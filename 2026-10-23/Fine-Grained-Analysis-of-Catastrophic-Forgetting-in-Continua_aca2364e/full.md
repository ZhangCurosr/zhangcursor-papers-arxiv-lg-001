# Fine-Grained Analysis of Catastrophic Forgetting in Continual Learning via Algorithmic Stability

Guodong Zheng<sup>1</sup>, Yan Sun<sup>2</sup>, Peng Wang<sup>1</sup>, Enneng Yang<sup>1</sup>, Jiafei Lyu<sup>3</sup>, Xiaochun Cao<sup>1</sup>, Dacheng Tao<sup>2</sup>, Li Shen<sup>1</sup>

Abstract—Understanding catastrophic forgetting, in which learning new tasks degrades performance on previously learned tasks, remains a fundamental challenge in continual learning (CL). Existing theoretical analyses often rely on specific model or task structures and offer limited insight into how iterative optimization affects forgetting. To address this gap, we develop an algorithmic stability framework for expected population forgetting under SGD without prescribing a particular model architecture. Specifically, we decompose forgetting into intratask and cross-task generalization gaps and empirical forgetting. To bound the generalization gaps, we track how a sample perturbation affects updates within a task and propagates through subsequent tasks. We consider constant, intra-task decaying, and continuously decaying learning-rate schedules. Comparing the two decaying schedules allows us to characterize how restarting learning-rate decay at task boundaries affects the propagation of historical sample perturbations. For empirical forgetting, we derive bounds for all three schedules by retaining cumulative gradient alignment along the optimization trajectory, without requiring convexity. Combining these results yields upper bounds on expected forgetting that capture the effects of training set size, the number of updates per task, the total number of tasks, learning rate schedules, and cumulative gradient alignment. Finally, we conduct experiments on Split CIFAR-100 and Split MiniImageNet to examine the empirical implications of our theoretical findings.

Index Terms—Catastrophic forgetting, algorithmic stability, intra-task generalization gap, cross-task generalization gap, empirical forgetting.

## I. INTRODUCTION

Continual learning (CL) aims to learn from a sequence of non-stationary data streams and tasks over time. However, this classical training paradigm remains vulnerable to a fundamental challenge, known as catastrophic forgetting [1, 2], in which the model exhibits a substantial degradation in performance on previously learned tasks as it adapts to newly introduced tasks.

To mitigate catastrophic forgetting, several empirical methods have been proposed and have achieved promising results [3]. However, the theoretical understanding remains much less developed. Existing analyses have largely focused on the linear regime [4–8] or the neural tangent kernel (NTK) regime [9–11]. While linear models yield explicit forgetting bounds, the resulting guarantees depend on the assumed linear structure. NTK-based analyses extend the study to neural networks, but typically rely on infinite-width limits. Moreover, most of these studies analyze forgetting through closed-form solutions to the optimization problem, abstracting away the fine-grained dynamics of iterative updates and yielding coarse bounds. Recently, Zheng et al. [11] analyze forgetting dynamics through NTK gradient flow, which captures more detailed trajectories than conventional static analyses. Taheri et al. [12] characterize forgetting in two-layer quadratic neural networks in the kernel regime, deriving explicit empirical forgetting bounds for orthogonal XOR-cluster tasks and using algorithmic stability to control delayed generalization. These studies provide insights into how forgetting evolves during training, while their guarantees remain tied to specific kernel approximations, network architectures, or task structures. This motivates an analysis of training dynamics, without prescribing a particular network architecture, that tracks both within-task updates and the propagation of their effects across tasks.

In this work, we develop an algorithmic stability analysis of expected forgetting in CL. We quantify forgetting as the expected increase in population risk on historical tasks and decompose it into intra-task generalization gaps (IGG), crosstask generalization gaps (CGG), and empirical forgetting (EF). To bound the generalization terms, we characterize stability under a sample perturbation in either the current task or a historical task. We follow SGD trajectories to track how the perturbation affects updates within that task and propagates through subsequent tasks. This analysis yields generalization bounds for constant, intra-task decaying, and continuously decaying learning rate schedules in the convex setting, and for the two decaying schedules in the non-convex setting. Under intra-task decay, the learning rate decreases from the same initial value within each task, whereas under continuous decay, it continues decreasing across task boundaries. Comparing these schedules allows us to characterize how resetting the learning rate at task boundaries affects the propagation of historical sample perturbations. For EF, we analyze changes in empirical risk on historical tasks along the optimization trajectory, retaining the cumulative inner products between gradients of historical tasks and those of the current task. The EF analysis covers all three learning rate schedules without requiring convexity. Combining these results yields bounds on expected population forgetting that explicitly account for both generalization gaps and empirical risk changes.

As summarized in Table I, the bounds reveal distinct effects of learning-rate scheduling on generalization and empirical forgetting. For convex losses with step sizes at most 2/β, the IGG and CGG bounds scale as $\mathcal { O } ( N / n )$ under constant learning rates and $\mathcal { O } ( \log ( e N ) / n )$ under intra-task decay, while continuous decay gives bounds independent of N for historical tasks $i > 1 .$ , with the other parameters fixed. For non-convex losses and $i > 1$ , the IGG and CGG bounds under continuous decay are no larger than those under intra-task decay at the same parameter values. In particular, for fixed $c , \beta , L , T ,$ i with $i > 1$ the CGG bound under continuous decay is $\mathcal { O } ( 1 / n )$ uniformly in N. Thus, increasing the local training length does not impose an additional growth restriction for this bound to vanish as n increases. These results show how carrying learning-rate decay across task boundaries changes the dependence of the generalization bounds on local training length. The EF bound further captures the joint contribution of cumulative gradient alignment and a second-order term controlled by squared learning rates. With the other parameters fixed, this secondorder term scales as O(N), O(1), and $\mathcal { O } ( 1 / N )$ under constant, intra-task decaying, and continuously decaying learning rates, respectively. Combining the generalization and EF bounds characterizes how perturbation propagation and trajectory dependent gradient interactions jointly enter the population forgetting bound. Finally, we conduct experiments on Split CIFAR-100 and Split MiniImageNet to examine the empirical implications of our theoretical findings.

TABLE I: Results for the IGG, CGG, and EF under convex and non-convex loss functions with three learning-rate (LR) schedules: constant (C), intra-task decaying (ID), and continuously decaying (CD). Here, T is the number of tasks and $i \in [ T - 1 ]$ is the index of a previous task. N and n denote the number of iterations and training samples per task, respectively. L and $\beta$ are the Lipschitz and smoothness constants. We define $\begin{array} { r } { p = \frac { c \beta } { c \beta + 1 } } \end{array}$ , where c is the constant factor in the learning rate. The term $\mathcal { G } _ { T , i }$ <sub>i</sub> is cumulative gradient alignment defined in Equation (6). For the non-convex ID setting, the IGG and CGG upper bounds reported in the table apply when $2 c L ^ { 2 } < N$ and $\bar { 2 c } L ^ { 2 } ( e N ) ^ { c \beta ( T - i ) } < N$ , respectively. When a condition is not satisfied, the corresponding upper bound is $N / n$ . The full assumptions and detailed piecewise results are provided in Section IV.
<table><tr><td>Case</td><td>LR</td><td> ${ \mathbf { I G G } } \ \epsilon _ { i , i }$ </td><td> $\mathbf { C G G } \in { \boldsymbol { T } } , i$ </td><td> $\mathbf { E F } \widetilde { \epsilon } _ { T , i }$ </td></tr><tr><td rowspan="3">Connvex</td><td>C</td><td> $\textstyle { \frac { 2 c L ^ { 2 } N } { n } }$ </td><td> $\textstyle { \frac { 2 c L ^ { 2 } N } { n } }$ </td><td> $\begin{array} { r } { - \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 } N ( T - i ) } \end{array}$ </td></tr><tr><td>ID</td><td> $\frac { 2 c L ^ { 2 } \ln ( e N ) } { n }$ </td><td> $\frac { 2 c L ^ { 2 } \ln ( e N ) } { n }$ </td><td> $- \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 } \left( 2 - 1 / N \right) \left( T - i \right)$ </td></tr><tr><td>CD</td><td> $\begin{array} { r } { \frac { 2 c L ^ { 2 } } { n } \ln \left( 1 + \frac { 1 } { i - 1 } \right) } \end{array}$ </td><td> $\begin{array} { r } { \frac { 2 c L ^ { 2 } } { n } \ln \left( 1 + \frac { 1 } { i - 1 } \right) } \end{array}$ </td><td> $\begin{array} { r l } { - \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 } \left( 1 / ( i N ) - 1 / ( T N ) \right) } & { { } } \end{array}$ </td></tr><tr><td rowspan="3">Nonnn-cvvex</td><td>ID</td><td> $( 2 c L ^ { 2 } ) ^ { 1 - p } N ^ { p } \qquad 2 L ^ { 2 }$   $\overline { { p n } } ^ { -- } \overline { { \beta n } }$ </td><td> $\begin{array} { r } { e ^ { p ( T - i ) } K _ { N ^ { T } - i + 1 } - \frac { 2 L ^ { 2 } ( e N ) ^ { c \beta ( T - i ) } } { \beta n } } \end{array}$ </td><td> $\begin{array} { r } { - \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 } ( 2 - 1 / N ) ( T - i ) } \end{array}$ </td></tr><tr><td> $\mathrm { C D ^ { a } }$ </td><td> $\begin{array} { r } { \mathcal { C } _ { 1 } ^ { 1 } : ~ \mathcal { K } _ { i N } - \frac { ( i - 1 ) N } { n } - \frac { 2 L ^ { 2 } } { \beta n } } \end{array}$ </td><td> $\begin{array} { r } { \mathcal { C } _ { 2 } ^ { 1 } : ~ \mathcal { K } _ { T N } - \frac { ( i - 1 ) N } { n } - \frac { 2 L ^ { 2 } } { \beta n } ( T / i ) ^ { c \beta } } \end{array}$ </td><td></td></tr><tr><td></td><td>C2 :  $\begin{array} { r } { \frac { 2 L ^ { 2 } } { \beta n } \left[ \big ( \frac { i } { i - 1 } \big ) ^ { c \beta } - 1 \right] } \end{array}$ </td><td>C{2 :  $\begin{array} { r } { \frac { 2 L ^ { 2 } } { \beta n } \left[ \big ( \frac { T } { i - 1 } \big ) ^ { c \beta } - \big ( \frac { T } { i } \big ) ^ { c \beta } \right] } \end{array}$ </td><td> $\begin{array} { r l } { - \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 } \left( 1 / ( i N ) - 1 / ( T N ) \right) } & { { } } \end{array}$ </td></tr></table>

<sup>a</sup> Let $\mathcal { C } _ { 1 } = ( 2 c L ^ { 2 } ) ^ { 1 - p } ( i N ) ^ { p } - ( i - 1 ) N$ and $\mathcal { C } _ { 2 } = ( 2 c L ^ { 2 } ) ^ { 1 - p } ( T N ) ^ { p } - ( i - 1 ) N .$ . For $j = 1 , 2 , \mathcal { C } _ { j } ^ { 1 }$ denotes $0 < \mathcal { C } _ { j } < N ,$ and $ { \mathcal { C } } _ { j } ^ { 2 }$ denotes $\mathcal { C } _ { j } \leq 0 .$ . When ${ \dot { \mathcal { C } } } _ { j } \geq N$ , the corresponding bound is $N / n$

combining the IGG, CGG, and EF bounds, linking generalization with changes in empirical risk along the continual learning trajectory.

Our main contributions can be summarized as follows:

Empirical studies in CL. Continual learning has made substantial progress in empirical methods for mitigating catastrophic forgetting. We divide these methods into three main families: (1) replay-based methods, which either store data from previous tasks and replay it during training on the current task [13–15], or store gradient or feature information from prior tasks and constrain the current-task gradient to be orthogonal to the stored information [16–20]; (2) regularization-based methods, which aim to restrain updates of parameters that are important for previous tasks [21, 22], or maintain consistency of intermediate features by adding a regularization term to the loss [23]; and (3) architecture-based methods, which dynamically expand or adapt the model architecture to preserve knowledge from previous tasks [24–26].

## II. RELATED WORK

• We derive explicit IGG and CGG bounds for continual SGD by tracking sample perturbations within a task and through subsequent tasks. These bounds characterize the effects of sample size, local training length, the total number of tasks, and learning-rate schedules.

• We derive an EF bound in terms of cumulative gradient alignment and an explicit second-order term involving squared learning rates. The bound reveals how positive cumulative gradient alignment contributes to controlling empirical forgetting.

• We derive upper bounds on expected forgetting through

Theoretical analysis of CL. Recent theoretical studies on CL have mainly focused on catastrophic forgetting and generalization. Broadly speaking, the existing literature can be divided into two main lines of work: linear model analysis and NTK-based analysis. Under the linear regime, Evron et al. [4], Lin et al. [5] investigate forgetting and generalization under the assumptions of overparameterized linear models. Ding et al. [6] study forgetting in both overparameterized and underparameterized regimes while relaxing the data assumptions to fourth-moment conditions. Moreover, several works use linear models to analyze concrete methods for mitigating forgetting, including replay-based methods [8, 27, 28], regularizationbased methods [7, 29–31], and mixture-of-experts methods [32]. The other main line of theoretical work relies on the NTK regime. Specifically, Bennani et al. [33] establish generalization error bounds for CL based on the NTK and Rademacher complexity. Doan et al. [9] analyze forgetting by introducing the NTK overlap matrix as a task similarity metric. Karakida and Akaho [10] study forgetting and generalization error through learning-curve analysis between two tasks. Zheng et al. [11] further characterize the training-time dynamics of forgetting and generalization in the general CL setting. Taheri et al. [12] analyze forgetting in two-layer quadratic neural networks in the kernel regime, deriving explicit empirical forgetting bounds for orthogonal XOR-cluster tasks and using algorithmic stability to control delayed generalization. Liu et al. [34] derive generalization bounds for parameter-efficient finetuning in CL. However, these theoretical analyses have several limitations: (1) they rely on restrictive model assumptions, such as linear models, infinite-width models, or infinitesimal learning rates, which make the results less general; and (2) the upper bounds derived in the NTK setting often involve complicated kernel matrices, making them difficult to interpret and generalize. In contrast, we analyze forgetting through a fine-grained characterization of the training trajectory without restrictive model assumptions. Moreover, our bounds depend only on a few readily interpretable quantities, such as the training set size, local training length, learning-rate schedule, and gradient inner products between tasks.

The work most closely related to ours is that of Taheri et al. [12], who analyze forgetting using algorithmic stability. Our analysis differs in three aspects: (1) Their theoretical results are developed for two-layer neural networks with quadratic activa tions and fixed output-layer weights, whereas our analysis does not prescribe a particular network architecture. (2) Their explicit empirical forgetting guarantees rely on orthogonal XOR-cluster tasks and the specified network model. Our EF bound requires only that each sample loss be Lipschitz continuous and smooth with respect to the model parameters, without imposing these structural assumptions. Their delayed generalization analysis also uses Lipschitz and smoothness assumptions on the loss. (3) Their main forgetting and generalization theorems adopt constant learning rates, whereas we examine constant, intra-task decaying, and continuously decaying learning rates.

## III. PRELIMINARY

## A. Continual Learning Setting

We consider the standard CL setting with $T \geq 2$ sequential training tasks. For any positive integer n, let $[ n ] : = \{ 1 , \dots , n \}$ For each task $\tau \in [ T ] .$ , let $\mathcal { D } _ { \tau }$ denote the data distribution. To simplify the analysis, we assume that each task dataset contains n samples, and write $S _ { \tau } = ( z _ { \tau , 1 } , \dots , z _ { \tau , n } )$ for the corresponding training dataset drawn i.i.d. from $\mathcal { D } _ { \tau }$ . The datasets are independent across tasks. We denote the task sequence by $S _ { 1 : T } : = \{ S _ { 1 } , \ldots , S _ { T } \}$ , which collects the datasets from tasks 1 to $T .$ . Let f denote a differentiable loss function on the parameter space $\Omega = \mathbb { R } ^ { d }$ . We adopt stochastic gradient descent (SGD) as our update rule, which can be formulated as

$$
\boldsymbol { w } _ { \tau , t + 1 } = \boldsymbol { w } _ { \tau , t } - \eta _ { \tau , t } \nabla _ { \boldsymbol { w } } f ( \boldsymbol { w } _ { \tau , t } ; \boldsymbol { z } _ { \tau , i _ { t } } ) ,
$$

where $\eta _ { \tau , t } \geq 0$ is the learning rate and $i _ { t }$ is sampled uniformly from [n] with replacement, independently across all updates and independently of the datasets and initialization. Here, $w _ { \tau , t }$ denotes the model parameters at step t for task τ. Each task is trained for N steps indexed by $t = 0 , \ldots , N - 1$ . We write $w _ { \tau } = w _ { \tau , N }$ for the model parameters obtained after training on task $\tau ,$ with $w _ { \tau , 0 } = w _ { \tau - 1 }$ and an initialization $w _ { 0 }$ independent of all task datasets.

Furthermore, we define the population risk of w<sub>τ</sub> on $\mathcal { D } _ { i }$ as $R _ { \mathcal { D } _ { i } } ( w _ { \tau } )$ and the empirical risk of $w _ { \tau }$ on $S _ { i }$ as $R _ { S _ { i } } ( w _ { \tau } )$ given by

$$
R _ { { \mathcal { D } } _ { i } } ( w _ { \tau } ) : = \mathbb { E } _ { z \sim { \mathcal { D } } _ { i } } f ( w _ { \tau } ; z ) ,
$$

$$
R _ { S _ { i } } ( w _ { \tau } ) : = \frac { 1 } { n } \sum _ { k = 1 } ^ { n } f ( w _ { \tau } ; z _ { i , k } ) .
$$

We next adopt the definition of forgetting from Zheng et al. [11], Taheri et al. [12], which characterizes the model’s performance over the underlying data distribution. The formal definition is given in Definition 1.

Definition 1 (Forgetting in expectation). For any $i \in [ T ]$ , let $w _ { i } = \mathcal { A } ( S _ { 1 : i } )$ denote the model obtained by the randomized algorithm A after training on the task sequence $S _ { 1 : i }$ . We define forgetting in expectation as

$$
F _ { T } = \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } \mathbb { E } _ { S _ { 1 : T } , A } \big [ R _ { \mathcal { D } _ { i } } ( w _ { T } ) - R _ { \mathcal { D } _ { i } } ( w _ { i } ) \big ] .\tag{1}
$$

Discussion. Definition 1 generalizes backward transfer (BWT) [35], a conventional metric for evaluating forgetting, in two respects. First, BWT measures forgetting on a test dataset, whereas $F _ { T }$ measures forgetting over the underlying data distribution. Second, $F _ { T }$ takes an expectation over the randomness in both the training dataset and the learning algorithm. These extensions yield a theoretical measure of expected population forgetting, with smaller values of $F _ { T }$ indicating less forgetting.

## B. Algorithmic Stability

In the non-CL setting, standard algorithmic stability typically concerns perturbations in the current training data only. In $\mathrm { C L , }$ by contrast, the model is exposed to a sequence of tasks, so perturbations at different tasks can propagate differently through the subsequent training trajectory. To reflect this sequential structure, we propose intra-task uniform stability for perturbations on the current task and cross-task uniform stability for perturbations on a historical task.

Definition 2 (Intra-task uniform stability). For any $i \in [ T ]$ a randomized algorithm A is said to be $\epsilon _ { i , i }$ -uniformly stable on task i after training on i tasks if, for any datasets $S _ { 1 : i } , S _ { 1 : i } ^ { \prime } \in$ ${ \mathcal { Z } } ^ { i \times n }$ that differ in at most one example in task $i ,$ we have

$$
\operatorname* { s u p } _ { z } \mathbb { E } _ { \boldsymbol { \mathcal { A } } } [ f ( \boldsymbol { \mathcal { A } } ( S _ { 1 : i } ) ; z ) - f ( \boldsymbol { \mathcal { A } } ( S _ { 1 : i } ^ { \prime } ) ; z ) ] \leq \epsilon _ { i , i } .\tag{2}
$$

Definition 3 (Cross-task uniform stability). For any $i \in$ $[ T - 1 ]$ , a randomized algorithm A is said to be $\epsilon _ { T , i } . . u n i f o r m l y$ stable on task i after training on $T$ tasks if, for any datasets $S _ { 1 : T } , S _ { 1 : T } ^ { \prime } \in \mathcal { Z } ^ { T \times n }$ that differ in at most one example in task $i ,$ we have

$$
\operatorname* { s u p } _ { z } \mathbb { E } _ { A } [ f ( { \cal A } ( S _ { 1 : T } ) ; z ) - f ( { \cal A } ( S _ { 1 : T } ^ { \prime } ) ; z ) ] \leq \epsilon _ { T , i } .\tag{3}
$$

In the following analysis, $\epsilon _ { i , i }$ and $\epsilon _ { T , i }$ denote the infima of the nonnegative constants satisfying Definitions 2 and 3, respectively.

Remark 1. As shown in Definition 2, the stability measure $\epsilon _ { i , i } – u n i f o r m$ stability characterizes the algorithm’s stability to perturbations in the data of the current task during CL. In contrast, Definition 3 extends this notion to perturbations in a historical task, thereby capturing how the effect of a local change can propagate through the subsequent training trajectory. This reflects the cumulative of stability in CL, where the influence of earlier tasks may persist and affect the final model after training on all tasks.

In this work, we follow Hardt et al. [36], Wang et al. [37], Chen et al. [38], Liu et al. [39], Zhu et al. [40] to adopt the standard assumptions used in algorithmic stability analysis:

Assumption 1 (L-Lipschitz). For every $z \in { \mathcal { Z } } _ { : }$ , the function $f ( \cdot ; z ) : \Omega \to \mathbb { R }$ is L-Lipschitz, that is, for all $u , v \in \Omega$

$$
| f ( u ; z ) - f ( v ; z ) | \leq L \| u - v \| .
$$

Assumption 2 (β-smooth). For every $z \in { \mathcal { Z } } ,$ the function $f ( \cdot ; z ) : \Omega \to \mathbb { R }$ is β-smooth, that is, for all $u , v \in \Omega$

$$
\begin{array} { r } { \| \nabla f ( u ; z ) - \nabla f ( v ; z ) \| \leq \beta \| u - v \| . } \end{array}
$$

Assumption 3 (Convex). For every $z \in { \mathcal { Z } } ,$ , thefunction $f ( \cdot ; z )$ $\Omega \to \mathbb { R }$ is convex, that $i s ,$ for all $u , v \in \Omega$

$$
f ( u ; z ) \geq f ( v ; z ) + \langle \nabla f ( v ; z ) , u - v \rangle .
$$

## IV. THEORETICAL ANALYSIS

In this section, we provide a theoretical analysis of catastrophic forgetting. To comprehensively analyze the forgetting measure $F _ { T }$ in Equation (1), it is important to derive a finegrained decomposition that allows us to understand forgetting from multiple perspectives:

$$
\begin{array} { r l } { F _ { T } = \cfrac { 1 } { T - 1 } \displaystyle \sum _ { i = 1 } ^ { T - 1 } [ \underbrace { \mathbb { E } _ { S _ { 1 : T } , A } [ R _ { \mathcal { D } _ { i } } ( w _ { T } ) - R _ { S _ { i } } ( w _ { T } ) ] } _ { \mathrm { C o s s - T a s k ~ G e n e r a l i z a t i o n ~ G a p } }  } & { } \\ { + \underbrace { \mathbb { E } _ { S _ { 1 : T } , A } [ R _ { S _ { i } } ( w _ { T } ) - R _ { S _ { i } } ( w _ { i } ) ] } _ { \mathrm { E m p i r i c a l ~ F o r g e t i o n } } } & { } \\ { + \underbrace { \mathbb { E } _ { S _ { 1 : T } , A } [ R _ { S _ { i } } ( w _ { i } ) - R _ { \mathcal { D } _ { i } } ( w _ { i } ) ] } _ { \mathrm { H r r a r - T a s k ~ G e n e r a l i z a t i o n ~ G a p } } ] . } \end{array}\tag{4}
$$

Based on the decomposition in Equation (4), we analyze the forgetting measure $F _ { T }$ from the following three aspects.

(1) Intra-Task Generalization Gap (Section IV-A). This term captures the generalization behavior of the trained model on the current task. A smaller absolute intra-task generalization gap indicates closer agreement between the empirical risk and the population risk on the current task.

(2) Cross-Task Generalization Gap (Section IV-B). This term reflects the generalization behavior of the model trained on later tasks when evaluated on earlier task distributions. A smaller absolute cross-task generalization gap indicates closer agreement between the empirical risk and the population risk on previous tasks.

(3) Empirical Forgetting (Section IV-C). This term captures how much performance on the training set of task i deteriorates after learning T tasks. A smaller empirical forgetting indicates that the model forgets less of the knowledge from the training set of a previous task.

In what follows, we derive upper bounds for these three terms under convex and non-convex loss functions, respectively, to provide a comprehensive understanding of catastrophic forgetting in CL.

## A. Intra-Task Generalization Gap

The intra-task generalization gap (IGG) describes the generalization performance on the current task. We next present an important theorem showing that intra-task uniform stability implies an upper bound on the intra-task generalization gap in expectation. The result is stated in Theorem 1, and the corresponding proof is given in Appendix B-A.

Theorem 1. For any $i \in [ T ]$ , if A is $\epsilon _ { i , i }$ -uniformly stable, then

$$
\begin{array} { r } { | \mathbb { E } _ { S _ { 1 : i } , \boldsymbol { A } } [ R _ { \mathcal { D } _ { i } } ( \boldsymbol { A } ( S _ { 1 : i } ) ) - R _ { S _ { i } } ( \boldsymbol { A } ( S _ { 1 : i } ) ) ] | \leq \epsilon _ { i , i } . } \end{array}
$$

Theorem 1 shows that an intra-task uniformly stable algorithm will have a small IGG. This suggests that we can control the uniform stability itself, rather than directly bounding the generalization gap. We then provide the corresponding upper bound under both convex and non-convex loss functions.

First, we establish an upper bound for the case where the loss function is convex in Theorem 2, and the proof of Theorem 2 is given in Appendix B-B.

Theorem 2. Assume that the loss function $f ( \cdot ; z )$ satisfies Assumptions $I , 2 ,$ and 3. Suppose that SGD is run for N steps on each task $\tau \in [ i ]$ with step sizes $\begin{array} { r } { \eta _ { \tau , t } \leq \frac { 2 } { \beta } } \end{array}$ . Then,

$$
\epsilon _ { i , i } \leq \frac { 2 L ^ { 2 } } { n } \sum _ { t = 0 } ^ { N - 1 } \eta _ { i , t } .\tag{5}
$$

Remark 2. Theorem 2 implies that, under convex lossfunctions, the IGG can be bounded in terms of the cumulative learning rate within task i. This bound is consistent with the classical non-CL result of Hardt et al. [36]. Hence, in the convex setting, the bound on $\epsilon _ { i , i }$ is independent of previous tasks and depends only on the optimization process within the current task. Moreover, smaller learning rates and larger numbers of training samples can further reduce this upper bound.

Moreover, for different learning rate schedules, including the constant learning rate $\eta _ { \tau , t } = c ,$ the intra-task decaying learning rate $\textstyle \eta _ { \tau , t } = { \frac { c } { t + 1 } }$ , and the continuously decaying learning rate $\begin{array} { r } { \eta _ { \tau , t } = \frac { c } { ( \tau - 1 ) N + t + 1 } } \end{array}$ , Equation (5) can be further written as

$$
\epsilon _ { i , i } \leq \left\{ \begin{array} { l l } { \frac { 2 c L ^ { 2 } N } { n } , } & { \mathrm { i f ~ } \eta _ { \tau , t } = c , } \\ { \frac { 2 c L ^ { 2 } \ln ( e N ) } { n } , } & { \mathrm { i f ~ } \eta _ { \tau , t } = \frac { c } { t + 1 } , } \\ { \frac { 2 c L ^ { 2 } } { n } \ln \left( 1 + \frac { 1 } { i - 1 } \right) , } & { \mathrm { i f ~ } \eta _ { \tau , t } = \frac { c } { ( \tau - 1 ) N + t + 1 } , i > 1 . } \end{array} \right.
$$

Remark 3. The upper bound on $\epsilon _ { i , i }$ varies across learningrate schedules. Under a constant learning rate, the bound grows linearly with the number of SGD steps N on the current task. Under an intra-task decaying learning rate, this growth is reduced to $\mathcal { O } ( \ln ( e N ) )$ . With other parameters fixed, reducing the number of training steps yields a smaller upper bound under both schedules, providing a theoretical motivation for early stopping. Under the continuously decaying learningrate schedule, the bound coincides with that under intra-task decay when $i = 1$ . For $i > 1$ , the displayed upper bound is independent of N and is instead controlled by the task index i.

Before presenting the IGG bounds for non-convex losses, we introduce the following notation for a recurring term in the subsequent analysis:

$$
K _ { N } : = \frac { c \beta + 1 } { n c \beta } ( 2 c L ^ { 2 } ) ^ { \frac { 1 } { c \beta + 1 } } N ^ { \frac { c \beta } { c \beta + 1 } } .
$$

The subscript specifies the number of iterations, with $c , \beta , L ,$ and n held fixed. In particular, for any positive integer i,

$$
K _ { i N } = \frac { c \beta + 1 } { n c \beta } ( 2 c L ^ { 2 } ) ^ { \frac { 1 } { c \beta + 1 } } ( i N ) ^ { \frac { c \beta } { c \beta + 1 } } .
$$

Then, we derive an upper bound for the more challenging non-convex case in Theorem 3, and the proof of Theorem 3 is given in Appendix B-C.

Theorem 3. Assume that the loss function $f ( \cdot ; z ) \in [ 0 , 1 ]$ satisfies Assumptions 1 and 2, with $n \geq 2$ and $c , \beta \ > \ 0 .$ Suppose that SGD is run for N steps on each task $\tau \in [ i ]$ with learning rates $\eta _ { \tau , t }$ . Then,

(a) under intra-task decaying learning rates $\textstyle \eta _ { \tau , t } = { \frac { c } { t + 1 } } .$

$$
\epsilon _ { i , i } \leq \left\{ \begin{array} { l l } { K _ { N } - \frac { 2 L ^ { 2 } } { \beta n } , } & { i f 2 c L ^ { 2 } < N , } \\ { \frac { N } { n } , } & { i f 2 c L ^ { 2 } \geq N . } \end{array} \right.
$$

(b) under continuously decaying learning rates $\begin{array} { r l } { \eta _ { \tau , t } } & { { } = } \end{array}$ $\frac { c } { ( \tau - 1 ) N + t + 1 }$ and $i > 1 ;$

$$
\epsilon _ { i , i } \leq \left\{ \begin{array} { l l } { \frac { 2 L ^ { 2 } } { \beta n } \left[ \left( \frac { i } { i - 1 } \right) ^ { c \beta } - 1 \right] , } & { i f 2 c L ^ { 2 } \leq \mathcal C _ { 1 } ^ { \mathrm { C D } } , } \\ { \mathcal K _ { i N } - \frac { ( i - 1 ) N } { n } - \frac { 2 L ^ { 2 } } { \beta n } , } & { i f \mathcal C _ { 1 } ^ { \mathrm { C D } } < 2 c L ^ { 2 } < i N , } \\ { \frac { N } { n } , } & { i f 2 c L ^ { 2 } \geq i N , } \end{array} \right.
$$

where

$$
\mathcal { C } _ { 1 } ^ { \mathrm { C D } } = ( i - 1 ) N \left( \frac { i - 1 } { i } \right) ^ { c \beta } .
$$

Remark 4. Theorem 3 provides an upper bound for the IGG without assuming convexity ofthe lossfunction. Let $p = c \beta / ( 1 +$ $c \beta )$ . Under intra-task decaying learning rates, when $2 c L ^ { 2 } < N$ the bound scales as $\mathcal { O } ( N ^ { p } / n )$ for fixed $c , \beta , L > 0 ,$ , consistent with the non-convex SGD stability rate established by Hardt et al. [36] in the single-task setting. As $n \to \infty$ , the condition $N = o ( n ^ { 1 / p } )$ is sufficient for a vanishing IGG. Thus, the number of training steps can increase with the sample size, provided that it grows more slowly than $n ^ { 1 / p }$ . When $2 c L ^ { 2 } \geq N$ corresponding to a larger initial learning rate or Lipschitz constant relative to the training length, the bound becomes $N / n .$ In both cases, increasing the sample size or reducing the number of training steps decreases the displayed upper bound, with the other parameters fixed.

Remark 5. For $i > 1 ,$ , continuously decaying learning rates provide stronger control of the IGG bound as the training length increases. When $2 c L ^ { 2 } \leq \mathcal { C } _ { 1 } ^ { \mathrm { C D } }$ , the bound is independent of N and decreases with the task index i. Since $i \ \geq \ 2 ,$ it is at most $\frac { 2 L ^ { 2 } } { \beta n } \left( 2 ^ { c \beta } - 1 \right)$ . In the intermediate case $\mathcal { C } _ { 1 } ^ { \mathrm { C D } } <$ $2 c L ^ { 2 } < i N$ , the bound increases with N but remains below $\begin{array} { r } { \frac { 2 L ^ { 2 } } { \beta n } [ ( i / ( i - 1 ) ) ^ { c \beta } - 1 ] } \end{array}$ . Thus, the dependence on N in this case does not lead to unbounded growth. For the same ${ { c } , \beta , L , n , N }$ the displayed bound under continuous decay is no larger than that under intra-task decay, and the improvement is strict when $0 < 2 c L ^ { 2 } < i N$ . When 2c $L ^ { 2 } \geq i N$ , both schedules give the bound $N / n ,$ , which remains nontrivial for $N < n$ . Moreover, increasing the number of training samples and decreasing the initial learning rate can further reduce the upper bound under this learning-rate schedule.

## B. Cross-Task Generalization Gap

The cross-task generalization gap (CGG) captures the generalization performance on earlier tasks. Next, we present the relationship between cross-task uniform stability and the expected CGG, which reflects how perturbations on historical tasks propagate through training in the CL setting and affect CGG. The result is stated in Theorem 4, and the proof is given in Appendix $\mathrm { C } { \cdot } \mathrm { A }$

Theorem 4. For any $i \in [ T - 1 ]$ , if A is $\epsilon _ { T , : }$ <sub>i</sub>-uniformly stable, then

$$
\begin{array} { r } { | \mathbb { E } _ { S _ { 1 : T } , A } [ R _ { \mathcal { D } _ { i } } ( A ( S _ { 1 : T } ) ) - R _ { S _ { i } } ( A ( S _ { 1 : T } ) ) ] | \leq \epsilon _ { T , i } . } \end{array}
$$

Theorem 4 indicates that small cross-task uniform stability leads to a small CGG. We then analyze CGG under both convex and non-convex loss functions.

First, we establish an upper bound for the case where the loss function is convex in Theorem 5, and the proof of Theorem 5 is given in Appendix C-B.

Theorem 5. Assume that the loss function $f ( \cdot ; z )$ satisfies Assumptions $I , 2 ,$ and 3. Suppose that SGD is run for N steps on each task $\tau \in [ T ]$ with step sizes $\begin{array} { r } { \eta _ { \tau , t } \leq { \frac { 2 } { \beta } } } \end{array}$ . Then,

$$
\epsilon _ { T , i } \leq \frac { 2 L ^ { 2 } } { n } \sum _ { t = 0 } ^ { N - 1 } \eta _ { i , t } .
$$

Remark 6. Theorem 5 surprisingly shows that the bound for the CGG coincides with that for the IGG under convex loss functions. This indicates that, under convexity, the algorithmic stability of SGD is preserved throughout the entire sequence of tasks. Even as the model parameters are continuously updated using data from new tasks, the resulting bound on the expected generalization gap on historical tasks does not grow. This conclusion relies on the non-expansive property of convex and smooth loss functions, as established in Lemma 1. Therefore, the discussion of IGG bound in Remark 3 remains valid for CGG as well.

Then, we derive an upper bound for the more challenging non-convex case in Theorem 6, and the proof of Theorem 6 is given in Appendix C-C.

Theorem 6. Assume that the loss function $f ( \cdot ; z ) \in [ 0 , 1 ]$ satisfies Assumptions 1 and 2, with $n \geq 2$ and $c , \beta \ > \ 0$ Suppose that SGD is run for N steps on each task $\tau \in [ T ]$ with learning rates $\eta _ { \tau , t } f o r t = 0 , \dots , N - 1$ . Then,

(a) under intra-task decaying learning rates $\begin{array} { r } { \eta _ { \tau , t } = \frac { c } { t + 1 } . } \end{array}$

$$
\epsilon _ { T , i } \leq \left\{ \begin{array} { l l } { \mathcal { B } _ { 2 } ^ { 1 } , } & { i f \ : 2 c L ^ { 2 } ( e N ) ^ { c \beta ( T - i ) } < N , } \\ { \quad } \\ { \displaystyle \frac { N } { n } , } & { i f \ : 2 c L ^ { 2 } ( e N ) ^ { c \beta ( T - i ) } \geq N . } \end{array} \right.
$$

where

$$
\mathcal { B } _ { 2 } ^ { 1 } = \mathcal { K } _ { N } ( e N ) ^ { \frac { c \beta ( T - i ) } { c \beta + 1 } } - \frac { 2 L ^ { 2 } ( e N ) ^ { c \beta ( T - i ) } } { \beta n } .
$$

(b) under continuously decaying learning rates $\begin{array} { r l } { \eta _ { \tau , t } } & { { } = } \end{array}$ $\frac { c } { ( \tau - 1 ) N + t + 1 }$ and $i > 1 .$

$$
\epsilon _ { T , i } \leq \left\{ \begin{array} { l l } { \frac { 2 L ^ { 2 } } { \beta n } \left[ \left( \frac { T } { i - 1 } \right) ^ { c \beta } - \left( \frac { T } { i } \right) ^ { c \beta } \right] , } & { i f 2 c L ^ { 2 } \leq \mathcal C _ { 2 } ^ { \mathrm { C D } } , } \\ { \mathcal B _ { 2 } ^ { 2 } , } & { i f \mathcal C _ { 2 } ^ { \mathrm { C D } } < 2 c L ^ { 2 } < i N \left( \frac { i } { T } \right) ^ { c \beta } , } \\ { \frac { N } { n } , } & { i f 2 c L ^ { 2 } \geq i N \left( \frac { i } { T } \right) ^ { c \beta } . } \end{array} \right.
$$

where

$$
\begin{array} { c } { { \displaystyle { \mathcal { B } } _ { 2 } ^ { 2 } = K _ { T N } - \frac { ( i - 1 ) N } { n } - \frac { 2 L ^ { 2 } } { \beta n } \left( \frac T i \right) ^ { c \beta } , } } \\ { { \displaystyle { \mathcal { C } } _ { 2 } ^ { \mathrm { C D } } = ( i - 1 ) N \left( \frac { i - 1 } T \right) ^ { c \beta } . } } \end{array}
$$

For i = 1 under the continuously decaying schedule,

$$
\epsilon _ { T , 1 } \leq \left\{ \begin{array} { l l } { K _ { T N } - \frac { 2 L ^ { 2 } T ^ { c \beta } } { \beta n } , } & { i f 2 c L ^ { 2 } T ^ { c \beta } < N , } \\ { \quad } \\ { \frac { N } { n } , } & { i f 2 c L ^ { 2 } T ^ { c \beta } \geq N . } \end{array} \right.
$$

Remark 7. Under intra-task decaying learning rates, the displayed CGG bound is nondecreasing in the number of subsequent tasks and is no smaller than the corresponding IGG bound. When $2 c L ^ { 2 } ( e N ) ^ { c \beta ( T - i ) } < N ,$ , a sufficient condition for the CGG to vanish as $n  \infty$ is $N = \stackrel { \sim } { o } \big ( n ^ { 1 / [ p ( T - i + 1 ) ] } \big )$ with $c , \beta , L , T ,$ i fixed. Compared with the sufficient condition $N = o ( n ^ { 1 / p } )$ for the IGG, this condition imposes a stronger restriction on the training length, reflecting the additional effect of perturbation propagation across subsequent tasks. The negative term in B<sup>1</sup> partially offsets the positive term and therefore tightens the bound. Within this branch, the ratio of the magnitude of the negative term to the positive term increases with the number of subsequent tasks when the other parameters are fixed, but remains strictly smaller than $1 / ( c \beta + 1 )$ . Thus, retaining the negative correction provides a greater relative reduction for tasks followed by more subsequent training. Increasing the sample size or reducing the number of training steps also decreases the displayed upper bound, with the other parameters fixed.

Remark 8. Under continuous decay, the learning rate continues to decrease across task boundaries, whereas intra-task decay resets it to c at the beginning of each task. The corresponding upper bounds on perturbation amplification across subsequent tasks are $( T / i ) ^ { \bar { c } \beta }$ and $( e N ) ^ { c \beta ( \mathbf { \check { T } } - \mathbf { \check { i } } ) }$ , respectively. Continuous decay thus gives a smaller amplification bound that is independent of N. For $i > 1$ , the displayed CGG bound under continuous decay is also no larger than that under intratask decay with the same parameters. All three cases of this bound are bounded above by $\textstyle \frac { 2 L ^ { 2 } } { \beta n } [ ( T / ( i - 1 ) ) ^ { c \beta } - ( T / i ) ^ { c \beta } ] ,$ which coincides with the first-case expression when $2 c L ^ { 2 } \leq$ $( i - 1 ) N ( ( i - 1 ) / T ) ^ { c \beta }$ . This common upper bound decreases as i increases, with the other parameters fixed, and its independence from N extends the comparison beyond any particular training length. Consequently,forfixed $c , \beta , L , T ,$ i with $i > 1$ , the CGG bound is $\mathcal { O } ( 1 / n )$ uniformly in N and vanishes as $n  \infty$ without restricting the growth of the training length.

## C. Empirical Forgetting

Empirical Forgetting (EF) quantifies the degradation in performance on the training set of a previous task caused by learn ing subsequent tasks. We denote $\tilde { \epsilon } _ { T , i } = \mathbb { E } _ { S _ { 1 : T } , A } [ R _ { S _ { i } } ( w _ { T } ) -$ $R _ { S _ { i } } ( w _ { i } ) ]$ as the expected change in the empirical risk on task i between the completion of task i and the completion of task T. To characterize the interactions between task i and subsequent tasks during training, we define the cumulative gradient alignment (CGA) as

$$
\mathcal { G } _ { T , i } : = \sum _ { \tau = i + 1 } ^ { T } \sum _ { t = 0 } ^ { N - 1 } \eta _ { \tau , t } \mathbb { E } _ { S _ { 1 : T } , A } \langle \nabla R _ { S _ { i } } ( w _ { \tau , t } ) , \nabla R _ { S _ { \tau } } ( w _ { \tau , t } ) \rangle .\tag{6}
$$

Here, ${ \mathcal { G } } _ { T , i }$ accumulates the expected inner products between the gradients of task i and those of subsequent tasks, weighted by the corresponding learning rates. Positive and negative values indicate cumulative gradient agreement and interference, respectively. We then provide an upper bound on $\tilde { \epsilon } _ { T , i }$ in Theorem 7, and the proof is given in Appendix D.

Theorem 7. Assume that the loss function $f ( \cdot ; z )$ satisfies Assumptions 1 and 2. Suppose that SGD is run for N steps on each task $\tau \in [ T ]$ with learning rates $\eta _ { \tau , t }$ . Then, (a) under constant learning rates $\eta _ { \tau , t } = c \mathrm { . }$

$$
\tilde { \epsilon } _ { T , i } \leq - \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 } N ( T - i ) ;
$$

(b) under intra-task decaying learning rates $\begin{array} { r } { \eta _ { \tau , t } = \frac { c } { t + 1 } . } \end{array}$

$$
\tilde { \epsilon } _ { T , i } \leq - \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 } \left( 2 - \frac { 1 } { N } \right) ( T - i ) ;
$$

(c) under continuously decaying learning rates $\begin{array} { r l } { \eta _ { \tau , t } } & { { } = } \end{array}$ $\frac { c } { ( \tau - 1 ) N + t + 1 } :$

$$
\tilde { \epsilon } _ { T , i } \leq - \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 } \left( \frac { 1 } { i N } - \frac { 1 } { T N } \right) .
$$

Remark 9. Theorem 7 bounds empirical forgetting in terms of the cumulative inner products between the old-task and currenttask gradients and the squared learning rates. A positive $\mathcal { G } _ { T , i }$ reduces the upper bound, while a negative value increases it. For fixed $T , \ i , \ c , \ L ,$ and β, the second-order term grows linearly with N under a constant learning rate. Under intratask decay, it approaches $\beta c ^ { 2 } L ^ { 2 } ( T - i )$ as N increases and thus becomes only weakly dependent on N for large N. Under continuous decay, the term decreases proportionally to $1 / N$ These results hold for both convex and non-convex losses.

As discussed in Remark 9, a positive $\mathcal { G } _ { T , i }$ reduces the upper bound on empirical forgetting, indicating that positive gradient alignment accumulated over subsequent updates can help preserve performance on task i. This observation motivates the following connection with existing CL methods.

Relation to gradient alignment mechanisms. GEM [35] aims to mitigate negative BWT by projecting the current gradient onto the set of directions having nonnegative inner products with all previous-task memory gradients. Let $\mathcal { M } _ { i }$ denote the episodic memory containing samples from task i. At step t of task τ, the projected gradient is obtained by solving

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { { \widetilde { g } } _ { \tau , t } } } & { \frac { 1 } { 2 } \left\| \widetilde { g } _ { \tau , t } - \nabla f ( w _ { \tau , t } ; z _ { \tau , i _ { t } } ) \right\| ^ { 2 } } \\ { \mathrm { s . t . ~ } } & { \left. \widetilde { g } _ { \tau , t } , \displaystyle \frac { 1 } { | \mathcal { M } _ { i } | } \sum _ { z \in \mathcal { M } _ { i } } \nabla f ( w _ { \tau , t } ; z ) \right. \geq 0 , \quad \forall i < \tau . } \end{array}\tag{7}
$$

The parameters are then updated as $w _ { \tau , t + 1 } = w _ { \tau , t } - \eta _ { \tau , t } \widetilde { g } _ { \tau , t }$ To reduce the computational cost of projection, A-GEM [13] replaces these individual constraints with a single constraint based on a reference gradient estimated from samples drawn from the combined episodic memory. Both methods seek to reduce gradient interference with past data, which is consistent with the role of cumulative gradient alignment captured by $- \mathcal { G } _ { T , i }$ in our EF bound.

Observable CGA. To examine the role of gradient alignment in empirical forgetting, we accumulate the learning-rateweighted inner products between historical evaluation-loss gradients and the actual training gradients. We estimate the historical gradients using measurement buffers sampled from previous tasks and denote the resulting cumulative alignment by $\widehat { \mathcal { G } } _ { T , i }$ . Historical gradients are evaluated using the same loss computation as EF, while current gradients are those used for the training updates. The buffers are used exclusively for measurement. Details are provided in Appendix F.

C  
![](images/577341d601228656cd06f8f0450b64d9813739f1ae30b4ff5ee863cf4997993e.jpg)

ID  
![](images/2c14b6bad2d9592e334c93f66820e1996c50baca870026b135b063d9cff28840.jpg)

CD  
![](images/0e6fb720b90a947b025240884c12356719cebe2df03cb046c9d3be744cb8c661.jpg)  
Fig. 1: EF and CGA under different learning-rate schedules on Split CIFAR-100. C, ID, and CD denote constant, intra-task decaying, and continuously decaying learning rates, respectively. Curves show the means over three seeds, and shaded regions indicate one standard deviation.

As shown in Fig. 1, the estimated CGA is large and positive under constant learning rates, particularly for earlier tasks. Under intra-task decay, its magnitude is substantially smaller, although EF and $- \widehat { \mathcal { G } } _ { T , i }$ remain visibly separated. Under continuous decay, the two quantities are closer and exhibit similar fluctuations across historical tasks. We also report results on Split MiniImageNet and provide details of the CGA measurements in Appendix F.

## D. Forgetting in Expectation

In this section, we summarize the discussions in Sections IV-A, IV-B, and IV-C. Specifically, the forgetting measure $F _ { T }$ can be bounded by the following inequality:

$$
F _ { T } \leq \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } \left( \epsilon _ { T , i } + \epsilon _ { i , i } + \tilde { \epsilon } _ { T , i } \right) .\tag{8}
$$

We next provide upper bounds on $F _ { T }$ for convex and nonconvex losses in Theorems 8 and 9, respectively.

Theorem 8. Assume that the loss function $f ( \cdot ; z )$ satisfies Assumptions $I , 2 ,$ and 3, with $\beta > 0$ . Suppose that SGD is run for N steps on each task $\tau \in [ T ]$ with learning rates $\eta _ { \tau , t } \leq 2 / \beta$ . For $0 < c \leq 2 / \beta _ { \mathrm { { i } } }$ , the following bounds hold: (a) under constant learning rates $\eta _ { \tau , t } = c \mathrm { : }$

$$
F _ { T } \le - \frac { \sum _ { i = 1 } ^ { T - 1 } \mathcal { G } _ { T , i } } { T - 1 } + \frac { \beta c ^ { 2 } L ^ { 2 } N T } { 4 } + \frac { 4 c L ^ { 2 } N } { n } ;
$$

(b) under intra-task decaying learning rates $\begin{array} { r } { \eta _ { \tau , t } = \frac { c } { t + 1 } . } \end{array}$

$$
F _ { T } \leq - \frac { \sum _ { i = 1 } ^ { T - 1 } \mathcal { G } _ { T , i } } { T - 1 } + \frac { \beta c ^ { 2 } L ^ { 2 } T } { 4 } \left( 2 - \frac { 1 } { N } \right) + \frac { 4 c L ^ { 2 } \ln ( e N ) } { n } ;
$$

(c) under continuously decaying learning rates $\begin{array} { r l } { \eta _ { \tau , t } } & { { } = } \end{array}$ $\frac { c } { ( \tau - 1 ) N + t + 1 } :$

$$
\begin{array} { l } { { F _ { T } \leq \displaystyle - \frac { \sum _ { i = 1 } ^ { T - 1 } \mathcal { G } _ { T , i } } { T - 1 } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 N ( T - 1 ) } \left( \ln ( T - 1 ) + \frac { 1 } { T } \right) } } \\ { { \displaystyle + \frac { 4 c L ^ { 2 } } { n ( T - 1 ) } \ln \left( e N ( T - 1 ) \right) . } } \end{array}
$$

Theorem 9. Assume that the loss function $f ( \cdot ; z ) \in [ 0 , 1 ]$ satisfies Assumptions 1 and 2, with $n \geq 2$ and $c , \beta , L > 0$ Suppose that SGD is run for N steps on each task $\tau \in [ T ]$ with learning rates $\eta _ { \tau , t }$ . Then,

(a) under intra-task decaying learning rates $\begin{array} { r } { \eta _ { \tau , t } = \frac { c } { t + 1 } . } \end{array}$ , if $2 c L ^ { 2 } ( e N ) ^ { c \beta ( T - 1 ) } < N $ , then

$$
\begin{array} { l } { { \displaystyle F _ { T } \leq - \frac { \sum _ { i = 1 } ^ { T - 1 } \mathcal { G } _ { T , i } } { T - 1 } + \frac { \beta c ^ { 2 } L ^ { 2 } T } { 4 } \left( 2 - \frac { 1 } { N } \right) } } \\ { { \displaystyle \qquad + K _ { N } \left[ 1 + \frac { \left( e N \right) ^ { \frac { c \beta ( T - 1 ) } { c \beta + 1 } } - 1 } { ( T - 1 ) \left( 1 - ( e N ) ^ { - \frac { c \beta } { c \beta + 1 } } \right) } \right] } } \\ { { \displaystyle \qquad - \frac { 2 L ^ { 2 } } { \beta n } \left[ 1 + \frac { ( e N ) ^ { c \beta ( T - 1 ) } - 1 } { ( T - 1 ) \left( 1 - ( e N ) ^ { - c \beta } \right) } \right] . } } \end{array}\tag{9}
$$

(b) under continuously decaying learning rates $\begin{array} { r l } { \eta _ { \tau , t } } & { { } = } \end{array}$ $\frac { c } { ( \tau - 1 ) N + t + 1 } :$

$$
\begin{array} { l l } { F _ { T } \leq - \frac { \sum _ { i = 1 } ^ { T - 1 } \mathcal { G } _ { T , i } } { \bar { T } - 1 } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 N ( T - 1 ) } \left( \ln ( T - 1 ) + \frac { 1 } { T } \right) } \\ { \qquad + \frac { 2 L ^ { 2 } } { \beta n ( T - 1 ) } \frac { 2 c ^ { 6 } - 1 } { \ln 2 } \ln ( T - 1 ) + \frac { 1 } { ( T - 1 ) } \times } \\ { \qquad \left\{ \begin{array} { l l } { K _ { N } \left( 1 + T ^ { \frac { e ^ { - \phi } + 1 } { \theta n } } \right) - \frac { 2 L ^ { 2 } } { n \beta } \left[ 1 + \left( \frac { T } { T - 1 } \right) ^ { c \beta } \right] , } & { i f \mathcal { C } _ { 3 } ^ { 1 } , } \\ { K _ { N } - \frac { 2 L ^ { 2 } } { n \beta } + \frac { ( T - 1 ) ^ { 8 } } { n } , } & { i f \mathcal { C } _ { 3 } ^ { 2 } , } \end{array} \right. } \\ { \qquad \left\{ \begin{array} { l l } { K _ { T N } - \frac { 2 L ^ { 2 } } { n \beta } \left( \frac { T } { T - 1 } \right) ^ { c \beta } + \frac { N } { n } , } & { i f \mathcal { C } _ { 3 } ^ { 3 } , } \\ { \qquad \frac { N } { n \beta } , } & { i f \mathcal { C } _ { 3 } ^ { 3 } , } \end{array} \right. } \\ { \qquad \left\{ \begin{array} { l l } { K _ { T N } - \frac { 2 L ^ { 2 } } { n \beta } \left( \frac { T } { T - 1 } \right) ^ { c \beta } + \frac { N } { n } , } & { i f \mathcal { C } _ { 3 } ^ { 3 } , } \\ { \qquad \frac { N } { n \alpha } , } & { i f \mathcal { C } _ { 3 } ^ { 4 } , } \end{array} \right. } \end{array}
$$

where

$$
\begin{array} { r } { \mathcal { C } _ { 3 } ^ { 1 } : 2 c L ^ { 2 } < N , \quad 2 c L ^ { 2 } T ^ { c \beta } < N ( T - 1 ) ^ { c \beta + 1 } , } \\ { \mathcal { C } _ { 3 } ^ { 2 } : 2 c L ^ { 2 } < N , \quad 2 c L ^ { 2 } T ^ { c \beta } \geq N ( T - 1 ) ^ { c \beta + 1 } , } \\ { \mathcal { C } _ { 3 } ^ { 3 } : 2 c L ^ { 2 } \geq N , \quad 2 c L ^ { 2 } T ^ { c \beta } < N ( T - 1 ) ^ { c \beta + 1 } , } \\ { \mathcal { C } _ { 3 } ^ { 4 } : 2 c L ^ { 2 } \geq N , \quad 2 c L ^ { 2 } T ^ { c \beta } \geq N ( T - 1 ) ^ { c \beta + 1 } . } \end{array}\tag{10}
$$

The proofs of Theorems 8 and 9 are given in Appendix E. To obtain explicit bounds on $F _ { T } ,$ we evaluate the accumulated contributions of IGG, CGG, and EF across previous tasks. As shown in Equation (8), the upper bound on forgetting can be reduced by controlling $\epsilon _ { T , i } , \epsilon _ { i , i } ,$ and $\tilde { \epsilon } _ { T , i }$ across historical tasks $i \in [ T - 1 ]$ . If the IGG and CGG bounds and the magnitude of EF all vanish, then $F _ { T }$ also converges to zero. The strategies for controlling these terms have been discussed in Sections IV-A– IV-C. We summarize the implications of the bounds as follows: (1) Using decaying learning rates or increasing the number of training samples per task n can tighten the generalization bounds, with the other parameters held fixed.

(2) Smaller initial learning rates c can reduce the generalization bounds, while their dependence on the local training length N varies across learning-rate schedules.

(3) Positive cumulative alignment between the gradients of previous and subsequent tasks contributes to reducing the EF upper bound through the term $- \mathcal { G } _ { T , i }$

## V. EXPERIMENT

In this section, we present experiments on Split CIFAR-100 to examine the empirical implications of our theoretical analysis. We also conduct experiments on Split MiniImageNet, with the results reported in Appendix F.

## A. Experimental Setup

Experimental setting on Split CIFAR-100. We conduct experiments on the widely used CL benchmark Split CIFAR-100 [13]. This benchmark partitions the 100 classes of CIFAR-100 [41] into 10 disjoint tasks, with 10 classes per task. We use AlexNet as the backbone, following the setting of Serra et al. [42]. We train the model with SGD to match our theoretical setting. To verify our theoretical findings, we vary the number of epochs per task in {10, 25, 50, 75, 100}, the initial learning rate in $\{ 1 0 ^ { - 3 } , 5 \times 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 5 \times 1 0 ^ { - 2 } , 1 0 ^ { - 1 } \}$ , and the number of samples per task in {500, 1500, 2500, 3500, 4500}. To verify how the number of epochs per task affects the metrics, we fix $c = 0 . 1$ and use the original training set setting, same as in Split CIFAR-100. To verify how the initial learning rate affects the metrics, we fix the number of epochs per task to 50 and use the same training set setting. To examine the effect of the number of training samples per task, we fix $c = 0 . 1$ and use a common budget of $N = 3 5 5 0$ updates per task. Each experiment is repeated with random seeds 1, 2, and 3, and the figures report the mean and standard deviation across the three runs.

Observable metric. The definition in Equation (4) relies on the population risk, which is not directly observable. Therefore, we define an observable metric by replacing the population risk in the theoretical metric with the test loss. Specifically, let $R _ { t , i }$ and $\tilde { R } _ { t , i }$ denote the training loss and test loss of the model on task $i ,$ respectively, after training on task t. We report the absolute generalization gaps and signed empirical and test forgetting as follows. These observed absolute gaps are distinct from the absolute expected gaps controlled by the stability theorems:

$$
\begin{array} { r l r } {  { 1 \mathrm { G G } = \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } | R _ { i , i } - \tilde { R } _ { i , i } | , } } \\ & { } & \\ & { } & { \displaystyle \mathrm { C G } = \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } | \tilde { R } _ { T , i } - R _ { T , i } | , } \\ & { } & \\ & { } & { \displaystyle \mathrm { E F } = \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } ( R _ { T , i } - R _ { i , i } ) , } \\ & { } & \\ & { } & { \displaystyle F _ { T } = \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } ( \tilde { R } _ { T , i } - \tilde { R } _ { i , i } ) . } \end{array}
$$

In addition to forgetting, we evaluate the model’s predictive performance using average accuracy (ACC). Let $A _ { t , i }$ denote the test accuracy on task i after learning the first t tasks, and let $T$ be the total number of tasks. ACC is defined as

$$
\operatorname { A C C } = \frac { 1 } { T } \sum _ { i = 1 } ^ { T } A _ { T , i } .
$$

Compute resources. We conduct the experiments on an Intel Xeon Platinum 8160 @ 2.10 GHz system with an NVIDIA RTX A6000 GPU.

## B. Experimental Result

Effects of the learning rate and the number of epochs per task. As shown in Fig. 2, the two decaying learningrate schedules generally yield smaller generalization gaps and less forgetting than the constant schedule. In particular, continuous decay keeps forgetting close to zero across the these settings. Moreover, the smallest initial learning rate and the fewest training epochs generally yield the least forgetting. To examine whether these observations extend to a more challenging benchmark and a different backbone, we also conduct experiments on Split MiniImageNet using ResNet-18. The results, reported in Appendix F, show similar overall trends.

![](images/8cbd8b88873848ab413fbd42438a4468ba19a5681e1531611b4821e81cd52fd0.jpg)  
Fig. 2: IGG, CGG, EF, $F _ { T } ,$ and ACC under different learning rate schedules on Split CIFAR-100. The top row show results for different initial learning rates $c ,$ while the bottom row shows results for different numbers of training epochs per task.

Stability-plasticity dilemma. As shown in Fig. 2, training with a continuously decaying learning rate yields the lowest ACC among the three learning-rate schedules. Moreover, reducing the number of training epochs or the initial learning rate does not necessarily improve ACC, even when it reduces forgetting. These observations reflect the classical stabilityplasticity dilemma in CL [43–45], highlighting the difficulty of preserving past knowledge while learning new tasks.

![](images/5cb1ec3ac18f3cbcbf56544808789d9eb9f27bbd8d32e3db7b960643cd1e0c06.jpg)  
Fig. 3: IGG and CGG with different numbers of training samples per task on Split CIFAR-100.

Effect of the number of training samples per task. As shown in Fig. 3, both IGG and CGG decrease as the number of training samples per task increases, with the number of SGD updates per task held fixed. This trend is consistent with our theoretical bounds, which decrease with the sample size when the other parameters are fixed.

Effect of gradient alignment under different learning rate schedules. We compare original CL, GEM, and A-GEM on Split CIFAR-100 under three learning rate schedules. GEM enforces nonnegative inner products between the projected gradient and the memory gradients of previous tasks, whereas A-GEM uses a single constraint based on an aggregate memory gradient. As shown in Fig. 4, GEM achieves the highest ACC and BWT under the constant and intra-task decaying schedules, with larger performance gaps under the constant schedule. Under the continuously decaying schedule, however, all three methods exhibit similarly low ACC despite achieving BWT values close to zero. This pattern is consistent with the stability-plasticity dilemma. Overall, these results suggest that the benefits of gradient alignment depend on the learning rate schedule, in line with our theoretical analysis in Section IV-C, which relates EF to both CGA and learning rates.

![](images/6424227ff9e9d4b3babfbaeebad2839d5b794b808d55bd75234756130d22d1a2.jpg)

![](images/5c9d15365e231ed5c7c2ae702ba11d58a19a576fb39e0b182f3d3ea083b6dc4c.jpg)  
Fig. 4: ACC and BWT for original CL, GEM, and A-GEM under different learning-rate schedules on Split CIFAR-100. Higher ACC indicates better predictive performance, while higher BWT indicates less forgetting.

## VI. CONCLUSION

In this work, we propose an algorithmic stability based theoretical framework for analyzing catastrophic forgetting under both convex and non-convex loss functions with three learning-rate schedules. Specifically, we derive a fine-grained decomposition of forgetting into IGG, CGG, and EF. We further propose intra-task and cross-task stability to analyze IGG and CGG, and analyze EF through the optimization dynamics. Our theoretical results characterize how local training lengths, initial learning rates, learning-rate schedules, and cumulative gradient alignment affect the forgetting bounds. Finally, we validate our theoretical findings through several empirical studies.

Limitation. The theoretical analysis is carried out under classical assumptions such as L-Lipschitz continuity and $\beta -$ smoothness, which are standard in stability and generalization analyses. However, these assumptions may be restrictive in practice. It would be interesting to relax them and further strengthen our results in future work.

## REFERENCES

[1] M. McCloskey and N. J. Cohen, “Catastrophic interference in connectionist networks: The sequential learning problem,” in Psychology of learning and motivation. Elsevier, 1989, vol. 24, pp. 109–165.

[2] J. L. McClelland, B. L. McNaughton, and R. C. O’Reilly, “Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory.” Psychological review, vol. 102, no. 3, p. 419, 1995.

[3] Z. Wang, E. Yang, L. Shen, and H. Huang, “A comprehen sive survey of forgetting in deep learning beyond continual learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[4] I. Evron, E. Moroshko, R. Ward, N. Srebro, and D. Soudry, “How catastrophic can catastrophic forgetting be in linear regression?” in Conference on Learning Theory. PMLR, 2022, pp. 4028–4079.

[5] S. Lin, P. Ju, Y. Liang, and N. Shroff, “Theory on forgetting and generalization of continual learning,” in International Conference on Machine Learning. PMLR, 2023, pp. 21 078–21 100.

[6] M. Ding, K. Ji, D. Wang, and J. Xu, “Understanding forgetting in continual learning with linear regression,” in International Conference on Machine Learning. PMLR, 2024, pp. 10 978–11 001.

[7] X. Zhao, H. Wang, W. Huang, and W. Lin, “A statistical theory of regularization-based continual learning,” in International Conference on Machine Learning. PMLR, 2024, pp. 61 021–61 039.

[8] M. Banayeeanzade, M. Soltanolkotabi, and M. Rostami, “Theoretical insights into overparameterized models in multi-task and replay-based continual learning,” Transac tions on Machine Learning Research, 2025.

[9] T. Doan, M. A. Bennani, B. Mazoure, G. Rabusseau, and P. Alquier, “A theoretical analysis of catastrophic forgetting through the ntk overlap matrix,” in International Conference on Artificial Intelligence and Statistics. PMLR, 2021, pp. 1072–1080.

[10] R. Karakida and S. Akaho, “Learning curves for continual learning in neural networks: Self-knowledge transfer and

forgetting,” in International Conference on Learning Representations, 2022.

[11] G. Zheng, P. Wang, S. Hu, Q. Zheng, and L. Shen, “Understanding the dynamics of forgetting and generalization in continual learning via the neural tangent kernel,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 18 488–18 525.

[12] H. Taheri, A. Ghosh, and A. Mazumdar, “On the theory of continual learning with gradient descent for neural networks,” in Proceedings of the 43rd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 306. PMLR, 06–11 Jul 2026, pp. 118 009–118 036.

[13] A. Chaudhry, M. Ranzato, M. Rohrbach, and M. Elhoseiny, “Efficient lifelong learning with a-GEM,” in International Conference on Learning Representations, 2019.

[14] M. Riemer, I. Cases, R. Ajemian, M. Liu, I. Rish, Y. Tu, and G. Tesauro, “Learning to learn without forgetting by maximizing transfer and minimizing interference,” arXiv preprint arXiv:1810.11910, 2018.

[15] P. Buzzega, M. Boschini, A. Porrello, D. Abati, and S. Calderara, “Dark experience for general continual learning: a strong, simple baseline,” Advances in neural information processing systems, vol. 33, pp. 15 920– 15 930, 2020.

[16] M. Farajtabar, N. Azizan, A. Mott, and A. Li, “Orthogonal gradient descent for continual learning,” in International conference on artificial intelligence and statistics. PMLR, 2020, pp. 3762–3773.

[17] G. Saha, I. Garg, and K. Roy, “Gradient projection memory for continual learning,” in International Conference on Learning Representations, 2021.

[18] S. Lin, L. Yang, D. Fan, and J. Zhang, “Trgp: Trust region gradient projection for continual learning,” in The Tenth International Conference on Learning Representations, 2022.

[19] E. Yang, L. Shen, Z. Wang, S. Liu, G. Guo, and X. Wang, “Data augmented flatness-aware gradient projection for continual learning,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 5630–5639.

[20] E. Yang, L. Shen, Z. Wang, S. Liu, G. Guo, X. Wang, and D. Tao, “Revisiting flatness-aware optimization in continual learning with orthogonal gradient projection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[21] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska et al., “Overcoming catastrophic forgetting in neural networks,” Proceedings of the national academy of sciences, vol. 114, no. 13, pp. 3521–3526, 2017.

[22] F. Zenke, B. Poole, and S. Ganguli, “Continual learning through synaptic intelligence,” in International conference on machine learning. PMLR, 2017, pp. 3987–3995.

[23] Z. Li and D. Hoiem, “Learning without forgetting,” IEEE transactions on pattern analysis and machine intelligence,

vol. 40, no. 12, pp. 2935–2947, 2017.

[24] A. A. Rusu, N. C. Rabinowitz, G. Desjardins, H. Soyer, J. Kirkpatrick, K. Kavukcuoglu, R. Pascanu, and R. Hadsell, “Progressive neural networks,” arXiv preprint arXiv:1606.04671, 2016.

[25] J. Yoon, E. Yang, J. Lee, and S. J. Hwang, “Lifelong learning with dynamically expandable networks,” in International Conference on Learning Representations, 2018.

[26] Z. Wang, Z. Zhan, Y. Gong, G. Yuan, W. Niu, T. Jian, B. Ren, S. Ioannidis, Y. Wang, and J. Dy, “Sparcl: Sparse continual learning on the edge,” Advances in Neural Information Processing Systems, vol. 35, pp. 20 366– 20 380, 2022.

[27] G. Zheng, P. Wang, T. Sun, and L. Shen, “Towards understanding memory buffer based continual learning,” Neural Networks, p. 108522, 2025.

[28] J. Deng, Q. Wu, P. Ju, S. Lin, Y. Liang, and N. Shroff, “Unlocking the power of rehearsal in continual learning: A theoretical perspective,” in International Conference on Machine Learning. PMLR, 2025, pp. 13 277–13 326.

[29] H. Li, J. Wu, and V. Braverman, “Fixed design analysis of regularization-based continual learning,” in Conference on lifelong learning agents. PMLR, 2023, pp. 513–533.

[30] ——, “Memory-statistics tradeoff in continual learning with structural regularization,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 63 493– 63 534.

[31] G. Karpel, E. Moroshko, R. Levinstein, R. Meir, D. Soudry, and I. Evron, “Optimal l2 regularization in high-dimensional continual linear regression,” arXiv preprint arXiv:2601.13844, 2026.

[32] H. Li, S. Lin, L. Duan, Y. Liang, and N. Shroff, “Theory on mixture-of-experts in continual learning,” in The Thirteenth International Conference on Learning Representations, 2025.

[33] M. A. Bennani, T. Doan, and M. Sugiyama, “Generalisation guarantees for continual learning with orthogonal gradient descent,” arXiv preprint arXiv:2006.11942, 2020.

[34] J. Liu, Z. Ji, Y. Yu, J. Cao, Y. Pang, J. Han, and X. Li, “Parameter-efficient fine-tuning for continual learning: A neural tangent kernel perspective,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[35] D. Lopez-Paz and M. Ranzato, “Gradient episodic memory for continual learning,” Advances in neural information processing systems, vol. 30, 2017.

[36] M. Hardt, B. Recht, and Y. Singer, “Train faster, generalize better: Stability of stochastic gradient descent,” in International conference on machine learning. PMLR, 2016, pp. 1225–1234.

[37] P. Wang, L. Shen, Z. Tao, S. He, and D. Tao, “Generalization analysis of stochastic weight averaging with general sampling,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235. PMLR, 21–27 Jul 2024, pp. 51 442–51 464.

[38] J. Chen, H. Chen, B. Gu, G. Liu, Y. Wang, and W. Li, “Error analysis affected by heavy-tailed gradients for non-

convex pairwise stochastic gradient descent,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 15, 2025, pp. 15 803–15 811.

[39] Y. Liu, Q. Li, J. Tan, Y. Shi, L. Shen, and X. Cao, “Understanding the stability-based generalization of personalized federated learning,” in The Thirteenth International Conference on Learning Representations, 2025.

[40] M. Zhu, Y. Sun, L. Shen, B. Du, and D. Tao, “Stability and generalization for distributed sgda,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[41] A. Krizhevsky, “Learning multiple layers of features from tiny images,” Technical report, University of Tront, 2009.

[42] J. Serra, D. Suris, M. Miron, and A. Karatzoglou, “Overcoming catastrophic forgetting with hard attention to the task,” in International conference on machine learning. PMLR, 2018, pp. 4548–4557.

[43] S. Grossberg, “Adaptive resonance theory: How a brain learns to consciously attend, learn, and recognize a changing world,” Neural Networks, vol. 37, pp. 1–47, 2013.

[44] E. Yang, A. Tang, L. Shen, G. Guo, X. Wang, X. Cao, and J. Zhang, “Continual model merging without data: Dual projections for balancing stability and plasticity,” in Advances in Neural Information Processing Systems, vol. 38, Main Conference. Curran Associates, Inc., 2025, pp. 39 275–39 305.

[45] G. Zheng, E. Yang, X. Wang, Y. Chen, F. He, Q. Zheng, P. Wang, and L. Shen, “Plasticity activation via polar operator: A plug-in method for balancing stability and plasticity,” in Proceedings of the 43rd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research. PMLR, 06–11 Jul 2026, pp. 163 850–163 877.

![](images/8f585353ce4fb7a3b57442405089da369263a1fe1a81584257382797ec681f2a.jpg)  
Guodong Zheng is currently a postdoctoral researcher with Sun Yat-sen University, China. He received his Ph.D. degree from Huazhong University of Science and Technology, China. His work appeared in several top-tier ML conferences/journals, including ICML, ICLR, and Neural Networks. His current research interests include optimization, continual learning.

![](images/8054a3cdf88857c1be25c6ffe4ad7a3972cf3f8f5c65d66b8c7971b32a6df8a3.jpg)

Yan Sun is currently a Postdoctoral Researcher with Nanyang Technological University (NTU). He received the B.E. degree from the University of Science and Technology of China (USTC) and the Ph.D. degree from the University of Sydney. He has published 10+ first-author papers in leading machine learning venues, including NeurIPS, ICML, ICLR, and IEEE TPAMI. His research interests include machine learning, federated and distributed learning, optimization and generalization, and efficient training and adaptation of large-scale foundation models.

![](images/57099e782112a47c652dc5f24ca3447d5bbd2231acb749eb6cbf0be2560c8735.jpg)  
Peng Wang is currently a postdoctoral researcher with Sun Yat-sen University, China. He received his Ph.D. degree from Huazhong University of Science and Technology, China. His work appeared in several top-tier ML conferences/journals, including ICML, ICLR, Neural Networks, and Pattern Recognition. His current research interests include the generalization and optimization theories of deep learning.

![](images/21cbdaee97bed8f2ce8c147c8ce406eedb41b9bc13e0a626a3064bf9cbbde6a0.jpg)

Li Shen is currently an associate professor at Sun Yatsen University. Previously, he was a research scientist at JD Explore Academy, Beijing, and a senior researcher at Tencent AI Lab, Shenzhen. He received his bachelor’s degree and Ph.D. from the School of Mathematics, South China University of Technology. His research interests include efficient deep learning, efficient reinforcement learning, optimization and deep learning theory. He has served as the senior program committee for AAAI and IJCAI, area chairs for ICML, NeurIPS, ICLR, CVPR, and ACMMM.

He is also the associate editor for IEEE Transactions on Pattern Analysis and Machine Intelligence, IEEE Transactions on Knowledge and Data Engineering, and IEEE Transactions on Multimedia.

![](images/a094a8ae496573a056286d6e56f5b3eb12c096ce9345fc0b297142fb03593ea8.jpg)

Enneng Yang is currently a postdoctoral researcher at Sun Yat-sen University, China. He received his Ph.D. degree from the Software College, Northeastern University, China. His work appeared in several top-tier ML & IR conferences/journals, including IEEE TPAMI, IEEE TKDE, JMLR, ICML, NeurIPS, ICLR, ICCV, ACL, WWW, SIGIR, and AAAI. His research interests include machine learning and recommendation systems.

![](images/19b8b8929eef01316d7661532eaf5b702d58b0f01999c56427b2e60a89483a0d.jpg)

Jiafei Lyu is currently a Senior Researcher at Tencent. He received the bachelor’s and Ph.D. degrees from Tsinghua University in 2020 and 2025, respectively. He has published papers in NeurIPS, ICML, ICLR, AAAI, and the Journal of Artificial Intelligence Research. His research interests include RLHF and efficient decision-making with deep Reinforcement Learning.

![](images/c701e2c227f2e4d843bc9384ad75049969dd4035280f820168a0c09e0765b9f0.jpg)

Xiaochun Cao (Senior Member, IEEE) received the B.E. and M.E. degrees in Computer Science from Beihang University, Beijing, China, in 1999 and 2002, and the Ph.D. degree from the University of Central Florida, USA, in 2006. He worked as a Research Scientist at ObjectVideo Inc. for three years. From 2008 to 2012, he was a Professor at Tianjin University, China. Before joining Sun Yatsen University, Shenzhen, he was a Professor at the Institute of Information Engineering, Chinese Academy of Sciences. He is currently the Dean and

Professor with the School of Cyber Science and Technology at Sun Yat-sen University, Shenzhen. He has published over 200 papers and received the Piero Zamperoni Best Student Paper Award at ICPR in 2004 and 2010. He serves on the Editorial Boards of IEEE TPAMI and IEEE TIP.

![](images/f29d61a26b21bdb18003912e2502774cac03dcda955aec486e15c53c1d99169c.jpg)

Dacheng Tao (Fellow, IEEE) is currently a Distinguished University Professor with the College of Computing and Data Science, Nanyang Technologi cal University. He mainly applies statistics and math ematics to artificial intelligence and data science. His research is detailed in one monograph and over 200 publications in prestigious journals and conferences, with best paper awards, best student paper awards, and test-of-time awards. His publications have been cited over 112K times. He has an H-index of more than 160 in Google Scholar. He is a Fellow of Aus tralian Academy of Science, AAAS, and ACM. He received the 2015 and 2020 Australian Eureka Prize, the 2018 IEEE ICDM Research Contributions Award, and the 2021 IEEE Computer Society McCluskey Technical Achievement Award.

## APPENDIX A PRELIMINARIES ON ALGORITHMIC STABILITY

In this section, we introduce several preliminaries that will be used in the subsequent proofs. We adopt the notation of Hardt et al. [36].

Definition 4. For a nonnegative step size $\alpha \geq 0$ and a function $f \colon \Omega  \mathbb { R }$ , we define the gradient update rule $G _ { f , \alpha }$ as

$$
G _ { f , \alpha } ( w ) = w - \alpha \nabla f ( w ) .
$$

Definition 5 (η−expansive). An update rule is η−expansive $i f$

$$
\operatorname* { s u p } _ { v , w \in \Omega } \frac { \| G ( v ) - G ( w ) \| } { \| v - w \| } \leq \eta\tag{11}
$$

Definition 6 (σ−bounded). An update rule is σ−bounded if

$$
\operatorname* { s u p } _ { w \in \Omega } \left\| w - G ( w ) \right\| \leq \sigma\tag{12}
$$

Lemma 1. Assume that f is β-smooth. Then, the following properties hold.

(1). $G _ { f , \alpha } \ i s \ ( 1 + \alpha \beta )$ -expansive.

(2). Assume in addition that f is convex. Then, for any $\alpha \leq 2 / \beta ,$ the gradient update $G _ { f , \alpha }$ is 1-expansive.

(3). Assume in addition that f is γ-strongly convex. Then, for $\begin{array} { r } { \alpha \leq \frac { 2 } { \beta + \gamma } , G _ { f , \alpha } i s \left( 1 - \frac { \alpha \beta \gamma } { \beta + \gamma } \right) - e x p a n s i \nu e . } \end{array}$

## APPENDIX B INTRA-TASK GENERALIZATION GAP

A. Proof of Theorem 1

Theorem. If A is ϵ<sub>i,i</sub>-uniformly stable, then

$$
\begin{array} { r } { \left| \mathbb { E } _ { S _ { 1 : i } , A } [ R _ { \mathcal { D } _ { i } } ( \mathcal { A } ( S _ { 1 : i } ) ) - R _ { S _ { i } } ( \mathcal { A } ( S _ { 1 : i } ) ) ] \right| \leq \epsilon _ { i , i } . } \end{array}\tag{13}
$$

Proof. We first introduce the following notation. Let $S _ { 1 : i }$ and $S _ { 1 : i } ^ { \prime }$ be two independent datasets, and let $S _ { 1 : i } ^ { ( i , k ) }$ denote the sample obtained from $S _ { 1 : i }$ by replacing the k-th example $z _ { i , k }$ in the dataset $S _ { i }$ from task i with $z _ { i , k } ^ { \prime } .$ More explicitly,

$$
\begin{array} { r l } & { \quad S _ { 1 : i } = ( z _ { 1 , 1 } , \dotsc , z _ { 1 , n } , \dotsc , z _ { i , 1 } , \dotsc , z _ { i , n } ) , } \\ & { \quad S _ { 1 : i } ^ { \prime } = ( z _ { 1 , 1 } ^ { \prime } , \dotsc , z _ { 1 , n } ^ { \prime } , \dotsc , z _ { i , 1 } ^ { \prime } , \dotsc , z _ { i , n } ^ { \prime } ) , } \\ & { \quad S _ { 1 : i } ^ { ( i , k ) } = ( z _ { 1 , 1 } , \dotsc , z _ { 1 , n } , \dotsc , z _ { i , 1 } , \dotsc , z _ { i , k - 1 } , } \\ & { \quad \qquad z _ { i , k } ^ { \prime } , z _ { i , k + 1 } , \dotsc , z _ { i , n } ) . } \end{array}\tag{14}
$$

With these notations, we have

$$
\begin{array} { r l } { \mathbb { E } _ { S _ { 1 : i } } \mathbb { E } _ { A } \big [ R _ { S _ { i } } ( \boldsymbol { A } ( S _ { 1 : i } ) ) \big ] } & { } \\ { = \mathbb { E } _ { S _ { 1 : i } } \mathbb { E } _ { A } \bigg [ \cfrac { 1 } { n } \displaystyle \sum _ { k = 1 } ^ { n } f ( \boldsymbol { A } ( S _ { 1 : i } ) ; \boldsymbol { z } _ { i , k } ) \bigg ] } & { } \\ { = \mathbb { E } _ { S _ { 1 : i } } \mathbb { E } _ { S _ { 1 : i } ^ { \prime } } \mathbb { E } _ { A } \bigg [ \cfrac { 1 } { n } \displaystyle \sum _ { k = 1 } ^ { n } f ( \boldsymbol { A } ( S _ { 1 : i } ^ { ( i , k ) } ) ; \boldsymbol { z } _ { i , k } ^ { \prime } ) \bigg ] } & { } \\ { = \mathbb { E } _ { S _ { 1 : i } } \mathbb { E } _ { S _ { 1 : i } ^ { \prime } } \mathbb { E } _ { A } \bigg [ \cfrac { 1 } { n } \displaystyle \sum _ { k = 1 } ^ { n } f ( \boldsymbol { A } ( S _ { 1 : i } ) ; \boldsymbol { z } _ { i , k } ^ { \prime } ) \bigg ] + \delta } & { } \\ { = \mathbb { E } _ { S _ { 1 : i } } \mathbb { E } _ { A \times 1 } \mathbb { E } _ { X _ { i } } ( \boldsymbol { A } ( S _ { 1 : i } ) ) + \delta . } \end{array}
$$

where

$$
\begin{array} { r l } & { \delta = \mathbb { E } _ { S _ { 1 : i } } \mathbb { E } _ { S _ { 1 : i } ^ { \prime } } \mathbb { E } _ { \boldsymbol { \mathcal { A } } } \left[ \cfrac { 1 } { n } \displaystyle \sum _ { k = 1 } ^ { n } f ( \boldsymbol { \mathcal { A } } ( S _ { 1 : i } ^ { ( i , k ) } ) ; \boldsymbol { z } _ { i , k } ^ { \prime } ) \right. } \\ & { \qquad \left. - \cfrac { 1 } { n } \displaystyle \sum _ { k = 1 } ^ { n } f ( \boldsymbol { \mathcal { A } } ( S _ { 1 : i } ) ; \boldsymbol { z } _ { i , k } ^ { \prime } ) \right] . } \end{array}
$$

Under the assumption that $\mathcal { A }$ is $\epsilon _ { i , i }$ -uniformly stable, we have

$$
| \delta | \leq \epsilon _ { i , i } .
$$

Therefore,

$$
\left| \mathbb { E } _ { S _ { 1 : i } , A } [ R _ { \mathcal { D } _ { i } } ( A ( S _ { 1 : i } ) ) - R _ { S _ { i } } ( A ( S _ { 1 : i } ) ) ] \right| = | \delta | \le \epsilon _ { i , i } .
$$

## B. Proof of Theorem 2

Theorem. Assume that the loss function $f ( \cdot ; z )$ satisfies Assumptions 1, 2, and 3. Suppose that SGD is run for N steps on each task $\tau \in [ i ]$ with step sizes $\begin{array} { r } { \eta _ { \tau , t } \leq { \frac { 2 } { \beta } } } \end{array}$ . Then,

$$
\epsilon _ { i , i } \leq \frac { 2 L ^ { 2 } } { n } \sum _ { t = 0 } ^ { N - 1 } \eta _ { i , t } .
$$

Proof. Let $S _ { 1 : i }$ and $S _ { 1 : i } ^ { \prime }$ be two datasets consisting of samples from the first i tasks, and assume that they differ in only one example in task i. Consider the update sequences $G _ { 1 , 0 } , \dots , G _ { 1 , N - 1 } , \dots , G _ { i , 0 } , \dots , G _ { i , N - 1 }$ and $G _ { 1 , 0 } ^ { \prime } , \ldots , G _ { 1 , N - 1 } ^ { \prime } , \ldots , G _ { i , 0 } ^ { \prime } , \ldots , G _ { i , N - 1 } ^ { \prime }$ induced by running SGD on $S _ { 1 : i }$ and $S _ { 1 : i } ^ { \prime } ,$ respectively. The two runs use the same initialization and sampled indices.

Fix an arbitrary example $z \in { \mathcal { Z } }$ . By the L-Lipschitz property of $f ( \cdot ; z )$ , we have

$$
\begin{array} { r } { \mathbb { E } \big | f ( w _ { i , N } ; z ) - f ( w _ { i , N } ^ { \prime } ; z ) \big | \leq L \mathbb { E } [ \delta _ { i , N } ] , } \end{array}\tag{15}
$$

where $\delta _ { i , N } = \lVert \boldsymbol { w } _ { i , N } - \boldsymbol { w } _ { i , N } ^ { \prime } \rVert$

During the t-th step of training on task i, with probability $\textstyle { 1 - { \frac { 1 } { n } } }$ , SGD selects the same example in both $S _ { i }$ and $S _ { i } ^ { \prime } .$ In this case, the gradient update operators coincide, namely $G _ { i , t } =$ $G _ { i , t } ^ { \prime }$ , and we use the 1-expansivity of $G _ { i , t }$ , which follows from the convexity and smoothness of $f .$ . With probability $\textstyle { \frac { 1 } { n } }$ , the selected examples differ. In this case, both $G _ { i , t }$ and $G _ { i , t } ^ { \prime }$ are $\eta _ { i , t } L$ -bounded.

Combining these two cases yields

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ \delta _ { i , t + 1 } ] \leq \left( 1 - \frac { 1 } { n } \right) \mathbb { E } [ \delta _ { i , t } ] + \frac { 1 } { n } \mathbb { E } [ \delta _ { i , t } ] + \frac { 2 \eta _ { i , t } L } { n } } \\ { \displaystyle = \mathbb { E } [ \delta _ { i , t } ] + \frac { 2 \eta _ { i , t } L } { n } . } \end{array}
$$

Unrolling this recursion over the $N$ steps of task i, we obtain

$$
\mathbb { E } [ \delta _ { i , N } ] \leq \mathbb { E } [ \delta _ { i , 0 } ] + \sum _ { t = 0 } ^ { N - 1 } \frac { 2 \eta _ { i , t } L } { n } .\tag{16}
$$

Under the CL protocol, the initialization of task i is the final model from task $i - 1$ , so $w _ { i , 0 } = w _ { i - 1 , N }$ and $w _ { i , 0 } ^ { \prime } = w _ { i - 1 , N } ^ { \prime } .$ Therefore,

$$
\delta _ { i , 0 } = \lVert w _ { i , 0 } - w _ { i , 0 } ^ { \prime } \rVert = \lVert w _ { i - 1 , N } - w _ { i - 1 , N } ^ { \prime } \rVert = \delta _ { i - 1 , N } .
$$

Since the perturbation occurs only in task $i ,$ all previous tasks are identical in $S _ { 1 : i - 1 }$ and $S _ { 1 : i - 1 } ^ { \prime }$ . Hence, $w _ { i - 1 , N } = w _ { i - 1 , N } ^ { \prime } .$ which implies $\delta _ { i - 1 , N } = 0$ . Therefore,

$$
\mathbb { E } [ \delta _ { i , 0 } ] = 0 .
$$

Substituting this into Equation (16) gives

$$
\mathbb { E } [ \delta _ { i , N } ] \leq \frac { 2 L } { n } \sum _ { t = 0 } ^ { N - 1 } \eta _ { i , t } .
$$

Finally, plugging this bound into Equation (15), we obtain

$$
\mathbb { E } \big | f ( w _ { i , N } ; z ) - f ( w _ { i , N } ^ { \prime } ; z ) \big | \leq \frac { 2 L ^ { 2 } } { n } \sum _ { t = 0 } ^ { N - 1 } \eta _ { i , t } .
$$

This proves the theorem.

## C. Proof of Theorem 3

Lemma 2. Assume that $n \geq 2$ and that the loss function $f ( \cdot ; z )$ is nonnegative and L-Lipschitz for every z. Let $S _ { 1 : i }$ and $S _ { 1 : i } ^ { \prime }$ be two datasets consisting of samples from the first i tasks, differing in only one example in task i. Then, for every $z \in { \mathcal { Z } }$ and every $t _ { 0 } \in \{ 0 , 1 , \ldots , N \}$ , under the random update rule, we have

$$
\begin{array} { r l } & { \quad \mathbb { E } \big | f ( w _ { i } ; z ) - f ( w _ { i } ^ { \prime } ; z ) \big | } \\ & { \leq \frac { t _ { 0 } } { n } \operatorname* { s u p } _ { w , z } f ( w ; z ) + L \mathbb { E } [ \| w _ { i } - w _ { i } ^ { \prime } \| | \delta _ { i , t _ { 0 } } = 0 ] . } \end{array}
$$

The same inequality holds under the random permutation rule when $t _ { 0 } < n$

Proof. Let $S _ { 1 : i }$ and $S _ { 1 : i } ^ { \prime }$ be two datasets consisting of samples from the first i tasks, and assume that they differ in only one example in task i. Consider the update sequences $G _ { 1 , 0 } , \dots , G _ { 1 , N - 1 } , \dots , G _ { i , 0 } , \dots , G _ { i , N - 1 }$ and $G _ { 1 , 0 } ^ { \prime } , \ldots , G _ { 1 , N - 1 } ^ { \prime } , \ldots , G _ { i , 0 } ^ { \prime } , \ldots , G _ { i , N - 1 } ^ { \prime }$ induced by running SGD on $S _ { 1 : i }$ and $S _ { 1 : i } ^ { \prime } ,$ respectively, with the same initialization and sampled indices.

Let $\mathcal { E }$ denote the event that $\delta _ { i , t _ { 0 } } = 0$ . Then,

$$
\begin{array} { r l } & { \quad \mathbb { E } \big | f ( w _ { i } ; z ) - f ( w _ { i } ^ { \prime } ; z ) \big | } \\ & { = \mathbb { P } ( \mathcal { E } ) \mathbb { E } \big [ | f ( w _ { i } ; z ) - f ( w _ { i } ^ { \prime } ; z ) | \mid \mathcal { E } \big ] } \\ & { \quad + \mathbb { P } ( \mathcal { E } ^ { c } ) \mathbb { E } \big [ | f ( w _ { i } ; z ) - f ( w _ { i } ^ { \prime } ; z ) | \mid \mathcal { E } ^ { c } \big ] } \\ & { \le \mathbb { E } \big [ | f ( w _ { i } ; z ) - f ( w _ { i } ^ { \prime } ; z ) | \mid \mathcal { E } \big ] + \mathbb { P } ( \mathcal { E } ^ { c } ) \operatorname* { s u p } _ { w , z } f ( w ; z ) } \\ & { \le L \mathbb { E } \big [ \| w _ { i } - w _ { i } ^ { \prime } \| \mid \mathcal { E } \big ] + \mathbb { P } ( \mathcal { E } ^ { c } ) \operatorname* { s u p } _ { w , z } f ( w ; z ) . } \end{array}
$$

We now bound $\mathbb { P } ( \mathcal { E } ^ { c } )$ . Let $i ^ { * }$ denote the unique position at which $S _ { 1 : i }$ and $S _ { 1 : i } ^ { \prime }$ differ, and let I be the first step at which SGD selects the example $z _ { i ^ { * } }$ . If $I > t _ { 0 } ,$ , then the two executions are identical up to step $t _ { 0 } ,$ and thus $\delta _ { i , t _ { 0 } } = 0$ . Therefore,

$$
\mathbb { P } ( \mathcal { E } ^ { c } ) = \mathbb { P } ( \delta _ { i , t _ { 0 } } \neq 0 ) \le \mathbb { P } ( I \le t _ { 0 } ) .
$$

Under the random permutation rule, I is uniformly distributed over $\{ 1 , \ldots , n \}$ , so, for $t _ { 0 } < n$

$$
\mathbb { P } ( I \leq t _ { 0 } ) = \frac { t _ { 0 } } { n } .
$$

Under the random selection rule, let $A _ { t }$ be the event that $z _ { i ^ { * } }$ is selected at step t. Then

$$
\mathbb { P } ( I \le t _ { 0 } ) \le \mathbb { P } \Big ( \bigcup _ { t = 1 } ^ { t _ { 0 } } A _ { t } \Big ) \le \sum _ { t = 1 } ^ { t _ { 0 } } \mathbb { P } ( A _ { t } ) = \frac { t _ { 0 } } { n } .
$$

This proves the claim.

Theorem. Assume that the loss function $f ( \cdot ; z ) \in [ 0 , 1 ]$ satisfies Assumptions 1 and $^ { 2 , }$ with $n \geq 2$ and $c , \beta > 0$ . Suppose that SGD is run for $N$ steps on each task $\tau \in [ i ]$ with learning rates $\eta _ { \tau , t }$ . Then,

(a) under intra-task decaying learning rates $\textstyle \eta _ { \tau , t } = { \frac { c } { t + 1 } } { \mathrm { . } }$

$$
\epsilon _ { i , i } \leq \left\{ \begin{array} { l l } { K _ { N } - \frac { 2 L ^ { 2 } } { \beta n } , } & { i f 2 c L ^ { 2 } < N , } \\ { \frac { N } { n } , } & { i f 2 c L ^ { 2 } \geq N . } \end{array} \right.
$$

(b) under continuously decaying learning rates $\begin{array} { r l } { \eta _ { \tau , t } } & { { } = } \end{array}$ $\frac { c } { ( \tau - 1 ) N + t + 1 }$ and $i > 1 .$

$$
\epsilon _ { i , i } \leq \left\{ \begin{array} { l l } { \frac { 2 L ^ { 2 } } { \beta n } \left[ \left( \frac { i } { i - 1 } \right) ^ { c \beta } - 1 \right] , } & { i f 2 c L ^ { 2 } \leq \mathcal C _ { 1 } ^ { \mathrm { C D } } , } \\ { \mathcal K _ { i N } - \frac { \left( i - 1 \right) N } { n } - \frac { 2 L ^ { 2 } } { \beta n } , } & { i f \mathcal C _ { 1 } ^ { \mathrm { C D } } < 2 c L ^ { 2 } < i N , } \\ { \frac { N } { n } , } & { i f 2 c L ^ { 2 } \geq i N , } \end{array} \right.
$$

where

$$
\mathcal { C } _ { 1 } ^ { \mathrm { C D } } = ( i - 1 ) N \left( \frac { i - 1 } { i } \right) ^ { c \beta } .
$$

Proof. Let $S _ { 1 : i }$ and $S _ { 1 : i } ^ { \prime }$ be two datasets consisting of samples from the first i tasks, and assume that they differ in only one example in task i. Consider the update sequences $G _ { 1 , 0 } , \dots , G _ { 1 , N - 1 } , \dots , G _ { i , 0 } , \dots , G _ { i , N - 1 }$ and $G _ { 1 , 0 } ^ { \prime } , \ldots , G _ { 1 , N - 1 } ^ { \prime } , \ldots , G _ { i , 0 } ^ { \prime } , \ldots , G _ { i , N - 1 } ^ { \prime }$ induced by running SGD on $S _ { 1 : i }$ and $S _ { 1 : i } ^ { \prime } ,$ respectively. The two runs use the same initialization and sampled indices.

Based on Lemma 2, we have for every integer $t _ { 0 } ~ \in$ $\{ 0 , \ldots , N \}$ under independent uniform sampling with replacement,

$$
\mathbb { E } | f ( w _ { i } ; z ) - f ( w _ { i } ^ { \prime } ; z ) | \leq \frac { t _ { 0 } } { n } + L \mathbb { E } [ \| w _ { i } - w _ { i } ^ { \prime } \| \mid \delta _ { i , t _ { 0 } } = 0 ] .
$$

We let $\Delta _ { i , t } = \mathbb { E } [ \delta _ { i , t } | \delta _ { i , t _ { 0 } } = 0 ]$ . The case $L = 0$ is immediate, so we take $L > 0$ in the calculations below. Next, we consider two types of learning rates.

Integer cutoffs are handled by the following comparison for any nonnegative, nonincreasing function D on (0, N]:

$$
\begin{array} { r l r } {  { \operatorname* { m i n } _ { m \in \{ 0 , \dots , N \} } \{ m + \sum _ { t = m + 1 } ^ { N } D ( t ) \} = \sum _ { t = 1 } ^ { N } \operatorname* { m i n } \{ 1 , D ( t ) \} } } \\ & { } & { \leq \int _ { 0 } ^ { N } \operatorname* { m i n } \{ 1 , D ( u ) \} d u . } \\ & { } & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \end{array}\tag{17}
$$

Choose m as the number of indices for which $D ( t ) \geq 1$ to obtain the equality; monotonicity then gives the integral bound. We use this comparison to evaluate the cutoff through a finite integral.

(a) Under intra-task decaying learning rates $\textstyle \eta _ { \tau , t } = { \frac { c } { t + 1 } }$ , the t-th update uses $\eta _ { i , t - 1 } = c / t$ . For $t = t _ { 0 } + 1 , \ldots , N$

$$
\begin{array} { l } { \displaystyle \Delta _ { i , t } \leq ( 1 - \frac 1 n ) ( 1 + \eta _ { i , t - 1 } \beta ) \Delta _ { i , t - 1 } + \frac 1 n \Delta _ { i , t - 1 } + \frac { 2 \eta _ { i , t - 1 } L } { n } } \\ { \displaystyle \quad = ( \frac 1 n + ( 1 - \frac 1 n ) ( 1 + \frac { c \beta } { t } ) ) \Delta _ { i , t - 1 } + \frac { 2 c L } { t n } } \\ { \displaystyle \quad = ( 1 + ( 1 - \frac 1 n ) \frac { c \beta } { t } ) \Delta _ { i , t - 1 } + \frac { 2 c L } { t n } } \\ { \displaystyle \quad \leq \exp ( ( 1 - \frac 1 n ) \frac { c \beta } { t } ) \Delta _ { i , t - 1 } + \frac { 2 c L } { t n } . } \end{array}
$$

Unrolling this recursion from $t _ { 0 }$ to N yields the following bound, where the integral estimate is used for $t _ { 0 } \geq 1 { : }$

$$
\begin{array} { r l } { \Delta _ { \Delta x } \leq } & { \underset { x \leq t \leq t } { \sum } \left. \begin{array} { l } { \prod _ { i = 1 } ^ { n } \log ( 1 - \frac { 1 } { n } \frac { \sigma \beta } { h } ) \bigg \rbrace \frac { 2 \sigma L } { \tilde { \tau } ^ { 2 } } } \\ { \mathrm { l o s s i t y } } \\ { = } \\ { - \underset { x \leq t + 1 } { \sum _ { n = 1 } ^ { n } \sigma } \left. \cosh \left( 1 - \frac { 1 } { n } \right) \triangle _ { \Delta + 1 } ^ { \Delta } \right. \frac { 1 } { \tilde { \tau } ^ { 2 } } } \\ { \mathrm { l o s s i t y } } \\ { \leq } \\ { \varepsilon - \underset { x \leq t + 1 } { \sum _ { n = 1 } ^ { n } \sigma } \left( \left( 1 - \frac { 1 } { n } \right) \cosh \left( \frac { \sigma L } { \tilde { \tau } ^ { 2 } } \right) \right) \frac { 2 \tilde { \tau } L } { \tilde { \tau } ^ { 2 } } } \\ { \mathrm { l o s s i t y } } \\ { - \frac { 2 \tilde { \tau } L } { n } \chi _ { 1 } ^ { ( 1 - \phi ) } \leq \underset { x \leq t \leq 1 } { \sum _ { n = 1 } ^ { n } \sigma } \sum _ { x \leq t \leq 1 } ^ { \infty - 1 } \mathrm { s g s i t y } - 1 } \end{array} \right. } \\ & { \leq \int _ { x } \frac { 2 \tilde { \tau } L } { n } \chi _ { 2 } ^ { ( 1 - \phi ) } \underset { x \leq t \leq 1 } { \sum _ { n = 1 } ^ { n } \sigma } \left. \begin{array} { l } { \sum _ { i = 1 } ^ { n } \frac { \partial L } { \partial \tilde { \tau } ^ { 2 } } } \\ { \mathrm { l o s s i t y } } \\ { \partial _ { \Delta } ^ { \mathrm { e p t } } } \end{array} \right. } \\ &  \leq \frac { 2 \tilde { \tau } L } { n } \sum _ { i = 1 } ^ { n } \sqrt { 2 \tilde { \tau } _ { n } ^ { 2 } } \frac { 2 \tilde { \tau } L ^ { 2 } } { n } - \partial _  \Delta \end{array}\tag{18}
$$

Here, the first inequality after the sum uses $( N / t ) ^ { ( 1 - 1 / n ) c \beta } \leq$ $( N / t ) ^ { c \beta }$ . Therefore, we obtain

$$
\mathbb { E } \left| f ( w _ { i } ; z ) - f ( w _ { i } ^ { \prime } ; z ) \right| \leq { \frac { t _ { 0 } } { n } } + { \frac { 2 L ^ { 2 } } { \beta n } } \left[ \left( { \frac { N } { t _ { 0 } } } \right) ^ { c \beta } - 1 \right] .
$$

Differentiating the right-hand side with respect to the real variable $t _ { 0 }$ gives the stationary point

$$
t _ { 0 } ^ { \star } = ( 2 c L ^ { 2 } ) ^ { \frac { 1 } { c \beta + 1 } } N ^ { \frac { c \beta } { c \beta + 1 } } .
$$

Applying Equation (17) to the unrolled sum, with $D ( u ) =$ $2 c L ^ { 2 } \dot { N } ^ { c \beta } u ^ { - c \beta - 1 }$ , gives

$$
\begin{array} { r l } & { \mathbb { E } | f ( w _ { i } ; z ) - f ( w _ { i } ^ { \prime } ; z ) | } \\ & { \le \displaystyle \frac { 1 } { n } \int _ { 0 } ^ { N } \operatorname* { m i n } \lbrace 1 , 2 c L ^ { 2 } N ^ { c \beta } u ^ { - c \beta - 1 } \rbrace d u . } \end{array}
$$

When $t _ { 0 } ^ { \star } < N$ , splitting the integral at $t _ { 0 } ^ { \star }$ yields

$$
\begin{array} { l } { \displaystyle \mathbb { E } | f ( w _ { i } ; z ) - f ( w _ { i } ^ { \prime } ; z ) | \leq \frac { t _ { 0 } ^ { \star } } { n } + \frac { 2 c L ^ { 2 } N ^ { c \beta } } { n } \int _ { t _ { 0 } ^ { \star } } ^ { N } { u ^ { - c \beta - 1 } } d u } \\ { \displaystyle \qquad = \frac { c \beta + 1 } { n c \beta } t _ { 0 } ^ { \star } - \frac { 2 L ^ { 2 } } { \beta n } . } \end{array}
$$

When $t _ { 0 } ^ { \star } \geq N$ , the integrand equals 1 throughout $( 0 , N )$ , giving $N / n$ . Since $t _ { 0 } ^ { \star } < N$ is equivalent to $2 c L ^ { 2 } < N$ , substituting $t _ { 0 } ^ { \star }$ gives part (a).

(b) Under continuously decaying learning rates $\begin{array} { r l } { \eta _ { \tau , t } } & { { } = } \end{array}$ <sup>c</sup><sub>(τ−1)N+t+1</sub> , the t-th update uses C $\eta _ { i , t - 1 } = c / ( ( i - 1 ) N + t )$ For t = t<sub>0</sub> + 1, . . . , N,

$$
\begin{array} { r l } & { \quad \Delta _ { i , t } } \\ & { \leq \biggl ( 1 - \frac { 1 } { n } \biggr ) \bigl ( 1 + \eta _ { i , t - 1 } \beta \bigr ) \Delta _ { i , t - 1 } + \frac { 1 } { n } \Delta _ { i , t - 1 } + \frac { 2 \eta _ { i , t - 1 } L } { n } } \\ & { = \biggl ( \frac { 1 } { n } + \Bigl ( 1 - \frac { 1 } { n } \Bigr ) \biggr ( 1 + \frac { c \beta } { ( i - 1 ) N + t } \Bigr ) \biggr ) \Delta _ { i , t - 1 } } \\ & { \quad + \frac { 2 c L } { \bigl ( ( i - 1 ) N + t \bigr ) n } } \\ & { = \biggl ( 1 + \Bigl ( 1 - \frac { 1 } { n } \Bigr ) \frac { c \beta } { \bigl ( i - 1 \bigr ) N + t } \biggr ) \Delta _ { i , t - 1 } + \frac { 2 c L } { \bigl ( ( i - 1 ) N + t \bigr ) n } } \\ & { \leq \exp \biggl ( \biggl ( 1 - \frac { 1 } { n } \biggr ) \frac { c \beta } { \bigl ( i - 1 \bigr ) N + t } \biggr ) \Delta _ { i , t - 1 } + \frac { 2 c L } { \bigl ( ( i - 1 ) N + t \bigr ) n } . } \end{array}
$$

Unrolling this recursion from $t _ { 0 }$ to $N$ yields

$$
\begin{array} { r l } &  \begin{array} { r l } & { \mathrm { 1 . 8 s u p } } \\ & { \varepsilon _ { \perp } ^ { \mathrm { S P } } ( \frac { \hat { \mathbf { J } } } { \mathbf { C } } , \mathbf { u } ) ^ { \perp } ( \hat { \mathbf { J } } _ { 1 } + \frac { \mathbf { B } } { \mathbf { C } } \overline { { \mathbf { J } ( \mathbf { A } \cdot \mathbf { B } ) } } ^ { \perp } ) } \\ & { \varepsilon _ { \perp } ^ { \mathrm { S P } } ( \frac { \hat { \mathbf { J } } } { \mathbf { C } } , \mathbf { u } ) ^ { \perp } ( \hat { \mathbf { J } } _ { 1 } + \frac { \mathbf { B } } { \mathbf { C } } \overline { { \mathbf { J } ( \mathbf { A } \cdot \mathbf { B } ) } } ^ { \perp } ) } \\ & { \varepsilon _ { \perp } ^ { \mathrm { S P } } \frac { \varepsilon _ { \perp } ^ { \mathrm { S P } } \varepsilon _ { \perp } ^ { \mathrm { S P } } } { \sqrt { \mathbf { C } } } } \\ & { \varepsilon _ { \perp } ^ { \mathrm { S P } } \frac { \varepsilon _ { \perp } ^ { \mathrm { S P } } \varepsilon _ { \perp } ^ { \mathrm { S P } } ( \hat { \mathbf { J } } _ { 1 } ^ { \perp } - \mathbf { B } ) ^ { \perp } } { \sqrt { \mathbf { C } } } } \\ & { \varepsilon _ { \perp } ^ { \mathrm { S P } } \frac { \varepsilon _ { \perp } ^ { \mathrm { S P } } \varepsilon _ { \perp } ^ { \mathrm { S P } } } { \sqrt { \mathbf { C } } } ( \hat { \mathbf { J } } _ { 2 } ^ { \perp } + \frac { \mathbf { B } } { \mathbf { C } } \overline { { \mathbf { J } ( \mathbf { A } \cdot \mathbf { B } ) } } ^ { \perp } ) } \\ &  \varepsilon _ { \perp } ^ { \mathrm { S P } } \frac { \varepsilon _ { \perp } ^ { \mathrm { S P } } \varepsilon _ { \perp } ^ { \mathrm { S P } } } { \sqrt { \mathbf { C } } } ( \hat { \mathbf { J } } _ { 3 } ^ { \perp } -  \end{array} \end{array}\tag{19}
$$

where we again use $i N / ( ( i - 1 ) N + t ) \geq 1$ to replace the exponent $( 1 - 1 / n ) c \beta$ by $c \beta$ before taking the integral. Since $i > 1$ , this integral is finite also at $t _ { 0 } = 0$

Therefore, we have

$$
\begin{array} { r l r } & {  { \mathbb { E } | f ( w _ { i } ; z ) - f ( w _ { i } ^ { \prime } ; z ) | } } \\ & { } & { \leq \frac { t _ { 0 } } { n } + \frac { 2 L ^ { 2 } } { \beta n } [ ( \frac { i N } { ( i - 1 ) N + t _ { 0 } } ) ^ { c \beta } - 1 ] . } \end{array}\tag{20}
$$

Differentiating the right-hand side with respect to the real variable $t _ { 0 }$ gives the stationary point

$$
t _ { 0 } ^ { \star } = ( 2 c L ^ { 2 } ) ^ { \frac { 1 } { c \beta + 1 } } ( i N ) ^ { \frac { c \beta } { c \beta + 1 } } - ( i - 1 ) N .
$$

Applying Equation (17) to the unrolled sum, with $D ( u ) =$ $2 \bar { c } \bar { L } ^ { 2 } ( i \bar { N } ) ^ { c \beta } \bar { ( } ( i - 1 ) N + u ) ^ { - c \beta - 1 }$ , gives

$$
\begin{array} { l } { \displaystyle \mathbb { E } | f ( w _ { i } ; z ) - f ( w _ { i } ^ { \prime } ; z ) | } \\ { \displaystyle \leq \frac { 1 } { n } \int _ { 0 } ^ { N } \operatorname* { m i n } \left\{ 1 , \frac { 2 c L ^ { 2 } ( i N ) ^ { c \beta } } { ( ( i - 1 ) N + u ) ^ { c \beta + 1 } } \right\} d u . } \end{array}
$$

When $t _ { 0 } ^ { \star } \leq 0$ , the second argument of the minimum is at most 1 throughout $( 0 , N )$ , and evaluating the integral gives

$$
\mathbb { E } | f ( w _ { i } ; z ) - f ( w _ { i } ^ { \prime } ; z ) | \leq \frac { 2 L ^ { 2 } } { \beta n } \left[ \left( \frac { i } { i - 1 } \right) ^ { c \beta } - 1 \right] .
$$

When $0 < t _ { 0 } ^ { \star } < N$ , splitting the integral at $t _ { 0 } ^ { \star }$ gives

$$
\begin{array} { r l } & { \quad \mathbb { E } | f ( w _ { i } ; z ) - f ( w _ { i } ^ { \prime } ; z ) | } \\ & { \le \displaystyle \frac { t _ { 0 } ^ { \star } } { n } + \frac { 2 c L ^ { 2 } ( i N ) ^ { c \beta } } { n } \int _ { t _ { 0 } ^ { \star } } ^ { N } \big ( ( i - 1 ) N + u \big ) ^ { - c \beta - 1 } d u } \\ & { = \displaystyle \frac { c \beta + 1 } { n c \beta } \big ( ( i - 1 ) N + t _ { 0 } ^ { \star } \big ) - \frac { ( i - 1 ) N } { n } - \frac { 2 L ^ { 2 } } { \beta n } . } \end{array}
$$

When $t _ { 0 } ^ { \star } \geq N$ , the integrand equals 1 throughout $( 0 , N )$ , giving $N / n$ . The conditions $t _ { 0 } ^ { \star } \leq 0$ and $t _ { 0 } ^ { \star } \geq N$ are equivalent to $2 \dot { c } L ^ { 2 } \le ( i - 1 ) N ( ( i - \overset \sim { 1 } ) / i ) ^ { c \beta }$ and $2 c L ^ { 2 } \geq i N$ , respectively. Substituting $t _ { 0 } ^ { \star }$ gives part (b). The formulas agree at both boundaries. □

## APPENDIX C CROSS-TASK GENERALIZATION GAP

## A. Proof of Theorem 4

Theorem. If A is ϵ<sub>T,i</sub>-uniformly stable, then

$$
\begin{array} { r } { | \mathbb { E } _ { S _ { 1 : T } , A } [ R _ { \mathcal { D } _ { i } } ( A ( S _ { 1 : T } ) ) - R _ { S _ { i } } ( A ( S _ { 1 : T } ) ) ] | \leq \epsilon _ { T , i } . } \end{array}\tag{21}
$$

Proof. We first introduce the following notation. Let $S _ { 1 : T }$ and $S _ { 1 : T } ^ { \prime }$ be two independent samples, and let $S _ { 1 : T } ^ { ( i , k ) }$ denote the sample obtained from $S _ { 1 : T }$ by replacing the k-th example $z _ { i , k }$ in the dataset $S _ { i }$ from task i with $z _ { i , k } ^ { \prime } .$ . More explicitly,

$$
\begin{array} { r l } & { S _ { 1 : T } = ( z _ { 1 , 1 } , \ldots , z _ { 1 , n } , \ldots , z _ { T , 1 } , \ldots , z _ { T , n } ) , } \\ & { S _ { 1 : T } ^ { \prime } = ( z _ { 1 , 1 } ^ { \prime } , \ldots , z _ { 1 , n } ^ { \prime } , \ldots , z _ { T , 1 } ^ { \prime } , \ldots , z _ { T , n } ^ { \prime } ) , } \\ & { S _ { 1 : T } ^ { ( i , k ) } = ( z _ { 1 , 1 } , \ldots , z _ { i , k - 1 } , z _ { i , k } ^ { \prime } , z _ { i , k + 1 } , \ldots , z _ { T , n } ) . } \end{array}\tag{22}
$$

With these notations, we have

$$
\begin{array} { r l } & { \mathbb { E } _ { S _ { 1 : T } } \mathbb { E } _ { \lambda } [ R _ { S _ { 1 : T } } ] } \\ & { = \mathbb { E } _ { S _ { 1 : T } } \mathbb { E } _ { \lambda } \Bigg [ \displaystyle \frac { 1 } { n } \sum _ { k = 1 } ^ { n } f ( \mathcal { A } ( S _ { 1 : T } ) ; z _ { i , k } ) \Bigg ] } \\ & { = \mathbb { E } _ { S _ { 1 : T } } \mathbb { E } _ { S _ { 1 : T } } \mathbb { E } _ { \lambda } \Bigg [ \displaystyle \frac { 1 } { n } \sum _ { k = 1 } ^ { n } f ( \mathcal { A } ( S _ { 1 : T } ^ { ( i , k ) } ) ; z _ { i , k } ^ { \prime } ) \Bigg ] } \\ & { = \mathbb { E } _ { S _ { 1 : T } } \mathbb { E } _ { S _ { 1 : T } } \mathbb { E } _ { \lambda } \Bigg [ \displaystyle \frac { 1 } { n } \sum _ { k = 1 } ^ { n } f ( \mathcal { A } ( S _ { 1 : T } ) ; z _ { i , k } ^ { \prime } ) \Bigg ] + \delta } \\ & { = \mathbb { E } _ { S _ { 1 : T } } \mathbb { E } _ { S _ { 1 : T } } \mathbb { E } _ { \lambda } \Bigg [ \displaystyle \frac { 1 } { n } \sum _ { k = 1 } ^ { n } f ( \mathcal { A } ( S _ { 1 : T } ) ; z _ { i , k } ^ { \prime } ) \Bigg ] + \delta } \\ & { = \mathbb { E } _ { S _ { 1 : T } } \mathbb { E } _ { \lambda } [ R _ { T _ { 0 } } ( \mathcal { A } ( S _ { 1 : T } ) ) ] + \delta . } \end{array}
$$

where

$$
\begin{array} { r l } & { \delta = \mathbb { E } _ { S _ { 1 : T } } \mathbb { E } _ { S _ { 1 : T } ^ { \prime } } \mathbb { E } _ { \mathcal { A } } \left[ \cfrac { 1 } { n } \displaystyle \sum _ { k = 1 } ^ { n } f ( \mathcal { A } ( S _ { 1 : T } ^ { ( i , k ) } ) ; z _ { i , k } ^ { \prime } ) \right. } \\ & { \quad \left. \vphantom { \sum _ { k = 1 } ^ { n } } - \frac { 1 } { n } \displaystyle \sum _ { k = 1 } ^ { n } f ( \mathcal { A } ( S _ { 1 : T } ) ; z _ { i , k } ^ { \prime } ) \right] . } \end{array}
$$

Under the assumption that A is $\epsilon _ { T , i }$ -uniformly stable, we have

$$
| \delta | \leq \epsilon _ { T , i } .
$$

Therefore,

$$
\begin{array} { r } { | \mathbb { E } _ { S _ { 1 : T } , A } [ R _ { \mathcal { D } _ { i } } ( \mathcal { A } ( S _ { 1 : T } ) ) - R _ { S _ { i } } ( \mathcal { A } ( S _ { 1 : T } ) ) ] | = | \delta | \le \epsilon _ { T , i } . } \end{array}
$$

## B. Proof of Theorem 5

Theorem. Assume that the loss function $f ( \cdot ; z )$ satisfies Assumptions 1, 2, and 3. Suppose that SGD is run for N steps on each task $\tau \in [ T ]$ with step sizes $\begin{array} { r } { 0 \leq \eta _ { \tau , t } \leq \frac { 2 } { \beta } } \end{array}$ for $t = 0 , \ldots , N - 1 .$ . Then,

$$
\epsilon _ { T , i } \leq \frac { 2 L ^ { 2 } } { n } \sum _ { t = 0 } ^ { N - 1 } \eta _ { i , t } .
$$

Proof. Let $S _ { 1 : T }$ and $S _ { 1 : T } ^ { \prime }$ be two datasets containing $T$ tasks, differing in only one example in task i. Couple the two runs using the same initialization and sampled indices. Consider the update sequences $G _ { 1 , 0 } , \dots , G _ { 1 , N - 1 } , \dots , G _ { T , 0 } , \dots , G _ { T , N - 1 }$ and $G _ { 1 , 0 } ^ { \prime } , \ldots , G _ { 1 , N - 1 } ^ { \prime } , \ldots , G _ { T , 0 } ^ { \prime } , \ldots , G _ { T , N - 1 } ^ { \prime }$ induced by running SGD on $S _ { 1 : T }$ and $S _ { 1 : T } ^ { \prime } ,$ respectively.

First, fix an arbitrary example $z \in { \mathcal { Z } }$ . By the L-Lipschitz continuity of $f ( \cdot ; z )$ , we have

$$
\begin{array} { r } { \mathbb { E } \big | f ( w _ { T , N } ; z ) - f ( w _ { T , N } ^ { \prime } ; z ) \big | \leq L \mathbb { E } [ \delta _ { T , N } ] , } \end{array}\tag{23}
$$

where $\delta _ { T , N } = \lVert w _ { T , N } - w _ { T , N } ^ { \prime } \rVert$ . Under the CL framework, the initialization for each task is exactly the final model from the previous task, that is, $w _ { k , 0 } = w _ { k - 1 , N }$ and $w _ { k , 0 } ^ { \prime } = w _ { k - 1 , N } ^ { \prime }$ for any $k \in \{ 2 , \ldots , T \}$ . Hence, $\delta _ { k , 0 } = \delta _ { k - 1 , N }$ . To bound $\mathbb { E } [ \delta _ { T , N } ] .$ we now consider two cases.

(i) When the task index satisfies $j \in \{ i + 1 , i + 2 , \ldots , T \}$ the corresponding dataset for task j is identical in $S _ { 1 : T }$ and $S _ { 1 : T } ^ { \prime }$ . Therefore, $G _ { j , t } = G _ { i , t } ^ { \prime }$ holds for any $j \in \{ i + 1 , i +$ $2 , \ldots , T \}$ and $t \in \{ \bar { 0 } , \ldots , \bar { N } - 1 \}$ . This allows us to apply the 1-expansivity of $G _ { j , t }$ . Under this case, we have

$$
\begin{array} { r l } & { \mathbb { E } [ \delta _ { T , N } ] \leq \mathbb { E } [ \delta _ { T , N - 1 } ] \leq \cdot \cdot \cdot \leq \mathbb { E } [ \delta _ { T , 0 } ] } \\ & { ~ = \mathbb { E } [ \delta _ { T - 1 , N } ] \leq \cdot \cdot \cdot } \\ & { ~ \leq \mathbb { E } [ \delta _ { i + 1 , 0 } ] = \mathbb { E } [ \delta _ { i , N } ] . } \end{array}\tag{24}
$$

(ii) During the t-th step of training on task i, with probability $\textstyle { 1 - { \frac { 1 } { n } } } .$ , the example selected by SGD is identical in both $S _ { i }$ and $\ddot { S } _ { i } ^ { \prime } .$ . In this case, the gradient update operators are identical $( \mathrm { i . e . , } G _ { i , t } = G _ { i , t } ^ { \prime } ) .$ , and we can use the 1-expansivity of $G _ { i , t }$ due to the convexity and smoothness of f. With probability $\textstyle { \frac { 1 } { n } }$ the selected examples differ. In this scenario, we rely on the property that both $G _ { i , t }$ and $G _ { i , t } ^ { \prime }$ are $\eta _ { i , t } L$ -bounded.

Combining these two cases, we obtain the following recurrence relation for $t = 0 , \ldots , N - 1 ;$

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ \delta _ { i , t + 1 } ] \leq \left( 1 - \frac { 1 } { n } \right) \mathbb { E } [ \delta _ { i , t } ] + \frac { 1 } { n } \mathbb { E } [ \delta _ { i , t } ] + \frac { 2 \eta _ { i , t } L } { n } } \\ { \displaystyle = \mathbb { E } [ \delta _ { i , t } ] + \frac { 2 \eta _ { i , t } L } { n } . } \end{array}
$$

Unraveling this recursion from step N down to step 0 within task i, we get:

$$
\mathbb { E } [ \delta _ { i , N } ] \leq \mathbb { E } [ \delta _ { i , 0 } ] + \sum _ { t = 0 } ^ { N - 1 } \frac { 2 \eta _ { i , t } L } { n } .\tag{25}
$$

Since the data perturbation occurs only in the dataset for task $i ,$ the datasets for all prior tasks remain identical. Consequently, for any previous task $m \in \{ 1 , \ldots , i - 1 \}$ and any update $t \in \{ 0 , \ldots , N - 1 \}$ , the operators $G _ { m , t } = G _ { m , t } ^ { \prime }$ . Therefore, we can apply the 1-expansivity property recursively:

$$
0 \leq \mathbb { E } [ \delta _ { i , 0 } ] \leq \mathbb { E } [ \delta _ { i - 1 , N } ] \leq \cdot \cdot \cdot \leq \mathbb { E } [ \delta _ { 1 , 0 } ] = 0 .
$$

Substituting $\mathbb { E } [ \delta _ { i , 0 } ] = 0$ into Equation (25) and combining it with Equation (24), we obtain

$$
\mathbb { E } [ \delta _ { T , N } ] \leq \frac { 2 L } { n } \sum _ { t = 0 } ^ { N - 1 } \eta _ { i , t } .
$$

Finally, plugging this bound back into Equation (23), we arrive at the stability bound:

$$
\mathbb { E } | f ( w _ { T , N } ; z ) - f ( w _ { T , N } ^ { \prime } ; z ) | \leq \frac { 2 L ^ { 2 } } { n } \sum _ { t = 0 } ^ { N - 1 } \eta _ { i , t } .
$$

## C. Proof of Theorem 6

Lemma 3. Assume that the loss function $f ( \cdot ; z )$ is nonnegative and L-Lipschitz for every $z ,$ and that $n \geq 2 .$ . Let $S _ { 1 : T }$ and $S _ { 1 : T } ^ { \prime }$ be two datasets consisting of samples from T tasks, differing in only one example in task i. Then, for every $z \in { \mathcal { Z } }$ and every $t _ { 0 } \in \{ 0 , 1 , \ldots , N \}$ under the random update rule, we have

$$
\begin{array} { r l } & { \quad \mathbb { E } | f ( w _ { T } ; z ) - f ( w _ { T } ^ { \prime } ; z ) | } \\ & { \leq \displaystyle \frac { t _ { 0 } } { n } \operatorname* { s u p } _ { w , z } f ( w ; z ) + L \mathbb { E } [ \| w _ { T } - w _ { T } ^ { \prime } \| \mid \delta _ { i , t _ { 0 } } = 0 ] . } \end{array}
$$

The same inequality holds under the random permutation rule for $0 \leq t _ { 0 } \leq N$ and $t _ { 0 } < n$

Proof. Let E denote the event that $\delta _ { i , t _ { 0 } } = 0$ , where $t _ { 0 }$ is the step index within task i. Then, we have

$$
\begin{array} { r l } & { \mathbb { E } | f ( w _ { T } ; z ) - f ( w _ { T } ^ { \prime } ; z ) | } \\ & { = \mathbb { P } \{ \mathcal { E } \} \mathbb { E } [ | f ( w _ { T } ; z ) - f ( w _ { T } ^ { \prime } ; z ) | \mid \mathcal { E } ] } \\ & { \quad + \mathbb { P } \{ \mathcal { E } ^ { c } \} \mathbb { E } [ | f ( w _ { T } ; z ) - f ( w _ { T } ^ { \prime } ; z ) | \mid \mathcal { E } ^ { c } ] } \\ & { \le \mathbb { E } [ | f ( w _ { T } ; z ) - f ( w _ { T } ^ { \prime } ; z ) | \mid \mathcal { E } ] + \mathbb { P } \{ \mathcal { E } ^ { c } \} \operatorname* { s u p } _ { w , z } f ( w ; z ) } \\ & { \le L \mathbb { E } [ \| w _ { T } - w _ { T } ^ { \prime } \| \mid \mathcal { E } ] + \mathbb { P } \{ \mathcal { E } ^ { c } \} \operatorname* { s u p } _ { w , z } f ( w ; z ) . } \end{array}
$$

The bound on $\mathbb { P } \{ \mathcal { E } ^ { c } \}$ is consistent with Lemma 2. □

Theorem. Assume that the lossfunction $f ( \cdot ; z ) \in [ 0 , 1 ]$ satisfies Assumptions 1 and 2, with $n \geq 2$ and $c , \beta > 0$ . Suppose that SGD is run for N steps on each task $\tau \in [ T ]$ with learning rates $\eta _ { \tau , t } f o r t = 0 , \dots , N - 1$ . Then,

(a) under intra-task decaying learning rates $\begin{array} { r } { \eta _ { \tau , t } = \frac { c } { t + 1 } \colon } \end{array}$

$$
\epsilon _ { T , i } \leq \left\{ \begin{array} { l l } { \mathcal { B } _ { 2 } ^ { 1 } , } & { i f \ : 2 c L ^ { 2 } ( e N ) ^ { c \beta ( T - i ) } < N , } \\ { \quad } \\ { \displaystyle \frac { N } { n } , } & { i f \ : 2 c L ^ { 2 } ( e N ) ^ { c \beta ( T - i ) } \geq N . } \end{array} \right.
$$

where

$$
\mathcal { B } _ { 2 } ^ { 1 } = \mathcal { K } _ { N } ( e N ) ^ { \frac { c \beta ( T - i ) } { c \beta + 1 } } - \frac { 2 L ^ { 2 } ( e N ) ^ { c \beta ( T - i ) } } { \beta n } .
$$

(b) under continuously decaying learning rates $\begin{array} { r l } { \eta _ { \tau , t } } & { { } = } \end{array}$ $\frac { c } { ( \tau - 1 ) N + t + 1 }$ and $i > 1 .$

$$
\epsilon _ { T , i } \leq \left\{ \begin{array} { l l } { \frac { 2 L ^ { 2 } } { \beta n } \left[ \left( \frac { T } { i - 1 } \right) ^ { c \beta } - \left( \frac { T } { i } \right) ^ { c \beta } \right] , } & { i f 2 c L ^ { 2 } \leq \mathcal C _ { 2 } ^ { \mathrm { C D } } , } \\ { \mathcal B _ { 2 } ^ { 2 } , } & { i f \mathcal C _ { 2 } ^ { \mathrm { C D } } < 2 c L ^ { 2 } < i N \left( \frac { i } { T } \right) ^ { c \beta } , } \\ { \frac { N } { n } , } & { i f 2 c L ^ { 2 } \geq i N \left( \frac { i } { T } \right) ^ { c \beta } . } \end{array} \right.
$$

where

$$
\begin{array} { c } { { \displaystyle { \mathcal { B } } _ { 2 } ^ { 2 } = K _ { T N } - \frac { ( i - 1 ) N } { n } - \frac { 2 L ^ { 2 } } { \beta n } \left( \frac T i \right) ^ { c \beta } , } } \\ { { \displaystyle { \mathcal { C } } _ { 2 } ^ { \mathrm { C D } } = ( i - 1 ) N \left( \frac { i - 1 } T \right) ^ { c \beta } . } } \end{array}
$$

For $i = 1$ under the continuously decaying schedule,

$$
\epsilon _ { T , 1 } \leq \left\{ \begin{array} { l l } { K _ { T N } - \frac { 2 L ^ { 2 } T ^ { c \beta } } { \beta n } , } & { i f 2 c L ^ { 2 } T ^ { c \beta } < N , } \\ { \quad } \\ { \frac { N } { n } , } & { i f 2 c L ^ { 2 } T ^ { c \beta } \geq N . } \end{array} \right.
$$

Proof. Let $S _ { 1 : T }$ and $S _ { 1 : T } ^ { \prime }$ be two datasets containing $T$ tasks, differing in only one example in task i. Couple the two runs using the same initialization and sampled indices. Consider the update sequences $G _ { 1 , 0 } , \dots , G _ { 1 , N - 1 } , \dots , G _ { T , 0 } , \dots , G _ { T , N - 1 }$ and $G _ { 1 , 0 } ^ { \prime } , \ldots , G _ { 1 , N - 1 } ^ { \prime } , \ldots , G _ { T , 0 } ^ { \prime } , \ldots , G _ { T , N - 1 } ^ { \prime }$ induced by running SGD on $S _ { 1 : T }$ and $S _ { 1 : T } ^ { \prime }$ , respectively.

By Lemma 3, for any integer $t _ { 0 } \in \{ 0 , \ldots , N \}$ , we have

$$
\mathbb { E } \left| f ( w _ { T } ; z ) - f ( w _ { T } ^ { \prime } ; z ) \right| \leq \frac { t _ { 0 } } { n } + L \mathbb { E } [ \| w _ { T } - w _ { T } ^ { \prime } \| | \delta _ { i , t _ { 0 } } = 0 ] .\tag{26}
$$

Define $\Delta _ { \tau , t } : = \mathbb { E } [ \delta _ { \tau , t } \mid \delta _ { i , t _ { 0 } } = 0 ]$ . We first bound $\Delta _ { T , N \cdot } \mathrm { B y }$ construction, for every task $m \in \{ i + 1 , \ldots , T \}$ , the sampled examples at each step are identical in $S _ { 1 : T }$ and $S _ { 1 : T } ^ { \prime }$ . Hence, $G _ { m , t } = G _ { m , t } ^ { \prime }$ for all $t \in \{ 0 , \ldots , N - 1 \}$ , and we may use the $( 1 + \eta _ { m , t } \beta )$ -expansivity of $G _ { m , t } .$

Therefore, for any $t \in \{ 0 , \ldots , N - 1 \}$

(a) under intra-task decaying learning rates $\begin{array} { r } { \eta _ { \tau , t } = \frac { c } { t + 1 } . } \end{array}$

$$
\begin{array} { r l r } {  { \Delta _ { T , t + 1 } \leq ( 1 + \eta _ { T , t } \beta ) \Delta _ { T , t } } } \\ & { } & \\ & { } & { = ( 1 + \frac { c \beta } { t + 1 } ) \Delta _ { T , t } } \\ & { } & \\ & { } & { \leq \exp ( \frac { c \beta } { t + 1 } ) \Delta _ { T , t } . } \end{array}
$$

Unrolling this recursion from N down to 0 yields

$$
\begin{array} { r l } {  { \Delta _ { T , N } \le \prod _ { k = 0 } ^ { N - 1 } \exp ( \frac { c \beta } { k + 1 } ) \Delta _ { T , 0 } } } \\ & { = \exp ( { \alpha \beta \sum _ { k = 0 } ^ { N - 1 } \frac { 1 } { k + 1 } } ) \Delta _ { T , 0 } } \\ & { = \exp ( { c \beta \sum _ { k = 1 } ^ { N } \frac { 1 } { k } } ) \Delta _ { T , 0 } } \\ & { \le \exp ( { c \beta ( \ln N + 1 ) } ) \Delta _ { T , 0 } } \\ & { = ( e N ) ^ { e \beta } \Delta _ { T , 0 } . } \end{array}
$$

Under the CL protocol, the initialization of each task is the final model from the previous task, namely $w _ { k , 0 } = w _ { k - 1 , N }$ and $w _ { k , 0 } ^ { \prime } = w _ { k - 1 , N } ^ { \prime }$ for any $k \in \{ 2 , \ldots , T \}$ . Hence, $\delta _ { k , 0 } = \delta _ { k - 1 , N } .$ Therefore,

$$
\begin{array} { r l } & { \Delta _ { T , N } \leq ( e N ) ^ { c \beta } \Delta _ { T , 0 } } \\ & { \qquad = ( e N ) ^ { c \beta } \Delta _ { T - 1 , N } } \\ & { \qquad \leq \cdots \leq ( e N ) ^ { c \beta ( T - i ) } \Delta _ { i , N } . } \end{array}
$$

For $1 \le t _ { 0 } \le N$ , Equation (18) further gives

$$
\Delta _ { T , N } \leq \frac { 2 L ( e N ) ^ { c \beta ( T - i ) } } { \beta n } \left[ \left( \frac { N } { t _ { 0 } } \right) ^ { c \beta } - 1 \right] .
$$

Substituting this bound into Equation (26), we have

$$
\begin{array} { l } { \displaystyle \mathbb { E } | f ( w _ { T } ; z ) - f ( w _ { T } ^ { \prime } ; z ) | } \\ { \displaystyle \leq \frac { t _ { 0 } } { n } + \frac { 2 L ^ { 2 } ( e N ) ^ { c \beta ( T - i ) } } { \beta n } \left[ \left( \frac { N } { t _ { 0 } } \right) ^ { c \beta } - 1 \right] . } \end{array}
$$

Differentiating the right-hand side with respect to a positive real cutoff gives the stationary point

$$
\begin{array} { r } { \widehat { t } _ { 0 } = ( 2 c L ^ { 2 } ) ^ { \frac { 1 } { c \beta + 1 } } e ^ { \frac { c \beta ( T - i ) } { c \beta + 1 } } N ^ { \frac { c \beta ( T - i + 1 ) } { c \beta + 1 } } . } \end{array}
$$

For integer SGD steps, apply Equation (17) to the unrolled sum in Equation (18), after using $( N / t ) ^ { ( 1 - \mathrm { { i } } / n ) c \beta } \leq ( N / t ) ^ { c \beta }$ and the future-task factor. The corresponding function is

$$
D ( x ) = 2 c L ^ { 2 } ( e N ) ^ { c \beta ( T - i ) } N ^ { c \beta } x ^ { - c \beta - 1 } .
$$

When $0 < \widehat { t } _ { 0 } < N$ , evaluating the finite integral on both sides of $\widehat { t } _ { 0 }$ yields

$$
\begin{array} { r l } & { \quad \mathbb { E } | f ( w _ { T } ; z ) - f ( w _ { T } ^ { \prime } ; z ) | } \\ & { \le \displaystyle \frac { \widehat { t } _ { 0 } } { n } + \frac { 2 L ^ { 2 } ( e N ) ^ { c \beta ( T - i ) } } { \beta n } \left[ \left( \frac { N } { \widehat { t } _ { 0 } } \right) ^ { c \beta } - 1 \right] } \\ & { = \displaystyle \frac { c \beta + 1 } { n c \beta } \widehat { t } _ { 0 } - \frac { 2 L ^ { 2 } ( e N ) ^ { c \beta ( T - i ) } } { \beta n } . } \end{array}
$$

This evaluation also covers $\widehat { t } _ { 0 } < 1$ , since the comparison in Equation (17) permits a zero integer cutoff. When $\widehat { t } _ { 0 } \geq N$ the integral equals N, giving the bound $N / n .$ . The condition $\widehat { t } _ { 0 } < N$ is equivalent to $2 c L ^ { \stackrel { \smile } { 2 } } ( e N ) ^ { c \beta ( T - i ) } < N$ , which proves part (a). The two expressions coincide at equality. The case $L = 0$ follows directly.

(b) under continuously decaying learning rates $\begin{array} { r l } { \eta _ { \tau , t } } & { { } = } \end{array}$ $\frac { c } { ( \tau - 1 ) N + t + 1 }$

$$
\begin{array} { r l } & { \Delta _ { T , t + 1 } \leq ( 1 + \eta _ { T , t } \beta ) \Delta _ { T , t } } \\ & { \qquad = \left( 1 + \frac { c \beta } { ( T - 1 ) N + t + 1 } \right) \Delta _ { T , t } } \\ & { \qquad \leq \exp \left( \frac { c \beta } { ( T - 1 ) N + t + 1 } \right) \Delta _ { T , t } . } \end{array}
$$

For any $j \in \{ i + 1 , \ldots , T \}$ , unrolling this recursion from N down to 0 yields

$$
\begin{array} { r l } { \Delta _ { j , N } \leq \displaystyle \prod _ { k = 0 } ^ { N - 1 } \exp \left( \frac { c \beta } { ( j - 1 ) N + k + 1 } \right) \Delta _ { j , 0 } } & { } \\ { = \displaystyle \exp \left( c \beta \displaystyle \sum _ { k = 0 } ^ { N - 1 } \frac { 1 } { ( j - 1 ) N + k + 1 } \right) \Delta _ { j , 0 } } & { } \\ { = \displaystyle \exp \left( c \beta \displaystyle \sum _ { k = 1 } ^ { N } \frac { 1 } { ( j - 1 ) N + k } \right) \Delta _ { j , 0 } } & { } \\ { \leq \displaystyle \exp \left( c \beta \ln \left( \frac { j } { j - 1 } \right) \right) \Delta _ { j , 0 } } & { } \\ { = \displaystyle \left( \frac { j } { j - 1 } \right) ^ { c \beta } \Delta _ { j , 0 } . } \end{array}
$$

Under the CL protocol, the initialization of each task is the final model from the previous task, namely $w _ { k , 0 } = w _ { k - 1 , N }$ and $w _ { k , 0 } ^ { \prime } = w _ { k - 1 , N } ^ { \prime }$ for any $k \in \{ 2 , \ldots , T \}$ . Hence, $\delta _ { k , 0 } = \delta _ { k - 1 , N }$ Therefore,

$$
\begin{array} { l } { \displaystyle \Delta _ { T , N } \leq \left( \frac { T } { T - 1 } \right) ^ { c \beta } \Delta _ { T , 0 } } \\ { \displaystyle = \left( \frac { T } { T - 1 } \right) ^ { c \beta } \Delta _ { T - 1 , N } } \\ { \displaystyle \quad \leq \ldots } \\ { \displaystyle \quad \leq \left( \frac { T } { T - 1 } \right) ^ { c \beta } \cdots \left( \frac { i + 1 } { i } \right) ^ { c \beta } \Delta _ { i , N } } \\ { \displaystyle = \left( \frac { T } { i } \right) ^ { c \beta } \Delta _ { i , N } . } \end{array}
$$

Based on Equation (19), we can further obtain

$$
\Delta _ { T , N } \leq \frac { 2 L } { \beta n } \left[ \left( \frac { T N } { ( i - 1 ) N + t _ { 0 } } \right) ^ { c \beta } - \left( \frac { T } { i } \right) ^ { c \beta } \right] .
$$

Substituting this bound into Equation (26), we have

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left| f ( w _ { T } ; z ) - f ( w _ { T } ^ { \prime } ; z ) \right| } \\ { \displaystyle \leq \frac { t _ { 0 } } { n } + \frac { 2 L ^ { 2 } } { \beta n } \left[ \left( \frac { T N } { ( i - 1 ) N + t _ { 0 } } \right) ^ { c \beta } - \left( \frac { T } { i } \right) ^ { c \beta } \right] . } \end{array}\tag{27}
$$

Differentiating the right-hand side gives the stationary point of its continuous relaxation:

$$
\widehat { t } _ { 0 } = ( 2 c L ^ { 2 } ) ^ { \frac { 1 } { c \beta + 1 } } ( T N ) ^ { \frac { c \beta } { c \beta + 1 } } - ( i - 1 ) N .
$$

The integer cutoff is again justified by Equation (17), applied to the finite-sum recursion with

$$
D ( x ) = 2 c L ^ { 2 } ( T N ) ^ { c \beta } \big ( ( i - 1 ) N + x \big ) ^ { - c \beta - 1 } .
$$

If $\widehat { t } _ { 0 } \leq 0 .$ , choose $t _ { 0 } = 0$ in Equation (27). This gives

$$
\mathbb { E } \left| f ( w _ { T } ; z ) - f ( w _ { T } ^ { \prime } ; z ) \right| \leq \frac { 2 L ^ { 2 } } { \beta n } \left[ \left( \frac { T } { i - 1 } \right) ^ { c \beta } - \left( \frac { T } { i } \right) ^ { c \beta } \right] .
$$

If $0 < \widehat { t } _ { 0 } < N$ , evaluating the finite integral at this threshold gives

$$
\begin{array} { l } { \displaystyle \mathbb { E } | f ( w _ { T } ; z ) - f ( w _ { T } ^ { \prime } ; z ) | } \\ { \displaystyle \leq \frac { c \beta + 1 } { n c \beta } \big ( ( i - 1 ) N + \widehat { t } _ { 0 } \big ) - \frac { ( i - 1 ) N } { n } - \frac { 2 L ^ { 2 } } { \beta n } \left( \frac T i \right) ^ { c \beta } . } \end{array}
$$

$\mathrm { I f } \ \widehat { t } _ { 0 } \geq N$ , the integral equals N, yielding $N / n$ . The conditions $\widehat { t } _ { 0 } \leq 0$ and ${ \widehat { t } } _ { 0 } \geq N$ are respectively equivalent to $2 c L ^ { 2 } \leq$ $( i - 1 ) N ( ( i - 1 ) / T ) ^ { c \beta }$ and $\bar { 2 } c L ^ { 2 } \geq \dot { i } N ( \bar { i } / T ) ^ { c \beta }$ . These cases establish part (b), and the adjacent formulas agree at both boundaries. For i = 1, the local training is the same as in part (a), while the future-task factor is $T ^ { c \beta }$ . Substituting this factor into the analysis of part (a) gives the stated two-branch bound for the continuously decaying schedule.

□

## APPENDIX D EMPIRICAL FORGETTING

Theorem. Assume that the loss function $f ( \cdot ; z )$ satisfies Assumptions 1 and 2. Suppose that SGD is run for N steps on each task $\tau \in [ T ]$ with learning rates $\eta _ { \tau , t }$ . Then, (a) under constant learning rates $\eta _ { \tau , t } = c \mathrm { : }$

$$
\tilde { \epsilon } _ { T , i } \leq - \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 } N ( T - i ) ;
$$

(b) under intra-task decaying learning rates $\begin{array} { r } { \eta _ { \tau , t } = \frac { c } { t + 1 } \colon } \end{array}$

$$
\tilde { \epsilon } _ { T , i } \leq - \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 } \left( 2 - \frac { 1 } { N } \right) ( T - i ) ;
$$

(c) under continuously decaying learning rates $\begin{array} { r l } { \eta _ { \tau , t } } & { { } = } \end{array}$ $\frac { c } { ( \tau - 1 ) N + t + 1 } :$

$$
\tilde { \epsilon } _ { T , i } \leq - \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 } \left( \frac { 1 } { i N } - \frac { 1 } { T N } \right) .
$$

Proof. We first consider the impact on task i of performing one training step on task $\tau > i ,$ and then accumulate the empirical risk changes to obtain the final bound. Let $\mathcal { F } _ { \tau , t }$ contain all training datasets, the initialization, and the sampling indices before the current update. Since the current index $i _ { t }$ is sampled independently and uniformly from [n], we have

$$
\begin{array} { r l } & { \mathbb { E } [ \nabla f ( w _ { \tau , t } ; z _ { \tau , i _ { t } } ) \mid \mathcal { F } _ { \tau , t } ] = \nabla R _ { S _ { \tau } } ( w _ { \tau , t } ) , } \\ & { \mathbb { E } \big [ \| \nabla f ( w _ { \tau , t } ; z _ { \tau , i _ { t } } ) \| ^ { 2 } \mid \mathcal { F } _ { \tau , t } \big ] = \displaystyle \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \| \nabla f ( w _ { \tau , t } ; z _ { \tau , j } ) \| ^ { 2 } . } \end{array}
$$

The function $R _ { S _ { i } }$ is β-smooth. Applying the smoothness inequality to the SGD update and taking the conditional expectation gives

$$
\begin{array} { r l } & { \quad \mathbb { E } [ R _ { S _ { i } } ( w _ { \tau , t + 1 } ) - R _ { S _ { i } } ( w _ { \tau , t } ) \ \vert \ { \mathcal { F } } _ { \tau , t } ] } \\ & { \leq - \eta _ { \tau , t } \left. \nabla R _ { S _ { i } } ( w _ { \tau , t } ) , \nabla R _ { S _ { \tau } } ( w _ { \tau , t } ) \right. } \\ & { \quad \quad + \frac { \beta \eta _ { \tau , t } ^ { 2 } } { 2 n } \displaystyle \sum _ { j = 1 } ^ { n } \| \nabla f ( w _ { \tau , t } ; z _ { \tau , j } ) \| ^ { 2 } } \\ & { \quad \quad \leq - \eta _ { \tau , t } \left. \nabla R _ { S _ { i } } ( w _ { \tau , t } ) , \nabla R _ { S _ { \tau } } ( w _ { \tau , t } ) \right. + \frac { \beta L ^ { 2 } \eta _ { \tau , t } ^ { 2 } } { 2 } , } \end{array}
$$

where the last inequality uses the gradient bound implied by Assumption 1. Taking the full expectation and accumulating from $t = 0$ to $N - 1$ , we have

$$
\begin{array} { r l } & { \quad \mathbb { E } \left[ R _ { S _ { i } } ( w _ { \tau } ) - R _ { S _ { i } } ( w _ { \tau - 1 } ) \right] } \\ & { \leq - \displaystyle \sum _ { t = 0 } ^ { N - 1 } \eta _ { \tau , t } \mathbb { E } \left. \nabla R _ { S _ { i } } ( w _ { \tau , t } ) , \nabla R _ { S _ { \tau } } ( w _ { \tau , t } ) \right. + \frac { \beta L ^ { 2 } } { 2 } \displaystyle \sum _ { t = 0 } ^ { N - 1 } \eta _ { \tau , t } ^ { 2 } . } \end{array}
$$

Then, summing over the subsequent tasks gives

$$
\tilde { \epsilon } _ { T , i } \leq - \mathcal { G } _ { T , i } + \frac { \beta L ^ { 2 } } { 2 } \sum _ { \tau = i + 1 } ^ { T } \sum _ { t = 0 } ^ { N - 1 } \eta _ { \tau , t } ^ { 2 } .\tag{28}
$$

We next discuss the different learning rates. Under constant learning rates $\eta _ { \tau , t } = c ,$ , we have

$$
\tilde { \epsilon } _ { T , i } \leq - \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 } N ( T - i ) .
$$

Under intra-task decaying learning rates $\eta _ { \tau , t } = c / ( t + 1 )$ , we have

$$
\begin{array} { r l r } {  { \tilde { \epsilon } _ { T , i } \leq - \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 } ( T - i ) \sum _ { t = 0 } ^ { N - 1 } \frac { 1 } { ( t + 1 ) ^ { 2 } } } } \\ & { } & \\ & { } & { \leq - \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 } ( T - i ) ( 1 + \int _ { 1 } ^ { N } \frac { d x } { x ^ { 2 } } ) } \\ & { } & \\ & { } & { = - \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 } ( 2 - \frac { 1 } { N } ) ( T - i ) . } \end{array}
$$

Under continuously decaying learning rates $\eta _ { \tau , t } = c / ( \tau -$ $1 ) N + t + 1 )$ , we have

$$
\begin{array} { r l r } {  { \tilde { \epsilon } _ { T , i } \leq - \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 } \sum _ { k = i N + 1 } ^ { N N } \frac { 1 } { k ^ { 2 } } } } \\ & { } & { \leq - \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 } \int _ { i N } ^ { T N } \frac { d x } { x ^ { 2 } } } \\ & { } & { = - \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 } ( \frac { 1 } { i N } - \frac { 1 } { T N } ) . } \end{array}
$$

This completes the proof.

## APPENDIX E FORGETTING IN EXPECTATION

## A. Proof of Theorem 8

Proof. By Equation (4) and Theorems 1 and 4,

$$
F _ { T } \leq \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } \left( \epsilon _ { i , i } + \epsilon _ { T , i } + \tilde { \epsilon } _ { T , i } \right) .
$$

We apply the convex stability bounds and the EF bound under each learning-rate schedule.

(a) under constant learning rates $\eta _ { \tau , t } = c \mathrm { : }$

$$
\begin{array} { r l } & { F _ { T } < \displaystyle \frac { 1 } { T - 1 } \displaystyle \sum _ { i = 1 } ^ { T - 1 } ( \epsilon _ { i , i } + \epsilon _ { T , i } + \epsilon _ { T , i } ) } \\ & { \leq - \frac { 1 } { T - 1 } \displaystyle \sum _ { i = 1 } ^ { T - 1 } \delta \epsilon _ { i , i } } \\ & { \quad \quad + \frac { 1 } { T - 1 } \displaystyle \sum _ { i = 1 } ^ { T - 1 } \left[ \frac { \partial ^ { 2 } \bar { F } ^ { 2 } } { 2 } N ( T - \bar { u } ) + \frac { 4 \epsilon \bar { T } ^ { 2 } N } { N } \right] } \\ & { = - \frac { 1 } { T - 1 } \displaystyle \sum _ { i = 1 } ^ { T - 1 } \delta \epsilon _ { T , i } } \\ & { \quad \quad + \frac { 1 } { T - 1 } \displaystyle \left[ \frac { \partial ^ { 2 } \bar { F } ^ { 2 } N } { 2 } T ( T - 1 ) + \frac { 4 \epsilon \bar { T } ^ { 2 } N ( T - 1 ) } { \alpha } \right] } \\ & { = - \frac { 1 } { T - 1 } \displaystyle \sum _ { i = 1 } ^ { T - 1 } \delta \epsilon _ { T , i } + \frac { \partial \bar { \epsilon } ^ { 2 } \bar { L } ^ { 2 } N T } { \partial } + \frac { 4 \epsilon \bar { L } ^ { 2 } N } { \alpha } . } \end{array}
$$

(b) under intra-task decaying learning rates $\textstyle \eta _ { \tau , t } = { \frac { c } { t + 1 } } \colon$

$$
\begin{array} { r l } {  { F _ { \mathrm { Y } } \sum _ { \ell = 1 } ^ { \infty } \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { n - 1 } \omega _ { i , i } + \bar { \omega } _ { i , i } + \bar { \omega } _ { i , i } } } \\ & { \le - \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { \infty } \hat { \mathcal { G } } _ { i , i } + \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { - 1 } \Bigg [ \frac { \hat { \omega } _ { i } ^ { 2 } \hat { F } _ { i } ^ { 2 } } { 2 } \bigg ( 2 - \frac { 1 } { N } \bigg ) } \\ & { \quad \times ( T - i ) + \frac { \mathrm { d } \hat { \omega } _ { i } ^ { 2 } \hat { \mathbf { A } } _ { i } ( \hat { \omega } _ { i } ) } { 2 } \Bigg ] } \\ & { = - \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T } \frac { 1 } { \hat { \mathcal { G } } _ { i , i } } + \frac { 1 } { T - 1 } [ \frac { \partial \hat { F } _ { i } ^ { 2 } \hat { F } _ { i } ^ { 2 } } { 2 } \bigg ( 2 - \frac { 1 } { N } \bigg )  } \\ & { \quad \times  \frac { T ( T - 1 ) } { T - 1 } + \frac { \mathrm { d } \hat { \omega } _ { i } ^ { 2 } \hat { \mathbf { A } } _ { i } ( \hat { \omega } _ { i } ) ( T - 1 ) } { \hat { \mathcal { G } } _ { i } } ] } \\ & { = - \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { \infty } \hat { \mathcal { G } } _ { i , i } + \frac { \beta \hat { \omega } _ { i } ^ { 2 } \hat { \mathcal { G } } _ { i } ^ { 2 } T } { 4 } \Big ( 2 - \frac { 1 } { N } \Big ) } \\ &  \quad \times \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { \infty } \hat { \mathcal { G } } _ { i , i } + \frac  \beta \hat  \omega \end{array}
$$

For the continuous schedule, we use the finite harmonic-sum estimate

$$
\begin{array} { c } { \displaystyle \sum _ { i = 1 } ^ { T - 1 } \displaystyle \frac { 1 } { i } - \frac { T - 1 } { T } \leq 1 + \int _ { 1 } ^ { T - 1 } \displaystyle \frac { d u } { u } - \frac { T - 1 } { T } } \\ { \displaystyle = \ln ( T - 1 ) + \displaystyle \frac { 1 } { T } . } \end{array}\tag{29}
$$

This also holds for $T = 2 ,$ , when the integral is zero.

(c) under continuously decaying learning rates $\begin{array} { r l } { \eta _ { \tau , t } } & { { } = } \end{array}$ $\frac { c } { ( \tau - 1 ) N + t + 1 } \colon$

$$
\begin{array} { r l } & { \mathcal { F } _ { \overline { { S } } } ^ { \prime } } \\ & { \mathcal { F } _ { \overline { { S } } } ^ { \prime } } \\ & { \mathcal { F } _ { \overline { { S } } } ^ { \prime } } \\ & { \mathcal { F } _ { \overline { { S } } } ^ { \prime } = \frac { 1 } { T } \sum _ { \overline { { i } } \in \mathcal { D } _ { \overline { { S } } } ^ { \prime } } \frac { 1 } { T } \sum _ { \overline { { i } } \in \mathcal { D } _ { \overline { { S } } } ^ { \prime } } \frac { 1 } { T } \sum _ { \overline { { i } } \in \mathcal { D } _ { \overline { { S } } } ^ { \prime } } \frac { 1 } { 2 T } \left( \frac { 1 } { \overline { { S } } } - \frac { 1 } { T } \right) } \\ & { \mathcal { F } _ { \overline { { S } } } ^ { \prime } - \frac { 1 } { T } \sum _ { \overline { { i } } \in \mathcal { D } _ { \overline { { S } } } ^ { \prime } } \frac { 1 } { T } \sum _ { \overline { { i } } \in \mathcal { D } _ { \overline { { S } } } ^ { \prime } } \frac { 1 } { T } \left[ \frac { 1 } { 2 T } \sum _ { \overline { { i } } \in \overline { { S } } } \frac { 1 } { T } \right] } \\ & { \mathcal { F } _ { \overline { { S } } } ^ { \prime } \left( \overline { { S } } ^ { \prime } \right) _ { \overline { { D } } } ( \overline { { S } } ^ { \prime } ) _ { \overline { { S } } } \frac { 1 } { T } \sum _ { \overline { { i } } \in \mathcal { D } _ { \overline { { S } } } ^ { \prime } } \frac { 1 } { T } \left[ \frac { 1 } { 2 T } \sum _ { \overline { { i } } \in \overline { { S } } } \frac { 1 } { T } \right] } \\ &  - \frac { 1 } { T } \sum _ { \overline { { i } } \in \overline { { S } } } \frac { 1 } { T } \sum _  \overline { { i } } \in \mathcal { D } _  \overline  \end{array}
$$

Here, the generalization terms use $\textstyle \sum _ { t = 1 } ^ { N } 1 / t \leq \ln ( e N )$ for task 1 and the telescoping identity

$$
\sum _ { i = 2 } ^ { T - 1 } \ln \left( \frac { i } { i - 1 } \right) = \ln ( T - 1 ) .
$$

The last inequality follows from Equation (29). When $T = 2 ,$ the logarithmic sum is empty, so the same calculation applies. This completes the proof. □

## B. Proof of Theorem 9

Proof. By Equation (4) and the generalization bounds,

$$
F _ { T } \leq \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } \left( \epsilon _ { i , i } + \epsilon _ { T , i } + \tilde { \epsilon } _ { T , i } \right) .
$$

We evaluate these three contributions under each learning-rate schedule.

(a) Under intra-task decaying learning rates $\begin{array} { r } { \eta _ { \tau , t } ~ = ~ \frac { c } { t + 1 } } \end{array}$ Theorem 7 gives

$$
\begin{array} { r l } & { \quad \displaystyle \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } \tilde { \epsilon } _ { T , i } } \\ & { \le - \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 ( T - 1 ) } \left( 2 - \frac { 1 } { N } \right) \sum _ { i = 1 } ^ { T - 1 } ( T - i ) } \\ & { = - \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } T } { 4 } \left( 2 - \frac { 1 } { N } \right) . } \end{array}
$$

Under the condition $2 c L ^ { 2 } ( e N ) ^ { c \beta ( T - 1 ) } < N$ , for every $i \in$ $[ T - 1 ]$ we have

$$
2 c L ^ { 2 } \leq 2 c L ^ { 2 } ( e N ) ^ { c \beta ( T - i ) } \leq 2 c L ^ { 2 } ( e N ) ^ { c \beta ( T - 1 ) } < N .
$$

Thus, the interior expressions in Theorems 3 and 6 apply to all historical tasks. The IGG bound is the same for every task, so

$$
\frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } \epsilon _ { i , i } \le \frac { c \beta + 1 } { n c \beta } ( 2 c L ^ { 2 } ) ^ { \frac { 1 } { c \beta + 1 } } N ^ { \frac { c \beta } { c \beta + 1 } } - \frac { 2 L ^ { 2 } } { \beta n } .
$$

For CGG, setting $\textit { j } = \textit { T } - \textit { i }$ and summing the interior expressions gives

$$
\begin{array} { l } { { \displaystyle \sum _ { i = 1 } ^ { T - 1 } \epsilon _ { T , i } \leq \frac { c \beta + 1 } { n c \beta } ( 2 c L ^ { 2 } ) ^ { \frac { 1 } { c \beta + 1 } } N ^ { \frac { c \beta } { c \beta + 1 } } \sum _ { j = 1 } ^ { T - 1 } ( e N ) ^ { \frac { c \beta j } { c \beta + 1 } } } } \\ { { \displaystyle \qquad - \frac { 2 L ^ { 2 } } { \beta n } \sum _ { j = 1 } ^ { T - 1 } ( e N ) ^ { c \beta j } . } } \end{array}
$$

The two geometric sums are

$$
\begin{array} { l } { { \displaystyle \sum _ { j = 1 } ^ { T - 1 } ( e N ) ^ { \frac { c \beta j } { c \beta + 1 } } = \frac { ( e N ) ^ { \frac { c \beta ( T - 1 ) } { c \beta + 1 } } - 1 } { 1 - ( e N ) ^ { - \frac { c \beta } { c \beta + 1 } } } , } } \\ { { \displaystyle \sum _ { j = 1 } ^ { T - 1 } ( e N ) ^ { c \beta j } = \frac { ( e N ) ^ { c \beta ( T - 1 ) } - 1 } { 1 - ( e N ) ^ { - c \beta } } . } } \end{array}
$$

Both denominators are positive since $e N > 1$ and $c \beta > 0$ Dividing the CGG sum by $T - 1$ , adding the IGG and EF contributions, and collecting the common coefficients gives Equation (9). When $T = 2 .$ , each geometric sum contains one term, and the same identities apply.

(b) Under continuously decaying learning rates $\begin{array} { r l } { \eta _ { \tau , t } } & { { } = } \end{array}$ $\frac { c } { ( \tau - 1 ) N + t + 1 } \colon$

The unrolled CGG recursion in Appendix C-C gives, for every integer $t _ { 0 } \in \{ 0 , \ldots , N \}$

$$
\epsilon _ { T , i } \leq \frac { 1 } { n } \left[ t _ { 0 } + 2 c L ^ { 2 } ( T N ) ^ { c \beta } \sum _ { k = t _ { 0 } + 1 } ^ { N } \frac { 1 } { ( ( i - 1 ) N + k ) ^ { c \beta + 1 } } \right] .
$$

The summand is nonnegative and nonincreasing in k. Applying Equation (17) and substituting $v = ( i - 1 ) N + u$ therefore gives, for each $i \in [ T - 1 ]$

$$
\epsilon _ { T , i } \leq \frac { 1 } { n } \int _ { ( i - 1 ) N } ^ { i N } \operatorname* { m i n } \left. 1 , \frac { 2 c L ^ { 2 } ( T N ) ^ { c \beta } } { v ^ { c \beta + 1 } } \right. d v .
$$

The integrand is the same for all tasks. Summing the integrals over their adjacent intervals yields

$$
\sum _ { i = 1 } ^ { T - 1 } \epsilon _ { T , i } \leq \frac { 1 } { n } \int _ { 0 } ^ { ( T - 1 ) N } \operatorname* { m i n } \left\{ 1 , \frac { 2 c L ^ { 2 } ( T N ) ^ { c \beta } } { v ^ { c \beta + 1 } } \right\} d v .
$$

The two arguments of the minimum are equal at

$$
\widehat { v } _ { 0 } = ( 2 c L ^ { 2 } ) ^ { \frac { 1 } { c \beta + 1 } } ( T N ) ^ { \frac { c \beta } { c \beta + 1 } } .
$$

If $2 c L ^ { 2 } T ^ { c \beta } < N ( T - 1 ) ^ { c \beta + 1 }$ , then ${ \widehat { v } } _ { 0 } < ( T - 1 ) N$ . Splitting the integral at $\widehat { v } _ { 0 }$ gives

$$
\begin{array} { r l } & { \displaystyle \sum _ { i = 1 } ^ { T - 1 } \epsilon _ { T , i } \leq \frac { \widehat v _ { 0 } } { n } + \frac { 2 c L ^ { 2 } ( T N ) ^ { c \beta } } { n } \displaystyle \int _ { \widehat v _ { 0 } } ^ { ( T - 1 ) N } v ^ { - c \beta - 1 } d v } \\ & { \qquad = \displaystyle \frac { \widehat v _ { 0 } } { n } + \frac { 2 L ^ { 2 } ( T N ) ^ { c \beta } } { \beta n } \left[ \widehat v _ { 0 } ^ { - c \beta } - ( ( T - 1 ) N ) ^ { - c \beta } \right] } \\ & { \qquad = \displaystyle \frac { c \beta + 1 } { n c \beta } ( 2 c L ^ { 2 } ) ^ { \frac { 1 } { c \beta + 1 } } ( T N ) ^ { \frac { c \beta } { c \beta + 1 } } - \frac { 2 L ^ { 2 } } { \beta n } \left( \frac { T } { T - 1 } \right) ^ { c \beta } . } \end{array}
$$

If $2 c L ^ { 2 } T ^ { c \beta } \geq N ( T - 1 ) ^ { c \beta + 1 }$ , then $\widehat { v } _ { 0 } \geq ( T - 1 ) N$ , so the integrand equals 1 throughout the interval and

$$
\sum _ { i = 1 } ^ { T - 1 } \epsilon _ { T , i } \leq \frac { ( T - 1 ) N } { n } .
$$

For IGG, the same integer-cutoff comparison gives

$$
\epsilon _ { 1 , 1 } \leq \frac { 1 } { n } \int _ { 0 } ^ { N } \operatorname* { m i n } \left\{ 1 , \frac { 2 c L ^ { 2 } N ^ { c \beta } } { u ^ { c \beta + 1 } } \right\} d u .
$$

If $\begin{array} { r l r } { 2 c L ^ { 2 } } & { { } < } & { N , } \end{array}$ , the crossing point is $\begin{array} { r l } { \widehat { u } _ { 0 } } & { { } = } \end{array}$ $( 2 c L ^ { 2 } ) ^ { 1 / ( c \beta + 1 ) } N ^ { c \beta / ( c \beta + 1 ) } \quad < \quad N$ . Splitting the integral at this point gives

$$
\begin{array} { c } { \displaystyle \epsilon _ { 1 , 1 } \leq \frac { \widehat { u } _ { 0 } } { n } + \frac { 2 c L ^ { 2 } N ^ { c \beta } } { n } \int _ { \widehat { u } _ { 0 } } ^ { N } u ^ { - c \beta - 1 } d u } \\ { \displaystyle = \frac { c \beta + 1 } { n c \beta } ( 2 c L ^ { 2 } ) ^ { \frac { 1 } { c \beta + 1 } } N ^ { \frac { c \beta } { c \beta + 1 } } - \frac { 2 L ^ { 2 } } { \beta n } . } \end{array}
$$

If $2 c L ^ { 2 } \geq N$ , the integrand equals 1 on $[ 0 , N ]$ , giving $\epsilon _ { 1 , 1 } \leq$ $N / n$ . For every $i \geq 2 ,$ , choosing the integer cutoff $t _ { 0 } = 0$ in Equation (20) gives

$$
\epsilon _ { i , i } \leq \frac { 2 L ^ { 2 } } { \beta n } \left[ \left( 1 + \frac { 1 } { i - 1 } \right) ^ { c \beta } - 1 \right] .
$$

For $0 \leq x \leq$ ln 2, convexity of $e ^ { c \beta x }$ implies

$$
e ^ { c \beta x } - 1 \leq { \frac { 2 ^ { c \beta } - 1 } { \ln 2 } } x .
$$

Taking $x = \ln ( 1 + 1 / k )$ and summing over $k = 1 , \ldots , T - 2 ,$ we obtain

$$
\begin{array} { r l } & { \displaystyle \sum _ { i = 2 } ^ { T - 1 } \epsilon _ { i , i } \leq \frac { 2 L ^ { 2 } } { \beta n } \sum _ { k = 1 } ^ { T - 2 } \left[ \left( 1 + \frac { 1 } { k } \right) ^ { c \beta } - 1 \right] } \\ & { \qquad \leq \frac { 2 L ^ { 2 } } { \beta n } \frac { 2 ^ { c \beta } - 1 } { \ln 2 } \sum _ { k = 1 } ^ { T - 2 } \ln \left( \frac { k + 1 } { k } \right) } \\ & { \qquad = \frac { 2 L ^ { 2 } } { \beta n } \frac { 2 ^ { c \beta } - 1 } { \ln 2 } \ln ( T - 1 ) . } \end{array}
$$

For $T = 2$ , these sums are empty and both sides are zero. Thus, when $2 c L ^ { 2 } < N$

$$
\begin{array} { l } { \displaystyle \sum _ { i = 1 } ^ { T - 1 } \epsilon _ { i , i } \leq \frac { c \beta + 1 } { n c \beta } ( 2 c L ^ { 2 } ) ^ { \frac { 1 } { c \beta + 1 } } N ^ { \frac { c \beta } { c \beta + 1 } } } \\ { \displaystyle \qquad + \frac { 2 L ^ { 2 } } { \beta n } \left[ \frac { 2 ^ { c \beta } - 1 } { \ln 2 } \ln ( T - 1 ) - 1 \right] . } \end{array}
$$

When $2 c L ^ { 2 } \geq N$ , adding the first-task endpoint bound instead gives

$$
\sum _ { i = 1 } ^ { T - 1 } \epsilon _ { i , i } \leq \frac { N } { n } + \frac { 2 L ^ { 2 } } { \beta n } \frac { 2 ^ { c \beta } - 1 } { \ln 2 } \ln ( T - 1 ) .
$$

Finally, the finite harmonic sum satisfies

$$
\begin{array} { c } { \displaystyle \sum _ { i = 1 } ^ { T - 1 } \displaystyle \frac { 1 } { i } - \frac { T - 1 } { T } \leq 1 + \int _ { 1 } ^ { T - 1 } \displaystyle \frac { d u } { u } - \frac { T - 1 } { T } } \\ { \displaystyle = \ln ( T - 1 ) + \displaystyle \frac { 1 } { T } . } \end{array}
$$

Combining this inequality with Theorem 7 gives

$$
\begin{array} { l } { \displaystyle \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } \tilde { \epsilon } _ { T , i } } \\ { \leq - \displaystyle \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } \mathcal { G } _ { T , i } + \frac { \beta c ^ { 2 } L ^ { 2 } } { 2 N ( T - 1 ) } \left( \ln ( T - 1 ) + \frac { 1 } { T } \right) . } \end{array}
$$

The two IGG cases and the two cumulative CGG cases give the four combinations in Equation (10) after division by $T - 1$ and addition of the EF bound. In particular, when both cutoff points reach their endpoints, their combined contribution is

$$
{ \frac { 1 } { T - 1 } } \left( { \frac { N } { n } } + { \frac { ( T - 1 ) N } { n } } \right) = { \frac { N T } { n ( T - 1 ) } } .
$$

At $2 c L ^ { 2 } = N .$ , the first-task interior expression equals $N / n$ At $2 c L ^ { 2 } T ^ { c \beta } = N ( T - 1 ) ^ { c \beta + 1 }$ , the cumulative CGG interior expression equals $( T - 1 ) N / n$ . Thus the expressions agree on each boundary, including their intersection. The empty IGG tail at $T = 2$ was handled above, so all stated cases are covered. This completes the proof. □

## APPENDIX F

EXPERIMENTS ON SPLIT MINIIMAGENET WITH RESNET-18

Experimental setting on Split MiniImageNet. The Split MiniImageNet dataset is constructed by dividing the 100 classes of MiniImageNet into 20 sequential tasks, each containing 5 classes. We use a ResNet-18 backbone, following a similar architecture to that used by Lopez-Paz and

![](images/1a539726de697207c7f975c4cf3c8466e376bcced480f4c584b819d240cab999.jpg)  
Fig. 5: IGG, CGG, EF, $F _ { T } ,$ and ACC under different learning-rate schedules on the Split MiniImageNet benchmark. The top row shows results for different initial learning rates c, while the bottom row shows results for different numbers of training epochs per task.

Ranzato [35]. We vary the number of epochs per task in $\{ 1 0 , 2 5 , 5 0 , 7 5 , 1 0 0 \}$ , the initial learning rate in $\lbrace 1 0 ^ { - 4 } , 5 \times$ $\mathrm  \dot { 1 0 ^ { - 4 } } , 1 0 ^ { - 3 } , 5 \times 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 5 \times 1 0 ^ { - 2 } , \bar { 1 0 } ^ { - 1 } , 5 \times \bar { 1 0 } ^ { - 1 } \}$ , and the number of samples per task in {500, 1000, 1500, 2000}. To examine the effect of the number of epochs per task, we fix $c = 0 . 1$ and use the original training set setting. To examine the effect of the initial learning rate, we fix the number of epochs per task to 50 with the same training set setting. To examine the effect of the number of training samples per task, we fix $c = 0 . 1$ and use a common budget of $N = 1 6 0 0$ updates per task across all sample sizes. Each experiment is repeated with random seeds 1, 2, and 3, and the figures report the mean and standard deviation across the three runs.

Measurement of cumulative gradient alignment. For each task, we construct a measurement buffer by uniformly sampling batches without replacement from its fixed training-set evaluation partition. We retain four batches on Split CIFAR-100 and two batches on Split MiniImageNet, with up to 64 samples per batch, giving maximum buffer sizes of 256 and 128 samples per task, respectively. At step t of task $\tau ,$ let $g _ { \tau , t }$ denote the gradient of the mean training loss on the mini-batch actually used for the current SGD update. For each previous task $i < \tau ,$ we uniformly sample one batch from its buffer to obtain a gradient estimate $\widehat { g } _ { i , \tau , t }$ in evaluation mode using the corresponding task head. Both $\widehat { g } _ { i , \tau , t }$ and the current training gradient $g _ { \tau , t }$ are evaluated at the same parameters $w _ { \tau , t }$ before the update. We then compute

$$
\widehat { \mathcal { G } } _ { T , i } = \sum _ { \tau = i + 1 } ^ { T } \sum _ { t = 0 } ^ { N - 1 } \eta _ { \tau , t } \left. \widehat { g } _ { i , \tau , t } , g _ { \tau , t } \right. .
$$

Each inner product is multiplied by the learning rate used for that update and accumulated over all subsequent updates, without normalization by gradient norms, the number of updates, or the number of tasks. The buffers are used exclusively for measurement. We plot $- \widehat { \mathcal { G } } _ { T , i }$ alongside EF for each previous task $i ,$ reporting the mean and standard deviation across the three training seeds.

![](images/89c841853ed0d92ea4b31da43345459fa6f6f0a48b6e8aefe8fead54b4070c4d.jpg)  
Fig. 6: EF and CGA under different learning-rate schedules on Split MiniImageNet. C, ID, and CD denote constant, intra-task decaying, and continuously decaying learning rates, respectively. Curves show the means over three seeds, and shaded regions indicate one standard deviation.

Constant Intra-task decay Continuous decay  
![](images/13be7ffc8b0698acd1c4c9321e2ded8c2d7884e089d9a082e70b8af9bca526b4.jpg)

![](images/18b1c1cb446d3bdee6f8ead63824444849051f908d61e0af24328ddd52b87fcc.jpg)  
Training samples per task  
Fig. 7: IGG and CGG with different numbers of training samples per task on Split MiniImageNet.

![](images/e806bea12943a163a398cf6a322b9e06fc2bdd7595325c630cb6ade7462e4ff9.jpg)

![](images/bf11d51a454d1391b4d73cffe8634b34828fa7a8aaefbe0ffc5528337237b041.jpg)  
Fig. 8: ACC and BWT for original CL, GEM, and A-GEM under different learning-rate schedules on Split MiniImageNet. Higher ACC indicates better predictive performance, while higher BWT indicates less forgetting.