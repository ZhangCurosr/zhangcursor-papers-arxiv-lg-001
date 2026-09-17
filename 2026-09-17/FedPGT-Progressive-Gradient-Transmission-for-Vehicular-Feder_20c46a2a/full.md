# FedPGT: Progressive Gradient Transmission for Vehicular Federated Learning over Time-Varying Channels

Jintao Yan, Graduate Student Member, IEEE, Tan Chen, Yuxuan Sun, Member, IEEE, Sheng Zhou, Senior Member, IEEE and Zhisheng Niu, Fellow, IEEE

Abstract—Vehicular federated learning (VFL) enables privacypreserving collaborative model training for intelligent transportation systems, where communication resource allocation and gradient sparsification techniques have been explored to reduce communication overhead. However, vehicle mobility leads to rapidly varying channel conditions and transmission capacity, rendering predetermined resource allocation and sparsification decisions ineffective. In this paper, we propose FedPGT, a progressive gradient transmission scheme for VFL over time-varying channels, where vehicles progressively transmit high-magnitude gradient entries in response to instantaneous channel conditions. We establish a convergence bound that characterizes the impact of transmitted gradient entries and reveals diminishing-return behavior governed by a power-law decay. Motivated by this result, we formulate a stochastic optimization problem for online decision-making, where the main challenge lies in a cumulatively coupled, non-separable objective. To handle this challenge, we introduce per-slot surrogate transmission variables to decouple the long-term dependence across time slots and convert the original objective into an additive per-slot optimization problem, enabling a Lyapunov drift-plus-penalty approach for online scheduling. We further develop a low-complexity resource allocation algorithm for efficient online implementation. Experimental results demonstrate that the proposed scheme achieves a 3.65% accuracy improvement on the CIFAR-10 image classification task and a 12.66% reduction in average displacement error on the Argoverse trajectory prediction task compared with state-of-theart baselines, demonstrating its applicability to diverse learning tasks under highly dynamic vehicular environments.

Index Terms—Federated learning, mobility, vehicular networks, time-varying channels

## I. INTRODUCTION

The rapid development of vehicular networks has enabled a wide range of data-driven applications, such as cooperative perception, trajectory prediction, and intelligent route planning [1]. These applications rely on machine learning (ML) models that require continuous adaptation to cope with dynamic traffic conditions, evolving environments, and diverse driving behaviors. Traditional centralized learning frameworks require raw data to be uploaded to a central server for model training [2]– [4], which not only incurs excessive communication latency but also raises serious privacy concerns, particularly when sensitive sensor data are involved.

With the increasing deployment of on-board sensing and computing capabilities, vehicles are becoming capable of performing local model training using their own data. This trend has driven the adoption of federated learning in vehicular networks, where multiple vehicles collaboratively train a global model by exchanging model updates instead of raw data. Vehicular federated learning (VFL) enables timely model adaptation while preserving data privacy and has emerged as a promising paradigm for large-scale intelligent transportation systems [5], [6].

In a typical VFL process, each participating vehicle performs local training using its private data and periodically uploads model updates (e.g., gradients or parameter differences) to a roadside unit (RSU) or aggregation server. The server aggregates the received updates and broadcasts the updated global model to the vehicles for the next training round. This process repeats until convergence.

Despite its advantages, VFL faces significant challenges due to the highly dynamic nature of vehicular networks [7], [8]. First, vehicles remain within the coverage area of an RSU only for a limited sojourn time, which constrains the time available for model transmission. Second, wireless channel conditions fluctuate rapidly because of vehicle mobility, leading to time-varying uplink capacities and intermittent communication opportunities. As a result, reliable and efficient model transmission becomes a major bottleneck in VFL.

To alleviate the communication overhead, many existing works adopt model compression or gradient sparsification techniques, which reduce the number of transmitted parameters by retaining only the most informative components. In these approaches, the sparsification degree, which is defined as the number of gradient entries transmitted in each round, is typically fixed at the beginning of the model uploading [9], [10]. However, such a design is not well-suited for vehicular environments, where the achievable number of transmitted gradient entries during a round is highly uncertain due to rapidly varying channels and dynamic communication opportunities. Moreover, most existing sparsification-based convergence analyses assume a fixed set of participating devices [11], which does not hold in vehicular networks where the set of participating vehicles changes across training rounds. These limitations suggest that the number of transmitted gradient entries should adapt to the dynamically changing communication opportunities in vehicular networks, rather than being predetermined before transmission.

In this work, we propose FedPGT, a progressive gradient transmission scheme for VFL that adapts to time-varying communication conditions and dynamic vehicle participation. Instead of predefining the sparsification degree, vehicles progressively transmit high-magnitude gradient entries based on instantaneous communication conditions. The main contributions are summarized as follows:

• We establish a convergence analysis that characterizes the relationship between the number of transmitted gradient entries and the convergence performance of VFL, showing that increasing the number of transmitted gradient entries improves convergence performance with diminishing returns following a power-law decay. This result indicates that transmitting only a small number of high-magnitude gradient entries can preserve most of the convergence gain, which motivates progressive gradient transmission under limited communication opportunities.

• We propose a progressive gradient transmission scheme for VFL. We formulate a stochastic optimization problem, where the main challenge lies in a cumulatively coupled, non-separable objective. To handle this challenge, we introduce per-slot surrogate transmission variables to decouple the long-term dependence across time slots and construct an upper-bounding reformulation that converts the original problem into an additively separable perslot optimization problem. Based on this transformation, we develop a Lyapunov drift-plus-penalty approach for online scheduling and transmission decisions without requiring future channel information.

• We derive structural properties for the online resource allocation problem based on utility-based optimality conditions. In particular, we show that exclusive resource block (RB) allocation remains optimal despite relaxation and obtain an analytical power allocation expression. Based on these results, we develop a low-complexity online resource allocation algorithm for efficient implementation in dynamic vehicular environments.

• Experimental results show that, compared with state-ofthe-art benchmarks, the proposed scheme improves the test accuracy by 3.65% on the CIFAR-10 image classification task and reduces the average displacement error (ADE) by 12.66% on the Argoverse trajectory prediction task, demonstrating the effectiveness and applicability of FedPGT across diverse learning tasks.

The remainder of this paper is organized as follows. Section II presents a review of related works. Section III describes the system model. Section IV develops the convergence analysis and formulates the optimization problem. Section V introduces the proposed progressive gradient transmission scheme. The experimental evaluations are presented in Section VI, followed by the conclusions in Section VII.

## II. RELATED WORK

## A. Federated Learning in Vehicular Networks

Federated learning over wireless networks has attracted significant attention due to the tight coupling between learning performance and communication constraints. Comprehensive surveys such as [2]–[4] summarize recent advances in federated learning in wireless systems, highlighting challenges arising from limited bandwidth [12], [13], fading channels [14]–[16], and device heterogeneity [17].

Motivated by intelligent transportation applications, federated learning has been extended to vehicular networks, where high mobility and intermittent connectivity pose additional challenges. The authors of [18]–[21] study vehicle selection and resource optimization for vehicular edge FL, considering the limited sojourn time of vehicles. To enhance participation and robustness in dynamic environments, incentive mechanisms have been proposed to motivate vehicle collaboration in federated learning [22], [23]. In [24], mobility-aware decentralized federated learning frameworks have been developed to explicitly model vehicle mobility and multi-task learning dynamics in vehicular networks. Semi-asynchronous federated learning schemes have also been proposed to improve robustness against mobility-induced communication dynamics [25]. In addition, split federated learning and semantic communication enhanced frameworks have been investigated to improve communication efficiency in resource-constrained vehicular networks [26], [27]. These works demonstrate the potential of FL in vehicular systems but typically operate at the level of training rounds.

More recently, researchers have begun to investigate slotlevel communication and resource optimization in vehicular FL systems. The authors of [7] consider resource-constrained vehicular edge FL with highly mobile vehicles and analyze the impact of mobility on learning performance. The authors of [8] develop a dynamic scheduling scheme for vehicle-to-vehicle communications enhanced FL by adapting resource allocation to time-varying channels. These works explicitly consider slotlevel communication dynamics and online resource allocation. However, the structure of model update transmission is still typically predetermined before transmission. They do not explicitly model the progressive transmission of dominant gradient entries within each round under rapidly varying communication conditions, which is the key problem addressed in this work.

## B. Compression and Sparsification in Federated Learning

Communication-efficient learning has been widely studied through gradient and model compression techniques, among which sparsification transmits only a subset of gradient entries to reduce communication overhead. Early theoretical works typically assume a fixed number of transmitted gradient entries per round, such as top-k or random-k sparsification. A representative example is [28], which establishes convergence guarantees for k-sparsified SGD under standard assumptions. Qsparse-local-SGD [29] further combines sparsification with quantization and local updates, providing convergence results for both convex and non-convex objectives. More recent works investigate adaptive compression mechanisms that adjust the number of transmitted gradient entries across training rounds according to training dynamics or data heterogeneity. For instance, the authors of [30] formulate adaptive gradient sparsification in federated learning as an online decision problem to balance communication and computation. Hybrid compression schemes that integrate sparsification with quantization or encoding have also been explored to improve efficiency [11]. Nevertheless, most of these approaches determine the sparsification degree at the beginning of each training round and assume a fixed communication budget within the round.

Beyond compression-only designs, several studies jointly optimize sparsification and communication scheduling. The authors of [9] propose a flexible compression control mechanism for mobile edge FL to balance local computation and wireless transmission under energy constraints. The authors of [10] investigate joint gradient sparsification and device scheduling under limited communication resources. The authors of [31] further integrate sparsification with wireless resource allocation under privacy constraints. However, most existing joint designs assume that the sparsification structure or degree is decided a priori per round (or changes only at round boundaries), and they typically do not model the progressive, within-round transmission of dominant gradient entries under fast channel fluctuations, which is a key feature in vehicular networks considered in this work.

![](images/9f38b9938f659ec8f3fd6813ec4fcc2603693b4264669eb66cb9ee849baa5453.jpg)  
Fig. 1. The proposed FedPGT scheme.

## III. SYSTEM MODEL

## A. VFL Model

We study a VFL system, as depicted in Fig. 1, where an RSU coordinates the collaborative model training among vehicles within its coverage area through R rounds of training. During the $r ^ { \mathrm { t h } }$ training round, the set of vehicles participating in the model training are denoted by $\mathcal { N } ^ { ( r ) } = \{ 1 , 2 , \dots , \bar { N } ^ { ( r ) } \}$ Each vehicle $\textit { n } \in \mathcal { N } ^ { ( r ) }$ holds a local dataset drawn from distribution ${ \mathcal { X } } _ { n }$ over the input space $\mathcal { D } _ { n }$ . For a given sample $\mathbf { \Psi } _ { d } \in \mathcal { D } _ { n }$ , the loss function $f ( w , d )$ quantifies how well the model w fits the data. Accordingly, the local loss function for vehicle n is defined as the expected loss over its data distribution:

$$
F _ { n } ( \pmb { w } ) \triangleq \underset { { \pmb { d } } \sim { \pmb { \chi } } _ { n } } { \mathbb { E } } [ f ( \pmb { w } , { \pmb { d } } ) ] .
$$

Unlike conventional FL settings with a static client set, the participating vehicles in VFL vary over time due to mobility. We assume that vehicles participating in each round are sampled from an underlying distribution P. The global loss function is therefore defined as

$$
F ( \pmb { w } ) \triangleq \underset { n \sim \mathcal { P } } { \mathbb { E } } [ F _ { n } ( \pmb { w } ) ] .\tag{1}
$$

The objective is to minimize the global loss function in R communication rounds by optimizing the model parameters w. The index set of communication rounds is denoted by $\mathcal { R } = \{ 1 , 2 , \ldots , R \}$ . The VFL training procedure in each round comprises three key stages: local updates, gradient uploading, and model aggregation.

1) Local Updates: At the beginning of round $^ { r , }$ the RSU broadcasts the current global model $\bar { \mathbf { \chi } } _ { w } ( r - 1 )$ to all vehicles within its coverage. Upon reception, each vehicle $n \in \mathcal { N } ^ { ( r ) }$ computes its stochastic gradient based on stochastic gradient descent (SGD):

$$
\pmb { g } _ { n } ^ { ( r ) } = \frac { 1 } { | \mathcal { B } _ { n } ^ { ( r ) } | } \sum _ { \pmb { d } \in \mathcal { B } _ { n } ^ { ( r ) } } \nabla f \left( \pmb { w } ^ { ( r - 1 ) } , \pmb { d } \right) ,\tag{2}
$$

where $B _ { n } ^ { ( r ) } \subseteq { \mathcal { D } } _ { n }$ denotes a mini-batch sampled from ${ \mathcal { D } } _ { n } .$

2) Gradient Uploading: After local training, vehicles progressively transmit dominant gradient entries to the RSU according to the proposed transmission strategy. Let $\Psi _ { n } ^ { ( r ) } ( \cdot )$ denote the resulting gradient transmission operator for vehicle n in round r. The communication mechanism supporting this stage is described in Section III-C, while the detailed definition of $\bar { \Psi } _ { n } ^ { ( r ) } ( \cdot )$ is provided in Section III-D.

3) Model Aggregation: After receiving the uploaded gradi ents, the RSU aggregates them to update the global model:

$$
{ \pmb w } ^ { ( r ) } = { \pmb w } ^ { ( r - 1 ) } - \frac { \eta } { N ^ { ( r ) } } \sum _ { n \in \mathcal { N } ^ { ( r ) } } \Psi _ { n } ^ { ( r ) } ( { \pmb g } _ { n } ^ { ( r ) } ) ,\tag{3}
$$

and proceeds to the next training round.

## B. Computation Model

We adopt a standard computation model [32], [33] for local model updates. Let $N _ { \mathrm { f l o p } }$ denote the number of floating-point operations (FLOPs) required to process one training sample. For vehicle n in round $r ,$ the CPU clock frequency is denoted by $l _ { n } ^ { ( r ) }$ (cycles/s). Accordingly, the computation latency for local training is given by

$$
\rho _ { n } ^ { ( r ) } = \frac { N _ { \mathrm { f l o p } } | B _ { n } ^ { ( r ) } | } { l _ { n } ^ { ( r ) } } ,
$$

while the corresponding computation energy consumption is

$$
\xi _ { n } ^ { ( r ) } = \theta ( l _ { n } ^ { ( r ) } ) ^ { 2 } N _ { \mathrm { f l o p } } | B _ { n } ^ { ( r ) } | ,
$$

where $\theta$ denotes the effective switched capacitance coefficient determined by the processor chip architecture.

## C. Communication Model

We consider an OFDMA-based vehicular network operating over discrete time slots. For round $r ,$ the set of time slots is defined as $\mathcal { T } ^ { ( r ) } = \{ 1 , 2 , . . . , T \}$ , where T denotes the number of slots per round and τ represents the duration of each slot.

For downlink model distribution, the RSU broadcasts the global model to all vehicles using the entire bandwidth and sufficiently high transmission power. Therefore, following [7], [8], the downlink communication latency is ignored. For uplink gradient transmission, we consider a singleinput-multiple-output (SIMO) system, where each vehicle is equipped with a single antenna and the RSU is equipped with M antennas. The uplink bandwidth is equally divided into Z orthogonal resource blocks (RBs), each with bandwidth β. In each slot, the RSU allocates the RBs to vehicles, and the vehicles upload their gradients using the allocated RBs. We use $s _ { n , z } ( t )$ to denote the RB allocation indicator, where $s _ { n , z } ( t ) = 1$ if RB z is allocated to the vehicle n in slot t. Since one RB can only be allocated to one vehicle, $s _ { n , z } ( t )$ has the following constraints:

$$
\sum _ { n \in \mathcal { N } ^ { ( r ) } } s _ { n , z } ( t ) \leq 1 , \quad \forall z \in \mathcal { Z } , \forall t \in \mathcal { T } ^ { ( r ) } ,\tag{4}
$$

$$
\begin{array} { r } { s _ { n , z } ( t ) \in \{ 0 , 1 \} , \quad \forall n \in \mathcal { N } ^ { ( r ) } , \forall z \in \mathcal { Z } , \forall t \in \mathcal { T } ^ { ( r ) } . } \end{array}\tag{5}
$$

We denote the transmission power allocated to RB z by $p _ { n , z } ( t )$ , which is constrained by

$$
0 \leq \sum _ { z \in \mathcal { Z } } p _ { n , z } ( t ) \leq p _ { n } ^ { \operatorname* { m a x } } , \quad \forall n \in \mathcal { N } ^ { ( r ) } , \forall t \in \mathcal { T } ^ { ( r ) } .\tag{6}
$$

We denote the transmitted signal of vehicle n in slot t by $x _ { n } ( t )$ . The received signal at the RSU over RB z is given by

$$
y _ { n , z } ( t ) = s _ { n , z } ( t ) \sqrt { p _ { n , z } ( t ) } { \bf u } _ { n , z } ( t ) ^ { H } { \bf h } _ { n , z } ( t ) x _ { n } ( t ) + n _ { 0 } ,
$$

where ${ \pmb u } _ { n , z } ( t ) \in \mathbb { C } ^ { M \times 1 }$ is the receiver beamforming vector. $n _ { 0 } \sim \mathcal { C N } ( 0 , N _ { 0 } )$ denotes additive white Gaussian noise. The channel vector $\dot { h } _ { n , z } ( t ) \in \mathbb { C } ^ { M \times 1 }$ is modeled as

$$
\begin{array} { r } { { \bf h } _ { n , z } ( t ) = \sqrt { \psi _ { n } ( t ) } \varrho _ { n } ( t ) { \hat { \bf h } } _ { n , z } ( t ) , } \end{array}
$$

where $\sqrt { \psi _ { n } ( t ) }$ denotes the large-scale path loss, $\varrho _ { n } ( t )$ denotes the shadowing fading coefficient, and $\hat { h } _ { n , z } ( t )$ denotes the small-scale fading vector.

The RSU uses maximal-ratio combining receiver beamforming, i.e., ${ \bf { u } } _ { n , z } ( t ) = { \bf { h } } _ { n , z } ( t ) / \left\| { \bf { h } } _ { n , z } ( t ) \right\|$ . Then, the received signal-to-noise ratio (SNR) at the RSU from vehicle n over RB z in slot t is given by

$$
\Gamma _ { n , z } ( t ) = \frac { p _ { n , z } ( t ) \big | \big | h _ { n , z } ( t ) \big | \big | ^ { 2 } } { \beta N _ { 0 } } ,
$$

Based on the above model, the uplink transmission rate of vehicle n in slot t is given by

$$
A _ { n } ( t ) = \sum _ { z \in \mathcal { Z } } s _ { n , z } ( t ) \beta \log _ { 2 } ( 1 + \Gamma _ { n , z } ( t ) ) ,\tag{7}
$$

while the corresponding transmission energy consumption is

$$
e _ { n } ( t ) = \sum _ { z \in \mathcal { Z } } \tau s _ { n , z } ( t ) p _ { n , z } ( t ) .\tag{8}
$$

## D. Progressive Gradient Transmission

Based on the above communication model, vehicles progressively upload dominant gradient entries over the slots $t \in \mathcal { T } ^ { ( r ) }$ according to the available communication opportunities.

The communication overhead of each transmitted gradient entry consists of $\lceil \log _ { 2 } I \rceil$ bits for the index and b bits for the value, where I denotes the gradient dimension and $b = 3 2$ corresponds to single-precision floating-point representation. The number of new gradient entries that vehicle n can upload in slot t is integer-valued and capped by

$$
\kappa _ { n } ( t ) = \operatorname* { m i n } \left\{ \left\lfloor \frac { \tau A _ { n } ( t ) } { b + \lceil \log _ { 2 } I \rceil } \right\rfloor , I - \sum _ { t ^ { \prime } = 1 } ^ { t - 1 } \kappa _ { n } ( t ^ { \prime } ) \right\} ,\tag{9}
$$

where τ denotes the slot duration and $A _ { n } ( t )$ denotes the achievable uplink rate in slot t. Accordingly, the total number of uploaded gradient entries in round r is $\begin{array} { r l } { k _ { n } ^ { ( r ) } } & { { } = } \end{array}$ $\textstyle \sum _ { t \in T ^ { ( r ) } } \kappa _ { n } ( t )$ . For analytical tractability, the resource optimization in Sections V-B and V-C uses the continuous relaxation

$$
\bar { \kappa } _ { n } ( t ) = \frac { \tau A _ { n } ( t ) } { b + \lceil \log _ { 2 } I \rceil } ,\tag{10}
$$

To avoid allocating RBs or power to vehicles that have already uploaded all gradient entries, we impose

$$
\begin{array} { r l } & { s _ { n , z } ( t ) = 0 , \quad p _ { n , z } ( t ) = 0 , } \\ & { \forall n \in \mathcal { N } ^ { ( r ) } , \forall z \in \mathcal { Z } , \forall t \in \mathcal { T } ^ { ( r ) } , \displaystyle \sum _ { t ^ { \prime } = 1 } ^ { t - 1 } \kappa _ { n } ( t ^ { \prime } ) \geq I . } \end{array}\tag{11}
$$

After the RB and power allocation decisions are obtained, the actual number of transmitted entries is computed by the integer rounding and capping rule in (9). The continuous relaxation together with (11) is not exactly equivalent to (9). However, since the model dimension I is typically very large, the resulting approximation error is negligible.

The gradient transmission operator $\bar { \Psi } _ { n } ^ { ( \bar { r } ) } ( \pmb { g } _ { n } ^ { ( r ) } )$ is then formally defined as

$$
\left[ \Psi _ { n } ^ { ( r ) } ( { \pmb g } _ { n } ^ { ( r ) } ) \right] _ { i } = \left\{ \begin{array} { l l } { [ { \pmb g } _ { n } ^ { ( r ) } ] _ { i } , } & { i \in \mathcal { K } _ { n } ^ { ( r ) } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{12}
$$

where $\mathcal { K } _ { n } ^ { ( r ) }$ denotes the index set corresponding to the $k _ { n } ^ { ( r ) }$ largest-magnitude entries in $\mathbf { \Delta } _ { g _ { n } ^ { ( r ) } }$

## IV. PROBLEM FORMULATION

## A. Convergence Analysis

The goal of VFL is to minimize the global loss function (1). However, the impact of the number of transmitted gradient entries on the global loss is implicit. Therefore, we derive a convergence bound to characterize the relationship between gradient transmission and learning performance. We first introduce the definition of compressible gradients [34]–[38]. Definition 1 (Compressible Gradients). A vector $\pmb { g } \in \mathbb { R } ^ { I }$ is said to be compressible if the magnitudes of its entries, sorted in descending order, follow a power-law decay. Let $g _ { i }$ denote the i-th largest-magnitude entry of g. The vector g is compressible if there exist constants $C > 0$ and $\alpha > \frac { 1 } { 2 }$ such that

$$
| g _ { i } | \leq C i ^ { - \alpha } , \quad \forall i \in \{ 1 , 2 , \ldots , I \} ,\tag{13}
$$

where the compressibility exponent α characterizes the decay rate of the sorted entries, and a larger α indicates stronger compressibility. The scaling coefficient C characterizes the overall magnitude of the vector.

Prior works have shown that stochastic gradients in deep learning are typically compressible [34]–[36]. Accordingly, we make the following assumptions [8], [14]–[16], [39], [40].

Assumption 1: The stochastic gradient $\pmb { g } _ { n } ^ { ( r ) } \in \mathbb { R } ^ { I }$ generated by vehicle n in round r is compressible, i.e., there exist parameters $C _ { n } ^ { ( r ) } > 0$ and $\alpha _ { n } ^ { ( r ) } > \frac { 1 } { 2 }$ such that

$$
| g _ { n , i } ^ { ( r ) } | \leq C _ { n } ^ { ( r ) } i ^ { - \alpha _ { n } ^ { ( r ) } } , \quad \forall i \in \{ 1 , 2 , \ldots , I \} ,
$$

where $C _ { n } ^ { ( r ) }$ and $\alpha _ { n } ^ { ( r ) }$ can be estimated locally after each round by fitting the sorted gradient magnitudes to the power-law model.

Assumption 2: The variance of the stochastic gradient is bounded, i.e., $\begin{array} { r } { \mathbb { E } \left\| \pmb { g } _ { n } ^ { ( r ) } - \nabla F _ { n } \big ( \pmb { w } ^ { ( r - 1 ) } \big ) \right\| ^ { 2 } \leq \sigma ^ { 2 } } \end{array}$ , where the expectation is taken over the randomness of SGD.

Assumption 3: The data heterogeneity induced by vehicle sampling is bounded, i.e., $\underset { n \sim \mathcal { P } } { \mathbb { E } } \Vert \overset { \smile } { \nabla } F _ { n } ( \mathbf { \bar { w } } ) - \nabla F ( \pmb { \bar { w } } ) \Vert ^ { 2 } \le \delta ^ { 2 }$ for any model parameter w.

Assumption 4: The local loss function $F _ { n } ( \cdot )$ is L-smooth, i.e.,

$$
F _ { n } ( \pmb { w } ^ { \prime } ) \leq F _ { n } ( \pmb { w } ) + \langle \nabla F _ { n } ( \pmb { w } ) , \pmb { w } ^ { \prime } - \pmb { w } \rangle + \frac { L } { 2 } \left. \pmb { w } ^ { \prime } - \pmb { w } \right. ^ { 2 } .
$$

For a given gradient ${ \pmb g } _ { n } ^ { ( r ) } \in \mathbb { R } ^ { I }$ , the gradient approximation error is defined as $\begin{array} { r } { \left\| g _ { n } ^ { ( r ) } - \Psi _ { n } ^ { ( r ) } ( { \pmb g } _ { n } ^ { ( r ) } ) \right\| ^ { 2 } } \end{array}$

The following lemma characterizes the gradient approximation error under the compressible gradient assumption.

Lemma 1 (Gradient Approximation Error Bound). The gradient approximation error of $\mathbf { \Delta } _ { g _ { n } ^ { ( r ) } }$ is upper-bounded by

$$
\left\| g _ { n } ^ { ( r ) } - \Psi _ { n } ^ { ( r ) } ( { \pmb g } _ { n } ^ { ( r ) } ) \right\| ^ { 2 } \leq \frac { 2 ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } ( k _ { n } ^ { ( r ) } + 1 ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } { 2 \alpha _ { n } ^ { ( r ) } - 1 } .\tag{14}
$$

Proof: See Appendix A.

Lemma 1 characterizes the approximation error caused by transmitting only partial gradient entries as a function of the number of uploaded gradient entries $k _ { n } ^ { ( r ) }$ . Building on Lemma 1, we derive the convergence bound.

Theorem 1 (Convergence Bound). After R rounds of training, the expected squared gradient norm of the global loss function is upper-bounded by

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { R } \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } \left\| \nabla F ( { \boldsymbol w } ^ { ( r ) } ) \right\| ^ { 2 } } \\ & { \le \frac { 2 \left( F ( { \boldsymbol w } ^ { ( 0 ) } ) - \mathbb { E } [ F ( { \boldsymbol w } ^ { ( R ) } ) ] \right) } { \eta R } + \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \frac { 2 ( \sigma ^ { 2 } + \delta ^ { 2 } ) } { N ^ { ( r ) } } } \\ & { \displaystyle + \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \sum _ { n \in \mathcal { N } ^ { ( r ) } } \frac { 4 ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } ( k _ { n } ^ { ( r ) } + 1 ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } { N ^ { ( r ) } ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } . } \end{array}\tag{15}
$$

Proof: See Appendix B.

Remark 1. Theorem 1 shows that the convergence performance improves as the number of uploaded gradient entries $k _ { n } ^ { ( r ) }$ increases. However, the improvement exhibits diminishing marginal returns. Specifically, due to the power-law term $( k _ { n } ^ { ( r ) } + 1 ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) }$ , each additional transmitted entry yield progressively smaller reductions in the gradient approximation error, and therefore smaller improvements in convergence performance. Consequently, transmitting a small set of dominant gradient entries captures most of the benefit, while transmitting additional entries provides limited further improvement.

## B. Problem Formulation

Based on Theorem 1, we minimize the convergence upper bound in (15). This is equivalent to minimizing $\begin{array} { r } { \sum _ { n \in \mathcal { N } ^ { ( r ) } } \frac { \big ( C _ { n } ^ { ( r ) } \big ) ^ { 2 } \alpha _ { n } ^ { ( r ) } \big ( k _ { n } ^ { ( r ) } + 1 \big ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } { 2 \alpha _ { n } ^ { ( r ) } - 1 } } \end{array}$ in each round since other parameters are constant. The proposed scheme optimizes the communication resource allocation to indirectly control the progressive gradient transmission process. The optimization problem is formulated as

$$
P 0 : \operatorname* { m i n } _ { S ^ { ( r ) } , P ^ { ( r ) } } \sum _ { n \in \mathcal { N } ^ { ( r ) } } \frac { ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } } { 2 \alpha _ { n } ^ { ( r ) } - 1 } \left( \sum _ { t \in \mathcal { T } ^ { ( r ) } } \kappa _ { n } ( t ) + 1 \right) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) }\tag{16a}
$$

$$
\mathrm { s . t . } \quad \sum _ { t \in \mathcal { T } ^ { ( r ) } } e _ { n } ( t ) + \xi _ { n } ^ { ( r ) } \leq E _ { n } ^ { \mathrm { c o n s } } , \quad \forall n \in \mathcal { N } ^ { ( r ) } ,\tag{16b}
$$

$$
s _ { n , z } ( t ) = 0 , \quad p _ { n , z } ( t ) = 0 ,\tag{16c}
$$

$$
\forall n \in \mathcal { N } ^ { ( r ) } , \forall z \in \mathcal { Z } , \forall t \in \mathcal { T } ^ { ( r ) } , ( t - 1 ) \tau < \rho _ { n } ^ { ( r ) } ,
$$

$$
\mathrm { c o n s t r a i n t s ~ } ( 4 ) { - } ( 6 ) , ( 1 1 ) ,
$$

where $\begin{array} { r c l } { S ^ { ( r ) } } & { = } & { [ s ( 1 ) , . . . , s ( T ) ] } \end{array}$ denotes the RB allocation strategy, and $P ^ { ( r ) } ~ = ~ [ { \pmb p } ( 1 ) , . . . , { \pmb p } ( T ) ]$ denotes the power allocation strategy. Constraint (16b) ensures that the total energy consumption of each vehicle does not exceed the given energy budget. Constraint (16c) guarantees that uplink transmission starts only after local training is completed. Constraints (4)−(6) specify the feasible transmission and power allocation regions.

## V. PROGRESSIVE GRADIENT TRANSMISSION SCHEME

In this section, we present the proposed progressive gradient transmission scheme. We first employ a Lyapunov driftplus-penalty approach to transform the original stochastic optimization problem into an online mixed-integer nonlinear programming (MINLP) problem. Then, we derive key structural properties that facilitate the design of a joint RB and power allocation algorithm. Based on these results, the proposed algorithm performs utility-driven RB assignment and analytical power allocation for efficient online implementation. The overall optimization framework is illustrated in Fig. 2.

## A. Transformation of the Stochastic Optimization Problem

P0 is a stochastic optimization problem. The primary challenge in solving this problem lies in the uncertainty of future channel states and vehicle availability. High vehicle mobility results in rapidly varying communication conditions and unpredictable system evolution. Moreover, even with perfect future channel states and vehicle availability, solving P0 remains challenging due to the presence of integer and coupled decision variables.

![](images/a00002ab76ab2ff68857edd3d7debf20dca6ae24c1bcb68db744be2586ad7f0a.jpg)  
Fig. 2. Optimization framework of the proposed FedPGT scheme.

Lyapunov optimization is an efficient approach for stochastic optimization with unknown future channel states and vehicle availability and enables online decision-making based solely on current observations. However, standard Lyapunov frameworks are designed for objectives that are additively separable over time, such as the minimization of long-term average costs or the maximization of time-average utilities. In contrast, the objective of P0 is cumulatively coupled, depending on the sum of per-slot variables over the entire time horizon and preventing direct application of Lyapunov methods. To address this issue, we construct an upper-bounding surrogate reformulation that decouples the long-term dependence across time slots and converts the original cumulatively coupled objective into an additively separable surrogate objective, enabling Lyapunov-based per-slot optimization. The resulting surrogate optimization problem is given by

$$
P 1 \operatorname * { m i n } _ { S ^ { ( r ) } , P ^ { ( r ) } , \gamma ^ { ( r ) } } \sum _ { t \in { \cal T } ^ { ( r ) } } \sum _ { n \in { \cal N } ^ { ( r ) } } \frac { ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } \left( \gamma _ { n } ( t ) + \frac { 1 } { T } \right) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } { 2 \alpha _ { n } ^ { ( r ) } - 1 }\tag{17a}
$$

$$
{ \mathrm { s . t . } } 0 \leq \gamma _ { n } ( t ) \leq { \frac { I } { T } } , \forall n \in N ^ { ( r ) } ,\tag{17b}
$$

$$
\sum _ { t \in \mathcal { T } ^ { ( r ) } } \gamma _ { n } ( t ) = \sum _ { t \in \mathcal { T } ^ { ( r ) } } \kappa _ { n } ( t ) , \quad \forall n \in \mathcal { N } ^ { ( r ) } ,\tag{17c}
$$

$$
\mathrm { c o n s t r a i n t s ~ ( 4 ) } \mathrm { - } ( 6 ) , ( 1 1 ) , ( 1 6 \mathrm { b } ) , ( 1 6 \mathrm { c } ) .
$$

In P1, the surrogate variable $\gamma _ { n } ( t )$ represents the slot-wise contribution to the cumulative transmitted gradient entries. The objective of $P 1$ serves as an upper-bounding surrogate of the original cumulatively coupled objective in P0, while the equality constraint (17c) ensures long-term consistency between the surrogate variables and the actual transmitted gradient entries.

To handle the long-term constraints in the surrogate optimization problem, we introduce two types of virtual queues. First, for each vehicle $\textit { n } \in \mathcal { N } ^ { \left( r \right) }$ , the long-term energy constraint (16b) is enforced through the following virtual

energy queue:

$$
q _ { n } ( t + 1 ) = \operatorname* { m a x } \Bigg [ q _ { n } ( t ) + e _ { n } ( t ) - \frac { E _ { n } ^ { \mathrm { c o n s } } - \xi _ { n } ^ { ( r ) } } { T } , 0 \Bigg ] .\tag{18}
$$

The queue backlog $q _ { n } ( t )$ measures the accumulated energy deficit relative to the long-term energy budget. A larger queue backlog indicates that the vehicle has consumed excessive energy in previous slots, thereby imposing a stronger penalty on future transmission decisions.

Second, to enforce the equality constraint (17c), we introduce a PGT queue defined as

$$
\zeta _ { n } ( t + 1 ) = \zeta _ { n } ( t ) + \kappa _ { n } ( t ) - \gamma _ { n } ( t ) ,\tag{19}
$$

where the queue backlog $\zeta _ { n } ( t )$ measures the accumulated discrepancy between the actual transmitted gradient entries and the surrogate transmission variables. A larger queue backlog indicates that the number of transmitted entries exceeds the surrogate allocation, reducing the incentive to further transmit due to diminishing marginal convergence gains. Stabilizing this queue ensures long-term consistency between the surrogate reformulation and the original transmission process.

By applying the Lyapunov drift-plus-penalty framework to the surrogate optimization problem P1, we obtain the following per-slot optimization problem:

$$
P 2 : \operatorname* { m i n } _ { s ( t ) , p ( t ) , \gamma ( t ) } \sum _ { n \in \mathcal { N } ^ { ( r ) } } \frac { V ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } \left( \gamma _ { n } ( t ) + \frac { 1 } { T } \right) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } { 2 \alpha _ { n } ^ { ( r ) } - 1 }\tag{20a}
$$

$$
{ \mathrm { s . t . ~ \ c o n s t r a i n t s ~ ( 4 ) - ( 6 ) , ~ ( 1 1 ) , ~ ( 1 6 c ) , ~ ( 1 7 b ) . } }
$$

P2 is separable with respect to $\left( s ( t ) , p ( t ) \right)$  and $\gamma ( t )$ , since $\kappa _ { n } ( t )$ and $e _ { n } ( t )$ depend only on $\left( s ( t ) , p ( t ) \right)$ , while $\gamma _ { n } ( t )$ appears only in the terms $\begin{array} { r } { \frac { V ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } \big ( \gamma _ { n } ( t ) + \frac { 1 } { T } \big ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } { 2 \alpha _ { - } ^ { ( r ) } - 1 } \ - } \end{array}$ $\zeta _ { n } ( t ) \gamma _ { n } ( t )$ . Therefore, P2 can be decomposed into the following two subproblems. P3 determines the RB and power allocation decisions for the current slot:

$$
P 3 : \operatorname* { m i n } _ { s ( t ) , p ( t ) } \quad \sum _ { n \in \mathcal { N } ^ { ( r ) } } \left( q _ { n } ( t ) e _ { n } ( t ) + \zeta _ { n } ( t ) \kappa _ { n } ( t ) \right)\tag{21}
$$

s.t. constraints (4)−(6), (11) (16c).

The auxiliary variable $\gamma ( t )$ is optimized separately in $P 4 { : }$

$$
\begin{array} { r l } & { P 4 : \underset { \gamma ( t ) } { \operatorname* { m i n } } \sum _ { n \in \mathcal { N } ^ { ( r ) } } \frac { V ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } \left( \gamma _ { n } ( t ) + \frac { 1 } { T } \right) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } { 2 \alpha _ { n } ^ { ( r ) } - 1 } } \\ & { \quad \quad - \displaystyle \sum _ { n \in \mathcal { N } ^ { ( r ) } } \zeta _ { n } ( t ) \gamma _ { n } ( t ) } \end{array}\tag{22}
$$

s.t. constraints (17b).

Based on the above decomposition, we establish the following theorem to characterize the performance of the proposed Lyapunov-based online optimization framework. Superscript <sup>†</sup> denotes the solution obtained by the online algorithm (i.e., solving P3 and P4 in every slot), while <sup>∗</sup> denotes the optimal offline solution to P0.

Theorem 2. Suppose that all virtual queues are initialized to zero. Then, the performance gap between the solution obtained by the online algorithm and the optimal offline solution of P0 is bounded as

$$
\begin{array} { r l } & { \underset { n \in \mathcal { N } ^ { ( r ) } } { \sum } \frac { ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } } { 2 \alpha _ { n } ^ { ( r ) } - 1 } \bigg ( \underset { t \in \mathcal { T } ^ { ( r ) } } { \sum } \kappa _ { n } ^ { \dagger } ( t ) + 1 \bigg ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } \\ & { \leq \underset { n \in \mathcal { N } ^ { ( r ) } } { \sum } \frac { ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } ( k _ { n } ^ { ( r ) * } + 1 ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } { 2 \alpha _ { n } ^ { ( r ) } - 1 } } \\ & { + \underset { n \in \mathcal { N } ^ { ( r ) } } { \sum } \frac { T ^ { 2 - 2 \alpha _ { n } ^ { ( r ) } } \Phi } { V } + ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } T \sqrt { 2 \Phi } . } \end{array}\tag{23}
$$

The energy consumption of vehicle n is bounded by

$$
\xi _ { n } ^ { ( r ) } + \sum _ { t \in \mathcal { T } ^ { ( r ) } } e _ { n } ^ { \dagger } ( t ) \leq E _ { n } ^ { \mathrm { c o n s } } + T \sqrt { 2 \Phi } ,\tag{24}
$$

where $\begin{array} { r } { \epsilon _ { n } ( t ) \triangleq e _ { n } ( t ) - \frac { E _ { n } ^ { c o n s } - \xi _ { n } ^ { ( r ) } } { T } , \phi _ { n } \triangleq \operatorname* { m a x } _ { t } \{ | \epsilon _ { n } ( t ) | \} , \varphi _ { n } \triangleq } \end{array}$ $\mathrm { m a x } _ { t } \{ | \kappa _ { n } ( t ) - \gamma _ { n } ( t ) | \}$ and $\Phi \triangleq \operatorname* { m a x } _ { n } \{ ( \phi _ { n } ) ^ { 2 } + ( \varphi _ { n } ) ^ { 2 } \}$ Proof: See Appendix C. □

Theorem 2 characterizes the performance of the Lyapunovbased online framework under exact per-slot minimization of P3 and P4. Specifically, if the decomposed per-slot problems are solved optimally in each slot, the resulting online policy achieves a bounded gap with respect to the offline optimum of P0, while guaranteeing bounded long-term energy consumption for each vehicle.

Based on the above decomposition, we next solve the two subproblems P3 and P4, respectively.

P4 is a convex optimization problem, since its objective function is convex with respect to $\gamma ( t )$ and the feasible set defined by (17b) is convex. Therefore, the optimal solution can be directly obtained from the KKT conditions as

$$
\begin{array} { r } { \gamma _ { n } ^ { \dagger } ( t ) = \left\{ \begin{array} { l l } { \frac { I } { T } , } & { \zeta _ { n } ( t ) \geq 0 , } \\ { \left[ \left( \frac { V ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } } { - \zeta _ { n } ( t ) } \right) ^ { \frac { 1 } { 2 \alpha _ { n } ^ { ( r ) } } } - \frac { 1 } { T } \right] _ { 0 } ^ { \frac { I } { T } } , } & { \zeta _ { n } ( t ) < 0 , } \end{array} \right. } \end{array}\tag{25}
$$

where $[ a ] _ { 0 } ^ { \frac { I } { T } }$ is defined as min(max(a, 0), <sup>I</sup> ).

In contrast, P3 is a MINLP problem due to the binary RB allocation variables and the nonlinear coupling introduced by the power control constraints. As a result, solving P3 is more challenging and will be addressed in the following subsections.

## B. RB Allocation Criterion

First, we relax the binary RB allocation variables $s ( t )$ to continuous variables in [0, 1] and consider the following relaxed version of P3:

$$
\begin{array} { r l r } {  { P 3 : \operatorname* { m i n } _ { s ( t ) , p ( t ) } \sum _ { n \in \mathcal { N } ( t ) } \sum _ { z \in \mathcal { Z } } \frac { \zeta _ { n } ( t ) \tau \beta s _ { n , z } ( t ) \log _ { 2 } ( 1 + \Gamma _ { n , z } ( t ) ) } { b + \lceil \log _ { 2 } T \rceil } } } \\ & { } & { + \sum _ { n \in \mathcal { N } ( t ) } \sum _ { z \in \mathcal { Z } } q _ { n } ( t ) \tau s _ { n , z } ( t ) p _ { n , z } ( t ) \qquad ( 2 6 } \end{array}\tag{a}
$$

$$
\begin{array} { r } { \mathrm { s . t . ~ } 0 \leq s _ { n , z } ( t ) \leq 1 , \quad \forall n \in \mathcal { N } ( t ) , \forall z \in \mathcal { Z } , } \\ { \mathrm { c o n s t r a i n t s ~ } ( 4 ) , ( 6 ) , \qquad } \end{array}\tag{26b}
$$

where $\begin{array} { r } { \mathcal { N } ( t ) = \{ n \in \mathcal { N } ^ { ( r ) } : ( t - 1 ) \tau \geq \rho _ { n } ^ { ( r ) } , \sum _ { t ^ { \prime } = 1 } ^ { t - 1 } \kappa _ { n } ( t ^ { \prime } ) < } \end{array}$ $I \}$ denotes the set of vehicles that have completed local training and are eligible for transmission at slot t. Owing to constraint (16c), vehicles outside $\mathcal { N } ( t )$ are excluded from RB allocation. The relaxed problem P3 remains non-convex since the objective function is neither convex nor concave with respect to $s ( t )$ and ${ \pmb p } ( t )$ . Thus, obtaining a globally optimum solution is challenging. Nevertheless, we derive the following proposition characterizing the necessary optimality conditions based on the KKT conditions.

Proposition 1. A necessary condition for $s _ { n , z } ( t )$ being positive at the optimal solution is

$$
n = \arg \operatorname* { m i n } _ { n \in \mathcal { N } ( t ) } \Upsilon _ { n , z } ( t )\tag{27}
$$

where the cost function $\Upsilon _ { n , z } ( t )$ is defined as

$$
\Upsilon _ { n , z } ( t ) = \frac { \zeta _ { n } ( t ) \tau \beta \log _ { 2 } ( 1 + \Gamma _ { n , z } ( t ) ) } { b + \lceil \log _ { 2 } I \rceil } + \tau q _ { n } ( t ) p _ { n , z } ( t ) .\tag{28}
$$

Proof: See Appendix D.

This proposition characterizes a necessary optimality condition for RB allocation. Although it does not directly yield the globally optimal solution, it provides the basis for the proposed RB allocation criterion.

Although the binary RB allocation variables are relaxed to continuous variables in [0, 1], the following proposition shows that, for a given power allocation, the relaxed RB allocation subproblem admits an integral optimal solution. Therefore, the exclusive RB allocation property of OFDMA is preserved without loss of optimality.

Proposition 2. For a given power allocation, the relaxed RB allocation subproblem admits an integral optimal solution. Specifically, there exists an optimal solution in which each RB is exclusively assigned to one vehicle, i.e.,

$$
s _ { n ^ { * } , z } ( t ) = 1 , \quad s _ { n , z } ( t ) = 0 , \quad \forall n \neq n ^ { * } .
$$

Proof: Consider a fixed RB z. Let $\mathcal { N } _ { z } ( t )$ denote the set of vehicles that achieve the smallest $\Upsilon _ { n , z } ( t )$ on RB z, i.e.,

$$
\mathcal { N } _ { z } ( t ) = \left\{ n | \arg \operatorname* { m i n } _ { n \in \mathcal { N } ( t ) } \Upsilon _ { n , z } ( t ) \right\} .
$$

From the KKT conditions, we know that for the optimal solution, $s _ { n , z } ( t ) ~ > ~ 0$ implies user $\textit { n } \in \mathcal { N } _ { z } ( t )$ , and $s _ { n , z } ( t ) = 0$ for all $n \not \in \mathcal { N } _ { z } ( t )$ . When RB z is allocated to a certain vehicle, constraint (4) becomes $\begin{array} { r l } { \sum _ { n \in \mathcal { N } ( t ) } s _ { n , z } ( t ) = } & { { } } \end{array}$ $\begin{array} { r } { \sum _ { n \in \mathcal { N } _ { z } ( t ) } s _ { n , z } ( t ) = 1 } \end{array}$ , Moreover, from the definition of

Algorithm 1 Power allocation algorithm   
Sort $z \in \mathcal { Z } _ { n } ( t )$ in descending order of $\left\| h _ { n , z } ( t ) \right\| ^ { 2 } .$   
for $j = 1$ to $| \mathcal { Z } _ { n } ( t ) |$ do   
Select the first j RBs as ${ \mathcal { Z } } _ { n } ^ { * } ( t ) ;$   
Compute power allocation $p _ { n , z } ^ { \prime } ( t )$ for all $z \in \ z _ { n } ^ { * } ( t )$   
according to Proposition 3;   
if any $p _ { n , z } ^ { \prime } ( t ) \leq 0$ then   
break; {Stop: invalid RB found, do not update}   
else   
Update the current power allocation $p _ { n , z } ( t ) = p _ { n , z } ^ { \prime } ( t ) ;$   
end if   
end for   
$\mathcal { N } _ { z } ( t )$ , all vehicles $n \in \mathcal { N } _ { z } ( t )$ have the same value of $\Upsilon _ { n , z } ( t )$   
which is the smallest possible among all vehicles. As a result,   
the contribution of the objective (26a) from $\operatorname { R B } z \ i$ s   
$\sum _ { n \in \mathcal { N } _ { z } ( t ) } s _ { n , z } ( t ) \Upsilon _ { n , z } ( t ) = \Upsilon _ { n ^ { * } , z } ( t ) \sum _ { n \in \mathcal { N } _ { z } ( t ) } s _ { n , z } ( t ) = \Upsilon _ { n ^ { * } , z } ( t ) ,$   
where $n ^ { * } \in \mathcal { N } _ { z } ( t )$   
Therefore, regardless of how the weights $s _ { n , z } ( t )$ are dis  
tributed among vehicles in $\mathcal { N } _ { z } ( t )$ , the objective contribution   
remains unchanged as long as their sum equals 1.

Furthermore, $\Upsilon _ { n ^ { * } , z } ( t ) ~ \leq ~ 0$ holds because the transmit power $p _ { n ^ { * } , z } ( t )$ associated with the RB can be set to zero without violating any constraints. Therefore, assigning the RB to a vehicle is no worse than leaving it unassigned.

Hence, for the relaxed RB allocation subproblem with fixed power allocation, there exists an integral solution assigning RB z exclusively to one vehicle. This completes the proof. □

## C. Power Allocation

Then, the power allocation problem is considered. Given the   
RB allocation decision, the following proposition characterizes   
the optimal transmit power allocation strategy.   
Proposition 3. Given the RB allocation decision, the optimal   
power allocation for vehicle n is characterized as follows.   
$I f \zeta _ { n } ( t ) \geq 0 ,$ the optimal transmit power is   
p<sub>n,z</sub>(t) = 0, ∀z ∈ Z<sub>n</sub>(t). (29)   
$I f \ \zeta _ { n } ( t ) \ < \ 0$ and $q _ { n } ( t ) > 0 ,$ , the optimal transmit power   
over the positive-power RB set ${ \mathcal { Z } } _ { n } ^ { * } ( t )$ is   
$p _ { n , z } ( t ) = \operatorname* { m i n } \left( \frac { p _ { n } ^ { \mathrm { m a x } } + \sum _ { z ^ { \prime } \in \mathcal { Z } _ { n } ^ { * } ( t ) } \frac { \beta N _ { 0 } } { \left\| h _ { n , z ^ { \prime } } ( t ) \right\| ^ { 2 } } } { | \mathcal { Z } _ { n } ^ { * } ( t ) | } , \frac { - \zeta _ { n } ( t ) \beta } { q _ { n } ( t ) \ln 2 } \right)$   
$- \frac { \beta N _ { 0 } } { \left\| h _ { n , z } ( t ) \right\| ^ { 2 } } , \quad \forall z \in \mathcal { Z } _ { n } ^ { * } ( t ) .$ (30)   
$I f \zeta _ { n } ( t ) < 0$ and $q _ { n } ( t ) = 0 ,$ , the optimal transmit power is   
p<sup>max</sup><sub>n</sub> + P<sub>z′∈Z∗n(t)</sub> <sup>βN0</sup><sub>∥hn,z′</sub> <sub>(t)∥2</sub> βN<sub>0</sub>   
p<sub>n,z</sub>(t) =   
|Z<sup>∗</sup><sub>n</sub>(t)| ∥h<sub>n,z</sub>(t)∥<sup>2 (31)</sup>   
$\forall z \in { \mathcal { Z } } _ { n } ^ { * } ( t ) .$   
Proof: See Appendix E. □

```latex
Algorithm 2 Joint RB and power allocation algorithm
1: Initialize all $s _ { n , z } ( t ) = 0 , p _ { n , z } ( t ) = 0 ,$ set of unallocated
RBs $\mathcal { Z } ^ { \prime } ( t ) = \mathcal { Z }$ , and the set of RBs already allocated to
vehicle n $\mathcal { Z } _ { n } ^ { \dagger } ( t ) = \varnothing ;$
2: while $\mathcal { Z } ^ { \prime } \neq \emptyset$ do
3: for $n \in \mathcal { N } ( t )$ do
4: Let $\mathcal { Z } _ { n } ( i ) = \mathcal { Z } _ { n } ^ { \dag } ( t ) \cup \mathcal { Z } ^ { \prime } ( t ) :$
5: Obtain the power allocation $p _ { n , z } ( t )$ over $\mathcal { Z } _ { n } ( t )$ based
on Algorithm 1;
6: Compute $\Upsilon _ { n , z } ( t )$ according to (28);
7: end for
8: Find $\begin{array} { r } { ( n , z ) = \arg \operatorname* { m i n } _ { n \in \mathcal { N } ( t ) , z \in \mathcal { Z } ^ { \prime } } \Upsilon _ { n , z } ( t ) ; } \end{array}$
9: Allocate RB z to user n: set $s _ { n , z } ( t ) = 1 ;$
10: Update ${ \mathcal { Z } } ^ { \prime } \gets { \mathcal { Z } } ^ { \prime } \setminus \{ z \}$ and $\mathcal { Z } _ { n } ^ { \dagger } ( t )  \mathcal { Z } _ { n } ^ { \dagger } ( t ) \cup \{ z \} ;$
11: end while
12: for $n \in \mathcal { N } ( t )$ do
13: Perform Algorithm 1 over the assigned RBs
$\{ z | s _ { n , z } ( t ) = 1 \}$ to finalize $p _ { n , z } ( t ) ;$
14: end for
```

Proposition 3 gives a closed-form expression for power allocation. However, this requires prior knowledge of the subset of RBs to which vehicle n assigns positive power, i.e., $\mathcal { Z } _ { n } ^ { * } ( t )$ . To address this, we propose Algorithm 1, which incrementally searches for the subset ${ \mathcal { Z } } _ { n } ^ { * } ( t )$ . Specifically, it first sorts all candidate RBs in descending order of channel gain. Then, it sequentially adds RBs to the candidate set and computes the corresponding power allocation using (30). The process continues until a non-positive power value is encountered, at which point the search terminates. This ensures that only RBs receiving positive power are included in ${ \mathcal { Z } } _ { n } ^ { * } ( t )$ Remark 2. Propositions 1 and 3 characterize the structural properties of the optimal RB and power allocation strategies. The PGT queue $\zeta _ { n } ( t )$ regulates the transmission urgency according to the accumulated transmitted gradient entries. When $\zeta _ { n } ( t ) \geq 0 .$ , the vehicle has already transmitted a relatively large number of gradient entries, and the diminishing marginal convergence gain reduces the incentive for further transmissions. In this case, the transmit power becomes zero, and the corresponding RBs are released for other vehicles with more urgent transmission demands. When $\zeta _ { n } ( t ) < 0$ , the system allocates more communication resources to increase the transmitted gradient entries. The energy queue $q _ { n } ( t )$ controls the long-term energy consumption, where a larger queue backlog imposes a higher penalty on future power allocation and encourages more energy-efficient transmissions.

## D. Joint RB and Power Allocation Algorithm

Although Propositions 1–3 provide analytical characterizations for the RB and power allocation strategies, RB allocation and power allocation remain inherently coupled, since the RB utility depends on the power allocation, while the power allocation itself depends on the allocated RB set. We develop the iterative joint RB and power allocation algorithm shown in Algorithm 2 for solving P3.

The proposed algorithm iteratively allocates RBs according to their marginal utility contributions under adaptive power allocation. Initially, all RBs are unallocated. In each iteration, for every vehicle n, the RSU temporarily augments its currently allocated RB set $\mathcal { Z } _ { n } ^ { \dag } ( t )$ with the remaining unallocated RBs ${ \mathcal { Z } } ^ { \prime } ( t )$ . Based on this temporary RB set, the corresponding power allocation is recomputed using Algorithm 1. The resulting $\Upsilon _ { n , z } ( t )$ value is then evaluated for all candidate RBvehicle pairs.

Algorithm 3 The Overall Procedure of FedPGT   
Initialization: Set $\pmb q ( 1 ) = \mathbf { 0 }$ and $\zeta ( 1 ) = \mathbf { 0 } ;$   
for $t \in \mathcal { T } ^ { ( r ) }$ do   
Vehicles completing local training estimate $C _ { n } ^ { ( r ) }$ and $\alpha _ { n } ^ { ( r ) }$   
via power-law fitting;   
Construct the active transmission set $\mathcal { N } ( t ) ;$   
Observe current CSI $h ( t )$ and queue states $\mathbf { \boldsymbol { q } } ( t ) , \dot { \mathsf { \boldsymbol { \varsigma } } } ( t ) ;$   
Solve P4 to obtain $\gamma ^ { \ast } ( t ) ;$   
Solve $P 3$ to obtain $s ^ { * } ( t )$ and ${ \mathbf { } } p ^ { * } ( t )$ using Algorithm 2;   
Vehicles in $\mathcal { N } ( t )$ transmit gradient entries;   
Update virtual queues q(t + 1) and $\zeta ( t + 1 )$   
end for

The RB-vehicle pair $( n , z )$ yielding the smallest $\Upsilon _ { n , z } ( t )$ is selected, and RB z is assigned to vehicle n. The allocated RB is then removed from the unallocated RB set, and the procedure repeats until all RBs are assigned. Finally, after the RB allocation process is completed, each vehicle recomputes the power allocation over its finalized RB set using Algorithm 1.

## E. The Overall Procedure of FedPGT

The workflow of the proposed PGT algorithm is outlined in Algorithm 3. At the start of each round, the RSU broadcasts the global model and vehicles conduct local model updates. The compressibility parameters $C _ { n } ^ { ( r ) }$ and $\alpha _ { n } ^ { ( r ) }$ are then estimated locally by fitting the sorted gradient magnitudes to the power-law model in the log-log domain. In each slot, the RSU observes the current channel states and vehicle availability and solves the decomposed per-slot problems P3 and $P 4$ Vehicles progressively transmit gradient entries according to the obtained RB and power allocation decisions, and the virtual queues are updated based on the realized transmissions. This procedure continues over all slots within the round. By solving the per-slot problems online, PGT adapts to timevarying channels without requiring future system information and achieves a bounded performance gap.

The per-slot complexity is dominated by solving P3 and P4. P4 admits a closed-form solution, and its complexity scales linearly with the number of vehicles, i.e., $\mathcal { O } ( N ^ { ( r ) } )$ For P3, the RB assignment iterates over all unallocated RBs. In each iteration, Algorithm 1 is executed for each vehicle to recompute the corresponding power allocation, yielding complexity $\mathcal { O } ( N ^ { ( r ) } Z )$ per iteration. Since each RB is assigned once, the overall complexity of the RB allocation is $\mathcal { O } ( N ^ { ( r ) } Z ^ { 2 } )$ . Note that Algorithm 1 involves sorting RBs according to their channel gains. However, this sorting operation is only performed once for each vehicle at the beginning of each slot, and the resulting RB ordering is reused throughout subsequent executions. Therefore, the sorting step only introduces preprocessing complexity $\mathcal { O } ( N ^ { ( r ) } Z \log _ { 2 } Z )$ which is dominated by $\mathcal { O } ( N ^ { \left( r \right) } Z ^ { 2 } )$ .

TABLE I Simulation Parameters.
<table><tr><td rowspan=1 colspan=1>Simulation Parameters</td><td rowspan=1 colspan=1>Values</td></tr><tr><td rowspan=1 colspan=1>System Bandwidth</td><td rowspan=1 colspan=1>20 MHz</td></tr><tr><td rowspan=1 colspan=1>Maximum Transmission Power</td><td rowspan=1 colspan=1>0.2 W</td></tr><tr><td rowspan=1 colspan=1>Carrier Frequency</td><td rowspan=1 colspan=1>5.9 GHz</td></tr><tr><td rowspan=1 colspan=1>Vehicle Blockage Loss</td><td rowspan=1 colspan=1>max{0,N(5, 4)} dB</td></tr><tr><td rowspan=1 colspan=1>Shadowing Fading Std. Dev.</td><td rowspan=1 colspan=1>3 dB (LOS, NLOSv), 4 dB (NLOS)</td></tr><tr><td rowspan=1 colspan=1>Noise Power Spectrum Density</td><td rowspan=1 colspan=1>-174 dBm/Hz</td></tr><tr><td rowspan=1 colspan=1>Energy Consumption Coefficient</td><td rowspan=1 colspan=1>10−28[41], [42]</td></tr><tr><td rowspan=1 colspan=1>Energy Constraints</td><td rowspan=1 colspan=1>Randomly selected from 0.05 J to 0.1 J</td></tr><tr><td rowspan=1 colspan=1>Length of Time Slot</td><td rowspan=1 colspan=1>10 ms</td></tr><tr><td rowspan=1 colspan=1>Average Number of Vehicles</td><td rowspan=1 colspan=1>15</td></tr><tr><td rowspan=1 colspan=1>Number of RBs</td><td rowspan=1 colspan=1>50</td></tr></table>

Therefore, the per-slot complexity is $\mathcal { O } ( N ^ { ( r ) } Z ^ { 2 } )$ . Over T slots, the total complexity becomes $\mathcal { O } ( T N ^ { ( r ) } Z ^ { 2 } )$

## VI. EXPERIMENTAL SETUP

In this section, we evaluate the proposed FedPGT through simulations under vehicular network settings. We first introduce the simulation setup. Then, the performance of FedPGT under different vehicular network conditions is evaluated on both image classification and trajectory prediction tasks using the CIFAR-10 and Argoverse datasets.

## A. Simulation Setups

An urban grid road network is generated using SUMO [43], where an RSU is deployed at the center. Vehicle routing follows a Manhattan mobility pattern. At each intersection, vehicles move straight with probability 0.5 and turns left or right with probability 0.25 each.

Vehicle dynamics follow the Intelligent Driver Model (IDM). The acceleration of vehicle n is given by

$$
a _ { n } = a ^ { \mathrm { m a x } } \left[ 1 - \left( \frac { v _ { n } } { v ^ { \mathrm { m a x } } } \right) ^ { 4 } - \left( \frac { d _ { n } ^ { \prime } } { d _ { n } ^ { * } } \right) ^ { 2 } \right] ,
$$

where $v _ { n }$ is the instantaneous speed, $v ^ { \mathrm { m a x } }$ is the maximum allowable speed, and $d _ { n } ^ { \prime }$ is the headway distance.

The desired headway distance is given by

$$
d _ { n } ^ { * } = d ^ { \mathrm { s a f e } } + v _ { n } t ^ { \mathrm { d s t } } + \frac { v _ { n } \Delta v _ { n } } { 2 \sqrt { a ^ { \mathrm { m a x } } a ^ { \mathrm { b r a k e } } } } ,
$$

where $d ^ { \mathrm { s a f e } }$ is the safety distance, $t ^ { \mathrm { d s t } }$ is the desired time headway, $\Delta v _ { n }$ is the relative speed, and $a ^ { \mathrm { b r a k e } }$ denotes the comfortable braking deceleration.

The wireless channel model follows the 3GPP V2X specifications in TR 37.885 [44]. Both line-of-sight (LOS) and nonline-of-sight (NLOS) propagation conditions are considered.

The urban LOS path loss is modeled as $P L _ { \mathrm { L O S } } = 3 8 . 7 7 +$ $1 6 . 7 \log _ { 1 0 } ( d ) + 1 8 . 2 \log _ { 1 0 } ( f _ { c } )$ , and the NLOS path loss is $P L _ { \mathrm { N L O S } } = 3 6 . 8 5 + 3 0 \log _ { 1 0 } ( d ) + 1 8 . 9 \log _ { 1 0 } ( f _ { c } )$ , where d is the link distance and $f _ { c }$ is the carrier frequency. The shadowing coefficient follows a log-normal distribution, while the smallscale fading vector follows Rayleigh fading.

The key simulation parameters are summarized in Table I. Unless otherwise specified, the default parameter settings are adopted when evaluating the impact of individual variables.

![](images/3c82191a99cc8984c86de58efd375aa1b1ba8d3c0876d3fe88a3ae6098c59bb3.jpg)  
Fig. 3. The road network generated by SUMO.

![](images/e3a3341088b0acacb85cde7cba946093eff01b086aaf13b9affba1e8cc576aff.jpg)  
Fig. 4. Sorted Gradient Magnitudes in the log-log Domain.

![](images/9693d8ec5ac975780079f907a72e30ebbf7688888fe22a136947b627b9c82bdf.jpg)  
Fig. 5. Evolution of Estimated Compressibility Exponent $\alpha _ { n } ^ { ( r ) }$

## B. Model and Datasets

We evaluate the proposed FedPGT on both image classification and trajectory prediction tasks.

1) CIFAR-10: We first evaluate FedPGT on the CIFAR-10 dataset [45], which contains 50,000 training images and 10,000 test images from 10 classes. A non-i.i.d. data distribution is considered, where samples are partitioned according to labels and each vehicle holds data from two distinct classes.

A convolutional neural network is trained for image classification. The network consists of six convolutional layers, followed by ReLU activations, with alternating max-pooling and normalization operations. The final convolutional layer is connected to a fully connected layer and a softmax classifier. The batch size is randomly selected from {16, 32, 48}, and the learning rate is set to 0.1.

2) Argoverse: We further evaluate FedPGT on the Argoverse trajectory prediction dataset [46], which contains over 300,000 sequences collected in urban driving scenarios. Each sequence is sampled at 10 Hz, and the task is to predict future positions over a 3-second horizon. The dataset is divided into training, validation, and test sets with 205,942, 39,472, and 78,143 sequences, respectively. The data are uniformly partitioned into 40 subsets.

For trajectory prediction, we adopt the LaneGCN model [47], which integrates trajectory encoding and map information. It consists of an ActorNet for extracting trajectory features, a MapNet for modeling lane graph structures, and a FusionNet for combining these features to generate predictions. The batch size is randomly selected from {16, 32, 48}, and the learning rate is set to 0.1. We use the average displacement error (ADE) as the evaluation metric.

## C. Baseline Schemes

To evaluate the performance of the proposed FedPGT, we compare it with the following benchmark schemes.

1) V2V-Enhanced FL (V2VFL) [8]: This scheme adopts dynamic device scheduling with V2V-assisted communications. Transmission and scheduling decisions are updated at each time slot according to channel variations and vehicle mobility. Unlike FedPGT, only fully uploaded updates can contribute to the model aggregation.

2) Vehicular Edge FL (VEFL) [7]: VEFL considers vehicle mobility and fast channel variations. Resource allocation decisions are dynamically updated according to instantaneous channel states.

3) Joint Gradient Sparsification and Device Scheduling (JGSD) [10]: JGSD jointly considers device scheduling and gradient sparsification, where the sparsification degree is fixed at the beginning of each communication round.

4) Static Scheduling Algorithm (SA) [13]: This scheme performs device scheduling and resource allocation based on initial channel states and vehicle locations, without adapting to mobility or applying sparsification.

## D. Estimation of Gradient Compressibility Parameters

In this subsection, we illustrate how the compressibility parameters are estimated from the local gradients and investigate their evolution during federated training. According to Definition 1 and Assumption 1, $| g _ { n , i } ^ { ( r ) } | \leq C _ { n } ^ { ( \check { r } ) } i ^ { - \alpha _ { n } ^ { ( r ) } }$ , which implies that the sorted gradient magnitudes are upper-bounded by a linear function in the log-log domain: $\log _ { 1 0 } | g _ { n , i } ^ { ( r ) } | \ \leq$ $\log _ { 1 0 } C _ { n } ^ { ( r ) } - \alpha _ { n } ^ { ( r ) } \log _ { 1 0 } i$ . This motivates the estimation of the compressibility parameters through linear regression in the log-log domain.

Fig. 4 illustrates the sorted gradient magnitudes. The dominant gradient entries exhibit an approximately linear relationship in the log-log domain, indicating that a power-law model provides a good approximation of the gradient decay behavior. In contrast, the tail entries decrease more rapidly than the fitted power-law curve, primarily due to the large number of near-zero gradient components. Since Assumption 1 only requires the sorted gradient magnitudes to be upperbounded by a power-law function, such a faster decay remains consistent with the assumption. To estimate the compressibility parameters, we identify the dominant region that exhibits the strongest linear relationship in the log-log domain. A slidingwindow least-squares regression is performed over the sorted gradient magnitudes, and the interval achieving the highest coefficient of determination value is selected for parameter estimation. The resulting fitting parameters are then used as the scaling coefficient $C _ { n } ^ { \daleth _ { r } }$ and compressibility exponent $\alpha _ { n } ^ { ( r ) }$ in the proposed framework.

Fig. 5 shows the evolution of the estimated compressibility exponent $\alpha _ { n } ^ { ( r ) }$ during training. As training progresses, the estimated exponent gradually increases, indicating that the gradients become increasingly compressible near convergence. This is because the gradient magnitude becomes progressively concentrated on a few dominant entries as more model parameters approach stationary points. Consequently, fewer dominant gradient entries are required to accurately approximate the full gradient, further demonstrating the effectiveness of progressive gradient transmission.

![](images/5e84404046f5eec723ebdb1ef90f7e7ddde97d99384c2abdeac4dd1f147d82fd.jpg)  
Fig. 6. Final test accuracy of different methods under different vehicle speeds.

![](images/24bb354aabf28242cbe37ad71fc135d5caa126b6aa1d8026bbf8c0e13bea4bad.jpg)  
Fig. 7. Convergence performance of different methods when v<sup>max</sup> = 5 m/s.

![](images/bd1037be9acf4099b02fda3100d9d8d569a4e2a7b21f35432f422b66c5955523.jpg)  
Fig. 8. Convergence performance of different methods when $\mathbf { \bar { \rho } } _ { v } \mathbf { m a x } \mathbf { \bar { \rho } } = 2 5$ m/s.

![](images/16c68de3173ca83b4120ef8abf2bc218dad9580813aeaffa9c48b4060a51ec44.jpg)  
Fig. 9. Final test accuracy of different methods under different average number of vehicles.

![](images/864c3a0072117820ef6dd084937453e9df30c637b33a4294b031f9e268529e75.jpg)  
Fig. 10. Convergence performance when the average number of vehicles is 5.

![](images/e40dc7857450aa8cddc96cf047658b055974116378ff491db3c68629e979a8ed.jpg)  
Fig. 11. Convergence performance when the average number of vehicles is 25.

## E. Performance under Different Vehicle Speeds

In this subsection, we evaluate performance under varying maximum vehicle speeds $v ^ { \mathrm { m a x } }$ . Fig. 6 shows the final accuracy, while Figs. 7 and 8 present the convergence performance at $v ^ { \mathrm { m a x } } = 5$ m/s and $v ^ { \mathrm { m a x } } = 2 5$ m/s, respectively.

At low vehicle speeds, all schemes achieve comparable performance, as vehicles remain within the RSU coverage long enough to complete transmissions. Compared with the static case, FedPGT and V2VFL even achieve slight gains due to channel diversity introduced by mobility. When vehicles move at high speeds, performance degrades across all schemes. Higher mobility shortens the sojourn time and increases the probability of interrupted transmissions, reducing the number of successfully aggregated updates.

Among all schemes, FedPGT exhibits the strongest robustness against mobility, as it jointly incorporates dynamic resource allocation and progressive gradient transmission. Dynamic resource allocation adapts to time-varying channel conditions, while progressive gradient transmission allows partially transmitted updates to contribute to model aggregation under interrupted transmissions.

V2VFL and VEFL perform well at low speeds but degrade at high speeds. These schemes require full model uploads, and thus transmission interruptions caused by mobility lead to invalid updates. JGSD also degrades with increasing speed, but remains more robust than SA because partially transmitted model updates can still contribute to training. In contrast, SA suffers the most severe degradation, as it neither adapts to channel variations nor accounts for vehicle mobility.

## F. Performance under Different Numbers of Vehicles

We evaluate the performance of different schemes under varying average numbers of vehicles within the RSU coverage, as shown in Fig. 9, while Figs. 10 and 11 present the convergence performance with average numbers of vehicles of 5 and 25, respectively. As the number of vehicles increases, FedPGT and JGSD achieve steady performance improvement, while V2VFL, VEFL, and SA exhibit diminishing gains and gradually saturate. When the number of vehicles is small, the total bandwidth is sufficient to support most transmissions, and a large fraction of vehicles can successfully upload their updates. As a result, increasing the number of participants improves the convergence performance across all schemes.

As the number of vehicles increases, the total communication demand gradually exceeds the available resources. In this case, schemes requiring complete update transmissions (V2VFL, VEFL, and SA) cannot efficiently exploit the increasing number of vehicles, since only a limited number of updates can be successfully aggregated within each communication round. In contrast, FedPGT and JGSD continue to achieve stable performance gains as the number of vehicles grows, since partially transmitted gradient entries can still contribute to model aggregation under limited communication resources. This also verifies Theorem 1: collecting dominant gradient entries from more vehicles is more beneficial than collecting more complete updates from fewer vehicles.

## G. Performance under Different Total Bandwidth

We evaluate performance under varying total bandwidth. Fig. 12 shows the final accuracy, while Figs. 13 and 14 present the convergence performance under 10 MHz and 30 MHz bandwidth settings, respectively. When the bandwidth is sufficient (30 MHz), all schemes achieve comparable performance, since most vehicles can successfully complete their update transmissions. As the total bandwidth decreases from 30 MHz to 10 MHz, the overall performance degrades due to limited communication resources.

![](images/b3b5dddab1ec91af7dfc825d11199ff33051650827ac4a477244cd2fa0399c58.jpg)  
Fig. 12. Final test accuracy of different methods under different total bandwidth.

![](images/a27d9eba6035921e76006afa91e99bf911f4e579acd4514ccf260073ff707e61.jpg)  
Fig. 13. Convergence performance when the total bandwidth is 10 MHz.

![](images/b5e9d7d04e157ec675846a00ca35f11462bae7296249e455a5e4f0c3bd481bfc.jpg)  
Fig. 14. Convergence performance when the total bandwidth is 30 MHz.

![](images/023eddee6c708ec68dbfb4bc85dc09e225b5d5bbeb491edf74499e773f521624.jpg)  
Fig. 15. ADE for the trajectory prediction task when the total bandwidth is 20 MHz and $v ^ { \mathrm { m a x } } = 5 ~ \mathrm { m / s } .$

![](images/37a980c3ab6616959fdce5eed0b2d974a5770b76c62e00aab5ac701281c88d0e.jpg)  
Fig. 16. ADE for the trajectory prediction task when the total bandwidth is 20 MHz and $v ^ { \mathrm { m a x } } = 2 5 ~ \mathrm { m / s }$

![](images/211c52830bec7b71681591dca6dccbb3b23d145558db0f511273108d4fdbbd02.jpg)  
Fig. 17. ADE for the trajectory prediction task when the total bandwidth is 10 MHz and $v ^ { \mathrm { m a x } } = 5 ~ \mathrm { m / s } .$

In the low-bandwidth regime, schemes requiring complete update transmissions (V2VFL, VEFL, and SA) experience significant performance degradation, since limited bandwidth restricts the number of updates that can be successfully aggregated within each communication round. In contrast, FedPGT remains the most robust under limited bandwidth, as progressive gradient transmission enables dominant gradient entries from more vehicles to still contribute to model aggregation.

## H. Evaluation on Argoverse Trajectory Prediction Dataset

We further evaluate the performance of different schemes on the Argoverse trajectory prediction dataset. The results are shown in Figs. 15–17, where the average displacement error (ADE) is used as the evaluation metric. When the vehicle speed is low $( v ^ { \mathrm { m a x } } = 5 ~ \mathrm { m / s ) }$ and the bandwidth is sufficient (20 MHz), all schemes achieve comparable performance, with FedPGT achieving the lowest ADE. As the system operates under more challenging conditions, such as limited bandwidth (10 MHz) or high mobility $( v ^ { \mathrm { m a x } } ~ = ~ 2 5 ~ \mathrm { m / s ) }$ , the performance gap between FedPGT and the baselines becomes more significant. In these cases, FedPGT exhibits a clear advantage, demonstrating its robustness to both communication constraints and mobility. These results further verify that the advantages of FedPGT extend beyond image classification tasks to more complex spatio-temporal prediction problems.

## VII. CONCLUSION

In this paper, we proposed FedPGT, a progressive gradient transmission scheme for vehicular federated learning over time-varying wireless channels. We first established a convergence bound that reveals the diminishing-return effect of transmitted gradient entries, which motivates the optimization of progressive transmission according to marginal learning utility. To address the cumulatively coupled transmission objective, we developed an additively separable surrogate reformulation and introduced the PGT and energy queues to track transmission consistency and energy-budget deviation. This enables a Lyapunov-based tradeoff between learning utility and energy cost, allowing FedPGT to jointly adapt RB assignment and power allocation without future system information. Simulation results on CIFAR-10 and Argoverse demonstrated that FedPGT achieves more robust learning performance than existing vehicular federated learning schemes under high mobility and limited communication resources.

Future work will investigate the integration of semanticaware transmission and more advanced gradient compression techniques into the proposed FedPGT framework to further reduce communication overhead while preserving learning performance. We will also extend the framework to more practical vehicular federated learning scenarios, such as asynchronous model aggregation under dynamic connectivity.

## REFERENCES

[1] Y. Sun, W. Shi, X. Huang, S. Zhou, and Z. Niu, “Edge learning with timeliness constraints: Challenges and solutions,” IEEE Commun. Mag., vol. 58, no. 12, pp. 27–33, Dec. 2020.

[2] M. Chen, D. Gund ¨ uz, K. Huang, W. Saad, M. Bennis, A. V. Feljan, and¨ H. V. Poor, “Distributed learning in wireless networks: Recent progress and future challenges,” IEEE J. Sel. Areas Commun., vol. 39, no. 12, pp. 3579–3605, Oct. 2021.

[3] W. Xu, Z. Yang, D. W. K. Ng, M. Levorato, Y. C. Eldar, and M. Debbah, “Edge learning for b5g networks with distributed signal processing: Semantic communication, edge computing, and wireless sensing,” IEEE J. Sel. Top. Signal Process., vol. 17, no. 1, pp. 9–39, Jan. 2023.

[4] N. Jia, Z. Qu, B. Ye, Y. Wang, S. Hu, and S. Guo, “A comprehensive survey on communication-efficient federated learning in mobile edge environments,” IEEE Commun. Surv. Tutor., vol. 27, no. 6, pp. 3710– 3741, Dec. 2025.

[5] J. Posner, L. Tseng, M. Aloqaily, and Y. Jararweh, “Federated learning in vehicular networks: Opportunities and solutions,” IEEE Netw., vol. 35, no. 2, pp. 152–159, Mar. 2021.

[6] J. Yan, T. Chen, B. Xie, Y. Sun, S. Zhou, and Z. Niu, “Hierarchical federated learning: Architecture, challenges, and its implementation in vehicular networks,” ZTE Commun., vol. 21, no. 1, pp. 38–45, Mar. 2023.

[7] M. F. Pervej, R. Jin, and H. Dai, “Resource constrained vehicular edge federated learning with highly mobile connected vehicles,” IEEE J. Sel. Areas Commun., vol. 41, no. 6, pp. 1825–1844, May 2023.

[8] J. Yan, T. Chen, Y. Sun, Z. Nan, S. Zhou, and Z. Niu, “Dynamic scheduling for vehicle-to-vehicle communications enhanced federated learning,” IEEE Trans. Wireless Commun., vol. 24, no. 11, pp. 9373– 9390, Nov. 2025.

[9] L. Li, D. Shi, R. Hou, H. Li, M. Pan, and Z. Han, “To talk or to work: Flexible communication compression for energy efficient federated learning over heterogeneous mobile edge devices,” in Proc. IEEE INFOCOM, Vancouver, BC, Canada, May 2021, pp. 1–10.

[10] X. Lin, Y. Liu, F. Chen, X. Ge, and Y. Huang, “Joint gradient sparsification and device scheduling for federated learning,” IEEE Trans. Green Commun. Netw., vol. 7, no. 3, pp. 1407–1419, Aug. 2023.

[11] S. Hu, L. Jiang, and B. He, “Practical hybrid gradient compression for federated learning systems,” in Proc. Int. Joint Conf. Artif. Intell. (IJCAI), Macau SAR, China, Aug. 2024, pp. 458–465.

[12] M. Chen, Z. Yang, W. Saad, C. Yin, H. V. Poor, and S. Cui, “A joint learning and communications framework for federated learning over wireless networks,” IEEE Trans. Wireless Commun., vol. 20, no. 1, pp. 269–283, Oct. 2020.

[13] W. Shi, S. Zhou, Z. Niu, M. Jiang, and L. Geng, “Joint device scheduling and resource allocation for latency constrained wireless federated learning,” IEEE Trans. Wireless Commun., vol. 20, no. 1, pp. 453–467, Sept. 2020.

[14] M. M. Amiri and D. Gund ¨ uz, “Federated learning over wireless fading¨ channels,” IEEE Trans. Wireless Commun., vol. 19, no. 5, pp. 3546– 3557, May 2020.

[15] Y. Sun, S. Zhou, Z. Niu, and D. Gund ¨ uz, “Dynamic scheduling for¨ over-the-air federated edge learning with energy constraints,” IEEE J. Sel. Areas Commun., vol. 40, no. 1, pp. 227–242, Nov. 2021.

[16] G. Zhu, Y. Wang, and K. Huang, “Broadband analog aggregation for low-latency federated edge learning,” IEEE Trans. Wireless Commun., vol. 19, no. 1, pp. 491–506, Jan. 2020.

[17] T. Chen, J. Yan, Y. Sun, S. Zhou, D. Gund¨ uz, and Z. Niu, “Mobility¨ accelerates learning: Convergence analysis on hierarchical federated learning in vehicular networks,” IEEE Trans. Veh. Technol., vol. 74, no. 1, pp. 1657–1673, Jan. 2025.

[18] H. Xiao, J. Zhao, Q. Pei, J. Feng, L. Liu, and W. Shi, “Vehicle selection and resource optimization for federated learning in vehicular edge computing,” IEEE Trans. Intell. Transp. Syst., vol. 23, no. 8, pp. 11 073–11 087, Aug. 2022.

[19] X. Zhang, Z. Chang, T. Hu, W. Chen, X. Zhang, and G. Min, “Vehicle selection and resource allocation for federated learning-assisted vehicular network,” IEEE Trans. Mobile Comput., vol. 23, no. 5, pp. 3817–3829, May 2024.

[20] X. Cai, P. Zhao, S. Liu, Y. Fu, C. Li, and F. R. Yu, “Enhancing federated learning in connected and autonomous vehicles through cost optimization and advanced model selection,” IEEE Trans. Intell. Transp. Syst., vol. 26, no. 4, pp. 5276–5289, Apr. 2025.

[21] H. Tu, W. Wu, L. Chen, L. Li, X. Chen, and X. Shen, “FL in motion: Accelerating FL via mobility-aware vehicle selection and sparse training,” IEEE Trans. Mobile Comput., Jan. 2026, early Access.

[22] F. Fathi, M. Montazeri, B. N. Araabi, H. Du, D. Niyato, and H. Kebriaei, “Data-driven incentive mechanisms for federated learning in vehicular networks,” IEEE Trans. Veh. Technol., vol. 74, no. 7, pp. 10 175–10 186, Jul. 2025.

[23] A. Shan, C. Wu, Y. Lin, L. Zhong, J. Li, Y. Ji, and J. Chen, “Reinforcement learning-based incentive scheme for federated learning in vehicular networks,” IEEE Trans. Cogn. Commun. Netw., vol. 12, pp. 5148–5160, Dec. 2025.

[24] D. Chen, T. Deng, H. Huang, J. Jia, M. Dong, D. Yuan, and K. Li, “Mobility-aware multi-task decentralized federated learning for vehicular networks: Modeling, analysis, and optimization,” IEEE Trans. Mobile Comput., vol. 25, no. 2, pp. 2594–2610, Feb. 2026.

[25] Z. Jin, C. Yang, Y. Ye, L. Zhang, J. Shen, and J. Su, “Mobility-aware semi-asynchronous federated learning for vehicular networks,” IEEE Trans. Veh. Technol., vol. 75, no. 2, pp. 2001–2012, Feb. 2026.

[26] C. Li, Z. Zhao, H. Zhang, and D. Yuan, “Vehfsl: Hybrid federated split learning for resource-constrained vehicular networks,” IEEE Trans. Wireless Commun., vol. 25, pp. 8677–8691, Dec. 2025.

[27] L. Yu and Z. Chang, “Semantic communication-enhanced u-shaped split federated learning with adaptive compression for vehicular networks,” IEEE Trans. Wireless Commun., vol. 25, pp. 11 609–11 623, Feb. 2026.

[28] S. U. Stich, J.-B. Cordonnier, and M. Jaggi, “Sparsified SGD with memory,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), Montreal,´ Canada, Dec. 2018, pp. 4452–4463.

[29] D. Basu, D. Data, C. Karakus, and S. N. Diggavi, “Qsparse-local-sgd: Distributed SGD with quantization, sparsification, and local computations,” IEEE J. Sel. Areas Inf. Theory, vol. 1, no. 1, pp. 217–226, May 2020.

[30] P. Han, S. Wang, and K.-K. Leung, “Adaptive gradient sparsification for efficient federated learning: An online learning approach,” in Proc. IEEE Int. Conf. Distrib. Comput. Syst. (ICDCS), Singapore, Nov. 2020, pp. 300–310.

[31] K. Wei, J. Li, C. Ma, M. Ding, F. Shu, H. Zhao, W. Chen, and H. Zhu, “Gradient sparsification for efficient wireless federated learning with differential privacy,” Sci. China Inf. Sci., vol. 67, no. 4, p. 142303, Mar. 2024.

[32] X. Zhang, X. Zhou, M. Lin, and J. Sun, “ShuffleNet: An extremely efficient convolutional neural network for mobile devices,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Salt Lake City, UT, USA, Jun. 2018, pp. 6848–6856.

[33] Q. Zeng, Y. Du, K. Huang, and K. K. Leung, “Energy-efficient resource management for federated edge learning with CPU-GPU heterogeneous computing,” IEEE Trans. Wireless Commun., vol. 20, no. 12, pp. 7947– 7962, Dec. 2021.

[34] A. M. Abdelmoniem, A. Elzanaty, M. Alouini, and M. Canini, “An efficient statistical-based gradient compression technique for distributed training systems,” in Proc. 4th Conf. Mach. Learn. Syst. (MLSys), Virtual Conference, Apr. 2021, pp. 297–322.

[35] A. Elzanaty, A. Giorgetti, and M. Chiani, “Limits on sparse data acquisition: RIC analysis of finite gaussian matrices,” IEEE Trans. Inf. Theory, vol. 65, no. 3, pp. 1578–1588, Mar. 2019.

[36] A. Elzanaty, A. Giorgetti, and M. Chiani, “Lossy compression of noisy sparse sources based on syndrome encoding,” IEEE Trans. Commun., vol. 67, no. 10, pp. 7019–7031, Oct. 2019.

[37] S. Mallat, A Wavelet Tour of Signal Processing: The Sparse Way. Academic Press, Apr. 2009.

[38] R. Baraniuk, M. A. Davenport, M. F. Duarte, and C. Hegde, “An introduction to compressive sensing,” Online, Feb. 2011, available: https://legacy.cnx.org/content/col11133/1.5/.

[39] J. Wangni, J. Wang, J. Liu, and T. Zhang, “Gradient sparsification for communication-efficient distributed optimization,” in Proc. Adv. Neural Inf. Process. Syst. (NIPS), Montreal, Canada, Dec. 2018, pp. 1299–1309.´

[40] Y. Du, S. Yang, and K. Huang, “High-dimensional stochastic gradient quantization for communication-efficient edge learning,” IEEE Trans. Signal Process., vol. 68, pp. 2128–2142, Mar. 2020.

[41] Y. Mao, J. Zhang, and K. B. Letaief, “Dynamic computation offloading for mobile-edge computing with energy harvesting devices,” IEEE J. Sel. Areas Commun., vol. 34, no. 12, pp. 3590–3605, Dec. 2016.

[42] Z. Yang, M. Chen, W. Saad, C. S. Hong, and M. Shikh-Bahaei, “Energy efficient federated learning over wireless communication networks,” IEEE Trans. Wireless Commun., vol. 20, no. 3, pp. 1935–1949, Nov. 2021.

[43] P. A. Lopez, M. Behrisch, L. Bieker-Walz, J. Erdmann, Y.-P. Flotter ¨ od,¨ R. Hilbrich, L. Lucken, J. Rummel, P. Wagner, and E. Wießner,¨ “Microscopic Traffic Simulation using SUMO,” in Proc. IEEE Int. Conf. Intell. Transp. Syst. (ITSC), Maui, HI, USA, Nov. 2018, pp. 2575–2582.

[44] M. Harounabadi, D. M. Soleymani, S. Bhadauria, M. Leyh, and E. Roth-Mandutz, “V2X in 3GPP standardization: NR sidelink in release-16 and beyond,” IEEE Commun. Standards Mag., vol. 5, no. 1, pp. 12–21, Mar. 2021.

[45] A. Krizhevsky, V. Nair, and G. Hinton, “Learning multiple layers of features from tiny images,” Tech. Rep., Apr. 2009.

[46] M.-F. Chang, J. Lambert, P. Sangkloy, J. Singh, S. Bak, A. Hartnett, D. Wang, P. Carr, S. Lucey, D. Ramanan, and J. Hays, “Argoverse: 3d tracking and forecasting with rich maps,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Long Beach, CA, USA, Jun. 2019, pp. 8740–8749.

[47] M. Liang, B. Yang, R. Hu, Y. Chen, R. Liao, S. Feng, and R. Urtasun, “Learning lane graph representations for motion forecasting,” in Proc. European Conf. Comput. Vis. (ECCV), Glasgow, UK, Aug. 2020, pp. 541–556.

![](images/4e92f1b5904d8270101463f217e959a454712ab482ff5315fabeca8c8b68c3b7.jpg)

Jintao Yan (S’23) received the B.S. degree from the University of Electronic Science and Technology of China (UESTC), Chengdu, China, in 2022. He is currently working toward the Ph.D. degree with the Network Integration for Ubiquitous Linkage and Broadband Laboratory (Niulab), Department of Electronic Engineering, Tsinghua University. His research interests include edge intelligence, vehicular networks, federated learning and optimization theory.

![](images/a5adbf1fb124344a811030377c5b69aa610068b734542b08a6fa44a589ec738e.jpg)

Tan Chen (S’24) received the B.S. degree from the Department of Electronic Engineering, Tsinghua University, Beijing, China, in 2021. He is currently working toward the Ph.D. degree with the Network Integration for Ubiquitous Linkage and Broadband Laboratory (Niulab), Department of Electronic Engineering, Tsinghua University. His research interests include federated learning, vehicular networks and edge intelligence.

![](images/1ed78e7bae27df81903ce80d623e62f93eb5ba820b8ef15b065ee3a3687631b0.jpg)

Yuxuan Sun (S’18-M’20) received the Ph.D. degree in electronic engineering from Tsinghua University, Beijing, China, in 2020. From 2018 to 2019, she was a Visiting Student at the Department of Electrical and Electronic Engineering, Imperial College London, UK. From 2020 to 2022, she was a Post-Doctoral Researcher at the Department of Electronic Engineering, Tsinghua University, and a Visiting Researcher at Imperial College London. Currently, she is an Associate Professor at the School of Electronic and Information Engineering, Beijing

Jiaotong University, Beijing, China. She is the receipt of the Young Elite Scientists Sponsorship Program by CAST. She served as the Assistant to the Editor-in-Chief of IEEE TRANSACTIONS ON GREEN COMMUNICATIONS AND NETWORKING from 2020 to 2022. She has been the Secretary of IEEE ComSoC Emerging Technologies Standing Committee since 2022. Her research interests lie in the areas of edge computing, edge intelligence, and task-oriented communications.

![](images/231dd8011e768f5f8cd822230859977956efe8cabe55a06aa4c1705b433b3935.jpg)

Sheng Zhou (S’06-M’12-SM’24) received the B.E. and Ph.D. degrees in electronic engineering from Tsinghua University, Beijing, China, in 2005 and 2011, respectively. In 2010, he was a Visiting Student with the Wireless System Lab, Department of Electrical Engineering, Stanford University, Stanford, CA, USA. From 2014 to 2015, he was a Visiting Researcher with the Central Research Lab, Hitachi Ltd., Japan. He is currently an Associate Professor with the Department of Electronic Engineering, Tsinghua University. His research interests include cross-layer design for multiple antenna systems, mobile edge computing, vehicular networks, and green wireless communications. He received the IEEE ComSoc Asia–Pacific Board Outstanding Young Researcher Award in 2017, and IEEE ComSoc Wireless Communications Technical Committee Outstanding Young Researcher Award in 2020.

![](images/13a77d247c84b8c90a6a3fe688177329b4e237352ae5ffbf3b5b0c19d23e6265.jpg)

Zhisheng Niu (M’98-SM’99-F’12) graduated from Beijing Jiaotong University, China, in 1985, and got his M.E. and D.E. degrees from Toyohashi University of Technology, Japan, in 1989 and 1992, respectively. From 1992 to 1994, he worked for Fujitsu Laboratories Ltd., Japan, and in 1994 joined with Tsinghua University, Beijing, China, where he is now a Professor at the Department of Electronic Engineering. His major research interests include queueing theory, traffic engineering, mobile Internet, radio resource management of wireless networks, and green communication and networks.

Dr. Niu has been serving IEEE Communications Society since 2000 as Chair of Beijing Chapter (2000-2008), Director of Asia-Pacific Board (2008- 2009), Director for Conference Publications (2010-2011), Chair of Emerging Technologies Committee (2014-2015), Director for Online Contents (2018- 2019) and currently the Editor-in-Chief of IEEE TRANSACTIONS ON GREEN COMMUNICATIONS AND NETWORKING. He received the Best Paper Award of Asia-Pacific Board in 2013, Distinguished Technical Achievement Recognition Award of Green Communications and Computing Technical Committee in 2018, and Harold Sobol Award for Exemplary Service to Meetings & Conferences in 2019, all from the IEEE Communications Society. He was selected as a distinguished lecturer of IEEE Communications Society (2012- 2015) as well as IEEE Vehicular Technologies Society (2014-2018). He is a fellow of both IEEE and IEICE.

APPENDIX A PROOF OF LEMMA 1

Ranking the elements of $\pmb { g } _ { n } ^ { ( r ) }$ in descending order, where $| g _ { n , 1 } ^ { ( r ) } | \geq | \breve { g } _ { n , 2 } ^ { ( r ) } | \geq \ldots \geq | g _ { n , I } ^ { ( r ) } |$ , we have

$$
\begin{array} { r l } & { \| g _ { n } ^ { ( r ) } - \psi _ { n } ^ { ( r ) } ( g _ { n } ^ { ( r ) } ) \| ^ { 2 } } \\ & { = \displaystyle \sum _ { i = k \ell ^ { \prime } n ^ { \prime } + 1 } ^ { \ell } ( g _ { n } ^ { ( r ) } ) ^ { 2 } \leq \sum _ { i = k \ell ^ { \prime } n ^ { \prime } + 1 } ^ { \ell } ( C _ { n } ^ { ( r ) } ) ^ { 2 } i ^ { - \ell \alpha ( s ) } } \\ & { \quad \ : = i \displaystyle \sum _ { i = k \ell ^ { \prime } n ^ { \prime } + 1 } ^ { \ell } \sum _ { k = \ell } ^ { \ell } \sum _ { i = k \ell } ^ { \ell } ( C _ { n } ^ { ( r ) } ) ^ { 2 } i ^ { - 2 \ell \alpha ( s ) } } \\ & { \leq ( C _ { n } ^ { ( r ) } ) ^ { 2 } \left( k _ { n } ^ { ( r ) } + 1 \right) ^ { - 2 \alpha ( s ) } + \sum _ { i = k \ell ^ { \prime } n ^ { \prime } + 2 } ^ { \ell } ( C _ { n } ^ { ( r ) } ) ^ { 2 } i ^ { - 2 \ell \alpha ( s ) } } \\ & { \leq ( C _ { n } ^ { ( r ) } ) ^ { 2 } \left( k _ { n } ^ { ( r ) } + 1 \right) ^ { - 2 \alpha ( s ) } + ( C _ { n } ^ { ( r ) } ) ^ { 2 } \displaystyle \int _ { k _ { n } ^ { ( r ) } + 1 } ^ { \infty } x ^ { - \alpha ( s ) } d s ^ { \prime } } \\ & { = ( C _ { n } ^ { ( r ) } ) ^ { 2 } \left( ( k _ { n } ^ { ( r ) } + 1 ) ^ { - 2 \alpha ( s ) } + \frac { ( k _ { n } ^ { ( r ) } + 1 ) ^ { - ( 2 \alpha ( s ) } - 1 ) } { 2 \alpha ( s ) } \right) \cdot } \\ &  \leq \frac { 2 ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } ( k _ { n } ^ { ( r ) } + 1 ) ^ { - ( 2 \alpha ( s ) } - 1 ) }  2  \end{array}
$$

This completes the proof.

## APPENDIX B PROOF OF THEOREM 1

According to Assumption 4 and definition (1), the global loss function is also L-smooth. There is:

$$
\begin{array} { r l r } {  { F ( { \pmb w } ^ { ( r ) } ) - F ( { \pmb w } ^ { ( r - 1 ) } ) } } \\ & { } & { \le \Big \langle \nabla F ( { \pmb w } ^ { ( r - 1 ) } ) , { \pmb w } ^ { ( r ) } - { \pmb w } ^ { ( r - 1 ) } \Big \rangle + \frac { L } { 2 } \| { \pmb w } ^ { ( r ) } - { \pmb w } ^ { ( r - 1 ) } \| ^ { 2 } } \end{array}
$$

For the term $\big \langle \nabla F ( { \pmb w } ^ { ( r - 1 ) } ) , { \pmb w } _ { r + 1 } - { \pmb w } ^ { ( r ) } \big \rangle$ , we have

$$
\begin{array} { r l } & { ( ( \mathcal { X } ^ { 0 , 0 , 4 , n - 1 } , \mathcal { X } ^ { 0 , 1 } ) , \mathcal { X } ^ { 1 } ) = ( \begin{array} { l } { - \lambda _ { 1 } ^ { 2 } } \\ { \gamma _ { 1 } ^ { 2 } } \\ { \lambda _ { 2 } ^ { 2 } } \end{array} ) } \\ & { = - \sqrt { 5 6 0 0 0 ^ { - 1 } \gamma _ { 1 } ^ { 2 } } \frac { \lambda _ { 2 } ^ { 2 } } { 4 \lambda _ { 3 } ^ { 2 } } \frac { \lambda _ { 3 } ^ { 2 } } { 4 \lambda _ { 4 } ^ { 2 } } \frac { \zeta _ { 2 } ^ { 2 } \lambda _ { 3 } ^ { 2 } \lambda _ { 4 } ^ { 2 } } { 4 \lambda _ { 5 } ^ { 2 } } \frac { \zeta _ { 2 } ^ { 2 } \lambda _ { 4 } ^ { 2 } \lambda _ { 5 } ^ { 2 } } { 4 \lambda _ { 5 } ^ { 2 } } } \\ & { = \frac { \lambda _ { 1 } ^ { 2 } } { 2 } [ \Gamma \xi _ { 3 } \Gamma \Theta ^ { - 1 } ] ^ { 2 } + \frac { \lambda _ { 2 } ^ { 2 } } { 2 } [ \sum _ { s = 0 } ^ { \infty } \frac { \zeta _ { s } ^ { 2 } \lambda _ { 2 } ^ { 2 } } { 4 \lambda _ { 5 } ^ { 2 } } ] ^ { 2 s - 1 } } \\ & { \quad \times \frac { \lambda _ { 2 } ^ { 2 } } { 4 \lambda _ { 5 } ^ { 2 } } [ \Gamma \xi _ { 3 } ^ { - 1 } \xi _ { 4 } ^ { - 1 } ] ^ { 2 s - 1 } \frac { \zeta _ { 2 } ^ { 2 } \lambda _ { 4 } ^ { 2 } } { 4 \lambda _ { 5 } ^ { 2 } } \frac { \zeta _ { 2 } ^ { 2 } \lambda _ { 4 } ^ { 2 } } { 4 \lambda _ { 5 } ^ { 2 } } ] } \\ &  = - \frac { \lambda _ { 1 } ^ { 2 } } { 2 } [ \Gamma \Theta \Lambda \Theta ^ { - 1 } ] ^ { 2 } - \frac  \lambda _  1  \end{array}
$$

For the term $\left\| \pmb { w } ^ { ( r ) } - \pmb { w } ^ { ( r - 1 ) } \right\| ^ { 2 }$ , we have

$$
\frac { L } { 2 } \left\| \pmb { w } ^ { ( r ) } - \pmb { w } ^ { ( r - 1 ) } \right\| ^ { 2 } = \frac { \eta ^ { 2 } L } { 2 } \left\| \frac { \sum _ { n \in \mathcal { N } ^ { ( r ) } } \Psi _ { n } ^ { ( r ) } ( \pmb { g } _ { n } ^ { ( r ) } ) } { N ^ { ( r ) } } \right\| ^ { 2 } .\tag{34}
$$

Setting $\eta \le \frac { 1 } { L }$ , taking the expectation over stochastic data sampling on (32) and plugging (33) and (34), we have

$$
\begin{array} { r l } & { \displaystyle \frac { \eta } { 2 } \left\| \nabla F ( { \boldsymbol w } ^ { ( r - 1 ) } ) \right\| ^ { 2 } \le \mathbb { E } [ F ( { \boldsymbol w } ^ { ( r - 1 ) } ) ] - \mathbb { E } [ F ( { \boldsymbol w } ^ { ( r ) } ) ] } \\ & { \displaystyle + \frac { \eta ( \sigma ^ { 2 } + \delta ^ { 2 } ) } { N ^ { ( r ) } } + \sum _ { n \in \mathcal { N } ^ { ( r ) } } \frac { 2 \eta ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } ( k _ { n } ^ { ( r ) } + 1 ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } { N ^ { ( r ) } ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } . } \end{array}
$$

Taking a telescopic sum from $r = 1$ to $r = R ,$ we get

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { R } \sum _ { r = 0 } ^ { R - 1 } \mathbb { E } \left\| \nabla F ( { \boldsymbol w } ^ { ( r ) } ) \right\| ^ { 2 } } \\ & { \le \frac { 2 \left( F ( { \boldsymbol w } ^ { ( 0 ) } ) - \mathbb { E } [ F ( { \boldsymbol w } ^ { ( R ) } ) ] \right) } { \eta R } + \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \frac { 2 ( \sigma ^ { 2 } + \delta ^ { 2 } ) } { N ^ { ( r ) } } } \\ & { \displaystyle + \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \sum _ { n \in \mathcal { N } ^ { ( r ) } } \frac { 4 ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } ( k _ { n } ^ { ( r ) } + 1 ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } { N ^ { ( r ) } ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } . } \end{array}
$$

This completes the proof.

## APPENDIX C PROOF OF THEOREM 2

We define a quadratic Lyapunov function as

$$
L _ { n } ( t ) \triangleq \frac { 1 } { 2 } \sum _ { n \in \mathcal { N } ^ { ( r ) } } ( q _ { n } ( t ) ) ^ { 2 } + \frac { 1 } { 2 } \sum _ { n \in \mathcal { N } ^ { ( r ) } } ( \zeta _ { n } ( t ) ) ^ { 2 } ,
$$

and the Lyapunov drift of a single round as

$$
\begin{array} { r l r } & { \Delta _ { n } ( t ) \triangleq L _ { n } ( t + 1 ) - L _ { n } ( t ) } & \\ & { = \frac { 1 } { 2 } \left( q _ { n } ( t + 1 ) ) ^ { 2 } - ( q _ { n } ( t ) ) ^ { 2 } + ( \zeta _ { n } ( t + 1 ) ) ^ { 2 } - ( \zeta _ { n } ( t ) ) ^ { 2 } \right) } & \\ & { \leq \frac { 1 } { 2 } \left( ( q _ { n } ( t ) + \epsilon _ { n } ( t ) ) ^ { 2 } - ( q _ { n } ( t ) ) ^ { 2 } \right) } & { ( 3 ) } & \\ & { + \frac { 1 } { 2 } \left( ( \zeta _ { n } ( t ) + \kappa _ { n } ( t ) - \gamma _ { n } ( t ) ) ^ { 2 } - ( \zeta _ { n } ( t ) ) ^ { 2 } \right) } & \\ & { \leq \Phi + q _ { n } ( t ) \epsilon _ { n } ( t ) + \zeta _ { n } ( t ) \left( \kappa _ { n } ( t ) - \gamma _ { n } ( t ) \right) , } & \end{array}\tag{5}
$$

where $\begin{array} { r } { \epsilon _ { n } ( t ) \triangleq e _ { n } ( t ) - \frac { E _ { n } ^ { \mathrm { { c o n s } } } - \xi _ { n } ^ { ( r ) } } { T } , \phi _ { n } \triangleq \operatorname* { m a x } _ { t } \{ | \epsilon _ { n } ( t ) | \} , \varphi _ { n } \triangleq } \end{array}$ max<sub>t</sub> $\{ | \kappa _ { n } ( t ) - \gamma _ { n } ( t ) | \}$ and Φ <sup>≜</sup> ma $\mathfrak { c } _ { n } \{ ( \phi _ { n } ) ^ { 2 } + ( \varphi _ { n } ) ^ { 2 } \}$ . We define the overall drift as $\Delta _ { n } \triangleq \Delta _ { n } ( T + 1 ) - \Delta _ { n } ( 1 )$ , which is bounded by

$$
\Delta _ { n } \leq T \Phi + \sum _ { t \in \mathcal { T } ^ { ( r ) } } \left( q _ { n } ( t ) \epsilon _ { n } ( t ) + \zeta _ { n } ( t ) \left( \kappa _ { n } ( t ) - \gamma _ { n } ( t ) \right) \right) .
$$

Based on the definition of $q _ { n } ( t )$ , we have $q _ { n } ( t + 1 ) - q _ { n } ( t ) \leq$ $\phi _ { n } , \forall n \in \mathcal { N } , r \in \mathcal { R }$ . Therefore, for any $\epsilon _ { n } ( t )$ , we have

$$
q _ { n } ( t ) \epsilon _ { n } ( t ) \leq ( q _ { n } ( t ) - q _ { n } ( 1 ) ) \phi _ { n } \leq ( t - 1 ) ( \phi _ { n } ) ^ { 2 } .
$$

Similarly, there is

$$
\begin{array} { r } { \zeta _ { n } ( t ) ( \kappa _ { n } ( t ) - \gamma _ { n } ( t ) ) \leq ( t - 1 ) ( \varphi _ { n } ) ^ { 2 } . } \end{array}
$$

Therefore, we have

$$
\Delta _ { n } \leq \sum _ { t \in \mathcal { T } ^ { ( r ) } } ( t - 1 ) \Phi = T ^ { 2 } \Phi .
$$

Based on this, we have

$$
\begin{array} { r l } & { \Bigg | \displaystyle \sum _ { t \in \mathcal { T } ^ { ( r ) } } \kappa _ { n } ( t ) - \gamma _ { n } ( t ) \Bigg | = \Bigg | \sum _ { t \in \mathcal { T } ^ { ( r ) } } \zeta _ { n } ( t + 1 ) - \zeta _ { n } ( t ) \Bigg | } \\ & { = | \zeta _ { n } ( T + 1 ) | \leq \sqrt { 2 \Delta _ { n } } \leq T \sqrt { 2 \Phi } } \end{array}\tag{36}
$$

For energy consumption, we have

$$
\begin{array} { r l } & { \displaystyle \sum _ { t \in \mathcal { T } ^ { ( r ) } } \left( e _ { n } ^ { \dagger } ( t ) - \frac { E _ { n } ^ { \mathrm { c o n s } } - \xi _ { n } ^ { ( r ) } } { T } \right) } \\ & { \leq \displaystyle \sum _ { t \in \mathcal { T } ^ { ( r ) } } q _ { n } ( t + 1 ) - q _ { n } ( t ) \leq T \sqrt { 2 \Phi } . } \end{array}
$$

Adding $\begin{array} { r l } & { \frac { V ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } \big ( \gamma _ { n } ^ { \dagger } ( t ) + \frac { 1 } { T } \big ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } { 2 \alpha _ { n } ^ { ( r ) } - 1 } } \\ & { \mathrm { l e - r o u n d ~ d r i f t - p l u s - p e n a l t y ~ f u n c } } \end{array}$ on both sides of (35), the sing tion is bounded by

$$
\begin{array} { r l } & { \Delta _ { n } ( t ) + \frac { V ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } } { 2 \alpha _ { n } ^ { ( r ) } } \left( \gamma _ { n } ^ { \dagger } ( t ) + \frac { 1 } { T } \right) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } \\ & { \leq \Phi + q _ { n } ( t ) \epsilon _ { n } ^ { \dagger } ( t ) + \zeta _ { n } ( t ) \left( \kappa _ { n } ^ { \dagger } ( t ) - \gamma _ { n } ^ { \dagger } ( t ) \right) } \\ & { + \frac { V ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } } { 2 \alpha _ { n } ^ { ( r ) } } \left( \gamma _ { n } ^ { \dagger } ( t ) + \frac { 1 } { T } \right) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } \\ & { \leq \Phi + q _ { n } ( t ) \epsilon _ { n } ^ { * } ( t ) + \zeta _ { n } ( t ) \left( \kappa _ { n } ^ { * } ( t ) - \frac { k _ { n } ^ { ( r ) * } } { T } \right) } \\ & { + \frac { V ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } } { 2 \alpha _ { n } ^ { ( r ) } - 1 } \left( \frac { k _ { n } ^ { ( r ) * } + 1 } { T } \right) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } . } \end{array}\tag{37}
$$

Taking a telescopic sum for both sides, we have

$$
\begin{array} { r l } & { \Delta _ { n } + \displaystyle \sum _ { t \in \mathcal { T } ^ { ( r ) } } \frac { V ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } \left( \gamma _ { n } ^ { \dagger } ( t ) + \frac { 1 } { T } \right) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } { 2 \alpha _ { n } ^ { ( r ) } - 1 } } \\ & { \leq T ^ { 2 } \Phi + \frac { T ^ { 2 \alpha _ { n } ^ { ( r ) } } V ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } ( k _ { n } ^ { ( r ) * } + 1 ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } { 2 \alpha _ { n } ^ { ( r ) } - 1 } . } \end{array}\tag{38}
$$

For any positive sequence $\{ \kappa _ { n } ( t ) \} _ { t \in \mathcal { T } ^ { ( r ) } }$ , the function $f ( x ) =$ $x ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) }$ is convex and monotonically decreasing over $x >$ 0. By Jensen’s inequality, we have

$$
\begin{array} { l } { \displaystyle T ^ { 2 \alpha _ { n } ^ { ( r ) } } \bigg ( \sum _ { t \in \mathcal { T } ^ { ( r ) } } \gamma _ { n } ^ { \dagger } ( t ) + 1 \bigg ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } \\ { \leq \displaystyle \sum _ { t \in \mathcal { T } ^ { ( r ) } } \bigg ( \gamma _ { n } ^ { \dagger } ( t ) + \frac { 1 } { T } \bigg ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } \end{array}\tag{39}
$$

Also, since the function $g ( x ) = ( x + 1 ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) }$ is continuously differentiable and Lipschitz continuous on $x \geq 0 .$ , and its derivative satisfies

$$
\operatorname* { s u p } _ { x \geq 0 } | g ^ { \prime } ( x ) | = \operatorname* { s u p } _ { x \geq 0 } ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) ( x + 1 ) ^ { - 2 \alpha _ { n } ^ { ( r ) } } \leq 2 \alpha _ { n } ^ { ( r ) } - 1 ,
$$

it follows from the mean value theorem that, for any $A , B \geq 0 .$

$$
g ( x ) \leq g ( y ) + ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) | x - y | .
$$

Therefore, we have

$$
\begin{array} { r l } & { \Bigg ( \displaystyle \sum _ { t \in \mathcal { T } ^ { ( r ) } } \kappa _ { n } ^ { \dagger } ( t ) + 1 \Bigg ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } \leq \Bigg ( \displaystyle \sum _ { t \in \mathcal { T } ^ { ( r ) } } \gamma _ { n } ^ { \dagger } ( t ) + 1 \Bigg ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } \\ & { + \left( 2 \alpha _ { n } ^ { ( r ) } - 1 \right) \left| \displaystyle \sum _ { t \in \mathcal { T } ^ { ( r ) } } \kappa _ { n } ^ { \dagger } ( t ) - \gamma _ { n } ^ { \dagger } ( t ) \right| } \\ & { \leq \Bigg ( \displaystyle \sum _ { t \in \mathcal { T } ^ { ( r ) } } \gamma _ { n } ^ { \dagger } ( t ) + 1 \Bigg ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } + ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) T \sqrt { 2 \Phi } . } \end{array}
$$

Substituting (39) and (40) into (38), we have

$$
\begin{array} { r l } & { \underset { n \in \mathcal { N } ^ { ( r ) } } { \sum } \frac { ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } } { 2 \alpha _ { n } ^ { ( r ) } - 1 } \bigg ( \underset { t \in \mathcal { T } ^ { ( r ) } } { \sum } \kappa _ { n } ^ { \dagger } ( t ) + 1 \bigg ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } \\ & { \leq \underset { n \in \mathcal { N } ^ { ( r ) } } { \sum } \frac { ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } ( k _ { n } ^ { ( r ) * } + 1 ) ^ { - ( 2 \alpha _ { n } ^ { ( r ) } - 1 ) } } { 2 \alpha _ { n } ^ { ( r ) } - 1 } } \\ & { + \underset { n \in \mathcal { N } ^ { ( r ) } } { \sum } \frac { T ^ { 2 - 2 \alpha _ { n } ^ { ( r ) } } \Phi } { V } + ( C _ { n } ^ { ( r ) } ) ^ { 2 } \alpha _ { n } ^ { ( r ) } T \sqrt { 2 \Phi } . } \end{array}
$$

Theorem 2 is proved.

## APPENDIX D PROOF OF PROPOSITION 1

The Lagrangian of P3 is given by:

$$
\begin{array} { l } { \displaystyle \mathcal { L } = \sum _ { n \in N ^ { ( r ) } } \sum _ { z \in \mathcal { Z } } \frac { \zeta _ { n } ( t ) \tau \beta s _ { n , z } ( t ) \log _ { 2 } ( 1 + \Gamma _ { n , z } ( t ) ) } { b + \lceil \log _ { 2 } I \rceil } } \\ { + \sum _ { n \in N ^ { ( r ) } } \sum _ { z \in \mathcal { Z } } \tau q _ { n } ( t ) s _ { n , z } ( t ) p _ { n , z } ( t ) } \\ { + \sum _ { z \in \mathcal { Z } } \lambda _ { z } ( \displaystyle \sum _ { n \in N ^ { ( r ) } } s _ { n , z } ( t ) - 1 ) - \sum _ { n \in N ^ { ( r ) } } \sum _ { z \in \mathcal { Z } } s _ { n , z } s _ { n , z } ( t ) } \\ { + \sum _ { n \in N ^ { ( r ) } } \nu _ { n } ( \displaystyle \sum _ { z \in \mathcal { Z } } p _ { n , z } ( t ) - p _ { n } ^ { \mathrm { m a x } } ) - \sum _ { n \in N ^ { ( r ) } } \sum _ { z \in \mathcal { Z } } \chi _ { n , z } p _ { n , z } ( t ) . } \end{array}
$$

Then the KKT condition is given by:

$$
\begin{array} { r l } & { \frac { \zeta _ { n } ( t ) \tau \beta \log _ { 2 } ( 1 + \Gamma _ { n , z } ( t ) ) } { b + \vert \log _ { 2 } T \vert } + \tau q _ { n } ( t ) p _ { n , z } ( t ) } \\ & { + \lambda _ { z } - \varsigma _ { n , z } = 0 , \quad \forall n \in \mathcal { N } ^ { ( r ) } , \forall z \in \mathcal { Z } , } \\ & { \frac { \zeta _ { n } ( t ) \tau \beta s _ { n , z } ( t ) \frac { \Vert h _ { n , z } ( t ) \Vert ^ { 2 } } { \beta N _ { 0 } } } { \left( 1 + \frac { p _ { n , z } ( t ) \Vert h _ { n , z } ( t ) \Vert ^ { 2 } } { \beta N _ { 0 } } \right) \ln 2 } + \tau q _ { n } ( t ) s _ { n , z } ( t ) } \\ & { + \nu _ { n } - \chi _ { n , z } = 0 , \quad \forall n \in \mathcal { N } ^ { ( r ) } , \forall z \in \mathcal { Z } , } \end{array}\tag{42a}
$$

(42b)

$$
\lambda _ { z } \big ( \sum _ { n \in \mathcal { N } ^ { ( r ) } } s _ { n , z } ( t ) - 1 \big ) = 0 , \quad \forall z \in \mathcal { Z } ,\tag{42c}
$$

$$
\nu _ { n } ( \sum _ { z \in \mathcal { Z } _ { n } ( t ) } p _ { n , z } ( t ) - p _ { n } ^ { \mathrm { m a x } } ) = 0 , \quad \forall n \in \mathcal { N } ^ { ( r ) } ,\tag{42d}
$$

$$
\varsigma _ { n , z } s _ { n , z } ( t ) = 0 , \quad \forall n \in \mathcal { N } ^ { ( r ) } , \forall z \in \mathcal { Z } ,\tag{42e}
$$

$$
{ \chi } _ { n , z } p _ { n , z } ( t ) = 0 , \quad \forall n \in \mathcal { N } ^ { ( r ) } , \forall z \in \mathcal { Z } ,\tag{42f}
$$

$$
\lambda _ { n , z } , \varsigma _ { n , z } , \nu _ { n } , \chi _ { n , z } \geq 0 , \quad \forall n \in \mathcal { N } ^ { ( r ) } , \forall z \in \mathcal { Z } ,\tag{42g}
$$

where $\lambda _ { z } , \varsigma _ { n , z } , \nu _ { n }$ and $\chi _ { n , z }$ are non-negative Lagrange multipliers. Combining (42a) and (42e), for every $n \in \mathcal { N } ^ { ( r ) }$ and $z \in { \mathcal { Z } }$ , we have

$$
s _ { n , z } ( t ) \left( \frac { \zeta _ { n } ( t ) \tau \beta \log _ { 2 } ( 1 + \Gamma _ { n , z } ( t ) ) } { b + \lceil \log _ { 2 } T \rceil } + \tau q _ { n } ( t ) p _ { n , z } ( t ) + \lambda _ { z } \right) = 0 .
$$

Since $\varsigma _ { n , z } \geq 0 _ { : }$ , (42a) implies that

$$
\frac { \zeta _ { n } ( t ) \tau \beta \log _ { 2 } ( 1 + \Gamma _ { n , z } ( t ) ) } { b + \lceil \log _ { 2 } I \rceil } + \tau q _ { n } ( t ) p _ { n , z } ( t ) \geq - \lambda _ { z } .\tag{43}
$$

If $s _ { n , z } ( t )$ is non-zero, the left-hand side of (43) reaches its lower bound, $\lambda _ { z }$ . Therefore, the proposition is proved.

## APPENDIX E PROOF OF PROPOSITION 3

If $\zeta _ { n } ( t ) \geq 0$ , the objective function is nondecreasing with respect to each $p _ { n , z } ( t )$ . Therefore, the optimal solution is

$$
\begin{array} { r } { p _ { n , z } ( t ) = 0 , \quad \forall z \in \mathcal { Z } _ { n } ( t ) . } \end{array}
$$

If $\zeta _ { n } ( t ) < 0$ , we combine (42b) and (42f) to obtain

$$
p _ { n , z } ( t ) \left( \frac { \zeta _ { n } ( t ) \tau \beta s _ { n , z } ( t ) \frac { \| h _ { n , z } ( t ) \| ^ { 2 } } { \beta N _ { 0 } } } { \left( 1 + \frac { p _ { n , z } ( t ) \left\| h _ { n , z } ( t ) \right\| ^ { 2 } } { \beta N _ { 0 } } \right) \ln 2 } + \tau q _ { n } ( t ) s _ { n , z } ( t ) + \nu _ { n } \right) = 0 .
$$

Then, we have

$$
p _ { n , z } ( t ) = \operatorname* { m a x } \left( \frac { - \zeta _ { n } ( t ) \tau \beta } { ( \tau q _ { n } ( t ) + \nu _ { n } ) \ln 2 } - \frac { \beta N _ { 0 } } { \left\| h _ { n , z } ( t ) \right\| ^ { 2 } } , 0 \right) .
$$

if $\nu _ { n } > 0$ , the total power constraint is active, i.e.,

$$
\sum _ { z \in \mathcal { Z } _ { n } ^ { * } ( t ) } \left( \frac { - \zeta _ { n } ( t ) \tau \beta } { \left( \tau q _ { n } ( t ) + \nu _ { n } \right) \ln 2 } - \frac { \beta N _ { 0 } } { \left\| h _ { n , z } ( t ) \right\| ^ { 2 } } \right) = p _ { n } ^ { \mathrm { m a x } } ,
$$

where ${ \mathcal { Z } } _ { n } ^ { * } ( t )$ is the subset of RBs to which vehicle n assigns positive power. Then, there is

$$
\nu _ { n } = \frac { - | \mathcal { Z } _ { n } ^ { * } ( t ) | \zeta _ { n } ( t ) \tau \beta } { \left( p _ { n } ^ { \mathrm { m a x } } + \sum _ { z \in \mathcal { Z } _ { n } ^ { * } ( t ) } \frac { \beta N _ { 0 } } { \left\| h _ { n , z } ( t ) \right\| ^ { 2 } } \right) \ln 2 } - \tau q _ { n } ( t ) .
$$

Substituting $\nu _ { n }$ back gives

$$
p _ { n , z } ( t ) = \frac { p _ { n } ^ { \mathrm { m a x } } + \sum _ { z ^ { \prime } \in \mathcal { Z } _ { n } ^ { * } ( t ) } \frac { \beta N _ { 0 } } { \left\| h _ { n , z ^ { \prime } } ( t ) \right\| ^ { 2 } } } { | \mathcal { Z } _ { n } ^ { * } ( t ) | } - \frac { \beta N _ { 0 } } { \left\| h _ { n , z } ( t ) \right\| ^ { 2 } } .
$$

If $\nu _ { n } = 0$ , the optimal power allocation becomes

$$
p _ { n , z } ( t ) = \frac { - \zeta _ { n } ( t ) \beta } { q _ { n } ( t ) \ln 2 } - \frac { \beta N _ { 0 } } { \left\| h _ { n , z } ( t ) \right\| ^ { 2 } } .
$$

Combining the above cases, when $\zeta _ { n } ( t ) < 0$ and $q _ { n } ( t ) > 0$ the optimal power allocation is given by

$$
\begin{array} { r l } & { p _ { n , z } ( t ) = \operatorname* { m i n } \left( \frac { p _ { n } ^ { \mathrm { m a x } } + \sum _ { z ^ { \prime } \in \mathcal { Z } _ { n } ^ { * } ( t ) } \frac { \beta N _ { 0 } } { \left\| h _ { n , z ^ { \prime } } ( t ) \right\| ^ { 2 } } } , \frac { - \zeta _ { n } ( t ) \beta } { q _ { n } ( t ) \ln 2 } \right) } { | \mathcal { Z } _ { n } ^ { * } ( t ) | }  \\ & { \quad \quad \quad - \frac { \beta N _ { 0 } } { \left\| h _ { n , z } ( t ) \right\| ^ { 2 } } , \quad \forall z \in \mathcal { Z } _ { n } ^ { * } ( t ) . } \end{array}
$$

When $\zeta _ { n } ( t ) ~ < ~ 0$ and $q _ { n } ( t ) ~ = ~ 0$ , the objective function is monotonically decreasing with respect to the achievable transmission rate. Therefore, the total power constraint is active, and the optimal power allocation reduces to

$$
p _ { n , z } ( t ) = \frac { p _ { n } ^ { \mathrm { m a x } } + \sum _ { z ^ { \prime } \in \mathcal { Z } _ { n } ^ { * } ( t ) } \frac { \beta N _ { 0 } } { \| h _ { n , z ^ { \prime } } ( t ) \| ^ { 2 } } } { | \mathcal { Z } _ { n } ^ { * } ( t ) | } - \frac { \beta N _ { 0 } } { \| h _ { n , z } ( t ) \| ^ { 2 } } ,
$$

Therefore, the proposition is proved.