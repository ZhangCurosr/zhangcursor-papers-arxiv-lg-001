# Resilient Decentralized Wireless Federated Learning via Gradient Tracking with AdamW

Nguyen Van Thieu, Ti Ti Nguyen, Ons Aouedi, Vu Nguyen Ha, and Symeon Chatzinotas Interdisciplinary Centre for Security, Reliability and Trust (SnT), University of Luxembourg, Luxembourg {vanthieu.nguyen, titi.nguyen, ons.aouedi, vu-nguyen.ha, symeon.chatzinotas}@uni.lu

Abstract—Wireless Internet-of-Things (IoT) edge networks require decentralized learning (DecL) methods that can operate reliably under both heterogeneous local data and communicationconstrained wireless links. However, existing decentralized optimization schemes often incur substantial communication overhead and degraded performance when transmissions are constrained by strict airtime budgets, fading channels, and packet losses. This paper proposes QEF-GT-AdamW, a communicationefficient and outage-resilient algorithm for DecL over wireless communication (WCom) networks. The proposed method combines gradient tracking to mitigate the effect of non-IID data, AdamW-based adaptive optimization to improve training stability, and dual-stream biased quantization with error feedback to reduce communication payloads for both model and tracking exchanges. To address unreliable broadcast communication, the proposed framework further employs a local fallback strategy when scheduled packets are not successfully received. We explicitly model the effect of bandwidth, transmit power, airtime constraints, and fading channels on DecL performance, and establish convergence guarantees for the proposed algorithm under compressed and unreliable wireless communication. Experimental results on heterogeneous MNIST and CIFAR-10 settings show that QEF-GT-AdamW consistently improves robustness and convergence performance over representative DecL baselines while achieving favorable accuracy-communication trade-offs under limited wireless resources.

Index Terms—Decentralized Learning, Gradient Tracking, Error Feedback, Top-K Compression, Wireless Communication, Packet Loss

## I. INTRODUCTION

IoT wireless communication (WCom) networks increasingly rely on decentralized learning (DecL) to train models directly over distributed devices without a central coordinator [1]. While this paradigm improves scalability and robustness, its practical deployment is challenged by non-IID local data and unreliable WCom arising from limited bandwidth, strict airtime budgets, fading channels, and packet losses. These issues are tightly coupled in wireless-based DecL, where WCom constraints directly affect learning performance [2].

Among DecL optimization methods, gradient tracking (GT) is an effective mechanism for mitigating client drift under heterogeneous data [3], since it enables each node to asymptotically follow the network-wide gradient. However, classical

GT requires repeated exchanges of both model and tracking variables, which leads to substantial WCom overhead. Recent work such as GTAdam [4] combines GT with Adam-type adaptive momentum estimation to improve convergence speed and robustness, but existing GT-Adam methods are mainly developed under ideal communication assumptions. In parallel, gossip-based SGD [5] and compressed DecL methods [2], [6] reduce payload sizes through limited information exchange and quantization, yet they typically either lack the heterogeneity-handling capability of GT or do not explicitly address the interaction among adaptive optimization, compression, and unreliable wireless links. However, a practically deployable framework for wireless DecL under non-IID data and constrained communication remains missing.

Motivated by this gap, we develop QEF-GT-AdamW, a communication-efficient and outage-resilient DecL framework for wireless edge networks. The proposed approach combines GT for heterogeneity mitigation, AdamW-based adaptive updates, and dual-stream quantization with error feedback (EF) to reduce the payloads associated with exchanging both model and tracking variables, while a local fallback mechanism improves robustness when neighbor packets are not successfully received. In contrast to idealized DecL models, the framework explicitly models synchronized wireless learning with computation-dependent airtime, bandwidth and transmitpower constraints, and fading-induced outages. We also establish convergence guarantees for smooth convex objectives under compression and unreliable communication, together with a stronger linear-rate characterization under an additional gradient-dominance condition. Experiments on heterogeneous MNIST and CIFAR-10 settings over fading wireless channels show that the proposed method achieves favorable robustnessaccuracy-communication trade-offs compared with representative decentralized baselines.

## II. SYSTEM MODEL

We consider a DecL framework over a WCom network with N nodes, where each node holds a local dataset and exchanges information only with its neighbors over WCom channels. The goal is to minimize the overall average empirical loss under non-IID data distributions.

## A. DecL Background

1) Decentralized GT Learning: GT enables nodes to asymptotically follow the global gradient, even under heterogeneous local data distributions. Each node i maintains a local model $\mathbf { x } _ { i } ^ { k } \in \mathbb { R } ^ { d }$ and an auxiliary GT variable $\mathbf { y } _ { i } ^ { k } \in \mathbb { R } ^ { d }$ . Let $f _ { i } : \mathbb { R } ^ { d } $ R be the local loss function at node i, and let $\textbf { W } = ~ [ w _ { i j } ] ~ \in ~ \mathbb { R } ^ { N \times N }$ be a fixed doublystochastic baseline mixing matrix satisfying $\begin{array} { r } { \sum _ { j = 1 } ^ { N } w _ { i j } = \mathrm { ~ \dot { 1 } ~ } } \end{array}$ $\begin{array} { r } { \sum _ { i = 1 } ^ { N } w _ { i j } \ = \ 1 } \end{array}$ , and $w _ { i j } ~ \geq ~ 0$ . The classical GT updates [3] at iteration k are: $\begin{array} { r } { \mathbf { \dot { x } } _ { i } ^ { k + 1 } \ = \ \sum _ { j = 1 } ^ { N } w _ { i j } \mathbf { x } _ { j } ^ { k } \ - \ \alpha \mathbf { y } _ { i } ^ { k } \ } \end{array}$ , and $\begin{array} { r } { \mathbf { y } _ { i } ^ { k + 1 } = \sum _ { j = 1 } ^ { N } w _ { i j } \mathbf { y } _ { j } ^ { k } + \nabla f _ { i } ( \mathbf { x } _ { i } ^ { k + 1 } ) - \nabla f _ { i } ( \mathbf { x } _ { i } ^ { k } ) } \end{array}$ , where $\alpha > 0$ is a step size. This mechanism enables the tracking variables to follow the network-wide gradient and supports convergence toward the global objective

![](images/201bf8b3a597adf3f04e401d98a811d84773702c03cc6e3868bc9f17a997ef33.jpg)  
Fig. 1: A TS-based learning and information exchange framework.

$$
F ( \mathbf { x } ) \triangleq \frac { 1 } { N } \sum _ { i = 1 } ^ { N } f _ { i } ( \mathbf { x } ) .\tag{1}
$$

2) Adaptive Optimization and AdamW: Adam is an adaptive optimization algorithm that uses exponential moving averages of first- and second-order gradients to accelerate convergence. However, its coupling of weight decay with adaptive updates can lead to inconsistent regularization and reduced generalization, especially in decentralized settings. AdamW [7] addresses this by decoupling weight decay from gradient-based updates, leading to more stable optimization.

## B. DecL with Wireless Communication

1) Time-Slot based Learning and Information Exchange Framework: A synchronized time-slot (TS) protocol governs training and communication, as illustrated in Fig. 1. In each TS, nodes perform local updates and transmit over orthogonal subcarriers with limited airtime and unreliable links. Each TS has a strict deadline $\Delta t _ { \mathrm { m a x } } ^ { \mathrm { ( i t e r ) } }$ . Node i’s local computation time is modeled as $T _ { \mathrm { c o m p } , i } ^ { k } = R _ { i } ^ { k } \Delta T$ , where $R _ { i } ^ { k }$ follows a truncated normal distribution [8] and $\Delta T > 0$ denotes the basic computation-time unit. The remaining airtime available to node i for transmission is

$$
\begin{array} { r } { \Delta t _ { \mathrm { a i r } , i } ^ { k } = \Delta t _ { \mathrm { m a x } } ^ { \mathrm { ( i t e r ) } } - T _ { \mathrm { c o m p } , i } ^ { k } . } \end{array}\tag{2}
$$

This implies that devices with slower computation have less time to transmit, thus increasing the risk of outage.

2) Channel Model and Outage Probability: Let $d _ { i j }$ denote the Euclidean distance between nodes i and $j .$ The pathloss is given by $\nu _ { i j } = \nu _ { 0 } ( d _ { 0 } / d _ { i j } ) ^ { \zeta }$ , where $\zeta > 0$ is the pathloss exponent, $\nu _ { 0 }$ is the reference pathloss at distance $d _ { 0 } ,$ and $d _ { 0 }$ is the reference distance. The small-scale fading coefficient $h _ { i j } ^ { k }$ follows a Rician distribution with Rician factor $\kappa \geq 0$

Let $P _ { i }$ and $B _ { i }$ denote the transmission power and bandwidth of node $i ,$ respectively. The average SNR of the transmission from node i to node j is $\begin{array} { r } { \bar { \gamma } _ { i j } = \frac { \tilde { P _ { i } } \nu _ { i j } } { N _ { 0 } B _ { i } } } \end{array}$ , where $N _ { 0 }$ is the noise power spectral density. The outage probability at iteration k is estimated as [9]

$$
p _ { i j } ^ { \mathsf { o u t a g e , k } } = 1 - \mathcal { Q } _ { 1 } ^ { \mathsf { M } } \left( \sqrt { 2 \kappa } , \sqrt { \frac { 2 ( 1 + \kappa ) \gamma _ { i } ^ { \mathsf { t h } , \mathsf { k } } } { \bar { \gamma } _ { i j } } } \right) ,\tag{3}
$$

where $\mathcal { Q } _ { 1 } ^ { \mathsf { M } } ( \cdot , \cdot )$ is the first-order Marcum Q-function [9], and $\gamma _ { i } ^ { \mathrm { t h , k } }$ is the SNR threshold of node i at iteration k. The threshold $\gamma _ { i } ^ { \mathrm { t h , k } }$ is determined by the amount of information exchanged and the available airtime according to $B _ { i } \log _ { 2 } \Big ( 1 + \gamma _ { i } ^ { \mathrm { t h , k } } \Big ) = D _ { i } / \Delta t _ { \mathrm { a i r } , i } ^ { k } ,$ where $D _ { i }$ denotes the amount of information transmitted by node i, which depends on the model dimension d and the compression scheme.

## III. PROPOSED ALGORITHM: QEF-GT-ADAMW

This section presents QEF-GT-AdamW, which combines robust consensus mixing under packet losses, AdamW-based adaptive updates, GT for heterogeneous data, and dual-stream quantization with EF for both model and tracking exchanges.

## A. Initialization and Robust Mixing

Each node i maintains a local model $\mathbf { x } _ { i } ^ { k } \in \mathbb { R } ^ { d }$ , a gradienttracking variable $\mathbf { y } _ { i } ^ { k } \in \mathbb { R } ^ { d }$ , AdamW moment buffers $( \mathbf { m } _ { i } ^ { k } , \mathbf { v } _ { i } ^ { k } )$ and dual EF residuals $( \mathbf { e } _ { x , i } ^ { k } , \mathbf { e } _ { y , i } ^ { k } )$ . Let ${ \mathcal { N } } _ { i }$ denote the neighbor set of node i in the baseline communication graph, and let $\mathcal { N } _ { i } ^ { \mathrm { i n } }$ denote its set of potential in-neighbors. To prevent duplicate processing, node i stores the last accepted sequence number LastSeq [j] for each $j \in \mathcal { N } _ { i } ^ { \mathrm { i n } }$

To ensure that both the primal and tracking streams are initialized consistently before the first communication round, the warm start is performed as in Algorithm 1. Starting from an initial model $\mathbf { x } _ { i } ^ { 0 } .$ , node i computes its local gradient $\mathbf { g } _ { i } ^ { 0 } =$ $\nabla f _ { i } ( \mathbf { x } _ { i } ^ { 0 } )$ and sets $\mathbf { \dot { y } } _ { i } ^ { 0 } = \mathbf { g } _ { i } ^ { 0 }$ . The AdamW moment buffers are initialized to zero. For consistency with the subsequent EF recursion, the pre-initialization residuals are set to ${ \bf e } _ { x , i } ^ { - 1 } = { \bf 0 }$ and $\mathbf e _ { y , i } ^ { - 1 } = \mathbf 0$ . The initial model and tracking variables are then compressed using the corresponding compressors $\mathcal { Q } _ { x }$ and $\mathcal { Q } _ { y }$ before being broadcast to neighboring nodes.

At iteration $k ,$ node i accepts a decoded packet $( \hat { \mathbf { x } } _ { j } ^ { k } , \hat { \mathbf { y } } _ { j } ^ { k } )$ from neighbor $j$ only if its sequence number is larger than $\operatorname { L a s t S e q } _ { i } [ j ]$ ; otherwise, the packet is treated as missing. Here, the decoded vectors correspond to the compressor outputs, i.e., $\hat { \mathbf { x } } _ { j } ^ { k } = \mathbf { c } _ { x , j } ^ { k }$ and $\hat { \mathbf { y } } _ { j } ^ { k } = \mathbf { c } _ { y , j } ^ { k }$

Let $a _ { j  i } ^ { k } ~ \in ~ \{ 0 , 1 \}$ indicate successful reception, where $a _ { j  i } ^ { k } = 1$ means that node i successfully accepts the packet transmitted by node $j ,$ and $a _ { j  i } ^ { k } ~ = ~ 0$ otherwise. Under where $\alpha _ { k } \ > \ 0$ is the step size, $\varepsilon > 0$ is a small numerical constant, and $\lambda \geq 0$ is the decoupled weight-decay coefficient. After the primal update, node i computes the new local gradient $\mathbf { g } _ { i } ^ { k + 1 } = \nabla f _ { i } ( \mathbf { x } _ { i } ^ { k + 1 } )$ , and updates the tracking variable as

Algorithm 1 INITIALIZATION AT NODE i (WARM START)   
1: Given: mixing weights $\{ w _ { i j } \}$ ; stepsizes $\{ \alpha _ { k } \}$ ; AdamW   
parameters $( \beta _ { m } , \beta _ { v } , \varepsilon , \lambda )$ ; compressors $\mathcal { Q } _ { x } , \mathcal { Q } _ { y }$   
2: Initialize $\mathbf { x } _ { i } ^ { 0 } .$   
3: Compute $\mathbf { g } _ { i } ^ { 0 } = \nabla f _ { i } ( \mathbf { x } _ { i } ^ { 0 } )$ and set $\mathbf { y } _ { i } ^ { 0 } = \mathbf { g } _ { i } ^ { 0 } .$   
4: Set ${ \bf m } _ { i } ^ { 0 } = \dot { { \bf 0 } }$ and $\mathbf { v } _ { i } ^ { 0 } = \mathbf { 0 } .$   
5: Set $\mathbf { e } _ { x , i } ^ { - 1 } = \mathbf { 0 }$ and $\begin{array} { r } { \dot { \mathbf { e } } _ { y , i } ^ { - 1 } = \mathbf { 0 } . } \end{array}$   
6: Set $\mathrm { S e q } _ { i } \gets 0$ and LastSeq $_ i [ j ]  - 1$ for all $j \in \mathcal { N } _ { i } ^ { \mathrm { i n } } .$   
7: Warm-start compression: $\tilde { \mathbf { x } } _ { i } ^ { 0 } \gets \mathbf { x } _ { i } ^ { 0 } { + } \mathbf { e } _ { x , i } ^ { - 1 } , \mathbf { c } _ { x , i } ^ { 0 } \gets \mathcal { Q } _ { x } ( \tilde { \mathbf { x } } _ { i } ^ { 0 } )$   
$\mathbf { e } _ { x , i } ^ { 0 }  \tilde { \mathbf { x } } _ { i } ^ { 0 } - \mathbf { c } _ { x , i } ^ { 0 ^ { \top } } .$   
8: Similarly, $\tilde { \mathbf { y } } _ { i } ^ { 0 } \gets \mathbf { y } _ { i } ^ { 0 } + \mathbf { e } _ { y , i } ^ { - 1 } , \mathbf { c } _ { y , i } ^ { 0 } \gets \mathcal { Q } _ { y } ( \tilde { \mathbf { y } } _ { i } ^ { 0 } ) , \mathbf { e } _ { y , i } ^ { 0 } \gets \tilde { \mathbf { y } } _ { i } ^ { 0 } -$   
$\mathbf { c } _ { y , i } ^ { 0 } .$   
9: Broadcast packet $( \mathrm { S e q } _ { i } , \mathbf { c } _ { x , i } ^ { 0 } , \mathbf { c } _ { y , i } ^ { 0 } )$ to ${ \mathcal { N } } _ { i } .$   
the outage model in (3), $\mathrm { P r } ( a _ { j \to i } ^ { k } ~ = ~ 1 ) ~ = ~ 1 - p _ { j i } ^ { \mathrm { o u t a g e , k } } ,$   
To preserve stochastic mixing under packet losses, missing   
neighbor weights are reassigned to the self-loop:   
$b _ { i j } ^ { k } \triangleq w _ { i j } a _ { j  i } ^ { k } , \quad j \neq i , \qquad b _ { i i } ^ { k } \triangleq 1 - \sum _ { j \neq i } b _ { i j } ^ { k } .$ (4)   
Thus, the realized mixing matrix $\mathbf { B } ^ { k } = [ b _ { i i } ^ { k } ] \in \mathbb { R } ^ { N \times N }$ remains   
row-stochastic. The robust mixed variables used by node i are   
$\begin{array} { r } { \mathbf { X } _ { i } ^ { k } = b _ { i i } ^ { k } \mathbf { x } _ { i } ^ { k } + \sum _ { j \neq i } b _ { i j } ^ { k } \hat { \mathbf { x } } _ { j } ^ { k } } \end{array}$ , and $\begin{array} { r } { \mathbf { Y } _ { i } ^ { k } = b _ { i i } ^ { k } \mathbf { y } _ { i } ^ { k } + et { } { ' } \sum _ { j \neq i } b _ { i j } ^ { k } \hat { \mathbf { y } } _ { j } ^ { k } } \end{array}$   
Define the stacked vectors   
$\begin{array} { r } { \mathbf { x } ^ { k } \triangleq \operatorname { c o l } ( \mathbf { x } _ { 1 } ^ { k } , \ldots , \mathbf { x } _ { N } ^ { k } ) , \quad \mathbf { y } ^ { k } \triangleq \operatorname { c o l } ( \mathbf { y } _ { 1 } ^ { k } , \ldots , \mathbf { y } _ { N } ^ { k } ) , } \end{array}$   
$\begin{array} { r } { \hat { \mathbf { x } } ^ { k } \triangleq \mathrm { c o l } ( \hat { \mathbf { x } } _ { 1 } ^ { k } , \hdots , \hat { \mathbf { x } } _ { N } ^ { k } ) , \quad \hat { \mathbf { y } } ^ { k } \triangleq \mathrm { c o l } ( \hat { \mathbf { y } } _ { 1 } ^ { k } , \hdots , \hat { \mathbf { y } } _ { N } ^ { k } ) , } \end{array}$ (5)   
$\mathbf { X } ^ { k } \triangleq \operatorname { c o l } ( \mathbf { X } _ { 1 } ^ { k } , \ldots , \mathbf { X } _ { N } ^ { k } ) , \quad \mathbf { Y } ^ { k } \triangleq \operatorname { c o l } ( \mathbf { Y } _ { 1 } ^ { k } , \ldots , \mathbf { Y } _ { N } ^ { k } ) .$   
Furthermore, define $\mathbf { A } ^ { k } = [ A _ { i j } ^ { k } ] \in \mathbb { R } ^ { N \times N }$ by $A _ { i j } ^ { k } = b _ { i j } ^ { k }$ for   
$j \neq i$ and $0 \ { \mathrm { i f } } \ j = i .$ Then, the robust mixing operation admits   
the compact form   
$\mathbf { X } ^ { k } = ( \mathbf { B } ^ { k } \otimes \mathbf { I } _ { d } ) \mathbf { x } ^ { k } + ( \mathbf { A } ^ { k } \otimes \mathbf { I } _ { d } ) ( \hat { \mathbf { x } } ^ { k } - \mathbf { x } ^ { k } ) ,$ (6)   
with the analogous expression for $\mathbf { Y } ^ { k }$ . These mixed variables   
serve as the robust consensus states used in the local adaptive   
update and gradient-tracking steps described next.   
B. AdamW Update, GT, and Dual Quantization with EF   
Using the mixed tracking variable $\mathbf { Y } _ { i } ^ { k }$ as a robust estimate of   
the global gradient, node i updates its local moment estimates   
and applies bias correction according to   
$\mathbf { m } _ { i } ^ { k + 1 } = \beta _ { m } \mathbf { m } _ { i } ^ { k } + ( 1 - \beta _ { m } ) \mathbf { Y } _ { i } ^ { k } ,$   
v<sup>k+1</sup><sub>i</sub> = β<sub>v</sub>v<sup>k</sup><sub>i</sub> + (1 − β<sub>v</sub>)  Y<sup>k</sup><sub>i</sub> ⊙ Y<sup>k</sup><sub>i</sub>  , (7)   
mˆ <sup>k+1</sup><sub>i</sub> = m<sup>k+1</sup><sub>i</sub> /(1 − β<sup>k+1</sup><sub>m</sub> ), vˆ<sup>k+1</sup><sub>i</sub> = v<sup>k+1</sup><sub>i</sub> (1 − β<sup>k+1</sup><sub>v</sub> ), (8)   
where ⊙ denotes element-wise multiplication, and $\beta _ { m } , \beta _ { v } \in$   
(0, 1) are the momentum parameters. The primal model is   
subsequently updated using the AdamW rule:   
x<sup>k+1</sup><sub>i</sub> = X<sup>k</sup><sub>i</sub> − α<sub>k</sub> mˆ <sup>k+1</sup> + λX<sup>k</sup> (9)   
qvˆk+1 + ε

Algorithm 2 QEF-GT-ADAMW AT NODE i (ITERATION $k \geq 0 )$   
1: Reception (decode new packets only):   
2: for each $j \in \mathcal N _ { i }$ do   
3: if $a _ { j  i } ^ { k } = 1$ and received $\mathrm { S e q } _ { j } > \mathrm { L a s t S e q } _ { i } [ j ]$ then   
4: Decode payload into $( \hat { \mathbf { x } } _ { j } ^ { k } , \hat { \mathbf { y } } _ { j } ^ { k } ) ;$ set $\mathrm { L a s t S e q } _ { i } [ j ] \gets \mathrm { S e q } _ { j } .$   
5: else   
6: Treat $j$ as missing at iteration k.   
7: end if   
8: end for   
9: Robust Mixing with Local Fallback: Compute $\mathbf { X } _ { i } ^ { k }$ and $\mathbf { Y } _ { i } ^ { k }$   
using the dynamic weights defined in Section III-A.   
10: AdamW Update: Update $\mathbf { m } _ { i } ^ { k + 1 }$ and $\mathbf { v } _ { i } ^ { k + 1 }$ using $\mathbf { Y } _ { i } ^ { k }$ as in $( 7 ) .$   
Compute the bias-corrected moments $\hat { \mathbf { m } } _ { i } ^ { k + 1 }$ and $\hat { \mathbf { v } } _ { i } ^ { k + 1 }$ . Update   
the local model according to (9).   
11: Gradient Tracking Update: Compute $\mathbf { g } _ { i } ^ { k + 1 } = \nabla f _ { i } ( \mathbf { x } _ { i } ^ { k + 1 } )$ and   
update the tracking variable according to (10).   
12: Compression with EF Strategy: For each $z \in \{ x , y \}$ , compute   
$\tilde { \mathbf { z } } _ { i } ^ { k + 1 } , \mathbf { c } _ { z , i } ^ { k + 1 }$ , and $\mathbf { e } _ { z , i } ^ { k + 1 }$ using the EF recursion.   
13: Broadcast: Increment $\mathrm { S e q } _ { i } ~  ~ \mathrm { S e q } _ { i } + 1$ . Broadcast packet   
$( \mathrm { S e q } _ { i } , \mathbf { c } _ { x , i } ^ { k + 1 } , \mathbf { c } _ { y , i } ^ { k + 1 } )$ to ${ \mathcal { N } } _ { i } .$   
14: Set $\mathbf { g } _ { i } ^ { k }  \mathbf { g } _ { i } ^ { k + 1 } .$

$$
\mathbf { y } _ { i } ^ { k + 1 } = \mathbf { Y } _ { i } ^ { k } + \left( \mathbf { g } _ { i } ^ { k + 1 } - \mathbf { g } _ { i } ^ { k } \right) .\tag{10}
$$

To reduce WCom overhead, both $\mathbf { x } _ { i } ^ { k + 1 }$ and $\mathbf { y } _ { i } ^ { k + 1 }$ are compressed using separate compressors $\mathcal { Q } _ { x }$ and $\mathcal { Q } _ { y }$ with EF [10]. For $z \in \{ x , y \}$

$$
\tilde { \mathbf { z } } _ { i } ^ { k + 1 } = \mathbf { z } _ { i } ^ { k + 1 } + \mathbf { e } _ { z , i } ^ { k } \quad \mathrm { ( A d d \ r e s i d u a l ) } ,
$$

$$
\mathbf { c } _ { z , i } ^ { k + 1 } = \mathcal { Q } _ { z } \left( \tilde { \mathbf { z } } _ { i } ^ { k + 1 } \right) \quad ( \mathrm { C o m p r e s s } ) ,\tag{11}
$$

(12)

$$
{ \bf e } _ { z , i } ^ { k + 1 } = \tilde { \bf z } _ { i } ^ { k + 1 } - { \bf c } _ { z , i } ^ { k + 1 } \quad ( \mathrm { U p d a t e ~ r e s i d u a l } ) .\tag{13}
$$

Finally, node i increments its local sequence counter and transmits $( \mathrm { S e q } _ { i } , \mathbf { c } _ { x , i } ^ { k + 1 } , \mathbf { c } _ { y , i } ^ { k + 1 } )$ to its neighbors. The overall perround routine is summarized in Algorithm 2.

## C. Convergence Analysis

Reweighted Objective and Notation: For the convergence analysis, we specialize to a constant stepsize $\alpha _ { k } \equiv \alpha > 0$ . Due to packet-dependent local fallback, $\mathbf { B } ^ { \bar { k } }$ is row-stochastic but generally not doubly stochastic. Hence, the uniform network average is no longer preserved, and the convergence dynamics are governed by the Perron weighting induced by the effective mixing matrix [11].

Let $\bar { \mathbf { B } } \triangleq \mathbb { E } [ \mathbf { B } ^ { k } ]$ and let $\pi \ \succ \ 0$ be the normalized left Perron vector [12] satisfying $\pi ^ { \top } \bar { \mathbf { B } } = \pi ^ { \top }$ , and $\pi ^ { \top } { \bf 1 } = 1$ Accordingly, we analyze convergence with respect to the reweighted objective

$$
F _ { \pi } ( \mathbf { x } ) \triangleq \sum _ { i = 1 } ^ { N } \pi _ { i } f _ { i } ( \mathbf { x } ) , \qquad \mathbf { x } _ { \pi } ^ { \star } \in \arg \operatorname* { m i n } F _ { \pi } ( \mathbf { x } ) ,\tag{14}
$$

and define $\begin{array} { r } { F _ { \pi } ^ { \star } \triangleq F _ { \pi } ( \mathbf { x } _ { \pi } ^ { \star } ) } \end{array}$ . The Perron weights characterize the long-term asymmetry induced by packet-dependent

mixing. In the doubly-stochastic case, $\begin{array} { r l r } { \pi } & { { } = } & { \frac { 1 } { N } { \bf 1 } } \end{array}$ and $F _ { \pi }$ reduces to the original objective (1). For a stacked vector $\mathbf { z } = \operatorname { c o l } ( \mathbf { z } _ { 1 } , \dots , \mathbf { z } _ { N } ) \in \mathbb { R } ^ { N d }$ , define

$$
\bar { \mathbf { z } } _ { \pi } \triangleq ( \pi ^ { \top } \otimes \mathbf { I } _ { d } ) \mathbf { z } , \qquad \mathbf { I I } _ { \pi } ^ { \perp } \triangleq \mathbf { I } _ { N d } - ( \mathbf { 1 } \pi ^ { \top } \otimes \mathbf { I } _ { d } ) .\tag{15}
$$

Thus, $\| \mathbf { I I } _ { \pi } ^ { \perp } \mathbf { z } \|$ measures model disagreement from Perron consensus. Define

$$
\Xi ^ { k } \triangleq \frac { 1 } { N } \mathbb { E } \left[ \| \mathbf { \Pi } \mathbf { \Pi } \mathbf { \Pi } _ { \pi } ^ { \perp } \mathbf { x } ^ { k } \| ^ { 2 } \right] , \qquad \bar { \mathbf { g } } _ { \pi } ^ { k } \triangleq \sum _ { i = 1 } ^ { N } \pi _ { i } \nabla f _ { i } ( \mathbf { x } _ { i } ^ { k } ) .\tag{16}
$$

Assumption 1 (Objective regularity [13]). Each $f _ { i }$ is convex and L-smooth, and $F _ { \pi }$ admits at least one minimizer.

Assumption 2 (Compression [6]). For $z \in \{ x , y \}$ , the compressor $\mathcal { Q } _ { z } : \mathbb { R } ^ { d } \overset { \sim } {  } \mathbb { R } ^ { d }$ satisfies

$$
\begin{array} { r } { \mathbb { E } \| \mathcal { Q } _ { z } ( \mathbf { v } ) - \mathbf { v } \| ^ { 2 } \leq ( 1 - \delta _ { z } ) \| \mathbf { v } \| ^ { 2 } , \qquad 0 < \delta _ { z } \leq 1 , } \end{array}\tag{17}
$$

for all $\mathbf { v } \in \mathbb { R } ^ { d }$ . For deterministic Top-K, $\delta _ { z } = K _ { z } / d ,$ where $K _ { z }$ denotes the number of retained coordinates for stream z.

Assumption 3 (Random mixing [12]). The matrices $\{ \mathbf { B } ^ { k } \}$ are i.i.d. and row-stochastic, B<sup>¯</sup> is primitive, and there exists $\rho _ { \pi } \in ( 0 , 1 ]$ such that

$$
\begin{array} { r } { \mathbb { E } \left[ \left\| \Pi _ { \pi } ^ { \perp } ( \mathbf B ^ { k } \otimes \mathbf I _ { d } ) \mathbf { z } \right\| ^ { 2 } \right] \leq \left( 1 - \rho _ { \pi } \right) \left\| \Pi _ { \pi } ^ { \perp } \mathbf { z } \right\| ^ { 2 } . } \end{array}\tag{18}
$$

The current packet-delivery randomness is independent of the algorithmic states and compression variables generated before the communication phase of round k.

Assumption 4 (Bounded states [14]). There exist finite constants $R _ { x } , R _ { y } , G _ { \infty } > 0$ such that

$$
\frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { E } \| \mathbf { z } _ { i } ^ { k } \| ^ { 2 } \leq R _ { z } ^ { 2 } , \quad z \in \{ x , y \} , \quad \| \mathbf { Y } _ { i } ^ { k } \| _ { \infty } \leq G _ { \infty } .\tag{19}
$$

For AdamW preconditioner $\begin{array} { r } { \mathbf { D } _ { i } ^ { k + 1 } \triangleq \mathrm { d i a g } \left( \frac { 1 } { \sqrt { \hat { \mathbf { v } } _ { i } ^ { k + 1 } + \varepsilon } } \right) } \end{array}$ where the operations inside the diagonal are element-wise, Assumption 4 implies

$$
h _ { \operatorname* { m i n } } \mathbf { I } _ { d } \preceq \mathbf { D } _ { i } ^ { k + 1 } \preceq h _ { \operatorname* { m a x } } \mathbf { I } _ { d } , \quad h _ { \operatorname* { m i n } } = \frac { 1 } { G _ { \infty } + \varepsilon } , \quad h _ { \operatorname* { m a x } } = \frac { 1 } { \varepsilon } .\tag{20}
$$

For $z \in \{ x , y \}$ , the EF recursion gives

$$
\begin{array} { r } { \hat { \mathbf { z } } _ { i } ^ { k } = \mathcal { Q } _ { z } \left( \mathbf { z } _ { i } ^ { k } + \mathbf { e } _ { z , i } ^ { k - 1 } \right) , \quad \quad \mathbf { e } _ { z , i } ^ { k } = \mathbf { z } _ { i } ^ { k } + \mathbf { e } _ { z , i } ^ { k - 1 } - \hat { \mathbf { z } } _ { i } ^ { k } . } \end{array}\tag{21}
$$

Here, $\hat { \mathbf { z } } _ { i } ^ { k } \equiv \mathbf { c } _ { z , i } ^ { k }$ <sub>i</sub> denotes the decoded compressor output.

Lemma 1 (Communication-error bounds). Under Assumptions 2–4,

$$
\frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { E } \| \mathbf { e } _ { z , i } ^ { k } \| ^ { 2 } \leq C _ { e , z } R _ { z } ^ { 2 } , \quad C _ { e , z } = \frac { 2 ( 1 - \delta _ { z } ) ( 2 - \delta _ { z } ) } { \delta _ { z } ^ { 2 } } ,\tag{22}
$$

for $0 < \delta _ { z } < 1$ , while $C _ { e , z } = 0$ for $\delta _ { z } = 1 .$ . Defining

$$
\boldsymbol { \chi } _ { z , i } ^ { k } \triangleq \hat { \mathbf { z } } _ { i } ^ { k } - \mathbf { z } _ { i } ^ { k } , \qquad \boldsymbol { \mathcal { C } } _ { z } ^ { k } \triangleq \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { E } \| \boldsymbol { \chi } _ { z , i } ^ { k } \| ^ { 2 } .\tag{23}
$$

Then,

$$
\begin{array} { r } { \mathcal { C } _ { z } ^ { k } \le 4 C _ { e , z } R _ { z } ^ { 2 } . } \end{array}\tag{24}
$$

Furthermore, define $\chi _ { z } ^ { k } \triangleq$ col $\left( \boldsymbol { \chi } _ { z , 1 } ^ { k } , \ldots , \boldsymbol { \chi } _ { z , N } ^ { k } \right)$ . Let $\mathbf { w } ^ { k } \triangleq$ $( \pmb { \pi } ^ { \top } \otimes \mathbf { I } _ { d } ) \mathbf { X } ^ { k }$ and $\mathbf { r } _ { x } ^ { k } \triangleq \mathbf { w } ^ { k } - \bar { \mathbf { x } } _ { \pi } ^ { k }$ . Then,

$$
\mathscr { R } _ { x } ^ { k } \triangleq \mathbb { E } \Vert \mathbf { r } _ { x } ^ { k } \Vert ^ { 2 } \leq 2 N \kappa _ { B } \Xi ^ { k } + 2 N \kappa _ { A } \mathcal { C } _ { x } ^ { k } ,\tag{25}
$$

where $\kappa _ { B } \triangleq \mathbb { E } \left. \pmb { \pi } ^ { \top } ( \mathbf { B } ^ { k } - \bar { \mathbf { B } } ) \right. _ { 2 } ^ { 2 }$ , and $\kappa _ { A } \triangleq \mathbb { E } \left\| \pi ^ { \top } \mathbf { A } ^ { k } \right\| _ { 2 } ^ { 2 }$ Proof. Using (17), Young’s inequality with $\begin{array} { r } { \tau _ { z } = \frac { 2 ( 1 - \delta _ { z } ) } { \delta _ { z } } } \end{array}$ gives $\mathbb { E } \| \mathbf { e } _ { z , i } ^ { k } \| ^ { 2 } \leq \left( 1 - \frac { \delta _ { z } } { 2 } \right) \mathbb { E } \| \mathbf { e } _ { z , i } ^ { k - 1 } \| ^ { 2 } + \frac { ( 1 - \delta _ { z } ) ( 2 - \delta _ { z } ) } { \delta _ { z } } \mathbb { E } \| \mathbf { z } _ { i } ^ { k } \| ^ { 2 } .$

(26)

Averaging over all nodes and unrolling the recursion gives (22). Since $\boldsymbol { \chi } _ { z , i } ^ { k } = \mathbf { e } _ { z , i } ^ { k - 1 } - \mathbf { e } _ { z , i } ^ { k } , ( 2 4 )$ follows.

Furthermore, from $\mathbf { \bar { X } } ^ { k } = ( \mathbf { B } ^ { k } \otimes \mathbf { I } _ { d } ) \mathbf { x } ^ { k } + ( \mathbf { A } ^ { k } \otimes \mathbf { I } _ { d } ) { \pmb { \chi } } _ { x } ^ { k }$ and $( \mathbf { B } ^ { k } - \bar { \mathbf { B } } ) \mathbf { 1 } = \mathbf { 0 }$ , we obtain

$$
\mathbf { r } _ { x } ^ { k } = \left( \pi ^ { \top } ( \mathbf { B } ^ { k } - { \bar { \mathbf { B } } } ) \otimes \mathbf { I } _ { d } \right) \mathbf { I } _ { \pi } ^ { \bot } \mathbf { x } ^ { k } + \left( \pi ^ { \top } \mathbf { A } ^ { k } \otimes \mathbf { I } _ { d } \right) { \boldsymbol { \chi } } _ { x } ^ { k } .\tag{27}
$$

Independence and $\| \mathbf { a } + \mathbf { b } \| ^ { 2 } \leq 2 \| \mathbf { a } \| ^ { 2 } + 2 \| \mathbf { b } \| ^ { 2 }$ then give (25). □

We next express the Perron-average algorithm as an inexact gradient method. Define $\begin{array} { r } { \bar { \mathbf { Y } } _ { \pi } ^ { k } \triangleq et { } { ' } \sum _ { i = 1 } ^ { N } \pi _ { i } \mathbf { Y } _ { i } ^ { k } , \bar { \mathbf { g } } _ { \pi } ^ { k } \triangleq } \end{array}$ $\begin{array} { r } { \sum _ { i = 1 } ^ { N } \bar { \boldsymbol { \pi } } _ { i } \nabla f _ { i } ( \mathbf { x } _ { i } ^ { k } ) , ~ \mathbf { t } ^ { k } \triangleq \bar { \mathbf { Y } } _ { \boldsymbol { \pi } } ^ { k } - \bar { \mathbf { g } } _ { \boldsymbol { \pi } } ^ { k } } \end{array}$ , and $\begin{array} { r } { \mathcal { T } ^ { k } \triangleq \mathbb { E } \Vert \mathbf { t } ^ { k } \Vert ^ { 2 } } \end{array}$ . The tracking-stream compression is included in $\mathcal { T } ^ { k }$ because $\mathbf { Y } _ { i } ^ { k }$ is the received post-mixing tracking state. Let

$$
\mathbf { a } ^ { k } \triangleq \sum _ { i = 1 } ^ { N } \pi _ { i } \left( \mathbf { D } _ { i } ^ { k + 1 } \hat { \mathbf { m } } _ { i } ^ { k + 1 } - \mathbf { Y } _ { i } ^ { k } \right) , \quad \mathcal { A } ^ { k } \triangleq \mathbb { E } \| \mathbf { a } ^ { k } \| ^ { 2 } .\tag{28}
$$

With $\begin{array} { r } { \mathcal { M } ^ { k } \triangleq \sum _ { i = 1 } ^ { N } \pi _ { i } \mathbb { E } \left\| \hat { \mathbf { m } } _ { i } ^ { k + 1 } - \mathbf { Y } _ { i } ^ { k } \right\| ^ { 2 } } \end{array}$ , bounded preconditioning gives

$$
\mathcal { A } ^ { k } \leq 2 h _ { \operatorname* { m a x } } ^ { 2 } \mathcal { M } ^ { k } + 2 h _ { \Delta } ^ { 2 } d G _ { \infty } ^ { 2 } , \qquad \mathcal { M } ^ { k } \leq 4 d G _ { \infty } ^ { 2 } ,\tag{29}
$$

where $h _ { \Delta } \triangleq \operatorname* { m a x } \left\{ | h _ { \operatorname* { m i n } } - 1 | , | h _ { \operatorname* { m a x } } - 1 | \right\}$

By L-smoothness,

$$
\begin{array} { r } { \mathbb { E } \left\| \bar { \mathbf { g } } _ { \pi } ^ { k } - \nabla F _ { \pi } \left( \bar { \mathbf { x } } _ { \pi } ^ { k } \right) \right\| ^ { 2 } \leq L ^ { 2 } N \pi _ { \operatorname* { m a x } } \Xi ^ { k } , } \end{array}\tag{30}
$$

where $\pi _ { \operatorname* { m a x } } \triangleq$ max $1 \le i \le N  \pi _ { i }$

Taking the Perron-weighted average of the AdamW update and using $\mathbf { w } ^ { k } = \bar { \mathbf { x } } _ { \pi } ^ { k } + \mathbf { r } _ { x } ^ { k }$ yields the exact recursion

$$
\begin{array} { r } { \bar { \mathbf { x } } _ { \pi } ^ { k + 1 } = \bar { \mathbf { x } } _ { \pi } ^ { k } - \alpha \left[ \nabla F _ { \pi } \left( \bar { \mathbf { x } } _ { \pi } ^ { k } \right) + \pmb { \xi } ^ { k } \right] , } \end{array}\tag{31}
$$

where $\begin{array} { r } { \pmb { \xi } ^ { k } = \mathbf { a } ^ { k } + \mathbf { t } ^ { k } + \bar { \mathbf { g } } _ { \pi } ^ { k } - \nabla F _ { \pi } \left( \bar { \mathbf { x } } _ { \pi } ^ { k } \right) + \lambda \mathbf { w } ^ { k } - \frac { 1 } { \alpha } \mathbf { r } _ { x } ^ { k } . } \end{array}$

Define $\mathcal { E } _ { k } \triangleq \mathbb { E } \Vert \pmb { \xi } ^ { k } \Vert ^ { 2 }$ , and $\mathcal { W } ^ { k } \triangleq \mathbb { E } \| \mathbf { w } ^ { k } \| ^ { 2 }$ . Combining (25), (29), and (30) gives

$$
\begin{array} { r l } & { \mathcal { E } _ { k } \leq 5 \mathcal { A } ^ { k } + 5 \mathcal { T } ^ { k } + 5 \lambda ^ { 2 } \mathcal { W } ^ { k } } \\ & { \qquad + \left( 5 L ^ { 2 } N \pi _ { \operatorname* { m a x } } + \frac { 1 0 N \kappa _ { B } } { \alpha ^ { 2 } } \right) \Xi ^ { k } + \frac { 1 0 N \kappa _ { A } } { \alpha ^ { 2 } } \mathcal { C } _ { x } ^ { k } . } \end{array}\tag{32}
$$

Theorem 1 (Convergence for smooth convex objectives). Suppose Assumptions 1–4 hold and $0 < \alpha \leq 1 / ( 2 L )$ ). Define

$$
\widetilde { \mathbf { x } } _ { \pi } ^ { K } \triangleq \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \bar { \mathbf { x } } _ { \pi } ^ { k } , \qquad \overline { { \mathcal { E } } } _ { K } \triangleq \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathcal { E } _ { k } .\tag{33}
$$

Then,

$$
\begin{array} { r l } & { \mathbb { E } \left[ F _ { \pi } \left( \widetilde { \mathbf { x } } _ { \pi } ^ { K } \right) - F _ { \pi } ^ { \star } \right] } \\ & { \quad \leq \frac { \mathbb { E } \left\| \bar { \mathbf { x } } _ { \pi } ^ { 0 } - { \mathbf { x } } _ { \pi } ^ { \star } \right\| ^ { 2 } } { 2 \alpha K } + R _ { \star } \sqrt { \overline { { \mathcal { E } } } _ { K } } + \frac { \alpha ( 1 + 2 L \alpha ) } { 2 } \overline { { \mathcal { E } } } _ { K } , } \end{array}\tag{34}
$$

where Assumption 4 ensures E $\left\| \bar { \mathbf { x } } _ { \pi } ^ { k } - \mathbf { x } _ { \pi } ^ { \star } \right\| ^ { 2 } \leq R _ { \star } ^ { 2 }$ , with $R _ { \star } ^ { 2 } \triangleq$ $2 N \pi _ { \operatorname* { m a x } } R _ { x } ^ { 2 } + 2 \left\| \mathbf { x } _ { \pi } ^ { \star } \right\| ^ { 2 }$ . Thus, when the perturbations vanish, the standard $\mathcal { O } ( 1 / K )$ convex rate is recovered.

Proof. Let ${ \bf x } ^ { k } = \bar { \bf x } _ { \pi } ^ { k } , { \bf x } ^ { \star } = { \bf x } _ { \pi } ^ { \star }$ , and $\mathbf { g } ^ { k } = \nabla F _ { \pi } ( \mathbf { x } ^ { k } )$ . Using (31), we have

$$
\begin{array} { r } { \left\| { \mathbf { x } } ^ { k + 1 } - { \mathbf { x } } ^ { \star } \right\| ^ { 2 } = \left\| { \mathbf { x } } ^ { k } - { \mathbf { x } } ^ { \star } \right\| ^ { 2 } - 2 \alpha \left. { \mathbf { x } } ^ { k } - { \mathbf { x } } ^ { \star } , { \mathbf { g } } ^ { k } \right. } \\ { - 2 \alpha \left. { \mathbf { x } } ^ { k } - { \mathbf { x } } ^ { \star } , { \pmb { \xi } } ^ { k } \right. + \alpha ^ { 2 } \left\| { \mathbf { g } } ^ { k } + { \pmb { \xi } } ^ { k } \right\| ^ { 2 } . } \end{array}\tag{35}
$$

For a convex L-smooth function,

$$
\langle \mathbf { g } ^ { k } , \mathbf { x } ^ { k } - \mathbf { x } ^ { \star } \rangle \geq F _ { \pi } ( \mathbf { x } ^ { k } ) - F _ { \pi } ^ { \star } + \frac { 1 } { 2 L } \| \mathbf { g } ^ { k } \| ^ { 2 } .\tag{36}
$$

Using Young’s inequality and $\alpha \leq 1 / ( 2 L )$ therefore gives

$$
\begin{array} { r } { 2 \alpha \left[ F _ { \pi } ( \mathbf { x } ^ { k } ) - F _ { \pi } ^ { \star } \right] \leq \left\| \mathbf { x } ^ { k } - \mathbf { x } ^ { \star } \right\| ^ { 2 } - \left\| \mathbf { x } ^ { k + 1 } - \mathbf { x } ^ { \star } \right\| ^ { 2 } } \\ { + \ : 2 \alpha \left\| \mathbf { x } ^ { k } - \mathbf { x } ^ { \star } \right\| \| \pmb { \xi } ^ { k } \| + \alpha ^ { 2 } ( 1 + 2 L \alpha ) \| \pmb { \xi } ^ { k } \| ^ { 2 } . } \end{array}\tag{37}
$$

Taking expectations and applying Cauchy–Schwarz gives

$$
\mathbb { E } \left[ \left\| \mathbf { x } ^ { k } - \mathbf { x } ^ { \star } \right\| \| \pmb { \xi } ^ { k } \| \right] \leq R _ { \star } \sqrt { \mathcal { E } _ { k } } .\tag{38}
$$

Summing over $k = 0 , \ldots , K - 1$ , using $\begin{array} { r } { \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \sqrt { \mathcal { E } _ { k } } \le } \end{array}$ $\sqrt { \overline { { \mathcal { E } } } _ { K } }$ , and applying convexity of $F _ { \pi }$ proves (34). □

The next result characterizes the stronger regime in which the aggregate objective satisfies gradient dominance.

Corollary 1 (Linear rate under the PL condition). Under the conditions of Theorem $^ { l , }$ assume that for some $\mu _ { \mathrm { { P L } } } > 0$

$$
\frac { 1 } { 2 } \| \nabla F _ { \pi } ( \mathbf { x } ) \| ^ { 2 } \geq \mu _ { \mathrm { P L } } \left[ F _ { \pi } ( \mathbf { x } ) - F _ { \pi } ^ { \star } \right]\tag{39}
$$

over the region visited by the iterates, and let $0 < \alpha \leq 1 / ( 4 L )$ Define $\Delta _ { k } \triangleq \mathbb { E } \left[ F _ { \pi } \left( \bar { \mathbf { x } } _ { \pi } ^ { k } \right) - F _ { \pi } ^ { \star } \right]$ . Then,

$$
\Delta _ { k + 1 } \leq \rho \Delta _ { k } + \frac { 5 \alpha } { 4 } \mathscr { E } _ { k } , \qquad \rho \triangleq 1 - \alpha \mu _ { \mathrm { P L } } < 1 .\tag{40}
$$

Consequently,

$$
\Delta _ { K } \leq \rho ^ { K } \Delta _ { 0 } + \frac { 5 \alpha } { 4 } \sum _ { k = 0 } ^ { K - 1 } \rho ^ { K - 1 - k } \mathscr { E } _ { k } .\tag{41}
$$

$\begin{array} { r } { I f \mathcal { E } _ { k } \le \bar { \mathcal { E } } , } \end{array}$ , then lim sup $\begin{array} { r } { \kappa \infty \Delta _ { K } \leq \frac { 5 \bar { \mathcal { E } } } { 4 \mu _ { \mathrm { P L } } } . \ I f \mathcal { E } _ { k } \to 0 , } \end{array}$ , exact convergence follows. $I f \mathcal { E } _ { k } = \mathcal { O } ( \gamma ^ { k } ) f o r$ some $0 < \gamma < 1$ with $\gamma \neq \rho ,$ the convergence is linear with rate max $\{ \rho , \gamma \}$ . For $\gamma = \rho , \Delta _ { K } = \mathcal { O } ( K \rho ^ { K } )$

Proof. Let $\mathbf { x } ^ { k } = \bar { \mathbf { x } } _ { \pi } ^ { k }$ and $\mathbf { g } ^ { k } = \nabla F _ { \pi } ( \mathbf { x } ^ { k } )$ . By L-smoothness and (31),

$$
F _ { \pi } ( \mathbf { x } ^ { k + 1 } ) \leq F _ { \pi } ( \mathbf { x } ^ { k } ) - \alpha \left. \mathbf { g } ^ { k } , \mathbf { g } ^ { k } + \pmb { \xi } ^ { k } \right. + \frac { L \alpha ^ { 2 } } { 2 } \left\| \mathbf { g } ^ { k } + \pmb { \xi } ^ { k } \right\| ^ { 2 } .
$$

Using $- \left. \mathbf { g } ^ { k } , \pmb { \xi } ^ { k } \right. \leq \frac { 1 } { 4 } \| \mathbf { g } ^ { k } \| ^ { 2 } + \| \pmb { \xi } ^ { k } \| ^ { 2 }$ and $\left\| \mathbf { g } ^ { k } + \pmb { \xi } ^ { k } \right\| ^ { 2 } \leq$ $2 \| \mathbf { g } ^ { k } \| ^ { 2 } + 2 \| \pmb { \xi } ^ { k } \| ^ { 2 }$ , the condition $\alpha \leq 1 / ( 4 L )$ gives

$$
F _ { \pi } ( \mathbf { x } ^ { k + 1 } ) \leq F _ { \pi } ( \mathbf { x } ^ { k } ) - \frac { \alpha } { 2 } \| \mathbf { g } ^ { k } \| ^ { 2 } + \frac { 5 \alpha } { 4 } \| \pmb { \xi } ^ { k } \| ^ { 2 } .\tag{43}
$$

Applying (39) and taking expectations yields (40). Unrolling the recursion gives the remaining claims. □

Remark 1. Theorem 1 applies directly to smooth convex objectives, including the logistic regression setting used in our experiments. The PL condition is only an additional sufficient condition for the stronger linear-rate result and is not required by the general convex analysis. Model-stream compression enters through ${ \mathcal { C } } _ { x } ^ { k } ,$ , while tracking-stream compression is captured by $\mathcal { T } ^ { k }$

## IV. EXPERIMENT RESULTS AND DISCUSSION

We simulate a decentralized wireless network with $N = 1 5$ nodes deployed in $\mathrm { ~ a ~ } 2 0 0 0 \times 2 0 0 0 ~ \mathrm { m } ^ { 2 }$ area via Poisson-disk sampling with $d _ { \operatorname* { m i n } } = 2 5 0$ m. Directed links are formed within $r _ { \operatorname* { m a x } } = 7 5 0$ m. Channel parameters follow [15], with default transmit power $P _ { i } = 0 . 2 \ : \mathrm { W }$ . To evaluate the proposed method in the smooth convex setting of Theorem 1, we use a logistic regression model with Top-K biased compression.

![](images/3cdd01cafd13830a96e54f1ff119e82cde317138365c2f43f7ad11991ee14825.jpg)  
Fig. 2: Non-IID partition of MNIST and CIFAR10 dataset

## A. Datasets and Model

We evaluate MNIST [16] and CIFAR-10 [17] using 50,000 training samples and the standard test sets over 1000 epochs. Non-IID data are split across N local datasets $\{ \mathcal { D } _ { i } \} _ { i = 1 } ^ { N }$ , as shown in Fig. 2: MNIST uses a pathological label-skew split with 2–5 labels per node [18], while CIFAR-10 uses Dirichlet partitioning with concentration $\alpha = 0 . 5 \ [ 1 9 ]$

For comparison purposes, we also consider three representative DecL baselines: (1) CHOCO-SGD [5], a compressed decentralized SGD method with quantized communication; (2) GT-AdamW, which integrates classical GT with AdamW; and (3) QGT-AdamW, a quantized GT with AdamW. Training hyperparameters are as follows: for AdamW-based methods, the learning rate η is set to 0.005 (MNIST) and 0.001 (CIFAR-10) with $\beta _ { m } ~ = ~ 0 . 9$ and $\beta _ { v } ~ = ~ 0 . 9 9 9$ ; for CHOCO-SGD, $\eta = 0 . 0 0 5$ (MNIST), η = 0.001 (CIFAR-10) and consensus step-size $\gamma ~ = ~ 0 . 0 0 1$ for both datasets. All methods use a weight decay of 0.01. Communication compression uses Top-K sparsification retaining 10% of the coordinates, achieving 90% message reduction. Performance is evaluated using three metrics: (1) test accuracy, (2) global training loss, and (3) packet drop rate.

![](images/fd3fceb0770e48cb22b664f8d1954d63515251b536d722c5c86d512a88e58c48.jpg)

![](images/18f78fac6a24d808523fac209e1ff5de49ac9224120cc6cd58aede02c9e68414.jpg)

![](images/7c8795935b02162a851ada4e069d609acd93338de4320e51dc3c229c7690b1f9.jpg)

![](images/6a5492e427462a464c954540e185b54e1e4eab76fb7d14e77d30602092962316.jpg)

![](images/8da30b748e06fadcf82248a3f5eb84c6f1562ae4b3de01b9e618de0cb5df6047.jpg)

![](images/86ed3964a10aa69339ad979a2d28825a97e50a2e9e1c474b822dae9cb9290156.jpg)  
Fig. 3: Effect of Top-K density on test accuracy and training loss, with convergence curves for MNIST (first row) and CIFAR-10 (second row).

![](images/fc43d859626f617d7da259433315e075c9e2e92d2e5c38c547a3716483c1feb9.jpg)  
(a) Mean accuracy (MNIST)

![](images/824340f1ccc52d8a1c1e410185dc0384a66e361fd583ae03917b6ae8af8ca7e1.jpg)  
(b) Mean accuracy (CIFAR-10)

Average packet drop rate (%) vs Bandwidth (MHz)  
![](images/e6dff3986bbd800493b3d882b807757f524070e1e9b8eb84cb352d44835be06a.jpg)  
(c) Average packet drop rate versus bandwidth

Fig. 4: Final average accuracy and average packet drop rate versus bandwidth (MHz) under wireless deadlines and outage-based drops  
![](images/9edbbd6de578ef20eb81922c0b8e0e2325bcf14a3dd81c477a4a5f71f22a82cd.jpg)  
(a) Mean accuracy (MNIST)

![](images/6541682832509ca05e869c68ea32867f9b6c99a755e9a96c48ed723b6cb180a2.jpg)  
(b) Mean accuracy (CIFAR-10)

![](images/7dd57e9fe543d126c6d75dd5f716c273bc9cc1b0ec421ac2d5f6b166259810d2.jpg)  
(c) Average packet drop rate versus power  
Fig. 5: Final average accuracy and average packet drop rate versus transmission power (Watt) under wireless deadlines and outage-based drops.

## B. Results and Discussion

Fig. 3 illustrates the impact of quantization and EF on model performance. In this experiment, wireless impairments are disabled, and perfect communication is assumed at every iteration to isolate the effect of quantization only. Under this controlled setting, GT-AdamW serves as a full-precision reference because it does not incur quantization error. The results show that QEF-GT-AdamW consistently outperforms the baseline QGT-AdamW across all Top-K sparsifications. Moreover, QEF-GT-AdamW approaches the full-precision reference accuracy at Top-K sparsifications of approximately 30% for MNIST and 35% for CIFAR-10. In contrast, the baseline QGT-AdamW requires a substantially larger Top-K sparsification (around 90%) to achieve comparable accuracy. We further visualize the average training loss across all nodes for different Top-K sparsifications. For both datasets,

QEF-GT-AdamW yields uniformly lower loss than QGT-AdamW at the same Top-K sparsification, confirming that EF effectively compensates quantization bias and improves optimization quality.

Figs. 4 and 5 further explain the accuracy trends by reporting the average packet drop rate during training under bandwidth-limited and power-limited WCom, respectively, with the Top-K sparsification fixed at 10%. As can be seen, CHOCO-SGD consistently exhibits an almost zero packet drop rate over all settings because it exchanges only the model variable, resulting in a much smaller packet size. However, despite this highly reliable communication behavior, its learning accuracy remains significantly lower, since it does not explicitly mitigate client drift under non-IID data. By contrast, GT-based methods must exchange both the model and the gradient-tracking variables, which substantially enlarges the payload and makes them much more vulnerable to deadline violations and wireless outages. This effect is most severe for GT-AdamW, which does not use quantization. As shown in Fig. 4, its average packet drop rate is approximately 100% at 0.05, 0.1, and 0.2 MHz, remains very high at about 83% even at 0.5 MHz, and is still around 28% at 1 MHz. Only when the bandwidth increases to 2 MHz and above does the drop rate become small (about 4% at 2 MHz and nearly 0% at 10 MHz). A similar trend is observed in Fig. 5: GT-AdamW experiences about 96%, 91%, 82%, and 63% packet drops at 0.005, 0.01, 0.02, and 0.05 W, respectively, and its drop rate decreases gradually only at higher power levels (about 28% at 0.2 W, 9% at 0.75 W, and 4% at 2 W). These statistics explain the strong performance degradation of unquantized GT-AdamW in bandwidth- or power-limited regimes.

In contrast, the quantized GT variants, QGT-AdamW and QEF-GT-AdamW, achieve dramatically lower packet drop rates because both the model and tracking streams are compressed before transmission. In Fig. 4, their drop rate is about 67% at 0.05 MHz, decreases to roughly 10% at 0.1 MHz, and becomes essentially zero from 0.2 MHz onward. In Fig. 5, their drop rate is only about 8% at 0.005 W, around 1% at 0.01 W, and nearly zero for 0.02 W and above. Since QGT-AdamW and QEF-GT-AdamW use the same Top-K sparsification, they exhibit nearly identical drop-rate curves, which is expected because their transmitted packet sizes are almost the same. Therefore, the accuracy gain of QEF-GT-AdamW over QGT-AdamW does not come from a lower transmission failure probability, but from the optimization benefit of EF in compensating compression distortion across rounds. Overall, these results confirm that QEF-GT-AdamW achieves a more favorable trade-off among communication efficiency, robustness to wireless impairments, and learning effectiveness under heterogeneous data.

## V. CONCLUSION

In this paper, we propose QEF-GT-AdamW, a resilient and communication-efficient DecL algorithm for unreliable wireless edge networks. It combines AdamW, gradient tracking, and dual biased quantization-based EF to minimize payloads and stabilize optimization against non-IID data. Experiments on MNIST and CIFAR-10 show that QEF-GT-AdamW outperforms representative baselines in convergence speed and robustness, particularly in resource-limited environments.

## ACKNOWLEDGMENT

This work was supported by the Luxembourg National Research Fund (FNR) through the CHIST-ERA SHIELD project (Grant IN-TER/CHIST24/19023763/SHIELD) and the FNR AFR program via the FULFIL project (Grant 2000141).

## REFERENCES

[1] E. T. M. Beltran, M. Q. P´ erez, P. M. S. S´ anchez, S. L. Bernal,´ G. Bovet, M. G. Perez, G. M. P´ erez, and A. H. Celdr´ an, “Decentralized´ federated learning: Fundamentals, state of the art, frameworks, trends, and challenges,” IEEE Communications Surveys & Tutorials, vol. 25, no. 4, pp. 2983–3013, 2023.

[2] L. Chen, W. Liu, Y. Chen, and W. Wang, “Communication-efficient design for quantized decentralized federated learning,” IEEE Trans. Signal Process., vol. 72, pp. 1175–1188, 2024.

[3] S. Pu and A. Nedic, “Distributed stochastic gradient tracking methods,”´ Mathematical Programming, vol. 187, no. 1, pp. 409–457, 2021.

[4] G. Carnevale, F. Farina, I. Notarnicola, and G. Notarstefano, “GTAdam: Gradient tracking with adaptive momentum for distributed online optimization,” IEEE Trans. Control Netw. Syst., vol. 10, no. 3, 2022.

[5] A. Koloskova, S. Stich, and M. Jaggi, “Decentralized stochastic optimization and gossip algorithms with compressed communication,” in Int. Conf. Machine Learn. (ICML). PMLR, 2019, pp. 3478–3487.

[6] P. Richtarik, I. Sokolov, and I. Fatkhullin, “EF21: A new, simpler,´ theoretically better, and practically faster error feedback,” Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 34, pp. 4384–4396, 2021.

[7] P. Zhou, X. Xie, Z. Lin, and S. Yan, “Towards understanding convergence and generalization of adamw,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 9, pp. 6486–6493, 2024.

[8] J. Burkardt, “The truncated normal distribution,” Department Sci. Comput. Website, Florida State University, vol. 1, no. 35, p. 58, 2014.

[9] M. K. Simon and M.-S. Alouini, Digital communication over fading channels. John Wiley & Sons, 2004.

[10] S. P. Karimireddy, Q. Rebjock, S. Stich, and M. Jaggi, “Error feedback fixes signsgd and other gradient compression schemes,” in Int. Conf. Machine Learn. (ICML). PMLR, 2019, pp. 3252–3261.

[11] A. Nedic and A. Olshevsky, “Distributed optimization over time-varying´ directed graphs,” IEEE Transactions on Automatic Control, vol. 60, no. 3, pp. 601–615, 2014.

[12] S. Pu, W. Shi, J. Xu, and A. Nedic, “Push–pull gradient methods for´ distributed optimization in networks,” IEEE Trans. Automatic Control, vol. 66, no. 1, pp. 1–16, 2020.

[13] A. Koloskova, T. Lin, and S. U. Stich, “An improved analysis of gradient tracking for decentralized machine learning,” Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 34, pp. 11 422–11 435, 2021.

[14] A. Beznosikov, P. Richtarik, M. Diskin, M. Ryabinin, and A. Gasnikov,´ “Distributed methods with compressed communication for solving variational inequalities, with theoretical guarantees,” Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 35, pp. 14 013–14 029, 2022.

[15] T. T. Nguyen, L. B. Le, and Q. Le-Trung, “Computation offloading in MIMO based mobile edge computing systems under perfect and imperfect CSI estimation,” IEEE Trans. Services Comput., vol. 14, no. 6, pp. 2011–2025, 2019.

[16] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceed. IEEE, vol. 86, no. 11, pp. 2278–2324, 2002.

[17] A. Krizhevsky, G. Hinton et al., “Learning multiple layers of features from tiny images,” 2009.

[18] Y. Huang, L. Chu, Z. Zhou, L. Wang, J. Liu, J. Pei, and Y. Zhang, “Personalized cross-silo federated learning on non-IID data,” in Proceed. AAAI Conf. Artificial Intell., vol. 35, no. 9, 2021, pp. 7865–7873.

[19] L. Gao, H. Fu, L. Li, Y. Chen, M. Xu, and C.-Z. Xu, “Feddc: Federated learning with non-iid data via local drift decoupling and correction,” in Proceed. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 10 112–10 121.