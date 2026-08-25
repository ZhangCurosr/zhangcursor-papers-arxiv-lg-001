# A Momentum-Based Variance-Reduced Algorithm for Federated Multiobjective Optimization

Yong Zhao\*† Chunlin You† Minh N. Dao, and Zai-Yun Peng

August 24, 2026

## Abstract

Federated learning has traditionally been formulated as a single-objective optimization problem, primarily focused on maximizing model utility. In real-world applications, however, machine learning models often need to optimize multiple and potentially conflicting objectives simultaneously. This motivates federated multiobjective optimization (FMOO), which provides a natural framework for jointly handling multiple task-specific objectives in federated learning. In this paper, we propose a momentum-based variance-reduced algorithm for federated multiobjective optimization. The method incorporates a momentum-driven gradient estimator into the local updates to reduce the variance of stochastic updates, leading to an improved convergence rate. We establish theoretical guarantees showing that the expected Pareto stationarity measure of a randomly selected output iterate decays at a rate of $\mathcal { O } ( T ^ { - 2 / 3 } )$ , improving upon the $\mathcal { O } ( T ^ { - 1 / 2 } )$ rates established for existing methods such as FSMGDA and FedCMOO. Numerical experiments on federated multiobjective optimization benchmarks demonstrate the effectiveness and competitive performance of the proposed algorithm.

Keywords: Federated multiobjective optimization, nonconvex optimization, momentum-based variance reduction, communication complexity, stochastic multi-gradient methods. Mathematics Subject Classification (MSC 2020): 90C29, 90C15, 90C26.

## 1. Introduction

In recent years, multiobjective optimization (MOO) has emerged as a foundational paradigm for multi-agent and multi-task learning, with broad applications spanning multi-task neural network training [1], recommender systems [2], learning-to-rank tasks [3, 4], and economic or transportation scheduling [5-7]. Unlike single-objective optimization, MOO seeks to simultaneously optimize multiple interrelated and often conflicting objectives. Consequently, it is typically impossible to find a single solution that minimizes all objectives at once. Therefore, MOO algorithms instead aim to find a Pareto optimal/stationary solution, where any further improvement in one loss necessarily worsens another. Existing MOO algorithms are predominantly designed for centralized learning settings, where data from all participants must be aggregated on a central server for joint optimization. This centralized paradigm not only limits scalability in distributed multi-agent systems but also conflicts with increasingly stringent privacy and data localization requirements, particularly in sensitive domains such as healthcare and finance.

Federated learning (FL) has emerged as a transformative distributed machine learning paradigm that enables multiple participants to collaboratively train models without sharing raw data, thereby addressing critical challenges related to data privacy and data silos [8,9]. Since the seminal work by McMahan et al. [8], which introduced FedAvg as the first representative FL algorithm, substantial research efforts have been devoted to understanding and extending this approach. FedAvg follows a client-server architecture: in each communication round, participating clients perform several steps of local stochastic gradient descent (SGD) on their private datasets, after which the server aggregates the uploaded local model parameters via weighted averaging to update the global model. Under standard assumptions, it has been shown that FedAvg achieves a sublinear convergence rate when decaying learning rates are employed [10, 11]. However, despite SGD's statistically optimal convergence rate, its performance often depends on careful tuning of the learning rate, and this sensitivity is also inherited by FedAvg. To address these challenges, a wide range of methodological innovations have been proposed. Regularization-based methods [12-16] modify the local objective by introducing a proximal term that penalizes deviations from the global model, thereby promoting consistency across clients and mitigating client drift. Laplacian regularization is also employed in federated multitask learning to exploit relationships among client models [17]. Momentumbased techniques [18-21] accumulate historical gradient information to correct update directions, accelerating convergence and reducing bias induced by non-IID data. In response to difficulties in learning rate tuning and unstable local optimization, recent studies [22] have integrated adaptive optimization algorithms—such as Yogi [23], Adagrad [24] and Adam —into the federated setting. These methods enable automatic adjustment of step sizes based on gradient statistics at both the local and global levels, providing convergence guarantees under suitable assumptions for both IID and non-IID settings. Variance reduction (VR) techniques [25–30] further improve convergence stability by employing control variates or historical global gradient corrections to mitigate the client drift and variance caused by data heterogeneity. For more research progress on federated learning algorithms, please refer to [31]

Despite a rich body of literature on federated learning algorithms, most existing studies remain limited to single-objective optimization, where all clients jointly minimize one global loss function. However, in real-world applications, machine learning models often need to optimize multiple objectives simultaneously—such as improving model accuracy, and reducing training costs—which frequently conflict with one another. It is in this context that federated multiobjective optimization (FMOO) has emerged. By integrating the principles of federated learning and multiobjective optimization, it aims to achieve the synergistic optimization of multiple learning goals while preserving data privacy, thereby meeting the diverse demands of practical applications. Yang et al. [32] propose two federated multiobjective optimization (FMOO) algorithms called federated multi-gradient descent averaging (FMGDA) and federated stochastic multi-gradient descent averaging (FSMGDA) to solve FMOO, and establish the convergence of the two proposed algorithms under suitable conditions. Askin et al. [33] introduce FedCMOO, a communication-efficient federated multiobjective optimization algorithm that improves convergence performance without scaling communication costs with the number of objectives. Furthermore, the effectiveness and advantages of the proposed method are empirically validated through extensive comparisons with FSMGDA. Hartmann et al. [34] present the first comprehensive survey on the integration of multiobjective methods with federated learning, introducing a novel taxonomy to systematically classify existing works while offering insights into recent trends, open challenges, and future research directions.

Inspired by the work in [29–35], we propose FSMGDA-M-VR, a momentum-based variancereduced algorithm for federated multiobjective optimization. The design of our method is motivated by two related but previously separate research directions. The first is momentum-based variance reduction in federated learning, where methods such as STEM [30] improve the quality of stochastic descent directions and mitigate the effect of client-side sampling noise. The second is variance-reduced stochastic multiobjective optimization, where recent methods such as [35] show that tracking or momentum-type estimators can effectively reduce the bias and variance in multigradient construction. However, existing federated variance-reduction methods are mainly designed for single-objective optimization, while existing variance-reduced multiobjective methods are typically developed in centralized environments. Thus, they do not directly address the combined challenges of multiple conflicting objectives, heterogeneous clients, local stochastic updates, and server-client communication.

FSMGDA-M-VR addresses these challenges by reorganizing how objective-wise descent information and Pareto weights are computed in federated multiobjective training. Unlike FSMGDA [32], which aggregates accumulated stochastic gradients without explicit variance reduction, FSMGDA-M-VR constructs objective-wise momentum variance-reduced directions at each client. This reduces the stochastic error accumulated across local steps and provides more stable objective-wise information for server aggregation. It also differs from FedCMOO [33]. In FedCMOO, the server first approximates the Gram matrix of the objective gradients, updates the task weights through projected gradient steps, and then sends these weights to clients for weighted local updates. In contrast, FSMGDA-M-VR does not compute the Pareto weights before local training. Instead, clients first maintain objective-wise variance-reduced directions, and the server computes the Pareto balancing weights only after aggregating these directions. Therefore, the common descent direction is constructed from variance-reduced multiobjective information rather than from a pre-local-update weight vector.

This design leads to a clear communication-convergence trade-off. FedCMOO enjoys the advantage that its per-round communication cost can be independent of the number of objectives, whereas FSMGDA-M-VR requires each client to transmit one direction per objective and hence its per-round communication cost scales linearly with the number of objectives. The benefit is that the server receives richer objective-wise and variance-reduced information, which helps construct a more stable Pareto common descent direction. In this sense, FSMGDA-M-VR can be viewed as a federated multiobjective adaptation of momentum-based variance reduction, where objectivewise local momentum tracking, federated aggregation, and adaptive Pareto direction construction are integrated into a single training protocol. The main technical contribution lies in the convergence analysis of this scheme under federated multiobjective stochastic updates, showing that the estimator error can be controlled across local steps and communication rounds. Furthermore, we establish a convergence rate of $\mathcal { O } ( T ^ { - 2 / 3 } )$ , improving upon the $\mathcal { O } ( T ^ { - 1 / 2 } )$ rates of FSMGDA [32] and FedCMOO [33]. Extensive experiments further demonstrate the competitive performance of FSMGDA-M-VR over existing federated multiobjective optimization methods.

## 2. Preliminaries

FMOO aims at optimizing multiple objectives simultaneously:

$$
\operatorname* { m i n } _ { x \in \mathbb { R } ^ { d } } F ( x ) : = \left( f _ { 1 } ( x ) , . . . , f _ { S } ( x ) \right) ,\tag{2.1}
$$

where $x \in \mathbb { R } ^ { d }$ is the model parameter, and each $f _ { s }$ represents the objective function of task $s \in$ $[ S ] : = \{ 1 , \dots , S \}$ , defined by

$$
f _ { s } ( x ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } f _ { s , i } ( x ) ,
$$

with

$$
f _ { s , i } ( \boldsymbol { x } ) = \mathbb { E } _ { \xi _ { i } \sim \mathcal { D } _ { i } } [ F _ { s } ( \boldsymbol { x } ; \xi _ { i } ) ] ,
$$

Here, M denotes the number of clients, and the random variable $\xi _ { i }$ represents a local data point available at client i. We consider a synchronous full-participation federated setting, where all clients participate in each communication round with equal weights. Client dropout, communication compression, and quantization are not considered in the present analysis.

Unlike single-objective optimization, in MOO there generally does not exist a single solution that simultaneously minimizes all objectives. Therefore, a more appropriate notion of optimality in MOO is Pareto optimality, which is formally defined as follows.

Definition 2.1 (Pareto optimality). For any two solutions x, $\boldsymbol { y } \in \mathbb { R } ^ { d }$ , we say x dominates y if and only if $f _ { s } ( x ) \le f _ { s } ( y ) , \forall s \in [ S ]$ and $f _ { s } ( x ) < f _ { s } ( y ) , \exists s \in [ S ] . \mathrm { ~ A ~ }$ solution x $\in \mathbb { R } ^ { d }$ is Pareto optimal if it is not dominated by any other solution. A solution x $\in \mathbb { R } ^ { d }$ is weakly Pareto optimal if there does not exist a solution y such that $f _ { s } ( y ) < f _ { s } ( x ) , \forall s \in [ S ]$

Similar to single-objective non-convex optimization, finding a Pareto optimal solution in MOO can be computationally challenging, particularly in non-convex settings. Therefore, Pareto stationarity is commonly adopted as a practical first-order optimality criterion

Definition 2.2 (Pareto stationarity). A solution $\boldsymbol { x } \in \mathbb { R } ^ { d }$ is said to be Pareto stationary if there is no common descent direction $d \in \mathbb { R } ^ { d }$ such that $\nabla f _ { s } ( x ) ^ { \top } d < 0 , \forall s \in [ S ]$

Gradient-based MOO algorithms typically seek a common descent direction $d \in \mathbb { R } ^ { d }$ such that $\nabla f _ { s } ( x ) ^ { \top } d < 0$ for all $s \in [ S ]$ . If no such common descent direction exists at a given point, then according to Definition 2.2, x is a Pareto stationary solution. To determine such a direction, the multigradient method constructs the gradient matrix $G ( x ) : = [ \nabla f _ { 1 } ( x ) , \ldots , \nabla f _ { S } ( x ) ] \in \mathbb { R } ^ { d \times S }$ and identifies the optimal weight $\lambda ^ { * }$ by solving the quadratic optimization problem $\begin{array} { r } { \lambda ^ { * } ( x ) \in \arg \operatorname* { m i n } _ { \lambda \in \mathcal { C } } \| G ( x ) \lambda \| ^ { 2 } } \end{array}$ where $\begin{array} { r } { \mathcal { C } : = \{ \lambda \in [ 0 , 1 ] ^ { S } : \sum _ { s \in [ S ] } \lambda _ { s } = 1 \} } \end{array}$ . It then defines the multi-gradient direction as $\begin{array} { r } { g = G ( \boldsymbol { x } ) \lambda ^ { * } = \sum _ { \boldsymbol { s } \in [ S ] } \lambda _ { \boldsymbol { s } } ^ { * } \nabla f _ { \boldsymbol { s } } ( \boldsymbol { x } ) . \mathrm { ~ I f ~ } \dot { \lVert \boldsymbol { g } \rVert } = 0 . } \end{array}$ then x is a Pareto stationary solution; otherwise, $d = - g$ is a common descent direction, and the iteration is performed according to $x  x + \eta d ,$ or equivalently, $x  x - \eta g$ , where η denotes the learning rate, until a Pareto stationary solution is reached. The SMGD method follows a similar procedure to MGD, with the main difference being that the exact gradient matrix $G ( x )$ is replaced by its stochastic counterpart. Thus, $\| g \| ^ { 2 } = \| G ( x ) \lambda ^ { * } \| ^ { 2 }$ can serve as a stationarity measure for assessing the convergence of non-convex MOO algorithms.

Definition 2.3 (e-Pareto stationarity [36]). An iterate $x ^ { t }$ generated by a stochastic algorithm is said to be an e-Pareto stationary solution in expectation if

$$
\mathbb { E } \left[ \mathcal { G } ( x ^ { t } ) \right] \leq \epsilon ,
$$

where

$$
\mathcal { G } ( \boldsymbol { x } ) : = \operatorname* { m i n } _ { \lambda \in \mathcal { C } } \left\| \sum _ { s = 1 } ^ { S } \lambda _ { s } \nabla f _ { s } ( \boldsymbol { x } ) \right\| ^ { 2 } .
$$

The expectation is taken with respect to all the randomness generated by the algorithm up to iteration t.

To facilitate the subsequent convergence analysis, we introduce the following assumptions.

Assumption 2.1. (A1) For each $s \in [ S ] , f _ { s }$ is L-smooth, $i . e . , \| \nabla f _ { s } ( x ) - \nabla f _ { s } ( y ) \| \le L \| x - y \| , f o r$ any $x , y \in \mathbb { R } ^ { d } ,$

(A2) For each $s \in [ S ] , i \in [ M ]$ , and any realization $\xi _ { i }$ drawn from $\mathcal { D } _ { i } , F _ { s } ( \cdot ; \xi _ { i } )$ is L-smooth, i.e., $\begin{array} { r } { \| \nabla F _ { s } ( x ; \xi _ { i } ) - \nabla F _ { s } ( y ; \xi _ { i } ) \| \le L \| x - y \| , \quad \forall x , y \in \mathbb { R } ^ { d } } \end{array}$

By Assumption 2.1 (A2) and Jensen's inequality, each $f _ { s , i }$ is also L-smooth. Indeed,

$$
\begin{array} { r l } & { \| \nabla f _ { s , i } ( x ) - \nabla f _ { s , i } ( y ) \| = \| \mathbb { E } _ { \xi _ { i } } \left[ \nabla F _ { s } ( x ; \xi _ { i } ) - \nabla F _ { s } ( y ; \xi _ { i } ) \right] \| } \\ & { \qquad \le \mathbb { E } _ { \xi _ { i } } \left[ \| \nabla F _ { s } ( x ; \xi _ { i } ) - \nabla F _ { s } ( y ; \xi _ { i } ) \| \right] } \\ & { \qquad \le L \| x - y \| , \qquad \forall x , y . } \end{array}
$$

Assumption 2.2. There exists a constant $\sigma > 0$ such that, for any $x \in \mathbb { R } ^ { d } , s \in [ S ]$ , and $i \in [ M ]$

$$
\mathbb { E } _ { \xi _ { i } \sim \mathcal { D } _ { i } } \left[ \nabla F _ { s } ( x ; \xi _ { i } ) \right] = \nabla f _ { s , i } ( x ) ,
$$

and

$$
\begin{array} { r } { \mathbb { E } _ { \xi _ { i } \sim \mathcal { D } _ { i } } \left[ \| \nabla F _ { s } ( x ; \xi _ { i } ) - \nabla f _ { s , i } ( x ) \| ^ { 2 } \right] \leq \sigma ^ { 2 } . } \end{array}
$$

## 3. FSMGDA-M-VR algorithm

In what follows, we propose a momentum-based variance-reduced federated stochastic multigradient descent algorithm (FSMGDA-M-VR) for solving (2.1), as outlined in Algorithm 1.

At communication round t and local step k, each client i independently draws a sample $\xi _ { i } ^ { t , k } \sim \mathcal { D } _ { i }$ The samples are independent across clients, local steps, and communication rounds, but are not necessarily identically distributed across clients since the distributions $\{ \mathcal { D } _ { i } \} _ { i = 1 } ^ { M }$ may differ. For each fixed $( t , k , i )$ , the same sample $\xi _ { i } ^ { t , k }$ is used for all objectives $s \in [ S ]$ . In particular, the variancereduction mechanism evaluates $\check { \nabla } F _ { s } ( x _ { s , i } ^ { t , k } ; \xi _ { i } ^ { t , k } )$ and $\nabla F _ { s } ( x ^ { t - 1 } ; \xi _ { i } ^ { t , k } )$ using the same realization $\xi _ { i } ^ { t , k }$

As shown in Algorithm 1, after initializing the model parameters and historical gradients, the algorithm proceeds iteratively over multiple communication rounds. At each communication round t, all clients receive the current global model $x ^ { t }$ and, in parallel, perform K local update steps for each objective function $s \in [ S ]$ . In each local step $k ,$ client i computes a variance-reduced descent direction for objective s by combining the current stochastic gradient with a momentum-based correction term:

$$
\begin{array} { r } { \small g _ { s , i } ^ { t , k } = \nabla F _ { s } ( x _ { s , i } ^ { t , k } ; \xi _ { i } ^ { t , k } ) + ( 1 - \beta ) \left( g _ { s } ^ { t - 1 } - \nabla F _ { s } ( x ^ { t - 1 } ; \xi _ { i } ^ { t , k } ) \right) . } \end{array}
$$

The local model is then updated as $x _ { s , i } ^ { t , k + 1 } = x _ { s , i } ^ { t , k } - \eta g _ { s , i } ^ { t , k }$ . After completing all K local steps, each client computes the average gradient estimate for each objective $\begin{array} { r } { g _ { s , i } ^ { t } ~ = ~ \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } g _ { s , i } ^ { t , k } } \end{array}$ and sends the directions to the server. On the server side, the uploaded estimates are aggregated across clients to form $\begin{array} { r } { g _ { s } ^ { t } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } g _ { s , i } ^ { t } } \end{array}$ . To determine a multi-gradient direction that balances

Algorithm 1: FSMGDA-M-VR   
Input: Initial model $x ^ { - 1 } = x ^ { 0 } ,$ initialization batch size $B \in \mathbb { N } ,$ initial gradient estimate   
$\begin{array} { r } { \boldsymbol { g _ { s } ^ { - 1 } } = \frac { 1 } { B M } \sum _ { i = 1 } ^ { M } \sum _ { b = 1 } ^ { B } \boldsymbol { \nabla } F _ { s } ( \boldsymbol { x } _ { . } ^ { - 1 } ; \xi _ { i } ^ { b } ) } \end{array}$ for $s \in [ S ]$ with mutually independent samples   
$\{ \xi _ { i } ^ { b } : i \in [ M ] , b \in [ B ] \}$ and $\xi _ { i } ^ { b } \sim \mathcal { D } _ { i }$ , local learning rate $\eta > 0$ , global learning rate   
$\gamma > 0 ,$ and momentum parameter $\beta \in ( 0 , 1 )$   
Output: $( x _ { a } , \lambda _ { a } )$ , where a is chosen uniformly at random from $\{ 0 , 1 , \ldots , T - 1 \}$   
for $t = 0 , 1 , \ldots , T - 1$ do   
for each client i in parallel do   
Initial local model $x _ { s , i } ^ { t , 0 } = x ^ { t }$ for each s in parallel do   
for $k = 0 , 1 , \ldots , \dot { K } - 1$ do   
Compute direction $\begin{array} { r } { \small g _ { s , i } ^ { t , k } = \nabla F _ { s } ( x _ { s , i } ^ { t , k } ; \xi _ { i } ^ { t , k } ) + \left( 1 - \beta \right) \left( g _ { s } ^ { t - 1 } - \nabla F _ { s } ( x ^ { t - 1 } ; \xi _ { i } ^ { t , k } ) \right) } \end{array}$   
Update local model $x _ { s , i } ^ { t , k + 1 } = x _ { s , i } ^ { t , k } - \eta g _ { s , i } ^ { t , k }$   
end   
Aggregate local updates $\begin{array} { r } { g _ { s , i } ^ { t } = \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } g _ { s , i } ^ { t , k } } \end{array}$   
end   
end   
Compute $\begin{array} { r } { g _ { s } ^ { t } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } g _ { s , i } ^ { t } , \forall s \in [ S ] } \end{array}$   
Compute $\lambda ^ { t , * } \in [ 0 , 1 ] ^ { S }$ by solving   
$\operatorname* { m i n } _ { \lambda _ { s } ^ { t } \ge 0 } \left\| \sum _ { s \in [ S ] } \lambda _ { s } ^ { t } g _ { s } ^ { t } \right\| ^ { 2 } , \quad \mathrm { s . t . } \sum _ { s \in [ S ] } \lambda _ { s } ^ { t } = 1$   
Compute global multi-gradient direction $\begin{array} { r } { g ^ { t } = \sum _ { s \in [ S ] } \lambda _ { s } ^ { t , * } g _ { s } ^ { t } } \end{array}$   
Update global model $\boldsymbol { x } ^ { t + 1 } = \boldsymbol { x } ^ { t } - \gamma \boldsymbol { g } ^ { t }$   
end

$$
\operatorname* { m i n } _ { \lambda _ { s } ^ { t } \geq 0 , \sum _ { s \in [ S ] } \lambda _ { s } ^ { t } = 1 } \left\| \sum _ { s \in [ S ] } \lambda _ { s } ^ { t } g _ { s } ^ { t } \right\| ^ { 2 }
$$

$$
\lambda _ { s } ^ { t , * }
$$

These coefficients are used to construct the global direction $\begin{array} { r } { g ^ { t } = \sum _ { s = 1 } ^ { S } \lambda _ { s } ^ { t , * } g _ { s } ^ { t } } \end{array}$ . Finally, the global model is updated as $\boldsymbol { x } ^ { t + 1 } = \boldsymbol { x } ^ { t } - \gamma \boldsymbol { g } ^ { t }$

Remark 3.1. Relative to FSMGDA, the server-side Pareto aggregation mechanism remains unchanged. The main algorithmic modification is the introduction of an objective-wise momentum variance-reduced estimator into the local updates. The principal technical contribution is the analysis showing that this modification improves the stationarity rate from $\mathcal { O } ( T ^ { - 1 / 2 } ) \mathrm { t o } \mathcal { O } ( T ^ { - 2 / 3 } )$

## 4. Convergence analysis

In this section, we show that Algorithm 1 achieves a convergence rate of $\mathcal { O } ( T ^ { - \frac { 2 } { 3 } } )$ in the nonconvex case. To facilitate the theoretical analysis, we first introduce the filtration used to characterize the evolution of the iterates and the randomness generated during the execution of the algorithm.

Let

$$
\mathcal { F } ^ { 0 } = \sigma \left( \left\{ \xi _ { i } ^ { b } : 1 \leq i \leq M , 1 \leq b \leq B \right\} \right) ,
$$

where $\{ \xi _ { i } ^ { b } : 1 \le i \le M , 1 \le b \le B \}$ are the samples used to initialize $g _ { s } ^ { - 1 }$ , with $\xi _ { i } ^ { b } \sim \mathcal { D } _ { i }$ . These initialization samples are mutually independent and are independent of all samples generated in the subsequent communication rounds. Hence, the initial gradient estimates $g _ { s } ^ { - 1 } , s \in [ S ]$ , are ${ \mathcal { F } } ^ { 0 } .$ measurable. For each communication round $t , \mathcal { F } ^ { t }$ denotes the σ-algebra containing all randomness revealed before the beginning of round t. Within round t, we define the local-step filtration

$$
\mathcal { F } ^ { t , k } = \sigma \left( \mathcal { F } ^ { t } \cup \left\{ \xi _ { i } ^ { t , j } : 1 \leq i \leq M , 0 \leq j < k \right\} \right) , \qquad k = 0 , 1 , \ldots , K .
$$

Thus, $\mathcal { F } ^ { t , 0 } = \mathcal { F } ^ { t }$ , and $\mathcal { F } ^ { t , k }$ contains all randomness revealed before the k-th local update in round t. After completing the K local steps, we set

$$
\mathcal { F } ^ { t + 1 } = \mathcal { F } ^ { t , K } = \sigma \left( \mathcal { F } ^ { t } \cup \left\{ \xi _ { i } ^ { t , j } : 1 \leq i \leq M , 0 \leq j < K \right\} \right) .
$$

Under this definition, the local variables $x _ { s , i } ^ { t , k }$ and all global quantities available before the k-th local update, including $x ^ { t } , \ x ^ { t - 1 }$ , and $g _ { s } ^ { t - 1 }$ , are $\mathcal { F } ^ { t , k } .$ -measurable, whereas the fresh sample $\xi _ { i } ^ { t , k }$ is independent of $\mathcal { F } ^ { t , k }$ . Moreover, conditioned on $\mathcal { F } ^ { t , k }$ , the samples $\{ \xi _ { i } ^ { t , k } \} _ { i = 1 } ^ { M }$ are independent across clients and satisfy $\xi _ { i } ^ { t , k } \sim \mathcal { D } _ { i }$ . We use $\mathbb { E } [ \cdot ~ \vert ~ \mathcal { F } ^ { t , k } ]$ to denote the conditional expectation given the σ-algebra $\mathcal { F } ^ { t , k }$ , which contains all randomness revealed before the k-th local update in round t, and $\mathbb { E } [ \cdot ]$ to denote the expectation over all randomness in the algorithm. Throughout the proofs, we use $\textstyle \sum _ { i }$ to denote the summation over $i \in \{ 1 , \ldots , M \}$ and $\textstyle \sum _ { k }$ to denote the summation over $k \in \{ 0 , \ldots , K - 1 \}$ . For all $t \geq 0$ , we define the auxiliary variables as follows:

$$
\mathcal { E } _ { s } ^ { t } : = \mathbb { E } [ \| \nabla f _ { s } ( x ^ { t } ) - g _ { s } ^ { t } \| ^ { 2 } ] ,
$$

$$
U _ { s } ^ { t } : = \frac { 1 } { K M } \sum _ { i } \sum _ { k } \mathbb { E } [ \Vert x _ { s , i } ^ { t , k } - x ^ { t } \Vert ^ { 2 } ] ,
$$

$$
\zeta _ { s , i } ^ { t , k } : = \mathbb { E } [ x _ { s , i } ^ { t , k + 1 } - x _ { s , i } ^ { t , k } | \mathcal { F } ^ { t , k } ] ,
$$

$$
\Xi _ { s } ^ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } [ \| \zeta _ { s , i } ^ { t , 0 } \| ^ { 2 } ] .
$$

Furthermore, let $x ^ { - 1 } : = x ^ { 0 }$ and $\mathcal { E } _ { s } ^ { - 1 } : = \mathbb { E } \left[ \Vert \nabla f _ { s } ( x ^ { 0 } ) - g _ { s } ^ { - 1 } \Vert ^ { 2 } \right]$ . For the subsequent analysis, we also define the auxiliary initial aggregated multi-gradient direction using the same minimum-norm aggregation rule as in Algorithm 1. Specifically, let

$$
\lambda ^ { - 1 , * } \in \arg \operatorname* { m i n } _ { \lambda \in \mathcal { C } } \left\| \sum _ { s \in [ S ] } \lambda _ { s } g _ { s } ^ { - 1 } \right\| ^ { 2 } ,
$$

and define

$$
g ^ { - 1 } : = \sum _ { s \in [ S ] } \lambda _ { s } ^ { - 1 , * } g _ { s } ^ { - 1 } .
$$

By construction, $g ^ { - 1 }$ is ${ \mathcal { F } } ^ { 0 } .$ -measurable.

To establish the convergence properties of Algorithm 1, we further impose the following boundedness condition along the sequence of iterates generated by the algorithm.

Assumption 4.1. Let $\{ x ^ { t } \} _ { t \geq 0 }$ be the sequence generated by Algorithm 1. There exists a constant $D > 0$ such that

$$
\operatorname* { s u p } _ { t \geq 0 , s \in [ S ] } \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \left[ \| \nabla f _ { s , i } ( x ^ { t } ) \| ^ { 2 } \right] \leq D ^ { 2 } ,
$$

where the expectation is taken with respect to the randomness of the algorithm up to iteration t.

Remark 4.1. Assumption 4.1 controls the local expected gradients along the sequence generated by Algorithm 1. Specifically, it requires the client-averaged squared norms of the local expected gradients, evaluated along the generated trajectory, to be uniformly bounded in expectation. Such boundedness conditions are commonly adopted in the analysis of federated learning and federated multiobjective learning [13, 22, 30, 32], as well as in stochastic multiobjective optimization [36– 39]. Unlike a global bounded-gradient condition imposed for all $x ,$ Assumption 4.1 only requires boundedness along the sequence generated by the algorithm and is sufficient for our convergence analysis.

Our subsequent analysis will be based on the following foundational lemmas.

Lemma 4.1. Under Assumption 2.1, if $\begin{array} { r } { \gamma L \leq \frac { 1 } { 2 } } \end{array}$ , then, for all $t \geq 0$

$$
\mathbb { E } [ f _ { s } ( x ^ { t + 1 } ) ] \leq \mathbb { E } [ f _ { s } ( x ^ { t } ) ] - \frac { \gamma } { 4 } \mathbb { E } [ \Vert g ^ { t } \Vert ^ { 2 } ] + \frac { \gamma } { 2 } \mathcal { E } _ { s } ^ { t } , \quad s \in [ S ] .
$$

and

$$
\frac { 2 \left( \mathbb { E } [ f _ { s } ( x ^ { T } ) ] - f _ { s } ( x ^ { 0 } ) \right) } { \gamma } + \frac { 1 } { 2 } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \| g ^ { t } \| ^ { 2 } ] \leq \sum _ { t = 0 } ^ { T - 1 } \mathcal { E } _ { s } ^ { t } , \quad s \in [ S ] .
$$

Proof. By the L-smoothness of $f _ { s }$ we have

$$
\begin{array} { r l } & { f _ { s } ( x ^ { t + 1 } ) \leq f _ { s } ( x ^ { t } ) + \left. \nabla f _ { s } ( x ^ { t } ) , - \gamma g ^ { t } \right. + \displaystyle \frac { 1 } { 2 } L \| \gamma g ^ { t } \| ^ { 2 } } \\ & { \qquad = f _ { s } ( x ^ { t } ) + \left. \nabla f _ { s } ( x ^ { t } ) - g _ { s } ^ { t } , - \gamma g ^ { t } \right. - \gamma \left. g _ { s } ^ { t } , g ^ { t } \right. + \displaystyle \frac { 1 } { 2 } L \| \gamma g ^ { t } \| ^ { 2 } } \\ & { \qquad \leq f _ { s } ( x ^ { t } ) - \gamma \left. \nabla f _ { s } ( x ^ { t } ) - g _ { s } ^ { t } , g ^ { t } \right. - \gamma \| g ^ { t } \| ^ { 2 } + \displaystyle \frac { 1 } { 2 } L \| \gamma g ^ { t } \| ^ { 2 } } \\ & { \qquad \leq f _ { s } ( x ^ { t } ) + \displaystyle \frac { \gamma } { 2 } \| \nabla f _ { s } ( x ^ { t } ) - g _ { s } ^ { t } \| ^ { 2 } + \displaystyle \frac { \gamma } { 2 } \| g ^ { t } \| ^ { 2 } - \gamma \| g ^ { t } \| ^ { 2 } + \displaystyle \frac { 1 } { 2 } L \gamma ^ { 2 } \| g ^ { t } \| ^ { 2 } } \\ & { \qquad = f _ { s } ( x ^ { t } ) + \displaystyle \frac { \gamma } { 2 } \| \nabla f _ { s } ( x ^ { t } ) - g _ { s } ^ { t } \| ^ { 2 } - \gamma \left( \displaystyle \frac { 1 } { 2 } - \displaystyle \frac { 1 } { 2 } L \gamma \right) \| g ^ { t } \| ^ { 2 } , } \end{array}
$$

where the second inequality follows from the fact that $g ^ { t }$ is the minimum-norm element in conv $\{ g _ { s } ^ { t }$ $s \in [ S ] \}$ . By the first-order optimality condition of this projection problem,

$$
\langle g _ { s } ^ { t } - g ^ { t } , g ^ { t } \rangle \geq 0 , \quad \forall s \in [ S ] ,
$$

which implies that

$$
\langle g _ { s } ^ { t } , g ^ { t } \rangle \geq \| g ^ { t } \| ^ { 2 } , \quad \forall s \in [ S ] .
$$

With $\begin{array} { r } { \gamma \le \frac { 1 } { 2 L } } \end{array}$ and taking the global expectation, we obtain

$$
\mathbb { E } [ f _ { s } ( x ^ { t + 1 } ) ] \leq \mathbb { E } [ f _ { s } ( x ^ { t } ) ] - \frac { \gamma } { 4 } \mathbb { E } [ \Vert g ^ { t } \Vert ^ { 2 } ] + \frac { \gamma } { 2 } \mathcal { E } _ { s } ^ { t } .
$$

Then, summing the above inequality over $t = 0 , \ldots , T - 1$ yields

$$
\frac { 2 } { \gamma } \sum _ { t = 0 } ^ { T - 1 } \Big ( \mathbb { E } [ f _ { s } ( x ^ { t + 1 } ) - f _ { s } ( x ^ { t } ) ] \Big ) + \frac { 1 } { 2 } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \| g ^ { t } \| ^ { 2 } ] = \frac { 2 \Big ( \mathbb { E } [ f _ { s } ( x ^ { T } ) ] - f _ { s } ( x ^ { 0 } ) \Big ) } { \gamma } + \frac { 1 } { 2 } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \| g ^ { t } \| ^ { 2 } ] \leq \sum _ { t = 0 } ^ { T - 1 } \xi _ { s } ^ { t } ,
$$

which completes the proof.

To handle local updates and client sampling, we will also use the following technical lemmas.

Lemma 4.2 (See [25, Lemma 4]). Let $\{ X _ { 1 } , \ldots , X _ { \tau } \}$ be a collection $o f \mathbb { R } ^ { d } .$ -valued random variables that are potentially dependent. Suppose that $\mathbb { E } [ X _ { i } ] = \mu _ { i }$ and $\mathbb { E } [ \| X _ { i } - \mu _ { i } \| ^ { 2 } ] \le \sigma ^ { 2 } , i = 1 , \dots , \tau$ . Then

$$
\mathbb { E } \left[ \left. \sum _ { i = 1 } ^ { \tau } X _ { i } \right. ^ { 2 } \right] \leq \left. \sum _ { i = 1 } ^ { \tau } \mu _ { i } \right. ^ { 2 } + \tau ^ { 2 } \sigma ^ { 2 } .
$$

Suppose that $\{ \mathcal { G } _ { i } \} _ { i = 0 } ^ { \tau }$ is a fltration, i.e.,

$$
\mathcal { G } _ { 0 } \subseteq \mathcal { G } _ { 1 } \subseteq \cdots \subseteq \mathcal { G } _ { \tau }
$$

and that $X _ { i }$ is Gi-measurable for each $i = 1 , \ldots , \tau .$ Define

$$
\mu _ { i } = \mathbb { E } \left[ X _ { i } \mid { \mathcal { G } } _ { i - 1 } \right] ,
$$

and suppose that

$$
\begin{array} { r } { \mathbb { E } \left[ \| X _ { i } - \mu _ { i } \| ^ { 2 } | \mathcal { G } _ { i - 1 } \right] \leq \sigma ^ { 2 } , \qquad i = 1 , \dots , \tau . } \end{array}
$$

Then $\{ X _ { i } - \mu _ { i } \} _ { i = 1 } ^ { \tau }$ forms a martingale-difference sequence with respect to $\{ \mathcal { G } _ { i } \} _ { i = 0 } ^ { \tau }$ , and

$$
\mathbb { E } \left[ \left. \sum _ { i = 1 } ^ { \tau } X _ { i } \right. ^ { 2 } \right] \leq 2 \mathbb { E } \left[ \left. \sum _ { i = 1 } ^ { \tau } \mu _ { i } \right. ^ { 2 } \right] + 2 \tau \sigma ^ { 2 } .
$$

Lemma 4.3. Suppose that Assumptions 2.1–2.2 hold. $\begin{array} { r } { I f \gamma L \leq \sqrt { \frac { \beta K M } { 5 4 } } } \end{array}$ , then, for all $t \geq 1$

$$
\mathscr { E } _ { s } ^ { t } \leq ( 1 - \beta ) \mathscr { E } _ { s } ^ { t - 1 } + \frac { 4 } { \beta } L ^ { 2 } U _ { s } ^ { t } + \frac { 3 \beta ^ { 2 } \sigma ^ { 2 } } { K M } + \frac { 2 \beta } { 9 } \mathbb { E } [ \| g ^ { t - 1 } \| ^ { 2 } ] .
$$

and for $t = 0$

$$
\mathcal { E } _ { s } ^ { 0 } \leq ( 1 - \beta ) \mathcal { E } _ { s } ^ { - 1 } + \frac { 4 } { \beta } L ^ { 2 } U _ { s } ^ { 0 } + \frac { 3 \beta ^ { 2 } \sigma ^ { 2 } } { K M } .
$$

Proof. By the definition of $\mathcal { E } _ { s } ^ { t } .$ we have

$$
\begin{array} { l } { \displaystyle \mathcal { E } _ { \alpha } ^ { * } = \left[ \left[ \frac { 1 } { K ^ { \alpha } } \displaystyle \sum _ { j = 1 } ^ { N } \nabla F _ { + } ( L _ { \alpha } ^ { \alpha } ; \xi _ { \alpha } ^ { \beta } ) + \{ 1 - \beta \} \left( \alpha ^ { \beta - 1 } - \frac { 1 } { K ^ { \alpha } M } \displaystyle \sum _ { i = 1 } ^ { N } \nabla F _ { + } ( \alpha ^ { - 1 } ; \xi _ { \alpha } ^ { \beta } ) - \nabla f _ { i } ( \alpha ^ { \beta } ) ^ { * } \right) \right] ^ { 2 } \right] } \\ { \displaystyle = \mathbb { E } \left[ \left( 1 - \beta \langle \xi _ { \alpha } ^ { - 1 } \cdot \nabla f _ { i } ( \alpha ^ { - 1 } ) \rangle + \frac { 1 } { K ^ { \alpha } M } \displaystyle \sum _ { i = 1 } ^ { N } \nabla F _ { - } ( \alpha ^ { - 1 } ; \xi _ { \alpha } ^ { \beta } ) - \nabla f _ { i } ( \alpha ^ { \beta } ) ^ { * } \right) - \nabla f _ { i } ( \alpha ^ { \beta } ) ^ { * } \right] } \\ { \displaystyle \qquad + ( 1 - \beta ) \big ( \nabla f _ { - } \xi _ { \alpha } ^ { \alpha - 1 } \big ) - \frac { 1 } { K ^ { \alpha } M } \displaystyle \sum _ { i = 1 } ^ { N } \nabla F _ { + } ( \alpha ^ { - 1 } ; \xi _ { \alpha } ^ { \beta } ) \Big ] \Bigg ] ^ { 2 } } \\ { \displaystyle - ( 1 - \beta ) ^ { 2 } \xi _ { \alpha } ^ { * } \displaystyle \sum _ { i = 1 } ^ { N } \frac { 1 } { K ^ { \alpha } M } \displaystyle \sum _ { i = 1 } ^ { N } - \nabla f _ { i } ( \alpha ^ { - 1 } ; \xi _ { \alpha } ^ { \beta } ) \cdot \frac { 1 } { K ^ { \alpha } M } \displaystyle \sum _ { i = 1 } ^ { N } \nabla f _ { i } ( \alpha ^ { \beta } ; \xi _ { \alpha } ^ { \beta } ) - \nabla f _ { i } ( \alpha ^ { \beta } ) ^ { * } \Big ] \Bigg ] } \\  \displaystyle \ \end{array}
$$

By the AM-GM inequality and Assumption 2.1,

$$
\begin{array} { l } { \displaystyle \Lambda _ { 1 } \leq \beta ( 1 - \beta ) ^ { 2 } \mathcal { E } _ { s } ^ { t - 1 } + \frac { 1 } { \beta } \mathbb { E } \left[ \left\| \frac { 1 } { K M } \sum _ { i , k } \nabla f _ { s , i } ( x _ { s , i } ^ { t , k } ) - \nabla f _ { s } ( x ^ { t } ) \right\| ^ { 2 } \right] } \\ { \displaystyle \leq \beta ( 1 - \beta ) ^ { 2 } \mathcal { E } _ { s } ^ { t - 1 } + \frac { 1 } { \beta } \frac { 1 } { K ^ { 2 } M ^ { 2 } } \mathbb { E } \left[ \left\| \sum _ { i , k } \big ( \nabla f _ { s , i } ( x _ { s , i } ^ { t , k } ) - \nabla f _ { s , i } ( x ^ { t } ) \big ) \right\| ^ { 2 } \right] } \end{array}
$$

$$
\begin{array} { l } { \displaystyle \leq \beta ( 1 - \beta ) ^ { 2 } \mathcal { E } _ { s } ^ { t - 1 } + \frac { 1 } { \beta } \frac { 1 } { K M } \mathbb { E } \left[ \displaystyle \sum _ { i , k } \left\| \nabla f _ { s , i } ( x _ { s , i } ^ { t , k } ) - \nabla f _ { s , i } ( x ^ { t } ) \right\| ^ { 2 } \right] } \\ { \displaystyle \leq \beta ( 1 - \beta ) ^ { 2 } \mathcal { E } _ { s } ^ { t - 1 } + \frac { 1 } { \beta } \frac { 1 } { K M } \mathbb { E } \left[ \displaystyle \sum _ { i , k } L ^ { 2 } \| x _ { s , i } ^ { t , k } - x ^ { t } \| ^ { 2 } \right] } \\ { \displaystyle \leq \beta ( 1 - \beta ) ^ { 2 } \mathcal { E } _ { s } ^ { t - 1 } + \frac { 1 } { \beta } L ^ { 2 } U _ { s } ^ { t } . } \end{array}
$$

For $\Lambda _ { 2 }$ , we decompose it into three terms as

$$
\begin{array} { r l } & { \lambda _ { 2 } = \mathbb { E } \Bigg [ \Bigg | \frac { 1 } { K ^ { 2 } H _ { 2 } ^ { 3 } } \underset { m _ { \lambda } } { \sum } \Bigg ( \Gamma \hat { \mathbf { E } } \hat { \mathbf { E } } \hat { \mathbf { E } } \hat { \mathbf { E } } ( \hat { \mathbf { x } } _ { \lambda } ^ { \dagger } \hat { \mathbf { e } } _ { \lambda } ^ { \hat { \mathbf { x } } } \hat { \mathbf { e } } _ { \lambda } ^ { \hat { \mathbf { x } } } ) - \nabla F \delta ( \mathbf { x } ^ { \dagger } \hat { \mathbf { e } } _ { \lambda } ^ { \hat { \mathbf { x } } } ) \hat { \mathbf { e } } _ { \lambda } ^ { \hat { \mathbf { x } } } \hat { \mathbf { e } } _ { \lambda } ^ { \hat { \mathbf { x } } } \Bigg ) + \delta ( \frac { 1 } { K ^ { 2 } \hat { \mathbf { E } } \hat { \mathbf { E } } \hat { \mathbf { E } } \hat { \mathbf { E } } \hat { \mathbf { x } } } \hat { \mathbf { E } } \hat { \mathbf { E } } \hat { \mathbf { x } } ^ { \hat { \mathbf { x } } } \hat { \mathbf { e } } _ { \lambda } ^ { \hat { \mathbf { x } } } - \nabla F \delta ( \mathbf { x } ^ { \mathcal { \mathcal { \bar { \nu } } } } ) ) } \\ &  \quad + ( 1 - \Delta ) \Bigg ( \frac { 1 } { K ^ { 2 } H _ { 2 } ^ { 3 } } \sum ( \Gamma \hat { \mathbf { E } } \hat { \mathbf { E } } \hat { \mathbf { x } } \hat { \mathbf { E } } ( \mathbf { x } ^ { \hat { \mathcal { \nu } } } \hat { \mathbf { e } } _ { \lambda } ^ { \hat { \mathbf { x } } } ) - \nabla F \delta ( \mathbf { x } ^ { \mathcal { \nu } } \hat { \mathbf { e } } _ { \lambda } ^ { \hat { \mathbf { x } } } ) - \nabla F \delta ( \mathbf { x } ^ { \mathcal { \nu } } ) - \nabla F \delta ( \mathbf { x } ^ { \mathcal { \nu } } ) - \nabla F \delta ( \mathbf { x } ^  \mathcal  \nu \end{array}
$$

By Assumption 2.1,

$$
\begin{array} { l } { \displaystyle \Phi _ { 1 } = \frac { 3 } { K ^ { 2 } M ^ { 2 } } \mathbb { E } \left[ \left\| \displaystyle \sum _ { i , k } \left( \nabla F _ { s } ( x _ { s , i } ^ { t , k } ; \xi _ { i } ^ { t , k } ) - \nabla F _ { s } ( x ^ { t } ; \xi _ { i } ^ { t , k } ) \right) \right\| ^ { 2 } \right] } \\ { \displaystyle \quad \leq \frac { 3 } { K M } \mathbb { E } \left[ \displaystyle \sum _ { i , k } \left\| \nabla F _ { s } ( x _ { s , i } ^ { t , k } ; \xi _ { i } ^ { t , k } ) - \nabla F _ { s } ( x ^ { t } ; \xi _ { i } ^ { t , k } ) \right\| ^ { 2 } \right] } \\ { \displaystyle \quad \leq \frac { 3 } { K M } \displaystyle \sum _ { i , k } \mathbb { E } \left[ L ^ { 2 } \left\| x _ { s , i } ^ { t , k } - x ^ { t } \right\| ^ { 2 } \right] } \\ { \displaystyle = 3 L ^ { 2 } U _ { k } ^ { k } . } \end{array}
$$

By Assumption 2.2,

$$
\begin{array} { r l r } & { } & { \Phi _ { 2 } = \frac { 3 \beta ^ { 2 } } { K ^ { 2 } M ^ { 2 } } \mathbb { E } \left[ \left\| \displaystyle \sum _ { i , k } \left( \nabla F _ { s } ( x ^ { t } ; \xi _ { i } ^ { t , k } ) - \nabla f _ { s , i } ( x ^ { t } ) \right) \right\| ^ { 2 } \right] } \\ & { } & { = \frac { 3 \beta ^ { 2 } } { K ^ { 2 } M ^ { 2 } } \mathbb { E } \left[ \displaystyle \sum _ { i , k } \left\| \nabla F _ { s } ( x ^ { t } ; \xi _ { i } ^ { t , k } ) - \nabla f _ { s , i } ( x ^ { t } ) \right\| ^ { 2 } \right] } \end{array}
$$

$$
\begin{array} { l } { \displaystyle + 2 \sum _ { ( i , k ) \neq ( j , l ) } \left. \nabla F _ { s } ( x ^ { t } ; \xi _ { i } ^ { t , k } ) - \nabla f _ { s , i } ( x ^ { t } ) , \nabla F _ { s } ( x ^ { t } ; \xi _ { j } ^ { t , l } ) - \nabla f _ { s , j } ( x ^ { t } ) \right. \Bigg ] } \\ { \displaystyle \leq \frac { 3 \beta ^ { 2 } } { K ^ { 2 } M ^ { 2 } } \sum _ { i , k } \sigma ^ { 2 } = \frac { 3 \beta ^ { 2 } \sigma ^ { 2 } } { K M } . } \end{array}
$$

Again using Assumption 2.1, we have

$$
\begin{array} { r l } { \Phi _ { 1 } - \frac { \mathrm { d } \mathrm { d } } { \mathrm { d } t } - \frac { \mathrm { d } } { \mathrm { d } t } \Theta ^ { 2 } \operatorname* { m a x } ^ { 2 } \Bigg [ \Bigg | \sum _ { i = 1 } ^ { \infty } \Big \{ \mathrm { P r } { \cal A } _ { i } ^ { \mathrm { ( i ) } } + \mathrm { P r } { \cal A } _ { i } ^ { \mathrm { ( i ) } } - \mathrm { P r } { \cal A } _ { i } ^ { \mathrm { ( i ) } } + \mathrm { P r } { \cal A } _ { i } ^ { \mathrm { ( i ) } } - \mathrm { P r } { \cal A } _ { i } ^ { \mathrm { ( i ) } } - \mathrm { P r } { \cal A } _ { i } ^ { \mathrm { ( i ) } } \Big \} \Bigg | ^ { 2 } \Bigg ] } & { } \\ { - \frac { \mathrm { d } ( 1 - \mathrm { P r } ) ^ { 2 } } { \mathrm { d } t } \Bigg [ - \mathrm { P r } { \cal A } _ { i } ^ { \mathrm { ( i ) } } + \mathrm { P r } { \cal A } _ { i } ^ { \mathrm { ( i ) } } - \mathrm { P r } { \cal A } _ { i } ^ { \mathrm { ( i ) } } - \mathrm { P r } { \cal A } _ { i } ^ { \mathrm { ( i ) } } - \mathrm { P r } { \cal A } _ { i } ^ { \mathrm { ( i ) } } - \mathrm { P r } { \cal A } _ { i } ^ { \mathrm { ( i ) } } \Bigg \} \Bigg | ^ { 2 } } & { } \\ { + \frac { \mathrm { P r } } { \mathrm { d } t } \sum _ { i = 1 } ^ { \infty } \mathrm { P r } { \cal A } _ { i } ^ { \mathrm { ( i ) } } - \mathrm { P r } { \cal A } _ { i } ^ { \mathrm { ( i ) } } - \mathrm { P r } { \cal A } _ { i } ^ { \mathrm { ( i ) } } - \mathrm { P r } { \cal A } _ { i } ^ { \mathrm { ( i ) } } + \mathrm { P r } { \cal A } _ { i } ^ { \mathrm { ( i ) } } - \mathrm { P r } { \cal A } _ { i } ^ { \mathrm { ( i ) } } \Bigg \} } & { } \\  \quad \end{array}
$$

Substituting these three bounds, we obtain

$$
\Lambda _ { 2 } \leq 3 L ^ { 2 } U _ { s } ^ { t } + \frac { 3 \beta ^ { 2 } \sigma ^ { 2 } } { K M } + 1 2 ( 1 - \beta ) ^ { 2 } \frac { L ^ { 2 } } { K M } \mathbb { E } [ \| x ^ { t } - x ^ { t - 1 } \| ^ { 2 } ] .
$$

Therefore, for $t \geq 1$ ，

$$
\begin{array} { r l r } & { \displaystyle \mathcal E _ { s } ^ { t } \leq ( 1 - \beta ) \mathcal E _ { s } ^ { t - 1 } + \frac 4 \beta L ^ { 2 } U _ { s } ^ { t } + \frac { 3 \beta ^ { 2 } \sigma ^ { 2 } } { K M } + 1 2 ( 1 - \beta ) ^ { 2 } \frac { L ^ { 2 } } { K M } \mathbb E [ \| x ^ { t } - x ^ { t - 1 } \| ^ { 2 } ] } & \\ & { \displaystyle \quad \leq ( 1 - \beta ) \mathcal E _ { s } ^ { t - 1 } + \frac 4 \beta L ^ { 2 } U _ { s } ^ { t } + \frac { 3 \beta ^ { 2 } \sigma ^ { 2 } } { K M } + \frac { 2 \beta } { 9 } \mathbb E [ \| g ^ { t - 1 } \| ^ { 2 } ] , } & \end{array}
$$

where the last inequality is derived by $\begin{array} { r } { x ^ { t } = x ^ { t - 1 } - \gamma g ^ { t - 1 } \mathrm { ~ a n d ~ } \gamma L \leq \sqrt { \frac { \beta K M } { 5 4 } } } \end{array}$ . Similarly, for $t = 0$ , we can obtain

$$
\mathcal { E } _ { s } ^ { 0 } \leq ( 1 - \beta ) \mathcal { E } _ { s } ^ { - 1 } + \frac { 4 } { \beta } L ^ { 2 } U _ { s } ^ { 0 } + \frac { 3 \beta ^ { 2 } \sigma ^ { 2 } } { K M } ,
$$

which completes the proof.

Lemma 4.4. Suppose that Assumptions 2.1–2.2 hold. $I f \eta L K \le 1$ and $2 4 e ^ { 2 } ( \eta L ) ^ { 4 } K ^ { 4 } + 2 4 \eta ^ { 2 } L ^ { 2 } K \le$ $\frac { 1 } { 2 }$ , then

$$
U _ { s } ^ { t } \leq 4 e ^ { 2 } K ^ { 2 } \Xi _ { s } ^ { t } + 1 2 ( \eta K ) ^ { 2 } \left( 8 ( \eta K L ) ^ { 2 } + K ^ { - 1 } \right) \left( \beta ^ { 2 } \sigma ^ { 2 } + 4 ( \gamma L ) ^ { 2 } \mathbb { E } [ \| g ^ { t - 1 } \| ^ { 2 } ] \right) .
$$

Proof. Note that

$$
\begin{array} { r l } & { \zeta _ { s , i } ^ { t , k } = { \mathbb E } [ x _ { s , i } ^ { t , k + 1 } - x _ { s , i } ^ { t , k } \ \vert \ { \mathcal F } ^ { t , k } ] = { \mathbb E } [ - \eta g _ { s , i } ^ { t , k } \ \vert \ { \mathcal F } ^ { t , k } ] } \\ & { \qquad = - \eta \left( \nabla f _ { s , i } ( x _ { s , i } ^ { t , k } ) + ( 1 - \beta ) \left( g _ { s } ^ { t - 1 } - \nabla f _ { s , i } ( x ^ { t - 1 } ) \right) \right) . } \end{array}
$$

Throughout this proof, for a square-integrable random vector X, we use the scalar conditional variance

$$
\operatorname { V a r } ( X \mid { \mathcal { F } } ) : = \operatorname { \mathbb { E } } \left[ \| X - \operatorname { \mathbb { E } } [ X \mid { \mathcal { F } } ] \| ^ { 2 } \mid { \mathcal { F } } \right] .
$$

Thus, we have

$$
\begin{array} { r l } & { \mathbb { E } [ \| \zeta _ { s , i } ^ { t , j } - \zeta _ { s , i } ^ { t , j - 1 } \| ^ { 2 } ] = \mathbb { E } \left[ \left\| - \eta \left( \nabla f _ { s , i } ( x _ { s , i } ^ { t , j } ) + \left( 1 - \beta \right) \left( g _ { s } ^ { t - 1 } - \nabla f _ { s , i } ( x ^ { t - 1 } ) \right) \right) \right. } \\ & { \qquad \left. + \eta \left( \nabla f _ { s , i } ( x _ { s , i } ^ { t , j - 1 } ) + \left( 1 - \beta \right) \left( g _ { s } ^ { t - 1 } - \nabla f _ { s , i } ( x ^ { t - 1 } ) \right) \right) \right\| ^ { 2 } \right] } \\ & { \qquad = \mathbb { E } \left[ \left\| - \eta \nabla f _ { s , i } ( x _ { s , i } ^ { t , j } ) + \eta \nabla f _ { s , i } ( x _ { s , i } ^ { t , j - 1 } ) \right\| ^ { 2 } \right] } \\ & { \qquad \le \eta ^ { 2 } L ^ { 2 } \mathbb { E } [ \| x _ { s , i } ^ { t , j } - x _ { s , i } ^ { t , j - 1 } \| ^ { 2 } ] } \\ & { \qquad = \eta ^ { 2 } L ^ { 2 } \left( \mathbb { E } [ \| \zeta _ { s , i } ^ { t , j - 1 } \| ^ { 2 } ] + \mathbb { E } \left[ \mathrm { V a r } ( x _ { s , i } ^ { t , j } - x _ { s , i } ^ { t , j - 1 } \mid \mathcal { F } ^ { t , j - 1 } ) \right] \right) , } \end{array}
$$

where the last identity is the conditional bias-variance decomposition. Since

$$
\begin{array} { r l } &  \quad \mathbb { E } [ \operatorname { W r } ( x _  \hat { \mathcal { S } } _ { \hat { \mathcal { S } } _ { \hat { \mathcal { S } } _ { \hat { \mathcal { S } } _ { \hat { S } } _ { \hat { S } _ { \mathcal { S } } } } } } ^ { ( \mathcal { S } ) } \frac { x _ { \hat { \mathcal { S } } _ { \hat { S } _ { \hat { S } } _ { \hat { S } } } ^ { ( \mathcal { S } ) - 1 } } { \xi _ { \hat { S } _ { \hat { S } } } ^ { ( \mathcal { S } ) - 1 } } \mid \mathcal { F } ^ { \mathcal { S } - 1 } ) ] } \\ & { = \mathbb { E } [ \operatorname { W i n } ( - \mathcal { P } y _ { \hat { S } _ { \hat { S } _ { \mathcal { S } _ { \hat { S } _ { \mathcal { S } } } } } } ^ { ( \mathcal { S } ) - 1 } \mid \mathcal { F } ^ { \mathcal { S } } ) ^ { \frac { 1 } { 2 } } ] } \\ & { = \mathbb { E } [ \mathbb { E } [ \| - \operatorname { W g } _ { x _ { \hat { S } _ { \mathcal { S } _ { \hat { S } _ { \mathcal { S } } } } } } ^ { ( \mathcal { S } ) - 1 } - \operatorname { E g } _ { x _ { \hat { S } _ { \hat { S } _ { \mathcal { S } } } } } ^ { ( \mathcal { S } ) - 1 } \| \mathcal { F } ^ { \mathcal { S } - 1 } ] ] ^ { 2 } [ \mathcal { F } ^ { \mathcal { H } - \mathcal { S } - 1 } ] ] } \\ &  - \eta ^ { \mathcal { N } } \mathbb { E } [ \| \mathcal { F } \hat { F } \hat { F } _ { \mathcal { S } _ { \hat { S } _ { \hat { S } _ { \mathcal { S } } } } } ( x _ { \hat { S } _ { \hat { S } } } ^ { ( \mathcal { S } ) } \mid \mathcal { S } ^ { \mathcal { S } _ { \hat { S } _ { \hat { S } _ { \hat { S } } } } } ^ { ( \mathcal { S } ) - 1 } ) - \nabla f _  \mathcal { S }  \end{array}
$$

where the last inequality follows from Assumptions 2.1 and 2.2. Thus, we have

$$
\begin{array} { r l } & { \quad \mathbb { E } [ \| \zeta _ { s , i } ^ { t , j } - \zeta _ { s , i } ^ { t , j - 1 } \| ^ { 2 } ] } \\ & { \le \eta ^ { 2 } L ^ { 2 } \left( \mathbb { E } [ \| \zeta _ { s , i } ^ { t , j - 1 } \| ^ { 2 } ] + 3 \beta ^ { 2 } \eta ^ { 2 } \sigma ^ { 2 } + 6 \eta ^ { 2 } ( 1 - \beta ) ^ { 2 } L ^ { 2 } \mathbb { E } [ \| x ^ { t - 1 } - x _ { s , i } ^ { t , j - 1 } \| ^ { 2 } ] \right) } \\ & { \le \eta ^ { 2 } L ^ { 2 } \left( \mathbb { E } [ \| \zeta _ { s , i } ^ { t , j - 1 } \| ^ { 2 } ] + 3 \beta ^ { 2 } \eta ^ { 2 } \sigma ^ { 2 } + 1 2 \eta ^ { 2 } L ^ { 2 } \mathbb { E } [ \| x ^ { t - 1 } - x ^ { t } \| ^ { 2 } + \| x ^ { t } - x _ { s , i } ^ { t , j - 1 } \| ^ { 2 } ] \right) . } \end{array}
$$

Fix $k \in \{ 1 , \ldots , K - 1 \}$ . For every $j \in \left\{ 1 , \ldots , k - 1 \right\}$ , Young's inequality and $\eta L \leq K ^ { - 1 } \leq ( k + 1 ) ^ { - 1 }$ give

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ \| \zeta _ { s , i } ^ { t , j } \| ^ { 2 } ] \leq ( 1 + \frac { 1 } { k } ) \mathbb { E } [ \| \zeta _ { s , i } ^ { t , j - 1 } \| ^ { 2 } ] + ( 1 + k ) \mathbb { E } [ \| \zeta _ { s , i } ^ { t , j } - \zeta _ { s , i } ^ { t , j - 1 } \| ^ { 2 } ] } \\ { \leq ( 1 + \frac { 2 } { k } ) \mathbb { E } [ \| \zeta _ { s , i } ^ { t , j - 1 } \| ^ { 2 } ] + ( 1 + k ) \eta ^ { 2 } L ^ { 2 } \left( 3 \beta ^ { 2 } \eta ^ { 2 } \sigma ^ { 2 } + 1 2 \eta ^ { 2 } L ^ { 2 } \mathbb { E } [ \| x ^ { t - 1 } - x ^ { t } \| ^ { 2 } + \| x ^ { t } - x _ { s , i } ^ { t , j - 1 } \| ^ { 2 } ] \right) } \end{array}
$$

$$
\begin{array} { r l } & { \quad \mathrm { ( 1 ) } \quad \mathrm { ( 1 ) } \quad \mathrm { ( 2 ) } \quad \mathrm { ( 1 ) } \quad \mathrm { ( 2 ) } \quad \mathrm { ( 3 ) } \quad \mathrm { ( 2 ) } \mathrm { ( 1 ) } \quad \mathrm { ( 3 ) } \mathrm { ( 2 ) } \quad \mathrm { ( 2 ) } \mathrm { ( 1 ) } \quad \mathrm { ( 2 ) } \mathrm { ( 1 ) } \quad \mathrm { ( 2 ) } \mathrm { ( 2 ) } \quad \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \quad \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \quad \mathrm { ( 3 ) } \mathrm { ( 2 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } }  \\ &  \quad \mathrm { ( 1 ) } \quad \mathrm { ( 1 ) } \quad \mathrm { ( 2 ) } \mathrm { ( 1 ) } \quad \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm { ( 2 ) } \mathrm { ( 1 ) } \mathrm { ( 2 ) } \mathrm   \end{array}
$$

where the recursion is unrolled only over the index $j$ (with k fixed), and we use $1 + k \le 2 k$ , and $\left( 1 + \frac { 2 } { k } \right) ^ { k } \leq e ^ { 2 }$ . Define

$$
\Delta _ { j } : = x _ { s , i } ^ { t , j + 1 } - x _ { s , i } ^ { t , j } , \qquad \varepsilon _ { j } : = \Delta _ { j } - \zeta _ { s , i } ^ { t , j } , \qquad 0 \leq j \leq k - 1 .
$$

Then $\varepsilon _ { j }$ is $\mathcal { F } ^ { t , j + 1 }$ -measurable and $\mathbb { E } [ \varepsilon _ { j } \mid \mathcal { F } ^ { t , j } ] = 0$ . If $0 \leq j < \ell \leq k - 1$ , then $\varepsilon _ { j }$ is $\mathcal { F } ^ { t , \ell }$ -measurable, and hence the tower property yields the martingale orthogonality relation

$$
{  { \mathbb E } } \langle \varepsilon _ { j } , \varepsilon _ { \ell } \rangle = {  { \mathbb E } } \Big [ {  { \mathbb E } } \big [ \langle \varepsilon _ { j } , \varepsilon _ { \ell } \rangle \mid {  { \mathcal F } } ^ { t , \ell } \big ] \Big ] = {  { \mathbb E } } \Big [ \Big \langle \varepsilon _ { j } , {  { \mathbb E } } [ \varepsilon _ { \ell } \mid {  { \mathcal F } } ^ { t , \ell } ] \Big \rangle \Big ] = 0 .
$$

Consequently,

$$
\begin{array} { r l } { \mathbb { E } [ \| x _ { s , i } ^ { k , k } - x ^ { * } \| ^ { 2 } ] = } & { \mathbb { E } \left[ \left\| \displaystyle \sum _ { j = 0 } ^ { k - 1 } ( \zeta _ { s , j } ^ { i , j } + \xi _ { j } ) \right\| ^ { 2 } \right] } \\ & { \leq 2 \mathbb { E } \left[ \left\| \displaystyle \sum _ { j = 0 } ^ { k - 1 } \zeta _ { s , j } ^ { i , j } \right\| ^ { 2 } \right] + 2 \mathbb { E } \left[ \left\| \displaystyle \sum _ { j = 0 } ^ { k - 1 } \varepsilon _ { j } \right\| ^ { 2 } \right] } \\ & { = 2 \mathbb { E } \left[ \left\| \displaystyle \sum _ { j = 0 } ^ { k - 1 } \zeta _ { s , j } ^ { i , j } \right\| ^ { 2 } \right] + 2 \frac { k - 1 } { j = 0 } \mathbb { E } \| \| \varepsilon _ { j } \| ^ { 2 } ] } \\ & { = 2 \mathbb { E } \left[ \left\| \displaystyle \sum _ { j = 0 } ^ { k - 1 } \zeta _ { s , j } ^ { i , j } \right\| ^ { 2 } \right] + 2 \sum _ { j = 0 } ^ { k - 1 } \mathbb { E } \left[ \mathrm { V a r } ( \Delta _ { j } \mid \mathcal { F } ^ { i , j } ) \right] } \end{array}
$$

$$
\leq 2 k \sum _ { j = 0 } ^ { k - 1 } \mathbb { E } [ \| \zeta _ { s , i } ^ { t , j } \| ^ { 2 } ] + 2 \sum _ { j = 0 } ^ { k - 1 } \left( 3 \beta ^ { 2 } \eta ^ { 2 } \sigma ^ { 2 } + 1 2 \eta ^ { 2 } L ^ { 2 } \mathbb { E } [ \| x ^ { t - 1 } - x ^ { t } \| ^ { 2 } + \| x ^ { t } - x _ { s , i } ^ { t , j } \| ^ { 2 } ] \right) ,\tag{4.3}
$$

where the last line uses Cauchy-Schwarz and the conditional-variance bound proved above. By combining (4.2) and (4.3),

$$
\begin{array} { r l r } {  { \mathbb { E } [ \| x _ { s , i } ^ { t , k } - x ^ { t } \| ^ { 2 } ] } } \\ & { \le 2 k \sum _ { j = 0 } ^ { k - 1 } \Bigg ( e ^ { 2 } \mathbb { E } [ \| \zeta _ { s , i } ^ { t , 0 } \| ^ { 2 } ] + 1 2 e ^ { 2 } k ( \eta L ) ^ { 4 } \sum _ { j ^ { \prime } = 0 } ^ { j - 1 } \mathbb { E } [ \| x _ { s , i } ^ { t , j ^ { \prime } } - x ^ { t } \| ^ { 2 } ] + 8 k ^ { 2 } L ^ { 2 } \eta ^ { 4 } ( 3 \beta ^ { 2 } \sigma ^ { 2 } + 1 2 L ^ { 2 } \mathbb { E } [ \| x ^ { t } - x ^ { t - 1 } \| ^ { 2 } ] ) \Bigg ) } \\ & { } & { + 2 \sum _ { j = 0 } ^ { k - 1 } ( 3 \beta ^ { 2 } \eta ^ { 2 } \sigma ^ { 2 } + 1 2 \eta ^ { 2 } L ^ { 2 } \mathbb { E } [ \| x ^ { t - 1 } - x ^ { t } \| ^ { 2 } + \| x ^ { t } - x _ { s , i } ^ { t , j } \| ^ { 2 } ] ) . } \end{array}
$$

Summing (4.4) over $k = 0 , 1 , \ldots , K - 1$ , we have

$$
\begin{array} { r l } & { \frac { \displaystyle \sum _ { k = 0 } ^ { K - 1 } \| E \| | z _ { s - } ^ { t , k } - z ^ { s } \| | ^ { 2 } } { \displaystyle \sum _ { k = 0 } ^ { K - 1 } \| E \| | z _ { s - } ^ { t , k } - z ^ { s } \| | ^ { 2 } } ; } \\ & { \le \displaystyle \sum _ { k = 0 } ^ { K - 1 } \| \hat { z } _ { s - } ^ { t , k } - z _ { 0 } ^ { \mathrm { E } } \| | \hat { z } _ { s - } ^ { t , k } - z _ { 0 } ^ { s } \| ^ { 2 } \sum _ { k = 0 } ^ { K - 1 } \| z _ { s - } ^ { t , k } - z _ { 0 } ^ { s } \| ^ { 2 } \sum _ { k = 0 } ^ { K - 1 } z _ { 0 } ^ { s , k } - z _ { 1 } ^ { s } \| ^ { 2 } ; } \\ &  \le \displaystyle \sum _ { k = 0 } ^ { K - 1 } \| \hat { z } _ { s - } ^ { t , k } - z _ { 0 } ^ { s } \| \leq \frac { \sum _ { k = 0 } ^ { K - 1 } \| z _ { s - } ^ { t , k } - z _ { 0 } ^ { s } \| \leq \frac { \sum _ { k = 0 } ^ { K - 1 } \| z _ { s - } ^ { t , k } - z _ { 1 } ^ { s } \| ^ { 2 } } { ( \widehat { z } _ { s - } ^ { t , k } - z _ { 1 } ^ { t , k } ) ^ { 2 } } ; } \\ &  \qquad \times \displaystyle \sum _ { k = 0 } ^ { K - 1 } \| \sum _ { k = 0 } ^ { K - 1 } \| \hat { z } _ { s - } ^ { t , k } ( 3 ) \sigma _ { \sigma } ^ { 2 } \sigma ^ { 2 } \cdot 1 2 L ^ { 2 } \mathbb { L } \| z _ { s } ^ { t , k } - z _ { 1 } ^ { t , k } \| ^ { 2 } ) \} \quad \underbrace  \sum _ { k = 0 } ^ { K - 1 } \sum _ { k = 0 } ^ { K - 1 } \big ( 3 ) ^ { 2 \nu } \sigma ^ { 2 } \sigma ^ { 2 } \cdot 1 2 \sigma ^ { 2 } \end{array}
$$

For $\textcircled{1}$ , we have

$$
\sum _ { k = 0 } ^ { K - 1 } 2 k \sum _ { j = 0 } ^ { k - 1 } e ^ { 2 } \mathbb { E } [ \| \zeta _ { s , i } ^ { t , 0 } \| ^ { 2 } ] = \sum _ { k = 0 } ^ { K - 1 } 2 k ^ { 2 } e ^ { 2 } \mathbb { E } [ \| \zeta _ { s , i } ^ { t , 0 } \| ^ { 2 } ] \le 2 K ^ { 3 } e ^ { 2 } \mathbb { E } [ \| \zeta _ { s , i } ^ { t , 0 } \| ^ { 2 } ] .
$$

For $\textcircled{2}$ , we have

$$
\begin{array} { r l } & { \displaystyle \sum _ { k = 0 } ^ { K - 1 } \sum _ { k } ^ { k - 1 } 1 2 e ^ { 2 } k ( \eta L ) ^ { i } \overset { \_ { j } - 1 } { \underset { j = 0 } { \longrightarrow } } \mathbb { E } [ \| x _ { s , i } ^ { t , j ^ { \prime } } - x ^ { t } \| ^ { 2 } ] } \\ & { \displaystyle = \sum _ { k = 0 } ^ { K - 1 } 2 4 e ^ { 2 } k ^ { 2 } ( \eta L ) ^ { i } \overset { \_ { k - 1 } } { \underset { j = 0 } { \longrightarrow } } \big ( \mathbb { E } [ \| x _ { s , i } ^ { t , 0 } - x ^ { t } \| ^ { 2 } ] + \mathbb { E } [ \| x _ { s , i } ^ { t , 1 } - x ^ { t } \| ^ { 2 } ] + \dots + \mathbb { E } [ \| x _ { s , i } ^ { t , j - 1 } - x ^ { t } \| ^ { 2 } ] \big ) } \\ & { \displaystyle = \sum _ { k = 0 } ^ { K - 1 } 2 4 e ^ { 2 } k ^ { 2 } ( \eta L ) ^ { i } \big ( \mathbb { E } [ \| x _ { s , i } ^ { t , 0 } - x ^ { t } \| ^ { 2 } ] + \big ( \mathbb { E } [ \| x _ { s , i } ^ { t , 0 } - x ^ { t } \| ^ { 2 } ] + \mathbb { E } [ \| x _ { s , i } ^ { t , 1 } - x ^ { t } \| ^ { 2 } ] \big ) } \\ & { \displaystyle \quad + \dots + \big ( \mathbb { E } [ \| x _ { s , i } ^ { t , 1 } - x ^ { t } \| ^ { 2 } ] + \dots + \mathbb { E } [ \| x _ { s , i } ^ { t , k - 2 } - x ^ { t } \| ^ { 2 } ] \big ) \big ) } \\ &  \displaystyle \quad \dots + 4 e ^ { 2 } k ^ { 2 } ( \eta L ) ^ { 4 } \Big ( ( k - 2 ) \mathbb { E } [ \| x _ { s , i } ^ { t , 1 } - x ^ { t } \| ^ { 2 } ] + ( k - 3 ) \mathbb { E } [ \| x _ { s , i } ^ { t , 2 } - x ^ { t } \| ^ { 2 } ] + \dots + \mathbb  \end{array}
$$

$$
\begin{array} { r l } & { \le \displaystyle \sum _ { k = 0 } ^ { K } 2 4 4 e ^ { \lambda } t ^ { 2 } ( \mu \mathbb { E } _ { \lambda } ^ { t } \| \hat { s } _ { \xi , \xi } ^ { t _ { 1 } } - z ^ { t } \| ^ { 2 } \| - 1 6 \mathbb { E } \| \| s _ { \xi , \xi } ^ { t _ { 2 } } - z ^ { t } \| ^ { 2 } ] + \cdots + \mathbb { E } \mathbb { E } \| z _ { \lambda } ^ { t _ { 1 } } \| ^ { t _ { 2 } } e ^ { \lambda } - z ^ { t } \| ^ { 2 } \| } \\ & { \le - 2 4 e ^ { \lambda } ( \eta / t ) ^ { 4 } \displaystyle \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } ^ { \lambda } \Big ( \mathbb { E } \| z _ { \xi , \xi } ^ { t _ { 1 } } - x ^ { t } \| ^ { 2 } \Big ) + \mathbb { E } \mathbb { E } \| \| x _ { \xi , \xi } ^ { t _ { 2 } } - z ^ { t } \| ^ { 2 } \Big ) + \cdots + \mathbb { E } \mathbb { E } \mathbb { E } \| z _ { \xi , \xi } ^ { t _ { 1 } - \lambda - 2 } - z ^ { t } \| ^ { 2 } \Big ) } \\ & { = 2 4 e ^ { \lambda } ( \eta / t ) ^ { 4 } \left( 3 e ^ { \lambda } - z ^ { t } \right) \| ^ { 2 } \Big | } \\ & { \quad = 2 4 e ^ { \lambda } ( \eta / t ) ^ { 4 } \left( 3 e ^ { \lambda } - z ^ { t } \right) ^ { 2 } \Big | 2 + 4 \mathbb { S } \left( \mathbb { E } \| \| x _ { \xi , \xi } ^ { t _ { 1 } } - x ^ { t } \| ^ { 2 } \right) + 2 4 e ^ { \lambda } ( \mathbb { E } \| \| x _ { \xi , \xi } ^ { t _ { 2 } } - x ^ { t } \| ^ { 2 } ) + 1 5 \mathbb { E } \| x _ { \xi , \xi } ^ { t _ { 2 } } - x ^ { t } \| ^ { 2 } \Big | } \\ &  \quad \quad + \cdots + ( K - 1 ) ^ { 3 } \Big \{ \mathbb { E }  \end{array}
$$

For $\textcircled{3}$ , we have

$$
\begin{array} { r l } & { \quad \displaystyle \sum _ { k = 0 } ^ { K - 1 } 2 k \sum _ { j = 0 } ^ { k - 1 } 8 k ^ { 2 } L ^ { 2 } \eta ^ { 4 } \left( 3 \beta ^ { 2 } \sigma ^ { 2 } + 1 2 L ^ { 2 } \mathbb { E } [ \| x ^ { t } - x ^ { t - 1 } \| ^ { 2 } ] \right) } \\ & { = \displaystyle \sum _ { k = 0 } ^ { K - 1 } 1 6 k ^ { 4 } L ^ { 2 } \eta ^ { 4 } \left( 3 \beta ^ { 2 } \sigma ^ { 2 } + 1 2 L ^ { 2 } \mathbb { E } [ \| x ^ { t } - x ^ { t - 1 } \| ^ { 2 } ] \right) } \\ & { \le 1 6 K ^ { 5 } L ^ { 2 } \eta ^ { 4 } \left( 3 \beta ^ { 2 } \sigma ^ { 2 } + 1 2 L ^ { 2 } \mathbb { E } [ \| x ^ { t } - x ^ { t - 1 } \| ^ { 2 } ] \right) . } \end{array}
$$

For $\textcircled{4}$ , we have

$$
\begin{array} { r l } & { \quad \displaystyle \sum _ { k = 0 } ^ { K - 1 } 2 \sum _ { j = 0 } ^ { k - 1 } \left( 3 \beta ^ { 2 } \eta ^ { 2 } \sigma ^ { 2 } + 1 2 \eta ^ { 2 } L ^ { 2 } \mathbb { E } [ \| x ^ { t - 1 } - x ^ { t } \| ^ { 2 } ] \right) } \\ & { \quad \displaystyle = \sum _ { k = 0 } ^ { K - 1 } 2 k \left( 3 \beta ^ { 2 } \eta ^ { 2 } \sigma ^ { 2 } + 1 2 \eta ^ { 2 } L ^ { 2 } \mathbb { E } [ \| x ^ { t - 1 } - x ^ { t } \| ^ { 2 } ] \right) } \\ & { \quad \le 2 K ^ { 2 } \left( 3 \beta ^ { 2 } \eta ^ { 2 } \sigma ^ { 2 } + 1 2 \eta ^ { 2 } L ^ { 2 } \mathbb { E } [ \| x ^ { t - 1 } - x ^ { t } \| ^ { 2 } ] \right) . } \end{array}
$$

For $\textcircled{5}$ , we have

$$
\begin{array} { r l } & { \quad \displaystyle \sum _ { k = 0 } ^ { K - 1 } 2 \sum _ { j = 0 } ^ { k - 1 } 1 2 \eta ^ { 2 } L ^ { 2 } \mathbb { E } [ \| x ^ { t } - x _ { s , j } ^ { t } \| ^ { 2 } ] } \\ & { \quad \displaystyle \sum _ { k = 0 } ^ { K - 1 } 2 4 \eta ^ { 2 } L ^ { 2 } \left( \mathbb { E } [ \| x ^ { t } - x _ { s , k } ^ { t + 1 } \| ^ { 2 } ] + \cdots + \mathbb { E } [ \| x ^ { t } - x _ { s , i } ^ { t - 1 } \| ^ { 2 } ] \right) } \\ & { \quad \displaystyle - \sum _ { k = 0 } ^ { K - 1 } 2 4 \eta ^ { 2 } L ^ { 2 } \left( \mathbb { E } [ \| x ^ { t } - x _ { s , k } ^ { t + 1 } \| ^ { 2 } ] + \left( \mathbb { E } [ \| x ^ { t } - x _ { s , i } ^ { t + 1 } \| ^ { 2 } ] + \mathbb { E } [ \| x ^ { t } - x _ { s , i } ^ { t , 2 } \| ^ { 2 } ] \right) \right. } \\ & { \quad \left. \quad + \cdots + \left( \mathbb { E } [ \| x ^ { t } - x _ { s , i } ^ { t + 1 } \| ^ { 2 } ] + \cdots + \mathbb { E } [ \| x ^ { t } - x _ { s , i } ^ { t - 1 } \| ^ { 2 } ] + \mathbb { E } [ \| x ^ { t } - x _ { s , i } ^ { t , 2 } \| ^ { 2 } ] \right) \right) } \\ & { \quad \displaystyle - 2 4 \eta ^ { 2 } L ^ { 2 } \left( ( K - 2 ) \mathbb { E } [ \| x ^ { t } - x _ { s , i } ^ { t - 1 } \| ^ { 2 } ] + ( K - 3 ) \mathbb { E } [ \| x ^ { t } - x _ { s , i } ^ { t - 2 } \| ^ { 2 } ] + \cdots + \mathbb { E } [ \| x ^ { t } - x _ { s , i } ^ { t , K - 2 } \| ^ { 2 } ] \right) } \\ &  \quad \displaystyle \leq 2 4 \eta ^ { 2 } L ^ { 2 } \mathbb { K } \displaystyle \sum _ { k = 0 } ^  K - 1 \end{array}
$$

Combining $\textcircled{1} - \textcircled{5}$ yields

$$
\begin{array} { r l } {  { \sum _ { k = 0 } ^ { K - 1 } \exp ( \exp ( - \varepsilon ) \exp ( \exp x - \varepsilon ) \| ^ { 2 } ) } } \\ & { \overset { K \to \varepsilon } { \underset { \mathrm { t e n d } } { \sum } } \| \| x _ { \varepsilon , \delta } ^ { ( k , \varepsilon ) } - x ^ { [ 2 ] } \| ^ { 2 } } \\ & { \leq 2 \mathcal { K } ^ { \delta } e ^ { 2 } \| x \| _ { \mathcal { S } _ { \varepsilon } } ^ { \varepsilon } \| \| _ { \mathcal { S } _ { \varepsilon } } ^ { 2 } \| 1 + 2 4 \varepsilon ^ { 2 } \langle \partial _ { t } L _ { \varepsilon } ^ { 4 } \rangle K \frac { \varepsilon } { 6 + 1 } \sum _ { k = 0 } ^ { K } \mathbb { E } \| \| x _ { \varepsilon } ^ { ( k , \varepsilon ) } - x ^ { [ 2 ] } \| ^ { 2 } \big \vert + 1 6 K ^ { ^ { 2 } } L _ { f } ^ { 2 } u ^ { 4 } ( 3 \lambda ^ { 2 } \sigma ^ { 2 } + 1 2 L ^ { 2 } \mathbb { E } \| \| x ^ { 4 } - x ^ { 4 } \| ^ { 2 } ) \big \rangle } \\ & { \leq 2 K ^ { 2 } \delta ^ { 2 } e ^ { 2 } \| \partial _ { t } ^ { 2 } \sigma ^ { 2 } + 1 2 \sigma ^ { 2 } \langle \partial _ { t } L _ { \varepsilon } ^ { 3 } \| ^ { 2 } x ^ { 4 - 1 } - x ^ { [ 4 ] } \rangle \big \vert ^ { 2 } \big ) + 2 4 \eta ^ { 2 } \mathcal { L } _ { F } ^ { 2 } K \frac { \varepsilon } { 6 + 1 } \| x ^ { 3 - } x ^ { [ 6 ] } \| ^ { 2 } } \\ &  = 2 K ^ { 3 } e ^ { 2 } \| \| \tilde { \mathcal { S } } ^ { \varepsilon } \| _ { \mathcal { S } _ { \varepsilon } } ^ { 2 } \| ^ { 2 } + ( 4 K ^ { 8 } \pi ^ { 9 } t ^ { 2 } t ^ { 2 } + 6 K ^ { 2 } \sigma ^ { 2 } ) ( \beta ^ { 2 } \sigma ^ { 2 } + 4 L ^ { 2 } \mathbb  \end{array}
$$

where the last inequality follows from $2 4 e ^ { 2 } ( \eta L ) ^ { 4 } K ^ { 4 } + 2 4 \eta ^ { 2 } L ^ { 2 } K \le \frac { 1 } { 2 }$ . Thus, we obtain

$$
\sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ \| x _ { s , i } ^ { t , k } - x ^ { t } \| ^ { 2 } ] \le 4 K ^ { 3 } e ^ { 2 } \mathbb { E } [ \| \zeta _ { s , i } ^ { t , 0 } \| ^ { 2 } ] + 1 2 ( \eta K ) ^ { 2 } ( 8 K ^ { 3 } \eta ^ { 2 } L ^ { 2 } + 1 ) \left( \beta ^ { 2 } \sigma ^ { 2 } + 4 L ^ { 2 } \mathbb { E } [ \| x ^ { t } - x ^ { t - 1 } \| ^ { 2 } ] \right) ,
$$

Finally, summing over $i = 1 , \dots , M .$ dividing by KM, using the definitions of $U _ { s } ^ { t }$ and $\Xi _ { s } ^ { t }$ , and substituting $x ^ { t } - x ^ { t - 1 } = \gamma g ^ { t - 1 }$ , the conclusion follows.

Lemma 4.5. Suppose that Assumptions 2.1 and 4.1 hold. If max $\{ 1 1 5 2 e ^ { 2 } ( 1 - \beta ) ^ { 2 } \left( \eta K L \right) ^ { 2 } , 1 1 5 2 e ^ { 2 } ( 1 -$ $\beta ) ^ { 2 } \left( \eta \gamma K L ^ { 2 } \right) ^ { 2 } \} \le \beta ^ { 2 }$ , then the following inequality holds:

$$
\sum _ { t = 0 } ^ { T - 1 } \Xi _ { s } ^ { t } \le \frac { \beta ^ { 2 } } { 2 8 8 e ^ { 2 } K ^ { 2 } L ^ { 2 } } \sum _ { t = 0 } ^ { T - 1 } \left( \mathcal { E } _ { s } ^ { t - 1 } + \mathbb { E } [ \| g ^ { t - 1 } \| ^ { 2 } ] \right) + 4 \eta ^ { 2 } T D ^ { 2 } .
$$

Proof. Recall that $\zeta _ { s , i } ^ { t , 0 } = - \eta \left( \left( 1 - \beta \right) \left( g _ { s } ^ { t - 1 } - \nabla f _ { s , i } ( x ^ { t - 1 } ) \right) + \nabla f _ { s , i } ( x ^ { t } ) \right)$ . Consequently,

$$
\begin{array} { r l } & { \| \zeta _ { s , i } ^ { t , 0 } \| ^ { 2 } \leq 2 \eta ^ { 2 } \left( ( 1 - \beta ) ^ { 2 } \| g _ { s } ^ { t - 1 } \| ^ { 2 } + \| \nabla f _ { s , i } ( x ^ { t } ) - ( 1 - \beta ) \nabla f _ { s , i } ( x ^ { t - 1 } ) \| ^ { 2 } \right) } \\ & { \qquad = 2 \eta ^ { 2 } ( 1 - \beta ) ^ { 2 } \| g _ { s } ^ { t - 1 } \| ^ { 2 } + 2 \eta ^ { 2 } \left\| ( 1 - \beta ) \left( \nabla f _ { s , i } ( x ^ { t } ) - \nabla f _ { s , i } ( x ^ { t - 1 } ) \right) + \beta \nabla f _ { s , i } ( x ^ { t } ) \right\| ^ { 2 } } \\ & { \qquad \leq 2 \eta ^ { 2 } ( 1 - \beta ) ^ { 2 } \| g _ { s } ^ { t - 1 } \| ^ { 2 } + 4 \eta ^ { 2 } ( 1 - \beta ) ^ { 2 } \| \nabla f _ { s , i } ( x ^ { t } ) - \nabla f _ { s , i } ( x ^ { t - 1 } ) \| ^ { 2 } + 4 ( \eta \beta ) ^ { 2 } \| \nabla f _ { s , i } ( x ^ { t } ) \| ^ { 2 } } \\ & { \qquad \leq 2 \eta ^ { 2 } ( 1 - \beta ) ^ { 2 } \| g _ { s } ^ { t - 1 } \| ^ { 2 } + 4 ( \eta L ) ^ { 2 } ( 1 - \beta ) ^ { 2 } \| x ^ { t } - x ^ { t - 1 } \| ^ { 2 } + 4 ( \eta \beta ) ^ { 2 } \| \nabla f _ { s , i } ( x ^ { t } ) \| ^ { 2 } , } \end{array}
$$

where the last inequality follows from Assumption 2.1. By Assumption 4.1 and the definition of $\Xi _ { s } ^ { t }$ and summing over $t = 0 , \ldots , T - 1$ , we have

$$
\begin{array} { r l } {  { \sum _ { t = 0 } ^ { T - 1 } \Xi _ { s } ^ { t } = \sum _ { t = 0 } ^ { T - 1 } \displaystyle \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } [ \| \zeta _ { s , i } ^ { t , 0 } \| ^ { 2 } ] } } \\ & { \leq \sum _ { t = 0 } ^ { T - 1 } \sum _ { \ell = 0 } ^ { T - 1 } ( 2 \eta ^ { 2 } ( 1 - \beta ) ^ { 2 } \| g _ { s } ^ { t - 1 } \| ^ { 2 } + 4 ( \eta L ) ^ { 2 } ( 1 - \beta ) ^ { 2 } \| x ^ { t } - x ^ { t - 1 } \| ^ { 2 } + 4 ( \eta \beta ) ^ { 2 } \displaystyle \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \| \nabla f _ { s , i } ( x ^ { t } ) \| ^ { 2 } ) } \\ & { \leq \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \Bigg ( 4 \eta ^ { 2 } ( 1 - \beta ) ^ { 2 } ( \| g _ { s } ^ { t - 1 } - \nabla f _ { s } ( x ^ { t - 1 } ) \| ^ { 2 } + \| \nabla f _ { s } ( x ^ { t - 1 } ) \| ^ { 2 } ) + 4 ( \eta L ) ^ { 2 } ( 1 - \beta ) ^ { 2 } \| x ^ { t } - x ^ { t - 1 } \| ^ { 2 } } \\ & { \quad + 4 ( \eta \beta ) ^ { 2 } \displaystyle \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \| \nabla f _ { s , i } ( x ^ { t } ) \| ^ { 2 } \Bigg ) } \end{array}
$$

$$
\begin{array} { r l } & { \displaystyle \le 4 \eta ^ { 2 } ( 1 - \beta ) ^ { 2 } \sum _ { t = 0 } ^ { T - 1 } \xi _ { s } ^ { t - 1 } + 4 \eta ^ { 2 } ( 1 - \beta ) ^ { 2 } T D ^ { 2 } + 4 \eta ^ { 2 } ( 1 - \beta ) ^ { 2 } ( \gamma L ) ^ { 2 } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \| g ^ { t - 1 } \| ^ { 2 } ] + 4 \eta ^ { 2 } \beta ^ { 2 } T D ^ { 2 } } \\ & { \displaystyle = 4 \eta ^ { 2 } ( 1 - \beta ) ^ { 2 } \sum _ { t = 0 } ^ { T - 1 } \xi _ { s } ^ { t - 1 } + 4 \eta ^ { 2 } ( 1 - \beta ) ^ { 2 } ( \gamma L ) ^ { 2 } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \| g ^ { t - 1 } \| ^ { 2 } ] + 4 ( 1 - 2 \beta + 2 \beta ^ { 2 } ) \eta ^ { 2 } T D ^ { 2 } } \\ & { \displaystyle \le \frac { \beta ^ { 2 } } { 2 8 8 e ^ { 2 } K ^ { 2 } L ^ { 2 } } \sum _ { t = 0 } ^ { T - 1 } \xi _ { s } ^ { t - 1 } + \frac { \beta ^ { 2 } } { 2 8 8 e ^ { 2 } K ^ { 2 } L ^ { 2 } } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \| g ^ { t - 1 } \| ^ { 2 } ] + 4 \eta ^ { 2 } T D ^ { 2 } , } \end{array}
$$

where the last inequality follows from max $\{ 1 1 5 2 e ^ { 2 } ( 1 - \beta ) ^ { 2 } \left( \eta K L \right) ^ { 2 } , 1 1 5 2 e ^ { 2 } ( 1 - \beta ) ^ { 2 } \left( \eta \gamma K L ^ { 2 } \right) ^ { 2 } \} \le$ $\beta ^ { 2 }$

In the following theorem, we establish the convergence rate of the proposed algorithm.

Theorem 4.1. Suppose that Assumptions $\it { 2 . 1 - 2 . 2 }$ and 4.1 hold, and that $f _ { s } ( x )$ has a lower bound $f _ { s } ^ { \mathrm { m i n } }$ for all $\boldsymbol { x } \in \mathbb { R } ^ { d }$ . Let $\begin{array} { r } { \Delta : = \operatorname* { m a x } _ { s \in [ S ] } ( f _ { s } ( x ^ { 0 } ) - f _ { s } ^ { \operatorname* { m i n } } ) } \end{array}$ . If we set

$$
\beta = \left( { \frac { M K L ^ { 2 } \left( 1 + D ^ { 2 } \right) ^ { 2 } \Delta ^ { 2 } } { \sigma ^ { 4 } T ^ { 2 } } } \right) ^ { 1 / 3 } < 1 , \quad \gamma = \operatorname* { m i n } \left\{ { \frac { 1 } { 2 L } } , { \frac { 1 } { L } } { \sqrt { \frac { \beta K M } { 5 4 } } } \right\} ,
$$

$$
\eta K L \leq c _ { \eta } \operatorname * { m i n } \left\{ \beta , \frac { 1 } { \sqrt { K } } , \sqrt { \frac { \beta } { M } } , \beta ^ { 3 / 4 } \left( \frac { L \Delta } { ( K M ) ^ { 1 / 2 } T } \right) ^ { 1 / 2 } \right\} ,
$$

where $c _ { \eta } > 0$ is a sufficiently small absolute constant, and choose $\begin{array} { r } { B = \left\lceil \frac { K } { T \beta ^ { 2 } } \right\rceil } \end{array}$ , then Algorithm 1 admits the convergence guarantee

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left[ \mathcal { G } ( x _ { a } ) \right] \lesssim \left( \frac { L \left( 1 + D ^ { 2 } \right) \sigma \Delta } { K M T } \right) ^ { \frac 2 3 } ( 1 + S ^ { 2 } ) + \frac { L S ^ { 2 } \Delta } { T } } \\ { \displaystyle \quad \quad + \frac { ( 1 + S ^ { 2 } ) G _ { 0 } ^ { 2 } } { T } + \frac { ( 1 + S ^ { 2 } ) \sigma ^ { 2 } } { K M } \left( \frac { M K L ^ { 2 } ( 1 + D ^ { 2 } ) ^ { 2 } \Delta ^ { 2 } } { \sigma ^ { 4 } T ^ { 2 } } \right) ^ { 2 / 3 } , } \end{array}
$$

where $\begin{array} { r } { G _ { 0 } ^ { 2 } : = \frac { 1 } { S } \sum _ { s \in [ S ] } \left\| \nabla f _ { s } ( x ^ { 0 } ) \right\| ^ { 2 } } \end{array}$ . Alternatively, if $B = \Theta ( K T )$ and $\begin{array} { r } { \beta = \operatorname* { m a x } \{ \frac { 1 } { T } ,  ( \frac { M K L ^ { 2 } ( 1 + D ^ { 2 } ) ^ { 2 } \Delta ^ { 2 } } { \sigma ^ { 4 } T ^ { 2 } } ) ^ { 1 / 3 } \} } \end{array}$ then Algorithm 1 admits the convergence guarantee

$$
\begin{array} { r l r } & { } & { \mathbb { E } \left[ \mathcal { G } ( x _ { a } ) \right] \lesssim \frac { \sigma ^ { 2 } ( 1 + S ^ { 2 } ) } { K M T } + \left( \frac { L \left( 1 + D ^ { 2 } \right) \sigma \Delta } { K M T } \right) ^ { \frac { 2 } { 3 } } ( 1 + S ^ { 2 } ) } \\ & { } & { \quad + \frac { L S ^ { 2 } \Delta } { T } + \frac { ( 1 + S ^ { 2 } ) G _ { 0 } ^ { 2 } } { T } + \frac { ( 1 + S ^ { 2 } ) \sigma ^ { 2 } } { K M T ^ { 2 } } . \quad } \end{array}
$$

Proof. With the above choice of parameters, the conditions required in Lemmas 4.3–4.5 are satisfied. Indeed, by the definition of $\gamma ,$ we have

$$
\gamma \leq \frac { 1 } { 2 L } , \qquad \gamma L \leq \sqrt { \frac { \beta K M } { 5 4 } } .
$$

Moreover, since $\eta K L \leq c _ { \eta } \beta , \beta \leq c < 1$ , and $K \geq 1$ , by choosing $c _ { \eta } > 0$ sufficiently small, we obtain

$$
\begin{array} { r } { \eta K L \leq 1 , } \end{array}
$$

and

$$
2 4 e ^ { 2 } ( \eta L ) ^ { 4 } K ^ { 4 } + 2 4 \eta ^ { 2 } L ^ { 2 } K = 2 4 e ^ { 2 } ( \eta K L ) ^ { 4 } + 2 4 { \frac { ( \eta K L ) ^ { 2 } } { K } }
$$

$$
\leq 2 4 e ^ { 2 } c _ { \eta } ^ { 4 } + 2 4 c _ { \eta } ^ { 2 } \leq \frac { 1 } { 2 } .
$$

Similarly,

$$
1 1 5 2 e ^ { 2 } ( 1 - \beta ) ^ { 2 } ( \eta K L ) ^ { 2 } \leq 1 1 5 2 e ^ { 2 } c _ { \eta } ^ { 2 } \beta ^ { 2 } \leq \beta ^ { 2 } .
$$

Since $\gamma L \leq 1 / 2$ , we also have

$$
\begin{array} { r } { 1 1 5 2 e ^ { 2 } ( 1 - \beta ) ^ { 2 } ( \eta \gamma K L ^ { 2 } ) ^ { 2 } = 1 1 5 2 e ^ { 2 } ( 1 - \beta ) ^ { 2 } ( \eta K L ) ^ { 2 } ( \gamma L ) ^ { 2 } } \\ { \leq 1 1 5 2 e ^ { 2 } ( 1 - \beta ) ^ { 2 } ( \eta K L ) ^ { 2 } \leq \beta ^ { 2 } . } \end{array}
$$

Therefore, all the conditions in Lemmas 4.3–4.5 hold.

It follows from Lemmas 4.3 and 4.4 that

$$
\begin{array} { r l } & { \mathcal { E } _ { s } ^ { t } \leq ( 1 - \beta ) \mathcal { E } _ { s } ^ { t - 1 } + \frac { 2 \beta } { 9 } \mathbb { E } [ \| g ^ { t - 1 } \| ^ { 2 } ] + \frac { 3 \beta ^ { 2 } \sigma ^ { 2 } } { K M } + \frac { 4 } { \beta } L ^ { 2 } U _ { s } ^ { t } } \\ & { \qquad \leq ( 1 - \beta ) \mathcal { E } _ { s } ^ { t - 1 } + \frac { 2 \beta } { 9 } \mathbb { E } [ \| g ^ { t - 1 } \| ^ { 2 } ] + \frac { 3 \beta ^ { 2 } \sigma ^ { 2 } } { K M } } \\ & { \qquad + \frac { 4 } { \beta } L ^ { 2 } \left( 4 e ^ { 2 } K ^ { 2 } \Xi _ { s } ^ { t } + 1 2 ( \eta K ) ^ { 2 } \left( 8 ( \eta K L ) ^ { 2 } + K ^ { - 1 } \right) \left( \beta ^ { 2 } \sigma ^ { 2 } + 4 ( \gamma L ) ^ { 2 } \mathbb { E } [ \| g ^ { t - 1 } \| ^ { 2 } ] \right) \right) } \\ & { \qquad = ( 1 - \beta ) \mathcal { E } _ { s } ^ { t - 1 } + \frac { 2 \beta } { 9 } \mathbb { E } [ \| g ^ { t - 1 } \| ^ { 2 } ] + \frac { 3 \beta ^ { 2 } \sigma ^ { 2 } } { K M } + \frac { 1 6 ( \epsilon L K L ) ^ { 2 } } { \beta } \underline { { \alpha } } _ { s } ^ { t } + 4 8 \beta ( \eta K L ) ^ { 2 } \left( 8 ( \eta K L ) ^ { 2 } + K ^ { - 1 } \right) \sigma ^ { 2 } } \\ & { \qquad + \frac { 1 9 2 } { \beta } ( \eta K L ) ^ { 2 } \left( 8 ( \eta K L ) ^ { 2 } + K ^ { - 1 } \right) ( \gamma L ) ^ { 2 } \mathbb { E } [ \| g ^ { t - 1 } \| ^ { 2 } ] } \\ &  \qquad \leq ( 1 - \beta ) \mathcal { E } _ { s } ^ { t - 1 } + \frac { 4 \beta ^ { 2 } \sigma ^ { 2 } } { K M } + \frac { 5 \beta } { 1 8 } \mathbb { E } [ \| \end{array}
$$

where the third inequality follows from

$$
\left\{ \begin{array} { l l } { 4 8 \beta ( \eta K L ) ^ { 2 } \left( 8 ( \eta K L ) ^ { 2 } + K ^ { - 1 } \right) \le \displaystyle \frac { \beta ^ { 2 } } { K M } , } \\ { \displaystyle \frac { 1 9 2 ( \eta K L ) ^ { 2 } } { \beta } \left( 8 ( \eta K L ) ^ { 2 } + K ^ { - 1 } \right) ( \gamma L ) ^ { 2 } \le \displaystyle \frac { \beta } { 1 8 } . } \end{array} \right.
$$

Indeed, let $q : = \eta K L$ . The stated stepsize condition gives

$$
q ^ { 2 } \leq c _ { \eta } ^ { 2 } \frac { \beta } { M } , \qquad q ^ { 2 } \leq \frac { c _ { \eta } ^ { 2 } } { K } , \qquad 8 q ^ { 2 } + K ^ { - 1 } \leq \frac { 8 c _ { \eta } ^ { 2 } + 1 } { K } .
$$

Consequently,

$$
4 8 \beta q ^ { 2 } ( 8 q ^ { 2 } + K ^ { - 1 } ) \le 4 8 c _ { \eta } ^ { 2 } ( 8 c _ { \eta } ^ { 2 } + 1 ) \frac { \beta ^ { 2 } } { K M } \le \frac { \beta ^ { 2 } } { K M } ,
$$

provided that $4 8 c _ { \eta } ^ { 2 } ( 8 c _ { \eta } ^ { 2 } + 1 ) \leq 1$ . Moreover, the definition of $\gamma \ \mathrm { g i }$ ives $( \gamma L ) ^ { 2 } \leq \beta K M / 5 4 .$ SO

$$
\begin{array} { r l r } & { } & { \frac { 1 9 2 q ^ { 2 } } { \beta } ( 8 q ^ { 2 } + K ^ { - 1 } ) ( \gamma L ) ^ { 2 } \leq \frac { 3 2 } { 9 } K M q ^ { 2 } ( 8 q ^ { 2 } + K ^ { - 1 } ) } \\ & { } & { \qquad \leq \frac { 3 2 } { 9 } c _ { \eta } ^ { 2 } ( 8 c _ { \eta } ^ { 2 } + 1 ) \beta \leq \frac { \beta } { 1 8 } , } \end{array}
$$

where the last inequality holds if $c _ { \eta } ^ { 2 } ( 8 c _ { \eta } ^ { 2 } + 1 ) \leq 1 / 6 4$ . Thus, by choosing the absolute constant $c _ { \eta } > 0$ sufficiently small, both required inequalities hold.

Summing over t from 0 to $T - 1$ and applying Lemma 4.5, we get

$$
\sum _ { t = 0 } ^ { T - 1 } \mathcal { E } _ { s } ^ { t } \le ( 1 - \beta ) \sum _ { t = 0 } ^ { T - 1 } \mathcal { E } _ { s } ^ { t - 1 } + \frac { 4 \beta ^ { 2 } \sigma ^ { 2 } } { K M } T + \frac { 5 \beta } { 1 8 } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \Vert { g } ^ { t - 1 } \Vert ^ { 2 } ] + \frac { 1 6 ( e L K ) ^ { 2 } } { \beta } \sum _ { t = 0 } ^ { T - 1 } \Xi _ { s } ^ { t }
$$

$$
\begin{array} { r l } & { \leq ( 1 - \beta ) \displaystyle \sum _ { t = 0 } ^ { T - 1 } \xi _ { s } ^ { t - 1 } + \frac { 4 \beta ^ { 2 } \sigma ^ { 2 } } { K M } T + \frac { 5 \beta } { 1 8 } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \| g ^ { t - 1 } \| ^ { 2 } ] } \\ & { \quad + \frac { 1 6 ( e L K ) ^ { 2 } } { \beta } \left( \frac { \beta ^ { 2 } } { 2 8 8 e ^ { 2 } K ^ { 2 } L ^ { 2 } } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \left( \mathcal { E } _ { s } ^ { t - 1 } + \mathbb { E } [ \| g ^ { t - 1 } \| ^ { 2 } ] \right) + 4 \eta ^ { 2 } T D ^ { 2 } \right) } \\ & { = \left( 1 - \frac { 1 7 \beta } { 1 8 } \right) \displaystyle \sum _ { t = - 1 } ^ { T - 2 } \mathcal { E } _ { s } ^ { t } + \frac { \beta } { 3 } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \| g ^ { t - 1 } \| ^ { 2 } ] + \frac { 4 \beta ^ { 2 } \sigma ^ { 2 } } { K M } T + \frac { 6 4 } { \beta } ( e \eta K L ) ^ { 2 } T D ^ { 2 } . } \end{array}
$$

Thus,

$$
\frac { 1 7 \beta } { 1 8 } \sum _ { t = - 1 } ^ { T - 2 } \mathcal { E } _ { s } ^ { t } \le \mathcal { E } _ { s } ^ { - 1 } - \mathcal { E } _ { s } ^ { T - 1 } + \frac { \beta } { 3 } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \Vert g ^ { t - 1 } \Vert ^ { 2 } ] + \frac { 4 \beta ^ { 2 } \sigma ^ { 2 } } { K M } T + \frac { 6 4 } { \beta } ( e \eta K L ) ^ { 2 } T D ^ { 2 } ,
$$

which implies that

$$
\sum _ { t = - 1 } ^ { T - 2 } \mathcal { E } _ { s } ^ { t } \leq \frac { 1 8 } { 1 7 \beta } \mathcal { E } _ { s } ^ { - 1 } - \frac { 1 8 } { 1 7 \beta } \mathcal { E } _ { s } ^ { T - 1 } + \frac { 6 } { 1 7 } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \Vert g ^ { t - 1 } \Vert ^ { 2 } ] + \frac { 7 2 \beta \sigma ^ { 2 } } { 1 7 K M } T + \frac { 1 1 5 2 } { 1 7 \beta ^ { 2 } } ( e \eta K L ) ^ { 2 } T D ^ { 2 } .
$$

Therefore,

$$
\sum _ { t = - 1 } ^ { T - 1 } \mathcal { E } _ { s } ^ { t } \leq \frac { 1 8 } { 1 7 \beta } \mathcal { E } _ { s } ^ { - 1 } + \frac { 6 } { 1 7 } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \Big [ \| g ^ { t - 1 } \| ^ { 2 } \Big ] + \frac { 7 2 \beta \sigma ^ { 2 } } { 1 7 K M } T + \frac { 1 1 5 2 } { 1 7 \beta ^ { 2 } } ( e \eta K L ) ^ { 2 } T D ^ { 2 } .\tag{4.5}
$$

Notice that

$$
\sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \Big [ \| g ^ { t - 1 } \| ^ { 2 } \Big ] = \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \Big [ \| g ^ { t } \| ^ { 2 } \Big ] + \mathbb { E } \Big [ \| g ^ { - 1 } \| ^ { 2 } \Big ] - \mathbb { E } \Big [ \| g ^ { T - 1 } \| ^ { 2 } \Big ] .
$$

Hence,

$$
\sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \Big [ \| g ^ { t - 1 } \| ^ { 2 } \Big ] \leq \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \Big [ \| g ^ { t } \| ^ { 2 } \Big ] + \mathbb { E } \Big [ \| g ^ { - 1 } \| ^ { 2 } \Big ] .
$$

By Lemma 4.1,

$$
\frac { 2 \left( \mathbb { E } [ f _ { s } ( x ^ { T } ) ] - f _ { s } ( x ^ { 0 } ) \right) } { \gamma } + \frac { 1 } { 2 } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \Big [ \| g ^ { t } \| ^ { 2 } \Big ] \leq \sum _ { t = 0 } ^ { T - 1 } \mathcal { E } _ { s } ^ { t } \leq \sum _ { t = - 1 } ^ { T - 1 } \mathcal { E } _ { s } ^ { t } .
$$

Combining the above inequality with (4.5), we get

$$
\begin{array} { r } { \displaystyle \frac { 1 } { 2 } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \Big [ \| g ^ { t } \| ^ { 2 } \Big ] \leq \frac { 2 \left( f _ { s } ( x ^ { 0 } ) - \mathbb { E } [ f _ { s } ( x ^ { T } ) ] \right) } { \gamma } + \frac { 1 8 } { 1 7 \beta } \mathcal { E } _ { s } ^ { - 1 } + \frac { 6 } { 1 7 } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \Big [ \| g ^ { t } \| ^ { 2 } \Big ] } \\ { + \frac { 6 } { 1 7 } \mathbb { E } \Big [ \| g ^ { - 1 } \| ^ { 2 } \Big ] + \frac { 7 2 \beta \sigma ^ { 2 } } { 1 7 K M } T + \frac { 1 1 5 2 } { 1 7 \beta ^ { 2 } } ( e \eta K L ) ^ { 2 } T D ^ { 2 } , } \end{array}
$$

which implies that

$$
\begin{array} { l } { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \Big [ \| g ^ { t } \| ^ { 2 } \Big ] \leq \frac { 6 8 \left( f _ { s } ( x ^ { 0 } ) - \mathbb { E } [ f _ { s } ( x ^ { T } ) ] \right) } { 5 \gamma } + \frac { 3 6 } { 5 \beta } \xi _ { s } ^ { - 1 } + \frac { 1 4 4 \beta \sigma ^ { 2 } } { 5 K M } T } \\ { \displaystyle \qquad + \frac { 2 3 0 4 } { 5 \beta ^ { 2 } } ( e \eta K L ) ^ { 2 } T D ^ { 2 } + \frac { 1 2 } { 5 } \mathbb { E } \Big [ \| g ^ { - 1 } \| ^ { 2 } \Big ] } \\ { \leq \frac { 6 8 \Delta } { 5 \gamma } + \frac { 3 6 } { 5 \beta } \xi _ { s } ^ { - 1 } + \frac { 1 4 4 \beta \sigma ^ { 2 } } { 5 K M } T } \\ { \displaystyle \qquad + \frac { 2 3 0 4 } { 5 \beta ^ { 2 } } ( e \eta K L ) ^ { 2 } T D ^ { 2 } + \frac { 1 2 } { 5 } \mathbb { E } \Big [ \| g ^ { - 1 } \| ^ { 2 } \Big ] . } \end{array}\tag{4.6}
$$

Averaging (4.6) over $s \in [ S ]$ , we obtain

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ \| g ^ { t } \| ^ { 2 } \right] \leq \frac { 6 8 \Delta } { 5 \gamma } + \frac { 3 6 } { 5 \beta S } \sum _ { s \in [ S ] } \mathscr { E } _ { s } ^ { - 1 } + \frac { 1 4 4 \beta \sigma ^ { 2 } } { 5 K M } T } \\ & { \quad \quad \quad \quad \quad + \frac { 2 3 0 4 } { 5 \beta ^ { 2 } } ( e \eta K L ) ^ { 2 } T D ^ { 2 } + \frac { 1 2 } { 5 } \mathbb { E } \left[ \| g ^ { - 1 } \| ^ { 2 } \right] . } \end{array}\tag{4.7}
$$

Moreover, from (4.5), we have

$$
\begin{array} { r l } {  { \sum _ { t = 0 } ^ { T - 1 } \mathcal { E } _ { s } ^ { t } \le \frac { 1 8 } { 1 7 \beta } \mathcal { E } _ { s } ^ { - 1 } + \frac { 6 } { 1 7 } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \Big [ \| g ^ { t - 1 } \| ^ { 2 } \Big ] + \frac { 7 2 \beta \sigma ^ { 2 } } { 1 7 K M } T + \frac { 1 1 5 2 } { 1 7 \beta ^ { 2 } } ( e \eta K L ) ^ { 2 } T D ^ { 2 } } } \\ & { \le \frac { 1 8 } { 1 7 \beta } \mathcal { E } _ { s } ^ { - 1 } + \frac { 6 } { 1 7 } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \Big [ \| g ^ { t } \| ^ { 2 } \Big ] + \frac { 6 } { 1 7 } \mathbb { E } \Big [ \| g ^ { - 1 } \| ^ { 2 } \Big ] } \\ & { \quad \quad + \frac { 7 2 \beta \sigma ^ { 2 } } { 1 7 K M } T + \frac { 1 1 5 2 } { 1 7 \beta ^ { 2 } } ( e \eta K L ) ^ { 2 } T D ^ { 2 } . } \end{array}
$$

Substituting (4.6) into the above inequality gives

$$
\begin{array} { l } { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \xi _ { s } ^ { t } \leq \frac { 3 0 6 } { 8 5 \beta } \mathcal { E } _ { s } ^ { - 1 } + \frac { 2 4 \Delta } { 5 \gamma } + \frac { 7 2 \beta \sigma ^ { 2 } } { 5 K M } T + \frac { 1 1 5 2 } { 5 \beta ^ { 2 } } ( e \eta K L ) ^ { 2 } T D ^ { 2 } } \\ { \displaystyle \qquad + \frac { 1 0 2 } { 8 5 } \mathbb { E } \Big [ \| g ^ { - 1 } \| ^ { 2 } \Big ] . } \end{array}\tag{4.8}
$$

Summing (4.8) over $s \in [ S ]$ gives

$$
\begin{array} { r l } & { \displaystyle \sum _ { s \in [ S ] } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathcal { E } _ { s } ^ { t } \leq \frac { 3 0 6 } { 8 5 \beta } \displaystyle \sum _ { s \in [ S ] } \mathcal { E } _ { s } ^ { - 1 } + \frac { 2 4 S \Delta } { 5 \gamma } + \frac { 7 2 S \beta \sigma ^ { 2 } } { 5 K M } T } \\ & { \quad \quad \quad \quad + \frac { 1 1 5 2 S } { 5 \beta ^ { 2 } } ( e \eta K L ) ^ { 2 } T D ^ { 2 } + \frac { 1 0 2 S } { 8 5 } \mathbb { E } \left[ \| g ^ { - 1 } \| ^ { 2 } \right] . } \end{array}\tag{4.9}
$$

Hence,

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { T } \sum _ { \ell = 0 } ^ { T - 1 } \mathbb { E } \left[ \bigg | \displaystyle \sum _ { i < \ell | \ell | } \sum _ { s = 0 } ^ { T - 1 } \mathbb { E } \bigg [ \bigg | \bigg | \displaystyle \sum _ { \ell < s } \lambda _ { \ell } ^ { \ell + s } \nabla f _ { \ell } ( s ^ { \ell } ) - g _ { \ell } ^ { \ell } \bigg | \bigg | ^ { 2 } \right] + \displaystyle \frac { 2 } { T } \sum _ { \ell = 0 } ^ { T - 1 } \mathbb { E } \bigg [ \| s \| ^ { \ell } \| ^ { 2 } \bigg ] } & { } \\ { \displaystyle = \frac { 2 } { T } \sum _ { \ell = 0 } ^ { T - 1 } \mathbb { E } \bigg [ \bigg | \displaystyle \sum _ { \ell < s } \lambda _ { \ell } ^ { \ell + s } \Big ( \nabla f _ { \ell } ( s ^ { \ell } ) - g _ { \ell } ^ { \ell } \bigg ) \bigg | ^ { 2 } \bigg ] + \displaystyle \frac { 2 } { T } \sum _ { \ell = 0 } ^ { T - 1 } \mathbb { E } \bigg [ \| s ^ { \ell } \| ^ { 2 } \bigg ] } & { } \\ { \displaystyle \leq \frac { 2 } { T } \sum _ { \ell = 0 } ^ { T - 1 } \mathbb { E } \bigg [ \displaystyle \sum _ { \ell > s } \lambda _ { \ell } ^ { \ell } \Big | \nabla f _ { \ell } ( s ^ { \ell } ) - g _ { \ell } ^ { \ell } \bigg | ^ { 2 } \bigg ] + \displaystyle \frac { 2 } { T } \sum _ { \ell = 0 } ^ { T - 1 } \mathbb { E } \bigg [ \| s ^ { \ell } \| ^ { 2 } \bigg ] } & { } \\  \displaystyle \leq \frac { 2 } { T } \sum _ { \ell = 0 } ^ { T - 1 } \sum _ { \ell > 0 } ^ { T - 1 } \mathbb { E } \bigg [ \big \| \nabla f _ { \ell } ( s ^ { \ell } ) - g _ { \ell } ^ { \ell } \big \| ^ { 2 } \bigg ] - \displaystyle \frac { 2 } { T } \sum _ { \ell = 0 } ^ { T - 1 } \mathbb { E } \big [ \| s ^ { \ell } \| ^ \end{array}
$$

where the second inequality follows from the convexity of $\| \cdot \| ^ { 2 }$ . Plugging (4.7) and (4.9) into the

above inequality, we have

$$
\begin{array} { r l } & { \quad \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ \left\| \sum _ { s \in [ S ] } \lambda _ { s } ^ { t , * } \nabla f _ { s } ( x ^ { t } ) \right\| ^ { 2 } \right] } \\ & { \le \frac { 6 1 2 } { 8 5 \beta T } \sum _ { s \in [ S ] } \xi _ { s } ^ { - 1 } + \frac { 4 8 S \Delta } { 5 \gamma T } + \frac { 1 4 4 S \beta \sigma ^ { 2 } } { 5 K M } } \\ & { \quad \quad + \frac { 2 3 0 4 S } { 5 \beta ^ { 2 } } ( e \eta K L D ) ^ { 2 } + \frac { 2 0 4 S } { 8 5 T } \mathbb { E } [ \| g ^ { - 1 } \| ^ { 2 } ] + \frac { 7 2 } { 5 \beta S T } \sum _ { s \in [ S ] } \xi _ { s } ^ { - 1 } } \\ & { \quad \quad + \frac { 1 3 6 \Delta } { 5 \gamma T } + \frac { 2 8 8 \beta \sigma ^ { 2 } } { 5 K M } + \frac { 4 6 0 8 } { 5 \beta ^ { 2 } } ( e \eta K L D ) ^ { 2 } + \frac { 2 4 } { 5 T } \mathbb { E } [ \| g ^ { - 1 } \| ^ { 2 } ] . } \end{array}
$$

Finally, for $\begin{array} { r } { B = \left\lceil \frac { K } { T \beta ^ { 2 } } \right\rceil } \end{array}$ , noticing that

$$
g _ { s } ^ { - 1 } = \frac { 1 } { B M } \sum _ { i = 1 } ^ { M } \sum _ { b = 1 } ^ { B } \nabla F _ { s , i } ( x ^ { - 1 } ; \xi _ { i } ^ { b } ) ,
$$

we have

$$
\begin{array} { r l } { \xi _ { s } ^ { - 1 } } & { - \mathbb { E } [ \vert \nabla F _ { 2 } ( \cdot , \cdot ^ { - 1 } ) - g _ { s } \vert ^ { 1 } ] ^ { 2 } } \\ & { - \mathbb { E } \Bigg [ \Bigg \vert \Bigg \vert \frac { 1 } { B A } \sum _ { i , j = 1 } ^ { \infty } \nabla F _ { 2 , i } ( z ^ { - 1 } ) - \frac { 1 } { B A } \sum _ { i , j = 1 } ^ { \infty } ( \xi _ { s , j } ( z ^ { - 1 } ) , \xi _ { s , j } ^ { - 1 } ) \Bigg \vert ^ { 2 } \Bigg ] } \\ & { - \frac { 1 } { B ^ { 2 } M ^ { 2 } } \Bigg [ \Bigg \vert \frac { \partial } { \partial \xi _ { s } } \Bigg ( \nabla F _ { 2 , i } ( z ^ { - 1 } ) - \nabla F _ { 2 , i } ( z ^ { - 1 } ) \xi _ { s , j } ^ { - 1 } ) \Bigg \vert \Bigg ] ^ { 2 } \Bigg ] } \\ & { - \frac { 1 } { B ^ { 2 } M ^ { 2 } } \Bigg [ \Bigg \vert \frac { \partial } { \partial \xi _ { s } } \Bigg \vert \Bigg \vert \nabla F _ { 2 , i } ( \cdot ^ { 1 } ) - \nabla F _ { 2 , i } ( z ^ { - 1 } , \xi _ { s } ^ { - 1 } ) \Bigg \vert \Bigg \vert ^ { 2 } } \\ & { + 2 \sum _ { i , j = 1 } ^ { \infty } \left( \nabla F _ { 2 , i } ( z ^ { - 1 } ) - \nabla F _ { 2 , i } ( z ^ { - 1 } , \xi _ { s } ^ { - 1 } ) , \nabla F _ { 2 , i } ( z ^ { - 1 } ) - \nabla F _ { 2 , i } ( z ^ { - 1 } , \xi _ { s } ^ { - 1 } ) \right) \Bigg ] } \\ &  \quad + \frac { 1 } { 6 \cdot \rho _ { s } \rho _ { s } ^ { 2 } \rho _ { s } ^ { 2 } \rho _ { s } ^ { 2 } } \Bigg [ \Bigg \vert \nabla F _ { 2 , i } ( \cdot ^ { 1 } ) - \nabla F _ { 2 , i } ( z ^ { - 1 } , \xi _ { s } ^ { - 1 } ) \nabla F _ \end{array}
$$

Moreover, since $g ^ { - 1 }$ is obtained by applying the same Pareto aggregation rule to $\{ g _ { s } ^ { - 1 } \} _ { s \in [ S ] }$ , it holds that

$$
g ^ { - 1 } = \sum _ { s \in [ S ] } \lambda _ { s } ^ { - 1 , * } g _ { s } ^ { - 1 } ,
$$

where $\lambda ^ { - 1 , * } \in [ 0 , 1 ] ^ { S }$ solves

$$
\operatorname* { m i n } _ { \lambda _ { s } \geq 0 , } \left\| \sum _ { s \in [ S ] } \lambda _ { s } d _ { s } ^ { - 1 } \right\| ^ { 2 } .
$$

By the optimality of $\lambda ^ { - 1 , * }$ , choosing the uniform weight $\lambda _ { s } = 1 / S$ gives

$$
\mathbb { E } \| g ^ { - 1 } \| ^ { 2 } \leq \mathbb { E } \left[ \left\| \frac { 1 } { S } \sum _ { s \in [ S ] } g _ { s } ^ { - 1 } \right\| ^ { 2 } \right]
$$

$$
\leq \frac { 1 } { S } \sum _ { s \in [ S ] } \mathbb { E } \left[ \Vert g _ { s } ^ { - 1 } \Vert ^ { 2 } \right] .
$$

For each $s \in [ S ]$ , we have

$$
\begin{array} { r l } & { \mathbb { E } \left[ \Vert g _ { s } ^ { - 1 } \Vert ^ { 2 } \right] = \mathbb { E } \left[ \left. \nabla f _ { s } ( x ^ { - 1 } ) + g _ { s } ^ { - 1 } - \nabla f _ { s } ( x ^ { - 1 } ) \right. ^ { 2 } \right] } \\ & { \qquad \leq 2 \left. \nabla f _ { s } ( x ^ { - 1 } ) \right. ^ { 2 } + 2 \mathbb { E } \left[ \left. g _ { s } ^ { - 1 } - \nabla f _ { s } ( x ^ { - 1 } ) \right. ^ { 2 } \right] } \\ & { \qquad = 2 \left. \nabla f _ { s } ( x ^ { - 1 } ) \right. ^ { 2 } + 2 \mathcal { E } _ { s } ^ { - 1 } . } \end{array}
$$

Since $x ^ { - 1 } = x ^ { 0 }$ , let

$$
G _ { 0 } ^ { 2 } : = \frac { 1 } { S } \sum _ { s \in [ S ] } \left\| \nabla f _ { s } ( x ^ { 0 } ) \right\| ^ { 2 } .
$$

Then

$$
\mathbb { E } \Vert g ^ { - 1 } \Vert ^ { 2 } \leq 2 G _ { 0 } ^ { 2 } + \frac { 2 } { S } \sum _ { s \in [ S ] } \mathcal { E } _ { s } ^ { - 1 } \leq 2 G _ { 0 } ^ { 2 } + \frac { 2 \beta ^ { 2 } \sigma ^ { 2 } T } { K M } .
$$

Thus,

$$
\begin{array} { r c l } { \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ \left\| \displaystyle \sum _ { s : [ S ] } \lambda _ { s } ^ { t , * } \nabla f _ { s } ( x ^ { t } ) \right\| ^ { 2 } \right] \lesssim \frac { 1 } { \beta T } \displaystyle \sum _ { s : [ S ] } \xi _ { s } ^ { - 1 } + \frac { S \Delta } { \sqrt { T } } + \frac { S \beta \sigma ^ { 2 } } { K M } + \frac { S } { \beta ^ { 2 } } ( \eta K L D ) ^ { 2 } } \\ { \displaystyle } & { + \frac { 1 } { \beta T S } \displaystyle \sum _ { s \in [ S ] } \xi _ { s } ^ { - 1 } + \frac { \Delta } { \sqrt { T } } + \frac { \beta \sigma ^ { 2 } } { K M } + \frac { 1 } { \beta ^ { 2 } } ( \eta K L D ) ^ { 2 } } \\ { \displaystyle } & { + \frac { 1 + S } { T } \mathbb { E } [ \| g ^ { - 1 } \| ^ { 2 } ] } \\ { \displaystyle } & { \lesssim \frac { \beta \sigma ^ { 2 } } { K M } ( 1 + S ) + \frac { ( \eta K L ) ^ { 2 } } { \beta ^ { 2 } } D ^ { 2 } ( 1 + S ) + \frac { S \Delta } { \sqrt { T } } } \\ { \displaystyle } & { + \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { T } + \frac { ( 1 + S ) \beta ^ { 2 } \sigma ^ { 2 } } { K M } . } \end{array}\tag{4.10}
$$

By the choice of $\eta$

$$
\frac { ( \eta K L ) ^ { 2 } } { \beta ^ { 2 } } \lesssim \frac { L \Delta } { \beta ^ { 1 / 2 } ( K M ) ^ { 1 / 2 } T } .
$$

Moreover,

$$
\frac { 1 } { \gamma } \leq 2 L + \sqrt { 5 4 } \frac { L } { \sqrt { \beta K M } } ,
$$

which implies that

$$
\frac { S \Delta } { \gamma T } \lesssim \frac { L S \Delta } { T } + \frac { L S \Delta } { \beta ^ { 1 / 2 } ( K M ) ^ { 1 / 2 } T } .
$$

Therefore, from (4.10), we have

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ \left. \sum _ { s \in [ S ] } \lambda _ { s } ^ { t , s } \nabla f _ { s } ( x ^ { t } ) \right. ^ { 2 } \right] \lesssim \frac { \beta \sigma ^ { 2 } } { K M } ( 1 + S ) + \frac { L ( 1 + D ^ { 2 } ) \Delta } { \beta ^ { 1 / 2 } ( K M ) ^ { 1 / 2 } T } ( 1 + S ) + \frac { L S \Delta } { T } + \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { T } + \frac { ( 1 + S ) \beta ^ { 2 } \sigma ^ { 2 } } { K M } .
$$

Taking

$$
\beta = \left( \frac { M K L ^ { 2 } ( 1 + D ^ { 2 } ) ^ { 2 } \Delta ^ { 2 } } { \sigma ^ { 4 } T ^ { 2 } } \right) ^ { 1 / 3 }
$$

gives

$$
\begin{array} { l } { \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ \left\| \sum _ { s \in [ S ] } \lambda _ { s } ^ { t , * } \nabla f _ { s } ( x ^ { t } ) \right\| ^ { 2 } \right] } \\ { \displaystyle \lesssim \left( \frac { L ( 1 + D ^ { 2 } ) \sigma \Delta } { K M T } \right) ^ { 2 / 3 } ( 1 + S ) + \frac { L S \Delta } { T } + \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { T } } \\ { \displaystyle \quad + \frac { ( 1 + S ) \sigma ^ { 2 } } { K M } \left( \frac { M K L ^ { 2 } ( 1 + D ^ { 2 } ) ^ { 2 } \Delta ^ { 2 } } { \sigma ^ { 4 } T ^ { 2 } } \right) ^ { 2 / 3 } . } \end{array}
$$

Since $\lambda ^ { t , * } \in \mathcal { C }$ , by the definition of $\mathcal { G } ( \boldsymbol { x } ^ { t } )$ 2

$$
\mathcal { G } ( { \boldsymbol x } ^ { t } ) = \operatorname* { m i n } _ { \lambda \in \mathcal { C } } \left\| \sum _ { s \in [ S ] } \lambda _ { s } \nabla f _ { s } ( { \boldsymbol x } ^ { t } ) \right\| ^ { 2 } \leq \left\| \sum _ { s \in [ S ] } \lambda _ { s } ^ { t , * } \nabla f _ { s } ( { \boldsymbol x } ^ { t } ) \right\| ^ { 2 } .
$$

Taking expectations and averaging over $t = 0 , \ldots , T - 1$ , we obtain

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ \mathcal { G } ( { x } ^ { t } ) \right] \leq \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ \left. \sum _ { s \in [ S ] } \lambda _ { s } ^ { t , * } \nabla f _ { s } ( { x } ^ { t } ) \right. ^ { 2 } \right] .
$$

Since a is chosen uniformly at random from $\{ 0 , 1 , \ldots , T - 1 \}$

$$
\begin{array} { r l } {  { \mathbb { E } [ \mathscr { G } ( x _ { a } ) ] = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \mathscr { G } ( x ^ { t } ) ] } } \\ & { \leq \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \| \displaystyle \sum _ { s \in [ S ] } \lambda _ { s } ^ { t , * } \nabla f _ { s } ( x ^ { t } ) \| ^ { 2 } ] } \\ & { \lesssim ( \frac { L ( 1 + D ^ { 2 } ) \sigma \Delta } { K M T } ) ^ { \frac { 2 } { 3 } } ( 1 + S ) + \frac { L S \Delta } { T } + \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { T } } \\ & { \quad + \frac { ( 1 + S ) \sigma ^ { 2 } } { K M } ( \frac { M K L ^ { 2 } ( 1 + D ^ { 2 } ) \Delta ^ { 2 } } { \sigma ^ { 4 } T ^ { 2 } } ) ^ { \frac { 2 } { 3 } } . } \end{array}
$$

Similarly, if $B = \Theta ( K T )$ , then

$$
\mathcal { E } _ { s } ^ { - 1 } \leq \frac { \sigma ^ { 2 } } { B M } \lesssim \frac { \sigma ^ { 2 } } { K M T } .
$$

Moreover, by the same argument as above,

$$
\mathbb { E } \Vert d ^ { - 1 } \Vert ^ { 2 } \lesssim G _ { 0 } ^ { 2 } + \frac { \sigma ^ { 2 } } { K M T } .
$$

Using the preceding estimate, we have

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ \left\| \sum _ { s \in [ S ] } \lambda _ { s } ^ { t , * } \nabla f _ { s } ( x ^ { t } ) \right\| ^ { 2 } \right] } \\ & { \displaystyle \lesssim \frac { \sigma ^ { 2 } } { K M \beta T ^ { 2 } } ( 1 + S ) + \frac { \beta \sigma ^ { 2 } } { K M } ( 1 + S ) + \frac { ( \eta K L ) ^ { 2 } } { \beta ^ { 2 } } D ^ { 2 } ( 1 + S ) } \\ & { \quad + \frac { S \Delta } { \gamma T } + \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { T } + \frac { ( 1 + S ) \sigma ^ { 2 } } { K M T ^ { 2 } } . } \end{array}
$$

By the choice of $\eta ,$

$$
\frac { ( \eta K L ) ^ { 2 } } { \beta ^ { 2 } } \lesssim \frac { L \Delta } { \beta ^ { 1 / 2 } ( K M ) ^ { 1 / 2 } T } .
$$

Moreover, since

$$
\gamma = \mathrm { m i n } \left\{ \frac { 1 } { 2 L } , \frac { 1 } { L } \sqrt { \frac { \beta K M } { 5 4 } } \right\} ,
$$

it follows that

$$
\frac { 1 } { \gamma } \leq 2 L + \sqrt { 5 4 } \frac { L } { \sqrt { \beta K M } } .
$$

Thus,

$$
\frac { S \Delta } { \gamma T } \lesssim \frac { L S \Delta } { T } + \frac { L S \Delta } { \beta ^ { 1 / 2 } ( K M ) ^ { 1 / 2 } T } .
$$

Combining the above estimates gives

$$
\begin{array} { r l } & { \frac { 1 } { T } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ \left. \sum _ { s \in [ S ] } \lambda _ { s } ^ { t , * } \nabla f _ { s } ( x ^ { t } ) \right. ^ { 2 } \right] \lesssim \frac { \sigma ^ { 2 } } { K M \beta T ^ { 2 } } ( 1 + S ) + \frac { \beta \sigma ^ { 2 } } { K M } ( 1 + S ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad + \frac { L ( 1 + D ^ { 2 } ) \Delta } { \beta ^ { 1 / 2 } ( K M ) ^ { 1 / 2 } T } ( 1 + S ) + \frac { L S \Delta } { T } + \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { T } . } \end{array}
$$

Now take

$$
\beta = \operatorname * { m a x } \left\{ \frac { 1 } { T } , \left( \frac { M K L ^ { 2 } ( 1 + D ^ { 2 } ) ^ { 2 } \Delta ^ { 2 } } { \sigma ^ { 4 } T ^ { 2 } } \right) ^ { 1 / 3 } \right\} .
$$

Since $\beta \ge 1 / T$ , we have

$$
\frac { \sigma ^ { 2 } } { K M \beta T ^ { 2 } } \leq \frac { \sigma ^ { 2 } } { K M T } .
$$

Furthermore, by the definition of $\beta _ { i }$

$$
\frac { \beta \sigma ^ { 2 } } { K M } + \frac { L ( 1 + D ^ { 2 } ) \Delta } { \beta ^ { 1 / 2 } ( K M ) ^ { 1 / 2 } T } \lesssim \frac { \sigma ^ { 2 } } { K M T } + \left( \frac { L ( 1 + D ^ { 2 } ) \sigma \Delta } { K M T } \right) ^ { 2 / 3 } .
$$

Therefore,

$$
\begin{array} { r l r } {  { \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \| \sum _ { s \in [ S ] } \lambda _ { s } ^ { t , * } \nabla f _ { s } ( x ^ { t } ) \| ^ { 2 } ] \lesssim \frac { \sigma ^ { 2 } ( 1 + S ) } { K M T } + ( \frac { L ( 1 + D ^ { 2 } ) \sigma \Delta } { K M T } ) ^ { 2 / 3 } ( 1 + S ) } } \\ & { } & { + \frac { L S \Delta } { T } + \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { T } + \frac { ( 1 + S ) \sigma ^ { 2 } } { K M T ^ { 2 } } , } \end{array}
$$

which implies that

$$
\begin{array} { r l r } {  { \mathbb { E } [ \mathcal { G } ( x _ { a } ) ] \lesssim \frac { \sigma ^ { 2 } ( 1 + S ) } { K M T } + ( \frac { L ( 1 + D ^ { 2 } ) \sigma \Delta } { K M T } ) ^ { 2 / 3 } ( 1 + S ) } } \\ & { } & { + \frac { L S \Delta } { T } + \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { T } + \frac { ( 1 + S ) \sigma ^ { 2 } } { K M T ^ { 2 } } . } \end{array}
$$

The proof is complete.

Remark 4.2. Theorem 4.1 establishes a random-iterate Pareto stationarity guarantee for Algorithm 1. Specifically, the algorithm returns an iterate $x _ { a } .$ where $a$ is chosen uniformly at random from $\{ 0 , \ldots , T - 1 \}$ , and the corresponding expected Pareto stationarity measure $\mathbb { E } [ \mathcal { G } ( x _ { a } ) ]$ decreases at a rate of $\mathcal { O } ( T ^ { - 2 / 3 } )$ . This improves upon the $\mathcal { O } ( T ^ { - 1 / 2 } )$ rate established in [32,33].

Corollary 4.1. Under the conditions of Theorem $4 . 1 ,$ FSMGDA-M-VR attains an e-Pareto stationary solution with communication complexity

$$
\mathcal { O } \left( \operatorname* { m a x } \left\{ \frac { L ( 1 + D ^ { 2 } ) \sigma \Delta ( 1 + S ) ^ { \frac { 3 } { 2 } } } { K M \epsilon ^ { \frac { 3 } { 2 } } } , \frac { L S \Delta } { \epsilon } , \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { \epsilon } , \left( \frac { C _ { 1 } } { \epsilon } \right) ^ { \frac { 3 } { 4 } } , \frac { \sigma ^ { 2 } ( 1 + S ) } { K M \epsilon } , \left( \frac { ( 1 + S ) \sigma ^ { 2 } } { K M \epsilon } \right) ^ { \frac { 1 } { 2 } } \right\} \right) ,
$$

where

$$
C _ { 1 } : = \frac { ( 1 + S ) \sigma ^ { 2 } } { K M } \left( \frac { M K L ^ { 2 } ( 1 + D ^ { 2 } ) ^ { 2 } \Delta ^ { 2 } } { \sigma ^ { 4 } } \right) ^ { 2 / 3 } .
$$

Moreover, the corresponding sample complexity is

$$
\mathcal { O } \left( S M K \operatorname* { m a x } \left\{ \frac { L ( 1 + D ^ { 2 } ) \sigma \Delta ( 1 + S ) ^ { \frac { 3 } { 2 } } } { K M \epsilon ^ { \frac { 3 } { 2 } } } , \frac { L S \Delta } { \epsilon } , \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { \epsilon } , \left( \frac { C _ { 1 } } { \epsilon } \right) ^ { \frac { 3 } { 4 } } , \frac { \sigma ^ { 2 } ( 1 + S ) } { K M \epsilon } , \left( \frac { ( 1 + S ) \sigma ^ { 2 } } { K M \epsilon } \right) ^ { \frac { 1 } { 2 } } \right\} \right) .
$$

Proof. For the first case $\begin{array} { r } { B = \left\lceil \frac { K } { T \beta ^ { 2 } } \right\rceil } \end{array}$ , by Theorem 4.1,

$$
\begin{array} { r } { \mathbb { E } \left[ \mathcal { G } ( x _ { a } ) \right] \lesssim \left( \displaystyle \frac { L \left( 1 + D ^ { 2 } \right) \sigma \Delta } { K M T } \right) ^ { \frac 2 3 } ( 1 + S ) + \displaystyle \frac { L S \Delta } { T } \qquad } \\ { + \displaystyle \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { T } + \frac { ( 1 + S ) \sigma ^ { 2 } } { K M } \left( \frac { M K L ^ { 2 } ( 1 + D ^ { 2 } ) ^ { 2 } \Delta ^ { 2 } } { \sigma ^ { 4 } T ^ { 2 } } \right) ^ { 2 / 3 } . } \end{array}
$$

To obtain an e-Pareto stationary solution, it is sufficient, up to absolute constant factors hidden in $\lesssim ,$ to require

$$
\left( \frac { L ( 1 + D ^ { 2 } ) \sigma \Delta } { K M T } \right) ^ { \frac { 2 } { 3 } } ( 1 + S ) \leq \frac { \epsilon } { 4 } ,
$$

$$
\frac { L S \Delta } { T } \leq \frac { \epsilon } { 4 } , \qquad \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { T } \leq \frac { \epsilon } { 4 } ,
$$

and

$$
\frac { ( 1 + S ) \sigma ^ { 2 } } { K M } \left( \frac { M K L ^ { 2 } ( 1 + D ^ { 2 } ) ^ { 2 } \Delta ^ { 2 } } { \sigma ^ { 4 } T ^ { 2 } } \right) ^ { \frac 2 3 } \le \frac \epsilon 4 .
$$

The first inequality yields

$$
T \geq \frac { L ( 1 + D ^ { 2 } ) \sigma \Delta } { K M } \left( \frac { 4 ( 1 + S ) } { \epsilon } \right) ^ { \frac { 3 } { 2 } } ,
$$

and hence

$$
T = \mathcal { O } \left( \frac { L ( 1 + D ^ { 2 } ) \sigma \Delta ( 1 + S ) ^ { \frac { 3 } { 2 } } } { K M \epsilon ^ { \frac { 3 } { 2 } } } \right) .
$$

The second and third inequalities give

$$
T = \mathcal { O } \left( \frac { L S \Delta } { \epsilon } \right) , \qquad T = \mathcal { O } \left( \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { \epsilon } \right) .
$$

For the last inequality, let $\begin{array} { r } { C _ { 1 } : = \frac { ( 1 + S ) \sigma ^ { 2 } } { K M } \left( \frac { M K L ^ { 2 } ( 1 + D ^ { 2 } ) ^ { 2 } \Delta ^ { 2 } } { \sigma ^ { 4 } } \right) ^ { \frac { 2 } { 3 } } } \end{array}$ . Thus, it is sufficient to require

$$
\frac { C _ { 1 } } { T ^ { \frac 4 3 } } \leq \frac \epsilon 4 ,
$$

which yields

$$
T = \mathcal { O } \left( \left( \frac { C _ { 1 } } { \epsilon } \right) ^ { \frac { 3 } { 4 } } \right) .
$$

Therefore, for the first choice of $\begin{array} { r } { B = \left\lceil \frac { K } { T \beta ^ { 2 } } \right\rceil } \end{array}$ , it suffices to take

$$
T = { \cal O } \left( \operatorname * { m a x } \left\{ \frac { L ( 1 + D ^ { 2 } ) \sigma \Delta ( 1 + S ) ^ { \frac { 3 } { 2 } } } { K M \epsilon ^ { \frac { 3 } { 2 } } } , \frac { L S \Delta } { \epsilon } , \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { \epsilon } , \left( \frac { C _ { 1 } } { \epsilon } \right) ^ { \frac { 3 } { 4 } } \right\} \right) .
$$

Next, for the second case $B = \Theta ( K T )$ , Theorem 4.1 gives

$$
\mathbb { E } \left[ \mathcal { G } ( x _ { a } ) \right] \lesssim \frac { \sigma ^ { 2 } ( 1 + S ) } { K M T } + \left( \frac { L ( 1 + D ^ { 2 } ) \sigma \Delta } { K M T } \right) ^ { \frac 2 3 } ( 1 + S ) + \frac { L S \Delta } { T } + \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { T } + \frac { ( 1 + S ) \sigma ^ { 2 } } { K M T ^ { 2 } } .
$$

To obtain an e-Pareto stationary solution, it is sufficient, up to absolute constant factors hidden in $\lesssim ,$ to require

$$
\begin{array} { c } { { \displaystyle \frac { \sigma ^ { 2 } ( 1 + S ) } { K M T } \le \frac { \epsilon } { 5 } , } } \\ { { \displaystyle \left( \frac { L ( 1 + D ^ { 2 } ) \sigma \Delta } { K M T } \right) ^ { \frac { 2 } { 3 } } ( 1 + S ) \le \frac { \epsilon } { 5 } , } } \\ { { \displaystyle \frac { L S \Delta } { T } \le \frac { \epsilon } { 5 } , \qquad \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { T } \le \frac { \epsilon } { 5 } , } } \end{array}
$$

and

$$
\frac { ( 1 + S ) \sigma ^ { 2 } } { K M T ^ { 2 } } \leq \frac { \epsilon } { 5 } .
$$

These inequalities respectively yield

$$
T = \mathcal { O } \left( \frac { \sigma ^ { 2 } ( 1 + S ) } { K M \epsilon } \right) ,
$$

$$
T = \mathcal { O } \left( \frac { L ( 1 + D ^ { 2 } ) \sigma \Delta ( 1 + S ) ^ { \frac { 3 } { 2 } } } { K M \epsilon ^ { \frac { 3 } { 2 } } } \right) ,
$$

$$
T = \mathcal { O } \left( \frac { L S \Delta } { \epsilon } \right) , \qquad T = \mathcal { O } \left( \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { \epsilon } \right) ,
$$

and

$$
T = \mathcal { O } \left( \left( \frac { ( 1 + S ) \sigma ^ { 2 } } { K M \epsilon } \right) ^ { \frac { 1 } { 2 } } \right) .
$$

Therefore, for the second choice $B = \Theta ( K T )$ , it suffices to take

$$
T = \mathcal { O } \left( \operatorname* { m a x } \left\{ \frac { L ( 1 + D ^ { 2 } ) \sigma \Delta ( 1 + S ) ^ { \frac { 3 } { 2 } } } { K M \epsilon ^ { \frac { 3 } { 2 } } } , \frac { L S \Delta } { \epsilon } , \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { \epsilon } , \frac { \sigma ^ { 2 } ( 1 + S ) } { K M \epsilon } , \left( \frac { ( 1 + S ) \sigma ^ { 2 } } { K M \epsilon } \right) ^ { \frac { 1 } { 2 } } \right\} \right) .
$$

We next derive the communication complexity and the sample complexity.

• Communication complexity: In $\mathrm { F S M G D A - M \mathrm { - } V R } .$ each global iteration corresponds to one round of communication between the clients and the server. Therefore, the total communication complexity is equal to the total number of global rounds $T .$ Combining the above two cases, the communication complexity can be expressed as

$$
\mathcal { O } \left( \operatorname* { m a x } \left\{ \frac { L ( 1 + D ^ { 2 } ) \sigma \Delta ( 1 + S ) ^ { \frac { 3 } { 2 } } } { K M \epsilon ^ { \frac { 3 } { 2 } } } , \frac { L S \Delta } { \epsilon } , \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { \epsilon } , \left( \frac { C _ { 1 } } { \epsilon } \right) ^ { \frac { 3 } { 4 } } , \frac { \sigma ^ { 2 } ( 1 + S ) } { K M \epsilon } , \left( \frac { ( 1 + S ) \sigma ^ { 2 } } { K M \epsilon } \right) ^ { \frac { 1 } { 2 } } \right\} \right) .
$$

• Sample Complexity: Next, we estimate the total number of stochastic gradient computations.

At the initialization stage, the algorithm requires SMB stochastic gradient computations. At each local update, the algorithm computes stochastic gradients for all S objectives, and both the current stochastic gradient and the gradient at the previous global model are involved in the momentum estimator. Therefore, each communication round requires 2SM K stochastic gradient computations. Hence, the total number of stochastic gradient computations is

$$
S M B + 2 S M K T .
$$

It remains to compare the initialization cost SMB with the training-stage cost 2SMKT. We consider the following two cases.

(1) $B = \left\lceil \frac { K } { T \beta ^ { 2 } } \right\rceil$ . In this case,

$$
\frac { S M B } { 2 S M K T } = \frac { B } { 2 K T } .
$$

Since

$$
B = \left\lceil \frac { K } { T \beta ^ { 2 } } \right\rceil \leq \frac { K } { T \beta ^ { 2 } } + 1 ,
$$

we have

$$
\frac { B } { 2 K T } \leq \frac { 1 } { 2 T ^ { 2 } \beta ^ { 2 } } + \frac { 1 } { 2 K T } .
$$

Recall that

$$
\beta = \left( \frac { M K L ^ { 2 } ( 1 + D ^ { 2 } ) ^ { 2 } \Delta ^ { 2 } } { \sigma ^ { 4 } T ^ { 2 } } \right) ^ { 1 / 3 } ,
$$

then

$$
\frac { 1 } { T ^ { 2 } \beta ^ { 2 } } = \frac { \sigma ^ { 8 / 3 } } { M ^ { 2 / 3 } K ^ { 2 / 3 } L ^ { 4 / 3 } ( 1 + D ^ { 2 } ) ^ { 4 / 3 } \Delta ^ { 4 / 3 } } \cdot \frac { 1 } { T ^ { 2 / 3 } }  0 \qquad ( T  \infty ) .
$$

Therefore,

$$
\frac { S M B } { 2 S M K T }  0 .
$$

Thus, the initialization cost SM B is of lower order than the training-stage cost 2SM KT, and hence

$$
S M B + 2 S M K T = { \mathcal { O } } ( S M K T ) .
$$

(2) $B = \Theta ( K T )$ . In this case, we have $S M B = \Theta ( S M K T )$ , and hence

$$
S M B + 2 S M K T = { \mathcal { O } } ( S M K T ) ,
$$

which implies that the total number of stochastic gradient computations is also of order $\mathcal { O } ( S M K T )$

Combining the above two cases, for either choice of B, the total number of stochastic gradient computations is of order O(SM KT). Multiplying the above communication complexity by SMK, we obtain the sample complexity

$$
\mathcal { O } \left( S M K \operatorname* { m a x } \left\{ \frac { L ( 1 + D ^ { 2 } ) \sigma \Delta ( 1 + S ) ^ { \frac { 3 } { 2 } } } { K M \epsilon ^ { \frac { 3 } { 2 } } } , \frac { L S \Delta } { \epsilon } , \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { \epsilon } , \left( \frac { C _ { 1 } } { \epsilon } \right) ^ { \frac { 3 } { 4 } } , \frac { \sigma ^ { 2 } ( 1 + S ) } { K M \epsilon } , \left( \frac { ( 1 + S ) \sigma ^ { 2 } } { K M \epsilon } \right) ^ { \frac { 1 } { 2 } } \right\} \right) ,
$$

which completes the proof.

Remark 4.3. Compared with FSMGDA and FedCMOO, the proposed FSMGDA-M-VR improves the dependence on the target accuracy in terms of both communication-round complexity and the total number of stochastic gradient computations. Specifically, to obtain an e-Pareto stationary

solution in expectation, FSMGDA-M-VR requires

$$
T = { \cal O } \left( \operatorname* { m a x } \left\{ \frac { L ( 1 + D ^ { 2 } ) \sigma \Delta ( 1 + S ) ^ { \frac { 3 } { 2 } } } { K M \epsilon ^ { \frac { 3 } { 2 } } } , \frac { L S \Delta } { \epsilon } , \frac { ( 1 + S ) G _ { 0 } ^ { 2 } } { \epsilon } , \left( \frac { C _ { 1 } } { \epsilon } \right) ^ { \frac { 3 } { 4 } } , \frac { \sigma ^ { 2 } ( 1 + S ) } { K M \epsilon } , \left( \frac { ( 1 + S ) \sigma ^ { 2 } } { K M \epsilon } \right) ^ { \frac { 1 } { 2 } } \right\} \right)
$$

communication rounds. Hence, for fixed problem-dependent constants and in the high-accuracy regime where the first term dominates, its communication-round complexity is of order $\mathcal { O } ( \epsilon ^ { - 3 / 2 } )$ improving upon the $\mathcal { O } ( \epsilon ^ { - 2 } )$ communication-round complexity of FSMGDA and FedCMOO.

## 5. Numerical experiments

In this section, we evaluate Algorithm 1 on three federated multiobjective learning benchmarks. We first describe the experimental setup in Section 5.1 and then present the results in Sections 5.2–5.5.

## 5.1. Datasets and implementation

We consider the following experimental settings:

1) MultiMNIST: We construct a two-objective dataset from MNIST by placing two randomly selected digits in each image—one in the top-left corner and the other in the bottom-right corner—while preserving the original image size.

2) MNIST+FMNIST: We generate a new dataset by randomly combining one image from MNIST and one from FashionMNIST, following a construction method similar to MultiMNIST.

3) CIFAR10+MNIST: We synthesize images by superimposing a randomly selected MNIST digit at a fixed location onto all three channels of CIFAR10 images.

In our experiments, we employ convolutional neural networks of varying scales, with model sizes spanning from 34.6k parameters to 1.73m parameters in the largest model, ResNet-18. All architectures are based on a shared encoder for all the objective functions, coupled with task-specific decoders comprising the final one or two linear layers.

Unless the number of clients is varied in Section 5.5, the main experiments use 100 clients. To induce data heterogeneity across clients, we follow the implementation in [40| and use a Dirichlet distribution with concentration parameter $\alpha = 0 . 3$ to determine the class proportions assigned to each client, while keeping the total number of data points equal across clients. To ensure a fair comparison, the same Dirichlet client partition, training/validation split, model initialization, random seeds. and minibatch sequences are used for all methods. For the MultiMNIST and MNIST+FMNIST datasets, we employ a LeNet-like CNN encoder [1], with two linear layers per task as decoders, resulting in a total of 34.6k parameters. For the CIFAR10+MNIST dataset, we use a CNN encoder adapted from [40], along with a single linear layer decoder, totaling 1.73m parameters. In terms of training configuration, we apply the negative log-likelihood loss across all three datasets and incorporate random image rotation for data augmentation during training. The batch size is set to 128, and unless otherwise noted, each training round comprises 10 local steps. Each experiment is repeated with 10 different random seeds. We report the mean and standard deviation across the runs, with the shaded regions in the figures indicating one standard deviation around the mean.

All datasets come with predefined training and test splits, which we adopt together with the default validation splits when available. In cases where a validation split is not provided, we randomly partition the training set into 80% for training and 20% for validation. Throughout all experiments, constant learning rates are used without any schedule or decay. The learning rates are selected by running each method under the corresponding experimental setting and choosing the value that yields the best performance on the validation set.

For the MultiMNIST experiments, different learning-rate configurations are adopted across algorithms. Specifically, FSMGDA [32] uses a global learning rate of 2 and a local learning rate of 0.1; FedCMOO [33] employs a global learning rate of 1 and a local learning rate of 0.25; and FSMGDA-M-VR adopts a global learning rate of 1 with a local learning rate of 0.5. For the MNIST+FMNIST experiments, a similar configuration strategy is followed. FSMGDA uses global and local learning rates of 2 and 0.1, respectively; FedCMOO uses 1.2 and 0.2; and FSMGDA-M-VR uses 1.2 and 0.5. For the CIFAR10+MNIST experiments, FSMGDA and FedCMOO share the same learning-rate settings, with a global learning rate of 1.6 and a local learning rate of 0.3, whereas FSMGDA-M-VR uses a global learning rate of 1.6 and a local learning rate of 0.5.

## 5.2. Comparative analysis of algorithm performance

As shown in Figures 1 and 2, FSMGDA-M-VR demonstrates clear advantages over FedCMOO and FSMGDA in terms test accuracy and test loss on the MultiMNIST, MNIST+FMNIST, and CIFAR10+MNIST datasets. In terms of accuracy, FSMGDA-M-VR consistently achieves faster convergence in the early training rounds, reaching higher performance levels more rapidly than the other methods, particularly on the MultiMNIST and MNIST+FMNIST datasets, while matching or slightly surpassing the final accuracy of FedCMOO. Regarding test loss, FSMGDA-M-VR exhibits a steeper and more stable decline, indicating more efficient optimization and better robustness against gradient fluctuations. This smoother loss reduction, coupled with its rapid accuracy improvement, suggests that the momentum-based variance reduction mechanism effectively enhances training stability and convergence speed. Overall, FSMGDA-M-VR maintains communication efficiency while delivering superior or comparable optimization performance relative to FedCMOO and significantly outperforming FSMGDA in both convergence rate and stability.

![](images/90bfa08184167159ffadd3e1f768a68f4c0e21d88490aa168f44930ee5a0f330.jpg)

![](images/7aa1aa992b91dde43b4cd15f7702732d78d37112a2f5580be1ccc2a9140d876b.jpg)

![](images/91fa1ea355a6ea59e52213062c88eab4f2716179292c10ffd64ff27eb2d48b20.jpg)  
Figure 1: Mean test accuracy with MultiMNIST, MNIST+FMNIST and CIFAR10+MNIST datasets.

Table 1 reports the final test accuracies of the three algorithms on the three bi-objective tasks, reported as mean ± standard deviation over 10 different random seeds. FSMGDA-M-VR performs particularly well on MultiMNIST and CIFAR10+MNIST: it achieves the highest mean accuracy on the first objective and ties with FedCMOO in mean accuracy on the second objective of MultiMNIST, while obtaining the highest mean accuracies on both objectives of CIFAR10+MNIST. On MNIST+FMNIST, FSMGDA-M-VR achieves the highest mean accuracy on the first objective, whereas FedCMOO performs better on the second objective.

![](images/30b683ba711595e41d1a512b3c42913764adb57ce6da0b2298b6ac2464f9f58d.jpg)

![](images/6f7cfa06492f7f5e07042f41e7868ed164cf9ccf4a3957396cedfebf6c356842.jpg)

![](images/b26b8937277d9f275953d7c0cf1d7df3adafa1b84316dcd0a5d00f2a7a9f1a6a.jpg)

Figure 2: Mean test loss in MultiMNIST, MNIST+FMNIST and CIFAR10+MNIST datasets.
<table><tr><td>Experiment</td><td>MNIST+FMNIST</td><td>MultiMNIST</td><td>CIFAR10+MNIST</td></tr><tr><td>FSMGDA</td><td> $9 3 . 2 { \pm } 1 . 1 $  1  $7 3 . 4 { \pm } 0 . 9 $ </td><td> $9 1 . 1 { \pm } 0 . 6 $  1  $8 7 . 1 { \pm } 0 . 9 $ </td><td> $5 4 . 2 { \pm } 0 . 7 \ $  1  $9 3 . 3 { \pm } 1 . 1 $ </td></tr><tr><td>FedCMOO</td><td> $9 4 . 9 { \pm } 1 . 0 \ ,$  1  ${ \bf 7 7 . 5 { \pm 0 . 8 } }$ </td><td> $9 4 . 3 { \pm } 0 . 7 \ $  1  $9 1 . 1 \pm 0 . 8$ </td><td> $5 5 . 2 { \pm } 0 . 6 $  1  $9 3 . 7 { \pm } 1 . 0 $ </td></tr><tr><td>FSMGDA-M-VR</td><td> $\mathbf { 9 6 . 4 \bot 0 . 5 }$   $/ \ 7 6 . 0 { \pm } 0 . 4 $  7</td><td> $\mathbf { 9 4 . 5 } { \pm } \mathbf { 0 . 3 }$  1  ${ \bf 9 1 . 1 } { \bf \pm 0 . 3 }$ </td><td> ${ \bf 5 7 . 2 \pm 0 . 3 }$  1  $\mathbf { 9 6 . 0 { \pm } 0 . 5 }$ </td></tr></table>

Table 1: The final accuracies (%) of the first/second objectives in 2-objective settings. The bold values indicate the best accuracy for each objective.

## 5.3. Comparison in terms of sample access cost

Figures 3 and 4 report the mean test accuracy and mean test loss of FSMGDA, FedCMOO, and FSMGDA-M-VR against the cumulative number of samples accessed per client. On MultiMNIST, FSMGDA-M-VR achieves higher test accuracy than the other two methods over most of the considered range of sample accesses and yields the lowest final test loss. On MNIST+FMNIST, FSMGDA-M-VR generally achieves higher test accuracy over a substantial portion of the sample-access range and exhibits lower test loss than the other two methods. $\mathrm { O n }$ CIFAR10+MNIST, FedCMOO exhibits faster initial accuracy improvement, while FSMGDA-M-VR catches up in later stages and attains a comparable final accuracy; its final test loss is also close to the lowest among the three methods.

![](images/7c574065e16c3f2d4802e818bb2ae29743f3a20c1c5f8de8801b75b171cf2d48.jpg)

![](images/51c09b81c8c3f5aea11237d07e3f6223c025767b507e81d65428f86e43ffb028.jpg)

![](images/c859d0e44ecaa4869c4ad3e5452dad9cdf7fc1f978b8cffc62a5b586239a8606.jpg)  
Figure 3: Test accuracy versus cumulative samples accessed per client on the MultiMNIST, MNIST+FMNIST, and CIFAR10+MNIST datasets.

![](images/73a2658302266c368c8c820c39b5dd53fd9ec0ecaf99ff266d27e09c1667bfb8.jpg)

![](images/2f0803c6b7e21b66c8ca76f2e5c4b469d0088ffd7f450462afe116c3222d609a.jpg)

![](images/eb85498ff0be4f0f462795c0e67ee0bb7a74eb06bbb27512dfc019db1c6867c7.jpg)  
Figure 4: Test loss versus cumulative samples accessed per client on the MultiMNIST, MNIST+FMNIST, and CIFAR10+MNIST datasets.

## 5.4. Comparing the local training performance of FSMGDA-M-VR, FSMGDA, and FedCMOO

Figure 5 compares the average local training progress across clients in terms of loss decrease and accuracy improvement. It can be observed that FSMGDA-M-VR demonstrates superior local optimization performance compared to FSMGDA, while maintaining results comparable to FedCMOO. Specifically, in terms of local loss reduction, FSMGDA-M-VR consistently achieves a steeper and larger decrease in loss, especially during the early rounds, indicating that it optimizes local objectives more effectively and rapidly. Regarding local accuracy improvement, FSMGDA-M-VR also exhibits sharper gains and faster convergence in the initial training phase, with its trajectory closely aligning with that of FedCMOO across both datasets.

![](images/dcfa66811859f518cdbeeb70b750dfe0e1d16ceb2321eeffa500e3af01b28585.jpg)

![](images/1a87787fdef77a0841030edb1409c3c8f17c780c338eb974178cbefbc7a109e6.jpg)  
Figure 5: Comparison of average local training progress (loss ↓ and accuracy ↑) across clients among FSMGDA, FedCMOO, and FSMGDA-M-VR.

## 5.5. Analysis of hyperparameter effects on convergence

1) Impact of the batch size: Figures 6 and 7 compare the performance of FSMGDA-M-VR with different batch sizes on the MultiMNIST, MNIST+FMNIST, and CIFAR10+MNIST datasets. The results indicate that increasing the batch size generally leads to faster convergence, higher final test accuracy, and lower test loss, while also reducing the fluctuations of the training curves. On

MultiMNIST, the larger batch sizes, particularly 128 and 256, converge substantially faster than the batch size of 16 and attain higher final accuracy and lower loss. A similar trend is observed on MNIST+FMNIST, where the performance differences between batch sizes 128 and 256 become relatively small in the later stages of training. On CIFAR10+MNIST, the effect of the batch size is more pronounced: batch sizes 128 and 256 yield much faster convergence and noticeably better final accuracy and loss than the smaller batch sizes, especially compared with the batch size of 16.

![](images/a8c9b0ac644c46df916f5cb71a54034aea3c564466eac2c2bf9d85721172be10.jpg)

![](images/2b1f3f69649df845cdf1821bcf92c9d7ed9b548ee8f83e8a852a50766498e681.jpg)

![](images/45e4b3aa38fc9c8fa8e650792df5b5c1640098954e0fa3d61b98c52a999969f6.jpg)  
Figure 6: Mean test accuracy of different batch sizes on MultiMNIST, MNIST+FMNIST and CIFAR10+MNIST datasets.

![](images/deba187e67183121b5bf509ee6e0838f4dda2b35555863bf11539177b3205888.jpg)

![](images/b03dcb8fae1af891f5fd4b78f222f06d2084004f9d74f7b6b0c377aaa407f9ad.jpg)

![](images/0cfd957900abcba0770f78e9f3900d013bf719b2565a337d387c03b53e1f945f.jpg)  
Figure 7: Mean test loss of different batch sizes on MultiMNIST, MNIST+FMNIST and CIFAR10+MNIST datasets.

2) Impact of the number of clients: Figures 8 and 9 show how the number of clients affects the performance of FSMGDA-M-VR across the MultiMNIST, MNIST+FMNIST, and CI-FAR10+MNIST datasets. The results indicate that using more clients (e.g., 30) consistently leads to higher final test accuracy and lower final test loss. This improvement is most noticeable on the more heterogeneous CIFAR10+MNIST dataset, where a larger client pool better captures diverse data, and on MultiMNIST, where correlated objectives benefit from updates aggregated across more participants. On the relatively balanced MNIST+FMNIST tasks, increasing the number of clients has little impact on final test accuracy and final test loss.

3) Impact of local steps K: As shown in Figures 10 and 11, which depict the test accuracy and loss curves, moderately increasing the number of local steps K—typically within a low to medium range—effectively accelerates convergence, improves final accuracy, and reduces loss. For instance, increasing the number of local steps K from 1 to 5 yields clear and consistent improvements in both performance and training stability across all datasets. However, further increasing the number of local steps K, for example to 50, yields diminishing returns, with no significant additional

![](images/86ffe648da6e14b9f9127529723822cb902aa270f986af8e14da52e15cb344ee.jpg)

![](images/324af5634dfbf3804165a3bea5d246a26f78d6c6897feb79b5de13b5d3e01320.jpg)

![](images/f16cc6ba605d619cb8076599ac97806f4cb03079ec4b674aa68ed5c64a9f0613.jpg)

Figure 8: Mean test accuracy of different client numbers on MultiMNIST, MNIST+FMNIST and CIFAR10+MNIST datasets.  
![](images/9fefb22bc9268f5db8a9dc65bb31ac8376b715303a4d79554a50e009a3bd46ef.jpg)

![](images/004c4af34defdccaa7c6f2c9ff3343c56d8f823c1e8b68faa61fc1364bebf5cc.jpg)

![](images/641e1d23a4898c84a833ece3c5892837c4ff2133f59f4ad6d22c348d6bc37fe2.jpg)  
Figure 9: Mean test loss of different client numbers on MultiMNIST, MNIST+FMNIST and CIFAR10+MNIST datasets.

improvement in accuracy or reduction in loss.

![](images/351369792d75aad7a940f88596763023464793922b6a71b9d38da459e92015ae.jpg)

![](images/e62b397ede7dc51f9709ef4e7564d90c253c6ed96aa04a83136d39da61531abb.jpg)

![](images/e82a88d2e7d684669ae615e3d64fe80290614997078e0d657000f774313f02c5.jpg)  
Figure 10: Mean test accuracy for different numbers of local steps K.

## 6. Conclusion

This paper addressed federated multiobjective optimization by proposing a momentum-based variance-reduced algorithm that incorporates a momentum-driven gradient estimator into the local updates to reduce the variance and bias of stochastic updates. Theoretically, under the stated assumptions, the proposed algorithm achieves a convergence rate of $\mathcal { O } ( T ^ { - 2 / 3 } )$ , improving upon the $\mathcal { O } ( T ^ { - 1 / 2 } )$ rates of existing methods such as FSMGDA and FedCMOO. Extensive experiments on federated multiobjective optimization benchmarks further demonstrated the practical effectiveness and competitive performance of the proposed method.

![](images/be0195666f3629605175c8be9e05c7cfb1b0ce3e04a694bf9499617b122160b4.jpg)

![](images/a4706de49826875f9e4b2d52cf684c647599bd6e6753d51e462d508dc2741d45.jpg)

![](images/bd49cc4cd5953f26a919fef74cd888ac0906897af7a69fcfe5d0f97ad082eb6c.jpg)  
Figure 11: Mean test loss for different numbers of local steps K.

Acknowledgements. This work was supported in part by NSFC (No. 12271067), the Natural Science Foundation of Chongqing (No. CSTB2025NSCQ-LZX0060), the Science and Technology Research Program of Chongqing Education Commission (No. KJQN202400760), and the Australian Research Council (ARC DP230101749).

## References

[1] O. Sener, V. Koltun, Multi-task learning as multi-objective optimization, Advances in Neural Information Processing Systems, 31 (2018).

[2] T. C. Zhou, M. Momma, C. S. Dong, F. Yang, C. H. Guo, J. Shang, J. K. Liu, Multi-task learning on heterogeneous graph neural network for substitute recommendation, Proceedings of the 19th International Workshop on Mining and Learning with Graphs (MLG) (2023).

[3] D. Mahapatra, C. S. Dong, Y. T. Chen, M. Momma, Multi-label learning to rank through multi-objective optimization, Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 4605–4616 (2023).

[4] M. Momma, A. Bagheri Garakani, N. Ma, Y. Sun, Multi-objective ranking via constrained optimization, Companion Proceedings of the Web Conference 2020, 111–112 (2020).

[5] N. Mohamed Alshabibi, A.-H. Matar, M. H. Abdelati, Multi-objective mixed-integer linear programming for dynamic fleet scheduling, multi-modal transport optimization, and risk-aware logistics, Sustainability, 17(10): 4707 (2025).

[6] A. R. Jafarian-Moghaddam, Economical speed for optimizing the travel time and energy consumption in train scheduling using a fuzzy multi-objective model, Urban Rail Transit, 7(3): 191–208 (2021).

[7] X. F. Yang, Y. C. Qi, Research on optimization of multi-objective regional public transportation scheduling, Algorithms, 14(4): 108 (2021).

[8] B. H. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas, Communicationefficient learning of deep networks from decentralized data, Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS), 54: 1273–1282 (2017).

[9] P. Kairouz, H. B. McMahan, B. Avent, A. Bellet, M. Bennis, A. N. Bhagoji, K. Bonawitz, Z. Charles, G. Cormode, R. Cummings, et al., Advances and open problems in federated learning, Foundations and Trends in Machine Learning, 14(1–2): 1–210 (2021).

[10] S. U. Stich, S. P. Karimireddy, The error-feedback framework: Better rates for SGD with delayed gradients and compressed communication, Journal of Machine Learning Research, 21: 1–36 (2020).

[11] H. Yu, S. Yang, S. H. Zhu, Parallel restarted SGD with faster convergence and less communication: Demystifying why model averaging works for deep learning, Proceedings of the AAAI Conference on Artificial Intelligence, 33(01): 5693–5700 (2019).

[12] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, V. Smith, Federated optimization in heterogeneous networks, Proceedings of Machine Learning and Systems, 2: 429–450 (2020).

[13] T. Li, S. Y. Hu, A. Beirami, V. Smith, Ditto: Fair and robust federated learning through personalization, Proceedings of the 38th International Conference on Machine Learning (ICML), 139: 6357–6368 (2021).

[14] C. T. Dinh, N. H. Tran, T. D. Nguyen, Personalized federated learning with Moreau envelopes, Advances in Neural Information Processing Systems, 33: 21394–21405 (2020).

[15] Y. Mansour, M. Mohri, J. Ro, A. T. Suresh, Three approaches for personalization with applications to federated learning, arXiv preprint arXiv:2002.10619 (2020).

[16] Y. Sun, L. Shen, S. X. Chen, L. Ding, D. C. Tao, Dynamic regularized sharpness aware minimization in federated learning: Approaching global consistency and smooth landscape, Proceedings of the 40th International Conference on Machine Learning (ICML), 202: 32991– 33013 (2023).

[17] C. T. Dinh, T. T. Vu, N. H. Tran, M. N. Dao, H. Zhang, A new look and convergence rate of federated multitask learning with Laplacian regularization, IEEE Transactions on Neural Networks and Learning Systems, 35(6): 8075–8085 (2024).

[18] W. Liu, L. Chen, Y. F. Chen, W. Y. Zhang, Accelerating federated learning via momentum gradient descent, IEEE Transactions on Parallel and Distributed Systems, 31(8): 1754–1766 (2020).

[19] Z. Y. Huo, Q. Yang, B. Gu, L. Carin, H. Huang, Faster on-device training using new federated momentum algorithm, arXiv preprint arXiv:2002.02090 (2020).

[20] J. Xu, S. Wang, L. W. Wang, A. C.-C. Yao, FedCM: Federated learning with client-level momentum, arXiv preprint arXiv:2106.10874 (2021).

[21] J. H. Sun, X. D. Wu, H. Huang, A. D. Zhang, On the role of server momentum in federated learning, Proceedings of the AAAI Conference on Artificial Intelligence, 38(13): 15164–15172 (2024).

[22] S. J. Reddi, Z. Charles, M. Zaheer, Z. Garrett, K. Rush, J. Konečný, S. Kumar, H. B. McMahan, Adaptive federated optimization, International Conference on Learning Representations (ICLR) (2021).

[23] S. J. Reddi, M. Zaheer, D. Sachan, S. Kale, S. Kumar, Adaptive methods for nonconvex optimization, Advances in Neural Information Processing Systems, 31 (2018).

[24] R. Ward, X. X. Wu, L. Bottou, Adagrad stepsizes: Sharp convergence over nonconvex landscapes, Proceedings of the 36th International Conference on Machine Learning (ICML), 97: 6677–6686 (2019).

[25] S. P. Karimireddy, S. Kale, M. Mohri, S. Reddi, S. Stich, A. T. Suresh, SCAFFOLD: Stochastic controlled averaging for federated learning, Proceedings of the 37th International Conference on Machine Learning (ICML), 119: 5132–5143 (2020).

[26] X. F. Liang, S. H. Shen, J. C. Liu, Z. Pan, E. H. Chen, Y. F. Cheng, Variance reduced local SGD with lower communication complexity, arXiv preprint arXiv:1912.12844 (2019).

[27] T. Murata, T. Suzuki, Bias-variance reduced local SGD for less heterogeneous federated learning, Proceedings of the 38th International Conference on Machine Learning (ICML), 139: 7872–7881 (2021).

[28] L. Gao, H. Z. Fu, L. Li, Y. W. Chen, M. Xu, C.-Z. Xu, FedDC: Federated learning with non-IID data via local drift decoupling and correction, Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10112–10121 (2022).

[29] Z. H. Cheng, X. M. Huang, P. F. Wu, K. Yuan, Momentum benefits non-IID federated learning simply and provably, International Conference on Learning Representations (ICLR) (2024).

[30] P. Khanduri, P. Sharma, H. B. Yang, M. Y. Hong, J. Liu, K. Rajawat, P. K. Varshney, STEM: A stochastic two-sided momentum algorithm achieving near-optimal sample and communication complexities for federated learning, Advances in Neural Information Processing Systems, 34: 6050–6061 (2021).

[31] S. S. Yang, F. Y. Zhao, Z. H. Zhou, L. Shi, X. B. Ren, Z. B. Xu, Review of mathematical optimization in federated learning, CSIAM Transactions on Applied Mathematics, 6(2): 207- 249 (2025).

[32] H. B. Yang, Z. Q. Liu, J. Liu, C. S. Dong, M. Momma, Federated multi-objective learning, Advances in Neural Information Processing Systems, 36: 39602–39625 (2023).

[33] B. Askin, P. Sharma, G. Joshi, C. Joe-Wong, Federated communication-efficient multi-objective optimization, Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS), 258: 4627–4635 (2025).

[34] M. Hartmann, G. Danoy, P. Bouvry, Multi-objective methods in federated learning: A survey and taxonomy, arXiv preprint arXiv:2502.03108 (2025).

[35] H. Fernando, L. S. Chen, S. T. Lu, P.-Y. Chen, M. Liu, S. Chaudhury, K. Murugesan, G. W. Liu, M. Wang, T. Y. Chen, Variance reduction can improve trade-off in multi-objective learning, IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 6975–6979 (2024).

[36] S. J. Zhou, W. P. Zhang, J. Y. Jiang, W. L. Zhong, J. J. Gu, W. W. Zhu, On the convergence of stochastic multi-objective gradient manipulation and beyond, Advances in Neural Information Processing Systems, 35: 38103–38115 (2022).

[37] H. Fernando, H. Shen, M. Liu, S. Chaudhury, K. Murugesan, T. Y. Chen, Mitigating gradient bias in multi-objective learning: A provably convergent approach, International Conference on Learning Representations (ICLR) (2023).

[38] P. Y. Xiao, H. Ban, K. Y. Ji, Direction-oriented multi-objective learning: Simple and provable stochastic algorithms, Advances in Neural Information Processing Systems, 36: 4509–4533 (2023).

[39] M. J. Xu, P. Z. Ju, J. Liu, H. B. Yang, PSMGD: Periodic stochastic multi-gradient descent for fast multi-objective optimization, Proceedings of the AAAI Conference on Artificial Intelligence, 39(20): 21770–21778 (2025).

[40] D. A. E. Acar, Y. Zhao, R. Matas, M. Mattina, P. Whatmough, V. Saligrama, Federated learning based on dynamic regularization, International Conference on Learning Representations (ICLR) (2021).