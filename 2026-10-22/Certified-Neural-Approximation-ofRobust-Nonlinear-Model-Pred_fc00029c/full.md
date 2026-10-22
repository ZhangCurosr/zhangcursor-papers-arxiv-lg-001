# Certified Neural Approximation ofRobust Nonlinear Model Predictive Control

Hendrik Alsmeier <sup>a</sup>, Maik Pfefferkorn <sup>a</sup>, Rolf Findeisen\* <sup>a</sup>

<sup>a</sup>Control and Cyber-Physical Systems Laboratory, Technical University of Darmstadt, Landgraf-Georg-Str. 4, 64283 Darmstadt, Germany

## Abstract

Model predictive control is a high-performance control method, but it incurs comparatively high online computational costs across many control tasks and is challenging to realize in low-level hardware settings. To address this, we present a framework for generating neural network controllers that provably inherit the safety and stability guarantees of a robust model predictive policy. In contrast to approache based on post-verification or additional online safety filters, our method provides constructive, a priori design conditions that ensure preservation of recursive feasibility and stability before deployment. We treat the approximation error as a bounded input disturbance, which is addressed in the MPC design. This allows the neural network controller to directly inherit robustness from the underlying model predictive controller, which enables the transfer of guarantees to the approximation. The design process is an offline, constructive synthesis pipeline and can be implemented once Lipschitz bounds are specified. We further demonstrate in simulation that the learned controller reproduces the closed-loop behavior of the original controller while reducing online computation.

## 1 Introduction

Model predictive control (MPC) has been well-established as a reliable feedback control approach that is based on solving an underlying optimal control problem repeatedly and online in a receding horizon fashion [11, 21, 28]. As such, it can handle a variety of systems, can include input, state, and output constraints, and ensure near-optimal closed-loop behavior with respect to a given control objective. These properties have led to its widespread adoption in industry and research. However, the need to solve the optimal control problem online can impose a computational burden, posing implementation challenges in resource-constrained environments, such as embedded hardware, due to the demanding nature of the optimization process.

To address this challenge, various approaches have been proposed. For example, in the linear–quadratic case with linear dynamics, quadratic costs, and polytopic state and input constraints, explicit MPC yields a feedback law that is piecewise affine in the state. This structural property enables explicit MPC to compute the feedback law offline in closed form, thereby avoiding the need to solve an optimization problem online [1, 35]. However, the number of regions of the piecewise affine law grows, in the worst case, exponentially with the total number of inequality constraints, which in turn increases with the horizon length and the number of constrained states and inputs. Consequently, both the offline computation and memory complexity become rapidly intractable as the problem size grows.

This motivates the development of approximation- and learning-based approaches that mimic the implicit MPC feedback law [7, 8, 12, 15, 17, 20, 22–24, 27, 29, 33]. Among the available function approximators, neural networks (NNs) are particularly attractive and employed in this work due to their universal approximation capabilities [13, 14] and their ability to maintain a fixed evaluation effort with growing state dimension, input dimension, and dataset sizes. These properties have motivated numerous neural imitation-learning approaches for approximating MPC feedback laws. Nevertheless, the choice of function approximator alone does not determine how closed-loop guarantees are obtained. Existing approaches can be distinguished more coherently by how the approximation error is incorporated into the control design and how compliance with a prescribed error bound is certified.

One prominent guarantee-transfer paradigm consists of two steps. First, the MPC is designed to be robust against a prescribed approximation error, which is interpreted as a bounded disturbance acting on the system state. Second, an explicit approximation of the resulting MPC feedback law is constructed such that its error satisfies the prescribed bound. An early NN-based formulation of this approach was presented in [12], where Hoeffding’s inequality is used to validate statistically whether the learned controller satisfies the admissible input-error bound. If validation fails, additional samples are generated, and the learning procedure is repeated. Building on this framework, [24] extends the approach to nonlinear setpoint tracking and introduces a less conservative validation criterion that jointly accounts for approximation error and model mismatch. In [2], statistical post-validation is replaced by a deterministic condition that combines Lipschitz properties of the MPC feedback law and the NN with the coverage of the training data.

This disturbance-based guarantee-transfer principle is independent of the chosen function approximator. In [29], a Gaussian process is used to learn the MPC feedback law, and compliance with the required error bound is established through tailored training-data selection. Recently, [33] employs automatic kernel interpolation to construct an explicit surrogate whose approximation error is uniformly bounded by design, under the assumption that the feedback law belongs to the associated reproducing kernel Hilbert space (RKHS) with a known norm bound. Similarly, [30] uses kernel interpolation on the closed-loop reachable set and Monte Carlo reachability analysis to certify safety and closed-loop performance. These approaches share the disturbance-based guarantee-transfer principle of the NN methods above, but differ in the choice of approximator, the associated regularity assumptions, and the mechanism used to certify the approximation error.

Other NN-based approaches obtain guarantees through direct error certification, structural restrictions, or online correction mechanisms. In [8], semidefinite programming is used to bound the worst-case difference between the NN and MPC feedback laws, while [27] uses an optimization-based projection to preserve constraint satisfaction and stability. Building on constrained NN approximations of explicit MPC laws [7,23], deep piecewise-affine networks provide approximation guarantees by design [17]. For nonlinear MPC, [15] validates NN proposals online and uses an MPC-intrinsic fallback when needed, while [20, 22] learn MPC laws from trajectory data, with [22] improving data efficiency through gradient-informed Sobolev training.

Against this background, we develop a constructive framework for the certified neural approximation of robust nonlinear MPC. The framework adopts the disturbance-based viewpoint of [12,24,29,33], but derives a deterministic certificate from dataset coverage and Lipschitz bounds, thereby preserving closed-loop guarantees without relying on statistical validation or requiring membership of the MPC feedback law in a prescribed RKHS with a corresponding norm bound. We generalize the framework of [2] to MPC feedback laws that need not be globally Lipschitz continuous, establishing guarantee transfer under local regularity conditions along the closed-loop trajectory. To this end, we first design a robust MPC for a prescribed disturbance bound and derive conditions ensuring that a neural network trained on MPC state-input data satisfies this bound, thereby preserving safety and stability without post-verification or auxiliary safety mechanisms. We further validate these guarantees in closed-loop operation. Our main contributions are:

(i) A priori conditions are derived that link the Lipschitz properties of the MPC feedback law and the coverage of the training data to certified bounds on the approximation error.

(ii) A constructive offline procedure is developed to obtain a neural network approximator that satisfies these bounds, enabling systematic training of the network.

(iii) It is shown that, under these conditions, the learned neural network controller inherits the recursive feasibility and stability properties of the robust MPC, ensuring safe and reliable closed-loop operation.

The remainder of this article is structured as follows. In Section 2, we formulate the problem and introduce the robust MPC scheme. Section 3 introduces the basics of the neural networks used, details our theoretical results on the neural network approximation of the underlying MPC, and further includes an overview of the constructive procedure enabling the generation of the neural networks in line with the previously established theory. We illustrate the effectiveness of our proposed approach in Section 4 in simulation using a 4-dimensional continuously stirred tank reactor example before concluding in Section 5.

## 2 Robust MPC as a Basis for Certified Controller Approximation

We first show that the neural network approximation error enters the system as a bounded input disturbance. Using Lipschitz continuity of the system dynamics, this input disturbance can be equivalently reformulated as a bounded additive state disturbance. This reformulation allows us to cast the closed-loop system in a standard robust control setting. Based on this representation, we introduce the robust nonlinear MPC scheme that forms the basis for transferring safety and stability guarantees to the neural network controller. The presentation focuses on a robust MPC formulation for additive state disturbances, which is sufficient for our subsequent analysis. We conclude the section by recalling the key closed-loop properties of the robust MPC, in particular, recursive feasibility and input-to-state stability.

## 2.1 System Description and Control Task

We consider nonlinear discrete-time systems given by

$$
x _ { k + 1 } = f ( x _ { k } , u _ { k } ) ,\tag{1}
$$

where $x _ { k } \in \mathbb { R } ^ { n _ { x } }$ and $u _ { k } \in \mathbb { R } ^ { n _ { u } }$ describe the system state and input, respectively, and $k \in  { \mathbb { N } } _ { 0 }$ denotes the discrete time index. The transition map is given by $f : \mathbb { R } ^ { n _ { x } } \times \mathbb { R } ^ { n _ { u } } \to$ $\mathbb { R } ^ { n _ { x } }$ and captures the dynamics of the system. The objective is to stabilize the system (1) at the origin (w.l.o.g.) while satisfying input and state constraints given by

$$
\forall k \in \mathbb { N } : \ u _ { k } \in \mathcal { U } \subset \mathbb { R } ^ { n _ { u } } , \ x _ { k } \in \mathcal { X } \subset \mathbb { R } ^ { n _ { x } } .\tag{2}
$$

While model predictive control is well-suited for this task, it requires the online solution of a possibly large-scale optimization problem, typically through iterative methods. Thus, we aim to control the system (1) by a low-complexity neural network approximation $\kappa _ { \mathrm { N N } }$ of the model predictive feedback law $u = \kappa ( x )$ to reduce computational demands or enable formal verification of the explicit feedback law. However, the use of the neural network introduces an approximation error

$$
e ( x ) = \kappa ( x ) - \kappa _ { \mathrm { N N } } ( x ) ,\tag{3}
$$

which manifests as an additive disturbance on the control input. If this disturbance is appropriately accounted for in the MPC design, the safety and stability guarantees can be transferred to the neural network controller.

To do so, we rely on a pre-specified bound on the neural network’s approximation error, which we ensure through suitable data selection and training. That is, we assume that we know $\epsilon > 0$ such that the approximation error in (3) satisfies $\forall x \in \mathcal { X } : \ \| e \| \leq \epsilon . ^ { 1 }$ We then reformulate the resulting bounded input disturbance as a bounded additive disturbance acting on the state of the system (1) under the feedback law κ. This state disturbance is then addressed within a robust MPC formulation, allowing guarantees established for the closed-loop system under the robust MPC to carry over to any approximate controller that satisfies the prescribed error bound. The crucial step is to design a neural network that provably adheres to the predefined approximation error. To this end, we exploit Lipschitz continuity properties of both the MPC feedback law and the neural network, together with a data coverage criterion for the training data with respect to the domain X.

We make the following assumptions on the properties of the system and constraint sets.

Assumption 1 The transition map $f : \mathbb { R } ^ { n _ { x } } \times \mathbb { R } ^ { n _ { u } }  \mathbb { R } ^ { n _ { x } }$ is twice continuously differentiable in x and u on $\mathbb { R } ^ { n _ { x } } \times \mathbb { R } ^ { n _ { u } }$ denoted by $f \in \dot { C } ^ { 2 } ( \mathbb { R } ^ { n _ { x } } \times \mathbb { R } ^ { n _ { u } } , \mathbb { R } ^ { n _ { x } } )$ . Moreover, it holds that $f ( 0 , 0 ) = 0$

Note that the differentiability property in Assumption 1 is needed to apply a known result from parametric optimization regarding the structural properties of the MPC feedback law, which constitutes the first of two key components enabling the transfer of guarantees from the robust MPC to the neural network controller. In practice, many commonly considered systems satisfy this assumption.

Assumption 2 The sets U and X are compact and given by $\mathcal { U } : \stackrel { - } { = } \{ g _ { i } ^ { u } ( u ) \leq 0 \mid i = 1 , \ldots , n _ { c } ^ { u } \}$ and ${ \mathrm { \bar { \mathcal { X } } } } : = \{ g _ { i } ^ { x } ( x ) \leq$ $0 \mid i = { \mathrm { i } } , \ldots , n _ { c } ^ { x } \}$ with twice continuously differentiable functions $g _ { i } ^ { u } ( u ) \in \overset { \vartriangle } { C ^ { 2 } } ( \mathbb { R } ^ { n _ { u } } , \mathbb { R } )$ and $g _ { i } ^ { x } ( x ) \ \stackrel { \cdot } { \in } \ \stackrel { \cdot \cdot } { C ^ { 2 } } ( \mathbb { R } ^ { n _ { x } } , \mathbb { R } )$ Moreover, we have that $( 0 , 0 ) \in \operatorname { i n t } ( \mathcal { X } \times \mathcal { U } )$

As before, Assumption 2 is necessary to ensure the desired structural properties of the optimal control problem underlying the MPC feedback law. However, most practical MPC formulations employ polytopic or ellipsoidal sets for X and U, for which Assumption 2 holds.

## 2.2 From Input Approximation Error to State Disturbance

Most robust MPC formulations account for state disturbances. Therefore, we reformulate the approximation error as an equivalent state disturbance, enabling the application of a broad class of robust MPC formulations that provide a priori guarantees of closed-loop robustness to the neural network controller’s approximation error.

Assumptions 1 and 2 imply Lipschitz continuity of the transition map f with respect to both x and u on the compact domain $\bar { \mathcal X } \times \mathcal { U }$ . That is, there exist finite constants $L _ { f , x } > 0$ and $L _ { f , u } > 0$ such that

$$
\| f ( x , u ) - f ( x , u ^ { \prime } ) \| \leq L _ { f , u } \| u - u ^ { \prime } \| ,\tag{4a}
$$

$$
\| f ( x , u ) - f ( x ^ { \prime } , u ) \| \leq L _ { f , x } \| x - x ^ { \prime } \| ,\tag{4b}
$$

hold for all $x , x ^ { \prime } \in { \mathcal { X } }$ and $u , u ^ { \prime } \in \mathcal { U } .$ . Hence, (4a) implies that the bounded input approximation error induces a bounded additive disturbance w on the system states,

$$
\begin{array} { r l } & { x _ { k + 1 } = f ( x _ { k } , u _ { k } + e _ { k } ) \equiv f ( x _ { k } , u _ { k } ) + w _ { k } , \mathrm { ~ w i t h ~ } } \\ & { \| w _ { k } \| = \| f ( x _ { k } , u _ { k } + e _ { k } ) - f ( x _ { k } , u _ { k } ) \| \le L _ { f , u } \epsilon , } \end{array}\tag{5}
$$

exploiting knowledge of a bound ϵ > 0 on the approximation error. We emphasize that this approximation error bound is a design parameter. We will later show that it is indeed possible to construct a neural network approximator that satisfies any prescribed bound.

Having rewritten the approximation error as a bounded additive disturbance in (5), the closed-loop dynamics under the neural network controller fit the standard setting of robust MPC.

## 2.3 Robust MPC Design and Closed-Loop Guarantees

The general idea of robust MPC schemes is to analyze a nominal system model subject to appropriately tightened constraints, such that satisfaction of the tightened constraints by the undisturbed system implies satisfaction of the original constraints by the disturbed system. This enables the establishment of robust recursive feasibility guarantees in closed-loop operation. Furthermore, the optimal value function of a robust MPC scheme is typically used as a Lyapunov function candidate to derive robust stability guarantees for the closed-loop system.

Without loss of generality, we adopt the robust MPC scheme proposed in $[ 1 9 ] ^ { 2 }$ , whose underlying optimal control prob lem is given by

$$
\operatorname* { m i n } _ { \mathbf { u } _ { k } } \quad \sum _ { n = 0 } ^ { N - 1 } l ( \bar { x } _ { n | k } , u _ { n | k } ) + V _ { \mathrm { T } } ( \bar { x } _ { N | k } )\tag{6a}
$$

$$
\mathrm { s . ~ t . ~ } \forall n \in \{ 0 , \ldots , N - 1 \} :
$$

$$
\bar { x } _ { n + 1 | k } = f \bigl ( \bar { x } _ { n | k } , u _ { n | k } \bigr ) , \bar { x } _ { 0 | k } = x _ { k } ,\tag{6b}
$$

$$
\begin{array} { r } { u _ { n | k } \in \mathcal { U } \ominus \{ u \in \mathbb { R } ^ { n _ { u } } \ | \ \| u \| \le \epsilon \} , } \end{array}\tag{6c}
$$

$$
\bar { x } _ { n | k } \in \mathcal X \ominus \mathcal W _ { n } ,\tag{6d}
$$

$$
\bar { x } _ { N | k } \in \mathcal { X } _ { \mathrm { T } } = \{ x \in \mathbb { R } ^ { n _ { x } } \mid V _ { \mathrm { T } } ( x ) \leq \alpha _ { v } \} .\tag{6e}
$$

Here, ${ \bar { x } } _ { n \mid k }$ denotes the nominal state prediction n steps ahead of the current time $k , u _ { n | k }$ denotes the planned future control inputs n steps ahead of the current time $k ,$ and $N \in  { \mathbb { N } }$ , denotes the prediction horizon. The nominal state predictions are computed using the nominal system model (6b) initialized with the most recent state information $x _ { k }$ . The control objective is encoded in the cost function in (6a), which is composed of the stage cost $l : \mathbb { R } ^ { n _ { x } } \times \mathbb { R } ^ { n _ { u } } \to [ 0 , \infty )$ and the terminal cost $V _ { \mathrm { T } } : \mathbb { R } ^ { n _ { x } }  \lceil 0 , \infty $ . We assume the following for the stage and the terminal cost functions.

Assumption 3 The stage and the terminal cost functions are twice continuously differentiable, i.e., $l \in \check { C } ^ { 2 } ( \mathbb { R } ^ { n _ { x } } \times$ $\mathbb { R } ^ { n _ { u } } , [ 0 , \infty ) )$ and $V _ { \mathrm { T } } \stackrel { \cdot } { \in } \tilde { C } ^ { 2 } ( \mathbb { R } ^ { n _ { x } } , [ 0 , \infty ) )$ . Moreover, l and V<sub>T</sub> are positive definite w.r.t. to the origin.

Note that Assumption 3 is satisfied by many commonly used stage and terminal cost functions, including quadratic costs typically employed in regulation (and, by extension, tracking) problems. Moreover, Assumption 3 implies that the stage and terminal cost functions are Lipschitz continuous on the compact domain ${ \mathcal { X } } \times { \mathcal { U } }$ and X, respectively, with Lipschitz constants $L _ { l } , L _ { v } > 0$

We impose the input constraint in (6c), where $\ominus$ denotes the Pontryagin difference. This tightening of the original input constraints in (2) accounts for the approximation error introduced by the neural network controller.

Similarly, the state constraints in (2) appear in a tightened form in (6d), where they are imposed on the nominal system state. This tightening captures the effect of the (propagated) approximation error of the neural network controller on the system’s behavior. It is performed based on the sets

$$
\mathcal { W } _ { n } = \left\{ e \in \mathbb { R } ^ { n _ { x } } \mid \Vert e \Vert \leq \frac { L _ { f , x } ^ { n } - 1 } { L _ { f , x } - 1 } L _ { f , u } \epsilon \right\} .\tag{7}
$$

Here the sets $\mathcal { W } _ { n }$ overapproximate the possible — but a priori unknown — realizations of the mismatch between the nominal system and the disturbed system after n steps. Hence, the constraint tightening in (6d) ensures that the state constraints will be satisfied by the system (1) under the approximate neural network controller.

The terminal region constraint in (6e), with a suitably chosen parameter $\alpha _ { v } > 0 .$ is used to guarantee recursive feasibility and stability of the closed-loop system under the MPC.

The optimal control problem (6) is repeatedly solved at any time k with updated initial condition $x _ { k } ,$ and the first element $u _ { 0 | k } ^ { * }$ of the optimal input sequence $\mathbf { u } _ { k } ^ { * } = \big ( u _ { 0 | k } ^ { * } , \ldots , u _ { N - 1 | k } ^ { * } \big )$ is applied to the system (1). This implicitly defines the MPC feedback law

$$
u _ { k } = \kappa ( x _ { k } ) , \kappa ( x _ { k } ) : = u _ { 0 | k } ^ { * } .\tag{8}
$$

We assume that the following conditions hold [19].

Assumption 4 Let $\alpha > 0$ and define the set $\Phi : = \{ x \in \mathsf { \Gamma }$ $\mathcal { X } \mid \bar { V _ { \mathrm { T } } ( x ) } \le \alpha \}$ . Assume there exists a terminal feedback controller $u = \dot { \kappa _ { \mathrm { T } } } ( x ) , \kappa _ { \mathrm { T } } : \Phi  \mathcal { U }$ and a constant $\alpha _ { v }$ with $0 < \alpha _ { v } \le \alpha$ such that:

(i) The terminal controller satisfies the tightened input constraints when evaluated for $x \in \Phi$ , i.e., $\begin{array} { r } { \mathbf { \dot { \Phi } } \subset \left\{ x \in \mathcal { X } \mid \kappa _ { \mathrm { T } } ( x ) \in \mathcal { U } \ominus \left\{ u \in \mathbb { R } ^ { n _ { u } } \mid \| u \| \le \epsilon \right\} \right\} } \end{array}$

(ii) The set Φ and the terminal controller are such that $f ( x , \kappa _ { \mathrm { T } } ( x ) ) ~ \in ~ \mathcal { X } _ { \mathrm { T } }$ holds for all $x \in \Phi$ , where $\mathcal { X } _ { \mathrm { T } } = \{ x \in \mathbb { R } ^ { n _ { x } } \mid V _ { \mathrm { T } } ( x ) \leq \alpha _ { v } \}$ is the terminal region used in the optimal control problem in (6e). Consequently, Φ is robustly forward invariant for the system (5) under $u = \kappa _ { \mathrm { T } } ( x )$ .

(iii) The terminal cost $V _ { \mathrm { T } }$ is a Lyapunov function for the nominal system under the terminal controller on Φ, i.e., there exist class- $\kappa _ { \infty }$ functions $\underline { { \alpha } } , \overline { { \alpha } }$ satisfying

$$
\begin{array} { r l } & { \qquad \underline { { \alpha } } ( \| x \| ) \le V _ { \mathrm { T } } ( x ) \le \overline { { \alpha } } ( \| x \| ) , \ a n d } \\ & { \qquad V _ { \mathrm { T } } \big ( f ( x , \kappa _ { \mathrm { T } } ( x ) ) \big ) - V _ { \mathrm { T } } ( x ) \le - l \big ( x , \kappa _ { \mathrm { T } } ( x ) \big ) , } \end{array}
$$

for all $x \in \Phi .$

(iv) The neural network approximation error ϵ satisfies $\begin{array} { r } { L _ { f , u } \epsilon \leq \frac { \alpha - \alpha _ { v } } { L _ { v } L _ { f , x } ^ { N - 1 } } . } \end{array}$

Under these assumptions, the MPC defined by the optimal control problem (6) is recursive feasible and robustly stabilizes the system (1), as stated in the following theorem.

Theorem 5 (Recursive feasibility and stability [19].) Suppose Assumptions 1, 2, 3, and 4 hold. Then, if the optimal control problem (6) is initially feasible, it remains recursively feasible. Moreover, the origin is input-to-state stable for the disturbed system (5) under the MPCfeedback law (8).

Corollary 6 The set offeasible initial conditions, defined as ${ \mathcal { X } } _ { N } \doteq \{ x \in { \mathcal { X } } \mid$ Problem (6) admits a solution}, is robustly forward invariant for the disturbed system (5) under the MPC feedback law (8). That $i s , \ x \ \in \ \mathcal { X } _ { N } \quad \Longrightarrow$ $( f ( x , \kappa ( x ) ) + w ) \in \mathcal { X } _ { N }$ holds for all disturbances w satis-$f y i n g \parallel w \parallel \le L _ { f , u } \epsilon .$

These results establish the foundation for extending the safety and stability guarantees of the MPC scheme to the neural network controller, as discussed in the next section. <sup>3</sup>

## 3 Certified Neural Approximation of the MPC Feedback Law

We begin by reviewing the fundamentals of feed-forward neural networks, which we employ as function approximators to learn the MPC feedback law. Thereafter, we present our main contribution. First, we derive conditions under which the neural network is guaranteed to satisfy a prescribed approximation error bound. Second, we discuss how safety and stability guarantees of the original MPC transfer to the neural network controller. The section concludes with an algorithm for constructing the neural network controller.

## 3.1 Feed-Forward Neural Networks

We employ neural networks to approximate the feedback law κ , motivated by their ability to represent nonlinear functions accurately [13, 14], while retaining a fixed online evaluation structure that is independent of the size of the training dataset. This is particularly attractive for our approach, where large offline datasets are required, as discussed in Section 3.2. Specifically, we restrict ourselves to feed-forward neural networks trained offline via supervised learning techniques and ensure differentiability of the neural networks by choosing differentiable activation functions. Particularly, the $j ^ { \mathrm { t h } }$ layer of a NN can be written as

$$
h _ { j } = \sigma _ { j } \circ \zeta _ { j } ( h _ { j - 1 } ) : = \sigma _ { j } \left( A _ { j } h _ { j - 1 } + b _ { j } \right) .
$$

Here, $\zeta _ { j }$ denotes the affine transformation of the $j ^ { \mathrm { t h } }$ layer’s inputs, defined through the weight matrix $A _ { j } \in \mathbb { R } ^ { n _ { h _ { j } } \times n _ { h _ { j - 1 } } }$ and the bias vector $\bar { b _ { j } } \in \mathbb { R } ^ { n _ { h _ { j } } }$ . The nonlinear function $\sigma _ { j }$ $\mathbb { R } ^ { n _ { h _ { j } } }  \mathbb { R } ^ { n _ { h _ { j } } }$ encodes the element-wise application of an activation function to $\zeta _ { j } ( h _ { j - 1 } )$ to produce the output $h _ { j } . \mathrm { A }$ feed-forward NN is now the composition of all layers and their activations, denoted as

$$
\kappa _ { \mathrm { N N } } ( x ) : = \zeta _ { m } \circ \sigma _ { m - 1 } \circ \zeta _ { m - 1 } \circ . . . \circ \sigma _ { 1 } \circ \zeta _ { 1 } ( x ) ,\tag{9}
$$

where m is the total number of layers in the network. Since the activation functions $\sigma _ { j }$ are differentiable, and $\zeta _ { j }$ is affine and therefore differentiable, their composition (9) is also differentiable. This implies that the NN (9) is Lipschitz continuous on the compact domain X.

By exploiting Lipschitz continuity of the neural network, we derive conditions under which the NN satisfies a prescribed error bound in the next section. However, it is important to note that the results presented are not limited to feedforward neural networks. Rather, any neural network that is Lipschitz continuous on X can be used. We formalize this requirement in the more general assumption below.

Assumption 7 The neural network approximator κ<sub>NN</sub> is Lipschitz continuous on $\mathcal { X } , i . e .$ , there exists a constant $0 <$ $L _ { \mathrm { N N } } < \infty$ such that $\| \kappa _ { \mathrm { N N } } ( x _ { 1 } ) - \kappa _ { \mathrm { N N } } ( x _ { 2 } ) \| \le L _ { \mathrm { N N } } \| x _ { 1 } -$ x<sub>2</sub>∥ holds for all $x _ { 1 } , x _ { 2 } \in \mathcal { X }$

## 3.2 Approximation Error Bounds

The results presented here develop the Lipschitz-based approximation approach introduced in [2, Theorem 1] into a certified approximation setting for robust nonlinear MPC. As in that work, our approach relies primarily on two key aspects: (i) the careful design of the training data and the accuracy of the NN approximator on the training data, and (ii) regularity properties of the MPC feedback law (8). We begin by discussing the requirements for the training data and the neural network’s accuracy on it.

The neural network is trained on a dataset $\mathcal { D } = \{ ( x _ { i } , \kappa ( x _ { i } ) ) ~ |$ $i \in \{ 1 , \dots , n _ { \mathcal { D } } \} \}$ consisting of n<sub>D</sub> observations of the MPC feedback law (8) on the domain X. Specifically, we will explore coverage properties of the set of training inputs, $\{ \bar { x _ { i } } | i = 1 , . . . \bar { , } n _ { \bar { D } } \}$ , on X, as defined below.

Definition 8 Let $( A , \rho )$ be a metric space and let ∞ $> \delta >$ 0. A subset ${ \mathcal { P } } \subseteq A$ provides a δ-coverage of A if, for every $a \in { \mathcal { A } } ,$ there exists $p \in \mathcal P$ such that $\rho ( a , p ) \leq \delta .$

We further assume that an arbitrary prescribed training accuracy on a dataset D can be attained through a suitable choice of network architecture and training procedure, as formalized in Assumption 10. This assumption is motivated by Lemma 9, which establishes the existence of a neural network achieving any prescribed accuracy on a finite dataset.

Lemma 9 (Existence of an ϵ<sub>D</sub>-interpolating NN; proof in supplementary Section S3.) Let $\mathcal { D } = \dot { \{ ( x _ { i } , \breve { y _ { i } } ) \in \mathbb { R } ^ { \hat { d } } \times \bar { \mathbb { R } } ^ { p } \ \vert } $ $i \in \{ 1 , \dots , n _ { \mathcal { D } } \} \}$ be a dataset with pairwise distinct inputs x<sub>i</sub> and $\epsilon _ { \mathcal { D } } > 0$ a given constant. Then there exists afeedforward neural network $\tilde { f } : \mathbb { R } ^ { d }  \mathbb { R } ^ { p }$ with any continuous nonpolynomial activationfunction such that $\| \tilde { f } ( x _ { i } ) - y _ { i } \| \le \epsilon _ { \mathcal { D } }$ for all $i = 1 , \dots , n _ { \mathit { D } }$

Assumption 10 Given $\epsilon _ { \mathcal { D } } > 0$ and a training dataset $\mathcal { D } ,$ there exists a choice of training procedure and network architecture such that the trained network κ satisfies $\| \kappa ( x _ { i } ) - \kappa _ { \mathrm { N N } } ( x _ { i } ) \| \leq \epsilon _ { \mathcal { D } }$ for all $( x _ { i } , \kappa ( x _ { i } ) ) \in \mathcal { D }$

Lemma 9 establishes the existence of a feedforward neural network with any prescribed in-sample accuracy, whereas Assumption 10 additionally requires that such accuracy be attainable through a suitable training procedure and architecture. Both statements concern only the samples in $\mathcal { D } ;$ they imply neither convergence of a particular training procedure nor generalization beyond the training data. The training-set bound is extended to a uniform approximation bound below by combining it with the regularity properties of the MPC and NN policies and the coverage of the training inputs. We therefore begin by considering the regularity of the MPC feedback law (8). To this end, we note that the optimal control problem (6) is a parametric optimization problem, where the parameter is the initial state $x _ { k }$ in (6b). Specifically, we can express the nominal state $\bar { x } _ { n | k } , n \in \{ 0 , \ldots , N \}$ , in (6b) as a function of the initial state and the input (sub-)sequence, as $\bar { x } _ { n | k } = \psi ( n ; x _ { k } , \mathbf { u } _ { k } )$ . Since $\psi$ is defined via the n-fold application of the dynamics $f ,$ and $f$ is twice continuously differentiable in x and $u ,$ ψ is twice continuously differentiable in $x _ { k }$ and $\mathbf { u } _ { k }$ . Substituting $\psi ( n ; x _ { k } , { \mathbf { u } } _ { k } )$ for the states ${ \bar { x } } _ { n \mid k }$ in the cost function (6a) and the constraints (6d) and $( 6 \dot { \mathrm { e } } )$ enables us to write the optimal control problem (6) generically as

$$
\begin{array} { r l } { \underset { \mathbf { u } _ { k } } { \mathop { \operatorname* { m i n } } } } & { J ( \mathbf { u } _ { k } , x _ { k } ) } \\ { \mathrm { s . ~ t . ~ } } & { G _ { i } ( \mathbf { u } _ { k } , x _ { k } ) \leq 0 , \ i = 1 , \dots , N ( n _ { c } ^ { u } + n _ { c } ^ { x } ) + 1 . } \end{array}\tag{10}
$$

Again, since self-compositions and sums of twice continuously differentiable functions remain twice continuously differentiable, Assumptions $1 , 2 ,$ and 3 imply that the functions $J$ and $G _ { i } , \ i = 1 , \dots , N ( n _ { c } ^ { u } + n _ { c } ^ { x } ) ^ { \cdot } { \dot { + } } 1$ in (10), are twice continuously differentiable on the interior of $\mathcal { U } ^ { N } \times \mathcal { X }$ We will explore the following result from the field of parametric optimization.

Proposition 11 (Differentiability of optimal solutions $I 6 J . )$ Given $x _ { k } ,$ , suppose that u<sup>∗</sup> is an admissible point for the parametric optimization problem (10) which satisfies the Karush-Kuhn-Tucker conditions with Lagrange multiplier vector $\mu _ { k } ^ { * }$ . The corresponding active set is given as the index set $\bar { \mathcal { Z } } _ { \mathrm { a } } \overset { \cdot \bf { \sigma } } { = } \{ i \in \{ 1 , \dots , N ( n _ { c } ^ { u } + n _ { c } ^ { x } ) + 1 \} \mid \bar { G } _ { i } ( { \bf u } _ { k } ^ { * } , x _ { k } ) = 0 \}$ Furthermore, suppose that

(i) the linear independence constraint qualification (LICQ) is satisfied at $( \mathbf { u } _ { k } ^ { * } , x _ { k } )$ , i.e., the gradients $\nabla _ { \mathbf { u } _ { k } } G _ { i } ( \mathbf { u } _ { k } ^ { * } , x _ { k } ) , \ i \in \mathcal { T } _ { \mathrm { a } }$ , are linearly independent,

(ii) strict complementary slackness (SCS) holds, i.e., $\mu _ { k , i } ^ { * } > 0 f o r a l l i \in \mathcal { T } _ { \mathrm { a } } ,$ and

(iii) the second-order sufficient condition (SOSC) is satisfied at $\mathbf { \Psi } ( \mathbf { u } _ { k } ^ { * } , x _ { k } )$ , i.e., the Hessian of the Lagrangian at $\mathbf { \Psi } ( \mathbf { u } _ { k } ^ { * } , x _ { k } )$ is positive definite when restricted to the tangent space corresponding to the active constraints.

Then, there exist a neighborhood $\mathcal { N } ( x _ { k } ) \subset \mathcal { X }$ of x<sub>k</sub> and continuously differentiable functions u $: \mathcal { N } ( x _ { k } )  \mathcal { U } ^ { N }$ and $\pmb { \mu } : \mathcal { N } ( x _ { k } ) \overset { \cdot } {  } \bar { \mathbb { R } } ^ { ( N - 1 ) ( n _ { c } ^ { u } + \overset { \cdot } { n } _ { c } ^ { x } ) + 1 }$ such that

$$
( a ) \ \mathbf { u } ( x _ { k } ) = \mathbf { u } _ { k } ^ { * } , \mu ( x _ { k } ) = \mu _ { k } ^ { * } ,
$$

(b) the active set is constant in $\mathcal { N } ( \boldsymbol { x } _ { k } )$

(c) LICQ is satisfied for all $z \in \mathcal { N } ( x _ { k } )$ , and

(d) for $a l l z \in \mathcal { N } ( x _ { k } )$ , the pair $( { \bf u } ( x _ { k } ) , \mu ( x _ { k } ) )$ is a strong local minimum of the problem (10) with initial state $z ,$ i.e., LICQ, SCS, and SOSC are satisfied.

Corollary 12 Under the same conditions as in Proposition $^ { l l , }$ the MPC feedback law $( 8 )$ is locally Lipschitz continuous on the neighborhood $\mathcal { N } ( \boldsymbol { x } _ { k } )$ of $x _ { k } , \ i . e .$ , there exists a constant $0 < L _ { \mathrm { M P C } } ( x _ { k } ) < \infty$ such that $\| \kappa ( x _ { 1 } ) - \kappa ( x _ { 2 } ) \| \le$ $L _ { \mathrm { M P C } } ( x _ { k } ) \lVert x _ { 1 } - x _ { 2 } \rVert$ holds for all $x _ { 1 } , x _ { 2 } \in \mathcal { N } ( x _ { k } )$

We now derive an approximation bound linking dataset coverage and Lipschitz constants.

Theorem 13 (Approximation Error Bound.) Suppose Assumptions $I , 2 , { \hat { 3 } } , 4 , 7 ,$ and 10 hold. Furthermore, suppose that the current state $x _ { k }$ of the system (1) is such that the conditions in Proposition 11 are satisfied. Then, for a given training dataset ${ \mathcal { D } } ,$ given $\epsilon _ { \mathcal { D } } > 0 $ , and for a prescribed $\epsilon > \epsilon _ { \mathcal { D } } ,$ the approximation error is bounded $b y$

$$
\| \kappa ( x _ { k } ) - \kappa _ { \mathrm { N N } } ( x _ { k } ) \| \le \epsilon\tag{11}
$$

if there exists a training data point $( x ^ { \ast } , \kappa ( x ^ { \ast } ) ) \in \mathcal { D }$ that satisfies $\| x _ { k } - x ^ { * } \| \leq \bar { ( L _ { \mathrm { M P C } } ( x _ { k } ) + L _ { \mathrm { N N } } ) } ^ { - 1 } ( \epsilon - \epsilon _ { \mathcal { D } } )$ and ${ \boldsymbol { x } } ^ { * } \in \mathcal { N } ( { \boldsymbol { x } } _ { k } )$

PROOF. We begin by writing the left-hand side of (11) as

$$
\begin{array} { r l } & { \| \kappa ( x _ { k } ) - \kappa _ { \mathrm { N N } } ( x _ { k } ) \| = \| \kappa ( x _ { k } ) - \kappa ( x ^ { * } ) + \kappa _ { \mathrm { N N } } ( x ^ { * } ) } \\ & { \qquad - \kappa _ { \mathrm { N N } } ( x _ { k } ) + \kappa ( x ^ { * } ) - \kappa _ { \mathrm { N N } } ( x ^ { * } ) \| . } \end{array}
$$

Applying the triangle inequality, we obtain the upper bound

$$
\begin{array} { r l } & { \| \kappa ( x _ { k } ) - \kappa _ { \mathrm { N N } } ( x _ { k } ) \| \leq \| \kappa ( x _ { k } ) - \kappa ( x ^ { * } ) \| } \\ & { \qquad + \| \kappa _ { \mathrm { N N } } ( x ^ { * } ) - \kappa _ { \mathrm { N N } } ( x _ { k } ) \| } \\ & { \qquad + \| \kappa ( x ^ { * } ) - \kappa _ { \mathrm { N N } } ( x ^ { * } ) \| . } \end{array}
$$

By Assumption 7, and as ${ \boldsymbol { x } } ^ { * } \in \mathcal { N } ( { \boldsymbol { x } } _ { k } )$ , we obtain further that

$$
\begin{array} { r l } & { \| \kappa _ { \mathrm { N N } } ( x ^ { * } ) - \kappa _ { \mathrm { N N } } ( x _ { k } ) \| \leq L _ { \mathrm { N N } } \| x ^ { * } - x _ { k } \| , } \\ & { \qquad \| \kappa ( x _ { k } ) - \kappa ( x ^ { * } ) \| \leq L _ { \mathrm { M P C } } ( x _ { k } ) \| x _ { k } - x ^ { * } \| . } \end{array}
$$

Now, it follows straightforwardly that

$$
\begin{array} { r } { \| \kappa ( x _ { k } ) - \kappa _ { \mathrm { N N } } ( x _ { k } ) \| \leq ( L _ { \mathrm { M P C } } ( x _ { k } ) + L _ { \mathrm { N N } } ) \| x _ { k } - x ^ { * } \| } \\ { + \epsilon _ { \mathcal { D } } , ~ } \end{array}
$$

which, together with the requirement that $\| x _ { k } - x ^ { * } \| \ \leq$ $( L _ { \mathrm { M P C } } ( x _ { k } ) + L _ { \mathrm { N N } } ) ^ { - 1 } ( \epsilon - \dot { \epsilon _ { \mathcal { D } } } )$ , implies the result in (11).

We now turn to a detailed discussion of the conditions in Proposition 11.

Remark 14 Assuming that LICQ, SOSC, and SCS hold can be restrictive. However, these assumptions are often employed in explicit MPC and parametric optimization to ensure that the optimal solution is well-behaved, see, e.g., [5] or [34]. LICQ is typically ensured by design of the con straints (6c), (6d), and (6e), or via regularization or constraint relaxation techniques, see, e.g., [28,31]. SCS is usually only lost when arbitrarily small perturbations of the state $x _ { k }$ lead to active set changes. However, the sets on which this is the case are typically lower-dimensional subsets of X, and hence, Lebesgue null sets. As a result, it is unlikely in practice that a state sample lies in these sets. Satisfaction ofthe SOSC is more challenging; however, our approach allows for violations of the SOSC on Lebesgue-null sets, analogous to the SCS condition. For a more comprehensive overview of the regularity properties of parametric optimization problems, see, e.g., [9].

To better understand the implications of Theorem 13, we proceed by discussing some of its practical aspects.

Remark 15 If an upper bound $L _ { \mathrm { M P C } } ^ { \mathrm { m a x } }$ on the local Lipschitz constants $L _ { \mathrm { M P C } } ( x _ { k } )$ can be estimated over all regions of the state space where the MPC feedback law is locally Lipschitz continuous, Theorem 13 generalizes accordingly. Specifically, a state-independent distance bound $\delta : = ( L _ { \mathrm { M P C } } ^ { \mathrm { m a x } } + L _ { \mathrm { N N } } ) ^ { - 1 } ( \epsilon - \epsilon _ { \mathcal { D } } )$ can then be used to establish the result, at the cost of potentially increased conservatism. Hence, Theorem 13 provides a condition on the minimum coverage of X that must be achieved by the training inputs. To enforce the distance condition in Theorem 13, one can choose a parameter $0 < \eta \le \delta$ and construct the training dataset to provide an η-coverage of X. However, η-coverage alone does not ensure that the corresponding training point lies in the same neighborhood of local Lipschitz continuity as $x _ { k } .$ . These regions are determined by the underlying MPCproblem and may have arbitrary geometry; in particular, they need not be norm balls or have a uniform radius. For each state $x _ { k } ,$ the relevant neighborhood is the one associated with the corresponding region of the MPC problem. The result therefore remains conditional on the existence,for each considered state x , ofa training point that satisfies both the distance and neighborhood conditions of Theorem 13. The parameter η thus acts as a tuning parameter, which must be selected heuristically or via appropriate tuning strategies tailored to the specific application.

## 3.3 Recursive feasibility and Stability Guarantees

Building on the results of Theorem 13, we can now guarantee safety, i.e., constraint satisfaction, and stability of the system (1) under the neural network controller. Specifically, Theorem 13 provides conditions under which a neural network satisfies a prescribed bound on the approximation error, which can be used to design the underlying MPC to be robust against such errors, as described in Section 2. This robustness enables safe substitution of the MPC with the neural network controller. In particular, if the conditions of Theorem 13 are satisfied (point-wise) along the closedloop state trajectory of system (1) under the neural network controller, the safety and stability guarantees established in Theorem 5 are preserved. Moreover, under these conditions, the neural network controller maintains the invariance of the feasible set of initial conditions for the closed-loop system, as established in Corollary 6.

The following theorem provides a formal statement of the result under practical and realistically attainable conditions, explicitly accounting for stochastic disturbances that arise in closed-loop operation, which results in a probabilistic view on the results in Proposition 11 and Theorem 13.

## Theorem 16 Suppose the following conditions hold:

(i) Assumptions 1, 2, 3, 4, 7, and 10 hold.

(ii) The system (1) is subject to additive, bounded, independent, and identically distributed noise $\nu _ { k } \in \mathcal { W } _ { \nu } \subset$ $\mathbf { \mathbb { R } } ^ { n _ { x } } ,$ where

(a) $\mathcal { W } _ { \nu }$ is compact and satisfies $0 \in \operatorname { i n t } ( \mathcal { W } _ { \nu } )$

(b) $\nu _ { k }$ follows a continuous probability measure $\mu _ { \nu }$ that is absolutely continuous with respect to the Lebesgue measure λ, i.e., $\mu _ { \nu } \ll \lambda .$

(iii) The MPC problem (6) is designed to be robust with respect to the additive, bounded, total disturbance $\bar { w } _ { k } : =$ $w _ { k } + \nu _ { k } ,$ where $w _ { k }$ is defined as in (5).

(iv) The $s e t \mathcal { Z } : = \{ x \in \mathcal { X } _ { N } \mid L I C Q , S C S ,$ or SOSC is violated at $( \mathbf { u } ^ { * } , x ) \}$ , where $\mathbf { u } ^ { * }$ is the optimal solution of (6) with initial condition x, satisfies $\lambda ( \mathcal { Z } ) = 0 , i . e . ,$ the set of states where regularity fails is a null set with respect to the Lebesgue measure.

(v) The initial state satisfies $x _ { 0 } \in \mathcal { X } _ { N } \backslash \mathcal { Z } , i . e .$ ., the MPC problem is initially feasible, and the feedback law is regular at $x _ { 0 } .$

(vi) For each $x \in \mathcal { X } _ { N } \backslash \mathcal { Z } _ { : }$ , there exists a training data point $( x ^ { \ast } , \kappa ( x ^ { \ast } ) ) \in \mathcal { D }$ that satisfies $\| x - x ^ { * } \| \overset { \bar { } } { \leq } ( L _ { \mathrm { M P C } } ^ { \mathrm { m a x } } +$ $L _ { \mathrm { N N } } ) ^ { - 1 } ( \epsilon - \epsilon _ { \mathcal { D } } )$ and $x ^ { * } \in \mathcal { N } ( \stackrel { \cdot } { x } )$ .

Then, the neural network controller

(1) remains feasible along the closed-loop trajectory almost surely, and

(2) renders the closed-loop system almost surely input-tostate stable with respect to the origin and the disturbance $\bar { w } _ { k }$

PROOF. The proof proceeds in three main steps. First, we revisit the robust MPC design under the presence of an additional disturbance, as specified in condition (iia), and demonstrate the feasibility of the robust MPC formulation required by condition (iii). Second, we show that the feedback law of this robust MPC is almost surely sufficiently regular at the sampled closed-loop states $x _ { k }$ where it is evaluated, leveraging conditions (iia) and (iv). This guarantees the local Lipschitz continuity of the feedback law in neighborhoods around the observed closed-loop states. Third, leveraging conditions (i) and (v), we establish that the approximation error at any observed closed-loop state satisfies the bound stated in Theorem 13. Together, these results imply that the guarantees provided by the robust MPC design in Theorem 5 transfer to the neural network controller.

Step 1: We begin by considering the system (1) subject to the additive disturbance $\nu _ { k } .$ , given by $x _ { k + 1 } = f ( x _ { k } , u _ { k } ) + \nu _ { k }$

Analogously to the formulation in (5), we define a single additive disturbance term $\bar { w } _ { k }$ that captures both the effect of the approximation error induced by the neural network controller and the state disturbance $\nu _ { k }$ . Specifically,

$$
\begin{array} { r l } & { x _ { k + 1 } = f ( x _ { k } , u _ { k } + e _ { k } ) + \nu _ { k } \equiv f ( x _ { k } , u _ { k } ) + \bar { w } _ { k } , \mathrm { ~ w i t h } } \\ & { \quad \bar { w } _ { k } = f ( x _ { k } , u _ { k } + e _ { k } ) - f ( x _ { k } , u _ { k } ) + \nu _ { k } . } \end{array}
$$

This yields the bound $\| \bar { w } _ { k } \| \leq \| f ( x _ { k } , u _ { k } + e _ { k } ) - f ( x _ { k } , u _ { k } ) \| +$ $\lVert \nu _ { k } \rVert \leq L _ { f , u } \epsilon + \epsilon _ { \nu } ,$ , where $\epsilon _ { \nu } > 0$ is a constant such that $\mathcal { W } _ { \nu } \subseteq \{ \nu \ \overline { { | } } \ \| \nu \| \le \epsilon _ { \nu } \}$ . Such a constant $\epsilon _ { \nu } > 0$ exists due to the compactness of $\mathcal { W } _ { \nu }$ . Including $\epsilon _ { \nu }$ in (7) yields

$$
\mathcal { W } _ { n } = \left\{ e \in \mathbb { R } ^ { n _ { x } } \ | \ \| e \| \leq \frac { L _ { f , x } ^ { n } - 1 } { L _ { f , x } - 1 } \left( L _ { f , u } \epsilon + \epsilon _ { \nu } \right) \right\} .
$$

This establishes the feasibility of the robust MPC design required in condition (iii), which is henceforth assumed.

Step 2: Given condition (iia), for a fixed state $x _ { k } \in \mathcal { X } _ { N } \backslash \mathcal { Z } _ { * }$ the successor state $x _ { k + 1 }$ is a random variable determined by the disturbance $\nu _ { k }$ and follows the probability measure $\mu _ { x }$ . Specifically, $\mu _ { x }$ is the pushforward of $\mu _ { \nu }$ under the translation map $T ( z ) ~ = ~ z ^ { ' } + f ( x _ { k } , u _ { k } + e _ { k } )$ . Since the Lebesgue measure λ is translation invariant and $\mu _ { \nu } \ll \lambda$ by assumption, it follows that $\mu _ { x } \ll \lambda , \mathrm { i . e . , } \mu _ { x }$ is absolutely continuous with respect to the Lebesgue measure. Together with condition (iv), this yields $\mu _ { x } ( \bar { \mathcal { Z } } ) = 0 .$ , or equivalently, $\operatorname* { P r } [ x _ { k + 1 } \in \mathcal { X } _ { N } \setminus \mathcal { Z } ] = \bar { 1 }$ almost surely. By induction, and using condition (v), it follows that $\dot { x _ { k } } \in \mathcal { X } _ { N } \setminus \mathcal { Z }$ for all k, almost surely. Consequently, the regularity conditions of Proposition 11 are almost surely satisfied at all visited closed-loop states, implying local Lipschitz continuity of the MPC feedback law in neighborhoods of these states.

Step 3: The established (local) Lipschitz continuity of the MPC feedback law in neighborhoods of all visited closedloop states, together with conditions (i) and (vi), directly implies that the error at these states almost surely satisfies the bound stated in Theorem 13. Since, by condition (iii), the underlying MPC is robust with respect to both the approximation error induced by the neural network controller and the additive disturbance $\nu _ { k } .$ , the closed-loop guarantees established in Theorem 5 almost surely carry over to the neural network controller.

We conclude the theoretical analysis by briefly discussing the conditions of Theorem 16. Condition (i) requires Assumptions 1, 2, 3, 4, 7, and 10, whose implications and restrictiveness were discussed upon introduction.

Condition (ii) assumes bounded additive disturbances with a continuous distribution of compact support, encompassing, e.g., model-plant mismatch, external disturbances, and measurement noise. Condition (iia) is standard in robust and stochastic MPC, while (iib) covers, among others, uniform and truncated Gaussian distributions; the distribution itself need not be known explicitly or analytically tractable.

Condition (iii) requires robustness of the MPC against the combined disturbance, achievable through standard constraint tightening as described in Section 2.3.

Condition (iv) requires the irregularity set Z to have Lebesgue measure zero, supported by Remark 14 and empirical evidence reported in [34].

Finally, condition (v) requires initial MPC feasibility and excludes states where the feedback law lacks regularity.

## 3.4 Procedure for Certified NN Approximation

We now present a procedure to generate a NN controller that provides the guarantees discussed in Section 3.3 and discuss the limitations of implementing this method in a practical setting.

A possible implementation is summarized in Algorithm 1. The resulting NN is fixed at run time and does not require any online adaptation or re-training. In particular, once provisional bounds on the approximation error and the Lipschitz constants of the MPC law and the NN have been selected, sampling and training may be carried out in a single offline procedure, provided that the resulting network satisfies the prescribed bounds. Thus, enlarging the training dataset does not affect online execution times, as the evaluation time at deployment depends only on the NN architecture.

We first specify a bound $\epsilon _ { \mathcal { D } }$ on the expected error at the training points of the NN with respect to the underlying MPC law $\kappa ( x )$ . This bound may be chosen based on domain knowledge of the problem, an engineering guess, or a single preliminary NN training on an arbitrary initial dataset $\mathcal { D } _ { 0 }$ from a similar domain. Given such a bound, we select the design parameter ϵ in Theorem 13 so that $\epsilon \geq \epsilon _ { \mathcal { D } }$ , which can be chosen larger to include additional robustness margins.

Once ϵ has been fixed, we design the robust MPC as in Section 2.3. At this stage, state disturbances $\epsilon _ { \nu }$ can also be incorporated. This yields a robust MPC feedback law $\kappa ( x )$

Given the MPC design, we next provide bounds on the local Lipschitz constants of both the NN and the MPC on $\mathcal { X } .$ An upper bound $L _ { \mathrm { M P C } }$ for the feedback law (8) may be obtained via power iteration, finite-difference schemes, or specialized methods [26, 32]. Similarly, we fix $L _ { \mathrm { N N } }$ for the NN architecture and estimate the robustly forward-invariant set $\mathcal { X } _ { N }$ , on which the MPC law is defined and from which the training data are drawn.

With the bounds $\epsilon _ { D } , L _ { \mathrm { N N } }$ , and $L _ { \mathrm { M P C } }$ specified and Theorem 13, we compute the coverage radius δ. We then select a radius $\eta \leq \delta$ for the over-coverage. Having fixed $\eta ,$ we employ sub-sampling or space-filling methods [18] to compute an η-coverage of $\mathcal { X } _ { N }$ , which is also a δ-coverage of $\mathcal { X } _ { N }$ . Evaluating the MPC feedback law (8) on these points in open-loop yields the final training dataset D.

We then initialize the neural network with the chosen architecture and train it once on the fixed dataset D. The trained network must satisfy $\begin{array} { r } { \hat { \epsilon } _ { \mathcal { D } } ~ : = ~ \operatorname* { m a x } _ { ( x _ { i } , \kappa ( x _ { i } ) ) \in \mathcal { D } } \| \kappa ( x _ { i } ) ~ - } \end{array}$ $\kappa _ { \mathrm { N N } } ( x _ { i } ) \| \leq \epsilon _ { \mathcal { D } }$ and $\hat { L } _ { \mathrm { N N } } \leq L _ { \mathrm { N N } }$ . Because these quantities are known only after training, hyperparameter tuning or retraining may be required. Any network satisfying both inequalities is admissible, and no further refinement is needed.

```latex
Algorithm 1 Generation of NN Controller
Require: Constraint set X.
1: Specify or estimate a nominal approximation error
bound $\epsilon _ { \mathcal { D } }$
2: Choose a design approximation error bound $\epsilon \geq \epsilon _ { D } .$
3: Design a robust MPC using ϵ and, optionally, a distur
bance bound $\epsilon _ { \nu } ,$ obtaining the feedback law $\kappa ( x )$
4: Specify Lipschitz bounds $L _ { \mathrm { M P C } }$ and $L _ { \mathrm { N N } }$ on $\dot { \mathcal { X } } .$
5: Estimate the robustly forward invariant set $\mathcal { X } _ { N }$ of the
robust MPC.
6: Compute the required coverage radius $\begin{array} { r } { \delta = \frac { \epsilon - \epsilon _ { \mathcal { D } } } { L _ { \mathrm { M P C } } + L _ { \mathrm { N N } } } } \end{array}$
according to Theorem 13.
7: Select a radius $\eta \leq \delta$ for over-coverage.
8: Compute an η-coverage of $\mathcal { X } _ { N }$ and form the training
dataset D by evaluating $\kappa ( x )$ on the coverage points.
9: Train the neural network κ on ${ \mathcal { D } } ,$ such that it satisfies
$\hat { \epsilon } _ { \mathcal { D } } \leq \epsilon _ { \mathcal { D } } \leq \epsilon$ and $\hat { L } _ { \mathrm { N N } } \leq L _ { \mathrm { N N } } .$
10: Output the final NN controller $\kappa _ { \mathrm { N N } } ( x )$ , which approx
imates $\kappa ( x )$ on $\mathcal { X } _ { N }$ with error bound ϵ.
```

Remark 17 In practice, coverage-based certification suffers from the curse of dimensionality: for a bounded $n _ { x } –$ dimensional domain with characteristic width R, the required number of samples scales as $\begin{array} { r } { n _ { \mathcal { D } } ~ = ~ \mathcal { O } ( ( R / \delta ) ^ { n _ { x } } ) } \end{array}$ Thus, maintaining afixed coverage radius causes an increasing offline sampling and training effort with $n _ { x } ,$ whereas online evaluation remains determined by the NN architecture. Scalability also depends on δ, and hence on the acceptable approximation-error margin and the Lipschitz bounds.

## 4 Example

We evaluate the approach on an isothermal continuously stirred tank reactor in which the reaction $2 A + B  { \dot { C } } ,$ $C  2 D$ , and the selective removal of D are realized. The

continuous-time dynamics are

$$
\begin{array} { r l } & { \dot { c } _ { A } = d \left( c _ { A , f } - c _ { A } \right) - 2 k _ { 1 } c _ { A } ^ { 2 } c _ { B } , } \\ & { \dot { c } _ { B } = d \left( c _ { B , f } - c _ { B } \right) - k _ { 1 } c _ { A } ^ { 2 } c _ { B } , } \\ & { \dot { c } _ { C } = - d c _ { C } + k _ { 1 } c _ { A } ^ { 2 } c _ { B } - k _ { 2 } c _ { C } , } \\ & { \dot { c } _ { D } = - d c _ { D } + 2 k _ { 2 } c _ { C } - k _ { \mathrm { s e p } } c _ { D } . } \end{array}
$$

Here, $\mathit { x } ~ = ~ \left[ c _ { A } , c _ { B } , c _ { C } , c _ { D } \right]$ denotes the species concentrations in mol $\mathrm { L } ^ { - 1 }$ , and $\begin{array} { r } { u } { { } = { } \ c _ { A , f } } \end{array}$ is the controlled feed concentration of species A in mol $\mathrm { L } ^ { - 1 }$ . The dilution rate is $d = \mathrm { 0 . 1 } \mathrm { m i n ^ { - 1 } } ,$ and the fixed feed concentration is $c _ { B , f } ~ = ~ 0 . 7 5 \mathrm { m o l } \mathrm { L } ^ { - 1 }$ . The reaction parameters are $k _ { 1 } = \mathrm { { 3 . 2 L ^ { 2 } } / m i n ^ { 2 } } \ k _ { 2 } = 0 . 1 \mathrm { { m i n ^ { - 1 } } }$ and $k _ { \mathrm { s e p } } = 0 . 1 \mathrm { m i n } ^ { - 1 }$ The model is discretized using fourth-order Runge–Kutta with $T _ { s } = 0 . 5$ min.

Over ${ \mathcal { X } } \times { \mathcal { U } } .$ , the numerically estimated Lipschitz constants of the undisturbed discrete dynamics w.r.t. the 2-norm are $L _ { f , x } \ = \ 1 . 1 1$ and $L _ { f , u } ~ = ~ 0 . 0 5$ . We add $\tilde { \nu } _ { k } \ = \ [ \tilde { \nu } _ { 1 , k } , \tilde { \nu } _ { 2 , k } , \nu _ { 3 , k } , \nu _ { 4 , k } ] ^ { \top }$ , as an additive disturbance satisfying $\| \tilde { \nu } _ { k } \| _ { 2 } \le \epsilon _ { \nu }$ with $\epsilon _ { \nu } = 0 . 0 0 1$ to the discretized system.

## 4.1 Controller Design

The robust MPC uses the constraint sets $\mathcal { X } \ : = \ \{ x \}$ 0.05 mol $\mathrm { L } ^ { - 1 } \leq c _ { \gamma } \leq 0 . 4 5 \mathrm { m o l } \mathrm { L } ^ { - 1 } , \ \gamma \in \{ A , B , C , \dot { D } \} \}$ and $\mathcal { U } ~ : = ~ \{ u ~ \mid ~ 1 . 0 0$ mol $\mathrm { L } ^ { - 1 } ~ \le ~ u ~ \le ~ \bar { 1 . 5 0 } \mathrm { m o l } \mathrm { L } ^ { - 1 } \bar { \updownarrow }$ together with the quadratic stage and terminal costs $l ( \bar { x } , u ) = ( x - x _ { \mathrm { r e f } } ) ^ { \top } \bar { Q } ( \underline { { x } } - x _ { \mathrm { r e f } } ) \bar { + } ( u - u _ { \mathrm { r e f } } ) ^ { \top } R ( u - u _ { \mathrm { r e f } } )$ and $V _ { T } ( x ) = ( x - x _ { \mathrm { r e f } } ) ^ { \top } P ( x - x _ { \mathrm { r e f } } ) , \mathrm { r e s p e c t i v e l y } .$ , where $Q = \mathrm { d i a g } ( 2 , 1 , 1 , 1 ) , R = 1$ , and

$$
P = { \left[ \begin{array} { l l l l } { 8 . 7 7 } & { - 5 . 2 9 } & { 3 . 2 7 } & { 1 . 0 6 } \\ { - 5 . 2 9 } & { 1 4 . 1 9 } & { 2 . 2 3 } & { 0 . 6 9 } \\ { 3 . 2 7 } & { 2 . 2 3 } & { 1 1 . 8 4 } & { 3 . 6 9 } \\ { 1 . 0 6 } & { 0 . 6 9 } & { 3 . 6 9 } & { 8 . 2 6 } \end{array} \right] } ~ .
$$

The matrix P is obtained from fixed-point iterations of the discrete-time algebraic Riccati equation for the model linearized at the reference. The terminal level is set to $\alpha _ { v } ~ = ~ 0 . 1 0$ . We use the constant reference $\begin{array} { r l r } { x _ { \mathrm { r e f } } } & { { } = } & { [ 0 } \end{array}$ .25, 0.25, 0.25, 0.25]<sup>⊤</sup> mol $\mathrm { L } ^ { - 1 }$ and $\begin{array} { r l } { u _ { \mathrm { r e f } } } & { { } = } \end{array}$ $1 . 2 5 \mathrm { m o l } \dot { \mathrm { L } } ^ { - 1 }$ . We choose a prediction horizon of $N = 1 0$ steps, corresponding to a prediction horizon of 5.00 min. We conservatively set the designable approximation-error bound to $\ d t _ { \mathnormal { 1 } } \epsilon = 0 . 0 5$ , corresponding to 10 % of the input range.

## 4.2 Neural Network Architecture

We approximate the underlying robust MPC law by a fully connected feed-forward neural network (cf. Section 3.1) with 3 hidden layers, each containing 32 neurons, and Gaussian error linear unit activation functions. The network takes the 4-dimensional concentration state as input and outputs the scalar feed concentration $c _ { A , f } .$

## 4.3 Data Generation and Training

We seed $1 9 4 4 8 1 = 2 1 ^ { 4 }$ points equally spaced within the state constraint set. Using these points, we estimate a grid approximation $\hat { \mathcal { X } } _ { N } \subseteq \mathcal { X } _ { N }$ of the robustly forward-invariant set of the underlying MPC, which leaves 60 626 strictly feasible and converged grid points.

Before dataset generation, we estimate the Lipschitz constant of the MPC policy at 10 000 points in $\hat { \mathcal { X } } _ { N }$ using central finite differences with a step size of $1 \times 1 0 ^ { - 4 }$ . This yields $L _ { \mathrm { M P C } } = 3 . 1 5 3 1$ . For the NN architecture described in In Section 4.2, we prescribe the analytical design bound $L _ { \mathrm { N N } } ~ = ~ 3 . 5$ , which is subsequently enforced via weight projection during training [10]. The projection approach is further explained in supplementary Section S2 and in [10]. We choose the dataset error $\epsilon _ { \mathscr D } ~ = ~ 0 . 0 1$ and the overall approximation-error bound $\epsilon = 0 . 0 5$ . Therefore, we can calculate $\delta = 0 . 0 0 6 0 1 2 2$ , and we choose $\eta = \delta .$ . For the bounds and all subsequent analysis, the 2-norm was used.

We construct a candidate grid over the bounding box of $\hat { \mathcal { X } } _ { N }$ and retain points that pass a membership check with the invariant set. The final η-cover contains 6 417 883 centers. The MPC policy is evaluated at every center to obtain the final training dataset D. We then train a newly initialized NN on D for 50 epochs using a learning rate of $5 \times 1 0 ^ { - 3 }$ and a batch size of 128. The resulting NN achieves an $R ^ { 2 } .$ -score of 0.9999 on a validation set containing 10 000 points that are distinct from all training points. The maximum approximation error over the complete training dataset is $\hat { \epsilon } _ { \mathcal { D } } = 1 . 3 7 5 9 \times 1 0 ^ { - 3 } < 0 . 0 1$ . For the trained NNs Lipschitz bound obtained from the layer operator norms and activation Lipschitz constants is $\hat { L } _ { \mathrm { N N } } = 3 . 4 9 9 9$ . Thus, both quantities remain below the used design values.

All training, sampling, and simulations were performed in Python on a Lenovo ThinkPad T14 Gen 2a running Windows 11, equipped with an AMD Ryzen 7 PRO 5850U CPU and 32 GB RAM. MPC was implemented using IPOPT via CasADi [3], and the neural network using PyTorch [4, 25].

An additional 2-dimensional result is shown in supplementary Section S1

## 4.4 Results

We evaluate the trained NN on the simulated isothermal reactor in closed-loop using the robust MPC used for data generation as a comparison. The closed-loops for both controllers were simulated starting from 100 independently selected initial states for 160 steps with a sampling time of 0.5 min. To assess the controller near the boundary of the estimated invariant set, points near the eight coordinate-wise extrema of

Table 1

Controller computation times over 16,000 evaluations.
<table><tr><td></td><td colspan="3">Mean (ms) Max. (ms) Min. (ms)</td></tr><tr><td>Robust MPC</td><td>20.73</td><td>85.79</td><td>16.40</td></tr><tr><td>NN controller</td><td>0.59</td><td>1.85</td><td>0.48</td></tr></table>

Pointwise NN policy error over 16,000 evaluations.
<table><tr><td></td><td>Mean</td><td>Max.</td><td>Min.</td></tr><tr><td>Approximation error</td><td> $4 . 7 6 \times 1 0 ^ { - 5 }$ </td><td> $8 . 6 2 \times 1 0 ^ { - 4 }$ </td><td> $5 . 7 3 \times 1 0 ^ { - 9 }$ </td></tr></table>

the retained grid are included explicitly. The remaining initial states are sampled uniformly without replacement from $\hat { \mathcal { X } } _ { N }$ . None of the selected initial states were previously used for training or validation. For each initial state, the robust MPC and NN controller experience the same additional disturbance realization.

Fig. 1 shows the resulting trajectories in the $( c _ { A } , c _ { B } ) .$ - and (c<sub>C</sub>, c<sub>D</sub>)-projections of $\hat { \mathcal { X } } _ { N }$ . The enlarged regions show trajectories initialized close to different boundaries of the estimated admissible set. The trajectories generated by the robust MPC and its NN approximation remain close over the complete simulation horizon and approach the reference. No state-constraint or input-constraint violations are observed for either controller. In particular, all 16 000 evaluated states remain inside $\hat { \mathcal { X } } _ { N }$ . Set membership is evaluated in the full 4-dimensional state space and not solely from the 2-dimensional projections shown in Fig. 1.

Table 1 reports the computation times obtained from 16 000 sequential single-state controller evaluations. Both controllers are evaluated on the same hardware using a single CPU thread, and no batching is employed for the NN controller. The mean computation time decreases from 20.73 ms for the robust MPC to 0.59 ms for the NN controller, corresponding to a mean speed-up of approximately 35 times.

The pointwise approximation errors along the same independently generated trajectories are summarized in Table 2. The maximum observed error is $8 . 6 2 \times 1 0 ^ { - 4 }$ , which is below the prescribed approximation bound. No violation of the approximation bound occurs among the 16000 evaluations. The minimum error is $5 . 7 3 \times 1 0 ^ { - 9 }$ and is therefore close to the numerical precision relevant to this evaluation.

These independent closed-loop results are consistent with the theoretical guarantees established through the underlying robust MPC, the coverage, and the certified approximation bound. In particular, the absence of approximation error bound, state, and input constraint violations supports the conditions of Theorem 13 along all evaluated trajectories and demonstrates the applicability of Algorithm 1.

![](images/a28685402d22b946d13ad9baaab7c0f675c78363b3764727df2745fe1cba7489.jpg)  
Fig. 1. Projections of the estimated invariant set $\hat { \mathcal { X } } _ { N }$ and 100 closed-loop trajectories under the robust MPC and its NN approximation. Top: (c<sub>A</sub>, c<sub>B</sub>)-plane; bottom: (c<sub>C</sub>, c<sub>D</sub>)-plane. The insets show trajectories initialized near selected boundaries.

## 5 Conclusion

We developed a constructive framework for certified neural approximation of robust nonlinear model predictive controllers. By interpreting the approximation error as a bounded input disturbance and embedding Lipschitz-based error bounds into a robust MPC design, we established conditions under which recursive feasibility and input-to-state stability of the original controller are preserved under neural replacement. In contrast to post-verification approaches, the proposed method provides an offline design procedure that links dataset coverage and Lipschitz constants to explicit uniform approximation guarantees.

The framework combines results from parametric optimization, robust MPC theory, and Lipschitz continuity of neural networks to derive verifiable error bounds. The simulation study on a nonlinear isothermal CSRT model confirms that the neural controller reproduces the closed-loop behavior of the robust MPC within the certified invariant region while significantly reducing online computation time.

Future work will investigate domain partitioning to reduce conservatism induced by global Lipschitz bounds and data requirements associated with uniform coverage, as well as adaptive or trajectory-based sampling and mixtureof-experts architectures to improve scalability to higherdimensional systems. We will also consider relaxing the regularity assumptions on the MPC feedback law to broaden applicability of the proposed framework.

Declaration of Generative AI and AI-assisted technolo gies in the writing process

During the preparation of this work, the authors used Chat-GPT 5.5 and 5.6-Sol from OpenAI in order to improve language flow and wording. After using this tool/service, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

## Supplementary Material

Certified Neural Approximation of Robust Nonlinear Model Predictive Control

Hendrik Alsmeier, Maik Pfefferkorn, Rolf Findeisen

Summary. This document supplements the article “Certified Neural Approximation of Robust Nonlinear Model Predictive Control.” Section S1 presents an additional twodimensional numerical example. Section S2 describes the procedure used to enforce the prescribed neural-network Lipschitz bound during training. Section S3 provides the complete proof of Lemma 9 in the main article.

## S1 2-Dimensional Example

In addition to the continuous stirred tank reactor example presented in the main part of the paper, we further evaluate the approach on a nonlinear drone model describing vertical motion. The state $x = [ x _ { 1 } , x _ { 2 } ] = [ z , v _ { z } ]$ consists of altitude and vertical velocity, and the dynamics are given by

$$
\begin{array} { l } { { x _ { 1 , k + 1 } = x _ { 1 , k } + T _ { s } x _ { 2 , k } + \tilde { \nu } _ { 1 , k } , } } \\ { { x _ { 2 , k + 1 } = x _ { 2 , k } + T _ { s } \left( - g - k _ { z } x _ { 2 , k } ^ { 3 } + \displaystyle \frac { 1 } { m } u _ { k } \right) + \tilde { \nu } _ { 2 , k } . } } \end{array}
$$

Here, u is the input thrust force, $k _ { z } = 0 . 0 0 0 1 \mathrm { s m } ^ { - 2 }$ is the cubic drag coefficient in z-direction, $m = 1 \mathrm { k g }$ is the mass of the drone, $g = 9 . 8 1 \mathrm { m } \mathrm { s } ^ { - 2 }$ is the acceleration due to gravity, and $T _ { s } = 0 . 0 5$ is the discretization time. Additionally, $\tilde { \nu } _ { k } = [ \tilde { \nu } _ { 1 , k } , \tilde { \nu } _ { 2 , k } ] ^ { \intercal }$ denotes an additive disturbance satisfying $\| \tilde { \nu } _ { k } \| _ { 2 } \le \epsilon _ { \nu }$ with $\epsilon _ { \nu } = 0 . 0 0 1$ 1. The undisturbed system therefore has Lipschitz constants $L _ { f , x } = 1 . 0 5$ and $\dot { L } _ { f , u } = 0 .$ .05 w.r.t. the 2-norm, which can be calculated directly due to its control-affine structure.

## S1.1 Controller Design

For the robust MPC, we define the state constraint set $\mathcal { X }$ by $- 2 0 \mathrm { m } \leq x _ { 1 } \leq 4 0$ m and $- 2 5 \mathrm { { m s ^ { - 1 } } } \leq x _ { 2 } \leq 2 5 \mathrm { { m s ^ { - 1 } } }$ with positive velocity pointing upward, and the input constraint set U by $- 3 0 \mathrm { { N } } \leq u \leq 3 0 \mathrm { { N } }$

We use a quadratic stage cost $l ( x , u ) = ( x - x _ { \mathrm { r e f } } ) ^ { \top } Q ( x -$ $x _ { \mathrm { r e f } } ) + ( \bar { u ^ { } } - u _ { \mathrm { r e f } } ) ^ { \top } \bar { R ( } u - u _ { \mathrm { r e f } } ^ { } )$ , where $Q \ = \ \mathrm { d i a g } ( \dot { 5 } , 1 )$ and $R = 0 . 1$ . For the terminal cost, we choose $V _ { T } ( x ) =$ $( x - x _ { \mathrm { r e f } } ) ^ { \top } P ( x - x _ { \mathrm { r e f } } )$ with

$$
P = { \left[ \begin{array} { l } { 7 4 . 6 4 0 4 \ 1 5 . 9 8 9 9 } \\ { 1 5 . 9 8 9 9 \ 1 1 . 1 3 5 5 } \end{array} \right] }
$$

obtained via fixed-point iterations of the discrete-time algebraic Riccati equation. The terminal set is defined as the $\alpha _ { v } –$ level set of $V _ { T }$ with $\alpha _ { v } = 1 0 0$ . We set a constant reference of $x _ { \mathrm { r e f } } = [ 2 5 . 0 \mathrm { m } , 0 \mathrm { m } \mathrm { s } ^ { - 1 } ] ^ { \intercal }$ and $u _ { \mathrm { r e f } } = g \cdot m = 9 . 8 1 \mathrm { N }$

The designable approximation error bound is set to $\epsilon = 3$ which is small relative to the admissible range of u.

## S1.2 Horizon length and conservatism

For the robust MPC formulation in (6), the disturbance sets $W _ { n }$ may grow with the prediction step. Their radius $r _ { n }$ is

$$
r _ { n } = \frac { L _ { f , x } ^ { n } - 1 } { L _ { f , x } - 1 } L _ { f , u } ( \epsilon + \epsilon _ { \nu } ) .
$$

Hence, $W _ { n }$ remains bounded for $L _ { f , x } < 1$ , grows linearly for $L _ { f , x } = 1$ , and may grow geometrically for $L _ { f , x } > 1$

For the drone model, $L _ { f , x } ~ = ~ 1 . 0 5 ~ > ~ 1$ , and the disturbance sets therefore grow along the prediction horizon. With $L _ { f , u } = 0 . 0 5 , \epsilon = 3$ , and a chosen $N = 3 0$ , the radius of $\dot { W } _ { N - 1 } = W _ { 2 9 }$ is approximately

$$
r _ { 2 9 } = \frac { 1 . 0 5 ^ { 2 9 } - 1 } { 1 . 0 5 - 1 } ( 0 . 0 5 \cdot 3 + 0 . 0 0 1 ) \approx 9 . 4 1 .
$$

For comparison, the state domain has widths 60 and 50 in the altitude and velocity directions, respectively. Under the 2-norm used in the example, the tightening by $W _ { 2 9 }$ reduces these widths by $2 r _ { 2 9 } \approx 1 \dot { 8 } . 7 .$ , corresponding to reductions of approximately 31 % and $3 7 \%$ , respectively. This illustrates that the horizon-dependent tightening is already substantial for $N = 3 0$ . We therefore selected this horizon as a practical trade-off between improved look-ahead and increased conservatism.

A tube-based MPC formulation with an ancillary feedback controller could modify this propagation by stabilizing the error dynamics. Under suitable conditions ensuring contraction of the resulting closed-loop error dynamics, the corresponding uncertainty sets could remain bounded or grow substantially more slowly. However, the use of an ancillary feedback controller alone does not guarantee such behavior; establishing boundedness or contraction would require additional assumptions and analysis beyond the scope of the present framework.

## S1.3 Neural Network Architecture

We approximate the underlying robust MPC law by a fully connected feed-forward neural network (cf. Section 3.1) with 3 hidden layers, each containing 16 neurons and using a hyperbolic tangent activation function. The network takes the state x as input and outputs the control input u. This architecture is used for the initial and final training phases described in Section S1.4.

## S1.4 Data Generation and Training

We seed 20 814 initial points equally spaced within the state bounds. Using these points, we estimate a robustly forwardinvariant set $\hat { \mathcal { X } } _ { N } \subseteq \mathcal { X } _ { N }$ of the underlying MPC using a backwards-pruning approach.

With the same seed points, we perform an initial training of the NN architecture (cf. Section S1.3) for 200 epochs with a learning rate of $5 \times 1 0 ^ { - 3 }$ , which yields an approximation error of $\begin{array} { r } { \epsilon _ { \mathcal { D } } = 1 . 9 8 7 6 } \end{array}$ , rounded to $\epsilon _ { \mathcal { D } } = 2$ in the 2-norm.

Before dataset generation begins, we estimate the Lipschitz constants of both the MPC and the neural network on $\hat { \mathcal { X } } _ { N }$ using a power-iteration method based on repeated Jacobian–vector and vector–Jacobian products. For the MPC, we obtain $L _ { \mathrm { M P C } } = 1 9 . 8 8 7$ , which we round to $L _ { \mathrm { M P C } } = 2 0$ , and for the initially trained neural network, we obtain 21.436, from which we chose $L _ { \mathrm { N N } } = 2 5$ for caution. This results in $\delta = 0 . 0 2 2 2$ , based on which we set $\eta = \delta$ . We then compute an η-coverage of $\mathcal { X } _ { N }$ by calculating a bounding box of $\hat { \mathcal { X } } _ { N }$ on which a space-filling sampling algorithm is employed.

This results in 1 231 052 coverage points $x _ { i } .$ , which are used to construct the final training dataset D. These points provide an η-coverage of $\hat { \mathcal { X } } _ { N } , \mathrm { i . e . }$

$$
\operatorname* { s u p } _ { x \in \hat { \mathcal { X } } _ { N } } \operatorname* { m i n } _ { x _ { i } \in \mathcal { D } } \lVert x - x _ { i } \rVert _ { 2 } \leq \eta .
$$

Equivalently, the balls of radius η centered at the coverage points cover $\hat { \mathcal { X } } _ { N }$ . Consequently, two neighboring coverage points whose corresponding balls overlap are separated by at most $2 \eta .$ . This bound applies to neighboring points; it does not impose a distance bound on arbitrary pairs of coverage points. A cutout of the cover is shown in Fig. S1

We then train a newly initialized NN on the dataset D. The training concludes after 100 epochs with a learning rate of $5 \times 1 0 ^ { - 3 }$ and a batch size of 1080, achieving an $R ^ { 2 } .$ -score of 0.9999 calculated on a validation set of 10 000 data points, which are distinct from any training data points. Moreover, the trained NN achieves $\dot { \epsilon } _ { \mathcal { D } } = 1 . 9 \bar { 2 } 2 2$ and a Lipschitz estimate $L _ { \mathrm { N N } } ^ { \mathrm { a n , f i n a l } } = 2 3 . 2 2$ , both of which satisfy the initial bounds.

All training, sampling, and simulations were performed in Python on a Lenovo ThinkPad T14 Gen 2a running Windows 11, equipped with an AMD Ryzen 7 PRO 5850U CPU and 32 GB RAM. MPC was implemented using IPOPT via CasADi, and the neural network using PyTorch.

## S1.5 Results

We simulate the system in closed-loop, controlled by the robust MPC and its NN approximation for 100 random initial states. Specifically, the initial states are drawn from a uniform distribution over $\hat { \mathcal { X } } _ { N }$ . The closed-loop system is simulated for 15 s. We observe that the state trajectories under the robust MPC and the NN controller remain close over the whole simulation time and all converge to a neighborhood of the reference point (Figure S2). Furthermore, we can observe that no constraint violations occur, and that all trajectories stay inside $\hat { \mathcal { X } } _ { N }$ . Table S1 shows the mean, maximum, and minimum computation times for both the MPC and NN controller along the 100 trajectories, indicating a significant speed-up using the NN controller. For the same trajectories, Table S2 presents the occurring approximation error in $l _ { 2 } \cdot$ norm, all of which are below our allowed bound of $\epsilon = 3$ . The minimal value is given as 0 since the first observed digit is at the seventh decimal point and therefore not clearly distinguishable from numerical noise.

The results show how safety and stability guarantees established via the underlying MPC transfer to the neural network controller. Furthermore, the results suggest that Theorem 16 is satisfied along all 100 closed-loop trajectories, indicating that the feedback law of the designed MPC is well-behaved

![](images/940fcc33af74d2dab3d25029a73279c42f5031c394f4d919dc2aed490f488aea.jpg)  
Fig. S1. Small cutout of $\hat { \mathcal { X } } _ { N }$ in state space with coverage. Original coverage radius δ as a black line and red circle. Over-coverage radius η as a red line and blue circles. Data points x<sub>i</sub> are represented as orange points.

Table S1

Computation time for the robust MPC and NN controller.
<table><tr><td></td><td>Mean (ms)</td><td>Max (ms)</td><td>Min (ms)</td></tr><tr><td>MPC</td><td>41.79</td><td>213.98</td><td>24.65</td></tr><tr><td>NN approx.</td><td>0.52</td><td>1.98</td><td>0.44</td></tr></table>

and that the proposed Algorithm 1 is suited to compute the NN approximator.

Table S2

Approximation error over 100 trajectories for the robust MPC and NN controller.
<table><tr><td></td><td>Mean</td><td>Max</td><td>Min</td></tr><tr><td>Approx. errors</td><td>0.1943</td><td>1.3497</td><td> $7 . 5 4 4 9 \times 1 0 ^ { - 6 }$ </td></tr></table>

## S2 Weight-projected neural network training

The Lipschitz constant of the feed-forward neural network introduced in Section 3.1 of the main article can be bounded using the Lipschitz constants of its individual layers. Let $L _ { \sigma _ { j } }$ denote a Lipschitz constant of the activation function $\sigma _ { j } ,$ and let $A _ { j }$ denote the weight matrix of the j-th affine layer. Using the submultiplicativity of induced matrix norms and the composition of the network layers, we obtain the analytical upper bound

$$
\hat { L } _ { \mathrm { N N } } ( \theta ) = \| A _ { m } \| _ { 2 } \prod _ { j = 1 } ^ { m - 1 } \left( L _ { \sigma _ { j } } \| A _ { j } \| _ { 2 } \right) ,\tag{S1}
$$

where θ denotes the collection of all network parameters and m is the number of weight matrices. Fixed linear input or output transformations contribute additional induced-norm factors to (S1). Equivalently, they can be incorporated into the first or last affine map when evaluating the bound. Since the bias vectors $b _ { j }$ are constant additional factors and therefore do not affect differences between network outputs, they do not contribute to $\hat { L } _ { \mathrm { N N } } \ [ 1 0 ]$

To enforce a prescribed bound $L _ { \mathrm { N N } } > 0 .$ , the network is trained using projected gradient-based optimization. Starting from the parameters $\theta ^ { ( k ) }$ , an unconstrained optimizer step with respect to the supervised loss $\mathcal { L }$ first produces tentative parameters $\tilde { \theta } ^ { ( k + 1 ) }$ . These parameters are subsequently mapped back to the feasible parameter set

$$
\begin{array} { r } { \Theta _ { L } : = \left\{ \theta \Big | \widehat { L } _ { \mathrm { N N } } ( \theta ) \leq L _ { \mathrm { N N } } \right\} . } \end{array}
$$

We employ a uniform radial rescaling of all m weight matrices. The scaling factor is defined as

$$
\gamma ^ { ( k + 1 ) } = \left\{ \left( \begin{array} { l l } { \left( \frac { L _ { \mathrm { N N } } } { \hat { L } _ { \mathrm { N N } } \left( \tilde { \theta } ^ { ( k + 1 ) } \right) } \right) ^ { 1 / m } , } & { \hat { L } _ { \mathrm { N N } } \left( \widetilde { \theta } ^ { ( k + 1 ) } \right) > L _ { \mathrm { N N } } , } \\ { 1 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. \right.
$$

The updated parameters are then given by

$$
\begin{array} { r l } { { A _ { j } ^ { ( k + 1 ) } = \gamma ^ { ( k + 1 ) } \tilde { A } _ { j } ^ { ( k + 1 ) } , \qquad } } & { { j = 1 , \dots , m , } } \\ { { b _ { j } ^ { ( k + 1 ) } = \tilde { b } _ { j } ^ { ( k + 1 ) } , \qquad } } & { { j = 1 , \dots , m . } } \end{array}
$$

Because induced matrix norms are positively homogeneous, the resulting bound satisfies

$$
\hat { L } _ { \mathrm { N N } } \left( \boldsymbol { \theta } ^ { ( k + 1 ) } \right) = \left( \gamma ^ { ( k + 1 ) } \right) ^ { m } \hat { L } _ { \mathrm { N N } } \left( \boldsymbol { \tilde { \theta } } ^ { ( k + 1 ) } \right) \leq L _ { \mathrm { N N } } .
$$

Thus, every completed optimization step yields parameters that satisfy the prescribed analytical Lipschitz bound [10].

A soft penalty on violations of $L _ { \mathrm { N N } }$ may additionally be included in the training objective to reduce the magnitude and frequency of the rescaling steps. Related projectionbased approaches are discussed in greater detail in [10].

## S3 Existence of a neural network with prescribed finitedataset accuracy

This section provides the complete proof of Lemma 9 in the main article. Specifically, we show that, for any finite dataset with pairwise distinct inputs and any prescribed tolerance $\epsilon _ { D } > 0$ , there exists a feed-forward neural network whose approximation error at every data point is at most $\epsilon _ { \mathcal { D } }$ . The result concerns existence and finite-dataset accuracy only. This statement does not imply that a particular training algorithm finds such a network or that the resulting network satisfies an approximation bound outside the training dataset.

Lemma S1 (Existence ofan $\epsilon _ { \mathcal { D } }$ -interpolating NN.) Let D = $\{ ( x _ { i } , y _ { i } ) \in \mathbb { R } ^ { d } \times \mathbb { R } ^ { p } \ | \ i \in \{ 1 , \dots , n _ { \mathcal { D } } \} \}$ be a dataset with pairwise distinct inputs $x _ { i }$ and $\epsilon _ { \mathcal { D } } > 0$ a given constant. Then there exists a feedforward neural network $\tilde { f } :  { \mathbb { R } ^ { d } } \to$ $\mathbb { R } ^ { p }$ with any continuous non-polynomial activation function such that $\| \tilde { f } ( x _ { i } ) - y _ { i } \| \le \epsilon _ { \mathcal { D } } f o r a l l i = 1 , \dots , n _ { \mathcal { D } } .$

PROOF. The proof consists of two steps. First, we show that there exists a continuous function that interpolates the training data. Second, we apply the universal approximation theorem to conclude the existence of a neural network that approximates this interpolant with ϵ<sub>D</sub> accuracy. As a result, this neural network satisfies $\| { \tilde { f } } ( x _ { i } ) - y _ { i } \| \leq$ ϵ<sub>D</sub> point-wise on the training data, which proves the claim.

Step 1: Consider the set $\mathcal { S } = \{ x _ { i } \ | \ i \in \left\{ 1 , \dots , n _ { \mathcal { D } } \right\} \} \subset \mathbb { R } ^ { d }$ consisting of the training inputs. Since S is finite, it is closed. Now, define the function $g : S \to \mathbb { R } ^ { p } \ { \mathfrak { b } } \mathbf { y } \ g ( x _ { i } ) = y _ { i }$ for all $i = 1 , \dots , n _ { \mathcal { D } }$ . We now aim to construct a continuous extension of the function $g$ from the set S to the entire space $\mathbb { R } ^ { d }$ , yielding the desired interpolant. To this end, we consider the component functions $g _ { j } : S  \mathbb { R }$ , such that $g ( x ) =$ $( g _ { 1 } ( x ) , \bar { \ldots } , g _ { p } ( x ) )$ , and each component satisfies $g _ { j } ( x _ { i } ) =$ $y _ { i } ^ { ( j ) }$ , where $y _ { i } = ( y _ { i } ^ { ( 1 ) } , \dots , y _ { i } ^ { ( p ) } )$ . $\mathbf { A } \mathbf { s } \mathbb { R } ^ { d }$ , equipped with any p-norm, is a normal topological space and $S$ is closed, the Tietze extension theorem [16, Theorem 12.4] guarantees that each real-valued component function $g _ { j }$ can be continuously extended from $S \operatorname { t o } \mathbb { R } ^ { d }$ . We denote the continuous extension of $g _ { j }$ by $q _ { j } .$ , and define the continuous interpolant $\boldsymbol { q } : \mathbb { R } ^ { d } $ $\mathbb { R } ^ { p }$ of the training data by $q ( x ) = ( q _ { 1 } ( x ) , \ldots , q _ { p } ( x ) )$ .

![](images/07bc531d24fab410ec3260c4722d12941b2787fb8fd24031304913d05eb220c7.jpg)  
Fig. S2. 100 closed-loop state trajectories of the robust MPC and the NN controllers in state space with initial states in $\hat { \mathcal { X } } _ { N }$ , shown in white. The four insets provide enlarged views of selected trajectories for clarity.

Step 2: We consider a compact set $\mathcal { K } \subset \mathbb { R } ^ { d }$ of interest such that $s \subset \kappa$ . By the universal approximation theorem [13], for any $\delta ^ { \prime } > 0 .$ , there exists a neural network $\tilde { f }$ $\mathbb { R } ^ { d }  \mathbb { R } ^ { p }$ such that $\mathrm { s u p } _ { x \in \mathcal { K } } \| \tilde { f } ( x ) - q ( x ) \| _ { \infty } < \delta ^ { \prime }$ . The result can be extended to arbitrary p-norms with $1 \le p <$ ∞ by using norm equivalences in finite-dimensional real spaces. Specifically, for any $1 \leq p < \infty$ , there exists a constant $c _ { p } \geq 1$ such that $\| v \| _ { p } \leq c _ { p } \| v \| _ { \infty }$ for all $v \in \mathbb { R } ^ { p }$ and trivially, $c _ { p } = 1 \mathrm { i f } p = \infty .$ . Thus, and taking $\delta ^ { \prime } = \epsilon _ { \mathcal { D } } / c _ { p } ,$ we obtain sup $\nu _ { x i n K } \| \tilde { f } ( x ) - q ( x ) \| < \epsilon _ { \mathcal { D } }$ , and in particular, $\| \tilde { f } ( x _ { i } ) - y _ { i } \| \le \epsilon _ { \mathcal { D } }$ for all i, which concludes the proof.

## References

[1] A. Alessio and A. Bemporad. A survey on explicit model predictive control, pages 345–369. Springer Berlin Heidelberg, 2009.

[2] H. Alsmeier, L. Theiner, A. Savchenko, A. Mesbah, and R. Findeisen. Imitation learning of MPC with neural networks: error guarantees and sparsification. In Conf. Decis. Control, pages 4777–4782, 2024.

[3] J. A E Andersson, J. Gillis, G. Horn, J. B Rawlings, and M. Diehl. CasADi – A software framework for nonlinear optimization and optimal control. Math. Program. Comput., 11:1–36, 2019.

[4] J. Ansel, E. Yang, H. He, et al. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. In 29th ACM Int. Conf. Archit. Support Program. Lang. Oper. Syst., 2024.

[5] A. Bemporad, M. Morari, V. Dua, and E. N. Pistikopoulos. The explicit linear quadratic regulator for constrained systems. Automatica, 38:3–20, 2002.

[6] C. Buskens and H. Maurer.¨ Sensitivity analysis and real-time optimization of parametric nonlinear programming problems, pages 3–16. Springer Berlin Heidelberg, 2001.

[7] S. Chen, K. Saulnier, N. Atanasov, D. D. Lee, V. Kumar, G. J. Pappas, and M. Morari. Approximating explicit model predictive control using constrained neural networks. In Am. Control Conf., pages 1520–1527, 2018.

[8] R. Drummond, S. Duncan, M. Turner, P. Pauli, and F. Allgower.¨ Bounding the difference between model predictive control and neural networks. In Learn. Dyn. Cont. Conf., pages 817–829, 2022.

[9] A. V. Fiacco and Y. Ishizuka. Sensitivity and stability analysis for nonlinear programming. Ann. Oper. Res., 27:215–235, 1990.

[10] H. Gouk, E. Frank, B. Pfahringer, and M. J. Cree. Regularisation of neural networks by enforcing Lipschitz continuity. Mach. Learn., 110:393–416, 2021.

[11] L. Grune and J. Pannek. ¨ Nonlinear model predictive control: theory and algorithms. Springer, 2017.

[12] M. Hertneck, J. Kohler, S. Trimpe, and F. Allg ¨ ower. Learning an ¨ approximate model predictive controller with guarantees. Control Sys. Lett., 2:543–548, 2018.

[13] K. Hornik. Approximation capabilities of multilayer feedforward networks. Neural Netw., 4:251–257, 1991.

[14] K. Hornik. Some new results on neural network approximation. Neural Netw., 6:1069–1072, 1993.

[15] H. Hose, J. Kohler, M. N. Zeilinger, and S. Trimpe. Approximate ¨ nonlinear model predictive control with safety-augmented neural networks. IEEE Trans. Control Syst. Technol., 33:2490–2497, 2025.

[16] G. J. O. Jameson. Topology and normed spaces. Chapman and Hall, 1974.

[17] B. Karg and S. Lucia. Efficient representation and approximation of model predictive control laws via deep learning. Trans. Cybern., 50:3866–3878, 2020.

[18] M. Kiss, R. Toth, and M. Schoukens. Space-filling input design´ for nonlinear state-space identification. IFAC-PapersOnLine, 58:562– 567, 2024.

[19] D. Limon Marruedo, T. Alamo, and E. F. Camacho. Input-to-state´ stable MPC for constrained discrete-time nonlinear systems with bounded additive uncertainties. In Conf. Decis. Control, pages 4619– 4624, 2002.

[20] S. Lucia and B. Karg. A deep learning-based approach to robust nonlinear model predictive control. IFAC-PapersOnLine, 51:511– 516, 2018.

[21] S. Lucia, M. Kogel, P. Zometa, D. E. Quevedo, and R. Findeisen.¨ Predictive control, embedded cyberphysical systems and systems of systems–a perspective. Annu. Rev. Control, 41:193–207, 2016.

[22] L. Luken, D. Brandner, and S. Lucia. Sobolev training for data-¨ efficient approximate nonlinear MPC. IFAC-PapersOnLine, 56:5765– 5772, 2023.

[23] E. T. Maddalena, C. G. da S. Moraes, G. Waltrich, and C. N. Jones. A neural network architecture to learn explicit MPC controllers from data. IFAC-PapersOnLine, 53:11362–11367, 2020.

[24] J. Nubert, J. Kohler, V. Berenz, F. Allg¨ ower, and S. Trimpe. Safe¨ and fast tracking on a robot manipulator: robust MPC and neural network control. Robot. Autom. Lett., 5:3050–3057, 2020.

[25] A. Paszke, S. Gross, F. Massa, et al. Pytorch: An imperative style, high-performance deep learning library. In Adv. Neural Inf. Process. Syst., 2019.

[26] P. Pauli, A. Koch, J. Berberich, P. Kohler, and F. Allg ¨ ower. Training¨ robust neural networks using Lipschitz bounds. Control Syst. Lett., 6:121–126, 2021.

[27] J. A. Paulson and A. Mesbah. Approximate closed-loop robust model predictive control with guaranteed stability and constraint satisfaction. Control Syst. Lett., 4:719–724, 2020.

[28] J. B. Rawlings, D. Q. Mayne, and M. Diehl. Model predictive control: theory, computation, and design. Nob Hill Publishing, 2017.

[29] A. Rose, M. Pfefferkorn, H. H. Nguyen, and R. Findeisen. Learning a Gaussian process approximation of a model predictive controller with guarantees. In Conf. Decis. Control, pages 4094–4099, 2023.

[30] A. Rose, P. Schaub, and R. Findeisen. Safe and high-performance learning of model predictive control using kernel-based interpolation. In Am. Control Conf., pages 91–96, 2025.

[31] P. O. M. Scokaert and J. B. Rawlings. Feasibility issues in linear model predictive control. AIChE Journal, 45:1649–1659, 1999.

[32] D. Teichrib and M. Schulze Darup. Efficient computation of Lipschitz constants for MPC with symmetries. In Conf. Decis. Control, pages 6685–6691, 2023.

[33] A. Tokmak, C. Fiedler, M. N. Zeilinger, S. Trimpe, and J. Kohler. Automatic nonlinear MPC approximation with closed-loop¨ guarantees. Trans. Autom. Control, 2025.

[34] M. Zanon and S. Gros. Safe reinforcement learning using robust MPC. Trans. Autom. Control, 66:3638–3652, 2021.

[35] M. N. Zeilinger, C. N. Jones, and M. Morari. Real-time suboptimal model predictive control using a combination of explicit MPC and online optimization. Trans. Autom. Control, 56:1524–1534, 2011.