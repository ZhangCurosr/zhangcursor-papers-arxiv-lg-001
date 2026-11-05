# ConDR: Conservative Reachability through Neural Hamilton–Jacobi Subsolutions

Kendra Givens<sup>∗</sup>, Taylor T. Johnson<sup>∗</sup>, Bryan C. Ward<sup>∗</sup>, and Thomas Beckers<sup>†</sup>

<sup>∗</sup>Computer Science, Vanderbilt University, United States

{kendra.l.givens, taylor.johnson, bryan.ward}@vanderbilt.edu

<sup>†</sup>Institute of Measurement, Control and Microtechnology, Ulm University, Germany thomas.beckers@uni-ulm.de

Abstract—Hamilton-Jacobi (HJ) Reachability identifies the unsafe states of a dynamical system from which it can be forced into failure under worst-case disturbance. However, classical grid-based solvers scale exponentially with state dimension. DeepReach improves scalability by replacing the grid with a neural value function trained on the HJ-Isaacs (HJI) variational inequality (VI), but penalizes both sides of the residual equally. This allows the learned value to lie above the true one and unsafe states to be misclassified as safe. We propose Conservative DeepReach (ConDR) which aims to learn a subsolution of the HJI VI such that the learned unsafe set contains the true one. ConDR enforces the obstacle branch and terminal condition by construction, pushes the HJI residual to a safety margin ϵ below zero with a one-sided loss, and prevents the degenerate solution of classifying everything as unsafe. We prove that the exact solution of the margin-shifted VI that ConDR trains toward is a subsolution of the original VI so its unsafe set contains the true one and its value lies within ϵτ of the true value function. We compare against DeepReach and MADR in the established systems of Air3D, Air6D, a 9D multi-vehicle system, and a 13D quadrotor. ConDR reports the lowest rate of unsafe state misclassified and the lowest closed-loop collision rates with at most a small increase in intervention. On Air3D and Air6D, ConDR tolerates model mismatch that scales with ϵ. At matched false-negative rates, ConDR’s unsafe sets are smaller than the baselines showing the safety is not obtained by trivial overapproximation.

Index Terms—Hamilton-Jacobi, reachability, safety, dynamical systems, closed-loop, control

## I. INTRODUCTION

<sup>1</sup> As more complex and autonomous systems are deployed in safety-critical settings, verifying their behavior becomes increasingly important. These systems often have nonlinear dynamics, operate under control and state constraints, and may be affected by external or even adversarial disturbances. Hamilton-Jacobi (HJ) reachability provides worst-case safety guarantees under exactly these conditions. By solving for the value function, HJ reachability identifies the backward reachable tube (BRT), which represents the set of states from which the system can be forced into the target set, a chosen unsafe set of states, within the time horizon under worst-case disturbance. The value function can be used to recover the optimal control policy.

However, HJ reachability suffers from the curse of dimensionality. Classical HJ solvers use level-set methods on a discretized grid [1] where both memory and runtime grow exponentially with the dimension of the system state. Exact methods can extend beyond low-dimensional systems, but require the system to decompose into coupled subsystems [2]. Recently, neural approaches have improved the scalability. DeepReach [3] overcomes this limitation by representing the value function by an implicit neural representation (INR) [4] instead of a grid, which avoids direct dependence on spatial discretizations and makes higher-dimensional problems more tractable. The ExactBC [5] variant removes the terminal condition by construction. MPC-guided Adversarial DeepReach (MADR) [6] expands on DeepReach and ExactBC by augmenting the loss with a self-generated signal from model predictive control (MPC) rollouts, which improves convergence. The learned value function is then used as a least-restrictive safety filter in closed loop rollouts [7], [8]. Control barrier functions [9] provide an alternative safety filter, but require the barrier function to be provided or learned beforehand, whereas HJ reachability only requires the dynamics.

Despite the scalability benefits, neural HJ approaches recover the exact value function only in the limit of a vanishing residual over the whole domain [10]. However, this is never achieved in practice, as the residual is typically nonzero at most points. As a result, the sign of the residual matters, i.e., a positive residual pushes the learned value function above the true one and results in higher false negatives (FNs) where unsafe states are classified as safe. A negative residual only enlarges the learned unsafe set resulting in higher false positives (FPs). Since both sides are penalized equally, FNs can be as frequent as FPs. Deploying a learned value function in a real control system requires some protection against the FNs. There exist methods that can bound the error of the model training [11], [12], but they inflate the volume of the level-set by the size of the worst violation. Additionally, neither changes the training objective that produced the violation.

We propose Conservative DeepReach (ConDR), a neural HJ reachability method that trains towards a subsolution of the HJI VI, such that the target BRT overapproximates the true one. We enforce the obstacle branch [13] and terminal condition of the HJI VI at all times by construction, using a one-sided L1 loss to push the residual to the safe side with safety margin ϵ, and add a complementarity loss term to prevent the degenerate solution of declaring everything as unsafe. This results in smaller post-training corrections, which in turn yield a tighter tube volume and fewer collisions during closed-loop safetyfiltering rollouts, while tolerating model mismatch up to ϵ. Contributions Our key contributions are

1) ConDR: a conservative neural HJ reachability model that uses an easy to implement margin-shifted one-sided loss, trains toward a subsolution of the HJI VI, and composes with MADR.

2) Guarantees that the BRT of the exact margin-shifted VI solution contains the true BRT, overapproximated by at most $\epsilon \tau ,$ and that a value function whose residual is at most −ϵ remains a subsolution under model mismatch up to ϵ.

3) Evaluation against DeepReach and MADR under matched experimental setups on Air3D, Air6D, a 9D multi-vehicle system, and a 13D quadrotor where ConDR has the lowest FN, subsolution violations, and collision rates in every setting tested.

4) A tunable safety margin i.e. sweeping ϵ on Air3D and Air6D shows the tolerated model mismatch scales with ϵ with a linear accuracy cost in ϵ.

## II. BACKGROUND

A. HJ Reachability and Subsolution

Consider a dynamical system

$$
{ \dot { x } } ( t ) = f ( x ( t ) , u ( t ) , d ( t ) ) ,\tag{1}
$$

where $\boldsymbol { x } ( t ) \in \mathcal { X } \subset \mathbb { R } ^ { n }$ is the system state, $u ( t ) \in \mathcal { U } \subset \mathbb { R } ^ { m }$ is the control input, and $d ( t ) \in { \mathcal { D } } \subset \mathbb { R } ^ { k }$ is the disturbance input. The dynamics are given by $f : \mathcal { X } \times \mathcal { U } \times \mathcal { D }  \mathbb { R } ^ { n }$ , such that there always exists a unique solution for (1). HJ reachability can be viewed as a two-player zero-sum differential game in which one player (disturbance) attempts to steer the system toward an unsafe set while the other (controller) attempts to steer it away within a time horizon $T \in \mathbb { R } _ { > 0 }$ . Let the unsafe set $s$ be represented implicitly by a Lipschitz continuous function $g : \mathbb { R } ^ { n }  \mathbb { R }$ , such that

$$
S = \{ x \in \mathbb { R } ^ { n } \mid g ( x ) \leq 0 \} ,\tag{2}
$$

where $g ( x )$ is non-positive inside the unsafe set and positive outside it. The cost is defined as

$$
J ( x , u ( \cdot ) , d ( \cdot ) ) = \operatorname* { m i n } _ { s } g ( x ( s ) ) , s \in [ 0 , \tau ] , x ( 0 ) = x .\tag{3}
$$

where τ denotes the time remaining before the time horizon ends. Assuming the controller maximizes the cost to keep the system away from S and the disturbance minimizes it to push the system towards S, the value function $V : \mathbb { R } _ { > 0 } \times \mathbb { R } ^ { n } \to$ R is

$$
V ( \tau , x ) = \operatorname* { s u p } _ { u ( \cdot ) } \operatorname* { i n f } _ { d ( \cdot ) } \ J ( x , u ( \cdot ) , d ( \cdot ) ) .\tag{4}
$$

The backward reachable tube (BRT) B is the set of states whose trajectories can reach the unsafe set within the remaining time τ. It is given by the zero sub-level set of the value function

$$
B ( \tau ) = \{ x \in \mathbb { R } ^ { n } \mid V ( \tau , x ) \leq 0 \} .\tag{5}
$$

The value function is the solution to the Hamilton–Jacobi– Isaacs (HJI) variational inequality (VI) given by

$$
\begin{array} { c } { \operatorname* { m a x } \{ \partial _ { \tau } V - H ( x , \nabla _ { x } V ) , V - g ( x ) \} = 0 } \\ { V ( 0 , x ) = g ( x ) } \end{array}\tag{6}
$$

with the Hamiltonian H defined as

$$
H ( x , p ) = \operatorname* { m a x } _ { u \in \mathcal { U } } \operatorname* { m i n } _ { d \in \mathcal { D } } p ^ { \top } f ( x , u , d ) ,\tag{7}
$$

where $p = \nabla _ { x } V ( \tau , x )$ . We denote $\partial _ { \tau } V - H ( x , \nabla _ { x } V )$ as the differential branch and $V - g ( x )$ as the obstacle branch [13].

## B. DeepReach

DeepReach [3] represents the value function with a SIREN, an implicit neural representation, and trains it in an unsupervised, physics-informed manner [14] on the HJI VI. Deep Reach learns $V _ { \theta } ( \tau , x ) \approx V ( \tau , x )$ where θ denotes the network parameters, τ represents the time-to-go, and $x$ the spatial coordinates of the learned value function $V _ { \theta } \colon \mathbb { R } _ { > 0 } \times \mathbb { R } ^ { n } \to \mathbb { R }$ The original DeepReach required an additional loss term to satisfy the terminal condition. The ExactBC [5] variant removes that loss term and allows the model to satisfy the terminal condition by construction. It modifies the output of the neural network as

$$
V _ { \theta } ( \tau , x ) = g ( x ) + \tau \mathbf { N } \mathbf { N } _ { \theta } ( \tau , x )\tag{8}
$$

such that when $\tau = 0 , V _ { \theta } = g ( x )$ . The loss for ExactBC DeepReach is given by

$$
\mathcal { L } _ { \mathrm { { D R } } } = \mathbb { E } _ { \tau , x } [ | \operatorname* { m a x } ( \partial _ { \tau } V _ { \theta } - H ( x , \nabla _ { x } V _ { \theta } ) , V _ { \theta } - g ) | ] ,\tag{9}
$$

which penalizes violations on both sides of the residual equally allowing unsafe states to be misclassified as safe. The model is trained with a curriculum over time-to-go. Spatial gradients of the value function needed to evaluate the Hamiltonian and recover the control policy, are obtained through automatic differentiation. We refer to ExactBC DeepReach as DeepReach throughout.

## C. MADR

DeepReach’s residual only signal can be too weak for high-dimensional systems so DeepReach-MPC [15] added a supervised loss signal with the dataset obtained by model predictive control (MPC) rollouts. MPC-guided Adversarial DeepReach (MADR) [6] extends DeepReach MPC [15] to handle adversarial input. It runs MPC across sampled states and times with the disturbance chosen adversarially from the gradient of the current value function. MADR uses the running-minimum cost as an approximate value function label and an L1 loss term is used to penalize the distance between the $V _ { \theta }$ and the label. The labels are regenerated periodically from the model so no ground truth ever enters the model. An additional loss term is added that penalizes the network if $V _ { \theta }$ calls a state safe that was not safe in the rollout. However, MADR still inherits the DeepReach problem of penalizing both sides of the residual equally. We also use the ExactBC version of MADR, referred to as just MADR throughout.

## D. Problem Statement

Given a dynamical system (1), an unsafe set of states $s$ (2) and a time horizon $T ~ > ~ 0$ , we aim to learn a neural value function $V _ { \theta }$ approximating the largest subsolution of the margin-shifted HJI VI such that

$V _ { \theta } ( \tau , x ) \leq V _ { \mathrm { t r u e } } ( \tau , x ) , \forall ( \tau , x ) \in [ 0 , T ] \times \mathcal { X } .$ , where $V _ { \mathrm { t r u e } }$ represents the exact solution to the HJI VI (6)

• and the zero sub-level set $\{ x \ \in \ \mathcal { X } : \ V _ { \theta } ( \tau , x ) \ \leq \ 0 \}$ is an over-approximation containing the true Backward Reachable Tube $\vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert \vert$

At a high level, we aim to learn a value function $V _ { \theta }$ that lies below the true value function V so that its learned unsafe set over-approximates the true one to reduce the risk of misclassifying unsafe states as safe.

## III. METHODS

A subsolution for the HJI VI requires both the obstacle and differential branch in (6) to be non-positive. We build on DeepReach to encourage $V _ { \theta }$ to satisfy the subsolution property. ConDR constructs the output of the neural network to have a non-positive obstacle branch by construction and modifies the loss to train towards satisfying the differential branch without converging to the degenerate solution of declaring every state unsafe.

## A. HJI VI Subsolution

The true HJI VI solution satisfies both the obstacle and differential branch inequalities and has at least one of the branches to be an equality at every point. It additionally requires the terminal condition $V ( 0 , x ) ~ = ~ g ( x )$ . A valid subsolution drops the equality requirement i.e. both $V \leq g$ and $\partial _ { \tau } V - H ( x , \nabla _ { x } V ) \leq 0$ must be satisfied at every point without requiring the equality and requires a relaxed version of the terminal condition $V ( 0 , x ) \leq g ( x )$ . In the following, we present new approaches for the obstacle and the differential branch towards this goal.

## B. Obstacle Branch

We build on ExactBC’s exact construction of the terminal condition to also enforce the obstacle branch $V \ \leq \ g .$ . We constrain the correction to be nonpositive such that the exact construction then becomes

$$
V _ { \theta } ( \tau , x ) = g ( x ) - \tau \mathrm { s o f t p l u s } ( \mathbf { N N } _ { \theta } ( \tau , x ) ) .\tag{10}
$$

Therefore, the learned value function $V _ { \theta } ( \tau , x )$ is always less than or equal to $g ( x )$ for every $\tau > 0$ and $x \in \mathcal { X }$ as shown later in Proposition 1. In this way, we can satisfy the inequality of the obstacle branch by construction.

## C. Differential Branch

Unlike the obstacle branch, we cannot enforce the $V _ { \theta }$ to be a valid subsolution of the differential branch through construction. Therefore, we train towards it through the loss term. For that, let $r = \partial _ { \tau } V - H ( x , \nabla _ { x } V )$ be the residual and $r _ { \theta }$ denote the residual with the learned $V _ { \theta } .$ Rather than target $r _ { \theta } \leq 0 ,$ , we instead introduce a safety margin $\epsilon > 0$ to target $r _ { \theta } \leq - \epsilon .$ . We define the margin-shifted VI, whose exact solution we denote $V _ { \epsilon }$

$$
\begin{array} { c } { \displaystyle \operatorname* { m a x } \{ r _ { \epsilon } + \epsilon , V _ { \epsilon } - g ( x ) \} = 0 } \\ { \displaystyle V _ { \epsilon } ( 0 , x ) = g ( x ) . } \end{array}\tag{11}
$$

ConDR trains $V _ { \theta }$ toward $V _ { \epsilon }$ using the losses below. To encourage $r _ { \theta } \leq - \epsilon$ , we add the loss term

$$
\mathcal { L } _ { \mathrm { d i f f } } = \mathbb { E } [ \mathrm { r e l u } ( r _ { \theta } + \epsilon ) ] .\tag{12}
$$

If $r _ { \theta } \leq - \epsilon ,$ , the loss term is 0. When $r _ { \theta } > - \epsilon _ { * }$ the loss term grows linearly with the error. Since the model can only sample a finite number of points and will always have some nonzero residual, ϵ allows the model to satisfy the original inequality despite approximation error i.e. a residual error smaller than ϵ still leaves $r _ { \theta } ~ \le ~ 0$ and $V _ { \theta }$ remains a subsolution to the original HJI VI. Setting $\epsilon = 0$ recovers a one sided version of the original DeepReach loss.

This loss term alone only penalizes residuals above −ϵ and is indifferent to how far below −ϵ they ${ \bf g 0 }$ . This can lead to a degenerate solution that predicts everything as unsafe. The original HJI VI prevents this since at least one of the branches must be an equality at each point. We prevent the degenerate solution by encouraging at least one of the branches to be tight i.e. to train towards the largest subsolution of the marginshifted VI.

$$
\mathcal { L } _ { \mathrm { d e g } } = \mathbb { E } [ \operatorname* { m i n } ( | r _ { \theta } + \epsilon | , g ( x ) - V _ { \theta } ) ] .\tag{13}
$$

At each point, the loss equals the smaller of the two branches so that when one branch is tight, the loss is small.

We also add another loss term which penalizes residuals that exceed the margin on the safe side to aid in optimization. It is the negated version of the ${ \mathcal { L } } _ { \mathrm { d i f f } }$ loss given by

$$
\mathcal { L } _ { \mathrm { o p t } } = \mathbb { E } [ \mathrm { r e l u } ( - ( r _ { \theta } + \epsilon ) ) ] .\tag{14}
$$

The full objective is then

$$
\mathcal { L } = w _ { \mathrm { d i f f } } \mathcal { L } _ { \mathrm { d i f f } } + w _ { \mathrm { d e g } } \mathcal { L } _ { \mathrm { d e g } } + w _ { \mathrm { o p t } } \mathcal { L } _ { \mathrm { o p t } } ,\tag{15}
$$

where $w _ { \mathrm { d i f f } } , w _ { \mathrm { d e g } } , w _ { \mathrm { o p t } } \ \in \ \mathbb { R } \ > \ 0$ are hyperparameters that weight each loss term.

## D. Formal Guarantees

We prove that ConDR satisfies the obstacle branch and terminal condition exactly for every network in Proposition 1. In Theorem 1, we prove the solution of the margin-shifted VI that ConDR trains toward is a subsolution of the original VI. Corollary 1, proves that we can shift the BRT after training to contain the true BRT. Finally, in Theorem 2, we show that ConDR can tolerate model mismatch up to ϵ between the true and modeled Hamiltonian.

Proposition 1 (Exact Construction). For every $\theta \in \mathbb { R } ^ { p } ,$ where p denotes the number of parameters in the neural network of $V _ { \theta } ,$ , the construction (10) satisfies $V _ { \theta } ( 0 , x ) = g ( x )$ and $V _ { \theta } ( \tau , x ) < g ( x )$ for all $x \in \mathbb { R } ^ { n }$ and $\tau > 0$

Proof. $\mathbf { A } \mathbf { t } \boldsymbol { \tau } = 0$ , the second term in (10) is zero so $V _ { \theta } ( 0 , x ) =$ $g ( x )$ . For $\tau > 0$ , the correction is τ times a positive quantity.

Subtracting a positive number from g can only decrease $V _ { \theta }$ Therefore, $V _ { \theta } ( \tau , x ) = g ( x )$ at $\tau = 0$ and $V _ { \theta } ( \tau , x ) < g ( x )$ for all $\tau > 0$ □

Proposition 1 shows that regardless of the neural network’s parametrization, the obstacle branch and terminal condition will always be satisfied.

Theorem 1 (Bounding the Subsolution). Let V<sub>ϵ</sub> represent the exact solution to the margin shifted VI (11). Then, for all $( \tau , x ) \in [ 0 , T ] \times \mathbb { R } ^ { n } , V _ { t r u e } ( \tau , x ) - \epsilon \tau \leq V _ { \epsilon } ( \tau , x ) \leq V _ { t r u e } ( \tau , x )$ $T h e r e f o r e , f o r \ e \nu e r y \ \tau \in [ 0 , T ] , \ \{ x \in \mathbb { R } ^ { n } : V _ { t r u e } ( \tau , x ) \le 0 \} \subseteq$ $\{ x \in \mathbb { R } ^ { n } : V _ { \epsilon } ( \tau , x ) \leq 0 \} \subseteq \{ x \in \mathbb { R } ^ { n } : V _ { t r u e } ( \tau , x ) \leq \epsilon \tau \}$

Proof. Let $\begin{array} { l l l } { r [ V ] } & { = } & { \partial _ { \tau } V - H ( x , \nabla _ { x } V ) } \end{array}$ and let $B \_ =$ $V _ { \mathrm { t r u e } } ( \tau , x ) - \epsilon \tau$ so that $\partial _ { \tau } B = \partial _ { \tau } V _ { \mathrm { t r u e } } ( \tau , x ) - \epsilon .$ For the upper bound, $r [ V _ { \epsilon } ] \le - \epsilon \le 0 , V _ { \epsilon } \le g$ and $V _ { \epsilon } ( 0 , x ) = g \mathfrak { a }$ therefore, $V _ { \epsilon }$ is a subsolution to the original VI and lies below the true solution by the comparison principle [16]. Therefore, $V _ { \epsilon } \ \leq \ V _ { \mathrm { t r u e } }$ . For the lower bound, we will prove each branch and terminal condition. For the differential branch, $\nabla _ { x } B \ = \ \nabla _ { x } V _ { \mathrm { t r u e } } ,$ so $H ( x , \nabla _ { x } B ) \ = \ H ( x , \nabla _ { x } V _ { \mathrm { t r u e } } )$ . Additionally, $\partial _ { \tau } B = \partial _ { \tau } V _ { \mathrm { t r u e } } - \epsilon .$ . Therefore, $r [ B ] = r [ V _ { \mathrm { t r u e } } ] - \epsilon .$ Since $r [ V _ { \mathrm { t r u e } } ] ~ \leq ~ 0 , ~ r [ B ] ~ \leq ~ - \epsilon$ . For the obstacle branch, $B = V _ { \mathrm { t r u e } } - \epsilon \tau \leq V _ { \mathrm { t r u e } }$ since $\epsilon \tau \geq 0$ and $V _ { \mathrm { t r u e } } ~ \leq ~ g$ since $V _ { \mathrm { t r u e } }$ satisfies the obstacle branch (6). Therefore, $B \leq g .$ . For the terminal condition, $\tau = 0$ reduces the $\epsilon \tau$ term to zero. Therefore, $B ( 0 , x ) = V _ { \mathrm { t r u e } } ( 0 , x ) = g .$ As a consequence, B is a subsolution of the margin shifted VI and lower bounds $V _ { \epsilon }$ meaning that $V _ { \mathrm { t r u e } } - \epsilon \tau \le V _ { \epsilon } \le V _ { \mathrm { t r u e } }$ and $V _ { \epsilon }$ is at most ϵτ away from $V _ { \mathrm { t r u e } }$ □

Remark 1. The upper bound $V _ { \epsilon } \leq V _ { t r u e }$ follows because $V _ { \epsilon }$ is a subsolution of the original VI and lies below the exact solution $V _ { t r u e } .$ . However, V<sub>ϵ</sub> could be arbitrarily below $V _ { t r u e }$ and needs a valid lower bound to prevent the degenerate solution of declaring every state as unsafe. In general, $V _ { t r u e }$ is not a subsolution of the margin VI since $r [ V _ { t r u e } ] = 0$ whenever the differential branch is active and the margin VI requires $r \leq - \epsilon .$ . Therefore, we shift $r [ V _ { t r u e } ]$ by ϵ. We cannot add ϵ to $H ( x , \nabla _ { x } V )$ without changing $\nabla _ { x } V$ so we lower $\partial _ { \tau } V ~ b y ~ \epsilon$ instead.

Theorem 1 proves that the exact solution to the marginshifted VI that ConDR trains toward, $V _ { \epsilon }$ , lies below $V _ { \mathrm { t r u e } }$ and that its BRT never misclassifies an unsafe state. Additionally, the BRT of $V _ { \epsilon }$ overapproximates the true BRT by at most ϵτ. Theorem 1 concerns the exact subsolution $V _ { \epsilon }$ but, in practice, a trained neural network can only satisfy the residual approximately. Corollary 1 converts its worst violation into a level-set shift to certify the resulting backward reachable tube.

Corollary 1 (Certification). Let $V _ { \theta } \in C ^ { 1 }$ with $V _ { \theta } ( 0 , x ) = g$ and let $\delta \ge 0 . ~ I f ~ V _ { \theta } ( \tau , x ) \le g$ and $r _ { \theta } \leq \delta$ everywhere, then $V _ { \theta } ( \tau , x ) \le V _ { t r u e } ( \tau , x ) + \delta \tau \le V _ { t r u e } ( \tau , x ) + \delta T$ so for all $\tau \in$ $[ 0 , T ] ,$ , the true BRT is contained within the set $\{ x \in \mathbb { R } ^ { n }$ $V _ { \boldsymbol { \theta } } ( \tau , x ) \leq \delta T \}$

Proof. Let $W = V _ { \theta } - \delta \tau .$ , then $\nabla _ { x } W = \nabla _ { x } V _ { \theta }$ and $\partial _ { \tau } W =$ $\partial _ { \tau } V _ { \theta } - \delta$ so that $r [ W ] ~ = ~ r _ { \theta } - \delta ~ \leq ~ 0$ . Since $\delta \tau \geq 0 ,$ $W ~ \le ~ V _ { \theta } ( \tau , x ) ~ \le ~ g$ and $W ( 0 , x ) ~ = ~ g ( x )$ . Therefore, W is a subsolution of the original VI and $W ~ \leq ~ V _ { \mathrm { t r u e } }$ so $V _ { \theta } ( \tau , x ) \leq V _ { \mathrm { t r u e } } + \delta \tau .$ At any state with $V _ { \mathrm { t r u e } } \leq 0$ , this gives $V _ { \theta } ( \tau , x ) \leq \delta \tau \leq \delta T$ □

Corollary 1 converts a global bound δ of the residual violation into a level-set shift where the shifted, learned BRT $\{ x \in \mathbb { R } ^ { n } : V _ { \theta } ( \tau , x ) \leq \delta T \}$ contains the true one whenever $r _ { \theta } ~ \leq ~ \delta$ holds everywhere. However, this can only be a probabilistic estimate with finitely many samples.

Theorem 2 (Model Mismatch). Let $V _ { \theta } ( \tau , x ) ~ \leq ~ g$ with $\begin{array} { r l r } { V _ { \theta } ( 0 , x ) } & { { } = } & { g ( x ) } \end{array}$ . Let H<sup>ˆ</sup> be the Hamiltonian of the modeled/perturbed dynamics. If $r _ { \theta } ~ \leq ~ - \epsilon$ under H<sup>ˆ</sup> and $| H ( x , \nabla _ { x } V _ { \theta } ) - \hat { H } ( x , \nabla _ { x } V _ { \theta } ) | \leq \epsilon$ for all $( \tau , x ) \in [ 0 , T ] \times \mathbb { R } ^ { n }$ then $\partial _ { \tau } V _ { \theta } - H ( x , \nabla _ { x } V _ { \theta } ) \leq 0$ so $V _ { \theta }$ is a subsolution ofthe true system’s VI (6) and $V _ { \theta } \leq V _ { t r u e }$ by the comparison principle [16]. Therefore, the true system’s BRT is contained in the learned BRT.

Proof. The obstacle branch and terminal condition do not involve the dynamics and hold by hypothesis, for ConDR by construction (10), while the differential branch is satisfied because $\partial _ { \tau } V _ { \theta } ( \tau , x ) - H = ( \partial _ { \tau } V _ { \theta } ( \tau , x ) - \hat { H } ) + ( \hat { H } - H ) \leq$ $- \epsilon + \epsilon = 0$ so $V _ { \theta } ( \tau , x )$ is a subsolution of the true VI (6).

Theorem 2 proves that a residual trained ϵ below zero remains a valid subsolution under Hamiltonian perturbations up to ϵ.

## IV. EXPERIMENTAL SETUP

We evaluate ConDR on four systems from the DeepReach and DeepReach-MPC benchmarks all with horizon $T = 1$

## A. Systems

Air3D [17] is a three dimensional system modeling a twoaircraft collision-avoidance scenario in relative coordinates, relative position and heading, with the evader as the controller and the pursuer as the disturbance. Each has a bounded turn rate and a collision radius of 0.25. We compute the ground truth from a level-set solver on an $8 1 ^ { 3 }$ grid. Air6D is the same scenario in absolute coordinates with a six dimensional state, where the ground truth can be obtained from the three-dimensional game. Multi-vehicle 9D [3] is a non adversarial nine dimensional scenario with three cooperating Dubins car vehicles trying to avoid pairwise collisions. Quadrotor 13D [15], the highest dimensional system we evaluate on, is the DeepReach-MPC benchmark that models a non-adversarial single thirteen dimensional quadrotor with position, unit quaternion, velocity, and angular rate with the goal of avoiding the cylinder obstacle. For the Multi-vehicle 9D and Quadrotor 13D systems, the computational burden precludes the computation of a ground-truth solution.

## B. Models

We compare against the baselines DeepReach and MADR both with the ExactBC L1 loss variant. We also introduce ConDR-MADR as a composition of ConDR and MADR which combines the MADR MPC training signal with our ConDR loss. Both baselines are run from the released MADR implementation [6].

## C. Training

All the models use a three-layer SIREN network with 512 neurons per layer, the Adam optimizer with a learning rate of $2 { \cdot } 1 0 ^ { - 5 }$ , and the DeepReach’s curriculum over time-to-go, with 65,536 points per iteration (65,000 for the quadrotor). Air3D and Air6D train for 15k epochs with a 2k-epoch pretraining phase at $\tau = 0$ and a 10k curriculum stage, multi-vehicle 9D, for 60k epochs with a 50k curriculum, and quadrotor 13D for 104k epochs with a 1k pretraining phase. ConDR uses the safety margin $\epsilon = 0 . 0 2$ on Air3D, Air6D, and multi-vehicle 9D and $\epsilon = 0 . 0 8 5$ on the quadrotor. The loss weights are set to $\mathrm { w _ { d i f f } = 1 , w _ { d e g } = 1 . 2 , w _ { o p t } = 0 . 1 5 }$ for every experiment. The MPC loss label uses 500 states per refinement stage on Air3D, Air6D, and multi-vehicle 9D, and 5k on the quadrotor. Air3D is trained across 5 seeds while every other system is trained across 3 seeds. Every statistic reported is the mean ± the standard deviation (where shown) across these seeds.

## V. RESULTS

## A. Ground Truth Comparisons

We evaluate 395,678 Monte Carlo samples for Air6D and turn the absolute coordinates back into relative Air3D coordinates. All metrics are computed at $\tau = T$ . We report the Intersection over Union (IoU) between the learned and ground truth BRT, where IoU measures the overlap ratio and is given by $\begin{array} { r } { \mathrm { I o U } ( \tau ) = \frac { | \tilde { B } ( \tau ) \cap B ( \tau ) | } { | \tilde { B } ( \tau ) \cup B ( \tau ) | . } } \end{array}$ where $\tilde { B } ( \tau )$ denotes the predicted BRT and $B ( \tau )$ denotes the ground-truth BRT. We also report the FN and FP rates, the fraction of states where $V _ { \theta } \leq V _ { \mathrm { t r u e } } ,$ and the largest violation of the obstacle branch $V _ { \theta } - g \leq 0 .$

Table I shows that across both systems, ConDR achieves the lowest FN and the highest FP of all models as an overapproximation should, since it prefers to classify safe states as unsafe rather than the reverse. On Air3D, it has zero FNs on four out of five seeds with one seed as an outlier. Additionally, ConDR satisfies the subsolution property $V _ { \theta } \leq V _ { \mathrm { t r u e } }$ at 96% of states on Air3D and 99% on Air6D, against the baselines which show the wrong sign roughly half the time. ConDR also always satisfies the obstacle branch $V _ { \theta } - g \le 0$ , whereas DeepReach and MADR violate it by up to 0.045. DeepReach shows a slightly higher IoU than ConDR because IoU counts misclassification of safe and unsafe states equally. ConDR-MADR beats MADR on nearly every metric, but satisfies $V _ { \theta } \leq$ $V _ { \mathrm { t r u e } }$ at a lower rate than ConDR due to the MPC loss pulling away from the pure residual loss. In Fig. 1, we show the BRTs at the headings where the baselines miss most. In both Air3D and Air6D, ConDR shows a clear over-approximation of the BRT, whereas the other models show misclassifications on the BRT boundary.

![](images/b5edca4de0009af3ba7c135d5eb34083366308266904146037104c6774052514.jpg)  
Fig. 1. BRT classification at the slices where the baselines miss most. Top two rows: Air3D heading slices. Bottom two rows: Air6D pursuer-position slices with the evader fixed (triangle). Gray: true positives. Red: unsafe states called safe. Blue: safe states called unsafe. Black: true BRT boundary.

## B. Sampled Residual Bounds

For systems without ground truth, we estimate a residual bound using the distribution-free conformal bound [18] that Lin and Bansal [11] apply to rollouts and instead apply it to the VI residual. We sample $n ~ = ~ 2 0 0 , 0 0 0 ~ ( \tau , x )$ pairs uniformly from $[ 0 , T ] \times \mathcal { X }$ and compute the residual violation $v = { \mathrm { r e l u } } ( \operatorname* { m a x } ( r _ { \theta } , V _ { \theta } - g ) )$ which is zero if both inequalities are true. We set δ to be the violation at rank 198,137 so that it is slightly past the 99th percentile and with 0.999 confidence, this guarantees the violations be $\leq \delta$ on 99% of the sampling distribution. Although, this uses finite samples and does not hold globally as Corollary 1 requires, we can still use Corollary 1 to obtain a shifted tube $\{ x \in \mathbb { R } ^ { n } : V _ { \theta } \leq \delta T \}$ rather than a formal certificate for ConDR and ConDR-MADR. For DeepReach and MADR, whose obstacle branch is not satisfied by construction, but bounded by δ, we can use the same argument $W = V _ { \theta } - \delta \tau - \delta$ to give the level $\delta ( T + 1 )$

Table II shows the fraction of samples where relu $\iota \big ( \operatorname* { m a x } ( r _ { \theta } , V _ { \theta } - g ) \big ) \ > \ 0 .$ , the bound δ, and the volume of the shifted BRT as a fraction of the state space. ConDR violates the subsolution property on 1% to 3% of sampled states against DeepReach and MADR who violate on 25% to 52% of states. ConDR also reports the smallest shifted-tube volume on every system. The certificate can also be checked against Air3D’s ground truth where ConDR needs zero shift on four out of five seeds since its learned tube already covers 99% of the true BRT without correction.

TABLE I  
GROUND-TRUTH METRICS AT $\tau = T$ ON AIR3D (5 SEEDS) AND AIR6D (3 SEEDS), MEAN ± STD. FN: UNSAFE STATES CALLED SAFE. FP: SAFE STATES CALLED UNSAFE. $\operatorname { F R A C } ( V _ { \theta } \leq V _ { \operatorname { T R U E } } )$ : CONTAINMENT. max $( V _ { \theta } - g )$ : WORST OBSTACLE-BRANCH VIOLATION.
<table><tr><td></td><td>System Model</td><td>IoU ↑</td><td>FN↓</td><td>FP↓</td><td> ${ \mathrm { f r a c } } ( V _ { \theta } \leq V _ { \mathrm { t r u e } } )$ </td><td> $\operatorname* { m a x } ( V _ { \theta } - g ) \downarrow$ </td></tr><tr><td rowspan="4">Air3D</td><td>DeepReach</td><td> $0 . 9 3 9 \pm 0 . 0 0 8$ </td><td> $0 . 0 0 2 2 \pm 0 . 0 0 0 4$ </td><td> $0 . 0 0 5 9 \pm 0 . 0 0 0 8$ </td><td> $0 . 6 5 9 \pm 0 . 0 1 6$ </td><td>+0.018</td></tr><tr><td>ConDR</td><td> $0 . 9 2 9 \pm 0 . 0 0 4$ </td><td> $\mathbf { 0 . 0 0 0 9 \pm 0 . 0 0 1 8 }$ </td><td> $0 . 0 0 7 1 \pm 0 . 0 0 0 6$ </td><td> $\mathbf { 0 . 9 5 9 \pm 0 . 0 0 5 }$ </td><td>-0.005</td></tr><tr><td>MADR</td><td> $0 . 9 5 2 \pm 0 . 0 0 4$ </td><td> $0 . 0 2 6 6 \pm 0 . 0 0 3 5$ </td><td> $0 . 0 0 2 1 \pm 0 . 0 0 0 1$ </td><td> $0 . 4 4 0 \pm 0 . 0 4 3$ </td><td>+0.027</td></tr><tr><td>ConDR-MADR</td><td> $\mathbf { 0 . 9 6 3 \pm 0 . 0 0 2 }$ </td><td> $0 . 0 1 7 2 \pm 0 . 0 0 1 3$ </td><td> $\mathbf { 0 . 0 0 2 0 \pm 0 . 0 0 0 2 }$ </td><td> $0 . 7 4 0 \pm 0 . 0 2 8$ </td><td>-0.005</td></tr><tr><td rowspan="4">Air6D</td><td>DeepReach</td><td> $\mathbf { 0 . 9 5 0 \pm 0 . 0 0 2 }$ </td><td> $0 . 0 1 2 3 \pm 0 . 0 0 2 4$ </td><td> $\mathbf { 0 . 0 0 3 2 \pm 0 . 0 0 0 3 }$ </td><td> $0 . 4 6 6 \pm 0 . 0 9 6$ </td><td>+0.028</td></tr><tr><td>ConDR</td><td> $0 . 9 2 4 \pm 0 . 0 0 4$ </td><td> $\mathbf { 0 . 0 0 0 6 \pm 0 . 0 0 0 1 }$ </td><td> $0 . 0 0 6 5 \pm 0 . 0 0 0 4$ </td><td> $\mathbf { 0 . 9 9 4 \pm 0 . 0 0 1 }$ </td><td>-0.002</td></tr><tr><td>MADR</td><td> $0 . 9 3 3 \pm 0 . 0 0 6$ </td><td> $0 . 0 1 9 6 \pm 0 . 0 0 0 3$ </td><td> $0 . 0 0 4 0 \pm 0 . 0 0 0 5$ </td><td> $0 . 5 2 9 \pm 0 . 0 0 8$ </td><td>+0.045</td></tr><tr><td>ConDR-MADR</td><td> $0 . 9 3 6 \pm 0 . 0 0 3$ </td><td> $0 . 0 0 2 3 \pm 0 . 0 0 0 5$ </td><td> $0 . 0 0 5 2 \pm 0 . 0 0 0 3$ </td><td> $0 . 9 3 0 \pm 0 . 0 0 1$ </td><td>-0.004</td></tr></table>

TABLE II

SAMPLED RESIDUAL BOUND (n = 200K, η = 0.01, β = 0.001). VIOLATION: FRACTION OF SAMPLES VIOLATING A SUBSOLUTION INEQUALITY. $\delta _ { \mathrm { S U B } } \colon$ BOUND ON THE VIOLATION HOLDING ON A 1 − η FRACTION OF THE DOMAIN WITH CONFIDENCE 1 − β. SHIFTED VOL.: FRACTION OF THE BOX IN $\{ V _ { \theta } \leq \delta T \}$ FOR CONDR AND CONDR-MADR (EXACT OBSTACLE BRANCH) AND $\left\{ V _ { \theta } \leq \delta ( T + 1 ) \right\}$ FOR DEEPREACH AND MADR (OBSTACLE BRANCH VIOLATED UP TO δ).
<table><tr><td></td><td>DeepReach</td><td></td><td>ConDR MADR</td><td>ConDR- MADR</td></tr><tr><td>AIiD Violation↓</td><td>0.25</td><td>0.01</td><td>0.49</td><td>0.11</td></tr><tr><td> $\delta _ { \mathrm { s u b } }$  ↓</td><td>0.018</td><td>0.004</td><td>0.056</td><td>0.096</td></tr><tr><td>Shift. vol. ↓</td><td>0.116</td><td>0.097</td><td>0.162</td><td>0.152</td></tr><tr><td>AIiD Violation↓</td><td>0.52</td><td>0.03</td><td>0.50</td><td>0.11</td></tr><tr><td> $\delta _ { \mathrm { s u b } } ~ .$  →</td><td>0.056</td><td>0.033</td><td>0.158</td><td>0.144</td></tr><tr><td>Shift. vol. ↓</td><td>0.129</td><td>0.094</td><td>0.251</td><td>0.149</td></tr><tr><td>Violation ↓ 90</td><td>0.49</td><td>0.03</td><td>0.48</td><td>0.07</td></tr><tr><td> $\delta _ { \mathrm { s u b } } \downarrow$ </td><td>0.083</td><td>0.054</td><td>0.128</td><td>0.109</td></tr><tr><td>Shift. vol. ↓</td><td>0.379</td><td>0.247</td><td>0.491</td><td>0.314</td></tr><tr><td>Violation↓ 13D</td><td>0.49</td><td>0.03</td><td>0.48</td><td>0.09</td></tr><tr><td> $\delta _ { \mathrm { s u b } } \downarrow$ </td><td>0.231</td><td>0.158</td><td>0.327</td><td>0.316</td></tr><tr><td>Shift. vol. ↓</td><td>0.124</td><td>0.071</td><td>0.170</td><td>0.097</td></tr></table>

## C. Closed-loop Safety Filters

We deploy each learned value function as least-restrictive safety filters where the nominal controller acts unless $V _ { \theta } \leq 0 ,$ in which case the optimal control extracted from ∇V overrides it. For Air3D, Air6D, and multi-vehicle 9D, the nominal controllers set their turn rate to zero and fly straight. For Quadrotor 13D, the nominal controller flies at the same initial velocity and orientation. The adversary for Air3D and Air6D takes two forms: pure pursuit, a static policy of orienting towards the evader at max turn rate, or as a learned adversary which is the worst-case disturbance computed from a fixed DeepReach model and applied to all models. Multi-vehicle 9D and Quadrotor 13D do not have adversaries. We use two starting state procedures. On Air3D and Air6D, where ground truth exists, we sample verified safe states as the start. On all four systems, we sample starting states at a distance from the unsafe set that are unverified as safe. For each procedure, we sample 400 starting states per seed, shared across each method and report the collision and intervention rate.

In Table III, ConDR achieves the lowest, sometimes zero, collision rate across all the models, while ConDR-MADR beats MADR in five cases, ties in one, and is only worse by 0.005 in the other two cases. Neither ConDR nor ConDR-MADR has to increase the intervention rate in order to lower their collision rate. In every column, they intervene at the same rate or by no more than 0.005 more with the exception of the quadrotor which intervenes on 1.4% of the steps against DeepReach’s 0.6%. In Fig. 2, we show example trajectories for each system from the unverified distance start state procedure. We choose starts where the nominal controller collides on most of the trajectories.

TABLE III  
CLOSED-LOOP SAFETY FILTERING: COLLISION RATE (MEAN ± STD) / INTERVENTION FRACTION, 400 STARTS PER SEED. TOP: VERIFIED-SAFE STARTS ON AIR3D AND AIR6D UNDER PURE-PURSUIT AND LEARNED ADVERSARIES. BOTTOM: UNVERIFIED STARTS ON ALL FOUR SYSTEMS. BOLD: LOWEST COLLISION RATE.
<table><tr><td colspan="4">Verified-safe starts</td></tr><tr><td></td><td colspan="2">Air3D</td><td colspan="2">Air6D</td></tr><tr><td>Method</td><td>pursuit</td><td>learned</td><td>pursuit</td><td>learned</td></tr><tr><td>No filter</td><td>.178</td><td>.215</td><td>.140</td><td>.160 .046±.005 .040±.042 .034±.008 .098±.008</td></tr><tr><td>DeepReach</td><td>1.054</td><td>1.074</td><td>1.038</td><td>1.060</td></tr><tr><td>ConDR</td><td>1.055</td><td>1.069</td><td>.000±.000 .004±.005 .021±.005 .078±.010 1.037</td><td>1.062</td></tr><tr><td>MADR</td><td>1.053</td><td>1.073</td><td>.090±.017 .172±.010 .048±.006 .113±.015 1.037</td><td>1.060</td></tr><tr><td>ConDR-MADR</td><td>1.058</td><td>1.073</td><td>.077±.002 .153±.018 .043±.000 .118±.006 1.038</td><td>1.059</td></tr><tr><td>Unverified starts Method</td><td>Air3D</td><td>Air6D</td><td>9D</td><td>13D</td></tr><tr><td>No filter</td><td>.265</td><td>.182</td><td>.155</td><td>.085</td></tr><tr><td>DeepReach</td><td>1.094</td><td>1.069</td><td>.083±.043 .118±.008 .126±.005 .081±.003 1.052</td><td>1.006</td></tr><tr><td>ConDR</td><td>1.088</td><td>1.071</td><td>.047±.005 .096±.010 .042±.003 .052±.007 1.052</td><td>1.014</td></tr><tr><td>MADR</td><td>1.093</td><td>1.070</td><td>.219±.014 .132±.017 .122±.003 .079±.001 1.052</td><td>/.012</td></tr><tr><td>ConDR-MADR</td><td>7.093</td><td>1.067</td><td>.201±.018 .137±.007 .073±.012 .079±.001 1.053</td><td>/.013</td></tr></table>

## D. Model Mismatch

We repeat the distance margin safety filter experiment with the dynamics perturbed to evaluate ConDR’s robustness to model mismatch M. We scale the pursuer’s speed by $1 + M$ and the evader’s by $1 \ : - \ : M$ , while the filter and learned adversary still use the nominal model. We retrain ConDR for each $\epsilon \in \lbrace 0 , 0 . 0 2 , 0 . 0 5 , 0 . 1 0 \rbrace$ (three seeds each, five at $\epsilon = 0 . 0 2 )$

In Fig. 3, the left column shows that each model maintains its collision rate until the model mismatch roughly equals the safety margin ϵ and degrades after, showing consistency with Theorem 2. The $\epsilon = 0$ model fails almost as fast as DeepReach. The right column shows that the IoU falls roughly linearly in ϵ. The certificate violation fraction, where the model violates the subsolution inequality, shrinks as ϵ increases.

## E. Loss Ablation Study

In Table IV, we perform a loss ablation on ConDR and compare it to DeepReach. We add each loss component of ConDR to DeepReach one at a time and then remove them one at a time to test the effects of each term. The obstacle branch construction raises the subsolution property satisfaction from 66% to 91%, halves collisions, and improves IoU. As predicted by Section III-C, without the $\mathcal { L } _ { \mathrm { d e g } }$ loss, the model converges to the degenerate solution of classifying everything as unsafe. The $L _ { \mathrm { d e g } }$ prevents the degenerate solution, while $L _ { \mathrm { o p t } }$ increases the IoU and removes the last collisions. Adding the safety margin ϵ reduces the subsolution violations from 17%-26% to 0.4%-1.2%. Without $L _ { \mathrm { d e g } } ,$ but with $\mathcal { L } _ { \mathrm { o p t } }$ and the margin, the residual stays near the safety margin, but loses IoU.

## F. Volume at Matched Safety

To evaluate whether ConDR’s low FN rate is simply a consequence of trivially overapproximating the learned BRT, we slide the level-set threshold c and take the resulting BRT $\{ x \ \in \ \mathbb { R } ^ { n } \ : \ V _ { \theta } \ \leq \ c \}$ then measure the volume of each model where volume is the fraction of the state space occupied by the learned BRT. If ConDR and ConDR-MADR were just DeepReach with a conservative offset, then their volumes would be equal at matched FNs. However, Table V shows that for Air3D, ConDR’s and ConDR-MADR’s volume is smaller than DeepReach/MADR and on Air6D is within 0.0003. Therefore, our method’s conservatism is a result of where the loss places the BRT boundary and not simply from over-approximation.

![](images/43d5e94622f5235f4c7c9639237fd8edd6e649dc06a8895f09d266ece401ade7.jpg)  
Fig. 2. Rollouts from unverified starts: multi-vehicle 9D (8 starts), quadrotor 13D (16 starts). Red: collision

![](images/a0b1496de85d0dde88d4968b3a74d594e2e300d216c510522e95d85c02613198.jpg)

![](images/6661e1c4716ca5f9d4cbfd8127ac0becaa89d246c593858e478bcb36debafb84.jpg)

![](images/6942ceb1c6d32b5945898dbd7f77696525cae0dea9f52d00667c47952d04c12d.jpg)

![](images/105a784e03993fab5fe565c5f1ed8eea4a016d256ff844689ee0c3f7b12350a6.jpg)  
Fig. 3. Air3D and Air6D. Left: collision rate versus mismatch M; each margin ϵ holds until $M \approx \epsilon .$ . Right: IoU (blue) and certificate violation (red, log) versus ϵ.

We also evaluate the effect of sliding c in the closedloop safety filter experiment with verified safe starts. We measure the resulting collision and intervention rates for each model in Air3D and Air6D. Fig. 4 shows that ConDR has the fewest collisions at every c on both systems with equal or lower intervention. On Air3D, ConDR has a zerocollision rate, whereas the baselines only achieve this with $c \geq 0 . 0 3$ and must intervene 60%-80% more than ConDR at $c = 0 .$ . Therefore, shifting DeepReach/MADR after training can achieve ConDR’s level of safety but only at the cost of more interventions.

![](images/0adc49910713c376281f4e90d2ec50aa83df8ab0a9c677f074d9de852663ba57.jpg)

![](images/69600af481a81b58aca778a6e29b6a3742202784847041236cb53c288771854d.jpg)

![](images/cc85aa2218a94de074c4539e4a199182de5a948241b578e82e4d77fe7e6a1b67.jpg)

![](images/aa40081cf88f974f8f01bab448d2c314a93dd90414534021e19c73d5e8b190f8.jpg)  
Fig. 4. Collision rate (left) and intervention (right) as the filter level c is raised from 0 on verified-safe starts; dotted line at $c = \epsilon .$

TABLE IV  
LOSS ABLATION ON AIR3D (5 SEEDS). CHECK MARKS SHOW ACTIVE COMPONENTS; ROWS WITHOUT ${ \mathcal { L } } _ { \mathrm { d i f f } }$ USE THE TWO-SIDED RESIDUAL. <sup>†</sup>DEGENERATE SOLUTION.
<table><tr><td>Model</td><td>Obstacle construction</td><td> ${ \mathcal { L } } _ { \mathrm { d i f f } }$ </td><td> ${ \mathcal { L } } _ { \mathrm { d e g } }$ </td><td> $\mathcal { L } _ { \mathrm { o p t } }$ </td><td>ε</td><td>IoU ↑</td><td>FN↓</td><td>FP↓</td><td> $\operatorname { f r a c } ( V _ { \theta } \leq V _ { \mathrm { t r u e } } ) \uparrow$ </td><td>Subsol. viol. ↓</td><td>Collisions↓</td></tr><tr><td>DeepReach</td><td>I</td><td>I</td><td></td><td></td><td></td><td>0.939</td><td>0.0022</td><td>0.0059</td><td>0.659</td><td>0.253</td><td> $0 . 0 4 6 \pm 0 . 0 0 5$ </td></tr><tr><td></td><td>√</td><td>一</td><td></td><td></td><td>I</td><td>0.963</td><td>0.0014</td><td>0.0035</td><td>0.907</td><td>0.234</td><td> $0 . 0 1 9 \pm 0 . 0 0 6$ </td></tr><tr><td>†</td><td>√</td><td>√</td><td>一</td><td>一</td><td>0</td><td>0.086</td><td>0.0000</td><td>1.0000</td><td>1.000</td><td>0.000</td><td> $0 . 3 4 8 \pm 0 . 0 2 4$ </td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td>一</td><td>0</td><td>0.919</td><td>0.0002</td><td>0.0083</td><td>0.924</td><td>0.170</td><td> $0 . 0 2 2 \pm 0 . 0 0 5$ </td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>0</td><td>0.962</td><td>0.0021</td><td>0.0035</td><td>0.937</td><td>0.256</td><td> $0 . 0 1 3 \pm 0 . 0 0 5$ </td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td>一</td><td>0.02</td><td>0.898</td><td>0.0014</td><td>0.0106</td><td>0.949</td><td>0.004</td><td> $0 . 0 0 4 \pm 0 . 0 0 3$ </td></tr><tr><td></td><td>√</td><td>√</td><td>一</td><td>√</td><td>0.02</td><td>0.707</td><td>0.0006</td><td>0.0392</td><td>0.977</td><td>0.006</td><td> $\mathbf { 0 . 0 0 0 \ : \pm { \ : 0 . 0 0 0 } }$ </td></tr><tr><td></td><td>一</td><td>√</td><td>√</td><td>√</td><td>0.02</td><td>0.924</td><td>0.0107</td><td>0.0068</td><td>0.731</td><td>0.172</td><td> $0 . 0 9 0 \pm 0 . 0 1 9$ </td></tr><tr><td>ConDR</td><td>√</td><td>√</td><td>√</td><td>√</td><td>0.02</td><td>0.929</td><td>0.0009</td><td>0.0071</td><td>0.959</td><td>0.012</td><td> $\mathbf { 0 . 0 0 0 \ : \pm { \ : 0 . 0 0 0 } }$ </td></tr></table>

TABLE V

VOLUME OF $\{ V _ { \theta } \leq c \}$ AFTER SETTING c TO THE SMALLEST VALUE MEETING EACH FN TARGET, MEAN ± STD OVER SEEDS (AIR3D: 5, AIR6D: 3). BOLD: SMALLEST IN COLUMN.
<table><tr><td>Model</td><td> $\mathrm { F N } \le 0 . 0 5$ </td><td> $\mathrm { F N } \le 0 . 0 3$ </td><td> $\mathrm { F N } \le 0 . 0 1$ </td></tr><tr><td colspan="4">Air3D (exact BRT volume 0.0863)</td></tr><tr><td>DeepReach</td><td> $0 . 0 8 5 4 { \pm } 0 . 0 0 0 8$ </td><td> $0 . 0 8 7 6 { \scriptstyle \pm 0 . 0 0 0 8 }$ </td><td> $0 . 0 9 0 0 { \scriptstyle \pm 0 . 0 0 0 8 }$ </td></tr><tr><td>ConDR</td><td> $0 . 0 8 2 7 { \pm } 0 . 0 0 0 3$ </td><td> $0 . 0 8 5 0 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $\mathbf { 0 . 0 8 8 2 } \pm \mathbf { 0 . 0 0 0 3 }$ </td></tr><tr><td>MADR</td><td> $0 . 0 8 3 5 { \pm } 0 . 0 0 0 1$ </td><td> $0 . 0 8 5 6 { \pm } 0 . 0 0 0 2$ </td><td> $0 . 0 8 8 6 { \pm } 0 . 0 0 0 3$ </td></tr><tr><td>ConDR-MADR</td><td> $\mathbf { 0 . 0 8 2 2 { \scriptstyle \pm 0 . 0 0 0 1 } }$ </td><td> $\mathbf { 0 . 0 8 4 4 { \scriptstyle \pm 0 . 0 0 0 2 } }$ </td><td> $0 . 0 8 8 4 { \pm } 0 . 0 0 0 2$ </td></tr><tr><td colspan="4">Air6D (exact BRT volume 0.0736)</td></tr><tr><td>DeepReach</td><td> $0 . 0 7 1 6 { \pm } 0 . 0 0 0 1$ </td><td> $0 . 0 7 3 4 { \pm } 0 . 0 0 0 2$ </td><td> $0 . 0 7 6 0 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td></tr><tr><td>ConDR</td><td> $0 . 0 7 1 6 { \pm } 0 . 0 0 0 3$ </td><td> $0 . 0 7 3 6 { \pm } 0 . 0 0 0 3$ </td><td> $0 . 0 7 6 3 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td></tr><tr><td>MADR</td><td> $0 . 0 7 2 0 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $0 . 0 7 4 3 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $0 . 0 7 7 9 { \scriptstyle \pm 0 . 0 0 0 6 }$ </td></tr><tr><td colspan="4">ConDR-MADR  $\mathbf { 0 . 0 7 1 1 { \pm } 0 . 0 0 0 1 }$   $\mathbf { 0 . 0 7 3 0 { \overset { . } { \bot } } 0 . 0 0 0 2 }$ </td></tr></table>

## VI. CONCLUSION

We introduced ConDR that trains toward the solution of the margin-shifted VI. The solution itself is a subsolution of the original VI so its BRT contains the true BRT and is within ϵτ of the true value function. Across Air3D, Air6D, multivehicle 9D, and quadrotor 13D, ConDR achieves the lowest FN rates, the lowest closed-loop collision rates, and the fewest subsolution violations, all while having the tightest shifted BRT volume. Composing with MADR improves MADR’s FN, violation fraction, and shifted tube volume. It also lowers its closed-loop collision rates in five out of eight settings. While ConDR can only approximate the exact solution of the marginshifted VI, the guarantees of Theorem 1 hold for the exact solution. For the trained network we can compute the residual discrepancy with a high probability bound. Finally, ConDR is especially beneficial for higher dimensional systems whose ground truth cannot be computed and is simple to integrate into existing DeepReach applications.

## ACKNOWLEDGMENTS

Claude was used to assist with code for the experiments. All results were verified by the authors.

## REFERENCES

[1] S. Bansal, M. Chen, S. Herbert, and C. J. Tomlin, “Hamilton-jacobi reachability: A brief overview and recent advances,” in 2017 IEEE 56th Annual Conference on Decision and Control (CDC), 2017, pp. 2242– 2253.

[2] M. Chen, S. Herbert, and C. Tomlin, “Exact and efficient hamilton-jacobi guaranteed safety analysis via system decomposition,” 05 2017.

[3] S. Bansal and C. J. Tomlin, “Deepreach: A deep learning approach to high-dimensional reachability,” in 2021 IEEE International Conference on Robotics and Automation (ICRA), 2021, pp. 1817–1824.

[4] V. Sitzmann, J. N. Martel, A. W. Bergman, D. B. Lindell, and G. Wetzstein, “Implicit neural representations with periodic activation functions,” in arXiv, 2020.

[5] A. Singh, Z. Feng, and S. Bansal, “Exact imposition of safety boundary conditions in neural reachable tubes,” 2025.

[6] R. Teoh, S. Tonkens, W. Sharpless, A. Yang, Z. Feng, S. Bansal, and S. Herbert, “Madr: Mpc-guided adversarial deepreach,” 10 2025.

[7] J. F. Fisac, N. F. Lugovoy, V. Rubies-Royo, S. Ghosh, and C. J. Tomlin, “Bridging hamilton-jacobi safety analysis and reinforcement learning,” in 2019 International Conference on Robotics and Automation (ICRA), 2019, pp. 8550–8556.

[8] K.-C. Hsu, H. Hu, and J. F. Fisac, “The safety filter: A unified view of safety-critical control in autonomous systems,” 2023.

[9] A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada, “Control barrier functions: Theory and applications,” in 2019 18th European Control Conference (ECC), 2019, pp. 3420–3431.

[10] W. Hofgard, “Convergence guarantees for neural network-based hamilton-jacobi reachability,” 2024.

[11] A. Lin and S. Bansal, “Verification of neural reachable tubes via scenario optimization and conformal prediction,” 2024.

[12] J. Li, D. Lee, J. Lee, K. S. Dong, S. Sojoudi, and C. Tomlin, “Certifiable reachability learning using a new lipschitz continuous value function,” 2025.

[13] J. F. Fisac, M. Chen, C. J. Tomlin, and S. S. Sastry, “Reach-avoid problems with time-varying dynamics, targets and constraints,” 2014.

[14] M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” Journal of Computational Physics, vol. 378, pp. 686–707, 2019.

[15] Z. Feng, L. Qiu, and S. Bansal, “Bridging model predictive control and deep learning for scalable reachability analysis,” 2025.

[16] M. Bardi and I. Capuzzo-Dolcetta, Optimal Control and Viscosity Solutions ofHamilton–Jacobi–Bellman Equations. Boston, MA: Birkhauser,¨ 1997.

[17] I. Mitchell, A. Bayen, and C. Tomlin, “A time-dependent hamilton-jacobi formulation of reachable sets for continuous dynamic games,” IEEE Transactions on Automatic Control, vol. 50, no. 7, pp. 947–957, 2005.

[18] V. Vovk, “Conditional validity of inductive conformal predictors,” in Proceedings of the Asian Conference on Machine Learning, ser. Proceedings of Machine Learning Research, S. C. H. Hoi and W. Buntine, Eds., vol. 25. Singapore Management University, Singapore: PMLR, 04–06 Nov 2012, pp. 475–490.