# Adaptive Hybrid Subspace Levenberg–Marquardt Algorithm with Adequacy Monitor for Large-Scale Least Squares Problems

M. Duc Hoang & Timothy J. Lewis

Department of Mathematics, University of California, Davis, Davis, CA, USA

## ABSTRACT

The Levenberg–Marquardt (LM) algorithm is the most widely used method for solving nonlinear least-squares problems, as i combines the robustness of steepest descent with the fast local convergence of the Gauss-Newton method. However, its computational cost can become prohibitive for large-scale problems because each iteration requires solving a large damped linear system, and conventional step-acceptance strategies may require repeated solves as the damping parameter is adjusted. Despite this computational challenge, many large-scale least-squares problems exhibit effective low-dimensional structure, with only a small number of parameter-space directions strongly informed by the data. We propose an adaptive hybrid subspace Levenberg–Marquardt (HSLM) algorithm that constructs a low-dimensional subspace from complementary sources of gradient, memory, Krylov-subspace, and randomized curvature information and computes a spectrally damped LM step within this subspace. A distinguishing feature of the method is a deterministic adequacy monitor that quantifies how much descent information is captured by the reduced space and adaptively enriches the subspace when necessary. Step acceptance is decoupled from damping adjustment: Armijo backtracking determines the accepted step length, while the ratio of actual to predicted reduction is used solely to update the damping parameter, thereby avoiding repeated damped-system solves during step acceptance. For the HSLM algorithm, we establish global convergence to stationarity and prove local linear and superlinear convergence. Numerical experiments on neural-network training problems show that HSLM achieves convergence behavior comparable to classical and Krylov-subspace LM (KSLM) while substantially reducing per-iteration computationa cost, with increasing advantages observed as the parameter dimension grows.

Keywords: Levenberg-Marquardt, large-scale least squares problem, hybrid subspace, Adequacy Monitor

## 1 INTRODUCTION

Large-scale nonlinear least-squares problems arise in many areas, including physics, engineering, inverse modeling, and machine learning. Second-order methods, such as Levenberg–Marquardt (LM) algorithm, are attractive because of their robustness and fast local convergence, especially for ill-conditioned nonlinear least-squares problems [18, 21, 20]. However, their direct application becomes computationally expensive in high-dimensional problems, since each iteration requires solv ing large linear systems involving the Jacobian or an approximate Hessian [24, 11]. At the same time, many inverse problems possess intrinsic low-dimensional structure, in the sense that the data constrain only a relatively small subspace of the full parameter space, while directions outside this subspace are weakly identifiable or dominated by regularization. This observation is consistent with active-subspace ideas, which identify important directions along which the model or objective varies most strongly [8, 9]. This structure motivates reduced-space and low-rank variants of LM, which seek to approximate the full problem within a lower-dimensional subspace that captures the dominant data-informed and curvature directions.

Subspace and low-rank approximations have long been studied as a way to reduce the cost of LM-type methods for largescale inverse and nonlinear least-squares problems. These approaches include modified LM methods for large-scale inverse problems [27], general subspace frameworks for nonlinear equations and nonlinear least squares [29], randomized truncated SVD approximations of sensitivity matrices [4], and Krylov-based projection methods for highly parameterized inverse models [19]. In parallel, random embedding and randomized subspace optimization methods have been developed to reduce the dimension of the variable space itself [6, 25]. Related randomized techniques include stochastic Hessian approximation, as in SPAN [16], and randomized matrix-decomposition methods for identifying dominant low-rank subspaces [14]. Together, these works show that subspace approximations provide a natural route toward scalable LM-type methods by reducing the cost of each iteration while retaining important curvature and data-informed directions.

Despite these advances, most existing subspace methods lack an explicit deterministic mechanism to ensure that the chosen subspace remains adequate as the optimization progresses and local curvature directions change. Randomized subspace methods, for example, rely on probabilistic embedding properties to guarantee sufficient descent or curvature capture with high probability, but they do not necessarily exploit problem-specific curvature structure or incorporate memory from previous iterations. In addition, low-rank and Krylov-based approaches can efficiently approximate individual LM steps, but the quality of the reduced space may depend strongly on how well the current subspace captures the dominant local geometry. As a result, achieving both robust global convergence and fast local convergence within a scalable, matrix-free framework remains challenging.

In this work, we propose a hybrid subspace Levenberg-Marquardt (HSLM) algorithm that integrates ideas from spectral truncation, Krylov subspace methods, and randomized subspace methods. At each iteration, the method constructs a compact subspace that explicitly incorporates descent information via the gradient, previous accepted steps, and curvature information obtained from Hessian or Jacobian-vector probes. A projected-gradient adequacy measure is employed to monitor descent-relevant information in the subspace and to adaptively expand or refine it when necessary. To avoid repeatedly solving expensive damped systems, we also use Armijo backtracking line search for step acceptance and rely exclusively on the predicted-versus-actual reduction ratio to update the damping parameter. Under standard smoothness, regularity, and vanishing-approximation assumptions, we establish global convergence of the HSLM algorithm, with global convergence to stationarity and local linear/superlinear convergence rates. Finally, we implement the HSLM algorithm to train feedforward neural networks of varying problem sizes and compare its performance with the classical LM and Krylov subspace LM

algorithms.

## 2 BACKGROUND AND NOTATION

Nonlinear least-squares problems arise widely in parameter estimation, scientific computing, engineering, and machine learning, and are commonly formulated as

$$
\operatorname* { m i n } _ { x \in \mathbf { R } ^ { n } } \mathbf { F } ( x ) : = \operatorname* { m i n } _ { x \in \mathbf { R } ^ { n } } { \frac { 1 } { 2 } } \| \mathbf { r } ( x ) \| ^ { 2 } ,
$$

where $\mathbf { r } ( x ) : \mathbf { R } ^ { n }  \mathbf { R } ^ { m }$ is a vector of the residual. The Gauss–Newton method [11] solves nonlinear least-squares problems by linearizing the residual function at each step. At iteration k, the residual is approximated by $\mathbf { r } ( x _ { k } + s _ { k } ) \approx \mathbf { r } ( x _ { k } ) + \mathbf { J } ( x _ { k } ) s _ { k }$ leading to a linear least-squares subproblem where the update direction $s _ { k }$ satisfies the equations

$$
\left( \mathbf { J } ( x _ { k } ) ^ { \top } \mathbf { J } ( x _ { k } ) \right) s _ { k } = - \mathbf { J } ( x _ { k } ) ^ { \top } \mathbf { r } ( x _ { k } )
$$

where $\mathbf { J } ( x ) \in \mathbf { R } ^ { m \times n }$ is the Jacobian of the residuals and $\mathbf { H } ( x _ { k } ) \approx \mathbf { J } ( x _ { k } ) ^ { \top } \mathbf { J } ( x _ { k } )$ is the corresponding Gauss–Newton approximation to the Hessian matrix. Gauss–Newton exhibits second-order local convergence when residuals are small and the Jacobian has full rank. However, the method may fail when the Jacobian is ill-conditioned or the initial guess is far from the optimal solution. To improve robustness, Levenberg [18] and Marquardt [21] introduced a damping strategy that modifies the system as

$$
\left( \mathbf { J } ( x _ { k } ) ^ { \top } \mathbf { J } ( x _ { k } ) + \mu _ { k } \mathbf { I } _ { n } \right) s _ { k } = - \mathbf { J } ( x _ { k } ) ^ { \top } \mathbf { r } ( x _ { k } )\tag{1}
$$

where the damping term $\mu _ { k } > 0$ . In the classical Levenberg-Marquardt algorithm, the damping update strategy plays a key role in the method’s stability and efficiency. When the residual is effectively reduced, the damping parameter is decreased, bringing the algorithm closer to the Gauss-Newton direction. Conversely, when an iteration produces insufficient reduction in the residual, the damping parameter is increased, yielding a step closer to the steepest-descent direction.

$$
\mathbf { g } ( x ) = \nabla \mathbf { F } ( x ) = \mathbf { J } ( x ) ^ { \top } \mathbf { r } ( x )
$$

and denote the residual, the Jacobian matrix, the gradient and Gauss-Newton Hessian approximation at x<sub>k</sub> by

$$
r _ { k } = \mathbf { r } ( x _ { k } ) , \quad J _ { k } = \mathbf { J } ( x _ { k } ) , \quad g _ { k } = \mathbf { g } ( x _ { k } ) , \quad H _ { k } = \mathbf { H } ( x _ { k } ) .
$$

## 3 HYBRID SUBSPACE LEVENBERG MARQUART ALGORITHM

The HSLM algorithm is based on the observation that, although the optimization takes place in a high-dimensional parameter space, only a relatively small number of directions are typically needed to capture the dominant descent and curvature information. Rather than relying on a fixed reduced space, HSLM constructs an adaptive hybrid subspace at each iteration from multiple sources of information, including the gradient, recent optimization history, Krylov/Lanczos directions, and randomized curvature probes. A distinguishing feature of the method is a deterministic adequacy monitor that evaluates whether the current subspace captures sufficient descent information and adaptively enriches the subspace whenever necessary. Once an adequate subspace has been constructed, the algorithm computes a spectrally damped Levenberg–Marquardt step within the reduced space. Step acceptance is determined using Armijo backtracking, while the ratio of actual to predicted reduction is used solely to update the damping parameter, thereby avoiding repeated damped-system solves. The remainder of this section describes the subspace construction, the adequacy monitor, the reduced LM system, step acceptance and damping update.

## 3.1 Subspace construction

At each iteration, HSLM constructs an orthonormal basis $V _ { k } \in \mathbb { R } ^ { n \times p }$ from candidate directions that provide complementary information about the local optimization problem. Krylov directions provide an efficient way to incorporate local curvature information [26, 29], while randomized Hessian-vector products provide an inexpensive way to sample local curvature information without explicitly forming the Hessian [16]. Other dimension-reduction techniques include randomized subspace and embedding methods that construct lower-dimensional representations of high-dimensional optimization problems while approximately preserving relevant geometric information [6, 25], and truncated SVD approaches that approximate dominant directions of the sensitivity matrix [4]. Motivated by these complementary ideas, HSLM draws candidate and enrichment directions from four complementary sources:

• the normalized gradient, which provides first-order descent information;

• recent optimization steps, which encode local optimization history;

• Krylov/Lanczos vectors, which approximate dominant curvature directions; and

• randomized Gauss-Newton-vector probes $( J _ { k } ^ { \top } J _ { k } )$ w where w $\sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { n } )$ , which broaden spectral coverage and improve robustness when the local curvature is noisy or heterogeneous.

In this way, the method adapts the reduced space to the local geometry at each iteration while keeping the dominant linear algebra restricted to a low-dimensional subproblem.

## 3.2 Deterministic Adequacy Monitor

Although the candidate pool combines multiple complementary sources of information, there is no guarantee that the resulting reduced space adequately represents the current descent geometry. Therefore, before computing the reduced Levenberg–Marquardt step, we evaluate the quality of the subspace using a deterministic adequacy criterion. Specifically, we measure the fraction of the gradient energy captured by the current subspace through the projected-gradient energy ratio

$$
\eta _ { k } ^ { \mathrm { s u b } } : = \frac { \| V _ { k } ^ { \top } g _ { k } \| ^ { 2 } } { \| g _ { k } \| ^ { 2 } } \in [ 0 , 1 ] .\tag{2}
$$

The quantity $\eta _ { k } ^ { \mathrm { s u b } }$ takes values in [0,1], with larger values indicating that the subspace captures a greater proportion of the full gradient and therefore contains sufficient first-order descent information. If $\eta _ { k } ^ { \mathrm { s u b } }$ drops below a prescribed threshold $\eta _ { \mathrm { m i n } } \in ( 0 , 1 ]$ , additional candidate directions are generated from the pool described in Section 3.1, the basis is updated, and the adequacy test is repeated until the threshold is satisfied.

## 3.3 Reduced LM system

Once the adequacy criterion is satisfied, the reduced basis is fixed for the current iteration. The Levenberg–Marquardt step is then computed entirely within this subspace, reducing the n-dimensional damped system to a much smaller p-dimensional problem. Restricting the Jacobian to the reduced subspace gives the reduced Jacobian $J _ { k , p } : = J _ { k } V _ { k } \in \mathbb { R } ^ { m \times p }$

$$
J _ { k , p } = U _ { k } \Sigma _ { k , p } Z _ { k } ^ { \top } , \qquad \Sigma _ { k , p } = \mathrm { d i a g } ( \sigma _ { k , 1 } , \ldots , \sigma _ { k , p } ) .\tag{3}
$$

denote its singular value decomposition. This decomposition diagonalizes the reduced Gauss–Newton matrix restricted to the reduced subspace and provides a convenient basis for constructing the reduced LM system. In particular,

$$
\begin{array} { r } { V _ { k } ^ { \top } H _ { k } V _ { k } = ( J _ { k } V _ { k } ) ^ { \top } ( J _ { k } V _ { k } ) = J _ { k , p } ^ { \top } J _ { k , p } = Z _ { k } \Sigma _ { k , p } ^ { 2 } Z _ { k } ^ { \top } , } \end{array}\tag{4}
$$

and

$$
\begin{array} { r } { { V _ { k } ^ { \top } } g _ { k } = { V _ { k } ^ { \top } } ( J _ { k } ^ { \top } r _ { k } ) = ( J _ { k } V _ { k } ) ^ { \top } r _ { k } = J _ { k , p } ^ { \top } r _ { k } = Z _ { k } \Sigma _ { k , p } { U _ { k } ^ { \top } } r _ { k } . } \end{array}\tag{5}
$$

We restrict the search direction $s _ { k }$ to the reduced subspace by writing $s _ { k } = V _ { k } Z _ { k } y _ { k }$ . Using (4) and (5), the classical reduced Levenberg–Marquardt system becomes

$$
\left( \Sigma _ { k , p } ^ { 2 } + \mu _ { k } \mathbf { I } _ { p } \right) y _ { k } = - \Sigma _ { k , p } U _ { k } ^ { \top } r _ { k }
$$

This reduced formulation inherits the isotropic damping of the standard LM algorithm. However, because the singular values of the reduced Jacobian can vary substantially, treating all directions equally may lead to unnecessary damping in some directions and insufficient damping in others. To better accommodate ill-conditioning and varying local curvature, we replace the isotropic damping matrix $\mathbf { I } _ { p }$ with a diagonal, curvature-dependent damping matrix $D _ { k , p } ,$ giving

$$
B _ { k } y _ { k } = - \Sigma _ { k , p } U _ { k } ^ { \top } r _ { k } \qquad \mathrm { w h e r e } ~ B _ { k } : = \Sigma _ { k , p } ^ { 2 } + \mu _ { k } D _ { k , p } ,\tag{6}
$$

and

$$
D _ { k , p } : = \mathrm { d i a g } ( d _ { k , 1 } , \ldots , d _ { k , p } ) , \qquad d _ { k , i } : = \operatorname* { m a x } ( \sigma _ { k , i } ^ { 2 } , \delta ) .\tag{7}
$$

The threshold $\delta > 0$ guarantees that every damping coefficient remains strictly positive, preventing excessively large updates associated with very small singular values. Consequently, the amount of regularization adapts to the local curvature represented by the singular values: directions associated with larger singular values receive stronger damping, while flat

ter directions remain comparatively less damped. This curvature-adaptive damping improves numerical stability while preserving efficient progress along weakly constrained directions. The resulting reduced system (6) is therefore solved in the low-dimensional subspace, after which the corresponding search direction is mapped back to the original parameter space.

Remark 1. When the candidate subspace becomes large, its dimension may be controlled by compressing the basis using an SVD-based energy criterion. In practice, the basis can be maintained efficiently using incremental QR factorization together with an SVD of a small compressed matrix, following standard incremental SVD techniques [5].

## 3.4 Step acceptance and damping updates

After computing the reduced LM search direction $s _ { k } .$ , we determine an appropriate step length using Armijo backtracking [2]. Rather than accepting the full step unconditionally, Armijo backtracking performs a line search to ensure a sufficient decrease in the objective function. Specifically, Armijo backtracking starts with the full step $\left( t _ { k } = 1 \right)$ and repeatedly reduces the step length by a factor $\beta \in ( 0 , 1 )$ until the Armijo condition

$$
F ( x _ { k } + t _ { k } s _ { k } ) \leq F ( x _ { k } ) + \alpha t _ { k } g _ { k } ^ { \top } s _ { k } , \qquad \alpha \in ( 0 , 1 ) ,\tag{8}
$$

is satisfied. The accepted step length is therefore $t _ { k } = \beta ^ { j _ { k } }$ , where $j _ { k }$ is the smallest nonnegative integer for which the inequality holds. The iterate is updated according to

$$
\boldsymbol { x } _ { k + 1 } = \boldsymbol { x } _ { k } + t _ { k } \boldsymbol { s } _ { k } .
$$

Once a step has been accepted, we define the damped quadratic model associated with the reduced Levenberg–Marquardt system,

$$
m _ { k } ( s ) : = \frac { 1 } { 2 } \| r _ { k } + J _ { k } s \| ^ { 2 } + \frac { \mu _ { k } } { 2 } \| D _ { k , p } ^ { 1 / 2 } Z _ { k } ^ { \top } V _ { k } ^ { \top } s \| ^ { 2 } .\tag{9}
$$

This model combines the linearized least-squares objective with the curvature-adaptive damping introduced in Section 3.3 and provides the predicted reduction used to update the damping parameter. The agreement between the predicted reduction of the model and the actual reduction of the objective is measured by the gain ratio

$$
\rho _ { k } : = \frac { F ( x _ { k } ) - F ( x _ { k } + t _ { k } s _ { k } ) } { m _ { k } ( 0 ) - m _ { k } ( t _ { k } s _ { k } ) } .\tag{10}
$$

Following [20, 7], the damping parameter is updated according to the gain ratio using the standard rule

$$
\mu _ { k + 1 } = \left\{ \begin{array} { l l } { \mu _ { k } / \mu _ { d o w n } , } & { \rho _ { k } \geq \rho _ { \mathrm { h i g h } } } \\ { \mu _ { k } , } & { \rho _ { \mathrm { l o w } } \leq \rho _ { k } < \rho _ { \mathrm { h i g h } } } \\ { \mu _ { k } . \mu _ { u p } , } & { \rho _ { k } < \rho _ { \mathrm { l o w } } } \end{array} \right. .\tag{11}
$$

Unlike the classical LM algorithm, step acceptance and damping updates are decoupled: Armijo backtracking determines the step length, while the gain ratio is used to update the damping parameter. This separation avoids repeated dampedsystem solves during step acceptance while preserving the role of the damping parameter as a measure of local quadratic approximation quality.

## 3.5 Pseudocode

Algorithm Hybrid subspace Levenberg Marquardt   
Input: Initial guess $x _ { 0 } ,$ , Jacobian matrix $J _ { k } ,$ , gradient vector $g _ { k } ,$ , damping term $\mu _ { 0 } > 0 ,$ , singular value threshold $\delta > 0 ,$   
adequacy threshold $\eta _ { \mathrm { m i n } } \in ( 0 , 1 ]$ , Armijo constants $( \alpha , \beta )$ , ratio thresholds $( \rho _ { \mathrm { l o w } } , \rho _ { \mathrm { h i g h } } )$   
for $k = 0 , 1 , 2 , \ldots$ do   
Construct a candidate basis, then orthonormalize it to obtain $V _ { k }$   
Compute $\eta _ { k } ^ { \mathrm { s u b } } = \vert \vert \boldsymbol { V } _ { k } ^ { \top } \boldsymbol { g } _ { k } \vert \vert ^ { 2 } / \vert \vert \boldsymbol { g } _ { k } \vert \vert ^ { 2 } .$   
while $g _ { k } \neq 0$ and $\eta _ { k } ^ { \mathrm { s u b } } < \eta _ { \mathrm { m i n } }$ do   
Expand the subspace, then update $V _ { k } .$   
Recompute $\eta _ { k } ^ { \mathrm { s u b } } .$   
end while   
Form $J _ { k , p } = J _ { k } V _ { k }$ and its SVD $J _ { k , p } = U _ { k } \Sigma _ { k , p } Z _ { k } ^ { \top }$   
Set $D _ { k , p } =$ diag(max $( \sigma _ { k , i } ^ { 2 } , \delta ) )$ and $B _ { k } = \Sigma _ { k , p } ^ { 2 } + \mu _ { k } D _ { k , p } .$   
Solve $\boldsymbol { B _ { k } } \boldsymbol { y _ { k } } = - \boldsymbol { \Sigma _ { k , p } } \boldsymbol { U _ { k } ^ { \top } } \boldsymbol { r _ { k } }$ and set $s _ { k } = V _ { k } Z _ { k } y _ { k } .$   
Find smallest $j \geq 0$ such that $t _ { k } = \beta ^ { j }$ satisfies (8); set $\boldsymbol { x } _ { k + 1 } = \boldsymbol { x } _ { k } + t _ { k } \boldsymbol { s } _ { k }$   
Update $\mu _ { k + 1 }$   
end for

## 4 CONVERGENCE ANALYSIS

This section establishes the global and local convergence properties of HSLM. The analysis follows the standard framework for line-search Levenberg–Marquardt methods while accounting for the adaptive reduced subspace. Building on the classical analysis, we identify the additional conditions needed to ensure that restricting the LM step to an adaptive subspac preserves the relevant global and local convergence properties. For global convergence, the key ingredients are the deterministic adequacy condition, which ensures that the reduced space retains sufficient gradient information, and the Armijo line search, which guarantees sufficient decrease. For local convergence, we quantify how accurately the reduced system approximates the full Newton system and show how the resulting approximation errors determine the local convergence rate.

## 4.1 Global Convergence

We first impose two standard assumptions used in the global convergence analysis of line-search methods [2, 24]. Lower boundedness prevents indefinite decrease of the objective along the initial level set, ${ \mathcal { L } } : = \{ x : F ( x ) \leq F ( x _ { 0 } ) \}$ , while Lipschitz continuity of the gradient provides the smoothness needed to establish sufficient decrease.

Assumption 1 (Lower boundedness on the level set). F is bounded below on ${ \mathcal { L } } .$

Assumption 2 (Lipschitz gradient on the level set). ∇F is Lipschitz continuous on an open neighborhood of $\mathcal { L } \colon$ there exist $L _ { 1 } > 0$ such that $\| \nabla F ( x ) - \nabla F ( y ) \| \leq L _ { 1 } \| x - y \|$ for all $x , y \in { \mathcal { L } }$

The remaining assumptions account for the fact that the search direction is computed only in a reduced subspace. In a full-space method, a nonzero gradient directly provides a descent direction. In a subspace method, however, the projected gradient may be arbitrarily small if the chosen subspace is poorly aligned with the gradient. Therefore, we required the algorithm to construct or enrich the subspace until it captures a uniformly positive fraction of the full-gradient information. This deterministic adequacy condition is analogous to alignment and subspace-quality requirements appear in randomized and reduced subspace methods [6, 25].

Assumption 3 (Deterministic adequacy enforcement). There exists $\eta _ { \mathrm { m i n } } \in ( 0 , 1 ]$ such that the algorithm enforces $\eta _ { k } ^ { \mathrm { s u b } } \geq \eta _ { \mathrm { m i n } }$ for all iterations with $g _ { k } \neq 0 ,$ , where $\eta _ { k } ^ { \mathrm { s u b } }$ is defined in (2).

Finally, we assume the reduced problem to remain uniformly well conditioned. Positive definiteness of the reduced matrix guarantees that the reduced model is strictly convex and that the subspace step is uniquely defined, while uniform eigenvalue bounds prevent the reduced system from becoming nearly singular or excessively scaled. These bounds also provide the estimates needed to relate the step norm and predicted reduction to the projected gradient, and are standard conditioning requirements for subspace methods [11, 20], even when the reduced Jacobian is rank deficient.

Assumption 4 (Uniform boundedness of reduced conditioning). There exists $m _ { B } , M _ { B }$ such that $0 < m _ { B } \leq \lambda _ { \operatorname* { m i n } } ( B _ { k } ) \leq \lambda _ { \operatorname* { m a x } } ( B _ { k } ) \leq$ $M _ { B } < \infty$ for all k, where $B _ { k }$ is defined in (6).

Before establishing descent and convergence properties, we first verify that the reduced Levenberg–Marquardt system is well posed. The following lemma shows that the regularization term makes the reduced matrix positive definite whenever $\mu _ { k } > 0 .$ , ensuring that the reduced step exists uniquely.

Lemma 1 (Positive definiteness of the reduced system). $B _ { k } = \Sigma _ { k , p } ^ { 2 } + \mu _ { k } D _ { k , p }$ is symmetric positive definite.

Proof. Since $\Sigma _ { k , p } ^ { 2 } \succeq 0 , \mu _ { k } > 0$ and (7), hence $B _ { k } \succ 0 .$

The next result shows that the positive definiteness of the reduced system translates into descent for the full objective. In particular, the deterministic adequacy condition prevents the projected gradient from vanishing when $g _ { k } \neq 0$ , while $B _ { k } \succ 0$ ensures that the reduced LM solution produces a strictly negative directional derivative.

Lemma 2 (Descent direction). Under Assumption 3, $i f g _ { k } \ne 0 ,$ , then the reduced LM step $s _ { k }$ defined in (6) satisfies $g _ { k } ^ { \top } s _ { k } < 0$ Proof. From (5) and (6),

$$
\begin{array} { r l } & { g _ { k } ^ { \top } s _ { k } = g _ { k } ^ { \top } V _ { k } Z _ { k } y _ { k } } \\ & { \quad = ( V _ { k } ^ { \top } g _ { k } ) ^ { \top } Z _ { k } y _ { k } } \\ & { \quad = ( Z _ { k } \Sigma _ { k , p } U _ { k } ^ { \top } r _ { k } ) ^ { \top } Z _ { k } y _ { k } } \\ & { \quad = ( \Sigma _ { k , p } U _ { k } ^ { \top } r _ { k } ) ^ { \top } y _ { k } } \\ & { \quad = - ( \Sigma _ { k , p } U _ { k } ^ { \top } r _ { k } ) ^ { \top } B _ { k } ^ { - 1 } \Sigma _ { k , p } U _ { k } ^ { \top } r _ { k } . } \end{array}
$$

By Lemma 1, $B _ { k } ^ { - 1 } \succ 0 .$ , hence $g _ { k } ^ { \top } s _ { k } < 0$ whenever $g _ { k } \neq 0 .$

After showing $s _ { k }$ is a strict descent direction, we next verify that the line-search procedure is well defined. By Lipschitz continuity of the gradient, the first-order decrease along $s _ { k }$ dominates the quadratic Taylor remainder for sufficiently small step lengths, ensuring that the Armijo condition is eventually satisfied.

Lemma 3 (Existence of an Armijo step). Assume Assumption 2. $I f g _ { k } ^ { \top } s _ { k } < 0$ and $\alpha \in ( 0 , 1 )$ , then there exists $\bar { t } _ { k } > 0$ such that for all $t \in ( 0 , \bar { t } _ { k } ]$

$$
F \left( x _ { k } + t s _ { k } \right) \leq F \left( x _ { k } \right) + \alpha t g _ { k } ^ { \top } s _ { k } ,
$$

and Armijo backtracking terminates.

Proof. From Assumption 2, we have

$$
\begin{array} { r l r } { F ( x _ { k } + t s _ { k } ) - F ( x _ { k } ) = \displaystyle \int _ { 0 } ^ { 1 } \nabla F ( x _ { k } + y t s _ { k } ) ^ { \top } ( t s _ { k } ) d y } & { \quad \quad \mathrm { ( B y ~ F u n d a m e n t a l ~ T h e o r e m ~ o f ~ C a l c u l u s ) } } \\ & { } & { \quad = \displaystyle \int _ { 0 } ^ { 1 } \nabla F ( x _ { k } ) ^ { \top } ( t s _ { k } ) d y + \displaystyle \int _ { 0 } ^ { 1 } \left( \nabla F \big ( x _ { k } + y t s _ { k } \big ) - \nabla F ( x _ { k } ) \right) ^ { \top } ( t s _ { k } ) d y } \\ & { } & { \quad \le t g _ { k } ^ { \top } s _ { k } + \displaystyle \int _ { 0 } ^ { 1 } L _ { 1 } \| y s _ { k } \| \| t s _ { k } \| d y } \\ & { } & { \quad \le t g _ { k } ^ { \top } s _ { k } + L _ { 1 } t ^ { 2 } \| s _ { k } \| ^ { 2 } \displaystyle \int _ { 0 } ^ { 1 } y d y } \\ & { } & { \quad \le t g _ { k } ^ { \top } s _ { k } + \displaystyle \frac { L _ { 1 } } { 2 } t ^ { 2 } \| s _ { k } \| ^ { 2 } . } \end{array}
$$

Then,

$$
F ( x _ { k } + t s _ { k } ) - F ( x _ { k } ) - \alpha t g _ { k } ^ { \top } s _ { k } \leq ( 1 - \alpha ) t g _ { k } ^ { \top } s _ { k } + \frac { L _ { 1 } } { 2 } t ^ { 2 } \| s _ { k } \| ^ { 2 } .
$$

Since $g _ { k } ^ { \top } s _ { k } < 0$ , it suffices to choose

$$
0 < t \leq \bar { t } _ { k } : = - \frac { 2 ( 1 - \alpha ) g _ { k } ^ { \top } s _ { k } } { L _ { 1 } \| s _ { k } \| ^ { 2 } } .
$$

Backtracking over $t = \beta ^ { j _ { k } }$ will eventually produce $t \leq \bar { t } _ { k }$ for some $j _ { k }$

We now combine the preceding lemmas to establish global convergence. Specifically, we show that every infinite sequence generated by the algorithm has a subsequence along which the gradient norm converges to zero.

Theorem 1 (Global convergence to stationarity). Under Assumptions 1, 2, 3, and $^ { 4 , }$ either the algorithm terminates at a stationary point, or

$$
\operatorname* { l i m i n f } _ { k \to \infty } \left\| g _ { k } \right\| = 0 .
$$

Proof. Assume the algorithm does not terminate and suppose, for contradiction, that there exists $\varepsilon > 0$ such that $\left\| g _ { k } \right\| \geq \varepsilon$ for

all k. Lemma 3 shows Armijo backtracking holds for all $t \leq \bar { t } _ { k }$ with

$$
\bar { t } _ { k } = - \frac { 2 ( 1 - \alpha ) g _ { k } ^ { \top } s _ { k } } { L _ { 1 } \| s _ { k } \| ^ { 2 } } .
$$

To prevent $\bar { t } _ { k } \to 0 .$ , we want to bound $\| s _ { k } \|$ and $g _ { k } ^ { \top } s _ { k }$

Consider $y _ { k } = - B _ { k } ^ { - 1 } \Sigma _ { k , p } U _ { k } ^ { \top } r _ { k }$ and $\| s _ { k } \| = \| V _ { k } Z _ { k } y _ { k } \| = \| y _ { k } \|$ ,

$$
\begin{array} { r l } & { \| s _ { k } \| \leq \| B _ { k } ^ { - 1 } \| \left\| \Sigma _ { k , p } U _ { k } ^ { \top } r _ { k } \right\| } \\ & { \qquad \leq \frac { 1 } { \lambda _ { \operatorname* { m i n } } ( B _ { k } ) } \| \Sigma _ { k , p } U _ { k } ^ { \top } r _ { k } \| } \\ & { \qquad = \frac { 1 } { \lambda _ { \operatorname* { m i n } } ( B _ { k } ) } \| Z _ { k } \Sigma _ { k , p } U _ { k } ^ { \top } r _ { k } \| } \\ & { \qquad = \frac { 1 } { \lambda _ { \operatorname* { m i n } } ( B _ { k } ) } \| V _ { k } ^ { \top } g _ { k } \| . } \end{array}
$$

From Lemma 2 and (5),

$$
\begin{array} { r l } {  { - g _ { k } ^ { \top } s _ { k } = ( \Sigma _ { k , p } U _ { k } ^ { \top } r _ { k } ) ^ { \top } B _ { k } ^ { - 1 } \Sigma _ { k , p } U _ { k } ^ { \top } r _ { k } } } \\ & { \ge \frac { 1 } { \lambda _ { \operatorname* { m a x } } ( B _ { k } ) } \| \Sigma _ { k , p } U _ { k } ^ { \top } r _ { k } \| ^ { 2 } } \\ & { = \frac { 1 } { \lambda _ { \operatorname* { m a x } } ( B _ { k } ) } \| Z _ { k } \Sigma _ { k , p } U _ { k } ^ { \top } r _ { k } \| ^ { 2 } } \\ & { = \frac { 1 } { \lambda _ { \operatorname* { m a x } } ( B _ { k } ) } \| V _ { k } ^ { \top } g _ { k } \| ^ { 2 } . } \end{array}
$$

By Assumption 4 and (2),

$$
- g _ { k } ^ { \top } s _ { k } \ge \frac { 1 } { M _ { B } } \eta _ { \operatorname* { m i n } } \| g _ { k } \| ^ { 2 } \ge \frac { \eta _ { \operatorname* { m i n } } } { M _ { B } } \varepsilon ^ { 2 } > 0 .
$$

Then,

$$
\frac { - g _ { k } ^ { \top } s _ { k } } { \| s _ { k } \| ^ { 2 } } \geq \frac { \lambda _ { \operatorname* { m i n } } ( B _ { k } ) ^ { 2 } } { \lambda _ { \operatorname* { m a x } } ( B _ { k } ) } .
$$

From assumption 4, we have $\bar { t } _ { k } \geq \bar { t } = \frac { 2 ( 1 - \alpha ) m _ { B } ^ { 2 } } { M _ { B } L _ { 1 } } > 0 .$

Armijo backtracking then returns $t _ { k } \ge \mathrm { m i n } \{ 1 , \beta \bar { t } \}$ , which satisfy

$$
F ( x _ { k + 1 } ) \leq F ( x _ { k } ) + \alpha t _ { k } g _ { k } ^ { \top } s _ { k } \leq F ( x _ { k } ) - \alpha t _ { k } \frac { \eta _ { \operatorname* { m i n } } } { M _ { B } } \varepsilon ^ { 2 } .
$$

Thus, $F ( x _ { k } ) \to - \infty { \mathrm { a s } } k \to \infty .$ , contradicting Assumption 1. Therefore limin $\operatorname { f } _ { k \to \infty } \left\| g _ { k } \right\| = 0 .$

## 4.2 Local Convergence

Now, we introduce the additional assumptions required for the local convergence analysis. Unlike the global assumptions, which ensure sufficient descent and convergence toward stationarity, these conditions describe the local geometry of the

objective and require the subspace and curvature approximations to become increasingly accurate as the iterates approach a solution. The key identity for local analysis is the exact Hessian decomposition

$$
\nabla ^ { 2 } F ( x ) = J ( x ) ^ { \top } J ( x ) \ + S ( x ) \quad { \mathrm { ~ w h e r e ~ } } S ( x ) = \sum _ { i = 1 } ^ { m } r _ { i } ( x ) \nabla ^ { 2 } r _ { i } ( x )
$$

where $J ( x ) ^ { \top } J ( x )$ is the Gauss–Newton curvature and $S ( x )$ collects residual-dependent second-derivative. The Levenberg–Marquardt algorithm is built on the Gauss–Newton approximation $H ( x ) \approx J ^ { T } J$ and computes steps from a damped system. In this study, we analyze the local convergence of the proposed algorithm under the assumption of zero (or very small) residual at the optimal solutions.

We first impose the standard local regularity conditions used in Newton-type convergence analyses [11, 24]. Positive definiteness of the Hessian at $x ^ { \star }$ ensures that the solution is locally isolated and that the objective has stable quadratic curvature near the minimizer. Lipschitz continuity of the Hessian controls the remainder in the second-order Taylor expansion and is essential for deriving a local error recursion.

Assumption 5 (Strict local minimizer and Lipschitz Hessian). There exists a strict local minimizer x<sup>⋆</sup> such that $g ( x ^ { \star } ) = 0$ and $\nabla ^ { 2 } F ( x ^ { \star } ) \succ 0$ . Moreover, $\nabla ^ { 2 } F ( x )$ is Lipschitz in a neighborhood U of $x ^ { \star } \colon \| \nabla ^ { 2 } F ( x ) - \nabla ^ { 2 } F ( y ) \| \leq L _ { 2 } \| x - y \| f o r$ r all $x , y \in U$

The next assumption accounts for the evolving reduced subspace, which arise in reduced and randomized subspace methods [6, 25]. Global convergence requires only that the subspace retain a fixed amount of gradient information, whereas fast local convergence requires the subspace approximation to improve as $x _ { k }$ approaches $x ^ { \star }$ . Accordingly, the component of the gradient omitted by the subspace must vanish, the reduced space must become increasingly invariant under the local curvature operator, and the LM regularization must disappear as the subspace approximation becomes more accurate.

Assumption 6 (Vanishing subspace/curvature errors and regularization). Let $P _ { k } : = V _ { k } V _ { k } ^ { \top }$ and define $\gamma _ { k } : = \| ( I - P _ { k } ) g _ { k } \| / \| g _ { k } \|$ Suppose $x _ { k } \to x ^ { \star } , \gamma _ { k } \to 0 , \mu _ { k } \to 0 ,$ , and there exists $\varepsilon _ { k } \to 0$ such that

$$
\begin{array} { r } { \| H _ { k } P _ { k } - P _ { k } H _ { k } P _ { k } \| \leq \varepsilon _ { k } \| H _ { k } \| . } \end{array}
$$

Here, $\gamma _ { k }$ measures the relative portion of the gradient missed by the subspace, while $\varepsilon _ { k }$ measures the relative curvature (Hessian) information not captured by the subspace. As $\gamma _ { k } , \varepsilon _ { k } \to 0 .$ , the reduced subspace increasingly captures both the relevant gradient and curvature information near $x ^ { * }$ . Finally, the local curvature approximation used in the LM system must become sufficiently accurate near the solution. The following condition requires the Hessian approximation error to decrease at least linearly with the distance to $x ^ { \star }$ , which ensures that the Hessian approximation error becomes negligible at the rate required for fast local convergence. Consequently, its contribution to the local iterate-error recursion is of quadratic order, allowing the method to recover the fast local behavior of a Newton-type iteration.

Assumption 7 (Vanishing Hessian approximation error near x<sup>⋆</sup>). Assume there exist $c > 0$ such that for all x close enough to $x ^ { \star } ,$

$$
\| S ( x ) \| \leq c \| x - x ^ { \star } \| .
$$

To analyze the local convergence rate, we next quantify how accurately the reduced regularized step satisfies the full Newton equation. The following lemma provides a bound on the step norm and decomposes the exact Newton residual into four distinct error components: the gradient component omitted by the subspace, the lack of curvature invariance of the subspace, the LM regularization, and the Hessian-approximation error. This decomposition makes explicit the conditions under which the reduced LM step becomes an increasingly accurate inexact Newton step near $x ^ { \star }$

Lemma 4 (Exact Newton residual bound). Let $P _ { k } : = V _ { k } V _ { k } ^ { \top }$ and suppose $s _ { k } = V _ { k } Z _ { k } y _ { k }$ satisfies a reduced regularized system

$$
\begin{array} { r } { ( Z _ { k } ^ { \top } V _ { k } ^ { \top } H _ { k } V _ { k } Z _ { k } + R _ { k } ) y _ { k } = - Z _ { k } ^ { \top } V _ { k } ^ { \top } g _ { k } , } \end{array}
$$

where $R _ { k } = \mu _ { k } D _ { k , p } \succeq 0$ and $Z _ { k } ^ { \top } V _ { k } ^ { \top } H _ { k } V _ { k } Z _ { k } + R _ { k } \succeq m _ { B } I$ for some $m _ { B } > 0 .$ . Then:

1. ∥s<sub>k</sub>∥ = ∥y<sub>k</sub>∥ ≤ 1 ∥V<sup>⊤</sup><sub>k</sub> g<sub>k</sub>∥ ≤ 1 ∥g<sub>k</sub>∥. m<sub>B</sub> m<sub>B</sub>

2. Defining the exact Newton residual $R _ { k } ^ { e x a c t } : = \nabla ^ { 2 } F ( x _ { k } ) s _ { k } + g _ { k }$ ,

$$
\| R _ { k } ^ { e x a c t } \| \leq \| ( I - P _ { k } ) g _ { k } \| + \| H _ { k } P _ { k } - P _ { k } H _ { k } P _ { k } \| \| s _ { k } \| + \| R _ { k } \| \| y _ { k } \| + \| S _ { k } \| \| s _ { k } \| .
$$

Proof. (1) Let $B _ { k } : = Z _ { k } ^ { \top } V _ { k } ^ { \top } H _ { k } V _ { k } Z _ { k } + R _ { k } \succeq m _ { B } I , \mathrm { s o } \ \| B _ { k } ^ { - 1 } \| \leq \frac { 1 } { m _ { B } } .$ . Then

$$
y _ { k } = - B _ { k } ^ { - 1 } Z _ { k } ^ { \top } V _ { k } ^ { \top } g _ { k } \implies \left\| y _ { k } \right\| \leq \frac { 1 } { m _ { B } } \| Z _ { k } ^ { \top } V _ { k } ^ { \top } g _ { k } \| = \frac { 1 } { m _ { B } } \| V _ { k } ^ { \top } g _ { k } \| \leq \frac { 1 } { m _ { B } } \| g _ { k } \| .
$$

Since $V _ { k }$ has orthonormal columns and $Z _ { k }$ is orthogonal, $\| s _ { k } \| = \| V _ { k } Z _ { k } y _ { k } \| = \| y _ { k } \|$

(2) Let $R _ { k } ^ { e x a c t } : = ( H _ { k } + S _ { k } ) s _ { k } + g _ { k } = R _ { k } ^ { ( N ) } + S _ { k } s _ { k }$ where $R _ { k } ^ { ( N ) } = H _ { k } s _ { k } + g _ { k }$

Consider $R _ { k } ^ { ( N ) } = ( I - P _ { k } ) R _ { k } ^ { ( N ) } + P _ { k } R _ { k } ^ { ( N ) }$ and $s _ { k } = P _ { k } s _ { k }$ , we have

$$
\begin{array} { r l } { \| ( I - P _ { k } ) R _ { k } ^ { ( N ) } \| = \| ( I - P _ { k } ) H _ { k } P _ { k } s _ { k } + ( I - P _ { k } ) g _ { k } \| } & { } \\ { \ } & { } \\ { \leq \| ( I - P _ { k } ) H _ { k } P _ { k } \| \| s _ { k } \| + \| ( I - P _ { k } ) g _ { k } \| } & { } \\ { \ } & { } \\ { \qquad } & { = \| H _ { k } P _ { k } - P _ { k } H _ { k } P _ { k } \| \| s _ { k } \| + \| ( I - P _ { k } ) g _ { k } \| . } \end{array}\tag{12}
$$

By multiplying the reduced regularized system by $V _ { k } Z _ { k }$ , we also have

$$
V _ { k } Z _ { k } Z _ { k } ^ { \top } V _ { k } ^ { \top } H _ { k } V _ { k } Z _ { k } y _ { k } + V _ { k } Z _ { k } R _ { k } y _ { k } = - V _ { k } Z _ { k } Z _ { k } ^ { \top } V _ { k } ^ { \top } g _ { k }
$$

$$
P _ { k } H _ { k } s _ { k } + V _ { k } Z _ { k } R _ { k } y _ { k } = - P _ { k } g _ { k }
$$

$$
P _ { k } ( H _ { k } s _ { k } + g _ { k } ) = - V _ { k } Z _ { k } R _ { k } y _ { k }
$$

$$
P _ { k } R _ { k } ^ { ( N ) } = - V _ { k } Z _ { k } R _ { k } y _ { k } .
$$

Then

$$
\| P _ { k } R _ { k } ^ { ( N ) } \| = \| - V _ { k } Z _ { k } R _ { k } y _ { k } \| \leq \| V _ { k } \| \| Z _ { k } \| \| R _ { k } \| \| y _ { k } \| = \| R _ { k } \| \| y _ { k } \|\tag{13}
$$

Combining the estimates 12 and 13 yields the central local error recursion

$$
\begin{array} { r l } & { \| R _ { k } ^ { e x a c t } \| \leq \| R _ { k } ^ { ( N ) } \| + \| S _ { k } s _ { k } \| } \\ & { \qquad \leq \| ( I - P _ { k } ) R _ { k } ^ { ( N ) } \| + \| P _ { k } R _ { k } ^ { ( N ) } \| + \| S _ { k } s _ { k } \| } \\ & { \qquad \leq \| H _ { k } P _ { k } - P _ { k } H _ { k } P _ { k } \| \| s _ { k } \| + \| ( I - P _ { k } ) g _ { k } \| + \| R _ { k } \| \| y _ { k } \| + \| S _ { k } \| \| s _ { k } \| . } \end{array}
$$

Using the exact Newton residual bound in Lemma 4, we relate the local convergence rate to the combined subspace, curvature, regularization, and Hessian-approximation errors. Once full steps are accepted, a uniformly controlled residual yields local linear convergence, while vanishing residual components make the reduced LM step asymptotically equivalent to the exact Newton step and therefore produce superlinear convergence.

Theorem 2 (Local linear and superlinear convergence). Assume Assumptions $_ { 4 - 7 . }$ Let $e _ { k } : = x _ { k } - x ^ { \star }$ and assume eventually $t _ { k } = 1$ . Then:

1. $I f \theta _ { k } : = C ( \gamma _ { k } + \varepsilon _ { k } + \| R _ { k } \| + \| S _ { k } \| ) \leq { \bar { \theta } } < 1$ for all sufficiently large k, then $\| e _ { k + 1 } \| \le q \| e _ { k } \| f o r$ some $q < 1$ (linear convergence).

2. If $\theta _ { k } \to 0 ,$ , then $\| e _ { k + 1 } \| / \| e _ { k } \| \to 0$ (superlinear convergence).

Proof. Since $\nabla ^ { 2 } F ( x ^ { \star } ) \succ 0$ and $\nabla ^ { 2 } F$ is Lipschitz near $x ^ { \star }$ , there exist $m _ { 1 } , M _ { 1 } > 0$ and a neighborhood U such that $m _ { 1 } I \preceq$ $\nabla ^ { 2 } F ( x ) \preceq M _ { 1 } I$ for all $x \in U \left[ 1 1 \right]$ , 24]. Since $\nabla F ( x ^ { \star } ) = 0$ , we have

$$
\begin{array} { r l } & { g _ { k } = \nabla F ( x _ { k } ) - \nabla F ( x ^ { \star } ) = \displaystyle \int _ { 0 } ^ { 1 } \nabla ^ { 2 } F ( x ^ { \star } + t e _ { k } ) e _ { k } d t \qquad \quad \mathrm { ( B y ~ F u n d a m e n t a l ~ T h e o r e m ~ o f ~ C a l c u l u s ) } } \\ & { \qquad = \displaystyle \int _ { 0 } ^ { 1 } \left[ \nabla ^ { 2 } F ( x ^ { \star } + t e _ { k } ) - \nabla ^ { 2 } F ( x _ { k } ) e _ { k } \right] d t + \displaystyle \int _ { 0 } ^ { 1 } \nabla ^ { 2 } F ( x _ { k } ) e _ { k } d t } \\ & { \qquad = h _ { k } + \nabla ^ { 2 } F ( x _ { k } ) e _ { k } . } \end{array}
$$

where $\begin{array} { r } { h _ { k } = \int _ { 0 } ^ { 1 } \left[ \nabla ^ { 2 } F ( x ^ { \star } + t e _ { k } ) - \nabla ^ { 2 } F ( x _ { k } ) \right] e _ { k } d t } \end{array}$ . Using Hessian Lipschitz continuity, we also have

$$
\| h _ { k } \| \leq \| e _ { k } \| \int _ { 0 } ^ { 1 } \| \nabla ^ { 2 } F ( x ^ { \star } + t e _ { k } ) - \nabla ^ { 2 } F ( x _ { k } ) \| d t \leq \| e _ { k } \| \int _ { 0 } ^ { 1 } L _ { 2 } ( 1 - t ) \| e _ { k } \| d t = \frac { L _ { 2 } } { 2 } \| e _ { k } \| ^ { 2 }
$$

Define the Newton residual $R _ { k } ^ { e x a c t } : = \nabla ^ { 2 } F ( x _ { k } ) s _ { k } + g _ { k }$ . Since $H ( x ) = J ( x ) ^ { \top } J ( x )$ is continuous in a neighborhood of $x ^ { * }$ , there exists $M _ { H } > 0$ such that $\| H _ { k } \| \le M _ { H }$ for all sufficiently large k. By Lemma 4 and Assumption $^ { 6 , }$

$$
\| R _ { k } ^ { e x a c t } \| \leq \| H _ { k } P _ { k } - P _ { k } H _ { k } P _ { k } \| \| s _ { k } \| + \| ( I - P _ { k } ) g _ { k } \| + \| R _ { k } \| \| y _ { k } \| + \| S _ { k } \| \| s _ { k } \|
$$

$$
\leq \varepsilon _ { k } \| H _ { k } \| { \frac { 1 } { m _ { B } } } \| g _ { k } \| + \gamma _ { k } \| g _ { k } \| + \| R _ { k } \| { \frac { 1 } { m _ { B } } } \| g _ { k } \| + \| S _ { k } \| { \frac { 1 } { m _ { B } } } \| g _ { k } \|
$$

$$
\leq ( \frac { M _ { H } \varepsilon _ { k } } { m _ { B } } + \gamma _ { k } + \frac { \| R _ { k } \| } { m _ { B } } + \frac { \| S _ { k } \| } { m _ { B } } ) \| g _ { k } \| .
$$

At $t _ { k } = 1 , e _ { k + 1 } = e _ { k } + s _ { k }$ and

$$
\nabla ^ { 2 } F ( x _ { k } ) e _ { k + 1 } = \nabla ^ { 2 } F ( x _ { k } ) e _ { k } + \nabla ^ { 2 } F ( x _ { k } ) s _ { k } = ( g _ { k } - h _ { k } ) + ( R _ { k } ^ { e x a c t } - g _ { k } ) = R _ { k } ^ { e x a c t } - h _ { k } .
$$

Then,

$$
\| e _ { k + 1 } \| \leq \| \nabla ^ { 2 } F ( x _ { k } ) ^ { - 1 } \| ( \| R _ { k } ^ { e x a c t } \| + \| h _ { k } \| ) \leq \frac { 1 } { m _ { 1 } } \left[ \left( \frac { M _ { H } \varepsilon _ { k } } { m _ { B } } + \gamma _ { k } + \frac { \| R _ { k } \| } { m _ { B } } + \frac { \| S _ { k } \| } { m _ { B } } \right) \| g _ { k } \| + \frac { L _ { 2 } } { 2 } \| e _ { k } \| ^ { 2 } \right] .\tag{14}
$$

We also have

$$
\| g _ { k } \| = \| \nabla ^ { 2 } F ( x _ { k } ) e _ { k } + h _ { k } \| \leq \| \nabla ^ { 2 } F ( x _ { k } ) \| \| e _ { k } \| + \| h _ { k } \| \leq M _ { 1 } \| e _ { k } \| + \frac { L _ { 2 } } { 2 } \| e _ { k } \| ^ { 2 }\tag{15}
$$

Combine 14 and 15, we have

$$
\| e _ { k + 1 } \| \leq a \| e _ { k } \| ^ { 2 } + \theta _ { k } \| e _ { k } \| \qquad { \mathrm { ~ f o r ~ s o m e ~ c o n s t a n t ~ } } a > 0
$$

If $\theta _ { k } \le \bar { \theta } < 1$ eventually, choose $\bar { \theta } < q < 1$ and take sufficient large k so that $a \| e _ { k } \| \le q - \bar { \theta }$ , yielding $\| e _ { k + 1 } \| \leq q \| e _ { k } \|$

If $\theta _ { k } \to 0 .$ , then $\| e _ { k + 1 } \| = o ( \| e _ { k } \| )$ , giving superlinear convergence.

Remark 2. The local convergence analysis above is carried out using the Gauss–Newton/LM curvature approximation $H ( x ) =$ $J ( x ) ^ { \top } J ( x )$ . Therefore, the superlinear local behavior established in this section is only applicable to the zero-residual (or very small) regime, i.e.,

$$
r ( x ^ { \star } ) \approx 0 .
$$

In contrast, if $r ( x ^ { \star } ) \neq 0$ , then the additional term $S ( x )$ in $\nabla ^ { 2 } F ( x ^ { \star } )$ is generally nonzero, so $J ( x ^ { \star } ) ^ { \top } J ( x ^ { \star } )$ does not coincide with the true Hessian. In that case, the HSLM method is typically expected to be at best locally linear.

## 5 NUMERICAL EXPERIMENTS

We evaluate the HSLM algorithm by training feedforward neural networks for nonlinear function approximation. Feedforward neural networks are suitable test problems because their training can be formulated as a nonlinear least-squares problem whose size can be systematically increased by varying the network architecture and training dataset, while preserving the same approximation objective. Their use is motivated by their ability to represent complex nonlinear relationships and by classical universal approximation results [10, 15, 3]. Although the classical LM method is known for rapid convergence and strong robustness when training small- to medium-sized neural networks [1, 17, 28], its computational and memory costs increase significantly with the number of parameters and data points. In this section, we compare classical LM, Krylov-subspace LM, and HSLM across different network and dataset sizes and measure performance using training and validation errors, convergence behavior, and computational cost. Our experiments evaluate whether HSLM can maintain convergence behavior and solution quality comparable to classical LM while reducing computational cost as the problem size increases.

## 5.1 Experimental Setup

The neural network is trained to approximate the following Friedman function:

$$
f ( { \bf z } ) = 1 0 \sin ( \pi z _ { 1 } z _ { 2 } ) + 2 0 ( z _ { 3 } - \frac { 1 } { 2 } ) ^ { 2 } + 1 0 z _ { 4 } + 5 z _ { 5 }
$$

where $\mathbf { z } = ( z _ { 1 } , z _ { 2 } , . . . , z _ { 7 } ) \in [ - 1 , 1 ] ^ { 7 }$ . This modified nonlinear-regression benchmark, which was introduced by Friedman [12], combines trigonometric, linear, and polynomial components, making it well-suited for evaluating the nonlinear least-squares algorithms.

Training input samples are drawn uniformly from $[ - 1 , 1 ] ^ { 7 }$ , and corresponding outputs are computed from Friedman’s function. Three training-set sizes are considered (Table 1), while 1000 noise-free samples are reserved for validation. To assess robustness, additive zero-mean Gaussian noise with standard deviation equal to 5% of the standard deviation of the target signal, $\sigma _ { Y }$ , is applied to the training output,

$$
\hat { \mathbf { y } } = f ( \mathbf { z } ) + \varepsilon \qquad \mathrm { w h e r e } \quad \varepsilon \sim \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } ) , \quad \sigma = 0 . 0 5 \sigma _ { Y }
$$

A fully connected multilayer perceptron (MLP) with two hidden layers is used to approximate the target function [28]. For an input vector $\mathbf { z } \in \mathbf { R } ^ { 7 }$ , the hidden layers use hyperbolic tangent activation function, while the output layer is linear

$$
\begin{array} { c } { { { \bf { h } } ^ { ( 1 ) } = \operatorname { t a n h } \left( { \bf W } ^ { ( 1 ) } { \bf z } + { \bf b } ^ { ( 1 ) } \right) , } } \\ { { { \bf { h } } ^ { ( 2 ) } = \operatorname { t a n h } \left( { \bf W } ^ { ( 2 ) } { \bf h } ^ { ( 1 ) } + { \bf b } ^ { ( 2 ) } \right) , } } \\ { { { \bf { y } } = { \bf W } ^ { ( 3 ) } { \bf h } ^ { ( 2 ) } + { \bf b } ^ { ( 3 ) } . } } \end{array}
$$

where $\mathbf { W } ^ { ( l ) }$ and $\mathbf { b } ^ { ( l ) }$ denote the weight matrices and bias vectors of layer l, respectively.

Three network architectures, with increasing numbers of training parameters $( \mathbf { W } ^ { ( l ) }$ and $\mathbf { b } ^ { ( l ) } )$ are described in Table 1, allowing the dimension of the nonlinear least-squares problem to systematically increase while using the same underlying regression task.

Table 1. Neural network configurations with increasing model complexity and training data size. Network architecture is reported as the number of neurons in the input layer, the two hidden layers, and the output layer, respectively.
<table><tr><td>Network</td><td>Architecture</td><td>Trainable Parameters</td><td>Training data size</td></tr><tr><td>1</td><td> $\overline { { 7 \to 3 5 \to 2 0 \to 1 } }$ </td><td>1021</td><td>10000</td></tr><tr><td>2</td><td> $7  6 0  2 5  1$ </td><td>2031</td><td>20000</td></tr><tr><td>3</td><td> $7  8 0  4 0  1$ </td><td>3921</td><td>40000</td></tr></table>

By collecting all trainable weights and biases into a single parameter vector x, training the neural network can be formulated as a nonlinear least-squares problem

$$
\operatorname* { m i n } _ { \mathbf { x } } { \frac { 1 } { 2 } } \| \mathbf { r } ( \mathbf { x } ) \| ^ { 2 }
$$

where the residual vector is

$$
\mathbf { r } ( \mathbf { x } ) = \left[ \begin{array} { c } { y _ { 1 } ( \mathbf { x } ) - \hat { y } _ { 1 } } \\ { y _ { 2 } ( \mathbf { x } ) - \hat { y } _ { 2 } } \\ { \vdots } \\ { y _ { N } ( \mathbf { x } ) - \hat { y } _ { N } } \end{array} \right] ,
$$

with $y _ { i } ( \mathbf { x } ) , \hat { y _ { i } }$ denoting the network prediction for the $i ^ { t h }$ training input and the corresponding noisy target output. The Jacobian matrix

$$
\mathbf { J } ( \mathbf { x } ) = { \frac { \partial \mathbf { r } } { \partial \mathbf { x } } }
$$

is computed efficiently using a modified backpropagation procedure, which is described in detail in [13].

In our implementation, the maximum subspace dimension in both Krylov subspace LM and HSLM is limited to 10% of the parameter dimension. HSLM constructs the reduced basis using the hybrid strategy described in Section 3, whereas Krylov-subspace LM uses Lanczos-based projection of the scaled LM system, consistent with the Krylov subspace framework described in [19]. Additional implementation details are provided in Appendix A.

## 5.2 Results

We compare the three algorithms in terms of convergence behavior and computational efficiency as the size of the nonlinear least-squares problem increases. For each network architecture, 50 trials are performed using randomly generated initial weights $\mathbf { W } ^ { ( 1 ) }$ and bias $\mathbf { b } ^ { ( \mathbf { I } ) }$ . Figure 1 shows five representative runs for the largest network using identical initializations across the three algorithms, while Table 2 summarizes the average number of iterations and computational time per iteration over all 50 trials. We use the mean squared error (MSE) to quantify both the training error and the validation error.

![](images/e6195d121ec825d8f71bf42fc21f2198cc10e311c01c23e56cd317701d5ef854.jpg)  
Figure 1. Training results for five representative runs of network 3 for nonlinear function approximation. A1-A2. Levenberg–Marquardt algorithm. B1-B2. Krylov subspace Levenberg–Marquardt algorithm. C1-C2. Hybrid subspace Levenberg–Marquardt algorithm. The gray dashed line indicates the variance of the added noise, $\sigma ^ { 2 } \approx 0 . 6 2 2$ . These results indicate that HSLM achieves convergence behavior comparable to the classical LM and Krylov-subspace LM algorithms.

Table 2. Comparison in Computational Performance of Neural Network Training Algorithms
<table><tr><td rowspan="2"></td><td colspan="2">Classical LM</td><td colspan="2">Krylov Subspace LM</td><td colspan="2">HSLM</td></tr><tr><td>Iteration</td><td>Time / Iteration (s)</td><td>Iteration</td><td>Time / Iteration (s)</td><td>Iteration</td><td>Time / Iteration (s)</td></tr><tr><td>Network 1</td><td>41.7</td><td>0.492</td><td>41.1</td><td>0.262</td><td>52.7</td><td>0.131</td></tr><tr><td>Network 2</td><td>41.6</td><td>2.367</td><td>39.4</td><td>1.738</td><td>36.9</td><td>0.615</td></tr><tr><td>Network 3</td><td>34.9</td><td>15.830</td><td>36.5</td><td>13.082</td><td>35.4</td><td>3.665</td></tr></table>

All three algorithms achieve similar final training errors, with the training error approaching the variance of the added noise, $\sigma ^ { 2 } ,$ indicating that the networks capture the underlying Friedman function to approximately the noise level (see Ap pendix B for network 1 & 2). Despite this comparable convergence behavior, their computational costs differ substantially. HSLM reduces the average execution time per iteration relative to classical LM by factors of approximately 3.7, 3.8, and 4.3 for Networks 1, 2, and 3, respectively. Relative to Krylov-subspace LM, the corresponding speedups are approximately 2.0, 2.8, and 3.6. The increasing advantage with network size suggests that the reduced adaptive formulation becomes increasingly beneficial as the cost of the full LM system grows.

HSLM achieves this computational efficiency using relatively small adaptive subspaces. The deterministic adequacy monitor adjusts the subspace dimension according to the information required at each iteration. As shown in Appendix C, HSLM generally operates with a substantially smaller subspace than Krylov-subspace LM and still satisfies the prescribed projectedgradient adequacy criterion. The convergence behavior also depends on the composition of the subspace. Appendix D shows that randomized curvature probes alone, or randomized probes combined only with the gradient, converge more slowly than the full hybrid construction. This comparison supports the use of complementary gradient, memory, Krylov/Lanczos, and

randomized curvature information in HSLM.

## 6 DISCUSSION

We have developed an adaptive hybrid subspace Levenberg–Marquardt method for large-scale nonlinear least-squares problems. The framework provides convergence guarantees for a general hybrid subspace construction, so the choice of basis directions can be tailored to the structure of the problem. The adequacy monitor provides a systematic mechanism for integrating complementary basis directions while adaptively controlling the subspace dimension. The numerical experiments illustrate both aspects of this construction: Appendix C shows how the subspace dimension adapts during optimization, while Appendix D demonstrates the benefit of combining multiple sources of descent and curvature information. Because the convergence framework does not prescribe a specific candidate pool, the subspace construction can incorporate problemdependent directions, providing flexibility for other classes of large-scale nonlinear least-squares problems [22, 26, 19, 4].

In addition, separating step acceptance from damping adjustment allows HSLM to retain adaptive regularization while decoupling the two mechanisms [2, 24]. Armijo backtracking determines whether and how far to move along the computed direction, whereas the gain ratio is used only to adjust the damping parameter based on the quality of the local quadratic approximation [7, 20, 23]. This separation avoids the cost of repeated damped-system solves during step acceptance without eliminating the stabilizing role of LM regularization. Moreover, the predicted reduction remains positive for accepted Armijo step lengths and can be evaluated directly in the reduced coordinates (Appendix F), ensuring that the gain ratio is well defined.

The adequacy monitor ensures that the reduced space captures a prescribed fraction of the gradient, but it does not directly measure whether the subspace captures all curvature directions relevant to an effective LM step. Performance therefore still depends on the quality and diversity of the candidate directions used to construct and enrich the subspace. The hybrid construction used here addresses this issue by combining gradient, memory, Krylov/Lanczos, and randomized curvature information, but other problems may benefit from different or problem-specific candidate directions.

The scaling experiments in Appendix E help clarify a separate limitation related to computational cost. At a fixed trainingset size, HSLM becomes increasingly advantageous as the parameter dimension grows, indicating that the reduced-space construction is particularly effective in reducing parameter-dependent linear-algebra costs. However, these experiments do not examine the effect of increasing the number of training samples. As the training-set size grows, the cost of Jacobian evaluations and matrix-vector products is expected to become increasingly important and is not reduced directly by restricting the optimization step to a low-dimensional parameter subspace. In addition, the present numerical study is limited to neuralnetwork regression problems. Future work should therefore examine mini-batch and matrix-free HSLM variants for larger training sets, as well as applications to large-scale inverse problems with different Jacobian and curvature structures.

Overall, the results demonstrate that adaptive hybrid subspace construction can preserve convergence behavior comparable to full-space LM while substantially reducing the cost of linear-algebra computations as the parameter dimension increases. Together with the convergence analysis, these results suggest that HSLM provides a promising approach for retaining the favorable properties of LM-type methods while improving computational efficiency for large-scale problems.

## APPENDIX

## A IMPLEMENTATION DETAILS FOR THE SUBSPACE LEVENBERG–MARQUARDT METH-ODS

We implemented two subspace variants of the Levenberg–Marquardt method: a Krylov-subspace LM method and HSLM method. For the updating damping strategy, we use delayed gratification for all algorithms by choosing $\rho _ { l o w } = \rho _ { h i g h } = 0$ and $\alpha = 1 0 ^ { - 3 }$ . This appendix summarizes the main subspace construction rules, numerical tolerances, and expansion criteria for each method.

## A.1 Krylov-Subspace Levenberg–Marquardt Method

The Krylov basis is generated by the Lanczos process, initialized with the normalized gradient $q _ { 1 } = g _ { k } / \lVert g _ { k } \rVert _ { 2 }$ , which spans the Krylov subspace

$$
\mathcal { H } _ { p } ( H _ { k } , g _ { k } ) = \operatorname { s p a n } \{ g _ { k } , H _ { k } g _ { k } , \dotsc , H _ { k } ^ { p - 1 } g _ { k } \} .
$$

To reduce computational and memory costs, we avoid explicitly forming the Gauss–Newton matrix, $H _ { k } \nu = J _ { k } ^ { \top } ( J _ { k } \nu )$ , and instead compute the required matrix–vector products directly. The Lanczos iteration is terminated when either the Krylov subspace reaches the prescribed maximum dimension $d _ { \operatorname* { m a x } } = \lfloor 0 . 1 n \rfloor$ or the Lanczos residual satisfie $\beta _ { j } \leq 1 0 ^ { - 5 }$

Let $Q _ { k } = [ q _ { 1 } , \ldots , q _ { d _ { k } } ]$ denote the resulting orthonormal Krylov basis and $T _ { k } = Q _ { k } ^ { \top } H _ { k } Q _ { k }$ the projected tridiagonal matrix. Since the first Lanczos vector is the normalized gradient,

$$
Q _ { k } ^ { \top } g _ { k } = \| g _ { k } \| _ { 2 } e _ { 1 } ,
$$

where $e _ { 1 } = [ 1 , 0 , \ldots , 0 ] ^ { \top } \in \mathbb { R } ^ { d _ { k } }$ is the first canonical basis vector. Consequently, the reduced Levenberg–Marquardt system is

$$
( T _ { k } + \mu _ { k } { \bf I } ) z _ { k } = - \| g _ { k } \| _ { 2 } e _ { 1 } ,
$$

and the full-space step is recovered as $s _ { k } = Q _ { k } z _ { k }$ . If a trial step is rejected, the previously computed Krylov basis is retained and only the reduced system is solved with the updated damping parameter. This avoids recomputing the Jacobian and repeating the Lanczos process, thereby reducing the computational cost of rejected iterations.

## A.2 Adaptive Hybrid Reduced-Subspace Levenberg–Marquardt Method

At iteration k, the hybrid method constructs a reduced orthonormal basis $V _ { k } \in \mathbb { R } ^ { n \times p }$ by combining Lanczos–Krylov directions with stochastic curvature probes. The Lanczos process is initialized with the normalized gradient $q _ { 1 } = g _ { k } / \lVert g _ { k } \rVert _ { 2 }$ , and uses implicit Gauss–Newton matrix–vector products, $H _ { k } \nu = J _ { k } ^ { \top } ( J _ { k } \nu )$ . The initial candidate subspace consists of ⌊0.01n

random curvature probes and recent accepted step $s k _ { - 1 }$ . Each random probe is generated as

$$
\begin{array} { r } { d _ { j } = J _ { k } ^ { \top } J _ { k } \omega _ { j } , \qquad \omega _ { j } \sim \mathcal { N } ( 0 , I ) , } \end{array}
$$

normalized, and appended to the candidate matrix. The combined candidate matrix is then orthogonalized using a thin QR factorization, with linearly dependent directions removed. The quality of the reduced basis is measured by the fraction of the squared gradient norm captured by the subspace. If this quantity satisfies $\eta _ { k } ^ { \mathrm { s u b } } \geq 0 . 9 9$ , the current basis is accepted. Otherwise, the subspace is expanded by adding ⌊0.02n⌋ Lanczos directions and $\left\lfloor 0 . 0 1 n \right\rfloor$ random curvature probes.

Subspace expansion continues until either the target $\eta _ { k } ^ { \mathrm { s u b } } \geq 0 . 9 9$ is achieved or the maximum dimension reaches ⌊0.1n⌋. Since linearly dependent vectors are discarded during QR orthogonalization, the final numerical dimension of $V _ { k }$ may be smaller than the total number of generated candidate directions.

The hybrid method also employs backtracking, starting with $t _ { k } = 1$ and updating $t _ { k } \gets \beta t _ { k }$ , with at most ten trial step lengths evaluated, which is an implementation safeguard outside the theorem. The algorithm is terminated when the reduction in the objective function satisfies $F ( x _ { k } ) - F ( x _ { k + 1 } ) < 0 . 0 0 5$ , or when the maximum number of 10000 iterations is reached.

## A.3 Summary of Numerical Settings

Table 3. Numerical settings used for the Krylov and adaptive hybrid reduced-subspace LM methods.

<table><tr><td>Setting</td><td>Krylov LM</td><td>Hybrid reduced-subspace LM</td></tr><tr><td>Initial Krylov directions</td><td>Up to maximum</td><td></td></tr><tr><td>Initial random probes</td><td></td><td>0.01n</td></tr><tr><td>Krylov directions per expansion</td><td></td><td>0.02n</td></tr><tr><td>Random probes per expansion</td><td></td><td>0.01n</td></tr><tr><td>Adequacy threshold</td><td></td><td> $\eta _ { \mathrm { m i n } } = 0 . 9 9$ </td></tr><tr><td>Lanczos tolerance</td><td> $1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>QR dependence tolerance</td><td></td><td> $1 0 ^ { - 1 2 }$ </td></tr><tr><td>Reduced singular-value floor</td><td></td><td> $1 0 ^ { - 8 }$ </td></tr><tr><td>Initial damping μ</td><td>10</td><td>10</td></tr><tr><td>Damping decrease  $\mu _ { d o w n }$ </td><td>2</td><td>2</td></tr><tr><td>Damping increase</td><td>5</td><td>5</td></tr><tr><td>Step stopping tolerance</td><td>0.005</td><td>0.005</td></tr><tr><td>Maximum iterations</td><td>10000</td><td>10000</td></tr></table>

## B TRAINING AND VALIDATION RESULTS FOR NETWORK 1 & 2

Five representative training results obtained from Network 1 & 2 using the same initial guess demonstrate comparable convergence behavior for the three algorithms. A1-A2. Levenberg–Marquardt algorithm. B1-B2. Krylov subspace Levenberg–Marquardt algorithm. C1-C2. Hybrid subspace Levenberg–Marquardt algorithm. The gray dashed line indicates the variance of the added noise, $\sigma ^ { 2 } \approx 0 . 6 2 2$

## B.1 Network 1

![](images/13533ef2948253666187780997ce3ad1b655cf5b204afa3369f88bf8cd23e9b6.jpg)

## C ADAPTIVE SUBSPACE DIMENSION SELECTION USING DETERMINISTIC ADEQUACY MONITOR

![](images/71e941592f992f79a4320404a5709d2b61547dc87ab5f1a53e3ba72ddf04b3d9.jpg)  
The figure compares the subspace dimensions selected by Krylov Subspace LM and HSLM across iterations for three neural networks of increasing complexity. In the upper row, each panel shows a representative optimization run using the same initial guess for the corresponding network. In the lower row, each panel displays the corresponding projected-gradient energy, $\eta _ { \mathbf { k } } ^ { \mathrm { s u b } }$ , generated by HSLM from the same initial guess, illustrating how failure of the adequacy criterion triggers subspace enrichment during optimization. In contrast to Krylov-subspace LM, which uses a fixed subspace equal to 10% of the parameter dimension, HSLM dynamically adjusts the subspace dimension and generally satisfies the adequacy criterion using a substantially smaller subspace.

## D INFLUENCE OF CANDIDATE BASIS SOURCES IN SUBSPACE CONSTRUCTION ON AL-GORITHM PERFORMANCE

To examine how different sources of basis candidates in subspace construction influence the performance of the proposed method, we use the experimental setup described above for Network 3 to compare three difference subspace construction strategies: randomized Hessian probes only, randomized Hessian probes and gradient direction, and HSLM.

![](images/f3c0814311cb5828fd37bb96dd36e796bc2415ce7c889f27fa6758dee6a71767.jpg)

![](images/19300a62eeff1c9e18f49a9a163a6be1939c11f2004152d558b26cb9cd5074d4.jpg)

![](images/b1db0705016d94e5becf5ba06c9fc7a1182de75dc77e01903c5a7d543fd105fd.jpg)

![](images/cae60fda6f5ae7c3a1dfb4fef833bf17b3808c0b03aba72ef1cb0471632dbb5c.jpg)

![](images/bcd7d5067198543115ccb62c20dc4d439fb51e26cb2dec36c1ad8732b23548a7.jpg)

![](images/385f80a028392234b250b9e08ba100fc9bd6f8324af784c1f186589dd0dab2c1.jpg)  
Five representative training results obtained from Network 3 using the same initial guess demonstrate comparable convergence behavior for the three algorithms. A1-A2. Randomized Hessian/Gauss–Newton matrix–vector only B1-B2. Randomized Hessian/Gauss–Newton matrix–vector and gradient direction. C1-C2. HSLM algorithm. The gray dashed line indicates the variance of the added noise, $\sigma ^ { 2 } \approx 0 . 6 2 2$ . The figure compares training and validation errors for Network 3 using five representative initial guesses. The purely randomized Hessian method, whose subspace is constructed only from randomized Hessian probes, initially reduces error but struggles to reach the added noise level, indicating that the sampled curvature directions alone may omit important descent information. Adding the gradient direction produces more reliable convergence behavior toward a local optimal solution, but the reduction in both training and validation error remains slow. Meanwhile, HSLM achieves the strongest overall performance, reaching the lowest training and validation errors more rapidly and consistently across the five representative runs. These results demonstrate that augmenting randomized curvature information with additional structured directions improves the convergence behavior, with the full HSLM construction providing the strongest performance among the three variants considered.

## E COMPUTATIONAL EFFICIENCY OF HSLM IN INCREASING PARAMETER DIMENSION AT FIXED TRAINING-SET SIZE

We evaluate the efficiency of the proposed HSLM algorithm by comparing it with the hybrid Krylov-subspace and classical Levenberg–Marquardt algorithms on overparameterized feedforward neural networks. The experimental setup is identical to that described previously, using the same 40,000 training samples. To simulate measurement uncertainty, additive Gaussian noise with a standard deviation equal to 5% of the standard deviation of the target signal is added to the training data. The average number of iterations and the average execution time per iteration over 50 independent trials are summarized in Table 4.

Table 4. Comparison between algorithms for neural-network training
<table><tr><td rowspan="2"></td><td colspan="2">Classical LM</td><td colspan="2">Krylov Subspace LM</td><td colspan="2">HSLM</td></tr><tr><td>Iteration</td><td>Time / Iteration (s)</td><td>Iteration</td><td>Time / Iteration (s)</td><td>Iteration</td><td>Time / Iteration (s)</td></tr><tr><td>Network 1</td><td>43.9</td><td>1.044</td><td>45.3</td><td>0.941</td><td>49.8</td><td>0.464</td></tr><tr><td>Network 2</td><td>38.2</td><td>3.235</td><td>37.1</td><td>3.790</td><td>37.1</td><td>1.221</td></tr><tr><td>Network 3</td><td>34.9</td><td>15.830</td><td>36.5</td><td>13.082</td><td>35.4</td><td>3.665</td></tr></table>

For all three network architectures, the final training error is approximately equal to the variance of the additive noise, indicating that each algorithm successfully captures the underlying Friedman function and that the remaining error is primarily due to irreducible noise. While all three methods exhibit comparable convergence behavior, the proposed HSLM algorithm scales more effectively as the network size increases. Compared with the classical LM algorithm, HSLM reduces the execution time by factors of approximately 2.3, 2.6, and 4.3 for Networks 1, 2, and 3, respectively. Relative to the Krylov-subspace LM algorithm, HSLM achieves additional speedups of approximately 2.0, 3.1, and 3.6 for Networks 1, 2, and 3, respectively, demonstrating its enhanced computational performance for large overparameterized networks.

## F CLOSED-FORM PREDICTED REDUCTION

Lemma 5. Let $m _ { k }$ be defined by (9) and $s _ { k }$ by (6). Then for any $t \in ( 0 , 2 )$ ,

$$
m _ { k } ( 0 ) - m _ { k } ( t s _ { k } ) = \frac { 1 } { 2 } t ( 2 - t ) y _ { k } ^ { \top } B _ { k } y _ { k } > 0 \qquad f o r \ a l l \ t \in ( 0 , 2 )\tag{16}
$$

Proof. Using orthogonality of $V _ { k }$ and $Z _ { k }$ , the local quadratic approximation (9) restricted to this subspace becomes

$$
\begin{array} { l } { m _ { k } ( t s _ { k } ) = \displaystyle \frac 1 2 \| r _ { k } + t J _ { k } s _ { k } \| ^ { 2 } + \frac { \mu _ { k } } { 2 } t ^ { 2 } \| D _ { k , p } ^ { 1 / 2 } Z _ { k } ^ { \top } V _ { k } ^ { \top } s _ { k } \| ^ { 2 } } \\ { = \displaystyle \frac 1 2 \| r _ { k } + t J _ { k } ( V _ { k } Z _ { k } y _ { k } ) \| ^ { 2 } + \frac { \mu _ { k } } { 2 } t ^ { 2 } \| D _ { k , p } ^ { 1 / 2 } Z _ { k } ^ { \top } V _ { k } ^ { \top } ( V _ { k } Z _ { k } y _ { k } ) \| ^ { 2 } } \\ { = \displaystyle \frac 1 2 \| r _ { k } + t ( J _ { k } V _ { k } ) Z _ { k } y _ { k } \| ^ { 2 } + \frac { \mu _ { k } } { 2 } t ^ { 2 } \| D _ { k , p } ^ { 1 / 2 } y _ { k } \| ^ { 2 } } \\ { = \displaystyle \frac 1 2 \| r _ { k } + t J _ { k , p } Z _ { k } y _ { k } \| ^ { 2 } + \frac { \mu _ { k } } { 2 } t ^ { 2 } y _ { k } ^ { \top } D _ { k , p } y _ { k } . } \end{array}
$$

Since $J _ { k , p } Z _ { k } = U _ { k } \Sigma _ { k , p } ,$ define $q _ { k } : = U _ { k } ^ { \top } r _ { k }$ . Then

$$
\| r _ { k } + t U _ { k } \Sigma _ { k , p } y _ { k } \| ^ { 2 } = \| r _ { k } \| ^ { 2 } + 2 t y _ { k } ^ { \top } \Sigma _ { k , p } q _ { k } + t ^ { 2 } y _ { k } ^ { \top } \Sigma _ { k , p } ^ { 2 } y _ { k } ,
$$

using orthogonality of $U _ { k }$ . Hence

$$
m _ { k } ( t s _ { k } ) = \frac { 1 } { 2 } \| r _ { k } \| ^ { 2 } + t y _ { k } ^ { \top } \Sigma _ { k , p } q _ { k } + \frac { 1 } { 2 } t ^ { 2 } y _ { k } ^ { \top } ( \Sigma _ { k , p } ^ { 2 } + \mu _ { k } D _ { k , p } ) y _ { k } .
$$

Using the reduced LM system $( 6 ) , \Sigma _ { k , p } q _ { k } = - B _ { k } y _ { k }$ , then

$$
m _ { k } ( t s _ { k } ) = \frac 1 2 \| r _ { k } \| ^ { 2 } - t y _ { k } ^ { \top } B _ { k } y _ { k } + \frac 1 2 t ^ { 2 } y _ { k } ^ { \top } B _ { k } y _ { k } = m _ { k } ( 0 ) + \frac 1 2 t ( t - 2 ) y _ { k } ^ { \top } B _ { k } y _ { k } ,
$$

which rearranges to (16). Then,

$$
m _ { k } ( 0 ) - m _ { k } ( t s _ { k } ) = \frac { 1 } { 2 } t ( 2 - t ) y _ { k } ^ { \top } B _ { k } y _ { k } > 0 \quad \mathrm { ~ u n d e r ~ A s s u m p t i o n ~ } 3
$$

since $y _ { k } ^ { \top } B _ { k } y _ { k } > 0$ (Lemma 1) and $t \in ( 0 , 2 )$ ).

## REFERENCES

<sup>[1]</sup> N. Ampazis and S. J. Perantonis. Two highly efficient second-order algorithms for training feedforward networks. IEEE Trans. Neural Netw., 13(5):1064–1074, 2002.

<sup>[2]</sup> L. Armijo. Minimization of functions having lipschitz continuous first partial derivatives. Pacific Journal of Mathematics, 16(1):1–3, 1966.

<sup>[3]</sup> A. R. Barron. Universal approximation bounds for superpositions of a sigmoidal function. IEEE Transactions on Information Theory, 39(3):930–945, 1993.

<sup>[4]</sup> E. K. Bjarkason, O. J. Maclaren, J. P. O’Sullivan, and M. J. O’Sullivan. Randomized truncated svd levenberg–marquardt approach to geothermal natural state and history matching. Water Resources Research, 54(3):2376–2404, 2018.

<sup>[5]</sup> M. Brand. Incremental singular value decomposition of uncertain data with missing values. In European Conference on Computer Vision (ECCV), volume 2350 of Lecture Notes in Computer Science, pages 707–720, 2002.

<sup>[6]</sup> C. Cartis, J. M. Fowkes, and Z. Shao. A randomised subspace gauss-newton method for nonlinear least-squares. CoRR, abs/2211.05727, 2022.

<sup>[7]</sup> A. R. Conn, N. I. M. Gould, and P. L. Toint. Trust Region Methods. MOS-SIAM Ser. Optim. SIAM, Philadelphia, 2000.

<sup>[8]</sup> P. G. Constantine. Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter Studies. SIAM Spotlights. SIAM, Philadelphia, 2015.

<sup>[9]</sup> P. G. Constantine, C. Kent, and T. Bui-Thanh. Accelerating markov chain monte carlo with active subspaces. SIAM Journal on Scientific Computing, 38(5):A2779–A2805, 2016.

<sup>[10]</sup> G. Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals and Systems, 2(4):303–314, 1989.

<sup>[11]</sup> J. E. Dennis, Jr. and R. B. Schnabel. Numerical Methods for Unconstrained Optimization and Nonlinear Equations, volume 16 of Classics in Applied Mathematics. SIAM, Philadelphia, 1996.

<sup>[12]</sup> J. H. Friedman, E. Grosse, and W. Stuetzle. Multidimensional additive spline approximation. SIAM J. Sci. Statist. Comput., 4(2):291–301, 1983.

<sup>[13]</sup> M. T. Hagan and M. B. Menhaj. Training feedforward networks with the marquardt algorithm. IEEE Trans. Neural Netw., 5(6):989–993, 1994.

<sup>[14]</sup> N. Halko, P.-G. Martinsson, and J. A. Tropp. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Review, 53(2):217–288, 2011.

<sup>[15]</sup> K. Hornik, M. Stinchcombe, and H. White. Multilayer feedforward networks are universal approximators. Neural Networks, 2(5):359–366, 1989.

<sup>[16]</sup> X. Huang, X. Liang, Z. Liu, L. Li, Y. Yu, and Y. Li. SPAN: A stochastic projected approximate newton method. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1520–1527, 2020.

<sup>[17]</sup> C.-T. Kim and J.-J. Lee. Training two-layered feedforward networks with variable projection method. IEEE Transactions

on Neural Networks, 19(2):371–375, 2008.

<sup>[18]</sup> K. Levenberg. A method for the solution of certain non-linear problems in least squares. Quarterly of Applied Mathematics, 2(2):164–168, 1944.

<sup>[19]</sup> Y. Lin, D. O’Malley, and V. V. Vesselinov. A computationally efficient parallel Levenberg–Marquardt algorithm for highly parameterized inverse model analyses. Water Resour. Res., 52(9):6948–6977, 2016.

<sup>[20]</sup> K. Madsen, H. B. Nielsen, and O. Tingleff. Methods for Non-Linear Least Squares Problems. Informatics and Mathematical Modelling, Technical University of Denmark, Lyngby, Denmark, 2nd edition, 2004.

<sup>[21]</sup> D. W. Marquardt. An algorithm for least-squares estimation of nonlinear parameters. J. Soc. Indust. Appl. Math., 11(2):431–441, 1963.

<sup>[22]</sup> J. Martens. Deep learning via hessian-free optimization. In Proceedings of the 27th International Conference on Machine Learning, ICML’10, pages 735–742. Omnipress, 2010.

<sup>[23]</sup> J. J. More and D. C. Sorensen. Computing a trust region step. ´ SIAM Journal on Scientific and Statistical Computing, 4(3):553–572, 1983.

<sup>[24]</sup> J. Nocedal and S. J. Wright. Numerical Optimization. Springer Ser. Oper. Res. Financ. Eng. Springer, New York, 2nd edition, 2006.

<sup>[25]</sup> Z. Shao. On Random Embeddings and Their Application to Optimisation. PhD thesis, University of Oxford, Oxford, UK, 2021.

<sup>[26]</sup> O. Vinyals and D. Povey. Krylov subspace descent for deep learning. In Proceedings of the Fifteenth International Conference on Artificial Intelligence and Statistics, volume 22 of Proceedings of Machine Learning Research, pages 1261–1268. PMLR, 2012.

<sup>[27]</sup> C. R. Vogel and J. G. Wade. A modified levenberg–marquardt algorithm for large-scale inverse problems. In K. Bowers and J. Lund, editors, Computation and Control III, volume 15 of Progress in Systems and Control Theory, pages 367–378. Birkhauser, Boston, MA, 1993.¨

<sup>[28]</sup> B. M. Wilamowski and H. Yu. Improved computation for levenberg–marquardt training. IEEE Transactions on Neural Networks, 21(6):930–937, 2010.

<sup>[29]</sup> Y.-x. Yuan. Subspace methods for large scale nonlinear equations and nonlinear least squares. Optimization and Engineering, 10(2):207–218, 2009.