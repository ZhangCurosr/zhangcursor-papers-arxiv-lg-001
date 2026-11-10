# On the Global Convergence of Stochastic Adam for Physics-Informed Neural Networks

Yihang Gao<sup>\*</sup>

## Abstract

The Adam optimizer is widely used to train neural networks with highly nonlinear and nonconvex objective functions. Remarkably, Adam often drives the empirical training loss nearly to zero. This phenomenon has been partially explained by neural tangent kernel analyses of gradient descent (GD) and stochastic gradient descent (SGD). However, analyzing Adam presents additional challenges. Its first-moment momentum couples stochastic gradients across iterations, while its adaptive second-moment preconditioner depends on the same stochastic gradient used in the update. In this paper, we establish linear convergence guarantees for Adam in training sufficiently wide two-layer neural networks with either full-batch or minibatch gradients. For full-batch Adam, we construct a Lyapunov function to control the effect of historical momentum. For minibatch Adam, our analysis builds on a virtual-iterate representation, a conditional analysis of the dependence between stochastic gradients and adaptive preconditioners, and uniform concentration bounds for the second-moment estimates. We prove geometric decay of the empirical loss for full-batch Adam and in restricted expectation on a high-probability event for minibatch Adam. Beyond regression, we extend our convergence analysis to physics-informed neural networks (PINNs) for solving the Poisson equation, where differential operators and separate sampling of interior and boundary points introduce additional challenges. We establish global convergence guarantees for both deterministic and stochastic Adam under explicit conditions on network width and Adam’s parameters, characterizing the effects of minibatch size on convergence. These results help bridge the gap between Adam’s empirical success and its theoretical understanding, providing sharper convergence guarantees for PINNs than existing analyses.

## 1 Introduction

The Adam optimizer [22] has been widely used to train deep neural networks in various applications, including computer vision [21], natural language processing [35], and scientific machine learning [37]. Despite its empirical success, the theoretical understanding of Adam remains incomplete. In practice, Adam can often reduce the nonnegative training loss to nearly zero [4,8,22], suggesting that it can approach a global minimizer even for highly nonlinear and nonconvex objectives. However, classical optimization theory generally guarantees convergence to stationary points rather than global minimizers for nonconvex problems [6, 15]. This gap between empirical observations and existing theoretical guarantees motivates a closer study of the global convergence of Adam in neural network training.

One influential approach to understanding this phenomenon is based on the neural tangent kernel (NTK) framework [16], which describes the training dynamics of sufficiently wide neural networks through their tangent kernels. Based on this framework, several studies have established global convergence guarantees for gradient descent (GD) [9, 10, 28] and stochastic gradient descent (SGD) [1, 14, 25, 36, 39, 43] under suitable overparameterization conditions. These results show that sufficiently wide neural networks can maintain favorable local regularity properties throughout training, leading to convergence to global minima. However, most existing finite-width analyses focus on GD and SGD, and their techniques do not directly extend to Adam. Unlike GD and SGD, Adam uses exponentially weighted moving averages of gradients and squared gradients, which introduce dependence across iterations and adaptive preconditioning. In addition, the stochastic gradient is correlated with the adaptive preconditioner, making the conditional unbiasedness arguments used for SGD no longer directly applicable. These challenges motivate a separate convergence analysis for Adam.

The convergence properties of Adam and related adaptive optimization methods have also received considerable attention [6, 7, 31]. Existing studies have established convergence guarantees for convex and nonconvex problems, including convergence to stationary points [2, 23] and linear convergence under suitable Polyak–Łojasiewicz (PL) conditions [5]. Although these results provide theoretical insights, they do not fully explain why Adam can reduce the training loss to nearly zero when training neural networks. In particular, convergence to a stationary point does not imply global optimality, while the global PL condition is generally difficult to verify for highly nonlinear neural network objectives. Several studies have also established global convergence guarantees for Adam in specific neural network models [32, 42]. However, the global convergence of standard Adam in training finite-width neural networks, particularly under stochastic sampling, remains less understood.

Beyond standard regression problems, physics-informed neural networks (PINNs) [29] have become a popular approach for solving partial differential equations [20]. Unlike standard regression, PINN training involves differential operators acting on neural network outputs, together with boundary or initial conditions. These features introduce additional challenges in convergence analysis. Recent studies have extended NTK-based convergence theory to PINNs, establishing global convergence guarantees for GD [12], SGD [17], natural gradient descent [41], and implicit gradient descent [24, 40]. Related convergence results have also been established for physics-informed Kolmogorov–Arnold networks [13]. However, these studies mainly focus on optimization methods other than Adam. Although the training dynamics of Adam for PINNs have also been studied [11], global convergence guarantees for Adam remain limited, particularly under stochastic sampling.

To address these challenges, we study the global convergence of Adam in training two-layer neural networks for regression and physics-informed learning. We establish linear convergence guarantees under suitable conditions on the momentum parameters, damping parameter, step size, network width, and minibatch size. Our results provide a theoretical explanation for the empirical effectiveness of Adam in these settings. The main contributions are summarized as follows:

1. We first study deterministic Adam for neural network regression. We establish global linear convergence for any fixed choice of momentum parameters, provided that the network is sufficiently wide and the other algorithmic parameters satisfy suitable conditions. Our proof uses a Lyapunov function to control the effect of historical momentum and establishes NTK stability throughout training.

2. We extend the convergence analysis to stochastic Adam with minibatch sampling for regression. In this setting, additional conditions on the algorithmic parameters are needed to control the stochastic errors introduced by momentum and adaptive preconditioning. We develop a virtual-iterate analysis to handle historical momentum and a conditional analysis to control the bias arising from the dependence between stochastic gradients and adaptive preconditioners. Our results establish geometric decay of the empirical loss in restricted expectation on a high-probability event, under explicit conditions on network width and Adam’s parameters. These conditions and the convergence rate also characterize the dependence on minibatch size.

3. We further extend our analysis to physics-informed neural networks (PINNs) for solving the Poisson equation. The presence of differential operators and separate sampling of interior and boundary points introduces additional difficulties in controlling the training trajectory. By extending our convergence analysis to handle these difficulties, we establish analogous linear convergence guarantees with explicit width requirements and conditions on Adam’s parameters, including the dependence on the PDE and boundary minibatch sizes.

Notation. Throughout this paper, we use $\| \cdot \|$ to denote the Euclidean norm for vectors and the spectral norm for matrices. For any event A, the indicator function $\mathbf { 1 } _ { A }$ equals one if A occurs and zero otherwise, and $\mathcal { A } ^ { c }$ denotes its complement. Given a positive integer n, [n] is defined as the set $\{ 1 , 2 , \cdots , n \}$ . We write $a \lesssim b$ (respectively, $a \gtrsim b )$ if there exists a universal constant $C > 0$ such that $a \leq C b$ (respectively, $a \geq C b )$ . We write $a \asymp b$ if both $a \lesssim b$ and $a \gtrsim b$ hold. We use $\widetilde \Omega$ and $\widetilde { \Theta }$ analogously $\mathbf { t o } \gtrsim$ and $\asymp ,$ respectively, with polylogarithmic factors suppressed.

## 2 Convergence Analysis of Adam for Regression

Consider a regression problem with distinct training samples $x _ { 1 } , \ldots , x _ { n } \in \mathbb { R } ^ { d }$ and corresponding targets $y _ { 1 } , \ldots , y _ { n } \in \mathbb { R }$ . We assume that $\| x _ { i } \| \leq 1$ and $| y _ { i } | \le 1$ for all $i = 1 , \ldots , n$ . The regression problem is formulated as

$$
\operatorname* { m i n } _ { \theta } L ( \theta ) = \frac { 1 } { 2 n } \sum _ { i = 1 } ^ { n } ( f _ { \theta } ( x _ { i } ) - y _ { i } ) ^ { 2 } ,\tag{1}
$$

where $f _ { \theta }$ denotes a neural network parameterized by θ, which contains all trainable parameters.

Throughout this paper, we consider a two-layer feedforward neural network of the form

$$
f _ { \theta } ( x ) = \frac { 1 } { \sqrt { m } } \sum _ { r = 1 } ^ { m } a _ { r } \sigma ( w _ { r } ^ { T } \widetilde { x } ) , \qquad \theta = ( a _ { 1 } , w _ { 1 } , \ldots , a _ { m } , w _ { m } ) ,\tag{2}
$$

where $\widetilde { x } = ( x , 1 ) / \sqrt { 2 }$ is the augmented input vector that includes a bias term, $a _ { r } \in \mathbb { R }$ and $w _ { r } \in \mathbb { R } ^ { d + 1 }$ are trainable parameters, and $\sigma ( \cdot ) \ = \ \operatorname { t a n h } ( \cdot )$ is the hyperbolic tangent activation function. The network parameters are independently initialized as

$$
w _ { r } ^ { ( 0 ) } \sim \mathcal { N } ( 0 , I _ { d + 1 } ) , \qquad a _ { r } ^ { ( 0 ) } \sim \mathrm { U n i f o r m } ( \{ - 1 , 1 \} ) ,\tag{3}
$$

for $r = 1 , \ldots , m$ . Following previous studies [10, 12, 17], we adopt Rademacher initialization for the output weights $a _ { r }$ for analytical convenience. Our analysis can also be extended to more general initialization distributions.

We study the convergence of Adam applied to Problem (1). Let $\theta ^ { ( t ) } = ( a _ { 1 } ^ { ( t ) } , w _ { 1 } ^ { ( t ) } , \ldots , a _ { m } ^ { ( t ) } , w _ { m } ^ { ( t ) } )$ denote the network parameters at iteration t. At each iteration, Adam [22] computes a gradient

estimate $g ^ { ( t ) }$ of $\nabla _ { { \boldsymbol { \theta } } } L ( { \boldsymbol { \theta } } ^ { ( t ) } )$ . Given momentum parameters $\beta _ { 1 } , \beta _ { 2 } \in [ 0 , 1 )$ , a damping parameter $\epsilon > 0$ and a stepsize $\eta > 0$ , the moment estimates and their bias-corrected counterparts are computed as

$$
\begin{array} { r l } & { \boldsymbol { s } ^ { ( t ) } = \beta _ { 1 } \boldsymbol { s } ^ { ( t - 1 ) } + ( 1 - \beta _ { 1 } ) \boldsymbol { g } ^ { ( t ) } , \quad \boldsymbol { v } ^ { ( t ) } = \beta _ { 2 } \boldsymbol { v } ^ { ( t - 1 ) } + ( 1 - \beta _ { 2 } ) \boldsymbol { g } ^ { ( t ) \odot 2 } , } \\ & { \boldsymbol { \widehat { s } } ^ { ( t ) } = \cfrac { \boldsymbol { s } ^ { ( t ) } } { 1 - \boldsymbol { \beta } _ { 1 } ^ { t + 1 } } , \quad \boldsymbol { \widehat { v } } ^ { ( t ) } = \cfrac { \boldsymbol { v } ^ { ( t ) } } { 1 - \boldsymbol { \beta } _ { 2 } ^ { t + 1 } } , \quad \boldsymbol { D } ^ { ( t ) } = \mathrm { d i a g } \left( \cfrac { 1 } { \sqrt { \boldsymbol { v } ^ { ( t ) } } + \epsilon } \right) , } \end{array}\tag{4}
$$

where $s ^ { ( - 1 ) } = v ^ { ( - 1 ) } = 0$ . Here, $s ^ { ( t ) }$ and $v ^ { ( t ) }$ are the first- and second-moment estimates, respectively, and $\widehat { s } ^ { ( t ) }$ and $\widehat { v } ^ { ( t ) }$ are their bias-corrected counterparts. The matrix $D ^ { ( t ) }$ is the adaptive preconditioner. All powers, square roots, and divisions involving vectors are performed coordinatewise. The network parameters are then updated by

$$
\boldsymbol { \theta } ^ { ( t + 1 ) } = \boldsymbol { \theta } ^ { ( t ) } - \eta D ^ { ( t ) } \boldsymbol { \widehat { s } } ^ { ( t ) } .\tag{5}
$$

For a network of width $m ,$ , we choose the damping parameter and stepsize as

$$
\epsilon = \frac { \bar { \epsilon } } { \sqrt { m } } > 0 , \qquad \eta = \bar { \eta } \epsilon > 0 .\tag{6}
$$

Our main results provide guidelines for choosing $\bar { \epsilon }$ and $\bar { \eta } .$

Throughout our analysis, we assume that the Adam parameters $\beta _ { 1 } , \beta _ { 2 } , \epsilon$ , and $\eta$ are fixed across iterations, although they may depend on the training data, network width, and other problem parameters. For deterministic Adam, our convergence results allow arbitrary fixed $\beta _ { 1 } , \beta _ { 2 } \in [ 0 , 1 )$ . For stochastic Adam with minibatch sampling, we allow any fixed $\beta _ { 1 } \in [ 0 , 1 )$ , while $\beta _ { 2 } \in ( 0 , 1 )$ must be sufficiently close to one. The precise conditions on these parameters are specified in the main theorems. Unlike GD and SGD, Adam uses both first-moment momentum and second-momentbased adaptive preconditioning, as shown in (4). The resulting dependence on historical gradients and adaptive preconditioners introduces additional challenges to the convergence analysis. Consequently, existing techniques for GD and SGD cannot be directly applied.

The convergence analysis of Adam relies on the properties of the loss function $L ( \theta )$ . To study these properties, we define the normalized residuals

$$
R _ { i } ( \theta ) = \frac { f _ { \theta } ( x _ { i } ) - y _ { i } } { \sqrt { n } } , \qquad L ( \theta ) = \frac { 1 } { 2 } \| R ( \theta ) \| ^ { 2 } ,
$$

where $R ( \theta ) = ( R _ { 1 } ( \theta ) , \ldots , R _ { n } ( \theta ) ) ^ { \top } \in \mathbb { R } ^ { n }$ is the normalized residual vector. We further define

$$
J ( \theta ) = \nabla R ( \theta ) , \qquad K ( \theta ) = J ( \theta ) J ( \theta ) ^ { \top } ,\tag{7}
$$

where $J ( \theta ) \in \mathbb { R } ^ { n \times m ( d + 2 ) }$ is the Jacobian matrix of $R ( \theta )$ and $K ( \theta ) \in \mathbb { R } ^ { n \times n }$ is the associated Gram matrix. Since $\nabla L ( \theta ) = J ( \theta ) ^ { \top } R ( \theta )$ , we have

$$
\| \nabla L ( \theta ) \| ^ { 2 } = R ( \theta ) ^ { \top } K ( \theta ) R ( \theta ) \geq 2 \lambda _ { \operatorname* { m i n } } ( K ( \theta ) ) L ( \theta ) .\tag{8}
$$

This inequality relates the positive definiteness of the Gram matrix to the Polyak–Łojasiewicz (PL) inequality [19, 28]. In particular, if $\lambda _ { \operatorname* { m i n } } ( K ( \theta ) )$ remains uniformly bounded away from zero along the optimization trajectory, then the PL inequality holds along the trajectory. This property is important for establishing linear convergence of Adam. Our analysis therefore focuses on maintaining a positive lower bound on the minimum eigenvalue of the Gram matrix throughout training.

To this end, we define the limiting Gram matrix associated with the random initialization (3) as

$$
K _ { \infty } = \mathbb { E } _ { \theta ^ { ( 0 ) } } [ K ( \theta ^ { ( 0 ) } ) ] , \qquad \lambda _ { 0 } = \lambda _ { \operatorname* { m i n } } ( K _ { \infty } ) .\tag{9}
$$

Although $K ( \theta )$ changes during training, we will show that it remains close to its initial value, which can be controlled through $K _ { \infty }$ . The following lemma establishes the positive definiteness of $K _ { \infty }$

Lemma 1 (Positive definiteness of $K _ { \infty } )$ . The limiting Gram matrix $K _ { \infty }$ is positive definite, $i . e .$ $\lambda _ { 0 } > 0$

The proof is provided in Appendix B.1 and uses the full support of the Gaussian distribution and properties of the tanh activation function. Similar positive definiteness results have been established in the NTK literature [3, 9, 10]. However, extending the convergence analysis from GD and SGD to Adam requires new techniques to handle momentum and adaptive preconditioning, which constitute the main theoretical contributions of this work.

## 2.1 Convergence of Deterministic Adam

We first study the convergence of deterministic Adam, where the exact gradient $g ^ { ( t ) } = \nabla _ { \theta } L ( \theta ^ { ( t ) } )$ is used in (4) at each iteration. This setting allows us to study the effects of momentum and adaptive preconditioning without the randomness introduced by minibatch sampling. Nevertheless, the convergence analysis remains challenging due to the dependence on historical gradients and the time-varying preconditioner. Our first main result establishes linear convergence of deterministic Adam under suitable conditions on the network width and algorithmic parameters.

Theorem 1 (Regression with full gradients). Let $\delta \in ( 0 , 1 )$ . Consider Adam defined by $( 4 ) \mathrm { - } ( 5 )$ with full gradients $g ^ { ( t ) } = \nabla L ( \theta ^ { ( t ) } )$ and arbitrary fixed momentum parameters $\beta _ { 1 } , \beta _ { 2 } \in [ 0 , 1 )$ . Suppose that

$$
\bar { \epsilon } \gtrsim \sqrt { \log \frac { 2 } { \delta } } , \qquad 0 < \bar { \eta } \lesssim 1 - \beta _ { 1 } ,\tag{10}
$$

and that the network width satisfies

$$
m \gtrsim \frac { 1 } { \lambda _ { 0 } } \log \frac { n } { \delta } + \frac { 1 } { \lambda _ { 0 } ^ { 2 } } \log \frac { 2 } { \delta } .\tag{11}
$$

Then, with probability at least $1 - \delta$ over the random initialization (3), the Adam iterates satisfy

$$
L ( \theta ^ { ( t ) } ) \leq ( 1 - c \bar { \eta } \lambda _ { 0 } ) ^ { t } L ( \theta ^ { ( 0 ) } ) , \qquad t \geq 0 ,\tag{12}
$$

where $c > 0$ is a universal constant. Moreover, the parameter sequence $\{ \theta ^ { ( t ) } \}$ converges to a finite limit $\theta ^ { \star }$ satisfying $L ( \theta ^ { \star } ) = 0 .$

Proof sketch. The complete proof is provided in Appendix B. The proof consists of three main steps: concentration at initialization, stability of the Gram matrix, and a Lyapunov-based descent analysis.

First, under the network width condition (11), concentration inequalities imply that

$$
K ( \theta ^ { ( 0 ) } ) \stackrel { < } { = } \frac { 3 } { 4 } \lambda _ { 0 } I , \qquad L _ { 0 } : = L ( \theta ^ { ( 0 ) } ) \stackrel { < } { \sim } \log \frac { 2 } { \delta } ,
$$

with probability at least $1 - \delta .$ . As long as the output coefficients remain bounded, the residual Jacobian $J ( \theta )$ is Lipschitz continuous with constant $C / \sqrt { m }$ . Consequently,

$$
K ( \theta ) \succeq \frac { \lambda _ { 0 } } { 4 } I
$$

for all θ satisfying

$$
\begin{array} { r } { \| \theta - \theta ^ { ( 0 ) } \| \leq \mathcal { R } : = c _ { 0 } \sqrt { m \lambda _ { 0 } } , } \end{array}
$$

provided that the output coefficients remain bounded, where $c _ { 0 } > 0$ is a sufficiently small universal constant. By (8), this yields a uniform PL inequality in this neighborhood.

Next, we control the difference between the bias-corrected first-moment estimate $\widehat { s } ^ { ( t ) }$ and the current gradient $g ^ { ( t ) }$ . Define the normalized preconditioner $P ^ { ( t ) } = \epsilon D ^ { ( t ) }$ . As long as the output weights remain bounded and the loss does not exceed $L _ { 0 } ,$ , the coordinatewise gradient bounds and the damping condition imply that ${ \textstyle \frac { 1 } { 2 } } I \preceq P ^ { ( t ) } \preceq I$ . Combining the descent inequality for the loss function with the recurrence relation for $\widehat { s } ^ { ( t ) } - g ^ { ( t ) }$ , we construct the Lyapunov function

$$
\mathcal { E } ^ { ( t ) } : = L ( \theta ^ { ( t ) } ) + \frac { 4 \bar { \eta } } { 1 - \beta _ { 1 } } \Vert \widehat { s } ^ { ( t ) } - g ^ { ( t ) } \Vert ^ { 2 } ,
$$

which satisfies

$$
\mathcal { E } ^ { ( t + 1 ) } \leq ( 1 - c \bar { \eta } \lambda _ { 0 } ) \mathcal { E } ^ { ( t ) } .
$$

This contraction follows from the local lower bound on the Gram matrix and the smoothness of the loss function along each update segment.

To ensure that the preceding conditions hold throughout training, we establish two bounds on the cumulative parameter movement:

$$
\begin{array} { r } { \displaystyle \sum _ { t = 0 } ^ { T } \| \theta ^ { ( t + 1 ) } - \theta ^ { ( t ) } \| \lesssim \sqrt { \frac { L _ { 0 } } { \lambda _ { 0 } } } , } \\ { \displaystyle \sum _ { t = 0 } ^ { T } \displaystyle \operatorname* { m a x } _ { 1 \leq r \leq m } \| \theta _ { r } ^ { ( t + 1 ) } - \theta _ { r } ^ { ( t ) } \| \lesssim \frac { \sqrt { L _ { 0 } } } { \lambda _ { 0 } \sqrt { m } } , } \end{array}
$$

where ${ \theta _ { r } } = \left( { a _ { r } } , { w _ { r } } \right)$ denotes the parameters of the r-th neuron. The first bound follows from the Lyapunov descent analysis, while the second uses the exact bias-corrected moment weights. When m $\gtrsim L _ { 0 } / \lambda _ { 0 } ^ { 2 }$ , these bounds ensure that the iterates remain in the region where the Gram matrix is stable, and the output weights remain bounded. The bound on cumulative neuron-wise movement also ensures that each update segment stays in the region where the smoothness conditions hold. These properties allow us to close the induction and establish the contraction for all iterations.

Finally, the bias correction gives $\widehat s ^ { ( 0 ) } = g ^ { ( 0 ) }$ , so $\mathcal { E } ^ { ( 0 ) } = L _ { 0 }$ . Combining this identity with the contraction of $\mathcal { E } ^ { ( t ) }$ and the bound $L ( \theta ^ { ( t ) } ) \leq \mathcal { E } ^ { ( t ) }$ yields the claimed linear convergence rate. □

Discussion. Theorem 1 establishes linear convergence of deterministic Adam for arbitrary fixed momentum parameters $\beta _ { 1 } , \beta _ { 2 } \in [ 0 , 1 )$ , provided that the damping parameter, stepsize, and network width satisfy the conditions (10)–(11). The first-moment parameter $\beta _ { 1 }$ affects the admissible normalized stepsize through $\bar { \eta } \lesssim 1 - \beta _ { 1 }$ . Thus, a larger $\beta _ { 1 }$ requires a smaller stepsize under our conditions. We emphasize that our result guarantees convergence with fixed momentum parameters but does not establish any acceleration due to momentum.

The network width requirement in (11) consists of two terms with different roles. The term $\lambda _ { 0 } ^ { - 1 } \log ( n / \delta )$ ensures a positive lower bound on the minimum eigenvalue of the Gram matrix at initialization, while $\lambda _ { 0 } ^ { - 2 } \log ( 2 / \delta )$ controls parameter movement and maintains this lower bound throughout training. For fixed training data, $\lambda _ { 0 }$ is independent of the network width $m$ , and the normalized stepsize $\bar { \eta }$ can also be chosen independently of $m .$ . Therefore, once the width requirement is satisfied, the contraction factor $1 - c \bar { \eta } \lambda _ { 0 }$ does not depend on $m .$

The actual stepsize $\eta = \bar { \eta } \bar { \epsilon } / \sqrt { m }$ decreases as the network width m increases, while the contraction factor does not explicitly depend on $m .$ . To explain this, recall the normalized preconditioner

$P ^ { ( t ) } = \epsilon D ^ { ( t ) }$ , which allows us to rewrite the Adam update as

$$
\boldsymbol { \theta } ^ { ( t + 1 ) } = \boldsymbol { \theta } ^ { ( t ) } - \bar { \eta } P ^ { ( t ) } \widehat { s } ^ { ( t ) } .
$$

Under the gradient bounds and the damping condition, the eigenvalues of $P ^ { ( t ) }$ remain uniformly bounded above and away from zero throughout training. Thus, the width-dependent scaling of η compensates for the adaptive preconditioning, and the convergence rate is determined by the normalized stepsize η¯ rather than η.

## 2.2 Convergence of Stochastic Adam

In practice, computing the full gradient at each iteration can be expensive for large training datasets. Therefore, Adam is commonly implemented using stochastic gradients computed from minibatches. In this section, we extend our analysis to stochastic Adam and establish linear convergence guarantees under suitable conditions on the algorithmic parameters and network width.

At each iteration t, we sample a minibatch $B ^ { ( t ) } \subset [ n ]$ of size b uniformly without replacement, independently across iterations. The stochastic gradient is given by

$$
g ^ { ( t ) } = \frac { 1 } { b } \sum _ { i \in B ^ { ( t ) } } \big ( f _ { \theta ^ { ( t ) } } ( x _ { i } ) - y _ { i } \big ) \nabla _ { \theta } f _ { \theta ^ { ( t ) } } \big ( x _ { i } \big ) .\tag{13}
$$

The stochastic gradient is conditionally unbiased, satisfying $\mathbb { E } [ g ^ { ( t ) } \ | \ \mathcal { F } _ { t } ] = \nabla _ { \theta } L ( \theta ^ { ( t ) } )$ , where $\mathcal { F } _ { t }$ is the sigma-algebra generated by the initialization and all minibatches sampled before iteration t. Compared with deterministic Adam, the stochastic setting introduces additional difficulties due to gradient noise and the dependence between stochastic gradients, momentum, and adaptive preconditioners. To characterize the effect of minibatch sampling, we define

$$
\rho _ { b } = 1 + \frac { n - b } { b ( n - 1 ) \lambda _ { 0 } } ,\tag{14}
$$

for $n \geq 2$ . This quantity captures the effects of minibatch sampling and the minimum eigenvalue of the limiting Gram matrix.

Theorem 2 (Regression with minibatches). Let $\delta \in ( 0 , 1 ) , n \geq 2 ,$ , and $1 \leq b \leq n$ . Consider Adam defined by (4)–(5) with minibatch gradients (13) and anyfixed $\beta _ { 1 } \in [ 0 , 1 )$ . Let $\rho _ { b }$ be defined in (14), and suppose that

$$
\bar { \epsilon } \gtrsim \sqrt { \frac { 1 } { \delta } \log \frac { 2 } { \delta } } .\tag{15}
$$

Choose fixed $\beta _ { 2 } \in ( 0 , 1 )$ and $\bar { \eta } > 0$ such that

$$
0 < 1 - \beta _ { 2 } \lesssim \frac { 1 - \beta _ { 1 } } { \rho _ { b } \log \frac { \rho _ { b } } { ( 1 - \beta _ { 1 } ) \lambda _ { 0 } \delta } } , \qquad \bar { \eta } \asymp 1 - \beta _ { 2 } .\tag{16}
$$

Supposefurther that the network width satisfies

$$
m \gtrsim \frac { 1 } { \lambda _ { 0 } } \log \frac { n } { \delta } + \frac { 1 } { \delta \lambda _ { 0 } ^ { 2 } } \log \frac { 2 } { \delta } .\tag{17}
$$

Then, with probability at least $1 - \delta / 2$ over the random initialization (3), there exists an event $\mathcal { G }$ associated with minibatch sampling such that

$$
\begin{array} { r } { \operatorname { \mathbb P } ( \mathcal { G } \mid \theta ^ { ( 0 ) } ) \geq 1 - \delta / 2 , \qquad \operatorname { \mathbb E } \left[ L ( \theta ^ { ( t ) } ) \mathbf { 1 } _ { \mathcal { G } } \mid \theta ^ { ( 0 ) } \right] \leq C ( 1 - c \bar { \eta } \lambda _ { 0 } ) ^ { t } L ( \theta ^ { ( 0 ) } ) , \quad t \geq 0 , } \end{array}\tag{18}
$$

where $C , c > 0$ are universal constants. In particular, the joint probability of successful initialization and minibatch sampling is at least $1 - \delta .$

Moreover, on the high-probability event G, the parameter sequence $\{ \theta ^ { ( t ) } \}$ converges almost surely to a global minimizer $\theta ^ { \star }$ satisfying $L ( \theta ^ { \star } ) = 0$

Proofsketch. The complete proof is provided in Appendix C. The proof consists of three main steps: concentration at initialization, a conditional descent analysis, and a stopping argument to control the stochastic trajectory. Let $L _ { 0 } = L ( \theta ^ { ( 0 ) } ) , \gamma = 1 - \beta _ { 1 }$ , and $\nu = 1 - \beta _ { 2 }$

First, as in the proof of Theorem 1, the initialization bounds

$$
K ( \theta ^ { ( 0 ) } ) \stackrel { } { \mathop { = } } \frac { 3 } { 4 } \lambda _ { 0 } I , \qquad L _ { 0 } \stackrel { } { \mathop { \sim } } \log \frac { 2 } { \delta }
$$

hold with probability at least $1 - \delta / 2$ . We condition on an initialization satisfying these bounds. To handle the stochastic updates, we introduce an auxiliary network that is globally smooth and agrees with the original network whenever $| a _ { r } | \leq 2$ for all $r .$ . The regularity of the auxiliary network ensures that

$$
K ( \theta ) \succeq \frac { \lambda _ { 0 } } { 4 } I
$$

whenever

$$
\begin{array} { r } { \| \theta - \theta ^ { ( 0 ) } \| \leq \mathcal { R } : = c _ { 0 } \sqrt { m \lambda _ { 0 } } . } \end{array}
$$

We analyze Adam on this auxiliary network and show that its iterates agree with those of the original network on a suitable high-probability event.

Recall the filtration $\mathcal { F } _ { t }$ defined above, and write $\mathbb { E } _ { t } [ \cdot ] = \mathbb { E } [ \cdot \mid \mathcal { F } _ { t } ]$ . Using the exact covariance formula for uniform minibatch sampling, we obtain

$$
\mathbb { E } _ { t } \| g ^ { ( t ) } - \nabla L ( \theta ^ { ( t ) } ) \| ^ { 2 } \lesssim ( \rho _ { b } - 1 ) \| \nabla L ( \theta ^ { ( t ) } ) \| ^ { 2 }\tag{19}
$$

within the region of Gram matrix stability. To control the adaptive preconditioner, we bound each coordinate of the bias-corrected second-moment estimate $\widehat { v } ^ { ( t ) }$ by a scalar moving average of the sampled squared residuals. A uniform-in-time concentration bound for this stopped process then leads to the logarithmic restriction on $\nu$ in (16).

Next, we derive a conditional descent estimate that accounts for the dependence between the current stochastic gradient and the adaptive preconditioner. Recall that $P ^ { ( t ) } = \epsilon D ^ { ( t ) }$ . Using the exact bias-correction formulas, we introduce the virtual iterate

$$
Z ^ { ( t ) } = \theta ^ { ( t ) } - \bar { \eta } k ^ { ( t ) } P ^ { ( t - 1 ) } \widehat { s } ^ { ( t - 1 ) } , \qquad 0 \leq k ^ { ( t ) } \leq \frac { \beta _ { 1 } } { \gamma } ,\tag{20}
$$

where the deterministic coefficients $k ^ { ( t ) }$ are chosen to eliminate the contribution of historical firstmoment estimates from the virtual update. This allows us to separate the virtual update into a main descent term involving a preconditioner determined by past gradients and a remainder term controlled by the second moments of the stochastic gradients.

Combining the descent estimate with the first-moment recurrence, we construct the Lyapunov function

$$
\mathcal { E } ^ { ( t ) } : = L ( Z ^ { ( t ) } ) + C _ { 0 } \frac { \bar { \eta } \nu } { \gamma ^ { 2 } } \Vert \widehat { s } ^ { ( t - 1 ) } \Vert ^ { 2 } ,
$$

which satisfies

$$
\begin{array} { r } { \mathbb E _ { t } \mathcal E ^ { ( t + 1 ) } \le ( 1 - c \bar { \eta } \lambda _ { 0 } ) \mathcal E ^ { ( t ) } , \qquad L ( \theta ^ { ( t ) } ) \le 2 \mathcal E ^ { ( t ) } , } \end{array}
$$

for $t \geq T _ { * } \asymp \nu ^ { - 1 }$ , provided that the local Gram matrix, loss, and historical second-moment bounds hold. The first $T _ { * }$ iterations are treated separately, yielding

$$
\mathbb { E } [ \mathcal { E } ^ { ( T _ { * } ) } \mid \theta ^ { ( 0 ) } ] \lesssim L _ { 0 } .
$$

Since $\bar { \eta } T _ { * } \lesssim 1$ , this initial phase contributes only a universal constant factor to the final convergence bound.

To ensure that the local conditions hold throughout training, we introduce a stopping time for the auxiliary process, defined as the first iteration at which any of the prescribed energy, second-moment, or parameter movement bounds fails. The descent estimate controls the predictable part of the virtual iterate movement, while martingale maximal inequalities control the stochastic fluctuations. In addition, the exact bias-corrected moment weights allow us to control the cumulative movement of all neurons simultaneously. Under the network width and algorithmic parameter conditions, these estimates yield an event $\mathcal { G }$ with conditional probability at least $1 - \delta / 2$ . On this event, the iterates remain within the region of Gram matrix stability, and $| a _ { r } ^ { ( t ) } | \leq 3 / 2$ for all r and t.

Applying the conditional contraction estimate before the stopping time yields geometric decay of the loss in restricted expectation on $\mathcal { G }$ . On this event, the auxiliary and original Adam iterates coincide, including their gradients, moment estimates, and parameter updates. Thus, the convergence bound also holds for the original Adam iterates:

$$
\begin{array} { r } { \mathbb E \left[ L ( \theta ^ { ( t ) } ) \mathbf { 1 } _ { \mathcal { G } } \mid \theta ^ { ( 0 ) } \right] \leq C ( 1 - c \bar { \eta } \lambda _ { 0 } ) ^ { t } L _ { 0 } , \qquad t \geq 0 . } \end{array}
$$

Combining the probability bounds for initialization and minibatch sampling completes the proof. □

Discussion. The parameter $\rho _ { b }$ reflects the effects of minibatch sampling and the minimum eigenvalue of the limiting Gram matrix. As shown in (19), $\rho _ { b }$ − 1 controls the variance of the stochastic gradient relative to the squared norm of the full gradient. In particular, $\rho _ { b }$ decreases as the batch size b increases, with $\rho _ { 1 } = 1 + \lambda _ { 0 } ^ { - 1 }$ and $\rho _ { n } = 1$ . Moreover, the stochastic gradient variance decreases as the training loss approaches zero. This allows Adam to converge to zero training loss without requiring its parameters to decay over iterations.

The restriction on $\beta _ { 2 }$ is needed to control the fluctuations of the adaptive preconditioner and its dependence on the current stochastic gradient. Although $g ^ { ( t ) }$ is conditionally unbiased, the same minibatch is also used to compute $\widehat { v } ^ { ( t ) }$ . Thus, conditional unbiasedness alone is insufficient to establish descent for the preconditioned update. In particular, the Lyapunov descent analysis requires

$$
\frac { ( 1 - \beta _ { 2 } ) \rho _ { b } } { 1 - \beta _ { 1 } }
$$

to be sufficiently small. The additional logarithmic factor in (16) comes from controlling the secondmoment estimates uniformly over all iterations. The condition $\bar { \eta } \asymp 1 - \beta _ { 2 }$ allows us to choose the normalized stepsize and the second-moment parameter jointly. In particular, a larger minibatch size reduces $\rho _ { b }$ , allowing a larger value of $1 - \beta _ { 2 }$ and hence a larger normalized stepsize under our theoretical conditions. Moreover, (16) and $\rho _ { b } > 1$ imply a higher requirement on the secondmoment parameter $\beta _ { 2 }$ than on the first-moment parameter $\beta _ { 1 }$ . This is consistent with the default Adam settings $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9 9 [ 2 2 ]$

Compared with Theorem 1, the stochastic convergence result requires stronger conditions on the Adam parameters and a larger network width. The additional dependence on the confidence parameter $\delta$ arises from controlling the stochastic training trajectory with high probability. In particular, the stopping argument uses an energy threshold of order $\begin{array} { r } { \dot { S } \asymp \frac { 1 } { \delta } \log \frac { 2 } { \delta } } \end{array}$ , which ensures that the probability of exceeding this threshold is of order δ. The condition $\bar { \epsilon } ^ { 2 } \gtrsim S$ then provides the required control of the adaptive preconditioner. Moreover, the parameter movement bounds imply that the probability of leaving the region of Gram matrix stability is at most $\frac { C L _ { 0 } } { m \lambda _ { 0 } ^ { 2 } }$ . Together with $L _ { 0 } \lesssim \log ( 2 / \delta )$ on the initialization event, this yields the additional width requirement $\delta ^ { - 1 } \lambda _ { 0 } ^ { - 2 } \log ( 2 / \delta )$ . The initialization requirement $\lambda _ { 0 } ^ { - 1 } \log ( n / \delta )$ remains unchanged.

The stochastic analysis differs from the deterministic case in two main aspects: controlling the stochastic training trajectory and handling the dependence between stochastic gradients and adaptive preconditioners. In the full-gradient setting, this dependence does not involve sampling randomness, allowing arbitrary fixed $\beta _ { 2 } \in [ 0 , 1 )$ . Thus, when $b = n$ , Theorem 1 applies directly without the additional restrictions of the stochastic theorem. Moreover, for fixed training data and confidence level, the Adam parameters can be chosen independently of the network width $m .$ Hence, the contraction factor does not explicitly depend on $m .$ , despite the width-dependent scaling of the damping parameter and stepsize.

The convergence bound in (18) controls the expected loss restricted to a single high-probability event ${ \mathcal { G } } ,$ on which the iterates remain within the region of Gram matrix stability. This gives geometric decay in the restricted expectation. A similar localized convergence guarantee based on a highprobability event appears in the nonlinear least-squares analysis of SGD in [28, Theorem 3.1]. Our theorem also provides separate probability guarantees for initialization and minibatch sampling, accounting for both sources of randomness.

Finally, we next compare our convergence guarantees with those of GD and SGD. Under the same local regularity conditions, the loss function satisfies a local PL inequality with a constant of order $\lambda _ { 0 }$ . Moreover, the stochastic gradient satisfies

$$
\begin{array} { r } { \mathbb { E } _ { t } \| g ^ { ( t ) } \| ^ { 2 } \lesssim \rho _ { b } \| \nabla L ( \theta ^ { ( t ) } ) \| ^ { 2 } . } \end{array}
$$

A standard SGD descent argument with stepsize $h \asymp \rho _ { b } ^ { - 1 }$ then yields a conditional contraction of the loss with a contraction gap of order $\lambda _ { 0 } / \rho _ { b }$ , as long as the iterates remain in the region where the local conditions hold. A related result under global PL and strong-growth conditions is given in [36, Theorem 4]. Compared with previous results for GD and SGD [1,10,36,39,43], our analysis provides sharper width requirements and improved convergence-rate bounds. Compared with existing convergence analyses of Adam [5,7], which often assume the PL condition, our results establish explicit conditions on the network width and optimizer parameters, while also characterizing the effect of minibatch size on convergence.

## 3 Convergence Analysis of Adam for PINNs

We now extend our convergence analysis of Adam from regression to physics-informed neural networks (PINNs) [29]. Unlike standard regression, PINN training involves differential operators acting

on neural network outputs, which introduces additional challenges in analyzing the residual Jacobian and the associated Gram matrix. Following existing convergence analyses of GD and SGD for PINNs [12, 17, 41], we consider the Poisson equation with Dirichlet boundary conditions:

$$
\Delta u = f \quad \mathrm { i n } \Omega , \qquad u = \varphi \quad \mathrm { o n } \partial \Omega ,\tag{21}
$$

where $\Omega \subset \mathbb { R } ^ { d }$ is a bounded domain with $d \geq 1 , f$ is the source term, and $\varphi$ denotes the boundary data. Although we focus on the Poisson equation, our analysis can also be extended to other PDEs under suitable assumptions.

Let $x _ { 1 } , \dots , x _ { n _ { \mathrm { P D E } } } \in \Omega$ and $z _ { 1 } , . . . , z _ { n _ { \mathrm { B C } } } \in \partial \Omega$ denote the fixed interior and boundary collocation points, respectively, where $n _ { \mathrm { P D E } } , n _ { \mathrm { B C } } \ \geq \ 1$ . We assume that all collocation points are pairwise distinct and satisfy

$$
\| x _ { i } \| , \| z _ { j } \| \leq 1 , \qquad | f ( x _ { i } ) | , | \varphi ( z _ { j } ) | \leq 1 .
$$

We use the same two-layer neural network architecture and initialization as in (2) and (3), respectively, and denote its output by $u _ { \theta } ( x )$ . For a boundary penalty parameter $\alpha _ { \mathrm { B C } } > 0$ , we consider the empirical PINN loss

$$
L ( \theta ) = \frac { 1 } { 2 n _ { \mathrm { P D E } } } \sum _ { i = 1 } ^ { n _ { \mathrm { P D E } } } \left( \Delta u _ { \theta } ( x _ { i } ) - f ( x _ { i } ) \right) ^ { 2 } + \frac { \alpha _ { \mathrm { B C } } } { 2 n _ { \mathrm { B C } } } \sum _ { j = 1 } ^ { n _ { \mathrm { B C } } } \left( u _ { \theta } ( z _ { j } ) - \varphi ( z _ { j } ) \right) ^ { 2 } .\tag{22}
$$

Here, we regard $\alpha _ { \mathrm { B C } }$ as a fixed universal constant.

We use the same notation $L , R , J , K , K _ { \infty }$ , and $\lambda _ { 0 }$ as in the regression analysis, now defined for the PINN loss. Specifically, let $N = n _ { \mathrm { P D E } } + n _ { \mathrm { B C } }$ and define the normalized residual vector

$$
R ( \theta ) = \left( \big ( ( \Delta u _ { \theta } ( x _ { i } ) - f ( x _ { i } ) ) / \sqrt { n _ { \mathrm { P D E } } } \big ) _ { i } \right) \in \mathbb { R } ^ { N } , \qquad L ( \theta ) = \frac { 1 } { 2 } \| R ( \theta ) \| ^ { 2 } .\tag{23}
$$

As before, we define

$$
J ( \theta ) = \nabla R ( \theta ) , \qquad K ( \theta ) = J ( \theta ) J ( \theta ) ^ { \top } ,
$$

and

$$
K _ { \infty } = \mathbb { E } _ { \theta ^ { ( 0 ) } } [ K ( \theta ^ { ( 0 ) } ) ] , \qquad \lambda _ { 0 } = \lambda _ { \operatorname* { m i n } } ( K _ { \infty } ) .
$$

The gradient inequality (8) therefore also holds for the PINN loss. The following lemma establishes the positive definiteness of the limiting Gram matrix $K _ { \infty }$ in this setting.

Lemma 2 (Positive definiteness of the PINN Gram matrix). The limiting PINN Gram matrix $K _ { \infty }$ is positive definite, i.e., $\lambda _ { 0 } > 0$

The proof is provided in Appendix D.1. The positive definiteness follows from the properties of the activation function and the distinctness of the collocation points, without imposing an additional positive definiteness assumption on $K _ { \infty }$ . Similar results on the positive definiteness of PINN Gram matrix can also be found in [12, 17, 18, 41].

## 3.1 Convergence of Deterministic Adam

We first study the convergence of deterministic Adam for training PINNs. At each iteration, we use the exact gradient $g ^ { ( t ) } = \nabla L ( \theta ^ { ( t ) } )$ of the PINN loss (22) in the Adam updates $( 4 ) \mathrm { - } ( 5 )$ , with the damping parameter and the stepsize (6). Compared with regression, the PDE residual involves derivatives

of the network output with respect to its inputs, introducing additional terms in the residual Jacobian and its regularity estimates. This requires more careful control of the network parameters and the Gram matrix throughout training.

Theorem 3 (Poisson PINN with full gradients). Let $\delta \in ( 0 , 1 )$ and consider Adam withfull gradients $g ^ { ( t ) } = \nabla L ( \theta ^ { ( t ) } )$ and arbitrary fixed momentum parameters $\beta _ { 1 } , \beta _ { 2 } \in [ 0 , 1 )$ . Suppose that

$$
\bar { \epsilon } \gtrsim d \sqrt { \left( d ^ { 2 } + \log ^ { 2 } \frac { m d ^ { 2 } } { \delta \lambda _ { 0 } } \right) \log \frac { 2 } { \delta } } , \qquad 0 < \bar { \eta } \lesssim \frac { 1 - \beta _ { 1 } } { d ^ { 2 } } ,\tag{24}
$$

and that the network width satisfies

$$
m \gtrsim \left( d ^ { 2 } + \log ^ { 2 } \frac { m d ^ { 2 } } { \delta \lambda _ { 0 } } \right) \left( \frac { 1 } { \lambda _ { 0 } } \log \frac { N } { \delta } + \frac { d ^ { 2 } } { \lambda _ { 0 } ^ { 2 } } \log \frac { 2 } { \delta } \right) .\tag{25}
$$

Then, with probability at least $1 - \delta$ over the random initialization (3), the Adam iterates satisfy

$$
L ( \theta ^ { ( t ) } ) \leq ( 1 - c \bar { \eta } \lambda _ { 0 } ) ^ { t } L ( \theta ^ { ( 0 ) } ) , \qquad t \geq 0 ,\tag{26}
$$

where $c > 0$ is a universal constant. Moreover, the parameter sequence $\{ \theta ^ { ( t ) } \}$ converges to a finite limit $\theta ^ { \star }$ satisfying $L ( \theta ^ { \star } ) = 0$

Proof sketch. The complete proof is provided in Appendix D. The proof follows the same framework as in Theorem 1, with additional estimates to handle the PDE residuals. For convenience, define

$$
\ell _ { m } = d ^ { 2 } + \log ^ { 2 } \left( \frac { C _ { * } m d ^ { 2 } } { \delta \lambda _ { 0 } } \right) ,\tag{27}
$$

where $C _ { * } > 1$ is a universal constant.

First, under the network width condition (25), concentration inequalities imply that

$$
K ( \theta ^ { ( 0 ) } ) \stackrel { < } { = } \frac { 3 \lambda _ { 0 } } { 4 } I , \qquad L _ { 0 } : = L ( \theta ^ { ( 0 ) } ) \stackrel { < } { \sim } d ^ { 2 } \log \frac { 2 } { \delta } ,
$$

with probability at least $1 - \delta$ . Unlike regression, the PDE residuals involve derivatives of the neural network, which require additional control of the hidden weights. Specifically, when $\left| a _ { r } \right| \leq 2$ and $\lVert w _ { r } - w _ { r } ^ { ( 0 ) } \rVert \leq 1$ for all $r ,$ the residual Jacobian is Lipschitz continuous with constant $C \sqrt { \ell _ { m } / m }$ Combining this estimate with the initialization bound and a singular-value perturbation argument, we obtain

$$
K ( \theta ) \succeq \frac { \lambda _ { 0 } } { 4 } I
$$

whenever

$$
\lVert { \boldsymbol { \theta } } - { \boldsymbol { \theta } } ^ { ( 0 ) } \rVert \leq \mathcal { R } : = c _ { 0 } \sqrt { \frac { m \lambda _ { 0 } } { \ell _ { m } } } .
$$

This yields a local PL inequality, as in the regression analysis.

Next, we use the same Lyapunov-based descent analysis as in the proof of Theorem 1. Under the damping condition (24) and the local bounds on the network parameters and loss, the normalized preconditioner $P ^ { ( t ) } = \epsilon D ^ { ( t ) }$ satisfies ${ \scriptstyle { \frac { 1 } { 2 } } } I \preceq P ^ { ( t ) } \preceq I$ . Using the same Lyapunov function as in the regression analysis,

$$
\mathcal { E } ^ { ( t ) } : = L ( \theta ^ { ( t ) } ) + \frac { 4 \bar { \eta } } { 1 - \beta _ { 1 } } \Vert \widehat { s } ^ { ( t ) } - g ^ { ( t ) } \Vert ^ { 2 } ,
$$

we obtain

$$
\mathcal { E } ^ { ( t + 1 ) } \leq ( 1 - c \bar { \eta } \lambda _ { 0 } ) \mathcal { E } ^ { ( t ) } .
$$

The main difference from regression is the loss Hessian bound, which requires additional estimates for the PDE residuals. These estimates ensure that the descent inequality holds along each update segment.

As in the regression analysis, we establish two bounds on the cumulative parameter movement:

$$
\begin{array} { c } { { \displaystyle \sum _ { t = 0 } ^ { T } \| \theta ^ { ( t + 1 ) } - \theta ^ { ( t ) } \| \lesssim \sqrt { \frac { L _ { 0 } } { \lambda _ { 0 } } } , } } \\ { { \displaystyle \sum _ { t = 0 } ^ { T } \displaystyle \operatorname* { m a x } _ { 1 \leq r \leq m } \| \theta _ { r } ^ { ( t + 1 ) } - \theta _ { r } ^ { ( t ) } \| \lesssim \displaystyle \frac { 1 } { \lambda _ { 0 } } \sqrt { \frac { \ell _ { m } L _ { 0 } } { m } } , } } \end{array}
$$

where ${ \theta _ { r } } = \left( { a _ { r } , w _ { r } } \right)$ denotes the parameters of the r-th neuron. The first bound follows from the Lyapunov descent analysis, while the second uses the exact bias-corrected moment weights and the neuron-wise gradient bounds.

Under the width condition m $\gtrsim \ell _ { m } L _ { 0 } / \lambda _ { 0 } ^ { 2 }$ , these bounds allow us to close the induction argument. Specifically, the neuron-wise movement bound ensures that each update segment stays within the region where the network regularity estimates hold. This provides the required Hessian bound for the Lyapunov descent analysis. Meanwhile, the bound on total parameter movement ensures that the iterates remain within the region of Gram matrix stability. Together, these bounds establish the Lyapunov contraction for all iterations.

Finally, exact bias correction gives $\mathcal { E } ^ { ( 0 ) } ~ = ~ L _ { 0 }$ . Combining the Lyapunov contraction with $L ( \theta ^ { ( t ) } ) \leq \mathcal { E } ^ { ( t ) }$ yields the stated linear convergence rate. Furthermore, the finite cumulative parameter movement implies that the parameter iterates converge to a limit. □

Discussion. As in Theorem 1, our convergence guarantee allows arbitrary fixed momentum parameters $\beta _ { 1 } , \beta _ { 2 } \in [ 0 , 1 )$ , without requiring any additional relationship between them. Compared with regression, the main difference arises from the second-order spatial derivatives in the PDE residuals. These derivatives introduce additional dependence on the input dimension d in the regularity estimates. Moreover, controlling the maximum over the Gaussian-initialized neurons introduces logarithmic factors, captured by $\ell _ { m } .$ , in the network width and damping requirements. By separating these two types of estimates, we obtain an admissible normalized stepsize without additional logarithmic dependence on the network width m.

The two terms in the width requirement (25) serve different purposes. The first ensures a positive lower bound on the Gram matrix at initialization, while the second controls parameter movement to maintain this bound throughout training. Although the width requirement involves m on both sides, its dependence on m on the right-hand side is only logarithmic. Thus, the condition holds for all sufficiently large m when the other problem parameters are fixed. Unlike in regression, the required lower bound on ϵ¯ also depends logarithmically on $m .$ . Choosing ϵ¯ at this order and keeping η¯ fixed, the scaling (6) gives damping and stepsize parameters of order $m ^ { - 1 / 2 }$ up to logarithmic factors.

Finally, Theorem 3 establishes linear convergence of Adam for the empirical PINN loss evaluated at the collocation points. However, convergence of the empirical loss does not necessarily imply that the neural network converges to the true PDE solution throughout Ω. Such a guarantee requires additional generalization and PDE stability analysis, which are beyond the scope of this paper. We leave this question for future work.

## 3.2 Convergence of Stochastic Adam

We now study stochastic Adam applied to the empirical PINN loss (22). At each iteration t, we independently sample two minibatches $B _ { \mathrm { P D E } } ^ { ( t ) } \subset [ n _ { \mathrm { P D E } } ]$ and $B _ { \mathrm { B C } } ^ { ( t ) } \subset [ n _ { \mathrm { B C } } ]$ of sizes $b _ { \mathrm { P D E } }$ and $b _ { \mathrm { B C } } ,$ respectively, uniformly without replacement. The minibatches are sampled independently across iterations and independently of the initialization, while the collocation points remain fixed. The stochastic gradient is given by

$$
\begin{array} { c } { \displaystyle { g ^ { ( t ) } = \frac { 1 } { b _ { \mathrm { P D E } } } \sum _ { i \in B _ { \mathrm { P D E } } ^ { ( t ) } } \big ( \Delta u _ { \theta ^ { ( t ) } } ( x _ { i } ) - f ( x _ { i } ) \big ) \nabla _ { \theta } \Delta u _ { \theta ^ { ( t ) } } ( x _ { i } ) } } \\ { \displaystyle { + \frac { \alpha _ { \mathrm { B C } } } { b _ { \mathrm { B C } } } \sum _ { j \in B _ { \mathrm { B C } } ^ { ( t ) } } \big ( u _ { \theta ^ { ( t ) } } ( z _ { j } ) - \varphi ( z _ { j } ) \big ) \nabla _ { \theta } u _ { \theta ^ { ( t ) } } ( z _ { j } ) . } } \end{array}\tag{28}
$$

By uniform sampling, $g ^ { ( t ) }$ is a conditionally unbiased estimator of the full gradient $\nabla L ( \theta ^ { ( t ) } )$

To characterize the effects of minibatch sampling on the PDE and boundary residuals, we define

$$
\rho _ { b } = 1 + \frac { 1 } { \lambda _ { 0 } } \operatorname* { m a x } \left\{ d ^ { 2 } \frac { n _ { \mathrm { P D E } } - b _ { \mathrm { P D E } } } { b _ { \mathrm { P D E } } \left( n _ { \mathrm { P D E } } - 1 \right) } , \alpha _ { \mathrm { B C } } \frac { n _ { \mathrm { B C } } - b _ { \mathrm { B C } } } { b _ { \mathrm { B C } } \left( n _ { \mathrm { B C } } - 1 \right) } \right\} ,\tag{29}
$$

where $b = ( b _ { \mathrm { P D E } } , b _ { \mathrm { B C } } )$ denotes the pair of minibatch sizes. The factor $d ^ { 2 }$ arises from the derivatives in the PDE residuals, while the corresponding bounds for the boundary residuals does not rely on $d .$

Theorem 4 (Poisson PINN with minibatches). Let $\delta \in ( 0 , 1 )$ and n<sub>PDE</sub>, $n _ { \mathrm { B C } } \geq 2 , 1 \leq b _ { \mathrm { P D E } } \leq n _ { \mathrm { P D E } } ,$ $1 \leq b _ { \mathrm { B C } } \leq n _ { \mathrm { B C } }$ . Consider Adam with the minibatch gradients (28), the damping parameter and stepsize (6), and anyfixed $\beta _ { 1 } \in [ 0 , 1 )$ . Suppose that

$$
\bar { \epsilon } \gtrsim d \sqrt { \frac { 1 } { \delta } \left( d ^ { 2 } + \log ^ { 2 } \frac { m d ^ { 2 } } { \delta \lambda _ { 0 } } \right) \log \frac { 2 } { \delta } } .\tag{30}
$$

Choose fixed $\beta _ { 2 } \in ( 0 , 1 )$ and $\bar { \eta } > 0$ such that

$$
0 < 1 - \beta _ { 2 } \lesssim \frac { 1 - \beta _ { 1 } } { \rho _ { b } \log \Big ( \frac { \rho _ { b } d ^ { 2 } } { ( 1 - \beta _ { 1 } ) \lambda _ { 0 } \delta } \Big ) } , \qquad \bar { \eta } \asymp \frac { 1 - \beta _ { 2 } } { d ^ { 2 } } .\tag{31}
$$

Supposefurther that the network width satisfies

$$
m \gtrsim \left( d ^ { 2 } + \log ^ { 2 } \frac { m d ^ { 2 } } { \delta \lambda _ { 0 } } \right) \left( \frac { 1 } { \lambda _ { 0 } } \log \frac { N } { \delta } + \frac { d ^ { 2 } } { \delta \lambda _ { 0 } ^ { 2 } } \log \frac { 2 } { \delta } \right) .\tag{32}
$$

Then, with probability at least $1 - \delta / 2$ over the random initialization (3), there exists an event $\mathcal { G }$ associated with minibatch sampling such that

$$
\begin{array} { r l } { \mathbb { P } ( \mathcal { G } \mid \theta ^ { ( 0 ) } ) \geq 1 - \delta / 2 , } \\ { \mathbb { E } \left[ L ( \theta ^ { ( t ) } ) \mathbf { 1 } _ { \mathcal { G } } \mid \theta ^ { ( 0 ) } \right] \leq C ( 1 - c \bar { \eta } \lambda _ { 0 } ) ^ { t } L ( \theta ^ { ( 0 ) } ) , } & { \quad t \geq 0 . } \end{array}\tag{33}
$$

Here, $c , C > 0$ are universal constants. In particular, the joint probability of successful initialization and minibatch sampling is at least $1 - \delta .$

Moreover, on the high-probability event ${ \mathcal { G } } ,$ , the parameter sequence $\{ \theta ^ { ( t ) } \}$ converges almost surely to a global minimizer θ<sup>⋆</sup> satisfying $L ( \theta ^ { \star } ) = 0$

Proofsketch. The complete proof is provided in Appendix E. We follow the stochastic Adam analysis of Theorem 2, with additional regularity estimates for the PDE residuals.

As in the deterministic PINN analysis, the initialization bounds give

$$
K ( \theta ^ { ( 0 ) } ) \succeq \frac { 3 \lambda _ { 0 } } { 4 } I , \qquad L _ { 0 } : = L ( \theta ^ { ( 0 ) } ) \lesssim d ^ { 2 } \log \frac { 2 } { \delta } .
$$

We introduce an auxiliary network with smooth cutoffs on the output weights and hidden-weight movement from initialization. The auxiliary network agrees with the original PINN whenever $| a _ { r } | \leq$ 2 and $\lVert w _ { r } - w _ { r } ^ { ( 0 ) } \rVert \leq 1$ for all r. Its residual map is locally Lipschitz, and the associated Gram matrix remains uniformly positive definite whenever

$$
\lVert \theta - \theta ^ { ( 0 ) } \rVert \leq \mathcal { R } : = c _ { 0 } \sqrt { \frac { m \lambda _ { 0 } } { \ell _ { m } } } ,
$$

where $\ell _ { m }$ is defined in (27). We analyze the auxiliary process and show that its iterates agree with those of the original PINN on a suitable high-probability event.

Next, the independence of the PDE and boundary minibatches allows us to bound their conditional variances separately. As in the stochastic regression analysis, we obtain

$$
\begin{array} { r } { \mathbb { E } _ { t } \| g ^ { ( t ) } - G ^ { ( t ) } \| ^ { 2 } \lesssim ( \rho _ { b } - 1 ) \| G ^ { ( t ) } \| ^ { 2 } , } \end{array}
$$

where $G ^ { ( t ) } : = \nabla L ( \theta ^ { ( t ) } )$ and $\mathbb { E } _ { t }$ denotes conditional expectation given the sampling history. To control the adaptive preconditioner, let $U ^ { ( t ) }$ be the bias-corrected moving average of the sampled, weighted squared residuals. Using a common scalar bound, we obtain

$$
\widehat { v } _ { j } ^ { ( t ) } \lesssim \frac { \ell _ { m } } { m } U ^ { ( t ) } \quad \mathrm { f o r ~ a l l ~ } j , \qquad \sum _ { j } \widehat { v } _ { j } ^ { ( t ) } \lesssim d ^ { 2 } U ^ { ( t ) } .
$$

The first bound controls individual coordinates of the adaptive preconditioner, while the second controls the sum of the second-moment estimates. In particular, the latter depends on $d ^ { 2 }$ rather than $\ell _ { m } ,$ avoiding additional logarithmic factors in the subsequent descent analysis.

We use the same virtual iterate $Z ^ { ( t ) }$ as in (20) and construct the Lyapunov function

$$
\mathcal { E } ^ { ( t ) } = { L } ( Z ^ { ( t ) } ) + C _ { 0 } \frac { \bar { \eta } \nu } { \gamma ^ { 2 } } \| \widehat { s } ^ { ( t - 1 ) } \| ^ { 2 } ,
$$

where $\gamma = 1 - \beta _ { 1 }$ and $\nu = 1 - \beta _ { 2 }$ . Compared with regression, the PINN analysis requires additional control of the residuals along the segment connecting the actual and virtual iterates, as well as all possible virtual update segments. Under the network width condition, we establish a uniform Hessian bound of order $d ^ { 2 }$ along these segments. Combining this bound with the conditional descent analysis from the regression setting, we obtain

$$
\mathbb { E } _ { t } [ \mathcal { E } ^ { ( t + 1 ) } ] \leq ( 1 - c \bar { \eta } \lambda _ { 0 } ) \mathcal { E } ^ { ( t ) } ,
$$

provided that the required energy, second-moment, and parameter movement bounds hold.

Finally, we apply the stopping-time argument from the stochastic regression analysis, using the PINN-specific regularity bounds to control the training trajectory. As before, the energy descent estimate and martingale maximal inequalities control the cumulative parameter movement, while the bias-corrected moment weights provide neuron-wise bounds. Together, these estimates yield an event $\mathcal { G }$ with conditional probability at least $1 - \delta / 2$ , on which the auxiliary and original Adam iterates coincide for all iterations. Applying the conditional Lyapunov contraction before the stopping time then results in the convergence bound (33). □

Discussion. The factor $\rho _ { b }$ characterizes the relative variance of the stochastic gradient arising from the PDE and boundary minibatches. As the minibatch sizes increase, the corresponding sampling variance decreases and vanishes under full-batch sampling. The factor $d ^ { 2 }$ in the PDE contribution arises from the derivative bounds for the Laplacian features. For the boundary contribution, only one factor of $\alpha _ { \mathrm { B C } }$ remains, since the other is absorbed into the weighted loss when bounding the variance. When both minibatches are sampled in full, Theorem 3 applies with arbitrary fixed $\beta _ { 2 } ~ \in ~ [ 0 , 1 )$ The stochastic theorem also allows one group to be sampled in full while the other is sampled stochastically.

The conditions in (31) serve different purposes. As in stochastic regression, choosing $1 - \beta _ { 2 }$ sufficiently small controls the dependence between the current stochastic gradient and the adaptive preconditioner, as well as the accumulation of second-moment estimates. The logarithmic factor arises from controlling these estimates uniformly over iterations. The condition $\bar { \eta } \asymp ( 1 - \beta _ { 2 } ) / d ^ { 2 }$ further accounts for the additional $d ^ { 2 }$ factor in the PINN smoothness bounds. These parameter conditions do not introduce any additional dependence on the network width $m$

Compared with the deterministic result in Theorem 3, the stochastic analysis requires a threshold of order $d ^ { 2 } \delta ^ { - 1 } \log ( 2 / \delta )$ to control the stochastic training trajectory. This leads to an additional factor of $\delta ^ { - 1 / 2 }$ in the damping requirement (30) and $\delta ^ { - 1 }$ in the parameter movement term of the width requirement (32). As in the deterministic PINN analysis, controlling the maximum over the Gaussian-initialized neurons introduces logarithmic dependence on m in the damping and width requirements through $\ell _ { m }$ . In contrast, the averaged feature bounds allow us to control the stepsize and variance estimates without additional logarithmic dependence on $m .$ Consequently, the conditions on $\beta _ { 2 }$ and $\bar { \eta }$ in (31) do not explicitly depend on the network width. As in Theorem 2, the convergence bound (33) holds in expectation restricted to a single high-probability event $\mathcal { G }$ for all iterations. The probability statement accounts separately for initialization and minibatch sampling.

Table 1 compares the convergence guarantees for GD, SGD, and Adam in training PINNs. By adapting our analysis to GD and SGD, we obtain sharper width requirements and improved convergence rates compared with existing results [12,17]. In particular, the existing SGD analysis [17] does not explicitly characterize how the minibatch size affects the stepsize conditions and convergence rates. Our GD and SGD results, established in Appendix F, make explicit the dependence on the input dimension and minibatch size. Moreover, our analysis shows that Adam achieves comparable width requirements and convergence rates to GD and SGD, up to logarithmic factors in the stochastic setting, despite the additional challenges introduced by momentum and adaptive preconditioning.

## 4 Conclusion

In this paper, we study the global convergence of Adam in training two-layer neural networks for regression and physics-informed learning. Using the neural tangent kernel framework, we establish linear convergence guarantees for sufficiently wide networks with both full-batch and minibatch gradients in Adam. Our analysis addresses the dependence introduced by historical momentum and adaptive preconditioning, which prevents a direct application of existing GD and SGD analyses. In particular, we develop Lyapunov-based descent arguments, a virtual-iterate construction, and stochastic trajectory control to establish convergence without modifying the standard Adam updates. We further extend these results to physics-informed neural networks (PINNs) for the Poisson equation, where differential residuals require additional regularity estimates and more restrictive conditions on the network width and algorithmic parameters. These results provide a theoretical explanation for the convergence behavior of Adam in overparameterized neural networks.

<table><tr><td>Method</td><td>Sufficient width</td><td>Stepsize</td><td>Convergence rate</td></tr><tr><td>GD [12, Theorem 4.5]</td><td> $\widetilde \Omega \left( N ^ { 2 } \delta ^ { - 3 } \lambda _ { 0 } ^ { - 4 } \right)$ </td><td> $h \asymp \lambda _ { 0 }$ </td><td> $\lambda _ { 0 } ^ { 2 }$ </td></tr><tr><td>SGD [17, Theorem 3.1]</td><td> $\widetilde \Omega ( d ^ { 6 } \lambda _ { 0 } ^ { - 2 } + d ^ { 1 2 } \delta ^ { - 2 } \lambda _ { 0 } ^ { - 4 } )$ </td><td> $h = \widetilde { \Theta } \big ( \lambda _ { 0 } / d ^ { 9 / 2 } \big )$ </td><td> $\widetilde { \Theta } \left( \lambda _ { 0 } ^ { 2 } / d ^ { 9 / 2 } \right)$ </td></tr><tr><td>GD (Proposition 1)</td><td> $\widetilde \Omega \big ( d ^ { 2 } \lambda _ { 0 } ^ { - 1 } + d ^ { 4 } \lambda _ { 0 } ^ { - 2 } \big )$ </td><td> $h \asymp d ^ { - 2 }$ </td><td> $\lambda _ { 0 } / d ^ { 2 }$ </td></tr><tr><td>SGD (Proposition 1)</td><td> $\widetilde \Omega \big ( d ^ { 2 } \lambda _ { 0 } ^ { - 1 } + d ^ { 4 } \delta ^ { - 1 } \lambda _ { 0 } ^ { - 2 } \big )$ </td><td> $h \asymp ( d ^ { 2 } \rho _ { b } ) ^ { - 1 }$ </td><td> $\lambda _ { 0 } / ( d ^ { 2 } \rho _ { b } )$ </td></tr><tr><td>Full Adam (Theorem 3)</td><td> $\widetilde \Omega \big ( d ^ { 2 } \lambda _ { 0 } ^ { - 1 } + d ^ { 4 } \lambda _ { 0 } ^ { - 2 } \big )$ </td><td> $\bar { \eta } \asymp ( 1 - \beta _ { 1 } ) / d ^ { 2 }$ </td><td> $( 1 - \beta _ { 1 } ) \lambda _ { 0 } / d ^ { 2 }$ </td></tr><tr><td>Minibatch Adam (Theorem 4)</td><td> $\widetilde \Omega \big ( d ^ { 2 } \lambda _ { 0 } ^ { - 1 } + d ^ { 4 } \delta ^ { - 1 } \lambda _ { 0 } ^ { - 2 } \big )$ </td><td> $\bar { \eta } = \widetilde { \Theta } \big ( ( 1 - \beta _ { 1 } ) / ( d ^ { 2 } \rho _ { b } ) \big )$ </td><td> $\widetilde { \Theta } \big ( ( 1 - \beta _ { 1 } ) \lambda _ { 0 } / ( d ^ { 2 } \rho _ { b } ) \big )$ </td></tr></table>

Table 1: Comparison of network width requirements, stepsize, and convergence rate for GD, SGD, and Adam in training PINNs for the Poisson equation. The convergence rates are expressed as contraction gaps. The notation $\widetilde \Omega$ and $\widetilde { \Theta }$ hides logarithmic factors. Here, $\rho _ { b }$ is the minibatch sampling factor defined in (29), h is the stepsize for GD and SGD, and $\bar { \eta } = \eta / \epsilon$ is the normalized stepsize for Adam. The analysis of [12] treats the input dimension d as fixed and does not explicitly track its dependence. The analysis of [17] treats the stochastic gradient variance bound as a constant and does not explicitly characterize the dependence on minibatch size.

Several directions remain for future research. First, an interesting direction is to study the global convergence of quasi-Newton methods, particularly L-BFGS [26], for training PINNs. In practice, L-BFGS is widely used in scientific machine learning [29], often following an initial training stage with Adam [30]. Unlike GD, SGD, and Adam, L-BFGS uses curvature information through approximate inverse-Hessian updates. Recent convergence results for natural gradient descent in PINNs suggest that incorporating additional geometric information may improve convergence rates [41]. It would therefore be interesting to investigate whether similar improvements can be established for L-BFGS.

Second, although we establish global convergence guarantees for Adam, our theoretical results do not show an advantage over SGD in terms of convergence rates or network width requirements. This differs from empirical observations in some deep learning applications, where Adam-type methods often outperform SGD. Understanding this difference is an important direction for future research. In particular, it would be interesting to study how data distributions, gradient heterogeneity, and network architectures affect the convergence of Adam. Such an analysis may help identify conditions under which Adam achieves faster convergence than SGD.

Finally, our analysis is limited to two-layer neural networks and requires specific conditions on the damping parameter, momentum parameters, and stepsize, particularly in the stochastic setting. Relaxing these conditions and establishing stronger stochastic convergence guarantees are important directions for future research. Another direction is to extend the analysis to deeper neural networks and more general PDEs, where the NTK properties and optimization dynamics are more difficult to analyze. These extensions may provide a broader theoretical understanding of optimization methods for training neural networks.

## References

[1] Zeyuan Allen-Zhu, Yuanzhi Li, and Zhao Song. A convergence theory for deep learning via over-parameterization. In International Conference on Machine Learning, pages 242–252. PMLR, 2019.

[2] Anas Barakat and Pascal Bianchi. Convergence and dynamical behavior of the adam algorithm for nonconvex stochastic optimization. SIAM Journal on Optimization, 31(1):244–274, 2021.

[3] Lu´ıs Carvalho, Joao L Costa, Jos ˜ e Mour ´ ao, and Gonc¸alo Oliveira. The positivity of the neural ˜ tangent kernel. SIAM Journal on Mathematics ofData Science, 7(2):495–515, 2025.

[4] Matias D. Cattaneo, Jason Matthew Klusowski, and Boris Shigida. On the implicit bias of Adam. In International Conference on Machine Learning, volume 235, pages 5862–5906. PMLR, 2024.

[5] Kushal Chakrabarti and Mayank Baranwal. A methodology establishing linear convergence of adaptive gradient methods under PL inequality. In ECAI 2024: 27th European Conference on Artificial Intelligence, 19–24 October 2024, Santiago de Compostela, Spain–Including 13th Conference on Prestigious Applications ofIntelligent Systems (PAIS 2024), pages 2402–2409, 2024.

[6] Xiangyi Chen, Sijia Liu, Ruoyu Sun, and Mingyi Hong. On the convergence of a class of Adam-type algorithms for non-convex optimization. In International Conference on Learning Representations, 2019.

[7] Steffen Dereich, Arnulf Jentzen, and Sebastian Kassing. ODE approximation for the adam algorithm: General and overparametrized setting. arXiv preprint arXiv:2511.04622, 2025.

[8] Alemayehu Tamirie Deresse and Alemu Senbeta Bekela. A deep learning approach: physicsinformed neural networks for solving a nonlinear telegraph equation with different boundary conditions. BMC Research Notes, 18(1):77, 2025.

[9] Simon Du, Jason Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai. Gradient descent finds global minima of deep neural networks. In International Conference on Machine Learning, pages 1675–1685. PMLR, 2019.

[10] Simon S. Du, Xiyu Zhai, Barnabas Poczos, and Aarti Singh. Gradient descent provably optimizes over-parameterized neural networks. In International Conference on Learning Representations, 2019.

[11] Ghazal Farhani, Nima Hosseini Dashtbayaz, Alexander Kazachek, and Boyu Wang. A simple remedy for failure modes in physics-informed neural networks. Neural Networks, 183:106963, 2025.

[12] Yihang Gao, Yiqi Gu, and Michael Ng. Gradient descent finds the global optima of two-layer physics-informed neural networks. In International Conference on Machine Learning, pages 10676–10707. PMLR, 2023.

[13] Yihang Gao and Vincent Y. F. Tan. On the convergence of (stochastic) gradient descent for Kolmogorov–Arnold networks. IEEE Transactions on Information Theory, 71(9):7270–7291, 2025.

[14] Benjamin Gess and Sebastian Kassing. Exponential convergence rates for momentum stochastic gradient descent in the overparametrized setting. Mathematical Programming, pages 1–37, 2026.

[15] Saeed Ghadimi and Guanghui Lan. Stochastic first-and zeroth-order methods for nonconvex stochastic programming. SIAM Journal on Optimization, 23(4):2341–2368, 2013.

[16] Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and ´ generalization in neural networks. Advances in Neural Information Processing Systems, 31, 2018.

[17] Bangti Jin and Longjun Wu. Convergence of stochastic gradient methods for wide two-layer physics-informed neural networks for the poisson equation. Neural Networks, page 109335, 2026.

[18] Bangti Jin and Longjun Wu. The differential neural tangent kernel and its positivity. arXiv preprint arXiv:2607.10200, 2026.

[19] Hamed Karimi, Julie Nutini, and Mark Schmidt. Linear convergence of gradient and proximalgradient methods under the polyak-łojasiewicz condition. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 795–811. Springer, 2016.

[20] George Em Karniadakis, Ioannis G Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. Nature Reviews Physics, 3(6):422–440, 2021.

[21] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4396–4405. IEEE, 2019.

[22] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. International Conference for Learning Representations (ICLR), 2015.

[23] Haochuan Li, Alexander Rakhlin, and Ali Jadbabaie. Convergence of Adam under relaxed assumptions. Advances in Neural Information Processing Systems, 36:52166–52196, 2023.

[24] Ye Li, Song-Can Chen, and Sheng-Jun Huang. Implicit stochastic gradient descent for training physics-informed neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 8692–8700, 2023.

[25] Chaoyue Liu, Libin Zhu, and Mikhail Belkin. Loss landscapes and optimization in overparameterized non-linear systems and neural networks. Applied and Computational Harmonic Analysis, 59:85–116, 2022.

[26] Dong C Liu and Jorge Nocedal. On the limited memory BFGS method for large scale optimization. Mathematical Programming, 45(1):503–528, 1989.

[27] Colin McDiarmid et al. On the method of bounded differences. Surveys in Combinatorics, 141(1):148–188, 1989.

[28] Samet Oymak and Mahdi Soltanolkotabi. Overparameterized nonlinear learning: Gradient descent takes the shortest path? In International Conference on Machine Learning, pages 4951–4960. PMLR, 2019.

[29] Maziar Raissi, Paris Perdikaris, and George E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378:686–707, 2019.

[30] Pratik Rathore, Weimu Lei, Zachary Frangella, Lu Lu, and Madeleine Udell. Challenges in training PINNs: A loss landscape perspective. In International Conference on Machine Learning. PMLR, 2024.

[31] Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar. On the convergence of adam and beyond. In International Conference on Learning Representations, 2018.

[32] Xuan Tang, Han Zhang, Yuan Cao, and Difan Zou. Understanding the generalization of stochastic gradient adam in learning neural networks. Advances in Neural Information Processing Systems, 38:18120–18187, 2026.

[33] Joel A. Tropp. Freedman’s inequality for matrix martingales. Electronic Communications in Probability, 16:262–270, 2011.

[34] Joel A Tropp. An introduction to matrix concentration inequalities. Foundations and trends® in machine learning, 8(1-2):1–230, 2015.

[35] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017.

[36] Sharan Vaswani, Francis Bach, and Mark Schmidt. Fast and faster convergence of SGD for over-parameterized models and an accelerated perceptron. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 1195–1204. PMLR, 2019.

[37] Sifan Wang, Xinling Yu, and Paris Perdikaris. When and why pinns fail to train: A neural tangent kernel perspective. Journal of Computational Physics, 449:110768, 2022.

[38] David Williams. Probability with Martingales. Cambridge University Press, 1991.

[39] Jiaming Xu and Hanjing Zhu. Overparametrized multi-layer neural networks: Uniform concentration of neural tangent kernel and convergence of stochastic gradient descent. Journal of Machine Learning Research, 25(94):1–83, 2024.

[40] Xianliang Xu, Ting Du, Wang Kong, Bin Shan, Zhongyi Huang, and Ye Li. Convergence of implicit gradient descent for training two-layer physics-informed neural networks. East Asian Journal on Applied Mathematics, 17(1):59–94, 2027.

[41] Xianliang Xu, Wang Kong, Jiaheng Mao, Zhongyi Huang, and Ye Li. Fast convergence of natural gradient descent for over-parameterized physics-informed neural networks. In International Conference on Learning Representations, 2026.

[42] Difan Zou, Yuan Cao, Yuanzhi Li, and Quanquan Gu. Understanding the generalization of adam in learning neural networks with proper regularization. In International Conference on Learning Representations, 2023.

[43] Difan Zou and Quanquan Gu. An improved analysis of training over-parameterized deep neural networks. Advances in Neural Information Processing Systems, 32, 2019.

A Auxiliary Lemmas 23   
B Proof of Theorem 1 25   
B.1 Positive Definiteness of the Limiting Gram Matrix . 25   
B.2 Key Lemmas 26   
B.3 Completion of the Proof 33   
C Proof of Theorem 2 35   
C.1 An Auxiliary Network with Global Regularity 35   
C.2 Key Lemmas 38   
C.3 Completion of the Proof 45   
D Proof of Theorem 3 52   
D.1 Differential Features and Positive Definiteness of the Limiting Gram Matrix . 52   
D.2 Key Lemmas 54   
D.3 Completion of the Proof 60   
E Proof of Theorem 4 62   
E.1 An Auxiliary PINN with Global Residual Regularity 62   
E.2 Key Lemmas 64   
E.3 Completion of the Proof 68   
F Convergence Analysis of GD and SGD for PINNs 72

## A Auxiliary Lemmas

We collect several concentration inequalities used in our convergence analysis of Adam. For completeness, we state the results needed in our proofs and refer to the cited references for their proofs.

Lemma 3 (Matrix Chernoff inequality [34, Theorem 5.1.1]). Let $X _ { 1 } , \ldots , X _ { m }$ be independent symmetric random matrices in $\mathbb { R } ^ { n \times n }$ . Suppose that,for some $B > 0$

$$
0 \preceq X _ { r } \preceq B I , \qquad r = 1 , \ldots , m ,
$$

almost surely. Define

$$
\mu _ { \operatorname* { m i n } } = \lambda _ { \operatorname* { m i n } } \left( \sum _ { r = 1 } ^ { m } \mathbb { E } [ X _ { r } ] \right) > 0 .
$$

Then, for any $\alpha \in ( 0 , 1 )$

$$
\mathbb { P } \left( \lambda _ { \operatorname* { m i n } } \left( \sum _ { r = 1 } ^ { m } X _ { r } \right) \leq ( 1 - \alpha ) \mu _ { \operatorname* { m i n } } \right) \leq n \exp \left( { - \frac { \alpha ^ { 2 } \mu _ { \operatorname* { m i n } } } { 2 B } } \right) .\tag{34}
$$

Lemma 4 (Bounded-differences inequality [27]). Let $Z _ { 1 } , \ldots , Z _ { m }$ be independent random variables taking values in $\mathcal { Z } _ { 1 } , \ldots , \mathcal { Z } _ { m } ,$ respectively. Let $\begin{array} { r } { F : \prod _ { r = 1 } ^ { m } \mathcal { Z } _ { r } \to } \end{array}$ R be a measurable function satisfying $\mathbb { E } | F ( Z _ { 1 } , \ldots , Z _ { m } ) | < \infty .$ . Suppose there exist constants $b _ { 1 } , \dots , b _ { m } \geq 0 \mathrm { { ; } }$ , with $\textstyle \sum _ { r = 1 } ^ { m } b _ { r } ^ { 2 } > 0$ , such that

$$
| F ( z _ { 1 } , \ldots , z _ { r } , \ldots , z _ { m } ) - F ( z _ { 1 } , \ldots , z _ { r } ^ { \prime } , \ldots , z _ { m } ) | \leq b _ { r }
$$

for every r and all admissible arguments. Writing $Z = ( Z _ { 1 } , \ldots , Z _ { m } )$ , we have, for any $u > 0 _ { : }$

$$
\mathbb { P } \left( F ( Z ) - \mathbb { E } [ F ( Z ) ] \ge u \right) \le \exp \left( - \frac { 2 u ^ { 2 } } { \sum _ { r = 1 } ^ { m } b _ { r } ^ { 2 } } \right) .\tag{35}
$$

Lemma 5 (Conditional Bernstein inequality [33, Theorem 1.1]). Let $( \mathcal { F } _ { s } ) _ { s = 0 } ^ { N + 1 }$ be afiltration, and let X<sub>s</sub> be $\mathcal { F } _ { s + 1 }$ -measurable with $\mathbb { E } [ X _ { s } \mid \mathcal { F } _ { s } ] = 0 f o r s = 0 , \ldots , N$ . Suppose there exist deterministic constants $B , V > 0$ such that

$$
| X _ { s } | \leq B \quad a . s . , \qquad \sum _ { s = 0 } ^ { N } \mathbb { E } [ X _ { s } ^ { 2 } \mid { \mathcal F } _ { s } ] \leq V \quad a . s .
$$

Then, for any $u > 0$

$$
\mathbb { P } \left( \sum _ { s = 0 } ^ { N } X _ { s } \ge u \right) \le \exp \left( - \frac { u ^ { 2 } } { 2 ( V + B u / 3 ) } \right) .\tag{36}
$$

Proof. The lemma is a direct result of Freedman’s inequality; see [33, Theorem 1.1]. We include a short proof for completeness.

For any $\ 0 \textless h \textless 3 / B$ , expanding the exponential and using the conditional centering of $X _ { s } ,$ together with $k ! \geq 2 \cdot 3 ^ { k - 2 }$ for $k \geq 2 ,$ , yields

$$
\mathbb { E } [ e ^ { h X _ { s } } \mid \mathcal { F } _ { s } ] \le 1 + \frac { h ^ { 2 } \mathbb { E } [ X _ { s } ^ { 2 } \mid \mathcal { F } _ { s } ] } { 2 ( 1 - h B / 3 ) } \le \exp \left( \frac { h ^ { 2 } \mathbb { E } [ X _ { s } ^ { 2 } \mid \mathcal { F } _ { s } ] } { 2 ( 1 - h B / 3 ) } \right) ,
$$

where we have used

$$
\begin{array} { r } { \big | \mathbb { E } [ X _ { s } ^ { k } \mid \mathcal { F } _ { s } ] \big | \le B ^ { k - 2 } \mathbb { E } [ X _ { s } ^ { 2 } \mid \mathcal { F } _ { s } ] , \qquad k \ge 2 . } \end{array}
$$

Consequently, the process

$$
M _ { t } = \exp \left( h \sum _ { s = 0 } ^ { t - 1 } X _ { s } - \frac { h ^ { 2 } } { 2 ( 1 - h B / 3 ) } \sum _ { s = 0 } ^ { t - 1 } \mathbb { E } [ X _ { s } ^ { 2 } \mid \mathcal { F } _ { s } ] \right) , \qquad t = 0 , \dots , N + 1 ,
$$

is a nonnegative supermartingale with $M _ { 0 } = 1$ . Applying Markov’s inequality and the assumed bound on the accumulated conditional variance gives

$$
\mathbb { P } \left( \sum _ { s = 0 } ^ { N } X _ { s } \ge u \right) \le \exp \left( - h u + \frac { h ^ { 2 } V } { 2 ( 1 - h B / 3 ) } \right) .
$$

Choosing $h = u / ( V + B u / 3 )$ completes the proof.

Lemma 6 (Maximal inequalities [38]). Let $( \mathcal { F } _ { t } ) _ { t \geq 0 }$ be afiltration.

1. Let $( X _ { t } ) _ { t \geq t _ { 0 } }$ be a nonnegative supermartingale. Then, for any $a > 0$

$$
\mathbb { P } \left( \operatorname* { s u p } _ { t \geq t _ { 0 } } X _ { t } \geq a \right) \leq \frac { \mathbb { E } [ X _ { t _ { 0 } } ] } { a } .
$$

2. Let $( M _ { t } ) _ { t = 0 } ^ { N }$ be a square-integrable Euclidean-valued martingale with $M _ { 0 } = 0$ . Then,

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { 0 \leq t \leq N } \| M _ { t } \| ^ { 2 } \right] \leq 4 \mathbb { E } \| M _ { N } \| ^ { 2 } = 4 \sum _ { t = 0 } ^ { N - 1 } \mathbb { E } \| M _ { t + 1 } - M _ { t } \| ^ { 2 } .
$$

$$
\begin{array} { r } { I f \sum _ { t = 0 } ^ { \infty } \mathbb { E } \| M _ { t + 1 } - M _ { t } \| ^ { 2 } < \infty , t h e n } \end{array}
$$

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { t \geq 0 } \| M _ { t } \| ^ { 2 } \right] \leq 4 \sum _ { t = 0 } ^ { \infty } \mathbb { E } \| M _ { t + 1 } - M _ { t } \| ^ { 2 } .
$$

Proof. For the first assertion, let $\tau$ be the first time $X _ { t }$ reaches the threshold $^ { a , }$ truncated at a fixed horizon $N$ . By the nonnegativity and supermartingale property of $X _ { t }$ , we have

$$
a \mathbb { P } \left( \operatorname* { m a x } _ { t _ { 0 } \leq t \leq N } X _ { t } \geq a \right) \leq \mathbb { E } [ X _ { t _ { 0 } } ] .
$$

Letting $N  \infty$ and, if necessary, replacing a by $a - \varepsilon$ and then letting $\varepsilon \downarrow 0$ proves the first assertion.

For the second assertion, let $Z _ { t } = \| M _ { t } \|$ , which is a nonnegative submartingale. Define

$$
Z ^ { * } = \operatorname* { m a x } _ { 0 \leq t \leq N } Z _ { t } .
$$

Applying the stopping-time argument at the first crossing of a gives

$$
a \mathbb { P } ( Z ^ { * } \geq a ) \leq \mathbb { E } \left[ Z _ { N } \mathbf { 1 } _ { \{ Z ^ { * } \geq a \} } \right] .
$$

Integrating over $a > 0$ and applying the Cauchy–Schwarz inequality, we obtain

$$
\begin{array} { r l } & { \mathbb { E } [ ( Z ^ { * } ) ^ { 2 } ] \leq 2 \mathbb { E } [ Z _ { N } Z ^ { * } ] } \\ & { \qquad \leq 2 \big ( \mathbb { E } [ Z _ { N } ^ { 2 } ] \big ) ^ { 1 / 2 } \big ( \mathbb { E } [ ( Z ^ { * } ) ^ { 2 } ] \big ) ^ { 1 / 2 } . } \end{array}
$$

Consequently,

$$
\mathbb { E } [ ( Z ^ { * } ) ^ { 2 } ] \leq 4 \mathbb { E } [ Z _ { N } ^ { 2 } ] .
$$

The equality involving the sum of increment variances follows from the orthogonality of martingale increments. The infinite-horizon result then follows by monotone convergence under the stated summability condition. □

## B Proof of Theorem 1

Recall that $\theta _ { r } = ( a _ { r } , w _ { r } ) \in \mathbb { R } ^ { d + 2 }$ denote the parameters of the r-th neuron. For convenience, define

$$
\gamma = 1 - \beta _ { 1 } , \qquad P ^ { ( t ) } = \epsilon D ^ { ( t ) } = \mathrm { d i a g } \left( \frac { \epsilon } { \sqrt { \hat { v } ^ { ( t ) } } + \epsilon } \right) .\tag{37}
$$

By definition, $0 \prec P ^ { ( t ) } \preceq I$ , and the Adam update can be rewritten as

$$
\boldsymbol { \theta } ^ { ( t + 1 ) } = \boldsymbol { \theta } ^ { ( t ) } - \bar { \eta } P ^ { ( t ) } \widehat { s } ^ { ( t ) } .
$$

Throughout the proof, $C , c > 0$ denote universal constants that may change from line to line.

We first establish several lemmas on initialization, loss function regularity, and the convergence of Adam. We then combine these results to prove Theorem 1 by induction.

## B.1 Positive Definiteness of the Limiting Gram Matrix

Let $s _ { i r } = w _ { r } ^ { \top } \widetilde { x } _ { i }$ and $\theta _ { r } = \left( a _ { r } , w _ { r } \right)$ . Direct differentiation gives

$$
\nabla _ { \theta _ { r } } f _ { \theta } ( x _ { i } ) = \frac 1 { \sqrt { m } } \left( { \sigma } _ { a _ { r } \sigma ^ { \prime } ( s _ { i r } ) \widetilde { x } _ { i } } ^ { \sigma ( s _ { i r } ) } \right) ,\tag{38}
$$

$$
\nabla _ { \theta _ { r } } ^ { 2 } f _ { \theta } ( x _ { i } ) = \frac { 1 } { \sqrt { m } } \left( { 0 \atop \sigma ^ { \prime } ( s _ { i r } ) \widetilde { x } _ { i } } \quad { \sigma ^ { \prime } ( s _ { i r } ) \widetilde { x } _ { i } ^ { \top } } \right) .\tag{39}
$$

Consequently, the Gram matrix $K ( \theta ) = J ( \theta ) J ( \theta ) ^ { \top }$ can be expressed as

$$
K ( \theta ) _ { i j } = \frac { 1 } { n m } \sum _ { r = 1 } ^ { m } \left[ \sigma ( s _ { i r } ) \sigma ( s _ { j r } ) + a _ { r } ^ { 2 } \sigma ^ { \prime } ( s _ { i r } ) \sigma ^ { \prime } ( s _ { j r } ) \widetilde { x } _ { i } ^ { \top } \widetilde { x } _ { j } \right] .\tag{40}
$$

Both terms define positive semidefinite matrices. Taking expectations with respect to the initialization (3), we obtain

$$
\begin{array} { l } { { ( { \cal K } _ { \infty } ^ { ( a ) } ) _ { i j } = \displaystyle \frac { 1 } { n } { \mathbb E } _ { w } \left[ \sigma ( w ^ { \top } \widetilde { x } _ { i } ) \sigma ( w ^ { \top } \widetilde { x } _ { j } ) \right] , } } \\ { { \displaystyle ( { \cal K } _ { \infty } ^ { ( w ) } ) _ { i j } = \displaystyle \frac { \widetilde { x } _ { i } ^ { \top } \widetilde { x } _ { j } } { n } { \mathbb E } _ { w } \left[ \sigma ^ { \prime } ( w ^ { \top } \widetilde { x } _ { i } ) \sigma ^ { \prime } ( w ^ { \top } \widetilde { x } _ { j } ) \right] , } } \\ { { \displaystyle { \cal K } _ { \infty } = { \cal K } _ { \infty } ^ { ( a ) } + { \cal K } _ { \infty } ^ { ( w ) } \succeq { \cal K } _ { \infty } ^ { ( w ) } , \qquad \lambda _ { 0 } = \lambda _ { \operatorname* { m i n } } ( { \cal K } _ { \infty } ) . } } \end{array}\tag{41}
$$

Lemma 7. Both $K _ { \infty } ^ { ( w ) }$ and $K _ { \infty }$ are positive definite.

Proof. Since $K _ { \infty } \succeq K _ { \infty } ^ { ( w ) }$ , it suffices to show that $K _ { \infty } ^ { ( w ) }$ is positive definite. For any $c \in \mathbb { R } ^ { n }$ , we have

$$
c ^ { \top } K _ { \infty } ^ { ( w ) } c = \frac { 1 } { n } \mathbb { E } _ { w } \left[ \left\| \sum _ { i = 1 } ^ { n } c _ { i } \sigma ^ { \prime } ( w ^ { \top } \widetilde { x } _ { i } ) \widetilde { x } _ { i } \right\| ^ { 2 } \right] .
$$

Suppose that this quadratic form is zero. Since the integrand is nonnegative and continuous in $w ,$ and the Gaussian distribution has full support on $\mathbb { R } ^ { d + 1 }$ , it follows that

$$
\sum _ { i = 1 } ^ { n } c _ { i } \sigma ^ { \prime } ( w ^ { \top } \widetilde { x } _ { i } ) \widetilde { x } _ { i } = 0 , \qquad \mathrm { f o r ~ a l l ~ } w \in \mathbb { R } ^ { d + 1 } .
$$

We next show that $c _ { k } = 0$ for each $k = 1 , \dots , n .$ . Since the training inputs are pairwise distinct, there exists $v \in \mathbb { R } ^ { d }$ such that $v ^ { \top } x _ { 1 } , \ldots , v ^ { \top } x _ { n }$ are pairwise distinct. Indeed, any vector outside the finite union of proper hyperplanes

$$
\bigcup _ { i < j } \left\{ v \in \mathbb { R } ^ { d } : v ^ { \top } ( x _ { i } - x _ { j } ) = 0 \right\}
$$

satisfies this property. Fix any $k \in [ n ]$ and define

$$
\begin{array} { r } { w _ { A } = ( A v , - A v ^ { \top } x _ { k } ) \in \mathbb R ^ { d + 1 } , \qquad A > 0 . } \end{array}
$$

By the definition $\widetilde { x } _ { i } = ( x _ { i } , 1 ) / \sqrt { 2 }$ , we have

$$
w _ { A } ^ { \top } \widetilde { x } _ { i } = \frac { A } { \sqrt { 2 } } { v } ^ { \top } ( x _ { i } - x _ { k } ) .
$$

For $i = k$ , this expression equals zero, while for $i \neq k$ , it tends to either +∞ or −∞ as $A  \infty$ Since $\sigma =$ tanh satisfies $\sigma ^ { \prime } ( 0 ) = 1$ and $\sigma ^ { \prime } ( s )  0$ as $| s | \to \infty$ , substituting $w = w _ { A }$ into the preceding identity and letting $A \to \infty$ yields

$$
c _ { k } { \widetilde { x } } _ { k } = 0 .
$$

The last coordinate of $\widetilde { x } _ { k }$ is $1 / { \sqrt { 2 } } ,$ so $c _ { k } = 0$ . Since k was arbitrary, we conclude that $c = 0$ □

## B.2 Key Lemmas

Lemma 8 (Initialization). There exist universal constants $C _ { \mathrm { i n i t } } , C _ { 0 } > 0$ such that, for any $\delta \in ( 0 , 1 )$ $i f$

$$
m \geq { \frac { C _ { \mathrm { i n i t } } } { \lambda _ { 0 } } } \log { \frac { 2 n } { \delta } } ,
$$

then, with probability at least $1 - \delta$ over the random initialization (3), we have

$$
K ( { \theta } ^ { ( 0 ) } ) \succeq \frac { 3 } { 4 } { \lambda } _ { 0 } I , \qquad L ( { \theta } ^ { ( 0 ) } ) \leq S _ { 0 } , \qquad S _ { 0 } = C _ { 0 } \log \frac { 2 } { \delta } .\tag{42}
$$

Proof. For any $w \in \mathbb { R } ^ { d + 1 }$ , define the feature vectors

$$
\phi _ { i } ( w ) = \left( { \sigma } ^ { \top } \widetilde { x } _ { i } \right) , \qquad i = 1 , \ldots , n ,
$$

and the associated Gram matrix $H ( w ) \in \mathbb { R } ^ { n \times n }$ with entries

$$
H ( w ) _ { i j } = \frac { 1 } { n } \phi _ { i } ( w ) ^ { \top } \phi _ { j } ( w ) .
$$

Since $( a _ { r } ^ { ( 0 ) } ) ^ { 2 } = 1$ , it follows that

$$
K ( \theta ^ { ( 0 ) } ) = \frac { 1 } { m } \sum _ { r = 1 } ^ { m } H ( w _ { r } ^ { ( 0 ) } ) , \qquad \mathbb { E } [ H ( w _ { r } ^ { ( 0 ) } ) ] = K _ { \infty } .
$$

The matrices $H ( w _ { r } ^ { ( 0 ) } )$ are independent and positive semidefinite. Let $u = \mathrm { t a n h } ^ { 2 } ( s ) \in [ 0 , 1 ]$ . Since $\| \widetilde x _ { i } \| \le 1$ , we have

$$
\operatorname { t a n h } ^ { 2 } ( s ) + \operatorname { t a n h } ^ { \prime } ( s ) ^ { 2 } \| \widetilde { x } _ { i } \| ^ { 2 } \leq u + ( 1 - u ) ^ { 2 } \leq 1 .
$$

Consequently, $\| \phi _ { i } ( w ) \| ^ { 2 } \leq 1$ for all $i = 1 , \ldots , n$ , which implies

$$
\mathrm { t r a c e } ( H ( w ) ) \leq 1 , \qquad 0 \preceq H ( w ) \preceq I .
$$

Applying Lemma 3 with $X _ { r } = H ( w _ { r } ^ { ( 0 ) } ) , B = 1 , \mu _ { \mathrm { m i n } } = m \lambda _ { 0 } ,$ and $\alpha = 1 / 4$ gives

$$
\mathbb { P } \left( \lambda _ { \operatorname* { m i n } } ( K ( \theta ^ { ( 0 ) } ) ) < \frac { 3 } { 4 } \lambda _ { 0 } \right) \le n \exp \left( - \frac { m \lambda _ { 0 } } { 3 2 } \right) .
$$

Under the width condition on $m ,$ , this probability is at most $\delta / 2$

Next, we bound the initial loss. Define

$$
F ^ { ( 0 ) } = \frac { 1 } { \sqrt { n } } \bigl ( f _ { \theta ^ { ( 0 ) } } ( x _ { 1 } ) , \ldots , f _ { \theta ^ { ( 0 ) } } ( x _ { n } ) \bigr ) ^ { \top } = \frac { 1 } { \sqrt { m } } \sum _ { r = 1 } ^ { m } a _ { r } ^ { ( 0 ) } \Phi ( w _ { r } ^ { ( 0 ) } ) ,
$$

where

$$
\Phi ( w ) = \frac { 1 } { \sqrt { n } } \big ( \sigma ( w ^ { \top } \widetilde { x } _ { i } ) \big ) _ { i = 1 } ^ { n } .
$$

Since $\lVert \Phi ( w ) \rVert \leq 1$ and the output weights $a _ { r } ^ { ( 0 ) }$ are independent and centered, we obtain

$$
\begin{array} { r } { \mathbb { E } \| F ^ { ( 0 ) } \| ^ { 2 } \leq 1 , \qquad \mathbb { E } \| F ^ { ( 0 ) } \| \leq 1 . } \end{array}
$$

Moreover, replacing any one of the independent pairs $( a _ { r } ^ { ( 0 ) } , w _ { r } ^ { ( 0 ) } )$ changes $\| F ^ { ( 0 ) } \|$ by at most $2 / { \sqrt { m } }$ Applying Lemma 4 with $b _ { r } = 2 / \sqrt { m }$ and $u = \sqrt { 2 \log ( 2 / \delta ) }$ gives

$$
\mathbb { P } \left( \lVert F ^ { ( 0 ) } \rVert > 1 + \sqrt { 2 \log ( 2 / \delta ) } \right) \leq \frac { \delta } { 2 } .
$$

Since $\| ( y _ { i } / \sqrt { n } ) _ { i = 1 } ^ { n } \| \leq 1$ , on the complementary event we have

$$
L ( \theta ^ { ( 0 ) } ) \leq \frac { 1 } { 2 } \left( 2 + \sqrt { 2 \log ( 2 / \delta ) } \right) ^ { 2 } \leq C _ { 0 } \log \frac { 2 } { \delta } .
$$

Combining the two probability bounds by a union bound completes the proof.

Lemma 9 (Regularity and Gram matrix stability). Define

$$
S = \{ \theta : | a _ { r } | \leq 2 , 1 \leq r \leq m \} .
$$

There exist universal constants $H \geq 1 , C _ { J } , C _ { g } > 0 $ , and $c _ { 0 } > 0$ such that, for any $\theta , \vartheta \in S ,$

$$
\lVert J ( \theta ) \rVert \le 2 , \qquad \lVert J ( \theta ) - J ( \vartheta ) \rVert \le \frac { C _ { J } } { \sqrt { m } } \lVert \theta - \vartheta \rVert ,\tag{43}
$$

$$
\| \nabla ^ { 2 } L ( \theta ) \| \leq H , \qquad \| \nabla _ { \theta _ { r } } L ( \theta ) \| \leq C _ { g } \sqrt { \frac { L ( \theta ) } { m } } , \quad 1 \leq r \leq m .\tag{44}
$$

On the initialization event (42), let

$$
\mathcal { R } = c _ { 0 } \sqrt { m \lambda _ { 0 } } .
$$

Then, every $\theta \in S$ satisfying $\lVert { \boldsymbol { \theta } } - { \boldsymbol { \theta } } ^ { ( 0 ) } \rVert \leq \mathcal { R }$ obeys

$$
K ( \theta ) \succeq \frac { \lambda _ { 0 } } { 4 } I , \qquad \| \nabla L ( \theta ) \| ^ { 2 } \geq \frac { \lambda _ { 0 } } { 2 } L ( \theta ) .\tag{45}
$$

Proof. Recall from (38) and (39) that, with $s _ { i r } = w _ { r } ^ { \top } \widetilde { x } _ { i }$

$$
\begin{array} { r l } & { \nabla _ { \theta _ { r } } f _ { \theta } ( x _ { i } ) = \displaystyle \frac { 1 } { \sqrt { m } } \left( { { \sigma } _ { a _ { r } } } ^ { \sigma } ( { s } _ { i r } ) { { \widetilde x } _ { i } } \right) , } \\ & { \nabla _ { \theta _ { r } } ^ { 2 } f _ { \theta } ( x _ { i } ) = \displaystyle \frac { 1 } { \sqrt { m } } \left( { { \sigma } _ { { } ^ { \prime } } } ^ { 0 } ( { s } _ { i r } ) { { \widetilde x } _ { i } } \quad { { \sigma } ^ { \prime } } ( { s } _ { i r } ) { { \widetilde x } _ { i } } { { \widetilde x } _ { i } } ^ { \top } \right) . } \end{array}
$$

Moreover, the mixed second derivatives between different neurons vanish. Since $| \sigma | \leq 1 , | \sigma ^ { \prime } | \leq 1$ $| \sigma ^ { \prime \prime } | \le 2$ , and $\| \widetilde x _ { i } \| \le 1$ , we have, for any $\theta \in S$

$$
\| \nabla _ { \theta _ { r } } f _ { \theta } ( x _ { i } ) \| \leq \frac { 2 } { \sqrt { m } } , \qquad \| \nabla _ { \theta _ { r } } ^ { 2 } f _ { \theta } ( x _ { i } ) \| \leq \frac { C } { \sqrt { m } } .
$$

To verify the first bound, let $u = \operatorname { t a n h } ^ { 2 } ( s ) \in [ 0 , 1 ]$ , we use

$$
\sigma ( s ) ^ { 2 } + a _ { r } ^ { 2 } \sigma ^ { \prime } ( s ) ^ { 2 } \| \widetilde { x } _ { i } \| ^ { 2 } \leq u + 4 ( 1 - u ) ^ { 2 } \leq 4 .
$$

Summing the squared first-derivative bounds over all samples and neurons gives

$$
\| J ( \theta ) \| \leq \| J ( \theta ) \| _ { \mathrm { F } } \leq 2 .
$$

Since $s$ is convex, integrating the second-derivative bound along the line segment joining $\theta$ and $\vartheta$ gives

$$
\lVert J _ { i , r } ( \theta ) - J _ { i , r } ( \vartheta ) \rVert \leq \frac { C } { \sqrt { n m } } \lVert \theta _ { r } - \vartheta _ { r } \rVert ,
$$

where $J _ { i , \imath }$ <sub>r</sub> denotes the block of row i of $J$ corresponding to neuron r. Squaring and summing over i and r establishes the second inequality in (43).

Next, we establish the bounds in (44). By the Cauchy–Schwarz inequality,

$$
\begin{array} { l } { \displaystyle \| \nabla _ { \theta _ { r } } L ( \theta ) \| \leq \frac { 2 } { \sqrt { m } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | f _ { \theta } ( x _ { i } ) - y _ { i } | } \\ { \leq 2 \sqrt { \frac { 2 L ( \theta ) } { m } } . } \end{array}
$$

For the Hessian of the loss function, we have

$$
\nabla ^ { 2 } L ( \theta ) = J ( \theta ) ^ { \top } J ( \theta ) + \sum _ { i = 1 } ^ { n } R _ { i } ( \theta ) \nabla ^ { 2 } R _ { i } ( \theta ) .
$$

The second term is block diagonal with respect to the neuron parameters, and each block has spectral norm bounded by

$$
{ \frac { C } { \sqrt { n m } } } \sum _ { i = 1 } ^ { n } | R _ { i } ( \theta ) | \leq { \frac { C } { \sqrt { m } } } \| R ( \theta ) \| .
$$

Consequently,

$$
\| \nabla ^ { 2 } L ( \theta ) \| \leq 4 + \frac { C } { \sqrt { m } } \| R ( \theta ) \| \leq H ,
$$

where the last inequality follows from $| f _ { \theta } ( x _ { i } ) | \le 2 \sqrt { m }$ and $\| R ( \theta ) \| \le 2 \sqrt { m } + 1$ . Thus, the loss function is uniformly smooth on S, with a smoothness constant that does not depend on the network width or the hidden weights.

Finally, we establish the lower bound on the Gram matrix. For any unit vector $z \in \mathbb { R } ^ { n }$ , the initialization bound (42) and the Jacobian Lipschitz estimate (43) imply

$$
\begin{array} { r l } & { \| J ( \theta ) ^ { \top } z \| \ge \| J ( \theta ^ { ( 0 ) } ) ^ { \top } z \| - \| J ( \theta ) - J ( \theta ^ { ( 0 ) } ) \| } \\ & { \qquad \ge \sqrt { \displaystyle \frac { 3 \lambda _ { 0 } } { 4 } } - \frac { C _ { J } } { \sqrt { m } } \| \theta - \theta ^ { ( 0 ) } \| . } \end{array}
$$

Choose $c _ { 0 } > 0$ sufficiently small such that $C _ { J } c _ { 0 } \leq 1 / 4$ . Then, whenever $\lVert { \boldsymbol { \theta } } - { \boldsymbol { \theta } } ^ { ( 0 ) } \rVert \leq \mathcal { R }$ , we obtain

$$
\| J ( \theta ) ^ { \top } z \| \geq \frac { \sqrt { \lambda _ { 0 } } } { 2 } .
$$

Since this holds for every unit vector z, it follows that

$$
K ( \theta ) = J ( \theta ) J ( \theta ) ^ { \top } \succeq \frac { \lambda _ { 0 } } { 4 } I .
$$

The gradient inequality in (45) then follows from $\nabla L ( \theta ) = J ( \theta ) ^ { \top } R ( \theta )$ and $\| R ( \theta ) \| ^ { 2 } = 2 L ( \theta )$ □ Lemma 10 (Moment bounds and parameter movement). Suppose that ${ \theta ^ { ( 0 ) } , \dots , \theta ^ { ( T ) } \in \mathcal { S } }$

1. $I f L ( \theta ^ { ( k ) } ) \leq S$ for all $0 \leq k \leq T$ , then

$$
P ^ { ( t ) } \succeq \left( 1 + \frac { C _ { g } \sqrt { S } } { \bar { \epsilon } } \right) ^ { - 1 } I , \qquad 0 \leq t \leq T .\tag{46}
$$

In particular, $i f \bar { \epsilon } \ge C _ { g } \sqrt { S }$ , then $P ^ { ( t ) } \succeq I / 2$ for all $0 \leq t \leq T$ and any $\beta _ { 2 } \in [ 0 , 1 )$ .

## 2. Suppose further that

$$
L ( \theta ^ { ( k ) } ) \leq q ^ { k } L ( \theta ^ { ( 0 ) } ) , \qquad 0 \leq k \leq T ,
$$

where $q \in ( 0 , 1 )$ satisfies $1 - q \leq \gamma / 2$ . Then

$$
\| \widehat { s } _ { r } ^ { ( t ) } \| \leq 2 C _ { g } \sqrt { \frac { L ( \theta ^ { ( 0 ) } ) } { m } } q ^ { t / 2 } , \qquad 0 \leq t \leq T ,\tag{47}
$$

$$
\sum _ { t = 0 } ^ { T } \operatorname* { m a x } _ { 1 \leq r \leq m } \| \theta _ { r } ^ { ( t + 1 ) } - \theta _ { r } ^ { ( t ) } \| \leq \frac { 4 C _ { g } \bar { \eta } } { 1 - q } \sqrt { \frac { L ( \theta ^ { ( 0 ) } ) } { m } } .\tag{48}
$$

Proof. By the bias-correction formulas, the first- and second-moment estimates can be written as convex combinations of the historical gradients and their coordinatewise squares, respectively:

$$
\widehat { \boldsymbol { s } } ^ { ( t ) } = \sum _ { k = 0 } ^ { t } \frac { \gamma \beta _ { 1 } ^ { t - k } } { 1 - \beta _ { 1 } ^ { t + 1 } } \boldsymbol { g } ^ { ( k ) } ,\tag{49}
$$

$$
\widehat { v } ^ { ( t ) } = \sum _ { k = 0 } ^ { t } \frac { ( 1 - \beta _ { 2 } ) \beta _ { 2 } ^ { t - k } } { 1 - \beta _ { 2 } ^ { t + 1 } } g ^ { ( k ) \odot 2 } .\tag{50}
$$

When either momentum parameter is zero, the corresponding bias-corrected moment reduces to the current gradient or its coordinatewise square.

By the block-gradient bound in (44), every coordinate of $g ^ { ( k ) }$ is bounded in absolute value by $C _ { g } \sqrt { S / m }$ . Since the weights in (50) are nonnegative and sum to one, we obtain

$$
\widehat { v } _ { j } ^ { ( t ) } \leq \frac { C _ { g } ^ { 2 } S } { m }
$$

for every coordinate $j$ and all $0 \leq t \leq T$ . Substituting this bound into (37) and using $\epsilon = \bar { \epsilon } / \sqrt { m }$ proves (46).

For the second assertion, combining the block-gradient bound with (49) yields

$$
\lVert \widehat { s } _ { r } ^ { ( t ) } \rVert \leq C _ { g } \sqrt { \frac { L ( \theta ^ { ( 0 ) } ) } { m } } \sum _ { k = 0 } ^ { t } \frac { \gamma \beta _ { 1 } ^ { t - k } } { 1 - \beta _ { 1 } ^ { t + 1 } } q ^ { k / 2 } .
$$

For $\beta _ { 1 } > 0$ , let $z = \beta _ { 1 } / \sqrt { q }$ . Since $1 - q \le \gamma / 2$ , we have $z < 1$ and $1 - z \ge \gamma / 2$ . Indeed, setting $a = { \sqrt { q } } .$ , we obtain

$$
1 - a \le 1 - q \le \frac { \gamma } { 2 } , \qquad 1 - z = \frac { a - \beta _ { 1 } } { a } \ge \frac { \gamma } { 2 } .
$$

Consequently, the finite geometric sum satisfies

$$
\begin{array} { l } { { \displaystyle \sum _ { k = 0 } ^ { t } \frac { \gamma \beta _ { 1 } ^ { t - k } } { 1 - \beta _ { 1 } ^ { t + 1 } } q ^ { k / 2 } = q ^ { t / 2 } \frac { \gamma } { 1 - z } \frac { 1 - z ^ { t + 1 } } { 1 - \beta _ { 1 } ^ { t + 1 } } } } \\ { { \le 2 q ^ { t / 2 } , } } \end{array}
$$

where the last inequality follows from $z \geq \beta _ { 1 }$ . The same bound holds directly when $\beta _ { 1 } = 0$ , proving (47).

Finally, since $0 \prec P ^ { ( t ) } \preceq I .$ , the Adam update gives

$$
\operatorname* { m a x } _ { 1 \leq r \leq m } \| \theta _ { r } ^ { ( t + 1 ) } - \theta _ { r } ^ { ( t ) } \| \leq 2 C _ { g } \bar { \eta } \sqrt { \frac { L ( \theta ^ { ( 0 ) } ) } { m } } q ^ { t / 2 } .
$$

Summing over $t = 0 , \ldots , T$ and using $1 - \sqrt { q } \geq ( 1 - q ) / 2$ establishes (48).

Lemma 11 (Lyapunov descent and parameter movement). Let H be defined in Lemma 9, and suppose that

$$
0 < \bar { \eta } \leq \frac { \gamma } { 8 H } .
$$

Define

$$
\boldsymbol { e } ^ { ( t ) } = \widehat { \boldsymbol { s } } ^ { ( t ) } - \boldsymbol { g } ^ { ( t ) } , \qquad \ \boldsymbol { \mathcal { E } } ^ { ( t ) } = \boldsymbol { L } ( \boldsymbol { \theta } ^ { ( t ) } ) + \frac { 4 \bar { \eta } } { \gamma } \| \boldsymbol { e } ^ { ( t ) } \| ^ { 2 } , \qquad \boldsymbol { Q } ^ { ( t ) } = ( \boldsymbol { g } ^ { ( t ) } ) ^ { \top } \boldsymbol { P } ^ { ( t ) } \boldsymbol { g } ^ { ( t ) } .\tag{51}
$$

Suppose that, at iteration t,

$$
[ \theta ^ { ( t ) } , \theta ^ { ( t + 1 ) } ] \subset \mathcal { S } , \qquad K ( \theta ^ { ( t ) } ) \succeq \frac { \lambda _ { 0 } } { 4 } I , \qquad \frac { 1 } { 2 } I \preceq P ^ { ( t ) } \preceq I ,
$$

where $[ \theta ^ { ( t ) } , \theta ^ { ( t + 1 ) } ]$ denotes the line segment joining $\boldsymbol { \theta } ^ { ( t ) }$ and $\theta ^ { ( t + 1 ) }$ . Then, with $q = 1 - { \bar { \eta } } \lambda _ { 0 } / 3 2 \in$ (0, 1), we have

$$
\Delta ^ { ( t ) } : = \mathcal { E } ^ { ( t ) } - \mathcal { E } ^ { ( t + 1 ) } \geq \frac { \bar { \eta } } { 4 } Q ^ { ( t ) } + \bar { \eta } \Vert e ^ { ( t ) } \Vert ^ { 2 } ,\tag{52}
$$

$$
\begin{array} { r } { \mathcal { E } ^ { ( t + 1 ) } \leq q \mathcal { E } ^ { ( t ) } , } \end{array}\tag{53}
$$

$$
\lVert { \boldsymbol { \theta } } ^ { ( t + 1 ) } - { \boldsymbol { \theta } } ^ { ( t ) } \rVert ^ { 2 } \leq 8 \bar { \eta } \Delta ^ { ( t ) } .\tag{54}
$$

Moreover, ifthese conditions holdfor all $0 \leq t \leq T$ , then

$$
\sum _ { t = 0 } ^ { T } \lVert { \boldsymbol { \theta } } ^ { ( t + 1 ) } - { \boldsymbol { \theta } } ^ { ( t ) } \rVert \leq 3 2 \sqrt { \frac { L ( { \boldsymbol { \theta } } ^ { ( 0 ) } ) } { \lambda _ { 0 } } } .\tag{55}
$$

Proof. We first establish the Lyapunov descent analysis. The proof uses the smoothness of $L$ along the update segment, while the lower bound on the Gram matrix is needed only at the current iterate $\boldsymbol { \theta } ^ { ( t ) }$

Since $( P ^ { ( t ) } ) ^ { 2 } \preceq P ^ { ( t ) } \preceq I$ , the Adam update yields

$$
\lVert { \boldsymbol { \theta } } ^ { ( t + 1 ) } - { \boldsymbol { \theta } } ^ { ( t ) } \rVert ^ { 2 } \leq 2 \bar { \eta } ^ { 2 } \big ( Q ^ { ( t ) } + \lVert { \boldsymbol { e } } ^ { ( t ) } \rVert ^ { 2 } \big ) .\tag{56}
$$

Moreover, the weighted Cauchy–Schwarz and Young’s inequalities give

$$
| ( g ^ { ( t ) } ) ^ { \top } P ^ { ( t ) } e ^ { ( t ) } | \leq \frac { 1 } { 4 } Q ^ { ( t ) } + \| e ^ { ( t ) } \| ^ { 2 } .
$$

Applying the smoothness inequality along the update segment and using $H \bar { \eta } \leq 1 / 8$ , we obtain

$$
\begin{array} { r l } & { L ( \theta ^ { ( t + 1 ) } ) - L ( \theta ^ { ( t ) } ) \leq - \bar { \eta } Q ^ { ( t ) } - \bar { \eta } ( g ^ { ( t ) } ) ^ { \top } P ^ { ( t ) } e ^ { ( t ) } + \displaystyle \frac { H } { 2 } \| \theta ^ { ( t + 1 ) } - \theta ^ { ( t ) } \| ^ { 2 } } \\ & { \qquad \leq - \displaystyle \frac { \bar { \eta } } { 2 } Q ^ { ( t ) } + 2 \bar { \eta } \| e ^ { ( t ) } \| ^ { 2 } . } \end{array}\tag{57}
$$

Next, we control the tracking error $e ^ { ( t ) }$ . The exact bias-correction formula gives

$$
\widehat { s } ^ { ( t + 1 ) } = d _ { t + 1 } \widehat { s } ^ { ( t ) } + ( 1 - d _ { t + 1 } ) g ^ { ( t + 1 ) } , \qquad d _ { t + 1 } = \frac { \beta _ { 1 } ( 1 - \beta _ { 1 } ^ { t + 1 } ) } { 1 - \beta _ { 1 } ^ { t + 2 } } \leq \beta _ { 1 } .
$$

Consequently,

$$
e ^ { ( t + 1 ) } = d _ { t + 1 } \big ( e ^ { ( t ) } + g ^ { ( t ) } - g ^ { ( t + 1 ) } \big ) .
$$

For $\beta _ { 1 } > 0$ , applying $\| u + v \| ^ { 2 } \leq \| u \| ^ { 2 } / \beta _ { 1 } + \| v \| ^ { 2 } / \gamma$ yields

$$
\begin{array} { l } { \displaystyle \| e ^ { ( t + 1 ) } \| ^ { 2 } \leq \beta _ { 1 } \| e ^ { ( t ) } \| ^ { 2 } + \frac { \beta _ { 1 } ^ { 2 } } { \gamma } \| g ^ { ( t + 1 ) } - g ^ { ( t ) } \| ^ { 2 } } \\ { \leq \beta _ { 1 } \| e ^ { ( t ) } \| ^ { 2 } + \frac { 2 \beta _ { 1 } ^ { 2 } H ^ { 2 } \bar { \eta } ^ { 2 } } { \gamma } \big ( Q ^ { ( t ) } + \| e ^ { ( t ) } \| ^ { 2 } \big ) , } \end{array}\tag{58}
$$

where the second inequality follows from the smoothness of $L$ along the update segment and (56).   
When $\beta _ { 1 } = 0$ , we have $e ^ { ( t ) } = 0$ for all $t ,$ and the subsequent estimates hold directly.

Multiplying (58) by $4 \bar { \eta } / \gamma$ and combining it with (57), we obtain

$$
\mathcal { E } ^ { ( t + 1 ) } - \mathcal { E } ^ { ( t ) } \leq - \frac { 3 \bar { \eta } } { 8 } Q ^ { ( t ) } - \frac { 1 5 \bar { \eta } } { 8 } \| e ^ { ( t ) } \| ^ { 2 } ,
$$

where we use $8 ( H \bar { \eta } / \gamma ) ^ { 2 } \leq 1 / 8$ . This establishes (52). Furthermore, the lower bounds on the Gram matrix and the normalized preconditioner imply

$$
Q ^ { ( t ) } \geq \frac { 1 } { 2 } \| g ^ { ( t ) } \| ^ { 2 } \geq \frac { \lambda _ { 0 } } { 4 } L ( \theta ^ { ( t ) } ) .
$$

It follows that

$$
\begin{array} { l } { \displaystyle \Delta ^ { ( t ) } \geq \frac { \bar { \eta } \lambda _ { 0 } } { 1 6 } L ( \theta ^ { ( t ) } ) + \bar { \eta } \| e ^ { ( t ) } \| ^ { 2 } } \\ { \displaystyle \geq \frac { \bar { \eta } \lambda _ { 0 } } { 3 2 } \mathcal { E } ^ { ( t ) } , } \end{array}
$$

where the last inequality uses $\lambda _ { 0 } \leq 1 , H \geq 1$ , and $\bar { \eta } \leq \gamma / ( 8 H )$ . This proves the contraction estimate (53). Combining (56) with (52) further yields (54).

Finally, we establish the bound on the cumulative parameter movement. To avoid an additional factor of $\dot { \lambda } _ { 0 } ^ { - 1 / 2 }$ , we use a telescoping argument based on the square root of the Lyapunov function. Since $1 - q = \bar { \eta } \lambda _ { 0 } / 3 2$ , the contraction estimate implies

$$
\Delta ^ { ( t ) } \geq ( 1 - q ) \mathcal { E } ^ { ( t ) } .
$$

Whenever $\mathcal { E } ^ { ( t ) } > 0$ , we have

$$
\begin{array} { l } { \sqrt { \Delta ^ { ( t ) } } = \displaystyle \frac { \sqrt { \mathcal { E } ^ { ( t ) } } + \sqrt { \mathcal { E } ^ { ( t + 1 ) } } } { \sqrt { \Delta ^ { ( t ) } } } \left( \sqrt { \mathcal { E } ^ { ( t ) } } - \sqrt { \mathcal { E } ^ { ( t + 1 ) } } \right) } \\ { \leq \displaystyle \frac { 2 } { \sqrt { 1 - q } } \left( \sqrt { \mathcal { E } ^ { ( t ) } } - \sqrt { \mathcal { E } ^ { ( t + 1 ) } } \right) . } \end{array}
$$

If $\mathcal { E } ^ { ( t ) } = 0$ , the update vanishes and the same inequality holds trivially. Taking square roots in (54) and summing over $t = 0 , \ldots , T$ gives

$$
\begin{array} { r l r } & { } & { \displaystyle \sum _ { t = 0 } ^ { T } \| \theta ^ { ( t + 1 ) } - \theta ^ { ( t ) } \| \leq \sqrt { 8 \bar { \eta } } \sum _ { t = 0 } ^ { T } \sqrt { \Delta ^ { ( t ) } } } \\ & { } & { \leq 2 \sqrt { \frac { 8 \bar { \eta } } { 1 - q } } \sqrt { { \mathscr E } ^ { ( 0 ) } } . } \end{array}
$$

By exact bias correction, $\widehat { s } ^ { ( 0 ) } = g ^ { ( 0 ) } , \mathrm { s o } \mathcal { E } ^ { ( 0 ) } = L ( \theta ^ { ( 0 ) } )$ . Substituting $1 - q = \bar { \eta } \lambda _ { 0 } / 3 2$ completes the proof of (55). □

## B.3 Completion of the Proof

ProofofTheorem 1. We work on the initialization event established in Lemma 8. If $L _ { 0 } = 0$ , then $g ^ { ( 0 ) } = 0$ , and all subsequent gradients and moment estimates remain zero. Thus, the conclusion holds trivially. Henceforth, assume that $L _ { 0 } > 0$

Choose the universal constants in the theorem such that

$$
\bar { \epsilon } \geq C _ { g } \sqrt { S _ { 0 } } , \qquad 0 < \bar { \eta } \leq { \frac { \gamma } { 8 H } } , \qquad m \geq C _ { * } { \frac { S _ { 0 } } { \lambda _ { 0 } ^ { 2 } } } ,\tag{59}
$$

where $C _ { * } > 0$ is a sufficiently large universal constant to be specified below. Let

$$
q = 1 - \frac { \bar { \eta } \lambda _ { 0 } } { 3 2 } , \qquad \mathcal { R } = c _ { 0 } \sqrt { m \lambda _ { 0 } } ,
$$

as defined in the preceding lemmas. Under these conditions, $q \in ( 0 , 1 )$ and $1 - q \le \gamma / 2$

We prove by induction that, for every integer $t \geq 0$

$$
\begin{array} { c } { { \displaystyle \mathcal { E } ^ { ( t ) } \leq q ^ { t } L _ { 0 } , \qquad \displaystyle \operatorname* { m a x } _ { 1 \leq r \leq m } \| \theta _ { r } ^ { ( t ) } - \theta _ { r } ^ { ( 0 ) } \| \leq \frac { 1 } { 2 } , } } \\ { { \| \theta ^ { ( t ) } - \theta ^ { ( 0 ) } \| \leq \displaystyle \frac { \mathcal { R } } { 2 } . } } \end{array}\tag{60}
$$

The assertions hold at $t = 0$ , since $\mathcal { E } ^ { ( 0 ) } = L _ { 0 }$ and the parameter differences vanish. Suppose that they hold for all $0 \leq t \leq T$ . Then, for every $0 \leq k \leq T$ , we have

$$
| a _ { r } ^ { ( k ) } | \leq { \frac { 3 } { 2 } } , \qquad L ( \theta ^ { ( k ) } ) \leq q ^ { k } L _ { 0 } .
$$

In particular, $\theta ^ { ( k ) } \in { \mathcal { S } }$ for all $0 \leq k \leq T$ . By (46) and (59), we obtain $P ^ { ( T ) } \succeq I / 2$ . Moreover, the Gram matrix lower bound in (45) yields

$$
K ( { \boldsymbol { \theta } } ^ { ( T ) } ) \succeq \frac { \lambda _ { 0 } } { 4 } I .
$$

We next verify that the update segment from $\boldsymbol { \theta ^ { ( T ) } } \mathrm { t o } \boldsymbol { \theta ^ { ( T + 1 ) } }$ lies in S. By Lemma 10, which only requires the iterates through time $T$ and the bound $P ^ { ( t ) } \preceq I$ , we have

$$
\sum _ { t = 0 } ^ { T } \operatorname* { m a x } _ { 1 \leq r \leq m } \Vert \theta _ { r } ^ { ( t + 1 ) } - \theta _ { r } ^ { ( t ) } \Vert \leq \frac { 1 2 8 C _ { g } } { \lambda _ { 0 } } \sqrt { \frac { L _ { 0 } } { m } } \leq \frac { 1 } { 2 } ,\tag{61}
$$

where the last inequality follows by choosing $C _ { * }$ sufficiently large. Consequently,

$$
\operatorname* { m a x } _ { 1 \leq r \leq m } \| \theta _ { r } ^ { ( T + 1 ) } - \theta _ { r } ^ { ( 0 ) } \| \leq \frac { 1 } { 2 } ,
$$

which also implies $| a _ { r } ^ { ( T + 1 ) } | \leq 3 / 2$ for every r. Since S is convex, the entire segment $[ \theta ^ { ( T ) } , \theta ^ { ( T + 1 ) } ]$ lies in ${ \mathcal { S } } .$ . Thus, the smoothness condition required by Lemma 11 is verified before applying the descent estimate.

All the hypotheses of Lemma 11 are now satisfied at iteration T. Therefore,

$$
\mathcal { E } ^ { ( T + 1 ) } \leq q \mathcal { E } ^ { ( T ) } \leq q ^ { T + 1 } L _ { 0 } .
$$

Applying the cumulative parameter movement bound in the same lemma over iterations $0 , \ldots , T .$ we further obtain

$$
\begin{array} { r l r } {  { \| \theta ^ { ( T + 1 ) } - \theta ^ { ( 0 ) } \| \le \sum _ { t = 0 } ^ { T } \| \theta ^ { ( t + 1 ) } - \theta ^ { ( t ) } \| } } \\ & { } & \\ & { } & { \leq 3 2 \sqrt { \frac { L _ { 0 } } { \lambda _ { 0 } } } } \\ & { } & { \leq \frac { c _ { 0 } } { 2 } \sqrt { m \lambda _ { 0 } } = \frac { \mathcal { R } } { 2 } , } \end{array}
$$

where the last inequality again follows by taking $C _ { * }$ sufficiently large. Together with (61), this verifies all three assertions in (60) at time $T + 1$ , completing the induction. Consequently,

$$
L ( \theta ^ { ( t ) } ) \leq \mathcal { E } ^ { ( t ) } \leq \left( 1 - \frac { \bar { \eta } \lambda _ { 0 } } { 3 2 } \right) ^ { t } L ( \theta ^ { ( 0 ) } ) , \qquad t \geq 0 .
$$

It remains to verify that the conditions in (59) follow from the assumptions of Theorem 1. Lemma 8 requires m $\gtrsim \lambda _ { 0 } ^ { - 1 }$ log(2n/δ), while the induction requires $m \ \gtrsim \ S _ { 0 } / \lambda _ { 0 } ^ { 2 }$ . Since $S _ { 0 } ~ =$ $C _ { 0 } \log ( 2 / \delta )$ and $\lambda _ { 0 } \leq 1$ , we have

$$
\begin{array} { r } { \displaystyle \frac { 1 } { \lambda _ { 0 } } \log \displaystyle \frac { 2 n } { \delta } = \displaystyle \frac { 1 } { \lambda _ { 0 } } \log \displaystyle \frac { n } { \delta } + \displaystyle \frac { \log 2 } { \lambda _ { 0 } } \qquad } \\ { \displaystyle \leq \displaystyle \frac { 1 } { \lambda _ { 0 } } \log \displaystyle \frac { n } { \delta } + \displaystyle \frac { 1 } { \lambda _ { 0 } ^ { 2 } } \log \displaystyle \frac { 2 } { \delta } . } \end{array}
$$

Thus, both width conditions follow from (11), up to universal constants. Similarly, the damping and stepsize conditions in (59) are ensured by (10).

Finally, the finite-length bound (61) on the trajectory implies that $\{ \theta ^ { ( t ) } \}$ converges to a finite limit $\theta ^ { \star }$ . By the continuity of $L$ and (12), we have $L ( \theta ^ { \star } ) = 0$

## C Proof of Theorem 2

We use the network, loss function, initialization, and Adam iteration defined in the main text, with $\epsilon = \bar { \epsilon } / \sqrt { m } \mathrm { a n d } \eta = \bar { \eta } \epsilon$

For convenience, define

$$
\gamma = 1 - \beta _ { 1 } , \qquad \nu = 1 - \beta _ { 2 } , \qquad \rho = \rho _ { b } , \qquad \chi _ { b } = \frac { n - b } { b ( n - 1 ) } , \qquad \kappa = \frac { n } { b } .\tag{62}
$$

In particular, $\rho = 1 + \chi _ { b } / \lambda _ { 0 }$ . Define the filtration

$$
\mathcal F _ { t } = \sigma \big ( \theta ^ { ( 0 ) } , B ^ { ( 0 ) } , \ldots , B ^ { ( t - 1 ) } \big ) , \qquad \mathbb { E } _ { t } [ \cdot ] = \mathbb { E } [ \cdot \mid \mathcal F _ { t } ] .
$$

Then $\boldsymbol { \theta } ^ { ( t ) }$ and the moment estimates up to iteration t − 1 are $\mathcal { F } _ { t }$ -measurable. Unless otherwise stated, probabilities and expectations are conditional on a fixed initialization. We use $\mathcal { G }$ to denote the final high-probability sampling event.

We will make use of the initialization and regularity estimates established in Lemmas 8 and 9. In particular, the trace bound in the proof of Lemma 8 implies $\kappa \leq \rho$ . We also define

$$
T _ { * } = 1 + \left\lceil \frac { \log 2 } { - \log \beta _ { 2 } } \right\rceil .
$$

The proof consists of three main steps: constructing an auxiliary network with global regularity, establishing stochastic moment bounds and a conditional Lyapunov descent analysis, and controlling parameter movement through a stopping argument. We then use these results to prove the convergence guarantee for the Adam iterates.

## C.1 An Auxiliary Network with Global Regularity

Lemma 9 establishes the smoothness of the original loss function on $S = \{ \theta : | a _ { r } | \leq 2 , 1 \leq r \leq$ m}. In the stochastic setting, however, the conditional descent analysis must account for all possible realizations of the next minibatch. Even when the current iterate lies in ${ \mathcal { S } } ,$ the next update and the virtual iterates introduced later may leave this region. Consequently, the local smoothness bound alone is insufficient to justify the descent estimates. To overcome this difficulty, we construct an auxiliary network whose loss function is globally smooth and coincides with the original loss on S. We first analyze Adam on this auxiliary network and then show that its iterates coincide with those of the original network on a suitable high-probability event.

Let $\psi \in C _ { c } ^ { \infty } ( \mathbb { R } )$ be a fixed smooth function satisfying

$$
\psi ( a ) = a \quad \mathrm { f o r } | a | \leq 2 , \qquad \mathrm { s u p p } \psi \subset [ - 3 , 3 ] .
$$

Such a function can be constructed by multiplying a by a smooth cutoff function that equals one on $[ - 2 , 2 ]$ and vanishes outside $[ - 3 , 3 ]$ . We choose ψ independently of the training data, network width, and optimizer parameters, so that ψ and its first two derivatives are bounded by universal constants. Using the same parameter vector $\theta = ( a _ { 1 } , w _ { 1 } , \dots , a _ { m } , w _ { m } )$ , we define the auxiliary network

$$
f _ { \theta } ^ { \dagger } ( x ) = \frac { 1 } { \sqrt { m } } \sum _ { r = 1 } ^ { m } \psi ( a _ { r } ) \sigma ( w _ { r } ^ { \top } \widetilde { x } ) .\tag{63}
$$

Here, the superscript $\dagger$ denotes quantities associated with the auxiliary network. Correspondingly, we define the normalized residual vector, loss function, Jacobian, and Gram matrix by

$$
\begin{array} { l } { { R _ { i } ^ { \dagger } ( \theta ) = \displaystyle \frac { f _ { \theta } ^ { \dagger } ( x _ { i } ) - y _ { i } } { \sqrt { n } } } , \quad { \cal L } ^ { \dagger } ( \theta ) = \displaystyle \frac { 1 } { 2 } \| R ^ { \dagger } ( \theta ) \| ^ { 2 } , }  \\ { { J ^ { \dagger } ( \theta ) = \nabla R ^ { \dagger } ( \theta ) , \quad \quad K ^ { \dagger } ( \theta ) = J ^ { \dagger } ( \theta ) J ^ { \dagger } ( \theta ) ^ { \top } . } } \end{array}\tag{64}
$$

Lemma 12 (Properties of the auxiliary network). The auxiliary network satisfies the following properties.

1. For any $\theta \in S$ and input x, we have

$$
f _ { \theta } ^ { \dagger } ( x ) = f _ { \theta } ( x ) , \qquad \nabla _ { \theta } f _ { \theta } ^ { \dagger } ( x ) = \nabla _ { \theta } f _ { \theta } ( x ) .\tag{65}
$$

Consequently, $R ^ { \dagger } = R , L ^ { \dagger } = L , J ^ { \dagger } = J ,$ , and $K ^ { \dagger } = K$ on S. In particular, these identities hold at initialization.

2. There exists a universal constant $C > 0$ such that,for all θ, $\vartheta \in \mathbb { R } ^ { m ( d + 2 ) }$

$$
\| J ^ { \dagger } ( \theta ) \| \leq C , \qquad \| J ^ { \dagger } ( \theta ) - J ^ { \dagger } ( \vartheta ) \| \leq \frac { C } { \sqrt { m } } \| \theta - \vartheta \| ,
$$

$$
\| \nabla ^ { 2 } L ^ { \dag } ( \theta ) \| \leq C , \qquad \operatorname* { m a x } _ { i , r } \| \nabla _ { \theta _ { r } } f _ { \theta } ^ { \dag } ( x _ { i } ) \| \leq \frac { C } { \sqrt { m } } .\tag{66}
$$

3. On the initialization event (42), let

$$
\mathcal { R } = c _ { 0 } \sqrt { m \lambda _ { 0 } } ,
$$

where $c _ { 0 } > 0$ is a sufficiently small universal constant, chosen as in Lemma 9. Then, for every θ satisfying $\lVert { \boldsymbol { \theta } } - { \boldsymbol { \theta } } ^ { ( 0 ) } \rVert \leq \mathcal { R }$ , we have

$$
K ^ { \dagger } ( \theta ) \succeq \frac { \lambda _ { 0 } } { 4 } I , \qquad \| \nabla L ^ { \dagger } ( \theta ) \| ^ { 2 } \geq \frac { \lambda _ { 0 } } { 2 } L ^ { \dagger } ( \theta ) .\tag{67}
$$

The constants depend only on thefixed activation and cutofffunctions.

Proof. Since $\psi ( a ) = a$ and $\psi ^ { \prime } ( a ) = 1 \mathrm { f o r } | a | \leq 2$ , direct differentiation of (63) gives (65). Moreover, $| a _ { r } ^ { ( 0 ) } | = 1$ for all $^ { r , }$ so the auxiliary and original networks agree at initialization. In particular, their Gram matrices agree at initialization, and the auxiliary network has the same limiting Gram matrix $K _ { \infty }$ and minimum eigenvalue $\lambda _ { 0 }$

Next, we establish the global regularity bounds. The first- and second-order derivatives with respect to the parameters of neuron r are

$$
\nabla _ { \theta _ { r } } f _ { \theta } ^ { \dagger } ( x _ { i } ) = \frac { 1 } { \sqrt { m } } \left( \begin{array} { c } { { \psi ^ { \prime } ( a _ { r } ) \sigma ( s _ { i r } ) } } \\ { { \psi ( a _ { r } ) \sigma ^ { \prime } ( s _ { i r } ) \widetilde { x } _ { i } } } \end{array} \right) ,
$$

and

$$
\nabla _ { \theta _ { r } } ^ { 2 } f _ { \theta } ^ { \dagger } ( x _ { i } ) = \frac { 1 } { \sqrt { m } } \left( \begin{array} { l l } { { \psi ^ { \prime \prime } ( a _ { r } ) \sigma ( s _ { i r } ) } } & { { \psi ^ { \prime } ( a _ { r } ) \sigma ^ { \prime } ( s _ { i r } ) \widetilde { x } _ { i } ^ { \top } } } \\ { { \psi ^ { \prime } ( a _ { r } ) \sigma ^ { \prime } ( s _ { i r } ) \widetilde { x } _ { i } } } & { { \psi ( a _ { r } ) \sigma ^ { \prime \prime } ( s _ { i r } ) \widetilde { x } _ { i } \widetilde { x } _ { i } ^ { \top } } } \end{array} \right) .
$$

Since $\psi , \sigma ,$ and their derivatives up to second order are bounded, and $\| \widetilde { x } _ { i } \| \leq 1$ , both derivatives are bounded in norm by $C / \sqrt { m }$ uniformly over the parameter space. Furthermore, mixed second derivatives between different neurons vanish. As in the proof of Lemma 9, we obtain

$$
\| J ^ { \dagger } ( \theta ) \| \leq \| J ^ { \dagger } ( \theta ) \| _ { \mathrm { F } } \leq C ,
$$

and

$$
\lVert J ^ { \dagger } ( \theta ) - J ^ { \dagger } ( \vartheta ) \rVert _ { \mathrm { F } } ^ { 2 } \leq \frac { C } { m } \sum _ { r = 1 } ^ { m } \lVert \theta _ { r } - \vartheta _ { r } \rVert ^ { 2 } .
$$

These estimates establish the Jacobian bounds and the neuron-wise gradient bound in (66).

To establish the global smoothness of the auxiliary loss, we use

$$
\nabla ^ { 2 } L ^ { \dagger } ( \theta ) = ( J ^ { \dagger } ( \theta ) ) ^ { \top } J ^ { \dagger } ( \theta ) + \sum _ { i = 1 } ^ { n } R _ { i } ^ { \dagger } ( \theta ) \nabla ^ { 2 } R _ { i } ^ { \dagger } ( \theta ) .
$$

The second term is block diagonal with respect to the neuron parameters. By the same Cauchy– Schwarz argument as in Lemma 9, its operator norm is bounded by $\frac { C } { \sqrt { m } } \lVert R ^ { \dagger } ( \theta ) \dot { \parallel }$ . Since $\psi$ and $\sigma$ are bounded, we have $| f _ { \theta } ^ { \dagger } ( x _ { i } ) | \le C \sqrt { m }$ and hence $\lVert R ^ { \dagger } ( \theta ) \rVert \leq C \sqrt { m } + 1$ . Consequently,

$$
\lVert \nabla ^ { 2 } L ^ { \dag } ( \theta ) \rVert \leq C + \frac { C } { \sqrt { m } } \lVert R ^ { \dag } ( \theta ) \rVert \leq C .
$$

Thus, $L ^ { \dagger }$ is globally smooth, with a smoothness constant that does not depend on the network width. Finally, we establish the lower bound on the auxiliary Gram matrix. On the initialization event (42), the agreement between the two networks gives

$$
K ^ { \dagger } ( \theta ^ { ( 0 ) } ) = K ( \theta ^ { ( 0 ) } ) \succeq \frac { 3 \lambda _ { 0 } } { 4 } I .
$$

For any unit vector $z \in \mathbb { R } ^ { n }$ , the Jacobian Lipschitz bound implies

$$
\begin{array} { r l } & { \| ( J ^ { \dagger } ( \theta ) ) ^ { \top } z \| \ge \| ( J ^ { \dagger } ( \theta ^ { ( 0 ) } ) ) ^ { \top } z \| - \| J ^ { \dagger } ( \theta ) - J ^ { \dagger } ( \theta ^ { ( 0 ) } ) \| } \\ & { \quad \ge \sqrt { \frac { 3 \lambda _ { 0 } } { 4 } } - \frac { C } { \sqrt { m } } \| \theta - \theta ^ { ( 0 ) } \| } \\ & { \quad \ge \frac { \sqrt { \lambda _ { 0 } } } { 2 } , } \end{array}
$$

where the last inequality holds whenever $\lVert { \boldsymbol { \theta } } - { \boldsymbol { \theta } } ^ { ( 0 ) } \rVert \leq \mathcal { R }$ , provided that $c _ { 0 }$ is sufficiently small. Since this holds for every unit vector z, we obtain

$$
K ^ { \dagger } ( \theta ) \succeq \frac { \lambda _ { 0 } } { 4 } I .
$$

The gradient inequality in (67) follows from $\nabla L ^ { \dagger } ( \theta ) = ( J ^ { \dagger } ( \theta ) ) ^ { \top } R ^ { \dagger } ( \theta )$ and $\lVert R ^ { \dagger } ( \theta ) \rVert ^ { 2 } = 2 L ^ { \dagger } ( \theta )$

Auxiliary Adam and trajectory agreement. We apply Adam to the auxiliary loss $L ^ { \dagger }$ , using the same optimizer parameters, initialization, and minibatch sequence as the original algorithm. Denote the auxiliary iterates and moment estimates by $\theta _ { \dagger } ^ { ( t ) } , s _ { \dagger } ^ { ( t ) }$ , and $v _ { \dagger } ^ { ( t ) }$ , respectively, where $\theta _ { \dagger , r } ^ { ( t ) } = $ $( a _ { \dagger , r } ^ { ( t ) } , w _ { \dagger , r } ^ { ( t ) } )$ . We initialize

$$
\theta _ { \dagger } ^ { ( 0 ) } = \theta ^ { ( 0 ) } , \qquad s _ { \dagger } ^ { ( - 1 ) } = v _ { \dagger } ^ { ( - 1 ) } = 0 ,
$$

and compute the stochastic gradient

$$
g _ { \dagger } ^ { ( t ) } = \frac { 1 } { b } \sum _ { i \in B ^ { ( t ) } } \bigl ( f _ { \theta _ { \dagger } ^ { ( t ) } } ^ { \dagger } ( x _ { i } ) - y _ { i } \bigr ) \nabla _ { \theta } f _ { \theta _ { \dagger } ^ { ( t ) } } ^ { \dagger } ( x _ { i } ) .
$$

The moment estimates, bias corrections, and parameter updates follow (4)–(5), with the corresponding auxiliary quantities.

We now establish the agreement between the auxiliary and original Adam trajectories. For any integer $T \geq 0 , \operatorname { i f } \theta _ { \dagger } ^ { ( t ) } \in \mathcal { S }$ for all $0 \leq t \leq T$ , then the two algorithms generate identical parameter iterates through time $T + 1$ , as well as identical stochastic gradients and moment estimates through time $T$ . Indeed, both algorithms start from the same initialization. Whenever their iterates coincide at a point in ${ \mathcal { S } } ,$ , the agreement property (65) and the shared minibatch imply identical stochastic gradients. The moment recurrences and parameter updates are therefore identical, and the claim follows by induction. In the proof of Theorem 2, we establish a high-probability event $\mathcal { G }$ on which $| a _ { \uparrow , r } ^ { ( t ) } | \leq 3 / 2$ for all $r$ and t. Hence, the auxiliary and original Adam trajectories coincide for all iterations on $\mathcal { G }$

Notation for the remaining proof. For notational simplicity, we omit the superscript or subscript † from all auxiliary quantities in the following analysis, until the final comparison with the original algorithm. In particular, $\theta ^ { ( t ) } , \ L ( \theta ^ { ( t ) } )$ , and $g ^ { ( t ) }$ denote $\theta _ { \dagger } ^ { ( t ) } , \ { \cal L } ^ { \dagger } ( \theta _ { \dagger } ^ { ( t ) } )$ , and $g _ { \dagger } ^ { ( t ) }$ , respectively, and we write

$$
G ^ { ( t ) } = \nabla L ^ { \dagger } ( \theta _ { \dagger } ^ { ( t ) } )
$$

for the full gradient of the auxiliary loss. The same convention applies to $f , R , J , K , s , v , \widehat { s } , \widehat { v } , D$ , and the normalized preconditioner $P .$ . The filtration $\mathcal { F } _ { t }$ remains unchanged because the two algorithms share the same initialization and minibatch sequence. Throughout the subsequent analysis, global smoothness refers to the auxiliary loss $L ^ { \dagger }$ , while the lower bound on the Gram matrix is invoked only within the neighborhood specified in (67).

## C.2 Key Lemmas

Lemma 13 (Minibatch moment bounds). Let ${ \cal L } ^ { ( t ) } = { \cal L } ( \theta ^ { ( t ) } )$ and let $g _ { r } ^ { ( t ) }$ denote the block of the stochastic gradient $g ^ { ( t ) }$ corresponding to the parameters $\theta _ { r } = ( a _ { r } , w _ { r } )$ of neuron r. Define

$$
z _ { i } = ( f _ { \theta ^ { ( t ) } } ( x _ { i } ) - y _ { i } ) \nabla _ { \theta } f _ { \theta ^ { ( t ) } } ( x _ { i } ) .
$$

Then the stochastic gradient satisfies

$$
\mathbb { E } _ { t } [ g ^ { ( t ) } ] = G ^ { ( t ) } , \qquad \mathrm { C o v } _ { t } ( g ^ { ( t ) } ) = \frac { \chi _ { b } } { n } \sum _ { i = 1 } ^ { n } ( z _ { i } - G ^ { ( t ) } ) ( z _ { i } - G ^ { ( t ) } ) ^ { \top } .\tag{68}
$$

Moreover, thefollowing moment bounds holdfor all iterates:

$$
\mathbb { E } _ { t } \Vert g ^ { ( t ) } \Vert ^ { 2 } \leq C L ^ { ( t ) } , \qquad \operatorname* { m a x } _ { j } \mathbb { E } _ { t } | g _ { j } ^ { ( t ) } | ^ { 2 } \leq \frac { C L ^ { ( t ) } } { m } .\tag{69}
$$

If $K ( \theta ^ { ( t ) } ) \succeq \lambda _ { 0 } I / 4$ , we further have

$$
\begin{array} { r } { \mathbb { E } _ { t } \| g ^ { ( t ) } - G ^ { ( t ) } \| ^ { 2 } \leq C ( \rho _ { b } - 1 ) \| G ^ { ( t ) } \| ^ { 2 } , \qquad \mathbb { E } _ { t } \| g ^ { ( t ) } \| ^ { 2 } \leq C \rho _ { b } \| G ^ { ( t ) } \| ^ { 2 } . } \end{array}\tag{70}
$$

Next, define the sampled squared residual and its bias-corrected moving average by

$$
Y ^ { ( t ) } = \frac { 1 } { b } \sum _ { i \in { \cal B } ^ { ( t ) } } ( f _ { \theta ^ { ( t ) } } ( x _ { i } ) - y _ { i } ) ^ { 2 } , \qquad U ^ { ( t ) } = \sum _ { s = 0 } ^ { t } \frac { \nu \beta _ { 2 } ^ { t - s } } { 1 - \beta _ { 2 } ^ { t + 1 } } Y ^ { ( s ) } .\tag{71}
$$

Then the following bounds hold pathwise:

$$
\| g ^ { ( t ) } \| ^ { 2 } \leq C Y ^ { ( t ) } , \qquad \operatorname* { m a x } _ { r } \| g _ { r } ^ { ( t ) } \| ^ { 2 } \leq \frac { C Y ^ { ( t ) } } { m } , \qquad \operatorname* { m a x } _ { j } \widehat { v } _ { j } ^ { ( t ) } \leq \frac { C } { m } U ^ { ( t ) } .\tag{72}
$$

Furthermore, $Y ^ { ( t ) }$ satisfies

$$
\mathbb { E } _ { t } [ Y ^ { ( t ) } ] = 2 L ^ { ( t ) } , \qquad 0 \le Y ^ { ( t ) } \le 2 \kappa L ^ { ( t ) } , \qquad \mathrm { V a r } _ { t } ( Y ^ { ( t ) } ) \le 4 \kappa ( L ^ { ( t ) } ) ^ { 2 } .\tag{73}
$$

Proof. Let $I _ { i } = \mathbf { 1 } _ { \{ i \in B ^ { ( t ) } \} }$ denote the minibatch sampling indicators. Since $B ^ { ( t ) }$ is sampled uniformly without replacement,

$$
\mathbb { E } _ { t } [ I _ { i } ] = { \frac { b } { n } } , \qquad \mathbb { E } _ { t } [ I _ { i } I _ { j } ] = { \frac { b ( b - 1 ) } { n ( n - 1 ) } } , \quad i \neq j .
$$

Using $\begin{array} { r } { g ^ { ( t ) } = b ^ { - 1 } \sum _ { i = 1 } ^ { n } I _ { i } z _ { i } } \end{array}$ , we obtain the conditional unbiasedness and covariance identities in (68). The global derivative bounds in (66) imply

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| z _ { i } \| ^ { 2 } \leq C L ^ { ( t ) } ,
$$

together with the corresponding coordinatewise bound of order $L ^ { ( t ) } / m$ . The covariance identity therefore gives

$$
\begin{array} { r } { \mathbb { E } _ { t } \Vert g ^ { ( t ) } - G ^ { ( t ) } \Vert ^ { 2 } \leq C \chi _ { b } L ^ { ( t ) } . } \end{array}
$$

When $K ( \theta ^ { ( t ) } ) \succeq \lambda _ { 0 } I / 4$ , the gradient inequality (8) gives

$$
\| G ^ { ( t ) } \| ^ { 2 } \geq \frac { \lambda _ { 0 } } { 2 } L ^ { ( t ) } .
$$

Combining these estimates and using $\rho _ { b } = 1 + \chi _ { b } / \lambda _ { 0 }$ proves (70).

Next, Jensen’s inequality applied to the minibatch average, together with the global derivative bounds, yields the first two inequalities in (72). Taking conditional expectations gives (69). The bound on $\widehat { v } ^ { ( t ) }$ follows from the bias-corrected second-moment representation (50) and the coordinatewise gradient bounds.

Finally, we establish the scalar moment estimates. Let

$$
u _ { i } = ( f _ { \theta ^ { ( t ) } } ( x _ { i } ) - y _ { i } ) ^ { 2 } \geq 0 , \qquad \mu = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } u _ { i } = 2 L ^ { ( t ) } .
$$

By construction, $\mathbb { E } _ { t } [ Y ^ { ( t ) } ] = \mu$ and

$$
0 \leq Y ^ { ( t ) } \leq \frac { n } { b } \mu = \kappa \mu .
$$

Moreover, the covariance formula for uniform sampling without replacement gives

$$
\begin{array} { r l } & { \mathrm { V a r } _ { t } ( Y ^ { ( t ) } ) = \displaystyle \chi _ { b } \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } u _ { i } ^ { 2 } - \mu ^ { 2 } \right) } \\ & { \qquad \leq \displaystyle \chi _ { b } ( n - 1 ) \mu ^ { 2 } } \\ & { \qquad \leq \kappa \mu ^ { 2 } , } \end{array}
$$

where we used $\begin{array} { r } { \sum _ { i } u _ { i } ^ { 2 } \le ( \sum _ { i } u _ { i } ) ^ { 2 } } \end{array}$ . Substituting $\mu = 2 L ^ { ( t ) }$ completes the proof of (73). □

Lemma 14 (Uniform second-moment control). Let $0 < \nu \leq 1 / 2 , q = 1 - \Delta _ { q } \in ( 0 , 1 )$ , and $0 < \Delta _ { q } \leq \nu / 2$ . Suppose that $S > 0$ is deterministic and $I ^ { ( s ) } \in \{ 0 , 1 \}$ is $\mathcal { F } _ { s }$ -measurable, satisfying

$$
I ^ { ( s ) } L ( \theta ^ { ( s ) } ) \leq S q ^ { ( s - T _ { * } ) _ { + } } , \qquad s \geq 0 .
$$

Define the masked moving average

$$
\widetilde { U } ^ { ( t ) } = \sum _ { s = 0 } ^ { t } \frac { \nu \beta _ { 2 } ^ { t - s } } { 1 - \beta _ { 2 } ^ { t + 1 } } I ^ { ( s ) } Y ^ { ( s ) } .
$$

Then, for any $\delta _ { v } \in ( 0 , 1 / 2 )$ , there exist universal constants $C , c > 0$ such that

$$
\nu \kappa \log \frac { C } { \Delta _ { q } \delta _ { v } } \leq c \quad \Longrightarrow \quad \mathbb { P } \left( \operatorname* { s u p } _ { t \geq T _ { * } - 1 } \widetilde { U } ^ { ( t ) } > C S \right) \leq \delta _ { v } .\tag{74}
$$

Proof. Fix $t \geq T _ { * } - 1$ and let $j = ( t - T _ { * } ) _ { + }$ . Define the moving-average coefficients

$$
\alpha _ { t , s } = \frac { \nu \beta _ { 2 } ^ { t - s } } { 1 - \beta _ { 2 } ^ { t + 1 } } , \qquad 0 \leq s \leq t .
$$

$\mathtt { B y }$ the definition of $T _ { * }$ , we have $\alpha _ { t , s } \leq 2 \nu \beta _ { 2 } ^ { t - s }$ . Moreover, since $q = 1 - \Delta _ { q }$ and $\beta _ { 2 } = 1 - \nu _ { \mathrm { { ; } } }$

$$
1 - \frac { \beta _ { 2 } } { q } = \frac { \nu - \Delta _ { q } } { q } \geq \frac { \nu } { 2 } .
$$

Using these inequalities and summing the corresponding geometric series, we obtain

$$
\sum _ { s = 0 } ^ { t } \alpha _ { t , s } q ^ { ( s - T _ { * } ) _ { + } } \leq C q ^ { j } ,
$$

$$
\operatorname* { m a x } _ { 0 \leq s \leq t } \alpha _ { t , s } q ^ { ( s - T _ { * } ) _ { + } } \leq C \nu q ^ { j } ,
$$

$$
\sum _ { s = 0 } ^ { t } \alpha _ { t , s } ^ { 2 } q ^ { 2 ( s - T _ { * } ) _ { + } } \leq C \nu q ^ { 2 j } .
$$

Next, define the martingale differences

$$
X _ { s } = \alpha _ { t , s } I ^ { ( s ) } \big ( Y ^ { ( s ) } - 2 L ^ { ( s ) } \big ) , \qquad 0 \leq s \leq t .
$$

Since $I ^ { ( s ) }$ is $\mathcal { F } _ { s }$ -measurable and $\mathbb { E } _ { s } [ Y ^ { ( s ) } ] = 2 L ^ { ( s ) }$ , we have $\mathbb { E } _ { s } [ X _ { s } ] = 0$ . By (73), the assumed loss bound, and the weight estimates,

$$
| X _ { s } | \leq C \nu \kappa S q ^ { j } , \qquad \sum _ { s = 0 } ^ { t } \mathbb { E } _ { s } [ X _ { s } ^ { 2 } ] \leq C \nu \kappa S ^ { 2 } q ^ { 2 j } .
$$

Furthermore, the predictable component of the moving average satisfies

$$
\sum _ { s = 0 } ^ { t } \alpha _ { t , s } I ^ { ( s ) } 2 L ^ { ( s ) } \leq C S q ^ { j } .
$$

Applying the conditional Bernstein inequality in Lemma 5 with a sufficiently large constant multiple of S as the deviation threshold yields

$$
\mathbb { P } \left( \widetilde { U } ^ { ( t ) } > C S \right) \le \exp \left( - \frac { c q ^ { - j } } { \nu \kappa } \right) .
$$

To obtain a uniform bound over all iterations, let $A = c / ( \nu \kappa ) \geq 1$ . Since $q = 1 - \Delta _ { q } .$ , we have $q ^ { - j } \geq 1 + \Delta _ { q } j$ . A union bound therefore gives

$$
\begin{array} { r l } & { \mathbb { P } \left( \underset { t \geq T _ { * } - 1 } { \operatorname* { s u p } } \ \widetilde { U } ^ { ( t ) } > C S \right) \leq e ^ { - A } + \displaystyle \sum _ { j = 0 } ^ { \infty } e ^ { - A q ^ { - j } } } \\ & { \qquad \leq e ^ { - A } + \frac { e ^ { - A } } { 1 - e ^ { - A \Delta _ { q } } } } \\ & { \qquad \leq \frac { C e ^ { - A } } { \Delta _ { q } } , } \end{array}
$$

where the first term accounts for $t = T _ { * } - 1$ . Under condition (74), the last expression is at most $\delta _ { v } ,$ completing the proof. □

The following lemma provides a stochastic counterpart to Lemma 11. We use the normalized preconditioner $P ^ { ( t ) } = \epsilon D ^ { ( t ) }$ defined in (37). The bias-corrected moment representations (49) and (50) also hold for stochastic gradients. To establish conditional descent, we introduce a virtual iterate that separates the current stochastic gradient from the historical momentum contribution. Using the exact bias-correction formulas, we define, for $t \geq 1$

$$
\begin{array} { c } { { d ^ { ( t ) } = \displaystyle \frac { \beta _ { 1 } \left( 1 - \beta _ { 1 } ^ { t } \right) } { 1 - \beta _ { 1 } ^ { t + 1 } } , \qquad \omega ^ { ( t ) } = \displaystyle \frac { \nu } { 1 - \beta _ { 2 } ^ { t + 1 } } , \qquad k ^ { ( t ) } = \sum _ { s = t } ^ { \infty } \prod _ { j = t } ^ { s } d ^ { ( j ) } , \displaystyle } } \\ { { Z ^ { ( t ) } = \theta ^ { ( t ) } - \bar { \eta } k ^ { ( t ) } P ^ { ( t - 1 ) } \widehat { s } ^ { ( t - 1 ) } , } } \\ { { \zeta = C _ { 0 } \displaystyle \frac { \bar { \eta } \nu } { \gamma ^ { 2 } } , \qquad \mathcal { E } ^ { ( t ) } = L ( Z ^ { ( t ) } ) + \zeta \| \widehat { s } ^ { ( t - 1 ) } \| ^ { 2 } , } } \end{array}\tag{75}
$$

where $C _ { 0 } > 0$ is a sufficiently large universal constant.

Lemma 15 (Stochastic Lyapunov descent). Suppose that $t \geq T _ { * }$ and that, for a fixed universal constant $C _ { U } > 0$

$$
K ( { \theta } ^ { ( t ) } ) \succeq \frac { \lambda _ { 0 } } { 4 } I , \qquad L ( { \theta } ^ { ( t ) } ) \leq S , \qquad U ^ { ( t - 1 ) } \leq C _ { U } S .
$$

$I f \bar { \epsilon } \ge C \sqrt { S } , 0 < \nu \le c \gamma / \rho _ { b } ,$ and $0 < \bar { \eta } \le c \nu .$ , then

$$
\begin{array} { r l } & { { \mathcal { E } } ^ { ( t ) } - \mathbb { E } _ { t } [ { \mathcal { E } } ^ { ( t + 1 ) } ] \geq { \mathcal { D } } ^ { ( t ) } } \\ & { \qquad : = c \bar { \eta } \| G ^ { ( t ) } \| ^ { 2 } + c \gamma \zeta \| \widehat { s } ^ { ( t - 1 ) } \| ^ { 2 } } \\ & { \qquad \geq c \bar { \eta } \lambda _ { 0 } { \mathcal { E } } ^ { ( t ) } . } \end{array}\tag{76}
$$

Moreover, let $\Delta Z ^ { ( t ) } = Z ^ { ( t + 1 ) } - Z ^ { ( t ) }$ . Then

$$
\begin{array} { c } { \displaystyle | | \mathbb { E } _ { t } [ \Delta Z ^ { ( t ) } ] | | ^ { 2 } \leq C \bar { \eta } \mathcal { D } ^ { ( t ) } , } \\ { \displaystyle \mathbb { E } _ { t } \left[ \| \Delta Z ^ { ( t ) } - \mathbb { E } _ { t } [ \Delta Z ^ { ( t ) } ] \| ^ { 2 } \right] \leq \frac { C \bar { \eta } ^ { 2 } ( \rho _ { b } - 1 ) } { \gamma } \| G ^ { ( t ) } \| ^ { 2 } , } \end{array}\tag{77}
$$

where $\mathcal { D } ^ { ( t ) }$ is defined in (76). Finally, the following bounds hold globally for the auxiliary process, without assuming the local Gram matrix and loss bounds:

$$
L ( { \boldsymbol { \theta } } ^ { ( t ) } ) \leq 2 { \boldsymbol { \mathcal { E } } } ^ { ( t ) } , \qquad \| { \boldsymbol { \theta } } ^ { ( t ) } - { \boldsymbol { Z } } ^ { ( t ) } \| ^ { 2 } \leq C { \boldsymbol { \mathcal { E } } } ^ { ( t ) } .\tag{78}
$$

Proof. Fix $\mathcal { F } _ { t }$ and write $G \ : = \ : G ^ { ( t ) }$ and $h \ = \ { \widehat { s } } ^ { ( t - 1 ) }$ for brevity. Under the prescribed parameter conditions, we have $\beta _ { 2 } \geq \beta _ { 1 }$ and $\nu \leq 1 / 4$ . Moreover, for $t \geq T _ { * }$

$$
1 - d ^ { ( t ) } \leq 2 \gamma , \qquad \omega ^ { ( t ) } \leq 2 \nu , \qquad 1 - \omega ^ { ( t ) } \geq \frac { 1 } { 2 } .
$$

Since $\boldsymbol d ^ { ( t ) }$ is nondecreasing and bounded above by $\beta _ { 1 }$ , the definition of $k ^ { ( t ) }$ yields

$$
0 \leq k ^ { ( t ) } \leq \frac { \beta _ { 1 } } { \gamma } , \qquad k ^ { ( t ) } = d ^ { ( t ) } ( 1 + k ^ { ( t + 1 ) } ) .
$$

Consequently,

$$
1 \le c ^ { ( t ) } : = 1 + k ^ { ( t + 1 ) } - k ^ { ( t ) } = ( 1 + k ^ { ( t + 1 ) } ) ( 1 - d ^ { ( t ) } ) \le 2 .
$$

We first establish a bound relating the historical first- and second-moment estimates. Applying the weighted Cauchy–Schwarz inequality to their bias-corrected representations gives

$$
\begin{array} { l } { | h _ { j } | ^ { 2 } \leq \widehat { v } _ { j } ^ { ( t - 1 ) } \displaystyle \frac { \gamma ^ { 2 } ( 1 - \beta _ { 2 } ^ { t } ) } { \nu ( 1 - \beta _ { 1 } ^ { t } ) ^ { 2 } } \sum _ { s = 0 } ^ { t - 1 } \left( \frac { \beta _ { 1 } ^ { 2 } } { \beta _ { 2 } } \right) ^ { s } } \\ { \leq \displaystyle \frac { 4 \gamma } { \nu } \widehat { v } _ { j } ^ { ( t - 1 ) } . } \end{array}
$$

Here, we used $\beta _ { 1 } ^ { t } \le \beta _ { 2 } ^ { t } \le 1 / 2$ and $1 - \beta _ { 1 } ^ { 2 } / \beta _ { 2 } \ge \gamma$ . Furthermore, the convex combination

$$
\widehat { s } ^ { ( t ) } = d ^ { ( t ) } h + ( 1 - d ^ { ( t ) } ) g ^ { ( t ) }
$$

and the minibatch moment bounds in Lemma 13 imply

$$
\begin{array} { r } { \mathbb { E } _ { t } \| \widehat { s } ^ { ( t ) } \| ^ { 2 } \leq \beta _ { 1 } \| h \| ^ { 2 } + C \gamma \rho _ { b } \| G \| ^ { 2 } . } \end{array}\tag{79}
$$

Next, we analyze the virtual iterate update. For any gradient vector $g ,$ define

$$
P _ { t } ( g ) = \mathrm { d i a g } \left( \frac { \epsilon } { \epsilon + \sqrt { ( 1 - \omega ^ { ( t ) } ) \widehat { v } ^ { ( t - 1 ) } + \omega ^ { ( t ) } g ^ { \odot 2 } } } \right) , \qquad P ^ { - } = P _ { t } ( 0 ) .
$$

Substituting the bias-corrected moment recurrences into (75) yields the exact identity

$$
\begin{array} { l } { { \Delta Z ^ { ( t ) } = \bar { \eta } u _ { t } ( g ^ { ( t ) } ) , } } \\ { { \quad u _ { t } ( g ) = - c ^ { ( t ) } P _ { t } ( g ) g + k ^ { ( t ) } ( P ^ { ( t - 1 ) } - P _ { t } ( g ) ) h . } } \end{array}\tag{80}
$$

We decompose this update into a principal term involving the past-dependent preconditioner $P ^ { - }$ and a remainder:

$$
\begin{array} { l } { { u _ { t } ( g ) = - c ^ { ( t ) } P ^ { - } g + k ^ { ( t ) } ( P ^ { ( t - 1 ) } - P ^ { - } ) h + e _ { t } ( g ) , } } \\ { { e _ { t } ( g ) = ( P ^ { - } - P _ { t } ( g ) ) ( c ^ { ( t ) } g + k ^ { ( t ) } h ) . } } \end{array}\tag{81}
$$

We first bound the remainder $e _ { t } ( g )$ , which captures the dependence of the adaptive preconditioner on the current gradient. For each coordinate $j ,$ define

$$
A _ { j } = \sqrt { ( 1 - \omega ^ { ( t ) } ) \widehat { v } _ { j } ^ { ( t - 1 ) } } , \qquad B _ { j } = \sqrt { A _ { j } ^ { 2 } + \omega ^ { ( t ) } g _ { j } ^ { 2 } } .
$$

Then

$$
( P ^ { - } - P _ { t } ( g ) ) _ { j j } = \frac { \epsilon ( B _ { j } - A _ { j } ) } { ( \epsilon + A _ { j } ) ( \epsilon + B _ { j } ) } ,
$$

where

$$
B _ { j } - A _ { j } \le \sqrt { \omega ^ { ( t ) } } | g _ { j } | , \qquad A _ { j } ( B _ { j } - A _ { j } ) \le \omega ^ { ( t ) } g _ { j } ^ { 2 } .
$$

Combining these inequalities with $| h _ { j } | \le C \sqrt { \gamma / \nu } A _ { j }$ gives

$$
| e _ { t , j } ( g ) | \leq C \sqrt { \frac { \nu } { \gamma } } \frac { g _ { j } ^ { 2 } } { \epsilon } .
$$

$\mathtt { B y }$ Lemma 13, we therefore obtain

$$
\begin{array} { r l } {  { \| \mathbb { E } _ { t } \big [ e _ { t } ( g ^ { ( t ) } ) \big ] \| ^ { 2 } \leq \frac { C \nu } { \gamma \epsilon ^ { 2 } } ( \operatorname* { m a x } _ { j } \mathbb { E } _ { t } | g _ { j } ^ { ( t ) } | ^ { 2 } ) \mathbb { E } _ { t } \| g ^ { ( t ) } \| ^ { 2 } } } \\ & { \leq \frac { C \nu \rho _ { b } S } { \gamma \bar { \epsilon } ^ { 2 } } \| G \| ^ { 2 } . } \end{array}\tag{82}
$$

Thus, the dependence of the preconditioner on the current minibatch is captured by a controllable remainder.

We next bound the conditional variance of the virtual update. For each coordinate $j ,$ define

$$
p _ { j } ( z ) = \frac { \epsilon } { \epsilon + \sqrt { A _ { j } ^ { 2 } + \omega ^ { ( t ) } z ^ { 2 } } } .
$$

Direct differentiation yields

$$
| ( p _ { j } ( z ) z ) ^ { \prime } | \leq 1 , \qquad A _ { j } | p _ { j } ^ { \prime } ( z ) | \leq C \sqrt { \omega ^ { ( t ) } } .
$$

Together with the preceding historical moment bound, these inequalities imply that each coordinate of the mapping $u _ { t }$ is $C / \sqrt { \gamma } { - } \mathrm { I }$ ipschitz. When $A _ { j } = 0$ , the corresponding historical moment coordinate vanishes, and the same bound follows directly from

$$
p _ { j } ( z ) z = \frac { \epsilon z } { \epsilon + \sqrt { \omega ^ { ( t ) } } | z | } .
$$

Applying this Lipschitz estimate to two conditionally independent copies of $g ^ { ( t ) }$ , together with (70), yields

$$
\begin{array} { r l r } & { } & { \mathbb { E } _ { t } \left[ \| u _ { t } ( g ^ { ( t ) } ) - \mathbb { E } _ { t } [ u _ { t } ( g ^ { ( t ) } ) ] \| ^ { 2 } \right] \leq \displaystyle \frac { C } { \gamma } \mathbb { E } _ { t } \| g ^ { ( t ) } - G \| ^ { 2 } } \\ & { } & { \leq \displaystyle \frac { C ( \rho _ { b } - 1 ) } { \gamma } \| G \| ^ { 2 } . } \end{array}\tag{83}
$$

We now establish the conditional descent estimate. The historical second-moment bound and the damping condition ensure that

$$
p _ { * } I \preceq P ^ { - } \preceq I
$$

for some universal constant $p _ { * } > 0$ . Moreover,

$$
\| P ^ { ( t - 1 ) } - P ^ { - } \| \leq \omega ^ { ( t ) } .
$$

Indeed, letting $\ell _ { j } = \sqrt { \widehat { v } _ { j } ^ { ( t - 1 ) } } / \epsilon$ , we have

$$
\frac { ( P ^ { - } - P ^ { ( t - 1 ) } ) _ { j j } } { P _ { j j } ^ { - } } = \frac { \ell _ { j } - \sqrt { 1 - \omega ^ { ( t ) } } \ell _ { j } } { 1 + \ell _ { j } } \leq \omega ^ { ( t ) } .
$$

Taking conditional expectations in (81), we obtain

$$
\mathbb { E } _ { t } \big [ u _ { t } ( g ^ { ( t ) } ) \big ] = - c ^ { ( t ) } P ^ { - } G + \mathfrak { r } ^ { ( t ) } ,
$$

where

$$
\| { \boldsymbol { \mathfrak { r } } } ^ { ( t ) } \| \leq \alpha _ { \mathrm { e r r } } \| G \| + \frac { C \nu } { \gamma } \| h \| , \qquad \alpha _ { \mathrm { e r r } } ^ { 2 } \leq \frac { C \nu \rho _ { b } } { \gamma } .
$$

Under the prescribed parameter conditions, $\alpha _ { \mathrm { e r r } }$ is sufficiently small relative to $p _ { * }$

By global smoothness and the definition of the virtual iterate,

$$
\| \nabla L ( Z ^ { ( t ) } ) - G \| \leq \frac { C \bar { \eta } } { \gamma } \| h \| .
$$

Combining this estimate with the conditional mean and variance bounds for the virtual update, and using the smallness of $\alpha _ { \mathrm { e r r } }$ , we obtain

$$
\begin{array} { r l r } {  { \mathbb { E } _ { t } [ L ( Z ^ { ( t + 1 ) } ) ] - L ( Z ^ { ( t ) } ) } } \\ & { } & { \leq - c _ { 0 } \bar { \eta } \| G \| ^ { 2 } + \frac { C \bar { \eta } ^ { 2 } \rho _ { b } } { \gamma } \| G \| ^ { 2 } + \frac { C \bar { \eta } ( \nu + \bar { \eta } ) } { \gamma } \| G \| \| h \| + \frac { C \bar { \eta } ^ { 2 } ( \nu + \nu ^ { 2 } ) } { \gamma ^ { 2 } } \| h \| ^ { 2 } , } \end{array}
$$

where $c _ { 0 } > 0$ is a universal constant. By Young’s inequality and $\bar { \eta } \leq c \nu .$ , the mixed term can be bounded by

$$
\frac { C \bar { \eta } ( \nu + \bar { \eta } ) } { \gamma } \| G \| \| h \| \leq \frac { c _ { 0 } \bar { \eta } } { 2 } \| G \| ^ { 2 } + \frac { C \bar { \eta } \nu ^ { 2 } } { \gamma ^ { 2 } } \| h \| ^ { 2 } .
$$

The remaining historical-moment terms are also bounded by $C \bar { \eta } \nu ^ { 2 } \| h \| ^ { 2 } / \gamma ^ { 2 }$ . Consequently,

$$
\begin{array} { r l } & { \mathbb { E } _ { t } [ L ( Z ^ { ( t + 1 ) } ) ] - L ( Z ^ { ( t ) } ) } \\ & { \quad \le - c _ { 1 } \bar { \eta } \| G \| ^ { 2 } + \frac { C \bar { \eta } ^ { 2 } \rho _ { b } } { \gamma } \| G \| ^ { 2 } + \frac { C \bar { \eta } \nu ^ { 2 } } { \gamma ^ { 2 } } \| h \| ^ { 2 } , } \end{array}\tag{84}
$$

where $c _ { 1 } > 0$ is a universal constant.

We next establish the Lyapunov descent analysis. Recall that $\mathcal { E } ^ { ( t ) } = L ( Z ^ { ( t ) } ) + \zeta \| h \| ^ { 2 }$ , and $\begin{array} { r } { \zeta = C _ { 0 } \frac { \bar { \eta } \nu } { \gamma ^ { 2 } } } \end{array}$ . Combining (84) with ζ times (79), and using $\beta _ { 1 } = 1 - \gamma$ , yields

$$
\begin{array} { r l r } {  { \mathbb { E } _ { t } [ \mathcal { E } ^ { ( t + 1 ) } ] - \mathcal { E } ^ { ( t ) } } } \\ & { } & { \leq - \bar { \eta } ( c _ { 1 } - \frac { C \bar { \eta } \rho _ { b } } { \gamma } - \frac { C C _ { 0 } \nu \rho _ { b } } { \gamma } ) \| G \| ^ { 2 } - \gamma \zeta ( 1 - \frac { C \nu } { C _ { 0 } \gamma } ) \| h \| ^ { 2 } . } \end{array}
$$

Choose $C _ { 0 }$ sufficiently large and then take the universal constants in $\bar { \eta } \leq c \nu$ and $\nu \leq c \gamma / \rho _ { b }$ sufficiently small. Since $\rho _ { b } \geq 1$ , the two coefficients satisfy

$$
c _ { 1 } - \frac { C \bar { \eta } \rho _ { b } } { \gamma } - \frac { C C _ { 0 } \nu \rho _ { b } } { \gamma } \geq \frac { c _ { 1 } } { 2 } , \qquad 1 - \frac { C \nu } { C _ { 0 } \gamma } \geq \frac { 1 } { 2 } .
$$

Therefore,

$$
\begin{array} { l } { { \displaystyle { \mathcal E } ^ { ( t ) } - \mathbb E _ { t } [ { \mathcal E } ^ { ( t + 1 ) } ] \geq \frac { c _ { 1 } \bar { \eta } } { 2 } \| G \| ^ { 2 } + \frac { \gamma \zeta } { 2 } \| h \| ^ { 2 } } } \\ { { \displaystyle \geq \mathcal D ^ { ( t ) } } , } \end{array}
$$

after choosing the universal constant in $\mathcal { D } ^ { ( t ) }$ sufficiently small.

To establish the contraction estimate, global residual Lipschitz continuity and $\zeta \geq C \bar { \eta } ^ { 2 } ( k ^ { ( t ) } ) ^ { 2 }$ imply the comparisons in (78). Conversely, using the local Gram matrix lower bound, we obtain

$$
\begin{array} { l } { \displaystyle \mathcal { E } ^ { ( t ) } \leq 2 L ( \theta ^ { ( t ) } ) + 2 \zeta \| h \| ^ { 2 } } \\ { \displaystyle \leq \frac { 4 } { \lambda _ { 0 } } \| G \| ^ { 2 } + 2 \zeta \| h \| ^ { 2 } . } \end{array}
$$

Since $\bar { \eta } \lambda _ { 0 } \leq c \gamma$ , this estimate and the definition of $\mathcal { D } ^ { ( t ) }$ give

$$
{ \mathcal D } ^ { ( t ) } \geq c \bar { \eta } \lambda _ { 0 } { \mathcal E } ^ { ( t ) } ,
$$

completing the proof of (76).

Finally, we prove the virtual iterate movement bounds. From the conditional mean estimate, we have

$$
\begin{array} { r l } { \| \mathbb { E } _ { t } [ \Delta Z ^ { ( t ) } ] \| ^ { 2 } = \bar { \eta } ^ { 2 } \| \mathbb { E } _ { t } [ u _ { t } ( g ^ { ( t ) } ) ] \| ^ { 2 } } & { } \\ { \ } & { \leq C \bar { \eta } ^ { 2 } \left( \| G \| ^ { 2 } + \frac { \nu ^ { 2 } } { \gamma ^ { 2 } } \| h \| ^ { 2 } \right) } \\ { \ } & { \leq C \bar { \eta } \mathcal { D } ^ { ( t ) } . } \end{array}
$$

The conditional variance bound in (77) follows directly from (83) and $\Delta Z ^ { ( t ) } = \bar { \eta } u _ { t } ( g ^ { ( t ) } )$ . This completes the proof. □

## C.3 Completion of the Proof

We first show that the parameter conditions in Theorem 2 imply the concentration condition (74). Let

$$
q = 1 - c _ { E } \bar { \eta } \lambda _ { 0 } , \qquad \Delta _ { q } = 1 - q ,
$$

where $c _ { E } > 0$ is a sufficiently small universal constant, and choose $\delta _ { v }$ to be a fixed positive fraction of $\delta .$ Since $\kappa \leq \rho _ { b }$ and $\bar { \eta } \asymp \nu .$ , we have $\Delta _ { q } \asymp \nu \lambda _ { 0 }$ . Hence, it suffices to show that

$$
\nu \rho _ { b } \log { \frac { C } { \nu \lambda _ { 0 } \delta } }
$$

is sufficiently small.

Define

$$
\Lambda = \log \frac { C \rho _ { b } } { \gamma \lambda _ { 0 } \delta } \geq 1 .
$$

For a sufficiently small universal constant $c _ { \nu } > 0$ , the parameter condition (16) gives

$$
\nu \leq \nu _ { * } : = \frac { c _ { \nu } \gamma } { \rho _ { b } \Lambda } .
$$

Since the function $x \mapsto x \log ( C / ( x \lambda _ { 0 } \delta ) )$ is increasing on $( 0 , \nu _ { * } ]$ , we obtain

$$
\begin{array} { l } { \displaystyle \nu \rho _ { b } \log \frac { C } { \nu \lambda _ { 0 } \delta } \leq \frac { c _ { \nu } \gamma } { \Lambda } \log \frac { C \rho _ { b } \Lambda } { c _ { \nu } \gamma \lambda _ { 0 } \delta } } \\ { \leq { C c _ { \nu } } \big ( 1 + \log ( 1 / c _ { \nu } ) \big ) . } \end{array}
$$

Thus, choosing $c _ { \nu }$ sufficiently small ensures that (74) holds for all admissible ν, including arbitrarily small values.

The parameter conditions also imply

$$
\nu \leq \frac { c \gamma } { \rho _ { b } } , \qquad T _ { * } \asymp \nu ^ { - 1 } , \qquad \bar { \eta } T _ { * } \leq C , \qquad q ^ { - T _ { * } } \leq C , \qquad \Delta _ { q } \leq \frac { 1 } { 2 } \operatorname* { m i n } \{ \nu , \gamma \} .
$$

These bounds will be used in the stopping argument and the convergence analysis below.

Proof. Apply Lemma 8 with failure probability $\delta / 2$ . We condition on an initialization satisfying

$$
K ( { \theta } ^ { ( 0 ) } ) \succeq \frac { 3 \lambda _ { 0 } } { 4 } I , \qquad L _ { 0 } : = L ( { \theta } ^ { ( 0 ) } ) \leq S _ { 0 } : = C \log \frac { 2 } { \delta } .
$$

Throughout the following analysis, all probabilities and expectations are taken over minibatch sampling, conditional on this initialization. We work with the auxiliary model until the final step of the proof. If $L _ { 0 } = 0$ , all stochastic gradients vanish, and the conclusion follows immediately. Henceforth, assume that $L _ { 0 } > 0$

Define

$$
S = \frac { C _ { S } S _ { 0 } } { \delta } , \qquad \mathcal { R } = c _ { 0 } \sqrt { m \lambda _ { 0 } } , \qquad q = 1 - c _ { E } \bar { \eta } \lambda _ { 0 } ,
$$

where $C _ { S } > 0$ is a sufficiently large universal constant and $c _ { E } > 0$ is a sufficiently small universal constant, chosen consistently with the preceding lemmas. Under the assumptions of Theorem 2, we have

$$
\bar { \epsilon } \geq C \sqrt { S } , \qquad m \geq \frac { C S _ { 0 } } { \delta \lambda _ { 0 } ^ { 2 } } , \qquad \bar { \eta } T _ { * } \leq C , \qquad q ^ { - T _ { * } } \leq C , \qquad \Delta _ { q } : = 1 - q \leq \frac 1 2 \operatorname* { m i n } \{ \nu , \gamma \} .
$$

In particular, the parameter conditions of Lemmas 14 and 15 are satisfied.

Control of the initial iterations. By the bias-corrected first-moment representation (49) and the global bound $\mathbb { E } _ { t } \Vert g ^ { ( t ) } \Vert ^ { 2 } \leq C L ( \theta ^ { ( t ) } )$ , we obtain

$$
\mathbb { E } \Vert \widehat { s } ^ { ( t ) } \Vert ^ { 2 } \leq C M _ { t } , \qquad M _ { t } : = \operatorname* { m a x } _ { 0 \leq s \leq t } \mathbb { E } [ L ( \theta ^ { ( s ) } ) ] .
$$

Using the Lipschitz continuity of the residual vector and $P ^ { ( t ) } \preceq I$ , we obtain

$$
M _ { t + 1 } ^ { 1 / 2 } \leq ( 1 + C \bar { \eta } ) M _ { t } ^ { 1 / 2 } .
$$

Since $\bar { \eta } T _ { * } \leq C$ , it follows that $M _ { t } \le C L _ { 0 }$ for all $0 \leq t \leq T _ { * }$ . Moreover, Minkowski’s inequality gives

$$
\left( \mathbb { E } \left( \sum _ { s = 0 } ^ { T _ { * } - 1 } \lVert { \boldsymbol { \theta } } ^ { ( s + 1 ) } - { \boldsymbol { \theta } } ^ { ( s ) } \rVert \right) ^ { 2 } \right) ^ { 1 / 2 } \leq C \bar { \eta } T _ { * } \sqrt { L _ { 0 } } \leq C \sqrt { L _ { 0 } } .
$$

Using the Lipschitz continuity of the residual vector along the initial trajectory, together with $\bar { \eta } / \gamma \leq$ $C$ and $\zeta \leq C$ , we further obtain

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { 0 \leq s \leq T _ { * } } L ( \theta ^ { ( s ) } ) \right] + \mathbb { E } [ \mathcal { E } ^ { ( T _ { * } ) } ] + \mathbb { E } \left[ \left( \sum _ { s = 0 } ^ { T _ { * } - 1 } \| \theta ^ { ( s + 1 ) } - \theta ^ { ( s ) } \| \right) ^ { 2 } \right] \leq C L _ { 0 } .\tag{85}
$$

Define the $\mathcal { F } _ { T _ { * } }$ -measurable event

$$
\Pi = \left\{ \operatorname* { m a x } _ { 0 \leq s \leq T _ { * } } L ( \theta ^ { ( s ) } ) \leq S , \quad \sum _ { s = 0 } ^ { T _ { * } - 1 } \| \theta ^ { ( s + 1 ) } - \theta ^ { ( s ) } \| \leq \frac { \mathcal { R } } { 4 } \right\} .
$$

Applying Markov’s inequality to (85) yields

$$
\mathbb { P } ( \Pi ^ { c } ) \le \frac { C L _ { 0 } } { S } + \frac { C L _ { 0 } } { \mathcal { R } ^ { 2 } } .
$$

Stopping argument and energy control. On the event Π, define the stopping time $\tau$ as the first $t \geq T _ { * }$ at which any of the following conditions is violated:

$$
{ \mathcal E } ^ { ( t ) } \le S q ^ { t - T _ { * } } , \qquad U ^ { ( t - 1 ) } \le C _ { U } S , \qquad \mathrm { o r } \qquad \| \theta ^ { ( t ) } - \theta ^ { ( 0 ) } \| \le \frac { \mathcal R } { 2 } ,\tag{86}
$$

where $C _ { U } > 0$ is a sufficiently large universal constant. We set $\tau = T _ { * }$ on $\Pi ^ { c }$ and $\tau = \infty$ if none of these conditions is violated. Specifically, we let

$$
\tau = \operatorname* { i n f } \left\{ t \geq T _ { * } : { \mathcal E } ^ { ( t ) } > S q ^ { t - T _ { * } } \mathrm { o r } U ^ { ( t - 1 ) } > C _ { U } S \mathrm { o r } \| \theta ^ { ( t ) } - \theta ^ { ( 0 ) } \| > { \mathcal R } / 2 \right\} ,
$$

and inf $\mathcal { D } = \infty$ . Since all the stopping criteria are $\mathcal { F } _ { t }$ -measurable, $\tau$ is a stopping time. Define the predictable indicator

$$
I ^ { ( t ) } = \mathbf { 1 } _ { \Pi } \mathbf { 1 } _ { \left\{ t < \tau \right\} } , \qquad t \geq T _ { * } .
$$

On the event $\{ I ^ { ( t ) } = 1 \}$ , the global comparison in (78) gives

$$
L ( \theta ^ { ( t ) } ) \leq 2 \mathcal { E } ^ { ( t ) } \leq 2 S q ^ { t - T _ { * } } ,
$$

and the Gram matrix lower bound remains valid. Applying Lemma 15 with the loss bound 2S, we obtain

$$
\begin{array} { r } { \mathcal { E } ^ { ( t ) } - \mathbb { E } _ { t } [ \mathcal { E } ^ { ( t + 1 ) } ] \geq \mathcal { D } ^ { ( t ) } \geq ( 1 - q ) \mathcal { E } ^ { ( t ) } \qquad \mathrm { o n } \ \{ I ^ { ( t ) } = 1 \} . } \end{array}\tag{87}
$$

We first control the probability of violating the second-moment bound. Let $\delta _ { v } \ : = \ : c _ { v } \delta$ , where $c _ { v } \ > \ 0$ is a sufficiently small universal constant. Applying Lemma 14 with loss bound 2S and failure probability $\delta _ { v }$ to the predictable indicator

$$
\widetilde { I } ^ { ( s ) } = \left\{ \begin{array} { l l } { \mathbf { 1 } _ { \{ \mathrm { m a x } _ { 0 \leq j \leq s } L ( \theta ^ { ( j ) } ) \leq S \} } , } & { s < T _ { * } , } \\ { I ^ { ( s ) } , } & { s \geq T _ { * } . } \end{array} \right.
$$

By construction,

$$
\widetilde { I } ^ { ( s ) } L ( \theta ^ { ( s ) } ) \le 2 S q ^ { ( s - T _ { * } ) _ { + } } .
$$

On Π, all terms up to time $\tau - 1$ are retained in the masked moving average. Consequently, at any finite stopping time τ , the masked average coincides with $U ^ { ( \tau - 1 ) }$ , including the case $\tau = T _ { * }$ Choosing $C _ { U }$ sufficiently large to accommodate the constant in Lemma 14, we obtain

$$
\mathbb { P } \left( \Pi , \tau < \infty , U ^ { ( \tau - 1 ) } > C _ { U } S \right) \le \delta _ { v } .
$$

Next, we control the probability of violating the energy bound. By (87), the stopped process

$$
{ \mathbf { 1 } } _ { \Pi } q ^ { - ( ( t \wedge \tau ) - T _ { * } ) } \mathcal { E } ^ { ( t \wedge \tau ) }
$$

is a nonnegative supermartingale. Applying the maximal inequality in Lemma 6, together with the initial estimate (85), we have

$$
\mathbb { P } \left( \Pi , \tau < \infty , \mathcal { E } ^ { ( \tau ) } > S q ^ { \tau - T _ { * } } \right) \le \frac { C L _ { 0 } } { S } .
$$

Finally, we establish a contraction estimate for the expected energy before the stopping time. Define

$$
A _ { t } = \mathbb { E } [ I ^ { ( t ) } \mathcal { E } ^ { ( t ) } ] .
$$

Since $\mathcal { E } ^ { ( t + 1 ) } \geq 0$ and $I ^ { ( t + 1 ) } \leq I ^ { ( t ) }$ , the conditional descent estimate (87) implies

$$
\begin{array} { r } { \Delta _ { t } : = A _ { t } - A _ { t + 1 } \geq \mathbb E [ I ^ { ( t ) } { \mathcal D } ^ { ( t ) } ] } \\ { \geq ( 1 - q ) A _ { t } . } \end{array}\tag{88}
$$

Consequently,

$$
A _ { t } \leq q ^ { t - T _ { * } } A _ { T _ { * } } \leq C q ^ { t - T _ { * } } L _ { 0 } , \qquad t \geq T _ { * } ,
$$

where the last inequality follows from (85).

Control of parameter movement. We decompose the virtual iterate increment into its predictable and martingale components:

$$
\begin{array} { r } { Z ^ { ( t + 1 ) } - Z ^ { ( t ) } = W ^ { ( t ) } + \xi ^ { ( t + 1 ) } , } \end{array}
$$

where

$$
W ^ { ( t ) } = \bar { \eta } \mathbb { E } _ { t } [ u _ { t } ( g ^ { ( t ) } ) ] , \qquad \mathbb { E } _ { t } [ \xi ^ { ( t + 1 ) } ] = 0 .
$$

The conditional movement bound in Lemma 15 yields

$$
\mathbb { E } [ I ^ { ( t ) } \| W ^ { ( t ) } \| ^ { 2 } ] \le C \bar { \eta } \Delta _ { t } .
$$

The telescoping argument used in the deterministic proof also applies to the sequence $A _ { t }$ , giving

$$
\sqrt { \Delta _ { t } } \leq \frac { 2 } { \sqrt { 1 - q } } \left( \sqrt { A _ { t } } - \sqrt { A _ { t + 1 } } \right) .
$$

Consequently, Minkowski’s inequality and (88) imply

$$
\mathbb { E } \left[ \left( \sum _ { t \geq T _ { * } } I ^ { ( t ) } \| W ^ { ( t ) } \| \right) ^ { 2 } \right] \leq \frac { C \bar { \eta } L _ { 0 } } { 1 - q } \leq \frac { C L _ { 0 } } { \lambda _ { 0 } } .
$$

For the martingale increments, Lemma 15 further gives

$$
\begin{array} { r l } { \displaystyle \sum _ { t \geq T _ { * } } \mathbb { E } [ I ^ { ( t ) } \| \xi ^ { ( t + 1 ) } \| ^ { 2 } ] \leq \frac { C \bar { \eta } ^ { 2 } ( \rho _ { b } - 1 ) } { \gamma } \displaystyle \sum _ { t \geq T _ { * } } \mathbb { E } [ I ^ { ( t ) } \| G ^ { ( t ) } \| ^ { 2 } ] } & { } \\ { \leq \frac { C \bar { \eta } ( \rho _ { b } - 1 ) } { \gamma } L _ { 0 } } & { } \\ { \leq C L _ { 0 } . } \end{array}
$$

Applying the martingale maximal inequality in Lemma 6 to the stopped martingale, and combining the resulting bound with the initial movement estimate and the global comparison $\lVert \theta ^ { ( t ) } - Z ^ { ( t ) } \rVert ^ { 2 } \leq$ $C \mathcal { E } ^ { ( t ) }$ , we obtain

$$
\mathbb { E } \left[ \mathbf { 1 } _ { \Pi \operatorname* { s u p } _ { t \geq T _ { * } } } \left. Z ^ { ( t \wedge \tau ) } - \theta ^ { ( 0 ) } \right. ^ { 2 } \right] \leq \frac { C L _ { 0 } } { \lambda _ { 0 } } .\tag{89}
$$

The preceding estimates can first be established over finite horizons and then extended to the infinite horizon. In particular, the stopped sums include the update from $\tau - 1$ to τ .

We next bound the probability of leaving the region of Gram matrix stability. Consider a finite stopping time τ at which the parameter movement condition is violated while the energy condition remains satisfied. By the global comparison in (78) and the network width assumption,

$$
\| \theta ^ { ( \tau ) } - Z ^ { ( \tau ) } \| \leq C \sqrt { S } \leq \frac { \mathcal { R } } { 4 } .
$$

Therefore, such an exit necessarily implies

$$
\| Z ^ { ( \tau ) } - \theta ^ { ( 0 ) } \| > \frac { \mathcal { R } } { 4 } .
$$

Applying Markov’s inequality to (89), the probability of this event is bounded by $\frac { C L _ { 0 } } { \lambda _ { 0 } \mathcal { R } ^ { 2 } }$

Combining the preceding estimates for the initialization phase, energy, second-moment history, and parameter movement, we define the event

$$
\mathcal { G } _ { 0 } = \Pi \cap \{ \tau = \infty \} .
$$

Its failure probability satisfies

$$
\begin{array} { c } { { \displaystyle \mathbb { P } ( \mathcal { G } _ { 0 } ^ { c } ) \leq \frac { C L _ { 0 } } { S } + \frac { C L _ { 0 } } { \mathcal { R } ^ { 2 } } + \frac { C L _ { 0 } } { \lambda _ { 0 } \mathcal { R } ^ { 2 } } + \delta _ { v } } } \\ { { \displaystyle \leq \frac { \delta } { 4 } , } } \end{array}
$$

where the last inequality follows by choosing $C _ { S }$ and the universal constant in the network width requirement sufficiently large, with $\delta _ { v } = c _ { v } \delta$ for a sufficiently small constant $c _ { v } > 0$

On $\mathcal { G } _ { 0 }$ , the stopping time is infinite, so $I ^ { ( t ) } = 1$ for all $t \geq T _ { * }$ . Hence, the energy estimate (88), together with $L ( \theta ^ { ( t ) } ) \leq 2 \mathcal { E } ^ { ( t ) }$ , yields

$$
\mathbb { E } \left[ L ( \theta ^ { ( t ) } ) \mathbf { 1 } _ { \mathcal { G } _ { 0 } } \right] \leq C q ^ { t } L _ { 0 } , \qquad t \geq T _ { * } .\tag{90}
$$

For $t < T _ { * }$ , the same bound follows from (85) and $q ^ { - T _ { * } } \leq C$ . Thus, (90) holds for all $t \geq 0$

Return to the original network. It remains to control the movement of individual neurons, particularly their output coefficients, to establish agreement between the auxiliary and original Adam trajectories. Define the predictable indicator

$$
\widehat { I } ^ { ( s ) } = \left\{ \begin{array} { l l } { 1 , } & { s < T _ { * } , } \\ { I ^ { ( s ) } , } & { s \geq T _ { * } . } \end{array} \right.
$$

By the initial-phase estimate (85) and the stopped energy bound (88), we have

$$
\begin{array} { r } { \mathbb { E } \left[ \widehat { I } ^ { ( s ) } L ( \theta ^ { ( s ) } ) \right] \leq C L _ { 0 } q ^ { s } , \qquad s \geq 0 . } \end{array}
$$

Together with $\mathbb { E } _ { s } [ Y ^ { ( s ) } ] = 2 L ( \theta ^ { ( s ) } )$ , this implies

$$
\left( \mathbb { E } \left\| \widehat { I } ^ { ( s ) } \sqrt { Y ^ { ( s ) } } \right\| ^ { 2 } \right) ^ { 1 / 2 } \leq C \sqrt { L _ { 0 } } q ^ { s / 2 } .
$$

Define

$$
\mathcal { T } = \bar { \eta } \sum _ { t = 0 } ^ { \infty } \sum _ { s = 0 } ^ { t } \frac { \gamma \beta _ { 1 } ^ { t - s } } { 1 - \beta _ { 1 } ^ { t + 1 } } \widehat { I } ^ { ( s ) } \sqrt { Y ^ { ( s ) } } .
$$

Applying the geometric-weight estimate from the proof of Lemma 10, together with Minkowski’s inequality, yields

$$
\begin{array} { c } { { \displaystyle \left( \mathbb { E } \| \boldsymbol { \mathcal { T } } \| ^ { 2 } \right) ^ { 1 / 2 } \leq C \bar { \eta } \sqrt { L _ { 0 } } \sum _ { t = 0 } ^ { \infty } q ^ { t / 2 } } } \\ { { \leq \displaystyle \frac { C \sqrt { L _ { 0 } } } { \lambda _ { 0 } } . } } \end{array}
$$

On the event $\mathcal { G } _ { 0 }$ , all the indicators $\widehat { I } ^ { ( s ) }$ equal one. Therefore, the minibatch block-gradient bound in (72), the bias-corrected first-moment representation (49), and $P ^ { ( t ) } \preceq I$ imply

$$
\sum _ { t = 0 } ^ { \infty } \operatorname* { m a x } _ { 1 \leq r \leq m } \| \theta _ { r } ^ { ( t + 1 ) } - \theta _ { r } ^ { ( t ) } \| \leq \frac { C \mathcal { T } } { \sqrt { m } } \qquad \mathrm { o n ~ } \mathcal { G } _ { 0 } .
$$

For a sufficiently large universal constant $C _ { T } > 0$ , define

$$
\mathcal { G } = \mathcal { G } _ { 0 } \cap \left\{ \tau \leq C _ { T } \sqrt { \frac { L _ { 0 } } { \delta \lambda _ { 0 } ^ { 2 } } } \right\} .
$$

By Markov’s inequality and the preceding second-moment bound on $\tau ,$ we obtain

$$
\mathbb { P } ( \mathcal { G } ) \geq 1 - \frac { \delta } { 2 } .
$$

On ${ \mathcal { G } } ,$ , the cumulative neuron-wise movement satisfies

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 0 } ^ { \infty } \displaystyle \operatorname* { m a x } _ { 1 \leq r \leq m } \| \theta _ { r } ^ { ( t + 1 ) } - \theta _ { r } ^ { ( t ) } \| \leq C \sqrt { \frac { L _ { 0 } } { m \delta \lambda _ { 0 } ^ { 2 } } } } \\ & { \qquad \leq \displaystyle \frac { 1 } { 2 } , } \end{array}\tag{91}
$$

where the last inequality follows from the prescribed network width condition.

Restoring the auxiliary notation, we conclude that

$$
| a _ { \dagger , r } ^ { ( t ) } | \leq { \frac { 3 } { 2 } } , \qquad { \mathrm { f o r ~ a l l ~ } } r { \mathrm { ~ a n d ~ } } t \quad { \mathrm { o n ~ } } \mathcal { G } .
$$

Hence, the auxiliary iterates remain in $\boldsymbol { \mathcal { S } }$ throughout training. By the trajectory agreement established in Subsection C.1, the auxiliary and original Adam algorithms therefore generate identical iterates on $\mathcal { G } \mathrm { : }$

$$
\theta _ { \dagger } ^ { ( t ) } = \theta ^ { ( t ) } , \qquad L ^ { \dagger } ( \theta _ { \dagger } ^ { ( t ) } ) = L ( \theta ^ { ( t ) } ) , \qquad t \ge 0 .
$$

Since ${ \mathcal { G } } \subseteq { \mathcal { G } } _ { 0 }$ , the restricted expected-loss bound (90) for the auxiliary process transfers to the original Adam iterates:

$$
\begin{array} { r l } { \mathbb { E } \left[ L ( \boldsymbol { \theta } ^ { ( t ) } ) \mathbf { 1 } _ { \mathcal { G } } \right] = \mathbb { E } \left[ L ^ { \dagger } ( \boldsymbol { \theta } _ { \dagger } ^ { ( t ) } ) \mathbf { 1 } _ { \mathcal { G } } \right] } & { } \\ { \leq C \boldsymbol { q } ^ { t } L ( \boldsymbol { \theta } ^ { ( 0 ) } ) , \quad } & { t \geq 0 . } \end{array}
$$

Recalling that all probabilities and expectations are conditional on the fixed initialization, we obtain (18). Combining the initialization and sampling probabilities, the joint success probability is at least

$$
\left( 1 - \frac { \delta } { 2 } \right) ^ { 2 } \geq 1 - \delta .
$$

We then verify the parameter requirements. The initialization estimate requires

$$
m \gtrsim \lambda _ { 0 } ^ { - 1 } \log \frac { n } { \delta } ,
$$

up to universal constants, while the preceding trajectory control requires

$$
m \gtrsim \frac { S _ { 0 } } { \delta \lambda _ { 0 } ^ { 2 } } , \qquad \bar { \epsilon } \gtrsim \sqrt { \frac { S _ { 0 } } { \delta } } .
$$

Since $S _ { 0 } \asymp \log ( 2 / \delta )$ , these conditions are ensured by the width and damping requirements in Theorem 2.

Finally, (91) implies that, almost surely on $\mathcal { G }$

$$
\sum _ { t = 0 } ^ { \infty } \operatorname* { m a x } _ { 1 \leq r \leq m } \left( \lvert a _ { r } ^ { ( t + 1 ) } - a _ { r } ^ { ( t ) } \rvert + \lVert w _ { r } ^ { ( t + 1 ) } - w _ { r } ^ { ( t ) } \rVert \right) < \infty .
$$

Since $m$ is finite, the complete parameter sequence converges to a finite limit $\theta ^ { \star }$ on ${ \mathcal { G } } .$ . Moreover, by Fatou $\because \mathbf { \sigma } _ { \mathbf { S } }$ lemma, continuity of $L ,$ and the restricted expected-loss bound, we have

$$
\mathbb { E } \left[ L ( \theta ^ { \star } ) \mathbf { 1 } _ { \mathcal { G } } \mid \theta ^ { ( 0 ) } \right] \le \operatorname* { l i m i n f } _ { t \to \infty } \mathbb { E } \left[ L ( \theta ^ { ( t ) } ) \mathbf { 1 } _ { \mathcal { G } } \mid \theta ^ { ( 0 ) } \right] = 0 .
$$

Thus, $L ( \theta ^ { \star } ) = 0$ almost surely on ${ \mathcal { G } } ,$ which completes the proof.

## D Proof of Theorem 3

We use the PINN residual formulation and the associated Gram matrix defined in Section 3. We first establish the differential feature representations, initialization bounds, and regularity properties required for the PINN analysis. We then extend the moment bounds and Lyapunov-based descent analysis from the deterministic regression analysis to the PINN setting. Finally, we use these bounds to verify the required local conditions and complete the convergence proof by induction.

## D.1 Differential Features and Positive Definiteness of the Limiting Gram Matrix

Let $Q = \mathrm { d i a g } ( I _ { d } , 0 ) \in \mathbb { R } ^ { ( d + 1 ) \times ( d + 1 ) }$ . For any $w \in \mathbb { R } ^ { d + 1 }$ and input $\boldsymbol { x } \in \mathbb { R } ^ { d }$ , define

$$
s = w ^ { \top } \widetilde { x } , \qquad h ( w ) = \frac { 1 } { 2 } w ^ { \top } Q w .
$$

The PDE feature and its first two derivatives with respect to w are given by

$$
\begin{array} { r l } & { \quad \phi _ { \mathrm { P } } ( w ; x ) = h ( w ) \sigma ^ { \prime \prime } ( s ) , } \\ & { \nabla _ { w } \phi _ { \mathrm { P } } ( w ; x ) = Q w \sigma ^ { \prime \prime } ( s ) + h ( w ) \sigma ^ { \prime \prime \prime } ( s ) \widetilde { x } , } \\ & { \nabla _ { w } ^ { 2 } \phi _ { \mathrm { P } } ( w ; x ) = Q \sigma ^ { \prime \prime } ( s ) + \big ( Q w \widetilde { x } ^ { \top } + \widetilde { x } w ^ { \top } Q \big ) \sigma ^ { \prime \prime \prime } ( s ) + h ( w ) \sigma ^ { \prime \prime \prime \prime } ( s ) \widetilde { x } \widetilde { x } ^ { \top } . } \end{array}\tag{92}
$$

For the boundary feature $\phi _ { \mathrm { B } } ( w ; z ) = \sigma ( w ^ { \top } \widetilde { z } )$ , we have

$$
\nabla _ { w } \phi _ { \mathrm { B } } ( w ; z ) = \sigma ^ { \prime } ( w ^ { \top } \widetilde { z } ) \widetilde { z } , \qquad \nabla _ { w } ^ { 2 } \phi _ { \mathrm { B } } ( w ; z ) = \sigma ^ { \prime \prime } ( w ^ { \top } \widetilde { z } ) \widetilde { z } \widetilde { z } ^ { \top } .
$$

Since both layers are trained, we also use the identities

$$
\nabla _ { ( a , w ) } ( a \phi ) = \left( { \begin{array} { c } { \phi } \\ { a \nabla _ { w } \phi } \end{array} } \right) , \qquad \nabla _ { ( a , w ) } ^ { 2 } ( a \phi ) = \left( { \begin{array} { c c } { 0 } & { ( \nabla _ { w } \phi ) ^ { \top } } \\ { \nabla _ { w } \phi } & { a \nabla _ { w } ^ { 2 } \phi } \end{array} } \right) .\tag{93}
$$

Since $\sigma =$ tanh has bounded derivatives up to order four, the PDE feature and its first two derivatives are bounded in norm by $C ( 1 + \| w \| ^ { 2 } )$ . The corresponding boundary feature bounds are of order one. Consequently, the first- and second-order derivatives of the single-neuron contributions satisfy the same bounds when $| a | \le 2$ . These constants do not depend on the input dimension $d ,$ since $\| Q \| = 1$ and all augmented inputs have norm at most one.

For each $w \in \mathbb { R } ^ { d + 1 }$ , define the stacked feature vector

$$
\begin{array} { r } { \Phi ( w ) = \left( \binom { \left( \phi _ { \mathrm { P } } ( w ; x _ { i } ) / \sqrt { n _ { \mathrm { P D E } } } \right) _ { i = 1 } ^ { n _ { \mathrm { P D E } } } } { \left( \sqrt { \alpha _ { \mathrm { B C } } / n _ { \mathrm { B C } } } \phi _ { \mathrm { B } } ( w ; z _ { j } ) \right) _ { j = 1 } ^ { n _ { \mathrm { B C } } } } \right) \in \mathbb { R } ^ { N } . } \end{array}\tag{94}
$$

Denote its Jacobian with respect to w by $\nabla _ { w } \Phi ( w ) \in \mathbb { R } ^ { N \times ( d + 1 ) }$ . Then the limiting Gram matrix admits the representation

$$
K _ { \infty } = \mathbb { E } _ { w } \left[ \Phi ( w ) \Phi ( w ) ^ { \top } + \nabla _ { w } \Phi ( w ) \nabla _ { w } \Phi ( w ) ^ { \top } \right] ,\tag{95}
$$

where $w \sim \mathcal { N } ( 0 , I _ { d + 1 } )$ , since $( a _ { r } ^ { ( 0 ) } ) ^ { 2 } = 1$

ProofofLemma 2. Since the remaining contribution to $K _ { \infty }$ is positive semidefinite, it suffices to establish the positive definiteness of $\mathbb { E } [ \Phi ( w ) \Phi ( w ) ^ { \top } ]$

Suppose that a coefficient vector $( c _ { 1 } , \ldots , c _ { n _ { \mathrm { P D E } } } , d _ { 1 } , \ldots , d _ { n _ { \mathrm { B C } } } ) \in \mathbb { R } ^ { N }$ belongs to the null space of $\mathbb { E } _ { w } [ \bar { \Phi } ( w ) \Phi ( w ) ^ { \top } ]$ . Then the corresponding linear combination of the features vanishes almost surely under the Gaussian distribution. Since the features are continuous in w and the Gaussian distribution has full support on $\mathbb { R } ^ { d + 1 }$ , this identity holds for every $w \in \mathbb { R } ^ { d + 1 }$ . Absorbing the positive row normalization factors into the coefficients, we obtain

$$
\sum _ { i = 1 } ^ { n _ { \mathrm { P D E } } } c _ { i } \frac { w ^ { \top } Q w } { 2 } \sigma ^ { \prime \prime } ( w ^ { \top } \widetilde { x } _ { i } ) + \sum _ { j = 1 } ^ { n _ { \mathrm { B C } } } d _ { j } \sigma ( w ^ { \top } \widetilde { z } _ { j } ) = 0 , \qquad \forall w \in \mathbb { R } ^ { d + 1 } ,
$$

where $Q = \mathrm { d i a g } ( I _ { d } , 0 )$

We first show that all PDE coefficients vanish. Since the interior collocation points are pairwise distinct, there exists a unit vector $e \in \mathbb { R } ^ { d }$ such that $e ^ { \top } x _ { 1 } , \ldots , e ^ { \top } x _ { n _ { \mathrm { P D E } } }$ are pairwise distinct. Such a vector can be chosen outside the finite union of proper hyperplanes

$$
\bigcup _ { i < j } \left\{ e \in \mathbb { R } ^ { d } : e ^ { \top } ( x _ { i } - x _ { j } ) = 0 \right\} .
$$

Fix any $k \in \{ 1 , \dots , n _ { \mathrm { P D E } } \}$ and, for $A > 0$ and $s \in \mathbb { R }$ , define

$$
w _ { A } ( s ) = \sqrt { 2 } \left( { \ 4 e ^ { \cdot } } _ { s - A e ^ { \top } x _ { k } } \right) \in \mathbb { R } ^ { d + 1 } .
$$

By construction,

$$
w _ { A } ( s ) ^ { \top } \widetilde { x } _ { i } = A e ^ { \top } ( x _ { i } - x _ { k } ) + s , \qquad \frac { w _ { A } ( s ) ^ { \top } Q w _ { A } ( s ) } 2 = A ^ { 2 } .
$$

Substituting $w = w _ { A } ( s )$ into the preceding identity and dividing by $A ^ { 2 }$ , we obtain

$$
\sum _ { i = 1 } ^ { n _ { \mathrm { P D E } } } c _ { i } \sigma ^ { \prime \prime } \big ( A e ^ { \top } ( x _ { i } - x _ { k } ) + s \big ) + \frac { 1 } { A ^ { 2 } } \sum _ { j = 1 } ^ { n _ { \mathrm { B C } } } d _ { j } \sigma \big ( A e ^ { \top } ( z _ { j } - x _ { k } ) + s \big ) = 0 .
$$

Since $\sigma =$ tanh is bounded and $\sigma ^ { \prime \prime } ( s )  0 \mathrm { a s } | s |  \infty$ , letting $A \to \infty$ yields

$$
c _ { k } \sigma ^ { \prime \prime } ( s ) = 0 , \qquad \forall s \in \mathbb { R } .
$$

As $\sigma ^ { \prime \prime }$ is not identically zero, we conclude that $c _ { k } = 0$ . Because k was arbitrary, all PDE coefficients vanish.

It remains to show that the boundary coefficients vanish. The preceding identity now reduces to

$$
\sum _ { j = 1 } ^ { n _ { \mathrm { B C } } } d _ { j } \sigma ( w ^ { \top } \widetilde { z } _ { j } ) = 0 , \qquad \forall w \in \mathbb { R } ^ { d + 1 } .
$$

Since the boundary collocation points are pairwise distinct, we may similarly choose a unit vector $e \in \mathbb { R } ^ { d }$ such that their projections onto e are pairwise distinct. Fix $k \in \{ 1 , \dotsc , n _ { \mathrm { B C } } \}$ and define

$$
w _ { A } ( s ) = \sqrt { 2 } \left( { \cal A } e \atop s - { \cal A } e ^ { \top } z _ { k } \right) .
$$

Substituting this choice into the boundary identity gives

$$
\sum _ { j = 1 } ^ { n _ { \mathrm { B C } } } d _ { j } \sigma \big ( A e ^ { \top } ( z _ { j } - z _ { k } ) + s \big ) = 0 .
$$

Differentiating with respect to s, we obtain

$$
\sum _ { j = 1 } ^ { n _ { \mathrm { B C } } } d _ { j } \sigma ^ { \prime } \big ( A e ^ { \top } ( z _ { j } - z _ { k } ) + s \big ) = 0 .
$$

For any fixed s, letting $A \to \infty$ and using $\sigma ^ { \prime } ( s )  0 \mathrm { a s } | s |  \infty$ yields

$$
d _ { k } \sigma ^ { \prime } ( s ) = 0 .
$$

Since $\sigma ^ { \prime } ( 0 ) = 1$ , we conclude that $d _ { k } = 0$ . As k was arbitrary, all boundary coefficients vanish. Therefore, the original coefficient vector is zero, proving that $\mathbb { E } [ \Phi ( w ) \Phi ( w ) ^ { \top } ]$ and hence $K _ { \infty }$ are positive definite. □

## D.2 Key Lemmas

Lemma 16 (Initialization). Let $S _ { 0 } = C d ^ { 2 } \log ( 2 / \delta )$ and $0 < \delta < 1$ , where $C > 0$ is a sufficiently large universal constant. Suppose that

$$
m \gtrsim \frac { \ell _ { m } } { \lambda _ { 0 } } \log \frac { N } { \delta } ,
$$

where $\ell _ { m }$ is defined in (27). Then, with probability at least $1 - \delta$ over the random initialization, we have

$$
\begin{array} { r l r } {  { K ( \theta ^ { ( 0 ) } ) \succeq \frac { 3 \lambda _ { 0 } } { 4 } I , } \quad } & { L ( \theta ^ { ( 0 ) } ) \leq S _ { 0 } , } \\ & { } & { \displaystyle \operatorname* { m a x } _ { 1 \leq r \leq m } \big ( 1 + \| w _ { r } ^ { ( 0 ) } \| ^ { 2 } \big ) ^ { 2 } \leq C \ell _ { m } , } \\ & { } & { \displaystyle \frac { 1 } { m } \sum _ { r = 1 } ^ { m } \big ( 1 + \| w _ { r } ^ { ( 0 ) } \| ^ { 2 } \big ) ^ { 2 } \leq C d ^ { 2 } . } \end{array}\tag{96}
$$

Proof. For $w \sim \mathcal { N } ( 0 , I _ { d + 1 } )$ , the Gaussian moment function and Markov’s inequality yield

$$
\mathbb { P } ( \| w \| > B ) \le 2 ^ { ( d + 1 ) / 2 } \exp \left( - \frac { B ^ { 2 } } { 4 } \right) .
$$

Choose

$$
B ^ { 2 } = C _ { B } \left( d + \log \frac { C _ { * } m d ^ { 2 } } { \delta \lambda _ { 0 } } \right) ,
$$

where $C _ { B }$ and $C _ { * }$ are sufficiently large universal constants. By the definition of $\ell _ { m } .$ , we have

$$
( 1 + B ^ { 2 } ) ^ { 2 } \leq C \ell _ { m } .
$$

Moreover, a union bound gives

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 1 \leq r \leq m } \| w _ { r } ^ { ( 0 ) } \| > B \right) \leq \frac { \delta } { 4 } .
$$

Since the output weights are initialized from the Rademacher distribution, the Gram matrix at initialization admits the decomposition

$$
K ( { \theta } ^ { ( 0 ) } ) = \frac { 1 } { m } \sum _ { r = 1 } ^ { m } Z ( w _ { r } ^ { ( 0 ) } ) ,
$$

where

$$
\ b { Z } ( \ b { w } ) = \ b { \Phi } ( \ b { w } ) \ b { \Phi } ( \ b { w } ) ^ { \top } + \nabla _ { \ b { w } } \ b { \Phi } ( \ b { w } ) \nabla _ { \ b { w } } \ b { \Phi } ( \ b { w } ) ^ { \top } .
$$

The differential feature bounds established above, together with the normalization of the residual rows, imply

$$
0 \preceq Z ( w ) , \qquad \| Z ( w ) \| \leq \operatorname { t r a c e } ( Z ( w ) ) \leq C ( 1 + \| w \| ^ { 2 } ) ^ { 2 } .
$$

Consequently, the Gaussian moment bounds yield

$$
\mathbb { E } \Vert Z ( w ) \Vert ^ { 2 } \leq C d ^ { 4 } .
$$

In particular, these estimates do not introduce an additional dependence on the number of collocation points N.

To apply the matrix Chernoff inequality, define the truncated matrix

$$
Z ^ { B } ( w ) = Z ( w ) \mathbf { 1 } _ { \{ \| w \| \leq B \} } .
$$

By Cauchy–Schwarz and the Gaussian tail bound,

$$
\begin{array} { r l } { \| K _ { \infty } - \mathbb { E } [ Z ^ { B } ( w ) ] \| \le \mathbb { E } \left[ \| Z ( w ) \| \mathbf { 1 } _ { \{ \| w \| > B \} } \right] } & { } \\ & { \le \left( \mathbb { E } \| Z ( w ) \| ^ { 2 } \right) ^ { 1 / 2 } \mathbb { P } ( \| w \| > B ) ^ { 1 / 2 } } \\ & { \le C d ^ { 2 } \mathbb { P } ( \| w \| > B ) ^ { 1 / 2 } } \\ & { \le \displaystyle \frac { \lambda _ { 0 } } { 1 6 } , } \end{array}
$$

where the last inequality follows by choosing the constants in B sufficiently large. Since $K _ { \infty } \succeq \lambda _ { 0 } I ,$ we obtain

$$
\mathbb { E } [ Z ^ { B } ( w ) ] \succeq \frac { 1 5 \lambda _ { 0 } } { 1 6 } I .
$$

Moreover, the truncation ensures that

$$
0 \preceq Z ^ { B } ( w ) \preceq C \ell _ { m } I .
$$

By applying Lemma 3 to the independent truncated matrices $Z ^ { B } ( w _ { r } ^ { ( 0 ) } )$ , with relative deviation $1 / 5 ,$ we have

$$
\begin{array} { c l l } { \displaystyle \mathbb { P } \left( \lambda _ { \mathrm { m i n } } \left( \frac { 1 } { m } \sum _ { r = 1 } ^ { m } Z ^ { B } ( w _ { r } ^ { ( 0 ) } ) \right) < \frac { 3 \lambda _ { 0 } } { 4 } \right) } \\ { \displaystyle \qquad \leq N \exp \left( - \frac { c m \lambda _ { 0 } } { \ell _ { m } } \right) \leq \frac { \delta } { 4 } , } \end{array}
$$

where the last inequality follows from the given network width condition.

Finally, since $Z ( w ) \succeq Z ^ { B } ( w )$ for every w, the same lower bound holds for the original Gram matrix $K ( \theta ^ { ( 0 ) } )$ . We emphasize that the truncated matrices remain independent, as the concentration argument does not condition on the event controlling the maximum of the initialized weights.

Next, we establish the bound on the averaged Gaussian weights. Define the truncated random variables

$$
T _ { r } = \left( 1 + \| w _ { r } ^ { ( 0 ) } \| ^ { 2 } \right) ^ { 2 } \mathbf { 1 } _ { \{ \| w _ { r } ^ { ( 0 ) } \| \leq B \} } , \qquad r = 1 , \dots , m .
$$

Gaussian moment bounds imply

$$
\mathbb { E } [ T _ { r } ] \leq C d ^ { 2 } , \qquad \mathrm { V a r } ( T _ { r } ) \leq C d ^ { 4 } ,
$$

while the truncation ensures that $0 \leq T _ { r } \leq C \ell _ { m }$ . Applying the Bernstein inequality in Lemma 5 to the independent centered variables $T _ { r } - \mathbb { E } [ T _ { r } ]$ , we obtain

$$
\mathbb { P } \left( \frac { 1 } { m } \sum _ { r = 1 } ^ { m } T _ { r } > C d ^ { 2 } \right) \leq \exp \left( - \frac { c m d ^ { 2 } } { \ell _ { m } } \right) \leq \frac { \delta } { 4 } .
$$

Here, we used $\ell _ { m } \geq d ^ { 2 }$ and the prescribed width condition, which implies m $\gtrsim \frac { \ell _ { m } } { d ^ { 2 } } \log \frac { C } { \delta }$ , since $\lambda _ { 0 } \leq \alpha _ { \mathrm { B C } } ,$ , the boundary penalty is fixed, and $N \geq 2$ . On the event max $\| w _ { r } ^ { ( 0 ) } \| \leq B$ , the truncation is inactive, so

$$
T _ { r } = \left( 1 + \| w _ { r } ^ { ( 0 ) } \| ^ { 2 } \right) ^ { 2 }
$$

for every r.

Finally, we bound the initial loss. Fix the hidden weights satisfying the averaged bound established above, and define the normalized prediction vector

$$
F ^ { ( 0 ) } = \frac { 1 } { \sqrt { m } } \sum _ { r = 1 } ^ { m } a _ { r } ^ { ( 0 ) } \Phi ( w _ { r } ^ { ( 0 ) } ) ,
$$

where $\Phi ( \cdot )$ is defined in (94). Since the output weights are independent Rademacher random variables, conditioning on the hidden weights gives

$$
\mathbb { E } _ { a } \| F ^ { ( 0 ) } \| ^ { 2 } = \frac { 1 } { m } \sum _ { r = 1 } ^ { m } \| \Phi ( w _ { r } ^ { ( 0 ) } ) \| ^ { 2 } \leq C d ^ { 2 } ,
$$

where $\mathbb { E } _ { a }$ denotes expectation over the output weights $( a _ { 1 } , a _ { 2 } , \cdots , a _ { m } )$

Changing a single output weight $a _ { r } ^ { ( 0 ) }$ changes $\| F ^ { ( 0 ) } \|$ by at most $\frac { 2 } { \sqrt { m } } \lVert \Phi ( w _ { r } ^ { ( 0 ) } ) \rVert$ . The sum of the squares of these bounded-difference constants is therefore at most $C d ^ { 2 }$ . Applying Lemma 4, together with $\mathbb { E } _ { a } \| F ^ { ( 0 ) } \| \leq C d .$ , we have

$$
\| F ^ { ( 0 ) } \| \le C d \sqrt { \log \frac { 2 } { \delta } }
$$

with conditional probability at least $1 - \delta / 4$

Let y<sub>PINN</sub> denote the normalized target vector corresponding to the PDE source term and boundary data, then

$$
\| y _ { \mathrm { P I N N } } \| ^ { 2 } = \frac { 1 } { n _ { \mathrm { P D E } } } \sum _ { i = 1 } ^ { n _ { \mathrm { P D E } } } f ( x _ { i } ) ^ { 2 } + \frac { \alpha _ { \mathrm { B C } } } { n _ { \mathrm { B C } } } \sum _ { j = 1 } ^ { n _ { \mathrm { B C } } } \varphi ( z _ { j } ) ^ { 2 } \leq 1 + \alpha _ { \mathrm { B C } } .
$$

Consequently,

$$
\begin{array} { r l r } {  { L ( \theta ^ { ( 0 ) } ) = \frac { 1 } { 2 } \| F ^ { ( 0 ) } - y _ { \mathrm { P I N N } } \| ^ { 2 } } } \\ & { } & { \quad \leq \displaystyle \frac { 1 } { 2 } ( \| F ^ { ( 0 ) } \| + \| y _ { \mathrm { P I N N } } \| ) ^ { 2 } } \\ & { } & { \quad \leq C d ^ { 2 } \log \displaystyle \frac { 2 } { \delta } = S _ { 0 } . } \end{array}
$$

Combining the probability bounds for the Gaussian initialization, the initial Gram matrix, and the initial loss by a union bound completes the proof. □

Lemma 17 (Regularity and Gram matrix stability). On the initialization event (96), define the convex set

$$
S _ { \mathrm { P } } = \left\{ \theta : \left| a _ { r } \right| \leq 2 , \quad \left\| w _ { r } - w _ { r } ^ { ( 0 ) } \right\| \leq 1 , \quad 1 \leq r \leq m \right\} .
$$

Then, for any θ, $\vartheta \in S _ { \mathrm { P } }$ , we have

$$
\begin{array} { r } { \| J ( \theta ) \| ^ { 2 } \leq C d ^ { 2 } , } \end{array}
$$

$$
\lVert J ( \theta ) - J ( \vartheta ) \rVert \leq C \sqrt { \frac { \ell _ { m } } { m } } \lVert \theta - \vartheta \rVert ,\tag{97}
$$

$$
\lVert \nabla _ { { \boldsymbol { \theta } } _ { r } } L ( { \boldsymbol { \theta } } ) \rVert \leq C _ { g } \sqrt { \frac { \ell _ { m } L ( { \boldsymbol { \theta } } ) } { m } } ,
$$

$$
\lVert \nabla ^ { 2 } L ( \theta ) \rVert \leq C d ^ { 2 } + C \sqrt { \frac { \ell _ { m } } { m } } \lVert R ( \theta ) \rVert ,\tag{98}
$$

where the block-gradient bound holdsfor all $1 \leq r \leq m$

Furthermore, let

$$
\mathcal { R } = c _ { 0 } \sqrt { \frac { m \lambda _ { 0 } } { \ell _ { m } } } ,
$$

where $c _ { 0 } > 0$ is a sufficiently small universal constant. Then, every $\theta \in S _ { \mathrm { P } }$ satisfying $\lVert { \boldsymbol { \theta } } - { \boldsymbol { \theta } } ^ { ( 0 ) } \rVert \leq \mathcal { R }$ obeys

$$
K ( \theta ) \succeq \frac { \lambda _ { 0 } } { 4 } I , \qquad \| \nabla L ( \theta ) \| ^ { 2 } \geq \frac { \lambda _ { 0 } } { 2 } L ( \theta ) .\tag{99}
$$

Proof. On $S _ { \mathrm { P } }$ , the feature derivatives associated with neuron $r$ are bounded by $C ( 1 + \| w _ { r } ^ { ( 0 ) } \| ^ { 2 } )$ Accounting for the network scaling $m ^ { - 1 / 2 }$ and the normalization of the residual rows, we obtain

$$
\begin{array} { r l r } {  { \| J ( \theta ) \| ^ { 2 } \leq \| J ( \theta ) \| _ { \mathrm { F } } ^ { 2 } } } \\ & { } & { \leq \frac { C } { m } \sum _ { r = 1 } ^ { m } ( 1 + \| w _ { r } ^ { ( 0 ) } \| ^ { 2 } ) ^ { 2 } \leq C d ^ { 2 } . } \end{array}
$$

Since $S _ { \mathrm { P } }$ is convex, integrating the neuron-wise Hessian bounds along the segment joining $\theta$ and $\vartheta$ and summing over the residual rows yields

$$
\begin{array} { r l r } {  { \| J ( \theta ) - J ( \vartheta ) \| ^ { 2 } \leq \frac { C } { m } \sum _ { r = 1 } ^ { m } ( 1 + \| w _ { r } ^ { ( 0 ) } \| ^ { 2 } ) ^ { 2 } \| \theta _ { r } - \vartheta _ { r } \| ^ { 2 } } } \\ & { } & { \leq \frac { C \ell _ { m } } { m } \| \theta - \vartheta \| ^ { 2 } . } \end{array}
$$

Let $J _ { r } ( \theta )$ denote the column block of $J ( \theta )$ corresponding to neuron r. Since $\nabla _ { \theta _ { r } } L ( \theta ) \ =$ $J _ { r } ( \theta ) ^ { \top } R ( \theta )$ and

$$
\lVert J _ { r } ( \theta ) \rVert \leq \frac { C ( 1 + \lVert w _ { r } ^ { ( 0 ) } \rVert ^ { 2 } ) } { \sqrt { m } } ,
$$

the neuron-wise gradient bound follows. Moreover,

$$
\nabla ^ { 2 } L ( \theta ) = J ( \theta ) ^ { \top } J ( \theta ) + \sum _ { i = 1 } ^ { N } R _ { i } ( \theta ) \nabla ^ { 2 } R _ { i } ( \theta ) .
$$

The second term is block diagonal with respect to the neuron parameters. By applying Cauchy– Schwarz within the two residual groups, each block has operator norm at most

$$
\frac { C ( 1 + \| w _ { r } ^ { ( 0 ) } \| ^ { 2 } ) } { \sqrt { m } } \| R ( \theta ) \| .
$$

Combining this estimate with (97) establishes the Hessian bound in (98).

Finally, for any unit vector $z \in \mathbb { R } ^ { N }$ , the initialization bound and the Jacobian Lipschitz estimate imply

$$
\begin{array} { r l } & { \| J ( \theta ) ^ { \top } z \| \ge \| J ( \theta ^ { ( 0 ) } ) ^ { \top } z \| - \| J ( \theta ) - J ( \theta ^ { ( 0 ) } ) \| } \\ & { \qquad \ge \sqrt { \frac { 3 \lambda _ { 0 } } { 4 } } - C \sqrt { \frac { \ell _ { m } } { m } } \| \theta - \theta ^ { ( 0 ) } \| } \\ & { \qquad \ge \frac { \sqrt { \lambda _ { 0 } } } { 2 } , } \end{array}
$$

provided that $c _ { 0 }$ is sufficiently small. This yields $K ( \theta ) \succeq \lambda _ { 0 } I / 4$ . The gradient inequality then follows from $\nabla L ( \theta ) = J ( \theta ) ^ { \top } R ( \theta )$ and $\| R ( \theta ) \| ^ { 2 } = 2 L ( \theta )$ □

Let $\gamma = 1 - \beta _ { 1 } , L _ { 0 } = L ( \theta ^ { ( 0 ) } )$ , and $P ^ { ( t ) } = \epsilon D ^ { ( t ) }$ , as in the deterministic regression analysis. The bias-corrected moment representations remain unchanged. The following lemma extends the moment bounds and Lyapunov descent estimates in Lemmas 10 and 11 to the PINN setting, using the regularity properties established in Lemma 17. The required smoothness and Gram matrix conditions will be verified in the final induction argument.

Lemma 18 (Moment bounds and Lyapunov descent for PINNs). Work on the initialization event (96). Let $H _ { \mathrm { P } } \geq \operatorname* { m a x } \{ 1 , \lambda _ { 0 } \}$ , and suppose that

$$
0 < \bar { \eta } \leq \frac { \gamma } { 8 H _ { \mathrm { P } } } , \qquad q = 1 - \frac { \bar { \eta } \lambda _ { 0 } } { 3 2 } .
$$

Then $q \in ( 0 , 1 )$ and $1 - q \leq \gamma / 2$

Suppose that $\theta ^ { ( 0 ) } , \textrm { , . . . , } \theta ^ { ( T ) } \in S _ { \mathrm { P } } . \ I f L ( \theta ^ { ( k ) } ) \leq S f o r a l l \ 0 \leq k \leq T$ , then

$$
P ^ { ( t ) } \succeq \left( 1 + \frac { C _ { g } \sqrt { \ell _ { m } S } } { \bar { \epsilon } } \right) ^ { - 1 } I , \qquad 0 \leq t \leq T .\tag{100}
$$

If, in addition,

$$
L ( \theta ^ { ( k ) } ) \leq q ^ { k } L _ { 0 } , \qquad 0 \leq k \leq T ,
$$

then

$$
\| \widehat { s } _ { r } ^ { ( t ) } \| \leq 2 C _ { g } \sqrt { \frac { \ell _ { m } L _ { 0 } } { m } } q ^ { t / 2 } , \qquad 0 \leq t \leq T ,\tag{101}
$$

$$
\sum _ { t = 0 } ^ { T } \operatorname* { m a x } _ { 1 \leq r \leq m } \| \theta _ { r } ^ { ( t + 1 ) } - \theta _ { r } ^ { ( t ) } \| \leq \frac { 1 2 8 C _ { g } } { \lambda _ { 0 } } \sqrt { \frac { \ell _ { m } L _ { 0 } } { m } } .\tag{102}
$$

In particular, the latter bound includes the updatefrom iteration T to $T + 1$

Next, define the tracking error and Lyapunovfunction by

$$
e ^ { ( t ) } = \widehat { s } ^ { ( t ) } - g ^ { ( t ) } , \qquad { \pmb { \mathcal { E } } } ^ { ( t ) } = L ( \theta ^ { ( t ) } ) + \frac { 4 \bar { \eta } } { \gamma } \| e ^ { ( t ) } \| ^ { 2 } , \qquad Q ^ { ( t ) } = ( g ^ { ( t ) } ) ^ { \top } P ^ { ( t ) } g ^ { ( t ) } .
$$

For any iteration t satisfying

$$
\operatorname* { s u p } _ { \theta \in [ \theta ^ { ( t ) } , \theta ^ { ( t + 1 ) } ] } \| \nabla ^ { 2 } L ( \theta ) \| \le H _ { \mathrm { P } } , \qquad K ( \theta ^ { ( t ) } ) \succeq \frac { \lambda _ { 0 } } { 4 } I , \qquad \frac { 1 } { 2 } I \preceq P ^ { ( t ) } \preceq I ,
$$

thefollowing estimates hold:

$$
\Delta ^ { ( t ) } : = \mathcal { E } ^ { ( t ) } - \mathcal { E } ^ { ( t + 1 ) } \geq \frac { \bar { \eta } } { 4 } Q ^ { ( t ) } + \bar { \eta } \Vert e ^ { ( t ) } \Vert ^ { 2 } ,\tag{103}
$$

$$
\begin{array} { r } { \mathcal { E } ^ { ( t + 1 ) } \leq q \mathcal { E } ^ { ( t ) } , \qquad \| \theta ^ { ( t + 1 ) } - \theta ^ { ( t ) } \| ^ { 2 } \leq 8 \bar { \eta } \Delta ^ { ( t ) } . } \end{array}\tag{104}
$$

Moreover, if these smoothness, Gram matrix, and preconditioner bounds hold for all $0 \leq t \leq T$ then

$$
\sum _ { t = 0 } ^ { T } \lVert { \boldsymbol { \theta } } ^ { ( t + 1 ) } - { \boldsymbol { \theta } } ^ { ( t ) } \rVert \leq 3 2 { \sqrt { \frac { L _ { 0 } } { \lambda _ { 0 } } } } .\tag{105}
$$

Proof. Since $\bar { \eta } \lambda _ { 0 } \leq \gamma / 8$ , the stated properties of $q$ follow immediately. By the neuron-wise gradient bound in Lemma 17 and the convex combination representation (50), we have

$$
{ \widehat { v } } _ { j } ^ { ( t ) } \leq { \frac { C _ { g } ^ { 2 } \ell _ { m } S } { m } } .
$$

Together with $\epsilon = \bar { \epsilon } / \sqrt { m }$ , this establishes (100) for any $\beta _ { 2 } \in [ 0 , 1 )$ . Applying the geometric-weight argument from Lemma 10, with $C _ { g }$ replaced by $C _ { g } \sqrt { \ell _ { m } } ,$ gives (101) and (102). Since this argument uses only the gradients through iteration T and $P ^ { ( t ) } \preceq I$ , it also controls the candidate update at iteration $T + 1$

For the Lyapunov estimates, the proof of Lemma 11 requires smoothness only along the update segment and the Gram matrix lower bound only at its starting point. Under these conditions, we obtain

$$
\begin{array} { l } { \displaystyle { L ( \theta ^ { ( t + 1 ) } ) - L ( \theta ^ { ( t ) } ) \le - \frac { \bar { \eta } } { 2 } Q ^ { ( t ) } + 2 \bar { \eta } \| e ^ { ( t ) } \| ^ { 2 } , } } \\ { \displaystyle { \| e ^ { ( t + 1 ) } \| ^ { 2 } \le \beta _ { 1 } \| e ^ { ( t ) } \| ^ { 2 } + \frac { 2 \beta _ { 1 } ^ { 2 } H _ { \mathrm { P } } ^ { 2 } \bar { \eta } ^ { 2 } } { \gamma } \big ( Q ^ { ( t ) } + \| e ^ { ( t ) } \| ^ { 2 } \big ) . } } \end{array}
$$

The second bound follows from the exact bias-corrected first-moment recurrence; when $\beta _ { 1 } = 0$ , the tracking error vanishes. Multiplying this inequality by $4 \bar { \eta } / \gamma$ and combining it with the loss descent estimate establishes (103), since $H _ { \mathrm { P } } \bar { \eta } / \gamma \leq 1 / 8$

Moreover, the Gram matrix and preconditioner lower bounds imply $Q ^ { ( t ) } \geq \lambda _ { 0 } L ( \theta ^ { ( t ) } ) / 4$ . Consequently,

$$
\begin{array} { l } { \displaystyle \Delta ^ { ( t ) } \geq \frac { \bar { \eta } \lambda _ { 0 } } { 1 6 } L ( \theta ^ { ( t ) } ) + \bar { \eta } \| e ^ { ( t ) } \| ^ { 2 } } \\ { \displaystyle \geq \frac { \bar { \eta } \lambda _ { 0 } } { 3 2 } \mathcal { E } ^ { ( t ) } , } \end{array}
$$

where the last inequality uses $\bar { \eta } \lambda _ { 0 } \leq \gamma / 8$ . Finally, the squared-step and telescoping arguments in Lemma 11 yield (104) and (105), respectively, using $\mathcal { E } ^ { ( 0 ) } = L _ { 0 }$ from exact bias correction. □

## D.3 Completion of the Proof

ProofofTheorem 3. We work on the initialization event established in Lemma 16. If $L _ { 0 } = 0 $ , all residuals and gradients vanish, and the Adam iterates remain at initialization. Thus, the conclusion holds trivially. Henceforth, assume that $L _ { 0 } > 0$

Let $H _ { \mathrm { P } } ~ = ~ C _ { H } d ^ { 2 }$ , where $C _ { H } \ \ge \ \operatorname* { m a x } \{ 1 , \alpha _ { \mathrm { B C } } \}$ is a universal constant. In particular, $H _ { \mathrm { P } } \geq$ max{1, λ<sub>0</sub>}. The assumptions of Theorem 3 ensure that

$$
\bar { \epsilon } \geq C _ { g } \sqrt { \ell _ { m } S _ { 0 } } , \qquad 0 < \bar { \eta } \leq \frac { \gamma } { 8 H _ { \mathrm { P } } } , \qquad m \geq C _ { \mathrm { w } } \frac { \ell _ { m } S _ { 0 } } { \lambda _ { 0 } ^ { 2 } } ,\tag{106}
$$

where $C _ { \mathrm { w } } > 0$ is a sufficiently large universal constant. Define

$$
q = 1 - \frac { \bar { \eta } \lambda _ { 0 } } { 3 2 } , \qquad \mathcal { R } = c _ { 0 } \sqrt { \frac { m \lambda _ { 0 } } { \ell _ { m } } } ,
$$

as in Lemmas 18 and 17, respectively.

We prove by induction that, for every integer $t \geq 0$

$$
\begin{array} { c } { { \displaystyle \mathcal { E } ^ { ( t ) } \leq q ^ { t } L _ { 0 } , \qquad \displaystyle \operatorname* { m a x } _ { 1 \leq r \leq m } \| \theta _ { r } ^ { ( t ) } - \theta _ { r } ^ { ( 0 ) } \| \leq \frac { 1 } { 2 } , } } \\ { { \| \theta ^ { ( t ) } - \theta ^ { ( 0 ) } \| \leq \displaystyle \frac { \mathcal { R } } { 2 } . } } \end{array}\tag{107}
$$

These assertions hold at $t = 0$ , since $\mathcal { E } ^ { ( 0 ) } = L _ { 0 }$ and the parameter differences vanish. Suppose that they hold for all $0 \leq t \leq T$ . Then the iterates ${ \theta ^ { ( 0 ) } , \dots , \bar { \theta ^ { ( T ) } } }$ belong to $S _ { \mathrm { P } }$ and satisfy

$$
L ( \theta ^ { ( k ) } ) \leq q ^ { k } L _ { 0 } , \qquad 0 \leq k \leq T .
$$

By Lemma 18 and the damping condition in (106), we obtain

$$
P ^ { ( T ) } \succeq \frac { 1 } { 2 } I .
$$

Moreover, the Gram matrix lower bound in Lemma 17 gives

$$
K ( { \boldsymbol { \theta } } ^ { ( T ) } ) \succeq \frac { \lambda _ { 0 } } { 4 } I .
$$

We first verify the regularity conditions along the candidate update segment before applying the Lyapunov descent estimates. By (102) and (106),

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { r } \| \theta _ { r } ^ { ( T + 1 ) } - \theta _ { r } ^ { ( 0 ) } \| \leq \sum _ { t = 0 } ^ { T } \operatorname* { m a x } _ { r } \| \theta _ { r } ^ { ( t + 1 ) } - \theta _ { r } ^ { ( t ) } \| } & { } \\ { \leq \frac { 1 2 8 C _ { g } } { \lambda _ { 0 } } \sqrt { \frac { \ell _ { m } L _ { 0 } } { m } } \leq \frac { 1 } { 2 } . } \end{array}
$$

Hence, $\theta ^ { ( T + 1 ) } \in S _ { \mathrm { P } }$ , and convexity ensures that the entire update segment lies in $S _ { \mathrm { P } }$

The Jacobian bound in (97), together with the induction hypothesis, gives $\| g ^ { ( k ) } \| \le C d \sqrt { L _ { 0 } }$ for all $k \leq T$ . Using the convex combination representation (49) and $P ^ { ( T ) } \preceq I$ , we obtain

$$
\begin{array} { r } { \| \theta ^ { ( T + 1 ) } - \theta ^ { ( T ) } \| \leq \bar { \eta } \| \widehat { s } ^ { ( T ) } \| \leq C \bar { \eta } d \sqrt { L _ { 0 } } . } \end{array}
$$

Since $\| J ( \theta ) \| \leq C d$ throughout the segment, the fundamental theorem of calculus gives

$$
\begin{array} { r l r } & { } & { \underset { \theta \in [ \theta ^ { ( T ) } , \theta ^ { ( T + 1 ) } ] } { \operatorname* { s u p } } \left. R ( \theta ) \right. \leq \sqrt { 2 L _ { 0 } } + C d \Vert \theta ^ { ( T + 1 ) } - \theta ^ { ( T ) } \Vert } \\ & { } & { \leq C ( 1 + \bar { \eta } d ^ { 2 } ) \sqrt { L _ { 0 } } \leq C \sqrt { L _ { 0 } } , } \end{array}
$$

where the last inequality follows from $\bar { \eta } \leq \gamma / ( 8 H _ { \mathrm { P } } )$ and $H _ { \mathrm { P } } = C _ { H } d ^ { 2 }$

Consequently, the Hessian bound in (98) implies

$$
\begin{array} { r l r } {  { \operatorname* { s u p } _ { \theta \in [ \theta ^ { ( T ) } , \theta ^ { ( T + 1 ) } ] } \| \nabla ^ { 2 } L ( \theta ) \| } } \\ & { } & { \leq C d ^ { 2 } + C \sqrt { \frac { \ell _ { m } L _ { 0 } } { m } } } \\ & { } & { \leq C d ^ { 2 } + C \lambda _ { 0 } \leq H _ { \mathrm { P } } , } \end{array}
$$

where we used $L _ { 0 } \leq S _ { 0 }$ , the width condition in (106), and $\lambda _ { 0 } \leq \alpha _ { \mathrm { B C } }$ . Choosing $C _ { H }$ sufficiently large therefore verifies the segment smoothness condition in Lemma 18.

We can now apply Lemma 18 at iteration $T$ to obtain

$$
\mathcal { E } ^ { ( T + 1 ) } \leq q ^ { T + 1 } L _ { 0 } .
$$

Moreover, the cumulative parameter movement bound gives

$$
\begin{array} { r l r } {  { \| \theta ^ { ( T + 1 ) } - \theta ^ { ( 0 ) } \| \leq 3 2 \sqrt { \frac { L _ { 0 } } { \lambda _ { 0 } } } } } \\ & { } & { \leq \displaystyle \frac { c _ { 0 } } { 2 } \sqrt { \frac { m \lambda _ { 0 } } { \ell _ { m } } } = \frac { \mathcal { R } } { 2 } , } \end{array}
$$

where the second inequality follows from (106) by choosing $C _ { \mathrm { w } }$ sufficiently large. This completes the induction.

Since $L ( \theta ^ { ( t ) } ) \leq \mathcal { E } ^ { ( t ) }$ , we conclude that

$$
L ( \theta ^ { ( t ) } ) \leq \left( 1 - \frac { \bar { \eta } \lambda _ { 0 } } { 3 2 } \right) ^ { t } L _ { 0 } ,
$$

establishing the claimed linear convergence rate with $c = 1 / 3 2$ . The Gram matrix lower bound holds uniformly along the trajectory by Lemma 17.

Furthermore, letting $T \to \infty$ in (105) shows that the parameter trajectory has finite total length and hence converges to a limit $\theta ^ { \star }$ . By continuity of L and the convergence $L \big ( { \boldsymbol { \theta } } ^ { ( t ) } \big )  0$ , we have $L ( \theta ^ { \star } ) = 0$

Substituting $S _ { 0 } = C d ^ { 2 } \log ( 2 / \delta )$ into (106) recovers the width, damping, and stepsize conditions in Theorem 3, up to constants depending only on the fixed boundary penalty $\alpha _ { \mathrm { B C } }$ . The remaining width condition guarantees the initialization event in Lemma 16, which holds with probability at least $1 - \delta$

Finally, the finite-length bound (105) on the trajectory implies that $\{ \theta ^ { ( t ) } \}$ converges to a finite limit $\theta ^ { \star }$ . By the continuity of L and (26), we have $L ( \theta ^ { \star } ) = 0$ □

## E Proof of Theorem 4

We consider the empirical PINN loss and the independent PDE and boundary minibatches defined in the main text, under the fixed-penalty convention of Section 3. As in the stochastic regression analysis, let

$$
\gamma = 1 - \beta _ { 1 } , \qquad \nu = 1 - \beta _ { 2 } , \qquad T _ { * } = 1 + \left\lceil \frac { \log 2 } { - \log \beta _ { 2 } } \right\rceil .
$$

The proof adapts the stochastic regression argument by establishing global regularity for an auxiliary PINN, deriving the required minibatch moment bounds and conditional Lyapunov descent, and applying a stopping-time argument to show that the auxiliary and original Adam trajectories coincide.

Recall that $\ell _ { m }$ is defined in (27). Since $d \geq 1$ and $C _ { * }$ depends only on the fixed boundary penalty α<sub>BC</sub>,

$$
\ell _ { m } \asymp d ^ { 2 } + \log ^ { 2 } \frac { m d ^ { 2 } } { \delta \lambda _ { 0 } } .
$$

For brevity, we use $\ell _ { m }$ throughout the proof, while Theorem 4 states the dependence on these parameters explicitly.

## E.1 An Auxiliary PINN with Global Residual Regularity

We work on the initialization event of Lemma 16, with failure probability $\delta / 2$ . Replacing δ by $\delta / 2$ affects only the constants in the initialization bounds. Throughout this appendix, constants may depend on the fixed boundary penalty α <sub>C</sub> and the cutoff functions, but not on other problem or optimizer parameters.

We use the output-weight cutoff $\psi$ introduced in Appendix C. Unlike in the regression setting, we also introduce a cutoff for the hidden weights, since the derivatives of the PDE features can grow quadratically with the hidden-weight norm. Let $\chi _ { \mathrm { c u t } } : [ 0 , \infty )  [ 0 , 1 ]$ be a fixed smooth function satisfying $\chi _ { \mathrm { c u t } } ( s ) = 1$ for $s \leq 1$ and $\chi _ { \mathrm { c u t } } ( s ) = 0$ for $s \geq 4$ . Define the auxiliary network by

$$
\begin{array} { c } { { w _ { r } ^ { \dagger } = w _ { r } ^ { ( 0 ) } + \chi _ { \mathrm { c u t } } ( \| w _ { r } - w _ { r } ^ { ( 0 ) } \| ^ { 2 } ) ( w _ { r } - w _ { r } ^ { ( 0 ) } ) , } } \\ { { u _ { \theta } ^ { \dagger } ( x ) = \displaystyle \frac { 1 } { \sqrt { m } } \sum _ { r = 1 } ^ { m } \psi ( a _ { r } ) \sigma ( ( w _ { r } ^ { \dagger } ) ^ { \top } \widetilde { x } ) . } } \end{array}\tag{108}
$$

The corresponding unnormalized residuals are

$$
\begin{array} { r l } & { r _ { i } ^ { \mathrm { P D E , \dag } } ( \theta ) = \Delta u _ { \theta } ^ { \dag } ( x _ { i } ) - f ( x _ { i } ) , } \\ & { r _ { j } ^ { \mathrm { B C , \dag } } ( \theta ) = u _ { \theta } ^ { \dag } ( z _ { j } ) - \varphi ( z _ { j } ) . } \end{array}\tag{109}
$$

The map defining $w _ { r } ^ { \dagger } .$ , together with its first two derivatives, is uniformly bounded independently of d. We define $R ^ { \dagger } , L ^ { \dagger } , J ^ { \dagger }$ , and $K ^ { \dagger }$ as in (23), using the same residual normalization and boundary penalty.

Lemma 19 (Properties of the auxiliary PINN). On the initialization event of Lemma $I 6 ,$ the following properties hold.

1. The auxiliary and original residual maps, together with their Jacobians, coincide on the set S<sub>P</sub> defined in Lemma 17. In particular,

$$
L ^ { \dagger } ( \theta ) = L ( \theta ) , \qquad K ^ { \dagger } ( \theta ) = K ( \theta ) , \qquad \theta \in { \cal S } _ { \mathrm { P } } .
$$

2. For any parameter vectors $\theta , \vartheta ,$ , the auxiliary residual map satisfies

$$
\begin{array} { l } { \displaystyle \| J ^ { \dagger } ( \theta ) \| \le C d , \qquad \| R ^ { \dagger } ( \theta ) - R ^ { \dagger } ( \vartheta ) \| \le C d \| \theta - \vartheta \| , } \\ { \displaystyle \| J ^ { \dagger } ( \theta ) - J ^ { \dagger } ( \vartheta ) \| \le C \sqrt { \displaystyle \frac { \ell _ { m } } { m } } \| \theta - \vartheta \| , } \\ { \displaystyle \| \nabla ^ { 2 } L ^ { \dagger } ( \theta ) \| \le C d ^ { 2 } + C \sqrt { \displaystyle \frac { \ell _ { m } } { m } } \| R ^ { \dagger } ( \theta ) \| . } \end{array}\tag{110}
$$

Moreover, the unnormalized PDE and boundary residuals, defined in (109), satisfy

$$
\begin{array} { r l } & { \displaystyle | | \nabla r _ { i } ^ { \mathrm { P D E , \dagger } } ( \theta ) | | ^ { 2 } \leq C d ^ { 2 } , \qquad | | \nabla r _ { j } ^ { \mathrm { B C , \dagger } } ( \theta ) | | ^ { 2 } \leq C , } \\ & { \displaystyle | | \nabla _ { \theta _ { r } } r _ { i } ^ { \mathrm { P D E , \dagger } } ( \theta ) | | \leq \frac { C \left( 1 + \| w _ { r } ^ { ( 0 ) } \| ^ { 2 } \right) } { \sqrt { m } } , \qquad | | \nabla _ { \theta _ { r } } r _ { j } ^ { \mathrm { B C , \dagger } } ( \theta ) | | \leq \frac { C } { \sqrt { m } } . } \end{array}\tag{111}
$$

3. Let

$$
\mathcal { R } = c _ { 0 } \sqrt { \frac { m \lambda _ { 0 } } { \ell _ { m } } } ,
$$

where $c _ { 0 } > 0$ is sufficiently small. Then every parameter vector θ satisfying $\lVert { \boldsymbol { \theta } } - { \boldsymbol { \theta } } ^ { ( 0 ) } \rVert \leq \mathcal { R }$ obeys

$$
K ^ { \dagger } ( \theta ) \succeq \frac { \lambda _ { 0 } } { 4 } I , \qquad \| \nabla L ^ { \dagger } ( \theta ) \| ^ { 2 } \geq \frac { \lambda _ { 0 } } { 2 } L ^ { \dagger } ( \theta ) .\tag{112}
$$

Proof. When $\lVert w _ { r } - w _ { r } ^ { ( 0 ) } \rVert \leq 1$ , the hidden-weight cutoff map coincides with the identity, as does its derivative. Likewise, $\psi ( a ) = a$ and $\psi ^ { \prime } ( a ) = 1 \mathrm { f o r } | a | \leq 2$ . Since both cutoffs are independent of the spatial variable, we have

$$
\Delta u _ { \theta } ^ { \dagger } ( x ) = \frac { 1 } { \sqrt { m } } \sum _ { r = 1 } ^ { m } \psi ( a _ { r } ) \phi _ { \mathrm { P } } ( w _ { r } ^ { \dagger } ; x ) .
$$

Differentiating with respect to the network parameters shows that the residual maps and their Jacobians on $S _ { \mathrm { P } }$ coincide.

For arbitrary parameters, $\| w _ { r } ^ { \dagger } - w _ { r } ^ { ( 0 ) } \| \le 2$ . By the feature derivative formulas in (92) and the chain rule, each PDE neuron contribution and its first two parameter derivatives are bounded by $C ( 1 + \| w _ { r } ^ { ( 0 ) } \| ^ { 2 } ) / \sqrt { m }$ . The corresponding boundary bounds are $C / \sqrt { m }$ . These estimates rely on dimension-independent bounds for the derivatives of the cutoff map. Indeed, for the map $z \mapsto$ $\chi _ { \mathrm { c u t } } ( \| z \| ^ { 2 } ) z$ , the Jacobian is

$$
\chi _ { \mathrm { c u t } } ( \| z \| ^ { 2 } ) I + 2 \chi _ { \mathrm { c u t } } ^ { \prime } ( \| z \| ^ { 2 } ) z z ^ { \top } ,
$$

whose nonconstant cutoff terms are supported on $\| z \| \leq 2$ . The second derivative is uniformly bounded for the same reason.

The block-wise arguments in the proof of Lemma 17 now apply globally. The initialization bounds

$$
\frac { 1 } { m } \sum _ { r = 1 } ^ { m } ( 1 + \| w _ { r } ^ { ( 0 ) } \| ^ { 2 } ) ^ { 2 } \leq C d ^ { 2 } , \qquad \operatorname* { m a x } _ { r } ( 1 + \| w _ { r } ^ { ( 0 ) } \| ^ { 2 } ) ^ { 2 } \leq C \ell _ { m }
$$

yield the Jacobian bound and its Lipschitz estimate, respectively. The residual Lipschitz bound follows by integration. Moreover,

$$
\nabla ^ { 2 } L ^ { \dagger } = ( J ^ { \dagger } ) ^ { \top } J ^ { \dagger } + \sum _ { i = 1 } ^ { N } R _ { i } ^ { \dagger } \nabla ^ { 2 } R _ { i } ^ { \dagger } .
$$

The second term is block diagonal across neurons, with each block bounded by $C \sqrt { \ell _ { m } / m } \| R ^ { \dagger } \|$ establishing the Hessian estimate. Summing the squared neuron-wise gradient bounds gives (111).

Finally, since $J ^ { \dag } ( \theta ^ { ( 0 ) } ) ~ = ~ J ( \theta ^ { ( 0 ) } )$ , the global Jacobian Lipschitz estimate allows us to apply the singular-value perturbation argument of Lemma 17 throughout the ball $\lVert { \boldsymbol { \theta } } - { \boldsymbol { \theta } } ^ { ( 0 ) } \rVert \leq \mathcal { R }$ . This proves (112). □

We apply Adam to the auxiliary loss $L ^ { \dagger }$ with the same initialization, optimizer parameters, and minibatch sequences as the original algorithm. The trajectory-agreement argument from Appendix C applies with $S _ { \mathrm { P } }$ (defined in Lemma 17) in place of ${ \mathcal { S } } .$ . As long as the auxiliary iterates remain in $S _ { \mathrm { P } }$ , the two algorithms have identical gradients, moment estimates, and parameter updates. For the remainder of the proof, we suppress the superscript † on all auxiliary quantities. In particular, $G ^ { ( t ) } = \nabla L ( \theta ^ { ( t ) } )$ denotes the full gradient of the auxiliary loss. We establish agreement with the original trajectory at the end of the proof.

## E.2 Key Lemmas

Define the finite-population correction factors

$$
\chi _ { \mathrm { P D E } } = \frac { n _ { \mathrm { P D E } } - b _ { \mathrm { P D E } } } { b _ { \mathrm { P D E } } ( n _ { \mathrm { P D E } } - 1 ) } , \qquad \chi _ { \mathrm { B C } } = \frac { n _ { \mathrm { B C } } - b _ { \mathrm { B C } } } { b _ { \mathrm { B C } } ( n _ { \mathrm { B C } } - 1 ) } ,
$$

with each factor defined to be zero when the corresponding population size is one. Recall that

$$
\rho _ { b } = 1 + \frac { 1 } { \lambda _ { 0 } } \operatorname* { m a x } \{ d ^ { 2 } \chi _ { \mathrm { P D E } } , \alpha _ { \mathrm { B C } } \chi _ { \mathrm { B C } } \} ,
$$

and define

$$
\kappa = \operatorname* { m a x } \left\{ \frac { n _ { \mathrm { P D E } } } { b _ { \mathrm { P D E } } } , \frac { n _ { \mathrm { B C } } } { b _ { \mathrm { B C } } } \right\} .
$$

As in the stochastic regression analysis, let $\mathcal { F } _ { t }$ denote the sigma-algebra generated by the initialization and all PDE and boundary minibatches sampled before iteration t, and write

$$
\mathbb { E } _ { t } [ \cdot ] = \mathbb { E } [ \cdot \mid \mathcal { F } _ { t } ] .
$$

The following lemma provides bounds valid for all auxiliary parameter vectors, with only the relative variance bound requiring the local Gram matrix lower bound.

Lemma 20 (Two-group minibatch moment bounds). Define the sampled weighted squared residual by

$$
Y ^ { ( t ) } = \frac { 1 } { b _ { \mathrm { P D E } } } \sum _ { i \in B _ { \mathrm { P D E } } ^ { ( t ) } } \left( r _ { i } ^ { \mathrm { P D E } } ( \theta ^ { ( t ) } ) \right) ^ { 2 } + \frac { \alpha _ { \mathrm { B C } } } { b _ { \mathrm { B C } } } \sum _ { j \in B _ { \mathrm { B C } } ^ { ( t ) } } \left( r _ { j } ^ { \mathrm { B C } } ( \theta ^ { ( t ) } ) \right) ^ { 2 } .\tag{113}
$$

Then $\kappa \leq C \rho _ { b }$ , and the stochastic gradient satisfies

$$
\begin{array} { r l r } & { } & { \mathbb { E } _ { t } [ g ^ { ( t ) } ] = G ^ { ( t ) } , \qquad \| g ^ { ( t ) } \| ^ { 2 } \leq C d ^ { 2 } Y ^ { ( t ) } , \qquad \| g _ { r } ^ { ( t ) } \| ^ { 2 } \leq \frac { C \ell _ { m } } { m } Y ^ { ( t ) } , } \\ & { } & { \mathbb { E } _ { t } \| g ^ { ( t ) } \| ^ { 2 } \leq C d ^ { 2 } L ( \theta ^ { ( t ) } ) , \qquad \mathbb { E } _ { t } \| g ^ { ( t ) } - G ^ { ( t ) } \| ^ { 2 } \leq C \lambda _ { 0 } ( \rho _ { b } - 1 ) L \big ( \theta ^ { ( t ) } \big ) . } \end{array}\tag{114}
$$

In particular, $i f K ( \theta ^ { ( t ) } ) \succeq \lambda _ { 0 } I / 4$ , then

$$
\begin{array} { r } { \mathbb { E } _ { t } \| g ^ { ( t ) } - G ^ { ( t ) } \| ^ { 2 } \leq C ( \rho _ { b } - 1 ) \| G ^ { ( t ) } \| ^ { 2 } . } \end{array}
$$

The sampled squared residual also satisfies

$$
\begin{array} { r } { \mathbb { E } _ { t } [ Y ^ { ( t ) } ] = 2 L ( \theta ^ { ( t ) } ) , \qquad 0 \le Y ^ { ( t ) } \le 2 \kappa L ( \theta ^ { ( t ) } ) , \qquad \mathrm { V a r } _ { t } ( Y ^ { ( t ) } ) \le 4 \kappa L ( \theta ^ { ( t ) } ) ^ { 2 } . } \end{array}\tag{115}
$$

Define its bias-corrected exponential moving average by

$$
U ^ { ( t ) } = \frac { 1 - \beta _ { 2 } } { 1 - \beta _ { 2 } ^ { t + 1 } } \sum _ { s = 0 } ^ { t } \beta _ { 2 } ^ { t - s } Y ^ { ( s ) } .
$$

Then the Adam second-moment estimates satisfy

$$
\widehat { v } _ { j } ^ { ( t ) } \leq \frac { C \ell _ { m } } { m } U ^ { ( t ) } f o r e \nu e r y j , \qquad \sum _ { j } \widehat { v } _ { j } ^ { ( t ) } \leq C d ^ { 2 } U ^ { ( t ) } .\tag{116}
$$

Proof. The uniform-subset variance formula in Lemma 13 applies separately to the PDE and boundary groups. At $\boldsymbol { \theta } ^ { ( t ) }$ , define

$$
G _ { i } ^ { \mathrm { P D E } } = r _ { i } ^ { \mathrm { P D E } } \nabla r _ { i } ^ { \mathrm { P D E } } , \qquad G _ { j } ^ { \mathrm { B C } } = r _ { j } ^ { \mathrm { B C } } \nabla r _ { j } ^ { \mathrm { B C } } ,
$$

and let $\overline { G } ^ { \mathrm { P D E } }$ and $\overline { G } ^ { \mathrm { B C } }$ denote their respective population means. By independence of the two minibatches,

$$
\begin{array} { r l r } & { } & { \mathbb { E } _ { t } \Vert g ^ { ( t ) } - G ^ { ( t ) } \Vert ^ { 2 } = \chi _ { \mathrm { P D E } } \left( \frac { 1 } { n _ { \mathrm { P D E } } } \sum _ { i } \Vert G _ { i } ^ { \mathrm { P D E } } \Vert ^ { 2 } - \Vert \overline { { G } } ^ { \mathrm { P D E } } \Vert ^ { 2 } \right) } \\ & { } & { + \alpha _ { \mathrm { B C } } ^ { 2 } \chi _ { \mathrm { B C } } \left( \frac { 1 } { n _ { \mathrm { B C } } } \sum _ { j } \Vert G _ { j } ^ { \mathrm { B C } } \Vert ^ { 2 } - \Vert \overline { { G } } ^ { \mathrm { B C } } \Vert ^ { 2 } \right) . } \end{array}
$$

The boundary contribution contains $\alpha _ { \mathrm { B C } } ^ { 2 }$ , one factor of which is absorbed into the weighted loss. Using the row-gradient bounds in (111), we obtain

$$
\begin{array} { r } { \mathbb { E } _ { t } \| g ^ { ( t ) } - G ^ { ( t ) } \| ^ { 2 } \leq C \lambda _ { 0 } ( \rho _ { b } - 1 ) L ( \theta ^ { ( t ) } ) . } \end{array}
$$

The relative variance bound then follows from the local Gram matrix lower bound (112).

Applying Cauchy–Schwarz within each sampled group yields the neuron-wise estimate

$$
\| g _ { r } ^ { ( t ) } \| ^ { 2 } \leq \frac { C } { m } \left( 1 + \| w _ { r } ^ { ( 0 ) } \| ^ { 2 } \right) ^ { 2 } Y ^ { ( t ) } .
$$

Taking the maximum over neurons or summing the squared bounds gives the pointwise gradient estimates in (114). Taking conditional expectations gives the corresponding second-moment bound. Finally, averaging the coordinate-wise and summed gradient bounds with the nonnegative biascorrection weights establishes (116).

For the scalar moment bounds, we apply the same finite-population variance formula to the nonnegative populations $( r _ { i } ^ { \mathrm { P D E } } ) ^ { 2 }$ and $( r _ { j } ^ { \mathrm { B C } } ) ^ { 2 }$ . For any nonnegative population of size n with mean $\mu ,$ its population variance is bounded by $( n - 1 ) \mu ^ { 2 }$ . Since $( n - 1 ) \chi = n / b - 1$ , the variance of its uniformly sampled mean is at most $( n / b - 1 ) \mu ^ { 2 }$ . Applying this bound separately to the PDE and boundary groups, accounting for the boundary penalty and using independence, establishes the variance bound in (115). The expectation identity follows from unbiased sampling, while the almostsure bound follows from nonnegativity.

Uniform control of the scalar history. The proof of Lemma 14 uses only the three scalar moment bounds in (73). Since their two-group counterparts in (115) have the same form, the lemma applies directly to the present $Y ^ { ( t ) } , U ^ { ( t ) }$ , and $\kappa ,$ , with the same predictable masking argument under the condition

$$
\nu \kappa \log { \frac { C } { \Delta _ { q } \delta _ { v } } } \leq c .
$$

No union bound over parameter coordinates is needed, since (116) controls all coordinatewise second moments and their sum through the single scalar process $U ^ { ( t ) }$

Conditional Lyapunov descent. We continue to work with the auxiliary process, with $G ^ { ( t ) } =$ $\nabla L ( \theta ^ { ( t ) } )$ and $P ^ { ( t ) } = \epsilon D ^ { ( t ) }$ . We use the quantities $\boldsymbol { d } ^ { ( t ) } , \boldsymbol { k } ^ { ( t ) }$ , and $Z ^ { ( t ) }$ from (75), now defined using the PINN iterates. In particular,

$$
\begin{array} { r } { \qquad Z ^ { ( t ) } = \theta ^ { ( t ) } - \bar { \eta } k ^ { ( t ) } P ^ { ( t - 1 ) } \widehat { s } ^ { ( t - 1 ) } , \quad \quad \quad } \\ { \zeta = C _ { 0 } \displaystyle \frac { \bar { \eta } \nu } { \gamma ^ { 2 } } , \qquad \displaystyle \mathcal { E } ^ { ( t ) } = L ( Z ^ { ( t ) } ) + \zeta \| \widehat { s } ^ { ( t - 1 ) } \| ^ { 2 } . } \end{array}
$$

Here $C _ { 0 } ~ > ~ 0$ is a sufficiently large constant depending only on the fixed boundary penalty and cutoff functions. We also set $H _ { \mathrm { P } } = C _ { H } d ^ { 2 }$ , where $C _ { H } > 0$ is chosen sufficiently large to ensure that $H _ { \mathrm { P } } \geq \operatorname* { m a x } \{ 1 , \lambda _ { 0 } \}$ . The scalar history $U ^ { ( t ) }$ is defined in Lemma 20.

Lemma 21 (Stochastic Lyapunov descent for PINNs). Suppose that $t \geq T _ { * }$ and

$$
K ( { \theta } ^ { ( t ) } ) \succeq \frac { \lambda _ { 0 } } { 4 } I , \qquad L ( { \theta } ^ { ( t ) } ) \leq S , \qquad U ^ { ( t - 1 ) } \leq C _ { U } S .
$$

Assume further that

$$
m \ge \frac { C \ell _ { m } S } { \lambda _ { 0 } ^ { 2 } } , \qquad \bar { \epsilon } \ge C \sqrt { \ell _ { m } S } , \qquad 0 < \nu \le \frac { c \gamma } { \rho _ { b } } , \qquad 0 < \bar { \eta } H _ { \mathrm { P } } \le c \nu .
$$

Then the Lyapunov function satisfies

$$
\begin{array} { r l } & { { \mathcal E } ^ { ( t ) } - \mathbb { E } _ { t } [ { \mathcal E } ^ { ( t + 1 ) } ] \geq { \mathcal D } ^ { ( t ) } } \\ & { \qquad : = c \bar { \eta } \| G ^ { ( t ) } \| ^ { 2 } + c \gamma \zeta \| \widehat { s } ^ { ( t - 1 ) } \| ^ { 2 } } \\ & { \qquad \geq c \bar { \eta } \lambda _ { 0 } { \mathcal E } ^ { ( t ) } . } \end{array}\tag{117}
$$

Moreover, let $\Delta Z ^ { ( t ) } = Z ^ { ( t + 1 ) } - Z ^ { ( t ) }$ , then

$$
\begin{array} { c } { \displaystyle \| \mathbb { E } _ { t } [ \Delta Z ^ { ( t ) } ] \| ^ { 2 } \leq C \bar { \eta } \mathcal { D } ^ { ( t ) } , } \\ { \displaystyle \mathbb { E } _ { t } \| \Delta Z ^ { ( t ) } - \mathbb { E } _ { t } [ \Delta Z ^ { ( t ) } ] \| ^ { 2 } \leq \frac { C \bar { \eta } ^ { 2 } ( \rho _ { b } - 1 ) } { \gamma } \| G ^ { ( t ) } \| ^ { 2 } . } \end{array}\tag{118}
$$

Finally, the following comparison bounds hold globally for the auxiliary process

$$
L ( \theta ^ { ( t ) } ) \leq 2 \mathcal { E } ^ { ( t ) } , \qquad \| \theta ^ { ( t ) } - Z ^ { ( t ) } \| ^ { 2 } \leq \frac { C } { H _ { \mathrm { P } } } \mathcal { E } ^ { ( t ) } .\tag{119}
$$

These bounds do not require the local Gram matrix or loss bounds.

Proof. We adapt the proof of Lemma 15 by establishing the required smoothness bounds at the scale $H _ { \mathrm { P } }$ . This step is necessary because the auxiliary PINN loss has a residual-dependent Hessian bound rather than a uniform bound of order $H _ { \mathrm { P } }$

Condition on $\mathcal { F } _ { t }$ and write $h = \widehat { s } ^ { ( t - 1 ) }$ . The historical momentum bound used in the proof of Lemma 15, together with the trace bound in (116), yields

$$
\Vert h \Vert ^ { 2 } \leq \frac { C \gamma } { \nu } \sum _ { j } \widehat { v } _ { j } ^ { ( t - 1 ) } \leq \frac { C \gamma } { \nu } H _ { \mathrm { P } } S .
$$

Since $k ^ { ( t ) } \leq \beta _ { 1 } / \gamma$ , it follows that

$$
\sqrt { H _ { \mathrm { P } } } \| Z ^ { ( t ) } - \theta ^ { ( t ) } \| \le C \frac { \bar { \eta } H _ { \mathrm { P } } } { \sqrt { \gamma \nu } } \sqrt { S } \le C \sqrt { S } .
$$

For any realization of the current minibatches, the exact virtual update (80) satisfies

$$
\| \Delta Z ^ { ( t ) } \| \le \bar { \eta } \left( 2 \| g ^ { ( t ) } \| + \frac { \beta _ { 1 } } { \gamma } \| h \| \right) .
$$

Using $\| g ^ { ( t ) } \| ^ { 2 } \leq C H _ { \mathrm { P } } Y ^ { ( t ) }$ and $Y ^ { ( t ) } \le 2 \kappa L ( \theta ^ { ( t ) } )$ , we obtain

$$
\sqrt { H _ { \mathrm { P } } } \| \Delta Z ^ { ( t ) } \| \le C \left( \nu \sqrt { \kappa } + \sqrt { \frac { \nu } { \gamma } } \right) \sqrt { S } \le C \sqrt { S } ,
$$

where the last inequality uses $\kappa \lesssim \rho _ { b }$ and $\nu \rho _ { b } \lesssim \gamma$

By the global Lipschitz continuity of the auxiliary residual map, these estimates imply that $\| R \| \leq C { \sqrt { S } }$ throughout both segments $[ \theta ^ { ( t ) } , Z ^ { ( t ) } ]$ and $[ Z ^ { ( t ) } , Z ^ { ( t + 1 ) } ]$ . Consequently, the residualdependent Hessian bound gives

$$
\operatorname* { s u p } \| \nabla ^ { 2 } L \| \leq C H _ { \mathrm { P } } + C \sqrt { \frac { \ell _ { m } S } { m } } \leq C H _ { \mathrm { P } } + C \lambda _ { 0 } \leq C H _ { \mathrm { P } }
$$

on both segments, including any candidate update that exits the prescribed region. In particular, no Gram matrix lower bound is needed at the virtual iterates or at $\theta ^ { ( t + 1 ) }$

To transfer the remaining Adam estimates, consider the rescaling

$$
\begin{array} { r l r l r l r } { \widetilde { L } = \displaystyle \frac { L } { H _ { \mathrm { P } } } , } & { } & { \widetilde { g } = \displaystyle \frac { g } { H _ { \mathrm { P } } } , } & { } & { \widetilde { s } = \displaystyle \frac { s } { H _ { \mathrm { P } } } , } & { } & { \widetilde { v } = \displaystyle \frac { v } { H _ { \mathrm { P } } ^ { 2 } } , } \\ & { } & { \widetilde { \epsilon } = \displaystyle \frac { \epsilon } { H _ { \mathrm { P } } } , } & { } & { \widetilde { \eta } = H _ { \mathrm { P } } \bar { \eta } . } \end{array}
$$

The raw stepsize $\eta$ remains unchanged, and these substitutions preserve the parameter iterates, normalized preconditioners, and virtual iterates. The rescaled residual map $R / { \sqrt { H _ { \mathrm { P } } } }$ is globally Lipschitz, while the corresponding loss has a uniformly bounded Hessian on the two segments established above. The Gram matrix lower bound becomes $\lambda _ { 0 } / ( 4 H _ { \mathrm { P } } )$ , and the relative variance factor remains $\rho _ { b }$

Under this rescaling, the coordinatewise gradient bounds involve the factor $\ell _ { m } / H _ { \mathrm { P } }$ . In the proof of Lemma 15, these bounds enter only through the preconditioner lower bound and the normalizationbias estimate. Expressed in the original variables, the latter becomes

$$
\begin{array} { r l r } {  { \| \mathbb { E } _ { t } \big [ e _ { t } ( g ^ { ( t ) } ) \big ] \| ^ { 2 } \leq \frac { C \nu \rho _ { b } \ell _ { m } S } { \gamma \bar { \epsilon } ^ { 2 } } \| G ^ { ( t ) } \| ^ { 2 } } } \\ & { } & { \leq \frac { C \nu \rho _ { b } } { \gamma } \| G ^ { ( t ) } \| ^ { 2 } , ~ } \end{array}
$$

where $e _ { t }$ is defined in (81). The parameter conditions therefore provide the same uniform preconditioner lower bound and bias control as in the regression analysis.

The remaining steps of Lemma 15 rely only on the relative moment bounds, exact bias-correction identities, and smoothness along the two segments. Applying those arguments under the above rescaling establishes (117) and (118). Indeed, the rescaled Lyapunov coefficient satisfies $\widetilde { \zeta } = H _ { \mathrm { P } } \zeta$ so the corresponding energy is exactly $\mathcal { E } ^ { ( t ) } / H _ { \mathrm { P } }$

Finally, global residual Lipschitz continuity, together with $\zeta \geq C H _ { \mathrm { P } } \bar { \eta } ^ { 2 } ( k ^ { ( t ) } ) ^ { 2 }$ , yields the comparison bounds (119), as in Lemma 15. □

## E.3 Completion of the Proof

Proof of Theorem 4. Apply Lemma 16 with failure probability $\delta / 2 ,$ and fix an initialization in the resulting event. Define

$$
L _ { 0 } : = L ( \theta ^ { ( 0 ) } ) \leq S _ { 0 } : = C d ^ { 2 } \log \frac { 2 } { \delta } , \qquad S : = \frac { C _ { S } S _ { 0 } } { \delta } , \qquad \mathcal { R } : = c _ { 0 } \sqrt { \frac { m \lambda _ { 0 } } { \ell _ { m } } } ,
$$

where $C _ { S } > 0$ is a sufficiently large constant. Throughout the remainder of the proof, all probabilities and expectations are conditional on this initialization.

If $L _ { 0 } ~ = ~ 0$ , the iterates remain at initialization, and the conclusion follows immediately. We therefore assume $L _ { 0 } > 0$ . The width and damping conditions of Theorem 4 imply that

$$
m \ge \frac { C \ell _ { m } S } { \lambda _ { 0 } ^ { 2 } } , \qquad \bar { \epsilon } \ge C \sqrt { \ell _ { m } S } .
$$

We first analyze the auxiliary Adam process and then establish its agreement with the original trajectory.

Parameter conditions and initial iterations. Let $q = 1 - c _ { E } \bar { \eta } \lambda _ { 0 }$ , where $c _ { E } > 0$ is sufficiently small. Since $H _ { \mathrm { P } } \asymp d ^ { 2 }$ and $\bar { \eta } H _ { \mathrm { P } } \asymp \nu$ , the prescribed upper bound on ν implies that

$$
\begin{array} { c } { { \nu \rho _ { b } \lesssim \gamma , \qquad \bar { \eta } H _ { \mathrm { P } } T _ { * } \lesssim 1 , \qquad q ^ { - T _ { * } } \lesssim 1 , } } \\ { { 1 - q \leq \displaystyle \frac { 1 } { 2 } \operatorname* { m i n } \{ \nu , \gamma \} , \qquad \nu \kappa \log \displaystyle \frac { C } { ( 1 - q ) \delta } \leq c . } } \end{array}
$$

The last inequality follows from the monotonicity argument in the proof of Theorem 2, with $\lambda _ { 0 }$ replaced by $\lambda _ { 0 } / H _ { \mathrm { P } }$ and $\kappa \lesssim \rho _ { b }$ . Thus, the scalar concentration lemma applies to every admissible $\nu > 0$ , including values below the prescribed upper bound.

To control the initial iterations, we use the global Lipschitz continuity of the auxiliary residual map and the conditional second-moment estimate

$$
\begin{array} { r } { \mathbb { E } _ { t } \| g ^ { ( t ) } \| ^ { 2 } \leq C H _ { \mathrm { P } } L ( \theta ^ { ( t ) } ) . } \end{array}
$$

Let $M _ { t } = \operatorname* { m a x } _ { 0 \leq s \leq t } \mathbb { E } L ( \theta ^ { ( s ) } )$ . Since the bias-corrected first-moment weights form a convex combination,

$$
\mathbb { E } \| \widehat { s } ^ { ( t ) } \| ^ { 2 } \leq C H _ { \mathrm { P } } M _ { t } , \qquad M _ { t + 1 } ^ { 1 / 2 } \leq ( 1 + C \bar { \eta } H _ { \mathrm { P } } ) M _ { t } ^ { 1 / 2 } .
$$

Using these bounds and $\bar { \eta } H _ { \mathrm { P } } T _ { * } \lesssim 1$ , we obtain

$$
\mathbb { E } \operatorname* { m a x } _ { 0 \leq s \leq T _ { * } } L ( \theta ^ { ( s ) } ) + \mathbb { E } \mathcal { E } ^ { ( T _ { * } ) } \leq C L _ { 0 } ,
$$

$$
\mathbb { E } \left( \sum _ { s = 0 } ^ { T _ { * } - 1 } \lVert \theta ^ { ( s + 1 ) } - \theta ^ { ( s ) } \rVert \right) ^ { 2 } \leq \frac { C L _ { 0 } } { H _ { \mathrm { P } } } .\tag{120}
$$

Define the startup event

$$
\Pi = \left\{ \operatorname* { m a x } _ { 0 \leq s \leq T _ { * } } L ( \theta ^ { ( s ) } ) \leq S , \quad \sum _ { s = 0 } ^ { T _ { * } - 1 } \| \theta ^ { ( s + 1 ) } - \theta ^ { ( s ) } \| \leq \frac { \mathcal { R } } { 4 } \right\} .
$$

By Markov’s inequality,

$$
\mathbb { P } ( \Pi ^ { c } ) \leq \frac { C L _ { 0 } } { S } + \frac { C L _ { 0 } } { H _ { \mathrm { P } } \mathcal { R } ^ { 2 } } .
$$

Stopping and restricted energy. On the startup event Π, define τ as the first $t \geq T _ { * }$ at which any of the following conditions is violated:

$$
\mathcal { E } ^ { ( t ) } \leq S q ^ { t - T _ { * } } , \qquad U ^ { ( t - 1 ) } \leq C _ { U } S , \qquad \| \theta ^ { ( t ) } - \theta ^ { ( 0 ) } \| \leq \frac { \mathcal { R } } { 2 } .
$$

Specifically, we define

$$
\tau = \operatorname* { i n f } \left\{ t \geq T _ { * } : \begin{array} { l } { \mathcal { E } ^ { ( t ) } > S q ^ { t - T _ { * } } , \mathrm { ~ o r ~ } } \\ { U ^ { ( t - 1 ) } > C _ { U } S , \mathrm { ~ o r ~ } } \end{array} \right\} .
$$

Set $\tau = T _ { * }$ on $\Pi ^ { c }$ , and define the predictable indicators

$$
I ^ { ( t ) } = \mathbf { 1 } _ { \Pi } \mathbf { 1 } _ { \left\{ t < \tau \right\} } , \qquad t \geq T _ { * } .
$$

On each event $\{ I ^ { ( t ) } = 1 \}$

$$
L ( \theta ^ { ( t ) } ) \leq 2 \mathcal { E } ^ { ( t ) } \leq 2 S q ^ { t - T _ { * } } ,
$$

and the local Gram matrix lower bound holds. Thus, Lemma 21, applied with 2S in place of $S ,$ yields the stopped drift inequality.

We use the scalar-history masking argument from the stochastic regression proof. For $t < T _ { * }$ we retain a sampled residual only if the losses up to iteration t are bounded by S. For $t \geq T _ { * }$ , we use the indicators $I ^ { ( t ) }$ . The resulting predictable mask satisfies the loss bound $2 S q ^ { ( t - T _ { * } ) } +$ . On Π, all terms before $\tau$ are retained in the masked moving average. Thus, Lemma 14 bounds the probability of violating the second-moment condition by $\delta _ { v } ,$ , where $\delta _ { v }$ is a sufficiently small fixed fraction of $\delta .$

Moreover, the stopped process

$$
{ \mathbf { 1 } } _ { \Pi } q ^ { - ( ( t \wedge \tau ) - T _ { * } ) } \mathcal { E } ^ { ( t \wedge \tau ) }
$$

is a nonnegative supermartingale. Applying Lemma 6 together with (120) bounds the probability of an energy exit by $C L _ { 0 } / S$

Finally, define the restricted expected energy

$$
A _ { t } = \mathbb { E } [ I ^ { ( t ) } \mathcal { E } ^ { ( t ) } ] .
$$

The conditional drift inequality and the monotonicity of $I ^ { ( t ) }$ imply

$$
\begin{array} { r l } { A _ { t } - A _ { t + 1 } \geq \mathbb { E } [ I ^ { ( t ) } \mathcal { D } ^ { ( t ) } ] } & { { } } \\ { \geq ( 1 - q ) A _ { t } , \quad } & { { } A _ { t } \leq C q ^ { t - T _ { * } } L _ { 0 } . } \end{array}\tag{121}
$$

This argument uses only predictable indicators and does not condition on future survival.

Control of parameter movement. Decompose each virtual increment as

$$
\Delta Z ^ { ( t ) } = W ^ { ( t ) } + \xi ^ { ( t + 1 ) } , \qquad W ^ { ( t ) } = \mathbb { E } _ { t } [ \Delta Z ^ { ( t ) } ] ,
$$

where $\xi ^ { ( t + 1 ) }$ is a martingale difference. Combining the increment bounds in (118) with (121), and applying the telescoping and Minkowski arguments from the stochastic regression proof, we obtain

$$
\mathbb { E } \left( \sum _ { t \geq T _ { * } } I ^ { ( t ) } \| W ^ { ( t ) } \| \right) ^ { 2 } \leq \frac { C L _ { 0 } } { \lambda _ { 0 } } ,
$$

and

$$
\begin{array} { r l r } {  { \sum _ { t \geq T _ { * } } \mathbb { E } [ I ^ { ( t ) } \| \xi ^ { ( t + 1 ) } \| ^ { 2 } ] \leq \frac { C \bar { \eta } ( \rho _ { b } - 1 ) } { \gamma } L _ { 0 } } } \\ & { } & { \leq \frac { C L _ { 0 } } { H _ { \mathrm { P } } } . } \end{array}
$$

Combining these estimates with the startup bound (120) and the martingale maximal inequality gives

$$
\mathbb { E } \left[ \mathbf { 1 } _ { \Pi \operatorname* { s u p } _ { t \geq T _ { * } } } \left. Z ^ { ( t \wedge \tau ) } - \theta ^ { ( 0 ) } \right. ^ { 2 } \right] \leq \frac { C L _ { 0 } } { \lambda _ { 0 } } .\tag{122}
$$

Here we use $\lambda _ { 0 } \leq H _ { \mathrm { P } }$ , and the stopped sums include the increment that causes an exit.

If a radius exit occurs without an energy exit, the comparison bound (119) gives

$$
\| \theta ^ { ( \tau ) } - Z ^ { ( \tau ) } \| ^ { 2 } \leq \frac { C S } { H _ { \mathrm { P } } } \leq \frac { \mathcal { R } ^ { 2 } } { 1 6 } ,
$$

where the last inequality follows from the width condition. Consequently, the probability of such an exit is bounded by $C L _ { 0 } / ( \lambda _ { 0 } \mathcal { R } ^ { 2 } )$

Define the good sampling event $\mathcal { G } _ { 0 } = \Pi \cap \left\{ \tau = \infty \right\}$ . Combining the bounds for startup failure, energy exit, second-moment exit, and radius exit,

$$
\begin{array} { c } { \displaystyle { \mathbb { P } ( \mathcal { G } _ { 0 } ^ { c } ) \leq \frac { C L _ { 0 } } { S } + \frac { C L _ { 0 } } { H _ { \mathrm { P } } \mathcal { R } ^ { 2 } } + \frac { C L _ { 0 } } { \lambda _ { 0 } \mathcal { R } ^ { 2 } } + \delta _ { v } } } \\ { \displaystyle { \leq \frac { \delta } { 4 } } , } \end{array}
$$

provided that the constants in S and the width condition are chosen sufficiently large.

Since ${ \mathcal { G } } _ { 0 } \subseteq \{ I ^ { ( t ) } = 1 \}$ for every $t \geq T _ { * }$ , the restricted energy estimate (121) and the global loss comparison imply

$$
\mathbb { E } \left[ L ( \theta ^ { ( t ) } ) \mathbf { 1 } _ { \mathcal { G } _ { 0 } } \right] \leq C q ^ { t } L _ { 0 } , \qquad t \geq T _ { * } .
$$

For $t < T _ { * }$ , the same estimate follows from the startup bound and $q ^ { - T _ { * } } \lesssim 1$

Agreement with the original PINN. To establish uniform neuron-wise containment, define the second mask by

$$
\widehat { I } ^ { ( s ) } = \left\{ \begin{array} { l l } { 1 , } & { s < T _ { * } , } \\ { I ^ { ( s ) } , } & { s \geq T _ { * } . } \end{array} \right.
$$

Then $\mathbb { E } [ \widehat { I } ^ { ( s ) } L ( \theta ^ { ( s ) } ) ] \le C L _ { 0 } q ^ { s }$ . Define the scalar convolution

$$
\mathcal { T } = \bar { \eta } \sum _ { t \geq 0 } \sum _ { s = 0 } ^ { t } \frac { \gamma \beta _ { 1 } ^ { t - s } } { 1 - \beta _ { 1 } ^ { t + 1 } } \widehat { I } ^ { ( s ) } \sqrt { Y ^ { ( s ) } } .
$$

Using the exact bias-corrected moment weights, $\mathbb { E } _ { s } [ Y ^ { ( s ) } ] = 2 L ( \theta ^ { ( s ) } )$ , and $1 - q \lesssim \gamma$ , the argument from the stochastic regression proof gives

$$
\mathbb { E } [ \mathcal { T } ^ { 2 } ] \leq \frac { C L _ { 0 } } { \lambda _ { 0 } ^ { 2 } } , \qquad \sum _ { t > 0 } \operatorname* { m a x } _ { r } \| \theta _ { r } ^ { ( t + 1 ) } - \theta _ { r } ^ { ( t ) } \| \leq C \sqrt { \frac { \ell _ { m } } { m } } \mathcal { T } \quad \mathrm { o n } \mathcal { G } _ { 0 } .
$$

Now define

$$
\mathcal { G } = \mathcal { G } _ { 0 } \cap \left\{ \tau \leq C _ { T } \sqrt { \frac { L _ { 0 } } { \delta \lambda _ { 0 } ^ { 2 } } } \right\} .
$$

By Markov’s inequality, choosing $C _ { T }$ sufficiently large ensures $\mathbb { P } ( \mathcal { G } ) \ge 1 - \delta / 2$ . On this event, the total neuron-wise parameter movement satisfies

$$
\sum _ { t \ge 0 } \operatorname* { m a x } _ { r } \Vert \theta _ { r } ^ { ( t + 1 ) } - \theta _ { r } ^ { ( t ) } \Vert \leq C \sqrt { \frac { \ell _ { m } L _ { 0 } } { m \delta \lambda _ { 0 } ^ { 2 } } } \leq \frac { 1 } { 2 } ,\tag{123}
$$

where the last inequality follows from the width condition. Consequently, for every neuron $r$ and iteration $t ,$

$$
| a _ { r } ^ { ( t ) } | \leq \frac { 3 } { 2 } , \qquad \| w _ { r } ^ { ( t ) } - w _ { r } ^ { ( 0 ) } \| \leq \frac { 1 } { 2 } .
$$

Thus, the auxiliary trajectory remains in $S _ { \mathrm { P } }$ , where the auxiliary and original PINN residuals and their parameter derivatives coincide. Since both algorithms start from the same parameters and zero moments and use identical minibatches, their gradients, moment estimates, and iterates coincide by induction. The restricted expected-loss bound therefore transfers to the original Adam iterates on $\mathcal { G }$ establishing (33).

Finally, substituting $S _ { 0 } = C d ^ { 2 } \log ( 2 / \delta )$ recovers the width and damping requirements in Theorem $4 ,$ while the remaining width term ensures the initialization event. Combining the initialization probability of at least $1 - \delta / 2$ with the conditional sampling probability of at least $1 - \delta / 2$ gives a joint success probability of at least $1 - \delta$

Moreover, (123) implies that, almost surely on $\mathcal { G }$

$$
\sum _ { t = 0 } ^ { \infty } \operatorname* { m a x } _ { 1 \leq r \leq m } \left( | a _ { r } ^ { ( t + 1 ) } - a _ { r } ^ { ( t ) } | + \| w _ { r } ^ { ( t + 1 ) } - w _ { r } ^ { ( t ) } \| \right) < \infty .
$$

Since $m$ is finite, the complete parameter sequence converges to a finite limit $\theta ^ { \star }$ on $\mathcal { G }$ . Moreover, by Fatou $| \mathbf { \bar { s } } _ { }$ lemma, continuity of $L _ { i }$ , and the restricted expected-loss bound, we have

$$
\mathbb { E } [ L ( \theta ^ { \star } ) \mathbf { 1 } _ { \mathcal { G } } \mid \theta ^ { ( 0 ) } ] \leq \operatorname* { l i m i n f } _ { t  \infty } \mathbb { E } [ L ( \theta ^ { ( t ) } ) \mathbf { 1 } _ { \mathcal { G } } \mid \theta ^ { ( 0 ) } ] = 0 .
$$

Thus, $L ( \theta ^ { \star } ) = 0$ almost surely on ${ \mathcal { G } } _ { : }$ , which completes the proof.

## F Convergence Analysis of GD and SGD for PINNs

The following proposition establishes the GD and SGD convergence bounds reported in Table 1 under the assumptions of Section 3.

Proposition 1 (GD and SGD for the empirical PINN loss). Let $\delta \in ( 0 , 1 )$ and consider the update

$$
\boldsymbol { \theta } ^ { ( t + 1 ) } = \boldsymbol { \theta } ^ { ( t ) } - h \boldsymbol { g } ^ { ( t ) } .
$$

(i) For $G D ,$ let $g ^ { ( t ) } = \nabla L ( \theta ^ { ( t ) } )$ and $0 < h \lesssim d ^ { - 2 }$ . Under the width condition (25), with probability at least $1 - \delta$ over initialization,

$$
L ( \theta ^ { ( t ) } ) \leq ( 1 - c h \lambda _ { 0 } ) ^ { t } L ( \theta ^ { ( 0 ) } ) , \qquad t \geq 0 .
$$

(ii) For SGD, let $g ^ { ( t ) }$ be the minibatch gradient defined in (28), and suppose that $0 ~ < ~ h ~ \lesssim$ $( d ^ { 2 } \rho _ { b } ) ^ { - 1 }$ . Under the width condition (32), with probability at least $1 - \delta / 2$ over initialization, there exists a sampling event $\mathcal { G }$ such that

$$
\begin{array} { r l } { \mathbb { P } ( \mathscr { G } \mid \theta ^ { ( 0 ) } ) \geq 1 - \delta / 2 , } & { { } } \\ { \mathbb { E } \big [ L ( \theta ^ { ( t ) } ) \mathbf { 1 } _ { \mathscr { G } } \mid \theta ^ { ( 0 ) } \big ] \leq ( 1 - c h \lambda _ { 0 } ) ^ { t } L ( \theta ^ { ( 0 ) } ) , } & { { } \quad t \geq 0 . } \end{array}
$$

The joint success probability is at least $1 - \delta .$

Consequently, choosing $h \asymp d ^ { - 2 } f o r G D$ and $h \asymp ( d ^ { 2 } \rho _ { b } ) ^ { - 1 }$ for SGD gives contraction gaps of order $\lambda _ { 0 } / d ^ { 2 }$ and $\lambda _ { 0 } / ( d ^ { 2 } \rho _ { b } )$ , respectively.

Proof. Apply Lemma 16 with failure probability δ for GD and $\delta / 2$ for SGD. Fix an initialization in the corresponding event, and let $L _ { 0 } = L ( \theta ^ { ( 0 ) } ) \le S _ { 0 } : = C d ^ { 2 } \log ( 2 / \delta )$ and $\mathcal { R } = c _ { 0 } \sqrt { m \lambda _ { 0 } / \ell _ { m } }$ . We use the auxiliary PINN from Lemma 19 and suppress the superscript † throughout the proof. All probabilities and expectations below are conditional on the fixed initialization. We assume $L _ { 0 } > 0$ as the claim is immediate otherwise. The global auxiliary bounds give

$$
\| R ( { \boldsymbol { \theta } } ^ { ( t + 1 ) } ) \| \leq ( 1 + C h d ^ { 2 } \sqrt { \kappa } ) \| R ( { \boldsymbol { \theta } } ^ { ( t ) } ) \|
$$

for every minibatch realization.

One-step descent. Let $H _ { \mathrm { P } } = C _ { H } d ^ { 2 } \geq \lambda _ { 0 }$ , as in Appendix E. Suppose that $L ( \theta ^ { ( t ) } ) \leq S$ , m $\gtrsim$ $\ell _ { m } S / \lambda _ { 0 } ^ { 2 }$ , and $\boldsymbol { \theta } ^ { ( t ) }$ lies in the kernel neighborhood. The auxiliary regularity and sampling lemmas imply that, for every minibatch realization,

$$
\operatorname* { s u p } _ { \vartheta \in [ \theta ^ { ( t ) } , \theta ^ { ( t ) } - h g ^ { ( t ) } ] } \| R ( \vartheta ) \| \lesssim ( 1 + h H _ { \mathrm { P } } \sqrt { \kappa } ) \sqrt { S } \lesssim \sqrt { S } ,
$$

where the last inequality uses $\kappa ~ \lesssim ~ \rho _ { b }$ and $h H _ { \mathrm { P } } \rho _ { b } \lesssim 1$ . Hence, (110) gives a Hessian bound of $C H _ { \mathrm { P } }$ along the entire update segment, even if the candidate update leaves the kernel neighborhood. Writing ${ G } ^ { ( t ) } = \nabla L ( \theta ^ { ( t ) } )$ , unbiasedness and Lemma 20 yield

$$
\begin{array} { r l r } {  { \mathbb { E } _ { t } L ( \theta ^ { ( t + 1 ) } ) \leq L ( \theta ^ { ( t ) } ) - h \| G ^ { ( t ) } \| ^ { 2 } + C H _ { \mathrm { P } } h ^ { 2 } \rho _ { b } \| G ^ { ( t ) } \| ^ { 2 } } } \\ & { } & { \leq L ( \theta ^ { ( t ) } ) - \frac { h } { 2 } \| G ^ { ( t ) } \| ^ { 2 } } \\ & { } & { \leq q L ( \theta ^ { ( t ) } ) , \qquad q : = 1 - c h \lambda _ { 0 } \in ( 0 , 1 ) . } \end{array}\tag{124}
$$

For GD, we use full batches, so $\rho _ { b } = \kappa = 1$ and no conditional expectation is needed.

We first consider GD. Set $S = L _ { 0 }$ . Applying the descent and square-root telescoping arguments from the proof of Lemma 18 to the update $- h G ^ { ( t ) }$ , we obtain

$$
\sum _ { t } h \| G ^ { ( t ) } \| \lesssim \sqrt { \frac { L _ { 0 } } { \lambda _ { 0 } } } , \quad \quad \sum _ { t } \operatorname* { m a x } _ { r } \| \theta _ { r } ^ { ( t + 1 ) } - \theta _ { r } ^ { ( t ) } \| \lesssim \frac { 1 } { \lambda _ { 0 } } \sqrt { \frac { \ell _ { m } L _ { 0 } } { m } } .
$$

These bounds hold for every finite trajectory segment up to and including a candidate exit step. The second bound follows from the block-gradient estimate in Lemma 20, with full batches, and the geometric decay of $\sqrt { L ( \theta ^ { ( t ) } ) }$ . The width condition (25) ensures that the two bounds are at most $\mathcal { R } / 2$ and $1 / 2 ,$ , respectively. Thus, by induction, the iterates remain in the Gram matrix neighborhood and in $S _ { \mathrm { P } }$ . The auxiliary and original GD trajectories therefore coincide, proving (i).

We next consider SGD. Set $S = C _ { S } S _ { 0 } / \delta$ and define

$$
\tau = \operatorname* { i n f } \left. t \geq 0 : L ( \theta ^ { ( t ) } ) > S q ^ { t } \mathrm { o r } \| \theta ^ { ( t ) } - \theta ^ { ( 0 ) } \| > \mathcal { R } / 2 \right. , \qquad I ^ { ( t ) } = \mathbf { 1 } _ { \left. t < \tau \right. } .
$$

The width condition ensures that m $\gtrsim \ell _ { m } S / \lambda _ { 0 } ^ { 2 }$ . By (124), $q ^ { - ( t \wedge \tau ) } L ( \theta ^ { ( t \wedge \tau ) } )$ is a nonnegative supermartingale. Define $A _ { t } = \mathbb { E } [ I ^ { ( t ) } L ( \theta ^ { ( t ) } ) ]$ and $\Delta _ { t } = A _ { t } - A _ { t + 1 }$ . Then

$$
A _ { t } \leq q ^ { t } L _ { 0 } , \qquad \Delta _ { t } \geq \frac { h } { 2 } \mathbb { E } [ I ^ { ( t ) } \| G ^ { ( t ) } \| ^ { 2 } ] , \qquad \Delta _ { t } \geq ( 1 - q ) A _ { t } .
$$

As in Appendix E, we decompose the update into its predictable and martingale increments, − $- h G ^ { ( t ) }$ and $- \bar { h ( g ^ { ( t ) } - G ^ { ( t ) } ) }$ , respectively. Here, no startup period is needed. The key telescoping estimate is

$$
\sqrt { \Delta _ { t } } \leq \frac { 2 } { \sqrt { 1 - q } } \big ( \sqrt { A _ { t } } - \sqrt { A _ { t + 1 } } \big ) .
$$

Applying Minkowski’s inequality and the martingale maximal inequality in Lemma 6, first over finite horizons and then passing to the limit by monotone convergence, we obtain

$$
\begin{array} { r l } { \mathbb { E } \Bigg [ \Bigg ( \sum _ { t \geq 0 } I ^ { ( t ) } h \| G ^ { ( t ) } \| \Bigg ) ^ { 2 } \Bigg ] \lesssim \frac { h L _ { 0 } } { 1 - q } \lesssim \frac { L _ { 0 } } { \lambda _ { 0 } } , } & { } \\ { \sum _ { t \geq 0 } \mathbb { E } \big [ I ^ { ( t ) } h ^ { 2 } \| g ^ { ( t ) } - G ^ { ( t ) } \| ^ { 2 } \big ] \lesssim h ( \rho _ { b } - 1 ) L _ { 0 } \lesssim \frac { L _ { 0 } } { H _ { \mathrm { P } } } , } & { } \\ { \mathbb { E } \underset { t \geq 0 } { \operatorname* { s u p } } \| \theta ^ { ( t \wedge \tau ) } - \theta ^ { ( 0 ) } \| ^ { 2 } \lesssim \frac { L _ { 0 } } { \lambda _ { 0 } } . } & { } \end{array}
$$

The stopped sums include the increment that causes an exit. The maximal inequality for the loss supermartingale and Markov’s inequality for the parameter movement therefore give

$$
\mathbb { P } ( \tau < \infty ) \lesssim \frac { L _ { 0 } } { S } + \frac { L _ { 0 } } { \lambda _ { 0 } \mathcal { R } ^ { 2 } } \leq \frac { \delta } { 4 } .
$$

To establish neuron-wise containment, define $\begin{array} { r } { \mathcal { T } = h \sum _ { t \geq 0 } I ^ { ( t ) } \sqrt { Y ^ { ( t ) } } } \end{array}$ . By (115) and Minkowski’s inequality,

$$
\mathbb { E } [ \mathcal { T } ^ { 2 } ] \le 2 h ^ { 2 } \left( \sum _ { t \ge 0 } \sqrt { A _ { t } } \right) ^ { 2 } \lesssim \frac { L _ { 0 } } { \lambda _ { 0 } ^ { 2 } } .
$$

On $\{ \tau = \infty \}$ , the total neuron-wise parameter movement is bounded by $C \sqrt { \ell _ { m } / m } T$ . Define

$$
\mathcal { G } = \left\{ \tau = \infty \right\} \cap \left\{ \tau \leq C _ { T } \sqrt { \frac { L _ { 0 } } { \delta \lambda _ { 0 } ^ { 2 } } } \right\} .
$$

By Markov’s inequality, choosing $C _ { T }$ sufficiently large gives $\mathbb { P } ( \mathcal { G } ) \ge 1 - \delta / 2$ . On ${ \mathcal { G } } .$ , the width condition ensures that the total neuron-wise movement is at most $1 / 2$ . Thus, the auxiliary iterates remain in $S _ { \mathrm { P } }$ , and the original and auxiliary SGD trajectories coincide. Since $I ^ { ( t ) } = 1$ on $\mathcal { G }$ for every $t \geq 0$ , the bound $A _ { t } \leq q ^ { t } L _ { 0 }$ establishes (ii). □