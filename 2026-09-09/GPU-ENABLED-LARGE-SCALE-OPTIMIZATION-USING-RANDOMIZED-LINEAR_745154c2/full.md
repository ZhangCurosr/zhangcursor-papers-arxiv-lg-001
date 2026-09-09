# GPU-ENABLED LARGE-SCALE OPTIMIZATION USING RANDOMIZED LINEAR ALGEBRA

PREPRINT

Pratik Rathore<sup>\*</sup> Zachary Frangella<sup>\*</sup> Parth Nobel

Xuning Hu Madeleine Udell

Stanford University

pratikr@alumni.stanford.edu zfrangella@alumni.stanford.edu ptnobel@alumni.stanford.edu xuningh@stanford.edu udell@stanford.edu

## ABSTRACT

This paper introduces rlaopt, a PyTorch-based package for large-scale optimization and scientific computing using randomized numerical linear algebra (RandNLA). Despite substantial progress in RandNLA-based algorithms, few implementations combine GPU acceleration with a simple interface for specifying optimization problems. rlaopt addresses this gap by providing GPU-enabled solvers for positive-definite linear systems and convex empirical risk minimization with constraints and regularizers. These solvers use RandNLA to accelerate conjugate gradient (NyströmPCG), operator splitting (NysADMM), and stochastic gradient methods (SAPPHIRE). Moreover, rlaopt includes a modeling language that lets users specify problems using natural mathematical syntax. rlaopt automatically checks compatibility with the selected solver and performs the required problem decomposition. The solvers also support differentiation through their iterations, enabling applications such as hyperparameter tuning. Experiments on ridge regression, bounded multinomial logistic regression, and bounded elastic net identify when randomized preconditioning improves performance and demonstrate substantial speedups from GPU execution. The package is open-source under an Apache license, with source code at https://github.com/udellgroup/rlaopt and version 0.1.0 available on PyPI.

Keywords randomized linear algebra, stochastic optimization, operator splitting, scientific computing, machine learning

## 1 Introduction

Large-scale optimization lies at the heart of modern machine learning and scientific computing. Datasets routinely contain millions of samples and features, giving rise to optimization problems whose sheer scale demands efficient algorithms and hardware-aware implementations. Over the past two decades, randomized numerical linear algebra (RandNLA) has emerged as a powerful toolkit for addressing these challenges, producing algorithms that exploit lowrank structure and stochastic approximations to dramatically reduce computational costs [Halko et al. 2011; Mahoney 2011; Woodruff 2014; Martinsson and Tropp 2020]. However, a significant gap remains between the strong algorithmic foundations of RandNLA and the software available to practitioners: existing implementations of RandNLAbased methods lack the ability to leverage modern parallel hardware such as GPUs, and do not provide a simple, user-friendly syntax for modeling optimization problems. Consequently, practitioners who wish to apply RandNLAbased algorithms to large-scale problems must piece together ad hoc implementations, limiting both accessibility and performance.

We now describe two important problem classes where RandNLA-based algorithms offer significant advantages, and where the lack of high-quality implementations is particularly acute.

Large-scale linear systems. Dense linear systems of the form $( A + \lambda I ) x \ : = \ : b$ arise in kernel ridge regression [Schölkopf and Smola 2002], Gaussian process inference [Rasmussen and Williams 2005; Gardner et al. 2018], and other settings throughout machine learning and scientific computing. Direct methods such as Cholesky decomposition require $\mathcal { O } ( \bar { n } ^ { 3 } )$ computation and $O ( n ^ { 2 } )$ storage, limiting them to problems with $n \sim 1 0 ^ { 4 }$ . Iterative methods such as conjugate gradient scale more favorably, with per-iteration complexity $O ( n ^ { 2 } )$ , but converge slowly when the problem is ill-conditioned—a common occurrence in practice. RandNLA-based preconditioning, in particular the randomized Nyström preconditioner [Frangella et al. 2023], dramatically accelerates convergence by constructing high-quality low-rank approximations of the kernel matrix at modest cost. The resulting preconditioned conjugate gradient (NyströmPCG) method combines the scalability of iterative methods with robustness to ill-conditioning.

Large-scale optimization. Classical optimization methods, such as interior-point methods, produce high-accuracy solutions but rely on expensive matrix factorizations that limit their applicability to large-scale problems [O’Donoghue et al. 2016; Stellato et al. 2020; Applegate et al. 2021]. On the other end of the spectrum, first-order methods such as stochastic gradient descent (SGD) and its variants [Robbins and Monro 1951; Johnson and Zhang 2013; Defazio et al. 2014; Allen-Zhu 2018] scale to massive datasets but suffer from slow convergence on ill-conditioned problems and sensitivity to hyperparameters such as the learning rate [Nemirovski et al. 2009]. Operator splitting frameworks such as the alternating direction method of multipliers (ADMM) [Boyd et al. 2011] naturally handle constraints and nonsmooth regularizers, but also require the solution of subproblems that can be ill-conditioned. RandNLA techniques integrate naturally with both stochastic gradient methods and operator splitting: Nyström-based preconditioning accelerates ADMM by improving the conditioning of linear system subproblems [Zhao et al. 2022; Diamandis et al. 2026], and randomized curvature estimates yield preconditioned stochastic gradient methods with reliable default hyperparameters and fast convergence [Frangella et al. 2024b,a; Sun et al. 2025]. Crucially, the dominant operations in these RandNLA-enhanced methods are matrix-matrix and matrix-vector products, which are highly amenable to GPU acceleration.

To address the gap between RandNLA algorithms and practical software, we develop rlaopt, a PyTorch-based package for large-scale optimization and scientific computing. rlaopt includes GPU-enabled implementations of NyströmPCG for large-scale positive-definite linear systems, NysADMM [Zhao et al. 2022] for constrained convex optimization, and SAPPHIRE [Frangella et al. 2024a; Sun et al. 2025] for empirical risk minimization. To make these solvers accessible, we create a flexible modeling language inspired by disciplined convex programming [Grant et al. 2006] and CVXPY [Diamond and Boyd 2016] that allows users to specify optimization problems using natural mathematical syntax. rlaopt also supports differentiating through the solver, making it suitable for modern machine learning pipelines that require end-to-end gradient computation. Our implementation is open-sourced under an Apache license and is available at https://github.com/udellgroup/rlaopt. Documentation can be found at https://rlaopt.readthedocs.io, and version 0.1.0 is available on PyPI.

## 1.1 Contributions

Our contributions are as follows:

1. We develop rlaopt, an open-source, PyTorch-based software package for large-scale optimization using randomized linear algebra. rlaopt provides a shared solver interface for CPU and GPU execution.

2. We create a modeling language, inspired by disciplined convex programming [Grant et al. 2006] and CVXPY [Diamond and Boyd 2016], that lets users compose losses, regularizers, and constraints using natural mathematical syntax. rlaopt checks compatibility with the user’s selected solver and automatically introduces auxiliary variables and linear constraints for ADMM splitting.

3. We implement a suite of RandNLA-based algorithms within rlaopt, including NyströmPCG for large-scale positive-definite linear systems, NysADMM for constrained convex optimization via operator splitting, and SAPPHIRE for preconditioned, stochastic variance-reduced optimization.

4. We evaluate the methods in rlaopt against state-of-the-art competitor methods for large-scale ridge regression, bounded multinomial logistic regression, and bounded elastic net regression. The experiments characterize the types of problems for which the RandNLA-based solvers in rlaopt outcompete state-of-the-art competitors, and identify regimes where GPU provides significant speedups over CPU.

5. We show that rlaopt supports differentiating through the solver, enabling applications such as hyperparameter tuning and end-to-end learning within modern machine learning pipelines.

## 1.2 Roadmap

Section 2 formally defines the problem classes addressed by rlaopt. Section 3 surveys related work organized by problem class. Section 4 describes the modeling language and demonstrates its flexibility through examples. Section 5 explains how rlaopt automatically detects problem structure, checks compatibility with the user-selected solver, and performs the required decomposition. Section 6 presents performance benchmarks comparing CPU and GPU implementations across problem classes. Section 7 concludes the paper.

## 2 Problem Classes

In this section, we formally define the problem classes addressed by rlaopt. We consider two broad classes: positive definite linear systems (Section 2.1) and empirical risk minimization with constraints and regularizers (Section 2.2).

## 2.1 Positive-Definite Linear Systems

The first problem class consists of symmetric positive-definite (pd) linear systems of the form

$$
A x = b ,\tag{1}
$$

where $A \in \mathbb { R } ^ { n \times n }$ is symmetric and pd, $b \in \mathbb { R } ^ { n }$ , and $x \in \mathbb { R } ^ { n }$ is the unknown. A common special case is the regularized linear system

$$
( K + \lambda I ) x = b ,\tag{2}
$$

where $K \in \mathbb { R } ^ { n \times n }$ is symmetric positive-semidefinite (psd) and $\lambda > 0$ is a regularization parameter.

Linear systems of this form arise frequently in machine learning and scientific computing. A prominent example is kernel ridge regression (KRR), in which one solves

$$
\operatorname* { m i n i m i z e } _ { \boldsymbol { w } \in \mathbb { R } ^ { n } } \ \mathsf { \frac { 1 } { 2 } } \| \boldsymbol { K } \boldsymbol { w } - \boldsymbol { y } \| ^ { 2 } + \frac { \lambda } { 2 } \| \boldsymbol { w } \| _ { K } ^ { 2 } ,\tag{3}
$$

where $K \in \mathbb { R } ^ { n \times n }$ is a kernel matrix with entries $K _ { i j } = k ( x _ { i } , x _ { j } )$ for a kernel function $k ,$ and $y \in \mathbb { R } ^ { n }$ is the target vector. The optimality conditions of (3) yield the linear system $( \check { K } + \lambda I ) w ^ { \star } = y$ . Linear systems of the form (2) also arise in Gaussian process (GP) inference [Rasmussen and Williams 2005], where computing the posterior mean and variance requires solving dense linear systems involving a kernel matrix.

Direct methods such as Cholesky decomposition solve (1) in $\mathcal { O } ( n ^ { 3 } )$ time with $\mathcal { O } ( n ^ { 2 } )$ storage, limiting them to problems with $n \lesssim 1 0 ^ { 4 }$ . Iterative methods such as conjugate gradient (CG) scale more favorably, with per-iteration complexity $\mathcal { O } ( n ^ { 2 } )$ , but converge slowly when A is ill-conditioned. Ill-conditioning is common in practice: kernel matrices in machine learning often have rapidly decaying spectra, leading to large condition numbers [Caponnetto and DeVito 2007; Bach 2013; Tu et al. 2016; Ma and Belkin 2017; Belkin 2018]. rlaopt addresses this challenge by providing NyströmPCG, which uses a randomized Nyström preconditioner [Frangella et al. 2023] to dramatically accelerate the convergence of CG on ill-conditioned systems.

## 2.2 Empirical Risk Minimization with Constraints and Regularizers

The second problem class is composite convex optimization of the form

$$
\operatorname* { m i n i m i z e } _ { x \in \mathbb { R } ^ { n } } \quad f ( x ) + \sum _ { i = 1 } ^ { k } g _ { i } ( A _ { i } x - b _ { i } ) ,\tag{4}
$$

where $f \colon  { \mathbb { R } } ^ { n } \to  { \mathbb { R } }$ is smooth and convex, each $g _ { i } \colon \mathbb { R } ^ { m _ { i } }  \mathbb { R } \cup \{ + \infty \}$ is closed, convex, and proxable (i.e., its proximal operator can be evaluated efficiently), $A _ { i } \in \mathbb { R } ^ { m _ { i } \times n }$ , and $b _ { i } \in \mathbb { R } ^ { m _ { i } }$ <sup>i</sup>. This formulation is flexible enough to encode a wide range of machine learning problems, including empirical risk minimization (ERM) with constraints and regularizers.

In a typical ERM setting, the smooth component f takes the form

$$
f ( \boldsymbol { x } ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \ell _ { j } ( a _ { j } ^ { T } \boldsymbol { x } ) ,\tag{5}
$$

where $\{ ( a _ { j } , y _ { j } ) \} _ { j = 1 } ^ { N }$ is a training set with $a _ { i } \in \mathbb { R } ^ { n }$ and $y _ { j } \in \mathbb { R }$ , and $\ell _ { j }$ is a loss function. The nonsmooth terms $g _ { i }$ encode regularizers and constraints such as $\ell _ { 1 }$ regularization, box constraints, elastic net penalties, and indicator functions of convex sets.

Several concrete problems fit naturally into this framework:

• ℓ<sub>1</sub>-regularized logistic regression: $f$ is the average logistic loss and $g ( x ) = \mu \| x \|$ for some regularization weight $\mu > 0$

• Bounded elastic net: $\begin{array} { r } { f ( w ) = \frac { 1 } { 2 N } \| X w - y \| _ { 2 } ^ { 2 } } \end{array}$ , with $\begin{array} { r } { g _ { 1 } ( w ) = \lambda _ { 1 } \| w \| _ { 1 } + \frac { \lambda _ { 2 } } { 2 } \| w \| _ { 2 } ^ { 2 } } \end{array}$ and $g _ { 2 }$ encoding box constraints $0 \leq w \leq 1$

• Constrained multinomial regression: $f$ is the average cross-entropy loss for multiclass classification, with g encoding box constraints on the regression coefficients.

rlaopt provides several solvers for problems of the form (4). For problems where the full gradient of $f$ is available, rlaopt supports NysADMM [Zhao et al. 2022] which combines ADMM with Nyström-based preconditioning to accelerate linear system subproblems that arise in ADMM. For large-scale ERM problems where computing a full gradient of $f$ at each iteration is too expensive, rlaopt supports SAPPHIRE [Frangella et al. 2024a; Sun et al. 2025], a family of preconditioned stochastic variance-reduced methods that uses randomized curvature estimates to achieve fast convergence with reliable default hyperparameters. rlaopt also supports (accelerated) proximal gradient for problems where preconditioning is not needed.

## 3 Related Work

We survey related work organized by the problem classes introduced in Section 2. For each class, we describe existing algorithmic approaches and software, and discuss how rlaopt relates to and improves upon them.

## 3.1 Large-Scale Linear Systems

Direct methods. Direct methods such as Cholesky decomposition are the standard approach for solving dense pd linear systems [Golub and Van Loan 2013]. While they produce high-accuracy solutions, their $\mathcal { O } ( n ^ { 3 } )$ computational cost and $\mathcal { O } ( n ^ { 2 } )$ storage requirements render them impractical for problems with $n \gtrsim 1 0 ^ { 4 }$ . rlaopt targets the regime where direct methods are too expensive, providing iterative solvers that scale to much larger problem sizes.

Iterative methods and preconditioning. Conjugate gradient (CG) is the method of choice for large-scale pd linear systems, with per-iteration complexity ${ \bar { \mathcal { O } } } ( n ^ { 2 } )$ for dense systems. However, CG converges slowly when the system is ill-conditioned, and effective preconditioners are essential for practical performance. Randomized Nyström preconditioning [Frangella et al. 2023] constructs a high-quality low-rank approximation of the coefficient matrix using sketching techniques from RandNLA [Halko et al. 2011; Martinsson and Tropp 2020]. The resulting NyströmPCG method converges rapidly even on ill-conditioned problems, and its dominant cost—matrix-matrix products for constructing the preconditioner—is highly GPU-amenable. rlaopt provides a GPU-enabled implementation of NyströmPCG; Section 6 evaluates its performance across spectral decay rates and regularization levels.

Kernel methods. FALKON [Rudi et al. 2017] is a widely used solver for inducing points kernel ridge regression that uses Nyström preconditioning with CG. Gaussian process inference packages such as GPyTorch [Gardner et al. 2018] also rely on PCG-based linear solvers. rlaopt differs from these tools by targeting general pd linear systems rather than a specific application and providing a modeling language for specifying the problem.

## 3.2 Optimization Solvers

Interior-point methods. Interior-point methods (IPMs) are the gold standard for small-to-moderate-scale convex optimization, producing high-accuracy solutions with polynomial-time guarantees [Nesterov and Nemirovskii 1994]. Software implementations such as Clarabel [Goulart and Chen 2026] provide reliable IPM solvers, and recent work has extended these to GPU [Chen et al. 2026]. However, IPMs rely on matrix factorizations whose cost grows cubically with problem size, limiting their applicability to large-scale machine learning problems. rlaopt takes a complementary approach, using first-order and operator splitting methods enhanced with RandNLA to handle problems at scales where IPMs are impractical.

Operator splitting methods. The alternating direction method of multipliers (ADMM) [Gabay and Mercier 1976; Boyd et al. 2011] and related operator splitting methods [Ryu and Yin 2022] decompose composite optimization problems into simpler subproblems that can be solved independently. ADMM is the basis for several widely used solvers, including OSQP [Stellato et al. 2020] for quadratic programs and SCS [O’Donoghue et al. 2016] for conic programs. A key bottleneck in ADMM is solving a subproblem corresponding to the primal update. NysADMM [Zhao et al.

2022] accelerates inexact ADMM by solving the primal subproblem with NyströmPCG, using a randomized Nyström preconditioner to handle ill-conditioning; for non-quadratic smooth losses, the subproblem is formed from a secondorder approximation of the smooth term. GeNIOS [Diamandis et al. 2026] builds on this recipe with adaptive penalty parameter updates, periodic preconditioner refresh, an adaptive inexact PCG tolerance, ADMM over-relaxation, and primal/dual infeasibility detection, with an open-source Julia implementation. rlaopt’s NysADMM solver inherits these engineering choices from GeNIOS (with the exception of infeasibility detection), adapted to a PyTorch/GPU setting. A related line of work specializes operator splitting to linear programming: PDLP [Applegate et al. 2021] applies the primal-dual hybrid gradient method to a saddle-point formulation of LP with diagonal preconditioning, adaptive step sizes, and adaptive restarts, and cuPDLP [Lu and Yang 2025] ports this approach to GPUs. These solvers share rlaopt’s philosophy of scaling on GPU through matrix-vector products, but target LP specifically rather than the broader composite convex setting.

Proximal gradient methods. Proximal gradient methods [Parikh and Boyd 2014] and their accelerated variants are a natural fit for composite problems of the form (4) when the proximal operator of each $g _ { i }$ is cheap to evaluate. These methods have per-iteration costs dominated by gradient evaluations of the smooth component $f ,$ but converge slowly on ill-conditioned problems. rlaopt implements proximal gradient with optional Nyström-based preconditioning, which improves convergence on ill-conditioned objectives while preserving the simplicity of the proximal gradient framework.

Stochastic gradient methods. When the smooth component $f$ has a finite-sum structure, stochastic gradient methods such as SGD [Robbins and Monro 1951], Adam [Kingma and Ba 2014], SVRG [Johnson and Zhang 2013], SAGA [Defazio et al. 2014], and Katyusha [Allen-Zhu 2018; Kovalev et al. 2020] achieve low per-iteration costs by operating on mini-batches. However, these methods are sensitive to hyperparameters and converge slowly on ill-conditioned problems [Nemirovski et al. 2009]. The PROMISE framework [Frangella et al. 2024a] and SketchySGD [Frangella et al. 2024b] address these limitations by using randomized curvature estimates as preconditioners, yielding methods with reliable default hyperparameters and fast convergence on ill-conditioned problems. SAPPHIRE [Sun et al. 2025] generalizes this line of work to proximal settings, supporting nonsmooth regularizers and constraints within the preconditioned stochastic gradient framework. rlaopt provides GPU-enabled implementations of SAPPHIRE methods; Section 6 compares their CPU and GPU performance with deterministic baselines on bounded multinomial logistic regression.

Differentiable optimization. Several frameworks enable differentiating through the solution of optimization problems. cvxpylayers [Agrawal et al. 2019] embeds parametrized convex programs specified in CVXPY as differentiable layers, using implicit differentiation through the KKT conditions of the conic reformulation. This approach supports a broad class of disciplined convex programs. Its computational cost depends on the conic reformulation, the solver, and the linear algebra used for differentiation; the diffcp backend supports iterative linear solves for derivative evaluation. JAXopt [Blondel et al. 2022] provides a modular implicit differentiation framework in JAX with a wide range of solvers, including proximal gradient, L-BFGS, and OSQP. MPAX [Lu et al. 2024] is a JAX-based first-order solver for large-scale linear and quadratic programs that supports differentiation through unrolled solver iterations. These frame works are complementary to rlaopt: they cover different problem classes or ecosystems, while rlaopt contributes differentiable, RandNLA-based solvers in PyTorch that are specifically designed for large-scale, ill-conditioned prob lems.

## 4 Modeling Language

A central design goal of rlaopt is to provide a simple, expressive interface for specifying optimization problems. To this end, we develop a modeling language inspired by disciplined convex programming [Grant et al. 2006] and CVXPY [Diamond and Boyd 2016] that lets users construct objectives from composable building blocks using natural mathematical syntax. In this section, we describe the core abstractions of the modeling language and demonstrate its flexibility through examples.

## 4.1 Core Abstractions

The rlaopt modeling language is built around three core abstractions: variables, atoms, and solvers.

Variables. A Variable represents an unknown quantity to be optimized. Variables are created by specifying their shape and, optionally, a name and a device (CPU or GPU):

<table><tr><td>Category</td><td>Atom</td><td>Mathematical form</td></tr><tr><td rowspan="2">General</td><td>SumSquares(expr)</td><td> $\| \mathrm { e x p r } \| _ { 2 } ^ { 2 }$ </td></tr><tr><td>QuadForm(expr, Q)</td><td> $\mathrm { e x p r } ^ { T } Q \mathrm { e x p r }$ </td></tr><tr><td rowspan="9"></td><td>LinearRegression(w, loader) LogisticRegression(w, loader)</td><td> $\begin{array} { r } { { \frac { 1 } { N } } \sum _ { j } ( y _ { j } - z _ { j } ) ^ { 2 } } \end{array}$   $\begin{array} { r } { \frac { \mathrm { ~ i ~ } } { N } \sum _ { j } ^ { ^ { \prime } } [ \log ( 1 + e ^ { z _ { j } } ) - y _ { j } z _ { j } ] } \end{array}$ </td></tr><tr><td></td><td> $\begin{array} { r } { - \frac { 1 } { N } \sum _ { j } \log ( \operatorname { s o f t m a x } ( z _ { j } ) _ { c _ { j } } ) } \end{array}$ </td></tr><tr><td>MultinomialRegression(w, loader)</td><td>Poisson negative log-likelihood</td></tr><tr><td>Linear model PoissonRegression(w, loader)</td><td>Gamma negative log-likelihood</td></tr><tr><td>GammaRegression(w, loader)</td><td>InverseGaussianRegression(w, loader) Inverse Gaussian negative log-likelihood</td></tr><tr><td>CompoundPoissonGammaRegression</td><td></td></tr><tr><td> $( \mathtt { w } ,$  loader, power=q)</td><td>Tweedie loss,  $1 < q < 2$ </td></tr><tr><td>HuberRegression(w, loader, delta=δ)</td><td> $\begin{array} { r } { \frac { 1 } { N } \sum _ { j } h _ { \delta } \bigl ( y _ { j } - z _ { j } \bigr ) } \end{array}$ </td></tr><tr><td></td><td></td></tr><tr><td rowspan="5">Regularizers</td><td>L1Norm(w, λ) L2Norm(w, λ)</td><td> $\lambda \| w \| _ { 1 }$   $\lambda \| w \| _ { 2 }$ </td></tr><tr><td>LInfNorm(w, λ)</td><td> $\lambda \| w \| _ { \infty }$ </td></tr><tr><td>NucNorm(W, λ)</td><td> $\lambda \ddot { \parallel } W \parallel _ { * }$  (nuclear norm)</td></tr><tr><td></td><td></td></tr><tr><td> $\mathtt { E 1 a s t i c N e t } ( \mathtt { w } , \lambda _ { 1 } , \lambda _ { 2 } )$ </td><td> $\begin{array} { r } { \lambda _ { 1 } \| w \| _ { 1 } + \frac { \lambda _ { 2 } } { 2 } \| w \| _ { 2 } ^ { 2 } } \end{array}$ </td></tr><tr><td rowspan="8">Constraints</td><td>Box(w, lower, upper)</td><td> $\mathcal { T } [ \mathrm { l o w e r } \le w \le \mathrm { u p p e r } ]$ </td></tr><tr><td>NonNegative(w)</td><td> $\mathcal { T } [ w \geq 0 ]$ </td></tr><tr><td>Halfspace(w, c, upper)</td><td> $\mathcal { T } [ c ^ { T } w \leq \mathrm { u p p e r } ]$ </td></tr><tr><td> $\mathtt { L i n e a r E q u a l i t y ( w , \mathtt { A } , \mathtt { b } ) }$ </td><td> $\mathcal { T } [ A w = b ]$ </td></tr><tr><td>Polyhedron(w,  ${ \textsc { A } } , { \textsc { b } } , { \textsc { c } } , { \textsc { i } } , { \textsc { u } } )$ </td><td> $\mathcal { T } [ A w = b , l \leq C w \leq u ]$ </td></tr><tr><td>L1NormBall(w, r)</td><td> $\mathcal { T } [ \| w \| _ { 1 } \leq r ]$ </td></tr><tr><td>L2NormBall(w, r)</td><td> $\mathcal { T } [ \| w \| _ { 2 } \leq r ]$ </td></tr><tr><td> $\mathtt { L I n f N o r m B a l l } \left( \mathtt { w } , \mathtt { r } \right)$ </td><td> $\mathcal { T } [ | | w | | _ { \infty } \leq \dot { r } ]$ </td></tr></table>

Table 1: Atoms available in rlaopt. Linear model losses operate on data provided via a DataLoader. $\mathbf { \bar { \mathcal { I } } } [ \cdot ]$ denotes the indicator function of the given constraint set (zero when satisfied, +∞ otherwise). ∥ · ∥ denotes the nuclear norm (sum of singular values).

w = Variable((n,), name="w")   
beta = Variable((p, K), name="beta", device="cuda")

Variables can appear in mathematical expressions involving matrix multiplication, addition, and subtraction, using standard Python operators. For instance, $\texttt { X } \texttt { \textcircled { 1 } } \texttt { w } + \texttt { b - y }$ represents the affine expression $X w + b - y$

Atoms. An atom is a function with known mathematical properties (e.g., smoothness, or the availability of a proximal operator) that serves as a building block for constructing objectives. rlaopt provides atoms for common losses, regularizers, and constraints encountered in machine learning and scientific computing. Table 1 summarizes the available atoms.

Composing objectives. Objectives are constructed by combining atoms with the + operator, mirroring the mathematical structure of the problem. Scalar multiplication via \* is also supported. For example, the bounded elastic net problem

$$
\operatorname* { m i n i m i z e } _ { w , b } \quad \frac { 1 } { 2 N } \| X w + b - y \| _ { 2 } ^ { 2 } + \lambda _ { 1 } \| w \| _ { 1 } + \frac { \lambda _ { 2 } } { 2 } \| w \| _ { 2 } ^ { 2 } \quad \mathrm { s u b j e c t } \ \mathrm { t o } \quad 0 \leq w \leq 1
$$

is specified as:

w = Variable((X.shape[1],))   
b = Variable((1,))   
obj = SumSquares(X @ w + b - y) \* (0.5 / N) \   
+ ElasticNet(w, lam1, lam2) \   
+ Box(w, 0.0, 1.0)

The resulting objective obj automatically tracks the variables and atoms that compose it.

Solvers. Once an objective has been defined, the user selects a solver and its configuration. rlaopt provides a unified solver interface with two modes of operation: a stepped mode for fine-grained control over the optimization loop, and a direct mode for one-call solving. A complete reference of all solver configuration parameters, termination criteria, and their defaults is provided in Section A. In stepped mode, the user initializes the solver state and calls step iteratively (note that the NysADMM solver is accessed via the ADMM class in the API):

```julia
solver = ADMM(obj, config=ADMMConfig())
variable_values = obj.variable_values
state = solver.init_state(variable_values)
for _ in range(num_iters):
variable_values, state = solver.step(variable_values, state)
```

In direct mode, the user simply calls solve:

solver = ADMM(obj, config=ADMMConfig())   
result = solver.solve()   
x\_sol = result.variable\_values

Both modes return the solution as a dictionary mapping variable names to their optimal values.

Linear systems. For pd linear systems, rlaopt provides a separate LinSys interface. The user specifies a matrix, right-hand side, and regularization:

```python
lin_sys = LinSys(A, b, reg=1e-2)
```

The linear system is then solved with NyströmPCG:

```python
precond_config = NystromConfig(rank_init=50, base_damping=1e-2)
pcg_config = PCGConfig(preconditioner_config=precond_config)
solver = PCG(lin_sys, config=pcg_config)
params = lin_sys.w
state = solver.init_state(params)
for _ in range(100):
params, state = solver.step(params, state)
```

## 4.2 Differentiating Through the Solver

A key feature of rlaopt is its native support for differentiating through the optimization solver. In modern machine learning pipelines, users are often interested in optimizing with respect to a parameter of the optimization problem itself—for example, tuning a regularization parameter to minimize validation error, or learning loss function parameters in an end-to-end pipeline [Agrawal et al. 2019; Blondel et al. 2022; Lu et al. 2024]. Formally, consider a parametrized optimization problem $x ^ { \star } ( \theta ) = \operatorname * { a r g m i n } _ { x } \phi ( x ; \theta )$ , where $\theta \in \mathbb { R } ^ { d }$ is an external parameter. A common use case is hyperparameter tuning, in which one seeks to minimize an outer objective that depends on the solution of the inner problem:

$$
{ \underset { \theta \in \mathbb { R } ^ { d } } { \operatorname* { m i n i m i z e } } } \quad { \mathcal { L } } ( x ^ { \star } ( \theta ) ) ,\tag{6}
$$

where $\mathcal { L }$ is a validation loss or other performance metric. For example, one may wish to tune the regularization parameter $\mu$ in a lasso problem to minimize the prediction error on a held-out validation set:

$$
\operatorname* { m i n i m i z e } _ { \mu > 0 } \quad \frac { 1 } { N _ { \mathrm { v a l } } } \| X _ { \mathrm { v a l } } x ^ { \star } ( \mu ) - y _ { \mathrm { v a l } } \| _ { 2 } ^ { 2 } , \quad \mathrm { w h e r e } \quad x ^ { \star } ( \mu ) = \arg \operatorname* { m i n } _ { x } \frac { 1 } { N _ { \mathrm { r a i n } } } \| X _ { \mathrm { t r a i n } } x - y _ { \mathrm { t r a i n } } \| _ { 2 } ^ { 2 } + \mu \| x \| _ { 1 } .\tag{7}
$$

The solvers in rlaopt are implemented in a functional manner and support automatic differentiation by backpropagation through the optimization trajectory. By default, differentiation through the solver is disabled to avoid unnecessary overhead, but users can enable it by passing detach=False to the solver constructor. For the lasso tuning problem (7), this is expressed as:

```python
config = ProxGradConfig(
eta=float(N_train / (2 * torch.linalg.matrix_norm(A_train, ord=2)**2)),
use_linesearch=False, use_acceleration=False)
def outer_objective(mu):
x = Variable(torch.zeros(p, dtype=mu.dtype), name="x")
train_obj = SumSquares(A_train @ x - b_train) * (1 / N_train) \
+ L1Norm(x, scaling=mu)
solver = ProxGrad(train_obj, config, detach=False)
values = train_obj.variable_values
state = solver.init_state(values)
for _ in range(100):
values, state = solver.step(values, state)
return ((A_val @ values["x"] - b_val)**2).mean()
mu = torch.tensor(0.2, dtype=torch.float64)
for _ in range(40):
grad, value = torch.func.grad_and_value(outer_objective)(mu)
mu = (mu - 0.05 * grad).detach()
```

PyTorch’s automatic differentiation propagates gradients through the solver iterations, yielding the derivative of the validation loss with respect to $\mu$ without requiring the user to implement custom backward passes.

## 4.3 Comparison to Disciplined Convex Programming and CVXPY

The rlaopt modeling language shares disciplined convex programming (DCP) and CVXPY’s philosophy of composable atoms and natural mathematical syntax, but differs in how problems are prepared for a solver. DCP provides rules for constructing convex problems, while CVXPY transforms these problems into standard forms (e.g., conic forms) accepted by backend solvers [Grant et al. 2006; Diamond and Boyd 2016]. However, conic reformulations can significantly increase the problem size and obscure the structure of the original problem. In contrast, rlaopt preserves the composite structure $\begin{array} { r } { f ( x ) + \sum _ { i } g _ { i } ( A _ { i } x - b _ { i } ) } \end{array}$ and works directly with gradient and proximal oracles, avoiding unnecessary reformulation. This is particularly advantageous for machine learning problems such as logistic regression, where conic reformulation introduces many additional variables [Diamandis et al. 2026].

## 5 Automatic Detection of Problem Structure

A key feature of rlaopt is its ability to automatically detect the structure of a user-specified optimization problem and decompose it into a form amenable to the chosen solver. This section describes how rlaopt performs this decomposition for proximal gradient methods and ADMM-based methods.

## 5.1 Smooth-Nonsmooth Decomposition

Recall from Section 2 that rlaopt handles composite convex optimization problems of the form

$$
\operatorname* { m i n i m i z e } _ { x \in \mathbb { R } ^ { n } } \quad f ( x ) + \sum _ { i = 1 } ^ { k } g _ { i } ( A _ { i } x - b _ { i } ) ,\tag{8}
$$

where $f$ is smooth and convex, and each $g _ { i }$ is closed, convex, and proxable. When a user constructs an objective by composing atoms (as described in Section 4), rlaopt must determine which terms constitute the smooth component $f$ and which terms constitute the nonsmooth components $g _ { i }$ , along with their associated linear operators $A _ { i }$ and offsets $b _ { i }$

Every atom in rlaopt declares two key properties: whether it is smooth (i.e., differentiable everywhere) and whether it is proxable $( \mathrm { i . e . } ,$ , its proximal operator can be evaluated efficiently). For example, SumSquares is smooth, while L1Norm and Box are nonsmooth but proxable. Given a composite objective constructed via the + operator, rlaopt partitions the terms:

• All atoms with the smooth property form the smooth component $f .$

• All atoms without the smooth property form the nonsmooth components $g _ { 1 } , \ldots , g _ { k }$

This partition is performed automatically and requires no input from the user.

## 5.2 Splitting for Proximal Gradient Methods

For proximal gradient and accelerated proximal gradient methods, rlaopt requires that each nonsmooth atom $g _ { i }$ acts directly on a variable (rather than on an affine expression of variables). In this case, the proximal operator of $g _ { i }$ can be applied directly to the variable at each iteration, yielding the standard proximal gradient update

$$
\begin{array} { r } { \boldsymbol { x } ^ { k + 1 } = \mathbf { p r o x } _ { \eta g } \big ( \boldsymbol { x } ^ { k } - \eta \nabla f ( \boldsymbol { x } ^ { k } ) \big ) , } \end{array}\tag{9}
$$

where $\eta > 0$ is the step size and $\textstyle g = \sum _ { i = 1 } ^ { k } g _ { i }$

rlaopt validates two conditions for proximal gradient splitting:

1. Each nonsmooth atom must be proxable, meaning it takes a raw variable as input.

2. The nonsmooth atoms must operate on disjoint sets of variables, so that their proximal operators can be applied independently.

If either condition is violated (for example, if a nonsmooth atom receives an affine expression as input), rlaopt raises an error indicating that the problem structure is incompatible with proximal gradient methods and suggesting the use of ADMM instead.

## 5.3 Splitting for ADMM

ADMM handles a broader class of problems than proximal gradient, as it allows nonsmooth atoms to receive affine expressions of the variables as input. When a nonsmooth atom $g _ { i }$ acts on an affine expression $A _ { i } x - b _ { i }$ rather than a raw variable, rlaopt introduces an auxiliary variable $z _ { i }$ and rewrites the problem in the consensus form

$$
\begin{array} { r l r } { \underset { x , z _ { 1 } , \ldots , z _ { k } } { \mathrm { m i n i m i z e } } } & { f ( x ) + \sum _ { i = 1 } ^ { k } g _ { i } ( z _ { i } ) } \\ { \mathrm { s u b j e c t ~ t o } } & { A _ { i } x - z _ { i } = b _ { i } , } & { i = 1 , \ldots , k . } \end{array}\tag{10}
$$

Each atom provides a decompose method that performs this transformation. Given a nonsmooth atom $g _ { i }$ with input expression $A _ { i } x - b _ { i }$ , the decomposition produces:

1. A new auxiliary variable $z _ { i }$ whose shape matches the output dimension of $A _ { i } x - b _ { i }$

2. A new atom $g _ { i } ( z _ { i } )$ that is proxable (since it now acts on a raw variable).

3. The linear operator $A _ { i }$ and offset $b _ { i } ,$ , extracted from the affine expression.

The linear operator $A _ { i }$ is represented implicitly as a LinearOperator object that computes matrix-vector products $v \mapsto A _ { i } v$ and adjoint products $v \mapsto A _ { i } ^ { T } v$ without forming A explicitly. This is important for scalability, as the linear operators arising from data matrices in machine learning can be very large.

The ADMM algorithm then alternates between approximately solving the x-subproblem (a smooth optimization problem involving f and the quadratic penalty terms), applying the proximal operators of $g _ { i }$ to update each $z _ { i } ,$ , and updating the dual variables. The x-subproblem requires solving a linear system at each iteration, which rlaopt accelerates using Nyström-based preconditioning [Frangella et al. 2023; Zhao et al. 2022].

## 5.4 Solver Selection

Given a composite objective, the choice of solver depends on the problem structure. Table 2 summarizes the condition under which each solver family is applicable.

In practice, when the user specifies a solver, rlaopt validates that the problem structure is compatible and raises an informative error if it is not. This design gives users explicit control over the solver while preventing misuse through automatic structural checks.

<table><tr><td>Solver</td><td>Applicable when</td></tr><tr><td>NyströmPCG</td><td>Problem is a pd linear system  $A x = b .$ </td></tr><tr><td>Proximal gradient</td><td>All nonsmooth atoms act on raw variables and operate on disjoint vari- able sets.</td></tr><tr><td>NysADMM</td><td>Nonsmooth atoms may act on affine expressions of variables. Automat- ically introduces auxiliary variables and linear constraints.</td></tr><tr><td>SAPPHIRE</td><td>Smooth component f has a finite-sum structure  $\begin{array} { r } { f ( x ) = \frac { 1 } { N } \sum _ { j } f _ { j } ( x ) } \end{array}$  All nonsmooth atoms act on raw variables.</td></tr></table>

Table 2: Conditions for solver applicability in rlaopt.

## 5.5 Example: Splitting in Action

We illustrate the splitting process with a concrete example. Consider the $\ell _ { 1 }$ -regularized least-squares problem where an affine transformation of the decision variable is penalized:

$$
\operatorname* { m i n i m i z e } _ { \boldsymbol { w } } \ \frac { 1 } { 2 N } \| \boldsymbol { X } \boldsymbol { w } - \boldsymbol { y } \| _ { 2 } ^ { 2 } + \lambda \| \boldsymbol { C } \boldsymbol { w } - \boldsymbol { d } \| _ { 1 } ,\tag{11}
$$

where $C \in \mathbb { R } ^ { m \times n }$ and $d \in \mathbb { R } ^ { m }$ . The user specifies this problem as:

$$
\begin{array} { l } { \texttt { w = V a r i a b l e ( ( n , ) ) } } \\ { \texttt { o b j = S u m S q u a r e s ( X \odot \texttt { w - y ) * ( 0 . 5 / \texttt { N ) + L 1 N o r m ( C \odot \texttt { w - d } , \texttt { l a m } ) } } } } \end{array}
$$

For a proximal gradient solver, this problem is not directly compatible, because the L1Norm atom receives the affine expression $\texttt { C } \texttt { \textcircled { \circ } } \texttt { w } - \texttt { d }$ rather than a raw variable. For ADMM, rlaopt automatically detects this structure and decomposes the problem. The L1Norm atom’s decompose method introduces an auxiliary variable $z \in \mathbb { R } ^ { m }$ and rewrites the problem as

$$
\begin{array} { r l } { \underset { w , z } { \mathrm { m i n i m i z e } } } & { \frac { 1 } { 2 N } \| X w - y \| _ { 2 } ^ { 2 } + \lambda \| z \| _ { 1 } } \\ { \mathrm { s u b j e c t ~ t o } } & { C w - z = d . } \end{array}
$$

The linear operator $C$ and offset d are extracted from the affine expression, and the proximal operator of $\lambda \| \cdot \| _ { 1 }$ (soft-thresholding) is applied to z at each ADMM iteration. The w-subproblem involves minimizing $\frac { 1 } { 2 N } \lVert \ddot { X } w \stackrel { \cdot \cdot } { - }$ $y \| _ { 2 } ^ { 2 } + \frac { \rho } { 2 } \| C w - z ^ { k } - d + u ^ { k } \| _ { 2 } ^ { 2 }$ , which amounts to solving a pd linear system that rlaopt accelerates with Nyström preconditioning.

## 6 Experiments

We perform an extensive evaluation of rlaopt’s solvers on large-scale problems, comparing these solvers with stateof-the-art alternatives. We find that rlaopt’s solvers are particularly effective on large, dense, ill-conditioned prob lems, where randomized preconditioning and GPU acceleration provide significant speedups. On synthetic ridge regression, the benefit of NyströmPCG depends on the conditioning of the linear system, reaching a 7× speedup over CG on the largest problem with fast spectral decay and weak regularization, while obtaining more favorable scaling compared to LSQR [Paige and Saunders 1982] and LSMR [Fong and Saunders 2011]. On bounded multinomial logistic regression, SAPPHIRE does not outperform JAXopt’s accelerated proximal gradient (APG) [Beck and Teboulle 2009] and L-BFGS-B [Byrd et al. 1995] solvers when run on GPU, but could still be valuable for certain large-scale applications. On bounded elastic net, NysADMM solves several large problems with dense data, while the competing conic solvers SCS [O’Donoghue et al. 2016] and Clarabel [Goulart and Chen 2026; Chen et al. 2026] either run out of memory or reach the time limit. However, conic solvers are much faster than NysADMM when the data is sparse. We conclude with a demonstration of the differentiable optimization capabilities of rlaopt. Although the solvers in rlaopt do not universally outperform state-of-the-art solvers, we emphasize that the suite of solvers in rlaopt can solve a much wider range of problems than any one of the solvers that we compare against. Code for the experiments is available at https://github.com/pratikrathore8/rlaopt-experiments.

Our experiments require solutions to satisfy a set of common accuracy checks, which may be more stringent than necessary for applications requiring only low-to-moderate precision. These requirements may favor interior-point methods such as Clarabel and cuClarabel over first-order methods such as SCS and the methods in rlaopt. The relative performance of these solvers likely differs at looser accuracy levels.

Experimental setup. We use rlaopt 0.1.0 by installing it from PyPI. We compare solutions using common accuracy checks, independent of each solver’s stopping criterion. Ridge regression requires a relative residual less than $1 0 ^ { \dot { - } 6 }$ ; bounded multinomial regression and bounded elastic net require stationarity less than $1 0 ^ { - 4 }$ and feasibility violation less than $1 0 ^ { - 6 }$ . We use float64 arithmetic, 64 CPU cores and 128 GiB host memory per task, with one 141 GB NVIDIA H200 for GPU tasks. The time limits are 900 seconds for ridge regression and 3600 seconds for bounded elastic net and multinomial logistic regression. Section B gives additional details.

## 6.1 Large-Scale Ridge Regression with NyströmPCG

We solve

$$
\operatorname* { m i n } _ { w \in \mathbb { R } ^ { p } } \ \frac { 1 } { 2 } \| X w - y \| _ { 2 } ^ { 2 } + \frac { \lambda } { 2 } \| w \| _ { 2 } ^ { 2 } , \qquad ( X ^ { T } X + \lambda I ) w = X ^ { T } y .\tag{12}
$$

To control the conditioning of the objective in (12), we construct

$$
X = U \Sigma _ { \alpha } V ^ { T } , \qquad \Sigma _ { \alpha } = \mathrm { d i a g } ( i ^ { - \alpha / 2 } ) _ { i = 1 } ^ { r } , \qquad y = U g / \| g \| _ { 2 } , \quad g \sim \mathcal { N } ( 0 , I _ { r } ) ,\tag{13}
$$

where $r \ = \ \operatorname* { m i n } ( n , p ) , U \ \in \ \mathbb { R } ^ { n \times r }$ and $V \in \mathbb { R } ^ { p \times r }$ have orthonormal columns. We generate these columns from structured orthogonal matrices using the SORF construction of Yu et al. [2016, Eq. (5)]. All problems have $n \geq p ,$ so the matrix $X ^ { T } \breve { X } + \lambda I$ has eigenvalues $i ^ { - \alpha } + \lambda$ and condition number $( 1 + \lambda ) / ( p ^ { - \mathbf { \bar { \alpha } } } + \lambda )$ . Thus increasing α increases the condition number, while increasing λ decreases the condition number.

We use $\alpha \in \{ 0 . 5 , 1 , 2 \} , \lambda \in \{ 1 0 ^ { - 2 } , 1 0 ^ { - 4 } , 1 0 ^ { - 6 } \}$ , and three random seeds. The dimension sweeps for X comprise square matrices with $\overset { \mathcal { \bigcup } } { n } = p \in \{ 2 ^ { 1 0 } , 2 ^ { 1 2 } , 2 ^ { 1 4 } , 2 ^ { 1 6 } \}$ ; fixed $p = 2 ^ { 1 4 }$ with $n \in \{ 2 ^ { 1 4 } , 2 ^ { 1 5 } , 2 ^ { 1 6 } \}$ ; and fixed $n = 2 ^ { 1 6 }$ with $p \dot { \in } \{ 2 ^ { 1 0 } , 2 ^ { 1 2 } , 2 ^ { 1 4 } , 2 ^ { 1 6 } \}$ . These sweeps cover eight distinct shapes for X in total. We compare rank-128 NyströmPCG, CG [Hestenes and Stiefel 1952], QR, and LSQR on CPU, and replace LSQR with cuML’s LSMR solver [Raschka et al. 2020] on GPU. CG and NyströmPCG apply the normal matrix through products with X and $X ^ { T }$ without forming $X ^ { T } X$ . Preconditioner construction for NyströmPCG is included in the measured solve time.

Fig. 1 shows the fixed-n sweep at $\lambda = 1 0 ^ { - 6 }$ . NyströmPCG on GPU is faster than on CPU by 57×–128×. The direct QR baseline encounters memory limits as the problem grows, and fails completely at $n = p = 2 ^ { 1 6 }$ . LSQR and LSMR outperform NyströmPCG when the spectrum decays slowly or the regularization is strong, but NyströmPCG is superior for ill-conditioned problems. Fig. 2 isolates the effect of preconditioning at $n = p = { \bar { 2 } } ^ { 1 6 }$ . For $\alpha = 2$ and $\lambda = \mathrm { 1 0 ^ { - 6 } }$ , NyströmPCG is 7.23× faster than CG on $\operatorname { G P U } ;$ at $\alpha = 0 . 5$ and $\lambda = 1 0 ^ { - \overleftarrow { 2 } }$ , this ratio is merely 0.81. This is intuitive: the benefits of randomized preconditioning are greatest when the spectrum decays quickly (large α) and the regularization is weak (small λ). Section C includes the remaining dimension and regularization sweeps.

## 6.2 Bounded Multinomial Logistic Regression with SAPPHIRE

We solve multinomial logistic regression:

$$
\operatorname* { m i n } _ { W \in \mathbb { R } ^ { p \times K } } - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \log \bigl ( \mathrm { s o f t m a x } ( X W ) _ { i , c _ { i } } \bigr ) \quad \mathrm { s u b j e c t t o } \quad - 1 \le W \le 1 .\tag{14}
$$

We compare SAPPHIRE with accelerated projected gradient (APG) and L-BFGS-B from JAXopt. The datasets are CIFAR-10, SVHN, Fashion-MNIST, News20, and RCV1; Table 11 lists their dimensions and preprocessing. The main comparison disables JIT compilation in JAXopt for fairness, since rlaopt is not currently JIT-enabled (Fig. 16 compares with JIT-enabled JAXopt). Both configurations use the same common accuracy checks.

Fig. 3 shows that GPU SAPPHIRE succeeds on CIFAR-10, News20, and RCV1, taking approximately 576, 613, and 490 seconds, respectively. CPU SAPPHIRE succeeds on CIFAR-10 and News20 but times out on RCV1. On SVHN and Fashion-MNIST, SAPPHIRE reaches the iteration limit. APG is faster than SAPPHIRE on all three GPU instances on which SAPPHIRE succeeds. JIT compilation further improves the JAXopt results: GPU L-BFGS-B succeeds on all five datasets, and both JIT-enabled baselines are faster than SAPPHIRE on its three successful instances. Thus this experiment demonstrates GPU acceleration of SAPPHIRE, while also identifying a performance gap relative to these deterministic baselines. Although SAPPHIRE is outperformed by JAXopt, it uses minibatch gradients for its updates and evaluates full gradients periodically to check termination. For large, high-dimensional datasets where computing a full gradient at every iteration is too expensive, SAPPHIRE could be a more practical choice.

## 6.3 Bounded Elastic Net with NysADMM

We solve

$$
\operatorname* { m i n } _ { w \in \mathbb { R } ^ { p } , b \in \mathbb { R } } \ \frac { 1 } { 2 n } \| X w + b \mathbf { 1 } - y \| _ { 2 } ^ { 2 } + \lambda _ { 1 } \| w \| _ { 1 } + \frac { \lambda _ { 2 } } { 2 } \| w \| _ { 2 } ^ { 2 } \quad \mathrm { s u b j e c t t o } \quad 0 \leq w \leq 1 ,\tag{15}
$$

![](images/f9cf7b987229874e796bed9f20bc78316f1cb77f65e5bb149cf08b928916d5ce.jpg)  
Figure 1: Ridge regression with $n = 2 ^ { 1 6 }$ and $\lambda = 1 0 ^ { - 6 }$ , varying p and spectral decay $\alpha .$ Times include solver setup and, for NyströmPCG, preconditioner construction. Each point on the plot indicates median solve time over three seeds; the error bars provide the min-max range of solve times. As the problems become more ill-conditioned (larger $\alpha ,$ smaller λ), NyströmPCG scales better than the competing methods, especially on GPU.

![](images/26dd389c0a57bed28c1ae9008e05d861f3f36a559ec9dbe6ee0ac6d560de78ed.jpg)

![](images/55ec22781be172a8a3046a92ccb05239aab57d12ef10f08f369f053ed0c68367.jpg)  
Figure 2: CG time divided by NyströmPCG time at $n = p = 2 ^ { 1 6 }$ , including preconditioner construction. Ratios above one favor preconditioning. Inequalities denote lower bounds when CG times out.

![](images/55e69b9dcb1b61ffcdceae9ee4280b2212c58c5a4c7c5cc3b2c40e19cc75c5f4.jpg)  
Figure 3: Bounded multinomial regression with JAXopt JIT compilation disabled. Each successful marker represents a solve satisfying stationarity $\leq 1 0 ^ { - 4 }$ and feasibility violation $\leq 1 0 ^ { - 6 }$ . The JAXopt methods outperform SAPPHIRE in most instances. The JIT-enabled comparison appears in Fig. 16.

with an unregularized intercept and $\lambda _ { 1 } = \lambda _ { 2 } = 0 . 1 \lambda _ { \mathrm { m a x } } ,$ , where $\lambda _ { \operatorname* { m a x } } = \| X ^ { T } ( y - { \bar { y } } \mathbf { 1 } ) \| _ { \infty } / n$ . We compare NysADMM [Zhao et al. 2022] with SCS [O’Donoghue et al. 2016] on both CPU and GPU (using both indirect and direct linear system solvers in the backend), Clarabel on CPU [Goulart and Chen 2026], and cuClarabel on GPU [Chen et al. 2026]. The datasets comprise three dense random-feature problems, acsincome-rf, yearpredictionmsd-rf, and yolanda-rf, and two sparse problems, e2006 and realsim.

Fig. 4 shows that GPU NysADMM solves acsincome-rf and yearpredictionmsd-rf in approximately 3093 and 811 seconds, respectively, while no competing solver succeeds. This advantage does not extend to every dataset. On GPU, SCS direct solves yolanda-rf in 78 seconds and realsim in 8 seconds, compared with 965 and 579 seconds for NysADMM. Moreover, SCS direct on GPU solves e2006 in 53 seconds, while NysADMM times out. The results show that NysADMM is most effective on large, dense problems, while conic solvers are superior on sparse problems, even when they are high-dimensional.

## 6.4 GPU Speedups

Fig. 5 compares CPU and GPU times for the same solver and problem for the hardware systems described in Section B. NyströmPCG is 57–128× faster on GPU than CPU for the synthetic ridge experiments, SAPPHIRE is 3.23× faster on cifar10 and 4.63× faster on news20, and its rcv1 speedup is > 7.35× because the CPU run times out. The NysADMM speedups are lower bounds: CPU NysADMM times out on all five datasets, while its four successful GPU solves imply speedups exceeding 1.16–6.22×.

## 6.5 Differentiating Through the Solver

Finally, we demonstrate rlaopt’s ability to differentiate through the optimization solver. We consider the task of tuning the regularization parameter $\mu$ in a lasso problem to minimize the prediction error on a held-out validation set, as formulated in (7).

Fig. 6 shows the convergence of the outer gradient descent loop that optimizes $\mu .$ At each outer iteration, rlaopt solves the inner lasso problem using proximal gradient and then computes the gradient of the validation loss with

![](images/509af4c3a142a660242245f6fb4f01d27110190ecde9d9eb619e81cd32c939a6.jpg)

![](images/b3f04329e59b95175cf36401b5f14c2f39ed5fc15c15416152db72605a831937.jpg)  
Figure 4: Bounded elastic net. Successful runs satisfy the common stationarity and feasibility checks. Timeout means that the displayed attempt exhausted its 3600 second budget. Host/GPU memory failures, index overflow, and worker errors are distinguished. The two CPU direct SCS worker errors are probably memory-related, but we cannot confirm this definitively. NysADMM is the only solver that succeeds on two of the large, dense random-feature problems, while SCS and Clarabel are superior on the sparse problems.

![](images/b12451a262238b859381d975f6211bdb688f7983b5c7a7add2662b4a7dcb3dcc.jpg)  
Figure 5: GPU vs. CPU speedups for rlaopt solvers. Inequalities denote lower bounds obtained when the solver times out on CPU but succeeds on GPU. Problems where the solver fails on both CPU and GPU are not included.

![](images/017c136bb6ea0fa2f7eef7f0d28b059485bc5f4ed9e250c7c9d45dc2c3482045.jpg)  
Figure 6: Differentiating through rlaopt’s proximal gradient solver enables gradient-based tuning of the lasso regu larization parameter $\mu .$ The validation loss decreases as $\mu$ is optimized.

respect to $\mu$ via PyTorch’s automatic differentiation. The validation loss decreases steadily, demonstrating that rlaopt correctly propagates gradients through the solver and enables effective hyperparameter tuning.

## 7 Conclusion

We have developed rlaopt, an open-source, PyTorch-based software package for large-scale optimization using randomized linear algebra. rlaopt addresses a significant gap between the algorithmic foundations of RandNLA and the software available to practitioners, providing GPU-enabled implementations of NyströmPCG, NysADMM, and SAPPHIRE within a unified framework. We have introduced a flexible modeling language inspired by CVXPY that allows users to specify optimization problems using natural mathematical syntax, and described how rlaopt automatically detects problem structure to perform the appropriate decomposition for each solver. Our experiments show substantial GPU speedups and demonstrate that the benefit of randomized preconditioning depends on spectral decay and regularization. GPU NysADMM solves two dense bounded elastic-net instances on which no competing conic solver succeeds in our experiments, while conic solvers are faster on sparse instances. SAPPHIRE benefits from GPU execution, although JAXopt baselines are faster on the multinomial instances that SAPPHIRE solves. We also demonstrate differentiation through the solver, enabling gradient-based hyperparameter tuning.

There are three promising directions for future work. First, we can expand the set of supported atoms and solvers, including NysNewton-CG [Rathore et al. 2024], which combines NyströmPCG with Newton’s method, and ASkotch [Rathore et al. 2026], which combines RandNLA with sketch-and-project solvers [Gower and Richtárik 2015] for linear systems. Second, we can extend rlaopt to support more scientific computing applications from RandNLA, including stochastic trace estimation [Hutchinson 1990; Meyer et al. 2021] and spectral density estimation [Lin et al. 2016; Ubaru et al. 2017; Yao et al. 2020]. Finally, we can improve performance on sparse matrices and explore a JAX-based backend with JIT compilation.

rlaopt is available at https://github.com/udellgroup/rlaopt, with documentation at https://rlaopt.readthedocs.io and version 0.1.0 on PyPI.

## Acknowledgments

PR, ZF, and MU gratefully acknowledge support from the Office of Naval Research under award N000142412306, Air Force Office of Scientific Research under award FA9550-26-1-0012, the Alfred P. Sloan Foundation, the Stanford Institute for Human-Centered Artificial Intelligence (HAI), and from IBM Research as a founding member of Stanford Institute for Human-centered Artificial Intelligence. We would also like to acknowledge the SC cluster hosted by Stanford Computer Science, which provided the resources to run the experiments in this paper.

## References

Akshay Agrawal, Brandon Amos, Shane Barratt, Stephen Boyd, Steven Diamond, and Zico Kolter. 2019. Differen tiable Convex Optimization Layers. In Advances in Neural Information Processing Systems.

Zeyuan Allen-Zhu. 2018. Katyusha: The first direct acceleration of stochastic gradient methods. Journal ofMachine Learning Research 18, 221 (2018), 1–51.

David Applegate, Mateo Diaz, Oliver Hinder, Haihao Lu, Miles Lubin, Brendan O' Donoghue, and Warren Schudy. 2021. Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient. In Advances in Neural Information Processing Systems.

Francis Bach. 2013. Sharp analysis of low-rank kernel matrix approximations. In Conference on Learning Theory.

Amir Beck and Marc Teboulle. 2009. A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. SIAM Journal on Imaging Sciences 2, 1 (2009), 183–202.

Mikhail Belkin. 2018. Approximation beats concentration? An approximation view on inference with smooth radial kernels. In Conference On Learning Theory.

Mathieu Blondel, Quentin Berthet, Marco Cuturi, Roy Frostig, Stephan Hoyer, Felipe Llinares-Lopez, Fabian Pedregosa, and Jean-Philippe Vert. 2022. Efficient and Modular Implicit Differentiation. In Advances in Neural Information Processing Systems.

Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, and Jonathan Eckstein. 2011. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends® in Machine learning 3, 1 (2011), 1–122.

Richard H. Byrd, Peihuang Lu, Jorge Nocedal, and Ciyou Zhu. 1995. A Limited Memory Algorithm for Bound Constrained Optimization. SIAM Journal on Scientific Computing 16, 5 (1995), 1190–1208.

Andrea Caponnetto and Ernesto DeVito. 2007. Optimal rates for the regularized least-squares algorithm. Foundations ofComputational Mathematics 7 (2007), 331–368.

Yuwen Chen, Danny Tse, Parth Nobel, Paul Goulart, and Stephen Boyd. 2026. CuClarabel: GPU Acceleration for a Conic Optimization Solver. ACM Trans. Math. Softw. 52, 3 (2026).

Aaron Defazio, Francis Bach, and Simon Lacoste-Julien. 2014. SAGA: A fast incremental gradient method with support for non-strongly convex composite objectives. In Advances in Neural Information Processing Systems.

Theo Diamandis, Zachary Frangella, Shipu Zhao, Bartolomeo Stellato, and Madeleine Udell. 2026. GeNIOS: an (almost) second-order operator-splitting solver for large-scale convex optimization. Mathematical Programming Computation (2026).

Steven Diamond and Stephen Boyd. 2016. CVXPY: A Python-embedded modeling language for convex optimization. Journal ofMachine Learning Research 17, 83 (2016), 1–5.

David Chin-Lung Fong and Michael A. Saunders. 2011. LSMR: An Iterative Algorithm for Sparse Least-Squares Problems. SIAM Journal on Scientific Computing 33, 5 (2011), 2950–2971.

Zachary Frangella, Pratik Rathore, Shipu Zhao, and Madeleine Udell. 2024a. PROMISE: Preconditioned Stochastic Optimization Methods by Incorporating Scalable Curvature Estimates. Journal ofMachine Learning Research 25, 346 (2024), 1–57.

Zachary Frangella, Pratik Rathore, Shipu Zhao, and Madeleine Udell. 2024b. SketchySGD: Reliable stochastic optimization via randomized curvature estimates. SIAM Journal on Mathematics of Data Science 6, 4 (2024), 1173– 1204.

Zachary Frangella, Joel Tropp, and Madeleine Udell. 2023. Randomized Nyström preconditioning. SIAM J. Matrix Anal. Appl. 44, 2 (2023), 718–752.

Daniel Gabay and Bertrand Mercier. 1976. A dual algorithm for the solution of nonlinear variational problems via finite element approximation. Computers & mathematics with applications 2, 1 (1976), 17–40.

Jacob Gardner, Geoff Pleiss, Kilian Weinberger, David Bindel, and Andrew Gordon Wilson. 2018. GPyTorch: Black box matrix-matrix Gaussian process inference with GPU acceleration. In Advances in Neural Information Process ing Systems.

Gene Golub and Charles Van Loan. 2013. Matrix Computations - 4th Edition. Johns Hopkins University Press.

Paul J. Goulart and Yuwen Chen. 2026. Clarabel: An interior-point solver for conic programs with quadratic objectives. Mathematical Programming Computation (2026).

Robert Gower and Peter Richtárik. 2015. Randomized iterative methods for linear systems. SIAM J. Matrix Anal. Appl. 36, 4 (2015), 1660–1690.

Michael Grant, Stephen Boyd, and Yinyu Ye. 2006. Disciplined Convex Programming. Springer US, 155–210.

Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp. 2011. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM review 53, 2 (2011), 217–288.

Magnus R. Hestenes and Eduard Stiefel. 1952. Methods of Conjugate Gradients for Solving Linear Systems. J. Res. Nat. Bur. Standards 49, 6 (1952), 409–436.

Michael Frank Hutchinson. 1990. A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines. Communications in Statistics - Simulation and Computation 19, 2 (1990), 433–450.

Rie Johnson and Tong Zhang. 2013. Accelerating stochastic gradient descent using predictive variance reduction. In Advances in Neural Information Processing Systems.

Diederik Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

Dmitry Kovalev, Samuel Horváth, and Peter Richtárik. 2020. Don’t jump through hoops and remove those loops: SVRG and Katyusha are better without the outer loop. In International Conference on Algorithmic Learning Theory.

Lin Lin, Yousef Saad, and Chao Yang. 2016. Approximating spectral densities of large matrices. SIAM review 58, 1 (2016), 34–65.

Haihao Lu, Zedong Peng, and Jinwen Yang. 2024. MPAX: Mathematical Programming in JAX. arXiv preprint arXiv:2412.09734 (2024).

Haihao Lu and Jinwen Yang. 2025. cuPDLP.jl: A GPU Implementation of Restarted Primal-Dual Hybrid Gradient for Linear Programming in Julia. Operations Research 73, 6 (2025), 3440–3452.

Siyuan Ma and Mikhail Belkin. 2017. Diving into the shallows: A computational perspective on large-scale shallow learning. In Advances in Neural Information Processing Systems.

Michael Mahoney. 2011. Randomized Algorithms for Matrices and Data. Foundations and Trends in Machine Learning 3, 2 (11 2011), 123–224.

Per-Gunnar Martinsson and Joel Tropp. 2020. Randomized numerical linear algebra: Foundations and algorithms. Acta Numerica 29 (2020), 403–572.

Raphael Meyer, Cameron Musco, Christopher Musco, and David Woodruff. 2021. Hutch++: Optimal Stochastic Trace Estimation. 142–155.

Arkadi Nemirovski, Anatoli Juditsky, Guanghui Lan, and Alexander Shapiro. 2009. Robust stochastic approximation approach to stochastic programming. SIAM Journal on optimization 19, 4 (2009), 1574–1609.

Yurii Nesterov and Arkadii Nemirovskii. 1994. Interior-Point Polynomial Algorithms in Convex Programming. Society for Industrial and Applied Mathematics.

Brendan O’Donoghue, Eric Chu, Neal Parikh, and Stephen Boyd. 2016. Conic optimization via operator splitting and homogeneous self-dual embedding. Journal ofOptimization Theory and Applications 169 (2016), 1042–1068.

Christopher C. Paige and Michael A. Saunders. 1982. LSQR: An Algorithm for Sparse Linear Equations and Sparse Least Squares. ACM Trans. Math. Software 8, 1 (1982), 43–71.

Neal Parikh and Stephen Boyd. 2014. Proximal algorithms. Foundations and trends® in Optimization 1, 3 (2014), 127–239.

Sebastian Raschka, Joshua Patterson, and Corey Nolet. 2020. Machine Learning in Python: Main developments and technology trends in data science, machine learning, and artificial intelligence. arXiv preprint arXiv:2002.04803 (2020).

Carl Edward Rasmussen and Christopher Williams. 2005. Gaussian processes for machine learning. The MIT Press.

Pratik Rathore, Zachary Frangella, Jiaming Yang, Michał Derezinski, and Madeleine Udell. 2026. Have ASkotch: A´ Neat Solution for Large-scale Kernel Ridge Regression. arXiv preprint arXiv:2407.10070 (2026).

Pratik Rathore, Weimu Lei, Zachary Frangella, Lu Lu, and Madeleine Udell. 2024. Challenges in Training PINNs: A Loss Landscape Perspective. In Forty-first International Conference on Machine Learning.

Herbert Robbins and Sutton Monro. 1951. A stochastic approximation method. The Annals ofMathematical Statistic (1951), 400–407.

Alessandro Rudi, Luigi Carratino, and Lorenzo Rosasco. 2017. Falkon: An optimal large scale kernel method. In Advances in Neural Information Processing Systems.

Ernest Ryu and Wotao Yin. 2022. Large-Scale Convex Optimization: Algorithms & Analyses via Monotone Operators. Cambridge University Press.

Bernhard Schölkopf and Alexander Smola. 2002. Learning with kernels: support vector machines, regularization, optimization, and beyond. MIT Press.

Bartolomeo Stellato, Goran Banjac, Paul Goulart, Alberto Bemporad, and Stephen Boyd. 2020. OSQP: an operator splitting solver for quadratic programs. Mathematical Programming Computation 12, 4 (2020), 637–672.

Jingruo Sun, Zachary Frangella, and Madeleine Udell. 2025. SAPPHIRE: Preconditioned Stochastic Variance Reduc tion for Faster Large-Scale Statistical Learning. arXiv preprint arXiv:2501.15941 (2025).

Stephen Tu, Rebecca Roelofs, Shivaram Venkataraman, and Benjamin Recht. 2016. Large scale kernel learning using block coordinate descent. arXiv preprint arXiv:1602.05310 (2016).

Shashanka Ubaru, Jie Chen, and Yousef Saad. 2017. Fast Estimation of \$tr(f(A))\$ via Stochastic Lanczos Quadrature. SIAM J. Matrix Anal. Appl. 38, 4 (2017), 1075–1099.

David Woodruff. 2014. Sketching as a tool for numerical linear algebra. Foundations and Trends® in Theoretical Computer Science 10, 1–2 (2014), 1–157.

Zhewei Yao, Amir Gholami, Kurt Keutzer, and Michael Mahoney. 2020. PyHessian: Neural Networks Through the Lens of the Hessian. In 2020 IEEE International Conference on Big Data (Big Data).

Felix X. Yu, Ananda Theertha Suresh, Krzysztof Choromanski, Daniel N. Holtmann-Rice, and Sanjiv Kumar. 2016. Orthogonal Random Features. In Advances in Neural Information Processing Systems.

Shipu Zhao, Zachary Frangella, and Madeleine Udell. 2022. NysADMM: Faster composite convex optimization via low-rank approximation. In Proceedings of the 39th International Conference on Machine Learning.

## A Solver Configuration and Termination Criteria

This appendix provides a complete reference for the configuration parameters and termination criteria of each solver in rlaopt 0.1.0. All configuration classes use sensible defaults, allowing users to get started with minimal tuning.

## A.1 NyströmPCG Configuration

Solver configuration (PCGConfig). Table 3 lists the parameters of PCGConfig. The default IdentityConfig gives ordinary CG; using NystromConfig gives NyströmPCG.

<table><tr><td>Parameter</td><td>Type</td><td>Default</td><td>Description</td></tr><tr><td>preconditioner_config</td><td></td><td>PreconditionerConfigIdentityConfig()</td><td>Preconditioner strategy</td></tr><tr><td colspan="4">Table 3: PCGConfig parameters.</td></tr></table>

Stopping criteria (PCGStoppingCriteria). Table 4 lists the termination parameters for NyströmPCG. The solver terminates when the relative residual satisfies

$$
\| r _ { k } \| _ { 2 } \leq \mathsf { t o l } \cdot \| b \| _ { 2 } ,
$$

where $r _ { k } = b - A x _ { k }$ is the residual at iteration $k ,$ or when the iteration count reaches max\_iters. For multiple right-hand sides, the relative residual condition must hold for each right-hand side.

<table><tr><td>Parameter</td><td>Type</td><td>Default</td><td>Description</td></tr><tr><td>max_iters</td><td>int</td><td>1000</td><td>Maximum number of iterations.</td></tr><tr><td>tol</td><td>float</td><td>10⁻⁶</td><td>Relative tolerance for convergence.</td></tr></table>

Table 4: PCGStoppingCriteria parameters.

## A.2 NysADMM Configuration

Solver configuration (ADMMConfig). Table 5 lists the parameters of ADMMConfig. The NysADMM solver is accessed via the ADMM class in the API. Its default preconditioner is NystromConfig(rank\_init=50, base\_damping=0.0); the remaining parameters use the defaults in Table 10.

<table><tr><td>Parameter</td><td>Type</td><td>Default</td><td>Description</td></tr><tr><td>rho</td><td>float</td><td>1.0</td><td>Augmented Lagrangian penalty parameter.</td></tr><tr><td>rho_update_factor</td><td>float</td><td>2.0</td><td>Multiplicative factor for updating ρ during primal-dual balancing.</td></tr><tr><td>rho_update_threshold</td><td>float</td><td>10.0</td><td>Threshold ratio of primal to dual residual that triggers an update to ρ.</td></tr><tr><td>rho_update_freq</td><td>int</td><td>25</td><td>Frequency (in iterations) for checking and updating ρ.</td></tr><tr><td>alpha</td><td>float</td><td>1.6</td><td>Over-relaxation parameter</td></tr><tr><td>sigma</td><td>float</td><td> $1 0 ^ { - 6 }$ </td><td>(0 &lt; α &lt; 2). Regularization for the inexact linear</td></tr><tr><td>gamma</td><td>float</td><td>1.2</td><td>system solve. Exponent controlling the decay of the linear system solve tolerance</td></tr><tr><td>preconditioner_config</td><td>Preconditioner- Config</td><td>NystromConfig</td><td>(&gt; 1). Preconditioner for the linear system subproblem. Default: Nyström with</td></tr><tr><td>preconditioner_update_freq</td><td>int</td><td>20</td><td>rank 50. Frequency (in iterations) for updating the preconditioner.</td></tr></table>

Table 5: ADMMConfig parameters.

Stopping criteria (ADMMStoppingCriteria). Table 6 lists the termination parameters for NysADMM. The primal and dual residuals at iteration k are

$$
r _ { k } ^ { \mathrm { p r i } } = A x _ { k } - z _ { k } - b , \qquad r _ { k } ^ { \mathrm { d u a l } } = \nabla f ( x _ { k } ) + \rho _ { k } A ^ { T } u _ { k } ,
$$

where $u _ { k }$ is the scaled dual variable and $\rho _ { k }$ is the penalty parameter. The solver terminates when both residuals fall below their respective tolerances,

$$
\| r _ { k } ^ { \mathrm { p r i } } \| _ { 2 } \leq \epsilon _ { k } ^ { \mathrm { p r i } } \quad \mathrm { a n d } \quad \| r _ { k } ^ { \mathrm { d u a l } } \| _ { 2 } \leq \epsilon _ { k } ^ { \mathrm { d u a l } } ,
$$

where

$$
\begin{array} { r l } & { \epsilon _ { k } ^ { \mathrm { p r i } } = \sqrt { m } \epsilon _ { \mathrm { a b s } } + \epsilon _ { \mathrm { r e l } } \operatorname* { m a x } \bigl ( \| A x _ { k } \| _ { 2 } , \| z _ { k } \| _ { 2 } , \| b \| _ { 2 } \bigr ) , } \\ & { \epsilon _ { k } ^ { \mathrm { d u a l } } = \sqrt { n } \epsilon _ { \mathrm { a b s } } + \epsilon _ { \mathrm { r e l } } \| \rho _ { k } A ^ { T } u _ { k } \| _ { 2 } . } \end{array}
$$

Here m is the number of scalar constraints in the ADMM splitting and n is the number of scalar decision variables. These tolerances follow the standard ADMM convergence criterion [Boyd et al. 2011]. The solver also terminates when the iteration count reaches max\_iters.

<table><tr><td>Parameter</td><td>Type</td><td>Default</td><td>Description</td></tr><tr><td>max_iters</td><td>int</td><td>1000</td><td>Maximum number of iterations.</td></tr><tr><td>eps_abs</td><td>float</td><td> $1 0 ^ { - 4 }$ </td><td>Absolute tolerance for primal and dual residuals.</td></tr><tr><td>eps_rel</td><td>float</td><td> $1 0 ^ { - 4 }$ </td><td>Relative tolerance for primal and dual residuals.</td></tr></table>

Table 6: ADMMStoppingCriteria parameters.

## A.3 Proximal Gradient Configuration

Solver configuration (ProxGradConfig). Table 7 lists the parameters of ProxGradConfig. When the preconditioner is not the identity, both use\_linesearch and use\_acceleration must be False. Line search and automatic stepsize updates cannot be enabled at the same time. When the objective has a nonsmooth term and the preconditioner is non-identity, subproblem\_iters controls the number of accelerated proximal gradient iterations used to approximate the scaled proximal operator. This parameter is ignored when the objective is smooth or the preconditioner is the identity.

<table><tr><td>Parameter</td><td>Type</td><td>Default</td><td>Description</td></tr><tr><td>eta</td><td>float</td><td>1.0</td><td>Step size for the gradient update.</td></tr><tr><td>use_acceleration</td><td>bool</td><td>False</td><td>Whether to use Nesterov acceleration.</td></tr><tr><td>use_linesearch</td><td>bool</td><td>True</td><td>Whether to use backtracking line search.</td></tr><tr><td>precond_config</td><td>Preconditioner- Config</td><td></td><td>IdentityConfig() Preconditioner strategy.</td></tr><tr><td>subproblem_iters</td><td>int</td><td>20</td><td>Iterations for the scaled proximal solve.</td></tr><tr><td>auto_update_stepsize</td><td>bool</td><td>False</td><td>Whether to estimate the step size from local curvature.</td></tr><tr><td>precond_update_freq</td><td>int</td><td>10</td><td>Iterations between preconditioner and automatic stepsize updates.</td></tr></table>

Table 7: ProxGradConfig parameters.

Stopping criteria (GradSolverStoppingCriteria). Table 8 lists the termination parameters shared by proximal gradient and SAPPHIRE. The proximal gradient solver computes the gradient mapping after every iteration and terminates when

$$
\frac { 1 } { \eta _ { k } } \| x _ { k } - \mathbf { p r o x } _ { \eta _ { k } g } ( x _ { k } - \eta _ { k } \nabla f ( x _ { k } ) ) \| _ { 2 } \le \epsilon _ { \mathrm { a b s } } + \epsilon _ { \mathrm { r e l } } \| x _ { k } \| _ { 2 } ,\tag{16}
$$

where $\eta _ { k }$ is the current step size, $\epsilon _ { \mathrm { a b s } }$ and $\epsilon _ { \mathrm { r e l } }$ are the absolute and relative tolerances, and $\mathbf { p r o x } _ { \eta _ { k } g }$ denotes the proximal operator of $\eta _ { k } g .$ The left-hand side of $( 1 6 )$ vanishes at the optimum; for a smooth objective, the left-hand side reduces to the gradient norm. The solver also terminates when the iteration count reaches max\_iters.

<table><tr><td>Parameter</td><td>Type</td><td>Default</td><td>Description</td></tr><tr><td>max_iters</td><td>int</td><td>1000</td><td>Maximum number of iterations (minibatch updates for SAPPHIRE).</td></tr><tr><td>eps_abs</td><td>float</td><td>10-4</td><td>Absolute tolerance for the gradient mapping norm.</td></tr><tr><td>eps_rel</td><td>float</td><td>10⁻4</td><td>Relative tolerance for the gradient mapping norm.</td></tr></table>

Table 8: GradSolverStoppingCriteria parameters for proximal gradient and SAPPHIRE.

## A.4 SAPPHIRE Configuration

Solver configuration (SapphireConfig). Table 9 lists the parameters of SapphireConfig. The SAPPHIRE solver is accessed via the Sapphire class in the API. The base method can be "saga", "svrg", or "sgd"; the default is "saga". The default preconditioner uses NystromConfig with rank\_init=10, error\_tolerance=0.1, base\_damping=0.001, and damping\_mode="adaptive". The default maximum rank resolves to 10; other parameters use the defaults in Table 10. Automatic stepsize updates are enabled by default; to use a fixed stepsize eta, set auto\_update\_stepsize=False. The subproblem\_iters parameter has the same role as in proximal gradient.

<table><tr><td>Parameter</td><td>Type</td><td>Default</td><td>Description</td></tr><tr><td>base_method</td><td>str</td><td>&quot;saga&quot;</td><td>Base stochastic method: SAGA, SVRG, or SGD.</td></tr><tr><td>eta</td><td>float</td><td>0.1</td><td>Fixed step size when automatic updates are disabled.</td></tr><tr><td>precond_config</td><td>Preconditioner- Config</td><td>NystromConfig</td><td>Nyström preconditioner with the defaults specified in the text.</td></tr><tr><td>subproblem_iters</td><td>int</td><td>20</td><td>Iterations for the scaled proximal solve.</td></tr><tr><td>auto_update_stepsize</td><td>bool</td><td>True</td><td>Whether to estimate the step size from local curvature.</td></tr><tr><td>precond_update_freq</td><td>int</td><td>2</td><td>Epochs between preconditioner and automatic stepsize updates.</td></tr><tr><td>snapshot_update_freq</td><td>int</td><td>1</td><td>Epochs between snapshot updates (SVRG only).</td></tr><tr><td>check_termination_freq</td><td>int</td><td>1</td><td>Epochs between gradient-mapping evaluations.</td></tr></table>

Table 9: SapphireConfig parameters.

The minibatch size B is specified through the model’s DataLoader, rather than SapphireConfig. For the update frequencies, the implementation converts one epoch to ⌊N/B⌋ minibatch updates, where N is the number of training samples. The preconditioner is initialized on the first update and refreshed at the configured interval; snapshot updates apply only to SVRG.

Stopping criteria (GradSolverStoppingCriteria). SAPPHIRE uses the parameters in Table 8 and the gradient mapping condition in (16). The solver evaluates this mapping using a full gradient after the first minibatch update and every check\_termination\_freq epochs thereafter. Between these evaluations, it retains the most recently computed error for assessing convergence. The solver also terminates after max\_iters minibatch updates.

## A.5 Nyström Preconditioner Configuration

Preconditioner configuration (NystromConfig). Table 10 lists the parameters of NystromConfig, which controls the randomized Nyström preconditioner used by NyströmPCG, NysADMM, SAPPHIRE, and optionally proximal gradient.

## B Experimental Details

This appendix provides details and settings for the experiments in Section 6.

<table><tr><td>Parameter</td><td>Type</td><td>Default</td><td>Description</td></tr><tr><td>rank_init</td><td>int</td><td>(required)</td><td>Initial rank of the Nyström approximation.</td></tr><tr><td>rank_max</td><td>int or None</td><td>None</td><td>Maximum allowable rank. Defaults to rank_init if not specified.</td></tr><tr><td>num_power_iters</td><td>int</td><td>10</td><td>Number of power iterations for error estimation in rank adaptation.</td></tr><tr><td>error_tolerance</td><td>float</td><td> $1 0 ^ { - 2 }$ </td><td>Error tolerance for rank adaptation.</td></tr><tr><td>base_damping</td><td>float</td><td>(required)</td><td>Base damping for the regularized low-rank approximation; required</td></tr><tr><td>damping_mode</td><td>str</td><td>&quot;adaptive&quot;</td><td>and nonnegative. &quot;adaptive&quot;: adds the smallest retained approximate eigenvalue to base_damping. &quot;non_adaptive&quot;: uses base_damping only.</td></tr></table>

Table 10: NystromConfig parameters.

Table 11: Training dimensions and feature preprocessing. The column d gives the source feature dimension and $p$ the preprocessed feature dimension. Standardization transforms the columns to have zero mean and unit variance; normalization divides each nonzero row by its Euclidean norm. K is the number of classes for multinomial logistic regression.
<table><tr><td>Dataset</td><td>n</td><td>d</td><td>p</td><td>Feature preprocessing</td></tr><tr><td>ACSIncome-rf</td><td>1664500</td><td>11</td><td>1000</td><td>Standardize; Gaussian features</td></tr><tr><td>Yolanda-rf</td><td>400000</td><td>100</td><td>1000</td><td>Standardize; Gaussian features</td></tr><tr><td>YearPredictionMSD-rf</td><td>463715</td><td>90</td><td>4367</td><td>Normalize; ReLU features</td></tr><tr><td>E2006-tfidf</td><td>16087</td><td>150360</td><td>150360</td><td>Normalize</td></tr><tr><td>Real-sim</td><td>72309</td><td>20958</td><td>20958</td><td>Normalize</td></tr><tr><td>CIFAR-10 (K = 10)</td><td>50000</td><td>3072</td><td>3072</td><td>Normalize</td></tr><tr><td>SVHN (K = 10)</td><td>73257</td><td>3072</td><td>3072</td><td>Normalize</td></tr><tr><td>Fashion-MNIST (K = 10)</td><td>60000</td><td>784</td><td>784</td><td>Standardize</td></tr><tr><td>News20 (K = 20)</td><td>15935</td><td>62061</td><td>62061</td><td>Normalize</td></tr><tr><td>RCV1 (K = 51)</td><td>15564</td><td>47236</td><td>47236</td><td>Normalize</td></tr></table>

## B.1 Datasets and Preprocessing

Table 11 summarizes the training data used in the bounded problems. We use LIBSVM for yearpredictionmsd, e2006, realsim, cifar10, svhn, news20, and rcv1, (https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/), and OpenML (https://www.openml.org/) for acsincome (data ID 43141), yolanda (42705), and fashion-mnist (40996). For fashion-mnist, we use the first 60000 examples of the 70000-example OpenML distribution as the training set. Class labels are mapped to consecutive integers. The experiments measure optimization on the training objective only.

acsincome and yolanda’s targets are centered and divided by their population standard deviation; other targets retain their source values. In particular, realsim’s source labels serve as regression targets in the bounded elastic-net experiment. Sparse features remain sparse whenever the solver supports sparse matrices, while rlaopt forces them to be dense.

Random features. We follow the random features implementation from Frangella et al. [2024a]. For output dimension $m \ = \ p ,$ draw $G \in \mathbb { R } ^ { m \times d }$ with independent standard normal entries and set $\bar { W _ { \mathbf { \lambda } } } = \bar { G ^ { \prime } } \sqrt { m }$ . Gaussian features use $\phi ( x ) = \sqrt { 2 / m } \cos ( W x / \sigma + \theta )$ with $\sigma = 1$ and independent $\theta _ { j } \sim \mathrm { U n i f } [ 0 , 2 \pi ]$ . ReLU features use $\phi ( x ) = \operatorname* { m a x } ( W x , 0 )$ , coordinatewise.

Table 12: Solver tolerances obtained from the calibration procedure. Each value applies on both backends (CPU and/or GPU) where the method is available.
<table><tr><td>Problem</td><td>Solver</td><td>Native tolerance</td></tr><tr><td>Ridge</td><td>CG, NyströmPCG</td><td> $1 0 ^ { - 6 }$ </td></tr><tr><td>Ridge</td><td>SciPy LSQR, cuML LSMR</td><td> $1 0 ^ { - 9 }$ </td></tr><tr><td>Ridge</td><td>QR</td><td>N/A</td></tr><tr><td>Bounded multinomial logistic</td><td>SAPPHIRE</td><td> $1 0 ^ { - 7 }$ </td></tr><tr><td>Bounded multinomial logistic</td><td>JAXopt APG, L-BFGS-B</td><td> $1 0 ^ { - 6 }$ </td></tr><tr><td>Bounded elastic net</td><td>NysADMM, all SCS backends</td><td> $1 0 ^ { - 7 }$ </td></tr><tr><td>Bounded elastic net</td><td>Clarabel, cuClarabel</td><td> $1 0 ^ { - 1 0 }$ </td></tr></table>

## B.2 Common Accuracy Checks

All accuracy checks are computed in float64 from the primal variables in the optimization problem: they do not use dual variables. For ridge regression, the relative residual is

$$
\frac { \| ( X ^ { T } X + \lambda I ) w - X ^ { T } y \| _ { 2 } } { \| X ^ { T } y \| _ { 2 } } \leq 1 0 ^ { - 6 } .\tag{17}
$$

For the bounded problems, the accuracy checks measure stationarity with respect to the box and primal feasibility. Let $( a ) _ { + } = \operatorname* { m a x } ( a , 0 )$ and $\delta = 1 0 ^ { - 6 }$ . For one coordinate z with gradient g and bounds [ℓ, u], define

$$
v _ { \delta } ( z , g ; \ell , u ) = \left\{ \begin{array} { l l } { ( - g ) _ { + } , } & { z \leq \ell + \delta , } \\ { ( g ) _ { + } , } & { z \geq u - \delta , } \\ { | g | , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{18}
$$

At a lower bound a nonnegative gradient is stationary, and at an upper bound a nonpositive gradient is stationary. Feasibility is checked separately, so an infeasible coordinate cannot qualify solely because of this stationarity convention.

Bounded multinomial logistic regression. Let $P = \operatorname { s o f t m a x } ( X W )$ rowwise, let Y be the one-hot label matrix, and set $G = X ^ { T } ( P - Y ) / n$ . The stationarity and feasibility measures are

$$
\begin{array} { r } { s = \underset { j , k } { \operatorname* { m a x } } v _ { \delta } ( W _ { j k } , G _ { j k } ; - 1 , 1 ) , \qquad f = \underset { j , k } { \operatorname* { m a x } } \{ ( - 1 - W _ { j k } ) _ { + } , ( W _ { j k } - 1 ) _ { + } \} . } \end{array}\tag{19}
$$

Bounded elastic net. Let $r = X w + b \mathbf { 1 } - y$ . Because $w \ge 0 ,$ , the $\ell _ { 1 }$ penalty is linear on the feasible set, so we use $g = X ^ { T } r / n + \lambda _ { 2 } w + \lambda _ { 1 } { \bf 1 }$ . The checks are

$$
s = \operatorname* { m a x } \left\{ \operatorname* { m a x } _ { j } v _ { \delta } ( w _ { j } , g _ { j } ; 0 , 1 ) , \left| \frac { \mathbf { 1 } ^ { T } r } { n } \right| \right\} , \qquad f = \operatorname* { m a x } _ { j } \{ ( - w _ { j } ) _ { + } , ( w _ { j } - 1 ) _ { + } \} .\tag{20}
$$

Both bounded problems require $s \leq 1 0 ^ { - 4 }$ and $f \leq 1 0 ^ { - 6 }$

## B.3 Calibration and Production Refinement

Tolerances are not immediately comparable between solvers because their stopping criteria and scaling conventions differ. Calibration searches for the loosest native tolerance (for a single solver) whose solutions pass the common accuracy checks (Section B.2) on every calibration instance for a solver and backend. The purpose of calibration is to reduce the number of runs on the large datasets used in the experiments, while the common accuracy checks remain the criterion for accepting a production result. Calibration for ridge regression uses shapes $( 2 ^ { 8 } , 2 ^ { 8 } ) { \overline { { , } } } ( 2 ^ { 1 2 } , 2 ^ { 1 2 } )$ , and $( 2 ^ { 1 4 } , 2 ^ { 1 2 } )$ ), and all three values of α and λ used in the experiments. The calibration for bounded elastic net and multinomial logistic regression uses synthetic standardized Gaussian data with $( n , p ) \in \{ ( 2 ^ { 8 } , 2 ^ { 5 } ) , ( 2 ^ { 1 0 } , 2 ^ { 6 } ) , ( 2 ^ { 1 2 } , 2 ^ { 8 } ) \}$ , five classes for multinomial logistic regression, and candidate tolerances $1 0 ^ { - 4 } , \dotsc , 1 0 ^ { - \operatorname { \tilde { 1 0 } } }$ . Elastic net calibration include regularization fractions 0.1 and 0.01. These runs use a 900-second limit and at most 10000 iterations. Table 12 records the solver tolerances obtained from the calibration procedure.

Refinement rule. When running the experiments in the paper, we use the calibrated tolerances as the stopping criterion for each solver. However, some solvers may terminate successfully according to their calibrated tolerance but fail the common accuracy checks. For example, cuML LSMR may find a solution with its tolerance set to $1 0 ^ { - 9 }$ but still fail to obtain a residual less than $1 0 ^ { - 6 }$ . When this occurs, we rerun the solver with a stricter tolerance. The runtimes given in the main paper are the runtimes of the first attempt that reaches its calibrated tolerance (perhaps after a rerun with a strict tolerance) and passes the common accuracy checks. If no attempt qualifies, the plot displays the final completed attempt’s failure outcome.

Ridge outcomes. Six CPU LSQR and seven GPU cuML LSMR runs completed but had relative residuals between $1 . 0 \dot { 5 } \times 1 0 ^ { - 6 }$ and $3 . 0 9 \times 1 0 ^ { - 6 }$ . All 13 passed after reducing their native tolerance from $1 0 ^ { - 9 }$ to $1 0 ^ { - 1 0 }$ . The final figures for ridge regression include these reruns.

Elastic net and multinomial logistic outcomes. The only run that reached its tolerance while failing the common checks was direct SCS on GPU for yearpredictionmsd-rf. At native tolerance $1 0 ^ { - 7 }$ it terminated after 350 iterations in 365.56 seconds, with stationarity $1 . { \dot { 1 } } 7 8 { \dot { \times } } 1 0 ^ { - 2 }$ and feasibility violation $5 . 0 0 2 \times 1 0 ^ { - 6 }$ . We subsequently tried tolerances $1 0 ^ { - 8 }$ and $1 0 ^ { - 9 }$ , but both runs exhausted their 3600-second budgets without a qualifying solution. Therefore, this entry is listed as a timeout in the final figure.

## B.4 Hardware, Timing, and Solver Settings

CPU jobs run on nodes with two AMD EPYC 7763 processors (64 physical cores per socket); GPU jobs run on a node with two AMD EPYC 9565 processors (72 physical cores per socket) and NVIDIA H200 NVL GPUs. Each task receives 64 physical CPU cores and 128 GiB host memory; GPU tasks also receive one 141 GB GPU.

Ridge timing. Data generation, initial device placement, and a single warmup are outside the solve timer. The warmup for iterative methods is capped at ten iterations; QR performs a full warmup solve. There is one measured solve for each of the three seeds, with a 900-second solve limit, a separate 900-second startup limit, and a total iteration ceiling of $2 p$ . GPU timing synchronizes device work. Warmup failures are recorded separately from measured-solve failures; in particular, the CPU QR failures at the largest square shape occur during warmup.

Bounded elastic net and multinomial logistic timing. Each result is one cold solve, without warmup, with a 3600- second solve limit, a separate 1800-second startup limit, and at most 100000 iterations. Loading, preprocessing, random feature construction, and conversion to each solver’s input representation occur outside the timer. Native solver setup, factorization, JIT compilation (when enabled), and transfers internal to a solver call are included. rlaopt uses minibatches of size 256 for multinomial logistic regression. SCS is built with CPU/GPU direct/indirect backends; GPU direct SCS uses cuDSS. Clarabel uses QDLDL on CPU and cuClarabel uses cuDSS on GPU. The conic solvers use an equivalent formulation of (15) which avoids forming a dense Gram matrix:

$$
\begin{array} { r l } { \underset { r \in \mathbb { R } ^ { n } , w \in \mathbb { R } ^ { p } , b \in \mathbb { R } } { \mathrm { m i n i m i z e } } } & { \frac { 1 } { 2 n } \| r \| _ { 2 } ^ { 2 } + \frac { \lambda _ { 2 } } { 2 } \| w \| _ { 2 } ^ { 2 } + \lambda _ { 1 } \mathbf { 1 } ^ { T } w } \\ & { \mathrm { s u b j e c t ~ t o } } & { r = X w + b \mathbf { 1 } - y , } \\ & { 0 \leq w \leq 1 . } \end{array}\tag{21}
$$

Failure categories. Timeouts, iteration limits, host-memory failures, GPU-memory failures, index overflow, and worker errors remain distinct in the figures and outcome tables. For the two CPU direct SCS worker errors on acsincome-rf and yearpredictionmsd-rf, the worker connection closed without a terminal solver result. Despite investigation, we could not determine the exact cause of these failures, but there is a good chance of a memory-related issue, since Clarabel also had memory issues on these two datasets.

Table 13 records the software environment. Clarabel and cuClarabel use source revision ffa325c89fa90b7e86b745 fa61b1dca64daf3a06.

Differentiable optimization. The differentiable optimization experiment uses $2 ^ { 9 }$ training and $2 ^ { 7 }$ validation observations with $2 ^ { 6 }$ standard Gaussian features, $2 ^ { 4 }$ nonzero coefficients drawn from $\mathcal { N } ( 0 , 1 \overline { { / 2 } } ^ { 4 } )$ , and Gaussian noise with standard deviation 0.1. Each inner solve starts at zero and uses 100 proximal gradient steps with step size $n _ { \mathrm { t r a i n } } / ( 2 \lVert X _ { \mathrm { t r a i n } } \rVert _ { 2 } ^ { 2 } )$ . We take 40 outer steps of size 0.05 starting from $\mu = 0 . 2$

## B.5 Modeling Examples

These examples show how to define each problem, initialize the solver, and run a fixed number of iterations. The ridge operator applies $X ^ { T } X$ without materializing it:

Table 13: Software versions used for the experiments.
<table><tr><td>Software</td><td>Version</td></tr><tr><td>rlaopt</td><td>0.1.0</td></tr><tr><td>PyTorch</td><td>2.13.0+cu130</td></tr><tr><td>NumPy / SciPy</td><td>2.4.2 / 1.18.1</td></tr><tr><td>cuML</td><td>26.8.0</td></tr><tr><td>JAXopt</td><td>0.8.5</td></tr><tr><td>JAX</td><td>0.11.1</td></tr><tr><td>SCS (all four backends)</td><td>3.2.11</td></tr><tr><td>Julia</td><td>1.10.12</td></tr><tr><td>Clarabel / cuClarabel source project</td><td>0.11.0 (pinned revision below)</td></tr><tr><td>CUDA.jl / CUDSS.jl</td><td>5.11.3 / 0.6.5</td></tr><tr><td>cuDSS</td><td>0.7.1</td></tr><tr><td>CUDA container base</td><td>13.0.2</td></tr></table>

```python
x_op = aslinearoperator(X)
normal_op = x_op.T @ x_op
lin_sys = LinSys(normal_op, (X.T @ y).unsqueeze(-1), reg=reg)
precond = NystromConfig(rank_init=128, rank_max=128,
base_damping=reg, damping_mode="adaptive")
solver = PCG(lin_sys, PCGConfig(preconditioner_config=precond))
params = lin_sys.w
state = solver.init_state(params)
for _ in range(100):
params, state = solver.step(params, state)
```

For bounded multinomial logistic regression, the coefficient matrix has one column per class:

```julia
beta = Variable((X.shape[1], K), dtype=torch.float64, device=device)
loader = DataLoader(Dataset(X, y, device=device), batch_size=256)
model = MultinomialRegression(beta, loader, fit_intercept=False)
obj = model + Box(beta, lower=-1.0, upper=1.0)
solver = Sapphire(obj, config=SapphireConfig())
variable_values = obj.variable_values
state = solver.init_state(variable_values)
for _ in range(200):
variable_values, state = solver.step(variable_values, state)
```

For bounded elastic net, the model has an unregularized intercept:

w = Variable((X.shape[1],), dtype=torch.float64, device=device)   
loader = DataLoader(Dataset(X, y, device=device), batch\_size=256)   
model = LinearRegression(w, loader, fit\_intercept=True)   
lambd = 0.1 \* torch.linalg.vector\_norm(X.T @ (y - y.mean()),   
ord=float("inf")) / X.shape[0]   
obj = 0.5 \* model + ElasticNet(w, l1\_scaling=lambd,   
l2\_scaling=lambd) + Box(w, 0.0, 1.0)   
solver = ADMM(obj, config=ADMMConfig())   
variable\_values = obj.variable\_values   
state = solver.init\_state(variable\_values)   
for \_ in range(20):   
variable\_values, state = solver.step(variable\_values, state)

## C Additional Experimental Results

Figs. 7 to 9 show the square ridge problems across all three regularization levels. Figs. 10 to 12 and Figs. 13 and 14 complete the fixed-p and fixed-n sweeps, respectively. Fig. 15 summarizes the effects of preconditioning across all eight shapes, and Fig. 16 provides the JIT-enabled multinomial logistic regression comparison. They use the same common accuracy checks and resource limits as the figures in the main paper.

![](images/08404c81657a1458e15afd79dc6acf29141b2a02167801be242681b2d2dd6b6a.jpg)  
Figure 7: Square ridge problems, varying $n = p ,$ at $\lambda = 1 0 ^ { - 2 } .$ . Each point on the plot indicates median solve time over three seeds; the error bars provide the min-max range of solve times.

![](images/7995a83fbb6b91d2e06f589fc88387ac78f34782ac7a90980c8a2719dee7b2cd.jpg)  
Figure 8: Square ridge problems, varying $n = p ,$ at $\lambda = 1 0 ^ { - 4 }$ . Each point on the plot indicates median solve time over three seeds; the error bars provide the min-max range of solve times.

![](images/0165909e23b6064357bf61e7a3d92f32d4d95b39376d16a5a5f833b8c235c22c.jpg)  
Figure 9: Square ridge problems, varying $n = p ,$ at $\lambda = 1 0 ^ { - 6 }$ . Each point on the plot indicates median solve time over three seeds; the error bars provide the min-max range of solve times.

![](images/3f711aa43bea4ce4190d4d775cfdb6d636081465b44991cccc514a2c8512cfff.jpg)  
Figure 10: Ridge problems with $p = 2 ^ { 1 4 }$ , varying n, at $\lambda = 1 0 ^ { - 2 }$ . Each point on the plot indicates median solve time over three seeds; the error bars provide the min-max range of solve times.

![](images/1bc79d8efc07d4a53395c2debb28722a50da5871832957bb80e38cf16c7a827d.jpg)  
Figure 11: Ridge problems with $p = 2 ^ { 1 4 } ,$ , varying n, at $\lambda = 1 0 ^ { - 4 }$ . Each point on the plot indicates median solve time over three seeds; the error bars provide the min-max range of solve times.

![](images/dc6c2c56d596d182b50dbd6bc9e7914fc8f0d1209bd24aa8d0da271870c4a8c4.jpg)  
Figure 12: Ridge problems with $p = 2 ^ { 1 4 }$ , varying n, at $\lambda = 1 0 ^ { - 6 }$ . Each point on the plot indicates median solve time over three seeds; the error bars provide the min-max range of solve times.

![](images/c5ef0696c11991cebf827e513336106f914d56a00327d2d71d7cface6ec2dd72.jpg)  
Figure 13: Ridge problems with $n = 2 ^ { 1 6 } .$ , varying p, at $\lambda = 1 0 ^ { - 2 }$ . Each point on the plot indicates median solve time over three seeds; the error bars provide the min-max range of solve times.

![](images/47620beb31f0bac1e013601c4242032b89227632cd7538e9a618f7177bf5f9c1.jpg)  
Figure 14: Ridge problems with $n = 2 ^ { 1 6 }$ , varying p, at $\lambda = 1 0 ^ { - 4 }$ . Each point on the plot indicates median solve time over three seeds; the error bars provide the min-max range of solve times.

![](images/c7934d46f0a3c01628fcf3511ce71b6f8f71d758063ba5938422f5be660e0a07.jpg)

Preconditioning Speedup Across All Matrix Shapes  
![](images/19d530a41df5ea90a556315fd3a9582b7ca2bd3deb86758e7b62c1d64d3d5860.jpg)

![](images/38f30093a38ab19619f5ec7b9393fa61f9b12f5534f84d43b0b6b17e333da793.jpg)

![](images/3a57d3d0cc5479d5b6e6803f3665b85bfe7a9e75eb1eca3e7dd1a0f6de8f7016.jpg)

![](images/7619981a693e2cf2a88e67a611f14410664bea05f8ad1206d357fb1e6df0d132.jpg)

![](images/d67146c2e1439f7b9ea160d48668de0ce33fcf899d9b55fde097cb905a9060f3.jpg)

![](images/508b8520e8e1db9df15dae7631eacad19cdaf4f007694017678cf6a756b43a75.jpg)

![](images/48899be2b4e916f76660cc0ab11572f9ef745eb1b3275365cb95a686ef39c6de.jpg)

![](images/fbb6f1cb16a72378ef4b1d5b1fa6c6c3f3ea61e6acdf3a1d58ebf27b364f46e4.jpg)

![](images/10f95162af27359738b7199db43e98da60defbdd32cffacab2f61df009036685.jpg)

![](images/b75eee5ccb086ba26de0efadef0219d8badd2346f9155700bd64fff9fd02a204.jpg)

![](images/ca06a7c3a6ddb8369af410956820fed69b49f678308c28feb9550280c8e830de.jpg)

![](images/f7e187c377d8aeb279dea88db66c88ddf0fa2eb1270caa646333406630bdfe0b.jpg)

![](images/d8b7cdfb491337e2c5df2335947c3e8a863ada6ca39a6ff9a287032d093bbecc.jpg)

![](images/97ad24af72b9c9fa2aa155b3ac8af1bc04d115dda32f577d07eb7e641fbb4025.jpg)

![](images/4bd46622ed0132f123fff5f37257ed115146c37e2607bb26ae764a02bfdaf04d.jpg)  
Figure 15: CG solve time divided by NyströmPCG solve time across all eight shapes of X, spectral decay rates $\alpha ,$ and regularization levels λ. Inequalities denote lower bounds when CG times out.

![](images/f6a9391b6592bcbc9b69395ad7ded554edd95cd77ba9e92fbdf3993d12ade31f.jpg)  
Figure 16: Bounded multinomial logistic regression with JAXopt JIT compilation enabled and included in the solve time. The JIT-enabled baselines are faster than SAPPHIRE by at least one order of magnitude.