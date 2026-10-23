# Convex Equilibrium Potentials for Monge–Ampère: Curvature Bounds and A Posteriori Evaluation

Edison Vázquez<sup>1</sup> Yingdong Lu<sup>2</sup> Tomasz Nowicki<sup>2</sup>

<sup>1</sup>Departamento de Matemáticas, CINVESTAV, México

<sup>2</sup>IBM Research, Thomas J. Watson Research Center Yorktown Heights, New York, USA

## Abstract

Equilibrium neural networks define their hidden state as the fixed point of a contraction, so their efective depth is set at evaluation time rather than by the architecture. Using them for fully nonlinear PDEs raises two issues that explicit networks avoid: second derivatives may grow with the efective depth, and the potential is only ever evaluated approximately. We study nonnegative softplus equilibrium networks with contraction factor $q < 1$ , a convex class suited to the Monge–Ampère equation. A positivity cancellation in the resolvent shows that the Hessian grows at most like $( 1 - q ) ^ { - 2 }$ , rather than the (1 − q)<sup>−3</sup> of a direct estimate, and an explicit one-neuron family shows that this exponent is sharp; third derivatives satisfy a sharp $( 1 - q ) ^ { - 3 }$ bound. Residual-based a posteriori estimates then control the state, its input derivatives and, in two dimensions, the Hessian determinant at an approximate equilibrium. Combined with interval arithmetic on complete covers of the domain, they yield rigorous uniform bounds on the Monge–Ampère residual and, through a comparison argument, on the solution error of a fixed trained network. On two manufactured Dirichlet problems on the unit disk, all six trained models are certified with residual below 0.25 and boundary error below 0.01. Adaptive stopping halves float64 evaluation time at best, whereas certified adaptive evaluation remains costlier than fixed depth.

Keywords: equilibrium neural networks; input convex neural networks; Monge–Ampère equation;   
curvature bounds; a posteriori error estimation.

## 1 Introduction

Let $\Omega \subset \mathbb { R } ^ { n }$ be a bounded domain, and let $f : \Omega \to [ 0 , \infty )$ and $g : \partial \Omega  \mathbb { R }$ be given. The Dirichlet problem for the Monge–Ampère equation asks for a convex function $u ^ { * } : \Omega  \mathbb { R }$ such that

$$
\begin{array} { r } { \operatorname* { d e t } D ^ { 2 } u ^ { * } = f \quad \mathrm { i n } \ \Omega , \qquad u ^ { * } = g \quad \mathrm { o n } \ \partial \Omega . } \end{array}\tag{1}
$$

Convexity is not a technicality here; it is part of what a solution means. Without it the equation is not elliptic, uniqueness fails, and the comparison principle on which the theory rests is lost (Gutiérrez, 2016). A numerical method must therefore either enforce convexity or recover it in the limit.

Neural networks can enforce it by construction. An input convex neural network (Amos et al., 2017) composes nonnegative linear maps with convex, nondecreasing activations, so every member of the class is convex in its input. Nyström & Vestberg (2023) train such networks on the residual of (1) and estimate their error through the stability of the equation. We keep the same principle but let the depth of the network be decided by a fixed-point iteration rather than by the architecture. For each input $x ,$ with the parameters held fixed, we iterate

$$
z _ { \ell + 1 } ( x ) = \phi ( W z _ { \ell } ( x ) + A x + b ) , \qquad z _ { 0 } ( x ) = 0 ,
$$

where $\phi ( t ) = \log ( 1 + e ^ { t } )$ is the softplus function applied componentwise. Each step is a layer of an input convex network, and all layers share the parameters (W, A, b). When $W \geq 0$ and $\| W \| _ { \infty } \leq q$ for some real number $0 < q < 1$ , the iteration is a contraction and converges to a unique equilibrium $z ^ { * } ( x )$ , which defines the potentia

$$
\begin{array} { l } { { z ^ { * } ( x ) = \phi ( W z ^ { * } ( x ) + A x + b ) , } } \\ { { \ u _ { \theta } ( x ) = w ^ { \top } z ^ { * } ( x ) + a ^ { \top } x + c , } } \end{array}\tag{2}
$$

where $\theta$ collects the parameters which will be formally defined in (4). With $w \geq 0$ , the potential $u _ { \theta }$ is convex as the limit of convex iterates.

Such an equilibrium network is a weight-tied network of unbounded depth. Its parameter count does not depend on depth, and the number of iterations can be chosen at evaluation time, point by point, to trade accuracy against cost. The contraction gap $\delta = 1 - q$ sets the efective depth: the iteration error decays like $q ^ { \ell }$ , so each additional digit of accuracy costs about $\delta ^ { - 1 }$ log 10 iterations.

This flexibility has a price, and the Monge–Ampère equation exposes it. The operator is the determinant of the Hessian, so whatever happens to second derivatives is amplified in the residual. Two things can go wrong. The first is curvature. As $q \to 1$ the network becomes efectively deeper, and a direct estimate allows its Hessian to grow like $\delta ^ { - 3 }$ , precisely in the regime where the implicit model difers most from a shallow explicit one. The second is evaluation. The equilibrium is only ever computed approximately, so a determinant evaluated at a computed point describes an approximation of the network rather than the network itself. Neither issue arises for an explicit network, and neither is settled by convexity and contraction alone.

It is useful to separate three errors. The evaluation error compares a numerical evaluation with the exact equilibrium potential at fixed parameters. The PDE residual compares the exact determinant det $D ^ { 2 } u _ { \theta }$ with the prescribed right-hand side $f .$ The solution error compares $u _ { \theta }$ with $u ^ { * }$ . Reducing the first does not eliminate the second, and a pointwise bound on the second does not control the third uniformly. A guarantee about $u _ { \theta }$ as an approximation of $u ^ { * }$ must pass through all three.

This paper follows that chain. We first show that curvature is better behaved than the direct estimate suggests. A positivity cancellation,

$$
( I - \mathrm { d i a g } ( \phi ^ { \prime } ( s ) ) W ) ^ { - 1 } \phi ^ { \prime \prime } ( s ) \le { \bf 1 } ,\tag{3}
$$

reflects the fact that a softplus unit with slope close to one has small intrinsic curvature. It yields a Hessian bound of order $\delta ^ { - 2 }$ , and an explicit one-neuron family shows that this exponent cannot be improved. Third derivatives obey a sharp bound of order $\delta ^ { - 3 }$ , and positive weighted norms extend the analysis to recurrent matrices with spectral radius below one. We then turn to evaluation. Residual-based a posteriori estimates bound the error in the state, in its first and second input derivatives and, in two dimensions, in the Hessian determinant, at any approximate equilibrium. They apply whether the derivatives come from a finite unroll or from approximate linear solves, and they account for the perturbation of the coeficients themselves caused by the state error. Finally, we pass from points to the whole domain. Interval arithmetic on complete covers of the disk and its boundary gives rigorous uniform bounds on the Monge–Ampère residual, and a comparison argument converts them into a bound on the solution error.

We test this chain on two manufactured Dirichlet problems on the unit disk, with three trained networks each. All six networks are certified with residual below 0.25 and boundary error below 0.01. The experiments also measure how much adaptive evaluation saves and how much certification costs.

## 1.1 Related work

Convex neural representations. Input convex neural networks (ICNNs) enforce convexity through nonnegative hidden weights and convex, nondecreasing activations (Amos et al., 2017). Their construction follows standard composition rules for convex functions (Boyd & Vandenberghe, 2004). Convex neural potentials also arise in optimal transport: Makkuva et al. (2020) learn transport maps through gradients of ICNN potentials. These applications motivate control of derivatives as well as function values. Our model retains the positivity structure of an ICNN but defines its hidden state implicitly. Convexity follows from the established ICNN argument applied to successive fixed-point iterates; the curvature estimates below address the additional efect of the equilibrium resolvent.

Equilibrium models and implicit diferentiation. Deep equilibrium models replace a finite stack of layers by a fixed-point equation and diferentiate through its equilibrium (Bai et al., 2019). Monotone operator equilibrium networks provide a diferent route to well-posedness and convergent equilibrium solvers (Winston & Kolter, 2020). Non-Euclidean contraction analysis gives further conditions for existence, uniqueness and robustness of implicit networks (Jafarpour et al., 2021). Well-posedness conditions of Perron–Frobenius type for implicit models are given by El Ghaoui et al. (2021). Our nonnegative recurrent matrices admit a particularly simple positive weighted-norm formulation. For input derivatives, Huang et al. (2024, Proposition 1, equations (8)–(9)) derive first- and second-order identities for a general contractive fixed-point map. The identities used here specialize that analysis. Our a posteriori estimates additionally account for the perturbation of their coeficients when the equilibrium state is computed only approximately.

Monge–Ampère approximation. The analytical theory of the Monge–Ampère equation treats convexity as part of the solution concept and supplies comparison and stability tools (Gutiérrez, 2016). In numerical analysis, Froese & Oberman (2011) construct monotone finite-diference schemes converging to viscosity solutions. Neural approaches provide a diferent approximation framework. Physics-informed neural networks and the deep Galerkin method train neural representations using diferential equations and boundary or initial conditions (Raissi et al., 2019; Sirignano & Spiliopoulos, 2018). For the Dirichlet Monge–Ampère problem specifically, Nyström & Vestberg (2023) study ICNN approximations, derivative estimates and stability. Our experiments use a loss of the same form, with an equilibrium representation.

Curvature and error certification. Entesari et al. (2024) derive compositional curvature bounds for explicit neural networks, and Nyström & Vestberg (2023, Lemma 6.3) bound derivatives up to third order for explicit softplus ICNNs. For neural PDE approximations, Mishra & Molinaro (2023) relate generalization error to training error and sampling through stability properties of the underlying equation, while Eiras et al. (2024) develop regional error certification for physics-informed neural networks. We distinguish the error of evaluating an implicit potential from its PDE residual and its error relative to the PDE solution. The positive softplus resolvent yields the cancellation responsible for the inverse-quadratic curvature bound; residual estimates then propagate state and derivative errors to the determinant. Ball arithmetic (Johansson, 2017) encloses rounding efects in the regional implementation.

## 1.2 Organization and scope

Sections 2–3 introduce the equilibrium representation and the derivative identities needed for the analysis. Section 4 proves the curvature cancellation, the inverse-quadratic bound and its optimality, and states the sharp third-derivative extension. Section 5 develops the approximate-state certificate; Section 6 extends it to the full domain and to solution-error bounds. Sections 7–8 describe adaptive evaluation and the six-model study. The appendices contain supporting proofs, the weighted-norm and local-envelope extensions, and finite-unroll derivative recurrences. The guarantees concern evaluation of fixed trained models; they do not establish convergence of training or superiority to explicit ICNNs.

## 2 Assumptions, notation and the equilibrium

The representation analysis applies to $x \in \mathbb { R } ^ { n }$ with m hidden units representing $z ^ { * }$ . The determinant certificate and the experiments are stated for $n = 2$ . Parameters have sizes

$$
W \in \mathbb R ^ { m \times m } , \quad A \in \mathbb R ^ { m \times n } , \quad b , w \in \mathbb R ^ { m } , \quad a \in \mathbb R ^ { n } , \quad c \in \mathbb R .\tag{4}
$$

Throughout, parameters are fixed when diferentiating in $x ,$ and there exists a real number $q \in ( 0 , 1 )$ such that

$$
W \geq 0 , \qquad w \geq 0 , \qquad \| W \| _ { \infty } \leq q .\tag{H}
$$

Inequalities between vectors or matrices are entrywise unless stated otherwise. The norm $\| M \| _ { \infty }$ is the maximum absolute row sum; $\| M \| _ { 2 }$ is the spectral norm. Set

$$
\delta = 1 - q , \qquad K _ { A } : = \operatorname* { m a x } _ { 1 \leq r \leq m } \left\| A _ { r , : } \right\| _ { 2 } .\tag{5}
$$

The input matrix and afine readout are unrestricted in sign. The Hadamard product (or the entrywise product) is denoted by ⊙, and powers of vectors are also defined entrywisely.

Softplus ϕ is smooth, convex and 1-Lipschitz, with $0 < \phi ^ { \prime } < 1$ and $0 < \phi ^ { \prime \prime } \leq 1 / 4$ . Under (H), the state map (2) is a contraction. It has a unique equilibrium, and the ICNN composition argument implies convexity of the equilibrium potential $u \theta$ . For an approximate state $\hat { z } ,$ we have,

$$
\begin{array} { r } { \| z ^ { * } ( x ) - \hat { z } \| _ { \infty } \leq \rho _ { z } / \delta , \qquad \rho _ { z } = \| \hat { z } - \phi ( W \hat { z } + A x + b ) \| _ { \infty } . } \end{array}
$$

The precise statements and proofs are given in Appendix A.

## 3 Implicit derivatives and positive Hessian structure

At the exact equilibrium define

$$
\begin{array} { r l r } { s = W z ^ { * } + A x + b , } & { { } \gamma = \phi ^ { \prime } ( s ) \odot ( \mathbf { 1 } - \phi ^ { \prime } ( s ) ) , \quad \Psi = \mathrm { d i a g } ( \phi ^ { \prime } ( s ) ) , } & { } \\ { B = I - \Psi W , } & { { } \quad S = B ^ { - 1 } , \qquad V = ( I - W \Psi ) ^ { - 1 } A . } & { } \end{array}\tag{6}
$$

Write $A _ { : , i }$ for column i of A, $\begin{array} { r } { J _ { i } = \frac { \partial } { \partial x _ { i } } z ^ { * } ( x ) , K _ { i j } = \frac { \partial ^ { 2 } } { \partial x _ { i } \partial x _ { j } } z ^ { * } ( x ) } \end{array}$ and $J = [ J _ { 1 } , \ldots , J _ { n } ]$ . Note that the order of ΨW and WΨ in (6) matters.

The equilibrium is smooth, and implicit diferentiation gives

$$
B J _ { i } = \Psi A _ { : , i } , \qquad B K _ { i j } = \gamma \odot v _ { i } \odot v _ { j } , \qquad v _ { i } = A _ { : , i } + W J _ { i } = V _ { : , i } .
$$

The resolvent satisfies $S \geq I$ and $\begin{array} { r } { \| S \| _ { \infty } \le \delta ^ { - 1 } } \end{array}$ . In particular, with $\beta = ( S ^ { \top } w ) \odot \gamma$

$$
H : = D ^ { 2 } u _ { \theta } = V ^ { \top } \mathrm { d i a g } ( \beta ) V \succeq 0 , \qquad D ^ { 2 } z ^ { * } [ h , h ] = S [ \gamma \odot ( V h ) ^ { 2 } ] .
$$

These identities expose the positive structure used in the curvature argument. Appendix A gives the derivative derivations, the strict-convexity criterion and the two-dimensional determinant identity.

## 4 A curvature bound with an optimal gap exponent

Lemma 4.1 (Resolvent–curvature cancellation). Under assumption (H),

$$
0 \leq S \gamma \leq { \bf 1 } .\tag{7}
$$

Proof. Since $W \mathbf { 1 } \leq q \mathbf { 1 } \leq \mathbf { 1 }$ , we have

$$
0 \leq \gamma = \phi ^ { \prime } ( s ) \odot ( \mathbf { 1 } - \phi ^ { \prime } ( s ) ) \leq \mathbf { 1 } - \phi ^ { \prime } ( s ) \leq \mathbf { 1 } - \Psi W \mathbf { 1 } = B \mathbf { 1 } .
$$

Multiplication by the nonnegative matrix $S$ preserves entrywise inequalities and gives (7).

Theorem 4.2 (Uniform curvature bound). For every $x \in \mathbb { R } ^ { n }$ under assumption (H),

$$
\left\| D ^ { 2 } u _ { \theta } ( x ) \right\| _ { 2 } \leq C _ { 2 } , \qquad C _ { 2 } = \| w \| _ { 1 } K _ { A } ^ { 2 } \operatorname* { m i n } \left\{ \delta ^ { - 2 } , { \frac { 1 } { 4 \delta ^ { 3 } } } \right\} .\tag{8}
$$

The bound is uniform in the input and in the bias b. Width enters only through the displayed parameter norms.

Proof. For a Euclidean unit vector $h , \| A h \| _ { \infty } \leq K _ { A }$ . Since $\begin{array} { r } { \left. ( I - W \Psi ) ^ { - 1 } \right. _ { \infty } \leq \delta ^ { - 1 } } \end{array}$

$$
\begin{array} { r } { \| V h \| _ { \infty } \leq K _ { A } / \delta . } \end{array}
$$

Positivity and Lemma 4.1 imply

$$
0 \leq D ^ { 2 } z ^ { * } [ h , h ] = S [ \gamma \odot ( V h ) ^ { 2 } ] \leq \frac { K _ { A } ^ { 2 } } { \delta ^ { 2 } } S \gamma \leq \frac { K _ { A } ^ { 2 } } { \delta ^ { 2 } } \mathbf { 1 } .
$$

Multiply by $w ^ { \top }$ and take the supremum over unit h. Because $H \succeq 0$ , this supremum equals $\| H \| _ { 2 }$ . Alternatively, use $\begin{array} { r } { \| S \| _ { \infty } \le \delta ^ { - 1 } } \end{array}$ and $\| \gamma \| _ { \infty } \leq 1 / 4$ separately to obtain $\| H \| _ { 2 } \leq \| w \| _ { 1 } K _ { A } ^ { 2 } / ( 4 \delta ^ { 3 } )$ Taking the smaller estimate proves (8). □

Remark 4.3. The cancellation avoids multiplying a worst-case resolvent norm by a worst-case activation curvature independently. A softplus unit with slope close to one has small intrinsic curvature. The quadratic estimate is the smaller one precisely when $\delta < 1 / 4$ , or $q > 3 / 4$ . For $q \leq 3 / 4$ the direct cubic estimate may be tighter. The norms $\lVert \boldsymbol { w } \rVert _ { 1 }$ and $K _ { A }$ must remain in the statement: contraction alone does not uniformly bound the Hessian over arbitrary readouts and input weights.

Theorem 4.4 (Optimality of the exponent). For $0 < \varepsilon < 1$ , take one input and one hidden unit with

$$
W = q = 1 - \varepsilon , \quad A = w = 1 , \quad a = c = 0 , \quad b _ { \varepsilon } = \log ( 1 - \varepsilon ) - \varepsilon \log \varepsilon .
$$

$$
A t \ x = 0 ,
$$

$$
z ^ { * } ( 0 ) = - \log \varepsilon , \qquad u _ { \varepsilon } ^ { \prime \prime } ( 0 ) = \frac { 1 - \varepsilon } { \varepsilon ^ { 2 } ( 2 - \varepsilon ) ^ { 3 } } , \qquad \operatorname* { l i m } _ { \varepsilon \downarrow 0 } \varepsilon ^ { 2 } u _ { \varepsilon } ^ { \prime \prime } ( 0 ) = \frac { 1 } { 8 } .\tag{9}
$$

Thus no bound $\big \| D ^ { 2 } u _ { \theta } \big \| _ { 2 } \leq C \| w \| _ { 1 } K _ { A } ^ { 2 } ( 1 - q ) ^ { - p }$ with $p < 2$ and a constant C independent of the admissible parameters holds for this class.

Proof. With $z = - \log \varepsilon .$ , the preactivation is

$$
s = ( 1 - \varepsilon ) ( - \log \varepsilon ) + b _ { \varepsilon } = \log \frac { 1 - \varepsilon } { \varepsilon } .
$$

Then $\phi ( s ) = - \log \varepsilon = z$ and $\phi ^ { \prime } ( s ) = 1 - \varepsilon$ . Uniqueness identifies this state with $z ^ { * } ( 0 )$ . For a scalar input, (A.2) gives $z ^ { \prime } = \phi ^ { \prime } ( s ) / ( 1 - q \phi ^ { \prime } ( s ) )$ , hence $s ^ { \prime } = q z ^ { \prime } + 1 = 1 / ( 1 - q \phi ^ { \prime } ( s ) )$ ). Equation $\left( \mathrm { { A . 3 } } \right)$ yields

$$
z ^ { \prime \prime } = { \frac { \phi ^ { \prime } ( s ) ( 1 - \phi ^ { \prime } ( s ) ) } { ( 1 - q \phi ^ { \prime } ( s ) ) ^ { 3 } } } .
$$

Substitute $q = \phi ^ { \prime } ( s ) = 1 - \varepsilon$ and $1 - q \phi ^ { \prime } ( s ) = \varepsilon ( 2 - \varepsilon )$ to obtain (9). If a bound with exponent $p < 2$ held, then $\varepsilon ^ { 2 } u _ { \varepsilon } ^ { \prime \prime } ( 0 ) \leq C \varepsilon ^ { 2 - p } \to 0$ , contradicting the limit $1 / 8$ □

Remark 4.5 (Two-dimensional embedding and meaning of sharpness). Two independent copies with $W = ( 1 - \varepsilon ) I _ { 2 } , A = I _ { 2 }$ and $w = ( 1 , 1 ) ^ { \top }$ give a strictly convex two-dimensional potential with

$$
D ^ { 2 } u _ { \varepsilon } ( 0 ) = u _ { \varepsilon } ^ { \prime \prime } ( 0 ) I _ { 2 } , \qquad \operatorname * { d e t } D ^ { 2 } u _ { \varepsilon } ( 0 ) \sim \frac { 1 } { 6 4 \varepsilon ^ { 4 } } .
$$

Thus the curvature scaling also occurs within the two-dimensional class. The family varies parameters with ε and is not a training trajectory. Theorem 4.4 proves optimality of the exponent; it does not prove optimality of the constant in Theorem 4.2.

![](images/70ffc8b3dfbb8386ad677516c49dcd39edd857b0a51c5ca4b07f72cba058289f.jpg)

![](images/1cd13733de4a0a8b9688cce5053c41e6340c13ba721c5c37544daad79dc73e39.jpg)  
Figure 1: Analytic sharpness family from Theorem 4.4. Left: exact curvature and the two upper bounds for $A = w = 1$ . Right: the rescaled curvature tends to $1 / 8$ as the contraction gap decreases. Curves are evaluations of the displayed closed forms.

## 4.1 Sharp third-derivative bound

For a scalar trilinear form we use the norm obtained by taking the supremum over three independent Euclidean unit vectors.

Theorem 4.6 (Third derivatives). Under assumption (H), uniformly in $x _ { i }$

$$
\left\| D ^ { 3 } u _ { \theta } ( x ) \right\| \le ( 1 + 3 q ) \| w \| _ { 1 } K _ { A } ^ { 3 } \delta ^ { - 3 } .
$$

The inverse-gap exponent three is optimal over this class.

The proof is given in Appendix B.

Extension to weighted norms. The positive cancellation also gives a curvature estimate in a weighted norm whenever a positive vector η satisfies $W \eta \leq q \eta$ . Such a vector exists for every $\rho ( W ) < q < 1$ . Appendix C states and proves the extension, including the normalization and the spectral-radius endpoint caveat. The numerical experiments use the unweighted row norm.

## 5 A posteriori evaluation from an approximate state

Fix $\boldsymbol { x } \in \mathbb { R } ^ { 2 }$ and parameters satisfying (H). We consider numerical approximations to the equilibrium state and its first and second derivatives with respect to the input:

$$
\hat { z } \approx z ^ { * } ( x ) , \qquad \hat { J } _ { i } \approx \partial _ { x _ { i } } z ^ { * } ( x ) , \qquad \hat { K } _ { i j } \approx \partial _ { x _ { i } } \partial _ { x _ { j } } z ^ { * } ( x ) , \qquad i , j \in \{ 1 , 2 \} .
$$

Here, ${ \hat { J } } _ { i }$ denotes the ith column of an approximate Jacobian and approximates the exact derivative $J _ { i } = \partial _ { x _ { i } } z ^ { * } ( x )$ defined earlier. We assume $\hat { K } _ { 1 2 } = \hat { K } _ { 2 1 }$

For example, zˆ may be obtained by terminating the equilibrium iteration after finitely many steps, while ${ \hat { J } } _ { i }$ and $\hat { K } _ { i j }$ may be obtained by approximately solving the linear systems arising from implicit diferentiation. The estimates below concern numerical errors in evaluating a fixed neural representation, rather than the approximation error between $u _ { \theta }$ and the PDE solution $u ^ { * }$ . The approximate quantities need not solve their defining equations exactly and need not have been produced by the same algorithm.

Define

$$
{ \hat { s } } = W { \hat { z } } + A x + b , \quad { \hat { \Psi } } = \mathrm { d i a g } ( \phi ^ { \prime } ( { \hat { s } } ) ) , \quad { \hat { \gamma } } = \phi ^ { \prime \prime } ( { \hat { s } } ) , \quad { \hat { B } } = I - { \hat { \Psi } } W ,
$$

$$
\hat { v } _ { i } = A _ { : , i } + W \hat { J } _ { i } , \qquad U _ { i } = \left. \hat { v } _ { i } \right. _ { \infty } .\tag{10}
$$

The residuals, all computed at this same approximate state, are

$$
\rho _ { z } = \| \hat { z } - \phi ( \hat { s } ) \| _ { \infty } ,\tag{11}
$$

$$
\rho _ { i } = \left. \hat { B } \hat { J } _ { i } - \hat { \Psi } A _ { : , i } \right. _ { \infty } ,\tag{12}
$$

$$
{ \rho } _ { i j } = \left. \hat { B } \hat { K } _ { i j } - \hat { \gamma } \odot \hat { v } _ { i } \odot \hat { v } _ { j } \right. _ { \infty } .\tag{13}
$$

Introduce the following nonnegative quantities:

$$
E _ { z } = \rho _ { z } / \delta ,
$$

$$
\alpha = q E _ { z } / 4 ,\tag{14}
$$

$$
E _ { i } = ( \rho _ { i } + \alpha U _ { i } ) / \delta ,
$$

$$
\Delta _ { i } = q E _ { i } ,\tag{15}
$$

$$
\begin{array} { r } { Q _ { i j } = \alpha U _ { i } U _ { j } + \frac { 1 } { 4 } ( \Delta _ { i } U _ { j } + U _ { i } \Delta _ { j } + \Delta _ { i } \Delta _ { j } ) , } \end{array}\tag{16}
$$

$$
E _ { i j } = \frac { { { \rho } _ { i j } } + { { Q } _ { i j } } + \alpha q \left\| \hat { K } _ { i j } \right\| _ { \infty } } { \delta } .\tag{17}
$$

All these expressions use the approximate quantities, parameters and a valid contraction bound $q .$

Theorem 5.1 (State-aware derivative certificate). Under the preceding assumptions,

$$
\begin{array} { r } { \| \boldsymbol { z } ^ { * } - \boldsymbol { \hat { z } } \| _ { \infty } \leq E _ { z } , \quad \left\| \boldsymbol { J } _ { i } - \boldsymbol { \hat { J } } _ { i } \right\| _ { \infty } \leq E _ { i } , \quad \left\| \boldsymbol { K } _ { i j } - \boldsymbol { \hat { K } } _ { i j } \right\| _ { \infty } \leq E _ { i j } . } \end{array}\tag{18}
$$

For $\hat { H } _ { i j } = w ^ { \top } \hat { K } _ { i j }$ ，

$$
\begin{array} { r } { \left\| \boldsymbol H - \hat { \boldsymbol H } \right\| _ { 2 } \le E _ { H } , \qquad E _ { H } = \| \boldsymbol w \| _ { 1 } \sqrt { E _ { 1 1 } ^ { 2 } + 2 E _ { 1 2 } ^ { 2 } + E _ { 2 2 } ^ { 2 } } . } \end{array}\tag{19}
$$

The value approximation $\hat { u } = w ^ { \top } \hat { z } + a ^ { \top } x + c$ satisfies $| u _ { \theta } ( x ) - \hat { u } | \leq \| w \| _ { 1 } E _ { z }$

The proof is given in Appendix E.

Corollary 5.2 (Determinant, residual and convexity bounds). In two dimensions set

$$
E _ { \mathrm { d e t } } = 2 \left\| \hat { H } \right\| _ { 2 } E _ { H } + E _ { H } ^ { 2 } .\tag{20}
$$

Then | det H−det $\hat { H } | \leq E _ { \mathrm { d e t } }$ . For any prescribed $f ( x )$ , let $\hat { r } = |$ det ${ \hat { H } } - f ( x ) |$ and r = | det H −f(x)|. One has

$$
\operatorname* { m a x } \{ 0 , \hat { r } - E _ { \mathrm { d e t } } \} \leq r \leq \hat { r } + E _ { \mathrm { d e t } } ,\tag{21}
$$

and

$$
\lambda _ { \operatorname* { m i n } } ( H ) \geq \lambda _ { \operatorname* { m i n } } ( { \hat { H } } ) - E _ { H } .\tag{22}
$$

In particular, a strictly positive right-hand side in (22) certifies positive definiteness at that input.

The proof is given in Appendix E.

Why state error cannot be omitted. Even exact solves of the systems constructed with $\hat { B } .$ Ψ<sup>ˆ</sup> and $\hat { \gamma }$ only make $\rho _ { i }$ and $\rho _ { i j }$ vanish. They do not replace those coeficients by the exact-state coeficients. The α terms in (15)–(17) account for this discrepancy. If $q = 0$ , the derivative coeficients are independent of the state and these terms correctly vanish.

Arithmetic scope. Theorem 5.1 and Corollary 5.2 are inequalities for exact real quantities. Treating finite-precision residuals as exact upper bounds is insuficient for a validated machine certificate. Such an implementation must enclose residual evaluation, nonlinear functions, parameter norms, linear algebra and determinant/eigenvalue computations with outward rounding or explicit error bounds. The regional implementation described next supplies such enclosures for the saved model. Float64 diagnostic checks remain distinct from these validated bounds.

## 6 Local bounds, complete coverage and solution error

We extend the pointwise certificate to boxes and combine local guarantees over a complete cover. An alternative analytical construction based on local derivative envelopes is developed in Appendix D; it is separate from the interval-residual method evaluated here.

## 6.1 A regional residual route with controlled arithmetic

The regional method uses the following extension of the approximate-state certificate. It does not require third derivatives or the analytical envelopes of Appendix D.

Proposition 6.1 (Uniform certificate on a box). Fix a box $X \subset \mathbb { R } ^ { 2 }$ and constant candidates $\hat { z } , \hat { J } _ { i } , \hat { K } _ { i j }$ , with symmetric $\hat { K }$ . Suppose certified upper bounds for the residual norms in $( 1 1 ) - ( 1 3 )$ hold for every $x \in X$ , with the coeficients recomputed at $W \hat { z } + A x + b$ . Substitution of these upper bounds and outward enclosures of all nonnegative constants into (14)–(20) gives uniform derivative and determinant error bounds on X. In particular, if $F ( X )$ encloses $f ( X )$ and $\tilde { H } _ { i j } = w ^ { \top } \hat { K } _ { i j }$ , then

$$
\operatorname* { s u p } _ { x \in X } \vert \operatorname* { d e t } D ^ { 2 } u _ { \theta } ( x ) - f ( x ) \vert \leq \operatorname* { s u p } _ { y \in F ( X ) } \vert \operatorname* { d e t } \tilde { H } - y \vert + E _ { \operatorname* { d e t } } ( X ) .
$$

To preserve rigor in finite-precision arithmetic, the determinant det $\tilde { H }$ must be evaluated as an interval guaranteed to contain its exact value. The right-hand side must then be bounded from above using outward-rounded interval arithmetic, so that rounding errors cannot lead to an underestimate of the residual bound.

The proof is given in Appendix F.

Rigorous evaluation uses Arb ball arithmetic with 128-bit midpoint precision (Johansson, 2017). Efective binary64 parameters are imported as exact dyadic rationals and define the frozen real model being certified. Positivity and $q < 1$ are rechecked. This interpretation does not certify the rounding history of optimization. Nonlinear functions, residuals, coeficient norms and determinants are enclosed; scalar exports to binary64 are rounded outward. The implementation uses $\left\| \widetilde { H } \right\| _ { F }$ as a safe upper bound for its spectral norm. A reported floating determinant receives a further enclosure of its diference from the exact candidate determinant.

The initial state box $Z = \hat { z } + [ - E _ { z } , E _ { z } ] ^ { m }$ contains every $z ^ { * } ( x )$ for $x \in X$ . Intersections $Z \gets Z \cap \phi ( W Z + A X + b )$ preserve that property. Verified interval linear solves (Rump, 2010) for first and second derivatives may tighten the result; if a solve cannot be verified, the residual bound is retained. This is the implemented local-box method.

Corollary 6.2 (Complete disk and boundary cover). Let $\{ X _ { j } \}$ cover the closed unit disk, with residual bounds $R _ { j }$ from Proposition 6.1. Let $\{ I _ { k } \}$ cover [0, 2π], and suppose $B _ { k }$ bounds $| u _ { \theta } ( \cos t , \sin t ) -$ g(cos t, sin t)| for every $t \in I _ { k }$ . Then

$$
\left\| \operatorname* { d e t } D ^ { 2 } u _ { \theta } - f \right\| _ { L ^ { \infty } ( B _ { 1 } ) } \le R : = \operatorname* { m a x } _ { j } R _ { j } , \qquad \| u _ { \theta } - g \| _ { L ^ { \infty } ( \partial B _ { 1 } ) } \le B _ { \partial } : = \operatorname* { m a x } _ { k } B _ { k } .
$$

The proof is given in Appendix F.

The domain partition subdivides $[ - 1 , 1 ] ^ { 2 }$ into exact dyadic boxes and discards one only if an exact squared-distance test puts the entire closed box strictly outside the disk. Circle-touching boxes are retained. Boundary arcs use interval $\pi ,$ sine and cosine. Exact partition audits check coverage. At a work limit, coarse finite bounds are retained, not discarded. Coverage and tolerance attainment are therefore checked independently. A complete cover can yield a valid but uninformative upper bound.

## 6.2 From a uniform residual to a solution bound

The stability estimate of Nyström & Vestberg (2023, Proposition 6.1), under its convexity, continuity and data hypotheses, gives

$$
\lVert u _ { \theta } - u ^ { * } \rVert _ { L ^ { \infty } ( \Omega ) } \leq B _ { \partial } + C _ { n } \mathrm { d i a m } ( \Omega ) | \Omega | ^ { 1 / n } R ^ { 1 / n } .
$$

Here the network is viewed as the exact convex solution with its own right-hand side det $D ^ { 2 } u _ { \theta }$ and its own boundary trace. Smoothness and nonnegativity of the Hessian ensure that its classical determinant is its Aleksandrov density. The constant $C _ { n }$ depends only on the dimension but is not numerically explicit; on the unit disk a direct barrier gives an explicit coeficient. We give the classical smooth statement, suficient for P1 and P2; extension to nonsmooth Aleksandrov targets requires the corresponding measure comparison and sum inequality.

Proposition 6.3 (Explicit unit-disk barrier). Let $u , v \in C ^ { 2 } ( B _ { 1 } ) \cap C ( { \overline { { B } } } _ { 1 } )$ be convex and suppose that

$$
\begin{array} { r } { \left\| \operatorname* { d e t } D ^ { 2 } u - \operatorname* { d e t } D ^ { 2 } v \right\| _ { L ^ { \infty } ( B _ { 1 } ) } \le R , \qquad \| u - v \| _ { L ^ { \infty } ( \partial B _ { 1 } ) } \le B _ { \partial } . } \end{array}
$$

Then

$$
\begin{array} { r } { \boxed { \| u - v \| _ { L ^ { \infty } ( B _ { 1 } ) } \leq B _ { \partial } + \frac { 1 } { 2 } \sqrt { R } . } } \end{array}
$$

The proof is given in Appendix F.

For smooth targets on the unit disk, this standard comparison argument provides an explicit coeficient while retaining square-root dependence on the residual. The numerical refinement evaluates this barrier bound using interval arithmetic and outward rounding, using the previously certified R and $B _ { \partial }$ . For $\mathrm { P 1 / P 2 }$ we also bound $| u _ { \theta } - u ^ { * } |$ directly on boxes using the known manufactured solution; that separate bound does not require a stability constant.

## 7 Evaluation procedure and stopping criteria

An implementation can factor a single $m \times m$ matrix $\hat { B }$ at a given input and reuse it for the two first-derivative and three distinct second-derivative right-hand sides. Forming an explicit inverse is unnecessary. For dense matrices, factorization has the usual cubic cost in $m$

1. Obtain an approximate state zˆ and calculate (10) and $\rho _ { z }$ .

2. Obtain ${ \hat { J } } _ { i }$ and symmetric $\hat { K } _ { i j }$ , either by solving the approximate-state systems or by diferentiating a finite unroll.

3. Evaluate (12)–(13) using the same zˆ and form (14)–(17).

4. Form $\hat { H } , E _ { H }$ and $E _ { \mathrm { d e t } }$ , then report the interval (21) rather than only ${ \hat { r } } .$

5. If the required evaluation tolerance is not met, refine the state and/or derivative computations and recompute the bounds.

An absolute stopping rule is $E _ { \mathrm { d e t } } \leq \tau _ { \mathrm { a b s } }$ . If $\hat { r } > 0$ , a relative-to-observed-residual criterion is $E _ { \mathrm { d e t } } \le \eta \hat { r }$ for $0 < \eta < 1$ , which guarantees

$$
( 1 - \eta ) \hat { r } \leq r \leq ( 1 + \eta ) \hat { r } .
$$

The corresponding ratio relative to the true residual is bounded by $\eta / ( 1 - \eta )$ , not by η. Near $\hat { r } = 0$ an absolute tolerance remains necessary. No monotonicity of the computed bounds with iteration count is assumed.

For a finite input set X, Corollary 5.2 gives

$$
\operatorname* { m a x } _ { x \in X } r ( x ) \leq \operatorname* { m a x } _ { x \in X } \bigl ( \hat { r } ( x ) + E _ { \mathrm { d e t } } ( x ) \bigr ) .
$$

Replacing X by a continuous domain requires additional control between sample points. Section 6 supplies that control through regional certificates and a complete cover. Appendix D develops an alternative analytical construction. A sampled analogue for explicit ICNNs, combining Lipschitz constants with covering radii, is given in Nyström $\&$ Vestberg (2023, Lemma 6.2).

## 8 Numerical experiments

We investigate three questions: how the a posteriori bounds compare with observed evaluation errors, whether adaptive stopping reduces total evaluation time, and how pointwise guarantees extend to the full domain. A synthetic consistency experiment tests the derivative estimates. Two manufactured Dirichlet problems then assess accuracy and certification on trained networks. Float64 diagnostics and interval guarantees are reported separately.

## 8.1 Consistency of the derivative bounds

We generate 120 random networks in double precision using seed 20261015, with 40 networks at each $q \in \{ 0 . 3 , 0 . 7 , 0 . 9 5 \}$ , all with $m = 8$ and $n = 2$ . The entries of a positive matrix are uniform on [0.05, 1] and its rows are normalized to sum to q. Entries of A and b are independent zero-mean Gaussians with standard deviation 0.5, readout weights are uniform on [0.1, 1], and the input is uniform on $[ - 1 , 1 ] ^ { 2 }$

A reference state uses fixed-point iteration until its computed infinity-norm residual is at most $1 0 ^ { - 1 3 }$ . Reference derivatives solve (A.2)–(A.3). An approximate state uses 4, 8, 12 or 16 iterations, cycling through these depths. Its derivative solves are perturbed by Gaussian noise with a shared scale $1 0 ^ { U }$ , where U is uniform on $[ - 7 , - 3 ]$ for each case; second-derivative noise is symmetrized. This probes both state error and nonzero linear-system residuals.

Table 1 reports no violations. The largest computed reference-state residual bound $\rho _ { z } / ( 1 - q )$ is $1 . 9 9 \times 1 0 ^ { - 1 2 }$ . The maximum spectral diference between the linear-system Hessian and its factorization, divided by max $\{ 1 , \| H \| _ { 2 } \}$ , is $2 . 1 0 \times 1 0 ^ { - 1 5 }$

## 8.2 Problems, training and evaluation protocol

For both problems, $\Omega = B _ { 1 } ( 0 )$ and $g = u ^ { * } | _ { \partial \Omega }$ . The manufactured solutions are

$$
\begin{array} { c c c } { { \mathrm { P 1 } : } } & { { u ^ { * } ( x ) = x _ { 1 } ^ { 2 } + \frac { 1 } { 4 } x _ { 2 } ^ { 2 } , ~ } } & { { ~ } } & { { ~ f ( x ) = 1 , } } \end{array}\tag{23}
$$

$$
\begin{array} { r } { \mathrm { P 2 } : \quad u ^ { * } ( x ) = \frac { 1 } { 2 } \| x \| _ { 2 } ^ { 2 } + 0 . 1 ( x _ { 1 } ^ { 4 } + x _ { 2 } ^ { 4 } ) , \qquad f ( x ) = ( 1 + 1 . 2 x _ { 1 } ^ { 2 } ) ( 1 + 1 . 2 x _ { 2 } ^ { 2 } ) . } \end{array}\tag{24}
$$

Their Hessians are $\mathrm { d i a g } ( 2 , 1 / 2 )$ and diag $( 1 + 1 . 2 x _ { 1 } ^ { 2 } , 1 + 1 . 2 x _ { 2 } ^ { 2 } )$ , respectively. Both targets are smooth and strictly convex, so Proposition 6.3 applies.

<table><tr><td>Quantity</td><td>Maximum ratio Violations</td></tr><tr><td>State error</td><td>0.98924 0</td></tr><tr><td>First derivative error</td><td>0.77929 0</td></tr><tr><td>Second derivative error</td><td>0.69098 0</td></tr><tr><td>Hessian error</td><td>0.17737 0</td></tr><tr><td>Determinant error</td><td>0.07730 0</td></tr><tr><td>Curvature bound</td><td>0.20601 0</td></tr><tr><td>Cancellation  $S \gamma \leq { \bf 1 }$ </td><td>0.65309 0</td></tr></table>

Table 1: Independent float64 consistency experiment, 120 networks. Error rows show the largest observed error-to-bound ratio, maximizing over coordinates where applicable. The last row is $\operatorname* { m a x } _ { r } ( S \gamma ) _ { r }$ , whose bound is one. A violation means a ratio exceeding $1 + 1 0 ^ { - 8 }$ . These are numerical checks against approximate reference quantities.

We train one model per problem and seed 0, 1, 2, with $m = 1 2$ , nominal $q = 0 . 7 ,$ , and $L = 3 2$ iterations initialized at zero. The parametrization

$$
W _ { r s } = q \sigma ( \xi _ { r } ) \frac { e ^ { M _ { r s } } } { \sum _ { t } e ^ { M _ { r t } } } , \qquad w _ { r } = \frac { \phi ( \nu _ { r } ) } { m }
$$

uses unconstrained $\xi , M , \nu ,$ with $\sigma$ the logistic function, and ensures nonnegative weights with row sums below $q$ in exact arithmetic. The efective saved parameters are rechecked during interval certification. Training minimizes

$$
\mathcal { L } ( \theta ) = \frac { 1 } { { { N _ { \Omega } } } } \sum _ { x \in { X _ { \Omega } } } { \left( { \operatorname* { d e t } { D ^ { 2 } } { u _ { \theta , L } } ( x ) - f ( x ) } \right) ^ { 2 } } + \frac { 1 0 } { { { N _ { \partial } } } } \sum _ { x \in { X _ { \partial } } } { \left( { { u _ { \theta , L } } ( x ) - g ( x ) } \right) ^ { 2 } } .\tag{25}
$$

Interior values of $u ^ { * }$ are used only for evaluation. Training diferentiates the finite unroll; the study does not benchmark an equilibrium implicit backward pass.

Interior evaluation points are sampled uniformly in the disk using the reserved seed 900000+ seed. Boundary angles are $2 \pi ( k + 0 . 3 7 1 ) / 5 1 2 , k = 0 , \ldots , 5 1 1$ . Training and evaluation samples are distinct. Reference evaluations use converged float64 states and linear solves; the numerical stopping test for the state correction divided by $1 - q { \mathrm { ~ i s ~ } } 1 0 ^ { - 1 3 }$ . These reference values are diagnostics, not exact real values. The maximum reference determinant diagnostic across the models is $1 . 6 5 \times 1 0 ^ { - 1 1 }$ (rounded up).

## 8.3 Accuracy of the six trained models

Table 3 reports sample metrics for the saved equilibrium models. The relative error is $\| u - u ^ { * } \| _ { \ell ^ { 2 } ( X ) } / \| u ^ { * } \| _ { \ell ^ { 2 } ( X ) }$ Its mean and sample standard deviation across the three seeds are $0 . 2 8 4 7 \% \pm 0 . 0 3 3 3 \%$ for P1 and $0 . 2 9 4 1 \% \pm 0 . 0 1 7 8 \%$ for P2. These are descriptive summaries over three runs, not uniform error guarantees.

The smallest sampled Hessian eigenvalues range from 0.4566 to 0.4756 for P1 and from 0.9801 to 1.0003 for P2. Global convexity follows from the representation theorem, not from these samples. The learned row norms range from 0.5817 to 0.6584, all below $3 / 4 .$ . Consequently, the cubic branch in (8) is smaller for these trained models: the experiments test the evaluation certificate, while the near-critical exponent sharpness is demonstrated by the analytical family in Theorem 4.4.

## 8.4 Pointwise certificates and iteration depth

There are $6 \times 6 \times 2 0 4 8 = 7 3 { , } 7 2 8$ point/depth pairs. Every pair has both a float64 diagnostic and an Arb certificate. The float64 diagnostic declares a violation only if the observed determinant error exceeds its estimator plus the reference diagnostic and a 10<sup>−11</sup> numerical floor. No such violations are observed. The interval audit contains 73,728 finite certificate records at 128-bit precision, with no missing point/depth pairs.

<table><tr><td>Item</td><td>Executed configuration</td></tr><tr><td>Adam</td><td>1,200 steps; initial learning rate 0.02; factor 0.35 at steps 780 and 1,080; fresh 128 interior and 64 boundary points per step</td></tr><tr><td>L-BFGS</td><td>Up to 100 iterations, strong-Wolfe line search; fixed 512 interior and 128 boundary points</td></tr><tr><td>Evaluation</td><td>2,048 interior points and 512 boundary points per model</td></tr><tr><td>Depth sweep</td><td> $L \in \{ 4 , 8 , 1 2 , 1 6 , 2 4 , 3 2 \}$  on the same 2,048 interior points</td></tr><tr><td>Float64 timing</td><td>512 points/model; five repetitions; tolerances  $1 0 ^ { - 4 }$  and  $1 0 ^ { - 6 }$ </td></tr><tr><td>Interval timing Arithmetic</td><td>Separate eight-point/model cohort; five repetitions; same tolerances CPU, float64, one computational thread; Arb at 128-bit precision</td></tr></table>

Table 2: Protocol of the completed six-model run.

<table><tr><td>Model Rel.</td><td> $\ell ^ { 2 }$  (%) Max.</td><td> $\left| u - u ^ { * } \right|$ </td><td>Max. |det H − f|</td><td></td><td>Max. boundary</td><td> $\| W \| _ { \infty }$ </td></tr><tr><td>P1/0</td><td>0.2478</td><td>0.003927</td><td></td><td>0.045998</td><td>0.004150</td><td>0.6584</td></tr><tr><td>P1/1</td><td>0.2937</td><td>0.004282</td><td></td><td>0.054399</td><td>0.004573</td><td>0.6135</td></tr><tr><td>P1/2</td><td>0.3125</td><td>0.004958</td><td></td><td>0.052480</td><td>0.005022</td><td>0.5817</td></tr><tr><td>P2/0</td><td>0.2808</td><td>0.003626</td><td></td><td>0.102253</td><td>0.003961</td><td>0.6466</td></tr><tr><td>P2/1</td><td>0.2872</td><td>0.003799</td><td></td><td>0.080306</td><td>0.003872</td><td>0.6580</td></tr><tr><td>P2/2</td><td>0.3144</td><td>0.005618</td><td></td><td>0.089144</td><td>0.005954</td><td>0.6330</td></tr></table>

Table 3: Float64 sample metrics. A model label P1/0 means problem P1, seed 0. Interior maxima use 2,048 points; boundary maxima use 512 points. Values are rounded for display and are not interval bounds.

Figure 2 shows decreasing evaluation errors with depth. At depth 12, the median float64 bound/error ratio over points above the comparison floor ranges from 26.15 to 58.03 across the six models. Ratios at the floating-point floor are excluded from this statistic.

A fixed iteration threshold does not certify every point relative to its PDE residual. At depth 12, 62.99–89.11% of P1 points satisfy $E _ { \mathrm { d e t } } \le 0 . 0 5 \hat { r }$ , depending on the seed. At depth 16, the corresponding P2 fractions are 86.52–92.38%. All six sampled cohorts satisfy this float64 criterion at depth 32. Absolute tolerances remain necessary near vanishing residuals, as explained in Section 7.

![](images/75b09826904879d235030b0a5348739cd770f30ed5e033e887bfdc8c8ea61246.jpg)  
Figure 2: Determinant evaluation error over iteration depth for all six models. Each curve is the maximum over the same 2,048 evaluation points. Blue: float64 estimator; orange: outward-rounded Arb upper bound; green: error against the float64 reference. The gray line marks $1 0 ^ { - 1 1 }$ , the additive diagnostic floor; the reference diagnostic is added separately in violation counting. Curves near this scale should not be interpreted as exact-error measurements.

## 8.5 Cost of adaptive evaluation

The float64 benchmark compares fixed 32 iterations, fixed 32 plus the diagnostic certificate, and adaptive stopping on checkpoints 4, 8, 12, 16, 24, 32, 48, 64. The timer includes the certificate and management of the remaining active points. Methods are warmed up, and their timing order is permuted deterministically across five repetitions. We use the ratio of median times, $t _ { 3 2 } / t _ { \mathrm { a d a p t i v e } } .$ as the speedup measure. The baseline for this cohort is fixed-32 evaluation without a certificate.

At tolerance $1 0 ^ { - 4 }$ , adaptive float64 evaluation uses mean depths 11.66–16.05 and is 1.45–1.99 times faster across models. At $1 0 ^ { - 6 }$ it uses mean depths 14.95–20.84 and is 1.13–1.48 times faster. All 512 points per model are accepted at both tolerances, and observed errors against the reference lie below the requested tolerances. This stopping rule uses the float64 estimator without roundof enclosure.

![](images/ab34581a785d4f18a91c154d55b03492bb525db720558c16162a4f84dfd7ef35.jpg)

![](images/64c86de1ea91fc8c9e0f117b7febc897398aadc0539a2257e02a9f7a39fe5a52.jpg)  
Figure 3: Matched-cohort evaluation timings. Ratios above one favor adaptive evaluation; ratios below one favor fixed 32 iterations. Left: float64 estimator on 512 points/model, compared with fixed 32 without certification. Right: full Arb stopping on eight points/model, compared with fixed 32 plus Arb certification. Both comparisons include the adaptive certificate cost and use five repetitions. The cohorts and baselines difer; their throughputs must not be compared directly.

The separate Arb benchmark includes all interval work in both fixed and adaptive timings. All eight points per model are certified within each requested tolerance. Nevertheless, speedup ratios are 0.509–0.677 at $1 0 ^ { - 4 }$ and 0.376–0.472 at $1 0 ^ { - 6 }$ : adaptive Arb evaluation is approximately 1.48–2.66 times slower. Repeated certification outweighs the iteration savings in this implementation and cohort. The interval timing loops use fixed-then-adaptive order after warmup, unlike the interleaved float64 benchmark; timing-order efects and the small cohort limit broader performance conclusions.

## 8.6 Complete-domain and boundary certification

The initial domain protocol uses exact dyadic boxes, 128-bit Arb calculations, depth at most 8, a 20,000-box work budget, and residual target $R \leq 0 . 2 5$ . Boundary refinement uses trigonometric interval arcs, depth at most 12, an 8,192-arc budget and target $B _ { \partial } \leq 0 . 0 1$ . All six initial covers are complete, all satisfy the boundary target, and five satisfy the residual target. Initial domain-stage times range from 62.7 to 202.1 seconds in the original runtime.

For P2, seed 2, the initial residual bound is 0.4376936062, with 14,584 retained boxes and 502 boxes above the requested tolerance. We therefore perform a separately recorded, post hoc refinement of that fixed model. Only boxes above 0.25 are subdivided, prioritizing the largest bounds. The first stage permits depth 8; the second permits depth 9 if needed. Each stage is limited to 8,000 child-box evaluations and 120 seconds, with a cooperative time check before each complete four-child split. A parent is replaced only after all its intersecting children have certificates. Its valid bounds also apply to each child and are retained when tighter. Boundary certificates and model parameters are unchanged.

![](images/c2a0c44d4e082e9470f61274350650943b6c9ca848f78cf11b22977807a7ebaa.jpg)

![](images/fd4dd648372d2726a17095dd51399966fbf5e2566f0dec4e1a26c3a4fa1bbed0.jpg)  
Figure 4: Additional P2, seed-2 certification, without retraining. The residual upper bound decreases from 0.437694 to 0.254972 at depth 8 and then to 0.249951 at depth 9 (displayed bounds rounded up). The cover remains complete throughout. Stage 1 computes 1,850 new certificates and stage 2 computes 10, retaining 15,937 boxes at completion.

The additional refinement uses a separate CPU environment with one computational thread. Its measured incremental computation time is 3.32 seconds, excluding input/output and coverage audits. This reuses the existing domain and boundary certificates, runs in a diferent runtime, and is not a matched speed comparison with the original full-domain calculation. The exported partition is checked using exact arithmetic.

After refinement, every model satisfies $R < 0 . 2 5$ and $B _ { \partial } < 0 . 0 1$ (Table 4). Uniform residual bounds are approximately 2.44–5.06 times the corresponding sampled residual maxima.

<table><tr><td>Model</td><td>R</td><td> $B _ { \partial }$ </td><td>Direct U</td><td>Barrier U</td><td>Boxes</td></tr><tr><td>P1/0</td><td>0.232852</td><td>0.010000</td><td>0.107237</td><td>0.251273</td><td>3,332</td></tr><tr><td>P1/1</td><td>0.243891</td><td>0.009989</td><td>0.114571</td><td>0.256915</td><td>3,332</td></tr><tr><td>P1/2</td><td>0.217254</td><td>0.009996</td><td>0.110518</td><td>0.243048</td><td>3,332</td></tr><tr><td>P2/0</td><td>0.249993</td><td>0.009997</td><td>0.053797</td><td>0.259994</td><td>11,310</td></tr><tr><td>P2/1</td><td>0.249997</td><td>0.009999</td><td>0.065538</td><td>0.259997</td><td>11,668</td></tr><tr><td>P2/2</td><td>0.249951</td><td>0.009989</td><td>0.046754</td><td>0.259965</td><td>15,937</td></tr></table>

Table 4: Uniform upper bounds on the closed disk and its boundary. $\mathrm { P 2 / 2 }$ uses the additional refinement; all other rows use the original covers. Direct U bounds $\lvert u _ { \theta } - u ^ { * } \rvert$ using the known manufactured solution. Barrier $U = B _ { \partial } + \sqrt { R } / 2$ uses Proposition 6.3. Every displayed bound is rounded upward to six decimal places.

![](images/15af41f852ac25c1d5d3bfc54be09adb6a9d52ae1572bfb8802029bb34369494.jpg)

![](images/f4086a310585c75a7493706c24240395d42a9680b82167b910092cba348f3666.jpg)  
Figure 5: Samples and uniform bounds have diferent meanings. Left: observed PDE residual maxima versus interval upper bounds over the disk. Right: observed solution-error maxima, direct uniform bounds using the manufactured target, and the explicit barrier bound. P2/2 includes the final refinement. Curves connect model labels for readability; no trend across seeds is assumed.

The barrier expression is evaluated in Arb from the certified binary64 upper bounds and exported upward. For the six final covers it gives solution-error bounds of approximately 0.243–0.260. The direct manufactured-solution bounds, approximately 0.047–0.115, are tighter in every case. In particular, P2/2 has barrier bound 0.2599640704 and direct bound 0.0467530969, both rounded up. The direct bound is obtained independently and is unchanged by this refinement.

## 9 Discussion and limitations

Evaluation and training. The results certify evaluation of frozen networks. They do not prove convergence of sampled strong-loss training or control approximation, optimization and loss-discretization errors along a training sequence. Two manufactured problems and three seeds each provide limited empirical coverage, and no comparison with explicit ICNNs or other PDE solvers is conducted here.

Scope and tightness. The inverse-gap exponents are sharp over the network class, but optimal prefactors remain open. The weighted result and analytical local envelope are theoretical extensions; the implemented regional path uses unweighted interval residuals and verified solves. The disk barrier is stated for smooth convex targets. Its square-root dependence can give loose solution estimates, and the direct interval bounds also substantially exceed the observed sample errors.

Performance and numerical guarantees. Adaptive float64 evaluation reduces measured runtime, while full Arb stopping is slower in the tested implementation. Interval batching, reuse and stopping-policy design are possible future improvements, not demonstrated gains. Outward arithmetic, exact coverage checks and consistency tests support the implementation, but do not constitute formal verification of the complete software stack.

## 10 Conclusion

For nonnegative contractive softplus equilibrium networks, positivity yields $S \gamma \leq { \bf 1 }$ and a uniform Hessian bound with optimal inverse-gap exponent two. A residual-based perturbation argument controls derivatives computed at approximate states. Third derivatives admit sharp exponent three, and positive weighted norms extend the representation condition to $\rho ( W ) < 1$ . Regional interval residuals and complete boundary covers provide uniform bounds for frozen models, with an explicit residual-to-solution estimate on the disk. The six-model study supplies 73,728 point/depth interva certificates and, after one bounded fixed-model refinement, complete covers meeting the prescribed residual and boundary targets for every model. Adaptive evaluation saves time with the float64 estimator, whereas rigorous adaptive stopping remains more expensive than fixed-depth certification in the measured cohorts. These findings quantify both the usefulness and the cost of certified evaluation.

## Code availability

The code used to train and evaluate the models is publicly available at https://github.com/ EdisonVazquezG/monge-ampere-equilibrium.

## Use of AI tools

The authors used ChatGPT (OpenAI) and Claude (Anthropic) to assist with language editing, consistency checks of notation and statements, verification of bibliographic references against their sources. All mathematical results, proofs, experiments and conclusions are the authors’ own. The authors reviewed and edited all AI-assisted content and take full responsibility for the manuscript.

## References

Amos, Brandon, Xu, Lei, and Kolter, J. Zico. Input convex neural networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 146–155, 2017. URL https://proceedings.mlr.press/v70/amos17b.html.

Bai, Shaojie, Kolter, J. Zico, and Koltun, Vladlen. Deep equilibrium models. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://proceedings.neurips.cc/paper/2019/ h h/01386bd6d8 091 2 b4 7 7d 644d37b Ab h l.

Boyd, Stephen and Vandenberghe, Lieven. Convex Optimization. Cambridge University Press, 2004.

Eiras, Francisco, Bibi, Adel, Bunel, Rudy R., Dvijotham, Krishnamurthy, Torr, Philip, and Kumar, M. Pawan. Eficient error certification for physics-informed neural networks. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 12318–12347, 2024. URL https://proceedings.mlr.press/v235/eiras24a.html.

El Ghaoui, Laurent, Gu, Fangda, Travacca, Bertrand, Askari, Armin, and Tsai, Alicia Y. Implicit deep learning. SIAM Journal on Mathematics of Data Science, 3(3):930–958, 2021. doi: 10.1137/20M1358517.

Entesari, Taha, Sharifi, Sina, and Fazlyab, Mahyar. Compositional curvature bounds for deep neural networks. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 12527–12546, 2024. URL https://proceedings.mlr.press/v235/ entesari24a.html.

Figalli, Alessio. The Monge–Ampère Equation and Its Applications. Zurich Lectures in Advanced Mathematics. European Mathematical Society, Zürich, 2017.

Froese, Brittany D. and Oberman, Adam M. Convergent finite diference solvers for viscosity solutions of the elliptic Monge–Ampère equation in dimensions two and higher. SIAM Journal on Numerical Analysis, 49 (4):1692–1714, 2011. doi: 10.1137/100803092.

Gutiérrez, Cristian E. The Monge–Ampère Equation, volume 89 of Progress in Nonlinear Diferential Equations and Their Applications. Birkhäuser, second edition, 2016. doi: 10.1007/978-3-319-43374-5.

Huang, Yuhao, Wang, Qingsong, Onwunta, Akwum, and Wang, Bao. Eficient score matching with deep equilibrium layers. In The Twelfth International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ b73442f01ee75616fad938e1eb1f2427-Abstract-Conference.html.

Jafarpour, Saber, Davydov, Alexander, Proskurnikov, Anton, and Bullo, Francesco. Robust implicit networks via non-euclidean contractions. In Advances in Neural Information Processing Systems, volume 34, 2021. URL https://proceedings.nips.cc/paper\_files/paper/2021/hash/ 51a6ce0252d8fa6e913524bdce8db490-Abstract.html.

Johansson, Fredrik. Arb: Eficient arbitrary-precision midpoint-radius interval arithmetic. IEEE Transactions on Computers, 66(8):1281–1292, 2017. doi: 10.1109/TC.2017.2690633.

Makkuva, Ashok, Taghvaei, Amirhossein, Oh, Sewoong, and Lee, Jason. Optimal transport mapping via input convex neural networks. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 6672–6681, 2020. URL https: //proceedings.mlr.press/v119/makkuva20a.html.

Mishra, Siddhartha and Molinaro, Roberto. Estimates on the generalization error of physics-informed neural networks for approximating PDEs. IMA Journal of Numerical Analysis, 43(1):1–43, 2023. doi: 10.1093/imanum/drab093.

Nyström, Kaj and Vestberg, Matias. Solving the Dirichlet problem for the Monge–Ampère equation using neural networks. Journal of Computational Mathematics and Data Science, 8:100080, 2023. doi: 10.1016/j.jcmds.2023.100080. URL https://arxiv.org/abs/2110.03310.

Raissi, Maziar, Perdikaris, Paris, and Karniadakis, George Em. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational Physics, 378:686–707, 2019. doi: 10.1016/j.jcp.2018.10.045.

Rump, Siegfried M. Verification methods: Rigorous results using floating-point arithmetic. Acta Numerica, 19:287–449, 2010. doi: 10.1017/S096249291000005X.

Sirignano, Justin and Spiliopoulos, Konstantinos. DGM: A deep learning algorithm for solving partial diferential equations. Journal of Computational Physics, 375:1339–1364, 2018. doi: 10.1016/j.jcp.2018.08. 029.

Winston, Ezra and Kolter, J. Zico. Monotone operator equilibrium networks. In Advances in Neural Information Processing Systems, volume 33, 2020. URL https://papers.nips.cc/paper/2020/hash/ 798d1c2813cbdf8bcdb388db0e32d496-Abstract.html.

## A Equilibrium foundations and derivative identities

## A.1 Softplus and the equilibrium

Lemma A.1 (Derivatives of the Softplus function ϕ).

$$
0 < \phi ^ { \prime } ( t ) < 1 , \qquad \phi ^ { \prime \prime } ( t ) = \phi ^ { \prime } ( t ) ( 1 - \phi ^ { \prime } ( t ) ) \in ( 0 , 1 / 4 ] , \qquad \phi ^ { \prime \prime \prime } ( t ) = \phi ^ { \prime } ( t ) ( 1 - \phi ^ { \prime } ( t ) ) ( 1 - 2 \phi ^ { \prime } ( t ) ) .
$$

In particular $\phi$ is increasing, convex and 1-Lipschitz; both $\phi ^ { \prime }$ and $\phi ^ { \prime \prime }$ are 1/4-Lipschitz.

Proof. Diferentiate $\phi ( t ) = \log ( 1 + e ^ { t } )$ to obtain $\phi ^ { \prime } ( t ) = e ^ { t } / ( 1 + e ^ { t } )$ and the displayed identities. The bounds follow from the fact that $a ( 1 - a ) \leq 1 / 4$ and $| 1 - 2 a | \leq 1$ for any $a \in ( 0 , 1 )$ . The mean value theorem gives the Lipschitz statements. □

Proposition A.2 (Existence, convexity and state error). Under assumption (H), for every $x \in \mathbb { R } ^ { n }$ the fixed-point equation in (2) has a unique solution. Each component of $z ^ { * } ( x )$ and the scalar function $u _ { \theta } ( x )$ is convex. For any approximate state $\hat { z } \in \mathbb { R } ^ { m }$ , define

$$
\rho _ { z } = \lVert \hat { z } - \phi ( W \hat { z } + A x + b ) \rVert _ { \infty } .
$$

Then

$$
\| z ^ { * } ( x ) - \hat { z } \| _ { \infty } \leq \frac { \rho _ { z } } { \delta } .\tag{A.1}
$$

Moreover, $\mathrm { L i p } ( u _ { \theta } ) \leq \| w \| _ { 1 } K _ { A } / \delta + \| a \| _ { 2 }$ , where the input space carries the Euclidean norm.

Proof. For fixed $x ,$ let $T _ { x } ( z ) = \phi ( W z + A x + b )$ . Lemma A.1 gives

$$
\begin{array} { r } { \| T _ { x } ( z ) - T _ { x } ( \tilde { z } ) \| _ { \infty } \leq \| W \| _ { \infty } \| z - \tilde { z } \| _ { \infty } \leq q \| z - \tilde { z } \| _ { \infty } . } \end{array}
$$

The space $\mathbb { R } ^ { m }$ is complete, so Banach’s theorem gives a unique fixed point and convergence of the iterates. Starting from the constant function $z _ { 0 } ( x ) = 0$ , each component of $W z _ { \ell } ( x ) + A x + b$ is convex whenever those of $z _ { \ell }$ are convex. Composition with the increasing convex ϕ preserves convexity. Pointwise convergence $z _ { \ell } \to z ^ { * }$ then gives entrywise convexity of the limit. Nonnegative linear combination and addition of an afine function yield convexity of $u \theta$

The triangle inequality and contraction imply

$$
\begin{array} { r } { \left. \hat { z } - z ^ { * } \right. _ { \infty } \leq \rho _ { z } + q \left. \hat { z } - z ^ { * } \right. _ { \infty } , } \end{array}
$$

which proves (A.1). Finally, comparing the fixed-point equations at x, y gives

$$
\begin{array} { r } { \| { z } ^ { * } ( { x } ) - { z } ^ { * } ( { y } ) \| _ { \infty } \leq q \| { z } ^ { * } ( { x } ) - { z } ^ { * } ( { y } ) \| _ { \infty } + K _ { A } \| { x } - { y } \| _ { 2 } . } \end{array}
$$

Divide by δ and apply the readout.

## A.2 Implicit derivatives and Hessian structure

Proposition A.3 (Smoothness and derivative systems). The function $z ^ { * }$ is $C ^ { \infty }$ on $\mathbb { R } ^ { n }$ . For $1 \leq i , j \leq n$

$$
B J _ { i } = \Psi A _ { : , i } ,\tag{A.2}
$$

$$
B K _ { i j } = \gamma \odot v _ { i } \odot v _ { j } , \qquad v _ { i } = A _ { : , i } + W J _ { i } = V _ { : , i } .\tag{A.3}
$$

Furthermore $S \geq 0 , S \geq I , \| S \| _ { \infty } \leq \delta ^ { - 1 }$ , and $J = \Psi V$

Proof. Since $0 < \Psi < I$ on the diagonal, $\| \Psi W \| _ { \infty } \leq q$ and $\| W \Psi \| _ { \infty } \leq q$ . Both inverses in (6) therefore exist since the Neumann series converge. In particular

$$
S = \sum _ { \ell = 0 } ^ { \infty } ( \Psi W ) ^ { \ell } \geq I , \qquad \| S \| _ { \infty } \leq \sum _ { \ell = 0 } ^ { \infty } q ^ { \ell } = \delta ^ { - 1 } .
$$

For $F ( z , x ) = z - \phi ( W z + A x + b )$ the derivative in z equals B at a fixed point. The implicit function theorem yields a smooth local solution at every input. Uniqueness makes these local solutions agree on overlaps, giving a global smooth $z ^ { * }$

Diferentiate $z ^ { * } = \phi ( s )$ once to obtain $J _ { i } = \Psi ( W J _ { i } + A _ { : , i } )$ , hence (A.2). A second diferentiation gives

$$
K _ { i j } = \Psi W K _ { i j } + \gamma \odot ( W J _ { i } + A _ { : , i } ) \odot ( W J _ { j } + A _ { : , j } ) ,
$$

which proves (A.3). Finally, the identity $( I - \Psi W ) \Psi = \Psi ( I - W \Psi )$ implies $S \Psi A = \Psi ( I - W \Psi ) ^ { - 1 } A =$ $\Psi V .$ . Since $J = S \Psi A$ , it also follows that $A + W J = V$ □

Proposition A.4 (Hessian factorization). Let $\beta = ( S ^ { \top } w ) \odot \gamma$ . Then

$$
H : = D ^ { 2 } u _ { \theta } = V ^ { \top } \mathrm { d i a g } ( \beta ) V \succeq 0 .\tag{A.4}
$$

If all entries of w are strictly positive, then $H \succ 0$ if and only if rank $A = n$ . For $n = 2$ , writing $v _ { r 1 } , v _ { r 2 }$ for the entries of row r of V ,

$$
\operatorname* { d e t } H = \sum _ { 1 \leq r < t \leq m } \beta _ { r } \beta _ { t } ( v _ { r 1 } v _ { t 2 } - v _ { r 2 } v _ { t 1 } ) ^ { 2 } .\tag{A.5}
$$

Proof. For any $h \in \mathbb { R } ^ { n }$ , bilinearity of (A.3) gives

$$
{ \cal D } ^ { 2 } z ^ { * } [ h , h ] = S \big [ \gamma \odot ( V h ) ^ { 2 } \big ] .
$$

The afine readout has zero Hessian, so

$$
h ^ { \top } H h = ( S ^ { \top } w ) ^ { \top } [ \gamma \odot ( V h ) ^ { 2 } ] = \sum _ { r } \beta _ { r } ( V h ) _ { r } ^ { 2 } .
$$

This proves $\left( \mathrm { { A . 4 } } \right)$ . All $\beta _ { r }$ are nonnegative. If $w > 0$ , then $S \geq I$ and $\gamma > 0$ give $\beta > 0$ . Consequently H is positive definite exactly when V has full column rank. The invertibility of $I - W \Psi$ gives rank $V = \operatorname { r a n k } A$ . Formula (A.5) is Cauchy–Binet applied to diag ${ \sqrt { \beta } } ) V$ □

Remark A.5. Strict positivity of every entry of w is a suficient hypothesis for the rank criterion, not a necessary condition for strict convexity of every network. The factorization itself remains valid when some readout weights vanish. The two-dimensional determinant formula is also an algebraic identity, not a roundof guarantee for its numerical evaluation.

## B Proof of the third-derivative bound

This appendix proves Theorem 4.6, including optimality of the inverse-gap exponent.

Proof. Put $M = K _ { A } / \delta$ . For unit vectors $h _ { 1 } , h _ { 2 } , h _ { 3 }$ , write $v _ { i } = V h _ { i } , K _ { i j } = D ^ { 2 } z ^ { * } [ h _ { i } , h _ { j } ]$ , and $t = D ^ { 3 } z ^ { * } [ h _ { 1 } , h _ { 2 } , h _ { 3 } ]$ . Positivity of S and the cancellation $S \gamma \leq { \bf 1 }$ give

$$
| v _ { i } | \leq M \mathbf { 1 } , \qquad | K _ { i j } | = | S ( \gamma \odot v _ { i } \odot v _ { j } ) | \leq M ^ { 2 } \mathbf { 1 } .
$$

Diferentiating the second-derivative equation gives

$$
\begin{array} { r l } & { B t = \phi ^ { \prime \prime \prime } ( s ) \odot v _ { 1 } \odot v _ { 2 } \odot v _ { 3 } } \\ & { \qquad + \gamma \odot \big ( ( W K _ { 1 2 } ) \odot v _ { 3 } + ( W K _ { 1 3 } ) \odot v _ { 2 } + ( W K _ { 2 3 } ) \odot v _ { 1 } \big ) . } \end{array}
$$

Since $| \phi ^ { \prime \prime \prime } ( s ) | \le \gamma$ and $| W K _ { i j } | \le q M ^ { 2 } \mathbf { 1 }$ , multiplication by $S \geq 0$ yields $\vert t \vert \leq ( 1 + 3 q ) M ^ { 3 } S \gamma \leq$ $( 1 + 3 q ) M ^ { 3 } { \bf 1 }$ . Apply $w ^ { \top }$ and take the supremum over all three directions.

For sharpness use the one-neuron family of Theorem 4.4, for which $q = \phi ^ { \prime } ( s ) = 1 - \varepsilon { \mathrm { ~ a t ~ } } x = 0$ Direct scalar diferentiation gives

$$
u ^ { \prime \prime \prime } ( 0 ) = { \frac { \phi ^ { \prime \prime \prime } ( s ) } { ( 1 - q \phi ^ { \prime } ( s ) ) ^ { 4 } } } + { \frac { 3 q \phi ^ { \prime \prime } ( s ) ^ { 2 } } { ( 1 - q \phi ^ { \prime } ( s ) ) ^ { 5 } } } = { \frac { ( 1 - \varepsilon ) ( 1 - \varepsilon + \varepsilon ^ { 2 } ) } { \varepsilon ^ { 3 } ( 2 - \varepsilon ) ^ { 5 } } } .
$$

Thus $\varepsilon ^ { 3 } u ^ { \prime \prime \prime } ( 0 ) \to 1 / 3 2$ . This proves optimality of the exponent.

## C Positive weighted norms

Weighted contraction is a standard tool for implicit networks; see, for example, Jafarpour et al. (2021), and El Ghaoui et al. (2021) for Perron–Frobenius well-posedness conditions. The result here transports the softplus curvature cancellation to that setting.

Proposition C.1 (Weighted curvature). Let $W , w \geq 0$ and suppose that there exist $\eta > 0$ and $q < 1$ with $W \eta \leq q \eta$ . Normalize max $\eta _ { r } = 1$ and set

$$
\left. v \right. _ { \eta } = \operatorname* { m a x } _ { r } \left| v _ { r } \right| / \eta _ { r } , \qquad K _ { \eta } = \operatorname* { m a x } _ { r } \left. A _ { r , : } \right. _ { 2 } / \eta _ { r } .
$$

The equilibrium exists uniquely, is smooth and convex, and

$$
\left\| D ^ { 2 } u _ { \theta } \right\| _ { 2 } \leq ( w ^ { \top } \eta ) K _ { \eta } ^ { 2 } ( 1 - q ) ^ { - 2 } .
$$

Such a positive weight exists for every q with $\rho ( W ) < q < 1$

Proof. The η-weighted induced infinity norms of W, ΨW and $W \Psi$ are at most $q .$ The earlier contraction, positivity and smoothness arguments therefore apply. Moreover,

$$
\gamma \odot \eta \le ( I - \Psi ) \eta \le ( I - \Psi W ) \eta , \qquad S ( \gamma \odot \eta ) \le \eta .
$$

For a unit input direction, $| V h | \le K _ { \eta } \eta / ( 1 - q )$ . Because $0 < \eta _ { r } \leq 1$ , its entrywise square is at most $K _ { \eta } ^ { 2 } \eta / ( 1 - q ) ^ { 2 }$ . The directional Hessian formula then gives the stated bound. Finally, for $q > \rho ( W )$ the Neumann series gives $\eta _ { 0 } = ( I - W / q ) ^ { - 1 } \mathbf { 1 } > 0$ , and $W \eta _ { 0 } = q ( \eta _ { 0 } - { \bf 1 } ) \le q \eta _ { 0 } ;$ ; normalization preserves this inequality. □

Remark C.2 (The spectral-radius endpoint need not be attained). In general, in $\mathrm { d } { \ f _ { \eta > 0 } }$ max<sub>r</sub> $( W \eta ) _ { r } / \eta _ { r } =$ $\rho ( W )$ , but a strictly positive minimizing vector need not exist. For $\boldsymbol { W } = \left( \begin{array} { l } { r } { 1 } \\ { 0 } \end{array} \right)$ with $0 \leq r < 1$ , every positive η gives a first-row ratio $r + \eta _ { 2 } / \eta _ { 1 } > r$ . A positive Perron vector, when available, permits equality.

## D Analytical local envelopes

Proposition D.1 (Local envelopes). Let $Q = x _ { 0 } + [ - \rho , \rho ] ^ { n }$ and let $a _ { r } = \| A _ { r , : } \| _ { 2 }$ . Write $\Psi ( x ) =$ diag $\left( \phi ^ { \prime } ( s ( x ) ) \right)$ , with $s ( x ) = W z ^ { * } ( x ) + A x + b$ . Suppose a diagonal matrix Ψ bounds $\Psi ( x )$ entrywise on $Q$ and satisfies $0 \leq \overline { { \Psi } } \leq I$ . Set

$$
P = ( I - W \overline { { \Psi } } ) ^ { - 1 } , \qquad e = \sqrt { n } \rho P a .
$$

If $s _ { 0 } = s ( x _ { 0 } )$ , then $| s ( x ) - s _ { 0 } | \leq e$ on $Q$ . Consequently diag $\zeta ( \phi ^ { \prime } ( s _ { 0 } + e ) )$ is another valid upper envelope. Starting at $\overline { { \Psi } } = I$ , intersection with this new bound preserves validity at every step.

Proof. The Neumann expansion and $\Psi ( x ) \leq { \overline { { \Psi } } }$ imply $( I - W \Psi ( x ) ) ^ { - 1 } \leq P$ . For any unit direction $h , | D s ( x ) h | = | ( I - W \Psi ( x ) ) ^ { - 1 } A h | \leq P a$ . Integrate this inequality along the segment from $x _ { 0 }$ to $x ,$ which lies in $Q ,$ , and use $\| x - x _ { 0 } \| _ { 2 } \leq { \sqrt { n } } \rho$ . Since $\phi ^ { \prime }$ is increasing, the updated diagonal bounds $\Psi ( x )$ . The initial envelope is valid since $0 < \phi ^ { \prime } < 1$ , and induction proves the assertion. If only an enclosure $s _ { 0 } \in [ \underline { { s } } _ { 0 } , \overline { { s } } _ { 0 } ]$ is available, replace the upper endpoint by $\overline { { s } } _ { 0 } + e$ and the lower by $\underline { { s } } _ { 0 } - e ;$ an unverified center approximation cannot be substituted for $s _ { 0 }$ □

For any valid final envelope, recompute P and define $v = P a , \overline { { S } } = ( I - \overline { { \Psi } } W ) ^ { - 1 }$ . Let $\overline { \gamma }$ bound $\phi ^ { \prime \prime } ( t ) = \phi ^ { \prime } ( t ) ( 1 - \phi ^ { \prime } ( t ) )$ on the resulting preactivation intervals, including its maximum $1 / 4$ when the interval contains zero. Then

$$
\begin{array} { r l r l } & { k = \overline { { S } } ( \overline { { \gamma } } \odot v ^ { 2 } ) , \qquad } & & { C _ { 2 } ( Q ) = w ^ { \top } k , } \\ & { t = \overline { { S } } ( \overline { { \gamma } } \odot ( v ^ { 3 } + 3 v \odot W k ) ) , \qquad } & & { C _ { 3 } ( Q ) = w ^ { \top } t } \end{array}
$$

bound the Hessian and third-derivative norms on $Q .$ . These follow by the same mixed-direction absolute-value arguments used above. In two dimensions, $\operatorname { L i p } _ { Q }$ (det $D ^ { 2 } u _ { \theta } - f ) \leq 2 C _ { 2 } ( Q ) C _ { 3 } ( Q ) +$ $\operatorname { L i p } _ { Q } ( f )$ , whenever $f$ is Lipschitz on $Q \colon$ the derivative of the determinant pairs cof $( H )$ with a directional derivative of $H$ , and $\left. \cot ( H ) \right. _ { * } \leq 2 \left. H \right. _ { 2 }$ . This gives a center-plus-variation covering construction. A sampled analogue for explicit softplus ICNNs, combining Lipschitz constants of the residual with covering radii, appears in Nyström & Vestberg (2023, Lemmas 6.2–6.3); here the derivative envelopes account for the equilibrium resolvent. Its sharpness and computational cost are separate questions.

## E Proofs of the approximate-state certificates

## E.1 Proof of Theorem 5.1

Proof. The state estimate is Proposition A.2. It implies

$$
\begin{array} { r } { \| s - \hat { s } \| _ { \infty } \leq q E _ { z } . } \end{array}
$$

By Lemma A.1,

$$
\begin{array} { r } { \left\| \Psi - \hat { \Psi } \right\| _ { \infty } \leq \alpha , \qquad \left\| \gamma - \hat { \gamma } \right\| _ { \infty } \leq \alpha . } \end{array}\tag{E.1}
$$

Let $r _ { i } = \hat { B } \hat { J } _ { i } - \hat { \Psi } A _ { : , i }$ . Since $B J _ { i } = \Psi A _ { : , i }$ and $B = \hat { B } - ( \Psi - \hat { \Psi } ) W$ , direct subtraction gives

$$
B ( J _ { i } - \hat { J } _ { i } ) = ( \Psi - \hat { \Psi } ) \hat { v } _ { i } - r _ { i } .\tag{E.2}
$$

Multiplication by $S$ and use of $\begin{array} { r } { \| S \| _ { \infty } \le \delta ^ { - 1 } } \end{array}$ prove the first-derivative bound. In particular, if $e _ { i } = v _ { i } - { \hat { v } } _ { i } = W ( J _ { i } - { \hat { J } } _ { i } )$ , then $\| e _ { i } \| _ { \infty } \leq \Delta _ { i }$

For the second derivatives, set

$$
r _ { i j } = \hat { B } \hat { K } _ { i j } - \hat { \gamma } \odot \hat { v } _ { i } \odot \hat { v } _ { j } .
$$

Subtracting the approximate equation from (A.3) gives the exact identity

$$
\begin{array} { r l } & { B ( K _ { i j } - \hat { K } _ { i j } ) = \gamma \odot v _ { i } \odot v _ { j } - \hat { \gamma } \odot \hat { v } _ { i } \odot \hat { v } _ { j } } \\ & { \qquad + ( \Psi - \hat { \Psi } ) W \hat { K } _ { i j } - r _ { i j } . } \end{array}\tag{E.3}
$$

The diference of the nonlinear right-hand sides expands as

$$
\begin{array} { r l } & { \gamma \odot v _ { i } \odot v _ { j } - \hat { \gamma } \odot \hat { v } _ { i } \odot \hat { v } _ { j } } \\ & { \quad = ( \gamma - \hat { \gamma } ) \odot \hat { v } _ { i } \odot \hat { v } _ { j } + \gamma \odot ( e _ { i } \odot \hat { v } _ { j } + \hat { v } _ { i } \odot e _ { j } + e _ { i } \odot e _ { j } ) . } \end{array}
$$

Its infinity norm is at most $Q _ { i j }$ by $( \mathrm { E . 1 } ) , \| \gamma \| _ { \infty } \leq 1 / 4$ and the bounds on $e _ { i } , e _ { j }$ . The remaining coeficient term is at most αq $\left\| \hat { K } _ { i j } \right\| _ { \infty }$ . Applying S to (E.3) gives (17) and the second-derivative estimate.

Finally, $| H _ { i j } - \hat { H } _ { i j } | \leq \Vert w \Vert _ { 1 } E _ { i j }$ . The symmetry hypothesis implies $E _ { 1 2 } = E _ { 2 1 }$ , and the Frobenius norm dominates the spectral norm, proving (19). The value estimate follows directly from the readout and the state bound. □

## E.2 Proof of Corollary 5.2

Proof. Write $E = H - { \hat { H } }$ . The two-dimensional determinant identity gives

$$
\operatorname* { d e t } ( { \hat { H } } + E ) - \operatorname* { d e t } { \hat { H } } = \operatorname { c o f } ( { \hat { H } } ) : E + \operatorname* { d e t } E .
$$

For a $2 \times 2$ matrix, the cofactor matrix has the same two singular values as the original matrix, in reversed order. Its nuclear norm is therefore at most $2 \left\| { \hat { H } } \right\| _ { 2 } ^ { \infty }$ . The matrix dual-norm inequality and | det $E | \leq \| E \| _ { 2 } ^ { 2 }$ imply (20). The triangle and reverse triangle inequalities give (21). Since both Hessians are symmetric, the Rayleigh-quotient bound $v ^ { \top } E v \geq - \| E \| _ { 2 }$ for unit v proves (22).

## F Proofs of the regional and solution-error bounds

## F.1 Proof of Proposition 6.1

Proof. Apply Theorem 5.1 separately at every $x \in X$ with the same constant candidates. Every expression for an error magnitude is nondecreasing in its nonnegative upper-bound inputs, with a strictly positive lower bound for $1 - q$ . Hence uniform residual upper bounds give uniform output bounds. The determinant claim is the triangle inequality. Interval evaluation provides the required simultaneous enclosures; it is not an assumption that a floating-point residual is already an upper bound. □

## F.2 Proof of Corollary 6.2

Proof. Every point belongs to a certified covering element, so the appropriate local bound is at most the maximum. Parameterization of the circle transfers the angular bounds to the whole boundary. □

## F.3 Proof of Proposition 6.3

Proof. Set $b ( \boldsymbol { x } ) = \sqrt { R } ( \| \boldsymbol { x } \| _ { 2 } ^ { 2 } - 1 ) / 2$ . It vanishes on the circle, is convex, and satisfies $- \sqrt { R } / 2 \leq b \leq 0$ For every positive semidefinite $2 \times 2$ matrix $H$

$$
\operatorname* { d e t } ( H + { \sqrt { R } } I ) = \operatorname* { d e t } H + { \sqrt { R } } \operatorname { t r } H + R \geq \operatorname* { d e t } H + R .
$$

Thus $u + b - B _ { \partial }$ is convex, has boundary values at most those of $v ,$ and has determinant at least det $D ^ { 2 } u + R \geq \operatorname* { d e t } D ^ { 2 } v$ . The convex Monge–Ampère comparison principle gives $u + b - B _ { \partial } \leq v$ Interchanging u, v gives $v + b - B _ { \partial } \leq u$ . The two inequalities imply the result. The comparison principle is standard; see Figalli (2017, Theorem 2.10) or Nyström & Vestberg (2023, Theorem 2.2). □

## G Finite-unroll derivative recurrences

The certificate is independent of how its approximate quantities are obtained. For completeness, derivatives of the finite iteration can be computed without symbolic diferentiation. Start with $z _ { 0 } = 0 , J _ { 0 } = 0$ and $K _ { i j , 0 } = 0$ . At step $\ell ,$ define

$$
s _ { \ell } = W z _ { \ell } + A x + b , \quad d _ { \ell } = \phi ^ { \prime } ( s _ { \ell } ) , \quad \gamma _ { \ell } = \phi ^ { \prime \prime } ( s _ { \ell } ) , \quad v _ { i , \ell } = A _ { : , i } + W J _ { i , \ell } .
$$

Then the chain rule gives

$$
\begin{array} { r l } & { z _ { \ell + 1 } = \phi ( s _ { \ell } ) , } \\ & { J _ { i , \ell + 1 } = d _ { \ell } \odot v _ { i , \ell } , } \\ & { K _ { i j , \ell + 1 } = d _ { \ell } \odot ( W K _ { i j , \ell } ) + \gamma _ { \ell } \odot v _ { i , \ell } \odot v _ { j , \ell } . } \end{array}
$$

At depth $L ,$ take $\hat { z } = z _ { L } , \hat { J } _ { i } = J _ { i , I }$ and $\hat { K } _ { i j } = K _ { i j , L }$ . The certificate coeficients must be recomputed at ${ \hat { s } } = W z _ { L } + A x + b ;$ they are not automatically the coeficients from the preceding unroll step. For a finite unroll, the derivative residuals in (12)–(13) are generally nonzero.