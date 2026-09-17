# Learning Lyapunov Operators for Nonlinear Systems

Amartya Mukherjee, Maxwell Fitzsimmons, David C. Del Rey Fernandez, and Jun Liu´

Abstract— Constructing Lyapunov functions for nonlinear dynamical systems is a central problem in stability analysis, yet remains challenging. Lyapunov functions are commonly characterized as solutions to first-order partial differential equations (PDEs), but these solutions are typically obtained for single systems, limiting their reuse across systems. In this paper, we study the Lyapunov solution operator that maps a vector field to the corresponding Lyapunov function defined by a dissipation-based Lyapunov PDE. We establish that, on compact subsets of the domain of attraction and under exponential stability assumptions, this operator is well-defined, unique, and continuous with respect to perturbations of both the vector field and the dissipation function. These results provide a theoretical foundation for approximating Lyapunov functions uniformly over families of nonlinear systems. Building on these theoretical foundations, we employ Fourier Neural Operators (FNOs) as a data-driven approximation of the Lyapunov solution operator. Numerical experiments demonstrate that a single trained operator can accurately approximate the numerical Lyapunov functions across parameterized families of dynamics. This illustrates the potential of neural operators for approximating Lyapunov functions.

## I. INTRODUCTION

One of the longstanding challenges in nonlinear systems and control is the construction of Lyapunov functions [4]. While Lyapunov functions can essentially be characterized by solutions to partial differential equations (PDEs) and neural network solutions to such PDEs can effectively provide approximations to Lyapunov functions [9], solving PDEs for each system can still be time-consuming. Additionally, prior works that use neural networks to learn Lyapunov functions only aid in the verification of a single system [11], thus failing to generalize into systems with slightly different dynamics. This could pose difficulties in real-life systems.

Recently, operator learning has emerged as a paradigm for approximating mappings between function spaces, such as those defined by PDEs. Unlike conventional deep learning architectures that learn finite-dimensional mappings, neural operators generalize across function spaces, allowing them to learn solution operators of PDEs from data. This perspective is particularly attractive for Lyapunov analysis: the correspondence between vector fields and their associated Lyapunov functions can be viewed as an operator defined by a PDE constraint. Approximating this operator directly offers the promise of learning a single model that can compute Lyapunov functions for a large class of systems.

The Fourier Neural Operator (FNO) is an example of a neural operator [5], [7]. It lifts the input functions to a higher-dimensional feature space using a linear layer, followed by repeated applications of Fourier convolution layers. Each layer performs a fast Fourier transform (FFT), applies learnable filters in the frequency domain, and then inverts the transform to return to the spatial domain. This global convolution mechanism enables FNO to efficiently capture long-range dependencies in the input.

In this paper, we bridge the perspectives of Lyapunov stability and operator learning. We begin by formulating the Lyapunov stability condition as a PDE, thereby recasting the problem into the operator learning framework. We then establish regularity and continuity results for this PDE, showing that the assumptions required for the FNO universality theorem [6] hold in our setting. This provides theoretical justification for approximating Lyapunov operators using FNOs. We finally train an FNO on nonlinear systems and demonstrate that it can generate Lyapunov functions with small error with respect to the true Lyapunov function.

## II. PROBLEM FORMULATION

We consider autonomous nonlinear dynamical systems of the form

$$
{ \dot { x } } ( t ) = f ( x ( t ) ) , \quad x \in \mathbb { R } ^ { n } ,\tag{1}
$$

where the vector field $f : \mathbb { R } ^ { n } \to \mathbb { R } ^ { n }$ is continuously differentiable and satisfies $f ( 0 ) = 0 .$ . Let $\phi _ { f } ( t , \boldsymbol { x } )$ denote the flow of the system initialized at x at time $t = 0$

## A. Admissible class of dynamics

Define the function space

$$
\mathcal { F } _ { 0 } : = \{ f \in C ^ { 1 } ( \mathbb { R } ^ { n } ; \mathbb { R } ^ { n } ) \mid f ( 0 ) = 0 \} ,
$$

and the subclass of locally exponentially stable vector fields

$$
\mathcal { F } _ { s t } : =  f \in \mathcal { F } _ { 0 } | \ \frac { \partial f } { \partial x } ( 0 ) \ \mathrm { i s \ H u r w i t z }  .
$$

For $f \in \mathcal { F } _ { s t }$ , classical results guarantee that the origin is a locally exponentially stable equilibrium and that solutions exist and are unique in a neighborhood of the origin.

The domain of attraction of the origin is defined as

$$
D O A ( f ) : = \{ x \in \mathbb { R } ^ { n } \ | \operatorname* { l i m } _ { t \to \infty } \phi _ { f } ( t , x ) = 0 \} .\tag{2}
$$

A set Ω is called positively invariant if $\phi _ { f } ( t , x ) \in \Omega$ for all $t \geq 0$ and $x \in \Omega$

## B. Dissipation functions and Lyapunov PDE

Let $\omega : \mathbb { R } ^ { n }  \mathbb { R }$ be a twice continuously differentiable function satisfying

$$
\pmb { \omega } ( 0 ) = 0 , \quad \nabla \pmb { \omega } ( 0 ) = 0 , \quad \nabla ^ { 2 } \pmb { \omega } \succ 0 ,\tag{3}
$$

where $\nabla ^ { 2 }$ ω denotes the Hessian matrix. Define the admissible class

$$
\mathcal { W } _ { > 0 } : = \left\{ \omega \in C ^ { 2 } ( \mathbb { R } ^ { n } , \mathbb { R } ) \ \middle \vert \ \omega \ \mathrm { s a t i s f i e s ~ c o n d i t i o n s ~ } ( 3 ) \right\} .
$$

Given $f \in \mathcal { F } _ { s t }$ and $\omega \in { \mathcal { W } } _ { > 0 }$ , we consider Lyapunov functions defined as solutions of the first-order PDE

$$
\nabla V ( x ) \cdot f ( x ) = - \omega ( x ) , \quad V ( 0 ) = 0 .\tag{4}
$$

Equations of this form arise naturally in converse Lyapunov theory and characterize dissipation-based Lyapunov functions.

In applications, local Lyapunov functions are found to prove local asymptotic/exponential stability, but they are also used to estimate the size of $D O A ( f )$ . This is because if V is a Lyapunov function for f on $K = \{ x \in \mathbb { R } ^ { n } : V ( x ) < r \}$ for a fixed $r ,$ then K is positively invariant and $K \subset D O A ( f )$ Moreover, there is a Lyapunov function $V _ { 1 }$ for f such that $\{ x \in \mathbb { R } ^ { n } : V _ { 1 } ( x ) < r \} = D O A ( f )$ [15], [16].

## C. Integral representation and well-posedness

Let $K \subset D O A ( f )$ be compact and positively invariant. Under this assumption, the Lyapunov PDE (4) admits the integral representation

$$
V _ { f , \omega } ( x ) : = \int _ { 0 } ^ { \infty } \omega { \bigl ( } \phi _ { f } ( t , x ) { \bigr ) } d t , \quad x \in K ,\tag{5}
$$

which is well defined due to the exponential decay of trajectories and the regularity of ω.

Proposition 1 (Existence and uniqueness): Let $f \in \mathcal { F } _ { s t } , \omega \in$ $\mathcal { W } _ { > 0 } .$ , and let $K \subset D O A ( f )$ be compact and positively invariant. Then the following are equivalent:

1) There exists a unique function $V \in C ^ { 1 } ( K )$ satisfying (4)

2) This solution is given by the integral formula (5).

Proof: If (1) holds, then we see that

$$
\frac { d } { d t } \big ( V _ { f , \omega } \circ \phi _ { f } ( t , x ) \big ) = - \omega \circ \phi _ { f } ( t , x )
$$

for $t \geq 0$ and $x \in K$ . Integrating from 0 to T, we find

$$
V _ { f , \omega } \circ \phi _ { f } ( T , x ) - V _ { f , \omega } ( x ) = \int _ { 0 } ^ { T } - \omega \circ \phi _ { f } ( s , x ) d s .
$$

We note that, as ω is Lipschitz, the integrated solution is unique. Taking the limit as $T \to \infty$ yields the result, after noting that $\phi _ { f } ( T , x ) \to 0 { \mathrm { ~ i f ~ } } x \in K$ and $V ( 0 ) = 0$

If (2) holds then $\begin{array} { r } { V _ { f , \omega } \circ \phi _ { f } ( T , x ) - V _ { f , \omega } ( x ) = \int _ { 0 } ^ { T } - \omega \circ } \end{array}$ $\phi _ { f } ( s , x ) d s$ . By the mean value theorem for integrals, we obtain

$$
V _ { f , \omega } \circ \phi _ { f } ( T , x ) - V _ { f , \omega } ( x ) = - T \omega \circ \phi _ { f } ( s ^ { \prime } , x )
$$

for $s ^ { \prime } \in [ 0 , T ]$ . Dividing both sides by T and taking the limit $T  0$ yields the result. Finally, V is continuously differentiable on K as proved by [9]. ■

As a consequence, $V _ { f , \omega }$ is a Lyapunov function for f on K, and its sublevel sets define invariant subsets of the domain of attraction.

## D. The Lyapunov solution operator

Rather than constructing Lyapunov functions on a per-system basis, we adopt an operator-theoretic viewpoint. Define the Lyapunov solution operator

$$
\boldsymbol { \mathcal { G } } : \mathcal { F } _ { s t } \times \mathcal { W } _ { > 0 } , \quad \boldsymbol { \mathcal { G } } ( \boldsymbol { f } , \omega ) : = V _ { \boldsymbol { f } , \omega } ,
$$

mapping a vector field and dissipation function to the corresponding Lyapunov function. The central objective of this work is to establish continuity of this operator on compact subsets of the domain of attraction.

## E. Sobolev regularity of the data and solution

The universality results for FNOs are formulated for operators acting between Sobolev spaces, while Proposition 1 defines solutions to the Lyapunov PDE in the $C ^ { 1 }$ and $C ^ { 2 }$ space. To place the Lyapunov solution operator within this framework, it is necessary to ensure that both the input and output functions admit sufficient Sobolev regularity. The Sobolev regularity results here are standard in the analysis of PDEs and play a central role in neural operator theory [1]. They ensure that the Lyapunov PDE (4) is well defined pointwise while simultaneously allowing us to work in function spaces compatible with neural operator approximation. The next two theorems ensure that the solutions to the Lyapunov PDE can be embedded in appropriate Sobolev spaces.

Theorem 1 (Sobolev embedding [13]): Let $\Omega \subset \mathbb { R } ^ { n }$ be a bounded open domain with Lipschitz boundary, and let $k \geq 0$ be an integer. ${ \mathrm { ~ I f ~ } } s > n / 2 + k$ , then the Sobolev space $H ^ { s } ( \Omega )$ is continuously and compactly embedded in $C ^ { k } ( { \overline { { \Omega } } } )$ i.e., $H ^ { s } ( \Omega ) \hookrightarrow C ^ { k } ( \overline { { \Omega } } )$ . Moreover, there exists a constant $C > 0$ such that

$$
\| u \| _ { C ^ { k } ( \overline { { \Omega } } ) } \leq C \| u \| _ { H ^ { s } ( \Omega ) } , \quad \forall u \in H ^ { s } ( \Omega ) .
$$

Corollary 1 (Sobolev embedding for vector-valued functions): Let $\Omega \subset \mathbb { R } ^ { n }$ be a bounded open domain with Lipschitz boundary, and let $k \geq 0$ be an integer. If $s > n / 2 + k ,$ then the Sobolev space $H ^ { s } ( \Omega ; \mathbb { R } ^ { n } )$ consisting of $\mathbb { R } ^ { n }$ -valued functions with s-Sobolev regularity is continuously and compactly embedded in $C ^ { k } ( \overline { { \Omega } } ; \mathbb { R } ^ { n } )$ , i.e., $H ^ { s } ( \Omega ; \mathbb { R } ^ { n } ) \hookrightarrow C ^ { k } ( \bar { \Omega } ; \mathbb { R } ^ { n } )$ Moreover, there exists a constant $C > 0$ such that

$$
\| u \| _ { C ^ { k } ( \overline { { \Omega } } ; \mathbb { R } ^ { n } ) } \leq C \| u \| _ { H ^ { s } ( \Omega ; \mathbb { R } ^ { n } ) } , \quad \forall u \in H ^ { s } ( \Omega ; \mathbb { R } ^ { n } ) .
$$

The proof of Corollary 1 follows directly from the scalar case by applying the component-wise argument.

## III. CONTINUITY AND UNIVERSALITY OF THELYAPUNOV SOLUTION OPERATOR

In this section, we establish the main theoretical result of the paper: continuity of the Lyapunov solution operator with respect to perturbations of the vector field and the dissipation function. This property is essential for approximating Lyapunov functions uniformly over families of nonlinear systems.

## A. FNO universality theorem

The work of [6] provides the key conditions imposed on the input functions and output function of an operator so that the FNO universality theorem applies. In our setting, the operator is the Lyapunov operator, which parametrizes the Lyapunov PDE over the vector field f and the dissipation function ω. To apply the universality result, we must verify the following conditions:

1) the input vector fields belong to a Sobolev space $H ^ { s } ( \Omega ; \mathbb { R } ^ { n } )$ and the dissipation functions belong to a Sobolev space $H ^ { r } ( \Omega )$ with sufficiently high regularity;

2) the corresponding Lyapunov functions belong to a Sobolev space $H ^ { s ^ { \prime } } ( \Omega )$

3) the Lyapunov solution operator is continuous on compact subsets of the product of the input spaces.

Under these assumptions, the modified universality theorem of [6] for the Lyapunov PDE yields convergent approximation by FNOs. We restrict attention to compact subsets of the admissible classes $\mathcal { F } _ { s t }$ and $\mathcal { W } _ { > 0 } .$ , viewed as subsets of appropriate Sobolev spaces via Sobolev embedding.

Theorem 2 (Modification of Theorem 9 by $I 6 J \}$ Let $\Omega \subset \mathbb { R } ^ { n }$ be a bounded domain with Lipschitz boundary such that ${ \overline { { \Omega } } } \subset ( 0 , 2 \pi ) ^ { n }$ . Let $s , r , s ^ { \prime } \geq 0$ , and let

$$
\mathcal { H } _ { f } \subset \mathcal { F } _ { \mathrm { s t } } \cap H ^ { s } ( \Omega ; \mathbb { R } ^ { n } ) , \quad \mathcal { H } _ { \omega } \subset \mathcal { W } _ { > 0 } \cap H ^ { r } ( \Omega )
$$

be compact sets of admissible vector fields and dissipation functions, respectively. Assume that for each $( f , \omega ) \in \mathcal { K } _ { f } \times$ $\mathcal { H } _ { \omega } .$ , the Lyapunov PDE

$$
\nabla V ( x ) \cdot f ( x ) = - \omega ( x ) , \quad x \in \Omega ,\tag{6}
$$

admits a unique solution $V _ { f , \omega } \in H ^ { s ^ { \prime } } ( \Omega )$ , and that the associated Lyapunov solution operator

$$
\boldsymbol { \mathcal { G } } : \mathcal { H } _ { f } \times \mathcal { H } _ { \omega }  \boldsymbol { H } ^ { s ^ { \prime } } ( \Omega ) , \quad \boldsymbol { \mathcal { G } } ( f , \omega ) = V _ { f , \omega } ,
$$

is continuous with respect to the product topology induced by the $H ^ { s } ( \Omega ; \mathbb { R } ^ { n } )$ and $H ^ { r } ( \Omega )$ norms on the input and the $H ^ { s ^ { \prime } } ( \Omega )$ norm on the output. Let $\mathbb { T } ^ { n }$ denote the n-dimensional torus. Then, for every $\varepsilon > 0 ,$ , there exist

1) continuous linear extension operators

$$
E _ { f } : H ^ { s } ( \Omega ; \mathbb { R } ^ { n } ) \to H ^ { s } ( \mathbb { T } ^ { n } ; \mathbb { R } ^ { n } ) , \quad E _ { \omega } : H ^ { r } ( \Omega ) \to H ^ { r } ( \mathbb { T } ^ { n } ) ,
$$

2) a Fourier Neural Operator

$$
\mathcal { N } _ { \varepsilon } : H ^ { s } ( \mathbb { T } ^ { n } ; \mathbb { R } ^ { n } ) \times H ^ { r } ( \mathbb { T } ^ { n } )  H ^ { s ^ { \prime } } ( \mathbb { T } ^ { n } ) ,
$$

such that

$$
\operatorname* { s u p } _ { ( f , \omega ) \in \mathcal { H } _ { f } \times \mathcal { H } _ { \omega } } \big \| \mathcal { G } ( f , \omega ) - \mathcal { N } _ { \varepsilon } ( E _ { f } f , E _ { \omega } \omega ) \big | _ { \Omega } \big \| _ { H ^ { s ^ { \prime } } ( \Omega ) } < \varepsilon .
$$

In particular, the Lyapunov solution operator can be approximated arbitrarily well on compact subsets of admissible vector fields and dissipation functions by a suitable FNO.

The regularity conditions required for this theorem are satisfied in our setting. Specifically, from the Sobolev embedding results in Section II-E, for $s > n / 2 + 1$ we have $H ^ { s } ( \Omega ; \mathbb { R } ^ { n } ) \hookrightarrow$ $C ^ { 1 } ( \overline { { \Omega } } ; \mathbb { R } ^ { n } )$ , ensuring that vector fields are continuously differentiable. For the dissipation functions, we require $r > n / 2 + 2$ to guarantee $H ^ { r } ( \Omega ) \hookrightarrow C ^ { 2 } ( \overline { { \Omega } } )$ , which is the natural regularity for ω in the Lyapunov PDE. The continuity of the Lyapunov solution operator $\mathcal { G }$ is established in Theorem 3, providing the final ingredient needed to apply the universality result.

## B. Regularity of the Lyapunov PDE

Before turning to continuity, we briefly comment on regularity. Under Section II-E, the Lyapunov PDE admits a unique solution $V _ { f , \omega } \in C ^ { 1 } ( J )$ on compact invariant sets $J \subset D O A ( f )$ In numerical settings, vector fields and Lyapunov functions are often represented in Sobolev spaces. For sufficiently large $s > n / 2 + 1$ , classical Sobolev embedding results ensure continuous and compact embeddings

$$
H ^ { s } ( J ; \mathbb { R } ^ { n } ) \hookrightarrow C ^ { 1 } ( J ; \mathbb { R } ^ { n } ) , \quad H ^ { s } ( J ) \hookrightarrow C ^ { 1 } ( J ) ,
$$

which guarantee that the Lyapunov PDE is well defined in the function spaces used by neural operator architectures.

## C. Main continuity result

We begin by stating the central theorem.

Theorem 3 (Continuity of the Lyapunov solution operator): Let $f \in \mathcal { F } _ { s t }$ and $\omega \in \mathcal { W } _ { > 0 }$ . Let $K \subset D O A ( f )$ be compact. Then there exists a compact set,

$$
K \subset J \subset D O A ( f ) ,
$$

such that the Lyapunov solution operator

$$
\begin{array} { r } { \boldsymbol { \mathcal { G } } : \boldsymbol { \mathcal { F } } _ { s t } \times \boldsymbol { \mathcal { W } } _ { > 0 } \to C ^ { 1 } ( J ) , \quad \boldsymbol { \mathcal { G } } ( g , \boldsymbol { \psi } ) : = V _ { g , \boldsymbol { \psi } } , } \end{array}
$$

is continuous at $( f , \omega )$ when $\mathcal { F } _ { s t } \times \mathcal { W } _ { > 0 }$ is endowed with the norm

$$
\| g \| _ { C ^ { 1 } ( J ) } + \| \psi \| _ { C ^ { 2 } ( J ) } ,
$$

and $C ^ { 1 } ( J )$ is endowed with the uniform norm $\| \cdot \| _ { \infty , J } .$

Remark 1: Theorem 3 formalizes the intuition that small perturbations of the system dynamics and dissipation function induce small changes in the associated Lyapunov function on compact invariant sets. This robustness property provides the theoretical foundation for learning Lyapunov functions uniformly over families of nonlinear systems.

The proof of Theorem 3 proceeds in three steps. First, we establish robustness of exponential stability under perturbations of the vector field. Second, we show uniform decay of the Lyapunov function outside small sublevel sets. Finally, we prove uniform integrability of the Lyapunov integral representation.

## D. Robust exponential stability

We now establish robustness of local exponential stability with respect to perturbations of the vector field.

Lemma 1 (Robust exponential stability): Let $f \in \mathcal { F } _ { s t } , \omega \in$ $\mathcal { W } _ { > 0 }$ . There exist constants $r > 0 , \delta > 0 , M > 0$ , and $c > 0$ such that for any

$$
g \in \mathcal { F } _ { 0 } \quad \mathrm { w i t h } \quad \| g - f \| _ { C ^ { 1 } ( J ) } < r ,
$$

the origin remains locally exponentially stable for ${ \dot { x } } = g ( x )$ and

$$
\begin{array} { r } { \| \phi _ { g } ( t , x ) \| \le M e ^ { - c t } \| x \| , \quad \forall t \ge 0 , \ \| x \| \le \delta . } \end{array}
$$

Proof: Define the ball $B ( x , \eta ) : = \{ y \in \mathbb { R } ^ { n } : \| x - y \| < \eta \}$ and define $\overline { { B } } ( x , \eta )$ as its closure. Choose $\eta > 0$ such that $B ( 0 , \eta ) \subset D O A ( f )$ . Let $J = K \cup { \overline { { B } } } ( 0 , \eta )$ . Let $\begin{array} { r } { A : = \frac { \partial f } { \partial x } ( 0 ) , Q = } \end{array}$ $\nabla ^ { 2 } \omega ( 0 )$ . Since $f \in \mathcal { F } _ { s t } ,$ A is Hurwitz, and since ω $\in { \mathcal { W } } _ { > 0 } ,$ $Q \succ 0$ . Let $P \in \mathbb { R } ^ { n \times n }$ be the positive definite matrix that satisfies [4]

$$
P A + A ^ { T } P = - Q .
$$

Let $\lambda _ { \mathrm { m i n } } ( \cdot )$ and $\lambda _ { \operatorname* { m a x } } ( \cdot )$ refer to the smallest and largest eigenvalues respectively. Define $\begin{array} { r } { W ( x ) : = \frac { 1 } { 2 } x ^ { T } P ( x . } \end{array}$ then $W$ is a local Lyapunov function for $f .$ Let

$$
c _ { 1 } : = \frac { \lambda _ { \operatorname* { m i n } } ( Q ) } { 6 \lambda _ { \operatorname* { m a x } } ( P ) } .
$$

We will show that, for sufficiently small $\delta > 0$ and $r > 0 ,$ the function W is a strict local Lyapunov function for the system ${ \dot { x } } = g ( x )$ whenever $\| g - f \| _ { C ^ { 1 } ( J ) } < r .$

Since $f \in C ^ { 1 }$ , there exists $\delta _ { 1 } > 0$ such that $\delta _ { 1 } \leq \eta$ and

$$
\left\| \xi \right\| \leq \delta _ { 1 } \quad \Longrightarrow \quad \left\| { \frac { \partial f } { \partial x } } ( \xi ) - A \right\| \leq c _ { 1 } .
$$

Choose $r : = c _ { 1 }$ . Now let $g \in \mathcal { F } _ { 0 }$ satisfy $\| g - f \| _ { C ^ { 1 } ( J ) } < r ,$ , and let $\| x \| \leq \delta _ { 1 }$ . Since $g ( 0 ) = 0$ , Taylor’s theorem gives

$$
g ( x ) = \int _ { 0 } ^ { 1 } \frac { \partial g } { \partial x } ( t x ) d t x
$$

Hence,

$$
\begin{array} { l } { \displaystyle \frac { d } { d t } W ( \phi _ { g } ( t , x ) ) \Big | _ { t = 0 } = x ^ { T } P g ( x ) } \\ { = x ^ { T } P \int _ { 0 } ^ { 1 } \frac { \partial g } { \partial x } ( t x ) d t x } \\ { = x ^ { T } P A x + x ^ { T } P \int _ { 0 } ^ { 1 } \left( \frac { \partial g } { \partial x } ( t x ) - \frac { \partial f } { \partial x } ( t x ) \right) d t x } \\ { ~ + x ^ { T } P \int _ { 0 } ^ { 1 } \left( \frac { \partial f } { \partial x } ( t x ) - A \right) d t x . } \end{array}
$$

Since $\begin{array} { r } { x ^ { T } P A x = \frac { 1 } { 2 } x ^ { T } ( P A + A ^ { T } P ) x = - \frac { 1 } { 2 } x ^ { T } Q x , } \end{array}$ , we obtain

$$
\begin{array} { r l } & { \quad \displaystyle \frac { d } { d t } { \cal W } ( \phi _ { g } ( t , x ) ) \Big | _ { t = 0 } } \\ & { \le - \frac 1 2 \lambda _ { \operatorname* { m i n } } ( Q ) \| x \| ^ { 2 } + \| P \| \left\| \frac { \partial g } { \partial x } - \frac { \partial f } { \partial x } \right\| _ { \infty } \| x \| ^ { 2 } } \\ & { \quad + \| P \| \left\| \frac { \partial f } { \partial x } - A \right\| _ { \infty } \| x \| ^ { 2 } } \\ & { \le - \frac 1 2 \lambda _ { \operatorname* { m i n } } ( Q ) \| x \| ^ { 2 } + \| P \| r \| x \| ^ { 2 } + \| P \| \frac { \lambda _ { \operatorname* { m i n } } ( Q ) } { 6 \| P \| } \| x \| ^ { 2 } } \\ & { \le - \frac 1 2 \lambda _ { \operatorname* { m i n } } ( Q ) \| x \| ^ { 2 } + \frac 1 6 \lambda _ { \operatorname* { m i n } } ( Q ) \| x \| ^ { 2 } + \frac 1 6 \lambda _ { \operatorname* { m i n } } ( Q ) \| x \| ^ { 2 } } \\ & { = - \frac 1 6 \lambda _ { \operatorname* { m i n } } ( Q ) \| x \| ^ { 2 } . } \end{array}
$$

Using $2 W ( x ) \leq \lambda _ { \operatorname* { m a x } } ( P ) \| x \| ^ { 2 }$ , it follows that

$$
\frac { d } { d t } W ( \phi _ { g } ( t , x ) ) \Big | _ { t = 0 } \leq - \frac { \lambda _ { \operatorname* { m i n } } ( Q ) } { 3 \lambda _ { \operatorname* { m a x } } ( P ) } W ( x ) = - 2 c W ( x )
$$

for all $\| x \| \leq \delta _ { 1 }$

Therefore, along any trajectory of ${ \dot { x } } = g ( x )$ that remains in $B ( 0 , \delta _ { 1 } )$

$$
\frac { d } { d t } W \big ( \phi _ { g } ( t , x ) \big ) \leq - 2 c W \big ( \phi _ { g } ( t , x ) \big ) .
$$

By Gronwall’s inequality,¨

$$
W ( \phi _ { g } ( t , x ) ) \leq e ^ { - 2 c t } W ( x ) , \quad t \geq 0 .
$$

In particular, since W is decreasing, the sublevel set

$$
\begin{array} { r } { \Omega _ { \delta } : = \{ x \in \mathbb { R } ^ { n } : W ( x ) \leq \frac { 1 } { 2 } \lambda _ { \operatorname* { m i n } } ( P ) \delta ^ { 2 } \} } \end{array}
$$

is positively invariant whenever $\delta \leq \delta _ { 1 }$ . Choose $\delta = \delta _ { 1 }$ . Then every trajectory with $\| x \| \leq \delta$ remains in $B ( 0 , { \delta } ) \subset B ( 0 , \eta ) \subset$ J for all $t \geq 0 ,$ , so the above estimate is valid globally in time.

Finally, using the bounds relating W and $\| x \| ^ { 2 }$

$$
\begin{array} { r l r } {  { \| \phi _ { g } ( t , x ) \| ^ { 2 } \leq \frac { 2 } { \lambda _ { \operatorname* { m i n } } ( P ) } W ( \phi _ { g } ( t , x ) ) } } \\ & { } & { \leq \frac { 2 } { \lambda _ { \operatorname* { m i n } } ( P ) } e ^ { - 2 c t } W ( x ) } \\ & { } & { \leq \frac { \lambda _ { \operatorname* { m a x } } ( P ) } { \lambda _ { \operatorname* { m i n } } ( P ) } e ^ { - 2 c t } \| x \| ^ { 2 } . } \end{array}
$$

Hence,

$$
\| \phi _ { g } ( t , x ) \| \leq \sqrt { \frac { \lambda _ { \operatorname* { m a x } } ( P ) } { \lambda _ { \operatorname* { m i n } } ( P ) } } e ^ { - c t } \| x \| .
$$

Setting

$$
M : = \sqrt { \frac { \lambda _ { \operatorname* { m a x } } ( P ) } { \lambda _ { \operatorname* { m i n } } ( P ) } } ,
$$

we obtain

$$
\begin{array} { r } { \| \phi _ { g } ( t , x ) \| \le M e ^ { - c t } \| x \| , \quad \forall t \ge 0 , \ \| x \| \le \delta . } \end{array}
$$

This proves local exponential stability of the origin for ${ \dot { x } } =$ g(x).

## E. Uniform decay of the Lyapunov derivative

We next show that the Lyapunov function $V _ { f , \omega }$ decreases uniformly along trajectories outside small sublevel sets.

Lemma 2 (Uniform Lyapunov decay): Let $f \in \mathcal { F } _ { s t } , \omega \in$ $\mathcal { W } _ { > 0 } ,$ , and let $J \subset D O A ( f )$ be compact. Define $U ( \delta ) : = \{ x \in$ $\mathbb { R } ^ { n } : V _ { f , \omega } ( x ) < \delta \}$ . Then there exists $\delta _ { 0 } > 0$ such that for every $\delta \in ( 0 , \delta _ { 0 } )$ , there exist constants $c _ { \delta } > 0$ and $r _ { \delta } > 0$ such that for all

$$
g \in \mathcal { F } _ { 0 } \quad \mathrm { w i t h } \quad \| g - f \| _ { C ^ { 1 } ( J ) } < r _ { \delta } ,
$$

the inequality

$$
\nabla V _ { f , \omega } ( x ) \cdot g ( x ) \leq - c _ { \delta }
$$

holds for all $x \in J \backslash U ( \delta )$

Proof: Let $Q : = \nabla ^ { 2 } \omega ( 0 ) \succ 0$ , and denote $\lambda _ { \operatorname* { m i n } } ( Q ) > 0$ its smallest eigenvalue.

Let

$$
R : = \operatorname* { m a x } _ { x \in J } V _ { f , \omega } ( x ) , \quad K _ { 2 } : = \{ x \in \mathbb { R } ^ { n } : V _ { f , \omega } ( x ) \leq R \} .
$$

Then $K _ { 2 }$ is compact, positively invariant under $f ,$ and satisfies

$$
J \subseteq K _ { 2 } \subseteq D O A ( f ) .
$$

We first establish a lower bound on ω near the origin. Since $\omega \in C ^ { 2 }$ and $\nabla ^ { 2 } \omega ( 0 ) = Q \succ 0$ , there exists $\delta _ { 1 } > 0$ such that for all $\| x \| \leq \delta _ { 1 }$

$$
\| \nabla ^ { 2 } \omega ( x ) - Q \| \leq \frac { \lambda _ { \operatorname* { m i n } } ( Q ) } { 2 } .
$$

By Taylor’s theorem, for such x,

$$
\pmb { \omega } ( x ) = \frac { 1 } { 2 } x ^ { T } \nabla ^ { 2 } \pmb { \omega } ( \xi ) x
$$

for some $\xi$ on the segment between 0 and x. Hence,

$$
\begin{array} { r l } & { \omega ( x ) = x ^ { T } Q x + x ^ { T } \big ( \nabla ^ { 2 } \omega ( \xi ) - Q \big ) x } \\ & { \qquad \geq \lambda _ { \operatorname* { m i n } } ( Q ) \| x \| ^ { 2 } - \| \nabla ^ { 2 } \omega ( \xi ) - Q \| \| x \| ^ { 2 } } \\ & { \qquad \geq \displaystyle \frac { \lambda _ { \operatorname* { m i n } } ( Q ) } { 2 } \| x \| ^ { 2 } . } \end{array}\tag{7}
$$

Since $V _ { f , \omega }$ is continuous and $V _ { f , \omega } ( 0 ) = 0$ , there exists $\delta _ { 0 } > 0$ such that for all $\delta \in ( 0 , \delta _ { 0 } )$ ,

$$
U ( \delta ) \subset B ( 0 , \delta _ { 1 } ) .
$$

Fix such a δ, then for $x \in B ( 0 , \delta _ { 1 } ) \backslash U ( \delta )$ , we have $V _ { f , \omega } ( x ) \geq$ δ. By Taylor’s theorem, followed by Cauchy-Schwarz inequality,

$$
\delta \leq \| \nabla V _ { f , \omega } \| _ { \infty , K _ { 2 } } \| x \| \quad \Longrightarrow \quad \| x \| \geq { \frac { \delta } { \| \nabla V _ { f , \omega } \| _ { \infty , K _ { 2 } } } } .
$$

Combining with (7), we obtain

$$
\boldsymbol { \omega } ( \boldsymbol { x } ) \geq \frac { \lambda _ { \operatorname* { m i n } } ( Q ) } { 2 } \frac { \delta ^ { 2 } } { \Vert \nabla V _ { f , \omega } \Vert _ { \infty , K _ { 2 } } ^ { 2 } } , \quad \boldsymbol { x } \in B ( 0 , \delta _ { 1 } ) \setminus U ( \delta ) .\tag{8}
$$

Next, we establish a global lower bound on ω outside $U ( \delta )$ To separate our analysis into large and small level sets, we decompose the set

$$
K _ { 2 } \setminus U ( \delta ) = \left( K _ { 2 } \setminus B ( 0 , \delta _ { 1 } ) \right) \cup \big ( B ( 0 , \delta _ { 1 } ) \setminus U ( \delta ) \big ) .
$$

Since ω is continuous and strictly positive, the minimum away from the origin,

$$
m _ { 1 } : = \operatorname* { m i n } _ { x \in K _ { 2 } \backslash B ( 0 , \delta _ { 1 } ) } \omega ( x ) ,
$$

is strictly positive. Combining this with (8), we obtain

$$
\operatorname* { m i n } _ { x \in K _ { 2 } \setminus U ( \delta ) } \omega ( x ) \geq \operatorname* { m i n } \left\{ m _ { 1 } , \frac { \lambda _ { \operatorname* { m i n } } ( Q ) } { 2 } \frac { \delta ^ { 2 } } { \| \nabla V _ { f , \omega } \| _ { \infty , K _ { 2 } } ^ { 2 } } \right\} .
$$

Define

$$
c _ { \delta } : = \frac { 1 } { 2 } \operatorname* { m i n } \left\{ m _ { 1 } , \frac { \lambda _ { \operatorname* { m i n } } ( Q ) } { 2 } \frac { \delta ^ { 2 } } { \| \nabla V _ { f , \omega } \| _ { \infty , K _ { 2 } } ^ { 2 } } \right\} > 0 .
$$

Then,

$$
\operatorname* { m i n } _ { x \in K _ { 2 } \backslash U ( \delta ) } \omega ( x ) \geq 2 c _ { \delta } .\tag{9}
$$

Finally, we prove the main perturbation argument. Let

$$
r _ { \delta } : = \frac { c _ { \delta } } { \| \nabla V _ { f , \omega } \| _ { \infty , K _ { 2 } } } .
$$

Let $g \in \mathcal { F } _ { 0 }$ satisfy $\| g - f \| _ { C ^ { 1 } ( J ) } < r _ { \delta }$ . Then, for all $x \in K _ { 2 } \ \backslash$ $U ( \delta )$

$$
\begin{array} { r l } & { \nabla V _ { f , \omega } ( x ) \cdot g ( x ) = - \omega ( x ) + \nabla V _ { f , \omega } ( x ) \cdot \left( g ( x ) - f ( x ) \right) } \\ & { \qquad \leq - 2 c _ { \delta } + \| \nabla V _ { f , \omega } \| _ { \infty , K _ { 2 } } \| g - f \| _ { \infty , K _ { 2 } } } \\ & { \qquad < - 2 c _ { \delta } + \| \nabla V _ { f , \omega } \| _ { \infty , K _ { 2 } } r _ { \delta } } \\ & { \qquad = - c _ { \delta } . } \end{array}
$$

This proves the result.

## F. Uniform integrability of the Lyapunov integral

We now establish convergence and robustness of the integral representation.

Lemma 3 (Uniform integrability): Let $f \in \mathcal { F } _ { s t }$ and $\omega \in$ $\mathcal { W } _ { > 0 }$ . There exist constants $\delta > 0 , \ r > 0$ , and $C > 0$ such that for all

$g \in \mathcal { F } _ { 0 } , \ \psi \in \mathcal { W } _ { > 0 }$ with $\begin{array} { r } { \| g - f \| _ { C ^ { 1 } ( J ) } + \| \psi - \omega \| _ { C ^ { 2 } ( J ) } < r , } \end{array}$ the integral

$$
V _ { g , \psi } ( x ) = \int _ { 0 } ^ { \infty } \psi ( \phi _ { g } ( t , x ) ) d t
$$

is well defined for all $x \in J ,$ and

$$
\int _ { T } ^ { \infty } | \psi ( \phi _ { g } ( t , x ) ) | d t \leq C \delta
$$

for all $x \in J$ and all sufficiently large T.

Proof: Fix $\delta > 0$ sufficiently small as in Lemma 2, and define

$$
U ( \delta ) : = \{ x \in \mathbb { R } ^ { n } : V _ { f , \omega } ( x ) < \delta \} .
$$

Let $x \in K _ { 2 }$ and define the hitting time

$$
T ( \delta , x , g ) : = \operatorname* { i n f } \{ t \geq 0 : V _ { f , \omega } ( \phi _ { g } ( t , x ) ) \leq \delta \} .
$$

From Lemma 2, we have

$$
\cfrac { d } { d t } V _ { f , \omega } \big ( \phi _ { g } \big ( t , x \big ) \big ) \leq - c _ { \delta } \quad \mathrm { f o r } t \in [ 0 , T ( \delta , x , g ) ] .
$$

Integrating and using $V _ { f , \omega } ( \phi _ { g } ( T ( \delta , x , g ) , x ) ) = \delta$ , we obtain

$$
T ( \delta , x , g ) \leq \frac { V _ { f , \omega } ( x ) - \delta } { c _ { \delta } } .
$$

Since $V _ { f , \omega }$ is bounded on $K _ { 2 }$ , there exists a constant $T _ { \delta } > 0$ such that

$$
T ( \delta , x , g ) \leq T _ { \delta } \quad \mathrm { f o r ~ a l l ~ } x \in K _ { 2 } .
$$

By construction, $U ( \delta ) \subset B ( 0 , \delta _ { 1 } )$ for sufficiently small δ. From Lemma 1, there exist constants $M _ { 1 } > 0$ and $c _ { 1 } > 0$ such that for all $t \ge T \bigl ( \delta , x , g \bigr )$ ，

$$
\begin{array} { r } { \| \phi _ { g } ( t , x ) \| \le M _ { 1 } e ^ { - c _ { 1 } ( t - T ( \delta , x , g ) ) } \| \phi _ { g } ( T ( \delta , x , g ) , x ) \| . } \end{array}
$$

Since $\phi _ { g } ( T ( \delta , x , g ) , x ) \in U ( \delta )$ , we have

$$
\begin{array} { r } { \| \phi _ { g } ( T ( \delta , x , g ) , x ) \| \le \delta , } \end{array}
$$

and therefore

$$
\begin{array} { r } { \| \phi _ { g } ( t , x ) \| \le M _ { 1 } \delta e ^ { - c _ { 1 } ( t - T ( \delta , x , g ) ) } , \quad t \ge T ( \delta , x , g ) . } \end{array}\tag{10}
$$

Since ψ lies in a bounded subset of $C ^ { 2 } ( K _ { 2 } )$ , its gradient is uniformly bounded on $K _ { 2 }$ . Hence, there exists a constant $L > 0$ such that

$$
| \psi ( y ) | \leq L \| y \| , \quad \forall y \in K _ { 2 } .
$$

Using (10), for $t \ge T \bigl ( \delta , x , g \bigr )$

$$
| \psi ( \phi _ { g } ( t , x ) ) | \leq L \| \phi _ { g } ( t , x ) \| \leq L M _ { 1 } \delta e ^ { - c _ { 1 } \left( t - T ( \delta , x , g ) \right) } .
$$

Therefore,

$$
\int _ { T ( \delta , x , g ) } ^ { \infty } | \psi ( \phi _ { g } ( t , x ) ) | d t \leq L M _ { 1 } \delta \int _ { 0 } ^ { \infty } e ^ { - c _ { 1 } s } d s = \frac { L M _ { 1 } } { c _ { 1 } } \delta .
$$

Define $\begin{array} { r } { C : = \frac { L M _ { 1 } } { c _ { 1 } } } \end{array}$ . Then for all $x \in K _ { 2 }$

$$
\int _ { T ( \delta , x , g ) } ^ { \infty } \vert \psi ( \phi _ { g } ( t , x ) ) \vert d t \leq C \delta .
$$

Since $T ( \delta , x , g )$ is uniformly bounded over $K _ { 2 }$ , the integral defining $V _ { g , \psi } ( x )$ is finite for all $x \in J .$ ■

## G. Proof of Theorem 3

Proof: Let $K \subset D O A ( f )$ be compact. Define

$$
R : = \operatorname* { m a x } _ { x \in K } V _ { f , \omega } ( x ) , \quad J : = \{ x \in \mathbb { R } ^ { n } : V _ { f , \omega } ( x ) \leq R \} .
$$

Then J is compact, positively invariant under $f ,$ and satisfies

$$
K \subset J \subset D O A ( f ) .
$$

Fix $\varepsilon > 0$ . We will show that for $( g , \psi )$ sufficiently close to $( f , \omega )$ in $C ^ { 1 } ( J ) \times C ^ { 2 } ( J )$

$$
\| V _ { g , \psi } - V _ { f , \omega } \| _ { \infty , J } < \varepsilon .
$$

For $x \in J ,$ we use the integral representation

$$
V _ { f , \omega } ( x ) - V _ { g , \psi } ( x ) = \int _ { 0 } ^ { \infty } \left[ \omega ( \phi _ { f } ( t , x ) ) - \psi ( \phi _ { g } ( t , x ) ) \right] d t .
$$

We decompose the integrand:

$$
\begin{array} { r l } & { | \omega ( \phi _ { f } ( t , x ) ) - \psi ( \phi _ { g } ( t , x ) ) | \leq | \omega ( \phi _ { f } ( t , x ) ) - \omega ( \phi _ { g } ( t , x ) ) | } \\ & { \qquad + | \omega ( \phi _ { g } ( t , x ) ) - \psi ( \phi _ { g } ( t , x ) ) | . } \end{array}
$$

Fix $\delta > 0$ and define $\begin{array} { r } { T _ { \delta } : = \operatorname* { s u p } _ { x \in J } T ( \delta , x , g ) } \end{array}$ , which is finite by Lemma 3.

For $t \in [ 0 , T _ { \delta } ]$ , the flows $\phi _ { f }$ and $\phi _ { g }$ remain in J. By Gronwall¨ inequality, there exists $C _ { 1 } > 0$ such that

$$
\operatorname* { s u p } _ { x \in J , t \in [ 0 , T _ { \delta } ] } \Vert \phi _ { f } ( t , x ) - \phi _ { g } ( t , x ) \Vert \leq C _ { 1 } \Vert f - g \Vert _ { C ^ { 1 } ( J ) } .
$$

Since $\omega \in C ^ { 1 } ( J )$ , it is Lipschitz on J, so there exists $L _ { \omega } > 0$ such that

$$
| \omega ( { \phi } _ { f } ( t , x ) ) - \omega ( { \phi } _ { g } ( t , x ) ) | \leq L _ { \omega } \| { \phi } _ { f } ( t , x ) - { \phi } _ { g } ( t , x ) \| .
$$

Hence,

$$
\int _ { 0 } ^ { T _ { \delta } } | \omega ( \phi _ { f } ( t , x ) ) - \omega ( \phi _ { g } ( t , x ) ) | d t \leq C _ { 2 } \| f - g \| _ { C ^ { 1 } ( J ) }
$$

for some constant $C _ { 2 } > 0$

Similarly,

$$
\int _ { 0 } ^ { T _ { \delta } } | \omega ( \phi _ { g } ( t , x ) ) - \psi ( \phi _ { g } ( t , x ) ) | d t \leq T _ { \delta } \| \omega - \psi \| _ { \infty , J } .
$$

For $t \ge T \bigl ( \delta , x , g \bigr )$ , Lemma 3 yields

$$
\int _ { T ( \delta , x , g ) } ^ { \infty } \vert \psi ( \phi _ { g } ( t , x ) ) \vert d t \leq C \delta .
$$

Applying the same argument to $( f , \omega )$

$$
\int _ { T ( \delta , x , f ) } ^ { \infty } \vert \omega ( \phi _ { f } ( t , x ) ) \vert d t \le C \delta .
$$

Thus, the tail contribution satisfies

$$
\int _ { T _ { \delta } } ^ { \infty } | \omega ( \phi _ { f } ( t , x ) ) - \psi ( \phi _ { g } ( t , x ) ) | d t \leq 2 C \delta .
$$

Combining the estimates, we obtain

$$
\begin{array} { r } { \| \boldsymbol { V } _ { f , \omega } - \boldsymbol { V } _ { g , \psi } \| _ { \infty , J } \leq C _ { 2 } \| \boldsymbol { f } - \boldsymbol { g } \| _ { C ^ { 1 } ( J ) } + T _ { \delta } \| \omega - \psi \| _ { \infty , J } + 2 C \delta . } \end{array}
$$

First choose $\delta > 0$ such that $2 C \delta < \varepsilon / 3$ . Then choose $( g , \psi )$ sufficiently close to $( f , \omega )$ so that

$$
C _ { 2 } \| f - g \| _ { C ^ { 1 } ( J ) } < \varepsilon / 3 , \quad T _ { \delta } \| \omega - \psi \| _ { \infty , J } < \varepsilon / 3 .
$$

It follows that

$$
\| V _ { g , \psi } - V _ { f , \omega } \| _ { \infty , J } < \varepsilon .
$$

Therefore, G is continuous at $( f , \omega )$

## IV. NUMERICAL RESULTS

## A. Dataset Generation

We constructed datasets for three representative nonlinear dynamical systems: the damped Duffing oscillator, the inverted pendulum, and the Van der Pol oscillator. For each system, 1,000 parameterized instances were generated, with parameters sampled from uniform distributions as specified below. Lyapunov functions were obtained by numerically solving the PDE $\dot { V } = - x ^ { T } Q x$ using the LyZNet toolbox [8]. Q is a randomly sampled positive definite matrix. Each dataset was divided into 800 training, 100 validation, and 100 test samples. Each system was projected onto the grid $[ - 1 , 1 ] ^ { 2 }$ divided into a 64x64 grid during training of the FNO.

• Damped Duffing Equation:

$$
{ \dot { x } } _ { 1 } = x _ { 2 } ,\tag{11}
$$

$$
\dot { x } _ { 2 } = - \delta x _ { 2 } - \alpha x _ { 1 } - \beta x _ { 1 } ^ { 3 } ,\tag{12}
$$

with parameters sampled as $\alpha \sim U ( 1 , 1 0 ) , \beta \sim$ $U ( 0 . 1 , 2 . 0 ) , \delta \sim U ( 0 . 1 , 1 . 0 )$

• Inverted Pendulum:

$$
\begin{array} { r } { \dot { \theta } _ { 1 } = \theta _ { 2 } , } \end{array}\tag{13}
$$

$$
\dot { \theta } _ { 2 } = - c \theta _ { 2 } - \frac { c } { l } \sin \theta _ { 1 } ,\tag{14}
$$

with parameters $l \sim U ( 0 . 1 , 1 0 0 ) , c \sim U ( 0 . 1 , 1 0 )$

• Van Der Pol:

$$
{ \dot { x } } _ { 1 } = - x _ { 2 } ,\tag{15}
$$

$$
\dot { x } _ { 2 } = x _ { 1 } - \mu x _ { 2 } \big ( 1 - x _ { 1 } ^ { 2 } \big ) ,\tag{16}
$$

with parameters $\mu \sim U ( 0 . 0 1 , 1 . 0 )$

## B. Model Implementation

We use the implementation of FNO by [7]. For our experiments, we adopt an extended version of the standard FNO by integrating Adaptive Instance Normalization (AdaIN, [3]) to handle conditioning on the parameters Q. These parameters are flattened and passed through a small MLP to compute normalization constants that modulate the intermediate features at various stages of the network,

$$
\operatorname { A d a I N } ( x , \alpha ( Q ) , \beta ( Q ) ) = \alpha ( Q ) { \frac { x - \mu ( x ) } { \sigma ( x ) } } + \beta ( Q ) ,\tag{17}
$$

where $\mu ( x )$ and $\sigma ( x )$ are the mean and standard deviation of the layer, and $\alpha ( Q ) , \beta ( Q )$ are trainable MLPs that map the matrix Q to scalars that determine scaling and shifting in the normalized layer.

## C. Performance Comparison

We compared the FNO with DeepONet [10] in learning solutions to the Lyapunov PDE. We assessed the performance of these models by evaluating them on the testing set using the relative $L _ { 1 }$ error, which computes the $L _ { 1 }$ error between the predicted and the true Lyapunov function and divides by the $L _ { 1 }$ norm of the true Lyapunov function. This is the default evaluation metric used in [7], [2].

Table I summarizes the $L _ { 1 }$ errors on the test set. FNO achieved a lower error (0.0182) compared to DeepONet (0.6483), demonstrating its improved ability to approximate Lyapunov functions.

<table><tr><td>FNO</td><td>DeepONet</td></tr><tr><td>0.0182</td><td>0.6483</td></tr></table>

TABLE I: $L _ { 1 }$ errors of FNO and DeepONet on learning the Lyapunov PDE

## D. Visualization of Learned Functions

Figure 1 illustrates test cases across the systems. For each case, we plot the learned Lyapunov function against the ground-truth solution. The FNO reconstructions closely follow the true solutions, capturing its structure.

## V. CONCLUSION

In this work, we introduced a framework for learning Lyapunov functions of nonlinear dynamical systems using FNOs By formulating the Lyapunov condition as a PDE and leveraging the universality of FNOs on Sobolev spaces, we established theoretical guarantees ensuring that Lyapunov operators can be approximated with high fidelity. Our analysis demonstrated regularity and continuity properties that justify the application of neural operator learning in this setting. Through numerical experiments on nonlinear systems, we showed that FNOs achieve substantially lower approximation error compared to DeepONets.

A key challenge is extending this framework to higherdimensional dynamical systems. In principle, the universality of FNOs extends naturally to $\mathbb { R } ^ { n }$ , but practical implementation quickly becomes computationally prohibitive. Even for 3D problems, Fast Fourier Transforms (FFTs) scale as $O ( N ^ { 3 } \log N )$ in time and $O ( N ^ { 3 } )$ in memory, where N is the resolution along each dimension [14]. This growth imposes severe memory and runtime bottlenecks during training.

Finally, combining learned Lyapunov functions with controller design remains a promising avenue. By embedding Lyapunov certificates into feedback synthesis, one may learn controllers with formal guarantees. This would bridge datadriven stability analysis with practical control implementation, further motivating the development of scalable and interpretable neural operator architectures. An early step in this direction was accomplished using diffusion models [12].

## REFERENCES

[1] L. C. Evans. Partial differential equations, volume 19. American mathematical society, 2022.

![](images/e9c23e96e7c9561ec7d08c3515e201335f5d98cadc858fcf7cde40b0997fa671.jpg)  
Fig. 1: Comparison of learned Lyapunov functions (output V) against ground truth solutions (true V) for representative test cases across the inverted pendulum (Row 1), damped Duffing oscillator (Row 2), and Van der Pol oscillator (Row 3). Each subplot shows the input vector field components $( f _ { 1 } , f _ { 2 } )$ and the corresponding Lyapunov function values. The FNO reconstructions closely match the ground truth across all systems, closely matching the numerical reference solutions.

[2] M. Herde, B. Raonic, T. Rohner, R. Kappeli, R. Molinaro, ¨ E. de Bezenac, and S. Mishra. Poseidon: Efficient foundation models´ for pdes. Advances in Neural Information Processing Systems, 37:72525–72624, 2024.

[3] X. Huang and S. Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In Proceedings of the IEEE international conference on computer vision, pages 1501–1510, 2017.

[4] H. K. Khalil. Nonlinear Systems, Third Edition. Pearson Education, 2002.

[5] J. Kossaifi, N. Kovachki, Z. Li, D. Pitt, M. Liu-Schiaffini, R. J. George, B. Bonev, K. Azizzadenesheli, J. Berner, and A. Anandkumar. A library for learning neural operators, 2024.

[6] N. Kovachki, S. Lanthaler, and S. Mishra. On universal approximation and error bounds for fourier neural operators. Journal of Machine Learning Research, 22(290):1–76, 2021.

[7] N. B. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhattacharya, A. M. Stuart, and A. Anandkumar. Neural operator: Learning maps between function spaces. CoRR, abs/2108.08481, 2021.

[8] J. Liu, Y. Meng, M. Fitzsimmons, and R. Zhou. Tool lyznet: A lightweight python tool for learning and verifying neural lyapunov functions and regions of attraction. In Proceedings of the 27th ACM International Conference on Hybrid Systems: Computation and Control, pages 1–8, 2024.

[9] J. Liu, Y. Meng, M. Fitzsimmons, and R. Zhou. Physics-informed

neural network lyapunov functions: Pde characterization, learning, and verification. Automatica, 175:112193, 2025.

[10] L. Lu, P. Jin, and G. E. Karniadakis. Deeponet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators. arXiv:1910.03193, 2019.

[11] Y. Meng, R. Zhou, A. Mukherjee, M. Fitzsimmons, C. Song, and J. Liu. Physics-informed neural network policy iteration: Algorithms, convergence, and verification. In Forty-first International Conference on Machine Learning, 2024.

[12] A. Mukherjee, T. Quartz, and J. Liu. Manifold-guided stabilization of nonlinear dynamical systems with diffusion models. In 2025 American Control Conference (ACC), pages 3850–3855. IEEE, 2025.

[13] S. L. Sobolev. Sur un theor´ eme d’analyse fonctionnelle.\` Recueil Mathematique (Nouvelle s´ erie)´ , 4(46):471–497, 1938.

[14] C. Van Loan. Computational frameworks for the fast Fourier transform. SIAM, 1992.

[15] A. Vannelli and M. Vidyasagar. Maximal Lyapunov functions and domains of attraction for autonomous nonlinear systems. Automatica, 21(1):69–80, 1985.

[16] V. I. Zubov. Methods of A. M. Lyapunov and Their Application. Noordhoff, 1964.