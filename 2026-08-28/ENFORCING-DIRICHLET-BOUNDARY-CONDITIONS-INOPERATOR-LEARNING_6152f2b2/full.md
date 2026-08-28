# ENFORCING DIRICHLET BOUNDARY CONDITIONS INOPERATOR LEARNING

ANDREW M. STUART AND MARGARET TRAUTNER

Abstract. Operator learning in scientific machine learning is concerned with approximation of maps between infinite-dimensional function spaces; such maps frequently arise as the solution operators of partial diferential equations (PDEs). Neural operators have demonstrated broad empirical success at approximating such maps from data. However, most existing neural operator architectures enforce boundary conditions indirectly through training from data even though the boundary condition is often known exactly. Furthermore, existing modifications and approaches that do enforce boundary conditions explicitly sufer from impractical restrictions, including boundary smoothness, uniform grids, and separable, box-like domains. In this work, we propose an architecture which, independently of training, satisfies homogeneous Dirichlet boundary conditions, whilst simultaneously retaining the expressivity of existing kernel-integral neural operator architectures. This is achieved by enforcing the property that the output of each layer is contained in the span of a subset of the homogeneous Dirichlet eigenfunctions of the Laplacian on the output domain. The method requires only that the output domain be bounded with Lipschitz boundary and places no restriction on the choice of discretization, making it applicable to arbitrary mesh data and general geometries. We prove universal approximation for the resulting architecture; furthermore the approach we adopt in the analysis proves universality for a broad class of kernel-integral neural operators thereby uniting existing theory for a variety of operator learning methods. We validate the proposed method on maps defined by the coeficient to solution map in 2D PDEs: Darcy flow on a square domain and the Helmholtz equation on a circular domain. Comparisons are made with alternative methods.

## 1. Introduction

1.1. Motivation and Literature Review. Interest in, and the use of, neural network-based models for problems in science and engineering has exploded in the past decade. For many of these problems, the underlying phenomena are governed by partial diferential equations (PDEs), which map input functions to output functions via an infinite-dimensional operator. Operator learning encompasses a branch of machine-learning models that generalize to this infinite-dimensional function space setting; the models themselves are termed neural operators and early examples include DeepONet [31], PCA-Net [21, 8], random features [35] and the Fourier Neural Operator (FNO) [28]. For overviews of the field see [25, 3, 26].

Operator learning has proved empirically useful in two main cases. First, when the underlying equations of the system are known, operator learning can accelerate scientific computing by serving as a surrogate model for computationally expensive numerical simulations [37, 7, 3]. The second case is where a model is learned from data alone, as the governing equations are unknown [45, 46, 15]. However, in many settings, the reality falls in between these two cases; some information is known about the system, such as a boundary condition or conservation law [20], but the full descriptive equations remain unknown. In these cases, it is natural to want to enforce the known information without impeding the expressivity of the architecture or the ability to learn an approximator through stochastic optimization. It is this setting which largely motivates the present work: we enforce boundary conditions for neural operators that hold even without training while retaining the expressivity of the architecture. Furthermore, we develop the method in a manner that can act on arbitrary mesh data and thus on general geometries in any dimension.

Commonly used neural operator architectures rely on data to learn the boundary condition and thus enforce it only approximately. For instance, the DeepONet architecture multiplies two neural networks, one of which, the “branch” network, constructs basis functions from the input function, and the second, the “trunk” network, serves as a means to evaluate the output function on any point in the domain. Thus, the “trunk” network is largely responsible for learning the boundary condition; any point on the boundary should evaluate to match the boundary condition. However, there is no current mechanism to enforce this explicitly without fundamentally altering the expressivity of the architecture. Similarly, the FNO processes the input function through finite-dimensional neural networks acting pointwise both on the function and on a learned reweighting of the function in Fourier space via a kernel-integral operator [28]. By design, this architecture is natural for periodic boundary conditions on box-like domains with uniform grids. In practice, a positional encoding, or simply the coordinates $( x _ { 1 } , \ldots , x _ { d } )$ of the input domain, are appended to the input function values at that point. Thus, enforcement of non-periodic boundary conditions comes only by learning a map from this input positional encoding to the boundary value from data, and once again there is currently no mechanism to enforce this explicitly. These two architectures are representative of the larger class of neural operator architectures which enforce boundary conditions in a similar manner: PCA-Net, various modifications of Deep-ONet [1, 19], and other architectures involving kernel-integral operators, including wavelet neural operators [42] and the Laplace neural operator [10] all require data to enforce boundary conditions through approximation.

Several lines of research have made eforts to enforce boundary conditions explicitly in both operator learning and scientific machine learning (SciML) more broadly. Most notably, work on orthogonal polynomial neural operators (OPNOs) [29] uses a Shen transform [40] of Chebyshev polynomials to replace the Fourier basis and transform in the FNO with a basis that naturally satisfies Dirichlet, von Neumann, or Robin boundary conditions. The work also develops an eficient algorithm for computing these transforms and demonstrates that the resulting architecture satisfies boundary conditions to machine precision. A drawback of this work is that it requires data to be evaluated on Chebyshev-Gauss-Lobatto points, a variant of Gaussian grids [9]. This issue is remedied by the follow-up work [30] that develops an algorithm to extend the previous method to uniform grids. Both works prove universal approximation properties of their architecture and demonstrate machine precision satisfaction of the boundary conditions in various PDE examples. However, limitations of these methods are that they may only be generalized to separable domains; i.e. domains that can naturally be viewed as a d-dimensional box with uniform or Gaussian grids as opposed to arbitrary mesh coordinate points. A similar approach also appears in recently proposed work that is capable of enforcing the boundary conditions on arbitrarily curved quadrilateral domains only [14]. These caveats prevent the methods from extending to more complex geometries, which may well be the setting when direct enforcement of the boundary condition is most valuable.

Other approaches to enforce boundary conditions in operator learning include BOON [39], which uses a physics-informed refining procedure to enforce the constraint, as well as a method based on Q-learning that proposes an algorithm that refines the boundary conditions iteratively during training [12]. In non-operator learning approaches to SciML, boundary conditions have also been encoded explicitly in graph neural networks [22] and physics-informed neural networks [6]. Furthermore, the boundary condition problem has been considered in work that predates most current approaches in SciML, including for data-driven models based on kernel learning [18] and standard feedforward neural networks [32]. The approach in this latter work involves multiplication by a known function that satisfies a zero boundary condition and has also been carried forward to more recent work on convergence of Deep Galerkin Methods (DGMs) to approximate PDEs in SciML [24, 11]. Another very recent method proposes to learn function extensions separately from the main neural operator model, which allows additional domain generality [33].

1.2. Contributions. We make the following contributions to the state-of-the-art, in relation to enforcing Dirichlet boundary conditions within operator learning, as overviewed in the preceding subsection:

(C1) We propose an operator learning methodology that automatically satisfies Dirichlet boundary conditions, independent of training, by using the Dirichlet eigenfunctions of the Laplacian as a central part of the architecture.

(C2) We prove universal approximation for a general class of nonlocal neural operators, subsuming the proposed Dirichlet eigenfunction based architecture as a special case; these approximation results may be of independent interest beyond the application to enforcing Dirichlet boundary conditions. This general class includes neural operators using only a single hidden layer and single basis function as well as neural operators that perform approximation in a particular basis of interest. Suficient conditions on such basis functions are provided.

(C3) We implement the proposed method to approximate two representative operators defined by coeficient-to-solution maps arising in PDE problems: the Darcy flow equation on a square domain and the Helmholtz equation on a circular domain. We demonstrate the advantage of the proposed method over FNO, as well as over three possible alternative operator learning approaches that also automatically satisfy the boundary condition.

Specifically, we introduce a new, kernel-integral based, operator learning architecture in which the output space of each layer builds an approximation in the basis of Dirichlet eigenfunctions of the Laplacian in the output domain, contribution (C1). These eigenfunctions satisfy a zero boundary condition and thus, coupled with a zero-preserving activation function and an appropriate fixed bias, create an architecture that automatically satisfies the Dirichlet boundary condition of a PDE. In fact, replacing the final hidden layer of any kernel-integral neural operator, including FNO, with this Laplacian eigenfunction layer achieves the desired boundary condition without training. In addition, as a departure from previous work that automatically satisfies the boundary condition, this approach is applicable to complex geometries and domains; we assume only that the domain is bounded, with Lipschitz boundary. Furthermore, the method requires no restriction on choice of discretization or grid; it may be used on functions defined on arbitrary meshes. Beyond the choice of the eigenfunctions of the Laplacian as the basis elements, we establish universal approximation for a much broader class of potential basis functions, contribution (C2). The approach we adopt suggests practical extensions of the method to other classical basis choices, including various finite element bases. Numerical results demonstrate the practicality and competitiveness of the proposed approach, contribution (C3).

The remainder of the paper is organized as follows. In Section 2, we define the nonlocal neural operator, as well as the Dirichlet-boundary condition enforcing variant, and detail the assumptions on our choice of basis. In Section 3, we present the main theorems as well as critical lemmas supporting the conclusions; we defer the detailed proofs of the lemmas to the appendix. In Section 4, we showcase the results of various numerical experiments supporting the practical application of the proposed Laplacian eigenfunction layer. We make concluding remarks in Section 5. The more technical proofs, and some useful supplementary lemmas, are contained in the appendix. Appendix A contains lemmas describing finite-dimensional feedforward neural networks; Appendix B contains proofs that various familiar application settings satisfy the assumptions of our main theorems; Appendices C and D contain the proofs of the two main lemmas underlying our main results, Theorem 3.4 and Theorem 3.5; and Appendix E contains key results on the behavior of the eigenfunctions of the Laplacian.

## 2. Definition of Nonlocal Neural Operator

In this section we define the nonlocal neural operator (NNO) as in [27], repeated here in the context and notation of our work, as well as specify the Dirichlet layer that allows for exact enforcement of Dirichlet boundary conditions. We note that the NNO may also be referred to as a kernel-integral neural operator in the literature, and that a variety of common operator learning architectures fall into this general setting, including the Fourier Neural Operator (FNO) [28]. The NNO definition is given in Subsection 2.1. In Subsection 2.2, we define the Dirichlet layer variant, which is obtained through the particular choice of the eigenfunctions of the Laplacian as the basis functions in the hidden layer of the more general NNO definition.

Throughout this work, $\| \cdot \|$ and $\langle \cdot , \cdot \rangle$ denote the $L ^ { 2 } ( \Omega )$ norm and inner product, and |·| denotes the Euclidean norm. We let Ω be a bounded, open, Lipschitz domain in $\mathbb { R } ^ { \dot { d } }$ unless otherwise specified. Furthermore, we define $[ N ] : = \{ 1 , \dots , N \}$ . We adopt the convention that $C ^ { s } ( \Omega )$ is the Banach space with norm

$$
\| \eta \| _ { C ^ { s } ( \Omega ) } = \sum _ { | \alpha | \leq s } \operatorname* { s u p } _ { x \in \Omega } | ( \partial ^ { \alpha } \eta ) ( x ) | < \infty .
$$

2.1. Nonlocal Neural Operator. Let $\mathcal { X } ( \Omega ; \mathbb { R } ^ { o } ) , \mathcal { Y } ( \Omega ; \mathbb { R } ^ { o } )$ , and $\mathcal { V } ( \Omega ; \mathbb { R } ^ { o } )$ be $\mathrm { B a } -$ nach spaces of R<sup>o</sup>-valued functions over domain Ω. Given a channel width $d _ { c }$ , define the following lifting and projection layers as learnable feedforward neural networks<sup>1</sup> between finite-dimensional Euclidean spaces defined as Nemytskii operators via

$$
\mathcal { R } : \mathcal { X } ( \Omega ; \mathbb { R } ^ { k } ) \to \mathcal { V } ( \Omega ; \mathbb { R } ^ { d _ { c } } ) , \quad u ( x ) \mapsto R ( u ( x ) , x )\tag{2.1}
$$

and

$$
\begin{array} { r } {  { \mathcal { Q } } : \mathcal { V } (  { \Omega } ; \mathbb { R } ^ { d _ { c } } ) \to \mathcal { V } (  { \Omega } ; \mathbb { R } ^ { k ^ { \prime } } ) , \quad v ( x ) \mapsto Q ( v ( x ) ) . } \end{array}\tag{2.2}
$$

Note that $\mathcal { R }$ includes the positional encoding via pairing of $u ( x )$ with the domain variable x. Each hidden layer $\mathcal { L } _ { \ell } , \ell \in [ L ]$ , defines a mapping $\mathcal { V } ( \Omega ; \mathbb { R } ^ { d _ { c } } ) \to \mathcal { V } ( \Omega ; \mathbb { R } ^ { d _ { c } } )$ via

$$
( \mathcal { L } _ { \ell } v ) ( x ) = \sigma \left( W _ { \ell } v ( x ) + b _ { \ell } + \sum _ { m = 1 } ^ { M } \langle T _ { \ell , m } v , \psi _ { \ell , m } \rangle _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d _ { c } } ) } \phi _ { \ell , m } ( x ) \right) .\tag{2.3}
$$

Here the activation function $\sigma : \mathbb { R }  \mathbb { R }$ is a non-polynomial $C ^ { \infty }$ function which is extended to a Nemytskii operator acting component-wise on vector inputs: for a function $v ( x ) = ( v _ { 1 } ( x ) , \ldots , v _ { d _ { c } } ( x ) )$ , we define $\sigma ( v ( x ) ) = ( \sigma ( v _ { 1 } ( x ) ) , \ldots , \sigma ( v _ { d _ { c } } ( x ) ) )$ for $x \in \Omega$ . We may now define an $\mathrm { N N O ; }$ note that the definition includes an assumption on the properties of the activation function and that this assumption follows the definition.

Definition 2.1 (Nonlocal Neural Operator). The Nonlocal Neural Operator (NNO) is a mapping $\Psi : \mathcal { X } ( \Omega ; \mathbb { R } ^ { k } ) \to \mathcal { Y } ( \Omega ; \mathbb { R } ^ { k ^ { \prime } } )$ consisting of a lifting layer R, hidden layers $\mathcal { L } _ { 1 } , \ldots , \mathcal { L } _ { L }$ , and projection layer Q as defined in equations (2.1), (2.3), and (2.2), respectively, that acts on input $a \in \mathcal { X } ( \Omega ; \mathbb { R } ^ { k } )$ via

$$
\Psi ( a ) = \mathcal { Q } \circ \mathcal { L } _ { L } \circ \cdots \circ \mathcal { L } _ { 1 } \circ \mathcal { R } ( a ) ,
$$

and uses a non-polynomial $C ^ { \infty }$ activation function σ.

♢

We assume that the activation function σ used throughout the model is such that a feedforward, zero-preserving neural network can universally approximate the identity in $C ^ { s }$ on a closed ball in a finite dimension. This is true for ReLU for any $s \geq 0 \ [ 3 4 $ , Appendix B, Part $\left( \mathrm { i v } \right) ]$ , via $\sigma ( x ) - \sigma ( - x ) = x$ , which allows for exact replication of the identity. It is also true for GeLU for any integer $s \geq 0 ~ [ 4 4$ ， Lemma 3.1], which, unlike ReLU, also satisfies the $C ^ { \infty }$ condition.

Assumption 2.2. Let $I _ { d }$ be the identity matrix in d dimensions. Let $s _ { \sigma } \geq 0$ . We assume $\sigma$ is such that $\sigma ( 0 ) = 0$ and for any $\epsilon > 0$ and closed ball $\mathcal { K } \subset \mathbb { R } ^ { d }$ , there exists a number of layers $L ,$ matrices $\{ A _ { l } \} _ { l = 1 } ^ { L }$ , and biases $\{ b _ { l } \} _ { l = 1 } ^ { L }$ such that for neural network

$$
G ( x ) = A _ { L } ( \sigma ( A _ { L - 1 } \sigma ( \dots A _ { 1 } x + b _ { 1 } ) \dotsb + b _ { L - 1 } ) + b _ { L } ,
$$

$G ( 0 ) = 0$ , and it holds that $\| G - I _ { d } \| _ { C ^ { s _ { \sigma } } ( { \cal K } ) } < \epsilon .$

♢

Universal approximation results have been proved for many specific variants of the NNO that make a particular choice of the basis functions $\psi _ { \ell , m }$ and $\phi _ { \ell , m }$ in (2.3). For example, in the FNO these are the Fourier basis elements. In order to capture a broad class of possible architectures under the same theory, rather than proving multiple specific theorems for each specific basis choice, in this work we prove universality for any NNO whose basis functions satisfy mild conditions. Informally, if the map to be approximated produces an output set that can be approximated in some $L ^ { 2 } ( \Omega )$ basis $\{ \eta _ { j } \} _ { j = 1 } ^ { \infty } \subset C ^ { s ^ { \prime } } ( \overline { { \Omega } } )$ , then one can achieve $C ^ { s ^ { \prime } }$ approximation using only a single hidden layer and single pair of basis functions $\psi _ { 1 , 1 }$ , positive almost-everywhere and continuous, and $\phi _ { 1 , 1 }$ , nonzero and $C ^ { s ^ { \prime } }$ . This is formalized in Section 3. Additionally, if the set of functions $\{ \eta _ { j } \} _ { j = } ^ { \infty }$ only satisfies $\{ \eta _ { j } \} _ { j = 1 } ^ { \infty } \subset C ^ { s ^ { \prime } } ( \Omega )$ , then one can simply choose a finite subset of these functions to serve as the basis functions $\phi _ { L , m }$ in the final hidden layer and achieve universal approximation as well. This latter approach can be practically useful when one seeks an output function satisfying certain conditions. These mild conditions also allow one to interchange various choices of basis functions at each layer without violating universality. We emphasize that the general conditions we specify on the basis functions in Section 3 unite approximation theory for architectures with various choices of basis functions in one single result.

The prior work [27] showed that in fact one can set $M = 0 , L = 0$ , and ψ<sub>0</sub> ${ \bf \Phi } _ { , 0 } = \phi _ { 0 , 0 } \equiv 1$ in the NNO architecture and achieve universality via the resulting averaging neural operator (ANO). It is a remarkable fact, established in [27], that a single hidden layer taking the form

$$
( \mathcal { L } _ { 1 } v ) ( x ) = \sigma \left( W v ( x ) + \frac { 1 } { \vert \Omega \vert } \int v ( x ) \mathrm { d } x \right)
$$

is suficient for universal approximation. While a surprising and insightful theoretical result, the ANO is not competitive with other operator learning architectures. Intuitively, the ANO chooses a constant basis to carry the nonlocal information. In the case of, for instance, zero Dirichlet boundary conditions, this adds a constant value everywhere in the domain Ω and relies heavily on the dependence of the projection layer Q on the positional encoding $x ,$ carried to the input of Q through an identity block submatrix of W. The boundary conditions are enforced through the ability of the ReLU neural network Q to approximate arbitrary continuous functions: positions x on the boundary are mapped close to 0 as learned from data. In practice, one would never want to use such a restrictive architecture, especially in the case of Dirichlet boundary conditions. In this work we seek a more natural basis for maps with such boundary conditions.

2.2. Dirichlet Layer. In this subsection, we propose a layer of the nonlocal neural operator that automatically enforces Dirichlet boundary conditions even before training. In particular, we set the basis functions equal to the Dirichlet eigenfunctions of the Laplacian on $\Omega ;$ doing so automatically enforces a zero boundary condition without training, as we prove in Section 3. We refer to this modified layer as a Dirichlet layer, and define it explicitly below.

Definition 2.3 (Dirichlet Layer). The Dirichlet layer is a mapping $\mathcal { V } ( \Omega ; \mathbb { R } ^ { d _ { c } } ) \to$ $\mathcal { V } ( \Omega ; \mathbb { R } ^ { d _ { c } } )$ via equation (2.3), where $\psi _ { \ell , m } = \phi _ { \ell , m } = \eta _ { m }$ , where ℓ is the layer index and $\{ \eta _ { m } \} _ { m = 1 } ^ { \infty }$ are the Dirichlet eigenfunctions of the Laplacian on Ω. ♢

By using a Dirichlet layer with zero bias as the final hidden layer in an NNO, in tandem with a zero-preserving projection layer $\mathcal { Q } ,$ one may enforce homogeneous Dirichlet boundary conditions without training, as we prove in Section 3. One may use other choices of hidden layers prior to the Dirichlet layer, and the boundary conditions will still hold. We also point out that while this layer is designed to produce outputs with zero boundary conditions, for constant, nonzero boundary conditions, we may simply preprocess and postprocess the data. For instance, for a map $( u , \mathfrak { b } ) \mapsto v$ , where u and v are input and output functions, respectively, and b is the boundary condition, we may learn the map $( u , \mathfrak { b } ) \mapsto v - \mathfrak { b } \mathbf { 1 }$ , where 1 is the constant function equal to 1 on ${ \overline { { \Omega } } } ,$ , and then postprocess the output of the NNO by adding back b1.

Remark 2.4 (Computational Complexity). For the FNO, the fast Fourier transform allows the computational complexity of a hidden Fourier layer to be O(N log N), where N is the number of grid points, or $N = n ^ { d }$ where n is the number of points in a single dimension of the uniform grid. This complexity remains the dominant term even when the number of Fourier modes K is decreased. For a Dirichlet layer with M eigenfunctions forming the hidden layer (2.3), the complexity is O(NM). In practice, for reasonable mesh sizes, both for FNO and for the Dirichlet layer, we observe that K ≈ log N and M ≈ log N are more than suficient to achieve low error, and so the Dirichlet layer efectively shares the computational complexity of an FNO layer. This empirical observation is also supported by the numerical experiments in the existing work of [27], which demonstrates that fixing M for the nonlocal neural operator can be optimal in terms of the cost-accuracy tradeof between parameter count and model accuracy. These findings are corroborated by our own numerical experiments in Section 4. ♢

## 3. Theory

In this section, we unite theory for a broad class of kernel-integral neural operators under one umbrella. We do so via two theorems. The first theorem, Theorem 3.4, states that the nonlocal neural operator (NNO) architecture of Definition 2.1 is universal with only a single hidden layer with a single basis function. This is a generalization of the averaging neural operator result of [27]; in particular the single basis function no longer needs to be the constant function. In fact, universality of several well-known neural operator architectures may be viewed as special cases of this result; see Proposition 3.2 for further explanation.

The second main theorem we prove also concerns universality of the NNO architecture, but in a slightly more general setting where we are forced to use the basis functions present in the architecture itself to achieve the universality result. Namely, the first theorem requires that the output space be representable in a basis whose elements are continuous up to and including the boundary of the domain Ω. When this is true, finite-dimensional neural networks can approximate, to arbitrary accuracy, continuous functions on bounded sets in $\mathbb { R } ^ { d }$ , so the proof itself simply uses finite-dimensional neural networks to approximate basis functions. However, this is intuitively unappealing. If the idea is to approximate a function using hidden layers of the form (2.3) with some choice of basis elements, it is natural to suppose that the proof could use the fact that the architecture includes these basis elements already instead of reconstructing them using finite-dimensional neural networks. This achievement is the most appealing feature of Theorem 3.5; the proof uses the architecture’s own basis functions to obtain the universality result in addition to holding in a slightly more general setting. It is this latter theorem that applies to the Dirichlet eigenfunctions of the Laplacian as well, which for general Lipschitz domains are $C ^ { \infty } ( \Omega )$ but not necessarily $C ^ { s ^ { \prime } } ( \overline { { \Omega } } )$ for any $s ^ { \prime } > 0$

We require the following critical assumption about the ability of the chosen basis to approximate functions in the desired output set. Intuitively, one is allowed to first apply a smoother to the output set to be approximated, but then the chosen basis must uniformly approximate the smoothed set. Since we view the following assumption as rather general, we detail in Proposition 3.2 some familiar or relevant cases in which Assumption 3.1 holds. This assumption may be viewed informally as a consequence of separability of the relevant output space, compactness of the input set, and continuity of the underlying map of interest in relevant application settings. Since it is used in the context of the output space in subsequent results, we keep the “primed” notation for $\mathsf { K } ^ { \prime }$ and $s ^ { \prime } ,$ even though the assumption itself is self-contained. Recall that, unless otherwise specified in this paper, Ω is open; however the following assumption and proposition are formulated in a slightly more general setting.

Assumption 3.1 (Smoothed Approximation). Let $\Omega \subset \mathbb { R } ^ { d }$ be a bounded set that may be open or closed. A compact subset $\mathsf { K } ^ { \prime }$ of Banach space $B ( \Omega )$ , continuously embedded in $L ^ { 1 } ( \Omega )$ , satisfies the smoothed approximation assumption in $B ( \Omega )$ with respect to a set of functions $\{ \eta _ { j } \} _ { j = 1 } ^ { \infty }$ , if there exists a set $B \subset  { \mathbb { R } } ^ { d }$ such that $\{ \eta _ { j } \} _ { j = 1 } ^ { \infty }$ are defined on $B \supseteq \Omega$ , there exists a family of linear maps $\{ f _ { \delta } \} _ { \delta > 0 }$ , also defined on domain $B ,$ such that $f _ { \delta } : B ( \Omega ) \to B ( \Omega )$ for $\delta > 0 .$ , and the following hold:

(S1) (smoother convergence) li $\begin{array} { r } { \mathrm { 1 } _ { \delta \to 0 } \operatorname* { s u p } _ { w \in \mathsf { K } ^ { \prime } } \| f _ { \delta } ( w ) - w \| _ { \mathcal { B } ( \Omega ) } = 0 . } \end{array}$

(S2) (projection convergence) For each $\delta > 0$

$$
\operatorname* { l i m } _ { J \to \infty } \operatorname* { s u p } _ { w \in \mathsf { K } ^ { \prime } } \Big \| f _ { \delta } ( w ) - \sum _ { j = 1 } ^ { J } \langle f _ { \delta } ( w ) , \eta _ { j } \rangle _ { L ^ { 2 } ( B ) } \eta _ { j } \Big \| _ { B ( \Omega ) } = 0 .
$$

(S3) (continuous functional) For each $\delta > 0$ and $j \in \mathbb N$ , the map $w \mapsto \langle f _ { \delta } ( w ) , \eta _ { j } \rangle _ { L ^ { 2 } ( B ) }$ is a continuous functional on $B ( \Omega )$ .

♢

To establish some key examples where the Smoothed Approximation Assumption above holds, we present the following proposition.

Proposition 3.2. The following pairs of functions and Banach spaces satisfy the Smoothed Approximation Assumption 3.1:

(1) Let Ω be a bounded set in $\mathbb { R } ^ { d }$ with Lipschitz boundary. For a box B containing Ω in its interior, let $\{ \eta _ { j } \} _ { j = 1 } ^ { \infty }$ be the L<sup>2</sup>-orthogonal Fourier sine/cosine basis in $L _ { p e r } ^ { 2 } ( B )$ , and let K<sup>′</sup> be any compact subset of $C ^ { s ^ { \prime } } ( \overline { { \Omega } } )$ for integer $s ^ { \prime } \geq 0$ . Then K<sup>′</sup> satisfies the Smoothed Approximation Assumption 3.1 in $C ^ { s ^ { \prime } } ( \overline { { \Omega } } )$ with respect to $\{ \eta _ { j } \} _ { j = 1 } ^ { \infty }$

(2) Let Ω be a bounded set in $\mathbb { R } ^ { d }$ with Lipschitz boundary. Let $\{ \eta _ { j } \} _ { j = 1 } ^ { \infty }$ be Dirichlet eigenfunctions of the Laplacian on $B : = { \overline { { \Omega } } }$ . Let K<sup>′</sup> be any compact subset $o f C _ { 0 } ( \overline { { \Omega } } )$ . Then K<sup>′</sup> satisfies the Smoothed Approximation Assumption 3.1 in $C _ { 0 } ( \overline { { \Omega } } )$ with respect to $\{ \eta _ { j } \} _ { j = 1 } ^ { \infty }$

(3) Let $\Omega = [ - 1 , 1 ] ^ { d }$ . Let $\{ \eta _ { j } \} _ { j = 1 } ^ { \infty }$ be the tensorized Legendre polynomials defined on $B : = \Omega$ . Let K<sup>′</sup> be any compact subset of $C ^ { s ^ { \prime } } ( \Omega )$ for integer $s ^ { \prime } \geq 0$ Then K<sup>′</sup> satisfies the Smoothed Approximation Assumption 3.1 in $C ^ { s ^ { \prime } } ( \Omega )$ with respect to $\{ \eta _ { j } \} _ { j = 1 } ^ { \infty }$

Note that Case 1 encompasses the averaging neural operator and the FNO setting for $C ^ { s ^ { \prime } } ( \overline { { \Omega } } )$ approximation. We prove later that Case 2 covers the Dirichlet layer setting. Case 3 is the unweighted $L ^ { 2 } .$ -norm analogue of the tensorized Chebyshev polynomial basis used in [29]. Ultimately, all these settings are covered by the same theory in Theorems 3.4 and 3.5. Since the proof of Proposition 3.2 is technical, we defer it to Appendix B. The intuition behind each case is that we first apply a smoother to an extension of the set $\mathsf { K } ^ { \prime }$ and then we may approximate it in the specified basis to obtain Smoothed Approximation Assumption 3.1, (S2). The Smoothed Approximation Assumption 3.1, conditions (S1) and (S3), are obtained from properties of the smoothing and extension operators used.

3.1. Main Theorems. In this subsection we present our main theorems, Theorem 3.4 and 3.5, which prove universality for NNOs under general conditions on the basis functions. Through this result, we unite theory for kernel-integral neural operators under two statements. In Corollary 3.6, we observe that enforcing constant Dirichlet boundary conditions through the Dirichlet layer in Definition 2.3 retains universality while automatically enforcing the boundary condition.

We will use the following assumptions.

Assumption 3.3 (Single Basis Pair). We assume that $\eta _ { 1 }$ and $\eta _ { 2 }$ satisfy

(E1) $\eta _ { 1 } \geq 0$ on Ω.

(E2) $\eta _ { 1 } = 0$ only on a set of Lebesgue measure 0.

(E3) $\begin{array} { r } { \operatorname* { s u p } _ { y \in \Omega } | \eta _ { 1 } ( y ) | < \infty . } \end{array}$

(E4) $\eta _ { 1 } \in C ( \Omega )$

(E5) $\eta _ { 2 } \in C ^ { s ^ { \prime } } ( \overline { { \Omega } } )$ and $\eta _ { 2 } \neq 0$ on $\overline { { \Omega } } .$

♢

The first theorem addresses the case when approximation is possible using only a single hidden layer and pair of basis functions.

Theorem 3.4 (Single Basis Element). Let Ω be a bounded domain in $\mathbb { R } ^ { d }$ with Lipschitz boundary. Let $\Psi ^ { \dag } : C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { k } )  C ^ { s ^ { \prime } } ( \overline { { \Omega } } ; \mathbb { R } ^ { k ^ { \prime } } )$ be a continuous operator for integers $s , s ^ { \prime } \geq 0 .$ , and $\mathsf { K } \subset C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { k } )$ be a compact subset. Assume that $\Psi ^ { \dagger } ( { \sf K } )$ satisfies the Smoothed Approximation Assumption 3.1 in $C ^ { s ^ { \prime } } ( \Omega ; \mathbb { R } ^ { k ^ { \prime } } )$ with respect to some sequence of functions $\{ \xi _ { j } \} _ { j = 1 } ^ { \infty }$ . If these functions are also such that $\{ \xi _ { j } \} _ { j = 1 } ^ { \infty } \subset$ $C ^ { s ^ { \prime } } ( \overline { { \Omega } } )$ , then for any $\varepsilon > 0$ , there exists a nonlocal neural operator Ψ as defined in Definition ${ \it 2 . 1 } _ { } ,$ , with Assumption 2.2 on the activation σ satisfied with $s _ { \sigma } = s ^ { \prime }$ , such that

$$
\underset { u \in \mathsf { K } } { \operatorname* { s u p } } \Vert \Psi ^ { \dagger } ( u ) - \Psi ( u ) \Vert _ { C ^ { s ^ { \prime } } ( \Omega ) } < \varepsilon ,\tag{3.1}
$$

with a single hidden layer $( M = 1 )$ with any single basis function pair $\psi _ { 1 , 1 } : = \eta _ { 1 }$ and $\phi _ { 1 , 1 } : = \eta _ { 2 }$ satisfying Assumptions 3.3.

The second theorem addresses the slightly more general case where the Smoothed Approximation Assumption only holds for functions in $C ^ { s ^ { \prime } } ( \Omega )$ rather than in $C ^ { s ^ { \prime } } ( \overline { { \Omega } } )$ ).

It also builds the approximation in the proof by using basis functions contained in the NNO architecture itself. This justifies the practical intuition that by choosing a set of basis functions to use in the NNO architecture, one is projecting onto the span of the functions as part of the approximation.

Theorem 3.5 (Approximation Using NNO Basis Elements). Let Ω be a bounded domain in $\mathbb { R } ^ { d }$ with Lipschitz boundary. Let $\Psi ^ { \dag } : C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { k } )  C ^ { s ^ { \prime } } ( \overline { { \Omega } } ; \mathbb { R } ^ { k ^ { \prime } } )$ be a continuous operator for integers s, $s ^ { \prime } \geq 0$ , and $\mathsf { K } \subset C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { k } )$ be a compact subset. Assume that $\Psi ^ { \dagger }$ satisfies the Smoothed Approximation Assumption 3.1 in $C ^ { s ^ { \prime } } ( \Omega ; \mathbb { R } ^ { k ^ { \prime } } )$ with respect to some sequence of functions $\{ \xi _ { j } \} _ { j = 1 } ^ { \infty }$ . If these functions are also such that $\{ \xi _ { j } \} _ { j = 1 } ^ { \infty } \subset C ^ { s ^ { \prime } } ( \Omega )$ , then for any $\varepsilon > 0$ , there exists a nonlocal neural operator Ψ as defined in Definition $2 . 1 ,$ , with Assumption 2.2 on the activation $\sigma$ satisfied with $s _ { \sigma } = s ^ { \prime } .$ , such that

$$
\underset { u \in \mathsf { K } } { \operatorname* { s u p } } \Vert \Psi ^ { \dagger } ( u ) - \Psi ( u ) \Vert _ { C ^ { s ^ { \prime } } ( \Omega ) } < \varepsilon ,\tag{3.2}
$$

where the final hidden layer contains $\{ \xi _ { j } \} _ { j = 1 } ^ { J }$ as outer basis elements for some $J ;$ i.e. $\{ \xi _ { j } \} _ { j = 1 } ^ { J } \subset \{ \phi _ { L , m } \} _ { m = 1 } ^ { M }$ with corresponding inner basis functions $\{ \psi _ { m } : \phi _ { m } \equiv$ $\xi _ { j } , j \le j \}$ satisfying (E2)- (E4) in Assumptions 3.3 and an earlier hidden layer contains at least one basis function pair $\eta _ { 1 }$ satisfying $( E 1 ) \ – ( E _ { 4 } )$ and $\eta _ { 2 }$ satisfying (E2)-(E4).

The following corollary justifies the use of the Dirichlet layer for approximation as well as shows that it enforces zero boundary conditions exactly.

Corollary 3.6. Let Ω be a bounded Lipschitz domain in $\mathbb { R } ^ { d }$ , and let $\Psi ^ { \dagger } : C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { k } ) $ $C _ { 0 } ( \overline { { \Omega } } ; \mathbb { R } ^ { k ^ { \prime } } )$ be a continuous operator for nonnegative integer s. Let $\{ \eta _ { j } \} _ { j = 1 } ^ { \infty }$ be the eigenfunctions of the Dirichlet Laplacian on Ω. Let $\mathsf { K } \subset C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { k } )$ be a compact subset. Then for any $\varepsilon > 0$ , there exists a nonlocal neural operator Ψ as defined in Definition $\it { 2 . 1 }$ , with Assumption $\it 2 . 2$ on the activation σ satisfied with $s _ { \sigma } = 0$ , such that

$$
\operatorname* { s u p } _ { u \in \mathsf { K } } \| \Psi ^ { \dagger } ( u ) - \Psi ( u ) \| _ { C ( \Omega ) } < \varepsilon ,
$$

with the final NNO layer a Dirichlet layer as defined in Definition 2.3, and

$$
\operatorname* { s u p } _ { y \in \partial \Omega } | ( T ^ { * } \Psi ) ( y ) | = 0 ,
$$

where $T ^ { \ast } : H ^ { 1 } ( \Omega ) \to L ^ { 2 } ( \partial \Omega )$ is the trace operator.

Proof. By Proposition 3.2, Case $2 , \ \{ \eta _ { j } \} _ { j = 1 } ^ { \infty }$ satisfy the Smoothed Approximation Assumption with respect to $C _ { 0 } ( \overline { { \Omega } } )$ . Furthermore, the eigenfunctions of the Dirichlet Laplacian are in $C ^ { \infty } ( \Omega ) \cap H _ { 0 } ^ { 1 } ( \Omega )$ , so Theorem 3.5 applies. We note that the ground state eigenfunction $\eta _ { 1 }$ satisfies conditions (E1)-(E4) [16, Section 6.5]. By the construction in Lemma 3.11, the final hidden layer uses $\{ \eta _ { j } \} _ { j = 1 } ^ { J }$ as basis elements, with the remaining layers a finite-dimensional neural network that preserves 0 with smooth activation. Thus, Ψ enforces zero boundary condition on the output function in the trace sense. □

Remark 3.7. The approximation in Corollary 3.6 is stated for $C ( \Omega )$ only. For general Lipschitz Ω, while the eigenfunctions are in $C ^ { \infty } ( \Omega )$ , the norm $\| \eta _ { j } \| _ { C ^ { s ^ { \prime } } ( \Omega ) }$ can blow up as one approaches the boundary, for instance, at reentrant corners. This causes the Dirichlet eigenfunctions of the Laplacian and the space $C _ { 0 } ^ { s ^ { \prime } } ( \overline { { \Omega } } )$ to fail the Smoothed Approximation Assumption without additional assumptions on boundary regularity. ♢

The key lemmas underpinning the proofs of Theorems 3.4 and 3.5 are Lemma 3.8, which decomposes the true map into a sum of continuous functionals over the chosen basis, and Lemmas 3.10 and 3.11, which are devoted to approximation of the continuous functionals by an NNO with the architectures corresponding to each theorem. In Subsections 3.2 and 3.3, respectively, we describe and state these lemmas before returning to prove the two theorems in Subsection 3.4.

3.2. True Map: Approximation by Sum of Continuous Functions. The first step to establishing the approximation result is Lemma 3.8, which decomposes the true map $\Psi ^ { \dagger }$ into a finite sum of our basis elements $\eta _ { j }$ with coeficients determined by continuous functionals on the input function. This step is similar to that of $[ 2 7 ]$ except that we allow for the choice of basis to be more general than that of the Fourier basis. We note that this result may be useful beyond the setting of this work since one could substitute a variety of bases of interest; some are detailed in Proposition 3.2.

The following lemma decomposes the true map into a sum of the basis elements $\{ \eta _ { j } \} _ { j = 1 } ^ { J }$ weighted by continuous functionals from $L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to \mathbb { R }$

Lemma 3.8. Let Ω be a bounded Lipschitz domain in $\mathbb { R } ^ { d }$ , and let $s , s ^ { \prime }$ be nonnegative integers. Let $\Psi ^ { \dag } : C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { k } )  C ^ { s ^ { \prime } } ( \overline { { \Omega } } ; \mathbb { R } ^ { k ^ { \prime } } )$ be a continuous operator, and let $\{ \eta _ { j } \} _ { j = 1 } ^ { \infty } \subset C ^ { s ^ { \prime } } ( \Omega )$ . Let $\mathsf { K } \subset C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { k } )$ be a compact subset. Assume that $\mathsf { K } ^ { \prime } : =$ $\Psi ^ { \dagger } ( { \sf K } )$ satisfies the Smoothed Approximation Assumption 3.1 in $C ^ { s ^ { \prime } } ( \Omega )$ with respect to $\{ \eta _ { j } \} _ { j = 1 } ^ { \infty }$ . Then for any $\epsilon > 0$ , there exists $J \in \mathbb N$ , and continuous functionals $\alpha _ { 1 } , \ldots , \alpha _ { J } : L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to \mathbb { R }$ such that the operator $\widetilde { \Psi } ^ { \dagger } : L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to C ^ { s ^ { \prime } } ( \Omega ; \mathbb { R } ^ { k ^ { \prime } } )$ defined by

$$
\widetilde { \Psi } ^ { \dagger } ( u ) = \sum _ { j = 1 } ^ { J } \alpha _ { j } ( u ) \eta _ { j }
$$

satisfies

$$
\operatorname* { s u p } _ { u \in \mathsf { K } } \| \Psi ^ { \dagger } ( u ) - \widetilde \Psi ^ { \dagger } ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \leq \epsilon .
$$

Remark 3.9. The purpose of defining the approximation $\widetilde { \Psi } ^ { \dagger }$ as a mapping from $L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } )$ is to avoid needing to modify the proof for various values of s. Note that $C ^ { s } ( \Omega ; \mathbb { R } ^ { k } )$ is continuously embedded in $L ^ { 1 } ( \bar { \Omega } ; \mathbb { R } ^ { k } )$ ♢

We defer the proof of Lemma 3.8 to Appendix C.

3.3. NNO Approximation of Functionals and Basis Functions. In this subsection we describe and state the two lemmas that are key to proving Theorems 3.4 and 3.5. These lemmas show that we can approximate each term $\alpha _ { j } ( u ) \eta _ { j }$ in the sum in Lemma 3.8 by a NNO. The first lemma, Lemma 3.10, achieves the approximation in the setting of a single hidden layer and basis function pair, corresponding to Theorem 3.4. The second lemma, Lemma 3.11, applies to the slightly more general setting of Theorem 3.5 and uses the basis functions in the NNO architecture to represent the $\eta _ { j }$ in 3.8. The proofs of both lemmas may be found in Appendix D.

Lemma 3.10. Let $\alpha : L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to \mathbb { R }$ be a continuous functional and let $\textsf { K } \subset$ $L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } )$ be a compact set consisting of bounded functions su $\mathrm { p } _ { u \in \mathsf { K } } \| u \| _ { L ^ { \infty } ( \Omega ) } < \infty .$ Assume $\Omega \subset \mathbb { R } ^ { d }$ is a bounded Lipschitz domain. Assume, in addition, that $\eta \in$ $C ^ { s ^ { \prime } } ( \overline { { \Omega } } ; \mathbb { R } ^ { k ^ { \prime } } )$ for integer $s ^ { \prime } \geq 0$ . Then for any $\varepsilon > 0$ there exists a nonlocal neural operator $\widetilde { \alpha } : L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to C ^ { s ^ { \prime } } ( \Omega ; \mathbb { R } ^ { k ^ { \prime } } )$ as defined in Definition 2.1, with Assumption 2.2 on the activation satisfied with $s _ { \sigma } = s ^ { \prime }$ , such that

$$
\operatorname* { s u p } _ { u \in \mathsf { K } } \| \alpha ( u ) \eta - \widetilde { \alpha } ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } < \varepsilon\tag{3.3}
$$

with a single hidden layer $( M = 1 )$ and with any single basis function pair $\psi _ { 1 , 1 } : = \eta _ { 1 }$ and $\phi _ { 1 , 1 } : = \eta _ { 2 }$ satisfying Assumptions 3.3.

Lemma 3.11. Let $\alpha : L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to \mathbb { R }$ be a continuous functional and let $\textsf { K } \subset$ $L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } )$ be a compact set consisting of bounded functions sup $\ L _ { u \in \mathsf { K } } \| u \| _ { L ^ { \infty } ( \Omega ) } < \infty$ Assume $\Omega \subset \mathbb { R } ^ { d }$ is a bounded Lipschitz domain. Assume, in addition, that $\eta \in$ $C ^ { s ^ { \prime } } ( \Omega ; \mathbb { R } ^ { k ^ { \prime } } )$ for integer $s ^ { \prime } \geq 0$ . Then for any $\varepsilon > 0$ there exists a nonlocal neural operator $\widetilde { \alpha } : L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to C ^ { s ^ { \prime } } ( \Omega ; \mathbb { R } ^ { k ^ { \prime } } )$ as defined in Definition $2 . 1 ,$ with Assumption 2.2 on the activation satisfied with $s _ { \sigma } = s ^ { \prime }$ , such that

$$
\operatorname* { s u p } _ { u \in \mathsf { K } } \| \alpha ( u ) \eta - \widetilde { \alpha } ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } < \varepsilon ,\tag{3.4}
$$

where the final hidden layer contains η as an outer basis element, i.e. for some $m ^ { * } \le M , \phi _ { L , m ^ { * } } : = \eta$ with a corresponding inner basis function $\psi _ { L , m ^ { * } } \in L ^ { 1 } ( \Omega )$ satisfying $( E 2 ) \ – ( E 4 )$ in Assumptions 3.3 and an earlier hidden layer contains at least one basis function pair $\eta _ { 1 }$ satisfying $( E 1 ) \ – ( E _ { 4 } )$ and $\eta _ { 2 }$ satisfying $( E 2 ) \ – ( E 4 )$

Remark 3.12. Lemmas 3.10 and 3.11 are similar, but notable in two main diferences. First, Lemma 3.11 is slightly more general in that it only requires that η be $C ^ { s ^ { \prime } }$ on Ω rather than on the closure of Ω. In return, we sacrifice using only a single hidden layer and single basis function as in Lemma 3.10. However, the proof for Lemma 3.11 is appealing from an intuition standpoint because the basis functions in the NNO itself are actually used as the approximating basis, whereas in the proof of Lemma 3.10, as well as several other operator learning approximation proofs, a finite-dimensional neural network is employed to approximate η instead of using a basis function that is already part of the architecture. In practice, one never uses a single basis element, so we find it far more natural to view universal approximation as occurring through the lens of Lemma 3.11. ♢

3.4. Proofs of Theorems 3.4 and 3.5. We may now prove our main theorems. As the proofs are identical except for the use of diferent lemmas, we present them as one.

Proof of Theorems $\it 3 . 4$ and 3.5 . Fix any $\epsilon _ { 0 } > 0$ . By Lemma 3.8, there is $J > 0$ such that we may approximate $\Psi ^ { \dagger }$ by

$$
\widetilde { \Psi } ^ { \dagger } ( u ) = \sum _ { j = 1 } ^ { J } \alpha _ { j } ( u ) \eta _ { j } ,
$$

to obtain

$$
\operatorname* { s u p } _ { u \in \mathsf { K } } \| \Psi ^ { \dagger } ( u ) - \widetilde { \Psi } ^ { \dagger } ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } < \epsilon _ { 0 } .
$$

Then by Lemma 3.10 for Theorem 3.4, respectively Lemma 3.11 for Theorem 3.5, each $\alpha _ { j } \eta _ { j }$ may be approximated by an NNO $\widetilde { \alpha } _ { j }$ satisfying the appropriate architecture conditions corresponding to each theorem-lemma pair such that for any $\epsilon _ { 1 } > 0$

$$
\operatorname* { s u p } _ { u \in \mathsf { K } } \| \alpha _ { j } ( u ) \eta _ { j } - \widetilde { \alpha } _ { j } ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \leq \epsilon _ { 1 } , ~ \mathrm { f o r ~ a l l } ~ j \in [ J ] .
$$

Define $\begin{array} { r } { \Psi : = \sum _ { j = 1 } ^ { J } \widetilde { \alpha } _ { j } } \end{array}$ , the parallelization and subsequent sum in the final afine layer of the component NNOs. Then we have that

$$
\begin{array} { r l } {  { \operatorname* { s u p } _ { u \in \mathsf { K } } \| \Psi ^ { \dagger } ( u ) - \Psi ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } = \operatorname* { s u p } _ { u \in \mathsf { K } } \| \Psi ^ { \dagger } ( u ) - \displaystyle \sum _ { j = 1 } ^ { J } \widetilde \alpha _ { j } ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } } } \\ & { \le \operatorname* { s u p } _ { u \in \mathsf { K } } \| \Psi ^ { \dagger } - \displaystyle \sum _ { j = 1 } ^ { J } \alpha _ { j } ( u ) \eta _ { j } \| _ { C ^ { s ^ { \prime } } ( \Omega ) } + \| \sum _ { j = 1 } ^ { J } \alpha _ { j } ( u ) \eta _ { j } - \displaystyle \sum _ { j = 1 } ^ { J } \widetilde \alpha _ { j } ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } } \\ & { \le \epsilon _ { 0 } + J \operatorname* { m a x } _ { j \in [ J ] } \| \alpha _ { j } ( u ) \eta _ { j } - \widetilde \alpha _ { j } ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } } \\ & { \le \epsilon _ { 0 } + J \epsilon _ { 1 } . } \end{array}
$$

Choosing $\epsilon _ { 0 } < \frac { \epsilon } { 2 }$ and $\epsilon _ { 1 } < \frac { \epsilon } { 2 J }$ yields the results in (3.1) and (3.2).

## 4. Numerical Experiments

In this section we consider two operator learning problems defined by a coeficient to solution map in a PDE, one based on Darcy flow on a square domain and the other a variant of the Helmholtz equation on a circular domain. We implement a neural operator that enforces Dirichlet boundary conditions absolutely and test it on the two example problems. We demonstrate both the feasibility of our proposed method, and its competitiveness, in terms of cost-accuracy trade-of, in relation to other pre-existing methods.<sup>2</sup>

We first detail the architecture implementation in Subsection 4.1, and then we describe the data generation process for the Darcy and Helmholtz experiments in Subsections 4.2 and 4.3, respectively. We present and discuss results from both examples in Subsection 4.4.

4.1. Architecture. While the theory applies more generally, in implementation we make the natural choice of projecting onto the eigenfunctions of the Laplacian with Dirichlet boundary conditions on the domain associated with the application problem. In particular, we use the Dirichlet layer in Definition 2.3 as the final hidden layer in our architecture. By setting $W _ { \ell } = 0$ and $b _ { \ell } = 0$ , this layer will enforce zero Dirichlet boundary conditions absolutely for the output function, as proved in Corollary 3.6. Additionally, we enforce zero bias in the final neural network layer Q. GeLU activations are used throughout the model which, we recall, satisfy Assumptions 2.2 for any $s _ { \sigma } \geq 0$ , including the requirement that $\sigma ( 0 ) = 0$

We compare our method with a standard implementation of the FNO as well as three possible alternative methods to enforce the boundary conditions exactly via projection. Let $\{ \eta _ { j } \} _ { j = 1 } ^ { J }$ be the first J eigenfunctions of the Laplacian, and let the projection operator $\bar { P } _ { J }$ be defined via

$$
P _ { J } v = \sum _ { j = 1 } ^ { J } \langle \eta _ { j } , v \rangle _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d _ { c } } ) } \eta _ { j } .\tag{4.1}
$$

For the first comparison, we project the output of the trained FNO $\Psi _ { \mathrm { F N O } } ~ a f t e r$ training to obtain the prediction $P _ { J } \Psi _ { \mathrm { F N O } } ( a )$ for input $a .$ For the second comparison, we incorporate this projection as part of the pipeline during training. For the third comparison, we bypass neural networks entirely and simply perform regularized least squares (ridge regression) on the coeficients of the input and output functions projected onto the first $J$ eigenfunctions of the Laplacian. For this basic comparison, while the input does not satisfy Dirichlet boundary conditions, the eigenfunctions are still a basis for $L ^ { 2 } ( \Omega )$ . All of these approaches enforce the boundary condition absolutely. The last one corresponds to making a linear approximation to the nonlinear map $\Psi ,$ , whilst the proposed method and the other two comparisons all result in an approximation of the map Ψ which is nonlinear.

In what follows we aim to approximate $\Psi ^ { \dag } : \mathcal { X } ( \Omega ; \mathbb { R } ^ { o } )  \mathcal { Y } ( \Omega ; \mathbb { R } ^ { o } )$ acting on $^ { a , }$ coeficient of a PDE, to produce $u ,$ solution of the PDE. We assume we are given data $\{ a _ { n } , u _ { n } \} _ { n = 1 } ^ { N }$ with $a _ { n } \sim \mu$ i.i.d. and $u _ { n } = \Psi ^ { \dagger } ( a _ { n } )$ . The population objective function that we aim to approximately minimize, by choice of parameters $\theta$ in $\Psi ( \cdot ; \theta ) : \mathcal { X } ( \Omega ; \mathbb { R } ^ { o } ) \to \mathcal { Y } ( \Omega ; \mathbb { R } ^ { o } )$ , is

$$
\mathsf { J } ( \theta ) = \mathbb { E } ^ { a \sim \mu } \frac { \| \Psi ( a ; \theta ) - \Psi ^ { \dagger } ( a ) \| _ { L ^ { 2 } ( \Omega ) } } { \| \Psi ^ { \dagger } ( a ) \| _ { L ^ { 2 } ( \Omega ) } } .
$$

In practice we approximate this empirically using the data to obtain objective

$$
\mathsf { J } ^ { N } ( \theta ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \frac { \| \Psi ( a _ { n } ; \theta ) - u _ { n } \| _ { L ^ { 2 } ( \Omega ) } } { \| u _ { n } \| _ { L ^ { 2 } ( \Omega ) } } .
$$

4.2. Darcy Flow. Let $\Omega = [ 0 , 1 ] ^ { 2 }$ be the unit square. We are interested in the map from $a \in L ^ { \infty } ( \Omega ; \mathbb { R } )$ to $u \in H _ { 0 } ^ { 1 } ( \Omega ; \mathbb { R } )$ given by

$$
\begin{array} { r } { - \nabla \cdot ( a \nabla u ) = 1 , \quad x \in \Omega } \\ { u = 0 , \quad x \in \partial \Omega . } \end{array}
$$

In this example, we use the dataset associated with the original work on FNOs [28]. The coeficients a are randomly generated from $\mu : = \psi _ { \# } { \mathcal { N } } ( 0 , C )$ where $C =$ $( - \dot { \Delta } + 9 I ) ^ { - 2 }$ equipped with zero Neumann boundary conditions to define the inverse of the identity shifted Laplacian. The map ψ is a Nemytskii operator defined by a piecewise constant function taking values 3 and 12 on the negative and positive parts of the real line, respectively. The solution was obtained via a second-order finite-diference scheme on a 421 × 421 grid, as stated in [28]. In total the data set has 2048 samples. We set aside 248 of the samples for validation used only for hyperparameter tuning, and 200 of the samples for test data, which were not used to choose hyperparameters.

For this example, since the data are uniform and on the unit square, we used FNO layers in the DNO until the final layer to demonstrate the advantage of the DNO in modifying only the final layer. One can verify from Theorem 3.5 that combining diferent types of layers in this manner still satisfies the conditions for universality.

4.3. Helmholtz. Let Ω be the unit disk in $\mathbb { R } ^ { 2 }$ . We are interested in the map from $a \in L ^ { \infty } ( \Omega ; \mathbb { R } )$ and $b \in \mathbb { R }$ to the solution u given by

$$
\begin{array} { r } { - \Delta u - \omega ^ { 2 } a u = 0 , \quad x \in \Omega } \\ { u = b , \quad x \in \partial \Omega , } \end{array}
$$

where $\begin{array} { r } { \omega = \frac { 5 \pi } { 2 } } \end{array}$ . The coeficient a is randomly generated as a sum of Gaussians. The number n of Gaussians is uniformly sampled from integers in [[2, 7]], the centers of the Gaussians are uniformly sampled in Ω, the amplitudes are sampled uniformly $\sim ( 2 , 1 0 )$ , and the standard deviation is sampled uniformly $\sim ( 0 . 0 5 , 0 . 1 )$ . Following summation, the resulting field ea is normalized via

$$
a = \frac { \widetilde { a } - a _ { \mathrm { m i n } } } { a _ { \mathrm { m a x } } - a _ { \mathrm { m i n } } } .
$$

This ensures that a is non-negative and takes values in [0, 1], pointwise. The boundary condition b is uniformly sampled $\sim [ 0 . 2 5 , 5 ]$ . The numerical solution is obtained via the finite element method in FEniCSx using CG elements of degree 1 on a triangular mesh of the unit disk [5, 2, 4]. Since the operator $- \Delta - \omega ^ { 2 } a$ is not invertible for all choices of $^ { a , }$ the solution operator is unbounded unless we restrict our choice of input domain $( a , b )$ further. To do so, after obtaining numerical solution u, we reject the sample if $\begin{array} { r } { \frac { \operatorname* { m a x } _ { x \in \Omega } | u ( x ) | } { b } > 1 0 } \end{array}$ . This can be thought of as cutting out holes around the singularities in the domain $( a , b )$ ; it can also be thought of as conditioning the input measure as previously described. Doing so is necessary for the true solution map to exist on our choice of input set. To adapt the data for the FNO, we lift b to a function by $b \mapsto b \mathbf { 1 }$ and concatenate a and b1 as the input to the FNO. For the DNO, we also lift $b \mapsto b \mathbf { 1 }$ and transform the output data to be $u - b \mathbf { 1 }$ so that it has zero boundary condition. Then the map $( a , b \mathbf { 1 } ) \mapsto u - b \mathbf { 1 }$ is learned, and in postprocessing we add b1 back to obtain the true solution. For this example, we produced 11000 data samples and set aside 500 for validation data and 500 for test data.

Since the Helmholtz data are generated on a triangular mesh, we are able to pass this mesh directly to the DNO by using Dirichlet layers as all hidden layers. On the other hand, to adapt the data for the FNO, we interpolate the data to a uniform grid with approximately the same number of interior data points. We then pass to the FNO a box domain containing the circular domain with a zero mask applied outside the circular domain. For FNO, the error comparison of prediction to truth is performed on the interpolated grid instead of once more interpolating the FNO output back to the FEM mesh. In doing so, we give FNO a slight advantage because the error is not further increased via additional interpolation.

![](images/4153365b3a106196df64ed449bc7b54123ffa622562419864be3cc1736c86bfd.jpg)

Figure 1. DNO predictions on Darcy flow. Relative $L ^ { 2 }$ error for the best, median, and worst test samples, respectively: 0.482%, 0.772%, and 2.02%.  
![](images/bd1e7fa0a46800c21bd1de509ce482e9030f9498a0d3b945a191020e7fdca893.jpg)  
Figure 2. DNO predictions for the Helmholtz example. Relative $L ^ { 2 }$ error for the best, median, and worst test samples, respectively: 0.0283%, 0.130%, 4.45%.

4.4. Results and Discussion. For both PDE examples we ran a hyperparameter gridsearch, doing so for the DNO and for the FNO. And in both PDE examples, we varied the number of eigenfunctions (resp. Fourier modes) in each dimension, the channel width, the batch size, the number of layers, and the number of training epochs. For the Darcy problem, the resulting choice of hyperparameters for the DNO was 12 modes in each degree of freedom, a channel width of 64, 300 epochs, a batch size of 15, 4 FNO layers, and one final DNO layer. For the FNO, the resulting choice was the same with 12 modes in each dimension, a channel width of 64, 300 epochs, a batch size of 14, and 4 hidden layers. For the Helmholtz problem, the resulting choices for DNO were 16 modes, a width of 64, 300 epochs, a batch size of

![](images/9f61c787033044841984f94d1efb4ce4ec3ba2351e8f94e467c9a7ebd5e5db50.jpg)  
Figure 3. Darcy example hyperparameter search results.

Hyperparameter Optimization for Helmholtz  
![](images/5923906039e4f2175a82ffee1e4d0929a387635c215ac9e0d02bd11519b3e198.jpg)  
Figure 4. Helmholtz example hyperparameter search results.

![](images/0ef452cc52727a5bc8ea3165fb272aa05c2ecdccab4bbffb8a90bc92ed374c73.jpg)  
Figure 5. Data eficiency for the Darcy flow example for FNO and DNO as well as projected FNO before and after training.

![](images/90e0515037edbc649806245825fc6398690dc00a48ee9365f2f68debf25ff471.jpg)  
Figure 6. Data eficiency for the Helmholtz example for FNO and DNO as well as projected FNO before and after training and regression.

25, and 4 DNO layers. For FNO, the choices were 12 modes, a width of 64, a batch size of 25, 300 epochs, and 4 layers. We note that the training time of these models were comparable between the FNO and DNO in both examples. A visualization of the input and output functions as well as predictions for both problems can be seen in Figures 1 and 2, for the Darcy and Helmholtz examples, respectively. Figures 3 and 4 give a visualization of the relative validation error versus the training time for the hyperparameter search for the Darcy and Helmholtz examples, respectively.

These figures also give empirical evidence that the DNO is more robust to the suboptimal hyperparameter choices that will be made in practice.

We then took models with these hyperparameters and performed data eficiency comparisons of DNO, FNO, and the FNO projected onto the Laplacian eigenfunction basis both before and after training, as well as a basic ridge regression, as detailed in Subsection 4.1. The results of relative test error versus training data size N are shown in Figures 5 and 6. The shaded uncertainty regions in both figures represent two standard deviations away from the mean of five models trained on the same amount of data; the ridge regression example has no uncertainty as it results from a deterministic linear solve instead of from stochastic gradient descent. The poor performance of ridge regression demonstrates that the map Ψ being learned is indeed significantly nonlinear. For the Darcy example, the clearest benefit appears in the low data regime, where the DNO shows a $2 \times$ accuracy improvement over any of the other model choices; the other nonlinear model choices fail to beat even ridge regression. The benefit of DNO remains present for higher amounts of data as well; for 1600 data points the DNO shows a 20% accuracy improvement over FNO, which in turn beats both projection approaches. Indeed, it is somewhat surprising that neither projection approach improved on the basic FNO implementation; this implies that the benefit of DNO goes beyond simply getting the boundary condition right. For the Helmholtz example, all of the models perform similarly at extremely low levels of data $( N = 1 0 )$ , but the DNO shows roughly a 30% accuracy improvement over the other models for $N = 2 0$ and $N = 4 0$ . Surprisingly, in this example the benefit of DNO shows up most in the higher data regime, showing approximate $2 \times$ improvement in accuracy for $N = 5 0 0 0$ and $N = 1 0 0 0 0$ . In this Helmholtz example, projecting the FNO during training did yield some benefit in the high data regime, but, interestingly, projecting after training caused the error to more than double; we believe this error originates from the interpolation error between the native mesh of the FEM solver and the uniform mesh of the FNO. We emphasize that this phenomenon is not mitigated by increasing J in the projection operator $P _ { J }$

Intuitive visualizations of the best, median, and worst test samples are shown for DNO for the Darcy and Helmholtz experiments in Figures 1 and 2, respectively, at training data counts of $N = 1 6 0 0$ and $N = 1 0 0 0 0$ . We comment that for the Helmholtz example, the worst test sample occurs near the cutof for a singularity as described in Subsection 4.3 as expected; this was also the worst test sample for FNO.

## 5. Conclusion

We propose a choice of basis for the nonlocal neural operator that automatically satisfies Dirichlet boundary conditions by using the eigenfunctions of the Laplacian. The approximation theory developed for this approach additionally goes far beyond this specific setting and applies to a broad class of bases for $L ^ { 2 }$ with only Lipschitz boundary assumptions on a domain in any Euclidean space. We implement our proposed “Dirichlet Neural Operator” on two representative examples, including the Darcy flow problem on a square and the Helmholtz equation on a circular domain. The latter example in particular demonstrates the advantage of the method on arbitrary mesh points as well as with nonzero Dirichlet boundary. The ability of the method to enforce boundary conditions without training on arbitrary meshes and domains distinguishes it from prior work on the subject.

Several possible extensions are suggested by the work. First, since the theory applies to a broad class of possible bases, additional basis choices such as finite element bases could be implemented to achieve other desirable properties such as diferent boundary conditions or possibly boundedness guarantees. The choice of Fourier basis for the FNO has allowed for numerical analysis yielding rates in terms of model size parameters; similar rates could be achieved in diferent settings depending on the choice of basis. Furthermore, it remains an open question how to automatically enforce mixed boundary conditions on arbitrary domains that are defined separately on a partition $\Gamma _ { 1 } \cup \cdot \cdot \cdot \cup \Gamma _ { J } = \partial \Omega$ with operator learning.

## Acknowledgments

The authors thank Manuel Santana for advice on the numerical solution of the Helmholtz equation and Samuel Lanthaler for technical insights and advice.

AI acknowledgment: Claude Sonnet 4.6 was used to identify sources for theory on the eigenfunctions of the Laplacian. Claude Opus 4.8 made a key suggestion to achieve the proof of Lemma 3.10; namely, the exact “1” channel. Cursor was used as a code development aid, including in the numerical solution of the Helmholtz equation, implementation of the Dirichlet layer, and figure generation scripts. Claude Code with Opus 4.8 was used as a code development aid for the linear regression comparison. The authors take full responsibility for all the theoretical and numerical results in this manuscript. Furthermore, AI was not used to write any portion of the manuscript nor to determine which references to cite.

## References

1. Diab W Abueidda, Panos Pantidis, and Mostafa E Mobasher, DeepOKAN: Deep operator network based on Kolmogorov Arnold networks for mechanics problems, Computer Methods in Applied Mechanics and Engineering 436 (2025), 117699.

2. Martin S Alnæs, Anders Logg, Kristian B Ølgaard, Marie E Rognes, and Garth N Wells, Unified form language: A domain-specific language for weak formulations of partial diferential equations, ACM Transactions on Mathematical Software (TOMS) 40 (2014), no. 2, 1–37.

3. Kamyar Azizzadenesheli, Nikola Kovachki, Zongyi Li, Miguel Liu-Schiafini, Jean Kossaifi, and Anima Anandkumar, Neural operators for accelerating scientific simulations and design, Nature Reviews Physics 6 (2024), no. 5, 320–328.

4. Satish Balay, Shrirang Abhyankar, Mark Adams, Jed Brown, Peter Brune, Kris Buschelman, Lisandro Dalcin, Alp Dener, Victor Eijkhout, William Gropp, et al., PETSc users manual, (2019).

5. Igor A Baratta, Joseph P Dean, Jørgen S Dokken, Michal Habera, Jack HALE, Chris N Richardson, Marie E Rognes, Matthew W Scroggs, Nathan Sime, and Garth N Wells, DOLFINx: the next generation fenics problem solving environment, (2023).

6. Stefano Berrone, Claudio Canuto, Moreno Pintore, and Natarajan Sukumar, Enforcing Dirichlet boundary conditions in physics-informed neural networks and variational physicsinformed neural networks, Heliyon 9 (2023), no. 8.

7. Kaushik Bhattacharya, Lianghao Cao, George Stepaniants, Andrew Stuart, and Margaret Trautner, Learning memory and material dependent constitutive laws, arXiv preprint arXiv:2502.05463 (2025).

8. Kaushik Bhattacharya, Bamdad Hosseini, Nikola B. Kovachki, and Andrew M. Stuart, Model reduction and neural networks for parametric PDEs, The SMAI Journal of Computational Mathematics 7 (2021), 121–157 (en), Publisher: Soci´et´e de Math´ematiques Appliqu´ees et Industrielles.

9. John P Boyd, Chebyshev and Fourier spectral methods, Courier Corporation, 2001.

10. Qianying Cao, Somdatta Goswami, and George Em Karniadakis, Laplace neural operator for solving diferential equations, Nature Machine Intelligence 6 (2024), no. 6, 631–640.

11. Samuel N Cohen, Filippo de Feo, Jackson Hebner, and Justin Sirignano, Deep Hilbert– Galerkin methods for infinite-dimensional PDEs and optimal control, arXiv preprint arXiv:2603.19463 (2026).

12. Samuel N Cohen, Deqing Jiang, and Justin Sirignano, Neural q-learning for solving PDEs, Journal of Machine Learning Research 24 (2023), no. 236, 1–49.

13. Edward Brian Davies and Barry Simon, Ultracontractivity and the heat kernel for Schr¨odinger operators and Dirichlet Laplacians, Journal of Functional Analysis 59 (1984), no. 2, 335–395.

14. Suchuan Dong and Yuchuan Zhang, A novel method for enforcing exactly Dirichlet, Neumann and Robin conditions on curved domain boundaries for physics informed machine learning, arXiv preprint arXiv:2603.21909 (2026).

15. Xinghao Dong, Chuanqi Chen, and Jin-Long Wu, Data-driven stochastic closure modeling via conditional difusion model and neural operator, Journal of Computational Physics 534 (2025), 114005.

16. Lawrence C Evans, Partial diferential equations, 2nd ed., vol. 19, American Mathematical Society, Providence, Rhode Island, 2010.

17. David Gilbarg, Neil S Trudinger, David Gilbarg, and NS Trudinger, Elliptic partial diferential equations of second order, vol. 2, Springer, 1998.

18. Giorgio Gnecco, Marco Gori, and Marcello Sanguineti, Learning with boundary conditions, Neural computation 25 (2013), no. 4, 1029–1106.

19. Somdatta Goswami, Aniruddha Bora, Yue Yu, and George Em Karniadakis, Physics-informed deep neural operator networks, Machine learning in modeling and simulation: methods and applications, Springer, 2023, pp. 219–254.

20. Derek Hansen, Danielle C Maddix, Shima Alizadeh, Gaurav Gupta, and Michael W Mahoney, Learning physical models that can respect conservation laws, International Conference on Machine Learning, PMLR, 2023, pp. 12469–12510.

21. Jan S Hesthaven and Stefano Ubbiali, Non-intrusive reduced order modeling of nonlinear problems using neural networks, Journal of Computational Physics 363 (2018), 55–78, Publisher: Elsevier.

22. Masanobu Horie and Naoto Mitsume, Physics-embedded neural networks: Graph neural PDE solvers with mixed boundary conditions, Advances in Neural Information Processing Systems 35 (2022), 23218–23229.

23. Dunham Jackson, The theory of approximation, vol. 11, American mathematical society, 1930.

24. Deqing Jiang, Justin Sirignano, and Samuel N Cohen, Global convergence of deep Galerkin and PINNs methods for solving partial diferential equations, arXiv preprint arXiv:2305.06000 (2023).

25. Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar, Neural operator: Learning maps between function spaces with applications to PDEs, Journal of Machine Learning Research 24 (2023), no. 89, 1–97.

26. Nikola B Kovachki, Samuel Lanthaler, and Andrew M Stuart, Operator learning: Algorithms and analysis, Handbook of Numerical Analysis 25 (2024), 419–467.

27. Samuel Lanthaler, Zongyi Li, and Andrew M Stuart, Nonlocality and nonlinearity implies universality in operator learning, Constructive Approximation 62 (2025), 1–43.

28. Zongyi Li, Nikola Borislavov Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar, Fourier neural operator for parametric partial diferential equations, International Conference on Learning Representations, 2021.

29. Ziyuan Liu, Haifeng Wang, Hong Zhang, Kaijun Bao, Xu Qian, and Songhe Song, Render unto numerics: Orthogonal polynomial neural operator for PDEs with nonperiodic boundary conditions, SIAM Journal on Scientific Computing 46 (2024), no. 4, C323–C348.

30. Ziyuan Liu, Yuhang Wu, Daniel Zhengyu Huang, Hong Zhang, Xu Qian, and Songhe Song, Spfno: Spectral operator learning for PDEs with Dirichlet and Neumann boundary conditions, arXiv preprint arXiv:2312.06980 (2023).

31. Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis, Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators, Nature machine intelligence 3 (2021), no. 3, 218–229.

32. Kevin Stanley McFall and James Robert Mahan, Artificial neural network method for solution of boundary value problems with exact satisfaction of arbitrary boundary conditions, IEEE transactions on neural networks 20 (2009), no. 8, 1221–1233.

33. Sepehr Mousavi, Siddhartha Mishra, and Laura De Lorenzis, Imposing boundary conditions on neural operators via learned function extensions, arXiv preprint arXiv:2602.04923 (2026).

34. Ryumei Nakada and Masaaki Imaizumi, Adaptive approximation and generalization of deep neural network with intrinsic dimensionality, Journal of Machine Learning Research 21 (2020), no. 174, 1–38.

35. Nicholas H Nelsen and Andrew M Stuart, The random feature model for input-output maps between Banach spaces, SIAM Journal on Scientific Computing 43 (2021), no. 5, A3212– A3243.

36. Yu Netrusov and Yu Safarov, Weyl asymptotic formula for the Laplacian on domains with rough boundaries, Communications in mathematical physics 253 (2005), no. 2, 481–509.

37. Zhong Peng, Bo Yang, Yixian Xu, Feng Wang, Lian Liu, and Yi Zhang, Rapid surrogate modeling of electromagnetic data in frequency domain using neural operator, IEEE Transactions on Geoscience and Remote Sensing 60 (2022), 1–12.

38. Allan Pinkus, Approximation theory of the MLP model in neural networks, Acta Numerica 8 (1999), 143–195.

39. Nadim Saad, Gaurav Gupta, Shima Alizadeh, and Danielle C. Maddix, Guiding continuous operator learning through physics-based boundary constraints, The Eleventh International Conference on Learning Representations, 2023.

40. Jie Shen, Tao Tang, and Li-Lian Wang, Spectral methods: algorithms, analysis and applications, vol. 41, Springer Science & Business Media, 2011.

41. Tao Tang, Spectral and high-order methods with applications, Science Press Beijing, 2006.

42. Tapas Tripura and Souvik Chakraborty, Wavelet neural operator for solving parametric partial diferential equations in computational mechanics problems, Computer Methods in Applied Mechanics and Engineering 404 (2023), 115783.

43. Hermann Weyl, Das asymptotische verteilungsgesetz der eigenwerte linearer partieller diferentialgleichungen (mit einer anwendung auf die theorie der hohlraumstrahlung), Mathematische Annalen 71 (1912), no. 4, 441–479.

44. Konstantin Yakovlev and Nikita Puchkin, Approximation capabilities of feedforward neural networks with gelu activations, arXiv preprint arXiv:2512.21749 (2025).

45. Minglang Yin, Ehsan Ban, Bruno V Rego, Enrui Zhang, Cristina Cavinato, Jay D Humphrey, and George Em Karniadakis, Simulating progressive intramural damage leading to aortic dissection using DeepONet: an operator–regression neural network, Journal of the Royal Society Interface 19 (2022), no. 187.

46. Huaiqian You, Quinn Zhang, Colton J Ross, Chung-Hao Lee, Ming-Chen Hsu, and Yue Yu, A physics-guided neural operator learning approach to model biological tissues from digital image correlation measurements, Journal of Biomechanical Engineering 144 (2022), no. 12, 121012.

## Appendix A. Properties of finite-dimensional neural networks

Definition A.1. A finite-dimensional feedforward neural network $G : \mathbb { R } ^ { d _ { x } }  \mathbb { R } ^ { d _ { y } }$ takes the form of compositions

$$
G ( x ) = A _ { L } L _ { L - 1 } \circ \cdots \circ L _ { 1 } ( x ) + b _ { L } , \quad x \in \mathbb { R } ^ { d _ { x } }
$$

where $L _ { l } : \mathbb { R } ^ { d _ { l } }  \mathbb { R } ^ { d _ { l + 1 } } , d _ { 1 } = d _ { x } , d _ { L + 1 } = d _ { y } .$ , and

$$
L _ { l } ( x _ { l } ) = \sigma ( A _ { l } x _ { l } + b _ { l } ) , \quad x _ { l } \in \mathbb { R } ^ { d _ { l } } ,
$$

and $A _ { l } \in \mathbb { R } ^ { d _ { l + 1 } \times d _ { l } }$ is a matrix and $b _ { l } \in \mathbb { R } ^ { d _ { l + 1 } }$ is a vector. The activation $\sigma : \mathbb { R } $ R is a $C ^ { s }$ , non-polynomial function for some integer $s \geq 0$ , that is extended to act component-wise on vector inputs. ♢

The following result is a well-known fact about the approximation capability of finite-dimensional feedforward neural networks. We restate it here for convenience and refer to [38, Theorem 4.1] for the proof.

Lemma A.2. Let $\Omega \subset \mathbb { R } ^ { d }$ be a bounded domain, and assume $\sigma \in C ^ { s } ( \mathbb { R } )$ is not a polynomial. Let $u \in C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { k } )$ for some integer $s \geq 0$ . Then, for any nonpolynomial activation $\sigma \in C ^ { s }$ as in Definition A.1, and $f o r$ any $\epsilon > 0$ , there exists a finite-dimensional neural network ue such that

$$
\| u - \widetilde { u } \| _ { C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { k } ) } = \operatorname* { s u p } _ { | \alpha | \leq s } \operatorname* { s u p } _ { x \in \overline { { \Omega } } } | D ^ { \alpha } u ( x ) - D ^ { \alpha } \widetilde { u } ( x ) | \leq \epsilon .
$$

## Appendix B. Proof of Proposition 3.2

Here we provide a proof of Proposition 3.2, which details some familiar examples where the Smoothed Approximation Assumption holds. Throughout this proof, we refer to the three conditions of the Smoothed Approximation Assumption 3.1 by their enumerations (S1), (S2), and (S3).

Proof. Case 1: We note that this case is largely taken from the proof of $[ 2 7 ,$ Proposition A.6]. By [27, Lemma A.1], there exists a continuous extension mapping $\mathcal { E } : C ^ { s ^ { \prime } } ( \overline { { \Omega } } )  C _ { \mathrm { p e r } } ^ { s ^ { \prime } } ( B )$ . Then $\mathsf { K } _ { \mathrm { p e r } } ^ { \prime } : = \mathcal { E } ( \mathsf { K } ^ { \prime } )$ is a compact subset of $C _ { \mathrm { p e r } } ^ { s ^ { \prime } } ( B )$ , and we can identify $B \simeq  { \mathbb { T } } ^ { d }$ in a canonical way. Fix $\delta > 0$ , and let $w _ { \delta } ^ { \prime } : B \to \mathbb { R }$ be the standard mollification (see Definition D.3) of the periodic function $w ^ { \prime } : B \to \mathbb { R }$ We may then define the family of linear maps $\{ f _ { \delta } \} _ { \delta > 0 }$ by the composition of the mollification and periodic extension operator; i.e., $f _ { \delta } ( w ) : = \mathcal { E } ( w ) _ { \delta }$ . The standard mollifier satisfies (S1):

$$
\begin{array} { r l r } {  { \operatorname* { l i m } _ { \delta \to 0 } \operatorname* { s u p } _ { w \in \mathsf { K } ^ { \prime } } \| f _ { \delta } ( w ) - w \| _ { C ^ { s ^ { \prime } } ( \overline { { \Omega } } ) } = \operatorname* { l i m } _ { \delta \to 0 } \operatorname* { s u p } _ { w \in \mathsf { K } ^ { \prime } } \| \mathcal { E } ( w ) _ { \delta } - w \| _ { C ^ { s ^ { \prime } } ( \overline { { \Omega } } ) } } } \\ & { } & { \quad = \operatorname* { l i m } _ { \delta \to 0 } \operatorname* { s u p } _ { w ^ { \prime } \in \mathsf { K } _ { \mathrm { p e r } } ^ { \prime } } \| w _ { \delta } ^ { \prime } - w ^ { \prime } \| _ { C ^ { s ^ { \prime } } ( \overline { { \Omega } } ) } = 0 . } \end{array}
$$

Since the extension operator and the mollifier are continuous, property (S3) is also satisfied. It follows from classical results in harmonic analysis [23, Chapter 1, Section 3] that approximation by Fourier series on $C ^ { s ^ { \prime } } ( B )$ converges uniformly over the mollified set $\{ w _ { \delta } ^ { \prime } | \ w ^ { \prime } \in \mathsf { K } _ { \mathrm { p e r } } ^ { \prime } \}$

$$
\operatorname* { l i m } _ { J \to \infty } \operatorname* { s u p } _ { w ^ { \prime } \in \mathsf { K } _ { \mathrm { p e r } } ^ { \prime } } \Big \| w _ { \delta } ^ { \prime } - \sum _ { j = 1 } ^ { J } \langle w _ { \delta } ^ { \prime } , \eta _ { j } \rangle _ { L ^ { 2 } ( B ) } \eta _ { j } \Big \| _ { C _ { \mathrm { p e r } } ^ { s ^ { \prime } } ( B ) } = 0 .
$$

It can then be seen that

$$
\begin{array} { r l } { \displaystyle \underset { J \to \infty } { \operatorname* { l i m } } \underset { w \in \mathbb { K } ^ { \prime } } { \operatorname* { s u p } } \left\| f _ { \delta } ( w ) - \underset { j = 1 } { \overset { J } { \sum } } \langle f _ { \delta } ( w ) , \eta _ { j } \rangle _ { L ^ { 2 } ( B ) } \eta _ { j } \right\| _ { C ^ { s ^ { \prime } } ( \overline { { \Omega } } ) } } & { } \\ { = \underset { J \to \infty } { \operatorname* { l i m } } \underset { w ^ { \prime } \in \mathbb { K } _ { \mathrm { p e r } } ^ { \prime } } { \operatorname* { s u p } } \left\| w _ { \delta } ^ { \prime } - \underset { j = 1 } { \overset { J } { \sum } } \langle w _ { \delta } ^ { \prime } , \eta _ { j } \rangle _ { L ^ { 2 } ( B ) } \eta _ { j } \right\| _ { C ^ { s ^ { \prime } } ( \overline { { \Omega } } ) } } & { } \\ { \le \underset { J \to \infty } { \operatorname* { l i m } } \underset { w ^ { \prime } \in \mathbb { K } _ { \mathrm { p e r } } ^ { \prime } } { \operatorname* { s u p } } \left\| w _ { \delta } ^ { \prime } - \underset { j = 1 } { \overset { J } { \sum } } \langle w _ { \delta } ^ { \prime } , \eta _ { j } \rangle _ { L ^ { 2 } ( B ) } \eta _ { j } \right\| _ { C _ { \mathrm { p e r } } ^ { \prime } ( B ) } = 0 , } \end{array}
$$

satisfying property (S2).

Case 2: In what follows, for any $\gamma > 0$

$$
\Omega ^ { \gamma } : = \{ x \in \Omega : \mathrm { d i s t } ( x , \partial \Omega ) \geq \gamma \} ,\tag{B.1}
$$

and $\mathbf { 1 } _ { \Omega ^ { - } }$ is the indicator function on $\Omega ^ { \gamma }$ . For any $\delta > 0$ , define the smooth cutof mollification of $w \in { \mathsf { K } } ^ { \prime }$ by

$$
f _ { \delta } ( w ) = \rho _ { \delta } * ( \chi _ { 4 \delta } w ) ,
$$

where $\rho _ { \delta }$ is the standard mollifier as in Definition D.3, and $\chi _ { \delta } = \rho _ { \delta / 4 } * \mathbf { 1 } _ { \Omega ^ { 3 \delta / 4 } }$ . Then $\chi _ { \delta }$ satisfies

$$
\chi _ { \delta } ( x ) = \left\{ { \begin{array} { l l } { 1 } & { { \mathrm { ~ i f ~ } } \mathrm { d i s t } ( x , \partial \Omega ) \geq \delta , \ x \in \Omega } \\ { 0 } & { { \mathrm { ~ i f ~ } } \mathrm { d i s t } ( x , \partial \Omega ) \leq { \frac { \delta } { 2 } } , x \in \Omega , } \\ { 0 } & { { \mathrm { ~ i f ~ } } x \not \in \Omega , } \end{array} } \right.
$$

with $0 \leq \chi _ { \delta } \leq 1$ and $\| \nabla \chi _ { \delta } \| _ { \infty } \lesssim \delta ^ { - 1 }$ . Then $\chi _ { 4 \delta } w$ is continuous and equal to $0$ within $2 \delta$ of the boundary ∂Ω. We show first that property (S1) holds.

Recall that since $\mathsf { K } ^ { \prime }$ is compact, it is equicontinuous; i.e. there exists a modulus of continuity $\gamma$ such that for all $w \in { \mathsf { K } } ^ { \prime }$ , it holds that $| w ( x ) - w ( y ) | \leq \gamma ( | x - y | )$ for $x , y \in { \overline { { \Omega } } }$ , and that $\gamma ( t )  0$ as $t \to 0 ^ { + }$ . We bound

$$
\operatorname* { l i m } _ { \delta \to 0 } \operatorname* { s u p } _ { w \in \mathsf { K } ^ { \prime } } \| f _ { \delta } ( w ) - w \| _ { C ( \overline { { \Omega } } ) } \leq \operatorname* { l i m } _ { \delta \to 0 } \operatorname* { s u p } _ { w \in \mathsf { K } ^ { \prime } } \Big ( \underbrace { \| f _ { \delta } ( w ) - \chi _ { 4 \delta } w \| _ { C ( \overline { { \Omega } } ) } } _ { \mathrm { ( I ) } } + \underbrace { \| \chi _ { 4 \delta } w - w \| _ { C ( \overline { { \Omega } } ) } } _ { \mathrm { ( I I ) } } \Big ) .
$$

We bound components (I) and (II) separately. For any $v \in \{ \chi _ { 4 \delta } w : w \in \mathsf { K } ^ { \prime } \}$ and $x , y \in { \overline { { \Omega } } }$ , we have

$$
\begin{array} { l } { | v ( x ) - v ( y ) | \leq \displaystyle \operatorname* { s u p } _ { w \in { \mathsf { K } } ^ { \prime } } | ( \chi _ { 4 \delta } w ) ( x ) - ( \chi _ { 4 \delta } w ) ( y ) | } \\ { \leq \displaystyle \operatorname* { s u p } _ { w \in { \mathsf { K } } ^ { \prime } } | \chi _ { 4 \delta } ( x ) | | w ( x ) - w ( y ) | + | w ( y ) | | \chi _ { 4 \delta } ( x ) - \chi _ { 4 \delta } ( y ) | } \\ { \leq \gamma ( | x - y | ) + | w ( y ) | . } \end{array}\tag{B.2}
$$

We are now prepared to bound component (I):

$$
\begin{array} { r l } & { \displaystyle \operatorname* { l i m } _ { \delta \to 0 } \operatorname* { s u p } _ { w \in \mathbb { K } ^ { \star } } \| \rho _ { \delta } \ast ( \chi _ { 4 \delta } w ) - \chi _ { 4 \delta } w \| _ { C ( \overline { { \Omega } } ) } } \\ & { \qquad = \displaystyle \operatorname* { l i m } _ { \delta \to 0 } \operatorname* { s u p } _ { w \in \mathbb { K } ^ { \star } } \Big | \int _ { B _ { \delta } ( 0 ) } \rho _ { \delta } ( y ) ( ( \chi _ { 4 \delta } w ) ( x - y ) - ( \chi _ { 4 \delta } w ) ( x ) ) \mathrm { d } y \Big | } \\ & { \qquad \le \displaystyle \operatorname* { l i m } _ { \delta \to 0 } \operatorname* { s u p } _ { w \in \mathbb { K } ^ { \star } } \operatorname* { s u p } _ { w \in \mathbb { K } } \| ( \chi _ { 4 \delta } w ) ( x - y ) - ( \chi _ { 4 \delta } w ) ( x ) | } \\ & { \qquad = \displaystyle \operatorname* { l i m } _ { \delta \to 0 } \operatorname* { s u p } _ { w \in \mathbb { K } ^ { \star } } \operatorname* { s u p } _ { x \in \mathbb { K } } \Big \{ \operatorname* { s u p } _ { \delta } | w ( x - y ) - w ( x ) | , \operatorname* { s u p } _ { x \in \overline { { \Omega } } \cap \Sigma ^ { \delta } } | ( \chi _ { 4 \delta } w ) ( x - y ) - ( \chi _ { 4 \delta } w ) ( x ) | \Big \} } \\ & { \qquad = \displaystyle \operatorname* { l i m } _ { \delta \to 0 } \operatorname* { m a x } \Big \{ \gamma ( \delta ) , \gamma ( \delta ) } \\ & { \qquad \quad = \displaystyle \operatorname* { l i m } _ { \delta \to 0 } \operatorname* { s u p } _ { w \in \mathbb { K } ^ { \star } } \Big | \operatorname* { s u p } _ { x \in \mathbb { K } ^ { \star } } \operatorname* { s u p } _ { x \in \mathbb { K } ^ { \star } } | w ( x ) - \operatorname* { s u p } _ { \delta \in \overline { { \Omega } } \cap \Sigma ^ { \delta } } | \big | \leq } \\ &  \qquad \quad \quad \times \displaystyle \operatorname* { l i m } _  \delta \to  \end{array}
$$

by (B.2). Then,

$$
\operatorname* { l i m } _ { \delta \to 0 } \operatorname* { s u p } _ { w \in \mathsf { K } ^ { \prime } } \| \rho _ { \delta } \ast ( \chi _ { 4 \delta } w ) - \chi _ { 4 \delta } w \| _ { C ( \overline { { \Omega } } ) } \leq \operatorname* { l i m } _ { \delta \to 0 } \gamma ( \delta ) + \gamma ( 5 \delta ) = 0 ,
$$

since $w = 0 \mathrm { ~ o n ~ } \partial \Omega .$ and any $x \in \overline { { \Omega } } \setminus \Omega ^ { 5 \delta }$ is at a distance at most 5δ from the boundary. Bounding component (II) is simpler, as we see that

$$
\begin{array} { r l } & { \displaystyle \operatorname* { l i m } _ { \delta \to 0 } \displaystyle \operatorname* { s u p } _ { w \in \mathsf { K } ^ { \prime } } \| \chi _ { 4 \delta } w - w \| _ { C ( \overline { { \Omega } } ) } \leq \displaystyle \operatorname* { l i m } _ { \delta \to 0 } \displaystyle \operatorname* { s u p } _ { w \in \mathsf { K } ^ { \prime } } \displaystyle \operatorname* { s u p } _ { x \in \overline { { \Omega } } \setminus \Omega ^ { 4 \delta } } | \chi _ { 4 \delta } ( x ) w ( x ) - w ( x ) | } \\ & { \qquad \leq \displaystyle \operatorname* { l i m } _ { \delta \to 0 } \displaystyle \operatorname* { s u p } _ { w \in \mathsf { K } ^ { \prime } } \displaystyle \operatorname* { s u p } _ { x \in \overline { { \Omega } } \setminus \Omega ^ { 4 \delta } } | w ( x ) | } \\ & { \qquad \leq \displaystyle \operatorname* { l i m } _ { \delta \to 0 } \gamma ( | 4 \delta | ) = 0 . } \end{array}
$$

Thus property (S1) holds. It is clear that property (S3) also holds since $\chi _ { \delta }$ is continuous and the eigenfunctions of the Laplacian are $C ^ { \infty } ( \Omega )$ . For property (S2), set $B = \overline { { \Omega } }$ and note that for any fixed $\delta > 0 , f _ { \delta }$ is a bounded, continuous map from $C ( { \overline { { \Omega } } } )$ to $C _ { c } ^ { k } ( \overline { { \Omega } } )$ ) for any positive integer k, where $C _ { c } ^ { k } ( \overline { { \Omega } } )$ denotes the subspace of functions in $C ^ { k } ( { \overline { { \Omega } } } )$ with compact support in Ω and thus are 0 on the boundary. Then the image of K<sup>′</sup> under $f _ { \delta }$ , denoted $\mathsf { K } _ { \delta } ^ { \prime } ,$ , is compact in $C _ { c } ^ { k } ( \overline { { \Omega } } )$ . Applying Lemma E.1, we obtain convergence of the projection of $\mathsf { K } _ { \delta } ^ { \prime }$ onto the eigenfunctions of the Laplacian in $C ( { \overline { { \Omega } } } )$ , which achieves property (S2).

Case 3: Let $\{ P _ { n } \} _ { n = 0 } ^ { \infty }$ denote the Legendre polynomials, $P _ { n }$ of degree n, orthonormalized so that $\begin{array} { r } { \left( n + \frac { 1 } { 2 } \right) \int _ { - 1 } ^ { 1 } P _ { m } ( x ) P _ { n } ( x ) \mathrm { ~ d } x = \delta _ { m n } } \end{array}$ , the standard Kronecker delta [40, Equation 1.3.19]. We tensorize the Legendre polynomials in d dimensions in the following manner: for $\nu \in \mathbb { N } _ { 0 } ^ { d } .$ , let $\begin{array} { r } { \Phi _ { \nu } ( x ) = \prod _ { i = 1 } ^ { d } P _ { \nu _ { i } } ( x _ { i } ) } \end{array}$ . The set $\{ \Phi _ { \nu } \} _ { \nu \in  { \mathbb { N } } _ { 0 } ^ { d } }$ is then an orthonormal basis of $L ^ { 2 } ( \Omega )$ . Let $B ^ { \prime }$ be a bounded hypercube containing Ω in its interior. There exists a continuous, linear operator $\mathcal { E } : C ^ { s ^ { \prime } } ( \Omega )  C ^ { s ^ { \prime } } ( B ^ { \prime } )$ such that $\mathcal { E } ( u ) | _ { \Omega } = u$ [27, Lemma A.1]. Let $\rho _ { \delta }$ be the standard mollifier in Definition D.3, and define $f _ { \delta } ( w ) = ( \rho _ { \delta } * { \mathcal E } ( w ) ) | _ { \Omega }$ for $w \in C ^ { s ^ { \prime } } ( \Omega )$ . We assume $\delta < \mathrm { d i s t } ( \Omega , \partial B ^ { \prime } )$ so that the mollifier is defined on Ω. To prove property (S1), observe that for $| \alpha | \le s ^ { \prime } , D ^ { \alpha } f _ { \delta } ( w ) - D ^ { \alpha } w = \rho _ { \delta } * ( D ^ { \alpha } { \mathcal { E } } ( w ) ) - D ^ { \alpha } { \mathcal { E } } ( w ) )$ on Ω. Since $\mathsf { K } ^ { \prime }$ is compact in $C ^ { s ^ { \prime } } ( \Omega )$ and E is continuous, the set $\{ D ^ { \alpha } { \mathcal { E } } ( w ) : ~ w \in \mathsf { K ^ { \prime } } \}$ is compact for any choice of $| \alpha | \le s ^ { \prime }$ and thus is uniformly equicontinuous; denote by $\omega _ { \alpha }$ the modulus of continuity of this set. Since the set of α such that $| \alpha | \le s ^ { \prime }$ is finite, it follows that

$$
\begin{array} { r l } & { \underset { \delta \to 0 } { \operatorname* { l i m } } \underset { \alpha : | \alpha | \leq s ^ { \prime } } { \operatorname* { m a x } } \underset { w \in \mathsf { K } ^ { \prime } } { \operatorname* { s u p } } \Vert D ^ { \alpha } f _ { \delta } ( w ) - D ^ { \alpha } w \Vert _ { L ^ { \infty } ( \Omega ) } } \\ & { \quad \quad = \underset { \delta \to 0 } { \operatorname* { l i m } } \underset { \alpha : | \alpha | \leq s ^ { \prime } } { \operatorname* { m a x } } \underset { w \in \mathsf { K } ^ { \prime } } { \operatorname* { s u p } } \Vert \rho _ { \delta } \ast ( D ^ { \alpha } \mathcal { E } ( w ) ) - D ^ { \alpha } \mathcal { E } ( w ) \Vert _ { L ^ { \infty } ( \Omega ) } } \\ & { \quad \quad \leq \underset { \delta \to 0 } { \operatorname* { l i m } } \underset { \alpha : | \alpha | \leq s ^ { \prime } } { \operatorname* { m a x } } \omega _ { \alpha } ( \delta ) = 0 , } \end{array}
$$

so property (S1) holds. Property (S3) holds as well since $\rho _ { \delta } * { \mathcal { E } }$ is a bounded linear function from $C ^ { s ^ { \prime } } ( \Omega )$ to $L ^ { 2 } ( \Omega )$ , and each $\Phi _ { \nu }$ satisfies $\| \Phi _ { \nu } \| _ { L ^ { 2 } } = 1$ so $w \mapsto$ $\langle f _ { \delta } ( w ) , \Phi _ { \nu } \rangle$ is a bounded linear functional and thus a continuous functional.

We proceed with proving property (S2):

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { w \in \mathbb { K } ^ { * } } \left\| f _ { \delta } ( w ) - \displaystyle \sum _ { \| \nu \| _ { \infty } \leq J } \langle f _ { \delta } ( w ) , \Phi _ { \nu } \rangle \Phi _ { \nu } \right\| _ { C ^ { s ^ { \prime } } ( \Omega ) } } & { } \\ { = \displaystyle \operatorname* { s u p } _ { w \in \mathbb { K } ^ { * } } \| \displaystyle \sum _ { \| \nu \| _ { \infty } > J } \langle f _ { \delta } ( w ) , \Phi _ { \nu } \rangle \Phi _ { \nu } \| _ { C ^ { s ^ { \prime } } ( \Omega ) } } & { } \\ { \leq \displaystyle \operatorname* { s u p } _ { w \in \mathbb { K } ^ { * } } \displaystyle \sum _ { \| \nu \| _ { \infty } > J } | \langle f _ { \delta } ( w ) , \Phi _ { \nu } \rangle | \displaystyle \| \displaystyle \prod _ { i = 1 } ^ { d } P _ { \nu _ { i } } \big \| _ { C ^ { s ^ { \prime } } ( \Omega ) } } & { } \\ { \leq \displaystyle \operatorname* { s u p } _ { w \in \mathbb { K } ^ { * } } \displaystyle \sum _ { \| \nu \| _ { \infty } > J } | \langle f _ { \delta } ( w ) , \Phi _ { \nu } \rangle | \displaystyle \prod _ { i = 1 } ^ { d } \| } & { } \\ { \leq \displaystyle \operatorname* { s u p } _ { w \in \mathbb { K } ^ { * } } \displaystyle \sum _ { \| \nu \| _ { \infty } > J } | \langle f _ { \delta } ( w ) , \Phi _ { \nu } \rangle | \displaystyle \prod _ { i = 1 , \nu _ { i } \neq j } ^ { d } \| P _ { \nu _ { i } } \| _ { C ^ { s ^ { \prime } } ( [ - 1 , 1 ] ) } , } \end{array}
$$

where in the last line we have used that $\begin{array} { r } { D ^ { \alpha } \Phi _ { \nu } = \prod _ { i = 1 } ^ { d } P _ { \nu _ { i } } ^ { ( \alpha _ { i } ) } ( x _ { i } ) } \end{array}$ together with the fact that $\begin{array} { r } { \| P _ { 0 } \| _ { C ^ { s ^ { \prime } } ( [ - 1 , 1 ] ) } = \frac { 1 } { \sqrt { 2 } } < 1 } \end{array}$ , which allows the factors with $\nu _ { i } = 0$ to be discarded. Due to our chosen rescaling, $\| P _ { n } \| _ { L ^ { \infty } ( [ - 1 , 1 ] ) } \lesssim n ^ { 1 / 2 }$ , it follows from the Markov brothers’ inequality that $\| P _ { \nu _ { i } } \| _ { C ^ { s ^ { \prime } } ( [ - 1 , 1 ] ) } \leq C ( \nu _ { i } ) ^ { 2 s ^ { \prime } + \frac { 1 } { 2 } }$ for some constant C independent of $s ^ { \prime } { \mathrm { ; } }$ without loss of generality, we assume $C \geq 1$ . We are then left with

(B.3a)

$$
\begin{array} { r l r } {  { \operatorname* { s u p } _ { w \in \mathsf { K } ^ { \prime } } \big \| f _ { \delta } ( w ) - \displaystyle \sum _ { \| \nu \| _ { \infty } \leq J } \langle f _ { \delta } ( w ) , \Phi _ { \nu } \rangle \Phi _ { \nu } \big \| _ { C ^ { s ^ { \prime } } ( \Omega ) } } } \\ & { } & { \qquad \leq C _ { \quad \underbrace { w \in \mathsf { K } ^ { d } } _ { w \in \mathsf { K } ^ { \prime } } } \displaystyle \sum _ { \| \nu \| _ { \infty } > J } \big \vert \langle f _ { \delta } ( w ) , \Phi _ { \nu } \rangle \big \vert \displaystyle \prod _ { i = 1 , \nu _ { i } \neq 0 } ^ { d } ( \nu _ { i } ) ^ { 2 s ^ { \prime } + \frac { 1 } { 2 } } . } \end{array}\tag{B.3b}
$$

We introduce the Legendre operator

$$
\mathcal { L } _ { i } = - \partial _ { x _ { i } } \big ( ( 1 - x _ { i } ^ { 2 } ) \partial _ { x _ { i } } \big ) ,
$$

which is self-adjoint on $L ^ { 2 } ( \Omega )$ with zero boundary terms since $( 1 - x _ { i } ^ { 2 } )$ vanishes at $x _ { i } ~ = ~ \pm 1$ , and it holds that for $P _ { n }$ a polynomial in $x _ { i } , \mathcal { L } _ { i } P _ { n } = n ( n + 1 ) P _ { n }$ [41, Section 1.3]. Also note that the $\mathcal { L } _ { i }$ commute. This allows us to write $\Phi _ { \nu } =$ $\begin{array} { r } { \left( \prod _ { i = 1 , \nu _ { i } \neq 0 } ^ { d } \frac { \mathcal { L } _ { i } ^ { k } } { ( \nu _ { i } ( \nu _ { i } + 1 ) ) ^ { k } } \right) \Phi _ { \nu } } \end{array}$ . We may then bound quantity (I) by

$$
\begin{array} { l } { \displaystyle \underset { w \in \mathbb { K } ^ { \prime } } { \operatorname* { s u p } } \displaystyle \sum _ { \| \nu \| _ { \infty } > J } \vert \langle f _ { \delta } ( w ) , \Phi _ { \nu } \rangle \vert \displaystyle \prod _ { i = 1 , \nu _ { i } \neq 0 } ^ { d } ( \nu _ { i } ) ^ { 2 s ^ { \prime } + \frac { 1 } { 2 } } } \\ { \displaystyle \quad \le \underset { w \in \mathbb { K } ^ { \prime } } { \operatorname* { s u p } } \displaystyle \sum _ { \| \nu \| _ { \infty } > J } \vert \langle \big ( \displaystyle \prod _ { i = 1 , \nu _ { i } \neq 0 } ^ { d } \mathcal { L } _ { i } ^ { k } \big ) f _ { \delta } ( w ) , \Phi _ { \nu } \rangle \vert \displaystyle \prod _ { i = 1 , \nu _ { i } \neq 0 } ^ { d } ( \nu _ { i } ) ^ { 2 s ^ { \prime } + \frac { 1 } { 2 } - 2 k } } \\ { \displaystyle \quad \le \underset { w \in \mathbb { K } ^ { \prime } } { \operatorname* { s u p } } \displaystyle \sum _ { \| \nu \| _ { \infty } > J } \vert \big ( \displaystyle \prod _ { i = 1 , \nu _ { i } \neq 0 } ^ { d } \mathcal { L } _ { i } ^ { k } \big ) f _ { \delta } ( w ) \vert \displaystyle \prod _ { i = 1 , \nu _ { i } \neq 0 } ^ { d } ( \nu _ { i } ) ^ { 2 s ^ { \prime } + \frac { 1 } { 2 } - 2 k } . } \end{array}\tag{B.4}
$$

In the above, we have used the fact that $( \nu _ { i } ( \nu _ { i } + 1 ) ) \geq \nu _ { i } ^ { 2 }$ and that $\| \Phi _ { \nu } \| = 1$ Since $\mathcal { L } _ { i } f _ { \delta } ( w ) = - 2 x _ { i } \partial _ { x _ { i } } f _ { \delta } ( w ) + ( 1 - x _ { i } ^ { 2 } ) \partial _ { x _ { i } } ^ { 2 } f _ { \delta } ( w )$ and $x _ { i } \in [ - 1 , 1 ]$ , the operator $\textstyle \prod _ { i = 1 , \nu _ { i } \neq 0 } ^ { d } { \mathcal { L } } _ { i } ^ { k }$ is a diferential operator of order at most 2kd with coeficients bounded

on $\Omega ,$ so we can bound independently of $\nu ,$

$$
\| \left( \prod _ { i = 1 , \nu _ { i } \neq 0 } ^ { d } \mathcal { L } _ { i } ^ { k } \right) f _ { \delta } ( w ) \| \le C _ { k , d } \| f _ { \delta } ( w ) \| _ { C ^ { 2 k d } ( \Omega ) } \le C _ { k , d } C _ { k , d , \delta } \operatorname* { s u p } _ { w \in \mathbb { K } ^ { \prime } } \| w \| _ { C ^ { s ^ { \prime } } ( \Omega ) } ,
$$

which is bounded in the final inequality for each fixed δ since $\mathsf { K } ^ { \prime }$ is compact in $C ^ { s ^ { \prime } } ( \Omega )$ . Bound (B.4) then becomes

$$
C _ { k , d } C _ { k , d , \delta } \operatorname* { s u p } _ { w \in \mathsf { K } ^ { \prime } } \| w \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \sum _ { \underbrace { \| \nu \| _ { \infty } > J i = 1 , \nu _ { i } \neq 0 } _ { \mathrm { ( I I ) } } } \prod _ { \mathrm { ( I I ) } } ^ { d } ( \nu _ { i } ) ^ { 2 s ^ { \prime } + \frac { 1 } { 2 } - 2 k } .\tag{B.5}
$$

Let $b _ { \nu _ { i } } = ( \nu _ { i } ) ^ { 2 s ^ { \prime } + { \frac { 1 } { 2 } } - 2 k }$ if $\nu _ { i } > 0$ , and $b _ { 0 } = 1$ otherwise. Let $\begin{array} { r } { S = \sum _ { n = 1 } ^ { \infty } n ^ { 2 s ^ { \prime } + \frac { 1 } { 2 } - 2 k } } \end{array}$ and $\begin{array} { r } { S _ { J } = \sum _ { n = 1 } ^ { J } n ^ { 2 s ^ { \prime } + \frac { 1 } { 2 } - 2 k } } \end{array}$ . Assume integer $\begin{array} { r } { k > s ^ { \prime } + \frac { 3 } { 4 } } \end{array}$ so that S converges. Finally, note that the following sum-product factors as $\begin{array} { r } { \sum _ { \nu \in \mathbb { N } _ { 0 } ^ { d } } \prod _ { i = 1 } ^ { d } b _ { \nu _ { i } } = ( 1 + S ) ^ { d } } \end{array}$ . As a result, we may rewrite (II) and bound as

$$
\begin{array} { r l } { \displaystyle \sum _ { | \nu | | _ { \infty } > J } \prod _ { i = 1 , \nu _ { i } \ne 0 } ^ { d } ( \nu _ { i } ) ^ { 2 s ^ { \prime } + \frac { 1 } { 2 } - 2 k } = ( 1 + S ) ^ { d } - ( 1 + S _ { J } ) ^ { d } } & { } \\ { \displaystyle } & { = d \xi ^ { d - 1 } ( S - S _ { J } ) } \\ { \displaystyle } & { = d \xi ^ { d - 1 } \sum _ { n > J } n ^ { 2 s ^ { \prime } + \frac { 1 } { 2 } - 2 k } } \\ { \displaystyle } & { \le C _ { d , k , s ^ { \prime } } J ^ { \frac { 3 } { 2 } + 2 s ^ { \prime } - 2 k } , } \end{array}
$$

where we have used the mean value theorem in the second equality for some $\xi \in \mathbf { \Xi }$ $( 1 + S _ { J } , 1 + S )$ . Combining the above bound with (B.3), (B.4), and (B.5), we find that

$$
\begin{array} { l } { \displaystyle \operatorname* { s u p } _ { w \in { \mathsf { K } } ^ { \prime } } \left\| f _ { \delta } ( w ) - \displaystyle \sum _ { \| \nu \| _ { \infty } \leq J } \langle f _ { \delta } ( w ) , \Phi _ { \nu } \rangle \Phi _ { \nu } \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \right. } \\ { \displaystyle \qquad \leq C _ { k , d , \delta , s ^ { \prime } } \displaystyle \operatorname* { s u p } _ { w \in { \mathsf { K } } ^ { \prime } } \| w \| _ { C ^ { s ^ { \prime } } ( \Omega ) } J ^ { \frac { 3 } { 2 } + 2 s ^ { \prime } - 2 k } . } \end{array}
$$

Since δ is fixed and $\begin{array} { r } { k > s ^ { \prime } + \frac { 3 } { 4 } } \end{array}$ , the bound tends to 0 as $J \to \infty$ , proving (S2). □

## Appendix C. Proof of Lemma 3.8

Before proceeding with the proof, we repeat the lemma statement for ease of readability.

Lemma 3.8. Let Ω be a bounded Lipschitz domain in $\mathbb { R } ^ { d }$ , and let $s , s ^ { \prime }$ be nonnegative integers. Let $\Psi ^ { \dag } : C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { k } )  C ^ { s ^ { \prime } } ( \overline { { \Omega } } ; \mathbb { R } ^ { k ^ { \prime } } )$ be a continuous operator, and let $\{ \eta _ { j } \} _ { j = 1 } ^ { \infty } \subset C ^ { s ^ { \prime } } ( \Omega )$ . Let $\mathsf { K } \subset C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { k } )$ be a compact subset. Assume that $\mathsf { K } ^ { \prime } : =$ $\Psi ^ { \dagger } ( { \sf K } )$ satisfies the Smoothed Approximation Assumption $3 . 1$ in $C ^ { s ^ { \prime } } ( \Omega )$ with respect to $\{ \eta _ { j } \} _ { j = 1 } ^ { \infty }$ . Then for any $\epsilon > 0$ , there exists $J \in \mathbb { N } .$ , and continuous functionals $\alpha _ { 1 } , \ldots , \alpha _ { J } : L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to \mathbb { R }$ such that the operator $\widetilde { \Psi } ^ { \dagger } : L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to C ^ { s ^ { \prime } } ( \Omega ; \mathbb { R } ^ { k ^ { \prime } } )$ defined by

$$
\widetilde { \Psi } ^ { \dagger } ( u ) = \sum _ { j = 1 } ^ { J } \alpha _ { j } ( u ) \eta _ { j }
$$

satisfies

$$
\operatorname* { s u p } _ { u \in \mathsf { K } } \| \Psi ^ { \dagger } ( u ) - \widetilde \Psi ^ { \dagger } ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \leq \epsilon .
$$

Proof of Lemma 3.8. Step 1: Approximation by $\{ \eta _ { j } \} _ { j = 1 } ^ { \infty }$ . Without loss of generality, we may assume $k ^ { \prime } = 1$ , as we can always approximate $C ^ { s ^ { \prime } } ( \overline { { \Omega } } ; \mathbb { R } ^ { k ^ { \prime } } ) \ =$ $[ C ^ { s ^ { \prime } } ( \overline { { \Omega } } ; \mathbb { R } ) ] ^ { k ^ { \prime } }$ component-wise. Thus, denote by $C ^ { s ^ { \prime } } ( \overline { { \Omega } } )$ the space $C ^ { s ^ { \prime } } ( { \overline { { \Omega } } } ; \mathbb { R } )$

Let $\{ f _ { \delta } \} _ { \delta > 0 }$ be the family of maps in Assumptions 3.1, and B the domain in Assumptions 3.1 item (S2). Under these assumptions, we have that

$$
\operatorname* { l i m } _ { J \to \infty } \operatorname* { s u p } _ { w \in \mathsf { K } ^ { \prime } } \Big \| f _ { \delta } ( w ) - \sum _ { j = 1 } ^ { J } \langle f _ { \delta } ( w ) , \eta _ { j } \rangle _ { L ^ { 2 } ( B ) } \eta _ { j } \Big \| _ { C ^ { s ^ { \prime } } ( \Omega ) } = 0 ,
$$

and lim $\begin{array} { r } { \mathbb { 1 } _ { \delta \to 0 } \operatorname* { s u p } _ { w \in \mathsf { K } ^ { \prime } } \| f _ { \delta } ( w ) - w \| _ { C ^ { s ^ { \prime } } ( \Omega ) } = 0 } \end{array}$ . In particular, given $\epsilon > 0$ , we can first find $\begin{array} { r } { \delta = \delta ( \epsilon ) > 0 \ \mathrm { s u c h \ t h a t \ s u p } _ { w \in \mathsf { K } ^ { \prime } } \| f _ { \delta } ( w ) - w \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \leq \epsilon / 2 } \end{array}$ , and then $J \in \mathbb { N }$ such that $\begin{array} { r } { \operatorname* { s u p } _ { w \in \mathsf { K } ^ { \prime } } \| f _ { \delta } ( w ) - \sum _ { j = 1 } ^ { J } \langle f _ { \delta } ( w ) , \eta _ { j } \rangle _ { L ^ { 2 } ( B ) } \eta _ { j } \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \le \epsilon / 2 } \end{array}$ . It follows from the triangle inequality that

$$
\operatorname* { s u p } _ { w \in \mathsf { K } ^ { \prime } } \Big \| w - \sum _ { j = 1 } ^ { J } \langle f _ { \delta } ( w ) , \eta _ { j } \rangle _ { L ^ { 2 } ( B ) } \eta _ { j } \Big \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \leq \epsilon .\tag{C.1}
$$

As a consequence, it is clear that

$$
\begin{array} { l } { \displaystyle \underset { u \in { \mathsf K } } { \operatorname* { s u p } } \left\| \Psi ^ { \dagger } ( u ) - \displaystyle \sum _ { j = 1 } ^ { J } \langle f _ { \delta } ( \Psi ^ { \dagger } ( u ) ) , \eta _ { j } \rangle _ { L ^ { 2 } ( B ) } \eta _ { j } \right\| _ { C ^ { s ^ { \prime } } ( \Omega ) } } \\ { = \displaystyle \operatorname* { s u p } _ { w \in { \mathsf K } ^ { \prime } } \left\| w - \displaystyle \sum _ { j = 1 } ^ { J } \langle f _ { \delta } ( w ) , \eta _ { j } \rangle _ { L ^ { 2 } ( B ) } \eta _ { j } \right\| _ { C ^ { s ^ { \prime } } ( \Omega ) } \le \epsilon . } \end{array}
$$

Defining $\beta _ { j } ( u ) = \langle f _ { \delta } ( \Psi ^ { \dagger } ( u ) ) , \eta _ { j } \rangle _ { L ^ { 2 } ( B ) }$ we have that $\begin{array} { r } { \widetilde { \Psi } ^ { \dagger } ( u ) = \sum _ { j = 1 } ^ { J } \beta _ { j } ( u ) \eta _ { j } } \end{array}$ satisfies

$$
\operatorname* { s u p } _ { u \in \mathsf { K } } \| \Psi ^ { \dagger } ( u ) - \widetilde \Psi ^ { \dagger } ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \leq \epsilon .\tag{C.2}
$$

Step 2: Construction $o f \alpha _ { j }$ . The above inequality is almost the desired result, but $\beta _ { j }$ does not define a continuous functional $L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to \mathbb { R }$ . The remainder of the proof is very similar to that of the proof of [27, Proposition $\mathrm { A . 6 } ] ;$ we include it here for convenience. By [27, Lemma A.2], there exists a continuous operator $\mathcal { M } _ { \delta } : L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { k } )$ such that over the compact set $\mathsf { K } \subset C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { k } )$ , we have

$$
\operatorname* { s u p } _ { u \in \mathsf { K } } \| u - \mathcal { M } _ { \delta } u \| _ { C ^ { s } ( \overline { { \Omega } } ) } \to 0
$$

as $\delta  0$ Our goal is to define $\alpha _ { j } : L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \ \to$ R by $\alpha _ { j } ( u ) = \beta _ { j } ( \mathcal { M } _ { \delta } u )$ for suitably chosen $\delta > 0$ . Recall that, for fixed $\delta _ { 0 } > 0$ , the set $\mathsf { K } _ { \delta }$ defined by $\mathsf { K } _ { \delta } =$ $\cup _ { 0 \leq \delta \leq \delta _ { 0 } } { \mathcal { M } } _ { \delta } ( \mathsf { K } )$ is a compact subset of $C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { k } )$ by [27, Lemma $\mathrm { A . 4 } ]$ . Note that for any $j = 1 , \dots , J$ , by Assumptions 3.1 property (S3),together with the continuity of $\Psi ^ { \dagger }$ , we have a continuous mapping $\beta _ { j } : \mathsf { K } _ { \delta } \to \mathbb { I }$ R. Since $\mathsf { K } _ { \delta }$ is compact, it follows that there exists a modulus of continuity $\omega : [ 0 , \infty ) \to [ 0 , \infty )$ , continuous in its argument and satisfying $\omega ( 0 ) = 0$ , such that

$$
| \beta _ { j } ( u ) - \beta _ { j } ( u ^ { \prime } ) | \leq \omega ( \Vert u - u ^ { \prime } \Vert _ { C ^ { s } ( \overline { { \Omega } } ) } ) , \quad \forall u , u ^ { \prime } \in \mathsf { K } _ { \delta } ,
$$

holds for all $j = 1 , \dots , J .$ . It follows that

$$
| \beta _ { j } ( u ) - \beta _ { j } ( \mathcal { M } _ { \delta } u ) | \leq \omega ( \| u - \mathcal { M } _ { \delta } u ) \| _ { C ^ { s } ( \overline { { \Omega } } ) } ) ,
$$

for any $0 < \delta \leq \delta _ { 0 } . ~ \mathrm { B y } [ 2 7 $ , Lemma $\mathrm { A . 2 } ] , \mathrm { ~ } \mathcal { M } _ { \delta } u \mathrm { ~ \to ~ } u$ converges uniformly over $u \in { \mathsf { K } } .$ so that we may conclude

$$
\operatorname* { l i m } _ { \delta \to 0 } \operatorname* { s u p } _ { u \in \mathsf { K } } | \beta _ { j } ( u ) - \beta _ { j } ( \mathcal { M } _ { \delta } u ) | \leq \operatorname* { l i m } _ { \delta \to 0 } \omega ( \operatorname* { s u p } _ { u \in \mathsf { K } } \| u - \mathcal { M } _ { \delta } u \| _ { C ^ { s } ( \overline { \Omega } ) } ) = 0 .
$$

In particular, we can choose $\delta > 0$ suficiently small, to ensure that $\alpha _ { j } ( u ) : =$ $\beta _ { j } ( \mathcal { M } _ { \delta } u )$ satisfies

$$
\operatorname* { s u p } _ { u \in \mathsf { K } } | \beta _ { j } ( u ) - \alpha _ { j } ( u ) | \leq \frac { \epsilon } { J \operatorname* { m a x } _ { j = 1 , \ldots , J } \| \eta _ { j } \| _ { C ^ { s ^ { \prime } } ( \Omega ) } }\tag{C.3}
$$

for all $j = 1 , \dots , J ,$ . Since $\delta > 0$ , we also note that $\mathcal { M } _ { \delta } : L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to C ^ { s } ( \overline { { \Omega } } ; \mathbb { R } ^ { k } )$ is a continuous mapping, and hence $\alpha _ { j } = \beta _ { j } \circ { \mathcal { M } } _ { \delta }$ is continuous as a mapping $\alpha _ { i } : L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to \mathbb { R }$

Step 3: (Conclusion). Combining the results of (C.2) and (C.3), we obtain

$$
\begin{array} { l } { \displaystyle \operatorname* { s u p } _ { u \in \mathbb { K } } \left\| \Psi ^ { \dagger } ( u ) - \displaystyle \sum _ { j = 1 } ^ { J } \alpha _ { j } ( u ) \eta _ { j } \right\| _ { C ^ { \prime } ( \Omega ) } } \\ { \displaystyle \quad \leq \operatorname* { s u p } _ { u \in \mathbb { K } } \left\| \Psi ^ { \dagger } ( u ) - \displaystyle \sum _ { j = 1 } ^ { J } \beta _ { j } ( u ) \eta _ { j } \right\| _ { C ^ { \prime \prime } ( \Omega ) } + \displaystyle \operatorname* { s u p } _ { u \in \mathbb { K } } \left\| \displaystyle \sum _ { j = 1 } ^ { J } ( \beta _ { j } ( u ) - \alpha _ { j } ( u ) ) \eta _ { j } \right\| _ { C ^ { \prime \prime } ( \Omega ) } } \\ { \displaystyle \qquad \leq \operatorname* { s u p } _ { u \in \mathbb { K } } \left\| \Psi ^ { \dagger } ( u ) - \displaystyle \sum _ { j = 1 } ^ { J } \beta _ { j } ( u ) \eta _ { j } \right\| _ { C ^ { \prime \prime } ( \Omega ) } } \\ { \displaystyle \quad + \int _ { j = 1 , \dots , J } \left\| \eta _ { j } \right\| _ { C ^ { \prime \prime } ( \Omega ) } \displaystyle \operatorname* { m a x } _ { j = 1 , \dots , J } \operatorname* { s u p } _ { u \in \mathbb { K } } | \beta _ { j } ( u ) - \alpha _ { j } ( u ) | } \\ { \displaystyle \quad \leq 2 \epsilon . } \end{array}
$$

Since $\epsilon > 0$ was arbitrary, this proves the claim.

## Appendix D. Proofs of Lemma 3.10 and 3.11

Before we proceed with the proofs, we will need the following lemma, which allows us to approximate certain integrals by the integral of a finite-dimensional neural network weighted by a positive function.

Lemma D.1. Let Ω be a bounded domain in $\mathbb { R } ^ { d }$ , and let K be compact in $L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } )$ and assume that $\begin{array} { r } { M _ { \mathsf { K } } : = \operatorname* { s u p } _ { u \in \mathsf { K } } \| u \| _ { L ^ { \infty } } < \infty } \end{array}$ . Let $\xi \in C ( \overline { { \Omega } } ; \mathbb { R } ^ { k } ) \cap L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } )$ be given and fixed. Furthermore, let $\psi \in C ( \Omega ; \mathbb { R } _ { \geq 0 } )$ be such that $\mathrm { s u p } _ { y \in \Omega } | \psi ( y ) | < \infty$ and such that ψ is 0 only on a set of Lebesgue measure 0. Then for any $\epsilon > 0$ there exists a neural network $\widetilde { R } : \mathbb { R } ^ { k } \times \overline { { \Omega } }  \mathbb { R }$ such that

$$
\operatorname* { s u p } _ { u \in \mathsf { K } } \left| \int _ { \Omega } \widetilde { R } ( u ( y ) , y ) \psi ( y ) \mathrm { d } y - \int _ { \Omega } u ( y ) \cdot \xi ( y ) \mathrm { d } y \right| < \epsilon .
$$

Proof. Since $\psi \geq 0$ on Ω and equals 0 only on a set of measure $0 ,$ we may assume without loss of generality that $\begin{array} { r } { \int _ { \Omega } \psi ( y ) \ \mathsf { d } y = 1 } \end{array}$ since the final layer of $\widetilde { R }$ may be rescaled to account for the rescaling of ψ. Define $\Omega ^ { \gamma } : = \{ y \in \Omega : \psi ( y ) < \gamma \}$ and fix γ such that $\begin{array} { r } { M _ { \mathsf { K } } \| \boldsymbol { \xi } \| _ { C ( \overline { { \Omega } } ) } | \Omega ^ { \gamma } | < \frac { \epsilon } { 4 } } \end{array}$ . Then define

$$
\psi ^ { \gamma } ( y ) = \mathrm { m a x } ( \psi ( y ) , \gamma ) , \quad y \in \Omega .
$$

It holds that $\psi ^ { \gamma } \in C ( \Omega )$ . Choose a compact $F \subset \Omega$ , contained in the interior of Ω if Ω is closed, such that $\begin{array} { r } { M _ { \mathsf { K } } \| \xi \| _ { C ( \overline { { \Omega } } ) } | \Omega \setminus F | < \frac { \epsilon } { 4 } } \end{array}$ . Thus, $F$ does not touch $\partial \Omega$ , which enables the cutof function $\chi ,$ , that we now go on to define, to have space to deform continuously to 0. Define $\chi \in C ( \overline { { \Omega } } ; [ 0 , 1 ] )$ such that $\chi = 1$ on $F$ and 0 on $\partial \Omega$ . Then set

$$
g ( y ) = \left\{ \begin{array} { l l } { \frac { \chi ( y ) } { \psi ^ { \gamma } ( y ) } , \quad y \in \Omega , } \\ { 0 , \quad y \in \partial \Omega . } \end{array} \right.
$$

Since $\psi ^ { \gamma }$ is positive everywhere on Ω and $\chi$ is 0 on the boundary, $g$ is continuous on ${ \overline { { \Omega } } } ,$ and thus the function $R : ( a , y ) \mapsto ( a \cdot \xi ( y ) ) g ( y )$ is continuous on domain $[ - M _ { \mathsf { K } } , M _ { \mathsf { K } } ] ^ { k } \times \overline { { \Omega } }$ . By universality of ordinary neural networks, i.e. Lemma $\mathrm { { A . 2 } } .$ there exists a network $\widetilde { R }$ such that

$$
\operatorname* { s u p } _ { a \in [ - M _ { \mathsf { K } } , M _ { \mathsf { K } } ] ^ { k } , y \in \overline { { \Omega } } } | R ( a , y ) - \widetilde { R } ( a , y ) | < \frac { \epsilon } { 2 } .
$$

We proceed with the bound.

$$
\begin{array} { r l } & { \underset { u \in \mathbb { K } } { \operatorname* { s u p } } \left| \int _ { \Omega } \widetilde { R } ( u ( y ) , y ) \psi ( y ) \mathrm { d } y - \int _ { \Omega } u ( y ) \cdot \xi ( y ) \mathrm { d } y \right| } \\ & { \le \underset { u \in \mathbb { K } } { \operatorname* { s u p } } \int _ { \Omega } \left| \widetilde { R } ( u ( y ) , y ) - R ( u ( y ) , y ) \right| \psi ( y ) \mathrm { d } y + \left| \int _ { \Omega } R ( u ( y ) , y ) \psi ( y ) - u ( y ) \cdot \xi ( y ) \mathrm { d } y \right| } \\ & { \le \frac { \epsilon } { 2 } + \underset { u \in \mathbb { K } } { \operatorname* { s u p } } \left| \int _ { \Omega } u ( y ) \cdot \xi ( y ) \left( \frac { \chi ( y ) \psi ( y ) } { \psi ^ { \gamma } ( y ) } - 1 \right) \mathrm { d } y \right| , } \end{array}
$$

where we have used the normalization $\begin{array} { r } { \int _ { \Omega } \psi ( y ) \ \mathsf { d } y = 1 } \end{array}$ . Since $\begin{array} { r } { \frac { \psi ( y ) } { \psi ^ { \gamma } ( y ) } \leq 1 } \end{array}$ and $\chi \leq 1$ the expression $\frac { \chi ( y ) \psi ( y ) } { \psi ^ { \gamma } ( y ) } \in [ 0 , 1 ]$ . Additionally, $\begin{array} { r } { \frac { \chi ( y ) \psi ( y ) } { \psi ^ { \gamma } ( y ) } = 1 } \end{array}$ on $F \setminus \Omega ^ { \gamma }$ , so we have $\begin{array} { r } { \left| \frac { \chi ( y ) \psi ( y ) } { \psi ^ { \gamma } ( y ) } - 1 \right| \leq \mathbf { 1 } _ { \Omega \backslash F } + \mathbf { 1 } _ { \Omega ^ { \gamma } } } \end{array}$ . Therefore,

$$
\operatorname* { s u p } _ { u \in \mathsf { K } } \left| \int _ { \Omega } u ( y ) \cdot \xi ( y ) \left( \frac { \chi ( y ) \psi ( y ) } { \psi ^ { \gamma } ( y ) } - 1 \right) \mathrm { ~ d } y \right| \leq M _ { \mathsf { K } } \| \xi \| _ { C ( \overline { { \Omega } } ) } ( | \Omega \setminus F | + | \Omega ^ { \gamma } | ) .
$$

By our choices of $\gamma$ and $F _ { ; }$

$$
\operatorname* { s u p } _ { u \in \mathsf { K } } \left| \int _ { \Omega } \widetilde { R } ( u ( y ) , y ) \psi ( y ) \mathrm { d } y - \int _ { \Omega } u ( y ) \cdot \xi ( y ) \mathrm { d } y \right| < \frac { \epsilon } { 2 } + \frac { \epsilon } { 4 } + \frac { \epsilon } { 4 } = \epsilon .
$$

We also point out the following fact about continuous, nonzero functions.

Lemma D.2. Let Ω be a bounded set in $\mathbb { R } ^ { d }$ . Assume that η is continuous on Ω and not identically 0. Then for any $\epsilon > 0$ , there exists an open ball $\Gamma \subset \Omega$ and constant $\gamma$ such that

$$
\| \gamma \eta - \mathbf { 1 } \| _ { L ^ { \infty } ( \Gamma ) } < \epsilon .
$$

Proof. The result follows directly from the continuity of $\eta .$

Before proceeding with the proof of Lemma 3.11, we define the standard mollifier. This definition is taken from the standard text [16, Appendix C.4].

Definition D.3 (Standard Mollifier). Let Ω be a bounded Lipschitz domain, and define Ω<sup>ϵ</sup> as in (B.1). We say that $w _ { \epsilon } : \Omega ^ { \epsilon } \to \mathbb { R }$ is the standard mollification of $w \in L ^ { 1 } ( \Omega )$ , if

$$
w _ { \epsilon } : = \rho _ { \epsilon } \ast w , \mathrm { i n } \Omega ^ { \epsilon } \mathrm { f o r } \rho _ { \epsilon } ( x ) : = \frac { 1 } { \epsilon ^ { d } } \rho \left( \frac { x } { \epsilon } \right) ,\tag{D.1}
$$

where

$$
\rho ( x ) : = { \left\{ \begin{array} { l l } { C \exp \left( { \frac { 1 } { | x | ^ { 2 } - 1 } } \right) , } & { \quad { \mathrm { i f ~ } } | x | < 1 } \\ { 0 , } & { \quad { \mathrm { i f ~ } } | x | \geq 1 . } \end{array} \right. }
$$

is referred to as the standard mollifier.

♢

We now restate and prove the following two lemmas, stated originally in Section 3.3. Since the proofs begin in the same way, we initiate the proofs together and make a note when they diverge.

Lemma 3.10. Let $\alpha : L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to \mathbb { R }$ be a continuous functional and let $\textsf { K } \subset$ $L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } )$ be a compact set consisting of bounded functions $\begin{array} { r } { \operatorname* { s u p } _ { u \in \mathsf { K } } \| u \| _ { L ^ { \infty } ( \Omega ) } < \infty } \end{array}$ Assume $\Omega \subset \mathbb { R } ^ { d }$ is a bounded Lipschitz domain. Assume, in addition, that $\eta \in$ $C ^ { s ^ { \prime } } ( \overline { { \Omega } } ; \mathbb { R } ^ { k ^ { \prime } } )$ for integer $s ^ { \prime } \geq 0$ . Then for any $\varepsilon > 0$ there exists a nonlocal neural operator α : $L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to C ^ { s ^ { \prime } } ( \Omega ; \mathbb { R } ^ { k ^ { \prime } } )$ as defined in Definition 2.1, with Assumption 2.2 on the activation satisfied with $s _ { \sigma } = s ^ { \prime }$ , such that

$$
\operatorname* { s u p } _ { u \in \mathsf { K } } \| \alpha ( u ) \eta - \widetilde { \alpha } ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } < \varepsilon\tag{3.3}
$$

with a single hidden layer $( M = 1 )$ and with any single basis function pair $\psi _ { 1 , 1 } : = \eta _ { 1 }$ and $\phi _ { 1 , 1 } : = \eta _ { 2 }$ satisfying Assumptions 3.3.

Lemma 3.11. Let $\alpha : L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to \mathbb { R }$ be a continuous functional and let $\textsf { K } \subset$ $L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } )$ be a compact set consisting of bounded functions $\begin{array} { r } { \operatorname* { s u p } _ { u \in \mathsf { K } } \| u \| _ { L ^ { \infty } ( \Omega ) } < \infty . } \end{array}$ Assume $\Omega \subset \mathbb { R } ^ { d }$ is a bounded Lipschitz domain. Assume, in addition, that $\eta \in$ $C ^ { s ^ { \prime } } ( \Omega ; \mathbb { R } ^ { k ^ { \prime } } )$ for integer $s ^ { \prime } \geq 0$ . Then for any $\varepsilon > 0$ there exists a nonlocal neural operator $\widetilde { \alpha } : L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to C ^ { s ^ { \prime } } ( \Omega ; \mathbb { R } ^ { k ^ { \prime } } )$ as defined in Definition $2 . 1 ,$ with Assumption 2.2 on the activation satisfied with $s _ { \sigma } = s ^ { \prime }$ , such that

$$
\operatorname* { s u p } _ { u \in \mathsf { K } } \| \alpha ( u ) \eta - \widetilde { \alpha } ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } < \varepsilon ,\tag{3.4}
$$

where the final hidden layer contains η as an outer basis element, i.e. for some $m ^ { * } \le M , \phi _ { L , m ^ { * } } : = \eta$ with a corresponding inner basis function $\psi _ { L , m ^ { * } } \in L ^ { 1 } ( \Omega )$ satisfying $( E 2 ) \ – ( E 4 )$ in Assumptions 3.3 and an earlier hidden layer contains at least one basis function pair $\eta _ { 1 }$ satisfying $( E 1 ) \ – ( E _ { 4 } )$ and $\eta _ { 2 }$ satisfying $( E 2 ) \ – ( E 4 )$

Proof of Lemmas 3.10 and 3.11. We can identify any $u \in L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } )$ with a function in $\mathring { L } ^ { 1 } ( B ; \mathbb { R } ^ { k } )$ for any set $B \supset \Omega$ via an extension of $u ( x ) = 0$ for $x \in B \setminus \Omega$ Then the compact subset $\mathsf { K } \subset L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } )$ can be identified with a compact subset of $L ^ { 1 } ( B ; \mathbb { R } ^ { k } )$ ). Let $\rho _ { \delta }$ for $\delta > 0$ be the standard mollifier of Definition D.3. We denote by $u _ { \delta } ( x ) = ( u * \rho _ { \delta } ) ( x ) , \quad x \in \Omega$ , the mollification of u extended to B by 0 outside of Ω. Since ${ \sf K } \subset L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } )$ is compact,

$$
\operatorname* { l i m } _ { \delta \to 0 } \operatorname* { s u p } _ { u \in \mathsf { K } } \| u - u _ { \delta } \| _ { L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) } = 0 .
$$

Since $\alpha : L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to \mathbb { R }$ is continuous, it follows that $\alpha _ { \delta } : L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) \to \mathbb { R }$ defined by $\alpha _ { \delta } ( u ) = \alpha ( u _ { \delta } )$ for $\delta > 0$ , converges uniformly over K as $\delta  0 \backslash$

$$
\operatorname* { l i m } _ { \delta \to 0 } \operatorname* { s u p } _ { u \in { \mathsf { K } } } | \alpha ( u ) - \alpha _ { \delta } ( u ) | \leq \operatorname* { l i m } _ { \delta \to 0 } \operatorname* { s u p } _ { u \in { \mathsf { K } } } | \alpha ( u ) - \alpha ( u _ { \delta } ) | = 0 .\tag{D.2}
$$

Let $\{ \xi _ { j } \} _ { j = 1 } ^ { \infty }$ be any orthonormal basis of $L ^ { 2 } ( \Omega ; \mathbb { R } ^ { k } )$ , which we may additionally choose to be smooth; $\xi _ { j } \in C ^ { \infty } ( \Omega ; \mathbb { R } ^ { k } )$ . Since $L ^ { 2 } ( \Omega ) \hookrightarrow L ^ { 1 } ( \Omega )$ and $\{ u _ { \delta } : \ u \in \mathsf { K } \}$ is compact in $L ^ { 2 } ( \Omega )$ , due to being the image of compact K in $\dot { L } ^ { 1 }$ under the continuous map of the mollification, it holds that

$$
\operatorname* { l i m } _ { J \to \infty } \operatorname* { s u p } _ { u \in \mathsf { K } } \| u _ { \delta } - \sum _ { j = 1 } ^ { J } \langle u _ { \delta } , \xi _ { j } \rangle \xi _ { j } \| _ { L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } ) } = 0 .\tag{D.3}
$$

Then define

$$
\alpha _ { \delta , J } ( u ) = \alpha \left( \sum _ { j = 1 } ^ { J } \langle u _ { \delta } , \xi _ { j } \rangle \xi _ { j } \right) .
$$

Furthermore, using evenness of the mollifier, we have the identity

(D.4a)

$$
\begin{array} { r } { \langle u _ { \delta } , \xi _ { j } \rangle = \displaystyle \int _ { \Omega } u _ { \delta } ( y ) \cdot \xi _ { j } ( y ) \mathrm { d } y = \displaystyle \int _ { \mathbb R ^ { d } } ( u * \rho _ { \delta } ) ( y ) \cdot \xi _ { j } ( y ) \mathrm { d } y } \\ { = \displaystyle \int _ { \mathbb R ^ { d } } u ( y ) \cdot ( \xi _ { j } * \rho _ { \delta } ) \mathrm { d } y = \displaystyle \int _ { \Omega } u ( y ) \cdot \xi _ { j , \delta } ( y ) \mathrm { d } y } \end{array}\tag{D.4b}
$$

with the convention that u and $\xi _ { j }$ are expanded by 0 outside of $\Omega ,$ and

$$
\xi _ { j , \delta } ( y ) = ( \xi _ { j } * \rho _ { \delta } ) ( y ) , \quad y \in \overline { { \Omega } } .\tag{D.5}
$$

Thus, we may rewrite

$$
\alpha _ { \delta , J } ( u ) = \alpha \Big ( \sum _ { j = 1 } ^ { J } \langle u , \xi _ { j , \delta } \rangle \xi _ { j } \Big )\tag{D.6}
$$

From (D.2), (D.3), and continuity of $\alpha ,$ , for any $\epsilon _ { \alpha } > 0$ , it holds that for $\delta = \delta ( \mathsf { K } ) > 0$ small enough and $J = J ( \delta , { \sf K } ) \in$ N large enough,

$$
\operatorname* { s u p } _ { u \in \mathsf { K } } | \alpha ( u ) - \alpha _ { \delta , J } ( u ) | \leq \epsilon _ { \alpha } .\tag{D.7}
$$

We will thus seek to approximate $\alpha _ { \delta , J } ( u ) \eta$ by the NNO architecture. Recall that η<sub>1</sub> is the single basis function satisfying properties $\mathrm { ( E 1 ) { - } ( E 4 ) }$ , and thus satisfies the conditions on ψ in Lemma D.1. Additionally, $\xi _ { j , \delta } \ \in \ C ( \overline { { \Omega } } ; \mathbb { R } ^ { k } ) \cap L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } )$ and thus satisfies the conditions on ξ for the same lemma. Parallelizing the networks constructed in Lemma D.1, for any $\epsilon ^ { \prime } > 0$ , there exists a neural network $R ^ { \prime }$ $\mathbb { R } ^ { k } \times \overline { { \Omega } }  \mathbb { R } ^ { J } , ( v , x ) \mapsto R ^ { \prime } ( v , x ) = ( R _ { 1 } ( v , x ) , \ldots , R _ { J } ( v , x ) )$ such that

$$
\operatorname* { s u p } _ { u \in \mathsf { K } } \left| \int _ { \Omega } R _ { j } ( u ( y ) , y ) \eta _ { 1 } ( y ) \mathrm { d } y - \int _ { \Omega } u ( y ) \cdot \xi _ { j , \delta } ( y ) \mathrm { d } y \right| < \epsilon ^ { \prime }\tag{D.8}
$$

for $j = 1 , \dots , J$ . We assume $\epsilon ^ { \prime } < 1 $ . It is at this point that the two proofs diverge. Second half of the proof of Lemma 3.10: Recall that here $\eta _ { 1 }$ satisfies $\mathrm { ( E 1 ) } { - } ( \mathrm { E 4 ) }$ and $\eta _ { 2 } \in C ^ { s ^ { \prime } } ( \overline { { \Omega } } )$ and $\eta _ { 2 } \neq 0$ on $\overline { { \Omega } } .$ . Without loss of generality we may assume that $\eta _ { 2 } > 0$ on Ω. Since $\overline { { \Omega } }$ is compact and $\eta _ { 2 }$ is continuous and positive,

$$
\eta _ { 2 } ^ { - } : = \operatorname* { m i n } _ { x \in \overline { { \Omega } } } \eta _ { 2 } ( x ) > 0 , \quad \eta _ { 2 } ^ { + } : = \operatorname* { m a x } _ { x \in \overline { { \Omega } } } \eta _ { 2 } ( x ) < \infty .
$$

We retain $R ^ { \prime } = ( R _ { 1 } , \ldots , R _ { J } )$ and (D.8) from the first half of the proof, and abbreviate $c _ { j } : = \langle u , \xi _ { j , \delta } \rangle$ and $\begin{array} { r } { \widehat { c } _ { j } : = \int _ { \Omega } R _ { j } ( u ( y ) , y ) \eta _ { 1 } ( y ) } \end{array}$ dy so that (D.8) reads $\begin{array} { r } { \operatorname* { s u p } _ { u \in \mathsf { K } } \operatorname* { m a x } _ { j \in [ J ] } | \widehat { c } _ { j } - c _ { j } | \leq \epsilon ^ { \prime } } \end{array}$ with $\epsilon ^ { \prime } < 1 $ . We have that

$$
\operatorname* { m a x } _ { j \in [ J ] } | c _ { j } | \leq \| u _ { \delta } \| \leq \| u \| \leq \operatorname* { s u p } _ { u \in \mathsf { K } } \| u \| < \infty .
$$

As a result, ma $\begin{array} { r } { \mathbf { x } _ { j \in [ J ] } | \widehat { c } _ { j } | \leq \operatorname* { s u p } _ { u \in \mathsf { K } } \| u \| + 1 = : M _ { 1 } } \end{array}$ uniformly over K. By Lemma A.2, for any $\epsilon _ { \eta } > 0 .$ , there exists a finite-dimensional neural network $R _ { \eta }$ such that

$$
\| R _ { \eta } - \eta \| _ { C ^ { s ^ { \prime } } ( \overline { { \Omega } } ) } < \epsilon _ { \eta } \leq 1 .\tag{D.9}
$$

Define $M _ { e } = \| \eta \| _ { L ^ { \infty } ( \overline { { \Omega } } ) } + 1$ , which serves as a bound on $\| R _ { \eta } \| _ { L ^ { \infty } ( \overline { { \Omega } } ) }$ . Define the lifting layer of the NNO $\mathcal { R } : u ( x ) \mapsto R ( u ( x ) , x )$ by

$$
R ( u ( x ) , x ) = ( R _ { 1 } ( u ( x ) , x ) , \ldots , R _ { J } ( u ( x ) , x ) , 1 , R _ { \eta } ( x ) ) .\tag{D.10}
$$

We draw attention to the fact that the 1 is easily achieved exactly due to the final layer of finite-dimensional neural networks being afine in Definition A.1. Setting the bias term equal to this value of $y$ and the linear weights to 0 for the channel corresponding to 1 achieves the value exactly for every input. We set the single hidden layer of our NNO as

$$
( \mathcal { L } _ { 1 } v ) ( x ) = \sigma \Big ( A _ { 1 } ( W _ { 1 } v ( x ) + T _ { 1 , 0 } \int _ { \Omega } v ( y ) \eta _ { 1 } ( y ) \mathrm { d } y \eta _ { 2 } ( x ) ) + b _ { 1 } \Big ) .\tag{D.11}
$$

We set $T _ { 1 , 0 }$ to be a diagonal matrix, equal to the identity $I _ { J }$ for channels $1 , \ldots , J$ equal to $\frac { 1 } { \Vert \eta _ { 1 } \Vert _ { L ^ { 1 } ( \Omega ) } }$ on channel $J + 1$ and equal to 0 on the final $k ^ { \prime }$ entries (corresponding to the $R _ { \eta }$ block). We set $W _ { 1 }$ to copy the final $k ^ { \prime }$ entries $( R _ { \eta }$ block) and 0 the other entries. The end result of these choices is that before applying $A _ { 1 }$ and $b _ { 1 }$ and the activation, we obtain a vector

$$
z ( x ) : = ( \widehat { c } _ { 1 } \eta _ { 2 } ( x ) , \ldots \widehat { c } _ { J } \eta _ { 2 } ( x ) , \eta _ { 2 } ( x ) , R _ { \eta } ( x ) ) ,
$$

which for all $u \in \mathsf { K } , x \in \overline { { \Omega } }$ , lies in the box

$$
\mathcal { B } : [ - M _ { 1 } \eta _ { 2 } ^ { + } , M _ { 1 } \eta _ { 2 } ^ { + } ] ^ { J } \times [ \eta _ { 2 } ^ { - } , \eta _ { 2 } ^ { + } ] \times [ - M _ { e } , M _ { e } ] ^ { k ^ { \prime } } .
$$

Note that nonnegativity of $\eta _ { 1 }$ is used to achieve $\eta _ { 2 } ( x )$ in the second-to-last channel of z. Define $C _ { z } : = \operatorname* { s u p } _ { u \in \mathsf { K } } \| z \| _ { C ^ { s ^ { \prime } } ( \overline { \Omega } ) }$ . Set

$$
\widetilde Q ( \{ c _ { j } \} _ { j = 1 } ^ { J } ) = \alpha \Big ( \sum _ { j = 1 } ^ { J } c _ { j } \xi _ { j } \Big ) .
$$

$\widetilde { Q }$ is continuous on $[ - M _ { 1 } , M _ { 1 } ] ^ { J }$ , so by Lemma $\mathrm { { A . 2 } } .$ , for any $\epsilon _ { Q }$ , there exists a neural network $Q _ { 0 }$ with

$$
\| Q _ { 0 } - \widetilde Q \| _ { L ^ { \infty } ( [ - M _ { 1 } , M _ { 1 } ] ^ { J } ) } \le \epsilon _ { Q } < 1 .
$$

Define the bound $M _ { Q _ { 0 } } = \| Q _ { 0 } \| _ { L ^ { \infty } ( [ - M _ { 1 } , M _ { 1 } ] ^ { J } ) } \leq \| \widetilde { Q } \| _ { L ^ { \infty } ( [ - M _ { 1 } , M _ { 1 } ] ^ { J } ) } + 1$ . Define the function $G : B \to \mathbb { R } ^ { k ^ { \prime } }$ by

$$
G ( \{ a _ { j } \} _ { j = 1 } ^ { J } , b , e ) = Q _ { 0 } \left( \left\{ \frac { a _ { j } } { b } \right\} _ { j = 1 } ^ { J } \right) e .
$$

Since $b \geq \eta _ { 2 } ^ { - } > 0$ on $B ,$ the map $( a _ { j } , b ) \mapsto a _ { j } / b$ is $C ^ { \infty }$ on that domain, and $\sigma \in C ^ { \infty }$ by assumption, so $Q _ { 0 } \in C ^ { \infty } ( [ - M _ { 1 } , M _ { 1 } ] ^ { J } )$ and $G \in C ^ { \infty } ( B )$ . By Lemma A.2, there exists a finite-dimensional neural network $\mathcal { N }$ such that

$$
\| \mathcal { N } - G \| _ { C ^ { s ^ { \prime } } ( \mathcal { B } ) } < \epsilon _ { \mathcal { N } } .\tag{D.12}
$$

Set $A _ { 1 } , b _ { 1 }$ of $\mathcal { L } _ { 1 }$ defined in (D.11) to be the first layer weights and bias of $\mathcal { N } _ { ; }$ with Q realizing its remaining layers. Then

$$
\begin{array} { r } { \big ( \mathscr { Q } \circ \mathscr { L } _ { 1 } \circ \mathcal { R } \big ) ( u ) ( x ) = \mathcal { N } ( z ( x ) ) = : \widetilde { \alpha } ( u ) ( x ) . } \end{array}
$$

We note that $M _ { 1 } , \eta _ { 2 } ^ { - } , \eta _ { 2 } ^ { + } , M _ { e } , B ,$ and $M _ { Q _ { 0 } }$ , depend only on $\mathsf { K } , \eta , \eta _ { 2 } , \delta , J ,$ and on the fact that we assume the initial bounds of $\epsilon ^ { \prime } , \epsilon _ { \eta } , \epsilon _ { Q }$ as being $\leq 1 ;$ they are thus unafected when these bounds are subsequently decreased below 1.

We now estimate the error bound. Note that z is a $C ^ { s ^ { \prime } } ( \overline { \Omega } ; \mathbb { R } ^ { J + 1 + k ^ { \prime } } )$ map with components $\widehat { c } _ { j } \eta _ { 2 } , \eta _ { 2 }$ , and $R _ { \eta } \in C ^ { s ^ { \prime } }$ . We decompose

$$
\| \alpha ( u ) \eta - \widetilde { \alpha } ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \le \underbrace { \| \mathcal { N } \circ z - G \circ z \| _ { C ^ { s ^ { \prime } } ( \Omega ) } } _ { \mathrm { ( I ) } } + \underbrace { \| G \circ z - \alpha ( u ) \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } } _ { \mathrm { ( I I ) } } .
$$

For (I), by multivariate Fa\`a di Bruno’s formula,

$$
\begin{array} { r } { \mathrm { ( I ) } \le c _ { s ^ { \prime } , d } ( 1 + C _ { z } ) ^ { s ^ { \prime } } \epsilon _ { \mathcal { N } } = : C _ { \mathcal { N } } \epsilon _ { \mathcal { N } } , } \end{array}
$$

where $c _ { s ^ { \prime } , d } > 0$ is a constant depending on $s ^ { \prime }$ and d. For (II), due to the trick of obtaining a channel with $\eta _ { 2 } ( x )$ exactly, the division is exact, and since $\eta _ { 2 } ( x ) \geq$ $\eta _ { 2 } ^ { - } > 0$ , we have that

$$
G ( z ( x ) ) = Q _ { 0 } \left( \left\{ \frac { \widehat { c } _ { j } \eta _ { 2 } ( x ) } { \eta _ { 2 } ( x ) } \right\} _ { j = 1 } ^ { J } \right) R _ { \eta } ( x ) = Q _ { 0 } ( \{ \widehat { c } _ { j } \} ) R _ { \eta } ,
$$

with $Q _ { 0 } ( \{ \widehat { c } _ { j } \} )$ independent of x. Then

$$
\begin{array} { r l } & { \mathrm { ( I I ) } = \| Q _ { 0 } ( \{ \widehat { c } _ { j } \} _ { j = 1 } ^ { J } ) R _ { \eta } - \alpha ( u ) \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } } \\ & { \qquad \leq | Q _ { 0 } ( \{ \widehat { c } _ { j } \} ) _ { j = 1 } ^ { J } | \| R _ { \eta } - \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } + | Q _ { 0 } ( \{ \widehat { c } _ { j } \} _ { j = 1 } ^ { J } ) - \alpha ( u ) | \| \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } } \\ & { \qquad \leq M _ { Q _ { 0 } } \epsilon _ { \eta } + | Q _ { 0 } ( \{ \widehat { c } _ { j } \} _ { j = 1 } ^ { J } ) - \alpha ( u ) | \| \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } . } \end{array}
$$

We have that

$$
\begin{array} { r l } { | Q _ { 0 } \big ( \{ \widehat { c } _ { j } \} _ { j = 1 } ^ { J } \big ) - \alpha ( u ) | \leq | Q _ { 0 } \big ( \{ \widehat { c } _ { j } \} _ { j = 1 } ^ { J } \big ) - \widetilde { Q } \big ( \{ \widehat { c } _ { j } \} _ { j = 1 } ^ { J } \big ) | } & { } \\ { + | \widetilde { Q } \big ( \{ \widehat { c } _ { j } \} _ { j = 1 } ^ { J } \big ) - \widetilde { Q } \big ( \{ c _ { j } \} _ { j = 1 } ^ { J } \big ) | + | \widetilde { Q } \big ( \{ c _ { j } \} _ { j = 1 } ^ { J } \big ) - \alpha ( u ) | } & { } \\ { \leq \epsilon _ { Q } + | \alpha \big ( \displaystyle \sum _ { j = 1 } ^ { J } \widehat { c } _ { j } \xi _ { j } \big ) - \alpha \big ( \displaystyle \sum _ { j = 1 } ^ { J } c _ { j } \xi _ { j } \big ) | + | \alpha _ { \delta , J } ( u ) - \alpha ( u ) | . } & { } \end{array}
$$

Let $\omega _ { \alpha }$ be the modulus of continuity of α. Taking the supremum over K, we have

$$
\begin{array} { r l } & { \underset { u \in \mathbb { K } } { \operatorname* { s u p } } \left( \Pi \right) \le M _ { Q _ { 0 } } \epsilon _ { \eta } + \underset { u \in \mathbb { K } } { \operatorname* { s u p } } \left( \epsilon _ { Q } + \vert \alpha ( \underset { j = 1 } { \overset { J } { \sum } } \hat { c } _ { j } \xi _ { j } ) - \alpha ( \underset { j = 1 } { \overset { J } { \sum } } c _ { j } \xi _ { j } ) \vert + \vert \alpha _ { \delta , J } ( u ) - \alpha ( u ) \vert \right) \| \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } } \\ & { \qquad \le M _ { Q _ { 0 } } \epsilon _ { \eta } + ( \epsilon _ { Q } + \omega _ { \alpha } ( \vert \Omega \vert ^ { 1 / 2 } J ^ { 1 / 2 } \epsilon ^ { \prime } ) + \epsilon _ { \alpha } ) \| \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } , } \end{array}
$$

where we have used in the final inequality the conversion from the bound

$$
\| \sum _ { j = 1 } ^ { J } ( \widehat { c } _ { j } - c _ { J } ) \xi _ { j } \| _ { L ^ { 2 } ( \Omega ) } \leq J ^ { 1 / 2 } \epsilon ^ { \prime }
$$

to obtain the bound

$$
\| \sum _ { j = 1 } ^ { J } ( \widehat { c } _ { j } - c _ { J } ) \xi _ { j } \| _ { L ^ { 1 } ( \Omega ) } \leq | \Omega | ^ { 1 / 2 } J ^ { 1 / 2 } \epsilon ^ { \prime } ,
$$

using Cauchy-Schwarz. Collecting (I) and (II),

$$
\operatorname* { s u p } _ { u \in \mathbb { K } } \| \alpha ( u ) \eta - \widetilde { \alpha } ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \leq C _ { N } \epsilon _ { N } + M _ { Q _ { 0 } } \epsilon _ { \eta } + ( \epsilon _ { Q } + \omega _ { \alpha } ( | \Omega | ^ { 1 / 2 } J ^ { 1 / 2 } \epsilon ^ { \prime } ) + \epsilon _ { \alpha } ) \| \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } .
$$

Given $\varepsilon ,$ first fix $\delta , J$ via (D.7) such that $\epsilon _ { \alpha } \| \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } < \varepsilon / 5$ . The constants $C _ { \mathcal { N } }$ $M _ { Q _ { 0 } }$ , and the domain B are then determined. Choosing

$$
\begin{array} { r } { \epsilon _ { N } < \frac { \varepsilon } { 5 C _ { N } } , \quad \epsilon _ { \eta } < \frac { \varepsilon } { 5 M _ { Q _ { 0 } } } , \quad \epsilon _ { Q } < \frac { \varepsilon } { 5 \| \eta \| _ { C ^ { \alpha ^ { \prime } } ( \Omega ) } } , \quad \epsilon ^ { \prime } \mathrm { w i t h } \omega _ { \alpha } ( | \Omega | ^ { 1 / 2 } \sqrt { J } \epsilon ^ { \prime } ) < \frac { \varepsilon } { 5 \| \eta \| _ { C ^ { \alpha ^ { \prime } } ( \Omega ) } } } \end{array}
$$

gives $\begin{array} { r } { \operatorname* { s u p } _ { u \in \mathsf { K } } \| \alpha ( u ) \eta - \widetilde \alpha ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } < \varepsilon . } \end{array}$

Second half of the proof of Lemma $3 . 1 1 \cdot$

We may assume η is not identically zero, since if it were we could set all NNO weights to be zero and satisfy the lemma statement trivially. Since η and $\psi _ { L , m ^ { * } }$ ∗ are both continuous on $\Omega , \eta$ is not identically 0, and $\psi _ { L , m ^ { * } }$ equals 0 only on a set of Lebesgue measure $0 ,$ there exists an open ball $\widehat { \Gamma } \subset \Omega$ on which neither η nor $\psi _ { L , m ^ { * } }$ is equal to 0. It is important to note that this property implies that $\psi _ { L , m ^ { * } }$ ∗ does not change sign on $\widehat { \Gamma } .$ . For some $\epsilon _ { \gamma } > 0$ , let Γ be the open ball in $\widehat { \Gamma }$ described in Lemma D.2 for $\eta _ { 2 }$ on $\widehat { \Gamma }$ , and assume Γ has radius r. We then have that for some $\gamma$

$$
\| \gamma \eta _ { 2 } - \mathbf { 1 } \| _ { L ^ { \infty } ( \Gamma ) } < \epsilon _ { \gamma } .\tag{D.13}
$$

For $\delta _ { 1 } > 0 .$ , let $\widetilde { \Gamma } _ { \delta _ { 1 } }$ be the ball that shares a center with Γ and has radius $r - \delta _ { 1 }$ similarly, let $\widetilde { \Gamma } _ { \delta _ { 1 } / 2 }$ be the ball that shares a center with Γ and has radius $\begin{array} { r } { r \mathrm { ~ - ~ } \frac { \delta _ { 1 } } { 2 } } \end{array}$ Assume $\delta _ { 1 } < \frac { r } { 2 }$ . Let $\Gamma _ { \delta _ { 1 } }$ be the standard $\frac { \delta _ { 1 } } { 2 }$ -mollification of the indicator function $\mathbf { 1 } _ { \widetilde { \Gamma } _ { \frac { \delta _ { 1 } } { 2 } } }$ (see Definition D.3). Clearly, $\Gamma _ { \delta _ { 1 } } = 1$ inside $\widetilde \Gamma _ { \delta _ { 1 } }$ and 0 on $\Omega \backslash \Gamma ;$ ; furthermore, $\Gamma _ { \delta _ { 1 } } \in C ^ { \infty } ( \overline { { \Omega } } )$ . Since $\Gamma _ { \delta _ { 1 } }$ is a continuous map from $\overline { { \Omega } }  \mathbb { R }$ , for any $\epsilon _ { 0 } > 0$ , by Lemma A.2 there exists a neural network $\widetilde { R }$ such that

$$
\| \Gamma _ { \delta _ { 1 } } - \widetilde { R } \| _ { L ^ { \infty } ( \overline { { \Omega } } ) } < \epsilon _ { 0 } .\tag{D.14}
$$

For convenience, assume $\begin{array} { l } { \epsilon _ { 0 } < \frac { 1 } { 2 } } \end{array}$ such that the image of $\overline { { \Omega } }$ under $\widetilde { R }$ is contained in $[ - \frac { 1 } { 2 } , \frac { 3 } { 2 } ]$ We note that without loss of generality, we can assume that the $\eta _ { 1 }$ and η<sub>2</sub> pair belong to the first hidden layer, since any preceding hidden layers could simply have the weights $T \equiv 0$ and the remaining finite-dimensional neural network form either part of the lifting neural network R or approximate the identity, by assumption on σ. Finally, set the initial neural network R of the NNO to be the parallelization of $R ^ { \prime }$ from (D.8) and ${ \widetilde { R } } .$ Thus, $R : \mathbb R ^ { k } \times \overline { { \Omega } } \to \mathbb R ^ { J + 1 }$ . Let the first hidden layer of our NNO have the form

$$
( \mathcal { L } _ { 1 } v ) ( x ) = \sigma \Big ( A _ { 1 } ( W _ { 1 } v ( x ) + T _ { 1 , 1 } \int _ { \Omega } v ( y ) \eta _ { 1 } ( y ) \mathrm { d } y \eta _ { 2 } ( x ) ) + b _ { 1 } \Big ) ,\tag{D.15}
$$

where $W _ { 1 } = \mathrm { d i a g } ( [ { \bf 0 } _ { J } , 1 ] ) , T _ { 1 , 1 } = \mathrm { d i a g } ( [ \gamma { \bf 1 } _ { J } , 0 ] ) $ , and $A _ { 1 }$ and $b _ { 1 }$ are to be defined later. As in the proof of Lemma 3.10, we abbreviate $c _ { j } : = \langle u , \xi _ { j , \delta } \rangle$ and $\widehat { c } _ { j } : =$ $\begin{array} { r } { \int _ { \Omega } R _ { j } ( u ( y ) , y ) \eta _ { 1 } ( y ) } \end{array}$ dy so that (D.8) reads $\mathrm { s u p } _ { u \in \mathsf { K } }$ $\begin{array} { r } { \operatorname* { m a x } _ { j \in [ J ] } | \widehat { c } _ { j } - c _ { j } | \leq \epsilon ^ { \prime } } \end{array}$ with $\epsilon ^ { \prime } < 1 $

The end result of the choices for the first layer (D.15) and R is that before applying $A _ { 1 }$ and $b _ { 1 }$ and the activation, we obtain a vector

$$
z ( x ) = [ \widehat { c } _ { 1 } \gamma \eta _ { 2 } ( x ) , \ldots , \widehat { c } _ { J } \gamma \eta _ { 2 } ( x ) , \widetilde { R } ( x ) ] .
$$

Choose $M _ { 1 } > 0$ such that $\{ z ( x ) | u \in \mathsf { K } , x \in \Omega \} \subset [ - M _ { 1 } , M _ { 1 } ] ^ { J } \times [ - \frac 1 2 , \frac 3 2 ]$ , which is possible since $\mathrm { s u p } _ { y \in \Omega } | u ( y ) | < \infty$ and $\mathrm { s u p } _ { y \in \Omega } | \eta _ { 2 } ( y ) | < \infty$ by assumption.

Similarly, choose $M _ { 2 } > 0$ such that the image of the compact set ${ \sf K } \subset L ^ { 1 } ( \Omega ; \mathbb { R } ^ { k } )$ under the mapping

$$
( u , x ) \mapsto ( c _ { 1 } \gamma \eta _ { 2 } ( x ) , \ldots , c _ { J } \gamma \eta _ { 2 } ( x ) )
$$

is contained in $[ - M _ { 2 } , M _ { 2 } ] ^ { J }$ for all $u \in \mathsf { K }$ and $x \in \Omega$ . Let $M ^ { * } = \operatorname* { m a x } \{ M _ { 1 } , M _ { 2 } \}$ Now define

$$
\widetilde Q : \{ c _ { j } \} _ { j = 1 } ^ { J } \mapsto \alpha \left( \sum _ { j = 1 } ^ { J } c _ { j } \xi _ { j } \right) ,\tag{D.16}
$$

where $\{ c _ { j } \} _ { j = 1 } ^ { J } \in \mathbb { R } ^ { J }$ . Let $M _ { \alpha }$ be such that the image of $[ - M ^ { * } , M ^ { * } ] ^ { J }$ under $\widetilde { Q }$ is contained in $[ - M _ { \alpha } , M _ { \alpha } ]$ . Let id be the identity function, and let $\times ( a , b ) : =$ ab denote the scalar multiplication operator between the outputs of two functions a and b. By Lemma A.2, there exists a neural network $Q$ such that for any $\epsilon _ { 2 } > 0$ ，

$$
\| Q - \times ( \widetilde { Q } , \mathrm { i d } ) \| _ { L ^ { \infty } ( [ - M ^ { * } , M ^ { * } ] ^ { J } \times [ - \frac { 1 } { 2 } , \frac { 3 } { 2 } ] ) } < \epsilon _ { 2 } .\tag{D.17}
$$

Let D be the depth of Q and define $M _ { Q }$ such that the image of $\left[ - M ^ { * } , M ^ { * } \right] ^ { J } \times \left[ - \textstyle { \frac { 1 } { 2 } } , \frac { 3 } { 2 } \right]$ under $Q$ and under $\times ( \widetilde { Q } , \mathrm { i d } )$ is contained in $[ - M _ { Q } , M _ { Q } ]$

We may set the weight choices $A _ { 1 }$ and $b _ { 1 }$ of $\mathcal { L } _ { 1 }$ in (D.15) as well as $\{ W _ { i } \} _ { i = 2 } ^ { D } .$ and $\{ b _ { i } \} _ { i = 2 } ^ { D }$ of hidden layers $\mathcal { L } _ { 2 } , \ldots , \mathcal { L } _ { D }$ so that $\mathcal { L } _ { D } \circ \cdot \cdot \cdot \circ \mathcal { L } _ { 2 } \circ \sigma ( A _ { 1 } v + b _ { 1 } )$ exactly emulates the neural network $Q ( v )$ of (D.17) for input v. Within these layers, we may simply set $T _ { \ell , m } \equiv 0$

Set the number of hidden layers $L = D + 1$ . By assumption η is one of the basis elements $\phi _ { L , m ^ { * } }$ ∗ for some $m ^ { * } \leq M$ , and the corresponding $\psi _ { L , m ^ { * } }$ is such that $\textstyle \int _ { \widetilde { \Gamma } _ { \delta _ { 1 } } } \psi _ { L , m ^ { * } } ( y ) \ d y \neq 0$ for any $\delta _ { 1 }$ since we chose $\Gamma \subset { \widehat { \Gamma } }$ to be a set on which $\psi _ { L , m ^ { * } }$ did not equal 0. We may then set the last hidden layer $\mathcal { L } _ { L }$ of the NNO as

$$
\mathcal { L } _ { L } ( v ) = \sigma \Big ( A _ { L } \big ( \sum _ { m = 1 } ^ { M } T _ { L , m } \int _ { \Omega } v ( y ) \psi _ { L , m } ( y ) \mathrm { d } y \phi _ { L , m } \big ) + b _ { L } \Big ) ,\tag{D.18}
$$

where $T _ { L , m } = 0$ for $m \neq m ^ { * }$ and $\begin{array} { r } { T _ { L , m ^ { * } } = \frac { 1 } { \int _ { \tilde { \Gamma } _ { \delta _ { 1 } } } \psi _ { L , m ^ { * } } ( y ) \mathrm { d } y } } \end{array}$ . The next step is to ensure that the set

$$
B _ { m ^ { \star } } : = \{ T _ { L , m ^ { \star } } \int _ { \Omega } v ( y ) \psi _ { L , m ^ { \star } } ( y ) \mathrm { d } y \eta ( x ) : v ( y ) \in [ - M _ { Q } , M _ { Q } ] \mathrm { f o r } y \in \Omega , x \in \Omega \}
$$

is bounded. We have

$$
\operatorname* { s u p } _ { x \in \Omega } \left| T _ { L , m ^ { * } } \int _ { \Omega } v ( y ) \psi _ { L , m ^ { * } } ( y ) \mathrm { ~ d } y \eta ( x ) \right| \leq \left| T _ { L , m ^ { * } } \right| M _ { Q } \| \eta \| _ { L ^ { \infty } ( \Omega ) } \| \psi _ { L , m ^ { * } } \| _ { L ^ { 1 } ( \Omega ) } .\tag{D.19}
$$

The coeficient $T _ { L , m ^ { * } }$ is bounded for fixed $m ^ { * } , \delta _ { 1 }$ , and $\Gamma _ { \mathrm { {  } } }$ , and the remaining quantities are bounded by assumption. Thus we deduce there exists $M _ { m } ;$ ∗ such that $B _ { m ^ { * } } \subset [ - M _ { m ^ { * } } , M _ { m ^ { * } } ] ^ { k ^ { \prime } }$ . By Assumptions 2.2, for any $\epsilon _ { I } .$ , there exists an identity approximating neural network I that preserves 0 such that

$$
\| \mathcal { T } - \mathrm { i d } \| _ { C ^ { s ^ { \prime } } ( [ - M _ { m ^ { * } } , M _ { m ^ { * } } ] ^ { k ^ { \prime } } ) } < \epsilon _ { I } .\tag{D.20}
$$

Let I define the parameters $A _ { L }$ and $b _ { L }$ as well as those of the output neural network $\mathcal { Q } .$ . Denote by $\widetilde { \alpha }$ the resulting constructed NNO:

$$
\widetilde \alpha ( u ) : = \mathcal Q \circ \mathcal L _ { L } \circ \cdots \circ \mathcal L _ { 1 } \circ \mathcal R ( u ) .
$$

We now prove the approximation capability of $\widetilde { \alpha }$

$$
\begin{array} { l } { \displaystyle \underset { u \in \mathbb { K } } { \operatorname* { s u p } } \| \alpha ( u ) \eta - \widetilde { \alpha } ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \leq \displaystyle \operatorname* { s u p } _ { u \in \mathbb { K } } \| \alpha ( u ) \eta - \alpha _ { \delta , J } ( u ) \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } + \operatorname* { s u p } _ { u \in \mathbb { K } } \| \alpha _ { \delta , J } ( u ) \eta - \widetilde { \alpha } ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } } \\ { ( \mathrm { D } . 2 1 ) \qquad \leq \epsilon _ { \alpha } \| \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } + \displaystyle \operatorname* { s u p } _ { u \in \mathbb { K } } \| \alpha _ { \delta , J } ( u ) \eta - \widetilde { \alpha } ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } } \end{array}
$$

by (D.7). The first term is bounded by assumption on η. We expand the second term:

$$
\operatorname* { s u p } _ { u \in \mathbb { K } } \| \alpha _ { \delta , J } ( u ) \eta - \widetilde \alpha ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } = \operatorname* { s u p } _ { u \in \mathbb { K } } \| \alpha _ { \delta , J } ( u ) \eta - \mathcal { Q } \circ \mathcal { L } _ { L } \circ \cdot \cdot \circ \mathcal { L } _ { 1 } \circ R ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } .
$$

Putting together our definitions of $\mathcal { Q } , \mathcal { L } _ { L } , \ldots , \mathcal { L } _ { 1 }$ , and R and letting $\psi : = \psi _ { L , m ^ { * } }$ we have that

$$
\begin{array} { r l } {  { \mathcal { Q } \circ \mathcal { L } _ { L } \circ \cdots \circ \mathcal { L } _ { 1 } \circ \mathcal { R } ( u ) = } } \\ & { \quad \mathcal { Z } \bigl ( T _ { L , m ^ { * } } \int _ { \Omega } Q \bigl ( \gamma \int _ { \Omega } R ^ { \prime } ( u ( y ^ { \prime } ) , y ^ { \prime } ) \eta _ { 1 } ( y ^ { \prime } ) \ \mathrm { d } y ^ { \prime } \ \eta _ { 2 } ( y ) , \widetilde { R } ( y ) \bigr ) \psi ( y ) \ \mathrm { d } y \ \eta \bigr ) . } \end{array}\tag{D.22}
$$

The remainder of the proof simply involves assembling the individual bounds on each component one at a time. Note that we have already verified that the argument of each neural network – specifically, the arguments of each of $\mathcal { T } , Q , R ^ { \prime }$ , and $\widetilde { R }$ – falls within the correct domain of approximation. For visual clarity in the remainder of the bound, let $\mathbf { a } _ { f }$ denote the argument of function $f$ in the right-hand side of $( \mathrm { D } . 2 2 ) ; \mathrm { e . g . } { \mathbf a } _ { R ^ { \prime } }$ denotes $u ( y ^ { \prime } ) , y ^ { \prime }$ . Then

$$
\operatorname* { s u p } _ { u \in \mathsf { K } } \| \alpha _ { \delta , J } ( u ) \eta - \widetilde { \alpha } ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \leq \operatorname* { s u p } _ { \underline { { u \in \mathsf { K } } } } \| \alpha _ { \delta , J } ( u ) \eta - \mathbf { a } _ { \mathcal { T } } \| _ { C ^ { s ^ { \prime } } ( \Omega ) } + \operatorname* { s u p } _ { u \in \mathsf { K } } \| \mathcal { T } ( \mathbf { a } _ { \mathcal { T } } ) - \mathbf { a } _ { \mathcal { T } } \| _ { C ^ { s ^ { \prime } } ( \Omega ) } .
$$

For the second term,

$$
\begin{array} { r l } & { \displaystyle \operatorname* { s u p } _ { u \in { \mathsf { K } } } \| \mathcal { Z } ( \mathbf { a } _ { \mathcal { T } } ) - \mathbf { a } _ { \mathcal { Z } } \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \leq c _ { s ^ { \prime } , d } \epsilon _ { I } \big ( 1 + \| \mathbf { a } _ { \mathcal { Z } } \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \big ) } \\ & { \quad \quad \quad \quad \quad \leq c _ { s ^ { \prime } , d } \epsilon _ { I } \big ( 1 + | T _ { L , m ^ { * } } | \| \psi \| _ { L ^ { 1 } ( \Omega ) } M _ { Q } \| \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \big ) } \\ & { \quad \quad \quad \quad \leq C _ { 1 } \epsilon _ { I } } \end{array}\tag{D.24}
$$

by Fa\`a di Bruno’s formula, (D.19), and (D.20), where

$C _ { 1 } = c _ { s ^ { \prime } , d } ( 1 + C _ { \Gamma , \psi } \Vert \psi \Vert _ { L ^ { 1 } ( \Omega ) } M _ { Q } \Vert \eta \Vert _ { C ^ { s ^ { \prime } } ( \Omega ) } )$ , and $\begin{array} { r } { C _ { \Gamma , \psi } = \operatorname* { s u p } _ { 0 < \delta _ { 1 } < \frac { r } { 2 } } \Bigg | \frac { 1 } { \int _ { \widetilde { \Gamma } _ { \delta _ { 1 } } } \psi ( z ) \ : \mathsf { d } z } \Bigg | < } \end{array}$ $\infty .$ , the bound on $| T _ { L , m ^ { * } } | .$ , holds since $\psi \neq 0$ does not change sign on Γ and is continuous and bounded on Ω. Critically, we may take $\delta _ { 1 }  0$ without afecting $C _ { \Gamma , \psi }$ . For (I),

$$
\begin{array} { r l } & { \underset { u \in \mathsf { K } } { \operatorname* { s u p } } \| \alpha _ { \delta , J } ( u ) \eta - \mathbf { a } _ { \mathbb { Z } } \| _ { C ^ { s ^ { \prime } } ( \Omega ) } } \\ & { \qquad \le \underset { u \in \mathsf { K } } { \operatorname* { s u p } } \bigg | \alpha _ { \delta , J } ( u ) - T _ { L , m ^ { * } } \int _ { \Omega } Q ( \mathbf { a } _ { Q } ) \psi ( y ) \mathrm { d } y \bigg | \| \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } . } \end{array}\tag{D.25}
$$

The quantity $\| \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) }$ is bounded by assumption. Bounding the first factor,

$$
\begin{array} { r l r } {  { \operatorname* { s u p } _ { u \in \mathbb { K } } | \alpha _ { \delta , J } ( u ) - T _ { L , m ^ { * } } \int _ { \Omega } Q ( \mathbf { a } _ { Q } ) \psi ( y ) \mathrm { d } y | \leq } } \\ & { } & { \underbrace { \operatorname* { s u p } _ { u \in \mathbb { K } } | \alpha _ { \delta , J } ( u ) - T _ { L , m ^ { * } } \int _ { \Omega } \alpha ( \displaystyle \sum _ { j = 1 } ^ { J } \widehat { c } _ { j } \gamma \eta _ { 2 } ( y ) \xi _ { j } ) \widetilde { R } ( y ) \psi ( y ) \mathrm { d } y | } _ { \mathrm { ( I I ) } } } \\ & { } & { + | T _ { L , m ^ { * } } | \operatorname* { s u p } _ { u \in \mathbb { K } } \Big | \int _ { \Omega } \alpha ( \displaystyle \sum _ { j = 1 } ^ { J } \widehat { c } _ { j } \gamma \eta _ { 2 } ( y ) \xi _ { j } ) \widetilde { R } ( y ) \psi ( y ) \mathrm { d } y - \int _ { \Omega } Q ( \mathbf { a } _ { Q } ) \psi ( y ) \mathrm { d } y \Big | . } \end{array}
$$

The second term is bounded by

$$
\epsilon _ { 2 } | T _ { L , m ^ { * } } | \| \psi \| _ { L ^ { 1 } ( \Omega ) } = C _ { 2 } \epsilon _ { 2 }\tag{D.26}
$$

by equation (D.17), where $C _ { 2 } = C _ { \Gamma , \psi } \| \psi \| _ { L ^ { 1 } ( \Omega ) }$ . Continuing with the bound on (II):

$$
\begin{array} { r l } & { \displaystyle \operatorname* { s u p } _ { u \in \mathbb { K } } \Bigl | \alpha _ { \delta , J } ( u ) - T _ { L , m ^ { * } } \int _ { \Omega } \alpha \left( \sum _ { j = 1 } ^ { J } \widehat { c } _ { j } \gamma \eta _ { 2 } ( y ) \xi _ { j } \right) \widetilde { R } ( y ) \psi ( y ) { \mathrm { d } } y \Bigr | } \\ & { \quad \le \displaystyle \operatorname* { s u p } _ { u \in \mathbb { K } } \Bigl | \alpha _ { \delta , J } ( u ) - T _ { L , m ^ { * } } \int _ { \Omega } \alpha \left( \sum _ { j = 1 } ^ { J } \widehat { c } _ { j } \gamma \eta _ { 2 } ( y ) \xi _ { j } \right) \Gamma _ { \delta _ { 1 } } ( y ) \psi ( y ) { \mathrm { d } } y \Bigr | } \\ & { \quad \quad + | T _ { L , m ^ { * } } | \epsilon _ { 0 } \| \psi \| _ { L ^ { 1 } ( \Omega ) } M _ { \alpha } , } \end{array}\tag{D.27}
$$

by (D.14). Let $C _ { 3 } = C _ { \Gamma , \psi } \| \psi \| _ { L ^ { 1 } ( \Omega ) } M _ { \alpha }$ . Continuing to bound the first term in line (D.27),

$$
\begin{array} { r l } & { \quad \underset { u \in \mathbb { K } } { \operatorname* { s u p } } \left| \alpha _ { \delta , j } ( u ) - T _ { L , m } , \int _ { \Omega } \Omega \left( \displaystyle \sum _ { j = 1 } ^ { J } \widehat { c } _ { j } \gamma \eta _ { 2 } ( y ) \xi _ { j } \right) \Gamma _ { \delta _ { 1 } } ( y ) \psi ( y ) \mathrm { d } y \right| = } \\ & { \leq \underset { u \in \mathbb { K } } { \operatorname* { s u p } } \Bigg | \alpha _ { \delta , j } ( u ) - T _ { L , m } , \int _ { \Omega } \alpha \left( \displaystyle \sum _ { j = 1 } ^ { J } \widehat { c } _ { j } \xi _ { j } \right) \Gamma _ { \delta _ { 1 } } ( y ) \psi ( y ) \mathrm { d } y \Bigg | } \\ & { \quad + \underset { u \in \mathbb { K } } { \operatorname* { s u p } } \Bigg | T _ { L , m } , \int _ { \Omega } \alpha \Bigg ( \displaystyle \sum _ { j = 1 } ^ { J } \widehat { c } _ { j } \xi _ { j } \Big ) \Gamma _ { \delta _ { 1 } } ( y ) \psi ( y ) \mathrm { d } y } \\ & { \quad - T _ { L , m } , \displaystyle \int _ { \Omega } \alpha \left( \displaystyle \sum _ { j = 1 } ^ { J } \widehat { c } _ { j } \gamma \eta _ { 2 } ( y ) \xi _ { j } \right) \Gamma _ { \delta _ { 1 } } ( y ) \psi ( y ) \mathrm { d } y \Bigg | . } \end{array}
$$

The second supremum term is bounded by

$$
\vert T _ { L , m ^ { * } } \vert \Vert \psi \Vert _ { L ^ { 1 } ( \Gamma ) } \underset { u \in \mathsf { K } } { \operatorname* { s u p } } \underset { y \in \Gamma } { \operatorname* { s u p } } \big | \alpha \big ( \sum _ { j = 1 } ^ { J } \widehat { c } _ { j } \xi _ { j } \big ) - \alpha \big ( \sum _ { j = 1 } ^ { J } \widehat { c } _ { j } \gamma \eta _ { 2 } ( y ) \xi _ { j } \big ) \big | .\tag{D.28}
$$

Let $\omega _ { \alpha }$ be the modulus of continuity of $\alpha .$ . From the orthonormality of $\xi _ { j }$ in $L ^ { 2 } ( \Omega )$ and H¨older’s inequality, it holds that

$$
\lVert \sum _ { j = 1 } ^ { J } \widehat { c } _ { j } ( 1 - \gamma \eta _ { 2 } ( y ) ) \xi _ { j } \rVert _ { L ^ { 1 } ( \Omega ) } \leq | \Omega | ^ { 1 / 2 } \sum _ { j = 1 } ^ { J } | \widehat { c } _ { j } | | 1 - \gamma \eta _ { 2 } ( y ) | ,
$$

and using (D.8) and (D.13), we may then bound (D.28) by

$$
\lvert T _ { L , m ^ { * } } \rvert \| \psi \| _ { L ^ { 1 } ( \Gamma ) } \operatorname* { s u p } _ { u \in \mathsf { K } } \omega _ { \alpha } ( \epsilon _ { \gamma } | \Omega | ^ { 1 / 2 } J \operatorname* { m a x } _ { j \in [ J ] } ( | \langle u , \xi _ { j , \delta } \rangle | + 1 ) ) \leq | T _ { L , m ^ { * } } | \| \psi \| _ { L ^ { 1 } ( \Gamma ) } \omega _ { \alpha } ( \epsilon _ { \gamma } C _ { \Omega , J , \delta } ) .
$$

Next, we show that

$$
| T _ { L , m ^ { * } } | \| \psi \| _ { L ^ { 1 } ( \Gamma ) } = \frac { \int _ { \Gamma } | \psi ( y ) | \mathrm { ~ d } y } { | \int _ { \widetilde { \Gamma } _ { \delta _ { 1 } } } \psi ( y ) \mathrm { ~ d } y | }
$$

may be bounded independently of Γ given fixed $\widehat { \Gamma }$ through suficiently small choice of $\delta _ { 1 }$ . Since $\psi$ does not change sign on $\widehat { \Gamma } .$ , write $\begin{array} { r } { | \int _ { \widetilde { \Gamma } _ { \delta _ { 1 } } } \psi ( y ) \mathrm { d } y | = \int _ { \widetilde { \Gamma } _ { \delta _ { 1 } } } | \psi ( y ) | \mathrm { d } y = } \end{array}$ $\begin{array} { r } { \int _ { \Gamma } | \psi ( y ) | \ \mathrm { d } y - \int _ { \Gamma \backslash \widetilde { \Gamma } _ { \delta _ { 1 } } } | \psi ( y ) | \ \mathrm { d } y \geq \int _ { \Gamma } | \psi ( y ) | \ \mathrm { d } y - \| \psi \| _ { L ^ { \infty } ( \Omega ) } ^ { \cdot } d \omega _ { d } r ^ { d - 1 } \delta _ { 1 } } \end{array}$ , where $\omega _ { d }$ is the volume of the unit ball in $\mathbb { R } ^ { d }$ , and we have used the fact that the volume of the shell $\Gamma \setminus \\setminus \widetilde { \Gamma } _ { \delta _ { 1 } }$ is upperbounded by $d \omega _ { d } r ^ { d - 1 } \delta _ { 1 }$ . Since $\psi$ is continuous and does not equal 0 on $\begin{array} { r } { \Gamma , \int _ { \Gamma } | \psi ( y ) | \ \mathrm { d } y > 0 . } \end{array}$ so that $\begin{array} { r } { \delta _ { 1 } ^ { * } ( \Gamma ) : = \frac { \int _ { \Gamma } | \psi ( y ) | \ : \mathsf { d } y } { 2 \| \psi \| _ { L ^ { \infty } ( \Omega ) } d \omega _ { d } r ^ { d - 1 } } } \end{array}$ . Then for any $\delta _ { 1 }$ such that $0 < \delta _ { 1 } \leq \delta _ { 1 } ^ { * } ( \Gamma )$ , we have $| T _ { L , m ^ { * } } | \| \psi \| _ { L ^ { 1 } ( \Gamma ) } \leq 2 ,$ so (D.29) becomes simply

$$
| T _ { L , m ^ { * } } | \| \psi \| _ { L ^ { 1 } ( \Gamma ) } \operatorname* { s u p } _ { u \in \mathsf { K } } \omega _ { \alpha } ( \epsilon _ { \gamma } | \Omega | ^ { 1 / 2 } J \operatorname* { m a x } _ { j \in [ J ] } ( | \langle u , \xi _ { j , \delta } \rangle | + 1 ) ) \leq 2 \omega _ { \alpha } ( \epsilon _ { \gamma } C _ { \Omega , J , \delta } ) ,\tag{D.30}
$$

so long as we choose $\delta _ { 1 } < \delta _ { 1 } ^ { * } ( \Gamma )$ after choosing Γ. Now having dealt with the approximation of 1 by γη<sub>2</sub> on Γ, we continue to bound term (III):

$$
\begin{array} { r l } & { \underset { \mathrm { u \not = f } } { \operatorname* { s u p } } \biggl | \alpha _ { \delta , j } ( \mathrm { u } ) - T _ { i , \mathrm { r } , m } \cdot \displaystyle \int _ { \Omega } \alpha \biggl ( \underset { j = 1 } { \overset { j } { \sum } } \int _ { \Omega } R _ { j } \bigl ( \alpha _ { R , j } \bigr ) \eta _ { \mathrm { r } } ( j ^ { \prime } ) \mathrm { d } y ^ { \prime } \xi _ { j } \biggr ) \Gamma _ { \delta , \mathrm { \ t } } ( y ) \psi ( y ) \mathrm { d } y \biggr | } \\ & { \leq \underset { \mathrm { u \not = f } } { \operatorname* { s u p } } \biggl | \alpha _ { \delta , j } ( \mathrm { u } ) - T _ { i , \mathrm { r } , m } \cdot \displaystyle \int _ { \Omega } \alpha \biggl ( \underset { j = 1 } { \overset { j } { \sum } } \bigl ( u , \xi _ { j , \mathrm { r } } ( \mathrm { u } , \xi _ { j } ) \xi _ { j } \bigr ) \Gamma _ { \delta , \mathrm { \ t } } ( y ) \psi ( y ) \mathrm { d } y \biggr | } \\ & { \qquad \quad + \underset { \mathrm { u \not = f } } { \operatorname* { s u p } } \biggl | T _ { i , \mathrm { r } , m } \cdot \displaystyle \int _ { \Omega } \alpha \biggl ( \underset { j = 1 } { \overset { j } { \sum } } \int _ { \Omega } R _ { j } \bigl ( \alpha _ { R , j } \bigr ) \eta _ { \mathrm { r } } ( y ^ { \prime } ) \mathrm { d } y ^ { \prime } \xi _ { j } \biggr ) \Gamma _ { \delta , \mathrm { \ t } } ( y ) \psi ( y ) \mathrm { d } y } \\ & { \qquad - T _ { i , \mathrm { r } , m } \cdot \displaystyle \int _ { \Omega } \alpha \biggl ( \underset { j = 1 } { \overset { j } { \sum } } \langle u , \xi _ { j , \mathrm { r } } ( \mathrm { u } , \xi ) \xi _ { j } \rangle \Gamma _ { \delta , \mathrm { \ t } } ( y ) \psi ( y ) \mathrm { d } y \biggr | } \\ &  \qquad \leq ( \mathrm { W i } ) + [ T _ { i , \mathrm { r } , m } ] \psi \bigl | \alpha _   \end{array}\tag{D.31}
$$

by (D.8) and another application of H¨older’s inequality within the argument of $\omega _ { \alpha }$ Set $C _ { 4 } = C _ { \Gamma , \psi } \| \psi \| _ { L ^ { 1 } ( \Omega ) }$ . We can see that, within (IV),

$$
\begin{array} { r l } & { T _ { L , m ^ { * } } \displaystyle \int _ { \Omega } \alpha \left( \displaystyle \sum _ { j = 1 } ^ { J } \langle u , \xi _ { j , \delta } \rangle \xi _ { j } \right) \Gamma _ { \delta _ { 1 } } ( y ) \psi ( y ) \mathrm { d } y } \\ & { \displaystyle = \alpha \left( \displaystyle \sum _ { j = 1 } ^ { J } \langle u , \xi _ { j , \delta } \rangle \xi _ { j } \right) \left( 1 + \displaystyle \int _ { \Gamma \backslash \tilde { \Gamma } _ { \delta _ { 1 } } } \frac { 1 } { \int _ { \tilde { \Gamma } _ { \delta _ { 1 } } } \psi ( z ) \mathrm { d } z } \psi ( y ) \Gamma _ { \delta _ { 1 } } ( y ) \mathrm { d } y \right) . } \end{array}
$$

Since $\begin{array} { r } { \alpha ( \sum _ { j = 1 } ^ { J } \langle u , \xi _ { j , \delta } \rangle \xi _ { j } ) = \alpha _ { \delta , J } ( u ) } \end{array}$ by definition, so finally we have that

$$
\begin{array} { r l r } {  { \mathrm { ( I V ) } \le \operatorname* { s u p } _ { u \in \mathsf { K } } | \alpha _ { \delta , J } ( u ) | | \int _ { \Gamma \backslash \widetilde { \Gamma } _ { \delta _ { 1 } } } \frac { 1 } { \sqrt { \widetilde { \Gamma } _ { \delta _ { 1 } } \psi ( z ) \mathbb { d } z } } \psi ( y ) \Gamma _ { \delta _ { 1 } } ( y ) \mathbb { d } y | } } \\ & { } & { \le M _ { \delta , J } \| \psi \| _ { L ^ { \infty } ( \Omega ) } C _ { \Gamma , \psi } \omega _ { d } d r ^ { d - 1 } \delta _ { 1 } , } \end{array}\tag{D.32}
$$

where $\begin{array} { r } { M _ { \delta , J } = \operatorname* { s u p } _ { u \in \mathsf { K } } | \alpha _ { \delta , J } ( u ) | < \infty } \end{array}$ by continuity of $\alpha _ { \delta , J }$ and compactness of $\mathsf { K } , \omega _ { d }$ is the volume of the unit ball in $\mathbb { R } ^ { d }$ , and we have used the fact that the volume of the shell $\Gamma \setminus \\\\widetilde { \Gamma } _ { \delta _ { 1 } }$ is upperbounded by $d \omega _ { d } r ^ { d - 1 } \delta _ { 1 }$ . Let $C _ { 5 } = M _ { \delta , J } \lVert \psi \rVert _ { L ^ { \infty } ( \Omega ) } C _ { \Gamma , \psi } \omega _ { d } d r ^ { d - 1 }$

Bringing together the bounds of all the terms we have obtained, from equations (D.23), (D.24), (D.25), (D.26), (D.27), (D.30), (D.31), and (D.32), we have

$$
\begin{array} { r l } & { \displaystyle \operatorname* { s u p } _ { u \in \mathsf { K } } \| \alpha _ { \delta , J } ( u ) \eta - \widetilde \alpha ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \leq C _ { 1 } \epsilon _ { I } + \| \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \Big ( C _ { 2 } \epsilon _ { 2 } + C _ { 3 } \epsilon _ { 0 } } \\ & { \quad \quad \quad + \ : 2 \omega _ { \alpha } \big ( \epsilon _ { \gamma } C _ { \Omega , J , \delta } \big ) + C _ { 4 } \omega _ { \alpha } \big ( J ^ { 1 / 2 } \epsilon ^ { \prime } \big | \Omega \big | ^ { 1 / 2 } \big ) + C _ { 5 } \delta _ { 1 } \Big ) } \end{array}
$$

The choices of $\Omega , \delta , s ^ { \prime } , \eta , m ^ { * } , \psi , \widehat { \Gamma }$ , and J are fixed. For any $\varepsilon ^ { \prime } > 0 ,$ , we then may choose $\epsilon _ { \gamma }$ such that $\begin{array} { r l r } { \omega _ { \alpha } \mathopen { } \mathclose \bgroup \left( \epsilon _ { \gamma } C _ { \Omega , J , \delta } \aftergroup \egroup \right) } & { \leq } & { \frac { \varepsilon ^ { \prime } } { 1 4 \| \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } } } \end{array}$ The choice of ϵ deter- $\epsilon _ { \gamma }$ mines $\Gamma , \gamma ,$ and $r ,$ which determines the value of ${ \mathit { C } } _ { \Gamma , \psi }$ as well as the domain bounds $M ^ { * } , M _ { \alpha } .$ , and $M _ { Q }$ . Then constants $C _ { 1 } , \ C _ { 2 } , \ C _ { 3 } , \ C _ { 4 }$ , and $C _ { 5 }$ are determined. Choose $\epsilon ^ { \prime }$ such that $\begin{array} { r } { \omega _ { \alpha } \big ( J ^ { 1 / 2 } \epsilon ^ { \prime } | \Omega | ^ { 1 / 2 } \big ) \le \frac { \varepsilon ^ { \prime } } { 7 C _ { 4 } \| \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } } . } \end{array}$ . We choose $\delta _ { 1 }$ such that $\begin{array} { r } { \delta _ { 1 } \le \operatorname* { m i n } \left( \frac { \varepsilon ^ { \prime } } { 7 \| \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } C _ { 5 } } , \frac { r } { 2 } , \delta _ { 1 } ^ { \ast } ( \Gamma ) \right) , \epsilon _ { 2 } } \end{array}$ such that $\epsilon _ { 2 } < \frac { \varepsilon ^ { \prime } } { 7 C _ { 2 } \mathopen { } \mathclose \bgroup \left\| \eta \mathclose \bgroup \egroup \right\| _ { C ^ { s ^ { \prime } } ( \Omega ) } } , \epsilon _ { 0 }$ such that $\begin{array} { r } { \epsilon _ { 0 } < \frac { \varepsilon ^ { \prime } } { 7 C _ { 3 } \left\| \eta \right\| _ { C ^ { s ^ { \prime } } ( \Omega ) } } } \end{array}$ , and $\epsilon _ { I }$ such that $\begin{array} { r } { \dot { \epsilon _ { I } } < \frac { \varepsilon ^ { \prime } } { 7 C _ { 1 } } } \end{array}$ . Combining this result with (D.7), we have

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { u \in { \mathbb { K } } } \| \alpha ( u ) \eta - \widetilde \alpha ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } \leq \displaystyle \operatorname* { s u p } _ { u \in { \mathbb { K } } } \| \alpha ( u ) \eta - \alpha _ { \delta , J } ( u ) \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } + \displaystyle \operatorname* { s u p } _ { u \in { \mathbb { K } } } \| \alpha _ { \delta , J } \eta - \widetilde \alpha ( u ) \| _ { C ^ { s ^ { \prime } } ( \Omega ) } } & { } \\ { \leq \displaystyle \operatorname* { s u p } _ { u \in { \mathbb { K } } } | \alpha ( u ) - \alpha _ { \delta , J } ( u ) | \| \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } + \varepsilon ^ { \prime } } & { } \\ { \leq \epsilon _ { \alpha } \| \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } + \varepsilon ^ { \prime } . } & { } \end{array}
$$

Choosing $\varepsilon ^ { \prime } < \frac { \varepsilon } { 2 }$ and $\epsilon _ { \alpha } \leq \frac { \varepsilon } { 2 \| \eta \| _ { C ^ { s ^ { \prime } } ( \Omega ) } }$ , we have the desired result (3.4).

## Appendix E. Eigenfunctions of the Laplacian

Here we state two lemmas related to approximation by sum of nonlinear functionals and eigenfunctions of Laplacian. The first lemma gives a convergence result for projection onto the basis of the eigenfunctions of the Laplacian. Though results of this type are generally well-known [16, 17], we were unable to find a proof for this particular case in the literature, so we have included a proof here. We note that the compact support assumption of the set to be approximated is essential since, by construction, projection onto the eigenfunctions of the Laplacian nullifies at the boundary the projected function w and every power $\Delta ^ { k } w$ for integer $k \geq 0$

Lemma E.1 (Projection onto the Eigenfunctions of the Laplacian). Let Ω be a bounded Lipschitz domain in $\mathbb { R } ^ { d }$ . Let W be a bounded set in $C _ { c } ^ { k } ( \overline { { \Omega } } )$ , which is the subspace of $C ^ { k } ( { \overline { { \Omega } } } )$ whose functions have compact support in Ω. Assume $k > d { + } 1$ is an integer. Let $\{ \lambda _ { 1 } , \eta _ { 1 } \} , \{ \lambda _ { 2 } , \eta _ { 2 } \} , . .$ . be the eigenpairs of the Laplacian on Ω subject to homogeneous Dirichlet boundary conditions, ordered such that $0 < \lambda _ { 1 } \leq \lambda _ { 2 } , \cdot \cdot \cdot$ Let $\begin{array} { r } { P _ { J } w = \sum _ { j = 1 } ^ { J } \langle w , \eta _ { j } \rangle _ { L ^ { 2 } ( \Omega ) } \eta _ { j } } \end{array}$ be the projection of w onto the first J eigenfunctions over Ω. Then the approximation via the projection $P _ { J }$ converges uniformly in $C ( { \overline { { \Omega } } } )$ as $J \to \infty .$

$$
\operatorname* { l i m } _ { J \to \infty } \operatorname* { s u p } _ { w \in W } \Big \| w - \sum _ { j = 1 } ^ { J } \langle w , \eta _ { j } \rangle _ { L ^ { 2 } ( \Omega ) } \eta _ { j } \Big \| _ { C ( \overline { \Omega } ) } = 0 .
$$

Proof. Let $\begin{array} { r } { e _ { J } = w - P _ { J } w = \sum _ { i = J + 1 } ^ { \infty } \langle w , \eta _ { j } \rangle _ { L ^ { 2 } ( \Omega ) } \eta _ { j } } \end{array}$ since $\{ \eta _ { j } \} _ { j = 1 } ^ { \infty }$ form a basis for $L ^ { 2 } ( \Omega )$ . We recall that $\eta _ { j } \in C ^ { \infty } ( \Omega ) \cap C ( \overline { { \Omega } } )$ , and $\eta _ { j } = 0$ on ∂Ω [16].

We bound each inner product individually via

$$
\begin{array} { l } { \langle { \boldsymbol w } , { \boldsymbol \eta } _ { j } \rangle = \displaystyle \int _ { \Omega } w { \boldsymbol \eta } _ { j } \ \mathrm { d } x } \\ { = - \displaystyle \int _ { \Omega } \frac { 1 } { \lambda _ { j } } w \Delta { \boldsymbol \eta } _ { j } \ \mathrm { d } x } \\ { = \displaystyle \frac { 1 } { \lambda _ { j } } \int _ { \Omega } \nabla w \cdot \nabla { \boldsymbol \eta } _ { j } \ \mathrm { d } x - \frac { 1 } { \lambda _ { j } } \int _ { \partial \Omega } w ( \nabla { \boldsymbol \eta } _ { j } \cdot \widehat { \boldsymbol n } ) \ \mathrm { d } x } \\ { = \displaystyle \frac { 1 } { \lambda _ { j } } \int _ { \partial \Omega } \eta _ { j } ( \nabla w \cdot \widehat { \boldsymbol n } ) \ \mathrm { d } x - \frac { 1 } { \lambda _ { j } } \int _ { \Omega } \eta _ { j } \nabla \cdot ( \nabla w ) \ \mathrm { d } x } \\ { = - \displaystyle \frac { 1 } { \lambda _ { j } } \int _ { \Omega } \eta _ { j } \Delta w \ \mathrm { d } x . } \end{array}
$$

Since $w$ and all its derivatives have compact support in $\Omega .$ , we may iterate this procedure r times to obtain

$$
\langle w , \eta _ { j } \rangle = \frac { ( - 1 ) ^ { r } } { \lambda _ { j } ^ { r } } \langle \Delta ^ { r } w , \eta _ { j } \rangle .\tag{E.1}
$$

Furthermore, from [13, Lemma 2.1] combined with the unlabeled bound on $a _ { t } ( x , y )$ contained on $\left[ 1 3 , \mathrm { p } . \ 3 7 7 \right]$ , we can bound the eigenfunctions via

$$
\| \eta _ { j } \| _ { L ^ { \infty } ( \overline { { \Omega } } ) } \le e ^ { ( \lambda _ { j } / 2 ) t } ( 4 \pi t ) ^ { - d / 4 }
$$

for any $t > 0$ . Optimizing over t, we have the bound

$$
\| \eta _ { j } \| _ { L ^ { \infty } ( \overline { { \Omega } } ) } \le c _ { d } \lambda _ { j } ^ { d / 4 } ,\tag{E.2}
$$

where $c _ { d }$ is a constant dependent only on d. Then we may bound the tail

$$
\begin{array} { r l } {  { \big \| e _ { J } \big \| _ { C ( \overline { \Omega } ) } \leq \sum _ { j > J } \big | \langle w , \eta _ { j } \rangle \big | \big \| \eta _ { j } \big \| _ { L ^ { \infty } ( \overline { \Omega } ) } } \quad } & { } \\ & { \leq c _ { d } \displaystyle \sum _ { j > J } \lambda _ { j } ^ { d / 4 - r } | \langle \Delta ^ { r } w , \eta _ { j } \rangle \big | } \\ & { \leq c _ { d } \big ( \displaystyle \sum _ { j > J } \lambda _ { j } ^ { d / 2 - 2 r } \big ) ^ { 1 / 2 } \big ( \displaystyle \sum _ { j > J } | \langle \Delta ^ { r } w , \eta _ { j } \rangle | ^ { 2 } \big ) ^ { 1 / 2 } } \\ & { \leq c _ { d } \big \| \Delta ^ { r } w \big \| \big ( \displaystyle \sum _ { j > J } \lambda _ { j } ^ { d / 2 - 2 r } \big ) ^ { 1 / 2 } . } \end{array}
$$

We now must bound the summation. Weyl’s law states that the number $N ( \lambda )$ of these eigenvalues that are less than or equal to λ satisfies

$$
\operatorname* { l i m } _ { \lambda \to \infty } \frac { N ( \lambda ) } { \lambda ^ { d / 2 } } = ( 2 \pi ) ^ { - d } \omega _ { d } \mathrm { v o l } ( \Omega ) ,
$$

where $\omega _ { d }$ is the volume of the unit ball in $\mathbb { R } ^ { d } \ [ 3 6 , \ 4 3 ]$ . Thus, for any choice of $\epsilon ,$ there exists a $\lambda _ { \epsilon }$ such that for all $\lambda > \lambda _ { \epsilon }$

$$
\left| \frac { N ( \lambda ) } { \lambda ^ { d / 2 } } - ( 2 \pi ) ^ { - d } \omega _ { d } \mathrm { v o l } ( \Omega ) \right| < \epsilon .
$$

To handle duplicate eigenvalues, let $j ^ { * }$ be the largest index k such that $\lambda _ { k } \le \lambda _ { j }$ Note that if $\lambda _ { j }$ is not a duplicate eigenvalue, then $j ^ { * } = j$ . Then for $\lambda _ { j } > \lambda _ { \epsilon } .$

$$
\Big | \frac { j ^ { * } } { \lambda _ { j } ^ { d / 2 } } - ( 2 \pi ) ^ { - d } \omega _ { d } \mathrm { v o l } ( \Omega ) \Big | < \epsilon .
$$

Rearranging yields

$$
\frac { 1 } { \lambda _ { j } } < \left( \frac { \epsilon + ( 2 \pi ) ^ { - d } \omega _ { d } \mathrm { v o l } ( \Omega ) } { j ^ { * } } \right) ^ { 2 / d } ,
$$

which holds for $j > J _ { \epsilon }$ , where $J _ { \epsilon }$ is the smallest index such that $\lambda _ { J _ { \epsilon } } > \lambda _ { \epsilon }$

Then,

$$
\begin{array} { l } { \displaystyle \| e _ { J _ { \epsilon } } \| _ { C ( \overline { \Omega } ) } \leq c _ { d } \| \Delta ^ { r } w \| \left( \sum _ { j > J _ { \epsilon } } \left( \frac { \epsilon + ( 2 \pi ) ^ { - d } \omega _ { d } \mathrm { v o l } ( \Omega ) } { j } \right) ^ { \frac { 4 r } { d } - 1 } \right) ^ { 1 / 2 } } \\ { \leq c _ { d } \| \Delta ^ { r } w \| ( \epsilon + ( 2 \pi ) ^ { - d } \omega _ { d } \mathrm { v o l } ( \Omega ) ) ^ { \frac { 2 r } { d } - \frac { 1 } { 2 } } \left( \int _ { J _ { \epsilon } } ^ { \infty } \frac { 1 } { j ^ { \frac { 4 r } { d } - 1 } } \mathrm { d } j \right) ^ { 1 / 2 } } \end{array}
$$

For convergence, we require $\begin{array} { r } { r > \frac { d } { 2 } ; } \end{array}$ ; let $r$ be the smallest such integer that satisfies this constraint. So that $\Delta ^ { r } w$ is defined and in $L ^ { 2 } .$ , we require $2 r \leq k$ . By assumption, $W \subset C _ { c } ^ { k } ( \overline { { \Omega } } )$ for $k > d + 1$ , so $\operatorname* { s u p } _ { w \in W } \| \Delta ^ { r } w \|$ is finite for this choice of r. The bound becomes

$$
\| e _ { J } \| _ { C ( \overline { { \Omega } } ) } \leq c _ { d , W , r , \Omega } J ^ { 1 - \frac { 2 r } { d } } ,
$$

for $J > J _ { 1 }$ , where $\begin{array} { r } { c _ { d , W , r , \Omega } = \operatorname* { s u p } _ { w \in W } \| \Delta ^ { r } w \| ( 1 + ( 2 \pi ) ^ { - d } \omega _ { d } \mathrm { v o l } ( \Omega ) ) ^ { \frac { 2 r } { d } - \frac 1 2 } } \end{array}$ . We have set $\epsilon = 1$ , as it plays no role in the convergence. Since this is the tail bound, we have that

$$
\operatorname* { l i m } _ { J \to \infty } \operatorname* { s u p } _ { w \in W } \left\| w - \sum _ { j = 1 } ^ { J } \langle w , \eta _ { j } \rangle _ { L ^ { 2 } ( \Omega ) } \eta _ { j } \right\| _ { C ( \overline { { \Omega } } ) } = \operatorname* { l i m } _ { J \to \infty } \operatorname* { s u p } _ { w \in W } \left\| e _ { J } \right\| _ { C ( \overline { { \Omega } } ) } = 0 .
$$

![](images/3c65f44f1e288b63a569b4f654851df2b5b0ad962be745a8946b40759d6865fb.jpg)

Department of Computing and Mathematical Sciences<sub>,</sub> Caltech<sub>,</sub> Pasadena<sub>,</sub> CA Email address: astuart@caltech.edu

Department of Mathematics<sub>,</sub> ETH Zurich<sub>,</sub> Z¨ urich<sub>,</sub> Switzerland¨

Email address: margaret.trautner@sam.math.ethz.ch