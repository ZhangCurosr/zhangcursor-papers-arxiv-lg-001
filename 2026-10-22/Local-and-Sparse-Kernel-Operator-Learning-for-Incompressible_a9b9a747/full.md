# Local and Sparse Kernel Operator Learning for Incompressible Flows: An Operator Lagrange Formulation

Ramansh Sharma<sup>a</sup>, Matthew Lowery<sup>a</sup>, Houman Owhadi<sup>b</sup> and Varun Shankar<sup>a,∗</sup>

<sup>a</sup>Kahlert School ofComputing, University ofUtah, UT, USA

<sup>b</sup>Department ofComputing and Mathematical Sciences, California Institute ofTechnology, CA, USA

## A R T I C L E I N F O

## A BS T R AC T

We present a local and sparse kernel method for learning the solution operators of incompressible flows. It builds on our earlier property-preserving kernel method, whose global output reconstruction and dense operator regression scale poorly. Our method replaces the global output reconstruction by local divergence-free interpolants with divergence-free polynomials of any degree, solved in parallel on the GPU. These interpolants are divergence free, periodic on periodic domains, and, with a multiscale kernel, approach Kolmogorov statistics under refinement. A compactly supported kernel, positive semidefinite in any input dimension, makes the operator regression sparse. Its support radius, and hence the sparsity, is fitted to data, and optional afine polynomials in the input capture global trends. Operator Lagrange weights join the two. They cost one sparse solve per new input and act on the training fields before the output reconstruction, so their local coeficients are never formed. We show that, unlike direct evaluation, Lagrange evaluation does not amplify solve errors by the cancellation among spatial coeficients. On ten fluid benchmarks, validation selects sparse matrices nearly everywhere, with densities that mostly decrease as the training set grows, and afine augmentation lowers the error by more than an order of magnitude for forced isotropic turbulence. These sparse models are at least as accurate as our earlier dense method and more accurate than neural operators. Lagrange evaluation is as accurate as direct evaluation with local output reconstructions, and up to nearly three orders of magnitude more accurate with global ones. We also present GPU timings and storage.

## 1. Introduction

Uncertainty quantification, design, and inverse problems for incompressible flows require the velocity field for many initial conditions, boundary conditions, or forcing functions, and the repeated solution of the Navier–Stokes equations then dominates the cost [43, 59–61]. One remedy is to learn the solution operator, the map from the problem data to the velocity, from a collection of simulations [7]. Reduced-order models have long served this purpose [5, 42], and neural operators such as DeepONets, Fourier and graph neural operators, transformer-based models, and kernel neural operators now learn such maps from large data sets [32–34, 36, 37, 70]. We seek an operator learner for incompressible flows that satisfies four criteria. First, every prediction should be divergence free, and periodic on periodic domains, however accurately the operator itself is learned. Second, the cost of the output representation should grow linearly with the number of output points, so that fine three-dimensional outputs remain afordable. Third, the regression should handle ten thousand or more training functions without factoring a dense matrix. Fourth, every stage should run in parallel on a graphics processing unit (GPU).

Kernel methods ofer an attractive alternative to neural operators [2, 29, 38]. Batlle et al. [2] learn an operator from point samples in two steps. An operator regression, a kernel regression on the space of sampled inputs, maps the samples of the input function to the samples of the output function. An output reconstruction, a kernel interpolant of these samples in space (or spacetime), then recovers the output function. Since both steps are kernel interpolation or regression problems, the learned operator has a closed form, few parameters, and an error analysis. For flows, the output representation is the natural place to impose physics [30, 45]. In our previous work [53], we therefore regressed the coeficients of a divergence-free kernel interpolant of the output instead of its point samples, so that every prediction is exactly divergence free regardless of the accuracy of the operator regression. We called this the property-preserving kernel method (PPKM). PPKM satisfies the first criterion, but both of its kernel problems are global. Its operator regression factors one dense kernel matrix on all � training inputs, at a cost of $O ( N ^ { 3 } )$ operations and $O ( N ^ { 2 } )$ storage, and its output reconstruction solves one dense system on all output points. Both limits bind for three-dimensional turbulence, where a training set has ten thousand or more functions and an output has thousands of points. Meeting the other criteria poses three tasks. The output reconstruction must become local without losing the properties of the global one, the operator regression must become sparse, and the two must be joined without forming the local coeficients of every training field.

The first task draws on radial basis function-generated finite diferences (RBF-FD). RBF-FD methods replace a global radial basis function (RBF) interpolant, whose dense system couples all nodes, by small interpolants on the stencil of each node, so that every linear system has a fixed size and is independent of the others [20, 21, 55, 62, 69]. Over the last two decades, RBF-FD methods have been stabilized for convection-dominated problems [22, 49], applied to the Navier–Stokes equations [18, 55], extended to overlapped stencils, surfaces, and moving domains [48, 51, 52], and formulated in the least-squares sense [63]. They also have error bounds [10], node generators for irregular domains [50], and GPU implementations [6]. In particular, polyharmonic splines (PHS) augmented with polynomials converge at the order of the polynomial degree and have no shape parameter [3, 19]. For flows, the local interpolants must also be divergence free. Divergence-free kernels, obtained by applying $- \Delta I + \nabla \nabla ^ { \mathsf { T } }$ to a scalar kernel [11, 24, 39, 40, 66], have this property by construction and have been used both globally and in partition-of-unity form [12, 25], and Yelnyk [71] combined divergence-free PHS with polynomials in local Hermite interpolation for Stokes problems. However, the high order of PHS-based RBF-FD comes from the polynomials, so a local divergence-free interpolant of high order needs divergence-free polynomials of any degree, in two and three dimensions, that can be evaluated anywhere on the stencil.

Our approach is to build these polynomials by diferentiating products of Legendre polynomials, select an independent subset on each stencil by a column-pivoted QR factorization, and combine them with a divergence-free PHS on the stencils of RBF-FD, used for interpolation instead of diferentiation. Every local interpolant is then exactly divergence free, and building the stencils from periodic images of the nodes makes the output reconstruction periodic. For turbulence, we replace the scale-free PHS by the multiscale Matérn kernel of Sharma et al. [53], whose weights encode the Kolmogorov scaling of the velocity increments, so that the output reconstruction approaches prescribed Kolmogorov statistics as the nodes are refined. Matérn kernels and PHS are Green’s functions of closely related operators, and a Matérn kernel tends to a PHS as its length grows, so this change adds length scales without changing the construction. Since the local systems are independent and of fixed size, we assemble, factor, and evaluate them in batches on the GPU, and our output reconstruction is embarrassingly parallel.

For the second task, the dense kernel matrix of the operator regression can be approximated by low-rank, Nyström, and random-feature methods, inducing points, structured kernel interpolation, or sparse approximate factors of its inverse [1, 13, 17, 26, 44, 47, 56, 57, 68, 72]. In contrast, a compactly supported kernel gives a sparse matrix that can be factored exactly by sparse Cholesky. Its obstacle is the dimension of the input. A sampled input function has thousands of coordinates, but no compactly supported radial kernel is positive definite in every dimension [65]. Our kernel therefore determines its support from a few principal coordinates of the input, in which a compactly supported function is positive definite, and treats the remaining coordinates with a Matérn factor, which is positive definite in every dimension. The product is positive semidefinite in any input dimension. Its support radius sets the sparsity of the kernel matrix, and we choose it from the data instead of fixing it in advance. A restricted likelihood criterion, with a smooth count of the interacting training pairs as an optional penalty, fits the shape of the kernel and the regularization, and the error on validation fields then selects the radius, so that nearly dense matrices remain among the candidates when accuracy requires them.

However, a compactly supported kernel returns to zero away from the training inputs, so it represents trends in the input poorly, although many operators depend nearly linearly on part of their input. We therefore augment the operator regression with afine polynomials in the leading principal coordinates of the input, which it reproduces exactly, as in polynomially augmented kernel interpolation. The resulting saddle-point system is solved by a sparse Cholesky factorization of the kernel matrix followed by a QR factorization of the whitened polynomial columns, which avoids forming the Schur complement [4]. JAX has no sparse direct solver that keeps a factorization between calls, so we factor the kernel matrix on the GPU with cuDSS, through a primitive of our KernelPack-JAX framework that exposes cuDSS to compiled JAX programs [8].

The third task leads to operator Lagrange weights. Since the same kernel regresses every output coeficient, a prediction is a linear combination of the training outputs, with weights that depend only on the new input. These weights cost one sparse solve per new input, regardless of the size of the output, and they give the coeficients on every stencil at once (Figure 1). Because the output reconstruction is linear, the weights can also be applied to the sampled training fields before the output reconstruction, so the local coeficients of the training fields, which would take tens of times the storage of the fields themselves, are never formed. Weights and coeficients give the same prediction in exact arithmetic but round diferently. We show that direct evaluation through coeficients amplifies solve errors by a factor that measures the cancellation among the spatial coeficients, while Lagrange evaluation does not.

We test our method in two steps. On scattered nodes, our local output reconstruction converges at close to orders 2, 4, and 6, keeps the divergence and the periodic mismatch at rounding level, and approaches prescribed Kolmogorov statistics under refinement at the separations that the nodes resolve. Beyond a few thousand output points, it is also cheaper to set up than a global output reconstruction, and its memory grows only linearly with their number. We then compare kernels with a fixed and a fitted shape, each with and without afine augmentation, on ten fluid benchmarks with up to ten thousand training functions. Validation selects sparse kernel matrices on all benchmarks except forced isotropic turbulence and the unaugmented Taylor–Green models, and on most of them the selected density decreases as the training set grows. With these sparse matrices, the test errors decrease steadily with the number of training functions. At the training sizes of our earlier study [53], our models are at least as accurate as dense PPKM and the kernel method of Batlle et al. [2] on every data set, and more accurate than the neural operators reported there. Afine augmentation reduces the test error for forced isotropic turbulence by more than an order of magnitude, and together with the fitted shape it does the same for turbulent species transport. In contrast, the five flows whose training inputs vary along a single direction are insensitive to both choices. Finally, Lagrange evaluation is as accurate as direct evaluation with our local output reconstruction, whose amplification factors are at most about $1 0 ^ { 4 }$ , and up to nearly three orders of magnitude more accurate with global output reconstructions, whose amplification factors reach $6 \times 1 0 ^ { 9 }$

The remainder of this paper is organized as follows. Section 2 constructs the local property-preserving output reconstruction, Section 3 the sparse operator regression and its afine augmentation, and Section 4 the operator Lagrange weights, their rounding errors, and the GPU implementation. Section 5 presents the numerical experiments, and Section 6 our conclusions.

## 1.1. Problem formulation

We consider an incompressible fluid of constant density, governed by

$$
\frac { \partial { \bf u } } { \partial t } + ( { \bf u } \cdot \nabla ) { \bf u } = \nu \nabla ^ { 2 } { \bf u } - \frac { 1 } { \rho } \nabla p + \frac { 1 } { \rho } { \bf f } , \quad \mathrm { i n } \Omega \times ( 0 , T ] ,\tag{1}
$$

$$
\nabla \cdot \mathbf u = 0 , \quad \mathrm { i n } \Omega \times ( 0 , T ] ,\tag{2}
$$

$$
\mathbf { u } ( \mathbf { x } , 0 ) = \mathbf { u } _ { 0 } ( \mathbf { x } ) , \quad \mathrm { i n } \Omega ,\tag{3}
$$

$$
\begin{array} { r } { B \mathbf { u } = \mathbf { g } , \quad { \mathrm { o n ~ } } \partial \Omega \times ( 0 , T ] , } \end{array}\tag{4}
$$

where � is the velocity, � the pressure, � the kinematic viscosity, $\rho$ the density, � the body force, and  the boundary operator. Our operators map initial data (a velocity ${ \bf u } _ { 0 }$ , a vorticity, or, for a buoyancy-driven flow, a temperature), the boundary data �, or the forcing � to the velocity. The training data are simulations of (1) from Sharma et al. [53] (Appendix D.1).

Let  and  be separable Banach spaces of input and output functions, and let $\mathcal { G } : \mathcal { A }  \mathcal { V }$ be the operator we wish to learn. An input $a \in { \mathcal { A } }$ and its output $v \in \mathcal V$ are defined on domains $\Omega _ { a }$ and $\Omega _ { v }$ , which may be spatial, spacetime, or parametric, with

$$
a : \Omega _ { a } \to \mathbb { R } ^ { d _ { a } } , \quad v : \Omega _ { v } \to \mathbb { R } ^ { d } .\tag{5}
$$

Here $d _ { a }$ and $d$ are the numbers of input and output components, and $D _ { a }$ and $D _ { v }$ are the dimensions of the domains. We sample inputs at $X = \{ x _ { 1 } , \dots , x _ { n } \} \subset \Omega _ { a } \subset \mathbb { R } ^ { D _ { a } }$ , which gives vectors � $\in \mathbb { R } ^ { n d _ { a } }$ . For spatial outputs, $D _ { v } = d .$ , and we sample at $Y = \{ y _ { j } \} _ { i = 1 } ^ { m } \subset \Omega _ { v } \subset \mathbb { R } ^ { d }$ , which gives vectors $\mathbf { v } \in \mathbb { R } ^ { m d }$ stacked in component blocks, each holding one velocity component at all points of $Y$ , as in Sharma et al. [53]. For spacetime outputs, $D _ { v } = d + 1$ , and sampling the � spatial sites at $N _ { t }$ times gives vectors in $\mathbb { R } ^ { d m N _ { t } }$ (Section 2.4).

Given � training pairs $\{ ( a _ { i } , v _ { i } ) \} _ { i = 1 } ^ { N }$ with sampled vectors $\{ ( \mathbf { a } _ { i } , \mathbf { v } _ { i } ) \} _ { i = 1 } ^ { N }$ , we write the learned operator, as in Sharma et al. [53], as

$$
v \approx { \tilde { \mathcal { G } } } ( a ) = \chi \circ { \tilde { f } } \circ \varphi ( a ) ,\tag{6}
$$

![](images/eb72b1da9bc98f42da1a5b2e69adbbebe963d82c132af0bee04406ce4b803c21.jpg)  
Figure 1: Overview of our method for a new input $\mathbf { a } _ { * } .$ . The compactly supported kernel � of Section 3 gives a sparse matrix $K _ { U U } ^ { \theta }$ on the sampled training inputs (left and center). One sparse solve with the kernel values $k _ { U } ( \mathbf { a } _ { \ast } )$ gives the operator Lagrange weights $\ell ( { \mathbf { a } } _ { * } )$ of Section 4, shown here without their afine correction. The same weights give the coeficients $\mathbf { b } _ { \ast } ^ { j } = B _ { i } ^ { \top } \boldsymbol { \ell } ( \mathbf { a } _ { \ast } )$ of the local divergence-free interpolant of Section 2 on every stencil $S _ { j }$ of the output (right), where the rows of $B _ { j }$ are the coeficients of the training fields on $S _ { j }$ . In practice, we apply the weights to the sampled training fields before the output reconstruction, which gives the same prediction (Section 4.2).

where $\varphi$ samples the input, the operator regression $\tilde { f }$ maps the samples to the coeficients of the output, and the output reconstruction $\chi$ maps these coeficients to a function in space (or spacetime). This structure follows the formulation of Batlle et al. [2], in which $\tilde { f }$ maps input samples to output samples, with the diference that $\tilde { f }$ predicts the coeficients of a divergence-free interpolant. In PPKM [53], $\chi$ is a global divergence-free interpolant and $\tilde { f }$ a dense operator regression. Here, $\chi$ is a collection of local divergence-free interpolants (Section 2), and $\tilde { f }$ is a sparse operator regression (Section 3). Figure 1 shows how operator Lagrange weights join the two (Section 4).

We use the index � for training functions, � for stencils, � for components and coordinates, � for new (query) inputs and query points, and � for time. Each stencil has $n _ { s }$ nodes. A subscript ∗ marks a quantity of a new input, a superscript ⋆ a quantity at query points, and a hat a computed quantity in the error analysis. The superscript � denotes the transpose, � an identity matrix, and $\| \cdot \| _ { F }$ the Frobenius norm.

## 2. Local property-preserving output reconstruction

We now construct the output reconstruction map $\chi$ of (6). In PPKM [53], $\chi$ is a single divergence-free kernel interpolant on all � output points, so its linear system grows with the output resolution. We replace it by local interpolants, one on the stencil of each output point. Each local interpolant combines a divergence-free kernel with divergence-free polynomials, so that it is divergence free for any coeficient vector and reproduces low-degree divergence-free fields. Its system has a fixed size as � grows. We first describe the kernel, the polynomials, and the local system, and then the extensions to turbulent fields, periodic domains, and spacetime outputs. Nothing in this section depends on the operator regression of Section 3.

## 2.1. Divergence-free interpolation on stencils

## 2.1.1. The divergence-free kernel

The output points � serve as the nodes of the output reconstruction. For each node $y _ { j } ~ \in ~ Y$ , let the stencil $S _ { j } ~ = ~ \{ \tilde { y } _ { j , 1 } , \dots , \tilde { y } _ { j , n _ { s } } \}$ consist of its $n _ { s }$ nearest nodes. In the tests of Section 5.1 on bounded domains, we instead use the boundary-aware stencils of KernelPack-JAX [8], which keep only nodes visible from the center inside the domain. In periodic domains, some of these are translated images of nodes (Section 2.3). For smooth outputs, we use the cubic PHS $\phi ( x , y ) = \phi _ { 3 } ( \| x - y \| _ { 2 } )$ , with radial function

$$
\phi _ { 3 } ( r ) = r ^ { 3 } ,\tag{7}
$$

which has no shape parameter to choose (Section 2.2 relates it to the Matérn kernels that we use when a length scale is wanted). For the sixth-order tests of Section 5.1, we use $\phi ( x , y ) = \| x - y \| _ { \gamma } ^ { 5 }$ in the same way. Applying the operator $- \Delta I + \nabla \nabla ^ { \mathsf { T } }$ to a scalar kernel gives a matrix-valued kernel whose columns are divergence free [25, 39],

$$
\Phi _ { \mathrm { D F } } ( x , y ) = \left( - \Delta _ { x } I + \nabla _ { x } \nabla _ { x } ^ { \top } \right) \phi ( x , y ) .\tag{8}
$$

For the cubic PHS, with $z = x - y$ and $r = \| z \| _ { 2 } > 0$

$$
\Phi _ { \mathrm { D F } } ( x , y ) = - 3 d r I + 3 z z ^ { \top } / r , \qquad \Phi _ { \mathrm { D F } } ( y , y ) = 0 .\tag{9}
$$

Each column of $\Phi _ { \mathrm { D F } }$ has zero divergence in the distributional sense, and zero classical divergence away from its center [25]. At the center the kernel is continuous but not diferentiable. For example, along $z = t \mathbf { e } _ { k }$ , with ${ \bf e } _ { k }$ the �th coordinate vector, its �th diagonal entry is $- 3 ( d - 1 ) | t |$

## 2.1.2. Divergence-free polynomials of any degree

A PHS interpolant converges at the order set by the degree of its polynomial augmentation [3, 19]. A divergencefree local interpolant of high order therefore needs divergence-free vector polynomials of any degree �, in 2D and 3D, that can be evaluated at any point and not only at the stencil nodes. Our approach is to diferentiate scalar potentials. On $S _ { j }$ , let $\xi = ( y - \bar { y } _ { j } ) / { \rho _ { j } }$ be centered and scaled coordinates, where $\begin{array} { r } { \bar { y } _ { j } = n _ { s } ^ { - 1 } \sum _ { l } \tilde { y } _ { j , l } } \end{array}$ is the centroid of the stencil and $\rho _ { j } = \mathrm { m a x } _ { l }$ ${ \| \tilde { y } _ { j , l } - \bar { y } _ { j } \| _ { 2 } }$ its radius. From the products $\begin{array} { r } { J _ { \alpha } ( \xi ) = \prod _ { k } P _ { \alpha _ { k } } ( \xi _ { k } ) } \end{array}$ of normalized Legendre polynomials $P _ { n } ,$ of total degree $| \alpha | \leq \dot { \varpi } + 1$ , we form the rotated gradients in 2D and the curls of the potentials times the coordinate vectors in 3D,

$$
\begin{array} { r l } & { ~ g _ { \alpha } ( \boldsymbol { y } ) = \nabla _ { \boldsymbol { \xi } } ^ { \perp } J _ { \alpha } ( \boldsymbol { \xi } ) , } \\ & { ~ g _ { \alpha , k } ( \boldsymbol { y } ) = \nabla _ { \boldsymbol { \xi } } \times \big ( J _ { \alpha } ( \boldsymbol { \xi } ) \mathbf { e } _ { k } \big ) , } \end{array}
$$

$$
d = 2 ,
$$

$$
d = 3 , \quad k = 1 , 2 , 3 ,\tag{10}
$$

(11)

where � is a multi-index, $\nabla _ { \boldsymbol { \xi } } ^ { \perp } = ( \partial _ { \xi _ { 2 } } , - \partial _ { \xi _ { 1 } } ) ^ { \top }$ . These candidates have degree at most � and are divergence free in $\xi ,$ and hence in $y ,$ since the scaling is uniform. Together, they span the space of divergence-free vector polynomials of degree at most �, which has dimension $d \binom { \varpi + d } { d } - \binom { \varpi + d - 1 } { d }$ (Appendix C.1).

The candidates are not independent. Constant potentials give zero fields, diferent potentials can give the same field, and a particular stencil can add further dependencies. Let $g _ { 1 } , \ldots , g _ { n _ { \mathrm { c a n d } , j } }$ be the candidates, let $G _ { j } ( y ) \ =$ $[ g _ { 1 } ( y ) ~ \cdots ~ g _ { n _ { \mathrm { c a n d } . i } } ( y ) ] \in \mathbb { R } ^ { d \times n _ { \mathrm { c a n d } , j } }$ hold their values at $y ,$ and let $G _ { j } \in \mathbb { R } ^ { d n _ { s } \times n _ { \mathrm { c a n d } , j } }$ hold their values at the stencil nodes in component block order. We select an independent subset by column-pivoted QR,

$$
G _ { j } \Pi _ { j } = Q _ { j } R _ { j } ,\tag{12}
$$

and retain the first $n _ { P , j }$ pivoted columns, where $n _ { P , j }$ counts the diagonal entries of $R _ { j }$ larger than $1 0 ^ { - 1 2 }$ times the first. Let $I _ { j }$ hold the indices of the retained candidates, $G _ { j , I }$ the corresponding columns of $G _ { j } , Q _ { j , 1 }$ the first $n _ { P , j }$ columns of $Q _ { j } .$ , and $R _ { j , 1 1 } \in \mathbb { R } ^ { n _ { P , j } \times n _ { P , j } }$ the leading block of $R _ { j }$ . Then

$$
G _ { j , I } = Q _ { j , 1 } R _ { j , 1 1 } , \qquad G _ { j , I } R _ { j , 1 1 } ^ { - 1 } = Q _ { j , 1 } .\tag{13}
$$

The orthonormal columns of $Q _ { j , 1 }$ are only values at the nodes. To obtain functions, we apply the same triangular transformation to the retained candidate functions $G _ { j , I } ( y ) = [ g _ { c } ( y ) ] _ { c \in I _ { i } }$

$$
{ \bf P } _ { j } ( y ) = G _ { j , I } ( y ) R _ { j , 1 1 } ^ { - 1 } .\tag{14}
$$

By (13), $\mathbf { P } _ { j } = Q _ { j , 1 }$ at the stencil nodes, so the polynomial block of the local system is well conditioned. Every column of ${ \bf P } _ { j } ( y )$ is a fixed combination of divergence-free fields, and is thus divergence free everywhere. At a new point, we

evaluate the retained candidates and apply $R _ { j , 1 1 } ^ { - 1 }$ by a triangular solve. If the stencil geometry forces the rank below the full dimension, the interpolant reproduces only the retained subspace.

We batch this construction over the stencils. On stencils that are certainly of full rank, pivoting is unnecessary, and we use an unpivoted QR factorization instead (Appendix A.2).

## 2.1.3. The local system

On $S _ { j } ,$ we determine the kernel and polynomial coeficients together, from the interpolation conditions and the polynomial moment conditions. The saddle-point matrix

$$
\mathbf { A } _ { j } = \left[ \mathbf { \overline { { \mathbf { P } } } } _ { j } \mathbf { \overline { { \mathbf { \Gamma } } } } _ { j } \right] \in \mathbb { R } ^ { r _ { j } \times r _ { j } } ,\tag{15}
$$

has size $r _ { j } = d n _ { s } + n _ { P , j }$ , where $\Phi _ { j } \in \mathbb { R } ^ { d n _ { s } \times d n _ { s } }$ � is the Gram matrix of the divergence-free kernel on $S _ { j }$ . For training field �, the coeficient vector

$$
\mathbf { b } _ { i } ^ { j } = \left[ \mathbf { b } _ { i , \Phi } ^ { j } \right] \in \mathbb { R } ^ { r _ { j } }
$$

holds kernel coeficients $\mathbf { b } _ { i , \Phi } ^ { j } \in \mathbb { R } ^ { d n _ { s } }$ and polynomial coeficients ${ \bf b } _ { i , P } ^ { j } \in \mathbb { R } ^ { n _ { P , j } }$ , and defines the local interpolant

$$
\chi _ { i } ^ { j } ( y ) = \sum _ { l = 1 } ^ { n _ { s } } \Phi _ { \mathrm { D F } } ( y , \tilde { y } _ { j , l } ) \mathbf { b } _ { i , \Phi } ^ { j , l } + \mathbf { P } _ { j } ( y ) \mathbf { b } _ { i , P } ^ { j } ,\tag{16}
$$

where $\mathbf { b } _ { i , \Phi } ^ { j , l } \in \mathbb { R } ^ { d }$ holds the kernel coeficients of node �. Interpolation of the training field at the stencil nodes (with the value of the original node at a periodic image) and orthogonality of the kernel coeficients to the polynomials give

$$
\mathbf { A } _ { j } \mathbf { b } _ { i } ^ { j } = \left[ { \mathbf { \bar { \rho } } } _ { 0 } ^ { \phantom { \dagger } } \right] ,\tag{17}
$$

where $\mathbf { v } _ { i } \big | _ { S _ { i } } \in \mathbb { R } ^ { d n _ { s } }$ � holds the sampled velocities on the stencil. Since ${ \bf P } _ { j }$ has full column rank, $\mathbf { A } _ { j }$ is nonsingular whenever $\Phi _ { j }$ is positive definite on the null space of $\mathbf { P } _ { i } ^ { \top }$ . Conditional positive definiteness of the scalar PHS does not guarantee this for the matrix kernel on a particular stencil, so we check the solves numerically (Appendix A.2). We factor all local systems by batched LU on the GPU.

Evaluation To reconstruct a prediction at query points $y _ { q } ^ { \star } \in Y ^ { \star } , q = 1 , \ldots , m ^ { \star }$ , we assign each query point to a stencil $S _ { j ( q ) }$ and evaluate that stencil’s basis there. Let $\Phi _ { j ( q ) , q , \mathrm { c o m p } } ^ { \star } \in \mathbb { R } ^ { d \times d n _ { s } }$ hold the kernel values in the component block order of the coeficients (Appendix A.1) and $\mathbf { P } _ { j ( q ) , q } ^ { \star } \in \mathbb { R } ^ { d \times n _ { P , j ( q ) } }$ the polynomial values, and let

$$
\mathbf { A } _ { q } ^ { \star } = \left[ \Phi _ { j ( q ) , q , \mathrm { c o m p } } ^ { \star } \quad \mathbf { P } _ { j ( q ) , q } ^ { \star } \right] \in \mathbb { R } ^ { d \times r _ { j ( q ) } } .\tag{18}
$$

For coeficients $\mathbf { b } _ { * } ^ { j }$ predicted by the operator regression (Section 4.1), the reconstructed velocity is then

$$
\tilde { \mathbf { v } } _ { q } ^ { \star } = \mathbf { A } _ { q } ^ { \star } \mathbf { b } _ { \ast } ^ { j ( q ) } \in \mathbb { R } ^ { d } ,\tag{19}
$$

and $\tilde { \mathbf { v } } ^ { \star }$ stacks the $\tilde { \mathbf { v } } _ { q } ^ { \star }$ in component blocks. In the operator benchmarks, the query points are the nodes themselves, each with its own stencil. Appendix A.3 gives the assignment of other points.

The output reconstruction is divergence free on each stencil, for any coeficients. However, the local interpolants are not coupled, so the reconstructed field can jump where the assigned stencil changes, and its divergence is controlled only within each stencil.

On the other hand, every system has the fixed size $r _ { j } .$ , independent of the number of output points � and of the number of training functions �. Refining the output adds systems but does not enlarge them. The systems are also independent of each other, so their assembly, factorization, and evaluation are embarrassingly parallel, and we process them in batches on the GPU with no communication between stencils. In contrast, a global output reconstruction couples all output points in one dense system with �� unknowns, whose factorization costs $O ( ( d m ) ^ { 3 } )$ operations and does not decompose into independent parts. Section 5.1.5 measures both costs. Section 4.2 shows how the same systems serve every prediction.

## 2.2. Multiscale Matérn output reconstruction for turbulence

Turbulent velocity fields vary over many length scales, which a kernel can represent only if it has length scales of its own. Polyharmonic splines have none, since they are homogeneous (dilating the argument of $r ^ { 3 }$ or $r ^ { 5 }$ multiplies it by a constant, and so does the derived matrix kernel), and sums of dilated copies add nothing. Matérn kernels have one length each, and they are close relatives of the PHS. The Matérn kernels $M _ { 3 / 2 } ( r / \sigma )$ and $M _ { 5 / 2 } ( r / \sigma )$ of length $\sigma ,$ with $M _ { 3 / 2 } ( r ) = ( 1 + r ) e ^ { - r }$ and $M _ { 5 / 2 }$ given in (22) below, have Fourier transforms proportional to $( \sigma ^ { - 2 } + \Vert \mathbf { k } \Vert _ { , } ^ { 2 } ) ^ { - ( 3 + d ) / 2 }$ and $( \sigma ^ { - 2 } + \| \mathbf { k } \| _ { , } ^ { 2 } ) ^ { - ( 5 + d ) / 2 }$ in the wave vector �. They are therefore, up to constants, the Green’s functions of $( \bar { \sigma ^ { - 2 } } - \Delta ) ^ { ( 3 + d ) / 2 }$ and $( \sigma ^ { - 2 } - \mathbf { \bar { \Delta } } ) ^ { ( 5 + d ) / 2 } \sin \mathbb { R } ^ { d } [ 1 6 , 3 5 , 5 8 , 6 7 ]$ . Letting $\sigma $ ∞ removes the length and leaves $( - \Delta ) ^ { ( 3 + d ) / 2 }$ and $( - \Delta ) ^ { ( 5 + d ) / 2 }$ whose fundamental solutions are multiples of $r ^ { 3 }$ and $r ^ { 5 } \ [ 1 4 , 6 5 ]$ . The two Matérn kernels are thus the counterparts, with a length, of the PHS $r ^ { 3 }$ and $r ^ { 5 }$ . For example,

$$
M _ { 5 / 2 } \bigg ( \frac { r } { \sigma } \bigg ) = 1 - \frac { r ^ { 2 } } { 6 \sigma ^ { 2 } } + \frac { r ^ { 4 } } { 2 4 \sigma ^ { 4 } } - \frac { r ^ { 5 } } { 4 5 \sigma ^ { 5 } } + { \cal O } \bigg ( \frac { r ^ { 6 } } { \sigma ^ { 6 } } \bigg ) .\tag{20}
$$

The even terms are polynomials in $x - y$ , and after the operator $- \Delta I + \nabla \nabla ^ { \mathsf { T } }$ their contribution to an interpolant is a divergence-free polynomial of degree at most two, which the polynomial part of the local interpolant absorbs when $\varpi \ge 2$ . Since rescaling the kernel does not change the interpolant, the local interpolant with $M _ { 5 / 2 } ( r / \sigma )$ thus tends to the one with $r ^ { 5 } \mathrm { a s } \sigma \to \infty$ . We use the PHS when no length scale is wanted, for smooth outputs whose accuracy is set by the polynomial degree, and Matérn kernels when lengths are wanted. A mixture of Matérn kernels can have many lengths. This motivates localizing the multiscale Matérn construction of our earlier work [53], which follows Owhadi [41]. On $S _ { j }$ , we sum five Matérn kernels with dyadic lengths and weights

$$
\sigma _ { j , \kappa } = \frac { \sigma _ { j , 0 } } { 2 ^ { \kappa } } , \qquad \sigma _ { j , 0 } = c _ { \sigma } \rho _ { j } , \qquad \vartheta _ { j , \kappa } = \left( \frac { \sigma _ { j , \kappa } } { \sigma _ { j , 0 } } \right) ^ { 8 / 3 } , \qquad \kappa = 1 , \ldots , 5 ,\tag{21}
$$

where $\rho _ { j }$ is the stencil radius and $c _ { \sigma }$ a fixed factor. Each scale uses the scalar Matérn function

$$
M _ { 5 / 2 } ( r ) = \left( 1 + r + \frac { r ^ { 2 } } { 3 } \right) e ^ { - r }\tag{22}
$$

at the argument $r / \sigma _ { j , \kappa }$ . The weights are applied to the scalar kernels, before the divergence-free operator, whose two derivatives account for the 2 in the exponent $8 / 3 = 2 + 2 / 3$ . The remaining $2 / 3$ is the Kolmogorov exponent of the velocity increments in three dimensions. The local multiscale kernel is

$$
\Phi _ { j } ^ { \mathbf { M } } ( { x } , { y } ) = \frac { 1 } { c _ { j } } \left( - \Delta _ { x } I + \nabla _ { x } \nabla _ { x } ^ { \mathsf { T } } \right) \sum _ { \kappa = 1 } ^ { 5 } 3 \vartheta _ { j , \kappa } M _ { 5 / 2 } \left( \frac { \| x - y \| _ { 2 } } { \sigma _ { j , \kappa } } \right) , \qquad c _ { j } = ( d - 1 ) \sum _ { \kappa = 1 } ^ { 5 } \vartheta _ { j , \kappa } \sigma _ { j , \kappa } ^ { - 2 } ,\tag{23}
$$

normalized so that $\Phi _ { i } ^ { \mathbf { M } } ( y , y ) = I$ . The normalization rescales the coeficients and does not change the interpolant. We replace $\Phi _ { \mathrm { D F } }$ by $\dot { \Phi } _ { i } ^ { \mathrm { M } }$ in (15) and (18) and keep the polynomials, so each stencil still solves one system with $r _ { j }$ unknowns for the whole mixture. Unlike the cubic kernel, the mixture is smooth at its centers, so the reconstructed fields are classically divergence free everywhere on the stencil.

The weights build the prescribed increment statistics into the kernel [41]. Whether an interpolant on finite stencils reproduces those statistics also depends on the stencil extent and the node spacing, and we test this under refinement in Section 5.1 and for the turbulent benchmarks in Appendix A.5.

## 2.3. Periodic stencils

In a periodic domain, the nearest neighbors of a node near one face of the box can lie near the opposite face. Our stencils are therefore built from translated images of the nodes. If coordinate � has period $\Lambda _ { k }$ , a node $y _ { l }$ has images

![](images/f6a7a970eaae1897ec167a11e5ce792a0fdd7378ed040311b103f1b47af1d504.jpg)  
Figure 2: Periodic stencils in a box with periods $\Lambda _ { 1 }$ and $\Lambda _ { 2 }$ . Gray dots are nodes (dark) and their translated images (light). A stencil near a face (blue) and a stencil near a corner (red) are built from the nearest images given by (25). Dashed arrows link each image in a stencil to its node (circled), whose sampled velocity it carries.

$\begin{array} { r } { y _ { l } + \sum _ { k } \nu _ { k } \Lambda _ { k } \mathbf { e } _ { k } } \end{array}$ , with integers $\nu _ { k }$ over the periodic coordinates, which carry its index and its sampled velocity,

$$
v _ { i } \biggl ( y _ { l } + \sum _ { k } \nu _ { k } \Lambda _ { k } \mathbf { e } _ { k } \biggr ) = v _ { i } ( y _ { l } ) .\tag{24}
$$

For a stencil center $y _ { j }$ , we take the displacements $\delta _ { l } \ : = \ : y _ { l } - y _ { j }$ and shift every periodic component to its nearest representative,

$$
\delta _ { l , k } \gets \delta _ { l , k } - \Lambda _ { k } \mathrm { r o u n d } ( \delta _ { l , k } / \Lambda _ { k } ) .\tag{25}
$$

Since all periodic coordinates are shifted together, this also finds the right images near the edges and corners of the box, where a node’s neighbors are shifted in two or three coordinates (Figure 2). The stencil consists of the $n _ { s }$ nearest of the points $y _ { j } + \delta _ { l }$ , with one representative of each node, and the kernel and polynomials are evaluated in these translated coordinates. Translation does not change the divergence. Appendix A.4 gives the periods used for each benchmark.

## 2.4. Spacetime outputs

For outputs on $\Omega _ { v } = \Omega \times \Gamma$ , with Γ a time interval, PPKM [53] used a global product kernel

$$
\Phi _ { \mathrm { S T } } ( ( y , t ) , ( y ^ { \prime } , t ^ { \prime } ) ) = \Phi _ { \mathrm { D F } } ^ { \mathrm { M a t } } ( y , y ^ { \prime } ) \Phi _ { t } ( t , t ^ { \prime } ) ,\tag{26}
$$

with a divergence-free Matérn kernel in space and a scalar Matérn kernel $\Phi _ { t }$ in time, and ��� coeficients in one system. We instead keep each stencil and its spatial basis fixed over the sampled times $\boldsymbol { \mathcal { T } } = \{ t _ { s } \} _ { s = 1 } ^ { N _ { t } }$ and interpolate in time. Let $\mathbf { b } _ { i } ^ { j , s } \in \mathbb { R } ^ { r _ { j } }$ be the coeficients of training field � on $S _ { j }$ at time $t _ { s }$ . Every snapshot solves (17) with the same matrix $\mathbf { A } _ { j } .$ , and the target of the operator regression is the stacked vector

$$
\mathbf { b } _ { i , \mathcal { T } } ^ { j } = [ ( \mathbf { b } _ { i } ^ { j , 1 } ) ^ { \top } , \ldots , ( \mathbf { b } _ { i } ^ { j , N _ { t } } ) ^ { \top } ] ^ { \top } \in \mathbb { R } ^ { N _ { t } r _ { j } } .\tag{27}
$$

Between the sampled times, we interpolate with the Lagrange polynomials

$$
\omega _ { s } ( t ) = \prod _ { s ^ { \prime } \ne s } \frac { t - t _ { s ^ { \prime } } } { t _ { s } - t _ { s ^ { \prime } } } , \qquad s = 1 , \dots , N _ { t } ,\tag{28}
$$

which need no temporal kernel parameter and no further linear system. With $A _ { j } ^ { \star } ( y )$ the evaluation of the spatial basis on $S _ { j }$ as in (18), the spacetime output reconstruction of a prediction is

$$
\chi _ { * } ^ { j } ( y , t ) = \sum _ { s = 1 } ^ { N _ { t } } \omega _ { s } ( t ) \mathbf { A } _ { j } ^ { \star } ( y ) \mathbf { b } _ { * } ^ { j , s } .\tag{29}
$$

Its polynomial part is the tensor product of the retained spatial polynomials with polynomials of degree at most $N _ { t } - 1$ in time. Since the weights $\omega _ { s }$ do not depend on $y ,$ the spacetime output reconstruction is divergence free wherever the spatial one is. The local representation has $N _ { t } \sum _ { j } r _ { j }$ coeficients, but each system keeps the size $r _ { j }$ . Another sampled time adds only a right-hand side.

## 3. Sparse operator regression

Having fixed the output representation, we now learn its coeficients as functions of the input. Each output coeficient is a function on the space of sampled inputs, and we approximate all of them by kernel regression with the same kernel. PPKM [53] used a kernel with global support, whose $N \times N$ kernel matrix is dense, so that its Cholesky factorization costs $O ( N ^ { 3 } )$ operations and $O ( N ^ { 2 } )$ storage for � training functions. We now construct a compactly supported kernel that is positive semidefinite on inputs of any dimension, so that the kernel matrix is sparse. The rest of the section defines the operator interpolant built from this kernel and its afine augmentation, derives their linear systems, factors them, and chooses the kernel parameters.

## 3.1. A compactly supported kernel in any input dimension

To make the kernel matrix sparse, the kernel must vanish for inputs that are far apart, and it must remain positive definite on the space of sampled inputs, whose dimension $n d _ { a }$ can be in the thousands. These two requirements conflict for radial kernels. By Schoenberg’s theorem, a radial function that is positive definite in every dimension is a scale mixture of Gaussians and therefore never compactly supported [65]. Every compactly supported radial kernel [27, 64] is thus positive definite only up to some dimension $D _ { \mathrm { m a x } }$ . For the Wendland functions, the polynomial degree grows linearly with $D _ { \mathrm { m a x } }$ at fixed smoothness [64], so a function valid in thousands of dimensions would have a degree in the thousands. Tapering [23], which multiplies a globally supported kernel by a compactly supported one, inherits the same restriction from its taper. Low numerical rank of the training inputs does not remove the restriction either, since a new input need not lie in their span.

Our approach is to separate the two requirements. We let a low-dimensional projection of the input determine the support, and use a compactly supported radial function only there, in a dimension in which it is positive definite. The remaining coordinates enter through a Matérn factor, which is positive definite in every dimension but not compactly supported. The product is positive semidefinite in any input dimension and has the support of its first factor. In this paper, we use the $C ^ { 4 }$ Wendland function

$$
w ( t ) = ( 1 - t ) _ { + } ^ { 6 } \left( 1 + 6 t + \frac { 3 5 } { 3 } t ^ { 2 } \right) ,\tag{30}
$$

which is positive definite in up to three dimensions [64], so we project onto at most three coordinates. Another member of the Wendland family would allow a projection of higher dimension.

Let �̄ be the sampled input after centering and scaling each coordinate by its training mean and standard deviation (coordinates with zero training variance are set to zero). Let the columns of $V _ { z }$ be the first $n _ { z }$ principal directions of the standardized training inputs, where $n _ { z }$ is the smaller of three and their numerical rank (Appendix B.1), and split

$$
z ( \mathbf { a } ) = V _ { z } ^ { \mathsf { T } } \bar { \mathbf { a } } , \qquad \zeta ( \mathbf { a } ) = ( I - V _ { z } V _ { z } ^ { \mathsf { T } } ) \bar { \mathbf { a } } .\tag{31}
$$

For a support radius $\varrho > 0$ , a residual scale ${ \alpha _ { \perp } } > 0$ , and a positive diagonal matrix $D _ { \parallel }$ , our kernel is

$$
\lambda ( { \bf a } , { \bf a } ^ { \prime } ) = \eta ( { \bf a } ) \eta ( { \bf a } ^ { \prime } ) w \left( \frac { \| D _ { \| } ( z ( { \bf a } ) - z ( { \bf a } ^ { \prime } ) ) \| _ { 2 } } { \varrho } \right) M _ { 5 / 2 } \left( \frac { \alpha _ { \bot } \| \zeta ( { \bf a } ) - \zeta ( { \bf a } ^ { \prime } ) \| _ { 2 } } { \delta } \right) .\tag{32}
$$

Here $M _ { 5 / 2 }$ is the Matérn function (22), $\delta > 0$ is a residual length, and $\eta$ is a gate. The Wendland factor sets the support, since two inputs interact only if their principal coordinates are closer than $\varrho$ in the metric $D _ { \parallel }$ . The Matérn factor separates inputs with the same principal coordinates. Note that the kernel is not radial in the full input.

The gate switches the kernel of for inputs whose residual �(�) is much larger than those of the training inputs. With $\zeta _ { i } = \zeta ( \mathbf { a } _ { i } )$ , we set

$$
\delta = \operatorname* { m a x } \left\{ 0 . 0 1 \operatorname* { m a x } _ { i } \| \bar { \mathbf { a } } _ { i } \| _ { 2 } , 2 . 2 \operatorname* { m a x } _ { i } \| \zeta _ { i } \| _ { 2 } , 1 0 ^ { - 8 } \right\} ,\tag{33}
$$

and, with the polynomial $H ( t ) = 1 2 6 t ^ { 5 } - 4 2 0 t ^ { 6 } + 5 4 0 t ^ { 7 } - 3 1 5 t ^ { 8 } + 7 0 t ^ { 9 }$ , which increases from $H ( 0 ) = 0$ to $H ( 1 ) = 1$ since $H ^ { \prime } ( t ) = 6 3 0 t ^ { 4 } ( 1 - t ) ^ { 4 }$

$$
\eta ( \mathbf { a } ) = \hat { \eta } ( \Vert \zeta ( \mathbf { a } ) \Vert _ { 2 } / \delta ) , \qquad \hat { \eta } ( r ) = \left\{ \begin{array} { l l } { 1 , } & { 0 \leq r \leq 1 / 2 , } \\ { H ( 2 ( 1 - r ) ) , } & { 1 / 2 < r < 1 , } \\ { 0 , } & { r \geq 1 . } \end{array} \right.\tag{34}
$$

The gate is $C ^ { 4 }$ and positive for $\| \zeta ( \mathbf { a } ) \| _ { 2 } < \delta .$ Since $\| \zeta _ { i } \| _ { 2 } \leq \delta / 2 . 2$ , it equals one on every training input. A new input need not lie in the span of the principal directions $V _ { z }$ , and its gate decreases to zero as its residual grows.

Proposition 1 (Positive definiteness in any input dimension). For any finite sampled-input dimension, the kernel (32) is positive semidefinite. Its Gram matrix is strictly positive definite on distinct standardized inputs with positive gates. Consequently, adding a positive multiple ofthe identity gives a positive-definite kernel matrix, including when sampled inputs coincide or gates vanish.

Proofsketch. Both factors are positive definite functions composed with linear maps, and products and gate multiplications preserve positive semidefiniteness. Distinct inputs difer in their principal or their residual coordinates. Appendix C.2 gives the proof. □

The proposition holds for any orthogonal projection, so it needs no estimate of an intrinsic dimension. The principal components only choose the directions along which the support is cut of.

Fixed andfitted shapes We compare two choices of the shape parameters. The fixed-shape kernel uses $D _ { \parallel } = I$ and $\alpha _ { \perp } = 1$ . The fitted-shape kernel also fits $\alpha _ { \perp }$ and $D _ { \parallel } = \mathrm { d i a g } ( e ^ { \upsilon _ { 1 } } , \ldots , e ^ { \upsilon _ { n _ { z } } } )$ , with $\begin{array} { r } { \sum _ { k = 1 } ^ { n _ { z } } { \upsilon _ { k } } = 0 } \end{array}$ so that � alone sets the overall support and the $\upsilon _ { k }$ only redistribute it among the principal directions. Note that when $n _ { z } = 1$ the constraint forces $D _ { \parallel } = [ 1 ]$ , and when the residual coordinates are negligible we also keep ${ \alpha _ { \perp } } = 1$ . The fitted-shape kernel then has the shape of the fixed-shape one and difers from it only through its radius and regularization (Section 5.2.1 lists these cases).

## 3.2. The operator interpolant

We now use the kernel � to approximate the map from a sampled input to the local coeficients of its output. Let $\mathbf { b } _ { i } \in \mathbb { R } ^ { m _ { \mathrm { r h s } } }$ stack the local coeficients $\mathbf { b } _ { i } ^ { j }$ of training field � on all stencils (and at all sampled times), where $\begin{array} { r } { m _ { \mathrm { r h s } } = \sum _ { j } r _ { j } } \end{array}$ (times $N _ { t }$ for spacetime outputs). Every one of these coeficients is a function of the same input, so we approximate all of them with the same scalar kernel, i.e., with the matrix-valued kernel $\lambda ( \mathbf { a } , \mathbf { a } ^ { \prime } ) I$ . The operator regression $\tilde { f }$ of (6) is the operator interpolant

$$
\tilde { f } ( \mathbf { a } ) = \sum _ { i = 1 } ^ { N } \lambda ( \mathbf { a } , \mathbf { a } _ { i } ) \mathbf { c } _ { i } ,\tag{35}
$$

with coeficients $\mathbf { c } _ { i } \in \mathbb { R } ^ { m _ { 1 } }$ rhs determined by the regularized interpolation conditions

$$
\tilde { f } ( \mathbf { a } _ { i } ) + \theta \mathbf { c } _ { i } = \mathbf { b } _ { i } , \qquad i = 1 , \ldots , N ,\tag{36}
$$

where $\theta > 0$ is a regularization parameter. In the limit $\theta \to 0 , \tilde { f }$ interpolates the training coeficients, and for $\theta > 0$ it is their kernel ridge regression. Since $\chi$ is linear, the learned operator (6) is

$$
\tilde { \mathcal { G } } ( a ) = \sum _ { i = 1 } ^ { N } \lambda ( \mathbf { a } , \mathbf { a } _ { i } ) \chi ( \mathbf { c } _ { i } ) ,\tag{37}
$$

a combination of � fixed fields with weights that depend on the input. Every one of these fields is a local divergencefree output reconstruction, so $\tilde { \mathcal { G } } ( a )$ is divergence free on each stencil for every input, regardless of the accuracy of the operator regression.

## 3.3. Afine augmentation

Operators often depend nearly linearly on part of their input. A compactly supported kernel regression represents such a trend poorly, since it returns to zero away from the training inputs. As in polynomially augmented interpolation [4], we augment the operator regression with afine functions of up to 15 principal coordinates of the input (more than the at most three that set the support), which it reproduces exactly, and let the kernel represent the remainder. These polynomials in the input are unrelated to the divergence-free polynomials in space of Section 2, which are part of the output representation in every model.

Let the columns of $V _ { p }$ be up to 15 principal directions of the training inputs, let $\varsigma _ { k }$ be the training standard deviation along direction $k ,$ and set

$$
p ( \mathbf { a } ) = \left( 1 , \frac { ( V _ { p } ^ { \top } \bar { \mathbf { a } } ) _ { 1 } } { \varsigma _ { 1 } } , \ldots , \frac { ( V _ { p } ^ { \top } \bar { \mathbf { a } } ) _ { n _ { \mathrm { a f f } } - 1 } } { \varsigma _ { n _ { \mathrm { a f f } } - 1 } } \right) ^ { \top } ,\tag{38}
$$

with $n _ { \mathrm { a f f } } - 1 \leq 1 5$ limited by the rank of the training inputs. The augmented operator interpolant is

$$
\widetilde { f } ( \mathbf { a } ) = \sum _ { i = 1 } ^ { N } \lambda ( \mathbf { a } , \mathbf { a } _ { i } ) \mathbf { c } _ { i } + \beta ^ { \top } p ( \mathbf { a } ) ,\tag{39}
$$

with kernel coeficients $\mathbf { c } _ { i } ~ \in ~ \mathbb { R } ^ { m _ { \mathrm { r h s } } }$ and afine coeficients $\beta ~ \in ~ \mathbb { R } ^ { n _ { \mathrm { a f f } } \times m _ { \mathrm { r h s } } }$ . Its coeficients are determined by the regularized interpolation conditions of (36), now for the augmented ${ \tilde { f } } ,$ together with moment conditions on the kernel coeficients,

$$
\tilde { f } ( { \mathbf a } _ { i } ) + \theta { \mathbf c } _ { i } = { \mathbf b } _ { i } , \quad i = 1 , \ldots , N , \qquad \sum _ { i = 1 } ^ { N } p ( { \mathbf a } _ { i } ) { \mathbf c } _ { i } ^ { \top } = 0 .\tag{40}
$$

The moment conditions make (39) reproduce every afine function of the principal coordinates exactly, for any �. If $\mathbf b _ { i } = T ^ { \top } p ( \mathbf a _ { i } )$ for some matrix $T$ , then $\mathbf { c } _ { i } = 0$ and $\beta = T$ satisfy (40). As in (37), the augmented learned operator is

$$
\tilde { \mathcal { G } } ( a ) = \sum _ { i = 1 } ^ { N } \lambda ( \mathbf { a } , \mathbf { a } _ { i } ) \chi ( \mathbf { c } _ { i } ) + \sum _ { k = 1 } ^ { n _ { \mathrm { a f f } } } p _ { k } ( \mathbf { a } ) \chi ( \pmb { \beta } _ { k , : } ) ,\tag{41}
$$

a combination of $N + n _ { \mathrm { a f f } }$ fixed divergence-free fields. Without augmentation, $n _ { \mathrm { a f f } } = 0$ , the afine terms and the moment conditions disappear, and (39)–(41) reduce to (35)–(37).

Both interpolants have a variational characterization. The function $\tilde { f } = f _ { \mathcal { H } } + \beta ^ { \top } p$ minimizes

$$
\sum _ { i = 1 } ^ { N } \left\| f _ { \mathcal { H } } ( \mathbf { a } _ { i } ) + \beta ^ { \top } p ( \mathbf { a } _ { i } ) - \mathbf { b } _ { i } \right\| _ { 2 } ^ { 2 } + \theta \| f _ { \mathcal { H } } \| _ { \mathcal { H } } ^ { 2 }\tag{42}
$$

over the afine coeficients $\beta$ and the functions $f _ { \mathcal { H } }$ in the reproducing kernel Hilbert space  of $\lambda ( \mathbf { a } , \mathbf { a } ^ { \prime } ) I$ , since by the representer theorem the minimizing $f _ { \mathcal { H } }$ has the form $\begin{array} { r } { \sum _ { i } \lambda ( \cdot , { \bf a } _ { i } ) { \bf c } _ { i } } \end{array}$ , and (40) makes the gradient of the convex objective vanish (Appendix C.3). Without augmentation, $\beta$ is absent and (42) is the objective of kernel ridge regression.

## 3.4. Linear systems

We now write the conditions (36) and (40) as linear systems. For training inputs $U = \{ \mathbf { a } _ { i } \} _ { i = 1 } ^ { N }$ and $N _ { * }$ query inputs $U _ { * } = \{ \mathbf { a } _ { * } ^ { ( q ) } \} _ { q = 1 } ^ { N _ { * } }$ , we define the kernel matrice

$$
( K _ { U U } ) _ { i i ^ { \prime } } = \lambda ( { \bf a } _ { i } , { \bf a } _ { i ^ { \prime } } ) , \qquad ( K _ { U _ { \ast } U } ) _ { q i } = \lambda ( { \bf a } _ { \ast } ^ { ( q ) } , { \bf a } _ { i } ) , \qquad K _ { U U } ^ { \theta } = K _ { U U } + \theta I ,\tag{43}
$$

with � the regularization of (36), and the afine matrices $P _ { U }$ and $P _ { U _ { * } }$ with rows $( P _ { U } ) _ { i , : } \ = \ p ( \mathbf { a } _ { i } ) ^ { \mathsf { T } }$ and $( P _ { U _ { * } } ) _ { q , : } =$ $p ( \mathbf { a } _ { * } ^ { ( q ) } ) ^ { \top }$ . We keep every nonzero entry of $K _ { U U }$ , with no neighbors discarded by an approximate search. Its density nnz( $K _ { U U } ) / N ^ { 2 }$ , where nnz counts the nonzero entries, is thus set by $\varrho$ and the data, and the storage of its Cholesky factor also depends on the fill.

Since the kernel $\lambda ( \mathbf { a } , \mathbf { a } ^ { \prime } ) I$ does not couple diferent output coeficients, the conditions decouple into one system per stencil. On stencil $j ,$ let $B _ { i } \in \mathbb { R } ^ { N \times r _ { j } }$ hold the $\mathbf { b } _ { i } ^ { j }$ as rows, and let $C _ { j } \in \mathbb { R } ^ { N \times r _ { j } }$ and $\beta _ { i } \in \mathbb { R } ^ { n _ { \mathrm { a f f } } \times r _ { j } }$ hold the corresponding entries of the $\mathbf { c } _ { i }$ and the columns of $\beta .$ Evaluating (39) at the training inputs turns (40) into the first equation of

$$
\begin{array} { r l } { \left[ K _ { U U } ^ { \theta } \right. } & { { } P _ { U } } \\ { \left. P _ { U } ^ { \top } \right. } & { { } 0 \left. \right] \left[ \begin{array} { c } { C _ { j } } \\ { \beta _ { j } } \end{array} \right] = \left[ \begin{array} { c } { B _ { j } } \\ { 0 } \end{array} \right] , \qquad B _ { \ast , j } = K _ { U _ { \ast } U } C _ { j } + P _ { U _ { \ast } } \beta _ { j } , } \end{array}\tag{44}
$$

and evaluating it at the query inputs gives the second, where row $q$ of $B _ { * , j }$ holds the coeficients that $\tilde { f }$ predicts on stencil � for query input $q .$ The block $P _ { U } ^ { \top } C _ { j } = 0$ is the moment condition. Without augmentation, the same steps applied to (35) and (36) give

$$
K _ { U U } ^ { \theta } C _ { j } = B _ { j } , \qquad B _ { * , j } = K _ { U _ { * } U } C _ { j } .\tag{45}
$$

## 3.5. Sparse Cholesky and whitened QR

The system (44) is an indefinite saddle-point system. Forming the Schur complement $P _ { U } ^ { \top } ( K _ { U U } ^ { \theta } ) ^ { - 1 } P _ { U }$ squares the condition number of the whitened polynomial columns, as in the normal equations of least squares. We instead use the Cholesky–QR elimination of our unified kernel and polynomial interpolation [4]. Since $K _ { U U } ^ { \hat { \boldsymbol { \theta } } }$ is sparse and positive definite, we factor it by sparse Cholesky, whiten the polynomial columns, and factor them by thin QR. Suppressing the fill-reducing permutation, we obtain

$$
K _ { U U } ^ { \theta } = L _ { K } L _ { K } ^ { \top } , \qquad L _ { K } ^ { - 1 } P _ { U } = Q _ { P } R _ { P } , \qquad { \mathcal { E } } _ { U } = R _ { P } ^ { - 1 } Q _ { P } ^ { \top } L _ { K } ^ { - 1 } .\tag{46}
$$

Here $Q _ { P }$ has orthonormal columns and $R _ { P }$ is a nonsingular $n _ { \mathrm { a f f } } \times n _ { \mathrm { a f f } }$ triangular matrix, and we apply $\varepsilon _ { U }$ by triangular solves without forming $L _ { K } ^ { - 1 }$ . Block elimination of (44) then gives

$$
\beta _ { j } = \mathcal { E } _ { U } B _ { j } , \qquad K _ { U U } ^ { \theta } C _ { j } = B _ { j } - P _ { U } \beta _ { j } .\tag{47}
$$

Without augmentation, $Q _ { P } , R _ { P }$ , and $\varepsilon _ { U }$ are empty and the correction disappears. The same factors serve the parameter selection below and the prediction of Section 4.1, which never forms the $C _ { j }$

The size of the Cholesky factor, and the speed of its triangular solves, depend on the ordering of the training inputs. Since two inputs interact only if their principal coordinates are close, the sparsity pattern of $K _ { U U }$ resembles that of a discretization on a mesh in at most three dimensions, and we order it by nested dissection. For inputs of numerical rank one $( n _ { z } = 1 )$ , sorting the inputs by � even makes $K _ { U U }$ a band matrix (the matrix of an interval graph), whose Cholesky factor has no entries outside the band. However, the elimination of a band matrix is a single chain, which leaves the triangular solves of a prediction little parallelism on the GPU, while nested dissection gives a balanced elimination tree. Thus, nested dissection is our default, and the band ordering is preferable when memory is more limited than prediction time (Section 5.2.5). The ordering changes neither the operator regression nor its predictions beyond rounding.

## 3.6. Choosing the kernel parameters

The support radius �, the regularization �, and, for the fitted-shape kernel, the shape parameters remain to be chosen. The support radius is the most important of these, since it trades accuracy against sparsity. We use two criteria in turn. For the fitted-shape kernel, a restricted likelihood criterion fits the shape and the regularization from the training data alone. The error on validation fields then selects the radius and refines the regularization.

Let $\boldsymbol { F } ~ \in ~ \mathbb { R } ^ { N \times n _ { i } }$ � hold all sampled training outputs, with $n _ { v } ~ = ~ d m$ values per field, or �� $N _ { t }$ for spacetime outputs, after the amplitude normalization (72) where it applies, let $Q _ { 0 }$ be an orthonormal basis of the range of $P _ { U }$ $E _ { 0 } = ( I - Q _ { 0 } Q _ { 0 } ^ { \top } ) F$ , and $E = \sqrt { N } E _ { 0 } / \Vert E _ { 0 } \Vert _ { F }$ , and let

$$
\Pi _ { U } = ( K _ { U U } ^ { \theta } ) ^ { - 1 } - ( K _ { U U } ^ { \theta } ) ^ { - 1 } P _ { U } \left[ P _ { U } ^ { \top } ( K _ { U U } ^ { \theta } ) ^ { - 1 } P _ { U } \right] ^ { - 1 } P _ { U } ^ { \top } ( K _ { U U } ^ { \theta } ) ^ { - 1 } ,\tag{48}
$$

which we apply by sparse solves and never store. With $n _ { \mathrm { r e s } } = N - n _ { \mathrm { a f f } }$ , we minimize

$$
\begin{array} { r l } & { \mathcal { I } _ { c _ { d } } = \displaystyle \frac { 1 } { 2 } \left[ \frac { \log \operatorname* { d e t } K _ { U U } ^ { \theta } + 2 \log | \operatorname* { d e t } R _ { P } | } { n _ { \mathrm { r e s } } } + \log \left( \frac { \mathrm { t r } ( E ^ { \top } \Pi _ { U } E ) } { n _ { \mathrm { r e s } } } \right) \right] + c _ { d } \widetilde { d } , } \\ & { \widetilde { d } = \displaystyle \frac { 1 } { N ( N - 1 ) } \sum _ { i \neq i ^ { \prime } } \mathrm { s i g m o i d } \left( \frac { 1 - \| D _ { \mathbb { I } } ( z _ { i } - z _ { i ^ { \prime } } ) \| _ { 2 } / \varrho } { 0 . 0 3 } \right) . } \end{array}\tag{49}
$$

(50)

The bracket can be interpreted in two ways. In kernel terms, $\operatorname { t r } ( E ^ { \top } \Pi _ { U } E )$ is the squared norm, in the native space of the regularized kernel, of the augmented interpolants of the training outputs after their afine part is removed, and the log-determinants penalize kernels whose matrices have a large determinant, which are the more flexible ones. The bracket thus favors kernels that represent the outputs with a small norm, while penalizing overly flexible kernels. In probabilistic terms, if the outputs are modeled as samples of a Gaussian process with covariance $\lambda ,$ independent noise of variance $\theta ,$ and an afine mean with unknown coeficients, the bracket is, up to constants, the negative log restricted likelihood with the variance profiled out (with det $R _ { P } = 1$ without augmentation). We borrow the criterion from this setting, but use it only to select kernel parameters, not for uncertainty quantification.

The term $\widetilde { d } ,$ with $z _ { i } = z ( \mathbf { a } _ { i } )$ and the logistic function sigmoid(�) $) = 1 / ( 1 + e ^ { - x } )$ , is a smooth count of the interacting pairs, and its weight $c _ { d } \in \{ 0 , 1 \}$ trades accuracy against sparsity. We fit one candidate with each weight. All terms of $\mathcal { I } _ { c _ { d } }$ come from the sparse factors (Appendix B.3).

The criterion measures how well the kernel represents the training outputs, but our goal is the error of the predicted velocity for new inputs. Thus, we discard the radius that minimizes the likelihood criterion and choose the radius by validation alone. We first evaluate the five radii that give the densities 0.10, 0.25, 0.50, 0.75, and 0.90, with the regularization from the likelihood. We then take gradient steps in the radius and the regularization on the mean relative velocity error on validation fields, from four starts (half and a quarter of the largest of these radii, each with $\theta = 1 0 ^ { - 6 }$ and $1 0 ^ { - 3 } )$ , while the fitted shape stays unchanged. Among all candidates with density at most 0.90, we keep the one with the smallest validation error. The fixed-shape kernel goes through the same search without the likelihood step, with $\theta = 1 0 ^ { - 6 }$ at the five radii. All choices are made before the test fields are used $( \mathrm { A } \mathrm { I }$ ppendix B.2).

We thus compare four models, the fixed-shape and the fitted-shape kernel, each with and without afine augmentation. A comparison at the same augmentation isolates the efect of fitting the shape. Since all parameters are refit when the augmentation changes, a comparison with and without it measures the whole procedure.

## 4. Operator Lagrange prediction

We now combine the operator regression of Section 3 with the output reconstruction of Section 2. The direct way to predict is to solve (44) for the coeficients $C _ { j }$ and $\beta _ { j }$ of every stencil. This needs one operator right-hand side for every spatial coeficient, and the local coeficients of every training field. However, an operator regression with the same kernel for all outputs is a weighted combination of the training outputs, and the weights depend only on the new input. It sufices to solve once per new input for these weights and apply them to the whole output at once. Since the output reconstruction is linear, we can apply them even before the output reconstruction, to the sampled training fields.

## 4.1. Operator Lagrange weights

We first define the weights as functions on the input space. For an input �, let $k _ { U } ( \mathbf { a } ) = [ \lambda ( \mathbf { a } _ { i } , \mathbf { a } ) ] _ { i = 1 } ^ { N }$ . We define its operator Lagrange weights $\boldsymbol { \ell } ( { \mathbf { a } } ) \in { \mathbb { R } } ^ { N }$ by

$$
\begin{array}{c} \begin{array} { r l } { [ K _ { U U } ^ { \theta }  } & { { } P _ { U } } \\ {  P _ { U } ^ { \uparrow }  } & { { } 0 } \end{array} ] [ { \boldsymbol \ell } ( { \mathbf a } ) ] = [ k _ { U } ( { \mathbf a } ) ] ,  \end{array}\tag{51}
$$

where the multiplier $\mu ( \mathbf { a } )$ enforces the reproduction of afine functions. Block elimination with the factors (46) gives

$$
\ell ( \mathbf { a } ) = \big ( I - \mathcal { E } _ { U } ^ { \top } P _ { U } ^ { \top } \big ) ( K _ { U U } ^ { \theta } ) ^ { - 1 } k _ { U } ( \mathbf { a } ) + \mathcal { E } _ { U } ^ { \top } p ( \mathbf { a } ) ,\tag{52}
$$

so every weight $\ell _ { i }$ is a combination of the kernel functions $\lambda ( \cdot , { \bf a } _ { i ^ { \prime } } )$ and the afine functions, of the same form as (39). Since the matrix of (51) is the symmetric matrix of (44), the operator interpolant (39) and the learned operator (41)

take the Lagrange form

$$
\tilde { f } ( \mathbf { a } ) = \sum _ { i = 1 } ^ { N } \ell _ { i } ( \mathbf { a } ) \mathbf { b } _ { i } , \qquad \tilde { \mathcal { G } } ( a ) = \sum _ { i = 1 } ^ { N } \ell _ { i } ( \mathbf { a } ) \chi ( \mathbf { b } _ { i } ) ,\tag{53}
$$

a combination of the reconstructed training fields with weights that depend only on the input (Appendix C.4). Without augmentation, $\ell ( \mathbf { a } ) = ( K _ { _ { I I I I } } ^ { \theta } ) ^ { - 1 } k _ { U } ( \mathbf { a } )$ , and (53) is the Lagrange form of (35) and (37). The weights are regularized Lagrange (cardinal) functions of the training inputs [15, 46, 65]. For $\theta = 0$ , they are cardinal at the training inputs. For $\theta > 0 .$ , they are not, but they still reproduce the afine functions exactly.

Proposition 2 (Reproduction and preservation under shared weights). Assume $K _ { U U } ^ { \theta }$ is positive definite and $P _ { U }$ hasfull column rank. Then (52) is the unique weight vector in (51), and $P _ { U } ^ { \top } \ell ( { \mathbf a } ) = p ( { \mathbf a } )$ . In particular, the augmented model reproduces constants and the scaled principal coordinates in �(�). Let $\chi$ be any common linear output reconstruction ofthe training coeficient vectors, and let  be a linear constraint with $\mathcal { D } \chi ( \mathbf { b } _ { i } ) = 0 f o r$ every trainingfield. Then

$$
\chi \Biggl ( \sum _ { i } \ell _ { i } ( { \mathbf { a } } ) { \mathbf { b } } _ { i } \Biggr ) = \sum _ { i } \ell _ { i } ( { \mathbf { a } } ) \chi ( { \mathbf { b } } _ { i } ) , \qquad \mathcal { D } \chi \Biggl ( \sum _ { i } \ell _ { i } ( { \mathbf { a } } ) { \mathbf { b } } _ { i } \Biggr ) = 0 .\tag{54}
$$

Proofsketch. The factors (46) give $\mathcal { E } _ { U } P _ { U } = I$ , from which reproduction follows by substitution, and uniqueness follows from the block structure. The identities (54) are linearity. Appendix C.4 gives the details. □

With  the divergence on a stencil, the proposition shows that every prediction is divergence free on each stencil, for both kernels and with or without augmentation. Since the temporal weights (28) do not depend on the input, the operator weights also commute with interpolation in time.

We now turn to the discrete form. For a new input $\mathbf { a } _ { * }$ , restricting the first identity of (53) to stencil � gives the predicted coeficients

$$
\mathbf { b } _ { \ast } ^ { j } = B _ { j } ^ { \top } \boldsymbol { \ell } ( \mathbf { a } _ { \ast } ) ,\tag{55}
$$

and, with the weights of $N _ { * }$ new inputs as the rows of $\boldsymbol { L } \in \mathbb { R } ^ { N _ { * } \times N }$ , the second equation of (44) becomes

$$
B _ { * , j } = L B _ { j } .\tag{56}
$$

Figure 1 illustrates (55). One sparse solve gives the weights of a new input, and these weights produce the coeficients on every stencil. The number of operator solves now equals the number of new inputs and no longer depends on the size of the output representation. To compute (52), we first solve $K _ { U U } ^ { \theta } \ell _ { 0 } ( { \mathbf { a } } ) = k _ { U } ( { \mathbf { a } } )$ and then correct for the afine augmentation,

$$
\boldsymbol { \ell } ( \mathbf { a } ) = \ell _ { 0 } ( \mathbf { a } ) + \boldsymbol { \mathcal { E } } _ { U } ^ { \top } \left[ p ( \mathbf { a } ) - P _ { U } ^ { \top } \ell _ { 0 } ( \mathbf { a } ) \right] .\tag{57}
$$

Without augmentation, $L = K _ { U _ { \ast } U } ( K _ { U U } ^ { \theta } ) ^ { - 1 }$ , so that

$$
K _ { U U } ^ { \theta } L ^ { \mathsf { T } } = K _ { U _ { \ast } U } ^ { \mathsf { T } } .\tag{58}
$$

## 4.2. Applying the weights before the output reconstruction

With $\mathbf { b } _ { i }$ as in Section 3.2, let  be the linear map from a sampled output $\mathbf { v } _ { i }$ to its local output reconstruction $\chi ( \mathbf { b } _ { i } )$ . Since the same stencils, polynomial bases, and temporal weights serve every training field, linearity gives

$$
\begin{array} { l } { { \displaystyle { \widetilde { \mathcal { G } } ( a ) = \chi \left( \sum _ { i = 1 } ^ { N } \ell _ { i } ( { \mathbf { a } } ) { \mathbf { b } } _ { i } \right) = \sum _ { i = 1 } ^ { N } \ell _ { i } ( { \mathbf { a } } ) \chi ( { \mathbf { b } } _ { i } ) } } } \\ { { \displaystyle ~ = \mathcal { R } \left( \sum _ { i = 1 } ^ { N } \ell _ { i } ( { \mathbf { a } } ) { \mathbf { v } } _ { i } \right) } . } \end{array}\tag{59}
$$

Our GPU implementation uses the last form. It combines the sampled training fields with the weights and then applies  once per new input. At setup, adjoint solves with the local matrices $\mathbf { A } _ { j }$ turn  into compact evaluation weights on each stencil. Prediction then stores the sampled training fields, the evaluation weights, and the stencil indices, but none of the � sets of local coeficients. For the Taylor–Green benchmarks, whose responses we divide by the input amplitude before fitting (Section 5.2.1), the same identity holds for the normalized responses.

## 4.3. Rounding errors of direct and Lagrange evaluation

The forms in (59) agree in exact arithmetic, but they round diferently. The diference is important when a moderate velocity is the result of cancellation among large spatial coeficients, which can happen for a global interpolant. A solve error then enters the velocity amplified by the output reconstruction. We now quantify this for the model without augmentation and for evaluation through the coeficients, which is the setting of our comparison in Section 5.3. The analysis does not cover the afine correction (57) or the order of operations of Section 4.2.

Assume $K _ { U U } ^ { \theta }$ is symmetric and nonsingular. We fix one stencil and drop its index, writing $r = r _ { i } , B = B _ { i } \in \mathbb { R } ^ { N \times r }$ and $C = C _ { i } \in \mathbb { R } ^ { N \times r }$ , so that each row of � holds the spatial coeficients of one training field. Let $\mathbf { A } ^ { \star } \in \mathbb { R } ^ { n _ { \mathrm { e v a l } } \times r }$ stack the evaluation matrices (18) of the $n _ { \mathrm { e v a l } }$ velocity values reconstructed from this stencil, so that $B ( \mathbf { A } ^ { \star } ) ^ { \top }$ holds the reconstructed training velocities. The direct coeficients � and the Lagrange weights � satisfy

$$
K _ { U U } ^ { \theta } C = B , \qquad L K _ { U U } ^ { \theta } = K _ { U _ { * } U } ,\tag{60}
$$

and both give the prediction $V _ { * } ~ = ~ K _ { U , U } { \cal C } ( { \bf A } ^ { \star } ) ^ { \top } ~ = ~ L B ( { \bf A } ^ { \star } ) ^ { \top }$ . The same notation describes a global output reconstruction, with a single stencil.

Theorem 3 (Solve errors and multiplication rounding). For computed approximations $\hat { C }$ and ${ \widehat { L } } ,$ define the solve residuals by evaluating their defining equations in exact arithmetic,

$$
R _ { \mathrm { d i r } } = B - K _ { U U } ^ { \theta } \widehat { C } , \qquad R _ { \mathrm { L a g } } = K _ { U _ { * } U } - \widehat { L } K _ { U U } ^ { \theta } .\tag{61}
$$

Before subsequent multiplication rounding, the prediction errors satisfy

$$
\begin{array} { r l } & { K _ { U _ { * } U } \widehat { C } ( \mathbf { A } ^ { \star } ) ^ { \top } - V _ { * } = - L R _ { \mathrm { d i f } } ( \mathbf { A } ^ { \star } ) ^ { \top } , } \\ & { \qquad \widehat { L } B ( \mathbf { A } ^ { \star } ) ^ { \top } - V _ { * } = - R _ { \mathrm { L a g } } ( K _ { U U } ^ { \theta } ) ^ { - 1 } B ( \mathbf { A } ^ { \star } ) ^ { \top } . } \end{array}\tag{62}
$$

(63)

Define, with fl(⋅) denoting evaluation infloating point, $\widehat { V } _ { * , \mathrm { d i r } } = \mathrm { f l } ( \mathrm { f l } ( K _ { U , \iota \ U } \widehat { C } ) ( \mathbf { A } ^ { \star } ) ^ { \top } )$ and $\widehat { V } _ { \ast , \mathrm { L a g } } = \mathrm { f l } ( \mathrm { f l } ( \widehat { L } B ) ( \mathbf { A } ^ { \star } ) ^ { \top } )$

Suppose the coeficient products and the output reconstruction use standard floating-point dot products, possibly split into � component blocksfollowed by summation.

Assume no overflow or underflow, and set $\gamma _ { k } = k u _ { \mathrm { m a c h } } / ( 1 - k u _ { \mathrm { m a c h } } ) ,$ , where $u _ { \mathrm { m a c h } }$ is unit roundof and max $\{ N , r +$ $d \} u _ { \mathrm { m a c h } } < 1$ . With $\gamma _ { \mathrm { m u l t } } = \gamma _ { N } + \gamma _ { r + d } + \gamma _ { N } \gamma _ { r + d } ,$ , we have

$$
\begin{array} { r } { \| \widehat { V } _ { * , \mathrm { d i r } } - V _ { * } \| _ { F } \leq \| L \| _ { F } \| R _ { \mathrm { d i r } } \| _ { F } \| { ( \mathbf { A } ^ { \star } ) } ^ { \mathsf { T } } \| _ { F } + \gamma _ { \mathrm { m u l t } } \Big \| | K _ { U _ { * } U } | | \widehat { C } | | { ( \mathbf { A } ^ { \star } ) } ^ { \mathsf { T } } | \Big \| _ { F } , } \end{array}\tag{64}
$$

$$
\begin{array} { r } { \| \widehat { V } _ { * , \mathrm { L a g } } - V _ { * } \| _ { F } \leq \| R _ { \mathrm { L a g } } \| _ { F } \| ( K _ { U U } ^ { \theta } ) ^ { - 1 } B ( { \mathbf A } ^ { \star } ) ^ { \top } \| _ { F } + \gamma _ { \mathrm { m u l t } } \Big \| | \widehat { L } | | B | | ( { \mathbf A } ^ { \star } ) ^ { \top } | \Big \| _ { F } . } \end{array}\tag{65}
$$

Here absolute values are taken entrywise. The residual identities require symmetry and nonsingularity, but notpositive definiteness.

Proofsketch. Substituting the residuals into the two predictions gives (62)–(63). The standard bound for floating-point dot products then bounds the multiplications. Appendix C.5 gives the proof. □

The bounds become clearer with the amplification factor

$$
\alpha _ { \mathrm { s p a t i a l } } = \frac { \| B \| _ { F } \| ( \mathbf { A } ^ { \star } ) ^ { \top } \| _ { F } } { \| B ( \mathbf { A } ^ { \star } ) ^ { \top } \| _ { F } } \geq 1\tag{66}
$$

which measures how much the coeficients cancel in the output reconstruction of the training velocities. If $\lvert \lvert R _ { \mathrm { d i r } } \rvert \rvert _ { F } \leq$ $\eta _ { \mathrm { d i r } } \Vert B \Vert _ { F }$ and $\Vert R _ { \mathrm { L a g } } \Vert _ { F } \leq \eta _ { \mathrm { L a g } } \Vert K _ { U _ { \ast } U } \Vert _ { F }$ , the solve errors satisfy

$$
\| K _ { U _ { * } U } \widehat C ( { \mathbf { A } ^ { \star } } ) ^ { \top } - V _ { * } \| _ { F } \leq \eta _ { \mathrm { d i r } } \| L \| _ { F } \alpha _ { \mathrm { s p a t i a l } } \| B ( { \mathbf { A } ^ { \star } } ) ^ { \top } \| _ { F } ,\tag{67a}
$$

$$
\| \widehat { L } B ( { \bf A } ^ { \star } ) ^ { \top } - V _ { * } \| _ { F } \leq \eta _ { \mathrm { L a g } } \| K _ { U _ { * } U } \| _ { F } \| ( K _ { U U } ^ { \theta } ) ^ { - 1 } \| _ { F } \| B ( { \bf A } ^ { \star } ) ^ { \top } \| _ { F } .\tag{67b}
$$

In direct evaluation, the relative solve error $\eta _ { \mathrm { d i r } }$ is amplified by $\alpha _ { \mathrm { s p a t i a l } } .$ . In Lagrange evaluation, the coeficients have already been combined into velocities, and $\alpha _ { \mathrm { s p a t i a l } }$ does not appear. A global interpolant with a smooth kernel can have very large $\alpha _ { \mathrm { s p a t i a l } }$ , since its coeficients are large and cancel, while a local interpolant on a small stencil has moderate coeficients. The rounding of the multiplications, however, contains the entrywise product with $| ( \mathbf { A } ^ { \star } ) ^ { \top } |$ in both bounds (of $| \widehat { C } |$ in direct and of $| B |$ in Lagrange evaluation), so cancellation in the output reconstruction also enters the rounding, and the theorem does not make Lagrange evaluation more accurate in general. We test these predictions in Section 5.3.

## 4.4. GPU implementation

Algorithm 1 summarizes our implementation. Every floating-point computation runs on the GPU in double precision, from the normalization of the inputs and the stencil selection to the factorizations and the output reconstruction, using the routines of KernelPack-JAX [8]. Only the integer bookkeeping of the cell-based stencil search for large non-periodic node sets (Section 5.1.5) runs on the host. The kernel is assembled pair by pair over its support, in passes of bounded size, so that no dense kernel or distance matrix is ever formed.

However, JAX has no sparse direct solver that keeps a factorization between calls. Our implementation therefore factors $K _ { U U } ^ { \theta }$ with NVIDIA’s cuDSS through the primitive kernelpack.sparse.cudss of KernelPack-JAX, which exposes cuDSS to XLA, the compiler behind JAX. We use cuDSS in its symmetric positive-definite mode, with its nested-dissection ordering (Section 3.5). The factors stay on the GPU, and the right-hand sides of a batch of new inputs are solved as the columns of one call. A change of the support (i.e., of �) requires a new ordering and symbolic analysis, while a change of the values only requires a numerical refactorization. Appendix B.4 describes the batching and the checks of the timed runs.

Algorithm 1 Local and sparse kernel operator learning on the GPU.   
Require: training and validation pairs, new inputs, output points, spatial kernel and polynomial degree   
Output reconstruction:   
1: select the stencils, with periodic images where needed (Section 2.3)   
2: build the divergence-free polynomials and the local systems, and prepare  (Section 2.1)   
3: for spacetime outputs, add the temporal weights (28)   
Operator regression:   
4: standardize the training inputs and compute their principal directions   
5: for each choice of augmentation, keep the fixed shape or fit it by (49)   
6: select the radius and regularization on the validation fields   
7: assemble $K _ { U U } ^ { \theta }$ in sparse form and factor it with cuDSS   
8: with augmentation, form the whitened QR correction (46)   
Prediction:   
9: solve for the Lagrange weights (57) of a batch of new inputs   
10: combine the sampled training fields and apply  by (59)   
11: check the solve residuals and the reproduction of afine functions

## 4.5. Work and storage

Table 1 lists the work and storage of each step. Let $\mathcal { N } _ { K } = \mathrm { n n z } ( K _ { U U } )$ and $\sqrt { \mathrm { { f a c } } } = \mathrm { { n n z } } ( L _ { K } )$ , where $L _ { K }$ is the Cholesky factor of $K _ { U U } ^ { \theta }$ in (46). With $\mathcal { L } _ { i }$ the set of rows in which column � of $L _ { K }$ is nonzero, so that $\begin{array} { r } { \sum _ { i = 1 } ^ { N } | \mathcal { L } _ { i } | = \mathcal { N } _ { \mathrm { f a c } } , } \end{array}$ the sparse factorization takes, up to a constant factor,

$$
F _ { \mathrm { c h o l } } = \sum _ { i = 1 } ^ { N } | \mathcal { L } _ { i } | ^ { 2 } , \qquad \frac { \mathcal { N } _ { \mathrm { f a c } } ^ { 2 } } { N } \leq F _ { \mathrm { c h o l } } \leq N \mathcal { N } _ { \mathrm { f a c } } ,\tag{68}
$$

floating-point operations [9], where the lower bound follows from the Cauchy–Schwarz inequality and the upper bound from $| { \mathcal { L } } _ { i } | \leq N$ . For a dense factor, $\lvert \mathcal { L } _ { i } \rvert = N - i + 1$ and $F _ { \mathrm { c h o l } } \approx N ^ { 3 } / 3$ . Let $b _ { q }$ be the number of new inputs in a batch, let $m _ { \mathrm { r h s } }$ be the number of spatial coeficients, and let $n _ { v }$ be the number of sampled output values per field $( n _ { v } = d m .$

Work and storage. Here $\textstyle { \mathcal { N } } _ { K }$ and $\mathcal { N } _ { \mathrm { f a c } }$ are the numbers of nonzeros of the kernel matrix and of its Cholesky factor, $F _ { \mathrm { c h o l } }$ the work (68) of the factorization, $b _ { q }$ the number of new inputs in a batch, $n _ { \mathrm { a f f } }$ the number of afine columns, $n _ { v }$ the number of sampled output values per field, $m ^ { \star }$ the number of query points, $r _ { j }$ the size of the local system on stencil �, and $\begin{array} { r } { m _ { \mathrm { r h s } } = \sum _ { j } r _ { j } } \end{array}$ . The last two rows belong to direct evaluation through coeficients, which our predictions avoid. For spacetime outputs, the counts of these two rows are multiplied by $N _ { t } .$
<table><tr><td>Step</td><td>Work</td><td>Storage</td></tr><tr><td>Kernel factorization</td><td> $F _ { \mathrm { c h o l } }$ </td><td> $\mathcal { N } _ { K } , \mathcal { N } _ { \mathrm { f a c } }$ </td></tr><tr><td>Affine correction  $\left( n _ { \mathrm { a f f } } \right)$  columns)</td><td>solves with  $n _ { \mathrm { a f f } }$  right-hand sides,  $O ( N n _ { \mathrm { a f f } } ^ { 2 } )$   $\mathsf { Q R }$ </td><td> $O ( N n _ { \mathrm { a f f } } )$ </td></tr><tr><td>Lagrange weights,  $b _ { q }$  queries</td><td> $O ( b _ { q } { \mathcal { N } } _ { \mathrm { f a c } } ) + O ( b _ { q } N n _ { \mathrm { a f f } } )$ </td><td> $b _ { q } N \ ( { \mathsf { d e n s e } } )$ </td></tr><tr><td>Combining training fields Local systems  $( \mathsf { s e t u p } )$ </td><td> $O ( b _ { q } N n _ { v } )$ </td><td> $N n _ { v }$ </td></tr><tr><td>Local evaluation,  $m ^ { \star }$  points</td><td> $O ( r _ { i } ^ { 3 } )$  per stencil  $O ( \bar { b _ { q } } m ^ { \star } d ^ { 2 } n _ { s } )$  per snapshot</td><td> $O ( m ^ { \star } d ^ { 2 } n _ { s } )$  weights,  $O ( m ^ { \star } n _ { s } )$  indices</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Training coefficients Applying weights to coefficients</td><td> $O ( N \sum _ { j } r _ { j } ^ { 2 } )$   $O ( b _ { q } N m _ { \mathrm { r h s } } )$ </td><td> $N \sum _ { j } r _ { j }$ </td></tr></table>

or �� $N _ { t }$ for spacetime outputs). We count the work of a triangular solve as proportional to the number of nonzeros of the factor.

Two observations follow from the table. First, the local output reconstruction costs $O ( r _ { j } ^ { 3 } )$ per stencil, with $r _ { j } = d n _ { s } + n _ { P , j }$ fixed by the stencil size and the polynomial degree, so its cost grows linearly with the number of output points instead of cubically. Second, the cost of the operator regression is governed by the factor of $K _ { U U } ^ { \theta }$ , whose size depends on the fill as well as on the density of $K _ { U U }$ [9]. At a fixed density, the kernel matrix and its factor still grow as $N ^ { 2 }$ . For this reason, we report density, factor storage, and prediction time separately in Section 5.

Note that the Lagrange weights are dense even when $K _ { U U }$ is sparse. Direct evaluation instead solves $m _ { \mathrm { r h s } }$ righthand sides once and can reuse its coeficients for any number of new inputs, so the cheaper route depends on how many predictions are needed. Appendix B.3 gives the storage accounting of our experiments.

## 5. Numerical experiments

We now test our method in three steps. We first test the local output reconstruction alone, with exact fields in place of predictions, for convergence and for the three properties it preserves (Section 5.1). We then compare four operator models on ten fluid benchmarks as the training set grows, in accuracy, sparsity, and storage (Section 5.2). Finally, we compare direct and Lagrange evaluation in floating point and measure the cost of prediction (Section 5.3).

## 5.1. Local output reconstruction

Exact velocity fields in place of predictions separate the error of the output reconstruction from that of the operator regression. We use these tests to answer three questions. Does it converge at the order set by its polynomial degree on scattered nodes? Are its divergence and its periodic mismatch at rounding level? And does the multiscale Matérn kernel reproduce prescribed turbulence statistics as the nodes are refined?

## 5.1.1. Convergence on scattered nodes

We interpolate manufactured divergence-free fields in the smooth domains $\Omega _ { d } = \{ F _ { d } > 0 \} , d = 2 , 3$ , bounded by a quartic curve and a quartic surface. Each field is the curl of a trigonometric stream function $( d = 2 )$ or vector potential $( d = 3 )$ multiplied by $F _ { d } ^ { 2 }$ , so it vanishes on the boundary (Appendix D.2 gives $F _ { d }$ and the potentials). We use divergence-free polynomials of degree $\varpi = 1 , 3$ , and 5 for target orders 2, 4, and 6. Each stencil has $n _ { s } = 2 \lceil n _ { P } / d \rceil + 1$ nodes, where $n _ { P }$ is the dimension of the polynomial space, and both the PHS $( r ^ { 3 }$ for degrees 1 and 3, and $r ^ { 5 }$ for degree 5) and the multiscale Matérn kernel $( c _ { \sigma } = 1 6$ , which sets the Matérn lengths relative to the stencil radius in (21)) use the same stencils and polynomials. The node sets are four refinements generated by the node generator of KernelPack-JAX [8], with 172 to 1907 nodes in 2D and 197 to 2570 in 3D, and the stencils are boundary aware (Section 2.1).

![](images/231116446c522a9b6b307b893010253eec96f2a2245d21c6b448fe06509ed59e.jpg)  
Figure 3: Convergence of the local output reconstruction on scattered nodes in the domains $\Omega _ { 2 }$ (left) and $\Omega _ { 3 } \ ( \mathsf { r i g h t } )$ , with PHS and multiscale Matérn kernels and divergence-free polynomials. Relative error $E _ { \mathrm { s p } }$ at random evaluation points that are not nodes, against the sampled fill distance $h _ { Q }$ of (69). Dotted lines: $h _ { Q } ^ { 2 } , h _ { Q } ^ { 4 } , h _ { Q } ^ { 6 } .$

Let $v _ { h }$ be the output reconstruction of the exact field � from the nodes � of a node set with spacing ℎ. We measure its relative error on a fixed set � of uniformly random interior evaluation points (4096 in 2D and 32768 in 3D), against the fill distance sampled on the same set,

$$
E _ { \mathrm { s p } } = \left( \frac { \sum _ { \boldsymbol { y } \in \boldsymbol { Q } } \| \boldsymbol { v } _ { h } ( \boldsymbol { y } ) - \boldsymbol { v } ( \boldsymbol { y } ) \| _ { 2 } ^ { 2 } } { \sum _ { \boldsymbol { y } \in \boldsymbol { Q } } \| \boldsymbol { v } ( \boldsymbol { y } ) \| _ { 2 } ^ { 2 } } \right) ^ { 1 / 2 } ,
$$

$$
h _ { Q } = \operatorname* { m a x } _ { y \in Q } \operatorname* { m i n } _ { y ^ { \prime } \in Y } \| y - y ^ { \prime } \| _ { 2 } ,\tag{69}
$$

where each evaluation point uses the stencil of its nearest node.

Figure 3 shows the results. In 2D, both kernels converge at close to the target orders, with least-squares rates of 2.1, 3.9, and 5.8. In 3D, the rates over all four node sets are lower, 1.7–1.9, 3.4–3.7, and 5.0, but the sixth-order rate rises to 5.5 over the three finest node sets. The 3D deficit is likely due to the coarsest node sets, which are not yet in the asymptotic range. The two kernels give nearly the same errors on smooth fields, so the multiscale kernel does not reduce the accuracy on smooth fields.

## 5.1.2. Divergence and periodicity

Since every basis function is divergence free, the divergence of the output reconstruction should vanish up to rounding, however large its approximation error. Diferentiating each local expansion on its own stencil, we find a normalized root-mean-square (RMS) divergence at rounding level for both kernels, all degrees, and all node sets (Appendix D.2). We test periodic stencils on scattered nodes in $[ 0 , 2 \pi ) ^ { d }$ , comparing the reconstructed velocities and their Jacobians at periodically equivalent points on faces, edges, and corners. The diferences are again at rounding level for both kernels and all degrees, while the approximation error varies with degree and resolution as expected.

## 5.1.3. Kolmogorov statistics

The multiscale kernel builds a Kolmogorov law into its weights. We now test whether interpolation on finite stencils preserves this law. We reconstruct a three-dimensional solenoidal Fourier ensemble whose velocity increments follow the Kolmogorov law $S _ { 2 } ( r ) \ \propto \ r ^ { 2 / 3 }$ over the separations $0 . 0 3 \leq r \leq 0 . 8$ . The ensemble is a sum of 1514 Fourier modes with Gaussian coeficients perpendicular to their wave vectors and $\mathrm { a } - 5 / 3$ energy spectrum, restricted to $\Omega _ { 3 }$ (Appendix D.2.3). Below $r = 0 . 0 3$ , its finite bandwidth makes the increments smooth. For pairs of points $x _ { k } ( r )$ and $y _ { k } ( r )$ at distance � along eight fixed directions $k = 1 , \ldots , 8$ , we compute the second-order structure function exactly from the covariance of the ensemble,

$$
\begin{array} { l } { { S _ { 2 , k } ( r ) = \mathbb { E } \big [ \| v ( x _ { k } ( r ) ) - v ( y _ { k } ( r ) ) \| _ { 2 } ^ { 2 } \big ] , \qquad \| x _ { k } ( r ) - y _ { k } ( r ) \| _ { 2 } = r , } } \\ { { S _ { 2 } ( r ) = { \frac { 1 } { 8 } } \displaystyle \sum _ { k = 1 } ^ { 8 } S _ { 2 , k } ( r ) , } } \end{array}\tag{70}
$$

![](images/467e31feb4af223f6fe09667f7915b84155eb1bc28ee54b44086d2b9cb2f0802.jpg)

![](images/aa785af858d1fbd3b7c41061a0a55786decefda12e4f59ce46d46d35021d643b.jpg)  
Figure 4: Kolmogorov statistics of the local multiscale Matérn output reconstruction. Left: structure functions for degree five on three of the six node sets, and the exact reference. Right: worst relative error $E _ { S _ { 2 } } ( 0 . 8 )$ of (71) of the reconstructed structure function against the node spacing ℎ, with an $h ^ { 2 / 3 }$ guide.

both for the ensemble and, with $v _ { h }$ in place of $v ,$ for its output reconstruction $( S _ { 2 , k , h } )$ . We measure the worst relative error over the directions,

$$
E _ { S _ { 2 } } ( r ) = \operatorname* { m a x } _ { 1 \leq k \leq 8 } \frac { | S _ { 2 , k , h } ( r ) - S _ { 2 , k } ( r ) | } { S _ { 2 , k } ( r ) } .\tag{71}
$$

We use the multiscale Matérn kernel with polynomial degrees 1, 3, and 5 on six node sets, with node spacings ℎ from 0.26 to 0.05 (128 to 13567 nodes).

Figure 4 shows the structure functions approaching the reference under refinement. At the largest separation, $r = 0 . 8$ , the error decays like $h ^ { 0 . 6 4 } , h ^ { 0 . 7 3 }$ , and $h ^ { 0 . 6 7 }$ for the three degrees. These are fits over a finite range of ℎ, not asymptotic rates. For degree five, the fitted slope of $S _ { 2 }$ over the upper part of the range moves toward the reference slope. However, the error at small separations stays large even on the finest node set, since these separations are below the node spacing. To see whether locality is to blame, we compare local and global interpolation on one node set, with the same kernel and polynomials (Appendix D.2.3). Both have nearly the same error, at $r = 0 . 8$ and over the whole range, so the small-separation error is likely due to the resolution of the nodes and not to the stencils.

## 5.1.4. Output reconstruction in thefluid benchmarks

We now test the output reconstruction on the predictions of the operator benchmarks (Section 5.2.1). The data sets hold the velocity at more mesh points than the � output points that we use as nodes, and the remaining points lie between the nodes. We reconstruct the predicted nodal velocities of the model with the smallest validation error at $N = 1 0 0 0 0$ with the global divergence-free Matérn interpolant of our earlier work and with our local output reconstruction, and measure the error at these points (Table 2, Appendix A.5). Neither output reconstruction is uniformly more accurate. The global one is more accurate for both cylinder flows, the backward-facing step, and Taylor–Green vortices (space), by factors between 1.06 and 2.7. The local one is more accurate for the other six, by factors between 1.04 (turbulent species transport) and 10 (buoyancy-driven cavity), although each local system has at most 62 unknowns in 2D and 262 in 3D, against up to 21000 in the global system. For several flows, the errors between the nodes are much larger than those at the nodes, so they are set by the resolution of the � nodes and not by the operator regression. For both turbulent flows, the multiscale Matérn kernel has a lower error than the PHS, and validation selects $c _ { \sigma } = 1 6$ for both. Almost all stencils keep the full polynomial space, except in the species-transport mixer, whose geometry truncates the rank on more than half the stencils (Table 7 in Appendix A.2).

## 5.1.5. Cost of the local output reconstruction

We now compare the setup cost of our local output reconstruction with that of a global divergence-free output reconstruction, as in PPKM, on scattered nodes in the unit square and the unit cube. The global setup assembles the dense �� × �� Gram matrix of a divergence-free Matérn kernel and factors it by Cholesky. The local setup selects the stencils, builds the polynomial bases and the local systems, factors them, and forms the evaluation weights of

![](images/7f39c8fec71378f0973ec04933a172bba5f93dd345cb755bba1e8e0ef6d2a14a.jpg)

Table 2  
Test error between the nodes. We reconstruct the predicted nodal velocities of the 100 test inputs (model with the smallest validation error at $N = 1 0 0 0 0 )$ and measure the mean relative velocity error at the evaluation points, the mesh points of each data set that are not among its � nodes. Global: the divergence-free Matérn interpolant of Sharma et al. [53] on all nodes. Local: our output reconstruction with the cubic PHS, or with the multiscale Matérn kernel $\left( c _ { \sigma } = 1 6 \right)$ where marked (Appendix A.5 compares it with the PHS). Bold: smaller error in each row.
<table><tr><td>Data set</td><td>Nodes m</td><td>Evaluation points</td><td>Global</td><td></td><td>Local</td></tr><tr><td>Forced isotropic turbulence</td><td>7000</td><td>3000</td><td> $1 . 8 8 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 1 . 6 5 \times 1 0 ^ { - 2 } }$ </td><td>(multiscale)</td></tr><tr><td>Turbulent species transport</td><td>7000</td><td>4388</td><td> $1 . 5 2 \times 1 0 ^ { - 1 }$ </td><td> $\mathbf { 1 . 4 7 \times 1 0 ^ { - 1 } }$ </td><td>(multiscale)</td></tr><tr><td>Merging vortices</td><td>500</td><td>6947</td><td> $1 . 8 8 \times 1 0 ^ { - 1 }$ </td><td></td><td> $\mathbf { 1 . 3 5 \times 1 0 ^ { - 1 } }$ </td></tr><tr><td>Lid-driven cavity</td><td>1000</td><td>8566</td><td> $6 . 1 5 \times 1 0 ^ { - 1 }$ </td><td></td><td> $\mathbf { 8 . 7 1 \times 1 0 ^ { - 2 } }$ </td></tr><tr><td>Cylinder flow (no shedding)</td><td>1000</td><td>8520</td><td> $\mathbf { 8 . 6 7 \times 1 0 ^ { - 2 } }$ </td><td></td><td> $9 . 1 9 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Cylinder flow (vortex shedding)</td><td>1000</td><td>8520</td><td> $\mathbf { 9 . 6 5 \times 1 0 ^ { - 2 } }$ </td><td></td><td> $1 . 1 2 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>Backward-facing step</td><td>1000</td><td>6359</td><td> $\mathbf { 5 . 8 4 \times 1 0 ^ { - 3 } }$ </td><td></td><td> $1 . 0 4 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Buoyancy-driven cavity</td><td>5000</td><td>5514</td><td> $9 . 4 2 \times 1 0 ^ { - 2 }$ </td><td></td><td> $\mathbf { 9 . 3 1 \times 1 0 ^ { - 3 } }$ </td></tr><tr><td>Taylor-Green vortices (space)</td><td>500</td><td>6947</td><td> $\mathbf { 2 . 3 0 \times 1 0 ^ { - 4 } }$ </td><td></td><td> $6 . 2 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Taylor-Green vortices (spacetime)</td><td>500</td><td>6947</td><td> $1 . 9 9 \times 1 0 ^ { - 3 }$ </td><td></td><td> $\mathbf { 6 . 2 0 \times 1 0 ^ { - 4 } }$ </td></tr></table>

Figure 5: Setup time and memory of our local output reconstruction (degree four, cubic PHS) and of a global divergencefree Matérn output reconstruction (dense Gram matrix and Cholesky factor) against the number � of output points, on scattered nodes in the unit square (2D) and cube (3D). Dotted: the global cost extrapolated as $( d m ) ^ { 3 }$ in time and $( d m ) ^ { 2 }$ in memory beyond the largest size that a single dense factorization fits on our GPU. All times are measured after compilation.

Section 4.2, with the degree-four output reconstruction of the operator benchmarks. Both run on the GPU in double precision, and we time them after compilation. Figure 5 shows the results.

The local setup grows at most linearly in �, in both time and memory, as Table 1 predicts. The global setup grows like $( d m ) ^ { 3 }$ in time and $( d m ) ^ { 2 }$ in memory, and a single dense factorization no longer fits on our GPU beyond $d m \approx 2 . 5 { \times } 1 0 ^ { 4 }$ . It could be factored in blocks, at the same cubic cost. On small node sets, the dense global factorization is faster, since its work is done by a few large and eficient GPU kernels. The local setup becomes the faster one near $m = 3 \times 1 0 ^ { 3 }$ in 2D and $m = 5 \times 1 0 ^ { 3 }$ in 3D, and its memory is smaller at every size we tested, by a factor that grows linearly with �.

Two choices keep the local setup linear in �, and neither changes the output reconstruction beyond rounding. The polynomial bases are built by QR factorizations instead of singular value decompositions (Appendix A.2), and non-periodic node sets of more than 4096 nodes find their stencils by an exact search on a grid of cells, whose work grows linearly with �. Together, they make the setup of the output reconstruction a small part of the ofline cost of every operator benchmark (Appendix D.6.1).

Table 3  
The ten benchmarks: input and output of the operator, number of sampled input coordinates, number � of principal input $n _ { z }$ coordinates (the numerical rank of the training inputs, capped at three), spatial dimension �, number � of output points, and spatial kernel. Taylor–Green vortices (spacetime) has spacetime outputs, at the same 500 points at four times.
<table><tr><td>Data set</td><td>Input → output</td><td>Input coord.</td><td> $n _ { z }$ </td><td> $d$ </td><td>m</td><td>Spatial kernel</td></tr><tr><td>Forced isotropic turbulence</td><td>f → u(·, 0.5)</td><td>21000</td><td>3</td><td>3</td><td>7000</td><td>Five-scale Matérn</td></tr><tr><td>Turbulent species transport</td><td>inflow → u(·, 0.5)</td><td>1388</td><td>2</td><td>3</td><td>7000</td><td>Five-scale Matérn</td></tr><tr><td>Merging vortices</td><td>ω(·, 0) → u(·, 0.4)</td><td>500</td><td>3</td><td>2</td><td>500</td><td>Cubic PHS</td></tr><tr><td>Lid-driven cavity</td><td> $\mathbf { u } ( \cdot , 0 )  \mathbf { u } ( \cdot , 5 )$ </td><td>2000</td><td>1</td><td>2</td><td>1000</td><td>Cubic PHS</td></tr><tr><td>Cylinder flow (no shedding)</td><td>u(·, 0) → u(·, 10)</td><td>2000</td><td>1</td><td>2</td><td>1000</td><td>Cubic PHS</td></tr><tr><td>Cylinder flow (vortex shedding)</td><td>u(·, 0) → u(·, 10)</td><td>2000</td><td>1</td><td>2</td><td>1000</td><td>Cubic PHS</td></tr><tr><td>Backward-facing step</td><td>inflow → u(·, 5)</td><td>11</td><td>1</td><td>2</td><td>1000</td><td>Cubic PHS</td></tr><tr><td>Buoyancy-driven cavity</td><td> $T ( \cdot , 0 ) \to { \mathbf { u } } ( \cdot , 5 )$ </td><td>5000</td><td>1</td><td>2</td><td>5000</td><td>Cubic PHS</td></tr><tr><td>Taylor-Green vortices (space)</td><td>u(·, 0) → u(·, 1)</td><td>1000</td><td>1</td><td>2</td><td>500</td><td>Cubic PHS</td></tr><tr><td>Taylor-Green vortices (spacetime)</td><td> $\mathbf { u } ( \cdot , 0 )  \mathbf { u } ( \cdot , t ) , t \in [ 0 . 7 , 1 ]$ </td><td>1000</td><td>1</td><td>2</td><td>500</td><td>Cubic PHS</td></tr></table>

## 5.2. Sparse operator regression

## 5.2.1. Benchmarks and models

We use the ten data sets of our previous work [53], listed in Table 3. Eight are two-dimensional laminar flows, two of them the analytical Taylor–Green vortices, and two are three-dimensional turbulent flows, a Reynolds-averaged simulation of a static mixer and a large-eddy simulation of forced isotropic turbulence. In each, the input is an initial velocity, temperature, or vorticity, an inflow, or a body force, sampled at points of the domain or of its inflow boundary, and the output is the velocity at a fixed later time (or, for Taylor–Green vortices (spacetime), at four times), sampled at � output points. The output points are approximate Fekete points, i.e., well-spaced points selected from the mesh [53]. Appendix D.1 gives the geometries, parameter distributions, and simulations.

Each data set has 10000 training pairs, 100 validation pairs, and 100 test pairs. We fit every model on the first $N \in \{ 5 0 0 , 1 0 0 0 , 5 0 0 0 , 1 0 0 0 0 \}$ training pairs, so the training sets are nested, and the validation and test pairs are the same for every �. The validation and test pairs were also used in our previous study and during the development of the method, so our comparisons are retrospective. Freezing all choices before the final test runs prevents test-based selection in this study but does not make the test set new. Our error measure, the test error, is the relative velocity error $\| \widetilde { \mathbf { v } } ^ { \star } - \mathbf { v } ^ { \star } \| _ { 2 } / \| \mathbf { v } ^ { \star } \| _ { 2 }$ of the prediction $\tilde { \mathbf { v } } ^ { \star }$ of a test field $\mathbf { v } ^ { \star }$ at the output points, averaged over the 100 test fields (and, for Taylor–Green vortices (spacetime), over the four times). It measures the prediction for inputs that are used neither for fitting nor for validation.

The benchmarks difer less in their flows than in their inputs, and this afects how we interpret the results. In three of them, forced isotropic turbulence, turbulent species transport, and merging vortices, the inputs vary in at least two independent directions, and the operator is a multivariate map. In the other seven, the training inputs have numerical rank one. These inputs are multiples of one input function (e.g., the speed of the lid or the amplitude of the inflow), so the operator restricted to the data is a curve in the output space, and the operator regression is one-dimensional. Two of these, the Taylor–Green vortices, are degenerate in a further sense. Their outputs are proportional to the input amplitude, so we divide the responses by the amplitude before fitting and multiply by it after prediction. With $\mathcal { A } _ { i } > 0$ the amplitude of training input � relative to the median training amplitude and (�) that of a new input, we fit the normalized responses $\mathbf { v } _ { i } / \mathcal { A } _ { i }$ and predict

$$
\widetilde { \mathbf { v } } ( \mathbf { a } ) = \mathcal { A } ( \mathbf { a } ) \sum _ { i = 1 } ^ { N } \ell _ { i } ( \mathbf { a } ) \frac { \mathbf { v } _ { i } } { \mathcal { A } _ { i } } .\tag{72}
$$

The normalized operator is then constant, which afine augmentation reproduces, and Proposition 2 applies to the normalized responses. All four models use this normalization for both Taylor–Green data sets, and the other eight data sets use unnormalized responses. Taylor–Green vortices (spacetime) thus tests our spacetime output reconstruction but not the operator regression.

We compare four models (Section 3.6). The fixed-shape kernel uses $D _ { \parallel } = I$ and ${ \alpha _ { \perp } } = 1$ in (32), an isotropic metric on the principal coordinates and a unit scale for the residual coordinates. The fitted-shape kernel has its shape fitted by the restricted likelihood. Each is used with and without afine augmentation, and each chooses its support radius and regularization by validation. For inputs of rank one, the shape has no free parameters, so the fitted-shape kernel difers from the fixed-shape one only through the regularization that the likelihood proposes, which enters the validation search.

All four models share one local output reconstruction, with divergence-free polynomials of degree $\varpi \ = \ 4$ (dimension 20 in 2D and 85 in 3D) on stencils of $n _ { s } = 2 1$ and 59 nodes, the cubic PHS for the smooth flows, and the multiscale Matérn kernel with $c _ { \sigma } = 1 6$ (which sets its lengths $\sigma _ { j , \kappa } = c _ { \sigma } \rho _ { j } / 2 ^ { \kappa }$ relative to the stencil radius $\rho _ { j }$ in (21)) for the two turbulent flows. Section 5.1.4 compares these choices with alternatives. All computations run in double precision on one NVIDIA RTX 5090.

## 5.2.2. Sparsity and convergence in �

We first ask whether a sparse kernel matrix is compatible with accurate prediction. Validation chooses the support radius, and with it the density $d _ { K } = \mathrm { n n z } ( K _ { U U } ) / N ^ { 2 }$ of the kernel matrix, among candidates with densities up to 90%. If accuracy required a dense matrix, validation would select the largest candidates. Figure 6 (right) shows the densities selected at $N = 1 0 0 0 0$ for all ten benchmarks and all four models of Section 5.2.1. Validation selects 10% for the lid-driven cavity, both cylinder flows, and the buoyancy-driven cavity, about 33% for the backward-facing step, and between 9% and 25% for turbulent species transport and merging vortices. Nearly dense matrices are selected only for forced isotropic turbulence and for Taylor–Green without augmentation. The value 10% is the sparsest of the five prescribed densities that validation evaluates, although its gradient refinement can also reach sparser matrices.

Figure 7 shows how the selected density changes with �. For most benchmarks, it decreases as � grows, from between 20% and 90% at � = 500 to 10% at $N = 1 0 0 0 0$ for the lid-driven cavity and both cylinder flows, so the sparsity of the operator regression increases with the size of the training set. Forced isotropic turbulence stays at 90% for every �, and the backward-facing step at about 33%.

With these sparse matrices, the test errors decrease steadily as the training set grows. Figure 8 shows the errors for the three multivariate flows and for the lid-driven cavity, a representative flow with rank-one inputs. Least-squares fits give errors proportional to $N ^ { - \tau }$ with � between 0.4 and 1.3 for every benchmark and model (Table 9 in Appendix D.6), with two exceptions. With augmentation, the backward-facing step decays only like $N ^ { - 0 . 1 5 }$ . For Taylor–Green, the augmented models reproduce the constant normalized operator to rounding error, while the error of the unaugmented models grows slowly with �.

![](images/b63dd923a733140776eac76a251b1066c503287bfc55ff190d334f96e9602caa.jpg)  
Figure 6: All benchmarks at $N = 1 0 0 0 0$ . Left: test error of the fixed-shape kernel $\left( D _ { \parallel } = I , \alpha _ { \perp } = 1 \right)$ without augmentation, divided by the test error of each other model (values above one are improvements). Right: density $d _ { K }$ of the selected kernel matrix.

![](images/dcffaf88b4d7b5f855f763d1100e2b9a447c5ca28dea774e8c4459726cf7b5e9.jpg)  
Figure 7: Selected density $d _ { K }$ of the kernel matrix against the number � of training functions for the three multivariate flows and the lid-driven cavity. The afine columns are not counted. Figure 21 shows the other benchmarks.

## 5.2.3. Kernel shape and afine augmentation

We next compare the four models. Figure 6 (left) shows, for each benchmark at � = 10000, the error of the fixed-shape kernel without augmentation divided by the error of each other model. On the five rank-one flows other than Taylor–Green, these ratios lie between 0.98 and 1.36, so neither the fitted shape nor the augmentation changes the error by more than about a quarter. This is consistent with the operator regression being one-dimensional on these data.

On the multivariate flows, however, the ratios are larger and depend on the flow. For forced isotropic turbulence, augmentation reduces the error by a factor of 29, and by 39 together with the fitted shape, at the same density. For turbulent species transport, the fitted shape reduces it by a factor of 6, and by 12 together with augmentation, while doubling the selected density. For merging vortices, all four errors lie within 12% of each other. Augmentation also lowers the selected density for merging vortices and for both Taylor–Green benchmarks. Neither choice is uniformly better, so we select both by validation.

## 5.2.4. Comparison with earlier results

We now compare our models with the results that Sharma et al. [53] report on the same data sets for dense PPKM, for the kernel method of Batlle et al. [2] (VKM), which maps the samples of the input to the samples of the output, and for two neural operators, Geo-FNO [32] and Transolver [70], each trained with and without a divergence penalty. Table 4 lists these errors at the training sizes used there, together with the error of our model that has the smallest validation error among the four. The comparison has two caveats. The earlier errors are averages over all 200 held-out pairs, which include our 100 validation pairs, and the neural operators were trained in single precision.

Our models are at least as accurate as dense PPKM on all ten data sets, by factors between 1.06 (cylinder flow with vortex shedding) and about 56 (merging vortices), and the same holds against VKM. Against the better of the two neural operators, the factor ranges from 2 for forced isotropic turbulence to more than $1 0 ^ { 6 }$ for Taylor–Green. The gains over dense PPKM are likely due to the afine augmentation, the fitted kernel shape, and the validation of the radius

![](images/c3e341df61a1f9a5a6183c287a91530ab840c7f66ef4c5fec400090fb9e4369a.jpg)

![](images/7819b2193455ad5b6bc6a470ec3f1f4331891c9018a4945e86b39462dbf9861a.jpg)

![](images/b81159bd8da83f45f2686672c15eb37fd4cb8da207ae3970968619cabc9dcaa9.jpg)

![](images/fda653551771b42107d06ea6861c2f48d3e90ba778979961b11d11b741d8a5e2.jpg)  
training functions N  
Figure 8: Test error (mean relative velocity error of the predictions for the 100 test inputs) against the number � of training functions for the three multivariate flows and the lid-driven cavity. Dashed: with afine augmentation. Figure 20 shows the other benchmarks

## Table 4

Test error (mean relative velocity error of the predictions for held-out inputs) of our sparse local model (the one of the four with the smallest validation error) and of the methods reported by Sharma et al. [53] at the same training size �: dense PPKM, the kernel method of Batlle et al. [2] on output samples (VKM), and Geo-FNO and Transolver, each with the better of its errors with and without a divergence penalty. The earlier errors are averages over all 200 held-out pairs and ours over the 100 test pairs. Our models of the cylinder flow without shedding at $N = 1 0 0$ were fitted for this comparison, with the settings of the other training sizes. Bold: smallest test error in each row.
<table><tr><td>Data set</td><td>N</td><td>Ours</td><td>PPKM</td><td>VKM</td><td></td><td>Geo-FNO Transolver</td></tr><tr><td>Forced isotropic turbulence</td><td>10000</td><td> ${ \bf 7 . 1 \times 1 0 ^ { - 5 } }$ </td><td> $3 . 6 \times 1 0 ^ { - 4 }$ </td><td> $1 . 6 \times 1 0 ^ { - 4 }$ </td><td> $1 . 5 \times 1 0 ^ { - 4 }$ </td><td> $1 . 2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Turbulent species transport</td><td>10000</td><td> $\mathbf { 3 . 7 \times 1 0 ^ { - 5 } }$ </td><td> $2 . 4 \times 1 0 ^ { - 4 }$ </td><td> $2 . 4 \times 1 0 ^ { - 4 }$ </td><td> $5 . 0 \times 1 0 ^ { - 4 }$ </td><td> $2 . 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Merging vortices</td><td>500</td><td> $\mathbf { 1 . 9 \times 1 0 ^ { - 6 } }$ </td><td> $1 . 1 \times 1 0 ^ { - 4 }$ </td><td> $1 . 1 \times 1 0 ^ { - 4 }$ </td><td> $1 . 1 \times 1 0 ^ { - 3 }$ </td><td> $3 . 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Lid-driven cavity</td><td>10000</td><td> $\mathbf { 4 . 7 \times 1 0 ^ { - 7 } }$ </td><td> $1 . 4 \times 1 0 ^ { - 6 }$ </td><td> $1 . 5 \times 1 0 ^ { - 6 }$ </td><td> $1 . 5 \times 1 0 ^ { - 4 }$ </td><td> $2 . 1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Cylinder flow (no shedding)</td><td>100</td><td> $\mathbf { 2 . 8 \times 1 0 ^ { - 5 } }$ </td><td> $4 . 0 \times 1 0 ^ { - 5 }$ </td><td> $5 . 3 \times 1 0 ^ { - 5 }$ </td><td> $1 . 6 \times 1 0 ^ { - 4 }$ </td><td> $5 . 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Cylinder flow (vortex shedding)</td><td>10000</td><td> $\mathbf { 3 . 0 \times 1 0 ^ { - 7 } }$ </td><td> $3 . 1 \times 1 0 ^ { - 7 }$ </td><td> $2 . 3 \times 1 0 ^ { - 6 }$ </td><td> $3 . 7 \times 1 0 ^ { - 5 }$ </td><td> $1 . 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Backward-facing step</td><td>500</td><td> $\mathbf { 9 . 5 \times 1 0 ^ { - 8 } }$ </td><td> $1 . 5 \times 1 0 ^ { - 6 }$ </td><td> $2 . 9 \times 1 0 ^ { - 6 }$ </td><td> $9 . 3 \times 1 0 ^ { - 5 }$ </td><td> $7 . 2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Buoyancy-driven cavity</td><td>10000</td><td> $\mathbf { 4 . 5 \times 1 0 ^ { - 6 } }$ </td><td> $6 . 3 \times 1 0 ^ { - 6 }$ </td><td> $6 . 1 \times 1 0 ^ { - 6 }$ </td><td> $1 . 4 \times 1 0 ^ { - 4 }$ </td><td> $1 . 9 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Taylor-Green vortices (space)</td><td>5000</td><td> ${ \bf 6 . 0 \times 1 0 ^ { - 1 1 } }$ </td><td> $8 . 8 \times 1 0 ^ { - 1 0 }$ </td><td> $1 . 7 \times 1 0 ^ { - 9 }$ </td><td> $4 . 2 \times 1 0 ^ { - 4 }$ </td><td> $6 . 1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Taylor-Green vortices (spacetime)</td><td>5000</td><td> $\mathbf { 5 . 4 \times 1 0 ^ { - 1 1 } }$ </td><td> $8 . 7 \times 1 0 ^ { - 1 0 }$ </td><td> $8 . 7 \times 1 0 ^ { - 1 0 }$ </td><td> $7 . 8 \times 1 0 ^ { - 4 }$ </td><td> $2 . 2 \times 1 0 ^ { - 4 }$ </td></tr></table>

![](images/ae32a5536c6d6f62e16913ecb15011e6906c4eea38ef21550d2767f8bd95c3ff.jpg)

![](images/89e3a8ced781615db5544de23ff70a3137485cf98afe17edeb6e90b453c7433a.jpg)

![](images/f5cd0c91a6cb4a52b137879d288ed4fb2ddfbae43f0d1f1af6f78530d6b0f3a9.jpg)

![](images/dbb5d94a0361548264f39cc8b739d049c46ac0705fc936198e6ce645160c1824.jpg)  
training functions N  
Figure 9: Accounted storage of the fitted models (stored arrays and factors, not measured peak memory) against � for the three multivariate flows and the lid-driven cavity. Figure 22 shows the other benchmarks.

and regularization, and for Taylor–Green also to the amplitude normalization. Thus, at these training sizes, the sparse operator regression is no less accurate than the dense one.

In addition, the neural-operator predictions in Sharma et al. [53] have maximum pointwise divergences between about 0.1 and 20, while ours are divergence free on each stencil. The training times reported there, from seconds to about a minute and a half for the kernel methods with fixed shape parameters and from minutes to almost a day for the neural operators, are not comparable to our selection times, which include the search over radii and regularizations (Appendix D.6.1).

## 5.2.5. Storage

The storage of a model consists mainly of the Cholesky factor of its kernel matrix and the sampled training fields, which the prediction (59) uses in place of their local coeficients. Figure 9 shows the storage against �. Over our range of �, the training fields dominate, and the storage of most models grows roughly linearly in �, with least-squares exponents between 0.9 and 1.1. The exceptions are merging vortices, the backward-facing step, and the unaugmented Taylor–Green models, with exponents between 1.2 and 1.7, likely because the factor is a larger share of their storage (small outputs, or densities that do not decrease). The Cholesky factor itself grows like $N ^ { \bar { 1 . 4 } }$ to $N ^ { 2 . 4 }$ , since even a decreasing density leaves a number of nonzeros that grows faster than �, and it would eventually dominate for larger training sets.

The size of the factor depends on the ordering as well as on the density. With the nested-dissection ordering of Section 3.5, the factors at � = 10000 have between 1 and 4.3 times fewer entries (a median of 2.8) than with the approximate-minimum-degree ordering, with identical predictions. Forced isotropic turbulence gains nothing, since its nearly dense kernel matrix has a dense factor in any ordering. The fill also makes storage not proportional to the density. For example, the fitted-shape augmented species model has 2.4 times the density of the fixed-shape augmented one, but only 1.4 times as many factor entries. For this reason, we report storage separately from density (Table 11).

![](images/84caa7a2f343ac52a3a6dfc20050fb86cbb0b983b55abb31537ba54a55be1289.jpg)  
Figure 10: Median time to predict all 100 test fields after setup, including the kernel evaluation, the sparse solves, and the local output reconstruction. Fitting and setup are excluded (Appendix D.6.1).

## 5.3. Operator Lagrange weights

We now turn to the operator Lagrange weights of Section 4. Their main purpose is computational. They make a prediction cost one sparse solve per new input, regardless of the size of the output representation, and they let us combine the training fields before the output reconstruction. We first measure these savings. We then test the prediction of Theorem 3 that the weights are also more accurate than direct evaluation, but only when the output reconstruction amplifies solve errors.

## 5.3.1. Storage of the training data

Since the weights are applied to the sampled training fields, a prediction needs the $N n _ { v }$ sampled values, with $n _ { v } = d m .$ , and not the $N \textstyle \sum _ { j } r _ { j }$ local coeficients of the training fields. With one stencil per output point, the ratio of the two is $\begin{array} { r } { \sum _ { j } r _ { j } / ( d m ) = \tilde { n _ { s } } + n _ { P } / d } \end{array}$ when every stencil keeps the full polynomial space, of dimension $n _ { P } .$ . For our degree-four output reconstructions, the local coeficients would thus take 31 times the storage of the training fields in 2D and 87 times in 3D. For forced isotropic turbulence at � = 10000, for example, this is the diference between 1.7 and 147 GB in double precision.

## 5.3.2. Prediction cost

Figure 10 shows the time to predict all 100 test fields once the model is set up, including the evaluation of the kernel, the sparse solves for the weights, and the local output reconstruction. For the lid-driven cavity, a twentyfold increase in � raises this time seven- to eightfold, so prediction grows sublinearly in � over this range. The turbulent flows cost more at $N = 1 0 0 0 0$ , about three times the time of the lid-driven cavity for forced isotropic turbulence and 1.7–2.0 times for turbulent species transport, since they select denser kernels with more fill.

The ordering also afects the prediction time. Nested dissection makes the median prediction 1.3 times faster at � = 10000 than approximate minimum degree, since its smaller factors also give more parallel triangular solves. Fitting the kernel shape has no consistent efect on prediction time, but it adds ofline cost, since a fitted-shape mode takes several times longer to select than a fixed-shape one (Appendix D.6.1).

Amplification factor $\alpha _ { \mathrm { s p a t i a l } } = \| { B } \| _ { F } \| ( \mathbf { A } ^ { \star } ) ^ { \top } \| _ { F } / \| { B } ( \mathbf { A } ^ { \star } ) ^ { \top } \| _ { F }$ of (66), which measures the cancellation among the spatial coeficients, for $N = 1 \dot { 0 } 0 0$ training fields, for the global and for our local output reconstruction (stencils combined as in Section 5.3), and the largest ratio of the direct to the Lagrange error with global output reconstruction from Table 6. Bold: smaller amplification factor in each row.
<table><tr><td>Data set</td><td>Global  $\alpha _ { \mathrm { s p a t i a l } }$ </td><td>Local  $\alpha _ { \mathrm { s p a t i a l } }$ </td><td>Largest gain  $\left( { \mathsf { g l o b a l } } \right)$ </td></tr><tr><td>Forced isotropic turbulence</td><td> $3 . 7 \times 1 0 ^ { 9 }$ </td><td>396</td><td>1.2</td></tr><tr><td>Turbulent species transport</td><td> $2 . 9 \times 1 0 ^ { 9 }$ </td><td>9882</td><td>7.1</td></tr><tr><td>Merging vortices</td><td> $4 . 8 \times 1 0 ^ { 9 }$ </td><td>73</td><td>868.1</td></tr><tr><td>Lid-driven cavity</td><td> $5 . 8 \times 1 0 ^ { 9 }$ </td><td>13</td><td>34.5</td></tr><tr><td>Cylinder flow (no shedding)</td><td> $5 . 1 \times 1 0 ^ { 7 }$ </td><td>149</td><td>4.8</td></tr><tr><td>Cylinder flow (vortex shedding)</td><td> $5 . 2 \times 1 0 ^ { 7 }$ </td><td>149</td><td>2.0</td></tr><tr><td>Backward-facing step</td><td> $5 . 3 \times 1 0 ^ { 8 }$ </td><td>38</td><td>215.3</td></tr><tr><td>Buoyancy-driven cavity</td><td> $5 . 9 \times 1 0 ^ { 9 }$ </td><td>6</td><td>14.8</td></tr><tr><td>Taylor-Green vortices (space)</td><td> $1 . 6 \times 1 0 ^ { 7 }$ </td><td>73</td><td>6.1</td></tr><tr><td>Taylor-Green vortices (spacetime)</td><td> $1 . 9 \times 1 0 ^ { 2 }$ </td><td>145</td><td>1.0</td></tr></table>

## 5.3.3. Accuracy ofdirect and Lagrange evaluation

Recall the amplification factor $\alpha _ { \mathrm { s p a t i a l } } = \| { B } \| _ { F } \| ( \mathbf { A } ^ { \star } ) ^ { \top } \| _ { F } / \| { B } ( \mathbf { A } ^ { \star } ) ^ { \top } \| _ { F }$ of (66), which compares the size of the spatial coeficients � of the training fields and of the evaluation matrix $\mathbf { A } ^ { \star }$ with the size of the reconstructed velocities. It is large when large coeficients cancel in the output reconstruction. By (67), a solve error enters direct evaluation multiplied by $\alpha _ { \mathrm { s p a t i a l } } .$ , and Lagrange evaluation without this factor. The theorem therefore predicts a gain from Lagrange evaluation where $\alpha _ { \mathrm { s p a t i a l } }$ is large, and none where it is small. We test this prediction in two steps.

First, we compute $\alpha _ { \mathrm { s p a t i a l } }$ for all ten benchmarks, from 1000 training fields, for the global divergence-free Matérn output reconstruction of our earlier work and for our local output reconstruction. For the local output reconstruction, which has one $B _ { j }$ and one $\mathbf { A } _ { j } ^ { \star }$ per stencil, we combine the stencils as $\begin{array} { r l } {  { ( \sum _ { j } \| B _ { j } \| _ { F } ^ { 2 } \| \mathbf { A } _ { j } ^ { \star } \| _ { F } ^ { 2 } ) ^ { 1 / 2 } / \| B ( \mathbf { A } ^ { \star } ) ^ { \top } \| _ { F } } } \end{array}$ , which bounds the amplification of the solve errors summed over all stencils and reduces to $\alpha _ { \mathrm { s p a t i a l } }$ for a single global stencil. Table 5 shows that the coeficients of the global output reconstructions are large and cancel, with $\alpha _ { \mathrm { s p a t i a l } }$ above $1 0 ^ { 7 }$ up to $6 \times 1 0 ^ { 9 }$ , on all benchmarks except Taylor–Green vortices (spacetime), whose kernel is less flat. Our local output reconstructions have $\alpha _ { \mathrm { s p a t i a l } }$ between 6 and 150 in 2D, 396 for forced isotropic turbulence, and about $1 0 ^ { 4 }$ for turbulent species transport, where more than half the stencils have a truncated polynomial rank (Table 7).

Second, we evaluate the same operator models both ways, with everything else held fixed. These are the unaugmented models of Section 5.2, which the theorem covers: the fixed-shape kernel at the radii of the five prescribed densities and at its validated radius, and the fitted-shape kernel at its validated parameters, for � from 500 to 10000 (Appendix D.4). Table 6 lists the ratio of the direct error to the Lagrange error on all ten benchmarks.

The results agree with this prediction. With global output reconstruction, Lagrange evaluation is more accurate by up to a factor of 868 (merging vortices), with median gains above 5 for merging vortices, the lid-driven cavity, and the backward-facing step. With our local output reconstruction, the two evaluations agree to within a relative diference o $3 \times 1 0 ^ { - 6 }$ for every model, as the small $\alpha _ { \mathrm { s p a t i a l } }$ implies. A large $\alpha _ { \mathrm { s p a t i a l } }$ allows a gain but does not guarantee one, since the gain also requires a solve residual that is not negligible. Forced isotropic turbulence, for example, ha $\alpha _ { \mathrm { s p a t i a l } } \approx 4 \times 1 0 ^ { 9 }$ with its global output reconstruction but a gain of at most 1.2.

Over all 560 pairs, Lagrange evaluation is never less accurate than direct evaluation by more than 0.02%. Appendix D.4 follows one global model of the lid-driven cavity as � grows (Table 8). There, the gain grows from 1.3 to 5.1 together with the norm of the spatial coeficients, and solving the direct system for the sampled velocities instead of the coeficients removes it, which indicates that the large coeficients are the cause.

Lagrange evaluation thus serves our local output reconstruction through its cost and storage, and it would also protect a global or otherwise ill-conditioned output reconstruction, such as that of our earlier method, from the amplification of solve errors.

Ratio of the test error of direct evaluation to that of Lagrange evaluation, for the same unaugmented operator models (the fixed-shape kernel of Section 3.1 at the radii of the five prescribed densities and at its validated radius, and the fitted-shape kernel at its validated radius, for � = 500, 1000, 5000, and 10000) with global and local output reconstruction. Values above one favor Lagrange evaluation. Over all 560 pairs, the smallest ratio is 0.99987.
<table><tr><td rowspan="2">Data set</td><td colspan="3">Global output reconstruction</td><td colspan="2">Local output reconstruction</td></tr><tr><td>Models Median</td><td></td><td>Max</td><td>Models</td><td>Max |ratio-1|</td></tr><tr><td>Forced isotropic turbulence</td><td>28</td><td>1.00</td><td>1.2</td><td>28</td><td> $5 . 9 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>Turbulent species transport</td><td>28</td><td>1.00</td><td>7.1</td><td>28</td><td> $8 . 6 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Merging vortices</td><td>28</td><td>48.96</td><td>868.1</td><td>28</td><td> $4 . 2 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>Lid-driven cavity</td><td>28</td><td>5.07</td><td>34.5</td><td>28</td><td> $1 . 4 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Cylinder flow (no shedding)</td><td>28</td><td>1.05</td><td>4.8</td><td>28</td><td> $7 . 0 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>Cylinder flow (vortex shedding)</td><td>28</td><td>1.01</td><td>2.0</td><td>28</td><td> $2 . 4 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>Backward-facing step</td><td>28</td><td>16.51</td><td>215.3</td><td>28</td><td> $2 . 0 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>Buoyancy-driven cavity</td><td>28</td><td>1.47</td><td>14.8</td><td>28</td><td> $1 . 9 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>Taylor-Green vortices (space)</td><td>28</td><td>1.05</td><td>6.1</td><td>28</td><td> $1 . 7 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Taylor-Green vortices (spacetime)</td><td>28</td><td>1.00</td><td>1.0</td><td>28</td><td> $2 . 4 \times 1 0 ^ { - 6 }$ </td></tr></table>

## 6. Conclusions

In this article, we presented a local and sparse kernel method for learning the solution operators of incompressible flows. It keeps the guarantee of our earlier property-preserving method [53], that every prediction is divergence free, and removes the two costs of that method that grow with the problem size. Local divergence-free interpolants replace the global output reconstruction, so the output can be refined without enlarging any linear system, and the local systems are independent and solved in parallel on the GPU. The output reconstruction is divergence free, periodic on periodic domains, and, with the multiscale Matérn kernel, approaches Kolmogorov statistics under refinement at the separations that the nodes resolve. A compactly supported kernel that is positive semidefinite in any input dimension replaces the dense operator regression, so the kernel matrix is sparse. Its support radius, and with it the sparsity, is fitted to the data, and optional afine polynomials in the input, which the operator regression reproduces exactly, represent trends that a compactly supported kernel cannot. Operator Lagrange weights join the two, so each new input costs one sparse solve and the local coeficients of the training fields are never formed.

The local output reconstruction converges at close to the order of its polynomial degree on scattered nodes, is about as accurate between the nodes as the global output reconstruction (more accurate on six of ten benchmarks and less accurate on the other four) with systems that are smaller by about two orders of magnitude, is cheaper to set up than a global output reconstruction beyond a few thousand output points, with memory that grows linearly with their number, keeps the divergence and the periodic mismatch at rounding level, and, with the multiscale Matérn kernel, approaches prescribed Kolmogorov statistics at the separations that the nodes resolve.

In the operator benchmarks, the test errors decrease roughly like $N ^ { - \tau }$ with � between 0.4 and 1.3 in all but two degenerate cases, and the most accurate kernels are sparse on all benchmarks but one, with densities that mostly decrease as the training set grows. At the training sizes of our earlier study, these sparse models are at least as accurate as dense PPKM on every data set, and more accurate than the neural operators reported there by factors from 2 to more than $1 0 ^ { 6 }$ . Over our range of training sizes, the storage of most models grows roughly linearly in �. The benchmarks with rank-one inputs, apart from Taylor–Green, are insensitive to the choice of kernel and augmentation. The benchmarks with multivariate inputs are not, and the best choice difers between them. Afine augmentation gives the largest gain for forced isotropic turbulence, a fitted kernel shape for turbulent species transport, and neither helps for merging vortices. Our recommendation is therefore to select these choices by validation and to report error, density, and storage together.

Lagrange evaluation removes the need to store the local coeficients of the training fields, which would take $n _ { s } + n _ { P } / d$ times (the stencil size plus the number of polynomials per velocity component) the storage of the training fields themselves. Its accuracy agrees with our rounding analysis. With our local output reconstruction, whose amplification factors are at most about $1 0 ^ { 4 }$ , it is as accurate as direct evaluation. With global output reconstructions, whose amplification factors reach $6 \times 1 0 ^ { 9 }$ , it is up to nearly three orders of magnitude more accurate.

Several extensions are natural. First, benchmarks with richer input families, and a comparison with dense operator regression and with neural operators on identical test sets and hardware, would test these conclusions beyond our ten benchmarks. Second, our local interpolants are divergence free on each stencil but are not coupled across stencils. Blending them with a partition of unity would give a continuous field, but the blending weights introduce divergence. The divergence-free partition-of-unity method of Drake et al. [12] avoids this by blending potentials instead of velocities. Third, at a fixed density, the storage of the kernel matrix still grows quadratically with the number of training functions, which hierarchical or multilevel sparse factorizations could reduce. Finally, our rounding analysis covers the model without augmentation and evaluation through coeficients. Extending it to the afine correction and to evaluation before the output reconstruction would complete the analysis.

## Credit authorship contribution statement

Ramansh Sharma: conceptualization, methodology, software, investigation, validation, visualization, writing – original draft, review, editing.

Matthew Lowery: methodology, software

Houman Owhadi: conceptualization, methodology, supervision.

Varun Shankar: conceptualization, funding acquisition, methodology, project administration, resources, software, supervision, validation, visualization, writing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

Our implementation, including the scripts that build the tables and figures of this paper, is available in KernelPack-JAX at https: $: / / \mathrm { g } \mathrm { . }$ ithub.com/VarShankar/kernelpack-jax. The ten benchmark data sets are those of Sharma et al. [53] and are publicly available on Zenodo [54].

## Acknowledgements

RS, ML, and VS were supported by the National Science Foundation under DMS 2505986, Cooperative Agreement 2421782, and the Simons Foundation grant MPS-AI-00010515 awarded to the NSF-Simons AI Institute for Cosmic Origins — CosmicAI, https://www.cosmicai.org. VS was additionally supported by the Air Force Ofice of Scientific Research (AFOSR) under LRIR award FA9550-25-1-0042. HO acknowledges support from the Air Force Ofice of Scientific Research under MURI award number FOA-AFRL-AFOSR-2023-0004 (Mathematics of Digital Twins), the Department of Energy under award number DE-SC0023163 (SEA-CROGS: Scalable, Eficient, and Accelerated Causal Reasoning Operators, Graphs and Spikes for Earth and Embedded Systems) and the DoD Vannevar Bush Faculty Fellowship Program under ONR award number N00014-18-1-2363.

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work, the authors used both Codex (OpenAI) and Claude (Anthropic) to help clean up the manuscript and implement parts of the software. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

## A. Details of the local output reconstruction

## A.1. Ordering of vector coeficients

The coeficient vectors of (17) are ordered by component, with all values of one component before those of the next, but the kernel evaluation at a query point $y _ { q } ^ { \star }$ is naturally ordered by stencil node,

$$
\begin{array} { r } { \Phi _ { j ( q ) , q , \mathrm { p t } } ^ { \star } = \bigl [ \Phi _ { \mathrm { D F } } ( y _ { q } ^ { \star } , \tilde { y } _ { j ( q ) , 1 } ) \quad \cdots \quad \Phi _ { \mathrm { D F } } ( y _ { q } ^ { \star } , \tilde { y } _ { j ( q ) , n _ { s } } ) \bigr ] \in \mathbb { R } ^ { d \times d n _ { s } } . } \end{array}\tag{73}
$$

Retained divergence-free polynomial ranks of the local systems on the nodes of each benchmark, at degree four and with periodic images, from the construction above. The full ranks are 20 in 2D and 85 in 3D. Taylor–Green vortices (spacetime) has the stencils of Taylor–Green vortices (space).
<table><tr><td>Data set</td><td>Stencils</td><td>Retained rank</td><td>Truncated stencils</td></tr><tr><td>Turbulent species transport</td><td>7000</td><td>45-85</td><td>4338</td></tr><tr><td>Forced isotropic turbulence</td><td>7000</td><td>85</td><td>0</td></tr><tr><td>Buoyancy-driven cavity</td><td>5000</td><td>20</td><td>0</td></tr><tr><td>Lid-driven cavity</td><td>1000</td><td>20</td><td>0</td></tr><tr><td>Cylinder flow (no shedding)</td><td>1000</td><td>20</td><td>0</td></tr><tr><td>Cylinder flow (vortex shedding)</td><td>1000</td><td>20</td><td>0</td></tr><tr><td>Backward-facing step</td><td>1000</td><td>20</td><td>0</td></tr><tr><td>Taylor-Green vortices (space)</td><td>500</td><td>19-20</td><td>1</td></tr><tr><td>Merging vortices</td><td>500</td><td>19-20</td><td>1</td></tr></table>

Let $\Pi _ { d , n _ { s } }$ be the permutation with $\Pi _ { d , n _ { s } } \mathbf { b } _ { \Phi , \mathrm { c o m p } } = \mathbf { b } _ { \Phi , \mathrm { p t } }$ , where ${ \bf b } _ { \Phi , \mathrm { p t } }$ groups the � components of each node. The evaluation matrix in (18) is then

$$
\Phi _ { j ( q ) , q , \mathrm { c o m p } } ^ { \star } = \Phi _ { j ( q ) , q , \mathrm { p t } } ^ { \star } \Pi _ { d , n _ { s } } .\tag{74}
$$

## A.2. Polynomial basis and local solves on the GPU

We first remove the dependencies among the candidates that do not depend on the stencil, once, before sampling. A singular value decomposition of the fixed map from candidates to polynomial coeficients gives an orthonormal basis Ξ of its row space, shared by all stencils of one dimension and degree, with

$$
G _ { j } = G _ { j } \Xi \Xi ^ { \mathsf { T } } , \qquad \Xi ^ { \mathsf { T } } \Xi = I ,\tag{75}
$$

in exact arithmetic. Thus $G _ { j } \Xi$ has the nonzero singular values of $G _ { j } .$ , and we factor it by thin QR, $G _ { j } \Xi = \widetilde { Q } _ { j } \widetilde { R } _ { j }$ . On most stencils, $\widetilde { R } _ { j }$ is far from singular, and the polynomial basis is

$$
\mathbf { P } _ { j } ( y ) = G _ { j } ( y ) \boldsymbol { \Xi } \widetilde { R } _ { j } ^ { - 1 } ,\tag{76}
$$

whose values at the stencil nodes are the orthonormal columns of ${ \widetilde { Q } } _ { j } ,$ , as in (14). We use (76) when the eigenvalues of $\widetilde { R } _ { j } ^ { \intercal } \widetilde { R } _ { j }$ certify that its smallest singular value is at least $1 0 ^ { - 6 }$ times its largest, far above the rank threshold of $1 0 ^ { - 1 2 }$ since these eigenvalues carry absolute errors of the order of the unit roundof times the largest one. On the remaining stencils, the column-pivoted QR factorization $\widetilde { R } _ { j } \Pi _ { j } = \widehat { Q } _ { j } \widehat { R } _ { j }$ determines the rank as in (12), and the basis is (14) with $G _ { j , I }$ replaced by the retained columns of $G _ { j } \Xi \Pi _ { j } ^ { \phantom { } }$ . Both factorizations are batched over the stencils, and the columns of diferent stencils may difer, since every system is padded to the same width (below). A singular value decomposition on every stencil would decide the rank as well, but its batched form is slow on the GPU for these matrices $( 8 5 \times 8 5$ in 3D at degree four, about 10 ms each). On the species-transport node set, where most stencils are rank deficient, both rank decisions agree on every stencil, and the output reconstructions between the nodes agree to $1 0 ^ { - 1 2 }$

To keep the batches of fixed size, every system carries all 20 (2D) or 85 (3D) polynomial columns at degree four. The columns of discarded modes are set to zero and given a unit diagonal, which decouples their coeficients without changing the interpolant. Every system therefore has the padded width 62 (2D) or 262 (3D), and our costs and storage include the padding.

If the LU solution of a local system is not finite, we use the pseudoinverse of that system, also for its evaluation weights. The resulting field is still divergence free, but it need not interpolate or reproduce the full polynomial space. The pseudoinverse was never needed. The LU solutions are finite on every node stencil of all ten operator benchmarks, including the rank-deficient stencils of Table 7, and in the manufactured tests of Section 5.1 every stencil kept the full polynomial space.

## A.3. Evaluation away from the nodes

The operator benchmarks evaluate predictions at the nodes. To evaluate at another point, our GPU implementation selects a stencil centered at that point and builds its basis there. The comparisons of Table 2 use this rule. The

manufactured tests of Section 5.1 instead use the stencil of the nearest node, and their results between the nodes refer to that rule.

## A.4. Periodic stencils

The Taylor–Green flows and the merging vortices have period 2� in each direction. Forced isotropic turbulence has period 2� on the box $[ 0 , 2 \pi ] ^ { 3 }$ , which we supply explicitly, since the cell centers of its grid do not reach the faces of the box.

## A.5. Output reconstruction between the nodes

Table 2 reconstructs the test predictions at the nodes of the model with the smallest validation error among the four of Section 5.2.1, at $N = 1 0 0 0 0$ , so that the operator regression is that of Section 5.2 and only the output reconstruction changes. The evaluation points of Table 2 are all mesh or grid points of the data set that are not among the � output points. The local output reconstruction selects a stencil centered at each of them (Appendix A.3). The global output reconstruction is the divergence-free interpolant of Sharma et al. [53] with the scalar kernel $3 M _ { 5 / 2 } ( \varepsilon r )$ on all nodes, with the shape parameters of that work $( \varepsilon = 0 . 3$ for the lid-driven cavity, 0.0636 for both cylinder flows, 0.06 for the backward-facing step, 0.8 for the buoyancy-driven cavity, 0.02 for merging vortices and Taylor–Green vortices (space), and 1.5 for Taylor–Green vortices (spacetime)), applied at each sampled time for Taylor–Green vortices (spacetime). For the periodic data sets, it uses one representative of each set of periodically equivalent nodes.

We select the scale factor � of the multiscale kernel on the 100 validation fields, by reconstructing their exact nodal $c _ { \sigma }$ values and measuring the error between the nodes. For turbulent species transport, $c _ { \sigma } = 2 , 4 , 8 $ , and 16 give errors o 0.166, 0.158, 0.150, and 0.147, and the PHS gives 0.169. For forced isotropic turbulence, they give 0.0228, 0.0206, 0.0178, and 0.0165, and the PHS gives 0.0227. Both select $c _ { \sigma } = 1 6 $ , which the operator benchmarks use. For turbulent species transport, the error between the nodes remains much larger than at the nodes, so an accurate prediction at the nodes does not guarantee one everywhere.

We also compare the second-order structure functions

$$
S _ { 2 } ( r ) = \left. \left\| v ( x ) - v ( y ) \right\| _ { 2 } ^ { 2 } \right. _ { \| x - y \| \approx r }\tag{77}
$$

of the output reconstructions and of the reference fields on the same pairs of points between the nodes (256 anchors with up to four partners in each of 12 logarithmic distance bins, with periodic distances for forced isotropic turbulence), averaged over the 100 test fields (Figure 11). The relative discrepancy of the binned structure function is 1.7% for the PHS and 2.6% for the multiscale kernel for turbulent species transport, and 0.40% and 0.20% for forced isotropic turbulence. The multiscale kernel thus lowers the pointwise error between the nodes for both flows, but improves the binned increments only for forced isotropic turbulence. Since the reference fields come from Reynolds-averaged (RANS) and coarse large-eddy (LES) simulations, this comparison measures agreement with the reference, not with physical turbulence.

## B. Details of the operator regression

## B.1. Normalization and principal directions

At every �, we compute the normalization and the principal directions from the first � training inputs alone. Each coordinate is divided by its standard deviation, or set to zero if that vanishes, and the numerical rank uses the threshold max $\{ 1 0 ^ { - 1 0 } , 2 u _ { \mathrm { m a c h } } \operatorname* { m a x } ( N , n d _ { a } ) \}$ relative to the largest singular value, with $u _ { \mathrm { m a c h } }$ the unit roundof. The four models at one � share the normalization, the principal directions, and the gate (34), and the two models with augmentation share the afine matrix.

## B.2. Fitting and validation

The fitted-shape kernel starts from the radius $\varrho _ { 0 }$ that gives 25% density and from the regularization $\theta = 1 0 ^ { - 6 }$ , and takes 60 steps of the Adam optimizer [31] with learning rate 0.03 for each of $c _ { d } = 0$ and $c _ { d } = 1$ . Its log-radius is confined to $[ \log ( \varrho _ { 0 } / 8 ) , \log ( 4 \varrho _ { 0 } ) ]$ and its regularization to $\theta = \bar { e } ^ { t _ { \theta } } + 1 0 ^ { - 1 2 }$ with $t _ { \theta } \in [ \log ( 1 0 ^ { - 1 2 } )$ , log(0.1)]. The anisotropy uses $n _ { z } - 1$ free parameters $b _ { k } \in [ - 2 , 2 ]$ , completed by a zero, through $\tilde { \upsilon } _ { k } = \log ( 2 ) \operatorname { t a n h } ( b _ { k } )$ and $\begin{array} { r } { \upsilon _ { k } = \tilde { \upsilon } _ { k } - n _ { z } ^ { - 1 } \sum _ { k ^ { \prime } = 1 } ^ { n _ { z } } \tilde { \upsilon } _ { k ^ { \prime } } } \end{array}$ The residual scale lies in $[ 1 / 4 , 4 ]$ . It is inactive for the seven benchmarks with $n _ { z } = 1$ and stays at one for turbulent species transport.

![](images/1a71894740c3023df04b354315aff40f069d034f9332e0313ca30a4df272cf20.jpg)

![](images/cf4e3182d205c0f7a17239a8c1c86a26d2e65ff5b9ca4a2a42b3de272e22f66c.jpg)  
Figure 11: Second-order structure functions between the nodes of the reference fields and of the local PHS and multiscale Matérn output reconstructions of the test predictions at $N = 1 0 0 0 0$

Validation evaluates the radii of the five density budgets of Section 3.6. With $\varrho _ { \mathrm { m a x } }$ the largest of these, it also searches radii in $[ \varrho _ { \mathrm { m a x } } / 8 , \varrho _ { \mathrm { m a x } } ]$ from four starts (radii $\varrho _ { \mathrm { m a x } } / 2$ and $\varrho _ { \mathrm { m a x } } / 4$ , regularizations $1 0 ^ { - 6 }$ and $1 0 ^ { - 3 } )$ , each with 35 Adam steps on all 100 validation fields. A fitted-shape model repeats this for both of its fitted shapes, so its search is twice as long. We keep the model with the smallest validation error among those with density at most 0.90, and break ties by density.

## B.3. Work, storage, and timing

We assemble the kernel matrix by visiting all pairs of training inputs in blocks of bounded size and keeping those within the support. This keeps the memory bounded, but the work of the search is still quadratic in �, since compact support alone does not give a faster neighbor search. The likelihood and its gradient use the factors through

$$
\Pi _ { U } = { L _ { K } ^ { - \top } } ( I - Q _ { P } Q _ { P } ^ { \top } ) L _ { K } ^ { - 1 } ,\tag{78}
$$

with the entries of the inverse needed for the trace derivatives computed only on the support. When the outputs are wide, we replace � by the triangular factor of a thin QR factorization $E ^ { \mathsf { T } } = Q _ { E } R _ { E } .$ , since $\mathbf { \bar { \Sigma } } _ { E E } \mathsf { T } = R _ { E } ^ { \mathsf { T } } R _ { E }$

The storage we report is the sum of the stored model arrays (including the training fields), the sparse matrix arrays, cuDSS’s estimate of the memory of its factors, and the solve bufers. It is computed from the stored arrays instead of measured at its peak, and it excludes compiled code and temporary workspace. A kernel with $\mathcal { N } _ { K }$ nonzeros in compressed sparse row form takes $1 2 \mathcal { N } _ { K } + 4 ( N + 1 )$ bytes.

We report the setup of the output reconstruction, which all four models share, separately from the fitting and validation of each model. The warm prediction time of Figure 10 is the median of 11 repetitions of predicting all 100 test fields after the model is set up and compiled. The four models of a benchmark run in one process and share compiled code, so small diferences between their times are not significant.

## B.4. Batches and checks

The 100 test inputs are solved in four batches of 32 right-hand sides, the last one padded by repeated inputs. Repeated solves replay CUDA graphs (recorded sequences of GPU kernel launches). After every factorization, we check the residual of the solves, the reproduction of afine functions, and the agreement of the batched and compiled predictions with reference predictions. Every reported model passed these checks.

## C. Proofs

## C.1. Completeness of the polynomial candidates

In 2D, every divergence-free vector polynomial of degree at most � has a polynomial stream function of degree at most $\varpi + 1$ , so the rotated gradients of the scalar basis span the space. In 3D, split a divergence-free polynomial into

homogeneous parts $v _ { j }$ of degree $j ,$ each of which is divergence free. The identity

$$
\nabla _ { \xi } \times ( v _ { j } \times \xi ) = ( j + 2 ) v _ { j }\tag{79}
$$

shows that $v _ { j }$ is the curl of the potential $( v _ { j } \times \xi ) / ( j + 2 )$ of degree � + 1, and expanding the components of this potential in the scalar basis gives a combination of the candidates (11). For the dimension, the divergence maps the vector polynomials of degree at most � onto the scalar polynomials of degree at most $\varpi - 1$ (integrate in one coordinate to find a preimage), and rank–nullity gives $d \binom { \varpi + d } { d } - \binom { \overline { { \varpi + d - 1 } } } { d }$ .

## C.2. Positive definiteness of the kernel

ProofofProposition 1. The scaled principal coordinates $D _ { \parallel } z$ lie in $\mathbb { R } ^ { n _ { z } }$ with $n _ { z } \ \leq \ 3$ , so the Wendland factor is a positive semidefinite kernel. The Matérn factor is positive semidefinite on the residual space in every finite dimension. The Gram matrix of their product is the Schur product of two positive semidefinite matrices, and multiplying it on both sides by the diagonal matrix of gates keeps it positive semidefinite.

For strictness, group distinct standardized inputs by their distinct principal coordinates $z ^ { ( 1 ) } , \dots , z ^ { ( J ) }$ , and let $W$ be the Wendland Gram matrix on these, which is positive definite. Let $\Psi _ { \perp }$ be a feature map of the residual Matérn kernel, and for coeficients $c _ { i }$ let

$$
f _ { \iota } = \sum _ { \iota : z _ { i } = z ^ { ( \iota ) } } c _ { i } \eta ( { \bf a } _ { i } ) \Psi _ { \bot } ( \zeta _ { i } ) .\tag{80}
$$

The quadratic form of the product Gram matrix is $\begin{array} { r } { \sum _ { \iota , \iota ^ { \prime } } W _ { \iota \iota ^ { \prime } } \langle f _ { \iota } , f _ { \iota ^ { \prime } } \rangle \geq \lambda _ { \operatorname* { m i n } } ( W ) \sum _ { \iota } \| f _ { \iota } \| ^ { 2 } } \end{array}$ . Inputs in the same group difer in their residuals, so strict positive definiteness of the Matérn kernel gives $f _ { \iota } \neq 0$ whenever a coeficient of the group is nonzero and the gates are positive. (If the residual space is trivial, distinct inputs lie in distinct groups.) Finally, $c ^ { \mathsf { T } } ( K _ { U U } + \theta I ) c \geq \theta \| c \| _ { \gamma } ^ { 2 } > 0$ for every $c \neq 0$ and $\theta > 0$ □

The proof does not assume that the inputs lie in a low-dimensional afine subspace. Note that standardization can make two inputs equal if they difer only in coordinates with zero training variance. The regularization keeps the matrix positive definite in that case.

## C.3. Variational form of the operator regression

Let $f _ { \mathcal { H } } \in \mathcal { H }$ , and split $f _ { \mathcal { H } } = f _ { \| } + f _ { \perp }$ , where $f _ { \parallel }$ lies in the span of the functions $\lambda ( \cdot , { \bf a } _ { i } ) { \bf e } _ { k }$ and $f _ { \perp }$ is orthogonal to it. By the reproducing property, $f _ { \perp } ( { \bf a } _ { i } ) = 0$ for every �, so $f _ { \perp }$ leaves the data term of (42) unchanged and only increases the norm. A minimizer therefore has the form $\begin{array} { r } { f _ { \mathcal { H } } = \sum _ { i } \lambda ( \cdot , { \bf a } _ { i } ) { \bf c } _ { i } } \end{array}$ , with $\| f _ { \mathcal { H } } \| _ { \mathcal { H } } ^ { 2 } = \mathrm { t r } ( C ^ { \top } K _ { U U } C )$ , where � has the rows $\mathbf { c } _ { i } ^ { \mathsf { T } }$ . With � the matrix with rows $\mathbf b _ { i } ^ { \top }$ and $P = P _ { U }$ , the objective becomes the convex quadratic

$$
J ( C , \beta ) = \| K _ { U U } C + P \beta - B \| _ { F } ^ { 2 } + \theta \operatorname { t r } ( C ^ { \mathsf { T } } K _ { U U } C ) ,\tag{81}
$$

with gradients

$$
\nabla _ { C } J = 2 K _ { U U } \big ( K _ { U U } C + P \beta - B + \theta C \big ) , \qquad \nabla _ { \beta } J = 2 P ^ { \top } \big ( K _ { U U } C + P \beta - B \big ) .\tag{82}
$$

The conditions (40) state that $K _ { U U } C + P \beta - B = - \theta C$ and $P ^ { \mathsf { T } } C = 0$ . Both gradients therefore vanish, and every solution of (40) minimizes �, and hence (42). Without augmentation, the same argument applies with � and $\beta$ removed.

## C.4. Block elimination and reproduction

ProofofProposition 2. Write $K = K _ { U U } ^ { \theta } = L _ { K } L _ { K } ^ { \mathsf { T } }$ and $P = P _ { U }$ . Since � has full column rank, $L _ { K } ^ { - 1 } P = Q _ { P } R _ { P }$ with $R _ { P }$ invertible, and

$$
\begin{array} { r } { \mathcal { E } _ { U } P = R _ { P } ^ { - 1 } Q _ { P } ^ { \top } L _ { K } ^ { - 1 } P = I , \qquad K \mathcal { E } _ { U } ^ { \top } = P R _ { P } ^ { - 1 } R _ { P } ^ { - \top } . } \end{array}\tag{83}
$$

Let $\ell _ { 0 } = K ^ { - 1 } k _ { U } ( { \mathbf a } )$ and $\tilde { p } = p ( \mathbf { a } ) - P ^ { \top } \ell _ { 0 }$ . Then $\ell = \ell _ { 0 } + \mathcal { E } _ { U } ^ { \top } \tilde { p }$ satisfies

$$
\begin{array} { r } { P ^ { \top } \ell = P ^ { \top } \ell _ { 0 } + \widetilde { p } = p ( \mathbf { a } ) , \qquad K \ell = k _ { U } ( \mathbf { a } ) + P R _ { P } ^ { - 1 } R _ { P } ^ { - \top } \widetilde { p } , } \end{array}\tag{84}
$$

so (51) holds with $\mu = - R _ { P } ^ { - 1 } R _ { P } ^ { - \top } \tilde { p } .$ . If $( v , \omega )$ solves the homogeneous system, then $P ^ { \tau } v = 0$ and $v ^ { \top } K v = 0$ , so $v = 0$ and full column rank gives $\omega = 0$ . The solution is unique.

For targets in the space of afine functions, $B _ { j } \ = \ P _ { U } T _ { j }$ , we get $B _ { j } ^ { \intercal } \ell = T _ { j } ^ { \intercal } p ( { \bf a } )$ , which is reproduction. The coeficients of (44) are

$$
\begin{array} { r } { C _ { j } = \Pi _ { U } B _ { j } , \qquad \beta _ { j } = \mathcal { E } _ { U } B _ { j } , \qquad k _ { U } ( \mathbf { a } ) ^ { \top } C _ { j } + p ( \mathbf { a } ) ^ { \top } \beta _ { j } = \ell ( \mathbf { a } ) ^ { \top } B _ { j } , } \end{array}\tag{85}
$$

where the last identity follows by transposing the augmented system. Direct and Lagrange evaluation thus agree in exact arithmetic. Finally, linearity of $\chi$ and $_ D$ gives (54). For a nonzero common constraint $\mathcal { D } \chi ( \mathbf { b } _ { i } ) = f$ , the afine model also gives $\begin{array} { r } { D \chi ( \sum _ { i } \ell _ { i } \mathbf { b } _ { i } ) = f } \end{array}$ , since its constant column enforces $\textstyle \sum _ { i } \ell _ { i } = 1$ □

Without augmentation, $\ell = ( K _ { _ { I I I I } } ^ { \theta } ) ^ { - 1 } k _ { U } ( \mathbf { a } )$ . If $\theta = 0 ,$ , then $k _ { U } ( \mathbf { a } _ { i } )$ is the �th column of $K _ { U U } , \mathrm { s o } \ \ell ( { \bf a } _ { i } ) = { \bf e } _ { i }$ and the weights are cardinal. With $\theta > 0 .$ , the vector $k _ { U }$ excludes the regularization, and the weights are no longer cardinal.

## C.5. Rounding errors of direct and Lagrange evaluation

ProofofTheorem 3. The first residual equation gives $\widehat { C } = ( K _ { { U U J } } ^ { \theta } ) ^ { - 1 } ( B - R _ { \mathrm { d i r } } )$ , and multiplying by $K _ { U _ { * } U }$ and $( \mathbf { A } ^ { \star } ) ^ { \top }$ gives (62). The second gives $\widehat { L } = ( K _ { U _ { \ast } U } - R _ { \mathrm { L a g } } ) ( K _ { U U } ^ { \theta } ) ^ { - 1 }$ , and hence (63).

For the rounding, let $Z = \operatorname { f l } ( M _ { 1 } M _ { 2 } ) $ with $( M _ { 1 } , M _ { 2 } ) = ( K _ { U _ { \bot } U } , \widehat { C } )$ or $( \widehat { L } , B )$ . The standard bound for matrix products [28, Chapter 3] gives $| Z - M _ { 1 } M _ { 2 } | \leq \gamma _ { N } | M _ { 1 } | | M _ { 2 } |$ , and the product with $( \mathbf { A } ^ { \star } ) ^ { \top }$ satisfies $| \operatorname { f l } ( Z ( \mathbf { A } ^ { \star } ) ^ { \mathsf { T } } ) -$ $\bar { Z } ( \mathbf { A } ^ { \star } ) ^ { \top } \vert \leq \gamma _ { r + d } \vert \bar { Z } \vert \vert ( \mathbf { A } ^ { \star } ) ^ { \top } \vert$ , also when it is computed in component blocks (dot products of length at most � followed by at most $d - 1$ additions). Together,

$$
\Big | \operatorname { f l } ( \operatorname { f l } ( M _ { 1 } M _ { 2 } ) ( \mathbf { A } ^ { \star } ) ^ { \top } ) - M _ { 1 } M _ { 2 } ( \mathbf { A } ^ { \star } ) ^ { \top } \Big | \leq \gamma _ { \mathrm { m u l t } } | M _ { 1 } | | M _ { 2 } | | ( \mathbf { A } ^ { \star } ) ^ { \top } | .\tag{86}
$$

The triangle inequality and submultiplicativity of the Frobenius norm then give (64)–(65).

□

## D. Details of the numerical experiments

## D.1. Benchmarks

The ten data sets are those of our previous work [53], where their simulations are described in full. Each figure shows the first simulation of a data set. The color fields interpolate the simulation samples linearly (Appendix D.3).

## D.1.1. Flow past a cylinder

Two-dimensional flow passes a cylinder of radius 0.8 centered at (1.5, 7) in the channel $\Omega _ { a } = \Omega _ { v } = [ 0 , 2 0 ] \times [ 0 , 1 4 ]$ with an inflow on the left, freestream conditions at the top and bottom, no slip on the cylinder, and zero pressure at the outlet. The constant initial velocity varies within the vortex-shedding regime in one data set and below the onset of vortex shedding in the other. We learn $\mathcal { G } : \mathbf { u } ( x , 0 )  \mathbf { u } ( y , 1 0 ) \mathrm { o n } m = 1 0 0 0$ approximate Fekete points (well-spaced points selected from the mesh as in Sharma et al. [53]) of the 9520 mesh points.

## D.1.2. Taylor–Green vortices

The analytical Taylor–Green vortices decay on the periodic domain $\Omega _ { a } = \Omega _ { v } = [ 0 , 2 \pi ] ^ { 2 }$ , sampled at 500 Fekete points. For the spatial data set, we learn $\mathcal { G } : \mathbf { u } ( x , 0 )  \mathbf { u } ( y , 1 )$ ). For the spacetime data set, we learn the map from ${ \mathbf u } ( x , 0 )$ to the velocity on $\Omega \times [ 0 . 7 , 1 ]$ , with snapshots at $\mathcal { T } = \{ 0 . 7 , 0 . 8 , 0 . 9 , 1 . 0 \}$ and cubic interpolation in time (Section 2.4). Both use the amplitude normalization (72).

## D.1.3. Lid-driven cavity

In the cavity $\Omega _ { a } = \Omega _ { v } = [ 0 , 1 ] ^ { 2 }$ , the speed of the lid is drawn from a truncated normal distribution, and the fluid starts at rest, so the initial velocity is nonzero only on the lid. We learn $\mathcal { G } : \mathbf { u } ( x , 0 )  \mathbf { u } ( y , 5 )$ on $m = 1 0 0 0$ Fekete points selected from the 9566 mesh points.

## D.1.4. Backward-facing step

A parabolic inflow, scaled by a factor drawn from a truncated normal distribution, drives the flow over a step into the channel $\Omega _ { a } = \Omega _ { v } = [ 0 , 1 5 ] \times [ - 0 . 5 , 0 . 5 ]$ . We learn $\mathcal { G } : \mathbf { u } ( x , 0 )  \mathbf { u } ( y , 5 )$ on $m = 1 0 0 0$ Fekete points selected from the 7359 mesh points. The input has only 11 coordinates, the samples of the inflow profile.

![](images/10e97a1ae661e14052949bf946152f9a6a879cf4068168b0b6f658c9276e0d8a.jpg)  
Figure 12: Cylinder flow. (A) Initial speed. (B) Speed at � = 10, on the same color scale.

![](images/e52fdc7f2bf888fb6a6f2542f6ad2ab01de1375131ebad378c156be5fb4fad82.jpg)

(B) Speed at t = 1  
![](images/64d740de84331b0a2bf357e427cd6d6aa4a3dad2609134f6f16a6df3c3f1afde.jpg)  
Figure 13: Taylor–Green vortices. (A) Initial speed. (B) Speed at � = 1, on the same color scale.

## D.1.5. Buoyancy-driven cavity

In the cavity $\Omega _ { a } = \Omega _ { v } = [ 0 , 1 ] ^ { 2 }$ , an initial temperature field $T ( x , 0 )$ drives the flow by buoyancy. We learn the map from $T ( x , 0 )$ to the velocity at $t = 5$ on $m = 5 0 0 0$ Fekete points selected from the 10514 mesh points.

## D.1.6. Merging vortices

Two vortices merge on the periodic domain $\Omega _ { a } = \Omega _ { v } = [ 0 , 2 \pi ] ^ { 2 }$ , and the width of their cores varies across the data set. We learn $\mathcal { G } : \omega ( x , 0 )  \mathbf { u } ( y , 0 . 4 )$ from the initial vorticity on 500 Fekete points selected from the 7447 mesh points.

![](images/0a7906374a7796d62cf4964ad3296c08d2f54b206433239e4a4bc08ed27ce38b.jpg)  
Figure 14: Lid-driven cavity. (A) Initial speed. (B) Speed at � = 5, on the same color scale.

![](images/a55b16479bad2909934b3b26578df7c181243bfe7fc45772712dc09345018206.jpg)

![](images/7ace33e5847a19adb9203b8ab8d639e91f29cc81701b322b1b314ee484d4d805.jpg)  
Figure 15: Backward-facing step. (A) Inflow profile. (B) Speed at � = 5, with the transverse direction expanded.

## D.1.7. Turbulent species transport

Air and methane mix turbulently in a static Kenics mixer, whose three blades twist along the � axis, at Reynolds numbers between $1 0 ^ { 5 }$ and $1 0 ^ { 6 }$ . We learn the map from the inlet velocity (1388 samples) to the velocity at $t = 0 . 5$ on 7000 Fekete points selected from 11388 locations.

## D.1.8. Forced homogeneous isotropic turbulence

Forced isotropic turbulence evolves in the periodic box $\Omega _ { a } = \Omega _ { v } = [ 0 , 2 \pi ] ^ { 3 }$ , simulated by large-eddy simulation with the Smagorinsky model on $3 2 ^ { 3 }$ cells, with $\Delta t = 0 . 0 0 2$ , final time $T = 0 . 5$ , and Reynolds number $6 \times 1 0 ^ { 6 }$ . Every simulation starts from the divergence-free field

$$
\begin{array} { r l } & { u _ { x } ( x , 0 ) = \sin ( 2 x _ { 3 } ) + \cos ( 2 x _ { 2 } ) + \frac { 1 } { 2 } \big ( \sin ( 3 x _ { 3 } ) + \cos ( 3 x _ { 2 } ) \big ) + \frac { 1 } { 4 } \big ( \sin ( 5 x _ { 3 } ) + \cos ( 5 x _ { 2 } ) \big ) , } \\ & { u _ { y } ( x , 0 ) = \sin ( 2 x _ { 1 } ) + \cos ( 2 x _ { 3 } ) + \frac { 1 } { 2 } \big ( \sin ( 3 x _ { 1 } ) + \cos ( 3 x _ { 3 } ) \big ) + \frac { 1 } { 4 } \big ( \sin ( 5 x _ { 1 } ) + \cos ( 5 x _ { 3 } ) \big ) , } \end{array}
$$

![](images/4e180d843d632e26807737c28e613e6060a8d6b26b9ceced48df2306d8177f58.jpg)

![](images/4faa965967ee52af29fed65e90ae550722f40aefb098ea18066595520604f48b.jpg)  
Figure 16: Buoyancy-driven cavity. (A) Initial temperature. (B) Speed at $t = 5 ,$

![](images/e81e3bf3141db4de36c3e8db44560fdc30e139918c90132bf023f7d73254ef88.jpg)

![](images/69a22fe3e48dde96888753abbb35429ec52e42adb71cabb27ffa033ff26a2167.jpg)  
Figure 17: Merging vortices. (A) Initial vorticity. (B) Speed at $t = 0 . 4$

$$
\begin{array} { r } { u _ { z } ( x , 0 ) = \sin ( 2 x _ { 2 } ) + \cos ( 2 x _ { 1 } ) + \frac { 1 } { 2 } \big ( \sin ( 3 x _ { 2 } ) + \cos ( 3 x _ { 1 } ) \big ) + \frac { 1 } { 4 } \big ( \sin ( 5 x _ { 2 } ) + \cos ( 5 x _ { 1 } ) \big ) , } \end{array}\tag{87}
$$

and is driven by the steady divergence-free forcing

$$
\mathbf { f } ( x ) = { \left[ \begin{array} { l } { c _ { 1 } \left( \sin ( q _ { 1 } x _ { 3 } ) + \cos ( q _ { 2 } x _ { 2 } ) \right) } \\ { c _ { 2 } \left( \sin ( q _ { 1 } x _ { 1 } ) + \cos ( q _ { 2 } x _ { 3 } ) \right) } \\ { c _ { 3 } \left( \sin ( q _ { 1 } x _ { 2 } ) + \cos ( q _ { 2 } x _ { 1 } ) \right) } \end{array} \right] } ,\tag{88}
$$

![](images/cc9ade4b616ca8c614d5a4b4af180729c429359a8024011ae55e81186cf52c6c.jpg)

(B) Speed at � = 0.5  
![](images/a954042bab8a6ee16aa8fb8de75ea0aff86b9db6468f12c12852f8488983a075.jpg)  
Figure 18: Turbulent species transport. (A) Inlet velocity on the plane � = 0. (B) Speed at $t = 0 . 5$ on the whole domain.

whose amplitudes $c _ { 1 } , c _ { 2 } , c _ { 3 }$ are drawn from a normal distribution with standard deviation 0.15 truncated to [−0.3, 0.3], and whose distinct wavenumbers $q _ { 1 } , q _ { 2 }$ are drawn from {1, 2, 3}. We learn the map from the sampled forcing to the velocity at $t = 0 . 5$ on 7000 Fekete points of the grid.

## D.2. Manufactured tests of the output reconstruction

## D.2.1. Domains andfields

The convergence tests of Section 5.1 use the domains $\Omega _ { d } = \{ F _ { d } > 0 \}$ with

$$
\begin{array} { c } { F _ { 2 } ( x , y ) = 1 - x ^ { 2 } - 1 . 4 y ^ { 2 } - 0 . 1 5 x ^ { 4 } } \\ { - 0 . 1 0 x ^ { 2 } y ^ { 2 } - 0 . 2 0 y ^ { 4 } + 0 . 1 0 x ^ { 3 } , } \end{array}\tag{89}
$$

$$
\begin{array} { c } { F _ { 3 } ( x , y , z ) = 1 - x ^ { 2 } - 1 . 3 y ^ { 2 } - 1 . 7 z ^ { 2 } - 0 . 1 0 ( x ^ { 4 } + y ^ { 4 } + z ^ { 4 } ) } \\ { - 0 . 0 8 x ^ { 2 } y ^ { 2 } - 0 . 0 6 y ^ { 2 } z ^ { 2 } + 0 . 1 0 x y z . } \end{array}\tag{90}
$$

In 2D, the field is $v = ( \partial _ { \nu } \psi , - \partial _ { x } \psi )$ with

$$
\psi = F _ { 2 } ^ { 2 } \bigl [ \sin ( \pi x ) \sin ( 2 \pi y ) + 0 . 2 \cos ( 3 \pi x ) \sin ( \pi y ) \bigr ] ,\tag{91}
$$

and in 3D, $\boldsymbol { v } = \nabla \times \boldsymbol { A }$ with

$$
A = F _ { 3 } ^ { 2 } \big ( \sin ( \pi y ) \sin ( 2 \pi z ) , \sin ( \pi z ) \sin ( 2 \pi x ) , \sin ( \pi x ) \sin ( 2 \pi y ) \big ) ^ { \top } .\tag{92}
$$

Both fields are divergence free and vanish on the boundary.

The node generator of KernelPack-JAX [8] produced four node sets in each domain, at spacings 0.103, 0.0675, 0.0445, and 0.0295 in 2D (172 to 1907 nodes) and 0.22, 0.16, 0.118, and 0.088 in 3D (197 to 2570 nodes). The errors

![](images/db9521b3902697c845fe14a542ecfff47496934dcd919f31ebbc62dafab62625.jpg)  
Figure 19: Forced isotropic turbulence. (A) Forcing magnitude. (B) Speed at $t = 0 . 5 .$

use fixed sets of 4096 (2D) and 32768 (3D) uniformly random interior points. For degrees $\varpi = 1 , 3 , 5 ,$ , the divergencefree polynomial spaces have dimensions $n _ { P } = 5 ,$ , 14, 27 in 2D and 11, 50, 133 in 3D, and every stencil keeps the full space. At the finest level, the sixth-order errors are about $4 . 5 \times 1 0 ^ { - 6 }$ in 2D and $1 . 6 \times 1 0 ^ { - 3 }$ in 3D for both kernels.

## D.2.2. Divergence and periodicity

On the evaluation points where the output reconstruction is diferentiable (i.e., away from the centers of the cubic kernel), we measure the divergence relative to the size of the velocity gradient,

$$
D _ { \mathrm { r m s } } = \frac { \left( | Q _ { D } | ^ { - 1 } \sum _ { y \in Q _ { D } } | \nabla \cdot \boldsymbol { v } _ { h } ( y ) | ^ { 2 } \right) ^ { 1 / 2 } } { \operatorname* { m a x } \{ 1 , \left( | Q _ { D } | ^ { - 1 } \sum _ { y \in Q _ { D } } \| D \boldsymbol { v } _ { h } ( y ) \| _ { F } ^ { 2 } \right) ^ { 1 / 2 } \} } ,\tag{93}
$$

with the derivative of each local expansion taken on its own stencil. Its largest value over all smooth tests is $1 . 5 \times 1 0 ^ { - 1 3 }$ The periodic tests on $[ 0 , 2 \pi ) ^ { d }$ use the potentials

$$
\begin{array} { r l } & { \psi = \sin x \cos ( 2 y ) + 0 . 2 \cos ( 3 x ) \sin y , \qquad d = 2 , } \\ & { A = ( \sin y \sin ( 2 z ) , \sin z \sin ( 2 x ) , \sin x \sin ( 2 y ) ) ^ { \top } , \qquad d = 3 , } \end{array}\tag{94}
$$

(95)

on 657 (2D) and 1042 (3D) scattered nodes. We compare the reconstructed velocities and Jacobians at points that difer by integer multiples of the period, on the faces, edges, and corners of the box and at 128 interior points. The largest diferences are $1 . 4 \times 1 0 ^ { - 1 2 }$ for the velocities and $1 . 6 \times 1 0 ^ { - 1 2 }$ for the Jacobians.

## D.2.3. The turbulent ensemble

The statistical test of Section 5.1 uses the solenoidal Fourier ensemble

$$
v ( y ) = \sum _ { \mathbf { k } \in \mathcal { K } } \left[ \mathbf { c _ { k } } \cos ( \mathbf { k } \cdot y ) + \mathbf { s _ { k } } \sin ( \mathbf { k } \cdot y ) \right] , \qquad \mathbb { E } [ \mathbf { c _ { k } } \mathbf { c _ { k } ^ { \intercal } } ] = \mathbb { E } [ \mathbf { s _ { k } } \mathbf { s _ { k } ^ { \intercal } } ] = w _ { \mathbf { k } } \left( I - { \frac { \mathbf { k } \mathbf { k } ^ { \intercal } } { \| \mathbf { k } \| _ { 2 } ^ { 2 } } } \right) ,\tag{96}
$$

with independent Gaussian coeficients perpendicular to their wave vectors, 1514 modes up to wavenumber 1024, and weights $w _ { \mathbf { k } }$ that integrate a $\left\| \mathbf { k } \right\| _ { 2 } ^ { - 5 / \hat { 3 } }$ energy spectrum over radial shells, normalized so that $2 \sum _ { \bf k } w _ { \bf k } = 1$ . Its covariance is divergence free, and its structure function approximately follows the Kolmogorov law over the separations $0 . 0 3 \leq r \leq 0 . 8$ , which we fixed before any output reconstruction. At smaller separations, the finite bandwidth makes the increments smooth. We restrict the ensemble to $\Omega _ { 3 }$ and reconstruct it on six node sets with spacings 0.26, 0.19, 0.14, 0.101, 0.075, and 0.05 (128 to 13567 nodes). The structure functions (70) use 32 separations from $1 0 ^ { - 5 }$ to 0.8 in eight fixed directions and are computed exactly from the covariance, not from samples. For degree five, $E _ { S _ { 7 } } ( 0 . 8 )$ falls from 14% on the coarsest set to 4.6% on the finest, while the worst error over the whole range remains 64%.

For the comparison of local and global output reconstruction, we use the node set with 1770 nodes and one fixed five-scale Matérn kernel for both, with the same polynomials. The global system uses the covariance anchored at a point �, $K ( x , y ) - K ( x , a ) - K ( a , y ) + K ( a , a )$ , which is positive definite and gives the same interpolant, since the constant polynomials absorb the diference. For degree five, the errors $E _ { S _ { 2 } } ( 0 . 8 )$ are 9.9% (local) and 10.6% (global), the normalized mean square errors of the increments are 0.168 and 0.166, and the worst errors over the whole range are about 87% for both.

## D.3. Field figures

The field figures show the first simulation of each data set and are drawn by the MATLAB scripts in our repository. In two dimensions, the speed $\| \mathbf { u } \| _ { 2 }$ is interpolated linearly from the samples. Panels of the same quantity share a color scale, and the transverse direction of the backward-facing step is expanded.

## D.4. Direct and Lagrange evaluation

Table 6 evaluates seven unaugmented operator models for each data set and each $N \in \{ 5 0 0 , 1 0 0 0 , 5 0 0 0 , 1 0 0 0 0 \} \colon$ the fixed-shape kernel at the radii of the five prescribed densities of Section 3.6 (from 10% to 90%, with $\theta = 1 0 ^ { - 6 } )$ and at its validated radius and regularization, and the fitted-shape kernel at its validated parameters. We rebuild each model from its selected parameters and check that its Lagrange predictions at the nodes agree with those of Section 5.2 (largest relative diference $1 . 0 \times 1 0 ^ { - 1 0 } )$ . Each model is then evaluated through its coeficients and through its Lagrange weights, with the same normalization, factorization (cuDSS with nested dissection), support, and regularization, and with the same spatial basis. With the local output reconstruction, direct evaluation solves the kernel system for the coeficients of all stencils, in blocks of stencils, and Lagrange evaluation applies the weights to the same coeficients. The global output reconstruction is that of Appendix A.5, on the nodes. The test errors are measured at the nodes on the 100 test fields.

## Table 8

Test error at the nodes (mean relative velocity error over the 100 test inputs) of direct and Lagrange evaluation of the validated unaugmented fixed-shape model of the lid-driven cavity with global output reconstruction, of direct evaluation applied to the sampled velocities instead of the coeficients, and the Frobenius norm of the spatial coeficients of the training fields. Bold: smallest test error in each row.
<table><tr><td>N</td><td>Direct</td><td>Lagrange</td><td>Direct/Lagrange</td><td>Direct on samples</td><td> $\| B \| _ { F }$ </td></tr><tr><td>500</td><td> $1 . 5 4 \times 1 0 ^ { - 5 }$ </td><td> $\mathbf { 1 . 2 0 \times 1 0 ^ { - 5 } }$ </td><td>1.28</td><td> $\mathbf { 1 . 2 0 \times 1 0 ^ { - 5 } }$ </td><td> $1 . 5 8 \times 1 0 ^ { 9 }$ </td></tr><tr><td>1000</td><td> $8 . 3 9 \times 1 0 ^ { - 6 }$ </td><td> $4 . 2 5 \times 1 0 ^ { - 6 }$ </td><td>1.97</td><td> $\mathbf { 4 . 2 4 \times 1 0 ^ { - 6 } }$ </td><td> $2 . 2 4 \times 1 0 ^ { 9 }$ </td></tr><tr><td>5000</td><td> $3 . 1 9 \times 1 0 ^ { - 6 }$ </td><td> $9 . 5 4 \times 1 0 ^ { - 7 }$ </td><td>3.34</td><td> $\mathbf { 9 . 3 2 \times 1 0 ^ { - 7 } }$ </td><td> $5 . 0 0 \times 1 0 ^ { 9 }$ </td></tr><tr><td>10000</td><td> $3 . 4 2 \times 1 0 ^ { - 6 }$ </td><td> $6 . 6 4 \times 1 0 ^ { - 7 }$ </td><td>5.14</td><td> ${ \bf 6 . 4 2 \times 1 0 ^ { - 7 } }$ </td><td> $7 . 0 8 \times 1 0 ^ { 9 }$ </td></tr></table>

Table 8 follows the validated fixed-shape model of the lid-driven cavity with global output reconstruction as � grows. The Frobenius norm of the spatial coeficients of the training fields grows from $1 . 6 \times 1 0 ^ { 9 } \mathrm { t o } 7 . 1 \times 1 0 ^ { 9 }$ , which shows the cancellation that Theorem 3 warns about, and the ratio of the direct to the Lagrange error grows from 1.3 to 5.1. Direct evaluation applied to the sampled velocities instead of the coeficients is as accurate as Lagrange evaluation, so the gain arises in the output reconstruction, through the large coeficients.

## D.5. Software

Our implementation and the scripts of this paper are part of KernelPack-JAX (https: $/ / \mathrm { g i }$ thub.com/ VarShankar/kernelpack-jax). The operator models of Section 5.2 were fitted with scripts/run\_fastol\_ gpu\_campaign.py. The comparisons of Section 5.1.4, Section 5.3, and Appendices A.5 and D.4 rebuild these models with the released code, whose predictions agree with those of the original fits to $1 . 0 \times 1 0 ^ { - 1 0 }$ (Appendix D.4), and run on the same GPU with scripts/fastol\_gpu\_evaluation\_study.py. The directory FastOLPaper lists the scripts that produce every result and holds the scripts that build the tables and figures from them.

## D.6. All operator benchmarks

This section collects the results of all ten benchmarks for every training size. Figures 20 and 23 complete Figures 8 and 10, Figures 21 and 22 complete Figures 7 and 9, and Table 9 gives the rates at which the errors decrease with �. Tables 10 and 11 collect the errors, storage, and prediction times at $N = 1 0 0 0 0$ , and Tables 12– 21 list all 160 fitted models. Two entries show why we report storage separately from density. The fixed-shape and fitted-shape augmented species models have densities of 9.1% and 21.7% but 26 and 35 million factor entries, and the factor of the backward-facing step, at 33% density, has 38 million entries, three quarters of a dense one. All cost columns use the nested-dissection ordering of Section 3.5, measured by a replay of the selected models (scripts/fastol\_gpu\_cost\_replay.py). The test errors were computed with the approximate-minimum-degree ordering, and the predictions with the two orderings agree to $1 . 2 \times 1 0 ^ { - \hat { 1 0 } }$

## Table 9

Least-squares exponent � of the test error ∝ $N ^ { - \tau }$ over � ∈ {500, 1000, 5000, 10000}. Bold: fastest decay in each row.
<table><tr><td>Data set</td><td>Fixed</td><td>Fitted</td><td>Fixed + affine Fitted + affine</td></tr><tr><td>Forced isotropic turbulence</td><td>0.68</td><td>0.69</td><td>0.76</td></tr><tr><td>Turbulent species transport</td><td>0.85</td><td>1.09</td><td>0.66 1.03 1.26</td></tr><tr><td>Merging vortices</td><td>0.77</td><td>0.73</td><td>0.57 0.53</td></tr><tr><td>Lid-driven cavity</td><td>0.97</td><td>1.09</td><td>0.96 1.08</td></tr><tr><td>Cylinder flow (no shedding)</td><td>0.91</td><td>0.87</td><td>0.85 0.84</td></tr><tr><td>Cylinder flow (vortex shedding)</td><td>0.54</td><td>0.53</td><td>0.44 0.47</td></tr><tr><td>Backward-facing step</td><td>0.39</td><td>0.42</td><td>0.14 0.15</td></tr><tr><td>Buoyancy-driven cavity</td><td>0.82</td><td>0.91</td><td>0.92 0.96</td></tr><tr><td>Taylor-Green vortices (space)</td><td>-0.56</td><td>-1.06</td><td>2.81 3.24</td></tr><tr><td>Taylor-Green vortices (spacetime)</td><td>-0.58</td><td>-1.03</td><td>2.79 3.22</td></tr></table>

Table 10  
Test error (mean relative velocity error of the predictions for the 100 test inputs) at $N = 1 0 0 0 0 .$ . The two augmented models use the same afine columns. Each model is fitted and validated separately, and all four use the same local output reconstruction. Both Taylor–Green cases use the amplitude normalization (72). Bold: smallest test error in each row.
<table><tr><td>Data set</td><td>Fixed</td><td></td><td>Fitted Fixed + affine Fitted + affine</td><td></td></tr><tr><td>Forced isotropic turbulence</td><td> $2 . 8 0 \times 1 0 ^ { - 3 }$ </td><td> $1 . 6 2 \times 1 0 ^ { - 3 }$ </td><td> $9 . 7 7 \times 1 0 ^ { - 5 }$ </td><td> ${ \bf 7 . 1 4 \times 1 0 ^ { - 5 } }$ </td></tr><tr><td>Turbulent species transport</td><td> $4 . 3 4 \times 1 0 ^ { - 4 }$ </td><td> $6 . 8 6 \times 1 0 ^ { - 5 }$ </td><td> $1 . 8 9 \times 1 0 ^ { - 4 }$ </td><td> $\mathbf { 3 . 6 9 \times 1 0 ^ { - 5 } }$ </td></tr><tr><td>Merging vortices</td><td> $\mathbf { 3 . 0 2 \times 1 0 ^ { - 7 } }$ </td><td> $3 . 0 7 \times 1 0 ^ { - 7 }$ </td><td> $3 . 3 9 \times 1 0 ^ { - 7 }$ </td><td> $3 . 1 9 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Lid-driven cavity</td><td> $6 . 4 2 \times 1 0 ^ { - 7 }$ </td><td> $4 . 7 6 \times 1 0 ^ { - 7 }$ </td><td> $6 . 4 2 \times 1 0 ^ { - 7 }$ </td><td> $\mathbf { 4 . 7 2 \times 1 0 ^ { - 7 } }$ </td></tr><tr><td>Cylinder flow (no shedding)</td><td> $2 . 0 2 \times 1 0 ^ { - 7 }$ </td><td> $1 . 9 4 \times 1 0 ^ { - 7 }$ </td><td> $\mathbf { 1 . 7 0 \times 1 0 ^ { - 7 } }$ </td><td> $1 . 7 3 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Cylinder flow (vortex shedding)</td><td> $3 . 1 9 \times 1 0 ^ { - 7 }$ </td><td> $\mathbf { 2 . 9 5 \times 1 0 ^ { - 7 } }$ </td><td> $3 . 1 9 \times 1 0 ^ { - 7 }$ </td><td> $2 . 9 6 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Taylor-Green vortices (space)</td><td> $2 . 8 2 \times 1 0 ^ { - 6 }$ </td><td> $2 . 1 1 \times 1 0 ^ { - 6 }$ </td><td> $2 . 4 2 \times 1 0 ^ { - 1 0 }$ </td><td> ${ \bf 5 . 2 2 \times 1 0 ^ { - 1 1 } }$ </td></tr><tr><td>Taylor-Green vortices (spacetime)</td><td> $2 . 4 6 \times 1 0 ^ { - 6 }$ </td><td> $1 . 8 6 \times 1 0 ^ { - 6 }$ </td><td> $2 . 1 8 \times 1 0 ^ { - 1 0 }$ </td><td> $\mathbf { 4 . 7 0 \times 1 0 ^ { - 1 1 } }$ </td></tr><tr><td>Backward-facing step</td><td> $6 . 3 3 \times 1 0 ^ { - 8 }$ </td><td> $6 . 3 3 \times 1 0 ^ { - 8 }$ </td><td> $\mathbf { 5 . 8 6 \times 1 0 ^ { - 8 } }$ </td><td> $\mathbf { 5 . 8 6 \times 1 0 ^ { - 8 } }$ </td></tr><tr><td>Buoyancy-driven cavity</td><td> $5 . 3 8 \times 1 0 ^ { - 6 }$ </td><td> $\mathbf { 4 . 4 7 \times 1 0 ^ { - 6 } }$ </td><td> $5 . 3 8 \times 1 0 ^ { - 6 }$ </td><td> $4 . 6 2 \times 1 0 ^ { - 6 }$ </td></tr></table>

## D.6.1. Ofline cost

Table 22 gives the ofline costs at � = 10000. Fitting and validating a fitted-shape model takes 11–27 minutes, against 2–7 minutes for a fixed-shape model. Once the parameters are chosen, rebuilding the factorization takes about one to five seconds for most benchmarks and about 43 seconds for the dense forced-turbulence models. The setup of

![](images/941f616c9bb89e3ce4c2ea5f454bc6e0cfcef68d1150c99502c769e24a62e65f.jpg)  
Figure 20: Test error of the other six benchmarks. Dashed: with afine augmentation.

![](images/262471ffdb4bdfa9171c81340e9dad361e211673034e268d930cf6c480b13652.jpg)  
Figure 21: Selected density of the other six benchmarks. With augmentation, the Taylor–Green models select about 10% density at the two largest �, against 75% or 90% without it.

![](images/f03b6e926ce1a0a8d6410da1c109504482a99ffb35a600319b2210def7a9bfcc.jpg)

![](images/90f0effab6e832f166c0a0de0f885e14f67fef0354cb77899f28f8ab5938bf35.jpg)

![](images/2bb1d2cbf88b6d4621fd9735879230dac90335ef756e7764167d73d34fd28a7e.jpg)

![](images/ed622e2df274fe0270f3a639291b8f4a57d0b08e72d6679b16ece8b64a99e849.jpg)

![](images/0fb54eb53b4a200622b1094634630dcc0ded279c91ed5c2826b4c364884be8ef.jpg)

![](images/74c5b880432282d28555e0fd2a6e713248ade564e43c56f6cca459260f8f7664.jpg)  
Figure 22: Accounted storage of the other six benchmarks (Appendix B.3).

![](images/242e5d360bd0268dc70b564193a6a2f9dab1cb1888288da54001b6db31dac410.jpg)

![](images/9558a224904adf69770d2aa839f9b523fbe66a69fb7cb4dc1fc4f0c8af4c71d9.jpg)

![](images/61c160ef0479edcd6e6cda623c14c2859894f31174bd83a22448da48c35940e7.jpg)

![](images/00a08efb10c3f7af0e28cad56e81c72d397824c3e4af61619160de32f85838c2.jpg)

![](images/b696e30bd593db41ace96bca0fe5731f92fa45e535036eef5419965ed0a56b0a.jpg)

![](images/b541e486f154a1e964876e28ee215bdf75beec8dbb6d6a0eb214a80f5fc3aae3.jpg)  
Figure 23: Median warm time to predict all 100 test fields for the other six benchmarks.

Table 11  
Accounted storage in decimal GB / median warm prediction time in ms for 100 fields at � = 10000. Storage includes the stored model arrays, the solve bufers, and cuDSS’s estimate of the memory of its factors. It is not measured peak GPU memory. Times include operator Lagrange evaluation and local output reconstruction. Bold: smallest storage and smallest time in each row
<table><tr><td>Data set</td><td></td><td>Fixed</td><td></td><td>Fitted</td><td>Fixed + affine</td><td></td><td>Fitted + affine</td><td></td></tr><tr><td>Forced isotropic turbulence</td><td>10.680</td><td>235.2</td><td>10.680</td><td>234.0</td><td>10.685</td><td>244.1</td><td>10.682</td><td>233.4</td></tr><tr><td>Turbulent species transport</td><td>2.662 /</td><td>126.6</td><td>3.130</td><td>126.0</td><td>2.630</td><td>123.3</td><td>3.137 / 137.4</td><td></td></tr><tr><td>Merging vortices</td><td></td><td>1.397 / 79.6</td><td>1.029</td><td>/62.4</td><td>0.772</td><td>/37.7</td><td></td><td>0.767 / 39.0</td></tr><tr><td>Lid-driven cavity</td><td>1.135</td><td>/75.9</td><td>1.135</td><td>/71.5</td><td>1.136</td><td>/70.2</td><td></td><td>1.136 / 70.3</td></tr><tr><td>Cylinder flow (no shedding)</td><td>1.141</td><td>/73.8</td><td>1.141</td><td>/73.7</td><td></td><td>1.142 / 75.7</td><td>1.142</td><td>/74.0</td></tr><tr><td>Cylinder flow (vortex shedding)</td><td>1.140</td><td>/76.2</td><td>1.140</td><td>/75.3</td><td></td><td>1.141 / 76.2</td><td></td><td>1.141 / 74.2</td></tr><tr><td>Taylor-Green vortices (space)</td><td>3.606 /117.0</td><td></td><td></td><td>4.334 /119.0</td><td>0.899</td><td> / 52.3</td><td></td><td>0.899 / 54.6</td></tr><tr><td>Taylor-Green vortices (spacetime)</td><td>3.766</td><td>119.8</td><td>4.494</td><td>120.1</td><td>1.059</td><td>/ 59.0</td><td>1.059</td><td>/61.7</td></tr><tr><td>Backward-facing step</td><td>1.679 / 72.4</td><td></td><td>1.679</td><td>/68.8</td><td></td><td>1.657 / 66.3</td><td></td><td>1.657 / 65.3</td></tr><tr><td>Buoyancy-driven cavity</td><td>2.504 /189.4</td><td></td><td></td><td>2.504 / 184.9</td><td>2.505 / 177.0</td><td></td><td></td><td>2.505 /173.4</td></tr></table>

Table 12

Forced isotropic turbulence: all models. Test error is the mean relative velocity error of the predictions for the 100 test inputs. Density is $d _ { K } = \mathrm { n n z } ( K _ { U U } ) / N ^ { 2 }$ in percent. Factor is the number of entries of the cuDSS Cholesky factor in millions. GB is the accounted storage and ms the median warm time for all 100 fields. Bold: smallest test error at each �.
<table><tr><td>N</td><td>Operator model</td><td>Test error</td><td>Density (%)</td><td>Factor (10⁶)</td><td>GB</td><td>ms</td></tr><tr><td>500</td><td>Fixed shape</td><td>0.0214</td><td>90.00</td><td>0.125</td><td>0.465</td><td>60.9</td></tr><tr><td>500</td><td>Fitted shape</td><td>0.013</td><td>90.00</td><td>0.125</td><td>0.465</td><td>64.3</td></tr><tr><td>500</td><td>Fixed shape + affine</td><td> $9 . 3 7 \times 1 0 ^ { - 4 }$ </td><td>90.00</td><td>0.125</td><td>0.467</td><td>58.5</td></tr><tr><td>500</td><td>Fitted shape + affine</td><td> $\mathbf { 4 . 9 0 \times 1 0 ^ { - 4 } }$ </td><td>90.00</td><td>0.126</td><td>0.467</td><td>56.5</td></tr><tr><td>1000</td><td>Fixed shape</td><td>0.0157</td><td>90.00</td><td>0.990</td><td>0.750</td><td>64.8</td></tr><tr><td>1000</td><td>Fitted shape</td><td>0.0139</td><td>90.00</td><td>0.991</td><td>0.750</td><td>70.2</td></tr><tr><td>1000</td><td>Fixed shape + affine</td><td> $6 . 0 5 \times 1 0 ^ { - 4 }$ </td><td>90.00</td><td>0.990</td><td>0.752</td><td>65.2</td></tr><tr><td>1000</td><td>Fitted shape + affine</td><td> $\mathbf { 3 . 4 3 \times 1 0 ^ { - 4 } }$ </td><td>90.00</td><td>0.991</td><td>0.752</td><td>65.5</td></tr><tr><td>5000</td><td>Fixed shape</td><td> $5 . 1 6 \times 1 0 ^ { - 3 }$ </td><td>90.00</td><td>24.722</td><td>4.390</td><td>126.8</td></tr><tr><td>5000</td><td>Fitted shape</td><td> $4 . 6 9 \times 1 0 ^ { - 3 }$ </td><td>90.00</td><td>24.748</td><td>4.390</td><td>127.2</td></tr><tr><td>5000</td><td>Fixed shape + affine</td><td> $1 . 6 7 \times 1 0 ^ { - 4 }$ </td><td>90.00</td><td>24.722</td><td>4.394</td><td>126.9</td></tr><tr><td>5000</td><td>Fitted shape + affine</td><td> $\mathbf { 1 . 1 0 \times 1 0 ^ { - 4 } }$ </td><td>90.00</td><td>24.751</td><td>4.394</td><td>126.1</td></tr><tr><td>10000</td><td>Fixed shape</td><td> $2 . 8 0 \times 1 0 ^ { - 3 }$ </td><td>90.00</td><td>98.899</td><td>10.680</td><td>235.2</td></tr><tr><td>10000</td><td>Fitted shape</td><td> $1 . 6 2 \times 1 0 ^ { - 3 }$ </td><td>90.00</td><td>98.984</td><td>10.680</td><td>234.0</td></tr><tr><td>10000</td><td>Fixed shape + affine</td><td> $9 . 7 7 \times 1 0 ^ { - 5 }$ </td><td>90.00</td><td>98.899</td><td>10.685</td><td>244.1</td></tr><tr><td>10000</td><td>Fitted shape + affine</td><td> ${ \bf 7 . 1 4 \times 1 0 ^ { - 5 } }$ </td><td>90.00</td><td>98.651</td><td>10.682</td><td>233.4</td></tr></table>

the output reconstruction, which the four models share, takes 0.03–2.9 seconds after compilation, and a first setup, which also compiles, takes 1.6–24 seconds.

Turbulent species transport: all models. Test error is the mean relative velocity error of the predictions for the 100 test inputs. Density is $d _ { K } = \mathrm { n n z } ( K _ { U U } ) / N ^ { 2 }$ in percent. Factor is the number of entries of the cuDSS Cholesky factor in millions. GB is the accounted storage and ms the median warm time for all 100 fields. Bold: smallest test error at each �.
<table><tr><td>N</td><td>Operator model</td><td>Test error</td><td>Density (%)</td><td>Factor (10⁶)</td><td>GB</td><td>ms</td></tr><tr><td>500</td><td>Fixed shape</td><td> $4 . 3 9 \times 1 0 ^ { - 3 }$ </td><td>63.94</td><td>0.116</td><td>0.226</td><td>54.3</td></tr><tr><td>500</td><td>Fitted shape</td><td> $2 . 5 0 \times 1 0 ^ { - 3 }$ </td><td>90.00</td><td>0.129</td><td>0.228</td><td>52.8</td></tr><tr><td>500</td><td> $\mathsf { F i x e d \ s h a p e + a f f i n e }$ </td><td> $3 . 3 5 \times 1 0 ^ { - 3 }$ </td><td>40.98</td><td>0.097</td><td>0.224</td><td>53.1</td></tr><tr><td>500</td><td>Fitted shape + affine</td><td> $\mathbf { 2 . 4 8 \times 1 0 ^ { - 3 } }$ </td><td>44.60</td><td>0.096</td><td>0.224</td><td>54.0</td></tr><tr><td>1000</td><td>Fixed shape</td><td> $1 . 2 5 \times 1 0 ^ { - 3 }$ </td><td>89.71</td><td>1.000</td><td>0.278</td><td>57.1</td></tr><tr><td>1000</td><td>Fitted shape</td><td> $9 . 5 3 \times 1 0 ^ { - 4 }$ </td><td>85.62</td><td>0.962</td><td>0.276</td><td>57.8</td></tr><tr><td>1000</td><td>Fixed shape + affine</td><td> $1 . 4 9 \times 1 0 ^ { - 3 }$ </td><td>22.92</td><td>0.313</td><td>0.250</td><td>55.6</td></tr><tr><td>1000</td><td>Fitted shape + affine</td><td> $\mathbf { 8 . 7 5 \times 1 0 ^ { - 4 } }$ </td><td>75.00</td><td>0.988</td><td>0.273</td><td>59.1</td></tr><tr><td>5000</td><td>Fixed shape</td><td> $2 . 0 2 \times 1 0 ^ { - 4 }$ </td><td>69.06</td><td>24.743</td><td>1.858</td><td>98.2</td></tr><tr><td>5000</td><td>Fitted shape</td><td> $2 . 9 5 \times 1 0 ^ { - 4 }$ </td><td>90.00</td><td>24.744</td><td>2.036</td><td>101.7</td></tr><tr><td>5000</td><td>Fixed shape + affine</td><td> $\mathbf { 1 . 9 6 \times 1 0 ^ { - 4 } }$ </td><td>69.56</td><td>24.689</td><td>1.862</td><td>98.5</td></tr><tr><td>5000</td><td>Fitted shape + affine</td><td> $2 . 5 5 \times 1 0 ^ { - 4 }$ </td><td>66.71</td><td>24.746</td><td>1.838</td><td>98.3</td></tr><tr><td>10000</td><td>Fixed shape</td><td> $4 . 3 4 \times 1 0 ^ { - 4 }$ </td><td>10.40</td><td>24.431</td><td>2.662</td><td>126.6</td></tr><tr><td>10000</td><td>Fitted shape</td><td> $6 . 8 6 \times 1 0 ^ { - 5 }$ </td><td>21.95</td><td>33.613</td><td>3.130</td><td>126.0</td></tr><tr><td>10000</td><td>Fixed shape + affine</td><td> $1 . 8 9 \times 1 0 ^ { - 4 }$ </td><td>9.12</td><td>25.740</td><td>2.630</td><td>123.3</td></tr><tr><td>10000</td><td>Fitted  $\mathsf { s h a p e } + \mathsf { a f f i n e }$ </td><td> $\mathbf { 3 . 6 9 \times 1 0 ^ { - 5 } }$ </td><td>21.71</td><td>35.385</td><td>3.137</td><td>137.4</td></tr></table>

Table 14

Merging vortices: all models. Test error is the mean relative velocity error of the predictions for the 100 test inputs. Density is $d _ { K } = \mathrm { n n z } ( K _ { U U } ) / N ^ { 2 }$ in percent. Factor is the number of entries of the cuDSS Cholesky factor in millions. GB is the accounted storage and ms the median warm time for all 100 fields. Bold: smallest test error at each �.
<table><tr><td>N</td><td>Operator model</td><td>Test error</td><td>Density (%)</td><td>Factor (106)</td><td>GB</td><td>ms</td></tr><tr><td>500</td><td>Fixed shape</td><td> $2 . 8 5 \times 1 0 ^ { - 6 }$ </td><td>75.06</td><td>0.104</td><td>0.025</td><td>6.0</td></tr><tr><td>500</td><td>Fitted shape</td><td> $2 . 5 1 \times 1 0 ^ { - 6 }$ </td><td>75.00</td><td>0.106</td><td>0.025</td><td>6.8</td></tr><tr><td>500</td><td>Fixed shape + affine</td><td> $1 . 9 4 \times 1 0 ^ { - 6 }$ </td><td>37.36</td><td>0.065</td><td>0.021</td><td>7.9</td></tr><tr><td>500</td><td>Fitted shape + affine</td><td> $\mathbf { 1 . 6 6 \times 1 0 ^ { - 6 } }$ </td><td>33.64</td><td>0.060</td><td>0.021</td><td>6.9</td></tr><tr><td>1000</td><td>Fixed shape</td><td> $1 . 4 1 \times 1 0 ^ { - 6 }$ </td><td>63.81</td><td>0.386</td><td>0.060</td><td>7.3</td></tr><tr><td>1000</td><td>Fitted shape</td><td> $1 . 4 5 \times 1 0 ^ { - 6 }$ </td><td>60.31</td><td>0.387</td><td>0.058</td><td>8.6</td></tr><tr><td>1000</td><td>Fixed shape + affine</td><td> $8 . 6 0 \times 1 0 ^ { - 7 }$ </td><td>29.73</td><td>0.225</td><td>0.047</td><td>9.1</td></tr><tr><td>1000</td><td>Fitted shape + affine</td><td> $\mathbf { 8 . 4 6 \times 1 0 ^ { - 7 } }$ </td><td>29.04</td><td>0.217</td><td>0.047</td><td>7.4</td></tr><tr><td>5000</td><td>Fixed shape</td><td> $\mathbf { 3 . 5 4 \times 1 0 ^ { - 7 } }$ </td><td>24.58</td><td>6.404</td><td>0.433</td><td>24.9</td></tr><tr><td>5000</td><td>Fitted shape</td><td> $3 . 9 1 \times 1 0 ^ { - 7 }$ </td><td>25.00</td><td>6.356</td><td>0.436</td><td>25.2</td></tr><tr><td>5000</td><td>Fixed shape + affine</td><td> $3 . 6 6 \times 1 0 ^ { - 7 }$ </td><td>24.58</td><td>6.404</td><td>0.434</td><td>24.9</td></tr><tr><td>5000</td><td>Fitted shape + affine</td><td> $4 . 0 8 \times 1 0 ^ { - 7 }$ </td><td>18.33</td><td>4.892</td><td>0.369</td><td>21.7</td></tr><tr><td>10000</td><td>Fixed shape</td><td> $\mathbf { 3 . 0 2 \times 1 0 ^ { - 7 } }$ </td><td>24.67</td><td>26.353</td><td>1.397</td><td>79.6</td></tr><tr><td>10000</td><td>Fitted shape</td><td> $3 . 0 7 \times 1 0 ^ { - 7 }$ </td><td>15.85</td><td>18.009</td><td>1.029</td><td>62.4</td></tr><tr><td>10000</td><td>Fixed shape + affine</td><td> $3 . 3 9 \times 1 0 ^ { - 7 }$ </td><td>10.00</td><td>10.703</td><td>0.772</td><td>37.7</td></tr><tr><td>10000</td><td>Fitted shape + affine</td><td> $3 . 1 9 \times 1 0 ^ { - 7 }$ </td><td>10.00</td><td>10.047</td><td>0.767</td><td>39.0</td></tr></table>

Table 15  
Lid-driven cavity: all models. Test error is the mean relative velocity error of the predictions for the 100 test inputs. Density is $d _ { K } = \mathrm { n n z } ( K _ { U U } ) / N ^ { 2 }$ in percent. Factor is the number of entries of the cuDSS Cholesky factor in millions. GB is the accounted storage and ms the median warm time for all 100 fields. Bold: smallest test error at each �.
<table><tr><td>N</td><td>Operator model</td><td>Test error</td><td>Density (%)</td><td>Factor (10⁶)</td><td>GB</td><td>ms</td></tr><tr><td>500</td><td>Fixed shape</td><td> $1 . 2 0 \times 1 0 ^ { - 5 }$ </td><td>22.85</td><td>0.053</td><td>0.046</td><td>9.2</td></tr><tr><td>500</td><td>Fitted shape</td><td> $1 . 2 0 \times 1 0 ^ { - 5 }$ </td><td>22.85</td><td>0.053</td><td>0.046</td><td>10.8</td></tr><tr><td>500</td><td>Fixed shape + affine</td><td> $\mathbf { 1 . 1 5 \times 1 0 ^ { - 5 } }$ </td><td>21.49</td><td>0.048</td><td>0.046</td><td>10.4</td></tr><tr><td>500</td><td>Fitted shape + affine</td><td> $\mathbf { 1 . 1 5 \times 1 0 ^ { - 5 } }$ </td><td>25.00</td><td>0.055</td><td>0.047</td><td>9.2</td></tr><tr><td>1000</td><td>Fixed shape</td><td> $4 . 2 4 \times 1 0 ^ { - 6 }$ </td><td>19.99</td><td>0.190</td><td>0.096</td><td>10.6</td></tr><tr><td>1000</td><td>Fitted shape</td><td> $4 . 1 1 \times 1 0 ^ { - 6 }$ </td><td>25.00</td><td>0.223</td><td>0.098</td><td>11.4</td></tr><tr><td>1000</td><td>Fixed shape + affine</td><td> $4 . 1 3 \times 1 0 ^ { - 6 }$ </td><td>20.00</td><td>0.189</td><td>0.096</td><td>10.8</td></tr><tr><td>1000</td><td>Fitted shape + affine</td><td> $\mathbf { 4 . 1 0 \times 1 0 ^ { - 6 } }$ </td><td>25.00</td><td>0.223</td><td>0.098</td><td>11.1</td></tr><tr><td>5000</td><td>Fixed shape</td><td> $9 . 3 2 \times 1 0 ^ { - 7 }$ </td><td>10.00</td><td>2.600</td><td>0.538</td><td>34.4</td></tr><tr><td>5000</td><td>Fitted shape</td><td> $\mathbf { 6 . 7 3 \times 1 0 ^ { - 7 } }$ </td><td>10.00</td><td>2.600</td><td>0.538</td><td>34.6</td></tr><tr><td>5000</td><td>Fixed shape + affine</td><td> $9 . 3 2 \times 1 0 ^ { - 7 }$ </td><td>10.00</td><td>2.600</td><td>0.539</td><td>34.2</td></tr><tr><td>5000</td><td>Fitted shape + affine</td><td> $6 . 8 0 \times 1 0 ^ { - 7 }$ </td><td>10.00</td><td>2.600</td><td>0.539</td><td>34.7</td></tr><tr><td>10000</td><td>Fixed shape</td><td> $6 . 4 2 \times 1 0 ^ { - 7 }$ </td><td>10.00</td><td>11.198</td><td>1.135</td><td>75.9</td></tr><tr><td>10000</td><td>Fitted shape</td><td> $4 . 7 6 \times 1 0 ^ { - 7 }$ </td><td>10.00</td><td>11.198</td><td>1.135</td><td>71.5</td></tr><tr><td>10000</td><td>Fixed shape + affine</td><td> $6 . 4 2 \times 1 0 ^ { - 7 }$ </td><td>10.00</td><td>11.198</td><td>1.136</td><td>70.2</td></tr><tr><td>10000</td><td>Fitted shape + affine</td><td> $\mathbf { 4 . 7 2 \times 1 0 ^ { - 7 } }$ </td><td>10.00</td><td>11.198</td><td>1.136</td><td>70.3</td></tr></table>

Table 16

Cylinder flow (no shedding): all models. Test error is the mean relative velocity error of the predictions for the 100 test inputs. Density is $d _ { K } = \mathrm { n n z } ( K _ { U U } ) / N ^ { 2 }$ in percent. Factor is the number of entries of the cuDSS Cholesky factor in millions. GB is the accounted storage and ms the median warm time for all 100 fields. Bold: smallest test error at each �.
<table><tr><td>N</td><td>Operator model</td><td>Test error</td><td>Density (%)</td><td>Factor (106)</td><td>GB</td><td>ms</td></tr><tr><td>500</td><td>Fixed shape</td><td> $2 . 9 1 \times 1 0 ^ { - 6 }$ </td><td>89.11</td><td>0.129</td><td>0.053</td><td>9.8</td></tr><tr><td>500</td><td>Fitted shape</td><td> $2 . 4 2 \times 1 0 ^ { - 6 }$ </td><td>90.00</td><td>0.128</td><td>0.053</td><td>9.6</td></tr><tr><td>500</td><td>Fixed shape + affine</td><td> $2 . 2 7 \times 1 0 ^ { - 6 }$ </td><td>85.52</td><td>0.128</td><td>0.052</td><td>10.6</td></tr><tr><td>500</td><td>Fitted shape + affine</td><td> $\mathbf { 2 . 2 0 \times 1 0 ^ { - 6 } }$ </td><td>90.00</td><td>0.128</td><td>0.053</td><td>10.9</td></tr><tr><td>1000</td><td>Fixed shape</td><td> $9 . 2 1 \times 1 0 ^ { - 7 }$ </td><td>54.74</td><td>0.674</td><td>0.111</td><td>13.5</td></tr><tr><td>1000</td><td>Fitted shape</td><td> $9 . 2 1 \times 1 0 ^ { - 7 }$ </td><td>54.74</td><td>0.674</td><td>0.111</td><td>13.3</td></tr><tr><td>1000</td><td>Fixed shape + affine</td><td> $6 . 8 0 \times 1 0 ^ { - 7 }$ </td><td>45.10</td><td>0.581</td><td>0.107</td><td>13.4</td></tr><tr><td>1000</td><td>Fitted shape + affine</td><td> $\mathbf { 6 . 4 5 \times 1 0 ^ { - 7 } }$ </td><td>50.00</td><td>0.394</td><td>0.107</td><td>12.6</td></tr><tr><td>5000</td><td>Fixed shape</td><td> $1 . 9 7 \times 1 0 ^ { - 7 }$ </td><td>16.22</td><td>4.555</td><td>0.607</td><td>26.0</td></tr><tr><td>5000</td><td>Fitted shape</td><td> $1 . 9 7 \times 1 0 ^ { - 7 }$ </td><td>16.22</td><td>4.555</td><td>0.607</td><td>27.2</td></tr><tr><td>5000</td><td>Fixed shape + affine</td><td> $1 . 9 3 \times 1 0 ^ { - 7 }$ </td><td>10.00</td><td>2.726</td><td>0.540</td><td>20.5</td></tr><tr><td>5000</td><td>Fitted shape + affine</td><td> $\mathbf { 1 . 8 1 \times 1 0 ^ { - 7 } }$ </td><td>16.22</td><td>4.555</td><td>0.608</td><td>26.3</td></tr><tr><td>10000</td><td>Fixed shape</td><td> $2 . 0 2 \times 1 0 ^ { - 7 }$ </td><td>10.00</td><td>11.981</td><td>1.141</td><td>73.8</td></tr><tr><td>10000</td><td>Fitted shape</td><td> $1 . 9 4 \times 1 0 ^ { - 7 }$ </td><td>10.00</td><td>11.981</td><td>1.141</td><td>73.7</td></tr><tr><td>10000</td><td>Fixed shape + affine</td><td> $\mathbf { 1 . 7 0 \times 1 0 ^ { - 7 } }$ </td><td>10.00</td><td>11.981</td><td>1.142</td><td>75.7</td></tr><tr><td>10000</td><td>Fitted shape + affine</td><td> $1 . 7 3 \times 1 0 ^ { - 7 }$ </td><td>10.00</td><td>11.981</td><td>1.142</td><td>74.0</td></tr></table>

Table 17  
Cylinder flow (vortex shedding): all models. Test error is the mean relative velocity error of the predictions for the 100 test inputs. Density is $d _ { K } = \mathrm { n n z } ( K _ { U U } ) / N ^ { 2 }$ in percent. Factor is the number of entries of the cuDSS Cholesky factor in millions. GB is the accounted storage and ms the median warm time for all 100 fields. Bold: smallest test error at each �.
<table><tr><td>N</td><td>Operator model</td><td>Test error</td><td>Density (%)</td><td>Factor (10⁶)</td><td>GB</td><td>ms</td></tr><tr><td>500</td><td>Fixed shape</td><td> $1 . 3 5 \times 1 0 ^ { - 6 }$ </td><td>70.21</td><td>0.122</td><td>0.051</td><td>12.2</td></tr><tr><td>500</td><td>Fitted shape</td><td> $1 . 3 5 \times 1 0 ^ { - 6 }$ </td><td>70.21</td><td>0.122</td><td>0.051</td><td>9.7</td></tr><tr><td>500</td><td>Fixed shape + affine</td><td> $1 . 1 1 \times 1 0 ^ { - 6 }$ </td><td>68.61</td><td>0.122</td><td>0.051</td><td>12.6</td></tr><tr><td>500</td><td>Fitted shape + affine</td><td> $\mathbf { 1 . 0 9 \times 1 0 ^ { - 6 } }$ </td><td>75.00</td><td>0.125</td><td>0.051</td><td>12.0</td></tr><tr><td>1000</td><td>Fixed shape</td><td> $1 . 1 9 \times 1 0 ^ { - 6 }$ </td><td>27.92</td><td>0.262</td><td>0.099</td><td>11.6</td></tr><tr><td>1000</td><td>Fitted shape</td><td> $8 . 5 4 \times 1 0 ^ { - 7 }$ </td><td>25.00</td><td>0.231</td><td>0.098</td><td>11.9</td></tr><tr><td>1000</td><td>Fixed shape + affine</td><td> $\mathbf { 8 . 3 1 \times 1 0 ^ { - 7 } }$ </td><td>16.61</td><td>0.164</td><td>0.094</td><td>12.6</td></tr><tr><td>1000</td><td>Fitted shape + affine</td><td> $\mathbf { 8 . 3 1 \times 1 0 ^ { - 7 } }$ </td><td>16.61</td><td>0.164</td><td>0.094</td><td>10.1</td></tr><tr><td>5000</td><td>Fixed shape</td><td> $3 . 5 3 \times 1 0 ^ { - 7 }$ </td><td>10.00</td><td>2.710</td><td>0.539</td><td>23.8</td></tr><tr><td>5000</td><td>Fitted shape</td><td> $\mathbf { 3 . 3 0 \times 1 0 ^ { - 7 } }$ </td><td>10.00</td><td>2.710</td><td>0.539</td><td>25.6</td></tr><tr><td>5000</td><td>Fixed shape + affine</td><td> $3 . 5 3 \times 1 0 ^ { - 7 }$ </td><td>10.00</td><td>2.710</td><td>0.540</td><td>24.3</td></tr><tr><td>5000</td><td>Fitted shape + affine</td><td> $3 . 3 1 \times 1 0 ^ { - 7 }$ </td><td>10.00</td><td>2.710</td><td>0.540</td><td>24.4</td></tr><tr><td>10000</td><td>Fixed shape</td><td> $3 . 1 9 \times 1 0 ^ { - 7 }$ </td><td>10.00</td><td>11.865</td><td>1.140</td><td>76.2</td></tr><tr><td>10000</td><td>Fitted shape</td><td> $\mathbf { 2 . 9 5 \times 1 0 ^ { - 7 } }$ </td><td>10.00</td><td>11.865</td><td>1.140</td><td>75.3</td></tr><tr><td>10000</td><td>Fixed shape + affine</td><td> $3 . 1 9 \times 1 0 ^ { - 7 }$ </td><td>10.00</td><td>11.865</td><td>1.141</td><td>76.2</td></tr><tr><td>10000</td><td>Fitted shape + affine</td><td> $2 . 9 6 \times 1 0 ^ { - 7 }$ </td><td>10.00</td><td>11.865</td><td>1.141</td><td>74.2</td></tr></table>

Table 18

Taylor–Green vortices (space): all models. Test error is the mean relative velocity error of the predictions for the 100 test inputs. Density is $d _ { K } = \mathrm { n n z } ( K _ { U U } ) / N ^ { 2 }$ in percent. Factor is the number of entries of the cuDSS Cholesky factor in millions. GB is the accounted storage and ms the median warm time for all 100 fields. Bold: smallest test error at each �.
<table><tr><td>N</td><td>Operator model</td><td>Test error</td><td>Density (%)</td><td>Factor (106)</td><td>GB</td><td>ms</td></tr><tr><td>500</td><td>Fixed shape</td><td> $9 . 4 1 \times 1 0 ^ { - 7 }$ </td><td>46.17</td><td>0.094</td><td>0.028</td><td>6.5</td></tr><tr><td>500</td><td>Fitted shape</td><td> ${ \bf 7 . 7 2 \times 1 0 ^ { - 7 } }$ </td><td>50.00</td><td>0.097</td><td>0.029</td><td>7.7</td></tr><tr><td>500</td><td>Fixed shape + affine</td><td> $8 . 7 6 \times 1 0 ^ { - 7 }$ </td><td>40.07</td><td>0.083</td><td>0.028</td><td>7.7</td></tr><tr><td>500</td><td>Fitted shape + affine</td><td> $8 . 8 3 \times 1 0 ^ { - 7 }$ </td><td>50.00</td><td>0.097</td><td>0.029</td><td>6.7</td></tr><tr><td>1000</td><td>Fixed shape</td><td> $3 . 3 1 \times 1 0 ^ { - 7 }$ </td><td>51.76</td><td>0.422</td><td>0.068</td><td>7.6</td></tr><tr><td>1000</td><td>Fitted shape</td><td> ${ \bf 7 . 8 6 \times 1 0 ^ { - 9 } }$ </td><td>50.00</td><td>0.414</td><td>0.067</td><td>8.6</td></tr><tr><td>1000</td><td>Fixed shape + affine</td><td> $3 . 6 3 \times 1 0 ^ { - 8 }$ </td><td>32.16</td><td>0.288</td><td>0.060</td><td>8.0</td></tr><tr><td>1000</td><td>Fitted shape + affine</td><td> $1 . 0 3 \times 1 0 ^ { - 8 }$ </td><td>50.00</td><td>0.414</td><td>0.067</td><td>8.2</td></tr><tr><td>5000</td><td>Fixed shape</td><td> $2 . 4 5 \times 1 0 ^ { - 6 }$ </td><td>75.00</td><td>22.923</td><td>1.056</td><td>38.9</td></tr><tr><td>5000</td><td>Fitted shape</td><td> $2 . 5 1 \times 1 0 ^ { - 6 }$ </td><td>90.00</td><td>24.889</td><td>1.199</td><td>41.5</td></tr><tr><td>5000</td><td>Fixed shape + affine</td><td> $2 . 6 7 \times 1 0 ^ { - 1 0 }$ </td><td>10.00</td><td>2.644</td><td>0.339</td><td>14.2</td></tr><tr><td>5000</td><td>Fitted shape + affine</td><td> $\mathbf { 6 . 0 0 \times 1 0 ^ { - 1 1 } }$ </td><td>10.00</td><td>2.644</td><td>0.339</td><td>14.7</td></tr><tr><td>10000</td><td>Fixed shape</td><td> $2 . 8 2 \times 1 0 ^ { - 6 }$ </td><td>75.00</td><td>72.677</td><td>3.606</td><td>117.0</td></tr><tr><td>10000</td><td>Fitted shape</td><td> $2 . 1 1 \times 1 0 ^ { - 6 }$ </td><td>90.00</td><td>99.800</td><td>4.334</td><td>119.0</td></tr><tr><td>10000</td><td>Fixed shape + affine</td><td> $2 . 4 2 \times 1 0 ^ { - 1 0 }$ </td><td>10.00</td><td>11.728</td><td>0.899</td><td>52.3</td></tr><tr><td>10000</td><td>Fitted shape + affine</td><td> ${ \bf 5 . 2 2 \times 1 0 ^ { - 1 1 } }$ </td><td>10.00</td><td>11.728</td><td>0.899</td><td>54.6</td></tr></table>

Table 19  
Taylor–Green vortices (spacetime): all models. Test error is the mean relative velocity error of the predictions for the 100 test inputs. Density is $\dot { d } _ { K } = \mathrm { n n z } ( \dot { K } _ { U U } ) / N ^ { 2 }$ in percent. Factor is the number of entries of the cuDSS Cholesky factor in millions. GB is the accounted storage and ms the median warm time for all 100 fields. Bold: smallest test error at each �.
<table><tr><td>N</td><td>Operator model</td><td>Test error</td><td>Density (%)</td><td>Factor (106)</td><td>GB</td><td>ms</td></tr><tr><td>500</td><td>Fixed shape</td><td> $8 . 0 1 \times 1 0 ^ { - 7 }$ </td><td>46.23</td><td>0.093</td><td>0.052</td><td>6.4</td></tr><tr><td>500</td><td>Fitted shape</td><td> $\mathbf { 6 . 5 8 \times 1 0 ^ { - 7 } }$ </td><td>50.00</td><td>0.097</td><td>0.053</td><td>6.3</td></tr><tr><td>500</td><td>Fixed shape + affine</td><td> $7 . 3 8 \times 1 0 ^ { - 7 }$ </td><td>39.51</td><td>0.082</td><td>0.052</td><td>8.1</td></tr><tr><td>500</td><td>Fitted shape + affine</td><td> $7 . 6 5 \times 1 0 ^ { - 7 }$ </td><td>50.00</td><td>0.097</td><td>0.053</td><td>6.8</td></tr><tr><td>1000</td><td>Fixed shape</td><td> $2 . 7 1 \times 1 0 ^ { - 7 }$ </td><td>51.58</td><td>0.419</td><td>0.116</td><td>7.9</td></tr><tr><td>1000</td><td>Fitted shape</td><td> $9 . 3 0 \times 1 0 ^ { - 9 }$ </td><td>50.00</td><td>0.414</td><td>0.115</td><td>9.8</td></tr><tr><td>1000</td><td>Fixed shape + affine</td><td> $3 . 3 4 \times 1 0 ^ { - 8 }$ </td><td>32.27</td><td>0.284</td><td>0.108</td><td>11.0</td></tr><tr><td>1000</td><td>Fitted shape + affine</td><td> $\mathbf { 8 . 6 1 \times 1 0 ^ { - 9 } }$ </td><td>50.00</td><td>0.414</td><td>0.115</td><td>7.7</td></tr><tr><td>5000</td><td>Fixed shape</td><td> $2 . 1 4 \times 1 0 ^ { - 6 }$ </td><td>75.00</td><td>22.923</td><td>1.136</td><td>41.9</td></tr><tr><td>5000</td><td>Fitted shape</td><td> $2 . 1 9 \times 1 0 ^ { - 6 }$ </td><td>90.00</td><td>24.889</td><td>1.279</td><td>40.2</td></tr><tr><td>5000</td><td>Fixed shape + affine</td><td> $2 . 4 1 \times 1 0 ^ { - 1 0 }$ </td><td>10.00</td><td>2.644</td><td>0.419</td><td>16.8</td></tr><tr><td>5000</td><td>Fitted shape + affine</td><td> $\mathbf { 5 . 4 0 \times 1 0 ^ { - 1 1 } }$ </td><td>10.00</td><td>2.644</td><td>0.419</td><td>16.6</td></tr><tr><td>10000</td><td>Fixed shape</td><td> $2 . 4 6 \times 1 0 ^ { - 6 }$ </td><td>75.00</td><td>72.677</td><td>3.766</td><td>119.8</td></tr><tr><td>10000</td><td>Fitted shape</td><td> $1 . 8 6 \times 1 0 ^ { - 6 }$ </td><td>90.00</td><td>99.800</td><td>4.494</td><td>120.1</td></tr><tr><td>10000</td><td>Fixed shape + affine</td><td> $2 . 1 8 \times 1 0 ^ { - 1 0 }$ </td><td>10.00</td><td>11.728</td><td>1.059</td><td>59.0</td></tr><tr><td>10000</td><td>Fitted  $\mathsf { s h a p e } + \mathsf { a f f i n e }$ </td><td> $\mathbf { 4 . 7 0 \times 1 0 ^ { - 1 1 } }$ </td><td>10.00</td><td>11.728</td><td>1.059</td><td>61.7</td></tr></table>

Table 20

Backward-facing step: all models. Test error is the mean relative velocity error of the predictions for the 100 test inputs. Density is $d _ { K } = \mathrm { n n z } ( K _ { U U } ) / N ^ { 2 }$ in percent. Factor is the number of entries of the cuDSS Cholesky factor in millions. GB is the accounted storage and ms the median warm time for all 100 fields. Bold: smallest test error at each �.
<table><tr><td>N</td><td>Operator model</td><td>Test error</td><td>Density (%)</td><td>Factor (106)</td><td>GB</td><td>ms</td></tr><tr><td>500</td><td>Fixed shape</td><td> $2 . 4 3 \times 1 0 ^ { - 7 }$ </td><td>36.41</td><td>0.073</td><td>0.024</td><td>8.8</td></tr><tr><td>500</td><td>Fitted shape</td><td> $2 . 4 3 \times 1 0 ^ { - 7 }$ </td><td>36.41</td><td>0.073</td><td>0.024</td><td>8.6</td></tr><tr><td>500</td><td>Fixed shape + affine</td><td> $\mathbf { 9 . 4 6 \times 1 0 ^ { - 8 } }$ </td><td>34.94</td><td>0.068</td><td>0.024</td><td>9.6</td></tr><tr><td>500</td><td>Fitted shape + affine</td><td> $\mathbf { 9 . 4 6 \times 1 0 ^ { - 8 } }$ </td><td>34.94</td><td>0.068</td><td>0.024</td><td>8.6</td></tr><tr><td>1000</td><td>Fixed shape</td><td> $9 . 3 1 \times 1 0 ^ { - 8 }$ </td><td>49.59</td><td>0.350</td><td>0.059</td><td>11.6</td></tr><tr><td>1000</td><td>Fitted shape</td><td> $1 . 1 4 \times 1 0 ^ { - 7 }$ </td><td>25.00</td><td>0.236</td><td>0.050</td><td>8.4</td></tr><tr><td>1000</td><td>Fixed shape + affine</td><td> $\mathbf { 6 . 6 8 \times 1 0 ^ { - 8 } }$ </td><td>47.15</td><td>0.342</td><td>0.058</td><td>9.9</td></tr><tr><td>1000</td><td>Fitted shape + affine</td><td> $7 . 4 4 \times 1 0 ^ { - 8 }$ </td><td>25.00</td><td>0.236</td><td>0.050</td><td>8.4</td></tr><tr><td>5000</td><td>Fixed shape</td><td> $6 . 7 7 \times 1 0 ^ { - 8 }$ </td><td>29.08</td><td>7.711</td><td>0.504</td><td>25.8</td></tr><tr><td>5000</td><td>Fitted shape</td><td> $6 . 7 7 \times 1 0 ^ { - 8 }$ </td><td>29.08</td><td>7.711</td><td>0.504</td><td>26.1</td></tr><tr><td>5000</td><td>Fixed shape + affine</td><td> $\mathbf { 5 . 9 8 \times 1 0 ^ { - 8 } }$ </td><td>28.90</td><td>7.785</td><td>0.503</td><td>26.5</td></tr><tr><td>5000</td><td>Fitted shape + affine</td><td> $\mathbf { 5 . 9 8 \times 1 0 ^ { - 8 } }$ </td><td>28.90</td><td>7.785</td><td>0.503</td><td>26.2</td></tr><tr><td>10000</td><td>Fixed shape</td><td> $6 . 3 3 \times 1 0 ^ { - 8 }$ </td><td>32.98</td><td>40.506</td><td>1.679</td><td>72.4</td></tr><tr><td>10000</td><td>Fitted shape</td><td> $6 . 3 3 \times 1 0 ^ { - 8 }$ </td><td>32.98</td><td>40.506</td><td>1.679</td><td>68.8</td></tr><tr><td>10000</td><td>Fixed shape + affine</td><td> $\mathbf { 5 . 8 6 \times 1 0 ^ { - 8 } }$ </td><td>32.88</td><td>38.228</td><td>1.657</td><td>66.3</td></tr><tr><td>10000</td><td>Fitted shape + affine</td><td> $\mathbf { 5 . 8 6 \times 1 0 ^ { - 8 } }$ </td><td>32.88</td><td>38.228</td><td>1.657</td><td>65.3</td></tr></table>

Buoyancy-driven cavity: all models. Test error is the mean relative velocity error of the predictions for the 100 test inputs. Density is $d _ { K } = \mathrm { n n z } ( K _ { U U } ) / N ^ { 2 }$ in percent. Factor is the number of entries of the cuDSS Cholesky factor in millions. GB is the accounted storage and ms the median warm time for all 100 fields. Bold: smallest test error at each �.
<table><tr><td>N</td><td>Operator model</td><td>Test error</td><td>Density (%) Factor (10⁶)</td><td>GB</td><td>ms</td></tr><tr><td>500</td><td>Fixed shape</td><td> $\mathbf { 5 . 6 3 \times 1 0 ^ { - 5 } }$ </td><td>64.68</td><td>0.118 0.154</td><td>30.7</td></tr><tr><td>500</td><td>Fitted shape</td><td> $\mathbf { 5 . 6 3 \times 1 0 ^ { - 5 } }$ </td><td>64.68</td><td>0.118 0.154</td><td>29.6</td></tr><tr><td>500</td><td>Fixed shape + affine</td><td> $8 . 1 8 \times 1 0 ^ { - 5 }$ </td><td>10.00</td><td>0.024 0.148</td><td>29.1</td></tr><tr><td>500</td><td>Fitted shape + affine</td><td> $7 . 3 9 \times 1 0 ^ { - 5 }$ </td><td>10.00</td><td>0.024 0.148</td><td>28.6</td></tr><tr><td>1000</td><td>Fixed shape</td><td> $\mathbf { 3 . 0 3 \times 1 0 ^ { - 5 } }$ </td><td>17.33</td><td>0.170 0.298</td><td>32.5</td></tr><tr><td>1000</td><td>Fitted shape</td><td> $\mathbf { 3 . 0 3 \times 1 0 ^ { - 5 } }$ </td><td>17.33</td><td>0.170 0.298</td><td>32.2</td></tr><tr><td>1000</td><td>Fixed shape + affine</td><td> $3 . 0 4 \times 1 0 ^ { - 5 }$ </td><td>17.27</td><td>0.166 0.298</td><td>36.4</td></tr><tr><td>1000</td><td>Fitted shape + affine</td><td> $3 . 0 4 \times 1 0 ^ { - 5 }$ </td><td>17.27</td><td>0.166 0.298</td><td>32.6</td></tr><tr><td>5000</td><td>Fixed shape</td><td> $6 . 6 5 \times 1 0 ^ { - 6 }$ </td><td>10.00</td><td>2.838 1.143</td><td>111.5</td></tr><tr><td>5000</td><td>Fitted shape</td><td> $\mathbf { 5 . 0 2 \times 1 0 ^ { - 6 } }$ </td><td>10.00</td><td>2.838 1.143</td><td>104.1</td></tr><tr><td>5000</td><td>Fixed shape + affine</td><td> $6 . 6 5 \times 1 0 ^ { - 6 }$ </td><td>10.00</td><td>2.838 1.144</td><td>107.7</td></tr><tr><td>5000</td><td>Fitted shape + affine</td><td> $5 . 3 5 \times 1 0 ^ { - 6 }$ </td><td>10.00</td><td>2.838 1.144</td><td>109.9</td></tr><tr><td>10000</td><td>Fixed shape</td><td> $5 . 3 8 \times 1 0 ^ { - 6 }$ </td><td>10.00</td><td>11.947 2.504</td><td>189.4</td></tr><tr><td>10000</td><td>Fitted shape</td><td> $\mathbf { 4 . 4 7 \times 1 0 ^ { - 6 } }$ </td><td>10.00</td><td>11.947 2.504</td><td>184.9</td></tr><tr><td>10000</td><td>Fixed shape + affine</td><td> $5 . 3 8 \times 1 0 ^ { - 6 }$ </td><td>10.00</td><td>11.947 2.505</td><td>177.0</td></tr><tr><td>10000</td><td>Fitted shape + affine</td><td> $4 . 6 2 \times 1 0 ^ { - 6 }$ </td><td>10.00</td><td>11.947 2.505</td><td>173.4</td></tr></table>

Table 22  
Ofline and initialization costs at � = 10000. Selection includes kernel fitting, where applicable, and validation. Rebuild assembles and factors the selected operator. Initialization includes two synchronized 100-field prediction passes, compilation, and CUDA graph preparation. The four models share the setup of the output reconstruction and may reuse compiled code. The dash marks the selection time of the fixed-shape backward-facing-step model, which contains an external pause. Its raw time is in the released data. Bold: smallest time in each column for each data set.
<table><tr><td>Data set</td><td>Operator model</td><td>Selection (min)</td><td>Rebuild (s)</td><td>Initialization (s)</td></tr><tr><td rowspan="4">Forced isotropic turbulence</td><td>Fixed shape</td><td>3.83</td><td>43.27</td><td>3.71</td></tr><tr><td>Fitted shape</td><td>26.32</td><td>41.63</td><td>3.28</td></tr><tr><td>Fixed shape + affine</td><td>6.45</td><td>43.70</td><td>4.15</td></tr><tr><td>Fitted shape + affine</td><td>27.01</td><td>42.77</td><td>3.05</td></tr><tr><td rowspan="4">Turbulent species transport</td><td>Fixed shape</td><td>3.24</td><td>1.48</td><td>4.17</td></tr><tr><td>Fitted shape</td><td>25.89</td><td>4.64</td><td>3.70</td></tr><tr><td>Fixed shape + affine</td><td>5.79</td><td>1.76</td><td>6.05</td></tr><tr><td>Fitted shape + affine</td><td>24.01</td><td>4.93</td><td>3.79</td></tr><tr><td rowspan="4">Merging vortices</td><td>Fixed shape</td><td>4.81</td><td>0.91</td><td>3.35</td></tr><tr><td>Fitted shape</td><td>15.26</td><td>2.92</td><td>3.20</td></tr><tr><td>Fixed shape + affine</td><td>4.94</td><td>1.12</td><td>4.94</td></tr><tr><td>Fitted shape + affine</td><td>11.70</td><td>0.81</td><td>3.26</td></tr><tr><td rowspan="4">Lid-driven cavity</td><td>Fixed shape</td><td>2.12</td><td>0.86</td><td>2.37</td></tr><tr><td>Fitted shape</td><td>12.97</td><td>0.87</td><td>2.20</td></tr><tr><td>Fixed shape + affine</td><td>3.06</td><td>0.91</td><td>3.54</td></tr><tr><td>Fitted shape + affine</td><td>10.90</td><td>0.85</td><td>2.26</td></tr><tr><td rowspan="4">Cylinder flow (no shedding)</td><td>Fixed shape</td><td>2.32</td><td>0.87</td><td>2.58</td></tr><tr><td>Fitted shape</td><td>12.77</td><td>0.86</td><td>2.54</td></tr><tr><td>Fixed shape + affine</td><td>4.28</td><td>0.92</td><td>3.77</td></tr><tr><td>Fitted shape + affine</td><td>14.53</td><td>0.90</td><td>2.50</td></tr><tr><td rowspan="4">Cylinder flow (vortex shedding)</td><td>Fixed shape</td><td>2.12</td><td>0.85</td><td>2.47</td></tr><tr><td>Fitted shape</td><td>13.58</td><td>0.85</td><td>2.61</td></tr><tr><td>Fixed shape + affine</td><td>3.07</td><td>0.91</td><td>3.43</td></tr><tr><td>Fitted shape + affine</td><td>12.86</td><td>0.88</td><td>2.37</td></tr><tr><td rowspan="4">Taylor-Green vortices (space)</td><td>Fixed shape</td><td>3.41</td><td>3.18</td><td>3.61</td></tr><tr><td>Fitted shape</td><td>12.80</td><td>5.54</td><td>3.43</td></tr><tr><td>Fixed shape + affine</td><td>5.32</td><td>1.25</td><td>5.47</td></tr><tr><td>Fitted shape + affine</td><td>15.30</td><td>0.68</td><td>3.54</td></tr><tr><td rowspan="4">Taylor-Green vortices (spacetime)</td><td>Fixed shape</td><td>3.49</td><td>3.30</td><td>3.87</td></tr><tr><td>Fitted shape</td><td>16.10</td><td>3.60</td><td>3.60</td></tr><tr><td>Fixed shape + affine</td><td>4.82</td><td>0.88</td><td>5.22</td></tr><tr><td>Fitted shape + affine</td><td>15.47</td><td>0.69</td><td>3.64</td></tr><tr><td rowspan="4">Backward-facing step</td><td>Fixed shape</td><td></td><td>0.95</td><td>1.48</td></tr><tr><td>Fitted shape</td><td>14.23</td><td>0.99</td><td>1.31</td></tr><tr><td>Fixed shape + affine</td><td>3.25</td><td>3.27</td><td>2.53</td></tr><tr><td>Fitted shape + affine</td><td>13.42</td><td>0.94</td><td>1.36</td></tr><tr><td rowspan="4">Buoyancy-driven cavity</td><td>Fixed shape</td><td>2.11</td><td>1.65</td><td>2.72</td></tr><tr><td>Fitted shape</td><td>17.67</td><td>1.48</td><td>2.53</td></tr><tr><td>Fixed shape + affine</td><td>3.16</td><td>1.64</td><td>4.06</td></tr><tr><td>Fitted shape + affine</td><td>17.00</td><td>1.51</td><td>2.63</td></tr></table>

## References

[1] Bach, F.R., Jordan, M.I., 2005. Predictive low-rank decomposition for kernel methods, in: Proceedings of the 22nd International Conference on Machine Learning, pp. 33–40. doi:10.1145/1102351.1102356.

[2] Batlle, P., Darcy, M., Hosseini, B., Owhadi, H., 2024. Kernel methods are competitive for operator learning. Journal of Computational Physics 496, 112549.

[3] Bayona, V., Flyer, N., Fornberg, B., Barnett, G.A., 2017. On the role of polynomials in RBF-FD approximations: II. Numerical solution of elliptic PDEs. Journal of Computational Physics 332, 257–273. doi:10.1016/j.jcp.2016.12.008.

[4] Belianovich, M., Fasshauer, G.E., Narayan, A., Shankar, V., 2026. A unified framework for eficient kernel and polynomial interpolation. Jour nal of Approximation Software 3. URL: https://ojs.unito.it/index.php/JAS/article/view/12288, doi:10.13135/3103-1935 12288.

[5] Benner, P., Gugercin, S., Willcox, K., 2015. A survey of projection-based model reduction methods for parametric dynamical systems. SIAM Review 57, 483–531. doi:10.1137/130932715.

[6] Bollig, E.F., Flyer, N., Erlebacher, G., 2012. Solution to PDEs using radial basis function finite-diferences (RBF-FD) on multiple GPUs. Journal of Computational Physics 231, 7133–7151. doi:10.1016/j.jcp.2012.06.030.

[7] Boullé, N., Townsend, A., 2024. A mathematical guide to operator learning, in: Handbook of Numerical Analysis. Elsevier. volume 25, pp. 83–125.

[8] Cattaneo, A., Ballard, M.K., Kirby, R.M., Wright, G.B., Shankar, V., 2026. KernelPack-JAX: a meshless diferentiable GPU framework for node generation and forward and inverse PDE problems. Submitted to Journal of Computational Physics .

[9] Davis, T.A., 2006. Direct Methods for Sparse Linear Systems. volume 2 of Fundamentals of Algorithms. SIAM, Philadelphia, PA. doi:10.1137/1.9780898718881.

[10] Davydov, O., Schaback, R., 2016. Error bounds for kernel-based numerical diferentiation. Numerische Mathematik 132, 243–269. doi:10.1007/s00211-015-0722-9.

[11] Dodu, F., Rabut, C., 2004. Irrotational or divergence-free interpolation. Numerische Mathematik 98, 477–498. doi:10.1007/ s00211-004-0541-x.

[12] Drake, K.P., Fuselier, E.J., Wright, G.B., 2021. A partition of unity method for divergence-free or curl-free radial basis function approximation. SIAM Journal on Scientific Computing 43, A1950–A1974. doi:10.1137/20M1373505.

[13] Drineas, P., Mahoney, M.W., 2005. Approximating a Gram matrix for improved kernel-based learning, in: International Conference on Computational Learning Theory, pp. 323–337.

[14] Duchon, J., 1977. Splines minimizing rotation-invariant semi-norms in Sobolev spaces, in: Constructive Theory of Functions of Several Variables. Springer, Berlin, Heidelberg. volume 571 of Lecture Notes in Mathematics, pp. 85–100. doi:10.1007/BFb0086566.

[15] Fasshauer, G.E., 2007. Meshfree Approximation Methods with MATLAB. World Scientific.

[16] Fasshauer, G.E., Ye, Q., 2011. Reproducing kernels of generalized Sobolev spaces via a Green function approach with distributional operators. Numerische Mathematik 119, 585–611. doi:10.1007/s00211-011-0391-2.

[17] Fine, S., Scheinberg, K., 2001. Eficient SVM training using low-rank kernel representations. Journal of Machine Learning Research 2, 243–264.

[18] Flyer, N., Barnett, G.A., Wicker, L.J., 2016a. Enhancing finite diferences with radial basis functions: Experiments on the Navier–Stokes equations. Journal of Computational Physics 316, 39–62. doi:10.1016/j.jcp.2016.02.078.

[19] Flyer, N., Fornberg, B., Bayona, V., Barnett, G.A., 2016b. On the role of polynomials in rbf-fd approximations: I. interpolation and accuracy. Journal of Computational Physics 321, 21–38.

[20] Fornberg, B., Flyer, N., 2015a. A Primer on Radial Basis Functions with Applications to the Geosciences. SIAM, Philadelphia, PA. doi:10.1137/1.9781611974041.

[21] Fornberg, B., Flyer, N., 2015b. Solving PDEs with radial basis functions. Acta Numerica 24, 215–258. doi:10.1017/S0962492914000130.

[22] Fornberg, B., Lehto, E., 2011. Stabilization of RBF-generated finite diference methods for convective PDEs. Journal of Computationa Physics 230, 2270–2285. doi:10.1016/j.jcp.2010.12.014.

[23] Furrer, R., Genton, M.G., Nychka, D., 2006. Covariance tapering for interpolation of large spatial datasets. Journal of Computational and Graphical Statistics 15, 502–523. doi:10.1198/106186006X132178.

[24] Fuselier, E., 2008. Sobolev-type approximation rates for divergence-free and curl-free RBF interpolants. Mathematics of Computation 77, 1407–1423.

[25] Fuselier, E.J., Shankar, V., Wright, G.B., 2016. A high-order radial basis function (RBF) leray projection method for the solution of the incompressible unsteady Stokes equations. Computers & Fluids 128, 41–52.

[26] Gittens, A., Mahoney, M.W., 2013. Revisiting the Nyström method for improved large-scale machine learning, in: International Conference on Machine Learning, pp. 567–575.

[27] Gneiting, T., 2002. Compactly supported correlation functions. Journal of Multivariate Analysis 83, 493–508. doi:10.1006/jmva.2001. 2056.

[28] Higham, N.J., 2002. Accuracy and Stability of Numerical Algorithms. 2 ed., Society for Industrial and Applied Mathematics. doi:10.1137/ 1.9780898718027.

[29] Kadri, H., Duflos, E., Preux, P., Canu, S., Rakotomamonjy, A., Audifren, J., 2016. Operator-valued kernels for learning from functional response data. Journal of Machine Learning Research 17, 1–54.

[30] Khorrami, M.S., Goyal, P., Mianroodi, J.R., Svendsen, B., Benner, P., Raabe, D., 2024. A physics-encoded fourier neural operator approach for surrogate modeling of divergence-free stress fields in solids. arXiv preprint arXiv:2408.15408 .

[31] Kingma, D.P., Ba, J., 2015. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 .

[32] Li, Z., Huang, D.Z., Liu, B., Anandkumar, A., 2023. Fourier neural operator with learned deformations for PDEs on general geometries. Journal of Machine Learning Research 24, 1–26.

[33] Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A., 2020a. Fourier neural operator for parametric partial diferential equations. arXiv preprint arXiv:2010.08895 .

[34] Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A., 2020b. Neural operator: Graph kernel network for partial diferential equations. arXiv preprint arXiv:2003.03485 .

[35] Lindgren, F., Rue, H., Lindström, J., 2011. An explicit link between Gaussian fields and Gaussian Markov random fields: The stochastic partial diferential equation approach. Journal of the Royal Statistical Society Series B: Statistical Methodology 73, 423–498. doi:10.1111/ j.1467-9868.2011.00777.x.

[36] Lowery, M., Turnage, J., Morrow, Z., Jakeman, J.D., Narayan, A., Zhe, S., Shankar, V., 2024. Kernel neural operators (KNOs) for scalable, memory-eficient, geometrically-flexible operator learning. arXiv preprint arXiv:2407.00809 .

[37] Lu, L., Jin, P., Pang, G., Zhang, Z., Karniadakis, G.E., 2021. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence 3, 218–229.

[38] Mora, C., Yousefpour, A., Hosseinmardi, S., Owhadi, H., Bostanabad, R., 2025. Operator learning with gaussian processes. Computer Methods in Applied Mechanics and Engineering 434, 117581.

[39] Narcowich, F.J., Ward, J.D., 1994. Generalized Hermite interpolation via matrix-valued conditionally positive definite functions. Mathematics of Computation 63, 661–687.

[40] Narcowich, F.J., Ward, J.D., Wright, G.B., 2007. Divergence-free RBFs on surfaces. Journal of Fourier Analysis and Applications 13, 643–663.

[41] Owhadi, H., 2023. Gaussian process hydrodynamics. Applied Mathematics and Mechanics 44, 1175–1198. doi:10.1007/ s10483-023-2990-9.

[42] Quarteroni, A., Manzoni, A., Negri, F., 2016. Reduced Basis Methods for Partial Diferential Equations: An Introduction. Springer, Cham, Switzerland. doi:10.1007/978-3-319-15431-2.

[43] Queipo, N.V., Haftka, R.T., Shyy, W., Goel, T., Vaidyanathan, R., Tucker, P.K., 2005. Surrogate-based analysis and optimization. Progress in Aerospace Sciences 41, 1–28. doi:10.1016/j.paerosci.2005.02.001.

[44] Rahimi, A., Recht, B., 2007. Random features for large-scale kernel machines, in: Advances in Neural Information Processing Systems, pp. 1177–1184.

[45] Richter-Powell, J., Lipman, Y., Chen, R.T., 2022. Neural conservation laws: A divergence-free perspective. Advances in Neural Information Processing Systems 35, 38075–38088.

[46] Santin, G., Wittwar, D., Haasdonk, B., 2018. Greedy regularized kernel interpolation. arXiv preprint arXiv:1807.09575. URL: https: //arxiv.org/abs/1807.09575, doi:10.48550/arXiv.1807.09575.

[47] Schäfer, F., Katzfuss, M., Owhadi, H., 2021. Sparse cholesky factorization by kullback–leibler minimization. SIAM Journal on Scientific Computing 43, A2019–A2046. doi:10.1137/20M1336254.

[48] Shankar, V., 2017. The overlapped radial basis function-finite diference (RBF-FD) method: A generalization of RBF-FD. Journal of Computational Physics 342, 211–228. doi:10.1016/j.jcp.2017.04.037.

[49] Shankar, V., Fogelson, A.L., 2018. Hyperviscosity-based stabilization for radial basis function-finite diference (RBF-FD) discretizations of advection–difusion equations. Journal of Computational Physics 372, 616–639. doi:10.1016/j.jcp.2018.06.036.

[50] Shankar, V., Kirby, R.M., Fogelson, A.L., 2018a. Robust node generation for mesh-free discretizations on irregular domains and surfaces. SIAM Journal on Scientific Computing 40, A2584–A2608. doi:10.1137/17M114090X.

[51] Shankar, V., Narayan, A., Kirby, R.M., 2018b. RBF-LOI: Augmenting radial basis functions (RBFs) with least orthogonal interpolation (LOI) for solving PDEs on surfaces. Journal of Computational Physics 373, 722–735. doi:10.1016/j.jcp.2018.07.015.

[52] Shankar, V., Wright, G.B., Fogelson, A.L., 2021. An eficient high-order meshless method for advection-difusion equations on time-varying irregular domains. Journal of Computational Physics 445, 110633. doi:10.1016/j.jcp.2021.110633.

[53] Sharma, R., Lowery, M., Owhadi, H., Shankar, V., 2026. Fluids you can trust: Property-preserving operator learning for incompressible flows. arXiv preprint arXiv:2602.15472 .

[54] Sharma, R., Shankar, V., 2026. Operator-learning datasets for incompressible fluid flow. URL: https://doi.org/10.5281/zenodo. 22907213, doi:10 5281/ d 22907213.

[55] Shu, C., Ding, H., Yeo, K.S., 2003. Local radial basis function-based diferential quadrature method and its application to solve twodimensional incompressible Navier–Stokes equations. Computer Methods in Applied Mechanics and Engineering 192, 941–954. doi:10. / 7 ( ) 7.

[56] Si, S., Hsieh, C.J., Dhillon, I.S., 2014. Memory eficient kernel approximation, in: Proceedings of the 31st International Conference on Machine Learning, pp. 2507–2515.

[57] Snelson, E., Ghahramani, Z., 2006. Sparse gaussian processes using pseudo-inputs, in: Advances in Neural Information Processing Systems, pp. 1257–1264.

[58] Stein, M.L., 1999. Interpolation of Spatial Data: Some Theory for Kriging. Springer Series in Statistics, Springer, New York. doi:10.1007/ 978-1-4612-1494-6.

[59] Stuart, A.M., 2010. Inverse problems: A Bayesian perspective. Acta Numerica 19, 451–559. doi:10.1017/S0962492910000061.

[60] Sudret, B., Marelli, S., Wiart, J., 2017. Surrogate models for uncertainty quantification: An overview, in: 2017 11th European Conference on Antennas and Propagation (EUCAP), IEEE. pp. 793–797. doi:10.23919/EuCAP.2017.7928679.

[61] Tezzele, M., Demo, N., Stabile, G., Mola, A., Rozza, G., 2020. Enhancing CFD predictions in shape design problems by model and parameter space reduction. Advanced Modeling and Simulation in Engineering Sciences 7, 40. doi:10.1186/s40323-020-00177-0.

[62] Tolstykh, A.I., Shirobokov, D.A., 2003. On using radial basis functions in a “finite diference mode” with applications to elasticity problems. Computational Mechanics 33, 68–79. doi:10.1007/s00466-003-0501-9.

[63] Tominec, I., Larsson, E., Heryudono, A., 2021. A least squares radial basis function finite diference method with improved stability properties. SIAM Journal on Scientific Computing 43, A1441–A1471. doi:10.1137/20M1320079.

[64] Wendland, H., 1995. Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree. Advances in Computational Mathematics 4, 389–396.

[65] Wendland, H., 2005. Scattered Data Approximation. Cambridge University Press, Cambridge, UK.

[66] Wendland, H., 2009. Divergence-free kernel methods for approximating the Stokes problem. SIAM Journal on Numerical Analysis 47, 3158–3179.

[67] Whittle, P., 1954. On stationary processes in the plane. Biometrika 41, 434–449. doi:10.1093/biomet/41.3-4.434.

[68] Wilson, A.G., Nickisch, H., 2015. Kernel interpolation for scalable structured Gaussian processes, in: Proceedings of the 32nd Internationa Conference on Machine Learning, pp. 1775–1784.

[69] Wright, G.B., Fornberg, B., 2006. Scattered node compact finite diference-type formulas generated from radial basis functions. Journal of Computational Physics 212, 99–123. doi:10.1016/j.jcp.2005.05.030.

[70] Wu, H., Luo, H., Wang, H., Wang, J., Long, M., 2024. Transolver: A fast transformer solver for PDEs on general geometries. arXiv preprint arXiv:2402.02366 .

[71] Yelnyk, V., 2023. RBF method for solving Navier–Stokes equations. Degree project in computational science. KTH Royal Institute of Technology. URL: https://kth.diva-portal.org/smash/get/diva2:1849751/FULLTEXT01.pdf.

[72] Yu, F.X., Suresh, A.T., Choromanski, K.M., Holtmann-Rice, D., Kumar, S., 2016. Orthogonal random features, in: Advances in Neura Information Processing Systems, pp. 1975–1983.