# Geometry-Dependent Approximation and Stability Properties of Shallow ReLU<sup>k</sup> Networks on Bounded Euclidean Domains

Yuqing Liu<sup>1</sup> Tong Mao<sup>2</sup> Jinchao Xu<sup>3</sup>

## Abstract

On bounded Euclidean domains, we study how hidden-parameter geometry afects shallow ReLU<sup>k</sup> networks through results on approximation rates, saturation-type lower bounds, Gram and stifness matrix spectra, Bernstein and Marcinkiewicz–Zygmund inequalities, discrete leastsquares recovery, and inverse approximation.

Our analysis uses smooth localization and dyadic Fourier estimates. We quantify parameter geometry by the fill distance h and antipodal separation distance h. For the prescribed network space $L _ { n } ^ { \dot { k } } .$ representative approximation estimates take the form

$$
\operatorname* { i n f } _ { g _ { n } \in L _ { n } ^ { k } } \| f - g _ { n } \| _ { \mathcal L ^ { 2 } ( \Omega ^ { \prime } ) } \lesssim h ^ { r } \| f \| _ { \mathcal H ^ { r } ( \Omega ) } ,
$$

$$
\operatorname* { i n f } _ { g _ { n } \in L _ { n } ^ { k } } \| f - g _ { n } \| _ { \mathcal L ^ { 2 } ( \Omega ) } \gtrsim \underline { { h } } ^ { \frac { d + 2 k + 1 } { 2 } } \| f \| _ { \mathcal L ^ { 2 } ( \Omega ^ { \prime } ) } .
$$

The upper bound exploits target regularity up to the critical order $\textstyle { \frac { d + 2 k + 1 } { 2 } }$ , while the lower bound applies asymptotically to fixed targets smoother than this order. The upper estimate holds on an interior subdomain $\Omega ^ { \prime } { } _ { i }$ , whereas the lower estimate concerns the larger domain Ω.

We also obtain geometry-dependent spectral estimates that quantify conditioning and coeficient stability. Further results establish norm discretization and regularized recovery from function values, and infer target regularity from approximation rates under coeficient or separation conditions. Altogether, these results describe how parameter geometry governs approximation accuracy, the stability of coeficient representations, discrete recovery, and the regularity information contained in convergence rates.

## 1 Introduction

From a mathematical perspective, central questions in machine learning concern how target regularity and structure, network size, and available samples jointly determine achievable accuracy [23, 25, 65]. Addressing these questions requires quantitative approximation estimates and an analysis of stable recovery from finitely many function values.

Shallow neural networks provide a starting point for these questions and a reference for studying deeper architectures [91, 6, 25]. For an activation function $\sigma ,$ consider

$$
\Sigma _ { n } ^ { \sigma } : = \left\{ x \mapsto \sum _ { j = 1 } ^ { n } a _ { j } \sigma ( w _ { j } \cdot x + b _ { j } ) : w _ { j } \in \mathbb { R } ^ { d } , \quad a _ { j } , b _ { j } \in \mathbb { R } \right\} .\tag{1.1}
$$

To study how network size afects achievable accuracy, approximation theory provides density results under suitable activation assumptions [24, 42, 58] and quantitative width-dependent error bounds in Barron, variation, spectral Barron, and Sobolev spaces [6, 80, 90, 25, 71].

To study how available samples afect achievable accuracy, statistical learning theory combines approximation and estimation error analyses through empirical risk minimization, regularization, and capacity bounds [1, 9, 23]. For neural networks, this interplay yields risk bounds and nearminimax estimation guarantees [7, 88, 71, 69]; related analyses of random-design regression and random-feature learning quantify the roles of sample size, model complexity, and regularization [43, 97].

Alongside network size and available samples, the distribution of hidden parameters influences both approximation accuracy and recovery from discrete data. We study these relationships for shallow networks with the ReLU activation and its integral-order variants, defined by

$$
\sigma _ { k } ( t ) = \left\{ \begin{array} { l l } { t ^ { k } , } & { t > 0 , } \\ { 0 , } & { t \leq 0 , } \end{array} \right. \quad k = 0 , 1 , 2 , \ldots { } .
$$

Approximation rates for shallow ReL $\boldsymbol { { \mathrm { I U } } ^ { k } }$ networks have been studied in [104, 77, 117, 75]. In particular, [75] establishes the $\mathcal { O } ( n ^ { - \frac { r } { d } } )$ rate in the $\mathcal { L } ^ { 2 }$ norm for targets in $\mathcal { H } ^ { r }$ with $\textstyle 0 < r \leq { \frac { d + 2 k + 1 } { 2 } }$ showing that these networks can exploit target regularity well beyond the degree of individual neurons.

To examine this dependence, we normalize each nonzero hidden-parameter vector by positive homogeneity as $\theta = \binom { w } { b } \in \mathbb { S } ^ { d }$ . For a configuration $\Theta _ { n } = \{ \theta _ { j } ^ { * } \} _ { j = 1 } ^ { n }$ , with $\theta _ { j } ^ { * } = \binom { w _ { j } ^ { * } } { b _ { j } ^ { * } }$ , the associated networks form the space

$$
L _ { n } ^ { k } = L _ { n } ^ { k } ( \Theta _ { n } ) : = \operatorname { s p a n } \left\{ \sigma _ { k } ( w _ { j } ^ { * } \cdot x + b _ { j } ^ { * } ) : j = 1 , \dots , n \right\} = \operatorname { s p a n } \left\{ \sigma _ { k } ( \theta _ { j } ^ { * } \cdot \tilde { x } ) : j = 1 , \dots , n \right\} .\tag{1.2}
$$

Here we write

$$
\widetilde { x } : = { \binom { x } { 1 } } , \qquad x \in \mathbb { R } ^ { d } .
$$

Prescribed parameter distributions yield geometry-dependent approximation upper bounds [90, 67]. We ask how this geometry also afects coeficient stability, recovery from discrete data, and the limits of approximation.

For prescribed hidden parameters, computing the output coeficients by least squares reduces to solving a linear system whose coeficient matrix is the associated Gram matrix. Its spectrum therefore connects parameter geometry to the conditioning of this system and the stability of the computed coeficients [66, 19].

For prescribed parameter configurations, these issues lead to the following questions:

1. How does the distribution of hidden parameters afect the achievable approximation accuracy?

2. How does the parameter geometry determine the eigenvalue spectrum and condition number of the associated Gram matrices, and hence the stability of the coeficient representation?

3. How many function values are needed to recover the continuous approximation rate through discrete least squares?

4. What regularity of the target function can be inferred from a given approximation rate?

5. Does additional target smoothness always improve the approximation rate?

Classical polynomial, spline, and kernel approximation connects direct estimates with norm control, sampling stability, inverse regularity, and saturation [27, 99, 112]. This framework motivates a corresponding analysis of shallow networks.

For shallow neural networks, several parts of this framework have recently been established on the sphere: geometry-dependent approximation upper bounds [67], Sobolev Gram matrix spectra and condition numbers [66], and approximation lower bounds and saturation [76]. Bernstein inequalities, sampling estimates, discrete least-squares recovery, and inverse results further relate parameter geometry to discrete data and target regularity [68]. The approximation and spectral bounds are sharp under antipodal quasi-uniformity. Section 2 gives further details and comparisons.

Extending these results to bounded Euclidean domains raises additional dificulties. On the full sphere, harmonic expansions provide a common framework for approximation, spectral estimates, and sampling. A bounded Euclidean domain lifts to only part of $\mathbb { S } ^ { d }$ , and restriction may disrupt the spectral structure and norm comparisons; see [66, Section $7 ]$ . Spherical condition-number bounds, Bernstein inequalities, approximation lower bounds, and least-squares guarantees may therefore fail on general domains. Corresponding results require formulations that account for both parameter geometry and domain geometry.

In this paper, we study how hidden-parameter distributions afect the approximation properties of shallow $\operatorname { R e L U } ^ { k }$ networks on bounded Euclidean domains. We use smooth localization to work in a whole-space Fourier framework. Let $\Omega \subset \mathbb { R } ^ { d }$ be a bounded connected domain and $\Omega ^ { \prime } \Subset \Omega$ an interior subdomain, and choose a cutof satisfying

$$
\chi \in \mathcal { C } _ { c } ^ { \infty } ( \Omega ) , \quad 0 \leq \chi \leq 1 , \quad \chi = 1 \mathrm { ~ o n ~ } \Omega ^ { \prime } .\tag{1.3}
$$

The corresponding localized neurons are

$$
\varphi _ { \theta } ( x ) : = \chi ( x ) \sigma _ { k } ( w \cdot x + b ) , \quad \theta = { \binom { w } { b } } \in \mathbb { S } ^ { d } , \quad x \in \mathbb { R } ^ { d } ,\tag{1.4}
$$

and, for a parameter configuration $\Theta _ { n }$ , they span the localized network space

$$
\chi L _ { n } ^ { k } = \chi L _ { n } ^ { k } ( \Theta _ { n } ) : = \{ \chi g : g \in L _ { n } ^ { k } ( \Theta _ { n } ) \} = \mathrm { s p a n } \{ \varphi _ { \theta _ { j } ^ { * } } : j = 1 , \dots , n \} .\tag{1.5}
$$

The localized functions are supported away from ∂Ω, so Fourier analysis on $\mathbb { R } ^ { d }$ introduces no boundary singularities. Since $\chi = 1$ on $\Omega ^ { \prime }$ , estimates for the localized error also control the original approximation error on $\Omega ^ { \prime }$

To ensure that localization retains the singularity of each neuron, we choose a connected closed set $U \subset \Omega$ with $\Omega ^ { \prime } \Subset U ^ { \circ }$ and $\chi \geq c _ { 1 } > 0$ on $U$ , and consider parameters in the active set

$$
\begin{array} { r } { \mathcal A _ { \chi } : = \left\{ \binom { w } { b } \in \mathbb S ^ { d } : \{ x : w \cdot x + b = 0 \} \cap U \neq \emptyset \right\} . } \end{array}\tag{1.6}
$$

For the localized estimates, we take $\Theta _ { n } \subset { \mathcal { A } } _ { \chi }$ , so every singular hyperplane intersects a region where $\chi$ is uniformly positive.

Let ρ denote the geodesic distance on $\mathbb { S } ^ { d }$ , and write $\rho _ { \pm } ( \theta , \eta ) : = \operatorname* { m i n } \{ \rho ( \theta , \eta ) , \rho ( - \theta , \eta ) \}$ for the antipodal distance. For a parameter region $A \subset \mathbb { S } ^ { d }$ containing $\Theta _ { n } .$ , the fill distance and antipodal separation distance are

$$
h : = \operatorname* { s u p } _ { \theta \in \mathcal { A } } \operatorname* { m i n } _ { 1 \leq j \leq n } \rho ( \theta , \theta _ { j } ^ { * } ) , \quad \underline { { h } } : = \operatorname* { m i n } _ { i \neq j } \rho _ { \pm } ( \theta _ { i } ^ { * } , \theta _ { j } ^ { * } ) .
$$

In the localized setting, we use $\textstyle A = A _ { \chi } ;$ under antipodal quasi-uniformity, $h \approx \underline { { h } } \approx n ^ { - \frac { 1 } { d } }$

Our analysis uses localized Fourier estimates, integral representations, and spherical approximation tools. The main contributions are as follows.

1. Localized Fourier estimates. We estimate the norms of individual dyadic Fourier blocks of the localized neurons and establish the localization bound

$$
\left| \langle \Delta _ { q , s } \varphi _ { \theta } , \Delta _ { q , s } \varphi _ { \theta ^ { \prime } } \rangle _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \right| \lesssim _ { \gamma } \frac { 2 ^ { - q ( 2 ( k - s ) + 1 ) } } { \left( 2 ^ { q } \rho _ { \pm } ( \theta , \theta ^ { \prime } ) \right) ^ { \gamma } } .\tag{1.7}
$$

Here $\Delta _ { q , s }$ denotes a Littlewood–Paley Sobolev block, with $q \geq 1 , \gamma > d , 0 \leq s < k + \frac { 1 } { 2 }$ and $\rho _ { \pm } ( \theta , \theta ^ { \prime } ) > 0$ . These localized spectral estimates are key tools for approximation and harmonic analysis and underpin several results below.

2. Sharp spectral estimates for Gram and stifness matrices. For $\begin{array} { r } { 0 \leq s < k + \frac { 1 } { 2 } } \end{array}$ , the Sobolev Gram matrices are given by

$$
( K _ { s } ) _ { i j } : = \langle \varphi _ { \theta _ { i } ^ { \ast } } , \varphi _ { \theta _ { j } ^ { \ast } } \rangle _ { \mathcal { H } ^ { s } ( \mathbb { R } ^ { d } ) } = \int _ { \mathbb { R } ^ { d } } ( I - \bigtriangleup ) ^ { \frac { s } { 2 } } \varphi _ { \theta _ { i } ^ { \ast } } ( x ) ( I - \bigtriangleup ) ^ { \frac { s } { 2 } } \varphi _ { \theta _ { j } ^ { \ast } } ( x ) d x , \quad 1 \leq i , j \leq n .\tag{1.8}
$$

We obtain configuration-dependent bounds for all eigenvalues, ordered nonincreasingly. Under antipodal quasi-uniformity, these bounds are sharp:

$$
\lambda _ { j } ( K _ { s } ) \overset { } { \sim } n j ^ { - \frac { d + 2 \left( k - s \right) + 1 } { d } } , \quad \kappa ( K _ { s } ) \overset { } { \sim } n ^ { \frac { d + 2 \left( k - s \right) + 1 } { d } } , \quad 1 \leq j \leq n .\tag{1.9}
$$

The identity $\begin{array} { r } { \big \| \sum _ { j = 1 } ^ { n } a _ { j } \varphi _ { \theta _ { j } ^ { * } } \big \| _ { \mathcal { H } ^ { s } ( \mathbb { R } ^ { d } ) } ^ { 2 } = a ^ { \top } K _ { s } a } \end{array}$ , with $a = ( a _ { 1 } , \ldots , a _ { n } ) ^ { \top }$ , relates these spectral estimates to coeficient stability.

3. Configuration-dependent approximation lower bounds. For targets $f \in \mathcal { H } ^ { r } ( \Omega )$ with $\textstyle r > { \frac { d + { \overline { { 2 } } } k + 1 } { 2 } }$ , we prove, for suficiently large $n .$

$$
\operatorname* { i n f } _ { g _ { n } \in \chi L _ { n } ^ { k } } \| \chi f - g _ { n } \| _ { \mathcal L ^ { 2 } ( \mathbb R ^ { d } ) } \gtrsim n ^ { - \frac { 1 } { 2 } } \underline { { h } } ^ { k + \frac { 1 } { 2 } } \| \chi f \| _ { \mathcal L ^ { 2 } ( \mathbb R ^ { d } ) } .\tag{1.10}
$$

This also bounds the approximation error on Ω from below and gives the order $n ^ { - \frac { d + 2 k + 1 } { 2 d } }$ under antipodal quasi-uniformity. As a complement, we adapt the upper estimates of $[ 6 7 ]$ to parameters in $A _ { \chi }$ , obtaining interior approximation bounds with controlled output coeficients.

4. Bernstein and Marcinkiewicz– $\mathbf { - Z y }$ gmund inequalities. For the localized network spaces with parameters in $\mathcal { A } _ { \chi }$ , we control higher Sobolev norms and establish equal-weight norm discretization:

$$
\begin{array} { r l } & { \| g _ { n } \| _ { \mathcal { H } ^ { t } ( \mathbb { R } ^ { d } ) } \lesssim \displaystyle \underline { { h } } ^ { - ( t - s ) } \| g _ { n } \| _ { \mathcal { H } ^ { s } ( \mathbb { R } ^ { d } ) } , } \\ & { \| g _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } ^ { 2 } \lesssim \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } g _ { n } ( x _ { i } ) ^ { 2 } , \quad g _ { n } \in \chi L _ { n } ^ { k } . } \end{array}\tag{1.11}
$$

The Bernstein inequality holds for $\begin{array} { r } { 0 \leq s < t < k + \frac { 1 } { 2 } } \end{array}$ . The MZ inequality holds for antipodally quasi-uniform parameters and quasi-uniform samples in Ω when $k > \frac { d - 1 } { 2 }$ and m $\gtrsim n$ , with suficient oversampling conditions for smaller k.

5. Discrete least-squares recovery. For prescribed parameters covering the full sphere $\mathbb { S } ^ { d } .$ we obtain recovery estimates without assuming parameter quasi-uniformity. On bounded Lipschitz domains, with $\textstyle k > { \frac { d - 1 } { 2 } }$ and quasi-uniform samples satisfying $m \gtrsim \left( h ^ { \frac { 2 k + 1 } { d + 2 k + 1 } } \underline { { h } } ^ { \frac { d } { d + 2 k + 1 } } \right) ^ { \top } d$ with a suficiently small implicit constant, regularized discrete least squares gives

$$
\| f - f _ { n , m } \| _ { \mathcal { H } ^ { s } ( \Omega ^ { \prime } ) } \lesssim h ^ { r - s } \Big ( \frac { h } { \underline { { h } } } \Big ) ^ { \frac { d s } { d + 2 k + 1 } } \left\{ \| f \| _ { \mathcal { W } ^ { r , p } ( \Omega ) } , \quad \frac { d } { p } < r \leq \frac { d } { 2 } , \quad 2 < p < \infty , \right.\tag{1.12}
$$

where $0 \leq s <$ min $\{ r , k + { \frac { 1 } { 2 } } \}$ . The scheme uses localized neurons and the coeficient penalty $h ^ { 2 k + 1 } \| a \| _ { 2 } ^ { 2 }$ . Under antipodal quasi-uniformity, $m \gtrsim n$ sufices and the rate reduces to $n ^ { - \frac { r - s } { d } }$

6. Inverse approximation and structural conditions. Within the antipodally quasi-uniform setting, for $\begin{array} { r } { \mathbf { \bar { 0 } } < t < \frac { d + 2 k + 1 } { 2 } } \end{array}$ , approximation by two disjoint parameter families whose union has antipodal separation $\gtrsim n ^ { - \frac { 1 } { d } }$ gives

$$
\begin{array} { r l } & { \underset { g _ { n } \in \chi L _ { n } ^ { k } ( \Theta _ { n } ) } { \operatorname* { i n f } } \| f - g _ { n } \| _ { \mathcal L ^ { 2 } ( \mathbb R ^ { d } ) } + \underset { \widetilde { g } _ { n } \in \chi L _ { n } ^ { k } ( \widetilde { \Theta } _ { n } ) } { \operatorname* { i n f } } \| f - \widetilde { g } _ { n } \| _ { \mathcal L ^ { 2 } ( \mathbb R ^ { d } ) } \lesssim n ^ { - \frac { t } { d } } } \\ & { \implies f \in \mathcal B _ { 2 , \infty } ^ { t } ( \mathbb R ^ { d } ) . } \end{array}\tag{1.13}
$$

A coeficient budget yields another inverse result without nestedness. Further results identify the limits of inverse conclusions from unconstrained rates or qualitative interior convergence.

Section 2 compares these results with related work. Sections 3 and 4 give the preliminaries and main results. Section 5 proves the localized Fourier estimates, and Section 6 proves the remaining results.

## 2 Related work

We compare spectral, approximation, sampling, and inverse results, with particular attention to their domains, parameter configurations, and coeficient conditions.

## 2.1 Spectral estimates and conditioning

Spectral perspectives on neural networks include the spectral bias and frequency principle [93, 115], infinite-width neural tangent kernels [49], residual-network kernels on spheres [10], and kernelregression learning curves [13, 16]. Related results study integral-operator spectra approximated by randomly sampled kernel matrices [55, 96] and kernel-matrix spectra when data dimension and sample size grow proportionally [30]. These studies provide spectral descriptions of training, learning, and random kernel models, while our focus is the dependence of finite Gram and stifness spectra on neuron parameter geometry.

For unit-slope ReLU neurons with uniformly spaced breakpoints on [0, 1], [41] establishes the Gram eigenvalue estimates $\lambda _ { j } ( K _ { 0 } ^ { \mathrm { R } } ) ~ \stackrel { } { \sim } ~ n j ^ { - 4 }$ . For ordered breakpoints in one-dimensional Ritz discretizations, [15] gives $\kappa ( K _ { s } ^ { \mathrm { R } } ) \lesssim n h _ { \mathrm { m i n } } ^ { - ( 3 - 2 s ) } , s = 0 , 1$ , for the ReLU mass and derivative stifness matrices, where $h _ { \mathrm { m i n } }$ is the smallest partition interval. Uniform spacing gives upper bounds of order $n ^ { 4 }$ and $n ^ { 2 }$ , consistent with our estimates for $k = 1$ and $s = 0 , 1$ 1. Our results apply in arbitrary dimension, parameter geometry, and give configuration-dependent upper and lower bounds for all eigenvalues.

For shallow ReLU networks on d-dimensional unit balls, [121] obtains $\mu _ { j } \ \approx \ j ^ { - { \frac { d + 3 } { d } } }$ for the continuous $\mathcal { L } ^ { 2 }$ Gram operator with the uniform direction–bias measure on $\mathbb { S } ^ { \bar { d } - 1 } \times [ - 1 , 1 ]$ . Our estimates concern discrete Sobolev matrices of localized $\mathrm { R e L U } ^ { k }$ neurons and depend on parameter geometry without requiring a uniform direction–bias distribution.

For ReL $\boldsymbol { \mathrm { J } } ^ { k }$ neurons with both inputs and parameters on $\mathbb { S } ^ { d }$ , [66] gives geometric bounds for the eigenvalues of the mass and integer-order stifness matrices. Under antipodal quasi-uniformity,

$$
\lambda _ { j } ( K _ { s } ^ { \mathbb { S } } ) \asymp n j ^ { - \frac { d + 2 ( k - s ) + 1 } { d } } , \quad \kappa ( K _ { s } ^ { \mathbb { S } } ) \asymp n ^ { \frac { d + 2 ( k - s ) + 1 } { d } } , \quad 1 \leq j \leq n ,\tag{2.1}
$$

for integer $0 \leq s \leq k$ . These matrices use inner products over the full sphere. Our estimates retain this spectral exponent but apply to localized neurons on bounded Euclidean domains.

For standard piecewise linear nodal bases on shape-regular, quasi-uniform simplicial meshes, the finite element mass and Dirichlet stifness matrices satisfy $\kappa ( \bar { K _ { s } ^ { \mathrm { F E } } } ) \approx n ^ { \frac { 2 s } { d } } \ ( s = 0 , 1 )$ , where n is the number of degrees of freedom [52, 113]. This comparison depends on the basis: already in one dimension, ReLU hinges supplemented by afine functions and nodal hat functions can represent the same piecewise linear space while having substantially diferent Gram conditioning [121]. Connections with numerical discretization also include ReLU representations of linear finite element functions [38] and the MgNet framework linking multigrid with convolutional neural networks [40].

## 2.2 Approximation under prescribed parameter configurations

Universal approximation theorems establish density under suitable activation assumptions [24, 42, 58]; surveys describe the broader approximation theory [91, 25, 84]. Quantitative deep-network approximation theory relates accuracy to network size, depth, and target smoothness [119, 120, 73]. Related results treat deep convolutional approximation on spheres [32], simultaneous approximation of functions and derivatives on manifolds [122], anisotropic and mixed smoothness [116], and sparse compositional targets under parameter norm budgets [47].

For shallow networks with freely chosen hidden parameters, quantitative rates have been studied in Barron and variation spaces [6, 3, 53, 106, 104] and Sobolev spaces [80, 77, 117, 75]. The Barron-type rate $\mathcal { O } ( n ^ { - \frac { 1 } { 2 } } )$ is related to Maurey’s sampling principle [78, 92] and probabilistic or greedy approximation [50, 28, 74, 8], with refined estimates accounting for the activation, parameter geometry, and target regularity [3, 53, 114, 106, 104, 103, 82]. Related greedy analyses are developed in [61, 60]. For shallow $\mathrm { \bar { \mathrm { ~ R e L U } } } ^ { \mathrm { \bar { k } } }$ networks on bounded domains, [39] studies Radon-domain $\mathcal { L } ^ { p }$ ridge integral spaces, their Sobolev embeddings, and randomized approximation rates; at $p = 2$ , the integral space recovers the critical Sobolev space $\mathcal { H } ^ { \frac { d + 2 k + 1 } { 2 } } \ [ 6 7 ]$

Related work on prescribed ridge directions studies density conditions [64], Sobolev approximation on balls [90], and worst-case uniform approximation of Lipschitz functions [51]. Targetindependent centers also appear in spherical zonal networks [81], while fixed random neurons are used in Petrov–Galerkin and discontinuous Galerkin methods for diferential equations [100, 108]. For prescribed ReL $\boldsymbol { \mathbf { \mathit { I } } } ^ { k }$ parameters covering the full sphere, [67] gives bounded-domain $\mathcal { L } ^ { 2 }$ error of order $h ^ { r }$ for $f \in { \mathcal { H } } ^ { r } ( \Omega )$ with $\textstyle r \leq { \frac { d + 2 k + 1 } { 2 } }$ . Our focus below is configuration-dependent lower bounds and saturation for individual smooth targets.

Saturation results identify limiting rates for classical approximation schemes [72, 14, 99]. For degree-k piecewise polynomial spaces $V _ { h }$ on quasi-uniform meshes of size h, [63] establishes

$$
\operatorname* { i n f } _ { v \in V _ { h } } \| f - v \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } \gtrsim h ^ { k + 1 }\tag{2.2}
$$

for suficiently smooth fixed targets with a nonzero derivative of order $k + 1$ . For conditionally positive definite thin plate splines associated with $\triangle ^ { \ell }$ on a bounded domain Ω satisfying an interior cone condition, [98] proves

$$
\| f - I _ { X } f \| _ { \mathcal L ^ { \infty } ( K ) } = o ( h _ { X } ^ { 2 \ell } ) \quad \Longrightarrow \quad \bigtriangleup ^ { \ell } f = 0 \mathrm { ~ i n ~ } \Omega ,\tag{2.3}
$$

where $f \in \mathcal { C } ^ { 2 \ell } ( \Omega ) , I _ { X } f$ is the polynomially augmented interpolant at admissible centers $X$ , and the hypothesis holds for every compact $K \Subset \Omega$ as the fill distance $h _ { X }$ tends to zero.

Saturation has also been studied for neural network approximation operators. For one-dimensiona max-product operators generated by sigmoidal functions, [22] identifies the saturation order $n ^ { - 1 }$ with faster uniform convergence restricted to constant targets. For classical multivariate sigmoidal sampling operators, [21] shows that interior uniform convergence of order $o ( n ^ { - 2 } )$ forces $\triangle f = 0$ under suitable decay and moment conditions, where n denotes the grid resolution. These results identify saturation of prescribed sigmoidal approximation schemes. Our results relate hidden-parameter geometry to approximation lower bounds for shallow $\mathrm { R e L U } ^ { k }$ networks with general parameter configurations in $\mathcal { A } _ { \chi }$ and unrestricted output coeficients.

For prescribed $\mathrm { R e L U } ^ { k }$ neurons on $\mathbb { S } ^ { d }$ , [76] establishes the asymptotic lower bound

$$
\operatorname* { i n f } _ { g _ { n } \in L _ { n } ^ { k , \mathbb { S } } } \| f - g _ { n } \| _ { \mathcal L ^ { 2 } ( \mathbb { S } ^ { d } ) } \gtrsim n ^ { - \frac { 1 } { 2 } } \underline { { h } } ^ { k + \frac { 1 } { 2 } } \| f \| _ { \mathcal L ^ { 2 } ( \mathbb { S } ^ { d } ) } ,\tag{2.4}
$$

where $L _ { n } ^ { k , \mathbb { S } }$ is the span of the prescribed neurons and $f \in \mathcal { H } ^ { r } ( \mathbb { S } ^ { d } )$ is fixed with $r > \frac { d + 2 k + 1 } { 2 }$ Together with [67], this identifies the saturation order $n ^ { - \frac { d + 2 k + 1 } { 2 d } }$ under antipodal quasi-uniformity on the corresponding smooth parity class. Our lower bounds concern bounded Euclidean domains and use separation in $\mathcal { A } _ { \chi }$

## 2.3 Bernstein and Marcinkiewicz–Zygmund inequalities

Continuous and discrete norm estimates have been studied through Sobolev sampling inequalities and scattered-data approximation [85, 86, 2], general MZ inequalities [54], and their connection with weighted least squares [34]. Our focus is the dependence of Bernstein inequalities and equal weight norm discretization on prescribed hidden-parameter geometry.

Bernstein inequalities for spherical basis function spaces $V _ { Z } = \operatorname { s p a n } \{ \phi ( z \cdot \eta ) : z \in Z \}$ generated by positive definite Green kernels and their admissible perturbations are established in [83]. A representative $\mathcal { L } ^ { 2 }$ estimate is

$$
\| v \| _ { \mathcal { H } ^ { r } ( \mathbb { S } ^ { d } ) } \lesssim q _ { Z } ^ { - r } \| v \| _ { \mathcal { L } ^ { 2 } ( \mathbb { S } ^ { d } ) } , \quad v \in V _ { Z } , \quad 0 < r < \beta - \frac { d } { 2 } ,\tag{2.5}
$$

where $q _ { Z }$ is the center separation distance, $\beta$ is the kernel order, and the constant may depend on the mesh ratio of $Z .$ Our Bernstein estimates retain this separation-dependent scaling for localized afine $\mathrm { R e L U } ^ { k }$ neurons, with antipodal separation replacing the separation of kernel centers.

MZ inequalities for spherical polynomials at scattered samples are established in [79] and extended to difusion polynomials on compact manifolds in [33]. In the spherical setting, quasi-uniform samples $X = \{ x _ { i } \} _ { i = } ^ { m }$ <sub>1</sub> with fill distance $h _ { X }$ satisfy

$$
\| p \| _ { \mathcal { L } ^ { 2 } ( \mathbb { S } ^ { d } ) } ^ { 2 } \approx \sum _ { i = 1 } ^ { m } \omega _ { i } | p ( x _ { i } ) | ^ { 2 } , \quad p \in \Pi _ { L } , \quad L \lesssim h _ { X } ^ { - 1 } ,\tag{2.6}
$$

where $\Pi _ { L }$ is the space of spherical polynomials of degree at most $L , \ \omega _ { i }$ are positive partition weights, and the constant in $L \lesssim h _ { X } ^ { - 1 }$ is suficiently small. For quasi-uniform nodes, $\omega _ { i } \approx m ^ { - 1 }$ and the required sample size is of order $L ^ { d }$ , matching the space dimension. These polynomial estimates also support sample-based encoders for learning nonlinear functionals on spheres [118]. Our MZ estimates concern localized ReLU<sup>k</sup> spaces on bounded Euclidean domains instead of polynomial spaces.

## 2.4 Discrete least-squares recovery

Recovery from finite data is studied in statistical learning theory through empirical risk minimization, regularization, and capacity bounds [1, 9, 23], including neural-network risk bounds and near-minimax guarantees [7, 88], random-design regression [43], and random-feature learning [94, 97]. Related analyses treat empirical-feature regularization [35], distributed regularized least squares with Gaussian kernels [45], and hyper-kernel ridge regression for multi-index models [48].

Further sample-based guarantees cover spectral algorithms for functional linear regression [31], nonlinear functional regression with kernel-embedded deep networks [101], and neural-network distribution regression with two-stage sampling [102]. Iterative methods include early stopping and multi-pass stochastic gradient descent with general convex losses [44, 57], and truncated kernel stochastic gradient descent for least squares on spheres [5]. Least-squares and collocation formulations also appear in kernel methods [37], extreme learning machines, and neural-network methods for diferential equations [46, 95, 17, 111]. Our focus is recovery of continuous approximation rates from quasi-uniform function samples using prescribed hidden parameters.

Discrete least-squares approximation in finite-dimensional spaces from random evaluations is studied in [19, 18]. For an n-dimensional trial space, the sampling measure and weights based on its Christofel function in [20] yield high-probability stability in the regime

$$
m \stackrel { > } { \sim } n \log n , \quad \frac { 1 } { 2 } I _ { n } \preceq G _ { n , m } \preceq \frac { 3 } { 2 } I _ { n } ,\tag{2.7}
$$

where $G _ { n , m }$ is the weighted empirical Gram matrix in an $\mathcal { L } ^ { 2 } .$ -orthonormal basis. These results cover general trial spaces and give accuracy guarantees for stabilized least-squares estimators. Our result uses equal-weight quasi-uniform samples and a quadratic coeficient penalty. The suficient sampling density is expressed through parameter fill distance and antipodal separation, without assuming parameter quasi-uniformity or requiring a separate spectral check of $G _ { n , m }$

For radial basis spaces on spheres, [56] studies continuous and discrete least-squares surface fitting with separate center and data sets. On bounded Euclidean domains, [59] establishes continuous Sobolev error estimates for discrete least squares using positive definite kernels whose native spaces are equivalent to Sobolev spaces. Its stability conditions quantify the relationship between sampling density, center separation, and native-space regularization.

For ReLU<sup>k</sup> neurons on S<sup>d</sup>, [68] proves that equal-weight least squares recovers the continuous approximation rate when $k >  { \frac { d - 1 } { 2 } }$ , the parameters are antipodally quasi-uniform, and the samples are quasi-uniform with m ${ \gtrsim } n .$ For targets satisfying the required parity condition, the least-squares approximant $f _ { n , m } ^ { \mathbb { S } }$ satisfies

$$
\| f - f _ { n , m } ^ { \ S } \| _ { \mathcal { H } ^ { s } ( \mathbb { S } ^ { d } ) } \lesssim n ^ { - \frac { r - s } { d } } \left\{ \| f \| _ { \mathcal { H } ^ { r } ( \mathbb { S } ^ { d } ) } , \quad \frac { d } { p } < r \leq \frac { d } { 2 } , \quad 2 < p < \infty , \right.\tag{2.8}
$$

for $0 \leq s < \operatorname* { m i n } \{ r , k + { \frac { 1 } { 2 } } \}$ and a suficiently large constant in the sampling condition. The same work treats residual least squares for elliptic spectral multipliers and bounded-domain recovery through spherical lifting. Our formulation uses localized neurons with parameters covering the full sphere and a scalar quadratic coeficient penalty. It gives interior error estimates on bounded Lipschitz domains for the same two target-regularity regimes, without a parity condition or parameter quasiuniformity; the sampling density and error bounds depend explicitly on parameter coverage and separation.

## 2.5 Inverse approximation theory

Inverse approximation theory relates convergence rates to target regularity and membership in approximation spaces. Classical results cover polynomial and spline approximation, nonlinear approximation on adaptive partitions, and scattered-data approximation [11, 87, 27, 99, 26, 112]. Related neural-network studies describe approximation and variation spaces [106, 105] and establish converse results for recurrent models [110]. Our focus is how hidden-parameter geometry and coeficient growth determine the regularity information contained in shallow-network approximation rates.

Bernstein inequalities yield inverse theorems for polynomial and spline approximation [27, 99] and Besov characterizations through nested spherical basis function spaces [83]. Related kernel results include native-space inverse theorems for radial basis interpolation [98] and localized Bernstein estimates on bounded Euclidean domains [36]. For nested antipodally quasi-uniform parameter sets, our localized Bernstein inequality similarly gives

$$
\operatorname* { i n f } _ { g _ { n } \in \chi L _ { n } ^ { k } ( \Theta _ { n } ) } \lVert f - g _ { n } \rVert _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \lesssim n ^ { - \frac { t } { d } } \quad \Longrightarrow \quad f \in \mathcal { B } _ { 2 , \infty } ^ { t } ( \mathbb { R } ^ { d } ) , \quad 0 < t < k + \frac { 1 } { 2 } .\tag{2.9}
$$

This conclusion is limited by the regularity of individual neurons. Our inverse results beyond this threshold require coeficient control or simultaneous approximation on two suitably separated parameter families.

For deep networks, [47] gives approximation upper bounds under a budget on the product of layer Frobenius norms. Converse results for shallow networks characterize function spaces through bounded representations. For shallow ReLU networks, [29] proves that pointwise convergence to a continuous target under a uniform path-norm bound implies Barron-space membership. For a normalized $\mathrm { R e L U } ^ { \sim }$ dictionary D and approximants $\begin{array} { r } { g _ { n } = \sum _ { j = 1 } ^ { n } a _ { j , n } \phi _ { j , n } } \end{array}$ with $\phi _ { j , n } \in \mathcal { D }$ , the variationspace characterization in [105] gives

$$
\| f - g _ { n } \| _ { \mathcal L ^ { 2 } ( \Omega ) } \to 0 , \quad \operatorname* { s u p } _ { n } \sum _ { j = 1 } ^ { n } | a _ { j , n } | \leq M \quad \Longrightarrow \quad \| f \| _ { \mathcal D } \leq M ,\tag{2.10}
$$

where $\| \cdot \| _ { \mathcal { D } }$ denotes the variation norm. Sharp embeddings between quasi-Banach Besov spaces and shallow $\mathrm { R e L U } ^ { k }$ variation spaces are established in [62]. These results concern bounded representations and function-space embeddings; our inverse results relate growing coeficient budgets and quantitative approximation rates to Besov regularity.

For $\operatorname { R e L U } ^ { k }$ networks on $\mathbb { S } ^ { d }$ , [68] derives Sobolev inverse conclusions below $k + { \textstyle { \frac { 1 } { 2 } } }$ from a Bernstein inequality along nested antipodally quasi-uniform parameter sequences. The coeficient-controlled integral characterization in [67] reaches the critical Sobolev order $\textstyle { \frac { d + 2 k + 1 } { 2 } }$ , while [70] treats growing coeficient budgets. For quasi-uniform parameters and $\begin{array} { r } { g _ { n } ( \eta ) = \sum _ { j = 1 } ^ { \tilde { n } } a _ { j , n } \sigma _ { k } ( \bar { \theta _ { j , n } ^ { * } } \cdot \eta ) } \end{array}$ , the latter establishes

$$
\begin{array} { r l r } {  { \| f - g _ { n } \| _ { \mathcal L ^ { 2 } ( \mathbb S ^ { d } ) } \lesssim n ^ { - \frac { d + 2 k + 1 } { 2 d } + \alpha } , } } & { { ( n \sum _ { j = 1 } ^ { n } | a _ { j , n } | ^ { 2 } ) ^ { \frac { 1 } { 2 } } \lesssim n ^ { \alpha } } } \\ & { } & \\ & { \implies f \in { \mathcal B } _ { 2 , \infty , * } ^ { \frac { d + 2 k + 1 } { 2 } - \alpha d } ( \mathbb S ^ { d } ) , } & \end{array}\tag{2.11}
$$

for $\textstyle 0 < \alpha < { \frac { d + 2 k + 1 } { 2 d } }$ , where the subscript ∗ records the spherical parity restriction. It also shows that an unconstrained rate alone can fail to imply regularity above the neuron threshold, and recovers the inverse conclusion from two parameter sequences whose union is antipodally separated. Our results establish corresponding whole-space Besov conclusions for localized afine neurons with parameters in $\mathcal { A } _ { \chi }$

## 3 Preliminaries

Throughout the paper, for $A , B \ge 0$ , we write $A \lesssim B$ whenever $A \leq C B$ for some constant $C > 0$ . We write $A \gtrsim B$ when $B \lesssim A ,$ , and $A \gtrsim B$ when both $A \lesssim B$ and $A \gtrsim B$ hold. The implicit constants may change from one occurrence to another and depend only on the fixed parameters $d ,$ $k , s , r , \Omega ^ { \prime } , U ,$ Ω and $\chi$ unless otherwise stated. In the antipodal quasi-uniform setting $h \leq C ^ { * } \underline { { h } }$ they may additionally depend on the quasi-uniformity constant $C ^ { * }$ , but not on $n .$ . Dependence on any other parameters is stated explicitly or indicated by a subscript on $\gtrsim , \lesssim , \gtrsim$ . For example, if $A \leq C B$ with C depending on an index $\gamma ,$ we write $A \lesssim _ { \gamma } B$

We also use the notation $C _ { 1 } , C _ { 2 } , \ldots$ and $c _ { 1 } , c _ { 2 } , \ldots$ to denote constants, whose dependence will be clear from the context. Uppercase constants are considered to be large constants in the sense that the corresponding results remain valid if they are increased; lowercase constants are considered to be small constants in the sense that the results remain valid if they are decreased.

## 3.1 Network spaces and localization

We retain the domains $\Omega , \Omega ^ { \prime } ,$ , the set $U ,$ and the cutof $\chi$ introduced in Section 1. The prescribed network spaces, localized neurons, and localized spaces are defined in (1.2), (1.4), and (1.5). For the same prescribed parameters, the coeficient-constrained classes are

$$
\begin{array} { r l r } & { \displaystyle { \cal L } _ { n , M } ^ { k , \alpha } : = \Big \{ \displaystyle \sum _ { j = 1 } ^ { n } a _ { j } \sigma _ { k } \big ( w _ { j } ^ { * } \cdot \circ + b _ { j } ^ { * } \big ) : \left( n \sum _ { j = 1 } ^ { n } a _ { j } ^ { 2 } \right) ^ { \frac { 1 } { 2 } } \leq n ^ { \alpha } M \Big \} , } & \\ & { \displaystyle { \chi \cal L } _ { n , M } ^ { k , \alpha } : = \big \{ \chi f _ { n } : \ f _ { n } \in { \cal L } _ { n , M } ^ { k , \alpha } \big \} = \Big \{ \displaystyle \sum _ { j = 1 } ^ { n } a _ { j } \varphi \theta _ { j } ^ { * } : \left( n \sum _ { j = 1 } ^ { n } a _ { j } ^ { 2 } \right) ^ { \frac { 1 } { 2 } } \leq n ^ { \alpha } M \Big \} . } & \end{array}\tag{3.1}
$$

The active parameter set is given in (1.6), and the associated Gram and stifness matrices $K _ { s }$ are defined in (1.8).

## 3.2 Parameter and sample geometry

Throughout, | · | denotes the Euclidean norm on $\mathbb { R } ^ { d }$ and $\rho$ denotes the geodesic distance on the unit sphere

$$
\mathbb { S } ^ { d } = \{ x \in \mathbb { R } ^ { d + 1 } : | x | = 1 \} ,
$$

that is,

$$
\rho ( \theta , \eta ) = \operatorname { a r c c o s } ( \theta \cdot \eta ) , \quad \theta , \eta \in \mathbb { S } ^ { d } .\tag{3.2}
$$

Definition 3.1 (Quasi-uniformity and antipodal quasi-uniformity). Let $d \in \mathbb { N }$ . For a finite set $\Theta _ { n } = \{ \theta _ { j } ^ { * } \} _ { j = 1 } ^ { n } \subset \mathcal { A } \subset \mathbb { S } ^ { d }$ , define

$$
h = h _ { n } : = \operatorname* { s u p } _ { \theta \in \mathcal { A } } \operatorname* { m i n } _ { 1 \le j \le n } \rho ( \theta , \theta _ { j } ^ { * } ) , \qquad \tilde { h } = \tilde { h } _ { n } : = \operatorname* { m i n } _ { i \ne j } \rho ( \theta _ { i } ^ { * } , \theta _ { j } ^ { * } ) , \qquad h = \underline { { h } } _ { n } : = \operatorname* { m i n } _ { i \ne j } \rho _ { \pm } ( \theta _ { i } ^ { * } , \theta _ { j } ^ { * } ) ,
$$

where

$$
\rho _ { \pm } ( \theta , \eta ) = \operatorname* { m i n } \Big \{ \rho ( \theta , \eta ) , \rho ( \theta , - \eta ) \Big \} .
$$

The set $\Theta _ { n }$ is called quasi-uniform or antipodally quasi-uniform on A if there are constants such that

$$
h _ { n } \lesssim \tilde { h } _ { n } , \qquad h _ { n } \lesssim \underline { { h } } _ { n } ,\tag{3.3}
$$

respectively.

Similarly, a collection $X = \{ x _ { i } \} _ { i = 1 } ^ { m } \subset \Omega$ is called quasi-uniform on $\Omega \ i f h _ { X } \lesssim \underline { { h } } _ { X }$ , where

$$
h _ { X } : = \operatorname* { s u p } _ { x \in \Omega } \operatorname* { m i n } _ { 1 \leq i \leq m } | x - x _ { i } | , \qquad \underline { { h } } _ { X } : = \operatorname* { m i n } _ { i \neq j } | x _ { i } - x _ { j } | .
$$

## 3.3 Besov spaces and Sobolev spaces

We describe the Sobolev inner product using the Fourier transform, then introduce the dyadic blocks used in the Besov norms.

For $f \in \mathcal { L } ^ { 1 } ( \mathbb { R } ^ { d } ) \cap \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } )$ , define the Fourier transform by

$$
\widehat { f } ( \xi ) = \int _ { \mathbb R ^ { d } } f ( { \boldsymbol x } ) e ^ { - i { \boldsymbol \xi } \cdot { \boldsymbol x } } d { \boldsymbol x } , \quad { \boldsymbol \xi } \in \mathbb R ^ { d } .\tag{3.4}
$$

The transform extends to $\mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } )$ , with inverse understood in the $\mathcal { L } ^ { 2 }$ sense:

$$
f ( x ) = { \frac { 1 } { ( 2 \pi ) ^ { d } } } \int _ { \mathbb { R } ^ { d } } { \widehat { f } } ( \xi ) e ^ { i x \cdot \xi } d \xi , \quad \operatorname { i n } { \mathcal { L } } ^ { 2 } ( \mathbb { R } ^ { d } ) .\tag{3.5}
$$

Let $r \geq 0$ , and let $\triangle = \sum _ { j = 1 } ^ { d } \frac { \partial ^ { 2 } } { \partial x _ { j } ^ { 2 } }$ denote the Laplace operator. Parseval’s identity gives

$$
\int _ { \mathbb { R } ^ { d } } ( I - \bigtriangleup ) ^ { \frac { r } { 2 } } f ( x ) \overline { { ( I - \bigtriangleup ) ^ { \frac { r } { 2 } } g ( x ) } } d x = \frac { 1 } { ( 2 \pi ) ^ { d } } \int _ { \mathbb { R } ^ { d } } \widehat { f } ( \xi ) \overline { { \widehat { g } ( \xi ) } } ( 1 + | \xi | ^ { 2 } ) ^ { r } d \xi .
$$

Therefore, the inner product on $\mathcal { H } ^ { r } ( \mathbb { R } ^ { d } )$ is defined by

$$
\langle f , g \rangle _ { \mathcal { H } ^ { r } ( \mathbb { R } ^ { d } ) } : = \frac { 1 } { ( 2 \pi ) ^ { d } } \int _ { \mathbb { R } ^ { d } } \widehat { f } ( \xi ) \overline { { \widehat { g } ( \xi ) } } ( 1 + | \xi | ^ { 2 } ) ^ { r } d \xi .\tag{3.6}
$$

$$
\mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) = \mathcal { H } ^ { 0 } ( \mathbb { R } ^ { d } )
$$

$$
D \subset \mathbb { R } ^ { d }
$$

$$
\begin{array} { r } { \| f \| _ { \mathcal L ^ { 2 } ( D ) } ^ { 2 } : = \int _ { D } | f ( x ) | ^ { 2 } d x ; } \end{array}
$$

$$
\begin{array} { r } { \| f \| _ { \mathcal L ^ { 2 } ( \mathbb R ^ { d } ) } ^ { 2 } = \frac { 1 } { ( 2 \pi ) ^ { d } } \int _ { \mathbb R ^ { d } } | \widehat { f } ( \xi ) | ^ { 2 } d \xi } \end{array}
$$

To define the Besov norms, fix a smooth cutof function $\zeta$ satisfying the following conditions (see, e.g., [89, (3.6)]):

$$
\zeta \in \mathcal { C } ^ { \infty } (  { \mathbb { R } } ) , \quad \zeta \geq 0 , \quad \operatorname { s u p p } ( \zeta ) \subset [ 1 / 2 , 2 ] ,\tag{3.7}
$$

$$
\zeta ( t ) > c _ { 2 } > 0 , \qquad t \in [ 3 / 5 , 5 / 3 ] ,\tag{3.8}
$$

$$
\zeta ( t ) + \zeta ( 2 t ) = 1 , \quad t \in [ 1 / 2 , 1 ] .\tag{3.9}
$$

A standard consequence of these properties is the partition-of-unity identity

$$
1 = \sum _ { q = 0 } ^ { \infty } \zeta ( 2 ^ { - q } t ) , \qquad t \geq 1 ,
$$

$$
1 = ( 1 - \zeta ) ( t ) + \zeta ( t ) = ( 1 - \zeta ) ( t ) + \sum _ { q = 0 } ^ { \infty } \zeta ( 2 ^ { - q } t ) , \qquad 0 \leq t < 1 .\tag{3.10}
$$

Definition 3.2 (Littlewood–Paley Sobolev operators). For $s \geq 0$ , the Littlewood–Paley Sobolev operators $\Delta _ { q , s }$ are the Fourier multiplier operators

$$
\widehat { \Delta _ { q , s } f } ( \xi ) = \zeta _ { q , s } ( \xi ) \widehat { f } ( \xi ) , \quad q = 0 , 1 , \ldots ,\tag{3.11}
$$

given by the multipliers

$$
\zeta _ { 0 , s } ( \xi ) = \Big [ 1 - \sum _ { q = 1 } ^ { \infty } \zeta \Big ( \frac { | \xi | } { 2 ^ { q } } \Big ) \Big ] ( 1 + | \xi | ^ { 2 } ) ^ { s / 2 } , \qquad \zeta _ { q , s } ( \xi ) = \zeta \Big ( \frac { | \xi | } { 2 ^ { q } } \Big ) ( 1 + | \xi | ^ { 2 } ) ^ { s / 2 } , \quad q \geq 1 .\tag{3.12}
$$

For $q \geq 1$ , these operators localize to $| \xi | \lesssim 2 ^ { q }$ ; the block $q = 0$ contains the low frequencies. We call $\Delta _ { q , s } f$ a Littlewood–Paley Sobolev block.

Definition 3.3 (Besov and Sobolev spaces). For $r \geq 0 , 1 \leq p \leq \infty$ , the Besov spaces $B _ { 2 , p } ^ { r } ( \mathbb { R } ^ { d } )$ are defined by the following norms:

$$
\| f \| _ { \mathcal { B } _ { 2 , p } ^ { r } ( \mathbb { R } ^ { d } ) } = \left\{ \begin{array} { c c } { \displaystyle \left( \sum _ { q = 0 } ^ { \infty } \| \Delta _ { q , r } f \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } ^ { p } \right) ^ { \frac { 1 } { p } } , } & { \quad 1 \leq p < \infty } \\ { \displaystyle \operatorname* { s u p } _ { q \in \mathbb { N } _ { 0 } } \| \Delta _ { q , r } f \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } , } & { \quad p = \infty . } \end{array} \right.\tag{3.13}
$$

In particular, for $p = 2$ , the Besov space $B _ { 2 , 2 } ^ { r } ( \mathbb { R } ^ { d } )$ is equivalent to the Sobolev space $\mathcal { H } ^ { r } ( \mathbb { R } ^ { d } )$

For any bounded domain $\Omega \subset \mathbb { R } ^ { d }$ , the Besov spaces $B _ { 2 , p } ^ { r } ( \Omega )$ and Sobolev spaces $\mathcal { H } ^ { r } ( \Omega )$ are defined by the norms

$$
\| g \| _ { \mathcal { B } _ { 2 , p } ^ { r } ( \Omega ) } = \operatorname* { i n f } _ { f | _ { \Omega } = g } \| f \| _ { \mathcal { B } _ { 2 , p } ^ { r } ( \mathbb { R } ^ { d } ) } , \qquad \| g \| _ { \mathcal { H } ^ { r } ( \Omega ) } = \operatorname* { i n f } _ { f | _ { \Omega } = g } \| f \| _ { \mathcal { H } ^ { r } ( \mathbb { R } ^ { d } ) } .\tag{3.14}
$$

We also use the Bessel potential Sobolev spaces $\mathcal { W } ^ { r , p }$ for $r \geq 0$ and $1 < p < \infty$ , defined by

$$
\| f \| _ { \mathcal { W } ^ { r , p } ( \mathbb { R } ^ { d } ) } : = \| ( I - \bigtriangleup ) ^ { \frac { r } { 2 } } f \| _ { \mathcal { L } ^ { p } ( \mathbb { R } ^ { d } ) } , \quad \| g \| _ { \mathcal { W } ^ { r , p } ( \Omega ) } : = \operatorname* { i n f } _ { f | _ { \Omega = g } } \| f \| _ { \mathcal { W } ^ { r , p } ( \mathbb { R } ^ { d } ) } .\tag{3.15}
$$

For $p = 2 \ o r \ r \in \mathbb { N }$ , this norm is equivalent to the usual weak-derivative Sobolev norm on Lipschitz domains; for $p = 2 , \mathcal { W } ^ { r , 2 } = \mathcal { H } ^ { r }$ with equivalent norms.

Remark 3.1. In the summation (3.10), the dyadic supports have finite overlap, with at most two nonzero terms at each frequency. Hence the Littlewood–Paley square sum is equivalent to the Sobolev norm

$$
\| f \| _ { \mathcal H ^ { r } ( \mathbb R ^ { d } ) } ^ { 2 } = \frac { 1 } { ( 2 \pi ) ^ { d } } \int _ { \mathbb R ^ { d } } \big | \widehat f ( \xi ) \big | ^ { 2 } ( 1 + | \xi | ^ { 2 } ) ^ { r } d \xi .\tag{3.16}
$$

## 4 Main results

We present the main results in the network settings introduced above. These results relate hidden-parameter geometry to approximation accuracy, matrix spectra, recovery from discrete data, and inverse regularity.

We begin with the dyadic norm estimate for individual localized neurons and the of-diagonal estimate for their interactions. These are used in the spectral, approximation lower-bound, and norm-inequality proofs. Section 5 proves the two estimates separately.

Theorem 4.1 (Localized Fourier estimates). For every $\begin{array} { r } { 0 \leq s < k + \frac { 1 } { 2 } } \end{array}$ , there exists $q _ { 0 } \in \mathbb { N }$ such that

$$
\| \Delta _ { q , s } \varphi _ { \theta } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } ^ { 2 } \asymp 2 ^ { - q ( 2 ( k - s ) + 1 ) } , \quad q \geq q _ { 0 }
$$

uniformly for $\theta \in A _ { x }$

For every $\gamma > d$ and $\theta , \theta ^ { \prime } \in \mathcal { A } _ { \chi }$ with $\rho _ { \pm } ( \theta , \theta ^ { \prime } ) > 0$

$$
\Bigl | \langle \Delta _ { q , s } \varphi _ { \theta } , \Delta _ { q , s } \varphi _ { \theta ^ { \prime } } \rangle _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \Bigr | \lesssim _ { \gamma } \frac { 2 ^ { - q ( 2 ( k - s ) + 1 ) } } { ( 2 ^ { q } \rho _ { \pm } ( \theta , \theta ^ { \prime } ) ) ^ { \gamma } } , \qquad q \geq 1 .\tag{4.1}
$$

We next consider approximation in the prescribed network spaces and the spectra of the associated matrices.

Approximation upper bounds on bounded Euclidean domains were established in [67] using parameters covering the full sphere. The next theorem adapts these bounds to coverage of $\mathcal { A } _ { \chi }$ , with the same rates. This makes the coverage condition compatible with the spectral and approximation lower bounds, which require each singular hyperplane to intersect U, where χ is uniformly positive.

Theorem 4.2 (Approximation rates). Let $\begin{array} { r } { d , n \in \mathbb { N } , k \in \mathbb { N } _ { 0 } , r \leq \frac { d + 2 k + 1 } { 2 } , 0 \leq s < \operatorname* { m i n } \{ k + \frac { 1 } { 2 } , r \} } \end{array}$ and $\{ \theta _ { j } ^ { * } \} _ { j = 1 } ^ { n } \subset { \mathcal { A } _ { \chi } }$ . Then there exists $h _ { \delta }$ , independent of n and $\{ \theta _ { j } ^ { * } \} _ { j = 1 } ^ { n }$ , such that, whenever $h \leq h _ { \delta }$

$$
\operatorname* { i n f } _ { g _ { n } \in L _ { n } ^ { k } } \| f - g _ { n } \| _ { \mathcal H ^ { s } ( \Omega ^ { \prime } ) } \lesssim h ^ { r - s } \| f \| _ { \mathcal H ^ { r } ( \Omega ) } , \quad f \in \mathcal H ^ { r } ( \Omega ) .\tag{4.2}
$$

In particular, $i f \ \{ \theta _ { j } ^ { * } \} _ { j = 1 } ^ { n }$ is quasi-uniform on $\mathcal { A } _ { \chi }$ , then with $\textstyle \alpha = { \frac { d + 2 k + 1 - 2 r } { 2 d } }$ , there exists $M $ $\| f \| _ { \mathcal { H } ^ { r } ( \Omega ) }$ such that

$$
\operatorname* { i n f } _ { g _ { n } \in L _ { n , M } ^ { k , \alpha } } \| f - g _ { n } \| _ { \mathcal H ^ { s } ( \Omega ^ { \prime } ) } \lesssim n ^ { - \frac { r - s } { d } } \| f \| _ { \mathcal H ^ { r } ( \Omega ) } , \quad f \in \mathcal H ^ { r } ( \Omega ) .\tag{4.3}
$$

Remark 4.1 (Parameter coverage and separation). The upper bound (4.2) depends on the fill distance h and requires no antipodal separation. Quasi-uniformity gives $h \gtrsim n ^ { - { \frac { 1 } { d } } }$ . The spectral stability and lower bounds instead depend on $\underline { { h } } .$ adding nearly coincident neurons can reduce separation without substantially improving coverage. Under antipodal quasi-uniformity,

$$
h \stackrel { - } { \sim } \underline { { { h } } } \stackrel { } { \sim } n ^ { - \frac { 1 } { d } } ,\tag{4.4}
$$

so both kinds of estimates can be expressed in terms of the network width.

With hidden parameters prescribed, approximation reduces to choosing the output coeficients. Continuous $\mathcal { H } ^ { s }$ least squares in ${ \boldsymbol { \chi L } } _ { n } ^ { k }$ leads to a system with matrix $K _ { s } ,$ whose spectrum controls conditioning and coeficient stability.

Theorem 4.3 (Eigenvalues of Gram and stifness matrices). Let $d , n , k , s ,$ , and $\{ \theta _ { j } ^ { * } \} _ { j = 1 } ^ { n }$ be as in Theorem 4.2. Then the eigenvalues of $K _ { s }$ , ordered nonincreasingly, satisfy

$$
\operatorname* { m a x } \left\{ \underline { { h } } ^ { 2 ( k - s ) + 1 } \Big ( \frac { n } { j } \Big ) ^ { \frac { 2 ( k - s ) + 1 } { d } } , ( n ^ { \frac { 1 } { \lambda } } \underline { { h } } ) ^ { d + 2 ( k - s ) + 1 } n j ^ { - \frac { d + 2 ( k - s ) + 1 } { d } } \right\} \lesssim \lambda _ { j } ( K _ { s } ) \lesssim n j ^ { - \frac { d + 2 ( k - s ) + 1 } { d } } , \quad 1 \leq j \leq n .\tag{4.5}
$$

In particular, $i f \left\{ \theta _ { j } ^ { * } \right\} _ { j = 1 } ^ { n }$ is antipodally quasi-uniform,

$$
\lambda _ { j } ( K _ { s } ) \stackrel { } { \sim } n j ^ { - \frac { d + 2 ( k - s ) + 1 } { d } } , \quad 1 \le j \le n .\tag{4.6}
$$

The connection with coeficient stability follows from the identity

$$
a ^ { \top } K _ { s } a = \| f _ { n } \| _ { \mathcal H ^ { s } ( \mathbb R ^ { d } ) } ^ { 2 } , \quad f _ { n } = \sum _ { j = 1 } ^ { n } a _ { j } \varphi _ { \theta _ { j } ^ { * } } , \quad a = ( a _ { 1 } , \dots , a _ { n } ) ^ { \top } .\tag{4.7}
$$

Thus the smallest eigenvalue controls the coeficient norm, giving

$$
\| a \| _ { 2 } \lesssim \underline { { h } } ^ { - ( k - s + \frac { 1 } { 2 } ) } \| f _ { n } \| _ { \mathcal { H } ^ { s } ( \mathbb { R } ^ { d } ) } , \quad 0 \leq s < k + \frac { 1 } { 2 } ,\tag{4.8}
$$

including the $\mathcal { L } ^ { 2 }$ estimate at $s = 0$ . Under antipodal quasi-uniformity, $\kappa ( K _ { s } ) \asymp n ^ { \frac { d + 2 ( k - s ) + 1 } { d } }$ . This conditioning depends on the neuron representation: an invertible change of basis preserves the approximation space and best errors but can change the condition number.

The next theorem complements the approximation upper bounds by relating antipodal separation to a lower bound on the approximation error for fixed smooth targets.

Theorem 4.4 (Configuration-dependent lower bounds). Let $d , n , k , s$ , and $\{ \theta _ { j } ^ { * } \} _ { j = 1 } ^ { n }$ be as in Theorem 4.2, and suppose that $\textstyle r > { \frac { d + 2 k + 1 } { 2 } }$ . Then, for every $f \in \mathcal { H } ^ { r } ( \Omega )$ , there exists $n _ { 0 }$ , depending only on $d , r$ and $\| \chi f \| _ { \mathcal { H } ^ { r } ( \Omega ) } \| \chi f \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } ^ { - 1 }$ , such that, whenever $n \geq n _ { 0 }$

$$
\operatorname* { i n f } _ { g _ { n } \in L _ { n } ^ { k } } \| f - g _ { n } \| _ { \mathcal H ^ { s } ( \Omega ) } \gtrsim \operatorname* { i n f } _ { f _ { n } \in \mathcal X , L _ { n } ^ { k } } \| \chi f - f _ { n } \| _ { \mathcal L ^ { 2 } ( \mathbb R ^ { d } ) } \gtrsim n ^ { - \frac { 1 } { 2 } } \underline { h } ^ { k + \frac { 1 } { 2 } } \| \chi f \| _ { \mathcal L ^ { 2 } ( \Omega ) } \geq n ^ { - \frac { 1 } { 2 } } \underline { h } ^ { k + \frac { 1 } { 2 } } \| f \| _ { \mathcal L ^ { 2 } ( \Omega ^ { \prime } ) } .\tag{4.9}
$$

In particular, $i f \left\{ \theta _ { j } ^ { * } \right\} _ { j = 1 } ^ { n }$ is antipodally quasi-uniform,

$$
\operatorname* { i n f } _ { g _ { n } \in L _ { n } ^ { k } } \| f - g _ { n } \| _ { \mathcal H ^ { s } ( \Omega ) } \gtrsim \operatorname* { i n f } _ { f _ { n } \in \chi L _ { n } ^ { k } } \| \chi f - f _ { n } \| _ { L ^ { 2 } ( \mathbb R ^ { d } ) } \gtrsim n ^ { - \frac { d + 2 k + 1 } { 2 d } } \| \chi f \| _ { L ^ { 2 } ( \Omega ) } \geq n ^ { - \frac { d + 2 k + 1 } { 2 d } } \| f \| _ { L ^ { 2 } ( \Omega ^ { \prime } ) } .\tag{4.10}
$$

Remark 4.2 (Comparison of upper and lower bounds). For a fixed target $f \in \mathcal { H } ^ { r } ( \Omega )$ with $r \mathrm { ~ > ~ }$ $\textstyle { \frac { d + 2 k + 1 } { 2 } }$ , antipodal quasi-uniformity gives

$$
\begin{array} { r l r } & { } & { \underset { g _ { n } \in L _ { n } ^ { k } } { \operatorname* { i n f } } \| f - g _ { n } \| _ { \mathcal L ^ { 2 } ( \Omega ^ { \prime } ) } \lesssim n ^ { - \frac { d + 2 k + 1 } { 2 d } } \| f \| _ { \mathcal H ^ { \frac { d + 2 k + 1 } { 2 } } ( \Omega ) } , } \\ & { } & { \underset { g _ { n } \in L _ { n } ^ { k } } { \operatorname* { i n f } } \| f - g _ { n } \| _ { \mathcal L ^ { 2 } ( \Omega ) } \gtrsim n ^ { - \frac { d + 2 k + 1 } { 2 d } } \| \chi f \| _ { \mathcal L ^ { 2 } ( \mathbb R ^ { d } ) } , } \end{array}\tag{4.11}
$$

The lower bound holds for suficiently large $n ,$ with the threshold depending on $f .$ The powers of n match, but the domains difer: the upper bound is on $\Omega ^ { \prime }$ and the lower bound on $\Omega .$ . Thus the estimates give matching orders rather than a two-sided bound on one domain.

Bernstein inequalities are fundamental tools in approximation theory, controlling stronger norms of approximants in terms of weaker norms and the approximation scale.

Theorem 4.5 (Bernstein inequality). Let $d , n , k , s ,$ , and $\{ \theta _ { j } ^ { * } \} _ { j = 1 } ^ { n }$ be as in Theorem 4.2. Then, for $\begin{array} { r } { s < r < k + \frac { 1 } { 2 } } \end{array}$

$$
\| f _ { n } \| _ { \mathcal { H } ^ { r } ( \mathbb { R } ^ { d } ) } \lesssim \underline { { h } } ^ { - ( r - s ) } \| f _ { n } \| _ { \mathcal { H } ^ { s } ( \mathbb { R } ^ { d } ) } , \quad f _ { n } \in \chi L _ { n } ^ { k } .\tag{4.12}
$$

The restriction $\textstyle r < k + { \frac { 1 } { 2 } }$ reflects the Sobolev regularity of the individual localized neurons. This threshold difers from the target smoothness $\textstyle { \frac { d + 2 k + 1 } { 2 } }$ allowed in Theorem 4.2: the approximation spaces can exploit target regularity beyond the regularity of their basis functions.

Marcinkiewicz–Zygmund inequalities provide stable discretizations of continuous norms. Given $X = \{ x _ { i } \} _ { i = 1 } ^ { m }$ , the following theorem applies the equal-weight norm

$$
\| \boldsymbol { f } \| _ { \ell ^ { 2 } ( X ) } ^ { 2 } : = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \boldsymbol { f } ( \boldsymbol { x } _ { i } ) ^ { 2 } .\tag{4.13}
$$

to the localized network spaces.

Theorem 4.6 (Marcinkiewicz–Zygmund inequality). Let $d , n , k ,$ and $\{ \theta _ { j } ^ { * } \} _ { j = 1 } ^ { n }$ be as in Theorem $4 . 2 ,$ and fix $\begin{array} { r } { \frac { d } { 2 } < \tau < \frac { d + 1 } { 2 } } \end{array}$ . There exists $C _ { 1 } > 0$ such that, if $X = \{ x _ { i } \} _ { i = 1 } ^ { m } \subset \Omega$ is quasi-uniform and

$$
m \geq C _ { 1 } \left\{ \frac { h ^ { - d \left( 1 + \frac { ( d - 1 ) ( \tau - k - \frac { 1 } { 2 } ) } { 2 \tau ( k + 1 ) - 2 k - 1 } \right) } } { \underline { { h } } ^ { - d } } , \quad k \leq \frac { d - 1 } { 2 } , \right.\tag{4.14}
$$

then

$$
\| f _ { n } \| _ { \ell ^ { 2 } ( X ) } \approx \| f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } = \| f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } , \qquad f _ { n } \in \chi L _ { n } ^ { k } .\tag{4.15}
$$

The constants depend on $d , k , \tau , \Omega , U , \Omega ^ { \prime } , \chi .$ , and the quasi-uniformity constant of X.

Under antipodal quasi-uniformity and $\textstyle k > { \frac { d - 1 } { 2 } }$ , the order m $\gtrsim$ n is optimal for an MZ inequality: the lower bound requires the evaluation map into $\mathbb { R } ^ { m }$ to be injective, hence $m \geq n$ . For $\begin{array} { r } { k \leq \frac { d - 1 } { 2 } } \end{array}$ (4.14) is suficient; we do not establish the sharpness of the extra oversampling exponent.

The preceding Bernstein and MZ inequalities concern parameters in $\mathcal { A } _ { \chi }$ . For discrete leastsquares recovery, we use parameters covering the full sphere and add a quadratic coeficient penalty. To be precise, we denote the discrete least square solution

$$
f _ { n , m } = \sum _ { j = 1 } ^ { n } \alpha _ { j } \varphi _ { \theta _ { j } ^ { * } } ,\tag{4.16}
$$

where α is the unique minimizer of

$$
\alpha = \arg \operatorname* { m i n } _ { a \in \mathbb { R } ^ { n } } \left\{ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \Big ( \chi ( x _ { i } ) f ( x _ { i } ) - \sum _ { j = 1 } ^ { n } a _ { j } \varphi _ { \theta _ { j } ^ { * } } ( x _ { i } ) \Big ) ^ { 2 } + h ^ { 2 k + 1 } \| a \| _ { 2 } ^ { 2 } \right\} .
$$

The following result separates the roles of parameter coverage, antipodal separation, and sample density.

Theorem 4.7 (Regularized discrete least squares). Let $d , n , k , r , s$ be as in Theorem $4 . 2 ,$ , and $\{ \theta _ { j } ^ { * } \} _ { j = 1 } ^ { n } \subset \mathbb { S } ^ { d }$ . There exist $C _ { 5 } > 0$ such that for any

$$
m \geq C _ { 5 } \left( h ^ { \frac { 2 k + 1 } { d + 2 k + 1 } } \underline { { { h } } } ^ { \frac { d } { d + 2 k + 1 } } \right) ^ { - d }
$$

and $X = \{ x _ { i } \} _ { i = 1 } ^ { m } \subset \Omega$ being quasi-uniform, the unique minimizer $f _ { n , m }$ of (4.16) satisfies

$$
\| f - f _ { n , m } \| _ { \mathcal { H } ^ { s } ( \Omega ^ { \prime } ) } \lesssim h ^ { r - s } \Big ( \frac { h } { \underline { { h } } } \Big ) ^ { \frac { d s } { d + 2 k + 1 } } \left\{ \| f \| _ { \mathcal { W } ^ { r , p } ( \Omega ) } , \quad \frac { d } { p } < r \leq \frac { d } { 2 } , \quad 2 < p < \infty , \right.\tag{4.17}
$$

The constants depend only on $d , k , r , s , p , \Omega , \Omega ^ { \prime } , \chi ,$ , and the quasi-uniformity constant of X, independently $o f n , m$ , and the parameter configuration.

Remark 4.3 (Parameter geometry and computation). The penalty strength depends on parameter coverage rather than on the target regularity or sample locations. Writing $A _ { i j } : = \varphi _ { \theta _ { i } ^ { * } } ( x _ { i } )$ and $y _ { i } : = \chi ( x _ { i } ) f ( x _ { i } )$ , the coeficients satisfy

$$
\left( { \frac { 1 } { m } } A ^ { \top } A + h ^ { 2 k + 1 } I _ { n } \right) \alpha = { \frac { 1 } { m } } A ^ { \top } y .\tag{4.18}
$$

This system is positive definite even $i f$ the localized neurons are linearly dependent. On $\Omega ^ { \prime }$ , where $\chi = 1$ , the approximant is the shallow network $\textstyle \sum _ { j = 1 } ^ { n } \alpha _ { j } \sigma _ { k } ( w _ { j } ^ { * } \cdot x + b _ { j } ^ { * } )$ ).

Finally, we ask when approximation rates imply target regularity under coeficient or separation conditions.

Theorem 4.8 (Inverse approximation). Let $\begin{array} { r } { d \in \mathbb { N } , \ k \in \mathbb { N } _ { 0 } , 0 < \alpha < \frac { d + 2 k + 1 } { 2 d } } \end{array}$ , and let $\{ \Theta _ { n } \} _ { n = 1 } ^ { \infty }$ be a sequence of antipodally quasi-uniform sets on $\mathcal { A } _ { \chi }$

1. Let $f \in \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } )$ . If there exists $M > 0$ , independent of n, such that

$$
\operatorname* { i n f } _ { f _ { n } \in \chi L _ { n , M } ^ { k , \alpha } ( \Theta _ { n } ) } \| f - f _ { n } \| _ { \mathcal L ^ { 2 } ( \mathbb R ^ { d } ) } \lesssim n ^ { - \frac { d + 2 k + 1 } { 2 d } + \alpha } ,\tag{4.19}
$$

then

$$
f \in B _ { 2 , \infty } ^ { \frac { d + 2 k + 1 } { 2 } - \alpha d } ( \mathbb { R } ^ { d } ) .
$$

2. For every $M > 0$ , there exists

$$
f \in \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) \setminus \mathcal { B } _ { 2 , \infty } ^ { \frac { d + 2 k + 1 } { 2 } - \alpha d } ( \mathbb { R } ^ { d } )
$$

such that

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname* { i n f } _ { f _ { n } \in \chi L _ { n , M } ^ { k , \alpha } ( \Theta _ { n } ) } \| f - f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \Omega ^ { \prime } ) } = 0 .\tag{4.20}
$$

3. Suppose further that $\Theta _ { n } \subset \Theta _ { n + 1 }$ and

$$
\underline { { h } } _ { n } : = \operatorname* { m i n } _ { i \neq j } \rho _ { \pm } ( \theta _ { i , n } ^ { * } , \theta _ { j , n } ^ { * } ) \gtrsim n ^ { - \frac { 1 } { d } } , \qquad n \geq 2 .
$$

If α $\begin{array} { r } { \mathrm { ~ : ~ } > \frac { 1 } { 2 } } \end{array}$ , then every $f \in \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } )$ satisfying

$$
\operatorname* { i n f } _ { f _ { n } \in \chi L _ { n } ^ { k } ( \Theta _ { n } ) } \| f - f _ { n } \| _ { \mathcal L ^ { 2 } ( \mathbb R ^ { d } ) } \lesssim n ^ { - \frac { d + 2 k + 1 } { 2 d } + \alpha }\tag{4.21}
$$

belongs to $B _ { 2 , \infty } ^ { \frac { d + 2 k + 1 } { 2 } - \alpha d } ( \mathbb { R } ^ { d } ) . ~ I f ^ { \mathrm { ~ 0 ~ < ~ } \alpha \mathrm { ~ \leq ~ } \frac { 1 } { 2 } }$ , there exists $f \in \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } )$ satisfying (4.21) but not belonging to this Besov space.

4. Let $f \in \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } )$ , and suppose there exists another parameter sequence $\widetilde { \Theta } _ { n } = \{ \widetilde { \theta } _ { j , n } ^ { * } \} _ { j = 1 } ^ { n } \subset { \mathcal { A } _ { \chi } }$ such that

$$
\# ( \Theta _ { n } \cup \widetilde { \Theta } _ { n } ) = 2 n , \qquad \operatorname* { m i n } _ { \theta , \theta ^ { \prime } \in \Theta _ { n } \cup \widetilde { \Theta } _ { n } } \rho _ { \pm } ( \theta , \theta ^ { \prime } ) \stackrel { } { \sim } n ^ { - \frac { 1 } { d } } ,
$$

and

$$
\operatorname* { i n f } _ { f _ { n } \in \chi L _ { n } ^ { k } ( \Theta _ { n } ) } \lVert f - f _ { n } \rVert _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } + \operatorname* { i n f } _ { \tilde { f } _ { n } \in \chi L _ { n } ^ { k } ( \widetilde { \Theta } _ { n } ) } \lVert f - \widetilde { f } _ { n } \rVert _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \lesssim n ^ { - \frac { d + 2 k + 1 } { 2 d } + \alpha } .\tag{4.22}
$$

Then $f \in B _ { 2 , \infty } ^ { \frac { d + 2 k + 1 } { 2 } - \alpha d } ( \mathbb { R } ^ { d } )$

Remark 4.4 (Coeficient and separation conditions). Item 1 requires the coeficient bound

$$
\left( n \sum _ { j = 1 } ^ { n } a _ { j } ^ { 2 } \right) ^ { \frac { 1 } { 2 } } \leq M n ^ { \alpha }\tag{4.23}
$$

with M independent of $n .$ Its error is measured on $\mathbb { R } ^ { d } .$ , while item $\mathcal { Q }$ concerns only qualitative convergence on Ω<sup>′</sup>. Item $\it 4$ requires no coeficient bound, but its union must contain 2n distinct parameters with the stated antipodal separation, including cross-family pairs; separate quasi-uniformity is insuficient.

Remark 4.5 (Sobolev regularity at the endpoint). At $\alpha = 0 , / 6 \gamma ,$ Theorem $\it { 2 . 4 } ]$ gives the stronger conclusion

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname* { i n f } _ { g _ { n } \in L _ { n , M } ^ { k , 0 } } \| f - g _ { n } \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } = 0 \quad \Longrightarrow \quad f \in \mathcal { H } ^ { \frac { d + 2 k + 1 } { 2 } } ( \Omega ) ,\tag{4.24}
$$

for M independent of n. This result assumes parameters quasi-uniform on the entire $\mathbb { S } ^ { d }$ and qualitative convergence on $\Omega ;$ the present theorem concerns parameters in $\mathcal { A } _ { \chi }$ and whole-space approximation by localized networks.

## 5 Localization of the Littlewood–Paley Sobolev blocks

This section proves the two parts of Theorem 4.1. We retain their separate statements below to organize the proofs.

In Sections 5 and $6 ,$ we assume that $\Omega \subset \mathbb { B } ^ { d }$ . The case of a general bounded Ω follows by rescaling the spatial variable. We switch from the spherical parametrization to the unit-normal

parametrization. For $\theta = { \binom { w } { b } } \in \mathcal { A } _ { \chi }$ , the active condition implies $| w | \geq c _ { 3 } > 0 ;$ ; hence the normalization map

$$
{ \binom { w } { b } } \longmapsto \left( { \frac { w } { | w | } } , { \frac { b } { | w | } } \right)
$$

is well defined on $\mathcal { A } _ { \chi }$ . By positive homogeneity,

$$
\sigma _ { k } ( w ^ { \top } x + b ) = | w | ^ { k } \sigma _ { k } \left( \frac { w } { | w | } ^ { \top } x + \frac { b } { | w | } \right) ,
$$

and the normalized parameters form a compact subset $\boldsymbol { B } _ { \boldsymbol { \chi } } \subset \mathbb { S } ^ { d - 1 } \times [ - 1 , 1 ]$ . Since the spherical distance on $\mathbb { S } ^ { d }$ is equivalent to the Euclidean chordal distance and the above normalization is bi-Lipschitz on the active set, we use the normalized parametrization throughout this section. For $\theta = { \binom { w } { b } } , \theta ^ { \prime } = ( w ^ { \prime } , b ^ { \prime } ) \in \mathcal { B } _ { \chi }$ , define

$$
\varrho ( \theta , \theta ^ { \prime } ) : = \left( | w - w ^ { \prime } | ^ { 2 } + | b - b ^ { \prime } | ^ { 2 } \right) ^ { 1 / 2 }
$$

and

$$
\underline { { \varrho } } _ { \pm } ( \theta , \theta ^ { \prime } ) : = \operatorname* { m i n } \{ \varrho ( \theta , \theta ^ { \prime } ) , \varrho ( \theta , - \theta ^ { \prime } ) \} .
$$

In the Fourier proofs below, separation is measured by $\varrho$ and $\varrho \pm$ , with $| w | = 1$ for $\theta = { \binom { w } { b } } \in B _ { \chi }$ Positive homogeneity and the bi-Lipschitz normalization transfer the estimates to $\mathcal { A } _ { \chi }$

Theorem 5.1 (Localization property). For every $0 \leq s < k + \textstyle { \frac { 1 } { 2 } } , \gamma > d _ { \cdot }$ , and $\theta , \theta ^ { \prime } \in \mathcal { A } _ { \chi }$ with $\rho _ { \pm } ( \theta , \theta ^ { \prime } ) > 0$

$$
\Bigl | \langle \Delta _ { q , s } \varphi _ { \theta } , \Delta _ { q , s } \varphi _ { \theta ^ { \prime } } \rangle _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \Bigr | \lesssim _ { \gamma } \frac { 2 ^ { - q ( 2 ( k - s ) + 1 ) } } { ( 2 ^ { q } \rho _ { \pm } ( \theta , \theta ^ { \prime } ) ) ^ { \gamma } } , \qquad q \geq 1 .\tag{5.1}
$$

## 5.1 Active-parameter normalization and the high-frequency representation

We first express the inner product in (5.1) using Parseval’s identity:

$$
\begin{array} { r l } & { \langle \Delta _ { q , s } \varphi _ { \theta } , \Delta _ { q , s } \varphi _ { \theta ^ { \prime } } \rangle _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } = \displaystyle \frac { 1 } { ( 2 \pi ) ^ { d } } \int _ { \mathbb { R } ^ { d } } \zeta _ { q , s } ( \xi ) ^ { 2 } \widehat { \varphi _ { \theta } } ( \xi ) \overline { { \widehat { \varphi _ { \theta ^ { \prime } } } ( \xi ) } } d \xi } \\ & { \qquad = \displaystyle \frac { 1 } { ( 2 \pi ) ^ { d } } \int _ { \mathbb { R } ^ { d } } \zeta \Big ( \frac { | \xi | } { 2 ^ { q } } \Big ) ^ { 2 } ( 1 + | \xi | ^ { 2 } ) ^ { s } \widehat { \varphi _ { \theta } } ( \xi ) \overline { { \widehat { \varphi _ { \theta ^ { \prime } } } ( \xi ) } } d \xi . } \end{array}\tag{5.2}
$$

We derive a Fourier representation of $\varphi _ { \theta }$ from the distributional Fourier transform of $\sigma _ { k }$ . The activation has polynomial growth and is not in ${ \mathcal { L } } ^ { 2 } ( \mathbb { R } )$ , so its transform is a tempered distribution. For $k = 0$ , the inversion identity below is understood distributionally and almost everywhere; symmetric inversion assigns the midpoint value at the jump. Substituting $t = w ^ { \top } x + b$ into the Fourier inversion formula

$$
\sigma _ { k } ( t ) = \frac { 1 } { 2 \pi } \int _ { \mathbb { R } } \widehat { \sigma _ { k } } ( \tau ) e ^ { i \tau t } d \tau
$$

yields

$$
\sigma _ { k } ( w ^ { \top } x + b ) = \frac { 1 } { 2 \pi } \int _ { \mathbb R } \widehat { \sigma _ { k } } ( \tau ) e ^ { i \tau ( w ^ { \top } x + b ) } d \tau = \frac { 1 } { 2 \pi } \int _ { \mathbb R } \widehat { \sigma _ { k } } ( \tau ) e ^ { i \tau b } e ^ { i ( \tau w ) ^ { \top } x } d \tau .\tag{5.3}
$$

Using the standard modulation formula for the Fourier transform,

$$
\mathcal { F } [ \chi ( \circ ) e ^ { i ( \tau w ) ^ { \top } \circ } ] ( \xi ) = \widehat { \chi } ( \xi - \tau w ) ,
$$

we have

$$
\begin{array} { l } { \displaystyle \varphi _ { \theta } ( x ) = \chi ( x ) \sigma _ { k } ( w ^ { \top } x + b ) = \frac { 1 } { 2 \pi } \int _ { \mathbb { R } } \widehat { \sigma _ { k } } ( \tau ) e ^ { i \tau b } e ^ { i ( \tau w ) ^ { \top } x } \chi ( x ) d \tau } \\ { \displaystyle \quad = \frac { 1 } { ( 2 \pi ) ^ { d + 1 } } \int _ { \mathbb { R } } \widehat { \sigma _ { k } } ( \tau ) e ^ { i \tau b } \int _ { \mathbb { R } ^ { d } } \widehat { \chi } ( \xi - \tau w ) e ^ { i \xi \cdot x } d \xi d \tau . } \end{array}
$$

That is,

$$
\widehat { \varphi \theta } ( \xi ) = \frac { 1 } { 2 \pi } \int _ { \mathbb { R } } \widehat { \sigma _ { k } } ( \tau ) e ^ { i \tau b } \widehat { \chi } ( \xi - \tau w ) d \tau .\tag{5.4}
$$

We split the Fourier transform into three parts. The first is estimated through pairwise interactions; the other two decay rapidly. Let $\vartheta , \varsigma \in { \mathcal { C } } ^ { \infty } ( \mathbb { R } )$ be smooth symmetric functions satisfying

$$
\begin{array} { r l r } & { \displaystyle \vartheta ( \tau ) = 0 \quad \mathrm { f o r } | \tau | \leq 1 , \qquad \vartheta ( \tau ) = 1 } & { \mathrm { f o r } | \tau | \geq 2 , } \\ & { \displaystyle \varsigma ( \tau ) = 0 \quad \mathrm { f o r } | \tau | \leq \frac { 1 } { 8 } \mathrm { ~ a n d } | \tau | \geq 8 , \qquad \varsigma ( \tau ) = 1 } & { \mathrm { f o r } \frac { 1 } { 4 } \leq | \tau | \leq 4 } \end{array}
$$

We then write

$$
\widehat { \varphi \theta } ( \xi ) = \frac { 1 } { 2 \pi } \big ( I _ { 1 } ( \theta , q ; \xi ) + I _ { 2 } ( \theta , q ; \xi ) + I _ { 3 } ( \theta ; \xi ) \big ) ,\tag{5.5}
$$

where

$$
\begin{array} { r l } & { I _ { 1 } ( \theta , q ; \xi ) = \displaystyle \int _ { \mathbb { R } } \varsigma \big ( \frac { \tau } { 2 ^ { q } } \big ) \vartheta ( \tau ) \widehat { \sigma _ { k } } ( \tau ) e ^ { i \tau b } \widehat { \chi } ( \xi - \tau w ) d \tau , } \\ & { I _ { 2 } ( \theta , q ; \xi ) = \displaystyle \int _ { \mathbb { R } } \Big ( 1 - \varsigma \big ( \frac { \tau } { 2 ^ { q } } \big ) \Big ) \vartheta ( \tau ) \widehat { \sigma _ { k } } ( \tau ) e ^ { i \tau b } \widehat { \chi } ( \xi - \tau w ) d \tau , } \\ & { I _ { 3 } ( \theta ; \xi ) = \displaystyle \int _ { \mathbb { R } } ( 1 - \vartheta ( \tau ) ) \widehat { \sigma _ { k } } ( \tau ) e ^ { i \tau b } \widehat { \chi } ( \xi - \tau w ) d \tau } \end{array}\tag{5.6}
$$

In the sense of distributions,

$$
\partial _ { t } ^ { k + 1 } \sigma _ { k } = k ! \delta _ { 0 } .
$$

Consequently, away from the origin,

$$
{ \widehat { \sigma _ { k } } } ( \tau ) = k ! ( i \tau ) ^ { - k - 1 } ,\tag{5.7}
$$

Thus $\widehat { \sigma _ { k } }$ is homogeneous of degree $- ( k + 1 )$ away from the origin, and $\widehat { \vartheta \sigma _ { k } }$ is an ordinary smooth function. For every integer $\ell \geq 0$

$$
\Bigl | \Bigl ( \frac { d } { d \tau } \Bigr ) ^ { \ell } ( \vartheta \widehat { \sigma _ { k } } ) ( \tau ) \Bigr | \lesssim _ { \ell } | \tau | ^ { - k - 1 - \ell } , \qquad | \tau | \geq 1 .\tag{5.8}
$$

Lemma 5.1. For every $\gamma > 0$

$$
\begin{array} { r } { | I _ { 3 } ( \theta ; \xi ) | \lesssim _ { \gamma } ( 1 + | \xi | ) ^ { - \gamma } , } \end{array}
$$

uniformly for $\theta \in B _ { \chi }$

Proof. Applying the Fourier modulation formula again, we obtain

$$
\mathcal { F } ^ { - 1 } \left[ I _ { 3 } ( \boldsymbol { \theta } ; \circ ) \right] ( \boldsymbol { x } ) = 2 \pi \boldsymbol { \chi } ( \boldsymbol { x } ) \mathcal { F } ^ { - 1 } \left[ ( 1 - \boldsymbol { \vartheta } ) \widehat { \sigma _ { k } } \right] ( \boldsymbol { w } ^ { \top } \boldsymbol { x } + \boldsymbol { b } ) .
$$

The compactly supported distribution $( 1 - \vartheta ) \widehat { \sigma _ { k } }$ has finite order, so its inverse transform is smooth. Hence $\mathcal { F } ^ { - 1 } [ I _ { 3 } ( \theta ; \circ ) ]$ is smooth and supported in Ω, with all seminorms $\big | \mathcal { F } ^ { - 1 } [ I _ { 3 } ( \theta ; \circ ) ] \big | _ { \mathcal { C } ^ { \gamma } ( \mathbb { R } ^ { d } ) }$ uniformly bounded for $\theta \in B _ { \chi }$ . Its Fourier transform therefore decays rapidly:

$$
\begin{array} { r } { | I _ { 3 } ( \theta ; \xi ) | \lesssim _ { \gamma } ( 1 + | \xi | ) ^ { - \gamma } , \quad \xi \in \mathbb R ^ { d } . } \end{array}
$$

Lemma 5.2. For every $\gamma > 0$

$$
\begin{array} { r } { \lVert \zeta _ { q , s } ( \circ ) I _ { 2 } ( \theta , q ; \circ ) \rVert _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \lesssim _ { \gamma } 2 ^ { - q \gamma } , \quad q \geq 1 . } \end{array}\tag{5.9}
$$

uniformly for $\theta \in B _ { \chi }$ . Thus we may insert smooth cutofs in $\tau$ to restrict $| \tau | \stackrel { } { \sim } 2 ^ { q }$ , with a remainder that decays uniformly faster than any power of $2 ^ { - q }$

Proof. By (3.7), $\zeta _ { q , s } ( \circ ) I _ { 2 } ( \theta , q ; \circ )$ is supported in $\{ \xi \in \mathbb { R } ^ { d } : 2 ^ { q - 1 } \leq | \xi | \leq 2 ^ { q + 1 } \}$

$\mathrm { I f } \ | \tau | \le 2 ^ { q - 2 }$ , then, since $| w | = 1$ , we have

$$
| \xi - \tau w | \geq | \xi | - | \tau | \geq 2 ^ { q - 2 } .
$$

Since $\chi \in \mathcal { C } _ { c } ^ { \infty } ( \mathbb { R } ^ { d } ) , \widehat { \chi }$ is a Schwartz function:

$$
\begin{array} { r } { | \widehat \chi ( \xi - \tau w ) | \overset { < } \sim ( 1 + | \xi - \tau w | ) ^ { - \gamma - 1 } \overset { < } \sim 2 ^ { - q ( \gamma + 1 ) } . } \end{array}\tag{5.10}
$$

As $\vartheta = 0$ for $| \tau | \leq 1$ , the bound (5.8) gives

$$
\int _ { - 2 ^ { q - 2 } } ^ { 2 ^ { q - 2 } } \left| \vartheta ( \tau ) \widehat { \sigma _ { k } } ( \tau ) e ^ { i \tau b } \widehat { \chi } ( \xi - \tau w ) \right| d \tau \lesssim _ { \gamma } \int _ { - 2 ^ { q - 2 } } ^ { 2 ^ { q - 2 } } 2 ^ { - q ( \gamma + 1 ) } d \tau \lesssim 2 ^ { - q \gamma } .
$$

If $| \tau | \geq 2 ^ { q + 2 }$ , then

$$
\begin{array} { r } { | \xi - \tau w | \geq | \tau | - | \xi | \geq | \tau | - 2 ^ { q + 1 } \geq 2 ^ { q + 1 } . } \end{array}
$$

Using (5.8) and (5.10), we obtain

$$
\int _ { | \tau | \geq 2 ^ { q + 2 } } \left| \vartheta ( \tau ) \widehat { \sigma _ { k } } ( \tau ) e ^ { i \tau b } \widehat { \chi } ( \xi - \tau w ) \right| d \tau \lesssim \int _ { | \tau | \geq 2 ^ { q + 2 } } | \tau | ^ { - k - 1 } ( 1 + | \tau | ) ^ { - \gamma } d \tau \lesssim 2 ^ { - q ( k + \gamma ) } .
$$

Thus $I _ { 2 } ( \theta , q ; \circ )$ decays pointwise to any prescribed order on $| \xi | \in [ 2 ^ { q - 1 } , 2 ^ { q + 1 } ]$ . Increasing that order by $\lceil { \frac { d } { 2 } } + \dot { s } \rceil \ \mathrm { g i v e s }$ (5.9). □

## 5.2 Preliminary estimates for $I _ { 1 }$

Recall that the normalization above gives

$$
\begin{array} { r } { | w | = | w ^ { \prime } | = 1 . } \end{array}
$$

We aim to show

$$
\left| \int _ { \mathbb { R } ^ { d } } \zeta _ { q , s } ( \xi ) ^ { 2 } I _ { 1 } ( \theta , q ; \xi ) \overline { { I _ { 1 } ( \theta ^ { \prime } , q ; \xi ) } } d \xi \right| \lesssim \frac { 2 ^ { - q ( 2 ( k - s ) + 1 ) } } { ( 2 ^ { q } \varrho _ { \pm } ( \theta , \theta ^ { \prime } ) ) ^ { \gamma } } .\tag{5.11}
$$

It sufices to prove the estimate for $q \geq 4$ . On the support of the cutofs defining $I _ { 1 } , \vartheta ( \tau ) = \vartheta ( \tau ^ { \prime } ) =$ 1, so

$$
\begin{array} { r l } & { \quad \displaystyle \int _ { \mathbb { R } ^ { d } } \zeta _ { q , s } ( \xi ) ^ { 2 } I _ { 1 } ( \theta , q ; \xi ) \overline { { I _ { 1 } ( \theta ^ { \prime } , q ; \xi ) } } d \xi } \\ & { = \displaystyle \int _ { \mathbb { R } ^ { d } } \zeta _ { q , s } ( \xi ) ^ { 2 } \iint _ { \mathbb { R } ^ { 2 } } \varsigma \big ( \frac { \tau } { 2 q } \big ) \varsigma \big ( \frac { \tau ^ { \prime } } { 2 q } \big ) \widehat { \sigma _ { k } } ( \tau ) \overline { { \widehat { \sigma _ { k } } ( \tau ^ { \prime } ) } } e ^ { i ( \tau b - \tau ^ { \prime } b ^ { \prime } ) } \widehat { \chi } ( \xi - \tau w ) \overline { { \widehat { \chi } ( \xi - \tau ^ { \prime } w ^ { \prime } ) } } d \tau d \tau ^ { \prime } d \xi . } \end{array}
$$

Let $\varsigma _ { + } ( t ) = { \left\{ \begin{array} { l l } { \varsigma ( t ) , } & { \quad t \geq 0 , } \\ { 0 , } & { \quad t < 0 , } \end{array} \right. }$ and define ${ \cal J } _ { + } ( \theta , \theta ^ { \prime } , q )$ and ${ \cal J } _ { - } ( \theta , \theta ^ { \prime } , q )$ by

$$
\begin{array} { l } { { \displaystyle { J _ { + } ( \theta , \theta ^ { \prime } , q ) = \iint _ { \mathbb { R } ^ { 2 } } \varsigma _ { + } \big ( \frac { \tau } { 2 q } \big ) \varsigma _ { + } \big ( \frac { \tau ^ { \prime } } { 2 q } \big ) e ^ { i ( \tau b - \tau ^ { \prime } b ^ { \prime } ) } \int _ { \mathbb { R } ^ { d } } \zeta _ { q , s } ( \xi ) ^ { 2 } \widehat { \sigma _ { k } } ( \tau ) \overline { { \widehat { \sigma _ { k } } ( \tau ^ { \prime } ) } } \widehat { \chi } ( \xi - \tau w ) \overline { { \widehat { \chi } ( \xi - \tau ^ { \prime } w ^ { \prime } ) } } d \xi d \tau d ^ { \prime } , } } } \\ { { \displaystyle { J _ { - } ( \theta , \theta ^ { \prime } , q ) = \iint _ { \mathbb { R } ^ { 2 } } \varsigma _ { + } \big ( \frac { \tau } { 2 q } \big ) \varsigma _ { + } \big ( - \frac { \tau ^ { \prime } } { 2 ^ { q } } \big ) e ^ { i ( \tau b - \tau ^ { \prime } b ^ { \prime } ) } \int _ { \mathbb { R } ^ { d } } \zeta _ { q , s } ( \xi ) ^ { 2 } \widehat { \sigma _ { k } } ( \tau ) \overline { { \widehat { \sigma _ { k } } ( \tau ^ { \prime } ) } } \widehat { \chi } ( \xi - \tau w ) \overline { { \widehat { \chi } ( \xi - \tau ^ { \prime } w ^ { \prime } ) } } d \xi d \tau d \tau ^ { \prime } . } } } \end{array}\tag{5.12}
$$

Using the symmetry properties of $\vartheta , \widehat { \sigma _ { k } } , \widehat { \chi } .$ and the integration domain, we obtain

$$
\begin{array} { l } { \displaystyle \iint _ { \mathbb R ^ { 2 } } \varsigma _ { + } \big ( - \frac { \tau } { 2 ^ { q } } \big ) c _ { + } \big ( - \frac { \tau ^ { \prime } } { 2 ^ { q } } \big ) \int _ { \mathbb R ^ { d } } \zeta _ { q , s } ( \xi ) ^ { 2 } \widehat { \sigma _ { k } } ( \tau ) \overline { { \widehat { \sigma _ { k } } ( \tau ^ { \prime } ) } } e ^ { i ( \tau b - \tau ^ { \prime } b ^ { \prime } ) } \widehat { \chi } ( \xi - \tau w ) \overline { { \widehat { \chi } ( \xi - \tau ^ { \prime } w ^ { \prime } ) } } d \xi d \tau d \tau ^ { \prime } = \overline { { J _ { + } ( \theta , \theta ^ { \prime } , q ) } } } \\ { \displaystyle \iint _ { \mathbb R ^ { 2 } } \varsigma _ { + } \big ( - \frac { \tau } { 2 ^ { q } } \big ) \varsigma _ { + } \big ( \frac { \tau ^ { \prime } } { 2 ^ { q } } \big ) \int _ { \mathbb R ^ { d } } \zeta _ { q , s } ( \xi ) ^ { 2 } \widehat { \sigma _ { k } } ( \tau ) \overline { { \widehat { \sigma _ { k } } ( \tau ^ { \prime } ) } } e ^ { i ( \tau b - \tau ^ { \prime } b ^ { \prime } ) } \widehat { \chi } ( \xi - \tau w ) \overline { { \widehat { \chi } ( \xi - \tau ^ { \prime } w ^ { \prime } ) } } d \xi d \tau d \tau ^ { \prime } = \overline { { J _ { - } ( \theta ^ { \prime } , \theta , q ) } } . } \end{array}
$$

The substitution $\tau ^ { \prime } \mapsto - \tau ^ { \prime }$ and the same symmetries give

$$
J _ { - } ( \theta , \theta ^ { \prime } , q ) = ( - 1 ) ^ { k + 1 } \iint _ { \mathbb { R } ^ { 2 } } \mathsf { S } + \left( \frac { \tau } { 2 ^ { q } } \right) \mathsf { C } + \left( \frac { \tau ^ { \prime } } { 2 ^ { q } } \right) e ^ { i ( \tau b + \tau ^ { \prime } b ^ { \prime } ) } \int _ { \mathbb { R } ^ { d } } \zeta _ { q , s } ( \xi ) ^ { 2 } \widehat { \sigma _ { k } } ( \tau ) \overline { { \widehat { \sigma _ { k } } ( \tau ^ { \prime } ) } } \widehat { \chi } ( \xi - \tau w ) \overline { { \widehat { \chi } ( \xi + \tau ^ { \prime } w ^ { \prime } ) } } d \xi d \tau d \tau ^ { \prime } .\tag{5.13}
$$

Thus both $J _ { + }$ and $J _ { - }$ are reduced to interactions between two positive normal frequencies. The two cases difer only in whether the second afine parameter is $( w ^ { \prime } , b ^ { \prime } ) \mathrm { o r } ( - w ^ { \prime } , - b ^ { \prime } )$

To prove (5.11), it therefore sufices to show

$$
| J _ { + } ( \theta , \theta ^ { \prime } , q ) | \lesssim \frac { 2 ^ { - q ( 2 ( k - s ) + 1 ) } } { ( 2 ^ { q } \varrho _ { \pm } ( \theta , \theta ^ { \prime } ) ) ^ { \gamma } } , \quad | J _ { - } ( \theta , \theta ^ { \prime } , q ) | \lesssim \frac { 2 ^ { - q ( 2 ( k - s ) + 1 ) } } { ( 2 ^ { q } \varrho _ { \pm } ( \theta , \theta ^ { \prime } ) ) ^ { \gamma } } .\tag{5.14}
$$

We estimate these terms using direction separation and bias separation. Since

$$
\underline { { \varrho } } _ { \pm } ( \theta , \theta ^ { \prime } ) = \operatorname* { m i n } \{ \varrho ( \theta , \theta ^ { \prime } ) , \varrho ( \theta , - \theta ^ { \prime } ) \} \asymp \operatorname* { m i n } \left\{ \vert w - w ^ { \prime } \vert + \vert b - b ^ { \prime } \vert , \vert w + w ^ { \prime } \vert + \vert b + b ^ { \prime } \vert \right\} ,
$$

the separation of the two afine parameters can be detected either by the directional component or by the bias component.

Case 1: When $| w - w ^ { \prime } | \gtrsim \varrho ( \theta , \theta ^ { \prime } ) \geq \varrho _ { \pm } ( \theta , \theta ^ { \prime } )$ , the decay of ${ \cal J } _ { + } ( \theta , \theta ^ { \prime } , q )$ is obtained from the overlap of the two Fourier packets

$$
\widehat { \chi } ( \xi - \tau w ) , \qquad \widehat { \chi } ( \xi - \tau ^ { \prime } w ^ { \prime } ) .
$$

Indeed, under dyadic localization, $| \tau | \stackrel { } { \sim } | \tau ^ { \prime } | \stackrel { } { \sim } 2 ^ { q }$ , so the distance between the centers of the two packets is given by

$$
| \tau w - \tau ^ { \prime } w ^ { \prime } | \quad \mathrm { o r } \quad | \tau w + \tau ^ { \prime } w ^ { \prime } | ,
$$

depending on the corresponding sign. Since $\widehat \chi$ is a Schwartz function, the overlap of the two packets decays rapidly as this distance increases, which provides the decay factor $( 1 + 2 ^ { q } \varrho _ { \pm } ( \theta , \theta ^ { \prime } ) ) ^ { - \gamma }$

Case 2: When $| b - b ^ { \prime } | \gtrsim \varrho ( \theta , \theta ^ { \prime } ) \geq \varrho _ { \pm } ( \theta , \theta ^ { \prime } )$ , the decay must come from the oscillation of the phase factor

$$
e ^ { i ( \tau b - \tau ^ { \prime } b ^ { \prime } ) } .
$$

Such oscillatory cancellation is standard in the literature $( \mathrm { s e e } , \mathrm { e . g . } , [ 1 0 7 , \mathrm { C h a p t e r \ 8 } ] )$

To separate the oscillatory variable from the frequency mismatch, we introduce the change of variables

$$
u = \tau + \tau ^ { \prime } , \qquad v = \tau - \tau ^ { \prime } .
$$

Then

$$
\tau = \frac { u + v } { 2 } , \qquad \tau ^ { \prime } = \frac { u - v } { 2 } , \qquad d \tau d \tau ^ { \prime } = \frac 1 2 d u d v .
$$

For the two coherent sign configurations, the phase can be written as

$$
\tau b - \tau ^ { \prime } b ^ { \prime } = \frac { u } { 2 } ( b - b ^ { \prime } ) + \frac { v } { 2 } ( b + b ^ { \prime } ) , \quad \tau b + \tau ^ { \prime } b ^ { \prime } = \frac { u } { 2 } ( b + b ^ { \prime } ) + \frac { v } { 2 } ( b - b ^ { \prime } ) .
$$

A similar argument applies to ${ \cal J } _ { - } ( \theta , \theta ^ { \prime } , q )$ . Thus, the two coherent contributions (5.12) and (5.13) can be written as

$$
J _ { \pm } ( \theta , \theta ^ { \prime } , q ) = \frac { 1 } { 2 } \int _ { \mathbb { R } } \int _ { \mathbb { R } } e ^ { \frac { i u } { 2 } \left( b \mp b ^ { \prime } \right) } e ^ { \frac { i v } { 2 } \left( b \pm b ^ { \prime } \right) } A _ { q , \pm } ( u , v ) d u d v ,\tag{5.15}
$$

where

$$
\begin{array} { r l } & { A _ { q , \pm } ( u , v ) = ( \pm 1 ) ^ { k + 1 } \varsigma _ { + } \left( \frac { u + v } { 2 ^ { q + 1 } } \right) \varsigma _ { + } \left( \frac { u - v } { 2 ^ { q + 1 } } \right) \vartheta \left( \displaystyle \frac { u + v } { 2 } \right) \widehat { \sigma _ { k } } \left( \displaystyle \frac { u + v } { 2 } \right) \vartheta \left( \displaystyle \frac { u - v } { 2 } \right) \widehat { \sigma _ { k } } \left( \displaystyle \frac { u - v } { 2 } \right) } \\ & { \qquad \times \displaystyle \int _ { \mathbb R ^ { d } } \zeta _ { q , s } ( \xi ) ^ { 2 } \widehat { \chi } \left( \xi - \frac { u + v } { 2 } w \right) \widehat { \chi } \left( \xi \mp \frac { u - v } { 2 } w ^ { \prime } \right) d \xi . } \end{array}\tag{5.16}
$$

Lemma 5.3. For all integers $\gamma _ { 1 } , \gamma _ { 2 } \geq 0$

$$
\iint _ { \mathbb R ^ { 2 } } | \partial _ { u } ^ { \gamma _ { 1 } } A _ { q , \pm } ( u , v ) | d u d v \lesssim _ { \gamma _ { 1 } , \gamma _ { 2 } } 2 ^ { - q ( 2 ( k - s ) + 1 ) } ( 2 ^ { - q } + | w \mp w ^ { \prime } | ) ^ { \gamma _ { 1 } } ( 1 + 2 ^ { q } | w \mp w ^ { \prime } | ) ^ { - \gamma _ { 2 } } .\tag{5.17}
$$

Proof. For either sign, Leibniz’s rule reduces the derivative estimates for $A _ { q , \pm }$ to the following factors:

$$
K _ { 1 , q , \pm } ( u , v ) = \int _ { \mathbb { R } ^ { d } } \zeta _ { q , s } ( \xi ) ^ { 2 } \widehat { \chi } \left( \xi - \frac { u + v } { 2 } w \right) \overline { { \widehat { \chi } \left( \xi \mp \frac { u - v } { 2 } w ^ { \prime } \right) } } d \xi ,
$$

$$
K _ { 2 , q } ( u , v ) = \vartheta \left( \frac { u + v } { 2 } \right) \widehat { \sigma _ { k } } \left( \frac { u + v } { 2 } \right) \vartheta \left( \frac { u - v } { 2 } \right) \widehat { \sigma _ { k } } \left( \frac { u - v } { 2 } \right) ,
$$

$$
K _ { 3 , q } ( u , v ) = \varsigma _ { + } \big ( { \frac { u + v } { 2 ^ { q + 1 } } } \big ) \varsigma _ { + } \big ( { \frac { u - v } { 2 ^ { q + 1 } } } \big ) .
$$

Making the change of variables $\begin{array} { r } { \eta = \xi - \frac { u + v } { 2 } w } \end{array}$ , we write

$$
K _ { 1 , q , \pm } ( u , v ) = \int _ { \mathbb { R } ^ { d } } \zeta _ { q , s } \left( \eta + \frac { u + v } 2 w \right) ^ { 2 } \widehat { \chi } ( \eta ) \overline { { \widehat { \chi } \left( \eta + \frac { u + v } 2 w \mp \frac { u - v } 2 w ^ { \prime } \right) } } d \eta ,
$$

and by Leibniz’s rule,

$$
\begin{array} { r l } & { | \partial _ { u } ^ { \beta } K _ { 1 , q , \pm } ( u , v ) | \lesssim \displaystyle \sum _ { \beta _ { 1 } + \beta _ { 2 } = \beta } \int _ { \mathbb { R } ^ { d } } \bigg | \partial _ { u } ^ { \beta _ { 1 } } \left[ \zeta _ { q , s } \left( \eta + \frac { u + v } { 2 } w \right) ^ { 2 } \right] \partial _ { u } ^ { \beta _ { 2 } } \left[ \widehat { \chi } ( \eta ) \widehat { \chi } \left( \eta + \frac { u + v } { 2 } w \mp \frac { u - v } { 2 } w ^ { \prime } \right) \right] \bigg | d \eta } \\ & { \lesssim \displaystyle \sum _ { \beta _ { 1 } + \beta _ { 2 } = \beta } \int _ { \mathbb { R } ^ { d } } 2 ^ { q ( 2 s - \beta _ { 1 } ) } \left| \widehat { \chi } ( \eta ) \overline { { \left[ \left( \frac { w \mp w ^ { \prime } } { 2 } \cdot \nabla \right) ^ { \beta _ { 2 } } \widehat { \chi } \right] \left( \eta + \frac { u + v } { 2 } w \mp \frac { u - v } { 2 } w ^ { \prime } \right) } } \right| d \eta . } \end{array}
$$

On the support of $K _ { 3 , q } ,$ the relations

$$
\tau = \frac { u + v } { 2 } , \qquad \tau ^ { \prime } = \frac { u - v } { 2 }
$$

give $\tau \lesssim \tau ^ { \prime } \lesssim 2 ^ { q } > 0$ , and hence

$$
\left| { \frac { u + v } { 2 } } w \mp { \frac { u - v } { 2 } } w ^ { \prime } \right| = | { \boldsymbol { \tau } } w \mp { \boldsymbol { \tau } } ^ { \prime } w ^ { \prime } | = \left( | { \boldsymbol { \tau } } - { \boldsymbol { \tau } } ^ { \prime } | ^ { 2 } + { \boldsymbol { \tau } } { \boldsymbol { \tau } } ^ { \prime } | w \mp w ^ { \prime } | ^ { 2 } \right) ^ { 1 / 2 } \gtrsim | v | + 2 ^ { q } | w \mp w ^ { \prime } | .
$$

Here $\nabla ^ { \beta _ { 2 } } \widehat { \chi }$ denotes the collection of derivatives of order $\beta _ { 2 }$ . Schwartz decay gives

$$
| \widetilde { \chi } ( \eta ) | \left| \nabla ^ { \beta _ { 2 } } \widehat { \chi } \left( \eta + \frac { u + v } { 2 } w \mp \frac { u - v } { 2 } w ^ { \prime } \right) \right| \lesssim _ { \beta _ { 2 } , \gamma _ { 2 } } ( 1 + | \eta | ) ^ { - d - 1 } ( 1 + | v | + 2 ^ { q } | w \mp w ^ { \prime } | ) ^ { - \gamma _ { 2 } - 2 } .
$$

Since $\begin{array} { r } { \int _ { \mathbb { R } ^ { d } } ( 1 + | \eta | ) ^ { - d - 1 } d \eta < \infty . } \end{array}$ , it follows that

$$
| \partial _ { u } ^ { \beta } K _ { 1 , q , \pm } ( u , v ) | \lesssim _ { \beta , \gamma _ { 2 } } 2 ^ { 2 q s } ( 2 ^ { - q } + | w \mp w ^ { \prime } | ) ^ { \beta } ( 1 + | v | + 2 ^ { q } | w \mp w ^ { \prime } | ) ^ { - \gamma _ { 2 } - 2 } .
$$

On the other hand, (5.8) gives, on the support of $A _ { q , \pm }$

$$
\begin{array} { r l } & { | \partial _ { u } ^ { \alpha } K _ { 2 , q } ( u , v ) | = \left| \partial _ { u } ^ { \alpha } \left[ \vartheta \left( \displaystyle \frac { u + v } { 2 } \right) \widehat { \sigma _ { k } } \left( \displaystyle \frac { u + v } { 2 } \right) \vartheta \left( \displaystyle \frac { u - v } { 2 } \right) \widehat { \sigma _ { k } } \left( \displaystyle \frac { u - v } { 2 } \right) \right] \right| } \\ & { \qquad \lesssim _ { \alpha } \displaystyle \sum _ { \alpha _ { 1 } + \alpha _ { 2 } = \alpha } | \partial _ { \tau } ^ { \alpha _ { 1 } } \left( \vartheta ( \tau ) \widehat { \sigma _ { k } } ( \tau ) \right) \partial _ { \tau ^ { \prime } } ^ { \alpha _ { 2 } } \left( \vartheta ( \tau ^ { \prime } ) \widehat { \sigma _ { k } } ( \tau ^ { \prime } ) \right) | \lesssim 2 ^ { - q ( 2 k + 2 + \alpha ) } . } \end{array}
$$

By Leibniz’s rule, on its support,

$$
\begin{array} { r l } & { | \partial _ { u } ^ { \gamma _ { 1 } } A _ { q , \pm } ( u , v ) | \lesssim _ { \gamma _ { 1 } } \displaystyle \sum _ { \alpha + \beta + \nu = \gamma _ { 1 } } | \partial _ { u } ^ { \beta } K _ { 1 , q , \pm } ( u , v ) \partial _ { u } ^ { \alpha } K _ { 2 , q } ( u , v ) \partial _ { u } ^ { \nu } K _ { 3 , q } ( u , v ) | } \\ & { \qquad \lesssim \displaystyle \sum _ { \alpha + \beta + \nu = \gamma _ { 1 } } 2 ^ { 2 q s } ( 2 ^ { - q } + | w \mp w ^ { \prime } | ) ^ { \beta } ( 1 + | v | + 2 ^ { q } | w \mp w ^ { \prime } | ) ^ { - \gamma _ { 2 } - 2 } \times 2 ^ { - q ( 2 k + 2 + \alpha ) } \times 2 ^ { - q s } } \\ & { \qquad \lesssim ( 2 ^ { - q } + | w \mp w ^ { \prime } | ) ^ { \gamma _ { 1 } } ( 1 + | v | + 2 ^ { q } | w \mp w ^ { \prime } | ) ^ { - \gamma _ { 2 } - 2 } 2 ^ { - q ( 2 ( k - s ) + 2 ) } . } \end{array}
$$

Integrating first with respect to $v ,$ we obtain, for every $\gamma _ { 2 } \geq 0$

$$
\int _ { \mathbb { R } } ( 1 + | v | + 2 ^ { q } | w \mp w ^ { \prime } | ) ^ { - \gamma _ { 2 } - 2 } d v = \frac 2 { \gamma _ { 2 } + 1 } ( 1 + 2 ^ { q } | w \mp w ^ { \prime } | ) ^ { - \gamma _ { 2 } - 1 } \lesssim _ { \gamma _ { 2 } } ( 1 + 2 ^ { q } | w \mp w ^ { \prime } | ) ^ { - \gamma _ { 2 } } .
$$

The u-interval has length $O ( 2 ^ { q } )$ , so

$$
\begin{array} { r l } & { \displaystyle \iint _ { \mathbb R ^ { 2 } } | \partial _ { u } ^ { \gamma _ { 1 } } A _ { q , \pm } ( u , v ) | d u d v \lesssim _ { \gamma _ { 1 } , \gamma _ { 2 } } \int _ { - 2 q + 4 } ^ { 2 ^ { q + 4 } } ( 2 ^ { - q } + | w \mp w ^ { \prime } | ) ^ { \gamma _ { 1 } } ( 1 + 2 ^ { q } | w \mp w ^ { \prime } | ) ^ { - \gamma _ { 2 } } 2 ^ { - q ( 2 ( k - s ) + 2 ) } d u } \\ & { \qquad \lesssim ( 1 + 2 ^ { q } | w \mp w ^ { \prime } | ) ^ { - \gamma _ { 2 } } 2 ^ { - q ( 2 ( k - s ) + 1 ) } ( 2 ^ { - q } + | w \mp w ^ { \prime } | ) ^ { \gamma _ { 1 } } . } \end{array}
$$

## 5.3 Proof of Theorem 5.1

Proof. By (5.5),

$$
| \left. \Delta _ { q , s } \varphi _ { \theta } , \Delta _ { q , s } \varphi _ { \theta ^ { \prime } } \right. _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \big | \le \frac { 1 } { ( 2 \pi ) ^ { d + 2 } } \sum _ { i , j = 1 , 2 , 3 } \left| \int _ { \mathbb { R } ^ { d } } \zeta _ { q , s } ( \xi ) ^ { 2 } I _ { i } ( \theta , q ; \xi ) \overline { { I _ { j } ( \theta ^ { \prime } , q ; \xi ) } } d \xi \right| ,
$$

where $I _ { 3 } ( \theta , q ; \circ )$ denotes $I _ { 3 } ( \theta ; \circ )$ . By Lemmas 5.1 and 5.2, it remains to prove

$$
\left| \int _ { \mathbb { R } ^ { d } } \zeta _ { q , s } ( \xi ) ^ { 2 } I _ { 1 } ( \theta , q ; \xi ) \overline { { I _ { 1 } ( \theta ^ { \prime } , q ; \xi ) } } d \xi \right| \lesssim \frac { 2 ^ { - q ( 2 ( k - s ) + 1 ) } } { ( 2 ^ { q } \varrho _ { \pm } ( \theta , \theta ^ { \prime } ) ) ^ { \gamma } } .\tag{5.18}
$$

For ${ \cal J } _ { + } ( \theta , \theta ^ { \prime } , q )$ , we use the two cases in Section 5.2.

Case 1: $| w - w ^ { \prime } | \gtrsim \varrho ( \theta , \theta ^ { \prime } ) \geq \varrho _ { \pm } ( \theta , \theta ^ { \prime } )$

In this case, the decay follows from the overlap of the two Fourier packets

$$
\widehat { \chi } ( \xi - \tau w ) , \qquad \widehat { \chi } ( \xi - \tau ^ { \prime } w ^ { \prime } ) .
$$

Indeed, applying the estimate in Lemma 5.3 with $\gamma _ { 1 } = 0 , \gamma _ { 2 } = \gamma$ to (5.15), and recalling that supp $( A _ { q , + } ) \subset [ - 2 ^ { q + 4 } , 2 ^ { q + 4 } ] ^ { 2 }$ , we obtain

$$
\begin{array} { r } { | J _ { + } ( \theta , \theta ^ { \prime } , q ) | \lesssim 2 ^ { - q ( 2 ( k - s ) + 1 ) } ( 1 + 2 ^ { q } | w - w ^ { \prime } | ) ^ { - \gamma } . } \end{array}
$$

Together with $| w - w ^ { \prime } | \gtrsim \varrho _ { \pm } ( \theta , \theta ^ { \prime } )$ , this proves the first estimate in (5.14).

Case 2: $| b - b ^ { \prime } | \gtrsim \varrho ( \theta , \theta ^ { \prime } ) \geq \varrho _ { \pm } ( \theta , \theta ^ { \prime } )$

In this case, the decay comes from the oscillation of the phase factor $e ^ { i ( \tau b - \tau ^ { \prime } b ^ { \prime } ) }$ . To see this, we integrate by parts γ times in u in (5.15) and apply Lemma 5.3 with $\gamma _ { 1 } = \gamma _ { 2 } = \gamma$ to obtain

$$
\begin{array} { l } { \displaystyle | J _ { + } ( \theta , \theta ^ { \prime } , q ) | = \left| \frac { 1 } { 2 } \Big ( \frac { 2 i } { b - b ^ { \prime } } \Big ) ^ { \gamma } \iint _ { \mathbb R ^ { 2 } } e ^ { \frac { i \pi } { 2 } ( b - b ^ { \prime } ) } e ^ { \frac { i \pi } { 2 } ( b + b ^ { \prime } ) } \partial _ { u } ^ { \gamma } A _ { q , + } ( u , v ) d u d v \right| } \\ { \displaystyle \lesssim | b - b ^ { \prime } | ^ { - \gamma } 2 ^ { - q ( 2 ( k - s ) + 1 ) } ( 2 ^ { - q } + | w - w ^ { \prime } | ) ^ { \gamma } ( 1 + 2 ^ { q } | w - w ^ { \prime } | ) ^ { - \gamma } } \\ { \displaystyle \quad = 2 ^ { - q ( 2 ( k - s ) + 1 ) } \Big ( \frac { 1 + 2 ^ { q } | w - w ^ { \prime } | } { 2 ^ { q } | b - b ^ { \prime } | } \Big ) ^ { \gamma } ( 1 + 2 ^ { q } | w - w ^ { \prime } | ) ^ { - \gamma } = 2 ^ { - q ( 2 ( k - s ) + 1 ) } ( 2 ^ { q } | b - b ^ { \prime } | ) ^ { - \gamma } . } \end{array}
$$

Together with $| b - b ^ { \prime } | \gtrsim \varrho _ { \pm } ( \theta , \theta ^ { \prime } )$ , this proves the first estimate in (5.14).

For ${ \cal J } _ { - } ( \theta , \theta ^ { \prime } , q )$ , replace $( w ^ { \prime } , b ^ { \prime } )$ by $\left( - w ^ { \prime } , - b ^ { \prime } \right)$ in the same argument. This gives (5.14) for all $\gamma \in  { \mathbb { N } } _ { 0 }$ , and hence all $\gamma > 0$ , proving (5.18) and Theorem 5.1.

## 5.4 Dyadic estimates for individual localized neurons

Theorem 5.2. For every $\begin{array} { r } { 0 \leq s < k + \frac { 1 } { 2 } } \end{array}$ , there exists $q _ { 0 } \in \mathbb { N }$ such that

$$
\begin{array} { r } { \| \Delta _ { q , s } \varphi _ { \theta } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } ^ { 2 } \ = 2 ^ { - q ( 2 ( k - s ) + 1 ) } , \quad q \geq q _ { 0 } } \end{array}
$$

uniformly for $\theta \in A _ { \chi }$

Proof. Again, it sufices to consider $\theta = { \binom { w } { b } } \in B _ { \chi }$ . Fix θ and, for $y \in w ^ { \perp }$ , define

$$
\chi _ { \theta } ( y ) = \chi ( y - b w ) .
$$

Using the orthogonal decomposition

$$
\begin{array} { r } { x = y + ( t - b ) w , \qquad y \in w ^ { \perp } , \quad t = w ^ { \top } x + b , } \end{array}
$$

we have

$$
\varphi _ { \theta } ( y + ( t - b ) w ) = \chi ( y + ( t - b ) w ) \sigma _ { k } ( t ) .
$$

By Leibniz’s rule and the identity

$$
\partial _ { t } ^ { k + 1 } \sigma _ { k } ( t ) = k ! \delta _ { 0 }
$$

in the sense of distributions,

$$
\partial _ { t } ^ { k + 1 } \left[ \chi ( y + ( t - b ) w ) \sigma _ { k } ( t ) \right] = k ! \chi _ { \theta } ( y ) \delta _ { 0 } ( t ) + R _ { \theta } ( y , t ) ,
$$

where

$$
R _ { \theta } ( y , t ) = \sum _ { j = 1 } ^ { k + 1 } \binom { k + 1 } { j } \frac { k ! } { ( j - 1 ) ! } ( w ^ { \top } \nabla ) ^ { j } \chi ( y + ( t - b ) w ) t _ { + } ^ { j - 1 } .
$$

Since $\chi \in \mathcal { C } _ { c } ^ { \infty } ( \mathbb { R } ^ { d } )$ and the normalized active parameter set is bounded, $R _ { \theta }$ is uniformly compactly supported and smooth with respect to y, and its derivative with respect to t is a finite measure. Consequently, for every $\gamma > 0$

$$
\begin{array} { r } { | \widehat { \chi _ { \theta } } ( \eta ) | \lesssim _ { \gamma } ( 1 + | \eta | ) ^ { - \gamma } , \qquad | \widehat { R _ { \theta } } ( \eta , \lambda ) | \lesssim _ { \gamma } | \lambda | ^ { - 1 } ( 1 + | \eta | ) ^ { - \gamma } } \end{array}
$$

uniformly for $\theta \in B _ { \chi }$ and $| \lambda | \geq 1$

Taking the Fourier transform in the variables $( y , t )$ gives

$$
( i \lambda ) ^ { k + 1 } e ^ { - i \lambda b } \widehat { \varphi _ { \theta } } ( \eta + \lambda w ) = k ! \widehat { \chi _ { \theta } } ( \eta ) + \widehat { R _ { \theta } } ( \eta , \lambda ) .
$$

Therefore,

$$
\widehat { \varphi _ { \theta } } ( \eta + \lambda w ) = \frac { e ^ { i \lambda b } k ! } { ( i \lambda ) ^ { k + 1 } } \widehat { \chi _ { \theta } } ( \eta ) + { \mathcal O } _ { \gamma } \left( | \lambda | ^ { - k - 2 } ( 1 + | \eta | ) ^ { - \gamma } \right)\tag{5.19}
$$

uniformly for $\theta \in B _ { \chi }$ and $| \lambda | \geq 1$

We first prove the upper bound. By the definition of $\Delta _ { q , s }$ and Parseval’s identity,

$$
\| \Delta _ { q , s } \varphi _ { \theta } \| _ { 2 } ^ { 2 } = \frac { 1 } { ( 2 \pi ) ^ { d } } \int _ { \mathbb { R } ^ { d } } \zeta \left( \frac { | \xi | } { 2 ^ { q } } \right) ^ { 2 } ( 1 + | \xi | ^ { 2 } ) ^ { s } | \widehat { \varphi _ { \theta } } ( \xi ) | ^ { 2 } d \xi .
$$

Writing $\xi = \eta +$ λw with $\eta \in w ^ { \perp }$ , we have $| \xi | \lesssim 2 ^ { q }$ on the support of the multiplier. We split this region into

$$
| \lambda | \geq 2 ^ { q - 2 } \quad \mathrm { a n d } \quad | \lambda | < 2 ^ { q - 2 } .
$$

For $| \lambda | \geq 2 ^ { q - 2 }$ , the Fourier representation (5.19) yields

$$
| \widehat { \varphi _ { \theta } } ( \eta + \lambda w ) | \lesssim _ { \gamma } | \lambda | ^ { - k - 1 } ( 1 + | \eta | ) ^ { - \gamma }
$$

and we obtain

$$
\begin{array} { r l } & { \displaystyle \int _ { | \lambda | \geq 2 ^ { q - 2 } } \zeta \left( \frac { | \xi | } { 2 ^ { q } } \right) ^ { 2 } ( 1 + | \xi | ^ { 2 } ) ^ { s } | \widehat { \varphi _ { \theta } } ( \xi ) | ^ { 2 } d \xi \leq \displaystyle \int _ { | \lambda | \geq 2 ^ { q - 2 } } \int _ { w ^ { \perp } } ( 1 + | \eta + \lambda w | ^ { 2 } ) ^ { s } | \widehat { \varphi _ { \theta } } ( \eta + \lambda w ) | ^ { 2 } d \eta d \lambda } \\ & { \displaystyle \lesssim \int _ { | \lambda | \geq 2 ^ { q - 2 } } | \lambda | ^ { - 2 ( k - s ) - 2 } d \lambda \lesssim 2 ^ { - q ( 2 ( k - s ) + 1 ) } . } \end{array}
$$

For $| \lambda | < 2 ^ { q - 2 }$ , the multiplier is supported on $2 ^ { q - 1 } \leq | \xi | \leq 2 ^ { q + 1 }$ , so $| \eta | \geq 2 ^ { q - 2 }$ . Tangential smoothness of $\varphi _ { \theta }$ gives

$$
\begin{array} { r } { \left| \widehat { \varphi _ { \theta } } ( \eta + \lambda w ) \right| \lesssim _ { \gamma } ( 1 + | \eta | ) ^ { - \gamma } . } \end{array}
$$

Choosing $\gamma$ suficiently large gives

$$
\int _ { | \lambda | < 2 ^ { q - 2 } } \zeta \left( \frac { | \xi | } { 2 ^ { q } } \right) ^ { 2 } ( 1 + | \xi | ^ { 2 } ) ^ { s } | \widehat { \varphi _ { \theta } } ( \xi ) | ^ { 2 } d \xi \lesssim 2 ^ { - q ( 2 ( k - s ) + 1 ) } .
$$

Hence, uniformly for $\theta \in B _ { \chi }$ ，

$$
\begin{array} { r } { \| \Delta _ { q , s } \varphi _ { \theta } \| _ { 2 } ^ { 2 } \lesssim 2 ^ { - q ( 2 ( k - s ) + 1 ) } . } \end{array}
$$

We next prove the lower bound. The Fourier transform of $\chi _ { \theta }$ on $w ^ { \perp } \simeq \mathbb { R } ^ { d - 1 }$ is

$$
\widehat { \chi _ { \theta } } ( \eta ) = \int _ { w ^ { \perp } } \chi ( y - b w ) e ^ { - i \eta \cdot y } d y , \qquad \eta \in w ^ { \perp } .
$$

Since each active hyperplane intersects $U _ { ☉ }$ , the uniform continuity of $\chi$ provides a hyperplane disk of fixed radius on which $\chi \geq \frac { c _ { 1 } } { 2 }$ ; hence Parseval’s identity on $w ^ { \perp }$ gives

$$
\operatorname* { i n f } _ { \theta \in \mathcal { B } _ { x } } \int _ { w ^ { \perp } } | \widehat { \chi _ { \theta } } ( \eta ) | ^ { 2 } d \eta = ( 2 \pi ) ^ { d - 1 } \operatorname* { i n f } _ { \theta \in \mathcal { B } _ { x } } \| \chi _ { \theta } \| _ { \mathcal { L } ^ { 2 } ( w ^ { \perp } ) } ^ { 2 } > 0 .
$$

Therefore, there exists $R > 0 .$ , independent of $\theta ,$ such that

$$
\int _ { | \eta | \leq R } | \widehat { \chi _ { \theta } } ( \eta ) | ^ { 2 } d \eta \geq c > 0 .
$$

Consider the region

$$
| \eta | \leq R , \qquad { \frac { 4 } { 5 } } 2 ^ { q } \leq \lambda \leq { \frac { 5 } { 4 } } 2 ^ { q } .
$$

For suficiently large $q ,$ this region satisfies

$$
\frac { 3 } { 5 } \leq \frac { | \eta + \lambda w | } { 2 ^ { q } } \leq \frac { 5 } { 3 } ,
$$

and hence

$$
\zeta \left( \frac { \lvert \eta + \lambda w \rvert } { 2 ^ { q } } \right) \geq c _ { 2 } > 0 .
$$

The leading term in (5.19) satisfies

$$
2 ^ { 2 q s } \int _ { | \eta | \le R } \int _ { \frac 4 5 2 ^ { q } } ^ { \frac 5 4 2 ^ { q } } | \lambda | ^ { - 2 k - 2 } | \widehat { \chi _ { \theta } } ( \eta ) | ^ { 2 } d \lambda d \eta \gtrsim 2 ^ { - q ( 2 k - 2 s + 1 ) } ,
$$

while the remainder is bounded by

$$
2 ^ { 2 q s } \int _ { | \eta | \leq R } \int _ { \frac { 4 } { 5 } 2 ^ { q } } ^ { \frac { 5 } { 4 } 2 ^ { q } } | \lambda | ^ { - 2 k - 4 } ( 1 + | \eta | ) ^ { - 2 \gamma } d \lambda d \eta \lesssim 2 ^ { - q } 2 ^ { - q ( 2 k - 2 s + 1 ) } .
$$

Hence, for suficiently large $q ,$ uniformly for $\theta \in B _ { \chi }$ χ

$$
\begin{array} { r } { \| \Delta _ { q , s } \varphi _ { \theta } \| _ { 2 } ^ { 2 } \gtrsim 2 ^ { - q ( 2 k - 2 s + 1 ) } . } \end{array}
$$

## 6 Proofs of the main results

## 6.1 Proof of Theorem 4.2

Proof. The set $\{ \theta _ { j } ^ { * } \} _ { j = 1 } ^ { n }$ can be extended to a collection $\{ \eta _ { i } ^ { * } \} _ { i = 1 } ^ { N } \subset \mathbb { S } ^ { d }$ with $N \gtrsim n , \operatorname* { m a x } _ { \eta \in \mathbb { S } ^ { d } } \operatorname* { m i n } _ { 1 \leq i \leq N } \rho ( \eta , \eta _ { i } ^ { * } ) =$ h and $\{ \eta _ { i } ^ { * } \} _ { i = 1 } ^ { N } \cap \mathcal { A } _ { \chi } = \{ \theta _ { j } ^ { * } \} _ { j = 1 } ^ { n }$ . Then, by [67], there exist $g _ { N } ( x ) = \sum _ { i = 1 } ^ { N } a _ { i } \sigma _ { k } ( \eta _ { i } ^ { * } \cdot \widetilde { x } )$ such that

$$
\lVert f - g _ { N } \rVert _ { \mathcal { H } ^ { s } ( \Omega ^ { \prime } ) } \leq \lVert f - g _ { N } \rVert _ { \mathcal { H } ^ { s } ( \Omega ) } \lesssim h ^ { r - s } \lVert f \rVert _ { \mathcal { H } ^ { r } ( \Omega ) } .\tag{6.1}
$$

Moreover, the coeficients can be chosen to satisfy

$$
\left( N \sum _ { j = 1 } ^ { N } a _ { j } ^ { 2 } \right) ^ { \frac { 1 } { 2 } } \lesssim N ^ { \alpha } \| f \| _ { \mathcal H ^ { r } ( \Omega ) }
$$

if $\{ \eta _ { i } ^ { * } \} _ { i = 1 } ^ { N }$ is quasi-uniform.

Neurons with $\eta _ { i } ^ { * } \notin { \mathcal { A } } _ { \chi }$ are either zero or equal to $( \eta _ { i } ^ { * } \cdot \widetilde { x } ) ^ { k }$ on $U .$ . Hence

$$
\varphi _ { \eta _ { i } ^ { * } } ( x ) \equiv 0 \quad \mathrm { o r } \quad \varphi _ { \eta _ { i } ^ { * } } ( x ) = ( \eta _ { i } ^ { * } \cdot \widetilde { x } ) ^ { k } , \qquad x \in \Omega ^ { \prime } .
$$

Let

$$
\begin{array} { r } { \mathcal A _ { \Omega ^ { \prime } } : = \left\{ \theta = \binom { w } { b } \in \mathbb S ^ { d } : \left\{ x : w ^ { \top } x + b = 0 \right\} \cap \Omega ^ { \prime } \neq \emptyset \right\} , } \end{array}
$$

so that we can write

$$
g _ { N } ( x ) = \sum _ { \{ i : \eta _ { i } ^ { * } \in A _ { \Omega ^ { \prime } } \} } a _ { i } \varphi _ { \eta _ { i } ^ { * } } ( x ) + p ( x ) , \quad x \in \Omega ^ { \prime } ,
$$

where $p$ is a polynomial of degree $\leq k$ . We now show that $p$ can be represented by the remaining basis functions. To this end, we prove

$$
\mathrm { s p a n } \{ \varphi _ { \eta _ { i } ^ { * } } | _ { \Omega ^ { \prime } } : \eta _ { i } ^ { * } \in \mathcal { A } _ { \chi } \setminus \mathcal { A } _ { \Omega ^ { \prime } } \} = \mathbb { P } _ { k } ( \Omega ^ { \prime } )\tag{6.2}
$$

where $\mathbb { P } _ { k } ( \Omega ^ { \prime } )$ is the space of polynomials of $\mathrm { d e g r e e } \le k$ on $\Omega ^ { \prime }$

Since $\Omega ^ { \prime } \Subset U ^ { \circ }$ , there exists δ such that

$$
\{ x + y : \ x \in \Omega ^ { \prime } , \ | y | \leq \delta \} \subset U .
$$

Define

$$
\theta ( \omega , t ) : = \frac { 1 } { \sqrt { 1 + ( t - \underset { x \in \overline { { \Omega } ^ { \prime } } } { \operatorname* { m i n } } \omega \cdot x ) ^ { 2 } } } \binom { \omega } { t - \underset { x \in \overline { { \Omega } ^ { \prime } } } { \operatorname* { m i n } } \omega \cdot x } , \qquad \omega \in \mathbb { S } ^ { d - 1 } , \ t \geq 0 ,
$$

so that $\theta ( \omega , t ) \in \mathbb { S } ^ { d }$ and, for $0 < t < \delta , \theta ( \omega , t ) \in \mathcal { A } _ { \chi } \setminus \mathcal { A } _ { \Omega ^ { \prime } }$

1. For any $x \in \Omega ^ { \prime }$

$$
{ \binom { x } { 1 } } \cdot { \binom { \omega } { t - \operatorname* { m i n } _ { x \in \Omega ^ { \prime } } \omega \cdot x } } = \omega \cdot x + t - \operatorname* { m i n } _ { x \in \overline { { \Omega ^ { \prime } } } } \omega \cdot x > 0 .
$$

2. Take $x _ { 0 } \in \overline { { \Omega ^ { \prime } } }$ with $\omega \cdot x _ { 0 } = \operatorname* { m i n } _ { x \in \overline { { \Omega ^ { \prime } } } } \omega \cdot x ;$ then $x _ { 0 } - t \omega \in U$ and

$$
\binom { x _ { 0 } - t \omega } { 1 } \cdot \binom { \omega } { t - \underset { x \in \Omega ^ { \prime } } { \operatorname* { m i n } } \omega \cdot x } = \omega \cdot x _ { 0 } - t \omega \cdot \omega + t - \underset { x \in \Omega ^ { \prime } } { \operatorname* { m i n } } \omega \cdot x = 0 .
$$

Define

$$
\mathcal { S } _ { \delta } : = \left. \theta ( \omega , t ) : ~ \omega \in \mathbb { S } ^ { d - 1 } , ~ \frac { \delta } { 3 } < t < \frac { 2 \delta } { 3 } \right.\tag{6.3}
$$

and $K = { \binom { d + k } { k } }$ . The class $\left\{ ( \vartheta \cdot \widetilde { x } ) ^ { k } : \vartheta \in S _ { \delta } \right\}$ spans $\mathbb { P } _ { k } \mathbf { : }$ : any annihilating linear functional induces a homogeneous polynomial in ϑ vanishing on the open spherical set ${ \cal { S } } _ { \delta }$ , so its coeficients, and hence the functional, vanish. Thus there exist $\vartheta _ { 1 } ^ { \ast } , \dots , \vartheta _ { K } ^ { \ast } \in { \mathcal { S } } _ { \delta }$ such that $( \vartheta _ { u } ^ { \ast } \cdot \widetilde { x } ) ^ { k } , u = 1 , \dots , K$ , are linearly independent. Equivalently,

$$
\operatorname* { d e t } \left( \int _ { \Omega ^ { \prime } } ( \vartheta _ { u } ^ { \ast } \cdot \widetilde { x } ) ^ { k } ( \vartheta _ { v } ^ { \ast } \cdot \widetilde { x } ) ^ { k } d x \right) _ { u , v = 1 } ^ { K } > 0 .
$$

Choose $h _ { \delta , 1 } > 0$ such that

$$
\left\{ \theta \in \mathbb { S } ^ { d } : \rho ( \theta , \mathcal { S } _ { \delta } ) \leq h _ { \delta , 1 } \right\} \subset \left\{ \theta ( \omega , t ) : \ \omega \in \mathbb { S } ^ { d - 1 } , \ 0 < t < \delta \right\}
$$

and choose $h _ { \delta , 2 } > 0$ such that

$$
\operatorname* { d e t } \left( \int _ { \Omega ^ { \prime } } ( \vartheta _ { u } \cdot \widetilde { x } ) ^ { k } ( \vartheta _ { v } \cdot \widetilde { x } ) ^ { k } d x \right) _ { u , v = 1 } ^ { K } > 0\tag{6.4}
$$

whenever $| \vartheta _ { u } - \vartheta _ { u } ^ { * } | < h _ { \delta , 2 }$

Set $h _ { \delta } = \operatorname* { m i n } \bigg \{ \frac { 1 } { 2 } \operatorname* { m i n } _ { u \neq v } \{ \rho ( \vartheta _ { u } ^ { \ast } , \vartheta _ { v } ^ { \ast } ) \} , h _ { \delta , 1 } , h _ { \delta , 2 } \bigg \}$ . Then, for the extended collection $\{ \eta _ { i } ^ { * } \} _ { i = 1 } ^ { N }$ , whose active subset is $\{ \theta _ { j } ^ { * } \} _ { j = 1 } ^ { n }$ , whenever $h < h _ { \delta }$ there exists a collection

$$
\{ \eta _ { i _ { u } } ^ { * } \} _ { u = 1 } ^ { K } \subset \{ \theta ( \omega , t ) : \ \omega \in \mathbb { S } ^ { d - 1 } , \ 0 < t < \delta \} \subset A _ { \chi } \backslash \mathcal { A } _ { \Omega ^ { \prime } }
$$

such that $| \eta _ { i , \ i } ^ { \ast } - \vartheta _ { u } ^ { \ast } | < h _ { \delta } , \ u = 1 , \dots , K$ . The corresponding determinant in (6.4) is therefore positive, so $( \bar { 6 . 2 } )$ holds. We next verify that replacing the polynomial part preserves the coeficient scale. By decreasing $h _ { \delta , 2 }$ if necessary, the continuity of the Gram matrix and (6.4) imply that there exists $c _ { 4 } = c _ { 4 } ( \delta ) > 0$ such that

$$
\lambda _ { \operatorname* { m i n } } \left( \int _ { \Omega ^ { \prime } } ( \vartheta _ { u } \cdot \widetilde { x } ) ^ { k } ( \vartheta _ { v } \cdot \widetilde { x } ) ^ { k } d x \right) _ { u , v = 1 } ^ { K } \geq c _ { 4 }\tag{6.5}
$$

whenever $| \vartheta _ { u } - \vartheta _ { u } ^ { * } | < h _ { \delta , 2 } , u = 1 , \ldots , K .$

Therefore, there exists a suficiently small $\varepsilon > 0 .$ independent of $n ,$ such that

$$
\begin{array} { r l } & { B ( \vartheta _ { u } ^ { \ast } , \varepsilon ) : = \left\{ \theta \in \mathbb S ^ { d } : \rho ( \theta , \vartheta _ { u } ^ { \ast } ) \leq \varepsilon \right\} \subset S _ { \delta } , \qquad u = 1 , \ldots , K , } \\ & { \operatorname { d i s t } \big ( B ( \vartheta _ { u } ^ { \ast } , \varepsilon ) , B ( \vartheta _ { v } ^ { \ast } , \varepsilon ) \big ) \geq \varepsilon , \qquad u \neq v . } \end{array}\tag{6.6}
$$

Decrease $h _ { \delta }$ so that $h _ { \delta } \leq \varepsilon .$ By the fill-distance condition, $B ( \vartheta _ { u } ^ { \ast } , \varepsilon ) \cap \{ \eta _ { i } ^ { \ast } \} _ { i = 1 } ^ { N } \neq \varnothing$ for every u whenever $h < h _ { \delta }$ . Choose $\eta _ { i _ { u } } ^ { * }$ in each intersection. Then (6.5) gives coeficients $\beta _ { 1 } , \ldots , \beta _ { K }$ such that

$$
p ( x ) = \sum _ { u = 1 } ^ { K } \beta _ { u } \varphi _ { \eta _ { i _ { u } } ^ { * } } ( x ) , \qquad x \in \Omega ^ { \prime } ,
$$

which proves (4.2).

Next, we assume quasi-uniformity and prove (4.3). Since each $B ( \vartheta _ { u } ^ { * } , \varepsilon )$ is independent of $n ,$ for suficiently large n we have

$$
n ^ { * } : = \operatorname* { m i n } _ { 1 \leq u \leq K } \# \Big ( B ( \vartheta _ { u } ^ { * } , \varepsilon ) \cap \{ \eta _ { i } ^ { * } \} _ { i = 1 } ^ { N } \Big ) \asymp n .
$$

Choose $n ^ { * }$ elements from each $B ( \vartheta _ { u } ^ { * } , \varepsilon ) \cap \{ \eta _ { i } ^ { * } \} _ { i = 1 } ^ { N }$ , denoted by

$$
\{ \eta _ { i _ { u , \ell } } ^ { * } : u = 1 , \ldots , K \} , \qquad \ell = 1 , \ldots , n ^ { * } ,
$$

such that $\eta _ { i _ { u . \ell } } ^ { * } \in B ( \vartheta _ { u } ^ { * } , \varepsilon ) \subset A _ { \chi } \setminus A _ { \Omega ^ { \prime } }$ . Again, for every ℓ, (6.5) shows that there exist coeficients $\beta _ { u , \ell }$ satisfying

$$
p ( x ) = \sum _ { u = 1 } ^ { K } \beta _ { u , \ell } \varphi _ { \eta _ { i _ { u , \ell } } ^ { * } } ( x ) , \qquad x \in \Omega ^ { \prime } ,
$$

and $\sum _ { u = 1 } ^ { K } | \beta _ { u , \ell } | ^ { 2 } \lesssim \| p \| _ { \mathcal L ^ { 2 } ( \Omega ^ { \prime } ) } ^ { 2 } .$

To prove (4.3), note that $h \gtrsim n ^ { - \frac { 1 } { d } }$ in the quasi-uniform setting, so it sufices to establish the coeficient bound. Averaging these representations gives

$$
p ( x ) = \sum _ { \ell = 1 } ^ { n ^ { * } } \sum _ { u = 1 } ^ { K } \alpha _ { u , \ell } \varphi _ { \eta _ { i _ { u , \ell } } ^ { * } } ( x ) , \qquad \alpha _ { u , \ell } : = \frac { \beta _ { u , \ell } } { n ^ { * } } ,
$$

and hence

$$
\sum _ { \ell = 1 } ^ { n ^ { * } } \sum _ { u = 1 } ^ { K } | \alpha _ { u , \ell } | ^ { 2 } \lesssim \frac { 1 } { n ^ { * } } \| p \| _ { \mathcal L ^ { 2 } ( \Omega ^ { \prime } ) } ^ { 2 } \lesssim \frac { 1 } { n } \| p \| _ { \mathcal L ^ { 2 } ( \Omega ^ { \prime } ) } ^ { 2 } .\tag{6.7}
$$

Moreover, since every neuron contributing to p is either zero or a polynomial $( \eta _ { i } ^ { * } \cdot \widetilde { x } ) ^ { k }$ on $\Omega ^ { \prime }$ and $\Omega ^ { \prime }$ is bounded, we have

$$
\| p \| _ { \mathcal { L } ^ { 2 } ( \Omega ^ { \prime } ) } \lesssim \sum _ { i = 1 } ^ { N } | a _ { i } | \leq \sqrt { N } \left( \sum _ { i = 1 } ^ { N } a _ { i } ^ { 2 } \right) ^ { 1 / 2 } .
$$

It follows that

$$
n \sum _ { \ell = 1 } ^ { n ^ { * } } \sum _ { u = 1 } ^ { K } | \alpha _ { u , \ell } | ^ { 2 } \lesssim \| p \| _ { \mathcal L ^ { 2 } ( \Omega ^ { \prime } ) } ^ { 2 } \lesssim N \sum _ { i = 1 } ^ { N } a _ { i } ^ { 2 } .\tag{6.8}
$$

Consequently, replacing the polynomial part of $g _ { N }$ by the above active neurons preserves, up to a constant, the coeficient bound of the original $g _ { N }$ □

## 6.2 Proof of Theorem 4.3

## 6.2.1 Upper bound

Proof. The upper estimate in Theorem 5.2 implies

$$
\operatorname* { s u p } _ { \theta \in A _ { \chi } } \left\| \varphi _ { \theta } \right\| _ { B _ { 2 , \infty } ^ { k + \frac { 1 } { 2 } } ( \mathbb R ^ { d } ) } \lesssim 1 .
$$

Thus there are subspaces $V _ { m } \subset \mathcal { H } ^ { s } ( \mathbb { R } ^ { d } )$ of dimension at most m such that

$$
\dim V _ { m } \leq m , \qquad \operatorname* { s u p } _ { \theta \in \mathcal A _ { x } } \operatorname* { i n f } _ { v \in V _ { m } } \Vert \varphi _ { \theta } - v \Vert _ { \mathcal H ^ { s } ( \mathbb R ^ { d } ) } ^ { 2 } \lesssim m ^ { - \frac { 2 ( k - s ) + 1 } { d } } , \qquad m \geq 1 .\tag{6.9}
$$

To see this, extend $\varphi _ { \theta }$ periodically, with Ω contained in the interior of a period cell $[ - A , A ] ^ { d } .$ Then standard Fourier series approximation (see, e.g., [109, Chapters 2 and 9]) of the extended periodic function yields the approximation rate $\mathcal { O } ( m ^ { - \frac { 2 ( k - s ) + 1 } { 2 d } } )$ . Finally, multiplying the periodic approximant by another smooth cutof $\chi _ { e } \in \mathcal { C } ^ { \infty } ( \mathbb { R } ^ { d } )$ such that

$$
\chi _ { e } ( x ) = 1 , \quad x \in \Omega , \qquad \chi _ { e } ( x ) = 0 , \quad x \notin [ - A , A ] ^ { d }
$$

yields the approximant v and the space $V _ { m }$

Let $\operatorname { P r o j } _ { V _ { m } }$ be the $\mathcal { H } ^ { s } ( \mathbb { R } ^ { d } )$ -orthogonal projection onto $V _ { m } .$ , and define

$$
E _ { m } : = \left( \Big \langle ( I - \mathrm { P r o j } _ { V _ { m } } ) \varphi _ { \theta _ { i } ^ { * } } , ( I - \mathrm { P r o j } _ { V _ { m } } ) \varphi _ { \theta _ { j } ^ { * } } \Big \rangle _ { \mathscr { H } ^ { s } ( \mathbb { R } ^ { d } ) } \right) _ { i , j = 1 } ^ { n } .
$$

Orthogonality gives

$$
\begin{array} { r } { \Big \langle \mathrm { P r o j } _ { V _ { m } } \varphi _ { \theta _ { i } ^ { * } } , ( I - \mathrm { P r o j } _ { V _ { m } } ) \varphi _ { \theta _ { j } ^ { * } } \Big \rangle _ { \mathcal { H } ^ { s } ( \mathbb { R } ^ { d } ) } = 0 , \qquad 1 \leq i , j \leq n , } \end{array}
$$

and hence

$$
K _ { s } - E _ { m } = \left( \left. \operatorname { P r o j } _ { V _ { m } } \varphi _ { \boldsymbol { \theta } _ { i } ^ { * } } , \operatorname { P r o j } _ { V _ { m } } \varphi _ { \boldsymbol { \theta } _ { j } ^ { * } } \right. _ { \mathcal { H } ^ { s } ( \mathbb { R } ^ { d } ) } \right) _ { i , j = 1 } ^ { n } .
$$

Therefore,

$$
0 \preceq E _ { m } \preceq K _ { s } , \qquad \mathrm { r a n k } ( K _ { s } - E _ { m } ) \leq \dim V _ { m } \leq m .
$$

Moreover, by the best-approximation property of $\operatorname { P r o j } _ { V _ { m } }$ and (6.9),

$$
\mathrm { t r } E _ { m } = \sum _ { i = 1 } ^ { n } \| ( I - \mathrm { P r o j } _ { V _ { m } } ) \varphi _ { \theta _ { i } ^ { * } } \| _ { \mathcal { H } ^ { s } ( \mathbb { R } ^ { d } ) } ^ { 2 } = \sum _ { i = 1 } ^ { n } \operatorname* { i n f } _ { v \in V _ { m } } \| \varphi _ { \theta _ { i } ^ { * } } - v \| _ { \mathcal { H } ^ { s } ( \mathbb { R } ^ { d } ) } ^ { 2 } \lesssim n m ^ { - \frac { 2 ( k - s ) + 1 } { d } } .
$$

Since $E _ { m }$ is positive semidefinite,

$$
\sum _ { \ell = 1 } ^ { m } \lambda _ { \ell } ( K _ { s } ) \geq \sum _ { \ell = 1 } ^ { m } \lambda _ { \ell } ( K _ { s } - E _ { m } ) = \operatorname { t r } ( K _ { s } - E _ { m } )
$$

and

$$
\sum _ { \ell = m + 1 } ^ { n } \lambda _ { \ell } ( K _ { s } ) = \operatorname { t r } K _ { s } - \sum _ { \ell = 1 } ^ { m } \lambda _ { \ell } ( K _ { s } ) \leq \operatorname { t r } K _ { s } - \operatorname { t r } ( K _ { s } - E _ { m } ) = \operatorname { t r } E _ { m } \lesssim n m ^ { - { \frac { 2 ( k - s ) + 1 } { d } } } .\tag{6.10}
$$

For $2 \leq j \leq n$ , take $m = \lfloor j / 2 \rfloor$ . Since the eigenvalues are nonincreasing,

$$
( j - m ) \lambda _ { j } ( K _ { s } ) \leq \sum _ { \ell = m + 1 } ^ { j } \lambda _ { \ell } ( K _ { s } ) \leq \sum _ { \ell = m + 1 } ^ { n } \lambda _ { \ell } ( K _ { s } ) \lesssim n m ^ { - \frac { 2 ( k - s ) + 1 } { d } } .
$$

Using $m \geq j / 3$ and $j - m \ge j / 2$ , we obtain

$$
\lambda _ { j } ( K _ { s } ) \lesssim \frac { n } { j - m } m ^ { - \frac { 2 ( k - s ) + 1 } { d } } \lesssim n j ^ { - \frac { d + 2 ( k - s ) + 1 } { d } } , \qquad 2 \leq j \leq n .
$$

For $j = 1$ , the uniform H<sup>s</sup> bound gives

$$
\lambda _ { 1 } ( K _ { s } ) \le \operatorname { t r } K _ { s } = \sum _ { i = 1 } ^ { n } \| \varphi _ { \theta _ { i } ^ { * } } \| _ { \mathcal { H } ^ { s } ( \mathbb { R } ^ { d } ) } ^ { 2 } \lesssim n .
$$

This proves the upper bound. The constants depend only on $d , k , s$ and the fixed localization data. □

## 6.2.2 Lower bound

Proof. Define

$$
\begin{array} { r } { Q _ { s } ( q ) : = \left( \left. \Delta _ { q , s } \varphi _ { \theta _ { i } ^ { * } } , \Delta _ { q , s } \varphi _ { \theta _ { j } ^ { * } } \right. _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \right) _ { i , j = 1 } ^ { n } , \qquad K _ { s } ^ { * } : = \displaystyle \sum _ { q = 0 } ^ { \infty } Q _ { s } ( q ) . } \end{array}
$$

By the definition of the Sobolev norm,

$$
( K _ { s } ) _ { i j } = \int _ { \mathbb { R } ^ { d } } ( I - \triangle ) ^ { \frac { s } { 2 } } \varphi _ { \theta _ { i } ^ { \ast } } ( x ) ( I - \triangle ) ^ { \frac { s } { 2 } } \varphi _ { \theta _ { j } ^ { \ast } } ( x ) d x = \left. \varphi _ { \theta _ { i } ^ { \ast } } , \varphi _ { \theta _ { j } ^ { \ast } } \right. _ { \mathscr { H } ^ { s } ( \mathbb { R } ^ { d } ) } , \qquad i , j = 1 , \dots , n .
$$

For each $f _ { n } = \sum _ { i = 1 } ^ { n } a _ { i } \varphi _ { \theta _ { i } ^ { * } }$ , the Littlewood–Paley norm equivalence gives

$$
a ^ { \top } K _ { s } a = \| f _ { n } \| _ { \mathcal H ^ { s } ( \mathbb R ^ { d } ) } ^ { 2 } \approx \sum _ { q = 0 } ^ { \infty } \| \Delta _ { q , s } f _ { n } \| _ { \mathcal L ^ { 2 } ( \mathbb R ^ { d } ) } ^ { 2 } = \sum _ { q = 0 } ^ { \infty } \sum _ { i , j = 1 } ^ { n } a _ { i } a _ { j } \left. \Delta _ { q , s } \varphi \theta _ { i } ^ { * } , \Delta _ { q , s } \varphi \theta _ { j } ^ { * } \right. _ { \mathcal L ^ { 2 } ( \mathbb R ^ { d } ) } = a ^ { \top } K _ { s } ^ { * } a .
$$

Then the Courant–Fischer principle [12, Chapter III] yields

$$
\lambda _ { j } ( K _ { s } ) \approx \lambda _ { j } ( K _ { s } ^ { * } ) , \qquad 1 \leq j \leq n ,\tag{6.11}
$$

where the eigenvalues are arranged in nonincreasing order. Hence it sufices to prove

$$
\lambda _ { j } ( K _ { s } ^ { * } ) \gtrsim \operatorname* { m a x } \left\{ \underline { { h } } ^ { 2 ( k - s ) + 1 } \Big ( \frac { n } { j } \Big ) ^ { \frac { 2 ( k - s ) + 1 } { d } } , ( n ^ { \frac 1 d } \underline { { h } } ) ^ { d + 2 ( k - s ) + 1 } n j ^ { - \frac { d + 2 ( k - s ) + 1 } { d } } \right\} , \qquad 1 \leq j \leq n .
$$

By Theorem 5.1, for any $\gamma > 0$ , we have

$$
| Q _ { s } ( q ) _ { i j } | \lesssim _ { \gamma } 2 ^ { - q ( 2 ( k - s ) + 1 ) } \left( 2 ^ { q } \rho _ { \pm } ( \theta _ { i } ^ { * } , \theta _ { j } ^ { * } ) \right) ^ { - \gamma } , \qquad q \geq 1 .\tag{6.12}
$$

Let $q _ { 0 } \in \mathbb { N }$ be as in Theorem 5.2. Then

$$
Q _ { s } ( q ) _ { i i } \stackrel { . . } { \sim } 2 ^ { - q ( 2 ( k - s ) + 1 ) } , \qquad q \ge q _ { 0 } , \quad 1 \le i \le n .\tag{6.13}
$$

Since spherical cap packing yields the annular count

$$
\# \left\{ j : 2 ^ { \ell } \underline { { h } } \le \rho _ { \pm } ( \theta _ { i } ^ { * } , \theta _ { j } ^ { * } ) < 2 ^ { \ell + 1 } \underline { { h } } \right\} \lesssim 2 ^ { \ell d } , \qquad \ell \le \log _ { 2 } \frac { 2 \pi } { h } ,
$$

Thus we obtain, for $\gamma > d ,$

$$
\begin{array} { c } { { \displaystyle \operatorname* { m a x } _ { i } \sum _ { j \neq i } \displaystyle | Q _ { s } ( q ) _ { i j } | \lesssim _ { \gamma } 2 ^ { - q ( 2 ( k - s ) + 1 ) } \sum _ { \ell = 0 } ^ { \infty } 2 ^ { \ell d } \operatorname* { m i n } \left\{ 1 , ( 2 ^ { q } 2 ^ { \ell } \underline { { { h } } } ) ^ { - \gamma } \right\} } } \\ { { \lesssim _ { \gamma } 2 ^ { - q ( 2 ( k - s ) + 1 ) } \left\{ ( 2 ^ { q } \underline { { { h } } } ) ^ { - d } , \quad 0 < 2 ^ { q } \underline { { { h } } } \leq 1 , \right. } }  \end{array}\tag{6.14}
$$

Together with (6.13), this yields

$$
\| Q _ { s } ( q ) \| _ { 2 } \leq \operatorname* { m a x } _ { i } \sum _ { j = 1 } ^ { n } | Q _ { s } ( q ) _ { i j } | \lesssim 2 ^ { - q ( 2 ( k - s ) + 1 ) } \left( 1 + ( 2 ^ { q } \underline { { h } } ) ^ { - d } \right) , \qquad q \geq q _ { 0 } .\tag{6.15}
$$

We establish the lower bound for $\lambda _ { j } ( K _ { s } ^ { * } )$ by choosing $q = q ( j )$ and bounding $\lambda _ { j } ( Q _ { s } ( q ) )$ from below. Let $j \le m < n$ . The trace of $Q _ { s } ( q )$ can be split into three parts:

$$
\mathrm { t r } Q _ { s } ( q ) = \sum _ { \ell = 1 } ^ { n } \lambda _ { \ell } ( Q _ { s } ( q ) ) = \sum _ { \ell = j } ^ { m } \lambda _ { \ell } ( Q _ { s } ( q ) ) + \sum _ { \ell = 1 } ^ { j - 1 } \lambda _ { \ell } ( Q _ { s } ( q ) ) + \sum _ { \ell = m + 1 } ^ { n } \lambda _ { \ell } ( Q _ { s } ( q ) ) ,\tag{6.16}
$$

The first estimate below follows from (6.13), the second from (6.15), and the third from (6.10). All corresponding constants are independent of n, q, m and j.

$$
\begin{array} { r l } & { \displaystyle \sum _ { \ell = 1 } ^ { n } \lambda _ { \ell } ( Q _ { s } ( q ) ) = \mathrm { t r } Q _ { s } ( q ) \asymp n 2 ^ { - q ( 2 ( k - s ) + 1 ) } , } \\ & { \displaystyle \sum _ { \ell = 1 } ^ { j - 1 } \lambda _ { \ell } ( Q _ { s } ( q ) ) \leq ( j - 1 ) \| Q _ { s } ( q ) \| _ { 2 } \lesssim j \big ( 1 + ( 2 ^ { q } \underline { { h } } ) ^ { - d } \big ) 2 ^ { - q ( 2 ( k - s ) + 1 ) } , } \\ & { \displaystyle \sum _ { \ell = m + 1 } ^ { n } \lambda _ { \ell } ( Q _ { s } ( q ) ) \leq \sum _ { \ell = m + 1 } ^ { n } \lambda _ { \ell } ( K _ { s } ^ { * } ) \lesssim \sum _ { \ell = m + 1 } ^ { n } \lambda _ { \ell } ( K _ { s } ) \lesssim n m ^ { - \frac { 2 ( k - s ) + 1 } { d } } . } \end{array}\tag{6.17}
$$

Thus, there exist constants $c _ { 5 } , C _ { 2 } ~ > ~ 0$ such that whenever $j \ \leq \ c _ { 5 } n , \ n \ \geq \ C _ { 2 } j ( 2 ^ { q } \underline { { { h } } } ) ^ { - d }$ , m $\geq$ min $\{ n , \lceil C _ { 2 } 2 ^ { q d } \rceil \}$ ,

$$
\sum _ { \ell = 1 } ^ { j - 1 } \lambda _ { \ell } ( Q _ { s } ( q ) ) \leq \frac { 1 } { 4 } \sum _ { \ell = 1 } ^ { n } \lambda _ { \ell } ( Q _ { s } ( q ) ) , \quad \sum _ { \ell = m + 1 } ^ { n } \lambda _ { \ell } ( Q _ { s } ( q ) ) \leq \frac { 1 } { 4 } \sum _ { \ell = 1 } ^ { n } \lambda _ { \ell } ( Q _ { s } ( q ) ) .
$$

Take $q = \lceil \log _ { 2 } ( C _ { 2 } \underline { { h } } ^ { - 1 } \left( \frac { j } { n } \right) ^ { \frac { 1 } { d } } ) \rceil$ and $m = \operatorname* { m i n } \{ n - 1 , \lceil C _ { 2 } 2 ^ { q d } \rceil \}$ , and increase $C _ { 2 }$ if necessary so that $m \geq j$ . We then obtain

$$
\begin{array} { l } { \displaystyle \sum _ { \ell = j } ^ { m } \lambda _ { \ell } ( Q _ { s } ( q ) ) \geq \displaystyle \sum _ { \ell = 1 } ^ { n } \lambda _ { \ell } ( Q _ { s } ( q ) ) - \displaystyle \sum _ { \ell = 1 } ^ { j - 1 } \lambda _ { \ell } ( Q _ { s } ( q ) ) - \displaystyle \sum _ { \ell = m + 1 } ^ { n } \lambda _ { \ell } ( Q _ { s } ( q ) ) } \\ { \displaystyle \qquad \geq \frac { 1 } { 2 } \displaystyle \sum _ { \ell = 1 } ^ { n } \lambda _ { \ell } ( Q _ { s } ( q ) ) \gtrsim n 2 ^ { - q ( 2 ( k - s ) + 1 ) } } \end{array}
$$

and consequently,

$$
\begin{array} { r l r } {  { \lambda _ { j } ( K _ { s } ) \asymp \lambda _ { j } ( K _ { s } ^ { * } ) \geq \lambda _ { j } ( Q _ { s } ( q ) ) \geq \frac { 1 } { m - j + 1 } \sum _ { \ell = j } ^ { m } \lambda _ { \ell } ( Q _ { s } ( q ) ) \gtrsim \frac { n 2 ^ { - q ( 2 ( k - s ) + 1 ) } } { \operatorname* { m i n } \{ n , C _ { 2 } 2 ^ { q d } \} } } } \\ & { } &  \asymp \operatorname* { m a x } \{ \frac { h ^ { 2 ( k - s ) + 1 } \Big ( \frac { n } { j } \Big ) ^ { \frac { 2 ( k - s ) + 1 } { d } } , ( n ^ { \frac { 1 } { d } } \underline { { h } } ) ^ { d + 2 ( k - s ) + 1 } n j ^ { - \frac { d + 2 ( k - s ) + 1 } { d } } \} . } \end{array}\tag{6.18}
$$

For $j > c _ { 5 } n$ , it sufices to consider the smallest eigenvalue $\lambda _ { n } ( K _ { s } )$ . In fact, by (6.14) and (6.13), there exists $C _ { 3 }$ such that for $2 ^ { q } \geq C _ { 3 } { \underline { { h } } } ^ { - 1 }$ ，

$$
Q _ { s } ( q ) _ { i i } - \sum _ { j \neq i } | Q _ { s } ( q ) _ { i j } | \gtrsim 2 ^ { - q ( 2 ( k - s ) + 1 ) } .\tag{6.19}
$$

Taking $q = \lceil \log _ { 2 } ( C _ { 3 } \underline { { h } } ^ { - 1 } ) \rceil$ gives

$$
\lambda _ { n } ( Q _ { s } ( q ) ) \geq \operatorname* { m i n } _ { a \in \mathbb { R } ^ { n } , \ | a | = 1 } { a ^ { \top } Q _ { s } ( q ) a \gtrsim 2 ^ { - q ( 2 ( k - s ) + 1 ) } \simeq \underline { { h } } ^ { 2 ( k - s ) + 1 } }\tag{6.20}
$$

and

$$
\lambda _ { j } ( K _ { s } ) \gtrsim \lambda _ { j } ( K _ { s } ^ { * } ) \geq \lambda _ { j } ( Q _ { s } ( q ) ) \geq \lambda _ { n } ( Q _ { s } ( q ) ) \gtrsim \underline { { h } } ^ { 2 ( k - s ) + 1 } , \quad j > c _ { 5 } n ,\tag{6.21}
$$

which agrees with the estimate in (6.18) and completes the proof.

## 6.3 Proof of Theorem 4.4

Proof. If $\underline { { { h } } } = 0 \mathrm { ~ o r ~ } \| \chi f \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } = 0$ then there is nothing to prove. Now we only consider ${ \underline { { h } } } \ > \ 0$ and $\| \chi f \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } > 0$ . Let $f _ { n } \ = \ \sum _ { j = 1 } ^ { n } a _ { j } \varphi _ { \theta _ { j } ^ { * } } \ \in \ \chi L _ { n } ^ { k }$ satisfy $\| \chi f - f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } \ \leq \ \frac { 1 } { 2 } \| \chi f \| _ { \mathcal { L } ^ { 2 } ( \Omega ) }$ . Since $\operatorname* { s u p } _ { x \in \Omega } \varphi _ { \theta } ( x ) ^ { 2 } \lesssim 1$ uniformly,

$$
n \| a \| _ { 2 } ^ { 2 } \gtrsim \operatorname* { s u p } _ { x \in \Omega } \Big ( n \sum _ { j = 1 } ^ { n } a _ { j } ^ { 2 } \Big ) \Big ( \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \varphi \theta _ { j } ^ { * } \big ( x \big ) ^ { 2 } \Big ) \geq \operatorname* { s u p } _ { x \in \Omega } \Big ( \sum _ { j = 1 } ^ { n } a _ { j } \varphi \theta _ { j } ^ { * } \big ( x \big ) \Big ) ^ { 2 } \gtrsim \| f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } ^ { 2 } \geq \frac { \| \chi f \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } ^ { 2 } } { 4 } .
$$

Let ${ \mathfrak { q } } = \lceil \log _ { 2 } ( C _ { 3 } \underline { { h } } ^ { - 1 } ) \rceil$ where $C _ { 3 }$ is the constant for which (6.19) holds. We estimate the highfrequency part of $f _ { n }$ as follows:

$$
\begin{array} { r l } & { \displaystyle \int _ { | \xi | \geq 2 ^ { \alpha } } | \widehat { f _ { n } } ( \xi ) | ^ { 2 } d \xi \geq \displaystyle \int _ { \mathbb { R } ^ { d } } \Big | \displaystyle \sum _ { q = q + 1 } ^ { \infty } \zeta \big ( \displaystyle \frac { | \xi | } { 2 ^ { q } } \big ) \widehat { f _ { n } } ( \xi ) \Big | ^ { 2 } d \xi \approx \displaystyle \int _ { \mathbb { R } ^ { d } } \displaystyle \sum _ { q = q + 1 } ^ { \infty } \big | \zeta \big ( \displaystyle \frac { | \xi | } { 2 ^ { q } } \big ) \widehat { f _ { n } } ( \xi ) \big | ^ { 2 } d \xi = \displaystyle \int _ { \mathbb { R } ^ { d } } \displaystyle \sum _ { q = q + 1 } ^ { \infty } \big | \widehat { \Delta _ { q , 0 } f _ { n } } ( \xi ) \big | ^ { 2 } d \xi } \\ & { \quad \quad \quad = ( 2 \pi ) ^ { d } \displaystyle \sum _ { q = q + 1 } ^ { \infty } a ^ { \top } Q _ { 0 } ( q ) a \gtrsim 2 ^ { - { \mathfrak { a } } ( 2 k + 1 ) } \| a \| _ { 2 } ^ { 2 } \gtrsim n ^ { - 1 } \underline { { h } } ^ { 2 k + 1 } \| \chi f \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } ^ { 2 } . } \end{array}
$$

On the other hand,

$$
\int _ { | \xi | \geq 2 ^ { \eta } } | \widehat { \chi f } ( \xi ) | ^ { 2 } d \xi \lesssim 2 ^ { - 2 \mathfrak { q r } } \int _ { \mathbb { R } ^ { d } } | \widehat { \chi f } ( \xi ) | ^ { 2 } ( 1 + | \xi | ^ { 2 } ) ^ { r } d \xi = ( 2 \pi ) ^ { d } 2 ^ { - 2 \mathfrak { q r } } \| \chi f \| _ { \mathcal { H } ^ { r } ( \mathbb { R } ^ { d } ) } ^ { 2 } \sim \underline { { h } } ^ { 2 r } \| \chi f \| _ { \mathcal { H } ^ { r } ( \Omega ) } ^ { 2 } .
$$

Since $\textstyle r > { \frac { d + 2 k + 1 } { 2 } }$ , the ratio $\frac { { \underline { { h } } } ^ { 2 r } } { n ^ { - 1 } { \underline { { h } } } ^ { 2 k + 1 } }$ tends to zero uniformly over parameter configurations. Thus there exists $n _ { 0 } .$ , depending on $d , r$ and $\| \chi f \| _ { \mathcal { H } ^ { r } ( \Omega ) } \| \chi f \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } ^ { - 1 }$ , such that, for $n \geq n _ { 0 }$ ，

$$
\begin{array} { l } { \displaystyle { \| \chi f - f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } ^ { 2 } \geq \frac { 1 } { ( 2 \pi ) ^ { d } } \int _ { | \xi | \geq 2 ^ { 4 } } | \widehat { f _ { n } } ( \xi ) - \widehat { \chi f } ( \xi ) | ^ { 2 } d \xi \geq \frac { 1 } { ( 2 \pi ) ^ { d } } \int _ { | \xi | \geq 2 ^ { 4 } } \frac { 1 } { 2 } | \widehat { f _ { n } } ( \xi ) | ^ { 2 } - 2 | \widehat { \chi f } ( \xi ) | ^ { 2 } d \xi } } \\ { \displaystyle \qquad \gtrsim n ^ { - 1 } \frac { h ^ { 2 k + 1 } } { \lambda } \| \chi f \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } ^ { 2 } . } \end{array}
$$

## 6.4 Proof of Theorem 4.5

Proof. Using the Sobolev norms in Definition 3.3, for $t \in \{ s , r \}$ we obtain

$$
\| f _ { n } \| _ { \mathcal H ^ { t } ( \mathbb R ^ { d } ) } ^ { 2 } \asymp \sum _ { q = 0 } ^ { \infty } \| \Delta _ { q , t } f _ { n } \| _ { \mathcal L ^ { 2 } ( \mathbb R ^ { d } ) } ^ { 2 } = \sum _ { q = 0 } ^ { \infty } \Big \| \sum _ { j = 1 } ^ { n } a _ { j } \Delta _ { q , t } \varphi \theta _ { j } ^ { * } \Big \| _ { \mathcal L ^ { 2 } ( \mathbb R ^ { d } ) } ^ { 2 } = \sum _ { q = 0 } ^ { \infty } a ^ { \top } Q _ { t } ( q ) a .\tag{6.22}
$$

Let ${ \mathfrak { q } } = \lceil \log _ { 2 } ( C _ { 3 } \underline { { h } } ^ { - 1 } ) \rceil$ where $C _ { 3 }$ is chosen so that (6.19) holds for both r and s. Then, for $1 \leq q \leq { \mathfrak { q } }$ the low-frequency blocks satisfy

$$
\begin{array} { r l } & { \| \Delta _ { q , r } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } ^ { 2 } = \displaystyle \frac { 1 } { ( 2 \pi ) ^ { d } } \int _ { 2 ^ { q - 1 } \leq | \xi | \leq 2 ^ { q + 1 } } ( 1 + | \xi | ^ { 2 } ) ^ { r } \zeta \big ( \frac { | \xi | } { 2 q } \big ) ^ { 2 } | \widehat { f _ { n } } ( \xi ) | ^ { 2 } d \xi } \\ & { \qquad \lesssim \displaystyle \frac { 2 ^ { 2 ( q + 1 ) ( r - s ) } } { ( 2 \pi ) ^ { d } } \int _ { 2 ^ { q - 1 } \leq | \xi | \leq 2 ^ { q + 1 } } ( 1 + | \xi | ^ { 2 } ) ^ { s } \zeta \big ( \frac { | \xi | } { 2 q } \big ) ^ { 2 } | \widehat { f _ { n } } ( \xi ) | ^ { 2 } d \xi \lesssim \displaystyle \frac { h ^ { - 2 ( r - s ) } \| \Delta _ { q , s } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } ^ { 2 } } { 2 } . } \end{array}
$$

For $q = 0 ,$ , the same argument applies, with the multipliers replaced by $\zeta _ { 0 , r }$ and $\zeta _ { 0 , s }$ .

For $q \geq { \mathfrak { q } } + 1$ , the high-frequency blocks satisfy

$$
\begin{array} { l } { { \displaystyle \sum _ { q = q + 1 } ^ { \infty } a ^ { \top } Q _ { r } ( q ) a \approx \sum _ { q = q + 1 } ^ { \infty } 2 ^ { - q ( 2 ( k - r ) + 1 ) } \| a \| _ { 2 } ^ { 2 } \lesssim 2 ^ { - q ( 2 ( k - r ) + 1 ) } \| a \| _ { 2 } ^ { 2 } \lesssim \underline { { { h } } } ^ { - 2 ( r - s ) } 2 ^ { - \mathsf { q } ( 2 ( k - s ) + 1 ) } \| a \| _ { 2 } ^ { 2 } } } \\ { { \displaystyle \qquad \asymp \underline { { { h } } } ^ { - 2 ( r - s ) } \sum _ { q = q + 1 } ^ { \infty } a ^ { \top } Q _ { s } ( q ) a } } \end{array}
$$

Combining these two estimates with (6.22) completes the proof.

## 6.5 Proof of Theorem 4.6

Define

$$
\mathcal { P } _ { \mathfrak { q } } = \sum _ { q = 0 } ^ { \mathfrak { q } } \Delta _ { q , 0 } , \qquad \mathcal { P } _ { \mathfrak { q } } ^ { c } = I - \mathcal { P } _ { \mathfrak { q } } = \sum _ { q = \mathfrak { q } + 1 } ^ { \infty } \Delta _ { q , 0 } , \qquad \mathfrak { q } > 0 ,
$$

for the low- and high-frequency cutofs, respectively. We first bound $\| \mathcal { P } _ { \mathfrak { q } } ^ { c } f _ { n } \| _ { \ell ^ { 2 } ( X ) }$ for large q. In this subsection, we use the original spherical parameters in $\mathcal { A } _ { \mathcal { X } } \subset \mathbb { S } ^ { d }$

$$
\| \mathcal { P } _ { \boldsymbol { \mathfrak { q } } } ^ { c } f _ { n } \| _ { \ell ^ { 2 } ( X ) } ^ { 2 } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathcal { P } _ { \boldsymbol { \mathfrak { q } } } ^ { c } f _ { n } ( x _ { i } ) ^ { 2 } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \Big ( \sum _ { j = 1 } ^ { n } a _ { j } \mathcal { P } _ { \boldsymbol { \mathfrak { q } } } ^ { c } \varphi _ { \boldsymbol { \theta } _ { j } ^ { * } } ( x _ { i } ) \Big ) ^ { 2 } = \boldsymbol { a } ^ { \top } \boldsymbol { A } _ { \boldsymbol { \mathfrak { q } } } ^ { c } \boldsymbol { a } ,\tag{6.23}
$$

where

$$
( A _ { \mathfrak { q } } ^ { c } ) _ { j , \ell } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathcal { P } _ { \mathfrak { q } } ^ { c } \varphi _ { \theta _ { j } ^ { * } } ( x _ { i } ) \mathcal { P } _ { \mathfrak { q } } ^ { c } \varphi _ { \theta _ { \ell } ^ { * } } ( x _ { i } ) .
$$

Lemma 6.1. Let $d , n , k$ , and $\{ \theta _ { j } ^ { * } \} _ { j = 1 } ^ { n }$ be as in Theorem 4.2. Let $X = \{ x _ { i } \} _ { i = 1 } ^ { m } \subset \Omega$ be quasi-uniform with $h _ { X } \leq \operatorname* { m i n } \{ 1 , \underline { { h } } \}$ , and fix $\begin{array} { r } { \frac { d } { 2 } < \tau < \frac { d + 1 } { 2 } } \end{array}$ . Then, $f o r \ { \mathfrak { q } } \geq \lceil \log _ { 2 } ( C _ { 3 } \underline { { h } } ^ { - 1 } ) \rceil$ and $f _ { n } = \sum _ { j = 1 } ^ { n } a _ { j } \varphi _ { \theta _ { j } ^ { * } }$ ，

$$
\| \mathcal { P } _ { \mathfrak { q } } ^ { c } f _ { n } \| _ { \ell ^ { 2 } ( X ) } \lesssim \| \mathcal { P } _ { \mathfrak { q } } ^ { c } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } + \| a \| _ { 2 } \left\{ h _ { X } ^ { k + \frac { 1 } { 2 } } \left[ 1 + ( h _ { X } \underline { { h } } ^ { - d } ) ^ { \frac { \tau - k - \frac { 1 } { 2 } } { 2 \tau - 1 } } \right] , \quad k \leq \frac { d - 1 } { 2 } , \right.\tag{6.24}
$$

Here $h _ { X } : = \operatorname* { s u p } _ { x \in \Omega } \operatorname* { m i n } _ { 1 < i < m } \left| x - x _ { i } \right|$ , and the implicit constant may depend on $d , k , \tau _ { \mathrm { { i } } }$ , and the quasiuniformity constant.

Proof. Denote by $\begin{array} { r } { K _ { \mathfrak { q } } : = \mathcal { F } ^ { - 1 } ( \sum _ { q = 0 } ^ { \mathfrak { q } } \zeta _ { q , 0 } ) } \end{array}$ the convolution kernel of $\mathcal { P } _ { \mathfrak { q } }$ . Then

$$
\mathcal { P } _ { \mathfrak { q } } ^ { c } f ( x ) = f ( x ) - \int _ { \mathbb { R } ^ { d } } K _ { \mathfrak { q } } ( y ) f ( x - y ) d y , \qquad x \in \mathbb { R } ^ { d } .
$$

$\mathrm { B y }$ the standard Littlewood–Paley kernel estimates (see [4, Chapter 2]), for any ${ \boldsymbol { \alpha } } \in  { \mathbb { N } } _ { 0 } ^ { d }$ and $\beta \geq 0$

$$
\begin{array} { r } { | \partial _ { y } ^ { \alpha } K _ { \mathfrak { q } } ( y ) | \lesssim _ { \alpha , \beta } 2 ^ { \mathfrak { q } ( d + | \alpha | ) } ( 1 + 2 ^ { \mathfrak { q } } | y | ) ^ { - \beta } . } \end{array}
$$

Moreover, since the multiplier of $\mathcal { P } _ { \mathfrak { q } }$ equals one in a neighborhood of the origin,

$$
\int _ { \mathbb { R } ^ { d } } K _ { \mathfrak { q } } ( y ) d y = 1 , \qquad \int _ { \mathbb { R } ^ { d } } y ^ { \alpha } K _ { \mathfrak { q } } ( y ) d y = 0 , \quad | \alpha | \geq 1 .
$$

Consequently, for any $\gamma \geq 0$ , the smooth compactly supported function satisfies

$$
\left| \mathcal { P } _ { \mathfrak { q } } ^ { c } \big [ \chi ( x ) ( w \cdot x + b ) ^ { k } \big ] \right| \lesssim _ { \gamma } 2 ^ { - \mathfrak { q } ( k + \gamma ) } ( 1 + | x | ) ^ { - \gamma } \lesssim _ { \gamma } 2 ^ { - \mathfrak { q } k } ( 1 + 2 ^ { \mathfrak { q } } | w \cdot x + b | ) ^ { - \gamma } ,
$$

uniformly in $\theta ,$ since the parameters are bounded and $\chi$ is fixed.

If $w \cdot x + b < 0$ , then $\varphi _ { \theta } ( x ) = 0$ , and the convolution integral only involves points $x - y$ on the opposite side of the hyperplane. If $w \cdot x + b > 0$ , the same observation applies to $\varphi _ { \theta } - \chi ( \circ ) ( w \cdot \circ + b ) ^ { k }$ In either case, crossing the hyperplane requires $| y | \geq c | w \cdot x + b |$ , and the corresponding truncated power is bounded by a constant times $| y | ^ { \bar { k } }$ . Hence, for $w \cdot x + b \neq 0$

$$
\begin{array} { r l r } {  { | \mathcal { P } _ { \mathfrak { q } } ^ { c } \varphi _ { \theta } ( x ) | \lesssim \Big | \mathcal { P } _ { \mathfrak { q } } ^ { c } \big [ \chi ( x ) ( w \cdot x + b ) ^ { k } \big ] \Big | + \int _ { | y | \geq c | w \cdot x + b | } | K _ { \mathfrak { q } } ( y ) | | y | ^ { k } d y } } \\ & { } & { \lesssim _ { \gamma } 2 ^ { - \mathfrak { q } k } ( 1 + 2 ^ { \mathfrak { q } } | w \cdot x + b | ) ^ { - \gamma } , } \end{array}
$$

where the integral is estimated using the kernel bound with $\beta > d + k + \gamma$ . On the hyperplane $w \cdot x + b = 0 .$ , the same bound follows from $| \varphi _ { \theta } ( x - y ) | \lesssim | y | ^ { k }$ and $\varphi _ { \theta } ( x ) = 0$ when $k \geq 1 ;$ when $k = 0 .$ it follows from the uniform boundedness of $\varphi _ { \theta }$ and $\| K _ { \mathfrak { q } } \| _ { \mathcal { L } ^ { 1 } ( \mathbb { R } ^ { d } ) }$ . Therefore, for every $k \in  { \mathbb { N } } _ { 0 }$

$$
\left| \mathcal { P } _ { \mathfrak { q } } ^ { c } \varphi _ { \theta } ( x ) \right| \lesssim _ { \gamma } 2 ^ { - \mathfrak { q } k } ( 1 + 2 ^ { \mathfrak { q } } | w \cdot x + b | ) ^ { - \gamma } , \qquad x \in \mathbb { R } ^ { d } .\tag{6.25}
$$

Define

$$
\mathcal { Z } _ { j } ( 0 ) : = \big \{ \ell : ~ \rho _ { \pm } ( \theta _ { j } ^ { * } , \theta _ { \ell } ^ { * } ) < \underline { { h } } \big \} , \quad \mathcal { Z } _ { j } ( u ) : = \big \{ \ell : ~ 2 ^ { u - 1 } \underline { { h } } \le \rho _ { \pm } ( \theta _ { j } ^ { * } , \theta _ { \ell } ^ { * } ) < 2 ^ { u } \underline { { h } } \big \} , \qquad u \ge 1 ,
$$

$$
\mathcal { Z } _ { j , \ell } ( 0 ) : = \left. i : \operatorname* { m a x } \left. | w _ { j } ^ { * } \cdot x _ { i } + b _ { j } ^ { * } | , | w _ { \ell } ^ { * } \cdot x _ { i } + b _ { \ell } ^ { * } | \right. < 2 ^ { -  q } \right.\tag{6.26}
$$

$$
\begin{array} { r } { \mathcal { Z } _ { j , \ell } ( v ) : = \big \{ i : 2 ^ { v - \mathfrak { q } - 1 } \leq \operatorname* { m a x } \big \{ | w _ { j } ^ { * } \cdot x _ { i } + b _ { j } ^ { * } | , | w _ { \ell } ^ { * } \cdot x _ { i } + b _ { \ell } ^ { * } | \big \} < 2 ^ { v - \mathfrak { q } } \big \} , \qquad v \geq 1 , } \end{array}
$$

so that, for each $j ,$ dyadic summation yields

$$
\begin{array} { r l } & { \displaystyle \sum _ { \ell = 1 } ^ { n } \left. ( A _ { \mathfrak { q } } ^ { c } ) _ { j , \ell } \right. \leq \frac { 1 } { m } \displaystyle \sum _ { u = 0 } ^ { \lceil \log _ { 2 } \frac { \pi } { \tilde { h } } \rceil } \displaystyle \sum _ { \ell \in \mathbb { Z } _ { j } ( u ) } \displaystyle \sum _ { v = 0 } ^ { \mathfrak { q } + 1 } \displaystyle \sum _ { i \in \mathbb { Z } _ { j , \ell } ( v ) } \left. \mathcal { P } _ { \mathfrak { q } } ^ { c } \varphi _ { \theta _ { j } ^ { * } } ( x _ { i } ) \mathcal { P } _ { \mathfrak { q } } ^ { c } \varphi _ { \theta _ { \ell } ^ { * } } ( x _ { i } ) \right. } \\ & { \qquad \lesssim _ { \gamma } \displaystyle \frac { 2 ^ { - 2 \mathfrak { q } _ { k } } } { m } \displaystyle \sum _ { u = 0 } ^ { \lceil \log _ { 2 } \frac { \pi } { \tilde { h } } \rceil } \displaystyle \sum _ { \ell \in \mathbb { Z } _ { j } ( u ) } \displaystyle \sum _ { v = 0 } ^ { \mathfrak { q } + 1 } \displaystyle \sum _ { i \in \mathbb { Z } _ { j , \ell } ( v ) } ( 1 + 2 ^ { \mathfrak { q } } \vert w _ { j } ^ { * } \cdot x _ { i } + b _ { j } ^ { * } \vert ) ^ { - \gamma } ( 1 + 2 ^ { \mathfrak { q } } \vert w _ { \ell } ^ { * } \cdot x _ { i } + b _ { \ell } ^ { * } \vert ) ^ { - \gamma } } \\ & { \qquad \lesssim \displaystyle \frac { 2 ^ { - 2 \mathfrak { q } _ { k } } } { m } \displaystyle \sum _ { u = 0 } ^ { \lceil \log _ { 2 } \frac { \pi } { \tilde { h } } \rceil } \displaystyle \sum _ { \ell \in \mathbb { Z } _ { j } ( u ) } \displaystyle \sum _ { v = 0 } ^ { \mathfrak { q } + 1 } \displaystyle \sum _ { i \in \mathbb { Z } _ { j , \ell } ( v ) } 2 ^ { - \gamma v } . } \end{array}
$$

Since $| w \cdot x + b | \leq \epsilon$ places x in a belt satisfying dist $( x , \{ w \cdot x + b = 0 \} ) \lesssim \epsilon$ , the points in $\boldsymbol { \mathcal { T } } _ { j , \ell } ( \boldsymbol { v } )$ satisfy

$$
\mathrm { d i s t } ( x , \{ w _ { j } ^ { * } \cdot x + b _ { j } ^ { * } = 0 \} ) + \mathrm { d i s t } ( x , \{ w _ { \ell } ^ { * } \cdot x + b _ { \ell } ^ { * } = 0 \} ) \leq C _ { 4 } 2 ^ { v - 9 }
$$

for some $C _ { 4 }$ . Consequently, the points lie in the set

$$
\Xi _ { j , \ell } ( \boldsymbol { v } ) : = \left\{ \boldsymbol { x } \in \Omega : ~ | \theta _ { j } ^ { * } \cdot \widetilde { \boldsymbol { x } } | \leq C _ { 4 } 2 ^ { v - \mathfrak { q } } , ~ | \theta _ { \ell } ^ { * } \cdot \widetilde { \boldsymbol { x } } | \leq C _ { 4 } 2 ^ { v - \mathfrak { q } } \right\} , \qquad 1 \leq j , \ell \leq n .
$$

For $d \geq 2$ and $\ell \neq j .$ , let $\theta _ { \ell , j }$ be the orthogonal projection of $\theta _ { \ell } ^ { * }$ onto the direction $\theta _ { j } ^ { * }$ and $\theta _ { \ell , j } ^ { \perp } =$ $\theta _ { \ell } ^ { * } - \theta _ { \ell , j } ;$ then $\lvert \theta _ { \ell , j } ^ { \perp } \rvert = \sqrt { 1 - ( \theta _ { j } ^ { * } \cdot \theta _ { \ell } ^ { * } ) ^ { 2 } } \approx \rho _ { \pm } ( \theta _ { j } ^ { * } , \theta _ { \ell } ^ { * } )$ and consequently, there exists $C _ { 4 } ^ { \prime }$ such that

$$
\begin{array} { r l } & { \Xi _ { j , \ell } ( v ) \subset \left\{ x \in \Omega : \ | \theta _ { j } ^ { * } \cdot \widetilde x | \leq C _ { 4 } 2 ^ { v - \mathfrak { q } } , \ | \theta _ { \ell , j } ^ { \perp } \cdot \widetilde x | \leq 2 C _ { 4 } 2 ^ { v - \mathfrak { q } } \right\} } \\ & { \qquad \subset \left\{ x \in \Omega : \ | \theta _ { j } ^ { * } \cdot \widetilde x | \leq C _ { 4 } 2 ^ { v - \mathfrak { q } } , \ \left| \displaystyle \frac { \theta _ { \ell , j } ^ { \perp } } { | \theta _ { \ell , j } ^ { \perp } | } \cdot \widetilde x \right| \leq C _ { 4 } ^ { \prime } \frac { 2 ^ { v - \mathfrak { q } } } { \rho _ { \pm } ( \theta _ { j } ^ { * } , \theta _ { \ell } ^ { * } ) } \right\} . } \end{array}
$$

Since $\theta _ { \ell , j } ^ { \perp } \perp \theta _ { j } ^ { * }$ , the normalized points $\frac { 1 } { \sqrt { | x | ^ { 2 } + 1 } } \binom { x } { 1 } \in \mathbb { S } ^ { d }$ lie in a rectangular cylinder with widths bounded by $C _ { 4 } 2 ^ { v - { \mathfrak { q } } }$ and $C _ { 4 } ^ { \prime } 2 ^ { v - \mathfrak { q } } \rho _ { \pm } ( \theta _ { j } ^ { * } , \dot { \theta _ { \ell } ^ { * } } ) ^ { - 1 }$ . This normalization preserves distances on Ω up to constants, so the number of samples in $\Xi _ { j , \ell } ( v )$ satisfies

$$
\begin{array} { r l } & { \# \mathbb { Z } _ { j , \ell } ( v ) \leq \# \big ( X \cap \Xi _ { j , \ell } ( v ) \big ) \lesssim m ( 2 ^ { ( v - \mathfrak { q } ) } + h _ { X } ) ( 2 ^ { ( v - \mathfrak { q } ) } \rho _ { \pm } ( \theta _ { j } ^ { * } , \theta _ { \ell } ^ { * } ) ^ { - 1 } + h _ { X } ) , } \\ & { \# \mathbb { Z } _ { j , j } ( v ) \leq \# \big ( X \cap \Xi _ { j , j } ( v ) \big ) \lesssim m ( 2 ^ { v - \mathfrak { q } } + h _ { X } ) . } \end{array}\tag{6.27}
$$

For $d \geq 2$ , the sum in (6.26) is bounded by

$$
\begin{array} { r l } { \displaystyle \sum _ { \ell = 1 } ^ { n } \big | ( A _ { \dagger } ^ { c } ) _ { j , \ell } \big | \lesssim _ { \gamma } \frac { 2 ^ { - 2 q k } } { m } \displaystyle \sum _ { u = 1 } ^ { \frac { 2 } { k } } \sum _ { \ell \in \mathcal { Z } _ { j } ( u ) } \sum _ { v = 0 } ^ { q + 1 } 2 ^ { - v } \mathcal { H } Z _ { j , \ell } ( v ) + \frac { 2 ^ { - 2 q k } } { m } \displaystyle \sum _ { v = 0 } ^ { q + 1 } 2 ^ { - v } \mathcal { H } Z _ { j , j } ( v ) } \\ { \lesssim 2 ^ { - 2 q k } \displaystyle \sum _ { u = 1 } ^ { \lceil \log _ { 2 } \frac { 1 } { k } \rceil } \displaystyle \sum _ { \ell \in \mathcal { Z } _ { j } ( u ) } \Big ( \frac { 2 ^ { - 2 q } } { \rho _ { \ell } ( \theta _ { j } ^ { * } , \theta _ { \ell } ^ { * } ) } + \frac { 2 ^ { - q } h _ { X } } { \rho _ { \ell } ( \theta _ { j } ^ { * } , \theta _ { \ell } ^ { * } ) } + h _ { X } ^ { 2 } \Big ) + 2 ^ { - 2 q k } \big ( 2 ^ { - q } + h _ { X } \big ) } \\ { \lesssim 2 ^ { - 2 q k } \displaystyle \sum _ { u = 0 } ^ { \lceil \log _ { 2 } \frac { 1 } { k } \rceil } \displaystyle 2 ^ { u d } \big ( 2 ^ { - 2 q } 2 ^ { - u } \underline { { h } } ^ { - 1 } + 2 ^ { - q } 2 ^ { - u } \underline { { h } } ^ { - 1 } h _ { X } + h _ { X } ^ { 2 } \big ) + 2 ^ { - 2 q k } \big ( 2 ^ { - q } + h _ { X } \big ) } \\  \lesssim 2 ^ { - 2 q k } \displaystyle \sum _ { u = 0 } ^ { \tau - 2 q } \big ( 2 ^ { - 2 q } + h _ { X } ^ { 2 } \big ) \displaystyle \frac { h ^ { - d } } { \rho _ { \ell } ( \theta _ { j } ^ { * } , \theta _ { \ell } ^ { * } ) } + 2 ^  \end{array}\tag{6.28}
$$

For $d = 1$ , the original spherical parameters satisfy $c _ { 3 } \leq | w _ { i } ^ { * } | \leq 1$ . Their antipodal separation implies that the breakpoints $- b _ { j } ^ { * } / w _ { j } ^ { * }$ are separated by $\gtrsim \underline { { h } } .$ . Since $2 ^ { \mathfrak { q } } \underline { { h } } ~ \gtrsim ~ 1$ and the samples are separated by $\gtrsim h _ { X }$ , for $\gamma > 1$ we have

$$
\operatorname* { s u p } _ { x \in \Omega } \sum _ { j = 1 } ^ { n } ( 1 + 2 ^ { \mathfrak { q } } \vert w _ { j } ^ { * } x + b _ { j } ^ { * } \vert ) ^ { - \gamma } \lesssim 1 , \qquad \operatorname* { s u p } _ { 1 \leq j \leq n } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } ( 1 + 2 ^ { \mathfrak { q } } \vert w _ { j } ^ { * } x _ { i } + b _ { j } ^ { * } \vert ) ^ { - \gamma } \lesssim 2 ^ { - \mathfrak { q } } + h _ { X } .
$$

Applying weighted Cauchy–Schwarz to $\mathcal { P } _ { \mathfrak { q } } ^ { c } f _ { n } ( x _ { i } ) = \sum _ { j = 1 } ^ { n } a _ { j } \mathcal { P } _ { \mathfrak { q } } ^ { c } \varphi _ { \theta _ { j } ^ { * } } ( x _ { i } )$ and using (6.25), we obtain

$$
\begin{array} { r } { \| \mathcal { P } _ { \mathfrak { q } } ^ { c } f _ { n } \| _ { \ell ^ { 2 } ( X ) } ^ { 2 } \lesssim 2 ^ { - 2 \mathfrak { q } k } ( 2 ^ { - \mathfrak { q } } + h _ { X } ) \| a \| _ { 2 } ^ { 2 } . } \end{array}
$$

Thus, the estimate below also holds when $d = 1$

Substituting the corresponding estimates into (6.23) gives, for $\mathfrak { q } \geq \lceil \log _ { 2 } ( C _ { 3 } \underline { { h } } ^ { - 1 } ) \rceil$

$$
\| \mathcal { P } _ { \mathfrak { q } } ^ { c } f _ { n } \| _ { \ell ^ { 2 } ( X ) } ^ { 2 } \lesssim 2 ^ { - 2 \mathfrak { q } k } \left[ \Big ( 2 ^ { - 2 \mathfrak { q } } + h _ { X } ^ { 2 } \Big ) \underline { { h } } ^ { - d } + 2 ^ { - \mathfrak { q } } + h _ { X } \right] \| a \| _ { 2 } ^ { 2 } .\tag{6.29}
$$

Applying the sampling inequality in [2] with $\tau > \frac { d } { 2 }$ to $\Delta _ { q , 0 } f _ { n }$ gives

$$
\begin{array} { r l } { \| \Delta _ { q , 0 } f _ { n } \| _ { \ell ^ { 2 } ( X ) } \lesssim \| \Delta _ { q , 0 } f _ { n } \| _ { \ell ^ { 2 } ( \mathbb { R } ^ { d } ) } + h _ { X } ^ { \tau } \| \Delta _ { q , 0 } f _ { n } \| _ { \mathcal { H } ^ { \tau } ( \mathbb { R } ^ { d } ) } } & { } \\ { \lesssim \| \Delta _ { q , 0 } f _ { n } \| _ { \ell ^ { 2 } ( \mathbb { R } ^ { d } ) } + h _ { X } ^ { \tau } 2 ^ { q ( \tau - k - \frac { 1 } { 2 } ) } \| a \| _ { 2 } , } & { \quad q \geq \lceil \mathrm { l o g } _ { 2 } ( C _ { 3 } \underline { { h } } ^ { - 1 } ) \rceil . } \end{array}\tag{6.30}
$$

By the block upper estimate and a diagonally dominant block whose frequency support lies above that of the multiplier of $\mathcal { P } _ { \mathfrak { q } }$ ,

$$
\sum _ { q = \mathfrak { q } + 1 } ^ { \infty } \| \Delta _ { q , 0 } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \lesssim 2 ^ { - \mathfrak { q } ( k + \frac { 1 } { 2 } ) } \| a \| _ { 2 } \lesssim \| \mathcal { P } _ { \mathfrak { q } } ^ { c } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } .
$$

Therefore, for ${ \mathfrak { p } } \geq { \mathfrak { q } }$

$$
\begin{array} { r l } & { \quad \displaystyle \| \mathcal { P } _ { \mathfrak { q } } ^ { c } f _ { n } \| _ { \ell ^ { 2 } ( X ) } = \Big \| \displaystyle \sum _ { q = 0 + 1 } ^ { \mathfrak { p } } \Delta _ { q , 0 } f _ { n } + \mathcal { P } _ { \mathfrak { p } } ^ { c } f _ { n } \Big \| _ { \ell ^ { 2 } ( X ) } } \\ & { \lesssim \| \mathcal { P } _ { \mathfrak { q } } ^ { c } f _ { n } \| _ { \ell ^ { 2 } ( \mathbb { R } ^ { d } ) } + \left[ h _ { X } ^ { r } \displaystyle \sum _ { q = 0 + 1 } ^ { \mathfrak { p } } 2 ^ { q ( \tau - k - \frac { 1 } { 2 } ) } + 2 ^ { - \mathfrak { p } k } \left( ( 2 ^ { - \mathfrak { p } } + h _ { X } ) \frac { h ^ { - \frac { d } { 2 } } + ( 2 ^ { - \mathfrak { p } } + h _ { X } ) ^ { \frac { 1 } { 2 } } } { \mathfrak { q } } \right) \right] \| a \| _ { 2 } . } \end{array}\tag{6.31}
$$

If $k \leq \frac { d - 1 } { 2 }$ , then $\tau > k + 1 / 2$ . Set ${ \mathfrak { p } } = \left\lceil \log _ { 2 } \left( h _ { X } ^ { - 1 } \operatorname* { m a x } \left\{ 1 , ( h _ { X } { \underline { { h } } } ^ { - d } ) ^ { \frac { 1 } { 2 \tau - 1 } } \right\} \right) \right\rceil$ , so that $2 ^ { - 9 } \lesssim h _ { X }$ and

$$
\begin{array} { r } { h _ { X } ^ { \tau } 2 ^ { \mathfrak { p } ( \tau - k - \frac { 1 } { 2 } ) } + 2 ^ { - \mathfrak { p } k } \left( h _ { X } \underline { { h } } ^ { - \frac { d } { 2 } } + h _ { X } ^ { \frac { 1 } { 2 } } \right) \lesssim h _ { X } ^ { k + \frac { 1 } { 2 } } \left[ 1 + ( h _ { X } \underline { { h } } ^ { - d } ) ^ { \frac { \tau - k - \frac { 1 } { 2 } } { 2 \tau - 1 } } \right] . } \end{array}
$$

If ${ \mathfrak { q } } \leq { \mathfrak { p } }$ , substitution into (6.31) proves the first case of (6.24). If ${ \mathfrak { q } } > { \mathfrak { p } }$ , apply (6.29) directly; its right-hand side is nonincreasing in ${ \mathfrak { q } } ,$ so the same bound follows.

For $\textstyle k > { \frac { d - 1 } { 2 } }$ , we take $\begin{array} { r } { \frac { d } { 2 } < \tau < } \end{array}$ < min $\textstyle \{ k + { \frac { 1 } { 2 } } , { \frac { d + 1 } { 2 } } \}$ . Letting ${ \mathfrak { p } } \to \infty$ in (6.31), we obtain

$$
\begin{array} { r } { \| \mathcal { P } _ { \mathfrak { q } } ^ { c } f _ { n } \| _ { \ell ^ { 2 } ( X ) } \lesssim \| \mathcal { P } _ { \mathfrak { q } } ^ { c } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } + h _ { X } ^ { \tau } 2 ^ { \mathfrak { q } ( \tau - k - \frac { 1 } { 2 } ) } \| a \| _ { 2 } = \| \mathcal { P } _ { \mathfrak { q } } ^ { c } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } + 2 ^ { - \mathfrak { q } ( k + \frac { 1 } { 2 } ) } ( h _ { X } 2 ^ { \mathfrak { q } } ) ^ { \tau } \| a \| _ { 2 } . } \end{array}
$$

This completes the proof.

Proof of Theorem $4 . 6 .$ We first establish the sampling estimates for the low-frequency part. If supp $\cdot \widehat { g } \subset B ( 0 , 2 ^ { \mathfrak { p } + 1 } )$ and $2 ^ { \mathfrak { p } } \leq h _ { X } ^ { - 1 }$ , then

$$
\| g \| _ { \ell ^ { 2 } ( { \cal X } ) } \lesssim \| g \| _ { \mathcal L ^ { 2 } ( { \mathbb R } ^ { d } ) } , \qquad \| g \| _ { \mathcal L ^ { 2 } ( \Omega ) } \lesssim \| g \| _ { \ell ^ { 2 } ( { \cal X } ) } + h _ { { \cal X } } 2 ^ { \mathfrak p } \| g \| _ { \mathcal L ^ { 2 } ( { \mathbb R } ^ { d } ) } .\tag{6.32}
$$

Indeed, let $\Omega _ { i }$ be the Voronoi cells of X in Ω. Then $| x - x _ { i } | \leq h _ { X }$ for $x \in \Omega _ { i }$ <sub>i</sub> and $| \Omega _ { i } | \lesssim h _ { X } ^ { d } \lesssim m ^ { - 1 }$ A smooth reproducing kernel and the mean value theorem give

$$
| g ( x ) - g ( x _ { i } ) | \lesssim h _ { X } 2 ^ { \mathfrak { p } } \int _ { \mathbb { R } ^ { d } } \frac { 2 ^ { \mathfrak { p } d } | g ( y ) | } { ( 1 + 2 ^ { \mathfrak { p } } | x - y | ) ^ { d + 1 } } d y , \qquad x \in \Omega _ { i } .
$$

Young’s inequality proves the second estimate in (6.32). For the first estimate, the same kernel and the separation of X give

$$
| g ( x _ { i } ) | ^ { 2 } \lesssim \int _ { \mathbb { R } ^ { d } } \frac { 2 ^ { \mathfrak { p } d } | g ( y ) | ^ { 2 } } { ( 1 + 2 ^ { \mathfrak { p } } | x _ { i } - y | ) ^ { d + 1 } } d y ,
$$

and the kernel satisfies

$$
\operatorname* { s u p } _ { y \in \mathbb { R } ^ { d } } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { 2 ^ { \mathfrak { p } d } } { ( 1 + 2 ^ { \mathfrak { p } } | x _ { i } - y | ) ^ { d + 1 } } \lesssim 1 .
$$

Let $f _ { n } = \sum _ { j = 1 } ^ { n } a _ { j } \varphi _ { \theta _ { j } ^ { * } }$ . Choose a suficiently small constant $c _ { 6 } > 0$ and take ${ \mathfrak { p } } = \lfloor \log _ { 2 } ( c _ { 6 } h _ { X } ^ { - 1 } ) \rfloor$ . By increasing $C _ { 1 }$ , we may assume

$$
{ \mathfrak { p } } \geq \lceil \log _ { 2 } ( C _ { 3 } { \underline { { h } } } ^ { - 1 } ) \rceil + 2 , \qquad h _ { X } 2 ^ { \mathfrak { p } } \leq c _ { 6 } , \qquad 2 ^ { - \mathfrak { p } } \approx c _ { 6 } ^ { - 1 } h _ { X } .\tag{6.33}
$$

Then, by Lemma 6.1,

$$
\| \mathcal { P } _ { \mathfrak { p } } ^ { c } f _ { n } \| _ { \ell ^ { 2 } ( X ) } \lesssim \| a \| _ { 2 } \left\{ h _ { X } ^ { k + \frac { 1 } { 2 } } \left[ 1 + \big ( h _ { X } \underline { { h } } ^ { - d } \big ) ^ { \frac { \tau - k - \frac { 1 } { 2 } } { 2 \tau - 1 } } \right] , \ : \ : \ : k \leq \frac { d - 1 } { 2 } , \right.
$$

On the other hand, by (4.7) and Theorem 4.3, $\| f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \gtrsim \underline { { h } } ^ { k + \frac { 1 } { 2 } } \| a \| _ { 2 }$ . Thus, for $2 ^ { \mathfrak { p } } \gtrsim \underline { { h } } ^ { - 1 }$ with a suficiently large implicit constant, we have

$$
\begin{array} { r } { \| f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \lesssim \| \mathcal { P } _ { \mathfrak { p } } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \gtrsim \underline { { h } } ^ { k + \frac { 1 } { 2 } } \| a \| _ { 2 } . } \end{array}\tag{6.34}
$$

Since $m ^ { \frac { 1 } { d } } \approx 2 ^ { \mathfrak { p } }$ , we can choose a suficiently large constant $C _ { 1 }$ such that

$$
m \geq C _ { 1 } \left\{ \frac { h ^ { - d \left( 1 + \frac { ( d - 1 ) ( \tau - k - \frac { 1 } { 2 } ) } { 2 \tau ( k + 1 ) - 2 k - 1 } \right) } } { \Delta } , \quad k \leq \frac { d - 1 } { 2 } , \right.
$$

implies $\| \mathcal { P } _ { \mathfrak { p } } ^ { c } f _ { n } \| _ { \ell ^ { 2 } ( X ) } \lesssim \| \mathcal { P } _ { \mathfrak { p } } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) }$ and (6.32) yields

$$
\| f _ { n } \| _ { \ell ^ { 2 } ( X ) } \leq \| \mathcal { P } _ { \mathfrak { p } } f _ { n } \| _ { \ell ^ { 2 } ( X ) } + \| \mathcal { P } _ { \mathfrak { p } } ^ { c } f _ { n } \| _ { \ell ^ { 2 } ( X ) } \lesssim \| \mathcal { P } _ { \mathfrak { p } } f _ { n } \| _ { \ell ^ { 2 } ( \mathbb { R } ^ { d } ) } \lesssim \| f _ { n } \| _ { \ell ^ { 2 } ( \mathbb { R } ^ { d } ) } .
$$

For the reverse inequality, the high-frequency block estimate and the coeficient lower bound give

$$
\begin{array} { r } { \| \mathcal { P } _ { \mathfrak { p } } ^ { c } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \lesssim 2 ^ { - { \mathfrak { p } } ( k + \frac { 1 } { 2 } ) } \| a \| _ { 2 } \lesssim ( 2 ^ { \mathfrak { p } } \underline { { h } } ) ^ { - ( k + \frac { 1 } { 2 } ) } \| f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } . } \end{array}
$$

Increasing $C _ { 1 }$ if necessary, we may assume that the last expression is at most $\begin{array} { r l } { { \frac { 1 } { 4 } } \left\| f _ { n } \right\| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } } & { } \end{array}$ . Since supp $f _ { n } ) \subset \Omega$ , the triangle inequality yields

$$
\begin{array} { r l } & { \| \mathcal { P } _ { \mathfrak { p } } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } \geq \| f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } - \| \mathcal { P } _ { \mathfrak { p } } ^ { c } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } \geq \frac { 3 } { 4 } \| f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } , } \\ & { \| \mathcal { P } _ { \mathfrak { p } } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \leq \| f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } + \| \mathcal { P } _ { \mathfrak { p } } ^ { c } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \leq \frac { 5 } { 4 } \| f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } . } \end{array}
$$

Consequently, $\begin{array} { r } { \| \mathcal { P } _ { \mathfrak { p } } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \leq \frac { 5 } { 3 } \| \mathcal { P } _ { \mathfrak { p } } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } } \end{array}$ . Taking $c _ { 6 }$ suficiently small and then increasing $C _ { 1 }$ if necessary, we can therefore absorb the remainder in (6.32) to obtain

$$
\frac { 1 } { 2 } \| \mathcal { P } _ { \mathfrak { p } } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } \lesssim \| \mathcal { P } _ { \mathfrak { p } } f _ { n } \| _ { \ell ^ { 2 } ( X ) } \leq \| f _ { n } \| _ { \ell ^ { 2 } ( X ) } + \| \mathcal { P } _ { \mathfrak { p } } ^ { c } f _ { n } \| _ { \ell ^ { 2 } ( X ) } .
$$

It follows that there exists $c _ { 7 } > 0$ such that

$$
c _ { 7 } \| f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } \leq \| f _ { n } \| _ { \ell ^ { 2 } ( X ) } + \| \mathcal { P } _ { \mathfrak { p } } ^ { c } f _ { n } \| _ { \ell ^ { 2 } ( X ) } .
$$

Increasing $C _ { 1 }$ further so that $\begin{array} { r } { \| \mathcal { P } _ { \mathfrak { p } } ^ { c } f _ { n } \| _ { \ell ^ { 2 } ( X ) } \leq \frac { c _ { 7 } } { 2 } \| f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } } \end{array}$ , we obtain

$$
{ \frac { c _ { 7 } } { 2 } } \| f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } \leq \| f _ { n } \| _ { \ell ^ { 2 } ( X ) } ,
$$

as desired.

## 6.6 Proof of Theorem 4.7

The proof uses high-frequency upper estimates for parameters on the full sphere. Unlike the spectral lower bounds, these estimates do not require the singular hyperplanes to intersect $U$ . We retain the Fourier cutofs $\mathcal { P } _ { \mathfrak { q } }$ and $\mathcal { P } _ { \mathfrak { q } } ^ { c }$ defined in Section 6.5.

Lemma 6.2. Let $L _ { n } ^ { k } = L _ { n } ^ { k } ( \Theta _ { n } )$ and ${ \underline { { h } } } > 0$ be as in Theorem $4 . 7 ,$ let $\textstyle 0 \leq t < k + { \frac { 1 } { 2 } }$ and $q \geq 1$ Then for any $v _ { n } = \sum _ { j = 1 } ^ { n } a _ { j } \varphi _ { \theta _ { j } ^ { * } } \in L _ { n } ^ { k }$

$$
\begin{array} { r } { \| \Delta _ { q , t } v _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \lesssim 2 ^ { - q ( k + \frac { 1 } { 2 } - t ) } \left( 1 + ( 2 ^ { q } \underline { { h } } ) ^ { - d } \right) ^ { \frac { 1 } { 2 } } \| a \| _ { 2 } . } \end{array}\tag{6.35}
$$

Consequently, for ${ \mathfrak { q } } \geq 1$ ，

$$
\begin{array} { r } { \| \mathcal { P } _ { \mathfrak { q } } ^ { c } v _ { n } \| _ { \mathcal { H } ^ { t } ( \mathbb { R } ^ { d } ) } \lesssim 2 ^ { - \mathfrak { q } ( k + \frac { 1 } { 2 } - t ) } \left( 1 + ( 2 ^ { \mathfrak { q } } \underline { { h } } ) ^ { - d } \right) ^ { \frac { 1 } { 2 } } \| a \| _ { 2 } . } \end{array}\tag{6.36}
$$

Proof. As in the preceding proofs, assume that $\Omega \subset \mathbb { B } ^ { d }$ . Write $v _ { n } = v _ { n } ^ { ( 0 ) } + v _ { n } ^ { ( 1 ) }$ , where $v _ { n } ^ { ( 0 ) }$ contains the terms with $| b _ { j } ^ { * } | \leq | w _ { j } ^ { * } |$ . For these parameters, $\begin{array} { r } { | w _ { j } ^ { * } | \geq \frac { 1 } { \sqrt { 2 } } } \end{array}$ and $\frac { | b _ { j } ^ { * } | } { | w _ { j } ^ { * } | } \leq 1$ . The single-neuron upper bound and the of-diagonal estimate used to prove (6.15) therefore remain valid, since their proofs require neither intersection with $U$ nor positivity of $\chi$ on the singular hyperplane. The same matrix upper-bound argument gives

$$
\begin{array} { r } { \| \Delta _ { q , t } v _ { n } ^ { ( 0 ) } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } ^ { 2 } \lesssim 2 ^ { - q ( 2 ( k - t ) + 1 ) } \left( 1 + ( 2 ^ { q } \underline { { h } } ) ^ { - d } \right) \| a \| _ { 2 } ^ { 2 } . } \end{array}
$$

For the remaining parameters, each neuron is either zero or a polynomial on $\mathbb { B } ^ { d }$ . After multiplication by $\chi ,$ these functions have uniformly bounded $\mathcal { H } ^ { \gamma } ( \mathbb { R } ^ { d } )$ norms for every fixed integer $\gamma .$ . Choose $\begin{array} { r } { \gamma > k + \frac { d + 1 } { 2 } } \end{array}$ . Antipodal packing gives $n \lesssim \underline { { h } } ^ { - d }$ , and hence

$$
\begin{array} { r } { \| \Delta _ { q , t } v _ { n } ^ { ( 1 ) } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \lesssim \sqrt { n } 2 ^ { - q ( \gamma - t ) } \| a \| _ { 2 } \lesssim 2 ^ { - q ( k + \frac { 1 } { 2 } - t ) } ( 2 ^ { q } \underline { { h } } ) ^ { - \frac { d } { 2 } } \| a \| _ { 2 } . } \end{array}
$$

Combining these estimates proves (6.35). Finite overlap and geometric summation give

$$
\| \mathcal { P } _ { \mathfrak { q } } ^ { c } v _ { n } \| _ { \mathcal { H } ^ { t } ( \mathbb { R } ^ { d } ) } ^ { 2 } \lesssim \sum _ { q = \mathfrak { q } + 1 } ^ { \infty } 2 ^ { - q ( 2 ( k - t ) + 1 ) } \left( 1 + ( 2 ^ { q } \underline { { h } } ) ^ { - d } \right) \| a \| _ { 2 } ^ { 2 } \lesssim 2 ^ { - { \mathfrak { q } } ( 2 ( k - t ) + 1 ) } \left( 1 + ( 2 ^ { \mathfrak { q } } \underline { { h } } ) ^ { - d } \right) \| a \| _ { 2 } ^ { 2 } ,
$$

which proves (6.36).

Lemma 6.3. Let $X \subset \Omega$ be quasi-uniform with fill distance $h _ { X }$ , let $2 \leq p < \infty$ , and let $\begin{array} { r } { \frac { d } { p } < \tau < } \end{array}$ $\textstyle { \frac { d } { p } } + 1$ . For suficiently small $h _ { X }$

$$
\begin{array} { r } { \| v \| _ { \ell ^ { 2 } ( X ) } \lesssim \| v \| _ { \mathcal L ^ { p } ( \mathbb R ^ { d } ) } + h _ { X } ^ { \tau } \| v \| _ { \mathcal W ^ { \tau , p } ( \mathbb R ^ { d } ) } , \qquad v \in \mathcal W ^ { \tau , p } ( \mathbb R ^ { d } ) . } \end{array}\tag{6.37}
$$

Proof. Rescaling Sobolev embedding on balls of radius comparable to $h _ { X }$ centred at the samples and summing with bounded overlap gives

$$
\frac { 1 } { m } \sum _ { i = 1 } ^ { m } | v ( x _ { i } ) | ^ { p } \lesssim \| v \| _ { \mathcal { L } ^ { p } ( \mathbb { R } ^ { d } ) } ^ { p } + h _ { X } ^ { \tau p } \| v \| _ { \mathcal { W } ^ { \tau , p } ( \mathbb { R } ^ { d } ) } ^ { p } ,
$$

where m $\mathit { 1 h _ { X } ^ { d } } \approx 1$ follows from quasi-uniformity. Finally, since $p \geq 2 .$ , the standard norm bound $\| \circ \| _ { \ell ^ { 2 } ( X ) } \leq \| \circ \| _ { \ell ^ { p } ( X ) }$ completes the proof. □

Proof of Theorem $4 . 7 .$ For $\begin{array} { r } { r > \frac { d } { 2 } } \end{array}$ , set $p = 2 ;$ otherwise, let $2 < p <$ ∞ satisfy $\begin{array} { r } { \frac { d } { p } < r \leq \frac { d } { 2 } } \end{array}$ . In this proof, we take the Fourier cutof scale

$$
\varepsilon : = h ^ { \frac { 2 k + 1 } { d + 2 k + 1 } } \underline { { h } } ^ { \frac { d } { d + 2 k + 1 } } \leq h ,\tag{6.38}
$$

and choose ${ \mathfrak { q } } ~ \geq ~ 1$ such that $2 ^ { - \mathfrak { q } } \ \lesssim \ \varepsilon .$ . By quasi-uniformity of X and the sampling condition $m \geq C _ { 5 } \varepsilon ^ { - d }$ , we have $h _ { X } \gtrsim m ^ { - \frac { 1 } { d } } \lesssim C _ { 5 } ^ { - \frac { 1 } { d } } \varepsilon$ , and hence $h _ { X } 2 ^ { \mathfrak { q } } \lesssim C _ { 5 } ^ { - \frac { 1 } { d } }$ . By increasing $C _ { 5 } .$ we may therefore assume that the sampling estimates apply and $h _ { X } 2 ^ { \mathfrak { q } }$ is suficiently small. The identity

$$
\varepsilon ^ { 2 k + 1 } \left( 1 + \left( { \frac { \varepsilon } { \underline { { h } } } } \right) ^ { d } \right) = \varepsilon ^ { 2 k + 1 } + h ^ { 2 k + 1 } \lesssim h ^ { 2 k + 1 }
$$

and Lemma 6.2 imply for $\textstyle 0 \leq t < k + { \frac { 1 } { 2 } }$ ，

$$
\Big \| \sum _ { j = 1 } ^ { n } a _ { j } \mathcal { P } _ { \ q } ^ { c } \varphi _ { \theta _ { j } ^ { * } } \Big \| _ { \mathcal { H } ^ { t } ( \mathbb { R } ^ { d } ) } \lesssim \varepsilon ^ { - t } h ^ { k + \frac { 1 } { 2 } } \| a \| _ { 2 } , \quad a \in \mathbb { R } ^ { n } .
$$

Choose $\begin{array} { r } { \frac { d } { 2 } < \tau _ { 2 } < \operatorname* { m i n } \{ k + \frac { 1 } { 2 } , \frac { d } { 2 } + 1 \} } \end{array}$ . Applying Lemma 6.3 with $p = 2$ , we obtain

$$
\Big \| \sum _ { j = 1 } ^ { n } a _ { j } \mathcal { P } _ { \mathfrak { q } } ^ { c } \varphi _ { \theta _ { j } ^ { * } } \Big \| _ { \ell ^ { 2 } ( X ) } \lesssim \big ( 1 + ( h _ { X } \varepsilon ^ { - 1 } ) ^ { \tau _ { 2 } } \big ) h ^ { k + \frac { 1 } { 2 } } \| a \| _ { 2 } \lesssim h ^ { k + \frac { 1 } { 2 } } \| a \| _ { 2 } , \quad a \in \mathbb { R } ^ { n } .\tag{6.39}
$$

Similarly, choose $\begin{array} { r } { \frac { d } { p } < \tau _ { p } < \operatorname* { m i n } \{ r , \frac { d } { p } + 1 \} } \end{array}$ . Lemma 6.3 and the standard Fourier cutof estimates yield

$$
\begin{array} { r l } & { \| \mathcal { P } _ { \mathfrak { q } } ^ { c } ( \chi f ) \| _ { \ell ^ { 2 } ( X ) } \lesssim \| \mathcal { P } _ { \mathfrak { q } } ^ { c } ( \chi f ) \| _ { \mathcal { L } ^ { p } ( \mathbb { R } ^ { d } ) } + h _ { X } ^ { \tau _ { p } } \| \mathcal { P } _ { \mathfrak { q } } ^ { c } ( \chi f ) \| _ { \mathcal { W } ^ { \tau _ { p } , p } ( \mathbb { R } ^ { d } ) } } \\ & { \qquad \lesssim \left( \varepsilon ^ { r } + h _ { X } ^ { \tau _ { p } } \varepsilon ^ { r - \tau _ { p } } \right) \| f \| _ { \mathcal { W } ^ { r , p } ( \Omega ) } \lesssim h ^ { r } \| f \| _ { \mathcal { W } ^ { r , p } ( \Omega ) } . } \end{array}\tag{6.40}
$$

By [67, Theorem 2.2], there exists $g _ { n } = \sum _ { j = 1 } ^ { n } a _ { j } \sigma _ { k } ( w _ { j } ^ { * } \cdot x + b _ { j } ^ { * } )$ such that

$$
\| g _ { n } - f \| _ { \mathcal L ^ { 2 } ( \Omega ) } \lesssim n ^ { - \frac { r } { d } } \| f \| _ { \mathcal H ^ { r } ( \Omega ) } , \qquad \| a \| _ { 2 } \lesssim h ^ { r - k - \frac { 1 } { 2 } } \| f \| _ { \mathcal H ^ { r } ( \Omega ) } .\tag{6.41}
$$

Taking $v _ { n } = \chi g _ { n }$ , then

$$
\| v _ { n } - \chi f \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } = \| \chi ( g _ { n } - f ) \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } \lesssim n ^ { - \frac { r } { d } } \| f \| _ { \mathcal { H } ^ { r } ( \Omega ) }\tag{6.42}
$$

and consequently

$$
\begin{array} { r l } & { \| \chi f - v _ { n } \| _ { \ell ^ { 2 } ( X ) } \leq \| \mathcal { P } _ { \mathfrak { q } } \bigl ( \chi f - v _ { n } \bigr ) \| _ { \ell ^ { 2 } ( X ) } + \| \mathcal { P } _ { \mathfrak { q } } ^ { c } \bigl ( \chi f \bigr ) \| _ { \ell ^ { 2 } ( X ) } + \| \mathcal { P } _ { \mathfrak { q } } ^ { c } v _ { n } \| _ { \ell ^ { 2 } ( X ) } } \\ & { \qquad \lesssim \| \chi f - v _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } + h ^ { r } \| f \| _ { \mathcal { W } ^ { r , p } ( \Omega ) } + h ^ { k + \frac { 1 } { 2 } } \| a \| _ { 2 } \lesssim h ^ { r } \| f \| _ { \mathcal { W } ^ { r , p } ( \Omega ) } . } \end{array}
$$

where the first term is follows by the sampling estimate for the low-frequency functions (6.32), second by (6.40), and third by (6.39). Together with the coeficient estimate in (6.41),

$$
\begin{array} { r } { \| \chi f - v _ { n } \| _ { \ell ^ { 2 } ( X ) } ^ { 2 } + h ^ { 2 k + 1 } \| a \| _ { 2 } ^ { 2 } \lesssim h ^ { 2 r } \| f \| _ { \mathcal { W } ^ { r , p } ( \Omega ) } ^ { 2 } . } \end{array}
$$

Since $f _ { n , m }$ is the minimizer of (4.16),

$$
\begin{array} { r } { \| \chi f - f _ { n , m } \| _ { \ell ^ { 2 } ( X ) } ^ { 2 } + h ^ { 2 k + 1 } \| \alpha \| _ { 2 } ^ { 2 } \lesssim h ^ { 2 r } \| f \| _ { \mathcal { W } ^ { r , p } ( \Omega ) } ^ { 2 } . } \end{array}\tag{6.43}
$$

In particular,

$$
\| \pmb { \alpha } \| _ { 2 } \lesssim h ^ { r - k - \frac { 1 } { 2 } } \| f \| _ { \mathcal { W } ^ { r , p } ( \Omega ) }\tag{6.44}
$$

Now we are ready to consider $\| \chi f - f _ { n , m } \| _ { \mathcal { L } ^ { 2 } ( \Omega ) }$ . Since $\chi f - f _ { n , m }$ is supported in supp $\chi \subset \Omega$ the bandlimited lower sampling estimate in (6.32) gives

$$
\begin{array} { r l } & { \quad \| \chi f - f _ { n , m } \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } \leq \| \mathcal { P } _ { \mathfrak { q } } ( \chi f - f _ { n , m } ) \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } + \| \mathcal { P } _ { \mathfrak { q } } ^ { c } ( \chi f - f _ { n , m } ) \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } } \\ & { \lesssim \| \mathcal { P } _ { \mathfrak { q } } ( \chi f - f _ { n , m } ) \| _ { \ell ^ { 2 } ( X ) } + h _ { X } 2 ^ { \mathfrak { q } } \| \mathcal { P } _ { \mathfrak { q } } ( \chi f - f _ { n , m } ) \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } + \| \mathcal { P } _ { \mathfrak { q } } ^ { c } ( \chi f - f _ { n , m } ) \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } } \\ & { \lesssim \| \mathcal { P } _ { \mathfrak { q } } ( \chi f - f _ { n , m } ) \| _ { \ell ^ { 2 } ( X ) } + h _ { X } 2 ^ { \mathfrak { q } } \| \chi f - f _ { n , m } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } . } \end{array}
$$

By increasing $C _ { 5 }$ , the last term can be absorbed, giving $\lVert \chi f - f _ { n , m } \rVert _ { \mathcal { L } ^ { 2 } ( \Omega ) } \lesssim \lVert \mathcal { P } _ { \mathfrak { q } } ( \chi f - f _ { n , m } ) \rVert _ { \ell ^ { 2 } ( X ) }$ Now by (6.43), (6.40), (6.39), and (6.44),

$$
\begin{array} { r } { \| \chi f - f _ { n , m } \| _ { \mathcal { L } ^ { 2 } ( \Omega ) } \lesssim \| \mathcal { P } _ { \mathfrak { q } } ( \chi f - f _ { n , m } ) \| _ { \ell ^ { 2 } ( X ) } \leq \| \chi f - f _ { n , m } \| _ { \ell ^ { 2 } ( X ) } + \| \mathcal { P } _ { \mathfrak { q } } ^ { c } ( \chi f - f _ { n , m } ) \| _ { \ell ^ { 2 } ( X ) } } \\ { \leq \| \chi f - f _ { n , m } \| _ { \ell ^ { 2 } ( X ) } + \| \mathcal { P } _ { \mathfrak { q } } ^ { c } \chi f \| _ { \ell ^ { 2 } ( X ) } + \| \mathcal { P } _ { \mathfrak { q } } ^ { c } f _ { n , m } \| _ { \ell ^ { 2 } ( X ) } \lesssim h ^ { r } \| f \| _ { \mathcal { W } ^ { r , p } ( \Omega ) } . } \end{array}
$$

Finally, the standard low-frequency bound and Lemma 6.2 give

$$
\begin{array} { r l } { \| \chi f - f _ { n , m } \| _ { \mathcal H ^ { s } ( \mathbb R ^ { d } ) } \leq \| \mathcal P _ { \mathfrak q } ( \chi f - f _ { n , m } ) \| _ { \mathcal H ^ { s } ( \mathbb R ^ { d } ) } + \| \mathcal P _ { \mathfrak q } ^ { c } ( \chi f ) \| _ { \mathcal H ^ { s } ( \mathbb R ^ { d } ) } + \| \mathcal P _ { \mathfrak q } ^ { c } f _ { n , m } \| _ { \mathcal H ^ { s } ( \mathbb R ^ { d } ) } } & { } \\ { \lesssim \varepsilon ^ { - s } \| \chi f - f _ { n , m } \| _ { \mathcal L ^ { 2 } ( \mathbb R ^ { d } ) } + \varepsilon ^ { r - s } \| f \| _ { \mathcal H ^ { r } ( \Omega ) } + \varepsilon ^ { - s } h ^ { k + \frac 1 2 } \| \alpha \| _ { 2 } } & { } \\ { \lesssim h ^ { r } \varepsilon ^ { - s } \| f \| _ { \mathcal W ^ { r , p } ( \Omega ) } = h ^ { r - s } \Big ( \frac { h } { h } \Big ) ^ { \frac { d s } { d + 2 k + 1 } } \| f \| _ { \mathcal W ^ { r , p } ( \Omega ) } . } & { } \end{array}
$$

Restricting to $\Omega ^ { \prime }$ and using $\chi = 1$ there proves the result.

## 6.7 Proof of Theorem 4.8

Proof. We prove each item in turn.

Proof of item 1. The low-frequency blocks are controlled by the $\mathcal { L } ^ { 2 }$ norm. For large $q ,$ take $n = n _ { q } : = \operatorname* { m a x } \{ n \in \mathbb { N } : 2 ^ { q } \geq C _ { 3 } \underline { { h } } _ { n } ^ { - 1 } \}$ . Then $n \gtrsim 2 ^ { q d }$ , and (6.15) gives

$$
\begin{array} { r } { \| \Delta _ { q , 0 } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } ^ { 2 } = a ^ { \top } Q _ { 0 } ( q ) a \lesssim 2 ^ { - q ( 2 k + 1 ) } \| a \| _ { 2 } ^ { 2 } \lesssim 2 ^ { - q ( d + 2 k + 1 - 2 \alpha d ) } . } \end{array}
$$

Together with (4.19), this gives

$$
\begin{array} { r l } & { \| \Delta _ { q , 0 } f \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \leq \| \Delta _ { q , 0 } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } + \| \Delta _ { q , 0 } ( f - f _ { n } ) \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \leq \| \Delta _ { q , 0 } f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } + \| f - f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } } \\ & { \qquad \lesssim 2 ^ { - q ^ { d + 2 k + 1 - 2 \alpha d } } , } \end{array}
$$

$$
\begin{array} { r } { \operatorname * { a n d } _ { \boldsymbol { q } \geq 0 } 2 ^ { \boldsymbol { q } ( \frac { d + 2 k + 1 } { 2 } - \alpha d ) } \| \Delta _ { \boldsymbol { q } , 0 } f \| _ { 2 } < \infty , \mathrm { i . e . , } f \in \mathcal { B } _ { 2 , \infty } ^ { \frac { d + 2 k + 1 } { 2 } - \alpha d } ( \mathbb { R } ^ { d } ) . } \end{array}
$$

Proof of item 2. Fix a cube $Q \Subset \Omega ^ { \prime }$ and choose a compactly supported orthonormal wavelet system on $\mathbb { R } ^ { d }$ with regularity greater than $\textstyle { \frac { d + 2 k + 1 } { 2 } }$ . There exists $q _ { 0 } \geq 0$ such that whenever $q \geq q _ { 0 }$ there is at least one wavelet $\psi _ { q }$ at scale $2 ^ { - q }$ such that supp $( \psi _ { q } ) \subset Q$ . The collection $\{ \psi _ { q } \} _ { q \ge q _ { 0 } }$ is orthogonal and, by the standard wavelet characterizations of Sobolev and Besov spaces,

$$
\Big \| \sum _ { q = q _ { 0 } } ^ { \infty } \alpha _ { q } \psi _ { q } \Big \| _ { \mathcal { H } ^ { \gamma } ( \mathbb { R } ^ { d } ) } ^ { 2 } \lesssim \sum _ { q = q _ { 0 } } ^ { \infty } 2 ^ { 2 q \gamma } | \alpha _ { q } | ^ { 2 } , \qquad \Big \| \sum _ { q = q _ { 0 } } ^ { \infty } \alpha _ { q } \psi _ { q } \Big \| _ { B _ { 2 , \infty } ^ { \gamma } ( \mathbb { R } ^ { d } ) } = \operatorname* { s u p } _ { q } 2 ^ { q \gamma } | \alpha _ { q } | .\tag{6.45}
$$

Let

$$
t = \frac { d + 2 k + 1 - 2 \alpha d } { 4 } , \qquad f : = \sum _ { q = q _ { 0 } } ^ { \infty } 2 ^ { - q t } \psi _ { q } ,
$$

so a direct computation using (6.45) yields $f \in \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } )$ and $f \not \in B _ { 2 , \infty } ^ { \frac { d + 2 k + 1 } { 2 } - \alpha d } ( \mathbb { R } ^ { d } )$ . Since $\operatorname { s u p p } ( f ) \subset$ $Q \Subset \Omega ^ { \prime }$ , this also implies $f | _ { \Omega ^ { \prime } } \notin B _ { 2 , \infty } ^ { \frac { d + 2 k + 1 } { 2 } - \alpha d } ( \Omega ^ { \prime } )$ . For $n \geq 2$ , set

$$
q _ { n } : = \left\lfloor { \frac { \alpha } { { \frac { d + 2 k + 1 } { 2 } } - t } } \log _ { 2 } n \right\rfloor , \qquad g _ { n } : = \sum _ { q = q _ { 0 } } ^ { q _ { n } } 2 ^ { - q t } \psi _ { q } .
$$

Then $q _ { n } \to \infty$ , and

$$
\| g _ { n } \| _ { \mathcal { H } ^ { \frac { d + 2 k + 1 } { 2 } } ( \mathbb { R } ^ { d } ) } ^ { 2 } \lesssim \sum _ { q = q _ { 0 } } ^ { q _ { n } } 2 ^ { q ( d + 2 k + 1 ) } 2 ^ { - 2 q t } \lesssim 2 ^ { 2 q _ { n } ( \frac { d + 2 k + 1 } { 2 } - t ) } \lesssim n ^ { 2 \alpha } .\tag{6.46}
$$

Moreover,

$$
\| f - g _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } ^ { 2 } = \sum _ { q > q _ { n } } 2 ^ { - 2 q t } \lesssim 2 ^ { - 2 q _ { n } t } \longrightarrow 0 .\tag{6.47}
$$

By Theorem 4.2, for every suficiently large n there exists $f _ { n } = \sum _ { j = 1 } ^ { n } a _ { j , n } \varphi _ { \theta _ { j , r } ^ { * } }$ such that

$$
\| g _ { n } - f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \Omega ^ { \prime } ) } \lesssim n ^ { - \frac { d + 2 k + 1 } { 2 d } } \| g _ { n } \| _ { \mathcal { H } ^ { \frac { d + 2 k + 1 } { 2 } } ( \Omega ) } \lesssim n ^ { - \frac { d + 2 k + 1 } { 2 d } + \alpha } ,\tag{6.48}
$$

and

$$
\left( n \sum _ { j = 1 } ^ { n } a _ { j , n } ^ { 2 } \right) ^ { \frac { 1 } { 2 } } \lesssim \left. g _ { n } \right. _ { \mathcal { H } ^ { \frac { d + 2 k + 1 } { 2 } } ( \Omega ) } \lesssim n ^ { \alpha } .\tag{6.49}
$$

Now (6.48) and (6.47) imply

$$
\| f - f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \Omega ^ { \prime } ) } \leq \| f - g _ { n } \| _ { \mathcal { L } ^ { 2 } ( \Omega ^ { \prime } ) } + \| g _ { n } - f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \Omega ^ { \prime } ) } \longrightarrow 0 .
$$

If the constant in (6.49) exceeds $M$ , rescale $f$ and its approximants by a suficiently small positive constant. This proves item 2.

## Proof of item 3.

Case $\textstyle 1 \colon \alpha > { \frac { 1 } { 2 } }$ . Let $f \in \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } )$ satisfy (4.21). For $q \geq 0$ , let $n _ { q } = 2 ^ { ( q + 1 ) d }$ and choose $v _ { q } ~ \in$ $\chi L _ { n _ { q } } ^ { k } ( \Theta _ { n _ { q } } )$ such that

$$
\| f - v _ { q } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \lesssim 2 ^ { - q ( \frac { d + 2 k + 1 } { 2 } - \alpha d ) } .\tag{6.50}
$$

Set $u _ { 0 } : = v _ { 0 }$ and $u _ { q } : = v _ { q } - v _ { q - 1 }$ for $q \geq 1$ . By nestedness, $u _ { q } \in \chi L _ { n _ { q } } ^ { k } ( \Theta _ { n _ { q } } )$ , and (6.50) gives $\| u _ { q } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \lesssim 2 ^ { - q ( \frac { d + 2 k + 1 } { 2 } - \alpha d ) }$ after adjusting the constant to include the first term. Take $r \in$ $( \frac { d + 2 k + \mathrm { i } } { 2 } - \alpha d , k + \frac { 1 } { 2 } )$ . Antipodal quasi-uniformity and Theorem 4.5 yield

$$
\begin{array} { r } { \| u _ { q } \| _ { \mathcal { H } ^ { r } ( \mathbb { R } ^ { d } ) } \lesssim n _ { q } ^ { \frac { r } { d } } \| u _ { q } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \lesssim 2 ^ { q ( r - \frac { d + 2 k + 1 } { 2 } + \alpha d ) } . } \end{array}\tag{6.51}
$$

As $f = \sum _ { q = 0 } ^ { \infty } u _ { q }$ in $\mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } )$ , equations (6.50)–(6.51) give, for $Q \in \mathbb { N } ,$

$$
\begin{array} { r l } & { \displaystyle \| \Delta _ { Q , 0 } f \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \leq \sum _ { q < Q } \| \Delta _ { Q , 0 } u _ { q } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } + \sum _ { q \geq Q } \| u _ { q } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \lesssim 2 ^ { - Q r } \sum _ { q < Q } \| \Delta _ { Q , 0 } u _ { q } \| _ { \mathcal { H } ^ { r } ( \mathbb { R } ^ { d } ) } + \sum _ { q \geq Q } \| u _ { q } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } } \\ & { \qquad \lesssim 2 ^ { - Q r } \displaystyle \sum _ { q < Q } 2 ^ { q ( r - \frac { d + 2 k + 1 } { 2 } + \alpha d ) } + \sum _ { q \geq Q } 2 ^ { - q ( \frac { d + 2 k + 1 } { 2 } - \alpha d ) } \lesssim 2 ^ { - Q ( \frac { d + 2 k + 1 } { 2 } - \alpha d ) } . } \end{array}\tag{6.52}
$$

Consequently, $f \in B _ { 2 , \infty } ^ { \frac { d + 2 k + 1 } { 2 } - \alpha d } ( \mathbb { R } ^ { d } )$

Case 2: $\textstyle 0 < \alpha < { \frac { 1 } { 2 } }$ . By nestedness, fix $\theta _ { 1 , 1 } ^ { * } \in \Theta _ { 1 }$ and set $f = \varphi _ { \theta _ { 1 , 1 } ^ { * } }$ . Then $f \in \chi L _ { n } ^ { k } ( \Theta _ { n } )$ for every $n ,$ so the approximation error in (4.21) is zero. On the other hand, Theorem 5.2 gives

$$
\| \Delta _ { q , 0 } f \| _ { 2 } ^ { 2 } \lesssim 2 ^ { - q ( 2 k + 1 ) } , \qquad q \geq q _ { 0 } .
$$

Therefore,

$$
2 ^ { q ( \frac { d + 2 k + 1 } { 2 } - \alpha d ) } \| \Delta _ { q , 0 } f \| _ { 2 } \approx 2 ^ { q d ( \frac { 1 } { 2 } - \alpha ) } \longrightarrow \infty ,
$$

which establishes the counterexample.

Case 3: $\begin{array} { r } { \alpha = \frac { 1 } { 2 } } \end{array}$ . By nestedness, after relabeling the points if necessary, we may write

$$
\Theta _ { n } = \{ \theta _ { 1 } ^ { * } , \ldots , \theta _ { n } ^ { * } \} , \qquad n = 1 , 2 \ldots .
$$

Then, for each $\nu ,$ we have

$$
\Theta _ { 2 ^ { \nu + 1 } } \backslash \Theta _ { 2 ^ { \nu } } = \{ \theta _ { 2 ^ { \nu } + 1 } ^ { * } , \theta _ { 2 ^ { \nu } + 2 } ^ { * } , \ldots , \theta _ { 2 ^ { \nu + 1 } } ^ { * } \} , \quad \nu = 1 , 2 , \ldots
$$

Set $a _ { 1 } = a _ { 2 } = 0$ . For each $\nu ,$ by (4.7) and Theorem 4.3, let $a ( \nu ) = ( a _ { 2 ^ { \nu } + 1 } , \dots , a _ { 2 ^ { \nu + 1 } } ) ^ { \top }$ be a unit vector in the eigenspace of the smallest eigenvalue of the matrix

$$
\begin{array} { r } { K _ { 0 } = K _ { 0 } ( \nu ) = \left( \langle \varphi _ { \theta _ { i } ^ { \ast } } , \varphi _ { \theta _ { j } ^ { \ast } } \rangle _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \right) _ { i , j = 2 ^ { \nu } + 1 } ^ { 2 ^ { \nu + 1 } } , } \end{array}
$$

so the corresponding function $g _ { \nu } : = \sum _ { j = 2 ^ { \nu } + 1 } ^ { 2 ^ { \nu + 1 } } a _ { j } \varphi _ { \theta _ { j } ^ { * } }$ satisfies

$$
\begin{array} { r } { \| g _ { \nu } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } ^ { 2 } = a ( \nu ) ^ { \top } K _ { 0 } ( \nu ) a ( \nu ) = \lambda _ { \operatorname* { m i n } } \big ( K _ { 0 } ( \nu ) \big ) \lesssim 2 ^ { - \nu \frac { 2 k + 1 } { d } } . } \end{array}\tag{6.53}
$$

Let

$$
f : = \sum _ { \nu = 1 } ^ { \infty } g _ { \nu } \in \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) , \qquad F _ { \nu } : = \sum _ { \ell = 1 } ^ { \nu } g _ { \ell } \in \chi L _ { 2 ^ { \nu + 1 } } ^ { k } ( \Theta _ { 2 ^ { \nu + 1 } } ) , \qquad \nu = 1 , 2 , \dots .
$$

For $2 ^ { \nu + 1 } \leq n < 2 ^ { \nu + 2 }$ , by nestedness, (4.21) holds with $\begin{array} { r } { \alpha = \frac { 1 } { 2 } . } \end{array}$

$$
\operatorname* { i n f } _ { f _ { n } \in \chi L _ { n } ^ { k } ( \Theta _ { n } ) } \| f - f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \leq \| f - F _ { \nu } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \leq \sum _ { \ell > \nu } \| g _ { \ell } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \lesssim 2 ^ { - \nu \frac { 2 k + 1 } { 2 d } } \lesssim n ^ { - \frac { 2 k + 1 } { 2 d } } .
$$

Next, we show that the stated Besov regularity fails. For each $\nu ,$ take $q _ { \nu } = \lceil \log _ { 2 } ( C _ { 3 } \underline { { h } } _ { 2 ^ { \nu + 1 } } ^ { - 1 } ) \rceil$ with $C _ { 3 }$ chosen so that (6.20) holds.

Let $a ^ { ( \nu ) } = ( 0 , 0 , a _ { 3 } , \ldots , a _ { 2 ^ { \nu + 1 } } ) ^ { \top }$ , and form $Q _ { 0 } ( q _ { \nu } )$ using $\Theta _ { 2 ^ { \nu + 1 } }$ . Taking $C _ { 3 }$ suficiently large for (6.20) gives

$$
\| \Delta _ { q \nu , 0 } F _ { \nu } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } ^ { 2 } = ( a ^ { ( \nu ) } ) ^ { \top } Q _ { 0 } ( q _ { \nu } ) a ^ { ( \nu ) } \gtrsim 2 ^ { - q _ { \nu } ( 2 k + 1 ) } \sum _ { j = 1 } ^ { 2 ^ { \nu + 1 } } a _ { j } ^ { 2 } \geq \nu 2 ^ { - q _ { \nu } ( 2 k + 1 ) } .
$$

Meanwhile,

$$
\| \Delta _ { q _ { \nu } , 0 } ( f - F _ { \nu } ) \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \lesssim \| f - F _ { \nu } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \lesssim 2 ^ { - \nu \frac { 2 k + 1 } { 2 d } } \lesssim 2 ^ { - q _ { \nu } ( k + \frac { 1 } { 2 } ) } ,
$$

and hence, for suficiently large $\nu ,$

$$
\begin{array} { r } { \| \Delta _ { q _ { \nu } , 0 } f \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \geq \| \Delta _ { q _ { \nu } , 0 } F _ { \nu } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } - \| \Delta _ { q _ { \nu } , 0 } ( f - F _ { \nu } ) \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \gtrsim \sqrt { \nu } 2 ^ { - q _ { \nu } ( k + \frac { 1 } { 2 } ) } } \end{array}
$$

Letting $\nu \to \infty$ gives $\| f \| _ { B _ { 2 , \infty } ^ { k + \frac { 1 } { 2 } } ( \mathbb { R } ^ { d } ) } = \infty .$

Proof of item 4. For each $n ,$ choose

$$
f _ { n } = \sum _ { j = 1 } ^ { n } a _ { j , n } \varphi _ { \theta _ { j , n } ^ { * } } , \qquad \widetilde { f } _ { n } = \sum _ { j = 1 } ^ { n } \widetilde { a } _ { j , n } \varphi _ { \widetilde { \theta } _ { j , n } ^ { * } }
$$

such that

$$
\| f - f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } + \| f - \widetilde { f } _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \lesssim n ^ { - \frac { d + 2 k + 1 } { 2 d } + \alpha } .
$$

The union contains $2 n$ distinct parameters with antipodal separation $\gtrsim n ^ { - 1 / d }$ . Applying (4.7) and Theorem 4.3 to the coeficients $a _ { 1 , n } , \ldots , a _ { n , n } , - { \widetilde { a } } _ { 1 , n } , \ldots , - { \widetilde { a } } _ { n , n }$ gives

$$
\| f _ { n } - \widetilde { f } _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } ^ { 2 } \gtrsim n ^ { - \frac { 2 k + 1 } { d } } \Big ( \sum _ { j = 1 } ^ { n } a _ { j , n } ^ { 2 } + \sum _ { j = 1 } ^ { n } \widetilde { a } _ { j , n } ^ { 2 } \Big ) .
$$

On the other hand,

$$
\begin{array} { r } { \| f _ { n } - \widetilde { f _ { n } } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \leq \| f - f _ { n } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } + \| f - \widetilde { f _ { n } } \| _ { \mathcal { L } ^ { 2 } ( \mathbb { R } ^ { d } ) } \lesssim n ^ { - \frac { d + 2 k + 1 } { 2 d } + \alpha } . } \end{array}
$$

Consequently,

$$
\sum _ { j = 1 } ^ { n } a _ { j , n } ^ { 2 } + \sum _ { j = 1 } ^ { n } \tilde { a } _ { j , n } ^ { 2 } \lesssim n ^ { \frac { 2 k + 1 } { d } } n ^ { - \frac { d + 2 k + 1 } { d } + 2 \alpha } = n ^ { 2 \alpha - 1 } .
$$

In particular,

$$
\left( n \sum _ { j = 1 } ^ { n } a _ { j , n } ^ { 2 } \right) ^ { \frac { 1 } { 2 } } \lesssim n ^ { \alpha } .
$$

Thus, for some $M$ independent of $n ,$ the approximants $f _ { n }$ belong to $\chi L _ { n , M } ^ { k , \alpha } ( \Theta _ { n } )$ and retain the rate (4.19). Item 1 now implies

$$
f \in { \cal B } _ { 2 , \infty } ^ { \frac { d + 2 k + 1 } { 2 } - \alpha d } ( \mathbb { R } ^ { d } ) ,
$$

which completes the proof.

## References

[1] M. Anthony and P. L. Bartlett. Neural Network Learning: Theoretical Foundations. Cambridge University Press, 1999.

[2] R. Arcang´eli and J. J. Torrens. Sampling inequalities in sobolev spaces. Journal of Approximation Theory, 182:18–28, 2014.

[3] F. Bach. Breaking the curse of dimensionality with convex neural networks. Journal of Machine Learning Research, 18(19):1–53, 2017.

[4] H. Bahouri, J.-Y. Chemin, and R. Danchin. Fourier Analysis and Nonlinear Partial Diferential Equations, volume 343 of Grundlehren der mathematischen Wissenschaften. Springer, 2011.

[5] J. Bai and L. Shi. Truncated kernel stochastic gradient descent on spheres. Mathematics of Computation, 95:2409–2480, 2026. Published electronically 18 July 2025.

[6] A. R. Barron. Universal approximation bounds for superpositions of a sigmoidal function. IEEE Transactions on Information theory, 39(3):930–945, 1993.

[7] A. R. Barron. Approximation and estimation bounds for artificial neural networks. Machine learning, 14:115–133, 1994.

[8] A. R. Barron, A. Cohen, W. Dahmen, and R. A. DeVore. Approximation and learning by greedy algorithms. Annals of Statistics, 36(1):64–94, 2008.

[9] P. L. Bartlett and S. Mendelson. Rademacher and gaussian complexities: Risk bounds and structural results. Journal of Machine Learning Research, 3:463–482, 2002.

[10] Y. Belfer, A. Geifman, M. Galun, and R. Basri. Spectral analysis of the neural tangent kernel for deep residual networks. Journal of Machine Learning Research, 25(184):1–49, 2024.

[11] S. N. Bernstein. Sur la meilleure approximation des fonctions continues par les polynˆomes du degr´e donn´e. I. Communications de la Soci´et´e Math´ematique de Kharkow. Deuxi\`eme S´erie, 13(2–3):49–144, 1912.

[12] R. Bhatia. Matrix Analysis, volume 169 of Graduate Texts in Mathematics. Springer, New York, 1997.

[13] B. Bordelon, A. Canatar, and C. Pehlevan. Spectrum dependent learning curves in kernel regression and wide neural networks. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 1024– 1034. PMLR, 2020.

[14] P. L. Butzer and R. J. Nessel. Fourier Analysis and Approximation: One Dimensional Theory. Birkh¨auser, Basel, 1971.

[15] Z. Cai, A. Doktorova, R. D. Falgout, and C. Herrera. Eficient shallow ritz method for onedimensional difusion-reaction problems. SIAM Journal on Scientific Computing, 47(6):S414– S435, 2025.

[16] A. Canatar, B. Bordelon, and C. Pehlevan. Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide neural networks. Nature Communications, 12:2914, 2021.

[17] J. Chen, X. Chi, W. E, and Z. Yang. Bridging traditional and machine learning-based algorithms for solving PDEs: The random feature method. Journal of Machine Learning, 1(3):268–298, 2022.

[18] A. Chkifa, A. Cohen, G. Migliorati, F. Nobile, and R. Tempone. Discrete least squares polynomial approximation with random evaluations—application to parametric and stochastic elliptic PDEs. ESAIM: Mathematical Modelling and Numerical Analysis, 49(3):815–837, 2015.

[19] A. Cohen, M. A. Davenport, and D. Leviatan. On the stability and accuracy of least squares approximations. Foundations of Computational Mathematics, 13(5):819–834, 2013.

[20] A. Cohen and G. Migliorati. Optimal weighted least-squares methods. The SMAI Journal of Computational Mathematics, 3:181–203, 2017.

[21] D. Costarelli. Saturation theorems for neural network operators by solving elliptic and hy perbolic PDEs with analytical and semi-analytical inverse problems, 2025.

[22] D. Costarelli and G. Vinti. Saturation classes for max-product neural network operators activated by sigmoidal functions. Results in Mathematics, 72:1555–1569, 2017.

[23] F. Cucker and D. X. Zhou. Learning theory: an approximation theory viewpoint, volume 24. Cambridge University Press, 2007.

[24] G. Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of control, signals and systems, 2(4):303–314, 1989.

[25] R. DeVore, B. Hanin, and G. Petrova. Neural network approximation. Acta Numerica, 30:327–444, 2021.

[26] R. A. DeVore. Nonlinear approximation. Acta Numerica, 7:51–150, 1998.

[27] R. A. DeVore and G. G. Lorentz. Constructive approximation, volume 303. Springer Science & Business Media, 1993.

[28] R. A. DeVore and V. N. Temlyakov. Some remarks on greedy algorithms. Advances in computational Mathematics, 5(1):173–187, 1996.

[29] W. E, C. Ma, and L. Wu. The barron space and the flow-induced function spaces for neural network models. Constructive Approximation, 55(1):369–406, 2022.

[30] N. El Karoui. The spectrum of kernel random matrices. The Annals of Statistics, 38(1):1–50, 2010.

[31] J. Fan, Z.-C. Guo, and L. Shi. Spectral algorithms for functional linear regression. Communications on Pure and Applied Analysis, 23(7):895–915, 2024.

[32] Z. Fang, H. Feng, S. Huang, and D.-X. Zhou. Theory of deep convolutional neural networks II: Spherical analysis. Neural Networks, 131:154–162, 2020.

[33] F. Filbir and H. N. Mhaskar. Marcinkiewicz–zygmund measures on manifolds. Journal of Complexity, 27(6):568–596, 2011.

[34] K. Gr¨ochenig. Sampling, Marcinkiewicz–Zygmund inequalities, approximation, and quadrature rules. Journal of Approximation Theory, 257:105455, 2020.

[35] X. Guo, J. Fan, and D.-X. Zhou. Sparsity and error analysis of empirical feature-based regularization schemes. Journal of Machine Learning Research, 17(89):1–34, 2016.

[36] T. Hangelbroek, F. J. Narcowich, C. Rieger, and J. D. Ward. An inverse theorem for compact lipschitz regions in R<sup>d</sup> using localized kernel bases. Mathematics of Computation, 87(312):1949–1989, 2018.

[37] T. Hangelbroek, F. J. Narcowich, and J. D. Ward. Extending data to improve stability and error estimates using asymmetric kansa-like methods to solve PDEs. SIAM Journal on Numerical Analysis, 64(4):1539–1560, 2026.

[38] J. He, L. Li, J. Xu, and C. Zheng. ReLU deep neural networks and linear finite elements. Journal of Computational Mathematics, 38(3):502–527, 2020.

[39] J. He and Z. Tian. Sharp Sobolev sandwich and approximation rates of Radon-domain L<sup>p</sup> ridge integral spaces for ReLU<sup>k</sup> networks, 2026. Preprint.

[40] J. He and J. Xu. MgNet: A unified framework of multigrid and convolutional neural network. Science China Mathematics, 62(7):1331–1354, 2019.

[41] Q. Hong, J. W. Siegel, Q. Tan, and J. Xu. On the activation function dependence of the spectral bias of neural networks, 2022.

[42] K. Hornik, M. Stinchcombe, and H. White. Multilayer feedforward networks are universal approximators. Neural networks, 2(5):359–366, 1989.

[43] D. Hsu, S. M. Kakade, and T. Zhang. Random design analysis of ridge regression. Foundations of Computational Mathematics, 14(3):569–600, 2014.

[44] T. Hu and Y. Lei. Early stopping for iterative regularization with general loss functions. Journal of Machine Learning Research, 23(339):1–36, 2022.

[45] T. Hu and D.-X. Zhou. Distributed regularized least squares with flexible Gaussian kernels. Applied and Computational Harmonic Analysis, 53:349–377, 2021.

[46] G.-B. Huang, Q.-Y. Zhu, and C.-K. Siew. Extreme learning machine: Theory and applications. Neurocomputing, 70(1–3):489–501, 2006.

[47] S. Huang, L. Fiorito, L. Rosasco, and T. Poggio. Learning sparse compositional functions with norm-constrained neural networks, 2026.

[48] S. Huang, H. Labarri\`ere, E. De Vito, T. Poggio, and L. Rosasco. Learning multi-index models with hyper-kernel ridge regression, 2025. Version 2 revised September 2026.

[49] A. Jacot, F. Gabriel, and C. Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems, volume 31, 2018.

[50] L. K. Jones. A simple lemma on greedy approximation in hilbert space and convergence rates for projection pursuit regression and neural network training. The Annals of Statistics, 20(1):608–613, 1992.

[51] P. Jorgensen and D. E. Stewart. Approximation properties of ridge functions and extreme learning machines. SIAM Journal on Mathematics of Data Science, 3(3):815–832, 2021.

[52] L. Kamenski, W. Huang, and H. Xu. Conditioning of finite element equations with arbitrary anisotropic meshes. Mathematics of Computation, 83(289):2187–2211, 2014.

[53] J. M. Klusowski and A. R. Barron. Approximation by combinations of ReLU and squared ReLU ridge functions with ℓ<sup>1</sup> and ℓ<sup>0</sup> controls. IEEE Transactions on Information Theory, 64(12):7649–7656, 2018.

[54] Y. Kolomoitsev and S. Tikhonov. Marcinkiewicz–zygmund inequalities in quasi-banach function spaces. Journal of Complexity, 93:101999, 2026.

[55] V. Koltchinskii and E. Gin´e. Random matrix approximation of spectra of integral operators. Bernoulli, 6(1):113–167, 2000.

[56] Q. T. Le Gia, F. J. Narcowich, J. D. Ward, and H. Wendland. Continuous and discrete least-squares approximation by radial basis functions on spheres. Journal of Approximation Theory, 143:124–133, 2006.

[57] Y. Lei, T. Hu, and K. Tang. Generalization performance of multi-pass stochastic gradient descent with convex loss functions. Journal of Machine Learning Research, 22(25):1–41, 2021.

[58] M. Leshno, V. Y. Lin, A. Pinkus, and S. Schocken. Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. Neural networks, 6(6):861– 867, 1993.

[59] S. Li, L. Ling, and K. C. Cheung. Discrete least-squares radial basis functions approximations. Applied Mathematics and Computation, 355:542–552, 2019.

[60] Y. Li. A new analysis of empirical interpolation methods and Chebyshev greedy algorithms. SIAM Journal on Numerical Analysis, 63(2):931–948, 2025.

[61] Y. Li and J. W. Siegel. Entropy-based convergence rates of greedy algorithms. Mathematical Models and Methods in Applied Sciences, 34(5):779–802, 2024.

[62] Y. Li and Y. Wang. Sharp embeddings between quasi-Banach Besov spaces and shallow ReLU variation spaces, 2026. Preprint.

[63] Q. Lin, H. Xie, and J. Xu. Lower bounds of the discretization error for piecewise polynomials. Mathematics of Computation, 83(285):1–13, 2014.

[64] V. Y. Lin and A. Pinkus. Fundamentality of ridge functions. Journal of Approximation Theory, 75(3):295–311, 1993.

[65] P. Liu, Y. Liu, X. Zhou, and D.-X. Zhou. Approximation of functionals on korobov spaces with fourier functional networks. Neural Networks, 182:106922, 2025.

[66] X. Liu, T. Mao, and J. Xu. Condition numbers and eigenvalue spectra of shallow networks on spheres, 2025. Preprint.

[67] X. Liu, T. Mao, and J. Xu. Integral representations of sobolev spaces via ReLU<sup>k</sup> activation function and optimal error estimates for linearized networks, 2025. Preprint.

[68] X. Liu, T. Mao, and J. Xu. Optimal neural network approximation via empirical least squares with deterministic samples, 2026. Preprint.

[69] Y. Liu. Generalization analysis for classification on korobov space, 2025.

[70] Y. Liu, T. Mao, J. W. Siegel, and J. Xu. Parameter bounds and inverse approximation for linearized relu<sup>k</sup> networks on the sphere. Unpublished manuscript, 2026.

[71] Y. Liu, T. Mao, and D.-X. Zhou. Approximation of functions from korobov spaces by shallow neural networks. Information Sciences, 670:120573, 2024.

[72] G. G. Lorentz. Inequalities and the saturation classes of Bernstein polynomials. In P. L. Butzer and J. Korevaar, editors, On Approximation Theory / Uber Approximationstheorie<sup>¨</sup> , volume 5 of International Series of Numerical Mathematics, pages 200–207. Birkh¨auser, Basel, 1964.

[73] J. Lu, Z. Shen, H. Yang, and S. Zhang. Deep network approximation for smooth functions. SIAM Journal on Mathematical Analysis, 53(5):5465–5506, 2021.

[74] Y. Makovoz. Random approximants and neural networks. Journal of Approximation Theory, 85(1):98–109, 1996.

[75] T. Mao, J. W. Siegel, and J. Xu. Approximation rates for shallow ReLU<sup>k</sup> neural networks on Sobolev spaces via the Radon transform. SIAM Journal on Mathematical Analysis, 58(2):1171–1186, 2026.

[76] T. Mao and J. Xu. Configuration-dependent lower bounds for approximation by shallow ReLU<sup>k</sup> networks on the sphere, 2025. Revised September 2026.

[77] T. Mao and D.-X. Zhou. Rates of approximation by relu shallow neural networks. Journal of Complexity, 79:101784, 2023.

[78] B. Maurey. Type et cotype dans les espaces munis de structures locales inconditionnelles. S´eminaire d’analyse fonctionnelle (dit “Maurey–Schwartz”), pages 1–25, 1973–1974. Expos´es 24 et 25.

[79] H. Mhaskar, F. Narcowich, and J. Ward. Spherical marcinkiewicz-zygmund inequalities and positive quadrature. Mathematics of computation, 70(235):1113–1130, 2001.

[80] H. N. Mhaskar. Approximation properties of a multilayered feedforward artificial neural network. Advances in Computational Mathematics, 1:61–80, 1993.

[81] H. N. Mhaskar. Function approximation with zonal function networks with activation functions analogous to the rectified linear unit functions. Journal of Complexity, 51:1–19, 2019.

[82] H. N. Mhaskar and T. Mao. Tractability of approximation by general shallow networks. Analysis and Applications, 22(3):535–568, 2024.

[83] H. N. Mhaskar, F. J. Narcowich, J. Prestin, and J. D. Ward. L<sup>p</sup> bernstein estimates and approximation by spherical basis functions. Mathematics of Computation, 79(271):1647–1679, 2010.

[84] H. N. Mhaskar, E. Tsoukanis, and A. D. Jagtap. An approximation theory perspective on machine learning. Neural Networks, 200:108841, 2026.

[85] F. J. Narcowich, J. D. Ward, and H. Wendland. Sobolev bounds on functions with scattered zeros, with applications to radial basis function surface fitting. Mathematics of Computation, 74(250):743–763, 2005.

[86] F. J. Narcowich, J. D. Ward, and H. Wendland. Sobolev error estimates and a bernstein inequality for scattered data interpolation via radial basis functions. Constructive Approximation, 24(2):175–186, 2006.

[87] S. M. Nikolskii. Approximation of Functions of Several Variables and Imbedding Theorems. Springer, Berlin, 1975.

[88] R. Parhi and R. D. Nowak. Near-minimax optimal estimation with shallow ReLU neural networks. IEEE Transactions on Information Theory, 69(2):1125–1140, 2023.

[89] P. Petrushev and Y. Xu. Localized polynomial frames on the interval with jacobi weights. Journal of Fourier Analysis and Applications, 11(5):557–575, 2005.

[90] P. P. Petrushev. Approximation by ridge functions and neural networks. SIAM Journal on Mathematical Analysis, 30(1):155–189, 1998.

[91] A. Pinkus. Approximation theory of the mlp model in neural networks. Acta numerica, 8:143–195, 1999.

[92] G. Pisier. Remarques sur un r´esultat non publi´e de B. Maurey. S´eminaire d’analyse fonctionnelle, pages 1–12, 1980–1981. Expos´e 5.

[93] N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht, Y. Bengio, and A. Courville. On the spectral bias of neural networks. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 5301–5310. PMLR, 2019.

[94] A. Rahimi and B. Recht. Random features for large-scale kernel machines. In Advances in Neural Information Processing Systems, pages 1177–1184, 2007.

[95] M. Raissi, P. Perdikaris, and G. E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378:686–707, 2019.

[96] L. Rosasco, M. Belkin, and E. De Vito. On learning with integral operators. Journal of Machine Learning Research, 11(30):905–934, 2010.

[97] A. Rudi and L. Rosasco. Generalization properties of learning with random features. In Advances in Neural Information Processing Systems, volume 30, pages 3215–3225, 2017.

[98] R. Schaback and H. Wendland. Inverse and saturation theorems for radial basis function interpolation. Mathematics of Computation, 71(238):669–681, 2002.

[99] L. L. Schumaker. Spline Functions: Basic Theory. Cambridge Mathematical Library. Cambridge University Press, 3 edition, 2007.

[100] Y. Shang, F. Wang, and J. Sun. Randomized neural network with Petrov–Galerkin methods for solving linear and nonlinear partial diferential equations. Communications in Nonlinear Science and Numerical Simulation, 127:107518, 2023.

[101] Z. Shi, J. Fan, L. Song, D.-X. Zhou, and J. A. K. Suykens. Nonlinear functional regression by functional deep neural network with kernel embedding. Journal of Machine Learning Research, 26(284):1–49, 2025.

[102] Z. Shi, Z. Yu, and D.-X. Zhou. Learning theory of distribution regression with neural networks. Constructive Approximation, 62(1):61–104, 2025.

[103] J. W. Siegel and J. Xu. High-order approximation rates for shallow neural networks with cosine and ReLU<sup>k</sup> activation functions. Applied and Computational Harmonic Analysis, 58:1– 26, 2022.

[104] J. W. Siegel and J. Xu. Optimal convergence rates for the orthogonal greedy algorithm. IEEE Transactions on Information Theory, 68(5):3354–3361, 2022.

[105] J. W. Siegel and J. Xu. Characterization of the variation spaces corresponding to shallow neural networks. Constructive Approximation, 57(3):1109–1132, 2023.

[106] J. W. Siegel and J. Xu. Sharp bounds on the approximation rates, metric entropy, and n-widths of shallow neural networks. Foundations of Computational Mathematics, 24(2):481– 537, 2024.

[107] E. M. Stein. Harmonic Analysis: Real-Variable Methods, Orthogonality, and Oscillatory Integrals. Princeton University Press, Princeton, NJ, 1993.

[108] J. Sun, S. Dong, and F. Wang. Local randomized neural networks with discontinuous Galerkin methods for partial diferential equations. Journal of Computational and Applied Mathematics, 445:115830, 2024.

[109] H. Triebel. Theory of Function Spaces. Birkh¨auser, Basel, 1983.

[110] S. Wang, Z. Li, and Q. Li. Inverse approximation theory for nonlinear recurrent neural networks. In The Twelfth International Conference on Learning Representations, 2024.

[111] Y. Wang and S. Dong. An extreme learning machine-based method for computational pdes in higher dimensions. Computer Methods in Applied Mechanics and Engineering, 418:116578, 2024.

[112] H. Wendland. Scattered Data Approximation, volume 17 of Cambridge Monographs on Applied and Computational Mathematics. Cambridge University Press, 2005.

[113] J. Xu. Iterative methods by space decomposition and subspace correction. SIAM review, 34(4):581–613, 1992.

[114] J. Xu. Finite neuron method and convergence analysis. Communications in Computational Physics, 28(5):1707–1745, 2020.

[115] Z.-Q. J. Xu, Y. Zhang, T. Luo, Y. Xiao, and Z. Ma. Frequency principle: Fourier analysis sheds light on deep neural networks. Communications in Computational Physics, 28(5):1746– 1767, 2020.

[116] Y. Yang and J. Fan. Approximation and learning of anisotropic and mixed smooth functions by deep ReLU neural networks, 2026. Version 2, 24 July 2026.

[117] Y. Yang and D.-X. Zhou. Optimal rates of approximation by shallow ReLU<sup>k</sup> neural networks and applications to nonparametric regression. Constructive Approximation, 62(2):329–360, 2025.

[118] Z. Yang, S. Huang, H. Feng, and D.-X. Zhou. Spherical analysis of learning nonlinear functionals. Constructive Approximation, 64:233–261, 2026.

[119] D. Yarotsky. Error bounds for approximations with deep ReLU networks. Neural Networks, 94:103–114, 2017.

[120] D. Yarotsky. Optimal approximation of continuous functions by very deep ReLU networks. In Proceedings of the 31st Conference On Learning Theory, volume 75 of Proceedings of Machine Learning Research, pages 639–649. PMLR, 2018.

[121] S. Zhang, H. Zhao, Y. Zhong, and H. Zhou. Why shallow networks struggle to approximate and learn high frequencies. Information and Inference: A Journal of the IMA, 14(3):iaaf022, 2025.

[122] H. Zhou and L. Shi. Expressive power of deep networks on manifolds: Simultaneous approximation, 2025. Preprint.