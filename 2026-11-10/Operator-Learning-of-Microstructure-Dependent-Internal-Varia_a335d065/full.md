# Operator Learning of Microstructure-Dependent Internal-Variable Models for Inelastic Homogenization: Approximation and Complexity

Lianghao Cao<sup>1,</sup>∗, George Stepaniants<sup>2</sup>, Andrew M. Stuart<sup>1</sup>

## Abstract

We study the learning of homogenized constitutive laws for inelastic heterogeneous materials, aiming to replace costly concurrent unit-cell simulations with surrogates that generalize across loading histories and microstructures. We focus on the recurrent neural operator framework, motivated by the explicit form of homogenized Kelvin–Voigt linear viscoelasticity. In this framework, internal variables enable a Markovian approximation of loading history dependence, and neural operators represent the microstructure-dependent constitutive response. For periodic Kelvin–Voigt viscoelasticity, we prove a universal approximation theorem showing that the model can approximate the homogenized constitutive law to any prescribed accuracy, uniformly in time and over a wide set of admissible strain trajectories and microstructural property fields, in two and three dimensions. This comprises a significant extension of the existing, somewhat limited, one-dimensional theory that uses explicit solutions for piecewise constant materials. Instead, in two and three dimensions, we adopt a more versatile approximation based on the spectral representation of the memory kernel that arises from homogenization. This also enables us to give an explicit bound on the internalvariable dimension, suficient for approximation, exhibiting logarithmic growth in the inverse relative stress error and in material contrast. Numerical experiments on viscoelasticity and crystal plasticity complement the theory, illustrating the roles of internal-variable dimension and sample complexity in learning homogenized responses from unit-cell data. Theoretical results are formally verified in Lean 4.

![](images/e1def7b123afe93c4383473a5526b764122b22233bbc7f803e5ab70c7e78e1cd.jpg)  
Figure 1: Graphical abstract.

## 1. Introduction

Macroscopic material behavior often reflects physical processes at much smaller length scales. Homogenization provides a systematic framework for deriving macroscopic constitutive models for multiscale materials by averaging the response of a unit cell that resolves the local microstructural behavior [1, 2, 3, 4, 5]. For inelastic heterogeneous materials, the homogenized constitutive model  depends both on the loading history and on the microstructural properties:

$$
\overline { { { P } } } ( t ) = \mathcal { H } ( \{ \overline { { { F } } } ( s ) \} _ { s \in [ 0 , t ] } ; M ) , \quad t \geq 0 ,\tag{1}
$$

where $\overline { { P } }$ is the macroscopic first Piola–Kirchhof stress tensor, $\overline { F }$ is the macroscopic deformation gradient, and M denotes the microstructural property field on the unit cell.

Typically, the form of is not known analytically, but it can be evaluated through unit-cell simulations. This makes macroscopic simulation via computational homogenization or $\mathrm { F E ^ { 2 } }$ [6, 7, 8] expensive, since the unit-cell problem must be repeatedly solved at macroscopic quadrature points and time steps. This has motivated recent eforts to learn from data [9, 10, 11, 12, 13, 14, 15]. Many existing approaches rely on standard machinelearning architectures applied to discretized loading histories and microstructural descriptions [9, 16, 17, 18, 19], and therefore approximate discretized input-output relations rather than the underlying continuum micromechanics. Since loading histories and microstructural properties are functions of time and space, neural operators [11, 20, 12, 21, 22] ofer a natural alternative: they learn mappings between function spaces, are hence interpretable in the standard language of continuum mechanics and can, in principle, generalize across discretizations. Following [12, 23], we consider the neural operator model

$$
\overline { { P } } ( t ) \approx \overline { { P } } _ { \mathrm { N O } } ( t ) = \mathcal { H } _ { \mathrm { N O } } ( \{ \overline { { F } } ( s ) \} _ { s \in [ 0 , t ] } ; M ) ,
$$

defined through a microstructure-dependent internal-variable dynamical system:

$$
\overline { { P } } _ { \mathrm { N O } } ( t ) = \mathcal { F } _ { \mathrm { N O } } ( z ( t ) , \xi ( t ) ; M ) ,\tag{2a}
$$

$$
\dot { \xi } ( t ) = \mathcal { G } _ { \mathrm { N O } } ( z ( t ) , \xi ( t ) ; M ) , \quad \xi ( 0 ) = 0 .\tag{2b}
$$

Here, $z ( t )$ denotes the collection of relevant instantaneous macroscopic variables at time $t ,$ for example, $\overline { { F } } ( t ) , \ \dot { \overline { { F } } } ( t )$ , or both. In this model, $\xi \in \mathbb { R } ^ { n _ { \xi } }$ is a vector of internal variables that encodes the deformation history [24, 25, 26, 27, 28, 12, 13, 29, 30, 31, 32], and $\mathcal { F } _ { \mathrm { N O } }$ and $\mathcal { G } _ { \mathrm { N O } }$ are the stress and kinetic response, respectively. The internal-variable dimension $n _ { \xi }$ and the response mappings $\mathcal { F } _ { \mathrm { N O } } , \mathcal { G } _ { \mathrm { N O } }$ are learned from data, and the response mappings are parameterized by neural operators with universal approximation properties [33, 20, 34]. In this formulation, each macroscopic material point stores only the current value of $\xi ( t ) \in \mathbb { R } ^ { n _ { \xi } }$ in place of the full loading history. This leads to an explicit, Markovian surrogate $\mathcal { H } _ { \mathrm { N O } }$ for the implicit, generally non-Markovian closure $\mathcal { H } .$ . The dimension $n _ { \xi }$ thus governs both the representational capacity of the model and the computational cost of the macroscopic simulation.

Remark 1.1. In the small strain setting the constitutive law (1) is typically formulated in terms of the macroscopic infinitesimal strain $\begin{array} { r } { \overline { { e } } = \frac { 1 } { 2 } ( \overline { { F } } + \overline { { F } } ^ { \top } ) - I } \end{array}$ and the Cauchy stress $\overline { { \sigma } }$ with $\overline { { \sigma } } \approx \overline { { P } }$ . Then (2) can be modified, replacing $\overline { F }$ with the macroscopic strain e and $\overline { { P } }$ with $\overline { { \sigma } }$

This formulation raises the following questions for the neural operator model $\mathcal { H } _ { \mathrm { N O } }$ in relation to the homogenized model :

(Q1) (Approximation) Can $\mathcal { H } _ { \mathrm { N O } }$ approximate  to any prescribed accuracy?

(Q2) (Internal-Variable Complexity) How do the approximation properties of $\mathcal { H } _ { \mathrm { N O } }$ depend on the dimension of internal variables $n _ { \xi } ?$

These questions are dificult to address in general because the unit-cell problem is typically not amenable to analysis. In this work, we answer these questions in two settings: Kelvin– Voigt viscoelasticity, for which we provide theory and numerical experiments, and finitedeformation crystal plasticity, for which we provide numerical experiments.

For Kelvin–Voigt viscoelasticity, the fine-scale constitutive model is

$$
\sigma ^ { \epsilon } ( x , t ) = \mathbb { V } ( x / \epsilon ) \partial _ { t } e ^ { \epsilon } ( x , t ) + \mathbb { E } ( x / \epsilon ) e ^ { \epsilon } ( x , t ) ,\tag{3}
$$

where $x \in \Omega$ denotes a macroscopic material point, $\epsilon > 0$ is the microstructure length scale, $\sigma ^ { \epsilon }$ and $e ^ { \epsilon }$ are the ϵ-scale stress and strain fields, and $\mathbb { V }$ and E are the microstructural viscosity and elasticity tensor fields that constitute M. Kelvin–Voigt viscoelasticity results in a fadingmemory model for the homogenized constitutive law in the limit $\epsilon  0$ [35, 36, 37, 38]. Specifically, Francfort and Suquet [39] show that at each material point, the macroscopic stress is governed by a constitutive law of the form

$$
\overline { { \sigma } } ( t ) = \mathbb { V } ^ { \mathrm { h o m } } \dot { \overline { { e } } } ( t ) + \mathbb { E } ^ { \mathrm { h o m } } \overline { { e } } ( t ) + \int _ { 0 } ^ { t } \mathbb { K } ( t - s ) \dot { \overline { { e } } } ( s ) \mathrm { d } s .\tag{4}
$$

Here, $\overline { { \sigma } }$ and $\overline { { e } }$ denote the macroscopic stress and strain, and ${ \dot { \overline { { e } } } } : = { \frac { \mathrm { d } } { \mathrm { d } t } } { \overline { { e } } } .$ . The homogenized viscosity and elasticity tensors $\mathbb { V } ^ { \mathrm { h o m } }$ and $\mathbb { E } ^ { \mathrm { h o m } }$ , as well as the memory kernel K, depend implicitly on the microstructural properties M. Thus, this homogenization result defines a constitutive law of the form $\overline { { \sigma } } ( t ) = \mathcal { H } ( \{ \overline { { e } } ( s ) \} _ { s \in [ 0 , t ] } ; M )$ . Our interest is in approximating the memory kernel to obtain an internal-variable Markovian model of the form $\begin{array} { r } { \overline { { \sigma } } _ { \mathrm { N O } } ( t ) = } \end{array}$ $\mathcal { H } _ { \mathrm { N O } } ( \{ \overline { { e } } ( s ) \} _ { s \in [ 0 , t ] } ; M )$

## 1.1. Main Results and Paper Overview

In this subsection, we summarize our main results and outline the rest of the paper. We formally verified the theoretical results in Lean 4 [40] using mathlib [41]; details appear below. The formalization and verification instructions are available in the accompanying GitHub repository<sup>3</sup>.

Result 1 (Homogenized Model). Periodic homogenization of Kelvin–Voigt viscoelasticity gives rise to a fading-memory constitutive law whose instantaneous response and memory kernel are determined by bounded linear operators arising from static unit-cell problems. The memory kernel admits a spectral representation whose relaxation-rate spectrum has bounded support and bounded total-variation norm. ◇

The full statement appears in Theorems 3.1 and 3.4, and both are formally verified. Building on the framework of [39], this model form and the spectral representation of the kernel provide the basis for the approximation theory that follows.

Result 2 (Universal Approximation). The parameters of the neural operator model $\mathcal { H } _ { \mathrm { N O } }$ may be chosen to approximate the true homogenized constitutive law to any prescribed accuracy, uniformly in time, over microstructures whose viscosity and elasticity tensor fields are uniformly bounded, coercive, and of bounded spatial variation, and over strain trajectories with uniformly bounded magnitudes and rates.

The full statement appears in Theorem 4.4 and is formally verified. This theorem addresses (Q1) and, although similar in form to the one-dimensional approximation results of [23], requires substantial new ideas in two and three spatial dimensions. The previous one-dimensional analysis relies on the fact that the Prony series approximation of the memory kernel [42, 43, 44, 45, 46, 38] is exact for piecewise-constant microstructures [13]. In two and three dimensions, we instead directly estimate the approximation error without resorting to a specific microstructure. The proof constructs a Prony-series internal-variable model whose stress response approximates the homogenized constitutive law uniformly over admissible strain trajectories and microstructures. Its continuous dependence on microstructure enables approximation by neural operators. Assuming that the neural operator model emulates this intermediate model, a quantitative answer to (Q2) is obtained by the approximation rate of the Prony series:

Result 3 (Internal-Variable Complexity). The suficient internal-variable dimension grows logarithmically with the inverse relative stress error and with each material contrast when the remaining parameters are held constant. ◇

The full statement appears in Theorem 4.5 and is formally verified. This scaling suggests that the required internal-variable dimension remains computationally manageable for uniformly accurate homogenization across microstructural instances with large property variations or a small tolerance for approximation error. It also supports the practical use of the internal-variable memory representation in macroscopic simulations.

We also present empirical results on training and testing the neural-operator model on the viscoelasticity and crystal plasticity datasets. We report the supporting numerical study in Section 5.

Result 4 (Empirical Complexity). For viscoelasticity, the accuracy of the trained model improves algebraically when the internal-variable dimension increases. For crystal plasticity, the trained model’s accuracy improves and then plateaus as the internal-variable dimension increases. ◇

The remainder of the paper is organized as follows. Section 2 introduces the periodic homogenization setting for Kelvin–Voigt viscoelasticity and defines the neural operator constitutive model. Section 3 presents the homogenization theory, identifies the operator form of the homogenized model, and provides the spectral representation of the memory kernel; together, these establish Result 1. Section 4 states the universal approximation theorem and the internal-variable complexity bound, corresponding to Results 2 and 3. Section 5 reports numerical experiments on viscoelasticity and crystal plasticity (Result 4), and Section 6 summarizes the findings.

Remark 1.2. Result 1 underpins the approximation and complexity results. It is therefore valuable to give intuition about why this result holds. If the divergence of the stress in (3) is balanced with an ϵ-independent body force f then this results in a dynamical system of the form

$$
\partial _ { t } \mathcal { L } _ { \mathbb { V } } ^ { \epsilon } u + \mathcal { L } _ { \mathbb { E } } ^ { \epsilon } u = f .
$$

Here, $u \ : = \ : u ^ { \epsilon }$ is the displacement field. Based on standard assumptions concerning the material properties, the elliptic operators $\mathcal { L } _ { \mathbb { V } } ^ { \epsilon }$ and $\mathcal { L } _ { \mathbb { E } } ^ { \epsilon }$ are associated with coercive and bounded quadratic forms on the displacement space. The relaxation rates of the problem are governed by the bounded operator $( \mathcal { L } _ { \mathbb { V } } ^ { \epsilon } ) ^ { - 1 } \mathcal { L } _ { \mathbb { E } } ^ { \epsilon }$ . If the coercivity and boundedness hold uniformly in ϵ, then this leads to upper and lower bounds, uniformly in ϵ, for the relaxation rates of the multiscale problem. Therefore, it is unsurprising that the relaxation rates appearing in K should satisfy uniform positive lower and upper bounds. However, additional efects come into play in the memory and, in particular, memory vanishes $i f \mathbb { V } = \mathbb { E }$ . Hence the operator $( \mathcal { L } _ { \mathbb { V } } ^ { \epsilon } ) ^ { - 1 } \mathcal { L } _ { \mathbb { E } } ^ { \epsilon }$ does not by itself determine the memory kernel; rather the analysis also quantifies the discrepancy between V and E.

## 2. Kelvin–Voigt Viscoelasticity: Homogenization and Neural Operator Model

In this section, we present the unit-cell formulation of the homogenized response of a Kelvin–Voigt viscoelastic heterogeneous material and introduce a neural-operator constitutive model designed to learn this response. In Section 2.1, we define the geometric and tensorial setting for the heterogeneous material and periodic unit cell. In Section 2.2, we pose the ϵ-scale problem and define the constitutive response through a periodic unit-cell problem, before coupling it to the macroscopic balance equations. In Section 2.3, we present the neural operator constitutive model.

## 2.1. Settings

Spatial and Temporal Domains. Let $\Omega \subset \mathbb { R } ^ { d }$ with $d \in \{ 1 , 2 , 3 \}$ be a bounded open material domain and $ { \mathbb { T } } ^ { d } : =  { \mathbb { R } ^ { d } } /  { \mathbb { Z } ^ { d } }$ the d-dimensional unit torus with normalized Lebesgue measure. Throughout, we identify 1-periodic functions with functions on $\mathbb { T } ^ { d }$ . For the homogenization problem, choose any terminal time $T > 0$ and let $t \in [ 0 , T ]$ denote time. ◇

Second and Fourth Order Tensors. Let $\mathcal { V } ^ { 2 }$ denote the set of symmetric 2nd-order tensors, equipped with the Frobenius inner product $( e _ { 1 } , e _ { 2 } ) _ { F } : = \mathrm { t r } ( e _ { 1 } ^ { \top } e _ { 2 } )$ and the induced norm $\| e \| _ { F }$ Let $\mathcal { V } ^ { 4 }$ denote the set of 4th-order tensors with major and minor symmetries. For $\mathbb { A } \in \mathcal { V } ^ { 4 }$ and $e \in \mathcal { V } ^ { 2 }$ , the action $\mathbb { A } e \in \mathcal { V } ^ { 2 }$ is defined by $\begin{array} { r } { ( \mathbb { A } e ) _ { i j } = \sum _ { k , l } \mathbb { A } _ { i j k l } e _ { k l } } \end{array}$ , and the Frobenius inner product on $\mathcal { V } ^ { 4 }$ is $\begin{array} { r } { ( \mathbb { A } , \mathbb { B } ) _ { F } = \sum _ { i , j , k , l } \mathbb { A } _ { i j k l } \mathbb { B } _ { i j k l } } \end{array}$ . The positive-definite subset $\mathcal { V } _ { + } ^ { 4 } \subset \mathcal { V } ^ { 4 }$ consists of tensors A with $( e , \mathbb { A } e ) _ { F } \geq \alpha \Vert \bar { e } \Vert _ { F } ^ { 2 }$ for some $\alpha > 0$ and all $e \in \mathcal { V } ^ { 2 }$ . Throughout, we assume tensor fields with values in $\mathcal { V } _ { + } ^ { 4 }$ are uniformly coercive, so α can be chosen independently of position. ◇

## 2.2. Microscopic, Unit-Cell, and Macroscopic Problems

The ϵ-Scale Problem. Let $\epsilon > 0$ be the microstructure length scale relative to the material domain Ω. Let $\mathbb { V } , \mathbb { E } : \mathbb { T } ^ { d } \to \mathcal { V } _ { + } ^ { 4 }$ be 1-periodic viscosity and elasticity tensor fields that characterize the microstructure. With $\mathbb { V } ^ { \epsilon } ( x ) = \mathbb { V } ( x / \epsilon ) , \mathbb { E } ^ { \epsilon } ( x ) = \mathbb { E } ( x / \epsilon )$ , and $\operatorname { s y m } ( \circ )$ the symmetric part, the ϵ-scale problem is:

$$
e ^ { \epsilon } ( x , t ) = \mathrm { s y m } ( \nabla _ { x } u ^ { \epsilon } ( x , t ) ) ,
$$

$$
x \in \Omega , t \in [ 0 , T ] ,\tag{5a}
$$

$$
\sigma ^ { \epsilon } ( x , t ) = \mathbb { V } ^ { \epsilon } ( x ) \partial _ { t } e ^ { \epsilon } ( x , t ) + \mathbb { E } ^ { \epsilon } ( x ) e ^ { \epsilon } ( x , t ) ,
$$

$$
x \in \Omega , t \in [ 0 , T ] ,\tag{5b}
$$

$$
- \nabla _ { x } \cdot \sigma ^ { \epsilon } ( x , t ) = f ( x , t ) ,
$$

$$
x \in \Omega , t \in [ 0 , T ] ,\tag{5c}
$$

$$
u ^ { \epsilon } ( x , 0 ) = u _ { 0 } ^ { \epsilon } ( x ) ,
$$

$$
x \in \Omega ,\tag{5d}
$$

$$
u ^ { \epsilon } ( x , t ) = 0 ,
$$

$$
x \in \partial \Omega .\tag{5e}
$$

Here, $u ^ { \epsilon } : \Omega \times [ 0 , T ] \to \mathbb { R } ^ { d }$ is the displacement, $e ^ { \epsilon } : \Omega \times [ 0 , T ] \to \mathcal { V } ^ { 2 }$ is the strain, $\sigma ^ { \epsilon }$ : $\Omega \times [ 0 , T ]  \mathcal { V } ^ { 2 }$ is the stress, $u _ { 0 } ^ { \epsilon } : \Omega \to \mathbb { R } ^ { d }$ is the initial condition, and $f : \Omega \times [ 0 , T ]   { \mathbb { R } ^ { d } }$ is an external force. ◇

The Unit-Cell Problem. At a fixed macroscopic material point, we prescribe a strain history $\overline { { e } } : [ 0 , T ]  \mathcal { V } ^ { 2 }$ and consider the periodic unit cell with the same material fields V, E that enter the ϵ-scale problem (5). A periodic displacement correction allows the local strain to vary while preserving its prescribed spatial average $\overline { { e } } ( t )$ . Denote the displacement corrector by χ and the cell strain and stress fields by $e ^ { ( 0 \bar { ) } }$ and $\boldsymbol { \sigma } ^ { ( 0 ) }$ , respectively. They satisfy the time-domain unit-cell problem

$$
\eta ( y , t ) = \mathrm { s y m } \left( \nabla _ { y } \chi ( y , t ) \right) ,
$$

$$
y \in  { \mathbb { T } } ^ { d } , \ t \in [ 0 , T ] ,\tag{6a}
$$

$$
e ^ { ( 0 ) } ( y , t ) = \overline { { { e } } } ( t ) + \eta ( y , t ) ,
$$

$$
y \in  { \mathbb { T } } ^ { d } , t \in [ 0 , T ] ,\tag{6b}
$$

$$
\begin{array} { r } { \sigma ^ { ( 0 ) } ( y , t ) = \mathbb { V } ( y ) \partial _ { t } e ^ { ( 0 ) } ( y , t ) + \mathbb { E } ( y ) e ^ { ( 0 ) } ( y , t ) , } \end{array}
$$

$$
y \in  { \mathbb { T } } ^ { d } , \ t \in [ 0 , T ] ,\tag{6c}
$$

$$
- \boldsymbol { \nabla } _ { y } \cdot \boldsymbol { \sigma } ^ { ( 0 ) } ( y , t ) = 0 ,
$$

$$
y \in  { \mathbb { T } } ^ { d } , t \in [ 0 , T ] ,\tag{6d}
$$

$$
\chi ( y , 0 ) = \chi _ { 0 } ( y ) ,
$$

$$
\boldsymbol { y } \in \mathbb { T } ^ { d } ,\tag{6e}
$$

$$
\int _ {  { \mathbb { T } } ^ { d } } \chi ( y , t ) \mathrm { d } y = 0 ,
$$

$$
t \in [ 0 , T ] ,\tag{6f}
$$

Here, χ<sub>0</sub> is chosen as the solution of the static elastic unit-cell problem corresponding to $\overline { { e } } ( 0 )$ . We call η the strain corrector. ◇

Homogenized Model and the Macroscopic Problem. The averaged unit-cell stress defines the constitutive map by

$$
\overline { { \sigma } } ( t ) : = \int _ { \mathbb { T } ^ { d } } \sigma ^ { ( 0 ) } ( y , t ) \mathrm { d } y = \mathcal { H } \big ( \{ \overline { { e } } ( s ) \} _ { s \in [ 0 , t ] } ; \mathbb { V } , \mathbb { E } \big ) .\tag{7}
$$

This is the small-strain analog of (1); see Remark 1.1. Solving (6) for sampled strain histories and material fields produces the stress trajectories used to train the neural operator model in Section 5.1. The homogenization theorem in Section 3 identifies the constitutive law obtained as $\epsilon \to 0$ in (5). With the initialization specified above, Appendix A.3 shows that the averaged response (7) coincides with that law.

Let $u ^ { ( 0 ) } : \Omega \times [ 0 , T ] \to \mathbb { R } ^ { d }$ denote the homogenized displacement. At each macroscopic point x, its strain history drives the unit-cell response in (7). Coupling this constitutive response to macroscopic compatibility and force balance gives the following problem:

$$
\overline { { e } } ( x , t ) = \mathrm { s y m } \left( \nabla _ { x } u ^ { ( 0 ) } ( x , t ) \right) ,
$$

$$
x \in \Omega , t \in [ 0 , T ] ,\tag{8a}
$$

$$
\boldsymbol { \overline { { \sigma } } } ( \boldsymbol { x } , t ) = \mathcal { H } \big ( \{ \boldsymbol { \overline { { e } } } ( \boldsymbol { x } , s ) \} _ { s \in [ 0 , t ] } ; \mathbb { V } , \mathbb { E } \big ) ,
$$

$$
x \in \Omega , t \in [ 0 , T ] ,\tag{8b}
$$

$$
- \nabla _ { x } \cdot \overline { { \sigma } } ( x , t ) = f ( x , t ) ,
$$

$$
x \in \Omega , t \in [ 0 , T ] ,\tag{8c}
$$

$$
u ^ { ( 0 ) } ( x , 0 ) = u _ { 0 } ^ { ( 0 ) } ( x ) ,
$$

$$
x \in \Omega ,\tag{8d}
$$

$$
u ^ { ( 0 ) } ( x , t ) = 0 ,
$$

$$
x \in \partial \Omega , t \in [ 0 , T ] ,\tag{8e}
$$

The unit-cell fields depend parametrically on x through the strain history $\overline { { e } } ( x , \circ )$ ; this parameter is suppressed in (6) and (7), because  acts pointwise in space. The homogenized initial condition $u _ { 0 } ^ { ( 0 ) }$ is specified in Theorem 3.1. ◇

## 2.3. Viscoelastic Neural Operator Constitutive Model

Since  is defined pointwise in space, we suppress x and define the neural operator model $\mathcal { H } _ { \mathrm { N O } }$ that approximates  in (7) at a given material point. Specializing (2) to the small strain setting, noting Remark 1.1, we consider

$$
\begin{array} { r } { \overline { { \sigma } } _ { \mathrm { N O } } ( t ) = \mathcal { H } _ { \mathrm { N O } } ( \{ \overline { { e } } ( s ) \} _ { s \in [ 0 , t ] } ; \mathbb { V } , \mathbb { E } ) , } \end{array}\tag{9}
$$

defined through the dynamical system

$$
\overline { { \sigma } } _ { \mathrm { N O } } ( t ) = \mathcal { F } _ { \mathrm { N O } } ( \overline { { e } } ( t ) , \dot { \overline { { e } } } ( t ) , \xi ( t ) ; \mathbb { V } , \mathbb { E } ) ,
$$

$$
t \in [ 0 , T ] ,\tag{10a}
$$

$$
\dot { \xi } ( t ) = \mathcal { G } _ { \mathrm { N O } } ( \dot { \overline { { e } } } ( t ) , \xi ( t ) ; \mathbb { V } , \mathbb { E } ) ,
$$

$$
t \in [ 0 , T ] ,\tag{10b}
$$

$$
\xi ( 0 ) = 0 ,\tag{10c}
$$

where $\xi : [ 0 , T ]  \mathbb { R } ^ { n _ { \xi } }$ is an internal variable, and only the strain rate enters the kinetic response $\mathcal { G } _ { \mathrm { N O } }$ . The choice of inputs to $\mathcal { F } _ { \mathrm { N O } }$ and $\mathcal { G } _ { \mathrm { N O } }$ made here reflects the structure of the Kelvin–Voigt homogenized model identified in Section 3. For other materials, diferent choices may be appropriate: finite strain may be relevant, the kinetic response (10b) may also depend on strain, or both (10a) and (10b) may depend only on strain and not strain rate [12].

We note that the response mappings defining (10) depend on functions summarizing material properties. To handle this, we use neural operator concepts. As well as operating on functions, the chosen neural operator must also allow Euclidean vector-to-function lifting and function-to-vector projection layers to accommodate the dependence on strain, strain rate, and internal variables. In this work, we use the Fourier neural mapping (FNM) [47] to achieve this.

Example 2.1 (Fourier Neural Mapping). Let $\mathsf { v e c } _ { 2 } : \mathcal { V } ^ { 2 } \to \mathbb { R } ^ { d _ { \mathcal { V } ^ { 2 } } }$ and $\mathsf { v e c } _ { 4 } : \mathcal { V } ^ { 4 } \to \mathbb { R } ^ { d _ { \mathcal { V } ^ { 4 } } }$ denote the Mandel vectorizations [48], which are isometric isomorphisms, and denote $d _ { \nu ^ { 2 } } =$ $d ( d + 1 ) / 2$ and $d _ { \mathcal { V } ^ { 4 } } = d _ { \mathcal { V } ^ { 2 } } ( d _ { \mathcal { V } ^ { 2 } } + 1 ) / 2$ . We parameterize $\mathcal { F } _ { \mathrm { N O } }$ and $\mathcal { G } _ { \mathrm { N O } }$ by two FNM instances $\Phi _ { \mathcal { F } }$ and $\Phi _ { \mathcal { G } } \mathrm { : }$

$$
{ \mathcal { F } } _ { \mathrm { N O } } = { \tt v e c } _ { 2 } ^ { - 1 } \circ \Phi _ { { \mathcal { F } } } \left( \left( { \tt v e c } _ { 2 } { \overline { { e } } } , { \tt v e c } _ { 2 } { \dot { \overline { { e } } } } , \xi \right) , \left( { \tt v e c } _ { 4 } \circ \mathbb { V } , { \tt v e c } _ { 4 } \circ \mathbb { E } \right) \right) ,\tag{11a}
$$

$$
\mathcal { G } _ { \mathrm { N O } } = \Phi _ { \mathcal { G } } \left( \left( \mathsf { v e c } _ { 2 } \dot { \overline { { e } } } , \xi \right) , ( \mathsf { v e c } _ { 4 } \circ \mathbb { V } , \mathsf { v e c } _ { 4 } \circ \mathbb { E } ) \right) .\tag{11b}
$$

Each FNM Φ takes a vector input $v _ { \mathrm { i n } }$ and a function input $M = ( \mathsf { v e c } _ { 4 } \circ \mathbb { V } , \mathsf { v e c } _ { 4 } \circ \mathbb { E } ) : \mathbb { T } ^ { d } $ $\mathbb { R } ^ { 2 d _ { \nu ^ { 4 } } }$ and returns $v _ { \mathrm { o u t } } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } }$ . It has three components:

(i) Lifting: A trainable pointwise map $S : \mathbb { R } ^ { d _ { \mathrm { i n } } + 2 d _ { \nu ^ { 4 } } } \to \mathbb { R } ^ { d _ { h ^ { ( 0 ) } } }$ defines $h ^ { ( 0 ) } ( y ) = S ( v _ { \mathrm { i n } } , M ( y ) )$ .

(ii) Fourier layers: For $\ell = 1 , \ldots , L$ , the layer $\mathcal { L } _ { \ell }$ maps $h ^ { ( \ell - 1 ) }$ to

$$
h ^ { ( \ell ) } ( y ) = \sigma _ { \mathrm { a c t } } ^ { ( \ell ) } \Bigl ( W _ { \ell } h ^ { ( \ell - 1 ) } ( y ) + \sum _ { k \in \Lambda _ { \ell } } P _ { \ell } ^ { ( k ) } \widehat { h ^ { ( \ell - 1 ) } } ( k ) e ^ { 2 \pi \mathrm { i } k ^ { \top } y } + b _ { \ell } \Bigr ) ,
$$

where $\sigma _ { \mathrm { a c t } } ^ { ( \ell ) }$ is the activation function at layer $\ell , \widehat { ( \circ ) }$ denotes the Fourier transform, $W _ { \ell }$ and $b _ { \ell }$ care trainable weights and biases applied pointwise, $P _ { \ell } ^ { ( k ) }$ are trainable Fourierspace weights satisfying conjugate symmetry, and $\Lambda _ { \ell } = \{ k \in \mathbb { Z } ^ { d } : | k _ { j } | \leq m _ { \ell , j } \}$ is the set of the retained Fourier modes.

(iii) Projection: A Fourier-space projection $\begin{array} { r } { \mathcal { G } h ^ { ( L ) } = \sum _ { k \in \Lambda _ { \mathrm { p r o j } } } P _ { \mathrm { p r o j } } ^ { ( k ) } \widehat { h ^ { ( L ) } } ( k ) \in \mathbb { R } ^ { d _ { \mathrm { p r o j } } } } \end{array}$ is followed by a trainable map $Q : \mathbb { R } ^ { d _ { \mathrm { p r o j } } }  \mathbb { R } ^ { d _ { \mathrm { o u t } } }$ . The weights $P _ { \mathrm { p r o j } } ^ { ( k ) }$ satisfy conjugate symmetry.

The resulting map is $\Phi ( v _ { \mathrm { i n } } , M ) = ( Q \circ \mathcal { G } \circ \mathcal { L } _ { L } \circ \cdots \circ \mathcal { L } _ { 1 } ) ( h ^ { ( 0 ) } )$

![](images/ad339d928ddc31034d2f686984b5deed17aadd29c151eeac2dfa6360fdc39c6a.jpg)  
Figure 2: Fourier neural mappings for the viscoelastic constitutive model. At each unit-cell spatial point and time, we combine the relevant macroscopic states and material fields to form spatial features, process them with Fourier layers, and project them to an output. Two separately parameterized instances predict macroscopic stress and the internal-variable rate in (11). Tensor vectorizations are omitted in the schematic.

## 3. Homogenization Theory and Memory Kernel

We revisit the viscoelastic homogenization theory of Francfort and Suquet [39] and identify the homogenized model  with fading memory in a form suitable for analysis. To identify the homogenization limit of the ϵ-scale problem and justify the constitutive response introduced in Section 2.2, we use the Laplace transform to reduce the evolution equation to a family of static elliptic problems to which classical periodic homogenization applies. In Section 3.1, we introduce the functional setting. In Section 3.2, we define the bounded linear operators that govern . In Section 3.3, we state the homogenization theorem and sketch its proof. In Section 3.4, we present the spectral representation of the memory kernel that arises from homogenization.

## 3.1. Functional Setting

We use standard notation for Lebesgue spaces $L ^ { p }$ , Sobolev spaces $W ^ { k , p }$ (with $H ^ { k } : =$ $W ^ { k , 2 } )$ ). Periodic function spaces on $\mathbb { T } ^ { d }$ carry the subscript per. The space $H _ { 0 } ^ { 1 } ( \Omega ;  \mathbb { R } ^ { d } )$ denotes functions with homogeneous Dirichlet boundary conditions.

Let X and Y be Banach spaces. We denote Bochner and Sobolev spaces for time-evolving functions as $L ^ { p } ( 0 , T ; X )$ and $W ^ { k , p } ( 0 , T ; X )$ (with $H ^ { k } : = W ^ { k , 2 } )$ . We write ${ \mathcal { L } } ( X , Y )$ for the space of bounded linear operators from X to Y .

For $\mathbb { A } \in L _ { \mathrm { p e r } } ^ { \infty } ( \mathbb { T } ^ { d } ; \mathcal { V } _ { + } ^ { 4 } )$ , we define A-weighted inner products $( e _ { 1 } , e _ { 2 } ) _ { \mathbb { A } } : = ( \mathbb { A } e _ { 1 } , e _ { 2 } ) _ { L _ { \mathrm { p e r } } ^ { 2 } }$ on $L _ { \mathrm { p e r } } ^ { 2 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 2 } )$ , with induced norms equivalent to the base norm. For a macroscopic tensor field $\mathbb { B } ^ { \cdot } \in L ^ { \infty } ( \Omega ; \mathcal { V } _ { + } ^ { 4 } )$ , we define $\begin{array} { r } { ( u , v ) _ { \mathbb { B } , \mathrm { s y m } } : = ( \mathbb { B } \mathrm { s y m } ( \nabla _ { x } u ) , \mathrm { s y m } ( \nabla _ { x } v ) ) _ { L ^ { 2 } } } \end{array}$ on $H _ { 0 } ^ { 1 } ( \Omega ;  \mathbb { R } ^ { d } )$ , which is an equivalent inner product by Korn’s inequality. Recall that the Frobenius inner product is used for second- and fourth-order tensors.

The displacement corrector in the unit-cell problem is sought in $\mathring { H } _ { \mathrm { p e r } } ^ { 1 } ( \mathbb { T } ^ { d } ; \mathbb { R } ^ { d } )$ , the subspace of $H _ { \mathrm { p e r } } ^ { 1 } ( \mathbb { T } ^ { d } ; \mathbb { R } ^ { d } )$ with zero mean. The strain corrector space is

$$
\begin{array} { r } { \mathcal { X } : = \left\{ \mathrm { s y m } ( \nabla _ { y } \chi ) \in L _ { \mathrm { p e r } } ^ { 2 } ( \mathbb { T } ^ { d } ; \mathcal { V } ^ { 2 } ) \mid \chi \in \mathring { H } _ { \mathrm { p e r } } ^ { 1 } ( \mathbb { T } ^ { d } ; \mathbb { R } ^ { d } ) \right\} , } \end{array}\tag{12}
$$

a closed subspace of $L _ { \mathrm { p e r } } ^ { 2 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 2 } )$ by Korn’s inequality [49]. The space  contains no nonzero spatially constant tensors.

## 3.2. Continuous Linear Operators for the Unit-Cell Problem

We define the operators governing the form of , assuming $\mathbb { V } , \mathbb { E } \in L _ { \mathrm { p e r } } ^ { \infty } ( \mathbb { T } ^ { d } ; \mathcal { V } _ { + } ^ { 4 } )$ . Whenever a macroscopic tensor $\overline { { e } } \in \mathcal { V } ^ { 2 }$ is an input to a function-space operator, we identify it with the constant field $y \mapsto \overline { e }$ on $\mathbb { T } ^ { d }$ . Table 1 summarizes these operators.

Strain Corrector Projectors. The orthogonal projectors $\mathcal { P } _ { \mathbb { V } } , \mathcal { P } _ { \mathbb { E } } \in \mathcal { L } ( L _ { \mathrm { p e r } } ^ { 2 } ( \mathbb { T } ^ { d } ; \mathcal { V } ^ { 2 } ) , \mathcal { X } )$ are defined by

$$
( \mathcal { P } _ { \mathbb { V } } e , \eta ) _ { \mathbb { V } } = ( e , \eta ) _ { \mathbb { V } } , \quad ( \mathcal { P } _ { \mathbb { E } } e , \eta ) _ { \mathbb { E } } = ( e , \eta ) _ { \mathbb { E } } \quad \forall \eta \in \mathcal { X } .\tag{13}
$$

For a static macroscopic strain $\overline { { e } } \in \mathcal { V } ^ { 2 }$ $\mathcal { P } _ { \mathbb { V } } \overline { { \it e } }$ and $\mathcal { P } _ { \mathbb { E } } \overline { { e } }$ are the negative strain correctors solving the purely viscous or elastic static unit-cell problems. For example, the action of the viscous projector $\mathcal { P } _ { \mathbb { V } }$ on e is defined by (14a) where $\chi \in \mathring { H } _ { \mathrm { p e r } } ^ { 1 } (  { \mathbb { T } } ^ { d } ;  { \mathbb { R } } ^ { d } )$ is the weak solution of (14b) and (14c):

$$
[ \mathcal { P } _ { \mathbb { V } } \overline { { e } } ] ( y ) = - \mathrm { s y m } ( \nabla _ { y } \chi ( y ) ) ,
$$

$$
\boldsymbol { y } \in \mathbb { T } ^ { d } ,\tag{14a}
$$

$$
\nabla _ { y } \cdot \mathbb { V } ( y ) \mathrm { s y m } ( \nabla _ { y } \chi ( y ) ) = - \nabla _ { y } \cdot \mathbb { V } ( y ) { \overline { { e } } } ,
$$

$$
\boldsymbol { y } \in \mathbb { T } ^ { d } ,\tag{14b}
$$

$$
\int _ {  { \mathbb { T } } ^ { d } } \chi ( y ) \mathrm { d } y = 0 .\tag{14c}
$$

where $\chi :  { \mathbb { T } ^ { d } } \to  { \mathbb { R } ^ { d } }$ is the displacement corrector given the macroscopic strain $\overline { { e } } \in \mathcal { V } ^ { 2 }$ Replacing V by E gives the corresponding elastic cell problem. The tensors V<sup>hom</sup> and E<sup>hom</sup> are the efective tensors obtained from these separate static problems by standard periodic elliptic homogenization. ◇

Strain Localization and Mismatch Operators. The complementary projectors $\mathcal { P } _ { \mathbb { V } } ^ { \perp } = \operatorname { I d } _ { L _ { \mathrm { p e r } } ^ { 2 } } -$ $\mathcal { P } _ { \mathbb { V } }$ and $\mathcal { P } _ { \mathbb { E } } ^ { \perp } = \mathrm { I d } _ { L _ { \mathrm { n e r } } ^ { 2 } } - \mathcal { P } _ { \mathbb { E } }$ return the corrected microscopic strain fields, where Id denotes the identity on $L _ { \mathrm { p e r } } ^ { 2 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 2 } )$ . The mismatch operator $\mathcal { D } _ { \mathbb { V } , \mathbb { E } } : = \mathcal { P } _ { \mathbb { V } } - \mathcal { P } _ { \mathbb { E } }$ quantifies the incompatibility between viscous flow and elastic equilibrium and drives memory efects in the homogenized model. ◇

Relaxation Operator. The operator $\mathcal { R } _ { \mathbb { V } , \mathbb { E } } \in \mathcal { L } ( \mathcal { X } )$ defined by

$$
( \mathcal { R } _ { \mathbb { V } , \mathbb { E } } \eta _ { 1 } , \eta _ { 2 } ) _ { \mathbb { V } } = ( \eta _ { 1 } , \eta _ { 2 } ) _ { \mathbb { E } } \quad \forall \eta _ { 1 } , \eta _ { 2 } \in \mathcal { X }\tag{15}
$$

is the Riesz representation map associated with $( \circ , \circ ) _ { \mathbb { E } }$ relative to $( \circ , \circ ) _ { \mathbb { V } }$ . Equivalently, $\mathcal { R } _ { \mathbb { V } , \mathbb { E } } = \left. \mathcal { P } _ { \mathbb { V } } W _ { \mathbb { V } , \mathbb { E } } \right| _ { \mathcal { X } }$ , where $W _ { \mathbb { V } , \mathbb { E } }$ denotes pointwise multiplication by $\mathbb { V } ^ { - 1 } \mathbb { E }$ . It governs relaxation dynamics by balancing elastic restoring forces against viscous resistance.

Given $\boldsymbol { w } : \mathbb { T } ^ { d }  \mathbb { R } ^ { d }$ in $H _ { \mathrm { p e r } } ^ { 1 } ( \mathbb { T } ^ { d } ; \mathbb { R } ^ { d } )$ with $\begin{array} { r } { \int _ {  { \mathbb { T } } ^ { d } } w ( y ) \mathrm { d } y = 0 } \end{array}$ , it is defined by (16a) where $v :  { \mathbb { T } } ^ { d } \to  { \mathbb { R } } ^ { d }$ is the weak solution of (16b) and (16c):

$$
[ \mathcal { R } _ { \mathbb { V } , \mathbb { E } ^ { \mathrm { S y m } } } ( \nabla _ { y } w ) ] ( y ) = \mathrm { s y m } ( \nabla _ { y } v ( y ) ) ,
$$

$$
\boldsymbol { y } \in \mathbb { T } ^ { d } ,\tag{16a}
$$

$$
\nabla _ { y } \cdot \mathbb { V } ( y ) \mathrm { s y m } ( \nabla _ { y } v ( y ) ) = \nabla _ { y } \cdot \mathbb { E } ( y ) \mathrm { s y m } ( \nabla _ { y } w ( y ) ) ,
$$

$$
\boldsymbol { y } \in \mathbb { T } ^ { d } ,\tag{16b}
$$

$$
\int _ {  { \mathbb { T } } ^ { d } } v ( y ) \mathrm { d } y = 0 .\tag{16c}
$$

The family $\{ \exp ( - t \mathcal { R } _ { \mathbb { V } , \mathbb { E } } ) \} _ { t \geq 0 }$ is a strongly continuous semigroup on $( \mathcal { X } , ( \circ , \circ ) _ { \mathbb { V } } )$ , generated by $- \mathcal { R } _ { \mathbb { V } , \mathbb { E } }$ . Note that the mapping from w to v given by (16b) and (16c) is defined by an operator $( \mathcal { L } _ { \mathbb { V } } ) ^ { - 1 } \mathcal { L } _ { \mathbb { E } }$ , with $\mathcal { L } _ { \mathbb { V } }$ and $\mathcal { L } _ { \mathbb { E } }$ elliptic operators; this is the cell problem analog of the family of operators $( \mathcal { L } _ { \mathbb { V } } ^ { \epsilon } ) ^ { - 1 } \mathcal { L } _ { \mathbb { E } } ^ { \epsilon }$ discussed in Remark 1.2 and the same remarks concerning boundedness apply here. ◇

## 3.3. Viscoelastic Homogenization with Fading Memory

We identify the homogenized constitutive model for Kelvin–Voigt viscoelasticity. We first define the weak ϵ-scale problem and admissible initial conditions, then state the homogenization theorem adapted from [39] and sketch its proof.

Weak ϵ-Scale Problem. Given $\mathbb { V } , \mathbb { E } \in L _ { \mathrm { o e r } } ^ { \infty } ( \mathbb { T } ^ { d } ; \mathcal { V } _ { + } ^ { 4 } ) , f \in L ^ { 2 } ( 0 , T ; L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) )$ , and initial condition $u _ { 0 } ^ { \epsilon }$ , find $u ^ { \epsilon } \in H ^ { 1 } ( 0 , T ; H _ { 0 } ^ { 1 } ( \Omega ; \mathbb { R } ^ { d } ) )$ such that $u ^ { \epsilon } ( 0 ) = u _ { 0 } ^ { \epsilon }$ in $H _ { 0 } ^ { 1 } ( \Omega ;  \mathbb { R } ^ { d } )$ and, for a.e. $t \in [ 0 , T ]$ 2

$$
( \dot { u } ^ { \epsilon } ( t ) , v ) _ { \mathbb { V } ^ { \epsilon } , \operatorname { s y m } } + ( u ^ { \epsilon } ( t ) , v ) _ { \mathbb { E } ^ { \epsilon } , \operatorname { s y m } } = ( f ( t ) , v ) _ { L ^ { 2 } } \quad \forall v \in H _ { 0 } ^ { 1 } ( \Omega ; \mathbb { R } ^ { d } ) .\tag{17}
$$

<table><tr><td rowspan=1 colspan=1>Name</td><td rowspan=1 colspan=1>Input</td><td rowspan=1 colspan=1>Output</td><td rowspan=1 colspan=1>PhysicalInterpretation</td><td rowspan=1 colspan=1>MathematicalInterpretation</td></tr><tr><td rowspan=1 colspan=1>Strain CorrectorProjectors $\mathcal { P } _ { \mathbb { V } }$ and $\mathcal { P } _ { \mathbb { E } }$ </td><td rowspan=1 colspan=1> $L _ { \mathrm { p e r } } ^ { 2 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 2 } )$ </td><td rowspan=1 colspan=1> $\mathcal { X }$ </td><td rowspan=1 colspan=1>Identify purelyviscous or elasticmicroscopiccorrector (negative)</td><td rowspan=1 colspan=1>OrthogonalProjectors</td></tr><tr><td rowspan=1 colspan=1>StrainLocalization $\mathcal { P } _ { \mathbb { V } } ^ { \perp }$ and $\mathcal { P } _ { \mathbb { E } } ^ { \perp }$ </td><td rowspan=1 colspan=1> $L _ { \mathrm { p e r } } ^ { 2 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 2 } )$ </td><td rowspan=1 colspan=1> $L _ { \mathrm { p e r } } ^ { 2 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 2 } )$ </td><td rowspan=1 colspan=1>Determine purelyviscous or elasticmicroscopic strain</td><td rowspan=1 colspan=1>ComplementaryProjectors</td></tr><tr><td rowspan=1 colspan=1>Strain CorrectorMismatch $\mathcal { D } _ { \mathbb { V } , \mathbb { E } }$ </td><td rowspan=1 colspan=1> $L _ { \mathrm { p e r } } ^ { 2 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 2 } )$ </td><td rowspan=1 colspan=1> $\mathcal { X }$ </td><td rowspan=1 colspan=1>Difference betweenpurely viscous andelastic microscopiccorrectors</td><td rowspan=1 colspan=1>Difference ofProjectors</td></tr><tr><td rowspan=1 colspan=1>Relaxation $\mathcal { R } _ { \mathbb { V } , \mathbb { E } }$ </td><td rowspan=1 colspan=1> $\mathcal { X }$ </td><td rowspan=1 colspan=1> $\mathcal { X }$ </td><td rowspan=1 colspan=1>Dictates therelaxationtimescales</td><td rowspan=1 colspan=1>RieszRepresentationMap</td></tr></table>

Table 1: Continuous linear operators governing the homogenized constitutive model.

Admissible Initial Conditions. A family $\{ u _ { 0 } ^ { \epsilon } \} _ { \epsilon > 0 } \subset H _ { 0 } ^ { 1 } ( \Omega ; \mathbb { R } ^ { d } )$ is admissible if there exists $g _ { 0 } \in L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } )$ such that each $u _ { 0 } ^ { \epsilon }$ uniquely solves $( u _ { 0 } ^ { \epsilon } , v ) _ { \mathbb { E } ^ { \epsilon } , \mathrm { s y m } } = ( g _ { 0 } , v ) _ { L ^ { 2 } }$ for all $v \in$ $H _ { 0 } ^ { 1 } ( \Omega ;  \mathbb { R } ^ { d } )$ ). ◇

Theorem 3.1 (Viscoelastic Homogenization). Suppose Ω is Lipschitz, $\{ u _ { 0 } ^ { \epsilon } \} _ { \epsilon > 0 }$ is an admissible family generated by $g _ { 0 } \in L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) , \mathbb { V } , \mathbb { E } \in L _ { \mathrm { p e r } } ^ { \infty } ( \mathbb { T } ^ { d } ; \mathcal { V } _ { + } ^ { 4 } )$ , and $f \in L ^ { 2 } ( 0 , T ; L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) )$ Define the homogenized viscosity and elasticity $\mathbb { V } ^ { \mathrm { h o i n } } ( \mathbb { V } ) , \mathbb { E } ^ { \mathrm { h o m } } ( \mathbb { E } ) \in \mathcal { V } _ { + } ^ { 4 }$ through the symmetric positive-definite bilinear forms:

$$
\left( \boldsymbol { \bar { e } } _ { 1 } , \mathbb { V } ^ { \mathrm { h o m } } ( \mathbb { V } ) \bar { \boldsymbol { e } } _ { 2 } \right) _ { F } = \left( \boldsymbol { \bar { e } } _ { 1 } , \mathcal { P } _ { \mathbb { V } } ^ { \perp } \boldsymbol { \bar { e } } _ { 2 } \right) _ { \mathbb { V } } , \quad \left( \boldsymbol { \bar { e } } _ { 1 } , \mathbb { E } ^ { \mathrm { h o m } } ( \mathbb { E } ) \boldsymbol { \bar { e } } _ { 2 } \right) _ { F } = \left( \boldsymbol { \bar { e } } _ { 1 } , \mathcal { P } _ { \mathbb { E } } ^ { \perp } \boldsymbol { \bar { e } } _ { 2 } \right) _ { \mathbb { E } } \quad \forall \boldsymbol { \bar { e } } _ { 1 } , \boldsymbol { \bar { e } } _ { 2 } \in \mathcal { V } ^ { 2 } ,\tag{18}
$$

and for each $t \geq 0$ , define the memory kernel $\mathbb { K } ( t ; \mathbb { V } , \mathbb { E } ) \in \mathcal { V } ^ { 4 }$ through the symmetric positive semidefinite bilinear form:

$$
\left( \overline { { e } } _ { 1 } , \mathbb { K } ( t ; \mathbb { V } , \mathbb { E } ) \overline { { e } } _ { 2 } \right) _ { F } = \left( \mathcal { R } _ { \mathbb { V } , \mathbb { E } } \exp \left( - t \mathcal { R } _ { \mathbb { V } , \mathbb { E } } \right) \mathcal { D } _ { \mathbb { V } , \mathbb { E } } \overline { { e } } _ { 1 } , \mathcal { D } _ { \mathbb { V } , \mathbb { E } } \overline { { e } } _ { 2 } \right) _ { \mathbb { V } } \quad \forall \overline { { e } } _ { 1 } , \overline { { e } } _ { 2 } \in \mathcal { V } ^ { 2 } .\tag{19}
$$

The unique weak solution $u ^ { \epsilon }$ of (17) converges weakly in $H ^ { 1 } ( 0 , T ; H _ { 0 } ^ { 1 } ( \Omega ; \mathbb { R } ^ { d } ) )$ to $u ^ { ( 0 ) }$ , the unique weak solution of the macroscopic problem (8), closed by the homogenized model with memory

$$
\mathcal { H } ( \{ \overline { { e } } ( s ) \} _ { s \in [ 0 , t ] } ; \mathbb { V } , \mathbb { E } ) : = \mathbb { V } ^ { \mathrm { h o m } } ( \mathbb { V } ) \dot { \overline { { e } } } ( t ) + \mathbb { E } ^ { \mathrm { h o m } } ( \mathbb { E } ) \overline { { e } } ( t ) + \int _ { 0 } ^ { t } \mathbb { K } ( t - s ; \mathbb { V } , \mathbb { E } ) \dot { \overline { { e } } } ( s ) \mathrm { d } s ,\tag{20}
$$

with homogenized initial condition $u _ { 0 } ^ { ( 0 ) }$ solving $( u _ { 0 } ^ { ( 0 ) } , v ) _ { \mathbb { E } ^ { \mathrm { h o m } } ( \mathbb { E } ) , \mathrm { s y m } } = ( g _ { 0 } , v ) _ { L ^ { 2 } }$ for all $v \in$ $H _ { 0 } ^ { 1 } ( \Omega ;  \mathbb { R } ^ { d } )$

Proof Sketch. We follow [39, Theorem 3.1]. First, consider $f ~ = ~ 0$ For admissible initial data, the semigroup estimate [39, (3.20)] gives an ϵ-independent bound for $u ^ { \epsilon }$ in $W ^ { 1 , \infty } ( 0 , \infty ; H _ { 0 } ^ { 1 } ( \Omega ; \mathbb { R } ^ { d } ) )$ . After passing to a subsequence, $u ^ { \epsilon }$ converges in weak-\* to $u ^ { ( 0 ) }$ on this space. Their Laplace transforms therefore satisfy $\hat { u } ^ { \epsilon } ( s )  \hat { u } ^ { ( 0 ) } ( s )$ in $H _ { 0 } ^ { 1 } ( \Omega ;  \mathbb { R } ^ { d } )$ for every $s > 0$ . By admissibility, $\hat { v } ^ { \epsilon } ( s ) : = s \hat { u } ^ { \epsilon } ( s ) - u _ { 0 } ^ { \epsilon }$ solves a static elliptic problem with coeficient $\mathbb { A } _ { s } ^ { \epsilon } ( x ) : = s ^ { - 1 } \mathbb { E } ( x / \epsilon ) + \mathbb { V } ( x / \epsilon )$ and load $- g _ { 0 } / s$ . Periodic homogenization identifies the limits of these problems and of the initial data.

Next, we identify the constitutive law. Expressing the dissipative corrector of [39] in terms of $\mathcal { R } _ { \mathbb { V } , \mathbb { E } }$ and $\mathcal { D } _ { \mathbb { V } , \mathbb { E } }$ gives the kernel (19). The cell-problem identity [39, (3.12)] then reads $\begin{array} { r } { \mathbb { A } _ { s } ^ { \mathrm { { h o m } } } = s ^ { - 1 } \mathbb { E } ^ { \mathrm { { h o m } } } ( \mathbb { E } ) + \mathbb { V } ^ { \mathrm { { h o m } } } ( \mathbb { V } ) + \mathbb { \hat { K } } ( s ) } \end{array}$ , where K<sup>ˆ</sup> is the Laplace transform of K. Uniqueness of the homogenized elliptic problems and injectivity of the Laplace transform recover (20) and identify the limit uniquely, giving convergence of the whole family. We establish the properties of the homogenized tensors and kernel in Appendix $\mathrm { A . 1 }$

Finally, for general $f \in L ^ { 2 } ( 0 , T ; L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) )$ , Duhamel’s formula expresses the forcing contribution as a time integral of homogeneous evolutions whose initial data solve viscous elliptic problems. Applying the transformed elliptic argument to these evolutions and using the uniform semigroup bounds to pass to the limit under the integral yields the stated result on [0, T]; see [39, (3.34)–(3.39)]. □

Remark 3.2. The identity between the unit-cell representation (7) and the constitutive law (20) is proved in Appendix A.3 by writing the unit-cell problem as an ordinary diferential equation on .

Remark 3.3. The kernel representation shows that the mismatch operator $\mathcal { D } _ { \mathbb { V } , \mathbb { E } }$ drives microscopic strain relaxation, while $\mathcal { R } _ { \mathbb { V } , \mathbb { E } }$ determines its decay rates. The memory kernel vanishes when the viscous and elastic strain correctors coincide for every macroscopic strain. In particular, this holds when $\mathbb { V } = c \mathbb { E }$ for a spatially constant $c > 0$ ，

## 3.4. Spectral Representation of the Memory Kernel

The operator form of the memory kernel in (19) leads to a representation of K as a Laplace–Stieltjes integral over relaxation rates. The next proposition formalizes this representation together with two useful properties: the rates lie in an interval determined by the coercivity and boundedness constants of the microstructural viscosity and elasticity, and the total variation of the spectral measure is bounded in terms of the same constants.

Proposition 3.4 (Spectral Representation of the Memory Kernel). Let $\mathbb { V } , \mathbb { E } \in L _ { \mathrm { p e r } } ^ { \infty } ( \mathbb { T } ^ { d } ; \mathcal { V } _ { + } ^ { 4 } )$ ， and let $\alpha _ { \mathbb { V } } , \beta _ { \mathbb { V } } , \alpha _ { \mathbb { E } } , \beta _ { \mathbb { E } } > 0$ satisfy

$$
\begin{array} { r } { \alpha _ { \mathbb { V } } \| e \| _ { L _ { \mathrm { p e r } } ^ { 2 } } ^ { 2 } \le \| e \| _ { \mathbb { V } } ^ { 2 } \le \beta _ { \mathbb { V } } \| e \| _ { L _ { \mathrm { p e r } } ^ { 2 } } ^ { 2 } , \qquad \alpha _ { \mathbb { E } } \| e \| _ { L _ { \mathrm { p e r } } ^ { 2 } } ^ { 2 } \le \| e \| _ { \mathbb { E } } ^ { 2 } \le \beta _ { \mathbb { E } } \| e \| _ { L _ { \mathrm { p e r } } ^ { 2 } } ^ { 2 } } \end{array}\tag{21}
$$

for all $e \in L _ { \mathrm { p e r } } ^ { 2 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 2 } )$ . Then there exists a finite positive semidefinite $\mathcal { V } ^ { 4 }$ -valued Borel

measure $\nu _ { \mathbb { V } , \mathbb { E } }$ , supported on $\mathcal { T } _ { \mathbb { V } , \mathbb { E } } : = [ \alpha _ { \mathbb { E } } / \beta _ { \mathbb { V } } , \beta _ { \mathbb { E } } / \alpha _ { \mathbb { V } } ]$ , such that

$$
\mathbb { K } ( t ; \mathbb { V } , \mathbb { E } ) = \int _ { \mathbb { Z } _ { \mathbb { V } , \mathbb { E } } } \lambda \exp ( - \lambda t ) { \mathrm { d } \nu _ { \mathbb { V } , \mathbb { E } } ( \lambda ) } \quad \forall t \geq 0 ,\tag{22}
$$

$$
\| \nu _ { \mathbb { V } , \mathbb { E } } \| _ { \mathrm { T V } ( \mathbb { Z } _ { \mathbb { V } , \mathbb { E } } ) } \le d _ { \mathcal { V } ^ { 2 } } \frac { \beta _ { \mathbb { V } } \beta _ { \mathbb { E } } } { \alpha _ { \mathbb { E } } } ,\tag{23}
$$

where the total variation of the measure is defined as

$$
\| \nu _ { \mathbb { V } , \mathbb { E } } \| _ { \mathrm { T V } ( \mathcal { T } _ { \mathbb { V } , \mathbb { B } } ) } : = \operatorname* { s u p } \bigg \{ \sum _ { i = 1 } ^ { m } \| \nu _ { \mathbb { V } , \mathbb { E } } ( B _ { i } ) \| _ { F } \biggm | m \in \mathbb { N } , B _ { 1 } , \ldots , B _ { m } \ a r e \ B o r e l \ s u b s e t s \ o f \mathcal { T } _ { \mathbb { V } , \mathbb { B } } , 
$$

$$
B _ { i } \cap B _ { j } = \varnothing \ \forall i \neq j , \bigcup _ { i = 1 } ^ { m } B _ { i } = { \mathcal { I } } _ { \mathbb { V } , \mathbb { E } } \Bigg \} .\tag{24}
$$

The proof is given in Appendix A.2. The interval in the proposition gives the range of relaxation rates that are relevant to the memory kernel, while the total-variation norm bound controls the total amount of memory efects arising from homogenization. While the constants $\alpha _ { \mathbb { V } } , \beta _ { \mathbb { V } } , \alpha _ { \mathbb { E } } , \beta _ { \mathbb { E } }$ are properties of the specific microstructural instances, they are shared among a wide class of microstructures that satisfy (21). Theorem 3.4 shows that the spectral support and the total variation bound are common to this class; this is a key component of the approximation theory presented in the next section.

## 4. Universal Approximation and Internal-Variable Complexity

We now use the spectral representation of the memory kernel from Section 3.4 to address the approximation and internal-variable complexity questions for Kelvin–Voigt viscoelasticity. The key step is to introduce a Prony-series internal-variable model that approximates the homogenized constitutive law  in (20) uniformly over admissible microstructures and strain trajectories and depends continuously on the microstructure. This intermediate model links the fading-memory representation of  to the neural operator internal-variable model $\mathcal { H } _ { \mathrm { N O } }$ in (9).

In Section 4.1, we specify the admissible microstructures, strain trajectories, and neuraloperator response classes. In Section 4.2, we state the universal approximation theorem and sketch its proof, with the full argument deferred to Appendix B. In Section 4.3, we derive the suficient internal-variable dimension bound, and in Section 4.4, we discuss continuity constraints on microstructure-dependent relaxation rates.

In the weak macroscopic problem, the strain and stress are Bochner functions $\overline { { e } } , \overline { { \sigma } } :$ $[ 0 , T ]  L ^ { 2 } ( \Omega ; \mathcal { V } ^ { 2 } )$ . Since $\mathcal { H }$ is local in space, we pass to a material-point trajectory formulation: for a.e. $x \in \Omega$ , we write $\overline { { e } } , \overline { { \sigma } } : [ 0 , T ]  \mathcal { V } ^ { 2 }$ . Moreover, the constitutive law (20) is independent of the terminal time T. For the approximation results in this section, we take $\mathcal { T } = [ 0 , \infty )$ as the time domain for this law and the neural operator model (10).

## 4.1. Settings

We specify the admissible classes of microstructures and strain trajectories, together with the neural-operator response classes that enter the approximation theorem.

Admissible Microstructural Properties. For $\mathbb { A } \in L _ { \mathrm { p e r } } ^ { 1 } ( \mathbb { T } ^ { d } ; \mathcal { V } ^ { 4 } )$ , we define the total variation seminorm by duality [50],

$$
| \mathbb { A } | _ { \mathrm { T V } } : = \operatorname* { s u p } \left\{ \int _ { \mathbb { T } ^ { d } } \left( \mathbb { A } ( y ) , \nabla _ { y } \cdot \phi ( y ) \right) _ { F } \mathrm { d } y \ \middle | \ \phi \in C _ { \mathrm { p e r } } ^ { 1 } \left( \mathbb { T } ^ { d } ; \mathcal { V } ^ { 4 } \otimes \mathbb { R } ^ { d } \right) , \ \Vert \phi \Vert _ { L _ { \mathrm { p e r } } ^ { \infty } } \leq 1 \right\} ,\tag{25}
$$

and we set $\| \mathbb { A } \| _ { \mathrm { B V } } : = \| \mathbb { A } \| _ { L _ { \mathrm { p e r } } ^ { 1 } } + | \mathbb { A } | _ { \mathrm { T V } }$ and $\mathrm { B V } : = \{ \mathbb { A } \in L _ { \mathrm { p e r } } ^ { 1 } ( \mathbb { T } ^ { d } ; \mathcal { V } ^ { 4 } ) \mid \| \mathbb { A } \| _ { \mathrm { B V } } < \infty \}$ . Given constants $\alpha _ { \mathrm { v i s c } } , \beta _ { \mathrm { v i s c } } , \gamma _ { \mathrm { v i s c } } , \alpha _ { \mathrm { e l a s } } , \beta _ { \mathrm { e l a s } } , \gamma _ { \mathrm { e l a s } } > 0$ , the admissible set

$$
\begin{array} { r } { \mathcal { M } _ { \alpha _ { \mathrm { v i s c } } , \beta _ { \mathrm { v i s c } } , \gamma _ { \mathrm { v i s c } } } \subset \left( \mathrm { B V } \cap L _ { \mathrm { p e r } } ^ { \infty } ( \mathbb { T } ^ { d } ; \mathcal { V } ^ { 4 } ) \right) ^ { 2 } } \\ { \alpha _ { \mathrm { e l a s } } , \beta _ { \mathrm { e l a s } } , \gamma _ { \mathrm { e l a s } } } \end{array}\tag{26}
$$

consists of pairs (V, E) satisfying:

$$
\alpha _ { \mathrm { v i s c } } \| e \| _ { L _ { \mathrm { p e r } } ^ { 2 } } ^ { 2 } \leq \| e \| _ { \mathbb { V } } ^ { 2 } \leq \beta _ { \mathrm { v i s c } } \| e \| _ { L _ { \mathrm { p e r } } ^ { 2 } } ^ { 2 } , \alpha _ { \mathrm { e l a s } } \| e \| _ { L _ { \mathrm { p e r } } ^ { 2 } } ^ { 2 } \leq \| e \| _ { \mathbb { E } } ^ { 2 } \leq \beta _ { \mathrm { e l a s } } \| e \| _ { L _ { \mathrm { p e r } } ^ { 2 } } ^ { 2 } ,\tag{27a}
$$

$$
\| \mathbb { V } \| _ { \mathrm { B V } } \leq \gamma _ { \mathrm { v i s c } } , \quad \| \mathbb { E } \| _ { \mathrm { B V } } \leq \gamma _ { \mathrm { e l a s } } .\tag{27b}
$$

for all $e \in L _ { \mathrm { p e r } } ^ { 2 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 2 } )$

Remark 4.1 (Bounded Variation). Bounded variation is a natural assumption on the material microstructure which gives compactness in $( L _ { \mathrm { p e r } } ^ { 2 } ) ^ { 2 }$ , but smoother classes $( e . g .$ , uniformly bounded in $W ^ { 1 , p } )$ also have this property. We consider this class of materials as it is natural for composite materials to exhibit discontinuous jumps in microstructural properties, but fractal-type oscillations are excluded. The bounded variation assumption is not needed in the proof of the spectral representation theorem above. However, the compactness it provides in $( L _ { \mathrm { p e r } } ^ { 2 } ) ^ { 2 }$ is essential for establishing the universal approximation theorem below; see Theorem B.2.

Admissible Strain Trajectories. Since the constitutive model depends on both e and $\dot { \bar { e } } ,$ we impose a uniform bound on both quantities. Let $\delta _ { \mathrm { s t r e s s } } > 0$ denote the prescribed stress scale. We define the set of admissible macroscopic strain trajectories by:

$$
\begin{array} { r } { \mathcal { E } _ { \delta _ { \mathrm { s t r e s s } } , \beta _ { \mathrm { v i s c } } , \beta _ { \mathrm { e l a s } } } : = \left\{ \bar { e } \in W ^ { 1 , \infty } ( \mathcal { T } ; \mathcal { V } ^ { 2 } ) \vert \beta _ { \mathrm { e l a s } } \Vert \bar { e } \Vert _ { L ^ { \infty } } + \beta _ { \mathrm { v i s c } } \Vert \dot { \bar { e } } \Vert _ { L ^ { \infty } } \leq \delta _ { \mathrm { s t r e s s } } \right\} . } \end{array}\tag{28}
$$

We emphasize that this admissible class excludes idealized step-strain tests.

Neural-Operator Response Classes. We state the theorem for an abstract class of neural operators, rather than for a particular architecture. For integers $d _ { \mathrm { i n } } , d _ { \mathrm { o u t } } \in \mathbb { N }$ , let

$$
\mathfrak { N } \subset \{ \Phi : \mathbb { R } ^ { d _ { \mathrm { i n } } } \times L _ { \mathrm { p e r } } ^ { 2 } ( \mathbb { T } ^ { d } ; \mathbb { R } ^ { 2 d _ { \nu ^ { 4 } } } )  \mathbb { R } ^ { d _ { \mathrm { o u t } } } \}
$$

be a class of continuous mappings with the following approximation properties.

(A1) Universal Approximation. For every compact ${ \mathcal { K } } _ { \mathrm { v e c } } \subset \mathbb { R } ^ { d _ { \mathrm { i n } } }$ , every compact ${ \kappa _ { \mathrm { f u n } } } \subset$ $L _ { \mathrm { p e r } } ^ { 2 } (  { \mathbb { T } } ^ { d } ;  { \mathbb { R } } ^ { 2 d _ { \nu ^ { 4 } } } )$ , every continuous $g : \mathcal { K } _ { \mathrm { v e c } } \times \mathcal { K } _ { \mathrm { f u n } } \to \mathbb { R } ^ { d _ { \mathrm { o u t } } }$ , and every $\varepsilon > 0$ , there exists $\Phi \in \mathfrak { N }$ such that $\begin{array} { r } { \operatorname* { s u p } _ { \mathcal { K } _ { \mathrm { v e c } } \times \mathcal { K } _ { \mathrm { f u n } } } \left. \left. \Phi - g \right. \right. \le \varepsilon } \end{array}$

(A2) Locally Lipschitz Universal Approximation. The class N satisfies (A1). In addition, the approximating Φ can be chosen so that, for every $M \in \kappa _ { \mathrm { f u n } }$ , the map $v \mapsto \Phi ( v , M )$ is locally Lipschitz.

For $n _ { \xi } \in \mathbb { N }$ , we define the stress and kinetic response classes as follows.

• Stress Response Class $\mathfrak { F } ( n _ { \xi } )$ . This class consists of all $\mathcal { F } _ { \mathrm { N O } }$ of the form (11a) with $\Phi _ { \mathcal { F } } \in \mathfrak { N } _ { \mathcal { F } }$ , where $\mathfrak { N } _ { \mathcal { F } }$ satisfies (A1) with $\left( d _ { \mathrm { i n } } , d _ { \mathrm { o u t } } \right) = \left( 2 d _ { \nu ^ { 2 } } + n _ { \xi } , d _ { \nu ^ { 2 } } \right)$

• Kinetic Response Class ${ \mathfrak { G } } ( n _ { \xi } )$ . This class consists of all $\mathcal { G } _ { \mathrm { N O } }$ of the form (11b) with $\Phi _ { \mathcal { G } } \in \mathfrak { N } _ { \mathcal { G } }$ , where $\mathfrak { N } _ { \mathcal { G } }$ satisfies (A2) with $\left( d _ { \mathrm { i n } } , d _ { \mathrm { o u t } } \right) = \left( d _ { \mathcal { V } ^ { 2 } } + n _ { \xi } , n _ { \xi } \right)$

Only the kinetic response requires (A2), as this local Lipschitz property is used to control the internal-variable evolution in (10). ◇

Remark 4.2 (Architecture). The set N contains the maps realizable by a chosen neuraloperator architecture as its weights and hyperparameters vary. Condition (A1) is an expressivity requirement: the architecture approximates continuous maps uniformly on compact sets [20]. Approximation results for a range of neural-operator architectures and function-space settings are available; see, e.g., [51, 52, 53, 54, 55, 34]. Condition (A2) strengthens (A1) by requiring local Lipschitz dependence on the vector input. In practice, (A2) holds when the relevant architectural components are locally Lipschitz, for example, when using ReLU or GELU activations.

Remark 4.3 (Fourier Neural Mapping). The FNM architecture in Section 2.3 satisfies both (A1) and (A2) when equipped with smooth, globally Lipschitz, non-polynomial activations. Universal approximation follows from [47] for linear lifting and projection maps S and Q, while the Lipschitz condition follows from the bounded linear layers and Lipschitz activations. If S and Q are instead chosen as feedforward neural networks, locally Lipschitz activations preserve (A2).

## 4.2. Universal Approximation Theorem

We state the main result, which establishes the existence of a neural operator constitutive model $\mathcal { H } _ { \mathrm { N O } }$ , as defined in (9), that approximates the homogenized constitutive law of Kelvin– Voigt viscoelasticity to any prescribed accuracy, uniformly in time and over a wide range of microstructural properties and strain trajectories.

Theorem 4.4 (Universal Approximation of Viscoelastic Homogenization). Let $d \in \{ 1 , 2 , 3 \}$ For any constants $\alpha _ { \mathrm { v i s c } } , \beta _ { \mathrm { v i s c } } , \gamma _ { \mathrm { v i s c } } , \alpha _ { \mathrm { e l a s } } , \beta _ { \mathrm { e l a s } } , \gamma _ { \mathrm { e l a s } } , \delta _ { \mathrm { s t r e s s } } > 0$ such that $0 < \alpha _ { \mathrm { e l a s } } \le \beta _ { \mathrm { e l a s } }$ and $0 < \alpha _ { \mathrm { v i s c } } \leq \beta _ { \mathrm { v i s c } }$ , let

$$
\begin{array} { r l } { \mathcal { M } : = \mathcal { M } _ { \alpha _ { \mathrm { v i s c } } , \beta _ { \mathrm { v i s c } } , \gamma _ { \mathrm { v i s c } } } , \qquad } & { \mathcal { E } : = \mathcal { E } _ { \delta _ { \mathrm { s t r e s s } } , \beta _ { \mathrm { v i s c } } , \beta _ { \mathrm { e l a s } } } . } \end{array}
$$

For any $\varepsilon > 0$ , there exist $n _ { \xi } \in \mathbb { N }$ and a pair of stress and kinetic response mappings

$$
( \mathcal { F } _ { \mathrm { N O } } , \mathcal { G } _ { \mathrm { N O } } ) \in \mathfrak { F } ( n _ { \xi } ) \times \mathfrak { G } ( n _ { \xi } ) ,
$$

such that

$$
\operatorname* { s u p } _ { ( \mathbb { V } , \mathbb { E } ) \in \mathcal { M } } \ \operatorname* { s u p } _ { \bar { \epsilon } \in \mathcal { E } } \ \operatorname* { s s s u p } _ { t \in \mathcal { T } } \ \big \| \mathcal { H } _ { \mathrm { N O } } ( \{ \overline { { e } } ( s ) \} _ { s \in [ 0 , t ] } ; \mathbb { V } , \mathbb { E } ) - \mathcal { H } ( \{ \overline { { e } } ( s ) \} _ { s \in [ 0 , t ] } ; \mathbb { V } , \mathbb { E } ) \big \| _ { F } \leq \varepsilon .\tag{29}
$$

Proof Sketch. The proof proceeds in four steps; full details are in Appendix B. The proof follows the approach introduced in [13, 23], but significantly extends it to facilitate approximation of the spectral measure $\nu _ { \mathbb { V } , \mathbb { E } }$ underlying the formula for the memory kernel appearing in Theorem 3.4; the preceding papers used approximation by piecewise-constant materials, whose memory kernels admit exact finite Prony-series representations in one dimension.

(S1) Establish the compactness of $\mathcal { M }$ in $( L _ { \mathrm { p e r } } ^ { 2 } ) ^ { 2 }$ and the continuity of the homogenized tensors with respect to $( \mathbb { V } , \mathbb { E } )$

(S2) Construct an intermediate Prony-series internal-variable model $\mathcal { H } _ { \mathrm { I V } }$ by discretizing the spectral representation of the memory kernel on a common interval of relaxation rates.

(S3) Approximate the associated stress and kinetic responses of $\mathcal { H } _ { \mathrm { I V } }$ by those of the neural operator model $\mathcal { H } _ { \mathrm { N O } }$ on compact input sets and control the induced trajectory error.

(S4) Choose $n _ { \xi }$ and the error tolerances for $\mathcal { F } _ { \mathrm { N O } }$ and $\mathcal { G } _ { \mathrm { N O } }$ to obtain (29).

(S1): Compactness and Continuity. The uniform BV and $L _ { \mathrm { p e r } } ^ { \infty }$ bounds make compact in $( L _ { \mathrm { p e r } } ^ { 2 } ) ^ { 2 }$ (Theorem B.2). Convergence of the material fields in $\dot { L } _ { \mathrm { p e r } } ^ { 2 }$ implies strong convergence of the associated unit-cell operators (Theorems B.1 and B.3). Consequently, the homogenized tensors $\mathbb { V } ^ { \mathrm { h o m } }$ and $\mathbb { E } ^ { \mathrm { h o m } }$ depend continuously on the microstructure, and the spectral measure in the memory representation is weakly continuous (Theorems B.4 and B.5).

(S2): Internal-Variable Approximation. We replace the spectral integral in Theorem 3.4 by a finite sum of exponentials, or Prony series. All relaxation rates lie in the common interval $\mathcal { T } = [ \lambda _ { \mathrm { m i n } } , \lambda _ { \mathrm { m a x } } ]$ , where $\lambda _ { \mathrm { m i n } } = \alpha _ { \mathrm { e l a s } } / \beta _ { \mathrm { v i s c } }$ and $\lambda _ { \operatorname* { m a x } } = \beta _ { \mathrm { e l a s } } / \alpha _ { \mathrm { v i s c } }$

We introduce the following weighted spectral measure and relaxation response:

$$
\mathrm { d } \mu _ { \Psi , \mathtt { E } } ( \lambda ) : = \frac { \lambda } { \beta _ { \mathrm { e l a s } } / \beta _ { \mathrm { v i s c } } + \lambda } \mathrm { d } \nu _ { \Psi , \mathtt { E } } ( \lambda ) , \qquad \mathcal { W } _ { e , t } ( \lambda ) : = \left( \frac { \beta _ { \mathrm { e l a s } } } { \beta _ { \mathrm { v i s c } } } + \lambda \right) \int _ { 0 } ^ { t } \exp ( - \lambda ( t - s ) ) \dot { \bar { e } } ( s ) \mathrm { d } s .
$$

This weighting will allow us to bound the stress error introduced by the Prony-series approximation with a constant independent of the elasticity contrast $\beta _ { \mathrm { e l a s } } / \alpha _ { \mathrm { e l a s } }$ appearing in (23) (Theorems B.6 and B.7). The memory contribution to stress can then be written as

$$
\int _ { 0 } ^ { t } \mathbb { K } ( t - s ) \dot { \overline { { e } } } ( s ) \mathrm { d } s = \int _ { \mathbb { Z } } \mathrm { d } \mu _ { \mathbb { V } , \mathbb { E } } ( \lambda ) \mathcal { W } _ { \overline { { e } } , t } ( \lambda ) .\tag{30}
$$

We interpolate $\mathcal { W } _ { \overline { { e } } , t }$ in ln λ on overlapping subintervals of bounded logarithmic width, using Chebyshev nodes on each subinterval (Figure 3 and Theorem B.8). Analyticity of the response and the bounded widths give a uniform geometric convergence rate (Theorem B.9).

The overlap permits continuous nonnegative weights summing to one, which combine the local interpolants without accumulating their errors.

Integrating the combined interpolant against $\mu _ { \mathbb { V } , \mathbb { E } }$ gives a model with memory kernel $\begin{array} { r } { \mathbb { K } ^ { ( n ) } ( t ) = \sum _ { k = 1 } ^ { n } \mathbb { K } _ { k } \exp ( - \lambda _ { k } t ) } \end{array}$ . The rates $\lambda _ { k }$ are shared by all admissible microstructures, while the coeficients $\mathbb { K } _ { k } ( \mathbb { V } , \mathbb { E } )$ depend continuously on the microstructure (Theorem B.10), as needed in (S3). Each exponential term introduces one tensorial internal variable $q _ { k } , \mathrm { g i v i n g }$ the model $\mathcal { H } _ { \mathrm { I V } }$

$$
\overline { { \sigma } } _ { \mathrm { I V } } ( t ) = \mathbb { V } ^ { \mathrm { h o m } } \dot { \overline { { e } } } ( t ) + \mathbb { E } ^ { \mathrm { h o m } } \overline { { e } } ( t ) + \sum _ { k = 1 } ^ { n } q _ { k } ( t ) ,\tag{31a}
$$

$$
\dot { q } _ { k } ( t ) = \mathbb { K } _ { k } \dot { \overline { { e } } } ( t ) - \lambda _ { k } q _ { k } ( t ) , \qquad q _ { k } ( 0 ) = 0 , \qquad k = 1 , \ldots , n .\tag{31b}
$$

![](images/3f165be787fddf37090faec01a8e2c412695455605b8cdb6f4dbc0087865b5fd.jpg)  
Figure 3: Prony-series construction by local Chebyshev interpolation (S2). The middle panel shows one component of the weighted relaxation response as λ varies, for a given strain history and time. Two nodes illustrate interpolation; shading marks the error. The resulting relaxation rates are shared across microstructures, while the tensor coeficients depend continuously on the microstructure.

Define the logarithmic relaxation span by $\textsf { R } : = \ln ( \lambda _ { \operatorname* { m a x } } / \lambda _ { \operatorname* { m i n } } )$ . For ${ \textsf { R } } > { \mathsf { 0 } } .$ , we use $J = 1 + \lceil { \mathsf { R } } \rceil$ subintervals and m nodes per subinterval, giving $n = J m$ and $n _ { \xi } = J m d _ { \nu ^ { 2 } }$ Theorem B.10 gives the uniform stress error bound

$$
\operatorname* { s u p } _ { ( \mathbb { V } , \mathbb { E } ) \in \mathcal { M } } \operatorname* { s u p } _ { \overline { { e } } \in \mathcal { E } } \left\| \overline { { \sigma } } - \overline { { \sigma } } _ { \mathrm { I V } } \right\| _ { L ^ { \infty } ( \mathcal { T } ; \mathcal { V } ^ { 2 } ) } < \kappa _ { \mathrm { I V } } 2 ^ { - ( m - 1 ) } ,
$$

where $\kappa _ { \mathrm { I V } }$ depends only on d and $\delta _ { \mathrm { s t r e s s } }$ . Increasing m therefore gives any prescribed accuracy, uniformly over admissible microstructures, strain trajectories, and time. In the next subsection, we use this estimate to bound the number of tensorial internal variables suficient for a prescribed stress accuracy. Theorem 4.6 explains how the weighted representation (30) and the local interpolation introduced in this step improve that bound.

(S3): Neural Operator Approximation. Uniform bounds on strain, strain rate, and the internal variables of $\mathcal { H } _ { \mathrm { I V } }$ let us choose a compact set of vector inputs with a margin for internal-variable errors (Theorem B.11). The continuity established in (S1)–(S2) and compactness of  permit uniform approximation of the stress and kinetic response mappings of $\mathcal { H } _ { \mathrm { I V } }$ by neural operators, with errors $\varepsilon _ { \mathcal { F } }$ and $\varepsilon _ { \mathcal { G } }$ (Theorem B.12). Choosing $\mathcal { G } _ { \mathrm { N O } }$ locally Lipschitz in its vector inputs gives a unique local solution $\xi _ { \mathrm { N O } }$

The terms $- \lambda _ { k } q _ { k }$ in (31) yield a bound, uniform in time, on the diference between the internal variables of $\mathcal { H } _ { \mathrm { N O } }$ and $\mathcal { H } _ { \mathrm { I V } }$ . For suficiently small $\varepsilon _ { \mathcal { G } }$ , the norm of this diference remains below the chosen margin, so the inputs remain in this compact set and the solution extends to all of $\tau$ . Thus Theorem B.13 gives

$$
\operatorname* { s u p } _ { ( \mathbb { V } , \mathbb { E } ) \in \mathcal { M } } \operatorname* { s u p } _ { \bar { e } \in \mathcal { E } } \left\| \overline { { \sigma } } _ { \mathrm { N O } } - \overline { { \sigma } } _ { \mathrm { I V } } \right\| _ { L ^ { \infty } ( \mathcal { T } ; \mathcal { V } ^ { 2 } ) } \leq \varepsilon _ { \mathcal { F } } + \frac { \sqrt { n } } { \lambda _ { \operatorname* { m i n } } } \varepsilon _ { \mathcal { G } } .
$$

The first term bounds the error in approximating the stress response mapping; the second accounts for the error in the internal variables.

$( S \% )$ : Combination of the Approximation Errors. Choose any $\varepsilon _ { \mathrm { I V } } \in ( 0 , \varepsilon )$ and then n so that the stress error in (S2) is at most $\varepsilon _ { \mathrm { I V } }$ . For this choice of $n ,$ choose $\varepsilon _ { \mathcal { F } }$ and $\varepsilon _ { \mathcal { G } }$ so that the error in (S3) is at most $\varepsilon - \varepsilon _ { \mathrm { { I V } } }$ and the stability condition holds. The triangle inequality gives (29). □

## 4.3. On the Suficient Dimension of the Internal Variable

The proof of Theorem 4.4 constructs the intermediate model $\mathcal { H } _ { \mathrm { I V } }$ from a Prony series, then approximates it by the neural operator model $\mathcal { H } _ { \mathrm { N O } }$ without increasing the dimension of the internal variable model implied by $\mathcal { H } _ { \mathrm { I V } }$ . Each term in the series introduces one tensorial internal variable, so bounding $n _ { \xi }$ reduces to bounding the number of terms needed for the target accuracy. The local Chebyshev interpolation estimate in (S2) provides this bound. For a prescribed stress error tolerance $\varepsilon > 0$ , the following theorem expresses it in terms of the logarithmic relaxation span R and the relative stress error $\mathsf { E } : = \varepsilon / \delta _ { \mathrm { s t r e s s } } .$

Theorem 4.5 (Suficient Internal-Variable Dimension). In the setting of Theorem $4 . 4 ,$ use the parameters R and E defined above. $I f \mathsf { R } = 0 .$ , set $N : = d \nu ^ { 2 }$ . If $\mathsf { R } > 0$ , define

$$
N : = d _ { \mathcal { V } ^ { 2 } } ( 1 + \lceil \mathsf { R } \rceil ) \left( 1 + \left\lceil \log _ { 2 } ^ { + } \left( \frac { 8 d _ { \mathcal { V } ^ { 2 } } ( 1 + \sec ( 3 / 4 ) ) } { \mathsf { E } } \right) \right\rceil \right) .
$$

Here $\log _ { 2 } ^ { + } ( x ) : = \operatorname* { m a x } \{ 0 , \log _ { 2 } x \}$ . Then there exist response mappings $( \mathcal { F } _ { \mathrm { N O } } , \mathcal { G } _ { \mathrm { N O } } ) \in \mathfrak { F } ( N ) \times$ ${ \mathfrak { G } } ( N )$ attaining (29). Thus N is a suficient internal-variable dimension.

Proof. If $\textsf { R } = \textsf { 0 }$ , take $n = 1 ;$ the approximation by $\mathcal { H } _ { \mathrm { I V } }$ is exact. Otherwise, choose $m = 1 + \lceil \log _ { 2 } ^ { + } ( \kappa _ { \mathrm { I V } } / \varepsilon ) \rceil$ and $n = ( 1 + \lceil { \mathsf { R } } \rceil ) m$ , with $\kappa _ { \mathrm { I V } }$ as in (62). The strict bound in (62) ensures that the uniform error in the stress predicted by $\mathcal { H } _ { \mathrm { I V } }$ is less than ε. In either case, choose $\varepsilon _ { \mathrm { I V } }$ strictly between this error and ε. The tolerances in (69) then bound the remaining stress error by $\varepsilon - \varepsilon _ { \mathrm { { I V } } }$ through Theorem B.13. The triangle inequality yields (29) with $n d _ { \mathscr { V } ^ { 2 } } = N$ □

<table><tr><td rowspan=1 colspan=1>Name</td><td rowspan=1 colspan=1>Definition</td><td rowspan=1 colspan=1>Interpretation</td></tr><tr><td rowspan=1 colspan=1>LogarithmicRelaxation Span</td><td rowspan=1 colspan=1> $\mathsf { R } : = \ln \left( \frac { \beta _ { \mathrm { v i s c } } \beta _ { \mathrm { e l a s } } } { \alpha _ { \mathrm { v i s c } } \alpha _ { \mathrm { e l a s } } } \right)$ </td><td rowspan=1 colspan=1>Logarithm of the ratio of the largestto the smallest relaxation rate</td></tr><tr><td rowspan=1 colspan=1>Viscosity Contrast</td><td rowspan=1 colspan=1> $\mathsf { C } _ { \mathrm { v i s c } } : = \frac { \beta _ { \mathrm { v i s c } } } { \alpha _ { \mathrm { v i s c } } }$ </td><td rowspan=1 colspan=1>Ratio between the maximum and minimumviscosity</td></tr><tr><td rowspan=1 colspan=1>Elasticity Contrast</td><td rowspan=1 colspan=1> $\mathsf { C } _ { \mathrm { e l a s } } : = \frac { \beta _ { \mathrm { e l a s } } } { \alpha _ { \mathrm { e l a s } } }$ </td><td rowspan=1 colspan=1>Ratio between the maximum and minimumelasticity</td></tr><tr><td rowspan=1 colspan=1>Relative Stress Error</td><td rowspan=1 colspan=1> $\mathsf { E } : = \frac { \varepsilon } { \delta _ { \mathrm { s t r e s s } } }$ </td><td rowspan=1 colspan=1>Ratio between the stress error toleranceand the prescribed stress scale</td></tr></table>

Table 2: Non-dimensional groups for the suficient internal-variable dimension.

Define the viscosity and elasticity contrasts by $\mathsf { C } _ { \mathrm { v i s c } } : = \beta _ { \mathrm { v i s c } } / \alpha _ { \mathrm { v i s c } }$ and ${ \sf C } _ { \mathrm { e l a s } } : = \beta _ { \mathrm { e l a s } } / \alpha _ { \mathrm { e l a s } } .$ respectively. The non-dimensional parameters are summarized in Table 2. The bound in Theorem 4.5 has the form

$$
N = \mathcal { O } _ { d } \left( ( 1 + \mathsf { R } ) \left( 1 + \ln ^ { + } ( 1 / \mathsf { E } ) \right) \right) , \qquad \mathsf { R } = \ln ( \mathsf { C } _ { \mathrm { v i s c } } \mathsf { C } _ { \mathrm { e l a s } } ) ,
$$

where ln $^ { + } x = \operatorname* { m a x } \{ 0 , \ln x \}$ and the implicit constant depends only on d. In particular,

$$
N = \mathcal { O } \big ( \mathrm { l n } ( 1 / \mathsf { E } ) \big ) , \mathsf { E } \to 0 , \mathsf { C } _ { \mathrm { v i s c } } , \mathsf { C } _ { \mathrm { e l a s } } \mathrm { ~ h e l d ~ c o n s t a n t } ,
$$

$$
N = \mathcal { O } \big ( \ln \mathsf { C } _ { \mathrm { v i s c } } \big ) , \quad \mathsf { C } _ { \mathrm { v i s c } } \to \infty , \quad \mathsf { E } , \mathsf { C } _ { \mathrm { e l a s } } \mathrm { ~ h e l d ~ c o n s t a n t } ,\tag{32}
$$

$$
N = \mathcal { O } \big ( \ln \mathsf { C } _ { \mathrm { e l a s } } \big ) , \quad \mathsf { C } _ { \mathrm { e l a s } } \to \infty , \quad \mathsf { E } , \mathsf { C } _ { \mathrm { v i s c } } \mathrm { ~ h e l d ~ c o n s t a n t } .
$$

Remark 4.6 (Choices in the Prony-Series Construction). Simpler Prony-series constructions also establish Theorem 4.4, but reweighting and local interpolation improve the dependence of the suficient internal-variable dimension on the material contrasts.

Without reweighting, interpolation with m Chebyshev nodes per subinterval and the totalvariation bound for ν in (23) give a relative stress error bound of order $\mathsf C _ { \mathrm { e l a s } } 2 ^ { - ( m - 1 ) }$ . At fixed relative accuracy E and viscosity contrast $\complement _ { \mathrm { v i s c } } ,$ choosing m from this estimate gives a suficient internal-variable dimension of order $( \ln { \mathsf { C } _ { \mathrm { e l a s } } } ) ^ { 2 }$

For the weighted representation, consider a single interpolant with n Chebyshev nodes on $[ \ln \lambda _ { \mathrm { m i n } } , \ln \lambda _ { \mathrm { m a x } } ]$ . The Chebyshev interpolation estimate for analytic functions [56, Theorem 8.2] yields a relative stress error bound of order $\mathsf { R } ( 1 + 1 / \mathsf { R } ) ^ { - ( n - 1 ) } \ f o r \ \mathsf { R } \ge 1$ . At fixed E, this estimate gives a suficient internal-variable dimension of order R ln R.

## 4.4. Continuity Constraints on Adaptive Relaxation Rates

Relaxation rates adapted to each microstructure’s spectral measure may yield Pronyseries models with fewer internal variables. Such a construction must preserve continuous dependence on $( \mathbb { V } , \mathbb { E } ) \in \mathcal { M }$ to support the neural-operator approximation step in Theorem 4.4. Our construction uses the same relaxation rates throughout and lets only the Prony-series coeficients $\{ { \mathbb K } _ { i } \} _ { i = 1 } ^ { n }$ vary. The following proposition shows that adapting the interpolation interval to the spectral edge can fail to preserve continuity with respect to the microstructure. We emphasize that this does not exclude placing the adaptive relaxation rate continuously based on alternative spectral information.

Proposition 4.7 (Spectral Edge Discontinuity). There is a choice of the constants defining the admissible set in (26) for which the following holds. For each $( \mathbb { V } , \mathbb { E } ) \in \mathcal { M }$ , let the largest value of the spectrum of $\mathcal { R } _ { \mathbb { V } , \mathbb { E } }$ be $\Lambda _ { + } ( \mathbb { V } , \mathbb { E } )$ . Then the map $( \mathbb { V } , \mathbb { E } ) \mapsto \Lambda _ { + } ( \mathbb { V } , \mathbb { E } )$ is not continuous on .

Proof. Choose $\alpha _ { \mathrm { v i s c } } = \beta _ { \mathrm { v i s c } } = \alpha _ { \mathrm { e l a s } } = 1 , \beta _ { \mathrm { e l a s } } = 2$ , and $\gamma _ { \mathrm { v i s c } } , \gamma _ { \mathrm { e l a s } }$ large enough to contain the sequence below. Set $B _ { m } \subset \mathbb { T } ^ { d }$ to be the strip $\{ 0 < y _ { 1 } < 1 / m \}$ and let $\mathbf { 1 } _ { B _ { m } }$ denote the indicator function on $B _ { m }$ . Set $\mathbb { V } _ { m } \equiv \mathbb { E } \equiv \mathbb { V } \equiv \mathbb { I }$ and $\mathbb { E } _ { m } = ( 1 + \mathbf { 1 } _ { B _ { m } } ) \mathbb { I }$ , where $\mathbb { I } \in \mathcal { V } ^ { 4 }$ is the identity operator on $\mathcal { V } ^ { 2 }$ . Then $( \mathbb { V } _ { m } , \mathbb { E } _ { m } ) \in \mathcal { M }$ and $( \mathbb { V } _ { m } , \mathbb { E } _ { m } ) \  \ ( \mathbb { V } , \mathbb { E } )$ in $L _ { \mathrm { p e r } } ^ { 1 }$ and $L _ { \mathrm { p e r } } ^ { 2 }$ . We have $\mathcal { R } _ { \mathbb { V } , \mathbb { E } } = \mathrm { I d } _ { \mathcal { X } } , \Lambda _ { + } ( \mathbb { V } , \mathbb { E } ) = 1$ , where $\operatorname { I d } _ { X }$ is the identity on $\mathcal { X }$ . Furthermore, $\mathcal { R } _ { \mathbb { V } _ { m } , \mathbb { E } _ { m } } \eta = \eta + \mathcal { P } _ { \mathbb { I } } ( \mathbf { 1 } _ { B _ { m } } \eta )$ for all $\eta \in { \mathcal { X } }$ . Choose normalized strain correctors $\eta _ { m } \in \mathcal { X }$ supported in $B _ { m }$ . For example, take a nonconstant smooth function $\psi _ { m }$ supported in $B _ { m }$ and a nonzero $y _ { * } ~ \in ~ \mathbb { R } ^ { d }$ ; after subtracting the mean of $\psi _ { m } y .$ and normalizing its symmetric gradient in $L _ { \mathrm { p e r } } ^ { 2 } ,$ , one obtains such an $\eta _ { m }$ . Then $\mathbf { 1 } _ { B _ { m } } \eta _ { m } = \eta _ { m }$ and $\mathcal { P } _ { \mathbb { I } } \eta _ { m } = \eta _ { m } .$ so $\mathcal { R } _ { \mathbb { V } _ { m } , \mathbb { E } _ { m } } \eta _ { m } = 2 \eta _ { m }$ . Thus $\Lambda _ { + } ( \mathbb { V } _ { m } , \mathbb { E } _ { m } ) \geq 2$ . Conversely, since $1 \leq 1 + \mathbf { 1 } _ { B _ { m } } \leq 2$ , we have $( \mathcal { R } _ { \mathbb { V } _ { m } , \mathbb { E } _ { m } } \eta , \eta ) _ { L _ { \mathrm { p e r } } ^ { 2 } } \leq 2 \| \eta \| _ { L _ { \mathrm { p e } } ^ { 2 } } ^ { 2 }$ for all $\eta \in \mathcal { X }$ . Therefore, $\Lambda _ { + } ( \mathbb { V } _ { m } , \mathbb { E } _ { m } ) \leq 2$ , and $\Lambda _ { + } ( \mathbb { V } _ { m } , \mathbb { E } _ { m } ) = 2$ per for all $m .$ , whereas $\Lambda _ { + } ( \mathbb { V } , \dot { \mathbb { E } } ) = 1$ □

## 5. Numerical Examples

This section presents two sets of numerical experiments that complement the theory. In Section 5.1, we study Kelvin–Voigt viscoelasticity and examine whether the numerical results corroborate our theory. In Section 5.2, we turn to crystal plasticity and show that the neural operator model can be applied in a nonlinear inelastic setting beyond the scope of the theory. For consistency, all numerical datasets comprise trajectories restricted to the time interval [0, 1].

## 5.1. Viscoelasticity

We consider learning 2D plane-stress Kelvin–Voigt viscoelasticity. This subsection first describes the dataset, model architecture, and training objective in Section 5.1.1. In Section 5.1.2, we link the numerical results to our theory: the stress prediction error decreases as $n _ { \xi }$ increases, increases with material contrast, and also depends on the number of training samples. Section 5.1.3 shows that the internal variables of the trained neural operator model admit a tensorial interpretation after a linear change of coordinates, but the fitted dynamics remain coupled and are not necessarily dissipative. Finally, Section 5.1.4 evaluates models trained on Voronoi microstructures on a separate particulate dataset and shows that they retain predictability under this input distributional shift.

## 5.1.1. Setup

The viscous and elastic tensors are parameterized by Young’s modulus E and Poisson ratio ν, i.e., $\mathbb { V } ( y ) = \mathbb { A } ( E _ { \mathbb { V } } ( y ) , \nu _ { \mathbb { V } } ( y ) )$ and $\mathbb { E } ( y ) = \mathbb { A } ( E _ { \mathbb { E } } ( y ) , \nu _ { \mathbb { E } } ( y ) )$ , where $\mathtt { A } ( E , \nu ) e = E e / ( 1 +$ $\nu ) + ( E \nu \mathrm { t r } ( e ) / ( 1 - \nu ^ { 2 } ) ) I$ for $e \in \mathcal { V } ^ { 2 }$ , where I is the second-order identity tensor. Thus, the microstructural input consists of the four scalar fields $M = \left( E _ { \mathbb { V } } , \nu _ { \mathbb { V } } , E _ { \mathbb { E } } , \nu _ { \mathbb { E } } \right)$ , so that $d _ { M } = 4$ The baseline microstructures are periodic Voronoi tessellations with randomly sampled 5–15 seeds, whose regions carry independently sampled modulus and Poisson-ratio values. Details of the Voronoi dataset are given in Appendices C.4 and C.5. For each microstructure, we sample a macroscopic strain trajectory and compute the corresponding averaged stress from numerical solutions of the unit-cell problem. We provide details on strain-trajectory sampling and the numerical solver in Appendix C.2 and Appendix C.1, respectively. In Figure 4, we visualize samples from the Voronoi dataset.

![](images/5a7b49b793a615f48d37e95c85de7c405e4e9b5084ecfdf00e604509e17b40c7.jpg)  
Figure 4: Representative samples from the viscoelasticity dataset. For each sample, the left panels show the elastic and viscous microstructural fields, with the modulus values plotted on a logarithmic scale. The right panels show the prescribed macroscopic strain trajectory, the resulting homogenized stress components, and the stress predicted by the memoryless model $( \mathbb { K } = 0 )$

Using a Fourier neural mapping architecture for both the stress and kinetic response mappings in (11), we solve the internal-variable dynamics with the forward Euler method. Let $\overline { { \sigma } } _ { \mathrm { N O } } ^ { \left( j \right) }$ denote the predicted stress trajectory for the jth training sample. We train the model by minimizing the loss function given by

$$
\mathrm { L o s s } = \frac { 1 } { n _ { \mathrm { t r a i n } } } \sum _ { j = 1 } ^ { n _ { \mathrm { t r a i n } } } \left( \mathrm { e r r } _ { \mathrm { r e l } } \big ( \overline { { \sigma } } _ { \mathrm { N O } } ^ { ( j ) } , \overline { { \sigma } } ^ { ( j ) } \big ) ^ { 2 } + \alpha \left\| \mathcal { G } _ { \mathrm { N O } } \big ( 0 , 0 ; M ^ { ( j ) } \big ) \right\| ^ { 2 } \right) ,\tag{33}
$$

where the relative stress prediction error is defined by

$$
\mathrm { e r r } _ { \mathrm { r e l } } ( \overline { \sigma } _ { \mathrm { N O } } , \overline { \sigma } ) = \left( \frac { \int _ { 0 } ^ { T } \| \overline { \sigma } _ { \mathrm { N O } } ( t ) - \overline { \sigma } ( t ) \| _ { F } ^ { 2 } \mathrm { d } t } { \int _ { 0 } ^ { T } \| \overline { \sigma } ( t ) \| _ { F } ^ { 2 } \mathrm { d } t } \right) ^ { 1 / 2 } .\tag{34}
$$

Here $T = 1$ . The penalty term enforces that the kinetic response vanishes at zero strain rate and zero internal state; see [23, Appendix E] for a discussion of this penalty term. We tune the weighting constant α along with the other architectural hyperparameters. Details of the architecture and training are given in Appendix C.3.

## 5.1.2. Internal-Variable Complexity and Sample Complexity

We examine how test error depends on internal-variable dimension, material contrast, and the number of training samples. The first two quantities are related to the internalvariable complexity estimate in Theorem 4.5. The sample-complexity study is empirical, since the approximation theorem does not address finite-sample efects. Our theory relies on an existence result for the internal-variable neural-operator model, while the experiments use a prescribed architecture, finite training samples, and non-convex stochastic optimization. The empirical rates reported below therefore measure the trained neural operator model’s behavior under these additional constraints.

The left panel of Figure 5 shows relative test error versus internal-variable dimension for a model trained on 4096 samples from the baseline dataset, where the elastic and viscous moduli in each Voronoi region are sampled from [0.5, 2.0] and [0.02, 0.2], respectively. The memoryless model with $\mathbb { K } \equiv 0$ has about 25% relative error, showing that memory efects should be accounted for in this experimental setup. For the trained neural operator models, the mean error decreases from 5.6% at $n _ { \xi } ~ = ~ 1$ to 2.9% at $n _ { \xi } ~ = ~ 2 5$ , with an empirical algebraic decay exponent of 0.21. We show a median-error prediction example for the baseline Voronoi dataset at $n _ { \xi } = 2 5$ in Figure C.1.

We next test the efect of material contrast at internal-variable dimension $n _ { \xi } ~ = ~ 1 5$ Starting from the baseline dataset, we generate new datasets by varying one elasticity or viscosity contrast at a time while keeping the other unchanged, and train a neural operator model using 4096 samples from each dataset. The corresponding elastic and viscous modulus ranges are listed in Appendix C.5. The middle panel of Figure 5 shows that the mean test error increases monotonically in both sweeps, with an empirical algebraic exponent of about 0.4.

The number of training samples required to achieve a given prediction accuracy is also practically important, since each sample requires solving a unit-cell problem. We additionally report a sample-complexity study at $n _ { \xi } = 1 5$ , varying the training set size from 128 to 4096 while keeping the architecture and training procedure unchanged. As shown in the right panel of Figure 5, the mean relative stress error decreases from 19.7% at 128 samples to 3.2% at 4096 samples, with an algebraic decay exponent of 0.8 over the range 1024–4096 samples. Over this range, the decay with training samples is steeper than the dependence on the internal-variable dimension in the left panel of Figure 5, indicating that finite-sample efects remain dominant in the studied regime.

![](images/0b45002ac8f043e985f59340e0af6ba6e08392849ea62b7b01aedb484d6fc101.jpg)  
Figure 5: Internal-variable and sample-complexity trends for learning viscoelasticity on the Voronoi dataset. $( L e f t )$ Relative stress error versus internal-variable dimension $n _ { \xi }$ for testing samples in the baseline dataset. $( M i d d l e )$ Relative stress error versus material contrast at $n _ { \xi } = 1 5$ , varying one material contrast at a time from the baseline. $( R i g h t )$ Relative stress error versus the number of training samples for $n _ { \xi } = 1 5$ . Markers show the mean; shaded bands show one standard deviation.

## 5.1.3. Connection with a Prony-series Internal-Variable Model

The approximation theory constructs the neural operator model in (10) through an intermediate Prony-series model of the form (31). We examine whether the trained neural operator model exhibits this structure. For the trained model with $n _ { \xi } ~ = ~ 1 5$ , we fit an afine approximation to its stress and kinetic responses $\left( \mathcal { F } _ { \mathrm { N O } } , \mathcal { G } _ { \mathrm { N O } } \right)$ as functions of strain, strain rate, and internal-variable inputs. We perform the fit across 64 microstructures, with 1024 samples of macroscopic strain trajectories per microstructure, and achieve an average coeficient of determination $R ^ { 2 }$ of 99%. In two dimensions, a symmetric tensor has $d _ { \mathcal { V } ^ { 2 } } = 3$ independent components. An invertible linear change of coordinates therefore expresses the 15 scalar internal variables as $n _ { \xi } / d _ { \nu ^ { 2 } } = 5$ tensorial variables. The fitted linear part takes the form

$$
\overline { { \sigma } } _ { \mathrm { N O } } ( t ) = \mathbb { V } _ { \mathrm { N O } } ^ { \mathrm { h o m } } \dot { \overline { { e } } } ( t ) + \mathbb { E } _ { \mathrm { N O } } ^ { \mathrm { h o m } } \overline { { e } } ( t ) + \sum _ { i = 1 } ^ { 5 } q _ { \mathrm { N O } } ^ { ( i ) } ( t ) ,\tag{35a}
$$

$$
\dot { q } _ { \mathrm { N O } } ^ { ( i ) } ( t ) = \mathbb { K } _ { \mathrm { N O } } ^ { ( i ) } \dot { \overline { { e } } } ( t ) + \sum _ { j = 1 } ^ { 5 } \mathbb { L } _ { \mathrm { N O } } ^ { ( i j ) } q _ { \mathrm { N O } } ^ { ( j ) } ( t ) , \quad q _ { \mathrm { N O } } ^ { ( i ) } ( 0 ) = 0 ,\tag{35b}
$$

![](images/703092bbd7baf0437d5769ac7911f7bc4bacf8a2d530f1e960df665ef42eeb10.jpg)  
Figure 6: Visualizations of the eigenvalues (λ) of the fitted internal-variable block operator $\mathbb { L } _ { \mathrm { N O } }$ across microstructure samples. The operator $\mathbb { L } _ { \mathrm { N O } }$ is defined by its $( i , j )$ -block $\mathbb { L } _ { \mathrm { N O } } ^ { ( i j ) }$ as in (35). $( l e f t )$ Eigenvalue distribution across 64 microstructure samples. $( r i g h t )$ The eigenvalues obtained from 6 diferent microstructure samples are shown separately. The shaded area uses a linear scale for the real part of the eigenvalue, whereas the unshaded area uses a log scale.

where $q _ { \mathrm { N O } } ^ { ( i ) } : [ 0 , 1 ]  \mathcal { V } ^ { 2 }$ are tensorial internal variables of the trained neural operator model and V<sup>hom</sup>, E<sup>hom</sup>, NO NO $\mathbb { K } _ { \mathrm { N O } } ^ { ( i ) } , \mathbb { L } _ { \mathrm { N O } } ^ { ( i j ) } \in \mathcal { L } ( \gamma ^ { 2 } , \gamma ^ { 2 } )$ are fitted, microstructure-dependent linear maps, with $i , j \in \{ 1 , . . . , 5 \}$ . Thus, after the coordinate transformation, these internal variables admit a stress-like tensorial interpretation. Their fitted dynamics, however, remain coupled and need not be dissipative, unlike the decoupled, dissipative dynamics in (31). In Figure 6, we visualize the eigenvalues of the block operator $\mathbb { L } _ { \mathrm { N O } }$ across the sampled microstructures and for 6 microstructures separately. The $( i , j )$ block of this operator is $\mathbb { L } _ { \mathrm { N O } } ^ { ( i j ) }$ in (35). The positive real parts show that some modes are not dissipative, while the nonzero imaginary parts indicate coupling-induced oscillations. In general, the negative real eigenvalues are separated on a logarithmic scale for each microstructure, eficiently capturing a wide range of relaxation timescales. Details of the linear model fitting are provided in Appendix C.6.

## 5.1.4. Testing on Particulate Microstructures

Finally, we test whether models trained on Voronoi tessellations transfer to a diferent class of microstructures. To this end, we generate a dataset of particulate microstructures: randomly placed 1–20 circular inclusions that are relatively stif and elastic, with a relatively soft, viscous background. We keep the loading-path sampling and the overall range of material properties similar to those of the baseline Voronoi dataset. Details of the particulate dataset are described in Appendix C.7.

In Figure 7, we plot the test error computed on the particulate dataset as a function of the internal-variable dimension. For $n _ { \xi } = 2 5$ , the mean relative stress error increases from 2.9% on the in-distribution Voronoi test samples to 5.8% on the particulate samples. Across all tested internal-variable dimensions, mean errors range from 5.8% to 8.0% on the particulate dataset, substantially lower than the memoryless model’s mean error of 16.9% $( \mathbb { K } = 0 )$ . Figure C.2 shows a median-error prediction example for the particulate dataset at $n _ { \xi } = 2 5$ . These results indicate that the trained neural operator model retains a substantial advantage over the memoryless model despite the shift in the microstructure geometry.

![](images/8d5d88a8710455e9a2e43a5838d0cb53abd493a41c3b22ab9881dd9c38685670.jpg)  
Figure 7: (left) A representative sample of the particulate microstructural property field. (right) Internalvariable complexity trend for models trained on Voronoi microstructures and evaluated on particulate mi crostructures. Markers show the mean; shaded bands show one standard deviation.

## 5.2. Crystal Plasticity

We next apply the neural operator model to homogenized crystal plasticity, a nonlinear inelastic setting. An explicit homogenized constitutive law exists for a one-dimensional, small-strain model without hardening [12, 23]; here we consider three-dimensional, finitestrain polycrystals, which require numerical homogenization. We describe the governing cell problem, dataset, and learning setup in Section 5.2.1, then examine empirical internal variable and sample complexity in Section 5.2.2.

## 5.2.1. Setup

We consider single-phase, face-centered cubic (FCC) Voronoi polycrystals on the unit cell $\mathbb { T } ^ { 3 }$ . The local response combines the multiplicative decomposition, cubic Hookean elasticity, and rate-dependent slip on the twelve FCC 111 110 systems [57, 58], implemented in DAMASK 3.0.2 [59].

For a prescribed macroscopic deformation-gradient history $\overline { { F } } ~ : ~ [ 0 , T ] ~  ~ \mathbb { R } ^ { 3 \times 3 }$ with det $\overline { { F } } > 0$ and $\overline { { \boldsymbol { F } } } ( 0 ) = \boldsymbol { I }$ , the quasistatic cell problem seeks a periodic displacement cor-

rector $\chi$ and the local plastic state satisfying

$$
F ( y , t ) = \overline { { F } } ( t ) + \nabla _ { y } \chi ( y , t ) , \qquad y \in  { \mathbb { T } } ^ { 3 } , \ t \in [ 0 , T ] ,
$$

$$
- \nabla _ { y } \cdot P ( y , t ) = 0 ,\tag{36a}
$$

$$
y \in \mathbb { T } ^ { 3 } , t \in [ 0 , T ] ,\tag{36b}
$$

$$
\int _ {  { \mathbb { T } } ^ { 3 } } \chi ( y , t ) \mathrm { d } y = 0 ,
$$

$$
t \in [ 0 , T ] .\tag{36c}
$$

Here $F$ and $P$ are the deformation gradient and the first Piola–Kirchhof stress on the unit cell, respectively. We specify the constitutive relation that maps $F$ to $P$ on the unit cell and the initially stress-free state in Appendix D.1. The averaged stress $\begin{array} { r } { \overline { { P } } ( t ) = \int _ { \mathbb { T } ^ { 3 } } P ( y , t ) } \end{array}$ dy defines the history-dependent constitutive response to be learned.

The dataset varies three microstructural features: grain geometry, crystallographic texture, and grainwise slip-law parameters. For each sample, the periodic Voronoi tessellation has $N _ { g } \in \{ 1 6 , \ldots , 3 2 \}$ grains with randomly sampled seed locations. Grain orientations are drawn from a mixture of random, single-mode, and two-mode texture families, resulting in samples with both clustered and difuse orientation distributions. The microstructural input $M : \mathbb { T } ^ { 3 }  \mathbb { R } ^ { d _ { M } }$ has $d _ { M } = 7$ channels: four orientation-quaternion components and the initial slip resistance, saturation slip resistance, and hardening scale. We randomly sample the three slip-law parameters for each grain around the copper reference values, while keeping the elastic constants and the remaining power-law parameters unchanged. Details of the microstructure sampling are given in Appendix D.2.

For each sample, we construct a loading history $\overline { F }$ by sampling the volumetric logarithmic strain, deviatoric logarithmic strain, and rotation vector, interpolating them in time, and combining the resulting stretch and rotation as described in Appendix D.3. This construction guarantees a positive determinant and produces multiaxial, nonproportional, and sign-changing loading paths. In the numerical implementation, we set $T = 1$ and solve (36) using DAMASK’s spectral solver on a $3 2 \times 3 2 \times 3 2$ voxel grid with 128 loading steps to obtain the averaged stress history ${ \overline { { P } } } .$ . Representative microstructures, loading paths, and stress responses are shown in Figure 8.

We approximate this history-dependent response with the following neural operator model, extended from [12] to include material dependence:

$$
\begin{array} { r l } & { \overline { { P } } _ { \mathrm { N O } } ( t ) = \mathcal { F } _ { \mathrm { N O } } ( \overline { { F } } ( t ) , \xi ( t ) ; M ) , } \\ & { \qquad \dot { \xi } ( t ) = \mathcal { G } _ { \mathrm { N O } } ( \overline { { F } } ( t ) , \xi ( t ) ; M ) , \quad \xi ( 0 ) = 0 . } \end{array}
$$

Here, we represent both the stress and kinetic responses $\mathcal { F } _ { \mathrm { N O } }$ and $\mathcal { G } _ { \mathrm { N O } }$ using Fourier neural mappings. We use the forward Euler method to solve the internal-variable dynamics. We train the model by minimizing a loss function similar to (33), which sums the mean relative stress-prediction error and a penalty term enforcing $\mathcal { G } _ { \mathrm { N O } } ( I , 0 ; M ) = 0$ . Details of the architecture and training are given in Appendix D.4.

## 5.2.2. Internal-Variable Complexity and Sample Complexity

The left panel of Figure 9 shows relative test error versus internal-variable dimension for a model trained on 4096 samples. For the trained neural operator models, the mean error decreases from 25% at $n _ { \xi } = 1$ $9 . 5 \%$ at $n _ { \xi } = 1 5$ . We observe a plateau in testing error after reaching $n _ { \xi } = 1 0$ . A median-error prediction example for $n _ { \xi } = 2 5$ is shown in Figure D.1.

Samples #1  
![](images/cc21c4cfa3ef4378bf332d24237a9c9c34ae50017d0e4c12d977ad92cd3c79cf.jpg)  
Figure 8: Representative samples from the crystal plasticity dataset. For each sample, the left panels show the grain geometry, the grainwise slip-law parameters, and the pole figures corresponding to the crystallographic texture. The right panels show the prescribed macroscopic displacement-gradient trajectory $( { \overline { { F } } } - I )$ and the resulting averaged first Piola–Kirchhof stress (P). Sample $\# 1$ is drawn from the randomtexture family, whereas $\# 2$ is drawn from the single-mode-texture family.

We additionally report a sample-complexity study at $n _ { \xi } = 2 5$ , varying the training set size from 512 to 8192 while keeping the architecture and training procedure unchanged. As shown in the right panel of Figure 9, the mean relative stress error decreases from 30.9% at 512 samples to 8.9% at 8192 samples. While finite-sample efects remain dominant relative to the internal variable dimension, the error decays more slowly with training samples than in viscoelasticity, indicating that homogenized crystal plasticity requires more data to learn well. This may partly reflect limitations of the data-generation strategy for learning sheardominated modes. The normal stress components have large magnitudes, and the mean relative test errors are about 6% for normal components and 30% for shear components, compared with an aggregate error of about 9%. The larger shear errors are consistent with the visualization of the median-error test sample shown in Figure D.1. Further investigation of how these modes are learned could guide the design of more informative loading paths and active learning strategies to reduce data requirements; see, e.g., [60, 61].

![](images/5a00bfa6d5c8a2ca488f0b516b2360bd27c262de4877c525a7f1b4845484181a.jpg)

![](images/2eef7832b31e88811cc76573fc78f6f6237949f6d109b1c6cbc3eabafe10fc5e.jpg)  
Figure 9: Internal-variable and sample-complexity trends for learning homogenized crystal plasticity. $( L e f t )$ Relative stress error versus internal-variable dimension $n _ { \xi } .$ $( R i g h t )$ Relative stress error versus the training sample size for $n _ { \xi } = 2 5$ . Markers show the mean; shaded bands show one standard deviation.

## 6. Conclusion

This paper studied microstructure-dependent recurrent neural operator models for learning inelastic homogenization. In these models, neural operators represent microstructuredependent stress and kinetic responses, with a finite-dimensional internal variable that encodes the loading history, giving a Markovian representation of a history-dependent constitutive law. We proved that this model can approximate the homogenized constitutive law of Kelvin–Voigt viscoelasticity and quantified the internal-variable dimension suficient to achieve a prescribed accuracy.

The analysis relied on an operator formulation of the homogenized Kelvin–Voigt law and its spectral representation of the memory kernel. Based on this, we construct a Pronyseries internal-variable model with relaxation rates shared across microstructures and continuous microstructure-dependent coeficients. Combining this model with neural-operator approximation of the stress and kinetic responses yielded uniform approximation over time, admissible microstructures, and strain trajectories. The resulting suficient internal-variable dimension grows logarithmically with the inverse relative stress error and with each material contrast, holding the remaining parameters constant.

The numerical experiments complement these conclusions. For Kelvin–Voigt viscoelasticity, the learned models substantially improve on the naive Markovian model without memory. The test error decreases as the internal-variable dimension or training dataset size increases, and increases as the material contrast grows. Our results indicate that the learned internal variable can be transformed into tensorial, stress-like variables, although their dynamics are generally coupled rather than exhibiting the decoupled structure arising in the theory. The models trained on Voronoi microstructures also retain predictive accuracy for particulate microstructures. For crystal plasticity, we show that the same datadriven modeling framework can learn nonlinear inelastic homogenization, demonstrating the framework’s practicality beyond linear viscoelasticity theory.

There are numerous directions for future study: theoretical, methodological, and applied. A potential improvement of the complexity result involves sharpening the internal-variable complexity estimates using exponential-sum approximation of completely monotone functions [62, 63, 64, 65, 66]. Related history-variable constructions [67] motivate alternative memory representations, for which continuous dependence on microstructure remains a requirement. Other directions include parameter and sample complexity analyses [68, 69] and extensions to nonlinear inelasticity [70, 71]. Finally, an important future direction is to develop methods that enforce thermodynamical consistency and well-posedness of the macroscopic balance equation. A good starting point is learning energetic formulations [72, 30, 15] with microstructure dependence.

## Acknowledgment

This work was supported by the U.S. Ofice of Naval Research through the Multidisciplinary University Research Initiative (MURI) program on Data-Driven Closure Relations under award number N00014-23-1-2654. George Stepaniants was supported by an NSF Mathematical Sciences Postdoctoral Research Fellowship (MSPRF) under award number 2402074. Andrew Stuart was also supported by the U.S. Department of Defense through a Vannevar Bush Faculty Fellowship under award number N00014-22-1-2790; this award also partially supported Lianghao Cao. The authors thank Kaushik Bhattacharya for suggestions that helped improve this paper.

OpenAI ChatGPT assisted in formulating proof strategies. OpenAI GPT-6.1-Sol suggested the bound improvement discussed in Theorem 4.6. OpenAI GPT-5.3-codex in GitHub Copilot helped improve an earlier version of the viscoelasticity data-generation and training code published in [23] and implement the crystal plasticity data-generation pipeline. We used OpenAI Codex GPT-6 Astra to formalize our theoretical results in Lean 4.35.0, and we used Lean Comparator to verify the proofs against separately specified theorem statements. OpenAI GPT-6.1 Sol assisted in producing TikZ code for Figures 1 to 3. The authors take full responsibility for the manuscript’s content.

## References

[1] A. Bensoussan, J.-L. Lions, G. Papanicolaou, Asymptotic Analysis for Periodic Structures, volume 5 of Studies in Mathematics and Its Applications, North-Holland, 1978. doi:10.1016/S0168-2024(08) 70192-9.

[2] E. Sánchez-Palencia, Non-Homogeneous Media and Vibration Theory, volume 127 of Lecture Notes in Physics, Springer, 1980. doi:10.1007/3-540-10000-8.

[3] G. Allaire, Homogenization and two-scale convergence, SIAM Journal on Mathematical Analysis 23 (1992) 1482–1518. doi:10.1137/0523084.

[4] V. V. Jikov, S. M. Kozlov, O. A. Oleinik, Homogenization of Diferential Operators and Integral Functionals, Springer, 1994. doi:10.1007/978-3-642-84659-5.

[5] G. A. Pavliotis, A. M. Stuart, Multiscale Methods: Averaging and Homogenization, volume 53 of Texts in Applied Mathematics, Springer, New York, NY, 2008. doi:10.1007/978-0-387-73829-1.

[6] F. Feyel, J.-L. Chaboche, FE<sup>2</sup> multiscale approach for modelling the elastoviscoplastic behaviour of long fibre SiC/Ti composite materials, Computer Methods in Applied Mechanics and Engineering 183 (2000) 309–330. doi:10.1016/S0045-7825(99)00224-8.

[7] C. Miehe, J. Schröder, J. Schotte, Computational homogenization analysis in finite plasticity: simulation of texture development in polycrystalline materials, Computer Methods in Applied Mechanics and Engineering 171 (1999) 387–418. doi:10.1016/S0045-7825(98)00218-7.

[8] M. G. D. Geers, V. G. Kouznetsova, W. A. M. Brekelmans, Multi-scale computational homogenization: trends and challenges, Journal of Computational and Applied Mathematics 234 (2010) 2175–2182. doi:10.1016/j.cam.2009.08.077.

[9] F. Ghavamian, A. Simone, Accelerating multiscale finite element simulations of history-dependent materials using a recurrent neural network, Computer Methods in Applied Mechanics and Engineering 357 (2019) 112594. doi:10.1016/j.cma.2019.112594.

[10] H. J. Logarzo, G. Capuano, J. J. Rimoli, Smart constitutive laws: Inelastic homogenization through machine learning, Computer Methods in Applied Mechanics and Engineering 373 (2021) 113482. doi:10. 1016/j.cma.2020.113482.

[11] B. Liu, N. Kovachki, Z. Li, K. Azizzadenesheli, A. Anandkumar, A. M. Stuart, K. Bhattacharya, A learning-based multiscale method and its application to inelastic impact problems, Journal of the Mechanics and Physics of Solids 158 (2022) 104668. doi:10.1016/j.jmps.2021.104668.

[12] B. Liu, E. Ocegueda, M. Trautner, A. M. Stuart, K. Bhattacharya, Learning macroscopic internal variables and history dependence from microscopic models, Journal of the Mechanics and Physics of Solids 178 (2023) 105329. doi:10.1016/j.jmps.2023.105329.

[13] K. Bhattacharya, B. Liu, A. M. Stuart, M. Trautner, Learning markovian homogenized models in viscoelasticity, Multiscale Modeling & Simulation 21 (2023) 641–679. doi:10.1137/22M1499200.

[14] M. Karimi, K. Bhattacharya, A learning-based multiscale model for reactive flow in porous media, Water Resources Research 60 (2024) e2023WR036303. doi:10.1029/2023WR036303.

[15] M. Raj, L. Cao, A. Stuart, K. Bhattacharya, A neural-network framework to learn history-dependent constitutive laws and identifiability of internal variables, Journal of the Mechanics and Physics of Solids 217 (2026) 106807. doi:10.1016/j.jmps.2026.106807.

[16] M. Mozafar, R. Bostanabad, W. Chen, K. Ehmann, J. Cao, M. A. Bessa, Deep learning predicts path-dependent plasticity, Proceedings of the National Academy of Sciences 116 (2019) 26414–26420. doi:10.1073/pnas.1911815116.

[17] Z. Liu, C. T. Wu, M. Koishi, A deep material network for multiscale topology learning and accelerated nonlinear modeling of heterogeneous materials, Computer Methods in Applied Mechanics and Engineering 345 (2019) 1138–1168. doi:10.1016/j.cma.2018.09.020.

[18] M. B. Gorji, M. Mozafar, J. N. Heidenreich, J. Cao, D. Mohr, On the potential of recurrent neural networks for modeling path dependent plasticity, Journal of the Mechanics and Physics of Solids 143 (2020) 103972. doi:10.1016/j.jmps.2020.103972.

[19] C. Rao, Y. Liu, Three-dimensional convolutional neural network (3D-CNN) for heterogeneous material homogenization, Computational Materials Science 184 (2020) 109850. doi:10.1016/j.commatsci. 2020.109850

[20] N. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhattacharya, A. M. Stuart, A. Anandkumar, Neural operator: Learning maps between function spaces with applications to PDEs, Journal of Machine Learning Research 24 (2023) 1–97.

[21] K. Bhattacharya, N. B. Kovachki, A. Rajan, A. M. Stuart, M. Trautner, Learning homogenization for elliptic operators, SIAM Journal on Numerical Analysis 62 (2024) 1844–1873. doi:10.1137/23M1585015.

[22] K. Bhattacharya, Multiscale modeling of materials and neural operators, MRS Bulletin 51 (2026) 913–923. doi:10.1557/s43577-026-01117-8.

[23] K. Bhattacharya, L. Cao, G. Stepaniants, A. M. Stuart, M. Trautner, Learning memory and material dependent constitutive laws, The SMAI Journal of Computational Mathematics 12 (2026) 219–267. doi:10.5802/smai-jcm.148.

[24] B. D. Coleman, M. E. Gurtin, Thermodynamics with internal state variables, The Journal of Chemical Physics 47 (1967) 597–613. doi:10.1063/1.1711937.

[25] J. Rice, Inelastic constitutive relations for solids: An internal-variable theory and its application to metal plasticity, Journal of the Mechanics and Physics of Solids 19 (1971) 433–455. doi:10.1016/ 0022-5096(71)90010-X

[26] B. Halphen, N. Q. Son, Sur les matériaux standard généralisés, Journal de Mécanique 14 (1975) 39–63.

[27] X. He, J.-S. Chen, Thermodynamically consistent machine-learned internal state variable approach for data-driven modeling of path-dependent materials, Computer Methods in Applied Mechanics and Engineering 402 (2022) 115348. doi:10.1016/j.cma.2022.115348.

[28] R. E. Jones, A. L. Frankel, K. L. Johnson, A neural ordinary diferential equation framework for modeling inelastic stress response via internal state variables, Journal of Machine Learning for Modeling and Computing 3 (2022) 1–35. doi:10.1615/JMachLearnModelComput.2022042917.

[29] M. Rosenkranz, K. A. Kalina, J. Brummund, W. Sun, M. Kästner, Viscoelasticty with physicsaugmented neural networks: model formulation and training methods without prescribed internal variables, Computational Mechanics 74 (2024) 1279–1301. doi:10.1007/s00466-024-02477-1.

[30] M. Flaschel, P. Steinmann, L. De Lorenzis, E. Kuhl, Convex neural networks learn generalized standard material models, Journal of the Mechanics and Physics of Solids 200 (2025) 106103. doi:10.1016/j. jmps.2025.106103.

[31] R. E. Jones, J. N. Fuhg, An attention-based neural ordinary diferential equation framework for modeling inelastic processes, Computer Methods in Applied Mechanics and Engineering 445 (2025) 118186. doi:10.1016/j.cma.2025.118186.

[32] W. Qiu, S. Huang, C. Reina, Bridging statistical mechanics and thermodynamics away from equilibrium: A data-driven approach for learning internal variables and their dynamics, Journal of the Mechanics and Physics of Solids 203 (2025) 106211. doi:10.1016/j.jmps.2025.106211.

[33] L. Lu, P. Jin, G. Pang, Z. Zhang, G. E. Karniadakis, Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators, Nature Machine Intelligence 3 (2021) 218–229. doi:10.1038/s42256-021-00302-5.

[34] S. Lanthaler, Z. Li, A. M. Stuart, Nonlocality and nonlinearity implies universality in operator learning, Constructive Approximation 62 (2025) 261–303. doi:10.1007/s00365-025-09718-3.

[35] B. D. Coleman, W. Noll, An approximation theorem for functionals, with applications in continuum mechanics, Archive for Rational Mechanics and Analysis 6 (1960) 355–370. doi:10.1007/BF00276168.

[36] B. D. Coleman, W. Noll, Foundations of linear viscoelasticity, Reviews of Modern Physics 33 (1961) 239–249. doi:10.1103/RevModPhys.33.239.

[37] L. Tartar, Memory efects and homogenization, Archive for Rational Mechanics and Analysis 111 (1990) 121–133. doi:10.1007/BF00375404.

[38] M. Ortiz, Linear viscoelasticity: Mechanics, analysis and approximation, Archives of Computational Methods in Engineering 33 (2026) 4133–4153. doi:10.1007/s11831-025-10426-1.

[39] G. A. Francfort, P. M. Suquet, Homogenization and mechanical dissipation in thermoviscoelasticity, Archive for Rational Mechanics and Analysis 96 (1986) 265–293.

[40] L. de Moura, S. Ullrich, The Lean 4 theorem prover and programming language, in: Automated Deduction—CADE 28, volume 12699 of Lecture Notes in Computer Science, Springer, 2021, pp. 625– 635. doi:10.1007/978-3-030-79876-5\_37.

[41] The mathlib Community, The Lean mathematical library, in: Proceedings of the 9th ACM SIGPLAN International Conference on Certified Programs and Proofs, CPP 2020, ACM, 2020, pp. 367–381. doi:10.1145/3372885.3373824.

[42] M. Baumgaertel, H. H. Winter, Determination of discrete relaxation and retardation time spectra from dynamic mechanical data, Rheologica Acta 28 (1989) 511–519. doi:10.1007/BF01332922.

[43] N. W. Tschoegl, The Phenomenological Theory of Linear Viscoelastic Behavior: An Introduction, 1

ed., Springer Berlin, Heidelberg, 1989. URL: https://doi.org/10.1007/978-3-642-73602-5. doi:10. 1007/978-3-642-73602-5.

[44] M. Baumgaertel, H. Winter, Interrelation between continuous and discrete relaxation time spectra, Journal of Non-Newtonian Fluid Mechanics 44 (1992) 15–36. doi:10.1016/0377-0257(92)80043-W.

[45] S. Park, R. Schapery, Methods of interconversion between linear viscoelastic material functions. part i—a numerical method based on prony series, International Journal of Solids and Structures 36 (1999) 1653–1675. doi:10.1016/S0020-7683(98)00055-9.

[46] G. Beylkin, L. Monzón, On approximation of functions by exponential sums, Applied and Computational Harmonic Analysis 19 (2005) 17–48. doi:10.1016/j.acha.2005.01.003.

[47] D. Z. Huang, N. H. Nelsen, M. Trautner, An operator learning perspective on parameter-to-observable maps, Foundations of Data Science 7 (2025) 163–225. doi:10.3934/fods.2024037.

[48] P. Helnwein, Some remarks on the compressed matrix representation of symmetric second-order and fourth-order tensors, Computer Methods in Applied Mechanics and Engineering 190 (2001) 2753–2770. doi:10.1016/S0045-7825(00)00263-2.

[49] P. G. Ciarlet, Mathematical Elasticity: Three-Dimensional Elasticity, Classics in Applied Mathematics, Society for Industrial and Applied Mathematics, Philadelphia, PA, 2021. doi:10.1137/1. 9781611976786.

[50] L. Ambrosio, N. Fusco, D. Pallara, Functions of Bounded Variation and Free Discontinuity Problems, Oxford University Press, 2000. doi:10.1093/oso/9780198502456.001.0001.

[51] T. Chen, H. Chen, Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems, IEEE Transactions on Neural Networks 6 (1995) 911–917. doi:10.1109/72.392253.

[52] K. Bhattacharya, B. Hosseini, N. B. Kovachki, A. M. Stuart, Model reduction and neural networks for parametric PDEs, The SMAI Journal of Computational Mathematics 7 (2021) 121–157. doi:10.5802/ smai-jcm.74.

[53] N. Kovachki, S. Lanthaler, S. Mishra, On universal approximation and error bounds for fourier neural operators, Journal of Machine Learning Research 22 (2021) 1–76. URL: http://jmlr.org/papers/ v22/21-0806.html.

[54] N. H. Nelsen, A. M. Stuart, The random feature model for input-output maps between banach spaces, SIAM Journal on Scientific Computing 43 (2021) A3212–A3243. doi:10.1137/20M133957X.

[55] D. Luo, T. O’Leary-Roseberry, P. Chen, O. Ghattas, Dimension reduction for derivative-informed operator learning: An analysis of approximation errors, Journal of Machine Learning Research 27 (2026) 1–94. URL: https://www.jmlr.org/papers/v27/25-1861.html.

[56] L. N. Trefethen, Approximation Theory and Approximation Practice, extended edition ed., Society for Industrial and Applied Mathematics, Philadelphia, PA, 2019. doi:10.1137/1.9781611975949.

[57] R. J. Asaro, Crystal plasticity, Journal of Applied Mechanics 50 (1983) 921–934. doi:10.1115/1. 3167205.

[58] F. Roters, P. Eisenlohr, L. Hantcherli, D. D. Tjahjanto, T. R. Bieler, D. Raabe, Overview of constitutive laws, kinematics, homogenization and multiscale methods in crystal plasticity finite-element modeling: Theory, experiments, applications, Acta Materialia 58 (2010) 1152–1211. doi:10.1016/j.actamat. 2009.10.058.

[59] F. Roters, M. Diehl, P. Shanthraj, P. Eisenlohr, C. Reuber, S. L. Wong, T. Maiti, A. Ebrahimi, T. Hochrainer, H.-O. Fabritius, S. Nikolov, M. Friak, N. Fujita, N. Grilli, K. G. F. Janssens, N. Jia, P. J. J. Kok, D. Ma, F. Meier, E. Werner, M. Stricker, D. Weygand, D. Raabe, DAMASK – the Düsseldorf advanced material simulation kit for modelling multi-physics crystal plasticity, damage, and thermal phenomena from the single crystal up to the component scale, Computational Materials Science 158 (2019) 420–478. doi:10.1016/j.commatsci.2018.04.030.

[60] K. Bhattacharya, L. Cao, A. Stuart, Optimal experimental design for reliable learning of historydependent constitutive laws, Computer Methods in Applied Mechanics and Engineering 457 (2026) 119022. doi:10.1016/j.cma.2026.119022.

[61] Y. Zhang, K. Bhattacharya, Iterated learning and multiscale modeling of history-dependent archi-

tectured metamaterials, Mechanics of Materials 197 (2024) 105090. doi:10.1016/j.mechmat.2024. 105090.

[62] D. W. Kammler, Chebyshev approximation of completely monotonic functions by sums of exponentials, SIAM Journal on Numerical Analysis 13 (1976) 761–774. doi:10.1137/0713062.

[63] D. Braess, W. Hackbusch, Approximation of 1/x by exponential sums in [1, ), IMA Journal of Numerical Analysis 25 (2005) 685–697. doi:10.1093/imanum/dri015.

[64] G. Beylkin, L. Monzón, Approximation by exponential sums revisited, Applied and Computational Harmonic Analysis 28 (2010) 131–149. doi:10.1016/j.acha.2009.08.011.

[65] Y. M. Koyama, Exponential sum approximations of finite completely monotonic functions, 2023. URL: https://arxiv.org/abs/2301.08931. doi:10.48550/arXiv.2301.08931. arXiv:2301.08931.

[66] D. Darrow, G. Stepaniants, A spectral theory of scalar Volterra equations, Communications of the American Mathematical Society 6 (2026) 634–725. doi:10.1090/cams/64.

[67] I. Romero, M. Ortiz, Identification of optimal history variables and corresponding hereditary laws in linear viscoelasticity, Computer Methods in Applied Mechanics and Engineering 461 (2026) 119122. doi:10.1016/j.cma.2026.119122.

[68] S. Lanthaler, A. M. Stuart, The parametric complexity of operator learning, IMA Journal of Numerical Analysis 46 (2026) 647–712. doi:10.1093/imanum/draf028.

[69] N. B. Kovachki, S. Lanthaler, H. Mhaskar, Data complexity estimates for operator learning, 2024. URL: https://arxiv.org/abs/2405.15992. doi:10.48550/arXiv.2405.15992. arXiv:2405.15992.

[70] F. Masi, I. Stefanou, Evolution TANN and the identification of internal variables and evolution equations in solid mechanics, Journal of the Mechanics and Physics of Solids 174 (2023) 105245. doi:10.1016/j.jmps.2023.105245.

[71] M. Ortiz, P. Pedregal, Optimal history encoding for elastic-plastic hereditary laws: Sharp input and constitutive approximation, 2026. URL: https://arxiv.org/abs/2607.09974. doi:10.48550/arXiv. 2607.09974. arXiv:2607.09974, preprint.

[72] S. Huang, Z. He, B. Chem, C. Reina, Variational Onsager neural networks (VONNs): A thermodynamics-based variational learning strategy for non-equilibrium PDEs, Journal of the Mechanics and Physics of Solids 163 (2022) 104856. doi:10.1016/j.jmps.2022.104856.

[73] A. Logg, K.-A. Mardal, G. Wells (Eds.), Automated Solution of Diferential Equations by the Finite Element Method, volume 84 of Lecture Notes in Computational Science and Engineering, Springer, Berlin, Heidelberg, 2012. doi:10.1007/978-3-642-23099-8.

[74] M. S. Alnæs, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg, C. Richardson, J. Ring, M. E. Rognes, G. N. Wells, The FEniCS project version 1.5, Archive of Numerical Software 3 (2015) 9–23. doi:10.11588/ans.2015.100.20553.

[75] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Köpf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, S. Chintala, PyTorch: An imperative style, high-performance deep learning library, in: Advances in Neural Information Processing Systems, volume 32, Curran Associates, Inc., 2019, pp. 8024–8035. URL: https://proceedings.neurips.cc/paper/2019/hash/ bdbca288fee7f92f2bfa9f7012727740-Abstract.html.

## Appendix A. Homogenized Tensors and Memory Kernel

This appendix proves symmetry and definiteness of the homogenized tensors and memory kernel in Appendix A.1, establishes the spectral representation of the kernel in Appendix A.2, and derives the homogenized model directly from the unit-cell problem in Appendix A.3.

## A.1. Symmetry and Definiteness of the Homogenized Tensors and Memory Kernel

We verify the symmetry and definiteness asserted in Theorem 3.1. Let $\mathbb { V } , \mathbb { E } \in L _ { \mathrm { p e r } } ^ { \infty } ( \mathbb { T } ^ { d } ; \mathcal { V } _ { + } ^ { 4 } )$ and let $\mathbb { V } ^ { \mathrm { h o m } } ( \mathbb { V } ) , \mathbb { E } ^ { \mathrm { h o m } } ( \mathbb { E } )$ , and $\mathbb { K } ( t ; \mathbb { V } , \mathbb { E } )$ be defined by (18) and (19).

Since $\mathcal { V } ^ { 2 }$ is finite-dimensional, it sufices to verify the symmetry and sign properties of the bilinear forms in (18) and (19). For $\mathbb { A } \in \{ \mathbb { V } , \mathbb { E } \}$ , let $B _ { \mathbb { A } } ( \overline { { e } } _ { 1 } , \overline { { e } } _ { 2 } ) : = ( \overline { { e } } _ { 1 } , \mathcal { P } _ { \mathbb { A } } ^ { \perp } \overline { { e } } _ { 2 } ) _ { \mathbb { A } }$ . Because $\mathcal { P } _ { \mathbb { A } } ^ { \perp }$ is the A-orthogonal projector onto the orthogonal complement of $x ,$ it is self-adjoint and idempotent with respect to $( \circ , \circ ) _ { \mathbb { A } }$ . Therefore $\mathcal { B } _ { \mathbb { A } } ( \overline { { e } } _ { 1 } , \overline { { e } } _ { 2 } ) = ( \mathcal { P } _ { \mathbb { A } } ^ { \perp } \overline { { e } } _ { 1 } , \mathcal { P } _ { \mathbb { A } } ^ { \perp } \overline { { e } } _ { 2 } ) _ { \mathbb { A } } = \mathcal { B } _ { \mathbb { A } } ( \overline { { e } } _ { 2 } , \overline { { e } } _ { 1 } )$ and $\begin{array} { r } { B _ { \mathbb { A } } ( \overline { { e } } , \overline { { e } } ) = \| \mathcal { P } _ { \mathbb { A } } ^ { \perp } \overline { { e } } \| _ { \mathbb { A } } ^ { 2 } \geq 0 } \end{array}$ . If $\begin{array} { r } { B _ { \mathbb { A } } ( \overline { { e } } , \overline { { e } } ) = 0 } \end{array}$ , then $\mathcal { P } _ { \mathbb { A } } ^ { \perp } \overline { { e } } = 0$ , so $\overline { { e } } = \mathcal { P } _ { \mathbb { A } } \overline { { e } } \in \mathcal { X }$ . Since e is constant and $\mathcal { X }$ contains no nonzero constant tensors, it follows that $\overline { { e } } = 0$ . Thus $\boldsymbol { B } _ { \mathbb { A } }$ is positive definite, so $\mathbb { V } ^ { \mathrm { h o m } } ( \mathbb { V } ) , \mathbb { E } ^ { \mathrm { h o m } } ( \mathbb { E } ) \in \mathcal { V } _ { + } ^ { 4 }$ . For the kernel, set $\mathcal { A } _ { t } : = \mathcal { R } _ { \mathbb { V } , \mathbb { E } } \exp ( - t \mathcal { R } _ { \mathbb { V } , \mathbb { E } } )$ for $t \geq 0$ . Since $\mathcal { R } _ { \mathbb { V } , \mathbb { E } }$ is positive and self-adjoint on $( \mathcal { X } , ( \circ , \circ ) _ { \mathbb { V } } )$ , the spectral theorem implies that $\mathcal { A } _ { t } ~ = ~ g _ { t } ( \mathcal { R } _ { \mathbb { V } , \mathbb { E } } )$ , with $g _ { t } ( \lambda ) : = \lambda \exp ( - t \lambda )$ , is also positive and self-adjoint. Hence $( \mathcal { A } _ { t } \mathcal { D } _ { \mathbb { V } , \mathbb { E } } \overline { { e } } _ { 1 } , \mathcal { D } _ { \mathbb { V } , \mathbb { E } } \overline { { e } } _ { 2 } ) _ { \mathbb { V } } = ( \mathcal { D } _ { \mathbb { V } , \mathbb { E } } \overline { { e } } _ { 1 } , \mathcal { A } _ { t } \mathcal { D } _ { \mathbb { V } , \mathbb { E } } \overline { { e } } _ { 2 } ) _ { \mathbb { V } }$ and $( \mathcal { A } _ { t } \mathcal { D } _ { \mathbb { V } , \mathbb { E } } \overline { { e } } , \mathcal { D } _ { \mathbb { V } , \mathbb { E } } \overline { { e } } ) _ { \mathbb { V } } \ge 0$ , so $\mathbb { K } ( t ; \mathbb { V } , \mathbb { E } ) \in \mathcal { V } ^ { 4 }$ is positive semidefinite for all $t \geq 0$

## A.2. Proof of Theorem 3.4

Write $\mathcal { R } : = \mathcal { R } _ { \mathbb { V } , \mathbb { E } } , \mathcal { D } : = \mathcal { D } _ { \mathbb { V } , \mathbb { E } } , \nu : = \nu _ { \mathbb { V } , \mathbb { E } }$ , and $\mathcal { T } : = \mathcal { T } _ { \mathbb { V } , \mathbb { E } }$ , and set $\lambda _ { \operatorname* { m i n } } : = \alpha _ { \mathbb { E } } / \beta _ { \mathbb { V } }$ and $\lambda _ { \operatorname* { m a x } } : = \beta _ { \mathbb { E } } / \alpha _ { \mathbb { V } }$ , so $\mathcal { T } = [ \lambda _ { \mathrm { m i n } } , \lambda _ { \mathrm { m a x } } ]$ . By the Riesz representation theorem, $\mathcal { R } \in \mathcal { L } ( \mathcal { X } )$ is the self-adjoint positive operator on $( \mathcal { X } , ( \circ , \circ ) _ { \mathbb { V } } )$ . The coercivity and boundedness assumptions yield, for all $\eta \in \mathcal { X } , \lambda _ { \operatorname* { m i n } } \Vert \eta \Vert _ { \mathbb { V } } ^ { 2 } \leq ( \mathcal { R } \eta , \eta ) _ { \mathbb { V } } \leq \lambda _ { \operatorname* { m a x } } \Vert \eta \Vert _ { \mathbb { V } } ^ { 2 }$ . Thus, the spectrum of is contained in $\mathcal { Z } .$ . By the spectral theorem, there exists a projection-valued measure $\Pi : = \Pi _ { \mathbb { V } , \mathbb { E } }$ supported on such that $\begin{array} { r } { \mathcal { { R } } \exp ( - t \mathcal { R } ) = \int _ { \mathcal { T } } \lambda \exp ( - \lambda t ) \mathrm { d } \Pi ( \lambda ) } \end{array}$ for all $t \geq 0$ . For each Borel set $B \subset \mathcal { Z }$ I<sub>define</sub> <sub>a</sub> <sub>bounded</sub> <sub>bilinear</sub> <sub>form</sub> <sub>on</sub> $\mathcal { V } ^ { 2 }$ by

$$
( \overline { { e } } _ { 1 } , \nu ( B ) \overline { { e } } _ { 2 } ) _ { F } : = \left( \Pi ( B ) { \mathcal { D } } \overline { { e } } _ { 1 } , \ { \mathcal { D } } \overline { { e } } _ { 2 } \right) _ { \mathbb { V } } .\tag{37}
$$

Since $\Pi ( B )$ is self-adjoint, $\nu ( B )$ is symmetric. Since $\mathcal { V } ^ { 2 }$ is finite-dimensional, (37) defines a unique tensor in $\mathcal { V } ^ { 4 }$ , and the strong countable additivity of Π implies that ν is countably additive. Moreover, for every Borel set $B \subset \tau$ and every $\overline { { e } } _ { 1 } , \overline { { e } } _ { 2 } \in \mathcal { V } ^ { 2 }$ , we have $\begin{array} { r } { | ( \overline { { e } } _ { 1 } , \nu ( B ) \overline { { e } } _ { 2 } ) _ { F } | \leq \| \Pi ( B ) { \mathcal D } \overline { { e } } _ { 1 } \| _ { \mathbb { V } } \| { \mathcal D } \overline { { e } } _ { 2 } \| _ { \mathbb { V } } \leq \| { \mathcal D } \overline { { e } } _ { 1 } \| _ { \mathbb { V } } \| { \mathcal D } \overline { { e } } _ { 2 } \| _ { \mathbb { V } } . } \end{array}$ , so ν is finite. Combining (19), the spectral representation of $\mathcal { R } \exp ( - t \mathcal { R } )$ , and (37) gives (22).

It remains to prove (23). First, for any Borel set $B \subset \mathcal { T }$ , the operator $\Pi ( B )$ is positive on $\mathcal { X }$ . Consequently, (37) gives, for any $\overline { { e } } \in \mathcal { V } ^ { 2 } , ( \overline { { e } } , \nu ( B ) \overline { { e } } ) _ { F } \geq 0$ Let $\{ \overline { { { e } } } _ { \alpha } \} _ { \alpha = 1 } ^ { d _ { \nu 2 } }$ be an orthonormal basis of $\mathcal { V } ^ { 2 }$ with respect to the Frobenius inner product. For any finite Borel partition $B _ { 1 } , \ldots , B _ { m }$ of $\mathcal { T } _ { : }$ , the inequality $\| \mathbb { A } \| _ { F } \le \operatorname { t r } ( \mathbb { A } )$ for positive semidefinite $\mathbb { A } \in \mathcal { V } ^ { 4 }$ gives $\begin{array} { r } { \sum _ { i = 1 } ^ { m } \| \nu ( B _ { i } ) \| _ { F } \leq \sum _ { i = 1 } ^ { m } \mathrm { t r } ( \nu ( B _ { i } ) ) = \mathrm { t r } ( \nu ( \underline { { \mathcal { T } } } ) ) = \sum _ { \alpha = 1 } ^ { d _ { \mathcal { V } ^ { 2 } } } \| \mathcal { D } \overline { { e } } _ { \alpha } \| _ { \mathbb { V } } ^ { 2 } } \end{array}$ , where the last identity follows from (37) and $\Pi ( \mathcal { T } ) = \operatorname { I d } _ { \mathcal { X } }$ . Also, for any $\overline { { e } } \in \mathcal { V } ^ { 2 }$ , since $\mathcal { P } _ { \mathbb { E } } \overline { { e } } \in \mathcal { X }$ and $\mathcal { P } _ { \mathbb { V } }$ is the V-orthogonal projector onto $\mathcal { X } _ { : }$

$$
\| \mathcal { D } \overline { { e } } \| _ { \mathbb { V } } ^ { 2 } = \| \mathcal { P } _ { \mathbb { V } } ( \overline { { e } } - \mathcal { P } _ { \mathbb { E } } \overline { { e } } ) \| _ { \mathbb { V } } ^ { 2 } \le \| \overline { { e } } - \mathcal { P } _ { \mathbb { E } } \overline { { e } } \| _ { \mathbb { V } } ^ { 2 } \le \frac { \beta _ { \mathbb { V } } } { \alpha _ { \mathbb { E } } } \| \overline { { e } } - \mathcal { P } _ { \mathbb { E } } \overline { { e } } \| _ { \mathbb { E } } ^ { 2 } \le \frac { \beta _ { \mathbb { V } } \beta _ { \mathbb { E } } } { \alpha _ { \mathbb { E } } } \| \overline { { e } } \| _ { F } ^ { 2 } .\tag{38}
$$

Applying this bound to each basis vector $\overline { { e } } _ { \alpha }$ and taking the supremum over all finite Borel partitions gives (23).

## A.3. Deriving the Homogenized Model from the Unit Cell Problem

We verify the unit-cell representation (7) by deriving the homogenized model in (20) directly from the unit cell problem (6), complementing the Laplace-transform proof of Theorem 3.1. Let $T > 0$ be arbitrary and finite, and write $\mathcal { R } : = \mathcal { R } _ { \mathbb { V } , \mathbb { E } }$ and $\mathcal { D } : = \mathcal { D } _ { \mathbb { V } , \mathbb { E } }$

Testing the balance law in (6) against $\phi \in \mathring { H } _ { \mathrm { p e r } } ^ { 1 } ( \mathbb { T } ^ { d } ; \mathbb { R } ^ { d } )$ and integrating by parts gives

$$
( { \dot { \overline { { e } } } } ( t ) + { \dot { \eta } } ( t ) , \psi ) _ { \mathbb { V } } + ( { \overline { { e } } } ( t ) + \eta ( t ) , \psi ) _ { \mathbb { E } } = 0 \quad { \mathrm { a . e . ~ } } t \in [ 0 , T ]\tag{39}
$$

where $\psi = \mathrm { s y m } ( \nabla _ { y } \phi ) \in \mathcal { X }$ and $\eta ( 0 ) = - \mathcal { P } _ { \mathbb { E } } \overline { { e } } ( 0 )$ . Since $\eta ( t ) , \dot { \eta } ( t ) \in \mathcal { X }$ for a.e. $t \in [ 0 , T ]$ , the definitions of $\mathcal { P } _ { \mathbb { V } } , \mathcal { P } _ { \mathbb { E } }$ , and  give $( \dot { \overline { { e } } } ( t ) , \psi ) _ { \mathbb { V } } = ( \mathcal { P } _ { \mathbb { V } } \dot { \overline { { e } } } ( t ) , \psi ) _ { \mathbb { V } } , ( \overline { { e } } ( t ) , \psi ) _ { \mathbb { E } } = ( \mathcal { R } \mathcal { P } _ { \mathbb { E } } \overline { { e } } ( t ) , \psi ) _ { \mathbb { V } }$ , and $( \eta ( t ) , \psi ) _ { \mathbb { E } } = ( \mathcal { R } \eta ( t ) , \psi ) _ { \mathbb { V } }$ for all $\psi \in \mathcal { X }$ and a.e. $t \in [ 0 , T ]$ . Hence (39) is equivalent to

$$
\begin{array} { r } { \dot { \eta } ( t ) + \mathcal { R } \eta ( t ) = - \mathcal { P } _ { \mathbb { Y } } \dot { \overline { { e } } } ( t ) - \mathcal { R } \mathcal { P } _ { \mathbb { E } } \overline { { e } } ( t ) , \quad \eta ( 0 ) = - \mathcal { P } _ { \mathbb { E } } \overline { { e } } ( 0 ) . } \end{array}\tag{40}
$$

Set $\omega ( t ) : = \eta ( t ) + \mathcal { P } _ { \mathbb { E } } \overline { { e } } ( t )$ for a.e. $t \in [ 0 , T ]$ . Then (40) and $\mathcal { D } = \mathcal { P } _ { \mathbb { V } } - \mathcal { P } _ { \mathbb { E } } \mathrm { g i v e } \dot { \omega } ( t ) + \mathcal { R } \omega ( t ) =$ $- \mathcal { D } \dot { \overline { { e } } } ( t )$ with $\omega ( 0 ) = 0$ . Integrating in time gives

$$
\omega ( t ) = - \int _ { 0 } ^ { t } \exp ( - ( t - s ) \mathcal { R } ) \mathcal { D } \dot { \overline { { e } } } ( s ) \mathrm { d } s .\tag{41}
$$

For any $\overline { { e } } _ { 1 } \in \mathcal { V } ^ { 2 }$ , the macroscopic stress satisfies $( { \overline { { e } } } _ { 1 } , { \overline { { \sigma } } } ( t ) ) _ { F } = ( { \overline { { e } } } _ { 1 } , { \dot { \overline { { e } } } } ( t ) + { \dot { \eta } } ( t ) ) _ { \mathbb { V } } + ( { \overline { { e } } } _ { 1 } , { \overline { { e } } } ( t ) +$ $\eta ( t ) ) _ { \mathbb { E } }$ for a.e. $t \in [ 0 , T ]$ . Using $\eta ( t ) = \omega ( t ) - \mathcal { P } _ { \mathbb { E } } \overline { { e } } ( t )$ gives $\overline { { e } } ( t ) + \eta ( t ) = \mathcal { P } _ { \mathbb { E } } ^ { \perp } \overline { { e } } ( t ) + \omega ( t )$ and $\dot { \bar { e } } ( t ) + \dot { \eta } ( t ) = \mathcal { P } _ { \mathbb { E } } ^ { \perp } \dot { \bar { e } } ( t ) + \dot { \omega } ( t )$ . Since $\mathcal { P } _ { \mathbb { E } } ^ { \perp } = \mathcal { P } _ { \mathbb { V } } ^ { \perp } + \mathcal { D }$ , this becomes

$$
\begin{array} { r } { ( \overline { { e } } _ { 1 } , \overline { { \sigma } } ( t ) ) _ { F } = \underbrace { ( \overline { { e } } _ { 1 } , \mathcal { P } _ { \overline { { \mathbb { V } } } } ^ { \perp \dot { \overline { { e } } } } ( t ) ) _ { \mathbb { V } } } _ { ( \overline { { e } } _ { 1 } , \mathbb { V } ^ { \mathrm { h o m } } \dot { \overline { { e } } } ( t ) ) _ { F } } + \underbrace { ( \overline { { e } } _ { 1 } , \mathcal { P } _ { \mathbb { E } } ^ { \perp } \overline { { e } } ( t ) ) _ { \mathbb { E } } } _ { ( \overline { { e } } _ { 1 } , \mathbb { E } ^ { \mathrm { h o m } } \overline { { e } } ( t ) ) _ { F } } + \underbrace { ( \overline { { e } } _ { 1 } , \mathcal { D } \dot { \overline { { e } } } ( t ) + \dot { \omega } ( t ) ) _ { \mathbb { V } } + ( \overline { { e } } _ { 1 } , \omega ( t ) ) _ { \mathbb { E } } } _ { = : R } . } \end{array}\tag{42}
$$

where the first two terms are identified by (18).

Since $\dot { \omega } ( t ) + \mathcal { R } \omega ( t ) = - \mathcal { D } \dot { \bar { e } } ( t )$ , we have $\mathcal { D } \dot { \overline { { e } } } ( t ) + \dot { \omega } ( t ) = - \mathcal { R } \omega ( t )$ The definitions of $\mathcal { P } _ { \mathbb { V } } , \mathcal { P } _ { \mathbb { E } }$ , and  imply $( \overline { { e } } _ { 1 } , \mathcal { R } \omega ( t ) ) _ { \mathbb { V } } = ( \mathcal { R } \mathcal { P } _ { \mathbb { V } } \overline { { e } } _ { 1 } , \omega ( t ) ) _ { \mathbb { V } }$ and $( \overline { { e } } _ { 1 } , \omega ( t ) ) _ { \mathbb { E } } = ( \mathcal { R } \mathcal { P } _ { \mathbb { E } } \overline { { e } } _ { 1 } , \omega ( t ) ) _ { \mathbb { V } } ,$ so $R = - ( \mathcal { R } D \bar { e } _ { 1 } , \omega ( t ) ) \mathbb { v }$ . Substituting (41) and using the self-adjointness and commutativity of  and exp( t ) on $( \mathcal { X } , ( \circ , \circ ) _ { \mathbb { V } } )$ gives

$$
R = \int _ { 0 } ^ { t } \bigl ( \mathcal { R } \exp ( - ( t - s ) \mathcal { R } ) \mathcal { D } \bar { e } _ { 1 } , \mathcal { D } \dot { \bar { e } } ( s ) \bigr ) _ { \mathbb { V } } \mathrm { d } s = \int _ { 0 } ^ { t } \bigl ( \bar { e } _ { 1 } , \mathbb { K } ( t - s ; \mathbb { V } , \mathbb { E } ) \dot { \bar { e } } ( s ) \bigr ) _ { F } \mathrm { d } s
$$

where the second equality is (19). Substituting into (42) yields (20).

## Appendix B. Supporting Results for Theorem 4.4

Throughout this appendix,  and  are the admissible sets in Theorem 4.4, and $\mathcal { T } =$ $[ 0 , \infty )$ . We first establish the compactness and continuity properties used in (S1).

Lemma B.1 (Strong Convergence of Multiplication Operators). Let $\{ \mathbb { A } _ { n } \} _ { n \in \mathbb { N } } \subset L _ { \mathrm { p e r } } ^ { \infty } ( \mathbb { T } ^ { d } ; \mathcal { V } ^ { 4 } )$ satisfy $\begin{array} { r l } { \operatorname* { s u p } _ { n \in \mathbb { N } } } & { { } \| \mathbb { A } _ { n } \| _ { L _ { \mathrm { p e r } } ^ { \infty } } \leq \kappa _ { \mathbb { A } } < \infty } \end{array}$ and $\mathbb { A } _ { n } \to \mathbb { A }$ in $L _ { \mathrm { p e r } } ^ { 1 } ( \dot { \mathbb { T } } ^ { d } ; \mathcal { V } ^ { 4 } )$ Then $\mathbb { A } \in L _ { \mathrm { p e r } } ^ { \infty }$ with $\| \mathbb { A } \| _ { L _ { \mathrm { p e r } } ^ { \infty } } \le \kappa _ { \mathbb { A } }$ and $\| ( \mathbb { A } _ { n } - \mathbb { A } ) e \| _ { L _ { \mathrm { p e r } } ^ { 2 } } \to 0$ for every $e \in L _ { \mathrm { p e r } } ^ { 2 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 2 } )$

Proof. Since $\mathbb { A } _ { n } \to \mathbb { A }$ in $L _ { \mathrm { p e r } } ^ { 1 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 4 } )$ and $\mathbb { T } ^ { d }$ has finite measure, a subsequence converges pointwise a.e. to A. Passing to the limit along this subsequence and using $\| \mathbb { A } _ { n } ( y ) \| _ { F } \le \kappa _ { \mathbb { A } }$ a.e. gives $\| \mathbb { A } ( y ) \| _ { F } \le \kappa _ { \mathbb { A } } \mathrm { a . e }$ . Hence $\mathbb { A } \in L _ { \mathrm { p e l } } ^ { \infty }$ and $\| \mathbb { A } \| _ { L _ { \mathrm { p e r } } ^ { \infty } } \le \kappa _ { \mathbb { A } }$ . Now take $e \in L _ { \mathrm { p e r } } ^ { 2 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 2 } )$ By the subsequence criterion in $L _ { \mathrm { p e r } } ^ { 2 } .$ it sufices to show that every subsequence of $\left\{ ( { \mathbb A } _ { n } - \right.$ $\mathbb { A } ) e \} _ { n \in \mathbb { N } }$ has a further subsequence converging to 0. Let $\mathbb { A } _ { n _ { j } }$ be any subsequence. Since $\mathbb { A } _ { n _ { j } } \ \to \ \mathbb { A }$ in $L _ { \mathrm { p e r } } ^ { 1 } .$ , after passing to a further subsequence we have $\mathbb { A } _ { n _ { i } } ( y )  \mathbb { A } ( y )$ for a.e. $\boldsymbol { y } \in \mathbb { T } ^ { d }$ . Therefore, $( \mathbb { A } _ { n _ { i } } ( y ) - \mathbb { A } ( y ) ) e ( y ) \to 0$ for a.e. $\boldsymbol { y } \in  { \mathbb { T } } ^ { d }$ , and $\| ( ( \mathbb { A } _ { n _ { j } } ( y ) - \mathbb { A } ( y ) ) e ( y ) ) \| _ { F } ^ { 2 } \le$ $\| \mathbb { A } _ { n _ { j } } ( y ) - \mathbb { A } ( y ) \| _ { F } ^ { 2 } \| e ( y ) \| _ { F } ^ { 2 } \leq ( 2 \kappa _ { \mathbb { A } } ) ^ { 2 } \| e ( y ) \| _ { F } ^ { 2 }$ for a.e. $\boldsymbol { y } \in \mathbb { T } ^ { d }$ . The right-hand side belongs to $\dot { L } _ { \mathrm { p e r } } ^ { 1 } (  { \mathbb { T } } ^ { d } )$ , so the dominated convergence theorem gives $\| ( \mathbb { A } _ { n _ { j } } - \mathbb { A } ) e \| _ { L _ { \mathrm { p e r } } ^ { 2 } } \to 0$ along the further subsequence. The subsequence criterion therefore implies $\| ( \mathbb { A } _ { n } - \tilde { \mathbb { A } } ) e \| _ { L _ { \mathrm { p e r } } ^ { 2 } } \to 0$ □

Lemma B.2 (Compactness of the Admissible Set in $L _ { \mathrm { p e r } } ^ { 2 } )$ . The admissible set defined in (26) is compact in $L _ { \mathrm { p e r } } ^ { 2 } ( \mathbb { T } ^ { d } ; \mathcal { V } ^ { 4 } ) \times L _ { \mathrm { p e r } } ^ { 2 } ( \mathbb { T } ^ { d } ; \mathcal { V } ^ { 4 } )$

Proof. Let $\{ ( \mathbb { V } _ { n } , \mathbb { E } _ { n } ) \} _ { n \in \mathbb { N } } \subset \mathcal { M }$ . Since $L _ { \mathrm { p e r } } ^ { 2 } \times L _ { \mathrm { p e r } } ^ { 2 }$ is a metric space, it sufices to prove sequential compactness. By (27b), $\{ \mathbb { V } _ { n } \} _ { n \in \mathbb { N } }$ and $\mathrm { \bar { \{ E } }  _ { n } \} _ { n \in \mathbb { N } }$ are bounded in BV. The compact embedding $\mathrm { B V } (  { \mathbb { T } } ^ { d } ) \hookrightarrow L ^ { 1 } (  { \mathbb { T } } ^ { d } )$ gives a subsequence (not relabeled) and tensors $\mathbb { V } , \mathbb { E } \in$ $L _ { \mathrm { p e r } } ^ { 1 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 4 } )$ such that $\mathbb { V } _ { n } \to \mathbb { V }$ in $L _ { \mathrm { p e r } } ^ { 1 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 4 } )$ and $\mathbb { E } _ { n } \to \mathbb { E }$ in $L _ { \mathrm { p e r } } ^ { 1 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 4 } )$ . Applying Theorem B.1 gives V, $\mathbb { E } \in L _ { \mathrm { p e r } } ^ { \infty } ( \mathbb { T } ^ { d } ; \mathcal { V } ^ { 4 } )$ , $\| \mathbb { V } \| _ { L _ { \mathrm { p e r } } ^ { \infty } } \leq \sqrt { d _ { \mathcal { V } ^ { 2 } } } \beta _ { \mathrm { v i s c } }$ , and $\| \mathbb { E } \| _ { L _ { \mathrm { p e r } } ^ { \infty } } \leq \sqrt { d _ { \mathcal { V } ^ { 2 } } } \beta _ { \mathrm { e l a s } } .$ Since $\| \mathbb { V } _ { n } - \mathbb { V } \| _ { L _ { \mathrm { p e r } } ^ { \infty } } \leq 2 \sqrt { d _ { \mathcal { V } ^ { 2 } } } \beta _ { \mathrm { v i s c } }$ and $\| \mathbb { E } _ { n } - \mathbb { E } \| _ { L _ { \mathrm { p e r } } ^ { \infty } } \leq 2 \sqrt { d _ { \mathcal { V } ^ { 2 } } } \beta _ { \mathrm { e l a s } }$ , Hölder’s inequality gives

$$
\begin{array} { r } { \| \mathbb { V } _ { n } - \mathbb { V } \| _ { L _ { \mathrm { p e r } } ^ { 2 } } ^ { 2 } \leq \| \mathbb { V } _ { n } - \mathbb { V } \| _ { L _ { \mathrm { p e r } } ^ { \infty } } \| \mathbb { V } _ { n } - \mathbb { V } \| _ { L _ { \mathrm { p e r } } ^ { 1 } } \leq 2 \sqrt { d _ { \mathcal { V } ^ { 2 } } } \beta _ { \mathrm { v i s c } } \| \mathbb { V } _ { n } - \mathbb { V } \| _ { L _ { \mathrm { p e r } } ^ { 1 } } \to 0 . } \end{array}
$$

Therefore, $\mathbb { V } _ { n } \to \mathbb { V }$ strongly in $L _ { \mathrm { p e r } } ^ { 2 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 4 } )$ . The same argument gives $\mathbb { E } _ { n } \to \mathbb { E }$ strongly in $L _ { \mathrm { { p e r } } } ^ { 2 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 4 } )$ . Lower semicontinuity of total variation under $L ^ { 1 }$ convergence implies $\| \mathbb { V } \| _ { \mathrm { B V } } \leq$ lim i $\begin{array} { r } { \operatorname * { n f } _ { n \to \infty } \| \mathbb { V } _ { n } \| _ { \mathrm { B V } } \leq \gamma _ { \mathrm { v i s c } } } \end{array}$ and $\| \mathbb { E } \| _ { \mathrm { B V } } \leq$ lim $\begin{array} { r } { \operatorname* { i n f } _ { n  \infty } \| \mathbb { E } _ { n } \| _ { \mathrm { B V } } \leq \gamma _ { \mathrm { e l a s } } , } \end{array}$ so the BV bounds are preserved. For the uniform ellipticity bounds, Theorem B.1 and Cauchy–Schwarz give, for every $e \in L _ { \mathrm { p e r } } ^ { 2 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 2 } )$

$$
\left| \int _ { \mathbb { T } ^ { d } } \left( e ( y ) , ( \mathbb { V } _ { n } ( y ) - \mathbb { V } ( y ) ) e ( y ) \right) _ { F } \mathrm { d } y \right| \leq \| e \| _ { L _ { \mathrm { p e r } } ^ { 2 } } \| ( \mathbb { V } _ { n } - \mathbb { V } ) e \| _ { L _ { \mathrm { p e r } } ^ { 2 } } \to 0 .
$$

Therefore, $\| e \| _ { \mathbb { V } _ { n } } \to \| e \| _ { \mathbb { V } }$ , and passing to the limit in the uniform ellipticity bounds for $\mathbb { V } _ { n }$ leads to the same bounds for V. The same argument applies to E. Therefore $( \mathbb { V } , \mathbb { E } ) \in \mathcal { M } . \quad \varTheta$

Proposition B.3 (Strong Convergence of the Unit-Cell Operators). Let $\{ ( \mathbb { V } _ { n } , \mathbb { E } _ { n } ) \} _ { n \in \mathbb { N } } \subset$ and assume that for some pair $( \mathbb { V } , \mathbb { E } ) , \mathbb { V } _ { n } \to \mathbb { V }$ and $\mathbb { E } _ { n } \to \mathbb { E }$ in $L _ { \mathrm { p e r } } ^ { 1 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 4 } )$ . We have

$$
\begin{array} { r } { \| ( \mathcal { P } _ { \mathbb { V } _ { n } } - \mathcal { P } _ { \mathbb { V } } ) e \| _ { L _ { \mathrm { p e r } } ^ { 2 } } \to 0 , \quad \| ( \mathcal { P } _ { \mathbb { E } _ { n } } - \mathcal { P } _ { \mathbb { E } } ) e \| _ { L _ { \mathrm { p e r } } ^ { 2 } } \to 0 \quad f o r \ e v e r y \ e \in L _ { \mathrm { p e r } } ^ { 2 } , } \end{array}\tag{43}
$$

$$
\| ( \mathcal { D } _ { \mathbb { V } _ { n } , \mathbb { E } _ { n } } - \mathcal { D } _ { \mathbb { V } , \mathbb { E } } ) e \| _ { L _ { \mathrm { p e r } } ^ { 2 } } \to 0 \quad { f o r \ e v e r y \ e \in L _ { \mathrm { p e r } } ^ { 2 } } ,\tag{44}
$$

$$
\begin{array} { r } { \| ( \mathcal { R } _ { \mathbb { V } _ { n } , \mathbb { E } _ { n } } - \mathcal { R } _ { \mathbb { V } , \mathbb { E } } ) \eta \| _ { L _ { \mathrm { p e r } } ^ { 2 } } \to 0 \quad { f o r \ e v e r y \ \eta \in \mathcal { X } } , } \end{array}\tag{45}
$$

Furthermore, for all $k \in \mathbb N$ , we have

$$
\| ( \mathcal { R } _ { \mathbb { V } _ { n } , \mathbb { E } _ { n } } ^ { k } - \mathcal { R } _ { \mathbb { V } , \mathbb { E } } ^ { k } ) \eta \| _ { L _ { \mathrm { p e r } } ^ { 2 } } \to 0 \quad { f o r \ e v e r y \ \eta \in \mathcal { X } } .\tag{46}
$$

Proof. By Theorem B.2, the limit pair $( \mathbb { V } , \mathbb { E } )$ belongs to $\mathcal { M }$ . Let $e \in L _ { \mathrm { p e r } } ^ { 2 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 2 } )$ and set $\eta _ { n } : = \mathcal { P } _ { \mathbb { V } _ { n } } e , \eta : = \mathcal { P } _ { \mathbb { V } } e$ . By (13), $( \eta _ { n } - e , \psi ) _ { \mathbb { V } _ { n } } = 0$ and $( \eta - e , \psi ) _ { \mathbb { V } } = 0$ for all $\psi \in \mathcal { X }$ . Choosing $\psi = \eta _ { n } - \eta$ , subtracting the two identities, and applying coercivity and Cauchy–Schwarz give

$$
\alpha _ { \mathrm { v i s c } } \Vert \eta _ { n } - \eta \Vert _ { L _ { \mathrm { p e r } } ^ { 2 } } ^ { 2 } \leq \Vert ( \mathbb { V } _ { n } - \mathbb { V } ) ( e - \eta ) \Vert _ { L _ { \mathrm { p e r } } ^ { 2 } } \Vert \eta _ { n } - \eta \Vert _ { L _ { \mathrm { p e r } } ^ { 2 } } .
$$

If $\| \eta _ { n } - \eta \| _ { L _ { \mathrm { p e r } } ^ { 2 } } > 0$ , we divide by this norm, and otherwise there is nothing to prove. Thus $\alpha _ { \mathrm { v i s c } } \| \eta _ { n } - \eta \| _ { L _ { \mathrm { p e r } } ^ { 2 } } \leq \| ( \mathbb { V } _ { n } - \mathbb { V } ) ( e - \eta ) \| _ { L _ { \mathrm { p e r } } ^ { 2 } } \to 0$ by Theorem B.1. Hence $\mathcal { P } _ { \mathbb { V } _ { n } } e \to \mathcal { P } _ { \mathbb { V } } e$ in $L _ { \mathrm { p e r } } ^ { 2 } .$ The same argument with $\mathbb { E } _ { n } \to \mathbb { E }$ gives $\mathcal { P } _ { \mathbb { E } _ { n } } e \to \mathcal { P } _ { \mathbb { E } } e$ , and therefore (43) and (44) follow.

Now set $\mathcal { R } _ { n } : = \mathcal { R } _ { \mathbb { V } _ { n } , \mathbb { E } _ { \tau } }$ and $\mathcal { R } : = \mathcal { R } _ { \mathbb { V } , \mathbb { E } }$ . For $\eta \in \mathcal { X }$ , let $\varrho _ { n } : = \mathcal { R } _ { n } \eta$ and $\varrho : = \mathcal { R } \eta$ . Using (15) for $\mathcal { R } _ { n }$ and $\mathcal { R } _ { : }$ , choosing $\psi = \varrho _ { n } - \varrho .$ , and arguing as above, we obtain

$$
\begin{array} { r } { \alpha _ { \mathrm { v i s c } } \Vert \varrho _ { n } - \varrho \Vert _ { L _ { \mathrm { p e r } } ^ { 2 } } \leq \Vert ( \mathbb { E } _ { n } - \mathbb { E } ) \eta \Vert _ { L _ { \mathrm { p e r } } ^ { 2 } } + \Vert ( \mathbb { V } _ { n } - \mathbb { V } ) \varrho \Vert _ { L _ { \mathrm { p e r } } ^ { 2 } } \to 0 , } \end{array}\tag{47}
$$

again by Theorem B.1. Thus (45) holds.

Testing the identity $( \mathcal { R } _ { n } \eta , \psi ) _ { \mathbb { V } _ { n } } ~ = ~ ( \eta , \psi ) _ { \mathbb { E } _ { n } }$ from (15) with $\psi ~ = ~ \mathcal { R } _ { n } \eta$ and using the coercivity of $\mathbb { V } _ { n }$ together with the upper bound on $\mathbb { E } _ { n }$ gives

$$
\operatorname* { s u p } _ { n \in \mathbb { N } } \| \mathcal { R } _ { n } \| _ { \mathcal { L } ( \mathcal { X } ) } \leq \beta _ { \mathrm { e l a s } } / \alpha _ { \mathrm { v i s c } } .\tag{48}
$$

The convergence of $\mathcal { R } _ { n } ^ { k }$ follows by induction on k. The case $k = 1$ is (47). If the claim holds for $k - 1$ , then $\mathcal { R } _ { n } ^ { k } \eta - \mathcal { R } ^ { k } \eta = \mathcal { R } _ { n } \big ( \mathcal { R } _ { n } ^ { k - 1 } \eta - \mathcal { R } ^ { k - 1 } \eta \big ) + ( \mathcal { R } _ { n } - \mathcal { R } ) \mathcal { R } ^ { k - 1 } \eta$ . The first term tends to 0 by the inductive hypothesis and (48), while the second tends to 0 by (47). Therefore, (46) holds. □

Corollary B.4 (Continuity of the Homogenized Tensors). Let $\{ ( \mathbb { V } _ { n } , \mathbb { E } _ { n } ) \} _ { n \in \mathbb { N } } \subset \mathcal { M }$ and assume that for some pair $( \mathbb { V } , \mathbb { E } ) , \mathbb { V } _ { n }  \mathbb { V }$ and $\mathbb { E } _ { n } \to \mathbb { E }$ in $L _ { \mathrm { p e r } } ^ { 1 } (  { \mathbb { T } } ^ { d } ;  { \mathcal { V } } ^ { 4 } )$ . Then $( \mathbb { V } , \mathbb { E } ) \in \mathcal { M }$ and we have $\mathbb { V } ^ { \mathrm { h o m } } ( \mathbb { V } _ { n } ) \to \mathbb { V } ^ { \mathrm { h o m } } ( \mathbb { V } )$ and $\mathbb { E } ^ { \mathrm { h o m } } ( \mathbb { E } _ { n } )  \mathbb { E } ^ { \mathrm { h o m } } ( \mathbb { E } )$

Proof. By Theorem B.2, the limit pair $( \mathbb { V } , \mathbb { E } )$ belongs to . Let $\overline { { e } } _ { 1 } , \overline { { e } } _ { 2 } \in \mathcal { V } ^ { 2 }$ . Using (18) and adding and subtracting $( \overline { { e } } _ { 1 } , \mathcal { P } _ { \mathbb { V } } ^ { \perp } \overline { { e } } _ { 2 } ) _ { \mathbb { V } _ { n } }$ , we obtain

$$
\begin{array} { r l r } & { } & { \displaystyle \big | \big ( \overline { { e } } _ { 1 } , \mathbb { V } ^ { \mathrm { h o m } } ( \mathbb { V } _ { n } ) \overline { { e } } _ { 2 } \big ) _ { F } - \big ( \overline { { e } } _ { 1 } , \mathbb { V } ^ { \mathrm { h o m } } ( \mathbb { V } ) \overline { { e } } _ { 2 } \big ) _ { F } \big | \leq \big | \big ( \overline { { e } } _ { 1 } , ( \mathcal { P } _ { \mathbb { V } _ { n } } ^ { \perp } - \mathcal { P } _ { \overline { { \mathbb { V } } } } ^ { \perp } ) \overline { { e } } _ { 2 } \big ) _ { \mathbb { V } _ { n } } \big | } \\ & { } & { \displaystyle + \left| \int _ { \mathbb T ^ { d } } \big ( \overline { { e } } _ { 1 } , ( \mathbb { V } _ { n } - \mathbb { V } ) [ \mathcal { P } _ { \overline { { \mathbb { V } } } } ^ { \perp } \overline { { e } } _ { 2 } ] ( y ) \big ) _ { F } { \mathrm { d } } y \right| . } \end{array}
$$

The first term is bounded by $\left\| \mathbb { V } _ { n } \right\| _ { L _ { \mathrm { p e r } } ^ { \infty } } \left\| \overline { { e } } _ { 1 } \right\| _ { F } \left\| \big ( \mathcal { P } _ { \mathbb { V } _ { n } } - \mathcal { P } _ { \mathbb { V } } \big ) \overline { { e } } _ { 2 } \right\| _ { L _ { \mathrm { p e r } } ^ { 2 } }$ and therefore converges to 0 by the uniform $L _ { \mathrm { p e r } } ^ { \infty }$ bound on $\mathbb { V } _ { n }$ and Theorem B.3. The second term is bounded by $\| ( { \mathbb Y } _ { n } -$ $\mathbb { V } ) \overline { { e } } _ { 1 } \| _ { L _ { \mathrm { p e r } } ^ { 2 } } \| \mathcal { P } _ { \mathbb { V } } ^ { \perp } \overline { { e } } _ { 2 } \| _ { L _ { \mathrm { p e } } ^ { 2 } }$ and also converges to 0 by Theorem B.1. Therefore, $( \overline { { { e } } } _ { 1 } , \mathbb { V } ^ { \mathrm { h o m } } ( \mathbb { V } _ { n } ) \overline { { { e } } } _ { 2 } ) _ { F } $ $\left( \overline { { e } } _ { 1 } , \mathbb { V } ^ { \mathrm { h o m } } ( \mathbb { V } ) \overline { { e } } _ { 2 } \right) _ { F }$ for all $\overline { { e } } _ { 1 } , \overline { { e } } _ { 2 } \in \mathcal { V } ^ { 2 }$ . Since $\mathcal { V } ^ { 2 }$ is finite-dimensional, this is equivalent to $\mathbb { V } ^ { \mathrm { h o m } } ( \mathbb { V } _ { n } ) \to \mathbb { V } ^ { \mathrm { h o m } } ( \mathbb { V } )$ in $\mathcal { V } ^ { 4 }$ . The argument for $\mathbb { E } ^ { \mathrm { h o m } }$ is identical. □

Lemma B.5 (Weak Convergence of the Spectral Measure). Suppose $\{ ( \mathbb { V } _ { n } , \mathbb { E } _ { n } ) \} _ { n \in \mathbb { N } } \subset \mathcal { M }$ satisfies $\mathbb { V } _ { n } \to \mathbb { V }$ and $\mathbb { E } _ { n } \to \mathbb { E }$ in $L _ { \mathrm { p e r } } ^ { 1 } ( \mathbb { T } ^ { d } ; \mathcal { V } ^ { 4 } )$ . Then the sequence of measures $\nu _ { \mathbb { V } _ { n } , \mathbb { E } _ { n } }$ , as defined by (37), converges weakly to $\nu _ { \mathbb { V } , \mathbb { E } }$ . That is, for any continuous function $f \in C ( \mathcal { T } )$ with $\mathcal { T } : = [ \alpha _ { \mathrm { e l a s } } / \beta _ { \mathrm { v i s c } } , \beta _ { \mathrm { e l a s } } / \alpha _ { \mathrm { v i s c } } ]$ and any $\overline { { e } } _ { 1 } , \overline { { e } } _ { 2 } \in \mathcal { V } ^ { 2 }$ 7

$$
\operatorname* { l i m } _ { n \to \infty } \int _ { \mathbb { T } } f ( \lambda ) \mathrm { d } \big ( \overline { e } _ { 1 } , \nu _ { \mathbb { V } _ { n } , \mathbb { E } _ { n } } ( \lambda ) \overline { e } _ { 2 } \big ) _ { F } = \int _ { \mathbb { T } } f ( \lambda ) \mathrm { d } \big ( \overline { e } _ { 1 } , \nu _ { \mathbb { V } , \mathbb { E } } ( \lambda ) \overline { e } _ { 2 } \big ) _ { F } .
$$

Proof. By Theorem B.2, the limit pair $( \mathbb { V } , \mathbb { E } )$ belongs to $\mathcal { M }$ . Write $\nu _ { n } : = \nu _ { \mathbb { V } _ { n } , \mathbb { E } _ { n } } , \nu : = \nu _ { \mathbb { V } , \mathbb { E } } .$ $\mathcal { R } _ { n } : = \mathcal { R } _ { \mathbb { V } _ { n } , \mathbb { E } _ { n } } , \mathcal { R } : = \mathcal { R } _ { \mathbb { V } , \mathbb { E } } , \mathcal { D } _ { n } : = \mathcal { D } _ { \mathbb { V } _ { n } , \mathbb { E } _ { n } }$ , and $\mathcal { D } : = \mathcal { D } _ { \mathbb { V } , \mathbb { E } }$ . For $g \in C ( \mathcal { T } )$ , let $g ( \mathcal { R } _ { n } ) \mathrel { \mathop : } =$ $\begin{array} { r } { \int _ { \mathbb { Z } } g ( \lambda ) \mathrm { d } \Pi _ { \mathbb { V } _ { n } , \mathbb { E } _ { n } } ( \lambda ) } \end{array}$ , and define $g ( \mathcal { R } )$ analogously. Then (37) gives $\begin{array} { r } { \int _ { \mathcal { T } } f ( \lambda ) \mathrm { d } ( \overline { { e } } _ { 1 } , \nu _ { n } \overline { { e } } _ { 2 } ) _ { F } \ = } \end{array}$ $\left( f ( \mathcal { R } _ { n } ) \mathcal { D } _ { n } \overline { { e } } _ { 1 } , \mathcal { D } _ { n } \overline { { e } } _ { 2 } \right) _ { \mathbb { V } _ { n } }$ , and the same identity holds with n replaced by the limit pair $( \mathbb { V } , \mathbb { E } )$ Therefore, it sufices to prove

$$
\Big ( f ( \mathcal { R } _ { n } ) \mathcal { D } _ { n } \overline { { e } } _ { 1 } , \mathcal { D } _ { n } \overline { { e } } _ { 2 } \Big ) _ { \mathbb { V } _ { n } } \to \Big ( f ( \mathcal { R } ) \mathcal { D } \overline { { e } } _ { 1 } , \mathcal { D } \overline { { e } } _ { 2 } \Big ) _ { \mathbb { V } } .\tag{49}
$$

For any ${ \varepsilon _ { p } } > 0$ , the Weierstrass approximation theorem gives a polynomial p such that

$$
\operatorname* { s u p } _ { \lambda \in \mathcal { T } } | f ( \lambda ) - p ( \lambda ) | \leq \varepsilon _ { p } .
$$

Adding and subtracting the polynomial terms gives

$$
\begin{array} { r l } & { \left| \left( f ( \mathcal { R } _ { n } ) \mathcal { D } _ { n } \overline { { e } } _ { 1 } , \mathcal { D } _ { n } \overline { { e } } _ { 2 } \right) _ { \mathbb { V } _ { n } } - \left( f ( \mathcal { R } ) \mathcal { D } \overline { { e } } _ { 1 } , \mathcal { D } \overline { { e } } _ { 2 } \right) _ { \mathbb { V } } \right| } \\ & { \leq \left| \left( ( f - p ) ( \mathcal { R } _ { n } ) \mathcal { D } _ { n } \overline { { e } } _ { 1 } , \mathcal { D } _ { n } \overline { { e } } _ { 2 } \right) _ { \mathbb { V } _ { n } } \right| + \left| \left( p ( \mathcal { R } _ { n } ) \mathcal { D } _ { n } \overline { { e } } _ { 1 } , \mathcal { D } _ { n } \overline { { e } } _ { 2 } \right) _ { \mathbb { V } _ { n } } - \left( p ( \mathcal { R } ) \mathcal { D } \overline { { e } } _ { 1 } , \mathcal { D } \overline { { e } } _ { 2 } \right) _ { \mathbb { V } } \right| } \\ & { \qquad + \left| \left( ( p - f ) ( \mathcal { R } ) \mathcal { D } \overline { { e } } _ { 1 } , \mathcal { D } \overline { { e } } _ { 2 } \right) _ { \mathbb { V } } \right| . } \end{array}\tag{50}
$$

For any $\eta \in \mathcal { X }$

$$
\Vert ( f - p ) ( \mathcal { R } _ { n } ) \eta \Vert _ { \mathbb { V } _ { n } } ^ { 2 } = \int _ { \mathbb { Z } } \vert f ( \lambda ) - p ( \lambda ) \vert ^ { 2 } \mathrm { d } ( \Pi _ { \mathbb { V } _ { n } , \mathbb { E } _ { n } } ( \lambda ) \eta , \eta ) _ { \mathbb { V } _ { n } } \le \varepsilon _ { p } ^ { 2 } \Vert \eta \Vert _ { \mathbb { V } _ { n } } ^ { 2 } ,
$$

and the same estimate holds with $\left( \mathcal { R } _ { n } , \mathbb { V } _ { n } , \Pi _ { \mathbb { V } _ { n } , \mathbb { E } _ { n } } \right)$ replaced by $\left( \mathcal { R } , \mathbb { V } , \Pi _ { \mathbb { V } , \mathbb { E } } \right)$ . Cauchy– Schwarz and (38) therefore show that the first and third terms in (50) are each bounded by $\varepsilon _ { p } \beta _ { \mathrm { v i s c } } \beta _ { \mathrm { e l a s } } \| \overline { { e } } _ { 1 } \| _ { F } \| \overline { { e } } _ { 2 } \| _ { F } / \alpha _ { \mathrm { e l a s } }$

For the polynomial term, set $\phi _ { n } : = p ( \mathcal { R } _ { n } ) \mathcal { D } _ { n } \overline { { e } } _ { 1 } , \phi : = p ( \mathcal { R } ) \mathcal { D } \overline { { e } } _ { 1 } , \psi _ { n } : = \mathcal { D } _ { n } \overline { { e } } _ { 2 }$ , and $\psi : = \mathcal { D } \overline { { e } } _ { 2 }$ Then the middle term in (50) is $| ( \phi _ { n } , \psi _ { n } ) _ { \mathbb { V } _ { n } } - ( \phi , \psi ) _ { \mathbb { V } } | . \mathrm { ~ B y ~ } ( 4 4 ) , \psi _ { n } \to \psi$ strongly in $L _ { \mathrm { p e r } } ^ { 2 }$ . If $\begin{array} { r } { p ( \lambda ) = \sum _ { k = 0 } ^ { K } c _ { k } \lambda ^ { k } } \end{array}$ , then $\begin{array} { r } { \phi _ { n } - \phi = \sum _ { k = 0 } ^ { K } c _ { k } \left( \mathcal { R } _ { n } ^ { k } \mathcal { D } _ { n } \overline { { e } } _ { 1 } - \mathcal { R } ^ { k } \mathcal { D } \overline { { e } } _ { 1 } \right) } \end{array}$ . For $k = 0 , \ldots , K$ 2

$$
\mathcal { R } _ { n } ^ { k } \mathcal { D } _ { n } \overline { { e } } _ { 1 } - \mathcal { R } ^ { k } \mathcal { D } \overline { { e } } _ { 1 } = \mathcal { R } _ { n } ^ { k } ( \mathcal { D } _ { n } - \mathcal { D } ) \overline { { e } } _ { 1 } + ( \mathcal { R } _ { n } ^ { k } - \mathcal { R } ^ { k } ) \mathcal { D } \overline { { e } } _ { 1 } .
$$

The first term tends to 0 because (44) gives $\mathcal { D } _ { n } \overline { { e } } _ { 1 } \  \ \mathcal { D } \overline { { e } } _ { 1 }$ in $L _ { \mathrm { p e r } } ^ { 2 }$ and (48) implies that $\{ \mathcal { R } _ { n } ^ { k } \} _ { n }$ is uniformly bounded on $\mathcal { X }$ for fixed k. The second term tends to 0 by (46) with $\eta = \mathcal { D } \overline { { e } } _ { 1 } \in \mathcal { X }$ . Hence $\phi _ { n } \to \phi$ strongly in $L _ { \mathrm { p e r } } ^ { 2 }$ . We now write

$$
\begin{array} { r l r } {  { ( \phi _ { n } , \psi _ { n } ) _ { \mathbb { V } _ { n } } - ( \phi , \psi ) _ { \mathbb { V } } = \big ( ( \phi _ { n } - \phi ) , \psi _ { n } \big ) _ { \mathbb { V } _ { n } } + \int _ { \mathbb { T } ^ { d } } \phi ( y ) : \big ( \mathbb { V } _ { n } ( y ) - \mathbb { V } ( y ) \big ) : \psi _ { n } ( y ) \mathrm { d } y } } \\ & { } & { \quad + \big ( \phi , ( \psi _ { n } - \psi ) \big ) _ { \mathbb { V } } . } \end{array}
$$

The first and third terms tend to 0 by Cauchy–Schwarz, the uniform $L _ { \mathrm { p e r } } ^ { \infty }$ bounds on $\mathbb { V } _ { n }$ and V, and the strong $L _ { \mathrm { p e r } } ^ { 2 }$ convergence of $\phi _ { n }$ and $\psi _ { n }$ . The integral term tends to 0 by Theorem B.1, since $\mathbb { V } _ { n } \to { \dot { \mathbb { V } } }$ in $L _ { \mathrm { p e r } } ^ { 1 }$ and $\left( \psi _ { n } \right)$ is bounded in $L _ { \mathrm { p e r } } ^ { 2 }$ . Consequently, the polynomial term in (50) converges to 0. Taking lim ${ \mathrm { s u p } } _ { n \to \infty }$ in $( 5 0 )$ and then letting $\varepsilon _ { p } \to 0$ gives (49). □

We next construct the Prony-series approximation used in (S2).

Lemma B.6 (Weighted Spectral Measure). Set $\lambda _ { \operatorname* { m i n } } : = \alpha _ { \mathrm { { e l a s } } } / \beta _ { \mathrm { { v i s c } } } , \ : \lambda _ { \operatorname* { m a x } } : = \beta _ { \mathrm { { e l a s } } } / \alpha _ { \mathrm { { v i s c } } } ,$ and $\tau : = [ \lambda _ { \operatorname* { m i n } } , \lambda _ { \operatorname* { m a x } } ]$ . For $( \mathbb { V } , \mathbb { E } ) \in \mathcal { M }$ , define the positive semidefinite tensor-valued measure

$$
\mathrm { d } \mu _ { \mathbb { V } , \mathbb { E } } ( \lambda ) : = \frac { \lambda } { \beta _ { \mathrm { e l a s } } / \beta _ { \mathrm { v i s c } } + \lambda } \mathrm { d } \nu _ { \mathbb { V } , \mathbb { E } } ( \lambda ) .\tag{51}
$$

Then

$$
\operatorname* { s u p } _ { ( \mathbb { V } , \mathbb { E } ) \in \mathcal { M } } \| \mu _ { \mathbb { V } , \mathbb { E } } \| _ { \mathrm { T V } ( \mathcal { Z } ) } < 2 d _ { \mathcal { V } ^ { 2 } } \beta _ { \mathrm { v i s c } } .\tag{52}
$$

Moreover, $( \mathbb { V } , \mathbb { E } ) \mapsto \mu _ { \mathbb { V } , \mathbb { E } }$ is weakly continuous under $L _ { \mathrm { p e r } } ^ { 1 }$ convergence in .

Proof. By $( 3 7 ) , \ \nu _ { \mathbb { V } , \mathbb { E } } ( B )$ is positive semidefinite for every Borel set $B \subset \mathcal { T }$ , so the same holds for $\mu _ { \mathbb { V } , \mathbb { E } } ( B )$ . For $s > 0$ , let $\mathbb { A } _ { s } = s ^ { - 1 } \mathbb { E } + \mathbb { V }$ , as in the proof of Theorem 3.1. The Laplace-transform identity there and (22) give

$$
\int _ { \mathcal { T } } \frac { \lambda } { s + \lambda } \mathrm { d } \nu _ { \mathbb { V } , \mathbb { E } } ( \lambda ) = \hat { \mathbb { K } } ( s ) = \mathbb { A } _ { s } ^ { \mathrm { h o m } } - s ^ { - 1 } \mathbb { E } ^ { \mathrm { h o m } } - \mathbb { V } ^ { \mathrm { h o m } } .
$$

The variational characterization of static homogenization, with a constant trial strain, gives $( \overline { e } , \mathbb { A } _ { s } ^ { \mathrm { h o m } } \overline { e } ) _ { F } \le ( s ^ { - 1 } \beta _ { \mathrm { e l a s } } + \beta _ { \mathrm { v i s c } } ) \| \overline { e } \| _ { F } ^ { 2 }$ . Since periodic strain correctors have zero mean, the same characterization gives $( \overline { { e } } , \mathbb { V } ^ { \mathrm { h o m } } \overline { { e } } ) _ { F } \geq \alpha _ { \mathrm { v i s c } } \Vert \overline { { e } } \Vert _ { F } ^ { 2 }$ and $( \overline { { e } } , \mathbb { E } ^ { \mathrm { { h o m } } } \overline { { e } } ) _ { F } \geq \alpha _ { \mathrm { e l a s } } \Vert \overline { { e } } \Vert _ { F } ^ { 2 }$ . At $s =$ $\beta _ { \mathrm { e l a s } } / \beta _ { \mathrm { v i s c } } ,$ these bounds imply

$$
( \overline { { e } } , \mu _ { \mathbb { V } , \mathbb { E } } ( \mathcal { T } ) \overline { { e } } ) _ { F } \leq \left( 2 \beta _ { \mathrm { v i s c } } - \alpha _ { \mathrm { v i s c } } - \frac { \beta _ { \mathrm { v i s c } } \alpha _ { \mathrm { e l a s } } } { \beta _ { \mathrm { e l a s } } } \right) \| \overline { { e } } \| _ { F } ^ { 2 } .
$$

The inequality $\| \mathbb { A } \| _ { F } \le \operatorname { t r } \mathbb { A }$ for positive semidefinite $\mathbb { A } \in \mathcal { V } ^ { 4 }$ gives $\| \mu _ { \mathbb { V } , \mathbb { E } } \| _ { \mathrm { T V } ( \mathcal { Z } ) } \le \mathrm { t r } \mu _ { \mathbb { V } , \mathbb { E } } ( \mathcal { Z } )$ as in the proof of Theorem 3.4. Taking the trace in the preceding bound and using $\alpha _ { \mathrm { v i s c } } , \alpha _ { \mathrm { e l a s } } > 0$ proves (52). Finally, for every $f \in C ( \mathcal { T } )$ , the function $\lambda \mapsto f ( \lambda ) \lambda / ( \beta _ { \mathrm { e l a s } } / \beta _ { \mathrm { v i s c } } +$ $\lambda )$ is continuous. Weak continuity follows from Theorem B.5. □

Lemma B.7 (Weighted Relaxation Response). For ${ \overline { { e } } } \in { \mathcal { E } } , t \geq 0$ , and $\lambda \in \mathbb { C }$ , define

$$
\mathcal { W } _ { \overline { { e } } , t } ( \lambda ) : = \left( \frac { \beta _ { \mathrm { { e l a s } } } } { \beta _ { \mathrm { { v i s c } } } } + \lambda \right) \int _ { 0 } ^ { t } \exp ( - \lambda ( t - s ) ) \dot { \overline { { e } } } ( s ) \mathrm { d } s .\tag{53}
$$

For each fixed $\overline { { e } }$ and t, every tensor component of $\mathcal { W } _ { \overline { { e } } , t } ( \lambda )$ is an entire function of λ. If $\mathrm { R e } \lambda > 0$ , then

$$
\operatorname* { s u p } _ { \overline { { e } } \in \mathcal { E } } \operatorname* { s u p } _ { t \geq 0 } \Vert \mathcal { W } _ { \overline { { e } } , t } ( \lambda ) \Vert _ { F } \leq \left( 1 + \frac { | \lambda | } { \mathrm { R e } \lambda } \right) \frac { \delta _ { \mathrm { s t r e s s } } } { \beta _ { \mathrm { v i s c } } } .\tag{54}
$$

Here the Frobenius norm for complex tensors is $\begin{array} { r } { \| Z \| _ { F } = ( \sum _ { i , j = 1 } ^ { d } | Z _ { i j } | ^ { 2 } ) ^ { 1 / 2 } } \end{array}$

Proof. For each $t \geq 0$ , diferentiation under the integral on the finite interval $[ 0 , t ]$ proves analyticity. Write $r =$ Re $\lambda > 0 , c = | \lambda | / r \geq 1$ , and $\begin{array} { r } { z ( t ) = \int _ { 0 } ^ { t } \exp ( - \lambda ( t - s ) ) \dot { \bar { e } } ( s ) } \end{array}$ ds. Integration by parts gives

$$
z ( t ) = \overline { { e } } ( t ) - \exp ( - \lambda t ) \overline { { e } } ( 0 ) - \lambda \int _ { 0 } ^ { t } \exp ( - \lambda ( t - s ) ) \overline { { e } } ( s ) \mathrm { d } s .
$$

Consequently, $\| z ( t ) \| _ { F } \leq [ 1 + \exp ( - r t ) + c ( 1 - \exp ( - r t ) ) ] \| \bar { e } \| _ { L ^ { \infty } } \leq ( 1 + c ) \| \bar { e } \| _ { L ^ { \infty } }$ . By the definition of $z ,$ , we also have $| \lambda | \| z ( t ) \| _ { F } \leq c \| \dot { \overline { { e } } } \| _ { L ^ { \infty } }$ . Hence

$$
\| \mathcal { W } _ { \overline { { e } } , t } ( \lambda ) \| _ { F } \leq \frac { \beta _ { \mathrm { e l a s } } } { \beta _ { \mathrm { v i s c } } } ( 1 + c ) \| \overline { { e } } \| _ { L ^ { \infty } } + c \| \dot { \overline { { e } } } \| _ { L ^ { \infty } } \leq \frac { 1 + c } { \beta _ { \mathrm { v i s c } } } \left( \beta _ { \mathrm { e l a s } } \| \overline { { e } } \| _ { L ^ { \infty } } + \beta _ { \mathrm { v i s c } } \| \dot { \overline { { e } } } \| _ { L ^ { \infty } } \right) .
$$

The definition of $\mathcal { E }$ gives (54).

Definition B.8 (Prony Series by Local Chebyshev Interpolation). A Prony-series approximation of $\mathbb { K } ( \circ ; \mathbb { V } , \mathbb { E } )$ is a kernel

$$
\mathbb { K } ^ { ( n ) } ( t ; \mathbb { V } , \mathbb { E } ) : = \sum _ { k = 1 } ^ { n } \mathbb { K } _ { k } ( \mathbb { V } , \mathbb { E } ) \exp ( - \lambda _ { k } t ) ,\tag{55}
$$

with microstructure-independent rates $\lambda _ { k } \in { \mathcal { T } } . \ I f \lambda _ { \operatorname* { m i n } } = \lambda _ { \operatorname* { m a x } }$ , set $n = 1 , \lambda _ { 1 } = \lambda _ { \operatorname* { m i n } }$ , and

$$
\mathbb { K } _ { 1 } ( \mathbb { V } , \mathbb { E } ) : = \int _ { \mathbb { Z } } \lambda \mathrm { d } \nu _ { \mathbb { V } , \mathbb { E } } ( \lambda ) .\tag{56}
$$

Otherwise, set $\mathsf { R } = \ln ( \lambda _ { \mathrm { m a x } } / \lambda _ { \mathrm { m i n } } ) , L = \lceil \mathsf { R } \rceil , h = \mathsf { R } / L \leq 1$ , and $J = L + 1$ . For $j = 1 , \dots , J$ define

$$
\begin{array} { r l } & { \tau _ { j } = ( j - 1 ) h , \qquad a _ { j } = \operatorname* { m a x } \{ 0 , \tau _ { j } - h \} , \qquad b _ { j } = \operatorname* { m i n } \{ \mathsf { R } , \tau _ { j } + h \} , } \\ & { \qquad \theta _ { j } ( \lambda ) = \operatorname* { m a x } \left\{ 1 - \frac { \big | \ln ( \lambda / \lambda _ { \operatorname* { m i n } } ) - \tau _ { j } \big | } { h } , 0 \right\} , \qquad \lambda \in \mathcal { T } . } \end{array}
$$

The functions $\theta _ { j }$ are continuous and nonnegative, satisfy $\textstyle \sum _ { j = 1 } ^ { J } \theta _ { j } = 1$ on $\mathcal { T } _ { i }$ and are supported on $\mathcal { T } _ { j } = [ \lambda _ { \mathrm { m i n } } \exp ( a _ { j } ) , \lambda _ { \mathrm { m i n } } \exp ( b _ { j } ) ]$ . For the reference coordinate $x \in [ - 1 , 1 ]$ , define the local changes of variables by

$$
g _ { j } ( x ) = \lambda _ { \mathrm { m i n } } \exp \left( \frac { a _ { j } + b _ { j } } { 2 } + \frac { b _ { j } - a _ { j } } { 2 } x \right) , \qquad \psi _ { j } ( \lambda ) = \frac { 2 \ln ( \lambda / \lambda _ { \mathrm { m i n } } ) - a _ { j } - b _ { j } } { b _ { j } - a _ { j } } .\tag{57}
$$

For $m \in \mathbb { N }$ , let $x _ { i } = \cos ( ( 2 i - 1 ) \pi / ( 2 m ) ) , i = 1 , \ldots , m$ , be the Chebyshev nodes, and let $\ell _ { i }$ be the associated Lagrange basis polynomials. Set

$$
\lambda _ { j i } = g _ { j } ( x _ { i } ) , \qquad \mathbb { K } _ { j i } ( \mathbb { V } , \mathbb { E } ) = \left( \frac { \beta _ { \mathrm { e l a s } } } { \beta _ { \mathrm { v i s c } } } + \lambda _ { j i } \right) \int _ { T } \theta _ { j } ( \lambda ) \ell _ { i } ( \psi _ { j } ( \lambda ) ) \mathrm { d } \mu _ { \mathbb { V } , \mathbb { E } } ( \lambda ) .\tag{58}
$$

Use the single index $k = ( j - 1 ) m + i ;$ , so that $n = J m , \lambda _ { k } = \lambda _ { j i }$ , and ${ \mathbb K } _ { k } = { \mathbb K } _ { j i }$

Lemma B.9 (Local Approximation of the Weighted Response). Suppose $\lambda _ { \operatorname* { m i n } } < \lambda _ { \operatorname* { m a x } }$ , and use the response in (53) and the maps and nodes in Theorem B.8. For ${ \overline { { e } } } \in { \mathcal { E } } , t \geq 0 .$ , and $j = 1 , \dots , J _ { \mathrm { { \scriptscriptstyle 3 } } }$ , set $f _ { j } ( x ) = \mathcal { W } _ { \bar { e } , t } ( g _ { j } ( x ) )$ . The interpolant $\begin{array} { r } { p _ { j } ( x ) = \sum _ { i = 1 } ^ { m } \ell _ { i } ( x ) f _ { j } ( x _ { i } ) } \end{array}$ has degree at most $m - 1$ and satisfies

$$
\operatorname* { s u p } _ { \bar { e } \in \mathcal { E } } \operatorname* { s u p } _ { t \geq 0 } \operatorname* { m a x } _ { 1 \leq j \leq J } \operatorname* { s u p } _ { x \in [ - 1 , 1 ] } \| f _ { j } ( x ) - p _ { j } ( x ) \| _ { F } \leq \frac { 4 \delta _ { \operatorname { s t r e s s } } } { \beta _ { \operatorname { v i s c } } } \left( 1 + \sec \frac { 3 } { 4 } \right) 2 ^ { - ( m - 1 ) } .\tag{59}
$$

Proof. Let $B _ { 2 }$ be the closed Bernstein ellipse with parameter 2. Its imaginary semiaxis is $3 / 4$ . Since $( b _ { j } - a _ { j } ) / 2 \leq h \leq 1$ , (57) gives arg $g _ { j } ( z ) | \le 3 / 4$ on $B _ { 2 }$ . In particular, Re $g _ { j } ( z ) > 0$ and $| g _ { j } ( z ) | / \mathrm { R e } g _ { j } ( z ) \le \sec ( 3 / 4 )$ . By Theorem B.7, $f _ { j }$ is analytic on a neighborhood of $B _ { 2 }$ and

$$
\operatorname* { s u p } _ { z \in \mathcal { B } _ { 2 } } \| f _ { j } ( z ) \| _ { F } \leq \frac { \delta _ { \mathrm { s t r e s s } } } { \beta _ { \mathrm { v i s c } } } \left( 1 + \sec \frac { 3 } { 4 } \right) .
$$

Denote this bound by $M _ { f }$ . For any complex linear functional $\varphi$ of norm one, the Chebyshev coeficients $c _ { k }$ of $\varphi \circ f _ { j }$ satisfy $| c _ { k } | \le 2 M _ { f } 2 ^ { - k }$ [56, Theorem 8.1]. Let $I _ { m }$ denote interpolation at the roots of $T _ { m }$ . Each Chebyshev polynomial $T _ { k }$ interpolates to zero or to $\pm T _ { r }$ for some $r < m$ , so $\| I _ { m } T _ { k } \| _ { L ^ { \infty } ( - 1 , 1 ) } \leq 1$ . Since interpolation is exact for $k < m$ 2

$$
\| \varphi \circ f _ { j } - I _ { m } ( \varphi \circ f _ { j } ) \| _ { L ^ { \infty } ( - 1 , 1 ) } \leq 2 \sum _ { k = m } ^ { \infty } | c _ { k } | \leq 4 M _ { f } 2 ^ { - ( m - 1 ) } .
$$

Taking the supremum over $\varphi$ proves (59).

Proposition B.10 (Uniform Approximation by the Prony-Series Internal-Variable Model). Let the rates and coeficients be given by Theorem $B . 8 .$ Each coeficient ${ \mathbb K } _ { k }$ depends continuously on $( \mathbb { V } , \mathbb { E } ) \ \in \ { \mathcal { M } }$ in the $L _ { \mathrm { p e r } } ^ { 1 }$ topology. For $( \mathbb { V } , \mathbb { E } ) \ \in \ { \mathcal { M } }$ and $\overline { { e } } \in \mathcal { E }$ , define $q _ { k } \in W ^ { 1 , \infty } ( \tau ; \nu ^ { 2 } )$ by

$$
\begin{array} { r } { \dot { q } _ { k } ( t ) + \lambda _ { k } q _ { k } ( t ) = \mathbb { K } _ { k } ( \mathbb { V } , \mathbb { E } ) \dot { \overline { { e } } } ( t ) , \qquad q _ { k } ( 0 ) = 0 , \qquad k = 1 , \dots , n , } \end{array}\tag{60}
$$

and set

$$
\overline { { \sigma } } _ { \mathrm { I V } } ( t ) = \mathbb { V } ^ { \mathrm { h o m } } ( \mathbb { V } ) \dot { \overline { { e } } } ( t ) + \mathbb { E } ^ { \mathrm { h o m } } ( \mathbb { E } ) \overline { { e } } ( t ) + \sum _ { k = 1 } ^ { n } q _ { k } ( t ) .\tag{61}
$$

Denote the resulting constitutive model by $\mathcal { H } _ { \mathrm { I V } }$ . This is the homogenized constitutive law with K replaced by $\mathbb { K } ^ { ( n ) }$ $I f \lambda _ { \operatorname* { m i n } } = \lambda _ { \operatorname* { m a x } } .$ , the approximation is exact with $n = 1$ . Otherwise, with $n = J m$ 7

$$
\operatorname* { s u p } _ { ( \mathbb { V } , \mathbb { E } ) \in \mathcal { M } } \operatorname* { s u p } _ { \bar { e } \in \mathcal { E } } \operatorname* { s s s } _ { t \in \mathcal { T } } \Vert \mathcal { H } ( \{ \overline { { e } } ( s ) \} _ { s \in [ 0 , t ] } ; \mathbb { V } , \mathbb { E } ) - \overline { { \sigma } } _ { \mathrm { I V } } ( t ) \Vert _ { F } < \kappa _ { \mathrm { I V } } 2 ^ { - ( m - 1 ) } ,\tag{62}
$$

$$
\kappa _ { \mathrm { I V } } : = 8 d _ { \mathcal { V } ^ { 2 } } \left( 1 + \sec \frac { 3 } { 4 } \right) \delta _ { \mathrm { s t r e s s } } .
$$

Proof. For a singleton interval, coeficient continuity follows from (56) and Theorem B.5. Otherwise, each function $\lambda \mapsto \theta _ { j } ( \lambda ) \ell _ { i } ( \psi _ { j } ( \lambda ) )$ is continuous on , and the rates are independent of (V, E). Coeficient continuity then follows from (58) and Theorem B.6.

Duhamel’s formula gives

$$
q _ { k } ( t ) = \int _ { 0 } ^ { t } \exp ( - \lambda _ { k } ( t - s ) ) \mathbb { K } _ { k } ( \mathbb { V } , \mathbb { E } ) { \dot { \overline { { e } } } } ( s ) { \mathrm { d } } s .\tag{63}
$$

Since $\lambda _ { k } > 0$ and $\dot { \bar { e } } \in L ^ { \infty }$ , this solution is bounded, and (60) bounds its derivative. Summing (63) and using (55) proves the asserted constitutive representation. If  is a singleton, (22) and (56) give exactness.

For a nondegenerate interval, abbreviate $\mu ~ = ~ \mu _ { \mathbb { V } , \mathbb { E } }$ . By (22) and (51), the memory contribution to the stress equals $\begin{array} { r } { \int _ { \mathbb { Z } } \mathrm { d } \mu ( \lambda ) \mathcal { W } _ { \overline { { e } } , t } ( \lambda ) } \end{array}$ , where the tensor measure acts on the vector-valued integrand. The coeficients in (58) give the corresponding contribution in the Prony-series model, $\begin{array} { r l } { \sum _ { j = 1 } ^ { J } \int _ { \mathcal { T } } \theta _ { j } ( \boldsymbol { \lambda } ) \mathrm { d } \mu ( \boldsymbol { \lambda } ) p _ { j } ( \psi _ { j } ( \boldsymbol { \lambda } ) ) } & { { } } \end{array}$ . Hence their diference is

$$
\sum _ { j = 1 } ^ { J } \int _ { \mathcal { T } } \theta _ { j } ( \boldsymbol { \lambda } ) \mathrm { d } \mu ( \boldsymbol { \lambda } ) \big [ f _ { j } ( \psi _ { j } ( \boldsymbol { \lambda } ) ) - p _ { j } ( \psi _ { j } ( \boldsymbol { \lambda } ) ) \big ] .
$$

The interpolation bound applies on the support of $\theta _ { j }$ , where $\psi _ { j } ( \lambda ) \in [ - 1 , 1 ]$ . If $| \mu |$ denotes the total variation measure, the norm of this diference is at most

$$
\begin{array} { r l r } {  { \frac { 4 \delta _ { \mathrm { s t r e s s } } } { \beta _ { \mathrm { v i s c } } } ( 1 + \sec { \frac { 3 } { 4 } } ) 2 ^ { - ( m - 1 ) } \sum _ { j = 1 } ^ { J } \int _ { \mathbb { Z } } \theta _ { j } ( \lambda ) \mathrm { d } | \mu | ( \lambda ) = \frac { 4 \delta _ { \mathrm { s t r e s s } } } { \beta _ { \mathrm { v i s c } } } ( 1 + \sec { \frac { 3 } { 4 } } ) 2 ^ { - ( m - 1 ) } \| \mu \| _ { \mathrm { T V } ( \mathbb { Z } ) } } } \\ & { } & { < \kappa _ { \mathrm { I V } } 2 ^ { - ( m - 1 ) } , } \end{array}
$$

by the partition of unity and the uniform mass bound (52). The instantaneous terms coincide, so this proves (62). □

We now approximate the stress and kinetic responses of the Prony-series model and bound the resulting errors in the internal variables and stress, as required in (S3).

Lemma B.11 (Uniform Bound for the Internal Variable). Choose an approximation from Theorem B.8 with n terms, and let $\xi _ { \mathrm { I V } } : = \mathsf { v e c } _ { 2 } ^ { ( n ) } ( q _ { 1 } , \hdots , q _ { n } ) \in \mathbb { R } ^ { n d _ { \nu ^ { 2 } } }$ , where the $q _ { k }$ solve (60) and $\mathsf { v e c } _ { 2 } ^ { ( n ) }$ concatenates their Mandel vectors. There is a constant $\delta _ { \xi _ { \mathrm { I V } } } > 0$ such that

$$
\operatorname* { s u p } _ { ( \mathbb { V } , \mathbb { E } ) \in \mathcal { M } } \operatorname* { s u p } _ { \overline { { e } } \in \mathcal { E } } \| \xi _ { \mathrm { I V } } \| _ { L ^ { \infty } } \leq \delta _ { \xi _ { \mathrm { I V } } } .
$$

Proof. By (63), $\lambda _ { k } \geq \lambda _ { \operatorname* { m i n } } > 0$ , and (28),

$$
\| q _ { k } ( t ) \| _ { F } \leq \frac { \delta _ { \mathrm { s t r e s s } } } { \beta _ { \mathrm { v i s c } } \lambda _ { \operatorname* { m i n } } } \| \mathbb { K } _ { k } ( \mathbb { V } , \mathbb { E } ) \| _ { F } , \qquad k = 1 , \dots , n .
$$

Continuity of each coeficient and compactness of (Theorems B.2 and B.10) give $C _ { k } : =$ $\mathrm { s u p } _ { ( \mathbb { V } , \mathbb { E } ) \in \mathcal { M } } \| \mathbb { K } _ { k } ( \mathbb { V } , \mathbb { E } ) \| _ { F } < \infty$ . Since $\mathsf { v e c } _ { 2 } ^ { ( n ) }$ is an isometry, one may take

$$
\delta _ { \xi _ { \mathrm { I V } } } = 1 + \frac { \delta _ { \mathrm { s t r e s s } } } { \beta _ { \mathrm { v i s c } } \lambda _ { \mathrm { m i n } } } \left( \sum _ { k = 1 } ^ { n } C _ { k } ^ { 2 } \right) ^ { 1 / 2 } .
$$

Lemma B.12 (Uniform Approximation by Neural Operator Response Mappings). Consider an approximation from Theorem B.8 with n terms. For $\eta _ { \xi } > 0$ , let $\delta _ { \xi _ { \mathrm { I V } } } > 0$ be given by Theorem B.11. For $\xi \in \mathbb { R } ^ { n d _ { \nu ^ { 2 } } }$ , write $\xi = \mathsf { v e c } _ { 2 } ^ { ( n ) } ( q _ { 1 } , \dots , q _ { n } )$ with $q _ { 1 } , \dotsc , q _ { n } \in \mathcal { V } ^ { 2 }$ , and define the stress and kinetic responses associated with the Prony-series internal-variable model in Theorem B.10 by

$$
\begin{array} { r l } & { \mathcal { F } _ { \mathrm { I V } } ( \overline { { e } } , \dot { \overline { { e } } } , \xi ; \mathbb { V } , \mathbb { E } ) : = \mathbb { V } ^ { \mathrm { h o m } } ( \mathbb { V } ) \dot { \overline { { e } } } + \mathbb { E } ^ { \mathrm { h o m } } ( \mathbb { E } ) \overline { { e } } + \displaystyle \sum _ { i = 1 } ^ { n } q _ { i } , } \\ & { \quad \mathcal { G } _ { \mathrm { I V } } ( \dot { \overline { { e } } } , \xi ; \mathbb { V } , \mathbb { E } ) : = \mathsf { v e c } _ { 2 } ^ { ( n ) } \big ( \mathbb { E } _ { 1 } ( \mathbb { V } , \mathbb { E } ) \dot { \overline { { e } } } - \lambda _ { 1 } q _ { 1 } , \dots , \mathbb { K } _ { n } ( \mathbb { V } , \mathbb { E } ) \dot { \overline { { e } } } - \lambda _ { n } q _ { n } \big ) . } \end{array}
$$

Define the compact input set

$$
\begin{array} { r } { K _ { \mathrm { v e c } } : = \big \{ ( \bar { e } , \dot { \bar { e } } , \xi ) \in \mathcal { V } ^ { 2 } \times \mathcal { V } ^ { 2 } \times \mathbb { R } ^ { n d _ { \mathcal { V } ^ { 2 } } } \ \big | \ \beta _ { \mathrm { e l a s } } \| \overline { { e } } \| _ { F } + \beta _ { \mathrm { v i s c } } \| \dot { \bar { e } } \| _ { F } \leq \delta _ { \mathrm { s t r e s s } } , \| \xi \| \leq \delta _ { \xi _ { \mathrm { I V } } } + \eta _ { \xi } \big \} . } \end{array}
$$

Then, for any $\varepsilon _ { \mathcal { F } } , \varepsilon _ { \mathcal { G } } > 0$ , there exist response mappings $( \mathcal { F } _ { \mathrm { N O } } , \mathcal { G } _ { \mathrm { N O } } ) \in \mathfrak { F } ( n d _ { \mathcal { V } ^ { 2 } } ) \times \mathfrak { G } ( n d _ { \mathcal { V } ^ { 2 } } )$ such that

$$
\operatorname* { s u p } _ { ( \overline { { e } } , \overline { { e } } , \xi ) \in K _ { \mathrm { v e c } } } \left\| \mathcal { F } _ { \mathrm { N O } } ( \overline { { e } } , \dot { \overline { { e } } } , \xi ; \mathbb { V } , \mathbb { E } ) - \mathcal { F } _ { \mathrm { I V } } ( \overline { { e } } , \dot { \overline { { e } } } , \xi ; \mathbb { V } , \mathbb { E } ) \right\| _ { F } \leq \varepsilon _ { \mathcal { F } } ,\tag{64}
$$

$$
\begin{array} { r l } & { \underset { ( \overline { { e } } , \overline { { e } } , \xi ) \in { \cal K } _ { \mathrm { v e c } } } { \operatorname* { s u p } } \left\| \mathcal { G } _ { \mathrm { N O } } ( \dot { \overline { { e } } } , \xi ; \mathbb { V } , \mathbb { E } ) - \mathcal { G } _ { \mathrm { I V } } ( \dot { \overline { { e } } } , \xi ; \mathbb { V } , \mathbb { E } ) \right\| \leq \varepsilon _ { \mathcal { G } } . } \end{array}\tag{65}
$$

Moreover, for every $( \mathbb { V } , \mathbb { E } ) \in \mathcal { M }$ , the map $( \dot { \bar { e } } , \xi ) \mapsto \mathcal G _ { \mathrm { N O } } ( \dot { \bar { e } } , \xi ; \mathbb { V } , \mathbb { E } )$ is locally Lipschitz.

Proof. The set $\scriptstyle { { \mathcal { K } } _ { \mathrm { v e c } } }$ is compact by definition, and Theorem B.2 gives the compactness of . By Theorems B.4 and B.10, the maps $( \mathbb { V } , \mathbb { E } ) \mapsto \mathbb { V } ^ { \mathrm { h o m } } ( \mathbb { V } ) , ( \mathbb { V } , \mathbb { E } ) \mapsto \mathbb { E } ^ { \mathrm { h o m } } ( \mathbb { E } )$ , and $( \mathbb { V } , \mathbb { E } ) \mapsto \mathbb { K } _ { i } ( \mathbb { V } , \mathbb { E } )$ are continuous on , so $\mathcal { F } _ { \mathrm { { I V } } }$ and are continuous on ${ \mathcal { K } } _ { \mathrm { v e c } } \times { \mathcal { M } }$

Given $\varepsilon _ { \mathcal { F } } , \varepsilon _ { \mathcal { G } } > 0$ , define $\widehat { \mathcal { K } } _ { \mathcal { F } } : = \left\{ \big ( \mathsf { v e c } _ { 2 } \overline { { e } } , \mathsf { v e c } _ { 2 } \dot { \overline { { e } } } , \xi \big ) \ | \ \big ( \overline { { e } } , \dot { \overline { { e } } } , \xi \big ) \in \mathcal { K } _ { \mathrm { v e c } } \right\} , \ \widehat { \mathcal { K } } _ { \mathcal { G } } : = \left\{ \big ( \mathsf { v e c } _ { 2 } \dot { \overline { { e } } } , \xi \big ) \ | \ \right\}$ $( \overline { { e } } , \dot { \overline { { e } } } , \xi ) \in \mathcal { K } _ { \mathrm { v e c } } \}$ , and $\widehat { \mathcal { M } } : = \{ ( \mathsf { v e c } _ { 4 } \circ \mathbb { V } , \mathsf { v e c } _ { 4 } \circ \mathbb { E } ) \ | \ ( \mathbb { V } , \mathbb { E } ) \in \mathcal { M } \}$ b. These sets are compact as continuous images of $\kappa _ { \mathrm { v e c } }$ and $\mathcal { M }$ under the relevant vectorization maps. Define

$$
\begin{array} { r l } & { g _ { \mathcal { F } } \Big ( \big ( { \mathsf { v e c } } _ { 2 } \overline { { e } } , { \mathsf { v e c } } _ { 2 } \dot { \overline { { e } } } , \xi \big ) , \big ( { \mathsf { v e c } } _ { 4 } \circ { \mathbb { V } } , { \mathsf { v e c } } _ { 4 } \circ { \mathbb { E } } \big ) \Big ) : = { \mathsf { v e c } } _ { 2 } \big ( \mathcal { F } _ { \mathrm { I V } } \big ( \overline { { e } } , \dot { \overline { { e } } } , \xi ; { \mathbb { V } } , { \mathbb { E } } \big ) \big ) , } \\ & { \qquad g _ { \mathcal { G } } \Big ( \big ( { \mathsf { v e c } } _ { 2 } \dot { \overline { { e } } } , \xi \big ) , \big ( { \mathsf { v e c } } _ { 4 } \circ { \mathbb { V } } , { \mathsf { v e c } } _ { 4 } \circ { \mathbb { E } } \big ) \Big ) : = \mathcal { G } _ { \mathrm { I V } } ( \dot { \overline { { e } } } , \xi ; { \mathbb { V } } , { \mathbb { E } } ) . } \end{array}
$$

Then $g _ { \mathcal { F } }$ and $g _ { \mathcal { G } }$ are continuous on $\widehat { \mathcal { K } } _ { \mathcal { F } } \times \widehat { \mathcal { M } }$ and ${ \widehat { \cal K } } _ { \mathcal { G } } \times { \widehat { \cal M } }$ , respectively. Apply (A1) to $g _ { \mathcal { F } }$ and (A2) to $g _ { \mathcal { G } }$ to obtain maps $\Phi _ { \mathcal { F } }$ band $\Phi _ { \mathcal { G } }$ c bsatisfying

$$
\operatorname* { s u p } _ { \widehat { K } _ { \mathcal { F } } \times \widehat { \mathcal { M } } } \| \Phi _ { \mathcal { F } } - g _ { \mathcal { F } } \| \leq \varepsilon _ { \mathcal { F } } , \qquad \operatorname* { s u p } _ { \widehat { K } _ { \mathcal { G } } \times \widehat { \mathcal { M } } } \| \Phi _ { \mathcal { G } } - g _ { \mathcal { G } } \| \leq \varepsilon _ { \mathcal { G } } ,
$$

with $\Phi _ { \mathcal { G } }$ locally Lipschitz in its vector input for each microstructure. Let $\left( \mathcal { F } _ { \mathrm { N O } } , \mathcal { G } _ { \mathrm { N O } } \right) \in$ $\mathfrak { F } ( n d _ { \mathcal { V } ^ { 2 } } ) \times \mathfrak { G } ( n d _ { \mathcal { V } ^ { 2 } } )$ be the induced response mappings. The isometry of vec<sub>2</sub> gives (64) and (65) and preserves the local Lipschitz property of $\mathcal { G } _ { \mathrm { N O } }$ □

Proposition B.13 (Global Existence and Uniform Error Bounds). Consider an approximation from Theorem B.8 with n terms. Let $\varepsilon _ { \mathcal { F } } , \eta _ { \xi } > 0$ , let $0 < \varepsilon _ { \mathcal { G } } \leq \lambda _ { \operatorname* { m i n } } \eta _ { \xi }$ , and choose $\left( \mathcal { F } _ { \mathrm { N O } } , \mathcal { G } _ { \mathrm { N O } } \right)$ as in Theorem B.12. For every $( \mathbb { V } , \mathbb { E } ) \in \mathcal { M }$ and $\overline { { e } } \in \mathcal { E }$ , the initial value problem

$$
\dot { \xi } _ { \mathrm { N O } } ( t ) = \mathcal { G } _ { \mathrm { N O } } ( \dot { \overline { { e } } } ( t ) , \xi _ { \mathrm { N O } } ( t ) ; \mathbb { V } , \mathbb { E } ) \quad f o r ~ a . e . ~ t \in \mathcal { T } , \qquad \xi _ { \mathrm { N O } } ( 0 ) = 0 ,
$$

admits a unique solution $\xi _ { \mathrm { N O } } ~ \in ~ W ^ { 1 , \infty } ( \mathcal { T } ; \mathbb { R } ^ { n d _ { \mathcal { V } ^ { 2 } } } )$ . Let $\overline { { \sigma } } _ { \mathrm { N O } }$ be its stress response in (10). Then

$$
\operatorname* { s u p } _ { ( \mathbb { V } , \mathbb { E } ) \in \mathcal { M } } \operatorname* { s u p } _ { \overline { { e } } \in \mathcal { E } } \| \xi _ { \mathrm { N O } } - \xi _ { \mathrm { I V } } \| _ { L ^ { \infty } } \leq \frac { \varepsilon _ { \mathscr { G } } } { \lambda _ { \operatorname* { m i n } } } ,
$$

$$
\operatorname* { s u p } _ { ( \mathbb { V } , \mathbb { E } ) \in \mathcal { M } } \operatorname* { s u p } _ { \overline { { e } } \in \mathcal { E } } \| \overline { { \sigma } } _ { \mathrm { N O } } - \overline { { \sigma } } _ { \mathrm { I V } } \| _ { L ^ { \infty } } \leq \varepsilon _ { \mathcal { F } } + \frac { \sqrt { n } } { \lambda _ { \operatorname* { m i n } } } \varepsilon _ { \mathcal { G } } ,\tag{66}
$$

where $\xi _ { \mathrm { I V } }$ is the internal variable defined in Theorem B.11 and $\overline { { \sigma } } _ { \mathrm { I V } }$ is given by (61).

Proof. Let $( \mathbb { V } , \mathbb { E } ) \in \mathcal { M }$ and ${ \overline { { e } } } \in { \mathcal { E } }$ . For $j = 1 , 2$ , write $\xi _ { j } = \mathsf { v e c } _ { 2 } ^ { ( n ) } ( q _ { j , 1 } , \ldots , q _ { j , n } )$ . The definition of $\mathcal { G } _ { \mathrm { I V } }$ in Theorem B.12 gives

$$
\mathcal G _ { \mathrm { I V } } ( \dot { \bar { e } } , \xi _ { 1 } ; \Psi , \mathbb E ) - \mathcal G _ { \mathrm { I V } } ( \dot { \bar { e } } , \xi _ { 2 } ; \Psi , \mathbb E ) = \mathrm { v e c } _ { 2 } ^ { ( n ) } \big ( - \lambda _ { 1 } ( q _ { 1 , 1 } - q _ { 2 , 1 } ) , \dots , - \lambda _ { n } ( q _ { 1 , n } - q _ { 2 , n } ) \big ) .
$$

Since $\mathsf { v e c } _ { 2 } ^ { ( n ) }$ is an isometry,

$$
\big ( \xi _ { 1 } - \xi _ { 2 } \big ) ^ { \top } \Big ( \mathcal G _ { \mathrm { I V } } \big ( \dot { \vec { e } } , \xi _ { 1 } ; \mathbb { V } , \mathbb { E } \big ) - \mathcal G _ { \mathrm { I V } } \big ( \dot { \vec { e } } , \xi _ { 2 } ; \mathbb { V } , \mathbb { E } \big ) \Big ) \leq - \lambda _ { \operatorname* { m i n } } \| \xi _ { 1 } - \xi _ { 2 } \| ^ { 2 } .\tag{67}
$$

By Theorem B.12, the map $( \dot { \bar { e } } , \xi ) \ \mapsto \ \mathcal G _ { \mathrm { N O } } ( \dot { \bar { e } } , \xi ; \mathbb { V } , \mathbb { E } )$ is locally Lipschitz. Since $\dot { \bar { e } } \in$ $L ^ { \infty } ( \mathcal { T } ; \mathcal { V } ^ { 2 } )$ , Carathéodory’s theorem gives a unique maximal locally absolutely continuous solution $\xi _ { \mathrm { N O } }$ on $[ 0 , T _ { \mathrm { m a x } } )$ , where $T _ { \mathrm { m a x } }$ denotes its maximal existence time. Set $\zeta ( t ) : = \xi _ { \mathrm { N O } } ( t ) -$ $\xi _ { \mathrm { I V } } ( t )$ and define

$$
t _ { * } : = \operatorname* { s u p } \left\{ 0 \leq \tau < T _ { \operatorname* { m a x } } \mid \Vert \zeta ( t ) \Vert \leq \eta _ { \xi } { \mathrm { ~ f o r ~ a l l ~ } } t \in [ 0 , \tau ] \right\} .
$$

Since $\zeta ( 0 ) = 0$ , we have $t \mathrm { ~ , ~ } > ~ 0$ . On $[ 0 , t _ { * } )$ , Theorem B.11 and the definition of $\mathcal { E }$ give $( \overline { { e } } ( t ) , \dot { \overline { { e } } } ( t ) , \xi _ { \mathrm { N O } } ( t ) ) \in \mathcal { K } _ { \mathrm { v e c } }$ for a.e. t. Hence (65) applies there. For a.e. $t \in [ 0 , t _ { * } )$

$$
\begin{array} { r l } & { \dot { \zeta } ( t ) = \Big ( \mathcal { G } _ { \mathrm { N O } } ( \dot { \overline { { e } } } ( t ) , \xi _ { \mathrm { N O } } ( t ) ; \mathbb { V } , \mathbb { E } ) - \mathcal { G } _ { \mathrm { I V } } ( \dot { \overline { { e } } } ( t ) , \xi _ { \mathrm { N O } } ( t ) ; \mathbb { V } , \mathbb { E } ) \Big ) } \\ & { \qquad + \left( \mathcal { G } _ { \mathrm { I V } } ( \dot { \overline { { e } } } ( t ) , \xi _ { \mathrm { N O } } ( t ) ; \mathbb { V } , \mathbb { E } ) - \mathcal { G } _ { \mathrm { I V } } ( \dot { \overline { { e } } } ( t ) , \xi _ { \mathrm { I V } } ( t ) ; \mathbb { V } , \mathbb { E } ) \right) . } \end{array}
$$

Taking the Euclidean inner product with ζ(t) and using (65) and (67) gives

$$
\frac 1 2 \frac { \mathrm { d } } { \mathrm { d } t } \| \zeta ( t ) \| ^ { 2 } \leq \varepsilon _ { \mathcal { G } } \| \zeta ( t ) \| - \lambda _ { \operatorname* { m i n } } \| \zeta ( t ) \| ^ { 2 } .
$$

Therefore, for a.e. $t \in [ 0 , t _ { * } )$ such that $\begin{array} { r } { \| \zeta ( t ) \| > 0 , \frac { \mathrm { d } } { \mathrm { d } t } \| \zeta ( t ) \| \le \varepsilon _ { \mathcal G } - \lambda _ { \operatorname* { m i n } } \| \zeta ( t ) \| } \end{array}$ . Since $\zeta ( 0 ) = 0$ comparison yields

$$
\| \zeta ( t ) \| \le \lambda _ { \operatorname* { m i n } } ^ { - 1 } \varepsilon _ { \mathcal { G } } \big ( 1 - \exp ( - \lambda _ { \operatorname* { m i n } } t ) \big ) \le \lambda _ { \operatorname* { m i n } } ^ { - 1 } \varepsilon _ { \mathcal { G } }\tag{68}
$$

on $[ 0 , t _ { * } )$ . If $t _ { * } ~ < ~ T _ { \mathrm { m a x } }$ , continuity and $\varepsilon _ { \mathcal { G } } \leq \lambda _ { \mathrm { m i n } } \eta _ { \xi } \mathrm { ~ g i v e ~ } \| \zeta ( t _ { * } ) \| < \eta _ { \xi }$ , contradicting the definition of $t _ { * }$ . Thus $t _ { * } = T _ { \operatorname* { m a x } }$ . Since $\xi _ { \mathrm { N O } }$ remains bounded by $\delta _ { \xi _ { \mathrm { I V } } } + \eta _ { \xi }$ and e<sup>˙</sup> is bounded, the continuation criterion for Carathéodory ODEs gives $T _ { \mathrm { m a x } } = \infty$ . Moreover, (65) and the boundedness of $\mathcal { G } _ { \mathrm { I V } }$ on ${ \mathcal { K } } _ { \mathrm { v e c } } \times { \mathcal { M } }$ imply $\dot { \xi } _ { \mathrm { N O } } \in L ^ { \infty }$ , so $\xi _ { \mathrm { N O } } \in W ^ { 1 , \infty } ( \mathscr { T } ; \mathbb { R } ^ { n d _ { \nu ^ { 2 } } } )$ . Hence (68) holds on $\tau _ { \mathrm { { \ell } } }$ , and $\| \xi _ { \mathrm { N O } } - \xi _ { \mathrm { I V } } \| _ { L ^ { \infty } } \leq \lambda _ { \mathrm { m i n } } ^ { - 1 } \varepsilon _ { \mathcal { G } }$

With $( \overline { { e } } ( t ) , \dot { \overline { { e } } } ( t ) , \xi _ { \mathrm { N O } } ( t ) ) \in \mathcal { K } _ { \mathrm { v e c } }$ for $\mathrm { a . e . } \ t \in \mathcal { T }$ , we apply (64). By the definition of $\mathcal { F } _ { \mathrm { { I V } } }$ in Theorem B.12,

$$
\begin{array} { r } { \begin{array} { l l } { \overline { { \sigma } } _ { \mathrm { I V } } ( t ) = \mathcal { F } _ { \mathrm { I V } } ( \overline { { e } } ( t ) , { \dot { \overline { { e } } } ( t ) } , \xi _ { \mathrm { I V } } ( t ) ; \mathbb { V } , \mathbb { E } ) } & { \mathrm { f o r ~ a . e . ~ } t \in \mathcal { T } . } \\ { \| \overline { { \sigma } } _ { \mathrm { N O } } ( t ) - \overline { { \sigma } } _ { \mathrm { I V } } ( t ) \| _ { F } \leq \varepsilon _ { \mathcal { F } } + \left\| \mathcal { F } _ { \mathrm { I V } } ( \overline { { e } } ( t ) , { \dot { \overline { { e } } } ( t ) } , \xi _ { \mathrm { N O } } ( t ) ; \mathbb { V } , \mathbb { E } ) - \mathcal { F } _ { \mathrm { I V } } ( \overline { { e } } ( t ) , { \dot { \overline { { e } } } ( t ) } , \xi _ { \mathrm { I V } } ( t ) ; \mathbb { V } , \mathbb { E } ) \right\| _ { F } . } \end{array} } \end{array}
$$

Since $\mathcal { F } _ { \mathrm { { I V } } }$ is afine in $\xi$ and sums the n tensorial blocks, Cauchy–Schwarz gives

$$
\begin{array} { r } { \| \mathcal { F } _ { \mathrm { I V } } ( \overline { { e } } ( t ) , \dot { \overline { { e } } } ( t ) , \xi _ { \mathrm { N O } } ( t ) ; \mathbb { V } , \mathbb { E } ) - \mathcal { F } _ { \mathrm { I V } } ( \overline { { e } } ( t ) , \dot { \overline { { e } } } ( t ) , \xi _ { \mathrm { I V } } ( t ) ; \mathbb { V } , \mathbb { E } ) \| _ { F } \leq \sqrt { n } \| \xi _ { \mathrm { N O } } ( t ) - \xi _ { \mathrm { I V } } ( t ) \| . } \end{array}
$$

Therefore, $\| \overline { { \sigma } } _ { \mathrm { N O } } ( t ) - \overline { { \sigma } } _ { \mathrm { I V } } ( t ) \| _ { F } \leq \varepsilon _ { \mathcal { F } } + \sqrt { n } \lambda _ { \operatorname* { m i n } } ^ { - 1 } \varepsilon _ { \mathcal { G } }$ for a.e. $t \in \tau$ . Taking suprema over $( \mathbb { V } , \mathbb { E } ) \in \mathcal { M }$ and ${ \overline { { e } } } \in { \mathcal { E } }$ gives (66). □

Completion of the Proof of Theorem 4.4. Let $\varepsilon > 0$ and choose $\varepsilon _ { \mathrm { I V } } \in ( 0 , \varepsilon )$ . By Theorem B.10, choose a Prony-series model with n terms whose uniform stress error is at most ε<sub>IV</sub>. Let $\eta _ { \xi } > 0$ and set

$$
\varepsilon _ { \mathcal { F } } : = \frac { \varepsilon - \varepsilon _ { \mathrm { I V } } } { 2 } , \qquad \varepsilon _ { \mathcal { G } } : = \operatorname* { m i n } \left\{ \frac { \lambda _ { \operatorname* { m i n } } \eta _ { \xi } } { 2 } , \frac { \lambda _ { \operatorname* { m i n } } ( \varepsilon - \varepsilon _ { \mathrm { I V } } ) } { 2 \sqrt { n } } \right\} .\tag{69}
$$

Choose the response mappings $\left( \mathcal { F } _ { \mathrm { N O } } , \mathcal { G } _ { \mathrm { N O } } \right)$ by Theorem B.12. These tolerances satisfy the hypothesis of Theorem B.13 and $\varepsilon _ { \mathcal { F } } + \sqrt { n } \varepsilon _ { \mathcal { G } } / \lambda _ { \operatorname* { m i n } } \leq \varepsilon - \varepsilon _ { \mathrm { I V } }$ , so the neural operator model approximates the Prony-series model with uniform stress error at most $\varepsilon - \varepsilon _ { \mathrm { { I V } } }$ . The triangle inequality gives (29) with $n _ { \xi } = n d _ { \nu ^ { 2 } }$ □

## Appendix C. Implementation Details for Viscoelasticity

This appendix provides additional information on the numerical results for learning viscoelasticity.

## C.1. Numerical Solutions of the Unit Cell Problem

The cell problem is discretized using a $1 2 8 \times 1 2 8$ uniform triangular mesh with $\mathbb { P } _ { 1 }$ finite elements and periodic boundary conditions. The homogenization solver uses a mixed formulation for the displacement corrector together with a Lagrange multiplier enforcing a zero-mean constraint. We use an explicit fourth-order Runge–Kutta scheme with 1024 time steps on [0, 1]. The displacement field at each time step is obtained via a direct solver. The solver is implemented in FEniCS [73, 74].

## C.2. Trajectory Sampling

For each independent component $a \in \{ 1 1 , 2 2 , 1 2 \}$ of the macroscopic strain, we first sample the total number of control times $N _ { c }$ uniformly from $\{ 3 , \ldots , 2 1 \}$ . Let $0 = t _ { 0 } < t _ { 1 } <$ $\cdot \cdot \cdot < t _ { N _ { c } - 1 } = 1$ be the control times, with the interior points sampled uniformly in [0, 1] and then sorted. The strain is initialized by $\overline { { e } } _ { a } ( t _ { 0 } ) = 0$ . For each $k = 1 , \ldots , N _ { c } - 1$ , we sample an independent sign $v _ { a , k } \in \{ - 1 , 1 \}$ and set $\begin{array} { r } { \overline { { e } } _ { a } ( t _ { k } ) = \overline { { e } } _ { a } ( t _ { k - 1 } ) + \left( e _ { \operatorname* { m a x } } v _ { a , k } - \overline { { e } } _ { a } ( t _ { k - 1 } ) \right) \sqrt { t _ { k } - t _ { k - 1 } } . } \end{array}$ with $e _ { \mathrm { m a x } } = 0 . 2$ . The values between control times are obtained via piecewise cubic Hermite interpolating polynomials (PCHIP). This procedure is applied independently to the three strain components.

## C.3. Architecture and Training

For both the stress and kinetic response mappings in (11), each Fourier neural mapping uses GELU activation, 4 retained Fourier modes, a width of 32, a projection width of 64, and three layers. The temporal grids are downsampled to 256 steps for the training. The baseline dataset contains 4096 training, 1024 validation, and 2048 test samples. We use the explicit Euler method to evolve the dynamics of the internal variable. Training uses the AdamW optimizer with a batch size of 32, an initial learning rate of 10−<sup>3</sup>, and 400 epochs. A cosine learning-rate scheduler decreases the learning rate to $1 0 ^ { - 5 }$ by the final epoch. We use a penalty weight of $\alpha = 1$ for $n _ { \xi } = 1$ and $\alpha = 0 . 1$ for $n _ { \xi } \in \{ 5 , 1 0 , 1 5 \}$ and $\alpha = 0 . 0 1$ for $n _ { \xi } \in \{ 2 0 , 2 5 \}$ . The architecture and training are implemented in PyTorch [75].

## C.4. Voronoi Dataset

The baseline microstructures are piecewise constant on periodic Voronoi tessellations. The number of seeds is sampled uniformly from $\{ 5 , \ldots , 1 5 \}$ , and the seed locations are sampled uniformly in the unit cell. Within each Voronoi region, the elastic and viscous Poisson ratios are sampled independently from [0.05, 0.45]. The elastic and viscous Young’s moduli are sampled independently from log-uniform distributions, with baseline ranges [0.5, 2.0] and [0.02, 0.2], respectively.

## C.5. Datasets with Varied Modulus Ranges

For each material contrast shown in the middle panel of Figure 5, we first generate a separate Voronoi dataset. To vary the elastic contrast range, we either set the elastic modulus to 1 or sample it from the ranges [0.71, 1.41], [0.35, 2.83], or [0.25, 4]. Note that even when the elastic modulus is constant, the elastic tensor remains heterogeneous because the Poisson ratio varies across each Voronoi region. To vary the viscous modulus range, we sample the viscous modulus from [0.04, 0.1], [0.028, 0.14], [0.014, 0.28], or $[ 0 . 0 1 , 0 . 4 ]$ . Since the same Poisson ratio range is used throughout and the plane-stress isotropic tensor has smallest and largest eigenvalues $E / ( 1 + \nu )$ and $E / ( 1 - \nu )$ , the material contrast in this numerical study is $\beta / \alpha = ( E _ { \mathrm { m a x } } / E _ { \mathrm { m i n } } ) ( 1 + \nu _ { \mathrm { m a x } } ) / ( 1 - \nu _ { \mathrm { m a x } } )$ , where $\nu _ { \mathrm { m a x } } = 0 . 4 5$

## C.6. Fitting a Linear Model

For each microstructure, we evaluate the stress and kinetic responses of the trained neural operator model on the sampled trajectories and fit them by afine functions of the strain, strain rate, and internal state. After subtracting the fitted ofsets, the linear part has the form

$$
\mathsf { v e c } _ { 2 } ( \overline { { \boldsymbol { \sigma } } } _ { \mathrm { N O } } ) = A _ { \dot { e } } \mathsf { v e c } _ { 2 } ( \dot { \overline { { e } } } ) + A _ { e } \mathsf { v e c } _ { 2 } ( \overline { { e } } ) + B \boldsymbol { \xi } , \qquad \dot { \boldsymbol { \xi } } = C \mathsf { v e c } _ { 2 } ( \dot { \overline { { e } } } ) + D \boldsymbol { \xi } ,
$$

where $\xi \in \mathbb { R } ^ { n _ { \xi } }$ is the internal variable of the trained neural operator model. The tensorial form in (35) follows from a change of coordinates. For $n _ { \xi } = n d _ { \nu ^ { 2 } }$ with $n \in \mathbb N$ , introduce tensors $q _ { \mathrm { N O } } ^ { ( i ) } \in \mathcal { V } ^ { 2 } , i = 1 , \dots , n .$ , and seek an invertible matrix S such that

$$
\xi = S \mathsf { v e c } _ { 2 } ^ { ( n ) } ( q _ { \mathrm { N O } } ^ { ( 1 ) } , \hdots , q _ { \mathrm { N O } } ^ { ( n ) } ) , \qquad B S = [ I \ \cdots \ I ] ,
$$

where each identity block has size $d _ { \nu ^ { 2 } } \times d _ { \nu ^ { 2 } }$ . If B has full row rank, such an S exists, and the internal-variable contribution to the stress is $\begin{array} { r } { \sum _ { i } q _ { \mathrm { N O } } ^ { ( i ) } } \end{array}$ . The matrices $A _ { \dot { e } }$ and $A _ { e }$ are the Mandel representations of $\mathbb { V } _ { \mathrm { N O } } ^ { \mathrm { h o m } }$ and $\mathbb { E } _ { \mathrm { N O } } ^ { \mathrm { h o m } }$ . Partitioning $S ^ { - 1 } D S$ and $S ^ { - 1 } C$ into $d _ { \nu ^ { 2 } } \times d _ { \nu ^ { 2 } }$ blocks gives the Mandel representations of $\mathbb { L } _ { \mathrm { N O } } ^ { ( i j ) }$ and $\mathbb { K } _ { \mathrm { N O } } ^ { ( i ) }$ , respectively. For $n _ { \xi } = 1 5$ in two dimensions, $d _ { \mathcal { V } ^ { 2 } } = 3$ and hence $n = 5$ . Since $S ^ { - 1 } D S$ is similar to $D ,$ the eigenvalues plotted in Figure 6 are invariants of the fitted linear dynamics.

## C.7. Particulate Dataset

The particulate dataset consists of two-phase periodic microstructures with circular inclusions. In each sample, the elastic and viscous modulus and Poisson-ratio fields share the same inclusion geometry. The number of inclusions is drawn uniformly from $\{ 1 , \ldots , 2 0 \}$ , and their sizes are chosen to give a particle area fraction in [0.03, 0.40] subject to non-overlap in the periodic cell. The material fields are constant on each phase. We sample elastic moduli log-uniformly from [0.5, 1.0] in the matrix and from [1.0, 2.0] in the inclusions. Viscous moduli use the reverse ordering on the logarithmic scale over [0.02, 0.2], so that the matrix phase is more viscous. We sample elastic and viscous Poisson ratios independently from [0.05, 0.45] in both phases.

## C.8. Median-Error Prediction Visualization

Figures C.1 and C.2 show test samples whose relative stress errors are closest to the median for the baseline Voronoi and particulate datasets, respectively, using the trained neural operator model with $n _ { \xi } = 2 5$ . These examples display the four scalar material fields, the imposed strain path, the predicted and reference stress components, and the predicted internal-variable trajectory. The memoryless response indicates the part of the response not captured without internal variables.

![](images/c3e78ebf00dc53c40ee3b8c49dee0832f1f5773e3880a092e7472bdd5596330c.jpg)

![](images/5ea80def2afd44d30a3ab117c865ec92c622e016b0742266b83cc85c08594858.jpg)

![](images/1770a17f0f58610be55b69253ad6dd5139eeaa923d762026847b5e99fa4a3f16.jpg)

## The median-error test sample: Voronoi dataset

![](images/61e747ca79fffb3b336d6bb279b1bea061aa2024d67f7e56905639ee1aae3516.jpg)  
Figure C.1: Prediction by the trained neural operator model for a viscoelasticity sample with median relative stress error, using internal-variable dimension $n _ { \xi } = 2 5$ . The left panels show the microstructural property fields, while the right panels show the prescribed strain trajectory, the predicted and reference stress trajectory, and the predicted internal-variable trajectory. The memoryless prediction $( \mathbb { K } = 0 )$ is included for comparison.

## The median-error test sample: Particulate dataset

![](images/7106375f6cd9b76a68e5795a560d2146e0f9271fc81ddb4d90099cd66d323626.jpg)

![](images/edcf9bbf0823570687912c7ffa9e88e922bc9129c0bf50b4e2a5abb8f229babf.jpg)

![](images/5eb7163617d9973ad2d3a105f5b442d47d9faf074ad73afe35f9571829d76283.jpg)

![](images/d88abb0b246b1f7e8beb06787ce3b7aa5df27d3935a5590f3214b78e7b50dea2.jpg)  
Figure C.2: Prediction by the trained neural operator model for a particulate viscoelasticity sample with median relative stress error, using internal-variable dimension $n _ { \xi } = 2 5$ . The left panels show the microstructural property fields, while the right panels show the prescribed strain trajectory, the predicted and reference stress trajectory, and the predicted internal-variable trajectory. The memoryless prediction $( \mathbb { K } = 0 )$ is included for comparison.

## Appendix D. Implementation Details for Crystal Plasticity

This appendix specifies the local constitutive equations, data-generation procedure, and training setup for the crystal-plasticity experiments in Section 5.2.

## D.1. Constitutive Equations

The cell problem (36) is closed by the following local constitutive equations. We use the multiplicative decomposition into elastic and plastic distortions, $F = F _ { e } F _ { p }$ , with elastic Green–Lagrange strain and stress

$$
\begin{array} { r } { E _ { e } = \frac { 1 } { 2 } ( { F _ { e } ^ { \top } F _ { e } } - I ) , \qquad S _ { e } = \mathbb { C } _ { \mathrm { c r y s } } E _ { e } , \qquad P = F _ { e } S _ { e } F _ { p } ^ { - \top } . } \end{array}
$$

Here $\mathbb { C } _ { \mathrm { c r y s } }$ is the cubic single-crystal stifness and $S _ { e }$ is the elastic second Piola–Kirchhof stress. The stifness and slip systems are expressed in crystal coordinates. Plastic flow on the twelve FCC 111 110 systems satisfies

$$
\dot { F } _ { p } = L _ { p } F _ { p } , \qquad L _ { p } = \sum _ { \alpha = 1 } ^ { 1 2 } \dot { \gamma } ^ { \alpha } s ^ { \alpha } \otimes m ^ { \alpha } , \qquad \dot { \gamma } ^ { \alpha } = \dot { \gamma } _ { 0 } \left| \frac { \tau ^ { \alpha } } { r ^ { \alpha } } \right| ^ { n } \mathrm { s g n } ( \tau ^ { \alpha } ) , \qquad \tau ^ { \alpha } = S _ { e } ( s ^ { \alpha } \otimes m ^ { \alpha } ) ,
$$

where ${ \dot { \gamma } } ^ { \alpha }$ is the shear rate on system α, with unit slip direction $s ^ { \alpha }$ , unit plane normal $m ^ { \alpha }$ and slip resistance $r ^ { \alpha }$ . The resolved shear stress follows DAMASK’s small-elastic-strain approximation $F _ { e } ^ { \top } F _ { e } \simeq I$ in the plastic driving stress [59, Appendix C]. The slip resistances evolve by self and latent hardening according to

$$
\dot { r } ^ { \alpha } = h _ { 0 } \mathrm { s g n } \biggl ( 1 - \frac { r ^ { \alpha } } { r _ { \infty } } \biggr ) \bigg | 1 - \frac { r ^ { \alpha } } { r _ { \infty } } \bigg | ^ { a } \sum _ { \beta = 1 } ^ { 1 2 } Q _ { \alpha \beta } | \dot { \gamma } ^ { \beta } | .
$$

Here $h _ { 0 } ( y )$ is the hardening scale, $r _ { \infty } ( y )$ is the saturation resistance, and $Q _ { \alpha \beta }$ is the FCC slip-interaction matrix. The initial orientation $O _ { 0 } ( y )$ maps specimen coordinates to crystal coordinates and is incorporated into the initial plastic distortion. With initial slip resistance $r _ { 0 } ( y )$ , the initially stress-free state is specified by

$$
\chi ( y , 0 ) = 0 , \qquad F _ { p } ( y , 0 ) = O _ { 0 } ( y ) , \qquad r ^ { \alpha } ( y , 0 ) = r _ { 0 } ( y ) .
$$

Since $\overline { { F } } ( 0 ) = I$ , these conditions give $F _ { e } ( y , 0 ) = O _ { 0 } ( y ) ^ { \top }$ and $S _ { e } ( y , 0 ) = 0$

The elastic constants are $( C _ { 1 1 } , C _ { 1 2 } , C _ { 4 4 } ) = ( 1 6 8 . 9 , 1 2 1 . 8 , 7 5 . 8 ) \mathrm { G P a }$ , the reference shear rate is $\dot { \gamma } _ { 0 } = 3 \times 1 0 ^ { - 3 } \mathrm { s ^ { - 1 } }$ , the rate exponent is $n = 2 0$ , and the saturation exponent is $a \ = \ 0 . 6$ The seven coeficients defining $Q$ are $( 1 , 1 , 5 . 1 2 3 , 0 . 5 7 4 , 1 . 1 2 3 , 1 . 1 2 3 , 1 )$ for self, coplanar, collinear, Hirth, glissile I, glissile II, and Lomer interactions, respectively. The grainwise fields $r _ { 0 } , r _ { \infty }$ , and $h _ { 0 }$ are sampled as follows.

## D.2. Microstructure Sampling

Let $\{ G _ { j } \} _ { j = 1 } ^ { N _ { g } }$ denote the grains of the periodic Voronoi tessellation. The microstructural input is piecewise constant on grains, with $M ( y ) = ( q _ { j } , r _ { 0 , j } , r _ { \infty , j } , h _ { 0 , j } ) \in \mathbb { R } ^ { 7 }$ for $y \in G _ { j }$ . Here $q _ { j } \in \mathbb { R } ^ { 4 }$ is the orientation quaternion representing $O _ { 0 }$ on $G _ { j }$ . For the single- and two-mode textures, the spread angle is sampled over the range $5 ^ { \circ }$ to $3 5 ^ { \circ }$

We sample the three slip-law parameters around the copper reference values using correlated grainwise scale factors. For each sample, we draw $\delta \sim \mathrm { U n i f } [ 0 . 0 5 , 0 . 2 0 ]$ , a resistance scale $\bar { s } _ { r }$ log-uniformly from [0.80, 1.35], and a hardening scale $\bar { s } _ { h }$ log-uniformly from [0.80, 1.50]. For each grain $G _ { j }$ , we draw a standard Gaussian pair $( Z _ { r , j } , Z _ { h , j } )$ with correlation 0.65 and set

$$
s _ { r , j } = \bar { s } _ { r } \exp ( \delta Z _ { r , j } ) , \qquad s _ { h , j } = \bar { s } _ { h } \exp ( \delta Z _ { h , j } ) .
$$

We set

$$
r _ { 0 , j } = 3 1 \mathrm { M P a } s _ { r , j } , \qquad r _ { \infty , j } = 9 6 . 4 \mathrm { M P a } s _ { r , j } , \qquad h _ { 0 , j } = 0 . 3 5 \mathrm { G P a } s _ { h , j } .
$$

Thus $r _ { 0 , j }$ and $r _ { \infty , j }$ vary together with a constant ratio, while $h _ { 0 , j }$ varies through a distinct but positively correlated factor.

## D.3. Trajectory Sampling

We parameterize each macroscopic loading path by separating the rotation, volumetric stretch, and deviatoric stretch components. Specifically, we write

$$
\begin{array} { r } { \overline { { F } } ( t ) = R ( t ) \exp ( E ( t ) ) , \quad R ( t ) = \exp ( \mathrm { s k e w } ( \omega ( t ) ) ) , } \end{array}
$$

where $E ( t )$ is the logarithmic strain, $\boldsymbol { \omega } ( t ) \in \mathbb { R } ^ { 3 }$ is the rotation vector, and skew $\mathbf { \Pi } ( \omega ( t ) )$ is the associated skew symmetric matrix, i.e., skew $( w ) v = w \times v$ for $v , w \in \mathbb { R } ^ { 3 }$ . The logarithmic strain is decomposed as

$$
E ( t ) = \frac { \eta _ { \mathrm { v o l } } ( t ) } { 3 } I + E _ { \mathrm { d e v } } ( t ) , \quad \eta _ { \mathrm { v o l } } ( t ) = \log \operatorname* { d e t } \overline { { F } } ( t ) , \quad \mathrm { t r } E _ { \mathrm { d e v } } ( t ) = 0 .
$$

After choosing a basis $\{ B _ { a } \} _ { a = 1 } ^ { 5 }$ for trace-free symmetric $3 \times 3$ tensors, we have $E _ { \mathrm { d e v } } ( t ) =$ $\textstyle \sum _ { a = 1 } ^ { 5 } e _ { a } ( t ) B _ { a }$ Thus, each loading path is sampled through the nine scalar coordinates $( \eta _ { \mathrm { v o l } } ( t ) , e _ { 1 } ( t ) , \ldots , e _ { 5 } ( t ) , \omega _ { 1 } ( t ) , \omega _ { 2 } ( t ) , \omega _ { 3 } ( t ) )$ independently according to the procedure in $\mathrm { A p \mathrm { - } }$ pendix C.2. The initial scalar values are set so that $\overline { { F } } ( 0 ) = I$ . Once the scalar paths have been sampled and interpolated, the deformation gradient is reconstructed on the common output time grid. This reconstruction enforces det $\overline { { F } } ( t ) = \exp ( \eta _ { \mathrm { v o l } } ( t ) ) > 0$

## D.4. Architecture and Training

The architecture and training are similar to Appendix C.3, except that we use 8 Fourier modes, a width of 48, and a projection width of 96. We use a penalty weight of $\alpha = 0 . 0 0 1$ across all internal-variable dimensions. We train on trajectory data with a larger step size $( 1 / 6 4 )$ rather than the data-generation step size (1/128), and only on [0, 0.875] rather than the full unit time interval. This reduction lets us make the most of our computing resources; we use 4 Nvidia Tesla P100 GPUs for training. The testing is performed on the full trajectory data.

## D.5. Median-Error Prediction Visualization

Figure D.1 shows the median-error test sample for the crystal-plasticity dataset, including the sampled microstructure, loading path, stress response, and learned internal variables.

The median error test sample: Crystal plasticity dataset  
![](images/68c58cad824e3b1e345e3fd3503f834e5dce6758505d2baf7e9d1373b8f9b2a2.jpg)  
Figure D.1: (Top) The test sample of the crystal plasticity dataset with a median relative stress prediction error (7.5%) for the learned model. (Bottom left) The stress-prediction components, where solid lines denote the reference homogenized response and dashed-dotted lines denote the stress response predicted by the learned model. (Bottom right) The components of the internal variable predicted by the learned model.