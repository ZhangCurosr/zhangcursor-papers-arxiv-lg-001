# Asymptotic-Preserving Tensor Neural Networks for 3D High-Field Kinetic Equations with Applications to Uncertainty Quantification

Shi Jin<sup>1,2,3</sup> and Tian-ai Zhang<sup>∗1</sup>

<sup>1</sup>School of Mathematical Sciences, Shanghai Jiao Tong University, Shanghai, China <sup>2</sup>Institute of Natural Sciences, MOE-LSC, Shanghai Jiao Tong University, Shanghai, China <sup>3</sup>Shanghai Artificial Intelligence Laboratory, Shanghai, China

## Abstract

Kinetic equations in the high-field regime describe particle transport dominated by electric-field acceleration and collisional relaxation. These equations are widely used in simulations of plasmas and semiconductor devices. Typical models include the Vlasov– Poisson–Fokker–Planck (VPFP) and Boltzmann–Poisson systems. Their high dimensionality and multiscale nature pose two of the major challenges to accurate and eficient numerical solution. To address these challenges, we propose an asymptotic-preserving tensor neural network (APTNN) method. Specifically, to reduce high-dimensional computational costs and memory requirements, we adopt the canonical polyadic (CP) decomposition structure of TNNs to approximate their solutions. We show that the equation residuals retain this decomposition form, allowing high-dimensional loss integrals to be evaluated through combinations of onedimensional integrals. The resulting computational and storage costs are reduced from exponential to polynomial growth with dimension. To improve the prediction accuracy in the high-field limit, we enhance the baseline AP loss by incorporating the first moment equation. The added loss term directly constrains the limiting flux closure to reduce its approximation error in the high-field regime. The enhanced AP loss can still be evaluated dimension-wise. Numerical results show that APTNN improves prediction accuracy in the high-field regime with significantly reduced computational costs and memory usage. For a 3D3V VPFP problem with multiple random inputs, training takes less than 30 minutes on a single GPU.

Keywords: High-field kinetic equations; Tensor neural networks; Asymptotic-preserving methods; Uncertainty quantification

## 1 Introduction

Kinetic equations provide a statistical description of motion and interactions in a collection of particles [12]. They are fundamental to modeling plasma dynamics and semiconductor transport [10, 20]. Typical models include the Vlasov–Poisson–Fokker–Planck (VPFP) system for plasmas [24, 17] and the Boltzmann–Poisson system for semiconductors [25], both of which incorporate a self-consistent electric field. Of great physical interest is the high-field regime, in which electricfield acceleration and collisional relaxation jointly dominate particle transport [31, 3, 9]. This regime is important for modeling the Brownian motion of charged particles in a thermal bath and hot-electron transport in miniaturized semiconductor devices [24, 25].

Numerically solving kinetic equations in the high-field regime presents two major challenges: high dimensionality and multiple temporal and spatial scales. First, the distribution function is defined over phase space and time, involving seven variables in three-dimensional problems. For full tensor-product discretizations, the number of grid points grows exponentially with dimension, leading to the curse of dimensionality [27]. Moreover, uncertainty in initial data, boundary conditions or model parameters introduces additional random variables, which further increase the computational cost of uncertainty quantification (UQ) [23]. Second, the multiscale nature of these equations presents another challenge. As the characteristic scales become small, numerically resolving them can be prohibitively expensive. Therefore, an efective numerical scheme should reduce high-dimensional computational costs and capture the correct macroscopic limit without resolving the small scales.

In recent years, neural network (NN) methods have emerged as promising approaches to solving high-dimensional partial diferential equations (PDEs) [19, 13, 38]. These methods leverage the powerful approximation capabilities of NNs to predict PDE solutions. For example, physicsinformed neural networks (PINNs) [33] compute derivatives using automatic diferentiation (AD) and construct a loss from mean squared PDE residuals. Residuals of the initial and boundary conditions (IBCs) are also included. To improve computational eficiency in high dimensions, tensor neural networks (TNNs) employ a low-rank decomposition idea to modify NN architectures [37]. They feed each coordinate into a separate subnetwork and combine the outputs as sums of products to represent multivariate functions. This separable structure is also adopted in separable physics-informed neural networks (SPINNs) to reduce repeated network computations in function and derivative evaluation [7]. Recent studies have applied this approach to kinetic equations to reduce computational costs [32, 6]. However, the high dimensionality of kinetic equations makes loss evaluation on full tensor-product grids computationally expensive and memory intensive. In particular, GPU memory requirements grow exponentially with dimension and become a primary constraint on problem size. This bottleneck becomes more severe in UQ problems with additional random inputs.

Besides addressing the challenges of high dimensionality, NNs also need to accurately capture the multiscale behavior of kinetic equations. The asymptotic-preserving (AP) idea has been incorporated into neural solvers to preserve the correct macroscopic limit, leading to asymptoticpreserving neural networks (APNNs) [21, 29, 2]. For the high-field regime, APNN can be constructed by combining the macroscopic mass conservation law with the kinetic equations in the loss [22, 40]. However, the correct limiting behavior alone does not guarantee prediction accuracy in the limit. The baseline AP loss lacks a direct constraint on the limiting flux closure for the continuity equation. A misfit in this closure may reduce the predicted accuracy for the macroscopic evolution.

In this paper, we aim to develop an eficient and accurate NN method for high-dimensional kinetic equations in the high-field regime. First, we adopt TNNs approximation in variableseparation form. For their training loss, we show that a dimension-wise integration strategy can be introduced to express the high-dimensional losses as combinations of one-dimensional integrals. This avoids the full-grid loss evaluation used in previous TNN-based kinetic solvers [32, 6], reducing computational costs and memory usage from exponential to polynomial growth with dimension. Second, we enhance the AP loss by incorporating the first velocity-moment equation in the loss term. The added term directly constrains the flux closure and improves predictions of the macroscopic evolution in the high-field limit. The enhanced loss remains separable and allows eficient dimension-wise evaluation. As a result, we developed an asymptotic-preserving tensor neural network (APTNN) method. Numerical results show that APTNN improves prediction accuracy in the high-field limit while significantly reducing computational costs and memory usage. A multidimensional UQ example for a 3D3V problem requires less than 30 minutes of training on a single GPU.

The remainder of this paper is organized as follows. Sect. 2 introduces the kinetic equations in the high-field regime and their high-field limits. Sect. 3 presents the APTNN method, starting with the separable TNN structure and its approximation properties. It then constructs an AP loss with an enhanced constraint on the limiting flux closure and derives a dimension-wise formula for loss evaluation. Numerical results in Sect. 4 demonstrate the accuracy and eficiency of the proposed method across diferent dimensions and scale parameters. Multi-dimensional UQ examples are also included. Sect. 5 concludes the paper and discusses future research directions.

## 2 Model equations

This section introduces kinetic models in the high-field regime, where both electric force and collisions dominate the transport of charged particles. Typical applications include the simulation of plasmas and semiconductor devices. Let $f = f ( t , \mathbf { x } , \mathbf { v } ) \geq 0$ denote the distribution function of electrons at time $t \in [ 0 , T ]$ in the phase space $\Omega _ { \mathbf { x } } \times \mathbb { R } ^ { d }$ , where $\mathbf { x } \in \Omega _ { \mathbf { x } } \subset \mathbb { R } ^ { d } , \mathbf { v } \in \mathbb { R } ^ { d }$ and d $\in \{ 1 , 2 , 3 \}$ . Its time evolution under a self-consistent electrostatic potential $\phi$ is governed by

$$
\left\{ \begin{array} { l } { \displaystyle \partial _ { t } f + \mathbf { v } \cdot \nabla _ { \mathbf { x } } f + \frac { 1 } { \varepsilon } \mathbf { E } \cdot \nabla _ { \mathbf { v } } f = \frac { 1 } { \varepsilon } \mathcal { Q } ( f ) , } \\ { \displaystyle - \Delta _ { \mathbf { x } } \phi = \rho - h , \qquad \mathbf { E } = - \nabla _ { \mathbf { x } } \phi . } \end{array} \right.\tag{2.1}
$$

Here, E is the electric field and $\mathcal { Q }$ is a collision operator satisfying mass conservation

$$
\int _ {  { \mathbb { R } } ^ { d } } \mathscr { Q } ( f ) \mathrm { d }  { \mathbf { v } } = 0 .
$$

In the second Poisson equation,

$$
\rho ( t , \mathbf { x } ) = \int _ { \mathbb { R } ^ { d } } f ( t , \mathbf { x } , \mathbf { v } ) \mathrm { d } \mathbf { v }
$$

is the electron density and $h = h ( \mathbf { x } )$ is the given background charge density. The system satisfies the global charge-neutrality condition

$$
\int _ { \Omega _ { \mathbf { x } } } \int _ { \mathbb { R } ^ { d } } f ^ { 0 } ( \mathbf { x } , \mathbf { v } ) \mathrm { d } \mathbf { v } \mathrm { d } \mathbf { x } = \int _ { \Omega _ { \mathbf { x } } } h ( \mathbf { x } ) \mathrm { d } \mathbf { x } ,\tag{2.2}
$$

where $f ^ { 0 } = f ( 0 , { \bf x } , { \bf v } )$ is the initial distribution and h is set to 1 in this work. The scale parameter $\varepsilon > 0$ denotes the ratio of the electron mean free path to the characteristic length of the system. When $\varepsilon = O ( 1 )$ , the system is in the kinetic regime, whereas $\varepsilon \ll 1$ corresponds to the high-field regime. The asymptotic limit $\varepsilon \to 0$ is referred to as the high-field limit.

We consider two collision operators, namely the Fokker–Planck operator for the VPFP system and the Boltzmann operator for the semiconductor Boltzmann equation. For the VPFP system in electrostatic plasmas, the Fokker–Planck operator models Coulomb interactions between electrons and a surrounding thermal bath [24] and is given by

$$
\mathcal { Q } _ { \mathrm { F P } } ( f ) = \nabla _ { \mathbf { v } } \cdot ( \mathbf { v } f + \nabla _ { \mathbf { v } } f ) .\tag{2.3}
$$

Taking $\mathcal { Q } = \mathcal { Q } _ { \mathrm { F P } }$ in (2.1) and combining it with the electric force term, the Vlasov equation can be rewritten as

$$
\partial _ { t } f + \mathbf { v } \cdot \nabla _ { \mathbf { x } } f = { \frac { 1 } { \varepsilon } } { \mathcal { L } } _ { \mathbf { E } } f , \quad { \mathrm { w h e r e ~ } } { \mathcal { L } } _ { \mathbf { E } } f : = \nabla _ { \mathbf { v } } \cdot \left[ ( \mathbf { v } - \mathbf { E } ) f + \nabla _ { \mathbf { v } } f \right] .\tag{2.4}
$$

The combined operator $\mathcal { L } _ { \mathbf { E } }$ remains of Fokker–Planck type. Its null space is

$$
\mathcal { N } ( \mathcal { L } _ { \mathbf { E } } ) = \mathrm { S p a n } \{ M _ { \mathbf { E } } \} = \{ f = \rho M _ { \mathbf { E } } \} ,
$$

where

$$
M _ { \mathbf { E } } ( \mathbf { v } ) = ( 2 \pi ) ^ { - d / 2 } \exp ( - | \mathbf { v } - \mathbf { E } | ^ { 2 } / 2 )
$$

is the shifted Maxwellian that depends on $( t , \mathbf { x } )$ through E.

For the semiconductor Boltzmann equation, one considers the low density regime, in which the collision operator takes the form

$$
\mathcal { Q } _ { \mathrm { s e m i B } } ( f ) ( \mathbf { v } ) = \int _ { \mathbb { R } ^ { d } } \Psi ( \mathbf { v } , \mathbf { v } ^ { \prime } ) \left[ M ( \mathbf { v } ) f ( \mathbf { v } ^ { \prime } ) - M ( \mathbf { v } ^ { \prime } ) f ( \mathbf { v } ) \right] \mathrm { d } \mathbf { v } ^ { \prime } ,\tag{2.5}
$$

which is called the non-degenerate case [25]. Here, $\Psi ( \mathbf { v } , \mathbf { v } ^ { \prime } ) = \Psi ( \mathbf { v } ^ { \prime } , \mathbf { v } )$ is the symmetric scattering kernel and

$$
M ( \mathbf { v } ) = ( 2 \pi ) ^ { - d / 2 } \exp ( - | \mathbf { v } | ^ { 2 } / 2 )
$$

is the standard Maxwellian. This operator characterizes the interactions of electrons with the semiconductor crystal lattice and with one another. In contrast to the VPFP system, the high-field equilibrium of the semiconductor Boltzmann equation is not always available in explicit form. In this work, we consider the simplest time-relaxation approximation $\Psi \equiv 1$ , for which

$$
\mathcal { Q } _ { \mathrm { B G K } } ( f ) = \rho M - f .\tag{2.6}
$$

With this collision, (2.1) defines the BGK–Poisson system.

We next show the common high-field limit of the VPFP and BGK–Poisson systems. As $\varepsilon  0 ,$ f approaches a local high-field equilibrium. For the VPFP system, $\mathcal { N } ( \mathcal { L } _ { \bf E } ) = \mathrm { S p a n } \{ M _ { \bf E } \}$ implies $f  \rho M _ { \bf E }$ , whereas for the BGK–Poisson system, $f  \rho F _ { \bf E }$ , where F<sub>E</sub> satisfies

$$
\int _ { \mathbb { R } ^ { d } } F _ { \mathbf { E } } \mathrm { d } \mathbf { v } = 1 , \quad \mathbf { E } \cdot \nabla _ { \mathbf { v } } F _ { \mathbf { E } } = M - F _ { \mathbf { E } } , \quad F _ { \mathbf { E } } \geq 0 .\tag{2.7}
$$

Since both $\mathcal { Q } _ { \mathrm { F P } }$ and $\mathcal { Q } _ { \mathrm { B G K } }$ conserve mass, integrating (2.1) over velocity space yields the continuity equation

$$
\begin{array} { r } { \partial _ { t } \rho + \nabla _ { \mathbf x } \cdot \mathbf j = 0 , } \end{array}\tag{2.8}
$$

where $\begin{array} { r } { \mathbf { j } = \int _ { \mathbb { R } ^ { d } } \mathbf { v } f } \end{array}$ dv is the flux. Because both collision operators also satisfy the same first velocity-moment identity

$$
\int _ { \mathbb { R } ^ { d } } \mathbf { v } \mathcal { Q } ( f ) \mathrm { d } \mathbf { v } = - \mathbf { j } ,\tag{2.9}
$$

multiplying (2.1) by v and integrating over $\mathbb { R } ^ { d }$ gives

$$
\varepsilon \left( \partial _ { t } \mathbf { j } + \nabla _ { \mathbf { x } } \cdot \boldsymbol { \Pi } \right) + \mathbf { j } - \rho \mathbf { E } = 0 ,\tag{2.10}
$$

where $\begin{array} { r } { \pmb { \Pi } = \int _ { \mathbb { R } ^ { d } }  { \mathbf { v } } \otimes  { \mathbf { v } } f } \end{array}$ dv is the momentum flux tensor. Letting $\varepsilon \to 0$ in this equation yields $\mathbf { j } = \rho \mathbf { E }$ . Substituting this relation into (2.8), together with the Poisson equation, one obtains the high-field limit system

$$
\left\{ \begin{array} { l l } { \partial _ { t } \rho + \nabla _ { \mathbf { x } } \cdot ( \rho \mathbf { E } ) = 0 , } \\ { \qquad } \\ { - \Delta _ { \mathbf { x } } \phi = \rho - h , \qquad \mathbf { E } = - \nabla _ { \mathbf { x } } \phi . } \end{array} \right.\tag{2.11}
$$

This formal analysis can be made rigorous. For the VPFP system, the high-field limit was first established in one dimension [31] and later extended to multiple dimensions [16]. This result for the BGK–Poisson system was established in [3]. For more general Boltzmann operators, this topic remains an active area of research.

## 3 Methodology

This section presents the construction of the APTNN method. We first introduce the TNN architecture and establish an approximation result on the unbounded velocity domain. Then an enhanced AP loss function is formulated to reduce the approximation error in the limiting flux closure. Finally, we derive a dimension-wise formula for loss evaluation and analyze its computational cost and storage requirements.

## 3.1 The tensor neural network architecture

Standard PINNs can incur substantially higher computational costs as the input dimension D grows. This mainly stems from using coordinate-based network architectures that map input coordinates to the corresponding solution values. For (2.1), the main high-dimensional solution is the distribution $f ( t , \mathbf { x } , \mathbf { v } )$ with $D _ { f } = 1 + d _ { x } + d _ { v }$ variables, where $d _ { x }$ and $d _ { v }$ denote the spatial and velocity dimensions respectively. Introducing $d _ { z }$ random variables in UQ problems further increases the input dimension to $D _ { f } = 1 + d _ { x } + d _ { v } + d _ { z }$ . In the PINN framework, $f$ is approximated by a network $f _ { \theta }$ , where $\theta$ collects the trainable parameters. If N collocation points are used in each coordinate, $f _ { \theta }$ and its derivatives are evaluated at $N ^ { D _ { f } }$ tensor-product points.

A well-established approach to tackling high-dimensional challenges is canonical polyadic (CP) tensor decomposition [26, 1]. Its key idea is to represent a high-dimensional tensor as a finite sum of rank-one tensor products. Such separated representations are also suitable for approximating solutions of kinetic equations, since these solutions often evolve near a low-dimensional manifold and exhibit a low-rank structure [11, 39]. Thus we can approximate $f ( t , \mathbf { x } , \mathbf { v } )$ in CP decomposition form

$$
f ( t , \mathbf { x } , \mathbf { v } ) \approx \sum _ { r = 1 } ^ { r _ { f } } c _ { r } ^ { f } \xi _ { r } ^ { t } ( t ) \prod _ { i = 1 } ^ { d _ { x } } \xi _ { r } ^ { x _ { i } } ( x _ { i } ) \prod _ { j = 1 } ^ { d _ { v } } \xi _ { r } ^ { v _ { j } } ( v _ { j } ) ,\tag{3.1}
$$

where $r _ { f }$ is the separation rank and $\{ c _ { r } ^ { f } \} _ { r = 1 } ^ { r _ { f } }$ are the weights. For UQ problems, each rank-one term additionally contains $\Pi _ { m = 1 } ^ { d _ { z } } \xi _ { r } ^ { z _ { m } } ( z _ { m } )$

We use NNs to parameterize the CP representation of $f$ in (3.1), which is referred to as a tensor neural network (TNN) [37, 7]. The TNN comprises $D _ { f }$ independent feedforward subnetworks, each of which takes a scalar coordinate as input. Let s denote any coordinate, namely, $s = t$ $s = x _ { i }$ for $i = 1 , \ldots , d _ { x } .$ , or $s = v _ { j }$ for $j = 1 , \ldots , d _ { v }$ . The corresponding subnetwork maps s to an $r _ { f } .$ -dimensional output vector

$$
\pmb { \xi } ^ { s } ( s ; \theta _ { s } ) = \big ( \xi _ { 1 } ^ { s } ( s ; \theta _ { s } ) , \ldots , \xi _ { r _ { f } } ^ { s } ( s ; \theta _ { s } ) \big ) ^ { \top } .\tag{3.2}
$$

Each subnetwork is a feedforward neural network (FNN) with L hidden layers

$$
\pmb { \xi } ^ { s } ( s ; \theta _ { s } ) = \left( \pmb { F } _ { s } ^ { ( L + 1 ) } \circ \pmb { F } _ { s } ^ { ( L ) } \circ \cdots \circ \pmb { F } _ { s } ^ { ( 1 ) } \right) ( s ) .\tag{3.3}
$$

For $l = 1 , \ldots , L$ , the hidden layers are defined by $\mathbf { \nabla } _ { F _ { s } ^ { ( l ) } ( \mathbf { h } ) } = \sigma ( \mathbf { W } _ { s } ^ { ( l ) } \mathbf { h } + \mathbf { b } _ { s } ^ { ( l ) } )$ , where $\sigma$ is the activation function. The output layer ${ \pmb F } _ { s } ^ { ( L + 1 ) }$ is afine. The weight matrices and bias satisfy $\mathbf { W } _ { s } ^ { ( l ) } \in \mathbb { R } ^ { N _ { l } \times N _ { l - 1 } }$ and $\mathbf { b } _ { s } ^ { ( l ) } \in \mathbb { R } ^ { N _ { l } }$ , where $\left[ N _ { 0 } , N _ { 1 } , \dots , N _ { L } , N _ { L + 1 } \right]$ denotes the layer widths with $N _ { 0 } = 1$ and $N _ { L + 1 } = r _ { f }$ . The parameters of each subnetwork are collected in $\theta _ { s } = \{ \mathbf { W } _ { s } ^ { ( l ) } , \mathbf { b } _ { s } ^ { ( l ) } \} _ { l = 1 } ^ { L + 1 }$ The coeficients $\{ c _ { r } ^ { f } \} _ { r = 1 } ^ { r _ { f } }$ in (3.1) are also trainable parameters. The final output is obtained by multiplying the subnetwork outputs componentwise and taking a weighted sum

$$
f _ { \theta } ( t , \mathbf { x } , \mathbf { v } ) = \sum _ { r = 1 } ^ { r _ { f } } c _ { r } ^ { f } \xi _ { r } ^ { t } ( t ; \theta _ { t } ) \prod _ { i = 1 } ^ { d _ { x } } \xi _ { r } ^ { x _ { i } } ( x _ { i } ; \theta _ { x _ { i } } ) \prod _ { j = 1 } ^ { d _ { v } } \xi _ { r } ^ { v _ { j } } ( v _ { j } ; \theta _ { v _ { j } } ) ,\tag{3.4}
$$

where $\theta$ collects all subnetwork parameters and the trainable coeficients $\{ c _ { r } ^ { f } \} _ { r = 1 } ^ { r _ { f } }$ . Figure 1 shows the TNN architecture for $f _ { \theta }$ . The potential $\phi$ is also approximated by an independent TNN

$$
\phi _ { \theta } ( t , \mathbf { x } ) = \sum _ { r = 1 } ^ { r _ { \phi } } c _ { r } ^ { \phi } \eta _ { r } ^ { t } ( t ) \prod _ { i = 1 } ^ { d _ { x } } \eta _ { r } ^ { x _ { i } } ( x _ { i } ) ,\tag{3.5}
$$

![](images/aef801cf729afc974b82b32205465fd15d421ffedccd34d1464531a94ca821e9.jpg)  
Figure 1: Tensor neural network architecture for $f _ { \theta }$ . Each scalar coordinate is assigned an independent subnetwork. Only one spatial and one velocity subnetwork are shown. Here, σ denotes the activation function, and $\xi _ { r , \theta _ { i } } ^ { s }$ denotes the rth output component $\xi _ { r } ^ { s } ( s ; \theta _ { s } )$ of the subnetwork for coordinate s. To obtain the final output, subnetwork outputs with the same rank index are multiplied and combined using the trainable weights $c _ { r } ^ { f }$

where $r _ { \phi }$ is the separation rank and $\{ c _ { r } ^ { \phi } \} _ { r = 1 } ^ { r _ { \phi } }$ are trainable coeficients.

We establish a TNN approximation theorem in Sobolev spaces for the distribution f on the whole velocity space $\mathbb { R } ^ { 3 }$ . We first recall the approximation result in $H ^ { m } ( \Omega _ { 1 } \times \cdots \times \Omega _ { d } )$ on bounded domains [37, Theorem 1]:

Theorem 3.1. Let $\begin{array} { r } { \Omega = \prod _ { i = 1 } ^ { d } \Omega _ { i } } \end{array}$ be a Cartesian-product domain, where each $\Omega _ { i } \in \mathbb { R }$ is a bounded open interval. Consider a function $q \in H ^ { m } ( \Omega )$ . Assume that each subnetwork in the TNN is an FNN with one hidden layer and a non-polynomial activation function $\sigma \in C ^ { m } ( \mathbb { R } )$ . Then, for every $\delta > 0$ , there exist a rank $p \in \mathbb N$ and a TNN $\Psi _ { \theta }$ such that

$$
\| q - \Psi _ { \theta } \| _ { H ^ { m } ( \Omega ) } < \delta .\tag{3.6}
$$

We note that this theorem does not quantify the relation between separation rank $p$ and TNN the approximation error. Even for CP approximations, establishing a general rank–error relation remains an open research problem [37].

This theorem does not directly apply to the distribution $f ( t , x , v )$ , since the velocity is defined over the unbounded domain $\mathbb { R } ^ { 3 }$ . We therefore extend this approximation result to the domain $\mathcal { D } = [ 0 , T ] \times \Omega \times \mathbb { R } ^ { 3 }$ with $\begin{array} { r } { \Omega = \prod _ { i = 1 } ^ { d } \Omega _ { i } } \end{array}$ and $d \in \{ 1 , 2 , 3 \}$ . Let $\langle v \rangle = ( 1 + | v | ^ { 2 } ) ^ { 1 / 2 }$ . For $s \geq 0$ and

m $\in  { \mathbb { N } } _ { 0 }$ , we define a weighted Sobolev norm

$$
\left\| f \right\| _ { H _ { s } ^ { m } ( { \mathcal { D } } ) } ^ { 2 } : = \sum _ { \alpha + | \beta | + | \gamma | \leq m } \left\| \langle v \rangle ^ { s } \partial _ { t } ^ { \alpha } \partial _ { x } ^ { \beta } \partial _ { v } ^ { \gamma } f \right\| _ { L ^ { 2 } ( { \mathcal { D } } ) } ^ { 2 } ,\tag{3.7}
$$

where $\alpha \in  { \mathbb { N } } _ { 0 }$ is the time derivative order, and $\beta \in  { \mathbb { N } } _ { 0 } ^ { d }$ and $\gamma \in  { \mathbb { N } } _ { 0 } ^ { 3 }$ are multi-indices. We establish the TNN approximation theorem in this weighted Sobolev space:

Theorem 3.2. Let $f \in H _ { s } ^ { m } ( \mathcal { D } )$ be a particle density function. Then, for every $\delta > 0$ , there exists a rank $p \in \mathbb N$ and a TNN $f _ { \theta }$ such that

$$
\| f - f _ { \theta } \| _ { H _ { s } ^ { m } ( \mathcal { D } ) } < \delta .\tag{3.8}
$$

The proof can be found in Appendix A. For $s > 7 / 2$ , the TNN $f _ { \theta }$ can also approximate the velocity moments of $f$ up to second order, which will be needed to construct the loss. Specifically, the Cauchy–Schwarz inequality gives

$$
\left\| \int _ { \mathbb R ^ { 3 } } \varphi ( v ) ( f - f _ { \theta } ) d v \right\| _ { L ^ { 2 } ( ( 0 , T ) \times \Omega ) } \leq C _ { s } \| f - f _ { \theta } \| _ { H _ { s } ^ { m } ( \mathcal D ) } < C _ { s } \delta ,\tag{3.9}
$$

where $\varphi ( v ) \in \{ 1 , v _ { i } , v _ { i } v _ { j } , | v | ^ { 2 } \}$ with $i , j \in \{ 1 , 2 , 3 \}$ , and $\begin{array} { r } { C _ { s } = \left( \int _ { \mathbb { R } ^ { 3 } } \langle v \rangle ^ { 4 - 2 s } d v \right) ^ { 1 / 2 } } \end{array}$ is finite when $s > 7 / 2$

In practice, each velocity factor in TNN $f _ { \theta }$ is multiplied by a Gaussian envelope to enforce the tail decay of $f _ { \theta }$ in velocity [32]

$$
\widetilde { \xi } _ { r } ^ { v _ { i } } ( v _ { i } ) = \xi _ { r } ^ { v _ { i } } ( v _ { i } ) \exp \left[ - \frac { \tau _ { i } ^ { 2 } ( v _ { i } - \mu _ { i } ) ^ { 2 } } { 2 } \right] , \quad i = 1 , \dots , d _ { v } , \quad r = 1 , \dots , r _ { f } .\tag{3.10}
$$

Here, $\mu _ { i }$ and $\tau _ { i } > 0$ are trainable parameters initialized to 0 and 1, respectively. We still denote the modified velocity factors $\widetilde { \xi } _ { r } ^ { v _ { i } }$ by $\xi _ { r } ^ { v _ { i } }$ in what follows. Moreover, to improve numerical stability, each factor $\xi _ { r } ^ { s }$ and $\eta _ { r } ^ { s }$ in TNNs $f _ { \theta }$ and $\phi _ { \theta }$ is normalized in the $L ^ { 2 } \mathrm { - n o r m }$ , and the resulting scales are absorbed into $c _ { r } ^ { f }$ and $c _ { r } ^ { \phi }$ , respectively.

The TNN architecture provides three computational advantages. First, the number of network forward passes is reduced from $\mathcal { O } ( N ^ { D } )$ to $\mathcal { O } ( D N )$ . Second, the derivatives can be eficiently computed by forward-mode AD [7]. Third, macroscopic velocity moments are reduced from high-dimensional integrals to sums of products of one-dimensional integrals. Specifically, for a multi-index $\pmb { \alpha } = ( \alpha _ { 1 } , \dots , \alpha _ { d _ { v } } ) \in \mathbb { N } _ { 0 } ^ { d _ { v } }$ , we define $\begin{array} { r } { \mathbf { v } ^ { \alpha } = \prod _ { j = 1 } ^ { d _ { v } } v _ { j } ^ { \alpha _ { j } } } \end{array}$ and evaluate the velocity moment as

$$
\langle \mathbf { v } ^ { \alpha } f _ { \theta } \rangle = \sum _ { r = 1 } ^ { r _ { f } } c _ { r } ^ { f } \xi _ { r } ^ { t } ( t ) \prod _ { i = 1 } ^ { d _ { x } } \xi _ { r } ^ { x _ { i } } ( x _ { i } ) \prod _ { j = 1 } ^ { d _ { v } } \left( \int _ { \mathbb { R } } v _ { j } ^ { \alpha _ { j } } \xi _ { r } ^ { v _ { j } } ( v _ { j } ) \mathrm { d } v _ { j } \right) .\tag{3.11}
$$

The computational cost of moments is reduced from $\mathcal { O } ( N ^ { d _ { v } } )$ to $\mathcal { O } ( d _ { v } r _ { f } N )$ . This fast evaluation strategy will be used to construct the AP loss in Sect. 3.2.

## 3.2 Flux closure-enhanced AP loss

We first review the concept of AP loss and explain how it captures macroscopic evolution in the limit. Following the PINN framework, the TNNs $f _ { \theta }$ and $\phi _ { \theta }$ can be trained by minimizing a loss constructed by taking least square error of residuals of the kinetic system (2.1). $\mathrm { A s } \ \varepsilon  0$ , the kinetic equations formally reduce to

$$
\mathbf { E } \cdot { \boldsymbol { \nabla } } _ { \mathbf { v } } f = { \mathcal { Q } } ( f ) .\tag{3.12}
$$

This condition constrains only the local equilibrium and does not determine the macroscopic evolution for $\rho$ in the high-field limit. Hence the vanilla kinetic loss alone is insuficient to recover the correct high-field dynamics. To address this degeneracy in the limit, an asymptotic-preserving (AP) loss has been introduced for multiscale kinetic equations [21, 29]. Its key idea is to retain the original kinetic equation while incorporating macroscopic conservation laws into the loss. The resulting loss can thus preserve the essential macroscopic evolution as $\varepsilon \to 0$ . NNs trained with such losses are referred to as asymptotic-preserving neural networks (APNNs).

For (2.1), the baseline AP loss can be constructed by incorporating the continuity equation (2.8) into the loss [40]. Accordingly, we need to introduce another TNN $\rho _ { \boldsymbol { \theta } } ( t , \mathbf { x } )$ to approximate the density $\rho$

$$
\rho _ { \theta } ( t , \mathbf { x } ) = \sum _ { r = 1 } ^ { r _ { \rho } } c _ { r } ^ { \rho } \zeta _ { r } ^ { t } ( t ) \prod _ { i = 1 } ^ { d _ { x } } \zeta _ { r } ^ { x _ { i } } ( x _ { i } ) ,\tag{3.13}
$$

where $r _ { \rho }$ is the separation rank and $\{ c _ { r } ^ { \rho } \} _ { r = 1 } ^ { r _ { \rho } }$ are trainable coeficients. The one-dimensional factors $\zeta _ { r } ^ { t }$ and $\zeta _ { r } ^ { x _ { i } }$ are generated by independent subnetworks. The mass consistency condition $\rho _ { \theta } = \langle f _ { \theta } \rangle$ is also imposed in the loss. This AP loss provides a unified framework for the VPFP and BGK–Poisson systems. In particular, it does not require an explicit equilibrium distribution, which is not always available in the high-field regime.

We next examine how this baseline AP loss recovers the correct high-field limit. As $\varepsilon  0 .$ the kinetic residual yields the local equilibrium (3.12). Taking the first velocity moment of this equation and using (2.9), we obtain the flux closure

$$
\mathbf { j } = \rho \mathbf { E } .\tag{3.14}
$$

Substituting this relation into the continuity equation (2.8), together with the Poisson equation, recovers the high-field limit system (2.11). The loss therefore has the AP property. However, the limiting flux closure (3.14) is recovered from the kinetic residual without directly penalizing the closure error. This error can impair prediction accuracy for density evolution in the limit.

To improve prediction accuracy in the high-field limit, we enhance the baseline AP loss by incorporating the first velocity-moment equation (2.10)

$$
\varepsilon \left[ \partial _ { t } \langle \mathbf { v } f \rangle + \nabla _ { \mathbf { x } } \cdot \langle \mathbf { v } \otimes \mathbf { v } f \rangle \right] + \langle \mathbf { v } f \rangle - \rho \mathbf { E } = 0 .\tag{2.10}
$$

$\mathrm { A s } \ \varepsilon  0$ , this equation reduces to the flux closure (3.14). The corresponding loss term therefore

directly penalizes the closure error in the limit. Accordingly, we propose the following flux closure-enhanced AP formulation

$$
\left\{ \begin{array} { l l } { \displaystyle \partial _ { t } f + \mathbf { v } \cdot \nabla _ { \mathbf { x } } f + \frac { 1 } { \varepsilon } \mathbf { E } \cdot \nabla _ { \mathbf { v } } f = \frac { 1 } { \varepsilon } \mathcal { Q } ( f ) , } \\ { \displaystyle \partial _ { t } \rho + \nabla _ { \mathbf { x } } \cdot \langle \mathbf { v } f \rangle = 0 , } \\ { \varepsilon \left[ \partial _ { t } \langle \mathbf { v } f \rangle + \nabla _ { \mathbf { x } } \cdot \langle \mathbf { v } \otimes \mathbf { v } f \rangle \right] + \langle \mathbf { v } f \rangle - \rho \mathbf { E } = 0 , } \\ { \displaystyle \rho = \langle f \rangle , } \\ { \displaystyle - \Delta _ { \mathbf { x } } \phi = \rho - h , \quad \mathbf { E } = - \nabla _ { \mathbf { x } } \phi . } \end{array} \right.\tag{3.15}
$$

The added equation significantly improves the prediction accuracy, as domestrated by the ablation study in Figure 2. The enhanced AP formulation also has three advantages. First, the first moment equation is obtained exactly from the kinetic equation for every $\varepsilon > 0$ . Its inclusion alters neither the kinetic equation nor its solutions. Second, since the required velocity moments are computed directly from $f ,$ no additional solution variables are introduced. Third, these moments can be eficiently evaluated using TNN structure via (3.11).

We now specify the loss function associated with system (3.15). Its three solution variables $f , \rho ,$ , and $\phi ,$ are approximated by the TNNs $f _ { \theta } ( t , \mathbf { x } , \mathbf { v } ) , \rho _ { \theta } ( t , \mathbf { x } )$ , and $\phi _ { \theta } ( t , \mathbf { x } )$ , respectively. Here, $\theta$ collects all trainable parameters and $\mathbf { E } _ { \theta } = - \nabla _ { \mathbf { x } } \phi _ { \theta }$ . The residuals of (3.15) is defined as

$$
\begin{array} { r l } & { \mathcal { R } _ { \mathrm { k i n } } : = \varepsilon \big ( \partial _ { t } f _ { \theta } + \mathbf { v } \cdot \nabla _ { \mathbf { x } } f _ { \theta } \big ) + \mathbf { E } _ { \theta } \cdot \nabla _ { \mathbf { v } } f _ { \theta } - \mathcal { Q } _ { \theta } , } \\ & { \mathcal { R } _ { \mathrm { c o n t } } : = \partial _ { t } \rho _ { \theta } + \nabla _ { \mathbf { x } } \cdot \langle \mathbf { v } f _ { \theta } \rangle , } \\ & { \mathcal { R } _ { \mathrm { m o m } } : = \varepsilon \big [ \partial _ { t } \langle \mathbf { v } f _ { \theta } \rangle + \nabla _ { \mathbf { x } } \cdot \langle \mathbf { v } \otimes \mathbf { v } f _ { \theta } \rangle \big ] + \langle \mathbf { v } f _ { \theta } \rangle - \rho _ { \theta } \mathbf { E } _ { \theta } , } \\ & { \mathcal { R } _ { \mathrm { P } } : = - \Delta _ { \mathbf { x } } \phi _ { \theta } - \rho _ { \theta } + h , } \\ & { \mathcal { R } _ { \mathrm { m a s s } } : = \rho _ { \theta } - \langle f _ { \theta } \rangle , } \end{array}\tag{3.16}
$$

where $\mathcal { Q } _ { \theta } = \mathcal { Q } _ { \mathrm { F P } } ( f _ { \theta } )$ for the VPFP system, whereas $\mathcal { Q } _ { \theta } = \rho _ { \theta } M - f _ { \theta }$ for the BGK–Poisson system. Let $\begin{array} { r } { \Omega _ { \mathbf { x } } = \Pi _ { i = 1 } ^ { d _ { x } } \Omega _ { x _ { i } } } \end{array}$ denote the bounded spatial domain and $\begin{array} { r } { \Omega _ { \mathbf { v } } = \prod _ { j = 1 } ^ { d _ { v } } \Omega _ { v _ { j } } } \end{array}$ the truncated velocity domain for computation. The computational domains in both phase space and physical space are defined as

$$
\mathcal { D } _ { f } = ( 0 , T ) \times \Omega _ { \mathbf { x } } \times \Omega _ { \mathbf { v } } , \qquad \mathcal { D } _ { x } = ( 0 , T ) \times \Omega _ { \mathbf { x } } ,
$$

respectively. Their boundaries are denoted by $\Gamma _ { f } = ( 0 , T ) \times \partial ( \Omega _ { \mathbf { x } } \times \Omega _ { \mathbf { v } } )$ and $\Gamma _ { \rho } = \Gamma _ { \phi } =$ $( 0 , T ) \times \partial \Omega _ { \bf x }$ . The losses for the equation residuals and IBCs are then defined using squared $L ^ { 2 } \mathrm { - n o r m s }$

$$
\begin{array} { r l } & { \mathcal { L } _ { r } ( \theta ) : = \lambda _ { \mathrm { k i n } } \| \mathcal { R } _ { \mathrm { k i n } } \| _ { L ^ { 2 } ( \mathcal { D } _ { f } ) } ^ { 2 } + \lambda _ { \mathrm { c o n t } } \| \mathcal { R } _ { \mathrm { c o n t } } \| _ { L ^ { 2 } ( \mathcal { D } _ { x } ) } ^ { 2 } + \lambda _ { \mathrm { m o m } } \| \mathcal { R } _ { \mathrm { m o m } } \| _ { L ^ { 2 } ( \mathcal { D } _ { x } ) } ^ { 2 } } \\ & { \phantom { \mathcal { L } } + \lambda _ { \mathrm { P } } \| \mathcal { R } _ { \mathrm { P } } \| _ { L ^ { 2 } ( \mathcal { D } _ { x } ) } ^ { 2 } + \lambda _ { \mathrm { m a s s } } \| \mathcal { R } _ { \mathrm { m a s s } } \| _ { L ^ { 2 } ( \mathcal { D } _ { x } ) } ^ { 2 } , } \\ & { \mathcal { L } _ { \mathrm { b c } } ( \theta ) : = \| \mathcal { B } _ { f } [ f _ { \theta } ] \| _ { L ^ { 2 } ( \Gamma _ { f } ) } ^ { 2 } + \| \mathcal { B } _ { \rho } [ \rho _ { \theta } ] \| _ { L ^ { 2 } ( \Gamma _ { \rho } ) } ^ { 2 } + \| \mathcal { B } _ { \phi } [ \phi _ { \theta } ] \| _ { L ^ { 2 } ( \Gamma _ { \phi } ) } ^ { 2 } , } \\ & { \mathcal { L } _ { \mathrm { i c } } ( \theta ) : = \| f _ { \theta } ( 0 ) - f ^ { 0 } \| _ { L ^ { 2 } ( \Omega _ { \mathrm { x } } \times \Omega _ { \mathrm { v } } ) } ^ { 2 } + \| \rho _ { \theta } ( 0 ) - \rho ^ { 0 } \| _ { L ^ { 2 } ( \Omega _ { \mathrm { x } } ) } ^ { 2 } + \| \phi _ { \theta } ( 0 ) - \phi ^ { 0 } \| _ { L ^ { 2 } ( \Omega _ { \mathrm { x } } ) } ^ { 2 } . } \end{array}\tag{3.17}
$$

$\lambda _ { \mathrm { k i n } } , \lambda _ { \mathrm { c o n t } } , \lambda _ { \mathrm { m o m } } , \lambda _ { \mathrm { P } }$ , and $\lambda _ { \mathrm { m a s s } }$ are adjustable weights. The operators $B _ { f } , B _ { \rho } ,$ and $B _ { \phi }$ denote

the BC operators for $f , \rho$ and $\phi ,$ respectively. The ICs are assumed to be compatible, with $\rho ^ { 0 } = \langle f ^ { 0 } \rangle$ and − $\mathbf { \Delta } \cdot { \Delta _ { \mathbf { x } } } { \phi } ^ { 0 } = \rho ^ { 0 } - h$ . Finally, the total loss is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { A P T N N } } ( \theta ) = \mathcal { L } _ { r } ( \theta ) + \lambda _ { \mathrm { b c } } \mathcal { L } _ { \mathrm { b c } } ( \theta ) + \lambda _ { \mathrm { i c } } \mathcal { L } _ { \mathrm { i c } } ( \theta ) , } \end{array}\tag{3.18}
$$

where $\lambda _ { \mathrm { b c } }$ and $\lambda _ { \mathrm { i c } }$ are adjustable weights. We refer to the TNN-based method trained with this loss as the Asymptotic-Preserving Tensor Neural Network (APTNN). Due to the coordinateseparable structure of the TNN, the loss no longer needs to be evaluated on a full tensor-product grid. Instead, an eficient dimension-wise evaluation strategy will be presented in Sect. 3.3.

## 3.3 Dimension-wise evaluation of the loss

In this section, we show that the high-dimensional integrals in the APTNN loss (3.18) can be decomposed into to a series of one-dimensional integrals. Directly approximating these integrals using the standard Gauss–Legendre (GL) quadrature rule incurs prohibitive computational costs and memory requirements. If N quadrature points are used in each coordinate, evaluating the kinetic loss requires $N ^ { D _ { f } }$ quadrature points. This curse of dimensionality is further exacerbated in UQ problems by additional random variables. To reduce this computational burden, we exploit the coordinate-separable structure of TNNs to evaluate the loss dimension-wise [37, 15].

We first show that the equation residuals (3.16) for loss construction inherit separable structure of TNNs. It is straightforward to verify that the operations used in these residuals, including products, velocity integration, partial diferentiation and linear combinations of the resulting terms, preserve the separated structure. The Maxwellian M in the BGK collision is also rank-one separable in velocity. Therefore, $\mathcal { R } _ { \mathrm { k i n } } , \mathcal { R } _ { \mathrm { c o n t } }$ , each scalar component of $\mathcal { R } _ { \mathrm { m o m } } , \mathcal { R } _ { \mathrm { P } }$ and ${ \mathcal { R } } _ { \mathrm { m a s s } }$ all admit separated representations. Let $\mathcal { R } ( \mathbf { y } )$ denote any such residual component on the Cartesian-product domain $\begin{array} { r } { \mathcal { D } _ { \mathcal { R } } = \prod _ { \ell = 1 } ^ { D _ { \mathcal { R } } } \Omega _ { \ell } . } \end{array}$ where $\mathbf { y } = ( y _ { 1 } , \dots , y _ { D _ { \mathcal { R } } } )$ and $\Omega _ { \ell }$ is the one-dimensional domain of $y _ { \ell }$ . Here, $\mathcal { D } _ { \mathcal { R } }$ represents $\mathcal { D } _ { f }$ with dimension $D _ { \mathcal { R } } = D _ { f }$ for the kinetic residual, and $\mathcal { D } _ { x }$ with dimension $D _ { \mathcal { R } } = D _ { x }$ for the remaining equation residuals. We can write $\mathcal { R } ( \mathbf { y } )$ in a separated form:

$$
\mathcal { R } ( \mathbf { y } ) = \sum _ { k = 1 } ^ { K _ { \mathcal { R } } } a _ { k } \prod _ { \ell = 1 } ^ { D _ { \mathcal { R } } } R _ { k } ^ { ( \ell ) } ( y _ { \ell } ) ,\tag{3.19}
$$

where $K _ { \mathcal { R } }$ is the number of separated terms and $a _ { k }$ are scalar coeficients.

We apply the separated form of $\mathcal { R } ( \mathbf { y } )$ to decompose the equation loss $\mathcal { L } _ { r } \ ( 3 . 1 7 )$ into a series of one-dimensional integrals. Specifically, the high-dimensional integral in each residual loss can be rewritten as:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathcal { R } } : = \| \mathcal { R } ( \mathbf { y } ) \| _ { L ^ { 2 } ( \mathcal { D } _ { \mathcal { R } } ) } ^ { 2 } = \displaystyle \int _ { \prod _ { \ell = 1 } ^ { D _ { \mathcal { R } } } \Omega _ { \ell } } \left( \sum _ { k = 1 } ^ { K _ { \mathcal { R } } } a _ { k } \prod _ { \ell = 1 } ^ { D _ { \mathcal { R } } } R _ { k } ^ { ( \ell ) } ( y _ { \ell } ) \right) ^ { 2 } \mathrm { d } y _ { 1 } \cdot \cdot \cdot \cdot \mathrm { d } y _ { D _ { \mathcal { R } } } } \\ & { \quad = \displaystyle \sum _ { k = 1 } ^ { K _ { \mathcal { R } } } \sum _ { s = 1 } ^ { K _ { \mathcal { R } } } a _ { k } a _ { s } \prod _ { \ell = 1 } ^ { D _ { \mathcal { R } } } \left[ \displaystyle \int _ { \Omega _ { \ell } } R _ { k } ^ { ( \ell ) } ( y _ { \ell } ) R _ { s } ^ { ( \ell ) } ( y _ { \ell } ) \mathrm { d } y _ { \ell } \right] . } \end{array}\tag{3.20}
$$

One then approximates $\mathcal { L } _ { \mathcal { R } }$ using the GL quadrature rule. For each $\ell = 1 , \ldots , D _ { \mathcal { R } }$ , let

$\{ y _ { n } ^ { ( \ell ) } , w _ { n } ^ { ( \ell ) } \} _ { n = 1 } ^ { N _ { \ell } }$ denote the quadrature nodes and weights on $\Omega _ { \ell }$ . Each one-dimensional integral in (3.20) is approximated by

$$
G _ { k s } ^ { ( \ell ) } : = \sum _ { n = 1 } ^ { N _ { \ell } } w _ { n } ^ { ( \ell ) } R _ { k } ^ { ( \ell ) } \left( y _ { n } ^ { ( \ell ) } \right) R _ { s } ^ { ( \ell ) } \left( y _ { n } ^ { ( \ell ) } \right) , \qquad k , s = 1 , \ldots , K _ { \mathcal { R } } .\tag{3.21}
$$

Substituting (3.21) into (3.20), we obtain the dimension-wise approximation of $\mathcal { L } _ { \mathcal { R } }$

$$
{ \mathcal { L } } _ { \mathcal { R } } \approx \sum _ { k = 1 } ^ { K _ { \mathcal { R } } } \sum _ { s = 1 } ^ { K _ { \mathcal { R } } } a _ { k } a _ { s } \prod _ { \ell = 1 } ^ { D _ { \mathcal { R } } } G _ { k s } ^ { ( \ell ) } = \mathbf { a } ^ { \mathsf { T } } \left( G ^ { ( 1 ) } \odot \dots \odot G ^ { ( D _ { \mathcal { R } } ) } \right) \mathbf { a } ,\tag{3.22}
$$

where $G ^ { ( \ell ) } = [ G _ { k s } ^ { ( \ell ) } ] _ { k , s = 1 } ^ { K _ { \mathcal { R } } } \in \mathbb { R } ^ { K _ { \mathcal { R } } \times K _ { \mathcal { R } } }$ is a weighted Gram matrix, $\mathbf { a } = ( a _ { 1 } , \ldots , a _ { K _ { \mathcal { R } } } ) ^ { \mathsf { T } }$ is the coeficient vector, and $\odot$ denotes the Hadamard product. Thus, all residual losses can be evaluated using one-dimensional quadrature and Gram-matrix contractions, and then assembled to form the complete equation loss $\mathcal { L } _ { r }$ . Here, separable BCs and ICs are considered, and their losses ${ \mathcal { L } } _ { \mathrm { b c } }$ and ${ \mathcal { L } } _ { \mathrm { i c } }$ are also evaluated dimension-wise. Finally, the total APTNN loss can be evaluated through one-dimensionals quadrature instead of direct tensor-product quadrature.

We analyze the computational complexity and storage requirement for evaluating equation loss $\mathcal { L } _ { r }$ dimension-wise using (3.22). These estimates depend on the dimension $D _ { \mathcal { R } }$ , the number of separated terms $K _ { \mathcal { R } }$ and the quadrature sizes $N _ { \ell }$ . Here, $K _ { \mathcal { R } }$ is diferent for each equation residual in (3.16), since it depends on the TNN ranks $r _ { f } , r _ { \rho }$ and $r _ { \phi }$ together with the algebraic operations involved in each residual. Let $K _ { \mathrm { k i n } } ^ { \mathrm { F P } } , K _ { \mathrm { k i n } } ^ { \mathrm { B G K } } , K _ { \mathrm { c o n t } } , K _ { \mathrm { m o m } , i } , K _ { \mathrm { P } }$ , and $K _ { \mathrm { m a s s } }$ denote the numbers of separated terms in the corresponding equation residuals, where $K _ { \mathrm { m o m } , i }$ corresponds to the ith scalar component of ${ \mathcal { R } } _ { \mathrm { m o m } }$ . These numbers satisfy the following upper bounds:

$$
\begin{array} { r l r l } & { K _ { \mathrm { k i n } } ^ { \mathrm { F P } } \leq ( 1 + d _ { x } + d _ { v } ) r _ { f } + d _ { v } r _ { \phi } r _ { f } , } & & { \quad K _ { \mathrm { k i n } } ^ { \mathrm { B G K } } \leq ( d _ { x } + 2 ) r _ { f } + d _ { v } r _ { \phi } r _ { f } + r _ { \rho } , } \\ & { K _ { \mathrm { c o n t } } \leq r _ { \rho } + d _ { x } r _ { f } , } & & { \quad K _ { \mathrm { m o m } , i } \leq ( d _ { x } + 2 ) r _ { f } + r _ { \rho } r _ { \phi } , } \\ & { K _ { \mathrm { P } } \leq d _ { x } r _ { \phi } + r _ { \rho } + 1 , } & & { \quad K _ { \mathrm { m a s s } } \leq r _ { \rho } + r _ { f } . } \end{array}\tag{3.23}
$$

In particular, the products $E _ { \theta , i } \partial _ { v _ { i } } f _ { \theta }$ and $\rho _ { \boldsymbol { \theta } } E _ { \boldsymbol { \theta } , i }$ lead to multiplicative rank dependence into the separated term counts. They contribute at most $r _ { \phi } r _ { f }$ and $r _ { \rho } r _ { \phi }$ separated terms to $K _ { \mathrm { k i n } }$ and $K _ { \mathrm { m o m } , i }$ , respectively. The computational complexity and storage requirements for $\mathcal { L } _ { r }$ is derived by combining the bounds (3.23) with the loss evaluation formula (3.22), as follows:

Proposition 3.3. Assume that at most N quadrature points are used in each coordinate. Let $K _ { \mathrm { k i n } } , K _ { \mathrm { c o n t } } , K _ { \mathrm { m o m } , i } , K _ { \mathrm { P } }$ and $K _ { \mathrm { m a s s } }$ satisfy the bounds in (3.23), where $K _ { \mathrm { k i n } } = K _ { \mathrm { k i n } } ^ { \mathrm { F P } }$ for the VPFP system and $K _ { \mathrm { k i n } } = K _ { \mathrm { k i n } } ^ { \mathrm { B G K } }$ for the BGK–Poisson system. When the loss of equation residual $\mathcal { L } _ { r }$ in (3.17) is evaluated dimension-wise using (3.22), its computational complexity is

$$
\mathcal { O } \left( N \left[ D _ { f } K _ { \mathrm { k i n } } ^ { 2 } + D _ { x } \left( K _ { \mathrm { c o n t } } ^ { 2 } + \sum _ { i = 1 } ^ { d _ { v } } K _ { \mathrm { m o m } , i } ^ { 2 } + K _ { \mathrm { P } } ^ { 2 } + K _ { \mathrm { m a s s } } ^ { 2 } \right) \right] \right) ,\tag{3.24}
$$

where $D _ { f } = 1 + d _ { x } + d _ { v } + d _ { z }$ and $D _ { x } = 1 + d _ { x } + d _ { z }$ with $d _ { z } = 0$ in the deterministic setting. The

storage requirement for loss evaluation is

$$
\begin{array} { r l } & { \mathcal { O } \bigg ( N \left[ D _ { f } K _ { \mathrm { k i n } } + D _ { x } \left( K _ { \mathrm { c o n t } } + \sum _ { i = 1 } ^ { d _ { v } } K _ { \mathrm { m o m } , i } + K _ { \mathrm { P } } + K _ { \mathrm { m a s s } } \right) \right] } \\ & { \quad \quad + K _ { \mathrm { k i n } } ^ { 2 } + K _ { \mathrm { c o n t } } ^ { 2 } + \sum _ { i = 1 } ^ { d _ { v } } K _ { \mathrm { m o m } , i } ^ { 2 } + K _ { \mathrm { P } } ^ { 2 } + K _ { \mathrm { m a s s } } ^ { 2 } \bigg ) . } \end{array}\tag{3.25}
$$

Proof. We first estimate the computational cost. For any equation residual $\mathcal { R } ( \mathbf { y } )$ of the form (3.19), computing the Gram matrix $G ^ { ( \ell ) }$ in the ℓth coordinate used in (3.22) requires $\mathcal { O } ( N _ { \ell } K _ { \mathcal { R } } ^ { 2 } )$ operations. Since $N _ { \ell } \leq N$ , constructing all $D _ { \mathcal { R } }$ Gram matrices requires $\mathcal { O } ( N D _ { \mathcal { R } } K _ { \mathcal { R } } ^ { 2 } )$ operations. The Hadamard product and the final quadratic form require $\mathcal { O } ( D _ { \mathcal { R } } K _ { \mathcal { R } } ^ { 2 } )$ and $\mathcal { O } ( K _ { \mathcal { R } } ^ { 2 } )$ operations, respectively. Since these two costs are of no higher order than the Gram-matrix construction cost, summing the bounds $\mathcal { O } ( N D _ { \mathcal { R } } K _ { \mathcal { R } } ^ { 2 } )$ over all residual components in $\mathcal { L } _ { r }$ yields the result.

We next estimate the storage required for loss evaluation. For any residual $\mathcal { R } ( \mathbf { y } )$ , storing the values $R _ { k } ^ { ( \ell ) } ( y _ { n } ^ { ( \ell ) } )$ for Gram-matrix computation in (3.21) requires $N _ { \ell } K _ { \mathcal { R } }$ entries per coordinate, giving a total storage of $\mathcal { O } ( N D _ { \mathcal { R } } K _ { \mathcal { R } } )$ . The Hadamard product in (3.22) can be accumulated sequentially across coordinates, retaining only the current Gram matrix and the accumulated product matrix. Both matrices have $K _ { \mathcal { R } } ^ { 2 }$ entries, so the additional storage is $\mathcal { O } ( K _ { \mathcal { R } } ^ { 2 } )$ . Summing these storage bounds over all residual components in $\mathcal { L } _ { r }$ yields (3.25). □

To extract the leading-order rank dependence of both bounds in Proposition 3.3, we assume that the TNN ranks satisfy min $\{ r _ { f } , r _ { \rho } , r _ { \phi } \}$ ≫ max $\{ 1 , d _ { x } , d _ { v } \}$ . Since $\rho$ and $\phi$ are independent of velocity, we choose smaller ranks $r _ { \rho } , r _ { \phi } < r _ { f }$ . Under the above assumptions, both bounds are controlled by the kinetic residual contribution with $K _ { \mathrm { k i n } } = \mathcal { O } ( d _ { v } r _ { f } r _ { \phi } )$ . Hence, these bounds simplify as follows:

Corollary 3.4. Under the rank assumptions above, the computational complexity of the dimension-wise evaluation of the equation loss $\mathcal { L } _ { r }$ is

$$
\mathcal { O } \left( N D _ { f } d _ { v } ^ { 2 } r _ { f } ^ { 2 } r _ { \phi } ^ { 2 } \right) .\tag{3.26}
$$

The corresponding storage requirement is

$$
\mathcal { O } \Bigl ( N D _ { f } d _ { v } r _ { f } r _ { \phi } + d _ { v } ^ { 2 } r _ { f } ^ { 2 } r _ { \phi } ^ { 2 } \Bigr ) .\tag{3.27}
$$

For fixed TNN ranks and quadrature size, the computational and storage bounds have at most cubic and quadratic dependence on the physical dimensions, respectively, and at most linear dependence on the random dimension.

In contrast, for previous TNN methods applied to kinetic equations [32, 6], full-grid evaluation of the equation loss entails $\mathcal { O } ( N ^ { D _ { f } } )$ computational complexity and storage requirements.

## 4 Numerical results

In this section, we present numerical experiments to assess the accuracy, computational eficiency and scalability of APTNN. Reference solutions are computed using the AP schemes developed for the VPFP system [24] and the semiconductor Boltzmann equation [25]. We extend both schemes to the four- and six-dimensional phase spaces and implement them in Python. Both schemes employ a second-order IMEX time discretization, a MUSCL spatial discretization, and an FFT-based Poisson solver.

In APTNN numerical realization, each one-dimensional subnetwork adopts the modified MLP architecture [36] with the Swish activation function [34]. The subnetworks for the physical coordinates have 4 hidden layers of 64 neurons each, and those for the random coordinates have 3 hidden layers of 32 neurons. All weight matrices are initialized using the Glorot normal initializer. Periodic BCs are used in each spatial direction for all examples. These conditions are enforced as hard constraints by applying a Fourier embedding to each spatial input [28, 30]. Specifically, each input $x _ { i } \in [ a _ { i } , b _ { i } ]$ is mapped to $\gamma _ { i } ( x _ { i } ) = \left( \cos ( \omega _ { i } x _ { i } ) , \sin ( \omega _ { i } x _ { i } ) , \ldots , \cos ( J \omega _ { i } x _ { i } ) , \sin ( J \omega _ { i } x _ { i } ) \right)$ where $\omega _ { i } = 2 \pi / P _ { i }$ and $P _ { i } = b _ { i } - a _ { i }$ . We set $J = 1$ throughout. Accordingly, the boundary loss ${ \mathcal { L } } _ { \mathrm { b c } }$ in (3.18) is omitted. The remaining loss terms are evaluated with N Gauss–Legendre (GL) quadrature points in each coordinate direction. The networks are trained for 60,000 iterations using the Adam optimizer. The learning rate is initialized to 10<sup>−3</sup> and reduced by a factor of 0.9 every 1,000 iterations. The implementation uses JAX and Flax.

We report relative space–time $L ^ { 2 } \mathrm { - e r r o r s }$ for the density ρ and the electric field E. In particular, we focus on density perturbations around the uniform background $h = 1$ and define the relative error in $\rho - 1$ as

$$
e _ { \rho - 1 } = \frac { \| \rho _ { \mathrm { p r e d } } - \rho _ { \mathrm { r e f } } \| _ { L ^ { 2 } ( [ 0 , T ] \times \Omega _ { \mathbf { x } } ) } } { \| \rho _ { \mathrm { r e f } } - 1 \| _ { L ^ { 2 } ( [ 0 , T ] \times \Omega _ { \mathbf { x } } ) } } ,\tag{4.1}
$$

where $\rho _ { \mathrm { p r e d } }$ and $\rho _ { \mathrm { r e f } }$ denote the predicted and reference densities, respectively. The error e<sub>E</sub> of the electric field is defined analogously. For multi-dimensional problems, the norm of $\mathbf { E } = \left( E _ { 1 } , \ldots , E _ { d _ { x } } \right)$ is defined by $\begin{array} { r } { \| \mathbf { E } \| _ { L ^ { 2 } } ^ { 2 } = \sum _ { i = 1 } ^ { d _ { x } } \| E _ { i } \| _ { L ^ { 2 } } ^ { 2 } } \end{array}$ . The reference solutions are generated on a single node of an HPC cluster with two Intel Xeon Platinum 8358 processors (32 cores each, 2.6 GHz) and 512 GB of memory. All network training is performed on a single NVIDIA A100 GPU with 40 GB of memory. We note that no extensive hyperparameter tuning is performed for the networks.

## 4.1 AP property and prediction accuracy

We consider a one-dimensional Landau damping problem to examine the AP property of APTNN for both the VPFP system and the BGK–Poisson system. The initial distribution is given by

$$
f ^ { 0 } ( x , v ) = ( 1 + \alpha \cos ( k x ) ) \frac { e ^ { - v ^ { 2 } / 2 } } { \sqrt { 2 \pi } } ,\tag{4.2}
$$

where $\alpha = 0 . 1$ and $k = 0 . 2 5$ . The spatial domain is $x \in [ 0 , L ]$ with $L = 2 \pi / k$ and the velocity domain is truncated to $v \in [ - V , V ]$ with $V = 6$ . We set the terminal time $T = 1$ and consider $\varepsilon \in \{ 1 , 1 0 ^ { - 1 } , 1 0 ^ { - 2 } , 1 0 ^ { - 3 } \}$ . Reference solutions for both models are computed using 256 uniform spatial cells, 128 velocity grid points and a time step $\Delta t = 2 \times 1 0 ^ { - 3 }$ . We compare vanilla TNN, baseline APTNN and the proposed APTNN. The vanilla TNN loss is based on the original kinetic system, whereas the proposed APTNN adds the first velocity-moment equation to the baseline AP loss. We set the separation ranks to $r _ { f } = 5 0$ and $r _ { \phi } = 2 0$ for all three methods, and $r _ { \rho } = 2 0$ for both APTNN variants. The loss is evaluated with $N = 3 2 { \mathrm { ~ G L } }$ quadrature points in each coordinate direction, and all loss weights are set to one. With these settings, training the proposed APTNN takes approximately 4 minutes per run for both the VPFP and BGK–Poisson systems.

Figure 2 compares the relative $L ^ { 2 } .$ -errors of the density perturbation $\rho - 1$ and the electric field $E$ obtained using the three loss formulations. The vanilla TNN becomes inaccurate as ε decreases and therefore does not have the $\mathrm { A P }$ property. In contrast, both APTNN variants remain accurate for all tested ε, supporting their AP property. Incorporating the first velocity-moment equation into the AP loss further improves prediction accuracy. Specifically, for the VPFP system, the proposed APTNN reduces the errors from $\mathcal { O } ( 1 0 ^ { - 2 } )$ to $\mathcal { O } ( 1 0 ^ { - 3 } )$ . For the BGK–Poisson system, the baseline APTNN errors fluctuate between $1 0 ^ { - 3 }$ and $1 0 ^ { - 2 }$ , whereas the proposed APTNN stays around $1 0 ^ { - 3 }$

We next extend the simulations to $T = 1 0$ with $\varepsilon = 1$ to examine the evolution of the electric-field energy, defined by

$$
\mathcal { E } _ { E } ( t ) = \frac { 1 } { 2 } \int _ { \Omega _ { x } } | E ( t , x ) | ^ { 2 } \mathrm { d } x = \frac { 1 } { 2 } \| E ( t , \cdot ) \| _ { L ^ { 2 } ( \Omega _ { x } ) } ^ { 2 } .\tag{4.3}
$$

To stabilize training, we divide [0, 10] into two windows, [0, 5] and [5, 10], and train sequentially using a time-marching strategy [7]. The terminal prediction from the first window is used as the ICs for the second. The network ranks and quadrature settings remain unchanged. The loss weights for the kinetic, continuity and Poisson residuals are adjusted to 50 for both systems, while all other weights remain one. The network is trained for 60,000 iterations on each window, with a training time comparable to that of the $T = 1$ case. Figure 3 shows good agreement between the APTNN predictions and the reference solutions for both systems. The relative $L ^ { 2 } ( [ 0 , T ] )$ )-errors of the electric-field energy are $4 . 4 8 \times 1 0 ^ { - 3 }$ for the VPFP system and $3 . 0 6 \times 1 0 ^ { - 2 }$ for the BGK-Poisson system.

## 4.2 Eficiency of dimension-wise loss evaluation

This section assess the computational cost and memory usage of dimension-wise loss evaluation. It also checks the ability of APTNN to capture non-equilibrium dynamics. We consider a 2D2V VPFP problem with a double-stream initial distribution

$$
f ^ { 0 } ( { \bf x } , { \bf v } ) = \left[ 1 + \alpha \cos ( k x _ { 1 } ) \cos ( k x _ { 2 } ) \right] \frac { e ^ { - ( v _ { 1 } - 1 . 5 ) ^ { 2 } / 2 } + e ^ { - ( v _ { 1 } + 1 . 5 ) ^ { 2 } / 2 } } { 2 \sqrt { 2 \pi } } \frac { e ^ { - v _ { 2 } ^ { 2 } / 2 } } { \sqrt { 2 \pi } } ,\tag{4.4}
$$

Density perturbation $e _ { \rho - 1 }$ Electric field $e _ { E }$  
![](images/608c117716fe56f6fd229dca8e1d7e42b8861f99355a50f12744c8cc6b1d5557.jpg)

![](images/039811afcb1fc91d7e02f996d8a9c91677a1d05cd9cfe34b7079e1f62402ae96.jpg)  
Vanilla TNN Baseline APTNN APTNN

(a) VPFP system  
Density perturbation $e _ { \rho - 1 }$ Electric field $e _ { E }$  
![](images/6dcb9e8b0898969a83647b48796d01cf7fc66ceb750efca8f96243e5a00a2301.jpg)

![](images/02d708a797a6828c15336f7f34e362af7203edd39e7daff7ef7f665c991468f3.jpg)  
Vanilla TNN Baseline APTNN APTNN  
(b) BGK-Poisson system

Figure 2: AP property and prediction accuracy for the one-dimensional Landau damping problem. Relative space–time $L ^ { 2 } .$ errors of the density perturbation $\rho - 1$ and the electric field E are shown for (a) the VPFP system and (b) the BGK–Poisson system. The three methods are vanilla TNN, baseline APTNN and the proposed $\mathrm { A P T N N }$ , which adds the first velocity-moment equation to the baseline AP loss. The scale parameter ε decreases from left to right, from the kinetic to the high-field regime.

![](images/d1b387bfdf3a6c55a59a2b23b7f57ae94f1aa0d86b2d5a0fdbf6740a9acc9f39.jpg)  
(a) VPFP system

![](images/17585f34ddab2e00384684d14afd35f84046ce6e42bc77c8317817c8e3815898.jpg)  
(b) BGK–Poisson system  
Figure 3: Evolution of the electric-field energy $\mathcal { E } _ { E } ( t )$ for the one-dimensional Landau damping problem with $T = 1 0$ and $\varepsilon = 1$ for (a) the VPFP system and (b) the BGK–Poisson system. The solid and dashed curves denote the reference solutions and the APTNN predictions, respectively. The vertical axis is on a logarithmic scale.

where $\alpha = 0 . 0 5$ and $k = 0 . 5$ . The spatial domain is $( x _ { 1 } , x _ { 2 } ) \in [ 0 , 2 \pi / k ] ^ { 2 }$ and the velocity domain is truncated to $( v _ { 1 } , v _ { 2 } ) \in [ - 6 , 6 ] ^ { 2 }$ . We set $T = 1$ and $\varepsilon = 1 0 ^ { - 3 }$ . The reference solution is computed using 128 uniform cells in each spatial direction, 48 grid points in each velocity direction and a time step $\Delta t = 2 \times 1 0 ^ { - 3 }$

We compare dimension-wise and full-grid evaluation of the same loss. The former uses one-dimensional quadrature and Gram-matrix contractions, whereas the latter evaluates the loss directly using full tensor-product quadrature points. The quadrature orders are set as $N = 8 , 1 2 , 1 6 , 2 4 , 3 2 , 4 0 , 4 8 , 6 4$ . The separation ranks of TNNs are set to $r _ { f } = 5 0$ and $r _ { \rho } = r _ { \phi } = 2 0$ and all loss weights to one. For each $N .$ , both strategies are evaluated using the same network parameters. We time 30 synchronized training steps after JIT compilation and warm-up, and report the median wall-clock time per step and peak GPU memory usage. For all tested N, the absolute diference between the loss values computed by the two strategies is of order $1 0 ^ { - 1 3 }$ Hence, we report only the relative $L ^ { 2 } \mathrm { - e r r o r s }$ of $\mathrm { A P T N N }$ trained by dimension-wise strategy.

Figure 4 exhibits the time per training step, peak GPU memory usage and relative $L ^ { 2 }$ -errors in panels (a)–(c), respectively. Panels (a) and (b) show that dimension-wise evaluation maintains relatively stable time and memory costs over the tested range, requiring at most 12.5 ms per training step and 0.27 GB of peak GPU memory. In contrast, full-grid evaluation exhibits a steep increase in both metrics as N increases. Since this strategy computs the kinetic loss using all $N ^ { 5 }$ quadrature points, its time per training step rises from 7.77 ms to 838.13 ms, and its peak GPU memory usage rises from 0.023 GB to 15.37 GB. For $N \leq 1 6$ , the full-grid evaluation is slightly more eficient because the dimension-wise evaluation is dominated by the fixed rank-dependent cost of Gram-matrix contractions. However, the errors have not yet converged in this range as shown in panel (c). From $N = 2 4$ onward, dimension-wise evaluation becomes more eficient and its advantage grows rapidly with N.

These results demonstrate that dimension-wise evaluation requires substantially less computation and memory than full-grid evaluation to support fine quadrature for accurate predictions. This advantage becomes more pronounced in higher-dimensional kinetic and UQ problems, because the $N ^ { D _ { f } }$ full-grid size makes increasing N much more expensive.

Figure 5 compares the APTNN predictions for $\rho , E _ { 1 }$ and $E _ { 2 }$ with the reference solution at $T = 1$ using $N = 3 2$ . The relative space–time $L ^ { 2 } .$ -errors of the density perturbation $\rho - 1$ and the vector electric field $\mathbf { E } = ( E _ { 1 } , E _ { 2 } )$ are $5 . 6 2 \times 1 0 ^ { - 3 }$ and $2 . 5 5 \times 1 0 ^ { - 3 }$ , respectively. Therefore, APTNN can capture the non-equilibrium dynamics accurately. The training time is approximately 7 minutes. It is measured without per-step synchronization and is not directly comparable to the time per training step in Figure $4 ( \mathrm { a } )$

## 4.3 3D3V problem

We examine the accuracy of APTNN in high-dimensional settings using a 3D3V BGK–Poisson problem. The initial distribution is

$$
f ^ { 0 } (  { \mathbf { x } } ,  { \mathbf { v } } ) = \left[ 1 + \alpha \sum _ { i = 1 } ^ { 3 } \cos ( k x _ { i } ) \right] \prod _ { i = 1 } ^ { 3 } \frac { e ^ { - v _ { i } ^ { 2 } / 2 } } { \sqrt { 2 \pi } } ,\tag{4.5}
$$

![](images/627946c816035369a1a274b534b183e5599bd7800997ebaf9057e97483c7047c.jpg)  
(a) Time per training step

![](images/e0e337452bbe2cab4d418e595b774acce31b6a98e9e5a7f5bcb319d300c5256d.jpg)  
(b) Peak GPU memory

![](images/9464c9fe85842b8e54cb9e80bb3c739e8c1a7ef6fc2b9039d7cc60217a5e0fc0.jpg)  
(c) APTNN relative errors  
Figure 4: Computational cost, memory usage and errors for the 2D2V double-stream problem with $\varepsilon = 1 0 ^ { - 3 }$ and $T = 1$ as the quadrature order N varies. Panels (a) and (b) show the median training-step time and peak GPU memory, respectively, for both dimension-wise and full-grid loss evaluation. Panel (c) shows the relative space–time $L ^ { 2 }$ errors of the density perturbation $\rho - 1$ and the electric field E for APTNN trained by dimension-wise evaluation.

![](images/f5d34d5a78cdb3b00de50a2e71da96ae4fbc27ab6dcc15faaf67ff6242dc44aa.jpg)

![](images/018c11547ea75332c4a1ffdbe0ea94904e955227d82a42446cac3ae5ab5d5061.jpg)

![](images/56ecc6c419451780d9e8831944c49e2938340c4939027be8b334ea5a0aae70e8.jpg)  
Figure 5: Results for the 2D2V double-stream problem with $\varepsilon = 1 0 ^ { - 3 }$ at $T = 1$ using $N = 3 2$ . The top and bottom rows show the APTNN predictions and reference solutions, respectively. The columns show the density $\rho$ and the electric-field components $E _ { 1 }$ and $E _ { 2 }$ , respectively.

where $\alpha = 0 . 0 5$ and $k = 0 . 5$ . The spatial domain is $( x _ { 1 } , x _ { 2 } , x _ { 3 } ) \in [ 0 , 2 \pi / k ] ^ { 3 }$ and the velocity domain is truncated to $( v _ { 1 } , v _ { 2 } , v _ { 3 } ) \in [ - 6 , 6 ] ^ { 3 }$ . We set $T = 1$ and consider $\varepsilon \in \{ 1 , 1 0 ^ { - 3 } \}$ . The reference solutions are computed using 80 uniform cells in each spatial direction, 16 grid points in each velocity direction and a time step $\Delta t = 2 \times 1 0 ^ { - 3 }$ . The separation ranks of TNN are set to $r _ { f } = 5 0$ and $r _ { \rho } = r _ { \phi } = 2 0$ . The loss is evaluated using N = 32 GL quadrature points in each coordinate direction. For $\varepsilon = 1$ , the IC loss for f is assigned a weight of 200, while all other loss weights are set to 1. For $\varepsilon = 1 0 ^ { - 3 }$ , all loss weights are set to 1. The corresponding training times are approximately 12 and 14 minutes, respectively.

Figures 6 and 7 compare the APTNN predictions with the reference solutions at $T = 1$ for $\varepsilon = 1$ and $1 0 ^ { - 3 }$ , respectively. The predicted density and electric field agree well with the reference solutions in both regimes. For $\varepsilon = 1$ , the relative space–time $L ^ { 2 } .$ -errors of the density perturbation $\rho - 1$ and the vector electric field $\mathbf { E } = ( E _ { 1 } , E _ { 2 } , E _ { 3 } )$ are $8 . 7 8 \times 1 0 ^ { - 3 }$ and $7 . 0 2 \times 1 0 ^ { - 3 }$ respectively. The corresponding errors for $\varepsilon = 1 0 ^ { - 3 }$ are $1 . 9 8 \times 1 0 ^ { - 3 }$ and $1 . 6 1 \times 1 0 ^ { - 3 }$

![](images/fa33a0da2366bd631dc9b17118d6c226f654644b5534599dc11a97b4461ff54e.jpg)

![](images/7a30b09346592d6d8f31e51b8a217de0ba8d534ff820724d7db3fa3e038343f1.jpg)

![](images/17e7b5769534c169f0b90acc865b3f0635564a5155c300023b3059b95ba31bf0.jpg)

![](images/a483582d9f69e11b50b7522f14f4f2355e7aa51b894083e37a49e3d37fb2c5b7.jpg)  
Figure 6: Results for the 3D3V BGK–Poisson problem with $\varepsilon = 1$ at $T = 1$ . The top and bottom rows show the $\mathrm { A P T N N }$ predictions and reference solutions, respectively. The columns for $\rho , E _ { 1 }$ , and $E _ { 2 }$ show the $x _ { 1 } - x _ { 2 }$ plane at $x _ { 3 } = \pi$ , while the $E _ { 3 }$ column shows the $x _ { 1 }$ –x plane at $x _ { 2 } = \pi$

## 4.4 UQ for 3D3V problems

We consider the 3D3V VPFP system with random inputs to evaluate the performance of APTNN in multi-dimensional UQ problems. Two cases with random dimensions $d _ { z } = 3$ and $d _ { z } = 5$ are tested. For $d _ { z } = 3$ , the initial distribution is

$$
f _ { 3 } ^ { 0 } ( \mathbf { x } , \mathbf { v } , \mathbf { z } ) = \left[ 1 + \alpha \sum _ { i = 1 } ^ { 3 } ( 1 + z _ { i } ) \cos ( k x _ { i } ) \right] \prod _ { i = 1 } ^ { 3 } \frac { e ^ { - v _ { i } ^ { 2 } / 2 } } { \sqrt { 2 \pi } } .\tag{4.6}
$$

For $d _ { z } = 5$ , we take

$$
f _ { 5 } ^ { 0 } ( \mathbf x , \mathbf v , \mathbf z ) = \left[ 1 + \alpha \left( 1 + \frac { 1 } { 5 } \sum _ { m = 1 } ^ { 5 } z _ { m } \right) \sum _ { i = 1 } ^ { 3 } \cos ( k x _ { i } ) \right] \prod _ { i = 1 } ^ { 3 } \frac { e ^ { - v _ { i } ^ { 2 } / 2 } } { \sqrt { 2 \pi } } .\tag{4.7}
$$

![](images/cac64c17381e140b8b739ee99b3e03c589b1e9953bd8250236670f3f6bafdc45.jpg)

![](images/e99ffe3f6a2e84865ee58c7431fc89900f92a5a4c924660b1b3c4adb0abc00f5.jpg)

![](images/a9ca4b44f55a19fe0e6d03184d0dcbc818deae341a1ddf532190719f77ebb541.jpg)

![](images/35c0d3f125236f4e24a067697d14566c2815ab2398f46d4d3bef4501888fade9.jpg)  
Figure 7: Results for the 3D3V BGK–Poisson problem with $\varepsilon = 1 0 ^ { - 3 }$ at $T = 1$ . The top and bottom rows show the APTNN predictions and reference solutions, respectively. The columns for $\rho , E _ { 1 }$ , and $E _ { 2 }$ show the $x _ { 1 } - x _ { 2 }$ plane at $x _ { 3 } = \pi$ , while the $E _ { 3 }$ column shows the $x _ { 1 } - x _ { 3 }$ plane at $x _ { 2 } = \pi$

Here, $z _ { 1 } , \ldots , z _ { d _ { z } }$ are independent random variables uniformly distributed on $[ - 1 , 1 ]$ . The spatial and velocity domains, BCs and parameters α and k remain the same as in the preceding 3D3V problem. We set $\varepsilon = 1 0 ^ { - 4 }$ and $T = 1$ and evaluate the means and standard deviations of $\rho$ and E. The reference statistics are computed nonintrusively using a level-4 Smolyak sparse-grid rule based on nested Clenshaw–Curtis points [35, 8]. The deterministic solver uses 48 uniform cells in each spatial direction, 20 grid points in each velocity direction and a time step $\Delta t = 4 \times 1 0 ^ { - 3 }$ The separation ranks of TNNs are set to $r _ { f } = 5 0$ and $r _ { \rho } = r _ { \phi } = 3 0$ . The loss is evaluated using 32 GL quadrature points in each physical coordinate and 16 in each random coordinate. All loss weights are set to 1. The training time is approximately 23 minutes for $d _ { z } = 3$ and 25 minutes for $d _ { z } = 5$ . In contrast, full-grid loss evaluation would far exceed the memory capacity of a single GPU in both cases. Even for $d _ { z } = 3$ , evaluating the kinetic loss would require over 500 TB of memory.

Figures 8 and 9 compare the APTNN predictions with the reference statistics at $T = 1$ for $d _ { z } = 3$ and $d _ { z } = 5$ , respectively. The predicted means and standard deviations of both the density and the electric field agree closely with the reference statistics in both cases. For $d _ { z } = 3$ the relative space–time $L ^ { 2 } \mathrm { - e r r o r s }$ for the density are $3 . 8 7 \times 1 0 ^ { - 3 }$ for $\mathbb E [ \rho ] - 1$ and $1 . 0 0 \times 1 0 ^ { - 2 }$ for $\mathrm { S t d } [ \rho ]$ , while those for the electric field are $2 . 5 1 \times 1 0 ^ { - 3 }$ for E[E] and $7 . 3 1 \times 1 0 ^ { - 3 }$ for Std[E]. For $d _ { z } = 5$ , the corresponding errors are $3 . 8 1 \times 1 0 ^ { - 3 }$ and $1 . 4 7 \times 1 0 ^ { - 2 }$ for the density, and $2 . 4 4 \times 1 0 ^ { - 3 }$ and $1 . 1 3 \times 1 0 ^ { - 2 }$ for the electric field.

Finally, Table 1 summarizes the wall-clock training times for the representative experiments presented in Sect. 4.

x<sub>1</sub>  
x<sub>1</sub>  
![](images/726fdc05790f6b9574f8769b1a0419604498a0a0b256adfea780fc3e6d00fb7e.jpg)

![](images/05431a738f9a0f16c8fcfc13b25defc7d087766b6449d56af5704e1a1ce52eb0.jpg)

![](images/cb6bf3c8da1b5982e1496d346f66736afab1296167d464b8d5ee5527f155e3d8.jpg)

![](images/8f4dd55b99f3bfa8601d7d1b7dfbd3ae39be2f8e8cd7e1b96aac12208d9cf240.jpg)  
Figure 8: Means and standard deviations in the UQ problem for the 3D3V VPFP system with $d _ { z } = 3$ and $\varepsilon = 1 0 ^ { - 4 }$ at $T = 1$ . The top and bottom rows show the APTNN predictions and reference statistics, respectively. The columns show E[ρ], E[E ], Std[ρ] and $\mathrm { S t d } [ E _ { 1 } ]$ on the $x _ { 1 } - x _ { 2 }$ plane at $x _ { 3 } = \pi$

![](images/76970202dad953d4ec60fc3daf70e14c4acf2cc555f7c816e6d9936702aa3092.jpg)

![](images/5264504c2b4703e8bbd7ee59155fc029641821ca1c726c405ab63ce07c02cb2c.jpg)

![](images/182f8d3b4ef12d778234daad847e1cec4f052bdd863268b3364cfe41f61e8404.jpg)

![](images/fa2cad114d6c355915d47a902d127d4ea143d7a3ad6a5dd0755695304857215c.jpg)  
Figure 9: Means and standard deviations in the UQ problem for the 3D3V VPFP system with $d _ { z } = 5$ and $\varepsilon = 1 0 ^ { - 4 }$ at $T = 1$ . The top and bottom rows show the APTNN predictions and reference statistics, respectively. The columns show E[ρ], E[E ], Std[ρ] and $\mathrm { S t d } [ E _ { 1 } ]$ on the $x _ { 1 } - x _ { 2 }$ plane at $x _ { 3 } = \pi .$

Table 1: Wall-clock training times of APTNN for the numerical experiments in Sect. 4.
<table><tr><td>Problem</td><td>Model</td><td>Setting</td><td> $d _ { z }$  Training time (s)</td></tr><tr><td>1D1V</td><td>VPFP</td><td> $\varepsilon = 1 , T = 1$ </td><td>243.2</td></tr><tr><td>1D1V</td><td>BGK-Poisson</td><td> $\varepsilon = 1 , T = 1$ </td><td>245.0</td></tr><tr><td>2D2V</td><td>VPFP</td><td> $\varepsilon = 1 0 ^ { - 3 } , N = 3 2$ </td><td>439.4</td></tr><tr><td>3D3V</td><td>BGK-Poisson ε = 1</td><td></td><td>673.1</td></tr><tr><td>3D3V</td><td>BGK-Poisson</td><td> $\varepsilon = 1 0 ^ { - 3 }$ </td><td>730.2</td></tr><tr><td>3D3V UQ</td><td>VPFP</td><td> $\varepsilon = 1 0 ^ { - 4 }$ </td><td>1352.3</td></tr><tr><td>3D3V UQ</td><td>VPFP</td><td> $\varepsilon = 1 0 ^ { - 4 }$ </td><td>1524.8</td></tr></table>

## 5 Conclusion and discussion

In this paper, an asymptotic-preserving tensor neural network (APTNN) method is proposed for high-dimensional kinetic equations with uncertainty in the high-field regime, particularly for the Vlasov–Poisson–Fokker–Planck system and the BGK-Poisson system. To alleviate the curse of dimensionality, we adopt TNN to approximate the solutions as sums of products of one dimensional functions and apply dimension-wise integration to evaluate the loss, avoiding costly computations and storage on full tensor grids. To improve prediction accuracy in the high-field limit, we further incorporate the first velocity-moment equation into the baseline AP loss to explicitly constrain the flux closure. The enhanced AP loss can still be evaluated dimension-wise. Numerical results demonstrate that the APTNN method significantly reduces training time and memory usage while improving prediction accuracy in the high-field regime. Therefore, the proposed method integrates accurate prediction in the macroscopic limit with eficient highdimensional computation, achieving large-scale kinetic simulations through economical GPU usage.

The proposed method merits investigation in several directions. First, further reducing memory consumption is important for extending APTNN to a wider range of physical problems. Kinetic solutions do not always admit low-rank approximations. For example, nonlinear plasma instabilities [14] and the transport of discontinuities [18] may require larger ranks to maintain accuracy. The coupling between the distribution function and the electric field introduces multiplicative rank dependence into the number of separated residual terms. As shown in Corollary 3.4, Gram-matrix storage for loss evaluation grows quadratically with this number, making memory a potential bottleneck at larger ranks. Promising ways to reduce memory usage include blockwise Gram-matrix evaluation [4] and recomputation during backpropagation [5]. Second, the quantitative relationship between the separation ranks and approximation errors of TNNs requires further investigation. The current approximation results guarantee arbitrary accuracy but do not quantify the ranks needed to achieve it. Developing rank-dependent error estimates would guide adaptive rank selection and help balance accuracy and computational cost.

## Appendix A Proof of Theorem 3.2

Proof. Let $\delta > 0$ . We first approximate the distribution f by a function with compact velocity support. Specifically, choose a cutof function $\chi \in C _ { c } ^ { \infty } (  { \mathbb { R } } )$ with $0 \leq \chi \leq 1 , \chi = 1 \mathrm { { o n } } \left[ - 1 , 1 \right]$ , and supp $\chi \subset [ - 2 , 2 ]$ . For $R \geq 1$ , define $\begin{array} { r } { \chi _ { R } ( v ) = \prod _ { j = 1 } ^ { 3 } \chi ( v _ { j } / R ) } \end{array}$ and $f _ { R } = \chi _ { R } f$ . Then $f _ { R } = f$ on $v \in [ - R , R ] ^ { 3 }$ and $f _ { R }$ vanishes outside $[ - 2 R , 2 R ] ^ { 3 }$ in velocity.

To show that $f _ { R }$ converges to $f$ in $H _ { s } ^ { m } ( \mathcal { D } )$ , we estimate the mixed derivatives of the truncation error $f - f _ { R } = ( 1 - \chi _ { R } ) f$ . For any $\alpha , \beta , \gamma$ satisfying $\alpha + | \beta | + | \gamma | \leq m$ , the Leibniz formula gives

$$
\partial _ { t } ^ { \alpha } \partial _ { x } ^ { \beta } \partial _ { v } ^ { \gamma } ( f - f _ { R } ) = ( 1 - \chi _ { R } ) \partial _ { t } ^ { \alpha } \partial _ { x } ^ { \beta } \partial _ { v } ^ { \gamma } f - \underset { \underset { \nu \neq 0 } { \nu \leq \gamma } } { \sum _ { \nu \leq \gamma } } { \binom { \gamma } { \nu } } ( \partial _ { v } ^ { \nu } \chi _ { R } ) \partial _ { t } ^ { \alpha } \partial _ { x } ^ { \beta } \partial _ { v } ^ { \gamma - \nu } f ,\tag{A.1}
$$

where $\nu \in \mathbb { N } _ { 0 } ^ { 3 }$ and $\nu \leq \gamma$ means $\nu _ { j } \leq \gamma _ { j }$ for $j = 1 , 2 , 3$ . In the second term, we use the chain rule to obtain

$$
\partial _ { v } ^ { \nu } \chi _ { R } ( v ) = R ^ { - | \nu | } \prod _ { j = 1 } ^ { 3 } \chi ^ { ( \nu _ { j } ) } ( v _ { j } / R ) , \mathrm { ~ a n d ~ } \| \partial _ { v } ^ { \nu } \chi _ { R } \| _ { L ^ { \infty } ( \mathbb { R } ^ { 3 } ) } \leq C _ { \nu } R ^ { - | \nu | } \leq C _ { \nu } ,
$$

where $\begin{array} { r } { C _ { \nu } = \prod _ { j = 1 } ^ { 3 } \| \chi ^ { ( \nu _ { j } ) } \| _ { L ^ { \infty } ( \mathbb { R } ) } } \end{array}$ . Then $\alpha + | \beta | + | \gamma - \nu | = \alpha + | \beta | + | \gamma | - | \nu | \leq m$ implies that only derivatives of $f$ up to order m are involved in the second term. Moreover, since $\chi _ { R } = 1$ on $( - R , R ) ^ { 3 }$ , both the coeficients $1 - \chi _ { R }$ and $\partial _ { v } ^ { \nu } \chi _ { R }$ for $\nu \neq 0$ vanish when $| v | < R$ . Therefore, using (A.1) and the Cauchy–Schwarz inequality, we obtain the estimate

$$
\| f - f _ { R } \| _ { H _ { s } ^ { m } ( \mathcal { D } ) } ^ { 2 } \leq C \sum _ { \substack { \alpha + | \beta | + | \gamma | \leq m } } \int _ { \mathcal { D } \cap \{ | v | \geq R \} } \langle v \rangle ^ { 2 s } \left| \partial _ { t } ^ { \alpha } \partial _ { x } ^ { \beta } \partial _ { v } ^ { \gamma } f \right| ^ { 2 } d t d x d v ,\tag{A.2}
$$

where $C$ is independent of R. Since $f \in H _ { s } ^ { m } ( \mathcal { D } )$ , the right-hand side of (A.2) tends to zero as $R \to \infty$ . Thus, we can choose R suficiently large such that

$$
\| f - f _ { R } \| _ { H _ { s } ^ { m } ( \mathcal { D } ) } < \frac { \delta } { 2 } .\tag{A.3}
$$

We next construct a TNN approximation of $f _ { R }$ on the whole velocity space, starting with an approximation on a bounded domain. Choose $V > 2 R$ and define the bounded domain $K _ { V } = ( 0 , T ) \times \Omega \times ( - V , V ) ^ { 3 }$ . Since $\langle v \rangle ^ { s } \geq 1$ , one has $f _ { R } \in H ^ { m } ( K _ { V } )$ with $\| f _ { R } \| _ { H ^ { m } ( K _ { V } ) } \leq$ $\| f _ { R } \| _ { H _ { s } ^ { m } ( \mathcal { D } ) } < \infty$ . By Theorem 3.1, there exists a TNN $U _ { \theta }$ that approximates $f _ { R }$ arbitrarily closely in $H ^ { m } ( K _ { V } )$ and has the form

$$
U _ { \theta } ( t , x , v ) = \sum _ { r = 1 } ^ { p } A _ { r , \theta } ( t ) \prod _ { i = 1 } ^ { d } B _ { r , i , \theta } ( x _ { i } ) \prod _ { j = 1 } ^ { 3 } D _ { r , j , \theta } ( v _ { j } ) ,\tag{A.4}
$$

where $A _ { r , \theta } , B _ { r , i , \theta }$ and $D _ { r , j , \theta }$ are the rth output components of the subnetworks for $t , x _ { i }$ and $v _ { j }$ respectively. Then, to obtain an approximation on the whole velocity space, we multiply the velocity factors $D _ { r , j , \theta }$ by a smooth cutof function and extend them by zero. Specifically, choose $\eta \in C _ { c } ^ { \infty } ( - V , V )$ with $\eta = 1$ on $[ - 2 R , 2 R ]$ and define $C _ { r , j , \theta } ( v _ { j } ) = \eta ( v _ { j } ) D _ { r , j , \theta } ( v _ { j } )$ . Each $C _ { r , j , \theta }$ is then extended by zero outside $( - V , V )$ . Replacing $D _ { r , j , \theta }$ with $C _ { r , j , \theta }$ in $U _ { \theta }$ , we obtain

$$
S _ { \theta } ( t , x , v ) = \sum _ { r = 1 } ^ { p } A _ { r , \theta } ( t ) \prod _ { i = 1 } ^ { d } B _ { r , i , \theta } ( x _ { i } ) \prod _ { j = 1 } ^ { 3 } C _ { r , j , \theta } ( v _ { j } ) .\tag{A.5}
$$

Both $f _ { R }$ and $S _ { \theta }$ vanish on $\mathcal { D } \setminus K _ { V }$ , so it sufices to estimate their diference on $K _ { V }$ . Since $\textstyle \prod _ { j = 1 } ^ { 3 } \eta ( v _ { j } ) = 1$ on $\operatorname { s u p p } _ { v } { f _ { R } }$ , we have

$$
f _ { R } - S _ { \theta } = \left( \prod _ { j = 1 } ^ { 3 } \eta ( v _ { j } ) \right) ( f _ { R } - U _ { \theta } ) \quad \mathrm { o n ~ } K _ { V } .\tag{A.6}
$$

Taking the $H ^ { m } ( K _ { V } )$ -norm in (A.6) and applying the Leibniz formula, we obtain

$$
\| f _ { R } - S _ { \theta } \| _ { H ^ { m } ( K _ { V } ) } \leq C _ { m , \eta } \| f _ { R } - U _ { \theta } \| _ { H ^ { m } ( K _ { V } ) } ,\tag{A.7}
$$

where $C _ { m , \eta }$ depends only on m and η. Together with the bound $\langle v \rangle ^ { s } \le ( 1 + 3 V ^ { 2 } ) ^ { s / 2 }$ on $( - V , V ) ^ { 3 }$ (A.7) yields

$$
\begin{array} { r l r } {  { \| f _ { R } - S _ { \theta } \| _ { H _ { s } ^ { m } ( \mathcal { D } ) } \leq ( 1 + 3 V ^ { 2 } ) ^ { s / 2 } \| f _ { R } - S _ { \theta } \| _ { H ^ { m } ( K _ { V } ) } } } \\ & { } & { \leq C _ { V , \eta } \| f _ { R } - U _ { \theta } \| _ { H ^ { m } ( K _ { V } ) } < \frac { \delta } { 2 } , ~ } \end{array}\tag{A.8}
$$

where $C _ { V , \eta } = ( 1 + 3 V ^ { 2 } ) ^ { s / 2 } C _ { m , \eta }$ . The last inequality follows by choosing $U _ { \theta }$ such that $\parallel f _ { R } -$ $U _ { \theta } \vert \vert _ { H ^ { m } ( K _ { V } ) } < \delta / ( 2 C _ { V , \eta } )$

Finally, combining (A.3) and (A.8), the triangle inequality gives

$$
\begin{array} { l } { \displaystyle \| f - S _ { \theta } \| _ { H _ { s } ^ { m } ( \mathcal D ) } \leq \| f - f _ { R } \| _ { H _ { s } ^ { m } ( \mathcal D ) } + \| f _ { R } - S _ { \theta } \| _ { H _ { s } ^ { m } ( \mathcal D ) } } \\ { \displaystyle < \frac \delta 2 + \frac \delta 2 = \delta . } \end{array}
$$

## Acknowledgments

SJ is partially supported by the NSFC grant No. 12031013, and the Shanghai Commission of S&T grant No. 23JC1402300. The computations in this paper were run on the Siyuan-1 cluster supported by the Center for High Performance Computing at Shanghai Jiao Tong University.

## References

[1] M. Bachmayr, Low-rank tensor methods for partial diferential equations, Acta Numerica, 32 (2023), pp. 1–121.

[2] M. Bai, J. Chen, and K. Wu, Uniformly accurate structure-preserving neural surrogates for radiative transfer. arXiv preprint arXiv:2511.04991, 2025.

[3] C. Cercignani, I. M. Gamba, and C. D. Levermore, High field approximations to a Boltzmann–Poisson system and boundary conditions in a semiconductor, Applied Mathematics Letters, 10 (1997), pp. 111–117.

[4] B. Charlier, J. Feydy, J. A. Glaunès, F.-D. Collin, and G. Durif, Kernel operations on the GPU, with autodif, without memory overflows, Journal of Machine Learning Research, 22 (2021), pp. 1–6.

[5] T. Chen, B. Xu, C. Zhang, and C. Guestrin, Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016.

[6] W. Chen, G. Dimarco, and L. Pareschi, Micro-macro tensor neural surrogates for uncertainty quantification in collisional plasma, Journal of Computational Physics, 565 (2026). Article 115197.

[7] J. Cho, S. Nam, H. Yang, S.-B. Yun, Y. Hong, and E. Park, Separable physicsinformed neural networks, in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 23761–23788.

[8] C. W. Clenshaw and A. R. Curtis, A method for numerical integration on an automatic computer, Numerische Mathematik, 2 (1960), pp. 197–205.

[9] J. Coughlin and J. Hu, Eficient dynamical low-rank approximation for the Vlasov-Ampère-Fokker-Planck system, Journal of Computational Physics, 470 (2022). Article 111590.

[10] J. M. Dawson, Particle simulation of plasmas, Reviews of Modern Physics, 55 (1983), pp. 403–447.

[11] A. Dektor, A. Rodgers, and D. Venturi, Rank-adaptive tensor methods for highdimensional nonlinear PDEs, Journal of Scientific Computing, 88 (2021). Article 36.

[12] G. Dimarco and L. Pareschi, Numerical methods for kinetic equations, Acta Numerica, 23 (2014), pp. 369–520.

[13] W. E and B. Yu, The Deep Ritz method: A deep learning-based numerical algorithm for solving variational problems, Communications in Mathematics and Statistics, 6 (2018), pp. 1–12.

[14] L. Einkemmer, K. Kormann, J. Kusch, R. G. McClarren, and J.-M. Qiu, A review of low-rank methods for time-dependent kinetic simulations, Journal of Computational Physics, 538 (2025). Article 114191.

[15] Y. Feng, M. K. Ng, K. Tang, and Z. Zhang, Functional tensor train neural network for solving high-dimensional PDEs. arXiv preprint arXiv:2510.13386, 2025.

[16] T. Goudon, J. Nieto, F. Poupaud, and J. Soler, Multidimensional high-field limit of the electrostatic Vlasov–Poisson–Fokker–Planck system, Journal of Diferential Equations, 213 (2005), pp. 418–442.

[17] A. Gu and X. Zhang, Asymptotic-preserving particle-in-cell method for the magnetized Vlasov–Poisson–Fokker–Planck equation, Journal of Scientific Computing, 108 (2026). Article 49.

[18] W. Guo and J.-M. Qiu, A low rank tensor representation of linear transport and nonlinear Vlasov solutions and their associated flow maps, Journal of Computational Physics, 458 (2022). Article 111089.

[19] J. Han, A. Jentzen, and W. E, Solving high-dimensional partial diferential equations using deep learning, Proceedings of the National Academy of Sciences, 115 (2018), pp. 8505– 8510.

[20] C. Jacoboni and L. Reggiani, The Monte Carlo method for the solution of charge transport in semiconductors with applications to covalent materials, Reviews of Modern Physics, 55 (1983), pp. 645–705.

[21] S. Jin, Z. Ma, and K. Wu, Asymptotic-preserving neural networks for multiscale timedependent linear transport equations, Journal of Scientific Computing, 94 (2023). Article 57.

[22] S. Jin, Z. Ma, and T.-a. Zhang, Asymptotic-preserving neural networks for multiscale Vlasov–Poisson–Fokker–Planck system in the high-field regime, Journal of Scientific Computing, 99 (2024). Article 61.

[23] S. Jin, Q. Shao, and H. Wang, Time-asymptotic behavior of the Boltzmann equation with random inputs in whole space and its stochastic Galerkin approximation. SIAM Journal on Mathematical Analysis, to appear.

[24] S. Jin and L. Wang, An asymptotic preserving scheme for the Vlasov–Poisson–Fokker– Planck system in the high field regime, Acta Mathematica Scientia, 31 (2011), pp. 2219–2232.

[25] , Asymptotic-preserving numerical schemes for the semiconductor Boltzmann equation eficient in the high field regime, SIAM Journal on Scientific Computing, 35 (2013), pp. B799– B819.

[26] T. G. Kolda and B. W. Bader, Tensor decompositions and applications, SIAM Review, 51 (2009), pp. 455–500.

[27] S. Li, Z. Wang, S. S.-T. Yau, and Z. Zhang, Solving nonlinear filtering problems using a tensor train decomposition method, IEEE Transactions on Automatic Control, 68 (2023), pp. 4405–4412.

[28] L. Lu, X. Meng, S. Cai, Z. Mao, S. Goswami, Z. Zhang, and G. E. Karniadakis, A comprehensive and fair comparison of two neural operators (with practical extensions) based on fair data, Computer Methods in Applied Mechanics and Engineering, 393 (2022). Article 114778.

[29] Y. Lu, L. Wang, and W. Xu, Solving multiscale steady radiative transfer equation using neural networks with uniform stability, Research in the Mathematical Sciences, 9 (2022). Article 45.

[30] L. Lyu, K. Wu, R. Du, and J. Chen, Enforcing exact boundary and initial conditions in the deep mixed residual method, CSIAM Transactions on Applied Mathematics, 2 (2021), pp. 748–775.

[31] J. Nieto, F. Poupaud, and J. Soler, High-field limit for the Vlasov–Poisson–Fokker– Planck system, Archive for Rational Mechanics and Analysis, 158 (2001), pp. 29–59.

[32] J. Oh, S. Y. Cho, S.-B. Yun, E. Park, and Y. Hong, Separable physics-informed neural networks for solving the BGK model of the Boltzmann equation, SIAM Journal on Scientific Computing, 47 (2025), pp. C451–C474.

[33] M. Raissi, P. Perdikaris, and G. E. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations, Journal of Computational Physics, 378 (2019), pp. 686–707.

[34] P. Ramachandran, B. Zoph, and Q. V. Le, Searching for activation functions, in International Conference on Learning Representations (ICLR), Workshop Track, 2018.

[35] S. A. Smolyak, Quadrature and interpolation formulas for tensor products of certain classes of functions, Soviet Mathematics Doklady, 4 (1963), pp. 240–243.

[36] S. Wang, Y. Teng, and P. Perdikaris, Understanding and mitigating gradient flow pathologies in physics-informed neural networks, SIAM Journal on Scientific Computing, 43 (2021), pp. A3055–A3081.

[37] Y. Wang, H. Xie, and P. Jin, Tensor neural network and its numerical integration, Journal of Computational Mathematics, 42 (2024), pp. 1714–1742.

[38] Z. Wang, J. Xin, and Z. Zhang, DeepParticle: Learning invariant measure by a deep neural network minimizing Wasserstein distance on data generated from an interacting particle method, Journal of Computational Physics, 464 (2022). Article 111309.

[39] S. Zhang and J. Hu, A separable and asymptotic-preserving dynamical low-rank method for the Vlasov–Poisson–Fokker–Planck system, ESAIM: Mathematical Modelling and Numerical Analysis, 60 (2026), pp. 1715–1739.

[40] T.-a. Zhang and S. Jin, AP-MIONet: Asymptotic-preserving multiple-input neural operators for capturing the high-field limits of collisional kinetic equations, Journal of Scientific Computing, 104 (2025). Article 17.