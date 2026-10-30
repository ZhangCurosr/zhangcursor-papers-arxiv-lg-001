# A physics-informed uncertainty-aware surrogate for p-n junction diodes

Suchita Kulkarni,a

eInstitute of Physics, NAWI Graz, University of Graz, Universitätsplatz 5, A-8010 Graz, Austria

E-mail: suchita.kulkarni@uni-graz.at

ABSTRACT: We develop an uncertainty-aware physics-informed surrogate for a onedimensional p-n junction diode, conditioned on device length, doping and bias. Using the internal state of the device as predicted by the surrogate we develop a test-suite of quantities of interest including the terminal current, depletion width, peak field and stored charge. The formalism and the quantities are can be extended to asymmetric junction and two-dimensional diodes as shown by means of benchmark analysis. For the one-dimensional symmetric junction surrogate, we obtain 0.11% median $L _ { 1 }$ error on the electrostatic potential, and 1.5% and 0.61% on the electron and hole densities. Finally, we develop an uncertainty analysis which determines the validity of surrogate answer within the userspecified tolerance.

## Contents

1 Introduction 2   
2 Drift-diffusion system for diodes 3   
2.1 Governing equations and parametrization 3   
2.2 Boundary conditions 4   
2.3 Terminal current 4   
2.4 Quantities of interest and scoring metrics 5   
3 Methods 7   
3.1 Dataset 7   
3.2 Network architecture 9   
3.3 Loss functions 11   
3.4 Training and evaluation procedure 12   
3.5 Uncertainty estimation framework 13   
4 Benchmark results across geometries 15   
4.1 One dimensional diodes 15   
4.2 Two-dimensional planar diode 16   
5 One-dimensional symmetric diode detailed results 17   
5.1 State and derived quantities 18   
5.2 Terminal current prediction 18   
5.3 Extrapolation performance 19   
6 Uncertainty quantification for one-dimensional symmetric diode 20   
7 Conclusions 25   
8 Acknowledgements 26   
A Drift-diffusion theory 28   
B Error distributions and seed variation 29   
C $\hat { J } _ { \mathrm { g r a d } }$ analysis 29   
D Ablation of the physics term 32

## 1 Introduction

Designing a semiconductor device means solving the coupled electrostatic and carriertransport equations for a given geometry, doping profile and set of boundary conditions This is routinely done with technology computer-aided design (TCAD) packages such as Synopsys Sentaurus [1], Silvaco Atlas [2] and COMSOL Multiphysics [3], which solve the underlying equations accurately for one fully specified device and bias. Exploring a design space, however, means many such solves, each started from scratch, over a large space of geometries, doping profiles and operating conditions.

A faster route to explore this space is useful if it reliably returns quantities of interest (QoI) to a device physicist such as the current, the stored charge, the electric field and the depletion width. Physics-informed neural networks [4] are a natural candidate for designing such a surrogate. A physics-informed network represents the internal state of the device, the potential and the carrier densities, and is trained to satisfy the governing equations as well as to match reference data. The QoIs are then derived from this one state, so a new QoI does not require a new model. Because the equations are enforced across the device and not only at labelled points, the network is also constrained between the devices it was trained on, and in principle beyond them. Physics can enter through the architecture, the loss, or the training data. In this work, we test whether and for which quantities, these motivations hold in practice.

Surrogates and physics-informed networks for TCAD have recently attracted considerable attention. Machine-learning approaches to device simulation fall into three groups. Terminal response surrogates map device and bias parameters directly to terminal characteristics such as I-V or C-V curves or figures of merit [5-7]. They are fast and accurate within their training range but carry no internal state, so every new quantity needs a new model. Physics-informed solvers represent the potential and carrier densities with a network trained on the drift-diffusion residual, with or without reference data [8-12]. Meshnative and operator surrogates learn the fields on the TCAD mesh or grid, through graph networks [13, 14| or neural operators [15], and reach realistic two- and three-dimensional geometries. Uncertainty enters these works as ensemble disagreement used to select new simulations [13] or as a model-based error bound for a single device [16]. The present work asks what a physics-informed surrogate conditioned over device properties can reliably deliver by utilising conformal methods to estimate uncertainty.

In this work, we thus intend to answer two related questions, first the capability of a physics-informed approach to deliver QoIs and second, the reliability of those answers at query time when the ground truth is unknown. We construct a physics-informed surrogate conditioned on device parameters, e.g., device length, doping and forward bias, and derive five QoIs – current, depletion region, electric field, and the stored charges for electrons and holes – from its predicted internal state. We demonstrate the generalisation of our formalism and architecture by evaluating single-device benchmarks of an asymmetric junction and a two-dimensional planar diode. The full study is carried out for a family of one-dimensional symmetric abrupt p-n junction diodes as the minimal complete showcase. We characterise QoIs against the reference simulation and attach an interval that tells a device physicist when the surrogate can be trusted.

We present three findings. First, within our architecture, the terminal current cannot be obtained by differentiating the network's internal state. The derivative must be inversely proportional to the number density across up to fourteen decades within a few Debye lengths wide junction layer. This presents a rather challenging situation for the network The differentiated current is wrong by about an order of magnitude in long, highly doped devices as the slope fails to follow the required steepness. It even has the wrong sign in two dimensions, although the internal states themselves are accurate. An integral of the potential, the classic Gummel relation [17, 18] recovers the current. Second, conditioning the error on the depleted fraction of the device yields intervals for all five QoIs whose coverage holds on held-out devices, whereas the spread of an ensemble of networks covers only about half of the errors. Third, extrapolation in bias degrades beyond the training range and fails beyond the built-in potential, where the device enters a regime absent from the training data.

The paper is organised as follows. Section 2 sets up the drift-diffusion system, the current readout and the QoIs. Section 3 sets up the methodology we use. Section 4 presents the three single-device benchmarks, which establish that the formalism is not tied to one geometry or doping. Sections 5 and 6 address this question for the symmetric family, through the accuracy of each QoI and its calibrated interval. Section 7 concludes.

## 2 Drift-diffusion system for diodes

## 2.1 Governing equations and parametrization

We consider a $p  – n$ junction diode occupying a domain $\Omega \subset \mathbb { R } ^ { d }$ with net doping $N _ { \mathrm { n e t } } ( \mathbf { x } ) \mathbf { \mu } ^ { 1 }$ acceptors $n _ { A }$ on the p-side and donors $n _ { D }$ on the n-side, and an applied bias V. Under the classical drift-diffusion approximation, such a system is characterized by the driftdiffusion equations which involve Poisson's equation and the electron and hole continuity equations, coupled through the carrier transport relations and a recombination model. Here we consider Shockley-Read-Hall (SRH) recombination. The equations are given in Appendix A. We use the well-known quasi-Fermi-level (QFL) parametrization, in which the unknowns are the electrostatic potential and the electron and hole quasi-Fermi potentials, and each carrier current is a single derivative of one of them rather than a balance of drift and diffusion terms that cancel to many orders of magnitude.

We non-dimensionalise the system, with lengths scaled by l, the longest side of the device $( \ell = L$ in one dimension), and denote the normalized quantities by a tilde $( \widetilde { \cdot } ) ^ { \mathrm { ~ 2 ~ } }$ . We follow most definitions from [11], and convert them to the QFL formalism where necessary. In Eq. (A.13) we provide the complete list used for non-dimensionalisation.

The non-dimensionalised Poisson's equation and the continuity equations become

$$
\lambda ^ { 2 } \tilde { \nabla } ^ { 2 } \tilde { \phi } = \tilde { n } - \tilde { p } - \tilde { N } _ { \mathrm { n e t } } , \qquad \tilde { \nabla } \cdot \tilde { \mathbf { J } } _ { n } = \tilde { R } , \qquad \tilde { \nabla } \cdot \tilde { \mathbf { J } } _ { p } = - \tilde { R } ,\tag{2.1}
$$

with the dimensionless potentials $\tilde { \phi } ~ = ~ \psi / U _ { t }$ and $\tilde { v } _ { n , p } ~ = ~ \phi _ { n , p } / U _ { t }$ , where $\psi$ is the electrostatic potential and $\phi _ { n , p }$ are the quasi-Fermi potentials in volts, $\tilde { n } = \tilde { n } _ { i } \exp \bigl ( \tilde { \phi } - \tilde { v } _ { n } \bigr )$ $\tilde { p } = \tilde { n } _ { i } \exp \bigl ( \tilde { v } _ { p } - \tilde { \phi } \bigr )$ and $\tilde { \mathbf { J } } _ { n } = - \tilde { \mu } _ { n } \tilde { n } \tilde { \nabla } \tilde { v } _ { n } , \tilde { \mathbf { J } } _ { p } = - \tilde { \mu } _ { p } \tilde { p } \tilde { \nabla } \tilde { v } _ { p }$ , with,

$$
\tilde { N } _ { \mathrm { n e t } } ( \chi ) = \frac { N _ { \mathrm { n e t } } ( \mathbf { x } ) } { n _ { \mathrm { m a x } } } \longrightarrow - \frac { n _ { A } } { n _ { \mathrm { m a x } } } ( p \mathrm { - b u l k } ) , \qquad + \frac { n _ { D } } { n _ { \mathrm { m a x } } } ( n \mathrm { - b u l k } ) ,\tag{2.2}
$$

whose larger magnitude is 1 by construction and whose smaller magnitude is 10-lrl with $r = \log _ { 1 0 } ( n _ { A } / n _ { D } )$

## 2.2 Boundary conditions

The boundary ∂Ω consists of ideal ohmic contacts $\Gamma _ { c }$ and, for $d > 1$ , of insulating segments. On contact $^ { c , }$ held at the bias $V _ { c } \ ( V _ { }$ on the anode, 0 on the cathode), charge neutrality and equal quasi-Fermi levels give

$$
\tilde { v } _ { n } = \tilde { v } _ { p } = \frac { V _ { c } } { U _ { t } } , \qquad \tilde { \phi } = \frac { V _ { c } } { U _ { t } } + \mathrm { a s i n h } \frac { \tilde { N } _ { c } } { 2 \tilde { n } _ { i } } \qquad \mathrm { o n } ~ \Gamma _ { c } ,\tag{2.3}
$$

with $\tilde { N } _ { c }$ the net doping at the contact. On insulating segments

$$
\partial _ { \nu } \tilde { \phi } = 0 , \qquad \tilde { \bf J } _ { n } { \cdot } \nu = \tilde { \bf J } _ { p } { \cdot } \nu = 0 .\tag{2.4}
$$

The quasi-Fermi boundary values are independent of the doping, which enters the boundary data only through $\tilde { \phi } .$ The built-in potential of a junction is $V _ { \mathrm { b i } } = U _ { t } \ln \left( n _ { A } n _ { D } / n _ { i } ^ { 2 } \right)$ , and the potential drop across it at bias V is $V _ { \mathrm { b i } } - V$

## 2.3 Terminal current

The quantity read off a device is the terminal current, the flux of current through the biased contact $\Gamma _ { a }$

$$
J = J _ { \mathrm { r e f } } \ell ^ { d - 1 } \int _ { \Gamma _ { a } } \big ( \tilde { \mathbf { J } } _ { n } + \tilde { \mathbf { J } } _ { p } \big ) { \cdot } { \pmb { \nu } } d \tilde { \Gamma } ,\tag{2.5}
$$

with ν the inward unit normal. $\Gamma _ { a }$ is a point in one dimension and a line segment in two, so J is a current density in 1D and a current per unit depth in 2D. Therefore the current has unit of $\mathrm { A / c m ^ { 2 } }$ in 1D and A/cm in 2D.

Setting $U _ { n } = e ^ { - \tilde { v } _ { n } }$ and $U _ { p } = e ^ { \tilde { v } _ { p } }$ (not to be confused with the thermal voltage $U _ { t } )$ , the currents become

$$
{ \tilde { \mathbf { J } } } _ { n } = \tilde { \mu } _ { n } \tilde { n } _ { i } e ^ { \tilde { \phi } } \tilde { \nabla } U _ { n } , \qquad { \tilde { \mathbf { J } } } _ { p } = - \tilde { \mu } _ { p } \tilde { n } _ { i } e ^ { - \tilde { \phi } } \tilde { \nabla } U _ { p } ,\tag{2.6}
$$

and the continuity equations of Eq. (2.1) become, for both carriers,

$$
\tilde { \nabla } \cdot \bigl ( a \tilde { \nabla } U \bigr ) = \tilde { R } ,\tag{2.7}
$$

with $U _ { n , p }$ fixed on the contacts and $\partial _ { \nu } U = 0$ on insulating segments. For given $\tilde { \phi }$ and $\tilde { R }$ this is linear in $U ,$ so no derivative of the network output enters. Here $\tilde { \phi }$ and $\tilde { R }$ are the network's, and the end values of $U _ { n , p }$ are the network's values on the boundary of the modelled domain, which excludes a thin layer at each contact (Sec. 3.1). The readout current is Eq. (2.5) evaluated with the J of this solution.

In one dimension Eq. (2.7) integrates in closed form. The currents are $\tilde { J } _ { n } ( \chi ) ~ =$ $\tilde { J } _ { n 0 } + C ( \chi )$ and $\tilde { J } _ { p } ( \chi ) = \tilde { J } _ { p 0 } - C ( \chi )$ with $\begin{array} { r } { C ( \chi ) = \int _ { \chi _ { - } } ^ { \chi } \tilde { R } d \chi ^ { \prime } } \end{array}$ , and integrating $d U _ { n } / d \chi =$ ${ \tilde { J } _ { n } e ^ { - \tilde { \phi } } / ( \tilde { \mu } _ { n } \tilde { n } i ) }$ and $d U _ { p } / d \chi = - \tilde { J } _ { p } e ^ { \phi } / ( \tilde { \mu } _ { p } \tilde { n } _ { i } )$ across the modelled domain $[ \chi _ { - } , \chi _ { + } ]$ gives the currents at its first end.

$$
\tilde { J } _ { n 0 } = \frac { U _ { n } ( \chi _ { + } ) - U _ { n } ( \chi _ { - } ) - \int _ { \chi _ { - } } ^ { \chi _ { + } } C e ^ { - \tilde { \phi } } / ( \tilde { \mu } _ { n } \tilde { n } _ { i } ) d \chi } { \int _ { \chi _ { - } } ^ { \chi _ { + } } e ^ { - \tilde { \phi } } / ( \tilde { \mu } _ { n } \tilde { n } _ { i } ) d \chi } ,\tag{2.8}
$$

$$
\tilde { J } _ { p 0 } = - \frac { U _ { p } ( \chi _ { + } ) - U _ { p } ( \chi _ { - } ) - \int _ { \chi _ { - } } ^ { \chi _ { + } } C e ^ { \tilde { \phi } } / ( \tilde { \mu } _ { p } \tilde { n } _ { i } ) d \chi } { \int _ { \chi _ { - } } ^ { \chi _ { + } } e ^ { \tilde { \phi } } / ( \tilde { \mu } _ { p } \tilde { n } _ { i } ) d \chi } .\tag{2.9}
$$

The total $\tilde { J } _ { n 0 } + \tilde { J } _ { p 0 }$ is the terminal current, constant in $\chi$ because the recombination integrals cancel. For $d > 1$ we solve Eq. (2.7) by cell-centred finite volumes on a uniform grid of the modelled domain. Insulating faces carry no flux, which imposes $\partial _ { \nu } U = 0$ without further input. This is a post-processing step at evaluation time that takes the place of the 1D quadrature.

Eq. (2.9) involves only $\tilde { \phi } , \tilde { R } , \tilde { \mu } _ { n , p } , \tilde { n } _ { i }$ and the end values of $U _ { n , p } ,$ and the quasi-Fermi boundary values of Eq. (2.3) do not depend on $n _ { A }$ or $n _ { D }$ . The formulation therefore holds unchanged for $n _ { A } \neq n _ { D }$ and in any dimension.

## 2.4 Quantities of interest and scoring metrics

Given the network output, we derive five scalar QoIs, the terminal current $\hat { J } _ { \mathrm { R O } }$ , the depletion width $W$ , the peak field $\mathcal { E } _ { \mathrm { p e a k } }$ and the stored charges $Q _ { n } , Q _ { p }$ . They are the quantities for which Sec. 6 constructs an uncertainty. Additionally, we also consider $\psi , n , p ,$ R and $\Delta E _ { F }$ as fields over Ω. Finally, as said before we compute $\hat { J } _ { \mathrm { g r a d } }$ , but do not actively use it in our analysis. Below we define each of the QoI that we use.

1. Terminal current. We define two estimates. The differentiated current

$$
J _ { \mathrm { g r a d } } = J _ { \mathrm { r e f } } \ell ^ { d - 1 } \int _ { \Gamma _ { a } } \left( - \tilde { \mu } _ { n } \tilde { n } \tilde { \nabla } \tilde { v } _ { n } - \tilde { \mu } _ { p } \tilde { p } \tilde { \nabla } \tilde { v } _ { p } \right) \cdot \nu d \tilde { \Gamma } ,\tag{2.10}
$$

with $\tilde { n }$ and $\tilde { p }$ as defined above, evaluates the drift-diffusion flux; on the network output the gradients come from automatic differentiation. The readout current

$$
J _ { \mathrm { R O } } = J _ { \mathrm { r e f } } \ell ^ { d - 1 } \int _ { \Gamma _ { a } } \left( \tilde { \mu } _ { n } \tilde { n } _ { i } e ^ { \tilde { \phi } } \tilde { \nabla } U _ { n } - \tilde { \mu } _ { p } \tilde { n } _ { i } e ^ { - \tilde { \phi } } \tilde { \nabla } U _ { p } \right) \cdot \nu \ d \tilde { \Gamma } ,\tag{2.11}
$$

evaluates the same integral with the Slotboom fluxes from Eq. (2.7), which in one dimension reduces to the closed form of Eqs. (2.7) and (2.9). Throughout this work, we use $\hat { J } _ { \mathrm { R O } }$ as the terminal-current predicted by the network as QoI. We also evaluate $\hat { J } _ { \mathrm { g r a d } }$ and compare it with $\hat { J } _ { \mathrm { R O } }$ in App. C.

2. Peak electric field helps quantify the maximum local electrostatic stress and assess field redistribution with applied bias. It is defined as

$$
\pmb { \mathcal { E } } ( \mathbf { x } ) = - \frac { U _ { t } } { \ell } \tilde { \nabla } \tilde { \phi } , \qquad \pmb { \mathcal { E } } _ { \mathrm { p e a k } } = \operatorname* { m a x } _ { \mathbf { x } \in \Omega } | \pmb { \mathcal { E } } ( \mathbf { x } ) | .\tag{2.12}
$$

3. The depletion width sets the junction capacitance, $C _ { j } \approx \varepsilon / W$ per unit area in the depletion approximation. A real doping profile has no sharp edge, therefore we define the depletion region as the set D where |ε| exceeds 1% of its own peak, and its size $W = | D |$ (the width in one dimension, the area in two),

$$
D = \big \{ \mathbf x \in \Omega : ~ | \pmb { \mathcal E } ( \mathbf x ) | \ge 0 . 0 1 \mathcal E _ { \mathrm { p e a k } } \big \} , \qquad W = | D | .\tag{2.13}
$$

The threshold is a modelling choice. For orientation, the abrupt-junction depletion approximation (Eq. (A.8)) places most of the depletion region on the lighter-doped side.

4. The stored charges $Q _ { n } , Q _ { p }$ quantify electron, hole injection and charge storage, which influence transient response and switching behaviour. The stored charges are the carriers in excess of their equilibrium minority densities, summed over the quasineutral part of each side, that is, the part of the doped region that is not depleted. We denote these regions by

$$
\begin{array} { r } { \Omega _ { p } ^ { \mathrm { q n } } = \big \{ \mathbf x \in \Omega : \ N _ { \mathrm { n e t } } ( \mathbf x ) < 0 , \ \mathbf x \not \in D \big \} , \qquad \Omega _ { n } ^ { \mathrm { q n } } = \big \{ \mathbf x \in \Omega : \ N _ { \mathrm { n e t } } ( \mathbf x ) > 0 , \ \mathbf x \not \in D \big \} , } \end{array}\tag{2.14}
$$

with D from Eq. (2.13), and write

$$
Q _ { n , \mathrm { s t o r e d } } = q \int _ { \Omega _ { p } ^ { \mathrm { q n } } } \operatorname* { m a x } \left( n - n _ { p 0 } , \ 0 \right) d \Omega ,\tag{2.15}
$$

$$
Q _ { p , \mathrm { s t o r e d } } = q \int _ { \Omega _ { n } ^ { \mathrm { q n } } } \operatorname* { m a x } \left( p - p _ { n 0 } , \ 0 \right) d \Omega .\tag{2.16}
$$

Here $n _ { p 0 } = n _ { i } ^ { 2 } / n _ { A }$ and $p _ { n 0 } = n _ { i } ^ { 2 } / n _ { D }$ are the equilibrium minority densities, with $n _ { A }$ and $n _ { D }$ the acceptor and donor doping of the respective plateau $( n _ { p 0 } = p _ { n 0 }$ for a symmetric device). The measure dΩ is a length in one dimension and an area in two, SO $Q$ is a charge per unit area in 1D and per unit depth in 2D. The depletion region of Eq. (2.13) is computed for ground truth and network predictions independently.

5. Finally, we define the quasi-Fermi-level splitting $\Delta E _ { F } ( { \bf x } )$ . It is the local separation of the two quasi-Fermi potentials

$$
\Delta E _ { F } ( \mathbf { x } ) = U _ { t } \mathrm { l n } \left( \frac { n ( \mathbf { x } ) p ( \mathbf { x } ) } { n _ { i } ^ { 2 } } \right) = U _ { t } \big ( \tilde { v } _ { p } ( \boldsymbol { x } ) - \tilde { v } _ { n } ( \boldsymbol { x } ) \big ) .\tag{2.17}
$$

We therefore have five scalar quantities, and we need to establish the accuracy of network prediction. A scalar QoI $Q$ is scored at each (device, voltage) point by the relative error of the surrogate answer $\hat { Q }$ against ground truth,

$$
e = \left| \frac { \hat { Q } } { Q } - 1 \right| ,\tag{2.18}
$$

and a profile by its relative $L _ { 1 }$ error $\| { \hat { q } } - q \| _ { 1 } / \| q \| _ { 1 }$ over Ω. Over the surrogate dimensions we consider, not all quantities may always be measured, $\mathrm { e . g . }$ , depletion width is not defined for a fully depleted device, or an extremely tiny current offers no practical use. We therefore compute the relative error only where the corresponding QoI or profile is defined. The definition uses only the network's own output, and the current is measurable if the predicted current satisies $| \hat { J } _ { \mathrm { R O } } | \geq 1 0 ^ { - 6 } \mathrm { A / c m ^ { 2 } }$ , the depletion width is informative if the network's W is below 95% of the domain, and a stored charge is informative if the network's charge is positive. Finally, each quantity is evaluated on the domain it is defined on, which means the ground truth is evaluated over the entire device dimensions and for reasons explained below the network predictions are evaluated over a slightly smaller range.

## 3 Methods

## 3.1 Dataset

Ground truth is generated with the open-source TCAD solver DEVSIM 3 [19], solving the steady-state drift-diffusion system for a silicon p-n diode at T = 300 K $( U _ { t } = 2 5 . 8 9 \mathrm { m V }$ 2 $n _ { i } = 1 0 ^ { 1 0 } \mathrm { c m } ^ { - 3 } , \varepsilon = 1 1 . 7 \varepsilon _ { 0 } )$ with constant mobilities $\mu _ { n } = 1 4 1 7$ and $\mu _ { p } = 4 7 0 . 5 \mathrm { c m ^ { 2 } / V }$ s and equal lifetimes $\tau _ { n } = \tau _ { p } = 1 0 ^ { - 5 } \mathrm { s } \ [ 1 1 ]$ . The doping is smeared by a Gaussian kernel with standard deviation $w / { \sqrt { 2 } }$ with $w = 0 . 0 2 L$

$$
N _ { \mathrm { n e t } } ( x ) = \frac { n _ { D } - n _ { A } } { 2 } + \frac { n _ { D } + n _ { A } } { 2 } \mathrm { e r f } \bigg ( \frac { x - x _ { j } } { w } \bigg ) ,\tag{3.1}
$$

which tends to $- n _ { A }$ on the p-side and $+ n _ { D }$ on the n-side.

The profiles are interpolated by cubic splines in (φ, ln n, ln p) in one dimension and by a $C ^ { 1 }$ Clough-Tocher interpolant on the mesh nodes in two and converted to $( \tilde { v } _ { n } , \tilde { v } _ { p } )$ via the Boltzmann relations of Sec. 2.1. We observe that DEVSIM forces exact equilibrium on the contact node itself, which does not join smoothly onto the injected minority-carrier profile one cell inside, leading to a particularly fast-varying profile near contacts.

This fast varying profile is a particularly challenging feature of the dataset and the underlying physics. As it was preempted in sec. 2.3, we discuss our approach to address this challenge. At each contact the minority density rises sharply from its equilibrium value, so that the quasi-Fermi level of the minority carrier leaves its contact value within a few nanometres. This sharp change within a few nanometers of contact is very challenging to be reproduced by the network. We explicitly quantify the error introduced due to this trim within our analysis.

We therefore trim 1 nm of data for 1D devices and 1.4 nm of data for 2D devices. The trim makes the device 2 nm, 2.8 nm shorter in 1D, 2D, so a quantity computed on the trimmed window differs from the same quantity computed on the full device even for a perfect network. We measure the error introduced by this approximation for the 1D symmetric devices by computing the QoIs from DEVSIM's own solution twice, once on the full length L and once on [1 nm, L-1 nm]. The QoIs lose at most 1% at the 95th percentile. The largest losses are 1.5% for W, 1.4% for $Q _ { p } , 0 . 2 5 \%$ for the readout current, 0.0045% for the peak field and 4.2% for $Q _ { n }$ . They are concentrated in the devices shorter than 0.4 µm, where the 2 nm trim is up to 2% of the length. In the longer devices the largest losses are less than percent.

Our predicted quantities $\hat { Q }$ are therefore always computed on the trimmed domain, while the ground truth is always computed on the full device. The errors therefore also include any effect of the trim.

## 3.1.1 Benchmark data sets

![](images/f754e8bf0d9000f26f3bfa52cfc0cc919ba946b245b73805a20665ae5346fe2a.jpg)  
Figure 1. The 2D planar benchmark device (contact thickness exaggerated): $\mathrm { ~ a ~ p ~ } ^ { + }$ implant $( n _ { A } =$ $1 0 ^ { 1 8 } \mathrm { c m ^ { - 3 } }$ , 1.26 µm wide, $0 . 6 4 \mu \mathrm { m }$ deep) in an $\mathrm { n ^ { - } }$ substrate $( n _ { D } = 1 0 ^ { 1 6 } \thinspace \mathrm { c m ^ { - 3 } } )$ of width $3 \mu \mathrm { m }$ and height $2 \mu \mathrm { m }$ . The anode is a window on the top face, the cathode covers the bottom face, and the remaining boundary is insulating.

Besides the one-dimensional symmetric data set of Sec. 3.1, on which the detailed study of Secs. 5 and 6 rests, we show the framework applicability on three single-device benchmarks, where each device is solved for 16 forward voltages. Twelve are used for training jointly across all voltages and four are held-out for evaluation. Both 1D devices have $L = 1 . 8 0 \mu \mathrm { m } \mathrm { ; }$ the symmetric device has $n _ { A } = n _ { D } = 2 . 9 1 { \times } 1 0 ^ { 1 7 } \mathrm { c m } ^ { - 3 }$ , and the asymmetric device has $n _ { A } = 6 \times 1 0 ^ { 1 7 } \mathrm { c m ^ { - 3 } }$ and $n _ { D } = 6 \times 1 0 ^ { 1 6 }$ cm $^ { - 3 } \ ( n _ { A } / n _ { D } = 1 0 )$ . For 2D device we set $n _ { D } = 1 0 ^ { 1 6 } \mathrm { { c m } ^ { - 3 } }$ with the detailed geometry given in fig. 1.

For the 2D benchmark a quarter of the collocation and data points are drawn near the metallurgical junction. As in 1D, the modelled domain excludes the first two mesh rows for 2D devices. The benchmarks establish that the formalism and the readout are generic. A single-device fit, however, says nothing about devices the network has not seen. It requires a device family, held-out devices and calibration data, which we construct for the symmetric 1D diode in Secs. 5–6 .

## 3.1.2 1D symmetric surrogate dataset

For a 1D surrogate construction, we consider a family of devices which vary in length and doping concentration. Devices are drawn over the range of $L \in [ 1 0 ^ { - 5 } , 7 \times 1 0 ^ { - 4 } ]$ cm, $n _ { D } \in [ 1 0 ^ { 1 5 } , 1 0 ^ { 1 9 } ] \mathrm { c m ^ { - 3 ~ 4 } }$ . The upper bound on length is the largest length at which DEVSIM converged reliably under bias. Each device is solved at the forward voltages 0.04 to $V / V _ { b i } <$ 0.8 (with $V _ { b i }$ as in Sec. 2.2) in steps of 0.04 V and up to a maximum of 0.64 V, so devices with $n _ { D } \gtrsim 5 \times 1 0 ^ { 1 6 } \mathrm { c m ^ { - 3 } }$ reach only $0 . 6 \substack { - 0 . 8 V _ { b i } }$ . We dub one device-voltage pair as one anchor. We generate three datasets with each serving a specific role defined in Table 1.

<table><tr><td>Set</td><td>Role</td><td>Devices (anchors)</td></tr><tr><td>Training</td><td>fit weights</td><td>27 (404)</td></tr><tr><td>Held-out</td><td>validation</td><td>3 (45)</td></tr><tr><td>Calibration</td><td>analysis, intervals</td><td>310 (4586)</td></tr><tr><td>Test</td><td>all reported claims</td><td>246 (3647)</td></tr><tr><td>Extrapolation</td><td>V extrapolation</td><td>12 (→)</td></tr></table>

Table 1. Device sets and their roles. Anchors are counted after restricting to $V / V _ { b i } < 0 . 8$ and, for the calibration and test sets, to the supported range $\kappa \geq - 2 . 5$ of Sec. 3.5.

The first dataset consists of the 30 devices or 449 anchors with $V / V _ { b i } < 0 . 8$ , drawn using Latin hypercube sampling over $\log _ { 1 0 } ( n _ { D } ) , \log _ { 1 0 } ( L )$ . Three whole devices are withheld at the training time. A separate set of 310 devices is used to develop the analysis and to calibrate the error intervals of Sec. 6. The test set consists of 250 devices generated with Latin hypercube sampling in $\log _ { 1 0 } ( n _ { D } ) , \log _ { 1 0 } ( L )$ to evaluate final network performance. It was drawn, solved and evaluated after the networks, the calibration set, the error bars and the analysis were frozen, and it was scored once. Two of the 250 devices did not converge at the first voltage and are not part of the set. The remaining 248 devices are restricted as described above, which leaves 246. Finally, to test extrapolation, we generated additional 12 devices beyond the training range.

With $L , n _ { D }$ spanning about (1.8, 4) decades, the sampled devices cover $L / \lambda _ { D }$ from 1.5 to 3500 (the corners of the box $( L , n _ { D } )$ would give 0.8 to 5400). This includes devices only a Debye length or two long, in which the depletion region occupies the whole device, to long devices with a quasi-neutral bulk. Whether a device counts as fully depleted depends on how the depletion edge is defined, so we report results by the continuous depletion fraction $W _ { \mathrm { d e p } } / L$ rather than by a binary label.

As we set $w = 0 . 0 2 L$ , the doping transition is always resolved on the same fraction of the length, but junction smearing width does not scale with the depletion width. At $n _ { D } =$ $1 0 ^ { 1 9 } \mathrm { c m ^ { - 3 } }$ the zero-bias abrupt-junction depletion width is ≈ 17 nm, whereas w = 140 nm at the upper limit of the length range. This shows the variety of devices we consider in our surrogate.

## 3.2 Network architecture

Our architecture and training methodology is inspired by [11]. Three separate networks, φ-net, n-net and p-net, produce $\tilde { \phi } , \tilde { v } _ { n }$ and $\tilde { v } _ { p }$ at a point $\chi = ( { \bf x } - { \bf x } _ { c } ) / \ell$ of a device of dimension $d ,$ where $\mathbf { x } _ { c }$ is the centre of the device and l its longest side, so that $\ell = L$ and $\chi = x / L - 1 / 2 \in [ - 1 / 2 , 1 / 2 ]$ in one dimension. The full network architecture including the FiLM conditioning described below is shown in fig. 2. Each is a tanh multilayer perceptron with five hidden layers of 128 units and a linear read-out. The spatial input has 2d components, χ and ${ \pmb \xi } = \chi / \lambda$ with $\lambda = \lambda _ { D } / \ell$ . We found ξ to be a very useful variable because it lets the network see the position in units of the Debye length, which changes by orders of magnitude across the grid.

![](images/146ccf72fefdab2a6f841a1365886cf19ed49f1eea18e0f01a9b20506e303637.jpg)  
Figure 2. Network architecture representation

The device and bias enter through feature-wise linear modulation of every hidden layer. For each of the three networks, a small hypernetwork with one hidden layer of 32 tanh units maps the conditioning vector c to a scale $\gamma _ { l } = 1 + \Delta \gamma _ { l }$ and a shift $\beta _ { l }$ for each of the five hidden layers (width 128), and the layer computes

$$
h _ { l } = \operatorname { t a n h } \bigl ( \gamma _ { l } \odot \bigl ( W _ { l } h _ { l - 1 } + b _ { l } \bigr ) + \beta _ { l } \bigr ) , \qquad h _ { 0 } = \bigl ( \boldsymbol { \chi } , \boldsymbol { \xi } \bigr ) .\tag{3.2}
$$

The last hypernetwork layer is initialised to zero, so training starts from three unconditioned networks $( \gamma = 1 , \beta = 0 )$ . The conditioning vector is

$$
c = \Big ( \log _ { 1 0 } ( \sqrt { n _ { A } n _ { D } } / n _ { D } ^ { \mathrm { r e f } } ) , ~ \log _ { 1 0 } ( \ell / L ^ { \mathrm { r e f } } ) , ~ \log _ { 1 0 } ( \mu _ { p } / \mu _ { n } ) , ~ V / U _ { t } - V _ { c } , ~ s , ~ r , ~ { \bf g } \Big ) ,
$$

with $r = \log _ { 1 0 } ( n _ { A } / n _ { D } )$ , and for $d = 2$ , five geometry descriptors g. These are the log aspect ratio $\log _ { 1 0 } ( L _ { x } / L _ { y } )$ of the domain, the fraction of its face covered by the anode and by the cathode contact, and the two components of the offset of the metallurgical-junction centroid from the domain centre in units of l. In one dimension $\mathbf { g }$ is absent.

Among the scalar components, $n _ { D } ^ { \mathrm { r e f } } = 1 0 ^ { 1 5 } \mathrm { c m } ^ { - 3 } , L ^ { \mathrm { r e f } } = 3 . 9 7 \times 1 0 ^ { - 4 } \mathrm { c m } , s = \pm 1$ marks which side of the device is donor-doped, and r is the doping asymmetry. The mobility ratio and s are constant in this work, and $r = 0$ in the symmetric data of Secs. 5.1–5.3, whose networks were trained without this column and therefore have five hypernetwork inputs. The column $V / U _ { t } - V _ { c }$ subtracts a fixed $V _ { c ; }$ which represents a mean over applied bias $V / U _ { t }$

The final simplification concerns the offset of the hole quasi-Fermi level. The quasi-Fermi level $\tilde { v } _ { n } , \tilde { v } _ { p }$ profiles behave such that $\tilde { v } _ { n } = 0 , \tilde { v } _ { p } = V / U _ { t }$ for most of the device, while they change to $\tilde { v } _ { n } = V / U _ { t } , \tilde { v } _ { p } = 0$ only at the contacts, where they change rapidly. The network therefore must predict a large, nearly constant offset at every point for $\tilde { v } _ { p } ,$ on top of a spatial variation of only a few units. Secondly, our 1nm trim removes the layer in which $\tilde { v } _ { p }$ falls to zero. Therefore, instead of forcing the network to reconstruct this level, the p-network returns $f _ { p } ( { \boldsymbol { x } } , { \boldsymbol { \xi } } ; c )$ and the model sets $\tilde { v } _ { p } ( \boldsymbol { \chi } ) = f _ { p } ( \boldsymbol { \chi } , \pmb { \xi } ; c ) + V / U _ { t }$ . After the shift, $f _ { p }$ and $\tilde { v } _ { n }$ have the same range and nearly the same shape up to a sign, each with a flat plateau at zero.

## 3.3 Loss functions

We reconstruct carrier densities from the outputs as

$$
\tilde { n } = \tilde { n } _ { i } \exp \bigl ( \mathrm { c l i p } ( \tilde { \phi } - \tilde { v } _ { n } , \pm 4 0 ) \bigr ) , \qquad \tilde { p } = \tilde { n } _ { i } \exp \bigl ( \mathrm { c l i p } ( \tilde { v } _ { p } - \tilde { \phi } , \pm 4 0 ) \bigr ) ,\tag{3.3}
$$

(the clip at $\pm 4 0$ is a float32 overflow guard). Using definitions in Eq. (2.1) and Eq. (A.5), the pointwise residuals are

$$
\begin{array} { r l } & { { \boldsymbol r } _ { P } = \lambda ^ { 2 } \tilde { \nabla } ^ { 2 } \tilde { \phi } - \big ( \tilde { n } - \tilde { p } - \tilde { N } _ { \mathrm { n e t } } \big ) , } \\ & { { \boldsymbol r } _ { n } = \displaystyle \frac { 1 } { \tilde { \mu } _ { n } } \left( \tilde { \nabla } \cdot \tilde { \mathbf J } _ { n } - \tilde { R } \right) , \qquad { \boldsymbol r } _ { p } = \displaystyle \frac { 1 } { \tilde { \mu } _ { p } } \left( \tilde { \nabla } \cdot \tilde { \mathbf J } _ { p } + \tilde { R } \right) , } \end{array}\tag{3.4}
$$

where $\tilde { N } _ { \mathrm { n e t } }$ is the analytic smeared doping above.

For a minibatch B of anchors, with $N = 2 5 6$ collocation points $\chi _ { a , i }$ per anchor, the terms are mean-squared values over all points of the minibatch

$$
\mathcal { L } _ { \mathrm { P o i s s o n } } = \big \langle r _ { P } ^ { 2 } \big \rangle ,\tag{3.5}
$$

$$
\mathcal { L } _ { \mathrm { c o n t } } = \big \langle r _ { n } ^ { 2 } \big \rangle + \big \langle r _ { p } ^ { 2 } \big \rangle ,\tag{3.6}
$$

$$
\mathcal { L } _ { \mathrm { d a t a } } = \big \langle ( \tilde { \phi } - \tilde { \phi } ^ { D } ) ^ { 2 } \big \rangle + \big \langle ( \tilde { v } _ { n } - \tilde { v } _ { n } ^ { D } ) ^ { 2 } \big \rangle + \big \langle ( \tilde { v } _ { p } - \tilde { v } _ { p } ^ { D } ) ^ { 2 } \big \rangle ,\tag{3.7}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D } } = \big \langle \big ( \tilde { \phi } - \tilde { \phi } ^ { D } \big ) ^ { 2 } \big \rangle _ { D } + \big \langle \big ( \tilde { v } _ { n } - \tilde { v } _ { n } ^ { D } \big ) ^ { 2 } \big \rangle _ { D } + \big \langle \big ( \tilde { v } _ { p } - \tilde { v } _ { p } ^ { D } \big ) ^ { 2 } \big \rangle _ { D } , } \end{array}\tag{3.8}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { N } } = \langle ( \partial _ { \nu } \tilde { \phi } ) ^ { 2 } \rangle + \langle ( \tilde { \bf J } _ { n } { \cdot } \nu ) ^ { 2 } \rangle + \langle ( \tilde { \bf J } _ { p } { \cdot } \nu ) ^ { 2 } \rangle , } \end{array}\tag{3.9}
$$

where D denotes the DEVSIM value, (·〉 averages over the collocation points and $D$ $\langle \cdot \rangle _ { D }$ over the contact points of each anchor. The total loss is

$$
\mathcal { L } = \lambda _ { \mathrm { d a t a } } \mathcal { L } _ { \mathrm { d a t a } } + \lambda _ { \mathrm { D } } \mathcal { L } _ { \mathrm { D } } + \lambda _ { \mathrm { N } } \mathcal { L } _ { \mathrm { N } } + \lambda _ { \mathrm { P o i s s o n } } \mathcal { L } _ { \mathrm { P o i s s o n } } + \lambda _ { \mathrm { c o n t } } \mathcal { L } _ { \mathrm { c o n t } } ,\tag{3.10}
$$

with all weights equal to 1 without any delay, ramp or curriculum learning. The data term is evaluated at the same points as the collocation points drawn uniformly on the trimmed domain. For $d > 1$ a term $\mathcal { L } _ { \mathrm { N } }$ imposes Eq. (2.4) on 128 points per anchor on the insulating segments of the boundary of the modelled domain, with weight 1. Both kinds of boundary condition are therefore soft. The Dirichlet values on the contacts enter through $\mathcal { L } _ { \mathrm { D } }$ with DEVSIM's values at the contact points of the trimmed boundary. The Neumann conditions enter through $\mathcal { L } _ { \mathrm { N } }$

## 3.3.1 Density gating of the physics loss

Our training procedure involves minimizing loss via gradient descent, for which the gradient has to flow back through Boltzmann relations. This introduces density gating in loss minimization, in other words below some value of the electron and hole densities the loss becomes insensitive to their change and hence the network can no longer converge to the right solution. We discuss this in some details below, which justifies our choice of augmenting the network with DEVSIM generated data.

The residual $r _ { n }$ contains ñ through ${ \dot { \mathbf { J } } } _ { n }$ , even though neither ñ nor $\tilde { p }$ appears explicitly in the loss formula $\bigl < r _ { n } ^ { 2 } \bigr >$

Gradient descent requires $\partial \mathcal { L } _ { \mathrm { c o n t } } / \partial \tilde { v } _ { n }$ . Since $\mathcal { L } _ { \mathrm { c o n t } } = \left. r _ { n } ^ { 2 } \right. + \left. r _ { p } ^ { 2 } \right.$

$$
\frac { \partial \mathcal { L } _ { \mathrm { c o n t } } } { \partial \tilde { v } _ { n } } = \frac { \partial \langle r _ { n } ^ { 2 } \rangle } { \partial \tilde { v } _ { n } } + \frac { \partial \langle r _ { p } ^ { 2 } \rangle } { \partial \tilde { v } _ { n } } .\tag{3.11}
$$

The average $\langle \cdot \rangle$ is taken over fixed collocation points, therefore,

$$
\frac { \partial \langle r _ { n } ^ { 2 } \rangle } { \partial \tilde { v } _ { n } } = \bigg \langle \frac { \partial r _ { n } ^ { 2 } } { \partial \tilde { v } _ { n } } \bigg \rangle = 2 \bigg \langle r _ { n } \frac { \partial r _ { n } } { \partial \tilde { v } _ { n } } \bigg \rangle .\tag{3.12}
$$

Differentiating $r _ { n }$ with respect to $\tilde { v } _ { n }$ , and keeping the dominant terms gives,

$$
\frac { \partial r _ { n } } { \partial \tilde { v } _ { n } } = \frac { 1 } { \tilde { \mu } _ { n } } \tilde { \nabla } \cdot \left( \frac { \partial \tilde { \mathbf { J } } _ { n } } { \partial \tilde { v } _ { n } } \right) .\tag{3.13}
$$

It therefore suffices to compute ${ \partial \tilde { \mathbf { J } } _ { n } } / { \partial \tilde { v } _ { n } }$ which gives

$$
\frac { \partial { \tilde { \mathbf { J } } _ { n } } } { \partial { \tilde { v } _ { n } } } = - \tilde { \mu } _ { n } \tilde { n } \frac { \partial } { \partial { \tilde { v } _ { n } } } \big ( \tilde { \nabla } \tilde { v } _ { n } \big ) + \left( - \tilde { \mu } _ { n } \tilde { \nabla } \tilde { v } _ { n } \right) \frac { \partial { \tilde { n } } } { \partial { \tilde { v } _ { n } } } .\tag{3.14}
$$

Both terms carry a factor of $\tilde { n }$ The complete gradient of the continuity loss with respect to $\tilde { v } _ { n }$ is therefore proportional to $\tilde { n } ,$ wherever ñ is exponentially small, the gradient is exponentially suppressed irrespective of the magnitude of the residual $r _ { n }$ . The same argument applies to $r _ { p }$ via ${ \tilde { p } } ,$ and to the Poisson residual $r _ { P }$ via the charge term $\tilde { n } - \tilde { p } .$ It also applies if $\tilde { v } _ { n }$ is replaced by log $\tilde { n }$ as the network output.

There are two ways to address this shortcoming. First is to circumvent loss minimization, which for example is possible using a Gummel-type iterative solver as used in [9], and the second is to include data. We choose to include data as it also helps account for effects not included in the physics equations [20], and can allow for generalisation in the future.

## 3.4 Training and evaluation procedure

The network is trained using Adam optimiser for 20k epochs with an initial learning rate of $1 0 ^ { - 3 }$ . We drop the learning rate by one order of magnitude every 7k epochs. Additionally we clip gradient norm of each of network at 1.0. We train via drawing a batch of 64 out of the 404 training anchors such that every anchor is visited about 3.2k times. In every epoch each of the 64 anchors contributes 256 collocation points drawn uniformly over trimmed domain. The collocation set is resampled every epoch giving approximately 16k collocation points per step. The 128 contact points of the 64 anchors corresponding to the boundary conditions are fixed. We train the network for six random seeds.

We use the median results over the six seeds for our final analysis to be robust against outliers. For the profiles $( \tilde { \phi } , \tilde { v } _ { n } , \tilde { v } _ { p } )$ , which are a function of x we evaluate pointwise median. We checked that the mean and the geometric mean give the same accuracy.

## 3.5 Uncertainty estimation framework

This section describes our approach to quantify the error in surrogate predictions in absence of ground truth. The recipe in short is as follows. We run the surrogate on devices where we also have the DEVSIM answer, measure how large its errors are, and use those errors to attach an interval to every new prediction. Using our test dataset, we check that the DEVSIM value falls inside the surrogate prediction error bars about 90% of the time. To do this, we adopt localised conformal-type intervals, validated on held-out test devices with a device bootstrap.

We consider two sources of uncertainty. First, the random seed, which affects the weight initialization, the anchor selection in batch training and the drawing of collocation points. Second, the finite number of devices we sample, in other words, a different device draw would lead to different answers. The two require different treatments. The effect of the seed is reduced by taking the median over six networks. The effect of the finite device sample on the reported statistics is quantified by the device bootstrap, which resamples the test devices.

The seed to seed variation and thus the median over network predictions helps reduce random seed uncertainty, it does not tell us how wrong that prediction is. We establish that error bar using calibration dataset (see sec. 3.1), in the spirit of the split conformal prediction framework [21-23] (see [24] for an introduction)

We empirically observe that the size of the error depends on the depleted device fraction $\kappa _ { i } = \log _ { 1 0 } ( W _ { \mathrm { d e p } } / L )$ , where $W _ { \mathrm { d e p } }$ is computed using closed form given in $\mathrm { e q . \ ( A . 8 ) }$ Therefore, we model the error as a function of depletion fraction.

For a query with a depletion fraction $\kappa _ { 0 }$ , we weight each calibration anchor i by how close its $\kappa _ { i }$ is to $\kappa _ { 0 }$ κ0,

$$
w _ { i } \big ( \kappa _ { 0 } \big ) = \exp \left[ - \frac { ( \kappa _ { i } - \kappa _ { 0 } ) ^ { 2 } } { 2 h ^ { 2 } } \right] , \qquad h = 0 . 2 .\tag{3.15}
$$

This allows us to construct an error bar on calibration dataset C resulting in relative errors $e _ { i }$ (see eqn. 2.18),

$$
b \bigl ( \kappa _ { 0 } \bigr ) = \operatorname* { i n f } \{ b : \begin{array} { l l } { \displaystyle \sum _ { i \in \mathcal { C } } w _ { i } \bigl ( \kappa _ { 0 } \bigr ) \mathbf { 1 } \bigl [ e _ { i } \le b \bigr ] } \\ { \qquad 1 + \sum _ { i \in \mathcal { C } } w _ { i } \bigl ( \kappa _ { 0 } \bigr ) } \end{array}  \ge ~ 1 - \alpha \} , \qquad 1 - \alpha = 0 . 9 0 ,\tag{3.16}
$$

which is the weighted 90% quantile of the calibration errors, in which the query itself enters the normalisation with weight one and an infinite error, as in weighted conformal prediction. Where the calibration weight near $\kappa _ { 0 }$ is too small for any b to satisfy the condition, b = ∞ and we report no usable bound. This happens outside the calibration support, which is $- 2 . 5 \le \log _ { 1 0 } ( W _ { \mathrm { d e p } } / L ) \lesssim 0 . 8 5$ for the current $( W _ { \mathrm { d e p } } / L \lesssim 7 )$ , and we refuse to give an interval there. Test queries outside the support are excluded from all coverage and tolerance statements. Weighting the calibration scores by the distance to the query in this way is the idea of kernel-weighted and localised conformal prediction [25-27]. Thus for the current the interval

$$
J \in \left[ \frac { \hat { J } _ { \mathrm { R O } } } { 1 + b } , \ \frac { \hat { J } _ { \mathrm { R O } } } { 1 - b } \right]\tag{3.17}
$$

is constructed so that the true current lies within the interval for about 90% of calibration anchors. It is useful to note that b can be $\geq 1$ , in which case we obtain only one sided error with no upper limit on the interval. We report no usable bound for such a query.

Now we use the test devices to establish how often eq. (3.17) holds. This test depends on the set of test devices, therefore we resample the test devices 5 2k times with replacement [28, 29]. We then compute the coverage

$$
{ \mathrm { c o v e r a g e } } = { \frac { 1 } { | T | } } \sum _ { j \in { \mathcal { T } } } \mathbf { 1 } [ e _ { j } \leq b ( \kappa _ { j } ) ] ,\tag{3.18}
$$

evaluated on all bootstrapped test datasets $\tau$ with $e _ { j }$ the relative error and $b ( \kappa _ { j } )$ the error bar defined in eq. (3.16). This leads to a distribution for coverage, from which we report the 2.5-97.5 percentile interval. The construction is called validated for a quantity if this interval contains 0.90 or lies above it.

The validity of this procedure rests on the calibration devices being an exchangeable draw of the population on which it is applied, and we state how our design meets this. The exchangeable unit is the device, not the anchor: the voltages of a device lie on a regular grid, so its anchors are neither independent nor a random draw. Calibration and test devices are therefore drawn by the same Latin hypercube design, and the bootstrap resamples devices. The localisation variable $W _ { \mathrm { d e p } } / L$ is a many-to-one projection of $( L , n _ { D } , V )$ and is not a sufficient statistic for the error, so conditioning on it does not give coverage device by device. The coverage is marginal, and it holds to the extent that calibration and test distributions agree given $W _ { \mathrm { d e p } } / L ,$ which matched designs ensure. The weighted quantile carries no finitesample guarantee and the device-level exchangeability is only approximate, so we treat coverage as something to be measured on the test devices, not assumed. The networks are trained on data together with a physics penalty that is minimised, not enforced, so the physics residual alone does not bound the error of a derived quantity. The calibration attaches an error scale to it.

At query time, a user may specify a tolerance τ for the accuracy of the predictions. We accept a query if its bar satisfies $b ( \kappa _ { j } ) \leq \tau$ , in which case the surrogate vouches for the answer to within τ. To test if this holds, we use test devices and measure a hit rate ${ \begin{array} { r } { \sum _ { j { \mathrm { ~ a c c e p t e d } } } \mathbf { 1 } [ e _ { j } \leq \tau ] / \# } \end{array} }$ {accepted} which we expect to be at least about 0.90, because the errors lie below $b \leq \tau$ for 90% of the calibration anchors.

## 4 Benchmark results across geometries

This section shows that the formalism of Sec. 2 and its quantities of interest are not tied to one geometry or one doping family. Three single devices as mentioned in Sec. 3.1.1 are trained using the same training recipe and evaluated on four held-out voltages to obtain QoIs as defined in sec. 2.4. A detailed study of the one-dimensional symmetric diode follows in Sec. 5.

Table 2 summarises the mean relative errors for QoIs for the three devices. The largest errors are the $L _ { 1 }$ error of the hole density $p$ of the symmetric device (13%) and the stored charge $Q _ { n }$ (11-13% for all three devices); all other entries are below 8%. These are means over the four held-out voltages and include the less accurate voltages discussed below. This shows that the framework itself can produce reliable QoIs for a single device at unseen biases within training range.

<table><tr><td rowspan=1 colspan=1>Quantity</td><td rowspan=1 colspan=1>1D symmetric</td><td rowspan=1 colspan=1>1D asymmetric</td><td rowspan=1 colspan=1>2D planar</td></tr><tr><td rowspan=1 colspan=1> $\psi \left( L _ { 1 } \right)$ </td><td rowspan=1 colspan=1>0.42</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=1>0.11</td></tr><tr><td rowspan=1 colspan=1> $n \left( L _ { 1 } \right)$ </td><td rowspan=1 colspan=1>1.1</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>0.50</td></tr><tr><td rowspan=1 colspan=1> $p \ ( L _ { 1 } )$ </td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>5.0</td></tr><tr><td rowspan=6 colspan=1>Peak field $\mathcal { E } _ { \mathrm { p e a k } }$ Depletion size W = |D| (2D: area)Stored charge $Q _ { n }$ Stored charge $Q _ { p }$  $\Delta E _ { F } ~ ( L _ { 1 } )$ Terminal current $\hat { J } _ { \mathrm { R O } }$ </td><td rowspan=1 colspan=1>0.34</td><td rowspan=1 colspan=1>0.36</td><td rowspan=1 colspan=1>4.6</td></tr><tr><td rowspan=1 colspan=1>1.8</td><td rowspan=1 colspan=1>2.2</td><td rowspan=1 colspan=1>0.69</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>13</td></tr><tr><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>1.4</td><td rowspan=1 colspan=1>1.3</td></tr><tr><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.43</td><td rowspan=1 colspan=1>0.33</td></tr><tr><td rowspan=1 colspan=1>5.1</td><td rowspan=1 colspan=1>2.1</td><td rowspan=1 colspan=1>1.3</td></tr></table>

Table 2. Mean (over the four held-out voltages) relative error in percent for QoIs. For the current, the mean is over the held-out voltages at which the anode current density is at least $1 0 ^ { - 6 } \mathrm { A } / \mathrm { c m } ^ { 2 }$ (0.28, 0.44 and $0 . 6 0 \mathrm { V } ;$ for the 2D device the current per unit depth divided by the $0 . 8 \mu \mathrm { m }$ anode window).

## 4.1 One dimensional diodes

The preceding results are for symmetric junctions $\left( n _ { A } = n _ { D } \right)$ . To test whether the same construction carries over to $n _ { A } \neq n _ { D }$ using the recipe of Sec. 3 we train on the asymmetric device $( n _ { A } / n _ { D } = 1 0 )$ on 12 voltages and evaluate at those and at the four held-out voltages (Sec. 3.1.1).

Fig. 3 shows the internal state reconstruction for held-out devices at voltage 0.44 V. At this voltage the potential is reconstructed across orders of magnitude with residuals below about 0.12 mV for the symmetric device, and the relative error of the densities remains at the 0.1-1% level except in the last few nanometres at the contacts, where it reaches 10–30%. The means in Table 2 are larger because they include the held-out voltages 0.12 and 0.60 V.

The other two held-out voltages are not reproduced as well (Table 2). For asymmetric doping, at 0.60 V the potential RMSE is 5.3 mV, $Q _ { n }$ is off by 32% and the current by $6 \% ;$ at

0.12 V, where the current is below the measurable threshold, the depletion width is off by 6% and $Q _ { n }$ by 13%. Nevertheless, this shows that the formalism generalises to asymmetric devices as well.

![](images/382204fec79894a591d491ab93b235e2a90ebddc442bb2809b7347102dd6623f.jpg)

![](images/1dfb1db0dd20b6bef9418bbb596f5be186ea5e5c6c6113f69747abdc0ab884b7.jpg)  
Figure 3. State-space reconstruction of the symmetric (left) and the asymmetric (right) 1D benchmark at the held-out voltage $V = 0 . 4 4 \mathrm { V }$ , on the trimmed domain. Rows: potential (DEVSIM, solid black; network, dashed blue); electron (solid) and hole (dash-dot) densities; and the relative error of $n , p$ and $\psi$ (dashed line: 20%).

## 4.2 Two-dimensional planar diode

The 2D device has a curved junction below a narrow anode window and a full-width cathode, so the current enters through a small window and spreads laterally, and the field varies along the rounded junction. Figure 1 shows the geometry and the doping. Figure 4 shows the state at the held-out voltage 0.44V.

Overall the errors remain small for the internal state reconstruction as well as for the QoI predictions. The largest error is on the stored charge $Q _ { n }$ (13%, against 1.3% for $Q _ { p } )$ and the peak field is at 4.6% (Table 2).

The benchmarks show that our formalism can reproduce the state, the field-based quantities and the derivative-free current across a symmetric and an asymmetric 1D junction and a 2D planar junction. They do not show how far the answers can be trusted or whether they can be generalised over a family of devices. For these questions we concentrate in the rest of the paper on the one-dimensional symmetric diode as a minimal showcase for QoI determination and reliability construction.

![](images/f4b37c9b8909927a37c73fab8602242420800bccbdd333190cba07893836b47a.jpg)

![](images/65e973660b16fff745db121fc8bdabedbccbb2154c8852b1b531caad5aa73284.jpg)

![](images/afdf61274bbdc90ee76c9ffba066e224896a99e605c5adf1f3a6926e698ea343.jpg)

![](images/f07510da4856c6fb1821c401c00545fed176a86d8c860eaa09977a17c51f2d8a.jpg)

![](images/bd1e7cc3b9045ebe86b2d2e0f10e37ed6f1943e55af471d6c5854cd4f375abd1.jpg)

![](images/04a36c438aee75790227bf01189a7479929255abf43162fb7ad82a8ea28256e8.jpg)

![](images/e031eb2a6a6df39829fc92846e8fa10ea8fc91ed66058c4d4a21e2bae0332088.jpg)  
Figure 4. State of the 2D planar benchmark at the held-out voltage $V = 0 . 4 4 \mathrm { V } .$ on the trimmed model box (the anode is at the top, the cathode at the bottom). Rows: potential, $\log _ { 1 0 }$ n and $\log _ { 1 0 } p ;$ columns: DEVSIM, network, and network minus DEVSIM (the colour scale of the two density differences is clipped at the 99th percentile). Bottom: cuts along the vertical line through the middle of the anode window. The potential differs by at most 2.9 mV, at the corners of the junction. $\log _ { 1 0 } p$ differs by up to 1.7 in the last row next to the cathode; over the whole box its 99th percentile is 0.04.

## 5 One-dimensional symmetric diode detailed results

All results in this section and in Sec. 6 are for the six-seed ensemble on the 246 test devices, restricted to $V / V _ { b i } < 0 . 8$ . We discuss one quantity in the main text and delegate supporting tables and figures to Appendix B. What the physics term of the loss contributes is tested in Appendix D by training the same networks without it: in the median the supervised state and the current are as accurate without it, while the peak field and the residuals of the governing equations are clearly worse, and the error distributions have heavier tails.

## 5.1 State and derived quantities

The quantities of interest (defined in Sec. 2.4) are the terminal current, depletion width, peak field and stored minority charges, which a designer reads off a device, together with the profiles $\psi , n , p ,$ R and $\Delta E _ { F }$ that describe the state they are computed from. Table 3 shows the relative error on each QoI for the test, train and held-out devices. The distribution of the errors over devices and voltages is shown in Fig. 11 (Appendix B). The relative errors on the test devices are larger than on the training devices, by a factor of about 4–6 for $\psi ,$ n and the current.

<table><tr><td rowspan=1 colspan=1>Quantity</td><td rowspan=1 colspan=1>Test (246 devices)</td><td rowspan=1 colspan=1>Training (27 devices)</td><td rowspan=1 colspan=1>Held-out (3 devices)</td></tr><tr><td rowspan=1 colspan=2>Scalar QoIs (relative error)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=1> $\hat { J } _ { \mathrm { R O } }$ Depletion width WPeak field $\mathcal { E } _ { \mathrm { p e a k } }$ </td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.21</td><td rowspan=1 colspan=1>1.4</td></tr><tr><td rowspan=1 colspan=1>1.4</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>1.1</td></tr><tr><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.25</td></tr><tr><td rowspan=2 colspan=1>Stored charge $Q _ { n }$ Stored charge $Q _ { p }$ </td><td rowspan=1 colspan=1>0.99</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.44</td></tr><tr><td rowspan=1 colspan=1>2.9</td><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>4.3</td></tr><tr><td rowspan=6 colspan=3>Profiles (relative $L _ { 1 }$ error) $\psi$ np $R$  $\Delta E _ { F }$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.27</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4.9</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.68</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.70</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.32</td></tr></table>

Table 3. Median percent error of the ensemble against DEVSIM for the test, training and held-out devices.

The errors depend only weakly on the training seed and as shown in table 6, the ensemble accuracy is comparable to the average single network. Averaging the six networks lowers the median error, and to a large part, the error is shared by the seeds.

Figure 5 shows three example state-space reconstructions, the best, worst and one corresponding to high $L , n _ { D }$ device where the solution is particularly stiff. The best reconstruction is indistinguishable from DEVSIM, while for the worst potential is reproduced to within about 22 mV (5.8–9.0 mV RMS) and the worst densities have $L _ { 1 }$ errors of 49% and 39% respectively. The largest errors occur at the contacts. Finally, only the minority tails at the contacts deviate for the stiff anchor. This shows that the network can reconstruct state-space quantities over majority of the regions.

## 5.2 Terminal current prediction

Fig. 6 we show the predicted current $\hat { J } _ { \mathrm { R O } }$ as a function of applied bias for a range of devices. Over most of the regions, the predicted current tracks the DEVSIM truth. The maximal deviation occurs at low bias in long, highly doped devices, where the current is below $1 0 ^ { - 6 }$ $\mathrm { A / c m ^ { 2 } }$ (dashed). Above this value the largest error is a factor of 2.7 at 0.04V in a short, lightly doped device.

![](images/83bb425d1231ddc1e4b454f3c5808189f3fe085ab557138ec6115390081e3046.jpg)

![](images/f1c2e9169d04390e18de7ac6c496ce3b668c8a4d0c022cd556f46807c8d7c60f.jpg)

![](images/c5a610ad717977f237edf7c660b078304e94074549e3fb6996eb47f2a237f35c.jpg)  
Figure 5. State-space reconstruction on the best and worst mean $L _ { 1 }$ error for anchors of the unseen devices, and for an anchor of the stiff high-doping, long-device regime.

In anchors with current $> 1 0 ^ { - 6 } ~ \mathrm { A / c m ^ { 2 } }$ , the error is particularly correlated with fully depleted devices. In these cases $\hat { J } _ { \mathrm { R O } }$ defined in eq. 2.9 amplifies an error $\delta \tilde { \phi }$ in the potential as $e ^ { \delta \tilde { \phi } }$ . Therefore the failure is correlated with the accuracy of $\tilde { \phi }$ in devices only few Debye lengths long.

To complete the discussion of current estimation, in fig. 7, we compare $\hat { J } _ { \mathrm { R O } }$ and $\hat { J } _ { \mathrm { g r a d } }$ averaged over all devices and biases. We see that $\hat { J } _ { \mathrm { R O } }$ is always better than $\hat { J } _ { \mathrm { g r a d } }$ over the entire region, and in particular, $\hat { J } _ { \mathrm { g r a d } }$ can be wrong by more than order of magnitude compared to DEVSIM value specially in large L high $n _ { D }$ region. In appendix C, we discuss the failure of $\hat { J } _ { \mathrm { g r a d } }$ in detail.

Given this, we only consider $\hat { J } _ { \mathrm { R O } }$ for our uncertainty estimates in sec. 6.

## 5.3 Extrapolation performance

In Fig. 8, we show the six network median error for the extrapolation dataset. Between the end of the training range and $V _ { b i }$ the median error of the readout is 25% against 1.9% inside the training range. Above $V _ { b i }$ the median error is 74% and the approach fails, where the DEVSIM current becomes resistance-limited and flattens, while the readout keeps growing exponentially.

The failure above $V _ { b i }$ is related to the surrogate conditioning, which is restricted to a range of $( L , n _ { D } , V )$ and is modeled via FiLM [30]. Beyond their training ranges, the FiLM network extrapolates the response without any restrictions. Therefore, while the physics loss restricts behavior within an anchor, the FiLM response is data-driven. Above $V _ { b i }$ the solution changes character, the depletion region collapses and the current becomes limited by the resistance of the quasi-neutral regions. The corresponding profiles are never seen by the network, correspondingly the extrapolation fails.

![](images/ef24b2f8600a3a549871659f7f5b0d20ef0ba76e7d40e16782bc272436902816.jpg)  
Figure 6. Terminal current against voltage for six test devices spread over current level (top), and the ratio $\hat { J } _ { \mathrm { R O } } / J$ (bottom); each curve ends at $V / V _ { b i } = 0 . 8$ or at 0.64V. The shaded region and the open symbols mark points with $| J _ { \mathrm { D E V S I M } } | < 1 0 ^ { - 6 } \mathrm { A / c m ^ { 2 } } .$ the ratio is dashed below this value. The devices are evenly spaced in mean log-current over the test devices; a device with fewer than five points above the floor or with a wrong-sign readout at any voltage is replaced by the eligible device nearest in current level. Above the floor the current is reproduced over up to seven decades and the largest error occurs at low voltage in the device with $L = 0 . 1 2 \mu \mathrm { m }$ , where the current is small. Below the floor the readout is up to about 14 times too large (device with $L = 4 . 6 4 \mu \mathrm { m }$ at 0.04V).

## 6 Uncertainty quantification for one-dimensional symmetric diode

A surrogate delivers a point prediction with no uncertainty attached to it. That prediction is of limited practical value unless the surrogate can also say how wrong it might be, without consulting the ground-truth simulator. We show that an interval conditioned on a single variable covers the true value about ninety percent of the time across all tested regimes. A user can then query the surrogate which not only returns the prediction but also determines if the prediction is within the user-defined tolerance.

For this, we use conformal predictions as detailed in Sec. 3.5. Conformal prediction has recently been applied to physics-informed networks and neural PDE solvers [31-34].

![](images/6060b40367a6a290dc17c9cf12dcd324d0243763de5bd72d32cbf64e27c72dae.jpg)  
Figure 7. Error factor of the terminal current over the $( L , n _ { D } )$ plane, averaged over voltage, for two different networks. Left: $\hat { J } _ { \mathrm { R O } }$ of the six-network ensemble of the recipe. Right: $\hat { J } _ { \mathrm { g r a d } }$ from a single modified network, see Appendix C for details of the modifications. The error factor is $\operatorname* { m a x } ( r , 1 / r )$ for the ratio r of predicted to DEVSIM current.

Those works calibrate the network outputs. For our application the calibrated object is a scalar quantity derived from the surrogate state.

The error of the surrogate is not a fixed fraction of the prediction. A quasi-neutral device, where carrier transport is governed by diffusion across a short junction, is easier for the network than one that is nearly fully depleted, where the current is exponentially sensitive to a potential that must be resolved across a thin space-charge layer. We capture this by conditioning the 90% error bar (Eq. (3.16)) on the depleted fraction $W _ { \mathrm { d e p } } / L ,$ which is computable in closed form from $( L , n _ { D } , V )$ . As Fig. 9 shows, the bar is narrow across most of the accessible range and widens sharply as the device approaches full depletion. For the current the bar rises to about 0.9 at $W _ { \mathrm { d e p } } / L \approx 7 ,$ the edge of the calibration support. Beyond it no bar is available, and the interval has no upper limit.

Fig. 10 tests whether this construction holds on unseen devices. The bar was frozen after calibration and then applied to 246 held-out test devices. Coverage is close to the nominal level across the plot. Point estimates below 0.90 are the current at $W _ { \mathrm { d e p } } / L < 0 . 1$ (0.88), the peak field at 0.1–0.3 (0.89) and $\mathrm { a t } \geq 0 . 9 5 \ ( 0 . 8 7 )$ , the depletion width at 0.3–0.6 (0.87) and the hole charge at 0.3–0.6 (0.87). In each case the 95% interval reaches 0.90. For the current the coverage is 0.96 at 0.6–0.95 and 0.91 at ≥ 0.95. This shows that the bar is small where network is accurate and wide where it is not, thus producing a measure for reliability.

We observe that the range spanned by the six networks covers only 47-73% of the errors that we obtain here. The ensemble range is not a calibrated predictor, so this is not a comparison with other uncertainty methods. It shows that the seed variation is not a substitute for a calibration, because the networks agree more closely with each other than they agree with DEVSIM.

While the above results validate our approach, the underlying construction guarantees

![](images/23a9ef5ba708bf9e020491befdb718069db6ad4ddafd6d9030628821400a98cd.jpg)

![](images/390155158885ea2902fe21764feecf8c4ef1597fa161eb84dfd93129b0dff1b4.jpg)  
V (V)

![](images/3f24babde379d06969f6d6058dcc258be5f666c7ed5157ef87905fe79d2fe5f2.jpg)  
V (V)

![](images/fcf3482c6d3ec9f10b0aa9cd4167f0ddf9375876f15836d69a85ee81947841a3.jpg)  
V (V)  
Figure 8. Forward-bias extrapolation beyond the training range, for twelve devices. Top: relative error of the network's terminal current, against V . The large error at low V is the small-current regime and is unrelated to extrapolation. Bottom: current against V for the best, median and worst device by median error beyond the end of the training range. The median over devices is drawn where at least six devices contribute.

that nine out of ten queries will be correct. It does not identify which nine. However, a user with a specific tolerance τ can ask if the surrogate's answer is within the user tolerance.
<table><tr><td rowspan="2">QoI</td><td rowspan="2">Inform.</td><td rowspan="2">Coverage</td><td colspan="2"> $\tau = 0 . 0 5$ </td><td colspan="2"> $\tau = 0 . 1 0$ </td><td colspan="2"> $\tau = 0 . 2 5$ </td></tr><tr><td> $\operatorname { A c c } .$ </td><td>Hit</td><td> $\operatorname { A c c } .$ </td><td>Hit</td><td> $\operatorname { A c c } .$ </td><td>Hit</td></tr><tr><td>J</td><td>0.69</td><td> $0 . 9 1 _ { - 0 . 0 3 } ^ { + 0 . 0 3 }$ </td><td>0.80</td><td> $0 . 9 6 _ { - 0 . 0 3 } ^ { + 0 . 0 2 }$ </td><td>0.85</td><td> $0 . 9 8 _ { - 0 . 0 2 } ^ { + 0 . 0 1 }$ </td><td>0.92</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td></tr><tr><td> $W$ </td><td>0.78</td><td> $0 . 9 0 _ { - 0 . 0 3 } ^ { + 0 . 0 3 }$ </td><td>0.66</td><td> $0 . 9 9 _ { - 0 . 0 1 } ^ { + 0 . 0 1 }$ </td><td>0.80</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td>1.00</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td></tr><tr><td> $E _ { \mathrm { p e a k } }$ </td><td>1.00</td><td> $0 . 9 1 _ { - 0 . 0 2 } ^ { + 0 . 0 2 }$ </td><td>0.99</td><td> $0 . 9 9 _ { - 0 . 0 1 } ^ { + 0 . 0 0 }$ </td><td>1.00</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td><td>1.00</td><td> $1 . 0 0 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td></tr><tr><td> $Q _ { n }$ </td><td>0.79</td><td> $0 . 9 2 _ { - 0 . 0 2 } ^ { + 0 . 0 2 }$ </td><td>0.70</td><td> $0 . 9 6 _ { - 0 . 0 2 } ^ { + 0 . 0 1 }$ </td><td>0.84</td><td> $0 . 9 9 _ { - 0 . 0 1 } ^ { + 0 . 0 0 }$ </td><td>0.92</td><td> $0 . 9 9 _ { - 0 . 0 0 } ^ { + 0 . 0 0 }$ </td></tr><tr><td> $Q _ { p }$ </td><td>0.79</td><td> $0 . 9 0 _ { - 0 . 0 4 } ^ { + 0 . 0 4 }$ </td><td>0.00</td><td></td><td>0.23</td><td> $0 . 9 0 _ { - 0 . 0 9 } ^ { + 0 . 0 8 }$ </td><td>0.84</td><td> $0 . 9 4 _ { - 0 . 0 4 } ^ { + 0 . 0 3 }$ </td></tr></table>

Table 4. Tolerance statements on the test devices. “Accepted": fraction of informative queries whose bar is $\le \tau ; \mathrm { \Delta ^ { 6 1 t } } ^ { 5 }$ : fraction of those whose true relative error is $\leq \tau$ (target $0 . 9 0 ; - :$ no query accepted). $\mathrm { ^ { * } I n f o r m . } ^ { \mathrm { 3 } }$ : fraction of all queries on which the quantity is defined; for J it is the fraction with $| \hat { J } _ { \mathrm { R O } } | \geq 1 0 ^ { - 6 } \mathrm { A / c m ^ { 2 } }$ , and the other columns refer to these queries.

![](images/1e2e4ec10e85134838da0478f30e04a27f0ee4b009e57c47f2e0efc609006d56.jpg)

Figure 9. Local 90% error bar for all five quantities, as a function of depleted fraction. The bar is narrow where the device is quasi-neutral and widens as the depletion region grows. For $W _ { \mathrm { d e p } } / L > 0 . 6$ the stored charges $Q _ { n } , Q _ { p }$ and the depletion width are not applicable. The current has a bar up to $W _ { \mathrm { d e p } } / L \approx 7 ;$ beyond it the interval has no upper limit.  
![](images/24ccdb4a45a8a58e5e4cba8f3d87abab9859be77530c706257237ad328c15224.jpg)  
Figure 10. Coverage of the 90% interval by depleted-fraction regime for the five quantities on the test devices (points: coverage; bars: 95% device-bootstrap intervals; dashed: target 0.90). A dash marks a regime that is not applicable (fully depleted: the width and the stored charges are undefined).

Table 4 reports, for each quantity, the fraction of queries accepted at each tolerance as well as their hit rate which quantifies the fraction of accepted queries whose true error is indeed within τ. The hit rate target of 0.90 is met for the current and the other QoIs across the considered tolerances, with one exception. For $Q _ { p }$ no query is accepted at $\tau = 0 . 0 5$ ， and at $\tau = 0 . 1 0$ only 23% are accepted, with a hit rate of 0.90. $Q _ { p }$ is thus usable only at loose tolerances.

<table><tr><td rowspan="2">L (µm)</td><td rowspan="2"> $n _ { D }$   $( \mathrm { c m } ^ { - 3 } )$ </td><td rowspan="2">V (V)</td><td rowspan="2"> $W _ { \mathrm { d e p } } / L$ </td><td rowspan="2"> $J _ { \mathrm { p r e d } }$   $\mathrm { ( A / c m ^ { 2 } ) }$ </td><td rowspan="2"> $J _ { \mathrm { t r u t h } }$   $\mathrm { ( A / c m ^ { 2 } ) }$ </td><td colspan="5"> $b \leq \tau ? \ ( \tau = 0 . 1 0 )$ </td></tr><tr><td>J</td><td>W</td><td> $E _ { \mathrm { p e a k } }$ </td><td> $Q _ { n }$ </td><td> $Q _ { p }$ </td></tr><tr><td>6.9</td><td> $1 . 8 1 \times 1 0 ^ { 1 8 }$ </td><td>0.64</td><td>0.0032</td><td> $6 . 9 6 _ { - 0 . 2 5 } ^ { + 0 . 2 7 } \times 1 0 ^ { - 2 }$ </td><td> $7 . 0 0 \times 1 0 ^ { - 2 }$ </td><td>yes</td><td>no</td><td>yes</td><td>no</td><td>no</td></tr><tr><td>0.91</td><td> $2 . 6 5 \times 1 0 ^ { 1 8 }$ </td><td>0.48</td><td>0.025</td><td> $7 . 5 6 _ { - 0 . 1 6 } ^ { + 0 . 1 7 } \times 1 0 ^ { - 4 }$ </td><td> $7 . 6 0 \times 1 0 ^ { - 4 }$ </td><td>yes</td><td>yes</td><td>yes</td><td>yes</td><td>no</td></tr><tr><td>0.88</td><td> $3 . 4 5 \times 1 0 ^ { 1 7 }$ </td><td>0.64</td><td>0.05</td><td> $2 . 9 7 _ { - 0 . 0 5 } ^ { + 0 . 0 5 }$ </td><td>2.97</td><td>yes</td><td>yes</td><td>yes</td><td>yes</td><td>yes</td></tr><tr><td>0.26</td><td> $1 . 5 0 \times 1 0 ^ { 1 8 }$ </td><td>0.60</td><td>0.097</td><td> $5 . 1 0 _ { - 0 . 1 3 } ^ { + 0 . 1 4 } \times 1 0 ^ { - 1 }$ </td><td> $5 . 1 2 \times 1 0 ^ { - 1 }$ </td><td>yes</td><td>yes</td><td>yes</td><td>yes</td><td>no</td></tr><tr><td>0.15</td><td> $1 . 4 1 \times 1 0 ^ { 1 8 }$ </td><td>0.44</td><td>0.21</td><td> $2 . 2 5 _ { - 0 . 0 5 } ^ { + 0 . 0 5 } \times 1 0 ^ { - 3 }$ </td><td> $2 . 2 5 \times 1 0 ^ { - 3 }$ </td><td>yes</td><td>yes</td><td>yes</td><td>yes</td><td>no</td></tr><tr><td>0.27</td><td> $4 . 0 3 \times 1 0 ^ { 1 6 }$ </td><td>0.52</td><td>0.49</td><td> $1 . 3 8 _ { - 0 . 0 3 } ^ { + 0 . 0 3 }$ </td><td>1.38</td><td>yes</td><td>yes</td><td>yes</td><td>no</td><td>no</td></tr><tr><td>0.13</td><td> $2 . 9 4 \times 1 0 ^ { 1 6 }$ </td><td>0.48</td><td>1.3</td><td> $2 . 0 1 _ { - 0 . 2 0 } ^ { + 0 . 2 5 }$ </td><td>1.95</td><td>no</td><td>n.a.</td><td>yes</td><td>n.a.</td><td>n.a.</td></tr><tr><td>0.12</td><td> $1 . 4 6 \times 1 0 ^ { 1 5 }$ </td><td>0.04</td><td>8.7</td><td> $1 . 0 3 \times 1 0 ^ { - 5 }$ </td><td> $3 . 7 6 \times 1 0 ^ { - 6 }$ </td><td>no</td><td>n.a.</td><td>no</td><td>n.a.</td><td>n.a.</td></tr></table>

Table 5. Example test queries and their answers: eight measurable test queries at evenly spaced ranks of $W _ { \mathrm { d e p } } / L$ The columns $b \leq \tau$ state whether the error bar of the quantity is at most $\tau = 1 0 \%$ which needs no DEVSIM value; $J _ { \mathrm { t r u t h } }$ is shown for comparison. The last query lies outside the calibration support $( W _ { \mathrm { d e p } } / L \gtrsim 7 )$ , so no interval is issued.

Table 5 shows eight representative test queries spanning more than three decades of depleted fraction. In the middle of the distribution, where most practical devices operate, the interval is a few percent wide and the ground truth falls comfortably inside it. Toward fully depleted devices the interval opens to a factor of about 1.25 between its bounds $( W _ { \mathrm { d e p } } / L = 1 . 3 )$ , and eventually loses its upper limit, signalling that the surrogate can no longer vouch for the answer. The last query (a 2.7-fold overestimate at $W _ { \mathrm { d e p } } / L = 8 . 7 )$ lies outside the calibration support, where no interval is issued. Among the 2511 measurable test anchors inside the calibration support, 91% lie inside the interval; the other 9 anchors (3 devices) get no interval.

While we show that construction of such intervals is possible, this construction is not without limitations. First and foremost, the coverage we computed is marginal over queries and not a per device quantity, which is the most that can be expected without further assumptions. It also holds only approximately: the localised quantile has no finitesample guarantee and the device-level exchangeability of calibration and test devices is an approximation, so the coverage is established by the test devices and not proven. Third, intervals are offered only inside the calibration support $- 2 . 5 \le \log _ { 1 0 } ( W _ { \mathrm { d e p } } / L ) \lesssim 0 . 8 5$ . Below it, in the stiff corner of high doping and long devices $( W _ { \mathrm { d e p } } \sim 2 0 \mathrm { n m } )$ , the errors are 10–20 times larger than in the central bins and only about 1% of the devices fall there, too few to calibrate, so we report no interval. Fourth, this calibration and interval construction needs labelled devices not used in training. This requires additional simulations. For a 1D device, it is computationally affordable to run DEVSIM multiple times, however as the complexity of the devices increases, these additional simulations may become an extra challenge.

It is nevertheless one of the simplest methods to construct reliability test for a surrogate. In the future, we will consider extensions of this framework for 2D/3D devices, where we will analyse the above challenges in detail.

## 7 Conclusions

We presented a physics-informed surrogate for p-n junction diodes in which the network represents the electrostatic potential and the two quasi-Fermi levels. The formalism and the derivative-free current readout apply to one- and two-dimensional devices, as shown for a symmetric and an asymmetric 1D junction and a 2D planar junction, and from one internal state they deliver several quantities of interest for device physicists.

Going beyond these benchmarks, we built a detailed surrogate of a one-dimensional symmetric abrupt $p  – n$ diode in forward bias, conditioned on the doping, the device length and the applied bias. We used it to predict five quantities of interest, namely, the terminal current, the depletion width, the peak electric field and the stored charges $Q _ { n }$ and $Q _ { p }$

The quantities of interest are themselves not definition free. The terminal current can be computed by differentiating the network output or by a derivative-free readout based on the Gummel relation, and the readout is the better choice. The differentiated current has to follow a quasi-Fermi slope that varies over many decades across a thin junction layer, which is hard for the network to resolve, whereas the integral of the potential is accurate

We also constructed localised conformal-type error intervals for each quantity of interest, with the error calibrated as a function of a single variable, the depleted fraction $W _ { \mathrm { d e p } } / L$ , on 310 calibration devices drawn like the test devices. On 246 unseen test devices, drawn after everything else was frozen, the marginal coverage is between 0.90 and 0.92 for all five quantities, with a few regimes below nominal whose confidence intervals reach 0.90. The coverage is marginal, not per device, and the intervals are only offered within the calibration support. This turns the surrogate from a predictor into an uncertainty-aware one. At query time, a user can decide whether the surrogate's answer is accurate to within a tolerance of their choice.

The surrogate is reliable within its training range. Extrapolating in voltage degrades beyond that range, to a median error of 25% up to the built-in potential against 1.9% inside it, and it fails above the built-in potential, where the device enters a regime absent from the training data.

The scope of the study is limited. The detailed analysis covers one doping family, a symmetric abrupt junction with a fixed relative smearing of the doping, constant mobility, long lifetimes and forward bias only. Although the benchmarks demonstrate the formalism for non-trivial geometries for a single device, they do not show how well the network generalizes across a family of devices.

Several challenges remain. It is open whether the mesh-free training approach carries over to conditioning on 2D/3D device geometry, where geometry-aware architectures such as graph neural networks may be preferable. Furthermore, more distribution-aware uncertainty models could reduce the need for additional labelled simulations for calibration, at the price of modelling assumptions, and it is of interest which of the two approaches is preferable as device complexity grows.

Despite the limited scope, we believe this is a successful demonstration of a physicsinformed framework that returns not only the internal device state but also several quantities of interest, each with a reliability estimate.

## 8 Acknowledgements

We thank the University of Graz for support with the HPC computing. Additionally, we thank Andreas Windisch for useful discussions on the topic.

During the preparation of this work, we used generative AI for fast experimentation, generate and refine code, and improve the clarity of the manuscript. AI-generated suggestions informed aspects of the methodological development and analysis, which were subsequently critically evaluated, refined, and validated by the author. The author takes full responsibility for the scientific content, computational results, and conclusions presented in this work.

## References

[1] Synopsys, Inc., Mountain View, CA, Sentaurus Device User Guide. Version W-2024.09-SP1, Sept., 2024.

[2] SILVACO, SILVACO TCAD®, 2025.

[3] COMSOL AB, COMSOL MultiphysicsR v. 6.0, Stockholm, Sweden, 2024.

[4] M. Raissi, P. Perdikaris and G. E. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics 378 (Feb., 2019) 686–707.

[5] K. Mehta and H. Y. Wong, Prediction of FinFET current-voltage and capacitance-voltage curves using machine learning with autoencoder, IEEE Electron Device Letters 42 (2021) 136-139.

[6] S. Myung, W. Jang, S. Jin, J. M. Choe, C. Jeong and D. S. Kim, Restructuring TCAD system: Teaching traditional TCAD new tricks, in 2021 IEEE International Electron Devices Meeting (IEDM), 2021, 2204.09578, DOI.

[7] H. Y. Wong, AI and TCAD for inverse design and defect discovery: From simple machine learning to LLM, arXiv e-prints (2026) , [2609.07046].

[8] B. Cao, Y.-D. Wang, N.-E. Zhang, Y.-Z. Liang and W.-Y. Yin, A Physics-Informed Neural Networks Algorithm for Simulating Semiconductor Devices, in 2023 International Applied Computational Electromagnetics Society Symposium (ACES-China), pp. 1–3, Aug., 2023, DOI.

[9] Z. Cai, A. Huang, Y. Xiong, D. Mu, X. Miao and X. Wang, Multi-order Differential Neural Network for TCAD Simulation of the Semiconductor Devices, in Proceedings of the 61st ACM/IEEE Design Automation Conference, DAC '24, (New York, NY, USA), pp. 1–6, Association for Computing Machinery, Nov., 2024, DOI.

[10] B. Kim and M. Shin, A Novel Neural-Network Device Modeling Based on Physics-Informed Machine Learning, IEEE Transactions on Electron Devices 70 (Nov., 2023) 6021–6025.

[11] R. Riganti, M. G. C. Alasio, E. Bellotti and L. Dal Negro, DDNet: A Unified Physics-Informed Deep Learning Framework for Semiconductor Device Modeling, arXiv e-prints (Sept., 2025) arXiv:2509.08073, [2509.08073].

[12] A. Lu, Y. F. Chau and H. Y. Wong, Physics-informed neural network for predicting out-of-training-range TCAD solution with minimized domain expertise, arXiv e-prints (Aug., 2024), [2408.07921].

[13] L. Popryho, A. Sadeghi and I. Partin-Vaisband, Mesh-Native Physics-Informed Graph Surrogates for TCAD-in-the-Loop Design Space Exploration, arXiv e-prints (Sept., 2026) arXiv:2609.02988, [2609.02988].

[14] Y. Zhang, Z. Zhang, K. Chen and C. Wang, PCGD: Physics-Guided Conditional Graph Diffusion for TCAD Device Simulation, arXiv e-prints (June, 2026) arXiv:2606.29272, [2606.29272].

[15] K. Kim, S. Myung, Y. Choi, G. Kang, K. Yeom, S. Han et al., Neural drift-diffusion model based on operator learning in Fourier space, in 2024 International Conference on Simulation of Semiconductor Processes and Devices (SISPAD), IEEE, 2024.

[16] H. Zhang, A. M. Propp, B. Kinch, H. Owhadi and N. Trask, Structure-preserving neural surrogates with tractable uncertainty quantification, arXiv e-prints (June, 2026) , [2606.11650]. arXiv:2606.11650.

[17] J. L. Moll and I. M. Ross, The dependence of transistor parameters on the distribution of base layer resistivity, Proceedings of the IRE 44 (1956) 72–78.

[18] H. K. Gummel, A charge control relation for bipolar transistors, Bell System Technical Journal 49 (1970) 115–120.

[19] J. E. Sanchez, Devsim: A tcad semiconductor device simulator, Journal of Open Source Software 7 (2022) 3898.

[20] A. Petrosyan and A. Yepiskoposyan, Hybrid PINN-TCAD framework for sub-percent p−n junction simulation with spillover error quantification, Journal of Computational Electronics (11, 2025) .

[21] H. Papadopoulos, K. Proedrou, V. Vovk and A. Gammerman, Inductive confidence machines for regression, in Machine Learning: ECML 2002, vol. 2430 of Lecture Notes in Computer Science, pp. 345–356, Springer, 2002, DOI.

[22] V. Vovk, A. Gammerman and G. Shafer, Algorithmic Learning in a Random World. Springer, New York, 2005.

[23] J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani and L. Wasserman, Distribution-free predictive inference for regression, Journal of the American Statistical Association 113 (2018) 1094–1111, [1604.04173].

[24] A. N. Angelopoulos and S. Bates, Conformal prediction: A gentle introduction, Foundations and Trends in Machine Learning 16 (2023) 494–591. arXiv:2107.07511, titled “A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification".

[25] J. Lei and L. Wasserman, Distribution-free prediction bands for non-parametric regression, Journal of the Royal Statistical Society: Series B (Statistical Methodology) 76 (2014) 71–96.

[26] R. J. Tibshirani, R. Foygel Barber, E. J. Candes and A. Ramdas, Conformal Prediction Under Covariate Shift, arXiv e-prints (Apr., 2019) arXiv:1904.06019, [1904.06019].

[27] L. Guan, Localized conformal prediction: a generalized inference framework for conformal prediction, Biometrika (2023) . arXiv:2106.08460.

[28] C. A. Field and A. H. Welsh, Bootstrapping clustered data, Journal of the Royal Statistical Society: Series B (Statistical Methodology) 69 (2007) 369–390.

[29] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. Chapman & Hall, New York, 1993.

[30] E. Perez, F. Strub, H. de Vries, V. Dumoulin and A. Courville, FiLM: Visual reasoning with a general conditioning layer, in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, pp. 3942–3951, 2018, 1709.07871.

[31] L. Podina, M. Torabi Rad and M. Kohandel, Conformalized Physics-Informed Neural Networks, arXiv e-prints (May, 2024) arXiv:2405.08111, [2405.08111].

[32] Y. Yu, C. H. Ho and Y. Wang, A Conformal Prediction Framework for Uncertainty Quantification in Physics-Informed Neural Networks, arXiv e-prints (Sept., 2025) arXiv:2509.13717, [2509.13717].

[33] V. Gopakumar, A. Gray, L. Zanisi, T. Nunn, D. Giles, M. J. Kusner et al., Calibrated Physics-Informed Uncertainty Quantification, arXiv e-prints (Feb., 2025) arXiv:2502.04406, [2502.04406].

[34] V. Gopakumar, A. Gray, J. Oskarsson, L. Zanisi, D. Giles, M. J. Kusner et al., Uncertainty quantification of surrogate models using conformal prediction, Machine Learning: Science and Technology 7 (Feb., 2026) 015025.

## A Drift-diffusion theory

This appendix collects the equations that Sec. 2 uses in non-dimensional form. In physical units the problem is

$$
\begin{array} { r } { \nabla \cdot \bigl ( \varepsilon \nabla \psi \bigr ) + q \bigl ( p - n + N _ { \mathrm { n e t } } \bigr ) = 0 , } \end{array}\tag{A.1}
$$

$$
\nabla \cdot { \bf J } _ { n } - q R = 0 , \qquad \nabla \cdot { \bf J } _ { p } + q R = 0 ,\tag{A.2}
$$

$$
\mathbf { J } _ { n } = q \mu _ { n } n \mathbf { E } + q D _ { n } \nabla n \mathbf { J } _ { n } = q \mu _ { p } p \mathbf { E } - q D _ { p } \nabla p\tag{A.3}
$$

$$
\mathbf { E } = - \nabla \psi\tag{A.4}
$$

where ψ is the electrostatic potential, $N _ { \mathrm { n e t } } ( x ) = N _ { D } ( x ) - N _ { A } ( x )$ the net doping (signed ${ \mathrm { s t e p } } , - n _ { A }$ for $x < x _ { j }$ , +nD for $x > x _ { j } )$ , and

$$
R = \frac { n p - n _ { i } ^ { 2 } } { \tau _ { p } ( n + n _ { i } ) + \tau _ { n } ( p + n _ { i } ) }\tag{A.5}
$$

the SRH rate $( n _ { i }$ the intrinsic concentration, $\tau _ { n } , \tau _ { p }$ carrier lifetimes). The electron and hole current here is a sum over two terms indicating the drift and diffusion processes and for a successful physics solver they balance exactly in order to yield currents which vary over orders of magnitude as the device parameters and bias $( n _ { A } , n _ { D } , L , V )$ change.

Rather than demanding such balance from the neural network solutions, we adopt a useful well-known reformulation known as the quasi-Fermi level (QFL) parametrization.

In QFL parametrization, the state-space vectors $( \psi , n , p )$ are rewritten in terms of electron and hole quasi-Fermi potentials $\phi _ { n } , \phi _ { p }$ and are defined through the Boltzmann relations

$$
n = n _ { i } \exp \mathopen { } \mathclose \bgroup \left( \frac { \psi - \phi _ { n } } { U _ { t } } \aftergroup \egroup \right) , \qquad p = n _ { i } \exp \mathopen { } \mathclose \bgroup \left( \frac { \phi _ { p } - \psi } { U _ { t } } \aftergroup \egroup \right) ,\tag{A.6}
$$

with $U _ { t } = k T / q$ . The currents follow as

$$
{ \bf J } _ { n } = - q \mu _ { n } n \nabla \phi _ { n } , \qquad { \bf J } _ { p } = - q \mu _ { p } p \nabla \phi _ { p } .\tag{A.7}
$$

It is evident that the current, although algebraically equivalent to the drift-diffusion form, now does not require balancing of terms involving derivatives.

Although we do not use it in our training, for a planar abrupt junction (the onedimensional device) a characteristic length scale is the depletion width, which can be approximated as

$$
W _ { \mathrm { d e p } } ( V ) = \sqrt { \frac { 2 \varepsilon ( V _ { b i } - V ) } { q } \Big ( \frac { 1 } { n _ { A } } + \frac { 1 } { n _ { D } } \Big ) } , \qquad V _ { b i } = U _ { t } \ln \frac { n _ { A } n _ { D } } { n _ { i } ^ { 2 } } ,\tag{A.8}
$$

which reduces to $\sqrt { 4 \varepsilon ( V _ { b i } - V ) / ( q n _ { D } ) }$ with $V _ { b i } = 2 U _ { t } \ln ( n _ { D } / n _ { i } )$ for $n _ { A } = n _ { D }$ . The depletion region extends over $x _ { p } = W _ { \mathrm { d e p } } n _ { D } / ( n _ { A } + n _ { D } )$ on the p-side and $x _ { n } = W _ { \mathrm { d e p } } n _ { A } / ( n _ { A } + n _ { D } )$ on the n-side of the junction, so mostly into the lighter-doped side; for $n _ { A } \gg n _ { D }$ (one-sided junction) $W _ { \mathrm { d e p } } \to x _ { n } = \sqrt { 2 \varepsilon ( V _ { b i } - V ) / ( q n _ { D } ) }$

The non-dimensional quantities of Sec. 2 are defined by

$$
\tilde { \phi } = \psi / U _ { t } , \qquad \tilde { v } _ { n } = \phi _ { n } / U _ { t } = \tilde { \phi } - \ln ( n / n _ { i } ) \qquad \tilde { v } _ { p } = \phi _ { p } / U _ { t } = \tilde { \phi } + \ln ( p / n _ { i } )\tag{A.9}
$$

$$
\mu _ { \mathrm { m a x } } = \operatorname * { m a x } ( \mu _ { n } , \mu _ { p } ) , \qquad \lambda _ { D } = \sqrt { \varepsilon U _ { t } / ( q n _ { \mathrm { m a x } } ) } , \qquad \lambda = \lambda _ { D } / \ell ,\tag{A.10}
$$

$$
\tilde { \mu } _ { n } = { \mu _ { n } } / { \mu _ { \mathrm { m a x } } } , \quad \tilde { \mu } _ { p } = { \mu _ { p } } / { \mu _ { \mathrm { m a x } } } ,\tag{A.11}
$$

$$
\tilde { n } _ { i } = n _ { i } / n _ { \mathrm { m a x } } , \quad \tilde { \tau } _ { n } = \tau _ { n } \mu _ { \mathrm { m a x } } U _ { t } / \ell ^ { 2 } , \quad \tilde { \tau } _ { p } = \tau _ { p } \mu _ { \mathrm { m a x } } U _ { t } / \ell ^ { 2 } ,\tag{A.12}
$$

$$
J _ { \mathrm { r e f } } = \frac { q \mu _ { \mathrm { m a x } } U _ { t } n _ { \mathrm { m a x } } } { \ell } , \qquad R _ { \mathrm { r e f } } = \frac { \mu _ { \mathrm { m a x } } U _ { t } n _ { \mathrm { m a x } } } { \ell ^ { 2 } } , \qquad n _ { \mathrm { m a x } } = \operatorname * { m a x } ( n _ { A } , n _ { D } ) .\tag{A.13}
$$

## B Error distributions and seed variation

These tables and figures support results stated in Sec. 5 and Sec. 6. Each repeats or extends an exhibit that carries the claim in the main text.

For every QoI the error of the ensemble is below the mean error of a single network. The readout current is the quantity least affected by the seed (single-network errors 0.73– $1 . 1 \% )$ . The ensemble error lies below the whole range of the single networks for the readout current, $Q _ { n } , \psi , p ;$ R and $\Delta E _ { F }$ . For the other QoIs the best single network is as accurate as the ensemble or slightly better (for example 0.35% against 0.39% for the peak field).

## C $\hat { J } _ { \mathrm { g r a d } }$ analysis

In this appendix we investigate the origin of the failure of $\hat { J } _ { \mathrm { g r a d } }$ to obtain current. In order to assist the network, we add two more features to the training pipeline. First the built-in-potential column $\eta = \left( V _ { b i } - V \right) / U _ { t }$ in the conditioning vector. This is added such that the network is aware of how far it is from the built-in potential, where the current becomes resistance limited. Secondly, the size of the continuity residual differs strongly between anchors. In the scaled units the peak SRH recombination rate, which sets this size, is proportional to

![](images/808222e2be3731586d1c564a08e4b59c762e8a8ed6dc605fb701a99e4795fcbf.jpg)  
Figure 11. Distribution of the ensemble error (Eq. (2.18); relative $L _ { 1 }$ error for the profiles in the top row, relative error for the scalar QoIs in the bottom row) over the (device, voltage) points of the training, held-out and test devices, on a logarithmic axis. Boxes: quartiles with median; whiskers: 5th and 95th percentiles. Dots: the median error of each of the six single networks on that set, so the spread of the dots around the box is the variation between networks. The held-out set has three devices and is indicative only. Table 3 gives the test medians.

$$
R _ { \mathrm { p e a k , r e f } } = \frac { n _ { D , \mathrm { r e f } } / n _ { \mathrm { m a x } } } { 2 \sqrt { \tilde { \tau } _ { n } \tilde { \tau } _ { p } } } e ^ { V / 2 U _ { t } } ,\tag{C.1}
$$

with $n _ { D , \mathrm { r e f } } = 1 0 ^ { 1 5 } \mathrm { c m ^ { - 3 } }$ . It falls as $n _ { \mathrm { m a x } }$ increases and grows with the bias. As we train over a pool of anchors, the continuity loss therefore varies strongly from anchor to anchor, which is a mechanism similar to the density gating. To compensate, we divide the continuity residuals by $R _ { \mathrm { p e a k , r e f } }$ . This brings the losses of all anchors to a similar magnitude and allows for a better optimisation. The physical peak rate has $n _ { i }$ in place of $n _ { D , { \mathrm { r e f } } } ,$ SO $R _ { \mathrm { p e a k , r e f } }$ differs from it by the constant factor $n _ { D , \mathrm { r e f } } / n _ { i }$ , a numerical choice.

<table><tr><td rowspan=1 colspan=2>Quantity</td><td rowspan=1 colspan=1>Ensemble error</td><td rowspan=1 colspan=1>Single network $\mathrm { m e a n } \pm \mathrm { s t d }$ </td><td rowspan=1 colspan=1>Single networkmin-max</td></tr><tr><td rowspan=3 colspan=3>Scalar QoIs (relative error)Readout current $\hat { J } _ { \mathrm { R O } }$ Depletion width W</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1> $0 . 8 8 \pm 0 . 1 3$ </td><td rowspan=5 colspan=1>0.73-1.11.4-1.80.35-0.582.5-4.0</td></tr><tr><td rowspan=1 colspan=1> $1 . 6 \pm 0 . 1 4$ </td></tr><tr><td rowspan=3 colspan=2>Peak field $\mathcal { E } _ { \mathrm { p e a k } }$ Stored charge $Q _ { n }$ Stored charge $Q _ { p }$ </td><td rowspan=1 colspan=1> $0 . 3 9 _ { - 0 . 0 3 7 } ^ { + 0 . 0 4 3 }$ </td><td rowspan=1 colspan=1> $0 . 4 3 \pm 0 . 0 9 5$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $0 . 9 9 _ { - 0 . 1 4 } ^ { + 0 . 1 5 }$ </td><td rowspan=1 colspan=1> $1 . 3 \pm 0 . 1 2$ </td><td rowspan=1 colspan=1>1.2-1.5</td></tr><tr><td rowspan=1 colspan=1> $2 . 9 _ { - 0 . 4 5 } ^ { + 0 . 5 6 }$ </td><td rowspan=1 colspan=1> $3 . 4 \pm 0 . 5 9$ </td></tr><tr><td rowspan=1 colspan=4>Profiles (relative $L _ { 1 }$ error)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=2> $\psi$ n $p$ </td><td rowspan=1 colspan=1> $0 . 1 1 _ { - 0 . 0 1 2 } ^ { + 0 . 0 1 9 }$ </td><td rowspan=1 colspan=1> $0 . 1 3 \pm 0 . 0 0 5 0$ </td><td rowspan=3 colspan=1>0.12-0.141.3-1.80.74-1.1</td></tr><tr><td rowspan=1 colspan=1> $1 . 5 _ { - 0 . 2 6 } ^ { + 0 . 3 7 }$ </td><td rowspan=1 colspan=1> $1 . 6 \pm 0 . 1 5$ </td></tr><tr><td rowspan=1 colspan=1> $0 . 6 1 _ { - 0 . 0 7 2 } ^ { + 0 . 0 9 1 }$ </td><td rowspan=1 colspan=1> $0 . 9 1 \pm 0 . 1 4$ </td></tr><tr><td rowspan=1 colspan=2> $R$ </td><td rowspan=1 colspan=1> $0 . 7 2 _ { - 0 . 0 5 6 } ^ { + 0 . 0 5 4 }$ </td><td rowspan=1 colspan=1> $1 . 0 \pm 0 . 1 5$ </td><td rowspan=2 colspan=1>0.76-1.20.31-0.44</td></tr><tr><td rowspan=1 colspan=2> $\Delta E _ { F }$ </td><td rowspan=1 colspan=1> $0 . 2 9 _ { - 0 . 0 2 8 } ^ { + 0 . 0 3 1 }$ </td><td rowspan=1 colspan=1> $0 . 3 7 \pm 0 . 0 4 9$ </td></tr></table>

Table 6. Dependence on the random seed. The six networks differ only in their seed and have a common training dataset

With these two additions, we train a single network and evaluate it on the measurable anchors of test devices of an earlier Latin hypercube draw. On these anchors the differentiated current is within a factor of two of DEVSIM for 72% of the anchors, against 98% for the readout. These two features were introduced to give $\hat { J } _ { \mathrm { g r a d } }$ its best chance, so this comparison favours $\hat { J } _ { \mathrm { g r a d } }$ and yet falls short of delivering a high accuracy answer. Without the additional features, only 13% of the anchors lead to a current within a factor of two. We deliberately choose a loose acceptance criteria to demonstrate the $\hat { J } _ { \mathrm { g r a d } }$ weakness.

Figure 12 compares the best readout with the best differentiated current. The top row is the readout $\hat { J } _ { \mathrm { R O } }$ of the ensemble network. The bottom row is the differentiated current $\hat { J } _ { \mathrm { g r a d } }$ of the network with the η column and the $R _ { \mathrm { p e a k , r e f } }$ divisor. The two rows are different networks by design and use the same 120 unseen devices, and the figure compares the broad pattern of failure. Unlike the rest of this appendix, the maps include anchors with $| J _ { \mathrm { t r u t h } } | < 1 0 ^ { - 6 } \mathrm { A / c m ^ { 2 } }$ , where the DEVSIM current is close to its numerical floor, so part of the red region at the lowest voltages may reflect that floor. The readout is within a factor of two over nearly the whole plane at every voltage shown, with small regions of larger error at the highest doping. This corner shrinks as the voltage grows and is within a factor of two at 0.64 V. Anchors with $V / V _ { b i } \ge 0 . 8 \left( V _ { b i } = 2 U _ { t } \ln ( n _ { D } / n _ { i } ) \right)$ are excluded, so at 0.64 V only devices with $n _ { D } \gtrsim 5 \times 1 0 ^ { 1 6 } \mathrm { c m ^ { - 3 } }$ remain (65 devices) and the band of intermediate doping that fails above $0 . 8 V _ { b i }$ is not shown.

![](images/23be7c3dea96ffdebe67d7263d0096a9ec7792c6d9234c20b566f08fa6817790.jpg)  
Figure 12. Error factor of the readout current $\hat { J } _ { \mathrm { R O } }$ of the ensemble (top row) and of the differentiated current $\hat { J } _ { \mathrm { g r a d } }$ of the network with the η column and the $R _ { \mathrm { p e a k , r e f } }$ divisor (bottom row) over the $( L , n _ { D } )$ plane, for the same 120 unseen devices, including those with $| J _ { \mathrm { t r u t h } } | < 1 0 ^ { - 6 } \mathrm { A / c m ^ { 2 } }$ . The hatched region at 0.64 V marks the devices with $V \geq 0 . 8 V _ { b i }$ , which are excluded.

We find the failure to be associated with the device length in units of the Debye length, $L / \lambda _ { D }$ , and offer the following interpretation. A device that is many Debye lengths long is mostly quasi-neutral, and the carrier densities change by many decades across a thin layer around the junction. Neglecting the small recombination current, the electron current $J _ { n } = - q \mu _ { n } n \mathrm { d } v _ { n } / \mathrm { d } x$ has to be the same everywhere in the device, so the slope $\mathrm { d } v _ { n } / \mathrm { d } x$ has to follow $1 / n$ , that is change by many decades within a few Debye lengths. An error of the slope is further multiplied by the density, which amplifies the error.

Figure 13 shows the quasi-Fermi levels and their slopes along the three devices, for $L / \lambda _ { D } = 1 0$ , 97 and 858. While the agreement between DEVSIM and network predictions is excellent for $v _ { n }$ and $v _ { p }$ their slopes show a different picture. The slope of the network follows that of DEVSIM at $L / \lambda _ { D } = 1 0$ and 97 with small deviations, while at 858 it oscillates around it in the majority-carrier bulk on both sides of the junction, with amplitudes comparable to the slope itself. We have not established the origin of these oscillations, but for these three devices the figure illustrates that the $v _ { n } , v _ { p }$ slopes degrade as $L / \lambda _ { D }$ increases.

## D Ablation of the physics term

This appendix tests the contribution of physics in the final network prediction. To this end, we trained additional networks using the 1D symmetric train dataset with data and boundary term loss only. We train six such networks, with the same six seeds as the physics-informed ones, and score them on the test devices exactly as in the main text. Numbers are medians over the test anchors of this ensemble.

![](images/44d5035e3a02c6f9d46bfb45a7dbf0f55eea135b9209810378617eb13c9d3fe0.jpg)

![](images/0886d3c3eb3f0856b692008254328e57d002e65ef7b149ea9496265944887c7d.jpg)

![](images/d69831e61a01ed0e7c0095fbc05cd7320ccb52d8ca032c5c9fcce6ebc904b63f.jpg)

![](images/f5d21a91d1db32c78463cca5a46c54585e2766ec62d20c46a491bbaee17bf756.jpg)

![](images/abad2e749854b27f615eef3e0e48930df94c14a8aa1cedb1b98c382db2465e57.jpg)

![](images/979f423d1d5cd9405786e87eacca3cc94c5c7265d6b260afb367a3b358b77b74.jpg)

![](images/f81547b713a80a3e67f25f03792fa9515d144041b98f11e5cbb7c54bf044cde1.jpg)

![](images/5e0f6760645ffbb4a9017a016f8f5c5abce21a1cf8a58dcdd02c09db9383cd1d.jpg)

![](images/ae027867609330196c71ab97cdb8a5c41fab42e76db7478c1624db9111a7c969.jpg)  
Figure 13. Quasi-Fermi levels $v _ { n }$ and $v _ { p }$ (top row) and their slopes $\mathrm { { d } } v _ { n } / \mathrm { { d } } x$ (middle) and d $v _ { p } / \mathrm { d } x$ (bottom) along three unseen devices at $V \mathrm { ~ = ~ } 0 . 5 2 \ : \mathrm { V }$ , of the network with the η column and the SRH-peak divisor, for $L / \lambda _ { D }$ of 10, 97 and 858. Black: DEVSIM; dashed: network. The slope axes are symmetric logarithmic (linear below $1 0 0 \mathrm { V / c m } )$ . Each column is labelled with $L / \lambda _ { D }$ and $J _ { \mathrm { p r e d } } ^ { \mathrm { g r a d } } / J _ { \mathrm { t r u t h } }$

We observe that the median of the physics-informed and data network predictions are comparable on the supervised fields and on the current. However, Fig. 14 (left panel) shows that the error distributions of data only network have heavier tails. Relative to its own median, the 90th percentile of the error on $\psi _ { : }$ n and $\hat { J } _ { \mathrm { R O } }$ is about twice as large in the data-only networks as in the physics-informed ones.

The physics-informed advantage is particularly visible in the peak E field as seen in the right panel of Fig. 14. The error of the peak field $\mathcal { E } _ { \mathrm { p e a k } }$ is about 8 times larger in the data-only ensemble than in the physics-informed one, and the whole distribution is shifted. The relative $L _ { 1 }$ error of the field of a single network is larger by a factor of 2.3. The gap thus grows with the order of the derivative of $\psi$ on which a quantity depends: none for the values of $\psi ,$ n and $p ,$ about two for the field, and about eight for its peak. The quantities that depend on the field only through the depletion edges, $W$ and $Q _ { n } ,$ differ little.

We evaluate the residuals of the Poisson and of the two continuity equations at fixed collocation points. Compared with a physics-informed network, the median residual of the data-only networks is larger by a factor of about 60 for the Poisson equation, about 3000 for the electron and about 1000 for the hole continuity equation, and is larger for more than 90% of the anchors.

We expect a larger difference to arise if the physics model contains process activated by the field, such as impact ionisation. This may favour physics-informed training and will be tested in the future.

![](images/abb46d0fdb55480625236928345a3da52194e55bf118008c7b9bda4d6dd4aea8.jpg)

![](images/c0c103824c6868404c3a3dd1fa025630b4a4cb3b2e783f8b91cfbcf38087fcb5.jpg)  
Figure 14. Distribution of the error of each (device, voltage) point of the test devices on a logarithmic axis, for the physics-informed and the data-only six-network ensembles. Left: relative $L _ { 1 }$ error of $\psi ;$ right: relative error of the peak field. Solid lines: median, dashed lines: 90th percentile of each set. The text above each panel gives the median and the ratio of the 90th percentile to the median. Without the physics term the typical error of $\psi$ is the same and the tail is heavier; the distribution of the error of the peak field is shifted to larger values.