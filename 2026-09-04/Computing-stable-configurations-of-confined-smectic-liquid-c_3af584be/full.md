# Computing stable configurations of confined smectic liquid crystals with a deep variational framework

Yuchen Xie,<sup>1</sup> Baoming Shi,<sup>2</sup> Yucen Han,<sup>3</sup> and Lei Zhang<sup>4,</sup> <sup>∗</sup>

<sup>1</sup>School of Mathematical Sciences, Peking University, Beijing 100871, China

<sup>2</sup>Department of Applied Physics and Applied Mathematics,

Columbia University, New York, NY 10027, USA

<sup>3</sup>Center for Applied Mathematics, Renmin University of China, Beijing 100872, China

<sup>4</sup>School of Mathematical Sciences, Beijing International Center for Mathematical Research,

Center for Quantitative Biology, Center for Machine Learning Research, Peking University, Beijing 100871, China

Smectic liquid crystals are layered liquid-crystalline phases characterized by orientational order and periodic density modulation. Although their structures can be modeled using continuum theories, computing stable configurations remains challenging in complex geometries, particularly when the high-frequency density modulations associated with smectic layering should be resolved. We pro pose a deep variational framework (DVF) for computing these configurations within the modified Landau–de Gennes model, in which the coupled orientational and positional order parameters are represented on a regular reference domain while physical confinement is incorporated through coordinate mappings. A warmup penalty mitigates the spectral bias of neural networks toward smooth, nonlayered fields, enabling robust recovery of oscillatory smectic states. Comparisons with a neuralnetwork baseline and finite-diference relaxation demonstrate the essential role of this penalty and the numerical stability of the resulting layered states. The DVF reproduces experimentally estab lished smectic-A defect structures and layer morphologies across diverse confinement geometries and further predicts a chevron-like smectic-C state in a tangent-anchored sphere. Together, these results demonstrate the applicability of the DVF to computing stable smectic configurations across experimentally relevant confinement geometries and anchoring conditions.

## I. INTRODUCTION

Liquid crystals are soft condensed matter systems that combine fluidity with partial molecular order [1, 2]. Among their ordered phases, smectic liquid crystals are of particular interest because, in addition to orientational alignment, they develop a one-dimensional density modulation that organizes the material into layers [3]. This layered order makes smectics especially sensitive to confinement, surface anchoring, and geometric frustration, since the preferred layer structure is often dificult to accommodate in bounded or curved domains [4–6]. As a result, confined smectics can exhibit a rich variety of layer morphologies, defect structures, and metastable states. Two representative phases are smectic-A (SmA), in which the average molecular orientation is parallel to the layer normal, and smectic-C (SmC), in which the molecules are collectively tilted with respect to the layer normal. This distinction can lead to qualitatively diferent confined structures in the two phases [7].

To describe confined smectic liquid crystals within a continuum framework, both orientational and positional order must be resolved, with the latter corresponding to the density modulation associated with layering. The Landau–de Gennes (LdG) theory represents orientational order through a tensor field Q [1], whereas smectic phases require an additional positional order parameter [8–10]. We therefore adopt the modified Landau–de

Gennes (mLdG) model [10–12], in which Q is coupled to a scalar positional field u. This formulation provides a unified variational description of molecular orientation, layered density structure, and their associated defects [10, 13]. Within this framework, the confined configurations of interest are characterized as local minimizers of the corresponding free-energy functional.

Computing these minima is challenging because three numerical dificulties occur simultaneously. The confinement may be irregular or three-dimensional; the mLdG energy contains second derivatives of u and therefore leads to a fourth-order Euler–Lagrange (EL) equation [11]; and the density field becomes strongly oscillatory when many smectic layers fit inside the domain [14, 15]. Established discretization methods can provide accurate solutions, but accommodating complex geometry, higherorder regularity, and fine layer scales within a single implementation can be technically demanding [10–12]. Neural variational methods ofer a complementary approach by representing the order parameters with trainable functions and minimizing the free energy directly. A standard neural variational method such as the Deep Ritz method (DRM), however, remains susceptible to spectral bias [16–18]. This bias causes neural networks to learn low-frequency components before high-frequency ones and may consequently lead the optimization to a nearly homogeneous, nonlayered branch.

To address these dificulties, we employ a deep variational framework (DVF) to compute stable layered configurations of the coupled mLdG model in complex do mains. The DVF is adapted from our previous work on the Landau–Brazovskii model [19]. The neural network serves as a variational representation of $( u , Q )$ , rather than as a surrogate trained on labeled data, and is evaluated on a regular reference domain. A coordinate mapping incorporates the physical confinement [20], separating the treatment of complex boundaries from the representation of the order parameters. To recover the highfrequency density modulation, a warmup penalty promotes the prescribed smectic wavelength during the early optimization and is subsequently removed, so that the final configuration is governed by the original mLdG energy.

Our numerical results show that the DVF recovers stable layered configurations across diverse confinement settings, reproducing established morphologies while revealing additional geometry-dependent structures. We first assess the method in a square by comparison with the DRM, an ablated DVF without the warmup penalty, and a subsequent finite-diference relaxation. We then apply it to SmA confinement in representative two- and threedimensional (2D and 3D) geometries. The computed SmA minima reproduce corner-localized defects reported in polygonal continuum studies [21, 22], bipolar textures observed in spherical confinement [23, 24], and concentric layers found in particle simulations under homeotropic anchoring [25, 26]. In irregular 2D domains, the calculations further relate polygonal corner types to interior layer junctions or corner-localized dislocations and show that tactoid elongation regularizes the interior layers. In 3D, anchoring designed to induce a toroidal focal conic domain (TFCD) produces a localized near-surface orderreconstruction state, while the tangent-anchored sphere supports chevron-like SmC layers [7, 27]. Repeated optimizations also recover competing configurations under the same macroscopic conditions, indicating a frustrated mLdG energy landscape. The model and method are introduced in Secs. II–III, followed by numerical results in Sec. IV and conclusions in Sec. V.

## II. MATHEMATICAL MODELS

To describe confined smectic phases, we adopt the mLdG framework [10–12], a phenomenological continuum model formulated in terms of two coupled order parameters: a symmetric, traceless nematic tensor $Q \in$ $\mathbf { \hat { \mathbb { R } } } ^ { d \times d }$ and a scalar positional field $u \in \mathbb { R }$ , where $d = 2 , 3$ The tensor Q characterizes local orientational order: its principal eigenvector defines the director n, while its eigenvalues measure the strength and possible biaxiality of the order [1, 28]. The field u measures the deviation of the molecular density from its spatial average, so its periodic modulation represents the alternating high- and low-density regions common to smectic phases. Locally, a modulation of the form $u ( { \pmb x } ) \sim \cos ( { \pmb k } \cdot { \pmb x } )$ describes layers with normal $k / | k |$ and spacing $2 \pi / | k |$ . The coupling terms introduced below encode this preferred smectic layering by selecting the wave number and determining the phase-dependent relation between the layer normal and

the director n.

We seek local minimizers of the total free-energy functional. Over a physical domain Ω, the total energy is given by

$$
E ( \boldsymbol { Q } , \boldsymbol { u } ) = \int _ { \Omega } \left( f _ { \mathrm { L d G } } ( \boldsymbol { Q } ) + f _ { \mathrm { b s } } ( \boldsymbol { u } ) + f _ { \mathrm { i n t } } ( \boldsymbol { Q } , \boldsymbol { u } ) \right) \mathrm { d } x .\tag{1}
$$

The integrand consists of three contributions: the nematic bulk and elastic energy $f _ { \mathrm { L d G } }$ , the smectic bulk energy $f _ { \mathrm { b s } }$ , and the coupling term $f _ { \mathrm { i n t } }$ linking orientational and positional ordering.

The first term in Eq. (1) is the nematic free-energy density,

$$
f _ { \mathrm { L d G } } ( \boldsymbol { Q } ) = \frac { K } { 2 } | \boldsymbol { \nabla } \boldsymbol { Q } | ^ { 2 } + f _ { \mathrm { b n } } ( \boldsymbol { Q } ) ,\tag{2}
$$

where $K > 0$ is the elastic constant and $f _ { \mathrm { { b n } } }$ is the local bulk potential governing the transition between isotropic and nematically ordered states. We use the standard LdG potential in 3D and its reduced counterpart in 2D. Their explicit forms and the corresponding equilibrium scalar order $s _ { + }$ are collected in Appendix A. The connection between orientational order and the smectic density modulation is introduced through $f _ { \mathrm { i n t } }$ below.

The second term in Eq. (1) is the bulk smectic freeenergy density associated with the positional order parameter $u ,$ defined as

$$
f _ { \mathrm { b s } } ( u ) = \frac { a } { 2 } u ^ { 2 } + \frac { b } { 3 } u ^ { 3 } + \frac { c } { 4 } u ^ { 4 } ,\tag{3}
$$

where $a = \alpha _ { 2 } ( T - T _ { 2 } ^ { * } )$ is a temperature-dependent coeficient, with $\alpha _ { 2 } > 0$ and $T _ { 2 } ^ { * } < T _ { 1 } ^ { * }$ associated with the nematic–smectic transition, while b and c are materialdependent constants with $c > 0$ . In the present work, we set $b = 0$ , so the bulk potential is invariant under $\iota \mapsto - u$ [9]. When $a > 0$ , the homogeneous state $u = 0$ is favored, whereas $a < 0$ favors a nonzero density-modulation amplitude. Thus, $f _ { \mathrm { b s } }$ controls the onset and amplitude of positional order.

The spatial organization of this positional order is governed by the third term $f _ { \mathrm { i n t } }$ , whose phase-dependent form selects the layer wavelength and its relation to $Q .$ . For the SmA phase, we take

$$
f _ { \mathrm { i n t } } ( \pmb { Q } , \boldsymbol { u } ) = B _ { 0 } \left| D ^ { 2 } \boldsymbol { u } + \boldsymbol { q } ^ { 2 } \left( \frac { \pmb { Q } } { s _ { + } } + \frac { \pmb { I } _ { d } } { d } \right) \boldsymbol { u } \right| ^ { 2 } ,\tag{4}
$$

whereas for the SmC phase, we use

$$
\begin{array} { l } { \displaystyle f _ { \mathrm { i n t } } ( Q , u ) = B _ { 0 } \Biggl ( ( \Delta u + q ^ { 2 } u ) ^ { 2 } } \\ { \displaystyle \phantom { \frac { 1 } { 2 } } + \left( \mathrm { t r } \left( D ^ { 2 } u \left( \frac { Q } { s _ { + } } + \frac { I _ { d } } { d } \right) \right) + q ^ { 2 } u \cos ^ { 2 } \theta _ { 0 } \right) ^ { 2 } \Biggr ) . } \end{array}\tag{5}
$$

Here $B _ { 0 } > 0$ weights the energetic penalty for deviations from the preferred geometric relation between the layer normal and the nematic director: alignment in the SmA phase and a prescribed tilt angle $\theta _ { 0 }$ in the SmC phase. Moreover, $D ^ { \hat { 2 } } u$ denotes the Hessian matrix of $u , q$ is the characteristic wave number of the smectic layers, $\mathbf { \delta } _ { I _ { d } }$ is the d-dimensional identity matrix, and $s _ { + }$ is the equilibrium scalar order selected by $f _ { \mathrm { { b n } } }$

The preferred layered structure follows directly from the phase-dependent coupling terms in Eqs. (4) and (5). For a uniaxial state and a local plane wave,

$$
Q = s _ { + } \left( n \otimes n - \frac { I _ { d } } { d } \right) , \qquad u ( x ) = u _ { 0 } \cos ( k \cdot x ) ,
$$

we have

$$
\frac { { \pmb Q } } { s _ { + } } + \frac { { \pmb I } _ { d } } { d } = { \pmb n } \otimes { \pmb n } , \qquad { \pmb D } ^ { 2 } { \pmb u } = - ( { \pmb k } \otimes { \pmb k } ) { \pmb u } .
$$

The SmA coupling is minimized locally when

$$
\pmb { k } = \pmb { q } \pmb { n } ,\tag{6}
$$

so it selects layers normal to the director with spacing $2 \pi / q$ . For SmC, the two coupling terms instead impose

$$
| { \pmb k } | = q , \qquad | { \pmb k } \cdot { \pmb n } | = q \cos \theta _ { 0 } ,\tag{7}
$$

which selects the same wavelength and an angle $\theta _ { 0 }$ between the director and layer normal. Thus, the layered modulation of u is selected by the mLdG energy rather than imposed externally.

Throughout, boundary anchoring constrains $Q _ { i }$ whereas u is varied freely and satisfies the natural boundary conditions of the mLdG energy. The conditions used in the computations are summarized in Appendix C.

Hereafter we work with the dimensionless free-energy functional and omit bars for notational simplicity. The dimensionless parameter λ is proportional to the physical confinement length. A density modulation with material wave number q has efective wave number $\lambda q$ on the normalized domain and dimensionless layer spacing $2 \pi / ( \lambda q )$ . At fixed $q ,$ increasing λ places more layers in the same normalized domain, so u becomes more oscillatory and requires higher numerical resolution. The complete rescaling and the dimensionless SmA and SmC energies are given in Appendix A.

## III. DEEP VARIATIONAL FRAMEWORK

The mLdG model poses several computational challenges, including complex confinement geometries, higher-order derivatives, and strongly oscillatory smectic layers. To address them, we adapt the DVF to represent the coupled order parameters on a regular reference domain, incorporate physical confinement through coordinate mappings, and use a Fourier-based architecture with a wavelength-guided warmup penalty to facilitate the recovery of layered states. The overall framework is summarized in Fig. 1 and detailed below.

## A. Neural representation of the order parameters

We use a neural network as a trial function for the mLdG order parameters and determine its parameters directly through free-energy minimization. The network is evaluated on a regular reference domain $\widehat \Omega$ with coordinate $\xi ,$ and a hat denotes a field represented on this domain. Specifically, the network outputs

$$
( { \widehat { u } } _ { \theta } , { \widehat { Q } } _ { \theta } ) = { \mathcal { N } } _ { \theta } ( \pmb { \xi } ) ,\tag{8}
$$

where θ denotes the collection of all trainable network parameters. The corresponding fields on the physical domain are defined through the coordinate mapping by

$$
u _ { \theta } ( \pmb { x } ) = \widehat { u } _ { \theta } ( \mathcal { G } ^ { - 1 } ( \pmb { x } ) ) , \qquad \pmb { Q } _ { \theta } ( \pmb { x } ) = \widehat { \pmb { Q } } _ { \theta } ( \mathcal { G } ^ { - 1 } ( \pmb { x } ) ) .\tag{9}
$$

The coordinate mapping in Eq. (9) is specified in Sec. III B. Substituting $( u _ { \theta } , Q _ { \theta } )$ into the mLdG energy replaces the original field minimization by a finitedimensional optimization over θ. The loss used to carry out this optimization, including its auxiliary terms, is specified in Sec. III C; the optimized network represents the fields of a computed local energy minimum.

To construct these trial fields, we adapt the Fourierbased representation of the DVF developed in Ref. [19] to the mLdG order parameters. Specifically, the network architecture is

$$
\mathcal { N } _ { \theta } = \mathcal { Q } \circ \mathcal { H } _ { L } \circ \cdots \circ \mathcal { H } _ { 1 } \circ \mathcal { P } ,
$$

where $\mathcal { P }$ embeds the input coordinates to a higherdimensional feature space, $\mathcal { H } _ { \ell }$ is the ℓth Fourier layer, and $\mathcal { Q }$ projects the final features to $( \widehat { u } _ { \theta } , \widehat { Q } _ { \theta } )$ . Following the Fourier Neural Operator design [29], each Fourier layer combines spectral convolution with a pointwise transformation and nonlinear activation. This architecture provides a Fourier-based representation of the positional and orientational fields. The projection produces $\widehat { u } _ { \theta }$ and a symmetry-preserving parameterization of ${ \widehat { Q } } _ { \theta } ;$ the explicit 2D and 3D forms are given in Eq. (A1).

## B. Coordinate mapping for complex domains

To treat diferent confinement geometries within a common computational setting, we relate the physical domain Ω to a regular reference domain $\widehat \Omega$ through a coordinate transformation. Following the strategy of Phy-GeoNet [20], we let $\pmb { x } \in \Omega$ and ${ \pmb \xi } \in \widehat { \Omega }$ denote the physical and reference coordinates, respectively, and introduce a smooth one-to-one map ${ \pmb x } = { \mathcal G } ( { \pmb \xi } )$ from $\widehat \Omega$ to Ω. Equivalently, $\mathcal { G } ^ { - 1 }$ pulls the problem on the complex physical domain back to the regular reference domain. The neural network is then evaluated on a uniform grid in ${ \widehat { \Omega } } ,$ while the geometry of Ω is incorporated through G.

All quantities required by the mLdG energy are evaluated through this mapping. Volume integrals are transformed to Ω with the corresponding Jacobian factor, andb the gradients of Q and the higher-order derivatives of u are first computed on the reference grid and then converted to physical derivatives by the chain rule. The free energy can therefore be evaluated on the regular reference domain and diferentiated with respect to the network parameters during optimization. The explicit derivative transformations in 2D and 3D are given in Appendix B.

![](images/6a70291923f4d65c20cf0c030a5f2da260f3244403dbf30035f0116b49caea60.jpg)  
FIG. 1. Schematic of the deep variational framework (DVF) for computing stable configurations of confined smectic liquid crystals. A physical point $\pmb { x } \in \Omega$ is mapped to the reference coordinate $\pmb { \xi } \in \widehat { \Omega }$ through ${ \mathcal { G } } ^ { - 1 } ;$ , and the neural network $\mathcal { N } _ { \theta }$ is evaluated on a uniform reference grid to produce $( \widehat { u } _ { \theta } , \widehat { Q } _ { \theta } )$ . Symmetry and tracelessness of ${ \widehat { Q } } _ { \theta }$ are imposed by construction. The inset shows the Fourier-based architecture, consisting of an embedding layer, stacked Fourier layers, and a projection layer. The core loss combines the normalized modified Landau–de Gennes (mLdG) free energy $\mathcal { L } _ { \mathrm { e n e r g y } }$ , boundary-condition penalty ${ \mathcal { L } } _ { \mathrm { b c } } ,$ Euler–Lagrange (EL) residual regularization $\mathcal { L } _ { \mathrm { E L } }$ , and warmup penalty ${ \mathcal { L } } _ { \mathrm { p } }$

## C. Free-energy minimization and warmup penalty

Using the neural representation and coordinate mapping above, we optimize the network parameters within a Deep Ritz formulation [16], in which the mLdG problem is treated as free-energy minimization over the neural trial space. The total loss is

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { e n e r g y } } + w _ { \mathrm { b c } } \mathcal { L } _ { \mathrm { b c } } + w _ { \mathrm { E L } } \mathcal { L } _ { \mathrm { E L } } + w _ { \mathrm { p } } ( t ) \mathcal { L } _ { \mathrm { p } } .\tag{10}
$$

Here $\mathcal { L } _ { \mathrm { e n e r g y } }$ is the normalized mLdG free energy and remains the primary variational objective. The term ${ \mathcal { L } } _ { \mathrm { b c } }$ enforces anchoring of Q through a squared Dirichlet mismatch in most cases and a degenerate planar surface potential for 3D tangent anchoring. The explicit forms and their discrete implementation are given in Appendix C.

In a standard variational formulation, the free energy together with the prescribed boundary conditions defines the minimization problem. In our discrete neural implementation, however, minimizing only the energy and boundary penalty can admit grid-scale artifacts localized near defects. Such configurations may have an anomalously low sampled energy while remaining far from a stationary point of the continuum functional. We therefore include the EL residual $\mathcal { L } _ { \mathrm { E L } }$ , defined in Appendix C, as a small-weight stationarity regularization. It suppresses these under-resolved configurations without replacing the energy as the selection criterion; an unstable critical point can also have a small EL residual.

The principal auxiliary term for recovering the layered structure of the smectic is the warmup penalty ${ \mathcal { L } } _ { \mathrm { p } } .$ The inherent spectral bias of the network [17] makes a smooth, nearly homogeneous u an easily accessible basin during early training, even though the mLdG energy favors modulation at the wave number $\lambda q .$ The timedependent term $w _ { \mathrm { p } } ( t ) \mathcal { L } _ { \mathrm { p } }$ promotes this high-frequency content and is defined by

$$
\mathcal { L } _ { \mathrm { p } } = \left( \sum _ { v \in \mathcal { V } } ⨏ _ { \mathbb { S } ^ { d - 1 } } \bigg | \int _ { \Omega } v ( x ) \psi _ { \omega } ( x ) \mathrm { d } x \bigg | \mathrm { d } \omega - C _ { \mathrm { p } } \right) ^ { 2 } .\tag{11}
$$

Here $\mathcal { V } ~ = ~ \{ u \} ~ \cup ~ \left\{ \lambda ^ { - 2 } \frac { \partial ^ { 2 } u } { \partial x _ { i } \partial x _ { j } } \right\} _ { 1 \leq i \leq j \leq d }$ , and $\psi _ { \omega } ( { \pmb x } ) = $ cos $\left( q \lambda \omega \cdot x \right)$ , where ω $\in \mathbb { S } ^ { d - 1 }$ is a sampled direction and $C _ { \mathrm { p } } { \dot { > } } 0$ sets the target projection strength. Averaging over ω avoids selecting a preferred layer orientation, and the nonzero target $C _ { \mathrm { p } }$ makes the nonlayered field $u \equiv 0$ unfavorable during warmup. Since $w _ { \mathrm { p } } ( t )  0$ as $t \to \infty$ (see Appendix D), the final state is governed solely by the mLdG model. The role of the warmup penalty is tested by the ablation study in Sec. IV A, and the loss weights and schedules are reported in Appendix D.

## IV. NUMERICAL RESULTS

We now examine mLdG energy minima across a range of 2D and 3D confinements and anchoring conditions. The frustrated mLdG energy landscape supports multiple competing minima under the same physical conditions, as reflected by the distinct configurations obtained from independent initializations. For each parameter setting, we present the minimum with the lowest computed energy, while recognizing that the sampled states do not exhaust the full landscape. To compare the efects of geometry and anchoring within a common ordered regime, we fix $a = - 5 , b = 0 , c = 5 , q = 2 \pi$ , and $B _ { 0 } = \mathrm { 1 0 ^ { - 3 } }$ for $u ,$ and take $B = 6 4 0 0$ and $C = 3 5 0 0$ for $Q .$ . The 2D and 3D computations use the reduced and standard LdG bulk potentials, respectively, with their quadratic coeficients chosen to give the same equilibrium scalar order $s _ { + } = B / C \ [ 1 3 ]$ . Their explicit forms and parameter choices are given in Appendix A. Most results are reported at $\lambda ^ { 2 } = 3 0$ and 50; Appendix D gives the complete numerical settings.

![](images/168e07b3117b285b993173fa334ea99ca7f9c9f8d64c213bf7979f065c4d8a68.jpg)  
FIG. 2. Validation and ablation for square-confined smectic-A (SmA) at $\lambda ^ { 2 } = 3 0$ under tangent anchoring. (a) Orientational fields Q (top) and positional fields u (bottom) obtained with the Deep Ritz method (DRM), the deep variational framework (DVF) without the warmup penalty (DVF $\mathrm { w } / \mathrm { o } )$ , the full DVF, and finite-diference (FD) relaxation initialized from the full DVF solution. White segments show the director, while the background gives the scalar orientational order ${ \sqrt { \operatorname { t r } ( Q ^ { 2 } ) / 2 } } ;$ dark blue indicates reduced order. The DRM and DVF $\mathrm { w / o }$ yield nonlayered configurations with well order-reconstruction solution (WORS)-type orientational textures. The full DVF recovers the layered state and its D-like Q texture, both of which are retained after FD relaxation. (b) Pointwise diferences $\lvert u - u ^ { * } \rvert$ $| q _ { 1 } - q _ { 1 } ^ { * } |$ , and $| q _ { 2 } - q _ { 2 } ^ { * } |$ between the full DVF and FD-relaxed minima, where q<sub>1</sub> and q<sub>2</sub> are the independent components of the 2D tensor. The rightmost colorbar is shared by $q _ { 1 }$ and $q _ { 2 }$

## A. Validation and SmA minima in square confinement

We first verify our framework for SmA in the square confinement $\Omega \doteq [ - 1 , 1 ] ^ { 2 }$ under strong tangent anchoring (i.e., with the director parallel to each edge), which is a representative physical setting [13, 30, 31]. The validation is performed at $\lambda ^ { 2 } = 3 0$ . The neural comparison includes the standard DRM as a baseline, the DVF without the warmup penalty $\mathrm { ( D V F ~ w / o ) }$ as a direct ablation, and the full DVF. All three calculations use the same physical parameters and comparable optimization budgets. After the full DVF has converged, its fields are used to initialize a finite-diference (FD) relaxation. The resulting FD-relaxed state is then compared with the DVF solution in both morphology and energy.

TABLE I. Normalized free energy $\varepsilon$ and EL-residual measure L<sub>EL</sub> for square-confined SmA at $\lambda ^ { 2 } = 3 0$ under tangent anchoring, computed with the three neural settings and with FD relaxation initialized from the full DVF solution.
<table><tr><td>DRM</td><td>DVF  $\mathrm { w / o }$  DVF (Ours) FD-relaxed</td></tr><tr><td>ε  $- 1 . 2 7 6 7$ </td><td> $- 1 . 3 4 8 3$   $- 3 . 0 3 3 6$   $- 3 . 0 3 5 8$ </td></tr><tr><td>EL residual  $1 . 4 \times 1 0 ^ { 1 }$ </td><td> $1 . 3 \times 1 0 ^ { - 2 }$   $8 . 9 \times 1 0 ^ { - 2 }$   $1 . 2 \times 1 0 ^ { - 1 8 }$ </td></tr></table>

The qualitative efect of the warmup penalty is visible in Fig. 2(a). Neither the DRM nor DVF $\mathrm { w / o }$ develops the high-frequency modulation required for smectic layering. Their orientational fields instead resemble the well order-reconstruction solution (WORS), a characteristic configuration of a square-confined nematic under tangent anchoring in which two diagonal low-order lines divide the square into four sectors with approximately uniform director orientations [32, 33].

By contrast, the full DVF recovers a well-resolved oscillatory density field and an associated diagonal (D-like) $Q$ texture, in which the director is predominantly aligned with one diagonal of the square. Because DVF $\mathrm { w / o }$ and the full DVF difer only by the warmup penalty, their distinct outcomes isolate its role in the optimization. The penalty moves the early iterations away from the nearly homogeneous basin and toward the layered smectic branch, while its eventual removal ensures that it does not alter the final minimization objective.

The energy and EL residual in Table I distinguish the branches reached by the three neural calculations. The DRM has both a high energy and a large residual, whereas DVF $\mathrm { w / o }$ attains a small residual but remains on a higher-energy nonlayered branch. The latter is therefore close to a stationary nonlayered configuration, while the full DVF reaches the substantially lower-energy layered branch. This comparison shows that the warmup penalty primarily changes the basin accessed during optimization rather than simply reducing the residual.

The FD relaxation initialized from the DVF configuration preserves the layered D-like morphology and converges to a stationary state whose Hessian is positive definite. This confirms that the DVF recovers the basin of a stable layered minimum. Using the FD-relaxed state as the reference, Fig. 2(b) shows localized diferences in Q and phase-sensitive diferences in the oscillatory field $u ,$ while the overall D-like layered morphology is retained.

We next study the minima in square confinement at $\lambda ^ { 2 } = 1$ , 4.38, 30, and 100, which resolve the structural transition produced by increasing the efective confinement size. As shown in Fig. 3, the $\lambda ^ { 2 } = 1$ minimum is WORS-like: two line defects divide the square into four sectors whose directors follow the adjacent edges, while u remains nearly homogeneous. At $\bar { \lambda ^ { 2 } } = 4 . 3 8$ , the line defects begin to reconstruct, and a layered modulation appears. The minima at $\lambda ^ { 2 } ~ = ~ 3 0$ and 100 are instead D-like, with a predominantly diagonal bulk director and no extended defect lines across the interior. This transition reflects a change in the relative cost of accommodating the tangent boundary condition. At small λ, spatial variations carry a high elastic cost relative to bulk ordering, so the boundary alignment influences the entire square and the WORS-like profile provides a lowdistortion interpolation between the four edges [32]. As λ increases, retaining two line defects across the interior becomes unfavorable, and the D-like minima become energetically favorable because they eliminate the interior defects and preserve strong orientational and positional order through most of the bulk [13].

![](images/df625921820f55f07a3b0f4d4615ecd65fe3bb0b6e75cc30d122d62098c62a5a.jpg)  
FIG. 3. SmA minima in a square under tangent anchoring for $\lambda ^ { 2 } = 1 , 4 . 3 8$ , 30, and 100. Each column shows Q (top) and u (bottom). The sequence changes from a WORS-like nonlayered minimum, through the onset of layering and reconstruction of the diagonal low-order lines, to D-like layered minima. The visualization and color mapping are the same as in Fig. 2.

## B. SmA in 2D irregular domains

We next consider 2D irregular domains under tangent anchoring, as shown in Fig. 4. The equilateral triangle, regular pentagon, regular hexagon, and disk are studied at $\lambda ^ { 2 } = 3 0$ and 50, providing a progression from sharply cornered polygons to a smooth boundary. The minima in polygonal confinements are distinguished by how tangent anchoring is accommodated at splay and bend corners. In the equilateral triangle (Fig. 4(a)), all three vertices are splay-like, and the resulting boundary rotation is balanced by an interior $- 1 / 2$ defect [21, 34]. This low-order core anchors a threefold layer junction, while the layers remain nearly regular along the three edges. The minima in pentagonal and hexagonal confinements (Figs. 4(b) and 4(c)) belong to a diferent class: two vertices are splay-like, and the remaining vertices are bendlike [21, 22]. The orientational mismatch is thereby expelled from the interior and concentrated at the two splay corners. The layer structure responds by terminating or inserting layers near these corners, producing bound ary dislocations while preserving a nearly ideal layer arrangement in the central region. These two organizations persist at both values of $\lambda ^ { \overset { \triangledown } { 2 } }$ , although symmetry-related choices of the splay corners lead to diferently oriented minima.

![](images/82a47af30aa0d1a51cb8a75f91bb7bd28517848756329505c34b7c189caa5a84.jpg)  
FIG. 4. SmA minima in 2D irregular domains under tangent anchoring: (a) equilateral triangle, (b) regular pentagon, (c) regular hexagon, (d) disk, (e) wide spindle, and (f) narrow spindle. Rows (a)–(d) show $\lambda ^ { 2 } = 3 0$ (left) and 50 (right), whereas rows (e) and (f) show only $\lambda ^ { 2 } = \mathrm { 3 0 }$ Within each group, Q is shown on the left and u on the right. The triangle contains an interior orientational defect and a layer triple junction, whereas the pentagon and hexagon concentrate orientational defects and layer dislocations near selected corners while retaining comparatively regular interior layers. The disk and spindles develop bipolar organizations, with the defects approaching the tips as the spindle narrows. The color mapping is the same as in Fig. 2.

The disk (Fig. 4(d)) provides a smooth-boundary counterpart to these polygonal confinements. In the absence of corners that can accommodate the rotation required by tangent anchoring, the orientational field adopts a bipolar configuration with two boundary defects at opposite poles, while the smectic layers bend continuously between them. This confinement-induced localization of orientational and positional defects is consistent with experimental and simulation studies of confined smectics [6, 14, 15, 35].

To further explore the role of boundary geometry in organizing defects and smectic layers, we turn to the spindle-shaped domains in Figs. 4(e) and 4(f), which connect the preceding confinement geometries to tactoid-like shapes. Such pointed domains commonly arise during isotropic–nematic coexistence, where elasticity, anchoring, and anisotropic interfacial energy jointly determine the droplet shape and director field [36, 37]. The wide spindle in Fig. 4(e) exhibits a bipolar-like defect pair accompanied by pronounced curvature and distortion of the interior layers. As the spindle becomes narrower (Fig. 4(f)), regions of strong director variation become increasingly localized near the high-curvature tips, allowing the elongated central region to sustain a nearly uniform director and straighter, more regularly spaced layers. This comparison suggests that pointed confinement can concentrate orientational frustration near the tips while promoting a more uniformly ordered smectic interior.

## C. SmA in 3D confinements

We next extend the analysis of confined SmA minima from two to three dimensions and examine how the equilibrium structures respond to changes in confinement geometry and surface anchoring. We begin with tangent anchoring in the cube and sphere (Figs. 5(a) and 5(c)). In the cube, the director is approximately aligned with a body diagonal throughout much of the interior, whereas stronger distortions develop near the boundary. A nearly uniform bulk orientation cannot satisfy tangent anchoring simultaneously on all six faces, and the resulting mismatch is therefore accommodated primarily near the edges and corners. Correspondingly, the smectic layers remain comparatively regular in the interior and become distorted mainly near the faceted boundary. This localization of orientational distortion is consistent with the multistable tangent-anchored textures reported for nematic cuboids [38]. The spherical domain exhibits a different organization. Under tangent anchoring, the Q field adopts a bipolar texture with two antipodal low-order regions, while the smectic layers form a smoothly curved stack connecting the two poles. The strongest distortions are concentrated near these polar regions. In contrast to the cube, the smooth spherical boundary provides no edges or corners at which the anchoring mismatch can be localized, and the frustration is instead organized around the bipolar defect pair. Similar bipolar structures have been observed in confined nematic droplets and in particle simulations of spherical smectics [23, 24, 39].

Changing the spherical boundary condition from tangent to homeotropic anchoring leads to a qualitatively diferent minimum (Fig. 5(d)). The director is now approximately radial throughout the domain. Because the SmA layer normal preferentially follows the director, the smectic layers close into concentric, onion-like shells rather than forming the curved bipolar stack found un der tangent anchoring. Similar radial orientational order and spherical SmA layering have been reported in particle simulations [25, 26]. The comparison between Figs. 5(c) and 5(d) therefore shows that the same spherical confinement can support markedly diferent layered organizations: tangent anchoring produces a bipolar texture with distortions concentrated near two surface regions, whereas homeotropic anchoring favors a radially organized state with closed concentric layers.

![](images/ccf7bf225f7dd3909865da8b6de65c7759f156060a5271efdb4d7dfab70c05b0.jpg)  
FIG. 5. SmA minima in 3D under diferent geometries and anchoring conditions: (a) cube with tangent anchoring, (b) cube with TFCD-inducing anchoring, (c) sphere with tangent anchoring, and (d) sphere with homeotropic anchoring. In each row, the left and right groups correspond to $\lambda ^ { 2 } = 3 0$ and 50, respectively, and show Q together with u. White director lines are superimposed on the biaxiality $\beta = 1 - 6 ( \mathrm { t r } Q ^ { 3 } ) ^ { 2 } / ( \mathrm { t r } Q ^ { 2 } ) ^ { 3 }$ ; regions of enhanced biaxiality, which highlight defect cores and order-reconstruction regions, appear red. Smectic layers are represented by zero isosurfaces of $u .$ The four cases exhibit corner- and edge-localized distortions, a near-substrate order-reconstruction layer, bipolar curved layers, and concentric shells, respectively.

We finally consider the competing anchoring conditions associated with toroidal focal conic domains (TFCDs). In a canonical TFCD, the smectic layers form toroidal structures organized around axial and circular focal lines, mediating between vertical alignment in the bulk and radial in-plane alignment at the substrate [10, 40, 41]. To probe whether a related structure emerges in the present model, we impose vertical alignment at the top surface of the cube and radial alignment at the bottom (Fig. 5(b)). In both computed minima, the upper portion of the domain retains an almost vertical director together with a nearly uniform stack of horizontal layers. The reorientation required by the bottom boundary remains confined to a narrow region adjacent to the substrate rather than extending through the full height of the cube. Within this near-surface region, the magnitude of Q decreases and its biaxiality increases around a localized low-order core, indicating a biaxial order-reconstruction zone [42]. The smectic layers simultaneously develop focal-conic-like curvature around the core, thereby providing a local transition between the radial bottom alignment and the horizontal bulk stack while leaving the upper layers nearly undistorted. This near-substrate reconstruction is also observed in computations at smaller system sizes and persists after FD relaxation.

## D. SmC in spherical confinement

SmC difers from SmA by a finite preferred angle between the director and the layer normal. In conventional confined SmC cells, the onset of this tilt is often accompanied by a reduction in the equilibrium layer spacing across the SmA–SmC transition. When such contraction is frustrated by surface constraints, the layers can reorganize into oppositely tilted segments, giving rise to the characteristic chevron structures observed experimentally and in molecular simulations [43, 44]. Here we separate this conventional layer-contraction mechanism from the efect of the director–layer tilt itself. We therefore retain the same spherical domain, tangent anchoring, material parameters, and preferred wave number q as in the corresponding SmA calculation, but impose a preferred angle $\theta _ { 0 } = \pi / 6$ through the SmC coupling [12].

The resulting minimum is shown in Fig. 6. Compared with the tangent-anchored SmA state in Fig. 5(c), the large-scale orientational organization changes only weakly: both states exhibit a bipolar Q texture with two antipodal low-order regions. Their layer morphologies, however, are markedly diferent. The SmA state consists of a smoothly curved, approximately parallel stack, whereas the SmC layers undergo pronounced bending and reorientation near the spherical boundary, producing a surface-localized chevron-like pattern. Because the two calculations have the same preferred wave number q, this restructuring cannot be attributed to an imposed change in the equilibrium layer spacing and instead originates from the preferred SmC director–layer tilt.

This distinction can be understood geometrically. Tangent anchoring constrains the director to the local tangent plane of the sphere, while the SmC coupling favors a finite angle $\theta _ { 0 }$ between the director and the layer normal. Since the tangent plane varies continuously over the curved boundary, a single SmA-like layer stack cannot accommodate this preferred angle everywhere without introducing additional distortion. The incompatibility is therefore relieved primarily through spatially varying bending and reorientation of the layers near the surface, while the bipolar orientational texture is largely retained. The resulting structure is thus chevron-like in morphol ogy, but difers from the conventional planar chevron mechanism driven by layer contraction. This curvatureinduced response is qualitatively consistent with studies of tangent-anchored smectic shells, where SmC tilt similarly provides an additional route for accommodating geometric frustration through layer bending [7, 27].

![](images/8f626cebbe4961e3494cb560882f147d8f70a1cc7bce793d5f3fedcb7fdcd0ac.jpg)  
FIG. 6. SmC minimum in a spherical domain under tangent anchoring at $\theta _ { 0 } = \pi / 6$ . The Q field is shown by white director lines superimposed on a biaxiality map, and the layers are represented by zero isosurfaces of u. The bipolar orientational texture is accompanied by surface-localized, chevronlike bending of the layers.

## V. DISCUSSION AND CONCLUSION

This work develops a unified computational framework for stable configurations of confined smectics based on an adapted DVF that combines a Fourier representation, coordinate mappings, and a warmup penalty for wavelength guidance. The square-domain ablation shows that the warmup stage is essential for reliably accessing layered mLdG states in the tested regime, while subsequent FD relaxation confirms that the recovered configurations persist as minima under an independent spatial discretization.

Across the computed states, confinement geometry and anchoring organize the large-scale Q field, while director–layer coupling links this orientational organization to the resulting layer morphology. The calculations recover several established structures, including WORS-like and D-like square states, bipolar spherical textures, and TFCD-like and chevron-like morphologies [10, 13, 32, 43]. These results demonstrate how geometry, anchoring, and director–layer coupling jointly con trol defect localization and layer morphology in confined smectics.

The present study is limited to static states in fixed domains and to a finite range of parameters and boundary conditions. The DVF could be adapted to other freeenergy models with spatially modulated order, including those for cholesteric liquid crystals [45] and other layered materials. Its geometric scope could likewise be extended from the solid confinements considered here to shell geometries with inner and outer boundaries, particularly spherical shells [7, 27]. Beyond prescribed boundaries, difuse-interface LdG models already couple nematic order to a deformable interface and recover equilibrium droplet morphologies [46, 47]. A future neural variational formulation could represent both the order parameters and the free boundary by trainable functions and determine them simultaneously through minimization of a common free-energy objective.

## ACKNOWLEDGMENTS

L.Z. was supported by the National Natural Science Foundation of China (No. 12225102, T2321001, and 12288101) and the National Key Research and Development Program of China 2024YFA0919500. Y.H. was supported by Young Faculty Development Fund of the School of Mathematics at Renmin University of China and the Renmin University of China New Faculty’s fund.

## DATA AVAILABILITY

The codes that support the findings of this article are available from the authors upon reasonable request.

## Appendix A: Nematic bulk potentials and nondimensionalization

We first record the tensor parameterizations and dimension-dependent nematic bulk potentials used in the computations. Symmetry and tracelessness are enforced by writing

$$
Q ^ { \mathrm { 2 D } } = \binom { q _ { 1 } \quad q _ { 2 } } { q _ { 2 } \ : - q _ { 1 } } ,
$$

$$
\begin{array} { r } { Q ^ { \mathrm { 3 D } } = \left( { \begin{array} { c c c } { q _ { 1 } - q _ { 3 } } & { q _ { 2 } } & { q _ { 4 } } \\ { q _ { 2 } } & { - q _ { 1 } - q _ { 3 } } & { q _ { 5 } } \\ { q _ { 4 } } & { q _ { 5 } } & { 2 q _ { 3 } } \end{array} } \right) . } \end{array}\tag{A1}
$$

Thus, the network uses two scalar fields for $Q$ in 2D and five in 3D. In 3D, the standard LdG potential is

$$
\begin{array} { c c c } { { f _ { \mathrm { b n } } ^ { \mathrm { 3 D } } ( \pmb Q ) = \displaystyle \frac { A } { 2 } \mathrm { t r } ( \pmb Q ^ { 2 } ) - \displaystyle \frac { B } { 3 } \mathrm { t r } ( \pmb Q ^ { 3 } ) + \displaystyle \frac { C } { 4 } \big ( \mathrm { t r } ( \pmb Q ^ { 2 } ) \big ) ^ { 2 } , } } \\ { { s _ { + } ^ { \mathrm { 3 D } } = \displaystyle \frac { B + \sqrt { B ^ { 2 } - 2 4 A C } } { 4 C } . } } \end{array}
$$

Here $A = \alpha _ { 1 } ( T - T _ { 1 } ^ { * } )$ is temperature dependent, while $B , C > 0$ are material parameters. In 2D, $\mathrm { t r } ( Q ^ { 3 } ) = 0 .$ and choosing $A = - B ^ { 2 } \dot { / } ( 2 C )$ reduces the bulk potential to

$$
f _ { \mathrm { b n } } ^ { \mathrm { 2 D } } ( \pmb { Q } ) = - \frac { B ^ { 2 } } { 4 C } \mathrm { t r } ( \pmb { Q } ^ { 2 } ) + \frac { C } { 4 } \big ( \mathrm { t r } ( \pmb { Q } ^ { 2 } ) \big ) ^ { 2 } , \quad s _ { + } ^ { \mathrm { 2 D } } = \frac { B } { C } .
$$

The reduced potential is used in all 2D computations. Its diference from the 3D bulk potential results from reducing the tensor itself to the two-component form above, not merely from restricting the spatial domain to two dimensions. Since the chosen value of A has already been substituted, it does not appear as an independent coefi cient in this expression. The resulting quartic potential is the standard reduced Landau–de Gennes (rLdG) form used for 2D nematic equilibria [21].

We next summarize the rescaling used to obtain the dimensionless energies in the main text. Let L be the characteristic physical length of Ω, and set $\bar { \pmb x } = \pmb x / L$ , so that $\bar { \Omega } = L ^ { - 1 } \dot { \Omega } , \dot { \nabla } = L ^ { - 1 } \ddot { \nabla } , \Delta = L ^ { - 2 } \bar { \Delta } , D ^ { 2 } = L ^ { - 2 } \bar { D } ^ { 2 }$ and $\mathrm { d } \boldsymbol { x } = L ^ { d } \boldsymbol { \dot { \mathrm { C } } }$ x¯. Measuring the free energy in units of $K L ^ { d - 2 }$ gives $\bar { E } = E / ( K L ^ { \breve { d } - 2 } )$ We further introduce $\bar { \lambda } ^ { 2 } = 2 C \bar { L } ^ { 2 } / K , \bar { a } = a / ( 2 C ) , \bar { c } = c / ( 2 C ) , \bar { q } ^ { 2 } = K q ^ { 2 } / ( 2 C )$ and $\bar { B } _ { 0 } = 2 { B } _ { 0 } C / K ^ { 2 }$ , and set $b = 0$ as in the main text.

Substituting these relations into the dimensional freeenergy functional yields the following dimensionless energies for the SmA and SmC phases:

$$
\begin{array} { r l } & { \bar { E } _ { \mathrm { S m A } } ( \pmb { Q } , u ) = \displaystyle \int _ { \bar { \Omega } } \bigg \{ \bar { \lambda } ^ { 2 } \big [ f _ { u } ( u ) + f _ { Q } ( \pmb { Q } ) \big ] + \frac { 1 } { 2 } | \bar { \nabla } \pmb { Q } | ^ { 2 } } \\ & { \qquad + \displaystyle \frac { \bar { B } _ { 0 } } { \bar { \lambda } ^ { 2 } } \bigg | \bar { D } ^ { 2 } u + \bar { \lambda } ^ { 2 } \bar { q } ^ { 2 } \left( \frac { \pmb { Q } } { s _ { + } } + \frac { \pmb { I } _ { d } } { d } \right) u \bigg | ^ { 2 } \bigg \} \mathrm { d } \bar { x } , } \end{array}\tag{A2}
$$

$$
\begin{array} { l } { \displaystyle \bar { E } _ { \mathrm { S m C } } ( Q , u ) = \int _ { \bar { \Omega } } \Biggl \{ \bar { \lambda } ^ { 2 } \big [ f _ { u } ( u ) + f _ { Q } ( Q ) \big ] + \frac { 1 } { 2 } | \bar { \nabla } Q | ^ { 2 } } \\ { \displaystyle \qquad + \frac { \bar { B } _ { 0 } } { \bar { \lambda } ^ { 2 } } \big ( \bar { \Delta } u + \bar { \lambda } ^ { 2 } \bar { q } ^ { 2 } u \big ) ^ { 2 } } \\ { \displaystyle \qquad + \frac { \bar { B } _ { 0 } } { \bar { \lambda } ^ { 2 } } \Biggl [ \mathrm { t r } \left( \bar { D } ^ { 2 } u \left( \frac { Q } { s _ { + } } + \frac { I _ { d } } { d } \right) \right) + \bar { \lambda } ^ { 2 } \bar { q } ^ { 2 } u \cos ^ { 2 } \theta _ { 0 } \Biggr ] ^ { 2 } \Biggr \} \mathrm { d } \bar { x } , } \end{array}
$$

where

$$
\begin{array} { c } { { f _ { u } ( u ) = \displaystyle { \frac { \bar { a } } { 2 } } u ^ { 2 } + \displaystyle { \frac { \bar { c } } { 4 } } u ^ { 4 } , } } \\ { { f _ { Q } ( { \pmb Q } ) = \displaystyle { \frac { A } { 4 C } } \mathrm { t r } ( { \pmb Q } ^ { 2 } ) - \displaystyle { \frac { B } { 6 C } } \mathrm { t r } ( { \pmb Q } ^ { 3 } ) + \displaystyle { \frac { \bigl [ \mathrm { t r } \bigl ( { \pmb Q } ^ { 2 } \bigr ) \bigr ] ^ { 2 } } { 8 } } . } } \end{array}\tag{A3}
$$

The displayed nematic bulk term is used in 3D. For the 2D rLdG computations, it is replaced by

$$
- \frac { B ^ { 2 } } { 8 C ^ { 2 } } \mathrm { t r } ( { \bf { \cal Q } } ^ { 2 } ) + \frac { \left( \mathrm { t r } ( { \bf { \cal Q } } ^ { 2 } ) \right) ^ { 2 } } { 8 } ,
$$

which is the dimensionless form of $f _ { \mathrm { b n } } ^ { \mathrm { 2 D } }$ . The implementation minimizes the further normalized energy

$$
\mathcal { E } = \frac { \bar { E } } { \bar { \lambda } ^ { 2 } } .
$$

For each fixed $\bar { \lambda } ,$ this positive rescaling leaves the energy minima unchanged and keeps the reported values on a comparable numerical scale over the range of $\bar { \lambda }$ considered here. Bars are dropped in the main text, and $\mathcal { L } _ { \mathrm { e n e r g y } } = \mathcal { E }$ is used in all loss functions and reported energy values.

## Appendix B: Coordinate mapping and transformed derivatives

To treat irregular confinement geometries within a unified computational setting, we map the regular reference domain $\widehat { \Omega } = [ 0 , 1 ] ^ { d }$ , with $d = 2 , 3 ,$ , to the physical domain Ω through an invertible transformation

$$
\begin{array} { r } { \pmb { x } = \pmb { \mathcal { G } } ( \pmb { \xi } ) , \qquad \pmb { x } \in \Omega , \quad \pmb { \xi } \in \widehat { \Omega } . } \end{array}
$$

Mapped geometries are evaluated on this uniform reference grid, while the square and cube are evaluated directly on uniform grids in $[ - 1 , 1 ] ^ { d }$ . The two settings difer only by an afine change of reference coordinates. Physical derivatives for mapped geometries are recovered through the Jacobian of the transformation, which is assumed to remain nonsingular at the quadrature points.

Following the coordinate-transformation idea used in Ref. [20], the mapping is defined by prescribing a oneto-one correspondence between ∂Ω and $\partial \widehat \Omega$ . The inverse coordinates $\bar { \pmb { \xi } } ( { \pmb x } ) = \mathcal { G } ^ { - 1 } ( { \pmb x } )$ may be viewed as harmonic coordinates satisfying

$$
\Delta _ { \pmb { x } } \xi _ { k } = 0 , \qquad k = 1 , \ldots , d ,
$$

subject to the prescribed boundary correspondence. In practice, it is more convenient to solve for the forward map ${ \pmb x } = { \pmb G } ( { \pmb \xi } )$ . Writing $\pmb { x } = ( x _ { 1 } , \ldots , x _ { d } ) ^ { T }$ , the corresponding equations can be written as

$$
\begin{array} { r } { \mathrm { t r } \big ( J ^ { - T } D _ { \xi } ^ { 2 } x _ { i } J ^ { - 1 } \big ) = 0 , \qquad i = 1 , \dots , d , } \end{array}
$$

where $J ( \pmb { \xi } ) = \partial \pmb { x } / \partial \pmb { \xi }$ is the Jacobian matrix and $D _ { \pm } ^ { 2 } x _ { i }$ is the Hessian of $x _ { i }$ with respect to the reference coordinates.

Once the map ${ \pmb x } = { \pmb G } ( { \pmb \xi } )$ is known, all geometric quantities are evaluated on the reference grid. The Jacobian $J$ is obtained from the mapped coordinates, and its inverse $( J ^ { - 1 } ) _ { k i } = \partial \xi _ { k } / \partial x _ { i }$ gives the first-order geometric coeficients. For a scalar field $f ( { \pmb x } )$ , let

$$
{ \widehat { f } } ( \pmb { \xi } ) = f ( \pmb { \mathcal { G } } ( \pmb { \xi } ) )
$$

denote its pullback to the reference domain. The same transformation is applied componentwise to the tensor field $Q$ , and the volume element satisfies dx $= | \operatorname* { d e t } J | \operatorname { d } \pmb { \xi }$

The first-order derivatives in the physical domain are then given by

$$
\nabla _ { x } f = J ^ { - T } \nabla _ { \xi } \widehat { f } .\tag{B1}
$$

Thus, derivatives are evaluated on the uniform reference grid and converted to the physical domain through the inverse Jacobian.

For second-order derivatives, the nonlinearity of the coordinate map also enters. Applying the chain rule twice gives

$$
D _ { \pmb { x } } ^ { 2 } f = J ^ { - T } ( D _ { \pmb { \xi } } ^ { 2 } \widehat { f } ) J ^ { - 1 } + \sum _ { k = 1 } ^ { d } \frac { \partial \widehat { f } } { \partial \xi _ { k } } D _ { \pmb { x } } ^ { 2 } \xi _ { k } .\tag{B2}
$$

The first term is the standard transformation of the Hessian, while the second term accounts for the local curvature of the mapping.

The second-order geometric coeficients are obtained from the identity

$$
\xi _ { k } ( \mathcal { G } ( \xi ) ) = \xi _ { k } , \qquad k = 1 , \dots , d .
$$

Diferentiating twice with respect to $\boldsymbol { \xi }$ yields

$$
D _ { \pmb { x } } ^ { 2 } \xi _ { k } = - J ^ { - T } \left( \sum _ { l = 1 } ^ { d } ( J ^ { - 1 } ) _ { k l } D _ { \pmb { \xi } } ^ { 2 } x _ { l } \right) J ^ { - 1 } , k = 1 , \dots , d .\tag{B3}
$$

Therefore, once the forward map is available, Eqs. (B1)– (B3) provide all first- and second-order derivatives in the physical domain in terms of quantities defined on the reference grid.

In the present work, these formulas are used to evaluate the gradient terms of $Q$ and the higher-order derivatives of u within the same variational framework for both regular and irregular confinement geometries.

## Appendix C: Boundary conditions and auxiliary loss terms

## 1. Boundary conditions

Let $\Gamma _ { \mathrm { b c } } \subseteq \partial \Omega$ denote the anchored part of the boundary. Most calculations prescribe $Q _ { \mathrm { b c } }$ through

$$
\mathcal { L } _ { \mathrm { b c } } = \int _ { \Gamma _ { \mathrm { b c } } } | Q - Q _ { \mathrm { b c } } | ^ { 2 } \mathrm { d } S .\tag{C1}
$$

For 2D tangent anchoring and 3D homeotropic anchoring on the sphere, the targets are

$$
\begin{array} { r } { Q _ { \mathrm { b c } } ^ { \mathrm { 2 D } } = s _ { + } \left( { \displaystyle t \otimes t - \frac { I _ { 2 } } { 2 } } \right) , } \\ { Q _ { \mathrm { b c } } ^ { \mathrm { h o m } } = s _ { + } \left( { \nu \otimes \nu - \frac { I _ { 3 } } { 3 } } \right) . } \end{array}
$$

where t is the local unit tangent, applied edgewise on polygons [13, 21], and ν is the outward radial normal on the sphere. The TFCD-inducing cube uses the same penalty only on its top and bottom faces, with

$$
\begin{array} { r l r } & { } & { Q _ { \mathrm { t o p } } = s _ { + } \left( \pmb { e } _ { z } \otimes \pmb { e } _ { z } - \frac { \pmb { I } _ { 3 } } { 3 } \right) , } \\ & { } & { Q _ { \mathrm { b o t } } = s _ { + } \left( \pmb { e } _ { r } \otimes \pmb { e } _ { r } - \frac { \pmb { I } _ { 3 } } { 3 } \right) , } \end{array}
$$

where $\begin{array} { r } { { \textbf { \em e } } _ { z } ~ = ~ ( 0 , 0 , 1 ) } \end{array}$ is the vertical unit vector and $\boldsymbol { e } _ { r } = ( \cos \varphi , \sin \varphi , 0 )$ is the in-plane radial direction on the bottom face. Here $\Gamma _ { \mathrm { b c } }$ consists of the top and bottom faces; the four lateral faces are left unanchored [10, 40].

The 3D tangent cases instead use the degenerate planar surface loss

$$
\mathscr { L } _ { \mathrm { b c } } = \int _ { \partial \Omega } \left. \left( \pmb { Q } + \frac { s _ { + } } { 3 } I _ { 3 } \right) \pmb { \nu } ( \pmb { x } ) \right. ^ { 2 } \mathrm { d } S .\tag{C2}
$$

This places a uniaxial director in the tangent plane without selecting an in-plane direction [38]. It is used for the tangent-anchored cube and the SmA and SmC spheres.

In contrast, no boundary term is imposed on u. Since the mLdG energy depends on $D ^ { 2 } u$ , its natural boundary condition is

$$
{ \frac { \mathrm { d } } { \mathrm { d } \varepsilon } } E { \bigl ( } Q , u + \varepsilon v { \bigr ) } { \bigg | } _ { \varepsilon = 0 } = 0 \qquad { \mathrm { f o r ~ e v e r y ~ } } v \in H ^ { 2 } ( \Omega ) ,
$$

where the traces of v and $\partial _ { \nu } v$ are unrestricted. Thus neither u nor $\partial _ { \nu } \boldsymbol { u }$ is prescribed at ∂Ω [11, 13]; Q is anchored explicitly, while the boundary behavior of u is determined variationally.

## 2. Euler–Lagrange residual penalty

For completeness, we summarize the EL residual penalty introduced in Sec. III C. The formulas below correspond to the dimensionless energies E<sup>¯</sup> in Eqs. (A2) and (A3), using the bar-dropped parameter notation of the main text. It is convenient to define the equivalent stationary operators

$$
\mathcal { R } _ { Q } = - \frac { \delta \bar { E } } { \delta Q } , \qquad \mathcal { R } _ { u } = \frac { \lambda ^ { 2 } } { 2 B _ { 0 } } \frac { \delta \bar { E } } { \delta u } .
$$

Their common zero set is the EL system. Writing the parameterizations in Eq. (A1) as $Q = Q ( q _ { 1 } , . . . , q _ { m } )$ , with $m = 2$ in 2D and $m = 5$ in 3D, and using $\mathcal { E } = \bar { E } / \lambda ^ { 2 }$ , the implemented residuals are

$$
r _ { \alpha } = - \lambda ^ { - 2 } \mathcal { R } _ { Q } : \frac { \partial Q } { \partial q _ { \alpha } } , \qquad r _ { u } = 2 B _ { 0 } \lambda ^ { - 4 } \mathcal { R } _ { u } .\tag{C3}
$$

The regularization is their squared Euclidean norm,

$$
\mathcal { L } _ { \mathrm { E L } } = \int _ { \Omega } \left( \sum _ { \alpha = 1 } ^ { m } | r _ { \alpha } | ^ { 2 } + | r _ { u } | ^ { 2 } \right) \mathrm { d } x .
$$

Thus, ${ \mathcal { L } } _ { \mathrm { E L } }$ measures stationarity with respect to the scalar fields produced by the network. For mapped geometries, the derivatives and integral are evaluated using Appendix B.

SmA phase. For the d-dimensional SmA free energy, the orientational EL residual is

$$
\begin{array} { r l } & { { \mathcal { R } _ { Q } ^ { \mathrm { S m A } } } = \Delta Q - 2 \lambda ^ { 2 } B _ { 0 } q ^ { 4 } \frac { Q } { s _ { + } ^ { 2 } } u ^ { 2 } } \\ & { \qquad - \frac { 2 B _ { 0 } q ^ { 2 } } { s _ { + } } \left( u D ^ { 2 } u - \frac { \mathrm { t r } \left( u D ^ { 2 } u \right) } { d } I _ { d } \right) } \\ & { \qquad - \lambda ^ { 2 } \left[ \frac { A } { 2 C } Q - \frac { B } { 2 C } \left( Q ^ { 2 } - \frac { \mathrm { t r } \left( Q ^ { 2 } \right) } { d } I _ { d } \right) + \frac { \mathrm { t r } \left( Q ^ { 2 } \right) } { 2 } Q \right] , } \end{array}
$$

where the final bulk contribution is replaced in 2D by

$$
- \lambda ^ { 2 } \left( - \frac { B ^ { 2 } } { 4 C ^ { 2 } } + \frac { \mathrm { t r } ( Q ^ { 2 } ) } { 2 } \right) Q ,
$$

in accordance with the rLdG potential. The positional EL residual is

$$
\begin{array} { l } { \displaystyle \mathcal { R } _ { u } ^ { \mathrm { S m A } } = \Delta ^ { 2 } u + \lambda ^ { 4 } \left( \frac { a } { 2 B _ { 0 } } u + \frac { c } { 2 B _ { 0 } } u ^ { 3 } \right) } \\ { \displaystyle \phantom { \frac { A } { B _ { 0 } ^ { \mathrm { S m A } } } } + \lambda ^ { 2 } \nabla \cdot \left[ \nabla \cdot \left( q ^ { 2 } \left( \frac { Q } { s _ { + } } + \frac { I _ { d } } { d } \right) u \right) \right] } \\ { \displaystyle \phantom { \frac { A } { B _ { 0 } ^ { \mathrm { S m A } } } } + \lambda ^ { 4 } \left| q ^ { 2 } \left( \frac { Q } { s _ { + } } + \frac { I _ { d } } { d } \right) \right| ^ { 2 } u } \\ { \displaystyle \phantom { \frac { A } { B _ { 0 } ^ { \mathrm { S m A } } } } + \lambda ^ { 2 } D ^ { 2 } u : \left[ q ^ { 2 } \left( \frac { Q } { s _ { + } } + \frac { I _ { d } } { d } \right) \right] . } \end{array}
$$

SmC phase. For the SmC free energy, it is convenient to introduce the auxiliary scalar field

$$
{ \mathcal { W } } = \operatorname { t r } \left[ D ^ { 2 } u \left( { \frac { Q } { s _ { + } } } + { \frac { I _ { d } } { d } } \right) \right] + \lambda ^ { 2 } q ^ { 2 } u \cos ^ { 2 } \theta _ { 0 } .
$$

The corresponding residuals are then written as

$$
\begin{array} { r l } & { \mathcal { R } _ { \pmb { Q } } ^ { \mathrm { S m C } } = \Delta \pmb { Q } - \frac { 2 B _ { 0 } } { \lambda ^ { 2 } s _ { + } } \mathcal { W } \left( D ^ { 2 } \boldsymbol { u } - \frac { \Delta \boldsymbol { u } } { d } \pmb { I } _ { d } \right) } \\ & { \quad \quad \quad - \lambda ^ { 2 } \left[ \frac { A } { 2 C } \pmb { Q } - \frac { B } { 2 C } \left( \pmb { Q } ^ { 2 } - \frac { \mathrm { t r } ( \pmb { Q } ^ { 2 } ) } { d } \pmb { I } _ { d } \right) + \frac { \mathrm { t r } ( \pmb { Q } ^ { 2 } ) } { 2 } \pmb { Q } \right] , } \end{array}
$$

and

$$
\begin{array} { r l } & { \mathcal { R } _ { u } ^ { \mathrm { S m C } } = \Delta ^ { 2 } u + 2 \lambda ^ { 2 } q ^ { 2 } \Delta u + \lambda ^ { 4 } q ^ { 4 } u } \\ & { \quad \quad \quad + \lambda ^ { 4 } \left( \cfrac { a } { 2 B _ { 0 } } u + \cfrac { c } { 2 B _ { 0 } } u ^ { 3 } \right) } \\ & { \quad \quad \quad + \nabla \cdot \left[ \nabla \cdot \left( \mathcal { W } \left( \cfrac { \mathbf { Q } } { s _ { + } } + \cfrac { { I } _ { d } } { d } \right) \right) \right] + \lambda ^ { 2 } q ^ { 2 } \mathcal { W } \cos ^ { 2 } \theta _ { 0 } . } \end{array}
$$

Equation (C3) converts these compact stationary operators to the componentwise residuals used in the code and in Table I.

## Appendix D: Implementation details

Unless otherwise stated, all DVF computations use four Fourier layers with 64 hidden channels, Gaussian error linear unit (GELU) activation, and 16 retained Fourier modes in each direction [48]. We use 129 grid points per direction in 2D and 65 in 3D, with all fields and derivatives evaluated in double precision. AdamW is applied with a learning rate of $1 0 ^ { - 3 }$ and a weight decay of $1 0 ^ { - 2 }$ for 5000 iterations in 2D and 10000 in 3D [49]; the gradient norm is clipped at 1. Each 2D setting is computed from 100 independent initializations, while each 3D setting uses 50 independent initializations; the figures show the lowest-energy minimum found for each setting.

The loss weights are fixed as $w _ { \mathrm { b c } } = 1 0 0$ and $w _ { \mathrm { E L } } = 0 . 1$ in all experiments. The warmup penalty is applied with the time-dependent weight

$$
w _ { \mathrm { p } } ( t ) = \left\{ \begin{array} { l l } { w _ { \mathrm { p } } ^ { \mathrm { m a x } } , } & { t < 2 5 0 , } \\ { w _ { \mathrm { p } } ^ { \mathrm { m a x } } ( 5 0 0 - t ) / 2 5 0 , } & { 2 5 0 \leq t < 5 0 0 , } \\ { 0 , } & { t \geq 5 0 0 , } \end{array} \right.\tag{D1}
$$

with $( C _ { \mathrm { p } } , w _ { \mathrm { p } } ^ { \mathrm { m a x } } ) = ( 5 0 , 0 . 0 1 )$ in 2D and (80, 0.02) in 3D. The directional average in $\mathcal { L } _ { \mathrm { p } }$ is estimated using 32 independently sampled directions per iteration.

For the 3D computations, we additionally employ temporary orientational guidance during the first 500 iterations, analogous to that used for phases with narrow at-

[1] P.-G. de Gennes and J. Prost, The Physics of Liquid Crystals, 2nd ed., The International Series of Monographs on Physics, Vol. 83 (Oxford University Press, Ox ford, 1993).

[2] W. Wang, L. Zhang, and P. Zhang, Modelling and computation of liquid crystals, Acta Numer. 30, 765 (2021).

[3] P.-G. de Gennes, An analogy between superconductors and smectics A, Solid State Commun. 10, 753 (1972).

[4] S. Shojaei-Zadeh and S. L. Anna, Role of surface anchoring and geometric confinement on focal conic textures in smectic-A liquid crystals, Langmuir 22, 9986 (2006).

[5] H.-L. Liang, S. Schymura, P. Rudquist, and J. Lagerwall, Nematic-smectic transition under confinement in liquid crystalline colloidal shells, Phys. Rev. Lett. 106, 247801 (2011).

[6] P. A. Monderkamp, R. Wittmann, L. B. G. Cortes, D. G. A. L. Aarts, F. Smallenburg, and H. L¨owen, Topology of orientational defects in confined smectic liquid crystals, Phys. Rev. Lett. 127, 198001 (2021).

[7] A. Sharma, M. Magrini, Y. Han, D. M. Walba, A. Majumdar, and J. P. F. Lagerwall, How smectic-A and smectic-C liquid crystals resolve confinement-induced frustration in spherical shells, Soft Matter 20, 9586 (2024).

[8] J.-h. Chen and T. C. Lubensky, Landau–Ginzburg meanfield theory for the nematic to smectic-C and nematic to smectic-A phase transitions, Phys. Rev. A 14, 1202 (1976).

[9] M. Y. Pevnyi, J. V. Selinger, and T. J. Sluckin, Modeling smectic layers in confined geometries: Order parameter and defects, Phys. Rev. E 90, 032507 (2014).

[10] J. Xia, S. MacLachlan, T. J. Atherton, and P. E. Farrell,

traction basins in Ref. [19]. The objective in Eq. (10) is augmented by $w _ { \mathrm { d } } ( t ) \mathcal { L } _ { \mathrm { d } }$ , with

$$
\mathcal { L } _ { \mathrm { d } } = \frac { 1 } { \left| \Omega \right| } \int _ { \Omega } \left| Q _ { \theta } - Q _ { \mathrm { r e f } } \right| ^ { 2 } \mathrm { d } x .
$$

The reference field is constructed as $Q _ { \mathrm { r e f } } \ = \ s _ { + } ( { \pmb n } _ { \mathrm { r e f } } \ \otimes $ $n _ { \mathrm { r e f } } \mathrm { ~ - ~ } I _ { 3 } / 3 )$ The tangent-anchored cube and the tangent-anchored SmA and SmC spheres use the uniform direction $n _ { \mathrm { r e f } } = ( 1 , 1 , 1 ) / \sqrt { 3 }$ , whereas the homeotropic sphere uses the radial direction ${ \pmb n } _ { \mathrm { r e f } } = { \pmb x } / | { \pmb x } |$ . The TFCD calculation uses the regularized toroidal director ansatz adopted in Ref. [10]. No reference field is imposed on $u .$ The weight $w _ { \mathrm { d } } ( t )$ has a maximum value of 15 and decreases to zero after iteration 500.

The DRM baseline uses four residual multilayerperceptron blocks with 64 hidden channels and the same optimizer, precision, and 5000-iteration budget. At each iteration, it samples 2000 interior points and 200 points on each edge. For the FD cross-check in Sec. IV A, the DVF fields are interpolated to the square grid with spacing $h = 2 / 1 2 8$ and relaxed by Barzilai–Borwein gradient descent [50] until the squared discrete gradient norm is below $1 0 ^ { - 8 }$ . Local stability is assessed by testing the positive definiteness of the Hessian at the converged state.

Structural landscapes in geometrically frustrated smectics, Phys. Rev. Lett. 126, 177801 (2021).

[11] J. Xia and P. E. Farrell, Variational and numerical analysis of a Q-tensor model for smectic-A liquid crystals, ESAIM Math. Model. Numer. Anal. 57, 693 (2023).

[12] J. Xia and Y. Han, Simple tensorial theory of smectic C liquid crystals, Phys. Rev. Res. 6, 033232 (2024).

[13] B. Shi, Y. Han, C. Ma, A. Majumdar, and L. Zhang, A modified Landau–de Gennes theory for smectic liquid crystals: Phase transitions and structural transitions, SIAM J. Appl. Math. 85, 821 (2025).

[14] R. Wittmann, L. B. Cortes, H. L¨owen, and D. G. Aarts, Particle-resolved topological defects of smectic colloidal liquid crystals in extreme confinement, Nat. Commun. 12, 623 (2021).

[15] R. Wittmann, P. A. Monderkamp, J. Xia, L. B. G. Cortes, I. Grobas, P. E. Farrell, D. G. A. L. Aarts, and H. L¨owen, Colloidal smectics in button-like confinements: Experiment and theory, Phys. Rev. Res. 5, 033135 (2023).

[16] W. E and B. Yu, The deep Ritz method: A deep learningbased numerical algorithm for solving variational problems, Commun. Math. Stat. 6, 1 (2018).

[17] N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht, Y. Bengio, and A. Courville, On the spectral bias of neural networks, in Proceedings of the 36th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 97, edited by K. Chaudhuri and R. Salakhutdinov (PMLR, 2019) pp. 5301–5310.

[18] Z.-Q. J. Xu, Y. Zhang, T. Luo, Y. Xiao, and Z. Ma, Frequency principle: Fourier analysis sheds light on

deep neural networks, Commun. Comput. Phys. 28, 1746 (2020).

[19] Y. Xie, J. Yin, and L. Zhang, A geometry-adaptive deep variational framework for phase discovery in the Landau–Brazovskii model (2026), arXiv:2603.05161 [cond-mat.mtrl-sci].

[20] H. Gao, L. Sun, and J.-X. Wang, PhyGeoNet: Physicsinformed geometry-adaptive convolutional neural networks for solving parameterized steady-state PDEs on irregular domain, J. Comput. Phys. 428, 110079 (2021).

[21] Y. Han, A. Majumdar, and L. Zhang, A reduced study for nematic equilibria on two-dimensional polygons, SIAM J. Appl. Math. 80, 1678 (2020).

[22] Y. Han, J. Yin, P. Zhang, A. Majumdar, and L. Zhang, Solution landscape of a reduced Landau–de Gennes model on a hexagon, Nonlinearity 34, 2048 (2021).

[23] T. Lopez-Leon and A. Fernandez-Nieves, Topological transformations in bipolar shells of nematic liquid crystals, Phys. Rev. E 79, 021707 (2009).

[24] M. Zhou, Y.-W. Sun, Z.-W. Li, H.-W. Pei, B. Li, Y.-L. Zhu, and Z.-Y. Sun, Defect transition of smectic liquid crystals confined in spherical cavities, Soft Matter 19, 3570 (2023).

[25] A. P. Emerson and C. Zannoni, Monte carlo study of gay–berne liquid-crystal droplets, J. Chem. Soc., Faraday Trans. 91, 3441 (1995).

[26] S. P¨uschel-Schlotthauer, V. Meiwes Turri´on, C. K. Hall, M. G. Mazza, and M. Schoen, The impact of colloidal surface-anchoring on the smectic A phase, Langmuir 33, 2222 (2017).

[27] O. V. Manyuhina and M. J. Bowick, Thick smectic shells, Int. J. Non-Linear Mech. 75, 87 (2015).

[28] N. J. Mottram and C. J. Newton, Introduction to Qtensor theory (2014), arXiv:1409.3542 [cond-mat.soft].

[29] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar, Fourier neural operator for parametric partial diferential equations, in International Conference on Learning Representations (2021).

[30] C. Tsakonas, A. J. Davidson, C. V. Brown, and N. J. Mottram, Multistable alignment states in nematic liquid crystal filled wells, Appl. Phys. Lett. 90, 111913 (2007).

[31] L. B. G. Cortes, Y. Gao, R. P. A. Dullens, and D. G. A. L. Aarts, Colloidal liquid crystals in square confinement: isotropic, nematic and smectic phases, J. Phys.: Condens. Matter 29, 064003 (2016).

[32] G. Canevari, A. Majumdar, and A. Spicer, Order reconstruction for nematics on squares and hexagons: A Landau–de Gennes study, SIAM J. Appl. Math. 77, 267 (2017).

[33] J. Yin, Y. Wang, J. Z. Y. Chen, P. Zhang, and L. Zhang, Construction of a pathway map on a complicated energy landscape, Phys. Rev. Lett. 124, 090601 (2020).

[34] P. Rajamanickam, Y. Han, T. Alhinai, and A. Majumdar, Nematic equilibria in isosceles triangles: The efects of edge length and apex angle on solution landscapes in a reduced Landau–de Gennes framework (2026), arXiv:2603.01015 [cond-mat.soft].

[35] E. I. L. Jull, G. Campos-Villalobos, Q. Tang, M. Dijkstra, and L. Tran, Curvature-directed anchoring and defect structure of colloidal smectic liquid crystals in confinement, PNAS Nexus 3, pgae470 (2024).

[36] P. Prinsen and P. van der Schoot, Shape and directorfield transformation of tactoids, Phys. Rev. E 68, 021701 (2003).

[37] Y.-K. Kim, S. V. Shiyanovskii, and O. D. Lavrentovich, Morphogenesis of defects and tactoids during isotropic–nematic phase transition in self-assembled lyotropic chromonic liquid crystals, J. Phys.: Condens. Matter 25, 404202 (2013).

[38] B. Shi, Y. Han, A. Majumdar, and L. Zhang, Multistability for nematic liquid crystals in cuboids with degenerate planar boundary conditions, SIAM J. Appl. Math. 84, 756 (2024).

[39] J. K. Whitmer, X. Wang, F. Mondiot, D. S. Miller, N. L. Abbott, and J. J. de Pablo, Nematic-field-driven positioning of particles in liquid crystal droplets, Phys. Rev. Lett. 111, 227801 (2013).

[40] M.-J. Gim, D. A. Beller, and D. K. Yoon, Morphogenesis of liquid crystal topological defects during the nematic–smectic A phase transition, Nat. Commun. 8, 15453 (2017).

[41] J. H. Kim, Y. H. Kim, H. S. Jeong, M. Srinivasarao, S. D. Hudson, and H.-T. Jung, Thermally responsive microlens arrays fabricated with the use of defect arrays in a smectic liquid crystal, RSC Adv. 2, 6729 (2012).

[42] M. Ambroˇziˇc, S. Kralj, and E. G. Virga, Defect-enhanced nematic surface order reconstruction, Phys. Rev. E 75, 031708 (2007).

[43] T. P. Rieker, N. A. Clark, G. S. Smith, D. S. Parmar, E. B. Sirota, and C. R. Safinya, ”chevron” local layer structure in surface-stabilized ferroelectric smectic-C cells, Phys. Rev. Lett. 59, 2658 (1987).

[44] R. E. Webster, N. J. Mottram, and D. J. Cleaver, Molecular simulation of chevrons in confined smectic liquid crystals, Phys. Rev. E 68, 021706 (2003).

[45] A. R. Fialho, N. R. Bernardino, N. M. Silvestre, and M. M. Telo da Gama, Efect of curvature on cholesteric liquid crystals in toroidal geometries, Phys. Rev. E 95, 012702 (2017).

[46] D. Wu, B. Shi, Y. Han, P. Zhang, A. Majumdar, and L. Zhang, A difuse-interface Landau–de Gennes model for free boundary problems in the theory of nematic liquid crystals, SIAM J. Math. Anal. 57, 4358 (2025).

[47] B. Shi, D. Wu, L. Zhang, and P. Zhang, Analyzing the nematic liquid crystal droplet with an improved difuseinterface Landau–de Gennes model, SIAM J. Appl. Math. 86, 919 (2026).

[48] D. Hendrycks and K. Gimpel, Gaussian error linear units (GELUs) (2016), arXiv:1606.08415 [cs.LG].

[49] I. Loshchilov and F. Hutter, Decoupled weight decay regularization, in International Conference on Learning Representations (2019).

[50] J. Barzilai and J. M. Borwein, Two-point step size gradient methods, IMA J. Numer. Anal. 8, 141 (1988).