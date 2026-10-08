# An extended deep energy method for thermo-mechanical crack propagation

Han Zhang<sup>∗a</sup>, Mehrisadat Makki Alamdari<sup>a</sup>, Babak Shahbodagh<sup>a</sup>, Mohammad Vahab<sup>b</sup>, Cosmin Anitescu<sup>c</sup>, Timon Rabczuk<sup>c</sup>, and Elena Atroshchenko<sup>†a</sup>

<sup>a</sup>University of New South Wales, Sydney, NSW, Australia <sup>b</sup>Central Queensland University, Melbourne, VIC, Australia <sup>c</sup>Bauhaus-Universität Weimar, Weimar, Germany

## Abstract

Thermo-mechanical fracture couples transient heat conduction on a cracked domain with a crack that grows as the temperature and the displacement evolve. Recently, neural energy solvers have been proposed for phase-field modeling of fracture and later extended to represent a sharp crack through the network input. However, heat conduction on the cracked domain and crack propagation under the resulting thermal stresses have not yet been treated together in these solvers. We present an extended deep energy method for thermo-mechanical crack propagation in which the crack remains a sharp polyline throughout. The temperature and the displacement are represented by two networks that receive the crack through a scalar embedding function, discontinuous across the crack and smooth elsewhere. Both fields can therefore jump across the crack without a regularization length, and the displacement is further enriched near the tip by the Williams expansion with trainable amplitudes. The temperature is obtained by minimizing an incremental functional of transient heat conduction and the displacement by minimizing the thermoelastic potential energy, in a staggered sequence at every load step. The energies are estimated by Monte Carlo integration on points that are stratified over square background elements, densified near the tip and redrawn during training, which removes the spurious solutions that a fixed point set admits. Crack growth is decided by classical fracture mechanics criteria evaluated on the trained fields. The stress intensity factors (SIFs) are extracted by the interaction integral, which under thermal load includes the area term of Wilson and Yu, and each extraction is checked by a sweep of the contour radius before the crack may advance. The crack advances at the maximum hoop stress angle when the energy release rate of a kink in that direction reaches the critical value at the crack-tip temperature. The method is assessed on a stationary thermal edge crack, on single-edge notched tension and shear tests under thermal load, the latter with a functionally graded material, and on a notched cruciform specimen. The extracted SIF of the stationary crack agrees with the published value to 0.11%, initiation in the shear test agrees with an independent sharp-crack finite element solution to within one load step, and the predicted crack paths of the cruciform follow the published solutions under mechanical, thermal and combined loading.

Keywords: Thermo-mechanical fracture; Physics-informed machine learning; Extended Deep Energy Method; Discontinuity embedding; Crack propagation; Functionally Graded Materials

## 1 Introduction

Fracture under combined thermal and mechanical loading is a common failure mode in engineering practice, in the thermal shock of ceramics, in hot cracking during additive manufacturing [1], and in thermally loaded components of nuclear, aerospace and energy systems. A temperature gradient alone can generate stresses large enough to initiate and drive a crack, and the resulting paths are often curved and hard to anticipate. In quenching experiments on thin ceramic plates the crack patterns are periodic and hierarchical, and their geometry depends on the severity of the thermal load [2]. Predicting both the structural response and the crack path under such loading is therefore a standing task of computational fracture mechanics.

The theory of brittle fracture rests on the energy balance of Grifith [3], the stress intensity factors (SIFs) of Irwin [4] and the asymptotic crack-tip fields of Williams [5]. The J-integral of Rice [6] connects the energetic and the stress intensity descriptions and remains the standard tool for reading the crack driving force from a computed solution. It is path independent for a homogeneous elastic body without body forces or thermal strain, and under a non-uniform temperature field it has to be completed by an area term, which Section 5 recalls. Within Linear Elastic Fracture Mechanics (LEFM), the direction of growth under mixed-mode loading is commonly taken from the maximum hoop stress criterion of Erdogan and Sih [7] or from the minimum strain energy density criterion of Sih [8]. Cohesive zone models in the spirit of Barenblatt [9] and Dugdale [10] add a process zone governed by a traction-separation law, and they became the basis of the cohesive element methods for crack propagation [11, 12].

Whatever theory is adopted, a sharp-crack computation has to represent a moving discontinuity within a fixed discretization. Early finite element treatments required the mesh to conform to the crack faces and to be rebuilt whenever the crack advanced [13], which becomes cumbersome for curved paths that are not known in advance. Meshfree methods such as the Element-Free Galerkin method relaxed the meshing burden and were applied to crack growth early on [14]. The Partition of Unity framework [15] then led to the Extended Finite Element Method (XFEM) [16, 17], which enriches the approximation with a Heaviside function across the crack and with the Williams functions around the tip, so that the displacement jump and the $\sqrt { r }$ singularity are both captured on a mesh that does not conform to the crack. Some variants track the crack geometry with level set functions [18]. Fries and Belytschko review the method and its variants in [19].

Two extensions of the sharp-crack route to thermal loading are relevant here. Duflot [20] derived temperature enrichment functions for insulated and conducting cracks and extracted mixed-mode SIFs under thermal load with interaction integrals [21], and KC and Kim [22] gave the interaction integrals for thermal fracture of graded materials, whose auxiliary fields the graded case requires. Enrichment is not the only route to a sharp crack under thermal load. Ai and Augarde [23] model thermoelastic fracture with an adaptive cracking particle method that uses no enrichment functions and reaches the accuracy of the enriched methods with fewer degrees of freedom. All of these methods keep the crack sharp and connect directly to the classical propagation criteria, which is the property retained here. Their cost is the explicit tracking of the crack geometry, and topological events such as branching and merging need dedicated treatment.

The energy minimization formulation of Francfort and Marigo [24] avoids the tracking altogether. Its regularized implementation by Bourdin et al. [25] replaces the sharp crack by a difuse damage band whose width is set by a regularization length. Thermodynamically consistent phase-field formulations and operator split solvers followed from Miehe et al. [26, 27], the related gradient damage models were analyzed by Pham et al. [28], energy decompositions that restrict damage to tensile states [29] and Ginzburg-Landau type evolution equations [30] widened the modeling options, and the framework was extended to dynamic fracture [31]. The reviews of Ambati et al. [32] and Wu et al. [33] give the whole picture.

For thermo-mechanical fracture, Miehe et al. [34] extended the phase-field model to thermoelastic solids, and Bourdin et al. [35] and Sicsic et al. [36] showed that such models reproduce the periodic and hierarchical crack patterns of quenched ceramic plates. More recent work brought monolithic solvers for thermoelastic fracture [37], hybrid formulations for thermoelastic crack propagation [38], models with temperature-dependent fracture properties for hot cracking in additive manufacturing [1], and applications to Functionally Graded Materials (FGMs) [39], whose composition is engineered for severe thermal environments and at the same time decides where and how a crack grows. In the phase-field description, initiation, curved growth, merging and branching all emerge from the energy minimization without any tracking of the crack, and two of these solutions serve as references below. This description has two costs. The damage band has to be resolved by elements much smaller than the regularization length, which makes large problems expensive, and fracture mechanics quantities such as the SIFs are recovered only indirectly from the difuse fields. The method proposed in this paper keeps the sharp crack and the direct access to the SIFs, and shares with the phase-field description the principle of obtaining every field by minimizing an energy.

Over the same period, neural network solvers for partial diferential equations have matured. Physics-Informed Neural Networks (PINNs) [40] represent the solution by a network trained to minimize the residual of the governing equations, with software frameworks [41] and a growing body of applications reviewed by Karniadakis et al. [42]. In solid mechanics they have been used for forward and inverse problems [43], and residual formulations have been applied to heat conduction and conjugate heat transfer [44]. For boundary value problems the energy itself can serve as the loss. The Deep Ritz Method [45] introduced this idea for variational problems in general, and the Deep Energy Method (DEM) [46, 47] applied it to computational mechanics. In both, the potential energy is minimized directly and the derivatives come from automatic diferentiation. An energy loss involves lower derivatives than a strong-form residual and includes the natural boundary conditions of the functional, and the essential conditions can be imposed exactly through distance functions [48]. Weak form residuals [49] and mixed formulations with the stresses as additional outputs [50] improve the accuracy near stress concentrations. The energy still has to be integrated numerically over the domain, so these solvers dispense with a mesh in the sense of connectivity but not with a set of integration points.

The DEM has been combined with phase-field modeling of brittle fracture by Goswami et al., with transfer learning across the load increments [51] and with adaptive schemes for fourth-order models [52], and studied systematically by Manav et al. [53]. Operator learning variants predict crack paths across loading scenarios with DeepONets [54], Physics-Informed Neural Operators (PINOs) [55] combine operator learning with physics constraints, and the energy minimization has since been extended to higher-order anisotropic crack density functionals [56] and to a second physical field [57]. These works show that a network can reproduce phase-field solutions of brittle fracture by energy minimization, and they also show what the combination costs. A network that takes the coordinates as its input is globally smooth and learns the smooth, slowly varying part of a field much faster than its steep, localized part [58, 59, 60], whereas the phase-field solution is concentrated in a narrow band with steep gradients, so accurate training needs dense sampling in the band and often a specialized architecture, such as Fourier features [61], Sinusoidal Representation Networks (SIRENs) [62] or Kolmogorov-Arnold Networks (KANs) [63]. A continuous network of the coordinates alone cannot represent the displacement jump across a sharp crack at all, whatever its size. For that class of approximation the difuse description is therefore a necessity, and the network spends most of its capacity on the localized field it approximates worst.

Several recent works return to a discrete crack inside the neural approximation. One route decomposes the domain and assigns separate networks to the two sides of the discontinuity, following the Extended PINN (XPINN) formulation [64], and enriches them with the XFEM crack-tip functions, as Lotfalian et al. [65] do for several cracks, but the decomposition becomes cumbersome once the crack is curved and growing. The other route keeps a single network and encodes the crack in its input. Discontinuity-Embedded Neural Networks (DENNs) [66] add to the input a scalar embedding function that is discontinuous across the crack and smooth elsewhere, so that points on opposite faces map to well separated inputs and a displacement jump can be represented across the crack, its magnitude being left to the training. The Discontinuity-Embedded Deep Energy Method [67] places this crack in an energy formulation, and the Extended Deep Energy Method of Wang et al. [68] adds the Williams near-tip functions with trainable amplitudes, in analogy with XFEM. These formulations recover a sharp crack, need no regularization length and give direct access to the crack-tip quantities. They have been applied to isothermal elastic fracture, and in the present work they are extended to thermal loading.

Thermo-mechanical fracture presents several challenges. A transient heat conduction problem has to be solved on the cracked domain, the crack faces are adiabatic so the temperature jumps across them, the near-tip fields under thermal load have their own structure [20], and the fracture resistance may depend on the temperature at the tip [1]. So far the two ingredients, a temperature field solved by a network and an embedded sharp crack propagated from the trained fields, have been treated in separate studies. Coupled thermo-mechanics of a graded material has been solved with PINNs without a crack [69], and the embedded crack formulations have a crack without a temperature field. Where cracked bodies under thermal load have been treated with networks, the crack has been held fixed, the temperature prescribed rather than solved, and the analysis has stopped at the SIFs, in the energy-informed peridynamic network of Yu and Zhou [70] and in the enriched graded plates of Yadav et al. [71]. A network solver that solves the temperature on the cracked domain and grows the crack under a fracture criterion evaluated on the trained fields has not been reported.

In this paper we develop an extended deep energy method for thermo-mechanical crack propagation, in which the crack is a sharp polyline that both fields receive through one embedding function built from the current crack geometry, as in the discontinuity-embedded networks [66, 67]. Two networks share this input, one for the temperature and one for the displacement. The transient heat conduction equation is advanced in time by backward Euler, recast as an incremental functional and minimized, so the temperature follows from the same kind of minimization that delivers the displacement, and the two minimizations are staggered at every load step in the manner of the operator splits of phase-field computations [27]. The embedding lets the temperature take diferent values on the two crack faces, and the zero flux across the faces is the natural condition of the thermal functional, supplemented by a weak consistency term. The displacement includes the Williams expansion [5] near the tip with trainable amplitudes, as in XFEM and, for the isothermal case, in [68], while the temperature receives no tip enrichment, for the reasons given in Section 4. The energies are estimated on integration points that are redrawn during training and densified near the tip.

Crack growth follows LEFM. The SIFs are extracted by the interaction integral [21, 72], which under thermal load includes the area term of Wilson and Yu [73] in addition to the contour term. The area term does not appear in the isothermal form of the integral and is left out, for example, by Ghafari et al. [74]. Without it the extracted SIFs stay finite but drift with the contour radius [73]. Since the complete integral is path independent, a drift with the radius marks an unreliable reading, and in the present work every extraction is therefore read over a sweep of contour radii and refused when it drifts beyond a set tolerance. The direction of growth follows the maximum hoop stress criterion [7], the criterion is evaluated on the energy released by the kink about to be taken [75, 76], which keeps it sensitive to the sign of the thermal load, and the resistance is a temperature-dependent critical energy release rate at the crack-tip temperature.

The embedded crack and the near-tip enrichment are inherited from the works cited above, and the growth criteria are the established relations named. What is new is their combination with a temperature field solved on the cracked domain, the propagation of the crack from the trained fields, and the numerical procedures that make the extraction reliable: the evaluation of the area term on the trained fields, the sweep of the contour radius, and the resampling of the integration points.

The method is assessed on four benchmarks. A stationary thermal edge crack with a published SIF [77] verifies the extraction. A single-edge notched tension test under three thermal loads [38] and its functionally graded shear counterpart are compared with a phase-field reference and, for the shear test, with an independent sharp-crack finite element solution. A notched cruciform specimen [37] is compared with the published crack paths of six independent solutions under either load alone and of two under the combined load. Across these cases the method reproduces the crack paths and the influence of the thermal load on initiation and peak load, and the crack remains a sharp geometric entity from which the SIFs and the crack-tip temperature are read directly.

The paper is organized as follows. Section 2 states the governing equations and the energy functionals. Section 3 describes the crack geometry and the embedding function. Section 4 gives the two networks, their losses and the staggered solution. Section 5 sets out the extraction of the SIFs, its admissibility checks and the propagation criterion. Section 6 presents the four benchmarks, and Section 7 concludes. The numerical parameters, the cost and the reference solutions are given in the

appendices.

## 2 Governing equations and energy functional

Consider a cracked thermoelastic body subjected to transient thermal loading. The temperature field evolves in time through heat conduction, while the mechanical response is evaluated as a quasi-static equilibrium problem at each loading or time step. For a fixed crack geometry we treat the thermomechanical coupling in a one-way sense, in which the temperature field induces thermal expansion and therefore modifies the mechanical stress state, whereas heat generation due to deformation and the feedback of mechanical fields into the thermal balance are neglected. The mechanical solution nevertheless decides whether the crack advances, and the advanced crack changes the domain on which the next temperature field is solved, so the two fields are coupled through the crack geometry. This treatment is appropriate for quasi-static brittle fracture, where deformations remain small and slow, so the temperature changes produced by straining are negligible compared with those imposed by the thermal loading [20].

The crack is an internal geometrical discontinuity of the body. The thermal and mechanical fields are solved on the cracked domain, and the crack faces carry their own boundary conditions, as sketched in Fig. 1. It is held fixed while those fields are solved, and advanced only by the growth rule of Section 5. Table 1 collects the main symbols.

![](images/0945a635ed6ec0c67510be79a7359db099249d2818ee2ec2dd3c4d1fa69d6f5a.jpg)  
Fig. 1. The thermo-mechanical problem on a cracked body. The crack $\Gamma _ { c } = \Gamma _ { c } ^ { + } \cup \Gamma _ { c } ^ { - }$ has adiabatic, traction-free faces, whose outward normals $\mathbf { n } ^ { \pm }$ point into the opening, and the active $\operatorname { t i p } \mathbf { x } _ { \mathrm { t i p } }$ . Temperature and displacement are prescribed on $\Gamma _ { T }$ and $\Gamma _ { u } ,$ heat flux and traction on $\Gamma _ { q }$ and $\Gamma _ { t }$ . For a fixed crack the temperature enters the mechanical problem through the thermal strain $\varepsilon _ { \mathrm { t h } } .$ , and the mechanical field acts on the heat conduction problem only through the crack it advances.

## 2.1 Cracked domain and primary fields

Let the body occupy an open bounded domain $\Omega \subset \mathbb { R } ^ { 2 }$ containing an explicitly represented crack. The crack has two faces, $\Gamma _ { c } ^ { + }$ and $\Gamma _ { c } ^ { - }$ , which coincide with the crack curve as point sets and are distinguished by the side from which they are approached, and $\Gamma _ { c } = \Gamma _ { c } ^ { + } \cup \Gamma _ { c } ^ { - }$ <sup>−</sup> denotes the crack as a whole. The

Table 1. Main symbols. The load or time step index n appears as a superscript. Descriptive subscripts are upright $( \varepsilon _ { \mathrm { t h } } , T _ { \mathrm { t i p } } )$ . The mass density is written ϱ so that $\rho$ can denote the crack embedding function, and $\| \cdot \|$ is the Euclidean norm.
<table><tr><td>Symbol</td><td>Meaning</td><td>Symbol</td><td>Meaning</td></tr><tr><td> $\Omega , \Omega _ { c }$   $\Gamma _ { c } ^ { \pm }$ </td><td>body, cracked domain crack faces</td><td> $\rho ( \mathbf { x } )$ </td><td>crack embedding function</td></tr><tr><td> $\mathbf { u } , T$ </td><td>displacement, temperature</td><td> $\tilde { \bf x }$   $\theta _ { u } , \theta _ { T }$ </td><td>crack-aware network input network parameters</td></tr><tr><td> $\varepsilon , \varepsilon _ { \mathrm { e } } , \varepsilon _ { \mathrm { t h } }$ </td><td>total, elastic, thermal strain</td><td> $a _ { I } , a _ { I I }$ </td><td>trainable enrichment amplitudes</td></tr><tr><td> $\sigma , \psi _ { \mathrm { e } }$ </td><td></td><td> $( r , \theta ) , \beta$ </td><td>tip polar frame, tangent angle</td></tr><tr><td> $\varrho , C _ { p } , k$ </td><td>stress, elastic energy density density, specific heat, conductivity</td><td> $K _ { I } , K _ { I I } , J$ </td><td>extracted SIFs, J-integral</td></tr><tr><td> $\alpha , T _ { 0 }$ </td><td>expansion coefficient, reference temperature</td><td> $\Gamma _ { r } , M ^ { \mathrm { t h } }$ </td><td>extraction contour, thermal area</td></tr><tr><td> $E , \nu , E ^ { \prime }$ </td><td>elastic constants, effective modulus</td><td> $p$ </td><td>term path-independence exponent</td></tr><tr><td> $\Pi _ { T } ^ { n } , \Pi _ { u } ^ { n }$ </td><td>incremental energy functionals</td><td> $\theta _ { c }$ </td><td>kink angle</td></tr><tr><td> $G _ { c } ( T )$ </td><td>critical energy release rate</td><td> $k _ { I } , k _ { I I }$ </td><td>SIFs at the kinked tip</td></tr><tr><td> $T _ { \mathrm { t i p } } ^ { n }$ </td><td>crack-tip temperature</td><td> $G ( \theta _ { c } )$ </td><td>energy release rate of the kink</td></tr><tr><td> $\Delta \bar { a } , \delta$ </td><td>crack increment, separation margin</td><td> $\chi , \chi _ { c }$ </td><td>criterion ratio, threshold</td></tr><tr><td> $\mathbf { q } , \bar { q } , Q$ </td><td>heat flux, prescribed normal flux,</td><td> $\bar { \mathbf { t } } , \mathbf { b }$ </td><td>prescribed traction, body force</td></tr><tr><td> $\mathbf { n } , \mathbf { n } ^ { \pm }$ </td><td>heat source outward normal, on the crack faces</td><td> $\mathbf { p } _ { j } , \mathbf { t } _ { j } , \mathbf { n } _ { j }$ </td><td>crack points, segment tangent and</td></tr><tr><td></td><td></td><td></td><td>normal</td></tr><tr><td> $\mathbf { D } _ { u } , D _ { \mathrm { t i p } }$ </td><td>liftings of the correction and the enrichment</td><td> $\alpha _ { \rho } , \ell _ { \rho } , \delta _ { c }$ </td><td>embedding decay, endpoint overlap</td></tr></table>

cracked domain is

$$
\Omega _ { c } = \Omega \setminus \Gamma _ { c } .\tag{1}
$$

In two dimensions, the crack is an internal curve of zero measure. Hence, the energy integrals may be evaluated over the background domain Ω, provided that the admissible fields are allowed to have distinct traces on the two crack faces.

The primary unknowns are the displacement field u and the temperature field $T ,$

$$
\mathbf { u } ( \mathbf { x } , t ) = \{ u ( \mathbf { x } , t ) , v ( \mathbf { x } , t ) \} ^ { \mathrm { T } } , \qquad T = T ( \mathbf { x } , t ) , \qquad \mathbf { x } \in \Omega _ { c } , \quad t \in [ 0 , t _ { f } ] ,\tag{2}
$$

where $t _ { f }$ denotes the final time of the loading program. Inertial efects are not included in the mechanical formulation.

## 2.2 Transient heat conduction

The temperature field is governed by transient heat conduction. For an isotropic material, the heat flux $\mathbf { q }$ is described by Fourier’s law,

$$
\mathbf { q } = - k { \nabla } T ,\tag{3}
$$

where k denotes the thermal conductivity. Combining Fourier’s law with the local energy balance gives

$$
\varrho C _ { p } \frac { \partial T } { \partial t } - \nabla \cdot ( k \nabla T ) = Q \qquad \mathrm { i n ~ } \Omega _ { c } \times ( 0 , t _ { f } ] ,\tag{4}
$$

where $\varrho$ is the mass density, $C _ { p }$ is the specific heat capacity, and Q is a volumetric heat source. No internal heat is generated in the numerical examples, that is, $Q = 0$

The external boundary is partitioned for the thermal problem as

$$
\partial \Omega = \Gamma _ { T } \cup \Gamma _ { q } , \qquad \Gamma _ { T } \cap \Gamma _ { q } = \emptyset ,\tag{5}
$$

where the temperature is prescribed on $\Gamma _ { T }$ and the normal heat flux ${ \bf q } \cdot { \bf n } = - k \partial T / \partial n$ is prescribed on $\Gamma _ { q }$ , with n the outward unit normal. Along the crack, both faces are treated as adiabatic boundaries. The thermal boundary and initial conditions are then

$$
T = \bar { T } \quad \mathrm { o n ~ } \Gamma _ { T } , \qquad \mathbf { q } \cdot \mathbf { n } = \bar { q } \quad \mathrm { o n ~ } \Gamma _ { q } , \qquad \mathbf { q } \cdot \mathbf { n } ^ { \pm } = 0 \quad \mathrm { o n ~ } \Gamma _ { c } ^ { \pm } ,\tag{6}
$$

where $\mathbf { n } ^ { \pm }$ denotes the outward unit normal of the body on the crack face $\Gamma _ { c } ^ { \pm }$ , which points into the opening and is given in terms of the crack geometry in Section 3.1, together with the initial condition

$$
T ( \mathbf { x } , 0 ) = T _ { \mathrm { i n i } } ( \mathbf { x } ) \qquad \mathrm { i n } ~ \Omega _ { c } .\tag{7}
$$

The heat equation is advanced in time with the implicit backward Euler scheme, which is unconditionally stable for linear heat conduction [78]. The time step can therefore follow the load steps of the incremental analysis, and each step reduces to the minimization of a single functional, as shown below. Let $T ^ { n }$ and $T ^ { n - 1 }$ denote the temperatures at two consecutive time steps, and let $\Delta t = t _ { n } - t _ { n - 1 }$ . The time discrete heat equation is then

$$
\varrho C _ { p } \frac { T ^ { n } - T ^ { n - 1 } } { \Delta t } - \nabla \cdot \left( k \nabla T ^ { n } \right) = Q ^ { n } \qquad \mathrm { i n ~ } \Omega _ { c } .\tag{8}
$$

Equation (8) can be obtained as the Euler-Lagrange equation of the incremental thermal functional

$$
\Pi _ { T } ^ { n } [ T ^ { n } ] = \int _ { \Omega _ { c } } \left[ \frac { \varrho C _ { p } } { 2 \Delta t } \left( T ^ { n } - T ^ { n - 1 } \right) ^ { 2 } + \frac { k } { 2 } \left. \nabla T ^ { n } \right. ^ { 2 } - Q ^ { n } T ^ { n } \right] d \Omega + \int _ { \Gamma _ { q } } \bar { q } ^ { n } T ^ { n } d \Gamma ,\tag{9}
$$

in the sense that the stationarity condition of (9) with respect to $T ^ { n }$ recovers (8) together with the natural thermal boundary conditions.

## 2.3 Thermoelastic constitutive response

At each loading or time step, the current temperature field produces thermal expansion and consequently contributes to the stress state. Under the small strain assumption, the total strain is defined as

$$
\pmb { \varepsilon } = \frac { 1 } { 2 } \left( \nabla \mathbf { u } + \nabla \mathbf { u } ^ { \mathrm { T } } \right) ,\tag{10}
$$

and the thermal strain, measured relative to the stress-free reference temperature $T _ { 0 } ,$ is

$$
{ \varepsilon } _ { \mathrm { t h } } = \alpha \left( T - T _ { 0 } \right) \mathbf { I } ,\tag{11}
$$

where $\alpha$ is the coeficient of thermal expansion and I is the second-order identity tensor. The elastic strain is obtained by subtracting the thermal strain from the total strain,

$$
\varepsilon _ { \mathrm { e } } = \varepsilon - \varepsilon _ { \mathrm { t h } } .\tag{12}
$$

The elastic strain energy density is defined as

$$
\psi _ { \mathrm { e } } = { \frac { 1 } { 2 } } \varepsilon _ { \mathrm { e } } : \mathbb { C } : \varepsilon _ { \mathrm { e } } ,\tag{13}
$$

where C is the fourth-order elastic stifness tensor. The corresponding Cauchy stress follows from the Duhamel-Neumann thermoelastic law,

$$
\pmb { \sigma } = \mathbb { C } : \varepsilon _ { \mathrm { e } } .\tag{14}
$$

For an isotropic solid, this relation reduces to

$$
{ \pmb \sigma } = \lambda \mathrm { t r } \left( \varepsilon _ { \mathrm { e } } \right) { \bf I } + 2 \mu \varepsilon _ { \mathrm { e } } ,\tag{15}
$$

where λ and $\mu$ are the Lamé constants, related to the Young’s modulus E and the Poisson’s ratio ν by $\lambda = E \nu / [ ( 1 + \nu ) ( 1 - 2 \nu ) ]$ and $\mu = E / [ 2 ( 1 + \nu ) ]$ . Plane strain is assumed in the numerical examples unless stated otherwise. Under plane strain the out-of-plane thermal expansion is suppressed by the constraint $\varepsilon _ { z z } = 0$ and is carried by $\sigma _ { z z }$ instead. Eliminating $\sigma _ { z z }$ from the in-plane relations shows that the in-plane problem is that of (11) with the coeficient of thermal expansion replaced by the efective value $( 1 + \nu ) \alpha$ . In what follows α is understood as this efective value whenever plane strain is assumed, whereas the material tables list α itself. The energy then difers from the three-dimensional one only by a term that depends on the temperature alone, which afects neither the displacement nor the stresses. The material parameters $( E , \nu , \alpha ,$ and the thermal properties) may also vary with position, as in Functionally Graded Materials (FGMs). The formulation carries over unchanged, with all parameters evaluated pointwise, and this case is exercised in the graded shear test of Section 6.3.

In component form, the elastic strain energy density can be written as

$$
\psi _ { \mathrm { e } } = \frac { 1 } { 2 } \left( \varepsilon _ { \mathrm { e } , x x } \sigma _ { x x } + \varepsilon _ { \mathrm { e } , y y } \sigma _ { y y } + 2 \varepsilon _ { \mathrm { e } , x y } \sigma _ { x y } \right) ,\tag{16}
$$

which is equivalent to (13) under the two-dimensional tensorial strain convention.

## 2.4 Quasi-static mechanical equilibrium

Once the temperature field $T ^ { n }$ at time $t _ { n }$ is known, the mechanical response is obtained from quasistatic equilibrium,

$$
\begin{array} { r } { \nabla \cdot \pmb { \sigma } + \mathbf b = \mathbf 0 \qquad \mathrm { i n } \ \Omega _ { c } , } \end{array}\tag{17}
$$

where b denotes the body force. The external boundary is partitioned for the mechanical problem as

$$
\partial \Omega = \Gamma _ { u } \cup \Gamma _ { t } , \qquad \Gamma _ { u } \cap \Gamma _ { t } = \emptyset ,\tag{18}
$$

where displacement is prescribed on $\Gamma _ { u }$ and traction is prescribed on $\Gamma _ { t }$ . The boundary and crack-face conditions are

$$
{ \mathbf u } = \bar { { \mathbf u } } \quad \mathrm { o n ~ } \Gamma _ { u } , \qquad { \boldsymbol { \sigma } } \cdot { \mathbf n } = \bar { { \mathbf t } } \quad \mathrm { o n ~ } \Gamma _ { t } , \qquad { \boldsymbol { \sigma } } \cdot { \mathbf n } ^ { \pm } = { \mathbf 0 } \quad \mathrm { o n ~ } \Gamma _ { c } ^ { \pm } .\tag{19}
$$

The traction-free condition on the crack faces describes an open crack without contact. We retain this open crack assumption throughout. Loading states that close the crack locally would require a unilateral contact condition.

For a fixed crack and the temperature field $T ^ { n }$ , the displacement field $ { \mathbf { u } } ^ { n }$ minimizes the mechanical potential energy

$$
\Pi _ { u } ^ { n } [ \mathbf { u } ^ { n } ; T ^ { n } ] = \int _ { \Omega _ { c } } \psi _ { \mathrm { e } } d \Omega - W _ { \mathrm { e x t } } ^ { n } ,\tag{20}
$$

where $\psi _ { \mathrm { e } }$ is evaluated with the elastic strain formed from $ { \mathbf { u } } ^ { n }$ and $T ^ { n }$ , and the external work is

$$
W _ { \mathrm { e x t } } ^ { n } = \int _ { \Omega _ { c } } \mathbf { b } \cdot \mathbf { u } ^ { n } d \Omega + \int _ { \Gamma _ { t } } \bar { \mathbf { t } } ^ { n } \cdot \mathbf { u } ^ { n } d \Gamma .\tag{21}
$$

The examples of Section 6 introduce their loading through essential boundary conditions, on the displacement, the temperature, or both. If neither body forces nor surface tractions are prescribed, then $W _ { \mathrm { e x t } } ^ { n } = 0$ , and the mechanical problem reduces to the minimization of the stored elastic energy over the admissible displacement fields. When the essential conditions leave a rigid body mode free, the minimizer is defined only up to that motion, and the free mode is removed by an additional point constraint.

## 3 Explicit crack representation

The crack is represented explicitly as a sharp geometrical entity. Rather than splitting the physical domain into separate subdomains, the crack is encoded through an additional scalar input to the neural approximation. This allows a single global network to distinguish points located on opposite sides of the crack, even when their Euclidean coordinates nearly coincide. This construction follows the embedding function idea introduced for Discontinuity-Embedded Neural Networks (DENNs) by Zhao and Shao [66] and adopted in Deep Energy Method formulations [67, 68]. As a result, the approximation can represent a displacement jump across the crack while retaining a single global neural representation over the entire domain.

## 3.1 Crack geometry

The crack is described by an ordered set of control points

$$
\mathcal { P } = \left\{ \mathbf { p } _ { 0 } , \mathbf { p } _ { 1 } , \dots , \mathbf { p } _ { N _ { c } } \right\} , \qquad \mathbf { p } _ { j } \in \mathbb { R } ^ { 2 } ,\tag{22}
$$

and is represented as the polyline passing through these points,

$$
\Gamma _ { c } = \bigcup _ { j = 0 } ^ { N _ { c } - 1 } \Gamma _ { c } ^ { j } ,\tag{23}
$$

where the jth segment is defined by

$$
\Gamma _ { c } ^ { j } = \left\{ \mathbf { x } = \mathbf { p } _ { j } + \xi \left( \mathbf { p } _ { j + 1 } - \mathbf { p } _ { j } \right) , \quad 0 \leq \xi \leq 1 \right\} .\tag{24}
$$

Each segment is associated with a length and a unit direction vector,

$$
l _ { j } = \| { \bf p } _ { j + 1 } - { \bf p } _ { j } \| , \qquad { \bf t } _ { j } = \frac { { \bf p } _ { j + 1 } - { \bf p } _ { j } } { l _ { j } } ,\tag{25}
$$

and the corresponding unit normal is chosen as

$$
{ \bf n } _ { j } = \left\{ - t _ { j , 2 } , t _ { j , 1 } \right\} ^ { \mathrm { T } } .\tag{26}
$$

The face $\Gamma _ { c } ^ { + }$ is the one toward which $\mathbf { n } _ { j }$ points and $\Gamma _ { c } ^ { - }$ the other. The outward normals of the body on the two faces are therefore opposite, $\mathbf { n } ^ { + } = - \mathbf { n } _ { j }$ and $\mathbf { n } ^ { - } = \mathbf { n } _ { j }$ along segment j, and both point into the opening, as drawn in Fig. 1. The arc length along the crack is given by

$$
s _ { 0 } = 0 , \qquad s _ { j } = \sum _ { m = 0 } ^ { j - 1 } l _ { m } , \qquad L _ { c } = \sum _ { m = 0 } ^ { N _ { c } - 1 } l _ { m } ,\tag{27}
$$

where $L _ { c }$ denotes the total crack length.

The active crack tip is taken as the last point of the current crack path,

$$
\mathbf { x } _ { \mathrm { t i p } } = \mathbf { p } _ { N _ { c } } ,\tag{28}
$$

and its tangent and normal are inherited from the final segment,

$$
\mathbf { t } _ { \mathrm { t i p } } = \mathbf { t } _ { N _ { c } - 1 } , \qquad \mathbf { n } _ { \mathrm { t i p } } = \left\{ - t _ { \mathrm { t i p , 2 } } , t _ { \mathrm { t i p , 1 } } \right\} ^ { \mathrm { T } } .\tag{29}
$$

These two vectors define the local crack-tip frame used for both the near-tip enrichment of Section 4 and the crack-tip post-processing of Section 5.

When crack-face quantities are required, the crack is sampled along its arc length and the sampling points are shifted slightly along the local normal. Let $\mathbf { x } _ { c } ( s )$ be the point of the polyline at arc length $s \in [ 0 , L _ { c } ]$ and $\mathbf { n } ( s )$ the normal $\mathbf { n } _ { j }$ of the segment that contains it. The two face samples, one on the side of each face, are

$$
\mathbf { x } ^ { + } ( s ) = \mathbf { x } _ { c } ( s ) + \epsilon _ { c } \mathbf { n } ( s ) , \qquad \mathbf { x } ^ { - } ( s ) = \mathbf { x } _ { c } ( s ) - \epsilon _ { c } \mathbf { n } ( s ) ,\tag{30}
$$

where $\epsilon _ { c }$ is a small numerical ofset. The shifted points serve only to evaluate crack-face quantities, namely the flux consistency term of Section 4.2 and the displacement-jump diagnostic of Section 5.1, and the crack geometry remains the polyline $\Gamma _ { c } .$ The ofset is chosen between two limits. It should be larger than the single-precision noise floor of the trained fields, and small enough that the jump sampled across the crack is not attenuated by the normal decay of the embedding of Section 3.2. The two samples should also lie on opposite sides of the embedded discontinuity, which holds along the straight stretches of the polyline. Close to a corner the blending of Section 3.2 displaces the discontinuity slightly from the polyline, and a pair of samples there can fall on one side and read no jump. The values used in each example are listed in Table A.1, and Appendix A reports the attenuation they incur and the extent and the consequences of the corner efect.

## 3.2 Crack embedding function

The crack geometry enters the neural approximation through a scalar crack embedding function $\rho ( \mathbf { x } )$ which is constructed to be discontinuous across the crack and smooth in the intact region. Ideally it satisfies [66]

$$
\operatorname* { l i m } _ { \mathbf { x } \to \mathbf { x } _ { c } ^ { - } } \rho ( \mathbf { x } ) \neq \operatorname* { l i m } _ { \mathbf { x } \to \mathbf { x } _ { c } ^ { + } } \rho ( \mathbf { x } ) ,\tag{31}
$$

at every point $\mathbf { x } _ { c }$ of the crack, where $\mathbf { x } \to \mathbf { x } _ { c } ^ { \pm }$ denotes the one-sided limit taken from the side of the crack face $\Gamma _ { c } ^ { \pm }$ , together with

$$
\operatorname* { l i m } _ { \mathbf { x } \to \mathbf { x } _ { 0 } } \rho ( \mathbf { x } ) = \rho ( \mathbf { x } _ { 0 } ) , \qquad \operatorname* { l i m } _ { \mathbf { x } \to \mathbf { x } _ { 0 } } \nabla \rho ( \mathbf { x } ) = \nabla \rho ( \mathbf { x } _ { 0 } ) , \qquad \mathbf { x } _ { 0 } \in \Omega \setminus \Gamma _ { c } .\tag{32}
$$

The first condition provides the separation required to distinguish the two crack faces, while the second condition prevents the introduction of artificial discontinuities away from the crack. The implemented function given below meets these conditions up to three local departures, in a strip of width $\delta _ { c }$ ahead of the tip, in a neighborhood of each corner of the polyline, and on the lines normal to a segment through its endpoints, which are described in Appendix A.

For a single crack segment, the crack embedding function can be factorized [66, 68] as

$$
\rho ( { \bf x } ) = f _ { 1 } ( { \bf x } ) f _ { 2 } ( { \bf x } ) ,\tag{33}
$$

where $f _ { 1 }$ is a sign component and $f _ { 2 }$ is a localization component. The sign component is defined as

$$
f _ { 1 } ( { \bf x } ) = \mathrm { s g n } \left( D _ { s } ( { \bf x } ; \Gamma _ { c l } ) \right) ,\tag{34}
$$

where $D _ { s } ( \mathbf { x } ; \Gamma _ { c l } )$ is the signed distance to the extended crack line $\Gamma _ { c l }$ , the infinite straight line obtained by extending the crack segment beyond its two endpoints, and

$$
\operatorname { s g n } ( a ) = { \left\{ \begin{array} { l l } { - 1 , } & { a \leq 0 , } \\ { 1 , } & { a > 0 . } \end{array} \right. }\tag{35}
$$

Let $\hat { \bf x }$ be the closest point to x on the extended crack line and let n be the segment normal $\mathbf { n } _ { j }$ of (26), so that the signed distance is positive on the side of $\Gamma _ { c } ^ { + }$ and negative on the side of $\Gamma _ { c } ^ { - }$ . The signed and unsigned distances are then written in terms of the Euclidean distance as

$$
D _ { s } ( \mathbf { x } ; \Gamma _ { c l } ) = \mathrm { s g n } \left[ \mathbf { n } \cdot \left( \mathbf { x } - \hat { \mathbf { x } } \right) \right] \left\| \mathbf { x } - \hat { \mathbf { x } } \right\| ,\tag{36}
$$

$$
D ( \mathbf { x } ; \Gamma _ { c l } ) = \| \mathbf { x } - \hat { \mathbf { x } } \| .\tag{37}
$$

The localization component is constructed from the signed distance functions associated with the two endpoint extension lines,

$$
f _ { 2 } ( { \bf { x } } ) = \mathrm { R e L U } ^ { 2 } \left( D _ { s } ( { \bf { x } } ; \Gamma _ { c 1 } ) D _ { s } ( { \bf { x } } ; \Gamma _ { c 2 } ) \right) \exp \left( D ( { \bf { x } } ; \Gamma _ { c l } ) \right) ,\tag{38}
$$

where $\Gamma _ { c 1 }$ and $\Gamma _ { c 2 }$ are auxiliary extension lines at the two crack endpoints. The squared ReLU factor confines the embedding to the crack segment, while the distance-dependent factor controls its spatial variation.

Equation (33) may also be interpreted as a weighted sign function [66],

$$
\rho ( \mathbf { x } ) = \mathcal { W } ( \psi ( \mathbf { x } ) ) \mathrm { s g n } \left( \phi ( \mathbf { x } ) \right) ,\tag{39}
$$

where $\phi$ is the signed distance to the extended crack surface, $\psi$ is a signed distance measure associated with the crack tips or endpoint extensions, and W is a smooth weight. A convenient choice is

$$
\mathcal { W } ( \psi ) = \mathrm { R e L U } ^ { 2 } ( \psi ) = \left\{ \begin{array} { l l } { \psi ^ { 2 } , } & { \psi > 0 , } \\ { 0 , } & { \psi \le 0 . } \end{array} \right.\tag{40}
$$

This form separates the two roles of the embedding. The sign term distinguishes the two faces of the crack, while the weight localizes the discontinuity to the crack and leaves the function elsewhere as regular as the weight itself. The implementation given next follows this form.

For the propagating crack considered here, the same weighted sign structure is evaluated on the current polyline $\mathbf { p } _ { 0 } , \ldots , \mathbf { p } _ { N _ { c } }$ of Section 3.1, with segment tangents $\mathbf { t } _ { j }$ , normals $\mathbf { n } _ { j }$ , lengths $l _ { j }$ and arc lengths $s _ { j }$ at the segment starts, $j = 0 , \ldots , N _ { c } - 1$ . For a point x, each segment $j$ supplies a signed distance to its line, the position $t _ { j }$ of a foot point limited to the segment, and the squared distance $r _ { j } ^ { 2 }$ to that foot point,

$$
d _ { j } ( \mathbf { x } ) = \mathbf { n } _ { j } \cdot ( \mathbf { x } - \mathbf { p } _ { j } ) , \qquad t _ { j } ( \mathbf { x } ) = \operatorname* { m i n } ( l _ { j } , \operatorname* { m a x } ( 0 , \mathbf { t } _ { j } \cdot ( \mathbf { x } - \mathbf { p } _ { j } ) ) ) , \qquad r _ { j } ^ { 2 } ( \mathbf { x } ) = \lVert \mathbf { x } - \mathbf { p } _ { j } - t _ { j } \mathbf { t } _ { j } \rVert ^ { 2 } ,\tag{41}
$$

and the segment values are blended with Gaussian weights on the squared distance to the segment,

$$
w _ { j } ( { \bf x } ) = \frac { \exp \left( - r _ { j } ^ { 2 } / \sigma ^ { 2 } \right) } { \sum _ { k } \exp \left( - r _ { k } ^ { 2 } / \sigma ^ { 2 } \right) } , \qquad \tilde { d } ( { \bf x } ) = \sum _ { j } w _ { j } d _ { j } ,\tag{42}
$$

$$
\tilde { s } ( { \bf x } ) = \sum _ { j } w _ { j } ( s _ { j } + t _ { j } ) + \operatorname* { m a x } ( 0 , { \bf t } _ { N _ { c } - 1 } \cdot ( { \bf x } - { \bf p } _ { N _ { c } } ) ) - \operatorname* { m a x } ( 0 , - { \bf t } _ { 0 } \cdot ( { \bf x } - { \bf p } _ { 0 } ) ) ,\tag{43}
$$

with σ half the shortest segment length. The blending gives one representative signed distance $\tilde { d }$ and one projected arc length s˜ that are continuous across the segment junctions, where taking the nearest segment alone would introduce discontinuities. The two end terms of (43) extend the arc length beyond the two endpoints along the end tangents, so that s˜ is defined on the whole domain. In terms of these two fields, the implemented crack embedding function reads

$$
\rho ( { \bf x } ) = S ( { \bf x } ) \mathcal { W } _ { c } ( { \bf x } ) , \qquad S ( { \bf x } ) = \mathrm { s g n } \left( \tilde { d } ( { \bf x } ) \right) ,\tag{44}
$$

where the side indicator $S$ plays the role of the sign term in (39) and ${ \mathcal { W } } _ { c }$ is a continuous localization weight,

$$
\mathcal { W } _ { c } ( \mathbf { x } ) = \left[ \operatorname* { m i n } \left( 1 , \operatorname* { m a x } \left( 0 , \frac { L _ { c } + \delta _ { c } - \tilde { s } ( \mathbf { x } ) } { L _ { c } + \delta _ { c } } \right) \right) \right] ^ { 2 } \exp \left( - \alpha _ { \rho } \left( \frac { | \tilde { d } ( \mathbf { x } ) | } { \ell _ { \rho } } \right) ^ { p _ { \rho } } \right) ,\tag{45}
$$

where $\delta _ { c }$ is a small endpoint overlap, and $\alpha _ { \rho } , p _ { \rho }$ and the reference length $\ell _ { \rho }$ control the normal decay, with the values of Table A.1. The first factor is the squared window of (40) expressed in arc-length coordinates. It is held at one at the first control point p and behind it, decreases along the crack, and vanishes at a distance $\delta _ { c }$ beyond the active tip, so that the embedded jump closes continuously toward the tip, mirroring the physical crack opening, while the near-tip enrichment of Section 4 supplies the opening in the immediate tip neighborhood. The overlap keeps the embedded jump open up to the tip itself. In exchange, the network can place a small discontinuity over the distance $\delta _ { c }$ ahead of the tip, where the exact fields are continuous, and its size is measured in Appendix B.

This construction is the weighted sign form (39) with $\phi = \tilde { d }$ and $\mathcal { W } = \mathcal { W } _ { c }$ , and for a single segment it reduces to the factorization (33) up to two changes. The exponential in (45) decays away from the crack where that of (38) grows, which keeps the embedding bounded as the crack propagates without changing the jump across it, and the endpoint window is written in arc length rather than through the two endpoint lines, which is what lets it follow a polyline. Which face receives the positive sign of $S$ is a convention tied to the orientation of the segment normals. Fig. 2 evaluates the implemented function on a kinked polyline. The sign change follows the crack, the transverse decay follows $\alpha _ { \rho } ,$ , and the measured jump tracks the window formula and closes at $\delta _ { c }$ beyond the tip.

The crack-aware network input is then

$$
\tilde { \mathbf { x } } = \left\{ x , y , \rho ( \mathbf { x } ) \right\} ^ { \mathrm { T } } .\tag{46}
$$

Both the displacement and temperature networks use this augmented input, so that the two fields are represented with respect to the same crack geometry.

![](images/f878207a1b88272aeb59d4df797fefeb23a636c0a37129780aa1f72e23644e4b.jpg)  
(a)

![](images/a0b2803a8f737560f6ff6a3f4394054ac39dfcfb2cfa28dcf5e36f8544726356.jpg)  
(b)

![](images/9450656678397ff14808337ff1a8b0abb16c43197b17748afb956f14b43d964d.jpg)  
(c)  
Fig. 2. The crack embedding function (44)–(45) on the crack at load step 70 of a heated graded shear run of Section 6.3 $( \alpha _ { \rho } = 5 0 , p _ { \rho } = 1 , \delta _ { c } = 0 . 0 1 2 5 )$ . (a) $\rho$ changes sign across the crack and decays away from it, the color scale clipped at ±0.05 so that the sign change stays visible toward the tip. (b) Transverse profiles at three stations $s / L _ { c } ,$ with s the arc length. The jump is largest where the crack meets the boundary and nearly closed at the tip. $\mathrm { ( c ) }$ The embedded jump sampled at the ofset $\pm \epsilon _ { c } = 0 . 0 0 6$ , as markers, against $2 \eta ^ { 2 } e ^ { - \alpha _ { \rho } \epsilon _ { c } / \ell _ { \rho } }$ , as the dashed line, where $\eta$ is the arc-length window of (45) before squaring. The zero just past the notch corner is the corner efect of Section 3.1.

## 3.3 Local crack-tip frame

A local coordinate frame is attached to the active crack tip in order to evaluate the near-tip enrichment. The tip tangent is written as

$$
\mathbf { t } _ { \mathrm { t i p } } = \left\{ \cos { \beta } , \sin { \beta } \right\} ^ { \mathrm { T } } ,\tag{47}
$$

where $\beta$ is the angle between the tip tangent and the global x axis. The corresponding normal is

$$
\mathbf { n } _ { \mathrm { t i p } } = \left\{ - \sin \beta , \cos \beta \right\} ^ { \mathrm { T } } .\tag{48}
$$

For a point $\mathbf { x } = \{ x , y \} ^ { \mathrm { T } }$ , its relative position with respect to the active tip is

$$
\mathbf { r } = \mathbf { x } - \mathbf { x } _ { \mathrm { t i p } } ,\tag{49}
$$

and its projections onto the tangent and normal are

$$
x _ { 1 } = { \bf r } \cdot { \bf t } _ { \mathrm { t i p } } , \qquad x _ { 2 } = { \bf r } \cdot { \bf n } _ { \mathrm { t i p } } .\tag{50}
$$

The local polar coordinates are therefore

$$
r = \sqrt { x _ { 1 } ^ { 2 } + x _ { 2 } ^ { 2 } } , \qquad \theta = \mathrm { a t a n 2 } ( x _ { 2 } , x _ { 1 } ) .\tag{51}
$$

These coordinates are used in the crack-tip asymptotic enrichment described in Section 4.

## 4 Enriched deep energy formulation

The formulation follows the Deep Energy Method (DEM). The temperature and the displacement are represented by separate crack-aware networks. The incremental functionals (9) and (20) are the losses, and they are minimized with derivatives obtained by automatic diferentiation [46, 47, 51].

The networks take the physical coordinates of the example, which their first layer maps afinely onto $[ - 1 , 1 ] ^ { 2 }$ . Every length in the formulation, the embedding decay, the face ofsets, the contour radius and the enrichment decay, is therefore stated in physical units, as Table A.1 lists them. In what follows x and $y$ denote the physical coordinates, $L _ { d }$ the width of the bounding box of the specimen and $L _ { c }$ the current crack length. The displacement output is scaled by a reference displacement $U _ { \mathrm { r e f } }$ and the temperature output by a reference increment $\Delta T _ { \mathrm { r e f } } .$ , each of the order of the applied quantity, and all reported values are converted back. Scalings of this kind are routine in network-based solvers, and the expressions below are written in physical variables.

The crack embedding introduced in Section 3 is supplied as an additional input to both networks through the crack-aware input x˜ of (46). The displacement network is a multilayer perceptron with hyperbolic tangent activations. The temperature network has the same form except for its first layer, which applies a sinusoidal map with map parameter $\omega _ { 0 } = 3 0$ , in the spirit of Sinusoidal Representation Networks (SIRENs) [62], to help resolve steep thermal gradients. Nothing below depends on that choice, and the layer widths are listed in Table A.1.

Essential boundary conditions, when present, are imposed through hard constraints [46, 48]. The network correction is multiplied by a lifting function that vanishes on the constrained boundary, so the prescribed values are satisfied exactly. For prescribed values on the lower and upper edges of a unit square, one may take

$$
B ( y ) = y ( 1 - y ) , \qquad 0 \leq y \leq 1 .\tag{52}
$$

Other boundary layouts require a corresponding choice of lifting function and prescribed extension.

## 4.1 Temperature approximation

Let $\widehat { T } _ { \pmb { \theta } _ { T } } ( \tilde { \mathbf { x } } )$ denote the output of the thermal network, where $\pmb { \theta } _ { T }$ collects its trainable weights and biases and x˜ is the crack-aware input (46). The temperature at time step $t _ { n }$ is approximated as

$$
T ( \mathbf { x } , t _ { n } ) = \bar { T } ( \mathbf { x } , t _ { n } ) + D _ { T } ( \mathbf { x } ) \widehat { T } _ { \theta _ { T } } ( \tilde { \mathbf { x } } ) ,\tag{53}
$$

where $\bar { T }$ is a prescribed extension satisfying the essential thermal boundary conditions and $D _ { T }$ is a lifting function that vanishes on the prescribed thermal boundary. If the reference temperature is imposed on the lower edge and a temperature increment is imposed on the upper edge, one may choose, with B from (52),

$$
\begin{array} { r } { \bar { T } ( \mathbf { x } , t _ { n } ) = T _ { 0 } + y \Delta T _ { n } , \qquad D _ { T } ( \mathbf { x } ) = B ( y ) , } \end{array}\tag{54}
$$

which gives

$$
T ( \mathbf { x } , t _ { n } ) = T _ { 0 } + y \Delta T _ { n } + B ( y ) \widehat { T } _ { \pmb { \theta } _ { T } } ( \tilde { \mathbf { x } } ) .\tag{55}
$$

This construction enforces $T = T _ { 0 }$ at $y = 0$ and $T = T _ { 0 } + \Delta T _ { n }$ at $y = 1$ . Here, $T _ { 0 }$ is the reference temperature and $\Delta T _ { n }$ is the prescribed temperature increment at time step $t _ { n }$ . Other thermal boundary conditions can be accommodated by modifying $\bar { T }$ and $D _ { T }$

The temperature receives no near-tip enrichment, although its asymptotic behavior at the tip matches that of the displacement, $\sqrt { r }$ with an $r ^ { - 1 / 2 }$ gradient. The reason lies in what is evaluated from each field. The displacement enters the interaction integral of Section 5.1 through its strains on the contour, so an error in its $\sqrt { r }$ term propagates directly into $K _ { I }$ and $K _ { I I }$ , which is why the enrichment of Section 4.3 supplies that term. The temperature enters the growth model only through its values near the tip, through the thermal strain in the mechanical loss, and through its gradient in the area term (85) of the interaction integral, where the singular gradient is integrable. Thermal enrichment functions are available if the singular gradient itself has to be resolved [20]. The choice is tested on the stationary benchmark of Section 6.1 and against an independent conduction solution in Section 6.3.

The jump across the adiabatic faces is the one thermal feature a smooth network cannot represent, and the shared embedding admits it by giving the two faces distinct inputs. The zero normal flux on the faces is not imposed through the embedding. It is the natural condition of the thermal functional of Section 2.2, which the minimization satisfies in the weak sense, supplemented by the small consistency term of Section 4.2.

## 4.2 Thermal energy loss

At time step $t _ { n } ,$ the thermal network is trained by minimizing the time discrete heat conduction functional. Let

$$
\mathcal { Q } _ { T } = \{ \mathbf { x } _ { i } ^ { T } , w _ { i } ^ { T } \} _ { i = 1 } ^ { N _ { T } ^ { q } }\tag{56}
$$

denote the thermal integration points and weights. The integrals are estimated by Monte Carlo integration on a background mesh, sketched in Fig. 3. The bounding box of the specimen is partitioned into square elements of side $h ,$ and each element receives a fixed number $n _ { q }$ of samples drawn within it by Latin hypercube sampling, so that the samples are stratified by element. The weight of a sample is the area of its element divided by the number of samples in that element, $w _ { i } ^ { T } = h ^ { 2 } / n _ { q }$ , so that every element integrates to its own area and the sum in (58) is an unbiased estimate of the functional. The mesh has no connectivity and no shape functions, and its role is to stratify the samples. With every element sampled, the variance of the estimate is no larger than that of unstratified sampling over the whole domain [79], and no draw leaves a region wider than $h$ without a point, which bounds the size of a feature that the loss cannot observe. The element is also the unit in which the near-tip densification of Fig. 3 is stated. Concentrating integration points near the tip, where the integrand varies most rapidly, is common practice in related methods, such as the dedicated quadrature for the tip enrichment of XFEM [80] and the local refinement along the crack in the Deep Energy Method for phase-field modeling of brittle fracture [52]. Let $T _ { i } ^ { n - 1 }$ be the temperature at the previous time step evaluated at the same integration points, and define

$$
\begin{array} { r } { T _ { i } ^ { n } = T ( \mathbf { x } _ { i } ^ { T } , t _ { n } ) . } \end{array}\tag{57}
$$

![](images/226dda6a6e3ffad3b22e91c570f22d4dbc9b295da0b45689f0bd089fd42fedf9.jpg)  
Fig. 3. The integration scheme. The bounding box of the specimen is partitioned into square background elements of side $h ,$ each with $n _ { q }$ Latin hypercube samples of weight $h ^ { 2 } / n _ { q } .$ . An element whose center, marked by a cross, lies within a set radius of the crack tip, the dashed circle, receives a multiple of $n _ { q }$ samples, each weight divided by the same multiple, so that every element still integrates to its own area. The multiple is 8 in the sketch and on the notched tests and 48 on the cruciform. The set is redrawn at a fixed interval during training.

The discrete thermal loss is the quadrature evaluation of the incremental thermal functional (9) with $Q = 0$ and $\bar { q } = 0$ , supplemented by a crack-face consistency term,

$$
\mathcal { L } _ { T } ^ { n } = \sum _ { i = 1 } ^ { N _ { T } ^ { q } } w _ { i } ^ { T } \left[ \frac { \varrho C _ { p } } { 2 \Delta t } \left( T _ { i } ^ { n } - T _ { i } ^ { n - 1 } \right) ^ { 2 } + \frac { k } { 2 } \left. \nabla T _ { i } ^ { n } \right. ^ { 2 } \right] + \mathcal { L } _ { q , c } ^ { n } .\tag{58}
$$

If volumetric sources or prescribed fluxes are present, the corresponding linear terms of (9) are restored.

The integration points are redrawn periodically during training on the same background elements. The resampling is essential for the thermal loss. With a fixed point set the optimizer can develop a spurious transition layer that threads between the integration points, the kind of incorrect, approximately zero-energy solution reported by Manav et al. [53], which arises when the representation contains more localized detail than the integration rule can observe. Redrawing the points at a fixed interval removes the fixed gaps in which such a layer could hide. The same resampling is applied to the mechanical integration set $\mathcal { Q } _ { u } ,$ where it plays a milder stabilizing role.

The crack faces are adiabatic, and the zero-flux condition is naturally associated with the energy statement. As a numerical safeguard, the small heat flux consistency term in (58) is defined as

$$
\mathcal { L } _ { q , c } ^ { n } = \frac { \lambda _ { q } } { N _ { q } } \sum _ { j = 1 } ^ { N _ { q } } \left[ \mathbf { q } ^ { n } \left( \mathbf { x } _ { q , j } \right) \cdot \mathbf { n } _ { q , j } \right] ^ { 2 } ,\tag{59}
$$

where $\mathbf { x } _ { q , j }$ are points sampled on the crack faces, $\mathbf { n } _ { q , j }$ are the associated face normals, and

$$
\mathbf { q } ^ { n } = - k \nabla T ( \mathbf { x } , t _ { n } ) .\tag{60}
$$

The weight $\lambda _ { q }$ is kept small so that this term acts only as a weak consistency regularizer, rather than as the primary enforcement mechanism.

The trainable thermal parameters are the network weights and biases,

$$
\vartheta _ { T } = \{ \pmb { \theta } _ { T } \} .\tag{61}
$$

The thermal problem is therefore written as

$$
\vartheta _ { T } ^ { n } = \arg \operatorname* { m i n } _ { \vartheta _ { T } } \mathcal { L } _ { T } ^ { n } .\tag{62}
$$

## 4.3 Enriched displacement approximation

Once the temperature field has been determined, the displacement field is represented by a separate crack-aware network with trainable parameters $\pmb { \theta } _ { u }$ . Two elements of what follows are inherited. The first is the embedded crack input itself, which allows a single global network to represent a displacement jump and is the construction of the discontinuity-embedded networks of Zhao and Shao [66] and of the energy formulation built on them [67]. The second is the asymptotic enrichment with trainable amplitudes, classical in XFEM and combined with an embedded crack in an isothermal energy setting by Wang et al. [68]. The network output is written as

$$
\widehat { \mathbf { u } } _ { \pmb { \theta } _ { u } } ( \widetilde { \mathbf { x } } ) = \left[ \widehat { u } _ { \pmb { \theta } _ { u } } ( \widetilde { \mathbf { x } } ) , \widehat { v } _ { \pmb { \theta } _ { u } } ( \widetilde { \mathbf { x } } ) \right] ^ { \mathrm { T } } ,\tag{63}
$$

where $\tilde { \bf x }$ is again the crack-aware input (46). The displacement approximation at time step $t _ { n }$ is defined as

$$
\mathbf { u } ( \mathbf { x } , t _ { n } ) = { \bar { \mathbf { u } } } ( \mathbf { x } , t _ { n } ) + \mathbf { D } _ { u } ( \mathbf { x } ) { \widehat { \mathbf { u } } } _ { \theta _ { u } } ( { \tilde { \mathbf { x } } } ) + D _ { \mathrm { t i p } } ( \mathbf { x } ) \mathbf { u } _ { \mathrm { t i p } } ( \mathbf { x } ) .\tag{64}
$$

Here, u¯ is a prescribed extension satisfying the essential displacement boundary conditions, $\mathbf { D } _ { u }$ is the lifting of the network correction, a scalar function or a diagonal matrix when the two displacement components are constrained diferently, $\mathbf { u } _ { \mathrm { t i p } }$ is the near-tip enrichment defined below, and $D _ { \mathrm { t i p } }$ is a scalar lifting that vanishes on every constrained boundary. As in the thermal case (53), the liftings keep the essential conditions exactly satisfied. The enrichment receives one scalar lifting for both components, which preserves the angular structure of the Williams modes near the tip, and the matrix form of $\mathbf { D } _ { u }$ is therefore confined to the network correction. For example, if the displacements $\bar {  { \mathbf { u } } } _ { \mathrm { b } } ^ { n }$ and $\bar {  { \mathbf { u } } } _ { \mathrm { t } } ^ { n }$ are prescribed on the lower and upper edges, respectively, the extension may be chosen as

$$
\bar { \mathbf { u } } ( \mathbf { x } , t _ { n } ) = ( 1 - y ) \bar { \mathbf { u } } _ { \mathrm { b } } ^ { n } + y \bar { \mathbf { u } } _ { \mathrm { t } } ^ { n } ,\tag{65}
$$

with $\mathbf { D } _ { u } = D _ { \mathrm { t i p } } = B ( y )$ from (52), so that the trainable correction and the enrichment vanish on the prescribed boundaries. The displacement approximation then becomes

$$
\mathbf { u } ( \mathbf { x } , t _ { n } ) = ( 1 - y ) \bar { \mathbf { u } } _ { \mathrm { b } } ^ { n } + y \bar { \mathbf { u } } _ { \mathrm { t } } ^ { n } + B ( y ) \left[ \widehat { \mathbf { u } } _ { \theta _ { u } } ( \tilde { \mathbf { x } } ) + \mathbf { u } _ { \mathrm { t i p } } ( \mathbf { x } ) \right] .\tag{66}
$$

Other displacement boundary conditions can be handled by modifying u¯, $\mathbf { D } _ { u }$ and $D _ { \mathrm { t i p } }$ accordingly. The whole approximation, from the crack polyline to the two losses, is drawn in $\mathrm { F i g . 4 }$

The crack-tip contribution is introduced to represent the leading asymptotic displacement behavior near the crack tip. In the local crack-tip coordinate system, indicated by the subscript ℓ, the enriched displacement field is written as

$$
\begin{array} { r } { \mathbf { u } _ { \ell } ^ { \mathrm { t i p } } ( r , \theta ) = a _ { I } \Phi _ { I } ( r , \theta ) + a _ { I I } \Phi _ { I I } ( r , \theta ) , } \end{array}\tag{67}
$$

![](images/d2555c9e3cdd6e62711deb85c17f682300a4c199b112d1e20932398379988829.jpg)  
Fig. 4. Structure of the enriched approximation. The crack polyline defines the embedding, which enters both networks through the crack-aware input. Both network outputs are lifted onto the essential conditions, and the displacement is enriched near the tip by the Williams modes with trainable amplitudes. The thermal and mechanical losses are minimized in sequence, as set out in Section 4.5, the trained temperature entering the mechanical loss through the thermal strain.

where $( r , \theta )$ are the local polar coordinates centered at the crack tip, and $a _ { I }$ and $a _ { I I }$ are trainable scalar amplitudes associated with the mode I and mode II asymptotic fields. These amplitudes have the dimensions of SIFs but are not used as such, since their trained values also absorb the local value of the lifting at the tip and the decay introduced below. The SIFs used for propagation are extracted from the converged fields by the interaction integral of Section $5 ,$ on a contour away from the tip. The corresponding functions $\Phi _ { I }$ and $\Phi _ { I I }$ are the Williams mode I and mode II displacement fields [5]. Both carry the factor ${ \sqrt { r } } ,$ so their gradients carry $r ^ { - 1 / 2 }$ and it is the enrichment that supplies the singular strain the network cannot form on its own:

$$
\Phi _ { I } ( r , \theta ) = \frac { 1 } { 2 \mu } \sqrt { \frac { r } { 2 \pi } } \left[ \cos \frac { \theta } { 2 } \left( \kappa - 1 + 2 \sin ^ { 2 } \frac { \theta } { 2 } \right) \right] ,\tag{68}
$$

and

$$
\Phi _ { I I } ( r , \theta ) = \frac { 1 } { 2 \mu } \sqrt { \frac { r } { 2 \pi } } \left[ \sin \frac { \theta } { 2 } \left( 2 + \kappa + \cos \theta \right) \right] .\tag{69}
$$

Here, $\mu$ is the shear modulus and κ is the Kolosov constant, with $\kappa = 3 - 4 \nu$ for plane strain and $\kappa = ( 3 - \nu ) / ( 1 + \nu )$ for plane stress.

The local enriched displacement is rotated back to the global coordinate system by

$$
\begin{array} { r } { \mathbf { u } _ { \mathrm { t i p } } ^ { 0 } ( \mathbf { x } ) = \mathbf { R } ( \beta ) \mathbf { u } _ { \ell } ^ { \mathrm { t i p } } ( r , \theta ) , } \end{array}\tag{70}
$$

where $\beta$ is the angle between the local crack-tip tangent and the global x axis. Since the Williams expansion is intended only to enrich the near-tip response, the contribution is localized by a decay function,

$$
\begin{array} { r } { \mathbf { u } _ { \mathrm { t i p } } ( \mathbf { x } ) = \mathcal { D } _ { u } ( r ) \mathbf { u } _ { \mathrm { t i p } } ^ { 0 } ( \mathbf { x } ) , } \end{array}\tag{71}
$$

with

$$
\mathcal { D } _ { u } ( r ) = \exp \left[ - m _ { u } \frac { L _ { d } } { L _ { c } } r ^ { q _ { u } } \right] ,\tag{72}
$$

where $L _ { c }$ is the current crack length, $L _ { d }$ the width of the bounding box, and $m _ { u }$ and $q _ { u }$ are prescribed localization parameters. Scaling the exponent by $L _ { d } / L _ { c }$ widens the region in which the enrichment acts as the crack grows, so that the represented near-tip zone remains proportional to the crack size. The decay localizes the enrichment without giving it a finite support. The angle $\theta$ of (67) is measured from the tip tangent, so the Williams functions are discontinuous across the backward extension of that tangent, which coincides with the crack only up to the last corner. Behind a corner the enrichment therefore places an attenuated jump on a line that is not the crack. The mismatch behind the corner is left to the minimization, and the extraction contour of Section 5.1 stays on the straight stretch that ends at the tip, where the two lines coincide. The magnitude of the mode I enrichment with its decay is shown in Fig. 5.

![](images/214f47d60a25901ecee1b7f1b6f1ba22d7c1c03e8256a51c2e68f1e0b4cac09b.jpg)  
Fig. 5. Magnitude of the mode I enrichment with its decay, computed from (68) and (72) with the tension test’s $m _ { u } = 5$ and $q _ { u } = 1$ , on the kinked crack of Fig. 2.

## 4.4 Mechanical energy loss

For a fixed crack configuration and a trained temperature field, the mechanical network is obtained by minimizing the thermoelastic potential energy. Let

$$
\mathcal { Q } _ { u } = \{ \mathbf { x } _ { i } ^ { u } , w _ { i } ^ { u } \} _ { i = 1 } ^ { N _ { u } ^ { q } }\tag{73}
$$

denote the mechanical integration points and weights. This set is drawn independently of the thermal set $\mathcal { Q } _ { T }$ , on the same background elements with the same element weights, and is resampled in the same way. At each integration point, the strain tensor (10) is evaluated from the displacement field as

$$
\boldsymbol { \varepsilon } _ { i } ^ { n } = \varepsilon \left( \mathbf { u } ( \mathbf { x } _ { i } ^ { u } , t _ { n } ) \right) = \frac { 1 } { 2 } \left[ \nabla \mathbf { u } ( \mathbf { x } _ { i } ^ { u } , t _ { n } ) + \nabla \mathbf { u } ^ { \mathrm { T } } ( \mathbf { x } _ { i } ^ { u } , t _ { n } ) \right] .\tag{74}
$$

The elastic strain is then obtained by subtracting the thermal strain, as in (12),

$$
\begin{array} { r } { \varepsilon _ { \mathrm { e } , i } ^ { n } = \varepsilon _ { i } ^ { n } - \varepsilon _ { \mathrm { t h } , i } ^ { n } , \qquad \varepsilon _ { \mathrm { t h } , i } ^ { n } = \alpha \left( T _ { i } ^ { n } - T _ { 0 } \right) \mathbf { I } , } \end{array}\tag{75}
$$

where

$$
T _ { i } ^ { n } = T ( \mathbf { x } _ { i } ^ { u } , t _ { n } ) .\tag{76}
$$

The discrete mechanical loss is therefore written as

$$
\mathcal { L } _ { u } ^ { n } = \sum _ { i = 1 } ^ { N _ { u } ^ { q } } { w _ { i } ^ { u } \psi _ { \mathrm { e } } \left( \pmb { \varepsilon } _ { \mathrm { e } , i } ^ { n } \right) } - W _ { \mathrm { e x t } } ^ { n } ,\tag{77}
$$

where $\psi _ { \mathrm { e } }$ is the elastic strain energy density (13) of the thermoelastic material and $W _ { \mathrm { e x t } } ^ { n }$ is the external work (21) evaluated with the discrete displacement. For displacement-controlled cases without body forces or prescribed tractions, the external work term $W _ { \mathrm { e x t } } ^ { n }$ vanishes.

Since the essential displacement conditions are built into the admissible approximation, no Dirichlet penalty terms are required. The traction-free crack-face condition remains a natural condition of the mechanical energy statement. The trainable mechanical parameters are

$$
\vartheta _ { u } = \left\{ \theta _ { u } , a _ { I } , a _ { I I } \right\} ,\tag{78}
$$

which collect the network parameters $\pmb { \theta } _ { u }$ and the Williams amplitudes $a _ { I }$ and $a _ { I I }$ . The mechanical problem is therefore

$$
\vartheta _ { u } ^ { n } = \arg \operatorname* { m i n } _ { \vartheta _ { u } } \mathcal { L } _ { u } ^ { n } .\tag{79}
$$

During this minimization, the temperature field is fixed.

## 4.5 Staggered solution procedure

The staggered sequence in which the two fields are trained is the energy-based counterpart of the operator split solvers commonly used in phase-field modeling of brittle fracture [27]. The complete cycle, including the crack growth decision of Section 5, is summarized in the flow chart of Fig. 6.

At time step $t _ { n } ,$ the crack $\Gamma _ { c }$ is fixed and its embedding $\rho ( \mathbf { x } )$ is constructed. Given the previous temperature field $T ( \mathbf { x } , t _ { n - 1 } )$ , the thermal parameters are obtained by minimizing

$$
\vartheta _ { T } ^ { n } = \arg \operatorname* { m i n } _ { \vartheta _ { T } } \mathcal { L } _ { T } ^ { n } ,\tag{80}
$$

after which $T ( \mathbf { x } , t _ { n } )$ is fixed and supplied as the thermal loading for the mechanical solve,

$$
\vartheta _ { u } ^ { n } = \arg \operatorname* { m i n } _ { \vartheta _ { u } } \mathcal { L } _ { u } ^ { n } .\tag{81}
$$

Within one step, the solution sequence can be summarized as

$$
\Gamma _ { c }  \rho ( \mathbf { x } )  T ( \mathbf { x } , t _ { n } )  \mathbf { u } ( \mathbf { x } , t _ { n } )  \mathrm { c r a c k - t i p ~ q u a n t i t i e s } .\tag{82}
$$

Both minimizations are carried out with Adam [81], followed by L-BFGS iterations on the last draw of the points only at the unloaded step 0 and in the stationary benchmark of Section 6.1. The periodic resampling of Section 4.2 replaces the objective between iterations, so a quasi-Newton refinement on a fixed objective, such as L-BFGS [82], is not used within the resampling schedule, whereas Adam receives the change of the objective as gradient noise [83]. Each minimization runs until the mean loss over one resampling interval stops improving by a set relative amount, or until its iteration budget is spent, with the values given in Appendix A. Automatic diferentiation supplies all spatial derivatives required in the thermal gradient, strains, stresses, and energy densities. At each new time step or crack increment, both networks are initialized from the converged parameters of the previous configuration. Since the fields change only incrementally, this warm start substantially reduces the number of iterations per step, in the same spirit as the transfer learning strategy of [51].

Crack-tip quantities are extracted from the trained fields after the mechanical solve, and if the propagation criterion of Section 5 is met the crack is advanced, the embedding is rebuilt and the sequence is repeated on the new geometry. Throughout that repetition the previous temperature field $T ^ { n - 1 }$ is kept fixed, so the extensions within a load step repeat the solution at the same time level.

## 5 Crack driving force and propagation

The crack propagation treatment is set within Linear Elastic Fracture Mechanics (LEFM) [3, 4]. Once the two fields have been trained for the current crack, the growth decision is taken from the trained fields alone, in the steps set out in the subsections below. The mixed-mode SIFs $K _ { I }$ and $K _ { I I }$ are extracted at the tip. The crack-tip temperature is read from the thermal field and gives the fracture resistance $G _ { c } .$ The SIFs give the kink angle $\theta _ { c }$ and the energy release rate $G ( \theta _ { c } )$ of a kink in that direction. The extraction is then checked for admissibility, and a reading that fails the check authorizes no extension, except under the bounded release for a clipped sweep of Section 5.2. Otherwise the crack is extended along $\theta _ { c }$ whenever $G ( \theta _ { c } )$ reaches $G _ { c } ,$ after which the embedding is rebuilt and the fields are retrained. Every extension has the same length, the crack increment $\Delta a .$ , which is a numerical parameter of the method and is given per example in Table A.1. The cycle is summarized in the flow chart of Fig. 6.

![](images/7c767352feaf2a2f7df091a3364d0c383c5e94c6c5c269e3a6faeb5e2093602b.jpg)  
Fig. 6. Flow chart of the staggered solution and crack growth cycle. The ratio $\chi$ is formed before the admissibility checks of Section 5.2, since the first check acts on it. A reading that fails the checks authorizes no extension, except under the bounded release for a clipped sweep. While (94) holds and the extensions in the step are below their cap, the crack is extended and the fields are retrained on the new embedding.

## 5.1 Extraction of the stress intensity factors

The SIFs are extracted with the interaction integral [21, 72], in the form that holds for a body with a non-uniform temperature field [73, 20, 22]. Let $( x _ { 1 } , x _ { 2 } )$ denote the local tip coordinates of Section 3.3, let $\Gamma _ { r }$ be a circle of radius r around the tip with outward unit normal n, and let $A _ { r }$ be the disc that it encloses. Superposing the trained fields with a known auxiliary field $( \mathbf { u } ^ { \mathrm { a u x } } , \pmb { \sigma } ^ { \mathrm { a u x } } )$ gives the interaction integral as the sum of a contour term and an area term,

$$
I = I _ { \Gamma } + M ^ { \mathrm { t h } } ,\tag{83}
$$

with

$$
{ \cal I } _ { \Gamma } = \int _ { \Gamma _ { r } } \left[ W ^ { ( 1 , 2 ) } \delta _ { 1 j } - \sigma _ { i j } \frac { \partial u _ { i } ^ { \mathrm { a u x } } } { \partial x _ { 1 } } - \sigma _ { i j } ^ { \mathrm { a u x } } \frac { \partial u _ { i } } { \partial x _ { 1 } } \right] n _ { j } d \Gamma , \qquad W ^ { ( 1 , 2 ) } = \frac { 1 } { 2 } \left( \sigma _ { k l } \varepsilon _ { k l } ^ { \mathrm { a u x } } + \sigma _ { k l } ^ { \mathrm { a u x } } \varepsilon _ { \mathrm { e } , k l } \right) ,\tag{84}
$$

and

$$
M ^ { \mathrm { t h } } = \int _ { A _ { r } } \sigma _ { k k } ^ { \mathrm { a u x } } \frac { \partial \varepsilon _ { \mathrm { t h } } } { \partial x _ { 1 } } \mathrm { d } A ,\tag{85}
$$

where summation over repeated indices is implied and all quantities are expressed in the local tip frame. The strain of the trained field that enters $I _ { \Gamma }$ is the elastic strain $\varepsilon _ { \mathrm { e } }$ of (12), with the stress from the thermoelastic law (14), whereas the auxiliary field is isothermal, so its strain carries no thermal part and is written without the subscript. In the area term, $\varepsilon _ { \mathrm { t h } } = \alpha ( T - T _ { 0 } )$ is the scalar thermal strain of (11), carrying in plane strain the same efective coeficient $( 1 + \nu ) \alpha$ as Section 2.3, and $\sigma _ { k k } ^ { \mathrm { a u x } }$ is the in-plane trace of the auxiliary stress. The area term is the contribution that Wilson and Yu [73] derived for the thermoelastic J-integral, written here in its interaction form. It arises since the elastic energy density depends on position through the temperature, and it vanishes for a uniform temperature, so that the isothermal cases reduce to the contour term. Its contribution is discussed in Appendix C. Choosing the auxiliary field as the pure mode I or pure mode II Williams solution with unit SIF yields

$$
K _ { I } = { \frac { E ^ { \prime } } { 2 } } I ^ { ( I ) } , \qquad K _ { I I } = { \frac { E ^ { \prime } } { 2 } } I ^ { ( I I ) } , \qquad J = { \frac { K _ { I } ^ { 2 } + K _ { I I } ^ { 2 } } { E ^ { \prime } } } ,\tag{86}
$$

where $E ^ { \prime } = E / ( 1 - \nu ^ { 2 } )$ for plane strain and $E ^ { \prime } = E$ for plane stress.

In finite element practice the interaction integral is converted into an equivalent domain integral over a ring of elements around the tip, with a weight function that vanishes on its outer boundary [72], since the finite element stresses are discontinuous across element boundaries and a line integral through the elements is inaccurate. Neural networks are not subject to this limitation, since their derivatives are obtained by automatic diferentiation at any point. The contour term is therefore evaluated directly on the circle $\Gamma _ { r }$ by the trapezoidal rule with 2880 equally spaced points, with neither background elements nor a weight function, and the area term over the whole disc $A _ { r }$ by the midpoint rule on a polar grid of 80 radial by 360 angular intervals. The trace $\sigma _ { k k } ^ { \mathrm { a u x } }$ is singular at the tip but integrable against the r dr dθ measure, so no core region is excluded. The accuracy of this evaluation is checked on the stationary benchmark of Section 6.1 and, at every load step, by the radius sweep of Section 5.2.

For a graded material the auxiliary fields are evaluated with the material properties at the tip, as detailed with the graded test of Section 6.3. When the tip is so close to a boundary that no contour fits between them, the SIFs are estimated instead from the displacement jump between the crack faces behind the tip. This estimate is a diagnostic. It has no sweep behind it, so it never passes the admissibility check of Section 5.2 and never advances the crack.

The auxiliary field of (84) is the Williams solution for a straight crack, so the contour should not enclose a corner of the polyline. The stretch that ends at the tip is treated as straight as long as the segment directions stay within $5 ^ { \circ }$ of the direction of the last segment. After a kink this stretch is one crack increment $\Delta a$ long, and the contour radius is kept below it, which ties the radius to the crack increment rather than to the specimen size. The nominal radius is $r = 0 . 9 \Delta a$ in every propagating example, as sketched in Fig. $\mathrm { 7 ( a ) }$ . A contour that crosses the corner does not fail visibly. It returns a finite value evaluated with an auxiliary field that does not match the crack behind the contour, and such a value can place the post-kink mode I SIF about a third below the kinked crack solution, arrest the crack and produce a spurious rise in the reaction force.

## 5.2 Admissibility of an extraction

An extracted pair of SIFs enters the growth decision only if it passes the two checks below. A reading that fails either of them authorizes no extension and ends the growth cycle of that load step, and the criterion is evaluated again at the next load step. The number of load steps at which the checks refuse a reading is reported with each example.

The first check, called the convergence check below, acts on the criterion ratio $\chi$ of Section 5.5. Several extensions may be accepted within one load step, each followed by a partial retraining, and an extraction taken before the fields have settled is finite but does not yet describe the extended crack. An extraction is therefore rejected when its ratio exceeds a fixed multiple of the threshold, or exceeds the ratio of the previous extension of the same load step by more than a factor of two. The multiple is a set value, listed in Table A.1, and in the tests below a settled and an unsettled extension difer by an order of magnitude in this ratio.

The second check is a sweep of the contour radius. A correctly extracted SIF does not depend on the radius, whereas a trained field that contains no $1 / \sqrt { r }$ singularity returns a quantity that grows roughly as ${ \sqrt { r } } ,$ so the radius dependence measures whether the extraction is admissible at all. The integral (83) is evaluated on five radii, 0.5, 0.75, 1, 1.5 and 2 times the nominal radius r of Section 5.1. A radius that would reach past the corner of the polyline, or beyond nine tenths of the distance from the tip to the nearest boundary, is dropped, so the set may hold fewer than five radii over a narrower span. The drift across the set is expressed as one exponent

![](images/4fc9d7979f419e8ec1d3a6844f6d2746052b916a9862b0d788765eaf5aa6c6c6.jpg)  
(a)

![](images/617fb7b93ee9f5e03184ba6e6ff228f67078c4692857d475d985aa84f765c3b7.jpg)  
(b)  
Fig. 7. The two geometric rules of the extraction. (a) The nominal contour $r = 0 . 9 \Delta a$ stays inside the straight stretch that ends at the tip. A larger contour, such as $r = 1 . 5 5 \Delta a$ , dashed, would cross the kink, and the auxiliary field, a straight-crack solution, would no longer match the crack it encloses. (b) Schematic of the radius sweep, flat for a field with the $1 / \sqrt { r }$ singularity and drifting as $\sqrt { r }$ for a field without it, with the admissible drift $p < 0 . 1 1 7 2$ , 15% across a fourfold span, shaded.

$$
p = - \frac { \ln ( 1 - d ) } { \ln ( r _ { \operatorname* { m a x } } / r _ { \operatorname* { m i n } } ) } ,\tag{87}
$$

where d is the larger of the peak-to-peak variations of $K _ { I }$ and $K _ { I I }$ across the set, divided by the largest SIF magnitude in it, so that a small mode II component cannot inflate the drift, and $r _ { \mathrm { m a x } }$ and $r _ { \mathrm { m i n } }$ are the largest and the smallest radius retained. A reading passes the sweep when $p < 0 . 1 1 7 2$ the value obtained by allowing a 15% drift across a fourfold span, and the SIFs that then enter the criterion are the medians of $K _ { I }$ and $K _ { I I }$ over the retained radii. A reading fails when fewer than two radii fit or when the SIF of larger magnitude changes sign across the set, and a field without the singularity is expected to give $p \approx 0 . 5$ and to fail as well, as Fig. 7(b) illustrates. The sweep tests the singular structure of a reading rather than the completeness of the integral. Appendix C shows a reading without the area term that still passes on the stationary benchmark, where only the shape of the sweep, monotonic rather than flat, reveals the omission.

The dropping of radii has a consequence that the check allows for. A fresh kink leaves one increment of straight stretch, so the two largest radii are dropped and the retained set spans a factor of two only. A retained set that spans less than threefold is called clipped. Refusing every clipped sweep would stall the crack at each kink, since the straight stretch grows only when the crack moves. A reading that fails the sweep on a clipped set is therefore released to the growth decision, at most twice since the last sweep that passed over a wider span. A released reading advances the crack only if the criterion of Section 5.5 is then met, and the readings released in this way and the extensions taken on them are counted with each example.

## 5.3 Crack-tip temperature and fracture resistance

The crack-tip temperature is read from the trained thermal field. The exact temperature is continuous at the tip but its gradient is singular there, so a value read at the tip itself depends on how well the network resolves that gradient. The temperature is therefore taken as the mean over a small ring

centered on the tip,

$$
T _ { \mathrm { t i p } } ^ { n } = \frac { 1 } { N _ { T } } \sum _ { k = 1 } ^ { N _ { T } } T ( \mathbf { x } _ { \mathrm { t i p } } + r _ { T } \mathbf { e } ( \phi _ { k } ) , t _ { n } ) ,\tag{88}
$$

with $N _ { T } = 2 4$ equally spaced angles $\phi _ { k } , \mathbf { e } ( \phi _ { k } )$ the unit vector at that angle, ring points falling outside the specimen discarded, and the radius $r _ { T }$ given per example in Table $\mathrm { A . 1 }$ . The ring straddles both crack faces, so the mean averages the two face temperatures behind the tip. The comparison with an independent conduction solution in Section 6.3 supports this choice over the point value.

The material resistance to crack growth may vary with temperature, which is incorporated through a temperature-dependent critical energy release rate [1],

$$
G _ { c } ( T ) = G _ { c 0 } \left[ 1 - b _ { 1 } \frac { T - T _ { 0 } } { T _ { \mathrm { m a x } } } + b _ { 2 } \left( \frac { T - T _ { 0 } } { T _ { \mathrm { m a x } } } \right) ^ { 2 } \right] ,\tag{89}
$$

evaluated at the crack-tip temperature (88), where $G _ { c 0 }$ is the fracture resistance at the reference temperature $T _ { 0 } , \ T _ { \mathrm { m a x } }$ is a reference maximum temperature, and $b _ { 1 }$ and $b _ { 2 }$ are material parameters controlling the temperature dependence. We use $b _ { 1 } = 1 . 8 0$ and $b _ { 2 } = 1 . 1 0$ in the notched tension and shear tests, with $T _ { 0 }$ and $T _ { \mathrm { m a x } }$ taken as 300 K and 1000 K. With these values the bracketed factor stays above 0.26 at every temperature, so the resistance remains positive. The cruciform test of Section 6.4 uses a constant resistance, as stated there. For the functionally graded shear test, $G _ { c 0 }$ is replaced by the linearly graded value $G _ { c 0 } ( x _ { \mathrm { t i p } } )$ obtained from the interpolation (96), so that the spatial and thermal dependence of the resistance act multiplicatively.

## 5.4 Growth direction and driving force

The growth direction is determined from the extracted SIFs by the maximum hoop stress criterion [7]. The kink angle $\theta _ { c } ,$ measured from the current tip tangent, is

$$
\theta _ { c } = 2 \tan ^ { - 1 } \left( \frac { K _ { I } - \sqrt { K _ { I } ^ { 2 } + 8 K _ { I I } ^ { 2 } } } { 4 K _ { I I } } \right) \quad \mathrm { f o r } \ K _ { I I } \neq 0 , \qquad \theta _ { c } = 0 \quad \mathrm { f o r } \ K _ { I I } = 0 ,\tag{90}
$$

so that a crack under pure mode I advances along its current tangent. The magnitude of $\theta _ { c }$ is capped at the value listed in Table A.1, to avoid an excessive change of direction within a single crack increment. With $\beta$ denoting the angle between the current tip tangent and the global x axis, the global growth direction is

$$
\mathbf { d } _ { c } = \left\{ \cos ( \beta + \theta _ { c } ) , \sin ( \beta + \theta _ { c } ) \right\} ^ { \mathrm { T } } .\tag{91}
$$

The driving force is the energy released by an infinitesimal kink in the direction $\theta _ { c } ,$ , and not the rate of straight growth, since a crack that is about to turn does not release energy along its current tangent. Testing a mixed-mode criterion on the kink that the crack is about to take is the criterion of Nuismer [76]. To first order, the SIFs at the tip of an infinitesimal kink at angle $\theta$ are [75]

$$
\begin{array} { r l } & { k _ { I } ( \theta ) = \cos ^ { 3 } \left( \frac { \theta } { 2 } \right) K _ { I } - 3 \cos ^ { 2 } \left( \frac { \theta } { 2 } \right) \sin \left( \frac { \theta } { 2 } \right) K _ { I I } , } \\ & { k _ { I I } ( \theta ) = \cos ^ { 2 } \left( \frac { \theta } { 2 } \right) \sin \left( \frac { \theta } { 2 } \right) K _ { I } + \cos \left( \frac { \theta } { 2 } \right) \left[ 1 - 3 \sin ^ { 2 } \left( \frac { \theta } { 2 } \right) \right] K _ { I I } , } \end{array}\tag{92}
$$

where the lower case SIFs $k _ { I }$ and $k _ { I I }$ belong to the crack after the turn and $K _ { I }$ and $K _ { I I }$ to the crack before it. The full expansion is given by Amestoy and Leblond [84]. The corresponding energy release rate is

$$
G ( \theta ) = \frac { k _ { I } ^ { 2 } ( \theta ) + k _ { I I } ^ { 2 } ( \theta ) } { E ^ { \prime } } .\tag{93}
$$

Both relations are standard and are used unchanged. They are evaluated at $\theta = \theta _ { c }$ with the extracted SIFs, and the resulting $G ( \theta _ { c } )$ is the driving force that enters the propagation criterion of Section 5.5. For pure mode I, $\theta _ { c } = 0$ and $G ( \theta _ { c } )$ equals the straight-ahead rate J of (86). For pure mode II the angle $| \theta _ { c } | = \operatorname { a r c c o s } ( 1 / 3 ) \approx 7 0 . 5 3 ^ { \circ }$ gives $k _ { I } = ( 2 / \sqrt { 3 } ) | K _ { I I } |$ and $k _ { I I } = 0 ,$ , so that $\begin{array} { r } { G ( \theta _ { c } ) = \frac { 4 } { 3 } K _ { I I } ^ { 2 } / E ^ { \prime } } \end{array}$ and the turn releases a third more energy than straight growth. Where the cap of Table A.1 limits the angle, the capped angle is used for the direction and for the driving force alike, so the two cannot disagree. The choice of $G ( \theta _ { c } )$ over J matters under thermal load. The straight-ahead rate J [6] is even in both SIFs, so it does not distinguish a thermal stress that opens the crack from one of the same size that closes it, whereas $G ( \theta _ { c } )$ contains the product $K _ { I } K _ { I I }$ through (92) and distinguishes the two states whenever $K _ { I I } \neq 0$

## 5.5 Propagation criterion and crack update

The crack advances while the ratio χ of the driving force to the resistance meets or exceeds a prescribed threshold $\chi _ { c }$

$$
\chi \equiv \frac { G ( \theta _ { c } ) } { G _ { c } ( T _ { \mathrm { t i p } } ^ { n } ) } \geq \chi _ { c } .\tag{94}
$$

The classical Grifith condition is recovered by setting $\chi _ { c } = 1$ , which is the value used in every example reported below. For a stationary straight crack under pure mode I with a tensile $K _ { I }$ , (94) reduces to the familiar $K _ { I } = \sqrt { E ^ { \prime } G _ { c } }$ . The criterion does not by itself exclude growth from a closed state, which only a contact condition would, in line with the open crack assumption of Section 2.4. None of the quantities entering the criterion is calibrated against the results reported below.

When the criterion is met, the crack is advanced explicitly by appending a new control point to the crack polyline at

$$
\begin{array} { r } { \mathbf { x } _ { \mathrm { t i p } } ^ { n e w } = \mathbf { x } _ { \mathrm { t i p } } + \Delta a \mathbf { d } _ { c } . } \end{array}\tag{95}
$$

The embedding function $\rho ( \mathbf { x } )$ is then reconstructed from the new polyline. Since both networks share it, the updated geometry is passed consistently to the two fields, which are retrained from a warm start. The extraction, the checks and the criterion are then repeated on the new configuration within the same load step, and the crack is extended again while the criterion holds, up to the maximum number of extensions per load step listed in Table A.1. The cycle ends for that load step when the criterion is no longer met, an extraction is refused or that number is reached, and the analysis moves to the next load step.

Growth ends when the crack tip approaches the far boundary. Once the remaining ligament is thinner than the contour radius, the circle $\Gamma _ { r }$ no longer lies inside the material and the extraction ceases to be reliable. The specimen is therefore declared separated once the tip comes within a separation margin δ of the boundary, after which no further SIFs are extracted and the reaction force is set to zero. This is a termination rule of the computation, and the last increment before the physical separation of the two halves is not resolved. The margin and the resulting definition of the failure displacement are given with each example.

## 6 Numerical examples

The method is assessed on four problems. The first holds the crack stationary in a configuration whose SIF has a published value, and verifies the extraction of Section 5 on its own. The remaining three let the crack grow: a single-edge notched plate in tension, its functionally graded shear counterpart, and a notched cruciform specimen under combined thermal and mechanical loading. For the two notched tests the phase-field references are computed with an in-house finite element solver. The shear test is compared in addition with a sharp-crack finite element solution computed for this study, and the cruciform with published paths and SIFs. In the displacement-controlled cases the reaction force is computed as the derivative of the total potential energy with respect to the imposed boundary displacement, evaluated by automatic diferentiation. In the comparison figures of the propagating examples the reference solutions are labeled by their crack description, sharp crack or phase field, and the temperature shading in the geometry sketches is schematic. The load steps at which the admissibility checks of Section 5.2 refused a reading, and the readings released on a clipped sweep, are reported with the notched tests and, for the cruciform, in Appendix A. The numerical parameters of all four are collected in Appendix A.

## 6.1 Stress intensity factor of a stationary crack under thermal load

Before any crack is allowed to move, the extracted SIF is checked against a published value. The configuration is the thermal edge crack of Wang et al. [77], shown in Fig. 8: a strip of width $L = 0 . 5$ mm and height $W = 2 ~ \mathrm { { m m } }$ , held at temperature changes of −1 K and +1 K relative to the stress-free reference on its two vertical faces, with rollers $( v = 0 ,$ u free) on the top and bottom edges and an edge crack of length $a = 0 . 2 5$ mm running from the cold face to the center line. The top and bottom edges are adiabatic, the two vertical faces are traction free, and the crack faces are adiabatic and traction free. Plane strain is assumed, the conduction problem is steady, as in the reference, and the strip is computed in its physical dimensions. Since the crack does not move, this example is trained on a single draw of the integration points, and its training ends with L-BFGS iterations after Adam, as Table A.1 records. The cooled face is prevented from contracting by the rollers, so $\sigma _ { y y }$ goes into tension and the horizontal crack opens in pure mode I. The material data follow the reference, $E = 2 1 8 . 4 \times 1 0 ^ { 3 } ~ \mathrm { M P a }$ $\nu = 0 . 2 5 , \alpha = 1 . 6 7 \times 1 0 ^ { - 5 } ~ \mathrm { K } ^ { - 1 }$ , and the extracted SIF is normalized by $K _ { 0 } = E \alpha \Delta T _ { a } \sqrt { \pi a } / ( 1 - \nu )$ with α as quoted rather than its plane strain value $( 1 + \nu ) \alpha$ and $\Delta T _ { a } = 1 \textrm { K }$ the amplitude of the imposed change, for which the reference reports $K _ { I } / K _ { 0 } = 0 . 5 0 0$ . The temperature field in the strip is non-uniform, so the area term (85) of the interaction integral contributes to the extracted SIF.

![](images/bf5c45f94d70aebde9eddafc79073d75c3f85dac9f887b92e50a2437ceea5436.jpg)  
Fig. 8. The thermal edge crack of Wang et al. [77], with the imposed temperature diference drawn schematically. The rollers prevent the cooled face from contracting, which puts $\sigma _ { y y }$ into tension across the crack.

Fig. 9 collects the study. On grids of 48, 72 and 96 elements across the strip width, the extracted SIF is $K _ { I } / K _ { 0 } = 0 . 4 9 5 3$ , 0.4955 and 0.4995, within 0.94% of the published value on the coarsest grid and 0.11% on the finest. The comparison is made over three grids rather than as a single reading, since the reference reports its own convergence study, with errors of 10.3%, 6.5% and 0.6% across three grids and path independence reached only on the refined one. Fig. 9(b) shows the radius sweep of Section 5.2 at each resolution. Over a fourfold span of contour radii the reading drifts by 2.9%, 3.4% and 6.5%, giving $p = 0 . 0 2 1 , 0 . 0 2 5$ and 0.049, so every sweep passes the admissibility check $p < 0 . 1 1 7 2$ of Section 5.2, the drift growing with refinement while the reading itself moves toward the published value.

## 6.2 Single-edge notched tension test

A single-edge notched thermoelastic fracture benchmark considered by Tangella et al. [38] is adopted for the tensile test. This setting isolates a predominantly mode I fracture process and provides a direct check of the response under thermal loading.

![](images/2f09330a7a14725609f936a3a0b24335fe0a0792ebec30d39fd7dbb6121fe24e.jpg)  
(a)

![](images/7af92b3ae1e79aa07c6be89d64b45a2128ee0e9167cba60ccb797164337456c0.jpg)  
(b)  
Fig. 9. SIF of the stationary thermal edge crack, normalized by $K _ { 0 } = E \alpha \Delta T _ { a } \sqrt { \pi a } / ( 1 - \nu )$ with $\alpha = 1 . 6 7 \times$ $1 0 ^ { - 5 } ~ \mathrm { K } ^ { - 1 }$ and $\Delta T _ { a } = 1 \textrm { K } .$ (a) The three grids against the published value 0.500 of Wang et al. [77], with a band of 2%. (b) The radius sweep of each grid with its exponent $p ,$ all below the admissibility threshold 0.1172. The innermost contour of the finest grid lies 5% above the rest of its sweep.

The geometry and boundary conditions are shown in Fig. 10(a). The specimen is a 1 mm × 1 mm square plate with a horizontal edge notch of length 0.5 mm at mid-height, initially at $T _ { 0 } = 3 0 0 ~ \mathrm { K }$ The lower edge is fixed in both displacement components and held at $T _ { 0 }$ , the upper edge receives a monotonically applied vertical displacement with its horizontal component held at zero and the temperature $T _ { 0 } + \Delta T$ , and the two vertical edges are traction free and adiabatic. The reaction force is taken for a thickness of 1 mm, so that it is in newtons. Three thermal cases are considered, $\Delta T = - 5 0 \mathrm { ~ K ~ }$ , 0 K and +50 K, with the material parameters of Table 2, which follow the reference benchmark and are used in a consistent mm–N–s–K unit system. Each case is solved over 106 load steps of $1 0 ^ { - 5 }$ mm, with the temperature diference applied in equal increments over the first 20 steps and held from there on. The crack increment is $\Delta a = 0 . 0 2 5$ mm, a fortieth of the specimen width, which by the rule of Section 5.1 puts the extraction contour at 0.0225 mm.

![](images/9e6070ec60137ea7a92229ed3f557bf6edd1b1ec10606f1d7dcacb54546b1ba6.jpg)  
(a)

![](images/b71212084070cc50cc3b6ba3a6c4c9823498019c580dfbf9660dafb8cadaaa47.jpg)  
(b)  
Fig. 10. Geometry and boundary conditions of the single-edge notched specimen. (a) Tension, with a vertical displacement on the upper edge. (b) Shear, with a horizontal displacement on the upper edge. In both tests the lower edge is fixed and held at $T _ { 0 } ,$ the upper edge is held at $T _ { 0 } + \Delta T$ and the vertical edges are adiabatic. The temperature shading is schematic.

Table 2. Material parameters for the homogeneous single-edge notched tension test.
<table><tr><td>Parameter</td><td>Symbol Value</td><td></td></tr><tr><td>Young&#x27;s modulus</td><td>E</td><td> $3 4 0 \ \mathrm { G P a }$ </td></tr><tr><td>Poisson&#x27;s ratio</td><td> $\nu$ </td><td>0.22</td></tr><tr><td>Critical energy release rate</td><td> $G _ { c }$ </td><td> $4 2 . 4 7 ~ \mathrm { J / m ^ { 2 } }$ </td></tr><tr><td>Density</td><td> $\varrho$ </td><td> $2 4 5 0 ~ \mathrm { k g / m ^ { 3 } }$ </td></tr><tr><td>Thermal conductivity</td><td> $k$ </td><td> $3 0 0 ~ \mathrm { W / ( m K ) }$ </td></tr><tr><td>Specific heat capacity</td><td> $C _ { p }$ </td><td> $0 . 7 7 5 ~ \mathrm { J / ( k g K ) }$ </td></tr><tr><td>Thermal expansion coefficient</td><td> $\alpha$ </td><td> $8 . 0 \times 1 0 ^ { - 6 } ~ \mathrm { K } ^ { - 1 }$ </td></tr><tr><td>Reference temperature</td><td> $T _ { 0 }$ </td><td>300 K</td></tr></table>

The phase-field reference curves for both single-edge notched tests are computed with the solver of Appendix B, the AT2 model with regularization length $\ell = 0 . 0 1$ mm on elements of size about $\ell / 5$ with geometry, material data and boundary conditions matched to the corresponding run of the proposed method. On the tension test it reproduces the shift of initiation with the thermal load reported by Tangella et al. [38], with peak loads 8% to 13% below theirs, a diference discussed in Appendix B. The reference and the proposed method difer in one respect. The reference uses the fracture energy of Table 2 as a constant, as the published statement of the test does, whereas the proposed method evaluates (89) at the tip temperature, which moves its critical SIF by 1.5% between the cooled and the heated case, as reported below.

The force–displacement responses are compared with the finite element phase-field reference solutions in Fig. 11. The isothermal response remains linear until the crack initiates. Under thermal load the loading branch is steeper for the cooled specimen and shallower for the heated one, and the heated branch steepens once the temperature diference has been fully applied after 20 steps. The thermal load also moves the initiation point. Cooling brings it forward and heating delays it, which follows from the thermal strain. Expansion near the heated upper edge takes up part of the imposed displacement and reduces the elastic tensile strain at the notch, while cooling has the opposite efect. In this test the peak load falls at the last load step before the crack advances. The displacement at peak load moves from $0 . 3 5 \times 1 0 ^ { - 3 }$ mm at $\Delta T = - 5 0$ K to $0 . 6 6 \times 1 0 ^ { - 3 }$ mm at +50 K, an increase of 89%, while the peak load itself changes by 1.3% over the same range.

The peak load agrees with the reference to +4.1%, +1.3% and −0.8% at $\Delta T = - 5 0$ , 0 and +50 K, with no systematic sign, and the displacement at which it is reached is 5.4% to 5.9% smaller than in the reference in all three cases. Repeats of all three cases move the peak by at most 1.8% and initiation by at most one load step, which sets the resolution of this comparison at about 2%. After the peak the load drops within one or two load steps, and the ligament is fully separated 17% to 21% earlier in applied displacement. The post-peak branches difer by construction. A sharp crack whose resistance varies only with the tip temperature crosses the short remaining ligament as soon as the criterion is met, whereas the reference softens over a longer branch as its damage band approaches the far edge.

Fig. 12 shows the field evolution for the heating case with $\Delta T = + 5 0$ K, from the same run as the curves of Figs. 11 and 13. The notch is unchanged until step 67, when the crack crosses the whole remaining ligament within that single step, so step 66 is the last step before growth, at the peak load, and step 70 is after separation. Since the ligament is crossed within one or two load steps, this test is run without a separation margin, and its failure displacement is that of the load step in which the crack completes the crossing. By step 66 the imposed temperature diference has fully developed and the isotherms are drawn down around the notch tip, since the adiabatic faces let the heat from the upper edge reach the region below the notch only around the tip. At step 70 the vertical displacement separates across the crack and the two halves of the specimen move independently.

The crack-tip quantities that drive this response are shown in Fig. 13. $K _ { I }$ rises with the applied displacement in all three cases. The isothermal curve is linear. The thermal curves are ofset from it by the thermal stress at the notch, which keeps evolving after the ramp as the temperature difuses through the transient, so their slopes difer from the isothermal one, the cooled specimen reaching a given SIF at the smallest displacement and the heated one at the largest, in the same order as the initiation points of Fig. 11. The extracted $K _ { I I }$ stays below $3 . 6 ~ \mathrm { N / m m ^ { 3 / 2 } }$ throughout, against a $K _ { I }$ that passes 120, so the test is mode I to the accuracy of the extraction. The two thermal cases show a small mode II component of opposite signs, negative under heating and positive under cooling, the pattern the geometry suggests, since the imposed temperature diference is antisymmetric about the notch plane where the mechanical load is symmetric about it. The component is of the size of the extraction error. It is consistent with the slight deflection of the crack from the notch plane as it approaches the far edge, seen in the last column of ${ \mathrm { F i g . } }$ 12 for the heated case, and Tangella et al. [38] likewise report crack paths deflected by the thermal load in this test. The crack advances when $K _ { I }$ reaches between 120 and $1 2 3 ~ \mathrm { N / m m ^ { 3 / 2 } }$ in all three cases, which is the pure mode I form $K _ { I } = \sqrt { E ^ { \prime } G _ { c } }$ of the criterion (94), since the kink angle of every accepted extension stays within about one degree of straight ahead. The critical SIF therefore follows from the imposed $G _ { c }$ and is a consistency check rather than an independent test. It does separate the two efects of the thermal load. The critical SIF moves by 1.5% between the heated and the cooled case, through the temperature dependence of $G _ { c } ,$ while the peak-load displacement, the last before the crack advances, moves by 89%.

![](images/cd51f5e1fbf1c2572a020f3145c2032c18f1b899cf50bf04d5e646f62bacfd0c.jpg)  
Fig. 11. Force–displacement response of the single-edge notched tension test at $\Delta T = - 5 0 , 0 \mathrm { a n d } + 5 0 \mathrm { K }$ , cooled in blue, isothermal in black and heated in red, with the phase-field reference dashed.

![](images/aa7e9034193acfd4784b4ac474df23ec1a8502062f232994a67d3200842e57f5.jpg)  
Fig. 12. Temperature and vertical displacement in the single-edge notched tension test at $\Delta T = + 5 0 \mathrm { K } ,$ at load steps 0, 66 and 70 of 106. Step 66 is the last before the crack advances and step 70 follows separation. Each row shares one color scale. The region below the notch stays near the initial temperature since the crack faces are adiabatic.

![](images/6e83871a2427fd85a780dbf4ac6217a76b414a4f20659eec4bc57197adad50e2.jpg)  
(a)

![](images/dad5d3d4a77b0fd6950f9996f2294035e58c1c39664b9fd77b30a5e17391ce16.jpg)  
(b)  
Fig. 13. SIFs of the single-edge notched tension test against the applied displacement: (a) $K _ { I } , ( \mathrm { b } ) \ K _ { I I }$ , with the legend of (a). The test is mode I to the accuracy of the extraction. Readings after the tip has reached the far edge, where no contour fits inside the material, are not plotted.

The admissibility checks of Section 5.2 refused no extraction at a load step where the crack could have advanced and released no reading, so they did not alter the response of this test.

## 6.3 Single-edge notched shear test with functionally graded material

The same notched specimen as in Section 6.2 is considered under shear loading, where the crack path is no longer expected to remain straight. The lower edge is fixed in both displacement components, a horizontal displacement is applied monotonically on the upper edge with its vertical component held at zero, and the two vertical edges are traction free. The resulting mixed-mode crack driving force provides a direct test of the predicted propagation direction.

A functionally graded material is introduced in this test, so that the pointwise graded properties of Section 2.3 and the graded extraction of Section 5.1 are exercised. The test is the thermoelastic shear test of Tangella et al. [38], which their paper reports for a homogeneous material and their released code [85] implements in graded form, and the graded properties of Table 3 are taken from that code. No solution of the graded form is published, so both references for this test are computed here. The properties vary linearly in the horizontal direction, a generic material parameter m being interpolated as

$$
m ( x ) = m _ { 1 } + \frac { x } { L } \left( m _ { 2 } - m _ { 1 } \right) , \qquad 0 \leq x \leq L ,\tag{96}
$$

where $L \ = \ 1$ mm and $m _ { 1 }$ and $m _ { 2 }$ are the values at the left and right edges. In the extraction the auxiliary fields and (86) use the material properties at the current tip, and the stresses on the contour the pointwise graded properties. The terms of the interaction integral that involve the material gradient [22] are omitted, since across the contour of this test the graded properties vary by about 1%, which is the order of those terms.

The shear test is solved for $\Delta T = - 3 0 ~ \mathrm { K } , 0$ K, and +30 K over 141 load steps of $2 . 5 \times 1 0 ^ { - 5 }$ mm, with the temperature diference applied in equal increments over the first 12 steps and held from there on, and with the same crack increment $\Delta a = 0 . 0 2 5$ mm as the tension test.

The phase-field reference for this test is the solver of Section 6.2 with the graded properties of Table 3. It represents the notch by setting the damage variable to one along it, so its notch is a damaged band of width about ℓ rather than a slit, and it difers from the proposed method before either crack has grown. To separate this efect from any diference between the solvers, a second reference was computed with a sharp crack. It is a standard finite element solution of the same problem, with the notch as a seam of the mesh and the crack held at its initial length, which sufices up to initiation. Its SIFs are obtained without the interaction integral, so that it shares no part of the extraction under test, and the criterion of Section 5.5 then gives the load step at which the crack first grows, marked by stars in Fig. 14. The procedure and its check on the stationary benchmark of Section 6.1 are given in Appendix B.

Table 3. Material parameters for the functionally graded single-edge notched shear test, adopted from the implementation of Tangella et al. [85]. The values at $x = 0$ are those of the homogeneous tension test in Table 2.
<table><tr><td>Parameter</td><td>Symbol</td><td> $x = 0$ </td><td> $x = L$ </td></tr><tr><td>Young&#x27;s modulus</td><td>E</td><td> $3 4 0 \ \mathrm { G P a }$ </td><td> $4 5 0 \ \mathrm { G P a }$ </td></tr><tr><td>Poisson&#x27;s ratio</td><td> $\nu$ </td><td>0.22</td><td>0.22</td></tr><tr><td>Critical energy release rate</td><td> $G _ { c }$ </td><td> $4 2 . 4 7 ~ \mathrm { J / m ^ { 2 } }$ </td><td> $1 2 0 \ \mathrm { J / m ^ { 2 } }$ </td></tr><tr><td>Density</td><td>Q</td><td> $2 4 5 0 ~ \mathrm { k g / m ^ { 3 } }$ </td><td> $4 0 0 0 ~ \mathrm { k g / m ^ { 3 } }$ </td></tr><tr><td>Thermal conductivity</td><td> $k$ </td><td> $3 0 0 ~ \mathrm { W / ( m K ) }$ </td><td> $3 5 0 ~ \mathrm { W / ( m K ) }$ </td></tr><tr><td>Specific heat capacity</td><td> $C _ { p }$ </td><td> $0 . 7 7 5 ~ \mathrm { J } / ( \mathrm { k g } \mathrm { K } )$ </td><td> $0 . 7 7 5 ~ \mathrm { J } / ( \mathrm { k g } \mathrm { K } )$ </td></tr><tr><td>Thermal expansion coefficient</td><td> $\alpha$ </td><td> $8 . 0 \times 1 0 ^ { - 6 } ~ \mathrm { K } ^ { - 1 }$ </td><td> $8 . 0 \times 1 0 ^ { - 6 } ~ \mathrm { K } ^ { - 1 }$ </td></tr><tr><td>Reference temperature</td><td> $T _ { 0 }$ </td><td>300 K</td><td>300 K</td></tr></table>

The force–displacement responses are shown in Fig. 14. Against the sharp-crack solution the loading ramp of the proposed method agrees to within 1% to 2%, and initiation falls within one load step in all three thermal cases, 2% of the applied displacement at that point. This lies inside the run-to-run scatter of 0 to 2 load steps in initiation and 0.2% to 4.5% in peak load reported in Appendix A. Since the sharp-crack solution applies the same criterion and the same temperature-dependent toughness, the agreement tests the trained fields and the extraction. All three solutions reproduce the influence of the thermal load, the peak load rising monotonically with ∆T, cooling advancing fracture and heating delaying it.

The phase-field reference reaches its peak five load steps after the sharp-crack solution initiates in every case, and five to six after the proposed method. Two properties of the regularized description delay it. Its AT2 crack density function departs from the undamaged stress–strain response before the critical stress is reached [86], so damage and compliance accumulate ahead of the peak while a sharp crack stays elastic. Its discretization raises the efective toughness to $G _ { c } ( 1 + h / 2 \ell )$ [25], with h its element size, which raises a Grifith peak load by the square root of that factor. Its fracture energy is also held constant, where the proposed method and the sharp-crack solution evaluate it at the tip temperature.

The sharp-crack solution also supplies the temperature field, and with it a test of the choice of Section 4.1 to leave the temperature without tip enrichment. At load step 47 of the heated case, four steps before initiation, the adiabatic crack splits the field by 18.6 K of the 30 K imposed, the largest diference between the two face temperatures along the notch, against 18.5 K in the reference. On a 501 × 501 grid, excluding the row of the notch itself, the trained temperature agrees with the finite element field to 0.08 K root mean square. Farther than 0.05 mm from the tip the largest pointwise deviation is 0.43 K, which matches the discretization scatter of the reference itself, measured by repeating it on a mesh twice as fine. Closer to the tip the deviations are larger, and there the ring-sampled tip temperature of (88) agrees with the reference to 0.24 K.

Fig. 15 shows the fields of the heated case from the same run as Fig. 14. The crack leaves the notch at its tip and turns downward. The region below the crack is connected to the heated upper edge only around the tip, so it stays near the initial temperature and the isotherms wrap around the advancing tip, while the displacement jumps across the faces.

The downward kink is not prescribed. It follows from the extracted SIFs through the maximum hoop stress criterion of Section 5.4, and the resulting path is consistent with the phase-field reference. The SIFs are shown in Fig. 16. Before initiation the specimen is shear dominated. $K _ { I I }$ grows linearly with the applied displacement to between 140 and 175 $\mathrm { N } / \mathrm { m m ^ { 3 / 2 } }$ at the last step before growth, while $K _ { I }$ stays small and is ofset by the thermal load, to about $- 4 0 ~ \mathrm { N / m m ^ { 3 / 2 } }$ in the heated case and about +36 in the cooled one, since heating expands the material across the notch and cooling contracts it. The first extension leaves the notch plane at the kink angle of this mixed-mode state and turns the crack onto the plane where the shear nearly vanishes. $K _ { I I }$ drops to between −25 and $+ 1 1 ~ \mathrm { N / m m ^ { 3 / 2 } }$ and $K _ { I }$ rises to between $1 7 7$ and 217, so growth is mode I dominated from there. The criterion ratio χ then stays close to $\chi _ { c }$ for the rest of the run, as a Grifith criterion should in a displacement-controlled test, and the thermal ordering of initiation is visible in both panels, cooling reaching the threshold first.

![](images/e47cf39fe1bc4256b0a18d8761ed2401a1726f786be0af20f870747702e38bfa.jpg)

Fig. 14. Force–displacement response of the graded single-edge notched shear test at $\Delta T = - 3 0 { \mathrm { : } }$ 0 and +30 K, with the colors and line styles of Fig. 11. The stars mark initiation in the sharp-crack finite element solution, which holds the crack stationary.  
![](images/48ae318ec9a6b84222bfc5406fc632548ccb7c7f78e4eda9d8d6befa8ee6e62c.jpg)  
Fig. 15. Temperature and horizontal displacement in the graded single-edge notched shear test at $\Delta T = + 3 0 \mathrm { K }$ at load steps 0, 70 and 140 of 141. Each row shares one color scale. The region below the crack stays near the initial temperature since the crack faces are adiabatic.

In the heated case $K _ { I }$ is negative before initiation. With the traction-free faces of Section 2.4 this means that the faces interpenetrate slightly near the tip. The embedding admits a jump of either sign, and only a contact condition, keeping the faces from crossing and transmitting a compressive traction between them, would exclude a negative normal jump. The reading that authorized the first extension was taken in that state, $K _ { I } = - 4 0$ and $K _ { I I } = 1 7 8 ~ \mathrm { N / m m ^ { 3 / 2 } }$ at load step 51, with $\chi = 1 . 0 3$ and a kink angle of $- 7 5 ^ { \circ }$ , and the positive $K _ { I }$ of ${ \mathrm { F i g } } .$ . 16 is recorded after the crack has turned. Contact would relieve the compressive $K _ { I } .$ , which in the computation is what delays the heated case relative to the isothermal one, so under contact that delay would shrink and the first kink would lie nearer the pure mode II angle. The sharp-crack reference rests on the same traction-free faces, so the comparison with it is made under this assumption.

The admissibility checks of Section 5.2 refused the reading at three to five load steps per case, and two extensions of the heated case were taken on a clipped sweep.

![](images/46936e401633bd101fa52f623d37be15aa2d51dce1f7887afd901e7a35a6c103.jpg)  
(a)

![](images/b6d618ca2ae9fdd7eaabbdfbc5dad83ab9b24fe391e425bbaeb1a71ed8735264.jpg)  
(b)  
Fig. 16. SIFs of the graded shear test against the applied displacement: (a) $K _ { I } ,$ (b) $K _ { I I } ,$ with the legend of (a). $K _ { I I }$ grows while the notch is stationary and collapses when the first extension turns out of its plane, where $K _ { I }$ takes over. Readings after the tip has reached the separation margin δ of Section 5.5 are not plotted.

The computed curves end 11% to 19% earlier than the phase-field reference separates, in fractions of its separation displacement, but the two ends are diferent events. The computation stops when the tip comes within the separation margin of Section 5.5, here $\delta = 0 . 0 3$ mm, the last admitted extraction being taken at about 0.032 mm from the boundary with the contour at full size, whereas the reference continues to full separation. The crack covered the 0.03 mm preceding the margin in 18, 15 and 19 load steps, and crossing the margin at the same rate would take as many again, 15% to 18% of the displacement at the margin, so the margin alone could account for a diference of this size. Arrival at the margin is quantized by the crack increment, which is why the cooled and isothermal cases stop within one load step of each other.

## 6.4 Notched cruciform test under combined thermo-mechanical loading

The notched cruciform specimen has served as a benchmark of thermoelastic crack growth since Prasad et al. [87] introduced it and computed it with the dual boundary element method. It has since been solved with enriched finite elements [20, 88], meshfree and smoothed formulations [89, 90], a moving mesh technique [91], and phase-field models [37, 92]. This history makes it the one propagating example whose predicted paths can be laid over a body of independent published solutions.

The specimen and its boundary data are shown in Fig. 17. It is a cruciform plate with a 10 mm notch at $4 5 ^ { \circ }$ from its lower right re-entrant corner, and three load cases are run, following the benchmark’s standard program. Case I applies the mechanical load on the top edge alone, isothermally. Case II applies the temperatures alone, $+ 1 0 ~ ^ { \circ } \mathrm { C }$ on the top arm and $- 1 0 ~ ^ { \circ } \mathrm { C }$ on the bottom, with the top edge left traction free, as the published treatments leave it. Case III applies both loads at once. In every case the side arms are held at the reference temperature with $u = 0$ , and the bottom edge is fixed.

The published solutions do not all solve the same problem. They share the geometry and the three load cases, and fall into two families that difer in how the mechanical load is applied and in the material data. The original statement of Prasad et al. [87] applies a traction of 10 Pa on the top edge, with $\nu = 0 . 3$ and $\alpha = 1 . 6 7 \times 1 0 ^ { - 5 } ~ \mathrm { { ^ circ C ^ { - 1 } } }$ , and grows the crack by a prescribed number of increments, so it involves no fracture resistance. The restatement of Mandal et al. [37] prescribes a displacement of 0.05 mm on the top edge instead, with $\nu = 0 . 2 , \alpha = 6 . 0 \times 1 0 ^ { - 4 } \ { } ^ { \circ } \mathrm { C } ^ { - \mathrm { 1 } }$ and a fracture energy, in plane stress. We compute the second statement, with its material data and its plane stress condition unchanged, since the propagation criterion of Section 5.5 requires the fracture energy that only this statement supplies. The diferences in Poisson’s ratio, thermal expansion and plane condition among the solutions compared below are therefore diferences between the two published statements, not choices made here.

![](images/c7534f7df7a7974a57a7faaf3f0d86cf6df98bd7a23f9ff88433a89ee289f1c5.jpg)  
Fig. 17. Geometry of the notched cruciform specimen and the three load cases. (a) Case I, the top edge displacement $v ^ { * }$ alone, isothermal. (b) Case II, the temperatures alone, with the top edge traction free. (c) Case III, both. In all three the side arms are held at the reference temperature with u = 0 and the bottom edge is fixed. Dimensions are marked on (a). The temperature shading is schematic.

The material parameters are listed in Table 4, quoted exactly as Mandal et al. state them, in their consistent $\mathrm { N - m } { - } ^ { \circ } \mathrm { C }$ system. The vanishing mass density makes the conduction problem of Section 2.2 steady at each load step, which matches their treatment of the thermal field. The tensile strength $f _ { t }$ enters only their phase-field model, and the proposed method uses the fracture energy $G _ { f }$ as a constant $G _ { c 0 }$ , without the temperature dependence of (89), which the published statement does not contain. Each case is run for 101 load steps over which the imposed displacement and the imposed temperatures rise linearly from zero to their final values, so that in case III their ratio is the same at every step. At most one crack increment of $\Delta a = 4$ mm is taken per step, nearly the same fraction of the specimen width as in the single-edge notched tests, so that the growth is paced by the load stepping in the same way as the phase-field solution it is compared with [37]. A step whose ratio stays above the threshold after its extension keeps that surplus for the next step, with the consequence for the criterion ratio recorded in Appendix A. The test is run without a separation margin, and growth stops when the tip reaches the boundary of the specimen. The specimen occupies part of its square bounding box, and the corner regions outside the arms receive no integration points. Each case is run three times from diferent network initializations. The integration points are drawn independently in every run, so the three runs are independent repeats, and the spread quoted below is the spread over them.

Table 5 lists the published solutions used below with the statement each solves. When only one load acts, as in cases I and II, its amplitude cancels out of the growth direction. The two statements then still difer in Poisson’s ratio, in the plane condition and in the traction against the displacement on the top edge, but their published single-load paths agree with one another within the spreads quoted below, so all six solutions are drawn for these cases. When both loads act, as in case III, the ratio of the mechanical to the thermal load decides the path, the two families difer in thermal expansion by a factor of 36 and in how the mechanical load enters, and only the two displacement-controlled solutions are drawn. Merging the families into one band for case III would produce a spread of nearly $4 0 ^ { \circ }$ that no single problem exhibits.

Table 4. Material parameters for the notched cruciform test.
<table><tr><td>Parameter</td><td>Symbol Value</td><td></td></tr><tr><td>Young&#x27;s modulus</td><td>E</td><td> $2 1 8 . 4 \times 1 0 ^ { 3 } ~ \mathrm { N / m ^ { 2 } }$ </td></tr><tr><td>Poisson&#x27;s ratio</td><td> $\nu$ </td><td>0.2</td></tr><tr><td>Fracture energy</td><td> $G _ { f }$ </td><td> $2 . 0 \times 1 0 ^ { - 4 } \ \mathrm { N / m }$ </td></tr><tr><td>Tensile strength</td><td> $f _ { t }$ </td><td> $1 2 0 ~ \mathrm { N / m ^ { 2 } }$ </td></tr><tr><td>Thermal expansion coefficient</td><td> $\alpha$ </td><td> $6 . 0 \times 1 0 ^ { - 4 } ~ ^ { \circ } \mathrm { C } ^ { - 1 }$ </td></tr><tr><td>Density</td><td>0</td><td> $0 . 0 ~ \mathrm { k g / m ^ { 3 } }$ </td></tr><tr><td>Thermal conductivity</td><td> $k$ </td><td> $1 . 0 \ \mathrm { W / ( m } ^ { \circ } \mathrm { C ) }$ </td></tr><tr><td>Specific heat capacity</td><td> $C _ { p }$ </td><td> $1 . 0 ~ \mathrm { N m } / ( \mathrm { k g } ^ { \circ } \mathrm { C } )$ </td></tr><tr><td>Reference temperature</td><td> $T _ { 0 }$ </td><td> $0 ~ ^ { \circ } \mathrm { C }$ </td></tr></table>

Table 5. Published solutions of the notched cruciform specimen and the statement of the problem that each solves. Every source that states Young’s modulus gives $\bar { E } = 2 1 8 . 4 \times 1 0 ^ { 3 } ~ \mathrm { N / m ^ { 2 } }$ . The traction on the top edge is 10 Pa and the displacement 0.05 mm. An asterisk marks data that the source does not state, taken from Prasad et al. [87], to whose statement of the problem the source refers, and a dash marks what neither states. The last column gives the load cases in which the path is compared, and F marks the two solutions whose SIFs are compared.
<table><tr><td>Solution</td><td>Crack</td><td>Method</td><td>Load</td><td>ν</td><td> $\alpha \ ( ^ { \circ } \mathrm { C } ^ { - 1 } )$ </td><td>Plane</td><td>Used in</td></tr><tr><td>Wang [89]</td><td>sharp</td><td>meshfree</td><td>traction</td><td> $0 . 3 ^ { * }$ </td><td> $1 . 6 7 \times 1 0 ^ { - 5 * }$ </td><td></td><td>I, II</td></tr><tr><td>Nguyen et al. [88]</td><td>sharp</td><td>enriched elements</td><td>traction</td><td> $0 . 3 ^ { * }$ </td><td> $1 . 6 7 \times 1 0 ^ { - 5 * }$ </td><td></td><td>I, II</td></tr><tr><td>Chen et al. [90]</td><td>sharp</td><td>smoothed elements</td><td>traction</td><td>0.3</td><td> $1 . 6 7 \times 1 0 ^ { - 5 }$ </td><td>strain</td><td>I, II, F</td></tr><tr><td>Greco et al. [91]</td><td>sharp</td><td>moving mesh</td><td>traction</td><td>0.3</td><td> $1 . 6 7 \times 1 0 ^ { - 5 }$ </td><td>strain</td><td>I, II, F</td></tr><tr><td>Mandal et al. [37]</td><td>phase field</td><td>finite elements</td><td>displacement</td><td>0.2</td><td> $6 . 0 \times 1 0 ^ { - 4 }$ </td><td>stress</td><td>I, II, III</td></tr><tr><td>Chen et al. [92]</td><td>phase field</td><td>scaled boundary elements</td><td>displacement</td><td>0.2</td><td> $6 . 0 \times 1 0 ^ { - 4 }$ </td><td></td><td>I, II, III</td></tr><tr><td>Proposed method</td><td>sharp</td><td></td><td>displacement</td><td>0.2</td><td> $6 . 0 \times 1 0 ^ { - 4 }$ </td><td>stress</td><td></td></tr></table>

Fig. 18 lays the computed paths over the published ones. The three repeats of a case stay within 1.6 mm of one another, under half the crack increment, so one is drawn. To compare the paths by one number, each is summarized by its chord angle, the angle from the global x axis of the line from the notch tip to the point where the path crosses a circle of radius 12 mm about it, three crack increments from the tip and a distance that every published path reaches. The angle measures the initial direction of the path and not its whole course. Table 6 collects the angles. In case I the three repeats fall $1 . 5 ^ { \circ }$ to $1 . 7 ^ { \circ }$ above the published span and in case $\mathrm { ~ I ~ I ~ 2 . 4 ^ { \circ } }$ to $4 . 1 ^ { \circ }$ below it, the spans themselves being $5 . 7 ^ { \circ }$ and $1 8 . 9 ^ { \circ }$ wide, and in case III they fall $1 . 3 ^ { \circ }$ to $3 . 2 ^ { \circ }$ below the closer of the two displacement-controlled references, which are $5 . 4 ^ { \circ }$ apart. Beyond the chord radius the figure shows how far the agreement extends along each path.

Table 6. Chord angles of the crack paths at 12 mm from the notch tip, measured from the global x axis. The proposed method is given as the first of three repeats with the range over the three, the published solutions as the span of those drawn in Fig. 18, and the last column gives the distance of the repeats from the nearest published value.
<table><tr><td>Case</td><td>Proposed method</td><td>Published</td><td>Outside the published span</td></tr><tr><td>I, mechanical</td><td> $1 8 0 . 6 ^ { \circ } \ : \left( 1 8 0 . 5 ^ { \circ } \ : \mathrm { t o } \ : 1 8 0 . 7 ^ { \circ } \right)$ </td><td> $1 7 3 . 3 ^ { \circ } \mathrm { \ t o \ 1 7 9 . 0 ^ { \circ } }$ </td><td> $1 . 5 ^ { \circ }$  to  $1 . 7 ^ { \circ }$  above</td></tr><tr><td>II, thermal</td><td> $7 4 . 1 ^ { \circ } \ : \left( 7 2 . 9 ^ { \circ } \ : \mathrm { t o } \ : 7 4 . 6 ^ { \circ } \right)$ </td><td> $7 7 . 0 ^ { \circ } \mathrm { \ t o \ 9 5 . 9 ^ { \circ } }$ </td><td> $2 . 4 ^ { \circ }$  to  $4 . 1 ^ { \circ }$  below</td></tr><tr><td>III, both</td><td>128.7° (128.7° to 130.5°)</td><td> $1 3 1 . 9 ^ { \circ }$  and  $1 3 7 . 3 ^ { \circ }$ </td><td> $1 . 3 ^ { \circ } \mathrm { ~ t o ~ } 3 . 2 ^ { \circ }$  below</td></tr></table>

The SIFs are compared as well, in an exploratory way. Published SIFs exist only for the tractioncontrolled statement, each tabulated along the path of its own solution, and those of the proposed method are likewise read along its own path. In case I the normalization below needs the reaction force at each crack length, which the propagating runs do not record, so the crack is held at each position of the path it grew, the final load of the case is applied, the fields are trained and the SIFs are extracted. In cases II and III the SIFs are those of the propagating runs, at the load steps whose radius sweep passed, with the median taken over the steps spent at one crack length. Each family is normalized by its own load measure $K _ { 0 }$ , which is $\sigma _ { \mathrm { e f f } } \sqrt { \pi a _ { 0 } }$ in the mechanical case, where $\sigma _ { \mathrm { e f f } }$ is the reaction force on the loaded arm divided by the arm width at that crack length, and $( 1 - \nu _ { r } ) E \alpha \Delta T \sqrt { \pi a _ { 0 } }$ in the thermal case and in case III, with ∆T the amplitude at that load step, a the notch length and the crack advance $a - a _ { 0 }$ counted along the path. The factor $1 - \nu _ { r }$ , with $\nu _ { r } = 0 . 3$ the Poisson’s ratio of the traction-controlled statement, brings the thermal SIFs of the plane stress computation to the plane strain scale of the published curves. The families still difer in Poisson’s ratio, in plane condition, in how the load enters and in path, so part of any residual discrepancy belongs to the diference in problem statement rather than to the method, and the path comparison above remains the primary validation.

![](images/fb8c6d9a7e7eabcba750493361b96bbce46c0403ff965ecd2dc55e0dd2cc4259.jpg)  
(a)

![](images/02bef9a76dea1ebea526cced9901831f6df672ecf00146611663f97a1033e22e.jpg)  
(b)

![](images/62e88bab8ba913997f36fee437b4c7a0198c31fe5ea8d73bde2de81a858123b9.jpg)  
(c)  
Wang 2015, sharp crack, traction-controlled — Nguyen et al. 2017, sharp crack, traction-controlled Chen et al. 2016, sharp crack, traction-controlled Greco et al. 2021, sharp crack, traction-controlled -e- Mandal et al. 2021, phase field, displacement-controlled -e- Chen et al. 2024, phase field, displacement-controlled proposed method, sharp crack, displacement-controlled  
Fig. 18. Crack paths of the three load cases against the published solutions, labeled by crack description. (a) Case I, mechanical. (b) Case II, thermal. (c) Case III, both, with the two displacement-controlled references only, since the traction-controlled family solves a diferent combined problem. One of the three repeats is drawn per case, the other two staying within 1.6 mm of it.

Fig. 19 collects both modes for all three cases, with the curves of Chen et al. [90] and Greco et al. [91] overlaid where comparable ones exist. In case I the extracted $K _ { I }$ lies below the published curves over most of the growth history. The ratio at matched crack length averages 0.92 against Chen et al. and 0.95 against Greco et al., 0.93 over both, and ranges from 0.89 to 1.04 along the history, with the average moving by 0.26% between repeats. Mode II stays small, as it does for a crack that follows the maximum hoop stress direction, except at the notch before the first kink. Away from the notch $K _ { I I } / K _ { I }$ lies between −0.038 and +0.018 with a median magnitude of 0.013, against published values between −0.009 and +0.030.

![](images/471c14752cc7fc7392a3d82eb165831d636412c82cc26d84f801022bb103e552.jpg)

![](images/aff4c2faf46ac386a1437084c571e1dff643a3b82d6a7dea7c49d984d415b110.jpg)  
(b)

![](images/00fb28b2699e535e0653e25bb12ba7bc802902f5ca1e214a27afb23daab5881a.jpg)

![](images/fba3e30f641bf9699a4acb41aa3f2b60e0a0bb5650e1238ee893d8a09a3d31e2.jpg)

![](images/94c997518d5caf4b9b3524380515407e6feca38620e3a3fb689df415d451fafd.jpg)  
(e)

![](images/deeda68cfbcb0bb74d2b6ecde9933182497312a70abc16f01790324b364dd82d.jpg)  
(f)  
Chen et al. 2016, sharp crack, traction-controlled  
Greco et al. 2021, sharp crack, traction-controlled  
◆ proposed method, sharp crack, displacement-controlled

Fig. 19. Both SIFs against the crack advance $( a - a _ { 0 } ) / a _ { 0 }$ , read along the path the proposed method grows for itself, as the published curves are read along theirs. Case I is read with the crack held at each position of that path under the final load, cases II and III from the propagating runs. Markers give the median over the three repeats, which lie within 3% of each other in case I, 11% in case II and 9% in case III, except at $( a - a _ { 0 } ) / a _ { 0 } = 1 . 6$ in (c), where one repeat passed the radius sweep at a single load step, and at 2.8 and 3.2 in case II, where two repeats and one passed the radius sweep. Case III has no reference, since the only published SIFs under both loads are for a cruciform with cooled side arms.

In case II the extracted $K _ { I }$ rises and declines with the published curves but lies below them, by 0.015 in $K _ { I } / K _ { 0 }$ at $( a - a _ { 0 } ) / a _ { 0 } = 0 . 4$ , about 0.02 up to 2.0 and about 0.03 at 2.8, so that it declines more steeply and its ratio to the published curves falls from 0.88 to 0.48 over that range. At 2.8 the readings come from two repeats and at 3.2 from one. $K _ { I I } / K _ { 0 }$ stays between 0.004 and 0.019 against published values below 0.007. The residual grows with crack length, and since the two families difer in Poisson’s ratio, in plane condition and in path at once, the comparison does not attribute it to one of them. For case III no published SIFs exist for this combination of loads. Its three repeats agree in $K _ { I }$ to within 9%.

## 7 Conclusions

We have presented an extended deep energy method for thermo-mechanical crack propagation in which the crack is represented explicitly, as a polyline, and advanced by a criterion of classical fracture mechanics. The transient heat conduction equation is recast as an incremental functional and minimized in a staggered manner with the thermoelastic potential energy. Each field is represented by its own network, and the two networks share one embedding of the crack polyline, which admits the displacement jump and the temperature jump across the crack faces, the adiabatic condition on the faces following from the thermal functional itself. The displacement network is enriched with the leading asymptotic terms at the tip. The integration points are redrawn during training and densified near the tip. At each load step the SIFs are extracted from the trained fields by the interaction integral, the fracture energy is evaluated at the tip temperature, and the crack is extended when the criterion is met.

Under thermal load the interaction integral has to include the area term of Wilson and Yu in addition to the contour term. Without it, the SIF read from the same trained field changes with the contour radius, and with it, the reading on the stationary benchmark is path independent to between 3% and 7% across a fourfold span of radii. Every extraction is also taken as a sweep over contour radii, and a reading that drifts across the sweep by more than a declared tolerance authorizes no extension, so that the crack does not advance on SIFs the fields do not support.

The method was assessed on four benchmark problems. On the stationary edge crack under thermal load the extracted SIF agrees with the published value to 0.11% on the finest of three grids. On the single-edge notched tension test the peak load agrees with the phase-field reference to within 4.1% in the three thermal cases, with no systematic sign, and the displacement at peak load increases by 89% from the cooled to the heated case. On the graded shear test the crack paths agree with the phase-field reference, and initiation agrees to within one load step with an independent sharp-crack finite element solution in all three thermal cases. On the notched cruciform specimen the predicted chord angles fall 1.3<sup>◦</sup> to 4.1<sup>◦</sup> outside the nearest published curve, where the published solutions themselves spread over $5 . 4 ^ { \circ }$ to $1 8 . 9 ^ { \circ }$ per case. The SIFs of the cruciform were compared in an exploratory way with published curves that solve the traction-controlled statement of the benchmark rather than the displacementcontrolled one computed here. Read along the paths the method grows, the mechanical case gives a ratio averaging 0.93 over the growth history, and in the thermal case $K _ { I }$ follows the published decline below it by an ofset that grows from 0.015 to 0.03 in units of the load measure. Run-to-run scatter was measured on every example and is reported beside each result.

The scope of the study is limited in several respects. All examples are two-dimensional and quasistatic. A single crack is followed, and neither branching nor merging is treated. The coupling is one way for a fixed crack, so no mechanical dissipation returns to the thermal problem, and the mechanical solution enters the conduction problem only through the crack it advances. The crack faces are taken open and traction free, so a loading state that closes the crack would require a contact condition. In the heated shear case the first extension was taken from such a state, and contact would shorten the delay of initiation that the computation shows. The crack advances by a prescribed increment along a direction evaluated once per extension, so a path is resolved to that increment, and the last increment before separation is not resolved, which in the tension test means the whole post-peak branch. The extraction contour is tied to the increment and has to span about two background elements, which sets a lower limit on the increment for a given grid. The tolerance of the radius sweep is a chosen value rather than a derived bound. For the cruciform specimen the published solutions solve two statements of the problem that share the geometry but not the loads, and no published SIFs exist for the combined load case, so that case is compared on its path only. A propagating case takes from 3.5 to 12 hours on one GPU, as reported in Appendix A, and no propagating reference solution was timed on the same task.

Extending the method to multiple and merging cracks, to crack nucleation, to three-dimensional problems, and to a two-way coupling in which the mechanical field feeds back into the thermal one is left to further work.

## Data availability

Data will be made available on request.

## Code availability

The repository https://github.com/HannnZH/Extended-DEM-ThermoMechanical-Crack contains the solver, the phase-field and sharp-crack finite element reference solvers and the scripts that reproduce the numerical examples.

## Acknowledgements

This research was undertaken with the assistance of resources from the Gadi supercomputer [93] at the National Computational Infrastructure (NCI Australia), an NCRIS enabled capability supported by the Australian Government, and from the Katana cluster [94] at UNSW Sydney, supported by Research Technology Services.

## A Numerical parameters, cost and sensitivity

The numerical parameters of the four examples are collected in Table A.1. Each solve runs at most its stated number of Adam iterations and stops earlier when the mean loss over the last 500 iterations has fallen by less than $1 0 ^ { - 4 }$ of the mean over the 500 before it on three consecutive checks, made every 250 iterations, so that each mean spans one resampling interval. The partial retrainings after an extension, 1000 thermal and 3000 mechanical iterations in the notched tests and 1500 and 3000 in the cruciform, run to their count. The network weights are initialized randomly from the seed listed in Table A.1. The seed does not govern the draw of the integration points, so the repeats of the notched tests, run with the same seed, difer only in that draw, whereas those of the cruciform also difer in the seed.

The stationary crack and the cruciform have zero mass density, so their conduction problem is steady and their time increment is written as unity. The SIFs of the cruciform, 150 mm across, are read on a contour of 3.6 mm. Cases I and III use the displacement network of the notched tests with a decay length $\ell _ { \rho } / \alpha _ { \rho } = 5 h$ of the embedding, and case II a larger network, 30 902 parameters against 5402, with $\ell _ { \rho } / \alpha _ { \rho } = 0 . 3 h$

The stationary benchmark is the one example trained under load on a single draw of its points, 82 944 to 331 776 on its three grids, with 2000 L-BFGS iterations after Adam, and the agreement of its SIF with the published value is the check that no spurious layer of the kind described in Section 4.2 formed there.

The flux weight $\lambda _ { q }$ multiplies the mean squared normal flux $( k \partial T / \partial n ) ^ { 2 }$ over $N _ { q }$ points on the crack faces in physical units, so its natural scale is set by k and by the imposed temperature diference over the size of the specimen, which is why it difers by orders of magnitude between the examples. It was set in each example so that the term remains a regularizer behind the natural condition of the thermal functional. At convergence its share of the thermal loss is $1 . 5 \times 1 0 ^ { - 7 }$ of the conduction energy on the stationary crack and $2 . 5 \times 1 0 ^ { - 6 }$ on the heated shear test at load step 47, where the root mean square normal flux on the faces is 6% of the flux scale in the body.

Table A.1. Numerical parameters of the four examples. Layer widths run from the input $( x , y , \rho )$ to the output. The thermal network has a sinusoidal first layer with $\omega _ { 0 } = 3 0$ , the displacement network hyperbolic tangents throughout. All four examples share nine integration points per background element, the thermal network 3–50–50–50–1 and the learning rate $1 0 ^ { - 3 }$ . The stationary example has no crack increment and no redraw. Lengths are in millimeters, the notched tests being computed on the unit square of side 1 mm and the cruciform in its physical dimensions, and $\ell _ { \rho }$ of (45) is given in millimeters or in units of the background element size h. The cruciform entries marked † are those of cases I and III, and case II uses the displacement network 3–100–100–100–100–2 with $\alpha _ { \rho } = 2 0$
<table><tr><td></td><td>Stationary crack Section 6.1</td><td>Tension Section 6.2</td><td>Graded shear Section 6.3</td><td>Cruciform Section 6.4</td></tr><tr><td>Background elements across the width</td><td>48, 72, 96</td><td>100</td><td>100</td><td>150</td></tr><tr><td>Resampling interval</td><td>none</td><td>500 iterations</td><td>500 iterations</td><td>500 iterations</td></tr><tr><td>Displacement network Adam iterations per load</td><td>3-50-50-50-2</td><td>3-50-50-50-2</td><td>3-50-50-50-2 3000 / 3000,</td><td> $3 - 5 0 { \mathrm { - } } 5 0 { \mathrm { - } } 5 0 { \mathrm { - } } 2 ^ { \dagger }$ </td></tr><tr><td>step, thermal / mech.</td><td>8000</td><td>3000 / 3000</td><td>5000 / 5000 at step 1</td><td>2000 / 3000</td></tr><tr><td>Unloaded step 0, Adam then L-BFGS, thermal / mech.</td><td></td><td>5000 / 7000, then 2500 / 3000</td><td>3000 / 3000, then 3000 / 1500</td><td>5000 / 7000, then 2500 / 3000</td></tr><tr><td>L-BFGS iterations after Adam, loaded steps</td><td>2000</td><td></td><td></td><td></td></tr><tr><td>Time increment  $\Delta t$ </td><td>1 s, steady</td><td> $1 0 ^ { - 8 } \mathrm { ~ s ~ }$ </td><td> $2 . 5 \times 1 0 ^ { - 8 } \mathrm { ~ s ~ }$ </td><td>1 s, steady</td></tr><tr><td>Crack increment  $\Delta a$ </td><td>none</td><td>0.025 mm</td><td>0.025 mm</td><td>4 mm</td></tr><tr><td>Extraction contour r</td><td>3.2h (0.033, 0.022,</td><td> $0 . 9 \Delta a$ </td><td> $0 . 9 \Delta a$ </td><td> $0 . 9 \Delta a$ </td></tr><tr><td>Kink angle cap</td><td>0.017 mm)</td><td> $6 0 ^ { \circ }$ </td><td>80°</td><td></td></tr><tr><td>Tip densification radius,</td><td></td><td>0.08 mm, 8</td><td>0.08 mm, 8</td><td> $6 0 ^ { \circ }$  3 mm, 48</td></tr><tr><td>factor Tip temperature ring  $r _ { T }$ </td><td></td><td> $2 \times 1 0 ^ { - 3 }$  mm</td><td> $2 \times 1 0 ^ { - 3 }$  mm</td><td>0.25 ∆a</td></tr><tr><td> $( N _ { T } = 2 4 )$  Embedding decay  $\alpha _ { \rho } , \ell _ { \rho }$ </td><td>1,5h</td><td>20, 1 mm</td><td>50, 1 mm</td><td>1.2†, 6 mm</td></tr><tr><td> $( p _ { \rho } = 1 )$ </td><td></td><td></td><td></td><td></td></tr><tr><td>Endpoint overlap  $\delta _ { c } ~ ( \mathrm { m m } )$ </td><td>0  $1 0 ^ { - 4 }$ </td><td> $0 . 0 1 2 5$ </td><td> $0 . 0 1 2 5$ </td><td>2</td></tr><tr><td>Extraction face offset (mm)</td><td> $1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Flux face offset (mm)  $N _ { q }$ </td><td> $1 0 ^ { - 2 } , 2 0 0$ </td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Flux weight  $\lambda _ { q } ,$  samples Enrichment decay  $m _ { u }$ </td><td>5</td><td> $1 0 ^ { - 6 } , 1 5 0$  5</td><td> $1 0 ^ { - 6 } , 1 5 0$  5</td><td>10−8, 200 0.05</td></tr><tr><td> $( \mathrm { m m ^ { - 1 } } , q _ { u } = 1 )$  Extensions per load step, at</td><td></td><td></td><td></td><td></td></tr><tr><td>most</td><td></td><td>35</td><td>50</td><td>1</td></tr><tr><td>Ratio cap of the convergence check</td><td></td><td>5</td><td></td><td>100</td></tr><tr><td>Seeds</td><td>2025</td><td>2025</td><td>2025</td><td>2025, 1, 2</td></tr></table>

The implemented embedding of Section 3.2 departs from the conditions (31) and (32) in three local ways. First, the embedded jump closes only $\delta _ { c }$ beyond the tip, so (32) does not hold in a strip of width $\delta _ { c }$ ahead of the tip, whose efect on the temperature is measured in Appendix B. For $\delta _ { c } = 0 \ :$ the setting of the stationary benchmark, the jump closes at the tip and the enrichment alone supplies the opening. Second, near a corner the sign change of $\tilde { d }$ departs from the polyline, by up to 0.004 mm within one increment either side of the $7 4 ^ { \circ }$ corner at which the crack of Fig. 2 leaves the notch and by up to 0.0005 mm at corners of under $1 0 ^ { \circ }$ . Within those stretches two face samples at the ofsets of Table A.1 can fall on one side, and the flux term then acts on one face only, which is why Fig. 2(c) samples the jump at 0.006 mm. Third, the projected arc length has a discontinuous gradient on the lines normal to a segment through its endpoints, where the projection (41) and the end terms of (43) change slope. The networks inherit these gradient discontinuities on curves of zero area, which the redrawn integration points meet with probability zero and the extraction contour crosses at isolated

points.

Away from the corners the face ofsets attenuate the sampled jump through the normal decay of the embedding, by the factor $\exp [ - \alpha _ { \rho } ( \epsilon _ { c } / \ell _ { \rho } ) ^ { p _ { \rho } } ]$ , which with the values of Table A.1 reduces the sampled jump by at most 2.5% in the notched tests and 15% for the flux ofset of the cruciform.

The stationary benchmark trains and extracts in about 17 minutes on a laptop GPU at its finest grid. The shear cases take 7.3 hours isothermal and 11.5 to 11.9 hours under thermal load for their 141 load steps on one V100, the tension cases 8.4 to 9.0 hours for their 106, and a cruciform case 3.5 to 4 hours for its 101 on one H200. Both networks are retrained at every load step and after every extension, so that a shear case comprises about 160 staggered solves of 2.5 to 4.5 minutes each. In the sensitivity study below, 16 and 4 integration points per element instead of 9 changed the run time by +9% and −2%, and network widths between 30 and 70 by 2.4%, against 2.1% between repeats of the same configuration, so the cost is set by the number of optimizer iterations rather than by the size of the representation. No propagating reference solution was timed on the same task. The sharp-crack reference of Appendix B covers its three thermal cases to initiation in 90 seconds on one laptop CPU core but holds the crack stationary, and the cost of the phase-field references was not measured.

The three thermal cases of the graded shear test were each run twice, and Table A.2 gives the load step at which the crack first advances. The pairs initiate 2, 1 and 0 load steps apart at $\Delta T = 0 , - 3 0$ and +30 K, with peak loads 4.5%, 2.4% and 0.2% apart. Every run lies within one load step of the sharp-crack solution.

Table A.2. Load step at which the crack first advances in the graded shear test, two runs per case, against the two references of Section 6.3. One load step is $2 . 5 \times 1 0 ^ { - 5 }$ mm of applied displacement. The phase-field entry is the step of its peak load, since it has no discrete initiation. The two runs difer only in the draw of the integration points.
<table><tr><td></td><td> $\Delta T = - 3 0 \mathrm { ~ K ~ }$ </td><td> $\Delta T = 0$ </td><td> $\Delta T = + 3 0 \mathrm { ~ K ~ }$ </td></tr><tr><td>Proposed method</td><td>40</td><td>45</td><td>51</td></tr><tr><td>Proposed method, repeat</td><td>41</td><td>47</td><td>51</td></tr><tr><td>Sharp-crack finite elements</td><td>41</td><td>46</td><td>51</td></tr><tr><td>Phase field</td><td>46</td><td>51</td><td>56</td></tr></table>

The isothermal shear test was rerun with one numerical parameter changed at a time, and Table A.3 gives the outcomes against the reported run. The repeat of that run sets the scale, 4.5% on the peak and two load steps at initiation. Every variant that grows a crack peaks 2.0% to 4.1% above the baseline, within the 4.5% of the repeat. The variants initiate at step 46 or 47 against 45, and with the increment doubled the crack reaches the separation margin after 11 or 12 extensions against 23, over nearly the same crack length. Halving the increment with the grid unchanged puts the contour at 1.1 background elements, and the radius sweep then refuses every reading. The contour has to span about two background elements, so refining the increment requires refining the grid with it.

Table A.3. Sensitivity of the isothermal shear test to its numerical parameters, one change per row. Peak forces are read inside the growth window and given relative to the baseline. The repeat difers only in the draw of the integration points and sets the run-to-run scatter against which the other rows are read.
<table><tr><td>Variation</td><td>Peak</td><td>First growth</td><td>Extensions</td></tr><tr><td>Baseline</td><td>95.9 N</td><td>step 45</td><td>23</td></tr><tr><td>Repeat, integration point draw resampled</td><td>+4.5%</td><td>47</td><td>23</td></tr><tr><td>Increment halved to 0.0125 mm, contour at 1.1 elements</td><td>no growth</td><td></td><td>0</td></tr><tr><td>Increment doubled to 0.05 mm, radii 0.9 ∆a</td><td>+2.0%</td><td>46</td><td>11</td></tr><tr><td>Increment doubled, contour pinned at 0.0225 mm</td><td>+2.8%</td><td>46</td><td>12</td></tr><tr><td>Both networks narrowed to width 30</td><td>+2.9%</td><td>46</td><td>23</td></tr><tr><td>Both networks widened to 70</td><td>+4.1%</td><td>47</td><td>23</td></tr><tr><td>Quadrature coarsened to 4 points per element</td><td>+3.1%</td><td>47</td><td>23</td></tr><tr><td>Quadrature refined to 16 points per element</td><td>+2.8%</td><td>46</td><td>23</td></tr></table>

Table A.4 gives the readings of the nine cruciform runs refused by the two checks of Section 5.2 and those released on a clipped sweep. A sweep refusal follows either a drift beyond the threshold or a load step at which no contour fitted between the tip and the boundary. The convergence check has a cap of 100 in the cruciform rather than 5, since one extension per load step leaves the ratio above the threshold between steps by construction, and it refused only readings taken after the growth window, when the tip had reached the boundary region. A repeat of case III with that check removed grows over the same window of steps 29 to 49, with 18 extensions against 18 to 19 and a chord angle $0 . 7 ^ { \circ }$ from the reported run, inside the $1 . 9 ^ { \circ }$ scatter over repeats, since the readings the check had refused are refused by the sweep instead.

Table A.4. Readings refused and released in the nine cruciform runs over 101 load steps each, the reported run of each case first. The sweep refusals are split into a drift beyond the threshold and no fitting contour. The last column gives the readings released on a clipped sweep and, after the comma, the extensions taken on them, and an asterisk marks an entry that includes one extension taken on a single radius near the boundary.
<table><tr><td>Case</td><td>Run</td><td>Extensions</td><td colspan="2">Sweep refusals drift no contour</td><td>Convergence refusals</td><td>Released, extended</td></tr><tr><td>I</td><td>reported</td><td>14</td><td>6</td><td>37</td><td>0</td><td>0,0</td></tr><tr><td>I</td><td>repeat 1</td><td>14</td><td>16</td><td>37</td><td>0</td><td>0,0</td></tr><tr><td>I</td><td>repeat 2</td><td>15</td><td>7</td><td>36</td><td>0</td><td>1,1*</td></tr><tr><td>II</td><td>reported</td><td>7</td><td>37</td><td>0</td><td>0</td><td>2,0</td></tr><tr><td>II</td><td>repeat 1</td><td>7</td><td>33</td><td>0</td><td>0</td><td>3,0</td></tr><tr><td>II</td><td>repeat 2</td><td>9</td><td>29</td><td>0</td><td>0</td><td>2,0</td></tr><tr><td>III</td><td>reported</td><td>19</td><td>3</td><td>0</td><td>51</td><td>2,2*</td></tr><tr><td>III</td><td>repeat 1</td><td>18</td><td>4</td><td>2</td><td>50</td><td>0,0</td></tr><tr><td>III</td><td>repeat 2</td><td>18</td><td>2</td><td>0</td><td>52</td><td>1,1</td></tr></table>

Two further checks on the stationary benchmark of Section 6.1 test the construction. With the enrichment amplitude started from five values between −1.22 and +2.43, one of the wrong sign, the extracted SIF lies between 0.4921 and 0.4979, a spread of 1.2%, while the trained amplitudes finish 7.8% apart, so the SIF is set by the minimization and not by the initialization. Recomputing the three grids on a diferent GPU model moves no value by more than 0.7%.

## B Reference solutions

The phase-field reference of Sections 6.2 and 6.3 is the staggered finite element solver that served as the reference in Zhang et al. [83], extended to the thermoelastic case, on a uniform mesh of 512 × 512 bilinear quadrilateral elements of size about $\ell / 5 ,$ , finer than the $\ell / 2$ of common practice [26], with 789 507 degrees of freedom for the displacement and the phase field together. It uses the AT2 crack density function in its second-order form with the hybrid split of Ambati et al. [32], in which the full energy is degraded in the equilibrium equation and the tensile part of the spectral decomposition drives the phase field. Irreversibility is enforced through the history field of Miehe et al. [27], and the pre-existing crack is imposed by setting the phase field $\phi$ to one on the nodes within 1.5 element sizes of the notch line. Within each load increment the temperature is solved first, by one backward Euler step of the transient conduction equation on the real clock of the loading, with the conductivity degraded by $( 1 - \phi ) ^ { 2 }$ so that the crack faces are adiabatic and with the phase field of the previous increment, and is then held while the displacement and the phase field are alternated to a tolerance on the change of the phase field. The thermal strain uses the plane strain coeficient $( 1 + \nu ) \alpha$ , as the proposed method does. The fracture energy is constant in temperature and, in the graded test, linear in x as Table 3 states it.

The solver was checked against the published curves of Tangella et al. [38] for the tension test before use. Its peak loads are 13%, 8% and 9% lower at $\Delta T = - 5 0$ , 0 and +50 K and its displacements at peak 16%, 4% and 3% earlier, read from their published figure. The published solution regularizes over $\ell = 0 . 0 0 5$ mm at $h = \ell / 2$ , against $\ell = 0 . 0 1$ mm at $\ell / 5$ here, and the toughness inflation $G _ { c } ( 1 + h / 2 \ell )$ of that coarser ratio accounts for 6.6% of the peak diference on its own. Against the same published curves the peaks of the proposed method lie $7 \%$ to 10% below, close to the 12% by which that inflation raises a peak load at $h = \ell / 2$

The sharp-crack finite element reference of Section 6.3 is a separate implementation, with quadratic triangles for the displacement and linear ones for the temperature on a structured mesh of 80 elements across the specimen, the notch a seam of doubled nodes, the transient conduction advanced by backward Euler with the time step and the loading history of the proposed method, and the crack held at its initial length throughout.

The reference does not use the interaction integral of Section 5.1, and it obtains the SIFs in two steps. First, the potential energy Π is computed on the mesh with the notch at its length $a _ { 0 }$ and on two meshes with the notch extended straight ahead by $\Delta a$ and by $\Delta a / 2$ , the temperature being transferred to the extended meshes. The energy released per unit extension, $[ \Pi ( a _ { 0 } ) - \Pi ( a _ { 0 } + \Delta a ) ] / \Delta a$ , approximates the straight-ahead energy release rate $G .$ and the two increments are combined by Richardson extrapolation to remove the first-order error of that diference. G gives the magnitude $\sqrt { K _ { I } ^ { 2 } + K _ { I I } ^ { 2 } } = \sqrt { E ^ { \prime } G }$ . Second, the ratio of $K _ { I }$ to $K _ { I I }$ is the ratio of the normal to the tangential opening of the crack faces, each fitted against $\sqrt { r }$ between 2 and 12 element sizes behind the tip. The signs follow the Williams convention in a frame with x along the notch toward the tip, which gives $K _ { I I } > 0$ under the applied shear, and $K _ { I }$ takes the sign of the normal opening, negative in a thermal state that closes the faces. From there the two SIFs enter the criterion exactly as in the proposed method, through the kink angle and the energy release rate of the kink $G ( \theta _ { c } )$ of Section 5.4, compared with $G _ { c } ( T _ { \mathrm { t i p } } )$ of (89) at the temperature of the tip node, which is single in the seam. The load step at which the ratio first reaches unity is the initiation marker of Fig. 14. Extending the notch costs little for the finite element reference, whereas in the proposed method it would add two trainings to every extraction.

The implementation was checked before use. On the stationary thermal edge crack of Section 6.1, with 32, 64 and 128 elements across the strip, the same implementation returns $K _ { I } / K _ { 0 } = 0 . 4 8 0 8$ 0.4889 and 0.4928, which extrapolate to 0.4966 against the published 0.500, a diference of 0.7%, and the ratio of successive diferences is 2.02, the first-order convergence a crack-tip singularity gives. On the shear specimen its elastic stifness extrapolates to 88 802 N/mm from linear elements and to 88 778 from quadratic ones, the two agreeing to 0.03%. Doubling the 80 elements of the graded thermal runs leaves the load step at which the crack first advances unchanged and moves the reaction force by 0.1% and $G ( \theta _ { c } )$ by 0.3%.

The temperature comparison of Section 6.3 extends to the crack faces. The jump between the two face temperatures, read one grid spacing either side of the notch, agrees with the reference to 1.4 K on the faces except over the last crack increment, 0.025 mm, behind the tip. This region reaches closer to the tip than the pointwise comparison of Section 6.3, which leaves out the last 0.05 mm. Ahead of the tip, where the exact field is continuous, the same diference read across the extended notch line is 1.9 K in the trained field against 0.36 K in the reference within the endpoint overlap $\delta _ { c }$ of Section 3.2, and 0.5 K against 0.28 K beyond it. The excess within the overlap is the discontinuity that the overlap prolongs past the tip by construction, and beyond it the trained field is continuous but steeper than the reference.

## C Stress intensity factor extraction under thermo-mechanical loading

The area term (85) of the interaction integral is the interaction form of the area contribution that Wilson and Yu [73] derived for the thermoelastic J-integral. It has no counterpart in the isothermal contour form, and it is omitted in some works of the literature [74], so its efect is examined here.

When the area term is omitted, the extraction does not fail visibly. The SIFs stay finite and drift with the contour radius, which is the dependence that the radius sweep of Section 5.2 measures. The area integral carries no weight function. An evaluation in domain form multiplies the integrand by a ramp that falls to zero on the outer boundary of the integration domain, whereas the quantity needed here is the diference between a vanishing contour and $\Gamma _ { r } ,$ which is the integral over the enclosed disc with unit weight.

The efect of the area term was isolated on the stationary benchmark of Section 6.1. A field trained separately on the finest grid was read twice, with the area term and from the contour alone, and Fig. C.1 shows both sweeps. With the area term the sweep flattens at $K _ { I } / K _ { 0 } = 0 . 4 9 5 , 1 . 1 \%$ below the published value, with $p = 0 . 0 3 9$ . Its diference from the reading of Section 6.1 on the same grid lies within the 1.2% spread over initial enrichment amplitudes reported in Appendix A. From the contour alone the reading falls monotonically, by 9.7% from the innermost contour to the outermost, and the value reduced from the sweep drops to 0.484 with $p = 0 . 0 7 4$ . The two readings agree to 0.7% on the smallest contour and separate to 5.9% on the largest, since the omitted quantity is an area integral and grows with the contour. On this benchmark the contour-only reading still passes the admissibility check, and the shape of the sweep, flat or monotonic, is what separates the two. Where the thermal load contributes more to the SIFs the diference is larger. On the trained field of the thermal case of the cruciform of Section 6.4 at mid load, taken from the reported run, the contour-only reading gives $p = 0 . 5 4$ , more than four times the threshold, and the reading with the area term 0.18. The run itself recorded $p = 0 . 1 7 6$ at that step and did not advance the crack.

![](images/ae0526d56cdf379df42adbeaebe6df887eff5c261c2d49dbad1660297ba24c13.jpg)  
Fig. C.1. The area term isolated on the stationary benchmark. A field trained separately on the finest grid is read with the area term (85) and from the contour alone. With the area term the sweep flattens inside the band, and from the contour alone the reading falls monotonically. The dashed line is the published value of Wang et al. [77], the shaded band 2%.

## References

[1] H. Ruan, S. Rezaei, Y. Yang, D. Gross, and B.-X. Xu. A thermo-mechanical phase-field fracture model: Application to hot cracking simulations in additive manufacturing. Journal of the Mechanics and Physics of Solids, 172:105169, 2023.

[2] Y. Shao, X. Xu, S. Meng, G. Bai, C. Jiang, and F. Song. Crack patterns in ceramic plates after quenching. Journal of the American Ceramic Society, 93(10):3006–3008, 2010.

[3] A. A. Grifith. The phenomena of rupture and flow in solids. Philosophical Transactions of the Royal Society of London. Series A, 221:163–198, 1921.

[4] G. R. Irwin. Analysis of stresses and strains near the end of a crack traversing a plate. Journal of Applied Mechanics, 24(3):361–364, 1957.

[5] M. L. Williams. On the stress distribution at the base of a stationary crack. Journal of Applied Mechanics, 24(1):109–114, 1957.

[6] J. R. Rice. A path independent integral and the approximate analysis of strain concentration by notches and cracks. Journal of Applied Mechanics, 35(2):379–386, 1968.

[7] F. Erdogan and G. C. Sih. On the crack extension in plates under plane loading and transverse shear. Journal of Basic Engineering, 85(4):519–525, 1963.

[8] G. C. Sih. Strain-energy-density factor applied to mixed mode crack problems. International Journal of Fracture, 10(3):305–321, 1974.

[9] G. I. Barenblatt. The mathematical theory of equilibrium cracks in brittle fracture. Advances in Applied Mechanics, 7:55–129, 1962.

[10] D. S. Dugdale. Yielding of steel sheets containing slits. Journal of the Mechanics and Physics of Solids, 8(2):100–104, 1960.

[11] X.-P. Xu and A. Needleman. Numerical simulations of fast crack growth in brittle solids. Journal of the Mechanics and Physics of Solids, 42(9):1397–1434, 1994.

[12] G. T. Camacho and M. Ortiz. Computational modelling of impact damage in brittle materials. International Journal of Solids and Structures, 33(20–22):2899–2938, 1996.

[13] T. N. Bittencourt, P. A. Wawrzynek, A. R. Ingrafea, and J. L. Sousa. Quasi-automatic simulation of crack propagation for 2D LEFM problems. Engineering Fracture Mechanics, 55(2):321–334, 1996.

[14] T. Belytschko, Y. Y. Lu, and L. Gu. Element-free Galerkin methods. International Journal for Numerical Methods in Engineering, 37(2):229–256, 1994.

[15] J. M. Melenk and I. Babuška. The partition of unity finite element method: Basic theory and applications. Computer Methods in Applied Mechanics and Engineering, 139(1–4):289–314, 1996.

[16] T. Belytschko and T. Black. Elastic crack growth in finite elements with minimal remeshing. International Journal for Numerical Methods in Engineering, 45(5):601–620, 1999.

[17] N. Moës, J. Dolbow, and T. Belytschko. A finite element method for crack growth without remeshing. International Journal for Numerical Methods in Engineering, 46(1):131–150, 1999.

[18] M. Stolarska, D. L. Chopp, N. Moës, and T. Belytschko. Modelling crack growth by level sets in the extended finite element method. International Journal for Numerical Methods in Engineering, 51(8):943–960, 2001.

[19] T.-P. Fries and T. Belytschko. The extended/generalized finite element method: An overview of the method and its applications. International Journal for Numerical Methods in Engineering, 84(3):253–304, 2010.

[20] M. Duflot. The extended finite element method in thermoelastic fracture mechanics. International Journal for Numerical Methods in Engineering, 74(5):827–847, 2008.

[21] J. F. Yau, S. S. Wang, and H. T. Corten. A mixed-mode crack analysis of isotropic solids using conservation laws of elasticity. Journal of Applied Mechanics, 47(2):335–341, 1980.

[22] A. KC and J.-H. Kim. Interaction integrals for thermal fracture of functionally graded materials. Engineering Fracture Mechanics, 75(8):2542–2565, 2008.

[23] W. Ai and C. E. Augarde. Thermoelastic fracture modelling in 2D by an adaptive cracking particle method without enrichment functions. International Journal of Mechanical Sciences, 160:343–357, 2019.

[24] G. A. Francfort and J.-J. Marigo. Revisiting brittle fracture as an energy minimization problem. Journal of the Mechanics and Physics of Solids, 46(8):1319–1342, 1998.

[25] B. Bourdin, G. A. Francfort, and J.-J. Marigo. Numerical experiments in revisited brittle fracture. Journal of the Mechanics and Physics of Solids, 48(4):797–826, 2000.

[26] C. Miehe, F. Welschinger, and M. Hofacker. Thermodynamically consistent phase-field models of fracture: Variational principles and multi-field FE implementations. International Journal for Numerical Methods in Engineering, 83(10):1273–1311, 2010.

[27] C. Miehe, M. Hofacker, and F. Welschinger. A phase field model for rate-independent crack propagation: Robust algorithmic implementation based on operator splits. Computer Methods in Applied Mechanics and Engineering, 199(45–48):2765–2778, 2010.

[28] K. Pham, H. Amor, J.-J. Marigo, and C. Maurini. Gradient damage models and their use to approximate brittle fracture. International Journal of Damage Mechanics, 20(4):618–652, 2011.

[29] H. Amor, J.-J. Marigo, and C. Maurini. Regularized formulation of the variational brittle fracture with unilateral contact: Numerical experiments. Journal of the Mechanics and Physics of Solids, 57(8):1209–1229, 2009.

[30] C. Kuhn and R. Müller. A continuum phase field model for fracture. Engineering Fracture Mechanics, 77(18):3625–3634, 2010.

[31] M. J. Borden, C. V. Verhoosel, M. A. Scott, T. J. R. Hughes, and C. M. Landis. A phase-field description of dynamic brittle fracture. Computer Methods in Applied Mechanics and Engineering, 217–220:77–95, 2012.

[32] M. Ambati, T. Gerasimov, and L. De Lorenzis. A review on phase-field models of brittle fracture and a new fast hybrid formulation. Computational Mechanics, 55(2):383–405, 2015.

[33] J.-Y. Wu, V. P. Nguyen, C. T. Nguyen, D. Sutula, S. Sinaie, and S. P. A. Bordas. Phase-field modeling of fracture. Advances in Applied Mechanics, 53:1–183, 2020.

[34] C. Miehe, L.-M. Schänzel, and H. Ulmer. Phase field modeling of fracture in multi-physics problems. Part I. Balance of crack surface and failure criteria for brittle crack propagation in thermoelastic solids. Computer Methods in Applied Mechanics and Engineering, 294:449–485, 2015.

[35] B. Bourdin, J.-J. Marigo, C. Maurini, and P. Sicsic. Morphogenesis and propagation of complex cracks induced by thermal shocks. Physical Review Letters, 112(1):014301, 2014.

[36] P. Sicsic, J.-J. Marigo, and C. Maurini. Initiation of a periodic array of cracks in the thermal shock problem: A gradient damage modeling. Journal of the Mechanics and Physics of Solids, 63:256–284, 2014.

[37] T. K. Mandal, V. P. Nguyen, J.-Y. Wu, C. Nguyen-Thanh, and A. de Vaucorbeil. Fracture of thermo-elastic solids: Phase-field modeling and new results with an eficient monolithic solver. Computer Methods in Applied Mechanics and Engineering, 376:113648, 2021.

[38] R. G. Tangella, P. Kumbhar, and R. K. Annabattula. Hybrid phase-field modeling of thermoelastic crack propagation. International Journal for Computational Methods in Engineering Science and Mechanics, 23(1):29–44, 2022.

[39] Hirshikesh, S. Natarajan, R. K. Annabattula, and E. Martínez-Pañeda. Phase field modelling of crack propagation in functionally graded materials. Composites Part B: Engineering, 169:239–248, 2019.

[40] M. Raissi, P. Perdikaris, and G. E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational Physics, 378:686–707, 2019.

[41] L. Lu, X. Meng, Z. Mao, and G. E. Karniadakis. DeepXDE: A deep learning library for solving diferential equations. SIAM Review, 63(1):208–228, 2021.

[42] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, and L. Yang. Physics-informed machine learning. Nature Reviews Physics, 3(6):422–440, 2021.

[43] E. Haghighat, M. Raissi, A. Moure, H. Gomez, and R. Juanes. A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics. Computer Methods in Applied Mechanics and Engineering, 379:113741, 2021.

[44] S. Cai, Z. Wang, S. Wang, P. Perdikaris, and G. E. Karniadakis. Physics-informed neural networks for heat transfer problems. Journal of Heat Transfer, 143(6):060801, 2021.

[45] W. E and B. Yu. The deep Ritz method: A deep learning-based numerical algorithm for solving variational problems. Communications in Mathematics and Statistics, 6(1):1–12, 2018.

[46] E. Samaniego, C. Anitescu, S. Goswami, V. M. Nguyen-Thanh, H. Guo, K. Hamdia, X. Zhuang, and T. Rabczuk. An energy approach to the solution of partial diferential equations in computational mechanics via machine learning: Concepts, implementation and applications. Computer Methods in Applied Mechanics and Engineering, 362:112790, 2020.

[47] V. M. Nguyen-Thanh, X. Zhuang, and T. Rabczuk. A deep energy method for finite deformation hyperelasticity. European Journal of Mechanics - A/Solids, 80:103874, 2020.

[48] N. Sukumar and A. Srivastava. Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks. Computer Methods in Applied Mechanics and Engineering, 389:114333, 2022.

[49] E. Kharazmi, Z. Zhang, and G. E. Karniadakis. hp-VPINNs: Variational physics-informed neural networks with domain decomposition. Computer Methods in Applied Mechanics and Engineering, 374:113547, 2021.

[50] J. N. Fuhg and N. Bouklas. The mixed deep energy method for resolving concentration features in finite strain hyperelasticity. Journal of Computational Physics, 451:110839, 2022.

[51] S. Goswami, C. Anitescu, S. Chakraborty, and T. Rabczuk. Transfer learning enhanced physics informed neural network for phase-field modeling of fracture. Theoretical and Applied Fracture Mechanics, 106:102447, 2020.

[52] S. Goswami, C. Anitescu, and T. Rabczuk. Adaptive fourth-order phase field analysis using deep energy minimization. Theoretical and Applied Fracture Mechanics, 107:102527, 2020.

[53] M. Manav, R. Molinaro, S. Mishra, and L. De Lorenzis. Phase-field modeling of fracture with physics-informed deep learning. Computer Methods in Applied Mechanics and Engineering, 429:117104, 2024.

[54] S. Goswami, M. Yin, Y. Yu, and G. E. Karniadakis. A physics-informed variational DeepONet for predicting crack path in quasi-brittle materials. Computer Methods in Applied Mechanics and Engineering, 391:114587, 2022.

[55] Z. Li, H. Zheng, N. Kovachki, D. Jin, H. Chen, B. Liu, K. Azizzadenesheli, and A. Anandkumar. Physics-informed neural operator for learning partial diferential equations. ACM/IMS Journal of Data Science, 1(3):1–27, 2024.

[56] N. Plung˙e, P. Brommer, R. S. Edwards, and E. G. Kakouris. Deep learning-based phase-field modelling of brittle fracture in anisotropic media. International Journal for Numerical Methods in Engineering, 127(13):e70381, 2026.

[57] A. Dean and B. Bahtiri. A fracture-informed multiphysics self-sensing framework for phase-field fracture in piezoresistive materials using the deep energy method. Theoretical and Applied Fracture Mechanics, 148:105890, 2026.

[58] N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. A. Hamprecht, Y. Bengio, and A. Courville. On the spectral bias of neural networks. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 5301–5310, 2019. https://arxiv.org/abs/1806.08734.

[59] S. Wang, X. Yu, and P. Perdikaris. When and why PINNs fail to train: A neural tangent kernel perspective. Journal of Computational Physics, 449:110768, 2022.

[60] S. Wang, Y. Teng, and P. Perdikaris. Understanding and mitigating gradient flow pathologies in physics-informed neural networks. SIAM Journal on Scientific Computing, 43(5):A3055–A3081, 2021.

[61] M. Tancik, P. P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. T. Barron, and R. Ng. Fourier features let networks learn high frequency functions in low dimensional domains. In Advances in Neural Information Processing Systems, volume 33, pages 7537–7547, 2020. https://arxiv.org/abs/2006.10739.

[62] V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G. Wetzstein. Implicit neural representations with periodic activation functions. In Advances in Neural Information Processing Systems, volume 33, pages 7462–7473, 2020. https://arxiv.org/abs/2006.09661.

[63] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljačić, T. Y. Hou, and M. Tegmark. KAN: Kolmogorov–Arnold Networks, 2024. https://arxiv.org/abs/2404.19756.

[64] A. D. Jagtap and G. E. Karniadakis. Extended physics-informed neural networks (XPINNs): A generalized space-time domain decomposition based deep learning framework for nonlinear partial diferential equations. Communications in Computational Physics, 28(5):2002–2041, 2020.

[65] A. Lotfalian, M. R. Banan, and P. Broumand. eXtended deep energy physics-informed neural networks method for fracture mechanics problems. Engineering Fracture Mechanics, 345:112428, 2026.

[66] L. Zhao and Q. Shao. DENNs: Discontinuity-embedded neural networks for fracture mechanics. Computer Methods in Applied Mechanics and Engineering, 446:118184, 2025.

[67] L. Zhao and Q. Shao. DEDEM: Discontinuity embedded deep energy method for solving fracture mechanics problems, 2024. https://arxiv.org/abs/2407.11346.

[68] Y. Wang, Y. Lin, S. Goswami, L. Zhao, H. Zhang, J. Bai, C. Anitescu, M. S. Eshaghi, X. Zhuang, T. Rabczuk, and Y. Liu. Towards unified AI-driven fracture mechanics: The extended deep energy method (XDEM). Nature Communications, 17:8492, 2026.

[69] M. Raj, P. Kumbhar, and R. K. Annabattula. Physics-informed neural networks for solving thermo-mechanics problems of functionally graded material, 2021. https://arxiv.org/abs/2111. 10751.

[70] X.-L. Yu and X.-P. Zhou. A nonlocal energy-informed neural network for isotropic elastic solids with cracks under thermomechanical loads. International Journal for Numerical Methods in Engineering, 124(18):3935–3963, 2023.

[71] S. K. Yadav, V. Yadav, and R. Patil. Fracture analysis of functionally graded plates under thermomechanical loading using mesh enriched physics-informed neural networks (M-PINNs). Research Square preprint, 2026. https://doi.org/10.21203/rs.3.rs-9295110/v1.

[72] C. F. Shih, B. Moran, and T. Nakamura. Energy release rate along a three-dimensional crack front in a thermally stressed body. International Journal of Fracture, 30(2):79–102, 1986.

[73] W. K. Wilson and I.-W. Yu. The use of the J-integral in thermal stress crack problems. International Journal of Fracture, 15(4):377–387, 1979.

[74] D. Ghafari, S. Rash Ahmadi, and F. Shabani. XFEM simulation of a quenched cracked glass plate with moving convective boundaries. Comptes Rendus Mécanique, 344(2):78–94, 2016.

[75] B. Cotterell and J. R. Rice. Slightly curved or kinked cracks. International Journal of Fracture, 16(2):155–169, 1980.

[76] R. J. Nuismer. An energy release rate criterion for mixed mode fracture. International Journal of Fracture, 11(2):245–250, 1975.

[77] H. Wang, S. Tanaka, S. Oterkus, and E. Oterkus. Evaluation of stress intensity factors under thermal efect employing domain integral method and ordinary state based peridynamic theory. Continuum Mechanics and Thermodynamics, 35:1021–1040, 2023.

[78] T. J. R. Hughes. The Finite Element Method: Linear Static and Dynamic Finite Element Analysis. Dover Publications, Mineola, NY, 2000. Chapter 8, algorithms for parabolic problems.

[79] R. E. Caflisch. Monte Carlo and quasi-Monte Carlo methods. Acta Numerica, 7:1–49, 1998.

[80] E. Béchet, H. Minnebo, N. Moës, and B. Burgardt. Improved implementation and robustness study of the X-FEM for stress analysis around cracks. International Journal for Numerical Methods in Engineering, 64(8):1033–1056, 2005.

[81] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization, 2014. https://arxiv.org/ abs/1412.6980.

[82] D. C. Liu and J. Nocedal. On the limited memory BFGS method for large scale optimization. Mathematical Programming, 45(1–3):503–528, 1989.

[83] H. Zhang, M. Makki Alamdari, B. Shahbodagh, M. Vahab, C. Anitescu, T. Rabczuk, and E. Atroshchenko. A mesh-free multiresolution deep energy method with phase-field modeling of brittle fracture, 2026. https://arxiv.org/abs/2608.24126.

[84] M. Amestoy and J. B. Leblond. Crack paths in plane situations—II. Detailed form of the expansion of the stress intensity factors. International Journal of Solids and Structures, 29(4):465–501, 1992.

[85] R. G. Tangella, P. Kumbhar, and R. K. Annabattula. Thermo-elastic fracture using phase field method in FEniCS: source code for the single-edge notched specimen. Mechanics of Materials Group, Indian Institute of Technology Madras, 2022. https://home.iitm.ac.in/ratna/codes/ thermo-elastic-fracture-fenics/index.html.

[86] P. K. Kristensen, C. F. Niordson, and E. Martínez-Pañeda. An assessment of phase field fracture: crack initiation and growth. Philosophical Transactions of the Royal Society A, 379:20210021, 2021.

[87] N. N. V. Prasad, M. H. Aliabadi, and D. P. Rooke. Incremental crack growth in thermoelastic problems. International Journal of Fracture, 66:R45–R50, 1994.

[88] M. N. Nguyen, T. Q. Bui, N. T. Nguyen, T. T. Truong, and L. V. Lich. Simulation of dynamic and static thermoelastic fracture problems by extended nodal gradient finite elements. International Journal of Mechanical Sciences, 134:370–386, 2017.

[89] H. Wang. A meshfree variational multiscale methods for thermo-mechanical material failure. Theoretical and Applied Fracture Mechanics, 75:1–7, 2015.

[90] H. Chen, Q. Wang, G. R. Liu, Y. Wang, and J. Sun. Simulation of thermoelastic crack problems using singular edge-based smoothed finite element method. International Journal of Mechanical Sciences, 115–116:123–134, 2016.

[91] F. Greco, D. Ammendolea, P. Lonetti, and A. Pascuzzo. Crack propagation under thermomechanical loadings based on moving mesh strategy. Theoretical and Applied Fracture Mechanics, 114:103033, 2021.

[92] H. Chen, S. Natarajan, E. T. Ooi, and C. Song. Modeling of coupled thermo-mechanical crack propagation in brittle solids using adaptive phase field method with scaled boundary finite element method. Theoretical and Applied Fracture Mechanics, 129:104158, 2024.

[93] NCI Australia. Gadi supercomputer, 2019. https://doi.org/10.25914/608bfd1838db2.

[94] UNSW Sydney. Katana, 2010. https://doi.org/10.26190/669X-A286.