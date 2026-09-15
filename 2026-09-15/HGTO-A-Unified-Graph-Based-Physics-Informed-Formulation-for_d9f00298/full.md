# HGTO: A Unified Graph-Based Physics-Informed Formulation for Structural Topology Optimization

Kangzheng Liu<sup>a</sup>, Uday Kumar Punna<sup>a</sup>, Leixin Ma<sup>a,∗</sup>

<sup>a</sup>School for Engineering of Matter, Transport and Energy, Arizona State University, Tempe, 85287, AZ, USA

## Abstract

Density-based topology optimization is typically structured as a nested sequence of material updates, structural analyses, and sensitivity assessments. While neural density parameterization and dual-field physics-informed approaches provide data-free alternatives, most existing methods represent density and displacement as coordinate fields and make limited use of the discrete relationships inherent in the finite element mesh. The present study introduces HGTO, a unified graph-based formulation that extends complete neural topology optimization from coordinate space to finite-element graph space. Element densities are parameterized on the element graph derived from the mesh, and the structural state is determined on the corresponding node–element hypergraph. Finite element kinematics, numerical quadrature, constitutive response, and force assembly remain explicitly defined operations within the diferentiable computation. The material field and equilibrium state are therefore coupled through a common finite-element incidence structure. Numerical studies show compliance comparable to conventional density-based optimization at substantially lower computational cost than a representative coordinate-based dual-field neural method. The same coupled formulation accommodates high-resolution and irregular meshes, three-dimensional structures, finite deformation, and elastoplastic response.

Keywords: Topology optimization, Graph neural network, Hypergraph neural network, Physics-informed learning, Finite element method, Neural parameterization

## 1. Introduction

Topology optimization (TO) seeks the most efective distribution of material within a prescribed design domain under given loads, boundary conditions, and design constraints. Since the pioneering work of Bendsøe and Kikuchi [1] and the subsequent development of the Solid Isotropic Material with Penalization (SIMP) method [2, 3], density-based topology optimization has become one of the most established approaches for structural design. In a conventional SIMP formulation, the design domain is discretized into finite elements, an element-wise density field describes the material distribution, and finite element analysis is repeatedly performed to determine the structural response. The corresponding sensitivities are then used to update the design through the optimality criteria (OC) method, the Method of Moving Asymptotes (MMA), or related gradient-based algorithms [4]. Filtering and projection schemes are commonly introduced to suppress checkerboard artifacts, control feature sizes, and obtain nearly discrete material layouts [5–7]. Owing to decades of development in numerical algorithms and computational execution, SIMP continues to be a reliable and widely used method for both academic and engineering applications [8, 9].

Nevertheless, conventional topology optimization is still a nested iterative procedure. Every modification of the density field changes the structural stifness and therefore requires a new equilibrium analysis, followed by sensitivity evaluation and another design update. Repeated state solutions constitute a major part of the computational efort, particularly as the design resolution, number of loading cases, or physical complexity increases [10, 11]. In addition, design and mechanics are handled by diferent numerical procedures. The optimizer updates the density variables, a finite element solver evaluates the associated displacement field, and the resulting mechanical information is subsequently transferred back to the optimizer. Although this sequential organization is efective, it separates the representation of the material field from the calculation of the structural state, making it dificult to express the complete design problem within a single computational formulation.

The rapid development of deep learning has motivated extensive eforts to accelerate or reformulate topology optimization. A large class of methods uses data-driven neural networks to predict optimized material layouts directly from loads, boundary conditions, volume fractions, or partially converged designs. Convolutional neural networks, encoder– decoder models, generative adversarial networks, transfer-learning strategies, and difusion models have demonstrated that near-optimal topologies can be generated in milliseconds or seconds once training has been completed [12–15]. These methods are particularly attractive when many design problems are drawn from a fixed or closely related family of problems. Their performance, however, depends strongly on the availability and coverage of optimized training data. Generating the required datasets may itself require many conventional topologyoptimization simulations, while unseen loading conditions, support configurations, geometries, or resolutions may markedly reduce prediction quality. Moreover, similarity to a reference topology at the image or density-field level does not necessarily ensure structural equilibrium, an efective load path, satisfaction of local constraints, or manufacturability.

To avoid dependence on databases of optimized structures, another class of methods uses neural networks as design parameterizations rather than predictors. In these approaches, the density field is represented as

$$
\rho ( \mathbf { x } ) = f _ { \pmb { \theta } } ( \mathbf { x } ) ,\tag{1}
$$

where the network parameters θ replace conventional element-wise design variables and are optimized for each problem using the structural objective and design constraints. Hoyer et al., TOuNN, TONR, and subsequent implicit-neural-representation methods demonstrated that a neural density field can be optimized without labeled topology data [16–18]. Such representations provide shared design parameters, continuous evaluation of the density field, and control over its spatial frequency content. They can therefore be viewed as physicsdriven neural reparameterizations of the design variables. Their optimization behavior, however, remains strongly influenced by the chosen network architecture and optimization scheme [19]. More importantly, these approaches alter only the design representation: the structural response is still obtained from a conventional finite element analysis, and the mechanics calculation remains external to the neural parameterization. Most neural density representations are also coordinate-based multilayer perceptrons and therefore infer spatial relationships from point coordinates without explicitly using the connectivity of the underlying finite element mesh.

Physics-informed neural networks have enabled a further transition from neural density parameterization towards complete neural topology optimization. PINNTO replaced the finite element state analysis with an energy-based neural solver rooted in the deep-energy formulation [20, 21]. NTopo represented both the density and displacement fields using implicit neural networks, while retaining sensitivity filtering and an OC-generated target density to stabilize the design update [22]. CPINNTO combined a deep-energy displacement

PINN with a neural density model and formulated the complete topology-optimization process without using labeled data or conventional finite element analysis [23]. DMF-TONN further demonstrated that the density and displacement networks could be connected more directly, allowing the density representation to be updated without sensitivity filtering, OC updates, or repeated fitting to target densities [24]. More recently, DPNN-TO adopted a variational dualnetwork formulation with a sinusoidal displacement network and a Fourier-enhanced density network, extending the approach to high-resolution two- and three-dimensional problems, multiple loads, and multiple design constraints [25]. These developments established an important new interpretation of topology optimization: the material field and the structural state can be treated as coupled unknowns and determined within the same physics-driven optimization process.

The principal dificulty in these complete and dual-field formulations lies in approximating the structural state. Density and displacement are generally represented by coordinate-based neural networks and coupled via an energy formulation, a residual loss, or a design objective. Whenever the density field changes, the displacement network must be retrained or further optimized to recover the corresponding equilibrium solution. The accuracy of the topology update, therefore, depends directly on the quality of the neural state approximation. Errors in displacement, strain, stress, or numerical integration carry over to the compliance and design gradient. This dependence becomes more pronounced near solid–void interfaces, stress concentrations, complex boundaries, low-volume-fraction designs, and nonlinear constitutive regimes. Present methods consequently rely on problem-dependent choices of sampling points, network size, frequency range, learning rate, loss weights, and the number of displacementtraining iterations. Inaccurate state solutions may also produce asymmetric or mechanically inconsistent designs in otherwise symmetric problems [23–25].

A more essential limitation is that coordinate-based dual-field methods make limited use of the discrete mechanical structure already available in the computational mesh. The displacement network receives spatial coordinates and must recover displacement gradients, local material response, and equilibrium from sampled points. Element adjacency, node– element incidence, numerical quadrature, constitutive variables, and the assembly of element forces are not inherent parts of the neural representation. Existing complete PINN methods, therefore, integrate design and mechanics at the level of the optimization objective but not at the level of the underlying finite element structure. Their mesh-free representation provides flexibility in spatial sampling but also discards connectivity information, which is particularly useful for irregular domains, unstructured meshes, and element-based constitutive calculations.

Graph neural networks (GNNs) provide a natural means of retaining this information. A finite element mesh is inherently relational: neighboring elements define the local neighborhood of the material field, nodes carry mechanical degrees of freedom, and each finite element establishes a multi-node mechanical relation. Recent GNN-based topology-optimization methods have exploited this structure for near-optimal topology prediction over irregular domains, convergence speedup on unstructured meshes, as well as graph-based density regularization [26–28]. More recent graph neural-field methods directly parameterize element densities on the finite-element mesh and optimize the GNN using structural objectives derived from diferentiable finite-element analysis [29, 30]. These studies show that mesh connectivity provides an efective basis for parameterizing neural density. Nevertheless, the graph is used primarily to represent the material field, whereas the equilibrium problem is still solved by a separate matrix-based finite element procedure.

In parallel, graph-based computational mechanics has increasingly incorporated the discrete structure of numerical mechanics. Physics-informed graph neural Galerkin networks introduced graph-based variational formulations for irregular shapes and unstructured meshes [31]. Graphconvolutional deep-energy methods additionally demonstrated that finite-element shapefunction gradients can improve robustness relative to purely coordinate-based automatic diferentiation in demanding deformation problems [32]. Finite-element-inspired hypergraph networks then introduced an explicit node–element representation to preserve the higher-order relations of finite element meshes [33]. Most recently, the FEM-Informed Hypergraph Neural Network (FHGNN) formulated isoparametric mapping, shape-function gradients, Gauss-point strain and stress evaluation, constitutive updates, numerical integration, and internal-force assembly as prescribed node-to-element and element-to-node operations [34]. Unlike a datadriven surrogate, these operations are determined by the finite element formulation rather than learned from simulation data. FHGNN therefore provides a diferentiable finite-elementconsistent representation of structural mechanics and has been applied to three-dimensional and path-dependent elastoplastic problems. Its current formulation, however, addresses forward structural analysis with a prescribed material distribution.

The graph representations used for material design and finite element mechanics are closely related. The element graph used to propagate density information is the dual representation induced by the node–element incidence of the same finite element mesh. The same incidence relation also transfers nodal displacements to an element, evaluates the corresponding element response, and assembles the element forces back to the nodes. Thus, the graph used for density parameterization and the hypergraph used for structural mechanics are complementary views of the same finite element discretization.

Taken together, these developments point to a more fundamental opportunity for physicsinformed topology optimization: replacing the coordinate-based coupling of design and state with a mesh-native formulation derived directly from the finite element discretization. In existing dual-field approaches, density and displacement are coupled through the optimization objective, but the geometric and mechanical relations carried by the computational mesh remain largely outside the neural representation. Graph-based formulations, on the other hand, have shown that these relations can be retained explicitly and used as the computational structure for both material representation and structural mechanics. This suggests that the finite element mesh itself can serve as a common basis for describing material evolution and equilibrium, allowing the two fields to interact through the same discrete mechanical structure rather than through independently parameterized coordinate spaces. To the best of the authors’ knowledge, such a mesh-native formulation of complete physics-informed topology optimization has not yet been established.

To address this gap, we introduce HGTO, a unified graph-based physics-informed formulation for structural topology optimization. HGTO advances complete neural topology optimization by transitioning from coordinate-based field representations to finite-element graphs. The density field is parameterized on the element graph derived from the mesh, and the structural state is evaluated on the corresponding node–element hypergraph using prescribed finite-element operations. Since both representations are based on the same finiteelement incidence structure, material evolution and mechanical equilibrium are integrated within a single mesh-connected diferentiable formulation. Consequently, HGTO maintains the data-free optimization advantages of neural design parameterization while explicitly preserving mesh connectivity and the local mechanics inherent to the finite element discretization.

The main contributions of this work are summarized as follows:

1. A complete graph-based formulation of physics-informed topology optimization is introduced, extending the coupled density–displacement paradigm from coordinate neural fields to finite element graph representations derived from a common mesh incidence structure.

2. Finite element mechanics is retained explicitly within the diferentiable graph computation. Material interpolation, finite-element kinematics, numerical quadrature, constitutive evaluation, and force assembly are prescribed operations within the same optimization process used to update the neural density representation.

3. The common mesh-based representation preserves element connectivity and node– element relations throughout the optimization, providing a consistent basis for irregular design domains, unstructured discretizations, and diferent constitutive descriptions without changing the overall formulation.

The proposed framework is evaluated using a series of structural topology-optimization problems, with comparisons to conventional density-based topology optimization and the coordinate-based dual-field neural approach. The numerical studies examine the accuracy of mechanical solutions, the quality and convergence of optimized topologies, and the computational characteristics of the coupled design and analysis process.

The remainder of this paper is organized as follows. Section 2 presents the HGTO formulation. Sections 3 and 4 examine linear and nonlinear design problems, respectively. Section 5 discusses the findings and concludes with the limitations and future directions of the framework.

## 2. HGTO: coupled topology and physics fields

HGTO describes material distribution and structural equilibrium on two graph fields derived from the same finite-element mesh. The topology-field graph PINN generates element densities through learnable message passing. The physics-field hypergraph PINN uses these densities in its constitutive response and updates its nodal states to satisfy equilibrium. Element sensitivities then connect the mechanical response to the parameters of the topology field.

## 2.1. Problem formulation

Consider a design domain $\Omega \subset \mathbb { R } ^ { d }$ , with $d = 2$ or 3, discretized into $N _ { e }$ elements and $N _ { n }$ nodes. For a prescribed force vector f and homogeneous displacement supports, the minimum-compliance problem is

$$
\begin{array} { r l } { \underset { \rho } { \mathrm { m i n } } } & { C ( \rho ) = f ^ { \top } \boldsymbol { u } ^ { * } , } \\ { \mathrm { s u b j e c t ~ t o } } & { K ( \rho ) \boldsymbol { u } ^ { * } = \boldsymbol { f } , } \\ & { \displaystyle \sum _ { e = 1 } ^ { N _ { e } } V _ { e } \rho _ { e } \leq V _ { f } V _ { \Omega } , \qquad \rho _ { \mathrm { m i n } } \leq \rho _ { e } \leq 1 , } \end{array}\tag{2}
$$

where $V _ { e }$ is the volume of element e, $\begin{array} { r } { V _ { \Omega } = \sum _ { e } V _ { e } } \end{array}$ , and $V _ { f }$ is the prescribed material fraction. Displacements and equilibrium equations are expressed on the free degrees of freedom. The SIMP interpolation defines the element modulus and global stifness as

$$
E _ { e } ( \rho _ { e } ) = E _ { \mathrm { m i n } } + ( E _ { 0 } - E _ { \mathrm { m i n } } ) \rho _ { e } ^ { p } , \qquad K ( \rho ) = \sum _ { e } P _ { e } ^ { \mathsf { T } } \big [ E _ { e } ( \rho _ { e } ) \widehat { k } _ { e } \big ] P _ { e } .\tag{3}
$$

Here $E _ { 0 }$ and $E _ { \mathrm { m i n } } > 0$ are the solid modulus and stifness floor, $p$ is the penalization exponent, $\widehat { \pmb { k } } _ { e }$ is the element stifness at unit modulus, and $P _ { e }$ gathers the local displacement $\pmb { u } _ { e } = \pmb { P } _ { e } \pmb { u }$ The available material is fully used in these linear compliance problems. HGTO therefore imposes the volume equality through the density parameterization introduced below. Section 4 extends the state equation to finite-deformation and elastoplastic response.

![](images/fd79d98ca52bfaed3a3939629aefcab889237883a2a7b164aa454c248759c3f4.jpg)  
Figure 1: Graph representations of the same finite-element mesh. An element is a node in the topology graph $\mathcal { G } _ { \rho }$ and a hyperedge in the physics hypergraph $\mathcal { G } _ { u }$

## 2.2. Dual-field representation of the mesh

The incidence matrix $\pmb { H } \in \{ 0 , 1 \} ^ { N _ { n } \times N _ { e } }$ records the membership of mesh nodes in finite elements: $H _ { i e } = 1$ when node i belongs to element e. Together with the ordered local connectivity, it defines the complementary graph views in Fig. 1. In the topology graph $\mathcal { G } _ { \rho } = ( \mathcal { V } _ { e } , \mathcal { E } _ { e } )$ , each element is a graph node. Adjacent elements share a complete edge in two dimensions or a complete face in three dimensions. Their centroids provide spatial features, and their connections determine the neighborhoods used to update these features.

The physics hypergraph $\mathcal { G } _ { u } ~ = ~ ( \mathcal { V } _ { n } , \mathcal { E } _ { h } , H )$ associates displacement states with mesh nodes and represents each element as a hyperedge joining its constituent nodes. Density and constitutive parameters are stored with the element, together with its geometry and quadrature data. This representation retains the multi-node interactions of finite-element mechanics. An element density generated on $\mathcal { G } _ { \rho }$ is available at the corresponding hyperedge of $\mathcal { G } _ { u }$ , and its mechanical sensitivity returns through the same index.

Figure 2 presents the optimization loop. The topology field generates the current physical density, the physics field determines the associated equilibrium state, and the design gradient updates the topology-network parameters. Connectivity and reference geometry remain fixed as the material field and mechanical states evolve on the common discretization.

## 2.3. Topology-field graph PINN

## 2.3.1. Graph parameterization

The topology network assigns one scalar output to each element. Normalized centroids $\pmb { c } _ { e } \in [ 0 , 1 ] ^ { d }$ are embedded using the Fourier features

$$
\pmb { h } _ { e } ^ { ( 0 ) } = \gamma ( \pmb { c } _ { e } ) = \left[ \frac { \sin ( 2 \pi \pmb { F } \pmb { c } _ { e } ) } { \cos ( 2 \pi \pmb { F } \pmb { c } _ { e } ) } \right] , \qquad \pmb { F } _ { a b } \sim \mathcal { N } ( 0 , \sigma _ { F } ^ { 2 } ) ,\tag{4}
$$

(a) Problem setup (b) Coupled dual-field Graph PINN  
(c) Topology evolution  
![](images/b48dc463e07a8aa60fde98fb0b45fbad8461981632263dc576de6a6763272060.jpg)  
Figure 2: HGTO workflow. The topology graph generates physical densities, and the physics hypergraph evaluates the structural response. Element sensitivities are propagated back to update the topology network. The right-hand panels show the evolving design.

where $\pmb { F } \in \mathbb { R } ^ { n _ { F } \times d }$ is sampled at initialization and then held fixed. Two first-degree Chebyshev graph convolutions [35] propagate the features over adjacent elements:

$$
\begin{array} { r l r } & { { \pmb h } ^ { ( 1 ) } = \mathrm { R e L U } \left( { \pmb h } ^ { ( 0 ) } { \pmb W } _ { 0 } ^ { ( 0 ) } + \tilde { \pmb { L } } { \pmb h } ^ { ( 0 ) } { \pmb W } _ { 1 } ^ { ( 0 ) } + { \pmb b } ^ { ( 0 ) } \right) , } & \\ & { { \pmb z } = { \pmb h } ^ { ( 1 ) } { \pmb W } _ { 0 } ^ { ( 1 ) } + \tilde { \pmb { L } } { \pmb h } ^ { ( 1 ) } { \pmb W } _ { 1 } ^ { ( 1 ) } + { \pmb b } ^ { ( 1 ) } . } & \end{array}\tag{5}
$$

The normalized graph Laplacian L is rescaled as $\tilde { \pmb { L } } = 2 \pmb { L } / \lambda _ { \mathrm { m a x } } - \pmb { I }$ , using the spectral bound $\lambda _ { \operatorname* { m a x } } = 2$ . Each layer combines the features of an element with those of its neighbors, and the second layer produces the scalar logits $z _ { e } .$ . The trainable parameters θ comprise the shared weights and biases. The parameter count is independent of mesh resolution: for hidden width $h ,$ the two layers contain $N _ { \theta } = 4 n _ { F } h + 3 h + 1$ trainable parameters. The fixed Fourier features and their first-layer graph propagation can be cached for a given mesh.

## 2.3.2. Density filtering and volume constraint

The network outputs are converted to physical densities by spatial averaging and smooth projection. The filter averages nearby design values within a radius $r _ { f }$ , with weights determined by distance and element volume. For the logits $z ,$ the density map is

$$
\begin{array} { r l } & { \pmb { x } = \mathrm { s i g m o i d } ( z - \lambda _ { V } \mathbf { 1 } ) , } \\ & { \pmb { \rho } = \rho _ { \mathrm { m i n } } \mathbf { 1 } + ( 1 - \rho _ { \mathrm { m i n } } ) \mathcal { H } _ { \beta } ( \pmb { A } \pmb { x } ) , } \end{array}\tag{6}
$$

where A is the normalized filter matrix and $\mathcal { H } _ { \beta }$ is a smooth Heaviside projection. Increasing $\beta$ sharpens the solid–void boundary. The common shift $\lambda _ { V }$ is chosen to satisfy the material constraint after filtering and projection:

$$
\sum _ { e } V _ { e } \rho _ { e } ( z , \lambda _ { V } ) = V _ { f } V _ { \Omega } .\tag{7}
$$

Gradients are propagated through the filter, projection, and volume constraint when updating the topology network. The filter weights, projection function, and volume-constrained derivative are given in Section S1 of the supplementary material.

## 2.4. Hypergraph physics-field PINN

The physics field stores displacement as nodal attributes. For a fixed density field, a forward pass gathers the nodal attributes at each hyperedge, evaluates the element response, and returns force contributions to the nodes:

$$
{ \pmb u } _ { e } = { \pmb P } _ { e } { \pmb u } , \qquad { \pmb m } _ { e } ^ { u } = \Phi _ { e } \bigl ( { \pmb u } _ { e } , \rho _ { e } \bigr ) , \qquad { \pmb f } ^ { \mathrm { i n t } } = \sum _ { e } { \pmb P } _ { e } ^ { \top } { \pmb m } _ { e } ^ { u } .\tag{8}
$$

The physical message map $\Phi _ { e }$ contains the finite-element kinematics, constitutive response, and numerical integration. For small-strain elasticity,

$$
\varepsilon _ { e g } = B _ { e g } u _ { e } , \qquad \sigma _ { e g } = D _ { e } ( \rho _ { e } ) \varepsilon _ { e g } , \qquad m _ { e } ^ { u } = \sum _ { g } { \cal B } _ { e g } ^ { \top } \sigma _ { e g } \omega _ { e g } ,\tag{9}
$$

where $B _ { e g }$ is the strain–displacement matrix at quadrature point g and $\omega _ { e g }$ includes the quadrature weight and Jacobian determinant. The resulting message is $\pmb { m } _ { e } ^ { u } = E _ { e } \hat { \pmb { k } } _ { e } \pmb { u } _ { e } ,$ , and its associated strain energy is $\mathcal { W } _ { e } = \frac { 1 } { 2 } \pmb { u } _ { e } ^ { \top } \pmb { m } _ { e } ^ { u }$

Structural equilibrium is obtained by minimizing the total potential energy,

$$
\begin{array} { r l } { \mathcal { L } _ { u } ( u ; \boldsymbol { \rho } ) = \displaystyle \sum _ { e } \mathcal { W } _ { e } - \boldsymbol { f } ^ { \top } \boldsymbol { u } , } & { { } } \\ { \nabla _ { u } \mathcal { L } _ { u } = \boldsymbol { f } ^ { \mathrm { i n t } } - \boldsymbol { f } = R , \quad } & { { } u ^ { * } ( \boldsymbol { \rho } ) = \arg \displaystyle \operatorname* { m i n } _ { u } \mathcal { L } _ { u } . } \end{array}\tag{10}
$$

The free nodal displacements are the optimization variables. The message operators are prescribed by the element formulation, and displacement supports are imposed directly. Element geometry, quadrature data, and the incidence maps are cached before optimization.

For linear elasticity, suficient displacement constraints and a positive stifness floor make $\mathcal { L } _ { u }$ a strictly convex quadratic. Preconditioned conjugate gradients minimize this loss using node–element–node operator actions, with geometric or algebraic multigrid adapted to the mesh structure. For the nonlinear cases in Section 4, equilibrium is followed incrementally using the finite-deformation and elastoplastic operators derived in Sections S3 and S4 of the supplementary material.

## 2.5. Coupling and optimization

The topology field is trained using the equilibrium compliance,

$$
\mathcal { L } _ { \rho } ( \pmb { \theta } ) = \frac { \pmb { f } ^ { \top } \pmb { u } ^ { * } ( \pmb { \rho } ( \pmb { \theta } ) ) } { C _ { \mathrm { r e f } } } ,\tag{11}
$$

where $C _ { \mathrm { r e f } }$ is the initial compliance, held fixed during optimization. The volume constraint is incorporated through Eq. (7). For the fixed-load linear problem, diferentiating equilibrium gives the element sensitivity and the network gradient:

$$
\begin{array} { c } { { \displaystyle g _ { \rho , e } = \frac { d C } { d \rho _ { e } } = - p \rho _ { e } ^ { p - 1 } ( E _ { 0 } - E _ { \mathrm { m i n } } ) ( \boldsymbol { u } _ { e } ^ { * } ) ^ { \top } \hat { \boldsymbol k } _ { e } \boldsymbol { u } _ { e } ^ { * } , } } \\ { { \nabla _ { \theta } \mathcal { L } _ { \rho } = \displaystyle \frac { 1 } { C _ { \mathrm { r e f } } } \sum _ { e } g _ { \rho , e } \nabla _ { \theta } \rho _ { e } . } } \end{array}\tag{12}
$$

Each sensitivity is evaluated at its corresponding physics hyperedge and passed to the aligned element output of the topology graph. Automatic diferentiation propagates these sensitivities through the constrained density map and graph layers.

Continuation in the material penalization and projection sharpness guides the density field from a difuse initial distribution toward the final design. Adam updates the network parameters using the equilibrium objective and its gradient through a sequence of prescribed $( p , \beta )$ stages. Once the terminal values are reached, optimization continues until both the objective and physical density stabilize. The nonlinear examples use the same density–state coupling with incremental mechanical feedback and the schedules specified in Section 4.

Algorithm 1 summarizes the procedure. Convergence is assessed on volume-feasible, equilibrated designs at the final material parameters. A maximum update count limits the computation. The final design is evaluated with the terminal material parameters and state tolerance.

Algorithm 1 HGTO optimization   
Require: Mesh, supports, loading, material model, $V _ { f }$ , and optimization schedule   
Ensure: Physical density $\rho ^ { * }$ and equilibrium state   
1: Construct $\mathcal { G } _ { \rho } , \mathcal { G } _ { u }$ , and filter $A ;$ cache element data   
2: Initialize the topology parameters θ and mechanical state   
3: for each design update within the prescribed maximum do   
4: Select $p$ and $\beta$ from the continuation schedule   
5: Evaluate logits z and solve Eq. (7) for the physical density $\rho$   
6: Solve equilibrium, including the loading history when required   
7: Evaluate the objective and physical-density sensitivities $g _ { \rho }$   
8: if this is the initial evaluation then   
9: Set $C _ { \mathrm { r e f } }  C$   
10: end if   
11: Backpropagate $g _ { \rho } / C _ { \mathrm { r e f } }$ through the density map and graph network   
12: Update θ with Adam using the objective gradient   
13: if the final-stage objective and density meet the stopping tolerances then   
14: break   
15: end if   
16: end for   
17: Evaluate the final density at the prescribed terminal material parameters and state tolerance   
18: return $\rho ^ { * }$ and its equilibrium state

## 3. Results

The numerical studies examine the design quality, computational cost, and applicability of HGTO in comparison with conventional density-based optimization and coordinate-based dual-field neural TO. We begin with planar benchmarks to assess the quality and cost of the optimized designs, then examine finer discretizations, irregular domains, and three-dimensional structures. Finite-deformation and elastoplastic problems are considered in Section 4.

## 3.1. Experimental settings

SIMP–OC serves as the conventional reference, and NTopo [22] represents coordinatebased dual-field neural TO. NTopo follows the original formulation and training settings, with case-specific settings summarized in the supplementary material.

The design domain, loading, supports, and prescribed material fraction are held fixed within each numerical comparison. Final linear designs are evaluated with a common finiteelement model implemented in scikit-fem [36]. The linear material parameters are $E _ { 0 } = 1$ $E _ { \mathrm { m i n } } = 1 0 ^ { - 6 }$ , and $\nu = 0 . 3$ . HGTO and SIMP–OC use the same density filter and projection.

The neural computations run on an NVIDIA RTX 6000 Ada GPU. SIMP–OC uses two CPU threads on an AMD Ryzen Threadripper PRO 5995WX. Network configurations, stopping criteria, achieved volumes, and solver settings are given in the supplementary material. Each neural design is optimized for its prescribed problem without labeled topology data.

## 3.2. Two-dimensional benchmarks

The first examples are a centrally loaded cantilever, a half-MBB beam, and a cantilever subjected to an inclined eccentric load. Each uses a 120 × 40 mesh and a material fraction of 0.50. Figure 3 shows that HGTO recovers the principal members connecting the loads and supports, with compliance comparable to the conventional and dual-field neural references (Table 1).

The designs difer more in the subdivision of their interiors than in their stifness. For the half-MBB beam, HGTO places material in a few broad diagonal members and leaves larger open regions. The two reference methods introduce additional diagonals and smaller internal cells. The inclined-load HGTO design similarly contains fewer slender secondary members, particularly toward the loaded end. These arrangements retain the main loadbearing connections while concentrating material in fewer members. The centrally loaded cantilever shows closer agreement among all three methods, including the upper and lower chords and their diagonal connections.

The cost of obtaining these designs difers substantially. HGTO completes the three optimizations in 7.4–8.5 s, whereas NTopo requires 862–943 s. The graph-based formulation thus retains a learned material representation at a total cost of the same order as SIMP–OC on these small problems. The shared mesh representation connects the learned density field directly to the nodal equilibrium states used in the design update.

![](images/fb7709dadd29ec5658c1abd25b604643e6727db25d56ecdff49533a07d5d5c76.jpg)  
Figure 3: Optimized topologies for (a) a centrally loaded cantilever, (b) a half-MBB beam, and (c) a cantilever under inclined loading. The prescribed volume fraction is 0.50.

## 3.3. High-resolution optimization

HGTO and SIMP–OC optimize the cantilever on meshes ranging from 120×40 to 960×320 elements. HGTO uses the same network configuration and 16,577 trainable parameters at every resolution. Refinement increases the number of density outputs and mechanical states, while the graph layers continue to share their weights over the mesh.

The fine-resolution designs in Fig. 4 have similar compliance but diferent internal layouts. SIMP–OC develops a network of thin branches near the loaded end. HGTO retains fewer principal diagonals, with well-defined boundaries and junctions. Increasing the resolution resolves these members more closely without requiring a more heavily subdivided interior. The compact parameterization thus accommodates a detailed density field while preserving the broad organization of the material.

Table 1: Compliance C and total optimization time t (s) for the linear examples. † Update limit reached.
<table><tr><td rowspan="2">Case</td><td colspan="2">SIMP-OC</td><td colspan="2">NTopo</td><td colspan="2">HGTO</td></tr><tr><td>C</td><td>t</td><td>C</td><td>t</td><td>C</td><td>t</td></tr><tr><td>Cantilever</td><td>175.08</td><td>9.0</td><td>177.93</td><td>862.2</td><td>176.49</td><td>8.3</td></tr><tr><td>Half-MBB</td><td>191.49</td><td>9.0</td><td>194.79</td><td>942.5</td><td>195.47</td><td>7.4</td></tr><tr><td>Inclined load</td><td>132.72</td><td>26.9†</td><td>133.81</td><td>906.6</td><td>133.48</td><td>8.5</td></tr><tr><td>L-bracket</td><td>96.06</td><td>9.7</td><td>99.43</td><td>952.3</td><td>96.95</td><td>10.3</td></tr><tr><td>Perforated bracket</td><td>49.32</td><td>6.3</td><td>49.78</td><td>1001.1</td><td>48.71</td><td>14.2</td></tr><tr><td>3D cantilever</td><td>15.07</td><td>188.3</td><td>16.17</td><td>3886.0</td><td>15.23</td><td>24.3</td></tr><tr><td>Four-foot support</td><td>0.5374</td><td>377.2</td><td>0.5810</td><td>3781.3</td><td>0.5347</td><td>39.3</td></tr><tr><td>Torsion  $( C \times 1 0 ^ { 3 } )$ </td><td>8.135</td><td>337.0</td><td>10.148</td><td>3564.1</td><td>8.144</td><td>48.1</td></tr></table>

The runtime separation grows with problem size. HGTO takes 195.7 s on the finest mesh, containing 307,200 density elements, compared with 1,880.1 s for the CPU SIMP–OC implementation. The speedup is approximately fourfold at $4 8 0 \times 1 6 0$ and nearly tenfold at $9 6 0 \times 3 2 0$

![](images/f38e62881c55bf6fe793756de8bfdd3a561898d8a1095ff7e3dc42c70611fcab.jpg)

![](images/ef5a17efce99773210705a932bcb7e3c2fec43bd117e05ac74a497254c71240f.jpg)  
Figure 4: Cantilever designs on the (a) 480×160 and (b) $9 6 0 \times 3 2 0$ meshes, and (c) total computation time.

## 3.4. Irregular design domains

An L-bracket and a perforated bracket are considered at a material fraction of 0.40. The former introduces a re-entrant boundary, while the latter has an internal circular clearance and a nonuniform mesh. The topology-network architecture is retained, with its neighborhoods constructed from the element connections in each domain.

HGTO and SIMP–OC produce closely matched compliance in both cases. In the L-bracket, the vertical members join a fan of inclined members around the re-entrant region (Fig. 5a). This arrangement is similar across the three methods. In the perforated bracket, material separates into upper and lower paths around the clearance and rejoins near the load. HGTO leaves a larger open region between the hole and the loaded end, whereas the reference layouts introduce smaller internal branches there. The simpler subdivision preserves the overall stifness and the required clearance.

The change in geometry is handled through the mesh relations already used by the two fields. The topology graph follows the element neighborhoods around the opening, and the physics hypergraph retains the corresponding nodal connections. This correspondence also accommodates unequal element sizes in the perforated domain. HGTO obtains the two designs in 10–14 s, compared with approximately 16 min per case for the dual-field neural baseline. The common mesh representation accommodates these geometric changes at a cost of the same order as SIMP–OC.

Design domain (a) L-bracket

![](images/11ec2a2ab040cb750c106f626c68c492babea62e115961b8c62ee3cbf6f80293.jpg)  
SIMP–OC

![](images/e860bac482ab5868938aebdff159aa7572c77f024913186f7ec470cd8e801efa.jpg)  
NTopo

![](images/2a3e13e98ae6bb0e13c32b8b04e3bb5a5e24c9025b764c44da3d96ee60f76f5e.jpg)  
HGTO

![](images/626fe1bb3fc54dd01092dbd5df35ee9fa3add4e4ed052fd7fbb147834e650732.jpg)  
(b) Perforated bracket

![](images/cf1d32167a6ebcecb24893d12cd6721fb2f90ca8522408a80091e218f2bde4d7.jpg)

![](images/c9ca908d413d3753be3167cb64500228e32dd0725dec008f4a9b251595bf58d5.jpg)

![](images/928fa2829259244dd3d69129f3c61bba2a28d6033109e5204b84504438b9a9ec.jpg)

![](images/94ae53a54406b9ab071e30b09c6ca9f3a0da52664d1bcb972079702310c72b41.jpg)  
Figure 5: Optimized topologies for (a) the L-bracket and (b) the perforated bracket at a prescribed volume fraction of 0.40.

## 3.5. Three-dimensional structures

The three-dimensional examples comprise an end-loaded cantilever, a four-foot support, and a torsion member. The three cases use a common 3D network configuration and require diferent spatial arrangements of material. HGTO achieves compliance comparable to SIMP– OC throughout this set (Table 1).

Figure 6 shows how these arrangements change with the loading. The cantilever places material in upper and lower longitudinal members linked by webs. The four-foot support divides the load from the upper pad among four inclined legs. The torsion member places material around a hollow interior, forming the closed section visible in the midspan insets. The learned density representation accommodates both branching members and continuous walls, with the resulting forms closely following those obtained by SIMP–OC.

NTopo captures the broad exterior shapes, although its torsion design is about 25% more compliant than HGTO in the common evaluation. The section views show a more difuse material boundary for NTopo, while HGTO and SIMP–OC have more sharply defined walls. For this case, the distribution of material through the section distinguishes the designs more clearly than their similar exterior envelopes.

HGTO completes the 3D optimizations in 24–48 s. SIMP–OC takes 188–377 s, and NTopo approximately one hour. Relative to this dual-field neural baseline, HGTO reduces runtime by factors of 74–160. The eficiency of the coupled graph formulation therefore extends to volumetric density fields, where both the structural state and the material distribution contain substantially more variables.

Design domain

(a) Cantilever

![](images/02b231a85ad69286789f17893f775c7c2ae1a289d5eb894ba1cd2ef1eea45aba.jpg)

SIMP–OC

![](images/75e6b3bb98f0434ecd5b5591e3b62e27a225717c4104a4b2838d664cfaf41ffc.jpg)

NTopo

![](images/f4fff0939218a9680324a04c83fdb4ae5db55da145266751e023be273bc2d00e.jpg)  
HGTO

![](images/70e79c9aa082e81022d11708d1903c08bee234fc8ca093e9ac72c9fdb4c32f0e.jpg)  
(b) Four-foot support

![](images/900792a5f96f8f6f46a72b2d7800e0091aeb88dba5deac5ea38b600b89c34d6d.jpg)

![](images/7292bcfc7b93cf24a35cc3df4db8293eae0e7a476803754fa873af03967fdd4e.jpg)

![](images/192098c050a581f02c981c3e174e4e9fb32afc4fc7c3c555ebd8ce26d24014dc.jpg)

![](images/d6c5f23749e99d7343f2f03b31a637199e2fa72ee8af079862dd62d73d063a3b.jpg)  
(c) Torsion member

![](images/e5b6c093e43d4a1d0d20d768418cc8a795ecc2a5c6d1b49a7e1f0cbf4f38df17.jpg)  
Midspan sections

![](images/926b789e678e4602671f4dff57f129e2d0b068c9cb213d9f77700409d76f6831.jpg)

![](images/b64146bba2bba364c047bfead5a219957efb5b876aa4e77847a081582e6de608.jpg)

![](images/9ac70d096e36b2a0e5ffca4ae627e15e2e21517ad5b55af2eba64dde2fcbffaf.jpg)  
0

![](images/25314c07cde141f7f0a1430ea304d445419b1b348a54030e30b657222ce917e9.jpg)

![](images/1c248aa2b268a11450c4b24e14169879b0f7b3b122611c84441db8b5c3fe3efc.jpg)  
Figure 6: Three-dimensional topologies for (a) the cantilever, (b) the four-foot support, and (c) the torsion member. Surfaces are shown at $\rho = 0 . 5 ;$ the insets give midspan density sections of the torsion designs.

## 4. Nonlinear topology optimization

The following examples extend the coupled graph formulation to finite deformation and elastoplasticity. The material field is generated by the same topology-network construction, while the physical operators on the hypergraph account for the nonlinear response. The constitutive models, incremental state equations, and density sensitivities are derived in Sections S3 and S4 of the supplementary material.

## 4.1. Finite deformation under increasing load

A cantilever is optimized under a reference load $P _ { 0 }$ and a tenfold larger load $1 0 P _ { 0 } ,$ using a compressible Neo-Hookean model and a complementary-work objective. The NTopo baseline is adapted to the same nonlinear energy.

At $P _ { 0 } ,$ all three methods produce connected cantilever designs. HGTO and SIMP–OC form similar diagonally braced layouts and have closely matched objective values. The dual-field neural baseline finds a more subdivided arrangement with a lower objective value (Fig. 7a). HGTO takes 64.1 s, compared with approximately 860 s for either reference method (Table 2).

At $1 0 P _ { 0 } ,$ the displacement of the loaded port in the HGTO design reaches approximately 24% of the span. The deformation in Fig. 7d shows substantial rotation of the inclined members and the loaded end. The overall cantilever arrangement is retained, but the member inclinations and junctions change with the load level. Evaluating the evolving geometry allows these changes to enter the density update throughout loading.

HGTO converges in 130.5 s at the stronger load. SIMP–OC reaches a similar objective but stalls after 772.2 s, while the nonlinear NTopo adaptation fails to complete the optimization.

![](images/7ee1d88eb799b87f3bc2776318f13d84d4a9d2e5d578cfa3ad54355b4a768bf3.jpg)  
Figure 7: Cantilever optimization at (a) $P _ { 0 }$ and (b) $1 0 P _ { 0 }$ . Panels (c) and (d) show the corresponding HGTO deformations without amplification, with the undeformed designs in gray.

## 4.2. Nonlinear bridge

A doubly fixed bridge subjected to a distributed top load provides a second finitedeformation example. HGTO and SIMP–OC form two principal inclined members beneath the loaded region, with thinner branches extending toward the upper parts of the supports (Fig. 8). Their final objective values are nearly identical.

![](images/8fda710c8d89e146bcf5dbd0979577b07cf5d677d091476c82fdb044eaebeef3.jpg)

Table 2: Objective J and total optimization time t (s) for finite-deformation problems. NTopo uses the nonlinear adaptation. † Last accepted design before stagnation.
<table><tr><td></td><td colspan="2">SIMP-OC</td><td colspan="2">NTopo</td><td colspan="2">HGTO</td></tr><tr><td>Case</td><td>J</td><td>t</td><td>J</td><td>t</td><td>J</td><td>t</td></tr><tr><td>Cantilever:  $P _ { 0 }$ </td><td>0.000749</td><td>854.3</td><td>0.000660</td><td>859.7</td><td>0.000738</td><td>64.1</td></tr><tr><td>Cantilever:  $1 0 P _ { 0 }$ </td><td>0.07640</td><td>772.2†</td><td>Training failed</td><td></td><td>0.07360</td><td>130.5</td></tr><tr><td>Doubly fixed bridge</td><td>0.11064</td><td>418.8</td><td>Training failed</td><td></td><td>0.11086</td><td>62.0</td></tr></table>

The agreement also extends over the loading path. The load–displacement curves overlap closely as the central junction moves downward and the inclined members rotate. Thus, the similar final objectives are accompanied by similar structural responses during loading. HGTO reaches the design in 62.0 s, compared with 418.8 s for SIMP–OC. The nonlinear NTopo adaptation also fails to complete this case.

![](images/573ee9f9beab75def3a629959b31276e7d78e0711a1830b4faba412ee6bba5ce.jpg)

![](images/da99650a4ab7767e8ee01e0a0b2e585dd81d98a6806f10ed91e6a8ae4ca5e3e0.jpg)

![](images/ea00df20e0761230dc1c17ce82fc0369dcb024ebeeb3b50f4a13f0749a75d021.jpg)

![](images/48198ba5d980462cd511ef938538cb158c0b2ef982f1be00bbfff4746c7bff4d.jpg)  
Figure 8: Nonlinear bridge example: (a) design domain, $^ { ( \mathrm { b , c } ) }$ optimized topologies, (d) HGTO deformation at the final load, and (e) load–displacement curves. Deformation is shown without amplification.

## 4.3. Elastoplastic design

A perforated connection is optimized using either elastic or elastoplastic response. Both runs use the same material fraction, topology-network configuration, and graph construction, and minimize the peak-load work measure. The elastoplastic model stores and updates material history at the integration points of each hyperedge. The two final designs are then evaluated under a common elastoplastic loading–unloading cycle.

The main diagonal and lower member appear in both designs, with local diferences near the fixed boundary and the loaded-end junction (Fig. 9). These diferences have a pronounced efect on the response. The elastic-design curve is initially slightly stifer but bends toward larger displacements as plastic deformation accumulates. The elastoplastic design reaches the peak load with a smaller displacement, reducing the peak-load work by 27.5%. The ranking based on initial stifness therefore changes during loading.

(a) Design domain  
![](images/7cd5fc3231fccea121eae8b0f15b8c27ba111b803ce3c91b198f8338e2835ff1.jpg)

(b) Elastic feedback  
![](images/33352f65681ba174f7ae2139259e730bc679b471718b35346f2c04d067c57ac7.jpg)  
(c) Plastic feedback

![](images/fb19ba8765bfa1481ee1b39c41a1e71662a929d10f1b7ec2df66a3582e46bc5f.jpg)

(d) Loading and unloading  
![](images/825cd0808c1b154af35ac2e471e02905cee43eeeb3cd24019800e4161b283488.jpg)  
Port displacement δ/L (%)

![](images/2372b97fc0aa38f10e494d12a374983c4c4203bde90c5337fcef2b46a4bf66d9.jpg)  
Figure 9: Perforated connection optimized with (b) elastic and (c) elastoplastic response. Panel (d) compares both designs under the same elastoplastic loading–unloading cycle; (e,f) show equivalent plastic strain at the peak load.

The largest diferences in the strain fields occur at the fixed ends of the upper and lower members. The elastoplastic design reduces these concentrations, lowering the maximum equivalent plastic strain to approximately one quarter of the elastic-design value. Its plastically active material region is larger in the common evaluation, but the strain is less concentrated. Plastic deformation is therefore distributed over a broader material region at lower intensity.

On unloading, residual displacement at the loaded port decreases from 0.319% to 0.054% of the span, a reduction of 83.0%. This improvement follows from the design obtained under the loading objective. It shows that similar overall topologies can have substantially diferent permanent deformations, and that incorporating material history can improve the response through changes in member dimensions and connections.

## 5. Discussion and conclusions

The numerical studies show that HGTO retains the design flexibility of neural parameterization while achieving compliance comparable to SIMP–OC across the linear benchmarks, irregular domains, and three-dimensional examples. Several planar designs attain this stifness with fewer secondary members and larger internal openings. The fine-resolution cantilevers preserve these broad arrangements as their boundaries and junctions become more clearly resolved, while the three-dimensional designs form webs, branching supports, and closed hollow sections according to the loading. A compact graph parameterization can therefore accommodate diferent internal layouts without sacrificing structural performance. Shared network weights coordinate density changes across elements, and mesh adjacency supplies local spatial information. Filtering controls the feature scale, while projection sharpens the solid–void boundary.

For nonlinear design, the quality of mechanical feedback also afects the final structural response. HGTO completes the finite-deformation optimizations as the members rotate and the loaded boundaries move, and incorporates loading history in the elastoplastic connection. In the latter case, the design with slightly lower initial stifness develops less concentrated plastic strain and substantially less permanent deformation. Its principal members remain similar to those of the elastic-feedback design, but local changes in their dimensions and connections improve the response over the loading–unloading cycle. The benefit of nonlinear feedback is thus evident in the structural response even when the overall topology changes little. The hypergraph operators carry the evolving geometry and material history into the element sensitivities used to train the density field.

HGTO couples material representation and mechanics through a common finite-element structure. Shared element relations connect each density output to its constitutive response, nodal state, and design sensitivity. The mesh thus organizes both the trainable material field and the physical operations that evaluate it. This correspondence is retained as the problem changes: an opening alters the element neighborhoods and nodal connections, whereas a constitutive extension changes the response and internal variables at the physics hyperedges. The irregular and nonlinear examples show that these changes can be accommodated within the same topology-network construction and density–state coupling.

This integration also reduces the cost of neural material design. In the standard planar and three-dimensional comparisons, HGTO completes the optimization in seconds rather than the minutes to approximately one hour required by the representative dual-field neural baseline. Equilibrium is evaluated through prescribed element operations on the nodal states. Shared graph weights keep the trainable design representation compact as the mesh is refined, while cached element data and parallel physical operations support repeated state evaluation. The same 2D network configuration optimizes the 307,200-element cantilever in 195.7 s. HGTO thus combines the flexibility of a learned density field with the computational structure of finite-element mechanics, bringing design quality, eficiency, and adaptability into one diferentiable formulation.

The present implementation optimizes each problem separately on a fixed mesh and requires equilibrium evaluation at every density update. Spatial detail and feature size remain linked to the discretization and filter settings, while longer nonlinear loading histories add to the state-evaluation cost. Adaptive meshes and multiscale graph parameterizations could improve the treatment of diferent spatial scales, and transferring density-network parameters between related problems could reduce repeated optimization. The stresses and internal variables already carried by the physics hyperedges also provide a basis for extending the formulation to local stress constraints and residual-deformation objectives, broadening the structural requirements that can guide neural material design.

## Code availability

The source code will be made available at https://github.com/Liukz233/HGTO.

## References

[1] M. P. Bendsøe, N. Kikuchi, Generating optimal topologies in structural design using a homogenization method, Computer Methods in Applied Mechanics and Engineering 71 (2) (1988) 197–224. doi:10.1016/0045-7825(88)90086-2.

[2] M. P. Bendsøe, Optimal shape design as a material distribution problem, Structural Optimization 1 (4) (1989) 193–202. doi:10.1007/BF01650949.

[3] M. P. Bendsøe, O. Sigmund, Topology Optimization: Theory, Methods, and Applications, 2nd Edition, Springer Berlin, Heidelberg, 2004. doi:10.1007/978-3-662-05086-6.

[4] K. Svanberg, The method of moving asymptotes—a new method for structural optimization, International Journal for Numerical Methods in Engineering 24 (2) (1987) 359–373. doi:10.1002/nme.1620240207.

[5] B. Bourdin, Filters in topology optimization, International Journal for Numerical Methods in Engineering 50 (9) (2001) 2143–2158. doi:10.1002/nme.116.

[6] J. K. Guest, J. H. Prévost, T. Belytschko, Achieving minimum length scale in topology optimization using nodal design variables and projection functions, International Journal for Numerical Methods in Engineering 61 (2) (2004) 238–254. doi:10.1002/nme.1064.

[7] F. Wang, B. S. Lazarov, O. Sigmund, On projection methods, convergence and robust formulations in topology optimization, Structural and Multidisciplinary Optimization 43 (6) (2011) 767–784. doi:10.1007/s00158-010-0602-y.

[8] O. Sigmund, K. Maute, Topology optimization approaches: A comparative review, Structural and Multidisciplinary Optimization 48 (6) (2013) 1031–1055. doi:10.1007/ s00158-013-0978-6.

[9] E. Andreassen, A. Clausen, M. Schevenels, B. S. Lazarov, O. Sigmund, Eficient topology optimization in MATLAB using 88 lines of code, Structural and Multidisciplinary Optimization 43 (1) (2011) 1–16. doi:10.1007/s00158-010-0594-7.

[10] N. Aage, E. Andreassen, B. S. Lazarov, O. Sigmund, Giga-voxel computational morphogenesis for structural design, Nature 550 (7674) (2017) 84–86. doi:10.1038/nature23911.

[11] T. Buhl, C. B. W. Pedersen, O. Sigmund, Stifness design of geometrically nonlinear structures using topology optimization, Structural and Multidisciplinary Optimization 19 (2) (2000) 93–104. doi:10.1007/s001580050089.

[12] I. Sosnovik, I. Oseledets, Neural networks for topology optimization, Russian Journal of Numerical Analysis and Mathematical Modelling 34 (4) (2019) 215–223. doi:10.1515/ rnam-2019-0018.

[13] Y. Yu, T. Hur, J. Jung, I. G. Jang, Deep learning for determining a near-optimal topological design without any iteration, Structural and Multidisciplinary Optimization 59 (3) (2019) 787–799. doi:10.1007/s00158-018-2101-5.

[14] Z. Nie, T. Lin, H. Jiang, L. B. Kara, TopologyGAN: Topology optimization using generative adversarial networks based on physical fields over the initial domain, Journal of Mechanical Design 143 (3) (2021) 031715. doi:10.1115/1.4049533.

[15] F. Mazé, F. Ahmed, Difusion models beat GANs on topology optimization, Proceedings of the AAAI Conference on Artificial Intelligence 37 (8) (2023) 9108–9116. doi:10.160 9/aaai.v37i8.26093.

[16] S. Hoyer, J. Sohl-Dickstein, S. Greydanus, Neural reparameterization improves structural optimization, arXiv:1909.04240 (2019). arXiv:1909.04240, doi:10.48550/arXiv.190 9.04240.

[17] A. Chandrasekhar, K. Suresh, TOuNN: Topology optimization using neural networks, Structural and Multidisciplinary Optimization 63 (3) (2021) 1135–1149. doi:10.1007/ s00158-020-02748-4.

[18] Z. Zhang, Y. Li, W. Zhou, X. Chen, W. Yao, Y. Zhao, TONR: An exploration for a novel way combining neural network with topology optimization, Computer Methods in Applied Mechanics and Engineering 386 (2021) 114083. doi:10.1016/j.cma.2021.114083.

[19] R. V. Woldseth, N. Aage, J. A. Bærentzen, O. Sigmund, On the use of artificial neural networks in topology optimisation, Structural and Multidisciplinary Optimization 65 (10) (2022) 294. doi:10.1007/s00158-022-03347-1.

[20] E. Samaniego, C. Anitescu, S. Goswami, V. M. Nguyen-Thanh, H. Guo, K. Hamdia, X. Zhuang, T. Rabczuk, An energy approach to the solution of partial diferential equations in computational mechanics via machine learning: Concepts, implementation and applications, Computer Methods in Applied Mechanics and Engineering 362 (2020) 112790. doi:10.1016/j.cma.2019.112790.

[21] H. Jeong, J. Bai, C. P. Batuwatta-Gamage, C. Rathnayaka, Y. Zhou, Y. Gu, A physicsinformed neural network-based topology optimization (PINNTO) framework for structural optimization, Engineering Structures 278 (2023) 115484. doi:10.1016/j.engstruct. 2022.115484.

[22] J. Zehnder, Y. Li, S. Coros, B. Thomaszewski, NTopo: Mesh-free topology optimization using implicit neural representations, in: Advances in Neural Information Processing Systems, Vol. 34, 2021, pp. 10368–10381. arXiv:2102.10782.

[23] H. Jeong, C. Batuwatta-Gamage, J. Bai, Y. M. Xie, C. Rathnayaka, Y. Zhou, Y. Gu, A complete physics-informed neural network-based framework for structural topology optimization, Computer Methods in Applied Mechanics and Engineering 417 (2023) 116401. doi:10.1016/j.cma.2023.116401.

[24] A. Joglekar, H. Chen, L. B. Kara, DMF-TONN: Direct mesh-free topology optimization using neural networks, Engineering with Computers 40 (2024) 2227–2240. doi:10.100 7/s00366-023-01904-w.

[25] A. Singh, S. Chakraborty, R. Chowdhury, A dual physics-informed neural network for topology optimization, Journal of Computational Physics 551 (2026) 114666. doi: 10.1016/j.jcp.2026.114666.

[26] M. Seo, S. Min, Graph neural networks and implicit neural representation for near-optimal topology prediction over irregular design domains, Engineering Applications of Artificial Intelligence 123 (2023) 106284. doi:10.1016/j.engappai.2023.106284.

[27] Y. Joo, H. Choi, G.-E. Jeong, Y. Yu, Dynamic graph-based convergence acceleration for topology optimization in unstructured meshes, Engineering Applications of Artificial Intelligence 132 (2024) 107916. doi:10.1016/j.engappai.2024.107916.

[28] G. B. Gavris, W. Sun, Topology optimization with graph neural network enabled regularized thresholding, Extreme Mechanics Letters 71 (2024) 102215. doi:10.1016/j. eml.2024.102215.

[29] A. Tabarraei, S. A. Bhuiyan, Diferentiable graph neural fields for manufacturable topology optimization under stress constraints, Computers & Structures 330 (2026) 108360. doi:10.1016/j.compstruc.2026.108360.

[30] S. A. Bhuiyan, A. Tabarraei, Graph neural network-based topology optimization for eficient support structure design in additive manufacturing, Engineering with Computers 42 (2026) 103. doi:10.1007/s00366-026-02344-y.

[31] H. Gao, M. J. Zahr, J.-X. Wang, Physics-informed graph neural galerkin networks: A unified framework for solving pde-governed forward and inverse problems, Computer Methods in Applied Mechanics and Engineering 390 (2022) 114502. doi:10.1016/j.cm a.2021.114502.

[32] J. He, D. Abueidda, S. Koric, I. Jasiuk, On the use of graph neural networks and shapefunction-based gradient computation in the deep energy method, International Journal for Numerical Methods in Engineering 124 (4) (2023) 864–879. doi:10.1002/nme.7146.

[33] R. Gao, I. K. Deo, R. K. Jaiman, A finite element-inspired hypergraph neural network: Application to fluid dynamics simulations, Journal of Computational Physics 504 (2024) 112866. doi:10.1016/j.jcp.2024.112866.

[34] J. Yang, X. Chen, J. Zhao, FEM-informed hypergraph neural networks for eficient elastoplasticity, arXiv:2602.07364 (2026). arXiv:2602.07364, doi:10.48550/arXiv.2 602.07364.

[35] M. Deferrard, X. Bresson, P. Vandergheynst, Convolutional neural networks on graphs with fast localized spectral filtering, in: Advances in Neural Information Processing Systems, Vol. 29, 2016, pp. 3844–3852.

[36] T. Gustafsson, G. D. McBain, scikit-fem: A Python package for finite element assembly, Journal of Open Source Software 5 (52) (2020) 2369. doi:10.21105/joss.02369. URL https://doi.org/10.21105/joss.02369

[37] F. Wang, B. S. Lazarov, O. Sigmund, J. S. Jensen, Interpolation scheme for fictitious domain techniques and topology optimization of finite strain elastic problems, Computer Methods in Applied Mechanics and Engineering 276 (2014) 453–472. doi:10.1016/j. cma.2014.03.021.

[38] J. C. Simo, T. J. R. Hughes, Computational Inelasticity, Vol. 7 of Interdisciplinary Applied Mathematics, Springer, New York, 1998. doi:10.1007/b98904.

# Supplementary Material

HGTO: A Unified Graph-Based Physics-Informed Formulation for Structural Topology Optimization

## S1. Density mapping and optimization settings

## S1.1. Density filtering and projection

For element centroids $X _ { e }$ and volumes $V _ { e }$ , the filter weights and normalized averaging matrix are

$$
w _ { e j } = \operatorname* { m a x } \Big ( 0 , r _ { f } - \| \mathbf { X } _ { e } - \mathbf { X } _ { j } \| _ { 2 } \Big ) , \qquad A _ { e j } = \frac { V _ { j } w _ { e j } } { \sum _ { k } V _ { k } w _ { e k } } .\tag{S1}
$$

The distance weights define the averaging neighborhood, and the volume weights account for unequal element sizes. This is the density filter used for HGTO and SIMP–OC [5, 9].

For network logits z and a common shift $\lambda _ { V }$ , the bounded design variables and physical densities are

$$
\begin{array} { r } { x _ { e } = \mathrm { s i g m o i d } ( z _ { e } - \lambda _ { V } ) , \qquad \bar { x } = A x , } \\ { \rho _ { e } = \rho _ { \mathrm { m i n } } + ( 1 - \rho _ { \mathrm { m i n } } ) \mathcal { H } _ { \beta } ( \bar { x } _ { e } ) , \qquad } \\ { \mathcal { H } _ { \beta } ( s ) = \frac { \operatorname { t a n h } ( \beta / 2 ) + \operatorname { t a n h } [ \beta ( s - 1 / 2 ) ] } { 2 \operatorname { t a n h } ( \beta / 2 ) } . } \end{array}\tag{S2}
$$

The smooth projection sharpens the filtered density as $\beta$ increases [7]. The filter radius remains fixed during projection continuation. The shift is determined by

$$
F _ { V } ( z , \lambda _ { V } ) = \sum _ { e } V _ { e } \rho _ { e } ( z , \lambda _ { V } ) - V _ { f } V _ { \Omega } = 0 .\tag{S3}
$$

Since the density decreases monotonically with $\lambda _ { V }$ , this scalar equation is solved by a safeguarded Newton–bisection iteration.

## S1.2. Diferentiation of the volume constraint

Let $\pmb { v } = ( V _ { 1 } , \ldots , V _ { N _ { e } } ) ^ { \top }$ . The density Jacobian with the shift held fixed is

$$
\begin{array} { r } { \pmb { J } _ { 0 } = \mathrm { d i a g } \big [ ( 1 - \rho _ { \mathrm { m i n } } ) \pmb { \mathcal { H } } _ { \beta } ^ { \prime } ( \bar { x } _ { e } ) \big ] \pmb { A } \mathrm { d i a g } [ x _ { e } ( 1 - x _ { e } ) ] , } \end{array}\tag{S4}
$$

where

$$
\mathcal { H } _ { \beta } ^ { \prime } ( s ) = \frac { \beta \operatorname { s e c h } ^ { 2 } [ \beta ( s - 1 / 2 ) ] } { 2 \operatorname { t a n h } ( \beta / 2 ) } .\tag{S5}
$$

Diferentiating Eq. (S3) gives ${ \pmb v } ^ { \mathsf { T } } { \pmb J } _ { 0 } d z - { \pmb v } ^ { \mathsf { T } } { \pmb J } _ { 0 } { \pmb 1 } d \lambda _ { V } = 0$ . The full density derivative is therefore

$$
\frac { \partial \pmb { \rho } } { \partial z } = \pmb { J } _ { 0 } - \frac { ( \pmb { J } _ { 0 } \pmb { 1 } ) ( \pmb { v } ^ { \top } \pmb { J } _ { 0 } ) } { \pmb { v } ^ { \top } \pmb { J } _ { 0 } \pmb { 1 } } .\tag{S6}
$$

Its volume-weighted sum is zero, so the linearized density change preserves the prescribed volume. For physical-density sensitivities $g _ { \rho } ,$ , backpropagation evaluates

$$
\pmb { g } _ { z } = \pmb { J } _ { 0 } ^ { \top } \pmb { g } _ { \rho } - \frac { \pmb { g } _ { \rho } ^ { \top } \pmb { J } _ { 0 } \pmb { 1 } } { \pmb { v } ^ { \top } \pmb { J } _ { 0 } \pmb { 1 } } \pmb { J } _ { 0 } ^ { \top } \pmb { v } .\tag{S7}
$$

The products in Eq. (S7) use the sparse filter matrix and pointwise projection derivatives.

## S1.3. Network and optimization parameters

HGTO uses two first-degree Chebyshev layers and fixed Fourier features. The linear 2D studies use 64 frequencies, feature standard deviation 2, and hidden width 64, giving 16,577 trainable parameters. The linear 3D studies use 128 frequencies and width 128, giving 65,921 parameters. Network weights are shared across elements. Each HGTO run starts from a uniform physical-density field with random seed 0. The density floor is 0.001.

For the linear studies, the continuation stages are $( p , \beta ) = ( 1 , 1 ) , ( 2 , 2 ) , ( 3 , 4 ) , ( 3 , 8 )$ . The first three stages contain 50 updates each. Adam learning rates decrease from 0.01 to 0.001, with a 200-update reference length for learning-rate decay in the final stage. SIMP–OC uses the same continuation, with move limit $0 . 2 / \operatorname* { m a x } ( 1 , \beta )$

At the final material parameters, convergence requires a relative objective range below $1 0 ^ { - 3 }$ over ten checks and a maximum physical-density change below $5 \times 1 0 ^ { - 3 }$ . Both conditions must hold for five consecutive checks, with volume error below $1 0 ^ { - 7 }$ and relative equilibrium residual below $1 0 ^ { - 8 }$ . The total update limit is 1,000. Table S1 lists the achieved volumes and HGTO update counts.

Table S1: Prescribed and achieved volume fractions and HGTO update counts.
<table><tr><td>Case</td><td>Target volume</td><td>NTopo volume</td><td>HGTO updates</td></tr><tr><td>Cantilever  $1 2 0 \times 4 0$ </td><td>0.50</td><td>0.501115</td><td>282</td></tr><tr><td>Half-MBB</td><td>0.50</td><td>0.500576</td><td>280</td></tr><tr><td>Inclined load</td><td>0.50</td><td>0.499636</td><td>293</td></tr><tr><td>Cantilever  $2 4 0 \times 8 0$ </td><td>0.50</td><td>0.500707</td><td>257</td></tr><tr><td>Cantilever 480 × 160</td><td>0.50</td><td>0.500738</td><td>311</td></tr><tr><td>Cantilever  $9 6 0 \times 3 2 0$ </td><td>0.50</td><td>0.500734</td><td>344</td></tr><tr><td>L-bracket</td><td>0.40</td><td>0.400602</td><td>336</td></tr><tr><td>Perforated bracket</td><td>0.40</td><td>0.400812</td><td>228</td></tr><tr><td>3D cantilever</td><td>0.30</td><td>0.299229</td><td>243</td></tr><tr><td>Four-foot support</td><td>0.18</td><td>0.179505</td><td>257</td></tr><tr><td>Torsion member</td><td>0.20</td><td>0.200417</td><td>230</td></tr></table>

HGTO and SIMP–OC satisfy the volume constraint within $1 0 ^ { - 7 }$ . All HGTO runs meet the stopping criterion. The inclined-load SIMP–OC run reaches its update limit; the other listed SIMP–OC runs converge.

NTopo follows the original network and training settings [22], using hidden widths of 60 in 2D and 180 in 3D, with 200 and 100 outer iterations, respectively. Each configuration uses one initialization, with seed 42 for NTopo. Final linear designs are evaluated with the common material model specified in the main text.

## S2. Domains, resolution, and evaluation

The common linear evaluator uses bilinear quadrilaterals in 2D and trilinear hexahedra in 3D, with full quadrature. Two-dimensional linear cases use plane stress and unit thickness. HGTO mechanics uses FP64 node–element–node operator actions. The optimized physical densities are passed directly to the independent evaluator. All common compliance values use $p = 3 , E _ { 0 } = 1 , E _ { \mathrm { m i n } } = 1 0 ^ { - 6 }$ , and $\nu = 0 . 3$

The standard cantilever has a downward unit force at its right midpoint and a fully fixed left edge. The half-MBB beam restrains horizontal displacement along the left edge and vertical displacement at the lower-right corner, with a downward load at the upper-left corner.

Table S2: Linear design domains and HGTO/SIMP–OC filter radii. Grid dimensions count density elements. Radii are expressed in the same length units as the listed domains.
<table><tr><td>Case</td><td>Domain size</td><td>Discretization</td><td>Radius</td></tr><tr><td>Standard cantilevers / half-MBB</td><td> $1 2 0 \times 4 0$ </td><td> $1 2 0 \times 4 0$ </td><td>3</td></tr><tr><td>Refined cantilever</td><td> $2 4 0 \times 8 0$ </td><td> $2 4 0 \times 8 0$ </td><td>6</td></tr><tr><td>Fine cantilever</td><td> $4 8 0 \times 1 6 0$ </td><td> $4 8 0 \times 1 6 0$ </td><td>6</td></tr><tr><td>Finest cantilever</td><td> $9 6 0 \times 3 2 0$ </td><td> $9 6 0 \times 3 2 0$ </td><td>6</td></tr><tr><td>L-bracket</td><td></td><td>2 × 2 bounding box 4,800 quadrilaterals</td><td>0.09</td></tr><tr><td>Perforated bracket</td><td> $3 \times 2 \ \mathrm { b o u n d i n g }$ </td><td>box 3,852 quadrilaterals</td><td>0.12</td></tr><tr><td>3D cantilever</td><td> $4 8 \times 1 6 \times 1 6$ </td><td> $4 8 \times 1 6 \times 1 6$ </td><td>1.6</td></tr><tr><td>Four-foot support</td><td> $2 4 \times 3 2 \times 2 4$ </td><td> $2 4 \times 3 2 \times 2 4$ </td><td>1.6</td></tr><tr><td>Torsion member</td><td> $3 2 \times 2 4 \times 2 4$ </td><td> $3 2 \times 2 4 \times 2 4$ </td><td>1.6</td></tr></table>

The inclined cantilever is loaded at three-quarters of the right-edge height, with the force inclined inward by $2 5 ^ { \circ }$ from the downward direction.

For the L-bracket, the upper-right unit square is removed from a $2 \times 2$ domain. The upper mounting edge is fixed, and the right edge over $0 . 8 \leq y \leq 1$ carries a downward unit resultant. The perforated bracket has a central circular clearance of radius 0.5 and a nonuniform conforming quadrilateral mesh. Its left edge is fixed, and the right edge over $0 . 9 \leq y \leq 1 . 1$ carries a downward unit resultant. Removed regions contribute neither material volume nor mechanical energy. The same excluded regions and loaded boundaries are used for the neural baseline.

In 3D, the cantilever has a fixed left face and a $2 \times 2$ central loading patch on the right. The four-foot support fixes four separate $3 \times 3$ patches on its lower face and applies a downward unit resultant over a 4 × 4 top pad. Torsion is applied by opposed tractions on two $4 \times 3$ end patches, with zero net force and a unit moment about the longitudinal axis. All densities in these three domains are design variables.

Table S3: Compliance and total computation time for the cantilever resolution study.
<table><tr><td rowspan="2">Resolution</td><td colspan="2">SIMP-OC</td><td colspan="2">NTopo</td><td colspan="2">HGTO</td></tr><tr><td>C</td><td>t (s)</td><td>C</td><td>t (s)</td><td>C</td><td>t (s)</td></tr><tr><td> $1 2 0 \times 4 0$ </td><td>175.0823</td><td>9.02</td><td>177.9284</td><td>862.24</td><td>176.4933</td><td>8.27</td></tr><tr><td> $2 4 0 \times 8 0$ </td><td>175.7651</td><td>30.87</td><td>178.0319</td><td>862.08</td><td>176.8524</td><td>11.72</td></tr><tr><td> $4 8 0 \times 1 6 0$ </td><td>171.5841</td><td>197.52</td><td>178.4512</td><td>863.68</td><td>172.4415</td><td>47.35</td></tr><tr><td> $9 6 0 \times 3 2 0$ </td><td>168.5552</td><td>1880.10</td><td>178.9222</td><td>867.61</td><td>170.9963</td><td>195.75</td></tr></table>

The cantilever grids share a 3:1 aspect ratio. The filter radius divided by the span is 0.025 on the two coarser grids, then 0.0125 and 0.00625 on the two finer grids (Table S2). For the resolution study, the NTopo results use one density network trained with $1 5 0 \times 5 0$ samples per batch and evaluated on each listed grid; its reported time includes training and evaluation.

## S3. Finite-deformation formulation

## S3.1. Kinematics and material interpolation

The finite-deformation examples use a total-Lagrangian description. For reference coordinates X and nodal displacements ${ \pmb u } _ { a }$ , the deformation gradient at an integration point of

element e is

$$
\mathbf { \nabla } F = I + G , \qquad \mathbf { \nabla } G = \nabla _ { X } \pmb { u } = \sum _ { a \in e } \pmb { u } _ { a } \otimes \nabla _ { X } N _ { a } .\tag{S1}
$$

Here $N _ { a }$ are the element shape functions. The solid material is described by the twodimensional compressible Neo-Hookean energy

$$
\psi _ { \mathrm { N H } } ( { \pmb F } ) = \frac { \mu _ { 0 } } { 2 } ( { \pmb F } : { \pmb F } - 2 - 2 \ln j ) + \frac { \lambda _ { 0 } } { 2 } ( \ln j ) ^ { 2 } , \qquad j = \operatorname * { d e t } { \pmb F } > 0 ,\tag{S2}
$$

with $\mu _ { 0 } = E _ { 0 } / [ 2 ( 1 + \nu ) ]$ and $\lambda _ { 0 } = E _ { 0 } \nu / ( 1 - \nu ^ { 2 } )$ . The energy is evaluated with two-dimensional kinematics and has a plane-stress elastic response in the small-strain limit.

Low-density regions are treated with the energy interpolation of Wang et al. [37]. Define the stifness scale and kinematic switch as

$$
s ( \rho ) = \frac { E _ { \mathrm { m i n } } + ( E _ { 0 } - E _ { \mathrm { m i n } } ) \rho ^ { p } } { E _ { 0 } } ,\tag{S3}
$$

$$
\gamma ( \rho ) = \frac { \operatorname { t a n h } ( \beta _ { 0 } \eta _ { 0 } ) + \operatorname { t a n h } [ \beta _ { 0 } ( \rho ^ { p } - \eta _ { 0 } ) ] } { \operatorname { t a n h } ( \beta _ { 0 } \eta _ { 0 } ) + \operatorname { t a n h } [ \beta _ { 0 } ( 1 - \eta _ { 0 } ) ] } ,\tag{S4}
$$

where $\beta _ { 0 } = 5 0 0$ and $\eta _ { 0 } = 0 . 0 1$ . The small-strain companion energy is

$$
\psi _ { \mathrm { L } } ( G ) = \mu _ { 0 } \ \mathrm { s y m } G : \mathrm { s y m } G + \frac { \lambda _ { 0 } } { 2 } ( \mathrm { t r } G ) ^ { 2 } .\tag{S5}
$$

The energy used in the density-dependent mechanical calculation is

$$
\psi _ { \rho } ( \mathbf { G } ) = s ( \rho ) \left[ \psi _ { \mathrm { N H } } ( \pmb { F } _ { \gamma } ) - \psi _ { \mathrm { L } } ( \gamma \pmb { G } ) + \psi _ { \mathrm { L } } ( \pmb { G } ) \right] , \qquad \pmb { F } _ { \gamma } = \pmb { I } + \gamma \pmb { G } .\tag{S6}
$$

For $\gamma = 1$ , this expression recovers the Neo-Hookean energy scaled by s. For $\gamma = 0 ,$ , it reduces to the scaled linear energy. The switch therefore confines the finite-strain response to material regions while assigning the linear response to low-stifness regions. Both the HGTO and baseline nonlinear calculations use this interpolation.

## S3.2. Physical operators and incremental equilibrium

Diferentiating Eq. (S2) gives the first Piola stress and material tangent of the solid,

$$
\begin{array} { r } { P _ { \mathrm { N H } } ( \pmb { F } ) = \mu _ { 0 } ( \pmb { F } - \pmb { F } ^ { - \top } ) + \lambda _ { 0 } \ln j \pmb { F } ^ { - \top } , } \end{array}\tag{S7}
$$

$$
( \mathbb { A } _ { \mathrm { N H } } ) _ { i J k L } = \mu _ { 0 } \delta _ { i k } \delta _ { J L } + \lambda _ { 0 } { F _ { i J } ^ { - \mathsf { T } } } { F _ { k L } ^ { - \mathsf { T } } } + ( \mu _ { 0 } - \lambda _ { 0 } \ln j ) F _ { i L } ^ { - \mathsf { T } } F _ { k J } ^ { - \mathsf { T } } .\tag{S8}
$$

Writing $Q _ { \mathrm { L } } ( G ) = 2 \mu _ { 0 } \mathrm { s y m } G + \lambda _ { 0 } \mathrm { t r } ( G ) I$ and $\mathbb { A } _ { \mathrm { L } } = \partial Q _ { \mathrm { L } } / \partial G$ , the stress and tangent associated with Eq. (S6) are

$$
P _ { \rho } = s \left[ \gamma P _ { \mathrm { N H } } ( { F } _ { \gamma } ) - \gamma Q _ { \mathrm { L } } ( \gamma \pmb { G } ) + Q _ { \mathrm { L } } ( \pmb { G } ) \right] ,\tag{S9}
$$

$$
\mathbb { A } _ { \rho } = s \left[ \gamma ^ { 2 } \mathbb { A } _ { \mathrm { N H } } ( \pmb { F } _ { \gamma } ) + ( 1 - \gamma ^ { 2 } ) \mathbb { A } _ { \mathrm { L } } \right] .\tag{S10}
$$

These are prescribed operations on each physics hyperedge. For quadrature weight $\omega _ { e g } ,$ including the reference Jacobian determinant and thickness, the element force and tangent are

$$
( { \bf f } _ { e } ) _ { a i } = \sum _ { g } ( P _ { \rho } ) _ { i J } N _ { a , J } \omega _ { e g } ,\tag{S11}
$$

$$
( \boldsymbol { K } _ { e } ) _ { a i , b k } = \sum _ { g } N _ { a , J } ( \mathbb { A } _ { \rho } ) _ { i J k L } N _ { b , L } \omega _ { e g } .\tag{S12}
$$

Repeated spatial indices are summed. The incidence-based gather and scatter operations assemble these quantities as

$$
\mathbf { \Delta } f _ { \mathrm { i n t } } = \sum _ { e } { P _ { e } ^ { \mathrm { T } } f _ { e } } , \qquad \mathbf { \nabla } K _ { \mathrm { T } } = \sum _ { e } { P _ { e } ^ { \mathrm { T } } K _ { e } P _ { e } } ,\tag{S13}
$$

where $P _ { e }$ extracts element degrees of freedom from the global state. The reference shapefunction gradients and incidence are reused as the density and displacement fields evolve.

Loading is applied incrementally, with $\mathbf { \Delta } f _ { n } = \ell _ { n } \mathbf { \Delta } f _ { \operatorname* { m a x } }$ and $0 < \ell _ { n } \leq 1$ . At each increment, equilibrium on the free degrees of freedom is obtained from

$$
\begin{array} { r } { { \cal R } _ { n } ( { \pmb u } _ { n } , \pmb { \rho } ) = f _ { \mathrm { i n t } } ( \pmb u _ { n } , \pmb { \rho } ) - \pmb f _ { n } = \mathbf { 0 } . } \end{array}\tag{S14}
$$

A Newton step solves $\pmb { K } _ { \mathrm { T } } \Delta \pmb { u } = - \pmb { R } _ { n }$ , followed by $\pmb { u } \gets \pmb { u } + \alpha \Delta \pmb { u }$ . Backtracking controls the decrease in total potential energy and rejects trial states with nonpositive det ${ \cal { F } } _ { \gamma }$ . If the Newton direction is not a descent direction, it is replaced by the negative residual rescaled to the same norm before backtracking. Prescribed displacements are imposed directly. The converged state of the preceding load increment initializes the next increment.

## S3.3. Design objective and density derivative

The finite-deformation objective is twice the complementary work at the final load,

$$
J ( \pmb { \rho } ) = 2 \left[ \pmb { f } _ { \mathrm { m a x } } ^ { \top } \pmb { u } _ { N } - U ( \pmb { u } _ { N } , \pmb { \rho } ) \right] , \qquad U = \sum _ { e , g } \psi _ { \rho _ { e } } ( \pmb { G } _ { e g } ) \omega _ { e g } .\tag{S15}
$$

For linear elasticity, $\begin{array} { r } { U = \frac { 1 } { 2 } \pmb { f } _ { \mathrm { m a x } } ^ { \top } \pmb { u } _ { N } } \end{array}$ at equilibrium, so this objective reduces to compliance. At a converged nonlinear state, $\partial U / \partial { \pmb u } _ { N } = { \pmb f } _ { \mathrm { m a x } }$ . The terms involving $d { \bf { u } } _ { N } / d \rho _ { e }$ therefore cancel, giving

$$
\frac { d J } { d \rho _ { e } } = - 2 \frac { \partial U } { \partial \rho _ { e } } = - 2 \sum _ { g } \left. \frac { \partial \psi _ { \rho _ { e } } } { \partial \rho _ { e } } \right. _ { G _ { e g } } \omega _ { e g } .\tag{S16}
$$

The derivative includes both the stifness interpolation and the kinematic switch. Let $B = \psi _ { \mathrm { N H } } ( F _ { \gamma } ) - \psi _ { \mathrm { L } } ( \gamma G ) + \psi _ { \mathrm { L } } ( G )$ . Then

$$
\left. \frac { \partial \psi _ { \rho } } { \partial \rho } \right| _ { G } = s ^ { \prime } B + s \gamma ^ { \prime } \left[ P _ { \mathrm { N H } } ( F _ { \gamma } ) - { \cal Q } _ { \mathrm { L } } ( \gamma { \cal G } ) \right] : G ,\tag{S17}
$$

where

$$
s ^ { \prime } = \frac { p \rho ^ { p - 1 } ( E _ { 0 } - E _ { \mathrm { m i n } } ) } { E _ { 0 } } , \qquad \gamma ^ { \prime } = \frac { \beta _ { 0 } p \rho ^ { p - 1 } \mathrm { s e c h } ^ { 2 } [ \beta _ { 0 } ( \rho ^ { p } - \eta _ { 0 } ) ] } { \operatorname { t a n h } ( \beta _ { 0 } \eta _ { 0 } ) + \operatorname { t a n h } [ \beta _ { 0 } ( 1 - \eta _ { 0 } ) ] } .\tag{S18}
$$

The element sensitivity is propagated through the constrained density map in Section S1 and the topology network.

## S3.4. Problem and optimization settings

The solid parameters are $E _ { 0 } = 1 , \nu = 0 . 3$ , and $E _ { \mathrm { m i n } } = 1 0 ^ { - 6 }$ . HGTO uses 64 Fourier frequencies, feature standard deviation 2, and hidden width 64. Adam learning rates decrease from 0.01 to 0.001 over a 180-update reference schedule, followed by exponential decay with half-life 20 and floor 10<sup>−5</sup>. Gradient clipping is 0.5. Penalization reaches $p = 3$ by update 125 and projection reaches $\beta = 8$ by update 144. The density floor is 0.001 and the physical filter radius is 1. HGTO and SIMP–OC use twelve load increments, a final relative equilibrium tolerance of $1 0 ^ { - 8 }$ , and the joint physical stopping rule in Section S1.

The cantilever domain is $2 4 \times 6$ , discretized by $9 6 \times 2 4$ cells. The left edge is fixed. A downward resultant of 0.00125 or 0.0125 is applied to the right-edge port over $2 . 2 5 \leq y \leq 3 . 7 5$

The bridge domain is $2 4 \times 8 ,$ discretized by $9 6 \times 3 2$ cells, with both side edges fixed. A downward resultant of 0.1 acts over a three-unit segment centered on the upper edge. The bridge averages HGTO logits with their spanwise reflection while retaining the full state mesh.

Nonlinear SIMP–OC uses up to ten step halvings with volume restoration to keep relative objective increases below $1 0 ^ { - 8 }$ at unchanged continuation parameters. The strong cantilever terminates with stagnation when no candidate step is accepted, retaining the last accepted density.

The nonlinear NTopo comparison adapts the original training procedure to the finitedeformation model above. The weak-load cantilever completes its 200-iteration budget at volume fraction 0.450644. The strong-load cantilever and bridge terminate during state training after 27.84 s and 13.56 s, respectively, when the deformation determinant becomes nonpositive.

## S4. Elastoplastic formulation

## S4.1. Plane-strain model and local return mapping

The perforated connection uses small-strain associative $J _ { 2 }$ plasticity with linear isotropic hardening [38]. Its displacement field is planar, with $\varepsilon _ { z z } = \varepsilon _ { x z } = \varepsilon _ { y z } = 0$ . The in-plane engineering strains $[ \varepsilon _ { x x } , \varepsilon _ { y y } , \gamma _ { x y } ]$ are embedded in a three-dimensional symmetric tensor using $\varepsilon _ { x y } = \gamma _ { x y } / 2$ . The out-of-plane stress is retained in the yield calculation.

At each element density, the same SIMP scale from Eq. (S3) multiplies Young’s modulus, initial yield stress, and hardening modulus:

$$
E _ { e } = s ( \rho _ { e } ) E _ { 0 } , \qquad \sigma _ { y , e } = s ( \rho _ { e } ) \sigma _ { y 0 } , \qquad H _ { e } = s ( \rho _ { e } ) H _ { 0 } .\tag{S1}
$$

Poisson’s ratio is constant. This interpolation preserves the yield strain while scaling the local stress response with density. Define $\mu _ { e } = E _ { e } / [ 2 ( 1 + \nu ) ] , \kappa _ { e } = E _ { e } / [ 3 ( 1 - 2 \nu ) ]$ , and $\lambda _ { e } = \kappa _ { e } - 2 \mu _ { e } / 3$

The internal variables at an integration point are plastic strain $\varepsilon ^ { p }$ and accumulated equivalent plastic strain a. Starting from the committed state of load increment $n - 1$ , the elastic trial stress is

$$
\pmb { \sigma } _ { n } ^ { \mathrm { t r } } = \lambda _ { e } \mathrm { t r } \big ( \pmb { \varepsilon } _ { n } - \pmb { \varepsilon } _ { n - 1 } ^ { p } \big ) \pmb { I } + 2 \mu _ { e } \big ( \pmb { \varepsilon } _ { n } - \pmb { \varepsilon } _ { n - 1 } ^ { p } \big ) .\tag{S2}
$$

With $\pmb { \mathscr { s } } ^ { \mathrm { t r } } = \mathrm { d e v } \pmb { \sigma } ^ { \mathrm { t r } } , t = \| \pmb { \mathscr { s } } ^ { \mathrm { t r } } \|$ , and ${ \boldsymbol { n } } = s ^ { \mathrm { t r } } / t$ for $t > 0$ , the trial yield function is

$$
f ^ { \mathrm { t r } } = t - \sqrt { \frac { 2 } { 3 } } \left( \sigma _ { y , e } + H _ { e } a _ { n - 1 } \right) .\tag{S3}
$$

For $f ^ { \mathrm { t r } } \leq 0$ , the step is elastic and the internal variables are unchanged. Otherwise, the radial return gives

$$
\Delta \gamma = \frac { \operatorname* { m a x } ( f ^ { \mathrm { t r } } , 0 ) } { 2 \mu _ { e } + \frac { 2 } { 3 } H _ { e } } ,\tag{S4}
$$

$$
{ \pmb \sigma } _ { n } = { \pmb \sigma } _ { n } ^ { \mathrm { t r } } - 2 \mu _ { e } \Delta \gamma { \pmb n } ,\tag{S5}
$$

$$
\pmb { \varepsilon } _ { n } ^ { p } = \pmb { \varepsilon } _ { n - 1 } ^ { p } + \Delta \gamma \pmb { n } , \qquad a _ { n } = a _ { n - 1 } + \sqrt { \frac { 2 } { 3 } } \Delta \gamma .\tag{S6}
$$

The flow direction is set to zero when $t = 0$ . Each integration point stores its own internal variables on the corresponding physics hyperedge.

## S4.2. Consistent tangent and equilibrium over a load history

Let $\mathbb { I } _ { \mathrm { s y m } }$ denote the symmetric fourth-order identity and $\mathbb { I } _ { \mathrm { d e v } } = \mathbb { I } _ { \mathrm { s y m } } - \frac { 1 } { 3 } \pmb { I } \otimes \pmb { I }$ . On the plastic branch, define

$$
\vartheta = 1 - \frac { 2 \mu _ { e } \Delta \gamma } { t } , \qquad \bar { \vartheta } = \frac { 1 } { 1 + H _ { e } / ( 3 \mu _ { e } ) } - ( 1 - \vartheta ) .\tag{S7}
$$

The algorithmic tangent of the return map is

$$
\mathbb { C } _ { \mathrm { a l g } } = \kappa _ { e } \pmb { I } \otimes \pmb { I } + 2 \mu _ { e } \vartheta \mathbb { I } _ { \mathrm { d e v } } - 2 \mu _ { e } \bar { \vartheta } \pmb { n } \otimes \pmb { n } .\tag{S8}
$$

On the elastic branch it reduces to $\mathbb { C } _ { \mathrm { e l } } = \kappa _ { e } \pmb { I } \otimes \pmb { I } + 2 \mu _ { e } \mathbb { I } _ { \mathrm { d e v } }$ . The in-plane block, with engineering-shear conventions, gives the matrix ${ \cal D } _ { e g } ^ { \mathrm { a l g } }$ used in the element tangent.

Collect the integration-point histories into $\pmb q _ { n } = ( \pmb { \varepsilon } _ { n } ^ { p } , a _ { n } )$ and denote the return map by $G _ { n }$ . At each load increment,

$$
{ \pmb R } _ { n } ( { \pmb u } _ { n } , { \pmb q } _ { n - 1 } , \pmb \rho ) = \sum _ { e } { \pmb P } _ { e } ^ { \top } \sum _ { g } { \pmb B } _ { e g } ^ { \top } { \pmb \sigma } _ { e g , n } ^ { \vee } \omega _ { e g } - { \pmb f } _ { n } = { \bf 0 } ,\tag{S9}
$$

$$
\pmb q _ { n } = \pmb G _ { n } ( \pmb u _ { n } , \pmb q _ { n - 1 } , \pmb \rho ) ,\tag{S10}
$$

$$
K _ { n } = \frac { \partial R _ { n } } { \partial { \bf u } _ { n } } = \sum _ { e } P _ { e } ^ { \top } \sum _ { g } B _ { e g } ^ { \top } D _ { e g } ^ { \mathrm { a l g } } B _ { e g } \omega _ { e g } P _ { e } .\tag{S11}
$$

Here $\pmb { \sigma } ^ { \mathrm { V } } = [ \sigma _ { x x } , \sigma _ { y y } , \sigma _ { x y } ] ^ { \mathsf { T } }$ , and B maps element displacements to engineering strains. Nodal displacements are updated by Newton iterations using Eq. (S11). During those iterations, each local return is recomputed from $\pmb q _ { n - 1 }$ . The new history is committed only after equilibrium is reached. A new density field is evaluated from the initially stress-free state, with ${ \bf q } _ { 0 } = { \bf 0 }$ , over the full prescribed loading sequence.

## S4.3. Peak-load objective and history-dependent sensitivity

The design objective is the peak-load work measure

$$
W _ { \mathrm { p e a k } } ( \pmb { \rho } ) = \pmb { f } _ { N } ^ { \top } \pmb { u } _ { N } ,\tag{S12}
$$

where N is the final loading increment. This objective measures displacement under the prescribed peak force. Earlier load increments enter through the material history in Eqs. (S9) and (S10).

The discrete adjoint is obtained from the augmented functional

$$
\mathcal { A } = W _ { \mathrm { p e a k } } + \sum _ { n = 1 } ^ { N } \mathbf { \lambda } _ { n } ^ { \mathrm { T } } \mathbf { R } _ { n } + \sum _ { n = 1 } ^ { N } \mu _ { n } ^ { \mathrm { T } } ( \pmb { q } _ { n } - \pmb { G } _ { n } ) .\tag{S13}
$$

All equilibrium derivatives below are restricted to free degrees of freedom, and the initial internal state is fixed. Starting from $\pmb { \mu } _ { N } = \mathbf { 0 }$ , stationarity with respect to ${ \pmb u } _ { n }$ and $\mathbf { \delta q } _ { n - 1 }$ yields the backward recursion

$$
\pmb { K } _ { n } ^ { \mathsf { T } } \pmb { \lambda } _ { n } = - \delta _ { n N } \pmb { f } _ { N } + \left( \frac { \partial \pmb { G } _ { n } } { \partial \pmb { u } _ { n } } \right) ^ { \mathsf { T } } \pmb { \mu } _ { n } ,\tag{S14}
$$

$$
{ \pmb \mu } _ { n - 1 } = - \left( \frac { \partial { \pmb R } _ { n } } { \partial { \pmb q } _ { n - 1 } } \right) ^ { \top } { \pmb \lambda } _ { n } + \left( \frac { \partial { \pmb G } _ { n } } { \partial { \pmb q } _ { n - 1 } } \right) ^ { \top } { \pmb \mu } _ { n } , \qquad n > 1 .\tag{S15}
$$

The density gradient is then

$$
\frac { d W _ { \mathrm { p e a k } } } { d \pmb { \rho } } = \sum _ { n = 1 } ^ { N } \left[ \left( \frac { \partial \pmb { R _ { n } } } { \partial \pmb { \rho } } \right) ^ { \top } \pmb { \lambda _ { n } } - \left( \frac { \partial \pmb { G _ { n } } } { \partial \pmb { \rho } } \right) ^ { \top } \pmb { \mu _ { n } } \right] .\tag{S16}
$$

The partial derivatives hold ${ \bf u } _ { n }$ and ${ \bf q } _ { n } .$ <sub>−1</sub> fixed and include the density scaling of all three parameters in Eq. (S1). Local vector–Jacobian products are obtained by diferentiating the return map once at each converged load step. Residual derivatives include quadrature weights, whereas derivatives of the pointwise history update do not. The global adjoint solves use the converged consistent tangents. This procedure diferentiates the loading history without storing the global Newton iterations. Equation (S16) is subsequently propagated through the physical-density map and the topology network.

## S4.4. Design and loading–unloading comparison

The connection has $E _ { 0 } = 1 , \nu = 0 . 3 , \sigma _ { y 0 } = 0 . 0 0 1 5 , H _ { 0 } = 0 . 0 3 E _ { 0 }$ , and $E _ { \mathrm { m i n } } = 1 0 ^ { - 6 }$ . The $2 4 \times 1 2$ domain uses a 48 × 24 background grid with 1,100 retained cells after removing the hole centered at (10.5, 6.5) with radius 2. The left edge is fixed. A downward resultant of 0.0015 acts on the right edge over $1 . 5 \leq y \leq 4 . 5$ . The material fraction is 0.40.

Both designs use 32 fixed Fourier frequencies, feature standard deviation 1, hidden width 64, and uniform initialization. The Adam schedule decreases from 0.003 to 0.0003 over a 180-update reference length, followed by a half-life of 40 and floor $1 0 ^ { - 5 }$ . The joint stopping criterion and 1,000-update cap are the same as in Section S1. The plastic design follows twelve loading increments. The elastic control uses the same elastic constants with yielding suppressed and evaluates the peak load in one step. Both minimize Eq. (S12). The plastic equilibrium tolerance is $1 0 ^ { - 9 }$ and the tangent-solve tolerance is $1 0 ^ { - 1 0 }$

After optimization, both designs undergo twelve loading and twelve unloading increments with the elastoplastic model. This comparison evaluates peak and residual port displacements and equivalent plastic strain under the same loading cycle (Table S4). The elastic-feedback design has a nominal elastic work of $1 . 9 1 6 1 \times 1 0 ^ { - 4 }$ , increasing to $3 . 0 6 4 2 \times 1 0 ^ { - 4 }$ under elastoplastic evaluation. For the plastic-feedback design, the corresponding values are 2.0280 × $1 0 ^ { - 4 }$ and $2 . 2 2 2 9 \times 1 0 ^ { - 4 }$

Table S4: Elastoplastic response of the two optimized designs under the same loading–unloading cycle. Strain measures refer to material regions $( \rho > 0 . 5 )$
<table><tr><td>Quantity</td><td>Elastic feedback</td><td>Plastic feedback</td></tr><tr><td>Peak-load work  $\left( \times 1 0 ^ { 4 } \right)$ </td><td>3.0642</td><td>2.2229</td></tr><tr><td>Peak port displacement / span (%)</td><td>0.8512</td><td>0.6175</td></tr><tr><td>Residual port displacement / span (%)</td><td>0.3187</td><td>0.0541</td></tr><tr><td>Maximum equivalent plastic strain (%)</td><td>1.3513</td><td>0.3398</td></tr><tr><td>Plastically active integration points (%)</td><td>15.63</td><td>27.82</td></tr><tr><td>Maximum material strain over cycle (%)</td><td>1.8546</td><td>0.5901</td></tr></table>

Plastic-strain images show element averages of $a _ { N }$ at the peak load, using a common color scale over material regions with $\rho > 0 . 5$ . The reported maxima are integration-point values. The plastically active fraction counts integration points satisfying $a _ { N } > 1 0 ^ { - 8 }$ among those with $\rho > 0 . 5$ . Residual displacement is the magnitude of the mean vertical displacement at the loaded port after complete unloading. All normalized displacements use the span of 24.

## S5. Execution and timing comparison

The experiments use an NVIDIA RTX 6000 Ada GPU and an AMD Ryzen Threadripper PRO 5995WX CPU. CPU mechanics uses two threads, and timed jobs run serially. HGTO and SIMP–OC timings cover problem construction, solver and network setup, optimization, and final evaluation. NTopo timings additionally include process startup and output. Plotting and ofline evaluation of saved trajectories are excluded.

Linear HGTO uses geometric multigrid preconditioning on structured meshes. The nonuniform perforated mesh uses a CPU-built smoothed-aggregation hierarchy with symmetric GPU V-cycles. The 2D geometric hierarchy is reused for up to ten design updates while the stifness-change ratio remains at most two. Conjugate-gradient iterations use the current stifness.

The nonlinear GPU backend uses FP64 tangent assembly and cuDSS 0.8 for sparse LU factorization and forward/adjoint solves. SIMP–OC and the CPU HGTO reference use the same two-thread sparse direct mechanics backend. Final nonlinear designs are evaluated independently with CPU mechanics.

Table S5: Nonlinear computation times (s). CPU/GPU denotes the HGTO mechanics backend; its density network runs on the GPU in both versions. The strong-load OC run terminates with stagnation.
<table><tr><td>Case</td><td>OC (CPU)</td><td>HGTO (CPU)</td><td>HGTO (GPU)</td><td>CPU/GPU ratio</td></tr><tr><td>Weak cantilever</td><td>854.28</td><td>286.90</td><td>64.10</td><td>4.48</td></tr><tr><td>Strong cantilever</td><td>772.24</td><td>554.13</td><td>130.45</td><td>4.25</td></tr><tr><td>Doubly fixed bridge</td><td>418.75</td><td>377.03</td><td>61.95</td><td>6.09</td></tr></table>

Table S5 compares the HGTO mechanics backends with the density network on the GPU in both versions. The objective values agree to numerical precision. GPU mechanics reduces total runtime by factors of 4.25–6.09 relative to CPU mechanics.

## S6. Optimization trajectories and visualization

Figure S1 shows the saved $2 4 0 \times 8 0$ cantilever trajectories. Snapshot compliance is evaluated with the final penalization $p = 3 .$ , including snapshots from the continuation stages. Markers show the final designs, with NTopo represented by its endpoint after the prescribed training budget.

Density plots use the optimized physical fields. Three-dimensional surfaces use $\rho = 0 . 5$ and a shared view within each comparison. Nonlinear deformations are shown at their actual scale, with common spatial limits for the weak and strong cantilevers. Load–displacement curves are obtained from the final-design evaluations.

![](images/11f96538ae6ee67157bab7b21afa54f5d225e715569353a6c5e762a03ddeea11.jpg)  
Figure S1: Cantilever optimization histories evaluated at the final material parameters. Markers indicate the final designs.