# A variational physics-informed graph neural network for heterogeneous solid mechanics

Aashay Rajan Yadav<sup>a</sup>, Amiya Prakash Das<sup>a,∗</sup>, Ratna Kumar Annabattula<sup>a,∗</sup>

<sup>a</sup>Mechanics of Materials Lab, Department of Mechanical Engineering, Indian Institute of Technology Madras, Chennai, 600036, Tamil Nadu, India

## Abstract

Stress localization in heterogeneous solids is governed by the bimaterial interface, where the displacement field remains C<sup>0</sup>-continuous, while in-plane stresses jump due to the stifness mismatch. Coordinate-based physics-informed neural networks (PINNs) represent this jump via a prescribed regularization width or a weighted interface penalty, making their accuracy sensitive to how phase-contrast changes are handled. This work presents a variational, label-free physics-informed graph neural network (PI-GNN) in which the heterogeneity is carried by the discretization rather than by the trial field. The solver operates on a conforming adaptive mesh graph, assigns constitutive behavior per element, and minimizes the discrete total potential energy as a single unweighted objective in which only first derivatives appear. The discrete energy on piecewise-linear elements coincides with the finite element (FE) Ritz functional. Dirichlet conditions are enforced by construction, with no penalty term, no interface weight, and no prescribed transition width. Using one fixed architecture, optimizer, and loss across small-strain elasticity and finite-strain Neo-Hookean hyperelasticity in two and three dimensions, the von Mises error remains below 3.58% across a stifness-contrast sweep spanning $( E _ { \mathrm { i n c } } / E _ { \mathrm { m a t } } \in [ 1 0 ^ { - 2 } , 1 0 ^ { 2 } ] )$ , where a strongform PINN degrades to 5.58%, and its displacement error reaches 7.66% against 0.49% for the PI-GNN. A trained network halves the $( \sigma _ { x x } )$ error of an energy-based PINN (5.01% versus 10.94%). Training cost exceeds a single FE solve by more than an order of magnitude, so the construction is a variationally consistent, penalty-free interface representation for parametric surrogates and inverse identification rather than a replacement for a one-of FE analysis.

Keywords: Heterogeneous materials; Neo-Hookean materials; Deep energy method; Variational methods; Physics-informed neural networks; Graph neural networks

## 1. Introduction

Material heterogeneity arising from inclusions, voids, and bonded interfaces governs stress localization and failure nucleation, so accurate prediction of localized fields is central to material design. The representational dificulty lies at the bimaterial interface: the displacement field is $C ^ { 0 } .$ -continuous, and traction is continuous, whereas in-plane stresses jump across the stifness mismatch. Smoothing that jump misrepresents the stress gradients that govern damage nucleation [Henkes et al., 2022]. The finite element method (FEM) addresses this structurally by assigning material properties to elements on a conforming mesh.

Data-driven surrogates address this problem space for constitutive learning [Kirchdoerfer and Ortiz, 2016], stress recovery [Go et al., 2025], and composite homogenization [Maurizi et al., 2022, Xia et al., 2025]. Graph neural networks (GNNs) are mesh-native: the FE mesh becomes a graph whose message-passing layers propagate information along nodal connectivity [Sanchez-Gonzalez et al., 2020, Zhao et al., 2024], enabling learnable physics engines for deformation and crack propagation [Zhou and Feng, 2024, Wang et al., 2025]. Supervised training, however, requires large FE datasets, so the labeling burden is reassigned rather than removed. Physics-informed neural networks (PINNs) eliminate labels by enforcing governing equations directly [Raissi et al., 2019, Haghighat et al., 2021, Cuomo et al., 2022, Henkes et al., 2022, Hu et al., 2024, Ren and Lyu, 2024], but carry two costs: a multi-objective loss requiring ad hoc balancing and noisy second derivatives at steep stress gradients [Bai et al., 2023]. The deep energy method (DEM) removes both by minimizing the total potential energy as a single scalar objective with only first derivatives [Samaniego et al., 2020, Nguyen-Thanh et al., 2020, Fuhg and Bouklas, 2022, Huang and Peng, 2024].

Neither the residual nor the energy formulation addresses the representational bias of the coordinate-based MLP (Multi-Layer Perceptron) used as the trial field. A single smooth network cannot reproduce a stress jump, so the contrast is regularized through a prescribed transition width [Henkes et al., 2022]. Domain decomposition assigns separate networks per phase, but compatibility and traction matching re-enter as penalty terms [Jagtap and Karniadakis, 2020, Jagtap et al., 2020, Sarma et al., 2024]. Alternative trial spaces—hp-VPINNs [Kharazmi et al., 2021], mixed-form PINNs [Rezaei et al., 2022, Ren and Lyu, 2024]—still interpolate from a global coordinate map with no information about where phases meet. Evaluating the loss on an FE discretization [Zhang et al., 2025, Xiong et al., 2025, Wu et al., 2026, Rezaei et al., 2025] lets the mesh carry geometry and material assignment, but the trial field remains a dense coordinate-to-value map in which neighboring nodes exchange nothing during the forward pass.

GNNs close this gap by moving the discretization into the architecture. When the trial field is parametrized on the mesh graph, each message-passing step updates a nodal state based on its neighbors. Hence, a stifness contrast between adjacent elements is visible during the forward pass, not only in the assembled objective. Existing energy-based graph solvers remain single-phase: Gao et al. [2022] and He et al. [2023] minimize variational or energy functionals on mesh graphs under small and finite strain, respectively; Dalton et al. [2023] scales to three-dimensional (3D) geometry with anisotropic hyperelasticity; and further applications address thermal simulation [W¨urth et al., 2024], phase-field fracture [Feng and Zhou, 2025], laminated shells [Hu et al., 2026], and elastohydrodynamic lubrication [Brumand-Poor et al., 2025]. The most direct two-phase graph work [Guevara Garban et al., 2025] trains on FE stress labels with equilibrium as a weighted penalty. Label-free two-phase solvers remain coordinate-based [Henkes et al., 2022, Sarma et al., 2024, Rezaei et al., 2022], representing the interface through a penalty or transition width. What has not been reported is a variational graph solver, trained without labels, in which material heterogeneity is carried by the discretization itself—with no interface penalty and no transition width—under a single construction covering linear elasticity and finite-strain hyperelasticity.

This paper presents a physics-informed graph neural network (PI-GNN) that operates on a conforming mesh graph and minimizes the discrete total potential energy. Constitutive behavior is assigned per element, so the bimaterial interface emerges from the stifness contrast between adjacent elements exactly as in FEM. Because the discrete energy on $\mathcal { P } _ { 1 }$ elements coincides with the FE Ritz functional, the converged FE field is the exact minimizer of the training objective; FEM is therefore the benchmark, with training and inference costs reported separately.

The main contributions are:

• Heterogeneity carried by the discretization. A single unweighted energy functional is minimized across phases with no interface penalty or transition width. Across a sweep $E _ { \mathrm { i n c } } / E _ { \mathrm { m a t } } \in [ 1 0 ^ { - 2 } , 1 0 ^ { 2 } ]$ with all settings frozen, the von Mises error stays below 3.58% and displacement error below 0.5%; a strong-form PINN with fixed transition width reaches 5.58% and 7.66% (Section 3.2.2).

• Isolation of the message-passing contribution. At a fixed training budget, message passing lowers the error by 19-43% across a mesh sweep, with the margin widening as the mesh coarsens. At convergence, both variants settle on a similar loss, proving that message passing is a faster approach to the shared Ritz minimizer (Section 3.2.3).

• Benchmarking on a shared unseen mesh. The PI-GNN roughly halves the $\sigma _ { x x }$ error of an energy-based PINN (5.01% versus 10.94%; 7.57% versus 15.08%), with errors measuring transfer to an unseen discretization (Sections 3.2.1 and 3.2.2).

• Robustness under one fixed model. All verification and application examples—smooth and re-entrant inclusions, combined loading, and 3D torsion—use identical architecture, hyperparameters, and loss, with displacement $L ^ { 2 }$ errors below 1.5% in both 2D and 3D (Sections 3 and 4).

The remainder of the paper is organized as: Section 2 sets out the kinematics, constitutive relations, and variational GNN formulation. Section 3 reports verification and isolates the message-passing efect. Section 4 applies the framework to re-entrant inclusions, and 3D torsion; Section 5 closes with findings and future directions.

## 2. Preliminaries: Kinematics and PI-GNN Formulation

This section presents the theoretical framework and formulation of the PI-GNN solver. We first describe the governing equations and constitutive relations, followed by the variational principle of minimum potential energy, which serves as the physical basis for the solver. Finally, we describe the graph-based discretization, neural network architecture, and training scheme in Section 2.2.

## 2.1. Governing equations and material models

The body occupies a domain Ω with boundary $\partial \Omega = \Gamma _ { u } \cup \Gamma _ { t }$ , along which the displacement $\left( \Gamma _ { u } \right)$ and traction $\left( \Gamma _ { t } \right)$ boundary conditions are prescribed, respectively. Given the displacement field u, the kinematic quantities are:

$$
\mathbb { F } = \mathbb { I } + \nabla \mathbf { u } , \quad J = \operatorname* { d e t } \mathbb { F } , \quad \boldsymbol { \varepsilon } = \frac { 1 } { 2 } \Big ( \nabla \mathbf { u } + \nabla \mathbf { u } ^ { \top } \Big ) , \quad \mathbb { C } = \mathbb { F } ^ { \top } \mathbb { F } , \quad I _ { 1 } = \operatorname { t r } ( \mathbb { C } ) ,\tag{1}
$$

where $\mathbb { F }$ and $\mathbb { C }$ are the deformation gradient and right Cauchy-Green tensor, respectively. For a linear elastic (LE) material under the small-strain assumption, the strain-energy density Ψ reads:

$$
\Psi _ { \mathrm { L E } } ( \varepsilon ) = { \frac { \lambda } { 2 } } ( \operatorname { t r } \varepsilon ) ^ { 2 } + \mu ( \varepsilon { : } \varepsilon ) ,\tag{2}
$$

whereas for a finite strain hyperelastic material, we use the Neo-Hookean (NH) material model,

$$
\Psi _ { \mathrm { N H } } ( \mathbb { F } ) = \frac { \mu } { 2 } ( I _ { 1 } - 3 ) - \mu \ln J + \frac { \lambda } { 2 } ( \ln J ) ^ { 2 } .\tag{3}
$$

The Lam´e parameters related to the Young’s modulus (E) and Poisson’s ratio (ν) as

$$
\mu = \frac E { 2 ( 1 + \nu ) } , \qquad \lambda = \frac { E \nu } { ( 1 + \nu ) ( 1 - 2 \nu ) } ,\tag{4}
$$

governs the 3D and plane strain problems. The plane stress condition eliminates the outof-plane stress. At finite strain, this is imposed through $\partial \Psi _ { \mathrm { N H } } / \partial \mathbb { F } _ { 3 3 } = 0$ . And at small strain, the corresponding energy uses the reduced Lam´e parameter $\bar { \lambda } = E \nu / ( 1 - \nu ^ { 2 } )$ Both Equations (2) and (3) are used, but a given boundary value problem (BVP) employs one constitutive family throughout. The linear elastic energy for the small-strain examples of Section 3, and the Neo-Hookean energy for both phases in the finite-strain examples of Section 4. Within each family, the material parameters are piecewise constant, $( E _ { \mathrm { m a t } } , \nu _ { \mathrm { m a t } } ) \in \Omega _ { \mathrm { m a t } }$ and $( E _ { \mathrm { i n c } } , \nu _ { \mathrm { i n c } } ) \in \Omega _ { \mathrm { i n c } }$ . The total potential energy is given as:

$$
\Pi [ { \mathbf { u } } ] = \int _ { \Omega } \Psi \mathrm { d } \Omega - \int _ { \Omega } { \mathbf { b } } \cdot { \mathbf { u } } \mathrm { d } \Omega - \int _ { \Gamma _ { t } } { \mathbf { t } } \cdot { \mathbf { u } } \mathrm { d } \Gamma ,\tag{5}
$$

with body force b and prescribed traction t on the boundary $\Gamma _ { t }$

The principle of minimum potential energy states that the equilibrium displacement is the admissible field that minimizes Π (Equation (5)):

$$
\mathbf { u } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { u } \in \mathcal { U } } \Pi [ \mathbf { u } ] , \qquad \mathcal { U } = \{ \mathbf { u } : \mathbf { u } = \mathbf { u } _ { D } \mathrm { o n } \Gamma _ { u } \} ,\tag{6}
$$

since the stationarity condition $\delta \Pi = 0$ is the weak form of $\nabla \cdot { \pmb \sigma } + { \pmb b } = { \pmb 0 }$ together with the natural boundary condition ${ \pmb \sigma } \cdot { \bf n } = { \bf t }$ on $\Gamma _ { t }$ . The DEM framework proposed by Samaniego et al. [2020] uses Equation (6) directly as a training objective. The displacement is approximated by the output u<sub>θ</sub> of a neural network with parameters θ. The functional is reduced to an ordinary function $\Pi ( \pmb \theta ) = \Pi [ \mathbf { u } _ { \theta } ]$ minimized over θ with no reference solution required. The essential condition $\mathbf { u } _ { \theta } = \mathbf { u } _ { D }$ on $\Gamma _ { u }$ is not satisfied automatically by the network and is enforced either by construction or through a penalty term. In contrast, the traction condition on $\Gamma _ { t }$ is recovered naturally by the energy minimization.

## 2.2. PI-GNN solver formulation

The PI-GNN workflow is designed to solve BVPs on complex, heterogeneous domains without requiring labeled training data. Figure 1 illustrates the overall computational pipeline, which proceeds in a structured sequence: given a physical domain, we first generate an adaptive mesh and construct a graph representation $\mathcal { G }$ (Section 2.2.1). We then pass spatial coordinates and material labels through a message-passing GNN to compute node-wise displacements (Section 2.2.2). Using the displacements, we compute element-wise strains and evaluate the constitutive relations to assemble the discrete potential energy Π (Section 2.2.3). Finally, the GNN parameters θ are optimized by minimizing the energy.

## 2.2.1. Adaptive mesh generation and graph construction

The PI-GNN operates on a graph $\mathcal { G } = ( \nu , \mathcal { E } )$ built from a spatially adaptive mesh of collocation points, refined where the fields are discontinuous—at the outer boundary or at material interfaces—coarsened through the quiescent bulk. Mesh generation proceeds in three stages: an adaptive size field determines the local node spacing, collocation points are placed at that spacing, and a Delaunay triangulation turns the points into a mesh graph with per-element material labels. A single geometric primitive, the signed distance $\phi ( \mathbf { x } )$ 2 which is negative inside the inclusion, positive in the matrix, and zero on the interface, is used. The point-in-phase tests, boundary sampling, and the size field are all derived from $\phi ,$ such that the same pipeline is used for any circular (or re-entrant) inclusion and reduces to null when the inclusion interior is left unfilled.

Adaptive size field: The field $\rho ( \mathbf { x } )$ defines the local nodal spacing, with finer resolution prescribed geometric features and progressively coarser resolution away from them. For a feature k of characteristic length $\ell _ { k }$ , let $d _ { k } = | \phi _ { k } ( \mathbf { x } ) |$ denote the distance from x to its boundary. The spacing is prescribed as:

$$
\rho _ { k } ( { \bf x } ) = \operatorname* { m i n } ( \rho _ { \mathrm { m i n } , k } + g d _ { k } , \ \rho _ { \mathrm { m a x } } ) ,\tag{7}
$$

where $\rho _ { \mathrm { m i n } , k }$ is proportional to $\ell _ { k }$ specifies the minimum spacing at the feature boundary, g controls the rate of the coarsening, and $\rho _ { \mathrm { m a x } }$ denotes the far field spacing. The distancebased grading provides a gradual transition from fine to coarse resolution while avoiding an abrupt change over a prescribed refinement width. In contrast to a fixed-width refinement band, the transition width adapts to the prescribed spacing contrast and continues until $\rho _ { \mathrm { m a x } }$ is reached. The parameter $g$ can, therefore, be selected to control the sharpness of the resolution transition or determined from the desired near-field refinement extent. When multiple features are present, the most restrictive local spacing is retained,

![](images/2df58f953e9746fe45555ad0716b011b9248b6430d8e137504aa4c3bea486e81.jpg)  
Figure 1: Overview of the PI-GNN solver. The mesh is a graph: each node $i ~ \in ~ \nu$ carries the feature vector $\mathbf { f } _ { i } = [ \hat { X } _ { i } , \hat { Y } _ { i } , m _ { i } , \chi _ { i } ] ^ { \mathsf { T } }$ (Equation (9)), which the encoder lifts to a 64-dimensional embedding. The $\mathrm { ~ L ~ } = ~ 4$ message-passing layers (see Figure 3) exchange embeddings along mesh edges. Element strains $\varepsilon _ { e }$ and stresses $\pmb { \sigma } _ { e }$ (linear elastic formulation shown here for simplicity) are assembled over the element set $\tau$ into the potential energy $\Pi = \Pi _ { \mathrm { i n t } } - \Pi _ { \mathrm { e x t } }$ . The Π is minimized over $\pmb \theta$ using an Adam optimizer with no labeled data.

$$
\rho ( \mathbf x ) = \operatorname* { m i n } _ { k } \rho _ { k } ( \mathbf x ) ,\tag{8}
$$

ensuring that each feature remains adequately resolved, including in regions where neighboring holes or inclusions interact. A representative spacing field is shown in Figure 2.

Collocation point assignment: The geometric boundaries are discretized before the inner domains are filled. Nodes are placed along each external boundary and inclusion interface according to the local spacing ρ, with their positions determined by arc length. This process provides an approximately uniform distribution along smooth boundary segments while increasing the nodal density in regions of high curvatures, including re-entrant tips. The boundary and interface nodes are then fixed and treated as an exclusion region. The matrix and inclusion domains are independently filled using variable-radius Poisson-disc sampling. Pre-seeding the boundaries allows the interior sampling to automatically obtain the required clearance from all geometric interfaces. For each region, the procedure is as follows:

1. Seeding. The fixed boundary and interface nodes are inserted into a background search grid, and the region is seeded with one accepted interior point; each inclusion is seeded separately such that none is left empty.

2. Candidate generation. For a random active point, candidate points are drawn in the annulus region between one and two local spacings, with random orientation.

![](images/d7441354e7131f255bf18196f326d8b78a8c4e587e710a5da46533f24e07a8b7.jpg)  
Figure 2: Adaptive collocation size field $\rho ( \mathbf { x } )$ : finest at the interface boundary $( \rho _ { \mathrm { m i n } } )$ and grading outward to the uniform far-field spacing $\left( \rho _ { \mathrm { m a x } } \right)$

3. Acceptance. A candidate point is accepted if it lies within the intended phase and is suficiently separated from every existing node. The required separation is taken as the smaller of the local spacing at the candidate point and at the existing node. If this condition is satisfied for all existing nodes, the candidate point is added to the active set.

4. Termination. An active point is considered retired once it yields no admissible candidates, and sampling ceases when the active set is empty, indicating that the region has reached the specified density.

Triangulation and material graph: The complete point set, including the outer boundary, interfaces, and interiors, is triangulated using the Delaunay criterion [Virtanen et al., 2020]. The resulting nodes form the graph vertices $\nu ,$ while pairs of nodes sharing a triangle edge are connected by directed edges in $\mathcal { E } .$ . Each edge carries a fixed geometric attribute based on the relative positions of its endpoints, $[ \Delta x _ { i j } , \Delta y _ { i j } , \| \Delta \mathbf { x } _ { i j } \| ] ^ { \mathsf { T } }$ in 2D, with the same definition extended to $d = 3$ . These attributes are computed once from the mesh and are not trained.

Each node is assigned a phase label; inclusion if $\phi _ { k } \leq 0$ for any feature k, including nodes on the feature boundary, and matrix otherwise. Elements are then classified according to the phases of their vertices as all-matrix, all-inclusion, or interface elements when their vertices span both labels. Because the mesh conforms to the phase interfaces, an interface element does not physically contain both materials; rather, the mixed vertex labels arise from the convention used to assign interface nodes. Such interface elements are assigned the matrix properties, consistent with the underlying conforming FE discretization and the prescribed nodal phase convention, rather than as an approximation of the material interface.

For a perforated plate, the same construction is used with the hole left unfilled and the remaining domain treated as a single matrix phase. The hole boundary is refined to the same field size.

## 2.2.2. Graph neural network architecture

The GNN solver operates on the graph representation $\mathcal { G }$ using an Encoder-Processor-Decoder architecture [Dalton et al., 2022, 2023]. The encoder maps the nodal features and spatial coordinates to latent representations, which are subsequently updated through local message passing to capture neighborhood interactions. The decoder then maps the processed latent representations to nodal displacements, with the prescribed boundary conditions enforced exactly.

1. Encoder: Each node $i \in \mathcal V$ is assigned a normalized feature vector:

$$
\mathbf { f } _ { i } = \left[ \hat { X } _ { i } , ~ \hat { Y } _ { i } , ~ m _ { i } , ~ \chi _ { i } \right] ^ { \mathsf { T } } \in \mathbb { R } ^ { 4 } ,\tag{9}
$$

where $\hat { X _ { i } } , \hat { Y _ { i } }$ are normalized spatial coordinates, $m _ { i } \in \{ 0 , 1 \}$ is the material-phase label, and $\chi _ { i } \in \{ 0 , 1 \}$ indicates whether the node i lies on the matrix–inclusion interface. For the 2D formulation shown here; the node feature vector is $[ \hat { \mathbf { X } } _ { i } , \ m _ { i } , \ \chi _ { i } ] ^ { \mathsf { T } } \in \mathbb { R } ^ { d + 2 }$ where $d = 2 ;$ ; the same construction extends directly to 3D with $( d = 3 )$ . The decoder subsequently outputs the displacement vector $\tilde { \mathbf { u } } _ { i } \in \mathbb { R } ^ { d }$ . A shared linear layer maps the node features to a 64-dimensional latent space:

$$
{ \bf h } _ { i } ^ { ( 0 ) } = \mathrm { R e L U } ( { \bf W } _ { \mathrm { e n c } } { \bf f } _ { i } + { \bf b } _ { \mathrm { e n c } } ) ,\tag{10}
$$

where $\mathbf { W } _ { \mathrm { e n c } }$ and $\mathbf { b } _ { \mathrm { e n c } }$ are trainable parameters, and ReLU( ) is the rectified linear unit.

2. Processor: Information is propagated across the mesh using $\mathrm { ~ L ~ } = ~ 4$ message-passing layers (see Figure 3). At each layer $\ell = 1 , \ldots , \mathtt { L } .$ , node embeddings $\mathbf { h } _ { i } ^ { ( \ell - 1 ) }$ are updated via localized messages from incoming neighbors $j \in \mathcal { N } ( i )$ :

$$
\begin{array} { r } { \mathbf { m } _ { j  i } ^ { ( \ell ) } = \mathrm { R e L U } \Big ( \mathbf { W } _ { \mathrm { m s g } } ^ { ( \ell ) } \big [ \mathbf { h } _ { i } ^ { ( \ell - 1 ) } \ \lVert \ \mathbf { h } _ { j } ^ { ( \ell - 1 ) } - \mathbf { h } _ { i } ^ { ( \ell - 1 ) } \ \rVert \ \mathbf { e } _ { i j } \big ] + \mathbf { b } _ { \mathrm { m s g } } ^ { ( \ell ) } \Big ) , } \end{array}\tag{11}
$$

$$
\begin{array} { r } { \mathbf { h } _ { i } ^ { ( \ell ) } = \mathrm { R e L U } \left( \mathbf { W } _ { \mathrm { u p d } } ^ { ( \ell ) } \big [ \mathbf { h } _ { i } ^ { ( \ell - 1 ) } \ \big \lVert \ \mathbf { m } _ { i } ^ { ( \ell ) } \big ] + \mathbf { b } _ { \mathrm { u p d } } ^ { ( \ell ) } \right) , } \end{array}\tag{12}
$$

where $\begin{array} { r } { \mathbf { m } _ { i } ^ { ( \ell ) } = \frac { 1 } { \vert \mathcal { N } ( i ) \vert } \sum _ { j \in \mathcal { N } ( i ) } \mathbf { m } _ { j  i } ^ { ( \ell ) } } \end{array}$ is the mean-aggregated message, represents vector concatenation, and $\mathbf { W } ^ { ( \ell ) } , \mathbf { b } ^ { ( \ell ) }$ are layer-specific trainable weights and biases. The message in Equation (11) combines three quantities: the receiver state $\mathbf { h } _ { i } ^ { ( \ell - 1 ) }$ , the neighbor–receiver diference $\mathbf { h } _ { j } ^ { ( \ell - 1 ) } - \mathbf { h } _ { i } ^ { ( \ell - 1 ) }$ , and the edge geometry ${ \bf e } _ { i j }$ . The state diference emphasizes local contrast between neighboring nodes, which is particularly relevant near bimaterial interfaces where material properties change across adjacent elements. The edge attribute provides the relative direction and length of the connection, allowing the same message function to operate on edges with diferent geometries in an unstructured mesh. Because these edge features are computed directly from the mesh, the same learned message function can also be applied when the network is evaluated on an unseen discretization. Figure 3 illustrates the message construction, aggregation, and node update for a single layer ℓ.

3. Decoder and Boundary Enforcement: The final latent embeddings ${ \bf h } _ { i } ^ { \left( \mathrm { L } \right) }$ are decoded to unconstrained nodal displacements via a linear layer:

$$
\tilde { \mathbf { u } } _ { i } = \mathbf { W } _ { \mathrm { d e c } } \mathbf { h } _ { i } ^ { \mathrm { ( L ) } } + \mathbf { b } _ { \mathrm { d e c } } , \quad \tilde { \mathbf { u } } _ { i } \in \mathbb { R } ^ { 2 } ,\tag{13}
$$

where $\mathbf { W } _ { \mathrm { d e c } }$ and $\mathbf { b } _ { \mathrm { d e c } }$ are trainable decoder parameters. Dirichlet boundary conditions are enforced exactly by applying a multiplicative binary mask to the unconstrained predictions:

$$
\mathbf { u } _ { i } = \mathbf { q } _ { i } \odot \tilde { \mathbf { u } } _ { i } + ( \mathbf { 1 } - \mathbf { q } _ { i } ) \odot \mathbf { u } _ { D , i } ,\tag{14}
$$

where $\mathbf { q } _ { i } \in \{ 0 , 1 \} ^ { 2 }$ is the boundary mask indicating free (1) or prescribed (0) displacement components, ${ \bf u } _ { D , i }$ is the prescribed Dirichlet value, and $\odot$ denotes element-wise multiplication. This hard enforcement guarantees exact constraint satisfaction at every epoch without penalty terms in the loss.

![](images/4f44d158975885cdb1c4c313f15126d76a25197adbca974eeb2cd94500dedeff.jpg)  
Figure 3: Message-passing layer (ℓ) of the processor (L = 4 layers, Equations (11) and (12)). On each edge $( j  i )$ the receiver embedding, the neighbor-minus-receiver difference, and the edge attribute are concatenated, $( [ \mathbf { h } _ { i } ^ { ( \ell - \bar { 1 } ) } \ \lVert \ \mathbf { h } _ { j } ^ { ( \ell - \bar { 1 } ) } - \mathbf { h } _ { i } ^ { ( \ell - 1 ) } \ \rVert \ \mathbf { e } _ { i j } ] \in \mathbb { R } ^ { 1 3 1 } )$ 2 and mapped by shared weights $( \mathbf { W } _ { \mathrm { m s g } } ^ { ( \ell ) } )$ to $( \mathbf { m } _ { j  i } ^ { ( \ell ) } \in \mathbb { R } ^ { 6 4 } )$ ; the four boxes are one weight set reused edge-wise. The neighborhood mean $( \mathbf { m } _ { i } ^ { ( \ell ) } )$ is concatenated with $( \mathbf { h } _ { i } ^ { ( \ell - 1 ) } )$ and mapped by $( \mathbf { W } _ { \mathrm { u p d } } ^ { ( \ell ) } )$ to $( \mathbf { h } _ { i } ^ { ( \ell ) } )$

## 2.2.3. Discrete potential energy assembly

Using the conforming mesh, the continuous total potential energy functional is discretized over the computational domain. With continuous piecewise-linear $\left( \mathcal { P } _ { 1 } \right)$ elements the deformation gradient $\mathbb { F }$ and strain ε are constant within each constant-strain triangles (CSTs) in 2D and constant-strain tetrahedra in 3D. For piecewise-constant material properties, the internal-energy density is, therefore, constant within each element, making single-point quadrature suficient and exact for the internal energy. The domain integrals consequently reduce to algebraic sums over elements and boundary facets. Assuming no body forces, the discretized potential energy $\Pi ( \mathbf { u } _ { \theta } )$ is assembled as:

$$
\Pi ( { \mathbf { u } _ { \theta } } ) = \sum _ { e = 1 } ^ { N _ { e } } \Psi _ { e } V _ { e } - \sum _ { f = 1 } ^ { N _ { f } } \mathbf { t } _ { f } \cdot { \mathbf { u } ^ { f } } S _ { f } ,\tag{15}
$$

where $V _ { e }$ is the element e area in 2D (or volume in 3D), $\Psi _ { e }$ is the strain-energy density calculated from the predicted nodal displacements $\mathbf { u } _ { \theta }$ (using the linear elastic/hyperelastic model). $S _ { f }$ denotes the edge length in 2D (or facet area in 3D) of facet $f$ on the traction boundary $\Gamma _ { t } .$ , and $\mathbf { t } _ { f }$ denotes the prescribed traction vector on that facet. For spatially varying traction, $\mathbf { t } _ { f }$ is evaluated as the facet-average traction. Similarly, $\mathbf { u } ^ { f }$ denotes the predicted displacement averaged over the nodes of the facet. For a facet-wise constant traction and $\mathcal { P } _ { 1 }$ interpolation, single-point quadrature on each facet computes the external work exactly. Under the uniaxial tension condition shown in Figure 4, the contribution reduces to $t u _ { x } ^ { f } S _ { f }$ , where $u _ { x } ^ { f }$ is the mean longitudinal displacement of the facet nodes. For linear elastic $\mathcal { P } _ { 1 }$ elements with global displacement vector u, the discrete internal strain energy evaluates to $\begin{array} { r } { \sum _ { e = 1 } ^ { N _ { e } } \Psi _ { e } V _ { e } = \frac { 1 } { 2 } \mathbf { u } ^ { \top } \mathbf { K } \mathbf { u } } \end{array}$ , where K is the standard assembled FE stifness matrix, and the external work evaluates to $\mathbf { F } _ { \mathrm { e x t } } ^ { \mathsf { T } } \mathbf { u } .$ . The discrete functional $\begin{array} { r } { \Pi ( \mathbf { u } _ { \theta } ) = \frac { 1 } { 2 } \mathbf { u } ^ { \mathsf { T } } \mathbf { K } \mathbf { u } - \mathbf { F } _ { \mathrm { e x t } } ^ { \mathsf { T } } . } \end{array}$ u therefore coincides identically with the FE Ritz functional, and the converged FE displacement field is the exact minimizer of the discrete training loss. The network parameters θ are then optimized by minimizing this discrete potential energy as the sole training objective (Equation (16)). No PDE-residual term or boundary penalty terms are required; the essential boundary conditions are enforced hard-coded in the output layer.

$$
\mathcal { L } ( \pmb { \theta } ) = \Pi \big ( \mathbf { u } _ { \theta } \big ) .\tag{16}
$$

## 3. Numerical Verification and Reference Solver

This section details the verification suite and benchmarks the PI-GNN framework against numerical and analytical references. We first describe the standard displacement-based FE reference model, which serves as the verification baseline solver (Section 3.1). We then define the error metrics and present three verification problems (Section 3.2).

## 3.1. Finite element reference model

The predictions of the PI-GNN are benchmarked against a standard displacement-based FE model implemented in FEniCSx [Baratta et al., 2023]. The solvers share the same mesh, per-phase material assignment, and boundary data, and use identical Lam´e parameters: the shear modulus $\mu$ throughout, with the dilatational constant taken as $\bar { \lambda }$ for the plane stress plates and as $\lambda$ for the 3D example. The comparison, therefore, isolates the numerical approximation introduced by the neural network from any modeling discrepancy. Converged to solver tolerance, the FE field serves as the high-fidelity benchmark throughout. Each example shares the equilibrium problem

$$
\nabla \cdot { \pmb \sigma } = { \bf 0 } \quad \mathrm { i n } \ \Omega ,\tag{17}
$$

closed by the kinematics and per-phase constitutive laws and the boundary condition $\mathbf { u } = \bar { \mathbf { u } }$ on $\Gamma _ { u }$ and ${ \pmb \sigma } { \cdot } { \bf n } = { \bf t }$ on $\Gamma _ { t }$ . The discontinuous moduli $( E , \nu ) ( \mathbf { x } )$ or the Lam´e parameters enter the element integrals directly. Because the mesh conforms to $\Gamma _ { \mathrm { i n t } }$ , the perfect-bond interface conditions are satisfied automatically, without penalty or Lagrange-multiplier terms. Figure 4 shows the representative configuration used in the verification suite; a plate under uniaxial tension, for which the prescribed boundary conditions are:

$$
\begin{array} { c c c } { { \boldsymbol { u } _ { x } = 0 } } & { { \mathrm { o n ~ t h e ~ l e f t ~ e d g e , } } } & { { \mathbf { u } = \mathbf { 0 } } } & { { \mathrm { a t ~ p o i n t ~ } ( 0 , L ) , } } \\ { { \boldsymbol { \sigma } \cdot \mathbf { n } = ( T , 0 ) ^ { \top } } } & { { \mathrm { o n ~ t h e ~ r i g h t ~ e d g e . } } } & { { } } \end{array}\tag{18}
$$

The corner constraint eliminates the remaining rigid-body mode, while all other edges are traction-free.

For the linear elastic example, the displacement is interpolated with continuous piecewiselinear $\left( \mathcal { P } _ { 1 } \right)$ Lagrange elements, and the weak form reads

$$
\int _ { \Omega } { \pmb \sigma } ( { \mathbf { u } } _ { h } ) : \pmb { \varepsilon } ( { \mathbf { v } } ) \mathrm { d } \Omega = \int _ { \Gamma _ { t } } { \mathbf { t } } \cdot { \mathbf { v } } \mathrm { d } \Gamma ,\tag{19}
$$

where $\pmb { \sigma } ( \mathbf { u } _ { h } )$ is the stress conjugate to $\Psi _ { \mathrm { L E } }$ . The spatially varying $\mu ( \mathbf { x } )$ and $\bar { \lambda } ( { \bf x } )$ enter the bilinear form directly, requiring no modification of the formulation for homogeneous or composite domains. The resulting linear system is solved using a direct LU factorization with MUMPS [Amestoy et al., 2001]. Element stresses are subsequently recovered from $\mathbf { u } _ { h }$ using the same constitutive relation.

![](images/30288245aa40e083ae538234a7da8825bd853a1c3cafb85a3163642970b71895.jpg)  
Figure 4: Boundary conditions of the representative uniaxial tension configuration, applied identically in the FE reference model and in the PI-GNN.

For the hyperelastic example, the mesh, element type, and boundary treatment are retained, while the constitutive model is replaced by the Neo-Hookean energy $\Psi _ { \mathrm { N H } }$ , rendering the problem nonlinear. A total-Lagrangian formulation is adopted with respect to the reference configuration. The discrete equilibrium is obtained from the stationarity of the total potential energy, whose first variation defines the residual:

$$
\mathcal { R } ( \mathbf { u } _ { h } ; \mathbf { v } ) = \int _ { \Omega } \mathbb { P } : \nabla \mathbf { v } \mathrm { d } \Omega - \int _ { \Gamma _ { t } } \mathbf { t } \cdot \mathbf { v } \mathrm { d } \Gamma = 0 \qquad \forall \mathbf { v } \in \mathcal { V } _ { 0 } ,\tag{20}
$$

where $\mathbb { P } = { \partial \Psi _ { \mathrm { N H } } } / { \partial \mathbb { F } }$ is the first Piola-Kirchhof stress and $\mathcal { V } _ { 0 }$ is the space of admissible variations vanishing on $\Gamma _ { u }$ . The nonlinear system is solved using Newton-Raphson iteration with the consistent tangent ${ \partial \mathcal { R } } / { \partial { \bf u } _ { h } }$ assembled by automatic diferentiation of the residual. The prescribed nominal traction t on the reference boundary is applied incrementally in equal load steps to facilitate convergence at finite strain, with the direct LU solver reused at each iteration. The Cauchy stress used for comparison is subsequently recovered by push-forward,

$$
\begin{array} { r } { \pmb { \sigma } = { J } ^ { - 1 } \mathbb { P } \mathbb { F } ^ { \top } , \qquad \mathbb { P } = \mathbb { F } \mathbb { S } , \qquad \mathbb { S } = \mu \big ( \mathbb { I } - \mathbb { C } ^ { - 1 } \big ) + \lambda _ { 3 \mathrm { D } } \ln { J } \mathbb { C } ^ { - 1 } , } \end{array}\tag{21}
$$

where S denotes the second Piola-Kirchhof stress.

In linear elastic or hyperelastic regimes, the FE reference solution and the PI-GNN use the same mesh, per-element material assignment, and essential and natural boundary conditions. The comparison, therefore, isolates the discrepancy between the converged FE solution and its neural approximation without introducing discrepancies due to geometry, discretization, material assignment, or loading.

## 3.2. Verification problems and error metrics

The framework is verified across two problem classes: the first investigates perforated plates with single- and three-hole configurations that introduce stress concentrations. The second investigates a material interface with a stifness mismatch. These numerical examples probe the recovery of smooth elastic fields, sharp stress gradients, and heterogeneous response across an interface.

Where a closed-form solution is available, it is used as the reference; otherwise, the neural predictions are compared with the converged FE solution evaluated on the same mesh. This common discretization removes diferences in mesh resolution and geometry representation from the comparison, such that the reported error isolates the neural approximation relative to the FE reference. Body forces are neglected throughout both the FE and neural network formulations, with $\mathbf b = \mathbf 0$ in Equation (5).

Let α denote a scalar field such as a displacement component $u _ { x } , u _ { y }$ , a stress component $\sigma _ { x x } , \sigma _ { y y } , \sigma _ { x y } ,$ or the von Mises stress $\sigma _ { \mathrm { v M } }$ . Let $\alpha _ { \mathrm { F E M } }$ and $\alpha _ { \mathrm { P I - G N N } }$ denote the FE reference and PI-GNN predictions, respectively, evaluated at the same nodes. Accuracy is quantified using three complementary measures: the global relative $L ^ { 2 }$ error, the pointwise relative error $\mathrm { e r r } _ { \alpha } .$ , and the coeficient of determination $R ^ { 2 }$ . The global agreement between the two solutions is quantified using an element-area-weighted relative $L ^ { 2 }$ error for each field:

$$
L _ { \alpha } ^ { 2 } = \frac { \sqrt { \displaystyle \sum _ { e } A _ { e } \left( \alpha _ { \mathrm { P I - G N N } } ^ { e } - \alpha _ { \mathrm { F E M } } ^ { e } \right) ^ { 2 } } } { \displaystyle \sqrt { \displaystyle \sum _ { e } A _ { e } \left( \alpha _ { \mathrm { F E M } } ^ { e } \right) ^ { 2 } } } \times 1 0 0 \% ,\tag{22}
$$

where $A _ { e }$ is the area of element e and $\alpha ^ { e }$ the field value on that element, so that the metric weights each region by its geometric measure and is independent of mesh non-uniformity. The signed pointwise relative error at node i is evaluated as:

$$
\mathrm { e r r } _ { \alpha } ( \mathbf { x } _ { i } ) = \frac { \alpha _ { \mathrm { P I - G N N } } ( \mathbf { x } _ { i } ) - \alpha _ { \mathrm { F E M } } ( \mathbf { x } _ { i } ) } { | \alpha _ { \mathrm { F E M } } ( \mathbf { x } _ { i } ) | } \times 1 0 0 \% .\tag{23}
$$

The sign of $\mathrm { e r r } _ { \alpha }$ distinguishes over-prediction (+) from under-prediction ( ). To avoid the artificial amplification of the relative error in Equation (23) near zero-crossings and constrained boundaries, nodes satisfying $| \alpha _ { \mathrm { F E M } } ( \mathbf { x } _ { i } ) | ~ < ~ \delta$ max $_ j | \alpha _ { \mathrm { F E M } } ( \mathbf { x } _ { j } ) |$ are excluded, with a relative threshold of $\delta = 1 0 ^ { - 2 }$ . The spatial distribution of $\mathrm { e r r } _ { \alpha }$ is summarized by its median, which indicates systematic bias toward over- or under-prediction while remaining less sensitive to isolated outliers. The coeficient of determination measures the reference field variance captured by the prediction,

$$
R _ { \alpha } ^ { 2 } = 1 - \frac { \displaystyle \sum _ { i } \bigl ( \alpha _ { \mathrm { F E M } } ( \mathbf { x } _ { i } ) - \alpha _ { \mathrm { P I - G N N } } ( \mathbf { x } _ { i } ) \bigr ) ^ { 2 } } { \displaystyle \sum _ { i } \bigl ( \alpha _ { \mathrm { F E M } } ( \mathbf { x } _ { i } ) - \overline { { \alpha } } _ { \mathrm { F E M } } \bigr ) ^ { 2 } } ,\tag{24}
$$

where $\overline { { \alpha } } \mathrm { { F E M } }$ is the mean of the reference field. For the heterogeneous examples, $R _ { \alpha } ^ { 2 }$ is additionally evaluated separately in near-field and far-field regions. A node is classified as near-field if its distance to the boundary of any inclusion is less than or equal to twice the corresponding inclusion radius; all remaining nodes are classified as far-field. This partition separates the accuracy in regions containing steep gradients near material interfaces from that in the comparatively smooth bulk.

## 3.2.1. Single and multiple holes

The first verification example considers square plates with circular holes, providing a benchmark for localized stress concentrations. Three configurations of increasing geometric complexity are used to assess the PI-GNN. The smallest single-hole example first verifies the PI-GNN and the FE reference against the closed-form Kirsch solution, for which the finite plate approximates the infinite-plate idealization well $( a = 1 \mathrm { m m }$ , plate-to-diameter ratio of 10). The two remaining configurations, consisting of a larger single hole $( r = 4 \mathrm { m m } )$ and an asymmetric three-hole plate, additionally provide a comparison with the energy-based PINN [Li et al., 2021]. For neural comparisons, the solvers are trained using independent samples from the domain and subsequently evaluated, without further optimization, with a single forward pass on a common FE mesh that is not used during training. The converged FE solution on this mesh serves as the reference.

The PI-GNN versus PINN comparison is designed to isolate the efect of the displacementfield representation as far as possible. Both models minimize the same functional, Π = $U - W _ { \mathrm { e x t } }$ , using Adam with a constant learning rate of $\eta = 1 \times 1 0 ^ { - 3 }$ and a fixed budget of 30,000 epochs, without problem-specific tuning. The PINN contains approximately 21,000 trainable parameters, compared with approximately 17,000 for the PI-GNN L4H32, providing comparable network capacities and avoiding an advantage to the PI-GNN from a larger parameter count. The PI-GNN represents the displacement as a discrete nodal field and evaluates the energy element-wise on the mesh. In contrast, the PINN represents a continuous coordinate-based field and evaluates the energy by Monte Carlo integration over collocation points [Li et al., 2021]. Thus, both methods are compared under the same variational objective and optimization budget, while their spatial representations and energy discretizations remain distinct.

Analytical benchmark: The plate occupies $\Omega = [ - l / 2 , l / 2 ] ^ { 2 } \setminus \mathcal { B } _ { a } ( { \bf { 0 } } )$ with side length $l =$ 20 mm and hole radius $a = 1 \mathrm { m m }$ . The resulting plate-to-diameter ratio, $l / 2 a = 1 0$ , provides a finite approximation to the infinite-plate assumption underlying the Kirsch solution. The material is linear elastic steel $( E = 2 1 0 \mathrm { G P a } , \nu = 0 . 3 )$ under plane stress. A uniform tensile traction $T = 1 \mathrm { M P a }$ is applied on the right edge, with the Dirichlet conditions on the left edge (see Figure 4). The near-field region is defined as the annulus extending from the hole boundary to an outer radius 2a.

Figure 5a compares the longitudinal stress $\sigma _ { x x }$ of the FE reference solution (left) and the PI-GNN prediction (right). The PI-GNN reproduces the characteristic Kirsch distribution; $\sigma _ { x x }$ reaches $\approx 3 T$ at the hole crown $( \theta = \pm \pi / 2 )$ At the horizontal margins $( \theta = 0 , \pi )$ the pressure becomes compressive with a value $\approx - T$ , and approaches the far-field value $T$ away from the hole. Figure 5b traces $\sigma _ { x x }$ along the transverse line of symmetry $x = 0$ . The PI-GNN closely follows both the FE reference and the analytical Kirsch solution, recovering the maximum stress-concentration factor of 3 at the hole boundary and its decay toward the far-field value. The field-wise metrics in Table 1 shows good agreement, with the loadcarrying fields $u _ { x }$ and $\sigma _ { x x }$ achieving $R ^ { 2 } = 0 . 9 9$ both globally and in the near-field. The mesh convergence study for this case is shown in Section Appendix A.

![](images/90fbb6f674a362e618edd9d89ed4e99b618e5e039ac7b929f31235d8645bbf3b.jpg)

![](images/60d008b33d3d6e1be7327d9deb454761775a8a8e1749a6223e7cfbc1d0e4dc61.jpg)

(b)  
![](images/b2340cd76cf4690e0c5a0c6a6dcd26ab5102211cfa0ce77751733d90bf69b2f5.jpg)  
Figure 5: Thin plate with a single hole; $a \ = \ 1$ mm, plate-to-diameter ratio 10. (a) Longitudinal stress $\sigma _ { x x }$ from the FE reference solution (left) and the PI-GNN prediction (right). (b) $\sigma _ { x x }$ along the transverse line of symmetry $x = 0$ from the PI-GNN, the FE reference, and the analytical Kirsch solution, showing the decay of the stress-concentration factor from 3 at the hole boundary to unity in the far field.

Table 1: Comparison of the field-wise error for the thin plate with a single-hole $( a = 1 \mathrm { m m } )$ PI-GNN against the FE reference.
<table><tr><td colspan="3"></td><td colspan="2"> $R ^ { 2 }$ </td></tr><tr><td>Field</td><td> $L _ { \alpha } ^ { 2 }$  (%)</td><td>Global</td><td>Near-field</td><td></td></tr><tr><td> $u _ { x }$ </td><td>0.5700</td><td>0.9998</td><td>0.9992</td><td></td></tr><tr><td> $u _ { y }$ </td><td>0.5700</td><td>0.9996</td><td></td><td>0.9989</td></tr><tr><td> $\sigma _ { x x }$ </td><td>0.9397</td><td>0.9963</td><td></td><td>0.9964</td></tr><tr><td> $\sigma _ { \mathrm { v M } }$ </td><td>0.8879</td><td>0.9963</td><td></td><td>0.9964</td></tr></table>

Single hole $( r = 4 \mathrm { m m } )$ : The problem is posed on the full plate (side 20 mm), containing a central hole of radius $r = 4$ mm and subjected to the roller-and-pin conditions shown in Figure 4. This difers from the quarter-domain formulation with symmetry conditions used in the original benchmark [Li et al., 2021]. The PI-GNN is trained on its own conforming mesh, refined around the hole, containing 4368 nodes and 8221 elements. In contrast, the PINN is trained using 10,000 collocation points, which are resampled every 500 epochs via near-field densification. Both are then evaluated on a common FE mesh with 1865 nodes and 3420 elements, which is not used during training. The PI-GNN is evaluated by a forward pass on the mesh graph, while the PINN is evaluated by querying its coordinate network at the mesh nodes.

The displacement magnitude $( | | \mathbf { u } | | )$ and longitudinal stress $( \sigma _ { x x } )$ fields exhibit good qualitative agreement, with all three solvers accurately resolving the characteristic crown stress concentration of approximately 3T and its decay into the surrounding plate (see Figure 6). Quantitatively, the PI-GNN provides better agreement with the FE reference than the PINN, achieving a global $R ^ { 2 } = 0 . 9 9$ compared with 0.96 for the PINN. The same trend is observed in the near-field, where the PI-GNN and PINN achieve $R ^ { 2 } = 0 . 9 9$ and 0.97, respectively, indicating that the additional near-field sampling of the PINN does not substantially improve its accuracy in the stress-concentration region (see Table 2). The relative-error distributions further distinguish the two models. The PI-GNN errors are narrowly distributed around zero, with a median of 0.41% and a standard deviation of 5.24%, whereas the PINN exhibits a positive bias, with a median of +5.07%, and a broader distribution with a standard deviation of 9.69% (Figure 8c).

Three-hole configuration: To test the resolution of interacting stress fields, an asymmetric three-hole configuration is considered in which closely spaced holes of diferent radii generate overlapping stress concentrations. The square plate (side l = 20 mm) contains three circular holes of radii $r _ { 1 } = 1$ mm, $r _ { 2 } = 2 \mathrm { m m }$ , and $r _ { 3 } = 3$ mm centered at (8, 14), (12, 12), and (8, 7), respectively. The material, loading, and boundary conditions are identical to the single-hole example. The PI-GNN is trained on its own conforming mesh refined around the three holes, containing 4455 nodes and 8394 elements, while the PINN is trained using 25,000 collocation points resampled every 500 epochs. Both models are subsequently evaluated on a common unseen FE mesh containing 3158 nodes and 5837 elements, with the converged FE solution on this mesh serving as the reference.

The $\sigma _ { x x }$ fields are compared in Figure 7. The close spacing between the holes concentrates the load within the narrow ligaments separating adjacent holes, with the peak $\sigma _ { x x }$ reaching $\approx 5 T$ at the upper edge of the largest hole. Both neural solvers reproduce the overall stress pattern qualitatively, but the quantitative metrics distinguish their accuracy (Table 2). The PI-GNN achieves $R ^ { 2 } \approx 0 . 9 6$ both globally and in the near-field, compared with 0.94 for the PINN, while its relative $L _ { \sigma _ { x x } } ^ { 2 }$ error is approximately half that of PINN (7.57% versus 15.08%). The $\mathrm { e r r } _ { \sigma _ { x x } }$ distribution (Figure 8d) show a similar trend to the single-hole example: the PI-GNN remains centered (median 0.68%, standard deviation of 7.87%), whereas the PINN exhibits a larger negative bias and broader distribution, with a median of 6.76% and standard deviation of 11.53%. Notably, the sign of the PINN bias changes between the two geometries, from positive in the single-hole example to negative in the three-hole example, whereas the PI-GNN remains close to unbiased in both examples. These results are consistent with the limitation reported by Li et al. [2021], who noted that although the energy-based model qualitatively captures the global stress distribution, its local prediction error remains appreciable in regions surrounding stress concentrations.

FEM  
![](images/315563448a96b91e30ee01f6a45abcd695ca570be34e6ba00c062026e90037e8.jpg)

PINN  
![](images/411839faf08a36ef38a9a614cfb5ab958d60b8403375a63af50fd926e54a30b6.jpg)

PI-GNN  
![](images/98e8ac57cfad53310cfe76f9f5c868af74a8c152683633aa5aa58dfc17146355.jpg)

![](images/a55c174e764b331a7368571d1057a405411036f2256bbb0f39a40b738f6424f3.jpg)

(b)  
![](images/5423115e4ace79b1169f77f78d95b2bfae8df512c467ca48463b9990bc18b982.jpg)

(c)  
![](images/905180c678f9b1f8cfb2a2478660656a9741202f764a8b85832fcd5a6dbae939.jpg)

![](images/0e6973de10d0df6f9f5c11ba19557620566f7adf42a3abe4d58907f3bd375b24.jpg)  
(g)

![](images/7f686611058876570a3a94ac9038c9f5efa11b21d09fb60298b6a20afdc88851.jpg)  
(h)

(f)  
![](images/645c902f8645426b97200dbf272610a0f8f9b4de19a5d85da3f3392aa05b3352.jpg)  
(i)  
Figure 6: Discretizations and field predictions for the single-hole plate $( r = 4 \mathrm { m m } )$ . Top row: (a) Shared FE evaluation mesh (1865 nodes, 3420 elements). (b) Collocation points for the PINN energy estimate, sampled with near-field densification (10,000 points, resampled every 500 epochs) [Li et al., 2021]. (c) PI-GNN training mesh (4368 nodes, 8221 elements). The neural solvers are trained on (b) and (c), then evaluated on (a). Middle row: Displacement magnitude $( \lVert \mathbf { u } \rVert )$ fields evaluated on the FE reference mesh. (d) FE reference solution, (e) energy-based PINN prediction [re-implemented from Li et al., 2021], and (f) PI-GNN prediction. Bottom row: Longitudinal stress $( \sigma _ { x x } )$ fields evaluated on the FE reference mesh. (g) FE reference solution, (h) energy-based PINN prediction, and (i) PI-GNN prediction.

PINN

PI-GNN

![](images/01dd8423ebf010fda318af29880127c248a5c0a63017d84c30e2ae05241f0dc2.jpg)  
Figure 7: Top row: Displacement magnitude ( u ). Bottom row: longitudinal stress $\sigma _ { x x }$ for the asymmetric three-hole configuration [Li et al., 2021]. The peak $\sigma _ { x x } \approx 5 T$ develops in the ligament above the largest hole. (a) FE reference solution (FEniCSx). (b) Energy-based PINN prediction, [re-implemented here Li et al., 2021]. (c) PI-GNN prediction.

Table 2: Longitudinal stress $( \sigma _ { x x } )$ error metrics for the PINN and PI-GNN predictions against the FE reference on the unseen evaluation mesh, for the single-hole (r = 4 mm) and three-hole configuration. The $\mathrm { e r r } _ { \sigma _ { x x } }$ distribution is summarized by its median and standard deviation across nodes that carry more than 20% of the peak stress.
<table><tr><td rowspan="2">Metric</td><td colspan="2">Single hole</td><td colspan="2">Three holes</td></tr><tr><td>PINN</td><td>PI-GNN</td><td>PINN</td><td>PI-GNN</td></tr><tr><td> $L _ { \sigma _ { x x } } ^ { 2 } ~ ( \% )$ </td><td>10.9425</td><td>5.0128</td><td>15.0781</td><td>7.5738</td></tr><tr><td> $R ^ { 2 } ~ \mathrm { ( g l o b a l ) }$ </td><td>0.9686</td><td>0.9918</td><td>0.9384</td><td>0.9600</td></tr><tr><td> $R ^ { 2 } \ \mathrm { ( n e a r - f i e l d ) }$ </td><td>0.9715</td><td>0.9919</td><td>0.9419</td><td>0.9593</td></tr><tr><td>Std. dev. (%)</td><td>9.6903</td><td>5.2359</td><td>11.5258</td><td>7.8710</td></tr><tr><td>Median signed rel. error (%)</td><td>5.0698</td><td>-0.4116</td><td>-6.7578</td><td>-0.6793</td></tr></table>

Across the three configurations, the PI-GNN reproduces the analytical Kirsch solution for the single-hole example and achieves accuracy comparable to or higher than the PINN across the perforated-plate benchmarks, with the largest diferences observed in the nearfield regions surrounding the holes. The PI-GNN also maintains a median pointwise error close to zero across the considered configurations, whereas the PINN exhibits geometrydependent bias. Both comparisons are performed under discretization transfer, with each model trained on its own sampling of the domain and evaluated on a separate unseen FE mesh using the same architecture and training settings. The PI-GNN is, therefore, evaluated by a forward pass on an unseen mesh graph. In contrast, the coordinate-based PINN is evaluated by querying its learned continuous map at new coordinates. The sharper near-field accuracy observed for the PI-GNN is consistent with its mesh-based energy formulation, in which the internal energy is evaluated element-wise using exact element areas rather than estimated through Monte Carlo collocation [Li et al., 2021].

![](images/c0506f111267fd5e4574c92eb3900bff4ec089a37607ea758fb656d7db541087.jpg)

![](images/b5ad14fb4a562edeee635c5020d627cbacf81d4604cf90ef7288dd6e9cea1ff1.jpg)

![](images/30a338b14fd639df097ca6b6a3694883144ef202758b912aa8b9df144e85c854.jpg)

![](images/139cc3e3960f19136b93f613b74ea0ebcfb1f6a390a43e0c0f7aa3f8232549cd.jpg)  
Figure 8: Spatial density of the element-wise error $( \mathrm { e r r } _ { \| \mathbf { u } \| }$ and $\mathrm { e r r } _ { \sigma _ { x x } } )$ for the PINN and PI-GNN predictions against the FE reference, where the dashed vertical lines are the median error for the respective solver. (a) & (c) Single hole $( r = 4 \mathrm { m m } )$ . (b) & (d) Asymmetric three-hole configuration. Nodes carrying less than 20% of the peak stress are excluded to focus the comparison on the failure-relevant region.

## 3.2.2. Two-phase composite

The perforated plate benchmarks isolate geometric stress concentrations in a homogeneous material. Heterogeneous material systems introduce an additional challenge because the material properties are discontinuous across the interface. For a perfectly bonded bimaterial interface, the displacement field remains $C ^ { 0 } .$ -continuous and the traction is continuous, whereas the in-plane stress components exhibit finite discontinuities across the interface. An energy-based continuum micromechanics PINN is evaluated on the same two-phase benchmark to compare its accuracy with the proposed graph-based solver and to establish a reference error level for heterogeneous material systems [Henkes et al., 2022].

The benchmark consists of a square plate $\Omega = [ 0 , l ] ^ { 2 }$ with $l = 2 \mathrm { { m m } }$ , containing a circular inclusion, $\Omega _ { \mathrm { i n c } } = B _ { a } ( \mathbf { x } _ { c } )$ , of radius $a = 0 . 4$ mm centered at $\mathbf { x } _ { c } = ( 1 , 1 )$ mm, with the surrounding matrix occupying $\Omega _ { \mathrm { m a t } } = \Omega \backslash \overline { { \Omega } } _ { \mathrm { i n c } }$ . Both phases have $\nu _ { \mathrm { m a t } } = \nu _ { \mathrm { i n c } } = 0 . 4$ while $E _ { \mathrm { m a t } } = 1 5 0 0 \mathrm { M P a }$ and $E _ { \mathrm { i n c } } = 1 0 0 0 0 \mathrm { M P a }$ , yielding a stifness contrast of 6.67. A uniform tensile traction of $T = 0 . 0 2 5 \mathrm { M P a }$ is applied to the right edge, with the remaining boundaries subjected to the roller-and-pin constraints. The geometry, material properties, and loading are identical to those of the reference study.

Unlike the deep-energy baseline, which shares the variational formulation of the PI-GNN, the reference model is a strong-form, mixed-variable PINN that predicts $( u _ { x } , u _ { y } , \sigma _ { x x } ,$ $\sigma _ { y y } , \sigma _ { x y } )$ . Equilibrium and constitutive consistency are enforced through collocated residuals rather than a single energy functional. The implementation follows the code released by Henkes et al. [2022], retaining the published architecture and training settings: four hidden layers of 64 neurons with tanh activations and LeCun-uniform initialization, a uniform 128 128 collocation grid, and 5000 BFGS iterations, yielding a final training loss of $3 . 4 1 \times 1 0 ^ { - 6 }$ . Material heterogeneity is represented through the hyperbolic-tangent regularization of the Lam´e parameters across a transition band of width $\delta = 0 . 0 1$ . With further model related details in Section Appendix C.

The PI-GNN uses the same architecture and hyperparameters from the previous verification example and is trained for 50,000 epochs using Adam with $\eta = 1 0 ^ { - 3 }$ The two models, therefore, use diferent optimization formulations: a strong-form residual minimization with BFGS for the reference PINN and discrete energy minimization with Adam for the PI-GNN. The only modification to the released PINN implementation concerns the post-processing. Rather than evaluating the trained network on the $1 2 8 \times 1 2 8$ collocation grid used during training, the network is evaluated at an independent set of points, allowing both neural models and the FE reference to be compared on a common discretization (3034 nodes and 5944 elements), which is not used during training. The mesh conforms to the material interface, with 252 nodes on the interface. This interface-conforming discretization permits the stress field to take distinct values on either side of the interface while retaining the required displacement continuity across the perfectly bonded interface. The PI-GNN, therefore, represents the material discontinuity explicitly through the conforming mesh topology, without introducing a smoothing or regularization layer at the interface.

Figure 9 compares the $\lvert \lvert \mathbf { u } \rvert \rvert$ and $\sigma _ { x x }$ fields. Both models reproduce the characteristic load-transfer mechanism of a stif inclusion embedded in a compliant matrix: the inclusion carries a large fraction of the applied load, resulting in elevated $\sigma _ { x x }$ within the inclusion and stress shielding in the surrounding matrix. The quantitative comparison shows comparable accuracy for the two models (see Tables 3 and 4). The relative $L ^ { 2 }$ error in the displacement field is below 1% for both methods, with 0.86% for the PINN and 0.17% for the PI-GNN. For $\sigma _ { x x }$ , the corresponding errors are 3.02% and 3.78%, respectively, while the von Mises stress errors are 2.57% and 1.85%. All predicted fields achieve $R ^ { 2 } \ge 0 . 9 5$ globally and within the near-field region. The transverse normal and shear stresses exhibit larger relative $L ^ { 2 }$ errors, ranging from 15% to 28% for both models. These components carry little of the load under uniaxial tension; consequently, their larger relative errors primarily result from normalization by a small reference-field norm, while the corresponding absolute errors remain small (see Section 3.3). Overall, neither model exhibits a consistent accuracy advantage. This benchmark demonstrates that the proposed PI-GNN, without architecture or hyperparameter tuning, achieves accuracy comparable to that of a PINN specifically developed for heterogeneous continuum micromechanics.

Two diferences nevertheless emerge in the distribution of the errors. The first concerns the near-field region. Restricting the evaluation to the 2668 nodes located within twice the inclusion radius of the interface, the PI-GNN exhibits improved $R ^ { 2 }$ values for the displacement components, $\sigma _ { x x } .$ , and $\sigma _ { \mathrm { v M } }$ , whereas the PINN shows a slight reduction in $R ^ { 2 }$ for all fields except $\sigma _ { x y }$ . The signed errors show a similar trend: the PI-GNN remains centered near 0.0% for $u _ { x } , \sigma _ { x x } ,$ and $\sigma _ { \mathrm { v M } }$ , whereas the PINN exhibits biases of 0.86% for $u _ { x }$ . Although these diferences are small, they indicate a modest advantage for the PI-GNN in resolving the fields near the material interface.

The second diference concerns the interface’s representation. Owing to the conforming mesh, the PI-GNN captures the stress discontinuity across $\Gamma _ { \mathrm { i n t } }$ directly through material assignment of adjacent elements. In contrast, the coordinate-based PINN represents the material interface through a hyperbolic-tangent transition band of width $\delta \ : = \ : 0 . 0 1$ producing a more difuse transition in the predicted stress contours (see Figure 9). The corresponding pointwise error distributions are shown in Figure 10.

![](images/3d349423fab81e56b9533578f9db285d8e6bc4a9a7425132d2917954cccdbb07.jpg)  
(a)

![](images/8af944d7356cce41cf73e9eeb8e4355c4dce5adb538668347d3fc4ba70a0547b.jpg)  
(b)

![](images/361732e5e52d31b89325b1a267c7d42e93544f593644294e684b4b52b4c16001.jpg)  
(c)

![](images/9fc2ae1592714ec4008fd733945f4faa5062448c2014a838b65eddf4bc973f5f.jpg)  
(d)

![](images/c9f3e6d812c5fd977bef97949777ca7216ace67ad9f8fe4a2bae836735ad0413.jpg)  
(e)

![](images/a45a33e4fc93d2cf5b29872f8a31ab2c1c955b21d15e2691bb3e2583e6efe8f7.jpg)

![](images/c59855903290d2048359cc1fd018ecc44a12aa07acd32dedf091cff7a8041e3f.jpg)  
(g)

![](images/9404c83d37d5362a460d43c10492f5f5017f94b4e20756532a282f2fbe6e9fc4.jpg)  
(h)

(f)  
![](images/8119e26a16b06daa94271d8fd34929ce31e7bb0642e94f4dd4a1500451043faf.jpg)  
(i)  
Figure 9: Discretizations and field predictions for the circular inclusion. Top row: (a) Shared FE evaluation mesh (3034 nodes, 5944 elements). (b) The uniform point cloud of $1 2 8 \times 1 2 8$ grid on which the PINN of Henkes et al. [2022] is trained. (c) PI-GNN training mesh (3762 nodes, 7316 elements). The neural solvers are trained on (b) and (c), then evaluated on (a). Middle row: Displacement magnitude $( \lVert \mathbf { u } \rVert )$ fields evaluated on the FE reference mesh. (d) FE reference solution, (e) energy-based PINN prediction [Henkes et al., 2022], and (f) PI-GNN prediction. Bottom row: Longitudinal stress $( \sigma _ { x x } )$ fields evaluated on the FE reference mesh. (g) FE reference solution, (h) PINN prediction, and (i) PI-GNN prediction.

![](images/46a092b4e60c75a3d14eecf1f82c5427f0c5c503095e0ee8f6d4da932e82fb3f.jpg)

![](images/96c486e24d5cb06de5442dbf09892544de2a8e74d51edc75ae0fab9bff58c56c.jpg)  
Figure 10: Spatial density of the element-wise relative error for the PINN and PI-GNN predictions against their respective FE references for the circular inclusion. (a) $\mathrm { e r r } _ { | | \mathbf { u } | | } .$ (b) $\mathrm { e r r } _ { \sigma _ { x x } }$ . Nodes carrying less than 20% of the peak reference value are excluded, following Equation (23).

Table 3: $L ^ { 2 }$ errors for the PINN and the PI-GNN, each against the FE solution, on the common evaluation mesh of the circular inclusion.
<table><tr><td colspan="3">Relative Error</td></tr><tr><td>Field</td><td>PINN</td><td> $L _ { \alpha } ^ { 2 } ~ ( \% )$  PI-GNN</td></tr><tr><td> $\lvert \lvert \mathbf { u } \rvert \rvert$ </td><td>0.8622</td><td>0.1699</td></tr><tr><td> $\sigma _ { x x }$ </td><td>3.0223</td><td>3.7801</td></tr><tr><td> $\sigma _ { y y }$ </td><td>19.7799</td><td>27.4440</td></tr><tr><td> $\sigma _ { x y }$ </td><td>16.9558</td><td>14.5477</td></tr><tr><td> $\sigma _ { \mathrm { v M } }$ </td><td>2.5672</td><td>1.8468</td></tr></table>

Table 4: Field-wise $R ^ { 2 }$ scores and median signed relative errors. The near-field comprises the 2668 nodes within twice the inclusion radius of the interface.
<table><tr><td rowspan="3"></td><td colspan="3">PINN</td><td colspan="3">PI-GNN</td></tr><tr><td colspan="2"> $R ^ { 2 }$ </td><td rowspan="2">Median Error (%)</td><td colspan="2"> $R ^ { 2 }$ </td><td rowspan="2">Median Error</td></tr><tr><td>Global</td><td>Near-field</td><td>Global</td><td>Near-field</td></tr><tr><td> $u _ { x }$ </td><td>0.9989</td><td>0.9962</td><td>-0.8570</td><td>0.9978</td><td>0.9999</td><td>(%) 0.1521</td></tr><tr><td> $u _ { y }$ </td><td>0.9988</td><td>0.9960</td><td>0.9318</td><td>0.9947</td><td>0.9999</td><td>-0.0951</td></tr><tr><td> $\sigma _ { x x }$ </td><td>0.9820</td><td>0.9811</td><td>0.0921</td><td>0.9750</td><td>0.9855</td><td>0.1459</td></tr><tr><td> $\sigma _ { y y }$ </td><td>0.9740</td><td>0.9729</td><td>-0.4006</td><td>0.9582</td><td>0.9571</td><td>1.5097</td></tr><tr><td> $\sigma _ { x y }$ </td><td>0.9731</td><td>0.9743</td><td>-0.4342</td><td>0.9884</td><td>0.9882</td><td>0.0066</td></tr><tr><td> $\sigma _ { \mathrm { v M } }$ </td><td>0.9796</td><td>0.9773</td><td>0.8186</td><td>0.9838</td><td>0.9929</td><td>-0.0294</td></tr></table>

Accuracy under increasing stifness contrast: The PINN and PI-GNN comparison with the FE reference solution described above considers a single stifness contrast, $E _ { \mathrm { i n c } } / E _ { \mathrm { m a t } } = 6 . 6 7$ . To assess the robustness of both solvers to increasing material contrast, we repeat the comparison across twenty-five logarithmically spaced contrasts spanning $E _ { \mathrm { i n c } } / E _ { \mathrm { m a t } } \in [ 1 0 ^ { - 2 } , 1 0 ^ { 2 } ]$ , placed reciprocally symmetrically about unit contrast, varying only $E _ { \mathrm { i n c } } .$ . The matrix modulus, both Poisson’s ratios, geometry, applied traction, evaluation mesh, and all architectural and optimization settings are kept fixed. In particular, the PINN retains its transition width $\delta = 0 . 0 1 , 1 2 8 \times 1 2 8$ collocation grid, and BFGS iteration budget, with no problem-specific retuning. For each stifness contrast, the solvers are trained from three independent weight initializations and evaluated on the same mesh using the area-weighted relative $L ^ { 2 }$ (Equation (22)).

![](images/bc49ab2da842d8e5011efa05da45c05cac356c9d07ffe538792409dcd347dce7.jpg)  
(a)

![](images/48575a03851703758f85c2011305dc54ef73e634f787c56f96dc43acbc3a177a.jpg)  
(b)  
Figure 11: $L _ { \sigma _ { \mathrm { v M } } } ^ { 2 }$ error against the FE reference as a function of the stifness contrast, for the strong-form PINN [Henkes et al., 2022] and the PI-GNN. Twenty-five logarithmically spaced contrasts spanning $E _ { \mathrm { i n c } } / E _ { \mathrm { m a t } } \in [ 1 0 ^ { - 2 } , 1 0 ^ { 2 } ]$ , placed reciprocally symmetrically about unit contrast. Solid and dashed lines denote PINN and PI-GNN, respectively. The lines show the mean across three random seeds, and the shaded regions show the seed-level range. The solvers are evaluated on an interface-conforming mesh (3034 nodes, 5944 elements) that is not used in PI-GNN training, and the area-weighted metric is used. (a) is the soft inclusion branch and (b) the stif branch.

The two solvers exhibit comparable accuracy for stifness contrasts below approximately 2, with $\sigma _ { \mathrm { v M } }$ errors near 2.3% (see Figure 11). The errors begin to separate beyond a contrast of approximately 3: the PINN error increases linearly with log $( E _ { \mathrm { i n c } } / E _ { \mathrm { m a t } } )$ , reaching 5.58% at a contrast of 100, whereas the PI-GNN remains below 3.58%. The same trend is reflected in $R ^ { 2 }$ , with the PINN decreasing from a maximum of 0.983 to 0.878, while the PI-GNN remains near 0.976. The displacement error shows an even stronger separation, reaching 3.10% for the stif-inclusion example and 7.66% for the soft-inclusion example at a contrast of 100, compared with 0.20% and 0.49%, respectively, for the PI-GNN.

The PI-GNN is largely insensitive to whether the inclusion is stifer or softer, whereas the PINN degradation is associated with the fixed interface regularization width $\delta = 0 . 0 1$ A contrast-dependent choice of $\delta$ could reduce this degradation, but would introduce an additional problem-specific parameter. The comparison, therefore, does not claim superiority over an optimally tuned strong-form PINN. It rather demonstrates that the PI-GNN maintains a single configuration across the full contrast range without requiring an interface parameter to be adjusted. At a contrast of 100, the PINN error is 1.56 times the PI-GNN error for $\sigma _ { \mathrm { v M } }$ . Repeating the evaluation on the PI-GNN training mesh gives 3.07%, compared with 3.58% on the held-out mesh, indicating that the observed trend is not sensitive to the choice of evaluation mesh.

At the reference contrast of 6.67, the two solvers achieve comparable accuracy at similar computational cost, despite using diferent optimization strategies. The PI-GNN completes 50,000 Adam epochs in 2497 $| \mathrm { s } , $ compared with 1200 s for the PINN after 5000 BFGS iterations on the same hardware. Thus, first-order optimization of the discrete potential energy achieves accuracy comparable to that of quasi-Newton optimization of the strong-form residuals on this benchmark. The memory requirements of BFGS also increase with the number of trainable parameters, which can become an important consideration for larger 3D and finite-strain problems. More importantly, the PI-GNN achieves this accuracy without problem-specific tuning. Geometry is introduced exclusively through the conforming mesh, while material heterogeneity enters through element-wise material properties. In contrast, coordinate-based PINNs require an appropriate collocation strategy and, for sharp material interfaces, a suitable interface regularization width.

## 3.2.3. Isolating the efect of message passing

The comparisons above demonstrate an advantage over the collocation-based energy PINN of Li et al. [2021] and, at high stifness contrast, over the strong-form PINN of Henkes et al. [2022]. However, these comparisons do not isolate whether the observed advantage arises from the element-wise energy discretization or from message passing on the graph. To separate these efects, we construct a mesh-based PINN by modifying the existing PI-GNN while keeping its architecture and parameter count unchanged. Specifically, the neighborhood  (i) in Equation (11) is restricted to the node itself, eliminating information exchange between neighboring nodes. The resulting model, therefore, minimizes the same FE Ritz functional on the conforming mesh. Still, the mesh topology enters only through the assembly of the energy and does not influence the nodal trial field through message passing. This provides a direct comparison between mesh-based energy minimization with and without graph-based neighborhood interactions.

Two sweeps are performed on the two-phase inclusion problem. The first varies the mesh element size over nine values $h \in \left\lceil 0 . 0 1 , 0 . 0 5 \right\rceil$ at a fixed stifness contrast; the second varies the contrast over seven values $E _ { \mathrm { i n c } } / E _ { \mathrm { m a t } } \in \textup { [ } 1 0 ^ { - 2 } , 1 0 ^ { 2 } ]$ on a fixed mesh. Each configuration is trained from 3 random seeds (initializations). The models use a reduced setting for this study 17,250 parameters (32 hidden neurons and 4 layers) and 5000 Adam epochs at $\eta = 1 0 ^ { - 3 }$ The smaller network size and limited training budget assess the convergence speed of the mesh-based PINN and PI-GNN models. At this fixed budget, the PI-GNN consistently achieves the lower von Mises error in both the mesh sweep and the stifnessratio sweep.

Two trends indicate where the gain originates. The accuracy gain due to message passing increases as the mesh coarsens, from a 19% error drop at $h = 0 . 0 1$ to 43% at $h = 0 . 0 5$ (see Figure 12a & b). With (L = 4) message-passing layers, a node aggregates over a graph neighborhood spanning four element widths, so a single forward pass covers a larger fraction of the inclusion when the mesh is coarse; as (h) decreases, the same four layers span a shorter physical distance and the margin narrows, although it remains at (19%) on the finest mesh. The second trend is the stifness contrast dependence (Figure 12c & d). Here, the unit ratio refers to a homogeneous body, with no interface; both models yield errors near 0.3%, and the gap closes. Away from the unit contrast, the gain in error due to message passing increases in displacement and stress fields. At this budget, the gain is therefore largest where the neighborhood exchange carries the most information, namely, across the material interface. Message passing incurs training time that is 1.2 to 2.2 times that of the ablated model at the same parameter count, because each epoch evaluates a message on every edge.

The models exhibit increasing error as h decreases, which reflects the fixed optimization budget rather than a loss of consistency. In the reduced setting (17,250 trainable parameters and 5000 Adam epochs), the optimization budget is held fixed while the number of nodal unknowns increases with mesh refinement. Consequently, the finer meshes require optimization over larger discrete systems and are less fully converged within the prescribed epoch budget. At the same time, mesh refinement yields a more accurate discrete approximation of the continuum solution, making the remaining optimization error more apparent when measured against the refined reference solution. Thus, Figure 12a reflects the optimization progress and convergence rate under a fixed computational budget, rather than the accuracy of fully converged models. A representative run with 30,000 epochs is provided in Appendix C.2.

![](images/51af04a51867f94a72f642eaf3f8e255f476eec98a3f0473db22094d0953011c.jpg)  
(a)

![](images/461573e2929f83073ca70f5272605304c0f6213e3168d26533713235ff0121d7.jpg)  
(b)

![](images/80e003fe02b727fc7113bfb47307b8b3d278c4b7a8c9d26d6b0676cbc7f82ee4.jpg)  
(c)

![](images/b0dab4ddc28d12286b1fbef09e3f87e93080376da5f7829e20d5c3a66155be11.jpg)  
(d)  
Figure 12: Efect of message passing at a fixed parameter count. The mesh-based PINN is the PI-GNN with $\mathcal { N } ( i )$ restricted to the node itself. (a) $L _ { \sigma _ { \mathrm { v M } } } ^ { 2 }$ against the FE reference over the mesh sweep, and (b) the corresponding error reduction; h decreases to the right. (c) $L _ { \sigma _ { \mathrm { v M } } } ^ { 2 }$ over the stifness-contrast sweep, and (d) $L _ { \mathrm { u } } ^ { 2 }$ over the stifness-contrast sweep ; the dashed vertical line marks the homogeneous limit $E _ { \mathrm { i n c } } / E _ { \mathrm { m a t } } = 1$ . Lines show the mean across three random seeds, and shaded regions show the seed-level range.

## 3.3. Verification summary

The verification suite demonstrates that a single PI-GNN architecture can accurately reproduce smooth elastic fields, localized stress concentrations, multi-feature stress interactions, and discontinuous constitutive responses at heterogeneous interfaces, without requiring labeled data and per-case tuning. Table 5 summarizes the nodes and elements of the evaluation mesh, training configurations, and loss metrics for each example. The benchmark problems demonstrate that the proposed PI-GNN, without architecture or hyperparameter tuning, achieves accuracy comparable to that of a FEM/PINN.

For the linear elastic problems, the internal strain energy U and external work $W _ { \mathrm { e x t } }$ satisfy Clapeyron’s theorem $( 2 U = W _ { \mathrm { e x t } } )$ at equilibrium. Consequently, the total potential energy $\Pi = U - W \mathrm { e x t } = - U .$ , and the converged solution is expected to satisfy $\Pi / W _ { \mathrm { e x t } } =$ $- 0 . 5$ The convergence history is monitored to verify stabilization at this ratio. The converged PI-GNN energy is compared with the FE value using the relative diference $( \Pi _ { \mathrm { P I - G N N } } - \Pi _ { \mathrm { F E M } } ) / | \Pi _ { \mathrm { F E M } } |$

Table 5: Summary of evaluation mesh configuration, training, and metrics across the verification examples.
<table><tr><td></td><td colspan="2">Training mesh</td><td colspan="2">FE evaluation mesh</td><td colspan="4"></td></tr><tr><td>Problem</td><td>Nodes</td><td>Elements</td><td>Nodes</td><td>Elements</td><td>Epochs</td><td>η</td><td> $\Pi / W _ { \mathrm { e x t } }$ </td><td>errn</td></tr><tr><td>Single-hole</td><td>4368</td><td>8221</td><td>1865</td><td>3420</td><td>30,000</td><td>0.001</td><td>-0.49</td><td>+0.40</td></tr><tr><td>Three-hole</td><td>4455</td><td>8394</td><td>3158</td><td>5837</td><td>30,000</td><td>0.001</td><td>-0.50</td><td> $+ 1 . 9 7$ </td></tr><tr><td>Inclusion</td><td>3762</td><td>7316</td><td>3034</td><td>5944</td><td>50,000</td><td>0.001</td><td>-0.50</td><td>+0.13</td></tr></table>

Remark: In each example, the non-dominant stress components, such as $\sigma _ { y y }$ and $\sigma _ { x y }$ under uniaxial tension, carry a small fraction of the load and consequently exhibit larger relative $L ^ { 2 }$ errors, even though the dominant stress component and $\sigma _ { \mathrm { v M } }$ are closely reproduced. This diference is primarily a consequence of the component-wise normalization in Equation (22), where $L _ { \alpha } ^ { 2 }$ is divided by the norm of the corresponding reference component. Thus, for components of small magnitude, even a small absolute error can result in a comparatively large relative error. The nodal exclusion in Equation (23) does not alter this normalization, since it is applied independently to each stress component.

The small magnitude of these errors is further supported by the close recovery of the von Mises stress, which difers from the reference by at most 1.85%. An error in a nondominant component comparable in magnitude to the dominant stress would generally produce a substantially larger deviation in the resulting von Mises field and is therefore inconsistent with the observed agreement. The coeficients of determination for these components also remain above 0.95, although $R ^ { 2 }$ provides limited independent evidence in this setting (see Table 4). For components with means close to zero, the normalization in $R ^ { 2 }$ and the relative $L ^ { 2 }$ error are governed by quantities of similar magnitude, making the two measures strongly related. For components that are identically zero by construction, such as the in-plane normal stresses under pure torsion, the reference variance vanishes, and $R ^ { 2 }$ is consequently not informative. The same normalization argument applies to the non-dominant components in the remaining examples.

## 4. Numerical applications

This section applies the verified framework to three challenging problems for which no closed-form solution exists. In each example, a converged FE solution computed on the same conforming mesh serves as the reference, so the reported errors quantify only the approximation introduced by the PI-GNN. The applications extend the verification benchmarks in three complementary directions. The first considers re-entrant inclusions to assess the resolution of stress concentrations induced by concave corners. The second and third investigate a 3D reinforced cube and rod subjected to torsion, respectively. All problems are trained using the same network architecture and hyperparameters similar to Section 3, with a constant learning rate of $\eta = 1 \times 1 0 ^ { - 3 }$ for 100,000 epochs. The training and evaluation are performed on the same conforming mesh.

## 4.1. Material Discontinuities

A material discontinuity is introduced by embedding a stif inclusion within a compliant matrix to assess the PI-GNN to resolve heterogeneous constitutive behavior across bonded material interfaces. The geometry and material properties are adopted from Kamali et al. [2023]. Following the reference study, the surrounding matrix consists of a 10% w/v PVA hydrogel $( E _ { \mathrm { m a t } } = 1 5 0 0 \mathrm { P a } , \nu _ { \mathrm { m a t } } = 0 . 4 5 )$ , while the inclusion is a stifer 15% w/v PVA hydrogel $( E _ { \mathrm { i n c } } = 5 0 0 0 \mathrm { P a } , \nu _ { \mathrm { i n c } } = 0 . 3 5 )$ , resulting in a stifness contrast of $E _ { \mathrm { i n c } } / E _ { \mathrm { m a t } } \approx 3 . 3 3$ Both phases, matrix and inclusion, are modeled as Neo-Hookean solids.

## 4.1.1. Re-entrant inclusion

The circular inclusion example is extended to a non-convex inclusion geometry to evaluate the PI-GNN to resolve the steep stress gradients introduced by re-entrant corners. Unlike the smooth circular interface, the reversal of interface curvature at the re-entrant lobes produces localized stress concentrations that are challenging to capture. The geometry consists of a square hydrogel layer of side l = 2 mm containing a five-lobed re-entrant inclusion positioned slightly of the plate center, eliminating symmetry about both coordinate axes. The same Dirichlet boundary conditions are imposed, together with a uniform tensile traction of T = 50 Pa applied to the right edge. The conforming FE mesh consists of 5088 nodes and 9966 linear triangular elements, with local refinement along the re-entrant interface. The union of the refined regions defines the near-field domain used for the localized error analysis.

Figure 13 presents the stifer inclusion carries the elevated stress, reaching $\approx 1 0 0 \mathrm { P a }$ along the lobes aligned with the loading direction. In contrast, the matrix immediately above and below the inclusion is shielded and relaxes toward 40 Pa. The PI-GNN reproduces the position and magnitude of every lobe, including the steep gradients at the concave notches between adjacent lobes.

![](images/0a09b10d4aa73a5e917c1142dc1976b4325dbb7fb171790e47c3138614e63fd1.jpg)

![](images/f77663f7efa7c88be07797323a27eef63b397930efcac628f5e88f2d4a8c876a.jpg)  
(b)

![](images/64eb1c7326bd606770c840f10b149b7ec29671bbe7216338d78009b90398055e.jpg)

![](images/6661eba406cf32f410e957ababcdbab09e0250340dec1a443aa270cc22f6f0ec.jpg)  
(d)

![](images/bc32d6d06b6ace71da2a17e2a26100a67f0d0c3580aa696f875538160d509060.jpg)  
(e)  
Figure 13: Stif re-entrant inclusion in a soft hydrogel layer. (a) Reference configuration showing a conforming graph. (b) Deformed FEM configuration showing u ; (c) Deformed PI-GNN prediction showing $| | u | |$ ; (d) Deformed FEM configuration showing $\sigma _ { x x }$ . (e) Deformed PI-GNN prediction showing $\sigma _ { x x }$ . Both stress panels use the same color scale in Pa. The deformed configurations are scaled by a factor of 5.

The displacement components achieve $R ^ { 2 } \ge 0 . 9 9$ globally and in the near field, indicating that the irregular interface does not degrade the displacement solution (Table 6). The signed median errors for $u _ { x }$ and $u _ { y }$ are $- 0 . 7 1 6 1 \%$ and $+ 0 . 6 6 8 1 \%$ , indicating no appreciable bias in either component. The dominant stress $\sigma _ { x x }$ and the von Mises stress achieve $R ^ { 2 } \ge 0 . 9 8$ , with near-field values marginally exceeding their global counterparts, indicating consistently accurate recovery of the primary load-carrying fields.

Table 6: Error metrics comparing the PI-GNN predictions with FEM reference for reentrant inclusion.
<table><tr><td rowspan="2">Field  $L _ { \alpha } ^ { 2 } ~ ( \% )$ </td><td rowspan="2"></td><td rowspan="2">Median  $\mathrm { e r r } _ { \alpha }$  (%)</td><td colspan="2"> $R ^ { 2 }$ </td></tr><tr><td>Global</td><td>Near-field</td></tr><tr><td> $u _ { x }$ </td><td>0.7108</td><td>-0.7161</td><td>0.9956</td><td>0.9994</td></tr><tr><td> $u _ { y }$ </td><td>0.6211</td><td>0.6681</td><td>0.9971</td><td>0.9993</td></tr><tr><td> $\sigma _ { x x }$ </td><td>2.7457</td><td>-0.4408</td><td>0.9831</td><td>0.9908</td></tr><tr><td> $\sigma _ { \mathrm { v M } }$ </td><td>2.2346</td><td>-0.2898</td><td>0.9852</td><td>0.9935</td></tr></table>

The error-density plots show how the residuals are distributed across the domain (Figure 14). The displacement magnitude error is unimodal and tightly concentrated, with its median near $\mathrm { e r r } _ { | | \mathbf { u } | | } \approx - 0 . 3 \%$ and most of the probability mass between 1.0% and $+ 0 . 3 \%$ (see Figure 14a). The bias is therefore negligible and marginally negative. The von Mises error is centered near $\operatorname { e r r } _ { \sigma _ { \mathrm { v M } } } \approx - 0 . 3 \%$ , with a roughly symmetric core inside 3% (see Figure 14b). Further, the displacement field is recovered with $R ^ { 2 } > 0 . 9 9$ , and the stress concentrations at the concave notches are resolved without a local penalty, since the near-field $R ^ { 2 }$ exceeds the global value for every field. The error metrics indicate that the PI-GNN framework handles circular and re-entrant inclusions without bias, and that accuracy does not deteriorate at the interface, where the gradients are steepest.

![](images/d4d0602fd453a3a876ab3c016f12f2a3687228cdc9ef005eda2b57ca0761869e.jpg)

60  
![](images/e34cf7c91965ee52c55da544c7ef4f02319474b9acbbe9169a75c3d9bd1067f6.jpg)  
Figure 14: Relative error densities comparing PI-GNN and the FEM reference for the re-entrant inclusion. (a) Displacement magnitude. (b) von Mises Stress.

## 4.1.2. Re-entrant inclusion with combined normal-shear loading

The re-entrant inclusion problem is subjected to combined normal and shear loading, unlike the uniaxial example, where loading is carried entirely by the longitudinal normal stress, the applied traction $T = 1 0 0 \mathrm { P a }$ is now inclined at $4 5 ^ { \circ }$ to the x-axis. The resulting loading produces equal normal and transverse traction components, resulting in a coupled normalshear stress state. To ensure static equilibrium, the left boundary is fully clamped $( u _ { x } =$ $u _ { y } = 0 )$ instead of being supported by a roller with a single corner pin. The transverse component of the applied traction introduces a net vertical force and bending moment that cannot be balanced by the roller constraint, making a fully fixed boundary essential for a well-posed problem (see Figure 15a).

Figure 15b & c compare the reference and PI-GNN predictions of the $\sigma _ { x x }$ field on the deformed configuration, which exhibits both rigid-body rotation and finite stretch. In contrast to the pure tension, the inclined traction acting against the clamped boundary yields a bending response. The transverse component of the applied load generates a bending moment at the fixed edge, causing $\sigma _ { x x }$ to vary from compression in the upper region (approximately 200 Pa) to tension in the lower region (approximately 400 Pa), superposed on the axial tensile stress. As in the pure-tension example, the stif re-entrant inclusion attracts load into the lobes aligned with the resultant traction, while the concave notches between adjacent lobes continue to exhibit the steepest stress gradients.

![](images/5573195436f84dc6c1d59b73747828e9e8e01b259079a25d716881d03cc969ce.jpg)

![](images/99205000faaf60e7432634ddbbb671d34123e73b778e8fbe3f1817f4b9199142.jpg)  
(b)

![](images/d7c50364c4f659beae4734b3a2536d00b70c5a0be48389aa10c8f395a939a313.jpg)  
(c)

(a)  
![](images/84721539c73395c5d601329851f08e7e6c0481c4806161003709a70e1b83fcc6.jpg)  
(d)

![](images/3ab7dc3f7c122ffee4d8ce276fda1b431299f1cf447778ee669a8958431fba6c.jpg)  
(e)  
Figure 15: Stif re-entrant inclusion in a soft hydrogel layer under normal-shear loading. The stress fields are shown using the same scaled colormap. (a) Reference configuration showing conforming graph and updated boundary conditions. (b) Deformed FEM configuration showing u ; (c) Deformed PI-GNN prediction showing u ; (d) Deformed FEM configuration showing $\sigma _ { x x }$ . (e) Deformed PI-GNN prediction showing $\sigma _ { x x }$ . Both stress panels use the same color scale in Pa. The deformed configurations are scaled by a factor of 5.

The quantitative error metrics are summarized in Table 7. The displacement field is recovered with good accuracy, with $u _ { x }$ and $u _ { y }$ achieving $R ^ { 2 } > 0 . 9 9$ globally and in the nearfield region. Thus, neither the inclined loading nor the fully clamped boundary adversely afects the displacement prediction. The stress metrics provide a more demanding assessment. Under the uniaxial loading, the shear stress $\sigma _ { x y }$ was a low-magnitude component. Rotating the applied traction to $4 5 ^ { \circ }$ promotes $\sigma _ { x y }$ to a primary load-carrying stress, yielding a relative error of $L _ { \alpha } ^ { 2 } = 6 . 3 \%$ , while maintaining $R ^ { 2 } > 0 . 9 7$ for all stress components. The relative error of the dominant stress $\sigma _ { x x }$ increases from 2.8% to 6.8%, reflecting the transition from a predominantly tensile field to one containing both tensile and compressive regions. Nevertheless, its coeficient of determination remains high $\left( R ^ { 2 } \ge 0 . 9 9 \right)$ .

Compared to the uniaxial example, the σvM exhibits a moderate increase in relative error, from 2.2% to 4.9%, consistent with its positive-definite nature. Overall, the stress fields are recovered with high fidelity despite the increased complexity of the multiaxial stress state. Figure 16 shows the pointwise error distributions, providing further insights. The displacement-magnitude error exhibits a narrow distribution with a median of approximately 0.5%, indicating a small systematic underprediction of the displacement magnitude (see Figure 16a). In contrast, the relative error of the σvM is centered close to zero (median 0.8%), with most values confined within 6% (see Figure 16b). These distributions indicate that the displacement and stress predictions remain essentially unbiased under combined loading.

Table 7: Error metrics comparing the PI-GNN predictions and FEM reference for reentrant inclusion with normal-shear loading.
<table><tr><td rowspan="2">Field</td><td rowspan="2"> $L _ { \alpha } ^ { 2 } ~ ( \% )$ </td><td rowspan="2">Median (%)  $\mathrm { e r r } _ { \alpha }$ </td><td colspan="2"> $R ^ { 2 }$ </td></tr><tr><td>Global</td><td>Near-field</td></tr><tr><td> $u _ { x }$ </td><td>1.3539</td><td>-2.7129</td><td>0.9967</td><td>0.9997</td></tr><tr><td> $u _ { y }$ </td><td>0.5690</td><td>-0.4687</td><td>0.9940</td><td>0.9998</td></tr><tr><td> $\sigma _ { x x }$ </td><td>6.8704</td><td>-5.9344</td><td>0.9907</td><td>0.9905</td></tr><tr><td> $\sigma _ { x y }$ </td><td>6.2773</td><td>-1.6033</td><td>0.9776</td><td>0.9736</td></tr><tr><td> $\sigma _ { \mathrm { v M } }$ </td><td>4.8927</td><td>-0.8019</td><td>0.9776</td><td>0.9784</td></tr></table>

![](images/019285460ad15d5089e85d8c39b446387a0bfef133c3b9dbf96ef7c416a1746a.jpg)

![](images/017c64556b53da354408c21a091247891816545aae1f2a64abf99bff360532c6.jpg)  
Figure 16: Relative error densities comparing PI-GNN and the FEM reference for the normal-shear loaded re-entrant inclusion. (a) Displacement magnitude. (b) von Mises Stress.

## 4.2. Composite cube under torsion

We further consider a 3D composite cube under torsion to assess the generalizability of the proposed GNN and energy-based formulation beyond 2D problems. In this example, the geometry consists of a rubber cube (side 20 mm) containing a concentric nylon core of square cross-section, 8 mm on a side, extending over the full 20 mm length along the z-axis. The rubber matrix has shear and bulk moduli $\mu _ { \mathrm { m a t } } = 7 . .$ 41 MPa and $\kappa _ { \mathrm { m a t } } = 2 2 . 2 2 \mathrm { M P a }$ corresponding to $\nu _ { \mathrm { m a t } } \approx 0 . 3 5$ . The nylon core is assigned $E _ { \mathrm { i n c } } = 2 0 0 0 \mathrm { M P a }$ and $\nu _ { \mathrm { i n c } } = 0 . 3$ giving a stifness ratio of $E _ { \mathrm { i n c } } / E _ { \mathrm { m a t } } \approx 1 0 0$ . The face $z = z _ { \mathrm { m i n } }$ is fully fixed, $\mathbf { u } = \mathbf { 0 }$ , while the opposite face $z = z _ { \mathrm { m a x } }$ is subjected to the torsional traction $\mathbf { t } = \tau _ { 0 } ( - y , x , 0 )$ , producing a pure twisting moment about the cube axis. For $\tau _ { 0 } = 0 . 1 5 \mathrm { M P a }$ , the resulting torque is 4000 N mm.

The conforming mesh contains 9261 nodes and 48000 tetrahedral elements. The $d -$ dimensional architecture is instantiated with the feature vector $\mathbf { f } _ { i } = [ \hat { X } _ { i } , \hat { Y } _ { i } , \hat { Z } _ { i } , m _ { i } , \chi _ { i } ] ^ { \mathsf { T } } \in$ $\mathbb { R } ^ { 5 }$ and decoder output $\tilde { \mathbf { u } } _ { i } \in \mathbb { R } ^ { 3 }$ The Dirichlet mask is applied component-wise on the fixed face. The discrete energy is assembled element-wise over the tetrahedra, using the element volume $V _ { e }$ for the internal energy and integrating the torsional traction over the triangular facets of the loaded face, with facet area $S _ { f }$ . Here, a uniform grid whose nodes align with the nylon core’s faces is used, in contrast to the adaptive mesh used in previous examples.

Figure 17 compares the deformed configurations for the PI-GNN and reference solver. The fixed face holds the cube in place while the loaded face rotates about the z-axis. The u grows from near zero at the center of each cross-section toward the corners, reaching 4 mm at the free-face corners (see Figure 17c & d). The PI-GNN twist field (Figure 17d) is visually indistinguishable from the FEM reference (Figure 17c). The absolute error of displacement magnitude $e _ { \parallel \mathbf { u } \parallel }$ mapped onto the cube faces stays below $\approx 0 . 0 5$ mm over most of the volume with $\approx 0 .$ 1 mm near the corners (see Figure 17b). The stifness contrast between the nylon core and the surrounding rubber results in a distinct local error pattern.

![](images/91dd4fbf074a045cf15a2d0d40eaa1d19fc4d851ff322c1ebd8526ac813a2806.jpg)

![](images/230d12f65961532a59de1860e613f6fd654df6b38daf892990bb000f8450d785.jpg)

![](images/5d2f98cd106f0f154490a4d8b5bd82d7d4affaf5d24960f99ae8ffbf7da1300a.jpg)

![](images/7ef51013b217fc89d1d2546b4453358d6bd69558b531da8424bc590ba42a367e.jpg)  
Figure 17: Hyperelastic composite cube under torsion. (a) Reference configuration with the applied torque. (b) Absolute displacement error $( e _ { \parallel \mathbf { u } \parallel } )$ between the reference FE solution and the PI-GNN prediction, mapped onto the cube faces. (c) Deformed configuration obtained from the FE displacement field (u). (d) Deformed configuration from the PI-GNN prediction. Both deformed configurations are shown on the same 1:1 scale, with dimensions in mm.

Table 8 summarizes the error statistics for the torsion problem, indicating that the solution is resolved with accuracy comparable to the 2D inclusion examples. The in-plane displacements $u _ { x }$ and $u _ { y } ,$ which dominate the twist, and the axial component $u _ { z } ,$ tied to warping, reach $R ^ { 2 } \geq 0 . 9 9$ . The stress field is shear-dominated, as expected under torsion: the two load-carrying shear components $\sigma _ { x z } , \sigma _ { y z }$ , and invariant $\sigma _ { v M }$ reach $R ^ { 2 } \ge 0 . 9 9$ , all with signed median errors below 0.5%.

Table 8: Summary of error statistics for the composite cube.
<table><tr><td>Field</td><td> $L _ { \alpha } ^ { 2 } ~ ( \% )$ </td><td>Median  $\mathrm { e r r } _ { \alpha }$  (%)</td><td> $R ^ { 2 } ~ \mathrm { ( g l o b a l ) }$ </td></tr><tr><td> $u _ { x }$ </td><td>0.7651</td><td>0.2147</td><td>0.9999</td></tr><tr><td> $u _ { y }$ </td><td>0.8413</td><td>0.2990</td><td>0.9999</td></tr><tr><td> $\sigma _ { x z }$ </td><td>4.2821</td><td>0.2281</td><td>0.9993</td></tr><tr><td> $\sigma _ { y z }$ </td><td>4.2545</td><td>0.2269</td><td>0.9993</td></tr><tr><td> $\sigma _ { \mathrm { v M } }$ </td><td>3.1946</td><td>0.4636</td><td>0.9995</td></tr></table>

The error-density plots are consistent with these statistics (see Figure 18). The $\mathrm { e r r } _ { \Vert \mathbf { u } \Vert }$ (a) is unimodal with its median near 2.5%, so the network slightly under-predicts the twist amplitude. The von Mises error is broader and roughly symmetric about a small negative median $( \approx - 0 . 5 \% )$ , with most of the probability mass inside $\pm 4 \%$ (see Figure 18b). The spread is wider than the $\pm 2 \%$ core seen for the re-entrant inclusion, reflecting the noisier stress recovery from a denser 3D graph (refer section 4.1). The dominant shear stresses and the von Mises invariant are all recovered with $R ^ { 2 } > 0 . 9 9$ , and the twist field matches the reference to within a couple of percentage points.

![](images/c135660aab0c611eb3c2263cce836a41963a08f377e6192851cde733a0fee0cb.jpg)

![](images/ea3c9699e2643bcde6bf0a10eb0f6e4ea292efb45d293b10c9619aa81160481f.jpg)  
Figure 18: Comparing 3D PI-GNN against the FEM reference using $e r r _ { f }$ density. (a) Displacement magnitude. (b) von Mises Stress.

## 4.3. Composite rod with a re-entrant inclusion under torsion

The final example combines a re-entrant material interface and a fully 3D torsional stress state, on an unstructured mesh with a curved, traction-free lateral boundary. The geometry is a circular rod of radius R = 1 mm and height $H = 2 \mathrm { m m } \ ( H / 2 R = 1 )$ , containing a concentric five-lobed re-entrant inclusion extending over the full height of the rod. The phases are modeled as compressible Neo-Hookean solids using the same material properties as the hydrogel example. Reusing the material pair isolates the efects of 3D torsion and curved geometry while keeping the constitutive contrast fixed. The bottom face $z = 0$ is fully clamped, $\mathbf { u } = \mathbf { 0 }$ , while the top face $z = H$ is subjected to a torque $M _ { z } = 0 . 2 5 \mathrm { { \textmu N } }$ m (see Figure 19a). The torque is chosen to produce finite deformation, with a mean twist exceeding 20<sup>◦</sup>, and the lateral surface is traction-free.

The conforming mesh contains 38430 nodes and 214260 tetrahedra, generated by extruding an adaptively refined cross-sectional triangulation containing 1830 nodes and 3571 triangles through 20 layers. The network architecture is identical to that used for the cube under torsion, with $\mathbf { f } _ { i } \in \mathbb { R } ^ { 5 }$ and $\tilde { \mathbf { u } } _ { i } \in \mathbb { R } ^ { 3 }$ . The Dirichlet mask is applied component-wise on the clamped face, and the total energy is assembled element-wise over the tetrahedra, with the applied traction integrated over the triangular facets of the loaded face.

This example combines two aspects that have not previously been tested simultaneously. First, the unstructured graph is substantially denser than the cube’s, with 4.1 times as many nodes and 4.5 times as many elements, providing a test of message passing on a locally refined 3D graph. Second, the applied load places the response well beyond the smallstrain regime: the loaded face rotates by $2 6 . 0 5 ^ { \circ }$ , while the rim shear is $\gamma = \phi R / H = 0 . 2 2 7$ approximately an order of magnitude larger than the 0.02-0.05 range for which the linear torsion solution remains applicable. Consequently, the observed warping and second-order axial stresses represent genuine finite-strain efects rather than corrections within the smallstrain approximation.

Figure 19c & d compare the reference and predicted von Mises stress fields on the deformed configuration. The stif inclusion attracts load, with $\sigma _ { \mathrm { v M } }$ reaching approximately 360 Pa along the lobe flanks, while the inclusion core remains at approximately 100 Pa owing to its proximity to the twist axis and correspondingly low shear. The matrix between the lobe tips and the free surface sustains intermediate stresses of approximately 230 Pa. The PI-GNN captures the overall stress distribution, including the lobe peaks and steep gradients near the concave notches, with slightly smoother gradients in the surrounding matrix. The absolute error remains below 10 Pa at 85% of the nodes, corresponding to approximately 3% of the peak stress, and reaches a maximum of 37.9 Pa in localized regions near the lobe tips and re-entrant notches (see Figure 19b). The global deformation is similarly well reproduced: the mean twist of the loaded face is 25.876<sup>◦</sup>, compared with 26.051<sup>◦</sup> for the reference, corresponding to an error of 0.7%, while the peak displacement magnitude is 0.4714 mm, compared with 0.4745 mm.

![](images/8fed10d628f03c83d0d4fbb28380899b999cd0e525065b23f51a4d13c97f7938.jpg)

![](images/182a8af33552ec1840e1da7a477c45a92a601060bc002769a9657a624190a937.jpg)

![](images/e02a20760505e514500f2749374d4cfdaa6028a9f5549186bebabe96cfc15d15.jpg)  
(d)

![](images/a0e4dd55d20b27ed776c089b4b37e487fc3439aa37b717a58f64542bb58a546e.jpg)  
Figure 19: Hyperelastic composite rod with a five-lobed re-entrant inclusion under torsion. (a) Reference configuration showing the stif inclusion (dark red) embedded in the compliant matrix (blue), with torque applied about the rod axis at $z = H$ and the face $z = 0$ clamped. (b) Absolute error in the von Mises stress, $e _ { \sigma _ { \mathrm { v M } } } = | \sigma _ { \mathrm { v M } } ^ { \mathrm { P I - G N N } } - \sigma _ { \mathrm { v M } } ^ { \mathrm { F E M } } |$ , between the PI-GNN prediction and the FE reference. (c) Deformed FE configuration and (d) corresponding PI-GNN prediction, both showing $\sigma _ { \mathrm { v M } }$ . Panels (c) and (d) use the same stress scale, while panel (b) uses an independent error scale. Stresses are reported in Pa, and the deformed configurations are shown at a 1:1 scale.

Table 9 summarizes the error statistics. The displacement magnitude is recovered with $L _ { | | { \bf u } | | } ^ { 2 } = 0 . 8 3 \%$ , while the in-plane components have errors of 0.75% and 0.93%, with $R ^ { 2 } \geq$ 0.999 for both. Thus, the curved free surface and re-entrant interface have little efect on the predicted twist response. The load-carrying shear component $\sigma _ { \theta z }$ has $L _ { \alpha } ^ { 2 } = 5 . 3 3 \%$ $R ^ { 2 } = 0 . 9 7 8$ , and a median signed error of $- 0 . 4 1 \%$ . The von Mises stress shows similar accuracy, with $L _ { \alpha } ^ { 2 } = 5 . 0 0 \%$ and $R ^ { 2 } = 0 . 9 8 0$ . Restricting the evaluation to the interface region yields essentially the same $R ^ { 2 }$ values as the global evaluation, indicating that the near-interface region does not substantially reduce correlation.

Table 9: Summary of error statistics for the composite rod under torsion. The $L _ { \alpha } ^ { 2 }$ values are volume-weighted, except for $u _ { x }$ and $u _ { y } ,$ which are evaluated nodally; the volumeweighted $L ^ { 2 }$ error of the displacement vector is 0.8305%.
<table><tr><td colspan="3"></td><td colspan="2"> $R ^ { 2 }$ </td></tr><tr><td>Field</td><td> $L _ { \alpha } ^ { 2 } ~ ( \% )$ </td><td>Median  $\mathrm { e r r } _ { \alpha }$  (%)</td><td>Global</td><td>Near-field</td></tr><tr><td> $u _ { x }$ </td><td>0.7482</td><td>-0.4203</td><td>0.9999</td><td>0.9999</td></tr><tr><td> $u _ { y }$ </td><td>0.9258</td><td>1.1214</td><td>0.9999</td><td>0.9999</td></tr><tr><td> $\sigma _ { \theta z }$ </td><td>5.3262</td><td>-0.4147</td><td>0.9777</td><td>0.9777</td></tr><tr><td> $\sigma _ { \mathrm { v M } }$ </td><td>5.0012</td><td>-0.5091</td><td>0.9804</td><td>0.9805</td></tr></table>

The energy balance provides an independent check unavailable from pointwise metrics. The converged PI-GNN attains a total potential energy of $\Pi = - 5 . 9 7 8 \times 1 0 ^ { - 2 } \mathrm { k P a m m ^ { 3 } }$ against $- 6 . 0 2 2 \times 1 0 ^ { - 2 } \mathrm { k P a m m ^ { 3 } }$ for the FE reference field evaluated with the same discrete functional, a relative diference of +0.72% within just 30,000 epochs. The prediction, therefore, approaches the reference minimum from above, as required by the principle of minimum potential energy. The model uses the same architecture and learning rate as the preceding examples. It takes 37 237 s to train for 30,000 epochs on a single thread of Intel Xeon(R) Gold 5220R. Subsequently, returns the full field in a single forward pass in 12.42 s, against 672.35 s for the FEniCSx Newton–Raphson solve. While the training time for PIGNN is  55 times FEM solve, the inference time is  0.02 times FEM solve, which shows this method is a promising candidate for parametric surrogate models where multiple inferences need to be made from a single trained model.

## 5. Summary and Outlook

We presented a variational, label-free physics-informed graph neural network (PI-GNN) for two-phase heterogeneous solids under small-strain linear elasticity and finite-strain Neo-Hookean hyperelasticity in 2D and 3D. The solver minimizes the discrete total potential energy on a conforming mesh graph as a single scalar objective, with material heterogeneity carried at the element level. The interface emerges from the stifness contrast between neighboring elements exactly as in FEM, with no penalty term and no prescribed transition width. The discrete energy on $\mathcal { P } _ { 1 }$ elements coincides with the FE Ritz functional, and the converged FE field is the exact minimizer of the training loss. Because u is recovered from the $\mathcal { P } _ { 1 }$ element shape functions rather than by autograd, the trial field carries no $C ^ { 1 }$ requirement—the constraint that obliges coordinate-based energy methods to adopt tanh or sinusoidal activations—a ReLU network is admissible here.

Three sets of results establish the practical content of this construction. First, sweeping the stifness contrast over $E _ { \mathrm { i n c } } / E _ { \mathrm { m a t } } \in [ 1 0 ^ { - 2 } , 1 0 ^ { 2 } ]$ with all settings frozen, the PI-GNN von Mises error remains below 3.58% while a strong-form PINN with fixed transition width reaches 5.58%; the PI-GNN error depends on the magnitude of the contrast but not its sign, whereas the PINN accuracy depends on both. Second, the PI-GNN roughly halves the $\sigma _ { x x }$ error of an energy-based PINN on the perforated-plate benchmarks under discretization transfer [Li et al., 2021]. Third, ablating message passing while preserving the Ritz objective isolates what the graph contributes: at a fixed reduced training budget, the full model attains a 19-43% (increasing with element size) lower error across a mesh sweep, making it a faster-converging model. The ablated version of our model reaches the minima of the loss function only with suficient training budget, as discussed in (Appendix C.2). Message passing, therefore, accelerates the approach to the shared Ritz minimizer rather than lowering it, and the advantage that persists at convergence is especially for the nearinterface displacement field.

The principal limitation is that the framework remains a per-instance solver: each new BVP requires separate training. The accuracy is bounded by the $\mathcal { P } _ { 1 }$ Ritz minimizer, and the message-passing advantage narrows under strong mesh refinement and lenient training budgets. As shown in Section 4.3, the training time exceeds a single FE solve by 55 times. However, when we compare only the inference times, the PIGNN delivers the full field in a single forward pass, 50 times faster than a FEniCSx Newton-Raphson solve. These times are case-specific, and the speed-up of inference depends mainly on the number of elements in the graph. Another systematic feature recurs throughout the suite: stresses are recovered with lower accuracy than displacements because they are determined post hoc via element-wise constitutive evaluation.

Future work will upscale the solver to a parametric surrogate trained on families of geometry, loading, and material variations [Rezaei et al., 2022, Maurizi et al., 2022], a setting in which the unsupervised energy objective eliminates the FE dataset that data-driven surrogates require. The graph architecture also extends to elastoplasticity [Niu et al., 2023] and anisotropic hyperelasticity [Abueidda et al., 2021], and the self-supervised loss is wellsuited to inverse problems such as parameter identification and elasticity imaging [Kamali et al., 2023].

## Data availability

The data and source code required to reproduce the results presented in this study will be openly available at https://github.com/aashay-y/Variational-PIGNN-composites. The repository contains the implementation, input files, and documentation necessary to reproduce the numerical experiments and figures reported in this work.

## Conflict of Interest

The authors declare no conflicts of interest.

## Appendix A. Mesh Convergence study for Kirsch verification

Since the peak boundary stress from a $\mathcal { P } _ { 1 }$ (constant-strain) triangulation is mesh-dependent, we performed a native FEM mesh-convergence study before adopting the FEM fields as the verification benchmark. The benchmark is the classical single-hole problem: a 20 20 plane stress plate $( E = 2 1 0 \mathrm { G P a } , \nu = 0 . 3 )$ with one central circular hole of radius $r = 1$ 1， subjected to uniform traction $\sigma _ { 0 } = 1$ MPa on the right edge (see Section 3.2.1).

For a linear elastic problem, the stress concentration factor (SCF) is independent of E and the load magnitude; the peak stress $\sigma _ { x x }$ equals the SCF. We refined the characteristic element size at the hole boundary, $h _ { \mathrm { h o l e } } = 2 \pi r /$ (number of hole-boundary facets), from 0.03 to 0.15, while holding the far-field grading fixed, such that only the near-hole resolution is varied. The SCF is taken as the seed-averaged nodal peak $\sigma _ { x x } / \sigma _ { 0 }$ value. Here we used the same area-weighted cell nodal projection as used in the analytical/FEM/PI-GNN comparison pipeline. As the unstructured Poisson-disk places interior nodes randomly, each element size is meshed with five independent random number generator seeds and reported as mean one standard deviation. The per-size scatter collapses from $\sim 0 . 0 7 \mathrm { - } 0 . 1 1 \ ( h _ { \mathrm { h o l e } } \geq$ 0.10) to $0 . 0 1 8 - 0 . 0 3 9 \left( h _ { \mathrm { h o l e } } \leq 0 . 0 9 \right)$ , confirming the fine meshes are stable. In Figure A.1, the SCF increases from 2.88 (at $h _ { \mathrm { h o l e } } = 0 . 1 5 )$ to plateau at 3.04, consistent with the Kirsch infinite-plate value of 3.000 and the Heywood finite-width estimate of 3.032 $( d / W = 0 . 1 )$ Every mesh with $h _ { \mathrm { h o l e } } \leq 0 . 0 9$ falls inside the acceptance band $( 3 . 0 \pm 2 \% )$ , while coarser meshes are under-resolved and fall low.

The independent Kirsch check is also satisfied $\sigma _ { y y }  - 1$ at the hole side, and the peak $\sigma _ { x x }$ is located on the hole boundary at the crown. For the PI-GNN training mesh, with $h _ { \mathrm { h o l e } } \approx 0 . 0 2 5$ and 3577 nodes, the computed SCF is 3.04, which lies within the prescribed acceptance band. This confirms that the FE reference solution used for verification is suficiently mesh-converged with respect to the SCF.

![](images/3659e0b02e6b3b0118f10e6f4379fc11c918bfba90d63d383c8f1a205ac91221.jpg)  
Figure A.1: Mesh convergence of the SCF for the single-hole verification example (plane stress linear elasticity; $r = 1$ 2 $\mathrm { p l a t e } = 2 0 , \nu = 0 . 3 .$ , gross traction $\sigma _ { 0 } = 1 \mathrm { M P a } )$ . Markers show the seed-averaged FEM peak nodal $\sigma _ { x x } / \sigma _ { 0 }$ vs. the element size at the hole boundary $h _ { \mathrm { h o l e } }$ (coarse fine). The error bars denote the first standard deviation over five independent mesh seeds. The shaded region is the $3 . 0 \pm 2 \%$ acceptance band [2.94, 3.06]; green circles lie inside the band (acceptable), and red crosses fall outside it. Horizontal lines denote the Kirsch infinite-plate solution (3.0, dashed) and the Heywood finite-width value (3.032, dotted). The SCF converges into the band for $h _ { \mathrm { h o l e } } \leq 0 . 0 9$ , plateauing at  3.04.

## Appendix B. Architecture Ablation Study

An architecture ablation study is conducted to quantify the trade-of between prediction accuracy and model complexity for the proposed PI-GNN model. The ablation study is performed on a stif re-entrant inclusion, while keeping all other training settings fixed, including the learning rate, number of training epochs, material properties, dataset, and random seed (see Section 4.1). Only the network architecture is varied by changing the number of message-passing layers and the width of the hidden neurons.

Each architecture is evaluated using two competing objectives: (i) the area-weighted $L ^ { 2 }$ of the von Mises stress field, which measures prediction accuracy, and (ii) the number of trainable parameters, which serves as a measure of model complexity. Figure B.2 shows the resulting architectures and the corresponding Pareto front. The Pareto front identifies architectures for which no further reduction in prediction error is possible without increasing model complexity. Based on this trade-of, the configuration L4H64 is selected for all application examples, as it provides a favorable balance between accuracy and computational cost.

## Appendix C. Stifness-contrast sweep

The sweep in Section 3.2.2 considers twenty-five logarithmically spaced stifness contrasts spanning $E _ { \mathrm { i n c } } / E _ { \mathrm { m a t } } \in [ 1 0 ^ { - 2 } , 1 0 ^ { 2 } ]$ (with twelve contrasts on either side of the unit-contrast homogeneous limit), obtained by varying $E _ { \mathrm { i n c } }$ while keeping $E _ { \mathrm { m a t } } = 1 5 0 0 \mathrm { M P a } , \nu _ { \mathrm { m a t } } =$ $\nu _ { \mathrm { i n c } } ~ = ~ 0 . 4$ , the geometry, and the applied traction $T \ = \ 0 . 0 2 5 { \mathrm { M P a } }$ fixed. The PINN uses four hidden layers of 64 neurons with tanh activations, a 128 128 collocation grid, a transition width $\delta \ = \ 0 . 0 1$ , and 5000 BFGS iterations. The PI-GNN uses the L4H64 configuration, trained for 50,000 Adam epochs at $\eta = 1 0 ^ { - 3 }$ (see Appendix B). Both solvers are trained from three independent random initializations at each contrast. The sweep, therefore, compares the two methods under fixed configurations rather than isolating the efect of any individual hyperparameter.

![](images/9dc3940fbeebdde298300e0ca273165a9e713527e65f886a92be875653b39b36.jpg)  
Figure B.2: Trade-of between model complexity (trainable parameters) and areaweighted $L ^ { 2 }$ for the PI-GNN architecture sweep. Blue markers denote the evaluated configurations, the solid curve represents the Pareto front, and the selected architecture (L4H64) is labeled with a red star.

Both solvers are evaluated using the same implementation on a common held-out mesh with 3034 nodes and 5944 elements, which is also used for the FE reference solution. Although the PINN is trained without a mesh, its coordinate network is evaluated at the nodes of this mesh, allowing the element areas to be used consistently for both methods in Equation (22). This area weighting is important because the element areas vary by a factor of 168, with the smallest elements concentrated near the material interface, where the errors are also largest. For the PI-GNN at a contrast of 100, removing the area weights increases the reported relative $L ^ { 2 }$ error from 3.58% to 7.96%. The weighting, therefore, materially afects the reported error, making its consistent use for both methods essential for a meaningful comparison.

## Appendix C.1. The unit contrast limit

For unit contrast $E _ { \mathrm { i n c } } / E _ { \mathrm { m a t } } = 1$ , the PINN error is 0.002%, compared with 0.598% for the PI-GNN. This exceptionally small PINN error is specific to the homogeneous case and follows from its trial-function construction: the prescribed background field already coincides with the exact uniform stress solution, leaving only a vanishing correction for the network to learn. The PI-GNN has no analogous prescribed stress field and must recover the equilibrium state by minimizing the discrete potential energy. Thus, this isolated result does not indicate a general advantage in accuracy for the PINN and is not representative of heterogeneous cases.

The residual PI-GNN error is primarily due to incomplete optimization rather than to spatial discretization. For all three seeds, the stored network solutions have higher discrete energy than the FE solution, confirming that the discrete minimizer has not been reached. The corresponding element-wise error fields are essentially uncorrelated (correlation coeficient $= 0 . 0 1 3 , \ 0 . 0 1 3$ , and 0.026), further indicating optimization noise rather than a systematic spatial error. Consistent with this interpretation, reducing the learning rate to $1 0 ^ { - 4 }$ and selecting the lowest-energy checkpoint reduces the error from 0.598% to 0.368%.

The residual error also exhibits small-scale roughness in the predicted displacement field. A local least-squares plane fit, which preserves linear fields exactly, reduces the von Mises error by 33–44% after one pass and by 71% after eight passes, while leaving the FE solution unchanged. This confirms that part of the residual originates from optimization induced displacement roughness. Such smoothing is appropriate only for this homogeneous case, where the exact solution is linear, and it would not preserve the genuine gradients and stress discontinuities that arise at material interfaces. Therefore, the exceptionally small PINN error at unit contrast and its increasing error with stifness contrast arise from distinct mechanisms rather than a single monotonic diference in solver accuracy.

## Appendix C.2. Comparison of converged solutions

The Section 3.2.3 uses a fixed reduced budget of 5000 epochs since the objective was to compare the convergence speed and not the converged solutions. Since the models minimize the same functional, they can be compared directly once the optimization has converged. Here the same reduced model setting (L4H32, 17,250 parameters, $\eta = 1 0 ^ { - 3 } )$ but trained to 30,000 Adam epochs. The mesh is similar; fixed interface-conforming mesh (3034 nodes, 5944 elements) and for a fixed contrast $E _ { \mathrm { i n c } } / E _ { \mathrm { m a t } } = 1 0$

![](images/4776d8de8982c6415443d93000ae8821e2a9333c6c1d3c8c780545a004b0390c.jpg)  
Figure C.3: the total potential Π descends faster for PI-GNN in the first 3000 epochs. It reaches a Π value that the mesh-based PINN only attains near 30,000 epochs. The ablated model curve then slowly closes the gap.

The converged energies are $\Pi = - 5 . 8 8 5 \times 1 0 ^ { - 7 }$ (PI-GNN, epoch 29,997) and Π = $- 5 . 8 6 9 \times 1 0 ^ { - 7 }$ (mesh-based PINN, epoch 29,818). The FE reference $\Pi _ { \mathrm { F E M } } = - 5 . 8 8 9 \times 1 0 ^ { - 7 }$ $\mathrm { M P a } \mathrm { m m } ^ { 2 }$ . Therefore, we can consider that the models reach the minima after suficient epochs, but PIGNN converges faster.

Comparing error metrics for converged fields we get, $L _ { \sigma _ { \mathrm { v M } } } ^ { 2 } = 4 . 6 0 \%$ versus 4.64% so the advantage seen at 5000 epochs has closed. Displacement retains a gap of $L _ { \mathrm { u } } ^ { 2 } = 0 . 2 3 9 \%$ versus 0.709% while the interface band 0.250% versus 0.837% and near-field $R _ { u _ { x } } ^ { 2 } = 0 . 9 9 9 7$ versus 0.9985.

## References

A. Henkes, H. Wessels, R. Mahnken, Physics informed neural networks for continuum micromechanics, Computer Methods in Applied Mechanics and Engineering 393 (2022) 114790.

T. Kirchdoerfer, M. Ortiz, Data-driven computational mechanics, Computer Methods in Applied Mechanics and Engineering 301 (2016) 338–360.

M.-S. Go, H.-K. Noh, J. H. Lim, Real-time full-field inference of displacement and stress from sparse local measurements using physics-informed neural networks, Mechanical Systems and Signal Processing 224 (2025) 112009.

M. Maurizi, C. Gao, F. Berto, Predicting stress, strain and deformation fields in materials and structures with graph neural networks, Scientific Reports 12 (2022) 21834.

F. Xia, G. Shi, Z. Gao, J. Li, S. Xu, D. Ruan, Predicting deformation and stress-strain behaviour of lattice truss structures under compression using dual graph neural network, Composite Structures (2025) 119557.

A. Sanchez-Gonzalez, J. Godwin, T. Pfaf, R. Ying, J. Leskovec, P. Battaglia, Learning to simulate physical systems with graph networks, in: International Conference on Machine Learning, PMLR, 2020, pp. 8459–8468.

Y. Zhao, H. Li, H. Zhou, H. R. Attar, T. Pfaf, N. Li, A review of graph neural network applications in mechanics-related domains, Artificial Intelligence Review 57 (2024) 315.

X.-P. Zhou, K. Feng, MPNN based graph networks as learnable physics engines for deformation and crack propagation in solid mechanics, International Journal of Solids and Structures 291 (2024) 112695.

H. Wang, F. Rhiana, A. Shi, Z. Lu, M. Li, T. Wang, T. H. Nguyen, A. Lin, Physics-informed graph neural networks: Predicting the composite mechanical behavior of clay-infilled concrete structures with morphed honeycomb configurations, Engineering Structures 343 (2025) 121267.

M. Raissi, P. Perdikaris, G. E. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations, Journal of Computational Physics 378 (2019) 686–707.

E. Haghighat, M. Raissi, A. Moure, H. Gomez, R. Juanes, A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics, Computer Methods in Applied Mechanics and Engineering 379 (2021) 113741.

S. Cuomo, V. S. Di Cola, F. Giampaolo, G. Rozza, M. Raissi, F. Piccialli, Scientific machine learning through physics–informed neural networks: Where we are and what’s next, Journal of Scientific Computing 92 (2022) 88.

H. Hu, L. Qi, X. Chao, Physics-informed neural networks (pinn) for computational solid mechanics: Numerical frameworks and applications, Thin-Walled Structures 205 (2024) 112495.

X. Ren, X. Lyu, Mixed form based physics-informed neural networks for performance evaluation of twophase random materials, Engineering Applications of Artificial Intelligence 127 (2024) 107250.

J. Bai, T. Rabczuk, A. Gupta, L. Alzubaidi, Y. Gu, A physics-informed neural network technique based on a modified loss function for computational 2d and 3d solid mechanics, Computational Mechanics 71 (2023) 543–562.

E. Samaniego, C. Anitescu, S. Goswami, V. M. Nguyen-Thanh, H. Guo, K. Hamdia, X. Zhuang, T. Rabczuk, An energy approach to the solution of partial diferential equations in computational mechanics via machine learning: Concepts, implementation and applications, Computer Methods in Applied Mechanics and Engineering 362 (2020) 112790.

V. M. Nguyen-Thanh, X. Zhuang, T. Rabczuk, A deep energy method for finite deformation hyperelasticity, European Journal of Mechanics-A/Solids 80 (2020) 103874.

J. N. Fuhg, N. Bouklas, The mixed deep energy method for resolving concentration features in finite strain hyperelasticity, Journal of Computational Physics 451 (2022) 110839.

Z.-M. Huang, L.-X. Peng, Geometrically nonlinear bending analysis of laminated thin plates based on classical laminated plate theory and deep energy method, Composite Structures 344 (2024) 118314.

A. D. Jagtap, G. E. Karniadakis, Extended physics-informed neural networks (xpinns): A generalized spacetime domain decomposition based deep learning framework for nonlinear partial diferential equations, Communications in Computational Physics 28 (2020).

A. D. Jagtap, E. Kharazmi, G. E. Karniadakis, Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward and inverse problems, Computer Methods in Applied Mechanics and Engineering 365 (2020) 113028.

A. K. Sarma, S. Roy, C. Annavarapu, P. Roy, S. Jagannathan, Interface pinns (i-pinns): A physicsinformed neural networks framework for interface problems, Computer Methods in Applied Mechanics and Engineering 429 (2024) 117135.

E. Kharazmi, Z. Zhang, G. E. Karniadakis, hp-vpinns: Variational physics-informed neural networks with domain decomposition, Computer Methods in Applied Mechanics and Engineering 374 (2021) 113547.

S. Rezaei, A. Harandi, A. Moeineddin, B.-X. Xu, S. Reese, A mixed formulation for physics-informed neural networks as a potential solver for engineering problems in heterogeneous domains: Comparison with finite element method, Computer Methods in Applied Mechanics and Engineering 401 (2022) 115616.

N. Zhang, K. Xu, Z.-Y. Yin, K.-Q. Li, Y.-F. Jin, Finite element-integrated neural network framework for elastic and elastoplastic solids, Computer Methods in Applied Mechanics and Engineering 433 (2025) 117474.

W. Xiong, X. Long, S. P. A. Bordas, C. Jiang, The deep finite element method: A deep learning framework

integrating the physics-informed neural networks with the finite element method, Computer Methods in Applied Mechanics and Engineering 436 (2025) 117681.

C. Wu, C. Liu, Y. Guo, X. Guo, DFENN: A penalty-free variational framework coupling finite elements and neural networks via interface condensation, Journal of the Mechanics and Physics of Solids 215 (2026) 106703.

S. Rezaei, R. Najian Asl, S. Faroughi, M. Asgharzadeh, A. Harandi, R. Najafi Koopas, G. Laschet, S. Reese, M. Apel, A finite operator learning technique for mapping the elastic properties of microstructures to their mechanical deformations, International Journal for Numerical Methods in Engineering 126 (2025) e7637.

H. Gao, M. J. Zahr, J.-X. Wang, Physics-informed graph neural galerkin networks: A unified framework for solving pde-governed forward and inverse problems, Computer Methods in Applied Mechanics and Engineering 390 (2022) 114502.

J. He, D. Abueidda, S. Koric, I. Jasiuk, On the use of graph neural networks and shape-function-based gradient computation in the deep energy method, International Journal for Numerical Methods in Engineering 124 (2023) 864–879.

D. Dalton, D. Husmeier, H. Gao, Physics-informed graph neural network emulation of soft-tissue mechanics, Computer Methods in Applied Mechanics and Engineering 417 (2023) 116351.

T. W¨urth, N. Freymuth, C. Zimmerling, G. Neumann, L. K¨arger, Physics-informed MeshGraphNets (PI-MGNs): Neural finite element solvers for non-stationary and nonlinear simulations on arbitrary meshes, Computer Methods in Applied Mechanics and Engineering 429 (2024) 117102.

B. Feng, X.-P. Zhou, The novel physics-enhanced graph neural network for phase-field fracture modelling, Computer Methods in Applied Mechanics and Engineering 446 (2025) 118284.

Y. Hu, K. Fan, X. Li, J. Luo, J. Cui, Z. Huang, Physics-informed graph neural network for the thermoelastic post-buckling optimization of laminated composite shells, Computer Methods in Applied Mechanics and Engineering 449 (2026) 118593.

F. Brumand-Poor, M. Trautmann, K. Schmitz, Advancing deformation calculation: a physics-informed deep graph learning framework for hyperelastic materials, Advanced Modeling and Simulation in Engineering Sciences 12 (2025) 20.

M. R. Guevara Garban, Y. Chemisky, M. Cl´ement, E. Pruli´ere, Physics-informed graph neural networks<sup>´</sup> to reconstruct local fields considering finite strain hyperelasticity, International Journal for Numerical Methods in Engineering 126 (2025) e70193.

P. Virtanen, R. Gommers, T. E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, E. Burovski, P. Peterson, W. Weckesser, J. Bright, et al., Scipy 1.0: fundamental algorithms for scientific computing in python, Nature Methods 17 (2020) 261–272.

D. Dalton, H. Gao, D. Husmeier, Emulation of cardiac mechanics using graph neural networks, Computer Methods in Applied Mechanics and Engineering 401 (2022) 115645.

I. A. Baratta, J. P. Dean, J. S. Dokken, M. Habera, J. Hale, C. N. Richardson, M. E. Rognes, M. W. Scroggs, N. Sime, G. N. Wells, Dolfinx: the next generation fenics problem solving environment (2023).

P. R. Amestoy, I. S. Duf, J.-Y. L’Excellent, J. Koster, A fully asynchronous multifrontal solver using distributed dynamic scheduling, SIAM Journal on Matrix Analysis and Applications 23 (2001) 15–41.

W. Li, M. Z. Bazant, J. Zhu, A physics-guided neural network framework for elastic plates: Comparison of governing equations-based and energy-based approaches, Computer Methods in Applied Mechanics and Engineering 383 (2021) 113933.

A. Kamali, M. Sarabian, K. Laksari, Elasticity imaging using physics-informed neural networks: Spatial discovery of elastic modulus and poisson’s ratio, Acta Biomaterialia 155 (2023) 400–409.

S. Niu, E. Zhang, Y. Bazilevs, V. Srivastava, Modeling finite-strain plasticity using physics-informed neural network and assessment of the network performance, Journal of the Mechanics and Physics of Solids 172 (2023) 105177.

D. W. Abueidda, Q. Lu, S. Koric, Meshless physics-informed deep learning method for three-dimensional solid mechanics, International Journal for Numerical Methods in Engineering 122 (2021) 7182–7201.