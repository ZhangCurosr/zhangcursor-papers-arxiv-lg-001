# Data-driven discovery of intrinsic Lie point symmetries of diferential equation systems

Ilker Gurcan<sup>1</sup>

Kadir Ustun<sup>2</sup>

Ramazan Gokberk Cinbis<sup>1,3</sup>

<sup>1</sup>Department of Computer Engineering, Middle East Technical University, Ankara, Türkiye <sup>2</sup>ASELSAN Inc., Ankara, Türkiye

<sup>3</sup>ROMER Robotics and AI Center, Middle East Technical University, Ankara, Türkiye

## Abstract

Discovering continuous symmetries directly from diferentialequation data can reveal geometric structure for model reduction and scientific machine learning without symbolic determining equations or a closed-form representation of the governing system. We formulate intrinsic Lie point-symmetry discovery as an operator-theoretic problem on sampled jet-space data. Admissible generators are identified from a sampled determining operator and separated from redundant, solution-dependent, and geometrically trivial directions. The framework combines prolongation, sequential subspace restrictions, and a fixed nonlinear dictionary linear in trainable coeficients. A rank-stabilization preflight test diagnoses systems whose represented intrinsic symmetry algebra does not stabilize under dictionary refinement. On a controlled diferential-equation benchmark with an analytically known 11- dimensional represented intrinsic symmetry algebra, the method recovers the complete represented generator set, reproduces its Lie-bracket structure to numerical precision, and approximates a known generator beyond the training domain. Here we show that explicit intrinsic continuous-symmetry generators can be recovered from data obtained from exact or approximate diferential systems.

## 1 INTRODUCTION

Symmetry has long played a central role in both mathematics and the physical sciences. A landmark development was Galois’s characterization of the solvability of polynomial equations by radicals in terms of the structure of their associated permutation groups, now known as Galois groups; this connection became foundational to Galois theory and modern field theory [1, Part IV]. Meanwhile, a wide range of physical phenomena can be described mathematically by systems of diferential equations (DEs). Motivated in part by Galois’s ideas, Sophus Lie sought an analogous theory for DEs and developed continuous transformation groups as a systematic means of studying, simplifying, and integrating them [2]. However, the advanced machinery for both Lie groups and diferential systems was developed by Élie Cartan [3, 4]. When admitted symmetries are known, they can reduce the efective dimensionality or diferential complexity of a system, for example by lowering the order of an ordinary diferential equation (ODE) or reducing the number of independent variables of a partial diferential equation (PDE), thereby transforming the original problem into a more tractable reduced system [5, Example 2.65]. Beyond dimensional and order reduction, symmetries can also reveal invariant quantities and structures carrying direct physical meaning. For variational systems, for example, Noether’s theorem establishes a fundamental correspondence between continuous symmetries and conservation laws: invariance under time translations, spatial translations, and rotations is associated with conservation of energy, linear momentum, and angular momentum, respectively [6].

Standard infinitesimal Lie-symmetry analysis begins with the infinitesimal invariance condition, reviewed in Section 2. In the classical approach, a candidate symmetry generator is first prolonged to the required jet order and then restricted to the equation manifold. The resulting symbolic expression is subsequently split with respect to the unconstrained jet-fiber coordinates, yielding an overdetermined system of linear PDEs for the unknown infinitesimal coeficients of the candidate generator [5, 7]. The resulting system is known as the determining system of the given DE system. A first limitation of this approach is that it operates on an explicit symbolic representation of the governing equations and therefore cannot be applied directly to data sampled from the corresponding equation manifold. Moreover, systematic treatment of complicated determining systems may involve exact diferential-elimination and involutive-completion algorithms [8], which may be viewed as diferential analogues of Gaussian elimination in linear algebra. Such algorithms are naturally suited to diferential-polynomial or rational systems. Non-polynomial determining systems can still be simplified symbolically, but their treatment may require auxiliary variables, additional analytic assumptions, heuristic simplification, symbolic integration, or generalized separation [9]. Finally, the size and complexity of the determining system can grow rapidly as the numbers of independent variables, dependent variables, equations, or diferential orders increase. The resulting symbolic expression growth, hidden integrability conditions, and possible case distinctions can make the analysis substantially more dificult.

More structural symbolic approaches seek to extract information directly from an involutive or standard form of the determining system without explicitly integrating it. In particular, the associated initial data can be used to determine the number of arbitrary constants and functions in the local solution space and, consequently, the dimension or functional freedom of the admitted symmetry algebra [10]. Although this avoids the need to obtain closed-form expressions for the infinitesimal coeficients, such methods still require an explicit symbolic representation of the governing equations. Moreover, the resulting initial-data representation does not by itself provide an explicit symmetry vector field that can be evaluated throughout a prescribed region. To obtain such a regional representation, the infinitesimal coeficients must be reconstructed, for example through local Taylor expansions generated from the involutive system and its initial data [11], through symbolic integration when available, or through an appropriate numerical continuation of the locally specified symmetry data. The reconstruction becomes increasingly burdensome as the amount of free initial data grows. This distinction is particularly important for the present work, whose intended downstream applications require readily evaluable symmetry generators over sampled regions, including their eventual use in the construction of invariant G-coframes [12].

On the machine-learning (ML) side, symmetry has also become increasingly important for understanding and improving learned representations. Mallat interprets deep convolutional neural-network (CNN) representations through multiscale contractions and the progressive linearization of local or hierarchical symmetry transformations, thereby reducing irrelevant variability while preserving discriminative information [13]. More generally, incorporating previously known symmetries into neural architectures or learning constraints can improve sample eficiency and generalization [14, 15]. In particular, equivariant architectures exploit prescribed group actions to restrict the admissible hypothesis space, allowing the learned model to respect transformations that are known a priori to preserve the underlying problem structure.

In parallel, emerging methods for representing and solving diferential-equation systems have become increasingly data-driven. Neural ordinary diferential equation (Neural ODE) models parameterize an unknown continuoustime vector field by a neural network and obtain the corresponding trajectories through numerical integration [16], in contrast to conventional residual networks, whose layerwise transformations define a discrete-depth evolution [17]. Physics-informed neural networks (PINNs), on the other hand, parameterize an approximate solution u˜<sub>θ</sub> and train the learnable parameters θ by penalizing the governing diferential-equation residual together with prescribed initial and/or boundary conditions [18]; their use of neural representations is supported by the universal approximation capabilities of neural networks [19]. Koopman operator theory provides a complementary operator-theoretic description of nonlinear dynamical systems through the linear evolution of observable functions on the state space [20]. Originally introduced by Bernard Koopman for Hamiltonian systems in 1931 [21], the exact Koopman operator is generally infinitedimensional, motivating modern data-driven approaches that seek finite-dimensional observable subspaces or embeddings in which the dynamics can be represented approximately, or in special cases exactly, by linear evolution [20, 22].

Continuous symmetries have also recently been incorporated into data-driven methods for diferential equations. In many existing approaches, the relevant Lie group or Lie algebra is assumed to be known in advance and is subsequently encoded into the learning architecture, training data, or optimization constraints [23–25]. Of particular relevance to the present work are recent methods that extend beyond enforcing prescribed symmetries to discovering previously unspecified symmetries from models or dynamical-system data [26– 29]. LieStoNet targets projectable Lie point symmetries of stochastic diferential equations by first learning neural surrogates for the drift and difusion and then jointly learning generators under determining-equation and Lie-algebraic regularizers; its associated Fokker–Planck formulation also removes the infinite-dimensional additive solution sector structurally from the learned generator class [28]. LieDyn-Net similarly learns multiple Lie point-symmetry generators jointly from spatiotemporal ODE/PDE data through diferentiable neural surrogates, combining prolonged infinitesimalinvariance and finite-flow conditions with Lie-algebraic constraints [29]. These developments demonstrate growing interest in learning continuous symmetry structure directly from dynamical data, while also highlighting that the computational formulation depends strongly on the target system class and on how admissibility and algebraic structure are imposed.

The present work takes a diferent route. Rather than first reconstructing a dynamical surrogate and then learning a jointly constrained generator basis, we formulate the determining problem directly on sampledjet-space data associated with the diferential system. The sampled determining operator is used to construct a finite represented admissible space before nonlinear optimization begins; within the linear-DE classes treated here, the additive solution-dependent sector is removed at the represented-space level, and the remaining intrinsic directions are recovered sequentially under global novelty and geometric nontriviality restrictions. In this sense, the structural exclusion used for the Fokker–Planck superposition sector in LieStoNet and the represented additive-sector removal developed here share the same practical principle— a large solution-dependent symmetry sector should be removed structurally rather than left to compete with the target directions through soft optimization—but they realize that principle in diferent problem settings.

![](images/fe3361e9d9bb62c416f46bb99515613567acb96e3c910c2bcad1290258c06048.jpg)  
Figure 1: Several solution curves of $\Delta ^ { ( 1 ) }$ in (1) related by its rotational symmetry. The remaining solution curves are obtained by rotating the particular solution shown in light blue.

To illustrate the potential benefit of discovering symmetries directly from data, consider the following nonlinear firstorder ODE in the first jet bundle [30]:

$$
\Delta ^ { ( 1 ) } \left( x , u , u _ { x } \right) = u _ { x } - \frac { u ^ { 3 } + x ^ { 2 } u - u - x } { x ^ { 3 } + u ^ { 2 } x + u - x } = 0 .\tag{1}
$$

At first glance, (1) is neither separable nor suggestive of an obvious simplifying change of coordinates. If only jet-space data sampled from its equation manifold were available, identifying such a transformation would be even less direct. Fortunately, the equation admits SO(2) as a point-symmetry group. Under the polar-coordinate transformation

$$
x = r \cos \theta , \qquad u = r \sin \theta ,\tag{2}
$$

where $( r , \theta )$ form local coordinates on the total space $E$ and r is locally taken as the transformed independent variable, (1) becomes

$$
\overline { { \Delta ^ { ( 1 ) } } } \left( r , \theta _ { r } \right) = \frac { d \theta } { d r } - \frac { 1 } { r ( 1 - r ^ { 2 } ) } = 0 .\tag{3}
$$

A crucial observation is that rotations in the original $( x , u )$ coordinates become translations in the transformed angular coordinate θ. Indeed, the infinitesimal rotational generator

$$
v = - u \partial _ { x } + x \partial _ { u }\tag{4}
$$

is straightened to $\partial _ { \theta }$ in polar coordinates. Consequently, the transformed equation is independent of θ: the group coordinate has become cyclic. This is the local manifestation of symmetry reduction by a one-parameter group and, for the present first-order ODE, reduces the problem to a quadrature. Moreover, the originally nonseparable equation becomes explicitly separable in the symmetry-adapted coordinates.

Thus, knowledge of a single admitted symmetry converts a comparatively complicated diferential relation into a substantially simpler representation for analytical, numerical, or learning-based treatment. The natural question motivating the present work is then: what if such a useful symmetry were not known a priori and only sampled diferential-system data were available?

The limitations of the preceding approaches, together with the motivating example above, capture the central objective of this work: data-driven recovery ofthe intrinsic Lie point symmetries of a DE system from sampled data. The term intrinsic, as used here, is not intended as a canonical definition from the existing literature; its precise meaning is introduced in Section 2. A distinguishing feature of the proposed framework, developed in Section 3, is its explicit use of the jet-space formalism, which provides the natural geometric setting for DEs and their prolongations [31, Chapter 9]. Within this setting, the objective is to recover a finite representation of the intrinsic Lie algebra ${ \mathfrak { g } } _ { \mathrm { i n t } }$ directly from sampled jets. Our earlier numerical experiments indicated that imposing all admissibility, nontriviality, solutionindependence, and novelty requirements simultaneously leads to a dificult multi-objective optimization problem. We therefore adopt an operator-theoretic, feasible-space-first strategy: rather than asking an optimizer to discover admissible symmetry data from an unconstrained parameterization, we first construct a finite-dimensional admissible search space and subsequently optimize only within that space. Successive restrictions exclude directions associated with sampled geometric triviality, additive solution-dependence, and previously discovered generators, thereby progressively isolating admissible intrinsic candidates.

The structure of the admissible generator space also depends strongly on the class of diferential system under consideration. As discussed in Section $^ { 6 , }$ such structural information can be incorporated directly into the chosen finite representation, substantially simplifying the generator model. At the same time, the proposed framework is not intended to provide complete symmetry discovery for every class of DE systems, and we therefore make its present applicability boundaries explicit. In particular, even within the current practical scope, some systems may possess infinite-dimensional intrinsic symmetry algebras arising from arbitrary functional freedom. In the formal theory of DEs, such freedom is characterized through involutive completion, Cartan characters, and ultimately the Cartan–Kähler framework [31, Chapters 11 and 15]. A fully data-driven analogue of these involutivity tests has not yet been established here. Instead, we introduce a conservative rank-stabilization preflight (RSP) test: over a nested sequence of increasingly rich finite dictionaries, we monitor the dimension of the represented non-additive admissible symmetry space. If this dimension stabilizes persistently under successive refinements, the sequential discovery stage is allowed to proceed.

If stabilization fails, the algorithm terminates without making a completeness claim for the intrinsic symmetry algebra of the supplied diferential system.

Finally, the recovered generators are subjected to several complementary numerical validations. First, we test whether the admissible discovery space is exhausted once the known dimension of the benchmark intrinsic algebra has been reached. Second, we evaluate whether the discovered generators reproduce closure under the Lie-bracket operation to numerical precision. Unlike recent joint-learning approaches that impose Lie-bracket closure during symmetry discovery [28, 29], closure is deliberately treated here only as a post-recovery validation. In a sequential setting, the generators recovered at an intermediate stage can span a proper subset of the target algebra that is not itself closed; enforcing closure prematurely can therefore exclude legitimate represented directions, bias the search toward a smaller closed subalgebra, or induce premature termination. C develops the corresponding recovery-path obstruction in detail. Third, we impose a stronger global representation test: does a known analytic intrinsic symmetry generator of the benchmark system lie in the global linear span over R of the discovered generators throughout the sampled region? Together, these tests assess not only whether individual candidates satisfy the sampled infinitesimal symmetry conditions, but also whether the recovered collection behaves as a coherent finite-dimensional representation of the target intrinsic Lie algebra.

Before concluding this section, we summarize the organization of the remainder of the paper. Section 2 establishes the notation and mathematical prerequisites used throughout the manuscript. Readers already familiar with diferential geometry, Lie groups, and Cartan’s method of equivalence may skim this section primarily for notation before proceeding to the subsequent development. Section 3 presents the core of the proposed methodology. There, we formulate the optimization requirements associated with recovering the intrinsic symmetry algebra g<sub>int</sub> and develop the operatortheoretic approach that replaces the resulting multi-objective formulation with a substantially simpler optimization problem posed within an admissible search space. Section 4 discusses the construction of our Kolmogorov–Arnold Network (KAN)-inspired spline-based dictionary, explains why the so-called “nontriviality test” must remain loss-based rather than being imposed entirely through linear space restrictions, describes the implementation and limitations of the RSP test, and establishes sequential stopping criteria concluding that the admissible represented algebra has been exhausted. Section 5 presents the most recent numerical results of the framework, including (i) the benchmark DE system and the generic random shell sampling procedure, (ii) reconstruction accuracy of the target g<sub>int</sub>, (iii) the speed– tail ratios of the discovered generators, (iv) approximation of a known generator through a global constant-coeficient span fit, and (v) the locality of both the infinitesimal symmetry-condition quality and the reconstruction accuracy. Section 6 discusses the practical applicability of the framework, its present limitations, and the structural properties of DE systems underlying these boundaries. Finally, Section 7 summarizes the principal findings of this study and outlines the research directions that follow naturally from the current state of the proposed architecture.

## 2 MATHEMATICAL PRELIMINARIES AND PROBLEM FORMULATION

The purpose of this section is not to reiterate material already established in the literature, but rather to introduce the mathematical ingredients required to formulate the research problem. We refer the reader to the cited literature for more comprehensive treatments of the underlying theory. This section also establishes the mathematical notation used throughout the paper. Proofs of the theorems, lemmas, and propositions introduced in the manuscript are provided in D. Furthermore, a “Table of Symbols” for reader convenience is included in G.

In the first subsection, we formulate a DE system within the jetformalism, since the associated jet bundle provides the local geometric setting in which the research problem is posed. We then introduce Lie point transformations and their prolongations to jet space, together with the relation between the corresponding Lie group of continuous transformations and its Lie algebra, identified with the tangent space at the identity element e. These two subsections provide the mathematical foundation for stating the central infinitesimal criterion, namely the infinitesimal symmetry condition, in the third subsection, where the notion of a Lie point symmetry is formally introduced together with the associated determining equations. The following subsection distinguishes a solution section of the underlying fiber bundle from its prolonged solution graph and from the equation manifold in jet space. The prolongation and projection maps are introduced there to clarify the geometric distinction between sampling a particular solution graph and sampling the equation manifold more generically. We subsequently formalize what is meant in this work by an intrinsic symmetry generator and distinguish the intrinsic symmetry sector from solution-dependent symmetry sectors. These intrinsic generators are the smooth infinitesimal generators that constitute the principal objects to be recovered in the present study. Finally, we provide in A a concise introduction to Cartan’s equivalenceframework and formal integrability, both to place the RSP test within its broader geometric context and to motivate the future research direction discussed in Section 7. The following five subsections, however, contain all mathematical prerequisites required to understand the proposed discovery framework.

Geometric picture in $J ^ { ( 1 ) } ( \pi )$ for the ODE $u _ { x } = x u$

![](images/79fd825e2366d5268584bc930b9665b6fa7961e4b2106fa6344b22f34eb7c222.jpg)  
Figure 2: The equation manifold $\begin{array} { r l } { \Delta ^ { ( 1 ) } } & { { } = } \end{array}$ $\{ ( x , u ; u _ { x } ) \mid u _ { x } - x u = 0 \}$ in $J ^ { ( 1 ) } ( \pi )$ , together with a solution section $u = f ( x )$ and its first jet lift $x \mapsto j ^ { 1 } i _ { f } ( x )$

## 2.1 JET-SPACE GEOMETRY AND DIFFERENTIAL EQUATIONS

Let X denote the base space, locally coordinatized by the independent variables $\mathbf { x } = \left( x ^ { 1 } , \ldots , x ^ { p } \right)$ , and let π : $E \to \mathbf { X }$ be a fiber bundle whose fibers are locally coordinatized by the dependent variables $\mathbf { u } = \left( u ^ { 1 } , \ldots , u ^ { q } \right)$ . The total space E is the space in which the graphs of solutions to a given DE system are represented. Locally, E admits the usual productcoordinate representation $\mathbf { X } \times \mathbf { U }$ , with points denoted by the coordinate pair $( \mathbf { x } , \mathbf { u } )$ . Accordingly, a local solution section is represented as $\Gamma _ { f } : = ( \mathbf { x } , \mathbf { f } ( \mathbf { x } ) )$ , or equivalently by $\mathbf { u } = \mathbf { f } ( \mathbf { x } )$ . Assuming suficient smoothness so that mixed partial derivatives commute, we denote derivatives of a dependent variable $u ^ { \alpha }$ by

$$
u _ { x ^ { I } } ^ { \alpha } : = \frac { \partial ^ { | I | } u ^ { \alpha } } { \partial x ^ { i _ { 1 } } \cdot \cdot \cdot \partial x ^ { i _ { k } } } , \quad I = \left( i _ { 1 } , \ldots , i _ { k } \right) , \quad | I | = k ,\tag{5}
$$

where I is a multi-index specifying the independent coordinates with respect to which diferentiation is performed. The corresponding quantities $f _ { x ^ { I } } ^ { \alpha }$ denote the partial derivatives of the local representative $\ddot { \mathbf { f } } : \mathbf { X }  \mathbf { U }$ of a solution section. These derivative variables will serve as coordinates on the jet spaces introduced below.

The formal definition of a jet bundle is based on the equivalence of local Taylor expansions, up to a prescribed order s, of smooth local sections at a fixed base point $\mathbf { x } \in \mathbf { X } \left[ 3 2 \right.$ Chapter 2].

Definition 1 (Jet bundle). Let

$$
f ^ { \alpha } ( { \bf x } ) = \sum _ { 0 \le | J | \le s } \frac { u _ { x ^ { J } } ^ { \alpha } } { J ! } \left( { \bf x } - { \bf x } _ { 0 } \right) ^ { J } + R ^ { \alpha } ( { \bf x } , { \bf x } _ { 0 } )\tag{6}
$$

denote the order-s Taylor expansion of a smooth scalar component $f ^ { \alpha }$ of a local section about a fixed point $\mathbf { x } _ { 0 } \in \mathbf { X }$ where

$$
u _ { x ^ { J } } ^ { \alpha } : = \left. \frac { \partial ^ { | J | } f ^ { \alpha } } { \partial x ^ { J } } \right| _ { \mathbf { x } = \mathbf { x } _ { 0 } }\tag{7}
$$

are its derivative coeficients and $R ^ { \alpha } ( { \bf x } , { \bf x } _ { 0 } )$ denotes the corresponding remainder. For a vector-valued local representative f, let

$$
\mathbf { u } _ { k } : = \left\{ \mathbf { u } _ { I } : | I | = k \right\} , \quad \mathbf { u } _ { ( s ) } : = \left\{ \mathbf { u } _ { 0 } , \mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { s } \right\} ,\tag{8}
$$

where $\mathbf { u } _ { 0 } \equiv \mathbf { u }$ and $\mathbf { u } _ { k }$ collects all derivative coordinates of total order k for all dependent-variable components.

Two smooth local sections f and g are said to be equivalent to order s at $\mathbf { x } _ { 0 } ,$ , denoted by $\mathbf { f } \sim _ { s } \mathbf { g }$ at x<sub>0</sub>, if

$$
\left. { \frac { \partial ^ { | I | } f ^ { \alpha } } { \partial x ^ { I } } } \right| _ { \mathbf { x } = \mathbf { x } _ { 0 } } = \left. { \frac { \partial ^ { | I | } g ^ { \alpha } } { \partial x ^ { I } } } \right| _ { \mathbf { x } = \mathbf { x } _ { 0 } }\tag{9}
$$

for every $\alpha = 1 , \ldots , q$ and every multi-index I satisfying $0 \leq | I | \leq s .$ Equivalently, their coordinate sets $\mathbf { u } _ { ( s ) }$ coincide at x<sub>0</sub>. The corresponding equivalence class is

$$
[ \mathbf { f } ] _ { ( s ) } \big \vert _ { \mathbf { x } _ { 0 } } = \{ \mathbf { g } \mid \mathbf { g } \sim _ { s } \mathbf { f } \ a t \ \mathbf { x } _ { 0 } \} .\tag{10}
$$

The order-s jet bundle $J ^ { ( s ) } ( \pi )$ is the space of all such equivalence classes over all $\mathbf { x } \in \mathbf { X }$ . Each element $[ \mathbf { f } ] _ { ( s ) } \big | _ { \mathbf { x } }$ is called an s-jet ofa local section at x.

In local coordinates, $J ^ { ( s ) } ( \pi )$ is coordinatized by

$$
z ^ { ( s ) } = \left( \mathbf { x } , \mathbf { u } , \mathbf { u } _ { I } \right) , \quad 1 \leq | I | \leq s .\tag{11}
$$

Since mixed partial derivatives are assumed to commute, $u _ { x ^ { i _ { 1 } } \cdots x ^ { i _ { k } } } ^ { \alpha }$ is unchanged under permutations of the diferentiation indices. We therefore retain only one coordinate for each symmetric multi-index, equivalently taking $i _ { 1 } \leq \cdots \leq i _ { k }$ The order-k derivative coordinates consequently have the local structure of $S ^ { k } ( T _ { \mathbf { x } } ^ { * } \mathbf { X } )$ for each dependent-variable component. Hence, the number of derivative coordinates of order k for each component is

$$
{ \binom { p + k - 1 } { k } } ,\tag{12}
$$

and the local dimension of the order-s jet bundle [5, Chapter 2] is

$$
\dim J ^ { ( s ) } ( \pi ) = p + q { \binom { p + s } { s } } .\tag{13}
$$

The jet bundle carries the canonical projections

$$
\pi _ { 0 } ^ { s } : J ^ { ( s ) } ( \pi )  E , \quad \pi ^ { s } : J ^ { ( s ) } ( \pi )  { \bf X } ,\tag{14}
$$

with $\pi ^ { s } = \pi \circ \pi _ { 0 } ^ { s } $ . More generally, for $0 \leq r < s ,$ , the projection

$$
\pi _ { r } ^ { s } : J ^ { ( s ) } ( \pi )  J ^ { ( r ) } ( \pi )\tag{15}
$$

forgets all derivative coordinates of order greater than r.

Within the local representation in (11), the dependentvariable and derivative coordinates are treated as independent jet-space coordinates; they are constrained by derivative relations only after restriction to a prolonged section or another submanifold carrying the corresponding diferential relations, as discussed in Solution Graphs and Generic Sampling on the Equation Manifold. This local jet-coordinate representation is the one employed throughout the proposed computational framework.

With the jet formalism established above, we can now define an order-s DE system, which is naturally represented as a subset of $J ^ { ( s ) } ( \pi )$

Definition 2 (Diferential equation system). Let $z ^ { ( s ) } \in$ $J ^ { ( s ) } ( \pi )$ denote an s-jet and let ${ \Delta ^ { ( s ) } } : J ^ { ( s ) } ( \pi ) \to \mathbb { R } ^ { m }$ be a smooth vector-valued diferential function, whose m components represent the constituent diferential equations. An order-s DE system is defined by the zero locus

$$
\Sigma ^ { ( s ) } = \left\{ z ^ { ( s ) } \in J ^ { ( s ) } ( \pi ) \mid \Delta ^ { ( s ) } \left( z ^ { ( s ) } \right) = 0 \right\} .\tag{16}
$$

Here, s denotes the highest derivative order efectively appearing in $\Delta ^ { ( s ) }$ . We refer to $\Sigma ^ { ( s ) }$ as the equation manifold whenever the regularity condition stated below is satisfied.

Let $M \subset \mathbf { X }$ be an open subset and let

$$
i _ { f } : M \to E , \quad \mathbf { x } \longmapsto \Gamma _ { f } | _ { \mathbf { x } } = ( \mathbf { x } , \mathbf { f } ( \mathbf { x } ) ) ,\tag{17}
$$

be the smooth local section corresponding to $\Gamma _ { f } ,$ , satisfying $\pi \circ i _ { f } = \operatorname { i d } _ { M }$ . The order-s jet prolongation of this section is the smooth map

$$
j ^ { s } i _ { f } : M \to J ^ { ( s ) } ( \pi ) , \quad { \bf x } \longmapsto \left( { \bf x } , { \bf f } ( { \bf x } ) , { \bf f } _ { I } ( { \bf x } ) \right) ,\tag{18}
$$

where $1 \leq | I | \leq s .$ . The section $i _ { f } ,$ or equivalently its local representative $\mathbf { u } = \mathbf { f } ( \mathbf { x } )$ , is a solution ofthe DE system on M if

$$
\Delta ^ { ( s ) } \left( \left[ j ^ { s } i _ { f } \right] ( \mathbf { x } ) \right) = \mathbf { 0 } , \quad \forall \mathbf { x } \in M .\tag{19}
$$

Consequently, its prolonged solution graph,

$$
\Gamma _ { f } ^ { ( s ) } : = \left\{ \left[ j ^ { s } i _ { f } \right] ( { \bf x } ) \mid { \bf x } \in { \cal M } \right\} ,\tag{20}
$$

satisfies $\Gamma _ { f } ^ { ( s ) } \subseteq \Sigma ^ { ( s ) }$ . Since $\pi ^ { s } \circ j ^ { s } i _ { f } = \operatorname { i d } _ { M }$ , the map $j ^ { s } i _ { f }$ is a smooth embedding [33, Chapter 4], and $\Gamma _ { f } ^ { ( s ) }$ is therefore an embedded p-dimensional submanifold of $J ^ { ( s ) } ( \pi )$ Moreover, the restricted projection $\pi ^ { s } | _ { \Gamma _ { \epsilon } ^ { ( s ) } } : \Gamma _ { f } ^ { ( s ) } \to M$ is a f difeomorphism whose inverse is $j ^ { s } i _ { f }$

Throughout this work, we restrict attention to regular portions of the equation set by assuming that $\Delta ^ { ( s ) }$ has maximal rank m on its zero locus $\Sigma ^ { ( s ) }$ . Under this assumption, $\Sigma ^ { ( s ) }$ is a regular embedded submanifold of $J ^ { ( s ) } ( \pi )$ of codimension m [5, Theorem 1.13]. Hence,

$$
\dim \Sigma ^ { ( s ) } = \dim J ^ { ( s ) } ( \pi ) - m = p + q { \binom { p + s } { s } } - m .\tag{21}
$$

Accordingly, samples belonging to singular or rank-deficient portions of the equation set are excluded from the present formulation. Additional nondegeneracy conditions relevant to the infinitesimal symmetry criterion are introduced in Infinitesimal Symmetry Condition and Determining Equations.

Figure 2 summarizes the geometric concepts introduced in this subsection through the example of the first-order ODE $\Delta ^ { ( 1 ) } \left( x , u , u _ { x } \right) = u _ { x } - x u = 0 .$ . The solution section $u = f ( x )$ is prolonged to the curve $x \mapsto j ^ { 1 } i _ { f } ( x )$ in the first-order jet space, and, when f satisfies the ODE, this prolonged solution graph lies entirely within the equation manifold $\Sigma ^ { ( 1 ) }$

## 2.2 LIE POINT TRANSFORMATIONS AND PRO-LONGATION

Definition 3 (Lie group). A diferentiable manifold is a topological space that locally admits smooth coordinate representations in Euclidean space, thereby permitting the usual tools of diferential calculus to be defined intrinsically [34, Section 3.4]. A Lie group G is a diferentiable manifold endowed with a group structure for which the (left) multiplication and inversion maps

$$
\begin{array} { c } { { m _ { G } : G \times G \to G , \quad ( g , h ) \longmapsto g \cdot h , } } \\ { { \ i _ { G } : G \to G , \quad g \longmapsto g ^ { - 1 } , } } \end{array}\tag{22}
$$

are smooth. Here, $e \in G$ denotes the identity element, satisfying $e \cdot g = g \cdot e = g$ and $g \cdot g ^ { - 1 } = g ^ { - 1 } \cdot g = e f o r$ every $g \in G$

Besides multiplication on itself, a Lie group may act smoothly on another geometric space. In this work, $G$ acts on the total space E through the smooth left action

$$
\Phi : G \times E  E , \quad ( g , z ) \longmapsto g \cdot z ,\tag{23}
$$

which satisfies $e \cdot z = z$ and $( g \cdot h ) \cdot z = g \cdot ( h \cdot z )$ for all $g , h \in G$ and $z \in E$ . Such actions provide the geometric setting for Lie point transformations; for example, Olver considers the rotation group SO(2) acting on the $( x , u ) .$ space of a scalar ODE [5, Example 2.21]. The same abstract group may admit diferent realizations through its actions on diferent spaces [35, Chapter 3, Section II]. We use the symbol “·” for both the group multiplication in G and the action of G on $E$ whenever the intended operation is clear from the participating objects.

For symmetry analysis of DEs, it is generally suficient for the action to be defined only locally. Accordingly, the global action in (23) may be restricted to an appropriate open neighborhood of $e \times E$ in $G \times E$ , yielding a local transformation group whenever the identities of the groupaction in (23) hold wherever the corresponding compositions are defined [5, Section 1.2].

![](images/28865b515d25ffe29eb5292d1c84931a16e602202587f507fa4a1b6a59132b14.jpg)  
Figure 3: A schematic relation between a Lie algebra, its associated Lie group, and the induced action on E. The exponential map generates the one-parameter subgroup $g _ { v } ( \epsilon ) = \exp ( \epsilon v ) \in G$ , whose action produces the local flow $\mathrm { F l } _ { v } ^ { z } ( \epsilon ) = \exp ( \epsilon v ) \cdot z$ , tangent to $v | _ { z } \in T _ { z } E$ at ϵ = 0.

Definition 4 (Lie algebra). A finite-dimensional real Lie algebra g is a real vector space equipped with a bilinear and skew-symmetric binary operation

$$
[ \cdot , \cdot ] : { \mathfrak { g } } \times { \mathfrak { g } } \longrightarrow { \mathfrak { g } } , \quad ( v _ { i } , v _ { j } ) \longmapsto [ v _ { i } , v _ { j } ] ,\tag{24}
$$

called the Lie bracket, which satisfies the Jacobi identity [36, Section 3.1]:

$$
\left[ v _ { i } , [ v _ { j } , v _ { k } ] ] + [ v _ { j } , [ v _ { k } , v _ { i } ] ] + [ v _ { k } , [ v _ { i } , v _ { j } ] ] = 0 . \right.\tag{25}
$$

For a Lie group G, we adopt the right-invariant realization of its Lie algebra. Let $R _ { g } : G \to G$ denote right multiplication, $R _ { g } ( h ) = h \cdot g .$ A smooth vector field $v \in { \mathfrak { X } } ( G )$ is $r i g h t -$ invariant if

$$
d ( R _ { g } ) _ { h } \left( v \vert _ { h } \right) = \left. v \right| _ { h \cdot g } , \quad \forall g , h \in G .\tag{26}
$$

The space of all right-invariant vector fields on G is closed under the ordinary vector-field Lie bracket and forms a finite-dimensional Lie algebra [6, Section 29.1.2]. Evaluation at the identity, $\mathrm { e v } _ { e } ( v ) = v | _ { e }$ , provides a vector-space isomorphism between this space and $T _ { e } G ;$ hence, we iden-$t i f y \ { \mathfrak { g } } \cong T _ { e } G .$ . Conversely, every tangent vector $v | _ { e } \in T _ { e } G$ extends uniquely to a right-invariant vector field through $v | _ { q } = d ( R _ { g } ) _ { e } \left( v | _ { e } \right)$ . In local coordinates $\left( g ^ { 1 } , \ldots , g ^ { r } \right)$ on G, where r = dim G = dim g, such a vector field may be expressed as

$$
v _ { i } | _ { g } = \sum _ { a = 1 } ^ { r } \eta _ { i } ^ { a } ( g ) \partial _ { g ^ { a } } \in T _ { g } G .\tag{27}
$$

Thus, $i f \left\{ v _ { 1 } , \ldots , v _ { r } \right\}$ is a basis of g, its Lie-algebraic structure is encoded by the constants $c _ { i j } ^ { k } \in \mathbb { R }$ defined through

$$
[ v _ { i } , v _ { j } ] = \sum _ { k = 1 } ^ { r } c _ { i j } ^ { k } v _ { k } .\tag{28}
$$

The smooth left action Φ in (23) induces a corresponding infinitesimal realization of g in ${ \mathfrak { X } } ( E )$ . For each $z \in E ,$ , let $\Phi _ { z } : G \to E$ denote the orbit map $\Phi _ { z } ( g ) = g \cdot z$ . We identify the realization of $v \in { \mathfrak { g } }$ on E through $v | _ { z } = d ( \Phi _ { z } ) _ { e } \left( v | _ { e } \right)$

and, when no ambiguity can arise, use the same symbol v for the right-invariant vector field on G and its induced vector field on E. In the local bundle coordinates $( \mathbf { x } , \mathbf { u } )$ , the latter takes the form

$$
v = \sum _ { i = 1 } ^ { p } \xi ^ { i } \left( \mathbf { x } , \mathbf { u } \right) \partial _ { x ^ { i } } + \sum _ { \alpha = 1 } ^ { q } \varphi ^ { \alpha } \left( \mathbf { x } , \mathbf { u } \right) \partial _ { u ^ { \alpha } } ,\tag{29}
$$

where $\xi ^ { i } , \varphi ^ { \alpha } \in C ^ { \infty } ( E )$ . We refer to $\big \{ \xi ^ { i } \big \}$ and $\{ \varphi ^ { \alpha } \}$ as the base and fiber coeficients of v, respectively. The proposed framework does not reconstruct the Lie group G itself; rather, it recovers the infinitesimal generators represented as vector fields in ${ \mathfrak { X } } ( E )$ . Accordingly, the local representation in (29) is the principal form used throughout this work. Since these generators act on the independent and dependent variables, rather than directly on derivative coordinates, the local transformations they generate on E are Lie point transformations; their induced action on jet space is obtained through prolongation below.

The connection between a Lie group G and its Lie algebra g is provided by a special map called the exponential map [5, Section 1.4]:

$$
\exp : { \mathfrak { g } } \to G , \quad v \longmapsto \exp ( v ) .\tag{30}
$$

The exponential map is a local difeomorphism from a neighborhood of $0 \in { \mathfrak { g } }$ onto an open neighborhood of $e \in G$ Hence, every $g \in G$ suficiently close to e can be written as $g = \exp ( v )$ for some $v \in { \mathfrak { g } }$ . More generally, assuming ${ \mathfrak { g } } = \operatorname { s p a n } _ { \mathbb { R } } \left\{ v _ { 1 } , \dots , v _ { r } \right\}$ , every element g of the connected identity component $G _ { 0 } \subseteq G$ can be expressed as a finite product

$$
g = \exp \left( \epsilon ^ { 1 } v _ { i _ { 1 } } \right) \cdot \cdot \cdot \exp \left( \epsilon ^ { k } v _ { i _ { k } } \right) ,\tag{31}
$$

for some finite $k \in \mathbb N ,$ , where $\epsilon ^ { j } \in \mathbb { R }$ and $1 \leq i _ { j } \leq r$ for $j = 1 , \ldots , k \ [ 5 , ( 1 . 4 0 ) ]$ . Thus, although exp ${ \mathfrak { s g } } \to G$ need not itself be globally surjective, finite products of oneparameter subgroups generated by a basis of g recover the identity component $G _ { 0 }$

We exploit this correspondence in Section 4, where neural ODE solvers [16] are employed to approximate the flows generated by elements of g on $E .$ For $v \in { \mathfrak { g } }$ and $z \in E$ , the corresponding integral curve is

$$
\begin{array} { r } { \mathrm { F l } _ { v } ^ { z } ( \epsilon ) : = \Phi _ { z } ( \exp ( \epsilon v ) ) , \quad \mathrm { F l } _ { v } ^ { z } : I _ { z }  E , } \end{array}\tag{32}
$$

where $I _ { z } \subseteq \mathbb { R }$ is an interval containing $\epsilon = 0$ on which the local action is defined. Its tangent at the identity satisfies $d / _ { d \epsilon } \Phi _ { z } \left( \exp ( \epsilon v ) \right) \vert _ { \epsilon = 0 } \left. v \right. _ { z }$ , with $v \big | _ { z }$ given by (29) [5, Section 1.3]. Hence, $\mathrm { F l } _ { v } ^ { z }$ is simply the integral curve through z generated by the infinitesimal action of v. This correspondence between finite transformations and their infinitesimal generators yields the two equivalent formulations of the symmetry condition introduced in Infinitesimal Symmetry Condition and Determining Equations.

The diagram in Figure 3 summarizes the relationships introduced thus far. An infinitesimal generator $v \in { \mathfrak { g } }$ is mapped through the exponential map to a one-parameter subgroup of $G ,$ denoted by $g _ { v } ( \epsilon )$ . The action of $g _ { v } ( \epsilon )$ on $E _ { \mathrm { { : } } }$ , as defined in (23), then induces the flow $\mathrm { F l } _ { v } ^ { z } ( \epsilon )$ through each $z \in E$

For consistency with the right-invariant realization of g adopted above, we distinguish the left action of G on $E$ from the inverse right action of $G$ on itself. In particular, for $g , h \in G$ , let the latter be given by $h \longmapsto h \cdot g ^ { - 1 }$ . A smooth map $\rho : E \to G$ is then G-equivariant with respect to these two actions if

$$
\rho \left( g \cdot z \right) = \rho ( z ) \cdot g ^ { - 1 } , \quad g \in G , z \in E .\tag{33}
$$

Equivalently, $\rho$ intertwines the left action on $E$ with the inverse right action on G. This is the convention underlying the right regularization of a group action in the movingframe framework of Fels and Olver, where the lifted action on $G \times E$ takes the form $( h , z ) \longmapsto ( h \cdot g ^ { - 1 } , g \cdot z )$ [37, Section 3].

This convention also aligns the right-invariant Lie algebra on $G$ with the infinitesimal generators induced by the left action on E. In particular, Fels and Olver show that the correspondence between a right-invariant element $v \in { \mathfrak { g } }$ and its induced infinitesimal generator on the acted-upon manifold preserves the Lie-algebraic structure, while Theorem 3.10 expresses the associated group-direction diferentials of the lifted invariants directly in terms of these infinitesimalgenerator coeficients and the right Maurer–Cartan forms [37, Theorem 3.10]. We adopt the same left-action/right-invariant convention here both to avoid unnecessary sign changes in the infinitesimal realization in Section 3, to remain compatible with the moving-coframe formulation revisited in $\mathrm { A } ,$ , and to provide a consistent foundation for the future research trajectory discussed in Section 7.

Lastly, we introduce the prolongation of an infinitesimal generator $v \in { \mathfrak { X } } ( E )$ to a vector field on $J ^ { ( s ) } ( \pi )$ , in analogy with the prolongation of local sections introduced in (18). The infinitesimal action of v on a local section $\Gamma _ { f } \subset E$ must be “compatible” with the induced action on its prolonged graph $\Gamma _ { f } ^ { ( s ) } \subset J ^ { ( s ) } ( \pi )$ . This compatibility underlies the general prolongation formula of Olver [5, Theorem 2.36], which defines

$$
\begin{array} { r l r } {  { \operatorname { p r } ^ { ( s ) } ( v ) : = v + \sum _ { \alpha = 1 } ^ { q } \displaystyle \sum _ { 1 \leq | J | \leq s } \varphi _ { J } ^ { \alpha } \frac { \partial } { \partial u _ { x ^ { J } } ^ { \alpha } } , } } \\ & { } & { \ \varphi _ { J } ^ { \alpha } : = D _ { J } ( \varphi ^ { \alpha } - \sum _ { i = 1 } ^ { p } \xi ^ { i } u _ { x ^ { i } } ^ { \alpha } ) + \sum _ { i = 1 } ^ { p } \xi ^ { i } u _ { x ^ { J } x ^ { i } } ^ { \alpha } , } \end{array}\tag{34}
$$

where $D _ { J } ~ = ~ D _ { x ^ { i _ { 1 } } } \cdot \cdot \cdot D _ { _ { x ^ { i } | J | } }$ denotes the iterated total derivative associated with the symmetric multi-index $J .$ For a smooth diferential function $P ( \mathbf { x } , \mathbf { u } _ { ( s ) } )$ whose highest derivative order is s, the total derivative with respect to $x ^ { i }$ is

$$
D _ { x ^ { i } } P = \frac { \partial P } { \partial x ^ { i } } + \sum _ { \alpha = 1 } ^ { q } \sum _ { 0 \leq | J | \leq s } u _ { x ^ { J } x ^ { i } } ^ { \alpha } \frac { \partial P } { \partial u _ { x ^ { J } } ^ { \alpha } } ,\tag{35}
$$

where the case $| J | = 0$ corresponds to $u ^ { \alpha }$ itself [5, Proposition 2.35]. Thus, $\mathrm { p r } ^ { ( s ) } ( v )$ extends the infinitesimal action of v from the coordinates $( \mathbf { x } , \mathbf { u } )$ to all derivative coordinates required through order s.

## 2.3 INFINITESIMAL SYMMETRY CONDITION AND DETERMINING EQUATIONS

Jet-Space Geometry and Diferential Equations and Lie Point Transformations and Prolongation provide the mathematical machinery required to formulate the symmetry condition for a DE system.

Definition 5 (Symmetry condition). Let $z ^ { ( s ) } = \big ( \mathbf { x } , \mathbf { u } _ { ( s ) } \big )$ denote an s-jet and let $\Delta ^ { ( s ) } ( z ^ { ( s ) } ) = 0$ be an order-s DE system consisting of m equations. A Lie group G acting locally on $E$ is a symmetry group of the DE system if its prolonged action preserves the equation manifold $\Sigma ^ { ( s ) }$ Equivalently, for every $z ^ { ( s ) } \in \Sigma ^ { ( s ) }$ and every $g \in G f o r$ which the prolonged action is defined,

$$
\begin{array} { c c c c } { { z ^ { ( s ) } \in \Sigma ^ { ( s ) } } } & { { \Longrightarrow } } & { { g ^ { ( s ) } \cdot z ^ { ( s ) } \in \Sigma ^ { ( s ) } , } } & { { \forall g \in G , } } \end{array}\tag{36}
$$

where $g ^ { ( s ) }$ denotes the prolonged transformation induced by g on $J ^ { ( s ) } ( \pi )$

Under the regularity and local-solvability assumptions required by the infinitesimal invariance criterion, preservation $o f \Sigma ^ { ( s ) }$ by the connected identity component $G _ { 0 }$ is equivalently characterized by

$$
\begin{array} { r } { \operatorname { p r } ^ { ( s ) } ( v ) \left. \left[ \Delta _ { \nu } ^ { ( s ) } \right] \right. _ { \Sigma ^ { ( s ) } } = 0 , \qquad } \\ { \qquad \quad \nu = 1 , \dots , m , \quad \forall v \in { \mathfrak { g } } , } \end{array}\tag{37}
$$

which is known as the infinitesimal symmetry condition $I 5 ,$ Theorem 2.31]. Geometrically, (37) states that every prolonged infinitesimal generator $\mathrm { p r } ^ { ( s ) } ( v )$ is tangent to the equation manifold $\Sigma ^ { ( \bar { s } ) }$

Since $\Delta ^ { ( s ) }$ has maximal rank m along $\Sigma ^ { ( s ) }$ , the tangency condition can locally be expressed equivalently in the multiplier form

$$
\mathrm { p r } ^ { ( s ) } ( v ) \left[ \Delta _ { \nu } ^ { ( s ) } \right] = \sum _ { \mu = 1 } ^ { m } F _ { \nu } ^ { \mu } \left( z ^ { ( s ) } \right) \Delta _ { \mu } ^ { ( s ) } ,\tag{38}
$$

for suitable smoothfunctions $F _ { \nu } ^ { \mu }$ and $\nu = 1 , \ldots , m$ . Thus, the prolonged infinitesimal action preserves the diferential constraints modulo the constraints themselves.

The criterion in (37) is the symmetry condition employed throughout the proposed framework. Its converse implication, namely that such infinitesimal generators integrate to a local symmetry group, requires the standard nondegeneracy and local-solvability assumptions on the DE system [5, Section 2.6].

In Section 1, we noted that the classical approach to recovering g applies symbolic-manipulation algorithms to a system known as the determining system. After substituting the prolonged generator into (37) and, when appropriate, eliminating dependent highest-order derivatives using the DE system itself, the resulting symbolic expressions can be split with respect to the remaining functionally independent jet coordinates $\left\{ { \bf u } _ { I } \right\}$ , where $1 \leq | I | \leq s$ . Equating the corresponding independent coeficients to zero yields an overdetermined linear homogeneous PDE system for the unknown infinitesimal coeficients ξ(x, u) and φ(x, u) introduced in (29). We refer the reader to [5, Section 2.4] and [38, Section 3.2.4] for representative examples in which determining systems are constructed and subsequently solved.

## 2.4 SOLUTION GRAPHS AND GENERIC SAM-PLING ON THE EQUATION MANIFOLD

In Jet-Space Geometry and Diferential Equations, we introduced two geometric objects that are central to the proposed framework: the equation manifold $\Sigma ^ { ( s ) } \subset J ^ { ( s ) } ( \pi )$ and the prolonged solution graph $\Gamma _ { f } ^ { ( s ) } \subset J ^ { ( s ) } ( \pi )$ . Although every solution of the DE system determines a prolonged graph satisfying $\Gamma _ { f } ^ { ( s ) } \subseteq \Sigma ^ { ( s ) }$ , the two manifolds generally encode fundamentally diferent information. Since $\Gamma _ { f } ^ { ( s ) }$ is parameterized only by the $p$ independent variables, dim $\Gamma _ { f } ^ { ( s ) } = p .$ . In contrast, $\Sigma ^ { ( s ) }$ retains all jet-space degrees of freedom compatible with the DE constraints and, under the maximal-rank assumption, has the dimension given in (21). Consequently, $\Gamma _ { f } ^ { ( s ) }$ is a proper positive-codimension submanifold of $\Sigma ^ { ( s ) }$ whenever

$$
\mathrm { c o d i m } _ { \Sigma ^ { ( s ) } } \left( \Gamma _ { f } ^ { ( s ) } \right) = \mathrm { d i m } \Sigma ^ { ( s ) } - p > 0 .\tag{39}
$$

In particular, under (39), $\Gamma _ { f } ^ { ( s ) } \subsetneq \Sigma ^ { ( s ) }$

A particular $\Gamma _ { f } ^ { ( s ) }$ does not merely satisfy the DE constraints.

Its jet coordinates additionally satisfy the relations imposed by the specific solution ${ \bf f ( x ) } ; { \bf e . g . } u ^ { \alpha } = f ^ { \alpha } ( { \bf x } )$ and $u _ { x ^ { i } } ^ { \alpha } = f _ { x ^ { i } } ^ { \alpha } ( \mathbf { x } )$ . These relations constitute solution-specific diferential information and may artificially enlarge the apparent null space associated with the sampled infinitesimal symmetry condition in (37), thereby admitting superfluous infinitesimal generators. The mechanism used in the proposed framework to identify and eliminate such generators is introduced in Section 3.

Proposition 1 (Generic Shell Sampling). Let $\Sigma ^ { ( s ) } \subset$ $J ^ { ( s ) } ( \pi )$ be a regular equation manifold satisfying dim $\Sigma ^ { ( s ) } > p ,$ and let $i _ { f } : M \to E$ be any fixed local solution section of $\Delta ^ { ( s ) } \stackrel { \cdot } { = } \mathbf { 0 } ,$ , where $M \subset \mathbf { X }$ is open. Then, its prolonged solution graph

$$
\Gamma _ { f } ^ { ( s ) } = j ^ { s } i _ { f } ( M ) \subset \Sigma ^ { ( s ) }\tag{40}
$$

is an embedded positive-codimension submanifold $o f \Sigma ^ { ( s ) }$ and therefore has zero measure with respect to any smooth volume measure on $\Sigma ^ { ( s ) }$

Consequently, let

$$
\mathbf { S } ^ { \mathrm { o n } } : = \left\{ z _ { l } ^ { ( s ) } \in \Sigma ^ { ( s ) } \right\} _ { l = 1 } ^ { N _ { \mathrm { o n } } } , \quad N _ { \mathrm { o n } } \ge 1 ,\tag{41}
$$

be a finite collection of jet-space samples drawn from a probability distribution $\mu _ { \Sigma } ( s )$ that is absolutely continuous with respect to a smooth volume measure on $\dot { \Sigma ^ { ( s ) } }$ . Then,for everyfixed local solution $f ,$

$$
\mathbb { P } \left( \mathbf { S } ^ { \mathrm { o n } } \subset \Gamma _ { f } ^ { ( s ) } \right) = 0 .\tag{42}
$$

A proof of Proposition 1 is provided in D.1. We emphasize that the proposition does not imply that random sampling prevents samples from lying near a particular solution graph, nor does it imply uniform sampling over the family of all solutions. Rather, it states only that, under an absolutely continuous sampling law, exact concentration on any fixed positive-codimension solution graph occurs with probability zero.

## 2.5 INTRINSIC AND SOLUTION-DEPENDENT SYMMETRY SECTORS

The proposed architecture does not aim to recover the entire abstract Lie symmetry algebra g of a given DE system $\Delta ^ { ( s ) } = \mathbf { 0 }$ . Instead, we target a distinguished symmetry sector, termed the intrinsic symmetry algebra, whose generators capture structural transformations of the DE system without importing a priori information from any particular solution. We emphasize that “intrinsic” is not a canonical classification in classical Lie symmetry theory, but terminology introduced specifically for the purposes of the present work.

![](images/2099c53e394e2853146a62e31105ecae2382a73df41b7a34211976a9fd870639.jpg)  
Figure 4: A trivial generator (left panel) is tangent to every admissible solution graph, whereas a nontrivial generator (right panel) maps at least one admissible solution graph to a distinct solution graph.

Definition 6 (Prolonged characteristic information). Let $v \in { \mathfrak { g } }$ be represented in ${ \mathfrak { X } } ( E )$ as in (29). Its characteristic vector is defined by $I 5 ,$ , Characteristics of Symmetries]

$$
\begin{array} { c } { { \displaystyle { \bf Q } _ { v } = \left[ Q _ { v } ^ { 1 } , \dots , Q _ { v } ^ { q } \right] , } } \\ { { \displaystyle } } \\ { { \displaystyle Q _ { v } ^ { \alpha } \left( { \bf x } , { \bf u } _ { ( 1 ) } \right) : = \varphi ^ { \alpha } - \sum _ { i = 1 } ^ { p } u _ { x ^ { i } } ^ { \alpha } \xi ^ { i } , } } \end{array}\tag{43}
$$

where $\alpha = 1 , \ldots , q .$ Thus, $\mathbf { Q } _ { v }$ is naturally defined on $J ^ { ( 1 ) } ( \pi )$ and, $f o r s \ge 1$ , can be evaluated on $J ^ { ( s ) } ( \pi )$ through the canonical projection $\pi _ { 1 } ^ { s }$

We refer to $\mathbf { E } _ { v } ^ { ( 0 ) } : = \mathbf { Q } _ { v }$ as the zeroth-order characteristic information. Motivated by the characteristic form of the prolongationformula [5, (2.50)–(2.51)], we define the characteristic information available through an s-jet by

$$
\mathbf { E } _ { v } ^ { ( s - 1 ) } : = \left[ \mathbf { Q } _ { v } , D _ { J } \mathbf { Q } _ { v } \right] , 1 \leq | J | < s ,\tag{44}
$$

where all symmetric multi-indices J in the indicated range are included and $D _ { J }$ denotes the total derivative operator introduced in (35). We call $\mathbf { E } _ { v } ^ { ( s - 1 ) }$ theprolonged characteristic information ofv through order $s - 1$

For a Lie point generator, $\mathbf { Q } _ { v }$ depends on jet coordinates only through first order. Consequently, every $D _ { J } \mathbf { Q }$ with $| J | \le s - 1$ can be evaluated using jet information available in ${ \dot { \mathbf { \pi } } } ^ { ( s ) } ( \pi )$ . The choice in (44) therefore incorporates all successive characteristic information accessible at the native order of the DE data without requiring prolongation to $J ^ { ( s + 1 ) } ( \pi )$ . Higher-order characteristic information could be introduced by prolonging the DE system and its sampled jets further, but at additional computational cost.

The characteristic $\mathbf { Q } _ { v }$ measures the infinitesimal deformation of a solution graph modulo motion induced by reparameterization of its independent variables. In particular, $\mathbf { Q } _ { v }$ vanishes along a prescribed solution graph precisely when the corresponding point generator is tangent to that graph. This observation motivates the following DE-level notion of triviality.

Definition 7 (Nontriviality). Let v be a Lie point symmetry generator with characteristic vector $\mathbf { Q } _ { v } .$ . We call v trivial with respect to the DE system if

$$
\mathbf { Q } _ { v } \big \rvert _ { \Gamma _ { f } ^ { ( 1 ) } } \equiv \mathbf { 0 }\tag{45}
$$

for every admissible local solution $f .$ Equivalently, v is tangent to every admissible local solution graph. Otherwise, v is called nontrivial. Figure 4 illustrates this distinction geometrically. This definition is consistent with the classical equivalence ofgeneralized symmetries through their evolutionary representatives and the associated notion of trivial symmetries [5, Section 5.1, Equivalence and Trivial Symmetries].

Moreover, if (45) holdsfor a particular solution $f ,$ then all total derivatives of $\mathbf { Q } _ { v }$ along the prolonged solution also vanish to the available order, and hence $\mathbf { E } _ { v } ^ { ( s - 1 ) } \Big | _ { \Gamma _ { f } ^ { ( s ) } } \equiv \mathbf { 0 } .$

Remark 1. Definition 7 does not imply that a nontrivial generator must be transverse to every admissible solution graph. A genuine nontrivial symmetry may be tangent to, and therefore have vanishing characteristic on, a particular subset of solutions. DE-level triviality requires this tangency property for the entire admissible solution family.

Remark 2. Definition 7 is stated directly over the family ofadmissible local solutions and therefore does not require every point $z ^ { ( s ) }$ ofthefinite-order equation manifold $\Sigma ^ { ( s ) }$ to be realizable as the jet of an actual prolonged solution section. Under the regularity,formal-compatibility, involutivity, and local-existence assumptions discussed in A, the Cartan–Kähler framework ensures local integral manifolds through the relevant ordinary integral elements. On the corresponding solution-realizable regular locus, vanishing of $\mathbf { Q } _ { \imath }$ on every admissible solution is therefore equivalent to $\mathbf { Q } _ { v } | _ { \Sigma ^ { ( s ) } } \equiv \mathbf { 0 } ,$ with $\Sigma ^ { ( s ) }$ understood as the appropriately completed equation manifold. In the absence of these assumptions, Definition 7 provides the less restrictive solution-level formulation adopted in this work.

For a data-driven framework, evaluating only the zerothorder characteristic at finitely many jet samples may provide insuficient evidence of nontriviality. At any $z ^ { ( s ) } \in \Sigma ^ { ( s ) }$ $\mathbf { E } _ { v } ^ { ( s - 1 ) } \Big | _ { z ( s ) } = \mathbf { 0 }$ implies $\mathbf { Q } _ { v } | _ { z ^ { ( s ) } } = \mathbf { 0 }$ because $\mathbf { Q } _ { \imath }$ is itself a component of $\mathbf { E } _ { v } ^ { ( s - 1 ) }$ , whereas the converse need not hold. Consider, for example,

$$
\Delta ^ { ( 2 ) } ( x , u _ { ( 2 ) } ) = u _ { x x } = 0 .\tag{46}
$$

The vector field $v = x \partial _ { x }$ is a Lie point symmetry generator of (46) by (37), with characteristic $Q _ { v } = - x u _ { x } . \mathrm { A t }$ the on-shell jet $z _ { 0 } ^ { ( 2 ) } = ( 0 , 0 , 1 , 0 )$ , one obtains $Q _ { v } \big | _ { z _ { 0 } ^ { ( 2 ) } } = 0$ whereas $D _ { x } Q _ { v } = - u _ { x } - x u _ { x x }$ gives $\left. { \bf E } _ { v } ^ { ( 1 ) } \right| _ { z _ { 0 } ^ { ( 2 ) } } = [ 0 , - 1 ] \ne$ 0. Thus, the prolonged characteristic information can distinguish a pointwise zero of the characteristic from vanishing of the characteristic together with its available diferential information.

To make this distinction explicit, define

$$
\begin{array} { r l } & { \mathcal { Z } _ { E } ( v ) : = \left\{ z ^ { ( s ) } \in \Sigma ^ { ( s ) } \mid \mathbf { E } _ { v } ^ { ( s - 1 ) } \Big \vert _ { z ^ { ( s ) } } = \mathbf { 0 } \right\} , } \\ & { \mathcal { Z } _ { Q } ( v ) : = \left\{ z ^ { ( s ) } \in \Sigma ^ { ( s ) } \mid \mathbf { Q } _ { v } \big \vert _ { z ^ { ( s ) } } = \mathbf { 0 } \right\} . } \end{array}\tag{47}
$$

Since $\mathbf { Q } _ { v }$ is contained in $\mathbf { E } _ { v } ^ { ( s - 1 ) }$ , it follows immediately that $\mathcal { Z } _ { E } ( v ) \subseteq \mathcal { Z } _ { Q } ( v )$ . Section 3 exploits this inclusion to strengthen, rather than render mathematically conclusive, the sampled optimization objective associated with nontriviality.

Nontriviality is central to the symmetry sector targeted by the proposed framework and is also consistent with the future objective of constructing G-coframes from the recovered intrinsic generators and prescribed solution data, as discussed in Section 7.

As discussed in Section 6, certain classes of DE systems within the practical scope of the proposed framework may admit symmetry generators whose coeficients depend explicitly on particular solutions of the underlying DE system. We refer to such generators as solution-dependent generators. We also note that “solution-dependent generator” is not a canonical term in classical Lie symmetry theory.

Definition 8 (Solution-dependent vertical generator). Let $\Delta ^ { ( s ) } = \mathbf { 0 }$ be a DE system and let $v _ { \varphi }$ be one ofits Lie point symmetry generators having vanishing base coeficients; i.e.,

$$
v _ { \varphi } = \sum _ { \alpha = 1 } ^ { q } \varphi ^ { \alpha } \left( \mathbf { x } \right) \partial _ { u ^ { \alpha } } ,\tag{48}
$$

where $\varphi = \left( \varphi ^ { 1 } , \dots , \varphi ^ { q } \right)$ depends only on the independent coordinates x. $I f \varphi$ itself constitutes an admissible local solution of the same DE system, equivalently,

$$
\Gamma _ { \varphi } ^ { ( s ) } \subseteq \Sigma ^ { ( s ) } ,\tag{49}
$$

then $v _ { \varphi }$ is called a solution-dependent vertical generator. Its so-called “verticality” follows from the fact that $v _ { \varphi }$ takes values in the vertical bundle $V E : = \ker ( d \pi ) \subset T E .$

Such generators are excluded from the target symmetry sector for several reasons. First, in the canonical linear case, their flows act by additive translations within the solution family and therefore do not provide the structural transformations sought for the G-coframe construction discussed in Section 7. Second, as shown in Section 3, whenever the underlying linear PDE admits an infinite-dimensional local solution space, these generators constitute an infinite-dimensional Abelian ideal of its Lie symmetry algebra, which cannot be exhaustively represented by a finite computational architecture. Third, their construction requires a priori information about solutions of the DE system. This introduces a conceptual circularity for the proposed architecture, whose eventual purpose is precisely to exploit the recovered intrinsic symmetries to simplify and integrate the given DE system without presupposing its particular solutions. We refer collectively to such unwanted generators as the solution-dependent sector. For linear DE systems, the corresponding family induced by the superposition principle will be termed the additive solution subalgebra in Section 3.

Definitions 7 and 8 provide the necessary ingredients to define the intrinsic symmetry algebra.

Definition 9 (Intrinsic symmetry algebra). Let $\Delta ^ { ( s ) } = \mathbf { 0 }$ be a DE system with Lie symmetry algebra g. A Lie subalgebra ${ \mathfrak { g } } _ { \mathrm { i n t } } \leq { \mathfrak { g } }$ is called an intrinsic symmetry algebra if every nonzero $v \in { \mathfrak { g } } _ { \mathrm { i n t } } ( i )$ is nontrivial, and (ii) is solution-independent.

Remark 3. The presence ofa solution-dependent vertical component does not, by itself, render thefull symmetry generator solution-dependent. In particular, $i f v = v _ { \xi } + v _ { \varphi }$ with $v _ { \xi } \not \equiv { \mathbf { 0 } }$ , then v is not classified as solution-dependent in the sense ofDefinition 8, even $i f v _ { \varphi } ,$ considered separately, belongs to the solution-dependent vertical sector. Accordingly, the term solution-dependent generator is reserved herefor the purely vertical class characterized in Definition 8.

## 3 DATA-DRIVEN SYMMETRY DISCOV-ERY FRAMEWORK

Section 2 established the infinitesimal symmetry condition, nontriviality, solution independence, and the ultimate target of this work: the intrinsic symmetry algebra ${ \mathfrak { g } } _ { \mathrm { i n t } }$ introduced in Definition 9. This mathematical formulation must now be translated into a computational framework operating on sampled jet data.

We first formulate the requirements for recovering ${ \mathfrak { g } } _ { \mathrm { i n t } }$ as a multi-objective optimization problem. However, as discussed in the subsequent subsection, the resulting optimization becomes increasingly dificult to control as these requirements are imposed simultaneously. This motivates our operatortheoretic approach, in which the determining condition and the subsequent admissibility restrictions are made explicit through the underlying algebraic structure. The remainder of this section develops this formulation and its sequential recovery procedure.

## 3.1 INITIAL MULTI-OBJECTIVE OPTIMIZATION FORMULATION

The initial approach was to design an architecture that is entirely optimization-based and relies on a standard neuralnetwork parameterization. The requirements that each successively discovered generator $v _ { i } \in { \mathfrak { g } } _ { \mathrm { i n t } }$ must satisfy are (i) the infinitesimal symmetry condition, (ii) nontriviality, (iii) solution independence, and (iv) global linear independence. We formulate and discuss the corresponding optimization objectives in this order.

In Definition 5, we stated what it means for a transformation group to constitute a symmetry of a given DE system $\Delta ^ { ( s ) } = 0$ . For computational discovery, however, we adopt the infinitesimal symmetry condition in (37) rather than attempting to learn the finite group action in (23) directly. A finite group action must satisfy not only pointwise invariance, but also the nonlinear compatibility conditions associated with group composition, invertibility, and the existence of a local action on an open neighborhood of each $z \in E .$ . Consequently, learning such transformations directly does not naturally expose the linear structure underlying the symmetry problem.

By contrast, the infinitesimal generators form the Lie algebra g, which is a vector space, and the infinitesimal symmetry condition yields an overdetermined linear homogeneous determining system for the coeficient functions of each generator. This formulation therefore permits the use of ordinary calculus and, crucially for the framework developed below, linear-algebraic operations on represented generator spaces. Once a realization of the relevant Lie algebra has been recovered, its generators may be integrated through the exponential map to obtain the corresponding local oneparameter group actions, as described in (31). Thus, for the connected symmetry transformations generated locally from the identity, the infinitesimal formulation retains the grouplevel symmetry information while providing a substantially more tractable computational representation.

The condition (37) converts symmetry discovery into a local diferential condition that can be evaluated directly on jet-space samples. Moreover, the autograd facility of modern ML tools is particularly convenient for computing the prolongation of a candidate generator given in (34). For an on-shell sample $z _ { l } ^ { ( s ) } \in \mathbf { S } ^ { \mathrm { o n } }$ (see (41)), define the determining residual vector via (37)

$$
\mathbf { r } _ { l } ( v ) : = \left( \operatorname { p r } ^ { ( s ) } ( v ) \left[ \Delta _ { \nu } ^ { ( s ) } \right] \Big | _ { z _ { l } ^ { ( s ) } } \right) _ { \nu = 1 } ^ { m } .\tag{50}
$$

A naive optimization objective is then the squared determin-

ing residual

$$
\mathcal { L } _ { \mathrm { s y m } , l } ^ { \mathrm { r a w } } : = \| { \bf r } _ { l } ( v ) \| _ { 2 } ^ { 2 } .\tag{51}
$$

Although (51) directly encodes the infinitesimal symmetry condition, our preliminary experiments showed that its raw magnitude is not geometrically balanced. In particular, it depends both on the local scale of the defining equations and distribution of the prolonged generator coeficients. We therefore introduce two normalization factors.

The first accounts for the local scale of the equation manifold $\Sigma ^ { ( s ) }$ . Since

$$
{ \bf r } _ { l } ( v ) = { \mathrm J } _ { \Delta ^ { ( s ) } } \left( z _ { l } ^ { ( s ) } \right) { \bf b } \left( z _ { l } ^ { ( s ) } \right) .\tag{52}
$$

where b denotes the coordinate representation of $\mathrm { p r } ^ { ( s ) } ( v )$ in (34), the Jacobian $\mathrm { J } _ { \Delta ( s ) }$ represents the local normal information of $\Sigma ^ { ( s ) }$ . We therefore use

$$
\left\| \operatorname { J } _ { \Delta ^ { ( s ) } } \left( z _ { l } ^ { ( s ) } \right) \right\| _ { \mathrm { F } }\tag{53}
$$

as the first normalization factor, where $\Vert \cdot \Vert _ { \mathrm { F } }$ denotes the Frobenius norm. This prevents a large local scaling of the defining equations from disproportionately increasing the determining residual.

The second normalization concerns the prolonged generator itself. Suppose that any $v \in { \mathfrak { g } } _ { \mathrm { i n t } }$ , lifted to the order-s jet bundle, is realized as in (34) inside ${ \mathfrak { X } } \left( J ^ { ( s ) } ( \pi ) \right)$ . We call each subset of prolonged coeficients grouped according to its jet order $0 \leq | J | \leq s \mathrm { \ a }$ jet block. At order zero, we further separate the horizontal and vertical coeficients,

$$
b _ { 0 } ^ { h } : = \{ \xi ^ { i } \} , \quad b _ { 0 } ^ { v } : = \{ \varphi ^ { \alpha } \} ,\tag{54}
$$

while, for $| J | \ge 1 , b _ { | J | }$ collects the prolonged coeficients $\varphi _ { J } ^ { \alpha }$ of the corresponding order. Accordingly, we write

$$
\mathbf { b } : = ( b _ { k } ) _ { k = 1 } ^ { s + 2 } = \left( b _ { 0 } ^ { h } , b _ { 0 } ^ { v } ; b _ { | J | = 1 } , \ldots , b _ { | J | = s } \right) .\tag{55}
$$

The magnitudes of these blocks may difer substantially across jet orders, causing the residual to be dominated by a particular block rather than by the geometric direction represented by the prolonged generator as a whole. To compensate for this imbalance, for each block k we compute an instantaneous RMS scale at training step t,

$$
\begin{array} { r } { \widetilde { s } _ { k } ^ { t } : = \sqrt { \operatorname* { m e a n } \left[ \left( b _ { k } ^ { t } \right) ^ { 2 } \right] + \epsilon } , } \end{array}\tag{56}
$$

where the mean is taken over the corresponding block entries in the current sample batch and $\epsilon > 0$ is a suficiently small numerical regularizer. The running scale is then tracked through the exponential moving average (EMA)

$$
s _ { k } ^ { t }  \alpha s _ { k } ^ { t - 1 } + ( 1 - \alpha ) \tilde { s } _ { k } ^ { t } , \quad \alpha = 0 . 9 ,\tag{57}
$$

with $s _ { k } ^ { 0 }$ initialized from the first sample batch.

Using these running scales, we define the block-normalized magnitude of the prolonged generator at $z _ { l } ^ { ( s ) }$ as

$$
\| \operatorname { p r } ^ { ( s ) } ( v ) \| _ { z _ { l } ^ { ( s ) } } \| _ { \operatorname { b l k } } : = \sum _ { k = 1 } ^ { s + 2 } \frac { \| b _ { k } ( z _ { l } ^ { ( s ) } ) \| _ { 2 } } { ( s _ { k } + \epsilon ) \sqrt { D _ { k } } } ,\tag{58}
$$

where $D _ { k } : = \dim b _ { k }$ . The factor $\sqrt { D _ { k } }$ compensates for the natural $\ell _ { 2 } { \mathrm { - n o r m } }$ growth of a block with its dimension; without this correction, a higher-dimensional jet block could remain dominant even after RMS normalization.

Combining the local normal magnitude in (53) with the block-normalized generator magnitude in (58), we obtain the normalized determining residual

$$
\rho _ { l } ( v ) : = \frac { \mathbf { r } _ { l } ( v ) } { \left. \mathbf { J } _ { \Delta ^ { ( s ) } } \left( z _ { l } ^ { ( s ) } \right) \right. _ { \mathrm { F } } \left. \mathrm { p r } ^ { ( s ) } ( v ) \left( z _ { l } ^ { ( s ) } \right) \right. _ { \mathrm { b l k } } + \epsilon } .\tag{59}
$$

The two factors in the denominator render the residual angle-like: the first normalizes the local normal scale of the equation manifold, while the second normalizes the magnitude of the prolonged generator in a block-balanced metric. In particular, the latter also makes the symmetry objective insensitive to an overall rescaling of a nonzero candidate generator, so reducing the magnitude of v alone cannot improve the normalized determining residual.

For a sample batch ${ \bf S } _ { B } ^ { \mathrm { o n } } ~ \subset ~ { \bf S } ^ { \mathrm { o n } }$ of cardinality B, the infinitesimal-symmetry objective is finally taken as

$$
\mathcal { L } _ { \mathrm { s y m } } : = \frac { 1 } { B } \sum _ { z _ { l } ^ { ( s ) } \in \mathbf { S } _ { B } ^ { \mathrm { o n } } } \left\| \pmb { \rho } _ { l } ( v ) \right\| _ { 2 } ^ { 2 } .\tag{60}
$$

Thus, $\mathcal { L } _ { \mathrm { s y m } }$ penalizes the relative failure of the candidate prolonged vector field to remain tangent to the sampled equation manifold rather than the unnormalized magnitude of the determining residual itself.

Definition 9 requires every nonzero element of ${ \mathfrak { g } } _ { \mathrm { i n t } }$ to be nontrivial in the sense of Definition 7. Accordingly, every candidate accepted as a new direction of the recovered intrinsic algebra must satisfy a corresponding nontriviality requirement. One practical motivation for this restriction is discussed in Section 7, where the discovered intrinsic algebra is proposed for constructing a G-coframe on sampled data associated with a particular local solution section.

For this purpose, we use the prolonged characteristic information $\mathbf { E } _ { v } ^ { ( s - 1 ) }$ introduced in (44), rather than the zeroth-order characteristic $\mathbf { Q } _ { v }$ alone. From the characteristic zero sets defined in (47), we have $\mathcal { Z } _ { E } ( v ) \subseteq \mathcal { Z } _ { Q } ( v )$ . Therefore, $\mathbf { E } _ { v } ^ { ( s - 1 ) }$ provides a stronger finite-sample discriminator: a sample at which $\mathbf { Q } _ { v }$ vanishes may still carry nonzero higher-order characteristic information. Since $\mathbf { Q } _ { v }$ already depends on first-order jet coordinates, evaluating $D _ { J } \mathbf { Q } _ { v }$ for $1 \leq | J | < s$ requires successive total diferentiation only through order $s - 1$ , while using jet coordinates no higher than the native data order s. Thus, the stronger nontriviality diagnostic does not require lifting the data beyond $J ^ { ( s ) } ( \pi )$

There are nevertheless two distinct numerical degeneracies to prevent. First, since ${ \mathfrak { g } } _ { \mathrm { i n t } }$ is a vector space, it necessarily contains the zero element, although Definition 9 imposes nontriviality only on its nonzero elements. Moreover, the normalized symmetry objective in (60) is insensitive to an overall rescaling of a nonzero candidate. A separate magnitude condition is therefore required to prevent optimization from collapsing toward the zero vectorfield. Second, among candidates of visible magnitude, the optimization must discourage directions whose characteristic information vanishes, or becomes negligibly small relative to the prolonged response, on the sampled equation manifold.

For a batch $\mathbf { S } _ { B } ^ { \mathrm { o n } }$ of cardinality B, let

$$
\| v \| _ { B } : = \sqrt { \frac { 1 } { B } \sum _ { z _ { l } ^ { ( s ) } \in \mathbf { S } _ { B } ^ { \mathrm { o n } } } \left\| \mathrm { p r } ^ { ( s ) } ( v ) \big | _ { z _ { l } ^ { ( s ) } } \right\| _ { \mathrm { b l k } } ^ { 2 } } .\tag{61}
$$

We then impose the minimum-magnitude constraint

$$
c _ { \operatorname* { m i n } } : = \tau _ { \operatorname* { m i n } } - \| v \| _ { B } \leq 0 ,\tag{62}
$$

where $\tau _ { \operatorname* { m i n } } > 0$ . This condition forces the candidate generator to retain a numerically visible magnitude.

To detect sampled tangency to the corresponding solution structure, we define the relative prolonged-characteristic response at $z _ { l } ^ { ( s ) }$ by

$$
\mathcal { R } _ { \mathrm { n t r } , l } ( v ) : = \frac { \left\| \mathbf { E } _ { v } ^ { ( s - 1 ) } \Big \vert _ { z _ { l } ^ { ( s ) } } \right\| _ { 2 } } { \left\| \operatorname* { p r } ^ { ( s ) } ( v ) \big \vert _ { z _ { l } ^ { ( s ) } } \right\| _ { \mathrm { b l k } } + \epsilon _ { \mathrm { n t r } } } ,\tag{63}
$$

where $\epsilon _ { \mathrm { n { t r } } } > 0$ is a numerical regularizer. The corresponding inequality constraint is

$$
c _ { \mathrm { n t r } , l } : = \tau _ { \mathrm { n t r } } - \mathcal { R } _ { \mathrm { n t r } , l } ( v ) \leq 0 ,\tag{64}
$$

with $\tau _ { \mathrm { n t r } } > 0$ . The normalization in (63) makes this criterion relative to the magnitude of the prolonged response, rather than imposing an absolute characteristic threshold that depends on the overall scale of v.

The condition (64) should be interpreted as a sampled nontriviality gate rather than a global certification of Definition 7. The use of $\mathbf { E } _ { v } ^ { ( s - 1 ) }$ strengthens the available finite-order evidence through $\mathcal { Z } _ { E } ( v ) \subseteq \mathcal { Z } _ { Q } ( v )$ ; nevertheless, satisfaction of a numerical threshold on finitely many sampled jets does not by itself establish the corresponding property over the entire solution-realizable equation manifold. Stronger conclusions may be drawn under additional regularity, solutionrealizability, and analyticity assumptions, but these are not required by the optimization criterion employed here.

To incorporate the inequality constraints into the optimization, we use the quadratic exterior penalty

$$
\mathcal { P } \left( c \right) : = \frac { 1 } { 2 } \gamma \operatorname* { m a x } \left( 0 , c \right) ^ { 2 } ,\tag{65}
$$

where $\gamma > 0$ is increased during training according to the continuation schedule specified in Section 4. The nontriviality objective over the current batch is therefore

$$
\mathcal { L } _ { \mathrm { n t r } } : = \mathcal { P } \left( c _ { \mathrm { m i n } } \right) + \frac { 1 } { B } \sum _ { z _ { l } ^ { \left( s \right) } \in \mathbf { S } _ { B } ^ { \mathrm { o n } } } \mathcal { P } \left( c _ { \mathrm { n t r } , l } \right) .\tag{66}
$$

For the reasons discussed in Intrinsic and Solution-Dependent Symmetry Sectors, the solution-dependent sector consisting of generators of the form specified in Definition 8 must be excluded from ${ \mathfrak { g } } _ { \mathrm { i n t } }$ . Among the requirements considered in the present optimization formulation, however, our preliminary experiments showed that solution independence is the most dificult to encode reliably through a diferentiable objective, as detailed in Limitations of the Optimization Formulation.

Detecting a solution-dependent generator in the sense of Definition 8 requires three simultaneous conditions. The candidate must be approximately vertical, its vertical coeficients must be approximately independent of the dependent coordinates u, and those coeficients, when interpreted as a local section over x, must approximately satisfy the original DE system. Symbolically, these conditions can be summarized as

$$
\begin{array} { c } { { v \approx v _ { \varphi } , } } \\ { { \displaystyle \displaystyle \frac { \partial \varphi } { \partial { \bf u } } \approx { \bf 0 } \Longrightarrow \varphi \approx \varphi \left( { \bf x } \right) , } } \\ { { \displaystyle \Delta ^ { \left( s \right) } \Big \vert _ { \Gamma _ { \varphi } ^ { \left( s \right) } } \approx { \bf 0 } , \quad \Gamma _ { \varphi } ^ { \left( s \right) } = \left[ j ^ { s } i _ { \varphi } \right] \left( M \right) . } } \end{array}\tag{67}
$$

where $j ^ { s } i _ { \varphi }$ is the order-s prolongation map of a local section in (18) and $M \subset X$ is an open subset. The first two conditions identify whether the candidate lies near the purely vertical, u-independent class considered in Definition $\mathbf { \boldsymbol { 8 } ; }$ only in that regime should satisfaction of the third condition be penalized. The requirement that $v _ { \varphi }$ itself constitute a Lie point symmetry need not be tested again here, since it is imposed simultaneously through the infinitesimal-symmetry objective $\mathcal { L } _ { \mathrm { s y m } }$

We implement the first two decisions through smooth sigmoid gates. For each $z _ { l } ^ { ( s ) } \in \mathbf { S } _ { B } ^ { \mathrm { o n } }$ , let

$$
g _ { \mathrm { v e r } , l } : = \sigma \left( \kappa _ { \mathrm { v e r } } \left( \tau _ { \mathrm { v e r } } - \left. \pmb { \xi } \left( z _ { l } ^ { ( s ) } \right) \right. _ { 2 } \right) \right)\tag{68}
$$

and

$$
g _ { \mathrm { u I n d } , l } : = \sigma \left( \kappa _ { \mathrm { u I n d } } \left( \tau _ { \mathrm { u I n d } } - \left. \frac { \partial \varphi } { \partial \mathbf { u } } \left( z _ { l } ^ { ( s ) } \right) \right. _ { \mathrm { F } } \right) \right) ,\tag{69}
$$

where $\sigma$ denotes the logistic sigmoid, $\tau _ { \mathrm { v e r } } , \tau _ { \mathrm { u I n d } } > 0$ are activation thresholds, and $\kappa _ { \mathrm { v e r } } , \kappa _ { \mathrm { u I n d } } > 0$ control the corresponding gate sharpness. Hence, $g _ { \mathrm { v e r } , l } \approx 1$ when the horizontal component is negligible, while $g _ { \mathrm { u I n d } , l } \approx 1$ when the vertical coeficients exhibit negligible dependence on u.

Once both gates are active, we test whether $\varphi$ behaves as a local solution section. Let

$$
\mathcal { R } _ { \mathrm { d e p } , l } ( v ) : = \frac {  \Delta ^ { ( s ) } ( [ j ^ { s } i _ { \varphi } ] ( \mathbf { x } _ { l } ) )  _ { 2 } } {  \operatorname { p r } ^ { ( s ) } ( v )  _ { z _ { l } ^ { ( s ) } }  _ { \mathrm { b l k } } + \epsilon _ { \mathrm { d e p } } } ,\tag{70}
$$

where $\epsilon _ { \mathrm { d e p } } > 0$ is a numerical regularizer. A small value of $\mathcal { R } _ { \mathrm { d e p } , l }$ indicates that the vertical coeficient, interpreted as an approximately u-independent section, lies close to the solution manifold of the DE system relative to the magnitude of the candidate prolonged response.

The corresponding solution-dependence constraint is

$$
c _ { \mathrm { d e p } , l } : = g _ { \mathrm { v e r } , l } g _ { \mathrm { u I n d } , l } \left( \tau _ { \mathrm { d e p } } - \mathcal { R } _ { \mathrm { d e p } , l } ( v ) \right) \leq 0 ,\tag{71}
$$

where $\tau _ { \mathrm { d e p } } ~ > ~ 0$ . Thus, when either verticality or uindependence is absent, the corresponding gate suppresses the constraint. When both gates are active, however, a suficiently small DE residual yields $c _ { \mathrm { d e p } , l } > 0$ and the candidate is repelled from the detected solution-dependent sector.

Using the quadratic exterior penalty in (65), the solutiondependence loss over a batch of cardinality B is

$$
\mathcal { L } _ { \mathrm { d e p } } : = \frac { 1 } { B } \sum _ { z _ { l } ^ { ( s ) } \in \mathbf { S } _ { B } ^ { \mathrm { o n } } } \mathcal { P } \left( c _ { \mathrm { d e p } , l } \right) .\tag{72}
$$

Remark 4. The construction above is consistent with $R e \cdot$ mark 3. In particular, the general definition of a solutiondependent generator is restricted to the purely vertical class. Accordingly, a mixed generator with $\pm 0$ is not classified as solution-dependent merely because its vertical component is u-independent and satisfies the DE system. Up to the numerical tolerance encoded by $g _ { \mathrm { v e r } , l ; }$ , the verticality gate suppresses $\mathcal { L } _ { \mathrm { d e p } } f o r$ such mixed generators.

The final requirement difers fundamentally from the previously defined objectives because it is not a property of a single candidate in isolation. A finite-dimensional real Lie algebra is a vector space over the field of real constants, not over the ring of smooth functions (see Definition 4). Therefore, for any $w \in { \mathfrak { g } }$

$$
w = \sum _ { i = 1 } ^ { r } c ^ { i } v _ { i } , \quad \mathbf { C } : = [ c ^ { 1 } , \ldots , c ^ { r } ] \in \mathbb { R } ^ { r } ,\tag{73}
$$

where

$$
\mathfrak { g } = \operatorname { s p a n } _ { \mathbb { R } } \left\{ v _ { 1 } , \dotsc , v _ { r } \right\} , \quad \mathrm { d i m } \mathfrak { g } = r .\tag{74}
$$

Equivalently, for every $z \in E$

$$
w | _ { z } = \sum _ { i = 1 } ^ { r } c ^ { i } \left. v _ { i } \right| _ { z } ,\tag{75}
$$

with the same coeficient vector C holding over the entire domain. Hence, linear dependence in g cannot be established from an arbitrary pointwise relation whose coeficients are allowed to vary with z. The corresponding discovery objective must instead test whether one common vector of real constants explains the candidate over a suficiently large collection of evaluation points.

Global linear independence is an ambient vector-field property of the realized generators in ${ \mathfrak { X } } ( E )$ and should therefore not be inferred solely from their values over the constrained sample distribution used to enforce the symmetry condition. In the present optimization formulation, the independence test is evaluated from the zeroth-order generator coeficients $( \xi , \varphi )$ , rather than from their prolonged responses.

Let $\pi _ { 0 } ^ { s } : J ^ { ( s ) } ( \pi ) \to E$ denote the canonical projection onto the zeroth-order bundle. Depending on the structure of the DE system and the projection used to construct $\mathbf { S } ^ { \mathrm { o n } }$ , two situations may arise. First,

$$
\pi _ { 0 } ^ { s } \left( \Sigma ^ { ( s ) } \right) \subsetneq E ,\tag{76}
$$

in which case the on-shell samples probe only a proper subset of the ambient space on which the realized vector fields are defined. Second,

$$
\pi _ { 0 } ^ { s } \left( \Sigma ^ { ( s ) } \right) = E ,\tag{77}
$$

in which case the zeroth-order coordinates are not geometrically restricted to a proper subset of E. Even in the latter case, however, projection onto $\Sigma ^ { ( s ) }$ may induce correlations or nonuniform sampling among the zeroth-order coordinates in a finite dataset. For example, whenever the DE system locally determines a dependent coordinate in the form $u ^ { \alpha } = F ^ { \alpha } \left( \mathbf { x } , \mathbf { u } _ { I } \right)$ with $0 \leq | I | \leq s$ , the corresponding on-shell values of $u ^ { \alpha }$ are no longer sampled independently of the remaining jet coordinates after projection. Consequently, restricting the independence test to the finite on-shell sample cloud may produce sample-induced rank loss or poor conditioning, whereby globally distinct vector fields exhibit dependent, or nearly dependent, responses over the sampled region.

To reduce this degeneracy, we construct an auxiliary of-shell dataset

$$
\mathbf { S } ^ { \mathrm { o f f } } : = \left\{ z _ { l } ^ { ( s ) } \notin \Sigma ^ { ( s ) } \mid \left\| \Delta ^ { ( s ) } \left( z _ { l } ^ { ( s ) } \right) \right\| _ { 2 } > \epsilon _ { \mathrm { o f f } } \right\} ,\tag{78}
$$

where $\epsilon _ { \mathrm { o f f } } \ > \ 0$ and $l = 1 , \ldots , N _ { \mathrm { o f f } } = N _ { \mathrm { o n } }$ . Each ofshell sample is generated randomly within a prescribed neighborhood of its corresponding originator sample from $\mathbf { S } ^ { \mathrm { o n } }$ defined in (41), subject to the condition in (78). In particular, the fiber coordinates are perturbed away from their on-shell values, thereby relaxing correlations that may have been induced by projection onto the equation manifold. The auxiliary samples therefore remain computationally tied to the data-supported region while probing a less constrained neighborhood in the ambient jet space.

From $\mathbf { S } ^ { \mathrm { { o f f } } }$ , we then select a suficiently large fixed subset

$$
{ \bf S } _ { \mathrm { a n c } } ^ { \mathrm { o f f } } \subset { \bf S } ^ { \mathrm { o f f } } , \quad A _ { \mathrm { o f f } } : = \left| { \bf S } _ { \mathrm { a n c } } ^ { \mathrm { o f f } } \right| ,\tag{79}
$$

which we call the $o f { \bar { f } } .$ shell anchor dataset. Although the anchor samples themselves belong to $J ^ { ( s ) } ( \pi )$ , the independence objective evaluates only the corresponding zerothorder generator responses in $T E$ . The same anchor set is reused when comparing a candidate with the generators already discovered, so that the resulting independence test probes one common constant-coeficient relation over the entire sampled anchor domain rather than allowing unrelated pointwise dependencies. Of-shell anchoring therefore provides a more robust finite-sample surrogate for ambient linear independence while remaining localized to the neighborhood of the observed equation manifold.

Starting with the second discovered generator $v _ { 2 } ,$ global linear independence is enforced by separating a candidate $v _ { i } ,$ for $i > 1$ , into a component represented by the previously discovered generators and a residual component outside that represented span. Accordingly, we write

$$
v _ { i } = v _ { \perp , i } + \underbrace { \sum _ { j = 1 } ^ { i - 1 } c _ { i } ^ { j } v _ { j } } _ { v _ { \parallel , i } } , \quad c _ { i } ^ { j } \in \mathbb { R } ,\tag{80}
$$

where $v _ { \parallel , i } \in \operatorname { s p a n } _ { \mathbb { R } } \left\{ v _ { 1 } , \dots , v _ { i - 1 } \right\}$ denotes the component already represented by the previously discovered generators, while

$$
v _ { \bot , i } : = v _ { i } - v _ { \parallel , i }\tag{81}
$$

denotes the residual component associated with the candidate. The constant coeficient vector determining $v _ { \parallel , i }$ is obtained globally over the of-shell anchor dataset and is specified below.

Since the independence criterion is evaluated through the zeroth-order generator responses, we introduce the anchorinduced norm

$$
\| \boldsymbol { v } \| _ { \mathrm { a n c } } ^ { 2 } : = \frac { 1 } { A _ { \mathrm { o f f } } } \sum _ { z _ { a } ^ { ( s ) } \in \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o f f } } } \| \boldsymbol { v } \| _ { z _ { a } } \| _ { 2 } ^ { 2 } ,\tag{82}
$$

where $z _ { a } : = \pi _ { 0 } ^ { s } \left( z _ { a } ^ { ( s ) } \right) \in E$ denotes the zeroth-order projection of an anchor jet. Thus, a single norm aggregates the response of a generator over the entire fixed anchor domain rather than assessing independence pointwise.

Using (82), we define the relative novelty response of $v _ { i }$ as

$$
\mathcal { R } _ { \mathrm { n o v } } \left( v _ { i } \right) : = \frac { \| v _ { \perp , i } \| _ { \mathrm { a n c } } } { \| v _ { i } \| _ { \mathrm { a n c } } + \epsilon _ { \mathrm { n o v } } } , \quad i > 1 ,\tag{83}
$$

where $\epsilon _ { \mathrm { n o v } } > 0$ is a numerical regularizer. A small value of $\mathcal { R } _ { \mathrm { n o v } } \left( v _ { i } \right)$ indicates that the candidate is largely represented by the real span of the previously discovered generators, whereas a larger value indicates the presence of a substantial globally novel component over the anchor domain.

The corresponding linear-independence constraint is

$$
c _ { \mathrm { i n d } } : = \tau _ { \mathrm { n o v } } - \mathcal { R } _ { \mathrm { n o v } } \left( v _ { i } \right) \leq 0 ,\tag{84}
$$

where $\tau _ { \mathrm { n o v } } > 0$ specifies the minimum required relative novelty. Applying the quadratic exterior penalty in (65) yields

$$
\mathcal { L } _ { \mathrm { i n d } } : = \mathcal { P } \left( c _ { \mathrm { i n d } } \right) .\tag{85}
$$

Hence, ${ \mathcal { L } } _ { \mathrm { i n d } }$ penalizes candidates whose globally evaluated response is insuficiently distinct from the span already represented by the previously discovered generators.

To determine the constant coeficient vector $\begin{array} { r l } { \mathbf { C } _ { i } } & { { } : = } \end{array}$ $\left[ c _ { i } ^ { 1 } , \ldots , c _ { i } ^ { i - 1 } \right] \ \in \ \mathbb { R } ^ { i - 1 }$ in (80) in a “global” manner, we employ a single least-squares projection over the complete of-shell anchor dataset $\bar { \mathbf { S } } _ { \mathrm { a n c } } ^ { \mathrm { o f f } }$ . For each previously discovered generator $v _ { j }$ , define its stacked anchor response vector as

$$
Y _ { j } : = \mathrm { v e c } \left( \left( \left. v _ { j } \right| _ { z _ { a } } \right) _ { z _ { a } ^ { \left( s \right) } \in \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o f f } } } \right) \in \mathbb { R } ^ { A _ { \mathrm { o f f } } D } ,\tag{86}
$$

where $z _ { a } : = \pi _ { 0 } ^ { s } \left( z _ { a } ^ { ( s ) } \right) \in E$ and D = dim E. The old anchor response matrix is then

$$
R _ { i } : = \left[ \begin{array} { l l l } { | } & { } & { \quad | } \\ { Y _ { 1 } } & { \cdots } & { Y _ { i - 1 } } \\ { | } & { } & { \quad | } \end{array} \right] \in \mathbb { R } ^ { A _ { \mathrm { o f f } } D \times ( i - 1 ) } .\tag{87}
$$

Each column of $R _ { i }$ therefore represents one previously discovered generator over the entire of-shell anchor domain. Similarly, let

$$
Y _ { i } : = \mathrm { v e c } \left( \left( \left. v _ { i } \right| _ { z _ { a } } \right) _ { z _ { a } ^ { ( s ) } \in { \bf S } _ { \mathrm { a n c } } ^ { \mathrm { o f f } } } \right) \in \mathbb { R } ^ { A _ { \mathrm { o f f } } D }\tag{88}
$$

denote the stacked anchor response of the candidate generator under training.

The global coeficient vector is obtained from

$$
\mathbf { C } _ { i } : = \underset { \mathbf { C } \in \mathbb { R } ^ { i - 1 } } { \operatorname { a r g m i n } } \left\| Y _ { i } - R _ { i } \mathbf { C } \right\| _ { 2 } ^ { 2 } = R _ { i } ^ { \dagger } Y _ { i } ,\tag{89}
$$

where $R _ { i } ^ { \dagger }$ denotes the Moore–Penrose pseudoinverse. If $R _ { i }$ has full column rank, (89) reduces to

$$
\mathbf { C } _ { i } = \left( \boldsymbol { R } _ { i } ^ { \top } \boldsymbol { R } _ { i } \right) ^ { - 1 } \boldsymbol { R } _ { i } ^ { \top } \boldsymbol { Y } _ { i } .\tag{90}
$$

Thus, $R _ { i } { \bf C } _ { i }$ is the least-squares projection of $Y _ { i }$ onto the column space of $R _ { i } ,$ which represents the sampled real span of the previously discovered generators. The corresponding residual

$$
Y _ { \perp , i } : = Y _ { i } - R _ { i } \mathbf { C } _ { i } = \left( I - R _ { i } R _ { i } ^ { \dagger } \right) Y _ { i }\tag{91}
$$

is the anchor-space representation of the complementary component $v _ { \perp , i }$ introduced in (80). In particular, the anchorinduced norms in (82) satisfy

$$
\begin{array} { r } { \| v _ { i } \| _ { \mathrm { a n c } } = \displaystyle \frac { 1 } { \sqrt { A _ { \mathrm { o f f } } } } \left\| Y _ { i } \right\| _ { 2 } , } \\ { \| v _ { \perp , i } \| _ { \mathrm { a n c } } = \displaystyle \frac { 1 } { \sqrt { A _ { \mathrm { o f f } } } } \left\| Y _ { \perp , i } \right\| _ { 2 } . } \end{array}\tag{92}
$$

For every $i > 1$ , the remaining optimization requirements are imposed on the complementary component $v _ { \perp , i }$ rather than directly on the raw candidate $v _ { i }$ . Indeed, $v _ { \parallel , i }$ belongs to the real span of the previously accepted generators and therefore represents information already recovered by the framework. Allowing this component to contribute to the remaining objectives could mask a deficiency of the genuinely novel direction. Accordingly, the normalized determining residual $\rho _ { l } ( v )$ in (59), the minimum-magnitude and relative nontriviality constraints defined through (62) and (63), the relative solution-dependence response in (70), and the gates in (68) and (69) are evaluated using $v _ { \perp , i }$ for $i > 1$ . For the first discovered generator $v _ { 1 }$ , no previously recovered subspace exists and hence the raw realization is used directly.

## 3.2 LIMITATIONS OF THE OPTIMIZATION FOR-MULATION

Over several years, we attempted to impose all requirements introduced in Initial Multi-Objective Optimization Formulation simultaneously through diferent multiobjective optimization strategies. In practice, the resulting optimization became increasingly dificult to control. We summarize below the principal dificulties encountered, both to motivate the transition to the operator-theoretic formulation and to explain the need for the admissible restrictions introduced by the new methodology.

The most fundamental dificulty originates from the symmetry objective itself. Let a candidate generator be represented by a finite-dimensional neural-network parameter vector $\mathbf { \Theta } \Theta \in \mathcal { P }$ through the realization map

$$
\Psi : { \mathcal { P } } \to { \mathfrak { X } } ( E ) , \quad \Theta \longmapsto v _ { \Theta } .\tag{93}
$$

For a standard neural-network parameterization, Ψ is nonlinear in Θ. Let $\mathcal { D } _ { \Delta }$ denote the determining operator associated with the infinitesimal symmetry condition in (38). Although $\mathcal { D } _ { \Delta }$ is linear in the generator $v _ { \Theta }$ (see Operator-Theoretic Symmetry Representation), the composition $\mathcal { D } _ { \Delta } \circ \Psi$ is generally nonlinear in the trainable parameters. Consequently, the set

$$
\{ \Theta \in \mathcal { P } \mid \mathcal { D } _ { \Delta } \left[ v _ { \Theta } \right] = 0 \}\tag{94}
$$

does not, in general, constitute a fixed linear subspace of the parameter space.

The same issue appears after sampling. Let $\widehat { \mathcal { D } } _ { \Delta }$ denote the determining operator evaluated over the finite on-shell

dataset $\mathbf { S } ^ { \mathrm { o n } }$ . Its composition with the neural realization has the parameter-dependent Jacobian

$$
\mathrm { J } _ { \Theta } \left[ \widehat { \cal D } _ { \Delta } \circ \Psi \right] \left( \Theta \right) ,\tag{95}
$$

whose row and null spaces vary as Θ changes during training. Hence, an singular value decomposition (SVD) of this Jacobian characterizes only the local linearization of the sampled determining residual around the current parameter iterate; it does not expose a fixed symmetry nullspace that can be isolated once and preserved throughout optimization.

This behavior was also observed empirically in our preliminary experiments with $\mathcal { L } _ { \mathrm { s y m } }$ in (60). SVD analysis of the sampled residual Jacobian revealed weakly constrained right-singular directions associated with small singular values, typically with $\sigma _ { i } ^ { 2 }$ approaching $1 0 ^ { - 6 }$ and below. These directions were dificult to suppress consistently during gradient-based training and produced persistent row-space contamination, whereby components outside the desired local nullspace remained mixed with the learned representation. Thus, even before the remaining objectives were imposed, the nonlinear parameterization prevented the determining condition from being treated as a fixed linear-algebraic admissibility constraint.

A second dificulty arises from the direct competition between the infinitesimal-symmetry objective $\mathcal { L } _ { \mathrm { s y m } }$ and the global-independence objective $\mathcal { L } _ { \mathrm { i n d } }$ . Once several generators have been discovered, their span already constitutes a region of the represented function space in which the determining residual is small. Consequently, gradient descent on $\mathcal { L } _ { \mathrm { s y m } }$ tends to favor directions close to the previously recovered subspace. In contrast, ${ \mathcal { L } } _ { \mathrm { i n d } }$ explicitly penalizes such directions and pushes the new candidate toward a complementary component. The two objectives therefore exert competing pressures on the same trainable realization.

Projecting the remaining objectives onto $v _ { \perp , i } ,$ , as described in the final paragraph of Initial Multi-Objective Optimization Formulation, alleviates but does not eliminate this conflict. Since the determining operator is linear in the generator,

$$
\mathcal { D } _ { \Delta } \left[ v _ { \perp , i } \right] = \mathcal { D } _ { \Delta } \left[ v _ { i } \right] - \sum _ { j = 1 } ^ { i - 1 } c _ { i } ^ { j } \mathcal { D } _ { \Delta } \left[ v _ { j } \right] .\tag{96}
$$

Hence, if the previously discovered generators satisfy the infinitesimal symmetry condition only approximately, their residual errors are inherited by the determining residual of $v _ { \perp , i }$ . The same observation applies after finite sampling to $\widehat { \mathcal { D } } _ { \Delta }$ . Therefore, inaccuracies accumulated during earlier discovery stages propagate into the optimization of subsequent generators rather than remaining confined to the generators in which they originated.

Moreover, the coeficients $\mathbf { C } _ { i }$ defining $v _ { \parallel , i }$ are themselves recomputed from the current realization of $v _ { i }$ through the global least-squares projection over $\mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o f f } }$ . The complementary component presented to the remaining objectives is therefore not fixed during training, but changes together with the candidate realization. This additional coupling further complicates the simultaneous balancing of symmetry and novelty.

As a result, sequential discovery under the multi-objective formulation becomes progressively more sensitive to residual errors and optimization history as the number of recovered generators increases. A candidate must simultaneously approach the symmetry sector while being repelled from the portion of that sector already represented by the previously discovered generators, with neither requirement imposed as an exact subspace restriction.

A third dificulty concerns the solution-dependence objective ${ \mathcal { L } } _ { \mathrm { d e p } }$ . As established formally in Intrinsic Admissibility Constraints, a linear DE system possesses an additive solution sector $\mathcal { A } _ { \mathrm { s o l } } \subseteq { \mathfrak { g } }$ , which may be infinite dimensional. This sector extends the solution-dependent behavior discussed in Intrinsic and Solution-Dependent Symmetry Sectors and provides a potentially large family of directions that satisfy the infinitesimal symmetry condition exactly. Consequently, $\mathcal { L } _ { \mathrm { s y m } }$ alone has no mechanism for distinguishing these directions from the finite-dimensional intrinsic sector sought by the framework, described in Intrinsic and Solution-Dependent Symmetry Sectors.

The gated penalty ${ \mathcal { L } } _ { \mathrm { d e p } }$ attempts to repel the optimizer from this prohibited sector only when both the verticality and u-independence detectors in (68) and (69) are suficiently active. Whenever either gate becomes small, the solution-dependence penalty is correspondingly suppressed, while $\mathcal { L } _ { \mathrm { s y m } }$ continues to favor any direction that reduces the determining residual. For a large, and particularly an infinite-dimensional, additive solution sector, this creates numerous admissible symmetry directions that cannot be reliably excluded through a finite collection of smooth gates and scalar penalties.

The resulting optimization may therefore alternate between approaching directions associated with $\boldsymbol { A } _ { \mathrm { s o l } }$ under the symmetry objective and being repelled from them once the solution-dependence gates become active. This attraction– repulsion behavior makes ${ \mathcal { L } } _ { \mathrm { d e p } }$ highly sensitive to gate thresholds, sharpness parameters, relative loss weights, and the current optimization trajectory. More fundamentally, it attempts to exclude an entire symmetry sector through local penalty shaping rather than removing that sector from the admissible search space itself.

The nontriviality objective introduces a further numerical complication. Apart from the regularization term $\epsilon _ { \mathrm { n t r } }$ , the relative response ${ \mathcal { R } } _ { \mathrm { n t r } }$ in (63) is scale invariant because both $\mathbf { E } _ { v } ^ { ( s - 1 ) }$ and $\mathrm { p r } ^ { ( s ) } ( v )$ depend linearly on v. Consequently, reducing the magnitude of a candidate does not necessarily reduce its relative nontriviality response as long as the candidate remains above the numerical regularization floor. Thus, ${ \mathcal { R } } _ { \mathrm { n t r } }$ alone cannot ensure that the complementary component $v _ { \perp , i }$ remains numerically visible, which necessitates the independent minimum-magnitude constraint $c _ { \mathrm { m i n } }$

Near the zero-response regime, however, the normalized ratio becomes increasingly sensitive to the numerical regularizer and small fluctuations in the numerator and denominator. In practice, this required a smooth activation of the nontriviality constraint so that $c _ { \mathrm { n t r } }$ was not enforced aggressively before the candidate acquired suficient magnitude. Hence, even the sampled nontriviality requirement could not be represented by a single stable objective, but instead required coupled magnitude and activation mechanisms whose behavior depended on additional numerical thresholds.

Finally, the determining equations themselves contain structural information that is largely discarded when they are represented only through the scalar symmetry loss $\mathcal { L } _ { \mathrm { s y m } }$ . A generic nonlinear optimizer is not informed of restrictions on the admissible functional form of the infinitesimal coefficients, nor of possible decompositions of the represented symmetry space. It therefore tends to converge first toward symmetry directions that are comparatively easy to represent and optimize. Once such directions have been recovered and excluded by the novelty objective, optimization of the remaining directions may become progressively more dificult or stall altogether. For example, in the benchmark problem of Section 5, our preliminary optimization-based experiments repeatedly recovered four generators whose real span coincided with the empirically easy symmetry subspace

$$
\mathfrak { g } _ { \mathrm { e a s y } } : = \operatorname { s p a n } _ { \mathbb { R } } \left\{ \partial _ { x } , \partial _ { y } , \partial _ { t } , u \partial _ { u } \right\} .\tag{97}
$$

Subsequent optimization attempts failed to produce a fifth globally independent direction: the learned candidates remained within, or converged back toward, ${ \mathfrak { g } } _ { \mathrm { e a s y } }$ despite the action of ${ \mathcal { L } } _ { \mathrm { i n d } }$ . Here, the qualifier easy refers only to the empirical accessibility of this subspace under the original optimization dynamics and does not indicate an intrinsic algebraic classification.

This behavior further motivates treating the determining system as an operator carrying exploitable algebraic structure, rather than merely as a residual to be minimized. Structural restrictions implied by particular classes of DEs are discussed later in the context of applicability and scope in Section 6.

## 3.3 OPERATOR-THEORETIC SYMMETRY REPRE-SENTATION

Starting with the present subsection, we develop the new operator-theoretic approach used throughout the remainder of this section. The first objective is to replace the moving local nullspace encountered in Limitations ofthe Optimization Formulation by a fixed represented nullspace of the determining operator. This nullspace constitutes the first admissible restriction on the search space from which ${ \mathfrak { g } } _ { \mathrm { i n t } }$ will subsequently be recovered. Its dimension, however, remains dependent on the expressive space supplied by the chosen finite representation and therefore should not be identified a priori with the dimension of the full Lie symmetry algebra.

Let $\Delta ^ { ( s ) } = \mathbf { 0 }$ be an order-s DE system and let $v \in { \mathfrak { X } } ( E )$ be a candidate Lie point generator. At a given jet $z ^ { ( s ) } \in J ^ { ( s ) } ( \pi )$ the prolongation operator $\mathrm { p r } ^ { ( s ) } : \bar { \mathfrak { X } } ( E ) \bar {  } \mathfrak { X } ( J ^ { ( s ) } ( \pi ) )$ in (34) maps v to its order-s prolongation. Meanwhile,

$$
\operatorname J _ { \Delta ^ { ( s ) } } \left( z ^ { ( s ) } \right) : T _ { z ^ { ( s ) } } J ^ { ( s ) } \left( \pi \right) \to T _ { \mathbf { y } } \mathbb { R } ^ { m }\tag{98}
$$

denotes the Jacobian matrix of the DE system at that jet with $\mathbf { y } = \Delta ^ { ( s ) } \left( z ^ { ( s ) } \right)$ . We define the corresponding pointwise determining operator by

$$
\mathcal { D } _ { \Delta } \big \vert _ { z ^ { ( s ) } } \left[ v \right] : = \operatorname { J } _ { \Delta ^ { ( s ) } } \left( z ^ { ( s ) } \right) \operatorname { p r } ^ { ( s ) } \left( v \right) \bigg \vert _ { z ^ { ( s ) } } .\tag{99}
$$

Accordingly, the infinitesimal symmetry condition in (37) requires $\mathcal { D } _ { \Delta } | _ { z ^ { ( s ) } } \left[ v \right] = \mathbf { 0 }$ for every $\boldsymbol { z } ^ { ( s ) } \in \Sigma ^ { ( s ) }$ . Since prolongation is linear in v and $\mathrm { J } _ { \Delta ^ { ( s ) } } \left( z ^ { ( s ) } \right)$ acts linearly on the resulting tangent vector, $\mathcal { D } _ { \Delta } \big | _ { z ^ { ( s ) } }$ is itself a linear operator in the generator argument.

The dificulty identified in Limitations of the Optimization Formulation therefore originates not from the determining operator itself, but from the nonlinear parameterization of its argument. Let $\Psi$ in (93) denote the realization map associated with a standard neural-network parameterization. Although $\mathcal { D } _ { \Delta } \big | _ { z ^ { ( s ) } }$ is linear in $v ,$ the composition $\mathcal { D } _ { \Delta } | _ { z ^ { ( s ) } } \circ \Psi$ is generally nonlinear in $\Theta .$ . For convenience, define the corresponding parameterized residual map

$$
{ \bf r } _ { z ^ { ( s ) } } \left( \Theta \right) : = \left. { \mathcal { D } } _ { \Delta } \right| _ { z ^ { ( s ) } } \left[ v _ { \Theta } \right] .\tag{100}
$$

Its zero set in $\mathcal { P }$ is therefore not, in general, a fixed linear subspace. Instead, recovering feasibility requires successive local corrections in parameter space. For example, a damped Gauss–Newton step may be written as

$$
\begin{array} { l } { \displaystyle \delta \Theta _ { \mathrm { D G N } } : = \frac { \mathrm { a r g m i n } } { \delta \Theta } } \\ { \displaystyle \qquad \frac { 1 } { 2 } \Big \| { \bf r } _ { z ^ { ( s ) } } \left( \Theta \right) + \mathrm { J } _ { { \bf r } _ { z ^ { ( s ) } } } \left( \Theta \right) \delta \Theta \Big \| _ { 2 } ^ { 2 } + } \\ { \displaystyle \qquad \frac { \lambda } { 2 } \left\| \delta \Theta \right\| _ { 2 } ^ { 2 } . } \end{array}\tag{101}
$$

where $\lambda > 0$ is a damping parameter. Such a correction is determined by the Jacobian of the residual map at the current parameter iterate and therefore restores only local feasibility. As Θ changes, the corresponding local row and null spaces change as well.

The operator-theoretic formulation instead adopts a fixed functional representation that is exactly linear in its trainable coeficients. For this purpose, we are inspired by the splinebased functional representation employed by Kolmogorov– Arnold Networks (KANs) [39]. A standard multilayer KAN, however, remains nonlinear in its trainable parameters and therefore does not provide the fixed linear structure required here. We consequently retain only its spline-based representational principle and construct a fixed spline dictionary: the basis functions and their underlying grid are frozen, while only the expansion coeficients remain trainable. Hence, the represented infinitesimal coeficients may remain highly nonlinear functions of their inputs while depending linearly on the finite trainable parameter vector.

Let

$$
\Psi ^ { ( k ) } : \mathbb { R } ^ { P } \to \mathfrak { X } ( E ) , \quad \Theta \longmapsto v _ { \Theta }\tag{102}
$$

denote the resulting realization map associated with the fixed dictionary $D _ { ( k ) }$ of maximum interaction order $k ,$ , where $P$ is the total number of trainable parameters. By construction, $\Psi ^ { ( k ) }$ is linear in Θ. Since the determining operator is also linear in its generator argument, the composition

$$
\mathcal { D } _ { \Delta } \big | _ { z ^ { ( s ) } } \circ \Psi ^ { ( k ) } : \mathbb { R } ^ { P }  \mathbb { R } ^ { m }\tag{103}
$$

is a fixed linear operator for every fixed $z ^ { ( s ) }$ . Its kernel is therefore a fixed linear subspace of the parameter space $\mathbb { R } ^ { P }$ , rather than a parameter-dependent local tangent approximation. The precise construction of $D _ { ( k ) }$ , including its spline basis, interaction structure, grid specification, and the representational consequences of these choices, is deferred to Section 4.

To convert the pointwise operator in (99) into a finite computational object, we evaluate it over a fixed on-shell anchor dataset ${ \bf S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } \subset { \bf S } ^ { \mathrm { o n } }$ , with $A _ { \mathrm { o n } } : = | \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } | .$ . We define the sampled determining operator by stacking the pointwise determining responses over all anchor jets,

$$
\begin{array} { r l r } { \left. \widehat { \mathcal { D } } _ { \Delta } \right. _ { \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } } \left[ v \right] : = } & { } & \\ { \mathrm { v e c } \left( \left( \left. \mathcal { D } _ { \Delta } \right. _ { z _ { a } ^ { ( s ) } } \left[ v \right] \right) _ { z _ { a } ^ { ( s ) } \in \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } } \right) \in \mathbb { R } ^ { A _ { \mathrm { o n } } m } , } \end{array}\tag{104}
$$

where m is the number of independent equations constituting $\Delta ^ { ( s ) } = \mathbf { 0 }$ . Equivalently, each block of (104) is obtained from $\mathrm { J } _ { \Delta ^ { ( s ) } } \left( z _ { a } ^ { ( s ) } \right)$ acting on $\mathrm { p r } ^ { ( s ) } \left( v \right) \big \vert _ { z _ { n } ^ { ( s ) } }$ . Since every pointwise determining operator is linear in v, their stacked realization $\left. \widehat { \mathcal { D } } _ { \Delta } \right. _ { \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } }$ is also linear.

Composing the sampled determining operator with the fixed linear realization map $\Psi ^ { ( k ) }$ in (102) yields the represented sampled determining operator

$$
\widehat { \mathcal { D } } _ { \Delta } ^ { ( k ) } | _ { \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } } : =  \widehat { \mathcal { D } } _ { \Delta } | _ { \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } } \circ \Psi ^ { ( k ) } : \mathbb { R } ^ { P }  \mathbb { R } ^ { A _ { \mathrm { o n } } m } .\tag{105}
$$

The corresponding sampled determining residual response is therefore

$$
\widehat { \mathbf { r } } _ { \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } } \left( \Theta \right) : = \left. \widehat { D } _ { \Delta } ^ { ( k ) } \right| _ { \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } } \left[ \Theta \right] .\tag{106}
$$

Because both constituent maps in (105) are linear, the residual response is linear in Θ. Hence, its Jacobian

$$
\widehat { M } _ { \Delta } ^ { ( k ) } : = \frac { \partial \widehat { \mathbf { r } } _ { \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } } } { \partial \mathbf { \Theta } } \in \mathbb { R } ^ { { A } _ { \mathrm { o n } } m \times { P } }\tag{107}
$$

is independent of Θ. In particular,

$$
\widehat { \mathbf { r } } _ { \mathbf { S } _ { \mathrm { a n c } } ^ { 0 } } \left( \Theta \right) = \widehat { M } _ { \Delta } ^ { ( k ) } \Theta ,\tag{108}
$$

so the sampled determining equations reduce to the fixed homogeneous linear system

$$
\widehat { M } _ { \Delta } ^ { ( k ) } \Theta = \mathbf { 0 } .\tag{109}
$$

Thus, unlike the parameter-dependent local nullspace encountered under the nonlinear realization map, the present formulation exposes a fixed nullspace directly in the parameter space.

The first admissible restriction is consequently defined as

$$
\begin{array} { r } { \Theta \in \widehat { N } _ { \Delta } ^ { ( k ) } : = \ker \left( \widehat { M } _ { \Delta } ^ { ( k ) } \right) \subseteq \mathbb { R } ^ { P } . } \end{array}\tag{110}
$$

Accordingly, the infinitesimal symmetry condition is no longer imposed through the loss $\mathcal { L } _ { \mathrm { s y m } }$ in (60). Instead, all subsequent discovery operations are restricted directly to the represented sampled nullspace $\widehat { N } _ { \Delta } ^ { ( k ) }$

The relationship between this parameter-space nullspace and the corresponding represented vector fields should be stated carefully. In general,

$$
\ker \left( \left. \widehat { \mathcal { D } } _ { \Delta } \right| _ { \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } } \circ \boldsymbol { \Psi } ^ { ( k ) } \right) = \left( \boldsymbol { \Psi } ^ { ( k ) } \right) ^ { - 1 } \left( \ker \left. \widehat { \mathcal { D } } _ { \Delta } \right| _ { \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } } \right)\tag{111}
$$

If $\Psi ^ { ( k ) }$ is injective, its image satisfies

$$
\Psi ^ { ( k ) } \left( \widehat { N } _ { \Delta } ^ { ( k ) } \right) = \operatorname { i m } \left( \Psi ^ { ( k ) } \right) \cap \ker \left. \widehat { \mathcal { D } } _ { \Delta } \right| _ { \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } } .\tag{112}
$$

Hence, the recovered symmetry sector is necessarily restricted by the expressive richness of the chosen dictionary $D _ { ( k ) }$

Moreover, because (112) is constructed from finitely many on-shell anchors, $\widehat { N } _ { \Delta } ^ { ( k ) }$ should be interpreted as a represented sampled symmetry nullspace, rather than being identified automatically with the exact determining kernel over the entire equation manifold. Insuficient sampling or degeneracy of the representation may introduce additional null directions that do not correspond to genuine symmetries on $\Sigma ^ { ( s ) }$

Remark 5. Since $\Psi ^ { ( k ) }$ is linear in Θ, its derivative with respect to Θ is independent of any evaluation point $\mathbf { \Theta } _ { \Theta } .$ It is completely determined by the fixed dictionary $D _ { ( k ) }$ Consequently, rank deficiency or redundancy in the dictionary may introduce non-identifiable parameter directions and artificial contributions to $\widehat { N } _ { \Delta } ^ { ( k ) }$ . The construction of $D _ { ( k ) } ,$ , the distribution ofits spline basisfunctions over the prescribed grid, and the numerical rank safeguards used to control these efects are discussed in Section 4.

## 3.4 INTRINSIC ADMISSIBILITY CONSTRAINTS

Definition 9 requires every nonzero $v \in { \mathfrak { g } } _ { \mathrm { i n t } }$ to be both nontrivial and solution-independent. Consequently, the represented sampled symmetry nullspace $\widehat { N } _ { \Delta } ^ { \overline { { ( k ) } } }$ in (110) is generally broader than the target intrinsic sector, as it may also contain directions that are trivial or carry prohibited solution-dependent information. For linear DE systems, an important part of the latter sector can be identified and removed through an additional operator-theoretic restriction. For nonlinear DE systems, however, solution dependence does not generally admit the same linear decomposition. Moreover, as shown later in this subsection, nontriviality itself cannot be imposed simply by restricting to another vector subspace and therefore remains enforced through ${ \mathcal { L } } _ { \mathrm { n t r } }$ in (66).

For both homogeneous and inhomogeneous linear DE systems, the associated homogeneous operator generates an additive symmetry sector. Its algebraic structure is summarized below.

Proposition 2 (Additive solution subalgebra and ideal). Let

$$
\Delta ^ { ( s ) } : = \mathrm { L } ^ { ( s ) } \left[ \mathbf { u } \right] - \mathbf { g } = \mathbf { 0 }\tag{113}
$$

be a linear DE system, and define

$$
\begin{array} { r l } & { \mathcal { A } _ { \mathrm { s o l } } : = \Big \{ v _ { \varphi } = \displaystyle \sum _ { \alpha = 1 } ^ { q } \varphi ^ { \alpha } \left( { \bf x } \right) \partial _ { u ^ { \alpha } } \ \Big \vert } \\ & { \qquad \quad \mathrm { L } ^ { ( s ) } \left[ \left[ j ^ { s } i _ { \varphi } \right] \left( { \bf x } \right) \right] = { \bf 0 } , \ \forall { \bf x } \in { \cal M } \subset { \bf X } \Big \} . } \end{array}\tag{114}
$$

Then $\mathcal { A } _ { \mathrm { s o l } } \leq \mathfrak { g }$ and

$$
[ v _ { \varphi _ { 1 } } , v _ { \varphi _ { 2 } } ] = 0\tag{115}
$$

for all $v _ { \varphi _ { 1 } } , v _ { \varphi _ { 2 } } \in \mathcal { A } _ { \mathrm { s o l } }$ . Hence, $\boldsymbol { A } _ { \mathrm { s o l } }$ is an Abelian Lie subalgebra of g.

If, in addition, the Lie point symmetries under consideration admit the projectible fiber-afine form $v = \xi \left( \mathbf { x } \right) \cdot \partial _ { \mathbf { x } }$ + $\left( A \left( \mathbf { x } \right) \mathbf { u } + \mathbf { b } \left( \mathbf { x } \right) \right) \cdot \partial _ { \mathbf { u } }$ , then

$$
[ \mathfrak { g } , \mathcal { A } _ { \mathrm { s o l } } ] \subseteq \mathcal { A } _ { \mathrm { s o l } } ,\tag{116}
$$

and therefore $\mathcal { A } _ { \mathrm { s o l } } \triangleleft { \mathfrak { g } }$ is an Abelian ideal.

A proof of Proposition 2 is provided in D.2. The first statement follows directly from linearity: if u satisfies the original system and $\varphi$ belongs to the kernel of its associated homogeneous operator, then

$$
{ \mathrm { L } } ^ { \left( s \right) } \left[ \mathbf { u } + \varepsilon \varphi \right] - \mathbf { g } = \mathbf { L } ^ { \left( s \right) } \left[ \mathbf { u } \right] - \mathbf { g }\tag{117}
$$

for every suficiently small ε. Thus, $v _ { \varphi }$ generates additive motion within the solution family. For a homogeneous linear system, φ is itself a solution of the original DE system and the construction coincides directly with the solution-dependent vertical class of Definition 8. For an inhomogeneous linear system, $\varphi$ instead solves the associated homogeneous system; equivalently, it represents the diference between two solutions of the afine inhomogeneous problem.

Proposition 2 further shows that $\boldsymbol { A } _ { \mathrm { s o l } }$ inherits the functional freedom of ker $\mathrm { L } ^ { ( s ) }$ and may therefore be infinite dimensional. This is one of the reasons, in addition to those discussed in Intrinsic and Solution-Dependent Symmetry Sectors, for excluding the additive solution sector from the target intrinsic algebra. Nevertheless, removal of $\boldsymbol { A } _ { \mathrm { s o l } }$ does not in general imply that the remaining symmetry algebra is finite dimensional, since other forms of arbitrary functional freedom may persist, as discussed in A.

For nonlinear DE systems, there is generally no analogous additive solution subspace generated by superposition. In particular, the purely vertical solution-dependent generators of Definition 8 need not be closed under addition, since two admissible local solutions $\varphi _ { i }$ and $\varphi _ { j }$ do not generally imply that $\varphi _ { i } + \varphi _ { j }$ is also a solution. Hence, the corresponding solution-dependent sector does not generally admit the same linear operator-theoretic exclusion employed for $\boldsymbol { A } _ { \mathrm { s o l } }$ . The applicability of the proposed framework to nonlinear DE systems, together with the structural conditions associated with projectable fiber-afine generators, is discussed in Section 6.

To identify the representable additive solution sector of a linear DE system, we introduce another operator that is linear in the candidate generator. Let $v _ { \Theta } \in { \mathfrak { X } } ( E )$ be specified by the coeficient pair $( \xi _ { \Theta } , \varphi _ { \Theta } )$ . Define

$$
\mathcal { T } _ { \mathrm { n a d d } } \left[ v _ { \Theta } \right] : = \left[ \underset { \mathrm { V e c } \left( \partial _ { \mathbf { u } } \varphi _ { \Theta } \right) } { \xi _ { \Theta } } \right] ,\tag{118}
$$

where $\partial _ { \mathbf { u } } \varphi _ { \Theta }$ denotes the Jacobian of the vertical coeficients with respect to the dependent coordinates u. Both constituent operations are linear in $v _ { \Theta }$ and therefore $\mathcal { T } _ { \mathrm { n a d d } }$ is a linear operator. Its kernel consists of represented generators satisfying $\pmb { \xi } _ { \Theta } = \mathbf { 0 }$ and $\partial _ { \mathbf { u } } \varphi _ { \Theta } = \mathbf { 0 }$ , i.e., purely vertical generators whose coeficients depend only on x.

We evaluate (118) over the fixed on-shell anchor dataset and compose it with the linear realization map $\Psi ^ { ( k ) }$ . This yields the sampled represented operator

$$
\widehat { T } _ { \mathrm { n a d d } } ^ { ( k ) }  _ { \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } } : \mathbb { R } ^ { P }  \mathbb { R } ^ { A _ { \mathrm { o n } } ( p + q ^ { 2 } ) } ,\tag{119}
$$

whose action is obtained by stacking $\xi _ { \Theta }$ and $\partial _ { \mathbf { u } } \varphi _ { \Theta }$ over all samples in $\mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } }$ . Since $\Psi ^ { ( \dot { k } ) }$ and $\mathcal { T } _ { \mathrm { n a d d } }$ are both linear in Θ, the operator in (119) is represented by a fixed matrix.

Within the represented sampled symmetry nullspace $\widehat { N } _ { \Delta } ^ { ( k ) }$ simultaneous vanishing of these three responses identifies the representable purely vertical, u-independent symmetry sector. For a linear DE system, Proposition 2 then identifies this sector with the additive solutions of the associated homogeneous operator. Accordingly, we define the sampled represented additive solution sector

$$
\begin{array} { r } { \widehat { \mathcal { A } } _ { \mathrm { s o l } } ^ { ( k ) } : = \widehat { N } _ { \Delta } ^ { ( k ) } \cap \ker \left( \left. \widehat { \mathcal { T } } _ { \mathrm { n a d d } } ^ { ( k ) } \right| _ { \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } } \right) . } \end{array}\tag{120}
$$

As with the sampled determining nullspace, the qualifier sampled represented is essential: (120) identifies the portion of the additive sector visible within the finite dictionary $D _ { ( k ) }$ and over the chosen anchor set.

Since $\widehat { A } _ { \mathrm { s o l } } ^ { ( k ) }$ is a vector subspace of $\widehat { N } _ { \Delta } ^ { ( k ) }$ , we remove it directly from the admissible parameter space rather than attempting to repel the optimizer through ${ \mathcal { L } } _ { \mathrm { d e p } } .$ . Using the Euclidean inner product of the parameter space $\mathbb { R } ^ { P }$ , define

$$
\widehat { N } _ { \mathrm { n a d d } } ^ { ( k ) } : = \widehat { N } _ { \Delta } ^ { ( k ) } \cap \left( \widehat { \mathcal { A } } _ { \mathrm { s o l } } ^ { ( k ) } \right) ^ { \perp } .\tag{121}
$$

Because $\widehat { A } _ { \mathrm { s o l } } ^ { ( k ) } \subseteq \widehat { N } _ { \Delta } ^ { ( k ) }$ , this construction yields the directsum decomposition

$$
\widehat { N } _ { \Delta } ^ { ( k ) } = \widehat { A } _ { \mathrm { s o l } } ^ { ( k ) } \oplus \widehat { N } _ { \mathrm { n a d d } } ^ { ( k ) } .\tag{122}
$$

Hence, $\widehat { N } _ { \mathrm { n a d d } } ^ { ( k ) }$ retains only representatives whose coeficient vectors contain no component along the sampled represented additive solution sector.

Importantly, (121) removes not only generators lying entirely in $\widehat { A } _ { \mathrm { s o l } } ^ { ( k ) }$ , but also additive components occurring inside mixed representatives. This is consistent with Remark 3, which reserves the term solution-dependent generator for the purely vertical class and does not classify a mixed generator with a nonvanishing horizontal component as solution-dependent merely because it contains a solution-dependent vertical contribution. The present linear-DE restriction is deliberately stronger: components lying in the sampled represented additive solution sector are removed from every admissible representative before sequential discovery proceeds. The orthogonal complement in (121) is a computational choice induced by the finite coeficient representation and should not be interpreted as a canonical orthogonal decomposition of the abstract Lie symmetry algebra.

To examine whether nontriviality can likewise be imposed through a linear admissible restriction, we first exploit the linearity of the prolonged characteristic information with respect to the generator. For the fixed linear realization $\bar { v } _ { \Theta } = \Psi ^ { ( k ) } ( \mathbf { \bar { \Theta } } )$ , define the sampled prolongedcharacteristic operator

$$
\begin{array} { r } { \widehat { \mathcal { T } } _ { \mathrm { t r i v } } ^ { ( k ) } \Big | _ { \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } } \left[ \Theta \right] : = \mathrm { v e c } \left( \left( \mathbf { E } _ { v _ { \Theta } } ^ { ( s - 1 ) } \Big | _ { z _ { a } ^ { ( s ) } } \right) _ { z _ { a } ^ { ( s ) } \in \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } } \right) } \end{array}\tag{123}
$$

Because $\Psi ^ { ( k ) }$ is linear in $\mathbf { \Theta } \Theta , \mathbf { Q } _ { v }$ is linear in $v ,$ and the total derivative operators entering $\mathbf { E } _ { v } ^ { ( s - 1 ) }$ are linear, the operator in (123) is linear in Θ.

Proposition 3 (Sampled triviality and geometric quotient). Let $\widehat { N } _ { \mathrm { n a d d } } ^ { ( k ) } \subseteq \mathbb { R } ^ { P }$ be the represented admissible coeficient

subspace defined in (121), and let $\widehat { \mathcal { T } } _ { \mathrm { t r i v } } ^ { ( k ) } \Big | _ { \mathbf { S } _ { \mathrm { \alpha , \mathtt { m } } } ^ { \mathrm { o n } } }$ be the sampled prolonged-characteristic operator in (123). Then the sampled-trivial sector

$$
\widehat K _ { \mathrm { t r i v } } ^ { ( k ) } : = \widehat N _ { \mathrm { n a d d } } ^ { ( k ) } \cap \ker \left( \left. \widehat T _ { \mathrm { t r i v } } ^ { ( k ) } \right| _ { { \bf S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } } \right)\tag{124}
$$

is a real vector subspace. Moreover, the quotient

$$
\widehat { N } _ { \mathrm { g e o m } } ^ { ( k ) } : = \frac { \widehat { N } _ { \mathrm { n a d d } } ^ { ( k ) } } { \widehat { K } _ { \mathrm { t r i v } } ^ { ( k ) } }\tag{125}
$$

is naturally isomorphic to the image of the sampled prolonged-characteristic operator restricted to $\widehat { N } _ { \mathrm { n a d d } } ^ { ( k ) } ,$

$$
\widehat { N } _ { \mathrm { g e o m } } ^ { ( k ) } \cong \mathrm { i m } \left( \widehat { \mathcal { T } } _ { \mathrm { t r i v } } ^ { ( k ) } \Big | _ { { \bf S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } , \widehat { N } _ { \mathrm { n a d d } } ^ { ( k ) } } \right) .\tag{126}
$$

In contrast, the set of coeficient vectors having nonzero sampled prolonged-characteristic response,

$$
\widehat { \mathcal { V } } _ { \mathrm { n t r } } ^ { ( k ) } : = \widehat { N } _ { \mathrm { n a d d } } ^ { ( k ) } \setminus \widehat { K } _ { \mathrm { t r i v } } ^ { ( k ) } ,\tag{127}
$$

is not a vector subspace.

A proof for Proposition 3 is also included in D.3. Proposition 3 follows directly from linearity and the first isomorphism theorem. In particular, the directions whose prolonged characteristic information vanishes over the entire anchor dataset form a kernel and can therefore be identified as a linear subspace $\widehat { K } _ { \mathrm { t r i v } } ^ { ( k ) }$ . The set obtained by removing this kernel, however, is not closed under either scalar multiplication by zero or vector addition and hence cannot serve as another admissible coeficient subspace.

One could instead operate on the quotient $\widehat { N } _ { \mathrm { g e o m } } ^ { ( k ) } ,$ but its elements are equivalence classes modulo sampled-trivial directions rather than explicit generator representatives. Selecting a particular linear complement of $\widehat { K } _ { \mathrm { t r i v } } ^ { ( k ) }$ would provide representatives, but such a complement is not canonical and would introduce an additional representation-dependent choice. Furthermore, removal of the exact sampled kernel would not prevent discovery along near-kernel directions whose prolonged-characteristic response is nonzero but numerically negligible.

For these reasons, nontriviality remains enforced through the relative response and minimum-magnitude constraints constituting ${ \mathcal { L } } _ { \mathrm { n t r } }$ in (66), now restricted to candidates already lying in $\widehat { N } _ { \mathrm { n a d d } } ^ { ( k ) }$ . Thus, unlike the infinitesimal symmetry condition and the additive solution sector, nontriviality is not converted into a further linear admissible restriction.

## 3.5 SEQUENTIAL DISCOVERY AND NOVELTY RE-STRICTIONS

The admissible restrictions introduced in Operator-Theoretic Symmetry Representation and Intrinsic Admissibility

Constraints determine which represented candidate generators are eligible for the recovery of ${ \mathfrak { g } } _ { \mathrm { i n t } }$ . Novelty, however, is not an intrinsic property of a candidate in isolation, but is defined relative to the generators already discovered. In particular, every successive generator must satisfy

$$
v _ { i } \not \in \mathrm { s p a n } _ { \mathbb { R } } \left\{ v _ { 1 } , \dots , v _ { i - 1 } \right\} , \quad \forall i > 1 .\tag{128}
$$

In the initial optimization formulation of Initial Multi-Objective Optimization Formulation, this requirement was imposed by decomposing $v _ { i }$ into the complementary component $v _ { \perp , i }$ and the component $v _ { \parallel , \ast }$ represented by the previously discovered span. The operator-theoretic formulation developed below instead constructs a novelty restriction that excludes this previously represented subspace directly from the space available to each successive discovery step.

The need for a global test of linear independence nevertheless remains unchanged. In particular, linear independence is an ambient vector-field property over real constants, and finite evaluation over an inadequately sampled domain may still introduce accidental rank loss or poor conditioning. We therefore retain the of-shell anchor dataset $\mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o f f } }$ introduced in (79) when constructing the operator-theoretic novelty restriction.

We now construct the operator-theoretic novelty restriction using the same of-shell ambient vector-field responses introduced in the initial optimization formulation. For each $z _ { a } ^ { ( s ) } \in \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o f f } }$ , let

$$
\mathbf { y } _ { E , a } \left[ v _ { \Theta } \right] : = \left. v _ { \Theta } \right| _ { z _ { a } } = \left[ \pmb { \xi } _ { \Theta } \left( z _ { a } \right) \right] \in \mathbb { R } ^ { p + q } ,\tag{129}
$$

where $z _ { a } : = \pi _ { 0 } ^ { s } \left( z _ { a } ^ { ( s ) } \right)$ . Thus, although the anchors belong to $J ^ { ( s ) } ( \pi )$ , novelty is evaluated through the zeroth-order responses of the ambient vector fields in $T _ { z _ { a } } E ,$ , rather than through their prolongations.

Stacking the responses over the fixed of-shell anchor dataset gives

$$
\begin{array} { r l } & { Y _ { E } \left[ v _ { \Theta } \right] : = } \\ & { \qquad \mathrm { v e c } \left( \left( \mathbf { y } _ { E , a } \left[ v _ { \Theta } \right] \right) _ { z _ { a } ^ { \left( s \right) } \in \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o f f } } } \right) \in \mathbb { R } ^ { A _ { \mathrm { o f f } } \left( p + q \right) } . } \end{array}\tag{130}
$$

As in the initial formulation in Initial Multi-Objective Optimization Formulation, the horizontal and vertical ambient responses are measured using a common blockwise metric. Let $d _ { \xi } > 0$ and $d _ { \varphi } > 0$ denote the fixed horizontal and vertical scaling factors, respectively, obtained from the normalization statistics associated with the of-shell anchor set. Define the single-anchor scaling matrix

$$
W _ { E , 0 } : = \left[ \begin{array} { c c } { { d _ { \pmb { \xi } } ^ { - 1 } I _ { p } } } & { { 0 } } \\ { { 0 } } & { { d _ { \varphi } ^ { - 1 } I _ { q } } } \end{array} \right] ,\tag{131}
$$

and its corresponding stacked metric

$$
{ \cal W } _ { E } : = { \cal I } _ { A _ { \mathrm { o f f } } } \otimes { \cal W } _ { E , 0 } .\tag{132}
$$

The same metric is therefore applied to every of-shell anchor and, importantly, to both the previously discovered generators and the candidate generator. We define the metricnormalized stacked response by

$$
\widetilde { Y } _ { E } \left[ v _ { \Theta } \right] : = W _ { E } Y _ { E } \left[ v _ { \Theta } \right] .\tag{133}
$$

The normalization statistics entering $W _ { E }$ are fixed during construction of the novelty restriction and are independent of the current coeficient vector Θ. Hence, $\widetilde { Y } _ { E } \left[ v _ { \Theta } \right]$ remains linear in $\Theta .$ . Details of the normalization statistics and their numerical evaluation are deferred to Section 4.

For the $i ^ { \mathrm { t h } }$ discovery step, $i > 1$ , the previously discovered generators define the normalized old-response matrix

$$
\widetilde { R } _ { E , i } : = \left[ \widetilde { \underline { { Y } } } _ { E } \left[ v _ { 1 } \right] \quad \cdot \cdot \quad \widetilde { Y } _ { E } \left[ v _ { i - 1 } \right] \right] .\tag{134}
$$

Equations (133) and (134) are the metric-normalized counterparts of $Y _ { i }$ in (88) and $R _ { i }$ in (87), respectively.

Let $\widetilde { Q } _ { E , i }$ be an orthonormal basis for the column space of $\widetilde { R } _ { E , i }$ obtained through a rank-revealing QR factorization, such that

$$
\begin{array} { r } { \operatorname { c o l } \left( \widetilde { Q } _ { E , i } \right) = \operatorname { c o l } \left( \widetilde { R } _ { E , i } \right) , \quad \widetilde { Q } _ { E , i } ^ { \top } \widetilde { Q } _ { E , i } = I . } \end{array}\tag{135}
$$

Since the previously discovered generators and the metric $W _ { E }$ are fixed during the $i ^ { \mathrm { t h } }$ discovery step, $\widetilde { Q } _ { E , i }$ is also fixed. We therefore define the sampled represented novelty operator

$$
\begin{array} { r } { \widehat { \mathrm { B } } _ { E , i } ^ { ( k ) } \Big | _ { { \bf S } _ { \mathrm { a n c } } ^ { \mathrm { o f f } } } \left[ \Theta \right] : = \widetilde { Q } _ { E , i } ^ { \top } \widetilde { Y } _ { E } \left[ v _ { \Theta } \right] . } \end{array}\tag{136}
$$

Because the realization $v _ { \Theta } = \Psi ^ { ( k ) } \left( \Theta \right)$ is linear in Θ, while $W _ { E }$ and $\widetilde { Q } _ { E , i }$ are fixed, the operator in (136) is linear in Θ. Its kernel consists of coeficient directions whose metricnormalized, globally stacked of-shell ambient responses are orthogonal to the response space generated by the previously discovered generators.

Accordingly, for every $i > 1$ , the parameter space available to the next discovery step is restricted to

$$
\begin{array} { r } { \widehat { N } _ { \mathrm { n o v } , i } ^ { ( k ) } : = \widehat { N } _ { \mathrm { n a d d } } ^ { ( k ) } \cap \ker \left( \left. \widehat { \mathrm { B } } _ { E , i } ^ { ( k ) } \right| _ { { \mathrm { S } } _ { \mathrm { a n c } } ^ { \mathrm { o f f } } } \right) . } \end{array}\tag{137}
$$

For the first generator, no previously discovered direction exists, and hence

$$
\widehat N _ { \mathrm { n o v , 1 } } ^ { ( k ) } : = \widehat N _ { \mathrm { n a d d } } ^ { ( k ) } .\tag{138}
$$

Thus, each successive discovery is carried out directly in a coeficient subspace whose metric-normalized of-shell ambient response is orthogonal to the response space already recovered. In contrast to the initial optimization formulation, no parallel response component $v _ { \parallel , i }$ needs to be learned and subsequently suppressed through a novelty loss.

## 4 IMPLEMENTATION AND COMPUTA-TIONAL CONSIDERATIONS

Section 3 established the theoretical framework underlying the proposed discovery procedure, but deliberately deferred implementation details that are central to its computational realization and did not describe the sequential discovery algorithm in operational terms. The present section addresses these aspects without attempting to document every lowlevel implementation detail. For further implementationspecific information, we refer the reader to the accompanying Code Ocean capsule, which provides the source code with suficient documentation and a modular package structure for detailed inspection. We begin with the fixed-dictionary realization of candidate generators, which preserves the linear dependence on the coeficient vector required by the operator formulation in (103). We then introduce the RSP test used to assess whether the represented determining system stabilizes under successive enrichment of the model. This is followed by the numerical enforcement of nontriviality and the metric normalization used in constructing the novelty restriction. Finally, we describe the sequential recovery procedure, including the criteria used to accept individual generators, determine whether further discovery steps should be attempted, and terminate the process once the available represented symmetry directions have been exhausted. A schematic flowchart summarizing the resulting training and discovery procedure is provided in Fig. E.2.

## 4.1 FIXED DICTIONARY AND GENERATOR RE-ALIZATION

As established in Operator-Theoretic Symmetry Representation, the composition of the determining operator in (99) with the realization map in (102) must remain linear in the model parameters Θ. This property permits the candidate coeficients to be restricted directly to ker $\left( \widehat { M } _ { \Delta } ^ { ( k ) } \right)$ through (110), thereby excluding symmetryviolating directions before optimization. A conventional neural-network realization does not, in general, preserve this linear dependence on its trainable parameters. We therefore replace our initial neural-network model by a fixed spline-based dictionary that retains suficient representational richness while remaining exactly linear in Θ. The construction is inspired by the spline parametrization employed in Kolmogorov–Arnold Networks (KANs) [39], as previously noted in Operator-Theoretic Symmetry Representation; unlike a general trainable KAN architecture, however, the grid and basis functions are fixed and only their expansion coeficients are optimized.

Let $\pmb { \chi } = \left( \chi ^ { 1 } , \dots , \chi ^ { d } \right)$ denote the input to the generator realization, and let $\begin{array} { r } { \Omega = \prod _ { i = 1 } ^ { d } [ a _ { i } , b _ { i } ] \subset \mathbb { R } ^ { d } } \end{array}$ be a domain containing both the training and validation datasets. We intentionally do not identify d with $p + q$ , since, as discussed in Section $^ { 6 , }$ the input dimension of the generator realization may vary with the class and order of the underlying DE system. For each coordinate $\chi ^ { i }$ , let $\left\{ B _ { \nu _ { i } } ^ { ( r ) } ( \chi ^ { i } ) \right\} _ { \nu _ { i } = 1 } ^ { K _ { o } }$ denote a fixed degree-r B-spline basis constructed over $[ a _ { i } , b _ { i } ]$ on a grid of resolution o using the Cox–de Boor recursion [40], where, under the adopted convention that o denotes the number of uniform core-grid intervals, $K _ { o } = o + r$

To incorporate multivariate dependence while preserving linearity in the expansion coeficients, we construct tensorproduct interactions among the univariate spline bases. For a nonempty subset $S \subseteq \{ 1 , \ldots , d \}$ and an associated splineindex tuple $\nu _ { S } = ( \nu _ { i } ) _ { i \in S }$ , define

$$
B _ { S , \pmb { \nu } _ { S } } ^ { ( r ) } \left( \pmb { \chi } \right) : = \prod _ { i \in S } B _ { \nu _ { i } } ^ { ( r ) } \left( \chi ^ { i } \right) .\tag{139}
$$

We define the interaction order k as the maximum number of distinct input coordinates entering any such tensor product. Accordingly, the fixed dictionary is

$$
\begin{array} { r l r } & { } & { D _ { ( k ) } : = \Big \{ B _ { S , \nu _ { S } } ^ { ( r ) } ~ \Big | ~ \emptyset \neq S \subseteq \{ 1 , \dots , d \} , } \\ & { } & { | S | \leq k , ~ 1 \leq \nu _ { i } \leq K _ { o } , ~ i \in S \Big \} . } \end{array}\tag{140}
$$

Thus, $k = 1$ yields an additive dictionary consisting only of univariate spline features, whereas $k = 2$ augments these features by all pairwise tensor products associated with distinct input coordinates. Larger values of k analogously introduce higher-order coordinate interactions without altering the linear dependence of the realized generator on its coeficient vector Θ.

Remark 6. Throughout the remainder of this section, k denotes the maximum interaction order admitted by the nested dictionary $D _ { ( k ) } ;$ the same index appearing as the superscript (k) on $\Psi ^ { ( k ) }$ and the associated sampled operator and coeficient-space objects in Section 3 indicates that they are constructedfrom this dictionary level.

The total number of scalar features listed in $D _ { ( k ) }$ is

$$
n _ { ( k ) } ^ { D } : = \left| D _ { ( k ) } \right| = \sum _ { \ell = 1 } ^ { k } { \binom { d } { \ell } } K _ { o } ^ { \ell } ,\tag{141}
$$

where $K _ { o } ^ { \ell }$ is the number of ℓ-way tensor-product spline features associated with one fixed subset of ℓ input coordinates. We emphasize that $n _ { ( k ) } ^ { D }$ counts dictionary entries rather than the dimension of their linear span. Although the univariate collection associated with each individual coordinate forms a B-spline basis, the combined dictionary $D _ { ( k ) }$ may contain exact linear dependencies across diferent interaction orders, as discussed below.

For illustration, consider $d = 3$ and $K _ { o } = 4$ . The first-order feature block is

$$
\mathbf { B } _ { 1 } ^ { ( r ) } : = \left\{ B _ { \nu _ { i } } ^ { ( r ) } \left( \chi ^ { i } \right) \Big | i \in \{ 1 , 2 , 3 \} , \nu _ { i } \in \{ 1 , 2 , 3 , 4 \} \right\} ,\tag{142}
$$

and therefore $\big | { \bf B } _ { 1 } ^ { ( r ) } \big | = 3 \times 4 = 1 2$ . For $\ell \ = \ 2$ , each feature has the form $B _ { \nu _ { i } } ^ { ( r ) } \left( \chi ^ { i } \right) B _ { \nu _ { j } } ^ { ( r ) } \left( \chi ^ { j } \right)$ with $i < j , \mathrm { \bf ~ A }$ fixed coordinate pair $( i , j )$ admits $4 \times 4 = 1 6$ choices of the spline-index pair $( \nu _ { i } , \nu _ { j } )$ , while ${ \binom { 3 } { 2 } } = 3$ distinct coordinate pairs are available. Hence, $\left| \mathbf { B } _ { 2 } ^ { ( r ) } \right| = 3 \times 4 ^ { 2 } = 4 8$ , and

$$
n _ { ( 2 ) } ^ { D } = \left| { \bf B } _ { 1 } ^ { ( r ) } \right| + \left| { \bf B } _ { 2 } ^ { ( r ) } \right| = 6 0 .\tag{143}
$$

More generally, the interaction-order hierarchy induces the nested enrichment

$$
\begin{array} { c } { { \displaystyle D _ { ( k + 1 ) } = D _ { ( k ) } \cup \Bigg \{ \prod _ { m = 1 } ^ { k + 1 } B _ { \nu _ { i _ { m } } } ^ { ( r ) } \left( \chi ^ { i _ { m } } \right) \Big \vert } } \\ { { 1 \leq i _ { 1 } < . . . < i _ { k + 1 } \leq d , 1 \leq \nu _ { i _ { m } } \leq K _ { o } \Bigg \} . } } \end{array}\tag{144}
$$

Although the realization remains linear in the trainable coeficients $\mathbf { \Theta } _ { \Theta } ,$ its dependence on the dictionary inputs need not be linear. At the univariate level, each $B _ { \nu _ { i } } ^ { ( r ) } \left( \chi ^ { i } \right)$ is a localized degree-r piecewise-polynomial function, so linear combinations of these basis functions permit nonlinear dependence of the generator coeficients, including the components $\xi ^ { i }$ and $\varphi ^ { \alpha }$ , on their admissible inputs. Increasing the grid resolution o enriches this univariate dependence by introducing additional localized degrees of freedom, while increasing the spline degree r enlarges the local polynomial complexity. At the multivariate level, increasing the interaction order k introduces tensor-product features coupling progressively larger subsets of distinct input coordinates. Hence, the representational richness of $D _ { ( k ) }$ arises jointly from the local spline resolution and degree at the B-spline level and from cross-coordinate interactions at the dictionary level, while the realization remains exactly linear in Θ.

We emphasize, however, that this increasing representational richness should not be interpreted as an established completeness property of the finite dictionary. At present, we have not established a universal-approximation /density theorem or a quantitative approximation bound for the truncated KANinspired dictionary $D _ { ( k ) }$ . Consequently, failure to represent a particular symmetry direction may reflect limitations of the chosen finite values of $o , r ,$ and $k ,$ rather than absence of that symmetry from the underlying DE system. The computational role of successive dictionary enrichment and the corresponding rank-stabilization criterion are discussed in Rank-Stabilization Preflight (RSP) Test.

The nested enrichment in (144) does not, in general, imply a corresponding increase in the dimension of the represented function space. Exact cross-order redundancies induced by the partition-of-unity property of B-splines, together with additional rank deficiencies that may arise from insuficient sampling of their local supports, are discussed in B.

More explicitly, any scalar coeficient function represented

through the fixed dictionary can be written as

$$
f _ { \pmb \theta } \left( \pmb \chi \right) = \sum _ { \ell = 1 } ^ { n _ { ( k ) } ^ { D } } \theta ^ { \ell } \psi _ { \ell } \left( \pmb \chi \right) , \quad \psi _ { \ell } \in D _ { ( k ) } ,\tag{145}
$$

where θ denotes the corresponding coeficient block of the global parameter vector Θ. Thus, the potentially nonlinear dependence on $x$ is carried entirely by the fixed dictionary functions $\psi _ { \ell } .$ , whereas the dependence on the trainable coeficients remains linear. The coeficient blocks associated with the admissible components of $\boldsymbol { \xi }$ and $\varphi$ are then assembled into Θ to define the realization $v _ { \Theta } = \Psi ^ { ( k ) } \left( \Theta \right)$

The fixed dictionary $D _ { ( k ) }$ provides the functional building blocks of the generator realization, but the admissible dependence of the generator components on the coordinates is selected according to the class, linearity, and diferential order of the underlying DE system. Accordingly, the implementation instantiates either a structurally restricted projectible fiber-afine realization, when such a form is appropriate, or the more general point-field realization in which the coeficient functions are expanded directly over $D _ { ( k ) }$ . This choice determines both the efective dictionary input dimension d introduced above and the total coeficient dimension P entering the realization map $\Psi ^ { ( k ) }$ in (102).

In either case, the basis functions and spline grids remain fixed, and only their expansion coeficients are trainable. Hence, the resulting vector field $v _ { \Theta }$ remains linear in Θ, preserving the coeficient-space operator formulation developed in Operator-Theoretic Symmetry Representation. The mathematical conditions motivating the realization selected for each DE class, together with the corresponding applicability and completeness restrictions, are discussed separately in Section 6.

Although the framework operates at the infinitesimal level, a discovered generator can subsequently be integrated to recover its associated local one-parameter transformation. For a discovered generator $v _ { i }$ and an initial jet $z _ { 0 } ^ { ( s ) } \in$ $J ^ { ( s ) } \left( \pi \right)$ , we numerically solve

$$
\frac { \mathrm { d } z ^ { ( s ) } } { \mathrm { d } \epsilon } = \mathrm { p r } ^ { ( s ) } \left( v _ { i } \right) \Big | _ { z ^ { ( s ) } ( \epsilon ) } , \quad z ^ { ( s ) } \left( 0 \right) = z _ { 0 } ^ { ( s ) } .\tag{146}
$$

The resulting trajectory approximates the local prolonged flow defined in (32) and induced by the one-parameter subgroup exp (ϵv<sub>i</sub>). In practice, (146) is integrated using a diferentiable ODE solver of the type employed in neural-ODE frameworks [16]; hence, reconstruction of the finite transformation does not require a closed-form evaluation of the abstract Lie-group exponential in (30).

The reconstruction also connects directly with the grouplevel representation in (31). Once a basis $\{ v _ { 1 } , \ldots , v _ { r } \}$ of the recovered Lie algebra has been obtained, finite products of the numerically reconstructed one-parameter flows realize transformations in the identity-connected component $G _ { 0 }$ . In particular, suficiently near the identity, an ordered product

$$
g \left( \epsilon \right) = \exp \left( \epsilon ^ { 1 } v _ { 1 } \right) \cdot \cdot \cdot \exp \left( \epsilon ^ { r } v _ { r } \right)\tag{147}
$$

provides canonical coordinates of the second kind, with the ordering becoming relevant when the recovered algebra is non-Abelian. Although such coordinates are local, a connected transformation can in principle be constructed patchwise by composing a sequence of suficiently small second-kind transformations, each remaining within a neighborhood where the local parametrization is valid. The numerical results in Section 5 provide evidence for the practical need for such a patchwise strategy when transporting beyond the region over which a single learned local realization remains accurate; a systematic patchwise-transport construction is left for future work.

## 4.2 RANK-STABILIZATION PREFLIGHT (RSP) TEST

Although the applicability of the proposed framework is discussed in detail in Section 6, we note here that certain classes of DE systems may admit an infinite-dimensional ${ \mathfrak { g } } _ { \mathrm { i n t } }$ through arbitrary functional freedom in their determining systems. In contrast, any finite dictionary $D _ { ( k ) }$ necessarily yields only a finite-dimensional represented coeficient space and, consequently, a finite-dimensional represented geometric symmetry space. Symbolic and formal approaches address such functional freedom by analyzing the determining system after compatibility completion and reduction to an involutive or standard form [10, 11, 32]. For background on the geometric meaning of functional freedom and involutivity, we refer the reader to A, together with [31, Chapters 14–15] and [3, Chapters 4–5].

At present, we have not developed a numerical analogue capable of simultaneously diagnosing arbitrary functional freedom and carrying out the compatibility and involutivity analysis associated with Cartan–Kuranishi completion and the Cartan–Kähler framework [32, Chapter 7]. Developing such a procedure is beyond the scope of the present work, although Section 7 outlines a possible trajectory toward it within the operator-theoretic framework introduced here. Instead, we employ a computational RSP test: the represented determining system is recomputed under successive enrichments of $D _ { ( k ) }$ , and the evolution of its numerically resolved rank and nullity is monitored to determine whether the represented symmetry dimension stabilizes.

Choose a sequence of nested knot refinements for the KAN dictionaries introduced in Fixed Dictionary and Generator Realization, while initially fixing the interaction order k and spline degree r. For grid resolutions $o _ { 1 } < o _ { 2 } < \cdots$ , define the corresponding dictionary-generated spaces by

$$
\begin{array} { r } { \mathcal V _ { i } ^ { ( k , r ) } : = \mathrm { s p a n } \left( D _ { ( k ) , o _ { i } } ^ { ( r ) } \right) , \quad \mathcal V _ { 1 } ^ { ( k , r ) } \subseteq \mathcal V _ { 2 } ^ { ( k , r ) } \subseteq \cdots . } \end{array}\tag{148}
$$

Here, each knot grid is chosen as a refinement of its predecessor, so that the coarser spline space is exactly representable on the finer grid through standard knot refinement. Accordingly, $D _ { ( k ) , o _ { i + 1 } } ^ { ( r ) }$ provides a richer coeficient representation than $D _ { ( k ) , o _ { i } } ^ { ( r ) } ,$ although the individual B-spline basis functions themselves need not form nested sets.

For the uniform spline grids used here, this nesting can be enforced by choosing successive resolutions such that $o _ { i + 1 }$ $m _ { i } o _ { i }$ with $m _ { i } \in \mathbb { N }$ and $m _ { i } > 1$ . Over a fixed core interval, every knot of the coarser grid is then retained by the finer grid, and, at fixed spline degree r, standard knot refinement yields the required inclusion of the corresponding spline spaces. Thus, the refinement increases only the representational resolution while preserving the underlying physical domain.

Let $\Psi _ { i } ^ { ( k ) } : \mathbb { R } ^ { P _ { i } }  \mathfrak { X } ( E )$ denote the realization induced by $D _ { ( k ) , o _ { i } } ^ { ( r ) }$ , and let

$$
\widehat { N } _ { \Delta , i } ^ { ( k ) } : = \ker \left( \widehat { M } _ { \Delta , i } ^ { ( k ) } \right)\tag{149}
$$

be the corresponding represented sampled determining nullspace in (110). Since the coeficient dimensions $P _ { i }$ vary under refinement, the spaces $\widehat { N } _ { \Delta , i } ^ { ( k ) }$ are not compared by literal coeficient-space inclusion. Instead, after identifying a coarse realization with its representation on the refined spline grid, the nested function spaces in (148) imply the corresponding inclusion of represented sampled symmetry fields,

$$
\Psi _ { i } ^ { ( k ) } \left( \widehat { N } _ { \Delta , i } ^ { ( k ) } \right) \subseteq \Psi _ { i + 1 } ^ { ( k ) } \left( \widehat { N } _ { \Delta , i + 1 } ^ { ( k ) } \right) ,\tag{150}
$$

up to the numerical rank tolerance used in constructing the sampled kernels.

Intrinsic Admissibility Constraints identified one source of arbitrary functional freedom that is known a priori for linear DE systems, namely the additive solution sector $\boldsymbol { A } _ { \mathrm { s o l } }$ . Since this sector can be isolated directly within the represented determining kernel, we define the admissible space used by the preflight test as

$$
\widehat { N } _ { \mathrm { a d m } , i } ^ { ( k ) } : = \left\{ \begin{array} { l l } { \widehat { N } _ { \mathrm { n a d d } , i } ^ { ( k ) } , } & { \mathrm { l i n e a r ~ s y s t e m s } , } \\ { \widehat { N } _ { \Delta , i } ^ { ( k ) } , } & { \mathrm { n o n l i n e a r ~ s y s t e m s } . } \end{array} \right.\tag{151}
$$

The first branch removes the represented additive solution sector before the stabilization test is applied. As emphasized in Intrinsic Admissibility Constraints and A, however, this does not preclude other forms of arbitrary functional freedom in the exact determining system. At every finite refinement level, $\widehat { N } _ { \mathrm { a d m } , i } ^ { ( k ) }$ remains finite-dimensional; if unresolved functional freedom is present in the underlying determining equations, its numerical signature may instead appear as continued growth of the represented admissible dimension under refinement.

Raw determining nullity alone is not suficient for this purpose. It may increase because of additive directions, coefficient nonidentifiability, or sampled-trivial directions that do not represent additional intrinsic geometric symmetries. We therefore also evaluate the sampled prolonged-characteristic operator associated with each refinement level, $\widehat { \mathcal { T } } _ { \mathrm { t r i v } , i } ^ { ( k ) } , \mathrm { r e } -$ stricted to ${ \widehat { N } } _ { \mathrm { a d m } , i } ^ { ( k ) } ,$ and define

$$
\begin{array} { r l } & { d _ { o _ { i } } : = \mathrm { r a n k } \left( \left. \widehat { \mathcal { T } } _ { \mathrm { t r i v } , i } ^ { ( k ) } \right. _ { \widehat { N } _ { \mathrm { a d m } , i } ^ { ( k ) } } \right) } \\ & { \quad = \dim \left( \widehat { N } _ { \mathrm { a d m } , i } ^ { ( k ) } / \widehat { K } _ { \mathrm { t r i v } , i } ^ { ( k ) } \right) = \dim \left( \widehat { N } _ { \mathrm { g e o m } , i } ^ { ( k ) } \right) , } \end{array}\tag{152}
$$

where $\widehat K _ { \mathrm { t r i v } , i } ^ { ( k ) } = \mathrm { k e r } \left( \widehat T _ { \mathrm { t r i v } , i } ^ { ( k ) } \Big | _ { \widehat N _ { \mathrm { a d m } , i } ^ { ( k ) } } \right)$ and $\widehat { N } _ { \mathrm { g e o m } , i } ^ { ( k ) }$ is the sampled geometric quotient defined in Intrinsic Admissibility Constraints. We refer to $d _ { o _ { i } }$ as the represented geometric rank at refinement level $o _ { i }$

All ranks and nullspaces entering the preflight test are numerical. For a sampled matrix A with singular values $\sigma _ { 1 } \geq \sigma _ { 2 } \geq \cdot \cdot \cdot$ , we define

$$
\operatorname { r a n k } _ { \tau } \left( A \right) : = \# \left\{ j \ | \ \sigma _ { j } > \tau _ { \operatorname { r a n k } } \sigma _ { 1 } \right\} ,\tag{153}
$$

using the same relative threshold $\tau _ { \mathrm { r a n k } }$ throughout a refinement sequence. In practice, the stability of the inferred rank is additionally checked under a modest variation of this threshold so that a plateau is not determined by a single marginal singular value.

The refinement sequence is regarded as numerically stabilized only when the represented geometric rank remains unchanged over a prescribed number of successive refinements,

$$
d _ { o _ { j } } = d _ { o _ { j + 1 } } = \cdot \cdot \cdot = d _ { o _ { j + L _ { \mathrm { s t a b } } } } ,\tag{154}
$$

for some prescribed $L _ { \mathrm { s t a b } } \ge 1$ , while the associated numerical ranks remain robust to the chosen SVD tolerance. Such a plateau is interpreted operationally as stabilization of the represented intrinsic symmetry dimension along the selected dictionary-refinement path; it is not, by itself, a formal proof that the exact determining system possesses no additional functional freedom. If the represented geometric rank continues to increase, or if the inferred plateau is numerically unstable, the sequential discovery stage is not initiated and the framework makes no completeness claim for g<sub>int</sub>.

Although the construction above uses successive grid refinements at fixed k and r as the primary enrichment path, the same stabilization principle may be applied to any genuinely nested sequence of represented function spaces. In particular, interaction order or spline degree may also be enriched when warranted by the realization under consideration, provided that the resulting spaces admit a consistent nested comparison.

## 4.3 NUMERICAL ENFORCEMENT OF NONTRIVI-ALITY

Proposition 3 in Intrinsic Admissibility Constraints showed that sampled triviality is characterized by the kernel $\widehat { K } _ { \mathrm { t r i v } } ^ { ( k ) }$ of the prolonged-characteristic operator, while the corresponding geometric information is represented by the quotient $\widehat { N } _ { \mathrm { g e o m } } ^ { \bar { ( } k \bar { ) } }$ . Unlike the determining and novelty conditions, however, nontriviality cannot be imposed by restricting the candidate to another linear admissible subspace. Indeed, the nontrivial feasible set ${ \widehat { \mathcal { V } } } _ { \mathrm { n t r } } ^ { ( k ) }$ defined in (127) excludes the sampled-trivial kernel from the admissible coeficient space and therefore does not itself form a vector subspace.

Consequently, even after the linear determining, solutionindependence, and novelty restrictions have been imposed, optimization must still prevent a candidate generator from collapsing toward $\widehat { K } _ { \mathrm { t r i v } } ^ { ( k ) }$ , including the zero vector field. For this reason, we retain the nonlinear loss ${ \mathcal { L } } _ { \mathrm { n t r } }$ introduced in (66). The remainder ofthis subsection describes its numerical realization and the continuation strategy used to enforce the corresponding minimum-response condition during training.

The analytical form of ${ \mathcal { L } } _ { \mathrm { n t r } }$ , including the relative prolongedcharacteristic response, the minimum-magnitude requirement, and the quadratic exterior penalty P, was introduced in Initial Multi-Objective Optimization Formulation. We therefore do not repeat its derivation here. Instead, we describe the numerical mechanisms used to evaluate and stabilize these quantities during optimization, including the lowertail response statistic, smooth activation of the nontriviality penalty, regularization of small denominators, auxiliary speed-tail control, and continuation of the penalty parameter γ in (65).

Let $\mathcal { R } _ { \mathrm { n t r } , l } \left( v \right)$ in (63) denote the relative prolongedcharacteristic response of v at $z _ { l } ^ { ( s ) } \in \mathbf { S } ^ { \mathrm { o n } }$ . If only the mean response

$$
\overline { { \mathcal { R } } } _ { \mathrm { n t r } } \left( v \right) : = \frac { 1 } { N _ { \mathrm { o n } } } \sum _ { l = 1 } ^ { N _ { \mathrm { o n } } } { \mathcal { R } } _ { \mathrm { n t r } , l } \left( v \right)\tag{155}
$$

is monitored, then v may exhibit a strong prolongedcharacteristic response over most samples while becoming nearly trivial on a non-negligible portion of the on-shell dataset, without the average necessarily revealing this behavior. We therefore define

$$
\mathcal { R } _ { \mathrm { t a i l } } ^ { \rho } \left( v \right) : = q _ { \rho } \left( \left\{ \mathcal { R } _ { \mathrm { n t r } , l } \left( v \right) \right\} _ { l } \right) ,\tag{156}
$$

where $q _ { \rho }$ denotes the empirical ρ-quantile. For small $\rho ,$ typically $\rho \leq 0 . 1$ , we refer to $\mathcal { R } _ { \mathrm { t a i l } } ^ { \rho }$ as the lower-tail statistic.

Rather than enforcing the nontriviality penalty uniformly throughout training, we use this statistic to construct the smooth activation gate

$$
g _ { \mathrm { n t r } } ^ { \rho } \left( \boldsymbol { v } \right) : = \sigma \left[ \kappa _ { \mathrm { g n t r } } \left( \tau _ { \mathrm { g n t r } } - \mathcal { R } _ { \mathrm { t a i l } } ^ { \rho } \left( \boldsymbol { v } \right) \right) \right] ,\tag{157}
$$

where σ is the sigmoid function, while $\kappa _ { \mathrm { g n t r } } > 0$ and $\tau _ { \mathrm { g n t r } } > 0$ denote the gate-sharpening parameter and gateactivation threshold, respectively. Thus, $g _ { \mathrm { n t r } } ^ { \rho }$ approaches unity when the lower tail of the relative response becomes suficiently small and approaches zero when the sampled response remains safely away from the trivial sector.

The nontriviality loss in (66) is consequently evaluated in gated form as

$$
\mathcal { L } _ { \mathrm { t a i l } } : = \mathcal { P } \left( c _ { \mathrm { m i n } } \right) + \frac { g _ { \mathrm { n t r } } ^ { \rho } \left( v \right) } { B } \sum _ { z _ { l } ^ { \left( s \right) } \in { \bf S } _ { B } ^ { \mathrm { o n } } } \mathcal { P } \left( c _ { \mathrm { n t r } , l } \right) ,\tag{158}
$$

where $c _ { \mathrm { n t r } , l } = \tau _ { \mathrm { n t r } } - \mathcal { R } _ { \mathrm { n t r } , l } \left( v \right) \leq 0$ is the samplewise nontriviality inequality constraint introduced previously, and $c _ { \mathrm { m i n } }$ is the minimum-magnitude inequality constraint in (62). The lower-tail statistic therefore does not replace the samplewise nontriviality condition; rather, it controls when that condition is actively emphasized during optimization.

A separate numerical safeguard is used to discourage pathological spatial concentration of the recovered vector field. For $z _ { l } ^ { ( s ) } \in \mathbf { S } ^ { \mathrm { o n } }$ , let $z _ { l } = \pi _ { 0 } ^ { s } \left( z _ { l } ^ { ( s ) } \right) \in E$ and define

$$
s _ { l } : = \| v \| _ { z _ { l } } \| _ { 2 }\tag{159}
$$

as the pointwise speed of the candidate generator. During optimization, we use a smoother upper-tail statistic. Let $I _ { \eta }$ contain the indices of the largest fraction η of the sampled speeds, with

$$
\overline { { s } } _ { \eta } : = \frac { 1 } { | I _ { \eta } | } \sum _ { l \in I _ { \eta } } s _ { l } , \quad | I _ { \eta } | = \lceil \eta B \rceil .\tag{160}
$$

The corresponding speed-concentration penalty is

$$
\mathcal { L } _ { \mathrm { s p k } } : = \operatorname* { m a x } \left( 0 , \log \left( \frac { \overline { { s } } _ { \eta } } { s \mathbf { g } \left( q _ { 5 0 } \right) } \right) - \log \rho _ { \mathrm { s p k } } \right) ^ { 2 } ,\tag{161}
$$

where sg denotes a stop-gradient operation and $\rho _ { \mathrm { s p k } } \ >$ 1 is the tolerated upper-tail ratio. Detaching the median prevents the optimizer from reducing the upper-tail ratio by increasing the typical speed over the sampled domain; consequently, the gradient of this regularizer acts only to suppress the localized high-speed tail. The nontriviality objective is therefore augmented as

$$
\mathcal { L } _ { \mathrm { n t r } } ^ { \mathrm { r e g } } : = \mathcal { L } _ { \mathrm { t a i l } } + \lambda _ { \mathrm { s p k } } \mathcal { L } _ { \mathrm { s p k } } ,\tag{162}
$$

with $\lambda _ { \mathrm { s p k } } > 0$ controlling the strength of this auxiliary safeguard. In the present implementation, $\eta = 0 . 0 1$ , so that the regularizer acts on the mean of the largest 1% of sampled speeds.

Remark 7. Although the speed-concentration penalty (161) does not directly constrain derivatives ofthe recovered vector field, it suppresses localized amplitude spikes that, in our experiments, were associated with strong point-to-point variations and unstable numerical integration of corresponding finite flow $\mathrm { F l } _ { v } ^ { z } ( \epsilon )$

Finally, the quadratic exterior penalty P introduced in (65) is enforced through a continuation strategy in its penalty parameter γ. Rather than imposing the final penalty strength from the beginning of training, $\gamma$ is initialized at a smaller value and increased progressively toward its prescribed final value as optimization proceeds. This allows the candidate generator to first move toward a favorable region of parameter space before the nontriviality inequalities are enforced more sharply, thereby reducing numerical stifness during the early stages of training. The precise continuation schedule and associated implementation parameters are provided in the accompanying source code.

## 4.4 METRIC-NORMALIZED NOVELTY RESTRIC-TION

The metric-normalized novelty construction, including $W _ { E } .$ the normalized stacked responses in (132), and the sampled novelty operator $\widehat { \mathrm { B } } _ { E , i } ^ { ( k ) }$ , was introduced in Sequential Discovery and Novelty Restrictions. We therefore restrict the present discussion to the scale factors $d _ { \xi } > 0$ and $d _ { \varphi } > 0$ deferred from that section. The horizontal and vertical zeroth-order ambient responses over $\mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o f f } }$ are assigned the RMS scales $s _ { \xi }$ and $s _ { \varphi }$ according to $( 5 6 ) – ( 5 7 )$ , using the pooled responses of the current candidate and previously discovered generators. We then set

$$
d _ { \pm } : = ( s _ { \pm } + \epsilon ) \sqrt { p } , \quad d _ { \varphi } : = ( s _ { \varphi } + \epsilon ) \sqrt { q } ,\tag{163}
$$

where the dimensional factors compensate for the natural growth of the Euclidean block norm with the number of components.

The scale estimates are updated periodically through the EMA introduced in (57). At each refresh, the updated values are detached and used to construct a common metric $W _ { E } ,$ after which the metric-dependent novelty restriction in (137) is rebuilt. Both $W _ { E }$ and the resulting restriction are then held fixed until the next scheduled refresh. The same frozen $W _ { E }$ is applied to the previously recovered responses and to the current candidate-response operator, ensuring that both are compared within the same normalized ambient geometry.

## 4.5 SEQUENTIAL RECOVERY AND STOPPING CRITERIA

Sequential Discovery and Novelty Restrictions established the construction of the search subspace $\widehat { N } _ { \mathrm { n o v } , i } ^ { ( k ) }$ , but did not specify the sequential algorithm by which the generators of ${ \mathfrak { g } } _ { \mathrm { i n t } }$ are recovered one at a time or the conditions under which this process terminates. The present subsection addresses both aspects from an algorithmic perspective. The complete sequential discovery lifecycle, including initialization, optimization, validation, acceptance, novelty-space reconstruction, and termination branches, is summarized schematically in E.

We assume throughout that $\Delta ^ { ( s ) } = \mathbf { 0 }$ lies within the applicability scope specified in Section 6 and, whenever required, has passed the RSP test in Rank-Stabilization Preflight (RSP) Test. For the selected dictionary level, $\widehat { N } _ { \Delta } ^ { ( k ) }$ and subsequently $\widehat { N } _ { \mathrm { n a d d } } ^ { ( k ) }$ are constructed from $\mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } }$ before sequential optimization begins, as described in Operator-Theoretic Symmetry Representation and Intrinsic Admissibility Constraints. For nonlinear systems, where no additive solution sector is removed, we set $\widehat { N } _ { \mathrm { n a d d } } ^ { ( k ) } = \widehat { N } _ { \Delta } ^ { ( k ) }$ ; the DE classes for which this realization is applicable are discussed in Section 6. At discovery step $i = 1 , 2 , \dots$ , the initial linear search subspace is then

$$
\begin{array} { r } { \widehat { N } _ { \mathrm { n o v } , i } ^ { ( k ) } = \left\{ \begin{array} { l l } { \widehat { N } _ { \mathrm { n a d d } } ^ { ( k ) } , } & { i = 1 , } \\ { \widehat { N } _ { \mathrm { n o v } , i } ^ { ( k ) } \mathrm { i n } ( 1 3 7 ) , } & { i > 1 . } \end{array} \right. } \end{array}\tag{164}
$$

Thus, only the novelty restriction evolves from one accepted generator to the next, while the determining and nonadditive admissible restrictions remain fixed at the chosen dictionary level throughout the entire discovery of ${ \mathfrak { g } } _ { \mathrm { i n t } }$

No additional solvability-based Lie-bracket restriction is imposed on the successively recovered generators. As shown in C, enforcing a normal-subalgebra chain during sequential recovery can make admissibility depend on the numerically encountered recovery path and can exclude otherwise legitimate directions that remain representable by $D _ { ( k ) }$

Before optimization, we also evaluate the spatial concentration of a candidate initialization. Let $q _ { p } \left( v _ { \Theta } \right)$ denote the empirical $p ^ { \mathrm { t h } }$ percentile of the sampled pointwise speeds defined in (159). We define the speed-tail ratio

$$
\rho _ { 9 9 } \left( v _ { \Theta } \right) : = \frac { q _ { 9 9 } \left( v _ { \Theta } \right) } { q _ { 5 0 } \left( v _ { \Theta } \right) } .\tag{165}
$$

Large values of $\rho _ { 9 9 }$ indicate that the vector-field magnitude is concentrated over a comparatively small portion of the sampled region. Although $\rho _ { 9 9 }$ does not directly constrain derivatives of the vector field, such concentration was associated in our numerical experiments with strong point-to-point variations and poor numerical integration of the corresponding finite flow (see 7). We therefore also use it as an initialization safeguard.

To initialize the $i ^ { \mathrm { t h } }$ candidate, let $U _ { i }$ denote an orthonormal coeficient-space basis of $\widehat { N } _ { \mathrm { n o v } , i } ^ { ( k ) }$ and draw

$$
\mathbf { a } _ { i } \sim \mathcal { N } \left( \mathbf { 0 } , I _ { \mathrm { d i m } \left( \widehat { N } _ { \mathrm { n o v } , i } ^ { ( k ) } \right) } \right) .\tag{166}
$$

The initial coeficient vector is then

$$
\Theta _ { i } : = U _ { i } \frac { \mathbf { a } _ { i } } { \left\| \mathbf { a } _ { i } \right\| _ { 2 } } ,\tag{167}
$$

so that $\Theta _ { i }$ lies directly in the current linear search subspace. As discussed in B, the realization $\Psi ^ { ( k ) }$ need not be injective: distinct coeficient vectors may represent the same vector field because of exact or sampling-induced dependencies in $D _ { ( k ) }$ . This pathological situation is called dictionary nonidentifiability. Consequently, the magnitude of $\Theta _ { i }$ alone is not a reliable measure of the visibility of its realized vector field.

An initialization is admitted only if (i) its ambient response is visible and (ii) its speed distribution is not excessively concentrated. We therefore define the of-shell ambientresponse magnitude

$$
\begin{array} { l } { { \displaystyle { \mathrm { R M S } _ { E } } \left( { \boldsymbol { v } } _ { \Theta _ { i } } \right) : = } \ ~ } \\ { { \displaystyle \left[ \frac { 1 } { A _ { \mathrm { o f f } } } \sum _ { z _ { a } ^ { ( s ) } \in \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o f f } } } \left. { \boldsymbol { v } } _ { \Theta _ { i } } \right. _ { { \boldsymbol { \pi } } _ { 0 } ^ { s } \left( z _ { a } ^ { ( s ) } \right) } \right. _ { 2 } ^ { 2 } } \ } .  \end{array}\tag{168}
$$

The initialization is accepted when

$$
\mathrm { R M S } _ { E } \left( v _ { \Theta _ { i } } \right) > \tau _ { \mathrm { v i s } } , \quad \rho _ { 9 9 } \left( v _ { \Theta _ { i } } \right) < \tau _ { \mathrm { s p d } } ,\tag{169}
$$

where $\tau _ { \mathrm { v i s } } ~ > ~ 0$ and $\tau _ { \mathrm { s p d } } > 0$ denote the visibility and maximum speed-tail thresholds, respectively. Random initialization is repeated until these conditions are satisfied or the prescribed initialization budget is exhausted; the latter case is incorporated into the stopping logic discussed below.

During optimization, the metric-dependent novelty restriction is periodically refreshed every $t _ { \mathrm { n o v } }$ training steps using the most recent normalization statistics entering $W _ { E } .$ following the construction in Metric-Normalized Novelty Restriction. Since this refresh generally changes the orthonormal basis $U _ { i }$ of $\widehat { N } _ { \mathrm { n o v } , i } ^ { ( k ) }$ , the current coeficient vector $\Theta _ { i }$ is reprojected onto the updated search subspace. Between successive refreshes, let

$$
P _ { i } : = U _ { i } U _ { i } ^ { \mathsf { T } }\tag{170}
$$

denote the orthogonal projector onto $\widehat { N } _ { \mathrm { n o v } , i } ^ { ( k ) } .$ . At every training step $t ,$ the optimizer update $\delta \Theta _ { i } ^ { t }$ is replaced by

$$
\delta \Theta _ { i } ^ { t } \longmapsto P _ { i } \delta \Theta _ { i } ^ { t } ,\tag{171}
$$

so that optimization evolves only along directions permitted by the current determining, solution-independence, and novelty restrictions.

The regularized nontriviality objective $\mathcal { L } _ { \mathrm { n t r } } ^ { \mathrm { r e g } }$ in (162) is then optimized until the candidate $v _ { \Theta } ,$ satisfies the principal acceptance criteria

$$
\begin{array} { r l r } & { } & { \mathrm { R M S } _ { E } \left( v _ { \Theta _ { i } } \right) > \tau _ { \mathrm { v i s } } , } \\ & { } & { \rho _ { 9 9 } \left( v _ { \Theta _ { i } } \right) < \tau _ { \mathrm { s p d } } , } \\ & { } & { \mathcal { L } _ { \mathrm { s y m } } ^ { \mathrm { r a w } } < \tau _ { \mathrm { s y m } } , } \\ & { } & { \mathcal { R } _ { \mathrm { t a i l } } ^ { \rho } \left( v _ { \Theta _ { i } } \right) > \tau _ { \mathrm { n t r } } , } \end{array}\tag{172}
$$

on the corresponding validation samples. Here, the ambientresponse visibility criterion is evaluated on an of-shell validation set, whereas the speed-tail, determining-residual, and prolonged-characteristic criteria are evaluated on onshell validation samples. The first two conditions ensure that the candidate remains both numerically visible and suficiently well resolved across the sampled region; the latter two verify approximate symmetry admissibility and nontriviality independently of the training samples. Once all conditions in (172) are satisfied, v<sub>Θ</sub> is accepted as the $i ^ { \mathrm { t h } }$ recovered generator and incorporated into the construction of $\widehat { N } _ { \mathrm { n o v } , i + 1 } ^ { ( k ) }$ for the subsequent discovery step.

Finally, we specify the stopping criteria for the sequential discovery procedure. Two qualitatively distinct termination modes are distinguished: (i) clean exhaustion of the represented novel directions and (ii) recovery failure while a represented novel direction remains. This distinction is essential because only the first mode supports the conclusion that the available symmetry directions have been exhausted within the chosen stabilized representation.

The clean-exhaustion criterion is met immediately when dim $\left( \widehat { N } _ { \mathrm { n o v } , i } ^ { ( k ) } \right) = 0$ . In this case, all represented admissible directions have already been accounted for by the previously accepted generators, up to the numerical rank tolerances entering the novelty construction, and no further initialization is attempted. Clean exhaustion may also be declared when dim $\left( \bar { \widehat { N } } _ { \mathrm { n o v } , i } ^ { ( k ) } \right) > 0$ in coeficient space, but repeated sampling within this subspace, over a prescribed initialization budget, fails to produce any candidate satisfying $\mathrm { R M S } _ { E } \left( v _ { \Theta _ { i } } \right) > \tau _ { \mathrm { v i s } }$ . Such a situation indicates that the remaining coeficient directions are numerically negligible under the realized ambient-response map, for example because of residual dictionary nonidentifiability or samplinginduced rank deficiency. The sequential procedure is then terminated and the already accepted generators are taken to exhaust the numerically resolvable symmetry directions available within the chosen stabilized representation.

Failure of the second initialization condition in (169) has a diferent interpretation. If candidates with $\mathrm { R M S } _ { E } \left( v _ { \Theta _ { i } } \right) >$ $\tau _ { \mathrm { v i s } }$ can be produced, but repeated initialization yields only fields satisfying ρ<sub>99</sub> $( v _ { \Theta _ { i } } ) \ge \tau _ { \mathrm { s p d } }$ , then represented novelty is still present, but no initialization with the required numerical regularity has been found. This case is therefore classified as a recovery failure rather than as exhaustion of the represented space.

The same recovery-failure status is assigned when an initialization satisfying both conditions in (169) is admitted to optimization, but no trained candidate satisfies all acceptance criteria in (172) within the prescribed optimization budget. Such failure may arise, for example, because the determining residual remains above $\tau _ { \mathrm { s y m } }$ , the prolonged-characteristic response remains below the required nontriviality threshold, or the speed-tail criterion becomes unacceptable during optimization. In these cases, the existence of a visible direction in $\widehat { N } _ { \mathrm { n o v } , i } ^ { ( k ) }$ prevents the algorithm from interpreting unsuccessful recovery as structural exhaustion.

Accordingly, only clean exhaustion permits the sequential algorithm to conclude that the represented intrinsic symmetry directions have been exhausted at the selected dictionary level and numerical tolerances. Termination by recovery failure makes no such statement: the generators accepted before the failed step remain valid recovered directions, but no completeness claim is made for the remaining represented symmetry space.

## 5 NUMERICAL EXPERIMENTS

This section evaluates the proposed framework on the two-dimensional spatial wave equation, using the final manuscript-aligned implementation described in Section 4. The numerical study is designed to examine not only the sequential recovery of the represented intrinsic symmetry directions, but also the behavior of the progressively restricted search space, the algebraic consistency of the recovered generators, and the locality of the fixed dictionary representation. We first introduce the benchmark and experimental configuration, then present the sequential discovery results, and finally examine the reconstruction and extrapolation behavior of a known analytic symmetry generator.

## 5.1 BENCHMARK AND EXPERIMENTAL CONFIG-URATION

This subsection introduces the DE system used as the principal numerical benchmark and explains why it provides a suficiently rich test of the structural components of the proposed framework while retaining a well-understood analytic symmetry structure for validation. We then summarize the common numerical configuration used throughout the experiments.

We consider the wave equation in two spatial dimensions [41, Chapter 10],

$$
\Delta ^ { ( 2 ) } \left( x , y , t ; u _ { ( 2 ) } \right) = u _ { t t } - c ^ { 2 } \left( u _ { x x } + u _ { y y } \right) = 0 ,\tag{173}
$$

with the normalized wave speed $c = 1$ in all experiments. Equation (173) is a regular linear PDE of genuine order $s = 2$ and satisfies the hypotheses of Proposition 6. Hence, every admitted Lie point-symmetry generator has the PFA form in (187). Moreover, under the dictionary-refinement sequence specified below, the represented geometric rank $d _ { o _ { i } }$ defined in (152) stabilizes, so the benchmark passes the RSP criterion described in Rank-Stabilization Preflight (RSP) Test. It therefore lies within the admissible scope summarized in Applicability and Admissible Generator Realizations.

The Lie point-symmetry structure of (173) is also known analytically [5, Example 2.43]. After separating the additive solution sector associated with linear superposition, its finite-dimensional intrinsic symmetry algebra ${ \mathfrak { g } } _ { \mathrm { i n t } }$ has dimension 11. At the same time, linearity introduces the nontrivial additive solution sector $\boldsymbol { A } _ { \mathrm { s o l } }$ described in Intrinsic Admissibility Constraints, which must be removed from the represented determining nullspace before the RSP and intrinsic sequential discovery are applied. The benchmark therefore exercises, within a single DE system, the sampled determining restriction, additive-sector exclusion, represented-rank stabilization, nontriviality criterion, global novelty restriction, and graceful stopping mechanism. Its known finite-dimensional symmetry structure additionally provides an independent reference for validating the recovered dimension, Lie-algebraic relations, and individual reconstructed symmetry directions.

For all experiments, we generated samples directly on the equation shell of (173), rather than from a particular solution graph. The independent coordinates x were sampled on the uniform patch

$$
\Omega _ { \mathbf { X } } = [ 0 . 3 0 , 0 . 3 6 ] \times [ 0 . 3 0 , 0 . 3 6 ] \times [ 0 . 2 5 , 0 . 2 5 6 ] .\tag{174}
$$

using $1 0 1 \times 1 0 1 \times 1 1$ points in $( x , y , t )$ , corresponding to a common spacing of $6 \times 1 0 ^ { - 4 }$ and 112,211 jet-space samples. At each base point, the remaining jet-fiber coordinates were drawn independently from a standard normal distribution and subsequently projected onto the equation shell. For (173) with $c = 1$ , this projection is obtained directly by imposing $u _ { t t } = u _ { x x } + u _ { y y } .$

Each on-shell sample was paired with an of-shell sample obtained by perturbing the corresponding jet-space point along a randomly sampled unit direction with perturbation magnitude $1 0 ^ { - 2 }$ . The resulting paired dataset was randomly divided into 80% training and 20% validation subsets, yielding 89,769 and 22,442 samples, respectively. The on-shell $\mathbf { S } ^ { \mathrm { o n } }$ and paired of-shell $\mathbf { S } ^ { \mathrm { o f f } }$ components were subsequently supplied to the corresponding sampled operators and constraints described in Sections 3 and 4, without restricting the samples to a single solution manifold.

For the fixed generator realization, we used the dictionary $D _ { ( 2 ) }$ introduced in Fixed Dictionary and Generator Realization, with the independent coordinates ${ \boldsymbol { x } } = ( x , y , t )$ as inputs, maximum interaction order $k = 2 ,$ cubic B-splines with degree $r = 3 ,$ , and grid resolution $o = 8$ . The fixed core-grid intervals were $\left[ - 0 . 6 6 , 1 . 3 4 \right]$ for both x and $y$ and $[ - 1 . 0 5 , 0 . 9 5 ]$ for t. Thus, $K _ { o } = o + r = 1 1$ univariate Bspline functions were associated with each input coordinate, and the univariate and pairwise-interaction blocks together yielded $n _ { ( 2 ) } ^ { D } = 3 9 6$ scalar dictionary features. Under the PFA realization appropriate to (173), the three horizontal coeficients together with the multiplicative and additive vertical coeficients therefore resulted in a total coeficient dimension of $P = 1 9 8 0$

Before sequential discovery, we applied the RSP test of

Rank-Stabilization Preflight (RSP) Test over the nested gridrefinement sequence $o _ { 1 } = 4 < o _ { 2 } = 8 < o _ { 3 } = 1 6$ , while keeping the spline degree, interaction order, and physical core-grid intervals fixed. Numerical ranks were evaluated using a relative SVD threshold of $1 0 ^ { - 1 0 }$ ; a tolerance sweep showed that this tighter threshold was required to resolve the determining-operator rank at the coarsest refinement level, while the resulting rank was unchanged upon further reduction to $1 0 ^ { - 1 2 }$ . The represented geometric rank remained constant across all three refinements,

$$
d _ { 4 } = d _ { 8 } = d _ { 1 6 } = 1 1 ,\tag{175}
$$

thereby satisfying the stabilization criterion used to proceed with the nominal $o = 8$ realization in the subsequent experiments.

## 5.2 SEQUENTIAL DISCOVERY OF INTRINSIC SYMMETRIES

Having established stabilization of the represented geometric rank, we fix the nominal resolution $o = 8$ and examine the sequential recovery of the corresponding intrinsic symmetry directions. Before any generator optimization is performed, the sampled determining restriction $\widehat { M } _ { \Delta } ^ { ( 2 ) }$ has rank 231 in the $P = 1 9 8 0$ -dimensional coeficient representation, leaving a determining nullspace $\widehat { N } _ { \Delta } ^ { ( 2 ) }$ of dimension 1749. The large coeficient-space nullity reflects the substantial redundancy of the spline representation discussed in B and should therefore not be interpreted directly as the number of geometrically distinct symmetries. After removal of the represented additive solution sector, the resulting additive-free coeficient representation has dimension 11, in agreement with both the RSP result of the preceding subsection and the known intrinsic symmetry dimension of (173).

This reduction can also be verified at the response level. The sampled characteristic responses of the full determining nullspace, the represented additive sector, and the resulting intrinsic sector have numerical ranks 29, 18, and 11, respectively. In particular, the intrinsic response rank remains equal to 11 over relative SVD thresholds from $1 0 ^ { - 4 }$ to $1 0 ^ { - 1 2 }$ , indicating that the finite-dimensional target supplied to the sequential discovery stage is numerically well separated from the redundant coeficient directions. We therefore begin the first optimization step with an 11-dimensional novelty-restricted admissible space and remove one globally novel direction after each accepted generator.

As shown in Table 1, all eleven accepted generators satisfy the sampled determining condition to near machine precision on the independent validation set. The validation symmetry RMS remains between approximately $2 . 8 \times 1 0 ^ { - 1 5 }$ and $1 . 1 \times 1 0 ^ { - 1 3 }$ throughout the complete discovery sequence, several orders of magnitude below the prescribed acceptance threshold $\tau _ { \mathrm { s y m } } = 2 \times 1 0 ^ { - 8 }$ . At the same time, each accepted generator removes exactly one globally novel represented direction: dim $( T _ { i } )$ decreases successively from 11 to 1 as $v _ { 1 } , \ldots , v _ { 1 1 }$ are recovered. Thus, the progressive novelty restriction remains consistent with the represented geometric rank $d _ { 8 } = 1 1$ identified before optimization, rather than producing an early loss or duplication of an admissible direction.

Table 1: Sequential recovery of the represented intrinsic symmetry directions for the two-dimensional wave equation at the nominal grid resolution $o = 8 .$ . Here, dim $( T _ { i } )$ denotes the dimension of the novelty-restricted coeficient space immediately before optimization of the $i ^ { \mathrm { t h } }$ generator. The validation symmetry RMS and speed-tail ratio are evaluated for the accepted generator on the held-out validation set. Each accepted generator removes one globally novel direction; after the eleventh generator, the novelty-restricted space collapses to zero and the procedure terminates by clean represented exhaustion.
<table><tr><td>Generator</td><td>dim  $( T _ { i } )$ </td><td>Validation symmetry RMS</td><td>Speed-tail ratio</td></tr><tr><td> $v _ { 1 }$ </td><td>11</td><td> $3 . 3 3 \times 1 0 ^ { - 1 5 }$ </td><td>1.037</td></tr><tr><td> $v _ { 2 }$ </td><td>10</td><td> $5 . 6 8 \times 1 0 ^ { - 1 5 }$ </td><td>1.011</td></tr><tr><td>v3</td><td>9</td><td> $2 . 7 7 \times 1 0 ^ { - 1 5 }$ </td><td>1.003</td></tr><tr><td> $v _ { 4 }$ </td><td>8</td><td> $5 . 2 4 \times 1 0 ^ { - 1 5 }$ </td><td>1.002</td></tr><tr><td> $v _ { 5 }$ </td><td>7</td><td> $4 . 9 2 \times 1 0 ^ { - 1 4 }$ </td><td>1.648</td></tr><tr><td> $v _ { 6 }$ </td><td>6</td><td> $7 . 8 6 \times 1 0 ^ { - 1 4 }$ </td><td>1.661</td></tr><tr><td> $v _ { 7 }$ </td><td>5</td><td> $8 . 7 6 \times 1 0 ^ { - 1 4 }$ </td><td>1.654</td></tr><tr><td> $v _ { 8 }$ </td><td>4</td><td> $7 . 3 2 \times 1 0 ^ { - 1 4 }$ </td><td>1.658</td></tr><tr><td> $v _ { 9 }$ </td><td>3</td><td> $1 . 1 0 \times 1 0 ^ { - 1 3 }$ </td><td>1.651</td></tr><tr><td> $v _ { 1 0 }$ </td><td>2</td><td> $5 . 8 5 \times 1 0 ^ { - 1 4 }$ </td><td>1.993</td></tr><tr><td> $v _ { 1 1 }$ </td><td>1</td><td> $1 . 0 6 \times 1 0 ^ { - 1 3 }$ </td><td>1.969</td></tr></table>

The speed-tail ratio $\rho _ { 9 9 }$ introduced in Sequential Recovery and Stopping Criteria provides an additional view of how the recovered directions change as this novelty space is progressively exhausted. For the first four generators, ρ<sub>99</sub> remains close to unity, ranging from approximately 1.002 to 1.037, indicating an almost uniform distribution of pointwise generator speeds over the sampled domain. This transition has a natural interpretation from the known analytical symmetry structure of (173): the intrinsic algebra contains the particularly simple four-dimensional subspace

$$
\mathrm { s p a n } \left\{ \partial _ { x } , \partial _ { y } , \partial _ { t } , u \partial _ { u } \right\} .\tag{176}
$$

Once four independent directions have been recovered, the speed-tail ratio increases abruptly to approximately 1.65 for $v _ { 5 } , \ldots , v _ { 9 }$ , and further to approximately 1.97–1.99 for $v _ { 1 0 }$ and $v _ { 1 1 }$ . Since the sequential procedure does not constrain the recovered numerical basis to coincide with any particular analytical basis, we do not identify $v _ { 1 } , \ldots , v _ { 4 }$ individually with the four canonical generators above. Nevertheless, the coincidence between the dimension of this simple analytical sector and the observed transition after the fourth recovered direction is consistent with the procedure first resolving comparatively simple symmetry combinations before moving into the more spatially structured directions required to complete the represented algebra.

Nontriviality remains satisfied throughout the same sequence. The validation lower-tail nontriviality statistic exceeds the prescribed threshold $\tau _ { \mathrm { n t r } } = 2 . 0 \times 1 0 ^ { - 3 }$ for every accepted generator, with values ranging from $2 . 0 0 2 \times 1 0 ^ { - 3 }$ to $3 . 3 7 9 \times$ $1 0 ^ { - 3 }$ . The earlier directions generally lie more comfortably above this threshold, whereas several of the later generators approach it more closely; for example, the seventh generator yields approximately $2 . 0 0 2 \times 1 0 ^ { - 3 }$ . We do not interpret these values as defining a canonical ordering of symmetry complexity, since both the recovered basis and its sequential ordering are nonunique. Rather, together with the increase in $\rho _ { 9 9 }$ , they indicate that the remaining admissible directions become progressively more constrained by the simultaneous requirements of global novelty, sampled symmetry, and nontriviality.

After acceptance of $v _ { 1 1 }$ , the novelty-restricted represented space collapses to zero before initialization of a twelfth candidate. The run therefore terminates by clean represented exhaustion, rather than by a failure of initialization or optimization. In conjunction with the stabilized RSP value $d _ { 4 } = d _ { 8 } = d _ { 1 6 } = 1 1$ , this provides numerical evidence that all resolvable intrinsic directions within the chosen stabilized representation and sampling configuration have been recovered. Representative integral curves of $v _ { 4 } , v _ { 7 } .$ , and $v _ { 1 0 } ,$ selected from the three speed-tail regimes identified above, are shown in Fig. F.1 of F. These visualizations complement the speed-tail diagnostic by illustrating the increasingly nonuniform local flow geometry of the recovered directions, but they are not used in the numerical acceptance or stopping criteria. The remaining question is then algebraic rather than dimensional: whether the span of the eleven recovered generators is numerically closed under Lie brackets. We address this next through post-recovery bracket validation.

To assess whether the recovered span is also consistent with the Lie-algebraic structure of the intrinsic symmetry space, we performed a post-recovery bracket-closure test on ofshell samples. For each unordered pair $( v _ { i } , v _ { j } )$ , with $i < j .$ a single vector of constant coeficients $c _ { i j } ^ { k } \in \mathbb { R }$ was fitted over the of-shell fitting set according to

Table 2: Post-recovery Lie-bracket closure of the recovered intrinsic symmetry generators for the two-dimensional wave equation.
<table><tr><td>Quantity</td><td>Value</td></tr><tr><td>Recovered generators</td><td>11</td></tr><tr><td>Unordered generator pairs</td><td>55</td></tr><tr><td>Fitting samples</td><td>89769</td></tr><tr><td>Validation samples</td><td>22 442</td></tr><tr><td>Response-design rank</td><td>11</td></tr><tr><td>Response-design condition number</td><td> $1 . 5 5 \times 1 0 ^ { 1 }$ </td></tr><tr><td>Median held-out relative closure error</td><td> $1 . 6 4 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Maximum held-out relative closure error</td><td> $2 . 6 9 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>Numerically zero brackets</td><td>0</td></tr></table>

$$
[ v _ { i } , v _ { j } ] \approx \sum _ { k = 1 } ^ { 1 1 } c _ { i j } ^ { k } v _ { k } .\tag{177}
$$

The fitted coeficients were then held fixed and evaluated on the disjoint of-shell validation set. Thus, the test does not permit point-dependent coeficients to reproduce the bracket locally; rather, it directly tests whether each recovered bracket can be represented by one global element of the finite-dimensional recovered span. For each nonzero bracket pair, the held-out relative closure error is defined by

$$
\varepsilon _ { i j } ^ { \mathrm { b r } } : = \frac { \mathrm { R M S } \left( \left[ v _ { i } , v _ { j } \right] - \sum _ { k = 1 } ^ { 1 1 } c _ { i j } ^ { k } v _ { k } \right) } { \mathrm { R M S } \left( \left[ v _ { i } , v _ { j } \right] \right) } .\tag{178}
$$

The first numerical requirement for interpreting such a fit is that the sampled responses of the recovered generators remain linearly identifiable. To this end, we form a responsedesign matrix whose $i ^ { \mathrm { t h } }$ column contains the sampled components of the recovered generator $v _ { i }$ over the fitting anchors. As reported in Table 2, this matrix has numerical rank 11, equal to the number of recovered generators, with condition number approximately $1 . 5 5 \times 1 0 ^ { 1 }$ . Hence, the fitted structure coeficients are not obtained from a rank-deficient representation of the recovered basis. This is important because an underdetermined response design could otherwise produce nonunique coeficient vectors even when a small bracket residual is observed.

All 55 unordered generator pairs were classified as numerically nonzero. A bracket was regarded as numerically zero when its RMS did not exceed $1 \bar { 0 } ^ { - 1 0 }$ times the overall recovered-generator RMS, which for this experiment gives a zero-bracket threshold of approximately $1 . 0 0 \times 1 0 ^ { - 1 1 }$ The smallest observed bracket RMS was approximately $5 . 6 2 \times 1 0 ^ { - 3 }$ and therefore remained well above this threshold. The absence of numerically zero pairwise brackets in the recovered basis should not be interpreted as a disagreement with an analytical symmetry basis containing commuting generators. The pattern of vanishing brackets, as well as the individual structure constants, is basis dependent, and the sequential discovery procedure does not constrain the recovered generators to coincide with a canonical analytical basis. Generic linear combinations of such generators may therefore yield nonzero brackets for every recovered basis pair.

The held-out closure accuracy is summarized by the relative validation residuals. Across all 55 brackets, the median relative error is $1 . 6 4 \times 1 0 ^ { - 1 0 }$ , while the largest value is $2 . 6 9 \times 1 0 ^ { - 8 }$ , attained for $[ v _ { 7 } , v _ { 9 } ]$ . Importantly, these errors are evaluated using the same constant structure coeficients determined solely from the fitting samples; no refitting is performed on the validation set. The small held-out residuals therefore indicate that the fitted bracket relations persist across disjoint of-shell samples rather than merely reproducing the points used to determine the coeficients.

Taken together with the full recovered response rank and the preceding clean represented exhaustion, these results provide numerical evidence that the recovered 11-dimensional span is closed under Lie brackets within the sampled domain and to the numerical accuracy supported by the chosen finite representation. Accordingly, the recovered generators behave numerically as a finite-dimensional Lie algebra, rather than merely as eleven independently admissible symmetry directions. This conclusion is a sampled numerical closure statement and is not intended as an exact symbolic verification of the structure constants.

## 5.3 APPROXIMATION OF A KNOWN SYMMETRY GENERATOR AND LOCALITY OF THE REP-RESENTATION

The preceding experiments establish that the sequential procedure recovers the full represented intrinsic dimension and that the resulting span is numerically closed under Lie brackets. We finally examine two complementary properties that are not determined by these tests alone: whether the recovered numerical basis can accurately reconstruct a specific known analytical symmetry direction, and how far such a reconstruction can be continued away from the region in which the fixed dictionary was sampled and fitted. This provides both an independent check on the content of the recovered span and a direct illustration of the local character of the finite dictionary representation.

As a reference direction, we select the analytical inversionin-x symmetry of (173) [5, (2.65)],

$$
v _ { \mathrm { i n v } , x } = \left( x ^ { 2 } - y ^ { 2 } + t ^ { 2 } \right) \partial _ { x } + 2 x y \partial _ { y } + 2 x t \partial _ { t } - x u \partial _ { u } .\tag{179}
$$

Because the sequential discovery procedure does not constrain the recovered generators to coincide with a prescribed analytical basis, the relevant question is not whether one recovered generator reproduces (179) individually, but whether this direction lies in their global constant-coeficient span. We therefore construct

$$
\widetilde { v } _ { \mathrm { i n v } , x } : = \sum _ { i = 1 } ^ { 1 1 } a _ { i } v _ { i } , \quad a _ { i } \in \mathbb { R } .\tag{180}
$$

and determine the coeficients $a _ { i }$ by a global least-squares fit while keeping all recovered generators fixed. The fit uses both the original on-shell samples and their paired ofshell perturbations, so that the resulting coeficient vector represents a single global combination of the recovered basis rather than a point-dependent local reconstruction.

We first quantify the reconstruction accuracy on the fitting and held-out validation data and examine the pointwise agreement of $\boldsymbol { v } _ { \mathrm { i n v } , x }$ and $\widetilde { v } _ { \mathrm { i n v } , x }$ in the vicinity of the sampled domain. We then compare their finite integral curves as the flow is continued beyond this region. The latter comparison is intended to probe the locality of the fixed spline representation rather than the membership of the analytical generator in the recovered span, and it provides the numerical motivation for the patchwise transport strategy discussed below.

Table 3 summarizes the resulting constant-span reconstruction. The fitting problem is strongly overdetermined, with $1 . 3 0 \times 1 0 ^ { 6 }$ sampled scalar equations for the 11 coeficients in (180). The corresponding response matrix has full column rank 11 at the relative SVD threshold $1 0 ^ { - 1 0 }$ and condition number approximately $3 . 0 5 \times 1 0 ^ { 3 }$ ; no ridge regularization is applied. The relative RMS reported in the table is the absolute RMS reconstruction residual normalized by the RMS magnitude of the analytical target, while the cosine similarity measures the directional alignment of the stacked analytical and reconstructed vector-field responses.

The reconstruction error is essentially unchanged when the fitted coeficients are transferred to the held-out samples. On the fitting data, the relative and absolute RMS errors are $4 . 1 2 \times \bar { 1 0 } ^ { - 1 0 }$ and $9 . 8 2 \times 1 0 ^ { - 1 0 }$ , respectively, while the corresponding validation values are $4 . 0 9 \times 1 0 ^ { - 1 0 }$ and $9 . 8 9 \times 1 0 ^ { - 1 0 }$ . The cosine similarity is 1.000 to the reported precision for both sets. Importantly, the coeficients $a _ { i }$ are determined only from the fitting data and remain fixed during validation. Thus, a single global linear combination of the recovered generators reproduces the known inversion-inx symmetry to approximately $1 0 ^ { - 1 0 }$ relative accuracy on independent samples in the neighborhood represented by the fixed dictionary.

This agreement is also evident at the level of the induced finite flows. Fig. F.2 in F compares the integral curves generated by $v _ { \mathrm { i n v } , x }$ and $\widetilde { v } _ { \mathrm { i n v } , x }$ . The initial-point grid spans $x , y \in [ 0 . 2 , 0 . 4 ] , t \in [ 0 . 2 , 0 . 3 ]$ , and $u \in [ - 0 . 6 , 0 . 6 ]$ . In particular, the independent-coordinate ranges already extend substantially beyond the base-coordinate intervals used to generate the fitting and validation data in (174). The trajectories are integrated up to $\varepsilon _ { \mathrm { m a x } } = 0 . 1$ using 100 steps. Across all six displayed coordinate-plane projections, the reconstructed and analytical trajectories are visually indistinguishable at the plotted resolution, consistent with the pointwise reconstruction errors reported in Table 3.

This provides an independent numerical check that the recovered span contains an accurate representation of this analytical intrinsic symmetry direction, while leaving open the separate question of how that representation behaves when the flow carries points beyond the region in which the fixed dictionary is reliably constrained.

To examine the extent and eventual limitations of this extrapolation, we next repeat the finite-flow comparison after shifting only the initialization interval associated with the x coordinate. Relative to the preceding visualization, the $y ,$ t, and u ranges are left unchanged, while the initial x values are moved to [0.8, 0.86]. Fig. F.3 in F compares the analytical and reconstructed flows over the shorter interval $\varepsilon _ { \operatorname* { m a x } } = 0 . 0 4$ . Despite this reduced integration horizon, the departure between the two representations is now readily visible. In the $x - y$ and x–t projections, both the directions and the extents of the reconstructed trajectories deviate from their analytical counterparts, whereas the remaining projections primarily exhibit a mismatch in trajectory scale. The contrast with Fig. F.2 therefore shows that, although the recovered representation remains remarkably accurate well beyond the sampled x interval, this extrapolation should not be interpreted as a uniformly accurate global representation of the analytical vector field.

This behavior motivates a patchwise treatment of the recovered generators on larger domains. Rather than requiring a single fixed dictionary realization to extrapolate accurately throughout the entire domain, one may construct local representations on overlapping coordinate patches $\Omega _ { X } ^ { ( a ) }$ and transport the already recovered generator directions across their overlaps. In such a continuation, the novelty, linearindependence, and nontriviality conditions used to identify the generators need not be reimposed independently on each patch; their role is to determine the symmetry directions during the initial recovery. Subsequent patches instead provide local representations of those same directions, with overlap consistency supplying the information required to relate neighboring realizations. The present extrapolation experiment thus provides a direct numerical motivation for the patchwise generator-transport extension identified as future work in Section 7.

Table 3: Global constant-span reconstruction of the analytical inversion-in-x generator from the eleven recovered generators.
<table><tr><td></td><td>Relative RMS</td><td>Absolute RMS</td><td>Cosine similarity</td></tr><tr><td>Fitting</td><td> $4 . 1 2 \times 1 0 ^ { - 1 0 }$ </td><td> $9 . 8 2 \times 1 0 ^ { - 1 0 }$ </td><td>1.000</td></tr><tr><td>Validation</td><td> $4 . 0 9 \times 1 0 ^ { - 1 0 }$ </td><td> $9 . 8 9 \times 1 0 ^ { - 1 0 }$ </td><td>1.000</td></tr></table>

## 6 DISCUSSION, APPLICABILITY, AND LIMITATIONS

Sections 3 and 4 establish the theoretical construction and computational machinery used to recover represented intrinsic symmetry generators. Let $\mathfrak { g } _ { \mathrm { r e c } } : = \operatorname { s p a n } \left\{ v _ { 1 } , \ldots , v _ { r } \right\}$ denote the span of the generators accepted by the proposed framework. Because the determining, nontriviality, solutionindependence, and novelty conditions are enforced through finite-dimensional representations and sampled operators, ${ \mathfrak { g } } _ { \mathrm { r e c } }$ should be interpreted as a numerical approximation to a subspace of the target intrinsic algebra ${ \mathfrak { g } } _ { \mathrm { i n t } }$ , rather than as an exact algebraic identification. The preceding sections, together with B, already describe the principal representational and sampling mechanisms through which the equality ${ \mathfrak { g } } _ { \mathrm { r e c } } = { \mathfrak { g } } _ { \mathrm { i n t } }$ may fail. They do not, however, fully characterize the scope of the resulting completeness claim, the limitations of the RSP, or the distinction between sampled represented completeness and exact geometric completeness.

A separate issue concerns the class of DE systems to which the framework may be applied. The sequential-recovery flowchart in Fig. E.2 begins by asking whether the given system $\Delta ^ { ( s ) } = \mathbf { 0 }$ is admissible and halts before discovery when it lies outside the intended scope. The mathematical criteria underlying that decision, including the admissible generator realization and the assumptions under which it is justified, have so far been deferred.

Accordingly, this section clarifies (i) the admissible classes of DE systems, together with the structural assumptions and model restrictions required for each class; (ii) the conditions under which recovery may be regarded as geometrically complete within the represented and sampled setting, together with the limitations of such a claim; and (iii) the geometric symmetry structure described by ${ \mathfrak { g } } _ { \mathrm { r e c } } .$ , including the distinction between infinitesimal, identity-connected, and genuinely global symmetry information. Together, these questions delineate the applicability, the scope of completeness claims, and the geometric scope of the proposed framework.

## 6.1 APPLICABILITY AND ADMISSIBLE GENERA-TOR REALIZATIONS

In this subsection, we do not seek a canonical classification of DEs, but rather several classifications for understanding the scope of our current and future framework. We distinguish an ODE from a PDE, linear DE from a nonlinear DE, and finally, the order also matters in determining the scope of this framework. We also discuss when the projectable fiber-afine (PFA) restriction, that is briefly touched in Proposition 2, is mathematically legitimate and when the RSP is required. Finally, we conclude the completeness claim in accordance with all the evidence provided in this subsection.

Unlike linear DEs, nonlinear DEs do not possess the generic additive solution-symmetry sector discussed in Intrinsic Admissibility Constraints. In particular, the superposition principle no longer applies: linear combinations of solutions of a nonlinear DE system need not be solutions, and consequently there is no generic linear space analogous to $\boldsymbol { A } _ { \mathrm { s o l } }$ that can be projected out before sequential discovery. This does not preclude individual nonlinear DEs from admitting solution-dependent point symmetries; rather, such symmetries must be identified through the solution-independence criterion introduced in Intrinsic Admissibility Constraints, instead of through a universal linear nullspace projection. Moreover, as discussed in A, the corresponding determining system may retain other forms of arbitrary functional freedom.

For determined ODE systems of genuine order $s \ > \ 1$ however, the unrestricted Lie point-symmetry algebra is finite-dimensional; regardless of whether the underlying ODE system is linear or nonlinear.

Proposition 4 (Finite Dimensionality of Determined ODE Symmetries). Let $\Delta ^ { ( s ) } \ = \ \mathbf { 0 }$ be a regular, determined system of ODEs of genuine order $s \geq 2 ,$ , locally solvable [5, Section 2.6] for its highest-order derivatives as $\mathbf { u } _ { s } = \mathbf { F } \left( x , \mathbf { u } _ { ( s - 1 ) } \right)$ , where u $\mathbf { \Sigma } \in \mathbb { R } ^ { q }$ and $q \geq 1$ . Then its local Lie point-symmetry algebra ${ \mathfrak { g } } _ { \mathrm { p t } }$ isfinite-dimensional, with

$$
\mathrm { d i m } \left( { \mathfrak { g } } _ { \mathrm { p t } } \right) \leq \left\{ { \begin{array} { l l } { q ^ { 2 } + 4 q + 3 , } & { s = 2 , } \\ { q ^ { 2 } + s q + 3 , } & { s \geq 3 . } \end{array} } \right.\tag{181}
$$

Remark 8 (Determined System). A system is called $^ {  } d e \mathrm { . }$ termined” in here when the number of equations equals the number of dependent variables and the highest-order derivative Jacobian has full rank, so that the system is locally solvable for all highest-order derivatives. See also the maximal-rank assumption in Jet-Space Geometry and Diferential Equations.

The bounds in Proposition 4 are established for general determined ODE systems of the stated form and therefore apply, in particular, to the ODE systems considered here [42]. Finite dimensionality of the exact point-symmetry algebra, however, does not by itself imply that a chosen finite dictionary contains a suficiently rich realization of all of its generators. We therefore retain the RSP for regular, determined ODE systems with $s > 1$ . Stabilization of the represented geometric rank provides operational evidence that further enrichment along the chosen nested representation does not reveal additional resolvable intrinsic directions, whereas failure to stabilize prevents a completeness claim within the present framework.

As stated previously, there is no generic additive solution sector that can be removed from the determining nullspace ker $\widehat { \mathcal { D } } _ { \Delta }$ before the discovery begins. Candidate generators must instead satisfy the solution-independence requirement individually upon discovery. If a candidate is identified as a solution-dependent vertical generator, it is excluded from ${ \mathfrak { g } } _ { \mathrm { r e c } }$ as an approximation to ${ \mathfrak { g } } _ { \mathrm { i n t } } .$ , but its realized direction is retained in the novelty restriction to prevent subsequent rediscovery of the same generator.

The preceding conclusion changes fundamentally at $s =$ 1. Regular first-order ODEs, irrespective of linearity or nonlinearity, possess an unrestricted local point-symmetry structure with arbitrary functional freedom.

Proposition 5 (Residual Functional Freedom for First-Order ODEs). Let a regular determinedfirst-order ODE system with $q \geq 1$ dependent variables be locally written as

$$
\frac { d \mathbf { u } } { d x } = \mathbf { F } \left( x , \mathbf { u } \right) .\tag{182}
$$

Then, in a neighborhood ofevery regular point, there exist local point coordinates $( s , \mathbf { r } )$ in which the system takes the rectified form

$$
\frac { d \mathbf { r } } { d s } = \mathbf { 0 } .\tag{183}
$$

In these coordinates, every local Lie point-symmetry generator has theform

$$
v = A \left( s , \mathbf { r } \right) \partial _ { s } + \sum _ { \alpha = 1 } ^ { q } B ^ { \alpha } \left( \mathbf { r } \right) \partial _ { r ^ { \alpha } } ,\tag{184}
$$

where A is arbitrary and thefunctions $B ^ { \alpha } \left( \mathbf { r } \right)$ are arbitrary smoothfunctions ofthe local solution-leafcoordinates.

Moreover, the characteristic satisfies

$$
Q _ { v } ^ { \alpha } | _ { \Sigma ^ { ( 1 ) } } = B ^ { \alpha } \left( \mathbf { r } \right) .
$$

Hence, the arbitrary sector $A \left( s , \mathbf { r } \right) \partial _ { s }$ is precisely tangent to the solutionfoliation and is removed by the characteristicbased nontriviality criterion $\mathcal { L } _ { \mathrm { t a i l } }$ in (158), whereas the arbitraryfunctionalfamily

(185)

$$
\sum _ { \alpha = 1 } ^ { q } B ^ { \alpha } \left( \mathbf { r } \right) \partial _ { r ^ { \alpha } }\tag{186}
$$

remains nontrivial. Consequently, quotienting by the characteristic-zero tangent sector does not render the unrestricted local point-symmetry structurefinite-dimensional.

The proof of Proposition 5 is given in D.4. The proposition clarifies that first-order ODEs are not excluded because the characteristic-based nontriviality test becomes inefective at $s = 1$ . Rather, the characteristic correctly removes the arbitrary solution-tangent sector, while an independent arbitrary functional freedom remains in transformations of the local solution-leaf coordinates. Consequently, quotienting by the sampled-trivial sector does not reduce the unrestricted exact local point-symmetry structure to a finite-dimensional algebra. We therefore exclude all first-order ODEs outright from the scope of the current framework.

This result also identifies a fundamental limitation of the RSP for first-order ODEs. As the represented function space is enriched, additional directions associated with the arbitrary functions $B ^ { \alpha } \left( \mathbf { r } \right)$ may become resolvable, so failure of the represented geometric rank in (152) to stabilize appropriately prevents a completeness claim. Conversely, an apparent finite plateau cannot override the exact local structural result of Proposition 5: it signifies only stabilization within the chosen finite dictionary, sampled realization, and numerical rank tolerance, rather than exhaustion of the unrestricted pointsymmetry structure.

The situation is less definitive for nonlinear PDEs. Unlike regular, determined ODE systems with $s > 1$ , there is no corresponding general finite-dimensionality result for their unrestricted local Lie point-symmetry algebras. Depending on the structure of the governing equations, the determining system may close to a finite-dimensional algebra or may retain arbitrary functional degrees of freedom, as discussed from the formal-integrability viewpoint in A. In particular, nonlinear PDEs may admit infinite-dimensional Lie pointsymmetry algebras parameterized by arbitrary functions; the Liouville equation provides a classical example [43, 44]. Consequently, finite dimensionality of the target algebra cannot be inferred solely from nonlinearity, diferential order, or regularity.

For linear systems of suficiently high diferential order, the determining equations impose a stronger admissible restriction: admitted point-symmetry generators acquire a PFA form. We examine this structural simplification next.

Proposition 6 (Projectable Fiber-Afine Form). Let $\Delta ^ { ( s ) } : = $ $\bar { L ^ { ( s ) } } [ \mathbf { u } ] - \mathbf { f } = \mathbf { 0 }$ be a regular linear DE of genuine order s, with nondegenerate highest-order part in the standard sense of[5, Section $2 . 6 J ,$ every admitted Lie point-symmetry generator has theform

$$
v = \sum _ { i = 1 } ^ { p } \xi ^ { i } ( { \bf { x } } ) \partial _ { x ^ { i } } + \sum _ { \alpha = 1 } ^ { q } \left[ A _ { \ \beta } ^ { \alpha } ( { \bf { x } } ) u ^ { \beta } + b ^ { \alpha } ( { \bf { x } } ) \right] \partial _ { u ^ { \alpha } } ,\tag{187}
$$

provided that either $\Delta ^ { ( s ) } = \mathbf { 0 }$ is a PDE with $s \geq 2$ or is an ODE with $s \geq 3 ,$ , where summation over index $\beta$ is assumed.

Hence, the admitted point generators are projectable in the independent variables andfiber-afine in the dependent variables.

The proof of Proposition 6 is provided in D.5. For PDE systems, the preceding structural restrictions established in Section 3 and this section do not by themselves imply finite dimensionality of the residual intrinsic symmetry algebra. In the linear case, removal of the additive solution sector $\boldsymbol { A } _ { \mathrm { s o l } }$ eliminates the universal functional freedom induced by superposition, while the PFA form restricts the admissible dependence of the remaining generator coeficients and thereby permits a substantially simpler realization of the represented generators. Neither operation, however, guarantees that the corresponding determining system contains no additional arbitrary functional freedom.

Proposition 6 does not extend the unrestricted PFA characterization to linear PDEs of first order. Nevertheless, for a scalar first-order linear PDE, the symmetry conditions can be characterized explicitly once the admissible generator is restricted to the PFA class.

Proposition 7 (PFA Symmetries of a First-Order Linear PDE). Let $\Delta ^ { ( 1 ) } [ u ] : = L [ u ] - f = a ^ { i } \left( \mathbf { x } \right) u _ { i } + c \left( \mathbf { x } \right) u -$ $f \left( \mathbf { x } \right) = 0$ be a regular scalar first-order linear PDE, and let $v = \xi ^ { i } \left( \mathbf { x } \right) \partial _ { x ^ { i } } + \left[ A \left( \mathbf { x } \right) u + b \left( \mathbf { x } \right) \right] \partial _ { u }$ be a PFA vector field. Then v is an admitted Lie point-symmetry generator if and only ifthere exists a scalar smoothfunction $\mu \left( \mathbf { x } \right)$ such that

$$
\begin{array} { r } { [ \pmb { \xi } , \pmb { a } ] = \mu \pmb { a } , } \\ { \pmb { a } \cdot \nabla A + \pmb { \xi } \cdot \nabla c = \mu c , } \\ { \pmb { \xi } \cdot \nabla f - L [ b ] - A f = \mu f . } \end{array}\tag{188}
$$

The same condition admits a compact operator-theoretic formulation.

Proposition 8 (Operator Criterion for First-Order Linear PDEs). Under the assumptions of Proposition 7, let $L =$ $\mathbf { \boldsymbol { a } } \cdot \nabla + \boldsymbol { c }$ and $R = A - { \pmb \xi } \cdot \nabla .$ Then the PFA field $v =$ $\pmb { \xi } \cdot \partial _ { \mathbf x } + \left( A u + b \right) \partial _ { u }$ is a Lie point symmetry of $L [ u ] = f i f$ and only ifthere exists $\mu \in C ^ { \infty }$ such that

$$
[ L , R ] = \mu L , \quad L [ b ] + \left( R + \mu \right) f = 0 .\tag{189}
$$

The proofs of Propositions 7 and 8 are provided in D.6. These propositions characterize the admitted symmetries within the PFA realization; they do not establish that every unrestricted Lie point symmetry of a first-order linear PDE must possess this form. Accordingly, first-order linear PDEs may be treated under a prescribed PFA model class, but completeness of the resulting recovery is understood relative to that restricted realization unless an independent structural argument establishes its exhaustiveness for the DE under consideration. No analogous PFA reduction is required for the ODE classes considered here: first-order ODEs are excluded from the present scope, whereas regular determined ODEs with $s = 2$ are treated through an unrestricted realization whose target point-symmetry algebra is already known to be finite-dimensional by Proposition 4.

The preceding results establish the applicability limits adopted by the present framework. Regular, determined ODE systems of genuine order $s > 1$ possess finite-dimensional local Lie point-symmetry algebras and are therefore admissible, subject to the RSP used to assess whether the chosen represented function space is suficiently rich. First-order ODEs are excluded outright because arbitrary nontrivial functional freedom remains even after quotienting by the characteristic-zero solution-tangent sector. For linear PDEs with $s \geq 2 ,$ Proposition 6 justifies the projectable fiberafine realization after removal of the additive solution sector. Scalar first-order linear PDEs may also be treated within a prescribed PFA realization according to Propositions 7 and 8, although completeness in that case remains relative to the restricted realization unless its exhaustiveness is established independently. Nonlinear PDEs may likewise be analyzed within the represented framework, but no unrestricted exact completeness claim is made without additional structural information excluding unresolved arbitrary functional freedom.

Accordingly, the RSP should be interpreted as a test of represented geometric completeness, not as a proof of exact formal closure. For admissible ODE systems with $s > 1$ exact finite dimensionality is already guaranteed and the preflight therefore addresses representability of the finite target algebra. For PDE systems, by contrast, stabilization alone cannot establish the absence of additional functional degrees of freedom in the exact determining system. Failure to stabilize prevents a completeness claim in either case, but does not by itself imply that the exact symmetry algebra is infinite-dimensional.

For convenience, Fig. E.3 summarizes these distinctions through a decision flowchart indicating whether a given DE class lies within the present scope, which admissible generator realization is used, and whether the RSP is required before sequential recovery.

## 6.2 GEOMETRIC SCOPE OF THE RECOVERED SYMMETRIES

The proposed framework recovers infinitesimal Lie pointsymmetry generators and therefore characterizes the local Lie-algebraic structure associated with transformations connected continuously to the identity. As discussed in Fixed Dictionary and Generator Realization, finite transformations generated by the recovered vector fields may subsequently be reconstructed through numerical integration of their prolonged flows. This reconstruction, however, should not be interpreted as recovery of the full global symmetry group.

In particular, the Lie algebra determines only the local structure of the identity-connected component. Disconnected components and discrete symmetries are not encoded by infinitesimal generators and therefore lie outside the direct scope of the present discovery procedure. Moreover, the Lie algebra does not determine the global topology of the associated Lie group: distinct connected Lie groups may possess isomorphic Lie algebras, being related through diferent global quotients of the corresponding simply connected covering group [36, Sections 1.3 and 5.8]. Consequently, even exact recovery of ${ \mathfrak { g } } _ { \mathrm { i n t } }$ —and hence of algebraic constructions determined solely by it, such as its universal enveloping algebra—would not by itself identify the full global symmetry group.

Accordingly, ${ \mathfrak { g } } _ { \mathrm { r e c } }$ should be interpreted as a numerical approximation to the local intrinsic symmetry algebra g<sub>int</sub> within the represented domain. The extension of the corresponding generators and finite transformations across larger or overlapping regions is a separate geometric problem; the patchwise strategy motivated in Fixed Dictionary and Generator Realization is therefore left for future development.

## 7 CONCLUSION

In this work, we developed a data-driven framework for recovering intrinsic Lie point symmetries of DE systems from sampled jet-space data without prescribing the symmetry generators a priori. Starting from the conventional infinitesimal invariance condition, we first showed that imposing symmetry admissibility, nontriviality, solution independence, and novelty simultaneously leads to a dificult multi-objective optimization problem. We therefore reformulated the discovery task in an operator-theoretic, feasible-space-first form. At a fixed finite representation level, the sampled determining equations define a linear admissible space that is further restricted by removal of the additive solution sector when applicable and by global novelty with respect to previously recovered generators, while geometric nontriviality is enforced through the prolonged characteristic information. A fixed spline-based dictionary, nonlinear in its inputs but linear in its trainable coeficients, then makes these restrictions computationally explicit and permits the intrinsic generators to be recovered sequentially. The resulting procedure distinguishes clean exhaustion of the numerically resolvable represented directions from recovery failure, thereby separating a meaningful represented-completeness statement from unsuccessful optimization.

The framework was developed together with an explicit analysis of the structural assumptions under which such a completeness statement is meaningful. In particular, the RSP provides an operational test of represented geometric completeness under successive dictionary refinement, rather than a substitute for formal involutivity or a proof that the exact determining system contains no unresolved functional freedom. Within these limits, the two-dimensional spatial wave equation provides a particularly informative benchmark: its symmetry structure is well understood, its linearity introduces an additive solution-dependent sector that must be removed, and its remaining represented intrinsic sector permits the complete discovery pipeline to be examined within a stabilized finite representation. The numerical study therefore probes not only sequential generator recovery, but also the progressive restriction of the represented geometric space, numerical regularity of increasingly constrained generators, reconstruction of known analytic symmetry directions, and the locality of the learned representation. Accordingly, the recovered algebra ${ \mathfrak { g } } _ { \mathrm { r e c } }$ is interpreted as a numerical approximation to the local intrinsic symmetry structure g<sub>int</sub> within the represented and sampled domain, rather than as recovery of the full global symmetry group.

Several natural extensions follow from the distinction between recovering an intrinsic infinitesimal structure and exploiting that structure geometrically. Under the applicability conditions established in Section 6, an immediate continuation of the present work is the construction of a G-coframe adapted to ${ \mathfrak { g } } _ { \mathrm { r e c } } .$ , interpreted as a numerical approximation to the relevant finite-dimensional intrinsic symmetry structure. In particular, we plan to employ the cross-section and moving-frame constructions of Fels and Olver [12, 37] to obtain symmetry-adapted invariant coordinates and thereby reduce or re-express a given DE system $\Delta ^ { ( s ) } = 0$ . For practical applications, however, the governing DE cannot generally be considered independently of its prescribed initial and boundary data. The symmetry reduction must therefore be restricted to those recovered infinitesimal directions whose associated transformations preserve not only the DE, but also the relevant domain, boundary manifolds, and imposed data of the corresponding initial-boundary value problem [45]. Assuming that the numerically recovered generators identify, to the required accuracy, a Lie subalgebra with this invariance property, a diferentiable computational realization could combine infinitesimal invariance constraints with neural-ODE-based integration of the corresponding local flows [16] to construct invariant coordinates adapted to the associated connected subgroup ${ \widetilde { G } } _ { 0 } \subseteq G _ { 0 }$ . The resulting reduced DE system could then be solved, for example using PINNs [18], and mapped back to the original coordinates so that the accuracy and practical utility of the symmetry-adapted reconstruction can be quantified.

The current spline-based dictionary representation $D _ { ( k ) }$ is inherently local over $\Omega _ { \mathbf { X } }$ , and its extrapolation accuracy eventually deteriorates away from the training support, as demonstrated in Section 5. Independently rediscovering ${ \mathfrak { g } } _ { \mathrm { r e c } }$ over a collection of local domains $\Omega _ { \mathbf { X } } ^ { [ a ] } , a = 1 , 2 , \ldots ,$ would, however, run counter to the intended use of the proposed framework: the intrinsic algebra should be discovered once and subsequently reused across diferent domains and problems. Taking inspiration from the atlas construction of Floryan and Graham [46], we therefore envision a patchwise transport procedure over overlapping domains satisfying $\Omega _ { \mathbf { X } } ^ { [ a ] } \cap \dot { \Omega } _ { \mathbf { X } } ^ { [ a + 1 ] } \neq \emptyset$ . For each recovered generator, a patch-transfer operator

$$
T _ { a \to a + 1 } : \Theta _ { j } ^ { [ a ] } \longmapsto \Theta _ { j } ^ { [ a + 1 ] }\tag{190}
$$

could be constructed by requiring consistency of the corresponding local realizations over the overlap,

$$
v _ { j } ^ { [ a + 1 ] } \Big | _ { { \bf S } ^ { [ a , a + 1 ] } } \approx v _ { j } ^ { [ a ] } \Big | _ { { \bf S } ^ { [ a , a + 1 ] } } ,\tag{191}
$$

where $\Theta _ { j } ^ { [ a ] }$ denotes the local coeficient representation of the already recovered generator $v _ { j } .$ , and $\mathbf { S } ^ { [ a , a + 1 ] }$ is sampled from the overlap $\Omega _ { \mathbf { X } } ^ { [ a , \overline { { { a + 1 } } } ] } : = \bar { \Omega _ { \mathbf { X } } ^ { [ a ] } } \cap \Omega _ { \mathbf { X } } ^ { [ a + 1 ] }$ . In this way, patchwise continuation would transfer local representations of already discovered intrinsic directions rather than restart the sequential discovery procedure on every domain, thereby providing an atlas-like extension of ${ \mathfrak { g } } _ { \mathrm { r e c } }$ beyond the original training support.

Another application that we find particularly interesting is flow-based generative modeling, in which an invertible transformation transports a simple reference density $\mu \left( \mathbf { z } \right)$ toward a target density through a continuous flow [47]. If $\phi _ { t }$ denotes the associated flow map generated by a velocity field $\mathbf { w } _ { t }$ , with

$$
\frac { d } { d t } \phi _ { t } \left( \mathbf { z } \right) = \mathbf { w } _ { t } \left( \phi _ { t } \left( \mathbf { z } \right) , t \right) ,\tag{192}
$$

then the transported density $\rho _ { t }$ satisfies the Liouville (or continuity) equation

$$
\begin{array} { r } { \partial _ { t } \rho _ { t } + \nabla _ { \mathbf { z } } \cdot \left( \rho _ { t } \mathbf { w } _ { t } \right) = 0 . } \end{array}\tag{193}
$$

For a prescribed velocity field, (193) is linear in $\rho _ { t }$ . In densitydriven transport constructions such as the dual-ascent formulation of Tabak and Vanden-Eijnden [48], however, $\mathbf { w } _ { t }$ is itself determined from the evolving density and its spatial derivatives, so that substitution into (193) yields a nonlinear density-evolution PDE. This provides a natural setting in which to apply the proposed framework: intrinsic symmetries of the resulting density dynamics could first be recovered, followed by the construction of symmetry-adapted group and invariant coordinates. The generative transport could then be parameterized primarily in the invariant coordinates rather than requiring the learned flow to repeatedly represent motion along symmetry-equivalent directions. If successful, such a reduction could decrease the efective complexity of the transport problem and improve its numerical conditioning and eficiency. We emphasize the density-evolution formulation here because it lies directly within the DE-based Lie point-symmetry architecture developed in this work. A corresponding treatment formulated directly at the level of likelihood-based optimization or an associated variational functional would require additional machinery, including variational symmetries where appropriate [5, Chapter 4], and is therefore left outside the scope of the present framework.

The preceding three research directions represent concrete extensions of the present framework. A more foundational continuation is to strengthen the criterion by which closure of the represented determining structure is assessed. As emphasized throughout the manuscript, the current RSP is intentionally an operational represented test: stabilization of the sampled geometric rank defined in (152) does not establish formal integrability, involutivity, or the absence of unresolved arbitrary functional freedom, as discussed in A. A natural long-term objective is therefore to develop a numerical/operator-theoretic analogue of Cartan involutivity for the determining system itself. Such a procedure would examine the behavior of the determining equations under successive diferential prolongation—not to be confused with the prolongation operators introduced in Section 2; see Remark 10—and numerically monitor compatibility conditions, symbol growth, rank evolution across prolonged levels, and the appearance or disappearance of new diferential consequences. The objective would be to determine whether the determining system reaches a finite-type closure, in which case a finite represented symmetry space would be supported by the prolonged diferential structure itself, or instead retains persistent symbolic freedom indicative of an unresolved functional symmetry sector. Such a diagnostic would provide a substantially stronger foundation for completeness claims than stabilization of a finite sampled dictionary rank alone.

Taken together, the present work represents a step from data-driven approximation of individual infinitesimal symmetries toward a systematic operator-theoretic framework for recovering and exploiting the geometric structure of DE systems. The emphasis throughout has been not only on identifying admissible generators, but also on distinguishing what can be inferred reliably within a finite represented and sampled setting from what requires deeper geometric information. The future directions outlined above aim to extend this perspective from symmetry discovery toward symmetryadapted reduction, transport, probabilistic modeling, and ultimately stronger numerical characterizations of formal geometric closure. In this sense, the broader objective is to bring data-driven computation and the diferential-geometric structure of DEs into a common computational framework.

## ACKNOWLEDGEMENTS

I.G. was financially supported by the Scientific and Technological Research Council of Türkiye (TÜBİTAK) and the Council of Higher Education (YÖK) in Türkiye through the 2211-C and 100/2000 scholarship programs, respectively. The authors also gratefully acknowledge TÜBİTAK for providing access to the National Grid System of Türkiye (TRUBA), which was instrumental in meeting the computational requirements associated with the development and testing of the accompanying software.

I.G. gratefully acknowledges the sustained support and guidance of Prof. Peter J. Olver over the past three years. His contributions included providing crucial theoretical insights, helping resolve numerous challenges arising from the advanced mathematical background required for this research, and ofering guidance on state-of-the-art developments in continuous group symmetries of PDEs and G-coframe construction.

I.G. also gratefully acknowledges Pelin Çiloğlu for her extensive guidance and discussions on optimization theory, functional analysis, and real analysis. Her support during the earlier stages of I.G.’s mathematical development contributed substantially to the analytical foundation that later proved valuable in carrying out this research.

## DATA AVAILABILITY

No external datasets were used in this study. All datasets analyzed in the numerical experiments were generated computationally using the procedures described in the manuscript. The initial jet-space dataset consists of samples on Σ<sup>(s)</sup>, while the additional datasets used in the symmetry-recovery and generator-approximation experiments were generated from the recovered or prescribed symmetry generators. The data-generation routines and all information required to regenerate the datasets used in the reported numerical experiments will be made available together with the accompanying source code.

## CODE AVAILABILITY

The source code implementing the data-driven symmetrydiscovery framework and the complete computational workflow used for the numerical experiments in this study has been prepared in a Code Ocean capsule. The capsule includes the Python package source, the YAML configuration files defining the manuscript benchmark, scheduler-independent task wrappers for data generation, sequential symmetry discovery, custom-generator approximation, integral-curve generation, visualization, and Lie-bracket closure testing, together with curated software tests and documentation for installation and reproduction.

The software is distributed as a self-contained Python 3.12+ project and can be installed using the package metadata supplied with the release. The manuscript workflow can be executed either task-by-task or through the supplied fullworkflow driver. The accompanying README documents the reference software environment, the expected computational products, and reproducibility considerations associated with stochastic equation-shell sampling and acceleratordependent floating-point arithmetic. The numerical data used for the reported results are described separately in the Data availability statement. At the time of this preprint, the capsule is being kept private while journal peer review is ongoing; a public persistent link will be added to a future version of the preprint once the capsule is released.

## REFERENCES

[1] D.S. Dummit and R.M. Foote. Abstract Algebra. Wiley, 2003. ISBN 9780471433347. URL https://books. google.com/books?id=KIGbCgAAQBAJ.

[2] Sigurdur Helgason. Sophus lie and the role of lie groups in mathematics. MIT OpenCourseWare, 1990. Opening lecture at the Nordic Teachers Conference, Reykjavik.

[3] Thomas Andrew Ivey and Joseph M Landsberg. Cartan for beginners: diferential geometry via moving frames and exterior diferential systems, volume 61. American Mathematical Society Providence, RI, 2003.

[4] Robert B Gardner. The method ofequivalence and its applications. SIAM, 1989.

[5] Peter J Olver. Applications of Lie groups to diferential equations, volume 107. Springer Science & Business Media, 2000.

[6] Sadri Hassani. Mathematical physics: a modern introduction to its foundations. Springer Science & Business Media, 2013.

[7] Nail H Ibragimov. CRC handbook of Lie group analysis ofdiferential equations, volume 3. CRC press, 1995.

[8] Fritz Schwarz. An algorithm for determining the size of symmetry groups. Computing, 49(2):95–115, 1992.

[9] John Carminati and Khai Vu. Symbolic computation and diferential equations: Lie symmetries. Journal of Symbolic Computation, 29(1):95–116, 2000.

[10] Gregory J Reid. Finding abstract lie symmetry algebras of diferential equations without integrating determining equations. European Journal ofApplied Mathematics, 2(4):319–340, 1991.

[11] Gregory J Reid. Algorithms for reducing a system of pdes to standard form, determining the dimension of its solution space and calculating its taylor series solution. European Journal ofApplied Mathematics, 2(4):293–318, 1991.

[12] Mark Fels and Peter J Olver. Moving coframes: I. a practical algorithm. Acta Applicandae Mathematica, 51:161–213, 1998.

[13] Stéphane Mallat. Understanding deep convolutional networks. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 374(2065):20150203, 2016.

[14] Marc Finzi, Max Welling, and Andrew Gordon Wilson. A practical method for constructing equivariant multilayer perceptrons for arbitrary matrix groups. In International conference on machine learning, pages 3318–3328. PMLR, 2021.

[15] Danilo Jimenez Rezende, Sébastien Racanière, Irina Higgins, and Peter Toth. Equivariant hamiltonian flows. arXiv preprint arXiv:1909.13739, 2019.

[16] Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary diferential equations. arXiv preprint arXiv:1806.07366, 2018.

[17] Jens Behrmann, Will Grathwohl, Ricky TQ Chen, David Duvenaud, and Jörn-Henrik Jacobsen. Invertible residual networks. In International Conference on Machine Learning, pages 573–582. PMLR, 2019.

[18] Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal ofComputational Physics, 378:686–707, 2019.

[19] Patrick Kidger and Terry Lyons. Universal approximation with deep narrow networks. In Conference on learning theory, pages 2306–2327. PMLR, 2020.

[20] Steven L Brunton, Marko Budišić, Eurika Kaiser, and J Nathan Kutz. Modern koopman theory for dynamical systems. arXiv preprint arXiv:2102.12086, 2021.

[21] Bernard O Koopman. Hamiltonian systems and transformation in hilbert space. Proceedings of the national academy of sciences of the united states ofAmerica, 17(5):315, 1931.

[22] Marko Budišić, Ryan Mohr, and Igor Mezić. Applied koopmanism. Chaos: An Interdisciplinary Journal of Nonlinear Science, 22(4):047510, 2012.

[23] Johannes Brandstetter, Max Welling, and Daniel E Worrall. Lie point symmetry data augmentation for neural pde solvers. In International Conference on Machine Learning, pages 2241–2256. PMLR, 2022.

[24] Grégoire Mialon, Quentin Garrido, Hannah Lawrence, Danyal Rehman, Yann LeCun, and Bobak Kiani. Selfsupervised learning with lie symmetries for partial diferential equations. Advances in Neural Information Processing Systems, 36:28973–29004, 2023.

[25] Tara Akhound-Sadegh, Laurence Perreault-Levasseur, Johannes Brandstetter, Max Welling, and Siamak Ravanbakhsh. Lie point symmetry and physics-informed networks. In Advances in Neural Information Processing Systems, volume 36, 2023.

[26] Samuel E Otto, Nicholas Zolman, J Nathan Kutz, and Steven L Brunton. A unified framework to enforce, discover, and promote symmetry in machine learning. arXiv preprint arXiv:2311.00212, 2023.

[27] Behrooz Tahmasebi and Melanie Weber. Adaptive symmetry discovery for dynamical system identification. In Forty-third International Conference on Machine Learning, 2026.

[28] Shida Liu, Abhishek Gupta, Sumit Sinha, and L. Mahadevan. LieStoNet: Learning lie symmetries from spatiotemporal data for stochastic dynamical systems. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 76384–76408. PMLR, 2026. URL https://proceedings.mlr. press/v306/liu26aq.html.

[29] Shida Liu, Sumit Sinha, and Vishaal Krishnan. LieDyn-Net: Learning lie symmetries from spatiotemporal data, 2026. URL https://openreview.net/forum?id= VgZ8BJkneV. OpenReview preprint, ICLR 2026 submission.

[30] John Starrett. Solving diferential equations by symmetry groups. The American Mathematical Monthly, 114(9):778–792, 2007.

[31] Peter J Olver. Equivalence, invariants and symmetry. Cambridge University Press, 1995.

[32] Werner M. Seiler. Involution: The Formal Theory of Diferential Equations and Its Applications in Computer Algebra, volume 24 of Algorithms and Computation in Mathematics. Springer, Berlin, Heidelberg, 2010. doi: 10.1007/978-3-642-01257-7.

[33] John M Lee. Smooth manifolds. In Introduction to smooth manifolds, pages 1–31. Springer, 2013.

[34] Paul Renteln. Manifolds, tensors, andforms: an introduction for mathematicians and physicists. Cambridge University Press, 2013.

[35] Robert Gilmore. Lie Groups, Lie Algebras, and Some ofTheir Applications. Wiley-Interscience, New York, 1974. ISBN 9780471301790.

[36] Brian Hall. Lie groups, Lie algebras, and representations: an elementary introduction, volume 222. Springer, 2015.

[37] Mark Fels and Peter J Olver. Moving coframes: Ii. regularization and theoretical foundations. Acta Applicandae Mathematica, 55:127–208, 1999.

[38] George Bluman and Stephen Anco. Symmetry and integration methodsfor diferential equations, volume 154. Springer Science & Business Media, 2008.

[39] Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljacic, Thomas Hou, and Max Tegmark. Kan: Kolmogorov–arnold networks. In International conference on learning representations, volume 2025, pages 70367–70413, 2025.

[40] Carl de Boor. A Practical Guide to Splines, volume 27 of Applied Mathematical Sciences. Springer, New York, revised edition, 2001.

[41] Richard Haberman. Elementary applied partial diferential equations, volume 987. Prentice Hall Englewood Clifs, NJ, 1983.

[42] JC Ndogmo. Algebraic structure and maximal dimension of the symmetry algebra for arbitrary systems of odes. arXiv preprint arXiv:1606.08074, 2016.

[43] Giorgio Gubbiotti, Decio Levi, and Christian Scimiterna. On partial diferential and diference equations with symmetries depending on arbitrary functions. Acta Polytechnica, 56(3):193–201, 2016. doi: 10.14311/AP.2016.56.0193.

[44] Decio Levi, Luigi Martina, and Pavel Winternitz. Structure preserving discretizations of the liouville equation and their numerical tests. SIGMA, 11:080, 2015. doi: 10.3842/SIGMA.2015.080.

[45] Aldhlki T Jassim. Symmetry for initial boundary value problems of pdes. Journal of the College of Basic Education, 25(105):178–194, 2019.

[46] Daniel Floryan and Michael D Graham. Data-driven discovery of intrinsic dynamics. Nature Machine Intelligence, 4(12):1113–1120, 2022.

[47] Ivan Kobyzev, Simon JD Prince, and Marcus A Brubaker. Normalizing flows: An introduction and review of current methods. IEEE transactions on pattern analysis and machine intelligence, 43(11):3964–3979, 2020.

[48] Esteban G. Tabak and Eric Vanden-Eijnden. Density estimation by dual ascent of the log-likelihood. Communications in Mathematical Sciences, 8(1):217–233, 2010. doi: 10.4310/CMS.2010.v8.n1.a11.

[49] Michael E. Taylor. Partial Diferential Equations I: Basic Theory, volume 115 of Applied Mathematical Sciences. Springer, New York, 2 edition, 2011. ISBN 978- 1-4419-7054-1. doi: 10.1007/978-1-4419-7055-8.

[50] Gerald B. Folland. Real Analysis: Modern Techniques and Their Applications. John Wiley & Sons, 2 edition, 1999.

# Data-driven discovery of intrinsic Lie point symmetries of diferential equation systems

(Supplementary Material)

Ilker Gurcan<sup>1</sup>

Kadir Ustun<sup>2</sup>

Ramazan Gokberk Cinbis<sup>1,3</sup>

<sup>1</sup>Department of Computer Engineering, Middle East Technical University, Ankara, Türkiye <sup>2</sup>ASELSAN Inc., Ankara, Türkiye <sup>3</sup>ROMER Robotics and AI Center, Middle East Technical University, Ankara, Türkiye

## A GEOMETRIC VIEWPOINT AND FORMAL INTEGRABILITY

The material in this section is not required to follow the computational construction of the proposed framework. Nevertheless, it provides the geometric foundation for several statements made throughout the paper, including Remark 2, and clarifies the future trajectory of the present research. We therefore briefly review the relevant elements of Cartan’s exterior diferential-system formulation.

Definition 10 (Contact forms and contact ideal). Let $\Omega ^ { ( s ) , l } : = \Omega ^ { 1 } \left( J ^ { ( s ) } ( \pi ) \right)$ denote the space ofsmooth diferential l-forms on $J ^ { ( s ) } ( \pi ) ~ l ^ { 3 4 } ,$ , Section 3.11]. For example, $\begin{array} { r } { \omega = \sum _ { i } P _ { i } \big ( z ^ { ( s ) } \big ) d z ^ { ( s ) , i } \in \Omega ^ { ( s ) , 1 } } \end{array}$ is a one-form, where $P _ { i } : J ^ { ( s ) } ( \pi ) \to \mathbb R$ are its coeficients and $d z ^ { ( s ) , i }$ are the coordinate one-forms spanning $T ^ { * } J ^ { ( s ) } ( \pi )$

A one-form $\theta \in \Omega ^ { ( s ) , 1 }$ is called a contact form if

$$
( j ^ { s } i _ { f } ) ^ { * } \theta \equiv 0\tag{A.1}
$$

for every local section $i _ { f } : M \to E$ . Thus, the p-dimensional integral manifolds ofthe contact system satisfying the usual independence condition are locally the prolongations ofsections $o f \pi \ I 3 ,$ , Section 1.9].

In the jet coordinates introduced previously, the basic contact forms may be written as

$$
\theta _ { I } ^ { \alpha } = d u _ { I } ^ { \alpha } - \sum _ { i = 1 } ^ { p } u _ { I , i } ^ { \alpha } d x ^ { i } ,\tag{A.2}
$$

where $\alpha = 1 , \ldots , q , 0 \leq | I | \leq s - 1$ , and I = 0 gives $\begin{array} { r } { \theta ^ { \alpha } = d u ^ { \alpha } - \sum _ { i } u _ { i } ^ { \alpha } d x ^ { i } } \end{array}$ . Writing

$$
\theta _ { k } : = \{ \theta _ { I } ^ { \alpha } : | I | = k , \alpha = 1 , \ldots , q \} ,\tag{A.3}
$$

the order-s contact ideal is the diferential ideal [3, Appendix B.4] generated by $\pmb { \theta } _ { 0 } , \dots , \pmb { \theta } _ { s - 1 }$

$$
\mathcal { C } ^ { ( s ) } : = \langle \pmb { \theta } _ { 0 } , \ldots , \pmb { \theta } _ { s - 1 } \rangle _ { \mathrm { d i f f } } .\tag{A.4}
$$

For a local solution section $i _ { f }$ of the DE system, its prolonged graph $\Gamma _ { f } ^ { ( s ) } = j ^ { s } i _ { f } ( { \cal M } )$ therefore satisfies

$$
( j ^ { s } i _ { f } ) ^ { \ast } \left( \theta _ { I } ^ { \alpha } \right) \equiv 0 , \quad \alpha = 1 , \dots , q , \quad 0 \leq | I | < s .\tag{A.5}
$$

Since pull-back $\left( j ^ { s } i _ { f } \right) ^ { \ast }$ commutes with exterior diferentiation d and wedge products, every diferential form in $\mathcal { C } ^ { ( s ) }$ consequently vanishes when pulled back to $\Gamma _ { f } ^ { ( s ) }$

Definition 11 (Exterior diferential system). Let $\Delta ^ { ( s ) } = \mathbf { 0 }$ be a regular DE system with equation manifold $\Sigma ^ { ( s ) } \subset J ^ { ( s ) } ( \pi )$ and let $\iota _ { \Sigma } : \Sigma ^ { ( s ) } \hookrightarrow J ^ { ( s ) } ( \pi )$ denote the inclusion map. The diferential ideal

$$
\mathcal { T } _ { \mathrm { d i f f } } ^ { ( s ) } : = \iota _ { \Sigma } ^ { * } \mathcal { C } ^ { ( s ) }\tag{A.6}
$$

together with the independence form

$$
\Omega : = d x ^ { 1 } \wedge \dots \wedge d x ^ { p }\tag{A.7}
$$

defines the EDS associated with the DE system, subject to the independence condition $\Omega \neq 0$ on its admissible p-dimensional integral elements.

At a point $z ^ { ( s ) } \in \Sigma ^ { ( s ) }$ , a p-dimensional plane $\mathcal { E } _ { z ^ { ( s ) } } \subset T _ { z ^ { ( s ) } } \Sigma ^ { ( s ) }$ is called an integral element of the EDS [3, Definition 1.9.3] $i f$

$$
\omega | _ { \mathscr { E } _ { z ^ { ( s ) } } } = 0 \quad \forall \omega \in \mathcal { I } _ { \mathrm { d i f f } } ^ { ( s ) } , \quad \Omega | _ { \mathscr { E } _ { z ^ { ( s ) } } } \neq 0 .\tag{A.8}
$$

One may regard the independence condition $\Omega | _ { \varepsilon _ { \scriptscriptstyle z } ( s ) } \neq 0$ as restricting attention to p-dimensional integral elements that project nondegenerately onto the base space and are therefore locally graph-like. On the other hand, $\mathcal { T } _ { \mathrm { d i f f } } ^ { ( s ) }$ imposes the diferential constraints that an admissible integral element must satisfy in order to be compatible with the jet structure of a prolonged local section. Whether such pointwise integral elements (i) exist with the required dimension, (ii) vary smoothly, and (iii) can be “fitted together” to form a p-dimensional integral manifold is a separate integrability question [31, Chapters 14–15], which motivates the compatibility and involutivity considerations discussed below.

Consider the regular first-order ODE introduced previously in Jet-Space Geometry and Diferential Equations,

$$
\Delta ^ { ( 1 ) } : = u _ { x } - x u = 0 .\tag{A.9}
$$

Its first-order jet bundle has local coordinates $z ^ { ( 1 ) } = ( x , u , u _ { x } )$ , while its equation manifold is

$$
\Sigma ^ { ( 1 ) } = \left\{ ( x , u , u _ { x } ) \in J ^ { ( 1 ) } ( \pi ) : u _ { x } = x u \right\} .\tag{A.10}
$$

Since $p = q = s = 1$ , the contact ideal $\mathcal { T } _ { \mathrm { d i f f } }$ is generated by the single contact form

$$
\theta = d u - u _ { x } d x .\tag{A.11}
$$

Restricting it to $\Sigma ^ { ( 1 ) }$ gives

$$
\theta \big | _ { \Sigma ^ { ( 1 ) } } = d u - x u d x ,\tag{A.12}
$$

while the independence form is simply $\Omega = d x$

At a point $z ^ { ( 1 ) } = ( x , u , x u ) \in \Sigma ^ { ( 1 ) }$ , let $\mathcal { E } _ { z ^ { ( 1 ) } } \subset T _ { z ^ { ( 1 ) } } \Sigma ^ { ( 1 ) }$ be a one-dimensional integral element. The conditions

$$
\theta | _ { \mathcal { E } _ { z ^ { ( 1 ) } } } = 0 , \quad d x | _ { \mathcal { E } _ { z ^ { ( 1 ) } } } \neq 0\tag{A.13}
$$

imply that its tangent direction may be represented by

$$
w = \partial _ { x } + x u \partial _ { u } + u \left( 1 + x ^ { 2 } \right) \partial _ { u _ { x } } ,\tag{A.14}
$$

where $\partial _ { u }$ coeficient is given by the contact condition, while tangency to $\Delta ^ { ( 1 ) } = 0$ defines $\partial _ { u _ { x } }$ coeficient. Observe that w has the characteristic $Q _ { w } | _ { \Sigma ^ { ( 1 ) } } \equiv 0$ implying that w is trivial with respect to Definition 7. Consequently, fitting these integral elements together produces integral curves whose projection onto $E$ satisfies $u _ { x } = x u$ . Their solution sections are $u = f ( x ) = C e ^ { x ^ { 2 } / 2 }$ , and their prolonged graphs

$$
\Gamma _ { f } ^ { ( 1 ) } = \left\{ \left( x , C e ^ { x ^ { 2 } / 2 } , x C e ^ { x ^ { 2 } / 2 } \right) \right\}\tag{A.15}
$$

are one-dimensional integral manifolds of the EDS.

The previous example presents a case in which the admissible integral elements can be smoothly fitted together into genuine solution graphs. For a general DE system, however, the existence of pointwise integral elements does not guarantee such integrability. The obstructions to doing so lead to the notions of compatibility, prolongation, torsion, and involutivity considered next.

The regularity assumptions imposed on $\Sigma ^ { ( s ) }$ earlier ensure that, locally, the DE system defines a smooth diferential submanifold and satisfies the standard nondegeneracy conditions required for the geometric treatment of its solutions [5, Section 2.6]. Regularity alone, however, does not guarantee that admissible jets extend consistently to higher orders.

Formal compatibility is examined by successively prolonging the DE system through total derivatives defined (35) and projecting the resulting higher-order equation manifolds back to lower jet orders. If, for some $l \geq 0$

$$
\pi _ { s + l } ^ { s + l + 1 } \left( \Sigma ^ { \left( s + l + 1 \right) } \right) \subsetneq \Sigma ^ { \left( s + l \right) } ,\tag{A.16}
$$

the prolongation has revealed additional integrability (compatibility) conditions that must be incorporated into the system. Once no new independent lower-order conditions arise under further prolongation, the system isformally integrable [32, Section 2.3].

In Cartan’s EDS formulation, the same compatibility problem is encoded locally through the associated Pfafian system, its tableau, and torsion. We briefly introduce these objects next. To pass from the EDS formulation above to Cartan’s linear Pfafian description, we retain the same restricted contact forms but distinguish between their degree-one span and the diferential ideal they generate. Namely, let

$$
\begin{array} { r } { \boldsymbol { \mathcal { Z } } ^ { ( s ) } : = \operatorname { s p a n } _ { C ^ { \infty } ( \Sigma ^ { ( s ) } ) } \{  \theta _ { I } ^ { \alpha } | _ { \Sigma ^ { ( s ) } } : \alpha = 1 , \ldots , q ,  0 \le | I | < s \} , } \end{array}\tag{A.17}
$$

so that

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { d i f f } } ^ { ( s ) } = \Big \langle \mathcal { T } ^ { ( s ) } \Big \rangle _ { \mathrm { d i f f } } . } \end{array}\tag{A.18}
$$

Thus, $\boldsymbol { \mathcal { T } ^ { ( s ) } }$ and $\mathcal { T } _ { \mathrm { d i f f } } ^ { ( s ) }$ are generated by the same contact one-forms; the former denotes their Pfafian span in $\Omega ^ { 1 } ( \Sigma ^ { ( s ) } )$ , whereas the latter denotes the full diferential ideal they generate [3, Section 5.1].

Since the determining equations form an overdetermined linear PDE system for the infinitesimal coeficients, as described in Infinitesimal Symmetry Condition and Determining Equations, they may themselves be represented, after passage to the appropriate jet space and first-order formulation, as a linear Pfafian system [3, Example 5.1.4].

Remark 9. Throughout this appendix, we apply Cartan’s equivalence machinery to the original DE system. When the same construction is applied to its determining system, the collection $\left\{ d x ^ { 1 } , \ldots , d x ^ { p } ; d u ^ { 1 } , \ldots , d \bar { u } ^ { q } \right\}$ replaces $\left\{ d x ^ { 1 } , \ldots , d x ^ { p } \right\}$ as the independence one-forms, since the determining equations are posed over the total space E rather than the base space X. Accordingly, $( \xi , \varphi )$ become the dependent coordinates ofthe determining system.

Definition 12 (Linear Pfafian system). Let $\Delta ^ { ( s ) } = 0$ be a regular DE system with equation manifold $\Sigma ^ { ( s ) }$ , and let ${ \mathcal { T } } ^ { \left( s \right) } \subset { \Omega } ^ { 1 } \left( { \Sigma } ^ { \left( s \right) } \right)$ denote the Pfafian system generated by the restricted contact one-forms introduced in Definition 10. Its generated diferential ideal is $\mathcal { T } _ { \mathrm { d i f f } } ^ { ( s ) }$ in (A.6).

Let further $\omega ^ { i } : = d x ^ { i } , i = 1 , . . . , p ,$ satisfy the independence condition $\Omega : = \omega ^ { 1 } \wedge \cdot \cdot \cdot \wedge \omega ^ { p } \neq 0 ,$ , and define

$$
\mathcal T ^ { ( s ) } : = \mathcal T ^ { ( s ) } \cup \mathrm { s p a n } \left\{ \omega ^ { 1 } , \dots , \omega ^ { p } \right\} .\tag{A.19}
$$

The pair $\big ( \mathcal { T } ^ { ( s ) } , \mathcal { T } ^ { ( s ) } \big )$ is called a linear Pfafian system $i f$

$$
d \theta ^ { a } \equiv 0 \left( \mathrm { m o d } \mathcal { I } ^ { ( s ) } \right)\tag{A.20}
$$

for every local generator θ<sup>a</sup> $o f \mathcal { T } ^ { ( s ) } \left[ 3 , \right.$ Definition $5 . l . l { \mathit { l } } { \mathit { l } } .$

Let $\left\{ \tau ^ { \beta } \right\}$ be one-forms complementary to $\mathcal { I } ^ { ( s ) }$ in $T ^ { * } \Sigma ^ { ( s ) }$ . Then the corresponding structure equations take the local form

$$
d \theta ^ { a } \equiv A _ { \beta i } ^ { a } \tau ^ { \beta } \wedge \omega ^ { i } + T _ { i j } ^ { a } \omega ^ { i } \wedge \omega ^ { j } \left( \mathrm { m o d } \mathcal { T } ^ { ( s ) } \right) ,\tag{A.21}
$$

where $A _ { \beta i } ^ { a }$ and $T _ { i j } ^ { a }$ are the tableau and torsion coeficients, respectively $I 3 ,$ Section $5 . I J .$

At a point $z ^ { ( s ) } \in \Sigma ^ { ( s ) }$ , let $V ^ { * } \cong \mathcal { T } ^ { ( s ) } / \mathcal { T } ^ { ( s ) }$ be spanned by the independence forms $\omega ^ { i }$ , and let $W ^ { * } \cong { \mathcal { T } } ^ { ( s ) }$ be spanned by the Pfafian generators $\theta ^ { a }$ . The coeficients $A _ { \beta i } ^ { a }$ therefore define the tableau

$$
\mathbf { A } : = \mathrm { s p a n } \left\{ \mathbf { A } _ { \beta } \right\} \subseteq W \otimes V ^ { * } ,\tag{A.22}
$$

which encodes the admissible highest-order diferential freedom of the system [3, Section 4.1]. In the language of PDE theory, this role is closely related to that of the principal symbol, which captures the highest-order diferential structure of a diferential operator or system [49, Section 9].

After restricting the complementary forms to an integral element ${ \mathcal { E } } _ { z ^ { ( s ) } }$ , the independence condition permits the local graph representation

$$
\left( \tau ^ { \beta } - { \cal P } _ { i } ^ { \beta } \omega ^ { i } \right) \Big | _ { \mathscr { E } _ { z } ( s ) } = 0 ,\tag{A.23}
$$

where $P _ { i } ^ { \beta }$ are its slope coeficients. Substituting (A.23) into (A.21) and imposing $d \theta ^ { a } | _ { { \mathcal { E } } _ { z } ( s ) } = 0$ yields algebraic conditions on $P _ { i } ^ { \beta }$ . Their homogeneous part defines the first prolongation of the tableau,

$$
\mathbf { A } ^ { ( 1 ) } : = \left( \mathbf { A } \otimes V ^ { * } \right) \cap \left( W \otimes S ^ { 2 } V ^ { * } \right) \subseteq \mathbf { A } \otimes V ^ { * } ,\tag{A.24}
$$

which represents the admissible next-order diferential freedom associated with A [3, Section 4.1].

Remark 10. Although $\mathbf { A } \longmapsto \mathbf { A } ^ { ( 1 ) } , \operatorname { p r } ^ { ( s ) } : { \mathfrak { X } } ( E ) \to { \mathfrak { X } } \left( J ^ { ( s ) } ( \pi ) \right)$ in (34), and $j ^ { s } i _ { f } : M \to J ^ { ( s ) } ( \pi )$ are all referred to as prolongations, they act on diferent geometric objects. The first prolongs a tableau in tensor space, the second prolongs a vector field, and the third prolongs a local section. Their common feature is the encoding of higher-order diferential information, but their outputs serve distinct geometric purposes.

Definition 13 (Involutive tableau). Let $\left( \mathcal { T } ^ { \left( s \right) } , \mathcal { T } ^ { \left( s \right) } \right)$ be a linear Pfafian system with tableau A. The tableau A is called involutive ifit satisfies Cartan’s test, evaluated through its first prolongation $\mathbf { A } ^ { ( 1 ) }$ , as specified in Definition 11.7 of[31] or Definition 4.5.4 of [3].

At the tableau level, involutivity means that the tableau prolongation has exactly the dimension predicted by Cartan’s test, with subsequent higher-order freedom determined by the Cartan characters. If A is not involutive, the system is prolonged and $\mathbf { A } ^ { ( 1 ) }$ becomes the next tableau to be tested, now through $\mathbf { A } ^ { ( 2 ) }$ ; the same procedure is repeated at successive orders as needed [3, Sections 5.5–5.6]. Once an involutive tableau is reached, its Cartan characters quantify the remaining diferential freedom. In particular, a non-Frobenius determining system need not possess arbitrary functional freedom, since its tableau may eventually become trivial; i.e. $\mathbf { A } ^ { ( l ) } = 0$ for some $l \geq 1$ , after finitely many prolongations. Once an involutive nonzero tableau is reached, however, at least one Cartan character is positive, and the corresponding Cartan–Kähler data include arbitrary functions, thereby yielding an infinite-dimensional Lie point-symmetry algebra. See Definition 1.3.5 and Example 4.1.4 in [3] for the definition of a Frobenius system and the corresponding tableau, respectively.

The torsion coeficients in (A.21) are not all intrinsic obstructions to integrability, since the complementary forms $\{ \tau ^ { \beta } \}$ are not uniquely determined. In particular, they may be replaced locally by

$$
\widetilde { \tau } ^ { \beta } = \tau ^ { \beta } + P _ { i } ^ { \beta } \omega ^ { i } \quad \Big ( \mathrm { m o d } \mathcal { T } ^ { ( s ) } \Big ) .\tag{A.25}
$$

Such a change leaves the tableau unchanged, but modifies the apparent torsion by

$$
\widetilde { \mathbf { T } } = \mathbf { T } + \delta ( \mathbf { P } ) ,\tag{A.26}
$$

where

$$
\delta : \mathbf { A } \otimes V ^ { * } \longrightarrow W \otimes \Lambda ^ { 2 } V ^ { * }\tag{A.27}
$$

is the alternation map. Consequently, only the equivalence class

$$
[ \mathbf { T } ] \in \frac { W \otimes \Lambda ^ { 2 } V ^ { * } } { \mathrm { i m } \delta } = \mathrm { c o k e r } \delta\tag{A.28}
$$

is independent of the choice of complementary forms and is called the essential torsion [3, Section 5.5].

Indeed, after representing an integral element by (A.23), the condition $d \theta ^ { a } | _ { { \mathcal { E } } _ { z } ( s ) } = 0$ takes the afine form

$$
\delta ( \mathbf { P } ) + \mathbf { T } = \mathbf { 0 } { \mathrm { . } }\tag{A.29}
$$

With the sign convention adopted above, this is precisely the condition $\widetilde { \textbf { T } } = \textbf { 0 }$ under the complementary-coframe transformation (A.25). Hence, an admissible integral element can exist at $z ^ { ( s ) }$ only $\mathrm { i f } \left[ \mathbf { T } \right] = \mathbf { 0 }$ . Whenever this condition holds and $\mathbf { P } _ { 0 }$ is a particular solution to (A.29), the apparent torsion can be absorbed by the corresponding change of complementary coframe, while the remaining homogeneous freedom is precisely ker $\delta = \mathbf { A } ^ { ( 1 ) }$ . Equivalently, the admissible integral elements form an afine space modeled on $\mathbf { \bar { A } } ^ { ( 1 ) }$ ; i.e. $\mathbf { P } _ { 0 } +$ ker δ.

$\operatorname { I f } \left[ \mathbf { T } \right] = \mathbf { 0 }$ does not hold identically on $\Sigma ^ { ( s ) }$ , its vanishing imposes additional compatibility conditions and the system must be restricted to the corresponding locus

$$
\widetilde { \Sigma } ^ { \left( s \right) } : = \left\{ z ^ { \left( s \right) } \in \Sigma ^ { \left( s \right) } : \left[ \mathbf { T } \right] = \mathbf { 0 } \right\} .\tag{A.30}
$$

The linear Pfafian system and its tableau are then recomputed on $\widetilde \Sigma ^ { ( s ) }$ before Cartan’s involutivity test, introduced in Definition 13, is applied. Thus, the restriction $[ \mathbf { T } ] = \mathbf { 0 }$ provides the EDS-side manifestation of the same compatibility phenomenon expressed by the strict projection (A.16): it removes jets satisfying the original finite-order equations that cannot be extended consistently to admissible higher-order integral elements. Further prolongation may, however, revea additional integrability conditions not detected by the torsion at the current order.

The interplay between tableau involutivity and torsion gives rise to an iterative completion process. If the essential torsion does not vanish identically, restricting to $[ \mathbf { T } ] = \mathbf { 0 }$ in (A.30) introduces additional compatibility conditions, requiring the Pfafian system and its tableau to be recomputed. Conversely, if the resulting tableau is not involutive, prolongation produces a higher-order Pfafian system for which the torsion coeficients must again be evaluated. The complete Cartan procedure therefore alternates between these two operations until either inconsistency is detected or an involutive system is reached. Figure E.1 summarizes this process. The blue arrow indicates that, after prolongation, the one-forms defining the newly obtained Pfafian pair are relabeled as the current $\left( \mathcal { T } ^ { \left( s + l \right) } , \mathcal { T } ^ { \left( s + l \right) } \right)$ for the next iteration; the prolongation index l begins at 0. This flowchart also clarifies which components of Cartan’s procedure would ultimately be required to replace the RSP test described in Section 4.

The preceding Pfafian formulation also admits a complementary group-theoretic interpretation through Cartan’s method of equivalence. Gardner formulates an equivalence problem by choosing an adapted coframe and prescribing a matrix Lie group $G \subset G L ( n , \mathbb { R } )$ acting linearly on that coframe; admissible equivalences are therefore characterized, at first order, by Jacobians taking values in G [4, Lecture 1]. After lifting the problem to the corresponding G-space (principal bundle), the diferential freedom along the group fibers is encoded by the right-invariant Maurer–Cartan forms of G [31, Chapter 2]. Indeed, Gardner’s lifted structure equations contain the Maurer–Cartan matrix $d S S ^ { - 1 }$ , whose independent entries encode the Lie algebra of the structure group [4, Lectures 2–3].

## B SPLINE-DICTIONARY REDUNDANCY AND SAMPLING-INDUCED RANK DEFI-CIENCY

The nested construction in (144) provides increasing representational richness, but it also introduces an exact coeficientspace redundancy across interaction orders. In particular, the B-spline basis associated with each coordinate satisfies the partition-of-unity property on the core domain,

$$
\sum _ { \nu _ { j } = 1 } ^ { K _ { o } } B _ { \nu _ { j } } ^ { ( r ) } \left( \chi ^ { j } \right) = 1\tag{B.1}
$$

Hence, for any $i \neq j$

$$
B _ { \nu _ { i } } ^ { ( r ) } \left( \chi ^ { i } \right) = B _ { \nu _ { i } } ^ { ( r ) } \left( \chi ^ { i } \right) \sum _ { \nu _ { j } = 1 } ^ { K _ { o } } B _ { \nu _ { j } } ^ { ( r ) } \left( \chi ^ { j } \right) = \sum _ { \nu _ { j } = 1 } ^ { K _ { o } } B _ { \nu _ { i } } ^ { ( r ) } \left( \chi ^ { i } \right) B _ { \nu _ { j } } ^ { ( r ) } \left( \chi ^ { j } \right) ,\tag{B.2}
$$

i.e. coeficients across all $\nu _ { j } \mathbf { \ ' } \mathbf { s } ,$ for a fixed $\nu _ { i } ,$ are shared. Consequently, a first-order feature can be represented by a particular linear combination of second-order features, and analogous dependencies occur between successive higher interaction orders. The coeficient representation generated by $D _ { ( k ) }$ is therefore not necessarily identifiable, so the corresponding realization map need not be injective. This redundancy does not prevent genuinely higher-order dependence from being represented: the reduction in (B.2) occurs only for the specific coeficient pattern that reproduces the partition-of-unity sum.

A distinct source of degeneracy arises from the finite sampling of the spline dictionary. Let

$$
( \Phi _ { i } ) _ { l \nu } : = B _ { \nu } ^ { ( r ) } \left( \chi _ { l } ^ { i } \right) , l = 1 , \dots , N , \nu = 1 , \dots , K _ { o } .\tag{B.3}
$$

denote the evaluation matrix of the univariate B-spline basis associated with the coordinate $\chi ^ { i }$ over sampled inputs $\left\{ \chi _ { l } ^ { i } \right\} _ { l = 1 } ^ { N } .$ thereby each column corresponds to the sampled response of a particular B-spline basis function and $\Phi _ { i } \in \mathbb { R } ^ { N \times K _ { o } }$ . Although the B-spline functions are linearly independent on their underlying spline space, their restrictions to a finite sample set S with $| \mathbf { S } | = N$ need not remain numerically independent. If the samples insuficiently cover the supports of one or more basis functions, the corresponding columns of Φ may vanish, become weakly activated, or become nearly proportional to other sampled columns. Consequently, the sampled evaluation matrix may exhibit exact or approximate rank deficiency, even when no additional functional dependence is present in the continuous dictionary. In the approximately rank-deficient case, this manifests numerically as

$$
\sigma _ { \operatorname* { m i n } } \left( \Phi _ { i } \right) \ll \sigma _ { \operatorname* { m a x } } \left( \Phi _ { i } \right) ,\tag{B.4}
$$

whereas exact rank deficiency yields $\sigma _ { \mathrm { m i n } } \left( \Phi _ { i } \right) = 0$

This efect is amplified for tensor-product features in $D _ { ( k ) }$ , since weak activation of any univariate factor suppresses the corresponding multivariate feature. Thus, a mismatch between the spline grid and the efective sampled support, or more generally insuficient sample coverage of the spline supports, can introduce additional near-null directions in coeficient space. These sampling-induced deficiencies are distinguished from the exact structural redundancy in (B.2): the former depend on the particular sample set, whereas the latter are identities of the dictionary itself.

## C WHY SOLVABILITY IS NOT IMPOSED AS A DISCOVERY CONSTRAINT

A natural question arises in view of the limitations of the RSP and finite dictionary representation discussed in Section 4: “Why not impose solvability as an additional constraint during sequential recovery, so that the accepted generators are at least guaranteed to close into a solvable Lie subalgebra even when recovery of the entire intrinsic algebra cannot be certified?” The dificulty is that such a constraint changes the admissible recovery path itself and can therefore force premature termination inside a proper represented subalgebra. To make this point precise, we first recall the finite-dimensional characterization of solvability used in Definition 2.63 of [5].

Definition 14 (Solvable Lie Group and Lie Algebra). Let G be an r-dimensional Lie group ofpoint transformations with Lie algebra ${ \mathfrak { g } } _ { \mathrm { p t } }$ . The group G is called solvable ifthere exists a chain ofLie subgroups

$$
\{ e \} = G _ { 0 } \subset G _ { 1 } \subset \cdots \subset G _ { r } = G ,\tag{C.1}
$$

such that, for each $i = 1 , \ldots , r , G _ { i }$ is i-dimensional and $G _ { i - 1 }$ is a normal subgroup of $G _ { i } .$ . Equivalently, there exists a chain ofLie subalgebras

$$
\{ \mathbf { 0 } \} = \mathfrak { s } _ { 0 } \subset \mathfrak { s } _ { 1 } \subset \cdot \cdot \cdot \subset \mathfrak { s } _ { r } = \mathfrak { g } _ { \mathrm { p t } } ,\tag{C.2}
$$

with dim $( \mathfrak { s } _ { i } ) = i ,$ such that $\mathfrak { s } _ { i - 1 }$ is an ideal of ${ \mathfrak { s } } _ { i } f o r$ every $i = 1 , \ldots , r , i . e .$

$$
\left[ \mathfrak { s } _ { i - 1 } , \mathfrak { s } _ { i } \right] \subseteq \mathfrak { s } _ { i - 1 } .\tag{C.3}
$$

Let

$$
\mathfrak { g } _ { \mathrm { r e c } , i - 1 } : = \operatorname { s p a n } \left\{ v _ { 1 } , \ldots , v _ { i - 1 } \right\}\tag{C.4}
$$

denote the algebraic sector recovered after $i - 1$ accepted generators in the sequential recovery procedure detailed in Sequential Recovery and Stopping Criteria. Let $v _ { i }$ be a candidate for the next intrinsic direction under the assumptions that dim $\left( \widehat { N } _ { \mathrm { n o v } , i } ^ { ( k ) } \right) > 0$ and the initialization conditions in (169) are satisfied. If, in addition to the existing admissibility conditions, one enforces the normal-subalgebra condition required by the chain in (C.2),

$$
[ \mathfrak { g } _ { \mathrm { r e c } , i - 1 } , v _ { i } ] \subseteq \mathfrak { g } _ { \mathrm { r e c } , i - 1 } ,\tag{C.5}
$$

then a genuinely novel represented direction need not remain feasible. Consequently, the sequential procedure may terminate prematurely, even though additional intrinsic directions remain representable by the underlying finite dictionary. The resulting algebra can therefore be a proper subalgebra of the represented intrinsic sector solely because the imposed solvability constraint excludes otherwise admissible generators.

A simple example is the special linear Lie algebra ${ \mathfrak { s l } } ( 2 )$ with structure relations [36, Section 4.6]

$$
[ H , X ] = 2 X , [ H , Y ] = - 2 Y , [ X , Y ] = H .\tag{C.6}
$$

Suppose that after two recovery steps,

$$
{ \mathfrak { g } } _ { \mathrm { r e c , 2 } } = \operatorname { s p a n } \left\{ H , X \right\} .\tag{C.7}
$$

Although this two-dimensional space is a Lie subalgebra, it is not an ideal of ${ \mathfrak { s l } } ( 2 )$ , since

$$
[ Y , H ] = 2 Y \not \in { \mathfrak { g } } _ { \mathrm { r e c , 2 } } .\tag{C.8}
$$

More generally, every direction

$$
v _ { 3 } = a H + b X + c Y , \quad c \neq 0 ,\tag{C.9}
$$

that is linearly independent of $\mathfrak { g } _ { \mathrm { r e c } , 2 }$ satisfies

$$
[ v _ { 3 } , H ] = - 2 b X + 2 c Y \notin { \mathfrak { g } } _ { \mathrm { r e c } , 2 } .\tag{C.10}
$$

Hence no genuinely novel third generator can satisfy (C.5), despite the existence of the remaining represented direction required to recover the full three-dimensional algebra.

The dificulty is compounded by the fact that the sequential recovery order is not prescribed a priori. Diferent choices of initialization, dictionary resolution and interaction order, spline degree, optimization hyperparameters, and numerical tolerances can lead the algorithm to recover diferent admissible bases of the same represented intrinsic sector. Consequently, imposing the normal-subalgebra condition in (C.5) would make the set of admissible future directions depend not only on the underlying Lie algebra, but also on the particular basis and ordering encountered numerically during training. A legitimate generator could therefore be excluded solely because previously recovered generators happened to span a subalgebra that cannot be extended through the required normal chain in that order. Such a path-dependent restriction is incompatible with the objective of exhausting all numerically resolvable intrinsic directions represented by $D _ { ( k ) }$

Another structural issue is that solvability certifies neither finite dimensionality nor completeness of ${ \mathfrak { g } } _ { \mathrm { i n t } }$ . Indeed, any infinite-dimensional Abelian Lie algebra is solvable, since its first derived algebra is already trivial, showing that solvability alone does not imply finite dimensionality. Conversely, a finite-dimensional intrinsic algebra need not be solvable. Consider ${ \mathfrak { s l } } ( 2 )$ again. Its derived series (see Definition 3.14 in [36]) satisfies

$$
{ \mathfrak { s l } } ( 2 ) = : { \mathfrak { g } } _ { ( 0 ) } = { \mathfrak { g } } _ { ( 1 ) } = \cdot \cdot \cdot = { \mathfrak { g } } _ { ( j ) } = { \mathfrak { s l } } ( 2 ) ,\tag{C.11}
$$

for every finite $j ,$ and therefore never reaches the trivial algebra {0}. Hence, the finite-dimensional algebra ${ \mathfrak { s l } } ( 2 )$ is not solvable. Together with the premature-termination mechanism discussed above, these observations show that solvability is neither an appropriate completeness constraint for recovering all represented intrinsic directions nor a certificate of finite dimensionality. Passing the RSP test establishes only stabilization of the represented geometric rank under the prescribed dictionary refinement; it provides no reason to assume that the corresponding intrinsic algebra is solvable. Accordingly, solvability is not imposed as an admissibility or stopping condition in the present framework.

## D PROOFS

## D.1 GENERIC SHELL SAMPLING

This section provides the proof for Proposition 1. For the first statement, let $\pi ^ { s } : J ^ { ( s ) } \left( \pi \right) \to M$ denote the canonical jet projection in (14). The prolonged graph $\Gamma _ { f } ^ { ( s ) }$ is the image of the jet prolongation $j ^ { s } i _ { f } : M \to J ^ { ( s ) } \left( \pi \right)$ in (18), and

$$
\pi ^ { s } \circ j ^ { s } i _ { f } = \mathrm { i d } _ { M } .\tag{D.1}
$$

Hence $j ^ { s } i _ { f }$ is injective and its diferential has full rank dim $( M ) = p , \mathrm { s o } \Gamma _ { f } ^ { ( s ) }$ is an embedded p-dimensional submanifold. Since $\Gamma _ { f } ^ { ( s ) } \subset \Sigma ^ { ( s ) }$ and, by assumption, dim $\left( \Sigma ^ { ( s ) } \right) > p ,$ it follows that

$$
\mathrm { c o d i m } _ { \Sigma ^ { ( s ) } } \left( \Gamma _ { f } ^ { ( s ) } \right) = \mathrm { d i m } \left( \Sigma ^ { ( s ) } \right) - p > 0 .\tag{D.2}
$$

Therefore $\Gamma _ { f } ^ { ( s ) }$ is a positive-codimension embedded submanifold of $\Sigma ^ { ( s ) }$ and consequently has zero measure with respect to any smooth volume measure on $\Sigma ^ { ( s ) }$

For the second statement, let $\nu _ { \Sigma } ( s )$ denote a smooth volume measure on $\Sigma ^ { ( s ) }$ . The first statement gives

$$
\nu _ { \Sigma ^ { ( s ) } } \left( \Gamma _ { f } ^ { ( s ) } \right) = 0 .\tag{D.3}
$$

By assumption, the sampling distribution satisfies $\mu _ { \Sigma ^ { ( s ) } } \ll \nu _ { \Sigma ^ { ( s ) } }$ . By the definition of absolute continuity of measures [50, p. 88], every $\nu _ { \Sigma } ( s )$ -null set is also $\mu _ { \Sigma ^ { ( s ) } } { \mathrm { - n u l l } }$ , and hence

$$
\mu _ { \Sigma ^ { ( s ) } } \left( \Gamma _ { f } ^ { ( s ) } \right) = 0 .\tag{D.4}
$$

Thus, for every sampled jet $z _ { l } ^ { ( s ) }$ drawn from $\mu _ { \Sigma ^ { ( s ) } }$

$$
\mathbb { P } \left( z _ { l } ^ { ( s ) } \in \Gamma _ { f } ^ { ( s ) } \right) = 0 .\tag{D.5}
$$

Since

$$
\left\{ \mathbf { S } ^ { \mathrm { o n } } \subset \Gamma _ { f } ^ { ( s ) } \right\} \subseteq \left\{ z _ { l } ^ { ( s ) } \in \Gamma _ { f } ^ { ( s ) } \right\}\tag{D.6}
$$

for any $l \in \{ 1 , \ldots , N _ { \mathrm { o n } } \}$ , monotonicity of probability yields

$$
\mathbb { P } \left( \mathbf { S } ^ { \mathrm { o n } } \subset \Gamma _ { f } ^ { ( s ) } \right) = 0 ,\tag{D.7}
$$

which proves (42). Notice that no independence assumption among the finite collection of samples is required for this conclusion.

## D.2 ADDITIVE SOLUTION SUBALGEBRA AND IDEAL

This section provides the proof of Proposition 2. We first establish the general statement for a linear DE system

$$
\Delta ^ { ( s ) } : = \mathrm { L } ^ { ( s ) } \left[ \mathbf { u } \right] - \mathbf { g } = \mathbf { 0 } .\tag{D.8}
$$

Let $v _ { \varphi } \in \mathcal { A } _ { \mathrm { s o l } }$ , so that $\varphi$ belongs to the kernel of the associated homogeneous operator,

$$
\begin{array} { r } { \mathrm { L } ^ { ( s ) } \left[ \left[ j ^ { s } i _ { \varphi } \right] ( { \bf x } ) \right] = { \bf 0 } . } \end{array}\tag{D.9}
$$

The local one-parameter transformation generated by $v _ { \varphi }$ acts only in the dependent-variable directions and is given by

$$
\mathbf { u } \longmapsto \mathbf { u } + \varepsilon \varphi \left( \mathbf { x } \right) .\tag{D.10}
$$

By linearity of $\mathrm { L } ^ { ( s ) }$

$$
\begin{array} { c } { { \mathrm { L } ^ { \left( s \right) } \left[ { \bf { u } } + \varepsilon { \varphi } \right] - { \bf { g } } = \mathrm { L } ^ { \left( s \right) } \left[ { \bf { u } } \right] - { \bf { g } } + \varepsilon \mathrm { L } ^ { \left( s \right) } \left[ { \varphi } \right] } } \\ { { { \mathrm {  ~ \Omega ~ } } } } \\ { { = \mathrm { L } ^ { \left( s \right) } \left[ { \bf { u } } \right] - { \bf { g } } . } } \end{array}\tag{D.11}
$$

Hence the transformation preserves the solution set of the original, possibly inhomogeneous, linear DE system, and therefore every $v _ { \varphi } \in \mathcal { A } _ { \mathrm { s o l } }$ is an admitted Lie point-symmetry generator. Since ker $\dot { \mathrm { ~ L ~ } } ^ { ( s ) }$ is a vector space, $\boldsymbol { A } _ { \mathrm { s o l } }$ is closed under linear combinations.

It remains to verify closure under the Lie bracket. For any $v _ { \varphi _ { 1 } } , v _ { \varphi _ { 2 } } \in \mathcal { A } _ { \mathrm { s o l } }$ , the coeficient functions $\varphi _ { 1 } \left( \mathbf { x } \right)$ and $\varphi _ { 2 } \left( \mathbf { x } \right)$ depend only on the independent variables. Since both vector fields are purely vertical,

$$
v _ { \varphi _ { 1 } } \left( \varphi _ { 2 } ^ { \alpha } \right) = v _ { \varphi _ { 2 } } \left( \varphi _ { 1 } ^ { \alpha } \right) = 0 , \quad \alpha = 1 , \ldots , q .\tag{D.12}
$$

Consequently,

$$
[ v _ { \varphi _ { 1 } } , v _ { \varphi _ { 2 } } ] = \mathbf { 0 } .\tag{D.13}
$$

Thus $\mathcal { A } _ { \mathrm { s o l } } \leq \mathfrak { g }$ and is an Abelian Lie subalgebra of the admitted Lie point-symmetry algebra ${ \mathfrak { g } } _ { \mathrm { p t } }$

We now assume that the admitted Lie point-symmetry generators have the projectable fiber-afine form

$$
v = \sum _ { i = 1 } ^ { p } \xi ^ { i } \left( { \bf x } \right) \partial _ { x ^ { i } } + \sum _ { \alpha = 1 } ^ { q } \left( A _ { \beta } ^ { \alpha } \left( { \bf x } \right) u ^ { \beta } + b ^ { \alpha } \left( { \bf x } \right) \right) \partial _ { u ^ { \alpha } } ,\tag{D.14}
$$

where summation over $\beta$ is dropped for notational simplicity. Let $v _ { \varphi } \in \mathcal { A } _ { \mathrm { s o l } }$ be arbitrary. Since the coeficients $\xi ^ { i }$ are independent of u, the horizontal component of the Lie bracket vanishes,

$$
v _ { \varphi } \left( \xi ^ { i } \right) = 0 , \quad i = 1 , \ldots , p .\tag{D.15}
$$

For the vertical component, direct evaluation gives

$$
\begin{array} { l } { { { \displaystyle [ v , v _ { \varphi } ] = \sum _ { \alpha = 1 } ^ { q } \widetilde { \varphi } ^ { \alpha } \left( { \bf x } \right) \partial _ { u ^ { \alpha } } } , } } \\ { { { \displaystyle ~ \widetilde { \varphi } ^ { \alpha } : = \sum _ { i = 1 } ^ { p } \xi ^ { i } \partial _ { x ^ { i } } \varphi ^ { \alpha } - \sum _ { \beta = 1 } ^ { q } A _ { \beta } ^ { \alpha } \varphi ^ { \beta } } . } } \end{array}\tag{D.16}
$$

Thus the bracket is again purely vertical, and its coeficients depend only on the independent variables.

Because both v and $v _ { \varphi }$ are admitted Lie point-symmetry generators, closure of the admitted symmetry algebra under the Lie bracket implies that $[ \dot { v } , v _ { \varphi } ]$ is also admitted. Its local flow is an additive transformation

$$
\mathbf { u } \longmapsto \mathbf { u } + \varepsilon { \boldsymbol { \widetilde { \varphi } } } \left( \mathbf { x } \right) .\tag{D.17}
$$

Applying the same linearity argument used above to this admitted transformation gives

$$
\left. \mathrm { p r } ^ { ( s ) } ( v _ { \widetilde { \varphi } } ) \left[ \Delta ^ { ( s ) } \right] \right| _ { \Sigma ^ { ( s ) } } = \mathrm { L } ^ { ( s ) } [ \widetilde { \varphi } ] = \mathbf { 0 } .\tag{D.18}
$$

Hence

$$
[ v , v _ { \varphi } ] = v _ { \widetilde { \varphi } } \in \mathcal { A } _ { \mathrm { s o l } } .\tag{D.19}
$$

Since $v \in { \mathfrak { g } }$ and $v _ { \varphi } \in \mathcal { A } _ { \mathrm { s o l } }$ were arbitrary,

$$
[ \mathfrak { g } , \mathcal { A } _ { \mathrm { s o l } } ] \subseteq \mathcal { A } _ { \mathrm { s o l } } .\tag{D.20}
$$

Therefore, whenever the admitted symmetry algebra has the PFA form, $\boldsymbol { A } _ { \mathrm { s o l } }$ is an Abelian ideal of ${ \mathfrak { g } } .$

## D.3 SAMPLED TRIVIALITY AND GEOMETRIC QUOTIENT

This subsection provides the proof of Proposition 3. Let $V : = \widehat { N } _ { \mathrm { n a d d } } ^ { ( k ) }$ and

$$
T : = \left. \widehat { T } _ { \mathrm { t r i v } } ^ { ( k ) } \right. _ { \mathbf { S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } , V }\tag{D.21}
$$

for notational simplicity. Since $T$ is linear, ker $T$ is a vector subspace of $V .$ . By the definition of the sampled trivial sector in (124),

$$
\ker T = \widehat { K } _ { \mathrm { t r i v } } ^ { ( k ) } .\tag{D.22}
$$

This proves the first statement.

For the second statement, the First Isomorphism Theorem [1, Section 3.3] applied to the linear map $T$ gives

$$
V / \ker T \cong \operatorname { i m } \left( T \right) .\tag{D.23}
$$

Using the definition of the geometric quotient in (125), we therefore obtain

$$
\widehat { N } _ { \mathrm { g e o m } } ^ { ( k ) } = V / \widehat { K } _ { \mathrm { t r i v } } ^ { ( k ) } = V / \ker T \cong \operatorname { i m } \left( T \right) .\tag{D.24}
$$

Thus the quotient identifies coeficient directions that difer only by an element of the sampled trivial sector, while its image under the induced isomorphism records the corresponding sampled prolonged-characteristic information.

$$
{ \widehat { \mathcal { V } } } _ { \mathrm { n t r } } ^ { ( k ) } = V \setminus \ker T .\tag{D.25}
$$

Since $\mathbf { 0 } \in$ ker T, it follows immediately that $\mathbf { 0 } \not \in \widehat { \mathcal { V } } _ { \mathrm { n t r } } ^ { ( k ) }$ . Hence ${ \widehat { \mathcal { V } } } _ { \mathrm { n t r } } ^ { ( k ) }$ is not a vector subspace of V. Moreover, whenever this set is nonempty, any $\Theta \in \widehat { \mathcal { V } } _ { \mathrm { n t r } } ^ { ( k ) }$ also satisfies $- \Theta \in \widehat { \mathcal { V } } _ { \mathrm { n t r } } ^ { ( k ) }$ , whereas

$$
\mathbf { \Theta } \Theta + \left( - \mathbf { \Theta } \right) = \mathbf { 0 } \in \ker T .\tag{D.26}
$$

Thus the sampled nontrivial feasible set is not closed under vector addition, completing the third statement.

## D.4 FIRST-ORDER ODES AND RESIDUAL FUNCTIONAL FREEDOM

This appendix proves Proposition 5 and explains why the characteristic-based removal of solution-tangent symmetries does not eliminate the arbitrary functional freedom inherent to unrestricted first-order ODE point symmetries.

Associated with the first-order system (182) is the nonvanishing vector field

$$
X = \partial _ { x } + \sum _ { \alpha = 1 } ^ { q } F ^ { \alpha } \left( x , { \bf u } \right) \partial _ { u ^ { \alpha } }\tag{D.27}
$$

on the local total space E. Its integral curves are precisely the lifted solution curves of the ODE system. By the localflow-box theorem [3, Theorem 1.2.2], around every regular point there exist point coordinates $\left( s , r ^ { 1 } , \ldots , r ^ { q } \right)$ such that

$$
X = \partial _ { s } .\tag{D.28}
$$

The coordinates $r ^ { \alpha }$ are therefore constant along the integral curves of X and locally label the solution leaves. In these coordinates, the ODE system assumes the rectified form (183).

Consider a general local point vector field

$$
v = A \left( s , \mathbf { r } \right) \partial _ { s } + \sum _ { \alpha = 1 } ^ { q } B ^ { \alpha } \left( s , \mathbf { r } \right) \partial _ { r ^ { \alpha } } .\tag{D.29}
$$

Its first prolongation along the derivative coordinate $r _ { s } ^ { \alpha }$ has coeficient (see (34))

$$
\varphi _ { s } ^ { \alpha } = D _ { s } B ^ { \alpha } - r _ { s } ^ { \alpha } D _ { s } A .\tag{D.30}
$$

Restricting to the equation manifold $r _ { s } ^ { \alpha } = 0$ gives

$$
\varphi _ { s } ^ { \alpha } | _ { \Sigma ^ { ( 1 ) } } = \frac { \partial B ^ { \alpha } } { \partial s } .\tag{D.31}
$$

Therefore, the infinitesimal invariance condition is equivalent to

$$
{ \frac { \partial B ^ { \alpha } } { \partial s } } = 0 , \quad \alpha = 1 , \ldots , q ,\tag{D.32}
$$

while no corresponding restriction is imposed on $A \left( s , \mathbf { r } \right)$ . Hence every admitted point generator is locally of the form

$$
v = A \left( s , \mathbf { r } \right) \partial _ { s } + \sum _ { \alpha = 1 } ^ { q } B ^ { \alpha } \left( \mathbf { r } \right) \partial _ { r ^ { \alpha } } .\tag{D.33}
$$

For the rectified system, the characteristic components are

$$
Q _ { v } ^ { \alpha } = B ^ { \alpha } \left( \mathbf { r } \right) - r _ { s } ^ { \alpha } A \left( s , \mathbf { r } \right) .\tag{D.34}
$$

Thus, on $\Sigma ^ { ( 1 ) }$

$$
Q _ { v } ^ { \alpha } | _ { \Sigma ^ { ( 1 ) } } = B ^ { \alpha } \left( \mathbf { r } \right) .\tag{D.35}
$$

It follows that the arbitrary vector-field family $A \left( s , \mathbf { r } \right) \partial _ { s }$ has vanishing characteristic on the equation manifold and hence, is tangent to the solution leaves. This is precisely the sector removed by the characteristic-based nontriviality criterion $\mathcal { L } _ { \mathrm { t a i l } }$ in (158).

In contrast, whenever at least one $B ^ { \alpha } \left( \mathbf { r } \right)$ is nonzero, the characteristic is nonvanishing and the corresponding generator is nontrivial. Since the functions $B ^ { \alpha } \left( \mathbf { r } \right)$ are otherwise arbitrary, the resulting nontrivial quotient contains the infinitedimensional Lie algebra of local vector fields on the q-dimensional space of solution leaves. Therefore, removing the characteristic-zero tangent sector does not reduce the unrestricted local Lie point-symmetry structure of a regular first-order ODE system to a finite-dimensional algebra.

## D.5 PROJECTABLE FIBER-AFFINE FORM (PFA)

The proof of Proposition $^ { 6 , }$ which states that the specified classes of linear DEs admit only Lie point-symmetry generators of the PFA form in (187), is provided in this subsection. We begin from the general point-symmetry representation in (29). It is suficient to establish

$$
\partial _ { u ^ { \alpha } } \xi ^ { i } = 0 , \quad i = 1 , \dots , p , \ \alpha = 1 , \dots , q ,\tag{D.36}
$$

together with

$$
\partial _ { u ^ { \beta } } \partial _ { u ^ { \gamma } } \varphi ^ { \alpha } = 0 , \quad \alpha , \beta , \gamma = 1 , \dots , q .\tag{D.37}
$$

Indeed, (D.36) implies $\xi ^ { i } = \xi ^ { i } \left( \mathbf { x } \right)$ , while (D.37) implies that each fiber coeficient is afine in the dependent variables,

$$
\varphi ^ { \alpha } \left( \mathbf { x } , \mathbf { u } \right) = \sum _ { \beta = 1 } ^ { q } A _ { \beta } ^ { \alpha } \left( \mathbf { x } \right) u ^ { \beta } + b ^ { \alpha } \left( \mathbf { x } \right) ,\tag{D.38}
$$

which yields precisely the PFA form in (187).

Linear ODEs with $s \geq 3 .$ : For the ODE case, let x denote the single independent variable and write $u _ { r } ^ { \alpha } : = d ^ { r } u ^ { \alpha } / d x ^ { r }$ , with $u _ { 0 } ^ { \alpha } : = u ^ { \alpha }$ . A general order-s linear system may be written locally as

$$
\Delta ^ { \nu } = a _ { \alpha _ { s } } ^ { \nu } \left( x \right) u _ { s } ^ { \alpha _ { s } } + \sum _ { r = 0 } ^ { s - 1 } a _ { \alpha _ { r } } ^ { \nu } \left( x \right) u _ { r } ^ { \alpha _ { r } } - f ^ { \nu } \left( x \right) = 0 ,\tag{D.39}
$$

where summation over the repeated dependent-variable indices $\alpha _ { T }$ is understood. By the assumed nondegeneracy of the highest-order part in Section 2, the system can be solved locally for the highest derivatives and equivalently written as

$$
u _ { s } ^ { \alpha } = F ^ { \alpha } \left( x , \mathbf { u } _ { ( s - 1 ) } \right) ,\tag{D.40}
$$

where each $F ^ { \alpha }$ is afine in the lower-order jet coordinates [5, The Cauchy-Kovalevskaya Theorem].

For the general generator in (29), the one-dimensional specialization of (34) satisfies the recursion

$$
\varphi _ { r + 1 } ^ { \alpha } = D _ { x } \varphi _ { r } ^ { \alpha } - u _ { r + 1 } ^ { \alpha } D _ { x } \xi , \quad 0 \leq r < s ,\tag{D.41}
$$

with $\varphi _ { 0 } ^ { \alpha } : = \varphi ^ { \alpha }$ . Iterating (D.41) gives

$$
\varphi _ { s } ^ { \alpha } = D _ { x } ^ { s } \varphi ^ { \alpha } - \sum _ { r = 1 } ^ { s } { \binom { s } { r } } u _ { s - r + 1 } ^ { \alpha } D _ { x } ^ { r } \xi .\tag{D.42}
$$

The infinitesimal symmetry condition in (37) for the normal form (D.40) is therefore

$$
\left. \left\{ \varphi _ { s } ^ { \alpha } - \mathrm { p r } ^ { \left( s - 1 \right) } \left( v \right) \left[ F ^ { \alpha } \right] \right\} \right| _ { u _ { s } ^ { \alpha } = F ^ { \alpha } } = 0 .\tag{D.43}
$$

Because the lower jet coordinates $\mathbf { u } _ { ( s - 1 ) }$ remain functionally independent after eliminating $\mathbf { u } _ { s }$ , this identity may be split with respect to their independent monomials.

We first establish projectability. At the lowest order covered by the proposition, $s = 3$ , the recursion (D.41) produces in $\varphi _ { 3 } ^ { \alpha }$ the quadratic term

$$
- 3 u _ { 2 } ^ { \alpha } u _ { 2 } ^ { \beta } \partial _ { u ^ { \beta } } \xi .\tag{D.44}
$$

No term with the same quadratic dependence on $\mathbf { u } _ { 2 }$ is generated by $\mathrm { p r } ^ { ( 2 ) } \left( v \right) \left[ F ^ { \alpha } \right]$ or by the on-shell substitution for $\mathbf { u } _ { 3 }$ since $F ^ { \alpha }$ is afine in the lower jet coordinates. Splitting (D.43) with respect to the products $u _ { 2 } ^ { \alpha } u _ { 2 } ^ { \beta }$ therefore yields

$$
\partial _ { u ^ { \beta } } \xi = 0 , \qquad \beta = 1 , \ldots , q .\tag{D.45}
$$

For $s > 3 ,$ the same recursive structure produces the terms

$$
- \binom { s } { 2 } u _ { s - 1 } ^ { \alpha } u _ { 2 } ^ { \beta } \partial _ { u ^ { \beta } } \xi - s u _ { 2 } ^ { \alpha } u _ { s - 1 } ^ { \beta } \partial _ { u ^ { \beta } } \xi\tag{D.46}
$$

inside $\varphi _ { s } ^ { \alpha }$ . Again, neither the lower-order prolonged contribution in (D.43) nor elimination of the linearly occurring highest derivatives generates quadratic monomials with this derivative-order structure. For $\beta \neq \alpha$ , the coeficients of $u _ { s - 1 } ^ { \alpha } u _ { 2 } ^ { \beta }$ and $u _ { 2 } ^ { \alpha } u _ { s - 1 } ^ { \beta } \mathrm { a r e } - \binom { s } { 2 } \partial _ { u ^ { \beta } } \xi$ and $- s \partial _ { u } \beta \xi ,$ , respectively, whereas for $\beta = \alpha$ the two displayed contributions combine with the nonzero coeficient $- \left[ { \binom { s } { 2 } } + s \right] \partial _ { u ^ { \alpha } } \xi$ . Hence,

$$
\partial _ { u ^ { \beta } } \xi = 0 \quad \Longrightarrow \quad \xi = \xi \left( x \right)\tag{D.47}
$$

for every $s \geq 3 .$

With projectability established, all derivatives of ξ with respect to the dependent variables vanish. The remaining nonlinear dependence on u that is relevant to the highest prolongation therefore comes from $D _ { x } ^ { s } \varphi ^ { \alpha }$ . In particular, it contains

$$
s u _ { 1 } ^ { \beta } u _ { s - 1 } ^ { \gamma } \partial _ { u ^ { \beta } } \partial _ { u ^ { \gamma } } \varphi ^ { \alpha } .\tag{D.48}
$$

For $s = 3 .$ this is precisely the term $3 u _ { 1 } ^ { \beta } u _ { 2 } ^ { \gamma } \partial _ { u ^ { \beta } } \partial _ { u ^ { \gamma } } \varphi ^ { \alpha }$ . After (D.47) has been imposed, no other contribution to (D.43) has the same $u _ { 1 } ^ { \beta } u _ { s - 1 } ^ { \gamma }$ dependence. Splitting with respect to these independent jet monomials therefore gives

$$
\partial _ { u ^ { \beta } } \partial _ { u ^ { \gamma } } \varphi ^ { \alpha } = 0 , \quad \alpha , \beta , \gamma = 1 , \dots , q .\tag{D.49}
$$

Consequently,

$$
\xi = \xi \left( x \right) , \quad \varphi ^ { \alpha } = A _ { \beta } ^ { \alpha } \left( x \right) u ^ { \beta } + b ^ { \alpha } \left( x \right) ,\tag{D.50}
$$

which proves the PFA characterization for linear ODE systems of genuine order $s \geq 3$

The restriction $s \geq 3$ is essential: the unrestricted PFA conclusion does not hold in general for linear ODEs of first or second order. Rather than reproducing the corresponding lower-order determining systems, we illustrate these two exceptions by explicit counterexamples below.

For $s = 1$ , consider the scalar linear equation

$$
\Delta ^ { ( 1 ) } : = u _ { 1 } = 0\tag{D.51}
$$

and the point vector field

$$
v = u \partial _ { x } + u ^ { 2 } \partial _ { u } .\tag{D.52}
$$

Its first prolonged coeficient is

$$
\varphi _ { 1 } = D _ { x } \left( u ^ { 2 } \right) - u _ { 1 } D _ { x } \left( u \right) = 2 u u _ { 1 } - u _ { 1 } ^ { 2 } ,\tag{D.53}
$$

and therefore

$$
\left. \mathrm { p r } ^ { ( 1 ) } \left( v \right) \left[ \Delta ^ { ( 1 ) } \right] \right| _ { \Sigma ^ { ( 1 ) } } = \left. \varphi _ { 1 } \right| _ { u _ { 1 } = 0 } = 0 .\tag{D.54}
$$

Hence v is an admitted point-symmetry generator, although

$$
\partial _ { u } \xi = 1 \neq 0 , \qquad \partial _ { u } ^ { 2 } \varphi = 2 \neq 0 ,\tag{D.55}
$$

so it is neither projectable nor fiber-afine.

For $s = 2$ , consider similarly

$$
\Delta ^ { ( 2 ) } : = u _ { 2 } = 0\tag{D.56}
$$

and

$$
v = x u \partial _ { x } + u ^ { 2 } \partial _ { u } .\tag{D.57}
$$

The first two prolonged coeficients are

$$
\varphi _ { 1 } = u u _ { 1 } - x u _ { 1 } ^ { 2 }\tag{D.58}
$$

and

$$
\varphi _ { 2 } = D _ { x } \varphi _ { 1 } - u _ { 2 } D _ { x } \left( x u \right) = - 3 x u _ { 1 } u _ { 2 } .\tag{D.59}
$$

Consequently,

$$
\left. \mathrm { p r } ^ { ( 2 ) } \left( v \right) \left[ \Delta ^ { ( 2 ) } \right] \right| _ { \Sigma ^ { ( 2 ) } } = \left. \varphi _ { 2 } \right| _ { u _ { 2 } = 0 } = 0 ,\tag{D.60}
$$

so v is again an admitted point-symmetry generator. However,

$$
\partial _ { u } \xi = x , \qquad \partial _ { u } ^ { 2 } \varphi = 2 ,\tag{D.61}
$$

and hence the generator is not of PFA form. These examples show explicitly why the unrestricted ODE statement in Proposition 6 begins at genuine order $s = 3$

Linear PDEs with $s \geq 2 .$ Let $\mathbf { x } = \left( x ^ { 1 } , \ldots , x ^ { p } \right)$ , with $p \geq 2 .$ A linear PDE system of genuine order s may be written locally as

$$
\begin{array} { c } { { \displaystyle \Delta ^ { \nu } = \sum _ { \alpha = 1 } ^ { q } \sum _ { | I | = s } a _ { \alpha , I } ^ { \nu } \left( { \bf x } \right) u _ { I } ^ { \alpha } + } } \\ { { \displaystyle \sum _ { \alpha = 1 } ^ { q } \sum _ { | I | < s } a _ { \alpha , I } ^ { \nu } \left( { \bf x } \right) u _ { I } ^ { \alpha } - f ^ { \nu } \left( { \bf x } \right) = 0 . } } \end{array}\tag{D.62}
$$

As in the ODE case, the assumed nondegeneracy of the highest-order part permits the appropriate dependent order-s derivatives to be eliminated locally, while the remaining jet coordinates provide functionally independent coordinates on $\Sigma ^ { ( s ) }$

We first consider the lowest order covered by the proposition, $s = 2 .$ . From the prolongation recursion in (34), the first-order prolonged coeficients are

$$
\varphi _ { i } ^ { \alpha } = \partial _ { x ^ { i } } \varphi ^ { \alpha } + u _ { i } ^ { \beta } \partial _ { u ^ { \beta } } \varphi ^ { \alpha } - u _ { j } ^ { \alpha } \partial _ { x ^ { i } } \xi ^ { j } - u _ { j } ^ { \alpha } u _ { i } ^ { \beta } \partial _ { u ^ { \beta } } \xi ^ { j } .\tag{D.63}
$$

Applying the recursion once more gives

$$
\begin{array} { c } { { \varphi _ { i j } ^ { \alpha } = \cdots + u _ { i } ^ { \beta } u _ { j } ^ { \gamma } \partial _ { u ^ { \beta } } \partial _ { u ^ { \gamma } } \varphi ^ { \alpha } } } \\ { { { } } } \\ { { - u _ { k , j } ^ { \alpha } u _ { i } ^ { \beta } \partial _ { u ^ { \beta } } \xi ^ { k } - u _ { i , k } ^ { \alpha } u _ { j } ^ { \beta } \partial _ { u ^ { \beta } } \xi ^ { k } - u _ { k } ^ { \alpha } u _ { i , j } ^ { \beta } \partial _ { u ^ { \beta } } \xi ^ { k } + \cdots , } } \end{array}\tag{D.64}
$$

where the omitted terms are not required for the coeficient splittings below.

Consider first the terms containing one first-order and one second-order jet coordinate. Since (D.62) is linear, the lower-order prolonged contributions do not generate terms of this type via (D.63). Restriction to $\Sigma ^ { ( 2 ) }$ eliminates only the dependent linear combinations of the second-order derivatives as in the previous case of an ODE in the normal form. Consequently, after this elimination, the coeficients of the remaining independent first-order–second-order products in (D.64) are linear combinations of the principal coeficients $a _ { \alpha , I } ^ { \nu }$ multiplying $\partial _ { u ^ { \beta } } \xi ^ { k }$ . By the assumed nondegeneracy of the highest-order part, these coeficients can vanish identically only if

$$
\partial _ { u ^ { \beta } } \xi ^ { k } = 0 , \quad k = 1 , \dots , p , \ \beta = 1 , \dots , q .\tag{D.65}
$$

Thus the base coeficients are projectable,

$$
\xi ^ { k } = \xi ^ { k } \left( { \bf x } \right) .\tag{D.66}
$$

Once (D.65) is imposed, all terms involving derivatives of $\xi ^ { k }$ with respect to the dependent variables disappear. The quadratic first-order contribution

$$
u _ { i } ^ { \beta } u _ { j } ^ { \gamma } \partial _ { u ^ { \beta } } \partial _ { u ^ { \gamma } } \varphi ^ { \alpha }\tag{D.67}
$$

therefore remains among the structurally highest nonlinear terms in (D.64). Neither the lower-order prolonged terms nor the on-shell elimination of second-order derivatives can generate compensating quadratic products of the first-order jets, because the PDE system itself is linear. Splitting the determining equations with respect to the independent products $u _ { i } ^ { \beta } u _ { j } ^ { \gamma }$ and again using nondegeneracy of the principal part, yields

$$
\partial _ { u ^ { \beta } } \partial _ { u ^ { \gamma } } \varphi ^ { \alpha } = 0 , \quad \alpha , \beta , \gamma = 1 , \dots , q .\tag{D.68}
$$

Hence every admitted generator of a second-order linear PDE system satisfying the hypotheses of Proposition 6 is projectable and fiber-afine.

The same structure persists at every order $s > 2$ . Indeed, repeated use of the prolongation recursion gives, for $| I | = s ,$

$$
\varphi _ { I } ^ { \alpha } = D _ { I } \varphi ^ { \alpha } - \sum _ { k = 1 } ^ { p } \sum _ { \mathbf { 0 } < J \leq I } \binom { I } { J } \left( D _ { J } \xi ^ { k } \right) u _ { I - J , k } ^ { \alpha } .\tag{D.69}
$$

Here $J \le I$ denotes componentwise multi-index ordering, and

$$
{ \binom { I } { J } } : = { \frac { I ! } { J ! \left( I - J \right) ! } } = \prod _ { i = 1 } ^ { p } { \binom { I _ { i } } { J _ { i } } } ,\tag{D.70}
$$

is the corresponding multi-index binomial coeficient arising from the repeated Leibniz rule for $D _ { x ^ { i } }$ and $p > 1$ . The terms with $| J | = 1$ contain products of an order-s derivative with a first-order derivative multiplied by $\partial _ { u ^ { \beta } } \xi ^ { k }$ . Since the order-s shell constraints remain linear in the highest-order jets, elimination of the dependent highest-order derivatives leaves independent products of this type. Splitting their coeficients therefore gives (D.65) for every $s > 2$

After projectability has been established, $D _ { I } \varphi ^ { \alpha }$ contains, with nonzero combinatorial coeficients, products of the form

$$
u _ { I - e _ { i } } ^ { \beta } u _ { i } ^ { \gamma } \partial _ { u ^ { \beta } } \partial _ { u ^ { \gamma } } \varphi ^ { \alpha } , \quad | I | = s ,\tag{D.71}
$$

where $e _ { i }$ denotes the unit multi-index in the $x ^ { i }$ direction. These products involve an order- $\cdot ( s - 1 )$ jet and a first-order jet and are unafected by the linear elimination of the order-s on-shell variables. No lower-order term in the linear determining system can cancel these nonlinear jet products. Splitting with respect to them therefore yields (D.68). Consequently,

$$
\xi ^ { i } = \xi ^ { i } \left( { \bf x } \right) , \quad \varphi ^ { \alpha } = A _ { \beta } ^ { \alpha } \left( { \bf x } \right) u ^ { \beta } + b ^ { \alpha } \left( { \bf x } \right) ,\tag{D.72}
$$

which proves the PFA characterization for linear PDE systems of genuine order $s \geq 2 .$

The corresponding first-order linear PDE case is structurally diferent and does not admit the unrestricted conclusion above. The PFA subclass at $s = 1$ , together with its classical and operator-theoretic admissibility conditions, is treated separately in D.6.

## D.6 PFA SYMMETRIES OF A FIRST-ORDER LINEAR PDE

This subsection provides the proofs of Propositions 7 and 8. For the scalar first-order linear PDE in Proposition 7,

$$
\Delta ^ { \left( 1 \right) } \left[ u \right] = a ^ { i } \left( { \bf x } \right) u _ { i } + c \left( { \bf x } \right) u - f \left( { \bf x } \right) = 0 ,\tag{D.73}
$$

consider the prescribed PFA generator

$$
v = \xi ^ { i } \left( \mathbf { x } \right) \partial _ { x ^ { i } } + \left[ A \left( \mathbf { x } \right) u + b \left( \mathbf { x } \right) \right] \partial _ { u } .\tag{D.74}
$$

The first prolonged coeficient obtained from (34) is

$$
\varphi _ { i } = \partial _ { x ^ { i } } A u + A u _ { i } + \partial _ { x ^ { i } } b - u _ { j } \partial _ { x ^ { i } } \xi ^ { j } .\tag{D.75}
$$

For the present calculation, it is convenient to use the multiplier form of the infinitesimal symmetry condition in (38). Since the PDE is scalar, there exists locally a smooth multiplier λ such that

$$
\mathrm { p r } ^ { ( 1 ) } ( v ) \left[ \Delta ^ { ( 1 ) } \right] = \lambda \Delta ^ { ( 1 ) } .\tag{D.76}
$$

Substituting (D.75) into the left-hand side gives

$$
\begin{array} { r l } & { \mathrm { p r } ^ { \left( 1 \right) } \left( v \right) \left[ \Delta ^ { \left( 1 \right) } \right] = \left( \xi ^ { j } \partial _ { x ^ { j } } a ^ { i } - a ^ { j } \partial _ { x ^ { j } } \xi ^ { i } + A a ^ { i } \right) u _ { i } } \\ & { \quad \quad + \left( \vphantom { A ^ { 2 } } { a } \cdot \nabla A + \xi \cdot \nabla c + A c \right) u } \\ & { \quad \quad \quad + L [ b ] - \xi \cdot \nabla f . } \end{array}\tag{D.77}
$$

The left-hand side and $\Delta ^ { ( 1 ) }$ are afine in the jet coordinates $u _ { i }$ and u. Hence the identity (D.76), together with regularity of the genuine first-order PDE, forces the multiplier to be independent of these jet coordinates; we may therefore write $\lambda = \lambda \left( \mathbf { x } \right)$

Splitting (D.76) first with respect to the independent coordinates $u _ { i }$ yields

$$
[ \pmb { \xi } , \pmb { a } ] ^ { i } + A a ^ { i } = \lambda a ^ { i } .\tag{D.78}
$$

Introducing $\mu : = \lambda - A$ we obtain

$$
[ \pmb { \xi } , \pmb { a } ] = \mu \pmb { a } .\tag{D.79}
$$

Geometrically, this condition states that the rank-one characteristic distribution generated by a is preserved by the flow of $\pmb { \xi } = \xi ^ { i } ( \mathbf { x } ) \partial _ { x ^ { i } }$ up to a local rescaling.

Splitting next with respect to u gives

$$
\pmb { a } \cdot \nabla A + \pmb { \xi } \cdot \nabla c + A c = \lambda c ,\tag{D.80}
$$

and hence

$$
\pmb { a } \cdot \nabla A + \pmb { \xi } \cdot \nabla c = \mu c .\tag{D.81}
$$

Finally, the constant term gives

$$
L [ b ] - \pmb { \xi } \cdot \nabla f = - \lambda f ,\tag{D.82}
$$

or, using $\lambda = A + \mu ,$

$$
\pmb { \xi } \cdot \nabla f - L [ b ] - A f = \mu f .\tag{D.83}
$$

These are precisely the three conditions in (188).

Conversely, suppose that the conditions in (188) hold for some smooth $\mu \left( \mathbf { x } \right)$ . Setting $\lambda : = A + \mu$ in the preceding expansion gives

$$
\mathrm { p r } ^ { ( 1 ) } \left( v \right) \left[ \Delta ^ { ( 1 ) } \right] = \left( A + \mu \right) \Delta ^ { ( 1 ) } .\tag{D.84}
$$

Hence the multiplier condition (38) is satisfied, proving Proposition 7.

We now prove Proposition 8. Let

$$
L = \pmb { a } \cdot \nabla + c , \quad R = A - \pmb { \xi } \cdot \nabla .\tag{D.85}
$$

For an arbitrary smooth test function h, direct expansion of the commutator gives

$$
\left[ L , R \right] h = \left[ \pmb { \xi } , \pmb { a } \right] \cdot \nabla h + \left( \pmb { a } \cdot \nabla A + \pmb { \xi } \cdot \nabla c \right) h .\tag{D.86}
$$

Therefore, $[ L , R ] = \mu L$ holds if and only if

$$
[ \pmb { \xi } , \pmb { a } ] = \mu \pmb { a } , \quad \pmb { a } \cdot \nabla A + \pmb { \xi } \cdot \nabla c = \mu c ,\tag{D.87}
$$

which are exactly the first two conditions in (188). Moreover,

$$
L [ b ] + \left( R + \mu \right) f = L [ b ] + A f - \pmb { \xi } \cdot \nabla f + \mu f .\tag{D.88}
$$

Thus

$$
L [ b ] + ( R + \mu ) \ f = 0\tag{D.89}
$$

is equivalent to the third condition in (188). Hence the two operator identities in (189) are equivalent to the complete PFA symmetry criterion of Proposition 7, proving Proposition 8.

## E FLOWCHARTS FOR ALGORITHMS

![](images/9e8dade3b1e697cd1b47c7ae42d7fe75544f912db1bdd8a790c36c976e1d54d5.jpg)  
Figure E.1: Flowchart of Cartan’s iterative procedure for a linear Pfafian system, including torsion compatibility, prolongation, involutivity, and the resulting diferential freedom. The blue arrow indicates relabeling of the Pfafian one-form systems for the next iteration. The prolongation index l starts at 0.

![](images/b1cdb9288d231d1f179b14ca1f8dca4f6a3a3eeeb0acb21af2cad69d0039b549.jpg)  
Figure E.2: Flowchart of the sequential symmetry-discovery procedure. Starting from the DE system, its sampled datasets, and the prescribed hyperparameters, the algorithm constructs the represented determining and nonadditive admissible spaces, applies the sequential novelty restriction, initializes a candidate generator within the remaining search subspace, and optimizes the regularized nontriviality objective. Accepted generators are appended successively to the recovered intrinsic algebra and used to construct the novelty restriction for the next discovery step. Blue-bordered halt states indicate clean exhaustion of the numerically resolvable represented symmetry directions, whereas red-bordered halt states indicate recovery failure despite the presence of a represented candidate direction. The initial halt associated with a DE system outside the admissible scope of the framework is treated separately in the applicability flowchart of Section 6.

![](images/bd031c4fabc2744dad78e09381372baa2f8ce3850777b3a4eff64e55ee6dcade.jpg)  
Figure E.3: Applicability and admissible-generator realization of the proposed framework. Starting from the DE system $\Delta ^ { ( s ) } = \mathbf { 0 }$ and the associated on- and of-shell datasets, the flowchart first distinguishes ODEs from PDEs, then branches according to diferential order and linearity. First-order ODEs are excluded from the present scope because arbitrary nontrivial functional freedom remains after removal of the characteristic-zero solution-tangent sector. For regular, determined ODE systems with $s > 1$ , the local point-symmetry algebra is finite-dimensional; linear ODEs with $s \geq 3$ use the projectable fiber-afine (PFA) realization, whereas the remaining admissible ODE classes use the regular generator realization. For PDEs, linear systems with $s \geq 2$ use the PFA realization, while nonlinear PDEs and first-order linear PDEs use the regular realization in the present implementation. All retained branches are subjected to the RSP test. A stable represented geometric-rank plateau over the prescribed refinement window permits entry into sequential recovery, whereas failure to stabilize places the DE outside the completeness-certified scope of the current framework. Here, “Admissible” refers to admissibility within the stabilized finite represented and sampled setting; it does not by itself establish exact formal closure or exclude unresolved arbitrary functional freedom in the underlying determining system.

## F SUPPLEMENTARY NUMERICAL VISUALIZATIONS

![](images/8348a0c0947ccdf47cb8b62b626e1f5001593d87107f90924eedb54a63219157.jpg)

![](images/6fa4cd5f46ec539e0b697904328789de1f93b978cb54c2e570e5d992b089c457.jpg)

![](images/00c2a6ccd013a9b485135bd964e65031f7dad8700801c6c5b8a06f7674474f52.jpg)

![](images/741cfcb65ea4ae5c9650e47256dcdd54b22bd997d93f5d88c693017955942b05.jpg)

![](images/422e697fd1dad94f40ef43a134e51f7a6115a52b46833a0af22e1f32c3af9432.jpg)  
(a) $v _ { 4 } , \varepsilon _ { \mathrm { { m a x } } } = 0 . 1 5$

![](images/c39d60086afcbcf72540355c87d90ee512dabba8ce390c000343a8de14e561d0.jpg)

![](images/1f1529205700bf4fc017743c130297ff66d1e9fd4ff8a26cb0d67f9e42041580.jpg)

![](images/d8640b569e3f61caf266ace2cb8322fc35b538766b62b4410981fd9bb65a692e.jpg)

![](images/69092f4c0e7362d9dafdaa552aca2efd96324868d1122097b54bdc7c6935368c.jpg)

![](images/6d7fefd8528916228c502df332258678006224bee0e3f2cf8ad78ea0c78c9748.jpg)

![](images/8dc90ab0f5d013161645418a873ad4d4fc670621a342014c08d3c51f30adf35f.jpg)

(b) v<sub>7</sub>, ε<sub>max</sub> = 0.05  
![](images/3ad35ac497a22b017623f7be7b10099ac208892d224e4aed1c7fc93b1e09ce18.jpg)

![](images/52832f061ed9485d91dd2c2d5a2f4292671b76fbb35111b2c637dc4d731b9cc5.jpg)

![](images/f8a29ddcbad5924455d939317abd4f63bf9996ef85da3fc90b4af34e929ec21a.jpg)

![](images/25f736f11e1f61478db2d9dc3477786c098a3694a59543748ee47bdf87aa17f2.jpg)

![](images/3b1adc36aa6d807607ce264250f42d94e3aba336abd87655ac8eef70fde098d4.jpg)

![](images/d4c90fe47f70e6967007c81bd7b51402915d849225d0735a656f66f246057285.jpg)  
(c) $v _ { 1 0 } , \varepsilon _ { \mathrm { m a x } } = 0 . 0 3$

![](images/a732a81fcd0b08cc959abe1748b56b9434cad36d5b0c84361916f9831af3eadc.jpg)  
Figure F.1: Representative integral curves of three recovered generators selected from the low-, intermediate-, and high-speed-tail regimes reported in Table 1. The trajectories are shown through their projections onto the x– $\not { p } - \mathcal { Y } , \ : x - t , \ : y - t , \ : x - u , \ : y - u$ , and t–u coordinate planes, with the black markers indicating the initial points. The selected generators are (a) v<sub>4</sub>, with $\rho _ { 9 9 } = 1 . 0 0 2 $ and $\varepsilon _ { \mathrm { m a x } } = 0 . 1 5 ;$ (b) v<sub>7</sub>, with $\rho _ { 9 9 } = 1 . 6 5 4$ and $\varepsilon _ { \mathrm { m a x } } = 0 . 0 5 ;$ and $( \mathrm { c } ) \ v _ { 1 0 } .$ , with $\rho _ { 9 9 } = 1 . 9 9 3 $ and $\varepsilon _ { \mathrm { m a x } } = 0 . 0 3$ . The integration horizons are chosen separately so that the displayed trajectories remain within, or suficiently close to, the region in which the corresponding locally learned vector fields remain reliable; the behavior outside the training support is examined separately in Approximation of a Known Symmetry Generator and Locality of the Representation.

![](images/9a2c64a50b6ca8840290f07e239be7daf12884b2d9488b8e1c0d22e906be8089.jpg)  
Figure F.2: Integral-curve comparison between the analytical inversion-in-x generator $v _ { \mathrm { i n v } , x }$ and its constant-span reconstruction $\widetilde { v } _ { \mathrm { i n v } , x } ,$ integrated from the same initial points up $\mathrm { { t o } \varepsilon _ { \mathrm { { m a x } } } = 0 . 1 }$ using 100 steps.

![](images/40be5495b435abee87bf3e983c32113efdf0ca2f13104cc9e0d2947cdb14e770.jpg)

![](images/917928c4fdaac7abf9f153d69c07c8673ff9188bb2ebf14157c24939a74876e5.jpg)

![](images/5108eb2fac10885b243ad37b9cfb6724767ac3325d3d4058c2b04170baac4877.jpg)

![](images/785d04cc09557df1e5648a586384cd4fa891d1239baf06693ad1362d1cd35e43.jpg)

![](images/30f2b637302edee70d7ca6f507e438d898fcb9978bebf97d49d42cee95440366.jpg)

![](images/a89daadce93596080e8df1d128cf39a61cb5ebf0ccf315c588147d9f6df5c3d4.jpg)  
Figure F.3: Integral-curve comparison between the analytical inversion-in-x generator $v _ { \mathrm { i n v } , x }$ and its constant-span reconstruction $\widetilde { v } _ { \mathrm { i n v } , x }$ after shifting the initial x interval to [0.8, 0.86], with the flows integrated up to $\varepsilon _ { \operatorname* { m a x } } = 0 . 0 4$

## G TABLE OF SYMBOLS

The table lists the principal manuscript-wide meaning of recurring notation. When a symbol is locally overloaded, only its principal role is recorded; section-specific uses are defined at their point of introduction.

<table><tr><td>Symbol</td><td>Principal meaning</td><td>Context</td></tr><tr><td colspan="3">Jet-space and differential-equation geometry</td></tr><tr><td>x</td><td>Base space of the independent variables.</td><td>Jet geometry</td></tr><tr><td> $E , \pi : E \to \mathbf { X }$ </td><td>Total space and fiber-bundle projection over the base space.</td><td>Jet geometry</td></tr><tr><td> $\mathbf { x } , \mathbf { u }$ </td><td>Independent- and dependent-variable coordinate vectors, respectively.</td><td>Jet geometry</td></tr><tr><td> $p , \ q , \ m , \ s$ </td><td>Numbers of independent variables, dependent variables, and constituent differential equations, and the differential order of the DE system,</td><td>DE system</td></tr><tr><td> $J ^ { ( s ) } \left( \pi \right)$ </td><td>Order-s jet bundle of local sections of  $\overline { { \pi } } .$ </td><td>Jet geometry</td></tr><tr><td> $z ^ { ( s ) } , \mathbf { u } _ { ( s ) }$ </td><td>Generic order-s jet and the collection of dependent-variable coordinates and derivatives through order s.</td><td>Jet geometry</td></tr><tr><td> $\pi _ { 0 } ^ { s } , ~ \pi _ { r } ^ { s } , ~ \pi ^ { s }$ </td><td>Canonical projections from  $J ^ { ( s ) } \left( \pi \right)$  onto  $E , J ^ { ( r ) } \left( \pi \right)$  , and  $\mathbf { X } ,$  respectively.</td><td>Jet geometry</td></tr><tr><td> $\Delta ^ { ( s ) } , \Sigma ^ { ( s ) }$ </td><td>Vector-valued differential function defining the DE system and its regular DE system zero locus (equation manifold).</td><td></td></tr><tr><td> $i _ { f } , \ j ^ { s } i _ { f }$ </td><td>Local solution section and its order-s jet prolongation.</td><td>Solution geometry</td></tr><tr><td> $\Gamma _ { f } , \Gamma _ { f } ^ { ( s ) }$ </td><td>Solution graph in E and its order-s prolonged graph in jet space.</td><td>Solution geometry</td></tr><tr><td colspan="3">Lie symmetries and intrinsic symmetry structure</td></tr><tr><td>G, g</td><td>Lie group of local point transformations and its Lie algebra of infinitesimal Lie symmetry</td><td></td></tr><tr><td> ${ \mathfrak { X } } ( E )$ </td><td>generators. Space of smooth vector fields on the total space  $E .$ </td><td>Lie symmetry</td></tr><tr><td>U</td><td>Lie point-symmetry generator represented as a vector field on E.</td><td>Lie symmetry</td></tr><tr><td> $\xi , \varphi$ </td><td>Base (horizontal) and fiber (vertical) coefficient vectors of a point generator.</td><td>Generator representation</td></tr><tr><td> $\mathrm { p r } ^ { ( s ) } \left( v \right)$ </td><td>Order-s prolongation of the point generator v to  $J ^ { ( s ) } \left( \pi \right)$ </td><td>Prolongation</td></tr><tr><td> $D _ { x ^ { i } } , D _ { J }$ </td><td>Total derivative with respect  $\tan x ^ { i }$  and its multi-index iteration,</td><td>Prolongation</td></tr><tr><td> $\mathbf { Q } _ { v } , \mathbf { E } _ { v } ^ { ( s - 1 ) }$ </td><td>Characteristic vector of v and its prolonged characteristic information through order  $s - 1 .$ </td><td>Nontriviality</td></tr><tr><td> ${ \mathfrak { g } } _ { \mathrm { i n t } } , \ { \mathfrak { g } } _ { \mathrm { r e c } }$ </td><td>Target intrinsic symmetry algebra and the span of the generators accepted by the numerical recovery procedure, respectively.</td><td>Intrinsic symmetry</td></tr><tr><td> $\scriptstyle A _ { \mathrm { s o l } }$ </td><td>Additive solution subalgebra; under the stated projectable fiber-affine conditions, an Abelian ideal of the Lie symmetry algebra.</td><td>Linear DEs</td></tr><tr><td> $c _ { i j } ^ { k }$ </td><td>Constant structure coefficients in the recovered-basis Lie-bracket representation.</td><td>Lie algebra</td></tr><tr><td> $\mathrm { F l } _ { v } ^ { z } \left( \epsilon \right)$ </td><td>Local flow generated by v through z, with flow parameter €.</td><td>Finite transformations</td></tr><tr><td colspan="3">Sampled operators and represented admissible spaces</td></tr><tr><td> $\mathbf { S } ^ { \mathrm { o n } } , \mathbf { S } ^ { \mathrm { o f f } }$ </td><td>Finite on-shell jet dataset and auxiliary off-shell jet dataset.</td><td>Sampling</td></tr><tr><td> ${ \bf S } _ { \mathrm { a n c } } ^ { \mathrm { o n } } , { \bf S } _ { \mathrm { a n c } } ^ { \mathrm { o f f } }$ </td><td>Fixed on-shell and off-shell anchor datasets used to construct sampled operators and global novelty restrictions.</td><td>Sampling</td></tr><tr><td> $\mathcal { D } _ { \Delta } , \widehat { \mathcal { D } } _ { \Delta } , \widehat { \mathcal { D } } _ { \Delta } ^ { ( k ) }$ </td><td>Pointwise determining operator, its stacked sampled realization, and its represented sampled realization induced by  $D _ { ( k ) }$ </td><td>Sampled operators</td></tr><tr><td> $\Psi ^ { ( k ) } , \Theta , P$ </td><td>Fixed-dictionary linear realization map, its coefficient vector, and the dimension of the global coefficient space.</td><td>Representation</td></tr><tr><td> ${ \widehat { M } } _ { \Delta } ^ { ( k ) }$ </td><td>Fixed matrix representation (Jacobian) of the represented sampled determining residual with respect to Θ.</td><td>Sampled operators</td></tr><tr><td> $\widehat { N } _ { \Delta } ^ { ( k ) }$ </td><td> $\left( \widehat { M } _ { \Delta } ^ { ( k ) } \right)$  Represented sampled determining nullspace ker</td><td>Admissibility</td></tr><tr><td> $\mathcal { T } _ { \mathrm { n a d d } } , ~ \widehat { \mathcal { T } } _ { \mathrm { n a d d } } ^ { ( k ) }$ </td><td>Nonadditive-sector test and its sampled represented realization.</td><td>Additive-sector exclusion</td></tr><tr><td> $\widehat { A } _ { \mathrm { s o l } } ^ { ( k ) }$ </td><td> $\widehat { N } _ { \Delta } ^ { ( k ) }$  Sampled represented additive solution sector within</td><td>Additive-sector exclusion</td></tr><tr><td> $\widehat { N } _ { \mathrm { n a d d } } ^ { ( k ) }$ </td><td>Additive-free represented admissible coefficient subspace.</td><td>Admissibility</td></tr><tr><td> $\widehat { T } _ { \mathrm { t r i v } } ^ { ( k ) }$ </td><td>Sampled represented prolonged-characteristic operator used to identify sampled-trivial directions.</td><td>Nontriviality</td></tr><tr><td> $\widehat { K } _ { \mathrm { t r i v } } ^ { ( k ) } , \ \widehat { N } _ { \mathrm { g e o m } } ^ { ( k ) }$ </td><td>Sampled-trivial kernel and the corresponding sampled geometric quotient Nontriviality</td><td></td></tr><tr><td> $\widehat { \mathrm { B } } _ { E , i } ^ { ( k ) } , ~ \widehat { N } _ { \mathrm { n o v } , i } ^ { ( k ) }$ </td><td>modulo sampled-trivial directions. Sampled represented novelty operator and the novelty-restricted</td><td>Sequential discovery</td></tr><tr><td> $\widehat { N } _ { \mathrm { a d m } , i } ^ { ( k ) }$ </td><td>admissible search subspace at discovery step i. Represented admissible coefficient space used at refinement level ¿ in the RSP</td><td></td></tr><tr><td colspan="3">RSP test. Dictionary, stabilization, and numerical diagnostics</td></tr><tr><td> $\Omega _ { \mathbf { X } }$ </td><td>Base-coordinate sampling domain; superscripted variants denote local</td><td>Sampling / locality</td></tr><tr><td>X</td><td>patches in the proposed patchwise extension. Input-coordinate vector supplied to the fixed generator dictionary.</td><td>Fixed dictionary</td></tr><tr><td> $B _ { \nu _ { i } } ^ { ( r ) } \left( \chi ^ { i } \right)$ </td><td>Degree-r univariate B-spline basis function associated with dictionary coordinate  $\chi ^ { i }$ </td><td>Fixed dictionary</td></tr><tr><td> $D _ { ( k ) }$ </td><td>Fixed spline dictionary containing tensor-product interactions through</td><td>Fixed dictionary</td></tr><tr><td> $k , r , o , K _ { o }$ </td><td>maximum order k. Maximum interaction order, B-spline degree, core-grid resolution, and number of univariate B-spline functions per input coordinate, with</td><td>Fixed dictionary</td></tr><tr><td> $n _ { ( k ) } ^ { D }$ </td><td> $K _ { o } = o + r .$  Number of scalar features listed in the dictionary  $D _ { ( k ) } .$ </td><td>Fixed dictionary</td></tr><tr><td> $d _ { o _ { i } }$ </td><td>Represented geometric rank at grid-refinement level  $o _ { i } .$ </td><td>RSP</td></tr><tr><td> $\mathcal { R } _ { \mathrm { n t r } , l } \left( v \right) , \mathcal { R } _ { \mathrm { t a i l } } ^ { \rho } \left( v \right)$ </td><td>Relative prolonged-characteristic response at sample l and its empirical</td><td>Nontriviality</td></tr><tr><td> $\mathcal { L } _ { \mathrm { n t r } } ^ { \mathrm { r e g } }$ </td><td>lower-tail ρ-quantile. Regularized nontriviality objective used during sequential recovery.</td><td>Optimization</td></tr><tr><td> $\rho _ { 9 9 } \left( v \right)$ </td><td>Speed-tail ratio defined by the empirical 99th-to-50th percentile ratio of sampled pointwise speeds.</td><td>Numerical diagnostics</td></tr><tr><td> $\mathrm { R M S } _ { E } \left( v \right)$ </td><td>Off-shell ambient-response magnitude used to assess visibility of a represented vector field.</td><td>Sequential recovery</td></tr><tr><td>Trank, Tvis,  $\tau _ { \mathrm { s p d } } ,$  Tsym, Tntr</td><td>Principal numerical thresholds for rank determination, ambient visibility, speed-tail concentration, symmetry acceptance, and nontriviality,</td><td>Numerical implementation</td></tr><tr><td colspan="3">Exterior differential systems and formal integrability</td></tr><tr><td> $\mathcal { C } ^ { ( s ) } , \mathcal { T } _ { \mathrm { d i f f } } ^ { ( s ) }$ </td><td>Order-s contact differential ideal on jet space and its pullback to the</td><td>Formal integrability</td></tr><tr><td>Ω</td><td>equation manifold, respectively. Independence p-form  $d x ^ { 1 } \wedge \cdot \cdot \cdot \wedge d x ^ { p } .$ </td><td>Formal integrability</td></tr><tr><td> $\mathcal { E } _ { z } ( s )$ </td><td>p-dimensional integral element of the EDS at  $z ^ { ( s ) } \in \Sigma ^ { ( s ) }$ </td><td>Formal integrability</td></tr><tr><td> $\left( \boldsymbol { \mathcal { T } } ^ { ( s ) } , \boldsymbol { \mathcal { T } } ^ { ( s ) } \right)$ </td><td>Linear Pfaffian system formed from the restricted contact one-forms and</td><td>Formal integrability</td></tr><tr><td> ${ \bf A } , { \bf A } ^ { ( 1 ) }$ </td><td>the independence one-forms. Tableau of the linear Pfaffian system and its first prolongation.</td><td>Formal integrability</td></tr><tr><td> $[ \mathbf { T } ]$ </td><td>Essential-torsion equivalence class in the cokernel of the alternation map.</td><td>Formal integrability</td></tr></table>