# De Rham cohomological architectures for PINNs and PDEs with topological holes

Andrew Gracyk<sup>1</sup>

Department of Mathematics, Purdue University, West Lafayette, IN 47907, United States

## Abstract

We attempt to alleviate failure modes of classical physics-informed neural networks (PINNs) in scenarios of topological pathologies corresponding to punctures, holes, poles, and voids by introducing de Rham cohomological augmentations, or Hodge-de Rham PINNs. Vanilla PINNs often encounter pitfalls in hole-based settings due to discontinuous interfaces, nonzero closed path integration, path dependence, correspondence to exact forms via the exterior derivative, etc., therefore we absolve traditional failures via parameterizing a 1-form by the Hodge-Morrey-Friedrichs (HMF) decomposition for manifolds with boundary, drawing a connection to de Rham cohomology such as through the HMF isomorphism theorem. 1-forms can act as representations of scalar-valued PDE solutions through integral operator mappings. We propose performing a PINN-based loss on the parameterized 1-form through an associated PDE, which subsequently solves the potential-based PINN problem and moreover remedies the failures of vanilla PINNs via the Sobolev-regular operator under a Fr´echet diferentiation interconnection. The operator relationship often has a nontrivial closed-form and often requires integration or a case-by-case analysis, which we address thoroughly in our experiments. When an integrability condition is not met on our losses, we can equip exhaustion functions to train the networks. We demonstrate our methods on the Aharonov-Bohm efect, which, to our understanding, is largely nonexistent in PINN literature. We develop theoretical guarantees of our methods and demonstrate our pathological safeguards on a broad selection of topologically-adverse problems, including scenarios of inexact forms, and those of nontrivial harmonics and coexact terms.

Key words. de Rham cohomology, diferential topology, algebraic topology, algebraic geometry, musical isomorphism, Sobolev, 1-form, coexact, puncture, pole, Hodge decomposition, Hodge-Morrey-Friedrichs, Helmholtz-Hodge, Hodge-de Rham, harmonic representative, Aharonov-Bohm, PINN, physics-informed

AMS MSC Classifications (2020): 58A12, 58A10, 68T07, 65N99

## Contents

1 Introduction 3   
2 Notations and conventions 6   
3 Methods 7   
3.1 Setup 7   
3.2 Primary contribution . . 10   
3.3 Extracting the PINN solution from the form, and vice-versa 13   
4 Training 14   
5 Experiments 18   
6 Conclusions 18   
7 Acknowledgments 20   
8 AI statement 20   
A Theoretical discussion 24   
B Additional discussion 29   
C Manners of eficiency 30   
D Coexact formulas in generalized cases 32   
E The loss function in 3D 33   
F Experimental details 34   
F.1 Unit radial vortex 34   
F.2 Arctangent without zero 35   
F.3 Arctangent with exhaustion 37   
F.4 The Aharonov–Bohm efect with one pole 37   
F.5 The Aharonov-Bohm efect with two poles 39   
G Connections to complex and K¨ahler geometries 39   
G.1 A Pythagorean loss function 40   
G.2 An exact architecture 41   
G.3 Full de Rham K¨ahler architecture 43   
H K¨ahler manifold experiments 44   
H.1 K¨ahler manifold with vector-valued PINN success 44   
H.2 K¨ahler manifold with vector-valued PINN failure 45   
I A further scalar-valued PDE solution with nonzero coexact term example 45   
J Additional figures 46

## 1 Introduction

We address topological downfalls of vanilla (or baseline) physics-informed neural networks (PINNs) via strategies from topology and algebraic geometry via the de Rham cohomology. Topological pathologies accompany an ill-learnable baseline framework, so vanilla PINNs catastrophically fail on a variety of tasks that are largely associated with poles, holes, and voids on the domains of interest.

Scalar-valued PINNs, which are standard for a broad variety of PDE tasks, fail on topologically adverse domains. This is in part due to the fact that

$$
\oint _ { \gamma } d u _ { \theta } = \int _ { \partial \gamma } u _ { \theta } = 0 ,\tag{1.1}
$$

since scalar valued functions are implicitly associated through exact forms via du<sub>θ</sub> and the exterior derivative. We demonstrate the phenomenon that scalar PINNs are associated to exact forms in Figure 26. This is adverse because, in this work here, we will need to find scalar solutions such that the associated 1-form via the exterior derivative that solves the PDE is not exact, which we show in 30. In particular, taking the exterior derivative of a scalar PINN yields an exact form. This is not to say a scalar function can have zero association to a non-exact form, which is entirely how this work is developed. Indeed, we will associate non-exact 1-forms to scalar PDE solutions via Sobolev-regular operators, the Fr´echet derivative, and integration. More specifically, u is largely associated to an exact form because u is a globally defined, single-valued smooth<sup>1</sup> function, so its exterior derivative du (which the PINN computes via automatic diferentiation) is exact. We note

$$
d u = \sum _ { i = 1 } ^ { d } \frac { \partial u } { \partial x ^ { i } } d x ^ { i } , \quad d u ^ { \sharp } = \nabla u\tag{1.2}
$$

(assuming a flat metric), which is information PINNs usually use. To illustrate further that scalar PINNs associate with exact forms, note that for a line parameterization $c : [ 0 , 1 ]  \Omega , c ( 0 ) = c ( 1 ) = p $ , we evaluate the circulation around the loop

$$
\oint _ { \gamma } d u _ { \theta } = \int _ { \partial \gamma } u _ { \theta } = u _ { \theta } ( c ( 1 ) ) - u _ { \theta } ( c ( 0 ) ) = 0 .\tag{1.3}
$$

This too is reflected in Figure 26. In many of our experiments, we require the above to evaluate to a nonzero number.

Our methods can remedy discontinuous branch cuts or undesirable behavior caused by poles. We will term these discontinuous interfaces as branch cuts. The term branch cut largely has meaning in complex analysis but we use it here with meaning in topology for example to illustrate a topological obstruction to exactness. Moreover, functions on manifolds around poles can experience blow-up (this is a main reason our experiment of a vector-valued PINN in Figure 27 fails, although this blow-up is more of a soft blow-up than a divergence). One possible consideration is to equip our (compact) manifolds with exhaustion functions.

De Rham cohomology is the suitable tool for such purposes. As mentioned in the previous paragraph, poles can create topological obstructions that ordinary PINNs cannot work around. It is true that removing poles can create nontrivial cohomology. For example, removing poles in two-dimensional domains can introduce noncontractible loops, which generate nontrivial first de Rham cohomology. Moreover, the first de Rham cohomology group identifies closed 1-forms that cannot be represented as gradients of globally-defined scalar functions. De Rham cohomology allows us to turn the topological pitfalls into something parameterizable using the dimension of the quotient module (let us not concern ourselves with the associated ring for the sake of simplicity), since

$$
b _ { 1 } = \dim H _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } ) .\tag{1.4}
$$

Here, $b _ { 1 }$ is the first Betti number. For example, in $\mathbb { R } ^ { 2 }$ with m distinct punctures, we get $b _ { 1 } = m$

Moreover, de Rham cohomology has interconnections to the Hodge decomposition. We will work with compact, smooth manifolds with boundary in an empirical setting, so we will specifically work with the three-term Hodge-Morrey-Friedrichs (HMF) decomposition

$$
\Omega ^ { k } ( M ) = d ( \Omega ^ { k - 1 } ( M ) ) \oplus \delta ( \Omega _ { N } ^ { k + 1 } ( M ) ) \oplus \mathcal { H } _ { A } ^ { k } ( M ) ,\tag{1.5}
$$

which serves as the underpinning of our architectures here. Note the above of 1.5 can also be referred to as the Helmholtz-Hodge decomposition, although this is somewhat colloquial because the decomposition referred to this exactly is less-existent in literature. Note 1.5 is verbatim in Su et al. (2024b) in section 2.2, equation 2.13, and is also almost found verbatim in Bhatia et al. (2013) on page 1390. We also refer to D’Elia et al. (2020) Schwarz (1995) Weyl (1940) F. (1941). Note a historical reference is Helmholtz (1858), which is in German. We remark the standard Hodge decomposition is not suitable for our purposes here because the boundary is incompatible with this framework, i.e. integration by parts produces a nontrivial boundary term. For example, via Stokes’ theorem

$$
\langle d \alpha , \delta \beta \rangle _ { L ^ { 2 } } = \langle \alpha , \delta ^ { 2 } \beta \rangle _ { L ^ { 2 } } + \int _ { \partial { \cal M } } \alpha \wedge { \star } \delta \beta = \int _ { \partial { \cal M } } \alpha \wedge { \star } \delta \beta .\tag{1.6}
$$

The classical Hodge decomposition is intended for compact Riemannian manifolds without boundary. $\mathcal { H } _ { A } ^ { k }$ is absolute harmonic space instead of unconstrained harmonic space $\mathcal { H } ^ { k }$ , which we need because of our choice of manifolds. Note the above of 1.5 also has connections to the generalized Hodge-Morrey-Friedrichs decomposition, which is as follows Shonkwiler (2009) Morrey Jr. (1956) Friedrichs (1955):

$$
\Omega ^ { k } ( M ) = d ( \Omega _ { D } ^ { k - 1 } ( M ) ) \oplus \delta ( \Omega _ { N } ^ { k + 1 } ( M ) ) \oplus \mathcal { H } _ { A } ^ { k } ( M ) \oplus \mathcal { H } _ { \mathrm { e x } } ^ { k } ( M ) .\tag{1.7}
$$

We will work with 1.5 without loss of generality, although they are almost identical for our purposes because neural network $\phi _ { \theta }$ can capture $d ( \Omega _ { D } ^ { k - 1 } ( M ) )$ and the exact harmonic forms $\mathcal { H } _ { \mathrm { e x } } ^ { k } ( M )$ together, i.e. we can bundle the exact terms together. In particular, we could bundle the exact terms as

$$
\begin{array} { r l } & { d ( \Omega _ { D } ^ { k - 1 } ( M ) ) \oplus \mathcal { H } _ { \mathrm { e x } } ^ { k } ( M ) \longmapsto \ d ( \Omega ^ { k - 1 } ( M ) ) } \\ & { } \\ & { \phi _ { \theta } \in \Omega ^ { k - 1 } ( M ) \longmapsto \ d } \end{array}
$$

In 1.7, D stands for Dirichlet and N stands for Neumann. The boundary conditions are required to enforce $L ^ { 2 }$ orthogonality, i.e. a direct sum versus as an algebraic sum, and we do not lose expressive power in our ansatz by dropping these conditions. We will largely ignore the boundary conditions as in 1.5, which is fine for our purposes, and we will maintain expressivity in our ansatz of 3.18. De Rham cohomology is an algebraic means to measure topological obstruction via the quotient module $H _ { \mathrm { d R } } ^ { k } ( M ) = Z _ { \mathrm { d R } } ^ { k } / B _ { \mathrm { d R } } ^ { k }$ . There exists a canonical isomorphism

$$
H _ { \mathrm { d R } } ^ { k } ( M ) \cong { \mathcal { H } } _ { A } ^ { k } ( M ) ,\tag{1.8}
$$

via the Hodge-Morrey-Friedrichs theorem/isomorphism Cappell et al. (2006) (which is closely related to the Hodge isomorphism theorem Wells (1980) Warner (1983), but not the same), and the right-hand side of 1.8 will be the space in which our harmonic forms will live. Note that 1.8 can also be written in terms of D, N conditions, nuanced by absolute and relative cohomology. It is also true that

$$
\dim { \mathcal { H } } _ { A } ^ { 1 } ( M ) = \dim H _ { \mathrm { d R } } ^ { 1 } ( M ; \mathbb { R } ) = b _ { 1 } ( M ) .\tag{1.9}
$$

We will use this as well to parameterize our harmonic forms corresponding to the number of punctures. In all of our experiments, the number of punctures will be equal to the first Betti number $b _ { 1 }$ . Note that we have also chosen k = 1 in 1.9. In particular, the terms we will call h in this work will greatly afect the learning of topological obstruction. In 3.18, we will parameterize our 1-forms that are intended to solve the PINNs via utilizing a decomposed ansatz. We subsequently draw connections to these 1-forms to the associated globally-defined scalar-valued PINN via the techniques in section 3.3.

Related work. Huang et al. (2022) addresses a PINNs for a fluid-solid interface with no-slip conditions and the simulated PINNs contain fluid dynamics around topological holes. Zhou et al. (2025) also discusses PINNs for topological pitfalls via a Hybrid Boundary PINN (HB-PINN), which achieves SOTA performance on tasks such as 2D cylinder wake flow and the 2D blocked cavity flow. Hu et al. (2024) addresses PINNs under singularities and geometric irregularities for Poisson equations, although these methods are not entirely compatible with poles. Overall, these methods are partially consistent with ours but are not intended for smooth-violating topologies and failure modes of PINNs under pole corruption.

Bastian et al. (2026) discusses topological neural operators (TNOs), which lifts neural operators to more nontrivial topological domains. They also introduce Hierarchial TNOs for propagation of topological information. This work also discusses the Hodge decomposition, which is crucial to our work here.

Baking topological features into neural network architectures in general is not new in literature, although it is new in the manner we present it here. Crocoll et al. (2026) discusses GLENN, or neural networks for Ginzburg-Landau problems. Their approach is hybrid: it uses unsupervised learning and, in conjunction, learns topological constraints with finite elements. Wu et al. (2022) constructs a PINN by integrating the physics into the model directly via a ”cascaded deep neural network.” Liu et al. (2024) resolves topological nuances via rotation, driven by orbital angular momentum, which incorporates the topological features into the neural network ansatz, which has connections to our work here.

We discuss work relevant to de Rham cohomology in a machine learning setting. Hu et al. (2026) constructs de Rham subcomplexes generted by shallow ReLU networks. Su et al. (2024a) discusses manifold topologica learning via persistent de Rham-Hodge Laplacian. Longo et al. (2023), most closely aligned with ours, embeds valid geometry of the discrete de Rham complex directly into deep learning architectures. Their approach is based on finite element strategies with basis functions. This paper is mostly an approximation theory paper. This paper is not based on the Hodge decomposition, which is what our work here does. Chen et al. (2026) discusses a deep Ritz method for PDE learning on the de Rham complex and overall discusses PINNs in these scenarios. This work is not based on an Hodge ansatz, nor is it largely based in learning with poles as in our work, although this work has general connections to Hodge theory. These works, while compatible with our de Rham cohomological perspectives to machine learning, are mostly distant from the work we present here.

A substantial contribution of our experiments is a working PINN on the Aharonov-Bohm efect. To our knowledge, this experiment is largely nonexistent in PINN literature. Brevi et al. (2024) discusses PINNs for quantum systems based on ring-shaped domains, namely a Schr¨odinger eigenfunction system, which has connections to our work here. Holliday et al. (2023) also discusses PINNs for quantum eigenproblems on more nontrivial domains. De Raedt and Michielsen (1994) discusses numerical solutions for the Aharonov–Bohm efect with accuracy, although this work is not PINN-based. Ovall and Zhu (2025) studies a gauge choice that can make Schr¨odinger eigenfunctions less oscillatory and easier to approximate.

Additional discussion. Let us be convinced that making these PINNs vector-valued does not solve the problems we present here. We will show this with empirical evidence throughout this work. A vector-valued PINN cannot resolve the discontinuity issues we face here, such as in Figures 5, 16, since discontinuities are naturally incompatible with smooth activations; however, a vector-valued PINN does not necessarily correspond to an exact 1-form, therefore a vector-valued PINN can remedy this issue.

Let us discuss more with the example in F.4. If S, ρ as in this section are parameterized directly, we introduce the branch cut or jump discontinuity. Our ansatz succeeds because we never ask the network to learn the branch cut directly since we introduce

$$
S _ { \mathrm { t o t a l } } ( x , y ) = S _ { \mathrm { s m o o t h } } ( x , y ) + c _ { 1 } \frac { \theta } { 2 \pi } ,\tag{1.10}
$$

and the harmonic basis generator $\theta = \arctan ( y / x )$ , which is systemicatically constructed and built into the ansatz, learns the discontinuity for us. Let us try again to parameterize the solution as vector valued as

$\mathbf { v } _ { \boldsymbol { \theta } } ( \boldsymbol { x } , \boldsymbol { y } )$ . To get S from the vector PINN $\mathbf { v } _ { \theta }$ , we must perform numerical integration (this is exactly what we do anyway in section F.4). Because the space has a hole and

$$
\oint \mathbf { v } _ { \theta } \cdot d \mathbf { l } \neq 0 ,\tag{1.11}
$$

this integral is path-dependent. Our methods bypass path-independence since our methods (often) compute

$$
S ( \mathbf { x } ) = S ( \mathbf { x } _ { 0 } ) + \int _ { C } \alpha _ { \theta }\tag{1.12}
$$

$$
= S ( \mathbf { x } _ { 0 } ) + \underbrace { \int _ { C } d \phi _ { \theta } } _ { \mathrm { p a t h - i n d e p e n d e n t } } + \sum _ { j = 1 } ^ { N } \frac { c _ { j } } { 2 \pi } \underbrace { \int _ { C } d \theta _ { j } } _ { \mathrm { p a t h - d e p e n d e n t } } + \mathrm { v a n i s h i n g ~ c o e x a c t ~ p a r t }\tag{1.13}
$$

$$
= S ( \mathbf { x } _ { 0 } ) + \Big ( \phi _ { \theta } ( \mathbf { x } ) - \phi _ { \theta } ( \mathbf { x } _ { 0 } ) \Big ) + \sum _ { j = 1 } ^ { N } c _ { j } \left( \frac { \Delta \theta _ { j } ( C ) } { 2 \pi } \right)\tag{1.14}
$$

for suitable path C. Sometimes, we will only integrate the $\phi _ { \theta }$ term. As mentioned in the abstract of this work, a lot of our methods needs evaluation on a case-by-case basis. We illustrate a path-dependence phenomenon in Figure 29.

## 2 Notations and conventions

We will let $\Omega ^ { * } \subseteq \mathbb { R } ^ { d }$ be a punctured manifold of dimension $d ,$ or possibly a submanifold of $\mathbb { C } ^ { d }$ in the complex case (see section G). We will use $\alpha _ { t } \in \Gamma ( \Omega ^ { * } , \Lambda ^ { 1 } T ^ { * } \Omega ^ { * } )$ to denote a 1-form (we will omit the t subscript for simplicity), or correspondingly a smooth section of the cotangent bundle such that $\begin{array} { r } { T ^ { * } \Omega ^ { * } = \coprod _ { p \in \Omega ^ { * } } T _ { p } ^ { * } \Omega ^ { * } } \end{array}$ corresponding to a scalar valued PDE solution u : $\Omega ^ { * } \times [ 0 , T ]  \mathbb { R }$ (note that u is not necessarily $L ^ { 2 }$ , therefore it is of interest to equip an exhaustion function $f )$ . We will denote an exact component to α as $d \phi ,$ a coexact component $\delta \beta .$ , and h the corresponding harmonic representatives such that α obeys the decomposition $\begin{array} { r } { \alpha = d \phi + \delta \beta + \sum _ { j } c _ { j } h _ { j } } \end{array}$ . We will assume the gradient operator $\nabla = ( \partial _ { x _ { 1 } } , \ldots , \partial _ { x _ { d } } )$ is taken with respect to the flat Euclidean metric $g _ { i j } = \delta _ { i j }$ unless stated otherwise. We will use notation

$$
\oint _ { \gamma } \alpha = \int _ { a } ^ { b } \sum _ { j = 1 } ^ { d } \alpha _ { j } ( \gamma ( s ) , t ) { \frac { d \gamma ^ { j } } { d s } } ( s ) d s\tag{2.1}
$$

to denote the circulation along $\gamma$ for a smooth 1-form α section in the cotangent bundle, where $\begin{array} { r l } { \alpha _ { t } } & { { } = } \end{array}$ $\begin{array} { r } { \sum _ { i = 1 } ^ { d } \alpha _ { j } ( x , t ) d x ^ { j } , \gamma ( s ) = ( \gamma ^ { 1 } ( s ) , \ldots , \gamma ^ { d } ( s ) ) } \end{array}$ , and $[ a , b ]$ is a suitable domain for parameterizing $\gamma .$ We will use Φ, Ψ to denote the spatial diferential operators in the PDE to be learned for the scalar solution and on the components of $\alpha ^ { \sharp }$ respectively, where $\sharp$ is the sharp musical isomorphism operator. When we say cohomology, we refer to the k-th cohomology module as a quotient space of modules

$$
H ^ { k } = Z ^ { k } / B ^ { k } = { \frac { \ker ( d ^ { k } ) } { \operatorname { I m } ( d ^ { k - 1 } ) } } : = \{ \omega + \operatorname { I m } ( d ^ { k - 1 } ) \mid \omega \in \ker ( d ^ { k } ) \} ,\tag{2.2}
$$

via a coset definition, where $d ^ { k } : C ^ { k } \to C ^ { k + 1 }$ is a homomorphism as in the cochain complex

$$
\dotsb { \xrightarrow { { d } ^ { k - 2 } } } C ^ { k - 1 } { \xrightarrow { { d } ^ { k - 1 } } } C ^ { k } { \xrightarrow { { d } ^ { k } } } C ^ { k + 1 } { \xrightarrow { { d } ^ { k + 1 } } } \dotsb\tag{2.3}
$$

We will take in cohomology [α] to belong to the first de Rham cohomology group (see section 3.1). In this work, when we say a form ω is exact, we mean $\omega = d u$ for some u, where d is the exterior derivative, and we say a form is closed if $d \omega = 0$ . Because $d ^ { 2 } = 0$ , every exact form is automatically closed $( d ( d u ) = 0 )$ . We will denote $\theta$ to denote a neural network parameter and $g _ { \theta }$ to denote a neural network. Our notation is mostly standard.

## 3 Methods

## 3.1 Setup

![](images/7c396915f3d69192a085b734923bd2ca5026f427aa38e9963f9b088f82314328.jpg)  
(1)

![](images/80c5f4e3f2dc0d9ff667aed6341d5dd98086f5888882b23eeddbf29cf5141968.jpg)  
(2)

![](images/f5d23e9a3a799b64637a1da0baac4a5ecdee62618978201f0559290fa2cf60b8.jpg)  
(3)  
Figure 1: We plot success of a (1) Hodge-de Rham PINN and failure of a (2) vanilla PINN on a domain $\{ ( x , y ) | 1 <$ $x ^ { 2 } + y ^ { 2 } < 4 \}$ with a topological hole under the PDE $\Delta u = \frac { \partial ^ { 2 } u } { \partial x ^ { 2 } } + \frac { \partial ^ { 2 } u } { \partial y ^ { 2 } } = 0$ , where $\Delta$ is the Laplacian with respect to the flat metric. The true solution is (3). We use the loss $\begin{array} { r } { \mathcal { L } = \| \Delta u \| _ { L ^ { 2 } ( \Omega ^ { * } ; d \mu _ { \Omega } ^ { * } ) } ^ { 2 } + \left\| \frac { \partial u } { \partial x } x + \frac { \partial u } { \partial y } y \right\| _ { L ^ { 2 } ( \Gamma _ { \mathrm { i n } } ^ { * } ; d \mu _ { \mathrm { i n } } ^ { * } ) } ^ { 2 } + } \end{array}$ $\left. \nabla u - v _ { \mathrm { t a r g e t } } \right. _ { L ^ { 2 } ( \Gamma _ { \mathrm { o u t } } ^ { * } ; d \mu _ { \mathrm { o u t } } ^ { * } ) } ^ { 2 }$ . We train for 12,000 epochs and use a Fourier feature embedding as in Wang et al. (2023) with $\sigma = 1 . 0$ . We see complete failure of the PINN. By the Fundamental Theorem of Line Integrals, we have $\oint _ { \Gamma _ { \mathrm { o u t } } } \boldsymbol { \nabla } u \cdot d \boldsymbol { l } = u ( \mathrm { s t a r t } ) - u ( \mathrm { e n d } ) = 0$ , but $v _ { \mathrm { t a r g e t } }$ corresponds to a non-exact form. This PINN completely fails (as opposed to only failing at the discontinuity) because Neumann boundary conditions are used.

Consider a PDE over a punctured domain

$$
\Omega ^ { \ast } = \Omega \setminus \{ p _ { 1 } , \dots , p _ { m } \} ,\tag{3.1}
$$

where $m \in \mathbb { N }$ and $m < \infty$ . More generally, when $p _ { i }$ has nonzero measure, let $K _ { 1 } , \ldots , K _ { m } \subset \Omega$ be disjoint, compact sets. The punctured domain becomes

$$
\Omega ^ { * } = \Omega \setminus \bigcup _ { j = 1 } ^ { m } K _ { j } .\tag{3.2}
$$

Now, take the PDE to have the form such that $u \in \mathcal { B } : = L ^ { 2 } \left( 0 , T ; H ^ { k } ( \Omega ^ { * } ) \right) \cap H ^ { 1 } \left( 0 , T ; L ^ { 2 } ( \Omega ^ { * } ) \right)$ and

$$
\left\{ \begin{array} { l l } { \partial _ { t } u ( x , t ) + \Phi \left[ x , t , u , \partial _ { t } u , \nabla u , \nabla ^ { 2 } u , \dots \right] = 0 , } & { ( x , t ) \in \Omega ^ { * } \times ( 0 , T ] } \\ { u ( x , 0 ) = f ( x ) , } & { x \in \Omega ^ { * } . } \end{array} \right.\tag{3.3}
$$

A standard PINN consider the physics loss

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p h y s i c s } } = \mathbb { E } _ { ( x , t ) \sim \mathcal { U } ( \Omega ^ { * } ) \otimes \mathcal { U } ( [ 0 , T ] ) } [ \bigg | \partial _ { t } u ( x , t ) + \Phi \Big [ x , t , u _ { \theta } , \partial _ { t } u _ { \theta } , \nabla u _ { \theta } , \nabla ^ { 2 } u _ { \theta } , . . . \Big ] | ^ { 2 } ] . } \end{array}\tag{3.4}
$$

We will often consider $\alpha \in \mathcal { B } _ { \Lambda } : = L ^ { 2 } \left( 0 , T ; H ^ { k } \Lambda ^ { 1 } ( \Omega ^ { * } ) \right) \cap H ^ { 1 } \left( 0 , T ; L ^ { 2 } \Lambda ^ { 1 } ( \Omega ^ { * } ) \right)$ to be the target form such that $\alpha = d u$ locally, i.e. $\alpha = d u$ on any simply connected open set $U \subseteq \Omega ^ { * }$ , but not every $\alpha \in \Omega ^ { 1 } ( \Omega ^ { * } )$ satisfies this due to a coexact term. We will elaborate on general strategies regarding form and scalar conversions in sections 3.3 and 5. In this work, it is also of interest to consider time-independent PDEs, so a loss of

$$
\widetilde { \mathcal { L } _ { \mathrm { p h y s i c s } } } = \mathbb { E } _ { { x } \sim \mathcal { U } ( \Omega ^ { * } ) } \Bigg [ \Bigg | \Phi \left[ x , t , u _ { \theta } , \partial _ { t } u _ { \theta } , \nabla u _ { \theta } , \nabla ^ { 2 } u _ { \theta } , \dots \right] \Bigg | ^ { 2 } \Bigg ] .\tag{3.5}
$$

We have renamed $\Phi  \widetilde { \Phi [ \cdot ] }$ to account for a new operator. We will relax this discussion throughout 4, so we will discuss for the sake of simplicity time-dependent PDEs; however, some of our experiments will use time-independent PDEs (see Appendix F.1).

For a locally exact form, because $U$ has no holes, Poincar´e’s Lemma Bott and Tu (1982) guarantees that every closed form is exact on $U ;$ however, there does not exist a single global potential ϕ on all of $\Omega ^ { * }$ , and so $\alpha \neq d \phi$ for all $\phi : \Omega ^ { * } \to \mathbb { R }$ . Because a neural network $u _ { \theta } ( x )$ is globally defined, it is mathematically incapable of being $\phi .$ . We highlight this with an example:

Example 1. Consider the canonical 1-form for a puncture at the origin

$$
\alpha = { \frac { - y d x + x d y } { x ^ { 2 } + y ^ { 2 } } } .\tag{3.6}
$$

By computing the exterior derivative, we verify $d \alpha = 0$ everywhere on $\Omega ^ { * } = \mathbb { R } ^ { 2 } \backslash \{ 0 \}$ . Locally, let us integrate α to find a potential u. When we do this, we get the (smooth) polar angle

$$
u ( x , y ) = \theta = \arctan \left( { \frac { y } { x } } \right) .\tag{3.7}
$$

So locally, $\alpha = d u$ . Let us attempt to define $u = \theta$ globally on the punctured plane. If we start at coordinates $( 1 , 0 )$ and define $u = 0$ , we walk counter-clockwise along the unit circle, so u continuously increases. When we arrive back at $( 1 , 0 )$ , continuity dictates that $u = 2 \pi$ . Therefore, if u is scalar, a contradiction is achieved. In particular, for a typical neural network $\phi _ { \theta } : \Omega ^ { * } \to \mathbb { R } .$ , at a coordinate $( x , y )$ , the network by definition outputs a scalar. If the network tries to learn $u = \theta .$ , the network must satisfy one of two properties, impossible together:

I. it must either output a smooth function (up to smoothness of the activation) which fails to represent the 2π jump;

II. it must force a jump discontinuity, which ruins the derivative $d \phi _ { \theta } \colon$ ; automatic diferentiation is not intended for these types of scenarios, and smooth networks with smooth activations cannot handle a jump discontinuity.

In particular,

$$
\begin{array} { c c c } { { u _ { \theta } \in C ^ { k } ( \Omega ^ { * } ) \xrightarrow { } \qquad } } & { { \left\lfloor u _ { \theta } \right\rfloor = 0 } } \\ { { \mathrm { ! } } } & { { } } & { { \mathrm { ! } } } \\ { { \mathrm { ! } } } & { { } } & { { \qquad } } \\ { { u _ { \theta } \notin C ^ { k } ( \Omega ^ { * } ) \longleftarrow } } & { { \left\lfloor u _ { \theta } \right\rfloor \neq 0 } } \end{array}
$$

or more informally, a smooth solution cannot satisfy the discontinuity and if it satisfies the discontinuity, it breaks the PINN. We use jump operator notation $[ [ u _ { \theta } ] ] = u _ { \theta } ^ { + } - u _ { \theta } ^ { - }$ . Note that PINNs for discontinuities do exist Liu et al. (2023); it is more so that baselines fail, which Liu et al. (2023) details. Alternatively, if the jump discontinuity is exactly along the boundary, the PINN is not necessarily broken. For example, learning Im $( \phi _ { \theta } ) \subseteq ( 0 , 2 \pi )$ . Therefore we consider an ϵ-neighborhood or open set around the discontinuity to be of interest in the learning task, so $( - \epsilon , 2 \pi + \epsilon )$ is the walked-path in the example. Hence, α ̸= dϕ globally.

Hence, we can conclude the form α satisfies the de Rham obstruction

$$
[ { \boldsymbol { \alpha } } ] \in H _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } ) ,\tag{3.8}
$$

where we define

$$
Z _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } ) : = \mathrm { k e r } ( d \colon \Omega ^ { 1 } \to \Omega ^ { 2 } ) = \{ \alpha \in \Omega ^ { 1 } \mid d \alpha = 0 \} \qquad \mathrm { ( c l o s e d ~ f o r m s ) , }\tag{3.9}
$$

$$
B _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } ) : = \mathrm { I m } ( d \colon \Omega ^ { 0 } \to \Omega ^ { 1 } ) = \{ \alpha \in \Omega ^ { 1 } \mid \alpha = d \phi \mathrm { ~ f o r ~ s o m e ~ } \phi \in \Omega ^ { 0 } \} \qquad \mathrm { ( e x a c t ~ f o r m s ) , }\tag{3.10}
$$

$$
H _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } ) : = Z _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } ) / B _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } ) .\tag{3.11}
$$

Here, the quotient space $H _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } ) : = Z _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } ) / B _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } )$ is called the first de Rham cohomology group. The first de Rham cohomology group is isomorphic to the quotient space, of interest in the learning task in which the harmonic forms will belong, in the weighted $L ^ { 2 } \mathbf { . }$ -cohomology group

$$
H _ { L ^ { 2 } } ^ { 1 } ( \Omega ^ { * } , f ) = \frac { \ker \left( \Gamma _ { L ^ { 2 } ( f ) } ( \Omega ^ { * } , \Lambda ^ { 1 } T ^ { * } \Omega ^ { * } ) \xrightarrow { d } \Gamma _ { L ^ { 2 } ( f ) } ( \Omega ^ { * } , \Lambda ^ { 2 } T ^ { * } \Omega ^ { * } ) \right) } { \operatorname { I m } \left( \Gamma _ { L ^ { 2 } ( f ) } ( \Omega ^ { * } , \Lambda ^ { 0 } T ^ { * } \Omega ^ { * } ) \xrightarrow { d } \Gamma _ { L ^ { 2 } ( f ) } ( \Omega ^ { * } , \Lambda ^ { 1 } T ^ { * } \Omega ^ { * } ) \right) } ,\tag{3.12}
$$

and accounts for the holes in this setting. The learned network form belongs to $\alpha _ { \theta } \in \Gamma _ { L ^ { 2 } ( f ) } ( \Omega ^ { * } , \Lambda ^ { 1 } T ^ { * } \Omega ^ { * } ) \cap$ $\Omega ^ { 1 } ( \Omega ^ { * } ) . \ \Gamma _ { L ^ { 2 } ( f ) } ( \Omega ^ { * } , \Lambda ^ { 1 } T ^ { * } \Omega ^ { * } )$ is the setting for integrability and $\Omega ^ { 1 } ( \Omega ^ { * } )$ is the setting for smoothness (note $\Gamma ( \cdot )$ usually means smooth sections but not in this context). PINNs are generally over bounded and compact domains, and are typically done in $L ^ { 2 }$ settings for their scalar field equivalents, and their form analogs are no diferent; however, the poles can cause integrability issues. Theoretically, we can equip $\Omega ^ { * }$ with an exhaustion function $f$ to ensure a ”safe region” of the PINN is trained on; more specifically, the exhaustion functions make the entire domain ”safe” whereas, without exhaustion, there is (possible) blow-up near the pole. For $f$ to formally be considered an exhaustion function, it must satisfy $\{ x \in \Omega ^ { * } | \dot { f } ( x ) \leq c \}$ to be compact. By construction of $\Omega ^ { * }$ , the level sets are compact. Without loss of generality, we can always take $\Omega ^ { * }$ compact in an empirical setting anyway. The form $\alpha \theta$ will diverge at the pole, but the $e ^ { - f }$ weight will force decay to ensure an integrability condition is met, since a PINN loss function will require specific integrability. Indeed, PINN loss can explode, so consider the example

$$
\mathcal { L } _ { \mathrm { p h y s i c s } } \ = \int _ { 0 } ^ { 2 \pi } \int _ { 0 } ^ { R } \left( \frac { 1 } { r } \right) ^ { 2 } r d r d \theta = 2 \pi \int _ { 0 } ^ { R } \frac { 1 } { r } d r = 2 \pi \left[ \ln | r | \right] _ { 0 } ^ { R } = \infty .\tag{3.13}
$$

We witness the above ourselves in our experiments (see sections 5, F.2). In section F.2, we specifically cut-out a suficiently nice region around the pole to prevent divergence of the loss. By equipping an exhaustion function, we ensure a loss can be satisfied, although the exhaustion function ensures $[ \alpha ]$ belongs to this class but will not need to be computed. With an exhaustion function, the physics loss on the form is

$$
\mathcal { L } _ { \mathrm { p h y s i c s } } = \int _ { \Omega ^ { * } } \left. \mathrm { e m p i r i c a l ~ r i s k } ^ { \sharp } ( x ) \right. ^ { 2 } e ^ { - f ( x ) } d x \approx \frac { 1 } { N _ { f } } \sum _ { i = 1 } ^ { N _ { f } } \frac { e ^ { - f ( x _ { i } ) } } { p ( x _ { i } ) } \left. \mathrm { e m p i r i c a l ~ r i s k } ^ { \sharp } ( x _ { i } ) \right. ^ { 2 } ,\tag{3.14}
$$

(possibly) omitting time for a time-independent PDE. We perform an exhaustion function experiment in F.3.   
Thus, the weighted $L ^ { 2 }$ -de Rham cohomology group of 3.12 has connections to PINNs.

Returning to the example, by Stokes’ theorem, integrating the diferential form, which is exact, gives

$$
\oint _ { \gamma } d \phi = \phi ( p ) - \phi ( p ) = 0 ;\tag{3.15}
$$

however, in our example,

$$
d \alpha = 0 , \qquad \oint _ { \gamma } \alpha = 2 \pi\tag{3.16}
$$

for a loop γ surrounding the pole, so α is not exact. Since $d \alpha = 0$ , α is closed, meaning $\alpha \in Z _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } )$ , but the contour integral condition means it is not exact, so α $\notin B _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } )$ (by definiton, $B _ { \mathrm { d R } } ^ { 1 }$ is the collection of exact forms). Therefore, α represents a nontrivial cohomology class in the quotient space, meaning $[ \alpha ] \in H _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } )$ and moreover in a PINN setting

$$
[ \alpha ] \in H _ { L ^ { 2 } } ^ { 1 } ( \Omega ^ { * } , f ) \cong H _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } )\tag{3.17}
$$

of 3.12.

![](images/eaa9f1fa7c7c9a9f7bc767e23208324e14e81aefe667cb14df839dfb84ed89cf.jpg)  
Figure 2: We illustrate our architecture as in 3.2. We specialize to the case of $( x , y ) \in \mathbb { R } ^ { 2 }$

## 3.2 Primary contribution

We propose introducing an ansatz of the form, for any finite number of punctures,

$$
\alpha _ { \theta } = d \phi _ { \theta } + \delta \beta _ { \theta } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } h _ { j }\tag{3.18}
$$

where $h _ { j }$ are harmonic representatives of $H _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } )$ . In particular, the above architecture is essentially built on the Hodge decomposition but with absolute harmonic forms, or the Hodge-Morrey-Friedrichs (HMF) decomposition

$$
\Omega ^ { k } ( M ) = d ( \Omega ^ { k - 1 } ( M ) ) \oplus \delta ( \Omega _ { N } ^ { k + 1 } ( M ) ) \oplus \mathcal { H } _ { A } ^ { k } ( M ) .\tag{3.19}
$$

Note this is not the same as the Hodge decomposition because we must also account for the boundary (see section 1 and discussion surrounding 3.20). As we mentioned in 1, the Hodge decomposition is primarily intended for compact Riemannian manifolds without boundary. In PINNs in empirical settings, it is most reasonable to work with compact Riemannian manifolds with boundary. Here, $\phi _ { \theta } , \beta _ { \theta } , c _ { \theta , j }$ are all learnable: ϕ , $\beta _ { \theta }$ are outputted by the neural network and $\mathcal { C } \theta , j$ are learnable, single parameters. ϕ is a 0-form, i.e. uses a scalar potential, approximating true 0-form ϕ, $\bar { \phi } \in \Gamma ( \Lambda ^ { 0 } T ^ { * } M )$ is a smooth section. We will generally take $k = 1$ , so α is a 1-form. $\beta _ { \theta }$ is a 2-form, i.e. uses a vector potential whose dimension depends on the domain, which approximates true $\beta , \beta \in \Gamma ( \Lambda ^ { 2 } T ^ { * } M )$ . In 2D space, there is only one basis form dx $\land d y$ , so $\beta _ { \theta }$ is scalar. In 3D space, a 2-form has three basis elements dy ∧ dz, dz ∧ dx, dx $\wedge d y$ , so $\beta _ { \theta }$ has 3 output neurons. δ denotes the codiferential operator, i.e. the $L ^ { 2 } .$ -adjoint of the exterior derivative $d ,$ defined via the Hodge star ⋆ : $\Lambda ^ { k } T ^ { * } M \ { \stackrel { \sim } { \to } } \ \Lambda ^ { d - k } T ^ { * } M$ so that $\delta = ( - 1 ) ^ { d ( k - 1 ) + 1 } \star d \star$ George. $b _ { 1 }$ is the first Betti number. The exterior derivative d and codiferential operator δ act in chain

$$
0 \xrightarrow [ \delta ] { } \Omega ^ { 0 } ( \Omega ^ { * } ) \xrightarrow [ \delta ] { d } \Omega ^ { 1 } ( \Omega ^ { * } ) \xrightarrow [ \delta ] { d } \Omega ^ { 2 } ( \Omega ^ { * } ) \xrightarrow [ \delta ] { d } \Omega ^ { 2 } ( \Omega ^ { * } ) \xrightarrow [ \delta ] { d } \Omega ^ { 1 } ( \Omega ^ { * } )
$$

so the exterior diferential maps to a higher smooth form, and the codiferential maps below.

Since we will be working with compact manifolds with boundary in an empirical setting, we must justify the vanishing of the boundary term

$$
\langle d \alpha , \delta \beta \rangle _ { L ^ { 2 } } = \int _ { \partial { \cal M } } \alpha \wedge \star \delta \beta .\tag{3.20}
$$

In particular, for the decomposition in 3.19 to be an orthogonal direct sum, the spaces $d ( \Omega ^ { k - 1 } ( M ) )$ and $\delta ( \Omega ^ { k + 1 } ( M ) )$ must be mutually orthogonal. We need not address boundary terms in our methods here. We will omit boundary terms in our experiments, therefore the spaces of exact and coexact forms are not orthogonal. Our methods will work in our experiments because of the decomposition as in 3.19, and not the true Hodge decomposition. This hypothesis class is expressive, and we can find a 1-form in the above that solves the associated PDE problem. More specifically, in A, we will apply the HMF decomposition to the true underlying forms $\alpha ^ { \star }$ . The PINN is unconstrained: ϕ can take on anything on the boundary (notice the coexact and harmonic components do not). The math works out because, for example, we will examine

$$
\int _ { \partial M } \phi _ { \theta } \wedge \underbrace { \star \delta \beta ^ { \star } } _ { = 0 } = 0 ,\tag{3.21}
$$

so the second term vanishes and annihilates the boundary. We will examine other terms that behave similarly as well.

Let us discuss the space of absolute harmonic forms more closely in order to distinguish it from that which is in the typical Hodge decomposition. We define

$$
\mathcal { H } _ { A } ^ { k } ( M ) = \big \{ h \in \Omega ^ { k } ( M ; \mathbb { R } ) : d h = 0 , ~ \delta h = 0 , ~ \mathbf { n } h = 0 \big \} .\tag{3.22}
$$

Here, nh denotes the normal trace of h on the boundary, which corresponds to $\iota _ { \partial M } ^ { * } ( \star h ) = 0$ . For 1-forms, this condition simplifies to $\iota _ { \nu } h | _ { \partial M } = 0$ , where ν is the outward unit normal vector. As mentioned in 1, by the Hodge-Morrey-Friedrichs theorem Schwarz (1995), this specific space yields the canonical isomorphism to the de Rham cohomology

$$
\begin{array} { r } { \mathcal { H } _ { A } ^ { k } ( M ) \cong H _ { \mathrm { d R } } ^ { k } ( M ; \mathbb { R } ) , } \end{array}\tag{3.23}
$$

meaning its dimension is the k-th Betti number, dim<sub>R</sub> $\mathcal { H } _ { A } ^ { k } ( M ) = b _ { k }$ . Therefore, we can choose a finite harmonic basis $h _ { 1 } , \ldots , h _ { b _ { 1 } }$ representing the first cohomology group, even when the manifold has a boundary.

We write $\alpha \theta$ in 3.18 with spatial and temporal input

$$
\alpha _ { \theta } ( x , t ) = d \phi _ { \theta } ( x , t ) + \delta \beta _ { \theta } ( x , t ) + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( t ) h _ { j } ( x ) .\tag{3.24}
$$

In connection to the first de Rham cohomology group, we can note $[ \alpha _ { \theta } - \delta \beta _ { \theta } ] \in H _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } )$ due to the existence of the $c , h$ terms. In a vanilla PINN, learning a scalar field $u _ { \theta } : \Omega ^ { * } \to$ R means that the model only corresponds to exact forms, i.e. du $\in B _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } )$ . This is true because α = du is the form, hence exact. Again, this is problematic because exact forms integrate to zero over any closed loop, which we saw in Example 1 partially in 3.35 (which integrated to 2π because it is not exact) and also appeared in 1. Moreover, the PINN is constrained to the trivial cohomology class [0]. In particular, in de Rham cohomology, the quotient space $H _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } ) = Z _ { \mathrm { d R } } ^ { 1 } / B _ { \mathrm { d R } } ^ { 1 }$ groups closed forms into equivalence classes. Two closed forms α and $\alpha ^ { \prime }$ belong to the same equivalence class, or written as $[ \alpha ] = [ \alpha ^ { \prime } ]$ , if and only if they difer by an exact form, that is,

$$
\alpha \sim \alpha ^ { \prime } \iff \alpha - \alpha ^ { \prime } = d \phi .\tag{3.25}
$$

Because of this rule, the trivial cohomology class [0] is the set of all exact forms, and $[ \omega ] = [ d \phi ] = [ 0 ]$ , and exact forms write as [0].

If the domain has m punctures, then $b _ { 1 } = m$ . For exactly one puncture

$$
\alpha _ { \theta } = d \phi _ { \theta } + \delta \beta _ { \theta } + c d \varphi .\tag{3.26}
$$

Computationally speaking, we compute in 2D space

$$
d \phi _ { \theta } = \frac { \partial \phi _ { \theta } ( x , y , t ) } { \partial x } d x + \frac { \partial \phi _ { \theta } ( x , y , t ) } { \partial y } d y .\tag{3.27}
$$

and moreover,

$$
\delta \beta _ { \theta } = \frac { \partial \beta _ { \theta } ( x , y , t ) } { \partial y } d x - \frac { \partial \beta _ { \theta } ( x , y , t ) } { \partial x } d y .\tag{3.28}
$$

In 3D space, we have

$$
d \phi _ { \theta } = \frac { \partial \phi _ { \theta } ( x , y , z , t ) } { \partial x } d x + \frac { \partial \phi _ { \theta } ( x , y , z , t ) } { \partial y } d y + \frac { \partial \phi _ { \theta } ( x , y , z , t ) } { \partial z } d z .\tag{3.29}
$$

Also, $\beta _ { \theta } = \beta _ { \theta } ^ { x } d y \wedge d z + \beta _ { \theta } ^ { y } d z \wedge d x + \beta _ { \theta } ^ { z }$ dx $\land d y .$ , and since $\delta = { \star } d { \star } .$ , the codiferential becomes the curl

$$
\begin{array} { c } { \displaystyle \delta \beta _ { \theta } = \left( \frac { \partial \beta _ { \theta } ^ { z } ( x , y , z , t ) } { \partial y } - \frac { \partial \beta _ { \theta } ^ { y } ( x , y , z , t ) } { \partial z } \right) d x } \\ { \displaystyle + \left( \frac { \partial \beta _ { \theta } ^ { x } ( x , y , z , t ) } { \partial z } - \frac { \partial \beta _ { \theta } ^ { z } ( x , y , z , t ) } { \partial x } \right) d y } \\ { \displaystyle + \left( \frac { \partial \beta _ { \theta } ^ { y } ( x , y , z , t ) } { \partial x } - \frac { \partial \beta _ { \theta } ^ { x } ( x , y , z , t ) } { \partial y } \right) d z . } \end{array}\tag{3.30}
$$

Scaling this architecture to dimensions $n \geq 4$ introduces computational and analytical bottlenecks, therefore we omit these dimensions for simplicity. For example, the number of independent basis elements for a 2-form is given by the binomial coeficient $\textstyle { \binom { n } { 2 } } = { \frac { n ( n - 1 ) } { 2 } }$ , which is not reasonable for typical automatic diferentiation without high-dimensional extensions Kaminsky et al. (2026) Shahab et al. (2025). Moreover, analytically computing the first Betti number $b _ { 1 }$ and deriving closed-form, diferentiable expressions for the harmonic basis forms $h _ { j }$ for arbitrary manifolds in $\mathbb { R } ^ { n }$ is a nontrivial algebraic topology problem. If we cannot provide the explicit $h _ { j }$ forms to the network, the $c _ { j } h _ { j }$ sum cannot be evaluated.

We can note $\phi _ { \theta } , \beta _ { \theta }$ are mere approximations to the true forms, so these neural network representations belong to classes

$$
\begin{array} { r } { \mathcal { A } _ { \epsilon } ^ { 0 } ( \phi ^ { \star } ) : = \left\{ \phi _ { \theta } \in \mathcal { N } _ { \Theta } ^ { 0 } \ | \ \| \phi ^ { \star } - \phi _ { \theta } \| _ { H ^ { k } ( M ) } < \epsilon \right\} , } \end{array}\tag{3.31}
$$

$$
\begin{array} { r } { \mathcal { A } _ { \epsilon } ^ { 2 } ( \beta ^ { \star } ) : = \left\{ \beta _ { \theta } \in \mathcal { N } _ { \Theta } ^ { 2 } \ | \ \| \beta ^ { \star } - \beta _ { \theta } \| _ { H ^ { k } ( M ) } < \epsilon \right\} . } \end{array}\tag{3.32}
$$

Here, $\mathcal { N } _ { \Theta } ^ { 0 } \subseteq \Gamma ( \Lambda ^ { 0 } T ^ { * } M ) , \mathcal { N } _ { \Theta } ^ { 2 } \subseteq \Gamma ( \Lambda ^ { 2 } T ^ { * } M )$ , so they are smooth sections due to the smooth activations. The networks must account for perturbation and error since they are learned and imprecise, although corresponding to true forms. Note that we take M generalized here, but often $M = \Omega ^ { * }$ is the punctured domain, and we use the $H ^ { k }$ norm, which applies to forms

$$
\| \phi \| _ { H ^ { k } ( M ) } ^ { 2 } = \sum _ { j = 0 } ^ { k } \int _ { M } \left| \nabla ^ { j } \phi \right| _ { g } ^ { 2 } d \mu _ { g }\tag{3.33}
$$

Here, $\nabla \colon \Gamma ( E ) \to \Gamma ( T ^ { * } M \otimes E )$ is the Levi-Civita connection.

Architectural motivations. The three neural network terms $\phi _ { \theta } , \beta _ { \theta } , h _ { j }$ learn the following components to the PDE:

I. $d \phi _ { \theta }$ captures the irrotational (curl-free) part of the field. Recall α is a 1-form, so it ”corresponds” to a vector field via musical isomorphism. This term accommodates smooth, continuous dynamics.

II. $\delta \beta _ { \theta }$ captures the solenoidal (divergence-free) part of the field, for vortices, eddies, and rotational flows. A pure gradient field (dϕ) cannot model rotation because the curl of a gradient is always zero

$$
\operatorname { c u r l } ( \nabla h ) = \nabla \times \nabla h = 0 .\tag{3.34}
$$

III. $\sum _ { j } c _ { \theta , j } h _ { j }$ is the harmonic component that aptly characterizes the PINN as de Rham cohomological, since $d \phi _ { \theta } , \delta \beta _ { \theta }$ are more aptly the smooth, local vector field terms. Rather, $\sum _ { j } c _ { \theta , j } h _ { j }$ absorbs the holes into the learning. If a field circles around a puncture (like a magnetic field around a wire or a fluid vortex around a cylinder), it accumulates a non-zero value over that closed loop. This term mostly rather learns the nonzero value of the closed line integral. We can note

$$
\oint _ { \gamma _ { i } } \alpha _ { \theta } = \oint _ { \gamma _ { i } } d \phi _ { \theta } + \oint _ { \gamma _ { i } } \delta \beta _ { \theta } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } \oint _ { \gamma _ { i } } h _ { j } .\tag{3.35}
$$

By Stokes’ theorem, the line integral of $d \phi _ { \theta }$ over the closed loop evaluates to zero. The term $\delta \beta _ { \theta }$ is L<sup>2</sup>-orthogonal to the space of harmonic forms (we say this nonrigorously because our domains have boundary). Assuming the harmonic basis forms are selected such that they form a dual basis to the homology cycles, meaning $\oint _ { \gamma _ { i } } h _ { j } = \delta _ { i j }$ , where $\delta _ { i j }$ is the Kronecker delta, the integral simplifies

$$
\oint _ { \gamma _ { i } } \alpha _ { \theta } = c _ { \theta , i } .\tag{3.36}
$$

The parameter $c _ { \theta , i }$ represents the de Rham period of the field, or correspondingly the $2 \pi$ we saw in Example 1, in the i-th homology cycle.

## 3.3 Extracting the PINN solution from the form, and vice-versa

A major motivation for this work is we can perform the learning on the form $\alpha _ { \theta } ,$ , and we can convert, or at the very least extract, a PDE solution u corresponding to $\alpha \theta$ . To summarize, this is oftentimes problem-dependent, but the overall recipe can be captured in a few key steps. We refer to I for a specific example.

In general, we seek an operator Γ : $\mathcal { U } \subseteq H ^ { k + 1 } ( \Lambda ^ { 0 } ( \Omega ^ { * } ) ) \longrightarrow \mathcal { A } \subseteq H ^ { k } ( \Lambda ^ { 1 } ( \Omega ^ { * } ) )$

$$
\Gamma [ u ] = \alpha , \qquad \Gamma ^ { - 1 } [ \alpha ] = u ,\tag{3.37}
$$

invertible on its image, up to boundary conditions. The simplest case $\alpha = d u$ , so it is an exact form, but this is an oversimplification relative to our proposed architecture of 3.18, and we can handle more advanced setups.

In general, there is not a universal rule for determining a closed form for Γ. In our experiments section of 5, we will elaborate for each example on how to deduce Γ and convert between $\alpha , u$

Instead, let us consider the following, which is useful and encompassing of many scenarios but is not universally applicable for learning Γ broadly:

$$
\alpha = a d u ,\tag{3.38}
$$

where a is shorthand for the 0-form (function) $\boldsymbol { a } ( \boldsymbol { x } , t )$ . Crucially,

$$
d \alpha = d a \wedge d u ,\tag{3.39}
$$

which need not vanish. Thus, a scalar solution can induce an associated 1-form with a nonzero coexact component, which has connections to the example in Appendix I. Generally, we recover the PINN solution as

$$
u _ { \theta } ( x ) = u _ { \theta } ( x _ { 0 } ) + \int _ { x _ { 0 } } ^ { x } a ^ { - 1 } \alpha _ { \theta } ,\tag{3.40}
$$

or alternatively

$$
u _ { \theta } ( x ) = u _ { \theta } ( x _ { 0 } ) + \int _ { x _ { 0 } } ^ { x } a ^ { - 1 } d \phi _ { \theta } .\tag{3.41}
$$

It is in this correspondence, we derive a new PDE associated to diferential operator $\Phi$ as in 3.1 and operator Γ. The 1-form α will satisfy an associated PDE

$$
\left\{ \begin{array} { l l } { \partial _ { t } \alpha ( x , t ) + \Psi _ { \alpha } \Big [ x , t , \alpha , \partial _ { t } \alpha , \nabla \alpha , \nabla ^ { 2 } \alpha , \dots \Big ] = 0 , } & { ( x , t ) \in \Omega ^ { * } \times ( 0 , T ] } \\ & { \qquad \quad } \\ { \alpha ( x , 0 ) = \alpha _ { 0 } ( x ) , } & { x \in \Omega ^ { * } , } \end{array} \right.\tag{3.42}
$$

where $\Psi _ { \alpha } ( t , \cdot ) : H ^ { k } ( \Lambda ^ { 1 } ( \Omega ^ { * } ) ) \longrightarrow H ^ { k - m } ( \Lambda ^ { 1 } ( \Omega ^ { * } ) )$ is a diferential operator associated with Φ $\cdot : H ^ { s } ( \Lambda ^ { p } ( \Omega ^ { * } ) ) \longrightarrow$ $H ^ { s - m _ { \Phi } } ( \Lambda ^ { \dot { q } } ( \Omega ^ { * } ) ) , \Gamma : \mathcal { U } \subseteq H ^ { k + 1 } ( \Lambda ^ { 0 } ( \Omega ^ { * } ) ) \longrightarrow \mathcal { A } \subseteq H ^ { k } ( \Lambda ^ { 1 } ( \Omega ^ { * } ) )$ ) with possible initial and boundary data. Therefore, in section 4, we consider this PDE on the loss.

The reasoning as to why the previous discussion works is because, consider the PDE with associated operator $\Gamma , \partial _ { t } u + \Phi [ u ] = 0 , \alpha = \Gamma [ u ]$ . Via chain rule, $\partial _ { t } \alpha = D \Gamma [ u ] ( \partial _ { t } u )$ . Here, D represents the Fr´echet derivative. Therefore, substituting the diferential operator $\partial _ { t } \alpha = - D \Gamma [ u ] ( \Phi [ u ] )$ . When $\Gamma ^ { - 1 }$ exists on the admissible class (note that it does not always exist), substitute $u = \Gamma ^ { - 1 } [ \alpha ]$ , and so

$$
\partial _ { t } \alpha + D \Gamma [ \Gamma ^ { - 1 } \alpha ] \left( \Phi [ \Gamma ^ { - 1 } \alpha ] \right) = 0 ,\tag{3.43}
$$

and the new associated operator defines Ψ. Note that this does not guarantee Ψ is a local diferential operator, i.e. the inverse of Γ may require integration. Also, α must be in the image of Γ. Note that we are using notation so the applications are element-wise, i.e.

$$
\partial _ { t } \alpha ^ { \sharp } + \left( D \Gamma [ \Gamma ^ { - 1 } \alpha ] \left( \Phi [ \Gamma ^ { - 1 } \alpha ] \right) \right) ^ { \sharp } = \binom { 0 } { \vdots } .\tag{3.44}
$$

For us, the invertibility of Γ is important because it is more important empirically that we can extract u from α than vice versa. Consider the form PDE empirical risk

$$
\mathcal { R } _ { \alpha } = \partial _ { t } \alpha _ { \theta } + D \Gamma [ u _ { \theta } ] \big ( \Phi [ u _ { \theta } ] \big ) , \qquad u _ { \theta } = \Gamma ^ { - 1 } [ \alpha _ { \theta } ] .\tag{3.45}
$$

By chain rule (do not yet assume the risk identically vanishes), the risk rewrites

$$
\partial _ { t } \alpha _ { \theta } = D \Gamma [ u _ { \theta } ] ( \partial _ { t } u _ { \theta } )\tag{3.46}
$$

$$
\implies \mathcal { R } _ { \alpha } = D \Gamma [ u _ { \theta } ] ( \partial _ { t } u _ { \theta } ) + D \Gamma [ u _ { \theta } ] ( \Phi [ u _ { \theta } ] )\tag{3.47}
$$

$$
= D \Gamma [ u _ { \theta } ] \left( \partial _ { t } u _ { \theta } + \Phi [ u _ { \theta } ] \right) .\tag{3.48}
$$

The scalar PDE empirical risk can be written $\mathcal { R } _ { u } = \partial _ { t } u _ { \theta } + \Phi [ u _ { \theta } ]$ . Since the operator with its inverse satisfies $\Gamma ^ { - 1 } \circ \Gamma = I$ , by the chain rule for Fr´echet derivatives, and since the derivative of the identity operator is the identity (in general, this is not true, but it is true for Fr´echet derivatives)

$$
D ( \Gamma ^ { - 1 } \circ \Gamma ) [ u _ { \theta } ] = D \Gamma ^ { - 1 } [ \alpha _ { \theta } ] \circ D \Gamma [ u _ { \theta } ] = I .\tag{3.49}
$$

Returning to 3.47 and applying the operator $D \Gamma ^ { - 1 } [ \alpha _ { \theta } ]$ and using 3.49,

$$
D \Gamma ^ { - 1 } [ \alpha _ { \theta } ] ( \mathcal { R } _ { \alpha } ) = D \Gamma ^ { - 1 } [ \alpha _ { \theta } ] \left( D \Gamma [ u _ { \theta } ] ( \mathcal { R } _ { u } ) \right)
$$

$$
\mathit { \Pi } = \mathcal { R } _ { u } .\tag{3.50}
$$

(3.51)

## 4 Training

In this section, we develop general rules for training. There are more guidelines than absolute rules, and oftentimes, the losses are problem-dependent (see experiments in F). We develop the total loss

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } } ( \theta , c ) = \lambda _ { \mathrm { p h y s i c s } } \mathcal { L } _ { \mathrm { p h y s i c s } } + \lambda _ { \mathrm { I C } } \mathcal { L } _ { \mathrm { I C } } + \lambda _ { \mathrm { B C } } \mathcal { L } _ { \mathrm { B C } } , } \end{array}\tag{4.1}
$$

and include boundary data if permissible. The physics loss means the PDE is applied component-wise, and so

$$
\mathcal { L } _ { \mathrm { p h y s i c s } } = \mathbb { E } _ { ( x , t ) \sim \mathcal { U } ( \Omega ^ { * } ) \otimes \mathcal { U } ( [ 0 , T ] ) } \left[ \left. \partial _ { t } \alpha _ { \theta } ^ { \sharp } ( x , t ) + \Psi _ { \alpha } ^ { \sharp } \Big [ x , t , \alpha _ { \theta } , \partial _ { t } \alpha _ { \theta } , \nabla \alpha _ { \theta } , \dots , \nabla ^ { k - 1 } \alpha _ { \theta } , \dots \Big ] \right. _ { 2 } ^ { 2 } \right]\tag{4.2}
$$

$$
\begin{array} { r l } & { = \mathbb { E } _ { ( x , t ) \sim \mathcal { U } ( \Omega ^ { * } ) \otimes \mathcal { U } ( [ 0 , T ] ) } \left[ \displaystyle \sum _ { i = 1 } ^ { d } \left| \partial _ { t } \alpha _ { \theta , i } ( x , t ) + \Psi _ { \alpha , i } \Big [ x , t , \alpha _ { \theta } , \partial _ { t } \alpha _ { \theta } , \nabla \alpha _ { \theta } , \dots , \nabla ^ { k - 1 } \alpha _ { \theta } , \dots \Big ] \right| _ { 2 } ^ { 2 } \right] } \\ & { = \displaystyle \left\| \left\| \partial _ { t } \alpha _ { \theta } ^ { \sharp } ( x , t ) + \Psi _ { \alpha } ^ { \sharp } \Big [ x , t , \alpha _ { \theta } , \partial _ { t } \alpha _ { \theta } , \nabla \alpha _ { \theta } , \dots , \nabla ^ { k - 1 } \alpha _ { \theta } , \dots \Big ] \right\| _ { 2 } ^ { 2 } \right\| _ { L ^ { 1 } ( \Omega ^ { * } \times [ 0 , T ] ; d \mu ) } , } \end{array}\tag{4.3}
$$

(4.4)

for measure $\begin{array} { r } { d \mu = \frac { 1 } { | \Omega ^ { * } | T } d x d t } \end{array}$ . Note that it is possible the PDE is time-independent, so the above is more a generalization rather than a rule that always holds. In particular, the loss is element wise, so we examine

$$
\left\{ \left( \begin{array} { l } { \displaystyle \int \partial _ { t } \alpha _ { \theta , 1 } + \Psi _ { \alpha , 1 } } \\ { \vdots } \\ { \displaystyle \int \partial _ { t } \alpha _ { \theta , d } + \Psi _ { \alpha , d } } \end{array} \right) \left. \right| _ { \displaystyle \phi _ { \theta } \in \Omega ^ { k - 1 } ( M ) , \beta _ { \theta } \in \Omega ^ { k + 1 } ( M ) , h _ { j } \in \mathcal { H } ^ { k } ( M ) } \right\} ,\tag{4.5}
$$

since $\partial _ { t } \alpha _ { \theta } + \Psi [ x , t \alpha _ { \theta } , . . . ] \in \Omega ^ { 1 } ( \Omega ^ { * } )$ , and moreover

$$
\begin{array} { l } { \displaystyle \alpha _ { \theta } ( x , t ) = d \phi _ { \theta } ( x , t ) + \delta \beta _ { \theta } ( x , t ) + \overset { b _ { 1 } } { \sum _ { j = 1 } ^ { \infty } } c _ { \theta , j } ( t ) h _ { j } ( x ) } \\ { \displaystyle \alpha _ { \theta } ( x , t ) ^ { \sharp } = \left( \begin{array} { l } { \displaystyle \alpha _ { \theta , 1 } ( x , t ) } \\ { \displaystyle \alpha _ { \theta , 2 } ( x , t ) } \\ { \vdots } \\ { \displaystyle \alpha _ { \theta , d } ( x , t ) \Big ) } \end{array} \right) = \left( \begin{array} { l } { \displaystyle g ^ { 1 i } ( x ) \left( \partial _ { i } \phi _ { \theta } ( x , t ) + ( \delta \beta _ { \theta } ) _ { i } ( x , t ) + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( t ) ( h _ { j } ) _ { i } ( x ) \right) } \\ { \displaystyle g ^ { 2 i } ( x ) \left( \partial _ { i } \phi _ { \theta } ( x , t ) + ( \delta \beta _ { \theta } ) _ { i } ( x , t ) + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( t ) ( h _ { j } ) _ { i } ( x ) \right) } \\ { \vdots } \\ { \displaystyle g ^ { d i } ( x ) \left( \partial _ { i } \phi _ { \theta } ( x , t ) + ( \delta \beta _ { \theta } ) _ { i } ( x , t ) + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( t ) ( h _ { j } ) _ { i } ( x ) \right) } \end{array} \right) . } \end{array}\tag{4.6}
$$

(4.7)

$\alpha \theta$ is neural network representative of a 1-form. $\sharp : \Omega ^ { 1 } ( M ) \to { \mathfrak { X } } ( M )$ is the sharp musical isomorphism, or alternatively as a bundle isomorphism over the manifold $\sharp : T ^ { * } \Omega ^ { * } \to T \Omega ^ { * }$ (note that $\Omega ^ { * }$ is indeed a manifold). A 1-form is constructed as a linear combination of basis covectors $\begin{array} { r } { \alpha = \sum _ { i = 1 } ^ { d } \alpha _ { i } d x ^ { i } } \end{array}$ . When the PDE operator $\partial _ { t }$ +Ψ acts on the full form $\alpha .$ the output ${ \mathcal { P } } [ \alpha ]$ expands in the same basis, $( \overrightharpoon { \partial _ { t } } + \Psi ) \alpha = \mathrm { r i s k _ { 1 } } d x ^ { 1 } + \cdot \cdot \cdot + \mathrm { r i s k _ { { d } } } d x ^ { d }$ Here, risk<sub>i</sub> represents the residual of the i-th component.

We expand out the physics loss of 4.2 in 2D so we can see what exactly is happening here. The physics loss, with the form substituted in

$$
\begin{array} { l } { \displaystyle { \mathcal { L } _ { \mathrm { p h y s i c s } } =  | | \begin{array} { l } { \displaystyle { 2 } } \\ { \displaystyle { \sum _ { i = 1 } ^ { 2 } } | \partial _ { t } [ \frac { \partial \phi _ { \theta } ( x _ { 1 } , x _ { 2 } , t ) } { \partial x _ { i } } + ( - 1 ) ^ { i + 1 } \frac { \partial \beta _ { \theta } ( x _ { 1 } , x _ { 2 } , t ) } { \partial x _ { i \mathrm { m o d } 2 + 1 } } + \displaystyle { \sum _ { j = 1 } ^ { b _ { 1 } } } c _ { \theta , j } ( t ) h _ { j , i } ( x ) ] | } \\ { \displaystyle { + \Psi _ { \alpha , i } [ x , t , \{ \nabla ^ { k } [ \frac { \partial \phi _ { \theta } ( x _ { 1 } , x _ { 2 } , t ) } { \partial x _ { i } } + ( - 1 ) ^ { i + 1 } \frac { \partial \beta _ { \theta } ( x _ { 1 } , x _ { 2 } , t ) } { \partial x _ { i \mathrm { m o d } 2 + 1 } } + \sum _ { j = 1 } ^ { b _ { 1 } } } c _ { \theta , j } ( t ) h _ { j , i } ( x ) ] \} } _ { k } | ^ { 2 } | \displaystyle { \sum _ { l ^ { 1 } \{ ( \Omega ^ { * } \times [ 0 , T ] \div d _ { \theta } \} } ^ { 1 } \frac { \partial \beta _ { \theta } ( x _ { l } , x _ { l } , t ) } { \partial x _ { l } ( x _ { l } , t ) } + \frac { \partial \beta _ { \theta } ( x _ { l } , t ) } { \partial x _ { l } ( x _ { l } , t ) } } _ { k } | ^ { 2 } , } \end{array} | }  \\  \displaystyle  \displaystyle { + \nabla _ { \alpha , i } [ x , t , \{ x ^ { \prime } , t \} } \{ \nabla ^ { k } [ \frac { \partial \phi _ { \theta } ( x _ { 1 } , x _ { 2 } , t ) } { \partial x _ { i } } \end{array}\tag{4.9}
$$

In practice, it is easier to compute the form first then substitute into the PDE, although the above is exactly suggestive of this but expanded out rather than done in steps. We can also note via 3.24 $h _ { j } \in \mathcal { H } ^ { 1 } ( \Omega ^ { * } )$ is a covector field independent of time and c depends on time, so

$$
\begin{array} { r l } & { H ( x ) = \left( \begin{array} { c c c c } { | } & & { | } & & \\ { h _ { 1 } ^ { \sharp } ( x ) } & { h _ { 2 } ^ { \sharp } ( x ) } & { \ldots } & { h _ { h _ { 1 } } ^ { \sharp } ( x ) } \end{array} \right) \in \mathbb { R } ^ { d \times b _ { 1 } } , \qquad h _ { j } ^ { \sharp } ( x ) = \left( \begin{array} { c } { h _ { j , 1 } ( x ) } \\ { h _ { j , 2 } ( x ) } \\ { \vdots } \\ { h _ { j , d } ( x ) } \end{array} \right) , \qquad c _ { \theta } ( t ) : \mathbb { R } \to \mathbb { R } ^ { b _ { 1 } } , } \end{array}\tag{4.10}
$$

and so

$$
\mathrm { t o p o l o g i c a l ~ t e r m } = \sum _ { j = 1 } ^ { b _ { 1 } } \underbrace { c _ { \theta , j } ( t ) } _ { \mathrm { s c a l a r ~ t i m e - v a r y i n g ~ a m p l i t u d e } } \times \underbrace { h _ { j } ( x _ { 1 } , \dots , x _ { d } ) } _ { \mathrm { s p a t i a l ~ v e c t o r ~ f i e l d } } .\tag{4.11}
$$

Again, $b _ { 1 }$ is the first Betti number.

We develop the initial condition loss. Generally, the boundary loss is in terms of the PDE solution $u \in$ $\Gamma ( \Lambda ^ { 0 } T ^ { * } M )$ , not the neural ansatz 1-form α, but there exists the relationship $\nabla u ( x , t ) = \alpha ( x , t ) ^ { \sharp }$ (or equivalently $\alpha = d u )$ . To elaborate, we can note $( d _ { x } g ) ^ { \sharp } = \nabla _ { x } g$ . Therefore,

$$
\mathcal { L } _ { \mathrm { I C } } = \mathbb { E } _ { x \sim \mathcal { U } ( \Omega ^ { * } ) } \Bigg [ \Bigg \lVert \alpha _ { \theta } ^ { \sharp } ( x , 0 ) - \nabla _ { x } g ( x ) \Bigg \rVert _ { 2 } ^ { 2 } \Bigg ] = \Bigg \lVert \left. \alpha _ { \theta } ^ { \sharp } ( x , 0 ) - \nabla _ { x } g ( x ) \right. _ { 2 } ^ { 2 } \Bigg \rVert _ { L ^ { 1 } ( \Omega ^ { * } ; d \mu ^ { * } ) }\tag{4.12}
$$

We can compute $\alpha _ { \theta } ( x , 0 ) ^ { \sharp }$ with 4.7. In the 2D case, expanding 4.12 with our specific form of 3.18 in a similar format of 4.8,

$$
\mathcal { L } _ { \mathrm { I C } } = | | \sum _ { i = 1 } ^ { 2 } | [ \frac { \partial \phi _ { \theta } ( x _ { 1 } , x _ { 2 } , 0 ) } { \partial x _ { i } } + ( - 1 ) ^ { i + 1 } \frac { \partial \beta _ { \theta } ( x _ { 1 } , x _ { 2 } , 0 ) } { \partial x _ { i \mathrm { m o d } 2 + 1 } } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( 0 ) h _ { j , i } ( x _ { 1 } , x _ { 2 } ) ] - \frac { \partial g ( x _ { 1 } , x _ { 2 } ) } { \partial x _ { i } } | ^ { 2 } | _ { L ^ { 1 } ( \Omega ^ { * } \setminus d \mu ^ { * } ) } .\tag{4.13}
$$

Note that we can also write, and it is equivalent to the above,

$$
\begin{array} { r l } & { { \mathcal L } _ { \mathrm { I C } } = \Bigg | \Bigg | \Bigg | \Bigg | \Bigg | \alpha _ { \theta } ^ { \sharp } ( x _ { 1 } , x _ { 2 } , 0 ) - \nabla _ { x } g ( x _ { 1 } , x _ { 2 } ) \Bigg | \Bigg | _ { 2 } \Bigg | _ { L ^ { 1 } ( \Omega ^ { * } ; d \mu ^ { * } ) } } \\ & { = \Bigg | \Bigg | \Bigg | \left( \frac { \partial \phi _ { \theta } ( x _ { 1 } , x _ { 2 } , 0 ) } { \partial x _ { 1 } } + \frac { \partial \beta _ { \theta } ( x _ { 1 } , x _ { 2 } , 0 ) } { \partial x _ { 2 } } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( 0 ) h _ { j , 1 } ( x _ { 1 } , x _ { 2 } ) \right) - \left( \frac { \partial g ( x _ { 1 } , x _ { 2 } ) } { \partial x _ { 1 } } \right) \Bigg | \Bigg | _ { 2 } ^ { 2 } \Bigg | \Bigg | _ { L ^ { 1 } ( \Omega ^ { * } ; d \mu ^ { * } ) } \ . } \end{array}\tag{4.14}
$$

(4.15)

We develop a boundary loss in 2D, although we remark this loss can take diferent forms, thus we leave it as generalized as possible

$$
\mathcal { L } _ { \mathrm { B C } } = \left. \sum _ { i = 1 } ^ { 2 } \left. \left[ \frac { \partial \phi _ { \theta } ( x _ { 1 } , x _ { 2 } , t ) } { \partial x _ { i } } + ( - 1 ) ^ { i + 1 } \frac { \partial \beta _ { \theta } ( x _ { 1 } , x _ { 2 } , t ) } { \partial x _ { i } \operatorname* { m o d } 2 + 1 } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( t ) h _ { j , i } ( x _ { 1 } , x _ { 2 } ) \right] - B _ { i } ( x _ { 1 } , x _ { 2 } , t ) \right. ^ { 2 } \right. _ { \boldsymbol { L } ^ { 1 } ( \partial \Omega ^ { * } \times [ 0 , T ] ; d \sigma ) }\tag{4.16}
$$

for suitable boundary measure along $\partial \Omega ^ { * }$ , dσ.

Closed-form periodic losses. In certain physical systems, the topological invariants, i.e. the (closed path) circulation we saw in Example 1, can be known. In particular, in Example 1, we can call

$$
\oint _ { \gamma } \alpha - \oint _ { \gamma ^ { \prime } } \alpha = \iint _ { S } d \alpha = 0 \implies \oint _ { \gamma } \alpha = \oint _ { \gamma ^ { \prime } } \alpha = 2 \pi\tag{4.17}
$$

or the constant value $2 \pi$ , to be the topological invariant. This value is called a topological invariant because it is immune to (topological) distortions, stretches, jagged efects, movements, and deformations (continuously) among the path $\gamma .$ . In certain cases, these topological invariants, in settings such as fluid vortices or magnetic

fluxes, can be known a priori. Let us consider developing a topological term in the loss function. Let $q _ { i }$ denote the known invariant (or period) associated with the i-th homology cycle. We add the loss

$$
\mathcal { L } _ { \mathrm { p e r i o d } } = \sum _ { i = 1 } ^ { b _ { 1 } } \left| \oint _ { \gamma _ { i } } \alpha _ { \theta } - q _ { i } \right| ^ { 2 } .\tag{4.18}
$$

Here, $\gamma _ { i }$ is any closed path around the i-th hole. Recall we have 3.35, which we can simplify, and we get

$$
\oint _ { \gamma _ { i } } \alpha _ { \theta } = \oint _ { \gamma _ { i } } d \phi _ { \theta } + \oint _ { \gamma _ { i } } \delta \beta _ { \theta } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } \oint _ { \gamma _ { i } } h _ { j }\tag{4.19}
$$

$$
\stackrel { ! } { = } 0 + 0 + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } \delta _ { i j }\tag{4.20}
$$

$$
{ } = c _ { } \theta , i \cdot\tag{4.21}
$$

We have noted

$$
\oint _ { \gamma _ { i } } h _ { j } = \delta _ { i j } = { \Bigg \{ } 1 , \quad i = j\tag{4.22}
$$

i.e. the harmonic representative evaluates to the Kronecker delta around a closed path. $\oint _ { \gamma _ { i } } d \phi _ { \theta } = 0$ is due to Stokes’ theorem. We crucially remark it must be true that $\oint _ { \gamma _ { i } } \delta \beta _ { \theta } = 0$ in order to isolate the topological invariant, otherwise the above does not hold. Therefore, the loss term of 4.23 evaluates to

$$
\mathcal { L } _ { \mathrm { p e r i o d } } = \sum _ { i = 1 } ^ { b _ { 1 } } \left| c _ { \theta , i } - q _ { i } \right| ^ { 2 } .\tag{4.23}
$$

We leave this mostly as discussion and ultimately do not consider this throughout our experiments.

Closed-form harmonic representatives. The question arises how to choose h. Certainly we desire $\oint _ { \gamma _ { i } } h _ { j } = \delta _ { i j }$ but this does not give us a closed form for what $h _ { j }$ actually is. As a remark, we must have $\Delta \ddot { h } _ { j } = ( d \delta + \delta d ) h _ { j } = 0$ for Hodge Laplacian $\Delta .$ . In 2D space, for a domain with Euclidean metric, we have the representative

$$
h _ { j } = \frac { 1 } { 2 \pi } d \big ( \arg ( x - p _ { j } ) \big ) ,\tag{4.24}
$$

$\boldsymbol { x } = ( x _ { 1 } , x _ { 2 } )$ , and $x - p _ { j } = ( x _ { 1 } - x _ { 1 } ^ { j } ) + i ( x _ { 2 } - x _ { 2 } ^ { j } )$ , which is a 1-form specific to 2D and unique up to boundary conditions. For the j-th puncture, we set using 4.24

$$
h _ { j } = \frac { 1 } { 2 \pi } \left( \frac { - ( x _ { 2 } - x _ { 2 } ^ { j } ) } { ( x _ { 1 } - x _ { 1 } ^ { j } ) ^ { 2 } + ( x _ { 2 } - x _ { 2 } ^ { j } ) ^ { 2 } } d x _ { 1 } + \frac { x _ { 1 } - x _ { 1 } ^ { j } } { ( x _ { 1 } - x _ { 1 } ^ { j } ) ^ { 2 } + ( x _ { 2 } - x _ { 2 } ^ { j } ) ^ { 2 } } d x _ { 2 } \right) ,\tag{4.25}
$$

for punctures $p _ { j } = ( x _ { 1 } ^ { j } , x _ { 2 } ^ { j } )$ . Via sharp musical isomorphism ♯, we write

$$
h _ { j } ^ { \sharp } ( x _ { 1 } , x _ { 2 } ) = \frac { 1 } { 2 \pi } \left( \frac { - ( x _ { 2 } - x _ { 2 } ^ { j } ) } { ( x _ { 1 } - x _ { 1 } ^ { j } ) ^ { 2 } + ( x _ { 2 } - x _ { 2 } ^ { j } ) ^ { 2 } } \right) .\tag{4.26}
$$

Note that Gustafsson (2019) discusses the complex potential

$$
\Psi = \frac { \Gamma _ { k } } { 2 \pi i } \log ( z - p _ { k } ) + \mathrm { r e g u l a r ~ t e r m s } ,\tag{4.27}
$$

(redefining Ψ from earlier), and taking its real part and diferentiating gives $\textstyle { \frac { \Gamma _ { k } } { 2 \pi } } d \arg ( z - p _ { k } )$ $h _ { k }$ is normalized so that $\Gamma _ { k } = 1$ , which is a signed circulation integral evaluation. Note that other relevant references are Gustafsson (2024) Grotta-Ragazzo et al. (2024).

In 3D space, in order to have $b _ { 1 } > 0$ in 3D, the ”puncture” must be an infinite line, curve, or cylinder, i.e. a vortex filament or an infinite wire. If the hole is an infinite cylinder extending along the $x _ { 3 } .$ -axis and centered at $( x _ { 1 } ^ { j } , x _ { 2 } ^ { j } )$ , the harmonic representative is 2D form extended into $\mathbb { R } ^ { 3 }$ with a zero x -component, so

$$
h _ { j } ^ { \sharp } ( x _ { 1 } , x _ { 2 } , x _ { 3 } ) = \frac { 1 } { 2 \pi } \left( \frac { - ( x _ { 2 } - x _ { 2 } ^ { j } ) } { ( x _ { 1 } - x _ { 1 } ^ { j } ) ^ { 2 } + ( x _ { 2 } - x _ { 2 } ^ { j } ) ^ { 2 } } \right) .\tag{4.28}
$$

In general the 3D case is nontrivial, and it must be evaluated on a case-by-case basis.

## 5 Experiments

In all of our experiments, we will use Adam optimizer with initial learning rate $\gamma = 1 \mathrm { e } - 3$ (note that we will always take the learning rate on the $c _ { \theta }$ parameter(s) to be larger; see Figures 11, 12). We will take all networks to have a width of 200. Note that more sophisticated optimizers have shown competitive performance with PINNs, such as second-order optimizers, Kiyani et al. (2025) but our results are primarily motivated by remedying failures of vanilla PINNs and not by achieving as low error as possible. Thus, we find Adam optimizer reasonable for our purposes.

For brevity reasons, we list experimental details in Appendices F and H.

![](images/cda0abd4bc714f6ee27917ed9e567a8410b09c8e201618a06c9e4b7b3a9dfed9.jpg)  
(1)

![](images/65ca9d801c6f10fc85d051b66616742f0f949ee946be096fa5764ed5adff767b.jpg)  
(2)

![](images/efaa07c54c7f1e6af4ad2aea8446e3e637d6b55ffe15cec7f052a4547e8f6871.jpg)  
(3)  
Figure 3: We plot (1) success of a Hodge-de Rham PINN versus (2) the true scalar solution with (3) pointwise absolute error after about ∼ 30, 000 epochs on the Aharonov-Bohm experiment. We found a vanilla MLP worked better without a Fourier feature embedding. We lower the learning rate in late training.

## 6 Conclusions

We developed strategies to circumvent classical pitfalls of PINNs that encompass topological pathologies corresponding to poles. We parameterized 1-forms corresponding to the Hodge-Morrey-Friedrichs decomposition, which subsequently solves as associated PDE relevant to the PDE on the scalar-valued function. We can associate these two systems using an operator mapping, often involving an integral. We developed a theoretical framework establishing this connection in 3.3, also using the Fr´echet derivative. Our experiments were mostly comprehensive. We demonstrated our results on Aharonov–Bohm efect, which is largely nonexistent in PINN literature. Our primary limitation is that our methods are most suitable for low-dimensions due to the diferentiative and combinatorial properties of the coexact term.

![](images/f42392057aa8eef9f80c48ce2be88d7abacff79fc90a5418392bd316779491a9.jpg)  
(1)

![](images/a626f5422d23e873a80f2587f6947cb14719c4e1f53b3968620203514c78c3a4.jpg)  
(2)

![](images/9fb280075f3c159631a9f4dc890e65463e7172d0d67961151c8b0a0166ff020f.jpg)  
(3)  
Figure 4: We plot (1) catastrophic failure of a vanilla PINN versus (2) the true scalar solution with (3) pointwise absolute error after about ∼ 12, 000 epochs on the Aharonov-Bohm experiment.

![](images/65c40adcdb20ba70f976cab33f21611e1aa84b958cba7570ac1cf82636106fb9.jpg)  
(1)

![](images/71040684d5b7e5d53389b4e846e263dd49013d98c537e6d7b6ac5f8300ebdbe0.jpg)  
(2)

![](images/bd211a6fd665468ad57e65b4088c456d33732605c6a94a2118d2bfc2be041151.jpg)  
(3)  
Figure 5: We plot (1) success of a Hodge-de Rham PINN versus (2) the true scalar solution with (3) pointwise absolute error after about ∼ 30, 000 epochs on the Aharonov-Bohm experiment with two poles. We lower the learning rate in late training.

![](images/25926107752167edcb0c0ce38a12dfdbe6fc3ea0ff7e8d9c6c8d1446c8964e47.jpg)  
(1)

![](images/4e6e14c4530e0c9fed5096c3b4f1a6370d34353af91afb95103d44bc11386f4d.jpg)  
(2)

![](images/af54b0c115e6c737954f608e9281c43260db1fea1a053924d212019afc48a303.jpg)  
(3)  
Figure 6: We plot (1) catastrophic failure of a vanilla PINN versus (2) the true scalar solution with (3) pointwise absolute error after about ∼ 12, 000 epochs on the Aharonov-Bohm experiment with two poles.

![](images/5245100883e3783bd8523bd7c55af85e9162c607835afb5d6163ad87817d1030.jpg)  
(1)

![](images/b4df33f3df767954db6a95b5eef98f4e7624955b82fd54cdfe4daf6d2f776ac6.jpg)  
(2)

![](images/af407624ca9a96f529b1d017e5fb44ac4e83d16819a731ad9af81e80f25fa757.jpg)  
(3)  
Figure 7: We plot (1) catastrophic failure of a canonically-developed vector-valued PINN (so the neural network is now vector-valued) versus (2) the true scalar solution with (3) pointwise absolute error after about ∼ 12, 000 epochs on the Aharonov-Bohm experiment with two poles.

## 7 Acknowledgments

I would like to thank Jeremy Miller, Nicholas McCleerey, and Ben McReynolds at Purdue University in the Department of Mathematics for helping make this work possible.

## 8 AI statement

The project was AI-assisted. Both Gemini 3.1 Pro and GPT-6 were used. AI assisted most predominately in helping develop the experiments, which required significant user input. In particular, the code for this research was based on code that already existed, and it was tailored for this work. GPT-6 was able to identify that the Aharonov-Bohm experiment would be a good experiment to run. AI was also used to check grammar and polish small details. Undermind was used to help find references (references were manually entered, but AI was used to help prepare a BibTeX representation if one did not exist).

## References

Qusay S.A. Al-Zamil and James Montaldi. Witten–hodge theory for manifolds with boundary and equivariant cohomology. Diferential Geometry and its Applications, 30(2):179–194, 2012. ISSN 0926-2245. doi: https://doi.org/10.1016/j.difgeo.2011.11.002. URL https://www.sciencedirect.com/science/article/ pii/S0926224511001604.

Lennart Bastian, Samuel Leventhal, Mustafa Hajij, and Tolga Birdal. Topological neural operators, 2026. URL https://arxiv.org/abs/2606.09806.

Harsh Bhatia, Gregory Norgard, Valerio Pascucci, and Peer-Timo Bremer. The Helmholtz-Hodge decomposition-a survey. IEEE Transactions on Visualization and Computer Graphics, 19(8):1386–1404, 2013. doi: 10.1109/TVCG.2012.316.

Raoul Bott and Loring W. Tu. Diferential Forms in Algebraic Topology. Graduate Texts in Mathematics. Springer New York, 1982. doi: 10.1007/978-1-4757-3951-0. URL https://link.springer.com/book/10. 1007/978-1-4757-3951-0.

Lorenzo Brevi, Antonio Mandarino, and Enrico Prati. A tutorial on the use of physics-informed neural networks to compute the spectrum of quantum systems. Technologies, 12(10), 2024. ISSN 2227-7080. doi: 10.3390/technologies12100174. URL https://www.mdpi.com/2227-7080/12/10/174.

Sylvain Cappell, Dennis DeTurck, Herman Gluck, and Edward Y. Miller. Cohomology of harmonic forms on riemannian manifolds with boundary. Forum Mathematicum, 18, 2006. doi: 10.1515/forum.2006.046.

Jiarong Chen, Xia Ji, Haijun Yu, and Shuo Zhang. Penalty-free natural deep ritz method based on de rham complex for high-dimensional dirichlet boundary value problems, 2026. URL https://arxiv.org/abs/ 2607.00676.

Michael Crocoll, Christian D¨oding, Benjamin D¨orich, and Roland Maier. Glenn: Neural network-enhanced computation of ginzburg-landau energy minimizers, 2026. URL https://arxiv.org/abs/2603.19096.

Hans De Raedt and Kristel Michielsen. Algorithm to solve the time-dependent schr¨odinger equation for a charged particle in an inhomogeneous magnetic field: Application to the aharonov–bohm efect. Computer in Physics, 8(5):600–607, 09 1994. ISSN 0894-1866. doi: 10.1063/1.168483. URL https: //doi.org/10.1063/1.168483.

Marta D’Elia, Cynthia Flores, Xingjie Li, Petronela Radu, and Yue Yu. Helmholtz-hodge decompositions in the nonlocal framework. Journal of Peridynamics and Nonlocal Modeling, 2(4):401–418, 2020. ISSN 2522-8978. doi: 10.1007/s42102-020-00035-w. URL https://doi.org/10.1007/s42102-020-00035-w.

Jean-Pierre Demailly. Complex Analytic and Diferential Geometry. Universit´e de Grenoble I, Institut Fourier, Grenoble, France, 2012. URL https://www-fourier.univ-grenoble-alpes.fr/ demailly/ manuscripts/agbook.pdf.

Manfredo P. do Carmo. Diferential Geometry of Curves and Surfaces: Revised and Updated Second Edition. Dover Publications, New York, United States, 2016. ISBN 978-0-486-81797-2. URL http: //ebookcentral.proquest.com/lib/leeds/detail.action?docID=4767649.

P. F. The theory and applications of harmonic integrals. by w v. d. hodge. pp. ix, 281. 21s. 1941. (cambridge). The Mathematical Gazette, 25(267):320–323, 1941. doi: 10.2307/3606574.

K. O. Friedrichs. Diferential forms on riemannian manifolds. Communications on Pure and Applied Mathematics, 8(4):551–590, 1955. doi: https://doi.org/10.1002/cpa.3160080408. URL https: //onlinelibrary.wiley.com/doi/abs/10.1002/cpa.3160080408.

Mathew George. Hodge theorem. Lecture notes. URL https://people.math.osu.edu/george.924/Hodge\_ Theorem.pdf.

Clodoaldo Grotta-Ragazzo, Bj¨orn Gustafsson, and Jair Koiller. On the interplay between vortices and harmonic flows: Hodge decomposition of euler’s equations in 2d. Regular and Chaotic Dynamics, 29(2):241–303, 2024. ISSN 1468-4845. doi: 10.1134/S1560354724020011. URL https://doi.org/10.1134/S1560354724020011.

Bj¨orn Gustafsson. Two-dimensional potential theory with a view toward vortex motion: Energy, capacity and green functions. Geometric Mechanics, 01(04):245–324, 2024. doi: 10.1142/S2972458924300018. URL https://doi.org/10.1142/S2972458924300018.

Bj¨orn Gustafsson. Vortex motion and geometric function theory: the role of connections. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 377(2158):20180341, 09 2019. ISSN 1364-503X. doi: 10.1098/rsta.2018.0341. URL https://doi.org/10.1098/rsta.2018.0341.

H. Helmholtz. Uber integrale der hydrodynamischen gleichungen, welche den wirbelbewegungen entsprechen.,<sup>¨</sup> January 1858. URL https://doi.org/10.1515/crll.1858.55.25.

Elliott G. Holliday, John F. Lindner, and William L. Ditto. Solving two-dimensional quantum eigenvalue problems using physics-informed machine learning, 2023. URL https://arxiv.org/abs/2302.01413.

Kaibo Hu, Jindong Wang, and Jinchao Xu. Relu<sup>k</sup> neural de rham complexes, 2026. URL https://arxiv. org/abs/2607.22478.

Tianhao Hu, Bangti Jin, and Zhi Zhou. Solving poisson problems in polygonal domains with singularity enriched physics informed neural networks. SIAM Journal on Scientific Computing, 46(4):C369–C398, 2024. doi: 10.1137/23M1601195. URL https://doi.org/10.1137/23M1601195.

Yi Huang, Zhiyu Zhang, and Xing Zhang. A direct-forcing immersed boundary method for incompressible flows based on physics-informed neural network. Fluids, 7(2), 2022. ISSN 2311-5521. doi: 10.3390/fluids7020056. URL https://www.mdpi.com/2311-5521/7/2/56.

E. Huguet, J. Queva, and J. Renaud. The schouten–nijenhuis bracket, codiferential of products and generalized interior products of p-forms. International Journal of Geometric Methods in Modern Physics, 23(10): 2550262, 2026. doi: 10.1142/S0219887825502627. URL https://doi.org/10.1142/S0219887825502627.

Aubrey G. Jafer. The lamb-oseen vortex and paint marbling, 2018. URL https://arxiv.org/abs/1810. 04646.

Natan Kaminsky, Daniel Freedman, and Kira Radinsky. Overcoming PINNs failure modes in high dimension with low-rank fourier sum. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=fq3yjso8Tn.

Elham Kiyani, Khemraj Shukla, Jorge F. Urb´an, J´erˆome Darbon, and George Em Karniadakis. Optimizing the optimizer for physics-informed neural networks and kolmogorov-arnold networks. Computer Methods in Applied Mechanics and Engineering, 446:118308, 2025. ISSN 0045-7825. doi: https://doi.org/10.1016/j. cma.2025.118308. URL https://www.sciencedirect.com/science/article/pii/S0045782525005808.

Dongshuai Liu, Wen Zhang, Yanxia Gao, Dianyuan Fan, Boris A. Malomed, and Lifu Zhang. Physicsinformed neural network for nonlinear dynamics of self-trapped necklace beams. Opt. Express, 32(22): 38531–38549, Oct 2024. doi: 10.1364/OE.532126. URL https://opg.optica.org/oe/abstract.cfm? URI=oe-32-22-38531.

Li Liu, Shengping Liu, Hui Xie, Fansheng Xiong, Tengchao Yu, Mengjuan Xiao, Lufeng Liu, and Heng Yong. Discontinuity computing using physics-informed neural networks. Journal of Scientific Computing, 98(1), December 2023. ISSN 1573-7691. doi: 10.1007/s10915-023-02412-1. URL http://dx.doi.org/10.1007/ s10915-023-02412-1.

Marcello Longo, Joost A.A. Opschoor, Nico Disch, Christoph Schwab, and Jakob Zech. De rham compatible deep neural network fem. Neural Networks, 165:721–739, August 2023. ISSN 0893-6080. doi: 10.1016/j. neunet.2023.06.008. URL http://dx.doi.org/10.1016/j.neunet.2023.06.008.

Walther Mayer. Uber abstrakte topologie. <sup>¨</sup> Monatshefte f¨ur Mathematik und Physik, 36(1):1–42, 1929. ISSN 1436-5081. doi: 10.1007/BF02307601. URL https://doi.org/10.1007/BF02307601.

Charles B. Morrey Jr. A variational method in the theory of harmonic integrals. Communications on Pure and Applied Mathematics, 9(3):499–508, 1956. doi: https://doi.org/10.1002/cpa.3160090321. URL https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.3160090321.

Jefrey S. Ovall and Li Zhu. A canonical gauge for computing of eigenpairs of the magnetic schr¨odinger operator, 2025. URL https://arxiv.org/abs/2409.06023.

G¨unter Schwarz. Hodge Decomposition - A Method for Solving Boundary Value Problems. Lecture Notes in Mathematics. Springer Berlin Heidelberg, 1 edition, 1995. ISBN 978-3-540-60016-9. doi: 10.1007/ BFb0095978. URL https://link.springer.com/book/10.1007/BFb0095978.

Muhammad Luthfi Shahab, Fidya Almira Suheri, Rudy Kusdiantara, and Hadi Susanto. Physics-informed neural networks for high-dimensional solutions and snaking bifurcations in nonlinear lattices. Physica D: Nonlinear Phenomena, 481:134836, 2025. ISSN 0167-2789. doi: https://doi.org/10.1016/j.physd.2025.134836. URL https://www.sciencedirect.com/science/article/pii/S0167278925003136.

Clayton Shonkwiler. Poincare duality angles for riemannian manifolds with boundary, 2009. URL https: //arxiv.org/abs/0909.1967.

Zhe Su, Yiying Tong, and Guo-Wei Wei. Persistent de rham-hodge laplacians in eulerian representation for manifold topological learning. AIMS Mathematics, 9(10):27438–27470, 2024a. doi: 10.3934/math.20241333.

Zhe Su, Yiying Tong, and Guowei Wei. Hodge decomposition of vector fields in cartesian grids. In SIGGRAPH Asia 2024 Conference Papers, SA ’24, New York, NY, USA, 2024b. Association for Computing Machinery. ISBN 9798400711312. doi: 10.1145/3680528.3687602. URL https://doi.org/10.1145/3680528.3687602.

Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546.

Sifan Wang, Shyam Sankaran, Hanwen Wang, and Paris Perdikaris. An expert’s guide to training physicsinformed neural networks. CoRR, abs/2308.08468, 2023. URL https://doi.org/10.48550/arXiv.2308. 08468.

Frank W. Warner. Foundations of Diferentiable Manifolds and Lie Groups, volume 94 of Graduate Texts in Mathematics. Springer, New York, NY, 1983. doi: 10.1007/978-1-4757-1799-0. URL https://link. springer.com/book/10.1007/978-1-4757-1799-0.

Raymond O. Wells. Diferential Analysis on Complex Manifolds, volume 65 of Graduate Texts in Mathematics. Springer, New York, NY, 1980. doi: 10.1007/978-1-4757-3946-6. URL https://link.springer.com/ book/10.1007/978-1-4757-3946-6.

Hermann Weyl. The method of orthogonal projection in potential theory. Duke Mathematical Journal, 7(1):411 – 444, 1940. doi: 10.1215/S0012-7094-40-00725-6. URL https://doi.org/10.1215/ S0012-7094-40-00725-6.

Xiaofeng Wu, Ziling Wu, Sibi Chakravarthy Shanmugavel, Hang Z. Yu, and Yunhui Zhu. Physics-informed neural network for phase imaging based on transport of intensity equation. Opt. Express, 30(24):43398– 43416, Nov 2022. doi: 10.1364/OE.462844. URL https://opg.optica.org/oe/abstract.cfm?URI= oe-30-24-43398.

Shing-Tung Yau. Calabi’s conjecture and some new results in algebraic geometry. Proceedings of the National Academy of Sciences, 74(5):1798–1799, 1977. doi: 10.1073/pnas.74.5.1798. URL https://doi.org/10. 1073/pnas.74.5.1798.

ChuYu Zhou, Tianyu Li, Chenxi Lan, Rongyu Du, Guoguo Xin, Wei Li, Guoqing Wang, Xun Liu, and Hangzhou Yang. Hybrid boundary physics-informed neural networks for solving navier-stokes equations with complex boundary. In Advances in Neural Information Processing Systems 38, pages 152339–152375. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2025. doi: 10.52202/085713-4583.

<table><tr><td>Symbol</td><td>Description</td></tr><tr><td>α</td><td>1-form (smooth section of the cotangent bundle)</td></tr><tr><td>Ω</td><td>Domain/manifold without puncture</td></tr><tr><td>Ω*</td><td>Domain/manifold with puncture(s)</td></tr><tr><td>Φ</td><td>Differential operator of the scalar PDE</td></tr><tr><td>Ψ</td><td>Differential operator of the PDE on the 1-form</td></tr><tr><td>[α]</td><td>Cohomology class containing α</td></tr><tr><td> $H _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } )$ </td><td>First de Rham cohomology group</td></tr><tr><td> $\Lambda ^ { 1 } T ^ { * } ( \Omega ^ { * } )$ </td><td>Bundle of differential 1-forms</td></tr><tr><td> $\Omega ^ { 1 } ( \Omega ^ { * } )$ </td><td>Smooth 1-forms</td></tr><tr><td> $\Gamma ( \Omega ^ { * } , \Lambda ^ { 1 } T ^ { * } \Omega ^ { * } )$ </td><td>Smooth sections of the cotangent bundle</td></tr><tr><td></td><td>Weight/exhaustion function (level sets are compact)</td></tr><tr><td>dα</td><td>Exterior derivative of α</td></tr><tr><td>δβ</td><td>Coexact term  $( - 1 ) ^ { n ( k - 1 ) + 1 }$  *d*β</td></tr><tr><td>C</td><td>Learnable scalar corresponding to harmonic representative</td></tr><tr><td> $h$ </td><td>Harmonic representative</td></tr><tr><td> $b _ { 1 }$ </td><td>First Betti number</td></tr><tr><td> $H ^ { k } ( M ; \Lambda ^ { p } T ^ { * } M )$ </td><td>Sobolev space for differential p-forms</td></tr><tr><td>Γ</td><td>Operator mapping from scalar PDE solution u to 1-form α</td></tr><tr><td> $\Delta _ { H }$ </td><td>Hodge-de Rham Laplacian  $d \delta + \delta d$ </td></tr><tr><td>甘</td><td>Sharp musical isomorphism  $\Omega ^ { 1 } ( M ) \to { \mathfrak { X } } ( M )$ </td></tr><tr><td>★</td><td> $\Omega ^ { k } ( M ) \ { \stackrel { \sim } { \longrightarrow } } \ \Omega ^ { d - k } ( M )$  (sometimes true solution)</td></tr><tr><td>x(M)</td><td>Smooth sections/vector fields on M (alternative to Γ(TM)) do Carmo (2016)</td></tr></table>

## A Theoretical discussion

Theorem 1. Let $M = \Omega ^ { * }$ be a compact, oriented, smooth Riemannian manifold. Let $\alpha ^ { \star } = d \phi ^ { \star } + \delta \beta ^ { \star } + h ^ { \star } \in$ $L ^ { 2 } ( \Lambda ^ { 1 } M )$ be the true diferential form corresponding to a PDE system, i.e. $\alpha ^ { \star } = u ^ { \flat }$ where u is the true PDE solution associated with the form (not the scalar solution) and ♭: $T \Omega ^ { * } \to T ^ { * } \Omega ^ { * }$ is the flat musical isomorphism. Moreover, $d \phi ^ { \star } , \delta \beta ^ { \star }$ , and $h ^ { \star }$ are the exact, coexact, and harmonic components given by the Hodge-Morrey-Friedrichs (HMF) decomposition. $I f$ the PINN parameterizes an exact form, $\alpha _ { \theta } = d \phi _ { \theta }$ , then the $L ^ { 2 }$ approximation error is bounded below such that

$$
\operatorname* { i n f } _ { \theta } \| \alpha ^ { \star } - \alpha _ { \theta } \| _ { L ^ { 2 } } ^ { 2 } \geq \| \delta \beta ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } + \| h ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } .\tag{A.1}
$$

Proof. For a diferential form $\alpha \in L ^ { 2 } ( \Lambda ^ { k } M )$ on a compact manifold with boundary, the Hodge-Morrey-Friedrichs decomposition gives

$$
L ^ { 2 } ( \Lambda ^ { k } M ) = \mathcal { E } ^ { k } \oplus \mathcal { C } ^ { k } \oplus \mathcal { H } _ { A } ^ { k } .\tag{A.2}
$$

Here, ${ \mathcal { E } } ^ { k } = d ( \Omega ^ { k - 1 } ( M ) )$ is the space of exact forms, $\mathcal { C } ^ { k } = \delta ( \Omega _ { N } ^ { k + 1 } ( M ) )$ is the space of coexact forms with Neumann boundary conditions, and $\mathcal { H } _ { A } ^ { k }$ is the space of absolute harmonic forms. For a coexact form $\delta \beta ^ { \star } \in { \mathcal { C } } ^ { 1 }$ the relative boundary condition implies its normal component vanishes, meaning the pullback of its Hodge star is zero, i.e., $\iota _ { \partial M } ^ { * } ( { \star } \omega ) = 0$

We use notation

$$
L ^ { 2 } ( \Lambda ^ { k } M ) : = \left\{ \alpha { \mathrm { ~ i s ~ a ~ m e a s u r a b l e ~ } } k { \mathrm { - f o r m } } \biggm | \| \alpha \| _ { L ^ { 2 } } ^ { 2 } : = \int _ { \Omega ^ { * } } \alpha \wedge \star \alpha < \infty \right\} ,\tag{A.3}
$$

where ⋆ : $\Lambda ^ { k } \to \Lambda ^ { d - k }$ , and

$$
H ( d , \Lambda ^ { k - 1 } ) = \Big \{ \phi \in L ^ { 2 } ( \Lambda ^ { k - 1 } ) \Big | d \phi \in L ^ { 2 } ( \Lambda ^ { k } ) \Big \} .\tag{A.4}
$$

Here, $d \phi$ is the weak exterior derivative, defined via integration by parts, meaning there exists some $\eta \in L ^ { 2 } ( \Lambda ^ { k } )$ such that for all smooth, compactly supported test forms $\psi \in \mathring { \Omega } _ { c } ^ { \bar { d } - k + 1 } ( M )$ ，

$$
\int _ { \Omega ^ { * } } \eta \wedge \psi = ( - 1 ) ^ { k } \int _ { \Omega ^ { * } } \phi \wedge d \psi .\tag{A.5}
$$

When we write $d ( H ^ { k - 1 } )$ , we define the space of exact $L ^ { 2 }$ forms. Similarly, $\delta ( H ^ { k + 1 } )$ is the space of coexact $L ^ { 2 }$ forms, representing the image of the weak codiferential acting on $( k + 1 ) – \mathrm { f o r m s } \ \beta$ for which $\delta \beta \in L ^ { 2 } ( \Lambda ^ { k } )$

Let us consider the case of a scalar-valued neural network, or vanilla PINN, which means that $\alpha _ { \theta } = d \phi _ { \theta }$ is our learned form. Let us quantify the $L ^ { 2 }$ error from the true PDE form $\alpha ^ { \star } = d \phi ^ { \star } + \delta \beta ^ { \star } + h ^ { \star }$ . Recall the true PDE form cannot be expressed in terms of the vanilla PINN, which we saw in 1. The $L ^ { 2 }$ norm squared gives

$$
\| \alpha ^ { \star } - \alpha _ { \theta } \| _ { L ^ { 2 } } ^ { 2 } = \langle d ( \phi ^ { \star } - \phi _ { \theta } ) + \delta \beta ^ { \star } + h ^ { \star } , d ( \phi ^ { \star } - \phi _ { \theta } ) + \delta \beta ^ { \star } + h ^ { \star } \rangle _ { L ^ { 2 } } .\tag{A.6}
$$

Let us attempt to evaluate the inner product by distributing,

$$
\langle d ( \phi ^ { \star } - \phi _ { \theta } ) + \delta \beta ^ { \star } + h ^ { \star } , d ( \phi ^ { \star } - \phi _ { \theta } ) + \delta \beta ^ { \star } + h ^ { \star } \rangle _ { L ^ { 2 } }\tag{A.7}
$$

$$
= \langle d ( \phi ^ { \star } - \phi _ { \theta } ) , d ( \phi ^ { \star } - \phi _ { \theta } ) \rangle _ { L ^ { 2 } } + \langle \delta \beta ^ { \star } , \delta \beta ^ { \star } \rangle _ { L ^ { 2 } } + \langle h ^ { \star } , h ^ { \star } \rangle _ { L ^ { 2 } }\tag{A.8}
$$

$$
+ 2 \underbrace { \langle d ( \phi ^ { \star } - \phi _ { \theta } ) , \delta \beta ^ { \star } \rangle _ { L ^ { 2 } } } _ { = 0 } + 2 \underbrace { \langle d ( \phi ^ { \star } - \phi _ { \theta } ) , h ^ { \star } \rangle _ { L ^ { 2 } } } _ { = 0 } + 2 \underbrace { \langle \delta \beta ^ { \star } , h ^ { \star } \rangle _ { L ^ { 2 } } } _ { = 0 }\tag{A.9}
$$

$$
= \| d ( \phi ^ { \star } - \phi _ { \theta } ) \| _ { L ^ { 2 } } ^ { 2 } + \| \delta \beta ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } + \| h ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } .\tag{A.10}
$$

We have noted symmetry of the inner product, thus the terms with 2 appear. In particular, over the rea numbers, the inner product is symmetric, and in general obeys $\langle \phi , \beta \rangle _ { L ^ { 2 } } = \overline { { \langle \beta , \phi \rangle } } _ { L ^ { 2 } }$

Because $\| d ( \phi ^ { \star } - \phi _ { \theta } ) \| _ { L ^ { 2 } } ^ { 2 } \geq 0$ , in a neural network setting, we get the lower bound

$$
\operatorname* { i n f } _ { \theta } \| \alpha ^ { \star } - d \phi _ { \theta } \| _ { L ^ { 2 } } ^ { 2 } \geq \| \delta \beta ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } + \| h ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } ,\tag{A.11}
$$

which guarantees a scalar PINN $u _ { \theta } \to \mathbb { R }$ cannot optimize suficiently well. In A.9, we have noted:

I. of the terms evaluating to zero, the first term evaluates to (via integration by parts/an adjoint)

$$
\langle d ( \phi ^ { \star } - \phi _ { \theta } ) , \delta \beta ^ { \star } \rangle _ { L ^ { 2 } } = \langle \phi ^ { \star } - \phi _ { \theta } , \delta ^ { 2 } \beta ^ { \star } \rangle _ { L ^ { 2 } } + \int _ { \partial M } ( \phi ^ { \star } - \phi _ { \theta } ) \wedge { \star \delta \beta ^ { \star } } ,\tag{A.12}
$$

and applying the codiferential twice yields zero, i.e. $\delta ^ { 2 } = 0$ . Furthermore, $\star \delta \beta ^ { \star } | _ { \partial { \cal M } } = 0$ , so the boundary term is zero;

II. the second term annihilates since

$$
\langle d ( \phi ^ { \star } - \phi _ { \theta } ) , h ^ { \star } \rangle _ { L ^ { 2 } } = \langle \phi ^ { \star } - \phi _ { \theta } , \delta h ^ { \star } \rangle _ { L ^ { 2 } } + \int _ { \partial { \cal M } } ( \phi ^ { \star } - \phi _ { \theta } ) \wedge \star h ^ { \star } .\tag{A.13}
$$

Because $h ^ { \star }$ is an absolute harmonic form, it is co-closed, meaning $\delta h ^ { \star } = 0$ . Also, its normal trace vanishes on the boundary, so $\star h ^ { \star } | _ { \partial M } = 0$ . Thus, the integral is annihilated;

III. the third term becomes

$$
\langle \delta \beta ^ { \star } , h ^ { \star } \rangle _ { L ^ { 2 } } = \langle \beta ^ { \star } , d h ^ { \star } \rangle _ { L ^ { 2 } } - \int _ { \partial { \cal M } } h ^ { \star } \wedge \star \beta ^ { \star } .\tag{A.14}
$$

Because $h ^ { \star }$ is a harmonic form, it is closed, meaning $d h ^ { \star } = 0$ . Furthermore, since $\beta ^ { \star } \in \Omega _ { N } ^ { k + 1 } ( M )$ satisfies absolute boundary conditions, its normal trace vanishes and $\star \beta ^ { \star } | _ { \partial { \cal M } } = 0$ , meaning the pullback to ∂M is zero and the boundary term vanishes.

□

Our next theorem is built from PINN hypothesis classes. We do not use the standard Hodge decomposition here because we must account for boundary. Instead, let us define the hypothesis classes of ascending expressivity

$$
\mathcal { V } _ { 0 } : = d ( \Omega ^ { 0 } ( M ) )\tag{A.15}
$$

$$
\mathcal { V } _ { 1 } : = d ( \Omega ^ { 0 } ( M ) ) + \mathcal { H } _ { A } ^ { 1 } ( M )\tag{A.16}
$$

$$
\mathcal { V } _ { 2 } : = d ( \Omega ^ { 0 } ( M ) ) + \delta ( \Omega ^ { 2 } ( M ) ) + \mathcal { H } _ { A } ^ { 1 } ( M ) .\tag{A.17}
$$

These above of A.15 does not use direct sums is order to account for boundary; however, they fully account for possible 1-forms (they are expressive), or at least among their respective classes.

Theorem 2. Let $M = \Omega ^ { * }$ be a compact, oriented, smooth Riemannian manifold with boundary ∂M. Let $\alpha ^ { \star } \in L ^ { 2 } ( \Lambda ^ { 1 } M )$ be the true PDE form, and let $\alpha ^ { \star } = d \phi ^ { \star } + \delta \beta ^ { \star } + h ^ { \star }$ be its unique, orthogonal HMF decomposition. We have the following bounds for the unconstrained neural network hypothesis classes of the spaces of A.15:

$$
\left\{ \begin{array} { l l } { \operatorname* { i n f } _ { \alpha \in \mathcal { V } _ { 0 } } \| \alpha ^ { \star } - \alpha \| _ { L ^ { 2 } } ^ { 2 } = \| \delta \beta ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } + \| h ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } , } \\ { \quad } \\ { \operatorname* { i n f } _ { \alpha \in \mathcal { V } _ { 1 } } \| \alpha ^ { \star } - \alpha \| _ { L ^ { 2 } } ^ { 2 } = \| \delta \beta ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } , } \\ { \quad } \\ { \operatorname* { i n f } _ { \alpha \in \mathcal { V } _ { 2 } } \| \alpha ^ { \star } - \alpha \| _ { L ^ { 2 } } ^ { 2 } = 0 . } \end{array} \right.\tag{A.18}
$$

Remark. It is correct to say the true form obeys an orthogonal decomposition. It is not correct to say the neural network ansatz form obeys an orthogonal decomposition (this is why we use algebraic sum notation in 1.5).

Proof. By the uniqueness of the orthogonal HMF decomposition on the true PDE form, its components satisfy the boundary conditions $\star \delta \beta ^ { \star } | _ { \partial { \cal M } } = 0$ (absolute boundary conditions) and $\star h ^ { \star } | _ { \partial M } = 0$ (absolute boundary conditions).

Now, take $\alpha \in \mathcal { V } _ { 0 }$ , so that $\alpha = d \phi$ for some admissible 0-form $\phi .$ . Just as before in $\mathrm { A } . 7 ,$ we can write by simplifying the inner products

$$
\| \alpha ^ { \star } - \alpha \| _ { L ^ { 2 } } ^ { 2 } = \| d ( \phi ^ { \star } - \phi ) \| _ { L ^ { 2 } } ^ { 2 } + \| \delta \beta ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } + \| h ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } .\tag{A.19}
$$

As before, as in Theorem 1, the cross-terms vanish. Since $\| d ( \phi ^ { \star } - \phi ) \| _ { L ^ { 2 } } ^ { 2 } \geq 0$ , and equality is attained by choosing $\alpha = d \phi ^ { \star }$ , we obtain

$$
\operatorname* { i n f } _ { \alpha \in \mathcal { V } _ { 0 } } \| \alpha ^ { \star } - \alpha \| _ { L ^ { 2 } } ^ { 2 } = \| \delta \beta ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } + \| h ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } .\tag{A.20}
$$

This strengthens Theorem 1 by replacing the lower bound with an equality when optimization is over the specific hypothesis class.

Now, we consider the hypothesis space $\mathcal { V } _ { 1 } = d ( \Omega ^ { 0 } ( M ) ) + \mathcal { H } _ { A } ^ { 1 } ( M )$ . An arbitrary element of this space can be expressed as $\alpha = d \phi + h$ . Repeating an inner product argument as in $_ \mathrm { A . 7 } .$ , we first note

$$
\alpha ^ { \star } - \alpha = d ( \phi ^ { \star } - \phi ) + \delta \beta ^ { \star } + ( h ^ { \star } - h ) ,\tag{A.21}
$$

and it follows

$$
\begin{array} { r l } & { \| \alpha ^ { \star } - \alpha \| _ { L ^ { 2 } } ^ { 2 } = \langle d ( \phi ^ { \star } - \phi ) + \delta \beta ^ { \star } + ( h ^ { \star } - h ) , d ( \phi ^ { \star } - \phi ) + \delta \beta ^ { \star } + ( h ^ { \star } - h ) \rangle _ { L ^ { 2 } } } \\ & { \qquad = \| d ( \phi ^ { \star } - \phi ) \| _ { L ^ { 2 } } ^ { 2 } + \| \delta \beta ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } + \| h ^ { \star } - h \| _ { L ^ { 2 } } ^ { 2 } } \\ & { \qquad + 2 \underbrace { \langle d ( \phi ^ { \star } - \phi ) , \delta \beta ^ { \star } \rangle _ { L ^ { 2 } } } _ { = 0 } + 2 \underbrace { \langle d ( \phi ^ { \star } - \phi ) , h ^ { \star } - h \rangle _ { L ^ { 2 } } } _ { = 0 } + 2 \underbrace { \langle \delta \beta ^ { \star } , h ^ { \star } - h \rangle _ { L ^ { 2 } } } _ { = 0 } } \\ & { \qquad = \| d ( \phi ^ { \star } - \phi ) \| _ { L ^ { 2 } } ^ { 2 } + \| \delta \beta ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } + \| h ^ { \star } - h \| _ { L ^ { 2 } } ^ { 2 } . } \end{array}\tag{A.22}
$$

Again, cross-terms vanish. Therefore, optimizing over the hypothesis space, the exact and harmonic norms annihilate

$$
\operatorname* { i n f } _ { \alpha \in \mathcal { V } _ { 1 } } \| \alpha ^ { \star } - \alpha \| _ { L ^ { 2 } } ^ { 2 } = \| \delta \beta ^ { \star } \| _ { L ^ { 2 } } ^ { 2 } .\tag{A.23}
$$

Let us consider now the full hypothesis class $\nu _ { 2 }$ , and consider the network ansatz $\alpha = d \phi + \delta \beta + h$ . Note that an arbitrary $\alpha \in { \nu } _ { 2 }$ does not necessarily possess orthogonal components since the neural network does not correspond to the orthogonal HMF decomposition exactly, i.e. boundary terms are omitted; however, it is true the ansatz components $\phi ^ { \star } , \beta ^ { \star } , h ^ { \star }$ lie within the support of the universal approximator. If we consider the true form learned exactly $\phi = \phi ^ { \star } , \beta = \beta ^ { \star }$ , and $h = h ^ { \star }$ , the diference $\alpha ^ { \star } - \alpha$ vanishes. Thus, since it is possible the neural network learns the true form by expressivity,

$$
\operatorname* { i n f } _ { \alpha \in \mathcal { V } _ { 2 } } \| \alpha ^ { \star } - \alpha \| _ { L ^ { 2 } } ^ { 2 } = 0 .\tag{A.24}
$$

□

Remark. We can parameterize networks within the hypothesis classes of $\mathrm { A . 7 }$ via

$$
\begin{array} { r l } { \mathcal { V } _ { 0 } : } & { { } \ \omega _ { \theta } = d \phi _ { \theta } } \end{array}
$$

$$
\begin{array} { r l } { \mathcal { V } _ { 1 } : } & { { } \ \omega _ { \theta } = d \phi _ { \theta } + h _ { \theta } } \end{array}\tag{A.25}
$$

(A.26)

$$
\mathcal { V } _ { 2 } : ~ \omega _ { \theta } = d \phi _ { \theta } + \delta \beta _ { \theta } + h _ { \theta } .\tag{A.27}
$$

Theorem 3. Let $M = \Omega ^ { * } \subseteq \mathbb { R } ^ { d }$ be a smooth domain, and let $\gamma _ { i } : S ^ { 1 } \to M$ be a fixed smooth embedded closed curve representing a homology generator. Let the neural PINN form $\alpha \theta$ and the true diferential 1-form $\alpha ^ { \star }$ reside in the Sobolev space ${ \dot { H } } ^ { s } ( M ; \Lambda ^ { 1 } )$ . Suppose we have Sobolev regularity $\textstyle s > { \frac { d - 1 } { 2 } }$ . Then there exists a constant $\widetilde { C } _ { i } > 0$ such that the closed line error between the neural PINN form $\alpha \theta$ and the true form $\alpha ^ { \star }$ obeys

$$
\left| \oint _ { \gamma _ { i } } \alpha _ { \theta } - \oint _ { \gamma _ { i } } \alpha ^ { \star } \right| \leq \widetilde { C } _ { i } \| \alpha _ { \theta } - \alpha ^ { \star } \| _ { H ^ { s } ( M ; \Lambda ^ { 1 } ) } .\tag{A.28}
$$

Proof. Both the true and neural network-approximated forms reside in the Sobolev space $H ^ { s } ( M ; \Lambda ^ { 1 } )$ . In local coordinates, the error form can be written component-wise as

$$
\alpha _ { \theta } - \alpha ^ { \star } = \sum _ { j = 1 } ^ { d } ( \alpha _ { \theta , j } - \alpha _ { j } ^ { \star } ) d x ^ { j } ,\tag{A.29}
$$

where the component error $( \alpha _ { \theta , j } - \alpha _ { j } ^ { \star } )$ belongs to the scalar Sobolev space $H ^ { s } ( M )$ . Let $\gamma _ { i } : S ^ { 1 } \to M$ be the fixed smooth embedded closed curve representing the homology generator. In particular, we will examine a map

$$
( [ \alpha ] , [ \gamma ] ) \longmapsto \oint _ { \gamma } \alpha .\tag{A.30}
$$

Now, we will examine the pullback, which can be written in terms of the trace operator, which requires

$$
s > \frac { d - 1 } { 2 }\tag{A.31}
$$

to be arbitrary. In particular, we can write via a pullback

$$
\gamma _ { i } ^ { * } ( \alpha _ { \theta } - \alpha ^ { \star } ) = \sum _ { j = 1 } ^ { d } \operatorname { T r } _ { \gamma _ { i } } ( \alpha _ { \theta , j } - \alpha _ { j } ^ { \star } ) \frac { d \gamma _ { i } ^ { j } } { d t } d t ,\tag{A.32}
$$

for all $\textstyle s > { \frac { d - 1 } { 2 } }$ , where $\mathrm { T r } _ { \gamma _ { i } } : H ^ { s } ( { \cal M } )  H ^ { s - \frac { d - 1 } { 2 } } ( \gamma _ { i } )$ is the trace operator. The for all argument is because Sobolev spaces are nested, $H ^ { s _ { 2 } } \subseteq H ^ { s _ { 1 } }$ whenever $s _ { 2 } > s _ { 1 }$ , so meeting the minimum threshold is the only requirement. Due to our choice of $s ,$ the trace operator is bounded, so there exists a constant $C _ { \mathrm { t r a c e } , i } > 0$ such that for each component of the error

$$
\| \operatorname { T r } _ { \gamma _ { i } } ( \alpha _ { \theta , j } - \alpha _ { j } ^ { \star } ) \| _ { L ^ { 2 } ( \gamma _ { i } ) } \leq C _ { \mathrm { t r a c e } , i } \| \alpha _ { \theta , j } - \alpha _ { j } ^ { \star } \| _ { H ^ { s } ( M ) } ,\tag{A.33}
$$

since, under this condition, the trace operator $\mathrm { T r } _ { \gamma _ { i } }$ is bounded.

Now, because $\gamma _ { i }$ is a fixed smooth embedding, its tangent vector components $\frac { d \gamma _ { i } ^ { j } } { d t }$ are uniformly bounded. Let $\begin{array} { r } { L _ { i } = \operatorname* { m a x } _ { j , t } \left| \frac { d \gamma _ { i } ^ { j } } { d t } \right| } \end{array}$ . Let us now bound the error of the line integral in the network versus true form. Observe

$$
\left| \oint _ { \gamma _ { i } } \alpha _ { \theta } - \oint _ { \gamma _ { i } } \alpha ^ { \star } \right| = \left| \int _ { S ^ { 1 } } \gamma _ { i } ^ { \ast } ( \alpha _ { \theta } - \alpha ^ { \star } ) \right|\tag{A.34}
$$

$$
= \left| \int _ { S ^ { 1 } } \sum _ { j = 1 } ^ { d } \operatorname { T r } _ { \gamma _ { i } } ( \alpha _ { \theta , j } - \alpha _ { j } ^ { \star } ) \frac { d \gamma _ { i } ^ { j } } { d t } d t \right|\tag{A.35}
$$

$$
\leq \sum _ { j = 1 } ^ { d } \int _ { S ^ { 1 } } | \operatorname { T r } _ { \gamma _ { i } } ( \alpha _ { \theta , j } - \alpha _ { j } ^ { \star } ) | | | { \frac { d \gamma _ { i } ^ { j } } { d t } } | d t\tag{A.36}
$$

$$
\leq L _ { i } \sum _ { j = 1 } ^ { d } \int _ { S ^ { 1 } } \left| \operatorname { T r } _ { \gamma _ { i } } ( \alpha _ { \theta , j } - \alpha _ { j } ^ { \star } ) \right| \cdot 1 d t\tag{A.37}
$$

$$
\leq L _ { i } \sum _ { j = 1 } ^ { d } \left( \int _ { S ^ { 1 } } \left| \operatorname { T r } _ { \gamma _ { i } } ( \alpha _ { \theta , j } - \alpha _ { j } ^ { \star } ) \right| ^ { 2 } d t \right) ^ { 1 / 2 } \left( \int _ { S ^ { 1 } } 1 ^ { 2 } d t \right) ^ { 1 / 2 }\tag{A.38}
$$

$$
= L _ { i } \sum _ { j = 1 } ^ { d } | S ^ { 1 } | ^ { 1 / 2 } \left. \mathrm { T r } _ { \gamma _ { i } } ( \alpha _ { \theta , j } - \alpha _ { j } ^ { \star } ) \right. _ { L ^ { 2 } ( \gamma _ { i } ) }\tag{A.39}
$$

$$
\leq L _ { i } | S ^ { 1 } | ^ { 1 / 2 } C _ { \mathrm { t r a c e } , i } \sum _ { j = 1 } ^ { d } \| \alpha _ { \theta , j } - \alpha _ { j } ^ { \star } \| _ { H ^ { s } ( M ) }\tag{A.40}
$$

$$
\leq L _ { i } | S ^ { 1 } | ^ { 1 / 2 } C _ { \mathrm { t r a c e } , i } \sqrt { d } \left\| \alpha _ { \theta } - \alpha ^ { \star } \right\| _ { H ^ { s } ( M ; \Lambda ^ { 1 } ) }
$$

$$
= \widetilde { C } _ { i } \| \alpha _ { \theta } - \alpha ^ { \star } \| _ { H ^ { s } ( M ; \Lambda ^ { 1 } ) } .\tag{A.41}
$$

(A.42)

Remark. We are using the norms

I.

$$
\| f \| _ { L ^ { 2 } ( \gamma _ { i } ) } = \left( \int _ { S ^ { 1 } } | f ( \gamma _ { i } ( t ) ) | ^ { 2 } d t \right) ^ { 1 / 2 } ;\tag{A.43}
$$

II.

$$
\| u \| _ { H ^ { k } ( M ) } = \left( \sum _ { | \beta | \leq k } \int _ { M } | D ^ { \beta } u | ^ { 2 } d V \right) ^ { 1 / 2 }\tag{A.44}
$$

(this norm is applicable on the components of a form);

III.

$$
\| \alpha \| _ { H ^ { s } ( M ; \Lambda ^ { 1 } ) } = \left( \sum _ { j = 1 } ^ { d } \| \alpha _ { j } \| _ { H ^ { s } ( M ) } ^ { 2 } \right) ^ { 1 / 2 } .\tag{A.45}
$$

![](images/026bf5f41c1af3264420c83b42def069254c4838ed2a652c218aa929a37ab226.jpg)  
Figure 8: We plot that Theorem 3 holds on the unit radial vortex experiment with simple boundary condition. We plot a log-log plot demonstrating the bound of $\begin{array} { r } { \left| \oint _ { \gamma _ { i } } \alpha _ { \theta } - \oint _ { \gamma _ { i } } \alpha ^ { \star } \right| \leq \widetilde { C } _ { i } \| \alpha _ { \theta } - \alpha ^ { \star } \| _ { H ^ { s } ( M ; \Lambda ^ { 1 } ) } } \end{array}$ holds. We plot the errors across three distinct closed curves $\gamma _ { i } ,$ which are concentric circles with radii $r \in \{ 1 . 1 , 1 . 5 , 1 . 9 \}$ with respect to 8,000 training epochs. The x-axis is the $H ^ { s }$ form error and the y-axis is the path circulation error. The shaded region represents the desirable region such that the bound holds. The points are clustered in the low-error regime since this is where the training spends most of its time (loss converges quickly then slowly).

## B Additional discussion

Definition (informal). We say a space is contractible if it can be continuously shrunk to a point within itself, without tearing, breaking, or leaving the space. For example, a circle is not contractible because if it shrinks to a singular point, the circle empties the empty space that once existed in that hole.

In the case that the punctures $\{ p _ { j } \} _ { j }$ have zero measure or nonzero measure and correspond to disjoint, compact, contractible sets $\{ K _ { j } \} _ { j }$ , we get an isomorphism (between modules/vector spaces, so a bijective homomorphism; with respect to the field of real numbers $\mathbb { R } ;$ note for K¨ahler manifolds, the field is C)

$$
H _ { \mathrm { d R } } ^ { 1 } \left( \Omega \setminus \bigcup _ { j = 1 } ^ { m } K _ { j } \right) \cong H _ { \mathrm { d R } } ^ { 1 } \left( \Omega \setminus \left\{ p _ { 1 } , \dots , p _ { m } \right\} \right) .\tag{B.1}
$$

Note that the first Betti number remains $b _ { 1 } = m$ . From an algebraic topology perspective, replacing point punctures with contractible bodies does not change the cohomology. Recall $H _ { \mathrm { d R } } ^ { 1 }$ is indeed a quotient space too, so it is both a cohomology and a quotient space.

Since the modules are isomorphic, there exists a (linear) map $\Psi$ such that

$$
\mathrm { I . } \Psi ( [ \omega ] + [ \eta ] ) = \Psi ( [ \omega ] ) + \Psi ( [ \eta ] ) ;
$$

II. $\Psi ( c [ \omega ] ) = c \Psi ( [ \omega ] )$ for any $c \in \mathbb { R }$ (recall the field is R).

The Mayer-Vietoris sequence Mayer (1929) allows us to identify additional cohomology classes introduced by the punctures. We obtain the following long exact sequence in cohomology, where $K = \cup _ { j = 1 } ^ { m } \{ x \in \Omega \ |$ $d ( x , K _ { j } ) < \epsilon \}$ is the union of neighborhoods surrounding the punctures

![](images/7df06e8ae85eece0932e9c0a63d4141ec203422385fc736423f73eebc7d5751d.jpg)

Here, $\delta ^ { * }$ is the connecting homomorphism (not to be confused with our codiferential notation δ). Vanilla PINNs fail to capture the holes, and as we demonstrated in Example 1, fail to accommodate jump discontinuities. These topological nuances are reflected in the arrows since de Rham PINNs allow nontrivial mappings between cohomology classes, whereas vanilla PINNs mean that these mappings are trivial. In particular, vanilla PINNs fail because they are typically stuck in $B _ { \mathrm { d R } } ^ { 1 }$ , specifically when $\omega _ { \theta } = d \phi _ { \theta }$ , and $\delta ^ { * }$ is necessarily trivial for an exact form.

## C Manners of eficiency

In section 3.2, we discussed an orthogonally-decomposed architecture with the ansatz $\alpha _ { \theta } = d \phi _ { \theta } + \delta \beta _ { \theta } +$ $\textstyle \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } h _ { j }$ . In this section, we consider more eficient architecture and the trade-ofs they pose. For example, let us consider the alternative ansatz

$$
\alpha _ { \theta } ( x , t ) = \gamma _ { \theta } ( x , t ) + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( t ) h _ { j } ( x ) .\tag{C.1}
$$

Here, $\gamma _ { \theta }$ is a network that absorbs both exact and coexact terms, and the harmonic terms are still a priori. To understand the downfalls of C.1, we must understand further why our proposed de Rham PINN of 3.18 remedies the vanilla PINN. As we have discussed, vanilla PINNs are conducive to exact forms du<sub>θ</sub>. In the case of zero topological pathologies, the equation $\alpha _ { \theta } = d u _ { \theta }$ , where ${ \mathrm { I m } } ( u _ { \theta } ) \subseteq \mathbb { R }$ is the scalar PINN solution, and $\alpha \theta$ is the corresponding form. Therefore, the inclusion of the coexact and harmonic terms remedy this artifact. But what about C.1? If the form is parameterized directly, and we manually add a harmonic term, this should remedy the exact quality. Let us examine the following pitfalls, in order of least to greatest importance.

I. The Hodge-Morrey-Friedrichs decomposition is a way to enforce physical properties into the architecture itself, so it is more than just representational. By definition, exact forms are irrotational, $d ( d \phi ) = 0$ and co-exact forms are solenoidal or divergence-free, $\delta ( \delta \beta ) = 0$ . For example, rather than enforcing physical laws into a loss function with a further penalty term, we can systematically construct the network to obey governing laws. We refer to Liu et al. (2024) Longo et al. (2023) for some works that enforce governing laws into the architectures. Enforcing laws in architectures exists also in rather na¨ıve cases, i.e. forcing positive semi-definite with neural network output $A ^ { T } A$ , so our Hodge-decomposed architecture is a rather mature case of this. If the governing PDE demands specific properties, i.e. incompressible flows, the orthogonal ansatz can satisfy with laws with error zero, i.e. machine precision, and moreover constrain the hypothesis class for these laws. Take the example of conservation of mass

$$
\delta \alpha = 0 ,\tag{C.2}
$$

and we take

$$
\alpha _ { \theta } = \delta \beta _ { \theta } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } h _ { j }\tag{C.3}
$$

$$
\implies \delta \alpha _ { \theta } = \delta ( \delta \beta _ { \theta } ) + \delta \left( \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } h _ { j } \right) \equiv 0\tag{C.4}
$$

by construction. Therefore, by collapsing the exact and coexact terms into a supersuming network $\gamma _ { \theta } .$ all of these properties are lost. Were these laws to be enforced manually, the optimizer must now deal with this. This is not to say that a collective $\gamma _ { \theta }$ will fail to satisfy to train the objective to bring the error down. Therefore, this leads us to point III, where the pitfall of C.1 lies in an impossible representation.

II. If $\gamma _ { \theta } \in \Omega ^ { 1 } ( \Omega ^ { * } )$ is in an unrestricted hypothesis class, then certainly it is capable of absorbing the harmonic $c ( t ) h ( x )$ terms. What is to distinguish C.1 from even $\alpha _ { \theta } = \gamma _ { \theta } ?$ Gradient descent will inevitably distribute the harmonic-to-be-learned pathology among the closed-form harmonic ansatz and the network $\gamma _ { \theta }$ . Moreover, the harmonic term generally corresponds to aggressive or sharp physical characteristics such as $1 / r$ or $1 / r ^ { 2 }$ scaling near the punctures. We witnessed this scaling in the definition of 4.25. Networks sufer from spectral bias and it will be dificult to incorporate these sharp pathologies into a desirable global solution. This is mostly what we witnessed in the experiment of H.2. $\mathrm { B y }$ baking the closed-form harmonic form into the ansatz directly, we have a guarantee that the pathology will be learned by the Hodge-Morrey-Friedrichs decomposition. In a decomposed setting, the other networks will not ”fight” to learn these pathologies, as they have their own learning tasks to achieve.

III. Most importantly, with the bundled architecture, it may be impossible to infer the learned PDE solution. It is possible it may technically succeed but it may pose other issues instead, and if it were to succeed, it is computationally no diferent than using the Hodge decomposition directly, or at least almost. Using the Hodge-Morrey-Friedrichs decomposition may be necessary to extract the scalar field. The reason that the scalar PINN fails in the $\gamma _ { \theta }$ case is because

$$
d \gamma _ { \theta } \neq 0\tag{C.5}
$$

is (almost surely) guaranteed, which is a fatal problem.

The Hodge architecture guarantees $d ( d \phi _ { \theta } ) = 0$ to be true architecturally. This allows us to extract a path-independent potential from the exact component $\phi _ { \theta }$ . Note that $d ( \delta \beta _ { \theta } )$ can still be zero, but to acquire the scalar potential, we do not necessarily integrate the coexact term always (see F.1, F.5 for evidence). In particular, $d ( \delta \beta _ { \theta } ) \neq 0$ leads to path-dependence.

In some scenarios, we integrate the full form $\alpha \theta$ and not just $\phi _ { \theta }$ , such as in F.2, F.4. In these scenarios, parameterizing $\gamma _ { \theta }$ can work, but instead we must penalize $d \gamma _ { \theta } = 0$ in a loss function (which is necessary for path-independence), which is nontrivial and will not be solved exactly. $d \gamma _ { \theta }$ is crucial because if not, path-independence is violated because

$$
\int _ { \eta _ { 1 } } \gamma - \int _ { \eta _ { 2 } } \gamma = \oint _ { C } \gamma = \iint _ { \Sigma } d \gamma .\tag{C.6}
$$

Note that integrating the full form $\alpha \theta$ will also violate path-independence since

$$
d ( d \phi + \delta \beta ) = d ( \delta \beta ) \neq 0 .\tag{C.7}
$$

This is not a problem in our experiment of F.2 because we integrate w.r.t. a canonical loop and in F.4 because we penalized curl in the loss and because we only integrated horizontally. Moreover, $d \gamma _ { \theta }$ is nontrivial for the optimizer to solve and may afect well-behaved training on the other loss terms, i.e. due to spectral bias and blow-up. The Hodge architecture can bypass this. In particular, the computational adversities of the Hodge architecture are also reflected in the architecture with parameterizing $\gamma _ { \theta }$ because now additional, diferentiated terms must be in the loss too. The whole purpose of the bundled ansatz was bypassing the computational burdens of the exact and coexact diferentiation, so these burdens will appear (at least partially) in the bundled ansatz with an extended loss function.

Note the bundled ansatz fails as in H.2 but for reasons other than the above. The bundled ansatz will fail in this scenario due to spectral bias and the omission of the decay rather than path-dependence. So this is noteworthy as well.

## D Coexact formulas in generalized cases

We previously saw a simple closed-form solution for the coexact terms in 2D and 3D via 3.28 and 3.30. In this section, we elaborate on the closed formula for the coexact term for more generalized cases, since this term is combinatorial and the most nontrivial of the three terms of 3.18.

On a general smooth, d-dimensional Riemannian manifold $\Omega ^ { * } = M$ with metric $g$ and covariant derivative ∇ (i.e. choose the Levi-Civita connection), the formula is, in local coordinates

$$
( \delta \beta ) _ { i _ { 1 } \ldots i _ { k } } = - \sum _ { j = 1 } ^ { d } \sum _ { m = 1 } ^ { d } g ^ { j m } \nabla _ { m } \beta _ { j i _ { 1 } \ldots i _ { k } } .\tag{D.1}
$$

The indices are accurate because $\beta$ is $\mathrm { a } \left( k + 1 \right) - \mathrm { f o r m }$ . In flat Euclidean space (where $g ^ { j m } = \delta ^ { j m }$ and $\nabla _ { m }  \partial _ { m } )$ this simplifies to

$$
( \delta \beta ) _ { i _ { 1 } \dots i _ { k } } = - \sum _ { j = 1 } ^ { d } \partial _ { j } \beta _ { j i _ { 1 } \dots i _ { k } } .\tag{D.2}
$$

Notice this formula is consistent with equations 3.28, 3.30. In particular, this equation is consistent with 3.30 because

$$
( \delta \beta ) _ { x } = - ( \partial _ { x } \beta _ { x x } + \partial _ { y } \beta _ { y x } + \partial _ { z } \beta _ { z x } )\tag{D.3}
$$

$$
= \partial _ { y } \beta _ { x y } - \partial _ { z } \beta _ { z x }\tag{D.4}
$$

since $\beta _ { x x } = 0$ and $\beta _ { y x } = - \beta _ { x y } .$ . For alternative indices,

$$
( \delta \beta ) _ { y } = - ( \partial _ { x } \beta _ { x y } + 0 + \partial _ { z } \beta _ { z y } ) = \partial _ { z } \beta _ { y z } - \partial _ { x } \beta _ { x y } = \partial _ { z } \beta ^ { x } - \partial _ { x } \beta ^ { z }\tag{D.5}
$$

$$
( \delta \beta ) _ { z } = - ( \partial _ { x } \beta _ { x z } + \partial _ { y } \beta _ { y z } + 0 ) = \partial _ { x } \beta _ { z x } - \partial _ { y } \beta _ { y z } = \partial _ { x } \beta ^ { y } - \partial _ { y } \beta ^ { x } .\tag{D.6}
$$

This formula is convenient because it only requires first order automatic diferentiation in an empirical setting. Note that the above is also true for the $\Omega ^ { * }$ submanifold we have been using. For implementation purposes on non-Euclidean manifolds, it is often easier to formulate the coexact term on a manifold using the Hodge star ⋆ and the exterior derivative d. Note that this is more applicable for D.1, since D.2 is not hard to compute. The general operator for a p-form in an d-dimensional manifold on a non-flat space is

$$
\delta = ( - 1 ) ^ { d p + d + 1 } \star d \star .\tag{D.7}
$$

This can be written using the (antisymmetric) Levi-Civita tensor $\epsilon ,$ defined as

$$
\epsilon _ { i _ { 1 } i _ { 2 } \ldots i _ { d } } = { \left\{ \begin{array} { l l } { + 1 } & { { \mathrm { i f ~ } } \sigma { \mathrm { ~ i s ~ a n ~ e v e n ~ p e r m u t a t i o n } } } \\ { - 1 } & { { \mathrm { i f ~ } } \sigma { \mathrm { ~ i s ~ a n ~ o d d ~ p e r m u t a t i o n } } } \\ { 0 } & { { \mathrm { i f ~ a n y ~ i n d e x ~ i s ~ r e p e a t e d . } } } \end{array} \right. }\tag{D.8}
$$

Using this format, we can write (see Huguet et al. (2026) for relevant literature)

$$
( \delta \beta ) _ { m _ { 1 } . . . m _ { p - 1 } } = \frac { ( - 1 ) ^ { d p + d + 1 } } { p ! ( d - p ) ! } \sum _ { k = 1 } ^ { d } \sum _ { j _ { 1 } . . . j _ { d - p } = 1 } ^ { d } \sum _ { i _ { 1 } . . . i _ { p } = 1 } ^ { d } \epsilon _ { k j _ { 1 } . . . j _ { d - p } m _ { 1 } . . . m _ { p - 1 } \epsilon _ { i _ { 1 } . . . i _ { p } j _ { 1 } . . . j _ { d - p } } \partial _ { k } \beta _ { i _ { 1 } . . . i _ { p } } } .\tag{D.9}
$$

In the Euclidean case, this reduces to $\begin{array} { r } { ( \delta \beta ) _ { i _ { 1 } \dots i _ { p - 1 } } = - \sum _ { k = 1 } ^ { d } \partial _ { k } \beta _ { k i _ { 1 } \dots i _ { p - 1 } } } \end{array}$ . Alternatively, ϵ can be written via a determinant of Kronecker deltas

$$
\epsilon _ { i _ { 1 } i _ { 2 } \ldots i _ { d } } = \operatorname* { d e t } { \left( \begin{array} { l l l l } { \delta _ { 1 , i _ { 1 } } } & { \delta _ { 1 , i _ { 2 } } } & { \ldots } & { \delta _ { 1 , i _ { d } } } \\ { \delta _ { 2 , i _ { 1 } } } & { \delta _ { 2 , i _ { 2 } } } & { \ldots } & { \delta _ { 2 , i _ { d } } } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { \delta _ { d , i _ { 1 } } } & { \delta _ { d , i _ { 2 } } } & { \ldots } & { \delta _ { d , i _ { d } } } \end{array} \right) } = \prod _ { 1 \leq a < b \leq d } { \frac { i _ { b } - i _ { a } } { b - a } } .\tag{D.10}
$$

Notice the term is annihilated if $i _ { b } = i _ { a }$ since the numerator is zero. Strictly speaking, the Levi-Civita symbol ϵ defined above is a pseudo-tensor because its components do not change under coordinate transformations. In order for it to be a true tensor on a general Riemannian manifold, it must be scaled by the square root of the absolute determinant of the metric g

$$
E _ { i _ { 1 } \dots i _ { d } } = \sqrt { | g | } \epsilon _ { i _ { 1 } \dots i _ { d } } .\tag{D.11}
$$

In flat Euclidean space where $g$ is the identity matrix and ${ \sqrt { | g | } } = 1$ , the tensor $E$ and the symbol ϵ are identical.

Note that the generalized formula for $d \phi$ is always

$$
d \phi = \sum _ { i = 1 } ^ { d } \frac { \partial \phi } { \partial x ^ { i } } d x ^ { i }\tag{D.12}
$$

with respect to any local coordinate chart $( U , x ^ { 1 } , \ldots , x ^ { d } )$ . This holds for any manifold $\Omega ^ { * } = M$

## E The loss function in 3D

Here, we develop expanded loss functions as in 3. First, let us repeat 4.8. The physics loss is, with respect to the correct permutation σ

$$
\begin{array} { l l } { \mathcal { L } _ { \mathrm { p h y s i c s } } = \displaystyle { \| \begin{array} { l } { \displaystyle { \frac { 3 } { i - 1 } } | \partial _ { t } [ \frac { \partial \phi _ { \theta } ( x , t ) } { \partial x _ { i } } + ( \frac { \partial \beta _ { \theta } ^ { \sigma ^ { 2 } ( i ) } ( x , t ) } { \partial x _ { \sigma ( i ) } } - \frac { \partial \beta _ { \theta } ^ { \sigma ( i ) } ( x , t ) } { \partial x _ { \sigma ^ { 2 } ( i ) } } ) + \displaystyle { \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( t ) h _ { j , i } ( x ) } ] } } \\ { \displaystyle { + \Psi _ { \alpha , i } [ x , t , \{ \nabla ^ { k } [ \frac { \partial \phi _ { \theta } ( x , t ) } { \partial x _ { i } } + ( \frac { \partial \beta _ { \theta } ^ { \sigma ^ { 2 } ( i ) } ( x , t ) } { \partial x _ { \sigma ( i ) } } - \frac { \partial \beta _ { \theta } ^ { \sigma ( i ) } ( x , t ) } { \partial x _ { \sigma ^ { 2 } ( i ) } } ) + \displaystyle { \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( t ) h _ { j , i } ( x ) } ] \} } _ { k } \displaystyle { \| } ^ { 2 } \displaystyle { \| \begin{array} { l } { \displaystyle { \begin{array} { l } { \begin{array} { l } { \mathrm { I . 1 } } \\ { \mathrm { \Omega . \nabla . \nabla ~ } ( [ 0 . t ] } { x . t } ) } \end{array} } } \\ { \displaystyle { L ^ { 1 } ( \Omega ^ { - \kappa } [ 0 . T ] \cdot ( \partial { \Omega } ^ { \kappa } ( x , t ) ) } } \end{array} } } \end{array} ) }  \end{array} \end{array}\tag{E.2}
$$

Alternatively, this can be written using the Levi-Civita symbol $\epsilon _ { i p q }$ (as in D) to denote the cross-product tensor. We can also write

$$
\mathcal { L } _ { \mathrm { p h y s i c s } } = \| \sum _ { i = 1 } ^ { 3 } | \partial _ { t } [ \frac { \partial \phi _ { \theta } ( x , t ) } { \partial x _ { i } } + \sum _ { p = 1 } ^ { 3 } \sum _ { q = 1 } ^ { 3 } \epsilon _ { i p q } \frac { \partial \beta _ { \theta } ^ { q } ( x , t ) } { \partial x _ { p } } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( t ) h _ { j , i } ( x ) ] |
$$

$$
+ \Psi _ { \alpha , i } \Bigg [ x , t , \Bigg \{ \nabla ^ { k } \Bigg [ \frac { \partial \phi _ { \theta } ( x , t ) } { \partial x _ { i } } + \sum _ { p = 1 } ^ { 3 } \sum _ { q = 1 } ^ { 3 } \epsilon _ { i p q } \frac { \partial \beta _ { \theta } ^ { q } ( x , t ) } { \partial x _ { p } } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( t ) h _ { j , i } ( x ) \Bigg ] \Bigg \} _ { k } \Bigg ] ^ { 2 } \Bigg |\tag{E.3}
$$

The correct permutation is

$$
i = 1 \colon \sigma ( 1 ) = 2 , \ \sigma ^ { 2 } ( 1 ) = 3 \implies \partial _ { 2 } \beta ^ { 3 } - \partial _ { 3 } \beta ^ { 2 } ,\tag{E.4}
$$

$$
i = 2 \colon \sigma ( 2 ) = 3 , \ \sigma ^ { 2 } ( 2 ) = 1 \implies \partial _ { 3 } \beta ^ { 1 } - \partial _ { 1 } \beta ^ { 3 } ,\tag{E.5}
$$

$$
i = 3 \colon \sigma ( 3 ) = 1 , \ \sigma ^ { 2 } ( 3 ) = 2 \implies \partial _ { 1 } \beta ^ { 2 } - \partial _ { 2 } \beta ^ { 1 } .\tag{E.6}
$$

Writing the initial condition loss as in 4.14

$$
\mathcal { L } _ { \mathrm { I C } } = | | \sum _ { i = 1 } ^ { 3 } | [ \frac { \partial \phi _ { \theta } ( x , 0 ) } { \partial x _ { i } } + \sum _ { p = 1 } ^ { 3 } \sum _ { q = 1 } ^ { 3 } \epsilon _ { i p q } \frac { \partial \beta _ { \theta } ^ { q } ( x , 0 ) } { \partial x _ { p } } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( 0 ) h _ { j , i } ( x ) ] - \frac { \partial g ( x ) } { \partial x _ { i } } | ^ { 2 } | _ { L ^ { 1 } ( \Omega ^ { * } ; d \mu ^ { * } ) } .\tag{E.7}
$$

Note that we can also write this in its equivalent vector form, explicitly expanding the 3D curl, as

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { I C } } = \Bigg | \Bigg | \Bigg | \Bigg | \Bigg | \alpha _ { \theta } ( x , 0 ) ^ { \sharp } - \nabla _ { x } g ( x ) \Bigg | \Bigg | _ { 2 } \Bigg | _ { L ^ { 1 } ( \Omega ^ { * } ; d \mu ^ { * } ) } } \\ & { \qquad = \Bigg | \Bigg | \Bigg | | ( \frac { \partial \phi _ { \theta } ( x , 0 ) } { \partial x _ { 1 } } + \frac { \partial \beta _ { \theta } ^ { 3 } ( x , 0 ) } { \partial x _ { 2 } } - \frac { \partial \beta _ { \theta } ^ { 2 } ( x , 0 ) } { \partial x _ { 3 } } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( 0 ) h _ { j , 1 } ( x ) ) - ( \frac { \partial g ( x ) } { \partial x _ { 1 } } ) | \Bigg | _ { 2 } ^ { 2 } \Bigg | \Bigg | _ { L ^ { 1 } ( \Omega ^ { * } ; d \mu ^ { * } ) } \ : . } \\ & { \qquad | | ( \frac { \partial \phi _ { \theta } ( x , 0 ) } { \partial x _ { 2 } } + \frac { \partial \beta _ { \theta } ^ { 3 } ( x , 0 ) } { \partial x _ { 3 } } - \frac { \partial \beta _ { \theta } ^ { 3 } ( x , 0 ) } { \partial x _ { 1 } } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( 0 ) h _ { j , 2 } ( x ) ) - ( \frac { \partial g ( x ) } { \partial x _ { 2 } } ) | \Bigg | _ { 2 } ^ { 2 } \Bigg | \Bigg | _ { L ^ { 1 } ( \Omega ^ { * } ; d \mu ^ { * } ) } \ : . } \end{array}\tag{E.8}
$$

(E.9)

## F Experimental details

## F.1 Unit radial vortex

In this example, we consider the PDE learning task on the scalar solution $\Delta u \ : = \ : 0 .$ , i.e. a harmonic function. We use an annular domain $r \in \left\lceil 1 , 2 \right\rceil$ or equivalently $\{ ( x , y ) | 1 < x ^ { 2 } + y ^ { 2 } < 4 \}$ . We enforce $\begin{array} { r } { \nabla \cdot { \boldsymbol { \alpha } } = \frac { \partial \alpha _ { 1 } } { \partial x } + \frac { \partial \alpha _ { 2 } } { \partial y } = 0 , \nabla \times { \boldsymbol { \alpha } } = \frac { \partial \alpha _ { 2 } } { \partial x } - \frac { \partial \alpha _ { 1 } } { \partial y } = 0 } \end{array}$ (see F.2), so it is enforced $d \alpha = 0$ (closed) and $\delta \alpha = 0$ (co-closed), which imply

$$
\Delta _ { H } \alpha = ( d \delta + \delta d ) \alpha = 0 .\tag{F.1}
$$

We enforce boundary conditions $\alpha \cdot n = 0$ at $r = 1$ and $\alpha = ( - \frac { y } { 4 } , \frac { x } { 4 } )$ at $r = 2$ . We take the relation between $\alpha , u$ to be $\alpha = \nabla u \iff \alpha ^ { \sharp } = \nabla u$ , or equivalently $\alpha = d u$ , locally but not globally, so these hold true along any small, simply connected patch of the annulus. In particular, because the target field $\left( - { \frac { y } { 4 } } , { \frac { x } { 4 } } \right)$ on the annulus creates a non-zero circulation around the central hole, there is no single, continuous scalar function u that works for the entire domain. So α is a locally exact form.

We consider two cases with this experiment: (1) with a simple boundary condition; (2) with a more sophisticated boundary condition. By default, we will do this experiment with the simple case (such as Figure 1). To reiterate, in both cases, we consider the PDE physics loss on the form

$$
\mathcal { L } _ { \mathrm { p h y s i c s } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \nabla \cdot \boldsymbol { \alpha } ^ { i } \right) ^ { 2 } + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \nabla \times \boldsymbol { \alpha } ^ { i } \right) ^ { 2 } .\tag{F.2}
$$

We have denoted $\alpha ^ { i } = \alpha ( x ^ { i } , y ^ { i } )$ . For the simple case, we consider boundary loss

$$
\mathcal { L } _ { \mathrm { b o u n d a r y } } = \mathcal { L } _ { \mathrm { i n } } + \mathcal { L } _ { \mathrm { o u t } }\tag{F.3}
$$

$$
\mathcal { L } _ { \mathrm { i n } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \alpha _ { 1 } ^ { i } x ^ { i } + \alpha _ { 2 } ^ { i } y ^ { i } \right) ^ { 2 }\tag{F.4}
$$

$$
\mathcal { L } _ { \mathrm { o u t } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \left( \alpha _ { 1 } ^ { i } - \left( - \frac { y ^ { i } } { 4 } \right) \right) ^ { 2 } + \left( \alpha _ { 2 } ^ { i } - \frac { x ^ { i } } { 4 } \right) ^ { 2 } \right) .\tag{F.5}
$$

We refer to this as the unit radial vortex experiment. In the complicated case, we consider boundary loss

$$
\mathcal { L } _ { \mathrm { i n } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \left( \alpha _ { 1 } ^ { i } - \left( 0 . 5 x ^ { i } - y ^ { i } \right) \right) ^ { 2 } + \left( \alpha _ { 2 } ^ { i } - \left( - 0 . 5 y ^ { i } + x ^ { i } \right) \right) ^ { 2 } \right)\tag{F.6}
$$

$$
\mathcal { L } _ { \mathrm { o u t } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \left( \alpha _ { 1 } ^ { i } - \left( 0 . 5 x ^ { i } - \frac { y ^ { i } } { 4 } \right) \right) ^ { 2 } + \left( \alpha _ { 2 } ^ { i } - \left( - 0 . 5 y ^ { i } + \frac { x ^ { i } } { 4 } \right) \right) ^ { 2 } \right) .\tag{F.7}
$$

We refer to this as the complicated unit radial vortex experiment (see Figure 14). This experiment has scalar-valued and form PDEs, omitting boundary for simplicity

$$
\left\{ \begin{array} { l l } { \Delta { u } = 0 } \\ { \nabla \times \alpha = 0 } \\ { \nabla \cdot \alpha = 0 . } \end{array} \right.\tag{F.8}
$$

In this experiment, we extract the scalar solution from the PINN solution using

$$
u ( x , y ) = \phi _ { \theta } ( x , y ) + \frac { 1 } { 2 \pi } \cdot c _ { \theta } \cdot \arg ( x + i y ) ,\tag{F.9}
$$

where $x + i y = r e ^ { i \theta } , \arg ( x + i y ) = \theta$ (it is the same thing as arctangent but the quadrant is taken into account). In particular, $\beta _ { \theta }$ as in 3.18 is used in training, but it is not used to extract the PINN solution.

## F.2 Arctangent without zero

In this example, we construct an example over a 2D domain such that the coexact term is nonconstant and $\alpha =$ adu for some function a (see section 3.3). Moreover, the vanilla PINN should fail. Let $\Omega ^ { * } =$ $( [ - 1 , 1 ] \times [ - 1 , 1 ] ) \setminus \{ ( x , y ) | x ^ { 2 } + y ^ { 2 } \leq r ^ { 2 } \} = [ - 1 , 1 ] ^ { 2 } \setminus \{ ( x , y ) | x ^ { 2 } + y ^ { 2 } \leq r ^ { 2 } \}$ , and r is arbitrary (it is to prevent divergence at the pole). Let us choose $r ^ { 2 } = 0 . 5$ . Let us consider the true PDE solution

$$
u ( x , y , t ) = e ^ { - t } \theta = e ^ { - t } \arctan \left( \frac { y } { x } \right)\tag{F.10}
$$

corresponding to PDE

$$
\partial _ { t } u = \Delta u - u .\tag{F.11}
$$

A vanilla PINN will fail because it is forced to map continuous coordinates $( x , y )$ to a scalar field that jumps by $2 \pi e ^ { - t }$ across the branch cut. The 1-form satisfies $\alpha = a ( x , y ) d u$ using $a ( x , y ) = e ^ { x }$ . Taking the exterior derivative of $u ,$

$$
d u = e ^ { - t } d \theta = e ^ { - t } \left( \frac { - y } { x ^ { 2 } + y ^ { 2 } } d x + \frac { x } { x ^ { 2 } + y ^ { 2 } } d y \right) = e ^ { - t } h .\tag{F.12}
$$

Therefore, the target 1-form is

$$
\alpha = e ^ { x } d u = e ^ { x - t } h .\tag{F.13}
$$

We can note the coexact term is non-constant, and α is not exact. We check the exterior derivative of α is

$$
d \alpha = d ( e ^ { x } ) \wedge ( e ^ { - t } h ) = e ^ { x - t } d x \wedge \left( { \frac { - y d x + x d y } { x ^ { 2 } + y ^ { 2 } } } \right) = e ^ { x - t } { \frac { x } { x ^ { 2 } + y ^ { 2 } } } d x \wedge d y .\tag{F.14}
$$

Because $d \alpha \neq 0 ,$ , the 1-form α is not closed. Therefore, it cannot be exact globally or locally. Also, we note a harmonic term is active, since we can integrate α over the unit circle $\gamma \ ( r = 1$ , where $x = \cos \theta )$

$$
\oint _ { \gamma } \alpha = e ^ { - t } \int _ { 0 } ^ { 2 \pi } e ^ { \cos \theta } d \theta \neq 0 .\tag{F.15}
$$

Substituting du $= e ^ { - x } \alpha$ into the scalar PDE, we get the PDE on the 1-form

$$
\partial _ { t } \alpha + e ^ { x } \Delta _ { H } ( e ^ { - x } \alpha ) + \alpha = 0 .\tag{F.16}
$$

We note $\Delta _ { H } : = d \delta + \delta d , \Delta _ { H } = - \Delta$

We found empirical success with a specialized ansatz for this problem. A standard decomposition would be $\alpha _ { \theta } = \nabla \phi _ { \theta } + \nabla \times \beta _ { \theta } + c _ { \theta } ( t ) h$ . We modify the harmonic state by an $e ^ { x }$ envelope. We parameterize the 1-form as

$$
\alpha _ { \theta } = \nabla \phi _ { \theta } + \nabla \times \beta _ { \theta } + e ^ { x } c _ { \theta } ( t ) h ,\tag{F.17}
$$

where h is the standard harmonic $\begin{array} { r } { h = \frac { 1 } { 2 \pi ( x ^ { 2 } + y ^ { 2 } ) } ( - y d x + x d y ) } \end{array}$ . The PDE requires us to compute the Laplacian of $A _ { \theta } = e ^ { - x } \alpha _ { \theta }$ . Distributing the inverse envelope yields $A _ { \theta } = e ^ { - x } \nabla \phi _ { \theta } + e ^ { - x } \nabla \times \beta _ { \theta } + c _ { \theta } ( t ) h$ .

For the physics loss, sample points on a compact subset of $\mathbb { R } ^ { 2 } \times [ 0 , T ] , \{ ( x ^ { i } , y ^ { i } , t ^ { i } ) \} _ { i = 1 } ^ { N }$ , i.e. on $\Omega ^ { * } \times [ 0 , T ]$ Let $A _ { \theta } ^ { i } = e ^ { - x ^ { i } } \alpha _ { \theta } ^ { i }$ . The physics loss can be written

$$
\mathcal { L } _ { \mathrm { p h y s i c s } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left. \left( \partial _ { t } \alpha _ { \theta } ^ { i } - e ^ { x ^ { i } } \Delta ( A _ { \theta } ^ { i } ) + \alpha _ { \theta } ^ { i } \right) ^ { \sharp } \right. _ { 2 } ^ { 2 } ,\tag{F.18}
$$

again noting a vector formulation via the sharp musical isomorphism $\sharp : T ^ { * } M \to T M$ . The exact solution is as follows:

$$
\alpha _ { 1 } ^ { * } ( x , y , t ) = e ^ { x - t } \left( \frac { - y } { x ^ { 2 } + y ^ { 2 } } \right) , \qquad \alpha _ { 2 } ^ { * } ( x , y , t ) = e ^ { x - t } \left( \frac { x } { x ^ { 2 } + y ^ { 2 } } \right) .\tag{F.19}
$$

We write initial and boundary conditions

$$
\mathcal { L } _ { \mathrm { I C } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ \left( \alpha _ { 1 , \theta } ( x ^ { i } , y ^ { i } , 0 ) - \alpha _ { 1 } ^ { * } ( x ^ { i } , y ^ { i } , 0 ) \right) ^ { 2 } + \left( \alpha _ { 2 , \theta } ( x ^ { i } , y ^ { i } , 0 ) - \alpha _ { 2 } ^ { * } ( x ^ { i } , y ^ { i } , 0 ) \right) ^ { 2 } \right]\tag{F.20}
$$

$$
\mathcal { L } _ { \mathrm { B C } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ \left( \alpha _ { 1 , \theta } ( x _ { \mathrm { B C } } ^ { i } , y _ { \mathrm { B C } } ^ { i } , t ^ { i } ) - \alpha _ { 1 } ^ { * } ( x _ { \mathrm { B C } } ^ { i } , y _ { \mathrm { B C } } ^ { i } , t ^ { i } ) \right) ^ { 2 } + \left( \alpha _ { 2 , \theta } ( x _ { \mathrm { B C } } ^ { i } , y _ { \mathrm { B C } } ^ { i } , t ^ { i } ) - \alpha _ { 2 } ^ { * } ( x _ { \mathrm { B C } } ^ { i } , y _ { \mathrm { B C } } ^ { i } , t ^ { i } ) \right) ^ { 2 } \right] .\tag{F.21}
$$

The total PDE system is

$$
\left\{ \begin{array} { l l } { \partial _ { t } u = \Delta u - u } \\ { \partial _ { t } \alpha - e ^ { x } \Delta ( e ^ { - x } \alpha ) + \alpha = 0 } \\ { \alpha ( x , y , 0 ) = \alpha ^ { * } ( x , y , 0 ) } & { \mathrm { ~ f o r ~ } ( x , y ) \in \Omega ^ { * } } \\ { \alpha ( x , y , t ) = \alpha ^ { * } ( x , y , t ) } & { \mathrm { ~ f o r ~ } ( x , y ) \in \partial [ - 1 , 1 ] ^ { 2 } , \ t \in ( 0 , T ] . } \end{array} \right.\tag{F.22}
$$

To extract the true solution from the PINN solution, we compute

$$
u ( x , y , t ) = \left( \frac { 1 } { 2 \pi } \int _ { 0 } ^ { 2 \pi } e ^ { - \cos \xi } \Big [ - \alpha _ { \theta , 1 } ( \cos \xi , \sin \xi , t ) \sin \xi + \alpha _ { \theta , 2 } ( \cos \xi , \sin \xi , t ) \cos \xi \Big ] d \xi \right) \arctan \left( \frac { y } { x } \right) .\tag{F.23}
$$

where $\xi \in [ 0 , 2 \pi ]$

![](images/6fcb9f0a1ada089efdc2e73dcecdf93c1cb79261369c52cd98bac357837eb631.jpg)

![](images/b539c9a253646415ffb6c36e8d899ef4c6268898c170818335f8c429392e6be2.jpg)  
Figure 9: We plot the coeficient c(t) after 3,000 epochs at 5 discrete t values for the arctangent without zero experiment with (left) a typical Hodge-decomposed architecture; and (right) a specialized Hodge-decomposed architecture where a $e ^ { x }$ deformation is baked into architecture, so the ansatz is $A = e ^ { - x } \alpha _ { \theta } = e ^ { - x } \nabla \phi + e ^ { - x } \nabla \times \beta + c ( t ) h$ and α<sub>θ</sub> is a Hodge-decomposed learned form. As we can see, the true values of c are learned much more precisely with the architecture specialized for this experiment.

## F.3 Arctangent with exhaustion

In this section, we perform the same experiment as F.2 but now we reduce the domain around the pole, so we sample closer to the pole, and further equip an exhaustion function. We use the exhaustion function with associated weight

$$
f ( x , y ) = \frac { \epsilon } { x ^ { 2 } + y ^ { 2 } } , \qquad W ( x , y ) = \exp \left( - \frac { \epsilon } { x ^ { 2 } + y ^ { 2 } } \right) ,\tag{F.24}
$$

which decays rapidly and is overall a good choice. Now, the losses are

$$
\mathcal { L } _ { \mathrm { p h y s i c s } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } W ( x ^ { i } , y ^ { i } ) \left. \left( \partial _ { t } \alpha _ { \theta } ^ { i } - e ^ { x ^ { i } } \Delta ( A _ { \theta } ^ { i } ) + \alpha _ { \theta } ^ { i } \right) ^ { \sharp } \right. _ { 2 } ^ { 2 } ,\tag{F.25}
$$

$$
\mathcal { L } _ { \mathrm { I C } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } W ( x ^ { i } , y ^ { i } ) \left[ \left( \alpha _ { 1 , \theta } ( x ^ { i } , y ^ { i } , 0 ) - \alpha _ { 1 } ^ { * } ( x ^ { i } , y ^ { i } , 0 ) \right) ^ { 2 } + \left( \alpha _ { 2 , \theta } ( x ^ { i } , y ^ { i } , 0 ) - \alpha _ { 2 } ^ { * } ( x ^ { i } , y ^ { i } , 0 ) \right) ^ { 2 } \right] .\tag{F.26}
$$

We crucially remark regarding this experiment, without exhaustion function the PDE will be learned suficiently well around the pole, although the networks will not train well. This is moreso due to neural network properties such as smoothness and behaving suficiently well on of-training data. We refer to Figures 19, 20. The exhaustion function is what allows the networks to train and obtain a better and more robust solution.

## F.4 The Aharonov–Bohm efect with one pole

In this section, we describe our Aharonov–Bohm (AB) efect experiment. In this experiment, we enforce a $[ - 4 , 4 ] \times [ - 4 , 4 ] \subseteq \mathbb { R } ^ { 2 }$ domain and we enforce a coupled system of scalar-valued PDEs corresponding to the Madelung formulation of the Schr¨odinger equation

$$
{ \frac { \partial } { \partial x } } \left[ \rho \left( { \frac { \partial S } { \partial x } } - A _ { x } \right) \right] + { \frac { \partial } { \partial y } } \left[ \rho \left( { \frac { \partial S } { \partial y } } - A _ { y } \right) \right] = 0 ,\tag{F.27}
$$

$$
{ \frac { 1 } { 2 } } \left[ \left( { \frac { \partial S } { \partial x } } - A _ { x } \right) ^ { 2 } + \left( { \frac { \partial S } { \partial y } } - A _ { y } \right) ^ { 2 } \right] - { \frac { 1 } { 2 { \sqrt { \rho } } } } \left( { \frac { \partial ^ { 2 } { \sqrt { \rho } } } { \partial x ^ { 2 } } } + { \frac { \partial ^ { 2 } { \sqrt { \rho } } } { \partial y ^ { 2 } } } \right) = E .\tag{F.28}
$$

Here, $\begin{array} { r } { E = { \frac { p ^ { 2 } } { 2 m } } = { \frac { k ^ { 2 } } { 2 } } = { \frac { 3 ^ { 2 } } { 2 } } = 4 . 5 } \end{array}$ since we set $k = 3$ . We have denoted

$$
A _ { x } = - \frac { \Phi _ { B } } { 2 \pi } \frac { y } { x ^ { 2 } + y ^ { 2 } } , \quad A _ { y } = \frac { \Phi _ { B } } { 2 \pi } \frac { x } { x ^ { 2 } + y ^ { 2 } } ,\tag{F.29}
$$

setting $\Phi _ { B } = \pi$ . The scalar PDE solution is $\psi ( x , y ) = \sqrt { \rho ( x , y ) } e ^ { i S ( x , y ) }$ , where $( \rho , S ) = ( \rho _ { \theta } , S _ { \theta } ) \in \mathbb { R } ^ { 2 }$ is the neural network output. We plot only the real part of the true solution in $3 , ~ 4$ , whose true solution obeys $\psi _ { \mathrm { t r u e } } ( x , y ) = e ^ { i ( k x + \alpha \theta ) }$ , where k is as before and $\begin{array} { r } { \alpha = \frac { \Phi _ { B } } { 2 \pi } } \end{array}$ . We use Dirichlet boundary data $\rho ( - 4 , y ) =$ $1 , S ( - 4 , y ) = - 4 k , \forall y \in [ - 4 , 4 ]$

The vanilla PINN experiences catastrophic failure in the above because: (1) in the true setting, the magnetic vector potential A is a closed 1-form, so $\nabla \times \mathbf { A } = 0$ , but it is not exact because its circulation around the origin is non-zero $\oint \mathbf { A } \cdot d \mathbf { l } = \Phi _ { B } ; \mathbf { \tau } ( 2 )$ ∇S corresponds to an exact 1-form since $\boldsymbol { \nabla } S = ( d \boldsymbol { S } ) ^ { \sharp }$ . We cannot completely cancel or balance a non-exact 1-form A using only an exact 1-form ∇S. Here, we do not mean they directly cancel in a PDE residual $\nabla S - A$ , but rather, in the theoretical setting, ∇S must absorb the topology of A since

$$
\oint { \nabla } S \cdot d \mathbf { l }\tag{F.30}
$$

$$
= \oint \mathbf { P } \cdot d \mathbf { l } + \oint \mathbf { A } \cdot d \mathbf { l }\tag{F.31}
$$

$$
= 0 + \Phi _ { B }\tag{F.32}
$$

$$
= \Phi _ { B } ,\tag{F.33}
$$

where $\mathbf { P } \approx k \mathbf { e } _ { 1 }$ is true in the code, which is irrotational. In the PINN setting, $\oint { \nabla } S \cdot d \mathbf { l } = 0$

Turning to the Hodge-de Rham approach, we perform the PDE on the 1-form α

$$
\frac { \partial \alpha _ { 2 } } { \partial x } - \frac { \partial \alpha _ { 1 } } { \partial y } = 0\tag{F.34}
$$

$$
\frac { \partial } { \partial x } \left[ \rho ( \alpha _ { 1 } - A _ { x } ) \right] + \frac { \partial } { \partial y } \left[ \rho ( \alpha _ { 2 } - A _ { y } ) \right] = 0\tag{F.35}
$$

$$
{ \frac { 1 } { 2 } } \left[ ( \alpha _ { 1 } - A _ { x } ) ^ { 2 } + ( \alpha _ { 2 } - A _ { y } ) ^ { 2 } \right] - { \frac { 1 } { 2 { \sqrt { \rho } } } } \left( { \frac { \partial ^ { 2 } { \sqrt { \rho } } } { \partial x ^ { 2 } } } + { \frac { \partial ^ { 2 } { \sqrt { \rho } } } { \partial y ^ { 2 } } } \right) = E .\tag{F.36}
$$

To extract the scalar PDE solution from the PINN, we perform numerical integration for domain length in one direction $L = 4$ and set

$$
S _ { \theta } ( x , y ) = - k L + \int _ { - L } ^ { x } \alpha _ { \theta , 1 } ( x ^ { \prime } , y ) d x ^ { \prime }\tag{F.37}
$$

$$
S _ { \mathrm { t o t a l } , \theta } ( x , y ) = S _ { \theta } ( x , y ) + c _ { 1 } \frac { \theta } { 2 \pi }\tag{F.38}
$$

$$
\widetilde { \rho } _ { \theta } = \ln ( 1 + e ^ { \rho _ { \theta } } ) ,\tag{F.39}
$$

and subsequently take the prediction into the ansatz

$$
\psi = \sqrt { \widetilde { \rho _ { \theta } } } \cos ( S _ { \mathrm { t o t a l } , \theta } ) .\tag{F.40}
$$

In particular, the integration in F.37 is computed as

$$
\int _ { - L } ^ { x } \alpha _ { \theta , 1 } ( x ^ { \prime } , y ) d x ^ { \prime } = \int _ { - L } ^ { x } \frac { \partial \phi _ { \theta } } { \partial x ^ { \prime } } d x ^ { \prime } + c _ { 1 } \int _ { - L } ^ { x } h _ { 1 } ( x ^ { \prime } , y ) d x ^ { \prime }\tag{F.41}
$$

$$
= \phi _ { \theta } ( x , y ) - \phi _ { \theta } ( - L , y ) + c _ { 1 } \left( \frac { \theta ( x , y ) } { 2 \pi } - \frac { \theta ( - L , y ) } { 2 \pi } \right) .\tag{F.42}
$$

## F.5 The Aharonov-Bohm efect with two poles

In this experiment, we consider an identical setup to F.4 but now with two poles. We use the same PDE system F.27 but now we place two poles at $y = - 1 . 5 , y = 1 . 5$ with radii $r = 0 . 1 5$ . We plot our results in Figures 5, 6. We also take $k = 4$ and $\begin{array} { r } { E = \frac { 4 ^ { 2 } } { 2 } = 8 . 0 } \end{array}$ . Since A no longer corresponds to one pole, we take

$$
A _ { x } = - q _ { 1 } \frac { y - y _ { 1 } } { ( x - x _ { 1 } ) ^ { 2 } + ( y - y _ { 1 } ) ^ { 2 } } - q _ { 2 } \frac { y - y _ { 2 } } { ( x - x _ { 2 } ) ^ { 2 } + ( y - y _ { 2 } ) ^ { 2 } }\tag{F.43}
$$

$$
A _ { y } = q _ { 1 } \frac { x - x _ { 1 } } { ( x - x _ { 1 } ) ^ { 2 } + ( y - y _ { 1 } ) ^ { 2 } } + q _ { 2 } \frac { x - x _ { 2 } } { ( x - x _ { 2 } ) ^ { 2 } + ( y - y _ { 2 } ) ^ { 2 } } .\tag{F.44}
$$

The true solution obeys ψ $\mathrm { \Psi _ { t r u e } } ( x , y ) = e ^ { i ( k x + q _ { 1 } \theta _ { 1 } + q _ { 2 } \theta _ { 2 } ) }$ now. S is now calculated

$$
S _ { \theta } ( x , y ) = \phi _ { \theta } ( x , y ) + \left( - k L - \frac { 1 } { N _ { y } } \sum _ { y } \phi _ { \theta } ( - L , y ) \right)\tag{F.45}
$$

$$
S _ { \mathrm { t o t a l } , \theta } ( x , y ) = S _ { \theta } ( x , y ) + c _ { 1 } { \frac { \theta _ { 1 } } { 2 \pi } } + c _ { 2 } { \frac { \theta _ { 2 } } { 2 \pi } } ,\tag{F.46}
$$

where $\phi _ { \theta }$ is as in the exact part we have been using in the form ansatz. ${ \oint } \mathbf { A } \cdot d \mathbf { l } = \Phi _ { B }$ now reflects that the domain is $[ - 4 , 4 ] \times [ - 4 , 4 ] \times \{ P _ { 1 } , P _ { 2 } \}$ . The first Betti number is now $b _ { 1 } = 2 . ~ \widetilde { \rho }$ is still calculated as $\widetilde { \rho } _ { \theta } = \ln ( 1 + e ^ { \rho _ { \theta } } )$ .

## G Connections to complex and K¨ahler geometries

In this section, we consider when the domain corresponds to a complex or K¨ahler manifold.

Assume the domain to be considered corresponds to a K¨ahler manifold. It may appear that the (K¨ahler) form can be written as the Dolbeault Hessian of a K¨ahler potential

$$
\omega = i \partial { \overline { { \partial } } } \rho .\tag{G.1}
$$

Here, $\partial \colon \Omega ^ { p , q } ( M ) \to \Omega ^ { p + 1 , q } ( M ) , { \overline { { \partial } } } \colon \Omega ^ { p , q } ( M ) \to \Omega ^ { p , q + 1 } ( M )$ are the Dolbeault operators. The above of G.1 is admissible if the domain is $\mathbb { C } ^ { d } .$ however this domain is unbounded and not practical for a PINN in reality. A natural choice is to truncate $\mathbb { C } ^ { d }$ to a compact subset. A global K¨ahler potential $\rho$ cannot exist on a compact K¨ahler manifold (without boundary) because this forces $\rho$ to be constant if $\rho \in { \mathrm { S P S H } } ( M )$ Demailly (2012). We will always take our PINN domain to be a compact set.

It can be noted a compact<sup>2</sup> subset of $\mathbb { C } ^ { d }$ is indeed K¨ahler because it has a boundary. A compact K¨ahler manifold, for example a Calabi-Yau manifold by invoking the Calabi conjecture Yau (1977), is intended for cases without boundaries, and diferential equations such as Monge-Amp\`ere equations require boundary conditions. Indeed, a compact subset of $\mathbb { C } ^ { d }$ is K¨ahler with global potential since it acquires the flat metric. Specifically, the flat K¨ahler form is given by

$$
\omega = \frac { i } { 2 } \sum _ { j = 1 } ^ { d } d z _ { j } \wedge d \overline { { { z } } } _ { j } ,\tag{G.2}
$$

which admits the global K¨ahler potential

$$
\rho ( z ) = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { d } | z _ { j } | ^ { 2 } .\tag{G.3}
$$

Because of the existence of such a boundary, the maximum of the SPSH function is attained on the boundary. Therefore, the maximum principle, which prevents non-constant global SPSH potentials on compact manifolds, is not violated.

## G.1 A Pythagorean loss function

This section here is motivated by an orthogonal direct sum decomposition of a vector bundle. In what we develop here, we will rely on cross-terms dropping to zero, which happens because the manifold is equipped with a Hermitian metric. In particular, what we develop in this section works for any complex manifold with a Hermitian metric, not just for manifolds with a K¨ahler metric. To be more clear: the $\mathrm { P y }$ thagorean efect we establish here is for any manifold with Hermitian structure, since

$$
\Omega ^ { 1 } ( M ; \mathbb { C } ) = \Omega ^ { 1 , 0 } ( M ) \oplus ^ { \perp } \Omega ^ { 0 , 1 } ( M )\tag{G.4}
$$

does not require K¨ahlerness.

In G.24, we will split the harmonic terms due to the fact that the real Betti number $b _ { 1 } = h ^ { 1 , 0 } + h ^ { 0 , 1 }$ . More importantly, notice that we will split into two ”components,” which is because the 1-form splits

$$
\Omega ^ { 1 } = \Omega ^ { 1 , 0 } \oplus \Omega ^ { 0 , 1 } .\tag{G.5}
$$

The basis 1-form consists of 2d distinct directions: the d holomorphic directions $( d z _ { 1 } , \ldots , d z _ { d } )$ and the d anti-holomorphic directions $( d \overline { { z } } _ { 1 } , \ldots , d \overline { { z } } _ { d } )$ . We do not add the components into the same squared term because $d z _ { k }$ and $d \overline { { z } } _ { k }$ are orthogonal, so adding the two directly violates the Pythagorean theorem, i.e. it is equivalent to computing the vector norm $\boldsymbol { v } = ( x , y )$ as $( x + y ) ^ { 2 }$ instead of $x ^ { 2 } + y ^ { 2 }$ . To elaborate further, we rewrite G.15 for generalized form as, for any K¨ahler manifold,

$$
v = \sum _ { k = 1 } ^ { d } X _ { k } d z _ { k } + \sum _ { k = 1 } ^ { d } Y _ { k } d \overline { { z } } _ { k }\tag{G.6}
$$

$$
\implies \| v \| ^ { 2 } = \left. \sum _ { k = 1 } ^ { d } X _ { k } d z _ { k } + \sum _ { k = 1 } ^ { d } Y _ { k } d \bar { z } _ { k } , \sum _ { m = 1 } ^ { d } X _ { m } d z _ { m } + \sum _ { m = 1 } ^ { d } Y _ { m } d \bar { z } _ { m } \right.\tag{G.7}
$$

$$
= \left. \sum _ { k = 1 } ^ { d } X _ { k } d z _ { k } , \sum _ { m = 1 } ^ { d } X _ { m } d z _ { m } \right. + \left. \sum _ { k = 1 } ^ { d } X _ { k } d z _ { k } , \sum _ { m = 1 } ^ { d } Y _ { m } d \bar { z } _ { m } \right.\tag{G.8}
$$

$$
+ \left. \sum _ { k = 1 } ^ { d } Y _ { k } d \bar { z } _ { k } , \sum _ { m = 1 } ^ { d } X _ { m } d z _ { m } \right. + \left. \sum _ { k = 1 } ^ { d } Y _ { k } d \bar { z } _ { k } , \sum _ { m = 1 } ^ { d } Y _ { m } d \bar { z } _ { m } \right.\tag{G.9}
$$

$$
= \left. \sum _ { k = 1 } ^ { d } X _ { k } d z _ { k } , \sum _ { m = 1 } ^ { d } X _ { m } d z _ { m } \right. + 0 + 0 + \left. \sum _ { k = 1 } ^ { d } Y _ { k } d \bar { z } _ { k } , \sum _ { m = 1 } ^ { d } Y _ { m } d \bar { z } _ { m } \right.\tag{G.10}
$$

$$
= \sum _ { k = 1 } ^ { d } \sum _ { m = 1 } ^ { d } X _ { k } \overline { { X } } _ { m } \langle d z _ { k } , d z _ { m } \rangle + \sum _ { k = 1 } ^ { d } \sum _ { m = 1 } ^ { d } Y _ { k } \overline { { Y } } _ { m } \langle d \overline { { z } } _ { k } , d \overline { { z } } _ { m } \rangle\tag{G.11}
$$

$$
\mathrm  { f a t } \underbrace { m e t r i c } _ { k = 1 } \sp { d } X _ { k } \overline { { X } } _ { k } \delta _ { k k } + \sum _ { k = 1 } \sp { d } Y _ { k } \overline { { Y } } _ { k } \delta _ { k k }\tag{G.12}
$$

$$
= \sum _ { k = 1 } ^ { d } \left( | X _ { k } | ^ { 2 } + | Y _ { k } | ^ { 2 } \right) .\tag{G.13}
$$

Note that we use convention $z _ { k }$ and not $z ^ { k }$ , and so this distinguishes the above from Einstein notation. We can note the spatial terms with Φ depend on the full form $\alpha \theta$ rather than the holomorphic and anti-holomorphic components. Specifically, the diferentiated form into the PDE empirical risk is

$$
\mathcal { R } = \partial _ { t } ( \alpha _ { \theta } ) + \Phi ( \alpha _ { \theta } ) .\tag{G.14}
$$

We can note the first term in G.14

$$
\partial _ { t } ( \alpha _ { \theta } ) = \partial _ { t } \left( \frac { \partial \phi _ { \theta } } { \partial z } \right) d z + \partial _ { t } \left( \frac { \partial \psi _ { \theta } } { \partial \overline { { z } } } \right) d \overline { { z } } + . . .\tag{G.15}
$$

Due to interaction, i.e. advection $h ^ { i \bar { j } } \alpha _ { i } \nabla _ { \bar { j } } \alpha _ { k } + h ^ { \bar { i } j } \alpha _ { \bar { i } } \nabla _ { j } \alpha _ { k }$ , Φ depends on the entire form. We can also observe, applying the sharp musical isomorphism $\sharp$ to α, arranging into a $( T ^ { ( 1 , 0 ) } M \oplus T ^ { ( 0 , 1 ) } M )$ column vector

$$
\begin{array}{c} \begin{array} { l } { \displaystyle \alpha _ { \theta } ^ { \sharp } = ( \partial \phi _ { \theta } ) ^ { \sharp } + ( \bar { \partial } \psi _ { \theta } ) ^ { \sharp } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } h _ { j } ^ { \sharp } \qquad \in \mathcal { X } ( \mathcal { M } ) } \\ { \displaystyle } \\ { = ( \begin{array} { l } { \displaystyle ( \alpha _ { \theta } ^ { \sharp } ) ^ { 1 1 } } \\ { \vdots } \\ { \displaystyle ( \alpha _ { \theta } ^ { \sharp } ) ^ { d } } \\ { \displaystyle ( \alpha _ { \theta } ^ { \sharp } ) ^ { 1 1 } } \\ { \vdots } \\ { \displaystyle ( \alpha _ { \theta } ^ { \sharp } ) ^ { d } } \end{array} ) = ( h ^ { \bar { \mathbb { I } } k } ( \partial _ { k } \bar { \psi } _ { \theta } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( h _ { j } ) \bar { k } ) ) } \\ { \displaystyle ( \begin{array} { l } { \displaystyle \alpha _ { \theta } ^ { \sharp } } \\ { \vdots } \\ { \displaystyle ( \alpha _ { \theta } ^ { \sharp } ) ^ { d } } \end{array} ) } \\ { \displaystyle ( ( \alpha _ { \theta } ^ { \sharp } ) ^ { \sharp } ) } \end{array} ( h ^ { \bar { \mathbb { I } } k } ( \partial _ { k } \bar { \psi } _ { \theta } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( h _ { j } ) _ { k } ) )  \\ { \displaystyle \qquad \vdots } \\ { \displaystyle h ^ { \bar { \mathbb { I } } k } ( \partial _ { k } \bar { \psi } _ { \theta } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( h _ { j } ) _ { k } ) ) } \end{array}\tag{G.16}
$$

(G.17)

We crucially remark that the term that previously corresponded to the codiferential is now a mere Wirtinger derivative as in G.24. This is more computationally feasible (compare to E.1 as in Appendix E). Saying the codiferential is ”intractable” is a poor choice of words, because it is generally computable. Rather, computing the codiferential is a combinatorial problem: in 100D, $\beta _ { \theta }$ has 4,950 components. Therefore, this method is far superior for high dimensions.

## G.2 An exact architecture

In this section, we omit the coexact term as in the Hodge-Morrey-Friedrichs decomposition, and specifically discuss the remaining architecture in the context of the complex manifold. This case is notable because the physics loss of G.24 takes a diferent form due to the simplification of G.6. In the case of a complex manifold equipped with a Hermitian structure, we can parameterize our PINN for a complex-split exact form to be

$$
\alpha _ { \theta } = \partial \phi _ { \theta } + \overline { { { \partial } } } \psi _ { \theta } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } h _ { j } \qquad \in \Omega ^ { 1 } ( M ) .\tag{G.18}
$$

Note that $\partial \phi _ { \theta } \in \Omega ^ { 1 , 0 }$ and $\overline { { \partial } } \psi _ { \theta } \in \Omega ^ { 0 , 1 }$ . Here, the sum iterates over the Betti number $b _ { 1 }$ , and the $h _ { j }$ are the absolute harmonic 1-forms. We should be clear this does not parameterize any arbitrary 1-form. In fact, the hypothesis class of the above is

$$
\begin{array} { r } { \mathcal { H } = \partial \big ( C ^ { \infty } ( M ) \big ) + \overline { { \partial } } \big ( C ^ { \infty } ( M ) \big ) + \mathcal { H } _ { A } ^ { 1 } ( M ) \subsetneq \Omega ^ { 1 } ( M ) , } \end{array}\tag{G.19}
$$

so the above architecture $\alpha \theta$ is not suitable for all PINNs. This architecture fails to include the coexact (solenoidal) components of the Hodge-Morrey-Friedrichs decomposition.

Example 2. Suppose a PINN is required to learn the non-exact form $\beta = z _ { 2 } d z _ { 1 }$ . We parameterize and attempt to learn

$$
z _ { 2 } d z _ { 1 } = \partial \phi _ { \theta } + \sum _ { j } c _ { j } h _ { j } ^ { ( 1 , 0 ) } .\tag{G.20}
$$

Consider applying the Dolbeault operator $\partial .$ On the true form, we have

$$
\partial ( z _ { 2 } d z _ { 1 } ) = d z _ { 2 } \wedge d z _ { 1 } \neq 0 .\tag{G.21}
$$

However, applying $\partial$ to our parameterized ansatz yields

$$
\partial \left( \partial \phi _ { \theta } + \sum _ { j } c _ { j } h _ { j } ^ { ( 1 , 0 ) } \right) = \partial ^ { 2 } \phi _ { \theta } + \sum _ { j } c _ { j } \partial h _ { j } ^ { ( 1 , 0 ) } = 0 .\tag{G.22}
$$

This evaluates to zero because $\partial ^ { 2 } = 0$ by definition, and the absolute harmonic forms $h _ { j }$ are closed, so $d h _ { j } = 0 \implies \partial h _ { j } ^ { ( 1 , 0 ) } = 0$ . Thus, the network cannot represent $\beta ,$ regardless of the harmonic coeficients.

Let us consider attempting to learn

$$
z _ { 2 } d z _ { 1 } = { \frac { \partial \phi _ { \theta } } { \partial z _ { 1 } } } d z _ { 1 } + { \frac { \partial \phi _ { \theta } } { \partial z _ { 2 } } } d z _ { 2 } .\tag{G.23}
$$

Therefore, it must be learned

I. $\begin{array} { r } { \frac { \partial \phi _ { \theta } } { \partial z _ { 1 } } = z _ { 2 } ; } \end{array}$

II. $\begin{array} { r } { \frac { \partial \phi _ { \theta } } { \partial z _ { 2 } } = 0 . } \end{array}$

By Clairaut’s theorem, these two conditions contradict each other. Diferentiating the first equation with respect to $z _ { 2 }$ yields 1, while diferentiating the second equation with respect to $z _ { 1 }$ yields 0.

To expand the physics loss with this form similarly to 4.8, and the other loss terms follow accordingly, we write, under the flat metric

$$
\mathcal { L } _ { \mathrm { p h y s i c s } } = \| \sum _ { k = 1 } ^ { d } ( | \partial _ { t } [ \frac { \partial \phi _ { \theta } } { \partial z _ { k } } ( z , \overline { { z } } , t ) + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( t ) ( h _ { j } ^ { ( 1 , 0 ) } ) _ { k } ( z , \overline { { z } } ) ]\tag{G.24}
$$

$$
+ \Psi _ { \alpha , k } \left[ z , \overline { { { z } } } , t , \left\{ \frac { \partial \phi _ { \theta } } { \partial z _ { \ell } } ( z , \overline { { { z } } } , t ) + \frac { \partial \psi _ { \theta } } { \partial \overline { { { z } } } _ { \ell } } ( z , \overline { { { z } } } , t ) + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( t ) ( h _ { j } ) _ { \ell } ( z , \overline { { { z } } } ) \right\} \right] ^ { 1 }\tag{G.25}
$$

$$
+ [ \partial _ { t } [ \frac { \partial \psi _ { \theta } } { \partial \overline { { { z } } } _ { k } } ( z , \overline { { { z } } } , t ) + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( h _ { j } ^ { ( 0 , 1 ) } ) _ { \overline { { { k } } } } ( z , \overline { { { z } } } ) ] \nonumber\tag{G.26}
$$

$$
+ \Psi _ { \alpha , \overline { { { x } } } } \Bigg [ z , \overline { { { z } } } , t , \left\{ \frac { \partial \phi _ { \theta } } { \partial z _ { \ell } } ( z , \overline { { { z } } } , t ) + \frac { \partial \psi _ { \theta } } { \partial \overline { { { z } } } _ { \ell } } ( z , \overline { { { z } } } , t ) + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( t ) ( h _ { j } ) _ { \ell } ( z , \overline { { { z } } } ) \right\} _ { \ell } \Bigg ] \Bigg | ^ { 2 } \Bigg ) \Bigg | _ { L ^ { 1 } ( \Omega ^ { * } \times [ 0 , T ] ) }\tag{G.27}
$$

$$
= \int _ { [ 0 , T ] } \int _ { \Omega ^ { * } } \sum _ { k = 1 } ^ { d } ( | \partial _ { t } \Biggl [ \frac { \partial \phi _ { \theta } } { \partial z _ { k } } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( h _ { j } ) _ { k } \Biggr ] + \Psi _ { \alpha , k } [ \alpha _ { \theta } ] | ^ { 2 }\tag{G.28}
$$

$$
+ | \partial _ { t } [ \frac { \partial \psi _ { \theta } } { \partial \overline { { z } } _ { k } } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( h _ { j } ) _ { \overline { { k } } } ] + \Psi _ { \alpha , \overline { { k } } } [ \alpha _ { \theta } ] | ^ { 2 } ) \frac { \omega ^ { d } } { d ! } d t .\tag{G.29}
$$

Note that we have relaxed a constant of 2, which is unimportant in a machine learning context. Note that we also remark the following identity

$$
H _ { \mathrm { d R } } ^ { 1 } ( M ; \mathbb { C } ) \cong H _ { \overline { { \partial } } } ^ { 1 , 0 } ( M ) \oplus H _ { \overline { { \partial } } } ^ { 0 , 1 } ( M ) .\tag{G.30}
$$

can hold in cases K¨ahler manifolds, but this is afected by compactness, boundary, etc. In general, G.30 does not hold for compact K¨ahler manifolds with boundary, which is what we have here. The loss of G.24 is

closely related to this but does not use this exactly, and the above holds for any complex manifold with (flat) Hermitian structure.

Instead, we define the space of absolute harmonic 1-forms

$$
\mathcal { H } _ { A } ^ { 1 } ( M ) = \left\{ h \in \Omega ^ { 1 } ( M ; \mathbb { C } ) : d h = 0 , ~ \delta h = 0 , ~ { \star h } | _ { \partial M } = 0 \right\} .\tag{G.31}
$$

Here, ν denotes the outward unit normal. We can note

$$
{ \mathcal { H } } _ { A } ^ { 1 } ( M ) \cong H _ { \mathrm { d R } } ^ { 1 } ( M ; \mathbb { C } ) ,\tag{G.32}
$$

$$
\dim _ { \mathbb { C } } { \mathcal { H } } _ { A } ^ { 1 } ( M ) = b _ { 1 } .\tag{G.33}
$$

Therefore, we can choose a harmonic basis $h _ { 1 } , \ldots , h _ { b _ { 1 } }$ even when the manifold has a boundary. In general, G.30 is not available in our setting. We refer to Al-Zamil and Montaldi (2012) for relevant literature. In particular, in G.24, we use

$$
h _ { j } = h _ { j } ^ { 1 , 0 } + h _ { j } ^ { 0 , 1 } .\tag{G.34}
$$

This result follows since

$$
\Omega ^ { 1 } ( M ; \mathbb { C } ) = \Omega ^ { 1 , 0 } ( M ) \oplus ^ { \perp } \Omega ^ { 0 , 1 } ( M ) ,\tag{G.35}
$$

which we explain in the beginning of Appendix G.1.

## G.3 Full de Rham K¨ahler architecture

In light of Example 2, the full de Rham architecture is

$$
\alpha _ { \theta } = \partial \phi _ { \theta } + \overline { { { \partial } } } \psi _ { \theta } + \partial ^ { * } \gamma _ { \theta } ^ { 2 , 0 } + ( \partial ^ { * } + \overline { { { \partial } } } ^ { * } ) \gamma _ { \theta } ^ { 1 , 1 } + \overline { { { \partial } } } ^ { * } \gamma _ { \theta } ^ { 0 , 2 } + \sum c _ { j } h _ { j } .\tag{G.36}
$$

We now include a coexact term. Here, we have denoted

$$
\partial ^ { * } = i [ \Lambda , \overline { { { \partial } } } ] = i ( \Lambda \overline { { { \partial } } } - \overline { { { \partial } } } \Lambda )\tag{G.37}
$$

$$
\overline { { { \partial } } } ^ { * } = - i [ \Lambda , \partial ] = - i ( \Lambda \partial - \partial \Lambda ) ,\tag{G.38}
$$

where $L$ is the Lefschetz operator and Λ is its adjoint. The complexified de Rham cohomology in this case corresponds to an isomorphism

$$
H _ { \mathrm { d R } } ^ { k } ( \Omega ^ { * } , \mathbb { C } ) \cong \bigoplus _ { p + q = k } H _ { \ni } ^ { p , q } ( \Omega ^ { * } ) ,\tag{G.39}
$$

and because the learned form α is a 1-form, $H _ { \mathrm { d R } } ^ { 1 } ( \Omega ^ { * } , \mathbb { C } ) \cong H _ { \overline { { \partial } } } ^ { 1 , 0 } \oplus H _ { \overline { { \partial } } } ^ { 0 , 1 }$ . As we noted in G.2, the above does not hold in compact K¨ahler cases with boundary, so we mostly leave the above as a remark.

Note we can write

$$
H _ { L ^ { 2 } } ^ { p , q } ( \Omega ^ { * } , \psi ) = \frac { \ker \left( \Gamma _ { L ^ { 2 } ( \psi ) } ( \Omega ^ { * } , \Lambda ^ { p , q } T ^ { * } \Omega ^ { * } ) \xrightarrow { \overline { { \partial } } } \Gamma _ { L ^ { 2 } ( \psi ) } ( \Omega ^ { * } , \Lambda ^ { p , q + 1 } T ^ { * } \Omega ^ { * } ) \right) } { \operatorname { I m } \left( \Gamma _ { L ^ { 2 } ( \psi ) } ( \Omega ^ { * } , \Lambda ^ { p , q - 1 } T ^ { * } \Omega ^ { * } ) \xrightarrow { \overline { { \partial } } } \Gamma _ { L ^ { 2 } ( \psi ) } ( \Omega ^ { * } , \Lambda ^ { p , q } T ^ { * } \Omega ^ { * } ) \right) } ,\tag{G.40}
$$

and there exists a homomorphism $\iota : H _ { L ^ { 2 } } ^ { p , q } ( \Omega ^ { * } , \psi )  H _ { \overline { { \partial } } } ^ { p , q } ( \Omega ^ { * } )$ , mapping the weighted $L ^ { 2 }$ cohomology classes to the smooth Dolbeault classes. Note that this architecture introduces a combinatorial problem with the coexact term, which we previously saw in 3.

In this setting, the physics loss is

$$
\mathcal { L } _ { \mathrm { p h y s i c s } } = \| \sum _ { k = 1 } ^ { d } ( | \partial _ { t } [ \frac { \partial \phi _ { \theta } } { \partial z _ { k } } ( z , \overline { { z } } , t ) + [ \partial ^ { * } \gamma _ { \theta } ^ { 2 , 0 } ( z , \overline { { z } } , t ) + \overline { { \partial } } ^ { * } \gamma _ { \theta } ^ { 1 , 1 } ( z , \overline { { z } } , t ) ] _ { k } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( t ) ( h _ { j } ^ { ( 1 , 0 ) } ) _ { k } ( z , \overline { { z } } ) ] | _ { \displaystyle = 0 } )\tag{G.41}
$$

$$
\left. + \Psi _ { \alpha , k } \left[ z , \overline { { z } } , t , \left\{ \frac { \partial \phi _ { \theta } } { \partial z _ { \ell } } + \frac { \partial \psi _ { \theta } } { \partial \overline { { z } } _ { \ell } } + \left[ \partial ^ { * } \gamma _ { \theta } ^ { 2 , 0 } + ( \partial ^ { * } + \overline { { \partial } } ^ { * } ) \gamma _ { \theta } ^ { 1 , 1 } + \overline { { \partial } } ^ { * } \gamma _ { \theta } ^ { 0 , 2 } \right] _ { \ell } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( t ) ( h _ { j } ) _ { \ell } \right\} _ { \ell } \right] \right| ^ { 2 }\tag{G.42}
$$

$$
+ \left| \partial _ { t } \left[ \frac { \partial \psi _ { \theta } } { \partial z _ { k } } ( z , \overline { { z } } , t ) + \big [ \partial ^ { * } \gamma _ { \theta } ^ { 1 , 1 } ( z , \overline { { z } } , t ) + \overline { { \partial } } ^ { * } \gamma _ { \theta } ^ { 0 , 2 } ( z , \overline { { z } } , t ) \big ] _ { \overline { { k } } } + \sum _ { j = 1 } ^ { b _ { 1 } } c _ { \theta , j } ( t ) ( h _ { j } ^ { ( 0 , 1 ) } ) _ { \overline { { k } } } ( z , \overline { { z } } ) \right] \right| _ { \overline { { k } } } ,\tag{G.43}
$$

(G.44)

## H K¨ahler manifold experiments

In this section, we provide two experiments on complex manifolds. We provide two experiments since in H.1 the vector-valued PINN succeeds, while in H.2 the vector-valued PINN fails

![](images/84bbc3bdc03e3ed57b76f3250e3b6c97b8c6a0118800109561e338685ca5945a.jpg)  
(1)

![](images/b2cfd5cf90dca2ed30ee44174e672c3e01c859d2830e3c66fa397a8eb7117960.jpg)  
(2)

![](images/5699f3306608ceedf747647bca2345d5a1092b3c18a5c545a5cc7af0a84aa828.jpg)  
(3)  
Figure 10: We illustrate (1) success with 1-form learning; (2) the true solution; and (3) pointwise absolute error on the K¨ahler manifold experiment with a Pythagorean loss function.

## H.1 K¨ahler manifold with vector-valued PINN success

In this section, we perform an experiment on a compact subset of C, which is a K¨ahler manifold, and we include the Pythagorean loss function of G.1.

We choose domain $\Omega = \{ z \in \mathbb { C } \mid \operatorname { R e } ( z ) \in [ - 4 , 4 ] , \operatorname { I m } ( z ) \in [ - 4 , 4 ] \}$ , and we create 2 poles or disks $D _ { k }$ of radii $r _ { \mathrm { m i n } } = 0 . 1 5$ centered at complex numbers $z _ { k } = x _ { k } + i y _ { k }$ . We take $\Omega ^ { * } = \Omega \setminus \bigcup _ { k = 1 } ^ { 2 } D _ { k }$ , which constitutes a K¨ahler manifold with first Betti number $b _ { 1 } = 2$ . We place the pole locations $( 0 . 0 , 1 . 5 )$ and $( 0 . 0 , - 1 . 5 )$ . We consider the PDE on the 1-form

$$
\nabla _ { \alpha ^ { \sharp } } \alpha - \nu \Delta _ { H } \alpha = f .\tag{H.1}
$$

We use notation $\begin{array} { r } { x = \frac { 1 } { 2 } ( z + \overline { { z } } ) , y = \frac { - i } { 2 } ( z - \overline { { z } } ) } \end{array}$ . Here, $f$ is the vector field $\boldsymbol { f } = \left( f _ { u } , f _ { v } \right)$ and $\alpha = u d x + v d y$ , so

$$
\left\{ \begin{array} { r l } & { f _ { u } = u _ { \mathrm { t r u e } } \partial _ { x } u _ { \mathrm { t r u e } } + v _ { \mathrm { t r u e } } \partial _ { y } u _ { \mathrm { t r u e } } - \nu ( \partial _ { x } ^ { 2 } u _ { \mathrm { t r u e } } + \partial _ { y } ^ { 2 } u _ { \mathrm { t r u e } } ) } \\ & { f _ { v } = u _ { \mathrm { t r u e } } \partial _ { x } v _ { \mathrm { t r u e } } + v _ { \mathrm { t r u e } } \partial _ { y } v _ { \mathrm { t r u e } } - \nu ( \partial _ { x } ^ { 2 } v _ { \mathrm { t r u e } } + \partial _ { y } ^ { 2 } v _ { \mathrm { t r u e } } ) . } \end{array} \right.\tag{H.2}
$$

The physics loss takes the form

$$
\mathcal { L } _ { \mathrm { p h y s i c s } } = \frac { 1 } { N } \sum \left( | \mathcal { R } ^ { ( 1 , 0 ) } | ^ { 2 } + | \mathcal { R } ^ { ( 0 , 1 ) } | ^ { 2 } \right) .\tag{H.3}
$$

We write

$$
\left\{ \begin{array} { r l } & { R _ { u } = u _ { \theta } \partial _ { x } u _ { \theta } + v _ { \theta } \partial _ { y } u _ { \theta } - \nu ( \partial _ { x } ^ { 2 } u _ { \theta } + \partial _ { y } ^ { 2 } u _ { \theta } ) - f _ { u } } \\ & { R _ { v } = u _ { \theta } \partial _ { x } v _ { \theta } + v _ { \theta } \partial _ { y } v _ { \theta } - \nu ( \partial _ { x } ^ { 2 } v _ { \theta } + \partial _ { y } ^ { 2 } v _ { \theta } ) - f _ { v } , } \end{array} \right.\tag{H.4}
$$

which is exactly the empirical risk of H.1. We write $\begin{array} { r } { \mathcal { R } ^ { ( 1 , 0 ) } = \frac { 1 } { 2 } ( R _ { u } - i R _ { v } ) , \mathcal { R } ^ { ( 0 , 1 ) } = \frac { 1 } { 2 } ( R _ { u } + i R _ { v } ) } \end{array}$ . To extract the scalar solution, we take

$$
\left\{ \begin{array} { l l } { u = \mathrm { R e } \big ( \alpha ^ { ( 1 , 0 ) } + \alpha ^ { ( 0 , 1 ) } \big ) } \\ { v = \mathrm { R e } \big ( i ( \alpha ^ { ( 1 , 0 ) } - \alpha ^ { ( 0 , 1 ) } ) \big ) } \\ { \mathrm { S p e e d } = \sqrt { u ^ { 2 } + v ^ { 2 } } . } \end{array} \right.\tag{H.5}
$$

Take $r _ { k } ^ { 2 } = ( x - x _ { k } ) ^ { 2 } + ( y - y _ { k } ) ^ { 2 }$ to be the squared distance to the k-th vortex. We set $\Gamma _ { 1 } = 1 5 . 0 , \Gamma _ { 2 } = - 1 5 . 0 $ $\delta = 0 . 5$ , and $U = 1$ . The true solution is given by

$$
\left\{ \begin{array} { r l } & { u _ { k } ( x , y ) = - \frac { \Gamma _ { k } } { 2 \pi } \frac { y - y _ { k } } { r _ { k } ^ { 2 } } \left( 1 - e ^ { - r _ { k } ^ { 2 } / \delta ^ { 2 } } \right) } \\ & { v _ { k } ( x , y ) = \frac { \Gamma _ { k } } { 2 \pi } \frac { x - x _ { k } } { r _ { k } ^ { 2 } } \left( 1 - e ^ { - r _ { k } ^ { 2 } / \delta ^ { 2 } } \right) } \\ & { u _ { \mathrm { t r u e } } ( x , y ) = U + u _ { 1 } ( x , y ) + u _ { 2 } ( x , y ) } \\ & { v _ { \mathrm { t r u e } } ( x , y ) = v _ { 1 } ( x , y ) + v _ { 2 } ( x , y ) } \\ & { \mathrm { S p e e d } ( x , y ) = \sqrt { u ( x , y ) ^ { 2 } + v ( x , y ) ^ { 2 } } . } \end{array} \right.\tag{H.6}
$$

Notice the Hodge-de Rham PINN uses $u _ { \mathrm { t r u e } } , v _ { \mathrm { t r u e } } .$ , which implicitly incorporates $\left( 1 - e ^ { - r _ { k } ^ { 2 } / \delta ^ { 2 } } \right)$ . We also remark this term is the key to why the vector-valued PINN succeeds in this experiment.

## H.2 K¨ahler manifold with vector-valued PINN failure

In this experiment, the setup is exactly the same as H.1. Now, with $U = 2$ , we now set

$$
\left\{ \begin{array} { l l } { u _ { k } ( x , y ) = - \frac { \Gamma _ { k } } { 2 \pi } \frac { y - y _ { k } } { r _ { k } ^ { 2 } } } \\ { v _ { k } ( x , y ) = \frac { \Gamma _ { k } } { 2 \pi } \frac { x - x _ { k } } { r _ { k } ^ { 2 } } . } \end{array} \right.\tag{H.7}
$$

The reason the vector-valued vanilla PINN now fails is because we omitted the decay term $\left( 1 - e ^ { - r _ { k } ^ { 2 } / \delta ^ { 2 } } \right)$ We call this term the the Oseen-Lamb decay Jafer (2018) term.

## I A further scalar-valued PDE solution with nonzero coexact term example

Let us highlight very clearly that a scalar-valued PDE solution can have nonzero coexact term $\delta \beta .$ We remark this example especially complements other discussion in C. Choose the annular domain considered in Figure 1,

$$
\Omega ^ { * } = \{ ( x , y ) \mid 1 < x ^ { 2 } + y ^ { 2 } < 4 \}\tag{I.1}
$$

(which can be taken compactly in a PINN), and consider the PDE

$$
\left\{ \begin{array} { l l } { \nabla \cdot ( a ( x ) \nabla u ) = f , } & { x \in \Omega ^ { * } , } \\ { u = g , } & { x \in \partial \Omega ^ { * } , } \end{array} \right.\tag{I.2}
$$

Let the scalar solution and coeficient function be given by

$$
u ( x , y ) = x , \qquad a ( x , y ) = 3 + y .\tag{I.3}
$$

Since $a ( x , y ) > 1$ throughout the annulus, an ellipticity condition is satisfied. Moreover, the PDE is satisfied exactly

$$
\nabla \cdot ( a \nabla u ) = \frac { \partial } { \partial x } ( 3 + y ) = 0 .\tag{I.4}
$$

Thus, $u = x$ is an exact solution to the scalar elliptic PDE with boundary condition $g = x$ . The associated flux 1-form is given by

$$
\alpha = ( 3 + y ) d x .\tag{I.5}
$$

Taking its exterior derivative yields

$$
d \alpha = d y \wedge d x = - d x \wedge d y \neq 0 .\tag{I.6}
$$

By the Hodge decomposition, we get a rewriting as in 3.19 $\alpha = d \phi + \delta \beta + h$ . Taking exterior derivatives, since $d ^ { 2 } \phi = 0$ and $d h = 0$ (as h is closed and harmonic), we obtain

$$
d \alpha = d \delta \beta .\tag{I.7}
$$

Consequently,

$$
d \delta \beta = - d x \wedge d y \neq 0 ,\tag{I.8}
$$

which establishes that

$$
\delta \beta \neq 0 ,\tag{I.9}
$$

even though $u = x$ is smooth, globally defined, and satisfies the PDE exactly.

Let us elaborate the form-function conversion outlined in section 3.3 with this specific example. Consider the (isolated) PDE

$$
\nabla \cdot ( a \nabla u ) = f .\tag{I.10}
$$

Recall we are interested in $\alpha = a d u$ as established in 3.3. We examine the codiferential. On the Euclidean annulus, the codiferential of a 1-form is minus the divergence of its associated vector field. Consequently, $\delta \alpha = - f$ . We obtain the system of equations

$$
\left\{ \begin{array} { l l } { \delta \alpha = - f , } \\ { d ( a ^ { - 1 } \alpha ) = 0 , } \\ { \displaystyle \oint _ { \gamma _ { j } } a ^ { - 1 } \alpha = 0 , \quad j = 1 , \ldots , b _ { 1 } , } \\ { d u = a ^ { - 1 } \alpha . } \end{array} \right.\tag{I.11}
$$

Our networks retain the architecture of 3.18, and as before in section 3.3, we get

$$
u _ { \theta } ( x ) = u _ { \theta } ( x _ { 0 } ) + \int _ { x _ { 0 } } ^ { x } a ^ { - 1 } \alpha _ { \theta } .\tag{I.12}
$$

## J Additional figures

![](images/cc6f6776796b9f427915ed83188499e9d2cf2e83a3913012dbe3d5e9fc510329.jpg)  
Figure 11: We plot the value of c (there is one puncture, so one value) corresponding to our unit radius vortex experiment. We plot (green) the learned value where the learning rate of the c parameter is increased to 5e−2 and the remaining networks use learning rate $_ { 1 \mathrm { e - 3 } , }$ whereas the learning rate for (red) is the same as the remaining networks with 1e−3. We can see the high learning rate case approaches a value of 2π, which is the anticipated true learned value in this experiment.

![](images/44e40f6145254f71b7f3b1c418bd8dc6b5e5a8aef2615f2f8d4a86536ee75659.jpg)  
Figure 12: We plot the training losses over 3,000 epochs on our unit radial vortex experiment with (1) a relatively low learning rate of 1e−3 for all $\phi _ { \theta } , \beta _ { \theta } , c _ { \theta } ;$ and (2) a learning rate of 1e−3 for $\phi _ { \theta } , \beta _ { \theta }$ and a high learning rate of $5 \mathrm { e } - 2$ for $c _ { \theta }$ only. We set loss coeficients $\lambda _ { \mathrm { p h y s i c s } } = 1 , \lambda _ { \mathrm { b o u n d a r y } } = 1$

![](images/a1a832a13b5d999d58ae148e595164a87e9f8b0b6fb46be3e8dee99355a59995.jpg)  
Figure 13: We plot training loss on 3,000 epochs on the arctangent without zero experiment with a vanilla Hodgedecomposed ansatz $\alpha \theta$ and a specialized ansatz $e ^ { - x } \alpha _ { \theta }$ tailored for this specific problem. The tailored ansatz clearly outperforms. This figure complements 9.

![](images/b1f0ad4062c575bb63012d56c26226c6c32b3389db842a490225706992d8dccc.jpg)  
(1)

![](images/b9c639cfc8ca8da607f9f752b0ea500b16349500614208dc77f43bf53cb295bb.jpg)  
(2)

![](images/6b4e181a9eada21458c2a02ca23b32917b9de7d6dafd29cd8eb9940ef6b89c5d.jpg)  
(3)

Figure 14: We illustrate (1) success with 1-form learning, (2) catastrophic failure of a vanilla PINN, and (3) the true solution on the unit radial vortex experiment with complicated boundary condition.  
![](images/68655a27bbc74de5eef2cdf75662e79a2a96d36b40b8c3f94f16d2f78433c2ee.jpg)  
(1)

![](images/8fab90096df5f4503da2bf1d45b347d8340b8b79316fad0612873bc3d1e8efa7.jpg)  
(2)

![](images/043b69f08660ac576c73f1181850c65645ef02675de70e8fb0f32cedfcfaaf0f.jpg)  
(3)  
Figure 15: We illustrate (1) catastrophic failure of a canonically-developed vector-valued PINN; (2) the true solution; (3) pointwise absolute error after about ∼ 12, 000 epochs.

![](images/c5606d8d3ecbd2b3e02c46e554a9a31254520c579038beda7df06588a9bb7431.jpg)  
(1)

![](images/48ace430a3eb954cb5d12286fcbdc9ceeeea7618ca52a85535e57269dac3c234.jpg)  
(2)

![](images/77df9d06d1069b4d619c1f5194ccfa137b756364fe7c8fc504bb7d35e81d28ff.jpg)  
(3)  
Figure 16: We plot failure of a vanilla PINN on the arctangent without zero experiment. We plot (1) the PINN solution; (2) the true solution; (3) the absolute pointwise error. We train the PINN for ∼ 8, 000 epochs, and the los has plateaued. As we can see, the discontinuity is not learned, compatible with the discussion in 3.1.

![](images/3768cf32ed5e3250ce5be21ae9f26ed45698cfbea347f7baff67a73e7d0a6aa0.jpg)  
(1)

![](images/f21b8a9c23e4e788be96b4d5015d21bb0ad9093db3f8858257f9e4e0f4d5f9a6.jpg)  
(2)

![](images/63feeb2d035125172261e327086764d59845c865205b54513e5b0012ea095e68.jpg)  
(3)  
Figure 17: We plot success of the Hodge-de Rham PINN on the arctangent without zero experiment. We plot (1) the PINN solution; (2) the true solution; (3) the absolute pointwise error. We train the PINN for 3, 000 epochs.

![](images/423af7733ad92db82fd1f0d17f4ec8e1b8072e097170b762be012c392f46c27f.jpg)  
Figure 18: We plot the vector field according to the 1-form $\alpha = \alpha _ { 1 } d x + \alpha _ { 2 } d y$ on the arctangent without zero experiment.

![](images/7a66846081b5c36da306b2477ee18526ca63a905efd5d6a3408944a4bdf6c5e9.jpg)  
Figure 19: We plot training loss on the arctangent without zero experiment when we set the radius around the pole to $r = 0$ and we exclude an exhaustion function. As we can see, the loss is highly stochastic and blows-up.

![](images/e4370b3bf87080df5228a3211e6134165c2e8ccb967246226b6e185901356df7.jpg)  
Figure 20: We plot training loss on the arctangent with exhaustion experiment when we set the radius around the pole to $r = 0$ and we include an exhaustion function. As we can see, the loss is stable.

![](images/a0148840c58d91c88d4d07977c202e9a0018ebf8abe65dd55e60c5528c80dc63.jpg)  
(1)

![](images/e6c5a55287ac384bde6df3347bc101a5740eb13162d1edb3c735ac8595fb5648.jpg)  
(2)

![](images/b1b3d6e96cb8be3b8e72a6209e0c563097f38f612871d9e873f2c5a04abe1b89.jpg)  
(3)  
Figure 21: We plot partial success, partial failure of the arctangent without zero (exhaustion function excluded) experiment after 4,000 epochs but we unrestrict $\{ ( x , y ) | x ^ { 2 } + r ^ { 2 } < 0 . 2 \}$ on the test dataset, but it is omitted in the training data. We plot (1) the PINN solution; (2) the true solution; (3) absolute pointwise error. The learning on the previously-omitted annular region is solely due to out-of-distribution learning

![](images/2cb8b8b1ea634e07dd3af436d4dc5340d4bb7351123574cf2c6568f93fbf747b.jpg)  
(1)

![](images/6d85422a55611672efd3588e93750de7ff18e6453ad3ceccd0bad029deaec6a7.jpg)  
(2)

![](images/e58241551fc75dbdf61327761a5b1ee4eeee12b58fe91790d297e050628d00fa.jpg)  
(3)  
Figure 22: We plot the same figure as 21 but with exhaustion after 4,000 epochs. We plot (1) the Hodge-de Rham PINN solution; (2) the true solution; (3) absolute pointwise error. As we can see, there is significantly greater success since maximum pointwise error drops from $\sim 0 . 7  \sim 0 . 1 5$

![](images/f6909913d35f9708f55e2abd1ce7198cdbcd839237dcea1b12950bdbec1bee15.jpg)  
(1)

![](images/f9c03f8dec481c7d3bbfd4e6a4f97602535475fea28c66d3dd82b918df4a4c2f.jpg)  
(2)

![](images/51e4f42c9bd731e00d36cdada8f6a302480ffb94bf0bae372b74b6ccdfe2979c.jpg)  
(3)  
Figure 23: We illustrate (1) catastrophic failure of a vanilla PINN; (2) the true solution; and (3) pointwise absolute error on the K¨ahler manifold experiment as in Appendix H.1 with a Pythagorean loss function after ∼ 7, 000 epochs.

Re(z)  
![](images/5eba9e64bde2a8c384318d5108ddeb4d6a60737673249b716974abaded9aa166.jpg)  
(1)

![](images/cce073d44b3a957977a5360ae47c30d32fb947513a3e9204423018e64e618673.jpg)  
(2)

![](images/2487fc9bbcc9375fa347ad4e1a8cd19a3b5b1c34738917eae130aba80ef0ece6.jpg)  
(3)  
Figure 24: We illustrate (1) success of a vector-valued vanilla PINN; (2) the true solution; and (3) pointwise absolute error on the K¨ahler manifold experiment as in Appendix H.1 with a Pythagorean loss function after ∼ 12, 000 epochs.

![](images/9c48cb0f2b773a40aefc07b7e8261bda3cdcbbebd83830fac48423543c605319.jpg)

![](images/17c7a93cdcdd8aa2678272b16e44cc91b212f1e59da8e33c573c6a5751e5c4be.jpg)  
Figure 25: We sample circles and compute closed line integrals as in the K¨ahler (complex manifold domain) experiment as in Appendix H.1 on the true solution. We sample circles around a pole and compute the circulation value. The left demonstrates the sampled circles around the pole, and the right demonstrates the corresponding values. Color is used to demonstrate radii values. As we can see on the right, the circulation is nonzero and approaches the theoretical value Γ .

![](images/aea1a9e9af6db760395d3dcaed537f45b4fea5afd5b06234c4e850bd568f2eda.jpg)

![](images/aa29787eaef3047b582299f29e40971b50eca66b0c741b9106b41c7882344d0d.jpg)  
Figure 26: We plot the same as Figure 25 but we perform the sampling on the scalar-valued PINN solution that fails and not the true solution. As we can see, the corresponding circulation is zero.

![](images/9573f5acc4b54d44603a96172ab7ece56437aed7ffb3123e0bf2fa61d27c5503.jpg)  
(1)

![](images/f35d7027364994c777a40708616e042494b50b58dacb8020d3fe237001bc1913.jpg)  
(2)

![](images/f064302280ef558d4df0dd9e54acd51daf52170d86860a330476749afe9e532e.jpg)  
(3)  
Figure 27: We illustrate (1) catastrophic failure of a vector-valued PINN; (2) the true solution; and (3) pointwise absolute error on the K¨ahler manifold experiment as in Appendix H.2 with a Pythagorean loss function after ∼ 8, 000 epochs.

![](images/317ad88ef22aaee1fcfe26ef19c44d106510c7ccd9188e13d915c4f569e50436.jpg)  
(1)

![](images/dcad62ea2e23f192cbb0b2e82bcf65952f560815c9d8a39015b5afcdd2e7f1ab.jpg)  
(2)

![](images/910a0db6be46e96d7f2ebd71cd356997eeb7f5550a4a7da9c5e886e1268ed5f9.jpg)  
(3)  
Figure 28: We illustrate (1) success of a Hodge-de Rham PINN; (2) the true solution; and (3) pointwise absolute error on the K¨ahler manifold experiment as in Appendix H.2 with a Pythagorean loss function after ∼ 2, 000 epochs.

![](images/6634a64345fc5cb0bbc6ea04168f3cfaf666ba925f2cb8b9d3c8b28c3dd87989.jpg)  
(1)

![](images/d6610caa0b02f209867bc984441195d45a8e039766e9c5e361dc1af0e9aaf4f0.jpg)  
(2)

![](images/5023ec8da27e6ee0be6922919563350e9c94998daabebeb48a86eb88aeca7611.jpg)  
(3)  
Figure 29: We plot the diference in value between two paths from one point to another, possibly encompassing poles. We plot (1) a toy diagram illustrating what we convey; (2) the diference in path values for a vector-valued vanilla PINN; (3) the diference in path values for the true solution. This figure illustrates the vector-valued PINN solution exhibits diferent path-dependent behavior than the true solution.

![](images/3b8f1efc13d049725b12c703a1cdf74e4deae2ae560babd957edd4733914c5d6.jpg)

![](images/0e7b307dd25d4679eb2fdcc3c5889a2b9142c5849937d0d47de8b74f197f5b35.jpg)

![](images/755c1bfe978787c2457f4e6c94c465afce21f711c63f35e21238eccaff643184.jpg)

![](images/2fb166613318c731ce71a53e5c1420d62a4ad757566ee0935ff6ee8d933d5aff.jpg)

![](images/fdc7be3b813a1f2cb9f2454a235b3491a6c6acb3b43835f8dd63aa62100a8f9c.jpg)  
Figure 30: We illustrate that the Aharonov-Bohm experiment true solution corresponds to a non-exact form. In particular, the bottom left panel would be exactly zero if the exterior derivative corresponded to an exact form, which is what the bottom right panel demonstrates.

![](images/e90cd79c5e87b30f95061c7fbc7363f84d418331e7514b48c8d535f887695ced.jpg)

![](images/c02c5b2bcd6fe747e14a06fb0b2e5b27b7eb0c9e593538326be8d02b5522528f.jpg)

![](images/d42b8d57cdda83cc6211d9c523393cd2095ab967bc803b68e333427c33187f78.jpg)

![](images/d248d945dae106330578dc52b40292a1be0edfa1f04735ff5fa132c322ff3588.jpg)

![](images/6aab74e295349aabb8525af567920953d765e30e6a4075e0ec7d8983a1b93d53.jpg)

![](images/15f850b1d1e8ccf37edc3415bd24d63637568e1ee83842f12c7b5a86aaa46c91.jpg)

![](images/80864de1f7bd57b7bc8b6777b4c811b9c57ed4caaa1fec575a82266933465037.jpg)

![](images/9aaa359a88e3a97186dabeea4a61f7ee8108c62a9f643cbe673901d5639cb737.jpg)

![](images/bd8383dfb8ca42f9f126a1a187dd151ce7986d7e7a09004aebd121b08538fd3b.jpg)  
Figure 31: We plot the Aharonov-Bohm experiment with two poles image across training iteration in (top) 200; (middle) 500; (bottom) 1,000 epoch chunks. The branch cut at initialization is due to the harmonic term. In particular, the branch cut at initialization on the left-hand side is intentional to facilitate the learning of the branch cut after training. Note that we use sin(·) activation. We do not use a Fourier feature embedding as in Tancik et al. (2020) Wang et al. (2023)

![](images/e7cb3428669c8cf938b06dc16d5ba873458b0239d817ead598690d530ad2364e.jpg)  
(1)

![](images/4ba473eafbe15ad6676168b5b83c809ec116363082f3c9c50ec6c25c9ea1d5ad.jpg)  
(2)

![](images/0506064c04c2014f156a320d1ff4176f399abf3137e5e98fa48fe49144e516e5.jpg)  
(3)  
Figure 32: We provide a supplemental experiment of success of our 1-form methods on the K¨ahler manifold experiment with six punctures to illustrate success of our methods on scenarios of many punctures.