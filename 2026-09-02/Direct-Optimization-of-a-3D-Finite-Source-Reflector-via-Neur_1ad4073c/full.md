# Direct Optimization of a 3D Finite-Source Reflector via Neural-Network Parameterization

ROEL HACKING,<sup>1,\*</sup> LISA KUSCH,<sup>1</sup> MARTIJN ANTHONISSEN,<sup>1</sup> AND WILBERT IJZERMAN<sup>1,2</sup>

<sup>1</sup>Eindhoven University of Technology, PO Box 513, 5600 MB Eindhoven, The Netherlands <sup>2</sup>Signify, High Tech Campus 7, 5656 AE Eindhoven, The Netherlands r.g.j.hacking@tue.nl

Abstract: We present a direct optimization method for three-dimensional freeform reflectors that transform the light of a finite-étendue source into a prescribed far-field angular intensity distribution. The reflector profile is represented by a small neural network (a multilayer perceptron), which is trained end-to-end through a diferentiable ray-tracing objective. We furthermore parameterize the emission directions in gnomonic coordinates, and show how we use this to ensure that every emitted ray intersects the reflector. At each iteration, the network is converted to a bicubic spline representation for ray-tracing eficiency, and intersections with this smooth surface are solved by a damped Newton solve—with gradients computed via the implicit function theorem. The traced output distribution is compared with the desired target on a ‘soft’ histogram, under an �<sup>−1</sup>-type spectral weighting that emphasizes long-range transport of flux to improve convergence. Optimization is performed using a BFGS method with self-scaled Broyden updates and a plateau-perturbation rule to prevent stalling. The method converges reliably within seconds on a single GPU for all examples tested.

## 1. Introduction

Many optical applications—including illumination, beam shaping, and solar concentration— depend on precise transformation of some source light distribution into some desired target distribution [1–3]. To produce such transformation, we wish to design optical components, such as reflectors and lenses with freeform surfaces, that efectively realize these desired light transformations. Most commonly, these components are designed under the simplifying assumption of zero-étendue sources, i.e., point or parallel sources. Under these simpler, idealized assumptions, the reflector design problem can be cast as an optimal transport problem: Glimm and Oliker [4] formulated the single-reflector problem as a Monge–Kantorovich mass transfer problem, and Gangbo and Oliker [5] showed the existence of optimal maps for reflector-type problems. Romijn et al. [6,7] developed numerical methods for freeform lens and reflector design, under the assumption of point sources and far-field targets, while Bösel et al. [8,9] contributed ray-mapping techniques for the design of freeform surfaces for collimated beams. Neural networks have also been used to solve the Monge–Ampère equation—the partial diferential equation characterizing these zero-étendue formulations—with its transport boundary condition [10], and to directly predict freeform surface topologies that produce a prescribed irradiance distribution [11].

In practice, however, most light sources—such as LEDs—have non-zero étendue. That is, they have both spatial and angular extent. The resulting target distribution becomes a blurred version of the ideal point or parallel source response [12]. Due to the conservation of étendue, this blurring imposes a fundamental limit on the achievable sharpness of any target distribution, and the simpler optimal transport formulation no longer applies—at least not directly. Several methods have been proposed to account for this finite-étendue blurring resulting from realworld light sources. For example, Fournier et al. [13] construct the reflector as a point-source solution—via the supporting-ellipsoid method of Kochengin and Oliker [14]—for a ‘virtual’ target distribution. They then iteratively adjust this virtual target until a finite-source ray trace reproduces the prescribed target distribution. Benamou et al. [15] later made this search over virtual targets gradient-based, minimizing an entropic optimal-transport error through the same type of point-source parameterization—though at the cost of an entropic transport solve in every loss evaluation, a restriction to convex reflectors, and, so far, a restriction to the two-dimensional setting.

Another, related, strategy comes from signal processing; we treat the finite-source efect as a black-box convolution and use generic iterative deconvolution methods to counteract the smearing efects imposed by the finite source. For example, we might use additive iterative deconvolution methods—such as van Cittert [16, 17]—or multiplicative iterative methods—such as Richardson–Lucy [18, 19]. Kronberg [20] designed an inverse freeform reflector design framework which uses such iterative deconvolution methods to account for scattering-surface efects, which are similar to the blurring efects of finite sources. The general framework used by Kronberg [20] can also be applied to finite-source efects, as shown for the two-dimensional case in [21] and, via a hybrid neural–deconvolution scheme, for the three-dimensional case in [22]. However, such deconvolution methods require solving a zero-étendue subproblem in every iteration, which is a one-dimensional ordinary diferential equation in the two-dimensional setting [21], but becomes a two-dimensional optimal transport or partial diferential equation (PDE) problem in the fully three-dimensional case, unless we assume rotational symmetry. Thus, the deconvolution route becomes less attractive in the fully three-dimensional setting, due to the high per-iteration cost of solving the zero-étendue subproblem. Moreover, we have no guarantee that the deconvolution will converge to a solution within this setting, and may even diverge for certain problems.

A third line of work applies diferentiable ray tracing to optical design problems. This approach formulates the forward ray-tracing pipeline as a diferentiable function, and uses some combination of automatic diferentiation and manual adjoint derivation to compute gradients of the optical design objective with respect to the surface parameters. This then allows us to use first- or second-order optimization methods to optimize the surface parameters directly, though the resulting optimization problem may be highly non-convex. Much of the research within this diferentiable ray tracing and rendering framework has come from the computer graphics community [23, 24]. Here, the goal is generally to optimize a variety of scene parameters—such as geometry, light positions, and material parameters—to match some desired rendered image. This general approach can also be applied to optical design problems. For example, de Koning et al. [25] and Heemels et al. [26] combined automatic diferentiation with B-splines for freeform optical optimization with finite sources, with the former using a custom diferentiable ray tracer and the latter using the Mitsuba 3 renderer [27]. Closer to our setting, Sun et al. [28] optimized mesh-based reflectors and lenses end to end through diferentiable ray tracing, with face-based optimal-transport updates that pull the surface out of poor local optima, and the DifRayFlow engine [29] combines discrete optimal transport with diferentiable ray tracing for freeform design.

This paper presents a fully three-dimensional method for finite-source reflector design with a far-field target distribution. Our main contributions are as follows:

1. A fully three-dimensional formulation treating the four-dimensional source phase space (2D spatial × 2D angular) directly, parameterized so that every emitted ray is guaranteed to intersect the reflector.

2. A mesh-free diferentiable forward ray-tracing pipeline: the reflector profile, parameterized by a multilayer perceptron (MLP), is converted to a bicubic spline representation each iteration, rays are intersected with this smooth surface by a damped Newton solve, and gradients are propagated back through the intersection via the implicit function theorem.

3. An $H ^ { - 1 }$ -weighted soft-histogram loss: ‘soft’ histograms of the desired target distribution and of the ray-traced output distribution are compared in the frequency domain, in a way that emphasizes long-range transport of flux to improve convergence.

4. Three numerical examples, every solve completing in 2 to 8 seconds of wall-clock time on a single GPU, demonstrating the method’s performance and robustness.

The remainder of the paper is organized as follows. Section 2 formulates the problem, Section 3 explains the method, Section 4 presents some numerical results, Section 5 discusses our findings, and we finally conclude in Section 6.

## 2. Problem formulation

We formulate the reflector design problem in the geometric-optics regime: we model light as rays that propagate in straight lines and reflect of specular surfaces according to the law of reflection. The source model and the coordinate conventions that we use are introduced in Section 2.1, the way we parameterize the reflector is described in Section 2.2, the ray–reflector intersection and the guarantee that every emitted ray hits the reflector are discussed in Section 2.3, and the forward ray-tracing routine and optimization objective are presented in Section 2.4.

## 2.1. Physical setting and source model

We model the light source and the reflector in a Cartesian coordinate system, with the source plane at $z = 0$ . The reflector is a specular surface above this source plane, and our goal is to design it such that the resulting far-field angular intensity distribution matches some prescribed target distribution. We follow some of the notation of [21, 22], and an overview of the symbols used in this paper is given in Table 1.

The planar source occupies a compact region $\Omega \subset \mathbb { R } ^ { 2 }$ on the $z = 0$ plane. Each point ${ \bf q } _ { s } = { }$ $( q _ { s , 1 } , q _ { s , 2 } ) \in \Omega$ emits light into a set of directions $\hat { \mathbf { s } } \in \mathbb { S } _ { + } ^ { 2 }$ , where $\mathbb { S } _ { + } ^ { 2 } = \{ \bar { \hat { \mathbf { s } } } \in \mathbb { R } ^ { 3 } : | \hat { \mathbf { s } } | = 1 , s _ { 3 } > 0 \}$ is the open upper unit hemisphere. We represent these emission directions by their gnomonic coordinates

$$
\mathbf { p } _ { s } = G ( \hat { \mathbf { s } } ) : = \frac { ( s _ { 1 } , s _ { 2 } ) } { s _ { 3 } } \in \mathbb { R } ^ { 2 } ,\tag{1}
$$

with the inverse

$$
G ^ { - 1 } ( \mathbf { p } _ { s } ) = \frac { ( p _ { s , 1 } , p _ { s , 2 } , 1 ) } { \sqrt { 1 + | \mathbf { p } _ { s } | ^ { 2 } } } .\tag{2}
$$

Geometrically, ${ \bf p } _ { s }$ is the central projection of sˆ from the origin onto the tangent plane $z = 1$ , such that $\vert { \bf p } _ { s } \vert = \tan \alpha$ where � is the polar angle between sˆ and the �-axis.

The set of admissible emission directions, expressed in gnomonic coordinates, forms the angular source domain $A \subset \mathbb { R } ^ { 2 }$ . The domain � represents the angular extent of the source: a small � corresponds to a nearly collimated beam, whereas a large � corresponds to a wide-angle emitter. The étendue of the source is determined jointly by Ω and �. The full source phase space is $S = \Omega \times A \subset \mathbb { R } ^ { 4 }$ , and the source radiance is described by a normalized distribution function $f : S  \mathbb { R } _ { \geq 0 }$ satisfying $\begin{array} { r } { \int _ { S } f ( \mathbf { q } _ { s } , \mathbf { p } _ { s } ) \mathrm { d } \mathbf { q } _ { s } \mathrm { d } \mathbf { p } _ { s } = 1 } \end{array}$

In related finite-source formulations [22], source emission directions were parameterized by stereographic projections from the south pole. Here we adopt gnomonic coordinates because the ray–reflector intersection equation (7) becomes afine in the emission angular coordinate ${ \bf p } _ { s }$ , which considerably simplifies the ray-coverage analysis of Section 2.3 and the boundary conditions. Stereographic and gnomonic coordinates are related by a smooth, invertible change of coordinates on ${ \mathbb S } _ { + } ^ { 2 }$ ; they therefore represent the same physical ray directions. For the reflected (output) directions, we retain the north-pole stereographic projection.

## 2.2. Reflector surface

A reflector over Ω is given by the parametric map (cf. [21, 22])

$$
\mathbf { r } ( \mathbf { q } _ { t } ) = \left( \mathbf { q } _ { t } , 0 \right) + u ( \mathbf { q } _ { t } ) \hat { \mathbf { d } } ( \mathbf { q } _ { t } ) ,\tag{3}
$$

where $\mathbf { q } _ { t } = ( q _ { t , 1 } , q _ { t , 2 } ) \in \Omega$ is the reflector surface parameter, $u : \Omega \to { \mathbb { R } _ { > 0 } }$ is the unknown profile function, and $\hat { \mathbf { d } } : \Omega \to \mathbb { S } _ { + } ^ { 2 }$ is a prescribed field of unit directions. The direction field is constructed from its gnomonic representation $\beta ( \mathbf { q } _ { t } ) = ( d _ { 1 } / d _ { 3 } , d _ { 2 } / d _ { 3 } ) \in \mathbb { R } ^ { 2 }$ via

$$
\hat { \mathbf { d } } ( \mathbf { q } _ { t } ) = G ^ { - 1 } ( \pmb { \beta } ( \mathbf { q } _ { t } ) ) = \frac { ( \beta _ { 1 } , \beta _ { 2 } , 1 ) } { \sqrt { 1 + | \pmb { \beta } | ^ { 2 } } } .\tag{4}
$$

Note that � is not a simple height function: the reflector point r lies at distance � from the source plane along the direction $\mathbf { \breve { d } } .$ , not vertically. The vertical component of the reflector point is $u ( \mathbf { q } _ { t } ) \cdot d _ { 3 } ( \mathbf { q } _ { t } )$

Let $\left[ q _ { 1 , \mathrm { m i n } } , q _ { 1 , \mathrm { m a x } } \right] \times \left[ q _ { 2 , \mathrm { m i n } } , q _ { 2 , \mathrm { m a x } } \right] \mathrm { a n d } \left[ p _ { 1 , \mathrm { m i n } } , p _ { 1 , \mathrm { m a x } } \right] \times \left[ p _ { 2 , \mathrm { m i n } } , p _ { 2 , \mathrm { m a x } } \right] \times \left[ q _ { 2 , \mathrm { m i n } } , q _ { 2 , \mathrm { m a x } } \right] \times \left[ p _ { 2 , \mathrm { m i n } } , q _ { 2 , \mathrm { m a x } } \right] \times \left[ p _ { 2 , \mathrm { m i n } } , q _ { 2 , \mathrm { m a x } } \right]$ be the axis-aligned bounding boxes of Ω and $A ;$ we define the direction field via the componentwise afine map

$$
\beta _ { i } ( { \bf q } _ { t } ) = p _ { i , \mathrm { m i n } } + \frac { p _ { i , \mathrm { m a x } } - p _ { i , \mathrm { m i n } } } { q _ { i , \mathrm { m a x } } - q _ { i , \mathrm { m i n } } } ( q _ { t , i } - q _ { i , \mathrm { m i n } } ) , \quad i = 1 , 2 ,\tag{5}
$$

which linearly maps the spatial bounding box to the angular bounding box. The geometric significance of this construction is explained in Section 2.3.

## 2.3. Ray–reflector intersection and coverage condition

Consider a ray emitted from $( \mathbf { q } _ { s } , 0 )$ in unit direction $\hat { \mathbf { s } } = G ^ { - 1 } ( \mathbf { p } _ { s } )$ , described by the ray equation

$$
\mathbf { x } ( \tau ) = ( \mathbf { q } _ { s } , 0 ) + \tau \hat { \bf s } , \qquad \tau \geq 0 .\tag{6}
$$

The ray hits the reflector at parameter point $\mathbf { q } _ { t } \mathrm { i f } \mathrm { \mathbf { x } } ( \tau ) = \mathrm { \mathbf { r } } ( \mathbf { q } _ { t } )$ for some $\tau \geq 0$ . The third component of this equality determines the ray parameter, $\tau = u ( \mathbf { q } _ { t } ) d _ { 3 } ( \mathbf { q } _ { t } ) / s _ { 3 } ;$ substituting � into the first two components and using $( s _ { 1 } , s _ { 2 } ) = s _ { 3 } { \bf p } _ { s }$ from (1) gives

$$
{ \bf q } _ { s } = { \bf q } _ { t } + u ( { \bf q } _ { t } ) d _ { 3 } ( { \bf q } _ { t } ) ( \pmb { \beta } ( { \bf q } _ { t } ) - { \bf p } _ { s } ) .\tag{7}
$$

This is the key geometric relation. It is afine in ${ \bf p } _ { s }$ for fixed ${ \bf q } _ { t }$ ; this is the reason for working in gnomonic coordinates, since in stereographic or other nonlinear angular coordinates the corresponding equation would be nonlinear in the angular variable. We define the shadow map $F _ { { \bf { p } } _ { s } } \left( { \bf { q } } _ { t } \right) = { \bf { q } } _ { t } + u ( { \bf { q } } _ { t } ) d _ { 3 } ( { \bf { q } } _ { t } ) \left( \beta ( { \bf { q } } _ { t } ) - { \bf { p } } _ { s } \right)$ , which gives the source-plane footprint of a reflector point q<sub>�</sub> as seen from emission direction ${ \bf p } _ { s }$

For the degenerate profile $u \equiv 0$ the shadow map reduces to the identity, $F _ { \mathbf { p } _ { s } } = \mathrm { i d }$ , which trivially covers every source point, regardless of ${ \bf p } _ { s }$ . As the profile grows continuously from zero to �, the shadow map keeps covering all of Ω as long as the image of the boundary �Ω never folds inward across Ω. This requirement yields the outward-lean boundary condition: for every $\mathbf { q } _ { b } \in \partial \Omega$ with outward unit normal n,

$$
\mathbf { n } \cdot { \boldsymbol { \beta } } ( \mathbf { q } _ { b } ) \geq \operatorname* { s u p } _ { \mathbf { p } _ { s } \in A } \mathbf { n } \cdot \mathbf { p } _ { s } .\tag{8}
$$

The interpretation is that the reflector direction at each boundary point must lean at least as far outward (in gnomonic coordinates) as the most oblique incoming ray in that normal direction. When this condition holds, every emitted ray—from any source point ${ \bf q } _ { s } \in \Omega$ in any direction ${ \bf p } _ { s } \in { \cal A }$ —is guaranteed to hit the reflector. On rectangular domains, the afine map (5) meets the condition exactly. If Ω or � is not rectangular (e.g., a disk), we simply apply the afine map to their axis-aligned bounding boxes; the direction field then leans somewhat farther outward than strictly necessary, but coverage is guaranteed, which is what we actually care about for the sake of optimization.

## 2.4. Forward ray tracing and the design problem

Given a source sample $( \mathbf { q } _ { s } , \mathbf { p } _ { s } ) \in S$ , the forward ray-tracing pipeline proceeds as follows.

1. Emit: ray origin $( \mathbf { q } _ { s } , 0 )$ , direction $\hat { \mathbf { s } } = G ^ { - 1 } ( \mathbf { p } _ { s } )$

2. Intersect: find the intersection of the ray with the reflector surface r.

3. Reflect: $\hat { \mathbf { t } } = \hat { \mathbf { s } } - 2 ( \hat { \mathbf { s } } \cdot \hat { \mathbf { n } } )$ nˆ, where nˆ is the unit surface normal at the intersection point; since $| \hat { \mathbf { s } } | = 1$ , the reflected direction <sup>ˆ</sup>t is again a unit vector.

4. Project: compute the output angular coordinate $\mathbf { p } _ { t } = P _ { N } ( \hat { \mathbf { t } } )$ via north-pole stereographic projection (since reflected rays have $t _ { 3 } < 0 ) !$

$$
\mathbf { p } _ { t } = { \frac { \left( t _ { 1 } , t _ { 2 } \right) } { 1 - t _ { 3 } } } .\tag{9}
$$

This defines the forward map $\mathbf { T } _ { \theta } : S  \mathcal { T } , ( \mathbf { q } _ { s } , \mathbf { p } _ { s } ) \mapsto ( \mathbf { q } _ { t } , \mathbf { p } _ { t } )$ , where � denotes the parameters of � (specified in Section 3.1), $B \subset \mathbb { R } ^ { 2 }$ is the far-field angular target domain (in north-pole stereographic coordinates), and $\mathcal { T } : = \Omega \times B$ is the full target domain.

The map $\mathbf { T } _ { \theta }$ pushes the source distribution � on S forward to a joint distribution $\tilde { g } _ { \theta } (  { \mathbf { q } } _ { t } ,  { \mathbf { p } } _ { t } )$ on $\mathcal { T }$ . The physically relevant quantity is the marginal far-field angular distribution $g _ { \boldsymbol { \theta } } ( \mathbf { p } _ { t } )$ obtained by integrating out the spatial reflector coordinate $\mathbf { q } _ { t }$ . Via the change-of-variables formula applied to $\mathbf { T } _ { \theta } ^ { - 1 }$ (cf. [22, Eq. (1)] and [21, Eq. (7)]),

$$
g _ { \pmb \theta } ( \mathbf { p } _ { t } ) = \int _ { \Omega } f \big ( \mathbf { T } _ { \pmb \theta } ^ { - 1 } ( \mathbf { q } _ { t } , \mathbf { p } _ { t } ) \big ) \left| \operatorname* { d e t } \frac { \partial \mathbf { T } _ { \pmb \theta } ^ { - 1 } } { \partial ( \mathbf { q } _ { t } , \mathbf { p } _ { t } ) } \right| \mathrm { d } \mathbf { q } _ { t } .\tag{10}
$$

This $g _ { \boldsymbol { \theta } } ( \mathbf { p } _ { t } )$ represents the far-field angular intensity distribution produced by the reflector with profile � . The design problem is to find � (parameterized by �) such that $g _ { \pmb \theta } ( \mathbf { p } _ { t } ) \approx \hat { g } ( \mathbf { p } _ { t } )$ for a prescribed target distribution ${ \hat { g } } .$ . Evaluating (10) directly requires the inverse map $\mathbf { T } _ { \pmb { \theta } } ^ { - 1 }$ and a two-dimensional quadrature over Ω for each target point $\mathbf { p } _ { t } $ ; Section 3.5 discusses this approach and its limitations. Instead, we adopt a sample-based forward approach: the source phase space is covered by � fixed quasi-Monte Carlo samples, each carrying the local source density � as a weight, and these are ray-traced through the reflector via $\mathbf { T } _ { \pmb { \theta } } ;$ the resulting weighted output distribution is compared to $\hat { g }$ via a diferentiable loss (Section 3.3).

![](images/3909bf73f61154e5b5837f7e1c3df903394742f57ea708f1e12239beb21e76f8.jpg)  
Fig. 1. Schematic diagram of the 3D finite-source reflector problem. A planar source $\mathbf { o n } z = 0$ emits rays from spatial points $\mathbf { q } _ { s } = ( q _ { s , 1 } , q _ { s , 2 } )$ in directions sˆ. Each ray strikes the reflector surface $\mathbf { r } ( \mathbf { q } _ { t } ) = \left( \mathbf { q } _ { t } , 0 \right) + u ( \mathbf { q } _ { t } ) \hat { \mathbf { d } } ( \mathbf { q } _ { t } )$ , reflects about the surface normal nˆ , and propagates to the far field. The reflected direction <sup>ˆ</sup>t is projected stereographically to give the far-field angular coordinate $\mathbf { p } _ { t } = \left( p _ { t , 1 } , p _ { t , 2 } \right)$

## 3. Method

This section details the four components of our approach: the neural-network parameterization of the reflector profile (Section 3.1), the spline-compiled surface and GPU ray tracing (Section 3.2),

Table 1. Notation and symbols.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\Omega$ </td><td>Spatial support of the planar source (source base domain)</td></tr><tr><td> $A$ </td><td>Angular source domain (gnomonic coordinates)</td></tr><tr><td> $S : = \Omega \times A$ </td><td>Full source domain</td></tr><tr><td> $B$ </td><td>Far-field angular target domain (north-pole stereographic coordi- nates)</td></tr><tr><td> $\mathcal { T } : = \Omega \times B$ </td><td>Full target domain</td></tr><tr><td> $f ( \mathbf { q } _ { s } , \mathbf { p } _ { s } )$ </td><td>Normalized source distribution function over  $s$ </td></tr><tr><td> $\pmb \theta$ </td><td>Parameters of the profile function u (e.g., MLP weights and biases)</td></tr><tr><td> $\tilde { g } _ { \theta } (  { \mathbf { q } } _ { t } ,  { \mathbf { p } } _ { t } )$ </td><td>Pushforward of  $f$  under  $\mathbf { T } _ { \theta } ;$  joint distribution over  $\mathcal { T }$ </td></tr><tr><td> $g _ { \boldsymbol { \theta } } ( \mathbf { p } _ { t } )$ </td><td>Marginal far-field angular distribution (integral of  $\tilde { g } _ { \theta }$  over</td></tr><tr><td> $\hat { g } ( \mathbf { p } _ { t } )$ </td><td>Prescribed (desired) far-field angular target distribution</td></tr><tr><td> ${ \bf q } _ { s }$ </td><td>Source spatial coordinate in Ω</td></tr><tr><td> ${ \bf q } _ { t }$ </td><td>Reflector surface parameter in Ω</td></tr><tr><td> ${ \bf p } _ { s }$ </td><td>Gnomonic emission coordinate in  $A$ </td></tr><tr><td> $\mathbf { p } _ { t }$ </td><td>Stereographic far-field coordinate in  $B$ </td></tr><tr><td> $\hat { \bf s }$ </td><td>Unit emission direction  $( \in \mathbb { S } _ { + } ^ { 2 } )$ </td></tr><tr><td> $\hat { \mathbf { t } }$ </td><td>Unit reflected direction  $( t _ { 3 } < 0 )$ </td></tr><tr><td> $\mathbf { r } ( \mathbf { q } _ { t } )$ </td><td>Reflector surface point for parameter  ${ \bf q } _ { t }$ </td></tr><tr><td> $u ( \mathbf { q } _ { t } )$ </td><td>Reflector profile function</td></tr><tr><td> $\hat { \mathbf { d } } ( \mathbf { q } _ { t } )$ </td><td>Unit direction field of the reflector  $( \in \mathbb { S } _ { + } ^ { 2 } )$ </td></tr><tr><td> $\beta ( { \bf q } _ { t } )$ </td><td>Gnomonic representation of the direction field</td></tr><tr><td> $\hat { \bf n }$ </td><td>Unit normal to the reflector surface</td></tr><tr><td> $\mathbf { T }$ </td><td>Forward map from source to target,  $s \to \mathcal { T } ;$  written  $\mathbf { T } _ { \theta }$  when emphasizing dependence on the profile parameters</td></tr><tr><td> $G , \ G ^ { - 1 }$ </td><td>Gnomonic projection and its inverse</td></tr><tr><td> $P _ { N }$ </td><td>North-pole stereographic projection</td></tr></table>

the $H ^ { - 1 }$ -weighted soft-histogram loss (Section 3.3), and the quasi-Newton optimizer (Section 3.4).   
Section 3.5 discusses the relationship to prior lower-dimensional formulations.

## 3.1. Neural-network parameterization of the profile function

We parameterize the profile function � by an MLP with parameters �. The network takes as input the two-dimensional reflector coordinate $\left( q _ { t , 1 } , q _ { t , 2 } \right)$ , passes it through two hidden layers of 24 units each with the activation function $\phi ( x ) : = \operatorname { t a n h } ^ { 2 } ( x )$ , and produces a single scalar output. The raw network output is combined with learnable scale, quadratic, and bias terms and passed through a sigmoid function to ensure $u \in [ u _ { \mathrm { m i n } } , u _ { \mathrm { m a x } } ]$

$$
u _ { \theta } ( \mathbf { q } _ { t } ) = u _ { \mathrm { m i n } } + ( u _ { \mathrm { m a x } } - u _ { \mathrm { m i n } } ) \sigma \big ( \lambda \mathrm { M L P } _ { \theta } ( \mathbf { q } _ { t } ) + \gamma | \mathbf { q } _ { t } | ^ { 2 } + h _ { \mathrm { l o g i t } } \big ) ,\tag{11}
$$

where $\sigma$ is the sigmoid function, � (initialized to 0.1) is a learnable linear scale, $\gamma$ (initialized to 0.5) is a learnable quadratic scale, and $h _ { \mathrm { l o g i t } }$ is initialized so that $\sigma ( h _ { \mathrm { l o g i t } } ) = ( u _ { \mathrm { i n i t } } - u _ { \mathrm { m i n } } ) / ( u _ { \mathrm { m a x } } - u _ { \mathrm { m i n } } )$ Together with the network’s 697 weights and biases, � comprises 700 trainable parameters. In all experiments we set $u _ { \mathrm { m i n } } = 0 . 1 , u _ { \mathrm { m a x } } = 8 . 0$ , and $u _ { \mathrm { i n i t } } = 2 . 0 ;$ the same initial profile value is used for every example.

At initialization, ML $\mathbf { P } _ { \pmb { \theta } } ( \mathbf { q } _ { t } ) \approx 0$ across Ω and � is small, so $u _ { \pmb \theta } ( { \bf q } _ { t } ) \approx u _ { \mathrm { i n i t } }$ up to a small quadratic perturbation. This yields a near-constant profile that places the reflector at approximately uniform distance from the source plane along the direction field, providing a physically reasonable starting point for the optimization.

## 3.2. Spline-compiled surface and GPU ray tracing

Instead of triangulating the surface, the MLP profile $u _ { \theta }$ is compiled each iteration: it is evaluated on a regular 64 × 64 grid over the padded bounding box of $\Omega ,$ , and the resulting values define a bicubic B-spline interpolant of the profile. All ray tracing is performed against this smooth approximation surface directly. Because the intersection relation (7) is afine in ${ \bf p } _ { s }$ and only slightly nonlinear in $\mathbf { q } _ { t }$ , the intersection parameter ${ \bf q } _ { t }$ of each ray is found by a damped twodimensional Newton iteration. This requires relatively few iterations to converge to a solution that is suficiently accurate for our applications. The surface normal at the hit point then comes directly from the analytic gradient of the spline interpolant—though it is of course only an approximation of the ML $\mathbf { \varPhi } \mathbf { \bar { s } }$ normal at that point.

Gradient flow through the intersection uses the implicit function theorem: diferentiating (7) at the converged point yields the sensitivity of ${ \bf q } _ { t }$ (and thus the reflected direction) to local perturbations of the reflector profile and its normal in closed form. Simply using automatic diferentiation here—which would diferentiate through the iterative algorithm—might still be suficient as well, but the implicit function approach should be faster and more accurate, especially as the number of inner Newton iterations is increased. The complete pipeline—reflector profile compilation, Newton intersection, reflection, histogram accumulation—runs on a single GPU. On our hardware, a full loss-and-gradient evaluation with $N = 2 ^ { 2 2 }$ rays takes well under a millisecond.

## 3.3. $H ^ { - 1 }$ -weighted soft-histogram loss

We require a diferentiable loss that compares the weighted empirical distribution of the � ray-traced output samples $\{ \mathbf { p } _ { t } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ against a target distribution ${ \hat { g } } .$ . Standard histograms with hard indicator-function bins are non-diferentiable at the bin boundaries. We therefore accumulate rays in a ‘soft’ histogram: each sample deposits its source weight onto a grid of $M \times M$ bins $( M = 1 2 8 )$ through compactly supported B-spline kernels, such that the bin contents $h _ { j k } ( \pmb \theta )$ are smooth functions of the sample positions. This is necessary in order for the objective function to be optimized through gradient-based optimization algorithms. We choose this B-spline kernel as it is both smooth and compactly supported, which keeps the computation relatively cheap and comparable to a standard histogram. Related diferentiable histogram constructions have been explored by Avi-Aharon et al. [30], and B-spline-based density estimation by Kirkby et al. [31] and Zhao et al. [32]. A reference histogram $h ^ { \mathrm { r e f } }$ is precomputed once from the target distribution using the same binning and B-spline kernel, such that both sides of the comparison carry the same kernel smoothing. This ensures that an exact solution—if possible—would obtain a discrepancy of exactly 0.

Rather than penalizing the residual $h - h ^ { \mathrm { r e f } }$ uniformly across frequencies, we measure it under an $H ^ { - 1 }$ -type spectral weighting (a weighting that emphasizes low spatial frequencies),

$$
\mathcal { L } ( \pmb { \theta } ) = \frac { 1 } { \| h ^ { \mathrm { r e f } } \| _ { 2 } ^ { 2 } M ^ { 2 } } \sum _ { \mathbf { k } } \left( \frac { 1 } { 1 + | \mathbf { k } | ^ { 2 } / k _ { 0 } ^ { 2 } } + \epsilon _ { 0 } \right) \left| ( \overline { { h - h ^ { \mathrm { r e f } } } } ) ( \mathbf { k } ) \right| ^ { 2 } ,\tag{12}
$$

where $\hat { \cdot }$ denotes the discrete Fourier transform over the bin grid, k the spatial frequency, $k _ { 0 } = 1$ (one cycle across the histogram window), and $\epsilon _ { 0 } = 0 . 0 2$ a small flat floor. The low-pass factor is the discrete analogue of a squared $H ^ { - 1 }$ norm and encodes the transport nature of the problem, without requiring expensive Wasserstein distance or Sinkhorn divergence computations to be performed repeatedly. This formulation ensures that, even when the achieved and desired distributions are far apart in terms of Wasserstein distance, there is still a good gradient signal pushing the ray-traced distribution in the right direction by allowing it to consider longer-range transport, which more standard losses might miss. The final auxiliary loss term simply encourages the optimizer to push as much reflected flux as possible into the desired domain, which the main histogram loss is blind to. Each ray that fails to reach the desired target domain contributes to an escaping-flux fraction, with a weight that grows smoothly from zero with the distance by which that ray overshoots the domain boundary. This squared fraction is then added to the loss.

For evaluation purposes (as opposed to training), we use a standard hard histogram with $M _ { \mathrm { e v a l } } = 1 2 8$ bins per axis and report the relative $L ^ { \tilde { 2 } }$ error

$$
\varepsilon _ { L 2 } = { \frac { \| h - h ^ { \mathrm { r e f } } \| _ { 2 } } { \| h ^ { \mathrm { r e f } } \| _ { 2 } } } ,\tag{13}
$$

where ℎ and $h ^ { \mathrm { r e f } }$ now denote the hard histograms at resolution $M _ { \mathrm { e v a l } } .$ , computed from $2 ^ { 2 6 }$ fresh quasi-Monte Carlo samples independent of those used during training.

## 3.4. Quasi-Newton optimization

The quasi-Newton update uses a self-scaled Broyden family method. Self-scaled quasi-Newton methods, introduced by Oren and Luenberger [33] and further developed by Al-Baali [34], multiply the current Hessian-inverse approximation by a scalar factor before performing the standard secant update. We use the self-scaled Broyden variant of Urbán et al. [35], which performed best in their study of quasi-Newton training for physics-informed neural networks, where such methods substantially outperformed first-order optimizers such as Adam [36]. Step sizes come from the bracketing weak-Wolfe line search of Lewis and Overton [37]. The initial step size is warm-started from the previously accepted line-search step size, and the gradients of line-search trial points are computed only when the suficient-decrease (Armijo) condition is satisfied to avoid unnecessary gradient evaluations for rejected trials. When the search fails, which happens on non-convex landscapes, the optimizer falls back to a Hessian reset, then Armijo backtracking, then a fixed step.

Two mechanisms improve the reliability of the optimization on this non-convex landscape. First, a plateau-perturbation (basin-hopping) rule: whenever progress stalls, the iterate is perturbed by a small random step (with amplitude decreasing over successive perturbations, on a deterministic schedule) and the Hessian approximation is reset; each perturbation launches from—and the run finally returns to—the best iterate encountered, so the best solution found is never lost. This reliably restores progress where plain descent otherwise stalls. Runs initialized from the solution at a previous radius (the continuation strategy of Section 4.4) skip these perturbations: the continuation exists to keep the solution in the inherited basin, and perturbing a near-converged solution tends to find basins that improve the training loss only by exploiting the particular fixed sample set. Second, we penalize large pre-sigmoid values in (11): a profile pushed against $u _ { \mathrm { { m i n } } }$ or $u _ { \mathrm { m a x } }$ saturates the sigmoid, so gradients through it vanish and the optimizer cannot pull the profile back from the bound. Because the penalty acts on the pre-sigmoid values directly, it keeps pushing the profile away from the bounds even where the sigmoid has saturated.

Source samples are generated once from a deterministic low-discrepancy point set (a rank-one lattice with a seed-dependent random shift) and held fixed throughout optimization. This deterministic sampling is necessary for standard quasi-Newton methods, as they assume deterministic objectives; stochastic objectives would corrupt the curvature information and break the assumptions of the line search. Stochastic quasi-Newton variants [38] do exist, but we opted here for determinism as the ray tracer is suficiently fast to allow for large sample sizes, and the deterministic objective then allows us to converge faster and more reliably.

## 3.5. Relation to lower-dimensional formulations

Earlier lower-dimensional formulations [21] can—to some extent—exploit symmetry to reduce the source and target phase spaces from four dimensions (2D spatial × 2D angular) to two dimensions (1D spatial × 1D angular), which enables inverse-mapping- and quadrature-based losses at manageable cost. In the fully three-dimensional setting treated here, those constructions get expensive: the marginal far-field distribution in (10) needs a two-dimensional integral over Ω for each target location, and constructions based on intersecting source and target cells would have to operate in the four-dimensional phase space. Such constructions may still be extended to 3D, but would require more careful attention to the computational cost and eficiency of the quadrature algorithms to avoid excessive runtime.

The present method therefore takes a diferent route: a forward, sample-based pipeline with hardware-accelerated ray tracing and a smooth soft-histogram discrepancy. It evaluates no inverse maps and no four-dimensional cell intersections, has no trouble with discontinuous source distributions, and suits GPU-based quasi-Newton optimization. Both lines of work use compact neural parameterizations and quasi-Newton updates, but the objective construction and the optimization loop have little else in common.

## 4. Numerical results

We demonstrate the method on three examples of increasing complexity: a Gaussian-to-uniform redistribution (Section 4.2), a yin-yang image projection (Section 4.3), and a systematic sweep over the angular source extent (Section 4.4). Implementation details common to all examples are given in Section 4.1.

## 4.1. Implementation details

The method is implemented in CUDA C++: profile compilation, Newton intersection, reflection, histogram accumulation, the adjoint, and the network itself all execute on the GPU, with the optimizer state held in double precision on the host. Surface and ray computations use single precision (float32) on a single NVIDIA RTX 4090 GPU. All examples use $N = 2 ^ { 2 2 }$ quasi-Monte Carlo source samples and 128 × 128 soft-histogram bins, and there are no per-example settings: every hyperparameter, including the initial profile value and the loss-histogram resolution, is identical across the three examples. The only per-example diference is the wall-clock budget, which serves as the stopping criterion: 2 seconds per run for Examples A and B, and 8 seconds per radius for Example C. Because the order of parallel floating-point additions varies from run to run on a GPU, histogram and adjoint sums are accumulated in fixed-point integer arithmetic (integer addition is exactly associative), so the objective—and with it the entire optimization trajectory—is bitwise reproducible across runs on a given GPU.

## 4.2. Example A: truncated Gaussian source, uniform target

The first example uses a square spatial domain $\Omega = [ - 0 . 3 , 0 . 3 ] ^ { 2 }$ and a circular angular domain � of gnomonic radius 0.75 (cone half-angle arctan $0 . 7 5 \approx 3 7 ^ { \circ } )$ . The source distribution is a truncated Gaussian with zero mean and covariance matrix $0 . 2 I _ { 4 }$ on $\Omega \times A$ , and the target is a uniform distribution $\hat { g }$ on $[ - 0 . 3 , 0 . 3 ] ^ { 2 }$ in north-pole stereographic coordinates.

The optimizer reaches a relative $L ^ { 2 }$ error of $\varepsilon _ { L 2 } = 3 . 6 \%$ within the 2-second budget. Figure 2 shows the convergence history, the evolution of the reflector surface and the ray-traced far-field distribution at selected times, and the pointwise signed error with respect to the target. The error drops below $7 \%$ within the first quarter second and continues to improve steadily until the budget is exhausted. Throughout the optimization, the optimizer adapts the curvature of the reflector to redistribute flux from the center—where the Gaussian source is densest—toward the edges, in order to match the prescribed uniform target.

## 4.3. Example B: uniform source, yin-yang image target

The second example uses a circular spatial domain Ω of radius 0.3 and the same circular angular domain � of gnomonic radius 0.75 as Example A, with a uniform source distribution on $\Omega \times A$ . The target distribution �ˆ is a yin-yang image $( 5 1 2 \times 5 1 2$ grayscale, piecewise-constant interpolation) defined on $[ - 0 . 3 , 0 . 3 ] ^ { 2 }$

The optimizer converges to $\varepsilon _ { L 2 } = 6 . 4 \%$ within the 2-second budget. The results are shown in Figure 3. The yin-yang pattern emerges progressively from the initial near-uniform distribution as the reflector surface is shaped by the optimizer. The final result captures the main features of the prescribed yin-yang target, namely the two interlocking regions and the small dots contained within, though with visible blurring and some ringing artifacts, which we believe are inevitable given the non-zero étendue of the source. As an exact match is likely impossible, the optimizer instead finds a solution that minimizes the specified objective. Alternative objectives may therefore yield diferent trade-ofs between the various features of the target, though we leave such exploration to future work.

## 4.4. Example C: angular domain radius sweep

Examples A and B both use a fixed angular domain radius of $r _ { A } = 0 . 7 5$ . To study how the optimization result depends on the angular source extent, Example C systematically varies this angular radius over a wide range. The spatial domain is a disk Ω of radius 0.3, the source distribution is uniform on $\Omega \times A ( r _ { A } )$ where $A ( r _ { A } )$ denotes a disk of gnomonic radius $r _ { A }$ , and the target is uniform on $[ - 0 . 3 , 0 . 3 ] ^ { 2 }$ . The radius is swept over ten equally spaced values $r _ { A } = 2 k / 9$ $k = 0 , \ldots , 9$ , from the collimated limit $r _ { A } = 0$ up to $r _ { A } = 2$ (cone half-angle ≈ 63<sup>◦</sup>). Apart from the collimated case, which is solved independently, we proceed by continuation: we first optimize at the smallest non-zero radius, then sweep outward, initializing each solve from the solution at the preceding radius.

The results are shown in Figure 4 and summarized in Table 2. At $r _ { A } = 0$ (a parallel source in which all rays travel in the same direction), the relative $L ^ { 2 }$ error is $\varepsilon _ { L 2 } = 2 . 9 \%$ , the smallest across the sweep, consistent with the fact that the parallel case approaches a collimated-beam reflector problem for which highly accurate solutions are known to exist [8, 9]. As $r _ { A }$ increases, the error grows monotonically and gently, from 3.0% at $r _ { A } = 0 . 2 2$ to 4.0% at $r _ { A } = 2 . 0 \mathrm { - } \mathrm { a }$ factor of only 1.3 while the angular source extent grows by an order of magnitude. The monotone growth of the error is consistent with our physical expectation: as the angular extent of the source is increased, the attainable sharpness of the target is limited by the source étendue, and the optimizer must find solutions that are near this limit across the entire sweep.

(a)  
![](images/9431a42aebcfd5a245c1ed1ae77aac735f5c7c3a1214cb08cc0a389f88d91daa.jpg)

![](images/a9e425998a820face91f6191c504109900e24aec2a258f4dd88d89ee84c6c588.jpg)

![](images/772e1d0f3ad00fbcf9e4fe85fc60145715f59fdc76865836f41245ef6c2fd89b.jpg)

(c)  
![](images/ad921e3cbcae4a22a0faf06fa080c2924fe0c22fcb0c3f37933f3b9b3293a4a3.jpg)

![](images/c24fec1d6c76f2d571d2b045920f0f03f21124cbc333c4dee739243512da6445.jpg)

![](images/c5a1b69309147fdb9af96f570946c9e060195d6a8aa880e2136ad363a14d76bb.jpg)

![](images/c0e0131d2a79240bb9d2b06f85b0f1c56104188202751027d0719f386b99fcd4.jpg)

![](images/3bf680a76600c1d0862e23feea2c9dd6a3f125084737f8bc6505c07965aeeac2.jpg)

(d)  
![](images/33ca63eb8581bc2614ff3befbb8080275ad05e388e7b720d395bb08e15411019.jpg)

![](images/9f05579268bc27e21daad636ce551bc43ca9cd71b584ba55afcadd047f0e163e.jpg)

![](images/7b7f9d1144fc7bea3133095f90f2b4627d8db6195640a4082002a13c7918fb27.jpg)

![](images/e96d8d8f56416b779d9d15ec24dd039a2364bd53bfba8bc5cf7611d90a52442f.jpg)

![](images/d638de09ce7d6c895ceaaba1b7452d4d946dd8547878e79eb07e16311dde48d0.jpg)

(e)  
![](images/c0846264d720a41826c0a93a29bfed688dfb9618438489032a6705b74d97eb48.jpg)

![](images/f913ad585d8cc35bd7aef1de6203f723730fca6af985fad44b5b5751af8c3505.jpg)

![](images/f95bfb0b3decdb290fdfcd5fe9254b1591966f503698dffdb4b163090980007f.jpg)

![](images/916c30c560182961089f31296bf0194c08a86132f4ef8113f4ce4e707eb737d5.jpg)

![](images/1378398a9127d181bc36a8e860f75ac50d0c1102fff36d366b9afc7f0e835a06.jpg)  
Fig. 2. Example A: truncated-Gaussian source, uniform target. (a) Relative $L ^ { 2 }$ error over wall-clock time; the vertical dotted lines mark the snapshot times shown below. (b) Target distribution. (c) Reflector surface at selected times. (d) Ray-traced far-field distributions; the first panel uses its own color scale (peak annotated), while the remaining panels share a common scale with (b). (e) Signed error $e ~ = ~ h - h ^ { \mathrm { r e f } }$ individually scaled to ± max |�| per panel.

(a)  
![](images/f94cef0d25b3480ec48d93c8925f6201d81d160b2dd9f703bb9461ec7a23ca92.jpg)

![](images/4fce984d88f7bf22b82e99b148952e0deef3897e5cd89f513c47551c4b5fa7c4.jpg)

![](images/7e929023795ee8bbfae49d627bb3d884ee5f4268f08d2fc628f37b4ecf130764.jpg)

(c)  
![](images/b8ad0d48d7c1517c8500abd3223b5fec2166412ab1efd2a033ea4795909ec818.jpg)

![](images/4f2a482319c7427c5b2abbf419e485af3cb29c7c29ffa54305885ff909df8cd0.jpg)

![](images/5ea7e2fa2aec6b39c224965ac715fb3c7dcf560e58c77f0e71917506171a2992.jpg)

![](images/3e3176366aa6585df8af0feb496101759a83457dd855a01628490881a1a81e6c.jpg)

![](images/e846b66d442dc584eae1e652c04871e74655a73286f40e0cc73cd55151ae6d79.jpg)

(d)  
![](images/cc79ae34e9936d61817917e0023a93770b938e378e83dd4eacd9b34a4819ba2f.jpg)

![](images/add5f1520199a0561368cb9054a4e31de5e99da97d1a684b36443978aeffce31.jpg)

![](images/2fe7423c6b91880a574c1956c4766d26ad3c634004ebe0777bf96bffee6b9847.jpg)

![](images/86d60460b8d13962c023ae4258ec531b8364ee1f187fac5bf53f4619b95ef6f9.jpg)

![](images/6cbbff747f69b9c240ff2f239d2efa506c67cde3b0c4c6d2082d450e8e531f12.jpg)

(e)  
![](images/f33a75dcbfcdcbc63e59bd1cc4744cfdd715774b222b1497ffc6da24fa83c33e.jpg)

![](images/95ad3eb34afbd38a6eb280da237cd6e36114b2624953c2e74cb1c5f54556e37f.jpg)

![](images/7c0d876b9c4945c82ae2f22b8cff615d99c4a353863f43bd5bce67f6df8693ef.jpg)

![](images/84c12376e099628cbeb4ce6437880f8527e82ab5f1df34384425978fab971862.jpg)

![](images/32efe24c1ccd7fd0a8d81a80084f4b90dbcac0d0425f2474be63290bacb79533.jpg)  
Fig. 3. Example B: uniform source, yin-yang image target. (a) Relative $L ^ { 2 }$ error over wall-clock time; the vertical dotted lines mark the snapshot times shown below. (b) Target distribution. (c) Reflector surface at selected times. (d) Ray-traced far-field distributions; the first panel uses its own color scale (peak annotated), while the remaining panels share a common scale with (b). (e) Signed error, individually scaled to ± max |�| per panel.

Table 2. Summary of numerical results.
<table><tr><td>Example Ω</td><td></td><td>A</td><td>Source</td><td>Target</td><td>N</td><td> $\varepsilon _ { L 2 }$ </td><td>Time (s)</td></tr><tr><td>A</td><td>box  $[ - 0 . 3 , 0 . 3 ] ^ { 2 }$ </td><td> $\mathrm { d i s k } , r = 0 . 7 5$ </td><td>Trunc. Gauss.</td><td>Uniform</td><td> $2 ^ { 2 2 }$ </td><td>3.6%</td><td>2.0</td></tr><tr><td>B</td><td> $\mathrm { d i s k } , r = 0 . 3$ </td><td> $\mathrm { d i s k } , r = 0 . 7 5$ </td><td>Uniform</td><td>Yin-yang</td><td> $2 ^ { 2 2 }$ </td><td>6.4%</td><td>2.0</td></tr><tr><td> $\mathrm { ~ C ~ } ( r _ { A } = 0 )$ </td><td> $\mathrm { d i s k } , r = 0 . 3$ </td><td> $\mathrm { d i s k } , r _ { A } = 0$ </td><td>Uniform</td><td>Uniform</td><td> $2 ^ { 2 2 }$ </td><td>2.9%</td><td>8.0</td></tr><tr><td> $\mathrm { ~ C ~ } ( r _ { A } = 2 )$ </td><td> $\mathrm { d i s k } , r = 0 . 3$ </td><td> $\mathrm { d i s k } , r _ { A } = 2$ </td><td>Uniform</td><td>Uniform</td><td> $2 ^ { 2 2 }$ </td><td>4.0%</td><td>8.0</td></tr></table>

## 5. Discussion

In the numerical experiments the method converges within 2 to 8 seconds of wall-clock time on a single GPU. Three things make this possible: the spline-compiled surface with its cheap Newton intersection (a full loss-and-gradient evaluation with $2 ^ { 2 2 }$ rays takes well under a millisecond), the fully GPU-resident pipeline, and the compactly supported soft-histogram accumulation.

While those three components ensure that objective and gradient computations are fast, it is also important that our optimization can converge reliably in relatively few iterations. We believe there are three main components that allow for this. Firstly, our use of gnomonic coordinates yields an afine ray–reflector intersection equation, which ensures all rays intersect the reflector regardless of the current neural network parameter vector, and avoids any potential discontinuities in the objective function resulting from varying visibility. Secondly, the $H ^ { - 1 }$ -weighted loss additionally improves the reliability of the optimization by supplying a long-range transport signal, which dominates during early optimization (when there is a greater mismatch between achieved and desired target distributions). Finally, our use of plateau-perturbation helps to eliminate any further failures or premature optimizer terminations which we otherwise sometimes observed.

Compared with prior lower-dimensional inverse-mapping formulations, the present approach uses a forward ray-tracing pipeline combined with a soft-histogram loss. This change is motivated by the increased dimensionality: in 3D without rotational symmetry, evaluating the marginal far-field distribution $g _ { \theta } ( \mathbf { p } _ { t } )$ via (10) requires repeated two-dimensional quadrature over Ω for each far-field point, and the integrand develops singular behavior as the angular source domain shrinks. The forward approach avoids these issues entirely and, being based on sample statistics rather than pointwise density evaluation, naturally handles discontinuous source distributions. Compact $\bf M L P$ parameterizations (approximately 700 parameters) remain efective in this fully three-dimensional setting. As noted previously, eficient inverse-mapping constructions may still be efective, but would require carefully designed domain-specific quadrature methods to reduce the number of evaluation points if we wish to achieve performance similar to or better than that of the present forward map approach.

Several limitations of the current method should be noted. The optimization landscape is non-convex, and the quality of the reached optimum can still depend on the basin entered early in the run; the bitwise-reproducible pipeline implemented here makes this dependence deterministic rather than stochastic (at least within the same environment), but it surfaces as sensitivity to the initial profile height: in Example A, diferent choices of $u _ { \mathrm { i n i t } }$ reach local optima with errors between 3.2% and 6.5%. In the collimated limit of Example C, the residual error concentrates at the corners of the target, where the ray mapping must carry the smooth circular boundary of the source disk into the sharp corners of the square. The smooth neural network struggles to represent such examples accurately, though for larger étendue cases this is less relevant, as it is no longer the dominant error. Finally, the profile is represented through a fixed $6 4 \times 6 4$ spline compilation grid, which bounds the finest surface detail the optimizer can express.

(a)  
![](images/6fc5b761c17e5f2d32ea380e86d3083a889d40b7c1687231c3578f324b3567ec.jpg)

![](images/15ce8beca415fe48a2daf4701d03686cd63ff9287dd5eb57fa3f1c18cfd95518.jpg)

![](images/a485ae5c8c605daf6f3532cd1b12bbf15393db280cc2145cca49a808e43bc92f.jpg)

(c)  
![](images/08a2dbe1ed7969eec5afb3120860bce51e3a5fc899845108cd244df6f0fd60e8.jpg)

![](images/841daacd7cf924a9509dc33db776bbf873c26419096f0d8af0707f111f4d63ae.jpg)

![](images/5d5a0281790d359a84fe07ee0720f31e397575e34b6a88dd981c5a80f2c957f2.jpg)

![](images/c0c69f153989e5ef44ff27f1dd282151fa2db6af8492e86833c8175b50e38f30.jpg)

![](images/e62f5eee3da3f349739d98fda21931c1581066875132b7bf59e8620e04f3005b.jpg)

(d)  
![](images/b03f74262e0ac6c3a2f5dd2257c6180cdba3a21e6874f53616f8bcbb1a2e1a83.jpg)

![](images/e9b301fb5e0983b190f14f710df394e6dc689bab157d09851da076a3e3d0e503.jpg)

![](images/5b062a784b8e6f3a3d237e91856ef343228705f04e84a2302a872e6d414d07de.jpg)

![](images/5e61d8aa60f554e01d7e4fdc1cb2864faa53a6cf3b0452b7b8cdaca1c579fd3e.jpg)

![](images/892fa3d32ea6b866a17408a087b56513432437ddaa375c28ab23000b2d4f832f.jpg)

(e)  
![](images/e5223a87471159079538751152bf4ce199c04cfd4d42044caa7d54125190591d.jpg)

![](images/0ac92ad8084291bf498629bb535f8461cb1b885198439c8aedb89dd60f3a49ce.jpg)

![](images/85c6192809ecf710010ddbd51c9902582908f9a3bb1cc87bb37cf8fc2c8ff8d9.jpg)

![](images/214f173d2954106792f780f024d24da7877dea0fa0db910439f55287f8f04229.jpg)

![](images/db5208e34554fe779f7e9e0dec99b412d5e5a1b0f5aab94e024e67ac1332c768.jpg)  
Fig. 4. Example C: angular domain radius sweep. (a) Relative $L ^ { 2 }$ error versus angular radius; the vertical dotted lines mark the radii shown below. (b) Target distribution. (c) Reflector surface at selected radii, sampled over the illuminated source disk; each panel uses its own �-axis scale (annotated). (d) Ray-traced far-field distributions (common color scale). (e) Signed error, individually scaled to ± max |�| per panel.

We see several directions for future work. An adaptive spline compilation grid, concentrating resolution where the reflector curvature is highest, could improve accuracy at little extra cost, though at increased implementational complexity. Convexity-structured parameterizations, in the spirit of the convexity-constrained neural Monge–Ampère solver of [10], could reduce the remaining basin dependence of the non-convex landscape. Additionally, the currently unconstrained reflector shape may develop geometries that result in rays bouncing multiple times within the reflector, which the current model does not account for. Though we have not observed this problem during our experiments, constrained geometries (such as convexity constraints) may still be useful for more formally ensuring that rays may not bounce multiple times within the reflector. Finally, as noted previously, non-zero étendue sources may no longer admit exact solutions, and our choice of objective function therefore influences how we distribute this irreducible error. For example, we may wish to reduce the ringing observed in some of our results, and future research might therefore investigate alternative distribution discrepancies to distribute the error in a more desirable fashion.

## 6. Conclusion

We have presented a direct optimization method for fully three-dimensional finite-source reflector design, built on a neural surface parameterization compiled to smooth spline surfaces. The eficient diferentiable ray-tracing formulation we developed around this representation, based on an $H ^ { - 1 }$ -weighted soft-histogram loss, allows us to eficiently optimize close-to-zero étendue sources, as well as sources with both significant spatial and angular extent. Our method converges within just a few seconds on a single GPU across several examples, using a deterministic and exactly reproducible pipeline.

Funding. This work in the project MALIOD is funded by Holland High Tech—TKI HSTM via the PPS allowance scheme for public–private partnerships.

Disclosures. The authors declare no conflicts of interest.

Data availability. Data underlying the results presented in this paper are not publicly available at this time but may be obtained from the authors upon reasonable request.

## References

1. W. T. Welford and R. Winston, High Collection Nonimaging Optics (Academic Press, 1989).

2. R. Winston, J. C. Miñano, and P. Benítez, Nonimaging Optics (Elsevier Academic Press, 2005).

3. J. Chaves, Introduction to Nonimaging Optics (CRC Press, 2015), 2nd ed.

4. T. Glimm and V. Oliker, “Optical design of single reflector systems and the Monge–Kantorovich mass transfer problem,” J. Math. Sci. 117, 4096–4108 (2003).

5. W. Gangbo and V. Oliker, “Existence of optimal maps in the reflector-type problems,” ESAIM: Control. Optimisation Calc. Var. 13, 93–106 (2007).

6. L. B. Romijn, J. H. M. ten Thije Boonkkamp, and W. L. IJzerman, “Freeform lens design for a point source and far-field target,” J. Opt. Soc. Am. A 36, 1926 (2019).

7. L. B. Romijn, J. H. M. ten Thije Boonkkamp, and W. L. IJzerman, “Inverse reflector design for a point source and far-field target,” J. Comput. Phys. 408, 109283 (2020)

8. C. Bösel and H. Gross, “Ray mapping approach for the eficient design of continuous freeform surfaces,” Opt. Express 24, 14271 (2016).

9. C. Bösel, N. G. Worku, and H. Gross, “Ray-mapping approach in double freeform surface design for collimated beam shaping beyond the paraxial approximation,” Appl. Opt. 56, 3679 (2017).

10. R. Hacking, L. Kusch, K. Mitra, et al., “A neural network approach for solving the Monge–Ampère equation with transport boundary condition,” J. Comput. Math. Data Sci. 15, 100119 (2025).

11. J. Cerpentier and Y. Meuret, “Freeform surface topology prediction for prescribed illumination via semi-supervised learning,” Opt. Express 32, 6350 (2024).

12. S. Wei, Z. Zhu, W. Li, and D. Ma, “Compact freeform illumination optics design by deblurring the response of extended sources,” Opt. Lett. 46, 2770 (2021).

13. F. R. Fournier, W. J. Cassarly, and J. P. Rolland, “Designing freeform reflectors for extended sources,” in Proceedings ofSPIE, vol. 7423 (2009), p. 742302.

14. S. A. Kochengin and V. I. Oliker, “Determination of reflector surfaces from near-field scattering data,” Inverse Probl. 13, 363–373 (1997).

15. J.-D. Benamou, G. Chazareix, W. L. IJzerman, and G. Rukhaia, “Point source regularization of the finite source reflector problem,” J. Comput. Phys. 456, 111032 (2022).

16. P. H. van Cittert, “Zum Einfluß der Spaltbreite auf die Intensitätsverteilung in Spektrallinien. II,” Zeitschrift für Physik 69, 298–308 (1931).

17. H. C. Burger and P. H. van Cittert, “Wahre und scheinbare Intensitätsverteilung in Spektrallinien,” Zeitschrift für Physik 79, 722–730 (1932).

18. W. H. Richardson, “Bayesian-based iterative method of image restoration,” J. Opt. Soc. Am. 62, 55 (1972).

19. L. B. Lucy, “An iterative technique for the rectification of observed distributions,” The Astron. J. 79, 745 (1974).

20. V. C. E. Kronberg, “Inverse freeform reflector design with a scattering surface,” Ph.D. thesis, Eindhoven University of Technology (2024).

21. R. Hacking, L. Kusch, K. Mitra, et al., “Neural-network methods for two-dimensional finite-source reflector design,” arXiv:2604.02184 (2026). Submitted to Machine Learning: Science and Technology.

22. R. Hacking, L. Kusch, K. Mitra, et al., “Hybrid neural and deconvolution approach for finite-source reflector design,” EPJ Web Conf. 335, 02011 (2025).

23. T.-M. Li, M. Aittala, F. Durand, and J. Lehtinen, “Diferentiable Monte Carlo ray tracing through edge sampling,” ACM Trans. on Graph. 37, 1–11 (2018).

24. M. Nimier-David, D. Vicini, T. Zeltner, and W. Jakob, “Mitsuba 2: a retargetable forward and inverse renderer,” ACM Trans. on Graph. 38, 1–17 (2019).

25. B. de Koning, A. Heemels, A. Adam, and M. Möller, “Gradient descent-based freeform optics design for illumination using algorithmic diferentiable non-sequential ray tracing,” Optim. Eng. 25, 1203–1235 (2024).

26. A. Heemels, B. de Koning, M. Möller, and A. Adam, “Optimizing freeform lenses for extended sources with algorithmic diferentiable ray tracing and truncated hierarchical B-splines,” Opt. Express 32, 9730 (2024).

27. W. Jakob, S. Speierer, N. Roussel, et al., “Mitsuba 3 renderer,” https://mitsuba-renderer.org (2022).

28. Y. Sun, B. Deng, and J. Zhang, “End-to-end surface optimization for light control,” ACM Trans. on Graph. 44, 1–19 (2025).

29. L. Wang, J. Chang, Y. Wu, et al., “DifRayFlow: A diferentiable freeform optical design engine based on discret optimal transport,” Photonics 12, 1243 (2025).

30. M. Avi-Aharon, A. Arbelle, and T. Riklin Raviv, “Diferentiable histogram loss functions for intensity-based image-to-image translation,” IEEE Trans. on Pattern Anal. Mach. Intell. 45, 11642–11653 (2023).

31. J. L. Kirkby, Á. Leitao, and D. Nguyen, “Nonparametric density estimation and bandwidth selection with B-spline bases: A novel Galerkin method,” Comput. Stat. & Data Anal. 159, 107202 (2021).

32. Y. Zhao, M. Zhang, Q. Ni, and X. Wang, “Adaptive nonparametric density estimation with B-spline bases,” Mathematics 11, 291 (2023).

33. S. S. Oren and D. G. Luenberger, “Self-scaling variable metric (SSVM) algorithms, Part I: Criteria and suficient conditions for scaling a class of algorithms,” Manag. Sci. 20, 845–862 (1974).

34. M. Al-Baali, “Numerical experience with a class of self-scaling quasi-Newton algorithms,” J. Optim. Theory Appl. 96, 533–553 (1998).

35. J. F. Urbán, P. Stefanou, and J. A. Pons, “Unveiling the optimization process of physics informed neural networks: How accurate and competitive can PINNs be?” J. Comput. Phys. 523, 113656 (2025).

36. D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in International Conference on Learning Representations (ICLR), (2015).

37. A. S. Lewis and M. L. Overton, “Nonsmooth optimization via quasi-Newton methods,” Math. Program. 141, 135–163 (2013).

38. P. Moritz, R. Nishihara, and M. I. Jordan, “A linearly-convergent stochastic L-BFGS algorithm,” in International Conference on Artificial Intelligence and Statistics (AISTATS), (2016), pp. 249–258.