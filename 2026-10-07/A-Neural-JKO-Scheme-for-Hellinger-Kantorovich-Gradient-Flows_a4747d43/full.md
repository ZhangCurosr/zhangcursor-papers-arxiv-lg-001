# A Neural JKO Scheme for Hellinger–Kantorovich Gradient Flows via Monge-Growth Pairs

Geuntaek Seo<sup>1,\*</sup> Cheolhyeong Kim<sup>2,\*</sup> Hwijae Son<sup>3,†</sup> Hyung Ju Hwang<sup>1,†</sup>

<sup>1</sup>Department of Mathematics, Pohang University of Science and Technology

<sup>2</sup>Samsung Electronics

<sup>3</sup>Department of Mathematics, Konkuk University

gtseo@postech.ac.kr hspst440@gmail.com hwijaeson@konkuk.ac.kr hjhwang@postech.ac.kr

These authors contributed equally to this work. <sup>†</sup>Corresponding authors.

## Abstract

We develop a mesh-free neural JKO scheme for advection–reaction–difusion equations with a gradient-flow structure in the Hellinger–Kantorovich (HK) geometry of unbalanced optimal transport. Each update is parametrized by a spatial map and a mass-changing factor, allowing spatial redistribution and local mass creation or loss to be treated jointly within a single variational step. Their cone action bounds the squared HK distance from above, yielding a suficient condition for discrete energy dissipation through comparison with the identity pair. Minimizing the pair objective over all admissible pairs recovers the exact JKO minimum when the source and a minimizer have positive densities. We establish existence and mass bounds for JKO minimizers and, under additional assumptions, obtain positivity and regularity together with a discrete Euler–Lagrange equation and a metric–dissipation identity. The self-consistent chemical potential is then nonincreasing along an optimal map. There exist parametric pairs whose endpoint densities and objective values converge to those of an exact JKO minimizer, provided a regular-pair approximation hypothesis holds. Finally, we show that a primal–dual gap controls objective suboptimality and, for Boltzmann entropy, the L<sup>1</sup> density error, assuming exact-step regularity, positive-semidefinite interactions, and global dual feasibility. Numerical experiments examine pointwise agreement with the PDE, energy dissipation, and the roles of transport, reaction, and fully implicit interactions.

Keywords: Hellinger–Kantorovich distance, unbalanced optimal transport, JKO scheme, advection–reaction–difusion equations, neural networks, minimizing movements 2020 Mathematics Subject Classification: 35K57, 35Q92, 49Q22, 65N75, 68T07

## 1 Introduction

## 1.1 Hellinger–Kantorovich gradient flows and the computational problem

Variational time discretization provides a natural way to construct gradient-flow evolutions. In Wasserstein space, the Jordan–Kinderlehrer–Otto (JKO) scheme identifies the Fokker–Planck equation as a gradient flow of free energy [1, 2]. Classical Wasserstein geometry preserves total mass, whereas advection–reaction–difusion systems generally combine spatial transport with local creation or destruction of mass.

The Hellinger–Kantorovich distance, denoted by HK and also called the Wasserstein–Fisher– Rao distance, is an unbalanced optimal transport metric that couples transport with reaction on finite nonnegative measures [3, 4, 5, 6, 7]. The distance admits static, cone, dynamic, and dual formulations. This geometry has been used to develop minimizing-movement schemes for advection–reaction–difusion equations and to study hard-congestion limits [8, 9, 10]. Subsequent work has established Monge representations, regularity of dual potentials, and local geometric properties [11, 12, 13].

We study the HK gradient flow of a free energy $\mathcal { F }$ consisting of internal, external, and nonlocal interaction terms. For positive constants a and $b ,$ the formal flow takes the form

$$
\partial _ { t } \rho = a \operatorname { d i v } ( \rho \nabla g _ { \rho } ) - b \rho g _ { \rho } , \qquad g _ { \rho } = U ^ { \prime } ( \rho ) + V + W [ \rho ] .\tag{1.1}
$$

Here a and b weight spatial transport and local mass change, respectively. Throughout this paper, we take $a = 1$ and $b = 4$ , corresponding to the normalization $\mathsf { H } = \mathsf { H } _ { 1 , 4 }$ . Both terms are driven by the same chemical potential $g _ { \rho }$ . The associated variational time discretization is the HK-JKO scheme

$$
\mu _ { k + 1 } ^ { \tau } \in \underset { \sigma \in \mathcal { M } _ { + } ( \Omega ) } { \arg \operatorname* { m i n } } \left\{ \mathcal { F } ( \sigma ) + \frac { 1 } { 2 \tau } \mathsf { H K } ^ { 2 } ( \sigma , \mu _ { k } ^ { \tau } ) \right\} .
$$

Here $\mathcal { M } _ { + } ( \Omega )$ denotes the space of finite nonnegative Borel measures on the spatial domain Ω. Whenever a minimizer exists and the current energy is finite, it satisfies the one-step dissipation inequality. Computing a step nevertheless requires minimizing an objective that combines the energy of the next measure with the unbalanced optimal transport cost between the current and next measures. Grid-based discretizations are efective in low dimensions but scale poorly with dimension, motivating mesh-free parametric methods [14, 15].

A parametric method must both represent the next measure and approximate the metric term. Whether the metric approximation is an upper or a lower bound matters for energy descent. Restricting the maximization in the HK dual problem to a prescribed class of feasible potentials gives a lower bound on the squared distance, so descent of the resulting objective need not imply descent of the exact JKO objective. We instead use a feasible primal cone action associated with a spatial map and a growth factor. Under the positive-density hypotheses stated below, optimization over all such pairs is equivalent to the exact JKO problem. Every admissible parametric pair still provides an upper bound on the exact JKO objective at its induced measure.

## 1.2 The primal formulation

Our formulation combines two choices. Each update acts on the current measure, so the learned displacement describes a single time step. A feasible primal cone action then allows the same pair to define the candidate measure and bound its exact proximal cost from above (Table 1).

Restarted update. We call $( T , q )$ a Monge-growth pair : $T$ transports positions and $q ^ { 2 }$ changes the mass before transport, giving

$$
T _ { \# } ( q ^ { 2 } \mu ) .
$$

Here $T _ { \# } \nu$ denotes the pushforward of a measure ν under $T .$ , defined by $( T _ { \# } \nu ) ( A ) = \nu ( T ^ { - 1 } ( A ) )$ for every Borel set $A \subset \Omega$ . The term “growth” includes both mass creation $( q > 1 )$ and mass loss $( q < 1 )$ . We call updates with $T =$ id pure-reaction updates; they are labeled “growth-only” in the experiments. The scheme is restarted in the sense that a fresh pair maps the current measure to the next candidate at each step. Density values and the sampling law must therefore be propagated between steps; Section 4 $\mathrm { g i }$ ves the exact formulas and the density fitting used in recursive runs.

Primal metric representation. A pair $( T , q )$ is admissible for µ if $T : \Omega \to \Omega$ and $q : \Omega \to$ $[ 0 , \infty )$ are Borel and $\textstyle \int _ { \Omega } q ^ { 2 } d \mu < \infty$ . Its induced measure is $T _ { \# } ( q ^ { 2 } \mu )$ , and its Monge-growth

action is

$$
A _ { \mu } ( T , q ) : = \int _ { \Omega } \left[ 1 + q ( x ) ^ { 2 } - 2 q ( x ) \cos \Bigl ( \mathsf { d } _ { \Omega } ( x , T ( x ) ) \wedge { \frac { \pi } { 2 } } \Bigr ) \right] d \mu ( x ) .\tag{1.2}
$$

In the pair formulation, unrestricted optimization ranges over all admissible Monge-growth pairs, whereas restricted optimization ranges only over a prescribed subclass, such as the admissible pairs representable by the chosen neural network class. The cone formulation gives the one-sided bound

$$
{ \sf K } ^ { 2 } \big ( \mu , T _ { \# } ( q ^ { 2 } \mu ) \big ) \leq \mathcal { A } _ { \mu } ( T , q ) ,\tag{1.3}
$$

which is tight at an optimal unrestricted pair when the two measures have positive densities. We therefore optimize

$$
\mathcal { T } _ { \tau } ( T , q \mid \mu ) : = \mathcal { F } \big ( T _ { \# } ( q ^ { 2 } \mu ) \big ) + \frac { 1 } { 2 \tau } \mathcal { A } _ { \mu } ( T , q ) .\tag{1.4}
$$

Every term of this objective can be written as an integral with respect to the current measure;   
see (4.7)–(4.8).

Equivalence to the exact JKO problem. Writing

$$
\mathcal { T } _ { \tau } ( \sigma \mid \mu ) : = \mathcal { F } ( \sigma ) + \frac { 1 } { 2 \tau } { \sf H K } ^ { 2 } ( \sigma , \mu ) ,
$$

the cone-action upper bound gives

$$
\mathcal { T } _ { \tau } ( T _ { \# } ( q ^ { 2 } \mu ) \mid \mu ) \le \mathcal { T } _ { \tau } ( T , q \mid \mu )\tag{1.5}
$$

for every admissible pair. Proposition 3.1 shows that, under the positive-density hypotheses used below, optimization over all admissible pairs has the same optimal value as the exact JKO problem:

$$
\operatorname* { m i n } _ { \sigma \in { \mathcal { M } } _ { + } ( \Omega ) } { \mathcal { I } } _ { \tau } ( \sigma \mid \mu ) = \operatorname* { i n f } _ { T , q \geq 0 } { \mathcal { I } } _ { \tau } ( T , q \mid \mu ) .
$$

A restricted pair class may overestimate the metric term, but the unrestricted formulation is exactly equivalent to the JKO problem. Since the identity pair has zero action, comparison with that pair gives the descent test of Proposition 3.3. Section 3 proves both statements. Theorem 5.4 then combines the primal objective with a globally feasible dual lower bound to estimate the density error. Numerical optimization, quadrature, and density fitting introduce additional errors, discussed in Section 4.

<table><tr><td></td><td>Restricted dual value  $( \leq { \mathsf { H K } } ^ { 2 } )$ </td><td>Primal cone action  $( \geq \mathsf { H K } ^ { 2 } )$ </td></tr><tr><td>Cumulative state  $\widehat { \mu } _ { k } =$   $( S _ { k } ) _ { \# } ( r _ { k } ^ { 2 } \mu _ { 0 } )$ </td><td>Density can be evaluated from a fixed reference law. A dual potential or c-transform is required, and the lower bound alone does not certify exact JKO descent.</td><td>A feasible pair can certify descent, but cumulative displacement and representation complexity may grow.</td></tr><tr><td>Restarted state  $\widehat { \mu } _ { k + 1 } = T _ { \# } ( q ^ { 2 } \widehat { \mu } _ { k } )$ </td><td>Each update starts from the current measure. Dual optimization is still required, and the lower bound alone does not certify exact JKO descent.</td><td>This paper: one pair defines the next measure and an upper bound on its JKO objective. The unrestricted problem is exact; density and sampling information must be propagated between steps.</td></tr></table>

Table 1: Here $( S _ { k } , r _ { k } )$ is the cumulative Monge-growth pair representing the approximate state $\widehat { \mu } _ { k }$ , with $r _ { k } \ge 0$ . Restarting parametrizes each update from the current state; it does not reset accumulated errors. The one-step descent criterion for the pair endpoint relies on the primal upper bound and comparison with the identity pair.

## 1.3 Main results and contributions

We first establish the exact JKO step that the neural update is meant to approximate, then examine the efect of parametric restriction and the accuracy of a computed candidate.

Contributions and analytical ingredients. First, Theorem A establishes existence and mass control, with positivity under an additional entropy condition. Theorem B identifies the structure of this exact step. For fully implicit interactions, it adapts the comparison argument of [13, Sec. 4] by establishing monotonicity of the self-consistent chemical potential $\Phi _ { 1 }$ . This yields oscillation-dependent density bounds. The metric–dissipation identity follows by combining the discrete Euler equation with the HK potential identity of [11], using the manifold Monge formula of [12] for the torus case. Appendix C.1 details the modified comparison argument.

Second, we determine when the pair formulation retains this variational target. The Monge representation of [11, 12] gives exact equivalence before parametric restriction. A neural restriction may introduce both a density error and a cone-action overestimate. Under an additional regular-pair approximation hypothesis, Theorem C transfers pair approximation to convergence of the endpoint density and objective values, and shows that the metric overestimate vanishes.

Third, we assess an individual computed pair. Comparison with the identity gives the energy-descent criterion of Proposition 3.3, but descent alone does not quantify accuracy. Under positive-semidefinite interactions and global dual feasibility, Theorem D combines a primal upper bound with a dual lower bound for the JKO minimum, using HK duality [4, 7] and convexity. Their gap controls objective suboptimality and, for Boltzmann entropy, the $L ^ { 1 }$ density error, without a parametric approximation hypothesis.

Theorem A (Theorem 2.3). Let U be convex, lower semicontinuous, finite at 0, and superlinear; let V be continuous; let W be continuous and symmetric; and assume that the scalar function $\Gamma _ { \tau , \bar { m } }$ in (2.7) is coercive. Then the exact step exists and every minimizer satisfies a quantitative mass bound. If $U ^ { \prime } ( 0 + ) = - \infty$ , its density is strictly positive and has full support.

Theorem B (Theorem 2.6). Under the standing entropy assumptions and two-sided bounds on the source density, the self-consistent chemical potential is nonincreasing along an optimal Monge map. The new density has explicit two-sided bounds, satisfies the exact discrete Euler– Lagrange equation, and obeys $\mathsf { H } ^ { 2 } ( \mu _ { 1 } , \bar { \mu } ) = \tau ^ { 2 } I _ { g } ( \rho _ { 1 } )$ . If, in addition, V and W are Lipschitz, then the new density is Lipschitz. The result covers bounded convex domains and flat tori, including $W = 0$

Theorem C (Theorem 5.2). If a regular optimal Monge-growth pair is approximable in the chosen admissible class, there are parametric pairs whose endpoint densities, JKO objective values, and pair-objective values converge to those of the exact step. Their cone actions converge to the squared HK distance. At fixed τ, this conclusion extends to any fixed number of steps under the corresponding approximation hypothesis at each step; no rate uniform in the number of steps is proved.

Theorem D (Theorem 5.4). Under the exact-step regularity assumptions and positive semidefinite interaction, the gap between a globally feasible dual lower bound and the Monge-growth primal upper bound controls both the JKO objective gap and the internal-energy convexity remainder relative to the unique exact step. For Boltzmann entropy, it controls the squared $L ^ { 1 }$ error by $2 ( \nu ( \Omega ) + \mu _ { 1 } ( \Omega ) ) \mathcal { R }$ , where ν is the computed measure and $\mu _ { 1 }$ the exact step; (2.8) bounds the latter mass. No approximation assumption on the parametric class is used, but global dual feasibility must be verified.

## 1.4 Relation to previous work

Hellinger–Kantorovich geometry and minimizing movements. The HK distance admits complementary dynamic, cone, entropy–transport, and dual formulations [3, 4, 5, 6, 7].

Alternating Wasserstein–Fisher–Rao JKO steps were proved to converge under suitable compactness assumptions in [8]; direct HK minimizing movements for scalar reaction–difusion equations were constructed in [9]; and HK geometry was used in a Hele–Shaw-type tumor-growth limit in [10]. More recent work develops fine geodesic and Monge properties, regularity of unbalanced transport, and an EVI theory for geodesically semiconvex energies [11, 12, 13]. Our exact-step analysis concerns a fully implicit proximal problem in which the interaction field depends on the candidate measure. As explained above, the comparison scheme is rooted in [13, Sec. 4], but the self-consistent chemical potential replaces the density as the monotone quantity and necessitates the modified oscillation-sensitive argument of Appendix C.1.

Numerical optimal transport and neural JKO solvers. Entropic scaling made discrete unbalanced transport, barycenters, and unbalanced gradient-flow steps accessible through Sinkhorn-type iterations [16], while Eulerian Benamou–Brenier discretizations with operator splitting provide structure-preserving Wasserstein JKO solvers in low dimensions [17]. Minimax estimation of transport-growth pairs in Gaussian–Hellinger unbalanced transport is studied in [18]. Neural methods instead seek mesh-free parametrizations of Wasserstein or related minimizing movements: convex transport potentials are used in [14], variational representations of density terms in [19], direct state parametrization in [15], blockwise invertible flows in [20], and Lagrangian neural ODEs with density updates in [21]. These are primarily balanced or masspreserving precedents. Neural unbalanced transport methods combine learned maps with mass reweighting between prescribed distributions [22, 23, 24]. TIGON learns velocity and growth fields from temporal snapshots using a dynamic WFR formulation [25]. For a prescribed free energy, we use a primal cone action for fully implicit HK-JKO steps, obtaining energy-descent criteria and, together with feasible dual lower bounds, a posteriori error estimates.

Fixed-reference, dual, and resampling-based formulations. A semi-dual approach to Wasserstein JKO steps uses a fixed reference measure and an unbalanced-transport reformulation [26]. A broader dual Lagrangian framework for balanced and unbalanced path problems appears in the literature [27]. Local continuous normalizing flows have also been combined with importance correction and resampling [28]. In our restarted formulation, each pair acts on the current measure and represents only one time step. The trade-of is the loss of direct density evaluation from a fixed base law. We address this with change-of-variables formulas for density propagation and growth-weighted particle updates.

## 1.5 Scope and organization

We distinguish the formal flow (1.1), the exact iterates $\mu _ { k + 1 } ^ { \tau } = S _ { \tau } ( \mu _ { k } ^ { \tau } )$ , and the idealized parametric iterates with exact density propagation

$$
\widehat { \mu } _ { k + 1 } ^ { \tau } = ( T _ { \theta , k + 1 } ) _ { \# } \big ( q _ { \theta , k + 1 } ^ { 2 } \widehat { \mu } _ { k } ^ { \tau } \big ) = u _ { \theta , k + 1 } ^ { \tau } d x .\tag{1.6}
$$

Recursive numerical runs additionally apply a mass-corrected density fit after each pair update (Algorithm 1).

The analysis is time-discrete and covers the exact HK-JKO step, its Monge-growth pair formulation, and guarantees for admissible computed pairs. We do not prove convergence of the stochastic optimizer or a τ ↓ 0 limit for the trained scheme. Remarks 5.3 and 5.7 detail the assumptions and limitations of the approximation and a posteriori results.

Section 2 develops the exact-step theory, and Section 3 presents the Monge-growth formulation. Section 4 describes the neural algorithm, and Section 5 gives the approximation and a posteriori results. Section 6 presents numerical experiments.<sup>1</sup> Proofs are collected in Appendices A–E, and implementation details in Appendix F.

## 2 Setting and the exact HK-JKO step

## 2.1 The minimal setting

Throughout, Ω denotes either a compact convex subset of $\mathbb { R } ^ { d }$ with nonempty interior or the flat torus $\mathbb { T } ^ { d } .$ In the Euclidean case, we write $D = \operatorname { i n t } \Omega$ when the open domain must be distinguished from its closure. We denote by $\mathcal { M } _ { + } ( \Omega )$ the space of finite nonnegative Borel measures on $\Omega .$ The symbol dx denotes Lebesgue measure in the Euclidean case and the Riemannian volume measure on $\mathbb { T } ^ { d } ; \ { \mathsf { d } } _ { \Omega }$ denotes the corresponding Euclidean or flat-geodesic distance. For a real-valued function $f ,$ we write $f _ { - } : = \operatorname* { m a x } \{ - f , 0 \}$

We reserve $W ( x , y )$ for the interaction kernel. For $\mu \in \mathcal { M } _ { + } ( \Omega )$ , define the associated interaction potential

$$
W [ \mu ] ( x ) : = \int _ { \Omega } W ( x , y ) d \mu ( y ) ,
$$

and, for a density $\rho ,$ write $W [ \rho ] : = W [ \rho d x ]$ . The free energy is

$$
\mathcal F ( \mu ) = \left\{ \begin{array} { l l } { \displaystyle \int _ { \Omega } U ( \rho ) d x + \int _ { \Omega } V \rho d x + \frac { 1 } { 2 } \int \int _ { \Omega \times \Omega } W ( x , y ) \rho ( x ) \rho ( y ) d x d y , } & { \mu = \rho d x , } \\ { + \infty , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

For later use, denote the scalar interaction-energy functional separately by

$$
\mathcal { W } ( \mu ) : = \frac { 1 } { 2 } \int \int _ { \Omega \times \Omega } W ( x , y ) d \mu ( x ) d \mu ( y ) .
$$

Thus $W$ denotes the two-point kernel, $W [ \mu ]$ its induced one-point potential, and $\mathcal { W } ( \mu )$ the scalar interaction energy.

Assumption 2.1 (Interaction kernel). We use the following regularity levels:

$$
\begin{array} { r l } { ( \mathbf { W 0 } ) } & { \boldsymbol { W } \in C _ { \mathrm { s y m } } ( \Omega \times \Omega ) , \qquad ( \mathbf { W 1 } ) \quad \boldsymbol { W } \in W _ { \mathrm { s y m } } ^ { 1 , \infty } ( \Omega \times \Omega ) , } \end{array}
$$

where $C _ { \mathrm { s y m } } ( \Omega \times \Omega ) : = \{ F \in C ( \Omega \times \Omega ) : F ( x , y ) = F ( y , x ) \}$ , and similarly for $W _ { \mathrm { s y m } } ^ { 1 , \infty } ( \Omega \times \Omega )$

The internal energy is subject to the following standing assumption.

Assumption 2.2 (Standing assumptions on the internal energy). Whenever the pointwise regularity theorem is invoked,

$$
U \in C ( [ 0 , \infty ) ) \cap C ^ { 2 } ( ( 0 , \infty ) ) , \qquad U ^ { \prime \prime } > 0 , \qquad U ^ { \prime } ( 0 + ) = - \infty , \qquad U ^ { \prime } ( + \infty ) = + \infty .
$$

Thus $U ^ { \prime } : ( 0 , \infty ) \to \mathbb { R }$ is a strictly increasing bijection. Assumptions on $V , W$ and coercivity of the full JKO objective are stated in each theorem.

The principal example is the Boltzmann entropy $U ( r ) = r \log r - r ;$ a negative contribution from the quadratic interaction energy requires the separate coercivity hypothesis in Theorem 2.3.

For a positive density $\rho ,$ set

$$
\begin{array} { r } { P [ \rho d x ] = V + W [ \rho ] , \qquad g _ { \rho } = U ^ { \prime } ( \rho ) + P [ \rho d x ] . } \end{array}
$$

On a bounded domain the formal no-flux condition, with n the outward unit normal, is

$$
\rho \nabla g _ { \rho } \cdot \mathsf { n } = 0 \qquad \mathcal { H } ^ { d - 1 } { \mathrm { - a . e . ~ o n ~ } } \partial \Omega .
$$

## 2.2 HK formulations and normalization

The Hellinger–Kantorovich metric. Throughout we use the normalization $\mathsf { H } = \mathsf { H } _ { 1 , 4 }$ characterized by the dynamic formulation [4, 6]

$$
\mathsf { H } ^ { 2 } ( \mu _ { 0 } , \mu _ { 1 } ) = \operatorname* { i n f } \int _ { 0 } ^ { 1 } \int _ { \Omega } ( | v _ { t } | ^ { 2 } + 4 | \alpha _ { t } | ^ { 2 } ) d \mu _ { t } d t ,
$$

where the infimum is taken over narrowly continuous curves $( \mu _ { t } ) _ { t \in [ 0 , 1 ] }$ in $\mathcal { M } _ { + } ( \Omega )$ and Borel velocity and reaction fields $v _ { t }$ and $\alpha _ { t }$ with finite action, satisfying

$$
\partial _ { t } \mu _ { t } + \operatorname { d i v } ( \mu _ { t } v _ { t } ) = 4 \alpha _ { t } \mu _ { t } , \qquad \mu | _ { t = 0 } = \mu _ { 0 } , \quad \mu | _ { t = 1 } = \mu _ { 1 } .
$$

The continuity equation is understood weakly, with no boundary flux in the Euclidean case and periodicity on the torus. This dynamic formulation corresponds to the choice $a = 1 , b = 4$ in (1.1). The equivalent entropy–transport, dual, and one-potential formulations, together with the elementary bounds used in the proofs, are recorded below.

Set $F ( s ) = s \log s - s + 1 , F ^ { * } ( a ) = e ^ { a } - 1$ , and

$$
\begin{array} { r } { c _ { \mathsf { H K } } ( x , y ) = - \log \left( \cos ^ { 2 } ( \mathsf { d } _ { \Omega } ( x , y ) \wedge \frac { \pi } { 2 } ) \right) . } \end{array}
$$

The entropy–transport formulation is [4, 7]

$$
\mathsf { H } ^ { 2 } ( \mu _ { 0 } , \mu _ { 1 } ) = \operatorname* { i n f } _ { \gamma } \left\{ \int F \left( \frac { d \gamma _ { 0 } } { d \mu _ { 0 } } \right) d \mu _ { 0 } + \int F \left( \frac { d \gamma _ { 1 } } { d \mu _ { 1 } } \right) d \mu _ { 1 } + \int c _ { \mathsf { H } } d \gamma \right\} ,\tag{2.1}
$$

where $\gamma _ { i } \ll \mu _ { i }$ . Its dual is

$$
{ \sf H } ^ { 2 } ( \mu , \nu ) = \operatorname* { s u p } _ { z _ { 0 } + z _ { 1 } \leq c _ { \sf K } } \left[ - \int F ^ { * } ( - z _ { 0 } ) d \mu - \int F ^ { * } ( - z _ { 1 } ) d \nu \right] ,\tag{2.2}
$$

and the one-potential form is

$$
\mathsf { H } ^ { 2 } ( \mu , \nu ) = \operatorname* { s u p } _ { v } \left[ - \int F ^ { * } ( - v ^ { c } ) d \mu - \int F ^ { * } ( - v ) d \nu \right] , \qquad v ^ { c } ( x ) = \operatorname* { i n f } _ { y } \{ c _ { \mathsf { H K } } ( x , y ) - v ( y ) \} .\tag{2.3}
$$

The formulation actually used in this paper is the equivalent cone, or semi-coupling, form $[ 7 , 4 ] ;$ with $\mathfrak { c } ( x , y , a , b ) : = a ^ { 2 } + b ^ { 2 } - 2 a b \cos ( \mathtt { d } _ { \Omega } ( x , y ) \wedge \frac { \pi } { 2 } )$

$$
\mathsf { H } ^ { 2 } ( \mu _ { 0 } , \mu _ { 1 } ) = \operatorname* { m i n } \left\{ \int _ { \Omega \times \Omega } \mathsf { c } \left( x , y , r _ { 0 } ( x , y ) , r _ { 1 } ( x , y ) \right) d \gamma \ : \ \pi _ { \# } ^ { 0 } ( r _ { 0 } ^ { 2 } \gamma ) = \mu _ { 0 } , \quad \pi _ { \# } ^ { 1 } ( r _ { 1 } ^ { 2 } \gamma ) = \mu _ { 1 } \right\} ,\tag{2.4}
$$

the minimum being taken over $\gamma \in \mathcal { M } _ { + } ( \Omega \times \Omega )$ and Borel $r _ { 0 } , r _ { 1 } \ge 0$ with $r _ { i } \in L ^ { 2 } ( \gamma )$ . Only one consequence of (2.4) is used below: every admissible triple $( \gamma , r _ { 0 } , r _ { 1 } )$ yields an upper bound for $\mathsf { H } ^ { 2 }$ . In particular, $\gamma = ( \mathrm { i d } , T ) _ { \# } \mu$ with $( r _ { 0 } , r _ { 1 } ) = ( 1 , q )$ gives $( 1 . 3 ) , \ \gamma = ( T , T ^ { \prime } ) _ { \# } \mu$ with $( r _ { 0 } , r _ { 1 } ) = ( q , q ^ { \prime } )$ gives the common-base bound used in Lemma 3.2, and the diagonal choice $\gamma = ( \mathrm { i d } , \mathrm { i d } ) _ { \# } ( \rho _ { 0 } d x )$ with $( r _ { 0 } , r _ { 1 } ) = ( 1 , \sqrt { \rho _ { 1 } / \rho _ { 0 } } )$ gives the pure-reaction competitor. The latter reads

$$
\mathsf { H } ^ { 2 } ( \rho _ { 0 } d x , \rho _ { 1 } d x ) \leq \int ( \sqrt { \rho _ { 0 } } - \sqrt { \rho _ { 1 } } ) ^ { 2 } d x \leq \| \rho _ { 0 } - \rho _ { 1 } \| _ { L ^ { 1 } } ,\tag{2.5}
$$

and

$$
{ \sf H } ( \mu , \nu ) \geq | \sqrt { \mu ( \Omega ) } - \sqrt { \nu ( \Omega ) } | , \qquad { \sf H } ^ { 2 } ( \mu , 0 ) = \mu ( \Omega ) .\tag{2.6}
$$

For $\tau > 0$ , define

$$
\mathcal { T } _ { \tau } ( \sigma \mid \mu ) : = \mathcal { F } ( \sigma ) + \frac { 1 } { 2 \tau } { \sf H K } ^ { 2 } ( \sigma , \mu ) ,
$$

and, whenever a minimizer exists, choose

$$
S _ { \tau } ( \mu ) \in \underset { \sigma \in \mathcal { M } _ { + } ( \Omega ) } { \mathrm { a r g m i n } } ~ \mathcal { T } _ { \tau } ( \sigma \mid \mu ) .
$$

The exact recursion generated by this operator is

$$
\mu _ { k + 1 } ^ { \tau } = S _ { \tau } ( \mu _ { k } ^ { \tau } ) , \qquad k = 0 , 1 , \ldots ,
$$

and is kept notationally distinct from the neural recursion (1.6) throughout. In both recursions the step is fully implicit in the interaction: a candidate r dx enters through

$$
g _ { r } = U ^ { \prime } ( r ) + V + W [ r ] ,
$$

and not through a field frozen at the preceding state.

## 2.3 Theorem A: existence, mass control, positivity

For $\bar { m } = \bar { \mu } ( \Omega )$ ), V := min<sub>Ω</sub> V , and $\begin{array} { r } { \underline { { W } } : = \operatorname* { m i n } _ { ( x , y ) \in \Omega \times \Omega } W ( x , y ) } \end{array}$ , we define

$$
\Gamma _ { \tau , \bar { m } } ( m ) = | \Omega | U \left( \frac { m } { | \Omega | } \right) + { \underline { { V } } } m + \frac { { \underline { { W } } } } { 2 } m ^ { 2 } + \frac { 1 } { 2 \tau } ( \sqrt { m } - \sqrt { \bar { m } } ) ^ { 2 } .\tag{2.7}
$$

Theorem 2.3 (Existence, mass control, and positivity). Let Ω be either a compact convex subset $o f \mathbb { R } ^ { d }$ with nonempty interior or the flat torus $\mathbb { T } ^ { d }$ . Assume that $U : [ 0 , \infty ) \to \mathbb { R }$ is convex, lower semicontinuous, finite at 0, and superlinear. Let $V \in C ( \Omega )$ and suppose that (W0) holds. Assume also that $\Gamma _ { \tau , \bar { m } } ( m )  + \infty \ a s \ m  \infty$ . Then $\mathcal { I } _ { \tau } ( \cdot \ | \ \bar { \mu } )$ has a minimizer $\mu _ { 1 } = \rho _ { 1 } d x$ . Every minimizer satisfies

$$
\mu _ { 1 } ( \Omega ) \leq M _ { \tau , \bar { \mu } } : = \operatorname * { s u p } \left\{ m \geq 0 : \Gamma _ { \tau , \bar { m } } ( m ) \leq | \Omega | U ( 0 ) + \frac { \bar { m } } { 2 \tau } \right\} < \infty .\tag{2.8}
$$

If $U ^ { \prime } ( 0 + ) = - \infty$ , then $\rho _ { 1 } > 0$ almost everywhere and supp $\mu _ { 1 } = \Omega$ . Here $U ^ { \prime } ( 0 + )$ denotes the right derivative of the convex function U at 0, which exists in $[ - \infty , + \infty )$ ; no diferentiability of $U$ is assumed in this theorem.

Idea of the proof (Appendix A). The proof uses the direct method, with a scalar coercivity estimate to control the total mass. Testing against $\sigma = 0$ bounds the infimum by $| \Omega | U ( 0 ) +$ $\bar { m } / ( 2 \tau )$ , because $\mathsf { H } \mathsf { K } ^ { 2 } ( 0 , \bar { \mu } ) = \bar { \mu } ( \Omega )$ ; this is where the unbalanced geometry helps, since the zero measure is an admissible competitor at finite cost. Jensen’s inequality, the elementary bounds $V \geq \underline { { V } }$ and $W \geq \underline { { W } }$ , and the mass estimate (2.6) then bound ${ \mathcal { I } } _ { \tau } ( \sigma \mid { \bar { \mu } } )$ from below by the scalar function $\Gamma _ { \tau , \bar { m } }$ of the total mass alone. This reduction is the only place where the negative contribution of the quadratic interaction energy must be controlled, and it is why the coercivity hypothesis is stated as a condition on one scalar function rather than as a sign condition on $W$

Superlinearity of U upgrades the resulting mass bound to uniform integrability through the de la Vall´ee-Poussin criterion, so no singular part can appear in the limit, while narrow convergence of $\sigma _ { n } \otimes \sigma _ { n }$ on the compact product space handles the interaction term.

Strict positivity is obtained by adding ε1<sub>B</sub> dx on a putative vacuum set $B ;$ the extra proximal cost is at most $\varepsilon | B | / ( 2 \tau )$ because $F ( 0 ) = 1$ , whereas the internal-energy increment is $| B | \big ( U ( \varepsilon ) - U ( 0 ) \big )$ , which dominates every $O ( \varepsilon )$ term when $U ^ { \prime } ( 0 + ) = - \infty$

Corollary 2.4 (Explicit mass bound). If the interaction energy is nonnegative on nonnegative measures and $U ( r ) \geq a r - C _ { a }$ with $a > \| V _ { - } \| _ { \infty }$ , then

$$
\mu _ { 1 } ( \Omega ) \leq \frac { C _ { a } | \Omega | + | \Omega | U ( 0 ) + \bar { m } / ( 2 \tau ) } { a - \| V _ { - } \| _ { \infty } } .\tag{2.9}
$$

## 2.4 Monge representation on domains and tori

The next proposition supplies the optimal Monge-growth pairs used in both Theorem B and the exact pair reformulation in Section 3.1. Its proof depends on whether the domain is Euclidean or a flat torus. The subsequent arguments use the representation (2.10), positivity of $q ,$ and essential injectivity of $T .$

Proposition 2.5 (Monge representation and essential injectivity). Let Ω be either a compact convex subset of $\mathbb { R } ^ { d }$ with nonempty interior or the flat torus $\mathbb { T } ^ { d }$ , and let $\mu _ { i } = \rho _ { i } d x$ with $\rho _ { i } > 0$ almost everywhere. There is an optimal Monge-growth pair $( T , q )$ from µ<sub>0</sub> to $\mu _ { 1 }$ satisfying

$$
T _ { \# } ( q ^ { 2 } \mu _ { 0 } ) = \mu _ { 1 } , \qquad \# ^ { 2 } ( \mu _ { 0 } , \mu _ { 1 } ) = \int _ { \Omega } ( 1 + q ^ { 2 } - 2 q \cos ( \mathsf { d } _ { \Omega } ( x , T ( x ) ) \wedge \frac \pi 2 ) ) d \mu _ { 0 } .\tag{2.10}
$$

Moreover, $q > 0 ~ \mu _ { 0 }$ -almost everywhere and $T$ may be chosen essentially injective. The same statement holds with $\mu _ { 0 }$ and $\mu _ { 1 }$ interchanged.

Idea of the proof (Appendix B). The Euclidean case is Theorem 3.3(4)–(5) and Corollary 3.5 of [11]. Since both densities are positive, their supports both equal Ω, so each support lies in the $\pi / 2$ -neighborhood of the other; this is precisely the strong-reduction condition of [11, Definition 2.6]. The torus case follows by combining the Lipschitz-potential and Monge results of [12], with essential injectivity obtained from uniqueness of the entropy–transport optimizer for the reversed pair. Appendix B also records the explicit form of $( T , q )$ in terms of the logarithmic dual potential.

Orientation convention. The forward pair defines the numerical update, $T _ { \# } ( q ^ { 2 } \mu _ { k } ) = \mu _ { k + 1 }$ with cone action integrated against $\mu _ { k }$ . The reversed pair maps $\mu _ { k + 1 }$ back to $\mu _ { k } { \mathrm { : } }$ its sourceside dual potential appears in Theorem $2 . 6 ( \mathrm { i i i } )$ , whose Euler equation holds at the new density. Proposition 2.5 provides both orientations.

Accordingly, (2.14) integrates against $\rho _ { k + 1 }$ , while $\mathcal { A } _ { \mu _ { k } }$ integrates against $\rho _ { k }$ . Their leadingorder agreement in τ supplies the diagnostic reference values. The forward and reversed generating potentials also have opposite leading-order signs; see Remark F.1.

## 2.5 Theorem B: density bounds, the discrete Euler equation, and the metric– dissipation identity

Let us define

$$
I _ { g } ( \rho ) = \int _ { \Omega } ( | \nabla g _ { \rho } | ^ { 2 } + 4 | g _ { \rho } | ^ { 2 } ) \rho d x .
$$

Theorem 2.6 (Density bounds and the exact discrete Euler equation). Under the assumptions of Theorem 2.3, we further assume Assumption 2.2. Let $\bar { \mu } = \bar { \rho }$ dx with $\bar { \rho } \in L ^ { \infty } ( \Omega )$ and $0 ~ <$ $c _ { 0 } \leq \bar { \rho } \leq C _ { 0 }$ almost everywhere. For a minimizer $\mu _ { 1 } = \rho _ { 1 } d x$ , set $M _ { 1 } = M _ { \tau , \bar { \mu } }$ and

$$
B _ { 1 } = \| V \| _ { \infty } + M _ { 1 } \| W \| _ { \infty } , \qquad \omega _ { 1 } = \mathrm { e s s o s c } V + 2 M _ { 1 } \| W \| _ { \infty } , \qquad c _ { * } = ( U ^ { \prime } ) ^ { - 1 } ( - B _ { 1 } - 1 ) ,
$$

where ess osc<sub>Ω</sub> $f : = \mathrm { e s s } \mathrm { s u p } _ { \Omega } f - \mathrm { e s s } \operatorname* { i n f } _ { \Omega } f$ denotes the essential oscillation. Then:

(i) there exists an essentially injective optimal pair $( T , q )$ from µ¯ to $\mu _ { 1 }$ such that

$$
\Phi _ { 1 } ( T x ) \le \Phi _ { 1 } ( x ) , \qquad \Phi _ { 1 } : = U ^ { \prime } ( \rho _ { 1 } ) + V + W [ \mu _ { 1 } ] ,\tag{2.11}
$$

for µ¯-almost every x;

(ii) the density satisfies

$$
\begin{array} { r l } & { 0 < \underline { { \rho } } _ { 1 } : = ( U ^ { \prime } ) ^ { - 1 } ( U ^ { \prime } ( \operatorname* { m i n } \{ c _ { 0 } , c _ { * } \} ) - \omega _ { 1 } ) \leq \rho _ { 1 } , } \\ & { \qquad \rho _ { 1 } \leq ( U ^ { \prime } ) ^ { - 1 } ( \operatorname* { m a x } \{ B _ { 1 } , U ^ { \prime } ( C _ { 0 } ) \} + \omega _ { 1 } ) = : \overline { { \rho } } _ { 1 } < \infty ; } \end{array}\tag{2.12}
$$

(iii) for the reversed pair $( \mu _ { 1 } , \bar { \mu } )$ , there is a potential $\phi _ { 1 } ~ \in ~ W ^ { 1 , \infty } ( \Omega )$ such that $2 \phi _ { 1 }$ is the coeficient of $\mu _ { 1 }$ in the transformed dual (C.12). With this normalization,

$$
g _ { \rho _ { 1 } } : = U ^ { \prime } ( \rho _ { 1 } ) + V + W [ \rho _ { 1 } ] = - \frac { \phi _ { 1 } } { \tau } \qquad a . e . ,\tag{2.13}
$$

and

$$
\mathsf { H } ^ { 2 } ( \mu _ { 1 } , \bar { \mu } ) = \tau ^ { 2 } I _ { g } ( \rho _ { 1 } ) , \qquad \mathcal { F } ( \mu _ { 1 } ) + \frac { \tau } { 2 } I _ { g } ( \rho _ { 1 } ) \le \mathcal { F } ( \bar { \mu } ) .\tag{2.14}
$$

Here $g _ { \rho _ { 1 } } = - \phi _ { 1 } / \tau$ is Lipschitz by part (iii), and $I _ { g } ( \rho _ { 1 } )$ uses its weak gradient. If, in addition, $V \in W ^ { 1 , \infty } ( \Omega )$ and (W1) holds, then $\rho _ { 1 } \in W ^ { 1 , \infty } ( \Omega )$

Remark 2.7. The local case follows by setting $W = 0 , B _ { 1 } = \| V \| _ { \infty } , a n d \omega _ { 1 } = \mathrm { e s s o s c } _ { \Omega } V .$

Idea of the proof (Appendix C). Two independent mechanisms are combined.

The first is a comparison principle for the step. On a set A, replacing the optimal Mongegrowth pair by the identity map with the constant growth factor $\sqrt { \mu _ { 1 } ( T ( A ) ) / \bar { \mu } ( A ) }$ does not increase the HK cost, by Cauchy–Schwarz (Lemma C.3). Consequently an ordering violation of the self-consistent chemical potential $\Phi _ { 1 }$ along $T$ would allow one to move mass from a region of high $\Phi _ { 1 }$ to a region of low $\Phi _ { 1 }$ without increasing the metric cost and with strictly lower energy, contradicting minimality. This gives $\Phi _ { 1 } \circ T \leq \Phi _ { 1 }$ (Lemma C.4). Using this monotonicity and the restriction identity for an essentially injective Monge pair (Lemma C.1), we compare suitable local mass variations on regions of high or low chemical potential. Comparing their efects on the energy and the metric cost yields the two-sided density bounds (2.12).

The second is a dual computation, carried out for the reversed pair from $\mu _ { 1 }$ to $\bar { \mu } ,$ since the Euler–Lagrange equation is an identity at the new density. For the source component ψ supplied by Lemma C.7, that pair satisfies $1 - \psi = q$ cos r and $\nabla \psi = - 2 q \sin r e$ , where $\boldsymbol { r } = \mathsf { d } _ { \Omega } ( x , T x )$ and e is the initial unit tangent of the selected geodesic. With $\phi = \psi / 2$ this yields the pointwise identity

$$
| \nabla \phi | ^ { 2 } + 4 | \phi | ^ { 2 } = 1 + q ^ { 2 } - 2 q \cos r ,
$$

whose integral is exactly $\mathsf { H K } ^ { 2 }$ (Corollary C.9). Taking the first variation of the JKO objective with respect to the endpoint density and using uniqueness of that source component identifies $\phi _ { 1 } = - \tau g _ { \rho _ { 1 } }$ , which is the Euler–Lagrange equation (2.13); substituting it back into the identity gives ${ \sf H } ^ { 2 } ( \mu _ { 1 } , \bar { \mu } ) = \tau ^ { 2 } I _ { g } ( \rho _ { 1 } )$ . If V and W are additionally Lipschitz, then $U ^ { \prime } ( \rho _ { 1 } ) = - \phi _ { 1 } / \tau - V -$ $W [ \rho _ { 1 } ]$ is Lipschitz. Since $( U ^ { \prime } ) ^ { - 1 }$ is Lipschitz on the interval supplied by part (ii), the density is then Lipschitz as well.

The comparison argument uses the essentially injective optimal pair from Proposition 2.5, the only part of the proof that depends on the state space.

## 3 The Monge-growth formulation

## 3.1 Exact reparametrization and pair stability

The first point is variational: before any finite-dimensional restriction, the restarted pair variables reproduce the exact JKO problem.

Proposition 3.1 (One-step HK-JKO reparametrization). Let $\mu$ have a positive density and assume that $\mathcal { I } _ { \tau } ( \cdot \mid \mu )$ admits a positive-density minimizer. Then

$$
\operatorname* { m i n } _ { \sigma \in { \mathcal { M } } _ { + } ( \Omega ) } { \mathcal { I } } _ { \tau } ( \sigma \mid \mu ) = \operatorname* { i n f } _ { T , q \geq 0 } { \mathcal { I } } _ { \tau } ( T , q \mid \mu ) .\tag{3.1}
$$

$I f \ \mu _ { + }$ is a positive-density minimizer and $( T _ { * } , q _ { * } )$ is an optimal Monge-growth pair from $\mu$ to $\mu _ { + }$ , then

$$
( T _ { * } ) _ { \# } ( q _ { * } ^ { 2 } \mu ) = \mu _ { + } , \qquad A _ { \mu } ( T _ { * } , q _ { * } ) = { \sf H } ^ { 2 } ( \mu , \mu _ { + } ) ,\tag{3.2}
$$

and $( T _ { * } , q _ { * } )$ attains the right-hand side of (3.1).

Proof. For every pair $( T , q )$ , (1.3) gives

$$
\mathcal { T } _ { \tau } \bigl ( T _ { \# } ( q ^ { 2 } \mu ) \mid \mu \bigr ) \leq \mathcal { T } _ { \tau } ( T , q \mid \mu ) .
$$

Since the left-hand side is bounded below by the minimum of the JKO problem, the infimum over $( T , q )$ is no smaller than that minimum. Conversely, let $\mu _ { + }$ be a positive-density minimizer. Proposition 2.5 supplies an optimal pair $( T _ { * } , q _ { * } )$ satisfying (3.2). Substitution gives

$$
{ \mathcal { T } } _ { \tau } ( T _ { * } , q _ { * } \mid \mu ) = { \mathcal { I } } _ { \tau } ( \mu _ { + } \mid \mu ) ,
$$

which proves the reverse inequality and attainment.

Applying Proposition 2.5 successively, every finite exact HK-JKO sequence with positive densities admits an essentially injective restarted Monge-growth representation whose cone action equals the squared HK distance at each step.

For the approximation arguments below, the same cone representation also gives direct stability of the induced endpoint with respect to the pair. We will often work with a current measure

$$
\mu = \rho d x , \qquad \rho \in C ( \Omega ) , \qquad 0 < \underline { { { \rho } } } \leq \rho \leq \overline { { { \rho } } } .
$$

Lemma 3.2 (Stability of the pushforward in the pair). Let $\mu \in \mathcal { M } _ { + } ( \Omega )$ and let $( T , q ) , ( T ^ { \prime } , q ^ { \prime } )$ be admissible pairs. Then

$$
\mathsf { H } ^ { 2 } \big ( T _ { \# } ( q ^ { 2 } \mu ) , T _ { \# } ^ { \prime } ( q ^ { \prime 2 } \mu ) \big ) \leq \int _ { \Omega } \Big [ ( q - q ^ { \prime } ) ^ { 2 } + q q ^ { \prime } \mathsf { d } _ { \Omega } ( T , T ^ { \prime } ) ^ { 2 } \Big ] d \mu .\tag{3.3}
$$

In particular, if max $\left\{ q , q ^ { \prime } \right\} \leq A$ on Ω, then

$$
\begin{array} { r } { \mathsf { H } \big ( T _ { \# } ( q ^ { 2 } \mu ) , T _ { \# } ^ { \prime } ( q ^ { \prime 2 } \mu ) \big ) \le \mu ( \Omega ) ^ { 1 / 2 } \Big ( \| q - q ^ { \prime } \| _ { L ^ { \infty } } + A \| \mathsf { d } _ { \Omega } ( T , T ^ { \prime } ) \| _ { L ^ { \infty } } \Big ) . } \end{array}\tag{3.4}
$$

Proof. Set $S = ( T , T ^ { \prime } )$ and $\lambda = S _ { \# } \mu$ , and define

$$
a = \frac { d S _ { \# } ( q ^ { 2 } \mu ) } { d \lambda } , \qquad b = \frac { d S _ { \# } ( q ^ { \prime 2 } \mu ) } { d \lambda } , \qquad c = \frac { d S _ { \# } ( q q ^ { \prime } \mu ) } { d \lambda } .
$$

Conditional Cauchy–Schwarz gives $c \leq \sqrt { a b } \lambda \mathrm { - a . e }$ . Thus λ with amplitudes $( { \sqrt { a } } , { \sqrt { b } } )$ is admissible in (2.4), even when $S$ is not injective. Since the truncated cosine is nonnegative, its cost is at most the following common-base integral:

$$
{ \sf H } ^ { 2 } \big ( T _ { \# } ( q ^ { 2 } \mu ) , T _ { \# } ^ { \prime } ( q ^ { \prime 2 } \mu ) \big ) \leq \int _ { \Omega } \Big [ q ^ { 2 } + q ^ { \prime 2 } - 2 q q ^ { \prime } \cos \big ( { \sf d } _ { \Omega } ( T , T ^ { \prime } ) \wedge \frac \pi 2 \big ) \Big ] d \mu ,
$$

of which (1.3) is the case $( T ^ { \prime } , q ^ { \prime } ) = ( \mathrm { i d } , 1 )$ . Writing $s : = \mathsf { d } _ { \Omega } ( T , T ^ { \prime } ) \wedge \frac { \pi } { 2 }$ and $q ^ { 2 } + q ^ { \prime 2 } - 2 q q ^ { \prime } \cos s =$ $( q - q ^ { \prime } ) ^ { 2 } + 2 q q ^ { \prime } ( 1 - \cos s )$ , the bound 1 − cos $s \leq s ^ { 2 } / 2 \leq \mathsf { d } _ { \Omega } ( T , T ^ { \prime } ) ^ { 2 } / \bar { 2 }$ gives (3.3), and $\sqrt { x + y } \leq$ ${ \sqrt { x } } + { \sqrt { y } }$ gives (3.4). □

Together with Proposition 3.1, estimate (3.4) converts pair approximation into a one-step state error whenever max $\{ q _ { \theta } , q _ { * } \} \le A$

$$
\begin{array} { r } { \mathsf { H } \big ( \widehat { \mu } _ { k + 1 } , S _ { \tau } \widehat { \mu } _ { k } \big ) \leq \widehat { \mu } _ { k } ( \Omega ) ^ { 1 / 2 } \Big ( \| q _ { \theta } - q _ { * } \| _ { L ^ { \infty } } + A \| \mathsf { d } _ { \Omega } ( T _ { \theta } , T _ { * } ) \| _ { L ^ { \infty } } \Big ) . } \end{array}\tag{3.5}
$$

## 3.2 Primal action versus restricted duality

Even after parametric restriction, the cone action gives an upper bound on the squared HK distance, while a feasible dual potential gives a lower bound. The upper bound yields the following suficient condition for energy descent.

Writing $r ( x ) : = \mathsf { d } _ { \Omega } ( x , T ( x ) ) \wedge \frac { \pi } { 2 }$ , the cone integrand splits as

$$
1 + q ^ { 2 } - 2 q \cos r = ( 1 - q ) ^ { 2 } + 2 q ( 1 - \cos r ) .\tag{3.6}
$$

Accordingly, $\mathcal { A } _ { \mu } = \mathcal { A } _ { \mu } ^ { \mathrm { r e } } + \mathcal { A } _ { \mu } ^ { \mathrm { t r } }$ , where

$$
A _ { \mu } ^ { \mathrm { r e } } ( T , q ) : = \int _ { \Omega } ( 1 - q ) ^ { 2 } d \mu , \qquad A _ { \mu } ^ { \mathrm { t r } } ( T , q ) : = \int _ { \Omega } 2 q \left( 1 - \cos r \right) d \mu .
$$

These are the reaction and transport components of the action. Their ratios to the total action, when nonzero, are called the reaction and transport shares and are reported in Section 6. The identity pair reproduces the source with zero action:

$$
( \mathrm { i d } ) _ { \# } ( 1 ^ { 2 } \mu ) = \mu , \qquad \mathcal { A } _ { \mu } ( \mathrm { i d } , 1 ) = 0 , \qquad \mathbb { Z } _ { \tau } ( \mathrm { i d } , 1 \mid \mu ) = \mathcal { F } ( \mu ) .\tag{3.7}
$$

An acceptance certificate for a computed state. A useful consequence of the one-sided formulation is a certificate for an arbitrary admissible pair, independently of how that pair was obtained. Combining (1.5) with the identity competitor gives the following criterion.

Proposition 3.3 (One-step descent criterion with slack). Let $\mu \in \mathcal { M } _ { + } ( \Omega )$ satisfy $\mathcal { F } ( \mu ) < \infty$ let $( T , q )$ be admissible for $\mu ,$ let $\delta \geq 0$ , and set $\mu _ { + } : = T _ { \# } ( q ^ { 2 } \mu )$ . If

$$
\begin{array} { r } { \mathcal { T } _ { \tau } ( T , q \mid \mu ) \le \mathcal { T } _ { \tau } ( \mathrm { i d } , 1 \mid \mu ) - \delta = \mathcal { F } ( \mu ) - \delta , } \end{array}\tag{3.8}
$$

then

$$
\mathcal { F } ( \mu _ { + } ) + \frac { 1 } { 2 \tau } \mathsf { H K } ^ { 2 } ( \mu _ { + } , \mu ) \le \mathcal { F } ( \mu ) - \delta .\tag{3.9}
$$

Indeed, (1.3) gives

$$
\mathcal { F } ( \mu _ { + } ) + \frac { 1 } { 2 \tau } \mathsf { H } ^ { 2 } ( \mu _ { + } , \mu ) \le \mathcal { F } ( \mu _ { + } ) + \frac { 1 } { 2 \tau } \varLambda _ { \mu } ( T , q ) = \mathcal { T } _ { \tau } ( T , q \mid \mu ) ,
$$

and (3.8) completes the argument. The case $\delta = 0$ is the acceptance criterion in Algorithm 1; summing the inequalities over steps accumulates the verified slacks in the dissipation bound. A strictly positive slack excludes the identity step. In numerical use the criterion is exact only when the relevant functional values have controlled quadrature error. Moreover, energy decay alone does not exclude a nontrivial pure-reaction update, which is why the transport and reaction shares are monitored in Remark 4.1 and Section 6.

Primal upper bounds and restricted dual lower bounds. Primal and dual representations are equivalent before restriction. The primal upper bound and the dual lower bound remain valid after restriction. For an admissible pair $( T _ { \theta } , q _ { \theta } )$ and the induced candidate $\nu _ { \theta } : = ( T _ { \theta } ) _ { \# } ( q _ { \theta } ^ { 2 } \mu )$ , the cone action gives

$$
\mathcal { F } ( \nu _ { \theta } ) + \frac { 1 } { 2 \tau } \mathsf { H } ^ { 2 } ( \mu , \nu _ { \theta } ) \leq \mathcal { F } ( \nu _ { \theta } ) + \frac { 1 } { 2 \tau } \mathcal { A } _ { \mu } ( T _ { \theta } , q _ { \theta } ) = \mathcal { T } _ { \tau } ( T _ { \theta } , q _ { \theta } \mid \mu ) ,\tag{3.10}
$$

which together with (3.7) yields the descent estimate (3.8)–(3.9) for every pair satisfying the criterion. Independently of acceptance, every admissible pair supplies an upper bound for the exact proximal cost, and the same variables generate the next state. Writing $D _ { \mathsf { H K } } ( \varphi ; \mu , \nu )$ for the objective of (2.2)–(2.3) at a feasible potential φ, a maximization restricted to a finite class $\mathcal { D } _ { \Theta } \subset \mathcal { D }$ has the opposite error, $\begin{array} { r } { \operatorname* { s u p } _ { \varphi \in \mathcal { D } _ { \Theta } } D _ { \mathsf { W } } \le \mathsf { H K } ^ { 2 } ( \mu , \nu ) } \end{array}$ , so descent of an objective built from such a value does not by itself guarantee descent of the exact JKO functional. The price of the primal choice is that a restricted class may strictly overestimate $\mathsf { H K } ^ { 2 } ;$ ; this is a representation error, and at the unrestricted level the bound is attained by an optimal pair. Duality remains available as the lower bound in the a posteriori estimate

$$
D _ { \mathsf { H } } ( \varphi ; \mu , \nu _ { \theta } ) \leq \mathsf { H } ^ { 2 } ( \mu , \nu _ { \theta } ) \leq \mathcal { A } _ { \mu } ( T _ { \theta } , q _ { \theta } ) ,
$$

which can be used to assess the tightness of a learned Monge-growth representation. The same one-sided structure underlies the reaction-only indicator of Remark 4.1; the held-out-sample or enriched-quadrature acceptance test in Algorithm 1 remains empirical unless its error is controlled rigorously.

## 4 Neural realization and algorithm

## 4.1 Pair objective and neural parametrization

Assume that T is an orientation-preserving $C ^ { 1 }$ difeomorphism of Ω and write

$$
J _ { T } ( x ) : = | \operatorname* { d e t } D T ( x ) | .
$$

If $T _ { \# } ( q ^ { 2 } \rho d x ) = u$ dx, then the change-of-variables formula gives

$$
u ( T ( x ) ) = \frac { q ( x ) ^ { 2 } \rho ( x ) } { J _ { T } ( x ) } .\tag{4.1}
$$

Thus (1.4) becomes

$$
\begin{array} { l } { \displaystyle \mathcal { Z } _ { \tau } ( T , q \mid \rho d x ) = \int _ { \Omega } U \bigg ( \frac { q ( x ) ^ { 2 } \rho ( x ) } { J _ { T } ( x ) } \bigg ) J _ { T } ( x ) d x + \int _ { \Omega } V ( T ( x ) ) q ( x ) ^ { 2 } \rho ( x ) d x } \\ { \displaystyle \qquad + \frac { 1 } { 2 } \iint _ { \Omega \times \Omega } W ( T ( x ) , T ( y ) ) q ( x ) ^ { 2 } q ( y ) ^ { 2 } \rho ( x ) \rho ( y ) d x d y } \\ { \displaystyle \qquad + \frac { 1 } { 2 \tau } \int _ { \Omega } \left[ 1 + q ( x ) ^ { 2 } - 2 q ( x ) \cos \left( \mathrm { d } _ { \Omega } ( x , T ( x ) ) \wedge \frac { \pi } { 2 } \right) \right] \rho ( x ) d x . } \end{array}\tag{4.2}
$$

We use one parameter symbol θ for the map and growth networks:

$$
T _ { \theta } = T _ { \theta _ { T } } , \qquad q _ { \theta } = q _ { \mathrm { r e f } } \exp ( Q _ { \theta _ { q } } ) , \qquad \theta = ( \theta _ { T } , \theta _ { q } ) .
$$

Here $q _ { \mathrm { r e f } }$ is a fixed positive reference profile. The experiments use $q _ { \mathrm { r e f } } = 1$ and optimize the growth network jointly with the map. An optional pure-reaction reference can instead be defined by freezing the interaction at the current density and solving the following equation pointwise by bisection:

$$
\begin{array} { r } { q \Big ( 1 + 2 \tau \left[ U ^ { \prime } ( q ^ { 2 } \rho ) + V + W [ \rho ] \right] \Big ) = 1 , \qquad q = q _ { \mathrm { r e f } } . } \end{array}\tag{4.3}
$$

For $W = 0 ,$ , this is the exact Euler equation for the pure-reaction subproblem. For $W \neq 0 ,$ the fully implicit equation contains $W [ q ^ { 2 } \rho ]$ and couples spatial points, so (4.3) supplies only a frozen-field reference. Any such competitor is evaluated with the same fully implicit objective as the trained pair.

The choice $Q _ { \theta _ { q } } = 0$ reproduces the reference profile. At the level of continuous functions, the fixed positive continuous factor $q _ { \mathrm { r e f } }$ can be absorbed into the log-growth function by adding log $q _ { \mathrm { r e f } }$ . Approximation by a network with a bounded output remains subject to Assumption 5.1. Appendix F.3 describes the bounded neural output and initialization used in the experiments. Given a current density $\rho ,$ the notation $u _ { \theta }$ denotes the density determined by

$$
( T _ { \theta } ) _ { \# } ( q _ { \theta } ^ { 2 } \rho d x ) = u _ { \theta } d x .
$$

The map must be admissible by construction rather than made invertible only through a soft penalty. The following constructions have this property; experiment-specific variants are detailed in Appendix F.3.

A constrained residual map. The experiments in Section 6 compose two blocks of this form. On a box $\begin{array} { r } { \Omega = \prod _ { j } [ l _ { j } , h _ { j } ] } \end{array}$ of widths $w _ { j }$ we set

$$
T _ { \theta _ { T } } ( x ) = x + s ( x ) \odot \delta _ { \theta _ { T } } ( x ) , \qquad s _ { j } ( x ) = \sin \Bigl ( \pi { \frac { x _ { j } - l _ { j } } { w _ { j } } } \Bigr ) ,\tag{4.4}
$$

where $\odot$ is the componentwise product and $\delta _ { \theta _ { T } }$ is a bounded network output with $| \delta _ { j } | \le A$ and $| \delta | \le \pi / 2 - \varepsilon$ . Only the factor of coordinate $j$ multiplies component $j$ of the displacement. This enforces the domain constraint: since sin $( \pi t ) / \pi \leq \operatorname* { m i n } \{ t , 1 - t \}$ , the bound $A \le w _ { j } / \pi$ gives $| s _ { j } \delta _ { j } | \leq \mathrm { d i s t } ( x _ { j } , \{ l _ { j } , h _ { j } \} )$ , so $T _ { \theta _ { T } } ( \Omega ) \subset \Omega$ with each boundary face preserved: the normal coordinate is fixed, while tangential motion along the face is allowed. $\mathrm { O n }$ the torus $s \equiv 1$ and the network takes a periodic embedding of the spatial coordinates as input.

The residual network uses continuously diferentiable activations; ReLU is excluded from the map class. For ReLU, the location at which the map derivative jumps can depend on the parameters. Diferentiating the integrated internal energy then produces an interface term that pointwise automatic diferentiation does not capture. The radial displacement bound also uses a smooth contraction. For $c = \pi / 2 - \varepsilon$ and $r _ { 0 } = c / 2$ , a raw displacement $z$ of norm r is unchanged for $r \leq r _ { 0 }$ and otherwise becomes $f ( r ) z / r$ , where

$$
f ( r ) = r _ { 0 } + ( c - r _ { 0 } ) \operatorname { t a n h } \left( \frac { r - r _ { 0 } } { c - r _ { 0 } } \right) .
$$

This cap is $C ^ { 2 }$ at $r = r _ { 0 }$ , has radial derivative and tangential scaling at most one, and preserves each component bound. Thus it preserves the contraction estimate below while avoiding a moving derivative discontinuity.

Orientation and injectivity are obtained from a contraction estimate rather than from a Jacobian penalty. Writing L for a Lipschitz bound of the network in the normalized input variable, one has the pointwise bound

$$
\| D \big ( s \odot \delta _ { \theta _ { T } } \big ) ( x ) \big \| _ { 2 } \ \leq \ \frac { A ( \pi + 2 L ) } { w _ { \mathrm { m i n } } } \ = : \ \kappa _ { \theta } ,
$$

because the diagonal part contributes max<sub>j</sub> $| s _ { j } ^ { \prime } | | \delta _ { j } | \le \pi A / w _ { \mathrm { m i n } }$ and the remaining part contributes $\| s \| _ { \infty } \operatorname { L i p } ( \delta _ { \theta _ { T } } ) \leq 2 A L / w _ { \operatorname* { m i n } } ;$ on the torus the same argument gives $\kappa _ { \theta } \leq 2 \pi A L / w _ { \mathrm { { m i n } } }$ If $\kappa _ { \theta } < 1$ , then $D T _ { \theta _ { T } } = I + D ( s \odot \delta _ { \theta _ { T } } )$ is invertible with det $D T _ { \theta _ { T } } > 0$ and $T _ { \theta _ { T } }$ is injective because its displacement is a strict contraction.

On a box, write G = int Ω. The inverse function theorem makes $T _ { \theta _ { T } } ( G )$ open in $\mathbb { R } ^ { d }$ and hence contained in G. Boundary preservation gives $T _ { \theta _ { T } } ( G ) = T _ { \theta _ { T } } ( \Omega ) \cap G$ , which is relatively closed in G by compactness. Connectedness gives $T _ { \theta _ { T } } ( G ) = G$ , and taking closures yields $T _ { \theta _ { T } } ( \Omega ) = \Omega$ On the torus, the image is likewise open by local invertibility and closed by compactness, hence equals the connected torus. Thus $T _ { \theta _ { T } }$ is an orientation-preserving difeomorphism of Ω onto itself, as is any finite composition of such blocks, and (4.1) is valid.

The implementation bounds each residual block using network coeficients and the spatial input derivatives, with tighter interval bounds when needed. It scales the displacement to enforce contraction, the domain constraint, and the displacement bound. Validation checks these bounds again; a candidate that fails is rejected. The sampled minimum of $J _ { T _ { \theta } }$ remains an additional diagnostic.

A neural ODE flow. One convenient alternative is the time-one flow of a neural ordinary diferential equation,

$$
\frac { d } { d t } \Phi _ { \theta , t } ( x ) = v _ { \theta _ { T } } \bigl ( t , \Phi _ { \theta , t } ( x ) \bigr ) , \qquad \Phi _ { \theta , 0 } ( x ) = x , \qquad T _ { \theta _ { T } } ( x ) = \Phi _ { \theta , 1 } ( x ) .\tag{4.5}
$$

On a bounded domain, $v _ { \theta _ { T } } ( t , \cdot )$ is taken $C ^ { 1 }$ in space, globally Lipschitz, and satisfies $v _ { \theta _ { T } } \cdot \mathsf { n } = 0$ H<sup>d−1</sup>-almost everywhere on ∂Ω; the stronger condition $v _ { \theta _ { T } } = 0$ on ∂Ω is a simple suficient choice. The flow preserves the boundary and maps Ω difeomorphically onto itself. On the torus, the vector field is periodic. In either case the exact flow is orientation preserving and Liouville’s formula gives

$$
\log J _ { T _ { \theta _ { T } } } ( x ) = \int _ { 0 } ^ { 1 } \operatorname { d i v } v _ { \theta _ { T } } \bigl ( t , \Phi _ { \theta , t } ( x ) \bigr ) d t , \qquad J _ { T _ { \theta _ { T } } } ( x ) > 0 ,\tag{4.6}
$$

as in continuous normalizing flows [29, 30]. Other invertible architectures may be used if they map Ω onto itself and provide a controlled positive Jacobian. The ODE construction includes the identity map through $v _ { \theta _ { T } } = 0$ . The experiments use composed residual maps with Jacobians evaluated from the network derivatives; their common architecture and density model are described in Appendix F.3. For either map class, the approximation result below requires Assumption 5.1 for the regular optimal pair under consideration.

## 4.2 Sampling, mass propagation, and the algorithm

Let ${ \mathfrak { m } } = \mu ( \Omega )$ and $\bar { \mu } = \mu / \mathfrak { m }$ . All terms in (4.2) can be evaluated from samples of the current measure. For example,

$$
\int _ { \Omega } U \left( \frac { q ^ { 2 } \rho } { J _ { T } } \right) J _ { T } d x = \mathfrak { m } \mathbb { E } _ { X \sim \bar { \mu } } \left[ \frac { J _ { T } ( X ) } { \rho ( X ) } U \left( \frac { q ( X ) ^ { 2 } \rho ( X ) } { J _ { T } ( X ) } \right) \right] ,\tag{4.7}
$$

whereas the interaction is

$$
{ \frac { \mathfrak { m } ^ { 2 } } { 2 } } \mathbb { E } _ { X , Y \sim { \bar { \mu } } } \left[ W ( T ( X ) , T ( Y ) ) q ( X ) ^ { 2 } q ( Y ) ^ { 2 } \right] .\tag{4.8}
$$

For a single batch of independent samples, the interaction is estimated by the of-diagonal U-statistic

$$
{ \frac { { \mathfrak { m } } ^ { 2 } } { 2 B ( B - 1 ) } } \sum _ { i \neq j } W ( T ( X _ { i } ) , T ( X _ { j } ) ) q ( X _ { i } ) ^ { 2 } q ( X _ { j } ) ^ { 2 } .\tag{4.9}
$$

The interaction estimator is unbiased when the $X _ { i }$ are independent draws from $\bar { \mu } ,$ since each term with $i \neq j$ has the product sampling law. The proximal term is m times the empirical average of the integrand in (1.2). If a uniformly selected subset of size $s \geq 2$ is used, its mass weights are $w _ { i } = \mathfrak { m } q ( X _ { i } ) ^ { 2 } / s$ and the estimator is

$$
{ \frac { s } { 2 ( s - 1 ) } } \sum _ { i \neq j } w _ { i } w _ { j } W ( T ( X _ { i } ) , T ( X _ { j } ) ) .
$$

The correction $s / ( s - 1 )$ is fixed by the sample count and does not depend on the learned growth weights. When sampling from the current normalized measure, it draws independently, with replacement, from a density-weighted finite proposal pool. Conditional on that pool the of-diagonal estimator is unbiased for the corresponding discrete sampling law; approximation of the current continuum measure by the pool remains a separate sampling error.

The interaction benchmark instead uses deterministic periodic quadrature. If $h _ { i }$ are spatial quadrature weights and $\omega _ { i } = h _ { i } \rho ( X _ { i } )$ , its pair energy is

$$
{ \frac { 1 } { 2 } } \sum _ { i , j } \omega _ { i } \omega _ { j } q ( X _ { i } ) ^ { 2 } q ( X _ { j } ) ^ { 2 } W ( { \cal T } ( X _ { i } ) , { \cal T } ( X _ { j } ) ) .\tag{4.10}
$$

This full weighted double sum includes the diagonal and has no U-statistic correction. A shifted grid with more quadrature nodes checks the trained pair; it is distinct from the training grid but is not a statistically independent random sample. Quadrature refinement is an empirical discretization check, not a rigorous continuum error bound. The neural parametrization is the same for stochastic and deterministic integration.

The normalized sampling law must also be propagated with the growth weights. Let $\pi =$ $\mu / { \mathfrak { m } }$ and let $\mu _ { + } = T _ { \# } ( q ^ { 2 } \mu )$ . Then

$$
Z : = \int _ { \Omega } q ^ { 2 } d \pi , \qquad { \mathfrak { m } } _ { + } = { \mathfrak { m } } Z , \qquad \pi _ { + } : = { \frac { \mu _ { + } } { { \mathfrak { m } } _ { + } } } = T _ { \# } \left( { \frac { q ^ { 2 } } { Z } } \pi \right) .\tag{4.11}
$$

Consequently, if $X _ { i } ~ \sim ~ \pi$ and $Y _ { i } ~ = ~ T ( X _ { i } )$ , the points $Y _ { i }$ represent $\pi _ { + }$ only after attaching normalized weights $w _ { i } = q ( X _ { i } ) ^ { 2 } / \sum _ { j } q ( X _ { j } ) ^ { 2 }$ or after resampling according to these weights. The empirical measure of the unweighted points $T ( X _ { i } )$ represents the next normalized measure only when $q$ is constant. At the transformed sample locations the density update is

$$
\log u _ { + } ( T ( x ) ) = \log \rho ( x ) + 2 \log q ( x ) - \log J _ { T } ( x ) .\tag{4.12}
$$

The formulas in (4.11) are exact; replacing Z and $\pi _ { + }$ by empirical estimates introduces a separate particle/quadrature error. Importance resampling plays an analogous role in iterative neural JKO sampling [28].

If rigorous error bounds $| \widehat { \mathcal { I } } _ { \tau } - \mathcal { I } _ { \tau } | \leq \varepsilon _ { I , k }$ and $| \widehat { \mathcal { F } } - \mathcal { F } | \le \varepsilon _ { F , k }$ are available, the stronger acceptance condition

$$
\widehat { \mathcal { T } } _ { \tau } ( \theta ) + \varepsilon _ { I , k } \leq \widehat { \mathcal { F } } ( \widehat { \mu } _ { k } ^ { \tau } ) - \varepsilon _ { F , k }
$$

implies the exact descent criterion (3.8) with $\delta = 0$ . The empirical guard used in the experiments does not provide these quadrature error bounds.

The inner loop needs density values at its evaluation points. Reusing a batch supplies values only at the stored points; a density model allows evaluation at new points. All recursive trajectory experiments fit a positive neural density after each step, with architectures listed in Table 2 and fitting details in Appendix F.3.

Algorithm 1 Restarted Monge-growth neural HK-JKO scheme with mass-corrected propagation   
1: Set $\widehat { \mu } _ { 0 } ^ { \tau } = \rho _ { 0 } d x , \mathfrak { m } _ { 0 } = \widehat { \mu } _ { 0 } ^ { \tau } ( \Omega )$ , and $\pi _ { 0 } = \widetilde { \mu } _ { 0 } ^ { \tau } / \mathfrak { m } _ { 0 }$   
2: for $k = 0 , 1 , \ldots , K - 1$ do   
3: Write $\widehat { \mu } _ { k } ^ { \tau } = u _ { \theta , k } ^ { \tau } d x$ and choose current-measure sampling, importance sampling, or de  
terministic quadrature with source-mass weights $\omega _ { i } .$   
4: Initialize a fresh admissible pair from the current-density predictor (Appendix F.3), or   
start from (id, 1) in the independent pair-scaling and certificate tests.   
5: for training iterations do   
6: Use the training points $X _ { 1 } , \ldots , X _ { B }$ and weights from the chosen rule, refreshing   
samples according to the experiment protocol.   
7: Evaluate $T _ { \theta } ( X _ { i } ) , q _ { \theta } ( X _ { i } ) , J _ { T _ { \theta } } ( X _ { i } )$ , and $u _ { \theta , k } ^ { \tau } ( X _ { i } )$   
8: Approximate (4.2), using (4.9) for iid interaction samples or (4.10) for deterministic   
quadrature.   
9: Update θ to decrease the objective. Compare saved parameter iterates (checkpoints)   
using one fixed training estimator throughout the step.   
10: end for   
11: Restore the selected parameters, fix the network normalization used for evalua  
tion, enforce map contraction, and recompute the training gradient and growth   
residual. On a sample independent of training (held out) or a distinct finer   
quadrature grid, evaluate ${ \widehat { \cal T } } _ { \tau } ( \theta )$ and $\widehat { \mathcal { T } } _ { \tau } ( \mathrm { i d } , 1 ) = \widehat { \mathcal { F } } ( \widehat { \mu } _ { k } ^ { \tau } )$ . Reject the candidate if   
its objective exceeds the identity value beyond the recorded tolerance or another   
required validation check fails. This is an empirical acceptance test.   
12: The common neural solver uses strict rejection: if no candidate passes after the   
prescribed attempts, terminate with a failure; do not advance physical time or   
propagate an identity/growth fallback. The identity is the acceptance baseline,   
and the analytic pure-growth competitor is diagnostic only.   
13: Record $\mathcal { A } ^ { \mathrm { t r } } , \mathcal { A } ^ { \mathrm { r e } }$ from (3.6), min<sub>i</sub> $J _ { T _ { \theta } } ( X _ { i } )$ , the applicable certified map-contraction bound   
(Appendix F.3), and $\operatorname* { m a x } _ { i } \mathsf { d } _ { \Omega } ( X _ { i } , T _ { \theta } ( X _ { i } ) )$   
14: Freeze the accepted pair $( T _ { \theta , k + 1 } , q _ { \theta , k + 1 } )$ and choose transition points with source-mass   
weights $\omega _ { i } ;$ current-measure samples use $\omega _ { i } = \mathfrak { m } _ { k } / B$   
15: Set $Y _ { i } = T _ { \theta , k + 1 } ( X _ { i } ) , { \widetilde w } _ { i } = \omega _ { i } q _ { \theta , k + 1 } ( X _ { i } ) ^ { 2 }$ , and $\widehat { Z } _ { k + 1 } = \mathfrak { m } _ { k } ^ { - 1 } \sum _ { i } \widetilde { w } _ { i }$   
16: Update ${ \mathfrak { m } } _ { k + 1 } = { \mathfrak { m } } _ { k } { \widehat { Z } } _ { k + 1 }$ and $w _ { i } = \widetilde { w } _ { i } / \sum _ { j } \widetilde { w } _ { j }$   
17: Retain $\sum _ { i } w _ { i } \delta _ { Y _ { i } }$ as a weighted approximation of $\pi _ { k + 1 }$ , or resample the $Y _ { i }$ according to   
$( w _ { i } ) _ { i }$   
18: (Density model.) The recursive runs of Section 6 use density-model propagation:   
the log-density values (4.12) at the mapped points $Y _ { i }$ are fitted by a positive   
neural density model. Output coeficients are fitted by variable projection, and   
a scalar shift restores sampled mass. The fit must pass the recorded validation   
tolerances for density error, mass error, and energy increase; otherwise the step   
is rejected. This projection adds a per-step representation error.   
19: Use the target values log $u _ { \mathrm { p a i r } } ( Y _ { i } ) \stackrel { \cdot } { = } \log u _ { \theta , k } ^ { \hat { \tau _ { k } } } ( \dot { X _ { i } } ) + 2 \log q _ { \theta , k + 1 } ( X _ { i } ) - \log J _ { T _ { \theta , k + 1 } } ( X _ { i } ) .$   
20: Define the ideal pair endpoint $\nu _ { k + 1 } = ( T _ { \theta , k + 1 } ) _ { \# } ( q _ { \theta , k + 1 } ^ { 2 } \widehat { \mu } _ { k } ^ { \tau } )$   
21: For the density-model runs set $\widehat { \mu } _ { k + 1 } ^ { \tau } = \mathcal { P } _ { k } \nu _ { k + 1 }$ , where $\mathcal { P } _ { k }$ is the accepted mass-corrected   
fit; exact propagation would instead set $\widehat { \mu } _ { k + 1 } ^ { \tau } = \nu _ { k + 1 }$   
22: end for

The cone action and its upper bound apply to the pair-induced measure $\nu _ { k + 1 }$ before density fitting. The fitted density is checked separately for representation error, mass error, and energy increase. These empirical checks do not certify its exact HK-JKO objective or transfer the convergence result for exact pair propagation to the trained recursion. Weighted-particle propagation is an alternative, but repeated reweighting can reduce particle diversity; its efective sample size should therefore be monitored.

Remark 4.1 (Cutof and reaction-only degeneracy). Where $\mathsf { d } _ { \Omega } ( x , T _ { \theta } ( x ) ) > \pi / 2$ , the cone integrand is independent of the map and its $\theta _ { T } { - } g r a d i e n t$ vanishes. Although the exact optimal Monge map stays below the cutof by Corollary C.8, a neural iterate may enter the region where the map gradient of the cone cost vanishes. If map learning then stalls, the update can collapse to the pure-reaction pair $T _ { \theta } = \mathrm { i d }$ , producing the reaction-only limit $\partial _ { t } \rho = - 4 \rho g _ { \rho }$ while still satisfying the descent criterion with $\delta = 0$ . Energy monotonicity is therefore insuficient: the maximal displacement and the transport share in (3.6) should be monitored. This can be mitigated by constraining the displacement radius, by choosing the transport–reaction scaling so that diam $\Omega < \pi / 2$ , or by using a parametrization for which the cutof is inactive. Penalties or smoothing, however, define a modified numerical objective and do not by themselves guarantee either $T _ { \theta } ( \Omega ) \subset \Omega$ or ${ { J } _ { { { T } _ { \theta } } } } > 0$

The transport share can also be used as an optional additional acceptance criterion. Fix $\kappa \in ( 0 , 1 )$ and accept a trained pair only $i f ,$ in addition to the descent criterion (3.8), it satisfies

$$
\begin{array} { r } { \mathcal { A } _ { \mu } ^ { \mathrm { t r } } ( T , q ) \geq \kappa \mathcal { A } _ { \mu } ( T , q ) . } \end{array}\tag{4.13}
$$

Since (4.13) only shrinks the accepted set, Proposition 3.3 applies verbatim to every pair that passes both criteria, while $T = \mathrm { i d }$ is now rejected whenever $\mathcal { A } _ { \mu } ( T , q ) > 0$ . Splitting the metric– dissipation identity (2.14) supplies the theoretical reference share $\textstyle \int | \nabla g _ { \rho _ { 1 } } | ^ { 2 } \rho _ { 1 } d x / I _ { g } ( \rho _ { 1 } )$ for interpreting choices of κ. Like the descent criterion itself, this additional test is empirical unless its quadrature error is controlled rigorously.

## 5 Guarantees for parametric and computed steps

This section gives two complementary guarantees. Theorem C is an a priori approximation result: if a regular optimal pair is approximable, there exist admissible parametric pairs whose endpoint densities and objective values converge to the exact ones. Theorem D gives an a posteriori error bound for an admissible computed measure under additional convexity and dualfeasibility assumptions. It does not require convergence of the optimizer or an approximation assumption on the parametric class.

## 5.1 Theorem C: one-step representability

Assumption 5.1 (Approximation of a regular Monge-growth pair). Let $( T , q )$ be a Mongegrowth pair that is regular in the following sense: T is an orientation-preserving $C ^ { 1 }$ difeomorphism of Ω onto itself with inf $J _ { T } > 0 _ { ; }$ , and $q \in C ( \Omega )$ satisfies $0 <$ inf $q \leq \operatorname* { s u p } q < \infty$ . The chosen network classes admit parameters $\theta _ { n }$ such that each $T _ { \theta _ { r } }$ is an orientation-preserving $C ^ { 1 }$ difeomorphism of Ω onto itself with inf<sub>Ω</sub> $J _ { T _ { \theta _ { n } } } > 0$ , and

$$
\lVert T _ { \theta _ { n } } - T \rVert _ { C ^ { 1 } ( \Omega ) } + \lVert q _ { \theta _ { n } } - q \rVert _ { L ^ { \infty } ( \Omega ) } \longrightarrow 0 .\tag{5.1}
$$

Surjectivity onto Ω is included because Appendix D compares induced densities through globally defined inverses $T _ { \theta _ { n } } ^ { - 1 }$ . The constrained residual map (4.4), its finite compositions, and the neural ODE flow (4.5) are onto under the respective conditions in Section 4.1.

Theorem 5.2 (One-step variational approximation and asymptotic tightness of the cone action). Assume the hypotheses of Theorem 2.3. Let $\bar { \mu } = \bar { \rho }$ dx with $\bar { \rho } \in C ( \Omega )$ and $C ^ { - 1 } \leq \bar { \rho } \leq C$ on Ω for some $C \geq 1$ , and let

$$
\mu _ { * } = \rho _ { * } d x \in \mathop { \mathrm { a r g m i n } } _ { \sigma \in \mathcal { M } _ { + } ( \Omega ) } \mathcal { I } _ { \tau } ( \sigma \mid \bar { \mu } ) .
$$

Suppose that an optimal Monge-growth pair $( T _ { * } , q _ { * } )$ from $\bar { \mu }$ to $\mu _ { * }$ is regular and satisfies the approximation property of Assumption 5.1 in the chosen neural class. Then, for every $\varepsilon > 0$ there is a neural pair $( T _ { \theta } , q _ { \theta } )$ such that, with

$$
( T _ { \theta } ) _ { \# } ( q _ { \theta } ^ { 2 } \bar { \mu } ) = u _ { \theta } d x = : \nu _ { \theta } ,
$$

we have

$$
\| u _ { \theta } - \rho _ { * } \| _ { L ^ { \infty } ( \Omega ) } \leq \varepsilon ,\tag{5.2}
$$

$$
0 \leq \mathcal { T } _ { \tau } ( T _ { \theta } , q _ { \theta } \mid \bar { \mu } ) - \mathcal { T } _ { \tau } ( \mu _ { * } \mid \bar { \mu } ) \leq \varepsilon ,\tag{5.3}
$$

and

$$
0 \leq \mathcal { I } _ { \tau } ( \nu _ { \theta } \mid \bar { \mu } ) - \mathcal { I } _ { \tau } ( \mu _ { * } \mid \bar { \mu } ) \leq \varepsilon .\tag{5.4}
$$

Moreover, the primal cone majorization is quantitatively tight:

$$
0 \leq \mathcal { A } _ { \bar { \mu } } ( T _ { \theta } , q _ { \theta } ) - { \sf H } ^ { 2 } ( \bar { \mu } , \nu _ { \theta } ) \leq 2 \tau \varepsilon .\tag{5.5}
$$

Remark 5.3 (Role and scope of the regular-pair assumption). Assumption 5.1 imposes regularity and approximation conditions, as in classical convergence analyses. It separates the regularity of the exact step from the approximation capability of the chosen neural class. Theorem 5.2 transfers the assumed pair approximation to density and objective convergence and vanishing cone-action overestimation. The $C ^ { 1 } \times L ^ { \infty }$ topology is a natural suficient choice because the induced density (4.1) depends on the Jacobian of the transport map: approximation only in $L ^ { 2 } ( \bar { \mu } )$ , with bounded growth factors, controls the HK distance between the endpoints (Lemma 3.2) but does not by itself control their internal energy.

The exact theory provides essential injectivity but does not establish the stronger regularity assumed here. For the approximating pairs, the two-block residual architecture satisfies the structural requirements: under the contraction and domain constraints of Section $4 . 1 ,$ its maps are orientation-preserving $C ^ { 1 }$ difeomorphisms of Ω onto itself with Jacobian bounded away from zero, and its growth factors are positive and continuous. Every accepted map satisfies its contraction certificate, with positive sampled Jacobians (Tables 3 and $\it 4 )$ . The experiments also provide evidence of small-time consistency: the rescaled-field errors of independently trained pairs decrease under time-step refinement (Table $\mathbf { \epsilon } _ { 4 } ) ,$ , and the relative $L ^ { 2 }$ residual of the PDE decreases from 1.8% to 0.07% (Figure 1). These observations do not establish the fixed-τ approximation property (5.1). Proposition 3.3 and Theorem $5 . 4$ provide complementary guarantees without this assumption. For the potential-generated parametrization, suficient conditions and the remaining regularity issue are discussed in Appendix F.1.

Idea of the proof (Appendix D). The argument transfers convergence through the change-ofvariables formula. Assumption 5.1 gives $T _ { \theta _ { n } } \to T _ { * }$ in $C ^ { 1 }$ and $q _ { \theta _ { n } } \to q _ { * }$ <sub>∗</sub> uniformly; since ${ { J } _ { { { T } _ { * } } } }$ is bounded away from zero, so are the ${ { J } _ { { { T } _ { \theta _ { n } } } } }$ for large $n _ { \mathrm { : } }$ , and the inverse maps converge uniformly. Writing the induced densities through (4.1) at the same target point then gives $u _ { \theta _ { n } } \ \to \ \rho _ { * }$ uniformly, with all densities eventually bounded above and away from zero, so the three energy terms and the cone integrand converge and the pair objective converges to $\mathcal { T } _ { \tau } ( T _ { * } , q _ { * } \ | \ \bar { \mu } )$ . By Proposition 3.1 the latter equals the JKO minimum, and every competitor is bounded below by that minimum, which makes the gap nonnegative and vanishing. The tightness statement (5.5) follows algebraically from the definitions:

$$
A _ { \bar { \mu } } ( T _ { \theta } , q _ { \theta } ) - { \sf H } ^ { 2 } ( \bar { \mu } , \nu _ { \theta } ) = 2 \tau \big [ \mathcal { Z } _ { \tau } ( T _ { \theta } , q _ { \theta } \mid \bar { \mu } ) - \mathcal { I } _ { \tau } ( \nu _ { \theta } \mid \bar { \mu } ) \big ] ,
$$

and the right-hand side is dominated by the pair-objective gap. This last identity also shows directly how the metric overestimate is measured: it is exactly the excess of the pair objective over the exact JKO objective at the induced state.

For fixed τ , Corollary D.2 in Appendix D extends this recovery result to any fixed finite number of JKO steps, provided the regular-pair approximation hypothesis holds at each step.

## 5.2 Theorem D: an a posteriori error estimate from a feasible dual pair

Theorem C establishes approximation by a parametric class, conditional on a regular optimal pair. We now bound the error of an individual computed measure. Its pair objective gives an upper bound on the exact JKO objective by (3.10). A globally feasible HK dual pair gives a lower bound on the squared distance. When the interaction is positive semidefinite, this dual bound also leads to a lower bound for the exact JKO minimum. The diference between the primal and dual bounds therefore controls the candidate’s JKO objective gap.

Theorem 5.4 also bounds the internal-energy convexity remainder $U ( u ) - U ( \rho _ { 1 } ) - U ^ { \prime } ( \rho _ { 1 } ) ( u$ $\rho _ { 1 } )$ integrated over Ω. For Boltzmann entropy, this gives an $L ^ { 1 }$ density error bound. The proof and the explicit gap decomposition appear in Appendix E.

Theorem 5.4 (Theorem D: an a posteriori error estimate for an admissible pair). Assume the hypotheses of Theorem 2.6 and suppose that $W$ is positive semidefinite, in the sense that

$$
\iint _ { \Omega \times \Omega } W ( x , y ) d \eta ( x ) d \eta ( y ) \ge 0 \qquad f o r e v e r y f i n i t e s i g n e d m e a s u r e \eta .
$$

Then the exact step $\mu _ { 1 } = \rho _ { 1 } d x = S _ { \tau } ( \bar { \mu } )$ is unique. Let $( T , q )$ be any admissible pair for µ¯ such that $\nu : = T _ { \# } ( q ^ { 2 } \bar { \mu } ) = \ i$ dx and $\mathcal { T } _ { \tau } ( T , q \mid \bar { \mu } ) < \infty$ . Let $\zeta , \xi$ be bounded Borel functions satisfying the global dual-feasibility condition

$$
\zeta ( x ) + \xi ( y ) \leq c _ { \sf H K } ( x , y ) \qquad f o r \ e v e r y \ x , y \in \Omega .\tag{5.6}
$$

Put $\begin{array} { r } { U ^ { * } ( p ) : = \operatorname* { s u p } _ { s \geq 0 } \{ p s - U ( s ) \} } \end{array}$ and

$$
a : = V + W [ u ] + \frac { 1 } { 2 \tau } \big ( 1 - e ^ { - \zeta } \big ) , \qquad \mathcal { R } : = \mathcal { Z } _ { \tau } ( T , q \mid \bar { \mu } ) + \int _ { \Omega } U ^ { * } ( - a ) d x + \mathcal { W } ( \nu ) - \frac { 1 } { 2 \tau } \int _ { \Omega } \bigl ( 1 - e ^ { - \xi } \bigr ) d \bar { \mu } .\tag{5.7}
$$

Then $\mathcal { R } \geq 0$ and

$$
0 \leq \mathcal { I } _ { \tau } ( \nu \mid \bar { \mu } ) - \mathcal { I } _ { \tau } ( \mu _ { 1 } \mid \bar { \mu } ) \leq \mathcal { R } .\tag{5.8}
$$

Moreover,

$$
\int _ { \Omega } \bigl [ U ( u ) - U ( \rho _ { 1 } ) - U ^ { \prime } ( \rho _ { 1 } ) ( u - \rho _ { 1 } ) \bigr ] d x \leq \mathcal { R } .\tag{5.9}
$$

Corollary 5.5 (Boltzmann $L ^ { 1 }$ error bound). Under the hypotheses of Theorem $5 . 4 ,$ , assume in addition that $U ( s ) = s \log s - s$ . Then

$$
\| u - \rho _ { 1 } \| _ { L ^ { 1 } ( \Omega ) } ^ { 2 } \le 2 \bigl ( \nu ( \Omega ) + \mu _ { 1 } ( \Omega ) \bigr ) \mathcal { R } \le 2 \bigl ( \nu ( \Omega ) + M _ { \tau , \bar { \mu } } \bigr ) \mathcal { R } .\tag{5.10}
$$

The gap decomposes into a Fenchel–Young residual for the discrete Euler relation and a primal–dual metric gap; the exact decomposition is Proposition E.1 in Appendix E.

Remark 5.6 (Constructing a feasible pair; monotonicity of the error bound in $\zeta )$ . Global feasibility (5.6) need not be verified after the fact; it can be arranged by construction. Fix any bounded Borel $\xi$ and set

$$
\zeta ( x ) : = \operatorname* { i n f } _ { y \in \Omega } \left\{ c _ { \mathsf { \mathsf { H K } } } ( x , y ) - \xi ( y ) \right\} .
$$

Then (5.6) holds for every pair $( x , y )$ by definition, and $- \operatorname* { s u p } _ { \Omega } \xi \leq \zeta \leq - \operatorname* { i n f } _ { \Omega } \xi$ because $c _ { \mathsf { H K } } \geq$ 0 and $c _ { \mathsf { H K } } ( x , x ) = 0$ , so $\zeta$ is admissible in Theorem 5.4. If moreover the transport–reaction weighting is normalized so that diam $\Omega < \pi / 2$ , then c is Lipschitz on $\Omega \times \Omega$ and $\zeta$ inherits its Lipschitz constant.

A pointwise lower bound for the exact c-transform preserves feasibility and gives a possibly larger error bound. Suppose $\zeta _ { - } \leq \zeta$ pointwise. Then ζ is again feasible. Moreover, $\zeta \mapsto a$ is nondecreasing by (5.7), and $U ^ { * }$ is nondecreasing on $\mathbb { R } ,$ so replacing ζ by ζ<sub>−</sub> can only increase $\mathcal { R }$ . Hence any rigorous pointwise lower bound for the exact c-transform yields a valid, possibly larger, error bound. For Lipschitz ξ and diam $\Omega < \pi / 2$ , one such lower bound is available on a finite grid $\{ y _ { j } \} \subset \Omega$ with fill radius $h : = \operatorname* { s u p } _ { y \in \Omega }$ min<sub>j</sub> ${ \mathsf { d } } _ { \Omega } ( y , y _ { j } )$

$$
\zeta _ { - } ( x ) : = \operatorname* { m i n } _ { i } \{ c _ { \mathsf { t } \mathsf { K } } ( x , y _ { j } ) - \xi ( y _ { j } ) \} - L h \leq \zeta ( x ) , \qquad L : = \mathrm { L i p } ( c _ { \mathsf { t } \mathsf { K } } ) + \mathrm { L i p } ( \xi ) ,\tag{5.11}
$$

uniformly in x. This correction is what distinguishes a rigorous evaluation from the batchwise minimum excluded in Remark 5.7. An unadjusted minimum over sampled points is an upper bound $f o r \zeta$ and does not guarantee feasibility, whereas subtracting Lh restores it. Global feasibility can therefore be enforced by construction. A useful numerical error bound still requires deterministic control of the three quadratures in (5.7) and a suficiently tight dual choice.

Remark 5.7 (Scope). Theorem 5.4 avoids approximation and regular-pair assumptions, but it applies only to $W = 0$ or, more generally, a positive-semidefinite interaction and requires global dual feasibility. Numerical use additionally requires a rigorous evaluation of the gap. In particular, replacing the exact c-transform by a batchwise minimum does not verify (5.6), whereas the corrected variant (5.11) of Remark 5.6 does. The estimate controls only the induced density, not the map-growth pair or its cone action. Finite-horizon propagation would require an additional stability estimate for $S _ { \tau }$ and rigorously controlled per-step gaps; no such propagation is proved here.

## 6 Numerical experiments

All main experiments use the common neural solver. Time steps, physical horizons and optimizer budgets are reported in Appendix F.3. The code for the numerical experiments is available at https://github.com/tyty4/hkflow.

Trajectory states are fitted with the positive neural density family in Table 2. Masses, moments, and reference errors are quadrature estimates; Gaussian references are used only for evaluation; the trainable density family also contains nonquadratic log densities. Finite-box errors include truncation. Comparisons use a common seed and are descriptive, not statistical significance claims.

Time-dependent error curves are shown for $t \geq 5 \tau$ , omitting the first five time levels; other diagnostics and tabulated values use the full trajectories.

Trajectory pairs are checked against the identity on held-out samples or enriched quadrature. The common neural solver terminates if its prescribed attempts fail validation; no fallback is propagated. Rejected attempts do not advance physical time. Ablation endpoints use the latest time attained by every compared trajectory; curves retain their own accepted clocks. Tables report these times, null/fallback counts, empirical guards, action shares, and map contraction bounds.

The “reference transport share” is a leading-order diagnostic at the numerical reference state, when available, or the projected neural endpoint. It is sensitive to diferentiation, quadrature, and small near-equilibrium action; it is not a finite-step transport plan. The optional transportshare acceptance criterion is not imposed. Implementation details are in Appendix F.3.

Table 2: Positive neural state families and the actual pair-training rules. Importance counts are draws before rejection outside the box.
<table><tr><td>Experiment</td><td> $d$ </td><td>pair-training rule</td><td>density family</td></tr><tr><td>Entropy-potential</td><td></td><td>1 Gauss-Legendre 257</td><td>mixed-activation FCNN</td></tr><tr><td>Entropy-potential</td><td></td><td>2 Gauss-Legendre 4225</td><td>mixed-activation FCNN</td></tr><tr><td>HK entropy flow</td><td></td><td>8 importance draws 2048</td><td>mixed-activation FCNN</td></tr><tr><td>Fokker-Planck type</td><td></td><td>2 Gauss-Legendre 1089</td><td>mixed-activation FCNN</td></tr><tr><td>Implicit interaction</td><td></td><td>2 periodic grid 961</td><td>mixed-activation FCNN</td></tr></table>

Here, “mixed-activation FCNN” denotes a fully connected neural network combining quadratic and tanh activations; see Appendix F.3.

<table><tr><td>Experiment</td><td> $d$ </td><td>rel.  $L ^ { 2 }$ </td><td>maxk rel.  $L ^ { 2 }$ </td><td> $\overline { { { A ^ { \mathrm { t r } } } / { A } } }$ </td><td>reference</td><td>min  $J _ { T }$ </td><td>sup  $\lVert D \delta \rVert$ </td></tr><tr><td>Entropy-potential</td><td>1</td><td>2.006e-05</td><td>9.819e-03</td><td>0.150</td><td>0.145</td><td>0.981</td><td>0.539</td></tr><tr><td>Entropy-potential</td><td>2</td><td>5.319e-04</td><td>0.016</td><td>0.105</td><td>0.102</td><td>0.969</td><td>0.644</td></tr><tr><td>HK entropy flow</td><td>8</td><td>8.029e-03</td><td>8.029e-03</td><td>0.069</td><td>0.069</td><td>0.995</td><td>0.806</td></tr><tr><td>Fokker-Planck type</td><td>2</td><td>3.476e-03</td><td>3.476e-03</td><td>0.228</td><td>0.228</td><td>0.995</td><td>0.453</td></tr><tr><td>Implicit interaction</td><td>2</td><td>1.312e-03</td><td>2.434e-03</td><td>0.537</td><td>0.536</td><td>0.977</td><td>0.092</td></tr></table>

Table 3: Trajectory diagnostics. Action shares are cone-action-weighted; max $\iota \kappa _ { \theta , \ell } < 1$ certifies the two blocks of each accepted map.

## 6.1 One-step PDE consistency and small-time scaling

For $g ( \rho ) = \log \rho + V$ , the one-dimensional HK flow is

$$
\begin{array} { r } { \partial _ { t } \rho = \partial _ { x } ( \rho \partial _ { x } g ) - 4 \rho g . } \end{array}
$$

We compare the backward time increment and the endpoint PDE operator pointwise:

$$
L _ { \tau } : = \frac { \rho _ { \tau } - \rho _ { 0 } } { \tau } , \qquad F _ { \tau } : = \partial _ { x } ( \rho _ { \tau } \partial _ { x } g _ { \tau } ) - 4 \rho _ { \tau } g _ { \tau } , \qquad g _ { \tau } = \log \rho _ { \tau } + V .
$$

Each τ uses an independent one-step update from

$$
\begin{array} { r } { V ( x ) = \frac { 1 } { 2 } \big ( 1 - \cos ( \pi x ) \big ) , \qquad \rho _ { 0 } ( x ) = I _ { 0 } ( 0 . 2 ) ^ { - 1 } \exp ( 0 . 2 \cos ( \pi x ) ) , } \end{array}
$$

where $I _ { 0 }$ is the modified Bessel function. Since $\partial _ { x } g _ { 0 } = 0 . 3 \pi \sin ( \pi x )$ , the smooth positive source satisfies no flux at $x = \pm 1$ . The fitted chemical potential uses the common trainable FCNN with cosine inputs enforcing zero normal derivative; architecture, budget, seed, and grid are fixed across $\tau .$

Spatial derivatives are evaluated by automatic diferentiation in $\mathrm { C P U }$ , using fixed FCNN weights learned on MPS. We report the relative strong residual over the full interval,

$$
\mathcal { R } _ { \tau } ^ { \mathrm { s t r } } = \frac { \Vert L _ { \tau } - F _ { \tau } \Vert _ { L ^ { 2 } ( - 1 , 1 ) } } { \Vert L _ { \tau } \Vert _ { L ^ { 2 } ( - 1 , 1 ) } + \Vert F _ { \tau } \Vert _ { L ^ { 2 } ( - 1 , 1 ) } } .
$$

It includes truncation, projection, and training errors, so Figure 1 is an agreement diagnostic rather than a convergence-rate estimate.

The explicit forward predictor in Remark F.1 has the expansion

$$
T _ { \mathrm { p r e d } } = \mathrm { i d } - \tau \nabla g _ { \rho } + O ( \tau ^ { 2 } ) , \qquad q _ { \mathrm { p r e d } } ^ { 2 } = 1 - 4 \tau g _ { \rho } + O ( \tau ^ { 2 } ) .
$$

We test the optimized pairs against these fields using the same Neumann-compatible $\rho _ { 0 } , V$ as above. Each $\tau$ uses fresh dense map and growth networks, initialized at (id, 1) and optimized in the same pair objective. The predictor fields are used to evaluate the resulting pairs, without a predictor-fitting loss. The entropy quadrature uses the current-source score through the integration-by-parts correction (F.3). Architecture, quadrature, and optimizer budget are fixed across τ (Appendix F.3). Figure 2 and Table 4 report errors over the full interval. Separate spatial and optimizer refinements check that the observed field errors are resolved; this is an empirical one-step consistency test.

![](images/dea10c36b644d75e0f48e01034a01757c06b8fffe1647fdbc56981d4b53e9491.jpg)

![](images/7465962492a38c68320e5866cc54bfd63e4269cdc9d7df1f26b7264f7a86f571.jpg)

![](images/4b127672774dafe1fff22b13e43d93e954e62a57d9ecd244832fa9432d0cbef9.jpg)  
Figure 1: Pointwise one-step PDE consistency under τ refinement. Left and center: $L _ { \tau }$ and $F _ { \tau }$ on a common scale. Right: $\mathcal { R } _ { \tau } ^ { \mathrm { s t r } }$

Small-time scaling of the optimized forward Monge--growth pair  
![](images/5393ac71c968508f23b32be3cd5c3879e3f19465004f0d3aea9f9cb3f2ca9f8c.jpg)  
Figure 2: Small-time pair scaling. Left: pair deviations; center: rescaled-field errors; right: profiles at the smallest τ.

<table><tr><td>T</td><td> $\| T - \operatorname { i d } \| _ { L ^ { 2 } ( \rho ) }$ </td><td> $\| q ^ { 2 } - 1 \| _ { L ^ { 2 } ( \rho ) }$ </td><td>velocity error</td><td>reaction error</td><td>min  $J _ { T }$ </td></tr><tr><td>0.020</td><td>0.014</td><td>0.050</td><td>0.234</td><td>0.146</td><td>0.958</td></tr><tr><td>0.010</td><td>8.169e-03</td><td>0.027</td><td>0.133</td><td>0.081</td><td>0.975</td></tr><tr><td>5.000e-03</td><td>4.370e-03</td><td>0.014</td><td>0.071</td><td>0.043</td><td>0.987</td></tr><tr><td>2.500e-03</td><td>2.265e-03</td><td>7.014e-03</td><td>0.037</td><td>0.022</td><td>0.993</td></tr><tr><td>1.250e-03</td><td>1.153e-03</td><td>3.542e-03</td><td>0.019</td><td>0.011</td><td>0.996</td></tr></table>

Table 4: Small-time pair diagnostics on $[ - 1 , 1 ]$ . Velocity and reaction errors are relative $L ^ { 2 } ( \rho _ { 0 } d x )$ errors of $( T - \operatorname { i d } ) / \tau$ and $( q ^ { 2 } - 1 ) / \tau$ against $- g _ { 0 } ^ { \prime }$ and $- 4 g _ { 0 }$

The relative strong PDE residual decreases from 0.0176 at $\tau = 0 . 0 2$ to 0.000743 at $\tau =$ 0.000625. In the independent pair-scaling test, the smallest step $\tau = 0 . 0 0 1 2 5$ gives velocity and reaction field errors 0.019 and 0.0113, respectively. These finite-resolution observations do not by themselves establish an asymptotic rate.

## 6.2 Entropy–potential relaxation and the role of both pair components

On $[ - 1 , 1 ] ^ { d }$ we take

$$
U ( s ) = s \log s - s , \quad \quad V ( x ) = 2 0 | x | ^ { 2 } ,
$$

$$
\rho _ { 0 } ( x ) = \prod _ { j = 1 } ^ { d } C _ { \beta } \exp \bigl ( - 2 0 x _ { j } ^ { 2 } + \beta ( x _ { j } - x _ { j } ^ { 3 } / 3 ) \bigr ) , \qquad \beta = 4 ,
$$

$$
C _ { \beta } ^ { - 1 } = \frac { 1 } { 2 } \int _ { - 1 } ^ { 1 } \exp \bigl ( - 2 0 s ^ { 2 } + \beta ( s - s ^ { 3 } / 3 ) \bigr ) \ d s .
$$

![](images/eee9d74f9743957985c4464b5be12f28584b55bca02efe0b3b8485d9424cd9c9.jpg)  
Figure 3: One-dimensional relaxation. Step labels count attempts; finite-volume references and diagnostics use the corresponding accepted physical time.

The source has mass $2 ^ { d }$ and satisfies the no-flux condition $\partial _ { j } ( \log \rho _ { 0 } + V ) = \beta ( 1 - x _ { j } ^ { 2 } ) = 0$ on each face. The asymmetric tilt produces directed transport with bounded chemical-potential variation. We compare with a finite-volume trajectory and the equilibrium $\rho _ { \infty } = e ^ { \frac { { \bf \hat { \rho } } } { \tilde { \rho } } 2 0 | x | ^ { 2 } }$ at accepted physical times.

To assess both HK mechanisms, we compare the full pair with pure growth $( T _ { \theta } = \operatorname { i d } )$ and transport only $( q \theta = 1 )$ , reporting trajectory error and transport share alongside energy.

![](images/2a58b072b6a6d731803033fc4f279d99aff35815c11563223dc0b72b3fc48bd8.jpg)  
Figure 4: Full pair versus pure-growth and transport-only restrictions.

<table><tr><td>Configuration</td><td> $\mathcal { F }$ </td><td>energy monotone?</td><td>rel.  $L ^ { 2 }$ </td><td> $\overline { { { A ^ { \mathrm { t r } } } / { A } } }$ </td><td>reference</td><td>common t</td><td>reached t</td></tr><tr><td>full pair  $( T _ { \theta } , q _ { \theta } )$ </td><td>-0.395</td><td>yes</td><td>1.308e-03</td><td>0.150</td><td>0.145</td><td>0.800</td><td>0.800</td></tr><tr><td>pure growth,  $T _ { \theta } = \mathrm { i d }$ </td><td>-0.395</td><td>yes</td><td>0.019</td><td>0</td><td>0.109</td><td>0.800</td><td>0.800</td></tr><tr><td>transport only,  $q _ { \theta } = 1$ </td><td>1.237</td><td>yes</td><td>3.726</td><td>1.000</td><td>0.316</td><td>0.800</td><td>0.800</td></tr></table>

Table 5: Pair-restriction diagnostics at a common attained physical time. Transport shares use the prefix ending at that time. The energy-monotonicity flag uses $E _ { k + 1 } - E _ { k } \leq 1 0 ^ { - 7 } ( 1 + | E _ { k } | )$ a tighter tolerance than the acceptance guard.

In one dimension, the final relative $L ^ { 2 }$ error to the finite-volume trajectory is $2 . 0 1 \times 1 0 ^ { - 5 }$ at $t = 2 . 5$ . In two dimensions, the final relative $L ^ { 2 }$ error to the finite-volume trajectory is 0.000532 at t = 1.2. At the common ablation time, full-pair, pure-growth and transport-only errors are

Difference (neural - reference)  
Difference (neural - reference)  
![](images/b4aae00636fc569977e3f204bb1c6b8ae46aa75ddcea6c914fb93529a1af8f4a.jpg)  
Two-dimensional entropy-potential relaxation

![](images/fd4bceb8eb17f268e47dd87cc1caa46a7ba27bb8bd2ca76d7daf030d3b972133.jpg)

![](images/9e1069be4ef79f4b712cabd0c99a6f9349b2c851abcdd0ad666bd124bc537c3c.jpg)

![](images/3cc356741f29317812fd1efda2a5b031c11fd1a4adf7b286feb3e0ae19d9394a.jpg)

![](images/caeb1a4e2d7729ba523152dad8cab3ae13841ebeedc61d35a05aeba52878c1e7.jpg)

![](images/0b63dd8f0ea1730cd4395e71fd75b40dfdeb5762b52b68230d7b1037292848a6.jpg)

![](images/2fce2235b4b2e07e106b0326a1aeabaa0f698e5104a614988b35200b310347be.jpg)

![](images/2f04058ce90dcaef1da98f29206554ae70d90cc5dfafbc2e9f5729c8be030abe.jpg)

![](images/819bc893f70f725961dc6a69b98dec986dda28f7f63f1342aa67147df6ca3cd0.jpg)

![](images/6fa5b13a06169893ec9f0fe663d1a995896d964eb0756cd4eab716a4d91cdccb.jpg)

![](images/b042d5d730e66dea385512b6dbda9a446183070734f3cfd835e35485d4703a38.jpg)  
Two-dimensional entropy-potential cross-sections and residual

![](images/3579fb1aa99a509becab50e70e7089b07ac3fa1f27b5157029be4020406d3fca.jpg)

![](images/6330f2443202e4aece73424a84e2600aec174858323702f5cb13ad3fe2f404f2.jpg)

![](images/4c83f341ac0a7ecf644ede495ce003e762e0cdeb26125ba2cf78ba07274545cf.jpg)  
Two-dimensional entropy-potential diagnostics

![](images/89eb53f0ef586100afbfdb891b1d0310ef5b8be10c4e59e0368fcb11ac656a48.jpg)

![](images/be5858045a0abf9abd91ee450b11c458012196aece79e613a4904be73186f21f.jpg)

![](images/6a52bfa74e9e736310f1321d7560f813aa83247ccd64b25b8fdef32740afd8eb.jpg)

![](images/e36d5764b486c30a408d0061746ca5946b21fd8993a341172147253d387819b2.jpg)  
Figure 5: Two-dimensional relaxation. Top: reference, neural, and signed error; middle: axial sections; bottom: trajectory diagnostics.

0.00131, 0.0193, 3.73, respectively. The full pair improves trajectory agreement even though endpoint energy alone does not rank these constrained dynamics by accuracy.

## 6.3 Gaussian entropy and confinement flows

For Boltzmann entropy, the HK equation is

$$
\partial _ { t } \rho = \Delta \rho + \boldsymbol { \nabla } \cdot ( \rho \boldsymbol { \nabla } V ) - 4 \rho ( \log \rho + V ) .
$$

We consider $V = 0$ in eight dimensions and $\begin{array} { r } { V ( x ) = \frac { 1 } { 2 } ( x - \mu ) ^ { \top } \Sigma ^ { - 1 } ( x - \mu ) } \end{array}$ in two dimensions. Gaussian initial data admit whole-space Gaussian-ansatz references for density, mass, mean, and covariance. Both learned states use the same positive FCNN as the non-Gaussian and interaction tests. Its latent log density combines quadratic and tanh features, so the approximation family is not restricted to Gaussians. These cases reproduce Gaussian dynamics within this larger approximation family; finite-box errors also include truncation.

Figure 6 shows entropy flow on $[ - 3 . 5 , 3 . 5 ] ^ { 8 }$ , with unshown coordinates fixed at zero. It uses full-input dense maps, learned neural growth, and Gaussian importance proposals whose moments are estimated from the current neural density; Table 9 checks density and mass errors at fixed $T = 0 . 1 5$ under time refinement. Figure 7 shows confinement and mean motion. Initial data and numerical settings are specified in Appendix F.3. FP2D uses distinct deterministic Gauss–Legendre rules; enriched checks assess integration error but do not establish optimizer convergence.

Eight-dimensional HK entropy flow at t= 0.150: coordinate slices

![](images/e68c36d110753b77dd7abdc20a4f0218e9f6658e8d661afaf8cfcda97e29bd65.jpg)

![](images/274b4c9d4ad66aa9d40e18e721e846600ea6c481419b285d69a7dc18430f8388.jpg)

![](images/8e54710bbf7aa42ad770b9c4c3ce35b192e1657cd2ee3a8a902ec27d9fa2d701.jpg)

![](images/db3f79940450302f803daf4475cf45178223e479db260580d3198b1e8aa08b4f.jpg)

![](images/6051eb309df4f79cbed843447d968cc0a9b1e69f522f2949807c6ba992042319.jpg)

![](images/a97c7c693b85d8e79b481be55c85d2def25dd884414b2b68516750b879122f88.jpg)

![](images/06fbc04ec65614c7c6e506b378fdd65cd3b8fc56d93a8365ee3b26c3117b6353.jpg)

![](images/c40c528831589b19028346db25412e69a9fd8e4b927ac3db87519a158b2c514c.jpg)

![](images/39349d342b3341299929006b39e96e6709f487f36db876a336ac83536544fe7d.jpg)

![](images/cbfab07472380b602187061ebc112640276aeb717cc1e8642754a2c25f064a60.jpg)

![](images/7fa4aca09b128c78e9a1f284afc56e1b50c31a88ef98819dc2e9621ff9f1988b.jpg)

![](images/7391e3f3b55e61944895fcc41d7393d15acb1e0f1b9e492c9be187ad1e67ba39.jpg)

![](images/379a406761053267174520a0d00221fab4a6aad6d8412670621cae27fa4f86da.jpg)

![](images/019159deca32ea85ea264db22c7b9f4cdbe79b3595f7de3cb1350ff19be1d566.jpg)

![](images/fcc2eddcbfee50362d8ffa11e7eb9155c7e91c06f563ab7efa5cfef29a7ecdfd.jpg)  
Figure 6: Eight-dimensional Gaussian entropy flow. Top: final coordinate sections; bottom: mass, energy, and error to the whole-space Gaussian reference.

For eight-dimensional entropy flow, the recorded endpoint relative $L ^ { 2 }$ density error is 0.00803 at $t = 0 . 1 5$ . For FP2D, the recorded endpoint relative $L ^ { 2 }$ density error is 0.00348 at $t = 1 . 2$ These values use the reported error quadratures and whole-space Gaussian references; they include time discretization, finite-box, learned-pair and density-fit errors.

![](images/1dc1bf80e4f0405b6a924f11a58927b8ff1da5763dd90b9cd24892c22c2a89db.jpg)  
Figure 7: Two-dimensional confinement test: density and mean-path comparisons with mass, energy, density error, and mean error.

## 6.4 Coercive fully implicit interaction benchmark

On the flat two-dimensional torus we use

$$
W ( x , y ) = \frac { 1 } { 2 } - \frac { 1 } { 2 } \exp \left( - \frac { | x - y | ^ { 2 } } { 2 ( 0 . 3 5 ) ^ { 2 } } \right) ,
$$

with $V ( x ) = 0 . 4 \substack { + 0 . 5 ( 2 - \cos ( \pi x _ { 1 } ) - \cos ( \pi x _ { 2 } ) ) }$ . Each coordinate of $x - y$ is reduced to $[ - 1 , 1 )$ , giving the same periodic kernel in the neural objective and the FFT reference. The shift preserves $\nabla _ { x } W$ and ensures coercivity, but changes the reaction term through $W [ \rho ]$ . We compare the fully implicit field $W [ \rho ]$ with a frozen preceding-state field, using full deterministic pair quadrature and enriched validation grids. The common initial density is specified in Appendix F.3.

Table 6 compares endpoint density errors and full-periodic PDE residuals, applying the same operator to the independent reference. Fixed-horizon time refinement and separate spatial and optimizer checks are given in Appendix F.3. This comparison does not establish a uniform accuracy advantage for either treatment. The strong-form residual is a grid-dependent finitediference diagnostic; the refinement checks do not establish spatial convergence.
<table><tr><td>interaction treatment</td><td>rel. L2</td><td>mass</td><td>true energy</td><td>transport share</td><td>residual (neural/ref.)</td><td>common t</td><td>reached t</td><td>null</td><td>fallback</td></tr><tr><td>fully implicit</td><td>1.312e-03</td><td>0.844</td><td>-0.940</td><td>0.050</td><td>0.021/4.787e-03</td><td>0.300</td><td>0.300</td><td>0</td><td>0</td></tr><tr><td>frozen field</td><td>1.133e-03</td><td>0.844</td><td>-0.940</td><td>0.051</td><td>0.021/4.787e-03</td><td>0.300</td><td>0.300</td><td>0</td><td>0</td></tr></table>

Table 6: Implicit/frozen comparison at the common attained physical time. Density errors use the bilinearly interpolated $6 4 ^ { 2 }$ reference; residuals use the same full-periodic operator for both trajectories on the refined $1 2 8 ^ { 2 }$ grid. Because the reference is advanced using the same spatial operator, its residual primarily reflects temporal discretization. Null and fallback counts cover all attempted steps; the transport share is that of the last update reaching the common time.

Coercive entropy--interaction flow with $W ( z ) = 0 . 5 - 0 . 5 \ : \mathrm { e } ^ { - \vert z \vert ^ { 2 } / ( 2 ( 0 . 3 5 ) ^ { 2 } ) } ;$ fully implicit versus frozen field  
![](images/3517e299c1e1f9606f62890caee6dc0e2e873b1fda45ce77cdcd7ce6ca9fae19.jpg)

![](images/eb444246241597e037ddd28bf7b1aeb8cce23b18d11ffc924ff5d37cb4bec5a1.jpg)

![](images/dde384244dfe1af968f4cb0f4e5ed2c8cf4c0af40aa79e28bee083070c03f6b6.jpg)

![](images/e17a8f358adff976c7768da2f185a0f6d7817ca1e14181d75fac26415a94c69a.jpg)

![](images/556229722cc528eb9cf40877dd3f3c9b94d9fabe97eb666986eb61894edcd1f7.jpg)

![](images/75959aea124822936f5b78f7ed9882bf0e84cac629edc84fc8053bd439f1ee76.jpg)

![](images/83b0660c4da92f6cd475a9e86cab827059f505f8d34cd4a05e1a1d6d1df95580.jpg)

![](images/dcdcad93efaa417fb9421819145c64e5970627e142df98818cb49755f97fbb59.jpg)

![](images/6acb3c90d5b38291d7d078ba2583aefb6187c37a57b7950f62c93abb365cb681.jpg)

![](images/34e987faf6e5d070c373694d181ccb6de9bb37dabb77d6ac632e6f05820b7394.jpg)

![](images/d1dc197111f902db4c0d1466dfb7fee06f671c04ca932fa7ce81ccad287587ed.jpg)

![](images/5d677ab323a1ab48825fe24a2bb2c47604e8e854f9a824a6896ee53778b05a9d.jpg)  
Figure 8: Interaction benchmark: periodic reference, fully implicit update, and frozen-field ablation at matched times.

Coercive interaction benchmark: implicit and frozen-field JKO steps (requested t=0.3)  
![](images/466d3f3699aa6f54b9e6d3aad5592c4c3fd0c50d5f7a196116377fda6bb87144.jpg)

![](images/3176978685a654aea7c65dc1464c6f29eb78481aba1921c4347817a262103318.jpg)

![](images/6fe5334bb4826b23bc99b1419829f7bee849dd307653ae5b5ad6b03b4ad9bd9b.jpg)

![](images/93cfdf00277ed2d187928976a387da5481259439425ce6c548ccc43a0ce49e68.jpg)

![](images/1089a202fe99d65c946e0e39814bbd18de4bd3983e47bd41dc9d71490e68616e.jpg)

![](images/a41db40161b09caf5569f8850dbc902da0901689e38f84a863f16baaeddeba0a.jpg)  
Figure 9: Interaction diagnostics: mass, common fully implicit energy, trajectory error, transport share, and guard margin.

## 6.5 Finite-grid primal–dual certificate

For a one-dimensional Boltzmann step with $W = 0$ , we evaluate the optimized pair’s density $u _ { \theta }$ by exact map inversion and change of variables. The discrete candidate consists of its values on the fixed 65-node grid; source, potential, and pair solver are specified in Appendix F.3. A nonnegative entropy–transport coupling gives a primal upper value for this same candidate. A grid dual pair is shifted until

$$
\zeta _ { i } + \xi _ { j } \leq c _ { \mathsf { H K } } ( x _ { i } , x _ { j } ) \quad \mathrm { f o r ~ e v e r y ~ g r i d ~ p a i r } \ ( i , j ) ,
$$

and hence gives a certified lower value. The gap $R _ { h }$ and a rigorous mass upper bound yield the discrete analogue of Corollary 5.5. The radius applies to the exact finite-grid minimizer; the comparison with a numerical grid iterate is diagnostic only, and its own primal–dual gap is reported separately. No continuum certificate is claimed. The discrete bounds are evaluated in floating-point arithmetic with a numerical feasibility margin; they are not interval-arithmetic proofs of the reported decimal values.

![](images/ced4e4430452411e63b00d727ebf7e6666c6be965a964d3e65309c35537e226f.jpg)  
Figure 10: Finite-grid certificate across network widths, with depth fixed. Left: primal and dual values; center: gap, observed error, and certified radius; right: directly evaluated $u _ { \theta }$ and the numerical grid reference.

<table><tr><td>width</td><td>primal upper</td><td>dual lower</td><td>gap  $R _ { h }$ </td><td>observed  $L ^ { 1 }$ </td><td>certified radius</td><td>metric gap  $/ 2 \tau$ </td><td>feasibility violation</td></tr><tr><td>8</td><td>-0.627</td><td>-0.627</td><td>9.783e-05</td><td>8.534e-03</td><td>0.022</td><td>1.261e-09</td><td>-1.000e-10</td></tr><tr><td>16</td><td>-0.627</td><td>-0.627</td><td>9.330e-05</td><td>8.159e-03</td><td>0.022</td><td>1.378e-10</td><td>-1.000e-10</td></tr><tr><td>32</td><td>-0.627</td><td>-0.627</td><td>9.346e-05</td><td>8.371e-03</td><td>0.022</td><td>5.154e-10</td><td>-1.000e-10</td></tr><tr><td>64</td><td>-0.627</td><td>-0.627</td><td>9.360e-05</td><td>8.370e-03</td><td>0.022</td><td>1.779e-10</td><td>-1.000e-10</td></tr><tr><td>128</td><td>-0.627</td><td>-0.627</td><td>9.208e-05</td><td>8.485e-03</td><td>0.022</td><td>2.226e-10</td><td>-1.000e-10</td></tr></table>

Table 7: Finite-grid certificate. Feasibility is checked over all grid pairs. The scaled metric gap measures coupling-solve accuracy with the candidate density fixed.

Across the tested widths, the largest relative growth-equation RMS and maximum residuals are $9 . 0 2 \times 1 0 ^ { - 5 }$ and 0.000583. The observed $L ^ { 1 }$ distance to the numerical grid iterate is at most 0.00853, while the certified radii range from 0.0216 to 0.0223. The radii remain valid across all widths; the measured errors and gaps need not decrease monotonically with width.

For the numerical grid reference itself, the primal upper value is −0.6267, the feasible dual lower value is −0.6267, and the gap is $2 . 7 4 6 \times 1 0 ^ { - 9 }$ . The same discrete error estimate gives it an $L ^ { 1 }$ radius 0.000118. These computed bounds, rather than the optimizer iteration budget alone, describe reference accuracy. Dual feasibility is checked with an outward numerical margin in floating-point arithmetic.

## 7 Discussion and conclusion

We developed a Monge-growth neural minimizing-movement scheme for HK dynamics with fully implicit pair interactions. At each step, a fresh spatial map and growth factor act on the current measure through a single-level objective. The key design choice is the one-sided primal approximation. Since the cone action majorizes $\mathsf { H K } ^ { 2 }$ and vanishes at the identity pair, every admissible pair satisfying the descent criterion has an exact, unprojected endpoint obeying the discrete energy inequality; the computed density fit is checked separately. The criterion with $\delta = 0$ is compatible with $T = \mathrm { i d }$ , so the transport–reaction split is needed to detect reaction-only degeneration.

For the exact step, Theorems 2.3 and 2.6 establish existence, mass control, positivity, density bounds, the discrete Euler–Lagrange equation, and the metric–dissipation identity on bounded convex domains and flat tori. With Lipschitz V and W, the density is Lipschitz as well. At the parametric level, Theorem 5.2 gives one-step recovery under a regular-pair approximation hypothesis; Corollary D.2 extends this recovery to each fixed finite horizon by finite induction. These are conditional representability results and do not imply convergence of the trained stochastic solver. In particular, approximation by the fixed two-block architecture used in the experiments is not proved.

Under a positive-semidefinite interaction, Theorem 5.4 complements the descent criterion with a one-step state-error estimate. A globally feasible dual pair supplies a rigorous lower bound for the exact JKO minimum. Combined with the primal cone-action upper bound, it controls the JKO-objective gap and the convexity remainder of U, with an explicit $L ^ { 1 }$ bound for Boltzmann entropy. Remark 5.7 records the limitations of this estimate and the additional ingredients needed for finite-horizon propagation. Section 6 checks the small-step pair structure, evaluates a deterministic-grid version of the primal–dual certificate, and tests the transport– reaction and fully implicit design choices by ablation. The 8D example reproduces Gaussian dynamics; projection errors, optimizer work, and interaction residual checks are reported in Appendix F.3. These experiments do not replace the missing continuum quadrature control or step-uniform estimates. A general $\tau \downarrow 0$ theory would additionally require stepwise estimates uniform in the step index.

## Acknowledgements

Geuntaek Seo was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government(MSIT) (RS-2023-00219980). Hwijae Son was supported by National Research Foundation of Korea (NRF) grants funded by the Korean government (Ministry of Science and ICT, MSIT) (No. RS-2026-25491905 and RS-2026-25506577). Hyung Ju Hwang was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (RS-2023-00219980 and RS-2022-00165268).

## A Proof of Theorem 2.3 (Theorem A)

Proof of Theorem 2.3. We prove existence, the quantitative mass estimate, strict positivity in this order.

Step 1: a coercive lower bound depending only on the mass. The zero measure is admissible and

$$
\mathcal { I } _ { \tau } ( 0 \mid \bar { \mu } ) = | \Omega | U ( 0 ) + \frac { \bar { m } } { 2 \tau } , \qquad \bar { m } : = \bar { \mu } ( \Omega ) ,\tag{A.1}
$$

because ${ \sf H } ^ { 2 } ( 0 , \bar { \mu } ) = \bar { m }$ . Hence the infimum of the one-step functional is finite. Moreover, every competitor with finite energy is of the form $\sigma = r d x$ . Writing $\begin{array} { r } { m = \sigma ( \Omega ) = \int _ { \Omega } } \end{array}$ r dx, Jensen’s inequality gives

$$
\int _ { \Omega } { U ( r ) d x } \geq | \Omega | U { \Biggr ( } { \frac { m } { | \Omega | } } { \Biggr ) } .
$$

Since $V \geq \underline { { V } }$ on Ω and $W \geq \underline { W }$ on $\Omega \times \Omega$

$$
\int _ { \Omega } V d \sigma \geq \underline { { { V } } } m , \qquad \frac 1 2 \int \int _ { \Omega \times \Omega } W ( x , y ) d \sigma ( x ) d \sigma ( y ) \geq \frac { W } { 2 } m ^ { 2 } .
$$

Finally, the mass estimate (2.6) yields

$$
{ \sf H } ^ { 2 } ( \sigma , \bar { \mu } ) \geq ( \sqrt { m } - \sqrt { \bar { m } } ) ^ { 2 } .
$$

Combining the preceding inequalities, we obtain the scalar lower bound

$$
{ \mathcal { I } } _ { \tau } ( \sigma \mid { \bar { \mu } } ) \geq \Gamma _ { \tau , { \bar { m } } } ( m ) .\tag{A.2}
$$

This estimate is the point at which the possibly negative quadratic part of the interaction is taken into account. The assumed coercivity of $\Gamma _ { \tau , \bar { m } }$ therefore controls the total mass even when the interaction energy itself is not nonnegative.

Step 2: compactness of a minimizing sequence. Let $\sigma _ { n } = r _ { n }$ dx be a minimizing sequence. By discarding finitely many terms, we may assume

$$
\mathcal { T } _ { \tau } ( \sigma _ { n } \mid \bar { \mu } ) \leq \mathcal { I } _ { \tau } ( 0 \mid \bar { \mu } ) + 1 .
$$

If $m _ { n } = \sigma _ { n } ( \Omega )$ , then (A.2) and the coercivity of $\Gamma _ { \tau , \bar { m } }$ imply

$$
\operatorname* { s u p } _ { n } m _ { n } < \infty .\tag{A.3}
$$

The potential and interaction terms are bounded from below on this mass-bounded family:

$$
\int _ { \Omega } V r _ { n } d x \geq \underline { { V } } m _ { n } , \qquad \frac 1 2 \iint W ( x , y ) r _ { n } ( x ) r _ { n } ( y ) d x d y \geq \frac { W } { 2 } m _ { n } ^ { 2 } .
$$

Since the proximal term is nonnegative and the objective values are bounded from above, it follows that

$$
\operatorname* { s u p } _ { n } \int _ { \Omega } U ( r _ { n } ) d x < \infty .\tag{A.4}
$$

We spell out why this gives compactness of the densities. Convexity and finiteness of U provide an afine lower bound $U ( s ) \geq - a s - b$ on $[ 0 , \infty )$ for suitable $a , b \geq 0$ . Thus

$$
\Psi ( s ) : = U ( s ) + a s + b
$$

is nonnegative and remains superlinear. From (A.3) and (A.4),

$$
\operatorname* { s u p } _ { n } \int _ { \Omega } \Psi ( r _ { n } ) d x < \infty .
$$

The de la Vall´ee-Poussin criterion shows that $( r _ { n } ) _ { n }$ is uniformly integrable. By the Dunford– Pettis theorem, after passing to a subsequence, there is $r \in L ^ { 1 } ( \Omega )$ such that

$$
r _ { n }  r \qquad \mathrm { w e a k l y ~ i n ~ } L ^ { 1 } ( \Omega ) .\tag{A.5}
$$

The positive cone of $L ^ { 1 }$ is weakly closed, so $r \geq 0$ . Setting $\sigma = r d x , \ ( \mathrm { A . 5 } )$ also implies the narrow convergence $\sigma _ { n }  \sigma$ in $\mathcal { M } _ { + } ( \Omega )$ . In particular, the weak $L ^ { 1 }$ argument rules out the creation of a singular part in the limit; this is the compactness role played by the superlinearity of U.

Step 3: passage to the limit. The convex integral functional is weakly lower semicontinuous on $L ^ { 1 }$ , hence

$$
\int _ { \Omega } U ( r ) d x \leq \operatorname* { l i m } _ { n \to \infty } \operatorname* { i n f } _ { \Omega } U ( r _ { n } ) d x .\tag{A.6}
$$

Because $V \in C ( \Omega ) \subset L ^ { \infty } ( \Omega )$

$$
\int _ { \Omega } V r _ { n } d x \longrightarrow \int _ { \Omega } V r d x .
$$

For the interaction term, narrow convergence together with the uniform mass bound implies

$$
\sigma _ { n } \otimes \sigma _ { n } \longrightarrow \sigma \otimes \sigma \qquad \mathrm { n a r r o w l y ~ i n ~ } \mathcal { M } _ { + } ( \Omega \times \Omega ) .
$$

Indeed, the convergence is immediate for functions of the form $f ( x ) g ( y )$ , and finite sums of such functions are uniformly dense in $C ( \Omega \times \Omega )$ . Since W is continuous and bounded on $\Omega \times \Omega$

$$
{ \frac { 1 } { 2 } } \int \int W ( x , y ) d \sigma _ { n } ( x ) d \sigma _ { n } ( y ) \longrightarrow { \frac { 1 } { 2 } } \int \int W ( x , y ) d \sigma ( x ) d \sigma ( y ) .\tag{A.7}
$$

The squared Hellinger–Kantorovich distance is lower semicontinuous under narrow convergence of finite measures; this follows, for instance, from its entropy–transport formulation (2.1); see [7]. Consequently,

$$
{ \sf H } ^ { 2 } ( \sigma , \bar { \mu } ) \leq \operatorname* { l i m } _ { n  \infty } { \sf H } ^ { 2 } ( \sigma _ { n } , \bar { \mu } ) .
$$

Together with (A.6) and (A.7), this gives

$$
{ \mathcal { T } } _ { \tau } ( \sigma \mid { \bar { \mu } } ) \leq \operatorname* { l i m i n f } _ { n  \infty } { \mathcal { T } } _ { \tau } ( \sigma _ { n } \mid { \bar { \mu } } ) .
$$

Thus $\sigma = \rho _ { 1 }$ dx is a minimizer.

Step 4: the quantitative mass bound. Let $\mu _ { 1 } = \rho _ { 1 } d x$ be any minimizer and put $m _ { 1 } =$ $\mu _ { 1 } ( \Omega )$ . Minimality against the zero measure, (A.1), and (A.2) give

$$
\Gamma _ { \tau , \bar { m } } ( m _ { 1 } ) \leq \mathcal { I } _ { \tau } ( \mu _ { 1 } \mid \bar { \mu } ) \leq | \Omega | U ( 0 ) + \frac { \bar { m } } { 2 \tau } .
$$

The scalar sublevel appearing in (2.8) is nonempty because it contains $m = 0$ , and it is bounded by the coercivity of $\Gamma _ { \tau , \bar { m } }$ . Hence $M _ { \tau , \bar { \mu } } < \infty$ and $m _ { 1 } \leq M _ { \tau , \bar { \mu } }$

Step 5: strict positivity when $U ^ { \prime } ( 0 + ) = - \infty$ . Assume now that $U ^ { \prime } ( 0 + ) = - \infty$ and suppose, by contradiction, that $B : = \{ x \in \Omega : \rho _ { 1 } ( x ) = 0 \}$ has positive Lebesgue measure.

We first estimate the proximal term. For $\delta > 0$ , choose an entropy–transport plan $\gamma$ for $( \mu _ { 1 } , \bar { \mu } )$ , with marginals $\gamma _ { 0 } , \gamma _ { 1 }$ , such that

$$
\int _ { \Omega } F \bigg ( \frac { d \gamma _ { 0 } } { d \mu _ { 1 } } \bigg ) \ : d \mu _ { 1 } + \int _ { \Omega } F \bigg ( \frac { d \gamma _ { 1 } } { d \bar { \mu } } \bigg ) \ : d \bar { \mu } + \int _ { \Omega \times \Omega } c _ { \mathsf { H K } } \ : d \gamma \leq \mathsf { H K } ^ { 2 } ( \mu _ { 1 } , \bar { \mu } ) + \delta .\tag{A.8}
$$

Since $\gamma _ { 0 } \ll \mu _ { 1 }$ and $\mu _ { 1 } ( B ) = 0$ , we have $\gamma _ { 0 } ( B ) = 0$ . For $\varepsilon > 0$ , if we define

$$
\mu _ { \varepsilon } : = \mu _ { 1 } + \varepsilon \mathbb { 1 } _ { B } d x ,
$$

then $\mu _ { 1 } \ll \mu _ { \varepsilon }$ , and hence $\gamma _ { 0 } \ll \mu _ { \varepsilon }$ . Thus the same plan $\gamma$ is admissible for $( \mu _ { \varepsilon } , \bar { \mu } )$ , and

$$
\frac { d \gamma _ { 0 } } { d \mu _ { \varepsilon } } = \frac { d \gamma _ { 0 } } { d \mu _ { 1 } } \quad \mathrm { o n ~ } \Omega \setminus B , \qquad \frac { d \gamma _ { 0 } } { d \mu _ { \varepsilon } } = 0 \quad \mathrm { o n ~ } B .
$$

Therefore

$$
\int _ { \Omega } F \left( \frac { d \gamma _ { 0 } } { d \mu _ { \varepsilon } } \right) d \mu _ { \varepsilon } = \int _ { \Omega \setminus B } F \left( \frac { d \gamma _ { 0 } } { d \mu _ { 1 } } \right) d \mu _ { 1 } + \int _ { B } F ( 0 ) d \mu _ { \varepsilon } = \int _ { \Omega } F \left( \frac { d \gamma _ { 0 } } { d \mu _ { 1 } } \right) d \mu _ { 1 } + \varepsilon | B | ,\tag{A.9}
$$

where we used $F ( 0 ) = 1$ . The second entropy term and the transport-cost term are unchanged. Since ${ \sf H } ^ { 2 } ( \mu _ { \varepsilon } , \bar { \mu } )$ is the infimum over all admissible entropy–transport plans, using this same $\gamma$ as a competitor gives

$$
\mathsf { H } ^ { 2 } ( \mu _ { \varepsilon } , \bar { \mu } ) \leq \int _ { \Omega } F \left( \frac { d \gamma _ { 0 } } { d \mu _ { \varepsilon } } \right) d \mu _ { \varepsilon } + \int _ { \Omega } F \left( \frac { d \gamma _ { 1 } } { d \bar { \mu } } \right) d \bar { \mu } + \int _ { \Omega \times \Omega } c _ { \mathsf { W } } d \gamma .
$$

By combining (A.8) and (A.9), and letting $\delta \downarrow 0$ , we obtain

$$
{ \sf H } ^ { 2 } ( \mu _ { \varepsilon } , \bar { \mu } ) \leq { \sf H } ^ { 2 } ( \mu _ { 1 } , \bar { \mu } ) + \varepsilon | B | .\tag{A.10}
$$

Since W is symmetric, the interaction increment has the exact expansion

$$
\begin{array} { c } { { \displaystyle { \frac { 1 } { 2 } \int \int W ( x , y ) d \mu _ { \varepsilon } ( x ) d \mu _ { \varepsilon } ( y ) - \frac { 1 } { 2 } \int \int W ( x , y ) d \mu _ { 1 } ( x ) d \mu _ { 1 } ( y ) } } } \\ { { \displaystyle { = \varepsilon \int _ { B } W [ \mu _ { 1 } ] ( x ) d x + \frac { \varepsilon ^ { 2 } } { 2 } \iint _ { B \times B } W ( x , y ) d x d y . } } } \end{array}
$$

Using minimality of $\mu _ { 1 }$ and (A.10), we obtain

$$
\begin{array} { l } { \displaystyle 0 \le \mathcal { T } _ { \tau } ( \mu _ { \varepsilon } \mid \bar { \mu } ) - \mathcal { T } _ { \tau } ( \mu _ { 1 } \mid \bar { \mu } ) } \\ { \displaystyle \quad \le \vert B \vert \big ( U ( \varepsilon ) - U ( 0 ) \big ) + \varepsilon \int _ { B } \big ( V + W [ \mu _ { 1 } ] \big ) d x + \frac { \varepsilon ^ { 2 } } { 2 } \iint _ { B \times B } W ( x , y ) d x d y + \frac { \varepsilon \vert B \vert } { 2 \tau } . } \end{array}\tag{A.11}
$$

Because $V , W$ are bounded and $\mu _ { 1 }$ has finite mass, all terms on the right-hand side except the internal-energy increment are bounded from above by $C \varepsilon | B | + C \varepsilon ^ { 2 } | B | ^ { 2 }$ , with C independent of ε. Dividing (A.11) by $\varepsilon | B |$ gives

$$
0 \leq \frac { U ( \varepsilon ) - U ( 0 ) } { \varepsilon } + C + C \varepsilon | B | .
$$

The right-hand side tends to $- \infty \mathrm { \ a s \ } \varepsilon \downarrow 0$ because $U ^ { \prime } ( 0 + ) = - \infty$ , which is impossible. Thus $\rho _ { 1 } > 0$ almost everywhere. Every nonempty relatively open subset of Ω has positive Lebesgue measure, so it has positive $\mu _ { 1 } { \mathrm { - m a s s } } ;$ hence supp $\mu _ { 1 } = \Omega$ □

Proof of Corollary 2.4. We only need a coercive estimate in the total mass. We obtain it by retaining the afine lower bound for the internal energy and the negative part of the external potential, and then compare the minimizer with the zero measure.

Let $\sigma = r$ dx have mass $m$ . Under the assumptions of the corollary,

$$
\int _ { \Omega } U ( r ) d x \geq a m - C _ { a } | \Omega | , \qquad \int _ { \Omega } V d \sigma \geq - \| V _ { - } \| _ { \infty } m ,
$$

while the interaction and proximal terms are nonnegative. Therefore

$$
\mathcal { I } _ { \tau } ( \sigma \mid \bar { \mu } ) \geq ( a - \| V _ { - } \| _ { \infty } ) m - C _ { a } | \Omega | .
$$

For a minimizer $\mu _ { 1 }$ , comparison with the zero measure gives

$$
( a - \| V _ { - } \| _ { \infty } ) \mu _ { 1 } ( \Omega ) - C _ { a } | \Omega | \leq | \Omega | U ( 0 ) + \frac { \bar { m } } { 2 \tau } .
$$

Since $a > \| V _ { - } \| _ { \infty }$ , rearranging proves (2.9).

## B Proof of Proposition 2.5

Proof of Proposition 2.5. We treat the two state spaces separately. In the Euclidean case the cited Monge theorem applies directly after extension by zero; on the torus we use the logarithmic-potential formulation and then verify essential injectivity from the reversed optimizer.

In the Euclidean case, extend the measures by zero outside Ω. Since both densities are positive almost everywhere, both supports equal Ω. Thus each support lies in the $\pi / 2 \cdot$ -neighborhood of the other, so the pair is strongly reduced in the sense of [11, Definition 2.6]; equivalently, there is no non-reduced marginal part to discard. Theorem $3 . 3 ( 4 ) { - } ( 5 )$ and Corollary 3.5 of [11] then give the Monge-growth representation, positivity of the growth factor, and essential injectivity. Convexity keeps every selected segment in Ω.

We next turn to the flat torus, where the same conclusion is obtained through the Riemannian entropy–transport theory.

Now let $\Omega =  { \mathbb { T } } ^ { d }$ and regard it as a compact flat Riemannian manifold. The two measures have full support and form an admissible pair for the HK cost, since $c _ { \mathsf { H K } } ( x , x ) = 0$ . The Lipschitzpotential and Monge results of [12, Sections 2.2 and 3.4.1] give finite Lipschitz logarithmic potentials $\left( z _ { 0 } , z _ { 1 } \right)$ and a unique entropy–transport optimizer

$$
\gamma = ( \mathrm { i d } , T ) _ { \# } \gamma _ { 0 } , \qquad \gamma _ { 0 } = e ^ { - z _ { 0 } } \mu _ { 0 } , \qquad \gamma _ { 1 } = e ^ { - z _ { 1 } } \mu _ { 1 } ,
$$

concentrated on the graph of the Riemannian c-exponential map

$$
T ( x ) = \mathrm { e x p } _ { x } \bigg ( - \arctan \bigg ( \frac { | \nabla z _ { 0 } ( x ) | } { 2 } \bigg ) \frac { \nabla z _ { 0 } ( x ) } { | \nabla z _ { 0 } ( x ) | } \bigg )
$$

for $\mu _ { 0 }$ -almost every diferentiability point, with the usual convention when $\nabla z _ { 0 } = 0$ . The corresponding growth factor is

$$
q ( x ) = e ^ { - z _ { 0 } ( x ) } \left( 1 + \frac { 1 } { 4 } | \nabla z _ { 0 } ( x ) | ^ { 2 } \right) ^ { 1 / 2 } > 0 ,
$$

and the Monge formulation gives both identities in (2.10).

It remains to verify essential injectivity of the torus map. For this we compare the forward optimizer with the optimizer of the reversed problem.

Apply the same result to the reversed pair $( \mu _ { 1 } , \mu _ { 0 } )$ and denote its Monge map by $S ,$ so that the reversed optimizer is $( \operatorname { i d } , S ) _ { \# \gamma _ { 1 } }$ . The entropy–transport cost is symmetric under transposition, so $\gamma ^ { \mathsf { T } }$ is optimal for the reversed pair; by its uniqueness, $\gamma ^ { \mathsf { T } } = ( \mathrm { i d } , S ) _ { \# } \gamma _ { 1 }$ . A plan that is concentrated both on the graph of $T$ read forwards and on the graph of $S$ read backwards satisfies $S ( T ( x ) ) = x$ for γ -almost every $x ,$ so $T$ is injective of a γ -null set. Since $z _ { \mathrm { 0 } }$ is finite on the compact torus, $e ^ { - z _ { 0 } }$ is bounded away from zero and $\gamma _ { 0 }$ and $\mu _ { 0 }$ are equivalent; therefore $T$ is essentially injective with respect to $\mu _ { 0 }$ as well. Interchanging the two measures gives the assertion for either ordering. All distances, gradients, and exponential maps in this case are understood in the flat Riemannian sense. □

## C Proof of Theorem 2.6 (Theorem B)

Notation used in this appendix. We write the source as $\mu _ { 0 } ~ = ~ \rho _ { 0 }$ dx with $c _ { \operatorname* { m i n } } ~ = ~ c _ { 0 }$ and $c _ { \mathrm { m a x } } = C _ { 0 }$ The symbol $\Phi _ { 1 }$ denotes the chemical potential (C.1), while $\phi _ { 1 }$ denotes the dual potential of Theorem $2 . 6 ( \mathrm { i i i } )$ ; the relation $\phi _ { 1 } = - \tau \Phi _ { 1 }$ is proved below, not assumed in the density-bound argument. The first subsection uses the forward pair $\mu _ { 0 } ~ \to ~ \mu _ { 1 }$ , whereas Subsection C.2 uses the reversed pair $\mu _ { 1 } \to \mu _ { 0 } ;$ the symbol $T$ is reused for these diferent maps. Subsection C.2 is moreover stated for a generic ordered pair $( \mu , \nu )$ . In its application to a JKO step, $\mu$ is the minimizer (the new state) and $\nu$ is the reference measure (the preceding state).

## C.1 Monotonicity of the discrete chemical potential along the optimal map and density bounds

Throughout this subsection the pair $( T , q )$ is oriented forward, from the source to the new state. Let Ω be either of the two state spaces in Theorem 2.6, and let

$$
\mu _ { 0 } = \rho _ { 0 } d x , \qquad 0 < c _ { \operatorname* { m i n } } \leq \rho _ { 0 } \leq c _ { \operatorname* { m a x } } < \infty ,
$$

and let $\mu _ { 1 } = \rho _ { 1 } d x = S _ { \tau } ( \mu _ { 0 } )$ . By Theorem 2.3, $\rho _ { 1 } > 0$ almost everywhere. By Proposition 2.5, let $( T , q )$ be a positive, essentially injective optimal Monge-growth pair satisfying (2.10). Choose Borel representatives of the self-consistent field and density and define the chemical potential by

$$
P _ { 1 } = P [ \mu _ { 1 } ] = V + W [ \mu _ { 1 } ] , \qquad \Phi _ { 1 } ( x ) = U ^ { \prime } ( \rho _ { 1 } ( x ) ) + P _ { 1 } ( x ) .\tag{C.1}
$$

All pointwise relations in this subsection are understood for these representatives after null-set modifications. Since $q > 0$ almost everywhere and $T _ { \# } ( q ^ { 2 } \mu _ { 0 } ) = \mu _ { 1 }$ , these modifications do not afect the asserted µ<sub>0</sub>-almost-everywhere statements.

Lemma C.1 (Restriction identity for an essentially injective Monge pair). After modifying $T$ on a $\mu _ { 0 } - n u l l$ set, there is a Borel set $G \subset \Omega$ of full µ<sub>0</sub>-measure on which $T$ is injective. For every Borel set $A \subset G$ 2

$$
T _ { \# } ( q ^ { 2 } \mu _ { 0 } | _ { A } ) = \mu _ { 1 } | _ { T ( A ) } , \qquad \mu _ { 1 } ( T ( A ) ) = \int _ { A } q ^ { 2 } d \mu _ { 0 } .
$$

The same identities hold after completing the measures.

Proof. We show that restricting the source of an essentially injective Monge pair restricts the target to the corresponding image set, with no additional mass coming from outside.

Choose a Borel full-measure set on which the Borel representative of $T$ is injective. By the Lusin–Souslin theorem, $T ( A )$ is Borel for every Borel A in that set. For a Borel set $B$

$$
T _ { \# } ( q ^ { 2 } \mu _ { 0 } | _ { A } ) ( B ) = \int _ { A \cap T ^ { - 1 } ( B ) } q ^ { 2 } d \mu _ { 0 } = \int _ { G \cap T ^ { - 1 } ( B \cap T ( A ) ) } q ^ { 2 } d \mu _ { 0 } = \mu _ { 1 } ( B \cap T ( A ) ) ,
$$

where injectivity on G and $\mu _ { 0 } ( \Omega \setminus G ) = 0$ are used in the middle equality. Completion gives the final assertion. □

The comparison argument below adapts the density-estimate scheme of [13, Sec. 4] to the present fully implicit JKO step. The essential diference is that the localized first variation involves the self-consistent chemical potential $\Phi _ { 1 }$ of $( \mathrm { C . 1 } )$ , rather than $U ^ { \prime } ( \rho _ { 1 } )$ alone. Consequently, the relevant invariant sets are superlevel sets of $\Phi _ { 1 }$ , and converting this invariance into density bounds introduces the field-oscillation loss $\omega _ { 1 }$ . In this correspondence, Lemma C.4 and Propositions C.5–C.6 play the roles of [13, Lem. 4.1 and Props. 4.2–4.3], respectively.

For the remainder of this subsection, fix a full-µ<sub>0</sub>-measure Borel set $G \subset \Omega$ on which $T$ is injective.

Lemma C.2 (Localized first variation of the full energy). Assume $V \in L ^ { \infty } ( \Omega )$ , and $W \in$ $L ^ { \infty } ( \Omega \times \Omega )$ is symmetric, $\rho > 0$ almost everywhere, and $h \in L ^ { \infty } ( \Omega )$ is supported in a set $E$ on which $0 < c \leq \rho \leq C$ . Then

$$
{ \mathcal { F } } ( ( \rho + s h ) d x ) - { \mathcal { F } } ( \rho d x ) = s \int _ { \Omega } \left( U ^ { \prime } ( \rho ) + V + W [ \rho ] \right) h d x + o ( s )
$$

as $s \downarrow 0 .$

Proof. We compute the first variation term by term. The internal and external parts are immediate under the localized two-sided density bound, while the interaction contributes the linear field $W [ \rho ]$ plus a quadratic remainder.

For suficiently small $s ,$ the values of $\rho + s h$ on $E$ remain in a compact subset of $( 0 , \infty )$ . The mean-value theorem and dominated convergence give the internal-energy derivative. Symmetry of $W$ gives the exact interaction expansion

$$
\mathscr { W } ( ( \rho + s h ) d x ) - \mathscr { W } ( \rho d x ) = s \int _ { \Omega } W [ \rho ] h d x + \frac { s ^ { 2 } } { 2 } \int \int W ( x , y ) h ( x ) h ( y ) d x d y .
$$

The last term is $O ( s ^ { 2 } )$ , which proves the claim.

Lemma C.3 (Localized target-replacement competitor). Let $( T , q )$ satisfy (2.10), and let $A \subset$ G be Borel with $0 < \mu _ { 0 } ( A ) < \infty$ . Put

$$
m _ { 0 } = \mu _ { 0 } ( A ) , \qquad m _ { 1 } = \mu _ { 1 } ( T ( A ) ) = \int _ { A } q ^ { 2 } d \mu _ { 0 } , \qquad \alpha = \frac { m _ { 1 } } { m _ { 0 } } ,
$$

and

$$
\widetilde { \mu } _ { 1 } = \mu _ { 1 } - \mu _ { 1 } | _ { T ( A ) } + \alpha \mu _ { 0 } | _ { A } .
$$

Then ${ \mathsf W } ^ { 2 } ( \mu _ { 0 } , \widetilde { \mu } _ { 1 } ) \le { \mathsf W } ^ { 2 } ( \mu _ { 0 } , \mu _ { 1 } )$ . Consequently, for

$$
\mu _ { 1 } ^ { s } = ( 1 - s ) \mu _ { 1 } + s \widetilde { \mu } _ { 1 } = \mu _ { 1 } + s \big ( \alpha \mu _ { 0 } \big | _ { A } - \mu _ { 1 } \big | _ { T ( A ) } \big ) , \qquad s \in [ 0 , 1 ] ,
$$

it holds ${ \sf H K } ^ { 2 } ( \mu _ { 0 } , \mu _ { 1 } ^ { s } ) \leq { \sf H K } ^ { 2 } ( \mu _ { 0 } , \mu _ { 1 } )$

Proof. We first construct a competitor that replaces the target mass generated from A by a diagonal growth branch supported on A. The choice of α preserves the amount of target mass.

By Lemma C.1, the target generated by the restriction to $A$ is exactly $\mu _ { 1 } | _ { T ( A ) }$ . Define a new admissible pair $( \widetilde { T } , \widetilde { q } )$ by keeping $( T , q )$ outside A and setting $\widetilde { T } = \mathrm { i d } , \widetilde { q } = \dot { \sqrt { \alpha } }$ on A. Its pushforward is $\bar { \tilde { T } } _ { \# } ( \tilde { q } ^ { 2 } \mu _ { 0 } ) = \bar { \mu } _ { 1 }$ , again by Lemma C.1.

We next compare the cone action of this replacement with the original optimal branch. The calculation shows that the replacement cannot increase the metric cost.

The two pairs agree outside A, so only the contributions on A have to be compared. The new one is

$$
\int _ { A } \bigl ( 1 + \alpha - 2 \sqrt { \alpha } \bigr ) d \mu _ { 0 } = m _ { 0 } + m _ { 1 } - 2 \sqrt { m _ { 0 } m _ { 1 } } ,
$$

since $\alpha m _ { 0 } = m _ { 1 }$ and $\sqrt { \alpha } m _ { 0 } = \sqrt { m _ { 0 } m _ { 1 } }$ , while the original one is

$$
m _ { 0 } + m _ { 1 } - 2 \int _ { A } q \cos \left( { \mathsf { d } } _ { \Omega } ( x , T ( x ) ) \wedge { \frac { \pi } { 2 } } \right) d \mu _ { 0 } .
$$

By Cauchy–Schwarz, $\textstyle \int _ { A } q$ cos $\begin{array} { r } { \left( { \mathsf { d } } _ { \Omega } ( x , T ( x ) ) \wedge \frac { \pi } { 2 } \right) d \mu _ { 0 } \ \leq \ \int _ { A } q d \mu _ { 0 } \ \leq \ \mu _ { 0 } ( A ) ^ { 1 / 2 } \big ( \int _ { A } q ^ { 2 } d \mu _ { 0 } \big ) ^ { 1 / 2 } \ = \qquad } \end{array}$ $\sqrt { m _ { 0 } m _ { 1 } }$ , so $A _ { \mu _ { 0 } } ( \widetilde { T } , \widetilde { q } ) \le { \cal A } _ { \mu _ { 0 } } ( T , q )$ . Since the new pair is admissible whereas $( T , q )$ is optimal,

$$
\mathsf { H } ^ { 2 } ( \mu _ { 0 } , \widetilde { \mu } _ { 1 } ) \le \mathcal { A } _ { \mu _ { 0 } } ( \widetilde { T } , \widetilde { q } ) \le \mathcal { A } _ { \mu _ { 0 } } ( T , q ) = \mathsf { H } ^ { 2 } ( \mu _ { 0 } , \mu _ { 1 } ) .
$$

Having obtained the endpoint competitor, we pass to the interpolated measures used in the variation argument. Convexity of the squared HK distance is exactly what is needed here.

Finally, $\nu \mapsto { \mathsf { H K } } ^ { 2 } ( \mu _ { 0 } , \nu )$ is convex by the entropy–transport formulation, so

$$
\mathsf { H } ^ { 2 } ( \mu _ { 0 } , \mu _ { 1 } ^ { s } ) \leq ( 1 - s ) \mathsf { H } ^ { 2 } ( \mu _ { 0 } , \mu _ { 1 } ) + s \mathsf { H } ^ { 2 } ( \mu _ { 0 } , \widetilde { \mu } _ { 1 } ) \leq \mathsf { H } ^ { 2 } ( \mu _ { 0 } , \mu _ { 1 } ) .
$$

Lemma C.4 (Monotonicity of the discrete chemical potential along the optimal map). Under the assumptions of this subsection,

$$
\Phi _ { 1 } ( T x ) \le \Phi _ { 1 } ( x ) \qquad f o r \mu _ { 0 } - a l m o s t \ e v e r y \ x .\tag{C.2}
$$

Proof. We argue by contradiction. We first localize a strict violation of the desired monotonicity to a set on which the density and chemical potential are uniformly controlled.

Suppose the conclusion fails, so that $S : = \{ x \in G : \Phi _ { 1 } ( T x ) > \Phi _ { 1 } ( x ) \}$ has positive $\mu _ { 0 ^ { - } }$ measure. For rationals $a \ < \ b$ put $S _ { a , b } \ = \ \{ x \ \in \ G \ : \ \Phi _ { 1 } ( x ) \ < \ a \ < \ b \ < \ \Phi _ { 1 } ( T x ) \}$ . Then $\begin{array} { r } { S = \bigcup _ { a < b \mathrm { \ r a t i o n a l } } S _ { a , b } } \end{array}$ is a countable union, so $\mu _ { 0 } ( S _ { a , b } ) > 0$ for at least one such pair; fix it. Intersecting $S _ { a , b }$ with the sets

$$
\{ n ^ { - 1 } \leq q \leq n \} , \qquad \{ \rho _ { 1 } \geq n ^ { - 1 } \} \cap T ^ { - 1 } \{ \rho _ { 1 } \geq n ^ { - 1 } \} , \qquad \{ | \Phi _ { 1 } | \leq n \} \cap T ^ { - 1 } \{ | \Phi _ { 1 } | \leq n \}\tag{C.3}
$$

and letting $n  \infty$ , we may in addition assume that on the resulting set $A \subset G$ the factor q is bounded above and away from zero, and that $\rho _ { 0 } , \rho _ { 1 }$ and $\Phi _ { 1 }$ are bounded on $A \cup T ( A )$ with $\rho _ { 1 }$ bounded away from zero there. This is possible because $\rho _ { 0 }$ is bounded and strictly positive, $\rho _ { 1 } > 0$ almost everywhere, and $q > 0$ for $\mu _ { 0 }$ -almost every point; the sets $T ( A )$ are Borel by Lemma C.1.

We evaluate the variation of the full energy in the direction given by Lemma C.3. The strict ordering of $\Phi _ { 1 }$ on A and $T ( A )$ makes it negative.

Let $\eta = \alpha \mu _ { 0 } | _ { A } - \mu _ { 1 } | _ { T ( A ) }$ be the direction in Lemma C.3, that is, $\eta \ : = \ : h$ dx with $h \ =$ $\alpha \rho _ { 0 } \mathbb { 1 } _ { A } - \rho _ { 1 } \mathbb { 1 } _ { T ( A ) }$ . Then $h \in L ^ { \infty }$ , it is supported in $E = A \cup T ( A )$ , on which $\rho _ { 1 }$ is bounded above and away from zero, and $h _ { - } \leq \rho _ { 1 } \mathbb { 1 } _ { T ( A ) } \leq \rho _ { 1 }$ pointwise, so Lemma C.2 applies with $\rho = \rho _ { 1 }$ and yields

$$
{ \mathcal { F } } ( \mu _ { 1 } + s \eta ) - { \mathcal { F } } ( \mu _ { 1 } ) = s \int _ { \Omega } \Phi _ { 1 } d \eta + o ( s ) .
$$

Note that $\Phi _ { 1 } < a$ on A and $\Phi _ { 1 } > b$ on $T ( A )$ , the latter because every $y \in T ( A )$ is $y = T x$ for some $x \in A$ . Hence, using $\alpha \mu _ { 0 } ( A ) = m _ { 1 } = \mu _ { 1 } ( T ( A ) )$ from Lemma C.1 and the definition of $\alpha .$

$$
\int _ { \Omega } \Phi _ { 1 } d \eta = \alpha \int _ { A } \Phi _ { 1 } d \mu _ { 0 } - \int _ { T ( A ) } \Phi _ { 1 } d \mu _ { 1 } \leq a \alpha \mu _ { 0 } ( A ) - b \mu _ { 1 } ( T ( A ) ) = ( a - b ) \mu _ { 1 } ( T ( A ) ) < 0 .
$$

This $\mathrm { g i }$ ves the required contradiction: the energy decreases to first order, whereas the proximal term does not increase.

The energy therefore decreases for small $s > 0$ , while Lemma C.3 shows that the HK term does not increase. This contradicts the minimality of $\mu _ { 1 }$ □

Replacing G by its intersection with the full-measure Borel set on which (C.2) holds, we henceforth assume that T is injective on G and

$$
\Phi _ { 1 } ( T x ) \leq \Phi _ { 1 } ( x ) \qquad { \mathrm { f o r ~ e v e r y ~ } } x \in G .
$$

Recall the standing bounds

$$
0 < c _ { \operatorname* { m i n } } \le \rho _ { 0 } \le c _ { \operatorname* { m a x } } < \infty .
$$

Proposition C.5 (Upper density bound). Let $c _ { \mathrm { u p p } } > 0$ satisfy

$$
1 + 2 \tau \operatorname * { m i n } \{ U ^ { \prime } ( c _ { \mathrm { u p p } } ) + \underset { \Omega } { \mathrm { e s s } } \operatorname * { i n f } P _ { 1 } , 0 \} > 0 ,
$$

and set

$$
\Lambda = { \frac { 1 } { 1 + 2 \tau \operatorname * { m i n } \{ U ^ { \prime } ( c _ { \mathrm { { u p p } } } ) + \mathrm { { e s s } } \operatorname * { i n f } _ { \Omega } P _ { 1 } , 0 \} } } ( \geq 1 ) .\tag{C.4}
$$

Then

$$
\rho _ { 1 } ( x ) \le ( U ^ { \prime } ) ^ { - 1 } \biggl ( \operatorname* { m a x } \{ U ^ { \prime } ( c _ { \mathrm { u p p } } ) , U ^ { \prime } ( \Lambda ^ { 2 } c _ { \mathrm { m a x } } ) \} + \mathrm { e s s o s c } P _ { 1 } \biggr )\tag{C.5}
$$

for almost every $x \in \Omega$

Proof. We prove the upper bound by excluding a suficiently high superlevel set of the chemical potential. The monotonicity lemma makes this set backward-invariant under the optimal map, and the rest of the argument turns that invariance into a contradiction with minimality.

Fix $\varepsilon > 0$ , put $\Lambda _ { \varepsilon } = \Lambda + \varepsilon ,$ , and define

$$
\ell _ { \varepsilon } = \operatorname* { m a x } \left\{ U ^ { \prime } ( ( 1 + \varepsilon ) ^ { 2 } c _ { \mathrm { u p p } } ) , U ^ { \prime } ( \Lambda _ { \varepsilon } ^ { 2 } c _ { \mathrm { m a x } } ) \right\} , \qquad B = \{ x : \Phi _ { 1 } ( x ) \geq \ell _ { \varepsilon } + \exp \operatorname* { s u p } _ { \Omega } P _ { 1 } \} .
$$

On B one has $U ^ { \prime } ( \rho _ { 1 } ) = \Phi _ { 1 } - P _ { 1 } \geq \ell _ { \varepsilon }$ , hence, $U ^ { \prime }$ being increasing,

$$
\rho _ { 1 } \geq ( 1 + \varepsilon ) ^ { 2 } c _ { \mathrm { u p p } } \qquad \mathrm { a n d } \qquad \rho _ { 1 } \geq \Lambda _ { \varepsilon } ^ { 2 } c _ { \mathrm { m a x } } \qquad \mathrm { o n } ~ B .\tag{C.6}
$$

Moreover, (C.2) gives $T ^ { - 1 } ( B ) \cap G \subset B$ because if $x \in G$ and $T x \in B$ , then $\Phi _ { 1 } ( x ) \geq \Phi _ { 1 } ( T x ) \geq$ $\ell _ { \varepsilon } + \mathrm { e s s } \mathrm { s u p } _ { \Omega } P _ { 1 }$ . We write $B ^ { \prime } : = T ^ { - 1 } ( B ) \cap G$

Suppose that this high-potential set carries positive target mass, that is, assume $\mu _ { 1 } ( B ) > 0$ Since G has full µ<sub>0</sub>-measure, $T _ { \# } ( q ^ { 2 } \mu _ { 0 } ) = \mu _ { 1 }$ gives $\begin{array} { r } { \int _ { B ^ { \prime } } q ^ { 2 } d \mu _ { 0 } = \mu _ { 1 } ( B ) } \end{array}$ . Combining this with (C.6), with $B ^ { \prime } \subset B$ , and with $\rho _ { 0 } \leq c _ { \mathrm { m a x } }$ 2

$$
\int _ { B ^ { \prime } } q ^ { 2 } d \mu _ { 0 } = \mu _ { 1 } ( B ) = \int _ { B } \rho _ { 1 } d x \geq \Lambda _ { \varepsilon } ^ { 2 } c _ { \operatorname* { m a x } } | B | \geq \Lambda _ { \varepsilon } ^ { 2 } c _ { \operatorname* { m a x } } | B ^ { \prime } | \geq \Lambda _ { \varepsilon } ^ { 2 } \mu _ { 0 } ( B ^ { \prime } ) .\tag{C.7}
$$

In particular $\mu _ { 0 } ( B ^ { \prime } ) > 0$ ; If not, $\mu _ { 1 } ( B ) = 0$ due to the first equality in (C.7).

If $q \ < \ \Lambda _ { \varepsilon }$ held µ<sub>0</sub>-almost everywhere on $B ^ { \prime } { } _ { ; }$ , then $\textstyle \int _ { B ^ { \prime } } q ^ { 2 } d \mu _ { 0 } \ < \ \Lambda _ { \varepsilon } ^ { 2 } \mu _ { 0 } ( B ^ { \prime } )$ , contradicting $\mathrm { ( C . 7 ) }$ ; hence $A _ { 0 } : = B ^ { \prime } \cap \{ q \geq \Lambda _ { \varepsilon } \}$ has positive µ -measure. Truncating as in the proof of Lemma C.4 (see (C.3)), choose $A \subset A _ { 0 }$ of positive $\mu _ { 0 } \cdot$ -measure on which, in addition, $\rho _ { 1 }$ and $\Phi _ { 1 }$ are bounded on $T ( A ) { \mathrm { ; } }$ the lower bound (C.6) keeps $\rho _ { 1 }$ away from zero there. Put $\mu _ { 1 } ^ { * } = \mu _ { 1 } | _ { T ( A ) }$ and, for $0 < t < 1$

$$
\mu _ { 1 } ^ { t } = \mu _ { 1 } - ( 1 - t ^ { 2 } ) \mu _ { 1 } ^ { * } .
$$

We will show that, for $t < 1$ suficiently close to 1, this competitor has strictly smaller JKO objective than $\mu _ { 1 }$ , contradicting the minimality of $\mu _ { 1 }$

To exploit this large-growth region, we scale the growth factor down on A while leaving the map unchanged. We now quantify the resulting decrease in the cone action, and hence in the squared HK term: Let $( T _ { t } , q _ { t } )$ agree with $( T , q )$ outside A and equal $( T , t q )$ on A. $\mathrm { B y }$ Lemma C.1, $( T _ { t } ) _ { \# } ( q _ { t } ^ { 2 } \mu _ { 0 } ) = \mu _ { 1 } - ( 1 - t ^ { 2 } ) \mu _ { 1 } ^ { * } = \mu _ { 1 } ^ { t }$ , so $( T _ { t } , q _ { t } )$ is admissible for $\mu _ { 1 } ^ { t }$ and only its contribution on A difers from that of $( T , q )$ . Writing $r ( x ) = \mathsf { d } _ { \Omega } ( x , T ( x ) ) \wedge \frac { \pi } { 2 }$ , that diference is

$$
\int _ { A } \bigl [ ( 1 + q ^ { 2 } - 2 q \cos r ) - ( 1 + t ^ { 2 } q ^ { 2 } - 2 t q \cos r ) \bigr ] d \mu _ { 0 } = ( 1 - t ) \int _ { A } \bigl [ ( 1 + t ) q ^ { 2 } - 2 q \cos r \bigr ] d \mu _ { 0 } .
$$

Here $\begin{array} { r } { \int _ { A } q ^ { 2 } d \mu _ { 0 } = \mu _ { 1 } ( T ( A ) ) = \mu _ { 1 } ^ { * } ( \Omega ) } \end{array}$ by Lemma C.1, while cos $r \leq 1$ and $q \geq \Lambda _ { \varepsilon }$ on $A$ give $\begin{array} { r } { \int _ { A } q \cos { r } d \mu _ { 0 } \le \int _ { A } q d \mu _ { 0 } \le \Lambda _ { \varepsilon } ^ { - 1 } \int _ { A } q ^ { 2 } d \mu _ { 0 } } \end{array}$ . Since $( T , q )$ is optimal and $( T _ { t } , q _ { t } )$ is merely admissible, (2.10) and (1.3) yield

$$
\mathsf { H } ^ { 2 } ( \mu _ { 0 } , \mu _ { 1 } ) - \mathsf { H } ^ { 2 } ( \mu _ { 0 } , \mu _ { 1 } ^ { t } ) \geq \mathcal { A } _ { \mu _ { 0 } } ( T , q ) - \mathcal { A } _ { \mu _ { 0 } } ( T _ { t } , q _ { t } ) \geq ( 1 - t ) \left( 1 + t - \frac { 2 } { \Lambda _ { \varepsilon } } \right) \mu _ { 1 } ^ { * } ( \Omega ) .\tag{C.8}
$$

We next compare the energy change for the same perturbation. $\mathrm { O n } \ T ( A )$ the chemical potential has a uniform lower bound, so the localized first variation gives a matching first-order energy change.

Set $a = U ^ { \prime } ( c _ { \mathrm { u p p } } )$ + ess inf<sub>Ω</sub> $P _ { 1 }$ . Since $A \subset T ^ { - 1 } ( B )$ , one has $T ( A ) \subset B$ and therefore, by $( \mathrm { C } . 6 ) , \Phi _ { 1 } \geq U ^ { \prime } ( ( 1 + \varepsilon ) ^ { 2 } c _ { \mathrm { u p p } } ) \ \mathsf { - }$ + ess inf<sub>Ω</sub> $P _ { 1 } \geq a$ on $T ( A )$ . With $s = 1 - t ^ { 2 }$ and $h = - \rho _ { 1 } \mathbb { 1 } _ { T ( A ) }$ which satisfies $h _ { - } \leq \rho _ { 1 }$ , Lemma C.2 gives

$$
\mathcal { F } ( \mu _ { 1 } ) - \mathcal { F } ( \mu _ { 1 } ^ { t } ) = s \int _ { T ( A ) } \Phi _ { 1 } d \mu _ { 1 } + o ( s ) \geq ( 1 - t ) ( 1 + t ) a \mu _ { 1 } ^ { * } ( \Omega ) + o ( 1 - t ) .\tag{C.9}
$$

Combining (C.8) and (C.9), dividing by $1 - t .$ , we obtain

$$
\operatorname* { l i m } _ { t \uparrow \uparrow } \frac { \mathcal { I } _ { \tau } ( \mu _ { 1 } \mid \mu _ { 0 } ) - \mathcal { I } _ { \tau } ( \mu _ { 1 } ^ { t } \mid \mu _ { 0 } ) } { 1 - t } \geq \left[ \frac { 1 } { 2 \tau } \left( 2 - \frac { 2 } { \Lambda _ { \varepsilon } } \right) + 2 a \right] \mu _ { 1 } ^ { * } ( \Omega ) = \left[ \frac { 1 } { \tau } \left( 1 - \frac { 1 } { \Lambda _ { \varepsilon } } \right) + 2 a \right] \mu _ { 1 } ^ { * } ( \Omega ) .
$$

The right-hand side is positive; When $a \geq 0$ this is immediate, since $\Lambda = 1$ and $\Lambda _ { \varepsilon } > 1$ . When $a < 0$ one has $\Lambda = ( 1 + 2 \tau a ) ^ { - 1 }$ , so that

$$
\frac { 1 } { \tau } \left( 1 - \frac { 1 } { \Lambda } \right) + 2 a = \frac { 1 } { \tau } \big ( 1 - ( 1 + 2 \tau a ) \big ) + 2 a = - 2 a + 2 a = 0 ,
$$

and the expression at $\Lambda _ { \varepsilon } > \Lambda$ is strictly larger because $\Lambda \mapsto 1 - \Lambda ^ { - 1 }$ is increasing. Hence

$$
\mathcal { T } _ { \tau } ( \mu _ { 1 } ^ { t } \mid \mu _ { 0 } ) < \mathcal { T } _ { \tau } ( \mu _ { 1 } \mid \mu _ { 0 } )
$$

for $t < 1$ suficiently close to 1, contradicting the minimality of $\mu _ { 1 }$ . Therefore $\mu _ { 1 } ( B ) = 0$ , and since $\rho _ { 1 }$ is bounded below by a positive constant on B by $\left( \mathrm { C } . 6 \right)$ , also $| B | = 0$

The contradiction shows that the high-potential set is negligible. It remains only to translate this chemical-potential statement back into a pointwise upper bound for the density.

Let $\xi _ { \varepsilon }$ be defined by

$$
U ^ { \prime } ( \xi _ { \varepsilon } ) = \ell _ { \varepsilon } + \exp \operatorname * { o s c } _ { \Omega } P _ { 1 } .
$$

If $\rho _ { 1 } ( x ) \geq \xi _ { \varepsilon }$ , then $\Phi _ { 1 } ( x ) \geq \ell _ { \varepsilon } +$ ess osc<sub>Ω</sub> P<sub>1</sub> + ess inf<sub>Ω</sub> $P _ { 1 } = \ell _ { \varepsilon } + \mathrm { e s s } \mathrm { s u p } _ { \Omega } P _ { 1 }$ , so $\{ \rho _ { 1 } \ge \xi _ { \varepsilon } \} \subset B$ and hence $\rho _ { 1 } < \xi _ { \varepsilon }$ almost everywhere. Letting $\varepsilon \downarrow 0$ and using the continuity of $U ^ { \prime }$ proves (C.5). □

Proposition C.6 (Lower density bound). Choose $c _ { \mathrm { l o w } } > 0$ such that

$$
U ^ { \prime } ( c _ { \mathrm { l o w } } ) + \exp { \frac { \imath } { \Omega } } P _ { 1 } < 0 ,
$$

which is always possible under $U ^ { \prime } ( 0 + ) = - \infty$ . Set $\bar { c } = \operatorname* { m i n } \{ c _ { \mathrm { m i n } } , c _ { \mathrm { l o w } } \} . \ T h e$ n

$$
\rho _ { 1 } ( x ) \geq ( U ^ { \prime } ) ^ { - 1 } \bigg ( U ^ { \prime } ( \bar { c } ) - \underset { \Omega } { \operatorname { e s s } } \ : \mathrm { o s c } P _ { 1 } \bigg )\tag{C.10}
$$

for almost every $x \in \Omega$

Proof. The lower bound is proved by the analogous low-potential argument, but the competitor is diferent: increasing the target locally is achieved here by splitting source mass between the original transport branch and a diagonal branch. We therefore first identify a region where the optimal growth factor is strictly smaller than one, and then build this split competitor explicitly.

Fix $\varepsilon > 0$ , set $c _ { \varepsilon } = \bar { c } / ( 1 + \varepsilon )$ , and define

$$
B = \{ x : \Phi _ { 1 } ( x ) \leq U ^ { \prime } ( c _ { \varepsilon } ) + \underset { \Omega } { \operatorname { e s s i n f } } P _ { 1 } \} .
$$

$\operatorname { B y } \ ( \mathrm { C . 2 } ) , T ( B \cap G ) \subset B$ , and on B one has $\rho _ { 1 } \leq c _ { \varepsilon }$

Assume that the low-potential set has positive measure. Forward invariance implies that the average of $q ^ { 2 }$ on this set is strictly below one.

If $| B | > 0$ , then $\mu _ { 0 } ( B \cap G ) > 0$ and Lemma C.1 gives

$$
\int _ { B \cap G } q ^ { 2 } d \mu _ { 0 } = \mu _ { 1 } ( T ( B \cap G ) ) \leq \mu _ { 1 } ( B ) \leq c _ { \varepsilon } | B | \leq { \frac { 1 } { 1 + \varepsilon } } \mu _ { 0 } ( B \cap G ) .
$$

Here $\mu _ { 0 } ( \Omega \setminus G ) = 0$ and the positive lower bound on $\rho _ { 0 }$ imply $| B \setminus G | = 0$ , so the last step uses $\begin{array} { r } { c _ { \varepsilon } | B | = \frac { \bar { c } } { 1 + \varepsilon } | B \cap G | \le \frac { c _ { \operatorname* { m i n } } } { 1 + \varepsilon } | B \cap G | \le \frac { 1 } { 1 + \varepsilon } \mu _ { 0 } ( B \cap G ) } \end{array}$ , by $\bar { c } \le c _ { \mathrm { m i n } } \le \rho _ { 0 }$

We sharpen this average estimate into a pointwise one on a positive-measure subset: there must be a region on which $q$ is uniformly bounded by some $k _ { \varepsilon } < 1$

Choose

$$
k _ { \varepsilon } ^ { 2 } = \frac { 1 + \varepsilon / 2 } { 1 + \varepsilon } < 1 .
$$

There is a measurable $A _ { 0 } \subset B \cap G$ of positive µ<sub>0</sub>-measure on which $q \leq k _ { \varepsilon }$ : otherwise $q > k _ { \varepsilon }$ µ<sub>0</sub>-almost everywhere on $B \cap G$ , contradicting the previous display because $k _ { \varepsilon } ^ { 2 } > ( 1 + \varepsilon ) ^ { - 1 }$ Since $\rho _ { 1 } > 0$ almost everywhere, there is $\delta > 0$ such that $A : = A _ { 0 } \cap \left\{ \rho _ { 1 } \geq \delta \right\}$ still has positive µ -measure; on A one has $\delta \le \rho _ { 1 } \le c _ { \varepsilon }$ and $\rho _ { 0 } \mathbb { 1 } _ { A } \in L ^ { \infty }$ is nonnegative, so Lemma C.2 applies with $E = A$

We now construct the competitor. For $t < 1$ close to 1, set

$$
\mu _ { 1 } ^ { t } = \mu _ { 1 } + ( 1 - t ^ { 2 } ) \mu _ { 0 } | _ { A } .
$$

To estimate $\mathsf { H K } ^ { 2 } ( \mu _ { 0 } , \mu _ { 1 } ^ { t } )$ , we construct an admissible triple in the cone formulation (2.4), preserving the original transported target mass and adding mass on A through a diagonal branch. Admissibility supplies the required cost upper bound. Writing $s _ { t } : = 1 - t ^ { 2 }$ , we reduce the source radial weight of the original transport branch from 1 to $t ,$ while keeping its target radial weight $q$ unchanged, and assign the missing source contribution $s _ { t } \mu _ { 0 } { \big | } _ { A }$ to a diagonal branch with radial weights $\left( \sqrt { s _ { t } } , \sqrt { s _ { t } } \right)$ . Thus the original transported target mass is preserved, while the diagonal branch adds exactly $s _ { t } \mu _ { 0 } { \big | } _ { A }$ to the target. On the fixed-point set $T ( x ) = x ,$ however, these two branches coincide and their radial weights must be merged. Accordingly, set

$$
A _ { \mathrm { f i x } } : = A \cap \{ x : T ( x ) = x \} , \qquad A _ { \ast } : = A \setminus A _ { \mathrm { f i x } } ,
$$

and take

$$
\gamma = ( \mathrm { i d } , T ) _ { \# } \mu _ { 0 } | _ { A _ { * } } + ( \mathrm { i d } , \mathrm { i d } ) _ { \# } \mu _ { 0 } | _ { A } + ( \mathrm { i d } , T ) _ { \# } \mu _ { 0 } | _ { \Omega \backslash A } .
$$

These three pieces are mutually singular. Define $( r _ { 0 } , r _ { 1 } )$ by

$$
\begin{array} { r } { ( r _ { 0 } , r _ { 1 } ) = \left\{ \begin{array} { l l } { ( t , q ) , } & { \mathrm { o n ~ } ( \mathrm { i d } , T ) ( A _ { * } ) , } \\ { ( \sqrt { s _ { t } } , \sqrt { s _ { t } } ) , } & { \mathrm { o n ~ } ( \mathrm { i d } , \mathrm { i d } ) ( A _ { * } ) , } \\ { ( 1 , \sqrt { q ^ { 2 } + s _ { t } } ) , } & { \mathrm { o n ~ } ( \mathrm { i d } , \mathrm { i d } ) ( A _ { \mathrm { f x } } ) , } \\ { ( 1 , q ) , } & { \mathrm { o n ~ } ( \mathrm { i d } , T ) ( \Omega \setminus A ) . } \end{array} \right. } \end{array}
$$

Since $T = \mathrm { i d }$ on $A _ { \mathrm { f i x } }$ and $T _ { \# } ( q ^ { 2 } \mu _ { 0 } ) = \mu _ { 1 }$ , the marginals are

$$
\pi _ { \# } ^ { 0 } ( r _ { 0 } ^ { 2 } \gamma ) = t ^ { 2 } \mu _ { 0 } | _ { A _ { * } } + s _ { t } \mu _ { 0 } | _ { A _ { * } } + \mu _ { 0 } | _ { A _ { \mathrm { f x } } } + \mu _ { 0 } | _ { \Omega \backslash A } = \mu _ { 0 } ,
$$

$$
\begin{array} { r l } & { \pi _ { \# } ^ { 1 } ( r _ { 1 } ^ { 2 } \gamma ) = T _ { \# } ( q ^ { 2 } \mu _ { 0 } | _ { A _ { * } } ) + s _ { t } \mu _ { 0 } | _ { A _ { * } } + ( q ^ { 2 } + s _ { t } ) \mu _ { 0 } | _ { A _ { \mathrm { f i x } } } + T _ { \# } ( q ^ { 2 } \mu _ { 0 } | _ { \Omega \backslash A } ) } \\ & { \qquad = \mu _ { 1 } + s _ { t } \mu _ { 0 } | _ { A } = \mu _ { 1 } ^ { t } , } \end{array}
$$

so the triple is admissible.

We next estimate the metric efect of this split construction. On both the moving and fixed-point parts of A, the new cone cost is strictly smaller to first order because $q < 1$

On A<sub>∗</sub> its excess cost over the original optimal pair is the negative of

$$
( 1 - t ) { \big ( } 1 + t - 2 q \cos r { \big ) } , \qquad r ( x ) : = \mathsf { d } _ { \Omega } ( x , T ( x ) ) \wedge { \frac { \pi } { 2 } } .
$$

On $A _ { \mathrm { f i x } }$ the merged diagonal branch has cost $\left( 1 - \sqrt { q ^ { 2 } + s _ { t } } \right) ^ { 2 }$ . Since $1 = { \sqrt { t ^ { 2 } + s _ { t } } }$ and $a \mapsto$ $\sqrt { a ^ { 2 } + s _ { t } }$ is 1-Lipschitz on $[ 0 , \infty )$ 2

$$
\left( 1 - \sqrt { q ^ { 2 } + s _ { t } } \right) ^ { 2 } \leq ( t - q ) ^ { 2 } .
$$

Thus on $A _ { \mathrm { f i x } }$ the decrease relative to the original cost $( 1 - q ) ^ { 2 }$ is at least $( 1 - t ) ( 1 + t - 2 q )$ Using $q \leq k _ { \varepsilon }$ on A and cos $r \leq 1$ on $A _ { * }$ , the total cost decrease is therefore at least

$$
( 1 - t ) ( 1 + t - 2 k _ { \varepsilon } ) \mu _ { 0 } ( A ) .
$$

Since the triple above is merely admissible while $( T , q )$ is optimal, this gives

$$
\mathsf { H } ^ { 2 } ( \mu _ { 0 } , \mu _ { 1 } ) - \mathsf { H } ^ { 2 } ( \mu _ { 0 } , \mu _ { 1 } ^ { t } ) \geq ( 1 - t ) ( 1 + t - 2 k _ { \varepsilon } ) \mu _ { 0 } ( A ) > 0
$$

for t suficiently close to 1, because $k _ { \varepsilon } < 1$ . It remains to compare the energy. Since A lies in the low-potential set, adding mass there also decreases the energy to first order, which contradicts minimality.

With $s = 1 - t ^ { 2 }$ , the localized first variation gives

$$
\mathcal { F } ( \mu _ { 1 } ^ { t } ) - \mathcal { F } ( \mu _ { 1 } ) = s \int _ { A } \Phi _ { 1 } d \mu _ { 0 } + o ( s ) .
$$

Since $A \subset B$ and $\begin{array} { r } { U ^ { \prime } ( c _ { \varepsilon } ) + \exp { \operatorname* { i n f } _ { \Omega } P _ { 1 } } < U ^ { \prime } ( c _ { \mathrm { l o w } } ) + \exp { \operatorname* { i n f } _ { \Omega } P _ { 1 } } < 0 } \end{array}$ , the energy also decreases for t close to 1. This contradicts minimality, so $| B | = 0$

Thus the low-potential set is negligible. We finally convert this information into the asserted lower density bound and then let $\varepsilon \downarrow 0$

Define $\xi _ { \varepsilon }$ by

$$
U ^ { \prime } ( \xi _ { \varepsilon } ) = U ^ { \prime } ( c _ { \varepsilon } ) - \operatorname * { e s s } _ { \Omega } \operatorname { o s c } P _ { 1 } .
$$

If $\rho _ { 1 } ( x ) \leq \xi _ { \varepsilon }$ , then $\Phi _ { 1 } ( x ) \le U ^ { \prime } ( c _ { \varepsilon } )$ − ess osc<sub>Ω</sub> P<sub>1</sub> + ess sup P<sub>1</sub> = U<sup>′</sup>(c<sub>ε</sub>) + ess inf<sub>Ω</sub> P<sub>1</sub>, so $\{ \rho _ { 1 } \leq$ $\xi _ { \varepsilon } \} \subset B$ and hence $\rho _ { 1 } > \xi _ { \varepsilon }$ almost everywhere. Let $\varepsilon \downarrow 0$ , so that $c _ { \varepsilon } \uparrow \bar { c } ,$ , to obtain (C.10).

## C.2 Dual potential and discrete Euler equation

Here $( T , q )$ is always oriented from the first argument of HK to the second. In the application to Theorem 2.6 the ordered pair is $( \mu _ { 1 } , \mu _ { 0 } )$ , so this $T$ is the reversed map, from the new state to the source.

Transformed dual potentials. The logarithmic dual variables of (2.2) are convenient to rescale. Writing $K ( x , y ) : = \cos ^ { 2 } ( \mathtt { d } _ { \Omega } ( x , y ) \wedge \frac { \pi } { 2 } ) = e ^ { - c _ { \mathsf { W } } ( x , y ) }$ and substituting

$$
\psi : = 1 - e ^ { - z _ { 0 } } , \qquad \chi : = 1 - e ^ { - z _ { 1 } } ,\tag{C.11}
$$

into (2.2) turns the objective into a linear one, because − $F ^ { * } ( - z ) = 1 - e ^ { - z }$ for $F ^ { * } ( a ) = e ^ { a } - 1$ 2 and turns the constraint $z _ { 0 } + z _ { 1 } \le c _ { \mathsf { H K } }$ into $e ^ { - z _ { 0 } } e ^ { - z _ { 1 } } \geq e ^ { - c _ { \mathsf { H K } } }$ . Hence (2.2) is equivalent to

$$
{ \sf H } ^ { 2 } ( \mu , \nu ) = \operatorname* { s u p } \left\{ \int _ { \Omega } \psi d \mu + \int _ { \Omega } \chi d \nu \right\} ,\tag{C.12}
$$

the supremum being over Borel $\psi , \chi \leq 1$ subject to

$$
( 1 - \psi ( x ) ) ( 1 - \chi ( y ) ) \geq K ( x , y ) = \cos ^ { 2 } \Bigl ( \mathsf { d } _ { \Omega } ( x , y ) \wedge \frac { \pi } { 2 } \Bigr ) \qquad \mathrm { f o r ~ a l l ~ } x , y \in \Omega .\tag{C.13}
$$

We call $( \psi , \chi )$ a transformed dual pair; $\psi$ is its source component and $\chi$ its target component. The pair is saturated when each component is the envelope of the other with respect to (C.13), that is,

$$
\psi ( x ) = 1 - \operatorname* { s u p } _ { y \in \Omega } \frac { K ( x , y ) } { 1 - \chi ( y ) } , \qquad \chi ( y ) = 1 - \operatorname* { s u p } _ { x \in \Omega } \frac { K ( x , y ) } { 1 - \psi ( x ) } .\tag{C.14}
$$

Saturation fixes a canonical representative: replacing a feasible pair by the corresponding envelopes preserves feasibility and does not decrease the objective. Following [11], we call the source component $\psi$ of this envelope-normalized pair the tight source potential; it is the component diferentiated below.

Lemma C.7 (Regularity and uniqueness of transformed HK dual potentials). Let Ω be either a compact convex subset of $\mathbb { R } ^ { d }$ with nonempty interior $o r$ the flat torus $\mathbb { T } ^ { d }$ , and let $\mu = \rho d x$ 2 $\nu = \tilde { \rho } d x ~ s a t i s f y$

$$
0 < m \leq \rho , \tilde { \rho } \leq M < \infty \qquad a . e .
$$

There is a saturated optimal pair $( \psi , \chi )$ in (C.12) with $\psi , \chi \in W ^ { 1 , \infty } ( \Omega )$ . Its saturated source component $\psi$ is unique µ-almost everywhere. Moreover, for some $\delta > 0$

$$
1 - \psi \ge \delta , \qquad 1 - \chi \ge \delta \qquad o n \Omega ,\tag{C.15}
$$

and the associated Monge map stays uniformly below the cutof $\pi / 2$ . Moreover, $\delta$ and th e W<sup>1,∞</sup> $W ^ { 1 }$ norms of $\psi$ and $\chi$ may be chosen to depend only on m, M, and Ω.

Proof. We first establish existence, uniqueness, and Lipschitz regularity of the transformed dual potentials. The argument is quoted separately in the Euclidean and torus settings because the available regularity results are formulated diferently.

In the Euclidean case, both measures have compact full support, so each support lies in the π/2-neighborhood of the other and the pair is strongly reduced in the sense of [11, Definition 2.6]. The uniqueness of the optimal marginal densities from [11, Theorem 2.14], together with the tight-potential regularity and Monge results in [11, Theorems 3.3 and 3.6 and Corollary 3.5], yields a saturated Lipschitz transformed pair and the associated Monge map. In the present normalization the optimal marginal densities are $1 - \psi = \sigma _ { 0 }$ and $1 - \chi = \sigma _ { 1 }$ , so their uniqueness gives uniqueness of the saturated source component µ-almost everywhere. Moreover, the factors $1 - \psi$ and $1 - \chi$ are positive and continuous on the compact full supports and hence are bounded away from zero.

We now treat the torus using the logarithmic potentials and then transform back to the variables $( \psi , \chi )$ . On the flat torus both measures have full support. In particular they form an admissible pair for the HK cost, since for each x one may choose $y = x ,$ , for which $c _ { \mathsf { H K } } ( x , x ) = 0$ The Lipschitz-potential and Monge results of [12, Sections 2.2 and 3.4.1] yield a unique Lipschitz logarithmic optimal pair $\left( z _ { 0 } , z _ { 1 } \right)$ and the associated Monge map. The change of variables (C.11) gives the corresponding saturated pair in (C.12); uniqueness of $z _ { \mathrm { 0 } }$ gives uniqueness of the source component. Since $z _ { 0 } , z _ { 1 }$ are finite and Lipschitz on the compact torus, $e ^ { - z _ { 0 } }$ and $e ^ { - z _ { 1 } }$ have positive lower bounds.

Thus, in either geometry, after decreasing $\delta > 0$ if necessary, (C.15) holds. It remains to verify that the associated Monge map stays uniformly below the cutof. On the contact set,

$$
( 1 - \psi ( x ) ) ( 1 - \chi ( T ( x ) ) ) = \cos ^ { 2 } \left( { \tt d } _ { \Omega } ( x , T ( x ) ) \wedge \frac { \pi } { 2 } \right) .
$$

Hence (C.15) gives

$$
\cos ^ { 2 } \left( \mathsf { d } _ { \Omega } ( x , T ( x ) ) \wedge \frac { \pi } { 2 } \right) \ge \delta ^ { 2 } ,
$$

and therefore

$$
{ \mathsf { d } } _ { \Omega } ( x , T ( x ) ) \leq \operatorname { a r c c o s } ( \delta ) < { \frac { \pi } { 2 } }
$$

for µ-almost every x (after replacing δ by min $\{ \delta , 1 \}$ if necessary).

For the later Euler argument we need more than pairwise regularity: the bounds must be uniform over all density pairs satisfying the same two-sided estimates. We obtain this by compactness and the saturated-envelope representation.

To make this quantitative, consider all density pairs with the same m and M. Set

$$
a : = 1 - \psi \geq 0 , \qquad b : = 1 - \chi \geq 0 .
$$

The preceding argument gives positive continuous a, b on the compact space Ω. Since $0 \leq K \leq 1$ $K ( x , x ) = 1$ , and the pair is saturated, (C.14) implies

$$
\operatorname* { m a x } _ { \Omega } a = \frac { 1 } { \operatorname* { m i n } _ { \Omega } b } , \qquad \operatorname* { m a x } _ { \Omega } b = \frac { 1 } { \operatorname* { m i n } _ { \Omega } a } .
$$

Let $a _ { \mathrm { m a x } } : = \operatorname* { m a x } _ { \Omega } a$ . The saturation formula and $\operatorname { L i p } ( K ) \leq 1$ give

$$
| a ( x ) - a ( x ^ { \prime } ) | \leq \frac { \mathrm { L i p } ( K ) } { \operatorname* { m i n } _ { \Omega } b } { \mathsf { d } } _ { \Omega } ( x , x ^ { \prime } ) \leq a _ { \operatorname* { m a x } } { \mathsf { d } } _ { \Omega } ( x , x ^ { \prime } ) .
$$

If $x _ { 0 }$ is a maximizer of $a ,$ then $a \ge a _ { \mathrm { m a x } } / 2$ on $B _ { 1 / 2 } ( x _ { 0 } ) \cap \Omega$ . Since Ω is fixed, compact, and is either convex with nonempty interior or a flat torus,

$$
v _ { \Omega } : = \operatorname* { i n f } _ { x \in \Omega } | B _ { 1 / 2 } ( x ) \cap \Omega | > 0 ,
$$

where the balls are Riemannian in the torus case. The feasible pair (0, 0) in (C.12) shows that the optimal value is nonnegative, and hence

$$
\int _ { \Omega } a d \mu + \int _ { \Omega } b d \nu = \mu ( \Omega ) + \nu ( \Omega ) - \mathsf { \mathsf { H } } ^ { 2 } ( \mu , \nu ) \leq 2 M | \Omega | .
$$

Using $\rho \ge m$ therefore yields

$$
\frac { m a _ { \mathrm { m a x } } v _ { \Omega } } { 2 } \leq \int _ { \Omega } a d \mu \leq 2 M | \Omega | , \qquad a _ { \mathrm { m a x } } \leq \frac { 4 M | \Omega | } { m v _ { \Omega } } .
$$

The same argument for b gives the same bound for max b. Consequently

$$
a , b \geq \delta _ { 0 } : = \frac { m v _ { \Omega } } { 4 M | \Omega | } > 0 ,
$$

uniformly over the stated class of densities. The saturation formulas then also give $a , b \leq \delta _ { 0 } ^ { - 1 }$ and

$$
\mathrm { L i p } ( \psi ) , \mathrm { L i p } ( \chi ) \leq \frac { \mathrm { L i p } ( K ) } { \delta _ { 0 } } ,
$$

where we used the elementary inequality $\left| \operatorname* { s u p } _ { y } f ( y ) - \operatorname* { s u p } _ { y } g ( y ) \right| \leq \operatorname* { s u p } _ { y } | f ( y ) - g ( y ) |$ . So the full $W ^ { 1 , \infty }$ bounds depend only on $m , M$ , and Ω. This is the uniformity used in Lemma C.10 for the family of perturbed densities. □

Corollary C.8 (Identities for an optimal Monge-growth pair). Let $( T , q )$ be an essentially injective optimal Monge-growth pair from µ to ν, so that $T _ { \# } ( q ^ { 2 } \mu ) = \nu _ { \# }$ , and let T also be the Monge map of Lemma C.7. The two coincide by uniqueness of the entropy–transport optimizer in the two cases covered by that lemma. For the saturated representatives in Lemma C.7,

$$
1 - \psi ( x ) = q ( x ) \cos { r ( x ) } , \qquad 1 - \chi ( T ( x ) ) = \frac { \cos { r ( x ) } } { q ( x ) } , \qquad r ( x ) : = \mathtt { d } _ { \Omega } ( x , T ( x ) ) ,\tag{C.16}
$$

for $\mu - a . e . \ x$ on the transported part. In particular, for some $\varepsilon > 0$

$$
r ( x ) \leq { \frac { \pi } { 2 } } - \varepsilon \qquad f o r \mu \mathrm { - } a . e . x .
$$

Proof. We identify the density of the optimal entropy–transport plan relative to the source in two diferent ways. The primal minimization gives it in terms of $( T , q )$ , while dual optimality gives it in terms of the source potential; equating the two descriptions yields the claimed identities.

The entropy–transport optimizer is concentrated on the graph of $T _ { i }$ , so it has the form $\gamma = ( \mathrm { i d } , T ) _ { \# } ( h \mu )$ for a Borel $h \geq 0$ , and it remains to identify h. Its first marginal is $\gamma _ { 0 } = h \mu ,$ so $d \gamma _ { 0 } / d \mu = h$ . Its second marginal is $\gamma _ { 1 } = T _ { \# } ( h \mu )$ , while $\nu = T _ { \# } ( q ^ { 2 } \mu )$ ; since T is essentially injective, the two densities compare at the transported point, namely

$$
\frac { d \gamma _ { 1 } } { d \nu } \circ T = \frac { h } { q ^ { 2 } } \qquad \mu \mathrm { - a . e . , } \qquad \mathrm { h e n c e } \qquad \int _ { \Omega } F \left( \frac { d \gamma _ { 1 } } { d \nu } \right) d \nu = \int _ { \Omega } F \left( \frac { h } { q ^ { 2 } } \right) q ^ { 2 } d \mu .
$$

Substituting this and $\textstyle \int c _ { \mathsf { \boldsymbol { \mathsf { H } } } } d \gamma = \int c _ { \mathsf { \boldsymbol { \mathsf { H } } } } ( x , T ( x ) ) h d \mu$ into (2.1), the entropy–transport cost on the right-hand side of (2.1) becomes $\int _ { \Omega } \Theta d \mu$ with pointwise integrand

$$
\Theta = F ( h ) + q ^ { 2 } F \bigg ( { \frac { h } { q ^ { 2 } } } \bigg ) + c _ { \mathsf { H } } ( x , T ( x ) ) h .
$$

Being optimal among all plans, $\gamma$ is in particular optimal within the class of plans of this form, in which h is an unconstrained nonnegative Borel function. Since $F ^ { \prime } ( \sigma ) = \log \sigma$ , minimizing Θ in $h > 0$ pointwise therefore gives

$$
\log h + \log { \frac { h } { q ^ { 2 } } } + c _ { \mathsf { H K } } ( x , T ( x ) ) = 0 , \qquad \mathrm { t h a t ~ i s } \qquad { \frac { h ^ { 2 } } { q ^ { 2 } } } = e ^ { - c _ { \mathsf { H K } } ( x , T ( x ) ) } = \cos ^ { 2 } r ,
$$

so $h = q$ cos r.

Let $\left( z _ { 0 } , z _ { 1 } \right)$ be the optimal logarithmic dual pair corresponding to $( \psi , \chi )$ through (C.11). We apply Fenchel’s inequality

$$
F ( s ) + F ^ { * } ( - z ) + z s \geq 0 , \qquad { \mathrm { w i t h ~ e q u a l i t y ~ i f ~ a n d ~ o n l y ~ i f ~ } } s = e ^ { - z } ,
$$

to $( s , z ) = ( h , z _ { 0 } )$ and $( s , z ) = ( d \gamma _ { 1 } / d \nu , z _ { 1 } )$ . Using the identities $\textstyle \int z _ { 0 } h d \mu = \int z _ { 0 } ( x ) d \gamma ( x , y )$ and $\begin{array} { r } { \int _ { \Omega } z _ { 1 } ( y ) \frac { d \gamma _ { 1 } } { d \nu } ( y ) d \nu ( y ) = \int z _ { 1 } ( y ) d \gamma ( x , y ) } \end{array}$ , the entropy–transport objective (2.1) minus the dual objective (2.2) equals the sum of the corresponding integrals against $\mu$ and $\nu ,$ together with

$$
\int _ { \Omega \times \Omega } \bigl ( c _ { \mathsf { H K } } ( x , y ) - z _ { 0 } ( x ) - z _ { 1 } ( y ) \bigr ) d \gamma ( x , y ) .
$$

All three terms are nonnegative, and their sum is zero at optimality. Hence equality holds in both Fenchel inequalities, giving

$$
h = e ^ { - z _ { 0 } } = 1 - \psi , \qquad \frac { d \gamma _ { 1 } } { d \nu } = e ^ { - z _ { 1 } } = 1 - \chi ,
$$

$\mu -$ and ν-almost everywhere, respectively, by $\left( \mathrm { C . 1 1 } \right)$ . Combining these identities with $h = q$ cos r and the previously established relation $( d \gamma _ { 1 } / d \nu ) \circ T = h / q ^ { 2 }$ yields

$$
1 - \psi = q \cos r , \qquad 1 - \chi ( T ( x ) ) = { \frac { h ( x ) } { q ( x ) ^ { 2 } } } = { \frac { \cos r ( x ) } { q ( x ) } } \quad { \mathrm { f o r ~ } } \mu { \mathrm { - a . e . ~ } } \ x .
$$

Finally, the lower bounds in (C.15) give

$$
\cos ^ { 2 } r ( x ) = ( 1 - \psi ( x ) ) ( 1 - \chi ( T ( x ) ) ) \geq \delta ^ { 2 }
$$

for some $0 < \delta < 1$ . Thus $r ( x ) \leq$ arccos $\delta < \pi / 2$ for $\mu \mathrm { - a . e . } ~ x$ , which proves the uniform bound below the cutof. □

Corollary C.9 (Discrete HK identity for the source-side dual potential). Assume the hypotheses ofLemma C.7. Let ψ be its unique saturated source component, let $( T , q )$ be the associated Monge pair, and set $\phi = \psi / 2$ . Then

$$
\int _ { \Omega } \left( | \nabla \phi | ^ { 2 } + 4 | \phi | ^ { 2 } \right) d \mu = \mathsf { W } ^ { 2 } ( \mu , \nu ) .\tag{C.17}
$$

Proof. We prove the identity by first expressing the gradient of the source dual potential in terms of the Monge pair. The derivation is slightly diferent in Euclidean space and on the torus, but both give the same pointwise formula.

Let $r = \mathsf { d } _ { \Omega } ( x , T ( x ) ) < \pi / 2$ , and let $e ( x )$ be the initial unit tangent at x of the minimizing geodesic running from x to $T ( x )$ , with $e \ = \ 0$ when $r \ = \ 0 ;$ in the Euclidean case $e ( x ) \ =$ $( T ( x ) - x ) / | T ( x ) - x |$ . Fix a point at which the Lipschitz function $\psi$ is diferentiable.

At an interior diferentiability point x, fix $y = T ( x )$ . By (C.14), the function

$$
x ^ { \prime } \longmapsto 1 - { \frac { K ( x ^ { \prime } , y ) } { 1 - \chi ( y ) } } - \psi ( x ^ { \prime } )
$$

is nonnegative and vanishes at $x ^ { \prime } = x$ . Its gradient therefore vanishes there, giving

$$
\nabla \psi ( x ) = - \frac { \nabla _ { x } K ( x , T ( x ) ) } { 1 - \chi ( T ( x ) ) } .
$$

For $r > 0$ , we have $K ( x , y ) = \cos ^ { 2 } r$ and $\nabla _ { x } \mathsf { d } _ { \Omega } ( x , y ) = - e ( x )$ at $y = T ( x )$ . The chain rule therefore gives

$$
\nabla _ { x } K ( x , T ( x ) ) = - 2 \cos r \sin r \nabla _ { x } \mathrm { d } _ { \Omega } ( x , T ( x ) ) = 2 \cos r \sin r e ( x ) .
$$

The resulting formula also holds when $r = 0$ , since $K ( \cdot , y )$ is diferentiable at $x = y$ with zero gradient. Inserting this together with $1 - \chi ( T ( x ) ) = \cos r / q$ from (C.16),

$$
\nabla \psi ( x ) = - { \frac { 2 \cos r \sin r e ( x ) } { \cos r / q } } = - 2 q ( x ) \sin r ( x ) e ( x ) .
$$

For $\Omega = \mathbb { T } ^ { d } .$ set $z = - \log ( 1 - \psi )$ , which is the logarithmic potential $z _ { \mathrm { 0 } }$ of (C.11). The Monge formula in [12, Section 3.4.1] gives, at almost every diferentiability point of z,

$$
T ( x ) = \exp _ { x } \left( - r ( x ) { \frac { \nabla z ( x ) } { | \nabla z ( x ) | } } \right) , \quad r ( x ) = \mathrm { d } _ { \Omega } ( x , T ( x ) ) = \arctan \left( { \frac { | \nabla z ( x ) | } { 2 } } \right) , \quad q ( x ) = { \frac { e ^ { - z ( x ) } } { \cos { r ( x ) } } } ,
$$

with the evident convention when $\nabla z ( x ) = 0$ . Thus $e = - \nabla z / | \nabla z |$ and, from $| \boldsymbol { \nabla } z | = 2$ tan r and $e ^ { - z } = q \cos r ,$

$$
\nabla \psi = e ^ { - z } \nabla z = - q \cos r | \nabla z | e = - 2 q \sin r e .
$$

Hence the same gradient identity holds on both state spaces, with the Riemannian gradient on the torus. By (C.16), $\psi = 1 - q \cos r$ . Therefore

$$
| \nabla \phi | ^ { 2 } = q ^ { 2 } \sin ^ { 2 } r , \qquad 4 | \phi | ^ { 2 } = ( 1 - q \cos r ) ^ { 2 } ,
$$

and hence

$$
| \nabla \phi | ^ { 2 } + 4 | \phi | ^ { 2 } = 1 + q ^ { 2 } - 2 q \cos r .
$$

Integrating and using (2.10) proves (C.17).

Lemma C.10 (Euler–Lagrange identity for the fully implicit HK-JKO step). Let $\mu ~ = ~ \rho$ dx minimize

$$
\mathcal { F } ( \sigma ) + \frac { 1 } { 2 \tau } \mathsf { H } ^ { 2 } ( \sigma , \nu ) ,
$$

assume Assumption 2.2, $V \in C ( \Omega ) , ( \mathbf { W 0 } )$ , and $\nu = \widetilde { \rho } d x$ with $0 < m \leq \rho , \tilde { \rho } \leq M$ . Let ϕ be the source-side dual potential from Corollary C.9. Then

$$
U ^ { \prime } ( \rho ) + V + W [ \rho ] = - { \frac { \phi } { \tau } } \quad \quad a . e . o n \Omega .\tag{C.18}
$$

In particular, the full chemical potential $g _ { \rho }$ has a globally Lipschitz representative.

Proof. We first compute the directional derivative of the squared HK term with respect to its first argument. The key point is that the saturated source dual potentials for the perturbed measures converge uniformly to the one at the minimizer.

For $h \in L ^ { \infty } ( \Omega )$ , the positive lower and upper density bounds permit the two-sided perturbation $\rho _ { \varepsilon } = \rho + \varepsilon h$ for suficiently small |ε|. Set $G ( \sigma ) = { \mathsf { H K } } ^ { 2 } ( \sigma , \nu )$ . There are then $0 < m ^ { \prime } \leq M ^ { \prime } <$ $\infty$ , independent of $\varepsilon ,$ with $m ^ { \prime } \le \rho _ { \varepsilon } \le M ^ { \prime }$ for all small |ε|. Let $( \psi _ { \varepsilon } , \chi _ { \varepsilon } )$ be saturated optimal transformed dual pairs for $( \rho _ { \varepsilon } d x , \nu )$ , and let $( \psi , \chi )$ be one for $( \mu , \nu )$ . For $\varepsilon > 0$ , using $( \psi , \chi )$ as a competitor at $\rho _ { \varepsilon }$ dx and $( \psi _ { \varepsilon } , \chi _ { \varepsilon } )$ as a competitor at $\mu$ in (C.12) gives the two envelope bounds

$$
\varepsilon \int _ { \Omega } \psi h d x \leq G ( \rho _ { \varepsilon } d x ) - G ( \mu ) \leq \varepsilon \int _ { \Omega } \psi _ { \varepsilon } h d x ,
$$

because the ν-integrals cancel in each comparison. By Lemma C.7, both components $( \psi _ { \varepsilon } , \chi _ { \varepsilon } )$ are uniformly bounded in $W ^ { 1 , \infty } ( \Omega )$ . By Arzel\`a–Ascoli on the compact space $\Omega ,$ every sequence $\varepsilon _ { n } \downarrow 0$ admits a subsequence along which the pair converges uniformly to some $( \psi _ { * } , \chi _ { * } )$

The envelope bounds above and the uniform sup-norm bounds imply $G ( \rho _ { \varepsilon _ { n } } d x ) \to G ( \mu )$ Together with $\rho _ { \varepsilon _ { n } } \to \rho$ in $L ^ { \infty } ( \Omega )$ and the uniform convergence of $\left( \psi _ { \varepsilon _ { n } } , \chi _ { \varepsilon _ { n } } \right)$ , this allows us to pass to the limit and obtain

$$
G ( \mu ) = \int _ { \Omega } \psi _ { * } d \mu + \int _ { \Omega } \chi _ { * } d \nu .
$$

The dual constraint also passes to the limit, so $( \psi _ { * } , \chi _ { * } )$ is feasible for $( \mu , \nu ) ;$ ; hence the equality above shows that it is optimal. Finally, the uniform positive lower bounds on $1 - \psi _ { \varepsilon _ { n } }$ and $1 - \chi _ { \varepsilon _ { n } }$ , together with uniform convergence, allow passage to the saturation identities $\left( \mathrm { C . 1 4 } \right)$

By Lemma C.7, uniqueness of the saturated optimal source component gives $\psi _ { * } = \psi _ { } \mu _ { - }$ almost everywhere. Since both functions are continuous and $\mu$ has full support, they agree everywhere. Every subsequential source limit is thus $\psi ,$ proving $\psi _ { \varepsilon }  \psi$ uniformly on Ω as $\varepsilon \downarrow 0$ . Therefore

$$
\operatorname* { l i m } _ { \varepsilon \downarrow 0 } \frac { G ( \rho _ { \varepsilon } d x ) - G ( \mu ) } { \varepsilon } = \int _ { \Omega } \psi h d x .
$$

Applying the same argument to −h yields the left derivative, so that

$$
\left. \frac { d } { d \varepsilon } \right| _ { \varepsilon = 0 } G ( \rho _ { \varepsilon } d x ) = \int _ { \Omega } \psi h d x = 2 \int _ { \Omega } \phi h d x .
$$

Therefore

$$
\left. \frac { d } { d \varepsilon } \right| _ { \varepsilon = 0 } \frac { 1 } { 2 \tau } \mathsf { H K } ^ { 2 } ( \rho _ { \varepsilon } d x , \nu ) = \frac { 1 } { \tau } \int _ { \Omega } \phi h d x .
$$

With the proximal derivative identified, we now diferentiate the full energy. Minimality for arbitrary bounded perturbations then yields the Euler–Lagrange identity.

The perturbed densities remain in a compact subset of $( 0 , \infty )$ , while $V$ and W are bounded. Thus diferentiation under the integral and symmetry of W give

$$
\frac { d } { d \varepsilon } \bigg \vert _ { \varepsilon = 0 } \mathcal { F } ( \rho _ { \varepsilon } d x ) = \int _ { \Omega } \bigl ( U ^ { \prime } ( \rho ) + V + W [ \rho ] \bigr ) h d x .
$$

Minimality for all bounded h yields (C.18).

Corollary C.11 (Discrete metric–dissipation identity). Under the hypotheses of Corollary C.9 and Lemma C.10,

$$
{ \sf H } ^ { 2 } ( \rho d x , \nu ) = \tau ^ { 2 } I _ { g } ( \rho ) .
$$

Proof. By (C.18), $\phi = - \tau g _ { \rho }$ . Substitute into (C.17).

Proof of Theorem 2.6. We now assemble the preceding results. We first obtain the forward Monge statement and the two-sided density bounds; these are the inputs needed to invoke the dual regularity theory for the reversed pair.

Proposition 2.5 and Lemma C.4 give part (i). By Theorem $2 . 3 , \mu _ { 1 } ( \Omega ) \leq M _ { 1 }$ , so

$$
\| P _ { 1 } \| _ { \infty } \leq B _ { 1 } , \qquad \mathrm { e s s o s c } P _ { 1 } \leq \omega _ { 1 } .
$$

In Proposition C.5, choose $c _ { \mathrm { u p p } } = ( U ^ { \prime } ) ^ { - 1 } ( B _ { 1 } )$ . Then $U ^ { \prime } ( c _ { \mathrm { u p p } } )$ +ess inf<sub>Ω</sub> $P _ { 1 } = B _ { 1 } +$ ess inf<sub>Ω</sub> $P _ { 1 } \geq 0$ because ess inf<sub>Ω</sub> $P _ { 1 } \ge - B _ { 1 }$ , so the minimum in (C.4) vanishes and $\Lambda = 1$ . With $c _ { \mathrm { m a x } } = C _ { 0 }$ (C.5) then reads $\rho _ { 1 } \le ( U ^ { \prime } ) ^ { - 1 } ( \operatorname* { m a x } \{ B _ { 1 } , U ^ { \prime } ( C _ { 0 } ) \}$ + ess osc<sub>Ω</sub> P<sub>1</sub>), which is the upper bound in (2.12). In Proposition $\mathrm { C . 6 } ,$ choose $c _ { \mathrm { l o w } } = c _ { * } = ( U ^ { \prime } ) ^ { - 1 } ( - B _ { 1 } - 1 )$ . Since ess inf<sub>Ω</sub> $P _ { 1 } \leq B _ { 1 }$ , one has $\begin{array} { r } { U ^ { \prime } ( c _ { * } ) + \mathrm { e s s } \operatorname* { i n f } _ { \Omega } P _ { 1 } \leq - 1 < 0 , } \end{array}$ , so $c _ { * }$ is admissible there, and $\bar { c } = \operatorname* { m i n } \{ c _ { 0 } , c _ { * } \}$ gives the lower bound.

We next work with the reversed ordering $( \mu _ { 1 } , \mu _ { 0 } )$ . The two-sided bounds just proved allow us to identify the discrete Euler equation and a Lipschitz representative of the chemical potential.

Apply Lemma C.7, Corollary C.9, and Lemma C.10 to the ordered pair $( \mu _ { 1 } , \mu _ { 0 } )$ , whose two densities are two-sided bounded by part (ii) and by hypothesis, respectively. Note that the Monge pair used in this second half runs from $\mu _ { 1 }$ to $\mu _ { 0 }$ and is therefore not the pair of part (i). Equation (C.18) gives (2.13), so that $\Phi _ { 1 } = - \phi _ { 1 } / \tau$ has a Lipschitz representative. Its weak gradient defines $I _ { g } ( \rho _ { 1 } )$ without requiring separate spatial derivatives of $V$ or $W [ \rho _ { 1 } ]$

We combine the elementary one-step energy inequality with the exact metric identity to obtain the dissipation statement. Since $\mathcal { F } ( \bar { \mu } ) < \infty$ , the minimality gives

$$
\mathcal { F } ( \mu _ { 1 } ) + \frac { 1 } { 2 \tau } \mathsf { H } ^ { \ast 2 } ( \mu _ { 1 } , \bar { \mu } ) = \mathcal { I } _ { \tau } ( \mu _ { 1 } \mid \bar { \mu } ) \leq \mathcal { I } _ { \tau } ( \bar { \mu } \mid \bar { \mu } ) = \mathcal { F } ( \bar { \mu } ) .\tag{C.19}
$$

Corollary C.11 gives the metric identity, and combining it with (C.19) gives the last inequality. Finally, if $V ~ \in ~ W ^ { 1 , \infty } ( \Omega )$ and (W1) holds, then $P _ { 1 }$ is Lipschitz, since $\mathrm { L i p } ( W [ \rho _ { 1 } ] ) ~ \leq$ $\mathrm { L i p } ( W ) \mu _ { 1 } ( \Omega )$ . By Lemma $\mathrm { C } . 7 , \ \phi _ { 1 }$ is Lipschitz, so $U ^ { \prime } ( \rho _ { 1 } ) = - \phi _ { 1 } / \tau - P _ { 1 }$ is Lipschitz. The continuous positive function $U ^ { \prime \prime }$ has a positive minimum on $[ \underline { { \rho } } _ { 1 } , \overline { { \rho } } _ { 1 } ]$ , hence $( U ^ { \prime } ) ^ { - 1 }$ is Lipschitz on $U ^ { \prime } ( [ \underline { { \rho } } _ { 1 } , \overline { { \rho } } _ { 1 } ] )$ . Solving (2.13) gives $\rho _ { 1 } \in W ^ { 1 , \infty } ( \Omega )$ under these additional assumptions. □

## D Proof of Theorem 5.2 and Corollary D.2 (Theorem C)

We first record the elementary fact about inverses that both proofs of this appendix use.

Lemma D.1 (Uniform convergence of inverses). Let Ω be compact, let $S , S _ { n } : \Omega  \Omega$ be continuous bijections of Ω onto itself, and suppose $S _ { n } \ \to \ S$ uniformly. Then $S _ { n } ^ { - 1 } \to S ^ { - 1 }$ uniformly.

Proof. We prove the claim by contradiction. A failure of uniform convergence of the inverse maps would produce convergent preimages whose limit violates injectivity of S.

Both S and the $S _ { n }$ are homeomorphisms, being continuous bijections of a compact Hausdorf space. Otherwise, some subsequence satisfies $\mathsf { d } _ { \Omega } ( S _ { n } ^ { - 1 } ( y _ { n } ) , S ^ { - 1 } ( y _ { n } ) ) \geq \delta$ for a fixed $\delta > 0$ and points $y _ { n } \in \Omega$ . By compactness we may assume $y _ { n }  y$ and $x _ { n } : = S _ { n } ^ { - 1 } ( y _ { n } ) \ \to \ x .$ Then $\mathrm { d } _ { \Omega } ( S ( x _ { n } ) , y _ { n } ) \ = \ \mathrm { d } _ { \Omega } ( S ( x _ { n } ) , S _ { n } ( x _ { n } ) ) \ \le \ \| S - S _ { n } \| _ { \infty } \ \to \ 0$ , so $S ( x ) \ = \ y$ by continuity, that is, $x = S ^ { - 1 } ( y )$ Since $S ^ { - 1 }$ is continuous, $S ^ { - 1 } ( y _ { n } )  S ^ { - 1 } ( y ) = x$ as well, contradicting ${ \mathsf { d } } _ { \Omega } ( x _ { n } , S ^ { - 1 } ( y _ { n } ) ) \geq \delta$ □

## D.1 One step

Proof of Theorem 5.2. We first show that neural approximation of the regular optimal pair implies uniform recovery of the induced density. We then pass to the pair objective, and finally use primal majorization to recover the original JKO objective and the cone action.

Let $( T _ { * } , q _ { * } )$ be the regular optimal Monge-growth pair. By Assumption 5.1, there are neural pairs $\left( T _ { \theta _ { n } } , q _ { \theta _ { n } } \right)$ such that

$$
T _ { \theta _ { n } } \to T _ { * } \quad \mathrm { i n ~ } C ^ { 1 } ( \Omega ) , \qquad q _ { \theta _ { n } } \to q _ { * } \quad \mathrm { u n i f o r m l y } .
$$

Since ${ { J } _ { { { T } _ { * } } } }$ is bounded away from zero and $D T _ { \theta _ { n } }  D T _ { * }$ uniformly, the same is true for ${ { J } _ { { { T } _ { \theta _ { n } } } } }$ for all suficiently large n. Assumption 5.1 makes $T _ { * }$ and every $T _ { \theta _ { n } }$ a bijection of Ω onto itself, so

Lemma D.1 applies and $T _ { \theta _ { n } } ^ { - 1 }  T _ { * } ^ { - 1 }$ uniformly; in particular $u _ { \theta _ { n } }$ below is defined at every point of Ω. If

$$
( T _ { \theta _ { n } } ) _ { \# } ( q _ { \theta _ { n } } ^ { 2 } \bar { \rho } d x ) = u _ { \theta _ { n } } d x ,
$$

then (4.1) gives, at a common point $y .$

$$
u _ { \theta _ { n } } ( y ) = \frac { q _ { \theta _ { n } } ( T _ { \theta _ { n } } ^ { - 1 } ( y ) ) ^ { 2 } \bar { \rho } ( T _ { \theta _ { n } } ^ { - 1 } ( y ) ) } { J _ { T _ { \theta _ { n } } } ( T _ { \theta _ { n } } ^ { - 1 } ( y ) ) } .
$$

The same formula holds for $\rho _ { * }$ with $( T _ { * } , q _ { * } )$ . Hence

$$
\| u _ { \theta _ { n } } - \rho _ { * } \| _ { L ^ { \infty } ( \Omega ) } \longrightarrow 0 .
$$

This proves (5.2) for all suficiently large n.

Having recovered the state uniformly, we now pass to the surrogate objective. Uniform convergence of maps, amplitudes, densities, and Jacobians is enough to pass to every term in $\mathcal { T } _ { \tau }$

The same change-of-variables formula and the positive continuous source put $\rho _ { * }$ and the approximating densities in a fixed positive compact interval. Finiteness and convexity of U imply its continuity there. Continuity of U, V , and $W$ on the relevant compact sets, uniform convergence of the maps and amplitudes, and convergence of the Jacobians imply convergence of the three energy terms in (4.2). The integrand in (1.2) also converges uniformly. Therefore

$$
{ \cal T } _ { \tau } ( T _ { \theta _ { n } } , q _ { \theta _ { n } } \mid \bar { \mu } ) \longrightarrow { \cal T } _ { \tau } ( T _ { * } , q _ { * } \mid \bar { \mu } ) .
$$

By Proposition 3.1,

$$
\begin{array} { r } { \mathcal { T } _ { \tau } ( T _ { * } , q _ { * } \mid \bar { \mu } ) = \mathcal { T } _ { \tau } ( \mu _ { * } \mid \bar { \mu } ) , } \end{array}
$$

and every admissible competitor has objective at least this minimum. Thus the diference in (5.3) is nonnegative and converges to zero.

We finally compare the surrogate value with the original JKO value. The primal upper bound sandwiches the exact JKO objective between the optimum and the surrogate objective, and the same comparison controls the excess cone action.

Since $\nu _ { \theta _ { n } } = u _ { \theta _ { n } }$ dx, the minimality and the primal majorization give

$$
\mathcal { T } _ { \tau } ( \mu _ { * } \mid \bar { \mu } ) \leq \mathcal { T } _ { \tau } ( \nu _ { \theta _ { n } } \mid \bar { \mu } ) \leq \mathcal { T } _ { \tau } ( T _ { \theta _ { n } } , q _ { \theta _ { n } } \mid \bar { \mu } ) .
$$

Hence the original JKO gap is bounded by the pair-objective gap. Moreover,

$$
\begin{array} { r l } & { 0 \leq A _ { \bar { \mu } } ( T _ { \theta _ { n } } , q _ { \theta _ { n } } ) - { \sf H K } ^ { 2 } ( \bar { \mu } , \nu _ { \theta _ { n } } ) } \\ & { ~ = 2 \tau \left[ \mathcal { T } _ { \tau } ( T _ { \theta _ { n } } , q _ { \theta _ { n } } \mid \bar { \mu } ) - \mathcal { T } _ { \tau } ( \nu _ { \theta _ { n } } \mid \bar { \mu } ) \right] } \\ & { ~ \leq 2 \tau \left[ \mathcal { T } _ { \tau } ( T _ { \theta _ { n } } , q _ { \theta _ { n } } \mid \bar { \mu } ) - \mathcal { T } _ { \tau } ( \mu _ { * } \mid \bar { \mu } ) \right] . } \end{array}
$$

Choosing n suficiently large proves (5.2)–(5.5).

## D.2 A fixed finite horizon

Corollary D.2 (Finite-horizon composition of the one-step recovery). Fix a horizon $\tau > 0$ and $0 < \tau \leq \tau$ , let $N _ { \tau } = \lfloor \tau / \tau \rfloor$ , and let $\mu _ { k + 1 } ^ { \tau } = S _ { \tau } ( \mu _ { k } ^ { \tau } ) = \rho _ { k + 1 } ^ { \tau } d x , 0 \leq k < N _ { \tau }$ , be exact ${ \sf H } { \sf - } J K O$ iterates satisfying Theorem 2.6, with $\rho _ { 0 } ^ { \tau }$ positive and continuous. For each k choose an optimal Monge-growth pair $( T _ { k + 1 } ^ { * } , q _ { k + 1 } ^ { * } )$ from $\mu _ { k } ^ { \tau }$ to $\mu _ { k + 1 } ^ { \tau }$ , and suppose that every chosen pair satisfies all of Assumption $5 . 1 ,$ including the network recovery property (5.1).

Then there exist neural pairs $( T _ { \theta _ { n , k + 1 } } , q _ { \theta _ { n , k + 1 } } )$ and positive continuous densities generated recursively from the current neural measure, with $u _ { n , 0 } ^ { \tau } : = \rho _ { 0 } ^ { \tau }$ ，

$$
\widehat { \mu } _ { n , 0 } ^ { \tau } = \mu _ { 0 } ^ { \tau } = u _ { n , 0 } ^ { \tau } d x , \qquad \widehat { \mu } _ { n , k + 1 } ^ { \tau } = ( T _ { \theta _ { n , k + 1 } } ) _ { \# } \bigl ( q _ { \theta _ { n , k + 1 } } ^ { 2 } \widehat { \mu } _ { n , k } ^ { \tau } \bigr ) = u _ { n , k + 1 } ^ { \tau } d x ,\tag{D.1}
$$

such that the pairs and the represented states converge,

$$
\operatorname* { m a x } _ { 0 \leq k \leq N _ { \tau } } \| u _ { n , k } ^ { \tau } - \rho _ { k } ^ { \tau } \| _ { L ^ { \infty } ( \Omega ) } + \operatorname* { m a x } _ { 0 \leq k < N _ { \tau } } \big ( \| T \theta _ { n , k + 1 } - T _ { k + 1 } ^ { * } \| _ { C ^ { 1 } ( \Omega ) } + \| q _ { \theta _ { n , k + 1 } } - q _ { k + 1 } ^ { * } \| _ { L ^ { \infty } ( \Omega ) } \big ) \longrightarrow 0 ,\tag{D.2}
$$

and both variational objectives are recovered uniformly over the horizon:

$$
\begin{array} { r l } & { \underset { 0 \leq k < N _ { \tau } } { \operatorname* { m a x } } \left| \mathcal { T } _ { \tau } ( T _ { \theta _ { n , k + 1 } } , q _ { \theta _ { n , k + 1 } } \mid \widehat { \mu } _ { n , k } ^ { \tau } ) - \mathcal { I } _ { \tau } ( \mu _ { k + 1 } ^ { \tau } \mid \mu _ { k } ^ { \tau } ) \right| \longrightarrow 0 , } \\ & { \underset { 0 \leq k < N _ { \tau } } { \operatorname* { m a x } } \left| \mathcal { I } _ { \tau } ( \widehat { \mu } _ { n , k + 1 } ^ { \tau } \mid \widehat { \mu } _ { n , k } ^ { \tau } ) - \mathcal { I } _ { \tau } ( \mu _ { k + 1 } ^ { \tau } \mid \mu _ { k } ^ { \tau } ) \right| \longrightarrow 0 . } \end{array}\tag{D.3}
$$

The cone action and its transport and reaction components converge likewise, uniformly in $k ,$ to $\tau ^ { 2 } I _ { g } ( \rho _ { k + 1 } ^ { \tau } )$ and to $\mathcal { A } _ { \mu _ { k } ^ { \tau } } ^ { \sharp } ( T _ { k + 1 } ^ { * } , q _ { k + 1 } ^ { * } )$ for $\sharp \in \{ \mathrm { t r } , \mathrm { r e } \}$

Proof of Corollary D.2. The finite-horizon statement is obtained by synchronizing the one-step approximations over the finitely many steps and then propagating the state error recursively. After the states are controlled, the objective and action limits follow from the same continuity arguments as in the one-step result.

For each $0 \leq k < N _ { \tau } ,$ Assumption 5.1 supplies neural pairs such that

$$
{ \cal T } _ { \theta _ { n , k + 1 } } \longrightarrow { \cal T } _ { k + 1 } ^ { * } \mathrm { i n } C ^ { 1 } ( \Omega ) , q _ { \theta _ { n , k + 1 } } \longrightarrow q _ { k + 1 } ^ { * } \mathrm { u n i f o r m l y } .
$$

Since the horizon contains finitely many steps, the sequences may be indexed by one common n so that these convergences hold simultaneously for every k.

We first propagate the density approximation. The induction uses the change-of-variables formula together with uniform convergence of the inverse maps and Jacobians.

Set $\widehat { \mu } _ { n , 0 } ^ { \tau } = \mu _ { 0 } ^ { \tau }$ and define the recursion in (D.1). We prove the density convergence by induction. Suppose that $u _ { n , k } ^ { \tau } \to \rho _ { k } ^ { \tau }$ uniformly. The change-of-variables formula gives, at $y \in \Omega$

$$
u _ { n , k + 1 } ^ { \tau } ( y ) = \frac { q _ { \theta _ { n , k + 1 } } ( T _ { \theta _ { n , k + 1 } } ^ { - 1 } ( y ) ) ^ { 2 } u _ { n , k } ^ { \tau } ( T _ { \theta _ { n , k + 1 } } ^ { - 1 } ( y ) ) } { J _ { T _ { \theta _ { n , k + 1 } } } ( T _ { \theta _ { n , k + 1 } } ^ { - 1 } ( y ) ) } .
$$

The same identity holds for $\rho _ { k + 1 } ^ { \tau }$ with $( u _ { n , k } ^ { \tau } , T _ { \theta _ { n , k + 1 } } , q _ { \theta _ { n , k + 1 } } )$ replaced by $( \rho _ { k } ^ { \tau } , T _ { k + 1 } ^ { * } , q _ { k + 1 } ^ { * } )$ . The Jacobians stay uniformly away from zero and the inverse maps converge uniformly. Hence $u _ { n , k + 1 } ^ { \tau } \to \rho _ { k + 1 } ^ { \tau }$ uniformly. The induction starts at $k = 0 ,$ , and finiteness of the horizon yields (D.2). In particular, for all suficiently large n, all source and induced densities lie in one positive compact interval.

With uniform state convergence established at every step, we can pass to the pair objective and the cone action step by step; finiteness of the horizon makes the convergence uniform in k.

Uniform convergence of the densities and amplitudes, $C ^ { 1 }$ convergence of the maps, and the formulas (4.2) and (1.2) imply

$$
\begin{array} { r l } & { \mathcal { T } _ { \tau } ( T _ { \theta _ { n , k + 1 } } , q _ { \theta _ { n , k + 1 } } \mid \widehat { \mu } _ { n , k } ^ { \tau } ) \longrightarrow \mathcal { T } _ { \tau } ( T _ { k + 1 } ^ { * } , q _ { k + 1 } ^ { * } \mid \mu _ { k } ^ { \tau } ) , } \\ & { \mathcal { A } _ { \widehat { \mu } _ { n , k } ^ { \tau } } ( T _ { \theta _ { n , k + 1 } } , q _ { \theta _ { n , k + 1 } } ) \longrightarrow \mathcal { A } _ { \mu _ { k } ^ { \tau } } ( T _ { k + 1 } ^ { * } , q _ { k + 1 } ^ { * } ) . } \end{array}
$$

The selected exact pair is optimal, so Proposition 3.1 and (2.14) give

$$
\begin{array} { r l } & { \mathcal { T } _ { \tau } ( T _ { k + 1 } ^ { * } , q _ { k + 1 } ^ { * } \mid \mu _ { k } ^ { \tau } ) = \mathcal { T } _ { \tau } ( \mu _ { k + 1 } ^ { \tau } \mid \mu _ { k } ^ { \tau } ) , } \\ & { \quad \mathcal { A } _ { \mu _ { k } ^ { \tau } } ( T _ { k + 1 } ^ { * } , q _ { k + 1 } ^ { * } ) = \mathsf { H K } ^ { 2 } ( \mu _ { k } ^ { \tau } , \mu _ { k + 1 } ^ { \tau } ) = \tau ^ { 2 } I _ { g } ( \rho _ { k + 1 } ^ { \tau } ) . } \end{array}
$$

This proves the first limit in (D.3) together with the cone-action convergence asserted at the end of the corollary.

We next recover the original JKO objective. Uniform density convergence gives convergence in HK, and the triangle inequality transfers this to the distance between consecutive approximate states.

By (2.5), uniform density convergence implies $\mathsf { H K } ( \widehat { \mu } _ { n , k } ^ { \tau } , \mu _ { k } ^ { \tau } ) \to 0$ for every k. Since HK is a metric, the triangle inequality gives

$$
\big | \mathsf { W } ( \widehat { \mu } _ { n , k + 1 } ^ { \tau } , \widehat { \mu } _ { n , k } ^ { \tau } ) - \mathsf { W } ( \mu _ { k + 1 } ^ { \tau } , \mu _ { k } ^ { \tau } ) \big | \leq \mathsf { H K } ( \widehat { \mu } _ { n , k + 1 } ^ { \tau } , \mu _ { k + 1 } ^ { \tau } ) + \mathsf { H K } ( \widehat { \mu } _ { n , k } ^ { \tau } , \mu _ { k } ^ { \tau } ) \longrightarrow 0 .
$$

The energy terms converge by uniform density convergence, so the original JKO objectives converge as well. This proves the second limit in (D.3).

It remains only to separate the limiting cone action into its transport and reaction components. This follows directly from uniform convergence of the two split integrands and of the source densities.

The two integrands in the splitting (3.6) converge uniformly, while the source densities converge uniformly; hence the transport and reaction components converge as asserted at the end of the corollary. □

## E Proof of Theorem 5.4 and Corollary 5.5 (Theorem D)

Proof of Theorem $5 . 4 .$ We prove the certificate in two stages. First we construct a computable lower bound for the exact JKO minimum from a feasible dual pair and the tangent inequality for the positive-semidefinite interaction. We then compare this lower bound with the primal cone-action value and convert the resulting objective gap into a state-error bound.

For $\sigma = s d x$ , every pair satisfying (5.6) is a competitor in (2.2), and therefore

$$
\mathsf { H } ^ { 2 } ( \sigma , \bar { \mu } ) \geq \int _ { \Omega } \bigl ( 1 - e ^ { - \zeta } \bigr ) s d x + \int _ { \Omega } \bigl ( 1 - e ^ { - \xi } \bigr ) d \bar { \mu } .\tag{E.1}
$$

In particular, $\sigma \mapsto { \mathsf { H K } } ^ { 2 } ( \sigma , { \bar { \mu } } )$ is convex in the flat structure of $\mathcal { M } _ { + } ( \Omega )$ , since it is a supremum of afine functionals. Positive semidefiniteness makes W convex, and $U ^ { \prime \prime } > 0$ makes $\scriptstyle \int U$ strictly convex on its finite domain. Existence from Theorem 2.3 thus gives uniqueness of $\mu _ { 1 }$ .

We now linearize the interaction energy at the candidate measure $\nu .$ Positive semidefiniteness is precisely what makes the quadratic interaction convex and validates this tangent lower bound.

The tangent inequality for the convex quadratic form at $\nu$ is

$$
\mathcal { W } ( \sigma ) \geq \int _ { \Omega } W [ u ] s d x - \mathcal { W } ( \nu ) .
$$

Indeed, setting $f ( t ) : = \mathcal { W } \big ( ( 1 - t ) \nu + t \sigma \big )$ , the positive semidefiniteness of W makes $f$ convex, so that $f ( 1 ) \geq f ( 0 ) + f ^ { \prime } ( 0 )$ . Since

$$
f ^ { \prime } ( 0 ) = \int _ { \Omega } W [ u ] ( s - u ) d x \quad { \mathrm { a n d } } \quad \int _ { \Omega } W [ u ] u d x = 2 \mathcal { W } ( \nu ) ,
$$

this gives precisely the displayed inequality.

Combining the metric minorant with the interaction tangent bound gives a pointwiseseparable lower bound for every competitor. Minimizing that lower bound through the convex conjugate of U produces the dual certificate.

Combining the interaction tangent inequality with (E.1) and using the definition of a in (5.7) gives

$$
\mathcal { I } _ { \tau } ( \sigma \mid \bar { \mu } ) \geq \int _ { \Omega } [ U ( s ) + a s ] d x - \mathcal { W } ( \nu ) + \frac { 1 } { 2 \tau } \int _ { \Omega } ( 1 - e ^ { - \xi } ) d \bar { \mu } .
$$

We now optimize the preceding lower bound in the density variable and compare it with the primal upper certificate supplied by $( T , q )$

Since in $\mathfrak { i } \mathrm { f } _ { s \ge 0 } \{ U ( s ) + a s \} = - U ^ { * } ( - a )$ pointwise,

$$
\operatorname* { m i n } _ { \sigma } \mathcal { I } _ { \tau } ( \sigma \mid \bar { \mu } ) = \mathcal { I } _ { \tau } ( \mu _ { 1 } \mid \bar { \mu } ) \geq - \int _ { \Omega } U ^ { * } ( - a ) d x - \mathcal { W } ( \nu ) + \frac { 1 } { 2 \tau } \int _ { \Omega } ( 1 - e ^ { - \xi } ) d \bar { \mu } .
$$

On the primal side, (3.10) gives $\mathcal { I } _ { \tau } ( \nu \mid \bar { \mu } ) \leq \mathcal { I } _ { \tau } ( T , q \mid \bar { \mu } )$ . Subtracting the two bounds proves (5.8) and $\begin{array} { r } { \mathcal { R } \geq 0 . \ \left( \mathrm { I f } \ \int _ { \Omega } U ^ { * } ( - a ) d x = + \infty \right. } \end{array}$ , then $\mathcal { R } = + \infty$ and the conclusions hold vacuously; the estimate is informative only when the dual choice makes this integral finite.)

We next convert the objective gap into a state error. For this purpose we use the exact step’s own normalized source dual potential and the Euler–Lagrange identity. Let $\phi _ { 1 }$ be the normalized source-side transformed dual potential in Theorem 2.6(iii), so that the coeficient

of the first argument in (C.12) is $2 \phi _ { 1 }$ ; write $\chi _ { 1 }$ for the companion target component. Since $( 2 \phi _ { 1 } , \chi _ { 1 } )$ is feasible in (C.12) for every first argument and optimal at $\mu _ { 1 }$ ，

$$
\mathsf { H } ^ { 2 } ( \sigma , \bar { \mu } ) \geq 2 \int _ { \Omega } \phi _ { 1 } s d x + \int _ { \Omega } \chi _ { 1 } d \bar { \mu } , \qquad \mathsf { H } ^ { 2 } ( \mu _ { 1 } , \bar { \mu } ) = 2 \int _ { \Omega } \phi _ { 1 } \rho _ { 1 } d x + \int _ { \Omega } \chi _ { 1 } d \bar { \mu } ,
$$

and subtracting the second from the first gives

$$
{ \sf H } ^ { 2 } ( \sigma , \bar { \mu } ) - { \sf H } ^ { 2 } ( \mu _ { 1 } , \bar { \mu } ) \geq 2 \int _ { \Omega } \phi _ { 1 } ( s - \rho _ { 1 } ) d x .
$$

Positive semidefiniteness also gives, by the tangent inequality applied at $\mu _ { 1 }$ instead of $\nu ,$

$$
\mathcal { W } ( \sigma ) - \mathcal { W } ( \mu _ { 1 } ) \geq \int _ { \Omega } W [ \rho _ { 1 } ] ( s - \rho _ { 1 } ) d x .
$$

Adding these to the exact identity $\begin{array} { r } { \int V d \sigma - \int V d \mu _ { 1 } = \int V ( s - \rho _ { 1 } ) d x } \end{array}$ and to $\begin{array} { r } { \int _ { \Omega } [ U ( s ) - U ( \rho _ { 1 } ) ] d x } \end{array}$ yields

$$
\mathcal { I } _ { \tau } ( \sigma \mid \bar { \mu } ) - \mathcal { I } _ { \tau } ( \mu _ { 1 } \mid \bar { \mu } ) \geq \int _ { \Omega } \bigl [ U ( s ) - U ( \rho _ { 1 } ) \bigr ] d x + \int _ { \Omega } \Bigl ( V + W [ \rho _ { 1 } ] + \frac { \phi _ { 1 } } { \tau } \Bigr ) ( s - \rho _ { 1 } ) d x ,
$$

and the Euler–Lagrange equation (2.13) identifies the bracket in the second integral as $- U ^ { \prime } ( \rho _ { 1 } )$ so that

$$
\mathcal { I } _ { \tau } ( \sigma \mid \bar { \mu } ) - \mathcal { I } _ { \tau } ( \mu _ { 1 } \mid \bar { \mu } ) \geq \int _ { \Omega } [ U ( s ) - U ( \rho _ { 1 } ) - U ^ { \prime } ( \rho _ { 1 } ) ( s - \rho _ { 1 } ) ] d x .
$$

Taking $\sigma = \nu$ and using (5.8) proves (5.9).

Proof of Corollary 5.5. For Boltzmann entropy, the convexity remainder becomes the relativeentropy integrand. We first compare it pointwise with squared Hellinger distance and then convert the latter into an $L ^ { 1 }$ estimate by Cauchy–Schwarz.

For $U ( s ) = s \log s - s$ one has $U ^ { \prime } ( s ) = \log s$ , so the integrand in (5.9) is

$$
U ( u ) - U ( \rho _ { 1 } ) - U ^ { \prime } ( \rho _ { 1 } ) ( u - \rho _ { 1 } ) = u \log \frac { u } { \rho _ { 1 } } - u + \rho _ { 1 } .
$$

It dominates the squared Hellinger integrand. With $t = \sqrt { u / \rho _ { 1 } }$ , the diference after division by $\rho _ { 1 }$ is

$$
2 t ^ { 2 } \log t - t ^ { 2 } + 1 - ( t - 1 ) ^ { 2 } = 2 t ( t \log t - t + 1 ) \geq 0 ,
$$

since t log $t - t + 1 \geq 0$ for $t \geq 0$ , with the continuous convention $0 \log 0 = 0$ . Hence

$$
u \log \frac { u } { \rho _ { 1 } } - u + \rho _ { 1 } \geq ( \sqrt { u } - \sqrt { \rho _ { 1 } } ) ^ { 2 } .
$$

The pointwise estimate gives Hellinger control. We now pass from Hellinger distance to $L ^ { 1 }$ using the factorization u $\mathbf { \bar { \rho } } \rho _ { 1 } = ( \sqrt { u } - \sqrt { \rho _ { 1 } } ) ( \sqrt { u } + \sqrt { \rho _ { 1 } } )$

Cauchy–Schwarz then gives

$$
\| u - \rho _ { 1 } \| _ { L ^ { 1 } } ^ { 2 } \leq \left( \int _ { \Omega } ( \sqrt { u } - \sqrt { \rho _ { 1 } } ) ^ { 2 } d x \right) \left( \int _ { \Omega } ( \sqrt { u } + \sqrt { \rho _ { 1 } } ) ^ { 2 } d x \right) \leq 2 \bigl ( \nu ( \Omega ) + \mu _ { 1 } ( \Omega ) \bigr ) \mathcal { R } .
$$

The remaining inequality in (5.10) is the mass bound (2.8).

Proposition E.1 (Decomposition of the gap). Under the hypotheses of Theorem $5 . 4 ,$

$$
\begin{array} { r l } & { \mathcal { R } = \underbrace { \displaystyle \int _ { \Omega } \left[ U ( u ) + a u + U ^ { * } ( - a ) \right] d x } _ { = : \ \mathcal { R } _ { \mathrm { E L } } \geq 0 } } \\ & { \quad \ + \underbrace { \frac { 1 } { 2 \tau } \left[ A _ { \bar { \mu } } ( T , q ) - \displaystyle \int _ { \Omega } \bigl ( 1 - e ^ { - \zeta } \bigr ) u d x - \int _ { \Omega } \bigl ( 1 - e ^ { - \xi } \bigr ) d \bar { \mu } \right] } _ { = : \ \mathcal { R } _ { \mathrm { m e t } } > 0 } . } \end{array}\tag{E.2}
$$

Proof. We verify the decomposition by direct algebra. After substitution, the candidatedependent terms cancel exactly; the two remaining pieces are nonnegative for independent reasons, namely Fenchel–Young and the primal–dual metric bounds.

Substituting a from (5.7), using $\begin{array} { r } { \int _ { \Omega } W [ u ] u d x = 2 \mathcal { W } ( \nu ) } \end{array}$ , and adding the two terms in (E.2), the contributions involving $1 - e ^ { - \zeta }$ cancel and the remaining terms are exactly R in (5.7). Moreover, $\mathcal { R } _ { \mathrm { E L } } \geq 0$ by Fenchel–Young, while

$$
A _ { \bar { \mu } } ( T , q ) \geq { \sf H } ^ { 2 } ( \bar { \mu } , \nu ) \geq \int _ { \Omega } ( 1 - e ^ { - \zeta } ) u d x + \int _ { \Omega } ( 1 - e ^ { - \xi } ) d \bar { \mu }
$$

by (1.3) and (E.1); hence $\mathcal { R } _ { \mathrm { m e t } } \geq 0$

For Boltzmann entropy and positive u one has $U ^ { \ast } ( p ) = e ^ { p }$ , so with $r = U ^ { \prime } ( u ) + a = \log u + a$ the identity $e ^ { - a } = u e ^ { - r }$ gives $\begin{array} { r } { \mathcal { R } _ { \mathrm { E L } } = \int _ { \Omega } u ( e ^ { - r } - 1 + r ) d x } \end{array}$ , which is ${ \frac { 1 } { 2 } } \int _ { \Omega } \boldsymbol { r } ^ { 2 } $ u dx to leading order. The decomposition therefore separates a discrete Euler residual from the primal–dual metric gap.

## F Design and implementation notes

This appendix records the potential-generated parametrization and its admissibility conditions, step-scaling limitations, and the implementation details for Algorithm 1 and Section 6.

## F.1 Potential-generated pairs and admissibility

Remark F.1 (Potential-generated and step-scaled pairs). Proposition 2.5 suggests parametrizing an optimal pair by one scalar network $z _ { \theta }$

$$
T _ { z _ { \theta } } ( x ) = \mathrm { e x p } _ { x } \bigg ( - \arctan \bigg ( \frac { | \nabla z _ { \theta } | } { 2 } \bigg ) \frac { \nabla z _ { \theta } } { | \nabla z _ { \theta } | } \bigg ) , \qquad q _ { z _ { \theta } } ( x ) = e ^ { - z _ { \theta } } \bigg ( 1 + \frac { | \nabla z _ { \theta } | ^ { 2 } } { 4 } \bigg ) ^ { 1 / 2 } ,\tag{F.1}
$$

with $T _ { z _ { \theta } } ( x ) = x$ when $\nabla z _ { \theta } ( x ) = 0$ . For $z _ { \theta }$ in the admissible class $\mathcal { Z } _ { \mathrm { a d m } }$ defined below, the cutof is inactive and

$$
A _ { \mu } ( T _ { z _ { \theta } , \boldsymbol { q } _ { z _ { \theta } } } ) = \int _ { \Omega } \left[ 1 - 2 e ^ { - z _ { \theta } } + e ^ { - 2 z _ { \theta } } \left( 1 + \frac { | \nabla z _ { \theta } | ^ { 2 } } { 4 } \right) \right] d \mu = \int _ { \Omega } \left( z _ { \theta } ^ { 2 } + \frac { | \nabla z _ { \theta } | ^ { 2 } } { 4 } \right) d \mu + O \big ( \| z _ { \theta } \| _ { C ^ { 1 } } ^ { 3 } \big ) .
$$

The reversed and forward orientations carry opposite leading-order signs: $z _ { \mathrm { r e v } } \approx - 2 \tau g _ { \rho _ { 1 } }$ and z<sub>fwd</sub> $\approx 2 \tau g _ { \rho 1 }$ . This motivates the scaled ansatz

$$
z _ { \theta } = \tau w _ { \theta } .\tag{F.2}
$$

For a smooth positive current density ρ with smooth $g _ { \rho } ,$ suppose that $z _ { \tau } : = 2 \tau g _ { \rho }$ belongs to $\mathcal { Z } _ { \mathrm { a d m } }$ for all suficiently small $\tau .$ . Then the explicit forward predictor gives, to first order,

$$
T _ { z _ { \tau } } = \mathrm { i d } - \tau \nabla g _ { \rho } + O ( \tau ^ { 2 } ) , \qquad q _ { z _ { \tau } } ^ { 2 } = 1 - 4 \tau g _ { \rho } + O ( \tau ^ { 2 } ) ,
$$

and therefore

$$
( T _ { z _ { \tau } } ) _ { \# } ( q _ { z _ { \tau } } ^ { 2 } \rho d x ) = \left[ \rho + \tau \{ \mathrm { d i v } ( \rho \nabla g _ { \rho } ) - 4 \rho g _ { \rho } \} \right] d x + o ( \tau )
$$

in the distributional sense. This is a first-order predictor calculation, not a consistency estimate for the exact minimizer. The admissibility and approximation requirements are stated below; Section F.2 discusses step scaling and step uniformity.

The parametrization (F.1) of Remark F.1 defines an admissible pair only after domain, injectivity, and minimizing-branch conditions are imposed. We therefore restrict it to

$$
\mathcal { Z } _ { \mathrm { a d m } } : = \left\{ z \in C ^ { 2 } ( \Omega ) \left[ \begin{array} { c c } { T _ { z } : \Omega \to \Omega \mathrm { ~ i s ~ a n ~ o r i e n t a t i o n - p r e s e r v i n g } } \\ { C ^ { 1 } \mathrm { ~ d i f f e o m o r p h i s m } , } & { \underset { \Omega } { \operatorname* { i n f } } J _ { T _ { z } } > 0 , } \\ { \hfill \mathrm { d } _ { \Omega } ( x , T _ { z } ( x ) ) = \arctan \left( \frac { \left| \nabla z \left( x \right) \right| } { 2 } \right) } & { \mathrm { f o r ~ e v e r y ~ } x \in \Omega \right) } \end{array} \right\} .
$$

On a bounded convex Euclidean domain the last identity follows once $T _ { z } ( \Omega ) \subset \Omega ;$ on the torus it additionally requires the exponential displacement to remain on the minimizing geodesic branch. The trade-of is the following. By Proposition 2.5 every optimal pair between positive densities has the form (F.1) for a Lipschitz $z ,$ so this parametrization matches the structure of an exact optimal pair, removes the cutof, and eliminates the directions along which $T$ and $q$ trade of at fixed pushforward. It does not, however, guarantee admissibility for an arbitrary network output, and standard uniform universal approximation [31] does not by itself yield Assumption 5.1 for this class: one must separately require a regular potential $z _ { * } \in C ^ { 2 } ( \Omega )$ and a network sequence with

$$
\| z _ { \theta _ { n } } - z _ { * } \| _ { C ^ { 2 } ( \Omega ) } \longrightarrow 0
$$

preserving the conditions defining ${ \mathcal { Z } } _ { \mathrm { a d m } } ,$ which then gives $T _ { z _ { \theta _ { n } } }  T _ { z }$ in $C ^ { 1 }$ and $q _ { z _ { \theta _ { n } } } \ \to \ q _ { z } ,$ uniformly.

Lemma C.7 relates $z _ { * }$ to the transformed dual potential; the exact-step theory supplies only Lipschitz regularity. The Euler–Lagrange equation (2.13) is algebraic in the density and potential and gives no elliptic regularity gain. Even with smoother V and $W .$ , the $C ^ { 2 }$ control needed here would require additional regularity theory for unbalanced optimal transport. The density bounds (2.12) and the uniform sub-cutof bound of Corollary C.8 identify the relevant non-degenerate regime; see also [12]. Establishing this additional regularity remains open here.

Under the step-scaled ansatz $z _ { \theta } = \tau w _ { \theta }$ , invertibility near the identity follows from a quantitative Hessian bound such as $\tau \| D ^ { 2 } w _ { \theta } \| _ { \infty } \ll 1$ together with the boundary or minimizingbranch condition; small τ alone is not suficient. Finally, $J _ { T _ { z _ { \theta } } }$ involves second derivatives, so the stochastic trace estimators available for (4.6) do not apply directly. The two parametrizations are complementary: (4.5) guarantees admissibility by construction, whereas (F.1) matches the exact optimal-pair structure under the additional restrictions above. The main representability theorem is formulated for a generic admissible pair class and applies to either choice.

## F.2 Step scaling and uniform estimates

The scaling $z _ { \theta } = \tau w _ { \theta }$ in Remark F.1 separates the $O ( \tau )$ potential from its O(1) profile. For admissible potentials $z _ { * } = \tau w _ { * }$ and $z _ { \theta }$ with uniformly bounded $C ^ { 1 }$ profiles, an $O ( \tau )$ approximation of $w _ { * }$ in $C ^ { 1 }$ yields $O ( \tau ^ { 2 } )$ errors in the map and growth factor of (F.1), which enter the one-step stability estimate (3.5).

The bounds of Theorem 2.6 do not, however, give step-uniform density control by direct iteration. For Boltzmann entropy, (2.12) incurs a factor $e ^ { \omega _ { 1 } }$ per step, where $\omega _ { 1 }$ is the oscillation of the self-consistent field; without additional τ-scaled control, this yields no τ-independent bound over $N _ { \tau } = \lfloor \tau / \tau \rfloor$ steps. Sharper estimates or a separate argument are needed. The results in the main text require neither this scaling nor step-uniform bounds.

## F.3 Experimental implementation and refinement checks

Parametrization and optimization. Each map composes two full-input dense residual FCNN blocks with certified contraction bounds. Scalar fields use trainable mixed features

$$
F _ { \eta } ( z ) = a _ { 0 } + a \cdot z + \sum _ { j = 1 } ^ { m } \bigl [ b _ { j } ( w _ { j } \cdot z + c _ { j } ) ^ { 2 } + d _ { j } \operatorname { t a n h } ( w _ { j } \cdot z + c _ { j } ) \bigr ] .
$$

The density is represented as $\exp ( \exp ( F _ { \eta } , - 3 0 0 , 3 0 ) )$ , replacing $F _ { \eta }$ by $F _ { \eta } - V$ for no-flux potential problems, with a fitted mass shift; a separate growth network uses $q _ { \eta } = \exp ( 4 \operatorname { t a n h } ( \tau F _ { \eta } / 4 ) )$ . Cosine inputs enforce the no-flux chemical-potential condition. Interaction uses four periodic inputs (one sine/cosine pair per coordinate), width 256, and width 512 in the capacity check.

Trajectory pairs, including the one-step PDE test, are initialized by fitting output coefficients to the source predictor $T \approx \mathrm { i d } - \tau \nabla g _ { \rho }$ , log $q \approx - 2 \tau g _ { \rho }$ of Remark F.1 before joint training; independent pair-scaling and certificate tests start from (id, 1). Joint L-BFGS with base budget N is followed by a conditioned stage of budget max(20, min $( 4 0 0 , \lfloor N / 2 \rfloor ) )$ ; when growth is learned, each stage includes growth polishing of the same objective with budget max(50, min(400, ⌊N/3⌋)). Recursive propagation fits the log density with up to 40 L-BFGS iterations for hidden parameters, variable projection for output coeficients, and sampled-mass restoration; the certificate uses the raw inverse-map pushforward. Across the six trajectories in Table 10, the per-fit Frobenius change in the density hidden-input weight matrix was at most $2 \times 1 0 ^ { - 5 }$ , and was zero at every interaction step. This diagnostic excludes hidden biases and output coeficients. Tables 8 and 2 give the base settings and pair-training rules.

The entropy objective uses the integration-by-parts identity

$$
C _ { \rho } ( T ) : = \int _ { \Omega } \rho \big ( \mathrm { d i v } \boldsymbol { v } + \nabla \log \rho \cdot \boldsymbol { v } \big ) d \boldsymbol { x } = 0 , \qquad \boldsymbol { v } = T - \mathrm { i d } ,\tag{F.3}
$$

for a smooth positive source $\rho ,$ with $v \cdot n = 0$ or a periodic lift. Its quadrature estimate is added with coeficient +1 in training and validation, using the current source’s score. This changes finite-quadrature values and gradients, but not the continuum objective.

Trainable networks use mps torch.float32; quadrature and linear solves use CPU float64. The seed is 1234.

<table><tr><td>Experiment</td><td></td><td>d domain</td><td>T</td><td></td><td>steps requested  $T = N \tau$ </td><td>widths  $( T , q , \rho )$ </td><td>base  $N$ </td></tr><tr><td>Small-τ pair</td><td></td><td>1 [−1,1]</td><td>0.00125-0.02</td><td>1</td><td></td><td>32,64,-</td><td>2400</td></tr><tr><td>Pointwise PDE diagnostic</td><td>1</td><td>[−1,1]</td><td>0.000625-0.02</td><td>1</td><td></td><td>64,64,64</td><td>3000</td></tr><tr><td>Finite-grid certificate</td><td>1</td><td>[−0.6, 0.6]</td><td>0.020</td><td>1</td><td>0.020</td><td>varied,64,-</td><td>2400</td></tr><tr><td>Entropy-potential</td><td>1</td><td>[−-1,1]</td><td>2.500e-03</td><td>1000</td><td>2.500</td><td>64,64,64</td><td>250</td></tr><tr><td>Entropy-potential</td><td>2</td><td> $[ - 1 , 1 ] ^ { 2 }$ </td><td>2.500e-03</td><td>480</td><td>1.200</td><td>64,64,64</td><td>250</td></tr><tr><td>HK entropy</td><td>8</td><td> $[ - 3 . 5 , 3 . 5 ] ^ { 8 }$ </td><td>6.250e-04</td><td>240</td><td>0.150</td><td>64,64,64</td><td>250</td></tr><tr><td>Fokker-Planck</td><td>2</td><td> $[ - 5 , 5 ] ^ { 2 }$ </td><td>1.250e-03</td><td>960</td><td>1.200</td><td>64,64,64</td><td>250</td></tr><tr><td>Coercive interaction</td><td>2</td><td> $[ - 1 , 1 ] ^ { 2 }$ </td><td>2.500e-03</td><td>120</td><td>0.300</td><td>256,256,256</td><td>400</td></tr><tr><td>Pair restriction</td><td>1</td><td> $[ - 1 , 1 ]$ </td><td>2.500e-03</td><td>320</td><td>0.800</td><td>64,64,64</td><td>250</td></tr></table>

Table 8: Executed settings. N excludes conditioned and growth-polishing stages. Certificate map widths are 8, 16, 32, 64, 128; a missing density width denotes a raw pushforward.

Data and references. Gaussian sources are $\mathcal { N } ( 0 , 0 . 4 5 ^ { 2 } I _ { 8 } )$ for entropy flow and $\mathcal { N } ( 0 , I _ { 2 } )$ for FP2D, restricted to their boxes. For FP2D, $\mu = ( 1 / 3 , 1 / 3 )$ and $\Sigma = \textstyle { \frac { 1 } { 8 } } \left( \begin{array} { c c } { { 5 } } & { { - 3 } } \\ { { - 3 } } & { { 5 } } \end{array} \right)$ . Projection/error evaluation uses $4 6 ^ { 2 } / 9 0 ^ { 2 }$ Gauss–Legendre nodes for FP2D and 4096/32768 importance draws for entropy 8D; 2D relaxation uses $9 1 ^ { 2 }$ projection nodes.

The certificate uses $\rho _ { 0 } = 0 . 7 5 + 0 . 3 5 e ^ { - 8 ( x + 0 . 1 8 ) ^ { 2 } } , V = 5 x ^ { 2 } + 0 . 2 5 x$ , 512 training nodes (also used in the small-time pair tests) and a 65-node certificate grid. Relaxation references use 600 cells in 1D and $1 2 8 ^ { 2 }$ in 2D, with $\Delta t = 1 0 ^ { - 4 }$ . The interaction source is $\rho _ { 0 } = { \textstyle { \frac { 1 } { 4 } } } [ 1 + 0 . 3 8 \cos ( \pi ( x _ { 1 } -$ $0 . 3 0 ) ) + 0 . 2 2 \sin ( \pi ( x _ { 2 } + 0 . 1 5 ) ) ]$ ; its reference includes potential and interaction drift.

The one-step PDE diagnostic uses 129 initial-fit nodes, 257 pair nodes, 1024 projection nodes and 601 uniform evaluation points, checked against 1201 points.

Refinement and propagation checks. Held-out rules check descent, density fitting and map admissibility; the one-step growth-stationarity tolerances are $1 0 ^ { - 4 }$ in relative RMS and $1 0 ^ { - 3 }$ in maximum residual. Table 9 reports independently optimized time refinements at fixed horizons.

<table><tr><td>Experiment</td><td>T</td><td>attained  $_ T$ </td><td>rel.  $L ^ { 2 }$ </td><td>rel. mass error</td><td>flow residual</td><td>reference residual</td></tr><tr><td>Entropy 8d</td><td>1.250e-03</td><td>0.150</td><td>0.025</td><td>0.022</td><td></td><td></td></tr><tr><td>Entropy 8d</td><td>6.250e-04</td><td>0.150</td><td>8.043e-03</td><td>0.012</td><td></td><td></td></tr><tr><td>Implicit interaction</td><td>2.500e-03</td><td>0.300</td><td>5.212e-04</td><td></td><td>0.021</td><td>4.787e-03</td></tr><tr><td>Frozen interaction</td><td>2.500e-03</td><td>0.300</td><td>4.211e-04</td><td></td><td>0.021</td><td>4.787e-03</td></tr><tr><td>Implicit interaction</td><td>5.000e-03</td><td>0.300</td><td>1.051e-03</td><td></td><td>0.021</td><td>9.533e-03</td></tr><tr><td>Frozen interaction</td><td>5.000e-03</td><td>0.300</td><td>9.422e-04</td><td></td><td>0.020</td><td>9.533e-03</td></tr></table>

Table 9: Fixed-horizon time refinement. Interaction errors and residuals use the refined reference grid and the same periodic operator for both trajectories.

Table 10 reports held-out fit errors relative to the unprojected endpoint and optimizer work in closure evaluations, not wall-clock time. All six stored trajectories satisfy $E _ { k + 1 } - E _ { k } \ \leq$ $1 0 ^ { - 7 } ( 1 + | E _ { k } | )$ for the propagated states; these checks do not bound accumulated projection error.
<table><tr><td>Case</td><td>steps fit rel.</td><td> $L ^ { 2 }$ </td><td></td><td>mass error pair evals./step fit evals./step</td></tr><tr><td>Relaxation 1D</td><td>1000</td><td>1.72e-05</td><td>5.29e-06</td><td>381.5 2.1</td></tr><tr><td>Relaxation 2D</td><td>480</td><td>4.78e-05</td><td>7.79e-06</td><td>693.3 2.8</td></tr><tr><td>Entropy 8D</td><td>240</td><td>1.92e-04</td><td>3.91e-05</td><td>499.5 5.0</td></tr><tr><td>FP2D</td><td>960</td><td>1.89e-04</td><td>1.22e-04</td><td>635.9 7.2</td></tr><tr><td>Implicit interaction</td><td>120</td><td>1.26e-05</td><td>7.12e-07</td><td>800.0 2.0</td></tr><tr><td>Frozen interaction</td><td>120</td><td>1.25e-05</td><td>5.96e-07</td><td>823.7 2.0</td></tr></table>

Table 10: Propagation and work diagnostics. Fit and mass errors are maxima over steps; work columns are mean objective-closure evaluations per step, including pair stages and density fitting. Pair and fit closures have diferent costs.

Interaction residuals use the same periodic finite-diference operator for learned and reference states, normalized by the sum of time-increment and PDE-activity norms. Because the reference is advanced using this spatial operator, its residual primarily reflects temporal discretization. Applying the operator to the learned continuum states also introduces spatial truncation error. At $T = 0 . 3$ , for the implicit state on the $1 2 8 ^ { 2 }$ grid, difusion and drift each have $L ^ { 2 }$ norm about 2.6, whereas their sum has norm about 0.03; the residual numerator has $L ^ { 2 }$ norm about 0.14% of that of either individual transport term. The nearest-image kernel has derivative kinks, so smooth-kernel conclusions are not invoked.

<table><tr><td>Treatment</td><td> $1 2 8 ^ { 2 }$ </td><td> $2 5 6 ^ { 2 }$ </td><td> $5 1 2 ^ { 2 }$ </td></tr><tr><td>Implicit</td><td>0.02069</td><td>0.02370</td><td>0.06650</td></tr><tr><td>Frozen</td><td>0.02123</td><td>0.02446</td><td>0.06700</td></tr></table>

Table 11: Interaction strong residuals at $T = 0 . 3 , \tau = 0 . 0 0 2 5$ . All grids evaluate the same final two learned states with the full-periodic operator, without retraining.

On the $6 4 ^ { 2 }$ grid the same operator gives a residual of 0.048 for the implicit state, so the residual first decreases and then increases under grid refinement. This behavior is consistent with $O ( h ^ { 2 } )$ truncation on coarse grids and with small pointwise evaluation errors amplified by second diferences in proportion to $h ^ { - 2 }$ on fine grids; no grid-converged strong-form residual is established. The $6 4 ^ { 2 }$ reference difers from the $1 2 8 ^ { 2 }$ reference at quarter time step by $6 . 9 5 7 \times 1 0 ^ { - 4 }$ in relative $L ^ { 2 }$ on the coarse grid, which limits the density comparison at this level.

## References

[1] R. Jordan, D. Kinderlehrer, and F. Otto. The variational formulation of the Fokker–Planck equation. SIAM J. Math. Anal., 29:1–17, 1998.

[2] L. Ambrosio, N. Gigli, and G. Savar´e. Gradient Flows in Metric Spaces and in the Space of Probability Measures. Birkh¨auser, Basel, 2nd edition, 2008.

[3] S. Kondratyev, L. Monsaingeon, and D. Vorotnikov. A new optimal transport distance on the space of finite Radon measures. Adv. Diferential Equations, 21:1117–1164, 2016.

[4] L. Chizat, G. Peyr´e, B. Schmitzer, and F.-X. Vialard. Unbalanced optimal transport: dynamic and Kantorovich formulations. J. Funct. Anal., 274:3090–3123, 2018.

[5] L. Chizat, G. Peyr´e, B. Schmitzer, and F.-X. Vialard. An interpolating distance between optimal transport and Fisher–Rao metrics. Found. Comput. Math., 18:1–44, 2018.

[6] M. Liero, A. Mielke, and G. Savar´e. Optimal transport in competition with reaction: the Hellinger–Kantorovich distance and geodesic curves. SIAM J. Math. Anal., 48:2869–2911, 2016.

[7] M. Liero, A. Mielke, and G. Savar´e. Optimal entropy–transport problems and a new Hellinger–Kantorovich distance between positive measures. Invent. Math., 211:969–1117, 2018.

[8] T. O. Gallou¨et and L. Monsaingeon. A JKO splitting scheme for Kantorovich–Fisher–Rao gradient flows. SIAM J. Math. Anal., 49:1100–1130, 2017.

[9] F. Fleißner. A minimizing movement approach to a class of scalar reaction–difusion equations. ESAIM Control Optim. Calc. Var., 27:18, 2021. doi: 10.1051/cocv/2020090.

[10] S. Di Marino and L. Chizat. A tumor growth model of Hele–Shaw type as a gradient flow. ESAIM Control Optim. Calc. Var., 26:103, 2020. doi: 10.1051/cocv/2020019.

[11] M. Liero, A. Mielke, and G. Savar´e. Fine properties of geodesics and geodesic λ-convexity for the Hellinger–Kantorovich distance. Arch. Ration. Mech. Anal., 247:112, 2023. doi: 10.1007/s00205-023-01941-1.

[12] T. Gallou¨et, R. Ghezzi, and F.-X. Vialard. Regularity theory and geometry of unbalanced optimal transport. J. Funct. Anal., 289(7):111042, 2025. doi: 10.1016/j.jfa.2025.111042.

[13] V. Laschos and A. Mielke. Evolutionary variational inequalities on the Hellinger– Kantorovich and spherical Hellinger–Kantorovich spaces. Commun. Partial Diferential Equations, 51(2–3):193–235, 2026. doi: 10.1080/03605302.2026.2650715.

[14] P. Mokrov, A. Korotin, L. Li, A. Genevay, J. M. Solomon, and E. Burnaev. Large-scale Wasserstein gradient flows. In Advances in Neural Information Processing Systems 34, 2021.

[15] M. S. Park, C. Kim, H. Son, and H. J. Hwang. The deep minimizing movement scheme. J. Comput. Phys., 494:112518, 2023. doi: 10.1016/j.jcp.2023.112518.

[16] L. Chizat, G. Peyr´e, B. Schmitzer, and F.-X. Vialard. Scaling algorithms for unbalanced optimal transport problems. Math. Comp., 87:2563–2609, 2018. doi: 10.1090/mcom/3303.

[17] J. A. Carrillo, K. Craig, L. Wang, and C. Wei. Primal dual methods for Wasserstein gradient flows. Found. Comput. Math., 22:389–443, 2022. doi: 10.1007/s10208-021-09503-1.

[18] D. Ponnoprat, N. Isobe, and M. Imaizumi. Minimax optimal estimation of transportgrowth pairs in unbalanced optimal transport. arXiv:2605.08705, 2026. doi: 10.48550/ arXiv.2605.08705.

[19] J. Fan, Q. Zhang, A. Taghvaei, and Y. Chen. Variational Wasserstein gradient flow. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of PMLR, pages 6185–6215, 2022.

[20] C. Xu, X. Cheng, and Y. Xie. Normalizing flow neural networks by JKO scheme. In Advances in Neural Information Processing Systems 36, pages 47379–47405, 2023.

[21] W. Lee, L. Wang, and W. Li. Deep JKO: Time-implicit particle methods for general nonlinear gradient flows. J. Comput. Phys., 514:113187, 2024. doi: 10.1016/j.jcp.2024. 113187.

[22] K. D. Yang and C. Uhler. Scalable unbalanced optimal transport using generative adversarial networks. In International Conference on Learning Representations, 2019.

[23] F. L¨ubeck, C. Bunne, G. Gut, J. Sarabia del Castillo, L. Pelkmans, and D. Alvarez-Melis. Neural unbalanced optimal transport via cycle-consistent semi-couplings. arXiv:2209.15621, 2022. doi: 10.48550/arXiv.2209.15621.

[24] L. Eyring, D. Klein, T. Uscidda, G. Palla, N. Kilbertus, Z. Akata, and F. Theis. Unbalancedness in neural Monge maps improves unpaired domain translation. In International Conference on Learning Representations, 2024.

[25] Y. Sha, Y. Qiu, P. Zhou, and Q. Nie. Reconstructing growth and dynamic trajectories from single-cell transcriptomics data. Nat. Mach. Intell., 6:25–39, 2024. doi: 10.1038/ s42256-023-00763-w.

[26] J. Choi, J. Choi, and M. Kang. Scalable Wasserstein gradient flow for generative modeling through unbalanced optimal transport. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of PMLR, pages 8629–8650, 2024.

[27] K. Neklyudov, R. Brekelmans, A. Tong, L. Atanackovic, Q. Liu, and A. Makhzani. A computational framework for solving Wasserstein Lagrangian flows. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of PMLR, pages 37461– 37485, 2024.

[28] J. Hertrich and R. Gruhlke. Importance corrected neural JKO sampling. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of PMLR, pages 23083–23119, 2025.

[29] R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud. Neural ordinary diferential equations. In Advances in Neural Information Processing Systems 31, 2018.

[30] W. Grathwohl, R. T. Q. Chen, J. Bettencourt, I. Sutskever, and D. Duvenaud. FFJORD: Free-form continuous dynamics for scalable reversible generative models. In International Conference on Learning Representations, 2019.

[31] M. Leshno, V. Ya. Lin, A. Pinkus, and S. Schocken. Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. Neural Networks, 6: 861–867, 1993. doi: 10.1016/S0893-6080(05)80131-5.