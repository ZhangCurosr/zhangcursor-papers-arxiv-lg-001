# A FINSLERIAN APPROACH FOR EMBEDDINGDIRECTED DATA

Gwendal Debaussart-Joniec<sup>∗</sup>   
ENS Paris-Saclay, Universite Paris-Saclay, CNRS, Centre Borelli´   
F-91190 Gif-sur-Yvette, France   
gwendal.debaussart@ens-paris-saclay.fr   
Theau Blanchard´   
UMR 1346 HeKA, Universite Paris Cit´ e, Inria, Inserm´   
GE Healthcare   
F-75015 Paris, France   
theau.blanchard@inria.fr   
Argyris Kalogeratos   
ENS Paris-Saclay, Universite Paris-Saclay, CNRS, Centre Borelli´   
F-91190 Gif-sur-Yvette, France   
argyris.kalogeratos@ens-paris-saclay.fr

## ABSTRACT

Many datasets carry an intrinsic directionality: citations point backward in time, cells differentiate along lineages, and traffic follows preferred routes. Spectral embedding methods, including most of their extensions to directed graphs, discard this information: they symmetrize the data and map it into a Euclidean space where asymmetry cannot be represented. We instead model directed data as sampled from a Finsler manifold, whose distance depends on the direction of travel, and study the kernel operator built from this asymmetric distance. Through a moment expansion of this operator, we show that its symmetric and antisymmetric parts separate geometry from direction. As the bandwidth of the kernel vanishes, the symmetric part converges to a weighted Laplacian, recovering diffusion maps in the Riemannian case, while the antisymmetric part converges to a first-order transport operator that encodes the directionality. We prove that the corresponding graph operators, built from finitely many samples, converge uniformly and almost surely to these limits. For Randers metrics, this vector field is explicit and yields an embedding algorithm recovering both the manifold structure, from the spectrum of the symmetric part, and the underlying drift. We illustrate the approach on synthetic directed graphs and point-clouds.

## 1 INTRODUCTION

A citation never points to the future, a cell rarely un-differentiates, influence in a social network is rarely returned in equal measure, and traffic or commodity flows between two locations are rarely symmetric. These are just some of the many examples of directed data, i.e. data with asymmetric relationships between entities. This directionality can be given explicitly, as in directed graphs (Debaussart-Joniec et al., 2026;

Chung, 2005; Cucuringu et al., 2020; Sevi et al., 2026), or arise from points in a continuous space, as in RNA velocity (Bergen et al., 2021) or lineage data (Zweig et al., 2026). For such data, Finsler geometry, an extension of Riemannian geometry allowing direction-dependent metrics, offers a natural mathematical language. Yet, Finsler geometry remains largely unexplored in machine learning or manifold learning.

Graph Laplacians and diffusion operators underlie many of the tools we rely on for structured data, including spectral clustering (Von Luxburg, 2007), dimensionality reduction (Donoho & Grimes, 2003), semisupervised learning, and graph neural networks (Kipf & Welling, 2016). Their theoretical grounding rests on a convergence result: when data is sampled from a compact Riemannian manifold, the graph Laplacian converges, as the sample size grows and the kernel bandwidth shrinks, to the Laplace–Beltrami operator (Coifman & Lafon, 2006; Belkin & Niyogi, 2001; Lafon, 2004; Hein et al., 2007; Singer, 2006; Calder & Trillos, 2022; Peoples & Harlim, 2025). This theory is symmetric by construction, as the kernel is a function of a symmetric distance.

Methods for directed graphs generally restore symmetry somewhere along the way, through the stationary distribution of a random walk (Chung, 2005; Zhou et al., 2003), an explicit symmetrization of the adjacency matrix (Satuluri & Parthasarathy, 2011), an encoding of the direction in the complex phase of a Hermitian matrix (Cucuringu et al., 2020; Fanuel et al., 2018; Zhang et al., 2021), or vertex measures (Sevi et al., 2026; Debaussart-Joniec et al., 2026). Directed embedding methods (Chen et al., 2007; Ou et al., 2016) keep the asymmetry in the objective, but, like the spectral ones, return points of a Euclidean space, in which the directionality of the data is lost. Closest to our work, Perrault-Joncas & Meila (2011) and Yuan et al. (2022) take continuous limits of Laplacian-type operators built from an asymmetric kernel, and obtain a diffusion term together with an advection term. There, as in other uses of asymmetric kernels (Wu et al., 2010; He et al., 2023; Tao et al., 2024), the asymmetry is introduced by hand, without an underlying geometry.

Finsler geometry is classical in mathematics (Finsler, 1918; Yeon, 2015; Bao et al., 2012; Shen, 2001; Ohta, 2021), and arises wherever the cost of moving depends on the direction of travel (Zermelo, 1931; Markvorsen, 2016; Gahtan et al., 2026; Asanjarani, 2021; Melonakos et al., 2008; Chen et al., 2024; Pfeifer, 2019). In machine learning, it has been used to model anisotropic or asymmetric structures (Lopez et al., 2021; Weber et al., 2024; Pouplin et al., 2023; Zweig et al., 2026; Shaska, 2025), and Dages et al.\` (2025); Dages et al.\` (2026) generalize multidimensional scaling to a Finslerian embedding space. These methods learn direction-dependent representations, but the Finsler metric usually has a prescribed form (Weber et al., 2024; Dages et al.\` , 2025).

This work proposes a principled approach for representing directed data, grounded in Finsler geometry: we model the data as sampled from a Finsler manifold, so that the asymmetry comes from an arbitrary underlying geometry rather than from a hand-crafted kernel, and we recover this geometry, through a Randers approximation of the metric, in the embedding space (see Figure 1). Our contributions are as follows.

• Continuum limits (Section 3). Building on the convergence theory of graph Laplacians, we derive a moment expansion of the Finsler kernel operator. Its symmetric part converges to a weighted Laplacian of the Binet–Legendre metric, recovering diffusion maps in the Riemannian case, while its antisymmetric part converges to the derivative along the centroid field of the unit ball, which encodes the directionality.

• Consistency (Section 4). We prove that the corresponding graph operators, built from finitely many samples, converge uniformly and almost surely to these limits, with an explicit rate.

• Directed embedding (Section 5). We introduce the moment-Randers metric, a Randers approximation of the Finsler metric with the same limiting operators, and show that it can be estimated from the two graph operators alone. This yields an algorithm returning both an embedding and a Randers metric on it.

• Experiments (Section 6). We illustrate the approach on point-clouds sampled from Randers and Matsumoto manifolds, as well as on directed stochastic block models.

![](images/544191c272f3dad8c666d0801e77dfe2aced109bf27e5aa850187660ccfa2ebd.jpg)  
Figure 1: Recovering geometry and directionality from Finsler manifolds. A kernel built from the Finsler distance turns sampled points into a directed graph. The operators $\mathcal { L } ^ { s }$ and ${ \mathcal { L } } ^ { a }$ of Theorem $^ 2$ recover the geometry and the directional structure, and are merged into a Randers metric on the embedding.

## 2 BACKGROUND AND SETTING

## 2.1 BASIC CONCEPTS

Notations are summarized in Tables 2 and $^ { 3 , }$ and Appendix B gives a self-contained introduction to Finsler geometry. We refer to Shen (2001) and Ohta (2021) for additional details.

Let M be an $m \cdot$ -dimensional manifold. A Finsler metric is a map $\mathcal { F } : T \mathcal { M }  [ 0 , \infty )$ , smooth and positive away from the zero section, that is positively homogeneous, $\mathcal { F } ( \boldsymbol { x } , t \boldsymbol { v } ) = t \mathcal { F } ( \boldsymbol { x } , \dot { \boldsymbol { v } } )$ for $t > 0 ,$ , sub-additive in $v ,$ and strongly convex. It plays on each tangent space $T _ { x } { \mathcal { M } }$ the role that a norm plays on $\mathbb { R } ^ { m }$ and, exactly as in the Riemannian case, it defines the length of a curve and a distance dst<sub>F</sub> $( x , y )$ as the infimum of the lengths of the curves joining $x$ to $y .$ . Riemannian metrics are the special case where $\mathcal { F } ( x , \cdot )$ is induced by an inner product. The essential difference is that homogeneity is only required for positive scalars: in general $\mathcal { F } ( \bar { x , } v ) \neq \mathcal { F } ( x , - v )$ , so that the distance dst satisfies the triangle inequality but not symmetry, since traveling from x to y need not cost as much as traveling back. The reverse metric $\mathcal { F } ( \boldsymbol { x } , - \boldsymbol { v } )$ is again a Finsler metric, whose distance is $( x , y ) \mapsto \mathrm { d s t } _ { \mathcal { F } } ( y , x )$ (Lemma 20).

This work relies on a subclass of Finsler metrics and one canonical construction. Randers metrics (Randers, 1941) are a natural generalization of Riemannian metrics. For a Riemannian metric $\alpha ( x , v ) = \sqrt { v ^ { \mathsf { T } } \mathbf { A } ( x ) v }$ and a vector field b on $\mathcal { M }$ , the Randers metric is defined as

$$
\mathcal { F } ( x , v ) = \sqrt { v ^ { \mathsf { T } } \mathbf { A } ( x ) v } + b ( x ) ^ { \mathsf { T } } v , \qquad \| b ( x ) \| _ { \mathbf { A } ^ { - 1 } } < 1 .\tag{1}
$$

The constraint on $b$ ensures that $\mathcal { F }$ is positive. Its indicatrix $\{ v \in T _ { x } \mathcal { M } : \mathcal { F } ( x , v ) = 1 \}$ is an ellipsoid whose center is shifted in a direction determined by $b ( x )$ , so that all the directional information is carried by the single vector field b. We write $\mathcal { B } _ { x } = \{ v \in \dot { T _ { x } } \mathcal { M } : \mathcal { F } ( x , v ) \leq 1 \}$ } for the unit ball of $\mathcal { F }$ in the tangent space $T _ { x } { \mathcal { M } }$ . The second moment of the uniform distribution on ${ \mathcal { B } } _ { x }$ yields a Riemannian metric canonically attached to $\mathcal { F } \colon$ the Binet–Legendre metric (Matveev & Troyanov, 2012),

$$
g _ { \mathrm { B L } } ^ { i j } ( x ) = \frac { \left( m + 2 \right) } { \lambda _ { x } \left( \mathcal { B } _ { x } \right) } \int _ { \mathcal { B } _ { x } } { v ^ { i } v ^ { j } \mathrm { d } \lambda _ { x } ( v ) } ,\tag{2}
$$

which is smooth whenever $\mathcal { F }$ is smooth, and coincides with A when $\mathcal { F }$ is Riemannian. The first moment of the same distribution is the centroid of the unit ball,

$$
\mathbf { c } ( x ) = \frac { 1 } { \lambda _ { x } ( \mathcal { B } _ { x } ) } \int _ { \mathcal { B } _ { x } } { v } \mathrm { d } \lambda _ { x } ( v ) \in T _ { x } \mathcal { M } ,\tag{3}
$$

which defines a vector field on ${ \mathcal { M } } .$ , the centroid field. It vanishes whenever $\mathcal { F }$ is reversible, since the unit ball is then symmetric about the origin, and thus measures the asymmetry of F at first order; for Randers metrics, it is an explicit function of A and b (Proposition 43). The pair $( \mathbf { c } , g _ { \mathrm { B L } } )$ , i.e. the first two moments of the unit ball, will carry respectively the direction and the geometry recovered by our operators.

Unlike a Riemannian manifold, a Finsler manifold carries no canonical volume. In this work, we use the Busemann–Hausdorffmeasure m (Definition 17). In a chart, with dx the Lebesgue measure on $\mathcal { M }$ and $\lambda _ { x }$ the Lebesgue measure on $T _ { x } { \mathcal { M } }$ in the coordinate basis $( \partial _ { i } | _ { x } )$ , it reads

$$
\mathrm { d } \mathfrak { m } _ { \mathrm { B H } } ( x ) = \sigma _ { \mathrm { B H } } ( x ) \mathrm { d } x , \qquad \sigma _ { \mathrm { B H } } ( x ) = \frac { \mathrm { V o l } _ { \mathrm { E u c } } ( \mathbb { B } ^ { m } ) } { \lambda _ { x } ( \mathcal { B } _ { x } ) } ,\tag{4}
$$

where $\operatorname { V o l } _ { \operatorname { E u c } } ( { \mathbb { B } } ^ { m } )$ is the volume of the Euclidean unit ball of $\mathbb { R } ^ { m }$ . Although dx and $\lambda _ { x }$ are chart-dependent, m<sub>BH</sub> is not. When $\mathcal { F }$ is Riemannian, m<sub>BH</sub> coincides with the Riemannian volume. We assume that the data are N i.i.d. samples $X _ { 1 } , \ldots , X _ { N }$ from a probability measure $\mathbb { P }$ on $\mathcal { M }$ , and denote by $\rho$ its density with respect to m<sub>BH</sub>, so that $\begin{array} { r } { \mathbb { P } ( x ) = \rho ( x ) \mathbb { d } \mathfrak { m } _ { \mathrm { B H } } ( x ) } \end{array}$

## 2.2 OUR SETTING

We make the following assumptions throughout:

(A1) The kernel $K : \mathbb { R } _ { + } \to \mathbb { R } _ { + }$ is a smooth, positive, monotone decreasing function with bounded derivative, and has a sub-exponential tail: there exist constants $C _ { K } , \nu _ { K } > 0$ such that for any $r \in \mathbb { R } _ { + }$ , $K ( r ) \le C _ { K } e ^ { - \nu _ { K } r }$

(A2) The manifold M is smooth, connected, compact and without boundary<sup>1</sup>, and the Finsler metric $\mathcal { F }$ is smooth and positive on the slit tangent bundle $T { \mathcal { M } } \backslash \{ 0 \}$ . Moreover, the sampling density $\rho$ is smooth and bounded away from zero by a constant $\rho _ { \mathrm { m i n } } > 0$

Given K satisfying (A1), we define the kernel $W : \mathcal { M } \times \mathcal { M }  \mathbb { R } _ { + }$ and its out/in-degrees as:

$$
W ( x , y ) = \frac { 1 } { \varepsilon ^ { m } } K \left( \frac { \mathrm { d } { \bf s } \mathrm { t } _ { \mathcal { F } } ( x , y ) } { \varepsilon } \right) , \quad d ( x ) = \int _ { \mathcal { M } } W ( x , y ) \mathrm { d } \mathbb { P } ( y ) , \quad d ^ { \prime } ( x ) = \int _ { \mathcal { M } } W ( y , x ) \mathrm { d } \mathbb { P } ( y ) ,
$$

where ε is the kernel bandwidth. We further assume that the degrees are bounded below: for any $x \in \mathcal { M }$ $d ( x ) , d ^ { \prime } ( x ) > d _ { \mathrm { m i n } } > 0$ . This is a standard assumption in the literature (Hein et al., 2007), and it is satisfied when the kernel is sub-exponential and $\rho$ is positive. We study the kernel through its associated forward and backward transport operators $\mathcal { G } _ { \mathcal { F } }$ and $\mathcal { G } _ { \mathcal { F } } ^ { \prime }$ . For any $f : \mathcal { M }  \dot { \mathbb { R } }$ smooth enough, these are defined as:

$$
\Lt _ { \mathcal { F } } [ f ] ( x ) = \int _ { \mathcal { M } } W ( x , y ) f ( y ) \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } ( y ) , \quad \Lt _ { \mathcal { F } } ^ { \prime } [ f ] ( x ) = \int _ { \mathcal { M } } W ( y , x ) f ( y ) \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } ( y ) .
$$

The backward operator is the transport operator of the reversed Finsler metric, and is the $L ^ { 2 } ( \mathcal { M } , \mathfrak { m } _ { \mathrm { B H } } ) { \cdot } \mathrm { a d j o i n }$ t of G<sub>F</sub>, so that $\mathcal { G } _ { \mathcal { F } }$ is self-adjoint precisely when F is reversible (Proposition 19, proved in Appendix D.1).

In practice, we only observe samples from P, whose density $\rho$ is unknown. To reduce its influence, we use a θ-normalization of the kernel $\dot { W }$ , similar to Coifman & Lafon (2006):

$$
W ^ { ( \theta ) } ( x , y ) = \frac { W ( x , y ) } { q _ { \varepsilon } ( x ) ^ { \theta } q _ { \varepsilon } ( y ) ^ { \theta } } , \quad \mathcal { G } _ { \mathcal { F } } ^ { ( \theta ) } [ f ] ( x ) = \int _ { \mathcal { M } } W ^ { ( \theta ) } ( x , y ) f ( y ) \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } ( y ) ,
$$

where $q _ { \varepsilon } ( x ) = ( d ^ { \prime } ( x ) + d ( x ) ) / 2$ is a normalization factor, and $\theta \in [ 0 , 1 ]$ . The backward operator is defined using $W ^ { ( \theta ) } ( y , x )$ instead. In general, the forward and backward operators differ; Section 3 exploits this difference to split the operator into a symmetric and an antisymmetric part.

## 3 CONTINUUM LIMITS: SEPARATING GEOMETRY FROM DIRECTION

Our starting point is a moment expansion of the transport operator $\mathcal { G } _ { \mathcal { F } }$ , the Finslerian counterpart of the expansions used to derive continuum limits of graph Laplacians (Coifman & Lafon, 2006; Singer, 2006). The overall procedure is illustrated in Figure 1.

Theorem 1. For any $f : \mathcal { M }  \mathbb { R }$ smooth enough, we have

$$
\displaystyle \mathcal { G } _ { \mathcal { F } } [ f ] ( x ) = m _ { 0 } f ( x ) + \varepsilon \mathcal { A } _ { 1 } [ f ] ( x ) + \frac { \varepsilon ^ { 2 } } { 2 } \mathcal { A } _ { 2 } [ f ] ( x ) + \mathcal { O } ( \varepsilon ^ { 3 } ) ,
$$

where $\displaystyle \mathcal { A } _ { 1 }$ and $\boldsymbol { \mathcal { A } } _ { 2 }$ are first- and second-order differential operators, whose coefficients are explicitly given in Appendix D.2. The leading term m<sub>0</sub> is the zeroth moment ofthe kernel $K ,$ , and does not depend on x.

We now split $\mathcal { G } _ { \mathcal { F } } ^ { ( \theta ) }$ into its symmetric and antisymmetric parts $\mathcal { G } _ { \mathcal { F } } ^ { ( \theta , s ) }$ and $\mathcal { G } _ { \mathcal { F } } ^ { ( \theta , a ) }$ , which is where the geometry and the direction separate:

$$
\mathscr { G } _ { \mathcal { F } } ^ { ( \theta ) } = \mathscr { G } _ { \mathcal { F } } ^ { ( \theta , s ) } + \mathscr { G } _ { \mathcal { F } } ^ { ( \theta , a ) } , \quad \mathrm { w h e r e } \quad \mathscr { G } _ { \mathcal { F } } ^ { ( \theta , s ) } = ( \mathscr { G } _ { \mathcal { F } } ^ { ( \theta ) } + \mathscr { G } _ { \mathcal { F } } ^ { ( \theta ) ^ { \prime } } ) / 2 \quad \mathrm { a n d } \quad \mathscr { G } _ { \mathcal { F } } ^ { ( \theta , a ) } = ( \mathscr { G } _ { \mathcal { F } } ^ { ( \theta ) } - \mathscr { G } _ { \mathcal { F } } ^ { ( \theta ) ^ { \prime } } ) / 2 .
$$

Define the associated transport operators $\mathcal { P } ^ { ( \theta , s ) }$ and $\mathcal { P } ^ { ( \theta , a ) }$ as

$$
\mathcal { P } ^ { ( \theta , \bullet ) } [ f ] = \frac { 1 } { \mathcal { G } _ { \mathcal { F } } ^ { ( \theta , s ) } [ \rho ] } \left( \mathcal { G } _ { \mathcal { F } } ^ { ( \theta , \bullet ) } [ \rho f ] - f \mathcal { G } _ { \mathcal { F } } ^ { ( \theta , \bullet ) } [ \rho ] \right) ,
$$

for $\bullet \in \{ s , a \}$ . The following theorem gives their limits as $\varepsilon \to 0$ , after rescaling by $\varepsilon ^ { - 2 }$ and $\varepsilon ^ { - 1 }$ , respectively. Theorem 2. The operators $\mathcal { P } ^ { ( \theta , \bullet ) }$ admit limiting operators $\mathcal { L } ^ { \bullet }$ as $\varepsilon \to 0 .$

$$
\mathcal { L } ^ { s } f : = \operatorname* { l i m } _ { \varepsilon  0 } \frac { 1 } { \varepsilon ^ { 2 } } \mathcal { P } ^ { ( \theta , s ) } [ f ] = \frac { c _ { 2 } } { \rho ^ { 2 ( 1 - \theta ) } } \mathrm { d i v } _ { B H } ( \rho ^ { 2 ( 1 - \theta ) } \nabla _ { B L } f ) ,\tag{5}
$$

$$
\mathcal { L } ^ { a } f : = \operatorname* { l i m } _ { \varepsilon \to 0 } \frac { 1 } { \varepsilon } \mathcal { P } ^ { ( \theta , a ) } [ f ] = \tilde { m } _ { 1 } ^ { i } \partial _ { i } f = c _ { 1 } \mathbf { c } ^ { i } \partial _ { i } f ,\tag{6}
$$

where $\tilde { m } _ { 1 } ( x )$ is the normalized first moment of the kernel K over the unit ball B (Definition 21), c is the centroidfield Equation $( 3 ) , c _ { 1 }$ and $c _ { 2 }$ are constants depending on the kernel and the dimension ofthe manifold, $\nabla _ { B L }$ is the gradient with respect to the Binet–Legendre metric g<sub>BL</sub> $o f \mathcal { F } _ { 3 }$ , and div $\cdot _ { B H }$ is the divergence with respect to the Busemann–Hausdorff measure ${ \mathfrak { m } } _ { \mathrm { B H } }$ . In particular, for any Finsler metric F, $\mathcal { L } ^ { s }$ is an elliptic operator, self-adjoint on $L ^ { 2 } ( \mathcal { M } , \overset { \sim } { \rho } { } ^ { 2 ( 1 - \theta ) } \mathfrak { m } _ { \mathrm { B H } } )$

This result is proved in Appendix D.3, and the constants are given in Proposition $3 5 ;$ for instance, $c _ { 1 } =$ $( m + 1 ) \mu _ { 1 } / ( m \mu _ { 0 } )$ and $c _ { 2 } = \mu _ { 2 } / ( 2 m \mu _ { 0 } )$ with $\begin{array} { r } { \mu _ { n } = \int _ { 0 } ^ { \infty } K ( r ) r ^ { \stackrel { \sim } { m } + n - 1 } \mathrm { d } r } \end{array}$ , see Section D.9 for more details. The symmetric limit is thus a weighted Laplacian, whose metric is the Binet–Legendre metric of $\mathcal { F }$ and whose reference measure is $\rho ^ { 2 ( 1 - \theta ) } \mathfrak { m } _ { \mathrm { B H } }$ . When $\theta = 1$ , it does not depend on the sampling density $\rho ,$ which is a desirable property for the embedding algorithm. In contrast, $\theta = 0$ keeps the whole drift induced by $\rho ,$ and $\theta = 1 / 2$ halves it (see also Section $\mathrm { D } . 4 )$ . The antisymmetric limit is the derivative along the centroid field: the first moment of the unit ball carries the direction, and the second moment the geometry. Note also that the finite-ε operator ${ \mathcal { P } } ^ { ( \theta , s ) }$ is always diagonalizable with real eigenvalues.

The divergence div $B H$ is taken with respect to the Busemann–Hausdorff measure, and not with respect to the Riemannian volume of $g _ { \mathrm { B L } }$ . The two coincide up to a constant factor for Berwald metrics, which are those whose connection depends on the position only and not on the direction (Definition 15). In that case, the symmetric limit simplifies as follows, proven in Appendix D.5.

Corollary 3 (Berwald and Riemannian metrics). If Fis a Berwald metric, then

$$
\begin{array} { r } { \mathcal { L } ^ { s } f = c _ { 2 } \left( \Delta _ { B L } f + 2 ( 1 - \theta ) \langle \nabla \log \rho , \nabla f \rangle _ { B L } \right) , } \end{array}\tag{7}
$$

where $\Delta _ { B L }$ is the Laplace–Beltrami operator of g<sub>BL</sub> and $\langle \nabla u , \nabla f \rangle _ { B L } = g _ { \mathrm { B L } } ^ { i j } \partial _ { i } u \partial _ { j } f .$ . If moreover $\mathcal { F }$ is $R i e \mathrm { - }$ mannian, i.e. $\mathcal { F } ( x , v ) = \sqrt { v ^ { \mathsf { T } } \mathbf { A } ( x ) v }$ , then $g _ { \mathrm { B L } } = \mathbf { A } , \mathcal { L } ^ { a } = 0$ , and Equation (7) is the θ-normalized diffusion generator ofCoifman & Lafon (2006).

## 4 DISCRETE OPERATORS AND CONVERGENCE GUARANTEES

In the discrete setting, where the manifold is only accessed through N points $X _ { 1 } , \ldots , X _ { N }$ sampled i.i.d. from $\mathbb { P } ,$ , the construction follows the same principle as the continuous one. From the empirical kernel matrix $\mathbf { W } _ { N } ( i , j ) = W ( X _ { i } , X _ { j } )$ , we define the out- and in-degrees $\begin{array} { r } { { \bf D } _ { N } ( i , i ) = \sum _ { i } { \bf W } _ { N } ^ { * } ( i , j ) } \end{array}$ and $\begin{array} { r } { \mathbf { D } _ { N } ^ { \prime } ( i , i ) = \sum _ { i } \mathbf { W } _ { N } ( j , i ) } \end{array}$ , and their half-sum ${ \bf Q } _ { N } = ( { \bf D } _ { N } + { \bf D } _ { N } ^ { \prime } ) / 2$ . The θ-normalized matrix and its symmetric and antisymmetric parts are then

$$
\begin{array} { r } { \mathbf { W } _ { N } ^ { ( \theta ) } = \mathbf { Q } _ { N } ^ { - \theta } \mathbf { W } _ { N } \mathbf { Q } _ { N } ^ { - \theta } , \qquad \mathbf { W } _ { N } ^ { ( \theta , \bullet ) } = \frac { 1 } { 2 } \big ( \mathbf { W } _ { N } ^ { ( \theta ) } \pm ( \mathbf { W } _ { N } ^ { ( \theta ) } ) ^ { \mathsf { T } } \big ) , } \end{array}
$$

with the sign + for $\bullet = s$ and − for ${ \bullet } = a$ . We write $\mathbf { D } _ { N } ^ { ( \theta , \bullet ) }$ for the diagonal matrix of row sums of $\mathbf { W } _ { N } ^ { ( \theta , \bullet ) }$ which discretizes $\mathcal { G } _ { \mathcal { F } } ^ { ( \theta , \bullet ) } [ \rho ]$ . Finally, for $\mathbf { f } = ( f ( X _ { 1 } ) , \ldots , f ( X _ { N } ) ) ^ { \mathsf { T } } \in \mathbb { R } ^ { N }$ , we define the discrete operators:

$$
\mathbf { P } ^ { ( \theta , \bullet ) } [ f ] = \big ( \mathbf { D } _ { N } ^ { ( \theta , s ) } \big ) ^ { - 1 } \Big ( \mathbf { W } _ { N } ^ { ( \theta , \bullet ) } \mathbf { f } - \mathbf { D } _ { N } ^ { ( \theta , \bullet ) } \mathbf { f } \Big ) , \qquad \mathrm { w i t h } \qquad \mathbf { L } ^ { s } [ f ] = \frac { \mathbf { P } ^ { ( \theta , s ) } [ f ] } { \varepsilon ^ { 2 } } , \quad \mathbf { L } ^ { a } [ f ] = \frac { \mathbf { P } ^ { ( \theta , a ) } [ f ] } { \varepsilon } .\tag{8}
$$

Both parts are normalized by the same $\mathbf { D } _ { N } ^ { ( \theta , s ) }$ , each is recentered by its own degree $\mathbf { D } _ { N } ^ { ( \theta , \bullet ) }$ , and each is rescaled by the order at which its limit appears in Theorem 2.

Theorem 4. Let $f \in \mathcal { C } ^ { 3 } ( \mathcal { M } )$ and let the bandwidth $\varepsilon = \varepsilon ( N )$ vary with the number of samples in such a way that $\varepsilon ( N ) \to 0$ and $\dot { N } \varepsilon ( N ) ^ { m + 2 } / \log N \to$ ∞ as $N \to \infty .$ . Then the discrete operators converge to the continuous ones, uniformly and almost surely: $f o r \bullet \in \{ s , a \}$

$$
\operatorname* { m a x } _ { 1 \leq i \leq N } | { \bf L } ^ { \bullet } [ f ] ( X _ { i } ) - { \mathcal L } ^ { \bullet } f ( X _ { i } ) | \xrightarrow [ N  \infty ] { a . s . } 0 .
$$

This result is proved in Appendix D.6, where the convergence rate is given explicitly.

## 5 MOMENT-RANDERS METRIC AND DIRECTED EMBEDDING

Theorem 2 shows that the operators $\mathcal { L } ^ { s }$ and ${ \mathcal { L } } ^ { a }$ depend on the Finsler metric $\mathcal { F }$ only through its Binet– Legendre metric, its Busemann–Hausdorff measure, and its centroid field. This motivates the definition of a Randers metric that has the same Binet–Legendre metric and centroid field as F, and that can be estimated from the graph operators of Section 4. Results stated in this section are proved in Appendix D.8, where additional details are also provided

## 5.1 MOMENT-RANDERS APPROXIMATION

Definition 5. (Moment-Randers metric) Let F be a Finsler metric on M with centroid field c and Binet– Legendre metric $g _ { \mathrm { B L } }$ . Set $\mathbf { H } ^ { - 1 } = g _ { \mathrm { B I . } } ^ { - 1 } - ( m + 2 ) \mathbf { c } \mathbf { c } ^ { \mathsf { T } }$ , and $\lambda = 1 - \| \mathbf { c } \| _ { \mathbf { H } } ^ { 2 } . \ I f \| \mathbf { c } ( x ) \| _ { \mathbf { H } } < 1$ for all $x \in \mathcal { M }$ , the moment-Randers metric $o f \mathcal { F }$ is the Randers metric Equation (1) with data $\left( \mathbf { A } _ { R } , b _ { R } \right)$

$$
\mathcal { F } _ { R } ( x , v ) = \sqrt { v ^ { \mathsf { T } } \mathbf { A } _ { R } ( x ) v } + b _ { R } ( x ) ^ { \mathsf { T } } v , \qquad \mathbf { A } _ { R } = \frac { \lambda \mathbf { H } + \mathbf { H } \mathbf { c } \mathbf { c } ^ { \mathsf { T } } \mathbf { H } } { \lambda ^ { 2 } } , \qquad b _ { R } = - \frac { \mathbf { H } \mathbf { c } } { \lambda } .
$$

Its drift satisfies $\| b _ { R } \| _ { \mathbf { A } _ { v } ^ { - 1 } } = \| \mathbf { c } \| _ { \mathbf { H } }$ , so the admissibility constraint of Equation (1) is exactly the hypothesis above. $\mathcal { F } _ { R }$ has the same Binet–Legendre metric and centroid field as F, but can have a different Busemann– Hausdorff measure<sup>2</sup>, and if Fis a Randers metric, then $\mathcal { F } _ { R } = \mathcal { F }$ as this is just the navigation map formulation (Section D.8). This means that ${ \mathcal { L } } ^ { a }$ is identical for both metrics, that $\mathcal { L } ^ { \bar { s } }$ only differs through the reference measure, and that the moment-Randers metric is a moment-matching Randers approximation of $\mathcal { F }$ . The construction of H relies on $g _ { \mathrm { B I } }$ and c. We relate the well-definedness of H and of the moment-Randers metric to the norm of c in the Binet–Legendre metric, which, as shown below, is accessible from the data.

Proposition 6. Let $\mathcal { F }$ be a Finsler metric and let $x \in \mathcal { M }$ be such that $\| \mathbf { c } ( x ) \| _ { g _ { \mathrm { B L } } } ^ { 2 } < ( m + 2 ) ^ { - 1 }$ . Then the matrix H(x) is well defined and positive definite, and

$$
\| { \mathbf { c } } ( x ) \| _ { g _ { \mathbb { R } ^ { \mathbf { L } } } } ^ { 2 } = \frac { \| { \mathbf { c } } ( x ) \| _ { \mathbf { H } } ^ { 2 } } { 1 + ( m + 2 ) \| { \mathbf { c } } ( x ) \| _ { \mathbf { H } } ^ { 2 } } , \qquad h e n c e \qquad \| { \mathbf { c } } ( x ) \| _ { \mathbf { H } } < 1 \Longleftrightarrow \| { \mathbf { c } } ( x ) \| _ { g _ { \mathbb { R } } } ^ { 2 } < ( m + 3 ) ^ { - 1 } .
$$

## 5.2 ESTIMATION THROUGH AN EMBEDDING

The quantities c and $g _ { \mathrm { B L } }$ cannot be observed directly. We therefore estimate them, through an embedding, from the operators $\mathcal { L } ^ { s }$ and ${ \mathcal { L } } ^ { a }$ , which are themselves approximated by the graph operators ${ \bf L } ^ { s }$ and ${ \bf L } ^ { a }$ (Theorem 4). Consider an embedding $\Psi ( x ) = ( \psi _ { 1 } ( x ) , \ldots , { \bar { \psi _ { \ell } } } ( x ) )$ and its Jacobian at $\dot { \boldsymbol { x } _ { } } , J ( \boldsymbol { x } ) \in \mathbb { R } ^ { \ell \times m }$ , assumed of full column rank m. By Equation (6), the push-forward of c by Ψ is recovered from the antisymmetric operator as

$$
V ( x ) : = \mathcal { L } ^ { a } [ \Psi ] ( x ) = c _ { 1 } J ( x ) \mathbf { c } ( x ) .
$$

Proposition 7. The carre du champ operator´ Γ associated with $\mathcal { L } ^ { s }$

$$
\Gamma ( x ) ^ { k l } = \frac { 1 } { 2 } \left( \mathcal { L } ^ { s } [ \psi _ { k } \psi _ { l } ] ( x ) - \psi _ { k } ( x ) \mathcal { L } ^ { s } [ \psi _ { l } ] ( x ) - \psi _ { l } ( x ) \mathcal { L } ^ { s } [ \psi _ { k } ] ( x ) \right) ,
$$

recovers the Binet–Legendre metric up to the embedding: $\Gamma ( x ) = c _ { 2 } J ( x ) g _ { \mathrm { B L } } ( x ) ^ { - 1 } J ( x ) ^ { \mathsf { T } }$ , where $c _ { 2 }$ is the constant ofTheorem 2.

Combining these results, the moment-Randers metric of $\mathcal { F }$ is well defined at x if and only $\begin{array} { r l r } {  { \mathrm { i f } \ \| \mathbf { c } ( x ) \| _ { g _ { \mathrm { B L } } } ^ { 2 } = } } \end{array}$ $\begin{array} { r } { \frac { c _ { 2 } } { c _ { 1 } ^ { 2 } } \| V ( x ) \| _ { \Gamma ^ { + } } ^ { 2 } < ( m + 3 ) ^ { - 1 } } \end{array}$ , where $\Gamma ^ { + }$ is the pseudo-inverse of Γ. This criterion only involves $\mathcal { L } ^ { s }$ and ${ \mathcal { L } } ^ { a }$ and allows us to define the embedded moment-Randers metric $\scriptstyle { \hat { \mathcal { F } } } _ { R }$

Definition 8. The embedded moment-Randers metric $\scriptstyle { \hat { \mathcal { F } } } _ { R }$ is defined as the moment-Randers metric (Definition 5) associated with the estimated Binet–Legendre metric $\hat { g } _ { B L }$ and the estimated centroid of the unit tangent ball cˆ. Let

$$
\begin{array} { l l } { { \hat { g } _ { B L } ( x ) ^ { - 1 } = \Gamma ( x ) , } } & { { \hat { \mathbf { c } } ( x ) = \frac { \sqrt { c _ { 2 } } } { c _ { 1 } } V ( x ) , } } \\ { { \hat { \mathbf { H } } ^ { - 1 } ( x ) = \Gamma ( x ) - ( m + 2 ) \frac { c _ { 2 } } { c _ { 1 } ^ { 2 } } V ( x ) V ( x ) ^ { \mathsf { T } } , } } & { { \hat { \lambda } ( x ) = 1 - \| \hat { \mathbf { c } } ( x ) \| _ { \hat { \mathbf { H } } } ^ { 2 } , } } \end{array}
$$

so that the embedded moment-Randers metric is given by

$$
\mathcal { \hat { F } } _ { R } ( x , v ) = \frac { 1 } { \hat { \lambda } ( x ) } \left( \sqrt { \hat { \lambda } ( x ) \| v \| _ { \hat { \mathbf { H } } } ^ { 2 } + \langle \hat { \mathbf { c } } ( x ) , v \rangle _ { \hat { \mathbf { H } } } ^ { 2 } } - \langle \hat { \mathbf { c } } ( x ) , v \rangle _ { \hat { \mathbf { H } } } \right) .
$$

Proposition 9. $I f \| \hat { \mathbf { c } } ( x ) \| _ { \hat { g } _ { B L } } < 1 / \sqrt { m + 3 }$ and J is of full column rank, then $\scriptstyle { \hat { \mathcal { F } } } _ { R }$ is related to the original moment-Randers metric $\mathcal { F } _ { R }$ by

$$
\hat { \mathcal { F } } _ { R } ( x , J ( x ) v ) = \frac { 1 } { \sqrt { c _ { 2 } } } \mathcal { F } _ { R } ( x , v ) , \quad \forall v \in T _ { x } \mathcal { M } .
$$

In particular, $\scriptstyle { \hat { \mathcal { F } } } _ { R }$ is a valid Randers metric on $J ( T _ { x } { \mathcal { M } } )$

## 5.3 EMBEDDING ALGORITHM

Algorithm 1 summarizes the procedure. The embedding Ψ can be any classical embedding, e.g. Isomap (Tenenbaum et al., 2000) or the leading eigenvectors of ${ \bf L } ^ { s }$ , in which case J has full column rank for ℓ large enough. $\mathbf { V } _ { i }$ and $\Gamma _ { i }$ are the discrete counterparts of V and Γ, obtained by replacing $\mathcal { L } ^ { \bullet }$ with $\mathbf { L } ^ { \bullet } ;$ here $Y ^ { k } \in \mathbb { R } ^ { N }$ is the k-th column of Y. When the data is given as a weighted directed graph (Section 6), ${ \bf W } _ { N }$ is its adjacency matrix.

Algorithm 1 Finsler directed embedding   
Require: Directed distances dst $( X _ { i } , X _ { j } )$ between samples, or a weighted directed graph with adjacency matrix ${ \bf W } _ { N } ;$   
kernel $K ,$ bandwidth $\varepsilon ,$ normalization ${ \bf \dot { \theta } } \in [ 0 , 1 ]$ , intrinsic dimension m, embedding dimension ℓ.   
Ensure: Embedded samples $Y _ { i } \in \mathbb { R } ^ { \ell }$ , drifts $\mathbf { V } _ { i }$ , and Randers metrics $\hat { \mathcal { F } } _ { R } ( Y _ { i } , \cdot )$   
1: $\mathbf { W } _ { N } ( i , j )  \varepsilon ^ { - m } K \bigl ( \mathrm { d s t } _ { \mathcal { F } } ( X _ { i } , X _ { j } ) / \varepsilon \bigr )$ {skipped if $\mathbf { W } _ { N }$ is given}   
2: $\mathbf { Q } _ { N }  ( \mathbf { D } _ { N } + \mathbf { D } _ { N } ^ { \prime } ) / 2 , \quad \mathbf { W } _ { N } ^ { ( \theta ) }  \mathbf { Q } _ { N } ^ { - \theta } \mathbf { W } _ { N } \mathbf { Q } _ { N } ^ { - \theta }$   
3: $\mathbf { W } _ { N } ^ { ( \theta , s ) } , \mathbf { W } _ { N } ^ { ( \theta , a ) } \gets$ symmetric and antisymmetric parts of $\mathbf { W } _ { N } ^ { ( \theta ) }$   
4: Build ${ \bf L } ^ { s }$ and ${ \bf L } ^ { a }$ from Equation $( 8 )$   
5: $\mathbf { Y } \gets ( \Psi ( X _ { 1 } ) , \dots , \Psi ( X _ { N } ^ { \textbf { \textit { n } } } ) ) ^ { \intercal } \in \mathbb { R } ^ { \tilde { N } \times \ell }$ {e.g. leading eigenvectors of ${ \bf L } ^ { s } \}$   
6: for $i = 1 , \ldots , N$ do   
7: $\begin{array} { r } { \mathbf { \dot { V } } _ { i }  ( \mathbf { L } ^ { a } \mathbf { Y } ) _ { i } , \quad \Gamma _ { i } ^ { k l }  \frac { 1 } { 2 } ( \mathbf { L } ^ { s } [ Y ^ { k } Y ^ { l } ] _ { i } - Y _ { i } ^ { k } \mathbf { L } ^ { s } [ Y ^ { l } ] _ { i } - Y _ { i } ^ { l } \mathbf { L } ^ { s } [ Y ^ { k } ] _ { i } ) } \end{array}$   
8: $\begin{array} { r } { \hat { { \bf c } } _ { i } \gets \frac { \sqrt { c _ { 2 } } } { c _ { 1 } } { \bf V } _ { i } , \quad \hat { { \bf H } } _ { i } ^ { - 1 } \gets \Gamma _ { i } - ( m + 2 ) \hat { { \bf c } } _ { i } \hat { { \bf c } } _ { i } ^ { \top } } \end{array}$   
9: if $\big \| \hat { \mathbf { c } } _ { i } \big \| _ { \Gamma _ { i } ^ { + } } ^ { 2 ^ { \bullet } } < ( m + 3 ) ^ { - 1 }$ then   
10: Set $\hat { \mathcal { F } } _ { R } ( Y _ { i } , \cdot )$ as in Definition 8   
11: end if   
12: end for

Table 1: Swiss roll. Relative error on $\| \hat { \mathbf { c } } \| _ { \hat { g } _ { B L } } ^ { 2 }$ and cosine between $V _ { i }$ and $c _ { 1 } J _ { i } \mathbf { c } ( X _ { i } )$ (medians over samples), and admissible fraction; medians over 3 seeds. Admissible means $\| \hat { \mathbf { c } } \| _ { \hat { g } _ { B L } } ^ { 2 } < ( m + 3 ) ^ { - 1 }$
<table><tr><td rowspan="3"> $\| \boldsymbol { b } \| _ { \mathbf { A } }$  -1</td><td colspan="3">Rel. error  $\| \hat { \mathbf { c } } \| _ { \hat { g } _ { B L } } ^ { 2 }$ </td><td colspan="3"> $\cos ( V _ { i } , J _ { i } \mathbf { c } )$ </td><td colspan="3">Admissible</td></tr><tr><td> $N = 1 0 0 0$ </td><td>2000</td><td>4000</td><td> $N = 1 0 0 0$ </td><td>2000</td><td>4000</td><td> $N = 1 0 0 0$ </td><td>2000</td><td>4000</td></tr><tr><td>0.1</td><td>0.17</td><td>0.14</td><td>0.13</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>0.3</td><td>0.15</td><td>0.13</td><td>0.12</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>0.5</td><td>0.13</td><td>0.11</td><td>0.10</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.98</td><td>0.99</td><td>0.99</td></tr><tr><td>0.7</td><td>0.11</td><td>0.10</td><td>0.09</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.89</td><td>0.91</td><td>0.91</td></tr><tr><td>0.9</td><td>0.09</td><td>0.08</td><td>0.07</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.46</td><td>0.47</td><td>0.50</td></tr></table>

## 6 EXPERIMENTS

Swiss roll dataset. We first consider a Randers metric on the Swiss roll, a 2D manifold embedded in $\mathbb { R } ^ { 3 }$ whose Riemannian part is supported on the manifold and whose drift b is tangent to it (Appendix C.1). We sample $N = 2 0 \bar { 0 } 0$ points from the manifold and construct a k-NN graph with $k = 1 0$ neighbors. With Isomap as Ψ, Figure 2a shows that we recover both the geometry of the manifold and the directionality of the edges. As c and $g _ { \mathrm { B L } }$ are known in closed form for a Randers metric (Proposition 43), Table 1 evaluates the estimated strength of the asymmetry for varying $\lVert b \rVert _ { \mathbf { A } ^ { - 1 } }$ and $N ,$ on a radius graph and with $J _ { i }$ estimated by finite differences (Appendix C.1). The cosine between the recovered and true drifts is 1.00 in every configuration, and the relative error on $\| \hat { \mathbf { c } } \| _ { \hat { g } _ { B L } } ^ { 2 }$ stays between 0.07 and 0.17. It decreases with N at every drift strength, although slowly. The bandwidth ε is the median Finsler distance from a sample to its 10-th nearest neighbor, and the graph construction is detailed in Appendix C.1. The rejections at large $\| b \| _ { \mathbf { A } _ { \alpha } ^ { - 1 } }$ are expected. All samples are admissible for the true metric, since $\mathcal { F } _ { R } = \mathcal { F }$ for a Randers metric, bu $\| \mathbf { c } \| _ { g _ { \mathrm { B I } } } ^ { 2 }$ approaches the threshold $( m + 3 ) ^ { - 1 }$ as the drift grows, so that small estimation errors suffice to cross it.

Approximating Matsumoto metrics. The method relies on a Randers approximation of the Finsler metric. We investigate whether the directional embeddings recovered from a non-Randers metric still capture the essential geometric properties of the data. We consider the Matsumoto metric (Matsumoto, 1992) induced by movement on a height map h over $[ - 3 . 2 , 3 . 2 ] \times [ - 2 . 4 , 2 . 4 ]$ (Matsumoto, 1989; Chansri et al., 2018), $\mathcal { F } = \alpha ^ { 2 } / ( \alpha - \beta )$ , where α is the Riemannian metric of the graph of h and $\beta = \mathrm { d } h$ (Appendix C.2). Figure 2c shows the height map and the drift recovered with Isomap embeddings. The drift is aligned with the gradient of h, the defining feature of the Matsumoto metric, which suggests that our method captures the main directional structure of non-Randers metrics.

![](images/b56819aa3c7cd71f7954733a72161697e894bdcef92297c1991a8f54aa5d8970.jpg)

![](images/70ec575304dc9b96098d9d5220256cbb932690163824c0177032d4d973ffea08.jpg)

![](images/83dc117597483ce10e4d24c1b873d0a60c0ffcdcf08a8443c88f4faa8ccc2a48.jpg)

![](images/ca746bf75c0b3eef7c76cbd50b844549df2264a930b01b27aaad7d255f9c5652.jpg)  
(a) 3D Swiss Roll Randers manifold.

![](images/4eecd2cc3406af0161e67ce26bec964f22d34dc8dfdef3df09d30c28604bd6ab.jpg)  
(b) DiSBM graph.

![](images/109c8b0184bd8a8332869b726711d3cc01893e8bf208e9fbe742e45407da917f.jpg)  
(c) Matsumoto manifold.  
Figure 2: Directional embeddings. Recovered directional embeddings on the three datasets. The top row shows the input data, and the bottom row shows the corresponding recovered embeddings and drifts.

Embedding directed graphs. Finally, we apply our method to general directed graphs, without an underlying manifold. We generate directed graphs from a Directed Stochastic Block Model (DiSBM, Appendix C.3). Figure 2b shows the leading eigenvectors of $\mathbf { L } ^ { s }$ for N = 1000 nodes and 15 communities: nodes cluster by community, and the drift follows the dominant, here backward, cyclic flow.

## 7 CONCLUSION AND PERSPECTIVES

We showed that Finsler geometry plays for directed graphs the role that Riemannian geometry plays for undirected ones. The symmetric and antisymmetric parts of the graph operators recover, respectively, the Binet–Legendre metric and the centroid field of the unit ball, which yields a moment-Randers approximation of the underlying metric and a directed embedding of the data. Experiments on Randers and Matsumoto manifolds and on directed stochastic block models show that both the geometry and the directionality are recovered. We hope that this work will open up new avenues for applied Finsler geometry in graph analysis. Future work could include the study of the spectral properties of the operators, the extension of the analysis to ‘standard’ directed graph operators such as the flow-based Laplacian (Chung, 2005), and the development of Finsler-based methods for directed graph analysis.

## AI USE STATEMENT

In this work, we used generative AI tools to assist in the writing of proofs, to design or provide feedback on research methodology or experiments, and to help with translation. We have not used generative AI tools to generate synthetic datasets, help develop theoretical models or conceptual frameworks, formulate mathematical claims, provide critical ingredients for proving mathematical claims, propose or refine hypotheses, implement methods, clean and reformat datasets, support qualitative and thematic data analysis, or interpret results. Additionally, we used generative AI tools to search for information and identify relevant literature. We have reviewed all AI-assisted work. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

This work proposes a novel method for representation learning. As such, there are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

## REPRODUCIBILITY STATEMENT

The code used to produce the results in this paper is available publicly at https://github.com/ Gwendal-Debaussart/Finsler-Embedding and is licensed under the MIT License. All datasets used in this work are synthetic and can be regenerated with the released code. The theoretical results in this work are accompanied by complete proofs in the appendix, and all assumptions are clearly stated. A primer is provided in the appendix to introduce the reader to the relevant concepts of Finsler geometry, and to help them check the definitions, notations and proofs used in the main text.

## AUTHOR CONTRIBUTIONS

This subsection follows the CRediT taxonomy.

Gwendal Debaussart-Joniec: Conceptualization, Formal Analysis, Investigation, Methodology, Validation, Visualization, Writing – original draft, Writing – review & editing.

Theau Blanchard: ´ Conceptualization, Formal Analysis, Investigation, Methodology, Software, Validation, Visualization, Writing – original draft, Writing – review & editing.

Argyris Kalogeratos: Supervision, Writing – review & editing.

## ACKNOWLEDGMENTS

The authors would like to deeply thank Jean-Marie Mirebeau for the discussions and feedback on the theoretical aspects of this work. Gwendal Debaussart-Joniec, and Argyris Kalogeratos acknowledge the support of the Industrial Analytics and Machine Learning (IdAML) Chair hosted at ENS Paris-Saclay, Universite´ Paris-Saclay. Theau Blanchard is funded by GEHealthcare.´

## REFERENCES

Georgios Arvanitidis, Soren Hauberg, Philipp Hennig, and Michael Schober. Fast and robust shortest paths on manifolds learned from data. In International Conference on Artificial Intelligence and Statistics, 2019.

Azam Asanjarani. A Finsler geometrical programming for the nonlinear complementarity problem of traffic equilibrium. Preprint arXiv:2109.01256, 2021.

David Bao, Shiing-Shen Chern, and Zhongmin Shen. An introduction to Riemann-Finsler geometry. Springer Science & Business Media, 2012.

Mikhail Belkin and Partha Niyogi. Laplacian eigenmaps and spectral techniques for embedding and clustering. In Advances in Neural Information Processing Systems, 2001.

Volker Bergen, Ruslan A Soldatov, Peter V Kharchenko, and Fabian J Theis. RNA velocity—current challenges and future perspectives. Molecular Systems Biology, 17(8):MSB202110282, 2021.

Jeff Calder and Nicolas Garcia Trillos. Improved spectral convergence rates for graph Laplacians on ε-graphs and k-NN graphs. Applied and Computational Harmonic Analysis, 60:123–175, 2022.

P Chansri, Pattrawut Chansangiam, and Sorin V Sabau. The geometry on the slope of a mountain. Preprint arXiv:1811.02123, 2018.

Da Chen, Jean-Marie Mirebeau, Huazhong Shu, and Laurent D Cohen. A region-based Randers geodesic approach for image segmentation. International Journal ofComputer Vision, 132(2):349–391, 2024.

Mo Chen, Qiong Yang, and Xiaoou Tang. Directed graph embedding. In International Joint Conference on Artificial Intelligence, 2007.

Fan Chung. Laplacians and the Cheeger inequality for directed graphs. Annals of Combinatorics, 9(1):1–19, 2005.

Ronald R Coifman and Stephane Lafon. Diffusion maps. ´ Applied and Computational Harmonic Analysis, 21(1):5–30, 2006.

Mihai Cucuringu, Huan Li, He Sun, and Luca Zanetti. Hermitian matrices for clustering directed graphs: insights and applications. In International Conference on Artificial Intelligence and Statistics, 2020.

Thomas Dages, Simon Weber, Ya-Wei Eileen Lin, Ronen Talmon, Daniel Cremers, Michael Lindenbaum,\` Alfred M. Bruckstein, and Ron Kimmel. Finsler multi-dimensional scaling: Manifold learning for asymmetric dimensionality reduction and embedding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Thomas Dages, Simon Weber, Daniel Cremers, and Ron Kimmel. Harnessing data asymmetry: Manifold\` learning in the Finsler world. Preprint arXiv:2603.11396, 2026.

Gwendal Debaussart-Joniec, Harry Sevi, Matthieu Jonckheere, and Argyris Kalogeratos. Parametrized power-iteration clustering for directed graphs. In International Conference on Machine Learning, 2026.

David L Donoho and Carrie Grimes. Hessian eigenmaps: Locally linear embedding techniques for highdimensional data. Proceedings ofthe National Academy ofSciences, 2003.

Michael Fanuel, Carlos M. Ala¨ ´ız, Angela Fern<sup>´</sup> andez, and Johan A.K. Suykens. Magnetic eigenmaps for´ the visualization of directed networks. Applied and Computational Harmonic Analysis, 44(1):189–199, 2018.

Paul Finsler. Uber Kurven und Fl <sup>¨</sup> achen in allgemeinen R ¨ aumen ¨ . Philos. Fak., Georg-August-Univ., 1918.

Barak Gahtan, Jacob Shpund, and Alex M Bronstein. Differentiable Randers-Finsler eikonal solvers. Computer Graphics Forum, pp. e70489, 2026.

Mingzhen He, Fan He, Lei Shi, Xiaolin Huang, and Johan A.K. Suykens. Learning with asymmetric kernels: Least squares and feature interpretation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(8):10044–10054, 2023.

Matthias Hein, Jean-Yves Audibert, and Ulrike von Luxburg. Graph Laplacians and their convergence on random neighborhood graphs. Journal ofMachine Learning Research, 8(6), 2007.

Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. Preprint arXiv:1609.02907, 2016.

Stephane S Lafon.´ Diffusion maps and geometric harmonics. Yale University, 2004.

John M. Lee. Introduction to Smooth Manifolds. Graduate Texts in Mathematics. Springer, 2 edition, 2012.

Federico Lopez, Beatrice Pozzetti, Steve Trettel, Michael Strube, and Anna Wienhard. Symmetric spaces for graph embeddings: A Finsler-Riemannian approach. In International Conference on Machine Learning, 2021.

Steen Markvorsen. A Finsler geodesic spray paradigm for wildfire spread modelling. Nonlinear Analysis: Real World Applications, 28:208–228, 2016.

Makoto Matsumoto. A slope of a mountain is a Finsler surface with respect to a time measure. Journal of Mathematics ofKyoto University, 29(1):17–25, 1989.

Makoto Matsumoto. Theory of Finsler spaces with (α, β)-metric. Reports on Mathematical Physics, 31(1): 43–83, 1992.

Vladimir S Matveev and Marc Troyanov. The Binet–Legendre metric in Finsler geometry. Geometry & Topology, 16(4):2135–2170, 2012.

John Melonakos, Eric Pichon, Sigurd Angenent, and Allen Tannenbaum. Finsler active contours. IEEE Transactions on Pattern Analysis and Machine Intelligence, 30(3), 2008.

Shin-ichi Ohta. Comparison Finsler Geometry. Springer Monographs in Mathematics. Springer, 2021.

Mingdong Ou, Peng Cui, Jian Pei, Ziwei Zhang, and Wenwu Zhu. Asymmetric transitivity preserving graph embedding. In ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016.

J. Wilson Peoples and John Harlim. Spectral convergence of symmetrized graph Laplacian on manifolds with boundary. Foundations ofData Science, 8(0):119–167, 2025.

Dominique Perrault-Joncas and Marina Meila. Directed graph embedding: an algorithm based on continuous limits of Laplacian-type operators. Advances in Neural Information Processing Systems, 24, 2011.

Christian Pfeifer. Finsler spacetime geometry in physics. International Journal of Geometric Methods in Modern Physics, 16(supp02):1941004, 2019.

Alison Pouplin, David Eklund, Carl Henrik Ek, and Søren Hauberg. Identifying latent distances with Finslerian geometry. Preprint arXiv:2212.10010, 2023.

Gunnar Randers. On an asymmetrical metric in the four-space of general relativity. Physical Review, 59(2): 195, 1941.

Frederik Mobius Rygaard and Søren Hauberg. GEORCE: A fast new control algorithm for computing¨ geodesics. Preprint arXiv:2505.05961, 2025.

Venu Satuluri and Srinivasan Parthasarathy. Symmetrizations for clustering directed graphs. In International Conference on Extending Database Technology, 2011.

Harry Sevi, Gwendal Debaussart-Joniec, Malik Hacini, Matthieu Jonckheere, and Argyris Kalogeratos. Generalized Dirichlet energy and graph Laplacians for clustering directed and undirected graphs. Transactions on Machine Learning Research, 2026.

Tony Shaska. Geometric learning and Finsler dissimilarity in weighted projective spaces. Preprint arXiv:2507.00001, 2025.

Yi-Bing Shen and Zhongmin Shen. Introduction to modern Finsler geometry. World Scientific Publishing Company, 2016.

Zhongmin Shen. Lectures on Finsler geometry. World Scientific, 2001.

Amit Singer. From graph to manifold Laplacian: The convergence rate. Applied and Computational Harmonic Analysis, 21(1):128–134, 2006. Special Issue: Diffusion Maps and Wavelets.

Stefan Sommer, Tom Fletcher, and Xavier Pennec. 1 - introduction to differential and Riemannian geometry. In Xavier Pennec, Stefan Sommer, and Tom Fletcher (eds.), Riemannian Geometric Statistics in Medical Image Analysis, pp. 3–37. Academic Press, 2020.

Qinghua Tao, Francesco Tonin, Alex Lambert, Yingyi Chen, Panagiotis Patrinos, and Johan A.K. Suykens. Learning in feature spaces via coupled covariances: asymmetric kernel SVD and Nystrom method. In¨ International Conference on Machine Learning, 2024.

Joshua B Tenenbaum, Vin de Silva, and John C Langford. A global geometric framework for nonlinear dimensionality reduction. Science, 290(5500):2319–2323, 2000.

Roman Vershynin. High-dimensional probability, volume 47 of Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2019.

Ulrike Von Luxburg. A tutorial on spectral clustering. Statistics and Computing, 17(4):395–416, 2007.

Simon Weber, Thomas Dages, Maolin Gao, and Daniel Cremers. Finsler-Laplace-Beltrami operators with\` application to shape analysis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Wei Wu, Jun Xu, Hang Li, and Satoshi Oyama. Asymmetric kernel learning. Technical report, Microsoft Research, 2010.

Won Dae Yeon. On the History of the Birth of Finsler Geometry at Gottingen. ¨ Journal for History of Mathematics, 2015.

Amber Yuan, Jeff Calder, and Braxton Osting. A continuum limit for the PageRank algorithm. European Journal ofApplied Mathematics, 33(3):472–504, 2022.

Ernst Zermelo. Uber das Navigationsproblem bei ruhender oder ver <sup>¨</sup> anderlicher Windverteilung. ¨ Journal of Applied Mathematics and Mechanics, 11(2):114–124, 1931.

Xitong Zhang, Yixuan He, Nathan Brugnone, Michael Perlmutter, and Matthew Hirn. MagNet: A neural network for directed graphs. In Advances in Neural Information Processing Systems, 2021.

Dengyong Zhou, Olivier Bousquet, Thomas Lal, Jason Weston, and Bernhard Scholkopf. Learning with¨ local and global consistency. Advances in Neural Information Processing Systems, 16, 2003.

Aaron Zweig, Mingxuan Zhang, David A Knowles, and Elham Azizi. Learning lineage-guided geodesics with finsler geometry. Preprint arXiv:2603.16708, 2026.

## APPENDIX CONTENTS

A Table of notations 14   
B A Primer of Riemannian and Finsler-Randers Geometry 14   
C Additional experimental results 19   
C.1 Swiss Roll dataset . 20   
C.2 Matsumoto Metric approximation 20   
C.3 Directed Stochastic Block Model 21   
D Proofs 23   
D.1 Forward and backward transport operators 23   
D.2 Proof of Theorem 1 24   
D.3 Proof of Theorem 2 30   
D.4 Links between the moments of the kernel and the unit ball 40   
D.5 Proof of Corollary 3 . 42   
D.6 Proof of Theorem 4 43   
D.7 Computation of the moments in the Randers setting 50   
D.8 The moment-Randers approximation 52   
D.9 Computation of the kernel moments 58

## A TABLE OF NOTATIONS

We provide in Tables 2 and 3 a summary of the main notations used throughout the paper.

## B A PRIMER OF RIEMANNIAN AND FINSLER-RANDERS GEOMETRY

In this section we provide a brief introduction to Riemannian geometry and its extension to Finsler geometry. For a more complete and formal introduction to those subjects, we refer the reader to Sommer et al. (2020) for Riemannian geometry, and to Ohta (2021) for a comprehensive treatment of Finsler geometry and Randers metrics. Throughout this paper, we use the Einstein summation convention, where an index repeated once as a superscript and once as a subscript is summed over, e.g. $\begin{array} { r } { \partial _ { i } f v ^ { i } = \sum _ { i = 1 } ^ { m } \partial _ { i } f v ^ { i } } \end{array}$

Riemannian geometry. Riemannian manifolds can be described in multiple ways. Generally, they are defined as smooth manifolds (i.e. topological spaces that are locally homeomorphic to Euclidean spaces, with smoothly compatible charts) endowed with a smoothly varying inner product on the tangent space at each point, called the Riemannian metric. This inner product allows one to define distances and angles on the manifold, see Figure 3b for an illustration. One may think of a Riemannian manifold as a curved space in which the local cost of crossing a region is governed by the metric. More formally, given a smooth manifold

Table 2: Notation table for the continuous setting. This table summarizes the notations attached to the manifold, the Finsler metric and the integral operators, with references to the equations where they are defined.  
Notation Description   
$\mathcal { M }$ Compact m-dimensional manifold embedded in $\overline { { \mathbb { R } ^ { D } } }$   
$\| w \| _ { M } , \langle w , w ^ { \prime } \rangle _ { M }$ $\| \boldsymbol { w } \| _ { M } ^ { \bar { 2 } } = \boldsymbol { w } ^ { \mathsf { T } } M \boldsymbol { w }$ and $\langle w , w ^ { \prime } \rangle _ { M } = w ^ { \mathsf { T } } M w ^ { \prime } ,$ , for $M \succ 0$   
$\mathcal { F } , \mathrm { d s t } _ { \mathcal { F } } ( \cdot , \cdot )$ Finsler metric and its associated distance   
Reversed Finsler metric   
$\alpha , \beta , \mathbf { A } , b$ Randers metric parameters   
$\lambda \mathcal { F } , i \mathcal { M }$ Reversibility constant $\begin{array} { r } { \operatorname* { s u p } _ { x , v } \mathcal { F } ( x , v ) / \mathcal { F } ( x , - v ) } \end{array}$ and injectivity radius of M (A2)   
$\mathfrak { m } _ { \mathrm { B H } } , \sigma _ { \mathrm { B H } }$ Busemann–Hausdorff measure on M and its density in a chart (Equation (4))   
$\lambda _ { x }$ Lebesgue measure on $T _ { x } { \mathcal { M } } .$ , in the coordinate basis   
$\mathrm { d i s t , V o l }$ Riemannian distance and volume induced on M by the ambient Euclidean norm   
$\mathcal { B } _ { x } , \mathcal { B } _ { x } ( r )$ Finsler unit ball $\{ \mathcal { F } ( x , \cdot ) \leq 1 \}$ and open ball $\{ \mathcal { F } ( \dot { x } , \cdot ) < r \}$ in the tangent space $T _ { x } { \mathcal { M } }$   
$B _ { \mathcal { F } } ^ { + } ( x , r )$ Forward metric ball $\{ y \in \mathcal { M } : \mathrm { d s t } _ { \mathcal { F } } ( x , y ) < r \}$ on the manifold M   
$\mathbb { B } ^ { \breve { m } }$ Euclidean unit ball of $\mathbb { R } ^ { m }$   
${ \mathcal { I } } _ { x }$ Indicatrix $\{ v \in T _ { x } \mathcal { M } : \mathcal { F } ( x , v ) = 1 \}$ , boundary of $\mathcal { B } _ { x }$   
$\mathbf { c } ( x ) , \mathbf { S } ( x )$ Centroid and raw second moment of the unit ball $\mathcal { B } _ { x }$ (Equations (22) and (23))   
$g _ { \mathrm { B L } }$ Binet–Legendre metric of F, with $g _ { \mathrm { B L } } ^ { i j } = ( m + 2 ) \mathbf { S } ^ { i j }$ (Equation (10))   
$\mathbb { P } , \rho$ Sampling distribution on M and its density $\rho { = } \mathrm { d } \mathbb { P } / \mathrm { d } { \mathfrak { m } } _ { \mathrm { B H } }$   
$K$ Radial kernel profile on $\mathbb { R } _ { + }$   
$W , W ^ { ( \theta ) }$ Directed kernel on ${ \mathcal { M } } \times { \mathcal { M } }$ and its θ-normalization   
$d , d ^ { \prime } , q _ { \varepsilon }$ Out- and in-degrees of $W ,$ and their half-sum $q _ { \varepsilon } = ( d { + } d ^ { \prime } ) / 2$   
$\mathcal { G } _ { \mathcal { F } } , \mathcal { G } _ { \mathcal { F } } ^ { \prime }$ Integral operators built from $W ; \mathcal { G } _ { \mathcal { F } } ^ { \prime } = \mathcal { G } _ { \mathcal { F } } = \mathcal { G } _ { \mathcal { F } } ^ { * }$ (Proposition 19)   
$\mathcal { G } _ { \mathcal { F } } ^ { ( \theta ) } , \mathcal { G } _ { \mathcal { F } } ^ { ( \theta , s ) } , \mathcal { G } _ { \mathcal { F } } ^ { ( \theta , a ) }$ θ-normalized integral operator, and its symmetric and antisymmetric parts   
${ \mathcal { P } } ^ { ( \theta , s ) } , { \mathcal { P } } ^ { ( \theta , a ) }$ Continuous transport operators built from $\mathcal { G } _ { \mathcal { F } } ^ { ( \theta , s ) }$ and $\mathscr { G } _ { \mathcal { F } } ^ { ( \theta , a ) }$ (Section 3)   
$\mathcal { L } ^ { s } , \mathcal { L } ^ { a }$ Their limits as $\varepsilon  0 \rangle$ a weighted Laplacian and a drift along c (Theorem 2)   
$\exp ^ { \mathcal { F } }$ Finsler exponential map of $\mathcal F$   
$G ^ { k } , N _ { i } ^ { i }$ Geodesic spray coefficients and nonlinear connection of F(Definition 27)   
$\tilde { \Gamma } _ { i j } ^ { k } , \mathrm { R i c } ^ { \mathcal { F } }$ Formal Christoffel symbols and Ricci curvature of F   
$\tau , S$ Distortion and S-curvature of F(Definition 18)

M, for any point $x \in \mathcal { M }$ we can define the tangent space $T _ { x } { \mathcal { M } }$ as the vector space of all tangent vectors at x. A Riemannian manifold is then a pair $( \mathcal { M } , \mathbf { A } )$ , where A is a smoothly varying inner product on the tangent space at each point,

$$
\langle \cdot , \cdot \rangle _ { \mathbf { A } ( x ) } : T _ { x } \mathcal { M } \times T _ { x } \mathcal { M }  \mathbb { R } , \qquad ( v , w ) \mapsto v ^ { \top } \mathbf { A } ( x ) w .
$$

In coordinates, components of tangent vectors carry upper indices, $v = \left( v ^ { i } \right)$ , while components of covectors, such as the partial derivatives $\partial _ { i } \bar { f }$ of a function, carry lower indices. We write $\mathbf { A } _ { i j }$ for the entries of A, so that $\langle v , w \rangle _ { \mathbf { A } } = \mathbf { A } _ { i j } v ^ { i } w ^ { j }$ , and $\mathbf { A } ^ { i j } = ( \mathbf { A } ^ { - 1 } ) _ { i j }$ for those of its inverse, so that $\mathbf { A } ^ { i k } \mathbf { A } _ { k j } = \delta _ { j } ^ { i }$ . The same convention applies to any metric. This inner product induces a natural local norm, $\| v \| _ { \mathbf { A } ( x ) } = \sqrt { v ^ { \mathsf { T } } \mathbf { A } ( x ) v }$ for $v \in T _ { x } { \mathcal { M } }$ , which in turn defines the length and energy of a curve $\gamma : [ 0 , 1 ] \to \mathcal { M }$

$$
L ( \gamma ) = \int _ { 0 } ^ { 1 } \| \dot { \gamma } ( t ) \| _ { \mathbf { A } ( \gamma ( t ) ) } \mathrm { d } t \quad \mathrm { ~ a n d ~ } \quad E ( \gamma ) = \int _ { 0 } ^ { 1 } \| \dot { \gamma } ( t ) \| _ { \mathbf { A } ( \gamma ( t ) ) } ^ { 2 } \mathrm { d } t .
$$

This lets us define the geodesic distance between two points as the length of the shortest path between them, which is also (up to reparametrization) the minimizer of the energy between these two points:

$$
\forall ( a , b ) \in { \mathcal { M } } ^ { 2 } , \operatorname { d s t } ( a , b ) = \operatorname* { m i n } _ { \gamma : [ 0 , 1 ] \to \mathcal { M } \atop \gamma ( 0 ) = a , \gamma ( 1 ) = b } L ( \gamma )\tag{9}
$$

Table 3: Notation table: discrete setting, hyper-parameters and constants. This table summarizes the notations attached to the sampled data, the matrices built from it, the hyper-parameters, and the moments and constants of the kernel.  
Notation Description   
$\overline { { X _ { 1 } , \ldots , X _ { N } } }$ A sequence of N i.i.d. samples from $\overline { { \mathbb { P } \ o n \ \mathcal { M } } }$   
${ \bf W } _ { N } , { \bf Q } _ { N }$ Empirical kernel matrix and half-sum of degrees ${ \bf Q } _ { N } = ( { \bf D } _ { N } + { \bf D } _ { N } ^ { \prime } ) / 2$ (Section 4)   
$\mathbf { D } _ { N } , \mathbf { D } _ { N } ^ { \prime }$ Diagonal matrices of the out- and in-degrees of $\mathbf { W } _ { N }$   
$\mathbf { W } _ { N } ^ { ( \theta ) } , \mathbf { W } _ { N } ^ { ( \theta , \bullet ) }$ θ-normalized matrix and its symmetric $( \bullet = s )$ and antisymmetric $( \bullet = a )$ parts   
$\mathbf { D } _ { \textrm { n r } } ^ { ( \bar { \theta } , \bullet ) }$ Diagonal matrix of the row sums of $\mathbf { W } _ { N } ^ { ( \theta , \bullet ) }$ , discretizing $\mathcal G _ { \mathcal F } ^ { ( \boldsymbol { \theta } , \bullet ) } [ \boldsymbol { \rho } ]$   
$\mathbf { P } ^ { ( \theta , \bullet ) }$ Discrete transport operators, discretizing $\mathscr { P } ^ { ( \theta , \bullet ) }$ (Equation (8))   
$\mathbf { L } ^ { s } , \mathbf { L } ^ { a }$ Rescaled discrete operators $\mathbf { P } ^ { ( \theta , s ) } / \varepsilon ^ { 2 }$ and $\mathbf { P } ^ { ( \theta , a ) } / \bar { \varepsilon }$ (Equation (8))   
$\hat { q } _ { N }$ Empirical counterpart of the normalization $q _ { \varepsilon }$   
$\Psi , Y _ { i }$ Spectral embedding $\Psi = \left( \psi _ { 1 } , \ldots , \psi _ { \ell } \right)$ of ${ \bf L } ^ { s }$ , and embedded points $Y _ { i } = \Psi ( X _ { i } )$   
$\mathbf { H } , \mathcal { F } _ { R }$ Moment-Randers matrix of Fand its moment-Randers metric (Definition 5)   
$\Gamma , V$ Estimated Binet–Legendre metric and drift, computed from $\mathcal { L } ^ { s } , \mathcal { L } ^ { a }$ (Proposition 7)   
$\hat { \mathbf { c } } , \hat { \mathbf { H } } , \hat { \mathcal { F } } _ { R }$ Empirical centroid, matrix, and moment-Randers metric estimated from the embedding   
$\varepsilon$ Kernel bandwidth parameter   
$\theta$ Normalization exponent of the kernel, $\theta \in [ 0 , 1 ]$   
$D$ Ambient dimension of the manifold M   
$m$ Intrinsic dimension of the manifold M   
$\ell$ Dimension of the embedding, i.e. number of eigenvectors of ${ \bf L } ^ { s }$ retained   
$m _ { n } , \Lambda ^ { k } , s _ { n } , \Upsilon$ Moments of the kernel K (Definition 21)   
$\tilde { m } _ { n }$ Normalized moments of the kernel, $\tilde { m } _ { n } = m _ { n } / m _ { 0 }$ (Proposition 35)   
$\mu _ { n }$ Raw moments of the radial profile, $\begin{array} { r } { \mu _ { n } = \int _ { 0 } ^ { + \infty } K ( r ) r ^ { m + n - 1 } \mathrm { d } r } \end{array}$   
$C _ { K } , \nu _ { K }$ Sub-exponential constants of the kernel: $\breve { K } ( r ) \leq C _ { K } e ^ { - \nu _ { K } r } \left( \mathrm { A } 1 \right)$   
$c _ { 1 }$ Constant $( m + 1 ) \mu _ { 1 } / ( m \mu _ { 0 } )$ appearing in ${ \mathcal { L } } ^ { a }$ (Theorem 2)   
$c _ { 2 }$ Constant $\mu _ { 2 } / ( 2 m \mu _ { 0 } )$ appearing in $\mathcal { L } ^ { \breve { s } }$ and Γ (Theorem 2 and Proposition 7)

Finsler geometry. Finsler manifolds are a natural extension of Riemannian manifolds that relax two of their defining constraints: the local cost need not be a quadratic form, and it need not be symmetric. This lets Finsler geometry model local asymmetry (the cost of crossing a region can depend on the direction of travel) while keeping a formulation close to that of Riemannian geometry, as illustrated in Figure 3c.

Definition 10 (Finsler metric). A Finsler metric on a smooth manifold M is a function $\mathcal { F } : T \mathcal { M }  [ 0 , \infty )$ such thatfor each $x \in \mathcal { M }$ and $v \in T _ { x } \mathcal { M }$ , thefollowing conditions hold:

• Positive homogeneity: $\mathcal { F } ( \boldsymbol { x } , t \boldsymbol { v } ) = t \mathcal { F } ( \boldsymbol { x } , \boldsymbol { v } )$ for all $t > 0$ and $v \in T _ { x } { \mathcal { M } } .$

• Strong convexity: The Hessian of $\mathcal { F } ^ { 2 }$ with respect to the velocity variable is positive definite for all nonzero $v \in T _ { x } { \mathcal { M } } .$

• Triangle inequality: $\begin{array} { r } { \mathcal { F } ( x , v + w ) \leq \mathcal { F } ( x , v ) + \mathcal { F } ( x , w ) f o r a l l v , w \in T _ { x } \mathcal { M } . } \end{array}$

Since homogeneity is only required for positive scalars $t ,$ these metrics induce local distances that depend on direction: we generally do not have $\mathcal { F } ( \boldsymbol { x } , \boldsymbol { v } ) = \mathcal { F } ( \boldsymbol { x } , - \boldsymbol { v } )$

Remark 11. In some textbooks, one may also encounter the notation $\mathcal { F } ( v )$ , dropping the explicit dependence on the base point x. Throughout this work, we will keep the explicit dependence on x on the Finsler-based quantities.

Definition 12 (Fundamental tensor). The fundamental tensor g of a Finsler metric $\mathcal { F }$ is given by

$$
g _ { i j } ( x , v ) = \frac { 1 } { 2 } \frac { \partial ^ { 2 } \mathcal { F } ^ { 2 } } { \partial v ^ { i } \partial v ^ { j } } ( x , v ) ,
$$

where $( x , v ) \in T \mathcal { M }$ and $v ^ { i }$ are the components ofv in a local coordinate system. Thefundamental tensor is a smoothly varying, positive definite bilinearform on the tangent space at each point, and it generalizes the notion ofa Riemannian metric to Finsler geometry.

A general Finsler tangent indicatrix $\{ v \in T _ { x } \mathcal { M } : \mathcal { F } ( x , v ) = 1 \}$ need not be an ellipsoid at all, as Finsler metrics are only required to be strongly convex, not quadratic. Nonetheless, the fundamental tensor always yields the best local Riemannian (quadratic) approximation of $\mathcal { F }$ at $( x , v )$

Similarly to Riemannian geometry, shortest paths under a Finsler metric are also minimizers of the energy of the curve. Using Euler’s homogeneous function theorem, one can show that the Finsler local energy satisfies

$$
\mathcal { F } ^ { 2 } ( \boldsymbol { x } , \boldsymbol { v } ) = \boldsymbol { v } ^ { \top } \boldsymbol { g } ( \boldsymbol { x } , \boldsymbol { v } ) \boldsymbol { v } = \| \boldsymbol { v } \| _ { \boldsymbol { g } ( \boldsymbol { x } , \boldsymbol { v } ) } ^ { 2 } .
$$

Consequently, we can define the length and energy of curves, and thus geodesic distances, on Finsler manifolds by replacing $\mathbf { A } ( \gamma _ { t } )$ with $g ( \gamma _ { t } , \dot { \gamma } _ { t } )$ in the definitions above. This lets us use the same class of algorithms as in the Riemannian case to solve the shortest-path problem, at marginal additional cost (Rygaard & Hauberg, 2025; Arvanitidis et al., 2019).

Balls. Several kinds of balls appear in this work. For $x \in \mathcal { M }$ and $r > 0$

$$
\mathcal { B } _ { x } = \big \{ v \in T _ { x } \mathcal { M } : \mathcal { F } ( x , v ) \leq 1 \big \} , \ \mathcal { B } _ { x } ( r ) = \big \{ v \in T _ { x } \mathcal { M } : \mathcal { F } ( x , v ) < r \big \} , \ B _ { \mathcal { F } } ^ { + } ( x , r ) = \big \{ y \in \mathcal { M } : \mathrm { d } \mathrm { s t } _ { \mathcal { F } } ( x , y ) < r \big \} .
$$

The first two live in the tangent space $T _ { x } \mathcal { M } \colon \mathcal { B } _ { x }$ is the (closed) unit tangent ball, whose boundary is the indicatrix $\mathcal { I } _ { x } = \{ v \in T _ { x } \mathcal { M } : \overline { { \mathcal { F } ( x , \overset { \cdot } { v } ) } } = 1 \}$ , and $\mathcal { B } _ { x } ( r )$ is the open tangent ball of radius $r ,$ which coincides with $r \mathcal { P } _ { x }$ up to its boundary. The third one is the forward metric ball and lives on the manifold ${ \mathcal { M } } .$ . Since dst<sub>F</sub> is not symmetric, it differs in general from the backward ball. Below the injectivity radius, $\exp _ { x } ^ { \mathcal { F } }$ maps $\mathcal { B } _ { x } ( r )$ onto $B _ { \mathcal { F } } ^ { + } ( x , r )$ . Finally, $\mathbb { B } ^ { m }$ denotes the Euclidean unit ball of $\mathbb { R } ^ { m }$ , and $\operatorname { V o l } _ { \operatorname { E u c } } ( { \mathbb { B } } ^ { m } )$ its volume. Unlike $\lambda _ { x } ( \mathcal { B } _ { x } )$ , it does not depend on x.

Randers metrics. A Randers metric (Randers, 1941) is a special type of Finsler metric that can be expressed as the sum of a Riemannian metric and a one-form.

Definition 13 (Randers metric). A Randers metric on a smooth manifold M is a Finsler metric $\mathcal { F }$ that can be expressed as

$$
\mathcal { F } ( \boldsymbol { x } , \boldsymbol { v } ) = \alpha ( \boldsymbol { x } , \boldsymbol { v } ) + \beta ( \boldsymbol { x } , \boldsymbol { v } ) ,
$$

where α is a Riemannian metric and β is a $1 - \mathit { f o r m } ,$ , such that $\alpha ( x , v ) ^ { 2 } = v ^ { \mathsf { T } } \mathbf { A } ( x ) v ,$ , with $\mathbf { A } : \mathcal { M }  \mathcal { S } _ { + + } ^ { D }$ ++ a smooth map from the manifold to the space of positive definite matrices, and $\beta ( x , v ) = b ( x ) ^ { \mathsf { T } } v$ , with $b : \mathcal { M } \to \mathbb { R } ^ { D }$ a smooth vector field on the manifold verifying $\| b ( x ) \| _ { \mathbf { A } ^ { - 1 } } < 1$

In coordinates, $\beta ( x , v ) = b _ { i } ( x ) v ^ { i } .$ , so that b carries a lower index as a covector. Its index is raised with A, $b ^ { i } = \mathbf { A } ^ { i j } b _ { j }$ , and $\| \dot { b } \| _ { \mathbf { A } ^ { - 1 } } ^ { 2 } = \dot { \mathbf { A } } ^ { i j } b _ { i } b _ { j } = b _ { i } b ^ { i }$

Randers metrics are a particular case of Finsler metrics that are generally easier to work with due to their specific structure: for such metrics we have an explicit form of g and $g ^ { - 1 }$ , enabling efficient implementation. Moreover, the Finsler indicatrix of a Randers metric is always an ellipsoid, simply not centered at the origin, obtained by translating the α-indicatrix by a vector determined by $b ( x )$

Example 14. Perhaps the simplest example ofa Randers metric is the Euclidean space $\mathbb { R } ^ { 2 }$ with the standard Euclidean metric $\bar { \alpha ( \boldsymbol { x } , \boldsymbol { v } ) } = \bar { \| \boldsymbol { v } \| _ { 2 } }$ and a constant vectorfield $b ( x ) = ( b _ { 1 } , b _ { 2 } )$ with $\bar { | | } b | | _ { 2 } < 1$ . In this case, the Randers metric is given by

$$
\mathcal { F } ( \boldsymbol { x } , \boldsymbol { v } ) = \| \boldsymbol { v } \| _ { 2 } + \boldsymbol { b } ^ { \top } \boldsymbol { v } = \sqrt { v _ { 1 } ^ { 2 } + v _ { 2 } ^ { 2 } } + b _ { 1 } v _ { 1 } + b _ { 2 } v _ { 2 } .
$$

The indicatrix of this Randers metric is an ellipse that is shifted in the direction of the vector b. One may verify that, in this context, geodesics are straight lines, but the distance between two points depends on the direction oftravel.

Berwald metrics. Among Finsler metrics, an important intermediate class, strictly more general than Riemannian metrics but with enough rigidity to retain some of their linear structure, is given by Berwald metrics.

Definition 15 (Berwald metric). A Finsler metric $\mathcal { F }$ is a Berwald metric if its formal Christoffel symbols $\tilde { \Gamma } _ { i j } ^ { k } ( x , v )$ (Ohta, 2021) do not depend on the direction $v ,$ i.e. $\tilde { \Gamma } _ { i j } ^ { k } ( x , v ) = \Gamma _ { i j } ^ { k } ( x )$ for some coefficients $\Gamma _ { i j } ^ { k }$ on ${ \mathcal { M } } .$

Every Riemannian metric is trivially Berwald, since its fundamental tensor $g _ { i j } ( x , v ) = \mathbf { A } _ { i j } ( x )$ does not depend on v to begin with. In the Randers case, $\mathcal { F } = \alpha + \beta$ is Berwald if and only if $\beta$ is parallel with respect to the Levi–Civita connection of $\alpha \left( \mathrm { O h t a } , 2 0 2 1 \right)$ , a condition on b that constant vector fields such as the one in the previous example trivially satisfy, but that a direction field varying freely across $\mathcal { M }$ , as we consider in our experiments, generally will not.

Binet–Legendre metric. The fundamental tensor $g ( x , v )$ of a Finsler metric depends on the direction v. A direction-free Riemannian metric is obtained by averaging over the unit tangent ball $\mathcal { B } _ { x }$ (Matveev & Troyanov, 2012).

Definition 16 (Binet–Legendre metric). The Binet–Legendre metric g<sub>BL</sub> of $\mathcal { F }$ is the Riemannian metric whose inverse is

$$
g _ { \mathrm { B L } } ^ { i j } ( x ) = \frac { m + 2 } { \lambda _ { x } ( \mathcal { B } _ { x } ) } \int _ { \mathcal { B } _ { x } } v ^ { i } v ^ { j } \mathrm { d } \lambda _ { x } ( v ) = ( m + 2 ) \mathbf { S } ^ { i j } ( x ) .\tag{10}
$$

The factor $m { + 2 }$ ensures that $g _ { \mathrm { B L } } = \mathbf { A }$ when $\mathcal { F } ( x , v ) = \sqrt { v ^ { \mathsf { T } } \mathbf { A } ( x ) v }$ is Riemannian (Appendix D.5). Here $g _ { \mathrm { B L } } ^ { i j }$ denotes the entries of the inverse of g , following the convention above. By contrast, the second moment $\mathbf { S } ^ { i j }$ of ${ \mathcal { B } } _ { x } .$ , like the kernel moment $m _ { 2 } ^ { i j }$ (Definition 21), is defined directly with upper indices, and is not the inverse of a matrix with lower indices. The metric $g _ { \mathrm { B L } }$ is what the symmetric limit $\mathcal { L } ^ { s }$ sees (Theorem 2), while the antisymmetric limit ${ \mathcal { L } } ^ { a }$ sees the centroid $\mathbf { c } ( x )$ of ${ \mathcal { B } } _ { x }$

Busemann–Hausdorff measure. Finsler manifolds carry no canonical volume; the two usual choices are the Busemann–Hausdorff and Holmes–Thompson measures, which coincide in the Riemannian case (Shen, 2001). We use the former.

Definition 17 (Busemann–Hausdorff measure). The Busemann–Hausdorff measure m<sub>BH</sub> is the measure on M which reads, in a chart, dm ${ \it \Delta } ( x ) = \sigma _ { \mathrm { B H } } ( x ) \mathrm { d } { \it \Delta }$ x, with density

$$
\sigma _ { \mathrm { B H } } ( x ) = \frac { \mathrm { V o l } _ { \mathrm { E u c } } ( \mathbb { B } ^ { m } ) } { \lambda _ { x } ( \mathcal { B } _ { x } ) } ,\tag{11}
$$

where $\lambda _ { x } ( \mathcal { B } _ { x } )$ is the Lebesgue volume $o f \mathcal { B } _ { x }$ in the coordinate basis $( \partial _ { i } | _ { x } )$ . Under a change of chart, σ<sub>BH</sub> and dx are multiplied by inversefactors, so that $\mathfrak { m } _ { \mathrm { B H } }$ does not depend on the chart.

Under ${ \mathfrak { m } } _ { \mathrm { B H } } .$ , every unit tangent ball has the volume of the Euclidean unit ball. This is what makes the zeroth kernel moment $m _ { 0 }$ constant (Proposition 35), so that the degree normalization behaves as in the Riemannian case.

Distortion and S-curvature. The Jacobian of $\exp _ { x } ^ { \mathcal { F } }$ with respect to m<sub>BH</sub> (Lemma 25) involves the mismatch between m and the direction-dependent volume $\sqrt { \operatorname* { d e t } g ( x , v ) } \mathrm { d } x$ , which is measured by the following two quantities.

Definition 18 (Distortion and $\mathbf { S } \cdot$ -curvature). For $z \in \mathcal { M }$ and $u \in T _ { z } \mathcal { M } \setminus \{ 0 \}$ , the distortion of $\mathcal { F }$ with respect to m<sub>BH</sub> $i s$

$$
\tau ( z , u ) = \log \left( \frac { \sqrt { \operatorname * { d e t } g _ { i j } ( z , u ) } } { \sigma _ { \mathrm { B H } } ( z ) } \right) ,
$$

![](images/4bd3c6d8d7c381c044e38424f6817f6af843fcb8043a2321fe0f726c6d07601a.jpg)  
(a) Euclidean distance through ambient space does not take into account the geometry of the manifold.

![](images/6b271483c051fa166d8ef941f584181e974a5cca60a470d141bad7533de55d81.jpg)  
(b) Riemannian distance follows a symmetric geodesic on the manifold. It captures the intrinsic geometry of the manifold without any directional information.

![](images/0e5a7827937132397aae4b99680d9b28eb527aa4f0c9d6c6d5d39cd2191d6b4d.jpg)  
(c) Finsler distance is directiondependent, with different paths for forward and reverse directions. It captures both the intrinsic geometry of the manifold and the directional information.  
Figure 3: Comparison of distance metrics on a manifold. Each panel shows the same manifold M with two points x and $_ { y , }$ but different notions of distance between them. (a) Euclidean distance is the straight line in ambient space. (b) Riemannian distance follows a symmetric geodesic on the manifold. (c) Finsler distance is direction-dependent, with different paths for forward (orange) and reverse (purple) directions.

and the S-curvature is its derivative along the geodesic $\gamma$ with $\gamma ( 0 ) = z$ and $\dot { \gamma } ( 0 ) = u ,$ i.e. $S ( z , u ) =$ $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } \tau ( \gamma ( t ) , \dot { \gamma } ( t ) ) \big | _ { t = 0 } . } \end{array}$

The S-curvature vanishes for Berwald metrics, and in particular for Riemannian ones (Shen & Shen, 2016, Proposition 4.3), but not for general Randers metrics. It produces the terms $s _ { 0 }$ and $s _ { 1 }$ of Definition 21, which are removed by the centering of $\mathcal { P } ^ { ( \theta , \bullet ) }$

Consequences of assumption (A2). The proofs use three uniform properties of $( { \mathcal { M } } , { \mathcal { F } } )$ , which follow from the compactness of M and the smoothness and positivity of Fon ${ \dot { T } } { \dot { \mathcal { M } } } \backslash \{ 0 \}$

(i) Bounded asymmetry. The reversibility constant $\begin{array} { r } { \lambda _ { \mathcal { F } } = \operatorname* { s u p } _ { x , v \neq 0 } \mathcal { F } ( x , v ) / \mathcal { F } ( x , - v ) } \end{array}$ is finite, as the supremum of a continuous function on the compact unit sphere bundle. Hence $\lambda _ { \mathcal { F } } ^ { - 1 } \mathrm { d s t } _ { \mathcal { F } } ( y , x ) \le$ dst<sub>F</sub> $( x , y ) \leq \lambda _ { \mathcal { F } }$ dst<sub>F</sub>(y, x).

(ii) Completeness. M is forward and backward complete, so that any two points are joined by a minimizing geodesic and $\exp ^ { \mathcal { F } }$ is defined on all of TM (Bao et al., 2012).

(iii) Injectivity radius. The injectivity radius is bounded below by some $i _ { \mathcal { M } } > 0 .$ , so that $\exp _ { x } ^ { \mathcal { F } }$ is a diffeomorphism from $\mathcal { B } _ { x } ( i _ { \mathcal { M } } )$ onto its image for every x.

## C ADDITIONAL EXPERIMENTAL RESULTS

We provide in this section additional details on the experiments presented in Section 6. We show how we construct the Randers metric and the drift for the Swiss Roll dataset, the Matsumoto metric approximation, and the Directed Stochastic Block Model (DiSBM).

On the illustrations. The presented method recovers the drift field as a pointwise quantity at each embedded sample. In the figures, we display the drift field as a vector field on a grid of points in the embedding space, obtained by interpolating the drift vectors at the embedded samples with a neural network.

## C.1 SWISS ROLL DATASET

We consider the Swiss Roll dataset $( X _ { i } ) _ { i }$ generated using sklearn with $N = 2 0 0 0$ points and a noise of 0.1. We consider this dataset as sampled from a Randers manifold. The Riemannian metric is given by:

$$
\mathbf { A } _ { i i } ^ { - 1 } ( z ) = \sum _ { k } \exp \left( - \frac { \| z - z _ { k } \| ^ { 2 } } { h } \right) + \xi , \quad \mathrm { a n d } \quad \mathbf { A } _ { i j } ^ { - 1 } = 0 \mathrm { ~ f o r ~ } i \neq j ,
$$

where $\xi = 1 0 ^ { - 3 }$ and h is chosen as the average distance to the 5th nearest neighbor.

We can then construct the drift as tangential to the dataset. Let $\varphi _ { i } = \operatorname { a t a n 2 } ( X _ { i , z } , X _ { i , x } )$ be the angle of the point $X _ { i }$ in the xz-plane. The initial tangent vector is given by $b ( X _ { i } ) = ( - \sin ( \varphi _ { i } ) , 0 , \cos ( \varphi _ { i } ) )$ ), which is then normalized so that $\| b ( X _ { i } ) \| _ { \mathbf { A } ^ { - 1 } } = 0 . 9$

The directed graph is computed as a KNN graph with $k = 1 0$ neighbors, and the edge weights, i.e. geodesic distances between nodes, are computed using the GEORCE algorithm (Rygaard & Hauberg, 2025). The ε in the Gaussian kernel is chosen as the standard deviation of the geodesic distances between the nodes. The points are then embedded in $\mathbb { R } ^ { 2 }$ using the Isomap algorithm. The embedding is then used to estimate the Randers metric and the drift following the method described in the paper.

Quantitative evaluation. For Table 1, we keep the same metric and vary the drift norm $\beta = \| b \| _ { \mathbf { A } ^ { - 1 } }$ and the number of points, with 3 seeds per configuration. Here, we use radius graphs rather than $k { \mathrm { - N N } }$ graphs. With a strong drift, the kernel extends further along the drift than across it, so that a k-NN graph would truncate it unevenly and bias its moments, and hence the limit of Theorem 2. The bandwidth ε is the median Finsler distance from a sample to its 10-th nearest neighbor. The graph is then truncated to the ball dst $( X _ { i } , X _ { j } ) < 3 \varepsilon$ , outside of which the kernel is negligible. As above, the edge weights are $K ( \mathrm { d s t } _ { \mathcal { F } } ( X _ { i } , X _ { j } ) / \varepsilon )$ with the Gaussian kernel, where dst is computed with GEORCE. In other words, the graph keeps every pair on which the kernel is non-negligible, and the geometry is carried by the weights.

The ground truth c and $g _ { \mathrm { B L } }$ is given by Proposition 43. The tangent plane of the roll is estimated by a local PCA on the 15 nearest neighbors. The Jacobian $J _ { i }$ of Isomap at X is estimated by least squares on the same neighbors, $Y _ { j } - Y _ { i } \approx J _ { i } ( \bar { X } _ { j } - X _ { i } )$ in these tangent coordinates. It is only needed for the direction of the drift, since $\| \dot { \hat { \mathbf { c } } } \| _ { \hat { g } _ { B L } }$ does not depend on the embedding. Table 1 reports the cosine between $V _ { i }$ and its limit $c _ { 1 } J _ { i } \mathbf { c } ( X _ { i } )$ : the direction of the drift is recovered at every strength and sample size.

## C.2 MATSUMOTO METRIC APPROXIMATION

We focus on the case of a graph sampled from a Matsumoto metric (Matsumoto, 1992), more precisely on the metric induced by movement on a height map (Matsumoto, 1989; Chansri et al., 2018). We define on $[ - 3 . 2 , 3 . 2 ] \times [ - 2 . 4 , \dot { 2 . } 4 ]$ the height function h:

$$
h ( x , y ) = 0 . 4 \sin ( x ) \cos ( y ) + 0 . 1 5 \exp \left( - { \left( \left( x + 1 \right) ^ { 2 } + \left( y - 0 . 5 \right) ^ { 2 } \right) } \right) - 0 . 1 5 \exp \left( - { \left( \left( x - 1 \right) ^ { 2 } + \left( y + 0 . 5 \right) ^ { 2 } \right) } \right) .
$$

The Matsumoto metric is then given by:

$$
\begin{array} { r l } & { \alpha ( ( x , y ) , v ) ^ { 2 } = ( 1 + \partial _ { x } h ( x , y ) ^ { 2 } ) v _ { x } ^ { 2 } + 2 \partial _ { x } h ( x , y ) \partial _ { y } h ( x , y ) v _ { x } v _ { y } + ( 1 + \partial _ { y } h ( x , y ) ^ { 2 } ) v _ { y } ^ { 2 } , } \\ & { \beta ( ( x , y ) , v ) = \partial _ { x } h ( x , y ) v _ { x } + \partial _ { y } h ( x , y ) v _ { y } , } \\ & { \mathcal F ( ( x , y ) , v ) = \cfrac { \alpha ( ( x , y ) , v ) ^ { 2 } } { \alpha \left( ( x , y ) , v \right) - \beta \left( ( x , y ) , v \right) } . } \end{array}
$$

We then sample $N \ = \ 3 0 0 0$ points uniformly on $[ - 3 . 2 , 3 . 2 ] \times [ - 2 . 4 , 2 . 4 ]$ that yield the samples $( x _ { i } , y _ { i } , h ( x _ { i } , y _ { i } ) ) _ { i }$ Similarly to the Swiss Roll dataset, we construct a directed graph from the sampled points using a KNN graph with $k = 1 0$ neighbors, and where the edge weights are computed using the GEORCE algorithm (Rygaard & Hauberg, 2025). The ε in the Gaussian kernel is chosen as the standard deviation of the geodesic distances between the nodes. The samples are then embedded in $\mathbb { R } ^ { 2 }$ using the Isomap algorithm. The embedding is then used to estimate the Randers metric and the drift following the method described in the paper.

## C.3 DIRECTED STOCHASTIC BLOCK MODEL

Formal definition. We start by giving a formal definition of the DiSBM. Let N be the number of nodes in the graph, B the number of communities, and $\boldsymbol n \in \mathbb { N } ^ { B }$ the number of nodes in each community. We denote by $\breve { \Pi } \in [ 0 , 1 ] ^ { B \times B }$ the block matrix that defines the connection probabilities between communities, such that $\begin{array} { r } { \dot { \sum _ { j } } \Pi _ { i , j } = \dot { 1 } } \end{array}$ for every i. We consider in particular the case of directed graphs that are generated from a circular block model, where the communities are arranged in a circle and the connection probabilities are defined as follows:

$$
\Pi _ { i , j } = { \left\{ \begin{array} { l l } { p } & { { \mathrm { i f ~ } } j = i + 1 \mod B , } \\ { q } & { { \mathrm { i f ~ } } j = i - 1 \mod B , } \\ { r } & { { \mathrm { i f ~ } } j = i , } \\ { 0 } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }
$$

where $p , q , r \in [ 0 , 1 ]$ and $p + q + r = 1$ . The probabilities $p , q , r$ are the probabilities of connecting to the next, previous, and same community, respectively. The adjacency matrix $\mathbf { Q } \in \{ 0 , 1 \} ^ { N \times N }$ of the graph is then generated by sampling each entry $\mathbf { Q } _ { i , j }$ independently according to a Bernoulli distribution with parameter $\Pi _ { c ( i ) , c ( j ) }$ , where $c ( i )$ is the community label of node i. The model is illustrated in Figure 4 for $B = 3$

Experiment setting. We consider a DiSBM with $B = 1 5$ communities, $N = 1 0 0 0 , r = 0 . 1 , p = 0 . 4$ and $q = 0 . 5$ . Note that $p > q$ is not necessary: as soon as $p \neq q ,$ , the flow between communities has a preferred orientation, forward if $p > q$ and backward if $q > p .$ . The community labels are drawn uniformly at random, and the adjacency matrix is generated according to the DiSBM model. The ε in the kernel is chosen as 1. The points are then embedded in $\mathbb { R } ^ { 2 }$ using the first two non-trivial eigenvectors of $\mathbf { L } ^ { s }$ , with $\theta = 1$ . The embedding is then used to estimate the Randers metric and the drift following the method described in the paper.

Direction of the drift. Although DiSBM has no underlying manifold, the direction of the flow between communities is known: forward if $p > q$ , backward if $q > p ,$ , and none if $p = q$ . We measure whether the recovered drift follows it as $p - q$ varies. The N, B and r parameters remains as in the previous paragraph. Let $Y _ { i } \in \mathbb { R } ^ { 2 }$ be the embedding of node i and $c ( i )$ its community. The center of community k is

$$
\mu _ { k } = { \frac { 1 } { | C _ { k } | } } \sum _ { j \in C _ { k } } Y _ { j } ,
$$

where $C _ { k } = \{ j : c ( j ) = k \}$ is the set of its nodes. The forward direction at community k is the tangent to the cycle of communities, $t _ { k } = \mu _ { k + 1 } - \mu _ { k - 1 }$ , with indices taken modulo B. The chord $\mu _ { k + 1 } - \mu _ { k }$ would be off by half the angle between two consecutive communities. The signed cosine of node i is the cosine between its drift $\mathbf { V } _ { i }$ and the forward direction at its community,

$$
s _ { i } = \frac { \left. \mathbf { V } _ { i } , t _ { c ( i ) } \right. } { \left\| \mathbf { V } _ { i } \right\| \left\| t _ { c ( i ) } \right\| } \in [ - 1 , 1 ] ,
$$

which is close to 1 when the drift points forward, close $\mathrm { t o } - 1$ when it points backward, and spread around 0 when it has no preferred direction. It is signed because it is measured against the forward direction whatever the actual flow, so that it changes sign when the flow is reversed. We also report the estimated strength of the asymmetry, $\| \hat { \mathbf { c } } _ { i } \| _ { \hat { g } _ { B L } }$ . Figure 5 shows that the drift points in the direction of the flow as soon as $p \neq q \colon$ the median signed cosine is already $\pm 0 . 8 9$ for $| p - q | = 0 . 0 2$ , and above 0.99 from $| p - q | = 0 . 1$ on. When $p = q .$ , the drift has no preferred direction. The strength of the asymmetry is symmetric in $p - q$ and grows linearly with $| p - q |$

![](images/5fb099549018f306622c69e3fff61e132f625090f37a8f32048d7bfd4d586f6e.jpg)  
(a) Block model. Each community sends its edges to the next community with probability p, to the previous one with probability q, and to itself with probability r.

![](images/8606d9703b7627e3e9a2d475cd084bbb1ad31a5be2051b267ac606b7bb755f39.jpg)  
(b) Sampled graph. Intra-community edges (gray) have no preferred orientation, while edges between communities mostly follow the cycle (colored by source), edges going against the cycle are less frequent (dashed).

Figure 4: Cyclic Directed Stochastic Block Model. Illustration with $B = 3$ communities and $p > q . \ ( \mathrm { a } )$ Connection probabilities of the block matrix Π. (b) A realization of the adjacency matrix $\mathbf { Q } .$  
![](images/c45c0be0f9b28b5951769df386df2df9df530000f103902a2ef7a612cc87c75c.jpg)  
(a) Signed cosine according to the direction of the flow.

![](images/0bc5d4a02b222d2f3f614e15e50d43423709d7a5889fdc49ac27d75f31865322.jpg)  
(b) Estimated strength of the asymmetry.  
Figure 5: Embedding of the DiSBM and its dynamics. Direction of the drift on the DiSBM as a function of $p - q ,$ with $N = 1 0 0 0$ nodes, $B = 1 5$ communities and $r = 0 . 1$ . Lines are medians over the nodes of 5 random graphs, and shaded bands their 10–90% range.

## D PROOFS

This appendix gathers the proofs of the results stated in the main text.

## D.1 FORWARD AND BACKWARD TRANSPORT OPERATORS

Proposition 19 (Relation between the forward and backward transport operators). The backward transport operator $\mathcal { G } _ { \mathcal { F } } ^ { \prime }$ is equal to the transport operator of the reversed Finsler metric , and is the $L ^ { 2 } ( \mathcal { M } , \mathfrak { m } _ { \mathrm { B H } } )$ adjoint of $\bar { \mathcal { G } } _ { \mathcal { F } }$ , i.e. $\mathcal { G } _ { \mathcal { F } } ^ { \prime } = ( \mathcal { G } _ { \mathcal { F } } ) ^ { * } = \dot { \mathcal { G } } _ { \overleftarrow { \mathcal { F } } }$

Before proving Proposition 19, we prove the following lemma, which allows us to formally work with the reversed Finsler metric and its associated distance.

Lemma 20 (Finsler metric and its reverse). The reversed Finsler metric is a Finsler metric on M. Moreover, its geodesics are the reverse ofthe geodesics of F, and the geodesic distance $d s t _ { \mathfrak { F } }$ is the reverse ofthe geodesic distance dst .

ProofofLemma 20. First, we show that is indeed a Finsler metric. By definition, $\overleftarrow { \mathcal { F } } ( x , v ) = \mathcal { F } ( x , - v )$ for all $x \in \mathcal { M }$ and $v \in T _ { x } \mathcal { M }$ . Hence $\overleftarrow { \mathcal { F } } ( x , v ) > 0$ for all $v \neq 0 ,$ , and $\overleftarrow { \mathcal { F } } ( x , 0 ) = 0$ . For any $t > 0$ , we have $\overleftarrow { \mathcal { F } } ( x , t v ) = \mathcal { F } ( x , - t v ) = t \mathcal { F } ( x , - v ) = t \overleftarrow { \mathcal { F } } ( x , v )$ , satisfying the positive homogeneity property. Finally, the Hessian of $\overleftarrow { \succcurlyeq } ^ { 2 }$ with respect to v is the same as the Hessian of $\mathcal { F } ^ { 2 }$ with respect to −v, which is positive definite by the definition of a Finsler metric. Hence, satisfies all three properties of a Finsler metric.

For the second claim, let $\gamma : [ 0 , T ] \to \mathcal { M }$ be the geodesic of F connecting two points $x , y \in { \mathcal { M } }$ , so that $\gamma ( 0 ) = x$ and $\gamma ( T ) = y$ . Then, the reverse curve $\sigma ( t ) = \gamma ( T - t )$ is a curve connecting y to x. We have that $\dot { \sigma } ( t ) = - \dot { \gamma } ( T - t )$ , so that

$$
\begin{array} { r l r } { \displaystyle \int _ { 0 } ^ { T } \overleftarrow { \mathfrak { F } } ( \sigma ( t ) , \dot { \sigma } ( t ) ) \mathrm { d } t = \int _ { 0 } ^ { T } \mathfrak { F } ( \sigma ( t ) , - \dot { \sigma } ( t ) ) \mathrm { d } t } \\ { \displaystyle } & { = \int _ { 0 } ^ { T } \mathfrak { F } ( \gamma ( T - t ) , \dot { \gamma } ( T - t ) ) \mathrm { d } t } & { \qquad \mathrm { b y ~ t h e ~ d e f i n i t i o n ~ o f ~ } \sigma } \\ { \displaystyle } & { = \int _ { 0 } ^ { T } \mathfrak { F } ( \gamma ( s ) , \dot { \gamma } ( s ) ) \mathrm { d } s } & { \qquad \mathrm { b y ~ c h a n g e ~ o f ~ v a r i a b l e ~ } s = T - t } \\ { \displaystyle } & { = \mathrm { d } s \mathfrak { t } _ { \mathfrak { F } } ( x , y ) . } \end{array}
$$

Since $\gamma$ is the geodesic of $\mathcal { F }$ , it minimizes the integral of $\mathcal { F }$ along curves connecting x and $y .$ Therefore, σ minimizes the integral of along curves connecting y and x, and is thus the geodesic of connecting y to x. This shows that the geodesics of are the reverse of the geodesics of $\mathcal { F }$ , and that the geodesic distance $\mathrm { d s t } _ { \overleftarrow { \mathcal { F } } } ( x , y ) = \mathrm { d s t } _ { \mathcal { F } } ( y , x )$ □

Applying this lemma, we can now prove Proposition 19.

Proof of Proposition 19. By Lemma 20, ds $\mathfrak { t } _ { \mathcal { F } } ( x , y ) = \mathrm { d s t } _ { \mathcal { F } } ( y , x )$ for all $x , y \in { \mathcal { M } }$ , so that, denoting by $W _ { \overleftarrow { \mathcal { F } } }$ the kernel built from in place of F,

$$
W ( x , y ) = \frac { 1 } { \varepsilon ^ { m } } K \left( \frac { \mathrm { d } \mathrm { s t } _ { \mathcal { F } } ( x , y ) } { \varepsilon } \right) = \frac { 1 } { \varepsilon ^ { m } } K \left( \frac { \mathrm { d } \mathrm { s t } _ { \mathcal { F } } ( y , x ) } { \varepsilon } \right) = W _ { \overleftarrow { \ast } } ( y , x ) .
$$

Hence $\begin{array} { r } { \mathcal { G } _ { \mathcal { F } } ^ { \prime } f ( x ) = \int _ { \mathcal { M } } W ( y , x ) f ( y ) \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } ( y ) = \int _ { \mathcal { M } } W _ { \mathfrak { F } } ( x , y ) f ( y ) \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } ( y ) = \mathcal { G } _ { \mathfrak { F } } f ( x ) . } \end{array}$

For the adjoint claim, by Fubini’s theorem,

$$
\begin{array} { l } { \displaystyle \langle \mathfrak { L } _ { \mathcal { F } } f , h \rangle = \iint _ { \mathcal { M } \times \mathcal { M } } W ( x , y ) f ( y ) h ( x ) \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } ( y ) \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } ( x ) } \\ { \displaystyle \quad = \int _ { \mathcal { M } } f ( y ) \left( \int _ { \mathcal { M } } W ( x , y ) h ( x ) \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } ( x ) \right) \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } ( y ) = \langle f , \mathfrak { G } _ { \mathcal { F } } ^ { \prime } h \rangle . } \end{array}
$$

## D.2 PROOF OF THEOREM 1

Despite its simple statement, the proof of Theorem 1 is rather dense, and requires a number of technical lemmas. For ease of reading, we first give a proof sketch, and defer the technical lemmas to the paragraphs below. We strongly encourage the interested reader to read Ohta (2021) and Shen & Shen (2016) for a comprehensive treatment of Finsler geometry, and Coifman & Lafon (2006) for a similar proof in the Riemannian case.

Proof sketch. The proof of Theorem 1 relies on the expansion of the integral operator $\mathcal { G } _ { \mathcal { F } }$ . To do $\mathbf { S O } ,$ we expand the function $f$ around a point x using the Finsler exponential map $\exp _ { x } ^ { \mathcal { F } }$ . This expansion will be used to expand the integral operator $\mathcal { G } _ { \mathcal { F } }$ in terms of the derivatives of $f$ at $x ,$ and the moments of the kernel $K$ This expansion is possible since we assume that the kernel $K$ is sub-exponential, and thus that the integral can be localized around $x$ for small $\varepsilon .$ The change of variable through $\exp _ { x } ^ { \mathcal { F } }$ brings a Jacobian, which we expand in terms of the density $\sigma _ { \mathrm { B H } }$ , the S-curvature and the Ricci curvature of $\mathcal { F }$ (Lemma 25). Once this expansion is obtained, we can then use Proposition 19 and the fact that ${ \mathcal { G } } _ { \mathcal { F } } ^ { \prime } = { \mathcal { G } } _ { \overleftarrow { \mathcal { F } } }$ to obtain the expansion of $\mathcal { G } _ { \mathcal { F } } ^ { \prime }$ . Finally, one can note the following identity:

$$
\mathcal { G } _ { \mathcal { F } } ^ { ( \theta ) } [ f ] = \frac { 1 } { { q _ { \varepsilon } } ^ { \theta } } \mathcal { G } _ { \mathcal { F } } \left[ \frac { f } { { q _ { \varepsilon } } ^ { \theta } } \right] ,
$$

so that the expansion of $\mathcal { G } _ { \mathcal { F } } ^ { ( \theta ) }$ can be easily obtained from the expansion of $\mathcal { G } _ { \mathcal { F } }$ . The final result is obtained by taking the symmetric and antisymmetric parts of the expansion of $\mathcal { G } _ { \mathcal { F } } ^ { ( \theta ) }$ , and applying the proper normalization.

We give here the form of the differential operators that appear in the expansion of $\mathcal { G } _ { \mathcal { F } }$ . To do so, we start by defining the kernel moments of K as follows:

Definition 21 (Kernel moments). Let $x \in \mathcal { M }$ and $n \in \mathbb { N } .$ . Using the formal Christoffel symbols $\tilde { \Gamma } _ { i j } ^ { k }$ , the Ricci curvature $R i c ^ { \mathcal { F } }$ and the S-curvature S (Definition 18) of the Finsler metric F, S<sup>˙</sup> denoting the derivative of S along the geodesic flow, we define

$$
m _ { n } ( x ) = \sigma _ { \mathrm { B H } } ( x ) \int _ { T _ { x } , \mathcal { M } } K ( \mathcal { F } ( x , v ) ) v ^ { \otimes n } \mathrm { d } \lambda _ { x } ( v ) ,
$$

$$
\Lambda ^ { k } ( x ) = \sigma _ { \mathrm { B H } } ( x ) \int _ { T _ { x } , \mathcal { M } } K ( \mathcal { F } ( x , v ) ) \tilde { \Gamma } _ { i j } ^ { k } ( x , v ) v ^ { i } v ^ { j } \mathrm { d } \lambda _ { x } ( v ) ,
$$

$$
s _ { n } ( x ) = \sigma _ { \mathrm { B H } } ( x ) \int _ { T _ { x } , \mathcal { M } } K ( \mathcal { F } ( x , v ) ) S ( x , v ) v ^ { \otimes n } \mathrm { d } \lambda _ { x } ( v ) ,
$$

$$
\begin{array} { l } { \displaystyle \Upsilon ( x ) = \sigma _ { \mathrm { B H } } ( x ) \int _ { T _ { x } \mathcal { M } } K ( \mathcal { F } ( x , v ) ) } \\ { \displaystyle \qquad \times \left( S ( x , v ) ^ { 2 } - \dot { S } ( x , v ) - \frac 1 3 R i c ^ { \mathcal { F } } ( x , v ) \right) \mathrm { d } \lambda _ { x } ( v ) , } \end{array}
$$

where $\sigma _ { \mathrm { B H } }$ is the density ofthe Busemann–Hausdorffmeasure (Definition 17).

With this definition at hand, we can clarify the definition of the differential operators $\mathcal { A } _ { 1 }$ and $\boldsymbol { \mathcal { A } } _ { 2 }$ in Theorem 1:

$$
\begin{array} { r l } & { \mathfrak { A } _ { 1 } [ f ] ( x ) = m _ { 1 } ^ { i } ( x ) \partial _ { i } f ( x ) - s _ { 0 } ( x ) f , } \\ & { \mathfrak { A } _ { 2 } [ f ] ( x ) = m _ { 2 } ^ { i j } ( x ) \partial _ { i } \partial _ { j } f ( x ) - \bigl ( \Lambda ^ { k } ( x ) + 2 s _ { 1 } ^ { k } ( x ) \bigr ) \partial _ { k } f ( x ) + \Upsilon ( x ) f ( x ) . } \end{array}
$$

Remark 22. This expansion is a generalization of the moment expansion of the kernel in the Riemannian setting, see e.g. Coifman & Lafon (2006, Lemma 8). Several differences are to be noted:

• Due to the non-reversibility of the metric, $m _ { 1 }$ does not vanish in general.

• The second-order term contains the term $\Lambda ^ { k } ( x )$ , which simplifies in the Riemannian setting, and the curvature term $\Upsilon ( x )$ , which only multiplies f and reduces to an average of the Ricci curvature in the Riemannian setting.

• The S-curvature brings the terms s and $s _ { 1 } ,$ , among which a zeroth-order term $- \varepsilon s _ { 0 } f$ atfirst order in ε, and contributes to Υ. These contributions vanish in the Riemannian setting, and more generally for Berwald metrics, whose S-curvature with respect to m is identically zero (Shen & Shen, 2016, Proposition 4.3).

We now turn to the technical lemmas that are used in the proof of Theorem 1.

## MAIN LEMMAS

We start by giving the expansion of a function f around a point x using the Finsler exponential map $\exp _ { x } ^ { \mathcal { F } }$ This expansion will be used to expand the integral operator $\mathcal { G } _ { \mathcal { F } }$ in terms of the derivatives of $f$ at x, and the moments of the kernel K. It is to be contrasted with the expansion using the Riemannian exponential map $\exp _ { x } ^ { \alpha }$ (Coifman & Lafon, 2006).

Lemma 23 (Function expansion using Finsler metric). Let $x \in \mathcal { M }$ and $v \in T _ { x } { \mathcal { M } }$ be a unit tangent vector at x. Let also $\varepsilon \in \mathbb { R } _ { * } ^ { + }$ . Then

$$
f ( \exp _ { x } ^ { \mathcal { F } } ( \varepsilon v ) ) = f ( x ) + \varepsilon \partial _ { i } f ( x ) v ^ { i } + \frac { \varepsilon ^ { 2 } } { 2 } \left( \partial _ { j } \partial _ { i } f ( x ) - \partial _ { k } f ( x ) \tilde { \Gamma } _ { i j } ^ { k } ( x , v ) \right) v ^ { j } v ^ { i } + O ( \varepsilon ^ { 3 } ) ,\tag{12}
$$

where $\tilde { \Gamma } _ { i j } ^ { k }$ is the Christoffel symbol ofthe Finsler metric $\mathcal { F } ^ { 3 }$

Proof. Let $\gamma ( \varepsilon ) = \exp _ { x } ^ { \mathcal { F } } ( \varepsilon v )$ , and $\phi ( \varepsilon ) = f ( \gamma ( \varepsilon ) )$ . We have that $\phi : \mathbb { R }  \mathbb { R }$ and $\phi ( 0 ) = f ( x )$ . Taylor expansion of $\phi$ around 0 yields:

$$
\phi ( \varepsilon ) = \phi ( 0 ) + \phi ^ { \prime } ( 0 ) \varepsilon + \frac { 1 } { 2 } \phi ^ { \prime \prime } ( 0 ) \varepsilon ^ { 2 } + O ( \varepsilon ^ { 3 } )
$$

Now we need to express $\phi ^ { \prime } ( 0 )$ and $\phi ^ { \prime \prime } ( 0 )$ in terms of the derivatives of $f$ at x.

For thefirst derivative, by the chain rule,

$$
\phi ^ { \prime } ( 0 ) = \frac { d } { d \varepsilon } f ( \gamma ( \varepsilon ) ) \bigg | _ { \varepsilon = 0 } = \partial _ { i } f ( \gamma ( \varepsilon ) ) \dot { \gamma } ^ { i } ( \varepsilon ) \bigg | _ { \varepsilon = 0 } = \partial _ { i } f ( x ) v ^ { i }
$$

For the second derivative, by the product rule,

$$
\begin{array} { l } { { \displaystyle { \phi ^ { \prime \prime } } ( \varepsilon ) = \frac { d } { d \varepsilon } ( d f ( \gamma ( \varepsilon ) ) \dot { \gamma } ( \varepsilon ) ) } } \\ { { \displaystyle ~ = \underbrace { \frac { d } { d \varepsilon } \Big ( d f ( \gamma ( \varepsilon ) ) \Big ) \dot { \gamma } ^ { i } ( \varepsilon ) } _ { \mathrm { ( A ) } } + \underbrace { d f ( \gamma ( \varepsilon ) ) \ddot { \gamma } ^ { i } ( \varepsilon ) } _ { \mathrm { ( B ) } } . } } \end{array}
$$

Using the chain rule again, the first term (A) gives

$$
\frac { d } { d \varepsilon } \Big ( d f ( \gamma ( \varepsilon ) ) \Big ) \dot { \gamma } ^ { i } ( \varepsilon ) \bigg | _ { \varepsilon = 0 } = \partial _ { j } \partial _ { i } f ( \gamma ( \varepsilon ) ) \dot { \gamma } ^ { j } ( \varepsilon ) \dot { \gamma } ^ { i } ( \varepsilon ) \bigg | _ { \varepsilon = 0 } = \partial _ { j } \partial _ { i } f ( x ) v ^ { j } v ^ { i } .
$$

For the second term (B), since $\gamma$ is a geodesic under the Finsler metric $\mathcal { F }$ , the geodesic equation (Ohta, 2021, Chap. 3.3) gives

$$
\ddot { \gamma } ^ { i } ( \varepsilon ) = - \tilde { G } ^ { i } ( \gamma ( \varepsilon ) , \dot { \gamma } ( \varepsilon ) ) = - \tilde { \Gamma } _ { j k } ^ { i } ( x , \dot { \gamma } ( \varepsilon ) ) \dot { \gamma } ( \varepsilon ) ^ { j } \dot { \gamma } ( \varepsilon ) ^ { k }
$$

where $\tilde { G } ^ { i }$ is the geodesic spray coefficient of the Finsler metric $\mathcal { F }$ , and $\tilde { \Gamma }$ is the Christoffel symbol of the Finsler metric F. Consequently,

$$
d f ( \gamma ( \varepsilon ) ) \ddot { \gamma } ^ { k } ( \varepsilon ) \bigg \vert _ { \varepsilon = 0 } = - \partial _ { k } f ( x ) \tilde { \Gamma } _ { i j } ^ { k } ( x , v ) v ^ { i } v ^ { j } ,
$$

where we changed the index from i to k. Hence, the second derivative of $\phi$ at 0 is

$$
\phi ^ { \prime \prime } ( 0 ) = ( \partial _ { j } \partial _ { i } f ( x ) - \partial _ { k } f ( x ) \tilde { \Gamma } _ { i j } ^ { k } ( x , v ) ) v ^ { j } v ^ { i } .
$$

The Taylor expansion of f around x is thus

$$
f ( \exp _ { x } ^ { \mathcal { F } } ( \varepsilon v ) ) = f ( x ) + \varepsilon \partial _ { i } f ( x ) v ^ { i } + \frac { \varepsilon ^ { 2 } } { 2 } \left( \partial _ { j } \partial _ { i } f ( x ) - \partial _ { k } f ( x ) \tilde { \Gamma } _ { i j } ^ { k } ( x , v ) \right) v ^ { j } v ^ { i } + O ( \varepsilon ^ { 3 } ) .
$$

This concludes the proof.

To use the previous expansion, we need to localize the integral operator $\mathcal { G } _ { \mathcal { F } }$ around a point $x \in \mathcal { M }$ , so that the integral is instead taken over the tangent space $T _ { x } { \mathcal { M } }$ , which can be identified with $\bar { \mathbb { R } } ^ { m }$ . This is possible thanks to the decay of the kernel K, which allows us to restrict the integral to a small neighborhood of $x .$

Lemma 24 (Integral localization). Let K satisfy (A1) and $x \in \mathcal { M }$ . Then, for any $f \in L _ { 1 } ( \mathcal { M } )$ smooth enough,

$$
\mathfrak { L } _ { \mathcal { F } } [ f ] ( x ) = \int _ { \mathcal { B } _ { x } ( r _ { 0 } / \varepsilon ) } K \left( \mathcal { F } ( x , v ) \right) f ( \exp _ { x } ^ { \mathcal { F } } ( \varepsilon v ) ) \mathcal { J } _ { x } ( \varepsilon v ) \mathrm { d } \lambda _ { x } ( v ) + O ( \varepsilon ^ { 3 } ) ,
$$

where $\mathcal { J } _ { x }$ is the Jacobian of the change of variable through the exponential map, and $0 < r _ { 0 } < i _ { \mathcal { M } }$ is a constant smaller than the injectivity radius of M, which does not depend on x.

Proof. We start by splitting the integral into two parts, one over the ball $B _ { \mathcal { F } } ^ { + } ( x , r _ { 0 } ) = \{ y \in \mathcal { M } : \mathrm { d s t } _ { \mathcal { F } } ( x , y ) <$ $r _ { 0 } \}$ and one over its complement.

On the far region. Since M is compact, f is bounded, and we denote $\begin{array} { r } { M _ { f } = \operatorname* { s u p } _ { y \in \mathcal { M } } | f ( y ) | < \infty } \end{array}$ . By definition, it holds that $\mathrm { d s t } _ { \mathcal { F } } ( x , y ) \geq r _ { 0 }$ for all $y \in \mathcal { M } \setminus B _ { \mathcal { F } } ^ { + } ( x , r _ { 0 } )$ . Using the sub-exponential property of the kernel (A1), it holds:

$$
\begin{array} { r l r } {  { \varepsilon ^ { - m } \int _ { \mathcal { M } \setminus B _ { \varepsilon } ^ { + } ( x , r _ { 0 } ) } K ( \frac { \mathrm { d } \mathrm { s t } _ { \mathcal { F } } ( x , y ) } { \varepsilon } ) f ( y ) \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } ( y ) \le \varepsilon ^ { - m } M _ { f } \int _ { \mathcal { M } \setminus B _ { \varepsilon } ^ { + } ( x , r _ { 0 } ) } C _ { K } \exp ( - \frac { \nu _ { K } r _ { 0 } } { \varepsilon } ) \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } ( y ) } } \\ & { } & { \le M _ { f } C _ { K } \varepsilon ^ { - m } \exp ( - \frac { \nu _ { K } r _ { 0 } } { \varepsilon } ) \mathfrak { m } _ { \mathrm { B H } } ( \mathcal { M } ) , } \end{array}
$$

which is $O ( \varepsilon ^ { N } )$ for any $N > 0 ;$ , and in particular $O ( \varepsilon ^ { 3 } )$

On the near region. Because $r _ { 0 } < i \mathcal { M }$ , the exponential map $\exp _ { x } ^ { \mathcal { F } }$ is a diffeomorphism from the tangent ball $\mathcal { B } _ { x } ( r _ { 0 } ) \subset T _ { x } \mathcal { M }$ onto the forward metric ball $B _ { \mathcal { F } } ^ { + } ( x , r _ { 0 } ) \subset \mathcal { M }$ (it is $C ^ { 1 }$ at the origin and smooth elsewhere), so that we can do the change of variable $y = \exp _ { x } ^ { \mathcal { F } } ( \varepsilon v )$ , with $v \in \mathcal { B } _ { x } ( r _ { 0 } / \varepsilon )$ . For such $v , \varepsilon \mathcal { F } ( x , v ) < r _ { 0 } < i _ { \mathcal { M } }$ so that the Finslerian distance writes dst<sub>F</sub> $\left( x , \exp _ { x } ^ { \mathcal { F } } ( \varepsilon v ) \right) = \varepsilon \mathcal { F } ( x , v )$ . The integral over the near region can then be rewritten as:

$$
\varepsilon ^ { - m } \int _ { B _ { \varepsilon } ^ { + } ( x , r _ { 0 } ) } K \left( \frac { \mathrm { d } \mathrm { s t } _ { \mathcal { F } } ( x , y ) } { \varepsilon } \right) f ( y ) \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } ( y ) = \int _ { \mathcal { B } _ { x } ( r _ { 0 } / \varepsilon ) } K \left( \mathcal { F } ( x , v ) \right) f ( \exp _ { x } ^ { \mathcal { F } } ( \varepsilon v ) ) \mathcal { J } _ { x } ( \varepsilon v ) \mathrm { d } \lambda _ { x } ( v ) ,
$$

where $\mathcal { J } _ { x } ( u )$ is the Jacobian of the change of variable $y = \exp _ { x } ^ { \mathcal { F } } ( u )$ , i.e. $\mathrm { d m } _ { \mathrm { B H } } ( y ) = \mathcal { J } _ { x } ( u ) \mathrm { d } \lambda _ { x } ( u )$ . The factor $\varepsilon ^ { - m }$ is absorbed in the change of variable $u = \varepsilon v$ . Finally, we can combine the two regions to get the desired result. □

Lemma 25 (Jacobian expansion). Let $x \in \mathcal { M } , \ v \in T _ { x } \mathcal { M } \setminus \{ 0 \}$ , and $\varepsilon > 0$ with $\varepsilon \mathcal { F } ( x , v ) < i _ { \mathcal { M } }$ . Then the Jacobian $\mathcal { J } _ { x }$ ofLemma 24 satisfies

$$
\mathcal { J } _ { x } ( \varepsilon v ) = \sigma _ { \mathrm { B H } } ( x ) \left( 1 - \varepsilon S ( x , v ) + \frac { \varepsilon ^ { 2 } } { 2 } \left( S ( x , v ) ^ { 2 } - \dot { S } ( x , v ) - \frac { 1 } { 3 } R i c ^ { \mathcal { F } } ( v ) \right) + { \mathbb O } \left( \varepsilon ^ { 3 } \right) \right) ,
$$

where $S$ and $\dot { S }$ are the $S \cdot$ -curvature (Definition 18) and its derivative along the geodesicflow, and where the constant in O depends neither on x nor on v.

Proof. By definition of the Jacobian, $\mathcal { J } _ { x }$ has the form:

$$
\begin{array} { r } { \mathcal { J } _ { x } ( \varepsilon v ) = \underbrace { \sigma _ { \mathrm { B H } } ( \exp _ { x } ^ { \mathcal { F } } ( \varepsilon v ) ) } _ { \mathrm { c h a n g e ~ o f ~ m e a s u r e ( \star ) } } \times \underbrace { \operatorname* { d e t } ( d ( \exp _ { x } ^ { \mathcal { F } } ) _ { \varepsilon v } ) } _ { \mathrm { c h a n g e ~ o f ~ v a r i a b l e ( \star \star ) } } , } \end{array}
$$

where $\sigma _ { \mathrm { B H } } ( \exp _ { x } ^ { \mathcal { F } } ( \varepsilon v ) )$ comes from the change of measure from the Busemann–Hausdorff measure to the Lebesgue measure on $T _ { x } { \mathcal { M } }$ , and de $\bar { \varrho } ( d ( \exp _ { x } ^ { \mathfrak { F } } ) _ { \varepsilon v } )$ comes from the change of variable $y = \exp _ { x } ^ { \mathcal { F } } ( \varepsilon v )$ . This proof is split in two parts, the expansion of $\sigma _ { \mathrm { B H } } ( \exp _ { x } ^ { \mathcal { F } } ( \varepsilon v ) )$ ) and the expansion of det $( d ( \exp _ { x } ^ { \mathcal { F } } ) _ { \varepsilon v } )$

Expansion $o f \left( \star \right)$ . By definition of the distortion (Definition 18), we have, for every $u \in T _ { z } \mathcal { M } \setminus \{ 0 \}$ , that $\sigma _ { \mathrm { B H } } ( z ) = \exp ( - \tau ( z , u ) ) \sqrt { \operatorname* { d e t } g ( z , u ) }$ . The left-hand side does not depend on $u ,$ so that we may choose the velocity of the geodesic $\gamma ,$ with ${ \dot { \gamma } } ( 0 ) { \dot { = } } x \operatorname { a n d } { \dot { \gamma } } ( 0 ) = v$ , at both of its ends:

$$
\sigma _ { \mathtt { B H } } ( x ) = \exp ( - \tau ( x , v ) ) \sqrt { \operatorname* { d e t } g ( x , v ) } , \qquad \sigma _ { \mathtt { B H } } ( \exp _ { x } ^ { \mathcal { F } } ( \varepsilon v ) ) = \exp ( - \tau ( \gamma ( \varepsilon ) , \dot { \gamma } ( \varepsilon ) ) ) \sqrt { \operatorname* { d e t } g ( \gamma ( \varepsilon ) , \dot { \gamma } ( \varepsilon ) ) } .
$$

Taking the ratio of these two identities, we have that

$$
\sigma _ { \mathrm { B H } } ( \exp _ { x } ^ { \mathcal { F } } ( \varepsilon v ) ) = \sigma _ { \mathrm { B H } } ( x ) \frac { \sqrt { \operatorname* { d e t } g ( \gamma ( \varepsilon ) , \dot { \gamma } ( \varepsilon ) ) } } { \sqrt { \operatorname* { d e t } g ( x , v ) } } \exp \big ( \tau ( x , v ) - \tau ( \gamma ( \varepsilon ) , \dot { \gamma } ( \varepsilon ) ) \big ) .
$$

By definition of the S-curvature (Definition 18), and since $t \mapsto \gamma ( s + t )$ is the geodesic with initial conditions $( \gamma ( s ) , \dot { \gamma } ( s ) )$ , the first two derivatives of $t \mapsto \tau ( \gamma ( t ) , \dot { \gamma } ( t ) )$ are $S ( \gamma ( t ) , \dot { \gamma } ( t ) )$ and $\dot { S } ( \gamma ( t ) , \dot { \gamma } ( t ) )$ ), so that

$$
\tau ( \gamma ( \varepsilon ) , \dot { \gamma } ( \varepsilon ) ) - \tau ( x , v ) = \varepsilon S ( x , v ) + \frac { \varepsilon ^ { 2 } } { 2 } \dot { S } ( x , v ) + O ( \varepsilon ^ { 3 } ) .
$$

Hence, we have that

$$
\sigma _ { \mathrm { B H } } ( \exp _ { x } ^ { \mathcal { F } } ( \varepsilon v ) ) = \sigma _ { \mathrm { B H } } ( x ) \frac { \sqrt { \operatorname* { d e t } g ( \gamma ( \varepsilon ) , \dot { \gamma } ( \varepsilon ) ) } } { \sqrt { \operatorname* { d e t } g ( x , v ) } } \left( 1 - \varepsilon S ( x , v ) + \frac { \varepsilon ^ { 2 } } { 2 } \left( S ( x , v ) ^ { 2 } - \dot { S } ( x , v ) \right) + O ( \varepsilon ^ { 3 } ) \right) .
$$

Expansion of $( \star \star )$ . Let $( e _ { 1 } , \ldots , e _ { m } )$ be a basis of $T _ { x } { \mathcal { M } }$ which is orthonormal for $g ( x , v )$ , and such that $e _ { m } = v / \mathcal { F } ( x , v )$ . For each $i ,$ we consider the variation of geodesics $\gamma _ { i } ( t , s ) = \exp _ { x } ^ { \mathcal { F } } \big ( t ( v + s e _ { i } ) \big )$ , and we define

$$
J _ { i } ( t ) = \partial _ { s } \gamma _ { i } ( t , 0 ) = d ( \exp _ { x } ^ { \mathcal { F } } ) _ { t v } ( t e _ { i } ) ,
$$

which is a Jacobi field along the geodesic $\gamma ( t ) = \exp _ { x } ^ { \mathcal { F } } ( t v )$ (Ohta, 2021, Prop. 5.5). For the sake of completeness, we check here its initial conditions $J _ { i } ( 0 )$ and ${ \dot { J } } _ { i } ( 0 )$ . First,

$$
J _ { i } ( 0 ) = \partial _ { s } \gamma _ { i } ( 0 , 0 ) = \partial _ { s } x = 0 .
$$

Second, $\partial _ { t } \gamma _ { i } ( 0 , s ) = v + s e _ { i }$ , and by the symmetry of second derivatives<sup>4</sup>, we have that $\partial _ { t } \partial _ { s } \gamma _ { i } ( 0 , 0 ) =$ $\partial _ { s } \partial _ { t } \gamma _ { i } ( 0 , 0 )$ . Thus, we get ${ \dot { J } } _ { i } ( 0 ) = e _ { i }$ . As a Jacobi field, $J _ { i }$ satisfies the Jacobi equation (Ohta, 2021, Def. 5.1):

$$
\ddot { J } _ { i } ( t ) + R ( t ) J _ { i } ( t ) = 0
$$

where R is the curvature tensor of the Finsler metric $\mathcal { F }$ , evaluated along the geodesic $\gamma ( t )$ . By definition, for $t = 0$ , we have that $R ( 0 ) J _ { i } ( 0 ) = 0$ . Hence, the Jacobi equation at $t = 0$ gives $\ddot { J } _ { i } ( 0 ) = \stackrel { \cdot } { 0 }$ . Differentiating the Jacobi equation gives

$$
\begin{array} { r } { \dddot { \boldsymbol J } _ { i } ( t ) + \dot { \boldsymbol R } ( t ) \boldsymbol J _ { i } ( t ) + \boldsymbol R ( t ) \dot { \boldsymbol J } _ { i } ( t ) = 0 , } \end{array}
$$

which, evaluated at $t = 0 ,$ , gives $\because { \ddot { J } } _ { i } ( 0 ) = - R ( 0 ) { \dot { J } } _ { i } ( 0 ) = - R ( 0 ) e _ { i }$ . To write these relations in matrix form, we express the $J _ { i } ( t )$ in the basis $( E _ { 1 } ( { t } ) , \ldots , E _ { m } ( { t } ) )$ of $T _ { \gamma ( t ) } \mathcal { M }$ obtained by parallel transport of $( e _ { 1 } , \ldots , e _ { m } )$ along $\gamma _ { \mathrm { : } }$ , for the Chern connection with reference vector γ˙ . This basis remains orthonormal for $g ( \gamma ( t ) , \dot { \gamma } ( t ) )$ , and the covariant derivatives along γ are the ordinary derivatives of the components (Ohta, 2021). Letting $\mathbf { J } ( t )$ be the matrix of the components of $J _ { 1 } ( t ) , \ldots , J _ { m } ( t )$ in this basis, and $\mathbf { R } ( t )$ the matrix of $R ( t )$ , we have that ${ \bf J } ( 0 ) = \ddot { \bf J } ( 0 ) = 0 , \dot { \bf J } ( 0 ) = I _ { m }$ and $\overleftrightarrow { \mathbf { J } } \left( 0 \right) = - \mathbf { R } ( 0 )$ . Hence, the Taylor expansion of J around 0 is given by

$$
{ \bf J } ( t ) = t I _ { m } - \frac { t ^ { 3 } } { 6 } { \bf R } ( 0 ) + O ( t ^ { 4 } ) .
$$

Since $J _ { i } ( \varepsilon ) / \varepsilon = d ( \exp _ { x } ^ { \mathcal { F } } ) _ { \varepsilon v } e _ { i } .$ , the matrix $\mathbf { J } ( \varepsilon ) / \varepsilon$ represents $d ( \exp _ { x } ^ { \mathcal { F } } ) _ { \varepsilon v }$ in the orthonormal bases $( e _ { i } )$ of $T _ { x } { \mathcal { M } }$ and $\left( E _ { i } ( \varepsilon ) \right)$ of $T _ { \gamma ( \varepsilon ) } \mathscr { M }$ . Its determinant is thus

$$
\begin{array} { l } { \displaystyle \operatorname* { d e t } \left( \frac { 1 } { \varepsilon } \mathbf { J } ( \varepsilon ) \right) = \operatorname* { d e t } \left( I _ { m } - \frac { \varepsilon ^ { 2 } } { 6 } \mathbf { R } ( 0 ) + O ( \varepsilon ^ { 3 } ) \right) } \\ { \displaystyle \qquad = 1 - \frac { \varepsilon ^ { 2 } } { 6 } \operatorname { t r } ( \mathbf { R } ( 0 ) ) + O ( \varepsilon ^ { 3 } ) } \\ { \displaystyle \qquad = 1 - \frac { \varepsilon ^ { 2 } } { 6 } \mathrm { R i c } ^ { \mathcal { F } } ( v ) + O ( \varepsilon ^ { 3 } ) , } \end{array}
$$

where $\operatorname { R i c } ^ { \mathcal { F } } ( v )$ is the Ricci curvature of $\mathcal { F }$ in the direction $v ,$ i.e. the trace of $R _ { v }$ in a basis which is orthonormal for $g ( x , v )$ . Since $e _ { m } = v / \mathcal { F } ( x , v )$ and $R _ { v } v = 0$ , only the $m - 1$ transverse directions contribute to this trace.

Going back to the original basis, the determinant in $( \star \star )$ is taken in the coordinate bases of $T _ { x } { \mathcal { M } }$ and $T _ { \gamma ( \varepsilon ) } { \mathcal { M } } .$ , and not in the orthonormal ones. Let P and $\mathbf { Q }$ be the matrices whose columns are the coordinates of $e _ { 1 } , \ldots , e _ { m }$ and of $E _ { 1 } ( \varepsilon ) , \ldots , E _ { m } ( \varepsilon )$ . The matrix of $d ( \exp _ { x } ^ { \mathcal { F } } ) _ { \varepsilon v }$ in the coordinate bases is ${ \bf Q } ( { \bf J } ( \varepsilon ) / \varepsilon ) { \bf P } ^ { - 1 }$ so that

$$
\operatorname* { d e t } \left( d ( \exp _ { x } ^ { \mathcal { F } } ) _ { \varepsilon v } \right) = \frac { \operatorname* { d e t } \mathbf { Q } } { \operatorname* { d e t } \mathbf { P } } \operatorname* { d e t } \left( \frac { 1 } { \varepsilon } \mathbf { J } ( \varepsilon ) \right) .
$$

The orthonormality of the two bases reads $\mathbf { P } ^ { \mathsf { T } } g ( x , v ) \mathbf { P } = I _ { m }$ and $\mathbf { Q } ^ { \top } g ( \gamma ( \varepsilon ) , \dot { \gamma } ( \varepsilon ) ) \mathbf { Q } = I _ { m }$ , so that det $\mathbf { P } =$ det $g ( x , v ) ^ { - 1 / 2 }$ and det $\mathbf { Q } = \operatorname* { d e t } g ( \gamma ( \varepsilon ) , \dot { \gamma } ( \varepsilon ) ) ^ { - 1 / 2 }$ . Both are positive when $( e _ { i } )$ is positively oriented, since the determinant of the coordinates of $\left( E _ { i } ( t ) \right)$ is then continuous in $t ,$ never vanishes, and equals detP at $t = 0$ . Hence

$$
\operatorname* { d e t } \left( d ( \exp _ { x } ^ { \mathcal { F } } ) _ { \varepsilon v } \right) = \frac { \sqrt { \operatorname* { d e t } g ( x , v ) } } { \sqrt { \operatorname* { d e t } g ( \gamma ( \varepsilon ) , \dot { \gamma } ( \varepsilon ) ) } } \left( 1 - \frac { \varepsilon ^ { 2 } } { 6 } \mathrm { R i c } ^ { \mathcal { F } } ( v ) + O ( \varepsilon ^ { 3 } ) \right) .
$$

Conclusion. Multiplying the expansions of (⋆) and (⋆⋆), the two ratios of $\scriptstyle { \sqrt { \operatorname* { d e t } g } }$ cancel exactly, and

$$
\mathcal { J } _ { x } ( \varepsilon v ) = \sigma _ { \mathrm { B H } } ( x ) \left( 1 - \varepsilon S ( x , v ) + \frac { \varepsilon ^ { 2 } } { 2 } \left( S ( x , v ) ^ { 2 } - \dot { S } ( x , v ) \right) + O ( \varepsilon ^ { 3 } ) \right) \left( 1 - \frac { \varepsilon ^ { 2 } } { 6 } \mathrm { R i } \mathrm { c } ^ { \mathcal { F } } ( v ) + O ( \varepsilon ^ { 3 } ) \right) .
$$

Expanding the product, the cross term $\varepsilon S ( x , v ) \cdot \textstyle { \frac { \varepsilon ^ { 2 } } { 6 } } \operatorname { R i c } ^ { \mathcal { F } } ( v )$ is of order $\varepsilon ^ { 3 }$ , so that

$$
\mathcal { J } _ { x } ( \varepsilon v ) = \sigma _ { \mathrm { B H } } ( x ) \left( 1 - \varepsilon S ( x , v ) + \frac { \varepsilon ^ { 2 } } { 2 } \left( S ( x , v ) ^ { 2 } - \dot { S } ( x , v ) - \frac { 1 } { 3 } \mathrm { R i } \mathrm { c } ^ { \mathcal { F } } ( v ) \right) + \mathcal { O } \left( \varepsilon ^ { 3 } \right) \right) .
$$

This concludes the proof.

## PROOF OF THE MOMENT EXPANSION OF THE KERNEL

We can now prove the moment expansion of the kernel of Theorem 1. The proof is a direct application of the previous lemmas. We recall that it states that, for any $f : \mathcal { M }  \mathbb { R }$ smooth enough,

$$
\mathcal { G } _ { \mathcal { F } } [ f ] ( x ) = m _ { 0 } f + \varepsilon \mathcal { A } _ { 1 } [ f ] ( x ) + \frac { \varepsilon ^ { 2 } } { 2 } \mathcal { A } _ { 2 } [ f ] ( x ) + \mathcal { O } ( \varepsilon ^ { 3 } ) .
$$

Proof. The proof utilizes Lemma 24 and both expansions of $f ( \exp _ { x } ^ { \mathcal { F } } ( \varepsilon v ) )$ and $\mathcal { J } _ { x } ( \varepsilon v )$ in $\varepsilon \ ( \mathrm { I }$ Lemmas 23 and 25). Putting those together, it holds that:

$$
\begin{array} { l } { \displaystyle \mathcal { E } _ { \mathcal { F } } [ f ] ( x ) = \int _ { \mathfrak { R } _ { x } ( r _ { 0 } / \varepsilon ) } K ( \mathcal { F } ( x , v ) ) f ( \exp _ { x } ^ { \mathcal { F } } ( \varepsilon v ) ) \mathcal { J } _ { x } ( \varepsilon v ) \mathrm { d } \lambda _ { x } ( v ) + O ( \varepsilon ^ { 3 } ) } \\ { \displaystyle \qquad = o _ { \mathtt { B H } } ( x ) \int _ { \mathcal { B } _ { x } ( r _ { 0 } / \varepsilon ) } K ( \mathcal { F } ( x , v ) ) \bigg [ f ( x ) + \varepsilon \partial _ { i } f ( x ) v ^ { i } } \\ { \displaystyle \qquad + \frac { \varepsilon ^ { 2 } } { 2 } \left( \partial _ { j } \partial _ { i } f ( x ) - \partial _ { k } f ( x ) \tilde { \Gamma } _ { i j } ^ { k } ( x , v ) \right) v ^ { j } v ^ { i } + O ( \varepsilon ^ { 3 } ) \bigg ] } \\ { \displaystyle \qquad \times \left[ 1 - \varepsilon S ( x , v ) + \frac { \varepsilon ^ { 2 } } { 2 } \left( S ( x , v ) ^ { 2 } - \dot { S } ( x , v ) - \frac { 1 } { 3 } \mathrm { R i } \mathrm { c } ^ { \mathcal { F } } ( x , v ) \right) + O ( \varepsilon ^ { 3 } ) \right] \mathrm { d } \lambda _ { x } ( v ) + O ( \varepsilon ^ { 3 } ) . } \end{array}
$$

Expanding the product and keeping the terms up to order $\varepsilon ^ { 2 }$ , the integrand is $K ( \mathcal { F } ( x , v ) )$ times

$$
\begin{array} { l } { f ( x ) + \varepsilon \left( \partial _ { i } f ( x ) v ^ { i } - S ( x , v ) f ( x ) \right) } \\ { \displaystyle + \frac { \varepsilon ^ { 2 } } { 2 } \Bigg [ \partial _ { i } \partial _ { j } f ( x ) v ^ { i } v ^ { j } - \partial _ { k } f ( x ) \tilde { \Gamma } _ { i j } ^ { k } ( x , v ) v ^ { i } v ^ { j } - 2 S ( x , v ) \partial _ { k } f ( x ) v ^ { k } } \\ { \displaystyle + \left( S ( x , v ) ^ { 2 } - \dot { S } ( x , v ) - \frac { 1 } { 3 } \mathrm { R i c } ^ { \mathcal { F } } ( x , v ) \right) f ( x ) \Bigg ] , } \end{array}
$$

up to terms of order $\varepsilon ^ { 3 }$ and higher. The moments of Definition 21 are integrals over the whole tangent space $\bar { T _ { x } } \mathcal { M }$ , whereas this integral is over $\mathcal { B } _ { x } ( r _ { 0 } / \varepsilon )$ . As in Lemma 24, the sub-exponential decay of $K ( \bar { \bf A } 1 )$ makes

the integral over $T _ { x } \mathcal { M } \setminus \mathcal { B } _ { x } ( r _ { 0 } / \varepsilon )$ exponentially small, so that the domain can be extended to $T _ { x } { \mathcal { M } }$ up to $O ( \varepsilon ^ { 3 } )$ . By collecting the terms of order $0 , \varepsilon$ and $\varepsilon ^ { 2 }$ , and including back the density $\sigma _ { \mathrm { B H } } ( x )$ , we recover the definitions of the moments (Definition 21) and thus

$$
\begin{array} { l } { \displaystyle \mathcal { G } _ { \mathcal { F } } [ f ] ( x ) = m _ { 0 } ( x ) f ( x ) + \varepsilon \left( m _ { 1 } ^ { i } ( x ) \partial _ { i } f ( x ) - s _ { 0 } ( x ) f ( x ) \right) } \\ { \displaystyle \qquad + \frac { \varepsilon ^ { 2 } } { 2 } \left( m _ { 2 } ^ { i j } ( x ) \partial _ { i } \partial _ { j } f ( x ) - \left( \Lambda ^ { k } ( x ) + 2 s _ { 1 } ^ { k } ( x ) \right) \partial _ { k } f ( x ) + \Upsilon ( x ) f ( x ) \right) + \Theta ( \varepsilon ^ { 3 } ) . } \end{array}
$$

Hence to recover the operators $\displaystyle \mathcal { A } _ { 1 }$ and $\boldsymbol { \mathcal { A } } _ { 2 }$ of Theorem 1, we define:

$$
\begin{array} { r l } & { \mathfrak { A } _ { 1 } [ f ] ( x ) = m _ { 1 } ^ { i } ( x ) \partial _ { i } f ( x ) - \mathfrak { s } _ { 0 } ( x ) f ( x ) , } \\ & { \mathfrak { A } _ { 2 } [ f ] ( x ) = m _ { 2 } ^ { i j } ( x ) \partial _ { i } \partial _ { j } f ( x ) - \left( \Lambda ^ { k } ( x ) + 2 \mathfrak { s } _ { 1 } ^ { k } ( x ) \right) \partial _ { k } f ( x ) + \Upsilon ( x ) f ( x ) . } \end{array}
$$

## D.3 PROOF OF THEOREM 2

This subsection is devoted to the proof of Theorem 2, which builds upon the previous expansion of the kernel. To do so, we first prove some properties of the reverse Finsler metric and its nonlinear connection.

## LEMMAS ON THE REVERSE FINSLER METRIC AND NONLINEAR CONNECTION

Some of the following properties are stated in Ohta (2021, Section 2.5), but are not proved there. We provide here proofs for completeness.

Proposition 26 (Geodesic spray of the reverse Finsler metric). Let $\overleftarrow { \mathcal { F } } ( x , v ) = \mathcal { F } ( x , - v )$ , and let $G ^ { k }$ and $\overleftarrow { G } ^ { k }$ be the geodesic spray coefficients of F and , respectively. Then, we have that $\overleftarrow { G } ^ { k } ( x , v ) = G ^ { k } ( x , - v )$ for all $v \in T _ { x } \mathcal { M }$

Proof. We first recall the relevant notations. For $x \in \mathcal { M }$ and $v \in T _ { x } { \mathcal { M } }$ , let $g _ { i j } ( x , v )$ and $\overleftarrow { g } _ { \ i j } ( x , v )$ be the fundamental tensors of Fand , respectively. By Definition 12, we have that:

$$
g _ { i j } ( x , v ) = \frac { 1 } { 2 } \frac { \partial ^ { 2 } \mathcal { F } ^ { 2 } ( x , v ) } { \partial v ^ { i } \partial v ^ { j } } \quad \mathrm { a n d } \quad \overleftarrow { g } _ { i j } ( x , v ) = \frac { 1 } { 2 } \frac { \partial ^ { 2 } \overleftarrow { \mathcal { F } } ^ { 2 } ( x , v ) } { \partial v ^ { i } \partial v ^ { j } } .
$$

And the geodesic spray coefficients (Ohta, 2021) are given by:

$$
\begin{array} { l } { { G ^ { k } ( x , v ) = \displaystyle \frac 1 4 g ^ { k l } ( x , v ) \left( \frac { \partial ^ { 2 } \mathcal { F } ^ { 2 } ( x , v ) } { \partial { x ^ { i } } \partial { v ^ { l } } } v ^ { i } - \frac { \partial \mathcal { F } ^ { 2 } ( x , v ) } { \partial { x ^ { l } } } \right) \mathrm { , } } } \\ { { \overleftarrow { G } ^ { k } ( x , v ) = \displaystyle \frac 1 4 \overleftarrow { g } ^ { k l } ( x , v ) \left( \frac { \partial ^ { 2 } \overleftarrow { \mathcal { F } } ^ { 2 } ( x , v ) } { \partial { x ^ { i } } \partial { v ^ { l } } } v ^ { i } - \frac { \partial \overleftarrow { \mathcal { F } } ^ { 2 } ( x , v ) } { \partial { x ^ { l } } } \right) \mathrm { . } } } \end{array}
$$

To start, we show that the fundamental tensor of the reverse Finsler metric is related to the fundamental tensor of the original Finsler metric by $\overleftarrow { g } _ { i j } ( x , v ) = g _ { i j } ( x , - v )$ . Indeed:

$$
\overleftarrow { g } _ { \mathit { i j } } ( x , v ) = { \frac { 1 } { 2 } } { \frac { \partial ^ { 2 } \overleftarrow { \mathcal { F } } ^ { 2 } ( x , v ) } { \partial v ^ { i } \partial v ^ { j } } } = { \frac { 1 } { 2 } } { \frac { \partial ^ { 2 } \mathcal { F } ^ { 2 } ( x , - v ) } { \partial v ^ { i } \partial v ^ { j } } } = g _ { i j } ( x , - v ) .
$$

This implies that $\overleftarrow { g } ^ { k l } ( x , v ) = g ^ { k l } ( x , - v )$ . Similarly,

$$
\frac { \partial ^ { 2 } \overleftarrow { \mathcal { F } } ^ { 2 } ( x , v ) } { \partial x ^ { i } \partial v ^ { l } } = - \frac { \partial ^ { 2 } \mathcal { F } ^ { 2 } ( x , - v ) } { \partial x ^ { i } \partial v ^ { l } } \quad \mathrm { a n d } \quad \frac { \partial \overleftarrow { \mathcal { F } } ^ { 2 } ( x , v ) } { \partial x ^ { l } } = \frac { \partial \mathcal { F } ^ { 2 } ( x , - v ) } { \partial x ^ { l } } .
$$

Putting everything together, we have that:

$$
\begin{array} { l } { { \overleftarrow { G } ^ { k } ( x , v ) = \displaystyle \frac { 1 } { 4 } g ^ { k l } ( x , - v ) \left( - \frac { \partial ^ { 2 } \mathcal { F } ^ { 2 } ( x , - v ) } { \partial x ^ { i } \partial v ^ { l } } v ^ { i } - \frac { \partial \mathcal { F } ^ { 2 } ( x , - v ) } { \partial x ^ { l } } \right) } } \\ { { \displaystyle \qquad = \frac { 1 } { 4 } g ^ { k l } ( x , - v ) \left( \frac { \partial ^ { 2 } \mathcal { F } ^ { 2 } ( x , - v ) } { \partial x ^ { i } \partial v ^ { l } } ( - v ) ^ { i } - \frac { \partial \mathcal { F } ^ { 2 } ( x , - v ) } { \partial x ^ { l } } \right) } } \\ { { \displaystyle \qquad = G ^ { k } ( x , - v ) . } } \end{array}
$$

This concludes the proof.

Next, we define the nonlinear connection of a Finsler metric, which is a key quantity in Finsler geometry. It is used to define the curvature tensor and the Ricci curvature of a Finsler metric, which appeared in Theorem 1. Definition 27 (Nonlinear connection). The nonlinear connection ofa Finsler metric $\mathcal { F }$ is defined as:

$$
N _ { j } ^ { i } ( x , v ) = { \frac { \partial G ^ { i } } { \partial v ^ { j } } } ( x , v ) .
$$

Lemma 28 (Nonlinear connection under reversal). Let $\left\{ \overline { { N } } _ { j } ^ { i } \right.$ and $N _ { j } ^ { i }$ be the nonlinear connections $o f ^ { \overleftarrow { \mathcal { F } } }$ and $\mathcal { F }$ respectively. Then $\overleftarrow { N } _ { j } ^ { i } ( x , v ) = - N _ { j } ^ { i } ( x , - v )$

Proof. Using Proposition 26 $( \overleftarrow { G } ^ { i } ( x , v ) = G ^ { i } ( x , - v ) )$ and the chain rule,

$$
\overleftarrow { N } _ { j } ^ { i } ( x , v ) = \frac { \partial \overleftarrow { G } ^ { i } } { \partial v ^ { j } } ( x , v ) = \frac { \partial } { \partial v ^ { j } } \left[ G ^ { i } ( x , - v ) \right] = - \frac { \partial G ^ { i } } { \partial v ^ { j } } ( x , - v ) = - N _ { j } ^ { i } ( x , - v ) .
$$

This allows us to prove the following property of the Ricci curvature of the reverse Finsler metric.

Proposition 29 (Ricci curvature of the reverse Finsler metric). Let $R i c ^ { \overleftarrow { \mathcal { F } } }$ and $R i c ^ { \mathcal { F } }$ be the Ricci curvatures $o f ^ { \overleftarrow { \mathcal { F } } }$ and Frespectively. Then $R i c ^ { \overleftarrow { \mathcal { F } } } ( x , v ) = R i c ^ { \mathcal { F } } ( x , - v )$

Proof. We first introduce a few auxiliary quantities. The curvature tensor of a Finsler metric $\mathcal { F }$ is defined as

$$
R _ { j } ^ { i } ( x , v ) = \frac { \partial G ^ { i } } { \partial x ^ { j } } ( x , v ) - \sum _ { k = 1 } ^ { m } \left( \frac { \partial N _ { j } ^ { i } } { \partial x ^ { k } } ( x , v ) v ^ { k } - \frac { \partial N _ { j } ^ { i } } { \partial v ^ { k } } ( x , v ) G ^ { k } ( x , v ) \right) - \sum _ { k = 1 } ^ { m } N _ { k } ^ { i } ( x , v ) N _ { j } ^ { k } ( x , v ) ,
$$

inducing a linear endomorphism $R _ { v } ( x , \cdot ) : T _ { x } { \mathcal { M } } \to T _ { x } { \mathcal { M } }$ defined as $\begin{array} { r } { R _ { v } ( x , w ) = \sum _ { i , j = 1 } ^ { m } R _ { j } ^ { i } ( x , v ) w ^ { j } \frac { \partial } { \partial x ^ { i } } \Bigr | _ { x } . } \end{array}$ The Ricci curvature is the trace of this endomorphism, for $v \neq 0 ;$

$$
\mathrm { R i c } ^ { \mathcal { F } } ( x , v ) = \sum _ { i = 1 } ^ { m - 1 } g ( x , v ) _ { k l } R _ { v } ( x , e _ { i } ) ^ { k } e _ { i } ^ { l } ,
$$

where $\mathcal { B } = \{ v / \mathcal { F } ( x , v ) \} \cup \{ e _ { i } \} _ { i = 1 } ^ { m - 1 } \subset T _ { x } \mathcal { M }$ is an orthonormal basis with respect to the fundamental tensor $g ( x , v )$ of F. We similarly define $\overleftarrow { R } _ { j } ^ { i } ( x , v ) , \overleftarrow { R } _ { v } ( x , w )$ and $\operatorname { R i c } ^ { \overleftarrow { \mathcal { F } } } \left( x , v \right)$ for the reverse Finsler metric

By Lemma 28, $\overleftarrow { N } _ { j } ^ { i } ( x , v ) = - N _ { j } ^ { i } ( x , - v )$ . Differentiating this identity in $x ^ { k }$ (which passes through unchanged, as in the proof of Proposition 26) and in $v ^ { k }$ (which picks up an extra sign, as in the proof of Lemma 28 itself) gives

$$
\frac { \partial \overleftarrow { N } _ { j } ^ { i } } { \partial x ^ { k } } ( x , v ) = - \frac { \partial N _ { j } ^ { i } } { \partial x ^ { k } } ( x , - v ) , \qquad \frac { \partial \overleftarrow { N } _ { j } ^ { i } } { \partial v ^ { k } } ( x , v ) = \frac { \partial N _ { j } ^ { i } } { \partial v ^ { k } } ( x , - v ) .
$$

Together with $\overleftarrow { G } ^ { k } ( x , v ) = G ^ { k } ( x , - v )$ (Proposition 26), we get

$$
\begin{array} { r l r } & { } & { \displaystyle { \frac { \partial \overline { { N } } _ { j } ^ { i } } { \partial x ^ { k } } } ( x , v ) v ^ { k } - \displaystyle { \frac { \partial \overline { { N } } _ { j } ^ { i } } { \partial v ^ { k } } } ( x , v ) \overleftarrow { G } ^ { k } ( x , v ) = - \displaystyle { \frac { \partial N _ { j } ^ { i } } { \partial x ^ { k } } } ( x , - v ) v ^ { k } - \displaystyle { \frac { \partial N _ { j } ^ { i } } { \partial v ^ { k } } } ( x , - v ) G ^ { k } ( x , - v ) } \\ & { } & { \displaystyle { = \frac { \partial N _ { j } ^ { i } } { \partial x ^ { k } } } ( x , - v ) ( - v ) ^ { k } - \displaystyle { \frac { \partial N _ { j } ^ { i } } { \partial v ^ { k } } } ( x , - v ) G ^ { k } ( x , - v ) , } \end{array}
$$

and $\begin{array} { r } { \sum _ { k } \overleftarrow { N } _ { k } ^ { i } ( x , v ) \overleftarrow { N } _ { j } ^ { k } ( x , v ) = \sum _ { k } N _ { k } ^ { i } ( x , - v ) N _ { j } ^ { k } ( x , - v ) } \end{array}$ (product of two sign-flipped terms). By Proposition 26:

$$
\frac { \partial \overleftarrow { G } ^ { i } } { \partial x ^ { j } } ( x , v ) = \frac { \partial G ^ { i } } { \partial x ^ { j } } ( x , - v ) .
$$

Putting everything together, we have that

$$
\begin{array} { c } { { { \overleftarrow { R } } _ { j } ^ { i } ( x , v ) = \displaystyle { \frac { \partial G ^ { i } } { \partial x ^ { j } } } ( x , - v ) - \sum _ { k = 1 } ^ { m } \left( \frac { \partial N _ { j } ^ { i } } { \partial x ^ { k } } ( x , - v ) ( - v ) ^ { k } - \frac { \partial N _ { j } ^ { i } } { \partial v ^ { k } } ( x , - v ) G ^ { k } ( x , - v ) \right) } } \\ { { - \displaystyle { \sum _ { k = 1 } ^ { m } } N _ { k } ^ { i } ( x , - v ) N _ { j } ^ { k } ( x , - v ) = R _ { j } ^ { i } ( x , - v ) , } } \end{array}
$$

so that $\overleftarrow { R } _ { v } ( x , w ) = R _ { - v } ( x , w )$ . Since the fundamental tensor of $\overleftarrow { \mathcal { F } } \mathrm { ~ i s ~ } \overleftarrow { g } \left( x , v \right) = g ( x , - v )$ (Proposition 26’s proof), a basis orthonormal for $g ( x , - v )$ is also orthonormal for $\overleftarrow { g } \left( x , v \right)$ , so we may use the same basis B for both. Hence

$$
\operatorname { R i c } ^ { \overleftarrow { \mathcal { F } } } { ( x , v ) } = \sum _ { i = 1 } ^ { m - 1 } \overleftarrow { g } \left( x , v \right) _ { k l } \overleftarrow { R } _ { v } { ( x , e _ { i } ) } ^ { k } e _ { i } ^ { l } = \sum _ { i = 1 } ^ { m - 1 } g ( x , - v ) _ { k l } R _ { - v } ( x , e _ { i } ) ^ { k } e _ { i } ^ { l } = \operatorname { R i c } ^ { \mathcal { F } } ( x , - v ) .
$$

Finally, we need the behavior of the S-curvature under reversal. Since $- \mathcal { B } _ { x }$ and ${ \mathcal { B } } _ { x }$ have the same volume<sup>5</sup>, $\mathcal { F }$ and share the Busemann–Hausdorff measure, and we denote by $ $ and $\dot { \bar { S } }$ the distortion, the S-curvature and its derivative along geodesics of , with respect to m<sub>BH</sub>.

Lemma 30 (S-curvature under reversal). For every $x \in \mathcal { M }$ and $v \in T _ { x } { \mathcal { M } } \setminus \{ 0 \}$ , we have that $\overleftarrow { \tau } \left( x , v \right) =$ $\tau ( x , - v ) , \stackrel {  } { S } ( x , v ) = - S ( x , - v ) a n d \stackrel {  } { S } ( x , v ) = \dot { S } ( x , - v ) .$

Proof. The distortion only involves the fundamental tensor and the density $\sigma _ { \mathrm { B H } }$ . Since $\overleftarrow { g } _ { i j } ( x , v ) =$ $g _ { i j } ( x , - v )$ (see the proof of Proposition 26) and $\sigma _ { \mathrm { B H } }$ is shared by $\mathcal { F }$ and , we get $\overleftarrow { \tau } \left( x , v \right) = \tau ( x , - v )$ Let γ be the geodesic of $\mathcal { F }$ with $\gamma ( 0 ) = x$ and $\dot { \gamma } ( 0 ) = - v ,$ defined on some interval $( - \delta , \delta )$ , and let $\bar { \gamma } ( t ) =$ $\gamma ( - t )$ , so that $\bar { \gamma } ( 0 ) = x$ and $\dot { \bar { \gamma } } ( 0 ) = \bar { v }$ . Using the geodesic equation of $\mathcal { F }$ and Proposition 26, we have that

$$
\ddot { \gamma } ^ { k } ( t ) = \ddot { \gamma } ^ { k } ( - t ) = - 2 G ^ { k } \big ( \gamma ( - t ) , \dot { \gamma } ( - t ) \big ) = - 2 G ^ { k } \big ( \bar { \gamma } ( t ) , - \dot { \bar { \gamma } } ( t ) \big ) = - 2 \overleftarrow { G } ^ { k } \big ( \bar { \gamma } ( t ) , \dot { \bar { \gamma } } ( t ) \big ) ,
$$

so that $\bar { \gamma }$ is the geodesic of with initial conditions $( x , v )$ . Hence, for $t \in \left( - \delta , \delta \right)$

$$
\overleftarrow { \gamma } \left( \bar { \gamma } ( t ) , \dot { \bar { \gamma } } ( t ) \right) = \tau \left( \bar { \gamma } ( t ) , - \dot { \bar { \gamma } } ( t ) \right) = \tau \left( \gamma ( - t ) , \dot { \gamma } ( - t ) \right) .
$$

Differentiating once and twice at $t = 0 \mathrm { g i v e s } \overleftarrow { S } ( x , v ) = - S ( x , - v ) \mathrm { a n d } \overleftarrow { S } ( x , v ) = \dot { S } ( x , - v )$

Lemma 31. The moments of the kernel for the reverse Finsler metric are related to the moments of the kernelfor the Finsler metric. In particular,for $n \in \mathbb { N }$ we have that:

$$
\begin{array} { l l } { { \overleftarrow { m } _ { n } ( x ) = ( - 1 ) ^ { n } m _ { n } ( x ) , \qquad \overleftarrow { \Lambda } ^ { k } ( x ) = \Lambda ^ { k } ( x ) , } } \\ { { \overleftarrow { s } _ { n } ( x ) = ( - 1 ) ^ { n + 1 } s _ { n } ( x ) , \qquad \overleftarrow { \Upsilon } ( x ) = \Upsilon ( x ) . } } \end{array}
$$

Proof. Since $\mathcal { F }$ and share the density $\sigma _ { \mathrm { B H } }$ , which does not depend on $v ,$ it factors out of all the integrals below, and we omit it. First, we show the relation between $\scriptstyle { \overleftarrow { m } } _ { n }$ and $m _ { n }$ . By the change of variable $u = - v ,$

$$
\begin{array} { l } { \displaystyle \overleftarrow { m } _ { n } ( x ) = \int _ { T _ { x } , \mathcal { M } } K ( \overleftarrow { \mathfrak { F } } ( x , v ) ) v ^ { \otimes n } \mathrm { d } \lambda _ { x } ( v ) } \\ { \displaystyle \qquad = \int _ { T _ { x } , \mathcal { M } } K ( \mathcal { F } ( x , - u ) ) ( - u ) ^ { \otimes n } \mathrm { d } \lambda _ { x } ( u ) } \\ { \displaystyle \qquad = ( - 1 ) ^ { n } \int _ { T _ { x } , \mathcal { M } } K ( \mathcal { F } ( x , u ) ) u ^ { \otimes n } \mathrm { d } \lambda _ { x } ( u ) } \\ { \displaystyle \qquad = ( - 1 ) ^ { n } m _ { n } ( x ) . } \end{array}
$$

The change of variable $u = - v$ has Jacobian determinant det $\ b ( - I _ { m } ) = ( - 1 ) ^ { m }$ , whose absolute value is 1, so that it preserves $\lambda _ { x } ,$ , and the sign factor $( - 1 ) ^ { n }$ comes only from the $v ^ { \otimes n }$ term. This argument is also used below for the other terms.

It remains to treat the $\overleftarrow { \Lambda } ^ { k } ( x ) , \overleftarrow { s } _ { n } ( x )$ and $\overleftarrow { \Upsilon } \left( x \right)$ terms. For $\overleftarrow { \Lambda } ^ { k } ( x )$ , recall that $\overleftarrow { \tilde { \Gamma } } _ { i j } ^ { k } ( v ) v ^ { i } v ^ { j } = 2 \overleftarrow { G } ^ { k } ( x , v )$ and $\tilde { \Gamma } _ { i j } ^ { k } ( x , v ) v ^ { i } v ^ { j } = 2 G ^ { k } ( x , v )$ . Using $\overleftarrow { G } ^ { k } ( x , v ) = G ^ { k } ( x , - v )$ (Proposition 26),

$$
\begin{array} { l l } { \overleftarrow { \Lambda } ^ { k } ( x ) = 2 \displaystyle \int _ { T _ { x } , \mathcal { M } } K ( \overleftarrow { \mathfrak { F } } ( x , v ) ) \overleftarrow { G } ^ { k } ( x , v ) \mathrm { d } \lambda _ { x } ( v ) } \\ { \displaystyle \quad = 2 \int _ { T _ { x } , \mathcal { M } } K ( \mathcal { F } ( x , - v ) ) G ^ { k } ( x , - v ) \mathrm { d } \lambda _ { x } ( v ) } \\ { \displaystyle \quad = 2 \int _ { T _ { x } , \mathcal { M } } K ( \mathcal { F } ( x , u ) ) G ^ { k } ( x , u ) \mathrm { d } \lambda _ { x } ( u ) } & { \qquad \mathrm { b y ~ c h a n g e ~ o f ~ v a r i a b l e ~ } u = - v } \\ { \displaystyle \qquad = \Lambda ^ { k } ( x ) . } \end{array}
$$

For the S-curvature terms, using Lemma 30 and the same change of variable,

$$
\begin{array} { l } { { \displaystyle { \overleftarrow { \overline { { s } } } } _ { n } ( x ) = \int _ { T _ { x } . \mathcal { M } } K ( { \overleftarrow { \overline { { \mathcal F } } } } ( x , v ) ) ^ { \overleftarrow { \overline { { S } } } } ( x , v ) v ^ { \otimes n } \mathrm { d } \lambda _ { x } ( v ) = - \int _ { T _ { x } . \mathcal { M } } K ( { \mathcal F } ( x , - v ) ) S ( x , - v ) v ^ { \otimes n } \mathrm { d } \lambda _ { x } ( v ) } } \\ { ~ } \\ { { \displaystyle ~ = ( - 1 ) ^ { n + 1 } \int _ { T _ { x } . \mathcal { M } } K ( { \mathcal F } ( x , u ) ) S ( x , u ) u ^ { \otimes n } \mathrm { d } \lambda _ { x } ( u ) = ( - 1 ) ^ { n + 1 } s _ { n } ( x ) } . } \end{array}
$$

Finally, for $\overleftarrow { \Upsilon } \left( x \right)$ , Lemma 30 gives $\overleftarrow { S } ( x , v ) ^ { 2 } - \overleftarrow { S } ( x , v ) = S ( x , - v ) ^ { 2 } - \dot { S } ( x , - v )$ , and Proposition 29 gives $\operatorname { R i c } ^ { \overleftarrow { \mathcal { F } } } ( x , v ) = \operatorname { R i c } ^ { \mathcal { F } } ( x , - v )$ , so that, by the change of variable $u = - v ,$

$$
\begin{array} { r l } & { \overleftarrow { \boldsymbol { \Upsilon } } \left( x \right) = \displaystyle \int _ { T _ { x } . \boldsymbol { \mathcal { M } } } K ( \overleftarrow { \boldsymbol { \mathcal { F } } } ( x , v ) ) \left( \overleftarrow { \boldsymbol { S } } \left( x , v \right) ^ { 2 } - \overleftarrow { \boldsymbol { S } } \left( x , v \right) - \frac { 1 } { 3 } \mathrm { R i c } ^ { \overleftarrow { \boldsymbol { \mathcal { F } } } } ( x , v ) \right) \mathrm { d } \lambda _ { x } ( v ) } \\ & { \qquad = \displaystyle \int _ { T _ { x } . \boldsymbol { \mathcal { M } } } K ( \boldsymbol { \mathcal { F } } ( x , - v ) ) \left( \boldsymbol { S } ( x , - v ) ^ { 2 } - \dot { S } ( x , - v ) - \frac { 1 } { 3 } \mathrm { R i c } ^ { \boldsymbol { \mathcal { F } } } ( x , - v ) \right) \mathrm { d } \lambda _ { x } ( v ) } \\ & { \qquad = \displaystyle \int _ { T _ { x } . \boldsymbol { \mathcal { M } } } K ( \boldsymbol { \mathcal { F } } ( x , u ) ) \left( \boldsymbol { S } ( x , u ) ^ { 2 } - \dot { S } ( x , u ) - \frac { 1 } { 3 } \mathrm { R i c } ^ { \boldsymbol { \mathcal { F } } } ( x , u ) \right) \mathrm { d } \lambda _ { x } ( u ) } \\ & { \qquad = \boldsymbol { \Upsilon } ( x ) . } \end{array}
$$

Putting everything together, we have shown that $\overleftarrow { m } _ { n } ( x ) = ( - 1 ) ^ { n } m _ { n } ( x ) , \overleftarrow { \Lambda } ^ { k } ( x ) = \Lambda ^ { k } ( x ) , \overleftarrow { s } _ { n } ( x ) =$ $( - 1 ) ^ { n + 1 } s _ { n } ( x )$ and $\overleftarrow { \Upsilon } \left( x \right) = \Upsilon ( x )$ , which concludes the proof. □

Recap. The goal of this subsection was to study the behavior of the reverse Finsler metric and its associated transport operator $\mathcal { G } _ { \overleftarrow { \mathcal { F } } }$ . We have shown that the moments of the kernel for the reverse Finsler metric are related to the moments of the kernel for the Finsler metric, with $\overleftarrow { m } _ { n } ( x ) = ( - 1 ) ^ { n } m _ { n } ( x ) , \overleftarrow { \Lambda } ^ { k } ( x ) =$ $\Lambda ^ { k } ( x ) , \overleftarrow { s } { } _ { n } ( x ) = ( - 1 ) ^ { n + 1 } s _ { n } ( x )$ and $\overleftarrow { \Upsilon } \left( x \right) = \Upsilon ( x )$ . This allows us to derive the moment expansion of the transport operator $\mathcal { G } _ { \overleftarrow { \mathcal { F } } }$ in terms of the moments of the kernel for $\mathcal { F }$ . In particular, we have the following proposition.

Proposition 32. The transport operator $\mathcal { G } _ { \overleftarrow { \mathcal { F } } }$ associated with the reverse Finsler metric has the following moment expansion:

$$
\mathcal { G } _ { \mathfrak { F } } [ f ] = m _ { 0 } f - \varepsilon \left( m _ { 1 } ^ { i } \partial _ { i } f - s _ { 0 } f \right) + \frac { \varepsilon ^ { 2 } } { 2 } \left( m _ { 2 } ^ { i j } \partial _ { i } \partial _ { j } f - \left( \Lambda ^ { k } + 2 s _ { 1 } ^ { k } \right) \partial _ { k } f + \Upsilon f \right) + O ( \varepsilon ^ { 3 } ) .
$$

This follows from Theorem 1 applied to , which shares the density $\sigma _ { \mathrm { B H } }$ with $\mathcal { F }$ , together with Lemma 31. Compared with the expansion of $\mathcal { G } _ { \mathcal { F } }$ , the terms of odd order in ε change sign under reversal, while the terms of even order do not.

## PROOF OF THE MOMENT EXPANSION OF THE SYMMETRIC AND ANTI-SYMMETRIC TRANSPORT OPERATORS

Using Proposition 32, the moment expansions of the symmetric and antisymmetric parts $\mathscr { G } _ { \mathcal { F } } ^ { s } = ( \mathscr { G } _ { \mathcal { F } } + \mathscr { G } _ { \frac {  } { \mathcal { F } } } ) / 2$ and $\mathcal { G } _ { \mathcal { F } } ^ { a } = ( \mathcal { G _ { F } } - \mathcal { G } _ { \overleftarrow { \mathcal { F } } } ) / 2$ of the transport operator follow directly: up to order $\varepsilon ^ { 2 }$ , the symmetric part retains the even-order terms and the antisymmetric part the odd-order ones. This yields

$$
\begin{array} { r l } & { \displaystyle \mathcal { G } _ { \mathcal { F } } ^ { s } [ f ] = m _ { 0 } f + \frac { \varepsilon ^ { 2 } } { 2 } \left( m _ { 2 } ^ { i j } \partial _ { i } \partial _ { j } f - \left( \Lambda ^ { k } + 2 s _ { 1 } ^ { k } \right) \partial _ { k } f + \Upsilon f \right) + O ( \varepsilon ^ { 3 } ) \quad \mathrm { a n d } } \\ & { \displaystyle \mathcal { G } _ { \mathcal { F } } ^ { a } [ f ] = \varepsilon \left( m _ { 1 } ^ { i } \partial _ { i } f - s _ { 0 } f \right) + O ( \varepsilon ^ { 3 } ) . } \end{array}
$$

It is important to note that, in practice, one does not observe the transport operator $\mathcal { G } _ { \mathcal { F } } [ f ]$ directly, as points are rarely sampled uniformly from the manifold. Instead, one typically observes $\mathcal { G } _ { \mathcal { F } } [ \rho f ]$ applied to the density $\rho$ of the sampling distribution. Looking at the expansion of $\mathbf { \dot { \mathcal { G } } } _ { \mathcal { F } } [ \rho \mathbf { \dot { \mathcal { f } } } ]$ , we have:

$$
\begin{array} { l } { { \displaystyle { \mathcal { G } } _ { \mathcal { F } } [ \rho f ] = m _ { 0 } \rho f + \varepsilon \left( m _ { 1 } ^ { i } \partial _ { i } ( \rho f ) - s _ { 0 } \rho f \right) } } \\ { { \displaystyle ~ + \frac { \varepsilon ^ { 2 } } { 2 } \left( m _ { 2 } ^ { i j } \partial _ { i } \partial _ { j } ( \rho f ) - \left( \Lambda ^ { k } + 2 s _ { 1 } ^ { k } \right) \partial _ { k } ( \rho f ) + \Upsilon \rho f \right) + O ( \varepsilon ^ { 3 } ) } } \end{array}
$$

so that every term is ‘polluted’ by the density $\rho .$ This calls for a normalization, which we now introduce.

## θ-NORMALIZED TRANSPORT OPERATOR

Our goal is now to define a θ-normalized transport operator $\mathcal { G } _ { \mathcal { F } } ^ { ( \theta ) }$ that mitigates the impact of the sampling density $\rho$ on the limit operator. We define $\mathcal { G } _ { \mathcal { F } } ^ { ( \theta ) }$ as:

$$
{ \mathcal G } _ { \mathcal F } ^ { ( \theta ) } [ f ] ( x ) = \frac { 1 } { q _ { \varepsilon } ( x ) ^ { \theta } } { \mathcal G } _ { \mathcal F } \left[ \frac { f } { q _ { \varepsilon } ^ { \theta } } \right] ( x ) ,
$$

where $q _ { \varepsilon } ( x )$ is a normalization factor. Specifically, we take $q _ { \varepsilon } = ( d ^ { \prime } + d ) / 2$ . This choice allows us to clean up the expansion of $\mathcal { G } _ { \mathcal { F } } ^ { ( \theta ) } [ \rho f ]$ and analyze its behavior as $\varepsilon  0$ . Since $q _ { \varepsilon }$ is invariant under reversal, the normalization $q _ { \varepsilon } ( x ) ^ { \theta } q _ { \varepsilon } ( y ) ^ { \theta }$ is symmetric in $( x , y )$ , and the symmetric and antisymmetric parts of $\mathcal { G } _ { \mathcal { F } } ^ { ( \theta ) }$ are obtained from those of G as

$$
\mathcal { G } _ { \mathcal { F } } ^ { ( \theta , s ) } [ f ] = \frac { 1 } { q _ { \varepsilon } \theta } \mathcal { G } _ { \mathcal { F } } ^ { s } \left[ \frac { f } { q _ { \varepsilon } \theta } \right] \quad \mathrm { a n d } \quad \mathcal { G } _ { \mathcal { F } } ^ { ( \theta , a ) } [ f ] = \frac { 1 } { q _ { \varepsilon } \theta } \mathcal { G } _ { \mathcal { F } } ^ { a } \left[ \frac { f } { q _ { \varepsilon } \theta } \right] .
$$

We use $q _ { \varepsilon }$ rather than $d ^ { \prime }$ or d alone because its expansion has no term of order $\varepsilon ,$ , which makes it a cleaner proxy for the sampling density $\rho .$ Indeed, the expansions of $d ^ { \prime }$ and d read

$$
\begin{array} { c } { { d = \displaystyle \mathcal { G } _ { \mathcal { F } } [ \rho ] = m _ { 0 } \rho + \varepsilon \left( m _ { 1 } ^ { i } \partial _ { i } \rho - s _ { 0 } \rho \right) } } \\ { { \displaystyle + \frac { \varepsilon ^ { 2 } } { 2 } \left( m _ { 2 } ^ { i j } \partial _ { i } \partial _ { j } \rho - \left( \Lambda ^ { k } + 2 s _ { 1 } ^ { k } \right) \partial _ { k } \rho + \Upsilon \rho \right) + O ( \varepsilon ^ { 3 } ) , } } \\ { { d ^ { \prime } = \displaystyle \mathcal { G } _ { \Xi } [ \rho ] = m _ { 0 } \rho - \varepsilon \left( m _ { 1 } ^ { i } \partial _ { i } \rho - s _ { 0 } \rho \right) } } \\ { { \displaystyle + \frac { \varepsilon ^ { 2 } } { 2 } \left( m _ { 2 } ^ { i j } \partial _ { i } \partial _ { j } \rho - \left( \Lambda ^ { k } + 2 s _ { 1 } ^ { k } \right) \partial _ { k } \rho + \Upsilon \rho \right) + O ( \varepsilon ^ { 3 } ) . } } \end{array}
$$

Hence $q _ { \varepsilon } = m _ { 0 } \rho + O ( \varepsilon ^ { 2 } )$ , whereas $d ^ { \prime }$ and d each carry a term of order ε. Moreover, $m _ { 0 } = m \mu _ { 0 } \mathrm { V o l } _ { \mathrm { E u c } } ( \mathbb { B } ^ { m } )$ does not depend on x (Section $\mathrm { D } . 4 )$ , so that $q _ { \varepsilon }$ is proportional to $\rho$ up to $O ( \varepsilon ^ { 2 } )$ . We now define $\phi ^ { ( \theta ) } ( x ) =$ $\rho ( x ) / q _ { \varepsilon } ( x ) ^ { \theta }$ , and look at the expansion of the observable quantity $\mathcal { G } _ { \mathcal { F } } ^ { ( \theta ) } [ \rho f ]$

$$
\begin{array} { r l r } {  { \mathcal { G } _ { \mathcal { F } } ^ { ( \theta ) } [ \rho f ] = \frac { 1 } { q _ { \varepsilon } \theta } \mathcal { G } _ { \mathcal { F } } [ \phi ^ { ( \theta ) } f ] } } \\ & { } & { = \frac { 1 } { q _ { \varepsilon } \theta } \Big [ m _ { 0 } \phi ^ { ( \theta ) } f + \varepsilon ( m _ { 1 } ^ { i } \partial _ { i } ( \phi ^ { ( \theta ) } f ) - s _ { 0 } \phi ^ { ( \theta ) } f ) } \\ & { } & { \quad + \frac { \varepsilon ^ { 2 } } { 2 } ( m _ { 2 } ^ { i j } \partial _ { i } \partial _ { j } ( \phi ^ { ( \theta ) } f ) - ( \Lambda ^ { k } + 2 s _ { 1 } ^ { k } ) \partial _ { k } ( \phi ^ { ( \theta ) } f ) + \Upsilon \phi ^ { ( \theta ) } f ) + O ( \varepsilon ^ { 3 } ) \Big ] . } \end{array}
$$

Expanding the derivatives with the Leibniz rule gives

$$
\begin{array} { r } { \partial _ { i } ( \phi ^ { ( \theta ) } f ) = \phi ^ { ( \theta ) } \partial _ { i } f + f \partial _ { i } \phi ^ { ( \theta ) } , \qquad } \\ { m _ { 2 } ^ { i j } \partial _ { i } \partial _ { j } ( \phi ^ { ( \theta ) } f ) = m _ { 2 } ^ { i j } \phi ^ { ( \theta ) } \partial _ { i } \partial _ { j } f + 2 m _ { 2 } ^ { i j } \partial _ { i } \phi ^ { ( \theta ) } \partial _ { j } f + m _ { 2 } ^ { i j } f \partial _ { i } \partial _ { j } \phi ^ { ( \theta ) } , } \end{array}
$$

where we used the fact that $m _ { 2 } ^ { i j } = m _ { 2 } ^ { j i }$ . Plugging these into the expansion of $\mathcal { G } _ { \mathcal { F } } ^ { ( \theta ) } [ \rho f ]$ , we obtain:

$$
\begin{array} { l } { { \displaystyle \mathcal { G } _ { \mathcal { F } } ^ { ( \theta ) } [ \rho f ] = \frac { 1 } { q _ { \varepsilon } \theta } \Bigg [ m _ { 0 } \phi ^ { ( \theta ) } f + \varepsilon \left( m _ { 1 } ^ { i } \left( \phi ^ { ( \theta ) } \partial _ { i } f + f \partial _ { i } \phi ^ { ( \theta ) } \right) - s _ { 0 } \phi ^ { ( \theta ) } f \right) } } \\ { { \displaystyle \qquad + \frac { \varepsilon ^ { 2 } } { 2 } \Bigg ( m _ { 2 } ^ { i j } ( \phi ^ { ( \theta ) } \partial _ { i } \partial _ { j } f + 2 \partial _ { i } \phi ^ { ( \theta ) } \partial _ { j } f + f \partial _ { i } \partial _ { j } \phi ^ { ( \theta ) } ) - \left( \Lambda ^ { k } + 2 s _ { 1 } ^ { k } \right) \left( \phi ^ { ( \theta ) } \partial _ { k } f + f \partial _ { k } \phi ^ { ( \theta ) } \right) } } \\ { { \displaystyle \qquad + \Upsilon \phi ^ { ( \theta ) } f \Bigg ) + O ( \varepsilon ^ { 3 } ) \Bigg ] . } } \end{array}\tag{13}
$$

We then define the symmetric and antisymmetric normalized transport operators $\mathcal { P } ^ { ( \theta , s ) }$ and $\mathcal { P } ^ { ( \theta , a ) }$ as

$$
\mathcal { P } ^ { ( \theta , \bullet ) } [ f ] = \frac { 1 } { d ^ { ( \theta , s ) } } \left( \mathcal { G } ^ { ( \theta , \bullet ) } [ \rho f ] - f \mathcal { G } ^ { ( \theta , \bullet ) } [ \rho ] \right)\tag{14}
$$

for $\bullet \in \{ s , a \}$ , where $d ^ { ( \theta , s ) } = \mathcal { G } _ { \mathcal { F } } ^ { ( \theta , s ) } [ \rho ]$ is the symmetric normalized degree defined below. This expression centers the operators $\mathcal { G }$ around their respective degrees, so that $\mathcal { P }$ vanishes on constant functions.

Expansion for the degrees of the θ-normalized operator. The in- and out-degrees $d ^ { \prime ( \theta ) }$ and $d ^ { ( \theta ) }$ of the θ-normalized kernel expand as

$$
\begin{array} { c } { { d ^ { ( \theta ) } = { \mathcal G } _ { \mathcal F } ^ { ( \theta ) } [ \rho ] = \displaystyle \frac { 1 } { q _ { \varepsilon } { } ^ { \theta } } \Big ( m _ { 0 } \phi ^ { ( \theta ) } + \varepsilon \Big ( m _ { 1 } ^ { i } \partial _ { i } \phi ^ { ( \theta ) } - s _ { 0 } \phi ^ { ( \theta ) } \Big ) } } \\ { { + \displaystyle \frac { \varepsilon ^ { 2 } } { 2 } \Big ( m _ { 2 } ^ { i j } \partial _ { i } \partial _ { j } \phi ^ { ( \theta ) } - \big ( \Lambda ^ { k } + 2 s _ { 1 } ^ { k } \big ) \partial _ { k } \phi ^ { ( \theta ) } + \Upsilon \phi ^ { ( \theta ) } \Big ) \Big ) + { \cal O } \big ( \varepsilon ^ { 3 } ) , } } \end{array}
$$

and $d ^ { \prime ( \theta ) } = \mathcal { G } _ { \overleftarrow { \mathcal { F } } } ^ { ( \theta ) } [ \rho ]$ has the same expansion, with the opposite sign in front of the term of order ε. By definition, the degrees of the symmetric and antisymmetric normalized kernels are

$$
\begin{array} { l } { { d ^ { ( \theta , s ) } = \mathcal { G } _ { \mathcal { F } } ^ { ( \theta , s ) } [ \rho ] = \frac { 1 } { 2 } \left( \mathcal { G } _ { \mathcal { F } } ^ { ( \theta ) } [ \rho ] + \mathcal { G } _ { \overline { { \mathcal { F } } } } ^ { ( \theta ) } [ \rho ] \right) = \frac { d ^ { ( \theta ) } + d ^ { \prime ( \theta ) } } { 2 } , } } \\ { { d ^ { ( \theta , a ) } = \mathcal { G } _ { \mathcal { F } } ^ { ( \theta , a ) } [ \rho ] = \frac { 1 } { 2 } \left( \mathcal { G } _ { \mathcal { F } } ^ { ( \theta ) } [ \rho ] - \mathcal { G } _ { \overline { { \mathcal { F } } } } ^ { ( \theta ) } [ \rho ] \right) = \frac { d ^ { ( \theta ) } - d ^ { \prime ( \theta ) } } { 2 } . } } \end{array}
$$

Plugging the expansions of $d ^ { \prime ( \theta ) }$ and $d ^ { ( \theta ) }$ into the above expressions, we have that:

$$
d ^ { ( \theta , s ) } = \frac { 1 } { q _ { \varepsilon } \theta } \left( m _ { 0 } \phi ^ { ( \theta ) } + \frac { \varepsilon ^ { 2 } } { 2 } \left( m _ { 2 } ^ { i j } \partial _ { i } \partial _ { j } \phi ^ { ( \theta ) } - \left( \Lambda ^ { k } + 2 s _ { 1 } ^ { k } \right) \partial _ { k } \phi ^ { ( \theta ) } + \Upsilon \phi ^ { ( \theta ) } \right) \right) + O ( \varepsilon ^ { 3 } ) ,\tag{15}
$$

$$
d ^ { ( \theta , a ) } = \frac { 1 } { { q _ { \varepsilon } } ^ { \theta } } \varepsilon \left( m _ { 1 } ^ { i } \partial _ { i } \phi ^ { ( \theta ) } - s _ { 0 } \phi ^ { ( \theta ) } \right) + O ( \varepsilon ^ { 3 } ) .\tag{16}
$$

Using the fact that $1 / ( 1 + x ) = 1 - x + O ( x ^ { 2 } )$ , and noting that the bracket of order $\varepsilon ^ { 2 }$ in Equation (15) is $\mathcal { A } _ { 2 } [ \phi ^ { ( \theta ) } ]$ (Definition 21), we can write the following expansion for $1 / d ^ { ( \theta , s ) }$

$$
\begin{array} { l } { \displaystyle \frac { 1 } { d ^ { ( \theta , s ) } } = \left( \frac { m _ { 0 } \phi ^ { ( \theta ) } } { { q _ { \varepsilon } } ^ { \theta } } \right) ^ { - 1 } \left( 1 + \frac { \varepsilon ^ { 2 } } { 2 m _ { 0 } \phi ^ { ( \theta ) } } { \mathcal A } _ { 2 } [ \phi ^ { ( \theta ) } ] + O ( \varepsilon ^ { 3 } ) \right) ^ { - 1 } } \\ { \displaystyle \qquad = \frac { { q _ { \varepsilon } } ^ { \theta } } { m _ { 0 } \phi ^ { ( \theta ) } } \left( 1 - \frac { \varepsilon ^ { 2 } } { 2 m _ { 0 } \phi ^ { ( \theta ) } } { \mathcal A } _ { 2 } [ \phi ^ { ( \theta ) } ] + O ( \varepsilon ^ { 3 } ) \right) } \\ { \displaystyle \qquad = \frac { { q _ { \varepsilon } } ^ { \theta } } { m _ { 0 } \phi ^ { ( \theta ) } } + O ( \varepsilon ^ { 2 } ) } \end{array}\tag{17}
$$

We now expand the symmetric and antisymmetric normalized transport operators $\mathcal { P } ^ { ( \theta , s ) }$ and $\mathcal { P } ^ { ( \theta , a ) }$ of Equation (14).

Expansion for the symmetric part. By putting together Equation (13) and Equation (15), we have that:

$$
\begin{array} { r l } & { \mathcal { G } _ { \mathcal { F } } ^ { ( \theta , s ) } [ \rho f ] - f d ^ { ( \theta , s ) } = \frac { \mathcal { E } ^ { 2 } } { 2 q _ { z } ^ { 6 } } \left( m _ { 2 } ^ { i j } \left( \phi ^ { ( \theta ) } \partial _ { i } \partial _ { j } f + 2 \partial _ { i } \phi ^ { ( \theta ) } \partial _ { j } f + f \partial _ { i } \partial _ { j } \phi ^ { ( \theta ) } - f \partial _ { i } \partial _ { j } \phi ^ { ( \theta ) } \right) \right. } \\ & { \quad \quad \quad \quad \quad \left. - \left( \Lambda ^ { k } + 2 s _ { 1 } ^ { k } \right) \left( \phi ^ { ( \theta ) } \partial _ { k } f + f \partial _ { k } \phi ^ { ( \theta ) } - f \partial _ { k } \phi ^ { ( \theta ) } \right) \right. } \\ & { \quad \quad \quad \quad \quad \left. + \Upsilon \left( \phi ^ { ( \theta ) } f - \phi ^ { ( \theta ) } f \right) \right) + O ( \varepsilon ^ { 3 } ) } \\ & { \quad \quad \quad \quad = \frac { \mathcal { E } ^ { 2 } } { 2 q _ { \varepsilon } ^ { 6 } } \left( m _ { 2 } ^ { i j } \left( \phi ^ { ( \theta ) } \partial _ { i } \partial _ { j } f + 2 \partial _ { i } \phi ^ { ( \theta ) } \partial _ { j } f \right) - \phi ^ { ( \theta ) } \left( \Lambda ^ { k } + 2 s _ { 1 } ^ { k } \right) \partial _ { k } f \right) + O ( \varepsilon ^ { 3 } ) . } \end{array}
$$

Hence, by Equation (17), we have that:

$$
\begin{array} { l } { { \displaystyle { \mathcal { P } } ^ { ( \theta , s ) } [ f ] = \frac { 1 } { d ^ { ( \theta , s ) } } \left( \mathcal { G } _ { \mathcal { F } } ^ { ( \theta , s ) } [ \rho f ] - f d ^ { ( \theta , s ) } \right) } } \\ { { \displaystyle ~ = \frac { q _ { \varepsilon } \theta } { m _ { 0 } \phi ^ { ( \theta ) } } \frac { \varepsilon ^ { 2 } } { 2 q _ { \varepsilon } \theta } \Bigg ( m _ { 2 } ^ { i j } \left( \phi ^ { ( \theta ) } \partial _ { i } \partial _ { j } f + 2 \partial _ { i } \phi ^ { ( \theta ) } \partial _ { j } f \right) - \phi ^ { ( \theta ) } \left( \Lambda ^ { k } + 2 s _ { 1 } ^ { k } \right) \partial _ { k } f \Bigg ) + O ( \varepsilon ^ { 3 } ) } } \\ { { \displaystyle ~ = \frac { \varepsilon ^ { 2 } } { 2 m _ { 0 } } \Bigg ( m _ { 2 } ^ { i j } \left( \partial _ { i } \partial _ { j } f + 2 \partial _ { i } \log \phi ^ { ( \theta ) } \partial _ { j } f \right) - \left( \Lambda ^ { k } + 2 s _ { 1 } ^ { k } \right) \partial _ { k } f \Bigg ) + O ( \varepsilon ^ { 3 } ) . } } \end{array}
$$

where we used $\partial _ { i } \log \phi ^ { ( \theta ) } = \partial _ { i } \phi ^ { ( \theta ) } / \phi ^ { ( \theta ) }$ . Note that the curvature term $\Upsilon ,$ , which only multiplies $f ,$ is removed by the centering.

Expansion for the anti-symmetric part. By putting together Equation (13) and Equation (16), we have that:

$$
\begin{array} { l } { { \displaystyle \mathcal { E } _ { \mathcal { F } } ^ { ( \theta , a ) } [ \rho f ] - f d ^ { ( \theta , a ) } = \frac { \mathcal { E } } { { q _ { \mathcal { F } } } ^ { \theta } } \left( m _ { 1 } ^ { i } \left( \phi ^ { ( \theta ) } \partial _ { i } f + f \partial _ { i } \phi ^ { ( \theta ) } \right) - s _ { 0 } \phi ^ { ( \theta ) } f - f \left( m _ { 1 } ^ { i } \partial _ { i } \phi ^ { ( \theta ) } - s _ { 0 } \phi ^ { ( \theta ) } \right) \right) + O ( \varepsilon ^ { 3 } ) } } \\ { { \displaystyle \qquad = \frac { \mathcal { E } } { { q _ { \mathcal { F } } } ^ { \theta } } m _ { 1 } ^ { i } \phi ^ { ( \theta ) } \partial _ { i } f + O ( \varepsilon ^ { 3 } ) . } } \end{array}
$$

In particular, the term $s _ { 0 }$ brought by the S-curvature only multiplies $f ,$ and is removed by the centering. Hence, by Equation (17), we have that:

$$
\begin{array} { r l } & { \mathcal { P } ^ { ( \theta , a ) } [ f ] = \displaystyle \frac { 1 } { d ^ { ( \theta , s ) } } \left( \mathcal { G } _ { \mathcal { F } } ^ { ( \theta , a ) } [ \rho f ] - f d ^ { ( \theta , a ) } \right) } \\ & { \quad \quad \quad = \displaystyle \frac { q _ { \varepsilon } ^ { \theta } } { m _ { 0 } \phi ^ { ( \theta ) } } \frac { \varepsilon } { q _ { \varepsilon } ^ { \theta } } m _ { 1 } ^ { i } \phi ^ { ( \theta ) } \partial _ { i } f + O ( \varepsilon ^ { 3 } ) } \\ & { \quad \quad \quad = \displaystyle \frac { \varepsilon } { m _ { 0 } } m _ { 1 } ^ { i } \partial _ { i } f + O ( \varepsilon ^ { 3 } ) . } \end{array}
$$

Consequently, if we introduce the normalized quantities $\tilde { m } _ { n } = m _ { n } / m _ { 0 } , \tilde { \Lambda } ^ { k } = \Lambda ^ { k } / m _ { 0 }$ and $\tilde { s } _ { 1 } ^ { k } = s _ { 1 } ^ { k } / m _ { 0 }$ , we have that:

$$
{ \mathcal { P } } ^ { ( \theta , s ) } [ f ] = \frac { \varepsilon ^ { 2 } } { 2 } \left( \tilde { m } _ { 2 } ^ { i j } \left( \partial _ { i } \partial _ { j } f + 2 \partial _ { i } \log \phi ^ { ( \theta ) } \partial _ { j } f \right) - \left( \tilde { \Lambda } ^ { k } + 2 \tilde { s } _ { 1 } ^ { k } \right) \partial _ { k } f \right) + O ( \varepsilon ^ { 3 } ) ,
$$

$$
{ \mathcal { P } } ^ { ( \theta , a ) } [ f ] = \varepsilon { \tilde { m } } _ { 1 } ^ { i } \partial _ { i } f + O ( \varepsilon ^ { 3 } ) .
$$

Impact of the normalization parameter. The antisymmetric normalized transport operator $\mathcal { P } ^ { ( \theta , a ) }$ depends neither on the sampling density $\rho$ nor on the normalization parameter θ. We thus focus on the impact of the normalization parameter θ on the symmetric normalized transport operator $\mathcal { P } ^ { ( \theta , s ) }$ . This normalization appears through the term $\partial _ { i } \log \phi ^ { ( \theta ) }$ , by definition of $\phi ^ { ( \theta ) } = \rho / { q _ { \varepsilon } } ^ { \theta }$ . We have that:

$$
\partial _ { i } \log \phi ^ { ( \theta ) } = \partial _ { i } \log \left( \frac { \rho } { q _ { \varepsilon } \theta } \right) = \partial _ { i } \log \rho - \theta \partial _ { i } \log q _ { \varepsilon } .
$$

Moreover, we have seen that $q _ { \varepsilon } = m _ { 0 } \rho + O ( \varepsilon ^ { 2 } )$ , where $m _ { 0 } = m \mu _ { 0 } \mathrm { V o l } _ { \mathrm { E u c } } ( \mathbb { B } ^ { m } )$ does not depend $\mathrm { o n } \ x \ ( { \mathrm { S e c } } -$ tion $_ { \mathrm { D . 4 ) } }$ , so that $\partial _ { i } \log q _ { \varepsilon } = \bar { \partial _ { i } } \log \rho + \dot { O } ( \varepsilon ^ { 2 } )$ . Hence, we have that:

$$
\partial _ { i } \log \phi ^ { ( \theta ) } = ( 1 - \theta ) \partial _ { i } \log \rho + O ( \varepsilon ^ { 2 } ) .
$$

The leading-order term in the expansion of $\mathcal { P } ^ { ( \theta , s ) }$ is thus

$$
\mathcal { P } ^ { ( \theta , s ) } [ f ] = \frac { \varepsilon ^ { 2 } } { 2 } \left( \tilde { m } _ { 2 } ^ { i j } \left( \partial _ { i } \partial _ { j } f + 2 ( 1 - \theta ) \partial _ { i } \log \rho \partial _ { j } f \right) - \left( \tilde { \Lambda } ^ { k } + 2 \tilde { s } _ { 1 } ^ { k } \right) \partial _ { k } f \right) + O ( \varepsilon ^ { 3 } ) .
$$

In particular, for $\theta = 1$ the sampling density disappears from the limit, exactly as in the Riemannian setting (Coifman & Lafon, 2006). Taking the limits for both operators when $\varepsilon \to 0$ yields $\mathcal { L } ^ { a } f = \tilde { m } _ { 1 } ^ { i } \partial _ { i } f$ , and

$$
\mathcal { L } ^ { s } f = \frac { 1 } { 2 } \tilde { m } _ { 2 } ^ { i j } \partial _ { i } \partial _ { j } f + ( 1 - \theta ) \tilde { m } _ { 2 } ^ { i j } \partial _ { i } \log \rho \partial _ { j } f - \frac { 1 } { 2 } \bigl ( \tilde { \Lambda } ^ { k } + 2 \tilde { s } _ { 1 } ^ { k } \bigr ) \partial _ { k } f .\tag{18}
$$

Divergence form of the symmetric limit. It remains to write Equation (18) in divergence form. This relies on the following identities between the coefficients of Theorem 1.

Lemma 33 (Divergence identities). For every $x \in \mathcal { M }$ , the kernel moments of Definition 21 satisfy

$$
s _ { 0 } = - { \frac { 1 } { 2 } } \mathrm { d i v } _ { B H } ( m _ { 1 } ) \qquad a n d \qquad \Lambda ^ { k } + 2 s _ { 1 } ^ { k } = - { \frac { 1 } { \sigma _ { \mathrm { B H } } } } \partial _ { j } \left( \sigma _ { \mathrm { B H } } m _ { 2 } ^ { j k } \right) ,
$$

where div $_ { B H } X = \sigma _ { \mathrm { B H } } ^ { - 1 } \partial _ { i } ( \sigma _ { \mathrm { B H } } X ^ { i } )$ . Equivalently, the operators ofTheorem 1 read

$$
\mathfrak { A } _ { 1 } f = m _ { 1 } ^ { i } \partial _ { i } f + \frac { 1 } { 2 } \mathrm { d i v } _ { B H } ( m _ { 1 } ) f , \qquad \mathfrak { A } _ { 2 } f = \frac { 1 } { \sigma _ { \mathrm { B H } } } \partial _ { j } \left( \sigma _ { \mathrm { B H } } m _ { 2 } ^ { j k } \partial _ { k } f \right) + \Upsilon f ,
$$

so that $\mathcal { A } _ { 1 }$ is skew-adjoint and $\boldsymbol { \mathcal { A } } _ { 2 }$ is self-adjoint on $L ^ { 2 } ( \mathcal { M } , \mathfrak { m } _ { \mathrm { B H } } )$

Proof. We work in a chart, write $\sigma = \sigma _ { \mathrm { B H } }$ , and $\begin{array} { r } { M _ { n } ( x ) = \int _ { T _ { \infty } , u } K ( \mathcal { F } ( x , v ) ) v ^ { \otimes n } \mathrm { d } \lambda _ { x } ( v ) } \end{array}$ , so that $m _ { n } = \sigma M _ { n }$ The proof relies on two classical facts about the geodesic spray coefficients $G ^ { k }$ of F(Proposition 26), whose geodesics satisfy $\ddot { \gamma } ^ { k } = - 2 G ^ { k } ( \gamma , \dot { \gamma } )$ . First, the S-curvature reads (Shen, 2001)

$$
S ( x , v ) = \frac { \partial G ^ { l } } { \partial v ^ { l } } ( x , v ) - v ^ { l } \partial _ { l } \log \sigma ( x ) .\tag{19}
$$

Second, $\mathcal { F }$ is constant along geodesics, so that differentiating $t \mapsto { \mathcal { F } } ( \gamma ( t ) , \dot { \gamma } ( t ) ) \ \mathrm { a t } \ t = 0$ gives

$$
v ^ { j } \frac { \partial \mathcal { F } } { \partial x ^ { j } } ( x , v ) = 2 G ^ { l } ( x , v ) \frac { \partial \mathcal { F } } { \partial v ^ { l } } ( x , v ) .\tag{20}
$$

Let $P$ be a polynomial in v. An integration by parts in v, followed by Equation (20), gives

$$
\begin{array} { r l } { \displaystyle \int _ { T _ { x } , \mathcal { M } } K ( \mathcal { F } ) \frac { \partial G ^ { l } } { \partial v ^ { l } } P \mathrm { d } \lambda _ { x } ( v ) = - \displaystyle \int _ { T _ { x } , \mathcal { M } } K ^ { \prime } ( \mathcal { F } ) \frac { \partial \mathcal { F } } { \partial v ^ { l } } G ^ { l } P \mathrm { d } \lambda _ { x } ( v ) - \displaystyle \int _ { T _ { x } , \mathcal { M } } K ( \mathcal { F } ) G ^ { l } \frac { \partial P } { \partial v ^ { l } } \mathrm { d } \lambda _ { x } ( v ) } & { } \\ { = - \displaystyle \frac { 1 } { 2 } \int _ { T _ { x } , \mathcal { M } } K ^ { \prime } ( \mathcal { F } ) v ^ { j } \frac { \partial \mathcal { F } } { \partial x ^ { j } } P \mathrm { d } \lambda _ { x } ( v ) - \displaystyle \int _ { T _ { x } , \mathcal { M } } K ( \mathcal { F } ) G ^ { l } \frac { \partial P } { \partial v ^ { l } } \mathrm { d } \lambda _ { x } ( v ) } & { } \\ { = - \displaystyle \frac { 1 } { 2 } \partial _ { j } \displaystyle \int _ { T _ { x } , \mathcal { M } } K ( \mathcal { F } ) v ^ { j } P \mathrm { d } \lambda _ { x } ( v ) - \displaystyle \int _ { T _ { x } , \mathcal { M } } K ( \mathcal { F } ) G ^ { l } \frac { \partial P } { \partial v ^ { l } } \mathrm { d } \lambda _ { x } ( v ) . } \end{array}
$$

The boundary terms vanish, and the derivative can be taken out of the integral, since $K ( \mathcal { F } ( x , v ) )$ and $K ^ { \prime } ( \mathcal { F } ( x , v ) )$ ) decay exponentially in v by $( \mathbf { A } 1 )$ , while $G ^ { l }$ , its derivatives and $P$ grow polynomially. Moreover, in a chart, $\lambda _ { x }$ is the Lebesgue measure of $\mathbb { R } ^ { m }$ , which does not depend on x.

First identity. Taking $P = 1$ , we get $\begin{array} { r } { \int K ( \mathcal { F } ) \partial _ { v ^ { l } } G ^ { l } \mathrm { d } \lambda _ { x } = - \frac { 1 } { 2 } \partial _ { j } M _ { 1 } ^ { j } } \end{array}$ . Hence, by Equation (19),

$$
s _ { 0 } = \sigma \int _ { T _ { x } , \mathcal { M } } K ( \mathcal { F } ) S \mathrm { d } \lambda _ { x } ( v ) = - \frac { \sigma } { 2 } \partial _ { j } M _ { 1 } ^ { j } - \partial _ { j } \sigma M _ { 1 } ^ { j } = - \frac { 1 } { 2 \sigma } \partial _ { j } \left( \sigma ^ { 2 } M _ { 1 } ^ { j } \right) = - \frac { 1 } { 2 } \mathrm { d i v } _ { B H } ( m _ { 1 } ) .
$$

Second identity. Taking $P = v ^ { k }$ , we get $\begin{array} { r } { \int K ( \mathcal { F } ) \partial _ { v ^ { l } } G ^ { l } v ^ { k } \mathrm { d } \lambda _ { x } = - \frac { 1 } { 2 } \partial _ { j } M _ { 2 } ^ { j k } - \int K ( \mathcal { F } ) G ^ { k } \mathrm { d } \lambda _ { x } } \end{array}$ . Moreover, $\tilde { \Gamma } _ { i j } ^ { k } ( x , v ) v ^ { i } v ^ { j } = 2 G ^ { k } ( x , v )$ , so that $\begin{array} { r } { \Lambda ^ { k } = 2 \sigma \int K ( \mathcal { F } ) G ^ { k } \mathrm { d } \lambda _ { x } } \end{array}$ . Hence, by Equation (19),

$$
\begin{array} { l } { { \Lambda ^ { k } + 2 s _ { 1 } ^ { k } = 2 \sigma \displaystyle \int _ { T _ { x } . \mathcal { M } } K ( \mathcal { F } ) \left( G ^ { k } + \frac { \partial G ^ { l } } { \partial v ^ { l } } v ^ { k } \right) \mathrm { d } \lambda _ { x } ( v ) - 2 \partial _ { j } \sigma M _ { 2 } ^ { j k } } } \\ { { \qquad = - \sigma \partial _ { j } M _ { 2 } ^ { j k } - 2 \partial _ { j } \sigma M _ { 2 } ^ { j k } = - \displaystyle \frac { 1 } { \sigma } \partial _ { j } \left( \sigma ^ { 2 } M _ { 2 } ^ { j k } \right) = - \frac { 1 } { \sigma } \partial _ { j } \left( \sigma m _ { 2 } ^ { j k } \right) . } } \end{array}
$$

The expressions of $\mathcal { A } _ { 1 }$ and $\displaystyle \mathcal { A } _ { 2 }$ follow by substitution in Definition 21. Finally, for smooth $f$ and $h ,$ Stokes’ theorem $( \mathrm { L e e } , 2 0 1 2$ , Theorem 16.11) on the compact manifold without boundary M gives $\begin{array} { r } { \int _ { \mathcal { M } } \mathrm { d i v } _ { B H } ( f h m _ { 1 } ) \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } = 0 } \end{array}$ , that is

$$
\int _ { \mathcal { M } } \mathcal { A } _ { 1 } [ f ] h \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } = - \int _ { \mathcal { M } } f \mathcal { A } _ { 1 } [ h ] \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } ,
$$

and similarly $\begin{array} { r } { \int _ { \mathcal { M } } \mathrm { d i v } _ { B H } \big ( h m _ { 2 } \nabla f - f m _ { 2 } \nabla h \big ) \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } = 0 } \end{array}$ gives the self-adjointness of $\boldsymbol { \mathcal { A } } _ { 2 }$

Remark 34. Lemma 33 is the infinitesimal counterpart of the adjointness relation $\mathcal { G } _ { \mathcal { F } } ^ { \prime } = \mathcal { G } _ { \mathcal { F } } ^ { * } \ ( P r o p o -$ sition 19). Indeed, plugging the expansions of Theorem 1 and Proposition 32 into $\begin{array} { r } { \int _ { \mathcal M } ^ { \infty } \mathcal G _ { \mathcal F } [ f ] \dot { h } \mathrm { d } \mathfrak m _ { \mathrm { B H } } = } \end{array}$ $\begin{array} { r } { \int _ { \mathcal { M } } f \mathcal { G } _ { \mathcal { F } } ^ { \prime } [ h ] \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } } \end{array}$ , the terms of order $\varepsilon$ and $\varepsilon ^ { 2 }$ show that $\displaystyle \mathcal { A } _ { 1 }$ is skew-adjoint and $\boldsymbol { \mathcal { A } } _ { 2 }$ is self-adjoint on $\dot { L ^ { 2 } } ( \mathcal { M } , \mathfrak { m } _ { \mathrm { B H } } )$ , which gives back the two identities after an integration by parts.

Dividing the second identity by the constant $m _ { 0 }$ , the second-order and first-order terms of Equation (18) combine into a divergence,

$$
\frac { 1 } { 2 } \tilde { m } _ { 2 } ^ { i j } \partial _ { i } \partial _ { j } f - \frac { 1 } { 2 } \bigl ( \tilde { \Lambda } ^ { k } + 2 \tilde { s } _ { 1 } ^ { k } \bigr ) \partial _ { k } f = \frac { 1 } { 2 \sigma _ { \mathrm { B H } } } \partial _ { j } \left( \sigma _ { \mathrm { B H } } \tilde { m } _ { 2 } ^ { j k } \partial _ { k } f \right) = \frac { 1 } { 2 } \mathrm { d i v } _ { B H } \left( \tilde { m } _ { 2 } \nabla f \right) ,
$$

where $\tilde { m } _ { 2 } \nabla f$ denotes the vector field $\tilde { m } _ { 2 } ^ { j k } \partial _ { k } f \partial _ { j }$ . Moreover, writing $\omega = \rho ^ { 2 ( 1 - \theta ) }$ , the Leibniz rule gives $\scriptstyle { \frac { 1 } { 2 \omega } }$ div<sub>BH</sub> $\begin{array} { r } { ( \omega X ) = \frac 1 2 \operatorname { d i v } _ { B H } ( X ) + ( 1 - \theta ) X ^ { j } \partial _ { j } } \end{array}$ log ρ for any vector field X. Taking $X = \tilde { m } _ { 2 } \nabla f$ , and using $\tilde { m } _ { 2 } ^ { j k } = c g _ { \mathrm { B L } } ^ { j k }$ (Proposition 35), we get

$$
\mathscr { L } ^ { s } f = \frac { 1 } { 2 \omega } \mathrm { d i v } _ { B H } \left( \omega \tilde { m } _ { 2 } \nabla f \right) = \frac { c _ { 2 } } { \rho ^ { 2 \left( 1 - \theta \right) } } \mathrm { d i v } _ { B H } \left( \rho ^ { 2 \left( 1 - \theta \right) } \nabla _ { \mathrm { B L } } f \right) .
$$

Finally, since M is compact and without boundary (assumption (A2)), Stokes’ theorem (Lee, 2012, Theorem 16.11) gives $\begin{array} { r } { \int _ { \mathcal { M } } \mathrm { d i v } _ { B H } \mathbf { \bar { ( } } X ) \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } = 0 } \end{array}$ for any smooth vector field X. Applying it to $X = u \omega \nabla _ { \mathrm { B L } } f$ , for smooth $u , f : \mathcal { M }  \mathbb { R }$ , gives

$$
\int _ { \mathcal { M } } u \mathcal { L } ^ { s } [ f ] \omega \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } = - c _ { 2 } \int _ { \mathcal { M } } \langle \nabla u , \nabla f \rangle _ { \mathrm { B L } } \omega \mathrm { d } \mathfrak { m } _ { \mathrm { B H } } ,
$$

which is symmetric in u and $f \colon \mathcal { L } ^ { s }$ is self-adjoint on $L ^ { 2 } ( \mathcal { M } , \omega \mathfrak { m } _ { \mathrm { B H } } )$ . Its principal part is $c _ { 2 } g _ { \mathrm { B L } } ^ { i j } \partial _ { i } \partial _ { j }$ , with $g _ { \mathrm { B L } } ^ { i j }$ positive definite, so that it is elliptic. This concludes the proof of Theorem 2.

## D.4 LINKS BETWEEN THE MOMENTS OF THE KERNEL AND THE UNIT BALL

We recall that the moments of the kernel are defined as (Definition 21):

$$
m _ { n } ( x ) = \sigma _ { \mathrm { B H } } ( x ) \int _ { T _ { x } , \mathcal { M } } K ( \mathcal { F } ( x , v ) ) v ^ { \otimes n } \mathrm { d } \lambda _ { x } ( v ) ,
$$

We denote the normalized moments of the kernel as $\tilde { m } _ { n } = m _ { n } / m _ { 0 }$ . We define the unit tangent ball and its indicatrix as:

$$
\mathcal { B } _ { x } = \{ v \in T _ { x } \mathcal { M } : \mathcal { F } ( x , v ) \leq 1 \} , \quad \mathcal { I } _ { x } = \{ v \in T _ { x } \mathcal { M } : \mathcal { F } ( x , v ) = 1 \} .
$$

We also introduce the volume, the centroid, and the raw second moment of the unit ball read

$$
\lambda _ { x } ( \mathcal { B } _ { x } ) = \int _ { \mathbb { B } _ { x } } \mathrm { d } \lambda _ { x } ( v ) , \quad \mathbf { c } ^ { i } ( x ) = \frac { 1 } { \lambda _ { x } ( \mathcal { B } _ { x } ) } \int _ { \mathbb { B } _ { x } } v ^ { i } \mathrm { d } \lambda _ { x } ( v ) , \quad \mathrm { a n d } \quad \mathbf { S } ^ { i j } ( x ) = \frac { 1 } { \lambda _ { x } ( \mathcal { B } _ { x } ) } \int _ { \mathbb { B } _ { x } } v ^ { i } v ^ { j } \mathrm { d } \lambda _ { x } ( v ) .
$$

Proposition 35. We can relate the moments ofthe kernel to the unit ball asfollows:

$$
\begin{array} { l l l } { { m _ { 0 } ( x ) = m \mu _ { 0 } \sigma _ { \mathrm { B H } } ( x ) \lambda _ { x } ( \mathcal { B } _ { x } ) = m \mu _ { 0 } \mathrm { V o l } _ { \mathrm { E u c } } ( \mathbb { B } ^ { m } ) , } } \\ { { \displaystyle \tilde { m } _ { 1 } ^ { i } ( x ) = \frac { ( m + 1 ) \mu _ { 1 } } { m \mu _ { 0 } } \mathbf { c } ^ { i } ( x ) , } } \\ { { \displaystyle \tilde { m } _ { 2 } ^ { i j } ( x ) = \frac { ( m + 2 ) \mu _ { 2 } } { m \mu _ { 0 } } \mathbf { S } ^ { i j } ( x ) , } } \end{array}
$$

where $\begin{array} { r } { \mu _ { n } = \int _ { 0 } ^ { \infty } K ( r ) r ^ { m + n - 1 } \mathrm { { c } } } \end{array}$ r are the raw moments ofthe radial kernel K.

Proof. Since $\mathcal { F }$ is positively homogeneous of degree 1 in the second variable, we use a polar change of variable $v = r u$ with $r \in \mathbb { R } _ { + }$ and $u \in \mathcal { I } _ { x }$ . Under this change of variable, we have that $\mathrm { d } \lambda _ { x } ( v ) \bar { = } r ^ { m - 1 } \mathrm { d } \bar { r } \mathrm { \dot { d } } \omega _ { x } ,$ where $\omega _ { x } ( A ) = m \lambda _ { x } ( \{ r u : u \in A , 0 \leq r \leq 1 \} )$ ) for $A \subset { \mathcal { I } } _ { x }$ . With this change of variable, the volume, centroids, and raw second moment of the unit ball as:

$$
\lambda _ { x } ( \mathcal { B } _ { x } ) = \int _ { \mathcal { I } _ { x } } \int _ { 0 } ^ { 1 } r ^ { m - 1 } \mathrm { d } r \mathrm { d } \omega _ { x } = \frac { 1 } { m } \int _ { \mathcal { I } _ { x } } \mathrm { d } \omega _ { x } ,\tag{21}
$$

$$
\mathbf { c } ^ { i } ( x ) = \frac { 1 } { \lambda _ { x } ( \mathcal { B } _ { x } ) } \int _ { \mathcal { I } _ { x } } \int _ { 0 } ^ { 1 } ( r u ) ^ { i } r ^ { m - 1 } \mathrm { d } r \mathrm { d } \omega _ { x } = \frac { 1 } { ( m + 1 ) \lambda _ { x } ( \mathcal { B } _ { x } ) } \int _ { \mathcal { I } _ { x } } u ^ { i } \mathrm { d } \omega _ { x } ,\tag{22}
$$

$$
\mathbf { S } ^ { i j } ( x ) = \frac { 1 } { \lambda _ { x } ( \mathcal { B } _ { x } ) } \int _ { \mathcal { I } _ { x } } \int _ { 0 } ^ { 1 } ( r u ) ^ { i } ( r u ) ^ { j } r ^ { m - 1 } \mathrm { d } r \mathrm { d } \omega _ { x } = \frac { 1 } { ( m + 2 ) \lambda _ { x } ( \mathcal { B } _ { x } ) } \int _ { \mathcal { I } _ { x } } u ^ { i } u ^ { j } \mathrm { d } \omega _ { x } .\tag{23}
$$

We now turn to the moments $m _ { n } ( x )$ of the kernel. Since $\sigma _ { \mathrm { B H } } ( x )$ does not depend on v, we compute them up to this factor. For the zeroth moment,

$$
\begin{array} { l } { \displaystyle \frac { m _ { 0 } ( { \boldsymbol { x } } ) } { \sigma _ { \mathrm { B H } } ( { \boldsymbol { x } } ) } = \int _ { T _ { x } \mathcal { M } } K ( \mathcal { F } ( { \boldsymbol { x } } , { \boldsymbol { v } } ) ) \mathrm { d } \lambda _ { x } ( { \boldsymbol { v } } ) } \\ { \displaystyle = \int _ { \mathcal { I } _ { x } } \int _ { 0 } ^ { \infty } K ( { \boldsymbol { r } } ) r ^ { m - 1 } \mathrm { d } { \boldsymbol { r } } \mathrm { d } \omega _ { x } } \\ { \displaystyle = \mu _ { 0 } \int _ { \mathcal { I } _ { x } } \mathrm { d } \omega _ { x } } \\ { \displaystyle = m _ { \mu _ { 0 } } \lambda _ { x } ( \mathcal { B } _ { x } ) } \end{array}
$$

$$
\begin{array} { r } { \mathrm { B y ~ t h e ~ c h a n g e ~ o f ~ v a r i a b l e ~ } v = r u } \\ { \mathrm { B y ~ F u b i n i ' s ~ t h e o r e m } } \\ { \mathrm { B y ~ E q u a t i o n ~ } ( 2 1 ) . } \end{array}
$$

Similarly, the first moment is given by:

$$
\begin{array} { r l r } { \displaystyle \frac { m _ { 1 } ^ { i } ( x ) } { \sigma _ { \mathrm { B H } } ( x ) } = \int _ { T _ { x } , \mathcal { M } } K ( \mathcal { F } ( x , v ) ) v ^ { i } \mathrm { d } \lambda _ { x } ( v ) } \\ { = \int _ { \mathcal { I } _ { x } } \displaystyle \int _ { 0 } ^ { \infty } K ( r ) ( r u ) ^ { i } r ^ { m - 1 } \mathrm { d } r \mathrm { d } \omega _ { x } } & { \qquad } & { \mathrm { B y ~ t h e ~ c h a n g e ~ o f ~ v a r i a b l e ~ } v = r u } \\ { = \mu _ { 1 } \displaystyle \int _ { \mathcal { I } _ { x } } u ^ { i } \mathrm { d } \omega _ { x } } & { \qquad } & { \mathrm { B y ~ F u b i n i ' s ~ t h e o r e m } } \\ { = ( m + 1 ) \mu _ { 1 } \lambda _ { x } ( \mathcal { B } _ { x } ) \mathbf { c } ^ { i } ( x ) } & { \qquad } & { \mathrm { B y ~ E q u a t i o n ~ } ( 2 2 ) . } \end{array}
$$

The second moment is given by:

$$
{ \begin{array} { r l r } { { \frac { m _ { 2 } ^ { i j } ( x ) } { \sigma _ { \mathrm { B H } } ( x ) } } = { \displaystyle \int _ { T _ { x } , \mathcal { M } } K ( \mathcal { F } ( x , v ) ) v ^ { i } v ^ { j } \mathrm { d } \lambda _ { x } ( v ) } } \\ { = { \displaystyle \int _ { \mathcal { I } _ { x } } \int _ { 0 } ^ { \infty } K ( r ) ( r u ) ^ { i } ( r u ) ^ { j } r ^ { m - 1 } \mathrm { d } r \mathrm { d } \omega _ { x } } \qquad } & { \mathrm { B y ~ t h e ~ c h a n g e ~ o f ~ v a r i a b l e ~ } v = r u } \\ { = \mu _ { 2 } \displaystyle \int _ { \mathcal { I } _ { x } } u ^ { i } u ^ { j } \mathrm { d } \omega _ { x } } & { \mathrm { B y ~ F u b i n i ' s ~ t h e o r e m } } \\ { = ( m + 2 ) \mu _ { 2 } \lambda _ { x } ( \mathcal { B } _ { x } ) \mathbf { S } ^ { i j } ( x ) } & { \mathrm { B y ~ E q u a t i o n ~ } ( 2 3 ) . } \end{array} }
$$

Multiplying by $\sigma _ { \mathrm { B H } } ( x )$ , Equation (11) gives $m _ { 0 } ( x ) = m \mu _ { 0 } \sigma _ { \mathrm { B H } } ( x ) \lambda _ { x } ( \mathcal { B } _ { x } ) = m \mu _ { 0 } \mathrm { V o l } _ { \mathrm { E u c } } ( \mathbb { B } ^ { m } )$ . Dividing the expression of $m _ { n }$ by $m _ { 0 }$ , in which $\sigma _ { \mathrm { B H } } ( x )$ cancels, yields the desired result for the normalized moments $\tilde { m } _ { n } = m _ { n } / m _ { 0 }$

## IMPLICATIONS FOR THE LIMIT OPERATORS

Zeroth moment. The zeroth moment $m _ { 0 } ( x )$ of the kernel appears in two ways: as the normalization factor in the definitions of $\tilde { m } , \tilde { \Lambda } ^ { k }$ and $\tilde { s } _ { 1 } ^ { k }$ , which is discussed at the end of this subsection, and as the leading coefficient of the expansion of Theorem 1. By Proposition 35, $m _ { 0 } = m \mu _ { 0 } \mathrm { V o l } _ { \mathrm { E u c } } ( \mathbb { B } ^ { m } )$ does not depend on x: the Busemann–Hausdorff measure is precisely the one for which small forward balls have, at leading order, the volume of Euclidean balls. As a consequence, $q _ { \varepsilon } = m _ { 0 } \rho + O ( \varepsilon ^ { 2 } )$ carries no geometric factor, and $\partial _ { i } \log \phi ^ { ( \theta ) } = ( 1 - \theta ) \partial _ { i } \log \rho + O ( \varepsilon ^ { 2 } )$ (Section D.3): the θ-normalization acts exactly as in the Riemannian setting, and $\dot { \theta } = 1$ removes the sampling density from the limit operator $\mathcal { L } ^ { s }$

First moment. By Proposition 35, the normalized first moment $\tilde { m } _ { 1 } ^ { i } ( x )$ is proportional to the centroid $\mathbf { c } ^ { i } ( x )$ of the unit ball ${ \mathcal { B } } _ { x }$ . If the Finsler metric $\mathcal { F }$ is reversible, the unit ball ${ \mathcal { B } } _ { x }$ is symmetric with respect to the origin, so that $\mathbf { c } ^ { i } ( x ) = 0$ . Hence, for reversible Finsler metrics, $\tilde { m } _ { 1 } ^ { i } = 0$ and the anti-symmetric limit operator ${ \mathcal { L } } ^ { a }$ vanishes<sup>6</sup>. When Fis not reversible, the unit ball is in general not symmetric, and its centroid defines a vector field c on $\mathcal { M }$ which measures the asymmetry of F at first order. Note that $\mathbf { c } ( x ) = 0$ does not imply that $\mathcal { F }$ is reversible at $x ,$ since a non-symmetric unit ball can have its centroid at the origin. For Randers metrics, the centroid is given explicitly in terms of b by Proposition $4 3 .$ This vector field is exactly what the anti-symmetric part of the graph Laplacian recovers: by Theorem $^ { 2 , }$

$$
\mathcal { L } ^ { a } f ( x ) = \tilde { m } _ { 1 } ^ { i } ( x ) \partial _ { i } f ( x ) = \frac { ( m + 1 ) \mu _ { 1 } } { m \mu _ { 0 } } \mathbf { c } ^ { i } ( x ) \partial _ { i } f ( x ) ,
$$

so that ${ \mathcal { L } } ^ { a }$ is the derivative along the centroid vector field, up to a constant that only depends on the kernel.

Second moment. By Proposition 35 and Equation (10), the normalized second moment is proportional to the raw second moment $\mathbf { S } ^ { i j } ( \boldsymbol { x } )$ of the unit ball, and hence to the Binet–Legendre metric $g _ { \mathrm { B I } }$ (Definition 16):

$$
\tilde { m } _ { 2 } ^ { i j } ( x ) = \frac { ( m + 2 ) \mu _ { 2 } } { m \mu _ { 0 } } { \bf S } ^ { i j } ( x ) = \frac { \mu _ { 2 } } { m \mu _ { 0 } } g _ { \mathrm { B L } } ^ { i j } ( x ) .
$$

Hence, up to the constant $2 c _ { 2 } = \mu _ { 2 } / ( m \mu _ { 0 } )$ of Theorem 2, the normalized second moment coincides with the Binet–Legendre metric, which is a Riemannian metric capturing the local geometry of $( { \mathcal { M } } , { \mathcal { F } } )$ . It depends on the kernel only through this constant, and does not depend on the bandwidth $\varepsilon .$

Normalization by $m _ { 0 } .$ . Dividing by $m _ { 0 }$ is what makes $\tilde { m } _ { 1 }$ and $\tilde { m } _ { 2 }$ the natural geometric objects governing the limit operators. Indeed, by Proposition 35, the raw moments $m _ { 1 }$ and $m _ { 2 }$ carry the factor $\sigma _ { \mathrm { B H } } ( x ) \lambda _ { x } ( \mathcal { B } _ { x } )$ as $m _ { 0 }$ does, which cancels in the ratio. The normalized moments thus only involve the centroid c and the raw second moment S, which are averages over the unit ball, and the kernel only enters through the constants $\mu _ { n } / \mu _ { 0 }$ . This is also why the Binet–Legendre metric, which is itself normalized by $\lambda _ { x } ( \mathcal { B } _ { x } )$ in Equation (10), appears in the limit.

## D.5 PROOF OF COROLLARY 3

We start from the divergence form of Theorem 2. Let $\sqrt { \operatorname* { d e t } g _ { \mathrm { B L } } }$ denote the density, in a chart, of the Riemannian volume vol of $g _ { \mathrm { B L } }$ , and let

$$
w = \frac { \sigma _ { \mathrm { B H } } } { \sqrt { \operatorname* { d e t } g _ { \mathrm { B L } } } } = \frac { \mathrm { V o l } _ { \mathrm { E u c } } ( \mathbb { B } ^ { m } ) } { \mathrm { v o l } _ { \mathrm { B L } } ( \mathcal { B } _ { x } ) }
$$

be the density of m<sub>BH</sub> with respect to vol<sub>BL</sub>, where we used Equation (11), and $\mathrm { v o l } _ { \mathrm { B L } } ( \mathcal { B } _ { x } ) \ =$ $\sqrt { \operatorname* { d e t } g _ { \mathrm { B L } } ( x ) } \lambda _ { x } ( \mathcal { B } _ { x } )$ is the volume of the unit ball of F for the Binet–Legendre metric. In particular, w does not depend on the chart. Since $\sigma _ { \mathrm { B H } } = w \sqrt { \operatorname* { d e t } g _ { \mathrm { B I } } }$ , we have div $\ O _ { B H } X = \mathrm { d i v _ { B L } } X + X ^ { i } \partial _ { i }$ log w for any vector field X, where div is the Riemannian divergence of $g _ { \mathrm { B I } }$ . Applying it in Theorem 2, and using div $\mathsf { \tilde { \Pi } } _ { \mathrm { B L } } ( \nabla _ { \mathrm { B L } } f ) = \Delta _ { \mathrm { B L } } f$ , we get, for any Finsler metric,

$$
\begin{array} { r } { \mathscr { L } ^ { s } f = c _ { 2 } \left( \Delta _ { \mathrm { B L } } f + 2 ( 1 - \theta ) \langle \nabla \log \rho , \nabla f \rangle _ { \mathrm { B L } } + \langle \nabla \log w , \nabla f \rangle _ { \mathrm { B L } } \right) . } \end{array}\tag{24}
$$

The discrepancy with a weighted Laplace–Beltrami operator of $g _ { \mathrm { B I } }$ is thus a drift along the gradient of log w, which measures the variations of the Binet–Legendre volume of the unit ball of $\mathcal { F }$ across M.

Berwald metrics. When $\mathcal { F }$ is Berwald, the parallel transport along any curve from x to y is a linear map $P : T _ { x } { \mathcal { M } }  T _ { y } { \mathcal { M } }$ which preserves $\mathcal { F }$ (Bao et al., 2012), so that $P ( \mathcal { B } _ { x } ) = \mathcal { B } _ { y }$ By Equation (10), the Binet–Legendre metric is built from the unit ball alone, and the change of variable $v = P u$ gives $\begin{array} { r l } { { g } _ { \mathrm { B L } } ^ { - 1 } ( y ) = } & { { } } \end{array}$ $P g _ { \mathrm { B L } } ^ { - 1 } ( x ) P ^ { \top }$ , the Jacobian of $P$ being cancelled by the normalization by the volume of the ball. Hence $P$ is an isometry from $( T _ { x } \mathcal { M } , g _ { \mathrm { B L } } ( x ) \bar { ) }$ to $( T _ { y } \mathcal { M } , g _ { \mathrm { B L } } ( y ) )$ which maps ${ \mathcal { B } } _ { x }$ onto ${ \mathcal { B } } _ { y } ,$ so that $\mathrm { v o l } _ { \mathrm { B L } } ( \mathcal { B } _ { x } ) =$ vol<sub>BL</sub> $( \mathcal { B } _ { y } )$ . Since M is connected, w is constant, and Equation (24) reduces to Equation (7).

Riemannian metrics. Let $\mathcal { F } ( x , v ) = \sqrt { v ^ { \mathsf { T } } \mathbf { A } ( x ) v }$ . Its fundamental tensor is $g _ { i j } ( x , v ) = \mathbf { A } _ { i j } ( x )$ for every $v ,$ hence does not depend on the direction, so F is Berwald. Its unit bal $\mathcal { B } _ { x } = \{ v : v ^ { \mathsf { T } } \mathbf { A } ( x ) v \leq 1 \}$ is the ellipsoid defined by $\mathbf { A } ( x )$ , whose second moment is given by the classical identity $\begin{array} { r l } { \int _ { \mathcal { B } _ { x } } v ^ { i } v ^ { j } \mathrm { d } \lambda _ { x } ( \acute { v } ) = } \end{array}$ $\frac { \lambda _ { x } ( \mathcal { B } _ { x } ) } { m + 2 } \mathbf { A } ^ { i j } ( x )$ . Substituting it into Equation (10) gives $g _ { \mathrm { B L } } ^ { i j } = \mathbf { A } ^ { i j } , \mathrm { i . e . } \ g _ { \mathrm { B L } } = \mathbf { A }$ . Equation (7) then reads

$$
\begin{array} { r } { \mathcal { L } ^ { s } f = c _ { 2 } \left( \Delta _ { \mathbf { A } } f + 2 ( 1 - \theta ) \langle \nabla \log \rho , \nabla f \rangle _ { \mathbf { A } } \right) , } \end{array}
$$

which is the θ-normalized diffusion generator of Coifman & Lafon (2006). Note also that the S-curvature of a Riemannian metric vanishes, so that $s _ { 0 } = s _ { 1 } = 0$ and, in this setting, Υ reduces to

$$
\Upsilon ( x ) = - \frac { 1 } { 3 } \sigma _ { \mathrm { B H } } ( x ) \int _ { T _ { x } \mathcal { M } } K ( \mathcal { F } ( x , v ) ) \mathrm { R i c } ^ { \mathcal { F } } ( x , v ) \mathrm { d } \lambda _ { x } ( v ) .
$$

Finally, a Riemannian metric is reversible, so v 7→ $K ( \mathcal { F } ( x , v ) )$ is an even function on $T _ { x } { \mathcal { M } }$ , and the change of variable $v \mapsto - v { \mathrm { ~ g i v e s ~ } } m _ { 1 } = 0$ . By Theorem $2 , \mathcal { L } ^ { a } f = \tilde { m } _ { 1 } ^ { i } \partial _ { i } f = 0$

## D.6 PROOF OF THEOREM 4

This subsection is devoted to the proof of Theorem 4. We first give the matrix forms of the discrete operators, then we prove the convergence of the empirical operators to their population counterparts, and finally we prove the convergence of the population operators to their limit.

## MATRIX FORMS

We start by recalling the basic definitions of the discrete operators. The matrices of Section 4 are all built from the empirical kernel matrix ${ \bf W } _ { N }$ , and the two members of the family $\mathbf { P } ^ { ( \theta , \bullet ) }$ take a slightly different shape. For $\bullet = s .$ , the matrix $\mathbf { D } _ { N } ^ { ( \theta , s ) }$ is by construction the degree matrix of $\mathbf { W } _ { N } ^ { ( \theta , s ) }$ , so that the two terms merge into

$$
\mathbf { P } ^ { ( \theta , s ) } [ f ] = \Big ( \big ( \mathbf { D } _ { N } ^ { ( \theta , s ) } \big ) ^ { - 1 } \mathbf { W } _ { N } ^ { ( \theta , s ) } - \mathbf { I } \Big ) \mathbf { f } ,
$$

a normalized-graph-Laplacian-type matrix, whose eigenvectors provide the embedding coordinates (Section 5.3). For $\bullet = a .$ , the matrix $\mathbf { D } _ { N } ^ { ( \theta , a ) }$ is not the degree matrix of $\mathbf { W } _ { N } ^ { ( \theta , s ) }$ , and the two terms remain separate:

$$
\mathbf { P } ^ { ( \theta , a ) } [ f ] = \big ( \mathbf { D } _ { N } ^ { ( \theta , s ) } \big ) ^ { - 1 } \mathbf { W } _ { N } ^ { ( \theta , a ) } \mathbf { f } - \big ( \mathbf { D } _ { N } ^ { ( \theta , s ) } \big ) ^ { - 1 } \mathbf { D } _ { N } ^ { ( \theta , a ) } \mathbf { f }
$$

is the estimator of the advection term, whose limit is $\varepsilon \tilde { m } _ { 1 } ^ { i } \partial _ { i } f .$

## FROM FINITE SAMPLES TO THE LIMIT OPERATOR: SETTING AND PROOF SKETCH

We now let $N \to \infty$ and $\varepsilon = \varepsilon ( N ) \to 0$ jointly. Throughout, $C$ denotes a constant depending on ${ \mathcal { M } } , { \mathcal { F } } , K$ $\rho , \theta$ and $f ,$ , but never on N or ε, and possibly changing from line to line.

It is convenient to extend the discrete objects to the whole manifold, so that they can be compared pointwise with their continuous counterparts. For $x , y \in { \mathcal { M } }$ , we set

$$
\begin{array} { l } { { \displaystyle \hat { d } _ { N } ( x ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } W ( x , X _ { j } ) , \qquad \hat { d } _ { N } ^ { \prime } ( x ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } W ( X _ { j } , x ) , \qquad \hat { q } _ { N } = \frac { \hat { d } _ { N } + \hat { d } _ { N } ^ { \prime } } { 2 } , } } \\ { { \displaystyle \hat { W } _ { N } ^ { ( \theta ) } ( x , y ) = \frac { W ( x , y ) } { \hat { q } _ { N } ( x ) ^ { \theta } \hat { q } _ { N } ( y ) ^ { \theta } } , \qquad \hat { W } _ { N } ^ { ( \theta , \bullet ) } ( x , y ) = \frac { \hat { W } _ { N } ^ { ( \theta ) } ( x , y ) \pm \hat { W } _ { N } ^ { ( \theta ) } ( y , x ) } { 2 } , } } \\ { { \displaystyle \hat { d } _ { N } ^ { ( \theta , \bullet ) } ( x ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \hat { W } _ { N } ^ { ( \theta , \bullet ) } ( x , X _ { j } ) , } } \end{array}
$$

with the sign + for $\bullet = s$ and $- \operatorname { f o r } \bullet = a$ , and finally

$$
\hat { \bf P } _ { N } ^ { ( \theta , \bullet ) } [ f ] ( x ) = \frac { 1 } { \hat { d } _ { N } ^ { ( \theta , s ) } ( x ) } \left( \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \hat { W } _ { N } ^ { ( \theta , \bullet ) } ( x , X _ { j } ) f ( X _ { j } ) - f ( x ) \hat { d } _ { N } ^ { ( \theta , \bullet ) } ( x ) \right) .\tag{25}
$$

We also write $\hat { \mathbf { L } } _ { N } ^ { s } = \varepsilon ^ { - 2 } \hat { \mathbf { P } } _ { N } ^ { ( \theta , s ) }$ and $\hat { \mathbf { L } } _ { N } ^ { a } = \varepsilon ^ { - 1 } \hat { \mathbf { P } } _ { N } ^ { ( \theta , a ) }$ for the rescaled versions, extending $\mathbf { L } ^ { s }$ and ${ \bf L } ^ { a }$ in the same way. Evaluated at the sample points $x = \ddot { X _ { i } }$ , these are exactly the matrices of Section 4, the factors $1 / N$ cancelling in the ratio Equation (25), so that ${ \hat { \mathbf { P } } } _ { N } ^ { ( \theta , \bullet ) } [ f ] ( X _ { i } ) = \left( \mathbf { P } ^ { ( \theta , \bullet ) } \mathbf { f } \right)$ and $\hat { \mathbf { L } } _ { N } ^ { \bullet } [ f ] ( X _ { i } ) = \left( \mathbf { L } ^ { \bullet } \mathbf { f } \right) _ { i }$ i Their continuous counterparts are $d , d ^ { \prime } , q _ { \varepsilon } , W ^ { ( \theta ) } , d ^ { ( \theta , s ) } = \mathcal { G } _ { \mathcal { F } } ^ { ( \theta , s ) } [ \rho ] , d ^ { ( \theta , a ) } = \mathcal { G } _ { \mathcal { F } } ^ { ( \theta , a ) } [ \rho ]$ and $\mathcal { P } ^ { ( \theta , \bullet ) }$

Two errors then separate the empirical operator from its limit, and we bound them one at a time. Writing the symmetric case for concreteness, and inserting the population operator $\mathcal { P } ^ { ( \theta , s ) }$ between the two, we have that:

$$
\big | \hat { \mathbf { L } } _ { N } ^ { s } [ f ] - \mathcal { L } ^ { s } f \big | \leq \underbrace { \frac { 1 } { \mathcal { E } ^ { 2 } } \big | \hat { \mathbf { P } } _ { N } ^ { ( \theta , s ) } [ f ] - \mathcal { P } ^ { ( \theta , s ) } [ f ] \big | } _ { \mathrm { f u c t u a t i o n } } + \underbrace { \frac { 1 } { \mathcal { E } ^ { 2 } } \big | \mathcal { P } ^ { ( \theta , s ) } [ f ] - \varepsilon ^ { 2 } \mathcal { L } ^ { s } f \big | } _ { \mathrm { b i a s } } .\tag{26}
$$

The bias is deterministic: it is the gap between the population operator and its limit, and it decreases with $\varepsilon .$ The fluctuation is stochastic: it is the gap between the empirical operator and its population counterpart, and it grows as ε decreases, since the kernel then concentrates and fewer samples contribute to each evaluation. The next two subsections bound each of them, and the last one combines the two and reads off the admissible scalings of $\varepsilon ( N )$

## BOUNDING THE BIAS

The deterministic part of the error is the expansion already established in Appendix D.3, read quantitatively rather than as a limit.

Lemma 36 (Bias). Let $f \in \mathcal { C } ^ { 3 } ( \mathcal { M } )$ . There exist $C , \varepsilon _ { 0 } > 0$ such that $f o r \varepsilon \leq \varepsilon _ { 0 }$

$$
\operatorname* { s u p } _ { x \in \mathcal { M } } \left. \mathcal { P } ^ { ( \theta , s ) } [ f ] ( x ) - \varepsilon ^ { 2 } \mathcal { L } ^ { s } f ( x ) \right. \leq C \varepsilon ^ { 3 } , \qquad \operatorname* { s u p } _ { x \in \mathcal { M } } \left. \mathcal { P } ^ { ( \theta , a ) } [ f ] ( x ) - \varepsilon \mathcal { L } ^ { a } f ( x ) \right. \leq C \varepsilon ^ { 2 } .
$$

Proof. The proof of Theorem 2 in Appendix D.3 establishes these expansions with an explicit remainder, inherited from the $\mathbb { O } ( \varepsilon ^ { 3 } )$ of Theorem 1. The remainder is uniform in $x ,$ since M is compact and the derivatives of $f$ are bounded. □

## BOUNDING THE FLUCTUATION

We now bound the stochastic part of the error, which is the deviation of the empirical operator from its population counterpart. The next proposition gives a uniform bound on this deviation, which is the main technical result of this section.

Proposition 37 (Fluctuation). Assume $\varepsilon = \varepsilon ( N ) \to 0$ with $N \varepsilon ^ { m } / \log N \to \infty$ , and let $f \in \mathcal { C } ^ { 1 } ( \mathcal { M } )$ . Then there is a constant $C$ such that, almost surely for N large enough,

$$
\operatorname* { s u p } _ { x \in \mathcal { M } } \left. \hat { \mathbf { P } } _ { N } ^ { ( \theta , \bullet ) } [ f ] ( x ) - \mathcal { P } ^ { ( \theta , \bullet ) } [ f ] ( x ) \right. \le C \varepsilon \zeta _ { N } , \qquad \bullet \in \{ s , a \} ,
$$

where

$$
\zeta _ { N } = \sqrt { \frac { \log N } { N \varepsilon ^ { m } } } .\tag{27}
$$

To prove this bound, we rely on standard concentration inequalities for sums of independent random variables, which give a bound on the deviation of the empirical average from its expectation. To do so, we use a δ-net argument<sup>7</sup>: we first control the deviation on a finite set of points, then extend it to the whole manifold by continuity, and conclude almost surely with the Borel–Cantelli lemma.

Proof of Proposition 37. Defining the net. The intrinsic distance dst is asymmetric, so that it yields two different notions of δ-net, depending on whether we use forward or backward balls. We therefore rely instead on the Euclidean distance $| \cdot | \operatorname { o f } \mathbb { R } ^ { \breve { D } }$ restricted to the tangent spaces of M, which is symmetric and induces a natural Riemannian distance dist on M. In particular, the distances dst and dist are comparable: the map $( x , v ) \mapsto \mathcal { F } ( x , v )$ is continuous and positive on the unit sphere bundle $\{ ( x , v ) \in T \mathcal { M } : | v | = \mathrm { \bar { 1 } } \}$ , which is compact by (A2), so that it is bounded there between two constants. By positive homogeneity, we get that

$$
\kappa _ { - } | v | \leq \operatorname { \mathcal { F } } ( x , v ) \leq \kappa _ { + } | v | , \qquad ( x , v ) \in T { \mathcal { M } } , \qquad \mathrm { f o r ~ s o m e ~ } 0 < \kappa _ { - } \leq \kappa _ { + } < \infty .
$$

Hence, for any $x , y \in { \mathcal { M } } .$

$$
\kappa _ { - } \mathrm { d i s t } ( x , y ) \leq \mathrm { d s t } _ { \mathcal { F } } ( x , y ) \leq \kappa _ { + } \mathrm { d i s t } ( x , y ) .\tag{28}
$$

Once this is established, we can define a δ-net according to this Riemannian distance. Counting the number of points in such a net is standard, and we give the result below for completeness. In this section, Vol denotes the Riemannian volume of M associated with the Riemannian metric that induces dist.

Lemma 38. There are $C _ { \mathit { 4 } l } , \delta _ { 0 } > 0$ such that, for every $\delta \leq \delta _ { 0 } ,$ , M admits a δ-net ofcardinality $n _ { \delta } \le C _ { { \mathcal { M } } } \delta ^ { - m }$

Proofsketch. The input is the volume of small balls. On a compact Riemannian manifold, the volume density in normal coordinates around x is $1 + \mathcal { O } ( r ^ { 2 } )$ , with a remainder that is uniform in x since the curvature is bounded and the injectivity radius is bounded below, hence there are $v _ { - } , v _ { + } , \delta _ { 0 } > 0$ such that $v _ { - } r ^ { m } \leq$ $\mathrm { V o l } \big ( B ( x , r ) \big ) \leq v _ { + } \dot { r ^ { m } }$ for every $x \in \mathcal { M }$ and every $r \leq \delta _ { 0 }$ . Let $x _ { 1 } , \ldots , x _ { n _ { \delta } }$ be a maximal δ-separated subset of M, which exists by compactness. It is a δ-net: any x at distance more than δ from all the $x _ { k }$ could be added to the family, contradicting maximality. Moreover the balls $B ( x _ { k } , \delta / 2 )$ are pairwise disjoint, by δ-separation, so that summing their volumes gives $n _ { \delta } v _ { - } ( \delta / 2 ) ^ { m } \leq \mathrm { V o l } ( \mathcal { M } )$ , which is the announced bound. The same packing argument, in the Euclidean setting, is detailed in Vershynin (2019, Section 4.2). □

Using this δ-net, the supremum over M of a function $g$ reduces to a maximum over the net points: picking for each x a net point $x _ { k }$ with dist ${ \bf \Phi } ( x , x _ { k } ) \le \delta$ , and denoting by $L _ { g }$ a Lipschitz constant of $g ,$

$$
\operatorname* { s u p } _ { x \in \mathcal { M } } | g ( x ) | \leq \operatorname* { m a x } _ { 1 \leq k \leq n _ { \delta } } | g ( x _ { k } ) | + L _ { g } \delta .
$$

Deterministic estimates on the kernel. Besides the net, we need three elementary properties of the kernel W: it is of size $\varepsilon ^ { - m }$ , it is concentrated at scale ε around the diagonal, and it varies at scale ε. We write $W ^ { \bullet } ( x , y ) = \bigl ( W ( x , y ) \pm W ( y , x ) \bigr ) / 2$ for its symmetric $( \bullet = s )$ and antisymmetric $( \bullet = a )$ parts, so that $| W ^ { \bullet } | \leq W ^ { s }$

Lemma 39 (Kernel at scale $\varepsilon )$ . There is a constant C such that, for every $\varepsilon \leq 1$ , every $x , x ^ { \prime } , y \in \mathcal { M }$ and every $p \in \{ 0 , 1 \}$ ,

$$
W ^ { s } ( x , y ) \leq \frac { K ( 0 ) } { \varepsilon ^ { m } } , \qquad W ^ { s } ( x , y ) \mathrm { d i s t } ( x , y ) \leq \frac { C } { \varepsilon ^ { m - 1 } } , \qquad \int _ { \mathcal { M } } W ^ { s } ( x , y ) \mathrm { d i s t } ( x , y ) ^ { p } \mathrm { d } \mathbb { P } ( y ) \leq C \varepsilon ^ { p } ,\tag{29}
$$

and, $f o r \bullet \in \{ s , a \}$

$$
\left| W ^ { \bullet } ( x , y ) - W ^ { \bullet } ( x ^ { \prime } , y ) \right| \leq \frac { C \mathrm { d i s t } ( x , x ^ { \prime } ) } { \varepsilon ^ { m + 1 } } .\tag{30}
$$

Proof. Since K is decreasing and $\begin{array} { r } { \mathbf { d s t } _ { \mathcal { F } } \geq 0 , } \end{array}$ , both $W ( x , y )$ and $W ( y , x )$ are bounded by $K ( 0 ) \varepsilon ^ { - m }$ , which gives the first bound. For the second one, the distance comparison inequality (Equation (28)) gives dst $\begin{array} { r } { \mathcal { F } ( x , y ) \ge \kappa _ { - } \mathrm { d i s t } ( x , y ) } \end{array}$ and $\mathrm { d s t } _ { \mathcal { F } } ( y , x ) \geq \kappa _ { - } \mathrm { d i s t } ( x , y )$ , so that $W ^ { s } ( x , y ) \leq \bar { \varepsilon } ^ { - m } \bar { K } \big ( \kappa _ { - } \mathrm { d i s t } ( x , y \bar { ) } / \varepsilon \big )$ Writing $\boldsymbol { r } = \boldsymbol { \kappa } _ { - } \mathrm { d i s t } ( x , y ) / \varepsilon$ , it holds that

$$
W ^ { s } ( x , y ) \mathrm { d i s t } ( x , y ) \leq \frac { \varepsilon ^ { 1 - m } } { \kappa _ { - } } r K ( r ) \leq \frac { \varepsilon ^ { 1 - m } } { \kappa _ { - } } \operatorname* { s u p } _ { r \geq 0 } r K ( r ) ,
$$

which is finite since K is sub-exponential $( \mathbf { A } 1 )$ . For the third one, we use that $\mathrm { V o l } \big ( B ( x , r ) \big ) \leq v _ { + } r ^ { m }$ for every $r > 0 \colon$ this is shown in the proof of Lemma 38 for $r \leq \delta _ { 0 } .$ , and it extends to $r > \delta _ { 0 }$ by enlarging $v _ { + }$ by $\mathrm { V o l } ( \mathcal { M } ) \delta _ { 0 } ^ { - m }$ . We split M into the annuli $A _ { k } = \{ y \in { \mathcal { M } } : k \varepsilon \leq \mathrm { d i s t } ( x , y ) < ( k + 1 ) \varepsilon \} , \ k \geq 0$ . On $A _ { k }$ , we have $W ^ { s } ( x , y ) \leq \varepsilon ^ { - m } K ( \kappa _ { - } k ) \leq C _ { K } \varepsilon ^ { - m } e ^ { - \nu _ { K } \kappa _ { - } k }$ and dis $\begin{array} { r } { \mathrm { ~ : ~ } ( x , y ) ^ { p } \leq \left( ( k + 1 ) \varepsilon \right) ^ { p } } \end{array}$ , and moreover $\mathrm { V o l } ( A _ { k } ) \leq v _ { + } { \big ( } ( k + 1 ) \varepsilon { \big ) } ^ { m }$ . Since $\mathrm { d } \mathbb { P } = \rho \mathrm { d } { \mathfrak { m } } _ { \mathrm { B H } }$ , where $\rho$ and the ratio of the densities of m and Vol are bounded on the compact manifold M, we have $\mathrm { d } \mathbb { P } \leq C d \mathrm { V o l }$ , and we get

$$
\int _ { \mathcal M } W ^ { s } ( x , y ) \mathrm { d i s t } ( x , y ) ^ { p } \mathrm { d } \mathbb P ( y ) \le \frac { C } { \varepsilon ^ { m } } \sum _ { k > 0 } e ^ { - \nu _ { K } \kappa _ { - } k } \big ( ( k + 1 ) \varepsilon \big ) ^ { m + p } = C \varepsilon ^ { p } \sum _ { k \ge 0 } ( k + 1 ) ^ { m + p } e ^ { - \nu _ { K } \kappa _ { - } k } ,
$$

and the series converges. Finally, for the Lipschitz bound, the triangle inequality for dst<sub>F</sub> gives dst<sub>F</sub> $( x , y ) \leq$ $\mathrm { d s t } _ { \mathcal { F } } ( \boldsymbol { x } , \boldsymbol { x } ^ { \prime } )$ +dst<sub>F</sub> $( x ^ { \prime } , y )$ and $\mathrm { d s t } _ { { \mathscr { F } } } ( x ^ { \prime } , y ) \leq \bar { \mathrm { d s t } } _ { { \mathscr { F } } } ( x ^ { \prime } , x ) + \mathrm { d s t } _ { { \mathscr { F } } } ( x , y )$ , so that, by Equation (28),

$$
\begin{array} { r } { \left| \mathrm { d s t } _ { \mathcal { F } } ( x , y ) - \mathrm { d s t } _ { \mathcal { F } } ( x ^ { \prime } , y ) \right| \leq \operatorname* { m a x } \{ \mathrm { d s t } _ { \mathcal { F } } ( x , x ^ { \prime } ) , \mathrm { d s t } _ { \mathcal { F } } ( x ^ { \prime } , x ) \} \leq \kappa _ { + } \mathrm { d i s t } ( x , x ^ { \prime } ) , } \end{array}
$$

and the same holds in the second argument. Since $K ^ { \prime }$ is bounded by (A1), we get $| W ( x , y ) - W ( x ^ { \prime } , y ) | \leq$ $\kappa _ { + } \| K ^ { \prime } \| _ { \infty } \varepsilon ^ { - m - 1 } \mathrm { d i s t } ( x , x ^ { \prime } )$ , and similarly for $x \mapsto W ( y , x )$ . The bound Equation (30) follows, since ${ \dot { W } } ^ { \bullet }$ is the half-sum or the half-difference of these two functions. □

We also need the normalization $q _ { \varepsilon }$ to be bounded above and away from zero so that the normalized kernel is well-defined. On the one hand, the assumption on the degrees gives $q _ { \varepsilon } = ( d ^ { \prime } + d ) / 2 > d _ { \operatorname* { m i n } }$ . On the other hand, since $\begin{array} { r } { q _ { \varepsilon } ( x ) = \int _ { \mathcal { M } } W ^ { s } ( x , y ) \mathrm { d } \mathbb { P } ( y ) } \end{array}$ , the case $p = 0$ of Equation (29) shows that $q _ { \varepsilon }$ is bounded. Hence, there are constants $C _ { q } , c _ { \theta }$ and $C _ { \theta }$ , independent of ε, such that for every $x \in \mathcal { M }$

$$
d _ { \operatorname* { m i n } } \leq q _ { \varepsilon } ( x ) \leq C _ { q } \qquad \mathrm { a n d } \qquad 0 < c _ { \theta } \leq q _ { \varepsilon } ( x ) ^ { - \theta } \leq C _ { \theta } .\tag{31}
$$

Reduction to numerator and denominator. To apply the δ-net argument, the natural choice is the function $g ( x ) = { \hat { \mathbf { P } } } _ { N } ^ { ( \theta , \bullet ) } [ f ] ( x ) - { \mathcal { P } } ^ { ( \theta , \bullet ) } [ f ] ( x )$ . However, this function is not itself an empirical average, so we apply this argument to its numerator and its denominator separately. One can remark that Equation (25) can be rewritten as a ratio of two empirical averages:

$$
\hat { \mathbf { P } } _ { N } ^ { ( \theta , \bullet ) } [ f ] ( x ) = \frac { \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \hat { W } _ { N } ^ { ( \theta , \bullet ) } ( x , X _ { j } ) ( f ( X _ { j } ) - f ( x ) ) } { \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \hat { W } _ { N } ^ { ( \theta , s ) } ( x , X _ { j } ) } .
$$

Moreover, by using the definition of θ-normalization, it holds:

$$
\hat { \mathbf { P } } _ { N } ^ { ( \theta , \bullet ) } [ f ] ( x ) = \frac { \frac { 1 } { N } \sum _ { j = 1 } ^ { N } W ^ { \bullet } ( x , X _ { j } ) \hat { q } _ { N } ( X _ { j } ) ^ { - \theta } ( f ( X _ { j } ) - f ( x ) ) } { \frac { 1 } { N } \sum _ { j = 1 } ^ { N } W ^ { s } ( x , X _ { j } ) \hat { q } _ { N } ( X _ { j } ) ^ { - \theta } } .
$$

Importantly, the same rewriting holds for the population operator $\mathcal { P } ^ { ( \theta , \bullet ) } [ f ]$ , with $\hat { q } _ { N }$ replaced by $q _ { \varepsilon }$ and the empirical average replaced by an integral over ${ \mathcal { M } } .$ . Finding bounds on the deviation of $\hat { \mathbf { P } } _ { N } ^ { ( \theta , \bullet ) } [ f ]$ from $\mathcal { P } ^ { ( \theta , \bullet ) } [ f ]$ therefore reduces to bounding the deviation of the numerator and denominator separately. Note that due to its definition, $\hat { q } _ { N }$ is itself an empirical average, so that the numerator and denominator are not sums of independent random variables.

Lemma 40 (Uniform deviation of an empirical average). Let $X _ { 1 } , \ldots , X _ { N }$ be i.i.d. with law $\mathbb { P } ,$ and let $\psi :$ $\mathcal { M } \times \mathcal { M }  \mathbb { R }$ be measurable, possibly depending on $N ,$ , and assume that for every $x , x ^ { \prime } , y \in \mathcal { M }$

$$
| \psi ( y , x ) | \leq B , \qquad \int _ { \mathcal { M } } \psi ( y , x ) ^ { 2 } \mathrm { d } \mathbb { P } ( y ) \leq \sigma ^ { 2 } , \qquad | \psi ( y , x ) - \psi ( y , x ^ { \prime } ) | \leq L \mathrm { d i s t } ( x , x ^ { \prime } ) ,
$$

where $B , \sigma$ and L may depend on N. Then, with $A _ { 0 } = 2 ( 5 m + 2 )$ , almost surely for N large enough,

$$
\operatorname* { s u p } _ { x \in \mathcal { M } } \left| \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \psi ( X _ { j } , x ) - \int _ { \mathcal { M } } \psi ( y , x ) \mathrm { d } \mathbb { P } ( y ) \right| \leq A _ { 0 } \left( \sigma \sqrt { \frac { \log N } { N } } + \frac { B \log N } { N } \right) + \frac { 2 L } { N ^ { 5 } } .\tag{32}
$$

Proof. Let $\delta = N ^ { - 5 }$ and let $x _ { 1 } , \ldots , x _ { n _ { \delta } }$ be a δ-net of ${ \mathcal { M } } .$ with $n _ { \delta } \le C _ { \mathcal { M } } N ^ { 5 m }$ by Lemma 38, which holds for N large so that $\delta \leq \delta _ { 0 }$ . For any $x \in \mathcal { M }$ , let $x _ { k }$ be a net point with dist $( x , x _ { k } ) \leq \delta$ . By the triangle inequality,

$$
\begin{array} { r l } { \displaystyle \left| \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \psi ( X _ { j } , x ) - \int _ { \mathcal { M } } \psi ( y , x ) \mathrm { d } \mathbb { P } ( y ) \right| \leq \displaystyle \left. \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \left( \psi ( X _ { j } , x ) - \psi ( X _ { j } , x _ { k } ) \right) \right| } & { } \\ { + \displaystyle \left. \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \psi ( X _ { j } , x _ { k } ) - \int _ { \mathcal { M } } \psi ( y , x _ { k } ) \mathrm { d } \mathbb { P } ( y ) \right. } & { } \\ { + \displaystyle \left. \int _ { \mathcal { M } } \left( \psi ( y , x _ { k } ) - \psi ( y , x ) \right) \mathrm { d } \mathbb { P } ( y ) \right. . } \end{array}
$$

For (A) and (C), the Lipschitz assumption gives $( \mathbf { A } ) \leq L \delta$ and $( \mathbf { C } ) \le L \delta$ , for every realization of the sample. For (B), fix a net point $x _ { k }$ . The net points do not depend on the sample, so $Y _ { j } = \psi ( X _ { j } , x _ { k } )$ are i.i.d. random variables with mean $\begin{array} { r } { \mu _ { k } = \int _ { \mathcal M } \psi ( y , x _ { k } ) \mathrm { d } \mathbb P ( y ) } \end{array}$ . By assumption, $Y _ { j }$ and $- Y _ { j }$ are bounded above by B and have second moment at most $\sigma ^ { 2 }$ . Bernstein’s inequality, applied to both of them, gives for any $t > 0$

$$
\mathbb { P } \left( \left| \frac { 1 } { N } \sum _ { j = 1 } ^ { N } Y _ { j } - \mu _ { k } \right| > t \right) \leq 2 \exp \left( - \frac { N t ^ { 2 } } { 2 ( \sigma ^ { 2 } + B t / 3 ) } \right) .\tag{33}
$$

We set $\begin{array} { r } { t = A _ { 0 } \left( \sigma \sqrt { \frac { \log N } { N } } + \frac { B \log N } { N } \right) } \end{array}$ and check that the exponent is then at least $A _ { 0 } \log N / 2$ . Let $S =$ $\sigma ^ { 2 } + B ^ { 2 } \log N / N$ , which we can assume positive, since otherwise $\psi = 0$ and there is nothing to prove. For the numerator, dropping the nonnegative cross term in the square, we have

$$
N t ^ { 2 } \geq N A _ { 0 } ^ { 2 } \left( \sigma ^ { 2 } \frac { \log N } { N } + B ^ { 2 } \frac { ( \log N ) ^ { 2 } } { N ^ { 2 } } \right) = A _ { 0 } ^ { 2 } S \log N .
$$

For the denominator, the inequality $2 a b \leq a ^ { 2 } + b ^ { 2 }$ gives $\sigma B \sqrt { \log N / N } \le S / 2$ , and we have $B ^ { 2 }$ log $N / N \le S$ so that

$$
\begin{array} { c } { 2 \left( \sigma ^ { 2 } + \displaystyle \frac { B t } { 3 } \right) = 2 \sigma ^ { 2 } + \displaystyle \frac { 2 A _ { 0 } } { 3 } \left( \sigma B \sqrt { \displaystyle \frac { \log N } { N } } + \displaystyle \frac { B ^ { 2 } \log N } { N } \right) } \\ { \leq 2 S + \displaystyle \frac { 2 A _ { 0 } } { 3 } \left( \displaystyle \frac { S } { 2 } + S \right) = ( 2 + A _ { 0 } ) S . } \end{array}
$$

Combining the two, the exponent in Equation (33) is at least $\frac { A _ { 0 } ^ { 2 } } { 2 + A _ { 0 } } \log N$ , and

$$
\frac { A _ { 0 } ^ { 2 } } { 2 + A _ { 0 } } \geq \frac { A _ { 0 } } { 2 } \qquad \Longleftrightarrow \qquad 2 A _ { 0 } \geq 2 + A _ { 0 } \qquad \Longleftrightarrow \qquad A _ { 0 } \geq 2 ,
$$

which holds since $A _ { 0 } = 2 ( 5 m + 2 )$ . Hence (B) exceeds t at the point $x _ { k }$ with probability at most $2 N ^ { - A _ { 0 } / 2 }$ Taking a union bound over the $n _ { \delta }$ net points, the maximum of (B) over the net exceeds t with probability at most

$$
2 n _ { \delta } N ^ { - A _ { 0 } / 2 } = 2 n _ { \delta } N ^ { - ( 5 m + 2 ) } \leq 2 C _ { \mathcal { M } } N ^ { - 2 } ,
$$

which is summable in N. By the Borel–Cantelli lemma, $( \mathtt { B } ) \le t$ for every net point, almost surely for N large enough. Combining the three terms gives Equation (32). □

To apply this lemma to the quantity of interest, we first need to remove the dependence of the kernel on the sample, which is done by replacing $\hat { q } _ { N }$ by its expectation $q _ { \varepsilon }$

We can apply Lemma 40 to $\hat { q } _ { N }$ since it is an empirical average of i.i.d. random variables, ${ \hat { q } } _ { N } ( x ) =$ $\begin{array} { r } { \frac { 1 } { N } \sum _ { j = 1 } ^ { N } W ^ { s } ( x , X _ { j } ) } \end{array}$ , with mean $q _ { \varepsilon } ( x )$ . By Lemma 39 and Equation (31), the kernel $\psi ( y , x ) = W ^ { s } ( x , y )$ satisfies the assumptions of the lemma with $B \leq K ( 0 ) \varepsilon ^ { - m } , \sigma ^ { 2 } \leq B q _ { \varepsilon } ( x ) \leq C \varepsilon ^ { - m }$ and $L \leq C \varepsilon ^ { - m - 1 }$ Recalling that $\zeta _ { N } ^ { 2 } = \log N / ( N \varepsilon ^ { m } )$ ), the right-hand side of Equation (32) is thus at most

$$
C \left( \sqrt { \frac { \log N } { N \varepsilon ^ { m } } } + \frac { \log N } { N \varepsilon ^ { m } } \right) + \frac { C } { \varepsilon ^ { m + 1 } N ^ { 5 } } = C \left( \zeta _ { N } + \zeta _ { N } ^ { 2 } \right) + \frac { C } { \varepsilon ^ { m + 1 } N ^ { 5 } } .
$$

The assumption $N \varepsilon ^ { m } / \log N \to \infty$ gives $\zeta _ { N } \to 0$ , as well as $\varepsilon ^ { - m } \leq N$ and $\varepsilon ^ { - 1 } \leq N ^ { 1 / m } \leq N$ for N large enough, so that the last term is at most $C N ^ { - 3 }$ . Since moreover $\zeta _ { N } \ge \sqrt { \log N / N } \ge N ^ { - 1 / 2 }$ and $\varepsilon \geq N ^ { - 1 }$ we have $N ^ { - 3 } \leq \varepsilon \zeta _ { N }$ , so that this term is negligible here and in all the bounds below. Hence, almost surely for N large enough,

$$
\operatorname* { s u p } _ { x \in \mathcal { M } } | \hat { q } _ { N } ( x ) - q _ { \varepsilon } ( x ) | \leq C \zeta _ { N } .\tag{34}
$$

Using this, we can rewrite the numerator of $\hat { \mathbf { P } } _ { N } ^ { ( \theta , \bullet ) } [ f ]$ as

$$
\begin{array} { l } { { \displaystyle \frac { 1 } { N } \sum _ { j = 1 } ^ { N } W ^ { \bullet } ( x , X _ { j } ) \hat { q } _ { N } ( X _ { j } ) ^ { - \theta } ( f ( X _ { j } ) - f ( x ) ) } } \\ { { \displaystyle \quad = \underbrace { \frac { 1 } { N } \sum _ { j = 1 } ^ { N } W ^ { \bullet } ( x , X _ { j } ) q _ { \ell } ( X _ { j } ) ^ { - \theta } ( f ( X _ { j } ) - f ( x ) ) } _ { ( \ast ) } } } \\ { { \displaystyle \quad \quad + \frac { 1 } { N } \sum _ { j = 1 } ^ { N } W ^ { \bullet } ( x , X _ { j } ) \left( \hat { q } _ { N } ( X _ { j } ) ^ { - \theta } - q _ { \ell } ( X _ { j } ) ^ { - \theta } \right) \left( f ( X _ { j } ) - f ( x ) \right) . } } \end{array}
$$

We can bound the second term (⋆⋆) by

$$
\begin{array} { l } { { \displaystyle | ( \star \star ) | \le \operatorname* { s u p } _ { y \in \mathcal { M } } \big | \widehat { q } _ { N } ( y ) ^ { - \theta } - q _ { \varepsilon } ( y ) ^ { - \theta } \big | \frac { 1 } { N } \sum _ { j = 1 } ^ { N } W ^ { s } ( x , X _ { j } ) | f ( X _ { j } ) - f ( x ) | } } \\ { { \displaystyle \qquad \le C \operatorname* { s u p } _ { y \in \mathcal { M } } \big | \widehat { q } _ { N } ( y ) - q _ { \varepsilon } ( y ) \big | \frac { 1 } { N } \sum _ { j = 1 } ^ { N } W ^ { s } ( x , X _ { j } ) | f ( X _ { j } ) - f ( x ) | } } \\ { { \displaystyle \qquad \le C \zeta _ { N } \frac { 1 } { N } \sum _ { j = 1 } ^ { N } W ^ { s } ( x , X _ { j } ) | f ( X _ { j } ) - f ( x ) | , } } \end{array}
$$

where we used that $| W ^ { \bullet } | \leq W ^ { s }$ , and that $t \mapsto t ^ { - \theta }$ is Lipschitz on $[ d _ { \operatorname* { m i n } } / 2 , 2 C _ { q } ]$ , which contains all the values of $\hat { q } _ { N }$ and $q _ { \varepsilon }$ for N large enough by Equations (31) and (34). Note that the supremum over $y \in \mathcal { M }$ controls in particular the values at the sample points $X _ { j }$ , even though they are random. The last factor is itself an empirical average of i.i.d. random variables, and we show below that it is at most Cε, so that $| ( \star \star ) | \leq C \varepsilon \zeta _ { N }$ The same reasoning applies to the denominator of $\hat { \mathbf { P } } _ { N } ^ { ( \theta , \bullet ) } [ f ]$ , without the factor $| f ( X _ { j } ) - f ( x ) | { : }$ its second term is at most $C \zeta _ { N } \hat { q } _ { N } ( x ) \leq C \zeta _ { N }$ , since $\hat { q } _ { N } \leq 2 C _ { q }$ for N large. To conclude, we need to bound (⋆), which is an empirical average of i.i.d. random variables. To do so, we must show that the lemma applies to the kernels $\begin{array} { r } { \dot { \psi } _ { \bullet } ( y , x ) = \dot { W ^ { \bullet } } ( x , y ) q _ { \varepsilon } ( y ) ^ { - \theta } ( f ( y ) - f ( x ) ) } \end{array}$ and $\psi _ { s } ( y , x ) = W ^ { s } ( x , y ) q _ { \varepsilon } ( y ) ^ { - \theta }$

By Lemma 39 and Equation (31), the kernel $\psi _ { s }$ satisfies the assumptions of Lemma 40 with $B \leq C \varepsilon ^ { - m }$ $\sigma ^ { \bar { 2 } } \leq C \varepsilon ^ { - m }$ and $L \leq \bar { C } \varepsilon ^ { - m - 1 }$ , exactly as for $\hat { q } _ { N }$ . The kernel $\psi _ { \bullet }$ carries the increment $f ( y ) - f ( x )$ , which is of size ε on the effective support of $\dot { W }$ . Since $\bar { \boldsymbol { f } } \in \mathcal { C } ^ { 1 } ( \mathcal { M } )$ is $L _ { f ^ { - } } \mathbf { I }$ Lipschitz and $| W ^ { \bullet } | \leq \dot { W } ^ { s }$ , Lemma 39 gives

$$
B \leq C _ { \theta } L _ { f } \operatorname* { s u p } _ { y \in \mathcal { M } } W ^ { s } ( x , y ) \operatorname { d i s t } ( x , y ) \leq C \varepsilon ^ { 1 - m } ,
$$

$$
\sigma ^ { 2 } \leq B \int _ { \mathcal { M } } | \psi _ { \bullet } ( y , x ) | \mathrm { d } \mathbb { P } ( y ) \leq C \varepsilon ^ { 1 - m } C _ { \theta } L _ { f } \int _ { \mathcal { M } } W ^ { s } ( x , y ) \mathrm { d i s t } ( x , y ) \mathrm { d } \mathbb { P } ( y ) \leq C \varepsilon ^ { 2 - m } ,
$$

and $L \leq C \varepsilon ^ { - m - 1 }$ , since

$$
\begin{array} { r } { | \psi _ { \bullet } ( y , x ) - \psi _ { \bullet } ( y , x ^ { \prime } ) | \leq 2 C _ { \theta } \| f \| _ { \infty } | W ^ { \bullet } ( x , y ) - W ^ { \bullet } ( x ^ { \prime } , y ) | + C _ { \theta } L _ { f } W ^ { s } ( x ^ { \prime } , y ) \mathrm { d i s t } ( x , x ^ { \prime } ) . } \end{array}
$$

The same bounds hold for the kernel $W ^ { s } ( x , y ) | f ( y ) - f ( x ) |$ of the last factor in the bound on $( \star \star )$ , using $\left| | f ( y ) - f ( x ) | - | f ( y ) - f ( x ^ { \prime } ) | \right| \leq L _ { f } \operatorname { d i s t } ( x , x ^ { \prime } )$ . For these kernels, B and σ are smaller than for $\psi _ { s }$ by a factor ε, and so is the first term of the right-hand side of Equation (32), the last one being negligible as seen above. Hence, almost surely for $N$ large enough,

$$
\operatorname* { s u p } _ { x \in \mathcal { M } } \left| \frac { 1 } { N } \sum _ { j = 1 } ^ { N } W ^ { \bullet } ( x , X _ { j } ) q _ { \varepsilon } ( X _ { j } ) ^ { - \theta } ( f ( X _ { j } ) - f ( x ) ) - \int _ { \mathcal { M } } W ^ { \bullet } ( x , y ) q _ { \varepsilon } ( y ) ^ { - \theta } ( f ( y ) - f ( x ) ) \mathrm { d } \mathbb { P } ( y ) \right| \leq C \varepsilon \zeta _ { N } ,
$$

and similarly $\begin{array} { r } { \frac { 1 } { N } \sum _ { j } W ^ { s } ( x , X _ { j } ) | f ( X _ { j } ) - f ( x ) | \leq \int _ { \mathcal { M } } W ^ { s } ( x , y ) | f ( y ) - f ( x ) | \mathrm { d } \mathbb { P } ( y ) + C \varepsilon \zeta _ { N } \leq C \varepsilon } \end{array}$ for every $x \in \mathcal { M }$ , which completes the bound on $( \star \star )$ . The same reasoning applies to the denominator:

$$
\operatorname* { s u p } _ { x \in \mathcal { M } } \left| \frac { 1 } { N } \sum _ { j = 1 } ^ { N } W ^ { s } ( x , X _ { j } ) q _ { \varepsilon } ( X _ { j } ) ^ { - \theta } - \int _ { \mathcal { M } } W ^ { s } ( x , y ) q _ { \varepsilon } ( y ) ^ { - \theta } \mathrm { d } \mathbb { P } ( y ) \right| \le C \zeta _ { N } .
$$

Thus, it remains to show that the ratio of these two empirical averages is close to the ratio of their expectations.

Combining the bounds on $( \star )$ and (⋆⋆), and their analogues for the denominator, we get that, almost surely for N large enough and for every $x \in \mathcal { M }$ , the numerator $\mathcal { N } _ { N } ( x )$ and the denominator $\mathcal { D } _ { N } ( x )$ of $\hat { \mathbf { P } } _ { N } ^ { ( \theta , \bullet ) } [ f ] ( x )$ satisfy

$$
| \mathcal { N } _ { N } ( x ) - \mathcal { N } ( x ) | \leq C \varepsilon \zeta _ { N } , \qquad | \mathfrak { D } _ { N } ( x ) - \mathfrak { D } ( x ) | \leq C \zeta _ { N } ,
$$

where $\begin{array} { r } { \mathcal { N } ( x ) = \int _ { \mathcal { M } } W ^ { \bullet } ( x , y ) q _ { \varepsilon } ( y ) ^ { - \theta } ( f ( y ) - f ( x ) ) \mathrm { d } \mathbb { P } ( y ) } \end{array}$ and $\begin{array} { r } { \mathfrak { D } ( x ) = \int _ { \mathcal { M } } W ^ { s } ( x , y ) q _ { \varepsilon } ( y ) ^ { - \theta } \mathrm { d } \mathbb { P } ( y ) } \end{array}$ are the numerator and the denominator of $\mathcal { P } ^ { ( \theta , \bullet ) } [ f ] ( x )$ , up to the common factor $q _ { \varepsilon } ( x ) ^ { - \theta } .$ . Moreover, since $\begin{array} { r } { \int _ { \mathcal { M } } W ^ { s } ( x , y ) \mathrm { d } \mathbb { P } ( y ) = q _ { \varepsilon } ( x ) } \end{array}$ , Equation $( 3 1 )$ and Lemma 39 give $\begin{array} { r } { \mathfrak { D } ( x ) \ge c _ { \theta } q _ { \varepsilon } ( x ) \ge c _ { \theta } d _ { \mathrm { m i n } } } \end{array}$ and $| \mathcal { N } ( x ) | \leq$ $\begin{array} { r } { \overleftarrow { C _ { \theta } } L _ { f } \int _ { \mathcal { M } } W ^ { s } ( x , y ) \mathrm { d i s t } ( x , y ) \mathrm { d } \mathbb { P } ( y ) \leq C \varepsilon } \end{array}$ . In particular $\mathcal { D } _ { N } ( x ) \geq c _ { \theta } d _ { \mathrm { m i n } } / 2$ for $N$ large. Adding and subtracting $\mathcal { N } ( x ) / \mathcal { D } _ { N } ( x )$ , we get

$$
\begin{array} { r l } & { \displaystyle \left. \hat { \mathbf { P } } _ { N } ^ { ( \theta , \bullet ) } [ f ] ( x ) - \mathcal { P } ^ { ( \theta , \bullet ) } [ f ] ( x ) \right. = \displaystyle \left. \frac { \mathcal { N } _ { N } ( x ) - \mathcal { N } ( x ) } { \mathfrak { D } _ { N } ( x ) } + \frac { \mathcal { N } ( x ) \left( \mathfrak { D } ( x ) - \mathfrak { D } _ { N } ( x ) \right) } { \mathfrak { D } _ { N } ( x ) \mathfrak { D } ( x ) } \right. } \\ & { \quad \quad \quad \quad \le \displaystyle \frac { 2 C \varepsilon \zeta _ { N } } { c _ { \theta } d _ { \operatorname* { m i n } } } + \frac { 2 C \varepsilon \cdot C \zeta _ { N } } { ( c _ { \theta } d _ { \operatorname* { m i n } } ) ^ { 2 } } \le C \varepsilon \zeta _ { N } . } \end{array}
$$

The second term is small because the deviation of the denominator, of order $\zeta _ { N }$ , is multiplied by $\mathcal { N } ( x )$ , which is of order <sub>which con</sub> $\varepsilon .$ . Since this holds for every<sub>udes the proof.</sub> $x \in \mathcal { M }$ , almost surely for $N$ large enough, this gives Proposition $3 7 .$ 7,

Combining the fluctuation and the bias gives the quantitative form of Theorem 4.

Theorem 41. Let $f \in \mathcal { C } ^ { 3 } ( \mathcal { M } )$ and let $\varepsilon = \varepsilon ( N ) \to 0$ with $N \varepsilon ^ { m } / \log N \to \infty$ . Then, almost surely for N large enough,

$$
\operatorname* { s u p } _ { x \in \mathcal { M } } \big | \hat { \mathbf { L } } _ { N } ^ { s } [ f ] ( x ) - \mathcal { L } ^ { s } f ( x ) \big | \le C \left( \varepsilon + \frac { \zeta _ { N } } { \varepsilon } \right) , \qquad \operatorname* { s u p } _ { x \in \mathcal { M } } \big | \hat { \mathbf { L } } _ { N } ^ { a } [ f ] ( x ) - \mathcal { L } ^ { a } f ( x ) \big | \le C \left( \varepsilon + \zeta _ { N } \right) ,\tag{35}
$$

with $\zeta _ { N }$ as in Equation (27). In particular, both right-hand sides vanish, and $\mathbf { L } ^ { \bullet } [ f ]  { \mathcal { L } } ^ { \bullet } f$ uniformly and almost surely, as soon as

$$
\varepsilon ( N ) \to 0 \qquad a n d \qquad { \frac { N \varepsilon ( N ) ^ { m + 2 } } { \log N } } \to \infty .\tag{36}
$$

Proof. By definition $\hat { \mathbf { L } } _ { N } ^ { s } = \varepsilon ^ { - 2 } \hat { \mathbf { P } } _ { N } ^ { ( \theta , s ) }$ , so that, splitting the error into its fluctuation and its bias, we have that:

$$
\begin{array} { r l } & { \displaystyle \big | \hat { \mathbf { L } } _ { N } ^ { s } [ f ] - \mathcal { L } ^ { s } f \big | \leq \frac { 1 } { \varepsilon ^ { 2 } } \big | \hat { \mathbf { P } } _ { N } ^ { ( \theta , s ) } [ f ] - \mathcal { P } ^ { ( \theta , s ) } [ f ] \big | + \frac { 1 } { \varepsilon ^ { 2 } } \big | \mathcal { P } ^ { ( \theta , s ) } [ f ] - \varepsilon ^ { 2 } \mathcal { L } ^ { s } f \big | } \\ & { \qquad \leq \frac { C \varepsilon \zeta _ { N } } { \varepsilon ^ { 2 } } + \frac { C \varepsilon ^ { 3 } } { \varepsilon ^ { 2 } } } \\ & { \qquad = C \left( \frac { \zeta _ { N } } { \varepsilon } + \varepsilon \right) . } \end{array}
$$

By Proposition 37 and Lemma 36

The second bound is obtained in the same way, with $\varepsilon ^ { - 1 }$ in place of $\varepsilon ^ { - 2 }$ . Under Equation (36), we have that $\zeta _ { N } / \varepsilon = \sqrt { \log N / ( N \varepsilon ^ { m + 2 } ) } \to 0 .$ , so that both bounds tend to 0. □

Remark 42 (Which scaling to choose). The two terms ofEquation (35) vary in opposite directions with $\varepsilon \colon a$ large bandwidth averages over many samples but blurs the geometry, while a small one isfaithful but noisy. Balancing ε against $\zeta _ { N } / \varepsilon$ gives the optimal choice

$$
\varepsilon ( N ) { \asymp } \left( \frac { \log N } { N } \right) ^ { \frac { 1 } { m + 4 } } , f o r a n e r r o r o f o r d e r \left( \frac { \log N } { N } \right) ^ { \frac { 1 } { m + 4 } } ,
$$

which exhibits the usual curse of dimensionality in the intrinsic dimension $m ,$ as in the Riemannian case (Singer, 2006; Hein et al., $2 0 0 7 )$ We note that the antisymmetric part is the easier of the two: being rescaled by $\varepsilon ^ { - 1 }$ instead of $\dot { \varepsilon } _ { \varepsilon } { } ^ { - 2 } ,$ , it only requires $N \varepsilon ^ { m } / \log N \to \infty$ , whereas the symmetric part requires the stronger condition Equation (36). In that sense, estimating the direction of the data is statistically cheaper than estimating its geometry.

## D.7 COMPUTATION OF THE MOMENTS IN THE RANDERS SETTING

If F is a Randers metric, the moments of the unit ball ${ \mathcal { B } } _ { x }$ can be computed in closed form. We recall that a Randers metric is defined as $\mathcal { F } ( x , v ) = \sqrt { v ^ { \mathsf { T } } \mathbf { A } ( x ) v } + b ( x ) ^ { \mathsf { T } } v$ , where $\mathbf { A } ( x )$ is a positive-definite matrix and $b ( x )$ is a vector field such that $\| b ( x ) \| _ { \mathbf { A } ^ { - 1 } } < 1$

Proposition 43. For a Randers metric, we have that:

$$
\lambda _ { x } ( \mathcal { B } _ { x } ) = \mathrm { V o l } _ { \mathrm { E u c } } ( \mathbb { B } ^ { m } ) ( 1 - \| b ( x ) \| _ { \mathbf { A } ^ { - 1 } } ^ { 2 } ) ^ { - \frac { m + 1 } { 2 } } \sqrt { \operatorname* { d e t } { \bf A } ^ { - 1 } ( x ) } ,\tag{37}
$$

$$
\mathbf { c } ^ { i } ( x ) = - \frac { b ^ { i } ( x ) } { 1 - \| b ( x ) \| _ { \mathbf { A } ^ { - 1 } } ^ { 2 } } ,\tag{38}
$$

$$
\mathbf { S } ^ { i j } ( x ) = \frac { R _ { x } ^ { 2 } } { m + 2 } \mathbf { E } ^ { i j } ( x ) + \mathbf { c } ^ { i } ( x ) \mathbf { c } ^ { j } ( x ) ,\tag{39}
$$

where $\begin{array} { r } { R _ { x } ^ { 2 } = \frac { 1 } { 1 - \| b ( x ) \| _ { \Lambda ^ { - 1 } } ^ { 2 } } , b ^ { i } ( x ) = \mathbf { A } ^ { i j } ( x ) b _ { j } ( x ) , \mathbf { E } _ { i j } ( x ) = \mathbf { A } _ { i j } ( x ) - b _ { i } ( x ) b _ { j } ( x ) a n d \mathbf { E } ^ { i j } = ( \mathbf { E } _ { i j } ) ^ { - 1 } = \mathbf { A } ^ { i j } + } \end{array}$ $\frac { b ^ { i } b ^ { j } } { 1 - \| b ( x ) \| _ { \mathbf { A } ^ { - 1 } } ^ { 2 } }$

Proof. Using (Ohta, 2021, Lemma 10.8), we have that:

$$
\mathcal { B } _ { x } = \{ v \in T _ { x } \mathcal { M } : \mathbf { E } _ { i j } ( x ) ( v - v _ { 0 } ) ^ { i } ( v - v _ { 0 } ) ^ { j } \leq R _ { x } ^ { 2 } \}
$$

where $\begin{array} { r } { \mathbf { E } _ { i j } ( x ) = \mathbf { A } _ { i j } ( x ) - b _ { i } ( x ) b _ { j } ( x ) , v _ { 0 } ^ { i } = - \frac { \mathbf { A } ^ { i j } ( x ) b _ { j } ( x ) } { 1 - \| b ( x ) \| _ { \mathbf { A } ^ { - 1 } } ^ { 2 } } } \end{array}$ and $\begin{array} { r } { R _ { x } ^ { 2 } = \frac { 1 } { 1 - \| b ( x ) \| _ { \mathsf { A } - 1 } ^ { 2 } } } \end{array}$ . This shows that the unit tangent ball of a Randers metric is an ellipsoid, described by the matrix $\mathbf { E } ( x )$ , centered at $v _ { 0 }$ and with a radius $R _ { x }$

We can compute the volume of the unit ball by using the change of variable $u = { \bf E } ^ { 1 / 2 } ( x ) ( v - v _ { 0 } )$ , which yields that:

$$
\lambda _ { x } ( \mathcal { B } _ { x } ) = \frac { R _ { x } ^ { m } } { \sqrt { \operatorname* { d e t } \mathbf { E } ( x ) } } \mathrm { V o l } _ { \mathrm { E u c } } ( \mathbb { B } ^ { m } ) = \mathrm { V o l } _ { \mathrm { E u c } } ( \mathbb { B } ^ { m } ) R _ { x } ^ { m } \sqrt { \operatorname* { d e t } \mathbf { E } ^ { - 1 } ( x ) }
$$

We have the expression of $R _ { x } ^ { m } \colon R _ { x } ^ { m } = ( 1 - \| b ( x ) \| _ { \mathbf { A } ^ { - 1 } } ^ { 2 } ) ^ { - \frac { m } { 2 } }$ . Moreover, the matrix determinant lemma yields $\operatorname* { d e t } ( \mathbf { E } ) = \operatorname* { d e t } ( \mathbf { A } - b b ^ { \boldsymbol { \mathsf { T } } } ) = \operatorname* { d e t } ( \mathbf { A } ) ( 1 - \| b ( x ) \| _ { \mathbf { A } ^ { - 1 } } ^ { 2 } )$ , which yields that $\sqrt { \operatorname * { d e t } \mathbf { E } ^ { - 1 } ( x ) } = \sqrt { \operatorname * { d e t } \mathbf { A } ^ { - 1 } ( x ) } ( 1 -$ $\| b ( x ) \| _ { \mathbf { A } ^ { - 1 } } ^ { 2 } \big ) ^ { - \frac { 1 } { 2 } }$ . Hence the expression of the volume of the unit ball as:

$$
\lambda _ { x } ( \mathcal { B } _ { x } ) = \mathrm { V o l } _ { \mathrm { E u c } } ( \mathbb { B } ^ { m } ) ( 1 - \| b ( x ) \| _ { \mathbf { A } ^ { - 1 } } ^ { 2 } ) ^ { - \frac { m + 1 } { 2 } } \sqrt { \operatorname* { d e t } \mathbf { A } ^ { - 1 } ( x ) } .
$$

Now we can compute the first moments. Because of the symmetry around the center $v _ { 0 }$ , we have that $\begin{array} { r } { \int _ { \mathcal { B } _ { x } } ( v - v _ { 0 } ) ^ { i } \mathrm { d } \lambda _ { x } ( \bar { v } ) = 0 } \end{array}$ . This yields that $\begin{array} { r } { \int _ { \mathcal { B } _ { x } } v ^ { i } \mathrm { d } \lambda _ { x } ( v ) = \lambda _ { x } ( \mathcal { B } _ { x } ) v _ { 0 } ^ { i } } \end{array}$ and thus:

$$
\mathbf { c } ^ { i } ( x ) = \frac { \int _ { \mathcal { B } _ { x } } v ^ { i } \mathrm { d } \lambda _ { x } ( v ) } { \lambda _ { x } ( \mathcal { B } _ { x } ) } = v _ { 0 } ^ { i } = - \frac { \mathbf { A } ^ { i j } ( x ) b _ { j } ( x ) } { 1 - \| b ( x ) \| _ { \mathbf { A } ^ { - 1 } } ^ { 2 } } = - \frac { b ^ { i } ( x ) } { 1 - \| b ( x ) \| _ { \mathbf { A } ^ { - 1 } } ^ { 2 } } .
$$

And to compute the second moments, we observe that:

$$
v ^ { i } v ^ { j } = ( v - v _ { 0 } ) ^ { i } ( v - v _ { 0 } ) ^ { j } + ( v - v _ { 0 } ) ^ { i } v _ { 0 } ^ { j } + v _ { 0 } ^ { i } ( v - v _ { 0 } ) ^ { j } + v _ { 0 } ^ { i } v _ { 0 } ^ { j } .
$$

Here the cross terms vanish when integrated over $\mathcal { B } _ { x }$ because of the symmetry around the center $v _ { 0 }$ . Using the change of variable $u = { \bf E } ^ { 1 / 2 } ( x ) ( v - v _ { 0 } )$ , we can compute the first term as:

$$
\begin{array} { r l } { \displaystyle \int _ { \mathcal { B } _ { x } } ( v - v _ { 0 } ) ^ { i } ( v - v _ { 0 } ) ^ { j } \mathrm { d } \lambda _ { x } ( v ) = \int _ { \{ u : | u | \| _ { 2 } ^ { 2 } < R _ { x } ^ { 2 } \} } ( \mathbf { E } ^ { - 1 / 2 } ( x ) u ) ^ { i } ( \mathbf { E } ^ { - 1 / 2 } ( x ) u ) ^ { j } \mathrm { d e t } ( \mathbf { E } ^ { - 1 / 2 } ( x ) ) \mathrm { d } u } & { } \\ & { \quad \quad = \operatorname* { d e t } ( \mathbf { E } ^ { - 1 / 2 } ( x ) ) ( \mathbf { E } ^ { - 1 / 2 } ( x ) ) _ { k } ^ { i } ( \mathbf { E } ^ { - 1 / 2 } ( x ) ) _ { l } ^ { j } \int _ { \{ u : | u | \| _ { 2 } ^ { 2 } < R _ { x } ^ { 2 } \} } u ^ { k } u ^ { l } \mathrm { d } u } \\ & { \quad = \operatorname* { d e t } ( \mathbf { E } ^ { - 1 / 2 } ( x ) ) ( \mathbf { E } ^ { - 1 / 2 } ( x ) ) _ { k } ^ { i } ( \mathbf { E } ^ { - 1 / 2 } ( x ) ) _ { l } ^ { j } \frac { R _ { x } ^ { m + 2 } } { m + 2 } \operatorname { V o l } _ { \mathrm { E u c } } ( \mathbb { B } ^ { m } ) \delta ^ { k l } } \\ & { \quad = \frac { R _ { x } ^ { m + 2 } } { m + 2 } \operatorname { V o l } _ { \mathrm { E u c } } ( \mathbb { B } ^ { m } ) \mathbf { E } ^ { i j } ( x ) \sqrt { \operatorname* { d e t } \mathbf { E } ^ { - 1 } ( x ) } } \\ & { \quad = \frac { R _ { x } ^ { 2 } } { m + 2 } \lambda _ { x } ( \mathcal { B } _ { x } ) \mathbf { E } ^ { i j } ( x ) . } \end{array}
$$

where we have that $\begin{array} { r } { \int _ { \{ u : \| u \| _ { 2 } ^ { 2 } < R _ { x } ^ { 2 } \} } u ^ { k } u ^ { l } \mathrm { d } u = \frac { R _ { x } ^ { m + 2 } } { m + 2 } \mathrm { V o l } _ { \mathrm { E u c } } \bigl ( \mathbb { B } ^ { m } \bigr ) \delta ^ { k l } } \end{array}$ by rotation invariance of the Euclidean ball.

So we have that:

$$
\begin{array} { c l c r } { { \displaystyle { \bf S } ^ { i j } ( x ) = \frac { \int _ { \mathfrak { B } _ { x } } v ^ { i } v ^ { j } \mathrm { d } \lambda _ { x } ( v ) } { \lambda _ { x } ( \mathcal { B } _ { x } ) } = \frac { \int _ { \mathfrak { B } _ { x } } ( v - v _ { 0 } ) ^ { i } ( v - v _ { 0 } ) ^ { j } \mathrm { d } \lambda _ { x } ( v ) + \lambda _ { x } ( \mathcal { B } _ { x } ) v _ { 0 } ^ { i } v _ { 0 } ^ { j } } { \lambda _ { x } ( \mathcal { B } _ { x } ) } } } \\ { { \displaystyle { = \frac { R _ { x } ^ { 2 } } { m + 2 } { \bf E } ^ { i j } ( x ) + v _ { 0 } ^ { i } v _ { 0 } ^ { j } } . } } \end{array}
$$

Remark 44. We can see that in the Randers setting, contrary to the Riemannian setting, the centroid of the unit ball is not zero but is shifted proportionally to the vector field b. Similarly, the second moment of the unit ball is not proportional to the inverse of the matrix A but has an additional term proportional to the outer product ofthe centroid with itself. When $b  0 ,$ , we recover the Riemannian setting with $\mathbf { c } ^ { i } ( x ) \to 0$ and $\begin{array} { r } { { \bf S } ^ { i j } ( x )  \frac { 1 } { m + 2 } { \bf A } ^ { i j } ( x ) } \end{array}$

Proposition 45. In the Randers setting, the moments of the kernel $m _ { 0 } ,$ , m˜ <sub>1</sub> and m˜ <sub>2</sub> are given by:

$$
m _ { 0 } ( x ) = m \mu _ { 0 } \mathrm { V o l } _ { \mathrm { E u c } } ( \mathbb { B } ^ { m } ) ,\tag{40}
$$

$$
\tilde { m } _ { 1 } ^ { i } ( x ) = - \frac { ( m + 1 ) \mu _ { 1 } } { m \mu _ { 0 } } \frac { b ^ { i } ( x ) } { 1 - \parallel b ( x ) \parallel _ { \mathbf { A } ^ { - 1 } } ^ { 2 } } ,\tag{41}
$$

$$
\tilde { m } _ { 2 } ^ { i j } ( x ) = \frac { \mu _ { 2 } } { \mu _ { 0 } } \frac { m + 2 } { m } \left( \frac { R _ { x } ^ { 2 } } { m + 2 } { \bf E } ^ { i j } ( x ) + { \bf c } ( x ) ^ { i } { \bf c } ( x ) ^ { j } \right) .\tag{42}
$$

Proof. This is a direct application of Proposition 43 and the expressions of the moments of the kernel in Proposition 35. □

## D.8 THE MOMENT-RANDERS APPROXIMATION

This subsection proves the results of Section 5. We proceed in three steps: we first check that the moment-Randers metric of Definition 5 is well defined, identify its unit ball, and show that it matches the centroid and the second moment of ${ \mathcal { F } } ;$ we then show how c and $g _ { \mathrm { B I } }$ are recovered from $\mathcal { L } ^ { s }$ and ${ \mathcal { L } } ^ { a }$ through an embedding; and we finally relate the resulting empirical metric $\hat { \mathcal { F } } _ { R }$ to $\mathcal { F } _ { R }$

Throughout, we work in a chart, so that $\mathbf { c } ( \boldsymbol { x } ) \in \mathbb { R } ^ { m } , \mathbf { S } ( \boldsymbol { x } ) \in \mathbb { R } ^ { m \times m }$ and ${ \mathcal { B } } _ { x } \subset \mathbb { R } ^ { m }$ . We recall from Equations (10), (22) and (23) that

$$
\mathbf { c } ( x ) = \frac { 1 } { \lambda _ { x } ( \mathcal { B } _ { x } ) } \int _ { \mathcal { B } _ { x } } v \mathrm { d } \lambda _ { x } ( v ) , \qquad \mathbf { S } ( x ) = \frac { 1 } { \lambda _ { x } ( \mathcal { B } _ { x } ) } \int _ { \mathcal { B } _ { x } } v v ^ { \top } \mathrm { d } \lambda _ { x } ( v ) , \qquad g _ { \mathrm { B L } } ( x ) ^ { - 1 } = ( m + 2 ) \mathbf { S } ( x ) ,
$$

so that the matrix H of Definition 5 can be written in the two equivalent ways

$$
\mathbf { H } ( x ) ^ { - 1 } = g _ { \mathrm { B L } } ( x ) ^ { - 1 } - ( m + 2 ) \mathbf { c } ( x ) \mathbf { c } ( x ) ^ { \mathsf { T } } = ( m + 2 ) { \bigl ( } \mathbf { S } ( x ) - \mathbf { c } ( x ) \mathbf { c } ( x ) ^ { \mathsf { T } } { \bigr ) } .
$$

## WELL-DEFINEDNESS OF THE MOMENT-RANDERS METRIC

We begin by proving Proposition $^ { 6 , }$ which gives a necessary and sufficient condition for H to be positive definite, and relates the norms of c in the two metrics. We restate it here for convenience:

Proposition 6. Let F be a Finsler metric and let $x \in \mathcal { M }$ be such that $\| \mathbf { c } ( x ) \| _ { g _ { \mathrm { B L } } } ^ { 2 } < ( m + 2 ) ^ { - 1 }$ . Then the matrix H(x) is well defined and positive definite, and

$$
\| { \mathbf { c } } ( x ) \| _ { { \mathbf { g } } _ { \mathrm { { R } } } } ^ { 2 } = \frac { \| { \mathbf { c } } ( x ) \| _ { \mathbf { H } } ^ { 2 } } { 1 + ( m + 2 ) \| { \mathbf { c } } ( x ) \| _ { \mathbf { H } } ^ { 2 } } , \qquad h e n c e \qquad \| { \mathbf { c } } ( x ) \| _ { \mathbf { H } } < 1 \Longleftrightarrow \| { \mathbf { c } } ( x ) \| _ { \mathbf { g } _ { \mathrm { { B L } } } } ^ { 2 } < ( m + 3 ) ^ { - 1 } .
$$

ProofofProposition 6. We first check that $\mathbf { H } ( x )$ is positive definite, hence invertible. The matrix ${ \bf H } ^ { - 1 }$ is a rank-one downdate of $g _ { \mathrm { B L } } ^ { - 1 }$ , so its positivity only depends on the size of c: for any $u \in T _ { x } ^ { * } \mathcal { M }$ , the Cauchy– Schwarz inequality gives $\begin{array} { r } { \langle \mathbf { c } , u \rangle ^ { 2 } \leq \| \mathbf { c } \| _ { g _ { \mathrm { B L } } } ^ { 2 } \| u \| _ { g _ { \mathrm { B I } } ^ { - 1 } } ^ { 2 } } \end{array}$ , so that

$$
u ^ { \mathsf { T } } \mathbf { H } ^ { - 1 } u = \left\| u \right\| _ { { g _ { \mathrm { B L } } ^ { - 1 } } } ^ { 2 } - ( m + 2 ) \langle \mathbf { c } , u \rangle ^ { 2 } \geq \left( 1 - ( m + 2 ) \| \mathbf { c } \| _ { { g _ { \mathrm { B L } } } } ^ { 2 } \right) \left\| u \right\| _ { { g _ { \mathrm { B L } } ^ { - 1 } } } ^ { 2 } ,
$$

which is positive for every $u \neq 0$ under the hypothesis. The bound is attained at $u = g _ { \mathrm { B L } } \mathbf { c } ,$ the equality case of Cauchy–Schwarz, where $u ^ { \mathsf { T } } \mathbf { H } ^ { - 1 } u = \| \mathbf { c } \| _ { g _ { \mathrm { B L } } } ^ { 2 ^ { \mathsf { ^ { * } } } } ( 1 - ( m + 2 ) \| \mathbf { c } \| _ { g _ { \mathrm { B L } } } ^ { 2 } )$ , so the hypothesis is also necessary when $\mathbf { c } \neq 0$

We now turn to the identity. We recall that $\mathbf { H } ^ { - 1 } = ( m + 2 ) ( \mathbf { S } - \mathbf { c } \mathbf { c } ^ { \mathsf { T } } )$ , so that $\begin{array} { r } { \mathbf { S } = \frac { 1 } { m + 2 } \mathbf { H } ^ { - 1 } + \mathbf { c c } ^ { \mathsf { T } } } \end{array}$ . Setting $\begin{array} { r } { A = \frac { 1 } { m + 2 } \mathbf { H } ^ { - 1 } } \end{array}$ , so that $A ^ { - 1 } = ( m + 2 ) { \bf H }$ , the Sherman–Morrison formula yields

$$
\mathbf { S } ^ { - 1 } = ( A + \mathbf { c } \mathbf { c } ^ { \mathsf { T } } ) ^ { - 1 } = A ^ { - 1 } - { \frac { A ^ { - 1 } \mathbf { c } \mathbf { c } ^ { \mathsf { T } } A ^ { - 1 } } { 1 + \mathbf { c } ^ { \mathsf { T } } A ^ { - 1 } \mathbf { c } } } = ( m + 2 ) \mathbf { H } - { \frac { ( m + 2 ) ^ { 2 } \mathbf { H } \mathbf { c } \mathbf { c } ^ { \mathsf { T } } \mathbf { H } } { 1 + ( m + 2 ) \mathbf { c } ^ { \mathsf { T } } \mathbf { H } \mathbf { c } } } .
$$

Hence

$$
\mathbf { c } ^ { \mathsf { T } } \mathbf { S } ^ { - 1 } \mathbf { c } = ( m + 2 ) \| \mathbf { c } \| _ { \mathbf { H } } ^ { 2 } - { \frac { ( m + 2 ) ^ { 2 } \| \mathbf { c } \| _ { \mathbf { H } } ^ { 4 } } { 1 + ( m + 2 ) \| \mathbf { c } \| _ { \mathbf { H } } ^ { 2 } } } = { \frac { ( m + 2 ) \| \mathbf { c } \| _ { \mathbf { H } } ^ { 2 } } { 1 + ( m + 2 ) \| \mathbf { c } \| _ { \mathbf { H } } ^ { 2 } } } ,
$$

and, because $g _ { \mathrm { B L } } ^ { - 1 } = ( m + 2 ) \mathbf { S }$

$$
\| { \boldsymbol { \mathbf { c } } } \| _ { g _ { \mathrm { B L } } } ^ { 2 } = { \boldsymbol { \mathbf { \mathbf { c } } } } ^ { \mathsf { T } } g _ { \mathrm { B L } } { \boldsymbol { \mathbf { \mathbf { c } } } } = { \frac { 1 } { m + 2 } } { \boldsymbol { \mathbf { \mathbf { c } } } } ^ { \mathsf { T } } { \boldsymbol { \mathbf { S } } } ^ { - 1 } { \boldsymbol { \mathbf { \mathbf { c } } } } = { \frac { \| { \boldsymbol { \mathbf { c } } } \| _ { \mathbf { H } } ^ { 2 } } { 1 + ( m + 2 ) \| { \boldsymbol { \mathbf { c } } } \| _ { \mathbf { H } } ^ { 2 } } } .
$$

The right-hand side is an increasing function of $\| \mathbf { c } \| _ { \mathbf { H } } ^ { 2 }$ , and

$$
\| { \bf c } \| _ { g _ { \mathrm { B L } } } ^ { 2 } < \frac { 1 } { m + 3 } \Longleftrightarrow ( m + 3 ) \| { \bf c } \| _ { \bf H } ^ { 2 } < 1 + ( m + 2 ) \| { \bf c } \| _ { \bf H } ^ { 2 } \Longleftrightarrow \| { \bf c } \| _ { \bf H } ^ { 2 } < 1 .
$$

This concludes the proof of Proposition 6.

The metric of Definition 5 is written in the $\alpha + \beta$ form Equation (1). The following remark shows that it can equivalently be written in the navigationform.

Remark 46 (Equivalent navigation form). With the $\alpha + \beta$ form of Definition 5, we have ${ \bf A } _ { R } = \lambda ^ { - 1 } { \bf H } +$ $\lambda ^ { - 2 } ( \mathbf { H } \mathbf { c } ) ( \mathbf { H } \mathbf { c } ) ^ { \top }$ and $b _ { R } = - \bar { \lambda ^ { - 1 } } \mathbf { H } \mathbf { c } ,$ , where $\lambda = 1 - \| \mathbf { c } \| _ { \mathbf { H } } ^ { 2 } .$ Hence

$$
v ^ { \mathsf { T } } \mathbf { A } _ { R } v = { \frac { \lambda \| v \| _ { \mathbf { H } } ^ { 2 } + \langle \mathbf { c } , v \rangle _ { \mathbf { H } } ^ { 2 } } { \lambda ^ { 2 } } } , \qquad b _ { R } ^ { \mathsf { T } } v = - { \frac { \langle \mathbf { c } , v \rangle _ { \mathbf { H } } } { \lambda } } ,
$$

so that $\mathcal { F } _ { R }$ can equivalently be written as

$$
\mathcal { F } _ { R } ( x , v ) = \frac { 1 } { \lambda ( x ) } \left( \sqrt { \lambda ( x ) \| v \| _ { \mathbf { H } } ^ { 2 } + \langle \mathbf { c } ( x ) , v \rangle _ { \mathbf { H } } ^ { 2 } } - \langle \mathbf { c } ( x ) , v \rangle _ { \mathbf { H } } \right) , \qquad \lambda ( x ) = 1 - \| \mathbf { c } ( x ) \| _ { \mathbf { H } } ^ { 2 } .\tag{43}
$$

Importantly, the admissibility condition $\| b _ { R } ( x ) \| _ { \mathbf { A } _ { R } ^ { - 1 } } < 1$ is equivalent to $\| \mathbf { c } ( x ) \| _ { \mathbf { H } } < 1$ , which is exactly the R   
hypothesis of Definition 5. Indeed,

$$
\begin{array} { r } { \| b _ { R } \| _ { { \mathbf { A } } _ { R } ^ { - 1 } } ^ { 2 } = b _ { R } ^ { \mathrm { T } } { \mathbf { A } } _ { R } ^ { - 1 } b _ { R } = \lambda ^ { 2 } \mathbf { c } ^ { \mathrm { T } } \mathbf { H } ^ { - 1 } \left( \lambda ( \mathbf { H } ^ { - 1 } - \mathbf { c c } ^ { \mathrm { T } } ) \right) \mathbf { H } ^ { - 1 } \mathbf { c } = \lambda ^ { 2 } ( \| \mathbf { c } \| _ { \mathbf { H } } ^ { 2 } - \| \mathbf { c } \| _ { \mathbf { H } } ^ { 4 } ) / \lambda ^ { 2 } = \| \mathbf { c } \| _ { \mathbf { H } } ^ { 2 } . } \end{array}
$$

We can now justify the name: $\mathcal { F } _ { R }$ is the unique Randers metric whose unit ball has the same centroid and the same second moment as ${ \mathcal { B } } _ { x }$ , and it therefore leaves $\tilde { m } _ { 1 }$ and $\tilde { m } _ { 2 }$ unchanged.

Proposition 47 (Moment matching). Let Fbe a Finsler metric with $\| \mathbf { c } ( x ) \| _ { \mathbf { H } } < 1$ for all $x \in \mathcal { M }$ , and let $\mathcal { F } _ { R }$ be its moment-Randers metric (Definition 5). Then $\mathcal { F } _ { R }$ is a Randers metric and:

1. its unit tangent ball is the ellipsoid $E _ { x } = \left\{ v \in T _ { x } \mathcal { M } : \| v - \mathbf { c } ( x ) \| _ { \mathbf { H } } \leq 1 \right\}$

2. F and $\mathcal { F } _ { R }$ have the same centroid and the same raw second moment, $\mathbf { c } _ { R } = \mathbf { c }$ and $\mathbf { S } _ { R } = \mathbf { S } ,$ ; consequently they have the same Binet–Legendre metric and the same normalized kernel moments $\tilde { m } _ { 1 }$ and $\tilde { m } _ { 2 } ;$

3. if Fis itselfa Randers metric, then $\mathcal { F } _ { R } = \mathcal { F }$

Proof. Unit tangent ball. We start by showing that $\mathcal { F } _ { R }$ is a valid Randers metric. $\mathrm { I f } \ \| \mathbf { c } ( x ) \| _ { \mathbf { H } } < 1$ , then $\lambda ( x ) > 0$ . Consequently we have that $\bar { \mathcal { F } } _ { R } ( x , v ) \bar { \geq } 0$ for all $v \in T _ { x } { \mathcal { M } }$ . Now if we assume that $\mathcal { F } _ { R } ( x , v ) = \mathcal { C }$ for some v $\in T _ { x } { \mathcal { M } }$ , then we have that $\lambda ( x ) \| v \| _ { \mathbf { H } } ^ { 2 } + \langle \mathbf { c } ( x ) , v \rangle _ { \mathbf { H } } ^ { 2 } = \langle \mathbf { c } ( x ) , v \rangle _ { \mathbf { H } } ^ { 2 }$ , which implies that $\lambda ( \dot { x } ) \lVert \dot { v } \rVert _ { \mathbf { H } } ^ { 2 } = 0$ Since $\lambda ( x ) > 0$ , we have that $\| v \| _ { \mathbf { H } } ^ { 2 } = 0$ , which implies that $v = 0$ . Therefore, $\mathcal { F } _ { R } ( x , v ) = 0$ if and only if $v = 0$ . Positive homogeneity and the triangle inequality are trivial. Hence, $\mathcal { F } _ { R }$ is a valid Randers metric.

Now let $v \in T _ { x } { \mathcal { M } }$ . We have the following chain of equivalences:

$$
\begin{array} { r l } & { \mathcal { F } _ { \mathbb { R } } ( x , v ) \leq 1 \Longleftrightarrow \sqrt { \lambda ( x ) \| v \| _ { \mathbb H } ^ { 2 } + \langle c ( x ) , v \rangle _ { \mathbb H } ^ { 2 } } - \langle c ( x ) , v \rangle _ { \mathbb H } \leq \lambda ( x ) } \\ & { \qquad \Longleftrightarrow \sqrt { \lambda ( x ) \| v \| _ { \mathbb H } ^ { 2 } + \langle c ( x ) , v \rangle _ { \mathbb H } ^ { 2 } } \leq \lambda ( x ) + \langle c ( x ) , v \rangle _ { \mathbb H } } \\ & { \qquad \Longleftrightarrow \lambda ( x ) \| v \| _ { \mathbb H } ^ { 2 } + \langle c ( x ) , v \rangle _ { \mathbb H } ^ { 2 } \leq \lambda ( x ) + \langle c ( x ) , v \rangle _ { \mathbb H } ^ { 2 } } \\ & { \qquad \Longleftrightarrow \lambda \langle c \rangle \| v \| _ { \mathbb H } ^ { 2 } + \langle c ( x ) , v \rangle _ { \mathbb H } ^ { 2 } \leq \lambda ( x ) ^ { 2 } + 2 \lambda ( x ) \langle c ( x ) , v \rangle _ { \mathbb H } + \langle c ( x ) , v \rangle _ { \mathbb H } ^ { 2 } } \\ & { \qquad \Longleftrightarrow \lambda \langle v \rangle \| v \| _ { \mathbb H } ^ { 2 } \leq \lambda ( x ) ^ { 2 } + 2 \lambda ( x ) \langle c ( x ) , v \rangle _ { \mathbb H } } \\ & { \qquad \Longleftrightarrow \| v \| _ { \mathbb H } ^ { 2 } \leq \lambda ( x ) + 2 \langle c ( x ) , v \rangle _ { \mathbb H } } \\ & { \qquad \Longleftrightarrow \| v - \mathbf c ( x ) + c ( x ) \| _ { \mathbb H } ^ { 2 } \leq \lambda ( x ) + 2 \langle c ( x ) , v \rangle _ { \mathbb H } } \\ & { \qquad \Longleftrightarrow \| v - \mathbf c ( x ) \| _ { \mathbb H } ^ { 2 } + 2 \langle v - \mathbf c ( x ) , c ( x ) \rangle _ { \mathbb H } + \| c ( x ) \| _ { \mathbb H } ^ { 2 } \leq 1 - \| c ( x ) \| _ { \mathbb H } ^ { 2 } + 2 \langle c ( x ) , v \rangle _ { \mathbb H } } \\ &  \qquad \Longleftrightarrow \| v - \mathbf c ( x ) \| _ { \mathbb H } ^ { 2 } + 2 \langle v , \mathbf c ( x ) \rangle _ { \mathbb H } - \end{array}
$$

Moment matching. By the first item above, the unit tangent ball of $\mathcal { F } _ { R }$ is $E _ { x } = \left\{ v \in T _ { x } \mathcal { M } : \lVert v - \mathbf { c } ( x ) \rVert _ { \mathbf { H } } \leq 1 \right\}$ Hence its centroid is $\mathbf { c } ( x )$

We now compute the second moment of the unit tangent ball of the moment-Randers metric:

$$
\begin{array} { l } { { \displaystyle { \bf S } _ { R } ( x ) = \frac { 1 } { \lambda _ { x } ( E _ { x } ) } \int _ { E _ { x } } v v ^ { \top } \mathrm { d } \lambda _ { x } ( v ) } \ ~ } \\ { { \displaystyle ~ = \frac { 1 } { \lambda _ { x } ( E _ { x } ) } \int _ { E _ { x } } ( v - \mathbf { c } ( x ) + \mathbf { c } ( x ) ) ( v - \mathbf { c } ( x ) + \mathbf { c } ( x ) ) ^ { \top } \mathrm { d } \lambda _ { x } ( v ) } \ ~ } \\ { { \displaystyle ~ = \frac { 1 } { \lambda _ { x } ( E _ { x } ) } \int _ { E _ { x } } ( v - \mathbf { c } ( x ) ) ( v - \mathbf { c } ( x ) ) ^ { \top } \mathrm { d } \lambda _ { x } ( v ) + \mathbf { c } ( x ) \mathbf { c } ( x ) ^ { \top } } , } \end{array}
$$

where we used that $\begin{array} { r } { \int _ { E _ { x } } ( v - \mathbf c ( x ) ) \mathrm { d } \lambda _ { x } ( v ) = 0 } \end{array}$ . Now, as $E _ { x }$ is an ellipsoid, we have that $\begin{array} { r } { \int _ { E _ { x } } ( v - \mathbf { c } ( x ) ) ( v - \mathbf { \rho } } \end{array}$ $\begin{array} { r } { \mathbf { c } ( \boldsymbol { x } ) ) ^ { \mathsf { T } } \mathrm { d } \lambda _ { x } ( \boldsymbol { v } ) = \frac { 1 } { m + 2 } \bar { \mathbf { H } } ( \boldsymbol { x } ) ^ { - 1 } } \end{array}$ , and thus

$$
\begin{array} { l } { \displaystyle { \mathbf { S } _ { R } ( x ) = \frac { 1 } { m + 2 } \mathbf { H } ( x ) ^ { - 1 } + \mathbf { c } ( x ) \mathbf { c } ( x ) ^ { \mathsf { T } } } } \\ { \displaystyle ~ = \mathbf { S } ( x ) } \end{array}
$$

where we used that $\mathbf { H } ( x ) ^ { - 1 } = ( m + 2 ) ( \mathbf { S } ( x ) - \mathbf { c } ( x ) \mathbf { c } ( x ) ^ { \mathsf { T } } )$

Uniqueness. Since $\tilde { \mathcal { F } } ( \boldsymbol { x } , \cdot )$ is positively homogeneous of degree 1 for any Finsler metric $\tilde { \mathcal { F } }$ , we have $\tilde { \mathcal { F } } ( x , v / t ) = \tilde { \mathcal { F } } ( x , v ) / t$ for $t > 0 .$ , so that $v / t \in \{ \tilde { \mathcal { F } } ( x , \cdot ) \leq 1 \} \Longleftrightarrow \tilde { \mathcal { F } } ( x , v ) \leq t ,$ , and thus

$$
\tilde { \mathcal { F } } ( x , v ) = \operatorname* { i n f } \{ t > 0 : v / t \in \{ \tilde { \mathcal { F } } ( x , \cdot ) \leq 1 \} \} , \qquad \forall v \in T _ { x } \mathcal { M } .
$$

In particular, a Finsler metric is determined by its unit tangent balls: two Finsler metrics with the same unit tangent ball at every $x \in \mathcal { M }$ are equal. The goal of this paragraph is to show that, for a Randers metric, the unit tangent ball is uniquely determined by its centroid and its raw second moment.

By definition, the centroid c and the raw second moment S of a Randers metric are given by

$$
\mathbf { c } = \frac { 1 } { \lambda _ { x } ( \mathcal { B } _ { x } ) } \int _ { \mathcal { B } _ { x } } { v } \mathrm { d } \lambda _ { x } ( v ) , \qquad \mathbf { S } = \frac { 1 } { \lambda _ { x } ( \mathcal { B } _ { x } ) } \int _ { \mathcal { B } _ { x } } { v } v ^ { \mathsf { T } } \mathrm { d } \lambda _ { x } ( v ) ,
$$

where $\mathcal { B } _ { x } = \{ v \in T _ { x } . \mathcal { M } : \mathcal { F } ( x , v ) \leq 1 \}$ is the unit tangent ball of the Randers metric. By Proposition 43, we have that $\mathcal { B } _ { x }$ is an ellipsoid, and by Proposition 45, we have that c and S are given by explicit formulas in terms of the parameters A and b of the Randers metric. These formulas are invertible, so that A and b can be recovered from c and S. Hence the unit tangent ball ${ \mathcal { B } } _ { x }$ is uniquely determined by c and S, and so is the Randers metric F. □

## RECOVERING THE CENTROID AND THE BINET–LEGENDRE METRIC FROM AN EMBEDDING

Let $\Psi = ( \psi _ { 1 } , \dots , \psi _ { \ell } ) : \mathcal { M } \to \mathbb { R } ^ { \ell }$ be a smooth embedding, and denote by $J ( \boldsymbol { x } ) \in \mathbb { R } ^ { \ell \times m }$ its Jacobian at x, whose k-th row is $\dot { \nabla } \psi _ { k } ( x ) ^ { \mathsf { T } }$ . We assume throughout that $J ( x )$ has full row rank m, so that $\ell \geq m$ and $J ( x ) ^ { + } = ( J ( x ) ^ { \top } J ( x ) ) ^ { - 1 } J ( x ) ^ { \top }$ is the pseudo-inverse of $J ( x ) , { \mathrm { i . e . ~ } } J ( x ) ^ { + } J ( x ) = I _ { m } .$ Operators are applied to Ψ coordinate-wise, so that $\mathcal { L } ^ { a } [ \Psi ] = ( \mathcal { L } ^ { a } [ \psi _ { 1 } ] , \dots , \mathcal { L } ^ { a } [ \dot { \psi _ { \ell } } ] ) ^ { \top }$ . Recall from Proposition 35 and Theorem 2 that the anti-symmetric operator ${ \mathcal { L } } ^ { a }$ is given by:

$$
\begin{array} { r } { \mathcal { L } ^ { a } [ f ] ( x ) = \big \langle \tilde { m } _ { 1 } ( x ) , \nabla f ( x ) \big \rangle = c _ { 1 } \big \langle \mathbf { c } ( x ) , \nabla f ( x ) \big \rangle , } \end{array}
$$

where $\mathbf { c } ( x )$ is the centroid of the unit ball $\mathcal { B } _ { x }$ and $c _ { 1 } = ( ( m + 1 ) \mu _ { 1 } ) / ( m \mu _ { 0 } )$ is a constant that depends on the kernel. Applying ${ \mathcal { L } } ^ { a }$ to the embedding Ψ yields:

$$
V ( \boldsymbol { x } ) : = \mathcal { L } ^ { a } [ \Psi ] ( \boldsymbol { x } ) = c _ { 1 } \left( \begin{array} { c } { \left. \mathbf { c } ( \boldsymbol { x } ) , \nabla \psi _ { 1 } ( \boldsymbol { x } ) \right. } \\ { \vdots } \\ { \left. \mathbf { c } ( \boldsymbol { x } ) , \nabla \psi _ { \ell } ( \boldsymbol { x } ) \right. } \end{array} \right) = c _ { 1 } J ( \boldsymbol { x } ) \mathbf { c } ( \boldsymbol { x } ) .
$$

Consequently, we can recover the centroid c(x) from $\mathcal { L } ^ { a } \Psi ( x )$ by

$$
\mathbf { c } ( x ) = \frac { 1 } { c _ { 1 } } J ( x ) ^ { + } \mathcal { L } ^ { a } \Psi ( x ) .
$$

The Binet–Legendre metric is recovered in the same way, from the symmetric part, through the carre du´ champ of $\mathcal { L } ^ { s }$ . The point is that this bilinear form only retains the principal part of $\mathcal { L } ^ { s }$ : the sampling density $\rho$ and the normalization θ, which enter $\mathcal { L } ^ { s }$ only through first-order terms, disappear. This is the content of Proposition 7, which we now prove.

Proof of Proposition 7. By Theorem 2, in a chart $\mathcal { L } ^ { s }$ is a second-order differential operator with principal part $c _ { 2 } g _ { \mathrm { B L } } ^ { i j } \partial _ { i } \partial _ { j }$ and no zeroth-order term. We denote by ξ the coefficients collecting all first-order contributions, so that $\begin{array} { r } { \mathcal { L } ^ { s } = c _ { 2 } g _ { \mathrm { B L } } ^ { i j } \partial _ { i } \partial _ { j } + \xi ^ { i } \partial _ { i } } \end{array}$ . For smooth $f$ and $h ,$ the Leibniz rule gives $\partial _ { i } ( f h ) = f \partial _ { i } h + h \partial _ { i } f$ and

$$
\partial _ { i } \partial _ { j } ( f h ) = f \partial _ { i } \partial _ { j } h + h \partial _ { i } \partial _ { j } f + \partial _ { i } f \partial _ { j } h + \partial _ { j } f \partial _ { i } h ,
$$

so that all terms involving ξ and all second derivatives of $f$ and h cancel in the combination $\mathcal { L } ^ { s } [ f h ] -$ $f \mathscr { L } ^ { s } [ h ] - h \mathscr { L } ^ { s } [ f ]$ . Using the symmetry of $g _ { \mathrm { B L } } ^ { i j }$ , we are left with

$$
\frac 1 2 \Big ( \mathscr { L } ^ { s } [ f h ] - f \mathscr { L } ^ { s } [ h ] - h \mathscr { L } ^ { s } [ f ] \Big ) = c _ { 2 } g _ { \mathrm { B L } } ^ { i j } \partial _ { i } f \partial _ { j } h .
$$

Applying this with $f = \psi _ { k }$ and $h = \psi _ { l }$ gives $\Gamma ^ { k l } = c _ { 2 } g _ { \mathrm { B L } } ^ { i j } \partial _ { i } \psi _ { k } \partial _ { j } \psi _ { l }$ , which is the $( k , l )$ entry of $\begin{array} { r } { c _ { 2 } J g _ { \mathrm { B L } } ^ { - 1 } J ^ { \mathsf { T } } , } \\ { \qquad \boxed { \begin{array} { r l r } \end{array} } } \end{array}$ with $\begin{array} { r } { c _ { 2 } = \frac { \mu _ { 2 } } { 2 m \mu _ { 0 } } . } \end{array}$

Proposition 7 gives the following corollary, which allows us to compute the norm of the drift in the Binet– Legendre metric from the embedding coordinates $\Psi ( x )$ and the operators $\mathcal { L } ^ { s }$ and ${ \mathcal { L } } ^ { a }$

Corollary 48. $H J ( x )$ has full column rank, then

$$
\Gamma ( x ) ^ { + } = \frac { 1 } { c _ { 2 } } J ( x ) ^ { + \top } g _ { \mathrm { B L } } ( x ) J ( x ) ^ { + } , \qquad a n d \qquad \| \mathbf { c } ( x ) \| _ { g _ { \mathrm { B L } } } ^ { 2 } = \frac { c _ { 2 } } { c _ { 1 } ^ { 2 } } \| V ( x ) \| _ { \Gamma ^ { + } } ^ { 2 } .
$$

Consequently, by Proposition 6, the moment-Randers metric of $\mathcal { F }$ is well defined at $x$ if and only if $\begin{array} { r } { \frac { c _ { 2 } } { c _ { 1 } ^ { 2 } } \| V ( \dot { x } ) \| _ { \Gamma ^ { + } } ^ { 2 } < ( \dot { m } + 3 \dot { ) } ^ { - 1 } } \end{array}$ , which is computable from $\mathcal { L } ^ { s }$ and ${ \mathcal { L } } ^ { a }$ alone.

Proof. Fix $x \in \mathcal { M }$ , write $M ( x ) = c _ { 2 } ^ { - 1 } J ( x ) ^ { + \top } g _ { \mathrm { B L } } ( x ) J ( x ) ^ { + }$ , and let $P ( x ) = J ( x ) J ( x ) ^ { + }$ be the orthogonal projector onto the range of $J ( x ) ^ { 8 }$ ; both $\Gamma ( x )$ and $M ( x )$ are symmetric. Using Proposition 7 and $J ( x ) ^ { \bar { + } } J ( x ) = I _ { m } ,$

$$
\Gamma ( x ) M ( x ) = J ( x ) g _ { \mathrm { B L } } ( x ) ^ { - 1 } { \left( J ( x ) ^ { + } J ( x ) \right) ^ { \top } } g _ { \mathrm { B L } } ( x ) J ( x ) ^ { + } = J ( x ) J ( x ) ^ { + } = P ( x ) .
$$

Since $P ( x )$ is symmetric, we also have $M ( x ) \Gamma ( x ) = P ( x )$ . Applying this, it holds that:

$$
\Gamma ( x ) M ( x ) \Gamma ( x ) = \Gamma ( x ) P ( x ) = \Gamma ( x ) , \qquad M ( x ) \Gamma ( x ) M ( x ) = M ( x ) P ( x ) = M ( x ) .
$$

Together with the symmetry of $\Gamma ( x ) M ( x )$ and $M ( x ) \Gamma ( x )$ , these are exactly the Penrose conditions characterizing the Moore–Penrose pseudoinverse, so $M ( x ) \stackrel { \cdot } { = } \Gamma ( x ) ^ { + }$ . The second identity then follows from $V ( x ) = c _ { 1 } J ( x ) \mathbf { c } ( x )$ and the definition of the pseudo-inverse:

$$
\left\| \mathbf { c } ( x ) \right\| _ { \mathcal { I } \mathrm { { R L } } } ^ { 2 } = \frac { 1 } { c _ { 1 } ^ { 2 } } V ( x ) ^ { \mathsf { T } } J ( x ) ^ { + \mathsf { T } } g _ { \mathrm { { B L } } } ( x ) J ( x ) ^ { + } V ( x ) = \frac { c _ { 2 } } { c _ { 1 } ^ { 2 } } V ( x ) ^ { \mathsf { T } } \Gamma ( x ) ^ { + } V ( x ) = \frac { c _ { 2 } } { c _ { 1 } ^ { 2 } } \left\| V ( x ) \right\| _ { \Gamma ^ { + } } ^ { 2 } .
$$

To sum up, we can recover both the direction and the magnitude, in the Binet–Legendre metric, of the vector field $\mathbf { c } ( x )$ from the vector field $V ( x )$ in the embedding space. Note that neither the sampling density $\rho$ nor the normalization exponent θ appears in $\| \mathbf { c } ( x ) \| _ { g _ { \mathrm { B L } } } ^ { 2 }$

## THE EMPIRICAL MOMENT-RANDERS METRIC

Building on the above, we can now define a metric from the observed quantities $\Gamma ( x )$ and $V ( x )$ . Following Section 5, set

$$
\begin{array} { l l } { { \hat { g } _ { B L } ( x ) ^ { - 1 } = \Gamma ( x ) , } } & { { \qquad \hat { \mathbf { c } } ( x ) = \frac { \sqrt { c _ { 2 } } } { c _ { 1 } } V ( x ) , } } \\ { { \hat { \mathbf { H } } ( x ) ^ { - 1 } = \Gamma ( x ) - ( m + 2 ) \frac { c _ { 2 } } { c _ { 1 } ^ { 2 } } V ( x ) V ( x ) ^ { \mathsf { T } } , } } & { { \qquad \hat { \lambda } ( x ) = 1 - \| \hat { \mathbf { c } } ( x ) \| _ { \hat { \mathbf { H } } } ^ { 2 } , } } \end{array}\tag{44}
$$

<sup>8</sup>Since $J ( x )$ has full column rank, $J ( x ) ^ { + } = \left( J ( x ) ^ { \mathsf { T } } J ( x ) \right) ^ { - 1 } J ( x ) ^ { \mathsf { T } } , \mathsf { s o } P ( x ) { = } J ( x ) { \left( J ( x ) ^ { \mathsf { T } } J ( x ) \right) } ^ { - 1 } J ( x ) ^ { \mathsf { T } }$ is symmetric.

with $\hat { \mathbf { H } } : = ( \hat { \mathbf { H } } ( x ) ^ { - 1 } ) ^ { + }$ . Let $\scriptstyle { \hat { \mathcal { F } } } _ { R }$ be defined by Definition 5 with $( \hat { \mathbf { c } } , \hat { \mathbf { H } } , \hat { \lambda } )$ , so that $\scriptstyle { \hat { \mathcal { F } } } _ { R }$ is defined as:

$$
\mathcal { \hat { F } } _ { R } ( x , v ) = \frac { 1 } { \hat { \lambda } ( x ) } \left( \sqrt { \hat { \lambda } ( x ) \| v \| _ { \hat { \mathbf { H } } } ^ { 2 } + \langle \hat { \mathbf { c } } ( x ) , v \rangle _ { \hat { \mathbf { H } } } ^ { 2 } } - \langle \hat { \mathbf { c } } ( x ) , v \rangle _ { \hat { \mathbf { H } } } \right) .
$$

Lemma $4 9 . \ J f J ( x )$ has full column rank, then $\hat { \mathbf { H } } ^ { - 1 }$ and cˆ are the pushforwards of ${ \bf H } ^ { - 1 }$ and c by the Jacobian $J ( x ) , i . e .$

$$
\hat { \mathbf { H } } ^ { - 1 } ( x ) = c _ { 2 } J ( x ) \mathbf { H } ^ { - 1 } ( x ) J ( x ) ^ { \top } , \qquad \hat { \mathbf { c } } ( x ) = \sqrt { c _ { 2 } } J ( x ) \mathbf { c } ( x ) , \qquad \hat { \mathbf { H } } ( x ) = \frac { 1 } { c _ { 2 } } J ( x ) ^ { + \top } \mathbf { H } ( x ) J ( x ) ^ { + } ,
$$

and $\hat { \lambda } ( x ) = \lambda ( x )$

In particular, this lemma implies that $\| \hat { \mathbf { c } } ( x ) \| _ { \hat { \mathbf { H } } } = \| \mathbf { c } ( x ) \| _ { \mathbf { H } } ,$ , so that the admissibility condition $\| \hat { \mathbf { c } } ( x ) \| _ { \hat { \mathbf { H } } } < 1$ is equivalent to $\| \mathbf { c } ( x ) \| _ { \mathbf { H } } < \bar { 1 }$ , which is exactly the hypothesis of Definition 5. Hence, if $J ( x )$ has full column rank, then $\scriptstyle { \hat { \mathcal { F } } } _ { R }$ is a valid Randers metric on the embedded tangent space $J ( T _ { x } { \mathcal { M } } )$

Proof. By Proposition 7, $\Gamma ( x ) = c _ { 2 } J ( x ) g _ { \mathrm { B L } } ( x ) ^ { - 1 } J ( x ) ^ { \mathsf { T } }$ , and $V ( x ) V ( x ) ^ { \mathsf { T } } = c _ { 1 } ^ { 2 } J ( x ) \mathbf { c } ( x ) \mathbf { c } ( x ) ^ { \mathsf { T } } J ( x ) ^ { \mathsf { T } }$ since $V ( x ) = c _ { 1 } J ( x ) \mathbf { c } ( x )$ . Substituting into the definition of $\hat { \mathbf { H } } ^ { - 1 } ( x )$ and using $\mathbf { H } ( x ) ^ { - 1 } = g _ { \mathrm { B L } } ( x ) ^ { - 1 } - ( m +$ $2 ) \dot { \mathbf { c } } ( \dot { \boldsymbol { x } } ) \mathbf { c } ( \boldsymbol { x } ) ^ { \top }$ (Definition 5),

$$
\begin{array} { r l } & { \hat { \mathbf { H } } ^ { - 1 } ( x ) = c _ { 2 } J ( x ) g _ { \mathrm { B L } } ( x ) ^ { - 1 } J ( x ) ^ { \mathsf { T } } - ( m + 2 ) c _ { 2 } J ( x ) \mathbf { c } ( x ) \mathbf { c } ( x ) ^ { \mathsf { T } } J ( x ) ^ { \mathsf { T } } } \\ & { \quad \quad \quad = c _ { 2 } J ( x ) \left( g _ { \mathrm { B L } } ( x ) ^ { - 1 } - ( m + 2 ) \mathbf { c } ( x ) \mathbf { c } ( x ) ^ { \mathsf { T } } \right) J ( x ) ^ { \mathsf { T } } } \\ & { \quad \quad \quad = c _ { 2 } J ( x ) \mathbf { H } ( x ) ^ { - 1 } J ( x ) ^ { \mathsf { T } } , } \end{array}
$$

and, similarly, $\begin{array} { r } { \hat { \mathbf { c } } ( x ) = \frac { \sqrt { c _ { 2 } } } { c _ { 1 } } V ( x ) = \frac { \sqrt { c _ { 2 } } } { c _ { 1 } } c _ { 1 } J ( x ) \mathbf { c } ( x ) = \sqrt { c _ { 2 } } J ( x ) \mathbf { c } ( x ) } \end{array}$

For $\hat { \mathbf { H } } ( x )$ : the identity $\hat { \mathbf { H } } ^ { - 1 } ( x ) = c _ { 2 } J ( x ) \mathbf { H } ( x ) ^ { - 1 } J ( x ) ^ { \mathsf { T } }$ just established has exactly the same form as $\Gamma ( x ) =$ $c _ { 2 } J ( x ) g _ { \mathrm { B L } } ^ { ' } ( x ) ^ { - 1 } J ( x ) ^ { \dagger }$ , with $g _ { \mathrm { B L } } ( x ) ^ { - 1 }$ replaced by $\dot { \mathbf { H } ( x ) } ^ { - 1 }$ . Since $\mathbf { H } ( x ) ^ { - 1 } \succsim 0$ (Proposition 6) and ${ \dot { J } } ( x )$ has full column rank, the proof of Corollary 48 applies verbatim under this substitution, giving ${ \hat { \mathbf { H } } } ( x ) =$ $\begin{array} { r } { \left( \hat { \mathbf { H } } ^ { - 1 } ( x ) \right) ^ { + } = c _ { 2 } ^ { - 1 } J ( x ) ^ { + \top } \mathbf { H } ( x ) J ( x ) ^ { + } } \end{array}$

It remains to compare the two norms. Using $J ( x ) ^ { + } J ( x ) = I _ { m }$

$$
\| \hat { \mathbf { c } } ( x ) \| _ { \hat { \mathbf { H } } } ^ { 2 } = c _ { 2 } \mathbf { c } ( x ) ^ { \mathsf { T } } J ( x ) ^ { \mathsf { T } } \cdot \frac { 1 } { c _ { 2 } } J ( x ) ^ { + \mathsf { T } } \mathbf { H } ( x ) J ( x ) ^ { + } \cdot J ( x ) \mathbf { c } ( x ) = \mathbf { c } ( x ) ^ { \mathsf { T } } \mathbf { H } ( x ) \mathbf { c } ( x ) = \| \mathbf { c } ( x ) \| _ { \mathbf { H } } ^ { 2 } ,
$$

so that $\hat { \lambda } ( x ) = 1 - \| \hat { \mathbf { c } } ( x ) \| _ { \hat { \mathbf { H } } } ^ { 2 } = 1 - \| \mathbf { c } ( x ) \| _ { \mathbf { H } } ^ { 2 } = \lambda ( x ) .$

Linking the two Randers metrics $\mathcal { F } _ { R }$ and $\scriptstyle { \hat { \mathcal { F } } } _ { R }$ together results in Proposition 9, which we restate here for convenience:

Proposition 9. $I f \| \hat { \mathbf { c } } ( x ) \| _ { \hat { g } _ { B L } } < 1 / \sqrt { m + 3 }$ and J is offull column rank, then $\scriptstyle { \hat { \mathcal { F } } } _ { R }$ is related to the original moment-Randers metric $\mathcal { F } _ { R }$ by

$$
\hat { \mathcal { F } } _ { R } ( x , J ( x ) v ) = \frac { 1 } { \sqrt { c _ { 2 } } } \mathcal { F } _ { R } ( x , v ) , \quad \forall v \in T _ { x } \mathcal { M } .
$$

In particular, $\scriptstyle { \hat { \mathcal { F } } } _ { R }$ is a valid Randers metric on $J ( T _ { x } { \mathcal { M } } )$

Proof of Proposition 9. By Corollary 48 and Proposition 6 the assumption $\| \hat { \mathbf { c } } ( x ) \| _ { \hat { g } _ { B L } } < ( m + 3 ) ^ { - 1 / 2 }$ is equivalent to $\| \mathbf { c } ( x ) \| _ { \mathbf { H } } < \bar { 1 }$ , so that $\mathcal { F } _ { R }$ is a valid Randers metric at x. By Lemma 49 and ${ \mathrm { \Delta } } J ^ { + } J = I _ { m }$ , for every $v \in T _ { x } { \mathcal { M } } .$

$$
\| J v \| _ { \dot { \mathbf { H } } } ^ { 2 } = \frac { 1 } { c _ { 2 } } \| v \| _ { \mathbf { H } } ^ { 2 } , \qquad \langle \hat { \mathbf { c } } , J v \rangle _ { \hat { \mathbf { H } } } = \sqrt { c _ { 2 } } \mathbf { c } ^ { \top } J ^ { \top } \frac { 1 } { c _ { 2 } } J ^ { + \top } \mathbf { H } J ^ { + } J v = \frac { 1 } { \sqrt { c _ { 2 } } } \langle \mathbf { c } , v \rangle _ { \mathbf { H } } .
$$

Substituting these and $\hat { \lambda } = \lambda$ into Equation (43),

$$
\hat { \mathcal { F } } _ { R } ( x , J ( x ) v ) = \frac { 1 } { \lambda } \left( \sqrt { \frac { \lambda } { c _ { 2 } } \| v \| _ { \mathbf { H } } ^ { 2 } + \frac { 1 } { c _ { 2 } } { \langle \mathbf { c } , v \rangle _ { \mathbf { H } } ^ { 2 } } } - \frac { 1 } { \sqrt { c _ { 2 } } } \langle \mathbf { c } , v \rangle _ { \mathbf { H } } \right) = \frac { 1 } { \sqrt { c _ { 2 } } } \mathcal { F } _ { R } ( x , v ) .
$$

In other words, $\hat { \mathcal { F } } _ { R } = c _ { 2 } ^ { - 1 / 2 } \Psi _ { * } \mathcal { F } _ { R }$ on the embedded tangent space $J ( x ) ( T _ { x } { \mathcal { M } } )$ : the moment-Randers construction commutes with the embedding, which is what makes it computable. Note that $\scriptstyle { \hat { \mathcal { F } } } _ { R }$ is a Randers metric on $J ( x ) ( T _ { x } { \mathcal { M } } )$ only, and not on the whole of $\mathbb { R } ^ { \ell }$ , since $\hat { \bf H } ^ { - 1 }$ has rank $m$

Discrete setting. All the quantities above are evaluated from the discrete operators of Equation $( 8 )$ by replacing $\mathcal { L } ^ { s }$ and ${ \mathcal { L } } ^ { a }$ with $\dot { \bf L } ^ { s }$ and ${ \bf L } ^ { a }$ , and Ψ with the embedded samples $Y _ { i } \overset { \cdot } { = } \Psi ( X _ { i } )$ . Writing $\psi _ { k } \in \mathbb { R } ^ { \tilde { N } }$ for the k-th embedding coordinate evaluated at the samples and ⊙ for the entrywise product,

$$
V _ { i } = \mathbf { L } ^ { a } [ \Psi ] _ { i } , \qquad \Gamma _ { i } ^ { k l } = \frac { 1 } { 2 } \Bigl [ \bigl ( \mathbf { L } ^ { s } ( \psi _ { k } \odot \psi _ { l } ) \bigr ) _ { i } - \psi _ { k } \bigl ( X _ { i } \bigr ) \bigl ( \mathbf { L } ^ { s } \psi _ { l } \bigr ) _ { i } - \psi _ { l } \bigl ( X _ { i } \bigr ) \bigl ( \mathbf { L } ^ { s } \psi _ { k } \bigr ) _ { i } \Bigr ] ,
$$

and $\begin{array} { r } { \| \mathbf { c } ( X _ { i } ) \| _ { g _ { \mathrm { B L } } } ^ { 2 } = \frac { c _ { 2 } } { c _ { 1 } ^ { 2 } } V _ { i } ^ { \mathsf { T } } \Gamma _ { i } ^ { + } V _ { i } } \end{array}$ by Corollary 48. Whenever this quantity is below $( m + 3 ) ^ { - 1 }$ , Equation (44) defines $\hat { \mathbf { c } } _ { i } , \hat { \mathbf { H } } _ { i }$ and $\hat { \lambda } _ { i }$ , and hence the metric $\hat { \mathcal { F } } _ { R } ( X _ { i } , \cdot )$ at the sample $X _ { i }$ . By Theorem $4 , \mathbf { L } ^ { s }$ and ${ \bf L } ^ { a }$ converge uniformly and almost surely to $\mathcal { L } ^ { s }$ and ${ \mathcal { L } } ^ { a }$ , so these empirical quantities converge to their continuous counterparts at every sample.

## D.9 COMPUTATION OF THE KERNEL MOMENTS

The values of the kernel moments $\mu _ { n }$ can be computed analytically for the usual kernel functions $K .$

EXPONENTIAL KERNEL

If K is the exponential kernel $K ( x ) = \exp ( - x )$ , then we can compute, for m $\in \mathbb { N } \mathrm { : }$

$$
\mu _ { n } = \int _ { 0 } ^ { + \infty } K ( r ) r ^ { n + m - 1 } \mathrm { d } r = \int _ { 0 } ^ { + \infty } e ^ { - r } r ^ { n + m - 1 } \mathrm { d } r = \Gamma ( n + m ) = ( n + m - 1 ) !
$$

GAUSSIAN KERNEL

If K is the Gaussian kernel $K ( x ) = \exp ( - x ^ { 2 } )$ , then we can compute, for $m \in \mathbb { N }$ , using the change of variable $u = r ^ { 2 }$ , so that $\mathrm { d } r = \mathrm { d } u / ( 2 \sqrt { u } ) ;$

$$
\begin{array} { l } { \displaystyle \mu _ { n } = \int _ { 0 } ^ { + \infty } K ( r ) r ^ { n + m - 1 } \mathrm { d } r = \int _ { 0 } ^ { + \infty } e ^ { - r ^ { 2 } } r ^ { n + m - 1 } \mathrm { d } r } \\ { \displaystyle \qquad = \frac { 1 } { 2 } \int _ { 0 } ^ { + \infty } e ^ { - u } u ^ { ( n + m - 1 ) / 2 - 1 / 2 } \mathrm { d } u } \\ { \displaystyle \qquad = \frac { 1 } { 2 } \int _ { 0 } ^ { + \infty } e ^ { - u } u ^ { ( n + m ) / 2 - 1 } \mathrm { d } u } \\ { \displaystyle \qquad = \frac { 1 } { 2 } \Gamma \left( \frac { n + m } { 2 } \right) } \end{array}
$$