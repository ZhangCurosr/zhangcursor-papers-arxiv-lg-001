# Grassmann–Plücker Parametrization of Convolutional Filter Subspaces: Regularity and Closed Embeddings

H. Yuan<sup>∗</sup> and H. Zuo <sup>†</sup>

## Abstract.

We propose a geometric parametrization of the filters in a single convolutional layer: the parameter is no longer an ordered family of filter vectors, but a fixed-dimensional subspace of the filter space. For one-dimensional finite-stride convolution, the filter-to-convolution-operator correspondence gives an injective linear map $\mathcal { C } : \mathcal { K }  H$ . This map sends filter subspaces in $\operatorname { G r } ( q , \kappa )$ to operator subspaces in $\operatorname { G r } ( q , H ) ;$ composing it with the Plücker embedding yields a projective parametrization $\Phi : \operatorname { G r } ( q , K ) \to \mathbb { P } ( \land ^ { q } H )$ . Using $T _ { U } \operatorname { G r } ( q , K ) \cong \operatorname { H o m } ( U , K / U )$ , we compute the diferential of $\Gamma _ { C }$ and prove that the diferential of Φ is injective at every point. We then use the vanishing equations for Plücker coordinates and the standard afine coordinates on a Grassmannian to prove that the sub-Grassmannian $\operatorname { G r } ( q , { \mathcal { C } } ( { K } ) ) \hookrightarrow \operatorname { G r } ( q , H )$ is a closed embedding, and hence that Φ is a closed embedding. Consequently, the parameter space is isomorphic to its projective image, the parametrization is finite and birational onto its image, every fiber is a singleton, and the resulting projective neural variety is smooth. For the first nontrivial case $k = 4$ and $q = 2$ , we also use Singular to eliminate the source Plücker coordinates and recover the image ideal directly, checking its dimension, degree, chart rank, and smoothness. This symbolic computation is a low-dimensional illustration rather than a substitute for the general proof. Finally, we discuss possible connections with filter redundancy and low-rank convolution, while distinguishing the geometric results proved here from application proposals that still require numerical validation.

Keywords. convolutional neural networks, Grassmannian, Plücker embedding, neural variety, closed embedding, finite birational map, low-rank representation.

MSC(2020). 14M15, 68T07.

1. Introduction. We study a projective parametrization of a class of subspaces of singlelayer convolution operators. A traditional single-filter parameter is a vector in the filter space, whereas here the parameter is generalized to a fixed-dimensional subspace of that space. Let K be the filter space and let H be the space of linear operators from the input space to the output space. Convolution gives a linear map

$$
{ \mathcal { C } } : { \mathcal { K } } \longrightarrow H .
$$

For $U \in \operatorname { G r } ( q , { \mathcal { K } } )$ , its image ${ \mathcal { C } } ( U )$ is a linear subspace of H. If C is injective, then dim ${ \mathcal { C } } ( U ) = q$ and hence there is a map

$$
\Gamma _ { \mathcal { C } } : \mathrm { G r } ( q , K ) \longrightarrow \mathrm { G r } ( q , H ) , \qquad U \longmapsto { \mathcal { C } } ( U ) .
$$

Composing it with the Plücker embedding gives the parametrization studied in this paper:

$$
\Phi = \mathrm { P l } _ { H } \circ \Gamma _ { \mathcal { C } } : \mathrm { G r } ( { \boldsymbol { q } } , { \boldsymbol { \mathcal { K } } } ) \longrightarrow \mathbb { P } ( { \boldsymbol { \wedge } } ^ { q } H ) .
$$

The body of the paper addresses three questions in order. First, we construct C directly from the finite-stride convolution formula and prove that C is injective. Second, we compute the diferentials of $\Gamma _ { C }$ and Φ and prove that the diferential of Φ is injective at every point. Third, setting $W = { \mathcal { C } } ( K )$ , we use Plücker coordinates to prove that the natural inclusion

$$
\mathrm { G r } ( q , W ) \longrightarrow \mathrm { G r } ( q , H )
$$

is a closed embedding, from which the closed-embedding property of Φ follows. Finiteness, birationality, uniqueness of fibers, and smoothness of the image are then consequences of the closed-embedding theorem. In addition, for the concrete choice $k = 4 , q = 2 , d = 5 , d ^ { \prime } = 2 , s =$ 1, we use Singular to eliminate the six source Plücker coordinates from the graph ideal, obtain the homogeneous ideal of the projective image, and compare the computation term by term with the main theorem.

The main result is stated explicitly as follows.

Theorem 1.1 (Main theorem). Let $\ b { K } = \mathbb { C } ^ { k }$ , let $H = \operatorname { H o m } ( E , F )$ , and let ${ \mathcal { C } } : { \mathcal { K } } \to H$ be the linear map induced by the finite-stride convolution in Definition 2.1. For every $1 \leq q \leq k$ , the parametrization

$$
\Phi = \mathrm { P l } _ { H } \circ \Gamma _ { \mathcal { C } } : \mathrm { G r } ( q , \mathcal { K } ) \longrightarrow \mathbb { P } ( \wedge ^ { q } H )
$$

is a closed embedding. If $X = \operatorname { G r } ( q , \mathcal { K } )$ and $Y = \Phi ( X )$ , then $\Phi : X { \stackrel { \sim } { \to } } Y$ is an isomorphism. In particular, Φ is finite and birational onto its image, $\# \Phi ^ { - 1 } ( y ) = 1$ for every $y \in Y$ , and

$$
\dim Y = q ( k - q ) , \qquad \operatorname { S i n g } ( Y ) = \emptyset .
$$

The closed-embedding assertion in the main theorem is proved in Theorem 5.5; its finiteness, birationality, uniqueness of fibers, dimension, and smoothness statements are established in Corollaries 6.2–6.5.

The family of functions realized by a neural network architecture can be studied as the image of a parameter-to-function map. In the related literature, such an image is often called a neuromanifold, or a neural variety in the algebraic setting [1, 8, 11]. Function spaces, singularities, and critical points of loss functions for linear convolutional networks have been studied from an algebraic-geometric perspective [8, 9]. Shahverdi, Marchetti, and Kohn further studied polynomial convolutional networks with monomial activation: their projective parametrization factors through a Segre–Veronese embedding and is regular and finite birational; they also discuss the dimension, degree, and singularities of the neural variety and critical points of a regression loss [11, Secs. 3.1, 4.1, and 4.2]. We adopt their notation for one-dimensional convolution, but study a diferent object: a higher-dimensional filter subspace is the single-layer parameter, and we record the complete Plücker coordinates of the corresponding operator subspace.

We consider only this single-layer linear subspace of convolution operators and its complete Plücker representation. Our theorems make no claims about nonlinear activations, multilayer compositions, loss functions, or performance in actual training.

## 2. One-Dimensional Convolution and the Grassmann–Plücker Parametrization.

2.1. Finite-stride convolution. All vector spaces in this paper are finite-dimensional complex vector spaces. Fix positive integers $k , d , d ^ { \prime }$ , and $s ,$ and assume that

$$
d = s ( d ^ { \prime } - 1 ) + k .\tag{2.1}
$$

Here k is the filter length, d is the input length, s is the stride, and $d ^ { \prime }$ is the output length. Set

$$
\begin{array} { r } { \mathcal { K } = \mathbb { C } ^ { k } , \qquad E = \mathbb { C } ^ { d } , \qquad F = \mathbb { C } ^ { d ^ { \prime } } . } \end{array}
$$

We index coordinates from 0. This convention agrees with the one-dimensional valid convolution used in the literature on polynomial convolutional networks [11, Sec. 3.1, Eq. (1)].

Definition 2.1 (Finite-stride convolution). For a filter

$$
w = ( w _ { 0 } , \dotsc , w _ { k - 1 } ) \in { \mathcal { K } }
$$

and an input

$$
x = ( x _ { 0 } , \ldots , x _ { d - 1 } ) \in E ,
$$

define $C _ { w } x \in F$ by

$$
( C _ { w } x ) _ { i } = \sum _ { j = 0 } ^ { k - 1 } w _ { j } x _ { s i + j } , \qquad 0 \leq i \leq d ^ { \prime } - 1 .\tag{2.2}
$$

By (2.1), $i f 0 \leq i \leq d ^ { \prime } - 1$ and $0 \leq j \leq k - 1$ , then

$$
0 \leq s i + j \leq s ( d ^ { \prime } - 1 ) + k - 1 = d - 1 ,
$$

so every input coordinate in (2.2) is defined.

Write

$$
H = \operatorname { H o m } ( E , F ) .
$$

Convolution induces the map

$$
{ \mathcal { C } } : { \mathcal { K } } \longrightarrow H , \qquad w \longmapsto C _ { w } .\tag{2.3}
$$

Proposition 2.2 (Linearity and injectivity of the convolution map). The map $\mathcal { C }$ defined in (2.3) is an injective linear map.

Proof. Let $a , b \in \mathbb { C } , w , w ^ { \prime } \in \mathcal { K }$ , and $x \in E$ . For every $0 \leq i \leq d ^ { \prime } - 1$ ，

$$
\begin{array} { l } { \displaystyle \left( C _ { a w + b w ^ { \prime } } x \right) _ { i } = \sum _ { j = 0 } ^ { k - 1 } ( a w _ { j } + b w _ { j } ^ { \prime } ) x _ { s i + j } } \\ { \displaystyle = a \sum _ { j = 0 } ^ { k - 1 } w _ { j } x _ { s i + j } + b \sum _ { j = 0 } ^ { k - 1 } w _ { j } ^ { \prime } x _ { s i + j } } \\ { \displaystyle = \left( a C _ { w } x + b C _ { w ^ { \prime } } x \right) _ { i } . } \end{array}
$$

Thus $C _ { a w + b w ^ { \prime } } = a C _ { w } + b C _ { w ^ { \prime } }$ , so C is linear.

We next prove injectivity. Let $w \in$ ker ${ \mathcal { C } } ,$ so that $C _ { w } = 0$ . For every $0 \leq j \leq k - 1$ , let $e _ { j } \in E$ be the jth standard basis vector. By (2.2), the zeroth output coordinate satisfies

$$
( C _ { w } e _ { j } ) _ { 0 } = \sum _ { \ell = 0 } ^ { k - 1 } w _ { \ell } ( e _ { j } ) _ { \ell } = w _ { j } .
$$

Because $C _ { w } = 0$ , we have $( C _ { w } e _ { j } ) _ { 0 } = 0$ , and therefore $w _ { j } = 0$ . This holds for every $0 \leq j \leq k - 1$ ， so $w = 0$ . Hence

$$
\ker C = \{ 0 \} { \mathrm { . } }
$$

Remark 2.3. In the concrete convolution model of Definition 2.1, the injectivity of C is a conclusion of Proposition 2.2, not an additional assumption. The geometric theorems below are stated for arbitrary injective linear maps, but when they are applied to the convolution considered here, the required injectivity has already been proved.

## 2.2. Filter-subspace parameters.

Definition 2.4 (Grassmannian). Let V be an n-dimensional complex vector space, and let $1 \leq q \leq n$ . Write

$$
\operatorname { G r } ( q , V ) = \{ U \subseteq V : U \ i s \ a \ q - d i m e n s i o n a l \ l i n e a r \ s u b s p a c e \} .
$$

Endowed with its standard algebraic-variety structure, this is the Grassmannian of q-planes in $V .$

Fix $1 \leq q \leq k$ . We take

$$
X = \operatorname { G r } ( q , \mathcal { K } )
$$

as the parameter space. A point $U \in X$ represents a q-dimensional subspace of the filter space.

Definition 2.5 (Convolution-induced Grassmann map). Define

$$
\Gamma _ { \mathcal { C } } : \mathrm { G r } ( q , K ) \longrightarrow \mathrm { G r } ( q , H ) , \qquad U \longmapsto \mathcal { C } ( U ) .\tag{2.4}
$$

Lemma 2.6 (Well-definedness of $\Gamma _ { \mathit { C } } )$ . For every $U \in \operatorname { G r } ( q , { \mathcal { K } } )$ , one has dim ${ \mathcal { C } } ( U ) = q$ Hence (2.4) indeed takes values in $\operatorname { G r } ( q , H )$

Proof. By Proposition 2.2, the restriction

$$
{ \mathcal { C } } | _ { U } : U \longrightarrow H
$$

is injective. The rank–nullity theorem gives

$$
\dim { \mathcal { C } } ( U ) = \dim U - \dim \ker ( { \mathcal { C } } | _ { U } ) = q - 0 = q .
$$

Thus ${ \mathcal { C } } ( U ) \in \operatorname { G r } ( q , H )$

## 2.3. The Plücker parametrization.

Definition 2.7 (Plücker map). Let V be a finite-dimensional complex vector space. For

$$
U = \operatorname { s p a n } ( u _ { 1 } , \dots , u _ { q } ) \in \operatorname { G r } ( q , V ) ,
$$

define

$$
\operatorname { P l } _ { V } ( U ) = [ u _ { 1 } \wedge \cdots \wedge u _ { q } ] \in \mathbb { P } ( \wedge ^ { q } V ) .\tag{2.5}
$$

If $v _ { 1 } , \ldots , v _ { q }$ is another basis of $U ,$ then there exists $A \in \operatorname { G L } _ { q } ( \mathbb { C } )$ such that

$$
( v _ { 1 } , \ldots , v _ { q } ) = ( u _ { 1 } , \ldots , u _ { q } ) A .
$$

Alternating multilinearity of the exterior product gives

$$
v _ { 1 } \wedge \cdot \cdot \cdot \wedge v _ { q } = \operatorname* { d e t } ( A ) u _ { 1 } \wedge \cdot \cdot \cdot \wedge u _ { q } .
$$

Since $\operatorname* { d e t } ( A ) \ \neq \ 0$ , the two vectors determine the same projective point, and thus (2.5) is independent of the choice of basis.

Definition 2.8 (Grassmann–Plücker parametrization and neural variety). Define

$$
\Phi = \mathrm { P l } _ { H } \circ \Gamma _ { \mathcal { C } } : \mathrm { G r } ( { \boldsymbol { q } } , { \boldsymbol { \mathcal { K } } } ) \longrightarrow \mathbb { P } ( { \boldsymbol { \wedge } } ^ { q } H ) .\tag{2.6}
$$

Its image

$$
Y = \Phi { \bigl ( } \mathrm { G r } ( q , K ) { \bigr ) }\tag{2.7}
$$

is called the Grassmann–Plücker neural variety of this single-layer convolutional model.

If $u _ { 1 } , \ldots , u _ { q }$ is a basis of U, then

$$
\Phi ( U ) = \left[ \mathcal { C } ( u _ { 1 } ) \wedge \cdot \cdot \cdot \wedge \mathcal { C } ( u _ { q } ) \right] .\tag{2.8}
$$

The linear map C induces a linear map

$$
{ \textstyle \bigwedge } ^ { q } { \mathcal { C } } : { \textstyle \bigwedge } ^ { q } { \mathcal { K } } \longrightarrow { \textstyle \bigwedge } ^ { q } H ,
$$

whose value on a decomposable vector is

$$
( \land ^ { q } { \mathcal { C } } ) ( u _ { 1 } \land \cdots \land u _ { q } ) = { \mathcal { C } } ( u _ { 1 } ) \land \cdots \land { \mathcal { C } } ( u _ { q } ) .
$$

Because C is injective, $\wedge ^ { q } { \mathcal { C } }$ is also injective and can be projectivized to give

$$
\mathbb { P } ( \wedge ^ { q } \mathcal { C } ) : \mathbb { P } ( \wedge ^ { q } \mathcal { K } ) \longrightarrow \mathbb { P } ( \wedge ^ { q } H ) .
$$

Equation (2.8) gives the commutative identity

$$
\boxed { \mathrm { P l } _ { H } \circ \Gamma _ { \mathcal { C } } = \mathbb { P } ( \wedge ^ { q } \mathcal { C } ) \circ \mathrm { P l } _ { \mathcal { K } } . }\tag{2.9}
$$

Thus “first send the filter subspace to the operator space and then apply the Plücker embedding” and “first take the Plücker coordinates of the filter subspace and then apply the linear map induced on the exterior power” give the same parametrization.

3. Local Coordinates on Grassmannians and the Plücker Embedding. This section recalls the standard facts about Grassmannians, tangent spaces, and the Plücker embedding that will be needed below. We give direct proofs of the local-coordinate and tangent-space statements used later, while citing the classical Plücker closed-embedding theorem. For background, see Harris [5, Lecture 6, pp. 63–67]; for the quotient-space representation, horizontal tangent spaces, and numerical matrix models of the Grassmann manifold, see Edelman, Arias, and Smith [3, Secs. 2.3.2 and 2.5].

3.1. Standard afine coordinates on a Grassmannian. Let V be an n-dimensional complex vector space with fixed ordered basis $e _ { 1 } , \ldots , e _ { n }$ . For a q-element index set

$$
I = \{ i _ { 1 } < \cdots < i _ { q } \} \subseteq \{ 1 , \ldots , n \} ,
$$

set

$$
V _ { I } = \operatorname { s p a n } ( e _ { i _ { 1 } } , \ldots , e _ { i _ { q } } ) , \qquad V _ { I ^ { c } } = \operatorname { s p a n } ( e _ { j } : j \notin I ) .
$$

Then $V = V _ { I } \oplus V _ { I ^ { c } }$ . Let

$$
\pi _ { I } : V \longrightarrow V _ { I }
$$

denote the projection along $V _ { I ^ { c } }$

Definition 3.1 (Standard Grassmann open set). Define

$$
\mathcal { U } _ { I } = \{ U \in \operatorname { G r } ( q , V ) : \pi _ { I } | _ { U } : U \to V _ { I } \ i s \ a n \ i s o m o r p h i s m \} .
$$

Proposition 3.2 (Graph coordinates). The map

$$
\mathrm { H o m } ( V _ { I } , V _ { I ^ { c } } ) \longrightarrow { \mathcal U } _ { I } , \qquad A \longmapsto \mathrm { G r a p h } ( A )
$$

is an isomorphism of afine varieties, where

$$
{ \mathrm { G r a p h } } ( A ) = \{ u + A ( u ) : u \in V _ { I } \} .
$$

In particular,

$$
\mathcal { U } _ { I } \cong \mathbb { C } ^ { q ( n - q ) } .
$$

Proof. If $A \in$ Hom $( V _ { I } , V _ { I ^ { c } } )$ , then

$$
\pi _ { I } ( u + A ( u ) ) = u
$$

for every $u \in V _ { I }$ . Hence

$$
\pi _ { I } | _ { \mathrm { G r a p h } ( A ) } : \mathrm { G r a p h } ( A ) \longrightarrow V _ { I }
$$

has inverse $u \mapsto u + A ( u )$ , and therefore $\mathrm { { G r a p h } } ( A ) \in { \mathcal { U } } _ { I }$

Conversely, if $U \in \mathcal { U } _ { I }$ , define

$$
A _ { U } = \pi _ { I ^ { c } } \circ ( \pi _ { I } | _ { U } ) ^ { - 1 } : V _ { I } \longrightarrow V _ { I ^ { c } } ,
$$

where $\pi _ { I ^ { c } } : V  V _ { I ^ { c } }$ is the projection along $V _ { I }$ . For $v \in U ,$ , put $u = \pi _ { I } ( v )$ . Then

$$
v = u + \pi _ { I ^ { c } } ( v ) = u + A _ { U } ( u ) ,
$$

and therefore $U = \mathrm { G r a p h } ( A _ { U } )$ . By construction, $A \mapsto { \mathrm { G r a p h } } ( A )$ and $U \mapsto A _ { U }$ are inverse maps.

With respect to the fixed bases, the matrix entries of A give $q ( n - q )$ afine coordinates. In these coordinates the two maps above are given, respectively, by the matrix entries and their identical recovery; hence both are regular. ■

Corollary 3.3 (Dimension and smoothness). $\operatorname { G r } ( q , V )$ is a smooth projective variety of $d i -$ mension $q ( n - q )$

Proof. The standard open sets $\mathcal { U } _ { I }$ cover $\operatorname { G r } ( q , V )$ , and by Proposition 3.2, each $\mathcal { U } _ { I }$ is isomorphic to $\mathbb { C } ^ { q ( n - q ) }$ . Thus every point of $\operatorname { G r } ( q , V )$ has an open neighborhood isomorphic to a smooth afine space, so $\operatorname { G r } ( q , V )$ is smooth of dimension $q ( n - q )$ . Its projectivity follows from the Plücker closed embedding in Theorem 3.6. ■

Lemma 3.4 (Irreducibility of the Grassmannian). $\operatorname { G r } ( q , V )$ is an irreducible algebraic variety.

Proof. Fix $U _ { 0 } \in \mathrm { G r } ( q , V )$ . The algebraic group $\operatorname { G L } ( V )$ acts on $\operatorname { G r } ( q , V )$ by

$$
( g , U ) \longmapsto g ( U ) .
$$

For an arbitrary $U \in \operatorname { G r } ( q , V )$ , choose bases of $U _ { 0 }$ and $U$ and extend each to a basis of $V .$ There is then some $g \in \operatorname { G L } ( V )$ with $g ( U _ { 0 } ) = U$ . Consequently, the orbit map

$$
\mathrm { G L } ( V ) \longrightarrow \mathrm { G r } ( q , V ) , \qquad g \longmapsto g ( U _ { 0 } )
$$

is surjective. The group $\operatorname { G L } ( V )$ is the nonempty principal open subset of the afine space $\operatorname { E n d } ( V )$ defined by det $\neq 0$ , so it is irreducible. The image of an irreducible space under a continuous map is irreducible, and hence $\operatorname { G r } ( q , V )$ is irreducible.

## 3.2. The tangent space.

Theorem 3.5 (Tangent space of a Grassmannian). For every $U \in \operatorname { G r } ( q , V )$ , there is a natural linear isomorphism

$$
T _ { U } \operatorname { G r } ( q , V ) \cong \operatorname { H o m } ( U , V / U ) .\tag{3.1}
$$

Proof. Choose a complementary subspace $L \subseteq V$ such that

$$
V = U \oplus L .
$$

By Proposition 3.2, an open neighborhood of $U$ is isomorphic to ${ \mathrm { H o m } } ( U , L )$ , with $U$ corresponding to the zero map. Therefore

$$
T _ { U } \operatorname { G r } ( q , V ) \cong T _ { 0 } \operatorname { H o m } ( U , L ) = \operatorname { H o m } ( U , L ) .
$$

If $\pi : V \to V / U$ is the quotient map, its restriction

$$
\pi | _ { L } : L \longrightarrow V / U
$$

is a linear isomorphism, and hence it induces

$$
\mathrm { H o m } ( U , L ) \stackrel { \sim } { \to } \mathrm { H o m } ( U , V / U ) , \qquad A \longmapsto \pi | _ { L } \circ A .
$$

The composition of these two isomorphisms gives (3.1).

We verify that this isomorphism is independent of the chosen complement. In any chosen graph coordinates, a tangent vector is represented by a first-order family of subspaces

$$
U _ { t } = \{ u + t { \widetilde A } ( u ) : u \in U \} ,
$$

where ${ \tilde { A } } : U \to V$ is linear. Its image in Hom $( U , V / U )$ is

$$
\begin{array} { r } { u \longmapsto [ \widetilde { A } ( u ) ] . } \end{array}
$$

If ${ \widetilde { A } } ^ { \prime }$ determines the same first-order family, then $\widetilde { A } ( u ) - \widetilde { A } ^ { \prime } ( u ) \in U _ { 1 }$ , so

$$
[ \widetilde { A } ( u ) ] = [ \widetilde { A } ^ { \prime } ( u ) ] \in V / U .
$$

Thus the resulting element is independent of both the lift and the complement.

3.3. Plücker coordinates and the classical embedding. Continue to use the fixed basis $e _ { 1 } , \ldots , e _ { n }$ of V. If a basis of $U \in \operatorname { G r } ( q , V )$ is arranged as the rows of a full-rank matrix

$$
M \in { \mathrm { M a t } } _ { q \times n } ( \mathbb { C } ) ,
$$

then for every q-element index set $I \subseteq \{ 1 , \ldots , n \}$ , define

$$
p _ { I } ( U ) = \operatorname* { d e t } ( M _ { I } ) ,\tag{3.2}
$$

where $M _ { I }$ is the $q \times q$ submatrix formed by the columns indexed by I. Replacing the basis matrix M by GM, where $G \in \operatorname { G L } _ { q } ( \mathbb { C } )$ , multiplies every $p _ { I }$ by det(G). Therefore

$$
[ p _ { I } ( U ) ] _ { | I | = q }
$$

is a well-defined system of projective coordinates, and it agrees with the exterior-product coordinates in Definition 2.7.

If an index is repeated, we set the corresponding $p _ { i _ { 1 } \cdots i _ { q } }$ equal to zero; interchanging two indices changes its sign. The following classical theorem is used as a cited result and is not reproved here; see [5, Lecture 6, pp. 63–67].

Theorem 3.6 (Plücker closed embedding). The Plücker map

$$
\operatorname { P l } _ { V } : \operatorname { G r } ( q , V ) \longrightarrow \operatorname { \mathbb { P } } ( \wedge ^ { q } V )
$$

is a closed embedding. Its image is the projective Grassmann variety, classically defined by the quadratic Plücker relations.

4. Regularity of the Grassmann–Plücker Parametrization.

## 4.1. Meaning of regularity.

Definition 4.1 (Regularity used in this paper). Let $f : X \to Z$ be a morphism from a smooth algebraic variety X to an algebraic variety Z. If the diferential

$$
\mathrm { d } f | _ { x } : T _ { x } X \longrightarrow T _ { f ( x ) } Z
$$

is injective for every $x \in X$ , then f is called a regular parametrization. Equivalently,

$$
\operatorname { r a n k } ( \mathrm { d } f | _ { x } ) = \dim X
$$

for every $x \in X$

Remark 4.2. In Definition 4.1, “regular” means that the diferential has maximal rank everywhere. This difers from the convention in algebraic geometry in which any morphism of algebraic varieties may be called a regular map. More precisely, we will prove that Φ is an immersion everywhere. We follow the use of “regular parametrization” in the literature on polynomial convolutional networks [11, Theorem 4.5].

## 4.2. The convolution-induced Grassmann map is a morphism.

Proposition 4.3. The map $\Gamma _ { \mathcal { C } } : \mathrm { G r } ( q , K ) \to \mathrm { G r } ( q , H )$ is a morphism of algebraic varieties.

Proof. Fix $U _ { 0 } \in \mathrm { G r } ( q , { \mathcal { K } } )$ and choose a complement L such that

$$
\mathcal { K } = U _ { 0 } \oplus L .
$$

Since C is injective,

$$
{ \mathcal { C } } ( K ) = { \mathcal { C } } ( U _ { 0 } ) \oplus { \mathcal { C } } ( L ) .
$$

Choose a subspace $R \subseteq H$ such that

$$
H = { \mathcal { C } } ( U _ { 0 } ) \oplus { \mathcal { C } } ( L ) \oplus R ,
$$

and set $Q = { \mathcal { C } } ( L ) \oplus R$ . Graph coordinates near $U _ { 0 }$ in the source Grassmannian are

$$
A \in { \mathrm { H o m } } ( U _ { 0 } , L ) \longmapsto { \mathrm { G r a p h } } ( A ) .
$$

For every such A,

$$
\begin{array} { r l } { \mathcal { C } ( \operatorname { G r a p h } ( A ) ) = \{ \mathcal { C } ( u ) + \mathcal { C } ( A u ) : u \in U _ { 0 } \} } \\ { = \operatorname { G r a p h } ( B _ { A } ) , } \end{array}
$$

where

$$
B _ { A } = { \mathcal { C } } | _ { L } \circ A \circ ( { \mathcal { C } } | _ { U _ { 0 } } ) ^ { - 1 } \in { \mathrm { H o m } } ( { \mathcal { C } } ( U _ { 0 } ) , Q ) .
$$

The map $A \mapsto B _ { A }$ is linear and hence regular in graph coordinates. Such graph-coordinate neighborhoods cover $\operatorname { G r } ( q , K )$ , so $\Gamma _ { \mathrm { { } } } c$ is a morphism.

By Theorem 3.6, Pl<sub>H</sub> is a morphism. Therefore $\Phi = \mathrm { P l } _ { H } \circ \Gamma _ { \mathcal { C } }$ is also a morphism.

4.3. Diferential of the convolution-induced Grassmann map. Fix $U \in \operatorname { G r } ( q , K )$ . Define a linear map on quotient spaces by

$$
{ \overline { { \mathcal { C } } } } _ { U } : K / U \longrightarrow H / { \mathcal { C } } ( U ) , \qquad [ v ] \longmapsto [ { \mathcal { C } } ( v ) ] .\tag{4.1}
$$

Lemma 4.4. The map $\overline { { \mathcal { C } } } _ { U }$ is well defined and injective.

Proof. If $[ v ] = [ v ^ { \prime } ]$ , then $v - v ^ { \prime } \in U$ , and therefore

$$
\begin{array} { r } { \mathcal { C } ( v ) - \mathcal { C } ( v ^ { \prime } ) = \mathcal { C } ( v - v ^ { \prime } ) \in \mathcal { C } ( U ) . } \end{array}
$$

Thus $[ { \mathcal C } ( v ) ] = [ { \mathcal C } ( v ^ { \prime } ) ]$ which proves well-definedness.

If ${ \overline { { \mathcal { C } } } } _ { U } ( [ v ] ) = 0$ , then ${ \mathcal { C } } ( v ) \in { \mathcal { C } } ( U )$ . Hence there exists $u \in U$ such that

$$
{ \mathcal { C } } ( v ) = { \mathcal { C } } ( u ) .
$$

It follows that ${ \mathcal { C } } ( v - u ) = 0$ . Since C is injective, $v - u = 0$ , so $v \in U$ and hence $[ v ] = 0$ . Thus $\overline { { \mathcal { C } } } _ { U }$ is injective. ■

Proposition 4.5 (Diferential formula). Under the natural isomorphisms

$$
T _ { U } \operatorname { G r } ( q , K ) \cong \operatorname { H o m } ( U , K / U )
$$

and

$$
T _ { \mathcal { C } ( U ) } \operatorname { G r } ( q , H ) \cong \operatorname { H o m } ( \mathcal { C } ( U ) , H / \mathcal { C } ( U ) ) ,
$$

one has, for every $A \in { \mathrm { H o m } } ( U , { \mathcal { K } } / U )$ ，

$$
\mathrm { d } \Gamma _ { } c | _ { U } ( A ) = \overline { { \mathcal { C } } } _ { U } \circ A \circ ( \mathcal { C } | _ { U } ) ^ { - 1 } .\tag{4.2}
$$

Proof. Choose a linear lift $\tilde { A } : U \to \kappa$ such that the quotient map $\pi _ { U } : { \mathcal { K } } \to { \mathcal { K } } / U$ satisfies

$$
\pi _ { U } \circ { \tilde { A } } = A .
$$

By the graph-coordinate description in Theorem 3.5, A is represented by the first-order family of subspaces

$$
U _ { t } = \{ u + t { \widetilde A } ( u ) : u \in U \} .
$$

Applying $\Gamma _ { \mathrm { { } } } c$ gives

$$
\begin{array} { r l } & { \Gamma _ { \mathcal { C } } ( U _ { t } ) = \mathcal { C } ( U _ { t } ) } \\ & { \qquad = \{ \mathcal { C } ( u ) + t \mathcal { C } ( \widetilde { A } ( u ) ) : u \in U \} . } \end{array}
$$

Thus the tangent vector in the target Grassmannian maps ${ \mathcal { C } } ( u ) \in { \mathcal { C } } ( U )$ to

$$
[ { \mathcal { C } } ( { \widetilde { A } } ( u ) ) ] \in H / { \mathcal { C } } ( U ) .
$$

On the other hand,

$$
A ( u ) = [ \widetilde { A } ( u ) ] \in \mathcal { K } / U ,
$$

so

$$
[ \mathcal { C } ( \widetilde { A } ( u ) ) ] = \overline { { \mathcal { C } } } _ { U } ( A ( u ) ) .
$$

Since ${ \mathcal { C } } | _ { U } : U \to { \mathcal { C } } ( U )$ is a linear isomorphism,

$$
\big ( \mathrm { d } \Gamma \mathcal { C } \big | _ { U } \big ( A \big ) \big ) ( \mathcal { C } ( u ) ) = \big ( \overline { { \mathcal { C } } } _ { U } \circ A \circ ( \mathcal { C } | _ { U } ) ^ { - 1 } \big ) ( \mathcal { C } ( u ) ) ,
$$

which proves (4.2).

If ${ \widetilde { A } } ^ { \prime }$ is another lift, then $( \widetilde { A } - \widetilde { A } ^ { \prime } ) ( U ) \subseteq U$ , so

$$
\mathcal { C } ( ( \widetilde { A } - \widetilde { A } ^ { \prime } ) ( U ) ) \subseteq \mathcal { C } ( U ) .
$$

Therefore the two lifts give the same class in $H / { \mathcal { C } } ( U )$ , and the formula for the diferential is independent of the choice of lift. ■

Corollary 4.6. For every $U \in \operatorname { G r } ( q , K )$ , the diferential $\mathrm { d } \Gamma _ { \scriptstyle { C } } | _ { \scriptstyle U }$ is injective.

Proof. Let $A \in { \mathrm { H o m } } ( U , { \mathcal { K } } / U )$ satisfy

$$
\mathrm { d } \Gamma c | _ { U } ( A ) = 0 .
$$

By (4.2), for every $u \in U$

$$
\overline { { { \mathcal { C } } } } _ { U } ( A ( u ) ) = 0 .
$$

Lemma 4.4 shows that $\overline { { \mathcal { C } } } _ { U }$ is injective, so $A ( u ) = 0$ . This holds for every $u \in U$ , and therefore $A = 0$ . Hence

$$
\ker ( \mathrm { d } \Gamma \mathcal { C } | _ { U } ) = \{ 0 \} .
$$

Remark 4.7 (Diferential kernel for a noninjective linear map). Let $T : V  H$ be any linear map, put $N = \ker T$ , and suppose that $U \in \operatorname { G r } ( q , V )$ satisfies $U \cap N = \{ 0 \}$ . Then dim $T ( U ) = q ,$ , and one can again define $\Gamma _ { T } ( U ) = T ( U )$ . The quotient map

$$
\overline { { T } } _ { U } : V / U \longrightarrow H / T ( U ) , \qquad [ v ] \longmapsto [ T ( v ) ]
$$

satisfies

$$
\begin{array} { r l } & { [ v ] \in \ker \overline { { T } } _ { U } \Longleftrightarrow T ( v ) \in T ( U ) } \\ & { \qquad \Longleftrightarrow \mathrm { t h e r e ~ e x i s t s ~ } u \in U \ \mathrm { s u c h ~ t h a t } \ T ( v - u ) = 0 } \\ & { \qquad \Longleftrightarrow v \in U + N . } \end{array}
$$

Consequently,

$$
\ker \overline { { T } } _ { U } = ( U + N ) / U
$$

and

$$
\ker ( \mathrm { d } \Gamma _ { T } | _ { U } ) = \mathrm { H o m } \left( U , { \frac { U + N } { U } } \right) .
$$

For the convolution considered here, $N = \{ 0 \}$ , so this formula reduces to ker $( \mathrm { d } \Gamma c | \upsilon ) = \{ 0 \}$

## 4.4. Diferential of the Plücker map.

Proposition 4.8. For every $S \in \mathop { \mathrm { G r } } ( q , H )$ , the diferential

$$
\mathrm { d } \mathrm { P l } _ { H } \left. _ { S } : T _ { S } \mathrm { G r } ( q , H ) \longrightarrow T _ { \mathrm { P l } _ { H } ( S ) } \mathbb { P } ( \wedge ^ { q } H ) \right.
$$

is injective.

Proof. By the classical Plücker closed-embedding theorem, Theorem 3.6 (see [5, Lecture 6, pp. 63–67]), Pl<sub>H</sub> is a closed immersion. Every closed immersion is unramified, and an unramified morphism induces an injective map on Zariski tangent spaces at every point [13, Tags 04XV and 0B2G]. Hence $\mathrm { d } \mathrm { P l } _ { H } | _ { S }$ is injective.

4.5. The regularity theorem.

Theorem 4.9 (Regularity of the single-layer parametrization). The diferential of

$$
\Phi : \mathrm { G r } ( q , { \cal K } ) \longrightarrow \mathbb { P } ( \wedge ^ { q } { \cal H } )
$$

is injective at every point. More precisely, for every $U \in \operatorname { G r } ( q , K )$ 2

$$
\operatorname { r a n k } ( \mathrm { d } \Phi | _ { U } ) = q ( k - q ) .
$$

Proof. By the chain rule,

$$
\mathrm { d } \Phi | _ { U } = \mathrm { d } \mathrm { P l } _ { H } | _ { \mathcal { C } ( U ) } \circ \mathrm { d } \Gamma _ { \mathcal { C } } | _ { U } .
$$

Suppose $A \in T _ { U } \mathrm { G r } ( q , { \cal K } )$ satisfies $\mathrm { d } \Phi | _ { U } ( A ) = 0$ . By Proposition 4.8, the map d $\operatorname { P l } _ { H } \left| c ( U ) \right.$ is injective, and therefore

$$
\mathrm { d } \Gamma _ { \mathcal { C } } | _ { U } ( A ) = 0 .
$$

Corollary 4.6 then shows that $\mathrm { d } \Gamma _  \} c \vert _ { U }$ is injective, so $A = 0$ . Hence

$$
\ker ( \mathrm { d } \Phi | _ { U } ) = \{ 0 \} .
$$

By Corollary 3.3,

$$
\dim T _ { U } \operatorname { G r } ( q , K ) = \dim \operatorname { G r } ( q , K ) = q ( k - q ) .
$$

It follows that

$$
\operatorname { r a n k } ( \mathrm { d } \Phi | _ { U } ) = q ( k - q ) .
$$

5. The Closed-Embedding Theorem.

5.1. The Grassmannian isomorphism induced by a linear isomorphism. Set

$$
W = { \mathcal { C } } ( K ) \subseteq H .\tag{5.1}
$$

By Proposition 2.2, restricting the codomain gives a linear isomorphism

$$
{ \widetilde { \mathcal { C } } } : \mathcal { K } \stackrel { \sim } { \to } W .
$$

Lemma 5.1. The map

$$
\alpha : \mathrm { G r } ( q , K ) \longrightarrow \mathrm { G r } ( q , W ) , ~ U \longmapsto { \widetilde { \mathcal { C } } } ( U )
$$

is an isomorphism of algebraic varieties, with inverse

$$
\beta : \mathrm { G r } ( q , W ) \longrightarrow \mathrm { G r } ( q , K ) , \qquad S \longmapsto \widetilde { \mathcal { C } } ^ { - 1 } ( S ) .
$$

Proof. For every $U \in \operatorname { G r } ( q , { \mathcal { K } } )$ and $S \in \operatorname { G r } ( q , W )$

$$
\beta ( \alpha ( U ) ) = \widetilde { \mathcal { C } } ^ { - 1 } ( \widetilde { \mathcal { C } } ( U ) ) = U
$$

and

$$
\alpha ( \beta ( S ) ) = \widetilde { \mathcal { C } } ( \widetilde { \mathcal { C } } ^ { - 1 } ( S ) ) = S .
$$

Thus α and $\beta$ are inverse set maps.

It remains to verify regularity. Fix a decomposition

$$
\mathcal { K } = U _ { 0 } \oplus L .
$$

Then

$$
W = \widetilde { \mathcal { C } } ( U _ { 0 } ) \oplus \widetilde { \mathcal { C } } ( L ) .
$$

In the corresponding graph coordinates, α is

$$
A \longmapsto \widetilde { \mathcal { C } } | _ { L } \circ A \circ ( \widetilde { \mathcal { C } } | _ { U _ { 0 } } ) ^ { - 1 } .
$$

This map is linear and therefore regular. Applying the same calculation to $\widetilde { \mathcal { C } } ^ { - 1 }$ shows that $\beta$ is also regular. Consequently, α is an isomorphism of algebraic varieties. ■

5.2. A Plücker-coordinate proof for a sub-Grassmannian. The inclusion of linear subspaces $W \subseteq H$ gives the natural map

$$
j : \mathrm { G r } ( q , W ) \longrightarrow \mathrm { G r } ( q , H ) , \qquad S \longmapsto S .\tag{5.2}
$$

Lemma 5.2 (Closed embedding of a sub-Grassmannian). The map j is a closed embedding. Proof. Write

$$
n = \dim W , \qquad m = \dim H .
$$

Choose a basis $e _ { 1 } , \ldots , e _ { n }$ of W and extend it to a basis

$$
e _ { 1 } , \ldots , e _ { n } , e _ { n + 1 } , \ldots , e _ { m }
$$

of H.

We first prove that $j$ is injective. If $S _ { 1 } , S _ { 2 } \in \operatorname { G r } ( q , W )$ and $j ( S _ { 1 } ) = j ( S _ { 2 } )$ , then $S _ { 1 }$ and $S _ { 2 }$ are equal as linear subspaces of $H .$ , and hence $S _ { 1 } = S _ { 2 }$ . Thus $j$ is injective.

We next describe its image. Define

$$
{ \cal Z } = \left\{ S \in \operatorname { G r } ( q , H ) : p _ { J } ( S ) = 0 { \mathrm { ~ f o r ~ e v e r y ~ } } J \not \subseteq \left\{ 1 , \ldots , n \right\} \right\} .\tag{5.3}
$$

Here $J \not \in \{ 1 , \ldots , n \}$ means that J contains at least one index greater than n. We prove that

$$
j ( \mathrm { G r } ( q , W ) ) = Z .\tag{5.4}
$$

If $S \in \operatorname { G r } ( q , W )$ , every basis vector of $S$ belongs to span $( e _ { 1 } , \ldots , e _ { n } )$ , so

$$
\wedge ^ { q } S \subseteq \wedge ^ { q } W .
$$

Therefore all Plücker coordinates containing an index greater than n vanish, and hence $j ( S ) \in$ $Z .$ This proves

$$
j ( \mathrm { G r } ( q , W ) ) \subseteq Z .
$$

Conversely, take $S \in Z$ . The Plücker coordinates of $S$ are not all zero, and every coordinate containing an index greater than n is zero. Hence there exists

$$
I = \{ i _ { 1 } < \cdots < i _ { q } \} \subseteq \{ 1 , \ldots , n \}
$$

with $p _ { I } ( S ) \neq 0$ . Thus $S$ lies in the standard open set $\mathcal { U } _ { I }$ . By reordering only $e _ { 1 } , \ldots , e _ { n } .$ , we may assume that $I = \{ 1 , \ldots , q \}$ . After normalizing $p _ { I }$ to 1, the subspace S has a unique row-space matrix

$$
M _ { S } = \left( I _ { q } \quad A \quad B \right) ,\tag{5.5}
$$

where the columns of A correspond to $e _ { q + 1 } , \ldots , e _ { n }$ , and the columns of $B$ correspond to $e _ { n + 1 } , \ldots , e _ { m } .$

Let $b _ { r \ell }$ be the entry of $B$ in row $r$ and in the column corresponding to $\mathbf { \mathit { e } } _ { \ell } ,$ where $1 \leq r \leq q$ and $n + 1 \leq \ell \leq m$ . Computing the corresponding minor of (5.5) gives

$$
p _ { \{ 1 , \dots , \widehat { r } , \dots , q , \ell \} } ( S ) = ( - 1 ) ^ { q - r } b _ { r \ell } p _ { \{ 1 , \dots , q \} } ( S ) = ( - 1 ) ^ { q - r } b _ { r \ell } .
$$

This index set contains $\ell > n$ . Since $S \in Z$ , the left-hand side is zero, and therefore

$$
b _ { r \ell } = 0
$$

for every $r , \ell .$ Thus $B = 0$ , every row of $M _ { S }$ belongs to $W$ , and $S \subseteq W$ . Hence $S \in j ( \mathrm { G r } ( q , W ) )$ , proving (5.4).

Under the Plücker embedding, (5.3) can be written as

$$
\operatorname { P l } _ { H } ( Z ) = \operatorname { P l } _ { H } ( \operatorname { G r } ( q , H ) ) \cap \mathbb { P } ( \land ^ { q } W ) .\tag{5.6}
$$

The space $\mathbb { P } ( \wedge ^ { q } W )$ is the projective linear subspace of $\mathbb { P } ( \land ^ { q } H )$ defined by the homogeneous linear equations

$$
p _ { J } = 0 , \qquad J \not \subset \{ 1 , . . . , n \} .
$$

By Theorem $3 . 6 , \mathrm { P l } _ { H } ( \mathrm { G r } ( q , H ) )$ is a closed subvariety. Therefore $\left( 5 . 6 \right)$ shows that $Z$ is a closed subvariety of $\operatorname { G r } ( q , H )$

Finally, we prove that $j$ identifies $\mathrm { G r } ( q , W )$ isomorphically with $Z .$ On any standard open set with $I \subseteq \{ 1 , \ldots , n \}$ , points of $\mathrm { G r } ( q , W )$ are represented by matrices

$$
\left( I _ { q } \quad A \right) ,
$$

whereas the corresponding points of $Z$ are represented by

$$
\left( I _ { q } \quad A \quad 0 \right) .
$$

Thus, in these afine coordinates, $j$ is

$$
A \longmapsto ( A , 0 ) ,
$$

and its inverse on the image is

$$
( A , 0 ) \longmapsto A .
$$

Both maps are given by coordinate polynomials and are therefore regular. These standard open sets cover $\mathrm { G r } ( q , W )$ and $Z ,$ , so

$$
j : \mathrm { G r } ( q , W ) \stackrel { \sim } {  } Z
$$

is an isomorphism. Since $Z$ is closed in $\operatorname { G r } ( q , H )$ , the map j is a closed embedding.

Remark 5.3. The proof of Lemma 5.2 uses only the injectivity of $j ,$ the vanishing equations for Plücker coordinates, and the standard afine coordinates on Grassmannians. On each standard afine open set, the closed-embedding property is verified directly by the coordinate map $A \mapsto ( A , 0 )$

## 5.3. The main closed-embedding theorem.

Theorem 5.4 (Closed embedding of the convolution-induced Grassmann map). The map

$$
\Gamma _ { \mathcal { C } } : \mathrm { G r } ( q , \mathcal { K } ) \longrightarrow \mathrm { G r } ( q , H )
$$

is a closed embedding.

Proof. By Lemma 5.1,

$$
\alpha : \operatorname { G r } ( q , K ) \ { \overset { \sim } { \to } } \operatorname { G r } ( q , W )
$$

is an isomorphism. By Lemma 5.2,

$$
j : \operatorname { G r } ( q , W ) \hookrightarrow \operatorname { G r } ( q , H )
$$

is a closed embedding. For every $U \in \operatorname { G r } ( q , K )$ 2

$$
( j \circ \alpha ) ( U ) = j ( { \mathcal { C } } ( U ) ) = { \mathcal { C } } ( U ) = \Gamma _ { \mathcal { C } } ( U ) .
$$

Thus

$$
\Gamma _ { \mathcal { C } } = j \circ \alpha .
$$

The composition of an isomorphism with a closed embedding is a closed embedding, so $\Gamma _ { \mathcal { C } }$ is a closed embedding.

Theorem 5.5 (Closed embedding of the Grassmann–Plücker parametrization). The parametrization

$$
\Phi : \mathrm { G r } ( q , { \cal K } ) \longrightarrow \mathbb { P } ( \wedge ^ { q } { \cal H } )
$$

is a closed embedding.

Proof. By Theorem 5.4, $\Gamma _ { C }$ is a closed embedding; by Theorem 3.6, $\mathrm { P l } _ { H }$ is a closed embedding. Therefore their composition

$$
\Phi = \mathrm { P l } _ { H } \circ \Gamma _ { \mathcal { C } }
$$

is a closed embedding.

Remark 5.6 (Relation to the regularity theorem). Theorem 5.5 also implies that the diferential of Φ is injective everywhere. This implication was not used to prove Theorem 4.9. The proof of regularity uses the direct diferential formula (4.2) for the convolution-induced Grassmann map together with the classical closed-embedding theorem for the Plücker map. It does not use the closed-embedding property of the full parametrization $\Phi ,$ which is established only in Theorem 5.5. Therefore the regularity argument and the proof of the main closed-embedding theorem are not circular.

6. Finite Birationality and Geometric Consequences. Continue to write

$$
X = \operatorname { G r } ( q , K ) , \qquad Y = \Phi ( X ) \subseteq \mathbb { P } ( \land ^ { q } H ) .
$$

Corollary 6.1 (Isomorphism between the parameter space and the neural variety). The set Y is a closed subvariety of $\mathbb { P } ( \land ^ { q } H )$ , and

$$
\Phi : X { \stackrel { \sim } { \to } } Y
$$

is an isomorphism of algebraic varieties.

Proof. By definition, a closed embedding identifies X isomorphically with a closed subvariety of the target. That closed subvariety is exactly the image $Y$ of $\Phi$ ■

Corollary 6.2 (Finiteness). The morphism

$$
\Phi : X \longrightarrow \mathbb { P } ( \land ^ { q } H )
$$

is finite. In particular, $\Phi : X  Y$ is $f i n i t e$

Proof. A closed embedding is a finite morphism [13, Tag 035C]. More explicitly, let $V =$ Spec A be an afine open subset of the target. Since Φ is a closed embedding, there is an ideal $I \subseteq A$ such that

$$
\Phi ^ { - 1 } ( V ) = \operatorname { S p e c } ( A / I ) .
$$

As an A-module, $A / I$ is generated by $1 + I ,$ , and hence is finitely generated. Therefore Φ is finite.

Corollary 6.3 (Birationality). The morphism $\Phi : X  Y$ is birational.

Proof. By Lemma 3.4, X is irreducible. By Corollary 6.1, $Y$ is isomorphic to $X$ and is therefore also irreducible. The same corollary gives an isomorphism of function fields

$$
\Phi ^ { * } : \mathbb { C } ( Y ) \stackrel { \sim } { \to } \mathbb { C } ( X ) .
$$

Thus $\Phi : X  Y$ is birational. In fact, it is stronger than a birational map: it is an isomorphism on all of X, not merely on a dense open subset. ■

Corollary 6.4 (Uniqueness of fibers). For every $y \in Y$ 2

$$
\# \Phi ^ { - 1 } ( y ) = 1 .
$$

Proof. By Corollary 6.1, $\Phi : X  Y$ is an isomorphism, so its underlying map of sets is bijective. Hence every $y \in Y$ has exactly one preimage. ■

Corollary 6.5 (Dimension and smoothness). The neural variety Y satisfies

$$
\dim Y = q ( k - q )
$$

and

$$
\operatorname { S i n g } ( Y ) = \varnothing .
$$

Proof. By Corollary 6.1, $Y \cong X = \operatorname { G r } ( q , K )$ . Isomorphisms preserve local rings and therefore preserve dimension and the property of a local ring being regular. By Corollary 3.3,

$$
\dim X = q ( \dim \mathcal K - q ) = q ( k - q )
$$

and X is smooth. Therefore Y has the same dimension and is smooth; that is, $\mathrm { S i n g } ( Y ) = \varnothing . \mathbb { I }$ 1

Corollary 6.6 (The rank-one case). $I f q = 1$ , then

$$
\mathrm { G r } ( 1 , { \mathcal { K } } ) = \mathbb { P } ( { \mathcal { K } } ) ,
$$

and the parametrization reduces to the projective linear embedding

$$
\Phi : \mathbb { P } ( K ) \longrightarrow \mathbb { P } ( H ) , \qquad [ w ] \longmapsto [ { \mathcal { C } } ( w ) ] .
$$

Proof. By definition, $\operatorname { G r } ( 1 , \kappa )$ is the set of one-dimensional subspaces of K, which is P(K). Moreover, $\wedge ^ { 1 } H = H$ , and $\operatorname { P l } _ { H } : \operatorname { G r } ( 1 , H )  \operatorname { \mathbb { P } } ( H )$ is the identity identification. Thus (2.8) becomes

$$
\Phi ( [ w ] ) = [ { \mathcal C } ( w ) ] .
$$

Since C is injective, its projectivization is a projective linear closed embedding.

6.1. A concrete nontrivial symbolic computation. To check the preceding conclusions in a completely reproducible low-dimensional case, take

$$
k = 4 , \qquad q = 2 , \qquad s = 1 , \qquad d ^ { \prime } = 2 , \qquad d = 5 .
$$

For $w = ( w _ { 0 } , w _ { 1 } , w _ { 2 } , w _ { 3 } ) \in \mathcal { K } = \mathbb { C } ^ { 4 }$ , Definition 2.1 gives

$$
C _ { w } = \left( \begin{array} { c c c c c } { { w _ { 0 } } } & { { w _ { 1 } } } & { { w _ { 2 } } } & { { w _ { 3 } } } & { { 0 } } \\ { { 0 } } & { { w _ { 0 } } } & { { w _ { 1 } } } & { { w _ { 2 } } } & { { w _ { 3 } } } \end{array} \right) \in H = \mathrm { H o m } ( \mathbb { C } ^ { 5 } , \mathbb { C } ^ { 2 } ) \cong \mathbb { C } ^ { 1 0 } .\tag{6.1}
$$

Let $E _ { 1 } , \ldots , E _ { 1 0 }$ be the standard basis of H in row-major order, and let $e _ { 1 } , \ldots , e _ { 4 }$ be the standard basis of K. Then

$$
\begin{array} { r } { \mathcal { C } ( e _ { i } ) = E _ { i } + E _ { i + 6 } , \qquad 1 \leq i \leq 4 . } \end{array}\tag{6.2}
$$

Let $p _ { i j } , 1 \leq i < j \leq 4$ , be the source Plücker coordinates on ${ \mathrm { G r } } ( 2 , 4 )$ . They satisfy

$$
p _ { 1 2 } p _ { 3 4 } - p _ { 1 3 } p _ { 2 4 } + p _ { 1 4 } p _ { 2 3 } = 0 .\tag{6.3}
$$

Write $z _ { a b } , 1 \leq a < b \leq 1 0$ , for the homogeneous coordinates on $\mathbb { P } ( \wedge ^ { 2 } H ) = \mathbb { P } ^ { 4 4 }$ . By (6.2), for every $1 \leq i < j \leq 4$

$$
\begin{array} { l } { { ( \wedge ^ { 2 } { \mathcal { C } } ) ( e _ { i } \wedge e _ { j } ) = ( E _ { i } + E _ { i + 6 } ) \wedge ( E _ { j } + E _ { j + 6 } ) } } \\ { { \qquad = E _ { i } \wedge E _ { j } + E _ { i } \wedge E _ { j + 6 } - E _ { j } \wedge E _ { i + 6 } + E _ { i + 6 } \wedge E _ { j + 6 } . } } \end{array}\tag{6.4}
$$

Hence the coordinates of the image satisfy

$$
z _ { i j } = z _ { i , j + 6 } = z _ { i + 6 , j + 6 } = p _ { i j } , \qquad z _ { j , i + 6 } = - p _ { i j } ,\tag{6.5}
$$

and the remaining 21 coordinates $z _ { a b }$ not occurring in (6.5) vanish.

Over the rational field, we used Singular 4.4.1 [12] to form the graph ideal defined by (6.3)–(6.5) and then eliminated the six source coordinates $p _ { i j }$ . The resulting elimination ideal $I _ { Y }$ is the homogeneous ideal generated by

(i) the 21 vanishing coordinates $z _ { a b }$ described above;

(ii) the three independent linear relations in (6.5) for each $1 \leq i < j \leq 4$ , giving 18 linear relations in total; and

(iii) the single quadratic relation

$$
z _ { 1 2 } z _ { 3 4 } - z _ { 1 3 } z _ { 2 4 } + z _ { 1 4 } z _ { 2 3 } = 0 .\tag{6.6}
$$

More precisely, reducing the generators of the elimination ideal by the standard basis of the expected ideal, and conversely, gives the zero ideal in both directions; the reduced Gröbner basis has 40 elements. Thus the calculation recovers the homogeneous image ideal, not only its set of points.

The Hilbert series of the homogeneous coordinate ring simplifies to

$$
\mathrm { H i l b } _ { \mathbb { C } [ z ] / I _ { Y } } ( t ) = \frac { 1 + t } { ( 1 - t ) ^ { 5 } } .\tag{6.7}
$$

Consequently,

$$
\dim Y = 4 = 2 ( 4 - 2 ) , \qquad \deg Y = 2 .
$$

On the standard chart $p _ { 1 2 } \neq 0 ;$ write

$$
U = \mathrm { r o w s p a n } \left( \begin{array} { l l l l } { { 1 } } & { { 0 } } & { { a } } & { { b } } \\ { { 0 } } & { { 1 } } & { { c } } & { { d } } \end{array} \right) .
$$

Then

$$
( z _ { 1 3 } , z _ { 1 4 } , z _ { 2 3 } , z _ { 2 4 } ) = ( c , d , - a , - b ) ,
$$

and therefore

$$
\operatorname * { d e t } { \frac { \partial ( z _ { 1 3 } , z _ { 1 4 } , z _ { 2 3 } , z _ { 2 4 } ) } { \partial ( a , b , c , d ) } } = 1 .\tag{6.8}
$$

This directly checks rank $\left( \mathrm { d } \Phi \right) = 4$ on this chart. Finally, in the six independent coordinates, the image is defined only by (6.6). All first partial derivatives of this quadratic vanish simultaneously only when the six coordinates are zero. Thus the afine cone is singular only at its vertex, and the corresponding projective image is smooth.

This computation therefore recovers the image ideal and checks the dimension, full chart rank, and smoothness predicted by the main theorem in the first concrete convolutional case with a nontrivial Plücker relation. It is a low-dimensional symbolic illustration, not a computer proof of the general theorem. The reproducible script is supplied as convolution $\cdot \mathtt { q } 2 .$ \_k4.sing.

## 7. Applications and Potential Value.

7.1. Status of the statements in this section. The results proved above are that Φ has injective diferential everywhere, Φ is a closed embedding, $X \cong Y$ , and $Y$ is a smooth projective variety. This section discusses modeling directions that these results may support, but does not assert the following as consequences of the theorems in this paper:

higher predictive accuracy, better generalization, shorter actual running time,

$$
\mathrm { o r ~ g r e a t e r ~ e m p i r i c a l ~ r o b u s t n e s s . }
$$

These properties depend on the particular network, data, optimization algorithm, and implementation, and must be validated by additional theoretical analysis or numerical experiments.

7.2. Change-of-basis redundancy and intrinsic degrees of freedom. Suppose that a convolutional layer contains m filters

$$
w _ { 1 } , \ldots , w _ { m } \in { \mathcal { K } } ,
$$

and let

$$
U = \mathrm { s p a n } ( w _ { 1 } , \ldots , w _ { m } ) , \qquad \mathrm { d i m } U = q < m .
$$

Choose a basis $u _ { 1 } , \ldots , u _ { q }$ of U. For every $A \in \operatorname { G L } ( q )$ , the family

$$
( v _ { 1 } , \ldots , v _ { q } ) = ( u _ { 1 } , \ldots , u _ { q } ) A
$$

is again a basis of U. Thus the right action of ${ \mathrm { G L } } ( q )$ on a basis matrix does not change the Grassmann point.

Let n = dim K. A full-rank $n \times q$ basis matrix has nq coordinates, whereas

$$
\dim \operatorname { G L } ( q ) = q ^ { 2 } .
$$

Removing the change-of-basis freedom leaves

$$
n q - q ^ { 2 } = q ( n - q ) = \dim \operatorname { G r } ( q , K ) .
$$

Thus the Grassmann parametrization removes the intrinsic redundancy caused by a choice of basis rather than deleting coordinates arbitrarily. This quotient-space viewpoint is the standard interpretation of the Grassmannian [5, Lecture 6, pp. 63–66].

It is important that the point U records only the subspace spanned by the filters. If the task requires recovering every $w _ { i } .$ , their coeficients in a chosen basis must also be stored. The closed-embedding theorem states only that the complete Plücker representation uniquely recovers $U ;$ it does not state that the discarded ordered family of filters can be recovered.

7.3. Low-rank filter families and convolutional compression. Suppose that the filters are approximately contained in a q-dimensional subspace

$$
U = \operatorname { s p a n } ( u _ { 1 } , \ldots , u _ { q } ) ,
$$

in the sense that there are coeficients $a _ { i j }$ with

$$
w _ { i } \approx \sum _ { j = 1 } ^ { q } a _ { i j } u _ { j } .
$$

By linearity of convolution in the filter,

$$
C _ { w _ { i } } \approx \sum _ { j = 1 } ^ { q } a _ { i j } C _ { u _ { j } } .
$$

Thus one may first compute q basic convolutional responses and then form the m approximate responses by linear combinations.

If $n = \dim \mathcal { K } ,$ directly storing m filters requires mn scalars. Storing a family of $q$ basis vectors and an $m \times q$ coeficient matrix requires

$$
q n + m q = q ( n + m )
$$

scalars. Under this elementary count, the latter representation reduces the number of parameters only if

$$
q ( n + m ) < m n , \qquad \mathrm { t h a t ~ i s , } \qquad q < { \frac { m n } { m + n } } .\tag{7.1}
$$

Therefore the use of Grassmann parameters does not automatically yield compression. One must also prove or observe that the filter family has suficiently low efective rank and account for the costs of basis orthogonalization, coeficient mixing, and the storage format.

Previous work has used channel or filter redundancy in convolutional filters to construct low-rank separable approximations that accelerate pretrained convolutional networks [2, 7]. Those works decompose a particular convolution tensor and report the corresponding experiments, whereas this paper records the subspace spanned by the filters. The connection is a modeling motivation, not an equivalence of models or algorithms.

## 8. Future Work.

8.1. Multilayer Grassmann parameter spaces. If the filter space in layer i is $\kappa _ { i }$ and the subspace dimension is $q _ { i }$ , then a formal multilayer parameter space is

$$
X _ { L } = \prod _ { i = 0 } ^ { L - 1 } \mathrm { G r } ( q _ { i } , \mathcal { K } _ { i } ) ,
$$

with dimension

$$
\dim X _ { L } = \sum _ { i = 0 } ^ { L - 1 } q _ { i } ( \dim { \cal K } _ { i } - q _ { i } ) .
$$

If every $q _ { i } = 1$ , each factor reduces to a projective filter space. This product alone, however, does not define a function parametrization for a deep network. The basis-independent output of the first layer naturally carries a factor $U _ { 0 } ^ { \vee }$ , and consequently the input representation of the next layer depends on the parameters of the preceding layer. A composition between layers that is compatible with changes of basis in every layer must first be defined before one can discuss the diferential, fibers, and closed-embedding property of the total map. The single-layer theorem cannot simply be applied layer by layer.

Deep models in which Grassmann data serve as layer inputs or representations have been constructed using full-rank mappings, reorthogonalization, projection pooling, and manifold backpropagation [6]. In those models a Grassmann point represents data or an activation subspace, whereas in the present paper a Grassmann point parametrizes a subspace of convolutional filters. The two settings cannot be directly identified, but the former provides techniques that may inform the numerical design of layers.

8.2. Nonlinear activation and change-of-basis equivariance. After a basis of U has been chosen, let the channel vector be $z \in \mathbb { C } ^ { q }$ . A change of basis replaces its coordinates by $G z$ where $G \in \operatorname { G L } ( q )$ . For the coordinatewise power activation

$$
\sigma _ { \boldsymbol { r } } ( z _ { 1 } , \ldots , z _ { q } ) = ( z _ { 1 } ^ { r } , \ldots , z _ { q } ^ { r } ) ,
$$

one generally has

$$
\sigma _ { r } ( G z ) \neq G \sigma _ { r } ( z ) .
$$

For example, if $r = 2$ and

$$
z = { \binom { 1 } { 1 } } , \qquad G = { \binom { 1 } { 0 } } \ 1 \biggr ) ,
$$

then

$$
\sigma _ { 2 } ( G z ) = { \binom { 4 } { 1 } } , \qquad G \sigma _ { 2 } ( z ) = { \binom { 2 } { 1 } } .
$$

Thus an ordinary coordinatewise activation depends on the chosen basis and does not automatically descend to a map that depends only on the Grassmann point.

One candidate intrinsic construction is the symmetric-tensor map

$$
\nu _ { r } : W \longrightarrow \mathrm { S y m } ^ { r } ( W ) , \qquad z \longmapsto z ^ { \odot r } ,
$$

because, for every linear map $G : W \to W ^ { \prime }$

$$
\mathrm { S y m } ^ { r } ( G ) ( z ^ { \odot r } ) = ( G z ) ^ { \odot r } .
$$

This construction, however, would make later layers act on parameter-dependent symmetrictensor spaces, so the total parametrization would have to be rebuilt. Equivariant networks for general matrix groups require both linear and nonlinear layers to accommodate the relevant group representations and may use gated or tensor-product nonlinearities [4]. These methods provide only design directions and do not themselves prove a multilayer extension of the present model.

9. Limitations. The results of this paper have the following explicit boundaries.

First, we study the complete Plücker representation of an operator subspace, not the full network function class obtained after arbitrary activation, readout, and classification layers. If the readout does not have the appropriate ${ \mathrm { G L } } ( q )$ invariance or equivariance, the output will depend on the choice of basis.

Second, the proof of injectivity concerns the full input space and the valid convolution of Definition 2.1. If the boundary conditions are changed, output coordinates are deleted, the input class is restricted, or redundant filter parameters are used, one must prove ker $\mathcal { C } = 0$ again.

Third, the theory is formulated over complex algebraic varieties. Practical optimization is usually carried out on real Grassmann manifolds. Although the linear-algebra formulas can be restricted to the real field, the topology of the real points, numerical stability, and optimization dynamics are not direct consequences of a complex-algebraic closed-embedding result.

Fourth, the ambient vector space for the Plücker representation has dimension

$$
\dim \wedge ^ { q } H = { \binom { \dim H } { q } } .
$$

Even using $Y \subseteq \mathbb { P } ( \wedge ^ { q } W )$ and dim $W = k$ , the number of coordinates can still be ${ \binom { k } { q } }$ . Plücker coordinates are therefore well suited to theoretical analysis, but in large-scale numerical computation they may be less economical than orthonormal basis matrices or projection matrices.

Fifth, we fix the subspace dimension $q$ and do not treat changes in efective rank during training or prove how to select $q$ automatically from data. We provide only one low-dimensional symbolic computation and no systematic training or data experiments. We therefore do not claim that the model has already improved accuracy, generalization, computational complexity, or robustness.

10. Conclusion. Starting from the concrete one-dimensional finite-stride convolution

$$
( C _ { w } x ) _ { i } = \sum _ { j = 0 } ^ { k - 1 } w _ { j } x _ { s i + j } ,
$$

we proved that the linear filter-to-convolution-operator map

$$
c : \kappa \longrightarrow H
$$

is injective. Consequently, for every q-dimensional filter subspace $U$ , the operator image ${ \mathcal { C } } ( U )$ is again q-dimensional, giving the parametrization

$$
\Phi = \mathrm { P l } _ { H } \circ \Gamma _ { \mathcal { C } } .
$$

Locally, using the natural isomorphism

$$
T _ { U } \operatorname { G r } ( q , K ) \cong \operatorname { H o m } ( U , K / U )
$$

and the diferential formula

$$
\mathrm { d } \Gamma \mathcal { C } \vert _ { U } ( A ) = \overline { { \mathcal { C } } } _ { U } \circ A \circ ( \mathcal { C } \vert _ { U } ) ^ { - 1 } ,
$$

we proved that $\mathrm { d } \Gamma _ { \mathrm { \it } } c | _ { U }$ is injective. The classical Plücker closed-embedding theorem, together with the general fact that closed immersions induce injective tangent maps, gives the injectivity of d $| \mathrm { P l } _ { H }$ , and hence

$$
\operatorname { r a n k } ( \mathrm { d } \Phi | _ { U } ) = q ( k - q ) .
$$

Globally, let $W = { \mathcal { C } } ( K )$ . The linear isomorphism $\kappa \cong W$ induces

$$
\operatorname { G r } ( q , { \mathcal { K } } ) \cong \operatorname { G r } ( q , W ) .
$$

Using Plücker coordinates, we proved that the image of

$$
\operatorname { G r } ( q , W ) \hookrightarrow \operatorname { G r } ( q , H )
$$

is defined exactly by the vanishing of all Plücker coordinates containing an index external to W, and that the map is $A \mapsto ( A , 0 )$ in standard afine coordinates. It is therefore a closed embedding, and consequently Φ is a closed embedding.

The resulting chain of strict implications is

C is injective =⇒ Φ is a closed embedding $\implies \Phi : X \xrightarrow { \sim } Y$

$$
\Longrightarrow { \left\{ \begin{array} { l l } { \Phi { \mathrm { ~ i s ~ f i n i t e ~ a n d ~ b i r a t i o n a l ~ o n t o ~ i t s ~ i m a g e , } } } \\ { \# \Phi ^ { - 1 } ( y ) = 1 , } \\ { \dim Y = q ( k - q ) , } \\ { \operatorname { S i n g } ( Y ) = \emptyset . } \end{array} \right. }
$$

These conclusions apply to the single-layer linear subspace of convolution operators defined in this paper and to its complete Plücker representation. If nonlinear activation, multilayer composition, output projection, or incomplete coordinate observation is introduced, then welldefinedness of the parametrization, injectivity of the convolution map, and the diferential and fiber structures must be checked again. The closed-embedding result in this paper cannot be transferred to such models without new proofs.

As a concrete check of the general conclusions, we also performed a Singular elimination computation for $k = 4 , q = 2 , d = 5 , d ^ { \prime } = 2 , s = 1$ . It directly recovered the image ideal generated by 21 coordinate vanishings, 18 linear identifications, and one Klein quadratic, and yielded dim $Y = 4 , \deg Y = 2$ , full diferential rank on the standard chart, and $\operatorname { S i n g } ( Y ) = \varnothing$ in complete agreement with the main theorem.

Declaration on the Use of Generative Artificial Intelligence. In accordance with the current SIAM editorial policy on artificial intelligence [10], the authors make the following declaration. Generative artificial intelligence tools, including OpenAI ChatGPT and Codex, were used to assist with the organization and linguistic revision of the manuscript, the drafting and revision of mathematical exposition and arguments, literature-search support, bibliographic preparation, and LAT<sub>E</sub>X formatting. All mathematical statements, proofs, citations, and bibliography entries included in the final manuscript were independently reviewed and verified by the authors. The authors assume responsibility for all content.

11. Acknowledgement. On behalf of all authors, the corresponding author states that there is no conflict of interest. H. Zuo acknowledges support from NSFC (grant No. 12671056) and BJNSF (grant No. 1252009).

## REFERENCES

[1] S.-I. Amari, H. Park, and T. Ozeki, Singularities afect dynamics of learning in neuromanifolds, Neural Computation, 18 (2006), pp. 1007–1065, https://doi.org/10.1162/089976606776241002.

[2] E. L. Denton, W. Zaremba, J. Bruna, Y. LeCun, and R. Fergus, Exploiting linear structure within convolutional networks for eficient evaluation, in Advances in Neural Information Processing Systems 27, Curran Associates, 2014, pp. 1269–1277, https://proceedings.neurips.cc/paper\_files/paper/2014/ hash/1adaeb993eba95859121a43ea61bd858-Abstract.html.

[3] A. Edelman, T. A. Arias, and S. T. Smith, The geometry of algorithms with orthogonality constraints, SIAM J. Matrix Anal. Appl., 20 (1998), pp. 303–353, https://doi.org/10.1137/S0895479895290954.

[4] M. Finzi, M. Welling, and A. G. Wilson, A practical method for constructing equivariant multilayer perceptrons for arbitrary matrix groups, in Proceedings of the 38th International Conference on Machine Learning, Proc. Mach. Learn. Res. 139, PMLR, 2021, pp. 3318–3328, https://proceedings.mlr.press v139/finzi21a.html.

[5] J. Harris, Algebraic Geometry: A First Course, Graduate Texts in Mathematics 133, Springer, New York, 1992.

[6] Z. Huang, J. Wu, and L. Van Gool, Building deep networks on Grassmann manifolds, in Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, AAAI Press, 2018, pp. 3279–3286, https://doi.org/10.1609/aaai.v32i1.11725.

[7] M. Jaderberg, A. Vedaldi, and A. Zisserman, Speeding up convolutional neural networks with low rank expansions, in Proceedings of the British Machine Vision Conference, BMVA Press, 2014, paper 88, pp. 1–13, https://doi.org/10.5244/C.28.88.

[8] K. Kohn, T. Merkh, G. Montúfar, and M. Trager, Geometry of linear convolutional networks, SIAM J. Appl. Algebra Geom., 6 (2022), pp. 368–406, https://doi.org/10.1137/21M1441183.

[9] K. Kohn, G. Montúfar, V. Shahverdi, and M. Trager, Function space and critical points of linear convolutional networks, SIAM J. Appl. Algebra Geom., 8 (2024), pp. 333–362, https://doi.org/10.1137 23M1565504.

[10] Society for Industrial and Applied Mathematics, SIAM Publications—Editorial Policy on Artificial Intelligence, Version 2.0, efective May 2026, https://epubs.siam.org/artificial-intelligence.

[11] V. Shahverdi, G. L. Marchetti, and K. Kohn, On the geometry and optimization of polynomial convolutional networks, in Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, Proc. Mach. Learn. Res. 258, PMLR, 2025, pp. 604–612, https://proceedings.mlr.press v258/shahverdi25a.html.

[12] W. Decker, G.-M. Greuel, G. Pfister, and H. Schönemann, Singular 4-4-1—A computer algebra system for polynomial computations, 2025, https://www.singular.uni-kl.de.

[13] The Stacks Project Authors, The Stacks Project, Tags 035C, 04XV, and 0B2G, https://stacks.math. columbia.edu/tag/035C, https://stacks.math.columbia.edu/tag/04XV, and https://stacks.math. columbia.edu/tag/0B2G.