# GRADED REPRESENTATION THEORY OF EQUIVARIANT NEURAL NETWORKS

MANI SHAYESTEHFAR

Abstract. Nonlinear activations can create equivariant interactions between irreducible representations that linear maps cannot. We use the Gaussian degree decomposition to extend ordinary polynomial degree to such nonlinear maps, and prove that for a fixed coordinatewise equivariant layer each degree factors into a polynomial determined by the linear maps and a scalar determined by the activation. This separates three distinct obstructions, coming from symmetry, coordinates, and activation.

## Contents

Introduction 1   
1. Preliminaries 5   
2. Degree-resolved equivariant maps 7   
3. Coordinatewise nonlinearities 11   
4. Activation spectra 17   
5. Architectures determined by the degree calculation 20   
6. Subset representations of the symmetric group 23   
7. Scope and future directions 28   
Appendix A. Supplementary proofs 29   
Appendix B. Hidden spaces with several coordinate orbits 30   
Appendix C. Representation formulae for subset modules 31   
References 33

## Introduction

Neural networks alternate linear maps with nonlinear activation functions. When the data carry symmetries, one often asks the network to respect them: rotating, translating or relabelling an input should have a prescribed efect on the output. Mathematically, the feature spaces are then representations of a group G, and the linear parts of the network are G-equivariant maps. This is precisely the viewpoint promoted by geometric deep learning [BBCV21].

Representation theory is very efective for understanding the linear part. A finitedimensional representation can be decomposed into irreducible pieces, and an equivariant linear map cannot mix two non-isomorphic irreducibles. In a suitable basis, this makes the linear maps much simpler.

The activation function changes the picture. It is nonlinear and is usually applied coordinatewise, so it can create interactions between irreducible pieces which are completely invisible to linear representation theory. The simplest example already occurs for the symmetric group S .

Let $S _ { 3 }$ permute the coordinates of

$$
\begin{array} { r } { \mathbb { R } ^ { 3 } = L _ { 0 } \oplus L _ { 1 } , \qquad L _ { 0 } = \mathrm { s p a n } \{ ( 1 , 1 , 1 ) \} , \qquad L _ { 1 } = \{ x : x _ { 1 } + x _ { 2 } + x _ { 3 } = 0 \} . } \end{array}
$$

The first summand is the trivial representation and the second is the two-dimensional standard representation. There is no nonzero equivariant linear map from $L _ { 1 }$ to $L _ { 0 }$ Coordinatewise squaring, however, gives

$$
p _ { 0 } ( x _ { 1 } ^ { 2 } , x _ { 2 } ^ { 2 } , x _ { 3 } ^ { 2 } ) = \frac { \left\| x \right\| ^ { 2 } } { 3 } ( 1 , 1 , 1 ) , \qquad x \in L _ { 1 } ,\tag{1}
$$

where $p _ { 0 }$ is orthogonal projection onto $L _ { 0 }$ . Thus a coordinatewise nonlinearity creates a quadratic interaction between two representation types which no linear equivariant map can connect.

This is the starting point of the paper. We ask not merely whether a nonlinear equivariant interaction exists, but at which degrees it occurs and whether a given neural network layer actually produces it.

From existence to degree. Gibson, Tubbenhauer and Williamson [GTW25] (GTW) study the first, degree-free version of this question. For irreducible real representations $L$ and K of a finite group, they show that a nonzero equivariant piecewise-linear map $L  K$ exists precisely when

$$
\ker L \subseteq \ker K .\tag{2}
$$

Here piecewise linear means continuous and afine on each of finitely many polyhedral pieces. This is already very diferent from the linear theory: nonlinear equivariant maps can connect many diferent irreducible representations.

Condition (2), however, does not say how the interaction occurs. For example, the interaction in (1) first appears in degree two. More generally, one would like to distinguish such interactions by the degree in which they occur. One would also like to distinguish two further questions: symmetry may permit a degree without the chosen coordinate system producing ${ \mathrm { i t } } ,$ and the coordinate system may produce it while the chosen activation suppresses it.

For polynomial maps there is an obvious notion of degree: the degree-k equivariant maps $L  K$ are

$$
{ \mathrm { H o m } } _ { G } ( \mathrm { S y m } ^ { k } L , K ) .
$$

Activations such as ReLU are not polynomial, so we need a way to extend this notion of degree. We put the standard Gaussian measure on $L .$ Just as monomials organise ordinary polynomials by degree, Hermite polynomials form an orthogonal basis for functions with respect to the Gaussian measure. Grouping the multivariate Hermite polynomials by their total degree gives an orthogonal decomposition, called the polynomial chaos decomposition.

Thus polynomial chaos extends the usual polynomial grading to non-polynomial activations such as ReLU. We write $\Pi _ { k }$ for the orthogonal projection onto the degree-k part.

This grading is classical. Its usefulness here is that it gives a common language for polynomial and non-polynomial activations. A ReLU layer, for example, has infinitely many Hermite components, and we can ask exactly which representation-theoretic interactions occur in each one.

There is already a useful answer before considering a particular architecture. If $L \neq 0$ then an allowed degree can never occur in isolation:

$$
\mathrm { H o m } _ { G } ( \mathrm { S y m } ^ { k } L , K ) \neq 0 \quad \Longrightarrow \quad \mathrm { H o m } _ { G } ( \mathrm { S y m } ^ { k + 2 m } L , K ) \neq 0 \qquad ( m \geq 0 ) .
$$

Thus the possible degrees consist of at most two tails, one even and one odd. For finite groups we also refine $( 2 )$ : writing N = ker $L$ , some polynomial degree occurs if and only if $K ^ { N } \neq 0$ (where $K ^ { N }$ denotes the N-fixed subspace of $K )$ , and the least such degree

is at most $\left| G / N \right| - 1$ . For irreducible K, this recovers the GTW kernel condition and additionally locates the interaction in degree.

What does one layer produce? Knowing that a degree-k equivariant map $L  K$ exists does not mean that a given neural network layer produces one. We now make this distinction precise.

Let $H = \mathbb { R } ^ { B }$ be a representation with a basis B which is permuted by G. A typical equivariant neural network layer has the form

$$
L { \stackrel { A } { \to } } H { \stackrel { \sigma v } { \longrightarrow } } H { \stackrel { R } { \to } } K .
$$

The first map A expresses the input in the hidden coordinates B, and the last map R sends the hidden features to the output space $K$ . Between them we apply the same scalar activation $\sigma : \mathbb { R }  \mathbb { R }$ separately to each hidden coordinate: if

$$
h = \sum _ { b \in \cal B } h _ { b } b , \quad \quad \mathrm { t h e n } \quad \quad \sigma _ { \cal B } ( h ) = \sum _ { b \in \cal B } \sigma ( h _ { b } ) b .
$$

Since $G$ only permutes the basis vectors in $B ,$ applying the same function to every coordinate commutes with the G-action. Hence $\sigma _ { B }$ is equivariant, and so is the whole layer whenever A and R are equivariant linear maps.

There are two separate reasons why a degree permitted by representation theory may fail to appear in this layer:

(a) the linear maps A and R may not produce the relevant polynomial interaction, or

(b) the activation σ may have no component in that degree.

The main calculation separates these two efects. Let $s ^ { 2 }$ be the common variance of the coordinates of AX when X is a standard Gaussian vector in L. Let $a _ { k } ( \sigma ; s )$ be the coeficient of the kth Hermite polynomial in the expansion of the one-variable function $z \mapsto \sigma ( s z )$ . Finally, for an ordinary input $x \in L .$ , set

$$
Q _ { k } ( x ) = R ( ( A x ) ^ { \odot k } ) ,
$$

where $( A x ) ^ { \odot k }$ is the coordinatewise kth power. Then

$$
\Pi _ { k } \big [ R \sigma _ { B } ( A x ) \big ] = \frac { a _ { k } ( \sigma ; s ) } { s ^ { k } } \Pi _ { k } Q _ { k } .\tag{3}
$$

This gives three separate questions:

(a) Does symmetry allow a degree-k equivariant map $L  K ?$

(b) Do the chosen linear maps A and R actually produce one?

(c) Does the activation σ retain degree k?

The first asks whether $\mathrm { H o m } _ { G } ( \mathrm { S y m } ^ { k } L , K ) \neq 0$ . The second asks whether $Q _ { k }$ is nonzero, and the third whether $a _ { k } ( \sigma ; s )$ is nonzero.

The second question also has an explicit answer. Put $C = A A ^ { * }$ , written in the coordinate basis of $\mathbb { R } ^ { B }$ (with G acting transitively). Then

$$
\| \Pi _ { k } Q _ { k } \| _ { L ^ { 2 } } ^ { 2 } = k ! \operatorname { t r } ( R C ^ { \circ k } R ^ { * } ) ,\tag{4}
$$

where $C ^ { \circ k }$ is obtained by taking the kth power of each entry of C. Hence

$$
Q _ { k } \neq 0 \quad \Longleftrightarrow \quad \mathrm { t r } ( R C ^ { \circ k } R ^ { * } ) > 0 .
$$

For $S _ { 4 }$ we give an example where a quadratic equivariant map exists, but the chosen coordinatewise layer produces no quadratic term at all, and changing the activation cannot fix this. For $C _ { 8 }$ the opposite phenomenon occurs: the layer does produce the first permitted polynomial interaction, but a positively homogeneous activation such as ReLU removes it. Adding a bias restores it.

Thus a nonlinear interaction can fail for three diferent reasons: it may be forbidden by symmetry, missed by the chosen linear maps, or removed by the activation. Distinguishing these three mechanisms is the main point of the paper.

Consequences for architectures. The same calculation answers two practical questions. For any fixed collection of layers, an explicit Gram matrix determines exactly which degree-k equivariant maps they can produce. Moreover, shifted ReLUs can realise arbitrary Hermite coeficients up to any prescribed finite degree. Using the regular representation then gives all equivariant polynomial directions, although not eficiently.

For ordinary ReLU there is a simple restriction: all odd Hermite degrees above one vanish. A shared bias removes this restriction generically. These degree-resolved interactions can also be used to design representation-aware architectures. In a companion paper by the author, the resulting source-to-target decomposition is used to build a higher-order graph network on unordered triples, and experiments show an improvement over both a standard GNN baseline and a parameter-matched control with the full linear $S _ { n }$ -equivariant mixing space [Sha26]. This addresses a question left open in [GTW25]: whether organising nonlinear equivariant networks by irreducible representation theory is useful in practice.

Subset features. The final part of the paper treats the action of $S _ { n }$ on r-element subsets. A degree-d monomial in these variables corresponds to an r-uniform multihypergraph with d edges. If the output is indexed by s-subsets, one simply marks an s-subset of vertices. Equivariant polynomial maps are therefore indexed by these marked multihypergraphs, up to relabelling.

This gives an explicit basis whose types stabilise at the sharp bound

$$
n = r d + s .
$$

For quadratic edge-to-vertex maps there are seven stable types, and all seven can be evaluated in $O ( n ^ { 2 } )$ operations, or $O ( n + m )$ for m nonzero edges.

Relation to existing work and scope. Our starting point is the piecewise-linear representation theory of [GTW25, Theorem 2H.3]. For irreducible real representations of a finite group, they characterise the existence of a nonzero equivariant piecewise-linear map by inclusion of the representation kernels. We refine this existence question by recording the polynomial degrees in which an interaction can occur. We then ask which of these interactions are produced by a specified coordinatewise layer and activation.

The compact-group structure theorem of [Aie26, Theorem 2.5 and Remark 2.6] gives a broader context for this finite-group analysis. Let Γ be a compact group with identity component $\Gamma ^ { \circ }$ , and equip its finite-dimensional real representations L and K with invariant inner products. Every equivariant piecewise-linear map $F : L  K$ has the form

$$
\begin{array} { r } { F ( v + w ) = \psi ( v ) + T w , \qquad v \in L ^ { \Gamma ^ { \circ } } , \quad w \in ( L ^ { \Gamma ^ { \circ } } ) ^ { \perp } , } \end{array}
$$

where $\psi : L ^ { \Gamma ^ { \circ } }  K ^ { \Gamma ^ { \circ } }$ is equivariant and piecewise linear, and $T : ( L ^ { \Gamma ^ { \circ } } ) ^ { \perp }  ( K ^ { \Gamma ^ { \circ } } ) ^ { - }$ ⊥ is an equivariant linear map. The joint action on the two fixed subspaces has finite image. Consequently, our finite-group degree criteria apply to this remaining nonlinear problem, and our layer criteria can test chosen coordinatewise constructions for ψ. For connected groups, nonlinear dependence is confined to the trivial subrepresentations. This restriction also explains why polynomial equivariants alone cannot characterise piecewiselinear existence for general compact groups.

At the level of layer construction, [PDLS24] classify the combinations of representations, coordinates and pointwise activations that give an equivariant activation map. We work with hidden permutation representations, where shared coordinatewise activations are equivariant, and analyse the interactions selected by the linear lift, activation and readout.

The analytic tools are classical. Gaussian chaos, Wick ordering and their interpretation through symmetric tensors are developed in [Jan97, Chapters 2–4]. Hermite expansions and the associated covariance identities also underlie the dual-activation framework of [DFS16]. We use these identities to separate the polynomial selected by an equivariant layer from the scalar coeficient supplied by its activation. This gives explicit Gram criteria for the vanishing of a degree component and for the degree subspace spanned by a specified family of layers.

Our spanning construction also connects with equivariant approximation theory. [Rav20] establishes universality using regular hidden actions, while [Yar22] obtains invariant and equivariant approximation through group averaging. Here regular representations provide an exact spanning construction in each polynomial degree. Matching finitely many Hermite components does not, by itself, control the remaining approximation error.

For subset representations, the marked-hypergraph basis follows the orbit-sum principle used by [PLK<sup>+</sup>23] to construct multigraph-indexed bases of equivariant graph polynomials. We formulate this construction for r-subset inputs and s-subset outputs, and determine its sharp stable range. The distinction between parameter sharing across dimensions and compatibility with embeddings already appears in [LD24]; our marked-orbit criterion makes that compatibility explicit for these polynomial maps.

The Gaussian measure here is only a reference measure for the grading, not an assumption about trained features, and the resulting one-layer obstructions need not persist under arbitrary composition.

Organisation. Sections 1 and 2 introduce polynomial and Gaussian degree and prove the finite-group degree-detection results. Section 3 studies coordinatewise layers and proves (3) and (4). Section 4 determines the relevant activation spectra, and Section 5 gives the resulting architecture-level spanning statements. Section 6 treats polynomial layers on subset features. Supplementary proofs and representation formulae are collected in the appendices.

Acknowledgement. This research was completed during a master’s project supervised by Dani Tubbenhauer, whom I thank for many discussions and for guidance throughout.

Use of AI. The initial backbone of the work was developed independently. Discussions with Claude Fable pointed towards sources such as [Jan97], where the connection between $L ^ { 2 } ( \gamma _ { L } )$ and $\mathrm { S y m } ^ { k } L$ was found. ChatGPT Sol assisted with polishing arguments and proofs, and with the literature review. After using each tool, the author reviewed and edited the content as needed, and takes full responsibility for the content of the work.

## 1. Preliminaries

All representations are finite-dimensional and real unless otherwise stated, and are equipped with fixed invariant inner products. The acting group G is finite unless compact groups are explicitly allowed. For a representation L, we write ker L for the kernel of the action; it is a normal subgroup of G.

1.1. Polynomial maps and symmetric powers. Polynomial degree is the basic grading used throughout the paper. We begin by recalling the standard identification between homogeneous polynomial maps and maps out of symmetric powers.

For a real representation L and $k \geq 1$ , define

$$
\mathrm { S y m } ^ { k } L = L ^ { \otimes k } \Big / \mathrm { s p a n } \{ t - \pi t : \ t \in L ^ { \otimes k } , \ \pi \in S _ { k } \} , \qquad \mathrm { S y m } ^ { 0 } L = \mathbb { R } ,
$$

where $S _ { k }$ acts by permuting tensor factors. The diagonal action of $G$ on $L ^ { \otimes k }$ commutes with this action, and therefore descends to $\mathrm { S y m } ^ { k } L$

For $x \in L$ , we write $x ^ { k }$ for the class of $x ^ { \otimes k }$ in $\mathrm { S y m } ^ { k } L$ . Pure powers span $\operatorname { S y m } ^ { k } L .$ , and the induced inner product is characterised by

$$
\left. x ^ { k } , y ^ { k } \right. = \left. x , y \right. ^ { k } .
$$

To avoid confusion, coordinatewise powers in a based permutation representation will instead be denoted by $x ^ { \odot k }$

For real representations L and $K$ , let $\mathrm { P o l y } _ { k } ( L , K )$ denote the space of homogeneous polynomial maps $P : L  K$ of degree k. The G-action is $( g \cdot P ) ( x ) = g P ( g ^ { - 1 } x )$ , so that

$$
\mathrm { P o l y } _ { k } ( L , K ) ^ { G } = \{ P \in \mathrm { P o l y } _ { k } ( L , K ) : P ( g x ) = g P ( x ) { \mathrm { ~ f o r ~ a l l ~ } } g \in G , x \in L \} .
$$

Similarly, G acts on Hom $( \mathrm { S y m } ^ { k } L , K )$ by $( g \cdot T ) ( y ) = g T ( g ^ { - 1 } y )$

For a finite index set I, multi-indices are written ${ \boldsymbol { \alpha } } \in  { \mathbb { N } } _ { 0 } ^ { I }$ , with $\textstyle | { \boldsymbol { \alpha } } | = \sum _ { i \in I } \alpha _ { i }$

Proposition 1.1. For every $k \geq 0$

$$
\mathrm { P o l y } _ { k } ( L , K ) ^ { G } \cong \mathrm { H o m } _ { G } ( \mathrm { S y m } ^ { k } L , K ) .
$$

See Section A.1 for a proof. The isomorphism sends $T$ to $x \mapsto T ( x ^ { k } )$ and follows from polarisation. We use this identification throughout to regard homogeneous degree-k equivariant polynomial maps as elements of Hom $\mathsf { \Omega } _ { 1 G } ( \mathrm { S y m } ^ { k } L , K )$

1.2. Piecewise-linear maps. Our nonlinear layers are piecewise linear, while the degree decomposition below lives in a Gaussian $L ^ { 2 }$ space. We first fix the class of maps to which both viewpoints apply. A map $F : L  K$ is piecewise linear if it is continuous and there is a finite covering of L by polyhedra on each of which $F$ is afine. We write Hom $\operatorname { r } _ { G } ^ { \mathrm { P L } } ( L , K )$ for the equivariant ones. We follow the afine-on-pieces convention of [GTW25], which includes constant maps. In particular, applying a continuous piecewise-linear scalar function with finitely many knots coordinatewise again gives a piecewise-linear map.

1.3. Gaussian spaces and polynomial chaos. Let L be an orthogonal representation of dimension m and let $\gamma _ { L }$ denote the standard Gaussian measure on $L .$ This measure is invariant under $O ( L )$ , hence under $G ,$ and has moments of every order. Its Hermite expansion provides the orthogonal grading used below.

For a representation $K$ , let $L ^ { 2 } ( \gamma _ { L } ; K )$ be the Hilbert space of square-integrable maps $L  K$ modulo null sets, with G acting unitarily by $( g \cdot F ) ( x ) = g F ( g ^ { - 1 } x )$

We use the probabilists’ Hermite polynomials $\mathrm { H e } _ { k } .$ , defined by the generating function exp $\begin{array} { r } { ( t x - t ^ { 2 } / 2 ) = \sum _ { k > 0 } \mathrm { H e } _ { k } ( x ) t ^ { k } / k ! } \end{array}$ and normalised so that $\mathbb { E } [ \mathrm { H e } _ { j } ( Z ) \mathrm { H e } _ { k } ( Z ) ] = k ! \delta _ { j k }$ for $Z \sim N ( 0 , 1 )$

In m variables, the products $\Pi _ { i } { \mathrm { H e } } _ { k _ { i } } ( x _ { i } )$ with $\textstyle \sum _ { i } k _ { i } = k$ span the kth polynomial chaos $\mathcal { H } _ { k } ( L )$ , and

$$
L ^ { 2 } ( \gamma _ { L } ) = \widehat { \bigoplus } _ { k \geq 0 } \mathcal { H } _ { k } ( L ) .\tag{5}
$$

Thus the chaos index is a polynomial degree after orthogonal removal of lower-degree terms. For example, $\mathrm { H e _ { 2 } } ( x ) = x ^ { 2 } - 1$ lies entirely in the second chaos. We write $\Pi _ { k }$ for the orthogonal projection onto $\mathcal { H } _ { k } ( L )$ and use the same symbol for the induced projection on K-valued maps:

$$
\Pi _ { k } : L ^ { 2 } ( \gamma _ { L } ; K ) \longrightarrow \mathcal { H } _ { k } ( L ) \otimes K .
$$

The chaos projection $\Pi _ { k }$ restricts to an $O ( L )$ -equivariant isomorphism from homogeneous degree-k polynomials onto $\mathcal { H } _ { k } ( L )$ , and hence realises the identification

$$
\mathcal { H } _ { k } ( L ) \cong \mathrm { S y m } ^ { k } ( L ^ { * } ) \cong \mathrm { S y m } ^ { k } L ,
$$

using the fixed invariant inner product (see [Jan97]). For a linear form $\boldsymbol { \ell } ( \boldsymbol { x } ) = \langle \boldsymbol { x } , \boldsymbol { a } \rangle$ with $\| a \| = s > 0 , \Pi _ { k } ( \ell ^ { k } ) = s ^ { k } \mathrm { H e } _ { k } ( \ell / s )$

If P is homogeneous of degree $k ,$ then $P - \Pi _ { k } P$ has degree strictly less than k. The space $\mathcal { H } _ { k } ( L )$ therefore consists of degree-at-most-k polynomials orthogonal to all lower-degree polynomials, rather than homogeneous polynomials themselves.

The following identity will be used in the factorisation and norm calculations.

Lemma 1.2 (Mehler). Let $( X , Y )$ be jointly standard normal with correlation $\rho$ . Then

$$
\operatorname { \mathbb { E } } [ \mathrm { H e } _ { a } ( X ) \mathrm { H e } _ { b } ( Y ) ] = a ! \rho ^ { a } \delta _ { a b } .
$$

Proof. This is a standard identity. See [Jan97, Theorem 3.9].

To use this Hilbert-space decomposition for the piecewise-linear maps above, we only need to check square-integrability.

Proposition 1.3. Let L and K be finite-dimensional. If $F : L  K$ is piecewise linear with finitely many polyhedral pieces, then $F \in L ^ { 2 } ( \gamma _ { L } ; K )$

Proof. Since F is afine on finitely many polyhedral pieces, there exist constants $C , D \geq 0$ such that $\| F ( x ) \| \leq C \| x \| + D$ for all $x \in L$ . Hence $\| \bar { \boldsymbol { F } } ( \boldsymbol { x } ) \| ^ { 2 } \leq 2 C ^ { 2 } \| \boldsymbol { x } \| ^ { 2 } + 2 D ^ { 2 }$ . If $X \sim \gamma _ { L } .$ then $\mathbb { E } \| X \| ^ { 2 } < \infty$ , so

$$
\int _ { L } \| F ( x ) \| ^ { 2 } d \gamma _ { L } ( x ) \leq 2 C ^ { 2 } \mathbb { E } \| X \| ^ { 2 } + 2 D ^ { 2 } < \infty .
$$

Thus $F \in L ^ { 2 } ( \gamma _ { L } ; K )$

Two continuous maps agreeing $\gamma _ { L }$ -almost everywhere agree everywhere, because $\gamma _ { L }$ has full support and the set where they difer is open. Consequently, the chaos components determine a continuous map uniquely.

## 2. Degree-resolved equivariant maps

We first identify the equivariant part of each polynomial chaos. We then extend the grading to several source summands and relate occurrence at some degree to the finite-group existence criterion.

2.1. The equivariant decomposition. The Gaussian grading is useful here only if it respects equivariance. The next theorem says that it does so, degree by degree.

Theorem 2.1. Let G be a finite or compact group acting orthogonally on finite-dimensional real spaces L and K. There is a canonical isomorphism of Hilbert spaces

$$
L ^ { 2 } ( \gamma _ { L } ; K ) ^ { G } \cong \widehat { \bigoplus } _ { k \geq 0 } \mathrm { H o m } _ { G } ( \mathrm { S y m } ^ { k } L , K ) .
$$

Proof. Tensoring (5) with the finite-dimensional space K gives $L ^ { 2 } ( \gamma _ { L } ; K ) = \widehat { \bigoplus _ { k } } ( \mathcal { H } _ { k } ( L ) \otimes K )$ Each chaos projection is $O ( L )$ -equivariant. Since $G \subseteq O ( L )$ , it is also G-equivariant and hence commutes with the projection onto G-fixed vectors. Therefore taking fixed points commutes with the orthogonal sum. Finally

$$
( { \mathcal { H } } _ { k } ( L ) \otimes K ) ^ { G } \cong ( \operatorname { S y m } ^ { k } ( L ^ { * } ) \otimes K ) ^ { G } \cong \operatorname { H o m } _ { G } ( \operatorname { S y m } ^ { k } L , K )
$$

using the $O ( L )$ -equivariant identification $\mathcal { H } _ { k } ( L ) \cong \mathrm { S y m } ^ { k } L$ above and the tensor–Hom identification. □

The key point is that each chaos projection is equivariant under the full orthogonal group. It therefore preserves the G-fixed subspace, allowing the Gaussian grading to restrict to equivariant maps.

Corollary 2.2. Every equivariant piecewise-linear $F : L  K$ has a unique sequence $( \Pi _ { k } F ) _ { k \geq 0 }$ with $\Pi _ { k } F \in \operatorname { H o m } _ { G } ( \operatorname { S y m } ^ { k } L , K )$ , and $F \mapsto ( \Pi _ { k } F ) _ { k }$ is injective on continuous square-integrable equivariant maps.

Proof. Square-integrability follows from Proposition 1.3. Orthogonality gives uniqueness in $L ^ { 2 }$ , and full support of the Gaussian measure gives pointwise uniqueness for continuous maps. □

Corollary 2.2 establishes injectivity but does not characterise the image of the transform. In particular, not every square-summable sequence of equivariant polynomial components arises from a piecewise-linear map. The finite-group existence questions considered here do not require such a characterisation. The distinction becomes essential for compact connected groups, as discussed in Section 7.

2.2. Multi-sources and multidegrees. Fix a finite orthogonal G-stable decomposition $V = \oplus _ { i \in I } W _ { i }$ into nonzero subrepresentations. The summands need not be irreducible or pairwise non-isomorphic. The multidegree refinement recorded below separates the contributions of the summands $W _ { i }$ within a single total degree. This allows the coordinatewise results of Section 3 to be applied to a direct-sum input without treating its summands separately.

Multi-source inputs are not used in our constructions; the following decompositions indicate how the theory extends to them.

Symmetric powers of a direct sum split by multidegree. Iterating $\operatorname { S y m } ( U \oplus W ) \cong$ Sym $U \otimes { \mathrm { S y } }$ m W gives the G-equivariant isomorphism

$$
\operatorname { S y m } ^ { k } V \cong \bigoplus _ { | \alpha | = k } \bigotimes _ { i \in I } \operatorname { S y m } ^ { \alpha _ { i } } W _ { i } .
$$

Hence for any $K ,$ ,

$$
\mathrm { H o m } _ { G } ( \mathrm { S y m } ^ { k } V , K ) \cong \bigoplus _ { | \alpha | = k } \mathrm { H o m } _ { G } \Bigl ( \bigotimes _ { i } \mathrm { S y m } ^ { \alpha _ { i } } W _ { i } , K \Bigr ) .
$$

Example 2.3. For $V = W _ { 1 } \oplus W _ { 2 }$ 2

$$
\mathrm { S y m } ^ { 2 } V \cong \mathrm { S y m } ^ { 2 } W _ { 1 } \oplus \left( W _ { 1 } \otimes W _ { 2 } \right) \oplus \mathrm { S y m } ^ { 2 } W _ { 2 } ,
$$

$$
\mathrm { S y m } ^ { 3 } V \cong \mathrm { S y m } ^ { 3 } W _ { 1 } \oplus ( \mathrm { S y m } ^ { 2 } W _ { 1 } \otimes W _ { 2 } ) \oplus ( W _ { 1 } \otimes \mathrm { S y m } ^ { 2 } W _ { 2 } ) \oplus \mathrm { S y m } ^ { 3 } W _ { 2 } .
$$

The mixed summands have no counterpart in a GTW interaction graph whose edges have a single source. Instead, they record simultaneous nonlinear dependence on two summands.

On the Gaussian side, orthogonality of the $W _ { i }$ makes the components of $x \sim \gamma _ { V }$ independent, so $\gamma _ { V } = \otimes _ { i } \gamma _ { W _ { i } }$ , and the Hilbert tensor product gives the multichaos decomposition

$$
L ^ { 2 } ( \gamma _ { V } ) \cong \bigoplus _ { \alpha \in \mathbb { N } _ { 0 } ^ { I } } \bigotimes _ { i } { \mathcal { H } } _ { \alpha _ { i } } ( W _ { i } ) ,
$$

whose total-degree-k part is the sum over $| { \boldsymbol { \alpha } } | = k$ . Write $\Pi _ { \alpha }$ for the projection onto the αth multichaos. Tensoring with K and taking fixed points degreewise, as in Theorem 2.1, gives

$$
L ^ { 2 } ( \gamma _ { V } ; K ) ^ { G } \cong \bigoplus _ { \alpha \in \mathbb { N } _ { 0 } ^ { I } } { \mathrm { H o m } } _ { G } \Bigl ( \bigotimes _ { i } \mathrm { S y m } ^ { \alpha _ { i } } W _ { i } , K \Bigr ) .
$$

2.3. Degree sets and detection. We now forget multiplicities and record only the degrees in which the target can occur. This is the information needed for the existence and parity questions below.

Definition 2.4. For representations $L , K$ define the degree set

$$
D ( L , K ) : = \{ k \geq 0 : \mathrm { H o m } _ { G } ( \mathrm { S y m } ^ { k } L , K ) \neq 0 \} ,
$$

and $d ( L , K ) : = \operatorname* { m i n } D ( L , K ) \in \mathbb { N } _ { 0 } \cup \{ \infty \}$ . We use min $\mathcal { D } = \infty$

If $K ^ { G } \neq 0$ , then degree zero records constant equivariant maps. When the question is transfer of input information, the first positive degree may therefore be the more relevant quantity.

Proposition 2.5. Let $L \neq 0$ , and let G be finite or compact. $I f k \in D ( L , K )$ , then $k + 2 m \in D ( L , K )$ for every $m \in  { \mathbb { N } } _ { 0 }$ . Consequently, for each $\varepsilon \in \{ 0 , 1 \}$ define

$$
d _ { \varepsilon } ( L , K ) : = \operatorname* { m i n } \{ k \in D ( L , K ) : k \equiv \varepsilon { \pmod { 2 } } \} .
$$

Then

$$
D ( L , K ) = ( d _ { 0 } ( L , K ) + 2 \mathbb { N } _ { 0 } ) \cup ( d _ { 1 } ( L , K ) + 2 \mathbb { N } _ { 0 } ) ,
$$

where a term is omitted $i f d _ { \varepsilon } ( L , K )$ is infinite.

Proof. Since G is finite or compact, L admits a G-invariant inner product, so $q ( x ) = \| x \| ^ { 2 }$ is a nonzero invariant quadratic polynomial. Hence multiplication by $q$ sends

$$
\mathrm { P o l y } _ { k } ( L , K ) ^ { G } \longrightarrow \mathrm { P o l y } _ { k + 2 } ( L , K ) ^ { G } .
$$

This map is injective because the polynomial ring on L is an integral domain. Hence $k \in D ( L , K ) \implies k + 2 \in D ( L , K )$ . Thus, within each parity, occurrence persists from the first occurring degree onward, giving the stated $D ( L , K )$ , with the corresponding term omitted when $d _ { \varepsilon } ( L , K ) = \infty$ □

Thus, once the first even and odd occurrences are known, the entire degree set is known. The remaining task is to determine those two thresholds.

Theorem 2.6 (Finite-group degree detection). Let $L \neq 0$ and K be (not necessarily irreducible) real G-representations, and put $N = \ker L$ and $Q = G / N$ . Then

$$
D ( L , K ) \neq \emptyset \quad \Longleftrightarrow \quad K ^ { N } \neq 0 .
$$

In this case $d ( L , K ) \leq | Q | - 1$ . If K is irreducible, the condition is equivalent to ker $L \subseteq$ ker $K$

Proof. Suppose $k \in D ( L , K )$ and choose $0 \neq T \in \operatorname { H o m } _ { G } ( \operatorname { S y m } ^ { k } L , K )$ . Since $N =$ ker L acts trivially on ${ \mathrm { S y m } } ^ { k } L .$ we have $\begin{array} { r } { \ i T ( y ) = T ( n y ) = T ( y ) } \end{array}$ for all $n \in N$ and $y \in \operatorname { S y m } ^ { k } L$ Thus im $T \subseteq K ^ { N }$ , and since $T \neq 0$ we obtain $K ^ { N } \neq 0$ . If K is irreducible, then $K ^ { N }$ is G-stable because $N \triangleleft G$ . Hence $K ^ { N } \neq 0$ implies $K ^ { N } = K$ . Equivalently, ker $L \subseteq$ ker K.

For the converse, assume $K ^ { N } \neq 0$ . Both L and $K ^ { N }$ descend to $Q = G / N$ , and L is faithful as a Q-representation. We argue in three steps.

A free orbit exists. For 1 $\neq q \in Q$ the fixed space $L ^ { q }$ is a proper subspace, since $L$ is faithful. The finite union of these proper subspaces cannot cover $L _ { i }$ , so some $v \in L$ has trivial stabiliser, and the orbit $Q v$ consists of $| Q |$ distinct points.

The orbit can be separated by a linear form. A functional $\ell \in L ^ { * }$ fails to separate two distinct orbit points qv and $q ^ { \prime } v$ precisely when $\ell ( q v - q ^ { \prime } v ) = 0$ . Since $q v - q ^ { \prime } v \neq 0 .$ , the set of such functionals is a proper hyperplane of $L ^ { * }$ . There are only finitely many pairs of orbit points, so a finite union of these hyperplanes cannot cover $L ^ { * }$ . Hence some $\ell \in L ^ { * }$ separates all points of $Q v$

Interpolate. With such an $\ell ,$ the Lagrange polynomial

$$
p ( x ) = \prod _ { 1 \neq q \in Q } \frac { \ell ( x ) - \ell ( q v ) } { \ell ( v ) - \ell ( q v ) }
$$

has degree at most $| Q | - 1$ , with $p ( v ) = 1$ and $p ( q v ) = 0$ . Fix $0 \neq w \in K ^ { N }$ and average:

$$
F ( x ) = \sum _ { q \in Q } p ( q ^ { - 1 } x ) q w .
$$

Reindexing $q = a q ^ { \prime }$ shows $F ( a x ) = a F ( x )$ , so $F$ is equivariant, and $F ( v ) = w \neq 0 .$ so $F \neq 0$ The action is linear, so each homogeneous component $F _ { j }$ of $F$ is separately equivariant, and some $F _ { j } \neq 0$ with $j \leq | Q | - 1$ . So $\mathrm { P o l y } _ { j } ( L , K ^ { N } ) ^ { Q } \ne \stackrel { \sim } { 0 }$ . Hence Hom $\mathsf { \Omega } _ { 2 } ( \mathrm { S y m } ^ { j } L , K ^ { N } ) \neq$ 0. Inflating along $G \ \to \ Q$ and composing with the inclusion $K ^ { N } \hookrightarrow K \mathrm { { \ g i v e s \ 0 } } \ne$ Hom $\mathsf { \Omega } _ { 1 G } ( \mathrm { S y m } ^ { j } L , K ^ { N } ) \subseteq$ Hom<sub>G</sub>(Sym<sup>j</sup> L, K), so $j \in D ( L , K )$ □

The key point here is that every irreducible representation of a finite group occurs somewhere in the symmetric algebra of a faithful representation. The interpolation proof gives the explicit bound $d ( L , K ) \leq | G / \ker L | - 1$ . No such argument is available for a connected group, which is consistent with the failure discussed in Section 7.

For irreducible $L , K$ , combining Theorem 2.6 with (2) identifies the GTW existence criterion with $D ( L , K ) \neq \emptyset$ . The following refinement also decides whether each parity occurs.

Corollary 2.7. Let $L \neq 0$ and let K be irreducible with ker $L \subseteq$ ker $K$ . Put $Q = G /$ ker $L _ { i }$ , so that Q acts faithfully on L and naturally on K.

(a) If no element of Q acts $a s - I _ { L }$ , then both even and odd degrees occur, with

$$
d \ k _ { 0 } ( L , K ) , d _ { 1 } ( L , K ) \leq 2 | Q | - 1 .
$$

(b) If some $z \in Q$ acts $a s - I _ { L }$ , then z is a central involution and acts on K as either $I _ { K } \ o r \ { - } I _ { K }$ . In the first case only even degrees occur, and in the second only odd degrees occur. In either case, the occurring parity has threshold at most $| Q | - 1$

Proof. Fix $\varepsilon \in \{ 0 , 1 \}$ and let $\widetilde Q = Q \times \{ \pm 1 \}$ act on L and K by

$$
( q , t ) \cdot x = t q x , \qquad ( q , t ) \cdot y = t ^ { \varepsilon } q y .
$$

A homogeneous Q-equivariant polynomial $P : L  K$ is ${ \cal \widetilde Q } .$ -equivariant exactly when its degree is congruent to $\varepsilon$ modulo 2.

Suppose first that no element of Q acts $\mathrm { a s } - I _ { L }$ . Then the $\widetilde { Q } \mathrm { - a c t i o n }$ on $L$ is faithful. By Theorem 2.6, there exists a nonzero Qe-equivariant homogeneous polynomial of degree at most $| \widetilde Q | - 1 = 2 | Q | - 1$ . Applying this for $\varepsilon = 0$ and $\varepsilon = 1$ gives both parities.

Now suppose $z \in Q$ acts $\mathrm { a s } - I _ { L }$ . Since $Q$ acts faithfully on $L ,$ we have $z ^ { 2 } = 1$ , and for every $q \in Q$ the elements $q z q ^ { - 1 }$ and z act identically on L. Hence z is central. The kernel of the Qe-action on L is therefore $\{ ( 1 , 1 ) , ( z , - 1 ) \}$

By Theorem 2.6, a degree of parity ε occurs exactly when this kernel acts trivially on K, that is, when

$$
( - 1 ) ^ { \varepsilon } z y = y \qquad { \mathrm { f o r ~ a l l ~ } } y \in K .
$$

Since z is central, its action on K lies in $\operatorname { E n d } _ { G } ( K )$ . Moreover $z ^ { 2 } = 1$ , so irreducibility implies that z acts as either $I _ { K }$ or $- I _ { K }$ . Hence exactly one parity occurs: even if z acts as $I _ { K }$ , and odd if it acts $\mathrm { a s } \ - I _ { K }$ . The faithful quotient has order $| Q |$ , so the corresponding threshold is at most $| Q | - 1$ □

2.4. Computing degree sets. The detection theorem decides whether some degree occurs, but not where the first occurrences lie. Molien’s formula packages the dimensions in all degrees into one generating series.

Proposition 2.8 (Molien series). Let G be finite and $c _ { k } = \dim \operatorname { H o m } _ { G } ( \operatorname { S y m } ^ { k } L , K )$ . Then

$$
\sum _ { k \geq 0 } c _ { k } t ^ { k } = { \frac { 1 } { | G | } } \sum _ { g \in G } { \frac { \chi _ { K } ( g ) } { \operatorname* { d e t } ( I - t \rho _ { L } ( g ) ) } } = { \frac { 1 } { | G | } } \sum _ { g \in G } \chi _ { K } ( g ) \exp { \Bigl ( } \sum _ { m \geq 1 } { \frac { t ^ { m } } { m } } \chi _ { L } ( g ^ { m } ) { \Bigr ) } .
$$

Here $\rho _ { L } : G \to { \mathrm { G L } } ( L )$ is the homomorphism defining the action of G on $L ,$ and $\chi _ { L }$ and $\chi _ { K }$ are the characters of L and K.

This is the usual character average; see Section A.2. The exponential form is useful computationally because it uses only the power-map characters $\chi _ { L } ( g ^ { m } )$ , which are fixedpoint counts for permutation representations. We use this form for subset representations in Theorem C.2.

When K is irreducible, the coeficient $c _ { k }$ need not equal the multiplicity of $K$ in $\operatorname { S y m } ^ { k } L$ over R, but rather

$$
c _ { k } = [ \operatorname { S y m } ^ { k } L : K ] \ { \mathrm { d i m } } _ { \mathbb { R } } \operatorname { E n d } _ { G } ( K ) .
$$

In small representations one can often read the degree set directly from the symmetric powers. The next two examples illustrate Theorem 2.6 and Proposition 2.5.

Example 2.9 (One-dimensional sources). Suppose dim $L = 1$ . Then $\mathrm { S y m } ^ { k } L = L ^ { \otimes k }$ . If $\chi$ is the real character of L, its order is $q \in \{ 1 , 2 \}$ . For the one-dimensional target $\chi ^ { a }$ $a \in \{ 0 , 1 \}$ ,

$$
D ( L , \chi ^ { a } ) = \{ k \geq 0 : k \equiv a { \pmod { q } } \} .
$$

Thus a trivial source reaches the trivial target in every degree, while a source of order two reaches the trivial target in even degrees and itself in odd degrees.

Example 2.10 (Rotation representations of $C _ { n } )$ . Let $C _ { n } = \left. a \right.$ , and let $\rho _ { s }$ and $\rho _ { t }$ be irreducible two-dimensional real rotation representations, with 2s, 2t ̸≡ 0 (mod n). Writing $( \rho _ { s } ) _ { \mathbb { C } } \cong W _ { s } \oplus W _ { - s }$ , define

$$
R _ { k } = \{ r \in \mathbb { Z } : | r | \leq k , \ r \equiv k { \pmod { 2 } } \} .
$$

Then $\mathrm { S y m } ^ { k } ( ( \rho _ { s } ) _ { \mathbb { C } } ) \cong \oplus _ { r \in R _ { k } } W _ { r s } .$ , and therefore

$$
D ( \rho _ { s } , \rho _ { t } ) = \{ k \geq 0 : \exists r \in R _ { k } { \mathrm { ~ w i t h ~ } } r s \equiv \pm t { \pmod { n } } \} .
$$

This set is nonempty exactly when $\operatorname* { g c d } ( s , n ) \mid t .$ , or equivalently when ker $\rho _ { s } \subseteq$ ker $\rho _ { t }$ . Since $R _ { k } \subseteq R _ { k + 2 }$ , occurrence persists within each parity, as predicted by Proposition 2.5.

## 3. Coordinatewise nonlinearities

The preceding sections determine which equivariant polynomial maps exist. We now ask which of these maps a coordinatewise layer produces.

The input representation L is arbitrary throughout and need not be irreducible or multiplicity-free. In particular, the results include interactions between components of a direct-sum input, without requiring a separate treatment of those components.

3.1. Coordinatewise layers and their degree components. There are two pieces of data to track: the polynomial produced by the hidden coordinates, and the scalar Hermite coeficient supplied by the activation. We introduce them separately.

Let L and K be finite-dimensional real orthogonal G-representations, regarded as the input and output spaces. Coordinatewise application is equivariant only after choosing coordinates that the group permutes. A hidden representation is an intermediate representation $H = \mathbb { R } ^ { B }$ with an orthonormal basis B permuted by G. Throughout this section we assume that the action on B is transitive.

No inclusion relations between $L , \ H$ , and K are assumed. They are connected by equivariant linear maps

$$
A : L \longrightarrow H , \qquad R : H \longrightarrow K .
$$

Thus a coordinatewise layer has the form

$$
\begin{array} { r } { L \stackrel { A } { \longrightarrow } H \stackrel { \sigma _ { B } } { \longrightarrow } H \stackrel { R } { \longrightarrow } K , } \end{array}
$$

where the nonlinearity is applied in the distinguished coordinate basis of H.

All adjoints are taken with respect to the fixed inner products. Hidden representations with several coordinate orbits are treated in Section B.

Definition 3.1. For an activation $\sigma : \mathbb { R }  \mathbb { R }$ , its coordinatewise extension to H is

$$
\sigma _ { B } \left( \sum _ { b \in B } z _ { b } b \right) = \sum _ { b \in B } \sigma ( z _ { b } ) b .
$$

The associated coordinatewise layer is

$$
F : L \longrightarrow K , \qquad F ( x ) = R \sigma _ { B } ( A x ) .
$$

Since $\sigma _ { B }$ commutes with permutations of $B ,$ the layer F is equivariant. We assume $A \neq 0$

Definition 3.2. For $b \in B .$ define the coordinate vector $a _ { b } = A ^ { * } b \in L$ . Thus the bth hidden coordinate is $( A x ) _ { b } = \langle x , a _ { b } \rangle$ . The coordinate Gram matrix is

$$
C = ( C _ { b b ^ { \prime } } ) _ { b , b ^ { \prime } \in \mathcal { B } } , \qquad C _ { b b ^ { \prime } } = \langle a _ { b } , a _ { b ^ { \prime } } \rangle .
$$

Thus $C$ is the matrix of $A A ^ { * }$ in the basis B.

Lemma 3.3. There is an $s > 0$ such that $\left\| a _ { b } \right\| = s$ for every $b \in B$ . Moreover,

$$
s ^ { 2 } = C _ { b b } = \frac { \mathrm { t r } C } { \mathrm { d i m } H } .
$$

For $X \sim \gamma _ { L }$ , every coordinate $( A X ) _ { b }$ therefore has distribution $N ( 0 , s ^ { 2 } )$

Proof. Equivariance and orthogonality give $a _ { g b } = g a _ { b }$ . Transitivity implies that all coordinate vectors have the same norm, and $A \neq 0$ implies that this norm is positive. Summing the diagonal entries of C gives the trace identity. Finally, $\langle X , a _ { b } \rangle$ is centred Gaussian with variance $\left\| \boldsymbol { a } _ { b } \right\| ^ { 2 }$ □

The activation enters the kth chaos through a single scalar coeficient.

Definition 3.4. Suppose $\sigma ( s Z )$ is square-integrable for $Z \sim N ( 0 , 1 )$ . Its Hermite coeficients at scale s are

$$
a _ { k } ( \sigma ; s ) = { \frac { 1 } { k ! } } \mathbb { E } [ \sigma ( s Z ) \mathrm { H e } _ { k } ( Z ) ] , \qquad a _ { k } ( \sigma ) : = a _ { k } ( \sigma ; 1 ) .
$$

We impose this integrability assumption throughout. It holds for every continuous piecewise-linear activation with finitely many knots, since such a function has at most linear growth.

The remaining degree-k information comes from the fixed lift and readout. It is encoded by the coordinatewise kth power.

Definition 3.5. For $\begin{array} { r } { z = \sum _ { b } z _ { b } b \in H } \end{array}$ , define

$$
z ^ { \odot k } : = \sum _ { b } z _ { b } ^ { k } b \quad ( k \geq 0 ) .
$$

The degree-k coordinate polynomial associated with A and R is

$$
Q _ { k } : L \longrightarrow K , \qquad Q _ { k } ( x ) = R \bigl ( ( A x ) ^ { \odot k } \bigr ) = \sum _ { b \in B } \langle x , a _ { b } \rangle ^ { k } R b .
$$

For matrices, ◦ denotes entrywise multiplication, and $C ^ { \circ k }$ denotes the entrywise power, with $C ^ { \circ 0 } = J$ , the all-ones matrix.

Coordinatewise powers commute with permutations, so $Q _ { k } \in \mathrm { P o l y } _ { k } ( L , K ) ^ { G }$ . On homogeneous degree-k polynomials, $\Pi _ { k }$ is an isomorphism onto $\mathcal { H } _ { k } ( L )$ . In particular,

$$
\Pi _ { k } Q _ { k } = 0 \quad \Longleftrightarrow \quad Q _ { k } = 0 .
$$

All $L ^ { 2 }$ norms below use the standard Gaussian on the source and the fixed inner product on the target.

With these definitions, the activation and the coordinate geometry separate as follows.

Theorem 3.6 (Coordinatewise factorisation and Gram formula). Under the preceding assumptions, for every $k \geq 0$

$$
\Pi _ { k } F = \frac { a _ { k } ( \sigma ; s ) } { s ^ { k } } \Pi _ { k } Q _ { k } ,\tag{6}
$$

$$
\| \Pi _ { k } F \| _ { L ^ { 2 } } ^ { 2 } = \frac { k ! | a _ { k } ( \sigma ; s ) | ^ { 2 } } { s ^ { 2 k } } \operatorname { t r } ( R C ^ { \circ k } R ^ { * } ) .
$$

Consequently,

$$
\begin{array} { r c l } { { \Pi _ { k } F \not = 0 } } & { { \Longleftrightarrow } } & { { a _ { k } ( \sigma ; s ) \not = 0 ~ a n d ~ Q _ { k } \not = 0 , } } \\ { { Q _ { k } \not = 0 } } & { { \Longleftrightarrow } } & { { \mathrm { t r } ( R C ^ { \circ k } R ^ { * } ) > 0 . } } \end{array}
$$

No irreducibility or multiplicity-free assumption is required.

The transitivity assumption is used only to give all hidden coordinates the same variance. The corresponding formula for several coordinate orbits is recorded in Section B.

Proof. For each b, the random variable $\langle X , a _ { b } \rangle / s$ is standard normal. Hence

$$
\sigma ( \langle x , a _ { b } \rangle ) = \sum _ { k \geq 0 } a _ { k } ( \sigma ; s ) \mathrm { H e } _ { k } ( \langle x , a _ { b } \rangle / s )
$$

in $L ^ { 2 } ( \gamma _ { L } )$ . The kth summand belongs to $\mathcal { H } _ { k } ( L )$ , and

$$
\Pi _ { k } \big ( \langle x , a _ { b } \rangle ^ { k } \big ) = s ^ { k } \mathrm { H e } _ { k } \big ( \langle x , a _ { b } \rangle / s \big ) .
$$

Multiplying by Rb and summing over b proves (6).

Applying Lemma 1.2 to the hidden coordinates gives

$$
\Big \langle \Pi _ { k } \big ( \langle x , a _ { b } \rangle ^ { k } \big ) , \Pi _ { k } \big ( \langle x , a _ { b ^ { \prime } } \rangle ^ { k } \big ) \Big \rangle _ { L ^ { 2 } } = k ! C _ { b b ^ { \prime } } ^ { k } .
$$

Therefore

$$
\| \Pi _ { k } Q _ { k } \| _ { L ^ { 2 } } ^ { 2 } = k ! \sum _ { b , b ^ { \prime } } C _ { b b ^ { \prime } } ^ { k } \langle R b , R b ^ { \prime } \rangle = k ! \operatorname { t r } ( R C ^ { \circ k } R ^ { * } ) .
$$

Combining this identity with (6) proves the norm formula. The nonvanishing statements follow from injectivity of $\Pi _ { k }$ on homogeneous degree-k polynomials. □

The theorem separates the three questions $( a ) { - } ( c )$ of the introduction. For fixed A and R, changing the activation only changes the scalar multiplying $\Pi _ { k } Q _ { k } ;$ it cannot select a diferent polynomial direction at degree k.

Example 3.7. Let $H = \mathbb { R } ^ { 3 }$ be the $S _ { 3 }$ permutation representation, and recall the trivial and standard subrepresentations, $L _ { 0 }$ and $L _ { 1 }$

Let $A : L _ { 1 } \hookrightarrow$ H be inclusion, and $R = p _ { L _ { 0 } }$ be orthogonal projection. Then $C = p _ { L } .$ and $s ^ { 2 } = \mathrm { d }$ im $L _ { 1 } / 3 = 2 / 3$ . As in (1),

$$
Q _ { 2 } ( x ) = \frac { 1 } { 3 } \left\| x \right\| ^ { 2 } \mathbf { 1 } , \qquad \Pi _ { 2 } Q _ { 2 } ( x ) = \frac { 1 } { 3 } ( \left\| x \right\| ^ { 2 } - 2 ) \mathbf { 1 } .
$$

Here $\mathbb { E } [ \left. X \right. ^ { 2 } ] = 2$ for $X \sim \gamma _ { L _ { 1 } }$ . Thus

$$
\Pi _ { 2 } { \cal F } ( x ) = \frac { a _ { 2 } ( \sigma ; \sqrt { 2 / 3 } ) } { 2 } ( \| x \| ^ { 2 } - 2 ) { \bf 1 } .
$$

The coordinate polynomial is nonzero, so this second-chaos interaction is present precisely when $a _ { 2 } ( \sigma ; \sqrt { 2 / 3 } ) \ne 0$

3.2. The submodule generated by the coordinates. We next explain why the polynomial $Q _ { k }$ can vanish even when Ho $\mathrm { 1 } _ { G } ( \mathrm { S y m } ^ { k } L , K )$ is nonzero.

Definition 3.8. For $k \geq 0$ , define

$$
Y _ { k } = \mathrm { s p a n } \{ a _ { b } ^ { k } : b \in \mathcal { B } \} \subseteq \mathrm { S y m } ^ { k } L .
$$

We call $Y _ { k }$ the coordinate-generated submodule of degree k. Define also

$$
\phi _ { k } : H \longrightarrow \mathrm { S y m } ^ { k } L , \qquad \phi _ { k } ( b ) = a _ { b } ^ { k } .
$$

The relation $a _ { g b } = g a _ { b }$ implies that $\phi _ { k }$ is equivariant. Thus $Y _ { k } = \operatorname { i m } \phi _ { k }$ is indeed a submodule.

Proposition 3.9. For every $k \geq 0 , \phi _ { k } ^ { * } \phi _ { k } = C ^ { \circ k }$ , and so

$$
Y _ { k } \cong ( \ker C ^ { \circ k } ) ^ { \perp }
$$

as G-representations. Proposition 1.1 gives $Q _ { k } ( x ) = R \phi _ { k } ^ { * } ( x ^ { k } )$ . Thus

$$
Q _ { k } \neq 0 \quad \Longleftrightarrow \quad R \phi _ { k } ^ { * } \neq 0 .
$$

Proof. The tensor inner product gives

$$
\begin{array} { r } { \langle a _ { b } ^ { k } , a _ { b ^ { \prime } } ^ { k } \rangle = \langle a _ { b } , a _ { b ^ { \prime } } \rangle ^ { k } = C _ { b b ^ { \prime } } ^ { k } . } \end{array}
$$

Hence $C ^ { \circ k }$ is the Gram matrix of the vectors $a _ { b } ^ { k }$ , proving $\phi _ { k } ^ { * } \phi _ { k } = C ^ { \circ k }$ . In particular, ker $\phi _ { k } = \ker C ^ { \circ k }$ . Restricting $\phi _ { k }$ to the orthogonal complement of its kernel gives the stated isomorphism onto $Y _ { k }$

For $z \in \operatorname { S y m } ^ { k } L ,$

$$
R \phi _ { k } ^ { * } z = \sum _ { b } \langle z , a _ { b } ^ { k } \rangle R b .
$$

Taking $z = x ^ { k }$ gives $Q _ { k } ( x )$ . Since pure powers span $\mathrm { S y m } ^ { k } L$ , the polynomial vanishes identically if and only if $R \phi _ { k } ^ { * }$ is zero. □

An arbitrary degree-k equivariant polynomial may use all of $\operatorname { S y m } ^ { k } L ,$ , whereas the coordinate construction sees only $Y _ { k }$ . A target can therefore occur in $\mathrm { S y m } ^ { k } L$ and still be invisible to the chosen coordinates. Even when it occurs in $Y _ { k } .$ , the readout R may kill the resulting direction.

3.3. Projection onto irreducible summands. When the lift is an inclusion and the readout is an orthogonal projection, the Gram criterion becomes especially transparent. Let W, $K \subseteq H$ be nonzero subrepresentations and take

$$
A : W \hookrightarrow H , \qquad R = p _ { K } : H \longrightarrow K .
$$

Writing $E _ { W }$ and $E _ { K }$ for the orthogonal projector matrices in B, we have

$$
C = E _ { W } , \qquad s ^ { 2 } = \frac { \dim W } { \dim H } , \qquad Q _ { k } ( x ) = p _ { K } ( x ^ { \odot k } ) \quad ( x \in W ) .
$$

The norm formula becomes

$$
\begin{array} { r } { \| \Pi _ { k } Q _ { k } \| _ { L ^ { 2 } ( \gamma _ { W } ; K ) } ^ { 2 } = k ! \operatorname { t r } ( E _ { K } E _ { W } ^ { \circ k } ) , } \end{array}
$$

which holds without a multiplicity-free assumption.

Suppose now that H is multiplicity-free, $H = \oplus _ { i = 0 } ^ { m } L _ { j }$ with the $L _ { j }$ pairwise nonisomorphic and irreducible. Let $L _ { 0 }$ be the trivial summand, let $E _ { j }$ be orthogonal projection onto $L _ { j }$ , and let $d _ { j } = \dim L _ { j }$ . Transitivity gives

$$
E _ { 0 } = \frac { J } { \dim \cal H } , \qquad ( E _ { j } ) _ { b b } = \frac { d _ { j } } { \dim \cal H } .
$$

When H is multiplicity-free, the matrix $E _ { W } ^ { \circ k }$ is determined by one scalar on each irreducible summand. We record these scalars next.

Definition 3.10. For the fixed source subrepresentation $W$ (not necessarily irreducible), define

$$
\kappa _ { j } ( k ) : = \frac { 1 } { d _ { j } } \mathrm { t r } ( E _ { j } E _ { W } ^ { \circ k } ) , \qquad k \geq 0 .
$$

We call $\kappa _ { j } ( k )$ the projector coeficient for the source $W .$ , target $L _ { j }$ and degree k.

Corollary 3.11. For each $k \geq 0 .$

(a) The Hadamard power of the projector decomposes as

$$
E _ { W } ^ { \circ k } = \sum _ { j = 0 } ^ { m } \kappa _ { j } ( k ) E _ { j } , \qquad \kappa _ { j } ( k ) \geq 0 .
$$

(b) For $Q _ { k } ( x ) = p _ { j } ( x ^ { \odot k } )$ with $x \in W$

$$
\begin{array} { r } { Q _ { k } \neq 0 \quad \Longleftrightarrow \quad \kappa _ { j } ( k ) > 0 , \qquad \| \Pi _ { k } Q _ { k } \| _ { L ^ { 2 } } ^ { 2 } = k ! d _ { j } \kappa _ { j } ( k ) . } \end{array}
$$

(c) The coordinate-generated submodule for the inclusion $W \hookrightarrow H$ is

$$
Y _ { k } \cong \bigoplus _ { j : \kappa _ { j } ( k ) > 0 } L _ { j } .
$$

Proof. Hadamard products of G-equivariant matrices are again equivariant. Thus $E _ { W } ^ { \circ k }$ is a self-adjoint G-endomorphism of $\textstyle H = \bigoplus _ { j = 0 } ^ { m } L _ { j }$ . By multiplicity-freeness and Schur’s lemma, together with self-adjointness, it acts on each $L _ { j }$ by a real scalar. Hence $E _ { W } ^ { \circ k } =$ $\Sigma _ { j = 0 } ^ { m } \lambda _ { j } ( k ) E _ { j }$ . Multiplying by $E _ { j }$ and taking traces proves $\begin{array} { r } { E _ { W } ^ { \circ k } = \sum _ { j = 0 } ^ { m } \kappa _ { j } ( k ) E _ { j } } \end{array}$

Since $E _ { W }$ is positive semidefinite, the Schur product theorem implies that $E _ { W } ^ { \circ k }$ is positive semidefinite. Its scalar on $L _ { j }$ is therefore nonnegative, so $\kappa _ { j } ( k ) \geq 0$

For the inclusion $W \hookrightarrow H$ and projection $p _ { j } : H \to L _ { j }$ , the norm formula gives

$$
\| \Pi _ { k } Q _ { k } \| _ { L ^ { 2 } } ^ { 2 } = k ! \operatorname { t r } ( E _ { j } E _ { W } ^ { \circ k } ) = k ! d _ { j } \kappa _ { j } ( k ) .
$$

Since $\Pi _ { k }$ is injective on homogeneous degree-k polynomials,

$$
Q _ { k } \neq 0 \quad \Longleftrightarrow \quad \kappa _ { j } ( k ) > 0 .
$$

Finally, Proposition 3.9 gives $Y _ { k } \cong ( \ker E _ { W } ^ { \circ k } ) ^ { \perp }$ . Since $E _ { W } ^ { \circ k }$ acts on $L _ { j }$ by $\kappa _ { j } ( k )$ , its kernel is the sum of those $L _ { j }$ with $\kappa _ { j } ( k ) = 0$ . Hence

$$
Y _ { k } \cong \bigoplus _ { j : \kappa _ { j } ( k ) > 0 } L _ { j } .
$$

When $W$ is an irreducible summand and $k = 2 ,$ these Hadamard coeficients are the usual Krein parameters, the structure constants for entrywise products of primitive idempotents, up to the chosen normalisation. Higher k gives iterated Hadamard products. See also Section C for the Johnson-scheme case.

Lemma 3.12. For each $k \geq 0$

$$
\kappa _ { \it j } ( k ) > 0 \implies \kappa _ { \it j } ( k + 2 ) > 0 .
$$

Equivalently, if $p _ { j } ( x ^ { \odot k } ) | _ { W }$ is nonzero, then $p _ { j } ( x ^ { \odot ( k + 2 ) } ) | _ { W }$ is nonzero. Hence, for each $j ,$ the supported degrees form at most one even tail and one odd tail.

Proof. For the trivial summand $L _ { 0 }$

$$
\kappa _ { 0 } ( 2 ) = \mathrm { t r } ( E _ { 0 } E _ { W } ^ { \circ 2 } ) = { \frac { \dim W } { \dim H } } .
$$

Using the decomposition from Corollary 3.11,

$$
E _ { W } ^ { \circ 2 } = { \frac { \dim W } { \dim H } } E _ { 0 } + B = { \frac { \dim W } { ( \dim H ) ^ { 2 } } } J + B ,
$$

where B is positive semidefinite. Therefore

$$
E _ { W } ^ { \circ ( k + 2 ) } = E _ { W } ^ { \circ k } \circ E _ { W } ^ { \circ 2 } = { \frac { \dim W } { ( \dim H ) ^ { 2 } } } E _ { W } ^ { \circ k } + E _ { W } ^ { \circ k } \circ B .
$$

The second term is positive semidefinite by the Schur product theorem. Its projector coeficients are therefore nonnegative, so

$$
\kappa _ { j } ( k + 2 ) \geq \frac { \dim W } { ( \dim H ) ^ { 2 } } \kappa _ { j } ( k ) .
$$

Since $W \neq 0$ , positivity of $\kappa _ { j } ( k )$ implies positivity of $\kappa _ { \mathscr { j } } ( k + 2 )$ . The equivalence with the coordinate polynomial follows from Corollary 3.11. □

This is the coordinate-level analogue of Proposition 2.5: within each parity, coordinate support also persists once it appears.

3.4. Examples of realisation and obstruction. The first example shows that a permitted quadratic map can be missed by the coordinate construction.

Example 3.13. Let $S _ { 4 }$ act on the six unordered pairs of $[ 4 ] : = \{ 1 , 2 , 3 , 4 \}$ , and let $H = \mathbb { R } ^ { ( \frac { [ 4 ] } { 2 } ) }$ be the corresponding permutation representation. Its standard subrepresentation is

$$
W = \left\{ x _ { a b } = u _ { a } + u _ { b } : u \in \mathbb { R } ^ { 4 } , \sum _ { a } u _ { a } = 0 \right\} .
$$

The map $u \mapsto x$ is equivariant and injective, since $\begin{array} { r } { u _ { a } = \frac { 1 } { 2 } \sum _ { b \ne a } x _ { a b } } \end{array}$ on its image. Thus W is a copy of the Specht module $S ^ { ( 3 , 1 ) }$ . Let $J _ { c }$ be the complementation operator on edge coordinates, defined by

$$
J _ { c } e _ { A } = e _ { [ 4 ] \backslash A } .
$$

$$
( x ^ { \odot 2 } ) _ { A ^ { c } } = ( x ^ { \odot 2 } ) _ { A } .
$$

For $x \in W$ , the zero-sum condition gives $x _ { A ^ { c } } = - x _ { A }$ . Thus $J _ { c }$ acts $\mathrm { a s } - 1$ on W, whereas

Since $J _ { c }$ is a self-adjoint involution, its two eigenspaces are orthogonal. Consequently,

$$
p _ { W } ( x ^ { \odot 2 } ) = 0 \qquad ( x \in W ) .
$$

Nevertheless, the map

$$
u \longmapsto \left( u _ { a } ^ { 2 } - { \frac { 1 } { 4 } } \sum _ { b } u _ { b } ^ { 2 } \right) _ { a = 1 } ^ { 4 }
$$

is a nonzero equivariant quadratic map from the standard vertex representation to itself. Transporting it through the identification above gives Hom $\ l _ { S _ { 4 } } ( \mathrm { S y m } ^ { 2 } W , W ) \neq 0$

Thus symmetry permits a quadratic interaction that the fixed layer $F ( x ) = p _ { W } \sigma _ { B } ( x )$ cannot produce in its second chaos. This holds for every activation satisfying the integrability assumption, including biased ReLU, because its coordinate polynomial $Q _ { 2 }$ is zero. The same argument applies in every even degree.

Equivalently, the Gram criterion in Theorem 3.6 gives $\mathrm { t r } ( E _ { W } E _ { W } ^ { \circ 2 } ) = 0$ in this example. For regular representations of finite abelian groups, every permitted degree has a nonzero coordinate polynomial.

Proposition 3.14. Let G be finite abelian, and let $H = \mathbb { R } G = \bigoplus _ { i } L _ { j }$ be its real regular representation, equipped with the regular coordinate basis. Let $W \subseteq H$ be any nonzero subrepresentation, and let $p _ { j }$ be orthogonal projection onto $L _ { j }$ . Then, $f o r$ every $k \geq 0$

$$
p _ { j } ( x ^ { \odot k } ) | _ { W } \not \equiv 0 \quad \Longleftrightarrow \quad \mathrm { H o m } _ { G } ( \mathrm { S y m } ^ { k } W , L _ { j } ) \neq 0 .
$$

Proof. After complexification, identify $H _ { \mathbb { C } }$ with functions $G \to \mathbb { C }$ . Since G is abelian, its regular representation is a direct sum of one-dimensional character spaces. The subrepresentation $W _ { \mathbb { C } }$ is therefore spanned by a subset of the character functions.

Consider the equivariant multiplication map

$$
\mu _ { k } : \mathrm { S y m } ^ { k } W _ { \mathbb { C } } \longrightarrow H _ { \mathbb { C } } , \qquad f _ { 1 } \cdot \cdot \cdot f _ { k } \longmapsto f _ { 1 } \odot \cdot \cdot \cdot \odot f _ { k } .
$$

A monomial in character functions is sent to their pointwise product, which is a nonzero character function of the same representation type. Thus every character occurring in $\operatorname { S y m } ^ { k } W _ { \mathbb { C } }$ occurs in im $\mu _ { k }$ , and conversely.

The polynomial $p _ { j } ( x ^ { \odot k } )$ corresponds to $p _ { \mathcal { I } } \mu _ { k }$ . Since pure powers span the symmetric power, this polynomial is nonzero exactly when a character of $( L _ { j } ) _ { \mathbb { C } }$ occurs in $\mathrm { S y m } ^ { k } W _ { \mathbb { C } }$ That is, Hom $\mathsf { \Omega } _ { 1 G } ( \mathrm { S y m } ^ { k } W , L _ { j } ) \neq 0$ □

The proposition detects nonzero degrees only; it does not say that the selected polynomial spans the full equivariant polynomial space. For these regular abelian representations, the presence of a permitted degree in the layer is therefore decided by the activation coeficient in Theorem 3.6.

## 4. Activation spectra

The coordinatewise factorisation separates the polynomial selected by a layer from the coeficient supplied by its activation. We now determine when $a _ { k } ( \sigma ; s )$ vanishes.

Definition 4.1. Let $s > 0$ , and suppose $\sigma ( s Z )$ is square-integrable for $Z \sim N ( 0 , 1 )$ . The Hermite spectrum of σ at scale s is the sequence $( a _ { k } ( \sigma ; s ) ) _ { k \geq 0 }$ . Its support is

$$
\operatorname { s u p p } a _ { \bullet } ( \sigma ; s ) = \{ k \in \mathbb { N } _ { 0 } : a _ { k } ( \sigma ; s ) \neq 0 \} .
$$

At scale one, write supp $a _ { \bullet } ( \sigma ) = \operatorname { s u p p } a _ { \bullet } ( \sigma ; 1 )$

For a layer satisfying the assumptions of Theorem 3.6,

$$
\{ k : \Pi _ { k } F \neq 0 \} = \{ k : Q _ { k } \neq 0 \} \cap \mathrm { s u p p } a _ { \bullet } ( \sigma ; s ) .
$$

Thus the activation support determines which of the nonzero coordinate polynomials contribute to the layer.

4.1. Positively homogeneous activations. We start with positively homogeneous activations, which include the identity, absolute value, and ReLU.

Definition 4.2. A function $\sigma : \mathbb { R }  \mathbb { R }$ is positively homogeneous of degree one if

$$
\sigma ( t x ) = t \sigma ( x ) \qquad ( t \geq 0 , ~ x \in \mathbb { R } ) .
$$

Writing $m _ { + } = \sigma ( 1 )$ and $m _ { - } = - \sigma ( - 1 )$ , such a function is necessarily of the form

$$
\sigma ( x ) = \left\{ \begin{array} { c c } { { m _ { + } x , } } & { { x \geq 0 , } } \\ { { m _ { - } x , } } & { { x \leq 0 , } } \end{array} \right. \ = \ \begin{array} { c c } { { \displaystyle { \frac { m _ { + } + m _ { - } } { 2 } x + \frac { m _ { + } - m _ { - } } { 2 } \ | x | } . } } \end{array}
$$

Every such function is continuous and piecewise linear.

From here on, homogeneous means positively homogeneous of degree one.

Theorem 4.3. Let $\sigma \neq 0$ be homogeneous, with slopes $m _ { + }$ and m<sub>−</sub> as above. For every $s > 0$

$$
a _ { k } ( \sigma ; s ) = s a _ { k } ( \sigma ) ,
$$

and its support is given by

$$
\begin{array} { l c l } { { m _ { + } = m _ { - } } } & { { \implies \ \mathrm { s u p p } a _ { \bullet } ( \sigma ) = \{ 1 \} , } } \\ { { m _ { + } = - m _ { - } } } & { { \implies \ \mathrm { s u p p } a _ { \bullet } ( \sigma ) = 2 \mathbb { N } _ { 0 } , } } \\ { { m _ { + } \neq \pm m _ { - } } } & { { \implies \ \mathrm { s u p p } a _ { \bullet } ( \sigma ) = \{ 1 \} \cup 2 \mathbb { N } _ { 0 } . } } \end{array}
$$

Proof. Positive homogeneity gives $\sigma ( s Z ) = s \sigma ( Z )$ , which proves the scaling identity.

The odd part of $\sigma$ is a multiple of $x = \mathrm { H e } _ { 1 } ( x )$ , which contributes only in degree one. The function |x| is even, so its odd Hermite coeficients vanish. Its degree-zero coeficient is $\mathbb { E } [ | Z | ] = 2 \varphi ( 0 ) > 0$ , where $\varphi ( t ) = ( 2 \pi ) ^ { - 1 / 2 } e ^ { - t ^ { 2 } / 2 }$ is the standard one-dimensional Gaussian density. For $r \geq 1$ , two integrations by parts give

$$
a _ { 2 r } ( | \cdot | ) = \frac { 2 } { ( 2 r ) ! } \mathrm { H e } _ { 2 r - 2 } ( 0 ) \varphi ( 0 ) .
$$

These coeficients are nonzero because

$$
\mathrm { H e } _ { 2 m } ( 0 ) = ( - 1 ) ^ { m } \frac { ( 2 m ) ! } { 2 ^ { m } m ! } .
$$

Thus $| x |$ has support exactly $2  { \mathbb { N } } _ { 0 }$ . The two parts of $\sigma$ have disjoint supports, so their nonzero coeficients cannot cancel. □

Example 4.4. The identity activation has support $\{ 1 \}$ , absolute value has support $2  { \mathbb { N } } _ { 0 }$ and ReL $\begin{array} { r } { \mathcal { \Lambda } ( x ) : = \operatorname* { m a x } ( x , 0 ) = \frac { x + | x | } { 2 } } \end{array}$ has support $\{ 1 \} \cup 2  { \mathbb { N } } _ { 0 }$

The scaling identity shows that changing the input scale cannot create a degree that is absent at scale one. In particular, no odd degree $k > 1$ occurs, for any σ in the class and any input scale.

Corollary 4.5. Let $F ( x ) = R \sigma _ { B } ( A x )$ satisfy the assumptions of Theorem ${ \it 3 . 6 , }$ with σ homogeneous. Then

$$
\Pi _ { k } F = 0 \qquad f o r \ e v e r y \ o d d \ k \ge 3 .
$$

The same conclusion holds for finite sums of such layers.

Proof. Write $\sigma ( z ) = \lambda z + \mu \left| z \right|$ with $\begin{array} { r } { \lambda = \frac { m _ { + } + m _ { - } } { 2 } } \end{array}$ and $\begin{array} { r } { \mu = \frac { m _ { + } - m _ { - } } { 2 } } \end{array}$ . The second summand is even, so the odd part of the layer is

$$
{ \frac { F ( x ) - F ( - x ) } { 2 } } = \lambda R A x ,
$$

which is linear and therefore lies in the first chaos, while an even map has no odd chaos components at all. Hence $\Pi _ { k } F = 0$ for every odd $k \geq 3 .$ , and the same argument applies to a finite sum of such layers. □

This statement applies to a single layer, or a sum of such layers, but it does not imply the same restriction for their compositions.

4.2. Bias and input scale. We now show how a shared additive bias changes this conclusion.

Definition 4.6. For a threshold $c \in \mathbb { R }$ , define

$$
\sigma _ { c } ( x ) : = \operatorname { R e L U } ( x - c ) .
$$

For fixed equivariant maps $A : L  H$ and $R : H \to K$ , write $F _ { c } ( x ) = R ( \sigma _ { c } ) { } _ { B } ( A x )$

The same threshold is applied to every hidden coordinate, so $F _ { c }$ remains equivariant. The identity $\sigma _ { c } ( s z ) = s \sigma _ { c / s } ( z )$ gives $a _ { k } ( \sigma _ { c } ; s ) = s a _ { k } ( \sigma _ { c / s } )$ . Thus the support depends on the ratio $c / s$ . The following formula determines it explicitly.

Corollary 4.7. For every $s > 0$ and $k > 2$

$$
a _ { k } ( \sigma _ { c } ; s ) = \frac { s } { k ! } \mathrm { H e } _ { k - 2 } ( c / s ) \varphi ( c / s ) .
$$

Moreover, $a _ { 0 } ( \sigma _ { c } ; s )$ and $a _ { 1 } ( \sigma _ { c } ; s )$ are strictly positive.

Proof. Put $u = c / s$ . For $k \geq 2$

$$
a _ { k } ( \sigma _ { c } ; s ) = { \frac { s } { k ! } } \int _ { u } ^ { \infty } ( z - u ) { \mathrm { H e } } _ { k } ( z ) \varphi ( z ) d z = { \frac { s } { k ! } } { \mathrm { H e } } _ { k - 2 } ( u ) \varphi ( u ) ,
$$

where the second equality follows from two integrations by parts using $( \mathrm { H e } _ { j } \varphi ) ^ { \prime } = - \mathrm { H e } _ { j + 1 } \varphi$ The cases $k = 0 , 1$ follow directly from the same integral and are strictly positive. □

Corollary 4.8. Fix A, $R \neq 0$ and a degree bound $D \in  { \mathbb { N } } _ { 0 }$ . Let s be the common coordinate scale. For all but finitely many thresholds $c \in \mathbb { R }$

$$
\Pi _ { k } F _ { c } \neq 0 \quad \Longleftrightarrow \quad Q _ { k } \neq 0 \qquad ( 0 \leq k \leq D ) .
$$

Proof. The coeficients in degrees zero and one never vanish. For $2 \le k \le D$ , vanishing occurs precisely when $c / s$ is a root of $\mathrm { H e } _ { k - 2 }$ . Each of these finitely many polynomials has finitely many roots. Outside the resulting finite set of thresholds, every activation coeficient through degree D is nonzero. Apply Theorem 3.6. □

Thus a suitable shared bias retains every nonzero coordinate polynomial through a chosen degree. It cannot recover a degree for which $Q _ { k } = 0$ , or select additional polynomial directions with A and R fixed.

Unlike the positively homogeneous case, changing the input scale at a fixed nonzero threshold can change activation support, because it changes $c / s$ . This concerns only the scalar activation coeficients, so any change to the lift must also be accounted for in the coordinate polynomials.

4.3. An activation obstruction for $C _ { 8 }$ . The ungraded ReLU obstruction in this example is already part of the cyclic-group classification in [GTW25, Theorem 3B.4]. Here we identify its first permitted degree and explain the efect of a bias.

The real regular representation of $C _ { 8 }$ decomposes as

$$
{ \mathbb { R } } [ C _ { 8 } ] \cong { \bf 1 } \oplus \varepsilon \oplus \rho _ { 1 } \oplus \rho _ { 2 } \oplus \rho _ { 3 } ,
$$

where $\varepsilon$ is the sign character and ρ is the two-dimensional representation in which a $\rho _ { j }$ generator acts by rotation through $2 \pi j / 8$

Consider the two projected layers from $\rho _ { 1 }$ to $\rho _ { 3 }$ and from $\rho _ { 3 }$ to $\rho _ { 1 }$ .

The element $a ^ { 4 }$ acts as −I on $\rho _ { 1 }$ and on $\rho _ { 3 } ,$ , so Corollary 2.7 already shows that only odd degrees occur in either direction. Example 2.10 locates the first of them:

$$
D ( \rho _ { 1 } , \rho _ { 3 } ) = D ( \rho _ { 3 } , \rho _ { 1 } ) = \{ 3 , 5 , 7 , . . . \} .
$$

Indeed, for the first direction the character condition is $r \equiv \pm 3$ (mod 8), while for the reverse direction it is $3 r \equiv \pm 1$ (mod 8), which is equivalent to the same condition. Together with $| r | \leq k$ and $r \equiv k$ (mod 2), this permits exactly the odd degrees $k \geq 3$

By Proposition 3.14, the coordinate polynomials are nonzero in every permitted degree. For a positively homogeneous activation, however, Theorem 4.3 gives

$$
a _ { k } ( \sigma ; s ) = 0 \qquad ( k = 3 , 5 , 7 , \ldots ) .
$$

Hence every chaos component of either projected layer vanishes. Since these layers are continuous, Gaussian almost-everywhere vanishing implies that they vanish identically.

A shifted ReLU with $c \neq 0$ changes this conclusion. Its third coeficient is nonzero by Corollary $4 . 7 ,$ so both projected layers acquire a nonzero third-chaos component.

This distinguishes the example from Example 3.13. There, the quadratic coordinate polynomial vanishes and changing the activation cannot repair it. Here, the cubic coordinate polynomial is already nonzero and the bias supplies its missing scalar coeficient.

## 5. Architectures determined by the degree calculation

The preceding results concern one component of one layer. We now describe what a collection of such components can span, how to control its activation coeficients, and how these statements lead to explicit polynomial layers.

5.1. The degree subspace of a fixed collection of components. At a fixed degree, each branch contributes at most one polynomial direction. We first compute the span of those directions.

Fix equivariant $A _ { h } : L \to H _ { h }$ and $R _ { h } : H _ { h } \to K$ for $1 \leq h \leq M$ , where each $H _ { h } = \mathbb { R } ^ { B _ { h } }$ has a transitive coordinate basis $B _ { h }$ and $A _ { h } \neq 0$ . Write

$$
\begin{array} { r } { Q _ { h , k } ( x ) = R _ { h } \big ( ( A _ { h } x ) ^ { \odot k } \big ) , \qquad S _ { k } : = \operatorname { s p a n } \{ Q _ { h , k } : 1 \leq h \leq M \} \subseteq \operatorname { P o l y } _ { k } ( L , K ) ^ { G } . } \end{array}
$$

Consider the layer $\begin{array} { r } { F _ { \theta } ( x ) = \sum _ { h } \theta _ { h } R _ { h } ( \sigma _ { h } ) _ { { \mathcal B } _ { h } } ( A _ { h } x ) } \end{array}$ , with independent real output coeficients $\theta _ { h }$

Proposition 5.1. $I f a _ { k } ( \sigma _ { h } ; s _ { h } ) \ne 0$ for all h, where $s _ { h } ^ { 2 } = \mathrm { t r } ( A _ { h } A _ { h } ^ { * } ) /$ dim $H _ { h }$ , then

$$
\left\{ \Pi _ { k } F _ { \theta } : \theta \in \mathbb { R } ^ { M } \right\} = \Pi _ { k } S _ { k } .
$$

Its dimension is the rank of the positive semidefinite Gram matrix

$$
\mathsf { G } _ { h \ell } ^ { ( k ) } = k ! \mathrm { t r } \left( R _ { h } ( A _ { h } A _ { \ell } ^ { * } ) ^ { \circ k } R _ { \ell } ^ { * } \right) .
$$

Here $A _ { h } A _ { \ell } ^ { * } : H _ { \ell } \to H _ { h }$ is represented in the two coordinate bases; its entrywise power is therefore well-defined even when $H _ { h }$ and $H _ { \ell }$ have diferent dimensions. If an activation coeficient vanishes, omit that component before forming the span and Gram matrix. In

particular, the whole degree-k equivariant space is spanned exactly when this rank equals dim $\mathrm { H o m } _ { G } ( \mathrm { S y m } ^ { k } L , K )$ .

Proof. Fix the degree k. By Theorem 3.6, the hth branch contributes

$$
\Pi _ { k } ( R _ { h } ( \sigma _ { h } ) _ { { \mathcal B } _ { h } } ( A _ { h } x ) ) = \frac { a _ { k } ( \sigma _ { h } ; s _ { h } ) } { s _ { h } ^ { k } } \Pi _ { k } Q _ { h , k } .
$$

Thus, when $a _ { k } ( \sigma _ { h } ; s _ { h } ) \ne 0$ , the activation only rescales the degree-k direction $\Pi _ { k } Q _ { h , k }$ selected by that branch. Since the coeficients $\theta _ { h }$ are independent, varying θ therefore gives exactly

$$
\{ \Pi _ { k } F _ { \theta } : \theta \in \mathbb { R } ^ { M } \} = \mathrm { s p a n } \{ \Pi _ { k } Q _ { h , k } : 1 \leq h \leq M \} = \Pi _ { k } S _ { k } .
$$

It remains to determine the dimension of this span. The matrix ${ \sf G } ^ { ( k ) }$ is precisely the Gram matrix of the vectors $\Pi _ { k } Q _ { h , k }$ . The same calculation as in Theorem 3.6 gives

$$
\mathsf { G } _ { h \ell } ^ { ( k ) } = k ! \mathrm { ~ t r } \Big ( R _ { h } ( A _ { h } A _ { \ell } ^ { * } ) ^ { \circ k } R _ { \ell } ^ { * } \Big ) , \quad \mathrm { a n d } \quad \mathrm { r a n k } \mathsf { G } ^ { ( k ) } = \mathrm { d i m } \Pi _ { k } S _ { k } .
$$

Finally, $\Pi _ { k }$ is an isomorphism on homogeneous degree-k polynomials, so this accessible space equals the full degree-k equivariant polynomial space exactly when

$$
\operatorname { r a n k } \mathsf { G } ^ { ( k ) } = \dim \operatorname { H o m } _ { G } ( \operatorname { S y m } ^ { k } L , K ) .
$$

If $a _ { k } ( \sigma _ { h } ; s _ { h } ) = 0$ , the corresponding branch contributes nothing in degree k and may simply be omitted. □

This is the degree-k space accessible for the chosen lifts $A _ { h }$ and readouts $R _ { h }$ . Varying those maps may enlarge it. Moreover, although each degree can be analysed separately, the same coeficients $\theta _ { h }$ act across all degrees, so the degree components cannot in general be chosen independently.

This also yields an approximation obstruction. For $T \in L ^ { 2 } ( \gamma _ { L } ; K ) ^ { G }$ and any degree bound $D \geq 0$ 2

$$
\operatorname* { i n f } _ { \theta } \| T - F _ { \theta } \| _ { L ^ { 2 } } ^ { 2 } \geq \sum _ { k = 0 } ^ { D } \mathrm { d i s t } _ { L ^ { 2 } } \big ( \Pi _ { k } T , \Pi _ { k } S _ { k } \big ) ^ { 2 } ,
$$

since the degree-k component of $F _ { \theta }$ lies in $\Pi _ { k } S _ { k }$ . Thus any component of $\Pi _ { k } T$ outside this space gives an unavoidable error for the fixed architecture under the Gaussian reference measure.

The preceding proposition treats one degree at a time, but the same branch coeficient acts across all degrees. To control several degrees independently, we need activations with prescribed initial Hermite coeficients.

Proposition 5.2. Fix $D \geq 0$ . There exist real $t _ { 0 } , \ldots , t _ { D }$ such that, for every $( b _ { 0 } , \ldots , b _ { D } ) \in$ $\mathbb { R } ^ { D + \bar { 1 } }$ , there are $\theta _ { 0 } , \dots , \theta _ { D } \in \mathbb { R }$ for which the continuous piecewise-linear function

$$
\sigma ( z ) = \sum _ { r = 0 } ^ { D } \theta _ { r } \operatorname { R e L U } ( z - t _ { r } )
$$

satisfies $a _ { k } ( \sigma ; 1 ) = b _ { k } \ f o r \ 0 \leq k \leq D$ . Thus the first $D + 1$ Hermite coeficients can be chosen independently.

Proof. For $0 \le k \le D$ , set $f _ { k } ( t ) : = a _ { k } ( \mathrm { R e L U } ( \cdot - t ) )$ . By linearity of the Hermite coeficients, if $\begin{array} { r } { \sigma ( z ) = \sum _ { r = 0 } ^ { D } \theta _ { r } \mathrm { R e L U } ( z - t _ { r } ) } \end{array}$ , then

$$
a _ { k } ( \sigma ; 1 ) = \sum _ { r = 0 } ^ { D } \theta _ { r } f _ { k } ( t _ { r } ) .
$$

It is therefore enough to show that $( f _ { 0 } ( t ) , \ldots , f _ { D } ( t ) )$ , as t varies, spans $\mathbb { R } ^ { D + 1 }$ . If not, there exist coeficients $c _ { 0 } , \ldots , c _ { D }$ , not all zero, such that $\textstyle \sum _ { k = 0 } ^ { D } c _ { k } f _ { k } ( t ) = 0$ for every t. Diferentiating $f _ { k } ( t )$ twice gives $\begin{array} { r } { f _ { k } ^ { \prime \prime } ( t ) = \frac { 1 } { k ! } \mathrm { H e } _ { k } ( t ) \varphi ( t ) } \end{array}$ . Hence

$$
0 = \sum _ { k = 0 } ^ { D } c _ { k } f _ { k } ^ { \prime \prime } ( t ) = \varphi ( t ) \sum _ { k = 0 } ^ { D } \frac { c _ { k } } { k ! } \mathrm { H e } _ { k } ( t ) .
$$

Since $\varphi ( t ) > 0$ and the Hermite polynomials are linearly independent, $c _ { k } = 0$ for every $k .$ Therefore the vectors $( f _ { 0 } ( t ) , \ldots , f _ { D } ( t ) )$ span $\mathbb { R } ^ { D + 1 }$ . Choose $t _ { 0 } , \ldots , t _ { D }$ giving a basis. Then, for any prescribed $\left( b _ { 0 } , \ldots , b _ { D } \right)$ , suitable coeficients $\theta _ { r }$ give $a _ { k } ( \sigma ; 1 ) = b _ { k }$ for all $0 \le k \le D$ □

For a lift with coordinate variance $s ^ { 2 }$ , the preceding proposition may be applied after rescaling the hidden coordinates by s. Choosing $b _ { k } = \delta _ { k j }$ produces a piecewise-linear activation whose chaos components vanish in every degree $0 , \ldots , D$ except degree j. Components above degree D may still be present.

There are two simpler ways to obtain an exact degree-j layer. Using the polynomial activation ${ \mathrm { H e } } _ { j }$ coordinatewise gives the pure jth-chaos component

$$
R ( \mathrm { H e } _ { j } ) _ { \mathcal { B } } ( A x / s ) = s ^ { - j } \Pi _ { j } \big [ R \big ( ( A x ) ^ { \odot j } \big ) \big ] .
$$

Alternatively, $R \big ( ( A x ) ^ { \odot j } \big )$ is itself an ordinary homogeneous polynomial of degree $j .$

Thus the piecewise-linear construction controls only finitely many chaos coeficients, whereas the Hermite activation isolates a single chaos degree exactly. In particular, Proposition 5.2 does not control the higher-degree tail.

5.2. Regular lifts span every degree. Regular representations of finite groups give a simple construction showing that coordinatewise powers can realise every equivariant polynomial map of a fixed degree. The obstruction of Example 3.13 is therefore a property of the chosen coordinates, not an intrinsic one.

Proposition 5.3. Let G be finite and let $L , K$ be real orthogonal G-representations. For $u \in L$ define the linear map

$$
A _ { u } : L \longrightarrow \mathbb R G , \qquad ( A _ { u } x ) _ { g } = \langle x , g u \rangle ,
$$

where RG carries the left regular action, and for $v \in K$ define $A _ { v } : K $ RG analogously and put $R _ { v } : = | G | ^ { - 1 } A _ { v } ^ { * }$ , that is, $\begin{array} { r } { R _ { v } z = | G | ^ { - 1 } \sum _ { g \in G } z _ { g } g v } \end{array}$ . Then $A _ { u }$ and $R _ { v }$ are equivariant, their coordinate polynomial is

$$
Q _ { u , v , k } ( x ) = R _ { v } { \big ( } ( A _ { u } x ) ^ { \odot k } { \big ) } = { \frac { 1 } { | G | } } \sum _ { g \in G } \left. x , g u \right. ^ { k } g v ,
$$

and for every $k \geq 0$ these polynomials span $\mathrm { P o l y } _ { k } ( L , K ) ^ { G }$ as u and v vary.

Proof. Equivariance of $A _ { u }$ is the identity $( A _ { u } ( h x ) ) _ { g } = \langle x , h ^ { - 1 } g u \rangle = ( A _ { u } x ) _ { h ^ { - 1 } g } ,$ and $R _ { v }$ is a multiple of the adjoint of an equivariant map, hence equivariant. Since $A _ { v } ^ { * } e _ { g } = g v$ , the displayed formula for $Q _ { u , v , k }$ follows.

Pure powers of linear forms span the homogeneous scalar polynomials of degree $k ,$ , so the maps $P _ { u , v } ( x ) = \langle \boldsymbol { x } , \boldsymbol { u } \rangle ^ { k }$ v span $\mathrm { P o l y } _ { k } ( L , K )$ . Averaging, that is,

$$
\mathcal { R } ( P ) ( x ) = | G | ^ { - 1 } \sum _ { g } g P ( g ^ { - 1 } x ) ,
$$

projects $\mathrm { P o l y } _ { k } ( L , K )$ onto $\mathrm { P o l y } _ { k } ( L , K ) ^ { G }$ , and orthogonality of the action gives $\mathcal { R } ( P _ { u , v } ) =$ $Q _ { u , v , k }$ . A projection maps a spanning set of the ambient space to a spanning set of its image. □

The proposition gives a universal, but generally ineficient, way to match any prescribed equivariant polynomial components through a finite degree bound D. For each required degree k, choose finitely many $Q _ { u , v , k }$ spanning the desired component, with u normalised so that the coordinates of $A _ { u } X$ have variance one.

By Proposition 5.2, one may choose a piecewise-linear activation whose Hermite coeficients vanish in all degrees $0 , \ldots , D$ except degree k. Then (6) shows that the corresponding layer contributes, up to degree D, only the chosen degree-k coordinate polynomial. Summing over the required degrees gives the prescribed chaos components.

Thus copies of the regular representation are suficient to recover every equivariant polynomial direction. The cost is that each regular block has |G| hidden coordinates, and spanning several directions may require several such blocks.

## 6. Subset representations of the symmetric group

We now make the ambient polynomial spaces from Section 2 explicit for an important family of permutation representations. Subset representations model vertex, edge, and higher-order subset features in equivariant graph networks. This section describes the full polynomial spaces against which the coordinate-selected spaces of Sections 3 and 5 can be compared.

Definition 6.1. Let $[ n ] = \{ 1 , \dots , n \}$ , with $[ 0 ] = \emptyset$ , and let $S _ { n }$ be its permutation group. For $r \geq 0$ , define

$$
X _ { n , r } : = { \binom { [ n ] } { r } } = \{ A \subseteq [ n ] : | A | = r \} .
$$

The r-subset representation is the real vector space

$$
V _ { n , r } : = \mathbb { R } [ X _ { n , r } ] = \bigoplus _ { A \in X _ { n , r } } \mathbb { R } e _ { A } ,
$$

equipped with the inner product for which the basis $\{ e _ { A } : A \in X _ { n , r } \}$ is orthonormal. The action of $S _ { n }$ is defined by $g \cdot e _ { A } = e _ { g A }$ , where $g A = \{ g ( a ) : a \in A \}$

The space $V _ { n , 0 } \cong \mathbb { R }$ is the trivial representation. Intuitively, $V _ { n , 1 }$ consists of vertex features, $V _ { n , 2 }$ of features on unordered pairs of distinct vertices, and $V _ { n , 3 }$ of features on unordered triples. Pair features may be viewed as real edge weights. Triple features are assigned to all triples, whether or not those triples form triangles in a particular graph. Throughout, these spaces contain arbitrary real-valued features on every subset of the indicated size.

Fix an input subset size $r \geq 1$ , an output subset size $s \geq 0$ , and a polynomial degree $d \geq 0$ . Our objective is to construct an explicit basis of

$$
\mathrm { P o l y } _ { d } ( V _ { n , r } , V _ { n , s } ) ^ { S _ { n } } \cong \mathrm { H o m } _ { S _ { n } } ( \mathrm { S y m } ^ { d } V _ { n , r } , V _ { n , s } ) .
$$

An element of this space has the form

$$
P ( x ) = \sum _ { Q \in X _ { n , s } } P _ { Q } ( x ) e _ { Q } ,
$$

where each $P _ { Q }$ is a homogeneous polynomial of degree d in the input coordinates $x _ { A }$ Equivariance means $P ( g \cdot x ) = g \cdot P ( x )$ , or equivalently, $P _ { g Q } ( g \cdot x ) = P _ { Q } ( x )$ for $g \in S _ { n } , Q \in$ $X _ { n , s }$

A degree-d input monomial $x _ { A _ { 1 } } \cdot \cdot \cdot x _ { A _ { d } }$ records d input subsets, with repetitions allowed. An output coordinate also specifies an s-subset Q. The basis constructed below is indexed by these configurations up to simultaneous relabelling of the vertices. Each basis map sums the monomials belonging to one such configuration type.

6.1. Symmetric powers are multihypergraph modules. To describe symmetric powers combinatorially, for a finite G-set X and $d \geq 0$ , let $\operatorname { M S e t } _ { d } ( X )$ denote the size-d multisets on X with the induced action.

Proposition 6.2. For a finite group G and a finite G-set X,

$$
\operatorname { S y m } ^ { d } ( \mathbb { R } [ X ] ) \cong \mathbb { R } [ \operatorname { M S e t } _ { d } ( X ) ]
$$

as G-representations. ${ H m } _ { \omega }$ represents an orbit $\omega \in \mathrm { M S e t } _ { d } ( X ) / G$ and $H _ { \omega } = \mathrm { S t a b } _ { G } ( m _ { \omega } )$ ， then

$$
\operatorname { S y m } ^ { d } ( \mathbb { R } [ X ] ) \cong \bigoplus _ { \omega \in \mathrm { M S e t } _ { d } ( X ) / G } { \mathrm { I n d } } _ { H _ { \omega } } ^ { G } \mathbf { 1 } .
$$

Proof. The monomials $\begin{array} { r } { e ^ { m } = \prod _ { x \in X } e _ { x } ^ { m ( x ) } } \end{array}$ for $m \in \operatorname { M S e t } _ { d } ( X )$ form a basis of $\operatorname { S y m } ^ { d } ( \mathbb { R } [ X ] )$ and $g e ^ { m } = e ^ { g m }$ . Each orbit therefore spans a permutation module isomorphic to $\mathbb { R } [ G / H _ { \omega } ] =$ ${ \mathrm { I n d } } _ { H _ { \omega } } ^ { G } \mathbf { 1 }$ □

Definition 6.3. An r-uniform multihypergraph on [n] with d edges is a multiset

$$
\mathcal { H } = \{ E _ { 1 } , \dots , E _ { d } \} , \qquad E _ { i } \in X _ { n , r } ,
$$

where repeated edges are allowed.

• A vertex is active if it lies in at least one edge of H. The subhypergraph obtained by deleting all isolated vertices is called the active core.

• Two multihypergraphs are isomorphic if there is a bijection between their vertex sets that preserves edge multiplicities. We write Aut(H) for the automorphism group of H.

• Let ${ \mathfrak { H } } _ { r , d }$ denote the set of isomorphism classes of r-uniform multihypergraphs with d edges and no isolated vertices. For $\mathcal { H } \in \mathfrak { H } _ { r , d }$ , write $v ( \mathcal { H } )$ for its number of vertices.

Take $X = X _ { n , r }$ . A monomial $e _ { A _ { 1 } } \cdot \cdot \cdot e _ { A _ { d } }$ corresponds to the edge multiset $\{ A _ { 1 } , \ldots , A _ { d } \}$ and two monomials lie in the same $S _ { n } { \mathrm { - o r b i t } }$ exactly when their active cores are isomorphic. Every core satisfies $v ( { \mathcal { H } } ) \leq r d$ , with equality precisely when its edges are pairwise disjoint.

For $v ( \mathcal { H } ) \leq n$ , let $P _ { \mathcal { H } } ( n )$ denote the span of all monomials whose active core is isomorphic to H.

The active core is exactly the orbit data, so it gives the desired decomposition of the symmetric power.

Theorem 6.4. For $n \geq 0 , r \geq 1$ and $d \geq 0$

$$
\operatorname { S y m } ^ { d } V _ { n , r } = \bigoplus _ { \stackrel { [ \mathcal { H } ] \in \mathfrak { H } _ { r , d } } { v ( \mathcal { H } ) \le n } } P _ { \mathcal { H } } ( n ) , \qquad w h e r e \qquad P _ { \mathcal { H } } ( n ) \cong \operatorname { I n d } _ { \operatorname { A u t } ( \mathcal { H } ) \times S _ { n - v ( \mathcal { H } ) } } ^ { S _ { n } } \mathbf { 1 } .
$$

The subgroup acts by automorphisms on a labelled copy of the core and by arbitrary permutations on its complement. For $n \geq r d$ , every core type occurs.

For each partition $\lambda \vdash n _ { : }$ , the multiplicity of the irreducible Specht module $S ^ { \lambda }$ is

$$
[ P _ { { \mathcal { H } } } ( n ) : S ^ { \lambda } ] = \dim ( S ^ { \lambda } ) ^ { \mathrm { A u t } ( { \mathcal { H } } ) \times S _ { n - v ( { \mathcal { H } } ) } } .
$$

Proof. See Section A.3.

This is a decomposition into orbit spans, not generally into irreducible representations. Each summand records one overlap pattern among the input subsets. Importantly, the indexing set stabilises for $n \geq r d$ , although the modules $P _ { \mathcal { H } } ( n )$ themselves still depend on n.

Example 6.5. For $r = 2 , d = 2$ and $n \geq 4$ , there are three core types: a repeated edge, two adjacent edges and two disjoint edges. Writing $e _ { a b } = e _ { \{ a , b \} }$ , the representatives are $e _ { 1 2 } ^ { 2 }$ e<sub>12</sub>e<sub>23</sub> and $e _ { 1 2 } e _ { 3 4 }$ , respectively. Hence

$$
\mathrm { S y m } ^ { 2 } V _ { n , 2 } = P _ { \mathrm { r e p } } ( n ) \oplus P _ { \mathrm { a d j } } ( n ) \oplus P _ { \mathrm { d i s j } } ( n ) .
$$

For general $^ { r , }$ the orbit of $e _ { A } e _ { B }$ is determined by $t = | A \cap B |$ , where max $( 0 , 2 r - n ) \leq t \leq r$ $\operatorname { I f } t < r ,$ its stabiliser is $S _ { t } \times ( S _ { r - t } \ : 2 \ : S _ { 2 } ) \times S _ { n - 2 r + t }$ . The factors permute the intersection, the two exclusive parts (allowing their exchange), and the unused vertices. If $t = r$ , the edge is repeated and the stabiliser is $S _ { r } \times S _ { n - r } .$

Symmetric powers of permutation modules have also been studied in the modular setting [Jia21]. Here the orbit description is useful because it keeps the overlap data visible without first decomposing into irreducibles. For example, at $n = r d$ the disjoint-edge summand is the Foulkes module $\mathrm { I n d } _ { S _ { r } \ell S _ { d } } ^ { S _ { r d } } \mathbf { 1 }$ , whose general irreducible decomposition is an open problem [PW16].

6.2. Polynomial equivariants: the marked-orbit basis. The marked-orbit basis describes all homogeneous polynomial maps from r-subset to s-subset features. We begin with the classical linear case.

Proposition 6.6. For finite G-sets $X , Y$ we have Hom $\mathsf { \iota } _ { G } ( \mathbb { R } [ X ] , \mathbb { R } [ Y ] ) \cong \mathbb { R } [ X \times Y ] ^ { G }$ , with basis the maps

$$
T _ { \mathcal O } ( e _ { x } ) = \sum _ { y : ( x , y ) \in \mathcal O } e _ { y }
$$

indexed by the G-orbits ${ \mathcal { O } } \subseteq X \times Y$

Proof. A linear map T with matrix $k _ { T } ( x , y )$ is equivariant if and only if $k _ { T } ( g x , g y ) =$ $k _ { T } ( x , y )$ , that is, if and only if k is constant on diagonal orbits. See also [GTW25, Lemma 2E.3] and [MBHSL19]. □

By Proposition 6.2 we may apply this with $X = \mathrm { M S e t } _ { d } ( X _ { n , r } )$ , which leads to the following description of the orbits.

Definition 6.7. A marked $d e g r e e { \boldsymbol { \cdot } } d \left( { \boldsymbol { r } } , { \boldsymbol { s } } \right)$ -interaction is a pair $( A , Q )$ with $\mathcal { A } \in \operatorname { M S e t } _ { d } ( X _ { n , r } )$ and $Q \in X _ { n , s } .$ , that is, an r-uniform d-edge multihypergraph on [n] together with a distinguished s-subset of vertices. Its type is its orbit under simultaneous relabelling. Marked vertices need not be active.

Theorem 6.8. $\mathrm { P o l y } _ { d } ( V _ { n , r } , V _ { n , s } ) ^ { S _ { n } } \cong \mathrm { H o m } _ { S _ { n } } ( \mathrm { S y m } ^ { d } V _ { n , r } , V _ { n , s } )$ , with a basis indexed by the diagonal S<sub>n</sub>-orbits ${ \mathcal { O } } \subseteq \operatorname { M S e t } _ { d } ( X _ { n , r } ) \times X _ { n , s }$ . For such an orbit, define

$$
\Psi _ { \mathcal { O } } ( x ) : = \sum _ { Q \in X _ { n , s } } \left( \sum _ { \mathcal { A } : ( \mathcal { A } , Q ) \in \mathcal { O } } x ^ { \mathcal { A } } \right) e _ { Q } , \qquad x ^ { \mathcal { A } } : = \prod _ { A \in X _ { n , r } } x _ { A } ^ { ^ { m _ { A } ( A ) } } .
$$

Then the maps $\Psi _ { \mathcal { O } }$ form a basis of ${ \mathrm { P o l y } } _ { d } ( V _ { n , r } , V _ { n , s } ) ^ { S _ { n } }$

Proof. By Proposition 6.2, a degree-d monomial is indexed by a multiset ${ \mathcal { A } } \in \operatorname { M S e t } _ { d } ( X _ { n , r } )$ Hence a polynomial map $V _ { n , r } ~  ~ V _ { n , s }$ is determined by coeficients $^ { c _ { A , Q } }$ indexed by $( \mathcal { A } , Q ) \in \mathrm { M S e t } _ { d } ( X _ { n , r } ) \times X _ { n , s }$ . By Proposition $6 . 6 .$ , such a map is S<sub>n</sub>-equivariant exactly when these coeficients are constant on diagonal $S _ { n } { \mathrm { - o r b i t s . } }$ . The corresponding orbit indicators are precisely the maps $\Psi _ { \mathcal { O } }$ □

Under the polarisation identification of Proposition 1.1, the usual multinomial factors only rescale the multiset basis vectors. These factors are constant on $S _ { n } { \mathrm { - o r b i t s } }$ , so they do not change the orbit indexing or the basis statement.

Orbit-indexed graph-polynomial bases also appear in $[ \mathrm { P L K ^ { + } 2 3 } ]$ .The orbit basis is explicit, but we also want its dimension and stable range. Fixing the active core reduces the count to the possible positions of the output mark.

Proposition 6.9. Fix an active core H with vertex set $U = V ( \mathscr { H } )$ , and write $v = v ( \mathcal { H } )$ . A marked type with core H is determined by $q : = | Q \cap U |$ together with the Aut(H)-orbit of the active part $Q \cap U \in \binom { U } { q }$ . Consequently,

$$
\dim \mathrm { H o m } _ { S _ { n } } ( \mathrm { S y m } ^ { d } V _ { n , r } , V _ { n , s } ) = \sum _ { \stackrel { [ \mathcal { H } ] \in \mathfrak { H } _ { r , d } } { v ( \mathcal { H } ) \leq n } } \sum _ { \substack { q = \operatorname* { m a x } ( 0 , s - ( n - v ( \mathcal { H } ) ) ) } } ^ { \operatorname* { m i n } ( s , v ( \mathcal { H } ) ) } | \binom { \vphantom { \int _ { q } ^ { q } } { v ( \mathcal { H } ) } } { q } / \mathrm { A u t } ( \mathcal { H } ) | .\tag{7}
$$

Proof. By Theorem 6.4 and Frobenius reciprocity, the contribution of a fixed core $\mathcal { H }$ is dim $\dot { ( V _ { n , s } ) } ^ { \mathrm { A u t } ( \mathcal { H } ) \times S _ { n - v } }$ . Since $V _ { n , s } = \mathbb { R } [ X _ { n , s } ]$ is a permutation representation, this dimension is the number of $( \operatorname { A u t } ( \mathcal { H } ) \times S _ { n - v } )$ -orbits on s-subsets $Q \subseteq [ n ]$

Write $q = | Q \cap U | . \ \mathrm { A u t } ( \mathcal { H } )$ acts on the active part $Q \cap U$ , while $S _ { n - v }$ is transitive on all $( s - q )$ -subsets of the inactive vertices. Thus, for fixed $q .$ , the orbit of Q is determined exactly by the A-orbit of $Q \cap U \in \binom { U } { q }$

Such a marking exists precisely when $0 \leq q \leq v$ and $0 \leq s - q \leq n - v$ , giving the stated limits. Summing over the core types proves (7). □

By Burnside’s lemma, evaluating (7) requires only the cycle structures of the automorphism groups. Theorem C.2 computes the same number by summing over conjugacy classes of $S _ { n }$ instead.

The counting formula also shows exactly when increasing n can no longer create new marked types.

Theorem 6.10. Let $r , d \geq 1$ and $s \geq 0$ , and write $D ( n ) : = \dim \operatorname { H o m } _ { S _ { n } } ( { \mathrm { S y m } } ^ { d } V _ { n , r } , V _ { n , s } )$ Then:

(a) $D ( n )$ is non-decreasing in n.

(b) For $n \geq r d + s$

$$
D ( n ) = \sum _ { [ \mathcal { H } ] \in \mathfrak { H } _ { r , d } } \sum _ { q = 0 } ^ { \operatorname* { m i n } ( s , v ( \mathcal { H } ) ) }  \binom { V ( \mathcal { H } ) } { q }  \operatorname { A u t } ( \mathcal { H } )  ,
$$

so $D ( n )$ is independent of n in this range.

(c) The threshold is sharp: $D ( r d + s - 1 ) < D ( r d + s )$

In the stable range, the basis $\{ \Psi \scriptscriptstyle O \}$ is indexed by a set independent of $n ,$ so one coeficient vector defines one equivariant polynomial layer simultaneously for every $n \geq r d + s$

Proof. In (7) the set of admissible types grows with n and each lower summation limit is non-increasing in $n ,$ so D is non-decreasing. For $n \geq r d + s$ we have $n - v \geq s$ for every core, so every lower limit is 0 and no term depends on $n ,$ which gives $( b )$

For sharpness, let $\mathcal { H } _ { 0 }$ be the disjoint type, with $v ( \mathcal { H } _ { 0 } ) = r d .$ . At $n = r d + s - 1$ the value $q = 0$ would require $s - q \leq n - v = s - 1$ , which fails, so $q = 0$ is inadmissible for $\mathcal { H } _ { 0 }$ Also when $s = 0$ the core $\mathcal { H } _ { 0 }$ does not embed at all. At $n = r d + s$ it becomes admissible and contributes exactly one further orbit, while no other term decreases. Hence the strict inequality. □

6.3. Worked example: seven quadratic edge-to-vertex maps. Take $( r , d , s ) =$ (2, 2, 1) and write $x _ { u v } = x _ { \{ u , v \} }$ . By Theorem 6.8 we must list the marked types: a repeated edge with the mark on it or of it; an adjacent pair with the mark at the centre, at an endpoint, or outside; a disjoint pair with the mark on an active vertex or outside. These give seven types (figure 1), hence dim Hom ${ \cal S } _ { n } ( \mathrm { S y m } ^ { 2 } V _ { n , 2 } , V _ { n , 1 } ) = 7$ for $n \geq 5$ . At $n = 3$ only four types fit, while at $n = 4$ only the disjoint pair with an outside mark is absent. Thus the dimensions are $4 , 6 , 7 , 7 , \ldots$ . for $n = 3 , 4 , 5 , 6 , \dots$

![](images/c9b20c862301d37091d95e13bdefb571edc976a7a419fba1a71f148fb11d27af.jpg)  
Figure 1. The seven marked interaction types for quadratic maps $V _ { n , 2 } $ $V _ { n , 1 }$ in the stable range. Filled vertices carry the output mark $v ;$ double lines denote a repeated input edge.

Writing the vth output coordinate gives the following basis, using the same labels as figure 1.

$$
\begin{array} { r l r l r l } & { \begin{array} { r l r l r l } & { \langle \mathbf { \phi } _ { 1 2 } | \mathbf { z } \rangle , \displaystyle - \sum _ { u = 1 } ^ { \infty } x _ { u = 1 } ^ { 2 } , } & & & { \mathrm { z e p o s s c i e d ~ d i d g e ~ , ~ m a t a t i o n : ~ o n } } \\ & { \langle \mathbf { \phi } _ { 1 2 } | \mathbf { z } \rangle , \quad \mathbf { w } _ { u = 1 } ^ { 2 } , } & & & { \mathrm { z e p e r a t i a l e d ~ i n g ~ , ~ m a t a t i o n : ~ o n } } \\ & { \langle \mathbf { \phi } _ { 2 3 } | \mathbf { z } \rangle , \quad \mathbf { w } _ { u = 1 } ^ { 2 } , } & & & { \mathrm { z e p e r a t i a l e d ~ i n g ~ , ~ m a t a t i o n : ~ o n } } \\ & { \langle \mathbf { \phi } _ { 3 2 } | \mathbf { z } \rangle , \quad \mathbf { \phi } _ { 1 3 } ^ { \mathrm { ~ o p ~ } } = : \displaystyle \sum _ { u = 1 } ^ { 2 } x _ { u = 1 } ^ { 2 } , } & & & { \mathrm { z e p t a i n g ~ r i a l e d ~ , ~ r i g h t . } } \\ & { \langle \mathbf { \phi } _ { 3 2 } | \mathbf { z } \rangle , \quad \mathbf { \phi } _ { 1 4 } ^ { \mathrm { ~ o p ~ } } = : \displaystyle \sum _ { u = 1 } ^ { 2 } x _ { u = 1 } ^ { 3 } x _ { u = 1 } ^ { 2 } , } & & & { \mathrm { z e p t a i n g ~ , ~ e p t a i n i t i o n : } } \end{array} , } \end{array}
$$

In the last sum, $E _ { 1 }$ and $E _ { 2 }$ are two-element subsets of [n] and their pair is unordered. Each map is an orbit sum of one marked type. Their monomial supports are disjoint, so the maps are independent and the dimension count shows that they form a basis.

The basis also shows the geometry directly. $\Phi _ { 1 }$ is a local square statistic at v, while $\Phi _ { 3 }$ and $\Phi _ { 4 }$ are quadratic contractions along length-two paths through or ending at v. The other four use configurations that are not local path contractions at v.

6.4. Evaluation of the seven quadratic operations. For implementation, the seven orbit sums can be evaluated without enumerating triples, quadruples or five-tuples. Write $X _ { u v } = x _ { u v }$ for u $\neq v$ and $X _ { v v } = 0$ , and define

$$
h = X { \bf 1 } , \qquad q _ { v } = \sum _ { u \ne v } x _ { u v } ^ { 2 } , \qquad S = \sum _ { u < v } x _ { u v } , \qquad T = \sum _ { u < v } x _ { u v } ^ { 2 } .
$$

Let

$$
W = \frac { 1 } { 2 } \sum _ { v } ( h _ { v } ^ { 2 } - q _ { v } ) , \qquad D _ { \mathrm { d i s j } } = \frac { 1 } { 2 } ( S ^ { 2 } - T ) - W .
$$

These are the total adjacent-pair and disjoint-pair products, respectively. All quantities refer to scalar, real-valued edge features.

Proposition 6.11. The seven basis maps of Section 6.3 satisfy

$$
\begin{array} { r l r l } & { ( \Phi _ { 1 } x ) _ { v } = q _ { v } , \qquad } & & { ( \Phi _ { 2 } x ) _ { v } = T - q _ { v } , } \\ & { ( \Phi _ { 3 } x ) _ { v } = \frac 1 2 ( h _ { v } ^ { 2 } - q _ { v } ) , \qquad } & & { ( \Phi _ { 4 } x ) _ { v } = ( X h ) _ { v } - q _ { v } , } \\ & { ( \Phi _ { 5 } x ) _ { v } = W - ( \Phi _ { 3 } x ) _ { v } - ( \Phi _ { 4 } x ) _ { v } , } \\ & { ( \Phi _ { 6 } x ) _ { v } = S h _ { v } - h _ { v } ^ { 2 } - ( X h ) _ { v } + q _ { v } , \qquad } & & { ( \Phi _ { 7 } x ) _ { v } = D _ { \mathrm { d i s j } } - ( \Phi _ { 6 } x ) _ { v } . } \end{array}
$$

For $n \geq 5$ , every homogeneous quadratic equivariant map $V _ { n , 2 } \to V _ { n , 1 }$ is therefore a unique linear combination of these seven maps and can be evaluated in $O ( n ^ { 2 } )$ arithmetic operations. With m nonzero edge features supplied sparsely, the cost is $O ( n + m )$

Proof. Expanding $h _ { v } ^ { 2 }$ separates repeated edges from unordered pairs of distinct edges incident to $v ,$ giving $\Phi _ { 3 }$ . Expanding $( X h ) _ { v }$ separates the returning path $v , u , v$ , which contributes $q _ { v } ,$ , from paths with three distinct vertices, giving $\Phi _ { 4 }$ . Every adjacent edge pair places v at its centre, at an endpoint, or outside. Subtracting the first two contributions from W gives $\Phi _ { 5 }$

For a fixed edge $\{ v , u \}$ , the sum of features on edges disjoint from it is $S - h _ { v } - h _ { u } + x _ { v u }$ Multiplying by $x _ { v u }$ and summing over u gives $\Phi _ { 6 }$ . The total product over unordered distinct edge pairs is $( S ^ { 2 } - T ) / 2$ . Subtract adjacent pairs to obtain $D _ { \mathrm { d i s j } }$ , then subtract those containing v to obtain $\Phi _ { 7 }$ . The first two identities are immediate from the definitions. Only edge sums, the two matrix–vector products X1 and $X h ,$ and vertexwise arithmetic are required. Completeness follows from Theorem 6.8. □

For dense inputs, the $O ( n ^ { 2 } )$ cost is of the same order as reading the $\binom { n } { 2 }$ edge coordinates; for sparse inputs, the $O ( n + m )$ bound is linear in the stored input size up to the vertex term.

An explicit quadratic layer is

$$
F _ { \theta } ( x ) = \sum _ { a = 1 } ^ { 7 } \theta _ { a } \Phi _ { a } ( x ) .
$$

Together with the equivariant linear terms and constant output, this spans all equivariant polynomial maps of degree at most two. The seven-dimensional statement concerns the quadratic polynomial space on the full real feature space. The quadratic products in $\Phi _ { a }$ are explicit operations of the architecture, so this is not a claim that finite ReLU networks can exactly realise quadratic functions on $\mathbb { R } ^ { \binom { n } { 2 } }$ . If inputs are restricted to Boolean adjacency matrices, additional identities such as $x _ { u v } ^ { 2 } = x _ { u v }$ may introduce dependencies between polynomial expressions.

## 7. Scope and future directions

Limitations. It remains open to characterise which sequences of chaos components come from equivariant piecewise-linear maps; see Corollary 2.2. The transform determines each such map uniquely, but equivariance and square-summability of the components do not ensure that the resulting map is piecewise linear. A complete structural picture is therefore still missing. Matching finitely many components with shifted ReLUs also leaves the higher-degree tail uncontrolled, so it gives no approximation bound on its own. For compact connected groups, polynomial existence does not sufice to detect piecewise-linear existence; the additional restrictions are discussed in the introduction.

At the architectural level, the Gram criterion describes the space available in each degree for fixed lifts and readouts. A characterisation of networks with prescribed width and depth, allowing these maps to be trained, remains open here. It must account for shared parameters across degrees and for interactions created by composition and joint processing of source components.

The general spanning construction can also be costly: each regular hidden block has |G| coordinates. For subset features, the multihypergraph orbit modules can still be reducible, so a full irreducible decomposition requires further work. Restricting the polynomial bases to Boolean inputs can introduce additional relations. Finally, the expressivity results do not establish guarantees about training or generalisation.

Constructing architectures. The main practical task is to turn these constructions into trainable networks with prescribed degree spaces. Choose source and target modules $L , K$ a degree budget D, and equivariant lifts $A _ { h } : L \to H _ { h }$ and readouts $R _ { h } : H _ { h } \to K$ , where each $H _ { h }$ is a transitive permutation module. Use the Gram test in Proposition 5.1 to select coordinate polynomials $R _ { h } ( ( A _ { h } x ) ^ { \odot k } )$ spanning the intended spaces, and choose activations with nonzero coeficients at the required degrees and scales. Independent branch weights then parametrise the computed span in each degree. For separate degree control, Section 5 provides coordinatewise powers for homogeneous polynomial layers, normalised Hermite activations for pure chaos layers, and shifted-ReLU combinations for matching coeficients through D. The polynomial constructions need not be piecewise linear, while finite ReLU coeficient matching leaves an uncontrolled higher-degree tail.

Graph networks. The companion paper [Sha26] applies this theory to predicting graph diameter and algebraic connectivity. The reported gains over a parameter-matched control with full linear $S _ { n ^ { - } } \mathrm { e q u i v a r i a n t }$ mixing concern one synthetic benchmark with a frozen GIN backbone. The subset representations studied here place this construction within a broader family of vertex, edge and higher-order features. In each degree, the marked-orbit basis gives all polynomial maps between these spaces, guiding layer construction and tests of which directions a proposed layer spans (Theorem 6.8). Its stable indexing allows coeficients to be shared across graph orders (Theorem 6.10).

A further direction is to adapt the GNN architecture to classify d-manifolds from their triangulations. Equivariance under vertex relabelling alone is insuficient: subdivisions and retriangulations, such as Pachner moves, can change both the vertex set and the incidence structure. Extending the theory to this setting requires specifying how features are transferred between triangulations and how network layers respect these transfers, with the aim of making the final prediction invariant under the chosen changes of triangulation.

## Appendix A. Supplementary proofs

This appendix supplies the polarisation and character-theoretic arguments used in the main text.

## A.1. Proof of Proposition 1.1.

Proof. Polarisation identifies a homogeneous polynomial map P of degree k with a symmetric k-linear map $\widetilde { P }$ satisfying $P ( x ) = \widetilde P ( x , \dots , x )$ , and the universal property of $\operatorname { S y m } ^ { k } L$ converts $\widetilde { P }$ into a unique linear $T _ { P } : \mathrm { S y m } ^ { k } L  K$ with $T _ { P } ( x ^ { k } ) = P ( x )$ . The two constructions are mutually inverse and natural in $L$ and K, so $P ( g x ) = g P ( x )$ for all $g$ and x if and only if $T _ { P }$ commutes with the G-action. □

## A.2. Proof of Proposition 2.8.

Proof. Diagonalising $\rho _ { L } ( g )$ over $\mathbb { C }$ gives $\begin{array} { r } { \sum _ { k } \chi _ { \mathrm { S y m } ^ { k } L } ( g ) t ^ { k } = \operatorname* { d e t } ( I - t \rho _ { L } ( g ) ) ^ { - 1 } } \end{array}$ , and taking logarithms,

$$
\log \operatorname* { d e t } \bigl ( I - t \rho _ { L } ( g ) \bigr ) ^ { - 1 } = \sum _ { m \geq 1 } \frac { t ^ { m } } { m } \operatorname { t r } \rho _ { L } ( g ^ { m } ) = \sum _ { m \geq 1 } \frac { t ^ { m } } { m } \chi _ { L } ( g ^ { m } ) .
$$

Real characters are self-conjugate, so $c _ { k } = \langle \chi _ { \mathrm { S y m } ^ { k } L } , \chi _ { K } \rangle$ , and averaging over G gives both displayed forms. □

## A.3. Proof of Theorem 6.4.

Proof. A basis of $\mathrm { S y m } ^ { d } V _ { n , r }$ is given by the monomials

$$
e ^ { m } = \prod _ { A \in X _ { n , r } } e _ { A } ^ { m ( A ) } , \qquad m : X _ { n , r } \to \mathbb { N } _ { 0 } , \qquad \sum _ { A } m ( A ) = d .
$$

The action of $S _ { n }$ is

$$
g \cdot e ^ { m } = e ^ { g m } , \qquad ( g m ) ( A ) = m ( g ^ { - 1 } A ) .
$$

Thus each monomial may be identified with an r-uniform multihypergraph on [n] having $m ( A )$ copies of the edge A.

A permutation g maps the active core of m isomorphically onto that of $g m$ . Conversely, any isomorphism of active cores extends arbitrarily to a permutation of their complements in [n]. Hence two monomials lie in the same $S _ { n }$ -orbit if and only if their active cores are isomorphic, and

$$
\operatorname { S y m } ^ { d } V _ { n , r } = \bigoplus _ { [ \mathcal { H } ] \in \mathfrak { H } _ { r , d } } P _ { \mathcal { H } } ( n ) .
$$

Fix a monomial with active core $\mathcal { H } ,$ , active vertex set $U$ , and $v = | U |$ . Its stabiliser consists precisely of automorphisms of the core together with arbitrary permutations of the inactive vertices. Therefore $\mathrm { S t a b } _ { S _ { n } } ( m ) \cong \mathrm { A u t } ( \mathcal { H } ) \times S _ { n - v ; }$ , and the corresponding orbit module is the associated permutation representation, $P _ { \mathcal { H } } ( n ) \cong { \mathrm { I n d } } _ { \mathrm { A u t } ( \mathcal { H } ) \times S _ { n - v } } ^ { S _ { n } } \ \mathbf { 1 }$

Since H has d edges, each of size $r , v ( \mathcal { H } ) \leq r d .$ . Thus every core type occurs once $n \geq r d .$ so the indexing set is independent of n in this range.

Finally, for $\lambda \vdash n .$ , absolute irreducibility of the real Specht modules and Frobenius reciprocity give

$$
\begin{array} { r l } & { [ P _ { \mathcal { H } } ( n ) : S ^ { \lambda } ] = \dim \operatorname { H o m } _ { S _ { n } } ( P _ { \mathcal { H } } ( n ) , S ^ { \lambda } ) } \\ & { \qquad = \dim \operatorname { H o m } _ { \operatorname { A u t } ( \mathcal { H } ) \times S _ { n - v } } \left( \mathbf { 1 } , \operatorname { R e s } S ^ { \lambda } \right) } \\ & { \qquad = \dim ( S ^ { \lambda } ) ^ { \operatorname { A u t } ( \mathcal { H } ) \times S _ { n - v } } . } \end{array}
$$

This proves the result.

## Appendix B. Hidden spaces with several coordinate orbits

The common scalar in Theorem 3.6 requires equal coordinate variances. With several coordinate orbits there is still an exact formula, but the scalar becomes a diagonal matrix.

Let $A : L \to \mathbb { R } ^ { B }$ and $R : \mathbb { R } ^ { B }  K$ be equivariant linear maps for an arbitrary permutation basis B. Put $C = A A ^ { * }$ and $s _ { b } = \sqrt { C _ { b b } }$ . Assume $\sigma ( s _ { b } \cdot ) \in L ^ { \bar { 2 } } ( \gamma _ { \mathbb { R } } )$ whenever $s _ { b } > 0$ . Let $D _ { k }$ be diagonal with

$$
( D _ { k } ) _ { b b } = \left\{ { \begin{array} { l l } { a _ { k } ( \sigma ; s _ { b } ) / s _ { b } ^ { k } , } & { s _ { b } > 0 , } \\ { \sigma ( 0 ) , } & { s _ { b } = 0 , \ k = 0 , } \\ { 0 , } & { s _ { b } = 0 , \ k > 0 . } \end{array} } \right.
$$

Each $s _ { b } ,$ and hence each diagonal entry, is constant on a coordinate orbit. Therefore $D _ { k }$ is equivariant.

Proposition B.1. For $F ( x ) = R \sigma _ { B } ( A x )$

$$
\Pi _ { k } F = \Pi _ { k } \bigl [ R D _ { k } \bigl ( ( A x ) ^ { \odot k } \bigr ) \bigr ] , \qquad \| \Pi _ { k } F \| _ { L ^ { 2 } } ^ { 2 } = k ! \operatorname { t r } ( R D _ { k } C ^ { \circ k } D _ { k } R ^ { * } ) .
$$

Proof. Expand each coordinate using its own variance. If $s _ { b } = 0$ , then $A ^ { * } b = 0$ and that coordinate is the constant $\sigma ( 0 )$ , contributing only to degree zero. For positive variances, the identity from Section 1.3 gives $\Pi _ { k } \left. x , A ^ { * } b \right. ^ { k } = s _ { b } ^ { k } \mathrm { H e } _ { k } ( \left. x , A ^ { * } b \right. / s _ { b } )$ . By Lemma 1.2, the corresponding cross inner product is $k ! C _ { b b ^ { \prime } } ^ { k }$ , including degree zero with $C ^ { \circ 0 } = J$ . Summing gives both assertions. □

Diferent orbit contributions can cancel after the readout. Thus separate nonzero tests for the orbits do not replace this combined formula. In particular, several feature channels in a hidden space do not require a new theory, but do require retaining their possibly diferent variances and cross terms.

## Appendix C. Representation formulae for subset modules

The Specht modules $S ^ { \lambda } , \lambda \vdash n .$ , are the irreducible real representations of $S _ { n }$ . They are absolutely irreducible, so character inner products below are ordinary multiplicities. For the two-row formula assume $0 \leq r \leq n / 2$ and recall that complementation gives $V _ { n , r } \cong V _ { n , n - r }$ for the remaining nonzero cases.

C.1. The subset module and its Johnson scheme. We finish with formulae specific to subset modules. They support the multiplicity-free and computational statements used in the main text.

Proposition C.1. $V _ { n , r } \cong \operatorname { I n d } _ { S _ { r } \times S _ { n - r } } ^ { S _ { n } } \mathbf { 1 }$ , and for $r \leq n / 2$

$$
V _ { n , r } \cong \bigoplus _ { j = 0 } ^ { r } S ^ { ( n - j , j ) } .
$$

In particular $V _ { n , r }$ is multiplicity-free, and $\chi ^ { ( n - j , j ) } ( g ) = \left| X _ { n , j } ^ { g } \right| - \left| X _ { n , j - 1 } ^ { g } \right| f o r \ : 0 \le j \le n / 2 ,$ with $X _ { n , - 1 } = \emptyset$

Proof. The action of $S _ { n }$ on r-subsets is transitive, with stabiliser $S _ { r } \times S _ { n - r }$ so $V _ { n , r } \cong$ $\mathrm { I n d } _ { S _ { r } \times S _ { n - r } } ^ { S _ { n } } \mathbf { 1 }$ . The decomposition of this Young permutation module is multiplicity-free and standard. See [Jam78, Sag01].

Finally, since $\chi _ { V _ { n , j } } ( g ) = | X _ { n , j } ^ { g } |$ and $V _ { n , j } \cong \oplus _ { i = 0 } ^ { j } S ^ { ( n - i , i ) }$ , subtracting the decompositions for j and $j - 1$ gives

$$
\chi ^ { ( n - j , j ) } ( g ) = | X _ { n , j } ^ { g } | - | X _ { n , j - 1 } ^ { g } | .
$$

Thus $V _ { n , r }$ is transitive and multiplicity-free, and Section 3.3 applies without modification. Here the orbit algebra is the Bose–Mesner algebra of the Johnson scheme $J ( n , r )$ . Its orbit indicator matrices are indexed by the intersection number $| A \cap A ^ { \prime } |$ , and the primitive idempotents $E _ { i }$ have entries depending only on that number, given by the dual eigenvalues (Q-numbers) of the scheme in the standard normalisation [BCN89, Section 9.1]. The Hadamard coeficients of Corollary 3.11 are therefore obtained from the Johnson Krein parameters in that normalisation.

C.2. A character formula. Theorem 6.4 describes the orbit geometry. For dimension and multiplicity calculations, Pólya theory gives a complementary formula that avoids enumerating hypergraphs up to isomorphism.

Theorem C.2 (Molien–Pólya formula). For $g \in S _ { n }$ let

$$
f _ { r } ( g ) : = \left| X _ { n , r } ^ { g } \right| = [ z ^ { r } ] \prod _ { c \in \mathrm { C y c } ( g ) } ( 1 + z ^ { | c | } ) ,
$$

the product being over cycles of g on $[ n ]$ , and define $h _ { d } ( g )$ by

$$
\sum _ { d \ge 0 } h _ { d } ( g ) t ^ { d } = \exp \Bigl ( \sum _ { m \ge 1 } \frac { t ^ { m } } { m } f _ { r } ( g ^ { m } ) \Bigr ) , \qquad e q u i v a l e n t l y \qquad d h _ { d } ( g ) = \sum _ { m = 1 } ^ { d } f _ { r } ( g ^ { m } ) h _ { d - m } ( g ) .
$$

Then for every $\lambda \vdash n$ and $0 \leq s \leq n$

$$
[ \mathrm { S y m } ^ { d } V _ { n , r } : S ^ { \lambda } ] = \frac { 1 } { n ! } \sum _ { g \in S _ { n } } h _ { d } ( g ) \chi ^ { \lambda } ( g ) ,
$$

$$
\dim \mathrm { H o m } _ { S _ { n } } ( \mathrm { S y m } ^ { d } V _ { n , r } , V _ { n , s } ) = { \frac { 1 } { n ! } } \sum _ { g \in S _ { n } } h _ { d } ( g ) f _ { s } ( g ) .
$$

Both sums depend on g only through its cycle type.

Proof. A subset is fixed by g if and only if it is a union of cycles of $^ { g , }$ which gives the generating-function expression for $f _ { r }$

By Proposition 6.2, $\chi _ { \mathrm { S y m } ^ { d } V _ { n , r } } ( g )$ is the number of g-invariant multisets of size d on $X _ { n , r } .$ that is, of multiplicity functions constant on the cycles of g acting on $X _ { n , r }$ . Hence

$$
\sum _ { d \geq 0 } \chi _ { \mathrm { { S y m } } ^ { d } V _ { n , r } } ( g ) t ^ { d } = \prod _ { c \in \mathrm { C y c } ( g \cap X _ { n , r } ) } ( 1 - t ^ { | c | } ) ^ { - 1 } ,
$$

and taking logarithms gives $\begin{array} { r } { \sum _ { m \geq 1 } \frac { t ^ { m } } { m } \left| X _ { n , r } ^ { g ^ { m } } \right| } \end{array}$ , since a point of $X _ { n , r }$ is fixed by $g ^ { m }$ exactly when its cycle length divides m. $\operatorname { A s } \left| X _ { n , r } ^ { g ^ { m } } \right| = f _ { r } ( g ^ { m } )$ , this shows $\chi _ { \mathrm { S y m } ^ { d } V _ { n . r } } ( g ) = h _ { d } ( g )$ which is Proposition 2.8 for the permutation representation $V _ { n , r }$ . The second formula is the first with $\chi _ { V _ { n , s } } = f _ { s }$ □

The recursion for $h _ { d }$ gives a direct computational procedure.

C.3. Support and stability. This supplementary subsection records how the irreducible support behaves as n grows; it is not needed for the coordinatewise layer calculations above.

Theorem ${ \bf C . 3 . } \qquad \mathrm { ( a ) } \ I f S ^ { \lambda }$ occurs in $P _ { \mathcal { H } } ( n )$ then $\lambda _ { 1 } \geq n - v ( { \mathcal { H } } )$ . Hence every constituent of $\mathrm { S y m } ^ { d } V _ { n , r }$ satisfies $n - \lambda _ { 1 } \leq r d$

(b) Fix r, d and a partition $\nu \vdash s ,$ and put $\lambda [ n ] = ( n - s , \nu )$ . For n such that $\lambda [ n ]$ is a partition, $m _ { \nu } ( n ) : = [ \mathrm { S y m } ^ { d } V _ { n , r } : S ^ { \lambda [ n ] } ]$ vanishes for all n $i f s > r d ,$ , and is independent of n $f o r n \ge s + r d i f s \le r d .$

Proof. (a) Since Aut $( { \mathcal { H } } ) \times S _ { n - v } \supseteq S _ { n - v }$ we get $[ P _ { \mathcal { H } } ( n ) : S ^ { \lambda } ] \leq \dim ( S ^ { \lambda } ) ^ { S _ { n - v } }$ , and iterated branching gives $( S ^ { \lambda } ) ^ { { \dot { S } } _ { m } } \not = 0$ if and only if $\lambda _ { 1 } \geq m$

(b) The first claim is (a). For the second, take $n \geq r d$ so that the type set is fixed. Write ${ \mathrm { I n d } } _ { \mathrm { A u t } ( \mathcal { H } ) } ^ { \dot { S _ { v } } } { \bf 1 } = \oplus _ { \mu } c _ { \mu } S ^ { \mu }$ and induce in stages; the contribution of H to the multiplicity of $S ^ { \lambda [ n ] }$ is $\begin{array} { r } { \sum _ { \mu } c _ { \mu } c _ { \mu , ( n - v ) } ^ { \lambda [ n ] } } \end{array}$ and by Pieri’s rule the Littlewood–Richardson coeficient $c _ { \mu , ( n - v ) } ^ { \lambda [ n ] }$ is 1 exactly when $\lambda [ n ] / \mu$ is a horizontal strip, that is when $\lambda [ n ] _ { i } \geq \mu _ { i } \geq \lambda [ n ] _ { i + 1 }$ for all $i ,$ and 0 otherwise.

Only one of those inequalities involves n, namely $n - s \geq \mu _ { 1 }$ . Since $\mu _ { 1 } \leq v \leq r d$ , it holds for every µ once $n \geq s + r d .$ , so the multiplicity is constant from there on. □

In the language of representation stability, the sequence $n \mapsto \mathrm { S y m } ^ { d } V _ { n , r }$ is a finitely generated FI-module, which also implies eventual constancy [CEF15]. The argument above gives an explicit suficient range for each padded Specht multiplicity. The sharpness statement in Theorem 6.10 concerns the full space of maps into s-subset features; it does not assert that every individual Specht multiplicity first stabilises at this bound.

## References

[Aie26] Valeriano Aiello. Piecewise linear equivariant maps for compact groups. arXiv preprint arXiv:2608.21645, 2026.

[BBCV21] Michael M. Bronstein, Joan Bruna, Taco Cohen, and Petar Veličković. Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. arXiv preprint arXiv:2104.13478, 2021.

[BCN89] Andries E. Brouwer, Arjeh M. Cohen, and Arnold Neumaier. Distance-Regular Graphs, volume 18 of Ergebnisse der Mathematik und ihrer Grenzgebiete. Springer, 1989.

[CEF15] Thomas Church, Jordan S. Ellenberg, and Benson Farb. FI-modules and stability for representations of symmetric groups. Duke Mathematical Journal, 164(9):1833–1910, 2015.

[DFS16] Amit Daniely, Roy Frostig, and Yoram Singer. Toward deeper understanding of neural networks: The power of initialization and a dual view on expressivity. In Advances in Neural Information Processing Systems, volume 29, pages 2253–2261, 2016.

[GTW25] Joel Gibson, Daniel Tubbenhauer, and Geordie Williamson. Equivariant neural networks and piecewise linear representation theory. In Modern Algebra, Volume 1: Representation Theory, volume 829 of Contemporary Mathematics, pages 157–192. American Mathematical Society, 2025.

[Jam78] Gordon D. James. The Representation Theory of the Symmetric Groups, volume 682 of Lecture Notes in Mathematics. Springer, 1978.

[Jan97] Svante Janson. Gaussian Hilbert Spaces, volume 129 of Cambridge Tracts in Mathematics. Cambridge University Press, 1997.

[Jia21] Yu Jiang. On the symmetric and exterior powers of young permutation modules. Journal of Algebra, 568:660–704, 2021.

[LD24] Eitan Levin and Mateo Díaz. Any-dimensional equivariant neural networks. In Proceedings of the 27th International Conference on Artificial Intelligence and Statistics, volume 238 of PMLR, pages 2773–2781, 2024.

[MBHSL19] Haggai Maron, Heli Ben-Hamu, Nadav Shamir, and Yaron Lipman. Invariant and equivariant graph networks. In International Conference on Learning Representations, 2019.

[PDLS24] Marco Pacini, Xiaowen Dong, Bruno Lepri, and Gabriele Santin. A characterization theorem for equivariant networks with point-wise activations. arXiv preprint arXiv:2401.09235, 2024.

[PLK<sup>+</sup>23] Omri Puny, Derek Lim, Bobak T. Kiani, Haggai Maron, and Yaron Lipman. Equivariant polynomials for graph neural networks. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of PMLR, pages 28191–28222, 2023.

[PW16] Rowena Paget and Mark Wildon. Minimal and maximal constituents of twisted foulkes characters. Journal of the London Mathematical Society, 93(2):301–318, 2016.

[Rav20] Siamak Ravanbakhsh. Universal equivariant multilayer perceptrons. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 7996–8006, 2020.

[Sag01] Bruce E. Sagan. The Symmetric Group: Representations, Combinatorial Algorithms, and Symmetric Functions, volume 203 of Graduate Texts in Mathematics. Springer, 2 edition, 2001.

[Sha26] Mani Shayestehfar. Nonlinear maps between irreducible representations in graph neural networks. In preparation, 2026.

[Yar22] Dmitry Yarotsky. Universal approximations of invariant maps by neural networks. Constructive Approximation, 55:407–474, 2022.

School of Mathematics and Statistics<sub>,</sub> University of Sydney<sub>,</sub> NSW 2006<sub>,</sub> Australia Email address: mani.shayestehfar@sydney.edu.au Personal Webpage: https://maani.info