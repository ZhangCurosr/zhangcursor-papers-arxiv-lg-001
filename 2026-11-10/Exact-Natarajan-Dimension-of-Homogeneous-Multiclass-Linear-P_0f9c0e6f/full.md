# Exact Natarajan Dimension of Homogeneous Multiclass Linear Predictors

Jiahua Liu<sup>a</sup>, Benchong Li<sup>a,∗</sup>

<sup>a</sup>School of Mathematics and Statistics, Xidian University, Xi’an, China

## Abstract

The Natarajan dimension is a fundamental measure of the capacity of multiclass hypothesis classes and plays an important role in analyzing their generalization ability. In this paper, by establishing lower and upper bounds, we show that the exact Natarajan dimension of homogeneous multiclass linear predictors with pairwise distinct class weights is $n ( k - 1 ) - 1$ , where n is the dimension of the feature space and k is the number of classes, for $n , k \geq 3 , n , k \in \mathbb { N }$

Keywords: Natarajan dimension, Multiclass classification, Homogeneous linear predictors, Shattering

## 1. Introductiono

In this paper, we consider multiclass classification. In this setting, a predictor assigns each feature vector a label from a finite set. Understanding the capacity of multiclass hypothesis classes is a basic question in learning theory. For binary classes,G this capacity is described by the Vapnik–Chervonenkis dimension (Vapnik and Chervonenkis, 1971; Blumer et al., 1989). To<sup>.</sup> extend this notion to multiclass prediction, Natarajan (1989) in-c troduced the Natarajan dimension, defined via shattering with two distinct candidate labels fixed at each point. Its role in multiclass learnability and empirical risk minimization has been stud-v ied in Ben-David et al. (1995); Daniely et al. (2015).8

Daniely et al. (2012) analyzed the sample complexity of several multiclass hypothesis classes using lower bounds for the Natarajan dimension and upper bounds for the graph dimension. Daniely and Shalev-Shwartz (2014) established an upper bound. on the Natarajan dimension of their generalized linear hypothesis class in terms of the dimension of the class-specific feature<sub>6</sub> representation. Related bounds are also available for decision2 trees, random forests, and neural networks (Jin, 2023).<sup>:</sup>

A standard approach associates a linear score with each la  
bel and predicts a label of maximum score (Shalev-Shwartz andX   
Ben-David, 2014). Let n denote the dimension of the feature<sup>r</sup> space and k the number of classes. For $x \in \mathbb { R } ^ { n }$ and $c \in [ k ]$ where $[ k ] = \{ 1 , \dots , k \}$ , the block feature map $\Psi ( x , c ) \in \mathbb { R } ^ { n k }$ places x in the c-th n-dimensional block and zeros in all other blocks. Predictors in the homogeneous multi-vector construction use scores $W \cdot \Psi ( x , c ) = w _ { c } \cdot x .$ , with arbitrary $W =$ $( w _ { 1 } , \ldots , w _ { k } ) \ \in \ \mathbb { R } ^ { n k }$ , and select the smallest maximizing label. For the unrestricted multi-vector class, Shalev-Shwartz and Ben-David (2014) gave the lower bound $( n - 1 ) ( k - 1 )$ and the upper bound nk on the Natarajan dimension for $n , k \geq 2$ . In addition, Pan and Li (2026) established the improved upper bound

$n ( k - 1 )$ for the class considered in their paper when $n , k \geq 2 .$ and determined its exact dimension to be k when $n = 2$ and $k \geq 3 .$ . The bound $n ( k - 1 )$ equals the number of free parameters after translating all class weights so that $w _ { 1 } = 0$ . We investigate whether this bound can be attained when the class weights are pairwise distinct.

In this paper, we denote by $\mathcal { H } _ { \Psi }$ the class of homogeneous multiclass linear predictors with pairwise distinct class weights and show that its Natarajan dimension is $n ( k - 1 ) - 1$ when $n , k \geq 3 .$ . For the lower bound, we choose class weights as vertices of a polytope and use its exposed edges to select feature vectors with exactly two maximizing labels. We select $n ( k - 1 ) - 1$ vectors whose comparison rows are linearly independent, allowing the score diferences within their candidate pairs to be adjusted independently. Suficiently small weight perturbations then realize every candidate labeling while preserving both candidates’ strict score advantage over other labels. For the upper bound, we assume that $n ( k - 1 )$ feature vectors can be shattered and interpolate the weights realizing their candidate labelings over a cube. Induction on the dimension of the feature space shows that the resulting comparison map is nonzero on the cube boundary. The associated sphere self-map has degree zero because it omits a point, but degree one because it is homotopic to the identity. This contradiction establishes the upper bound.

Section 2 introduces the necessary preliminaries and states the main theorem. We then prove the lower and upper bounds in Sections 3 and 4, respectively, and conclude the paper in Section 5.

## 2. Preliminaries

We first recall the definitions of shattering and the Natarajan dimension and specify the class of homogeneous multiclass linear predictors studied in this paper. We then introduce the geometric and topological concepts and results needed for the proofs. The section concludes with the main theorem.

## 2.1. Natarajan dimension

We follow the notation and definitions in Shalev-Shwartz and Ben-David (2014).

Definition 1 (Shattering). Let $\mathcal { H } \subseteq [ k ] ^ { x }$ . A finite set $S \subseteq { \mathcal { X } }$ is shattered by H if there exist fixed functions $f _ { 0 } , f _ { 1 } : S \to [ k ]$ with $f _ { 0 } ( x ) \neq f _ { 1 } ( x )$ for every $x \in S$ , such that for every $B \subseteq S$ there is $h _ { B } \in \mathcal { H }$ satisfying, for every $x \in S$

$$
h _ { B } ( x ) = \left\{ { \begin{array} { l l } { f _ { 0 } ( x ) , } & { x \in B , } \\ { f _ { 1 } ( x ) , } & { x \in S \setminus B . } \end{array} } \right.
$$

Using this notion of shattering, we define the Natarajan dimension as follows.

Definition 2 (Natarajan dimension). The Natarajan dimension of H, denoted by Ndim(H), is the maximal size of a set $S \subseteq { \mathcal { X } }$ shattered by H. If arbitrarily large finite sets are shattered, the dimension is infinite.

For $\begin{array} { r l r } { S } & { { } = } & { \left\{ x _ { 1 } , \ldots , x _ { m } \right\} } \end{array}$ , order each candidate pair $\{ f _ { 0 } ( x _ { i } ) , f _ { 1 } ( x _ { i } ) \}$ as $a _ { i } ~ < ~ b _ { i }$ Shattering requires every labeling in $\{ a _ { 1 } , b _ { 1 } \} \times \cdots \times \{ a _ { m } , b _ { m } \}$ to be realizable.

## 2.2. Homogeneous multiclass linear predictors

We now consider linear scores defined by the standard Euclidean inner product and use the associated Euclidean norm throughout. The score-maximization representation is standard (Daniely et al., 2012; Shalev-Shwartz and Ben-David, 2014).

Definition 3 (Homogeneous multiclass linear predictors). For $W = ( w _ { 1 } , \dots , w _ { k } ) \in \mathbb { R } ^ { n k }$ , define

$$
h _ { W } ( x ) = \underset { c \in [ k ] } { \arg \operatorname* { m a x } } w _ { c } \cdot x , \qquad x \in \mathbb { R } ^ { n } .\tag{1}
$$

If several labels maximize $w _ { c } \cdot x ,$ we choose the smallest one. The hypothesis class $\mathcal { H } _ { \Psi }$ consists of all such $h _ { W }$ with pairwise distinct class weights, that is, $w _ { c } \neq w _ { d }$ for all $c , d \in [ k ]$ with $c \neq d .$

Distinct weights can give equal scores at a particular feature vector. A common translation of all weights preserves score diferences, ties, and pairwise distinctness. We may therefore use efective weights

$$
v _ { 1 } = 0 , \qquad v _ { c } = w _ { c } - w _ { 1 } , \qquad c \in \{ 2 , \ldots , k \} .\tag{2}
$$

For each nonnegative integer $r ,$ define the efective parameter space corresponding to the feature space $\mathbb { R } ^ { r }$ by

$$
V _ { r } = \mathbb { R } ^ { r q } , \qquad \mathrm { d i m } V _ { r } = r q ,\tag{3}
$$

where $q = k - 1$ . For $v = ( v _ { 2 } , \ldots , v _ { k } ) \in V _ { r }$ , each $v _ { c } \in \mathbb { R } ^ { r }$ $c \in \{ 2 , \ldots , k \}$ . Setting $v _ { 1 } = 0$ gives the full tuple $( v _ { 1 } , \ldots , v _ { k } )$ of class weights. The admissible efective parameters form

$$
\mathcal { A } _ { r } = \{ v \in V _ { r } : v _ { c } \neq v _ { d } \mathrm { f o r } \mathrm { a l l } c , d \in [ k ] , c \neq d \} , \qquad v _ { 1 } = 0 .
$$

## 2.3. Faces ofconvex polytopes

We introduce some basic concepts from polytope theory that will be used below. For further details, we refer the reader to Ziegler (1995).

A convex polytope is the convex hull of finitely many points. Specifically, the convex hull of $u _ { 1 } , \ldots , u _ { s }$ consists of all convex combinations $\textstyle \sum _ { i = 1 } ^ { s } \lambda _ { i } u _ { i }$ , where $\lambda _ { i } \geq 0$ and $\textstyle \sum _ { i = 1 } ^ { s } \lambda _ { i } = 1$ . For Pa nonempty convex polytope $P \subseteq \mathbb { R } ^ { n }$ , let $A = \operatorname { a f f } ( P )$ denote the afine hull of $P ,$ namely the smallest afine subspace containing $P .$ If $P = { \mathrm { c o n v } } \{ u _ { 1 } , \dots , u _ { s } \}$ , then A consists of all afine combinations $\textstyle \sum _ { i = 1 } ^ { s } \lambda _ { i } u _ { i }$ , where $\lambda _ { i } \in$ R for $i \in [ s ]$ and $\textstyle \sum _ { i = 1 } ^ { s } \lambda _ { i } = 1$ P; the coeficients may be negative. Define

$$
L = \{ u - v : u , v \in A \} .
$$

The set L is a linear subspace of $\mathbb { R } ^ { n }$ , called the direction space of A. For any fixed $a _ { 0 } \in A$

$$
A = \{ a _ { 0 } + \ell : \ell \in L \} ,
$$

so $A$ is a translate of the linear space L. The dimension of $P$ is defined as the dimension of its afine hull; thus,

$$
\dim P = \dim L .
$$

Interiors and boundaries of polytopes are understood relative to their afine hulls.

For $a \in L \setminus \{ 0 \}$ , let $\alpha = \operatorname* { m a x } _ { x \in P } a \cdot x .$ . The hyperplane $\{ x \in A : a \cdot x = \alpha \}$ is a supporting hyperplane of $P ,$ and its intersection with $P .$

$$
F = \{ x \in P : a \cdot x = \alpha \} ,
$$

is the face exposed by $^ { a , }$ with a as an outward normal. Every nonempty proper face of a convex polytope admits such a representation. The sets $\mathcal { D }$ and $P$ are also regarded as faces. When d = dim $P \geq 1$ , faces of dimensions 0, 1, and $d - 1$ are called vertices, edges, and facets, respectively. In particular, the facets of a three-dimensional polytope are two-dimensional faces.

Every edge of a three-dimensional convex polytope is contained in exactly two facets and equals their intersection. If these facets are exposed by outward normals $a _ { 1 } , a _ { 2 }$ , then $a _ { 1 } + a _ { 2 }$ exposes their common edge: equality in the sum of the two supporting inequalities holds exactly when equality holds in both. Moreover, the vertex–edge graph of a convex polytope is connected. Any two facets of a three-dimensional convex polytope can also be joined by a sequence of facets in which consecutive facets share an edge.

For the cube $Q = [ - 1 , 1 ] ^ { m }$ , every nonempty face is obtained by fixing some coordinates to +1 or −1. Specifically, for $I \subseteq$ [m] and $\varepsilon _ { i } \in \{ - 1 , 1 \}$ , the set

$$
\left\{ z \in Q : z _ { i } = \varepsilon _ { i } { \mathrm { f o r } } { \mathrm { a l l } } i \in I \right\}
$$

is a face of dimension $m - | I |$ . For $z \in \partial Q$ , the smallest face containing z fixes precisely the coordinates satisfying $| z _ { i } | = 1$ all remaining coordinates vary freely in (−1, 1).

## 2.4. Homotopy, contractibility, and the degree ofsphere maps

The upper-bound proof compares sphere maps through homotopies and derives a contradiction from their degrees. We recall the required concepts and properties (Hatcher, 2002).

Let $X , Y$ be topological spaces. A homeomorphism is a continuous bijection with a continuous inverse. Two continuous maps $f , g : X \to Y$ are homotopic, written $f \simeq g$ , if there exists a continuous map $H : X \times [ 0 , 1 ] \to Y$ such that, for every $x \in X$

$$
H ( x , 0 ) = f ( x ) , \qquad H ( x , 1 ) = g ( x ) .
$$

The map H is called a homotopy from $f \ t o \ g$

A nonempty space X is contractible if its identity map is homotopic to a constant map. Both the constant map and the homotopy take values in $X$ . For example, a nonempty convex set is contractible: fixing $x _ { 0 } \in X$ , the map $H ( x , t ) = ( 1 -$ $t ) x + t x _ { 0 }$ gives a homotopy from the identity to the constant map $x \mapsto x _ { 0 }$

For an integer $d \geq 1$ , write $S ^ { d } = \{ x \in \mathbb { R } ^ { d + 1 } : \| x \| =$ 1}. Its d-th singular homology group with integer coeficients satisfies $H _ { d } ( S ^ { d } ; \mathbb { Z } ) \cong \mathbb { Z }$ . A continuous map ${ \bar { \boldsymbol { f } } } : S ^ { d } \to S ^ { d }$ induces a homomorphism on this group, necessarily given by multiplication by an integer. This integer is the degree of $f ,$ denoted by deg $f .$ . We use the following properties:

1. If $f \simeq g$ , then deg $f = \deg g .$

2. The identity satisfies $\begin{array} { r } { \mathrm { d e g } ( \mathrm { i d } _ { S ^ { d } } ) = 1 } \end{array}$

3. If $f ( S ^ { d } ) \neq S ^ { d }$ , then deg $f = 0$ . In particular, a constant map has degree zero.

The third property can be understood through contractibility. If f omits a point $p \in S ^ { d }$ , its image lies in $S ^ { d } \backslash \{ p \}$ . This space is homeomorphic to $\mathbb { R } ^ { d }$ and is therefore contractible, so $f$ is homotopic to a constant map.

In applications, sphere-valued homotopies can be constructed by normalizing nonzero vectors. If $B : X \times [ 0 , 1 ] $ $\mathbb { R } ^ { d + 1 } \backslash \{ \bar { 0 } \}$ is continuous, then

$$
\widehat { B } ( x , t ) = \frac { B ( x , t ) } { \| B ( x , t ) \| }
$$

is a sphere-valued homotopy. Thus, when using linear interpolation to construct such a homotopy, the interpolating vectors must remain nonzero throughout.

Finally, for $m \geq 2$ , the boundary of the cube $Q = [ - 1 , 1 ] ^ { m }$ is homeomorphic to $S ^ { m - 1 }$ through the radial map

$$
\mathcal { R } : \partial Q  S ^ { m - 1 } , \qquad \mathcal { R } ( z ) = \frac { z } { \| z \| } ,
$$

whose inverse is

$$
\mathcal { R } ^ { - 1 } ( y ) = \frac { y } { \operatorname* { m a x } _ { 1 \leq j \leq m } | y _ { j } | } .
$$

Consequently, if $F : Q  \mathbb { R } ^ { m }$ is continuous and $F ( z ) \neq 0$ for every $z \in \partial Q$ , we can define

$$
{ \widehat { F } } ( z ) = { \frac { F ( z ) } { \| F ( z ) \| } } , \qquad z \in \partial Q .
$$

Composing $\widehat F$ with $\mathcal { R } ^ { - 1 } : S ^ { m - 1 } \to \partial Q$ gives the sphere selfmap $\mathbf { \bar { \boldsymbol { f } } } : \bar { S ^ { m - 1 } }  S ^ { m - 1 }$ , defined by $f ( y ) ^ { - } = \widehat { F } ( \mathcal { R } ^ { - \bar { 1 } } ( y ) )$ ). We can therefore consider the degree of $f .$

## 2.5. Main result

We now state the main result for the hypothesis class $\mathcal { H } _ { \Psi }$

Theorem 1 (Exact dimension). For the class H<sub>Ψ</sub> ofpredictors with pairwise distinct class weights in Definition 3, $; i f n \ge 3$ and $k \geq 3 ,$ , then

$$
\mathrm { N d i m } ( \mathcal { H } _ { \Psi } ) = n ( k - 1 ) - 1 .\tag{4}
$$

Theorem 1 follows from Propositions 1 and 2: Proposition 1 in Section 3 constructs a shattered set of this size, and Proposition 2 in Section 4 proves that no larger set is shattered.

## 3. A lower bound

To establish the lower bound, we construct a set of $n ( k -$ 1) − 1 feature vectors together with two distinct candidate labels for each vector. We then show that every choice of one candidate label for each vector can be realized by a suitable choice of pairwise distinct class weights.

Proposition 1 (Lower bound). For $n , k \geq 3 ,$ , we have

$$
\mathrm { N d i m } ( \mathcal { H } _ { \Psi } ) \geq n ( k - 1 ) - 1 .\tag{5}
$$

The idea is to control the score diference within each candidate pair while keeping both candidate scores above those of all other labels. The proof proceeds in three steps. In Step 1, we choose class weights as vertices of a polytope and use its exposed edges to select feature vectors and fix their candidate label pairs. In Step 2, we form the comparison matrix $\tau$ , show that its rank is $n ( k - 1 ) - 1$ , and select $n ( k - 1 ) - 1$ independent rows to obtain a submatrix $T$ of full row rank. The corresponding feature vectors form the set to be shattered. In Step 3, we use a right inverse of $T$ to construct weight perturbations realizing every candidate labeling. A single suficiently small perturbation scale preserves the strict score advantage of both candidate labels over the other labels and keeps the class weights pairwise distinct.

Proof. Step 1: exposed edges and candidate label pairs. We first choose class weight vectors $u _ { i } \in \mathbb { R } ^ { n } , i \in [ k ]$ . For $k =$ $^ { 3 , }$ take $u _ { 1 } = 0 , u _ { 2 } = e _ { 1 }$ , and $u _ { 3 } ~ = ~ e _ { 2 }$ , where $e _ { 1 } , e _ { 2 }$ are the first two standard basis vectors of $\mathbb { R } ^ { n }$ . Their convex hull is a nondegenerate triangle. For $k \geq 4$ , choose distinct real numbers $t _ { 1 } , \ldots , t _ { k }$ and set

$$
u _ { i } = ( t _ { i } , t _ { i } ^ { 2 } , t _ { i } ^ { 3 } , 0 , \ldots , 0 ) , \qquad P = \mathrm { c o n v } \{ u _ { 1 } , \ldots , u _ { k } \} .\tag{6}
$$

Each $u _ { i }$ is a vertex: the functional with coeficient vector $( 2 t _ { i } , - 1 , 0 , \ldots , 0 )$ takes value $t _ { i } ^ { 2 } - ( t _ { j } - t _ { i } ) ^ { 2 }$ at $u _ { j } ,$ uniquely maximizing at $j = i ,$ Any four vertices are afinely independent, since the matrix with rows $( 1 , t _ { i } , t _ { i } ^ { 2 } , t _ { i } ^ { 3 } )$ has a nonzero Vandermonde determinant. Consequently $P$ is three-dimensional and each of its facets is a triangle. When $n > 3$ , these dimensions refer to its afine hull.

Let E be the edge set of $P ,$ or the three edges of the triangle when $k = 3 ,$ , identifying each edge $[ u _ { i } , u _ { j } ]$ with its endpoint index pair {i, j}. For each $\{ i , j \} \in E$ , define

$$
\begin{array} { r l } & { H _ { i j } = \{ x \in \mathbb { R } ^ { n } : ( u _ { j } - u _ { i } ) \cdot x = 0 \} , } \\ & { \Omega _ { i j } = \{ x \in H _ { i j } : ( u _ { i } - u _ { d } ) \cdot x > 0 \mathrm { ~ f o r ~ a l l ~ } d \in [ k ] \setminus \{ i , j \} \} . } \end{array}\tag{7}
$$

Fix an edge $\{ i , j \} \in E .$ . By the exposed-face property recalled in Section 2.3, there exists $x _ { 0 } \in \mathbb { R } ^ { n }$ such that the linear functional $u \mapsto u \cdot x _ { 0 }$ attains its maximum over conv $\{ u _ { 1 } , \ldots , u _ { k } \}$ exactly on $[ u _ { i } , u _ { j } ]$ . Thus the two endpoint weights have the same score at $x _ { 0 } ,$ , strictly greater than that of every other class. Hence $x _ { 0 } \in \Omega _ { i j } , \mathrm { s o } \Omega _ { i j }$ is nonempty. By definition, for every $\boldsymbol { x } \in \Omega _ { i j }$ and every $d \in [ \bar { k } ] \setminus \{ i , j \}$ ,

$$
u _ { i } \cdot x = u _ { j } \cdot x > u _ { d } \cdot x .\tag{8}
$$

Since $u _ { i } \neq u _ { j } ,$ , the space $H _ { i j }$ has dimension $n - 1$ . The set $\Omega _ { i j }$ is defined within $H _ { i j }$ by finitely many strict linear inequalities, so it is open in $H _ { i j }$ . Therefore, there exists $\eta > 0$ such that

$$
\left\{ x \in H _ { i j } : \| x - x _ { 0 } \| < \eta \right\} \subseteq \Omega _ { i j } .
$$

Diferences of points in this ball span $H _ { i j }$ . These diferences belong to span $\left( \Omega _ { i j } \right)$ , and hence span $\left( \Omega _ { i j } \right) = H _ { i j }$

For each edge $\{ i , j \} \in E$ , choose n − 1 linearly independent feature vectors in $\Omega _ { i j }$ , and assign the candidate labels i and j to each of them. All selected feature vectors are distinct. Within each edge, this follows from linear independence. Vectors selected for diferent edges cannot coincide, since the pair of maximizing class labels uniquely determines the edge.

Step 2: the rank of the edge comparisons. We next determine how many of these candidate comparisons can be varied independently. We form the comparison matrix T from all feature vectors chosen in Step 1, using the candidate pair assigned to each vector. The matrix has $| E | ( n - 1 )$ rows. For each edge $\{ i , j \} \in E$ with $i < j$ , and each of its $n - 1$ selected feature vectors $x ,$ the corresponding row is indexed by $( i j , x )$ and has k blocks of size $1 \times n$ . For $c \in [ k ]$ , its c-th block is defined by

$$
\mathcal { T } _ { ( i j , x ) , c } = \left\{ \begin{array} { l l } { - x ^ { \top } , } & { c = i , } \\ { x ^ { \top } , } & { c = j , } \\ { 0 _ { 1 \times n } , } & { c \notin \{ i , j \} . } \end{array} \right.
$$

Thus, for $Z = ( z _ { 1 } , \dots , z _ { k } ) \in \mathbb { R } ^ { n k }$

$$
( T Z ) _ { i j , x } = ( z _ { j } - z _ { i } ) \cdot x .
$$

We claim that

$$
\ker \mathcal { T } = \{ ( b + s u _ { 1 } , \ldots , b + s u _ { k } ) : b \in \mathbb { R } ^ { n } , \ s \in \mathbb { R } \} .\tag{9}
$$

Fix $Z \in$ ker T . For each edge $\{ i , j \} \in E ,$ the feature vectors chosen from $\Omega _ { i j }$ span $H _ { i j }$ . Hence, for every edge $\{ i , j \} \in E$ with $i < j ,$ , there exists $\lambda _ { i j } \in$ R such that

$$
z _ { j } - z _ { i } = \lambda _ { i j } ( u _ { j } - u _ { i } ) .\tag{10}
$$

On a triangular face with vertices $u _ { i } , u _ { j } , u _ { \ell }$ , where $i < j < \ell .$ put $a = u _ { j } - u _ { i }$ and $b ^ { \prime } = u _ { \ell } - u _ { i }$ . The identity $( z _ { j } - z _ { i } ) + ( z _ { \ell } -$ $z _ { j } ) - ( z _ { \ell } - z _ { i } ) = 0$ yields

$$
( \lambda _ { i j } - \lambda _ { j \ell } ) a + ( \lambda _ { j \ell } - \lambda _ { i \ell } ) b ^ { \prime } = 0 .
$$

Since $a , b ^ { \prime }$ are independent, the three edge multipliers agree. For $k = 3$ , this covers every edge. For $k \geq 4 .$ , triangular facets are connected through shared edges: radial projection from an interior point identifies the boundary with a triangulated sphere, and paths between facet interiors can avoid vertices and cross successive edges. Equality therefore propagates to every edge. Let s be the common multiplier. The vertex-edge graph is connected as well, so

$$
z _ { i } - z _ { 1 } = s ( u _ { i } - u _ { 1 } ) , \qquad z _ { i } = b + s u _ { i } , \quad b = z _ { 1 } - s u _ { 1 } .
$$

Conversely, every tuple of this form lies in ker T, because $( u _ { j } -$ $u _ { i } ) \cdot x = 0$ for $x \in H _ { i j }$ . The parameterization by $( b , s )$ is injective: if every $b + s u _ { i } = 0$ , subtracting equations for two distinct vertices gives $s = 0 ,$ and then $b = 0$ . This proves (9) and gives

$$
\dim \ker { \mathcal T } = n + 1 , \qquad \mathrm { r a n k } { \mathcal T } = n k - ( n + 1 ) = n ( k - 1 ) - 1 .\tag{11}
$$

Since rank $\mathcal { T } = n ( k - 1 ) - 1$ , we can select $m : = n ( k -$ $1 ) - 1$ linearly independent rows of T, obtaining a full-row-rank matrix $\boldsymbol { T } \in \mathbb { R } ^ { m \times n k }$ . Denote the corresponding distinct inputs by $x _ { 1 } , \ldots , x _ { m }$ and their fixed candidate pairs by $a _ { i } < b _ { i }$ . Thus, for any block vector $Z = ( z _ { 1 } , \dots , z _ { k } )$ , with $z _ { c } \in \mathbb { R } ^ { n }$ , we have

$$
( T Z ) _ { i } = ( z _ { b _ { i } } - z _ { a _ { i } } ) \cdot x _ { i } , \qquad i \in [ m ] .
$$

Step 3: a uniformperturbationfor all candidate labelings. It remains to realize every choice between the selected candidates without changing their dominance over the other labels. Take the right inverse of T

$$
R = T ^ { \top } ( T T ^ { \top } ) ^ { - 1 } , \qquad T R = I _ { m } .\tag{12}
$$

For each $\sigma \in \{ - 1 , + 1 \} ^ { m }$ , write $R \sigma = \left( \delta _ { 1 } , \ldots , \delta _ { k } \right)$ , where $\delta _ { c } = ( R \sigma ) _ { c } \in \mathbb { R } ^ { n }$ is the c-th vector block for each $c \in [ k ]$ These blocks depend on σ. Define the following uniform bounds over all choices:

$$
\begin{array} { r l } & { D = \underset { \sigma \in \{ - 1 , + 1 \} ^ { m } , \ c \in [ k ] } { \operatorname* { m a x } } \Vert ( R \sigma ) _ { c } \Vert > 0 , } \\ & { X = \underset { 1 \leq i \leq m } { \operatorname* { m a x } } \Vert x _ { i } \Vert > 0 . } \end{array}\tag{13}
$$

Define the base margin and weight separation explicitly by

$$
\begin{array} { r l r } & { \displaystyle { g = \operatorname* { m i n } _ { 1 \le i \le m , c \in \{ a _ { i } , b _ { i } \} } ( u _ { c } - u _ { d } ) \cdot x _ { i } > 0 , } } & \\ & { \displaystyle { d \in [ k ] \backslash \{ a _ { i } , b _ { i } \} } } & \\ & { \displaystyle { \rho = \operatorname* { m i n } _ { 1 \le c < d \le k } \| u _ { c } - u _ { d } \| > 0 . } } & \end{array}
$$

Both minima are taken over finite nonempty sets. Their positivity follows from the strict score inequalities above and the pairwise distinctness of $u _ { 1 } , \ldots , u _ { k }$ . Choose one $\varepsilon > 0$ , independently of σ, such that

$$
\begin{array} { r } { 2 \varepsilon D X < g , \qquad 2 \varepsilon D < \rho . } \end{array}\tag{14}
$$

With this ε fixed, take an arbitrary $\sigma \in \{ - 1 , + 1 \} ^ { m }$ and set $w _ { c } = u _ { c } + \varepsilon \delta _ { c }$ for every $c \in [ k ]$ , using the blocks of Rσ. For

every $i \in [ m ]$ , the identity $u _ { a _ { i } } \cdot x _ { i } = u _ { b _ { i } } \cdot x _ { i }$ , together with $T R \sigma = \sigma$ , gives

$$
\left( w _ { b _ { i } } - w _ { a _ { i } } \right) \cdot x _ { i } = \varepsilon \sigma _ { i } .\tag{15}
$$

For every $i \in [ m ]$ , every $c \in \{ a _ { i } , b _ { i } \}$ , and every $d \in \ [ k ] \ \backslash$ $\{ a _ { i } , b _ { i } \}$ , Cauchy–Schwarz yields

$$
\begin{array} { r } { ( w _ { c } - w _ { d } ) \cdot x _ { i } \geq g - \varepsilon \| \delta _ { c } - \delta _ { d } \| \| x _ { i } \| \geq g - 2 \varepsilon D X > 0 . } \end{array}\tag{16}
$$

For every $1 \leq c < d \leq k$ , the weights satisfy

$$
\| w _ { c } - w _ { d } \| \ge \rho - 2 \varepsilon D > 0 .\tag{17}
$$

For $\sigma _ { i } = - 1$ the unique winner is ${ { a } _ { i } } ;$ for $\sigma _ { i } = + 1$ it is $b _ { i } .$ . The inputs and candidate pairs were fixed before σ was chosen, and the same ε works for all $2 ^ { m }$ choices. This proves (5) with strict winning margins. □

Example 1 (An explicit five-point construction in $\mathbb { R } ^ { 3 } )$ . Take $u _ { 1 } = 0 , u _ { 2 } = e _ { 1 }$ , and $u _ { 3 } = e _ { 2 }$ . The exposed-edge regions are $\Omega _ { 1 2 } = \{ ( 0 , y , z ) : y < 0 \} , \Omega _ { 1 3 } = \{ ( x , 0 , z ) : x < 0 \}$ , and $\Omega _ { 2 3 } = \{ ( t , t , z ) : t > 0 \}$ . The following five distinct inputs lie in these regions:

<table><tr><td>2</td><td> $x _ { i }$   $\{ a _ { i } , b _ { i } \}$ </td></tr><tr><td>1 (0,−1,0) 2 (0, −1, 1) 3 (−1,0,0)</td><td>{1,2} {1,2} {1, 3} (−1,0,1) {1,3} {2,3}</td></tr></table>

For any $\sigma \in \{ - 1 , + 1 \} ^ { 5 }$ , set $\varepsilon = 1 / 2 0$ and

$$
\begin{array} { l } { { w _ { 1 } = 0 , } } \\ { { \begin{array} { l } { { \left( \sigma _ { 1 } - \sigma _ { 3 } - \sigma _ { 5 } \right) , } } \\ { { w _ { 2 } = e _ { 1 } + \varepsilon \left( \begin{array} { c } { { \sigma _ { 1 } - \sigma _ { 3 } - \sigma _ { 5 } } } \\ { { - \sigma _ { 1 } } } \\ { { \sigma _ { 2 } - \sigma _ { 1 } } } \end{array} \right) , } } \end{array} } } \\ { { w _ { 3 } = e _ { 2 } + \varepsilon \left( \begin{array} { c } { { - \sigma _ { 3 } } } \\ { { 0 } } \\ { { \sigma _ { 4 } - \sigma _ { 3 } } } \end{array} \right) . } } \end{array}
$$

Direct substitution gives $( w _ { b _ { i } } - w _ { a _ { i } } ) { \cdot } x _ { i } = \varepsilon \sigma _ { i }$ for all five points. The perturbation blocks have norms at most $D = { \sqrt { 1 4 } } ;$ ; the base quantities satisfy $g = \rho = 1$ and $X = { \sqrt { 2 } } .$ Hence $2 \varepsilon D X =$ $\sqrt { 2 8 } / 1 0 < 1$ and $2 \varepsilon D = \sqrt { 1 4 } / 1 0 < 1$ . Equations $( 1 6 ) – ( 1 7 )$ therefore ensure unique winners and pairwise distinct weights for all 32 candidate labelings. Thus the five-point set is shattered by ${ \mathcal { H } } _ { \Psi } .$ , giving $\mathrm { N d i m } ( \mathcal { H } _ { \Psi } ) \ge 5$

## 4. An upper bound

In this section, we prove the upper bound. We first establish comparison independence and strict avoidance of the candidate labels, then use induction and cube interpolation to derive a contradiction by computing the degree of the associated sphere map in two ways. Fix $k \geq 3$ . We use $V _ { r }$ from (3).

For the induction below, an indexed candidate system $( x _ { i } , a _ { i } , b _ { i } ) _ { i = 1 } ^ { m }$ records feature vectors $x _ { i }$ and fixed candidate labels $a _ { i } < b _ { i }$ in $[ k ]$ ; the feature vectors may coincide. Realizability of every candidate labeling by $\mathcal { H } _ { \Psi }$ means that, for every tuple $\left( y _ { 1 } , \ldots , y _ { m } \right)$ with $y _ { i } ~ \in ~ \{ a _ { i } , b _ { i } \}$ for all $i \in [ m ]$ , there exists $h \in \mathcal { H } _ { \Psi }$ satisfying $h ( x _ { i } ) = y _ { i }$ for all $i \in [ m ]$ . After projection, we retain each index and its candidate pair and apply the same realization condition to the projected feature vectors. Keeping the indices allows the induction to proceed without a general-position assumption.

## 4.1. Comparison independence and strict avoidance

The following lemma adapts the linear-dependence argument for homogeneous halfspaces in the proof of Theorem 9.2 of Shalev-Shwartz and Ben-David (2014) to the efective parameter space $V _ { r }$ , with the smallest-label tie rule made explicit.

Lemma 1. Let $( x _ { i } , a _ { i } , b _ { i } ) _ { i = 1 } ^ { m }$ be an indexed candidate system in $\mathbb { R } ^ { r }$ , with $a _ { i } < b _ { i } . \ F o r \ v = ( v _ { 2 } , \ldots , v _ { k } ) \in V _ { r } $ , let $h _ { v }$ denote the prediction rule associated with thefull tuple $( 0 , v _ { 2 } , \ldots , v _ { k } )$ Suppose that, for every $( y _ { i } ) _ { i = 1 } ^ { m } \in \{ a _ { 1 } , b _ { 1 } \} \times \cdots \times \{ a _ { m } , b _ { m } \}$ there exists $v \in V _ { r }$ such that $h _ { v } ( x _ { i } ) = y _ { i }$ for all $i \in [ m ] ,$ , under the smallest-label tie rule. Then the functionals

$$
\rho _ { i } ( v ) = ( v _ { b _ { i } } - v _ { a _ { i } } ) \cdot x _ { i } , \qquad 1 \leq i \leq m ,\tag{18}
$$

are linearly independent on $V _ { r }$ . Consequently $m \leq q r . \ I f m =$ $q r _ { ; }$ , the map

$$
T : V _ { r } \longrightarrow \mathbb { R } ^ { m } , \qquad T ( v ) = \bigl ( \rho _ { 1 } ( v ) , \ldots , \rho _ { m } ( v ) \bigr )\tag{19}
$$

is a linear isomorphism. For each fixed candidate pair, the $i n \textless$ dexed inputs carrying that pair are linearly independent.

Proof. Choosing $a _ { i }$ requires $\rho _ { i } ( v ) \leq 0 ,$ , whereas choosing $b _ { i }$ requires $\rho _ { i } ( v ) > 0$ , since a tie would favor $a _ { i } < b _ { i }$ . Suppose $\textstyle \sum _ { i } t _ { i } \rho _ { i } \ = \ 0$ is a nontrivial relation. Reverse its sign if nec-Pessary so that some $t _ { i } > 0$ . Choose $b _ { i }$ when $t _ { i } > 0 .$ , choose $a _ { i }$ when $t _ { i } ~ < ~ 0$ , and choose either candidate when $t _ { i } ~ = ~ 0$ At a realizing parameter, every $t _ { i } \rho _ { i } ( v )$ is nonnegative and at least one is strictly positive, contradicting the relation. A dependence among inputs with a common candidate pair would induce the same dependence among their functionals. The dimension bound and the isomorphism statement follow. In particular, under the hypotheses of the lemma, every $x _ { i }$ is nonzero. □

It shows that realizing every candidate labeling forces the comparison functionals to be linearly independent, thereby giving the upper bound n $\left( k - 1 \right)$ on the Natarajan dimension, which is consistent with the bound established in Pan and Li (2026).

The following lemma, together with interpolation, will be used in Step 3 of the proof of Proposition 2 below.

Lemma 2 (Strict avoidance). Let $k \geq 3 ,$ and let $( x _ { i } , a _ { i } , b _ { i } ) _ { i = 1 } ^ { m }$ be an indexed candidate system in $\mathbb { R } ^ { r }$ , with m $\geq 1$ . Suppose that,for every $( y _ { 1 } , \dots , y _ { m } ) \in \{ a _ { 1 } , b _ { 1 } \} \times \cdots \times \{ a _ { m } , b _ { m } \}$ , there exists $v \in A _ { r } ,$ corresponding to pairwise distinct weights, such that $h _ { v } ( x _ { i } ) = y _ { i }$ for all $i \in [ m ]$ , under the smallest-label tie rule. Then there exists a single u $\iota \in V _ { r } \backslash \{ 0 \}$ such that, for every $i \in [ m ]$ , the maximum class score at x under u is attained by exactly one label, which lies outside $\{ a _ { i } , b _ { i } \}$

Proof. Case $k = 3$ . Let $A , B , C$ denote the inputs with candidate pairs $\{ 1 , 2 \} , \{ 1 , 3 \}$ , and $\{ 2 , 3 \}$ , respectively. Choose a weight tuple $W = ( w _ { 1 } , \dots , w _ { k } )$ realizing the labeling

$$
A \mapsto 2 , \qquad B \mapsto 1 , \qquad C \mapsto 3 .
$$

By Lemma 1, the inputs in B are linearly independent. Hence there is $d \in \mathbb { R } ^ { r }$ such that $d \cdot x = 1$ for every $x \in B ;$ take $d = 0$ if B is empty. Replace w<sub>1</sub> by $w _ { 1 } + \varepsilon d .$ . On $B ,$ every margin of label 1 increases by $\varepsilon ,$ making label 1 a unique winner. On A and C, the desired winner already strictly beats label 1. Choose one positive ε small enough to preserve these strict inequalities simultaneously at all inputs in $A \cup C$ . Comparisons between labels 2 and 3 are unchanged.

Subtract the new first weight from all weights. Labels 1 on B and 3 on $C$ are now unique winners. On A, label 2 strictly beats label 1 but may tie with label 3 at a common positive score $h .$ Choose one $0 < \eta < 1 / 2$ , small enough to preserve all existing strict winning comparisons at all inputs, and replace

$$
w _ { 2 } \mapsto ( 1 - \eta ) w _ { 2 } , \qquad w _ { 3 } \mapsto ( 1 - 2 \eta ) w _ { 3 } .
$$

All existing strict margins remain positive, and a tie on A becomes a margin $\eta h > 0$ in favor of label 2. Thus, at every feature vector, the prescribed label uniquely maximizes the class score. Apply the permutation $1 \mapsto 2 \mapsto 3 \mapsto 1$ to the weights. The winners on $A , B , C$ become $3 , 2 , 1 .$ , respectively, and lie outside their candidate pairs.

Case $k \geq 4 .$ . Let π fix label 1 and cycle the others:

$$
2 \mapsto 3 \mapsto \cdots \mapsto k \mapsto 2 .
$$

For each $i \in [ m ]$ , choose $c _ { i } \in \{ a _ { i } , b _ { i } \} \setminus \{ 1 \}$ such that $\pi ( c _ { i } ) \notin$ $\{ a _ { i } , b _ { i } \}$ . Such an endpoint exists. If the pair contains 1, choose its other endpoint. Otherwise, if both images remained in the pair, that two-element set would be invariant under a cycle of length $k - 1 \geq 3$ , which is impossible.

Choose a weight tuple $W = ( w _ { 1 } , \dots , w _ { k } )$ realizing the labels $c _ { i }$ , and subtract the first weight from every class weight so that $w _ { 1 } = 0$ . Since $c _ { i } > 1$ , its winning score $w _ { c _ { i } } \cdot x _ { i }$ is strictly positive. Choose one $0 < \varepsilon < 1 / k$ , small enough to preserve all strict winning comparisons simultaneously for every input, and set

$$
\widetilde { w } _ { j } = ( 1 - \varepsilon j ) w _ { j } , \qquad 1 \leq j \leq k .
$$

eEvery previously strict winning comparison remains strict. $\operatorname { I f } c _ { i }$ originally tied with $j \neq c _ { i }$ , then $c _ { i } < j$ , and their common score h is positive. The new diference is

$$
( 1 - \varepsilon c _ { i } ) h - ( 1 - \varepsilon j ) h = \varepsilon ( j - c _ { i } ) h > 0 .
$$

Thus every chosen label becomes a unique winner. Relabel the weights by $\widetilde { w } _ { \pi ( j ) } ^ { \mathrm { n e w } } = \widetilde { w } _ { j }$ . Their winners are $\pi ( c _ { i } )$ , outside the eprescribed pairs.

In both cases, take the efective parameter of the final tuple to obtain u. It is nonzero because the system is nonempty and its winners are strict. The perturbations can be chosen small enough that the weights remain pairwise distinct. Relabeling is performed only after the winners have become unique, so no permutation invariance of the tie rule is assumed. □

## 4.2. Induction and cube interpolation

We now establish the upper bound for $\mathcal { H } _ { \Psi }$

Proposition 2 (Upper bound). For $n , k \geq 3 ,$ we have

$$
\mathrm { N d i m } ( \mathcal { H } _ { \Psi } ) \leq n ( k - 1 ) - 1 .\tag{20}
$$

Lemma 1 gives m $\leq q r$ and makes the comparison map $T :$ a linear isomorphism when $m = q r$ . Thus, it remains to exclude equality. The proof combines induction on the dimension of the feature space with two computations of the degree of a sphere self-map.

In Step 1, we set up the induction and interpolate the parameters realizing the candidate labelings over a cube Q. Applying the comparison map T gives a map $\bar { F }$ satisfying sign conditions on the cube faces. In Step $^ { 2 , }$ we use projection and the induction hypothesis to show that F does not vanish on $\partial Q$ , so its restriction to the boundary can be normalized. In Step 3, Lemma 2 and convex interpolation show that $F ( \partial Q )$ avoids a positive ray. The associated sphere self-map therefore omits a point and has degree zero. In Step 4, a small translation makes the face signs strict while the map remains nonzero on the boundary throughout the translation. We then show that the associated sphere selfmap is homotopic to the identity, giving degree one and yielding the required contradiction.

Proof. For each integer $r \geq 0 ,$ , let $P ( r )$ assert that, for every indexed candidate system with $q r$ inputs in $\mathbb { R } ^ { r }$ , there exists at least one candidate labeling that no parameter in $\mathcal { A } _ { r }$ realizes. We prove $P ( r )$ by induction for all $r \geq 0$ , including the lowerdimensional systems produced by projection.

Step 1: the induction and the interpolating map. For $r = 0 .$ the space $\mathbb { R } ^ { 0 }$ has only one vector, so no tuple of k pairwise distinct weights exists. Even the empty input list has one candidate labeling, which would require such a tuple. Hence $P ( 0 )$ holds.

Assume $r ~ \geq ~ 1$ and $P ( t )$ for every integer $0 \leq t < r ,$ and suppose $P ( r )$ fails. Fix an indexed candidate system $( x _ { i } , a _ { i } , b _ { i } ) _ { i = 1 } ^ { m } .$ with $m = q r \ge 2$ , for which every candidate labeling is realized by a parameter in $\mathcal { A } _ { r }$ . Lemma 1 makes $T : V _ { r } \ \to \ \mathbb { R } ^ { m }$ in (19) an isomorphism, and all inputs are nonzero. For each $s \in \{ - 1 , + 1 \} ^ { m }$ , choose one realization whose first class weight is zero, with efective parameter $\textstyle v _ { s } \in \ A _ { r }$ , where $s _ { i } ~ = ~ - 1$ selects $a _ { i }$ and $s _ { i } ~ = ~ + 1$ selects $b _ { i }$ . Each chosen tuple has pairwise distinct weights. Write $v _ { s } = ( v _ { s , 2 } , \ldots , v _ { s , k } )$ , where $v _ { s , c } \in \mathbb { R } ^ { r }$ for $2 \leq c \leq k ,$ , and set $v _ { s , 1 } = 0 .$

Set $Q = [ - 1 , 1 ] ^ { m }$ . For every $z \in Q$ and $s \in \{ - 1 , + 1 \} ^ { m }$ define

$$
\lambda _ { s } ( z ) = \prod _ { i = 1 } ^ { m } \frac { 1 + s _ { i } z _ { i } } { 2 } ,\tag{21}
$$

$$
G ( z ) = \sum _ { s \in \{ - 1 , + 1 \} ^ { m } } \lambda _ { s } ( z ) v _ { s } , \qquad F ( z ) = T ( G ( z ) ) .\tag{22}
$$

For every $z \in Q$ , the coeficients are nonnegative and sum to one. Write $\begin{array} { l l l } { G ( z ) } & { = } & { ( G _ { 2 } ( z ) , \dots , G _ { k } ( z ) ) } \end{array}$ , where $G _ { c } ( z ) \in \mathbb { R } ^ { r }$ for $2 ~ \leq ~ c ~ \leq ~ k$ , and set $G _ { 1 } ( z ) = 0$ . Write

$F ( z ) = ( F _ { 1 } ( z ) , \dots , F _ { m } ( z ) )$ . By the definition of $T ,$ for every $z \in Q$ and $i \in [ m ]$ , we have

$$
F _ { i } ( z ) = \rho _ { i } ( G ( z ) ) = \bigl ( G _ { b _ { i } } ( z ) - G _ { a _ { i } } ( z ) \bigr ) \cdot x _ { i } .
$$

A vertex s is said to contribute at $z \ \mathrm { i f } \ \lambda _ { s } ( z ) > 0$ . For each $i \in [ m ]$ , only vertices selecting $a _ { i }$ contribute on the face $z _ { i } =$ $^ { - 1 }$ , and only vertices selecting $b _ { i }$ contribute on the face $z _ { i } =$ +1. Thus, for every $i \in [ m ]$ and every $z \in Q$

$$
z _ { i } = - 1 \implies F _ { i } ( z ) \le 0 , \qquad z _ { i } = + 1 \implies F _ { i } ( z ) > 0 .\tag{23}
$$

The interpolated weights need not be pairwise distinct. Interpolation takes place in the full linear space; admissibility is $\mathrm { r e \mathrm { - } }$ quired for the vertex realizations and for the lower-dimensional system used below.

Step 2: exclusion ofboundary zeros. We next show that the map $F$ constructed in Step 1 is nonzero on $\partial Q$ . A boundary zero would yield a lower-dimensional candidate system that contradicts the induction hypothesis. Excluding such zeros will allow us to normalize the boundary restriction of F for the degree argument. Suppose $F ( z ) = 0$ for some $z \in \partial Q$ . Invertibility of $T$ gives $G ( z ) = 0$ . Let

$$
I = \{ i \in [ m ] : | z _ { i } | = 1 \} , \qquad J = [ m ] \setminus I .
$$

The set I is nonempty. On the smallest face containing $z ,$ the coordinates indexed by I are fixed and those indexed by J are free. More explicitly, a vertex contributes precisely when $s _ { i }$ = $z _ { i }$ for every $i \in I .$ For each such vertex,

$$
\lambda _ { s } ( z ) = \prod _ { j \in J } { \frac { 1 + s _ { j } z _ { j } } { 2 } } > 0 ,
$$

since $| z _ { j } | < 1$ on $J ;$ all other coeficients vanish. Thus the contributing vertices realize every candidate choice on J, while the choices on I remain fixed.

By (23), $z _ { i } = - 1$ for every $i \in I .$ . Hence every contributing vertex realization predicts $a _ { i }$ at $x _ { i }$ for each $i \in I . \mathrm { ~ H ~ } a _ { i } > 1$ for some $i \in I .$ , every contributing vertex would give that label a strictly positive score relative to label 1, whose score is zero. The convex average of these positive scores would also be positive, contradicting $G ( z ) = 0$ . Thus $a _ { i } = 1$ for every $i \in I .$ . For every $i \in I .$ , every $c \in \{ 2 , \ldots , k \}$ , and every $s \in \{ - 1 , + 1 \} ^ { m }$ with $\lambda _ { s } ( z ) > 0 .$

$$
v _ { s , c } \cdot x _ { i } \leq 0 .
$$

The weighted average of these nonpositive values, with positive coeficients $\lambda _ { s } ( z )$ for the contributing vertices, is $G _ { c } ( z ) \cdot x _ { i } = 0$ so every value is zero. Set

$$
U = \mathrm { s p a n } \{ x _ { i } : i \in I \} , \qquad \ell = \mathrm { d i m } U \geq 1 , \qquad L = ( U ^ { \perp } ) ^ { q } .\tag{24}
$$

Here $( U ^ { \perp } ) ^ { q }$ denotes the Cartesian product of $q = k - 1$ copies of $U ^ { \perp } ;$ ; explicitly,

$$
\begin{array} { c } { { L = \{ ( v _ { 2 } , \ldots , v _ { k } ) : v _ { c } \in U ^ { \perp } } } \\ { { \mathrm { f o r e v e r y } c = 2 , \ldots , k \} . } } \end{array}
$$

The notation $U ^ { q }$ analogously denotes the Cartesian product of q copies of U. All contributing parameters belong to $L ,$ whose dimension is $q ( r - \ell )$ . If $r - \ell = 0$ , every class weight of a contributing vertex is zero, contradicting pairwise distinctness. Hence $r - \ell \geq 1$

Project the inputs indexed by J orthogonally onto $U ^ { \perp }$ and use orthonormal coordinates on that space. For each contributing vertex $s ,$ all class weights in the tuple $( 0 , v _ { s , 2 } , \ldots , v _ { s , k } )$ lie in $U ^ { \mathbf { \breve { \perp } } }$ . Thus projection preserves every score and predicted label. For every contributing vertex s and every $1 \leq c < d \leq k$ , the diference $v _ { s , c } - v _ { s , d }$ is nonzero in $U ^ { \perp }$ , and the coordinate isomorphism preserves this property. The new tuples therefore still have pairwise distinct weights. For every candidate labeling on $J ,$ a contributing vertex provides a realization after projection. Applying Lemma 1 to these projected feature vectors in $\mathbb { R } ^ { r - \ell }$ gives

$$
| J | \leq q ( r - \ell ) .
$$

Identifying the comparison functionals with their Euclidean coeficient vectors, those indexed by I lie in $U ^ { q }$ and are independent. Indeed, $a _ { i } = 1$ makes their only nonzero coeficient block equal to $x _ { i }$ in block $b _ { i }$ . Since dim $U ^ { q } = q \ell ,$ we have $\lvert I \rvert \leq q \ell .$ Since $| I | + | J | = q r$ , both bounds are equalities:

$$
| I | = q \ell , \qquad | J | = q ( r - \ell ) .\tag{25}
$$

By (25), the projected tuples realize every candidate labeling of $q ( r - \ell )$ indexed feature vectors in dimension $r - \ell < r$ contradicting $P ( r - \ell )$ . Consequently

$$
\forall z \in \partial Q , \qquad F ( z ) \neq 0 .\tag{26}
$$

Step 3: an omitted direction and degree zero. Step 2 ensures that $F$ can be normalized on $\partial Q$ . We next combine Lemma 2 with convex interpolation to show that its boundary image avoids a positive ray. The resulting sphere map will then omit a point and have degree zero.

Fix the single parameter u supplied by Lemma 2. Take any $z \in \partial Q$ and choose an index $i \in [ m ]$ with $| z _ { i } | = 1$ , which exists because z is a boundary point. This index may depend on z. Set $c = a _ { i } \mathrm { i f } z _ { i } = - 1$ , and $c = b _ { i } \mathrm { i f } z _ { i } = + 1$ . Every vertex contributing at z has this same winner c at $x _ { i }$ . Hence, for every such vertex s and every $d \in [ k ]$

$$
( v _ { s , c } - v _ { s , d } ) \cdot x _ { i } \geq 0 .
$$

Convex averaging shows that label c attains the maximum score at $x _ { i }$ under the interpolated weights $G ( z )$ , possibly tied with other labels. For every $t > 0$ , however, tu has a unique winner outside $\{ a _ { i } , b _ { i } \} , \mathrm { s o } G ( z ) \neq t u$ . Since $T$ is a linear isomorphism and z was arbitrary, we obtain, for every $z ~ \in ~ \partial Q$ and every $t > 0 ,$

$$
F ( z ) \neq t T ( u ) .\tag{27}
$$

Let $S ^ { m - 1 }$ be the Euclidean unit sphere and let

$$
\begin{array} { r } { \mathcal { R } : \partial Q \longrightarrow S ^ { m - 1 } , \qquad \mathcal { R } ( z ) = z / \| z \| } \end{array}
$$

be the radial homeomorphism. By (26), the restriction of $F$ : $Q \to \mathbb { R } ^ { m }$ to $\partial Q$ can be normalized to give

$$
{ \widehat { F } } : \partial Q \longrightarrow S ^ { m - 1 } , \qquad { \widehat { F } } ( z ) = { \frac { F ( z ) } { \| F ( z ) \| } } .
$$

To obtain a sphere self-map whose degree we can compute, we use $\mathcal { R } ^ { - 1 } : S ^ { m - 1 }  \partial Q$ to define

$$
\begin{array} { c } { f : S ^ { m - 1 } \longrightarrow S ^ { m - 1 } , } \\ { f ( y ) = \widehat { F } ( \mathcal { R } ^ { - 1 } ( y ) ) } \\ { = \displaystyle \frac { F ( \mathcal { R } ^ { - 1 } ( y ) ) } { \| F ( \mathcal { R } ^ { - 1 } ( y ) ) \| } . } \end{array}
$$

Since $m \geq 2 .$ , the sphere has positive dimension. We use the standard facts that degree is invariant under homotopy, that the identity has degree one, and that a map omitting a point has degree zero because the punctured sphere is contractible (Hatcher, 2002). Since $u \ne 0$ and $T$ is invertible, $T ( u ) ~ \neq ~ 0$ . Equation (27) implies that f omits $T ( u ) / \| T ( u ) \|$ , so

$$
\deg f = 0 .\tag{28}
$$

Step 4: a homotopy to the identity and degree one. To obtain a contradiction with (28), we return to the face conditions in (23). A small translation of F will make these signs strict while keeping the map nonzero on the boundary. This will allow us to construct a homotopy from f to the identity and show that deg $f = 1$

First, compactness and (26) give

$$
\beta = \operatorname* { m i n } _ { z \in \partial Q } \| F ( z ) \| > 0 .
$$

Continuity and (23) also give

$$
\alpha = \operatorname* { m i n } _ { 1 \leq i \leq m } \ \operatorname* { m i n } _ { z \in Q \atop z _ { i } = 1 } F _ { i } ( z ) > 0 .
$$

Each inner minimum is over a compact face, and there are finitely many such faces. Write $\textbf { 1 } = \mathbf { \Gamma } ( 1 , \dots , 1 ) \ \in \ \mathbb { R } ^ { m }$ Choose a single $\delta > 0$ satisfying the bound below and define $F _ { \delta } : Q  \mathbb { R } ^ { m }$ by

$$
0 < \delta < \operatorname* { m i n } \left\{ \alpha , \frac { \beta } { 2 \sqrt { m } } \right\} , \qquad F _ { \delta } ( z ) = F ( z ) - \delta { \bf 1 } .\tag{29}
$$

For $0 \leq t \leq 1$ and $z \in \partial Q$

$$
\| F ( z ) - t \delta \mathbf { 1 } \| \ge \beta - \delta \sqrt { m } > 0 .
$$

Define $\widehat { F } _ { \delta } : \partial Q \to S ^ { m - 1 }$ by

$$
\widehat { F } _ { \delta } ( z ) = \frac { F _ { \delta } ( z ) } { \| F _ { \delta } ( z ) \| } , \qquad z \in \partial Q .
$$

Normalization therefore provides a homotopy between $\widehat F$ and $\widehat { F } _ { \delta }$ . Moreover, for every $i \in [ m ]$ and every $z \in Q$

$$
z _ { i } = - 1 \Longrightarrow ( F _ { \delta } ) _ { i } ( z ) < 0 , \qquad z _ { i } = + 1 \Longrightarrow ( F _ { \delta } ) _ { i } ( z ) > 0 .
$$

For every $t \in [ 0 , 1 ]$ and $z \in \partial Q$ , define

$$
H _ { t } ( z ) = ( 1 - t ) F _ { \delta } ( z ) + t z .
$$

Choose an index $i \in [ m ] \mathrm { w i t h } | z _ { i } | = 1$ . The two i-th coordinates in this convex combination both have the strict sign of $z _ { i } ,$ so $( H _ { t } ( z ) ) _ { i } \neq 0$ . Thus $H _ { t } ( z ) \neq 0$ for every $t \in [ 0 , 1 ]$ and every $z \in \partial Q$ . After normalization, this gives a homotopy from $\widehat { F } _ { \delta }$ bto R. The small translation uses the boundary nonvanishing already proved in Step 2; no strictification of the original vertex realizations is assumed. Precomposing both homotopies with $\mathcal { R } ^ { - 1 }$ gives

$$
\begin{array} { r l } & { f = \widehat { F } \circ \mathcal { R } ^ { - 1 } \simeq \widehat { F } _ { \delta } \circ \mathcal { R } ^ { - 1 } } \\ & { \qquad \simeq \mathcal { R } \circ \mathcal { R } ^ { - 1 } = \mathrm { i d } _ { S ^ { m - 1 } } . } \end{array}
$$

Hence deg $f = 1$ , contradicting (28).

Example 2 (continued). For $n = k = 3 .$ , the efective parameter space $V _ { 3 } = \mathbb { R } ^ { 6 }$ has dimension 6. By Lemma 1, any shattered set has at most six points. The induction and cube-interpolation argument in Proposition 2 excludes the six-point case, so

$$
\begin{array} { r } { \mathrm { N d i m } ( \mathcal { H } _ { \Psi } ) \le 3 ( 3 - 1 ) - 1 = 5 . } \end{array}
$$

The five-point construction in Example 1 attains this bound. Consequently,

$$
\mathrm { N d i m } ( \mathcal { H } _ { \Psi } ) = 5 .
$$

## 5. Conclusion

For homogeneous multiclass linear predictors with pairwise distinct class weights, we prove Ndim $( \mathcal { H } _ { \Psi } ) = n ( k - 1 ) - 1$ for $n , k \geq 3$ by establishing matching lower and upper bounds. The present result, together with the results in Shalev-Shwartz and Ben-David (2014) and Pan and Li (2026), determines the Natarajan dimension of this hypothesis class for all $n , k \geq 2$

## References

Ben-David, S., Cesa-Bianchi, N., Haussler, D., Long, P. M., 1995. Characterizations of learnability for classes of {0, . . . , n}-valued functions. Journal ofComputer and System Sciences 50 (1), 74–86. DOI: 10.1006/jcss.1995.1008.

Blumer, A., Ehrenfeucht, A., Haussler, D., Warmuth, M. K., 1989. Learnability and the Vapnik–Chervonenkis dimension. Journal of the ACM 36 (4), 929–965. DOI: 10.1145/76359.76371.

Daniely, A., Sabato, S., Shalev-Shwartz, S., 2012. Multiclass learning approaches: A theoretical comparison with implications, in: Pereira, F., Burges, C. J. C., Bottou, L., Weinberger, K. Q. (Eds.), Advances in Neural Information Processing Systems, vol. 25. Curran Associates, Inc., pp. 485–493. URL: https:// proceedings.neurips.cc/paper\_files/paper/2012/hash/ 19f3cd308f1455b3fa09a282e0d496f4-Abstract.html.

Daniely, A., Shalev-Shwartz, S., 2014. Optimal learners for multiclass problems, in: Balcan, M. F., Feldman, V., Szepesvári, C. (Eds.), Proceedings of the 27th Conference on Learning Theory, Barcelona, Spain. Proceedings of Machine Learning Research, vol. 35, PMLR, pp. 287–316. URL: https://proceedings.mlr.press/v35/ daniely14b.html.

Daniely, A., Sabato, S., Ben-David, S., Shalev-Shwartz, S., 2015. Multiclass learnability and the ERM principle. Journal of Machine Learning Research 16 (72), 2377–2404. URL: https://www. jmlr.org/papers/v16/daniely15a.html.

Hatcher, A., 2002. Algebraic Topology. Cambridge University Press, Cambridge. URL: https://pi.math.cornell.edu/\~hatcher/ AT/ATpage.html.

Jin, Y., 2023. Upper bounds on the Natarajan dimensions of some function classes, in: 2023 IEEE International Symposium on Information Theory (ISIT), Taipei, Taiwan. IEEE, pp. 1020–1025. DOI: 10.1109/ISIT54713.2023.10206618.

Natarajan, B. K., 1989. On learning sets and functions. Machine Learning 4 (1), 67–97. DOI: 10.1007/BF00114804.

Pan, Y., Li, B., 2026. Bounds on the Natarajan dimension of a class of linear multi-class predictors. Pattern Recognition Letters 200, 129– 134. DOI: 10.1016/j.patrec.2025.12.012.

Shalev-Shwartz, S., Ben-David, S., 2014. Understanding Machine Learning: From Theory to Algorithms. Cambridge University Press, Cambridge. DOI: 10.1017/CBO9781107298019.

Vapnik, V. N., Chervonenkis, A. Ya., 1971. On the uniform convergence of relative frequencies of events to their probabilities. Theory of Probability & Its Applications 16 (2), 264–280. DOI: 10.1137/1116025.

Ziegler, G. M., 1995. Lectures on Polytopes. Graduate Texts in Mathematics, vol. 152. Springer, New York. DOI: 10.1007/978-1-4613- 8431-1.