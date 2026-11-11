# Nonadaptive One-Sided Testing of Binary Rank

Nader H. Bshouty Faculty of Computer Science Technion Israel Institute of Technology bshouty@cs.technion.ac.il

## Abstract

The binary rank of a binary matrix is the minimum number of disjoint all-one rectangles that partition its one-entries. We study entry-query testing of binary rank at most d, with one-sided error and nonadaptive queries. We give a tester using ${ \tilde { O } } ( d ^ { 4 } { \bar { / } } \epsilon ^ { 2 } )$ queries, independent of the matrix dimensions and a general lower bound of $\tilde { \Omega } ( d ^ { 3 / 2 } / \epsilon )$

## 1 Introduction

Suppose a binary matrix records which people have which attributes. A set of people who all share a set of attributes gives an all-one rectangle in the matrix. We ask whether all the one-entries can be divided into a small number of such rectangles, with each one-entry belonging to exactly one rectangle. The minimum number needed is called the binary rank.

If the matrix is very large, even reading all its entries can be expensive. Property testing asks a less demanding question: does the matrix have binary rank at most a given integer d, or is it far from every such matrix? Here “far” means that at least an ϵ fraction of all entries must change. The tester may behave arbitrarily on matrices that lie between these two cases. A query asks for the value at one chosen row and column. Our goal is to use few queries, independently of the matrix dimensions.

We impose two requirements. The tester is nonadaptive: it chooses all queried positions before seeing any entries. It also has one-sided error: it always accepts a matrix of binary rank at most d. On every ϵ-far matrix, it must reject with probability at least 2/3. Probability refers to the tester’s random choices. Nonadaptivity allows all entries to be requested at once. One-sided error gives rejection a useful meaning: the entries already seen rule out every matrix of binary rank at most d.

## 1.1 Main results

We prove the following two results.

Theorem 1.1 (Nonadaptive upper bound). For every integer $d \geq 1$ and every $0 < \epsilon < 1$ , there is a randomized nonadaptive algorithm that, given entry-query access to a binary matrix $M \in \{ 0 , 1 \} ^ { n \times m }$ makes

$$
O \left( \frac { d ^ { 2 } ( d + \log ( 1 / \epsilon ) ) ^ { 2 } } { \epsilon ^ { 2 } } \right) = \tilde { O } \left( \frac { d ^ { 4 } } { \epsilon ^ { 2 } } \right) .
$$

queries and satisfies the following guarantees:

1. If M has binary rank at most d, the algorithm accepts with probability one.

2. If changing M into a matrix of binary rank at most d requires changing at least ϵnm entries, the algorithm rejects with probability at least $2 / 3$

Theorem 1.2 (Nonadaptive lower bound). Let $d \geq 4$ be an integer, let $0 < \epsilon \leq 1 / 1 6$ , and let N be an integer satisfying $N \geq ~ 1 0 2 4 d / \epsilon$ . Every randomized nonadaptive algorithm with entry-query access to matrices in $\{ 0 , 1 \} ^ { N \times N }$ that

1. accepts every matrix of binary rank at most d with probability one, and

2. rejects, with probability at least $2 / 3$ , every matrix for which at least $\epsilon N ^ { 2 }$ entries must be changed to obtain a matrix of binary rank at most d.

must make at least

$$
\Omega \left( \frac { d ^ { 3 / 2 } \log ( 1 / \epsilon ) } { \epsilon } \right)
$$

queries in the worst case.

## 1.2 Previous work and our contribution

Binary rank has both an algebraic description, by binary matrix factors, and a combinatorial description, by disjoint rectangles. We explain their equivalence in Section 2. Parnas’s survey [5] provides broader background on binary rank and related notions of rank.

Earlier general testing results already imply dimension-independent tests. Alon, Fischer, and Newman [1] studied testing bipartite graphs with forbidden induced subgraphs. Parnas, Ron, and Shraibman [7, Section 4] explain how that framework applies to binary rank, although the resulting dependence on d is very large. Nakar and Ron [4] developed a general framework for graph partition properties. Its specialization to binary rank gives a nonadaptive one-sided bound ${ \widetilde O } ( 2 ^ { 4 d } / \epsilon ^ { 4 } )$ .

More direct arguments gave better bounds. Parnas, Ron, and Shraibman [7] proved a nonadaptive one-sided bound ${ \cal O } ( 2 ^ { 2 d } / \epsilon ^ { 2 } )$ and an adaptive one-sided bound $O ( 2 ^ { 2 d } / \epsilon )$ . Bshouty [2] improved these to $\widetilde { \cal O } ( 2 ^ { d } / \epsilon ^ { 2 } )$ and $\widetilde { \cal O } ( 2 ^ { d } / \epsilon )$ , respectively. That paper also considers rectangles with bounded overlap. Here the rectangles are disjoint, which is the case $s = 1$ in its terminology.

Parnas [6] recently obtained an adaptive two-sided tester using ${ \cal \tilde { O } } ( d ^ { 3 } / \epsilon ^ { 2 } )$ queries. Thus polynomial query complexity itself is already known. Our upper bound obtains polynomial complexity with both nonadaptivity and one-sided error. It has a larger dependence on d at constant ϵ than that adaptive two-sided bound.

<table><tr><td>Upper-bound source</td><td>Model</td><td>Queries</td></tr><tr><td>Nakar-Ron [4, 7]</td><td>Nonadaptive, one-sided</td><td> $\widetilde { \cal O } ( 2 ^ { 4 d } / \epsilon ^ { 4 } )$ </td></tr><tr><td>Parnas-Ron-Shraibman [7] Nonadaptive, one-sided</td><td></td><td> $O ( 2 ^ { 2 d } / \epsilon ^ { 2 } )$ </td></tr><tr><td>Bshouty [2]</td><td>Nonadaptive, one-sided</td><td> $\widetilde { \cal O } ( 2 ^ { d } / \epsilon ^ { 2 } )$ </td></tr><tr><td>Parnas [6]</td><td>Adaptive, two-sided</td><td> $\widetilde { \cal O } ( d ^ { 3 } / \epsilon ^ { 2 } )$ </td></tr><tr><td>Theorem 3.4</td><td>Nonadaptive, one-sided</td><td> ${ \widetilde O } ( d ^ { 4 } / { \epsilon } ^ { 2 } )$ </td></tr></table>

Table 1: Selected upper bounds for binary rank at most d. The models difer as indicated.

## 1.3 The Upper Bound

One ingredient in our upper-bound proof is also known: the description of binary factors by cliques and independent sets in a graph, given in Parnas’s work [6, Lemma 9]. We explain below how we use this construction.

Let $M \in \{ 0 , 1 \} ^ { n \times m }$ be the matrix being tested. Our tester independently samples $t = \tilde { O } ( d ^ { 2 } / \epsilon )$ row indices and t column indices, each uniformly from the corresponding set of indices. It queries all entries of the resulting submatrix S. It accepts if $S$ has binary rank at most $d ,$ and rejects otherwise. Thus, all queries are chosen before any entry is read. The tester has one-sided error because every submatrix of a matrix of binary rank at most d also has binary rank at most $d .$

If S is accepted, it has binary rank at most d. Therefore, we can write $S \ = \ A B$ , where $A \in \{ 0 , 1 \} ^ { t \times d }$ and $B \in \{ 0 , 1 \} ^ { d \times t }$ , and multiplication uses ordinary arithmetic. In the proof, we select basis of the rows of A. That is, linearly independent rows of A that span its row space over R. We also select basis of the columns of B. Each selection contains at most d vectors.

We also form a graph with vertices $1 , \ldots , d .$ Two vertices are joined by an edge if some row of A has a 1 in both corresponding positions. No column of B can have a 1 in both positions of an edge, because its product with that row would then be at least 2. Consequently, the positions containing 1 in each row of A form a clique, and those in each column of B form an independent set.

For the proof, we also construct a matrix $H \in \{ 0 , 1 \} ^ { n \times m }$ that agrees with M at every sampled entry. Write ${ S _ { i j } } = { M _ { r _ { i } , c _ { j } } }$ , where $r _ { 1 } , \ldots , r _ { t }$ and $c _ { 1 } , \ldots , c _ { t }$ are the sampled row and column indices. To construct H, we use the selected basis rows of A, the selected basis columns of B, their positions within $A$ and B, and the graph. For each selected basis row i of $A ,$ we also use all entries of row $r _ { i }$ of M. For each selected basis column $j$ of $B ,$ we use all entries of column $c _ { j }$ of $M$ . The construction described below uses these vectors, positions, graph, and entries to define a matrix $H \in \{ 0 , 1 \}$ n×m of binary rank at most $d . ^ { 1 }$ We show that whenever S has binary rank at most $d ,$ the bases and graph obtained from its factorization $S = A B$ give a matrix H satisfying

$$
H _ { r _ { i } , c _ { j } } = M _ { r _ { i } , c _ { j } } = S _ { i j } \qquad { \mathrm { f o r ~ a l l ~ } } 1 \leq i , j \leq t .
$$

Thus, H agrees with M at every sampled entry.

For the probability analysis, fix a matrix M that is ϵ-far from every binary matrix of binary rank at most d. We consider all possible choices of the basis rows of A, the basis columns of B, their row and column numbers, and the graph. We examine each such choice separately. If row i of A is selected for its basis, we hold fixed the index $r _ { i }$ . Similarly, if column $j$ of B is selected for its basis, we hold fixed $c _ { j }$ . With these indices fixed, the construction defines a fixed matrix H. The other sampled indices remain independent and uniform, since all indices were sampled independently.

Since H has binary rank at most $d ,$ it difers from M on at least ϵnm entries. We bound the probability that H and M agree at every entry determined by a remaining sampled row and a remaining sampled column. We then bound the number of possible choices of bases, their row and column numbers, and the graph, and apply a union bound over all these choices. This bounds the probability of acceptance, because acceptance implies that at least one choice produces a matrix H that agrees with every queried entry.

## 1.4 The Lower Bound

Our lower-bound proof uses the fact that a one-sided tester can reject only when the entries it has queried rule out every binary matrix of binary rank at most d. If we can fill in the unqueried entries

to obtain such a matrix while preserving all the answers already given, then the tester must accept.   
We call such a filling a completion of the queried entries.

We construct random binary matrices that are ϵ-far from every binary matrix of binary rank at most d. We then show that, when the tester makes too few queries, the entries it observes often have a completion of binary rank at most d. Thus, although the entire matrix is far from the property, the queried entries do not provide enough information for a one-sided tester to reject it.

To obtain the dependence on ϵ, we place random bits in submatrices with diferent combinations of numbers of rows and columns. The proof controls both the distance of the entire matrix from the property and the probability that the queried entries still admit a completion of binary rank at most d. Together, these estimates give the explicit lower bound.

## 2 Background and notation

## 2.1 Rectangles and binary factors

For a positive integer a, write $[ a ] = \{ 1 , \dots , a \}$ . A binary matrix has entries in $\{ 0 , 1 \}$ . A rectangle is a set $R \times C$ of matrix positions, where R is a set of row indices and C is a set of column indices.

Definition 2.1. The binary rank br(M) of a binary matrix M is the smallest number of all-one rectangles whose disjoint union is the set of one-entries of M. The all-zero matrix has binary rank zero. We write $B _ { d } ^ { n , m }$ for the set of $n \times m$ binary matrices with binary rank at most $d .$

Matrix multiplication gives a useful way to record the rectangles. Assign one coordinate to each rectangle. Put $A _ { i u } = 1$ when row index i belongs to rectangle $u ,$ and put $B _ { u j } = 1$ when column index $j$ belongs to it. Then

$$
( A B ) _ { i j } = \sum _ { u = 1 } ^ { d } A _ { i u } B _ { u j }
$$

counts the rectangles containing position $( i , j )$ . Therefore br $( M ) \leq d$ is equivalent to the existence of a factorization

$$
M = A B , \qquad A \in \{ 0 , 1 \} ^ { n \times d } , \quad B \in \{ 0 , 1 \} ^ { d \times m } ,
$$

with ordinary integer multiplication. Conversely, each column u of A and row u of B define the rectangle $\{ i : A _ { i u } = 1 \} \times \{ j : B _ { u j } = 1 \}$ . Since each entry is zero or one, the rectangles cannot overlap. If fewer than d rectangles are needed, append zero columns to A and zero rows to B.

This difers from Boolean rank, where sums are replaced by logical OR, and the rectangles covering the entries equal to 1 may overlap. All linear algebra in this paper is over the real numbers.

The support of a binary vector $^ { a , }$ denoted by $\operatorname { s u p p } ( a )$ , is the set of indices at which its entries equal 1. Thus, for two binary vectors a and b of the same length, their inner product $a \cdot b$ equals $| \operatorname { s u p p } ( a ) \cap \operatorname { s u p p } ( b ) |$ . We will also use the fact that rank $( M ) \leq \mathrm { b r } ( M )$ , whereas the reverse inequality need not hold.

## 2.2 Distance, partial answers, and one-sided error

For $M \in \{ 0 , 1 \} ^ { n \times m }$ , the normalized Hamming distance from a matrix $H \in \{ 0 , 1 \} ^ { n \times m }$ is

$$
\operatorname { d i s t } ( M , H ) = { \frac { | \{ ( i , j ) : M _ { i j } \neq H _ { i j } \} | } { n m } } .
$$

The matrix M is $\epsilon { - } f a r$ from $B _ { d } ^ { n , m }$ when this distance is at least ϵ for every $H \in B _ { d } ^ { n , m }$

A transcript records each position queried by the tester and the value of the matrix entry at that position. We view it as a partial matrix: queried entries are specified and other entries are blank. A completion fills the blanks without changing specified entries. For example, the partial matrix

$$
\left( { \begin{array} { l l } { 1 } & { * } \\ { * } & { 1 } \end{array} } \right)
$$

has a binary-rank-one completion, obtained by filling both blanks with ones. But specifying its upper-right entry as zero rules out rank one.

This connection between partial answers and completions is the basis of the lower bound.

Lemma 2.2 (One-sided rejection requires an obstruction). Fix a seed for which the resulting deterministic nonadaptive tester accepts every matrix in $B _ { d } ^ { n , m }$ . It can reject a transcript only if that transcript has no completion in $B _ { d } ^ { n , m }$

Proof. Suppose there is a matrix $H \in B _ { d } ^ { n , m }$ that agrees with all the queried entries. The tester sees exactly the same answers on H, because its query positions were chosen before any answers. Since it accepts H, it must accept this transcript. □

## 2.3 Graphs and elementary probability

A proper coloring of a graph assigns colors to vertices so that endpoints of an edge have diferent colors. Each set of vertices with one color is therefore independent. We use cliques and independent sets to describe the rows of A and the columns of B in a binary factorization $M = A B$ . We use graph colorings to construct completions of partial matrices.

We need a bound for a sum of independent zero-one variables.

Lemma 2.3 (Chernof bound). Let Z be a sum of independent random variables taking values zero and one, and let $\mu = \mathbb { E } [ Z ]$ . For $0 < \delta < 1$ ，

$$
\operatorname* { P r } [ Z \leq ( 1 - \delta ) \mu ] \leq \exp ( - \delta ^ { 2 } \mu / 2 ) .
$$

## 3 A general polynomial upper bound

## 3.1 Globally valid reconstructions

Our tester accepts when the sampled submatrix has binary rank at most d. To analyze acceptance on an ϵ-far input M, we construct a binary matrix H with the same dimensions as M and binary rank at most d. The matrix H will agree with M at every sampled entry.

We construct H as the product of two binary matrices, the first with d columns and the second with d rows. However, such a product need not be binary. If a row of the first matrix and a column of the second both have 1 in two common positions, the corresponding entry of their product is at least 2. The following known graph construction ensures that every entry of the product is 0 or 1. We include its short proof because we use this construction to define H.

Lemma 3.1 (Parnas [6, Lemma 9]). Let $A \in \{ 0 , 1 \} ^ { n \times d }$ and $B \in \{ 0 , 1 \} ^ { d \times m }$ , and let G be a simple graph on [d]. Suppose the support<sup>2</sup> of every row of A is a clique in G, and the support of every column of B is an independent set in G. Then every entry of AB belongs to {0, 1}, and $\ker ( A B ) \leq d$

Conversely, if $M = A B$ is a binary matrix, then there exists a simple graph G on [d] such that the support of every row of A is a clique in G, and the support of every column of B is an independent set in G.

Proof. A clique and an independent set intersect in at most one vertex, so their binary vectors have inner product at most one. For the converse, put an edge $\{ u , v \}$ in G whenever some row of A contains both coordinates. A column of B cannot contain both endpoints of such an edge, since its product with that row would be at least two. □

We next construct a binary matrix H of the same dimensions as M and of binary rank at most d. The construction uses a graph, two lists of binary vectors, and entries in selected rows and columns of M. We will later show how to choose these so that H agrees with M on an accepted sample.

Choose lists of row indices $I = ( i _ { 1 } , \dots , i _ { k } )$ and column indices $J = \left( j _ { 1 } , \ldots , j _ { \ell } \right)$ of M, where $0 \leq k , \ell \leq d .$ . Choose binary matrices $X \in \{ 0 , 1 \} ^ { k \times d }$ , and $Y \in \{ 0 , 1 \} ^ { d \times \ell }$ , and a simple graph G on [d].

Let U be the real linear space spanned by the rows of X, and let V be the real linear space spanned by the columns of $Y \colon U = \operatorname { s p a n } \cdot$ {rows of $X \}$ , and V = span{columns of Y}. The span of an empty list is {0}. We use the notation

$$
M [ i , J ] = ( M _ { i , j _ { 1 } } , \ldots , M _ { i , j _ { \ell } } ) , \qquad M [ I , j ] = ( M _ { i _ { 1 } , j } , \ldots , M _ { i _ { k } , j } ) ^ { \mathsf { T } } .
$$

Thus, $M [ i , J ]$ contains the entries of row i in the selected columns, and $M [ I , j ]$ contains the entries of column j in the selected rows.

For each row $i \in [ n ]$ , consider all binary row vectors a of length d satisfying

$$
a \in U , \qquad \operatorname { s u p p } ( a ) { \mathrm { ~ i s ~ a ~ c l i q u e ~ i n ~ } } G , \qquad a Y = M [ i , J ] .
$$

The last equation requires a to have inner product $M _ { i , j _ { q } }$ with column q of $Y .$ , for every $q \in [ \ell ]$ . If such vectors exist, let $\widehat { a } _ { i }$ be the first one in lexicographic order. If no such vector exists, set $\widehat { a } _ { i } = 0$

Similarly, for each column $j \in [ m ]$ , consider all binary column vectors b of length d satisfying

$$
b \in V , \qquad \operatorname { s u p p } ( b ) { \mathrm { ~ i s ~ a n ~ i n d e p e n d e n t ~ s e t ~ i n ~ } } G , \qquad X b = M [ I , j ] .
$$

The last equation requires b to have inner product $M _ { i _ { p } , j }$ with row p of $X$ , for every $p \in [ k ]$ . Let $\widehat { b } _ { j }$ be the lexicographically first such vector, or set $\widehat { b } _ { j } = 0$ if none exists.

Define

$$
H _ { i j } = \widehat { a } _ { i } \widehat { b } _ { j } \qquad ( i \in [ n ] , \ j \in [ m ] ) .\tag{1}
$$

The support of every chosen row vector is a clique, and the support of every chosen column vector is an independent set. This remains true when we choose the zero vector, whose support is empty. By Lemma 3.1, every entry of H belongs to $\{ 0 , 1 \}$ . The chosen vectors give a binary factorization of H with d columns in the first matrix and d rows in the second. Hence $H \in B _ { d } ^ { n , m }$

For fixed $M , I , J , X , Y , G$ , the construction determines H uniquely. It always gives a matrix of binary rank at most $d ,$ although H need not agree with M at the selected entries. In particular, choosing the zero vector when no solution exists need not satisfy the corresponding equation.

Constructing all the vectors $\widehat { a } _ { i }$ uses all entries in the columns listed in J. Constructing all the vectors $\widehat { b } _ { j }$ uses all entries in the rows listed in I. These entries are used only to define H in the proof. The tester queries only the sampled submatrix and does not construct H.

The next lemma shows that, whenever the sampled submatrix S has binary rank at most $d ,$ we can choose $I , J , X , Y , G$ in the preceding construction so that H agrees with M at every sampled entry.

Lemma 3.2 (Encoding an accepted sample). Let S be a sampled $t \times t$ submatrix, where repeated sampled indices are permitted. $I f \operatorname { b r } ( S ) \leq d ,$ , some description of the preceding form, using at most d sampled row positions and d sampled column positions, produces a matrix H agreeing with every entry of S.

Proof. Write ${ S _ { i j } } = { M _ { r _ { i } , c _ { j } } }$ , where $r _ { 1 } , \ldots , r _ { t }$ and $c _ { 1 } , \ldots , c _ { t }$ are the sampled row and column indices. Since br $( S ) \leq d ,$ we can write

$$
S = A B , \qquad A \in \{ 0 , 1 \} ^ { t \times d } , \qquad B \in \{ 0 , 1 \} ^ { d \times t } .
$$

If necessary, add zero columns to A and zero rows to B to obtain these dimensions. Denote row i of A by $a _ { i }$ and column $j$ of B by $b _ { j }$ . Thus, $S _ { i j } = a _ { i } b _ { j }$

Choose linearly independent rows of A that span its row space over R. Let their positions in A be $p _ { 1 } , \ldots , p _ { k }$ , and let X be the matrix consisting of these rows. Similarly, choose linearly independent columns of B that span its column space over R. Let their positions in B be $q _ { 1 } , \ldots , q _ { \ell } .$ and let Y consist of these columns. Both k and ℓ are at most d.

In the preceding construction, use

$$
I = ( r _ { p _ { 1 } } , \ldots , r _ { p _ { k } } ) , \qquad J = ( c _ { q _ { 1 } } , \ldots , c _ { q _ { \ell } } ) .
$$

Choose the graph G supplied by Lemma 3.1 for the factorization $S = A B$ . Thus, the support of every row of A is a clique in G, and the support of every column of B is an independent set in G. By the choices of X and Y , the spaces U and V in the construction are the row space of A and the column space of B, respectively.

For each sampled row position i, the vector $a _ { i }$ satisfies all the conditions used to choose $\widehat { a } _ { r _ { i } }$ Indeed, $a _ { i } \in U$ , its support is a clique in $G ,$ and

$$
a _ { i } Y = ( S _ { i , q _ { 1 } } , \ldots , S _ { i , q \ell } ) = M [ r _ { i } , J ] .
$$

Therefore, at least one vector satisfies these conditions. The construction chooses such a vector $\widehat { a } _ { r _ { i } }$ which also satisfies $\widehat { a } _ { r _ { i } } Y = M [ r _ { i } , J ]$ . Subtracting gives

$$
( \widehat { a } _ { r _ { i } } - a _ { i } ) Y = 0 .
$$

Since the columns of Y span V , the vector $\widehat { a } _ { r _ { i } } - a _ { i }$ has inner product zero with every vector in V .

Similarly, for each sampled column position $j$ , the vector $b _ { j }$ belongs to $V$ , its support is an independent set in G, and

$$
X b _ { j } = ( S _ { p _ { 1 } , j } , \ldots , S _ { p _ { k } , j } ) ^ { \mathsf { T } } = M [ I , c _ { j } ] .
$$

Thus, the construction chooses a vector $\widehat { b } _ { c _ { j } } \in V$ satisfying the same equation. Consequently,

$$
X ( \widehat { b } _ { c _ { j } } - b _ { j } ) = 0 .
$$

Since the rows of X span U, every vector in U has inner product zero with $\widehat { b } _ { c _ { j } } - b _ { j }$

We can now compare H and S. For every $i , j \in [ t ]$ 2

$$
\begin{array} { c } { { H _ { r _ { i } , c _ { j } } - S _ { i j } = \widehat { a } _ { r _ { i } } \widehat { b } _ { c _ { j } } - a _ { i } b _ { j } } } \\ { { { } } } \\ { { { } = ( \widehat { a } _ { r _ { i } } - a _ { i } ) \widehat { b } _ { c _ { j } } + a _ { i } ( \widehat { b } _ { c _ { j } } - b _ { j } ) = 0 . } } \end{array}
$$

The first term is zero because $\widehat { b } _ { c _ { j } } \in V$ . The second is zero because $a _ { i } \in U$ . Hence $H _ { r _ { i } , c _ { j } } = S _ { i j } = M _ { r _ { i } , c _ { j } }$ at every sampled entry. □

## 3.2 A validation grid and the tester

We now know that an accepted sample has a short description. The next step is probabilistic: a fixed candidate that difers from the input in many places should fail on a random grid.

Lemma 3.3 (Avoiding a set of errors). Suppose two fixed binary matrices difer on at least an ϵ fraction of their entries. Sample a row indices and b column indices independently and uniformly, with replacement. The probability that the resulting grid contains no disagreement is at most

$$
e ^ { - \epsilon \operatorname* { m i n } \left\{ a , b \right\} } .
$$

Proof. Let $r _ { 1 } , \ldots , r _ { a }$ and $c _ { 1 } , \ldots , c _ { b }$ be the sampled row and column indices, and put $k = \operatorname* { m i n } \{ a , b \}$ The positions $( r _ { 1 } , c _ { 1 } ) , \ldots , ( r _ { k } , c _ { k } )$ are independent and uniformly distributed over all matrix positions. At each position, the probability of agreement is at most $1 - \epsilon$ . Therefore, the probability of agreement at all these positions is at most $( 1 - \epsilon ) ^ { k } \leq e$ −ϵ min{a,b} □

We can now analyze the tester. Its rule is simple: sample rows and columns, read their intersections, and accept exactly when these entries admit a binary factorization through d coordinates.

Theorem 3.4. For every $d \geq 1$ and $0 < \epsilon < 1$ , there is a nonadaptive one-sided tester for $B _ { d } ^ { n , m }$ using

$$
O \left( \frac { d ^ { 2 } ( d + \log ( 1 / \epsilon ) ) ^ { 2 } } { \epsilon ^ { 2 } } \right)
$$

entry queries.

Proof. Choose

$$
t = \left\lceil \frac { 6 4 d ( d + \log ( 1 / \epsilon ) ) } { \epsilon } \right\rceil .\tag{2}
$$

Independently sample row indices $r _ { 1 } , \ldots , r _ { t }$ uniformly from [n] and column indices $c _ { 1 } , \ldots , c _ { t }$ uniformly from [m], with replacement. Define $S _ { i j } = M _ { r _ { i } , c _ { j } } { \mathrm { ~ f o r ~ } } i , j \in [ t ]$ . The tester queries these entries and accepts if and only if br $( S ) \leq d .$ . If br $( M ) \leq d .$ , then br $( S ) \leq d$ for every sample, so the tester always accepts.

For the remainder of the proof, fix a matrix M that is ϵ-far from every matrix in $B _ { d } ^ { n , m }$ . All probabilities below are over the sampled indices.

Let D be the set of all choices $\theta = ( P , Q , X , Y , G )$ , where $P , Q \subseteq [ t ] , | P | , | Q | \leq d , X \in \{ 0 , 1 \} ^ { | P | \times d }$ $Y \in \{ 0 , 1 \} ^ { d \times | Q | }$ , and G is a simple graph on [d]. The sets P and Q specify positions in the sample. We list their elements in increasing order. For each $\theta \in \mathcal { D }$ , define $Z _ { \theta } = \left( ( r _ { i } ) _ { i \in P } , ( c _ { j } ) _ { j \in Q } \right)$ . Thus, $Z _ { \theta }$ records the sampled indices at the selected positions. Let $H ^ { \theta }$ be the matrix obtained from the construction (1), using $I = ( r _ { i } ) _ { i \in P } , J = ( c _ { j } ) _ { j \in Q }$ , together with $X , Y , G$ . Define the event

$$
E _ { \theta } = \left\{ H _ { r _ { i } , c _ { j } } ^ { \theta } = M _ { r _ { i } , c _ { j } } { \mathrm { ~ f o r ~ e v e r y ~ } } i , j \in [ t ] \right\} .
$$

Conditioning on any possible value $Z _ { \theta } = z$ makes $H ^ { \theta }$ fixed. The remaining sampled indices are still independent and uniform. Since $H ^ { \theta } \in B _ { d } ^ { n , m }$ , it difers from M on at least an ϵ fraction of entries. Moreover, $E _ { \theta }$ requires agreement on the grid formed by the remaining $t - | P |$ row samples and $t - | Q |$ column samples. Both numbers are at least $t - d .$ . Lemma 3.3 therefore bounds the conditional probability of $E _ { \theta }$ by $e ^ { - \epsilon ( t - d ) }$

We now obtain the probability bound:

$$
{ \begin{array} { r l } { \operatorname* { P r } _ { r , c } [ \operatorname { b r } ( S ) \leq d ] \leq \operatorname* { P r } \left[ \bigcup _ { \theta \in { \mathcal { D } } } E _ { \theta } \right] } & { { \mathrm { ( L e m m a ~ 3 . 2 ) } } } \\ { \leq \displaystyle \sum _ { \theta \in { \mathcal { D } } } \operatorname { P r } [ E _ { \theta } ] \leq | { \mathcal { D } } | \operatorname* { m a x } \operatorname* { P r } [ E _ { \theta } ] } & { { \mathrm { ( u n i o n ~ b o u n d ) } } } \\ { = | { \mathcal { D } } | \operatorname* { m a x } _ { \theta \in { \mathcal { D } } } \sum _ { z } \operatorname { P r } [ Z _ { \theta } = z ] \operatorname* { P r } [ E _ { \theta } \ | \ Z _ { \theta } = z ] } & { { \mathrm { ( l a w ~ o f ~ t o t a l ~ p r o b a b i l i t y ) } } } \\ { \leq | { \mathcal { D } } | \operatorname* { m a x } _ { \theta \in { \mathcal { D } } } \sum _ { z } \operatorname* { P r } [ Z _ { \theta } = z ] e ^ { - \epsilon ( t - d ) } } & { { \mathrm { ( t h e ~ c o n d i t i o n a l ~ b o u n d ~ a b o v e ) } } } \\ { = | { \mathcal { D } } | e ^ { - \epsilon ( t - d ) } } & { { \mathrm { ( t h e ~ p r o b a b i l i t i e s ~ s u m ~ t o ~ o n e ) } } . } \end{array} }\tag{3}
$$

Here each sum over z ranges over all possible values of $Z _ { \theta }$ .

It remains to count D and substitute the value of t. For $| P | = k$ , there are $\textstyle { \binom { t } { k } }$ choices of P and $2 ^ { k d }$ choices of X. The same count applies to $Q , Y$ . There are $2 ^ { \binom { d } { 2 } }$ choices of G. Hence

$$
| D | = 2 ^ { { \binom { d } { 2 } } } \left( \sum _ { k = 0 } ^ { d } { \binom { t } { k } } 2 ^ { k d } \right) ^ { 2 } \leq 2 ^ { 2 d ^ { 2 } + { \binom { d } { 2 } } } \left( \sum _ { k = 0 } ^ { d } { \binom { t } { k } } \right) ^ { 2 } \leq 2 ^ { 3 d ^ { 2 } } ( d + 1 ) ^ { 2 } ( t + 1 ) ^ { 2 d } \leq 2 ^ { 3 d ^ { 2 } } ( t + 1 ) ^ { 2 d + 2 } ,
$$

where the last inequality uses $t \geq d .$

Now it is easy to see that

$$
| D | e ^ { - \epsilon ( t - d ) } \leq 2 ^ { 3 d ^ { 2 } } ( t + 1 ) ^ { 2 d + 2 } e ^ { - \epsilon ( t - d ) } < \frac { 1 } { 3 } .
$$

Thus, the tester accepts an ϵ-far input with probability at most $1 / 3$ , and therefore rejects it with probability at least $2 / 3$ . The number of queries is at most

$$
t ^ { 2 } = O \left( \frac { d ^ { 2 } ( d + \log ( 1 / \epsilon ) ) ^ { 2 } } { \epsilon ^ { 2 } } \right) .
$$

## 4 Completions of partial transcripts

We turn to the lower bound. By Lemma 2.2, a one-sided tester cannot reject while its answers still admit a completion of binary rank at most d. Our task is therefore to build low-rank completions from sparse answers. A completion agrees with the input matrix at every queried entry. At unqueried entries, it may difer from the input matrix.

We construct a completion whose one-entries can be partitioned into a small number of rectangles. Fix any subset of h columns, where $0 \leq h \leq m$ . In each chosen column, fill every unqueried entry with zero. The ones in that column then form at most one rectangle, consisting of that column and the rows containing a one.

For the remaining columns, form a graph whose vertices are the rows. Join two rows by an edge if some remaining column contains a queried zero in one row and a queried one in the other. Color the vertices so that the endpoints of every edge have diferent colors.

Consider all rows of one color. In any remaining column, their queried entries have the same value, if any have been queried. Fill the unqueried entries in these rows with that value, or with zero if none has been queried. These rows now become identical on the remaining columns, so their one-entries form at most one rectangle. Rectangles from diferent colors use disjoint sets of rows. They use only columns outside the fixed set of h columns, so they are also disjoint from the rectangles constructed in those h columns. Thus, the completion has binary rank at most h plus the number of colors.

By Diestel [3, Proposition 5.2.1], every graph with e edges can be colored with at most $1 + { \sqrt { 2 e } }$ colors. Applying this bound to the row-conflict graph gives a completion of binary rank at most $h + 1 + \sqrt { 2 e }$ . We give the proof for completeness.

Lemma 4.1 (Column removal and row coloring). Let M be a partial binary matrix with m columns. Fix any subset of h columns, where $0 \leq h \leq m$ . Define the row-conflict graph to have one vertex for each row of M. Two vertices are joined by an edge if some column outside the fixed subset has a specified zero in one of the corresponding rows and a specified one in the other. If this graph has e edges, then M has a completion of binary rank at most

$$
h + 1 + { \sqrt { 2 e } } .
$$

Proof. Let k be the minimum number of colors needed by the row-conflict graph. We show that $k \leq 1 + \sqrt { 2 e }$ . If $k \leq 1$ , the claimed color bound holds. Otherwise, repeatedly remove vertices while the remaining graph still needs k colors. In the resulting minimal graph, every vertex has at least $k - 1$ neighbors. Indeed, take any vertex v and suppose that it has fewer than $k - 1$ neighbors. By the minimality of the graph, removing v leaves a graph that can be colored with $k - 1$ colors. In such a coloring, the neighbors of v use at most $k - 2$ colors. Therefore, at least one of the $k - 1$ colors is not used by any neighbor of v. Assigning that color to v gives a coloring of the entire graph with $k - 1$ colors, contradicting the assumption that it requires k colors.

This graph has at least k vertices, since fewer vertices can be colored with fewer colors. The sum of its vertex degrees is at least $k ( k - 1 )$ . Each edge is counted twice in that sum. The original graph therefore has $e \geq k ( k - 1 ) / 2$ edges. Since $( k - 1 ) ^ { 2 } \leq k ( k - 1 )$ , this gives $k \leq 1 + \sqrt { 2 e }$ □

The following corollary gives suficient bounds on h and e to ensure that this quantity is at most d.

Corollary 4.2. For $d \geq 4 ,$ a transcript is completable in binary rank at most d whenever $h < d / 2$ and $e < d ^ { 2 } / 3 2$ in Lemma $\it 4 . 1$

Proof. The completion rank is less than $d / 2 + 1 + d / 4 \leq d .$

As a first application, the number of specified entries alone already limits how large a rank can be forced. This explains why an exponent $3 / 2$ will appear in the query lower bound.

Corollary 4.3. Every partial binary matrix with $q \geq 1$ specified entries has a completion of binary rank at most $3 q ^ { 2 / 3 } + 1$

Proof. Let $u = \lceil q ^ { 1 / 3 } \rceil$ . Select all columns containing more than u specified entries. Their number is at most $q / u$ . Each other column with k specified entries contributes at most $( \mathit { \Sigma } _ { 2 } ^ { k } ) \le u k / 2$ conflict edges. Thus $e \leq u q / 2$ , and Lemma 4.1 gives a completion of rank at most $q / u + 1 + \sqrt { u q }$ . Here $u \geq q ^ { 1 / 3 }$ and $u \leq 2 q ^ { 1 / 3 }$ for $q \geq 1$ , so the two nonconstant terms are at most $q ^ { 2 / 3 }$ and $\sqrt { 2 } q ^ { 2 / 3 }$ 2 respectively. Their sum is at most $3 q ^ { 2 / 3 }$ □

Corollary 4.3 implies that every partial binary matrix with no completion of binary rank at most d must have $\Omega ( d ^ { 3 / 2 } )$ specified entries. To get a dependence on ϵ, we need more: we place random entries inside a small, randomly chosen submatrix and set all other entries to zero. We then show that a tester making too few queries is unlikely to query enough entries inside this submatrix to rule out a completion of binary rank at most $d .$

## 5 A general nonadaptive lower bound

## 5.1 Far matrices in a hidden rectangle

The hard inputs have random bits in a small rectangle and zeros elsewhere. The rectangle must be large enough that these bits cannot be repaired by changing only an ϵ fraction of the whole matrix. At the same time, we want to vary its shape while keeping its expected area fixed.

Fix $0 < \epsilon \leq 1 / 1 6$ , put $\rho = 8 \epsilon$ , and take $\alpha , \beta \in [ \rho , 1 ]$ with $\alpha \beta = \rho .$ On an $N \times N$ matrix, select rows independently with probability α and columns independently with probability $\beta .$ . Call the resulting rectangle an active rectangle. Inside the active rectangle put independent fair bits. Set all other entries to zero. Denote this distribution by $\mathcal { D } _ { \alpha , \beta }$ . An “active” row or column simply means one selected by this random experiment. For example, if $\rho = 1 / 8$ , we may use $( \alpha , \beta ) = ( 1 , 1 / 8 )$ $( 1 / 2 , 1 / 4 ) , \mathrm { o r } \ ( 1 / 8 , 1 )$ . Each has the same expected active area $N ^ { 2 } / 8$ , but a diferent shape.

Why should the resulting matrix be far from small binary rank? A fixed candidate matrix of binary rank at most d is unlikely to match many independent fair bits. There are many candidate matrices, but their number grows like $2 ^ { 2 N d }$ , while the number of random bits grows like $\epsilon N ^ { 2 }$ . Taking N large enough makes the latter dominate.

Lemma 5.1. Let $\alpha , \beta \in ( 0 , 1 ]$ satisfy $\alpha \beta = \rho : = 8 \epsilon$ $I f d \geq 1$ and $N \geq 1 0 2 4 d / \epsilon$ , a matrix from $\mathcal { D } _ { \alpha , \beta }$ is ϵ-far from $B _ { d } ^ { N , N }$ with probability at least 0.99.

Proof. We first show that the active rectangle is large. We then bound the chance of being close to one fixed candidate and sum over all candidates.

Let $a , b$ be the numbers of active rows and columns. Their expectations are $\alpha N$ and $\beta N$ Applying Lemma 2.3 with $\delta = 1 / 4$ to each count and using the union bound gives

$$
\operatorname* { P r } [ a < 3 \alpha N / 4 \mathrm { ~ o r ~ } b < 3 \beta N / 4 ] \le 2 e ^ { - \rho N / 3 2 } .
$$

Outside this event, the active area $A = a b$ is at least $9 \rho N ^ { 2 } / 1 6 = ( 9 / 2 ) \epsilon N ^ { 2 }$

Condition on these active sets. For any fixed binary matrix H, each of the A random bits disagrees with H independently with probability $1 / 2$ . Thus the number of disagreements has mean $A / 2$ . Since $\epsilon N ^ { 2 } \leq A / 4$ , Lemma 2.3 with $\delta = 1 / 2$ bounds the probability of at most $\epsilon N ^ { 2 }$ disagreements by $e ^ { - A / 1 6 } \leq$ exp $\left( - 9 \rho N ^ { 2 } / 2 5 6 \right)$ . Disagreements outside the active rectangle can only increase the distance, so ignoring them is safe.

There are at most $2 ^ { 2 N d }$ matrices in $B _ { d } ^ { N , N }$ . A union bound yields failure probability at most

$$
2 e ^ { - \rho N / 3 2 } + \exp \biggl ( 2 N d \log 2 - \frac { 9 \rho N ^ { 2 } } { 2 5 6 } \biggr ) < 0 . 0 1
$$

under the stated lower bound on $N$

## 5.2 The contribution of one queried column

Consider a tester whose queries have already been chosen. A column queried many times might reveal many active rows; we pay for that column by giving it its own rectangle. In a column queried few times, we instead count pairs of active queried rows that could create conflict edges. The threshold below balances these two ways of completing the answers.

Fix a deterministic nonadaptive query that makes at most $q$ queries and let $Q$ be its set of queried positions. Let $k _ { j }$ count its distinct queried positions in column $j$ . Thus $\textstyle \sum _ { j } k _ { j } = | Q | \leq q$ Call column $j$ heavy at scale $\alpha$ when $\alpha k _ { j } \geq { \sqrt { d } } .$ Here $\alpha k _ { j }$ is its expected number of queried active rows. The label “heavy” depends only on the fixed query set and on $\alpha ,$ not on the random active sets or their bits.

Lemma 5.2. For $d \geq 4$ , the rejection probability of any one-sided tester using $Q$ on $\mathcal { D } _ { \alpha , \beta }$ is at most

$$
{ \frac { 1 6 \rho } { d ^ { 3 / 2 } } } \sum _ { j : k _ { j } > 0 } k _ { j } f \left( { \frac { \alpha k _ { j } } { \sqrt { d } } } \right) , \qquad f ( x ) = \operatorname* { m i n } \{ x , x ^ { - 1 } \} .\tag{4}
$$

Proof. We use Corollary 4.2: unless there are many active heavy columns or many conflict edges elsewhere, the transcript has a low-rank completion. We bound the expectations of these two quantities and then apply Markov’s inequality.

Restrict the observed partial matrix to the active rows and active columns. Handle its heavy columns separately, and let h be their number. Then

$$
\mathbb { E } [ h ] = \beta | \{ j : \alpha k _ { j } \geq \sqrt { d } \} | .
$$

The formula holds because each heavy column is active with probability $\beta .$ . Let $e$ be the number of conflict edges on the remaining columns. A pair of queried rows is simultaneously active with probability $\alpha ^ { 2 }$ . Its column is active independently with probability $\beta .$ There are at most $k _ { j } ^ { 2 } / 2$ pairs in column $j .$ . Counting an edge more than once, and ignoring the further requirement that its two bits be opposite, only increase the bound. Linearity of expectation therefore gives

$$
\mathbb { E } [ e ] \leq \frac { \beta \alpha ^ { 2 } } { 2 } \sum _ { \alpha k _ { j } < \sqrt { d } } k _ { j } ^ { 2 } .
$$

Any completion on the active rectangle can be extended by zeros outside ${ \mathrm { i t } } ,$ matching every outside query without increasing binary rank. By Lemma 2.2 and Corollary 4.2, rejection therefore requires $h \geq d / 2$ or $e \geq d ^ { 2 } / 3 2$ . Markov’s inequality gives

$$
\begin{array} { l } { \displaystyle \operatorname* { P r } [ \mathrm { r e j e c t } ] \leq \frac { 2 \beta } { d } | \{ j : \alpha k _ { j } \geq \sqrt { d } \} | + \frac { 1 6 \beta \alpha ^ { 2 } } { d ^ { 2 } } \sum _ { \alpha k _ { j } < \sqrt { d } } k _ { j } ^ { 2 } } \\ { \leq \frac { 1 6 \beta } { d } \displaystyle \sum _ { j } \operatorname* { m i n } \{ 1 , \alpha ^ { 2 } k _ { j } ^ { 2 } / d \} . } \end{array}
$$

For $k _ { j } > 0$ , put $x = \alpha k _ { j } / \sqrt { d } .$ . Since min $\{ 1 , x ^ { 2 } \} = x f ( x )$ and $\alpha \beta = \rho ,$ each term in the last sum becomes the corresponding term of (4). Columns with $k _ { j } = 0$ contribute zero. □

## 5.3 Summing over aspect ratios

A single shape gives only part of the lower bound. Nonadaptivity now becomes crucial: the tester must use the same distribution of query sets no matter which shape hides the random bits. We use a sequence of shapes in which the row activation probability halves at each step and the column activation probability doubles. Their product, and hence the expected area, stays fixed.

Theorem 5.3. Let $d > 4$ and $0 < \epsilon \leq 1 / 1 6$ . For every $N \geq 1 0 2 4 d / \epsilon _ { \mathrm { i } }$ , any randomized nonadaptive one-sided tester for $B _ { d } ^ { N , N }$ that makes q queries satisfies

$$
q = \Omega \left( \frac { d ^ { 3 / 2 } } { \epsilon } \log \frac { 2 } { \epsilon } \right) .
$$

Proof. We apply the tester to several input distributions, each using a diferent choice of α and $\beta .$ Since each distribution produces an ϵ-far matrix with probability at least 0.99, the tester must reject with probability at least $0 . 9 9 \cdot ( 2 / 3 )$ under each distribution. However, Lemma 5.2 bounds the sum of these rejection probabilities in terms of the number of queries. Comparing the two bounds gives the required lower bound on the number of queries.

Step 1: choose the shapes. Recall that $\rho = 8 \epsilon$ . Set $T = \lfloor \log _ { 2 } ( 1 / \rho ) \rfloor$ and use the $T + 1$ distributions with

$$
\alpha _ { t } = 2 ^ { - t } , \qquad \beta _ { t } = \rho 2 ^ { t } , \qquad 0 \leq t \leq T .
$$

All these parameters lie in $[ \rho , 1 ]$ , and $\alpha _ { t } \beta _ { t } = \rho ,$ so Lemmas 5.1 and 5.2 apply.

Step 2: each column contributes little in total. Recall that $f ( x ) = \operatorname* { m i n } \{ x , x ^ { - 1 } \}$ . For any $x > 0$ , the sum $\scriptstyle \sum _ { t = 0 } ^ { T } f ( 2 ^ { - t } x )$ is at most 4. To verify this, begin with the first argument at most one. Its value is at most one, and subsequent values halve, so their sum is at most $1 + 1 / 2 + 1 / 4 + \cdot \cdot = 2$ For the arguments greater than one, read the sequence in the reverse order. Their reciprocals again start below one and halve, so their sum is at most two. Either part may be empty.

Summing Lemma 5.2 over these distributions for a fixed query set gives

$$
\sum _ { t = 0 } ^ { T } \operatorname* { P r } _ { \mathcal { D } _ { \alpha _ { t } , \beta _ { t } } } [ \operatorname { r e j e c t } ] \leq \frac { 1 6 \rho } { d ^ { 3 / 2 } } \sum _ { j : k _ { j } > 0 } k _ { j } \sum _ { t = 0 } ^ { T } f \left( \frac { 2 ^ { - t } k _ { j } } { \sqrt { d } } \right) \leq \frac { 6 4 \rho q } { d ^ { 3 / 2 } } .
$$

For a randomized tester, this estimate is first applied to every fixed random seed, using Lemma 2.2, and then averaged over seeds. For each seed, nonadaptivity fixes the same query set for every shape, and the worst-case budget guarantees $| Q | \le q$

Step 3: soundness forces many queries. By Lemma 5.1 and soundness, the tester rejects with probability at least $0 . 9 9 ( 2 / 3 ) > 3 / 5$ under each of these unconditioned distributions. Consequently,

$$
\frac { 3 } { 5 } ( T + 1 ) \leq \frac { 6 4 \rho q } { d ^ { 3 / 2 } } .
$$

It remains to compare the number of shapes with the desired logarithm. Write $u = \log _ { 2 } ( 1 / \rho ) \geq 1$ Then $T + 1 > u$ , and log $( 2 / \epsilon ) = ( u + 4 ) \log 2 \leq 5 u \log 2$ . Thus $T + 1$ is at least a constant times $\log ( 2 / \epsilon )$ . Substituting $\rho = 8 \epsilon$ into the preceding inequality proves the stated lower bound. □

## 6 Comparison and remaining questions

Table 2 summarizes the general bounds. The lower bound is a worst-case entry-query bound for suficiently large matrices.

For constant $\epsilon ,$ the bounds leave a gap between $\Omega ( d ^ { 3 / 2 } )$ and $O ( d ^ { 4 } )$ . Three concrete questions remain. First, can one strengthen the completion argument or choose a diferent far distribution to increase the exponent $3 / 2$ of $d \mathrm { ? }$ Second, can the reconstruction argument reduce the number of sampled rows and columns, or eliminate the basis-location logarithm? Third, can a sparse nonadaptive query pattern replace the full sampled grid while still supplying a one-sided rejection certificate?

<table><tr><td>Result Query complexity</td></tr><tr><td>Upper bound (Theorem 3.4)  $O ( d ^ { 2 } ( d + \log ( 1 / \epsilon ) ) ^ { 2 } / \epsilon ^ { 2 } ) = \tilde { O } ( d ^ { 4 } / \epsilon ^ { 2 } )$ </td></tr><tr><td>Lower bound (Theorem 5.3)  $\Omega ( d ^ { 3 / 2 } \log ( 1 / \epsilon ) / \epsilon )$   $\tilde { \Omega } ( d ^ { 3 / 2 } / \epsilon )$ </td></tr></table>

Table 2: General nonadaptive one-sided testing of binary rank at most d.

The two arguments address complementary obstacles. The upper bound limits how many lowrank explanations a sample can have. The lower bound shows how often sparse answers still admit such an explanation. Closing the gap requires improving at least one of these steps. Throughout, the property is binary rank at most d.

## References

[1] Noga Alon, Eldar Fischer, and Ilan Newman. Eficient testing of bipartite graphs for forbidden induced subgraphs. SIAM Journal on Computing, 37(3):959–976, 2007. https://doi.org/10. 1137/050627915.

[2] Nader H. Bshouty. On property testing of the binary rank. In 48th International Symposium on Mathematical Foundations of Computer Science (MFCS 2023), volume 272 of Leibniz International Proceedings in Informatics (LIPIcs), pages 27:1–27:14. Schloss Dagstuhl–Leibniz-Zentrum für Informatik, 2023. https://doi.org/10.4230/LIPIcs.MFCS.2023.27.

[3] Reinhard Diestel. Graph Theory, volume 173 of Graduate Texts in Mathematics. Springer, 5 edition, 2017.

[4] Yonatan Nakar and Dana Ron. On the testability of graph partition properties. In Approximation, Randomization, and Combinatorial Optimization. Algorithms and Techniques (APPROX/RANDOM 2018), volume 116 of Leibniz International Proceedings in Informatics (LIPIcs), pages 53:1–53:13. Schloss Dagstuhl–Leibniz-Zentrum für Informatik, 2018. https: //doi.org/10.4230/LIPIcs.APPROX-RANDOM.2018.53.

[5] Michal Parnas. Mathematical and computational perspectives on the Boolean and binary rank and their relation to the real rank. arXiv:2601.13900, 2026. https://arxiv.org/abs/2601.13900.

[6] Michal Parnas. Testing the binary rank with polynomial query complexity. arXiv:2609.10496, version 1, 2026. September 9, 2026. https://arxiv.org/abs/2609.10496.

[7] Michal Parnas, Dana Ron, and Adi Shraibman. Property testing of the boolean and binary rank. Theory Comput. Syst., 65(8):1193–1210, 2021.