# Linear Independence of Polynomial Compositions and Identifiability of Deep Neural Networks

Kathlén Kohn KTH Stockholm Digital Futures kathlen@kth.se

Giovanni Luca Marchetti AI4S AB Stockholm giovanni.marchetti@aiforscience.se

Alex Massarenti University of Ferrara msslxa@unife.it

Massimiliano Mella University of Ferrara mll@unife.it

## Abstract

Motivated by theoretical problems in deep learning, we conjecture that postcomposing a fixed number of pairwise distinct nonconstant polynomials with a generic polynomial of sufficiently large degree yields linearly independent polynomials. This generalizes Newman–Slater’s theorem on powers of polynomials. We establish several cases of this conjecture and its origin-passing variant: We prove the result for two polynomials, and for an arbitrary number of polynomials when their degrees are bounded. Furthermore, we show how the conjecture implies a complete understanding of the identifiability (i.e., parameter symmetries) of deep fully connected neural network architectures with generic polynomial activation functions. In particular, for network architectures with layer-specific activations of increasing degree, our established versions of the conjecture fully characterize the set of parameters yielding the same end-to-end network function. As a special case, we fully resolve the identifiability of shallow polynomial networks.

## 1 Introduction

One of the most fundamental questions for parametrized machine learning models is identifiability: Given a function parametrized by the model, which parameter choices yield that function? We investigate this question for the most classical deep learning architecture: multilayer perceptrons (MLPs). After fixing a so-called activation $\sigma \colon  { \mathbb { R } } \to  { \mathbb { R } }$ , such a model is parametrized by tuples of affine linear maps (of compatible dimensions): each tuple $( \alpha _ { 1 } , \dots , \alpha _ { L } )$ yields a function by alternating the affine linear maps with the activation as $\alpha _ { L } \circ \sigma \circ \cdots \circ \sigma \circ \alpha _ { 2 } \circ \sigma \circ \alpha _ { 1 }$ , where σ is applied entrywise to vectors. We focus on MLPs without bias vectors, which means that each $\alpha _ { i }$ is a linear map.

The identifiability question for MLPs has been answered for various choices of activations, e.g., for sigmoidal activations [3, 21] or hyperbolic tangent [17, 20]. For ReLU activations, identifiability is considered to be a challenging problem with only partial results available [12, 13, 6, 7, 5].

The recently established subfield of deep learning theory deemed neuroalgebraic geometry [9] proposes to investigate polynomial machine learning models. This is motivated by the fact that, on the one hand, the study of polynomial models is tractable via the rich toolbox of algebraic geometry and, on the other hand, the approximation capabilities of polynomials suggest that polynomial models can reflect the generally expected behavior of machine learning models. For MLPs with polynomial activation σ, the identifiability question is (essentially) resolved when σ is either linear (see, e.g., [18]) or a monomial of large degree [4, 2, 19, 10]. For the agenda of neuroalgebraic geometry to use polynomial models as approximators of arbitrary models, it would be most interesting to establish identifiability results when σ is a generic polynomial of large degree. However, the only known result in this setting is that, for a generic function parametrized by the model, there are finitely many parameter choices [16, Theorem 4.1]. Resolving the identifiability question would entail determining a precise condition for the term “generic” in the previous statement and identifying exactly which parameter choices occur in those finite fibers. We address this problem in this article.

We wish to point out that the identifiability of other $( \mathrm { i . e . }$ , non-MLP) polynomial machine learning models is known, namely for convolutional neural networks with a monomial activation of large degree [14] or a generic polynomial activation [16] or for single-head attention networks [8].

For MLPs with a generic polynomial activation of large degree, the identifiability question leads us to an open problem in commutative algebra. To motivate that problem, we recall the following known result due to Newman and Slater, which holds over any field K of characteristic zero:

Theorem 1.1 ([11]). Let $p _ { 1 } , \dotsc , p _ { k } , \ k \ \geq \ 2 $ , be multivariate polynomials that are nonzero and pairwise nonproportional. $H r > k ( k - 2 )$ , then $p _ { 1 } ^ { r } , \ldots , p _ { k } ^ { r }$ are linearly independent.

For $k \ = \ 3$ , that result can be viewed as a polynomial version of Fermat’s Last Theorem. The Newman–Slater result was the main ingredient in resolving identifiability for monomial MLPs [2, 4]. Intuitively, it says that plugging polynomials into a monomial activation of large degree separates the polynomials enough so that they become linearly independent. We conjecture that not only monomial activations behave like that, but almost all polynomial activations:

Conjecture 1.2. Let $k \geq 2 .$ . There is an integer $R ( k )$ such that, for every $r \geq R ( k )$ , there is a nonempty Zariski open subset $U _ { k , r } \subseteq \mathbb { K } [ z ] _ { \leq r }$ with the following property: For every $\sigma \in U _ { k , r }$ and all multivariate polynomials $p _ { 1 } , \ldots , p _ { k }$ that are nonconstant and pairwise distinct, the polynomials $\sigma ( p _ { 1 } ) , \ldots , \sigma ( p _ { k } )$ are linearly independent.

For our purposes of investigating polynomial MLPs, we need Conjecture 1.2 in its origin-passing variant, which is the verbatim statement as above but only considering polynomials σ and $p _ { j }$ that satisfy $\sigma ( 0 ) = 0$ and $p _ { j } ( 0 ) = 0$ for all $j = 1 , \dots , k$ (see Conjecture 3.1). Proving the origin-passing variant of the conjecture would resolve the identifiability question for MLPs with generic polynomial activations of large degree, as we explain in Section 4. In fact, MLP identifiability already follows from the weaker Conjecture 3.2.

We prove two restricted versions of Conjecture 1.2 and its origin-passing variant: First, we provide proofs for the case $k = 2$ (Propositions 2.1 and 3.3), which resolves identifiability for deep MLPs of width one (see Section 4). Second, we prove Conjecture 1.2 and its origin-passing variant when the bound R depends not only on the number k of polynomials but also on their maximum degree (Theorems 2.3 and 2.5 for polynomials with constant term and Theorems 3.4 and 3.5 for origin-passing polynomials). Note also that the original Newman–Slater result is independent of the degrees of the polynomials $p _ { 1 } , \ldots , p _ { k }$ . Our degree-dependent result solves the identifiability problem for shallow MLPs and for deep MLPs where the degrees of the generic polynomial activations grow from layer to layer (see Section 4).

Furthermore, Examples $2 . 7 , 3 . 6 ,$ and 3.7 give explicit families of fixed polynomial activations σ that turn every collection of nonconstant, pairwise distinct (and origin-passing) polynomials $p _ { 1 } , \ldots , p _ { k }$ into linearly independent $\sigma ( p _ { 1 } ) , \dots , \sigma ( p _ { k } )$ ). For a deep MLP with an origin-passing activation from such a family, identifiability holds without any growing-degree assumptions.

## 2 Linear independence of polynomial compositions

Let K be a field of characteristic zero, $k \in \mathbb { Z } _ { > 2 }$ , and $r \in \mathbb { Z } _ { > 0 }$ . Given a polynomial $\sigma \in \mathbb { K } [ z ] _ { \leq r }$ and polynomials $p _ { 1 } , \ldots , p _ { k } \in \mathbb { K } [ x _ { 1 } , \ldots , x _ { n } ]$ , we are interested in linear relations among the compositions $\sigma ( p _ { 1 } ) , \ldots , \sigma ( p _ { k } )$ . We first observe that, for all conjectures and results proved in this article, it is sufficient to consider univariate polynomials.

## 2.1 Reduction to one variable

Let $p _ { 1 } , \ldots , p _ { k }$ be pairwise distinct polynomials in $\mathbb { K } [ x _ { 1 } , \ldots , x _ { n } ]$ , and assume that there is a nontrivial linear combination

$$
\sum _ { i = 1 } ^ { k } c _ { i } \sigma ( p _ { i } ( x _ { 1 } , \dots , x _ { n } ) ) = 0
$$

in $\mathbb { K } [ x _ { 1 } , \ldots , x _ { n } ]$ . Composing with a general linear map $\ell : \mathbb { A } ^ { 1 } \to \mathbb { A } ^ { n }$ preserves the degrees of the $p _ { i }$ , keeps them pairwise distinct, and gives the relation

$$
\sum _ { i = 1 } ^ { k } c _ { i } \sigma ( p _ { i } ( \ell ( t ) ) ) = 0
$$

in $\mathbb { K } [ t ]$ . Moreover, if the $p _ { i }$ are origin-passing $( { \mathrm { i . e . , ~ } } p _ { i } ( 0 ) = 0 )$ , then so is the composition $p _ { i } \circ \ell .$ Henceforth, we assume $p _ { 1 } , \dotsc , p _ { k } \in \mathbb { K } [ t ]$

## 2.2 Two polynomials

The special case of two polynomials $( \mathrm { i } . \mathsf { e } . , k = 2 )$ can be easily settled. We now show that $R ( 2 ) = 3$ in Conjecture 1.2 and provide an explicit open set of activation polynomials.

Proposition 2.1. Let $r \geq 3$ and write $\sigma ( z ) = a _ { 0 } + a _ { 1 } z + \cdot \cdot \cdot + a _ { r } z ^ { r }$ . Set $\Delta _ { 2 } ( \sigma ) = 2 r a _ { r } a _ { r - 2 } -$ $( r - 1 ) a _ { r - 1 } ^ { 2 }$ and

$$
\Delta _ { 3 } ( \sigma ) = 3 r ^ { 2 } a _ { r } ^ { 2 } a _ { r - 3 } - 3 r ( r - 2 ) a _ { r } a _ { r - 1 } a _ { r - 2 } + ( r - 1 ) ( r - 2 ) a _ { r - 1 } ^ { 3 } .
$$

Then $U _ { r } = \{ \sigma \in \mathbb { K } [ z ] _ { < r } : a _ { r } \Delta _ { 2 } ( \sigma ) \Delta _ { 3 } ( \sigma ) \neq 0 \}$ is a nonempty Zariski open subset of $\mathbb { K } [ z ] _ { \leq r }$ such that, for every $\sigma \in U _ { r }$ and every pair of distinct nonconstant polynomials $p , q \in \mathbb { K } [ t ]$ , the polynomials $\sigma ( p )$ and $\sigma ( q )$ are linearly independent.

Proof. The set $U _ { r }$ is Zariski open and nonempty since $z ^ { r } + z ^ { r - 2 } + z ^ { r - 3 } \in U _ { r }$ . Let $\sigma \in U _ { r }$ <sub>r</sub> and assume that $s \sigma ( p ) = \sigma ( q )$ for some nonconstant $p , q \in \mathbb { K } [ t ]$ and $s \in \mathbb { K } ^ { * }$ . Comparing degrees gives deg $p = \deg q = : \delta . \operatorname { I f } b \in \mathbb { K } ^ { * }$ is the ratio of the leading coefficient of q to the leading coefficient of $p ,$ then comparison of leading coefficients in the equality $s \sigma ( p ) = \sigma ( q )$ gives $s = b ^ { r }$

Set $c : = q - b p$ . If c is nonconstant, then deg $c < \delta$ and $q ^ { r } - b ^ { r } p ^ { r }$ has degree $( r - 1 ) \delta + \deg c .$ , while every term of $\sigma ( q ) - b ^ { r } \sigma ( p )$ coming from powers smaller than r has degree at most $( r - 1 ) \delta$ . This contradicts $\sigma ( q ) = b ^ { r } \sigma ( p )$ . Hence, $c \in \mathbb { K }$

Since p is nonconstant, the substitution map $\mathbb { K } [ z ] \to \mathbb { K } [ t ] , f \mapsto f ( p )$ , is injective.Therefore, $\sigma ( b p +$ $c ) = \sigma ( q ) = b ^ { r } \sigma ( p ) $ implies $\sigma ( b z + c ) = \bar { b ^ { r } } \sigma ( z )$ Comparing the coefficients of $z ^ { r - 1 }$ gives $c = ( b - 1 ) a _ { r - 1 } / ( r a _ { r } )$ . Set $\alpha : = - a _ { r - 1 } / ( r a _ { r } )$ and $\widetilde { \sigma } ( z ) : = \sigma ( z + \alpha )$ Then, $c = ( 1 - b ) \alpha$ and $\widetilde \sigma ( b z ) ~ = ~ b ^ { r } \widetilde \sigma ( z )$ The coefficient of $z ^ { r - 1 }$ in $\widetilde { \sigma }$ is zero, while those of $z ^ { r - 2 }$ and $\overset { \cdot } { z } ^ { r - 3 }$ are $\Delta _ { 2 } ( \sigma ) \dot { / } ( 2 r a _ { r } )$ and $\Delta _ { 3 } ( \sigma ) / ( 3 r ^ { 2 } a _ { r } ^ { 2 } )$ , respectively. They are nonzero since $\sigma \in U _ { r }$ . Comparing the coefficients of $z ^ { r - 2 }$ and $z ^ { r - 3 }$ in $\begin{array} { r } { \dot { \widetilde { \sigma } } ( b z ) = b ^ { r } \widetilde { \sigma } ( z ) } \end{array}$ gives $b ^ { r - 2 } = b ^ { r }$ and $b ^ { r - 3 } = b ^ { r }$ , i.e., $b ^ { 2 } = b ^ { 3 } = 1$ Thus, $b = 1$ , which implies that $c = 0$ and $p = q$ □

## 2.3 Polynomials of equal degree

Next, we prove a version of Conjecture 1.2 where the lower bound R depends not only on the number k of polynomials but also on their degree. We begin with the case where the polynomials $p _ { 1 } , \ldots , p _ { k }$ have the same degree. This is the natural first setting for the problem since the coefficients of degree m polynomials form an affine space and the condition that $\sigma ( p _ { 1 } ) , \ldots , \sigma ( p _ { k } )$ are linearly dependent is closed in the corresponding parameter space. We will prove that, after fixing k and $m ,$ a general polynomial σ of sufficiently large degree separates every ordered k-tuple of pairwise distinct degree m polynomials in the sense that the compositions $\sigma ( p _ { 1 } ) , \ldots , \sigma ( p _ { k } )$ are linearly independent.

Lemma 2.2. Let $p _ { 1 } , \dotsc , p _ { k } \in \mathbb { K } [ t ]$ be pairwise distinct polynomials ofthe same degree m $\geq 1$ and let $( \lambda _ { 1 } , \cdots , \lambda _ { k } ) \in \mathbb { K } ^ { k } \setminus \left\{ ( 0 , \dots , 0 ) \right\}$ . Consider the K-linear map

$$
T _ { p , \lambda } : \mathbb { K } [ z ] _ { \leq r } \longrightarrow \mathbb { K } [ t ] _ { \leq r m } , \qquad \sigma \longmapsto \sum _ { i = 1 } ^ { k } \lambda _ { i } \sigma ( p _ { i } ( t ) ) .\tag{1}
$$

Then, rank $T _ { p , \lambda } \geq \lfloor ( r + 1 ) / k \rfloor$ and

$$
\mathrm { d i m } \mathrm { k e r } T _ { p , \lambda } \leq ( r + 1 ) - \left\lfloor { \frac { r + 1 } { k } } \right\rfloor .
$$

Proof. Set $q = t ^ { - 1 }$ . Since each $p _ { i }$ has degree m, we may write $p _ { i } ( t ) = t ^ { m } u _ { i } ( q )$ , where $u _ { i } ( q ) =$ $\alpha _ { i } + u _ { i , 1 } q + \cdot \cdot \cdot + u _ { i , m } q ^ { m } \in \mathbb { K } [ q ]$ and $\alpha _ { i } \in \mathbb { K } ^ { * }$ . Moreover, the power series u<sub>1</sub> $( q ) , \ldots , u _ { k } ( q )$ are pairwise distinct.

For every $h \geq 0 .$ , set $\begin{array} { r } { Q _ { h } ( t ) \ : = \ \sum _ { i } \lambda _ { i } p _ { i } ( t ) ^ { h } } \end{array}$ . Then, $Q _ { h } ( t ) ~ = ~ t ^ { m h } F _ { h } ( q )$ , where $F _ { h } ( q ) \ =$ $\Sigma _ { i } \lambda _ { i } u _ { i } ( q ) ^ { h }$ . Write $\begin{array} { r } { F _ { h } ( q ) = \sum _ { \nu > 0 } a _ { \nu } ( h ) q ^ { \nu } } \end{array}$ . If all sequences $a _ { \nu } ( h )$ were zero, then $F _ { h } ( q ) = 0$ for every $h \geq 0$ . In particular, ${ \cal F } _ { 0 } ( \bar { q } ) = \dots = { \cal F } _ { k - 1 } ( q ) = 0$ . The latter system of equations can be written as $V ( q ) ( \lambda _ { 1 } , \ldots , \lambda _ { k } ) ^ { \top } = 0$ , where

$$
V ( q ) : = \left( u _ { i } ( q ) ^ { h } \right) _ { 0 \leq h \leq k - 1 , 1 \leq i \leq k } .
$$

This Vandermonde matrix has determinant $\begin{array} { r } { \prod _ { i < j } ( u _ { j } ( q ) \ - \ u _ { i } ( q ) ) \ \ne 0 } \end{array}$ in $\mathbb { K } [ [ q ] ]$ Hence, $V ( q ) ( \lambda _ { 1 } , \ldots , \lambda _ { k } ) ^ { \top } = 0$ forces $\lambda _ { 1 } = \cdot \cdot \cdot = \lambda _ { k } = 0 .$ , a contradiction.

Let $\nu _ { 0 }$ be the smallest integer such that $a _ { \nu _ { 0 } } ( h )$ is not the zero sequence. The sequence $F _ { h } ( q )$ is annihilated by $\begin{array} { r } { \prod _ { i } ( E - u _ { i } ( \overline { { q } } ) ) } \end{array}$ , where E is the shift operator in h. Thus, there are $c _ { 0 } ( q ) , \ldots , c _ { k } ( q ) \in$ $\mathbb { K } [ [ q ] ]$ , with $c _ { k } ( q ) = 1$ , such that

$$
\sum _ { j = 0 } ^ { k } c _ { j } ( q ) F _ { h + j } ( q ) = 0\tag{2}
$$

for every $h \geq 0$ . Looking at the coefficient of $q ^ { \nu _ { 0 } }$ in $( 2 )$ , we see that all lower order terms vanish by the minimality of $\nu _ { 0 }$ and, setting $a ( h ) : = a _ { \nu _ { 0 } } ( h )$ , that

$$
\sum _ { j = 0 } ^ { k } c _ { j } ( 0 ) a ( h + j ) = 0 .\tag{3}
$$

The coefficients $c _ { j } ( 0 )$ are those of $\textstyle \prod _ { i } ( X - \alpha _ { i } )$ . In particular, $c _ { 0 } ( 0 )$ and $c _ { k } ( 0 )$ are nonzero. Therefore, the nonzero sequence $a ( h )$ cannot have k consecutive zero values. (Indeed, if $a ( n ) =$ $a ( n + 1 ) = \ldots = a ( n + k - 1 )$ for some n, applying (3) for $h = n$ yields $c _ { k } ( 0 ) a ( n + k ) = 0 .$ , i.e., $a ( n + k ) = 0$ . Then, applying (3) for $h = n + 1 , n + 2 , . . . _ $ , shows that $a ( h ) = 0$ for all $h \geq n .$ Similarly, applying (3) for ${ \bar { h = n } } - 1 { \mathrm { ~ g i v e s ~ } } c _ { 0 } ( 0 ) a ( n - 1 ) = 0 , { \mathrm { i . e . , } } a ( n - 1 ) = 0$ . Continuing this for $h = n - 2 , n - 3 , . .$ . shows that the whole sequence ${ \dot { a } } ( h )$ must be zero, a contradiction).

It follows that in every block of k consecutive integers, there is an h with $a _ { \nu _ { 0 } } ( h ) \neq 0$ . Hence, among $0 , \ldots , r$ , there are at least $\lfloor ( r + 1 ) / k \rfloor$ such integers. For each of them, deg $Q _ { h } = m h - \nu _ { 0 }$ , and these degrees are distinct. Thus, the corresponding $Q _ { h } = T _ { p , \lambda } ( z ^ { h } )$ are linearly independent. This gives the rank estimate, and the kernel estimate follows from dim $\mathbb { K } [ z ] _ { \leq r } = r + 1$ □

Theorem 2.3. Let $m , k \geq 1$ and assume

$$
r \geq k ^ { 2 } ( m + 2 ) - 1 .
$$

Then, there exists a nonempty Zariski open subset $U _ { k , m , r } \subseteq \mathbb { K } [ z ] _ { \leq r }$ such that, for every $\sigma \in U _ { k , m , r }$ and every k-tuple ofpairwise distinct polynomials $p _ { 1 } , \dotsc , p _ { k } \in \mathbb { K } [ t ]$ of degree m, the polynomials $\sigma ( p _ { 1 } ) , \ldots , \sigma ( p _ { k } )$ are linearly independent in $\mathbb { K } [ t ]$

Proof. Let $\mathcal { P } _ { m } \subseteq \mathbb { K } [ t ] _ { \leq m }$ be the open subset of polynomials of degree precisely m. Furthermore, let $\mathcal { P } _ { k , m } \subseteq \mathcal { P } _ { m } ^ { k }$ be the open subset of pairwise distinct k-tuples of such polynomials. Note that dim $\mathcal { P } _ { k , m } = k ( m + 1 )$ .

Consider the incidence variety

$$
\mathcal { I } _ { k , m , r } : = \left\{ ( [ \sigma ] , p _ { 1 } , \dotsc , p _ { k } , [ \lambda ] ) \in \mathbb { P } ( \mathbb { K } [ z ] _ { \leq r } ) \times \mathcal { P } _ { k , m } \times \mathbb { P } ^ { k - 1 } : \sum _ { i = 1 } ^ { k } \lambda _ { i } \sigma ( p _ { i } ( t ) ) = 0 \right\}\tag{4}
$$

and the projection $\pi _ { 2 , 3 } : \mathcal { I } _ { k , m , r } \to \mathcal { P } _ { k , m } \times \mathbb { P } ^ { k - 1 }$ . For fixed $( p , \lambda )$ , the fiber is $\mathbb { P } ( \ker T _ { p , \lambda } )$ , where $T _ { p , \lambda }$ is the map in (1). Set $L : = \lfloor ( r + 1 ) / k \rfloor .$ . By Lemma 2.2, every nonempty fiber has dimension at most $r - L$ . Hence,

$$
\dim \mathcal { I } _ { k , m , r } \leq k ( m + 1 ) + ( k - 1 ) + r - L .\tag{5}
$$

Since $r \geq k ^ { 2 } ( m + 2 ) - 1$ , we have $L \ge k ( m + 2 )$ . Substituting this in (5), we get dim $\mathcal { I } _ { k , m , r } \leq r - 1$ Now, consider the projection $\pi _ { 1 } : \mathcal { I } _ { k , m , r } \to \mathbb { P } ( \mathbb { K } [ z ] _ { \leq r } )$ . Its image variety $C : = \overline { { \pi _ { 1 } ( \mathcal { I } _ { k , m , r } ) } }$ is a proper closed subset of $\mathbb { P } ( \mathbb { K } [ z ] _ { \leq r } ) \cong \mathbb { P } ^ { r }$ . If $\widehat { C } \subseteq \mathbb { K } [ z ] _ { \leq r }$ denotes the affine cone over $C ,$ then $U _ { k , m , r } = \mathbb { K } [ z ] _ { \leq r } \setminus \widehat { C }$ is nonempty and open. By construction, a point $\sigma \in U _ { k , m , r }$ cannot yield a nontrivial relation among $\sigma ( p _ { 1 } ) , \ldots , \sigma ( p _ { k } )$ for any pairwise distinct $p _ { i }$ of degree m. □

## 2.4 Polynomials of bounded degree

We now pass from polynomials of the same fixed degree to polynomials of bounded degree. For fixed k and m, our goal is to show that, for general σ of sufficiently large degree, the compositions $\sigma ( p _ { 1 } ) , \ldots , \sigma ( p _ { k } )$ are linearly independent for every k-tuple of pairwise distinct nonconstant polynomials $p _ { i }$ of degree at most m. By treating simultaneously all possible degenerations in which some leading coefficients of the $p _ { i }$ vanish, the same incidence construction as above gives a closed bad locus of $\sigma .$

Lemma 2.4. Let $p _ { 1 } , \dotsc , p _ { k } \in ~ \mathbb { K } [ t ]$ be pairwise distinct nonconstant polynomials of degree at most m and let $( \lambda _ { 1 } , \cdots , \lambda _ { k } ) \in \mathbb { K } ^ { k } \setminus \{ ( 0 , \dots , 0 ) \}$ . Consider the K-linear map

$$
T _ { p , \lambda } : \mathbb { K } [ z ] _ { \leq r } \longrightarrow \mathbb { K } [ t ] _ { \leq r m } , \qquad \sigma \longmapsto \sum _ { i = 1 } ^ { k } \lambda _ { i } \sigma ( p _ { i } ( t ) ) .\tag{6}
$$

Then,

$$
\operatorname { r a n k } T _ { p , \lambda } \geq \left\lfloor { \frac { r } { k } } - { \frac { m ( k - 1 ) } { 2 } } \right\rfloor .\tag{7}
$$

Consequently,

$$
\mathrm { d i m } \mathrm { k e r } T _ { p , \lambda } \leq ( r + 1 ) - \left\lfloor \frac { r } { k } - \frac { m ( k - 1 ) } { 2 } \right\rfloor .\tag{8}
$$

Proof. Let ${ \cal { S } } : = \{ i : \lambda _ { i } \neq 0 \}$ and $D { : = } \operatorname* { m a x } _ { i \in S } \deg p _ { i } \geq 1$ . Further, let $I : = \{ i \in S : \deg p _ { i } = D \}$ and set $s : = | I | \geq 1$ . For $i \in I ,$ , write $p _ { i } ( t ) = t ^ { D } u _ { i } ( q )$ , where $q : = t ^ { - 1 }$ and $u _ { i } ( q ) \in \mathbb { K } [ q ]$ with $u _ { i } ( 0 ) \neq 0$ . Note that the $u _ { i } ( \boldsymbol { q } )$ , for $i \in I ,$ , are pairwise distinct.

For every $h \geq 0 ,$ , consider

$$
Q _ { h } ^ { \mathrm { t o p } } ( t ) : = \sum _ { i \in I } \lambda _ { i } p _ { i } ( t ) ^ { h } = t ^ { D h } F _ { h } ( q ) , \quad \mathrm { ~ w h e r e ~ } F _ { h } ( q ) : = \sum _ { i \in I } \lambda _ { i } u _ { i } ( q ) ^ { h } .
$$

Write $\begin{array} { r } { F _ { h } ( q ) = \sum _ { \nu > 0 } a _ { \nu } ( h ) q ^ { \nu } } \end{array}$ . As in the proof of Lemma 2.2, there is a smallest $\nu _ { 0 }$ such that $a _ { \nu _ { 0 } } ( h )$ is not the zero sequence.

Let

$$
D _ { I } : = \mathrm { o r d } _ { q } \left( \prod _ { i < j , i , j \in I } ( u _ { j } ( q ) - u _ { i } ( q ) ) \right) ,
$$

with $D _ { I } = 0 { \mathrm { i f } } s = 1$ . We claim that $\nu _ { 0 } \leq D _ { I }$ . If $\nu _ { 0 } > D _ { I }$ , then $F _ { h } ( q )$ is zero modulo $q ^ { D _ { I } + 1 }$ for all $h \geq 0$ . In particular, for the Vandermonde matrix

$$
V ( q ) : = \left( u _ { i } ( q ) ^ { h } \right) _ { 0 \leq h \leq s - 1 , i \in I }
$$

and the vector $\lambda _ { I } : = ( \lambda _ { i } ) _ { i \in I }$ , we obtain $V ( q ) \lambda _ { I } ^ { \top } = 0$ modulo $q ^ { D _ { I } + 1 }$ . Multiplying this system by the adjugate matrix of $V ( q )$ gives that det $V ( q ) \lambda _ { I } \ = \ 0$ modulo $q ^ { D _ { I } + 1 }$ . Since $\lambda _ { i } \neq 0$ for all $i \in I .$ , we conclude that det $V ( q )$ is divisible by $\bar { q } ^ { D _ { I } + 1 }$ . However, $\mathrm { o r d } _ { q }$ det $V ( q ) = D _ { I }$ , a contradiction.Therefore,

$$
\nu _ { 0 } \leq D _ { I } \leq D { \binom { s } { 2 } } \leq m { \binom { k } { 2 } } .\tag{9}
$$

As in the proof of Lemma 2.2, the sequence $a _ { \nu _ { 0 } } ( h )$ satisfies a recurrence $\textstyle \sum _ { j = 0 } ^ { s } c _ { j } a _ { \nu _ { 0 } } ( h + j ) = 0$ with $c _ { 0 } \neq 0$ and $c _ { s } \neq 0$ , obtained from $\begin{array} { r } { \prod _ { i \in I } ( E - u _ { i } ( q ) ) } \end{array}$ by taking the coefficient of $q ^ { \nu _ { 0 } }$ . Thus, it cannot have s consecutive zero values. Hence, among $\nu _ { 0 } + 1 , \ldots , r ,$ , there are at least $\lfloor ( r - \nu _ { 0 } ) / s \rfloor$ values of h with $a _ { \nu _ { 0 } } ( h ) \neq 0$ . Moreover, (9) gives

$$
\frac { r - \nu _ { 0 } } { s } \geq \frac { r } { s } - \frac { D ( s - 1 ) } { 2 } \geq \frac { r } { k } - \frac { m ( k - 1 ) } { 2 } .
$$

Therefore, there are at least as many values of h as the quantity in (7) with $a _ { \nu _ { 0 } } ( h ) \neq 0$ and $\nu _ { 0 } <$ $h \leq r$

For h with $a _ { \nu _ { 0 } } ( h ) \neq 0$ , we have that deg $Q _ { h } ^ { \mathrm { t o p } } = D h - \nu _ { 0 }$ . Moreover, deg $\begin{array} { r } { \langle \sum _ { i \in S \backslash I } \lambda _ { i } p _ { i } ( t ) ^ { h } \rangle \leq } \end{array}$ $( D - 1 ) h$ . Whenever $h > \nu _ { 0 }$ , the part of degree $D h - \nu _ { 0 }$ cannot be canceled and so $T _ { p , \lambda } ( z ^ { h } ) =$ $\textstyle \sum _ { i \in S } \lambda _ { i } p _ { i } ( t ) ^ { h }$ has degree $D h - \nu _ { 0 }$ . Therefore, those polynomials $T _ { p , \lambda } ( z ^ { h } )$ have distinct degrees and are linearly independent. This proves (7), and (8) follows from dim $\mathbb { K } [ z ] _ { \leq r } = r + 1$ □

## Theorem 2.5. Let $m , k \geq 1$ and assume

$$
r \geq k \left( k ( m + 1 ) + k + { \frac { m ( k - 1 ) } { 2 } } \right) = { \frac { k } { 2 } } ( 3 k m + 4 k - m ) .\tag{10}
$$

Then, there exists a nonempty Zariski open subset $U _ { k , m , r } \subseteq \mathbb { K } [ z ] _ { \leq 1 }$ <sub>r</sub> such that, for every $\sigma \in U _ { k , m , r }$ and every k-tuple of pairwise distinct nonconstant polynomials $p _ { 1 } , \dotsc , p _ { k } \in ~ \mathbb { K } [ t ]$ of degree at most m, the polynomials $\sigma ( p _ { 1 } ) , \ldots , \sigma ( p _ { k } )$ are linearly independent in $\mathbb { K } [ t ]$

Proof. Let $\mathcal { P } _ { < m } ^ { + } \subseteq \mathbb { K } [ t ] _ { \leq m }$ be the open subset of nonconstant polynomials. Moreover, let $\mathcal { P } _ { k , \leq m } ^ { + } \ \subseteq \ ( \mathcal { P } _ { \leq m } ^ { \mp } ) ^ { k }$ be the open subset of pairwise distinct k-tuples of such polynomials. Note that dim $\mathcal { P } _ { k , < m } ^ { + } = k ( m + 1 )$ .

Consider the incidence variety

$$
\mathfrak { I } _ { k , m , r } : = \left\{ ( [ \sigma ] , p _ { 1 } , \dotsc , p _ { k } , [ \lambda ] ) \in \mathbb { P } ( \mathbb { K } [ z ] _ { \leq r } ) \times \mathcal { P } _ { k , \leq m } ^ { + } \times \mathbb { P } ^ { k - 1 } : \sum _ { i = 1 } ^ { k } \lambda _ { i } \sigma ( p _ { i } ( t ) ) = 0 \right\} .\tag{11}
$$

The fiber over $( p , \lambda )$ under the projection $\pi _ { 2 , 3 } : \mathcal { I } _ { k , m , r } \to \mathcal { P } _ { k , < m } ^ { + } \times \mathbb { P } ^ { k - 1 }$ is $\mathbb { P } ( \ker T _ { p , \lambda } )$ , where $T _ { p , \lambda }$ is the map in (6). Set $L : = \lfloor r / k - m ( k - 1 ) / 2 \rfloor$ . By Lemma $2 . 4 .$ , every nonempty fiber has dimension at most $r - L$ . Hence,

$$
\dim \mathcal { I } _ { k , m , r } \leq k ( m + 1 ) + ( k - 1 ) + r - L .\tag{12}
$$

The assumption (10) gives $L \geq k ( m + 1 ) + k $ . Substituting this in (12), we get dim $\mathcal { I } _ { k , m , r } \leq r - 1$

Now, let $C : = \overline { { \pi _ { 1 } ( \mathcal { I } _ { k , m , r } ) } }$ , where $\pi _ { 1 } : { \mathcal { I } } _ { k , m , r } \to { \mathbb { P } } ( \mathbb { K } [ z ] \leq r )$ is the projection onto the first factor. Then, C is a proper closed subset of $\mathbb { P } ( \mathbb { K } [ z ] _ { \leq r } ) \cong \mathbb { P } ^ { r }$ . If ${ \widehat { C } } \subset { \mathbb { A } } ^ { r + 1 }$ denotes its affine cone, the open set $U _ { k , m , r } = \mathbb { A } ^ { r + 1 } \setminus \widehat { C }$ has the desired property by definition of the incidence variety. □

We wish to point out that it would also be interesting to prove the following weaker version of Conjecture 1.2, where the bound R only depends on the number k of polynomials, but the Zariski open subset U is allowed to depend on their maximum degree m:

Conjecture 2.6 (weak). Let $k \geq 2 .$ . There is an integer $R ( k )$ such that, for every $r \geq R ( k )$ and every m $\geq 1$ , there is a nonempty Zariski open subset $\breve { U } _ { k , r , m } \subseteq \mathbb { K } [ z ] _ { \leq r }$ with thefollowing property: For every $\sigma \in U _ { k , r , m }$ and all $p _ { 1 } , \dotsc , p _ { k } \in \mathbb { K } [ t ]$ of degree at most m that are nonconstant and pairwise distinct, the polynomials $\sigma ( p _ { 1 } ) , \ldots , \sigma ( p _ { k } )$ ) are linearly independent.

In fact, the origin-passing version of this weaker conjecture (see Conj. 3.2) is sufficient to resolve the problem of identifiability for polynomial MLPs, as we will explain in Section 4.

## 2.5 Examples for σ

We close this section with an explicit family of polynomials showing that the difficulty in Conjecture 1.2 is the openness statement rather than the existence of a single activation that works uniformly for all $p _ { i }$

Example 2.7. Let $k \geq 2$ , take any $\ell > k ( k - 2 )$ , and let $f \in U _ { d }$ be any polynomial from Proposition 2.1 for some $d \geq 3$ . Then $\sigma _ { \ell , f } ( z ) : = f ( z ) ^ { \ell }$ has the following property: For every choice of pairwise distinct nonconstant polynomials $p _ { 1 } , \ldots , p _ { k }$ , the polynomials $\sigma _ { \ell , f } ( p _ { 1 } ) , \ldots , \sigma _ { \ell , f } ( p _ { k } )$ are linearly independent. Indeed, Proposition 2.1 implies that $f ( p _ { 1 } ) , \ldots , f ( p _ { k } )$ are pairwise nonproportional, and so (due to $\ell > k ( k - 2 ) ,$ ) the Newman–Slater theorem 1.1 yields the claim.

For instance, both $f ( z ) = z ^ { 3 } + z + 1$ and $f ( z ) = z ^ { 3 } + z ^ { 2 } + . $ z belong to $U _ { 3 } { \mathrm { : } }$ : the corresponding values of $( \Delta _ { 2 } , \Delta _ { 3 } )$ are $( 6 , 2 7 )$ and $( 4 , - 7 )$ , respectively. Thus, $( z ^ { 3 } + z + 1 ) ^ { \bar { k } ( k - 2 ) + 1 }$ and $( z ^ { 3 } + z ^ { 2 } + { \bar { z } } ) ^ { k ( k - 2 ) + 1 }$ are two explicit choices. Note that the second one is origin-passing.

In particular, the polynomial $\sigma _ { \ell , f }$ belongs to $\mathbb { K } [ z ] _ { \leq r }$ for every $r \geq d \ell$ . Thus, Example 2.7 proves the existence part of Conjecture 1.2 after dropping the requirement that the good set be Zariski open.

## 3 Origin-Passing Polynomials

We now focus on the origin-passing analog of Conjecture 1.2 and its weaker version Conjecture 2.6. To this end, we denote by $\dot { \mathbb { K } } [ t ] ^ { 0 } : = \{ p \ \in \mathbb { K } [ t ] : p ( 0 ) = 0 \}$ and $\mathbb { K } [ t ] _ { < r } ^ { 0 } : = \mathbb { K } [ t ] ^ { 0 } \cap \mathbb { K } [ t ] \le t$ <sub>r</sub> the K-vector spaces of origin-passing polynomials (of degree at most r).

Conjecture 3.1 (origin-passing). Let $k \geq 2 .$ . There is an integer $R ( k )$ such that,for every $r \geq R ( k )$ there exists a nonempty Zariski open subset $U _ { k , r } \subseteq \mathbb { K } [ z ] _ { \leq { r } } ^ { 0 }$ with the following property: For every $\sigma \in U _ { k , r }$ and every collection of nonzero, pairwise distinct $p _ { 1 } , \dotsc , p _ { k } \in \mathbb { K } [ t ] ^ { 0 }$ , the polynomials $\sigma ( p _ { 1 } ) , \ldots , \sigma ( p _ { k } )$ are linearly independent.

Conjecture 3.2 (weak, origin-passing). Let $k \geq 2 .$ . There is an integer $R ( k )$ such that, for every $r \geq R ( k )$ and every $m \geq 1$ , there is a nonempty Zariski open subset $\check { U } _ { k , r , m } ~ \subseteq ~ \mathbb { K } [ \check { z } ] _ { < } ^ { 0 }$ with the following property: For every $\sigma \in U _ { k , r , m }$ and all $p _ { 1 } , \ldots , p _ { k } \in \mathbb { K } [ t ] _ { \leq m } ^ { 0 }$ that are nonzero and pairwise distinct, the polynomials $\sigma ( p _ { 1 } ) , \ldots , \sigma ( p _ { k } )$ are linearly independent.

We now prove analogues of the preceding results under the origin-passing assumption, following the same structure as in Section 2.

## 3.1 Two polynomials

Proposition 3.3. Suppose that $r \geq 2 ,$ , and let $\sigma ( z ) = a _ { 1 } z + a _ { 2 } z ^ { 2 } + \cdots + a _ { r } z ^ { r } \in \mathbb { K } [ z ] _ { < } ^ { 0 } ,$ be such that $a _ { r } , a _ { r - 1 } \neq 0 .$ . Then, for every pair of nonzero distinct $p , q \in \mathbb { K } [ t ] ^ { 0 }$ , the polynomials $\sigma ( p )$ and $\sigma ( q )$ are non-proportional.

Proof. Suppose that $s \sigma ( p ) = \sigma ( q )$ for two nonzero polynomials $p , q \in \mathbb { K } [ t ] ^ { 0 }$ and $s \in \mathbb { K } ^ { * }$ . Note that p and q must have equal degree, which we denote by δ. Let $b \in \mathbb { K } ^ { * }$ be the ratio of the highest-degree coefficient of q to that of $p .$ . By comparing the highest-degree coefficients of $s \sigma ( p )$ and $\sigma ( q )$ , we deduce that $s = b ^ { r }$

We wish to show that $q = b p$ . To this end, define $c : = q - b p \in \mathbb { K } [ t ] _ { < \delta } ^ { 0 }$ , and assume for contradiction that deg $c \geq 1$ . Then, $q ^ { r } - ( b p ) ^ { r }$ has degree $( r - 1 ) \delta + \deg c$ , while any other term in $\sigma ( q ) - s \sigma ( p )$ coming from lower powers $p ^ { \ i } , q ^ { \ i }$ , with $i ~ < ~ r ,$ , has degree at most $( r - 1 ) \delta .$ Therefore, the term $q ^ { r } - ( b p ) ^ { r }$ cannot be canceled out, leading to a contradiction with the equality $s \sigma ( p ) = \sigma ( q )$

Thus, c is a constant, and in fact must be 0 since $p$ and $q$ are origin-passing. We have shown that sσ $( p ) = \sigma ( b p )$ . Since p is nonconstant, the substitution map $\mathbb { K } [ z ]  \mathbb { K } [ t ] , \mathbf { \bar { \alpha } } f \mapsto f ( p )$ , is injective. Hence, $s \sigma ( z ) = \sigma ( b z )$ . By comparing the terms of degree $r - 1$ , we conclude that $s = b ^ { r } = b ^ { r - 1 } .$ ， implying that $b = 1$ , i.e., $q = p .$ □

## 3.2 Polynomials of equal degree

Theorem 3.4. Let $m , k \geq 1$ and assume $r \geq k ^ { 2 } ( m + 1 )$ . Then there exists a nonempty Zariski open subset $U _ { k , m , r } \subseteq \mathbb { K } [ z ] _ { < _ { 1 } } ^ { 0 }$ such that, for every $\sigma \in U _ { k , m , r }$ and every k-tuple ofpairwise distinct polynomials $p _ { 1 } , \ldots , p _ { k } ^ { - } \in \mathbb { K } [ t ] ^ { 0 }$ of degree m, the polynomials $\sigma ( p _ { 1 } ) , \ldots , \sigma ( p _ { k } )$ are linearly independent.

Proof. Let $\mathcal { P } _ { m } ^ { 0 } \subseteq \mathbb { K } [ t ] _ { \leq m } ^ { 0 }$ be the open subset of origin-passing polynomials of degree equal to $m _ { : }$ and let $\mathcal { P } _ { k , m } ^ { 0 } \subseteq ( \mathcal { P } _ { m } ^ { 0 } ) ^ { \bar { k } }$ be the set of pairwise distinct k-tuples. Note that dim $\mathcal { P } _ { k , m } ^ { 0 } = k m$ . For $p \in \mathcal { P } _ { k , m } ^ { 0 }$ and $\lambda \in \mathbb { K } ^ { k } \setminus \{ 0 \}$ , restrict the map in (1) to $T _ { p , \lambda } ^ { 0 } : \mathbb { K } [ z ] _ { \leq r } ^ { 0 }  \mathbb { K } [ t ] _ { \leq r m } ^ { 0 } .$ The proof of Lemma 2.2, applied to $h = 1 , \ldots , r$ , gives rank $T _ { p , \lambda } ^ { 0 } \ge \dot { L } : = \lfloor r / \bar { k } \rfloor :$ the sequence $a _ { \nu _ { 0 } } ( h )$ used there cannot vanish at k consecutive values, and the corresponding $T _ { p , \lambda } ^ { 0 } ( z ^ { h } )$ have distinct degrees.

We define the incidence variety $\mathcal { I } _ { k , m , r } ^ { 0 }$ as in (4), with $\mathcal { P } _ { k , m }$ replaced by $\mathcal { P } _ { k , m } ^ { 0 }$ and $\mathbb { P } ( \mathbb { K } [ z ] _ { \leq r } )$ by $\mathbb { P } ( \mathbb { K } [ z ] _ { < r } ^ { 0 } ) \cong \mathbb { P } ^ { r - 1 }$ . The same argument as in the proof of Theorem 2.3 gives

$$
\dim { \mathcal { I } } _ { k , m , r } ^ { 0 } \leq k m + ( k - 1 ) + ( r - L - 1 ) = r + k m + k - L - 2 .
$$

Since $r \geq k ^ { 2 } ( m + 1 )$ , we have $L \ge k ( m + 1 )$ , hence dim $\mathcal { I } _ { k , m , r } ^ { 0 } \leq r - 2 .$ .Therefore, the closure of its image under the projection onto the first factor is a proper subset of $\mathbb { P } ( \mathbb { K } [ z ] _ { < r } ^ { 0 } ) \cong \mathbb { P } ^ { r - 1 }$ , and the complement of the corresponding affine cone is the desired open subset $U _ { k , m , r }$ □

## 3.3 Polynomials of bounded degree

Theorem 3.5. Let $m , k \geq 1$ and assume

$$
r \geq k \left( k ( m + 1 ) + { \frac { ( m - 1 ) ( k - 1 ) } { 2 } } \right) = { \frac { k } { 2 } } ( 3 k m + k - m + 1 ) .
$$

Then, there exists a nonempty Zariski open subset $U _ { k , m , r } \subseteq \mathbb { K } [ z ] _ { \leq i } ^ { 0 }$ such that, for every $\sigma \in U _ { k , m , r }$ and every k-tuple of nonzero, pairwise distinct polynomials $p _ { 1 } , \bar { . . . } , p _ { k } \in \mathbb { K } [ t ] _ { \leq m } ^ { 0 }$ , the polynomials $\sigma ( p _ { 1 } ) , \ldots , \sigma ( p _ { k } )$ are linearly independent.

Proof. Let $\boldsymbol { \mathcal { P } } _ { k , \le m } ^ { 0 , + }$ be the open subset of $( \mathbb { K } [ t ] _ { < m } ^ { 0 } \setminus \{ 0 \} ) ^ { k }$ consisting of pairwise distinct k-tuples. Note that dim $\bar { \mathcal { P } } _ { k , < m } ^ { 0 , + } = k m$ . For $p \in \mathcal { P } _ { k , \leq m } ^ { 0 , + }$ and $\lambda \in \mathbb { K } ^ { k } , \lambda \neq 0$ , restrict the map in (6) to $T _ { p , \lambda } ^ { 0 } : \mathbb { K } [ z ] _ { \leq r } ^ { 0 }  \bar { \mathbb { K } } [ t ] _ { \leq r m } ^ { 0 } .$

In the following, we use the notation of the proof of Lemma 2.4. In particular, D is the maximal degree among the $p _ { i }$ with $\lambda _ { i } \neq 0$ , and we write $p _ { i } ( t ) = t ^ { D } u _ { i } ( t ^ { - 1 } )$ for the $p _ { i }$ of degree D. Since $p _ { i } ( 0 ) = 0$ , this time we have that deg $u _ { i } \leq D - 1 $ . Thus, (9) improves to $\nu _ { 0 } \leq ( D - 1 ) { \binom { s } { 2 } } \leq$ $( m - 1 ) \binom { k } { 2 }$ . Applying the same recurrence argument to $h = \nu _ { 0 } + 1 , \dots , r$ gives at least $\lfloor ( r - \nu _ { 0 } ) / s \rfloor$ linearly independent images. Since $s \leq$ k and $D \leq m ,$ , we obtain

$$
\mathrm { r a n k } T _ { p , \lambda } ^ { 0 } \geq L : = \left\lfloor \frac { r } { k } - \frac { ( m - 1 ) ( k - 1 ) } { 2 } \right\rfloor .
$$

We consider the incidence variety $\mathcal { I } _ { k , m , i } ^ { 0 , \leq }$ analogous to the one in (11), but with the parameter spaces $\mathbb { P } ( \mathbb { K } [ z ] _ { < r } ^ { 0 } )$ and $\mathcal { P } _ { k , \leq m } ^ { 0 , + }$ . The incidence argument in the proof of Theorem 2.5 then yields

$$
\dim { \mathcal { I } } _ { k , m , r } ^ { 0 , \leq } \leq k m + ( k - 1 ) + ( r - L - 1 ) = r + k m + k - L - 2 .
$$

The numerical assumption on r gives $L \ge k ( m { + } 1 )$ , hence dim $\begin{array} { r } { \mathcal { I } _ { k , m , r } ^ { 0 , \leq } \leq r - 2 . } \end{array}$ .Therefore, the closure of the image of $\mathfrak { I } _ { k , m , i } ^ { 0 , \le }$ under the projection onto the first factor is a proper subset of $\mathbb { P } ( \mathbb { K } [ z ] _ { \leq r } ^ { 0 } ) \cong$ $\mathbb { P } ^ { r - 1 }$ , and the complement of the corresponding affine cone is the required open subset $U _ { k , m , r } . \in \Sigma$

## 3.4 Examples for σ

The origin-passing activations $( z ^ { 3 } + z ^ { 2 } + z ) ^ { \ell }$ with $\ell > k ( k - 2 )$ ) from Example 2.7 already provide the existence of good activations for the two conjectures of this section. However, we can easily construct more such examples:

Example 3.6. Following the construction of Example 2.7, let $k \geq 2 , \ell > k ( k - 2 ) , d \geq 2 ,$ and $f = a _ { 1 } z + \ldots + a _ { d } z ^ { d }$ with $a _ { d - 1 } a _ { d } \neq 0$ Combining the Newman–Slater theorem 1.1 with Proposition 3.3 shows that $\sigma _ { \ell , f } : = f ^ { \ell }$ is good, that is, for any pairwise distinct, nonzero, originpassing polynomials $p _ { 1 } , \ldots , p _ { k }$ , the polynomials $\sigma _ { \ell , f } ( p _ { 1 } ) , \ldots , \sigma _ { \ell , f } ( p _ { k } )$ are linearly independent. The activation of minimal degree obtained from this construction is $( z ^ { 2 } + z ) ^ { k ( k - 2 ) + 1 }$

Example 3.7. Another example is $\sigma _ { r } : = ( z + 1 ) ^ { r } - 1$ , which is good in the above sense whenever $r \geq k ^ { \bar { 2 } } - 1$ . Indeed, let $p _ { 1 } , \ldots , p _ { k }$ be pairwise distinct, nonzero, origin-passing polynomials, and suppose that ${ \begin{array} { r } { \sum _ { i = 1 } ^ { k } \lambda _ { i } \sigma _ { r } ( p _ { i } ) = 0 } \end{array} }$ for some scalars $\lambda _ { i }$ . Setting $q _ { i } : = 1 + p _ { i }$ , this becomes

$$
- \left( \sum _ { i = 1 } ^ { k } \lambda _ { i } \right) 1 ^ { r } + \sum _ { i = 1 } ^ { k } \lambda _ { i } q _ { i } ^ { r } = 0 .\tag{13}
$$

Since $p _ { i } ( 0 ) = 0$ , all $q _ { i }$ have constant term 1. Therefore, the polynomials $1 , q _ { 1 } , \ldots , q _ { k }$ are pairwise nonproportional. The Newman–Slater theorem 1.1 shows that the $k + 1$ polynomials $1 ^ { r } , q _ { 1 } ^ { r } , \ldots , q _ { k } ^ { r }$ are linearly independent, so (13) implies $\lambda _ { 1 } = \cdot \cdot \cdot = \lambda _ { k } = 0$

## 4 Identifiability of Polynomial MLPs

In this section, we explain how Conjecture 3.2 resolves identifiability for deep multilayer perceptrons with sufficiently general polynomial activations. We first formally define such networks. Fix a function $\sigma \colon \mathbb { R }  \mathbb { R } , \mathfrak { i }$ a sequence of $L > 1$ positive integers $d _ { 0 } , \ldots , d _ { L }$ , and, for every $i = 1 , \ldots , L ,$ a matrix $W _ { i } \in \mathbb { R } ^ { d _ { i } \times d _ { i - 1 } }$

Definition 1. A multilayer perceptron (MLP) with architecture $\mathbf { d } = ( d _ { 0 } , \dots , d _ { L } )$ , activation function σ and weights $\mathbf { W } = \left( W _ { 1 } , \ldots , W _ { L } \right)$ is the function $f _ { \mathbf { W } } \colon \mathbb { R } ^ { d _ { 0 } } \to \mathbb { R } ^ { d _ { L } }$ given by the composition:

$$
f _ { \mathbf { W } } = W _ { L } \circ \sigma \circ \cdots \circ \sigma \circ W _ { 1 } ,\tag{14}
$$

where $\sigma$ is applied coordinate-wise.

We denote by $\mathcal { W } : = \oplus _ { i = 1 } ^ { L } \mathbb { R } ^ { d _ { i } \times d _ { i - 1 } }$ the parameter space of an MLP, and by $\varphi \colon \mathcal { W } \ni \mathbf { W } \mapsto f _ { \mathbf { W } }$ its parametrization map. Next, we introduce the notion of a subnetwork.

Definition 2. Let $0 < j < L$ and $1 \leq i \leq d _ { i }$ . The i-th neuron in layer j is called

• inactive if its incoming or outgoing weights vanish, i.e., $W _ { j } [ i , : ] = 0 \mathrm { o r } W _ { j + 1 } [ : , i ] = 0 ;$

• redundant if $W _ { j } [ i , : ]$ is nonzero but equal to $W _ { j } [ k , : ]$ for some $1 \leq k \leq d _ { j }$ with $k \neq i .$

We call W a subnetwork if it contains at least one inactive or redundant neuron.

Remark 4.1. If W is a subnetwork, then f<sub>W</sub> can be represented by an MLP with a smaller architecture. Indeed, if the i-th column of $W _ { j + 1 }$ vanishes, it can be removed together with the i-th row of $W _ { j }$ , obtaining weights for the architecture $( d _ { 0 } , \dotsc , d _ { j - 1 } , d _ { j } - 1 , d _ { j + 1 } , \dotsc , d _ { L } )$ whose associated function coincides with $f _ { \mathbf { W } }$ . Assuming $\sigma ( 0 ) \overset { \cdot } { = } 0$ , the same holds when the i-th row of $W _ { j }$ vanishes. Similarly, when the i-th and the k-th rows of $W _ { j }$ coincide, one can remove the i-th row of $W _ { j }$ and add the i-th column of $W _ { j + 1 }$ to its k-th column.

Definition 3. The weights $\mathbf { W } \in \mathcal { W }$ are identifiable if, for every $\mathbf { V } \in { \mathsf { W } }$ with $f _ { \mathbf { W } } = f _ { \mathbf { V } }$ , there exist permutation matrices $\bar { P } _ { 1 } \in \mathbb R ^ { d _ { 1 } \times d _ { 1 } } , \dots , P _ { L - 1 } \in \mathbb R ^ { d _ { L - 1 } \times d _ { L - 1 } }$ such that

$$
V _ { i } = P _ { i } W _ { i } P _ { i - 1 } ^ { \top }\tag{15}
$$

for every $i = 1 , \ldots , L$ , where $P _ { 0 }$ and $P _ { L }$ are identity matrices.

In other words, weights are identifiable if all other weights in the same fiber under $\varphi$ are obtained by permuting the neurons in each layer.

Corollary 4.1 (of Conjecture 3.2). Fix an architecture d $\begin{array} { c c l } { { \mathsf { I } } } & { { = } } & { { \left( d _ { 0 } , \ldots , d _ { L } \right) } } \end{array}$ Let $\begin{array} { r l } { D } & { { } : = } \end{array}$ r $\operatorname* { n a x } \{ d _ { 1 } , \dotsc , d _ { L - 1 } \}$ } and, using the notation in Conjecture 3.2, $l e t r \geq R ( 2 D )$ . There is a nonempty Zariski open subset $U \subseteq \mathbb { R } [ z ] { \overset { \sim } { \leq } } .$ with the following property: For every MLP of architecture d with activation $\sigma \in U$ , the weights $\mathbf { \bar { W } } \in \mathbf { \mathcal { W } }$ are identifiable if and only if they are not a subnetwork.

Proof. It is clear from Remark 4.1 that subnetworks are not identifiable. For the converse, we argue by induction on the depth $L .$ For $L = 1$ , the statement is straightforward. Thus, we assume $L > 1$ Note that a statement of Conjecture 3.2 for $2 D$ polynomials also applies to any smaller collection with the same degree bound $R ( 2 D )$ , since one can complete it to $2 D$ nonzero pairwise distinct origin-passing polynomials. For every architecture $( e _ { 0 } , \ldots , e _ { L - 1 } )$ satisfying $1 \ \leq \ e _ { i } \ \leq \ d _ { i }$ , the induction hypothesis provides a nonempty Zariski open subset of good activations in $\mathbb { R } [ z ] _ { \leq r } ^ { 0 }$ . We let $U ^ { \prime } \subseteq \mathbb { R } [ z ] _ { < _ { i } } ^ { 0 }$ be the intersection of those subsets. Moreover, setting $m : = r ^ { L - 2 }$ to be the maximum degree of the polynomials output by the network after $L - 1$ layers, we consider the nonempty Zariski open subset $U _ { 2 D , r , m }$ from Conjecture 3.2. We then define the nonempty Zariski open subset $U : = U ^ { \prime } \cap U _ { 2 D , r , m }$ of $\mathbb { R } [ z ] _ { \leq r } ^ { 0 }$ , and let $\sigma \in U$

We consider weights $\mathbf { W } \in \mathcal { W }$ for the architecture d that are not a subnetwork. Denote by $p _ { j }$ the output of the $j \mathrm { - t h }$ neuron in the penultimate layer of $f _ { \mathbf { W } }$ (before applying σ). In other words, $p _ { j }$ is the MLP with architecture $\mathbf { d } ^ { \prime } : = ( \bar { d } _ { 0 } , \dots , d _ { L - 2 } , \bar { 1 } )$ and weights $\mathbf { W } _ { j } ^ { \prime } : = ( \bar { W } _ { 1 } , \dots , W _ { L - 2 } , W _ { L - 1 } [ j , : ] )$

We claim that the $p _ { j }$ are nonzero polynomials. This follows from the inductive hypothesis as follows: If the weights $\mathbf { W } _ { i } ^ { \prime }$ are not a subnetwork for the architecture $\mathbf { d } ^ { \prime }$ , then they are identifiable (by the inductive hypothesis) and thus $p _ { j }$ cannot be zero (since otherwise it would be parametrized both by the nonzero weights $\mathbf { W } _ { j } ^ { \prime }$ and by the all-zero weights). However, even though W is not a subnetwork, the weights $\mathbf { W } _ { j } ^ { \prime }$ might be, since some entries of $W _ { L - 1 } [ j , : ]$ (but not all!) can be zero, resulting in inactive neurons at the $( L - 2 ) \ – \mathrm { t h }$ layer. As explained in Remark 4.1, these neurons can be removed. This operation may generate inactive neurons with vanishing outgoing weights at earlier layers, which can be removed iteratively. In this way, we eventually obtain a smaller MLP with nonsubnetwork weights parametrizing $p _ { j }$ . Now, we can apply the inductive hypothesis and conclude that these weights are identifiable for the smaller MLP. Hence, as above, we see that $p _ { j }$ cannot be zero.

Next, we claim that the $p _ { j }$ are pairwise distinct. If $p _ { j } = p _ { k }$ for some $j \neq k ,$ , consider the MLP with architecture d<sup>′</sup> and weights $( \hat { W _ { 1 } } , \ldots , W _ { L - 2 } , W _ { L - 1 } [ j , : ] - W _ { L - 1 } [ k , : ] )$ , which parametrizes the zero polynomial. Since $W _ { L - 1 } [ j , : ] \neq W _ { L - 1 } [ k , : ]$ ], the same reduction procedure as above yields a smaller MLP with non-subnetwork weights parametrizing zero, again contradicting that these weights must be identifiable by the inductive hypothesis.

Now, consider weights $\mathbf { V } \in { \mathsf { W } }$ such that $f _ { \mathbf { W } } = f _ { \mathbf { V } }$ . Let $q _ { k }$ be the output of the k-th neuron in the penultimate layer of $f _ { \mathbf { V } }$ (before applying σ). Then, for every $i = 1 , \ldots , d _ { L }$ ,

$$
\sum _ { j = 1 } ^ { d _ { L - 1 } } W _ { L } [ i , j ] \sigma ( p _ { j } ) - \sum _ { k = 1 } ^ { d _ { L - 1 } } V _ { L } [ i , k ] \sigma ( q _ { k } ) = 0 .\tag{16}
$$

Fix $j \in \left\{ 1 , \dots , d _ { L - 1 } \right\}$ . Since $W _ { L }$ has no vanishing columns, there exists i such that $W _ { L } [ i , j ] \neq 0$ For such an $i ,$ apply Conjecture 3.2 to (16), after removing zero terms and grouping equal polynomials $q _ { \bullet }$ and $p _ { \bullet }$ . It follows that $p _ { j } = q _ { k }$ for some $k .$

Since the $p _ { j }$ are pairwise distinct as $j$ varies, the resulting map $j \mapsto$ k is injective, hence bijective. Let $P _ { L - 1 }$ be the corresponding permutation matrix. The linear independence in (16), guaranteed by Conjecture 3.2, then gives $\bar { V _ { L } } = W _ { L } P _ { L - 1 } ^ { \top }$ . Moreover, the depth $L - 1$ networks with weights $( W _ { 1 } , \dots , W _ { L - 2 } , P _ { L - 1 } W _ { L - 1 } )$ and $( V _ { 1 } , \ldots , V _ { L - 1 } )$ parametrize the same function. The first network is not a subnetwork, so the inductive hypothesis yields permutation matrices $P _ { 1 } , \ldots , P _ { L - 2 }$ such that $V _ { i } = P _ { i } W _ { i } P _ { i - 1 } ^ { \top }$ for $i = 1 , \ldots , L - 2$ (with $P _ { 0 } = I )$ and $V _ { L - 1 } = P _ { L - 1 } W _ { L - 1 } P _ { L - 2 } ^ { \top }$ . This proves that W is identifiable. □

Remark 4.2. Even though Conjecture 3.2 is open in general, its established cases from Section 3 imply, via the same proof, the following specific identifiability results:

• Width-one networks: Proposition 3.3 gives the conclusion of Corollary 4.1 when all hidden widths are one, that is, $d _ { i } = 1$ for $1 \leq i \leq L - 1$

• Layerwise activations: Theorem 3.5 gives the analogue of Corollary 4.1 for layer-specific origin-passing activations. More precisely, consider

$$
f _ { \mathbf { W } } = W _ { L } \circ \sigma _ { L - 1 } \circ \cdots \circ \sigma _ { 1 } \circ W _ { 1 } ,
$$

where deg $\sigma _ { i } = r _ { i }$ . Set $m _ { 0 } : = 1$ and $m _ { i } : = r _ { 1 } \cdot \cdot \cdot r _ { i }$ . The outputs from the previous layer entering $\sigma _ { i }$ have degree at most $m _ { i - 1 }$ . Hence, it is enough to choose $r _ { i }$ recursively so that

$$
r _ { i } \geq d _ { i } ( 6 d _ { i } m _ { i - 1 } + 2 d _ { i } - m _ { i - 1 } + 1 ) ,
$$

and to choose $\sigma _ { i }$ in the nonempty Zariski open subset $U _ { 2 d _ { i } , m _ { i - 1 } , r _ { i } }$ given by Theorem 3.5.

• Shallow networks: As a particular case, Corollary 4.1 holds unconditionally for $L = 2$ Here, $m = 1$ and $k = 2 d _ { 1 }$ , so Theorem 3.5 gives the sufficient bound $r \geq 7 d _ { 1 } ^ { 2 }$

• Specific choices of activations: Furthermore, for deep MLPs with any of the specific activations from Section 3.4 (with $k = 2 D )$ , the weights are also identifiable if and only if they are not a subnetwork.

Remark 4.3. The results in this section concern MLPs without bias vectors. To deal with MLPs that have bias vectors, one generalizes Definition 1 by allowing affine linear maps instead of the linear maps $W _ { i } .$ . As discussed above, the identifiability of MLPs with bias vectors would follow from a strengthening of Conjectures 1.2 and 2.6 that asks for the $k + 1$ polynomials 1, $, \sigma ( p _ { 1 } ) , \ldots , \sigma ( p _ { k } )$ to be linearly independent.

## 5 Conclusion

We have related the identifiability of multilayer perceptrons with generic polynomial activations to a linear-independence problem for polynomial compositions. We established this linear independence in several special cases, resolving the identifiability question for some MLP architectures, most notably for shallow MLPs and deep MLPs whose activation degrees grow from layer to layer.

For all MLP architectures with a full identifiability characterization, the following remarkable result holds. For every MLP $f _ { \mathbf { W } }$ that is equivariant under some group action, there are weights V parametrizing the same function $( \mathrm { i } . \mathrm { e } . , f \mathbf { v } = f _ { \mathbf { W } } )$ such that every single layer $x \mapsto \sigma ( V _ { i } x )$ is equivariant [15]. Hence, for all such architectures, whenever one wishes to implement equivariant MLPs, it is sufficient to implement layerwise-equivariant ones.

Another application of identifiability is model stitching [1], which asks whether independently trained networks have compatible internal representations such that the lower layers of one network can be plugged into the upper layers of the other after only a simple alignment map. For MLP architectures with a full identifiability characterization, model stitching is possible since for two networks that are sufficiently well-trained so that they represent the same function, a permutation of the neurons in the “stitching layer” is the desired alignment map.

Overall, the identifiability of parametrized machine learning models is one of the most classical and fundamental questions in deep learning theory, lying at the heart of many further questions of interest (e.g., equivariant network design, model stitching). A full resolution of identifiability for all polynomial MLP architectures would be a foundational contribution to deep learning. Such a resolution could be achieved by proving the conjectures in this article, or possibly by other means. At their core, these conjectures ask whether the Newman–Slater theorem can be generalized from monomials to generic polynomials. Thus, beyond their consequences for neural-network identifiability, the conjectures are of independent interest in commutative algebra and algebraic geometry.

## Acknowledgements

We are immensely grateful to Vahid Shahverdi for numerous invaluable discussions on this topic. We also thank Jan Draisma and his wonderful group at the University of Bern for stimulating discussions. KK and GLM were supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation. GLM was supported by the AI4S initiative funded by the Knut and Alice Wallenberg Foundation. AM is a member of GN-SAGA and was partially supported by the PRIN 2022 project 20223B5S8L, “Birational Geometry ofModuli Spaces and Special Varieties”. MM is a member of GNSAGA and was partially supported by PRIN 2022 project 2022ZRRL4C “Multilinear Algebraic Geometry”.

## References

[1] Yamini Bansal, Preetum Nakkiran, and Boaz Barak. Revisiting model stitching to compare neural representations. In Advances in Neural Information Processing Systems, volume 34, 2021.

[2] Alexandru Craciun. Linear independence of powers for polynomials.˘ arXiv:2507.10163, 2025.

[3] Charles Fefferman. Reconstructing a neural net from its output. Revista Matemática Iberoamericana, 10(3):507–556, 1994.

[4] Bella Finkel, Jose Israel Rodriguez, Chenxi Wu, and Thomas Yahl. Activation thresholds and expressiveness of polynomial neural networks. arXiv:2408.04569, 2024.

[5] Johanna Marie Gegenfurtner, Moritz Grillo, and Guido Montúfar. The symmetries of threelayer relu networks. arXiv:2605.18319, 2026.

[6] Elisenda Grigsby, Kathryn Lindsey, and David Rolnick. Hidden symmetries of relu networks. In International Conference on Machine Learning, pages 11734–11760. PMLR, 2023.

[7] Moritz Grillo and Guido Montúfar. Most relu networks admit identifiable parameters. arXiv:2605.03601, 2026.

[8] Nathan W. Henry, Giovanni Luca Marchetti, and Kathlén Kohn. Geometry of lightning selfattention: Identifiability and dimension. In International Conference on Learning Representations, 2025.

[9] Giovanni Luca Marchetti, Vahid Shahverdi, Stefano Mereta, Matthew Trager, and Kathlén Kohn. Algebra unveils deep learning – an invitation to neuroalgebraic geometry. In International Conference on Machine Learning, 2025. URL https://arxiv.org/abs/2501. 18915.

[10] Alex Massarenti and Massimiliano Mella. The alexander-hirschowitz theorem for neurovarieties. arXiv:2511.19703, 2025.

[11] DJ Newman and Morton Slater. Waring’s problem for the ring of polynomials. Journal of Number Theory, 11(4):477–487, 1979.

[12] Henning Petzka, Martin Trimmel, and Cristian Sminchisescu. Notes on the symmetries of 2- layer relu-networks. In Proceedings ofthe northern lights deep learning workshop, volume 1, pages 6–6, 2020.

[13] Mary Phuong and Christoph H Lampert. Functional vs. parametric equivalence of relu networks. In International conference on learning representations, 2020.

[14] Vahid Shahverdi, Giovanni Luca Marchetti, and Kathlén Kohn. On the geometry and optimization of polynomial convolutional networks. In Artificial Intelligence and Statistics, 2025.

[15] Vahid Shahverdi, Giovanni Luca Marchetti, Georg Bökman, and Kathlén Kohn. Identifiable equivariant networks are layerwise equivariant. arXiv:2601.21645, 2026.

[16] Vahid Shahverdi, Giovanni Luca Marchetti, and Kathlén Kohn. Learning on a razor’s edge: Identifiability and singularity of polynomial neural networks. In The Fourteenth International Conference on Learning Representations, 2026.

[17] Héctor J Sussmann. Uniqueness of the weights for minimal feedforward nets with a given input-output map. Neural networks, 5(4):589–593, 1992.

[18] Matthew Trager, Kathlén Kohn, and Joan Bruna. Pure and spurious critical points: a geometric study of linear networks. In International Conference on Learning Representations, 2020.

[19] Konstantin Usevich, Clara Dérand, Ricardo Borsoi, and Marianne Clausel. Identifiability of deep polynomial neural networks. arXiv:2506.17093, 2025.

[20] Verner Vlaciˇ c and Helmut Bölcskei. Affine symmetries and neural network identifiability.´ Advances in Mathematics, 376:107485, 2021.

[21] Verner Vlaciˇ c and Helmut Bölcskei. Neural network identifiability for a family of sigmoidal ´ nonlinearities. Constructive Approximation, 55(1):173–224, 2022.