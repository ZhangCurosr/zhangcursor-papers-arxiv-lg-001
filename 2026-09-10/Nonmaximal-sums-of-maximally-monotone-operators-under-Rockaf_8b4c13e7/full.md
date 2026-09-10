# Nonmaximal sums of maximally monotone operators under Rockafellar’s constraint qualification

Weifeng Yang<sup>∗</sup>

## Abstract

We construct counterexamples to Rockafellar’s sum conjecture in which two maximally monotone operators satisfy the interior-domain condition but their sum is not maximally monotone. We give one counterexample on $c _ { 0 }$ and another on $\ell ^ { 1 }$ with its usual norm. We establish a general construction theorem that computes the entire monotone polar of a class of graphs, gives a necessary and suficient condition for their maximal monotonicity, and shows how a positive rank-one perturbation yields a nonmaximal sum under this condition. We verify the theorem’s hypotheses and its maximality criterion on $c _ { 0 } .$ , thereby obtaining a counterexample to the conjecture. Furthermore, we construct a bounded linear surjection from $\ell ^ { 1 }$ onto $c _ { 0 }$ and use it to obtain the counterexample on $\ell ^ { 1 }$

Keywords: maximally monotone operator, sum theorem, nonreflexive Banach space, monotone polar, rank-one operator.

## 1 Introduction

Let X be a real Banach space, let $X ^ { * }$ be its continuous dual, and let $A , B : X  X ^ { * }$ be maximally monotone operators. Their pointwise sum is defined by $( A + B ) x = \{ a + b : a \in A x , \ b \in B x \}$ Rockafellar’s sum theorem [1] states that $A + B$ is maximally monotone when X is reflexive and

$$
\operatorname { d o m } A \cap \operatorname { i n t } ( \operatorname { d o m } B ) \neq \emptyset .\tag{1}
$$

The unrestricted sum question asks whether Eq. (1) alone is suficient on an arbitrary real Banach space [2]. The interior here is taken in the norm topology of X, and maximality is taken in the full dual pair $X \times X ^ { * }$ . Bot¸, Bueno, and Simons [3] show that the unrestricted sum question is equivalent to its restriction to pairs with bounded common domain.

In general Banach spaces, sum theorems have been established under additional assumptions on the operators. Yao [4] proves maximality when A is of type (FPV) and B has full domain. Borwein and Yao [5] prove maximality under Eq. (1) when A is a linear relation. Voisei [6] treats the sum of an operator of type (FPV) with convex domain and a normal-cone operator. Verona and Verona [7, Main Result 1.1(b)] prove the sum theorem under Eq. (1) when A is C<sub>0</sub>-maximal: $A + N _ { C }$ is maximally monotone for every closed convex set C with dom A ∩ int $C \neq \varnothing$ , where $N _ { C }$ denotes the normal-cone operator of C. Bauschke, Borwein, Wang, and Yao [8] develop triangular operators and nonlinear examples. We use the triangular map of Bauschke et al. as a building block for our construction on $c _ { 0 }$

Suficient conditions for maximality of sums have also been obtained through convex representations of monotone operators. Fitzpatrick [9] associates a convex function with a monotone graph, and Burachik and Svaiter [10] study the associated family of convex representatives through enlargements. Using convex representations, Penot [11] develops results for compositions and sums in reflexive spaces, and Marques Alves and Svaiter [12] prove a sum theorem in general Banach spaces under a qualification on the domains of Fitzpatrick representations. Lower bounds by the duality products for a proper lower semicontinuous convex function and its conjugate yield a maximally monotone operator [13]. Voisei and Z˘alinescu [14] study the resulting class and its calculus.

In this paper, we construct counterexamples to Rockafellar’s sum conjecture on $c _ { 0 }$ and standard $\ell ^ { 1 }$ . We develop a general construction theorem that computes the entire monotone polar of a class of graphs, characterizes their maximality, and identifies a positive rank-one perturbation whose addition makes the sum nonmaximal. To satisfy the theorem’s assumptions on $c _ { 0 }$ , we couple copies of the triangular map through a Lipschitz curve, and then obtain a counterexample to the conjecture. A bounded linear surjection then transfers this counterexample to standard $\ell ^ { 1 }$

## 1.1 Contributions

The contributions of this paper are as follows.

(1) We establish a general construction theorem for counterexamples to Rockafellar’s sum conjecture (Theorem 1). Under Assumption 1, the theorem computes the entire monotone polar of the graph in Eq. (4) and characterizes its maximality by the criterion in Eq. (9). When this criterion holds, the graph defines a maximally monotone operator whose sum with a specified everywhere-defined positive rank-one operator is not maximally monotone, although the two operators satisfy Eq. (1).

(2) We construct an explicit counterexample on c<sub>0</sub> (Theorem 2) by verifying the assumptions of this theorem. The second operator is bounded, positive, rank-one, and defined everywhere, and the first operator has nonempty domain. Thus the original interior-domain condition holds. We also establish a finite radial bound at the origin for the first operator (Definition 2).

(3) We construct a counterexample on standard $\ell ^ { 1 }$ (Theorem 3) by pulling back the first operator through a specified bounded linear surjection onto $c _ { 0 }$ . By applying Lemma 1, we prove that both operators are maximally monotone in $\ell ^ { 1 } \times \ell ^ { \infty }$ . We establish nonmaximality of their sum by showing that (0, 0) lies outside its graph and is monotonically related to every point of that graph.

The paper is organized as follows. Section 2 introduces the notation and basic definitions. Section 3 proves the structural theorem and the pullback lemma. Section 4 constructs the counterexamples on $c _ { 0 }$ and standard $\ell ^ { 1 }$ and derives their stated consequences. Section 5 concludes the paper.

## 2 Preliminaries

All scalars are real. Put $\mathbb { N } = \{ 1 , 2 , \ldots \}$ and $\mathbb { N } _ { 0 } = \{ 0 , 1 , . . . \}$ . For a Banach space $X ,$ , we denote by $X ^ { * }$ its continuous dual, the space of continuous linear functionals on $X$ . We write $\langle x , a \rangle = a ( x )$ for $x \in X$ and $a \in X ^ { * }$ . For a real Hilbert space H, its inner product is denoted by $( u , v ) _ { H }$ . We use the norm topology for interiors and closures unless stated otherwise.

For a bounded linear map $P : X \to X ^ { * }$ , we use positive to mean $\langle x , P x \rangle \geq 0$ for every $x \in X$ 2 and rank one to mean dim(ran P) = 1.

Monotone polars and convex representations are studied in [15, 16]. Their relation to representable closures and maximal monotone extensions is examined in [17].

Definition 1 (Monotonicity and the monotone polar). For a graph $G \subset X \times X ^ { * }$ , define

$$
G ^ { \mu } = \{ ( z , p ) \in X \times X ^ { * } : \langle z - x , p - a \rangle \geq 0 \quad \forall ( x , a ) \in G \} .\tag{2}
$$

A point $( z , p )$ is monotonically related to $G { \mathrm { ~ i f ~ } } ( z , p ) \in G ^ { \mu }$ . The graph is monotone when $\langle x - y , a - b \rangle \geq 0$ for all $( x , a ) , ( y , b ) \in G$ , and maximally monotone when it has no proper monotone extension in this same full dual pair.

Proposition 1. A monotone graph $G \subset X \times X ^ { * }$ is maximally monotone if and only $i f G = G ^ { \mu }$

Proof. A point in $G ^ { \mu } \backslash G$ can be adjoined to $G _ { i }$ , and every point of a monotone extension belongs to $G ^ { \mu }$ □

We write dom $G = \left\{ x : \exists a , \ ( x , a ) \in G \right\} { \mathrm { ~ a n d ~ } } \operatorname { r a n } G = \left\{ a : \exists x , \ ( x , a ) \in G \right\}$

Definition 2 (Radial bound at the origin). For a nonempty graph G with $0 \not \in$ dom $G ,$ define

$$
\mathcal { V } ( G ) = \operatorname* { s u p } _ { ( x , a ) \in G } \frac { \operatorname* { m a x } \{ 0 , - \langle x , a \rangle \} } { \| x \| } .\tag{3}
$$

We say that G has a finite radial bound at the origin if $\mathcal { V } ( G ) < \infty$

For $G = \mathrm { g r a } T$ as in the preceding definition, the quantity of Verona and Verona [18, p. 1004] satisfies

$$
\begin{array} { r l } & { L ( 0 , 0 , T ) = \operatorname* { m a x } \left\{ 0 , \ \underset { ( x , a ) \in G } { \operatorname* { s u p } } \frac { - \langle x , a \rangle } { \| x \| } \right\} } \\ & { \quad \quad = \ \underset { ( x , a ) \in G } { \operatorname* { s u p } } \frac { \operatorname* { m a x } \{ 0 , - \langle x , a \rangle \} } { \| x \| } = \mathcal { V } ( G ) . } \end{array}
$$

For every $C \geq 0$ , Eq. (3) gives

$$
\mathcal { V } ( G ) \leq C \quad \Longleftrightarrow \quad \langle x , a \rangle \geq - C \| x \| \quad { \mathrm { f o r ~ a l l ~ } } ( x , a ) \in G .
$$

Thus, when finite, $\mathcal { V } ( G )$ is the least such constant. The bounds for our two operators are proved in Theorems 2 and 3, using Eqs. (21) and (28), respectively.

## 3 Maximal monotonicity and nonmaximal sums

In this section, we establish a general construction theorem for maximally monotone operators whose sums with an everywhere-defined positive rank-one operator are not maximally monotone. We also prove a lemma that transfers maximal monotonicity through a bounded linear surjection.

Let X be a real Banach space.

Definition 3 (Hilbert spaces and norms). Let $H _ { 0 }$ be a real Hilbert space with inner product $( \cdot , \cdot ) _ { H _ { 0 } }$ and induced norm

$$
\Vert u \Vert _ { H _ { 0 } } = \sqrt { ( u , u ) _ { H _ { 0 } } } \qquad ( u \in H _ { 0 } ) .
$$

Define $H = \mathbb { R } \oplus H _ { 0 }$ to be $\mathbb { R } \times H _ { 0 }$ equipped with the inner product

$$
\begin{array} { r } { ( ( s , u ) , ( t , v ) ) _ { H } = s t + ( u , v ) _ { H _ { 0 } } \qquad ( s , t \in \mathbb { R } , \ u , v \in H _ { 0 } ) . } \end{array}
$$

Its induced norm is

$$
\| ( s , u ) \| _ { H } = \sqrt { s ^ { 2 } + \| u \| _ { H _ { 0 } } ^ { 2 } } .
$$

Next, we give the following definition.

Definition 4 (The linear maps). Let $E : X ^ { * } \to X$ and $M : X ^ { * }  H$ be bounded linear maps, and let $g \in X ^ { * } \setminus \{ 0 \}$ . Define

$$
\begin{array} { c } { h = E g , } \\ { \displaystyle r ( a ) = \langle h , a \rangle , } \\ { L a = - E a + r ( a ) h , } \\ { K = \ker M \cap \ker r . } \end{array}
$$

We use $r ( a )$ as a scalar parameter and constrain M a to lie on a prescribed curve at that parameter. This specifies the graph as follows.

Definition 5 (The constrained graph). Let $\omega : ( 0 , \infty ) \to H _ { 0 }$ be ℓ-Lipschitz, where $0 \leq \ell \leq 1$ Define

$$
\begin{array} { r } { J = ( - \infty , - 1 ) \cup ( 1 , \infty ) , \qquad } \\ { F ( t ) = ( \mathrm { s g n } t , \omega ( | t | - 1 ) ) \qquad ( t \in J ) , } \end{array}
$$

where

$$
\operatorname { s g n } ( t ) = { \left\{ \begin{array} { l l } { 1 , } & { t > 0 , } \\ { 0 , } & { t = 0 , } \\ { - 1 , } & { t < 0 . } \end{array} \right. }
$$

The admissible dual variables and the graph are

$$
\begin{array} { l } { { D = \{ a \in X ^ { * } : r ( a ) \in J , \ M a = F ( r ( a ) ) \} , } } \\ { { \ } } \\ { { G = \{ ( L a , a ) : a \in D \} . } } \end{array}\tag{4}
$$

To compute the entire monotone polar of $G ,$ , we impose the following conditions on the linear maps, the curve, and its realizability in $X ^ { \ast }$

Assumption 1. For the maps and graph in Definitions 4 and $5 ,$ assume:

(H1) $M g = 0$ , and

$$
\langle E a , b \rangle + \langle E b , a \rangle = 2 ( M a , M b ) _ { H } \qquad ( a , b \in X ^ { * } ) .\tag{5}
$$

(H2) The subspace K satisfies

$$
\{ x \in X : \langle x , k \rangle = 0 { \mathrm { ~ f o r ~ a l l ~ } } k \in K \} = \mathbb { R } h .\tag{6}
$$

(H3) There exists $\eta \in H _ { 0 }$ such that

$$
\begin{array} { r l r } {  { \omega ( s ) \longrightarrow \eta \mathrm { ~ i n ~ } H _ { 0 } \quad \mathrm { a s ~ } s \downarrow 0 , } } \\ & { } & { 0 < S : = \| \eta \| _ { H _ { 0 } } ^ { 2 } < 1 , \ } \\ & { } & { \| \omega ( s ) \| _ { H _ { 0 } } ^ { 2 } \le S \qquad ( s > 0 ) . } \end{array}\tag{7}
$$

(H4) For every $t \in J$ , there exists $a ^ { t } \in X ^ { * }$ satisfying $r ( a ^ { t } ) = t$ and $M a ^ { t } = F ( t )$

By Assumption 1(H4), each parameter fibre $\{ a \in X ^ { * } : r ( a ) = t , \ M a = F ( t ) \}$ is nonempty and equals $a ^ { t } + K$ . Definition 5 includes every point in each fibre. Combining Eq. (5) and the Lipschitz bound on ω with these afine fibres, we obtain an explicit formula for $G ^ { \mu }$ . The following theorem shows that, under Assumption 1, G is maximally monotone if and only if no $( p , \tau ) \in X ^ { * } \times [ - 1 , 1 ]$ satisfies $r ( p ) = \tau$ and $M p = ( \tau , \eta )$ When this condition holds, adding the operator $\mathcal { P } x = \langle x , g \rangle g$ to the operator with graph G produces a nonmaximal sum. Therefore, it gives a general construction criterion for counterexamples satisfying the interiordomain condition in Eq. (1) as follows.

Theorem 1 (Construction of nonmaximal sums). For the graph G in Definition 5, suppose Assumption 1 holds and define

$$
\widehat { F } ( t ) = \left\{ \begin{array} { l l } { F ( t ) , } & { | t | > 1 , } \\ { ( t , \eta ) , } & { | t | \leq 1 . } \end{array} \right.
$$

Then the following statements hold.

(i) The entire monotone polar of G in $X \times X ^ { * }$ is

$$
G ^ { \mu } = \{ ( L p , p ) : p \in X ^ { * } , \ M p = { \widehat { F } } ( r ( p ) ) \} .\tag{8}
$$

This graph is the unique maximally monotone extension of G.

(ii) The graph G is maximally monotone if and only if

$$
\vec { \mathbb { f } } ( p , \tau ) \in X ^ { * } \times [ - 1 , 1 ] : \quad r ( p ) = \tau , \quad M p = ( \tau , \eta ) .\tag{9}
$$

(iii) Suppose Eq. (9) holds. Let A be the operator with graph G, and define $\mathcal { P } x = \langle x , g \rangle g$ . Then

$$
0 \not \in \mathrm { d o m } { \cal A } , \qquad \mathcal { V } ( G ) \leq S \| g \| .
$$

Both A and $\mathcal { P }$ are maximally monotone and satisfy

$$
\mathrm { d o m } { \cal A } \cap \mathrm { i n t } ( \mathrm { d o m } { \cal P } ) \neq \emptyset .
$$

Their sum is not maximally monotone. More precisely,

$$
( 0 , 0 ) \in \left( \operatorname { g r a } ( A + \mathcal { P } ) \right) ^ { \mu } \backslash \operatorname { g r a } ( A + \mathcal { P } ) .
$$

Proof. (i) The monotone polar and its maximality. Eq. (5) at $a = b = g$ gives $r ( g ) = 0$ . Using $M g = 0$ again gives

$$
\begin{array} { c } { { \langle E a , g \rangle = - r ( a ) , } } \\ { { \langle L a , g \rangle = r ( a ) , } } \\ { { \langle L a - L b , a - b \rangle = | r ( a ) - r ( b ) | ^ { 2 } - \| M a - M b \| ^ { 2 } . } } \end{array}\tag{10}
$$

Taking $t = 2$ in Assumption 1(H4) gives $ r ( a ^ { 2 } ) = 2$ , so $h \neq 0$

Extend ω to zero by $\omega ( 0 ) = \eta$ . Its Lipschitz bound persists by the norm limit. Define $\chi ( t ) = \operatorname* { m a x } \{ - 1 , \operatorname* { m i n } \{ 1 , t \} \}$ and $\zeta ( t ) = \operatorname* { m a x } \{ | t | - 1 , 0 \}$ . Then $\widehat { F } ( t ) = ( \chi ( t ) , \omega ( \zeta ( t ) ) )$ . On each of $[ 0 , \infty )$ and $( - \infty , 0 ]$ , the definitions give

$$
| \chi ( t ) - \chi ( u ) | + | \zeta ( t ) - \zeta ( u ) | = | t - u | .
$$

For $t \geq 0 \geq u$ , they give

$$
\begin{array} { c } { | \chi ( t ) - \chi ( u ) | + | \zeta ( t ) - \zeta ( u ) | \leq \chi ( t ) - \chi ( u ) + \zeta ( t ) + \zeta ( u ) } \\ { = t - u . } \end{array}
$$

Interchanging t, u covers the remaining case. Therefore

$$
| \chi ( t ) - \chi ( u ) | + | \zeta ( t ) - \zeta ( u ) | \leq | t - u | ,
$$

and

$$
\| \widehat { F } ( t ) - \widehat { F } ( u ) \| ^ { 2 } \leq | \chi ( t ) - \chi ( u ) | ^ { 2 } + \ell ^ { 2 } | \zeta ( t ) - \zeta ( u ) | ^ { 2 } \leq | t - u | ^ { 2 } .
$$

Eq. (10) proves monotonicity of both G and the set on the right of Eq. (8). It also proves that this latter set is contained in $G ^ { \mu }$

Conversely, let $( x , p ) \in G ^ { \mu }$ . For $a \in D$ and $t = r ( a )$ , Eq. (5) gives

$$
\langle x - L a , p - a \rangle = \langle x , p \rangle - \langle x + E p , a \rangle + 2 ( M a , M p ) _ { H } - t r ( p ) + t ^ { 2 } - \| M a \| ^ { 2 } .
$$

Replacing a by $a + \theta k$ , for all $\theta \in \mathbb { R }$ and $k \in K$ , forces $\langle x + E p , k \rangle = 0$ . Thus, $x = - E p + v h$ by Eq. (6). With

$$
\tau = \frac { v + r ( p ) } { 2 } , ~ \nu = \frac { v - r ( p ) } { 2 } ,
$$

the same expansion reduces exactly to

$$
\langle x - L a , p - a \rangle = ( t - \tau ) ^ { 2 } - \nu ^ { 2 } - \| F ( t ) - M p \| ^ { 2 } .\tag{11}
$$

By Assumption 1(H4), the polar inequality holds for every $t \in J$ . We distinguish two cases.

Case $\begin{array} { r } { \boldsymbol { { \mathit { 1 } } } \colon | { \boldsymbol { \tau } } | > 1 } \end{array}$ . Taking $t = \tau$ in Eq. (11) gives $\nu = 0$ and $M p = F ( \tau )$ . Hence $v = r ( p ) = \tau$ and $x = L p$

Case $\begin{array} { r } { \mathcal { Q } \colon | \tau | \leq 1 } \end{array}$ . Write $M p = ( b , z ) \in H$ . Letting $t \to 1 ^ { + }$ and $t  - 1$ <sup>−</sup>, respectively, in Eq. (11), and using Assumption 1(H3), gives

$$
( b - 1 ) ^ { 2 } + \| z - \eta \| ^ { 2 } + \nu ^ { 2 } \leq ( 1 - \tau ) ^ { 2 } ,
$$

$$
( b + 1 ) ^ { 2 } + \| z - \eta \| ^ { 2 } + \nu ^ { 2 } \leq ( 1 + \tau ) ^ { 2 } .
$$

Multiplying the two inequalities by the nonnegative weights $( 1 + \tau ) / 2$ and $( 1 - \tau ) / 2$ , respectively, and adding gives

$$
\begin{array} { r } { ( b - \tau ) ^ { 2 } + \| z - \eta \| ^ { 2 } + \nu ^ { 2 } \leq 0 . } \end{array}
$$

Consequently $M p = ( \tau , \eta ) , r ( p ) = v = \tau$ , and $x = L p$ . This proves Eq. (8). Here $p \in X ^ { * }$ is fixed, and only $F ( t )$ is passed to the limit in $H$

We have proved that $G ^ { \mu }$ is monotone. Any point monotonically related to $G ^ { \mu }$ is monotonically related to $G$ and therefore belongs to $G ^ { \mu }$ , so $G ^ { \mu }$ is maximally monotone. Since every monotone extension of G lies in $G ^ { \mu }$ , a maximal one must equal it.

(ii) Maximality of G. By Eq. (8) and Definition 5,

$$
G ^ { \mu } \setminus G = \{ ( L p , p ) : p \in X ^ { * } , \ r ( p ) \in [ - 1 , 1 ] , \ M p = ( r ( p ) , \eta ) \} .
$$

Since $G$ is monotone, it is maximally monotone if and only if $G = G ^ { \mu }$ . The displayed set is empty exactly when no $( p , \tau ) \in X ^ { * } \times [ - 1 , 1 ]$ satisfies $r ( p ) = \tau$ and $M p = ( \tau , \eta )$ , which is $\operatorname { E q }$ (9).

(iii) The radial bound and nonmaximality of the sum. Assume Eq. (9). For each $a \in D$

$$
\begin{array} { r l } & { | \langle L a , g \rangle | = | r ( a ) | > 1 , } \\ & { \quad \| L a \| > 1 / \| g \| , } \\ & { \langle L a , a \rangle = r ( a ) ^ { 2 } - 1 - \| \omega ( | r ( a ) | - 1 ) \| ^ { 2 } > - S . } \end{array}
$$

Thus $0 \not \in$ dom A and, for every $a \in D$

$$
\frac { \operatorname* { m a x } \{ 0 , - \langle L a , a \rangle \} } { \| L a \| } \leq \frac { S } { \| L a \| } < S \| g \| .
$$

Taking the supremum proves $\mathcal { V } ( G ) \leq S \Vert g \Vert$ . The map $\mathcal { P }$ is bounded, linear, positive, nonzero, and defined on all of $X$ . It is also the subdiferential of the continuous convex function $x \mapsto$ $\langle x , g \rangle ^ { 2 } / 2 ,$ , in accordance with [19]. We prove its maximality directly. Its monotonicity follows from $\langle x - y , \mathcal { P } x - \mathcal { P } y \rangle = \langle x - y , g \rangle ^ { 2 }$ . For an arbitrary point $( z , z ^ { \ast } ) \in ( \mathrm { g r a } \mathcal { P } ) ^ { \mu }$ , tests at $z + t v$ give

$$
- t \langle v , z ^ { * } - \mathcal { P } z \rangle + t ^ { 2 } \langle v , g \rangle ^ { 2 } \geq 0 \qquad ( t \in \mathbb { R } , \ v \in X ) .
$$

Both signs of arbitrarily small t force $z ^ { * } = \mathcal { P } z$ . Thus $\mathcal { P }$ is maximally monotone, and $L a ^ { 2 } \in$ dom A proves dom $\boldsymbol { \mathcal { A } } \cap$ int dom $\mathcal { P } \neq \emptyset$ . For every $a \in D$

$$
\langle L a , a + \mathcal { P } L a \rangle = 2 r ( a ) ^ { 2 } - 1 - \| \omega ( | r ( a ) | - 1 ) \| ^ { 2 } > 1 - S > 0 .
$$

The sum is monotone and its domain does not contain zero. Adjoining (0, 0) gives a proper monotone extension of $\mathrm { g r a } ( \mathcal { A } + \mathcal { P } )$ . Hence $\mathcal { A } + \mathcal { P }$ is not maximally monotone. □

Theorem 1 provides a criterion for constructing the first operator together with a rank-one perturbation that makes its sum nonmaximal. For a graph satisfying Assumption 1, part (i) identifies every possible additional polar point. Part (ii) therefore reduces maximality to excluding $p \in X ^ { * }$ and $\tau \in [ - 1 , 1 ]$ solving

$$
\begin{array} { c } { { r ( p ) = \tau , } } \\ { { M p = ( \tau , \eta ) . } } \end{array}
$$

Once this is proved, part (iii) supplies the second operator $\mathcal { P }$ and the point (0, 0) witnessing nonmaximality of the sum. Thus, to obtain the example on $c _ { 0 } .$ we will choose the maps and the curve so that the system has no solution in the continuous dual of $c _ { 0 }$ . The following lemma then transfers that example to standard $\ell ^ { 1 }$

Lemma 1 (Pullback by a bounded surjection). Let U, V be real Banach spaces and let $Q : U \to V$ be a bounded linear surjection. Suppose $C _ { Q } > 0$ and every $y \in V$ has a preimage $u \in U$ with $\| u \| \leq C _ { Q } \| y \|$ . If ${ \mathcal { M } } : V \Longrightarrow V ^ { * }$ is maximally monotone, then

$$
\operatorname { g r a } ( Q ^ { * } { \mathcal { M } } Q ) = \{ ( u , Q ^ { * } a ) : ( Q u , a ) \in \operatorname { g r a } { \mathcal { M } } \}
$$

is maximally monotone in the full dual pair $U \times U ^ { * }$ , where $Q ^ { * } : V ^ { * } \to U ^ { * }$ is the adjoint of $Q _ { i }$ defined by $\langle u , Q ^ { * } a \rangle = \langle Q u , a \rangle$ for $u \in U$ and $a \in V ^ { * }$

Proof. A maximally monotone graph is nonempty, since the empty graph admits a one-point monotone extension. Surjectivity therefore makes the pullback graph nonempty. For two of its points,

$$
\begin{array} { r } { \langle u - v , Q ^ { * } ( a - b ) \rangle = \langle Q u - Q v , a - b \rangle \geq 0 , } \end{array}
$$

so it is monotone.

Let $( v , v ^ { * } ) \in U \times U ^ { * }$ be monotonically related to $\mathrm { g r a } ( Q ^ { * } { \mathcal { M } } Q )$ . Fix $( u , Q ^ { * } a )$ in the graph. For each $k \in$ ker $Q$ , all $( u + t k , Q ^ { * } a )$ belong to it, and their polar inequalities have the form

$$
\langle v - u , v ^ { * } - Q ^ { * } a \rangle - t \langle k , v ^ { * } \rangle \geq 0 \qquad ( t \in \mathbb { R } ) .
$$

Thus $v ^ { * }$ annihilates ker $Q$ . Define $p ( y ) = \langle u , v ^ { * } \rangle$ when $Q u = y$ . Diferences of preimages prove well-definedness, and linear combinations of preimages prove linearity. The prescribed lifting bound gives $| p ( y ) | \leq C _ { Q } \| v ^ { * } \| \| y \|$ , so $p \in V ^ { * }$ and $v ^ { * } = Q ^ { * } p$ . For every $( y , a ) \in \mathrm { g r a } { \mathcal { M } }$ , choose $u \in U$ with $Q u = y$ . Then $( u , Q ^ { * } a )$ belongs to the pullback graph. The remaining inequalities are

$$
\langle Q v - y , p - a \rangle \geq 0 \qquad ( ( y , a ) \in \operatorname { g r a } \mathcal { M } ) .
$$

Maximality of M gives $( Q v , p ) \in \mathrm { g r a } \mathcal { M } .$ , and $( \boldsymbol { v } , \boldsymbol { v } ^ { * } )$ belongs to the pullback graph. □

## 4 Counterexamples on $c _ { 0 }$ and standard $\ell ^ { 1 }$

In this section, we apply Theorem 1 to construct a counterexample on $c _ { 0 }$ and then use Lemma 1 to obtain a counterexample on standard $\ell ^ { 1 }$

## 4.1 A counterexample on $c _ { 0 }$

Put $I = \mathbb { N } _ { 0 } \times \mathbb { N }$ and $Y = c _ { 0 } ( I )$ with the supremum norm. For every $x \in Y$ and $\epsilon > 0$ , only finitely many $( b , j ) \in I$ satisfy $| x _ { b , j } | \ge \epsilon$ . Its continuous dual is $\ell ^ { 1 } ( I )$ , with $\begin{array} { r } { \langle x , a \rangle = \sum _ { b , j } x _ { b , j } a _ { b , j } } \end{array}$ We also use this pairing for $x \in \ell ^ { \infty } ( I )$ and $a \in \ell ^ { 1 } ( I )$ . The unit coordinate vectors are denoted by $e _ { b , j }$

We index coordinates by blocks. The map m records the sum in each block, and E applies a triangular operator within that block. The additional constraint below couples the block sums through $F$

Definition 6 (Block maps). For $a \in \ell ^ { 1 } ( I )$ , define

$$
\begin{array} { c } { { ( m a ) _ { b } = \displaystyle \sum _ { j \geq 1 } a _ { b , j } , } } \\ { { ( E a ) _ { b , j } = a _ { b , j } + 2 \displaystyle \sum _ { k > j } a _ { b , k } . } } \end{array}\tag{12}
$$

Here m : $\ell ^ { 1 } ( I ) \to \ell ^ { 1 } (  { \mathbb { N } } _ { 0 } ) \subset \ell ^ { 2 } (  { \mathbb { N } } _ { 0 } )$ and $E : \ell ^ { 1 } ( I ) \to Y$ . Put

$$
\begin{array} { c } { { g = e _ { 0 , 1 } - e _ { 0 , 2 } \in Y ^ { * } , } } \\ { { h = E g = - e _ { 0 , 1 } - e _ { 0 , 2 } \in Y , } } \\ { { r ( a ) = \langle h , a \rangle = - a _ { 0 , 1 } - a _ { 0 , 2 } , } } \\ { { L a = - E a + r ( a ) h . } } \end{array}\tag{13}
$$

Definition 7 (The curve of block sums). For $s > 0$ and $n \geq 1$ , write

$$
\begin{array} { c } { { \omega _ { s } ( n ) = \displaystyle \frac { \operatorname* { m a x } \{ 1 - s n ^ { 1 / 4 } , 0 \} } { 4 n } , } } \\ { { W ( s ) = \displaystyle \sum _ { n \geq 1 } \omega _ { s } ( n ) ^ { 2 } , } } \\ { { \sigma = \displaystyle \sum _ { n \geq 1 } \frac { 1 } { 1 6 n ^ { 2 } } = \displaystyle \frac { \pi ^ { 2 } } { 9 6 } . } } \end{array}\tag{14}
$$

Each $\omega _ { s } , s > 0$ , has finite support. Set $\omega _ { 0 } = ( 1 / ( 4 n ) ) _ { n \geq 1 } \in \ell ^ { 2 } \setminus \ell ^ { 1 }$ . Define

$$
\begin{array} { r l } & { \quad J = ( - \infty , - 1 ) \cup ( 1 , \infty ) , } \\ & { \quad F ( t ) = ( \mathrm { s g n } t , \omega _ { | t | - 1 } ) \in \ell ^ { 1 } ( \mathbb { N } _ { 0 } ) \quad ( t \in J ) , } \\ & { \quad D = \{ a \in \ell ^ { 1 } ( I ) : r ( a ) \in J , \ m a = F ( r ( a ) ) \} . } \end{array}\tag{15}
$$

Definition 8 (The two operators on $c _ { 0 } ( I ) )$ . The operators and kernel are

$$
\begin{array} { r l } & { \operatorname { g r a } A = \{ ( L a , a ) : a \in D \} , } \\ & { \quad K = \{ k \in \ell ^ { 1 } ( I ) : m k = 0 , \ r ( k ) = 0 \} , } \\ & { \quad P x = \langle x , g \rangle g \quad ( x \in Y ) . } \end{array}\tag{16}
$$

For each $t \in J ,$ a vector satisfying $r ( a ^ { t } ) = t$ and $m a ^ { t } = F ( t )$ is

$$
a ^ { t } = - t e _ { 0 , 1 } + ( t + \mathrm { s g n } t ) e _ { 0 , 3 } + \sum _ { n \geq 1 } \omega _ { | t | - 1 } ( n ) e _ { n , 1 } .\tag{17}
$$

We now verify the hypotheses of Theorem 1 for the operators in Eq. (16). Lemmas 2 and 3 establish (H1) and (H2) of Assumption 1. Lemma 4 verifies the curve and fibre conditions (H3)–(H4) and proves the exclusion condition in Eq. (9).

Lemma 2 (The coordinate pairing identity). The maps m $: \ell ^ { 1 } ( I ) \to \ell ^ { 1 } ( \mathbb { N } _ { 0 } ) \subset \ell ^ { 2 } ( \mathbb { N } _ { 0 } )$ and $E , L : \ell ^ { 1 } ( I ) \to c _ { 0 } ( I )$ are bounded. For all $a , c \in \ell ^ { 1 } ( I ) , E q s . \ ( 1 \vartheta )$ and (19) below hold.

Proof. For $a \in \ell ^ { 1 } ( I ) , \| m a \| _ { 1 } \leq \| a \| _ { 1 }$ and $\| E a \| _ { \infty } \leq 2 \| a \| _ { 1 }$ . If a has finite support, then Ea has finite support: only finitely many blocks occur, and within a block no output occurs after the last nonzero input coordinate. Approximation by finite-support vectors and the bound prove $E a \in c _ { 0 } ( I )$ . Thus L also maps $\ell ^ { 1 } ( I )$ boundedly into $c _ { 0 } ( I )$

For $a , c \in \ell ^ { 1 } ( I )$ , absolute convergence permits exchanging the double sums. In each block the diagonal terms and the two triangular of-diagonal sums give

$$
\langle E a , c \rangle + \langle E c , a \rangle = 2 \sum _ { b } \Big ( \sum _ { j } a _ { b , j } \Big ) \Big ( \sum _ { j } c _ { b , j } \Big ) = 2 ( m a , m c ) _ { \ell ^ { 2 } } .\tag{18}
$$

The absolute sum of all products is at most $\| a \| _ { 1 } \| c \| _ { 1 }$ before the factor 2, so the identity holds for all $a , c \in \ell ^ { 1 } ( I )$ . Since $m g = 0 , h = E g$ , and $r ( g ) = 0$ , Eq. (18) implies $\langle E a , g \rangle = - r ( a )$ Consequently

$$
\begin{array} { c } { { \langle L a , a \rangle = r ( a ) ^ { 2 } - \| m a \| _ { 2 } ^ { 2 } , } } \\ { { \langle L a , g \rangle = r ( a ) , } } \\ { { \langle L a - L c , a - c \rangle = ( r ( a ) - r ( c ) ) ^ { 2 } - \| m a - m c \| _ { 2 } ^ { 2 } . } } \end{array}\tag{19}
$$

Lemma 3 (The annihilator of K in $c _ { 0 } ( I ) )$ . For K in $E q . \ ( { \it 1 6 } )$

$$
\{ z \in c _ { 0 } ( I ) : \langle z , k \rangle = 0 f o r e v e r y k \in K \} = \mathbb { R } h .
$$

Proof. Let $z \in c _ { 0 } ( I )$ annihilate K. In a block $b \geq 1$ , every diference $e _ { b , j } - e _ { b , k }$ is in K. Thus z is constant in that entire block. Since $z \in c _ { 0 } ( I )$ , this constant must be zero. In block zero, diferences between indices $j , k \geq 3$ belong to K, so the common tail value is also zero. Finally, $g = e _ { 0 , 1 } - e _ { 0 , 2 }$ belongs to K, forcing $z _ { 0 , 1 } = z _ { 0 , 2 }$ . It follows that z is a scalar multiple of $h .$ Conversely, $\langle h , k \rangle = r ( k ) = 0$ proves that every multiple of h annihilates K. □

Lemma 4 (Properties of the curve and its parameter fibres). The map $s \mapsto \omega _ { s }$ from $( 0 , \infty )$ to $\ell ^ { 2 } ( \mathbb { N } )$ satisfies Eq. (7) with $\eta = \omega _ { 0 } , S = \sigma < 1 / 8$ and a Lipschitz constant strictly less than one. For every $t \in J$ , the vector $a ^ { t }$ in $E q$ . (17) belongs to $D _ { i }$ and its parameter fibre is $a ^ { t } + K$ . No $p \in \ell ^ { 1 } ( I )$ and $\tau \in [ - 1 , 1 ]$ satisfy $r ( p ) = \tau$ and $m p = ( \tau , \omega _ { 0 } )$

Proof. For each $s > 0 , \ \omega _ { s } ( n ) = 0$ whenever $n \geq s ^ { - 4 }$ . Also $0 \leq \omega _ { s } ( n ) \leq 1 / ( 4 n )$ . Thus, $W ( s ) \leq \sigma < 1 / 8$ , and dominated convergence in $\ell ^ { 2 }$ gives $\omega _ { s }  \omega _ { 0 }$ and $W ( s ) \to \sigma$ as s decreases to zero. The comparison vector $\omega _ { 0 }$ is not in $\ell ^ { 1 }$ by divergence of the harmonic series.

Since $v \mapsto \operatorname* { m a x } \{ v , 0 \}$ is 1-Lipschitz on $\mathbb { R }$ ,

$$
\| \omega _ { s } - \omega _ { q } \| _ { 2 } ^ { 2 } \leq c | s - q | ^ { 2 } , \qquad c = \frac { 1 } { 1 6 } \sum _ { n > 1 } n ^ { - 3 / 2 } < \frac { 3 } { 1 6 } < 1 .\tag{20}
$$

The strict numerical bound follows from $\begin{array} { r } { \sum _ { n > 1 } n ^ { - 3 / 2 } < 1 + \int _ { 1 } ^ { \infty } x ^ { - 3 / 2 } d x = 3 } \end{array}$ . For t, u on the same branch of J, Eq. (20) gives $\| F ( t ) - F ( \bar { u } ) \| _ { 2 } \leq | t - u |$ . For $t = 1 + s$ and $u = - 1 - q$ , the squared distance is at most $4 + c ( s - q ) ^ { 2 } \leq ( 2 + s + q ) ^ { 2 }$ , and the opposite orientation is identical. Thus F is 1-Lipschitz on all of J.

The finitely supported vector $a ^ { t }$ in Eq. (17) satisfies $r ( a ^ { t } ) = t$ and $m a ^ { t } = F ( t )$ , proving nonemptiness for every $t \in J$ . The parameter fibre $\{ a : r ( a ) = t , \ m a = F ( t ) \}$ is precisely $a ^ { t } + K$ . For every $p \in \ell ^ { 1 } ( I )$ , the vector mp is in $\ell ^ { 1 } (  { \mathbb { N } } _ { 0 } )$ . Since $\omega _ { 0 } \notin \ell ^ { 1 } ( \mathbb { N } )$ , the last assertion follows. □

The preceding lemmas establish all the hypotheses of Theorem 1, including Eq. (9). Therefore, applying that theorem to the operators in Eq. (16) gives the following counterexample, with the quantitative bounds stated below.

Theorem 2 (A counterexample on c<sub>0</sub>). Let $I = \mathbb { N } _ { 0 } \times \mathbb { N }$ and $Y = c _ { 0 } ( I )$ with its usual supremum norm and continuous dual $Y ^ { * } = \ell ^ { 1 } ( I )$ . Let $A : Y  Y ^ { * }$ and the bounded positive rank-one map $P : Y  Y ^ { * }$ be defined by Eq. (16). Then:

1. A and P are maximally monotone in $Y \times Y ^ { * }$

2. dom $P = Y$ and dom A $\neq \emptyset _ { \mathrm { : } }$ , so dom $A \cap \operatorname { i n t } ( \operatorname { d o m } P ) \neq \emptyset$

3. Every $( x , a ) \in \operatorname { g r a } A$ satisfies $\| { x } \| _ { \infty } > 1 / 2$ and $\langle x , a \rangle \geq - 2 \sigma \| x \| _ { \infty }$ , where $\sigma = \pi ^ { 2 } / 9 6 < 1 / 8$

4. Every $( x , b ) \in \mathrm { g r a } ( A + P )$ satisfies $\langle x , b \rangle > 1 - \sigma > 7 / 8$

5. $( 0 , 0 )$ is monotonically related to the entire graph of $A + P$ and does not belong to that graph. Thus $A + P$ is not maximally monotone.

Proof. (1) and (2). Maximality and the domain intersection condition. Apply Theorem 1 with $X = Y , M = m , H _ { 0 } = \ell ^ { 2 } ( \mathbb { N } ) , \omega ( s ) = \omega _ { s } , \eta = \omega _ { 0 }$ and $S = \sigma$ . Lemmas 2–4, together with $g \neq 0$ and $m g = 0$ from Eq. (13), verify Assumption 1. The last assertion of Lemma 4 gives Eq. (9), so the theorem proves maximal monotonicity of both A and P.

Since $\| g \| _ { 1 } = 2$ , we have $\| \boldsymbol { P } \| \leq 4$ and $P e _ { 0 , 1 } = g \neq 0$ . The vector $a ^ { 2 } = - 2 e _ { 0 , 1 } + 3 e _ { 0 , 3 }$ in Eq. (17) belongs to D and gives $L a ^ { 2 } = - 6 e _ { 0 , 1 } - 8 e _ { 0 , 2 } - 3 e _ { 0 , 3 }$ . Hence $L a ^ { 2 } \in$ dom A, and dom $P = Y$ yields dom A ∩ int(dom $P ) \neq \emptyset$

(3). Bounds on gra A. For every $a \in D , t = r ( a )$ satisfies $| t | > 1$ . By Eq. (19) and $\| g \| _ { 1 } = 2$

$$
\begin{array} { l } { \| L a \| _ { \infty } \geq | t | / 2 > 1 / 2 , } \\ { \langle L a , a \rangle = t ^ { 2 } - 1 - W ( | t | - 1 ) > - \sigma . } \end{array}\tag{21}
$$

Eq. (21) gives $A ( 0 ) = \emptyset$ and, for every $a \in D$

$$
\frac { \operatorname* { m a x } \{ 0 , - \langle L a , a \rangle \} } { \| L a \| _ { \infty } } \leq \frac { \sigma } { \| L a \| _ { \infty } } < 2 \sigma .
$$

Hence $\langle L a , a \rangle \geq - 2 \sigma \| L a \| _ { \infty }$ for every $a \in D$ , and taking the supremum gives $\mathcal { V } ( \mathrm { g r a } A ) \leq 2 \sigma <$ $1 / 4$

(4) and (5). Nonmaximality of $A + P$ . To verify nonmaximality of $A + P .$ , take an arbitrary point of $\mathrm { g r a } ( A + P )$ . By Eqs. (16) and (19), it has the form $( L a , a + t g )$ with $a \in D$ and $t = r ( a )$ Thus Eq. (19) gives

$$
\langle L a , a + t g \rangle = 2 t ^ { 2 } - 1 - W ( | t | - 1 ) > 1 - \sigma > 7 / 8 .\tag{22}
$$

Eq. (22) shows that $( 0 , 0 )$ is monotonically related to every point of $\mathrm { g r a } ( A + P )$ . Since A and P are monotone, $\operatorname { g r a } ( A + P ) \cup \{ ( 0 , 0 ) \}$ is monotone. Eq. (21) gives $0 \not \in \mathrm { d o m } ( A + P )$ , so this union strictly contains $\mathrm { g r a } ( A + P )$ . Therefore $A + P$ is not maximally monotone. □

## 4.2 A counterexample on standard $\ell ^ { 1 }$

Next, we transfer the counterexample of Theorem 2 from $Y = c _ { 0 } ( I )$ to standard $\ell ^ { 1 }$ through a bounded linear surjection.

Let $Z = \ell ^ { 1 } ( \mathbb { N } )$ and $Z ^ { * } = \ell ^ { \infty } ( \mathbb { N } )$ , with their usual pairing. To construct the required surjection, fix an enumeration $\left( q _ { n } \right)$ of all finitely supported rational vectors in the closed unit ball of $Y$

Definition 9 (The operators on standard $\ell ^ { 1 } )$ . Define

$$
\begin{array} { r l r } {  { Q u = \sum _ { n \geq 1 } u _ { n } q _ { n } , } } \\ & { } & \\ & { } & { \displaystyle ( Q ^ { * } a ) _ { n } = \langle q _ { n } , a \rangle , } \\ & { } & { \mathrm { g r a } T = \{ ( u , Q ^ { * } a ) : a \in D , \ Q u = L a \} , } \\ & { } & { \quad f = Q ^ { * } g , } \\ & { } & { \quad B u = \langle u , f \rangle f . } \end{array}\tag{23}
$$

The next lemma provides the bounded preimages needed in Lemma 1, and no bounded linear right inverse of $Q$ is required.

Lemma 5 (The specified quotient onto $c _ { 0 } )$ . The map $Q : Z = \ell ^ { 1 } ( \mathbb { N } ) \to Y = c _ { 0 } ( I )$ of Eq. (23) is a bounded surjection with $\| Q \| = 1$ . Its adjoint satisfies $\| Q ^ { * } a \| _ { \infty } = \| a \| _ { 1 }$ for every $a \in \ell ^ { 1 } ( I )$ For the fixed enumeration $\left( q _ { n } \right)$ , a map $\mathcal { R } : Y  Z$ can be specified with

$$
\begin{array} { r } { Q \mathcal { R } ( x ) = x , \qquad \| \mathcal { R } ( x ) \| _ { 1 } \leq 2 \| x \| _ { \infty } . } \end{array}
$$

Proof. Finitely supported rational vectors in the closed unit ball of $Y$ are countable and dense in that ball: truncate a $c _ { 0 }$ vector and approximate each retained coordinate by a rational inside $[ - 1 , 1 ]$ . Fix an enumeration of all these vectors as $\left( q _ { n } \right)$ $\mathrm { F o r } \ u \in Z = \ell ^ { 1 } ( \mathbb { N } )$ , the sum defining Qu converges absolutely in the Banach space $Y ,$ and $\| Q u \| _ { \infty } \leq \| u \| _ { 1 }$ . Coordinate unit vectors occur among the $q _ { n }$ , so $\| Q \| = 1$

For an arbitrary $x \in Y$ , set $r _ { 0 } = x$ . If $r _ { k } \neq 0$ , choose the least index $n _ { k }$ for which $q _ { n _ { k } }$ satisfies $\| q _ { n _ { k } } - r _ { k } / \| r _ { k } \| _ { \infty } \| _ { \infty } < 1 / 2$ and set

$$
c _ { k } = \| r _ { k } \| _ { \infty } , \qquad r _ { k + 1 } = r _ { k } - c _ { k } q _ { n _ { k } } .\tag{24}
$$

If a residual vanishes, stop. Otherwise $\| r _ { k } \| _ { \infty } \leq 2 ^ { - k } \| x \| _ { \infty }$ and $\textstyle \sum _ { k } c _ { k } \leq 2 \| x \| _ { \infty }$ . Define $u _ { n } =$ $\scriptstyle \sum _ { k : n _ { k } = n } c _ { k }$ . Repeated indices are combined. Positivity of the $c _ { k }$ and summability give $u \in \ell ^ { 1 }$ with $\begin{array} { r } { \tilde { \| \boldsymbol { u } } \| _ { 1 } = \sum _ { k } c _ { k } \leq 2 \| \boldsymbol { x } \| _ { \infty } } \end{array}$ . Telescoping Eq. (24), then absolute convergence under regrouping, proves

$$
Q u = x , \qquad \| u \| _ { 1 } \leq 2 \| x \| _ { \infty } .\tag{25}
$$

For $x = 0$ , use $u = 0$ . This proves surjectivity with the asserted lifting bound.

Absolute convergence gives $\langle u , Q ^ { * } a \rangle = \langle Q u , a \rangle$ and $\| Q ^ { * } a \| _ { \infty } \leq \| a \| _ { 1 }$ . For each finite subset of $I ,$ its sign vector for a is a finitely supported rational vector of norm at most one and occurs in the enumeration. Taking these finite sign tests proves

$$
\| Q ^ { * } a \| _ { \infty } = \| a \| _ { 1 } \qquad ( a \in \ell ^ { 1 } ( I ) ) .\tag{26}
$$

In particular, $Q ^ { * }$ is injective. The least-index choices in $\operatorname { E q }$ . (24) specify

$$
\mathcal { R } ( x ) = \sum _ { k } c _ { k } e _ { n _ { k } } , \qquad \mathcal { R } ( 0 ) = 0 .
$$

The series converges in $\ell ^ { 1 }$ by the bound on $\sum _ { k } c _ { k }$ , and Eq. (25) gives both asserted properties. □

By Lemma 5, $Q \mathcal { R } ( L a ) = L a .$ , so the solutions of $Q u = L a$ are exactly $u = \mathscr { R } ( L a ) + v$ with $v \in$ ker Q. Substituting this expression into Eq. (23) gives

$$
\mathrm { g r a } T = \{ ( \mathcal { R } ( L a ) + v , Q ^ { * } a ) : a \in D , v \in \ker Q \} .\tag{27}
$$

This formula includes every preimage of each point in dom A. We can therefore apply Lemma 1 to the counterexample in Theorem 2. Together with the pairing identities below, this yields the following counterexample on standard $\ell ^ { 1 }$

Theorem 3 (A counterexample on standard $\ell ^ { 1 } )$ . On $Z ~ = ~ \ell ^ { 1 } ( \mathbb { N } )$ with its usual norm and continuous dual $Z ^ { * } = \ell ^ { \infty } ( \mathbb { N } )$ , let $T : Z \ni Z ^ { * }$ and $B : Z \to Z ^ { * }$ be defined by Eq. (23). Then:

1. $T$ is maximally monotone in the full dual pair $Z \times Z ^ { * }$

2. B is nonzero, bounded, positive, rank-one, and maximally monotone, with dom $B = Z$ . In particular, dom $T \cap$ int(dom $B ) \neq \emptyset$

3. $T ( 0 ) = \mathcal { O } ,$ and every $( u , b ) \in \mathrm { g r a } T$ satisfies $\| u \| _ { 1 } > 1 / 2$ and $\langle u , b \rangle \geq - 2 \sigma \Vert u \Vert _ { 1 }$ <sub>1</sub>. Thus $\mathcal { V } ( \mathrm { g r a } T ) \le 2 \sigma < 1 / 4$

4. Every $( u , c ) \in \mathrm { g r a } ( T + B )$ satisfies $\langle u , c \rangle > 1 - \sigma > 7 / 8$

5. (0, 0) belongs to the monotone polar of $\mathrm { g r a } ( T + B )$ and does not belong to $\mathrm { g r a } ( T + B )$ Hence $T + B$ is not maximally monotone.

Proof. (1) and (2). Maximality and the domain intersection condition. Eq. (27) identifies T with $Q ^ { * } A Q$ , including every preimage under $Q .$ . Lemma $5$ supplies the preimage bound with $C _ { Q } = 2 $ . Since A is maximally monotone by Theorem 2, Lemma 1 proves maximal monotonicity of $T$ in $Z \times Z ^ { * } = \ell ^ { 1 } \times \ell ^ { \infty }$

Let $f = Q ^ { * } g$ . By Eq. (26), $\| f \| _ { \infty } = \| g \| _ { 1 } = 2$ , so f is nonzero. The map $B : u \mapsto \langle u , f \rangle f$ is nonzero, linear, bounded with norm at most 4, positive, and rank-one. Moreover, for every $u \in Z .$

$$
B u = \langle u , Q ^ { * } g \rangle Q ^ { * } g = Q ^ { * } ( \langle Q u , g \rangle g ) = Q ^ { * } ( P ( Q u ) ) .
$$

Thus $B = Q ^ { * } P Q$ . Maximal monotonicity of P in Theorem 2 and Lemma 1 give maximal monotonicity of B. Its domain is all of $Z ,$ so int $( \operatorname { d o m } B ) = Z$

The vector $q _ { \circ } = - ( 3 / 4 ) e _ { 0 , 1 } - e _ { 0 , 2 } - ( 3 / 8 ) e _ { 0 , 3 }$ occurs as $q _ { n _ { 0 } }$ . For $u _ { \circ } = 8 e _ { n _ { 0 } } , Q u _ { \circ } = L a ^ { 2 }$ . Thus $( u _ { \circ } , Q ^ { * } a ^ { 2 } ) \in \mathrm { g r a } T$ , and dom T ∩ int(dom $B ) \neq \emptyset$

(3). Bounds on gra T. For $( u , Q ^ { * } a ) \in \mathrm { g r a } T$ , we have $Q u = L a$ . Thus the adjoint identity, $\| Q \| = 1$ and Eq. (21) transfer the bounds for A to $T .$ Writing $t = r ( a )$ , we obtain

$$
\begin{array} { c } { { \langle u , Q ^ { * } a \rangle = \langle L a , a \rangle = t ^ { 2 } - 1 - W ( | t | - 1 ) > - \sigma , } } \\ { { | | u | | _ { 1 } \geq | | Q u | | _ { \infty } = | | L a | | _ { \infty } > 1 / 2 . } } \end{array}\tag{28}
$$

Thus $T ( 0 ) = \emptyset$ and $\langle u , Q ^ { * } a \rangle \geq - 2 \sigma \Vert u \Vert _ { 1 }$ <sub>1</sub> on the entire graph, including every ker $Q$ translate. Consequently, $\mathcal { V } ( \mathrm { g r a } T ) \le 2 \sigma < 1 / 4$

(4) and (5). Nonmaximality of $T + B$ . For $( u , Q ^ { * } a ) \in \mathrm { g r a } T$ , we have $Q u = L a$ and $B u =$ $Q ^ { * } ( P ( L a ) )$ . The adjoint identity therefore transfers Eq. (22) to $T + B$ . With $t = r ( a )$

$$
\begin{array} { r l } & { \langle u , Q ^ { * } a + B u \rangle = \langle L a , a + P ( L a ) \rangle } \\ & { \qquad = 2 t ^ { 2 } - 1 - W ( | t | - 1 ) > 1 - \sigma > 7 / 8 . } \end{array}\tag{29}
$$

Eq. (29) shows that $( 0 , 0 )$ is monotonically related to every point of $\mathrm { g r a } ( T + B )$ . Since $T$ and B are monotone, gr $\mathsf { i } ( T + B ) \cup \{ ( 0 , 0 ) \}$ is monotone. Since $T ( 0 ) = \emptyset$ , this union strictly contains $\mathrm { g r a } ( T + B )$ . Therefore, $T + B$ is not maximally monotone. □

## 5 Conclusions

In this paper, we constructed counterexamples to Rockafellar’s sum conjecture. Specifically, we established a general construction theorem that computes the monotone polar of the proposed graphs, characterizes their maximal monotonicity and identifies when adding an everywheredefined positive rank-one operator produces a nonmaximal sum. We verified its hypotheses for explicit operators on $c _ { 0 }$ and transferred the resulting pair to standard $\ell ^ { 1 }$ through a bounded linear surjection. In both pairs, the two operators are maximally monotone and satisfy the original interior-domain condition, and their sum is not maximally monotone. We also proved a finite radial bound for the first operator in each pair.

## Use of AI

The author supplied the prior results and unsuccessful approaches, including some constructions based on blockwise triangular operators, together with a framework for positive rank-one perturbations, and guided their further development to investigate how maximality can fail under addition. The author also supplied obstruction results showing that a finite radial bound on a monotone graph need not survive maximal extension. Through iterative discussions of these materials, the author and GPT-5.6 Sol developed a general construction theorem characterizing maximality within this framework and identifying when an everywhere-defined rank-one monotone perturbation yields a nonmaximal sum. Under the author’s direction and using the aforementioned results as inputs, GPT-5.6 Sol worked out the detailed constructions and proof arguments, and the transfer from $c _ { 0 }$ to standard $\ell ^ { 1 }$

## Code availability.

The Lean formalization of the counterexample on $c _ { 0 }$ and the general pullback lemma, together with the pinned dependencies and build instructions, is available at https://github.com/ Weifeng-Yang/Rockafellar-sum-problem.

## References

[1] R. T. Rockafellar, “On the maximality of sums of nonlinear monotone operators,” Transactions of the American Mathematical Society, vol. 149, pp. 75–88, 1970.

[2] S. Simons, From Hahn–Banach to Monotonicity, 2nd ed., ser. Lecture Notes in Mathematics. Springer, 2008, vol. 1693.

[3] R. I. Bot¸, O. Bueno, and S. Simons, “The Rockafellar conjecture and type (FPV),” Set-Valued and Variational Analysis, vol. 24, pp. 381–385, 2016.

[4] L. Yao, “The sum of a maximal monotone operator of type (FPV) and a maximal monotone operator with full domain is maximal monotone,” Nonlinear Analysis: Theory, Methods & Applications, vol. 74, pp. 6144–6152, 2011.

[5] J. M. Borwein and L. Yao, “Maximality of the sum of a maximally monotone linear relation and a maximally monotone operator,” Set-Valued and Variational Analysis, vol. 21, pp. 603–616, 2013.

[6] M. D. Voisei, “A sum theorem for (FPV) operators and normal cones,” Journal of Mathematical Analysis and Applications, vol. 371, pp. 661–664, 2010.

[7] A. Verona and M. E. Verona, “On Rockafellar’s sum theorem in general Banach spaces,” Journal of Convex Analysis, vol. 29, no. 2, pp. 381–390, 2022.

[8] H. H. Bauschke, J. M. Borwein, X. Wang, and L. Yao, “Construction of pathological maximally monotone operators on non-reflexive Banach spaces,” Set-Valued and Variational Analysis, vol. 20, pp. 387–415, 2012, example numbering refers to the authors’ August 6, 2011 version.

[9] S. Fitzpatrick, “Representing monotone operators by convex functions,” in Workshop/Miniconference on Functional Analysis and Optimization (Canberra, 1988), ser. Proc. Centre Math. Anal. Austral. Nat. Univ. Canberra: Austral. Nat. Univ., 1988, vol. 20, pp. 59–65.

[10] R. S. Burachik and B. F. Svaiter, “Maximal monotone operators, convex functions and a special family of enlargements,” Set-Valued Analysis, vol. 10, pp. 297–316, 2002.

[11] J.-P. Penot, “The relevance of convex analysis for the study of monotonicity,” Nonlinear Analysis: Theory, Methods & Applications, vol. 58, pp. 855–871, 2004.

[12] M. Marques Alves and B. F. Svaiter, “A new qualification condition for the maximality of the sum of maximal monotone operators in general Banach spaces,” Journal of Convex Analysis, vol. 19, pp. 575–589, 2012.

[13] ——, “Brønsted–rockafellar property and maximality of monotone operators representable by convex functions in non-reflexive Banach spaces,” Journal of Convex Analysis, vol. 15, no. 4, pp. 693–706, 2008.

[14] M. D. Voisei and C. Z˘alinescu, “Strongly-representable monotone operators,” J. Convex Anal., vol. 16, no. 3–4, pp. 1011–1033, 2009.

[15] J.-E. Mart´ınez-Legaz and B. F. Svaiter, “Monotone operators representable by l.s.c. convex functions,” Set-Valued Analysis, vol. 13, pp. 21–46, 2005.

[16] H. H. Bauschke, D. A. McLaren, and H. S. Sendov, “Fitzpatrick functions: inequalities, examples, and remarks on a problem by S. Fitzpatrick,” J. Convex Anal., vol. 13, no. 3–4, pp. 499–523, 2006.

[17] O. Bueno, J.-E. Mart´ınez-Legaz, and B. F. Svaiter, “On the monotone polar and representable closures of monotone operators,” Journal of Convex Analysis, vol. 21, pp. 495–505, 2014.

[18] A. Verona and M. E. Verona, “Regular maximal monotone multifunctions and enlargements,” Journal of Convex Analysis, vol. 16, pp. 1003–1009, 2009.

[19] R. T. Rockafellar, “On the maximal monotonicity of subdiferential mappings,” Pacific Journal of Mathematics, vol. 33, pp. 209–216, 1970.