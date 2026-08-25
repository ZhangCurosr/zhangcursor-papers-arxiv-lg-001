# Mirror descent algorithms with logarithmic barriers

Alberto De Marchi<sup>∗</sup> Yura Malitsky<sup>†</sup> Adrien B. Taylor<sup>‡</sup>

## Abstract

This work derives convergence guarantees for mirror descent and proximal mirror descent algorithms when a logarithmic barrier is used as a distance-generating function. Standard approaches cannot be applied when the solution lies on the boundary, where the Bregman divergence blows up. We show that, in a specific setting, both methods enjoy an $O ( \log k / k )$ rate, which is also tight. In addition, our contributions include: (i) a new technique for handling the blow-up; (ii) a resolution of a gap in the theory of relative smoothness; and (iii) a comparison of the proposed approach with interior-point methods.

## 1 Introduction

This work considers the convex minimization problem

$$
\operatorname* { m i n } _ { x \in \mathbb { R } ^ { d } } f ( x ) \quad { \mathrm { s u b j e c t ~ t o } } \quad x \in { \mathcal { C } } \cap { \mathcal { S } } ,\tag{1}
$$

where $\mathcal { C } , \mathcal { S } \subset \mathbb { R } ^ { d }$ are nonempty closed convex sets, and $f \colon \mathcal { C } \cap \mathcal { S } \mathrm { ~  ~ } \mathbb { R }$ is a convex function. Here we decompose the constraint set into a simple set $s ,$ which is treated explicitly in the optimization problem, and a complicated set ${ \mathcal { C } } ,$ which is handled via the logarithmic barrier $h \colon { \mathcal { C } } \to \mathbb { R } \cup \{ + \infty \}$ . We consider this function h as a mirror, or distance-generating, function that defines the Bregman divergence $D _ { h } ( y , x ) : = h ( y ) - h ( x ) - \langle \nabla h ( x ) , y - x \rangle$ . With this, we can formulate the mirror descent algorithm

$$
\begin{array} { r } { x _ { k } = \underset { x \in \mathcal { S } } { \mathrm { a r g m i n } } \left\{ f ( x _ { k - 1 } ) + \langle \nabla f ( x _ { k - 1 } ) , x - x _ { k - 1 } \rangle + \frac { 1 } { \alpha _ { k } } D _ { h } ( x , x _ { k - 1 } ) \right\} , } \end{array}\tag{MD}
$$

or its proximal, or implicit, version

$$
\begin{array} { r } { x _ { k } = \underset { x \in \mathcal { S } } { \mathrm { a r g m i n } } \left. f ( x ) + \frac { 1 } { \alpha _ { k } } D _ { h } ( x , x _ { k - 1 } ) \right. , } \end{array}\tag{proxMD}
$$

where appropriate assumptions are needed to ensure that the iterates $( x _ { k } )$ are well-defined and feasible. We postpone the formal definitions and assumptions on $( f , h , \mathcal { C } )$ until Section 2. Mirror descent (MD) was originally introduced by [17] and later reformulated and popularized in its modern form by [2].

For the sake of presentation, the only thing one needs to know about a logarithmic barrier of C is that it is well-defined on int(C) and blows up on its boundary. Logarithmic barriers have not been particularly popular in Bregman-type methods; the exceptions only confirm our point, and we discuss this further later on. The main reason is that the classical analysis cannot handle them. Indeed, for either (MD) or (proxMD) with a fixed stepsize $\alpha _ { k } = \alpha$ for simplicity, and under any favorable conditions on $f ,$ the standard sublinear convergence rate is of the type

$$
f ( x _ { k } ) - f _ { \star } \leqslant \frac { D _ { h } ( x _ { \star } , x _ { 0 } ) } { \alpha k } ,
$$

where the optimal value $f _ { \star }$ is attained at some $x _ { \star }$ . The issue becomes apparent since, for most problems, we expect x<sub>⋆</sub> to lie on the boundary of ${ \mathcal { C } } ,$ in which case

$$
D _ { h } ( x _ { \star } , x _ { 0 } ) = h ( x _ { \star } ) - h ( x _ { 0 } ) - \langle \nabla h ( x _ { 0 } ) , x _ { \star } - x _ { 0 } \rangle = + \infty .
$$

The aim of this paper is to resolve this issue. We derive convergence rates for both (MD) and $\left( \mathrm { p r o x M D } \right)$ and show their tightness. We believe this is of interest in its own right, but along the way we also resolve a long-standing issue in the theory of relative smoothness and compare the proposed approach with interior-point methods.

Relative smoothness in mirror descent. As mentioned above, there are some exceptions where log-barriers have been used. One example is the relative smoothness condition [1, 3, 13]. We say that $f$ is L-relatively smooth with respect to h if

$$
D _ { f } ( x , y ) \leqslant L D _ { h } ( x , y ) \quad \forall x , y \in \operatorname { d o m } h .\tag{2}
$$

Under this condition, the analysis of mirror descent (MD) with $\begin{array} { r } { \alpha _ { k } = \frac { 1 } { L } } \end{array}$ largely follows the proof for gradient descent and yields the convergence bound $\begin{array} { r } { f ( x _ { k } ) - f _ { \star } \leqslant \frac { L D _ { h } ( x _ { \star } , x _ { 0 } ) } { k } } \end{array}$ , which, as already mentioned, becomes vacuous whenever $x _ { \star } \in \mathrm { b d } ( \mathcal { C } )$ . A log-barrier is often a convenient choice of h that ensures (2). Below, we provide two illustrative examples:

• D-optimal design [13, Section 2.2]. Let $H \in \mathbb { R } ^ { m \times d }$ with d > m and $\operatorname { r a n k } ( H ) = m$ , and $f ( x ) =$ $- \log \operatorname* { d e t } ( H \operatorname { d i a g } ( x ) H ^ { \top } ) , \mathcal { C } = \mathbb { R } _ { + } ^ { d } , \mathcal { S } = \{ x \colon \langle \mathbf { 1 } , x \rangle = 1 \}$ , and $\begin{array} { r } { \boldsymbol { h } ( \boldsymbol { x } ) = - \sum _ { i = 1 } ^ { d } \log \boldsymbol { x } _ { i } } \end{array}$ . We know that $f$ is 1-relatively smooth with respect to h.

• Poisson linear inverse problems [1, Section 5.2]. Let $\begin{array} { r } { f ( x ) = D _ { \phi } ( b , A x ) , \phi ( x ) = \sum _ { i } x _ { i } \log x _ { i } , S = \mathbb { R } ^ { d } . } \end{array}$ $\begin{array} { r } { \mathcal { C } = \mathbb { R } _ { + } ^ { d } , \mathrm { a n d } h ( x ) = - \sum _ { i = 1 } ^ { d } \log x _ { i } } \end{array}$ . We know that f is ∥b∥ -relatively smooth with respect to $h .$

Although these problems were among the motivations for introducing condition (2), until now there has been no theory providing quantitative convergence guarantees for (MD) in these settings.

Interior-point methods. The only area of optimization where log-barriers have found a stronghold, with rigorous convergence rates and complexity guarantees, is, of course, interior-point methods (IPMs); see, for instance, the classical [18] or [11] for historical perspectives. Thus, it was only natural for us to follow the same path and compare the Bregman approach with the IPM approach. While the usual presentation of IPMs does not closely follow the Bregman setting, the latter is more convenient for us.

In fact, classical IPMs can be viewed as lazy versions of (proxMD), in which the iterates are computed without updating the prox-center:

$$
\begin{array} { r } { x _ { k } = \underset { x \in S } { \mathrm { a r g m i n } } \left. f ( x ) + \frac { 1 } { \alpha _ { k } } D _ { h } ( x , x _ { 0 } ) \right. , } \end{array}\tag{IPM}
$$

where $h$ is a log-barrier for the set ${ \mathcal { C } } ,$ while $s$ typically represents afine constraints, and $x _ { 0 }$ is a specific point called the analytic center of ${ \mathcal { C } } \cap { \mathcal { S } } ;$ see, e.g., [18, Definition 5.3.3]. Classical examples include:

• linear programming: $\mathcal { C } = \left\{ x \in \mathbb { R } ^ { d } : a _ { i } ^ { \top } x \leqslant b _ { i } \right.$ for $i = 1 , \ldots , n \}$ with $a _ { 1 } , \dots , a _ { n } \in \mathbb { R } ^ { d } , b _ { 1 } , \dots , b _ { n } \in \mathbb { R }$ and $\begin{array} { r } { h ( x ) = - \sum _ { i } \log ( b _ { i } - \dot { a } _ { i } ^ { \top } x ) } \end{array}$

• semidefinite programming: $\begin{array} { r } { \mathcal { C } = \left\{ x \in \mathbb { R } ^ { d } : S ( x ) = \sum _ { i } x _ { i } A _ { i } \in \mathbb { S } _ { + } ^ { m } \right\} } \end{array}$ with $\mathbb { S } _ { + } ^ { m }$ the set of m×m positive semidefinite matrices, $A _ { 1 } , \ldots , A _ { d } \in \mathbb { S } ^ { m }$ symmetric matrices, and $h ( x ) = - \log \operatorname* { d e t } ( S ( x ) )$

Since (IPM) can be viewed as one iteration of (proxMD), one would expect the corresponding bounds to difer as follows:

$$
f ( x _ { k } ) - f _ { \star } \leqslant { \frac { D _ { h } ( x _ { \star } , x _ { 0 } ) } { \alpha _ { k } } } \qquad { \mathrm { w h e r e a s } } \qquad f ( x _ { k } ) - f _ { \star } \leqslant { \frac { D _ { h } ( x _ { \star } , x _ { 0 } ) } { \sum _ { i = 1 } ^ { k } \alpha _ { i } } }
$$

for (IPM) and (proxMD), respectively. Here the reader should not confuse the sequences $\left( \alpha _ { k } \right)$ in (proxMD) and (IPM) — they need not be the same.

One may therefore again be disappointed by the same issue as for mirror descent schemes: solutions typically lie on the boundary, where the guarantee explodes under classical choices of $h .$ This may also appear to be at odds with IPM-type results, where, when h is a ν-self-concordant barrier, one can obtain

$$
f ( x _ { k } ) - f _ { \star } \leqslant \frac { \nu } { \alpha _ { k } } ,
$$

see [18, Theorem 5.3.10]. Perhaps counterintuitively, in Section 4.3 we show that (proxMD) can be viewed as a direct alternative to IPMs, albeit with a slightly worse total complexity.

Further context. Related strategies that exploit the advantages of both approaches are numerous, which is natural given that IPMs and proximal-point methods are among the most iconic algorithmic tools in optimization. Indeed, augmented Lagrangian (AL) schemes [21] are proximal-point methods with $h = \textstyle { \frac { 1 } { 2 } } \| \cdot \| ^ { 2 }$ on dual problems, and modified barrier methods [19] are similar to AL (dual proximalpoint) but use Bregman geometries instead; see, e.g., [10, 23]. Bregman-proximal algorithms have also been used with regularized barriers and exponentially growing stepsizes [6, 25]. That said, to the best of our knowledge, such Bregman proximal methods were generally developed with entropy-like barriers that do not blow up at the boundary, as otherwise the complexity guarantees become void. Finally, another motivation for this study is that combining augmented Lagrangian and interior-point techniques does yield eficient solution methods; $\mathrm { s e e , e . g . }$ the quadratic programming solvers developed in [4, 20].

Roadmap. The most interesting part of the paper is the convergence result (Theorem 1 in Section 3) and the construction of the example showing that this rate is tight (Section 5). The proof for (MD) mainly follows that for (proxMD) but is slightly more technical (Theorem 2 in Section 4.1). As the proof of Theorem 1 is rather tedious, it may be simpler to gain an initial understanding by reading the informal analysis of mirror flow in Section 4.2. This part can also be skipped entirely. Finally, Section 6 presents a comparison of (proxMD) with IPMs, and is entirely independent of the previous main sections.

Warnings. Throughout the paper, we almost always remain agnostic about how the subproblems in (MD) and (proxMD) are solved. The only exception is the section on IPMs, where a more refined discussion is necessary.

We use $\nabla h ( x )$ to denote a chosen subgradient of h at $x ,$ with the choice clear from the context. For simplicity, the reader may assume that h is diferentiable. However, our main result (Theorem 1) uses only convexity of f and h and the log-barrier structure of $h .$

## 2 Preliminaries

Assumption 1 (Basic convex optimization setting). The sets $\mathcal { C } , \mathcal { S } \subset \mathbb { R } ^ { d }$ are nonempty, closed, and convex. The function $f \colon { \mathcal { C } } \cap { \mathcal { S } } \to \mathbb { R }$ is convex. The optimal value $f _ { \star } : = \operatorname* { m i n } _ { x \in \mathcal { C } \cap \mathcal { S } } f ( x )$ is finite and is attained at some $x _ { \star } \in \mathcal { C } \cap \mathcal { S }$

Later, when discussing (MD), we add more structural assumptions on $f \colon$ diferentiability and relative smoothness.

A nonempty convex set C always has a nonempty relative interior, $\mathrm { c i } ( { \mathcal { C } } ) \neq { \mathcal { D } } . ^ { 1 }$ We write rbd(C) for the relative boundary of C, that is ${ \mathrm { r b d } } ( { \mathcal { C } } ) = { \mathcal { C } } \setminus { \mathrm { r i } } ( { \mathcal { C } } )$

Logarithmic barrier. Suppose that the set C is parametrized by the convex function $\phi \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ as

$$
\phi ( x ) < 0 \mathrm { o n } \mathrm { r i } ( { \mathcal { C } } ) \quad { \mathrm { a n d } } \quad \phi ( x ) = 0 \mathrm { o n ~ r b d } ( { \mathcal { C } } ) .
$$

Then for $\nu > 0$ , we call $h ( x ) = - \nu \log ( - \phi ( x ) )$ a ν-log-barrier of the set $\mathcal { C } .$

Alternatively, one could say that h is a ν-log-barrier of C if it is (i) ν-exp-concave, namely the map $\begin{array} { r } { x \mapsto \exp ( { - \frac { h ( x ) } { \nu } } ) } \end{array}$ is concave, and $( \operatorname { i i } ) \ h ( x ) \to + \infty$ whenever $x  \mathrm { r b d } ( \mathcal { C } )$ The exp-concavity brings a useful characterization of h that is a strict improvement upon convexity of $h .$ . Note that this is also one of the key properties in the definition of ν-self-concordant barriers [18, Section 5.3].

Lemma 1 (Characterization of exp-concavity). Let $h : { \mathcal { C } } \to \mathbb { R } \cup \{ + \infty \}$ be closed, convex, proper, and ν-exp-concave. Then the following statements hold:

(i) For any $x \in$ dom h, $y \in { \mathcal { C } }$ and $s _ { x } ^ { h } \in \partial h ( x )$

$$
\nu \exp \left( \frac { h ( x ) - h ( y ) } { \nu } \right) + \langle s _ { x } ^ { h } , y - x \rangle \leqslant \nu .\tag{3}
$$

(ii) $I f$ in $\langle \cos h ) \neq \emptyset$ and h is twice continuously diferentiable on $i t ,$ then

$$
\nabla ^ { 2 } h ( x ) \succcurlyeq \frac { 1 } { \nu } \nabla h ( x ) \nabla h ( x ) ^ { \top } \qquad \forall x \in \mathrm { i n t } ( \mathrm { d o m } h ) .
$$

To see that the exp-concavity is a strict improvement over convexity, it sufices to apply $1 + t \leqslant e ^ { t }$ in (3). The parameter ν in the definition is important: roughly speaking, it measures the complexity of the set C. For illustration, consider the following example. Let C be given by

$$
{ \mathcal { C } } = \{ x \colon \phi _ { i } ( x ) \leqslant 0 \quad { \mathrm { f o r ~ } } i = 1 , \ldots , m \} ,
$$

where each $\phi _ { i }$ is convex and diferentiable. We have at least two possibilities for introducing a log-barrier. We may consider the barrier $\begin{array} { r } { h ( x ) = - \log \left( - \operatorname* { m a x } _ { i \in [ m ] } \phi _ { i } ( x ) \right) } \end{array}$ , which is simple but nondiferentiable. Alternatively, we may consider

$$
h ( x ) = - \sum _ { i = 1 } ^ { m } \log ( - \phi _ { i } ( x ) ) = - m \log ( - \Phi ( x ) ) , \quad \mathrm { w h e r e } \quad \Phi : \mathcal { C } \to \mathbb { R } : \Phi ( x ) = - \left( \prod _ { i = 1 } ^ { m } ( - \phi _ { i } ( x ) ) \right) ^ { 1 / m }
$$

which looks more intimidating but is diferentiable. Here we used the fact that the geometric mean of concave functions is concave on $\mathcal { C } .$

Assumption 2 (Well-posedness of the Bregman subproblems). For every stepsize $\alpha > 0$ and every $y \in \operatorname { r i } ( \mathcal { C } )$ , the Bregman subproblems associated with (proxMD) and (MD),

$$
\operatorname* { m i n } _ { x \in { \mathcal S } } \left\{ \alpha f ( x ) + h ( x ) - \langle \nabla h ( y ) , x \rangle \right\} \qquad { a n d } \qquad \operatorname* { m i n } _ { x \in { \mathcal S } } \left\{ \langle \alpha \nabla f ( y ) - \nabla h ( y ) , x \rangle + h ( x ) \right\} .
$$

admit minimizers. $B y$ construction, solutions to these problems automatically lie in $\operatorname { r i } ( \mathcal { C } )$

Note that Assumption 2 can be guaranteed in several standard ways, for example by coercivity of the subproblem objectives or by compactness of the feasible set $\mathcal { C } \cap \mathcal { S }$ . We keep it as an explicit assumption in order not to obscure the convergence analysis with existence issues.

Finally, the function h defines the Bregman divergence

$$
D _ { h } ( y , x ) = h ( y ) - h ( x ) - \langle \nabla h ( x ) , y - x \rangle , \qquad \forall x , y \in \operatorname { r i } ( \mathcal { C } ) .
$$

Similarly, we define $D _ { f }$ , which is used in Section 4.1.

A technical lemma. The following standard bound will be needed.

Lemma 2. Let $B \geqslant A > 0$ . Then

$$
\left( \sqrt { B } - \sqrt { A } \right) ^ { 2 } \leqslant \frac { B - A } { 4 } \log \left( \frac { B } { A } \right) .
$$

Proof. By the Cauchy-Schwarz inequality, we have

$$
{ \sqrt { B } } - { \sqrt { A } } = { \frac { 1 } { 2 } } \int _ { A } ^ { B } { \frac { 1 } { \sqrt { t } } } \mathrm { d } t \leqslant { \frac { 1 } { 2 } } \left( \int _ { A } ^ { B } 1 ^ { 2 } \mathrm { d } t \cdot \int _ { A } ^ { B } { \frac { 1 } { t } } \mathrm { d } t \right) ^ { 1 / 2 } = { \frac { 1 } { 2 } } { \sqrt { B - A } } \left( \log \left( { \frac { B } { A } } \right) \right) ^ { 1 / 2 } ,
$$

and the desired inequality follows.

## 3 Proximal mirror descent

Within this section, the following holds:

• Assumption 1 (main problem is well-defined);

• h is a ν-log-barrier of $\mathcal { C } ;$

• Assumption 2 (prox subproblem is well-defined);

$x _ { 0 } \in \mathrm { r i } ( \mathcal { C } )$

The main idea behind the convergence proof is similar to that in [24], for example, where the $O ( 1 / k )$ rate for gradient descent, implicit or explicit, is shown for the norm of the gradient. In the Euclidean case, this is done by accumulating terms $\| x _ { k } - x _ { k - 1 } \| ^ { 2 } = \alpha _ { k } ^ { 2 } \| \nabla f ( x _ { k } ) \| ^ { 2 }$ , whereas here we accumulate their analogues $D _ { h } ( x _ { k - 1 } , x _ { k } ) + D _ { h } ( x _ { k } , x _ { k - 1 } )$ . This, together with the exp-concavity property of logbarriers (3), is used to handle the blow-up at the boundary. The proof below may look somewhat synthetic because, after obtaining a proof based on the idea above, we refined it through the performance estimation framework [8, 22] to obtain tighter and cleaner estimates.

Lemma 3 (Lyapunov analysis of (proxMD)). Let $B _ { k } \geqslant 0 , x _ { k } \in \mathrm { r i } ( \mathcal { C } )$ , and let $x _ { k + 1 }$ be obtained by one iteration of (proxMD) with stepsize $\alpha _ { k + 1 }$ . Set $B _ { k + 1 } : = B _ { k } + \alpha _ { k + 1 }$ . Then

$$
B _ { k + 1 } \left( f ( x _ { k + 1 } ) - f _ { \star } \right) + \left. \nabla h ( x _ { k + 1 } ) , x _ { k + 1 } - x _ { \star } \right. \leqslant B _ { k } \left( f ( x _ { k } ) - f _ { \star } \right) + \left. \nabla h ( x _ { k } ) , x _ { k } - x _ { \star } \right. + H _ { k , \star }\tag{4}
$$

with ${ \cal H } _ { k } \leqslant \nu \ i f \ B _ { k } = 0$ and $\begin{array} { r } { H _ { k } \leqslant \frac { \nu } { 4 } \log \left( \frac { B _ { k + 1 } } { B _ { k } } \right) \ : i f B _ { k } > 0 . } \end{array}$

Proof. From optimality condition for $x _ { k + 1 }$ , we have that

$$
\langle \alpha _ { k + 1 } \nabla f ( x _ { k + 1 } ) + \nabla h ( x _ { k + 1 } ) - \nabla h ( x _ { k } ) , x _ { k + 1 } - x \rangle \leqslant 0 \quad \forall x \in \mathcal { C } \cap \mathcal { S } .\tag{5}
$$

In addition, convexity of f yields

$$
\alpha _ { k + 1 } \big ( f ( x _ { k + 1 } ) - f ( x ) \big ) + \langle \nabla h ( x _ { k + 1 } ) - \nabla h ( x _ { k } ) , x _ { k + 1 } - x \rangle \leqslant 0 \quad \forall x \in \mathcal { C } \cap \mathcal { S } .\tag{6}
$$

Now we add (6) with $x = x ,$ <sub>⋆</sub> and $\frac { B _ { k } } { \alpha _ { k + 1 } } \times ( 6 )$ with $x = x _ { k }$ , obtaining

$$
\begin{array} { r l } & { \alpha _ { k + 1 } \left( f ( x _ { k + 1 } ) - f _ { \star } \right) + B _ { k } \left( f ( x _ { k + 1 } ) - f ( x _ { k } ) \right) + \left. \nabla h ( x _ { k + 1 } ) - \nabla h ( x _ { k } ) , x _ { k + 1 } - x _ { \star } \right. } \\ & { \qquad + \frac { B _ { k } } { \alpha _ { k + 1 } } \left. \nabla h ( x _ { k + 1 } ) - \nabla h ( x _ { k } ) , x _ { k + 1 } - x _ { k } \right. \leqslant 0 . } \end{array}\tag{7}
$$

The two inner products with h can be transformed as follows:

$$
\begin{array} { r l } & { \langle \nabla h ( x _ { k + 1 } ) - \nabla h ( x _ { k } ) , x _ { k + 1 } - x _ { \star } \rangle + \frac { B _ { k } } { \alpha _ { k + 1 } } \langle \nabla h ( x _ { k + 1 } ) - \nabla h ( x _ { k } ) , x _ { k + 1 } - x _ { k } \rangle } \\ & { = \langle \nabla h ( x _ { k + 1 } ) , x _ { k + 1 } - x _ { \star } \rangle - \langle \nabla h ( x _ { k } ) , x _ { k } - x _ { \star } \rangle - \underbrace { \left( \frac { B _ { k } } { \alpha _ { k + 1 } } \langle \nabla h ( x _ { k + 1 } ) , x _ { k } - x _ { k + 1 } \rangle + \frac { B _ { k + 1 } } { \alpha _ { k + 1 } } \langle \nabla h ( x _ { k } ) , x _ { k + 1 } - x _ { k } \rangle \right) } _ { = : H _ { k } } . } \end{array}
$$

Substituting this back to (7) gives us

$$
B _ { k + 1 } \left( f ( x _ { k + 1 } ) - f _ { * } \right) + \left. \nabla h ( x _ { k + 1 } ) , x _ { k + 1 } - x _ { * } \right. \leqslant B _ { k } \left( f ( x _ { k } ) - f _ { * } \right) + \left. \nabla h ( x _ { k } ) , x _ { k } - x _ { * } \right. + H _ { k } .\tag{8}
$$

The last inequality would be perfect for telescoping if we could deal with the $H _ { k }$ terms. Fortunately, we can do this because of exp-concavity of h. Specifically,

$$
\begin{array} { r } { \nu \exp \left( \frac { h ( x _ { k } ) - h ( x _ { k + 1 } ) } { \nu } \right) + \langle \nabla h ( x _ { k } ) , x _ { k + 1 } - x _ { k } \rangle \leqslant \nu } \\ { \nu \exp \left( \frac { h ( x _ { k + 1 } ) - h ( x _ { k } ) } { \nu } \right) + \langle \nabla h ( x _ { k + 1 } ) , x _ { k } - x _ { k + 1 } \rangle \leqslant \nu . } \end{array}
$$

Hence, summing these inequalities (with appropriate weights), we obtain

$$
\begin{array} { r l } & { H _ { k } \leqslant \frac { B _ { k } + B _ { k + 1 } } { \alpha _ { k + 1 } } \nu - \frac { B _ { k } } { \alpha _ { k + 1 } } \nu \exp \left( \frac { h ( x _ { k + 1 } ) - h ( x _ { k } ) } { \nu } \right) - \frac { B _ { k + 1 } } { \alpha _ { k + 1 } } \nu \exp \left( \frac { h ( x _ { k } ) - h ( x _ { k + 1 } ) } { \nu } \right) } \\ & { \quad \leqslant \frac { \nu } { \alpha _ { k + 1 } } \left( B _ { k } + B _ { k + 1 } - 2 \sqrt { B _ { k } B _ { k + 1 } } \right) = \frac { \nu } { \alpha _ { k + 1 } } \left( \sqrt { B _ { k + 1 } } - \sqrt { B } _ { k } \right) ^ { 2 } , } \end{array}
$$

where the second inequality follows from the AM-GM inequality. When $B _ { k } = 0$ , one has $B _ { k + 1 } = \alpha _ { k + 1 }$ and hence $H _ { k } \leqslant \nu .$ . When $B _ { k } > 0 , H _ { k }$ can be bounded using Lemma 2:

$$
H _ { k } \leqslant \frac { \nu } { \alpha _ { k + 1 } } \left( \sqrt { B _ { k + 1 } } - \sqrt { B } _ { k } \right) ^ { 2 } \leqslant \frac { \nu } { 4 } \log \left( \frac { B _ { k + 1 } } { B _ { k } } \right) .
$$

Lemma 3 suggests telescoping the inequality at our disposal to obtain the main convergence results for (proxMD).

Theorem 1. Let $\{ x _ { i } \} _ { i = 0 , \ldots , k }$ be the iterates of (proxMD). For any $k \geqslant 1$ , it holds that

$$
f ( x _ { k } ) - f _ { \star } \leqslant \frac { \langle \nabla h ( x _ { 0 } ) , x _ { 0 } - x _ { \star } \rangle + 2 \nu + \frac { \nu } { 4 } \log \frac { A _ { k } } { A _ { 1 } } } { A _ { k } }\tag{9}
$$

where $\textstyle A _ { k } : = \sum _ { i = 1 } ^ { k } \alpha _ { i }$

Proof. Lemma 3 suggests defining the Lyapunov function

$$
\Psi _ { k } : = A _ { k } \ : \big ( f ( x _ { k } ) - f _ { \star } \big ) + \langle \nabla h ( x _ { k } ) , x _ { k } - x _ { \star } \rangle ,
$$

and telescoping the inequality to reach $\begin{array} { r } { \Psi _ { k } \leqslant \Psi _ { k - 1 } + H _ { k - 1 } \leqslant \ldots \leqslant \Psi _ { 0 } + \sum _ { i = 0 } ^ { k - 1 } H _ { i } } \end{array}$ . On top of this, we know that $H _ { 0 } \leqslant \nu$ (by setting $B _ { 0 } = 0 )$ and that $\begin{array} { r } { \sum _ { i = 1 } ^ { k - 1 } H _ { i } \leqslant \frac { \nu } { 4 } \sum _ { i = 1 } ^ { k - 1 } \log \left( \frac { A _ { i + 1 } } { A _ { i } } \right) = \frac { \nu } { 4 } \log \left( \frac { A _ { k } } { A _ { 1 } } \right) } \end{array}$ . It remains to use exp-concavity (3) as $\left. \nabla h ( x _ { k } ) , x _ { \star } - x _ { k } \right. \leqslant$ ν to deal with the second term in $\Psi _ { k }$ □

One can obtain many minor variations around this base theme, for instance by explicitly accounting for specific starting points $x _ { 0 }$ such that $\nabla h ( x _ { 0 } ) = 0$ . In this case, one can use $H _ { 0 } = A _ { 0 } = 0$ and reach

$$
f ( x _ { k } ) - f _ { \star } \leqslant \frac { \nu + \frac { \nu } { 4 } \log \frac { A _ { k } } { A _ { 1 } } } { A _ { k } } ,\tag{10}
$$

which can more readily be compared with classical bounds for (IPM). Starting from the analytic center $x _ { 0 } : = \mathrm { a r g m i n } _ { x \in S } h ( x )$ , whose definition implies $\langle \nabla h ( x _ { 0 } ) , y - x _ { 0 } \rangle \geqslant 0$ for all $y \in S ,$ one obtains $\langle \nabla h ( x _ { 0 } ) , x _ { 0 } - x _ { \star } \rangle \leqslant 0$ and therefore

$$
f ( x _ { k } ) - f _ { \star } \leqslant \frac { 2 \nu + \frac { \nu } { 4 } \log \frac { A _ { k } } { A _ { 1 } } } { A _ { k } } .
$$

Similarly, one can account for starting points minimizing $B _ { 0 } f ( x ) + h ( x )$ (with $B _ { 0 } > 0 )$ by adapting the proof with $\begin{array} { r } { B _ { k } : = B _ { 0 } + A _ { k } = B _ { 0 } + \sum _ { i = 1 } ^ { k } \alpha _ { i } } \end{array}$ with the convention that the empty sum is 0. In this case, we adapt the Lyapunov strategy to

$$
\Psi _ { k } : = B _ { k } \ : ( f ( x _ { k } ) - f _ { \star } ) + \langle \nabla h ( x _ { k } ) , x _ { k } - x _ { \star } \rangle ,
$$

and telescope the inequality to reach $\begin{array} { r } { \Psi _ { k } \leqslant \Psi _ { k - 1 } + H _ { k - 1 } \leqslant \ldots \leqslant \Psi _ { 0 } + \sum _ { i = 0 } ^ { k - 1 } H _ { i } } \end{array}$ . Here we have to use $\begin{array} { r } { \sum _ { i = 0 } ^ { k - 1 } H _ { i } \leqslant \frac { \nu } { 4 } \sum _ { i = 1 } ^ { k - 1 } \log \left( \frac { B _ { i + 1 } } { B _ { i } } \right) = \frac { \nu } { 4 } \log \left( \frac { B _ { k } } { B _ { 0 } } \right) } \end{array}$ , arriving at

$$
\begin{array} { r l } & { B _ { k } ( f ( x _ { k } ) - f _ { \star } ) - \nu \leqslant B _ { 0 } \left( f ( x _ { 0 } ) - f _ { \star } \right) + \langle \nabla h ( x _ { 0 } ) , x _ { 0 } - x _ { \star } \rangle + \frac { \nu } { 4 } \log \left( \frac { B _ { k } } { B _ { 0 } } \right) } \\ & { \qquad \leqslant \langle B _ { 0 } \nabla f ( x _ { 0 } ) + \nabla h ( x _ { 0 } ) , x _ { 0 } - x _ { \star } \rangle + \frac { \nu } { 4 } \log \left( \frac { B _ { k } } { B _ { 0 } } \right) . } \end{array}\tag{11}
$$

Specific stepsize schedules. The two corollaries below concern $\left( \mathrm { p r o x M D } \right)$ with specific stepsize schedules. A first classical choice $( \mathrm { e . g . }$ , akin to mirror descent) is that of constant stepsizes. Another possibility, more closely related to classical interior-point strategies is that of geometric stepsizes.

Corollary 1. Let $\alpha _ { k } = \alpha$ . Let $\{ x _ { i } \} _ { i = 0 , \ldots , k }$ be the iterates of (proxMD). For any $k \geqslant 1$ , it holds that

$$
f ( x _ { k } ) - f _ { \star } \leqslant \frac { \langle \nabla h ( x _ { 0 } ) , x _ { 0 } - x _ { \star } \rangle + 2 \nu + \frac { \nu } { 4 } \log k } { \alpha k } .\tag{12}
$$

That is, $\begin{array} { r } { f ( x _ { k } ) - f _ { \star } = O ( \frac { \log k } { k } ) } \end{array}$

Proof. In this case, $A _ { k } = \alpha k$

Corollary 2. Let $\mu > 0$ and $A _ { 1 } = \alpha _ { 1 } > 0$ , and define the sequence of stepsizes using $\alpha _ { k + 1 } = \mu A _ { k }$ . Let $\{ x _ { i } \} _ { i = 0 , \ldots , k }$ be the iterates of (proxMD). For any $k \geqslant 1$ , it holds that

$$
f ( x _ { k } ) - f _ { \star } \leqslant \frac { \langle \nabla h ( x _ { 0 } ) , x _ { 0 } - x _ { \star } \rangle + 2 \nu + ( k - 1 ) \frac { \nu } { 4 } \log ( 1 + \mu ) } { \alpha _ { 1 } ( 1 + \mu ) ^ { k - 1 } } .\tag{13}
$$

It follows that $f ( x _ { k } ) - f _ { \star } = O ( ( 1 + \tilde { \mu } ) ^ { - k } )$ for any $\tilde { \mu } < \mu$

Proof. In this case, for $k > 1$ we have $A _ { k } = ( 1 + \mu ) A _ { k - 1 } = ( 1 + \mu ) ^ { k - 1 } A _ { 1 } = ( 1 + \mu ) ^ { k - 1 } \alpha _ { 1 } .$

Similar bounds can directly be obtained by exploiting the possibility of an initialization at the analytic center $x _ { 0 }$ satisfying $\langle \nabla h ( x _ { 0 } ) , x _ { 0 } - x _ { \star } \rangle \leqslant 0$

Structured barriers. It is common to aim at improving the conditioning of the subproblems to be solved. For instance, [4, 6] add a quadratic regularization to the barriers. One could thus wonder whether the proofs above allow for such a degree of freedom given that quadratic functions are not exp-concave in general. For this, one can simply consider a structured distance-generating function of the form $h : = h _ { 1 } + h _ { 2 }$ where $h _ { 1 }$ is finite on the boundary (so $D _ { h _ { 1 } } ( x _ { \star } , x )$ does not blow up) and $h _ { 2 }$ is ν-exp-concave. In this case, the statement of Lemma 3 can be generalized to

$$
\begin{array} { r l } & { B _ { k + 1 } \left( f ( x _ { k + 1 } ) - f _ { \star } \right) + \langle \nabla h _ { 2 } ( x _ { k + 1 } ) , x _ { k + 1 } - x _ { \star } \rangle + D _ { h _ { 1 } } ( x _ { \star } , x _ { k + 1 } ) } \\ & { \qquad \leqslant B _ { k } \left( f ( x _ { k } ) - f _ { \star } \right) + \langle \nabla h _ { 2 } ( x _ { k } ) , x _ { k } - x _ { \star } \rangle + D _ { h _ { 1 } } ( x _ { \star } , x _ { k } ) + H _ { k } , } \end{array}
$$

with the same bounds on $H _ { k }$ , thereby resulting in bounds akin to that of Theorem 1

$$
f ( x _ { k } ) - f _ { \star } \leqslant \frac { D _ { h _ { 1 } } ( x _ { \star } , x _ { 0 } ) + \langle \nabla h _ { 2 } ( x _ { 0 } ) , x _ { 0 } - x _ { \star } \rangle + 2 \nu + \frac { \nu } { 4 } \log \frac { A _ { k } } { A _ { 1 } } } { A _ { k } }
$$

and of its variations.

## 4 Mirror descent

## 4.1 Mirror descent

Within this section, the following holds:

• Assumption 1 (main problem is well-defined);

• h is a ν-log-barrier of $\mathcal { C } ;$

• Assumption 2 (mirror subproblem is well-defined);

$x _ { 0 } \in \mathrm { r i } ( \mathcal { C } )$

$f , h \colon \operatorname { r i } ( \mathcal { C } ) \to \mathbb { R }$ are diferentiable<sup>2</sup> and f is relatively smooth with respect to $h .$

Let us recall the now-classical assumption of relative smoothness [1, 3, 13].

Definition 1 (Relative smoothness). Let $f , h \colon \operatorname { r i } ( \mathcal { C } ) \to \mathbb { R }$ be convex and diferentiable. We say that f is L-relatively smooth with respect to h on ri(C) if

$$
D _ { f } ( x , y ) \leqslant L D _ { h } ( x , y ) , \qquad \forall x , y \in \mathrm { r i } ( \mathcal { C } ) .
$$

Equivalently, when f and h are twice diferentiable, this means

$$
\nabla ^ { 2 } f ( x ) \prec L \nabla ^ { 2 } h ( x ) , \qquad \forall x \in \operatorname { r i } ( { \mathcal { C } } ) .
$$

For simplicity of presentation, this section focuses on mirror descent (MD) with the fixed-stepsize strategy $\alpha _ { k } = \alpha > 0$ . The proof below combines the standard mirror descent proof with the idea used to analyze the proximal mirror algorithm. However, it may look more convoluted, as we use one additional inequality not present in either analysis.

Lemma 4 (Lyapunov analysis of (MD)). Let $B _ { k } > 0 , x _ { k } \in \operatorname { r i } ( { \mathcal { C } } )$ , and let $x _ { k + 1 }$ be obtained by one iteration of (MD) with stepsize $\alpha _ { k + 1 } = \alpha \in \left( 0 , { \frac { 1 } { L } } \right)$ . Set $B _ { k + 1 } : = B _ { k } + 1$ . Then

$$
\begin{array} { r } { \alpha B _ { k + 1 } ( f ( x _ { k + 1 } ) - f _ { \star } ) + \left. \nabla h ( x _ { k + 1 } ) , x _ { k + 1 } - x _ { \star } \right. \leqslant \alpha B _ { k } ( f ( x _ { k } ) - f _ { \star } ) + \left. \nabla h ( x _ { k } ) , x _ { k } - x _ { \star } \right. + H _ { k , \star } } \end{array}\tag{14}
$$

with $\begin{array} { r } { H _ { k } \leqslant \frac { \nu } { B _ { k } } } \end{array}$

Proof. From optimality condition for $x _ { k + 1 }$ , we have that

$$
\langle \alpha \nabla f ( x _ { k } ) + \nabla h ( x _ { k + 1 } ) - \nabla h ( x _ { k } ) , x _ { k + 1 } - x \rangle \leqslant 0 \quad \forall x \in { \mathcal { C } } \cap { \mathcal { S } }\tag{15}
$$

Applying relative smoothness of $f ,$ we get

$$
\begin{array} { r } { f ( x _ { k + 1 } ) - f ( x _ { k } ) - \langle \nabla f ( x _ { k } ) , x _ { k + 1 } - x _ { k } \rangle \leqslant L \left[ h ( x _ { k + 1 } ) - h ( x _ { k } ) - \langle \nabla h ( x _ { k } ) , x _ { k + 1 } - x _ { k } \rangle \right] = L D _ { h } ( x _ { k + 1 } , x _ { k } ) , } \end{array}
$$

which combined with (15) gives us:

$$
\begin{array} { r } { \alpha f ( x _ { k + 1 } ) - \alpha f ( x _ { k } ) + \alpha \langle \nabla f ( x _ { k } ) , x _ { k } - x \rangle \leqslant \alpha L D _ { h } ( x _ { k + 1 } , x _ { k } ) + \langle \nabla h ( x _ { k + 1 } ) - \nabla h ( x _ { k } ) , x - x _ { k + 1 } \rangle , } \end{array}\tag{16}
$$

for any $x \in { \mathcal { C } } \cap { \mathcal { S } } .$ Convexity of f yields

$$
\alpha f ( x _ { k + 1 } ) - \alpha f ( x ) \leqslant \alpha L D _ { h } ( x _ { k + 1 } , x _ { k } ) + \langle \nabla h ( x _ { k + 1 } ) - \nabla h ( x _ { k } ) , x - x _ { k + 1 } \rangle \quad \forall x \in \mathcal { C } \cap \mathcal { S } .\tag{17}
$$

Now, let us add (17) with $x = x _ { \star }$ and $B _ { k } \times ( 1 7 )$ with $x = x _ { k }$ . This yields

$$
\begin{array} { r l } & { \alpha \left( f ( x _ { k + 1 } ) - f _ { \star } \right) + \alpha B _ { k } \left( f ( x _ { k + 1 } ) - f ( x _ { k } ) \right) - \left. \nabla h ( x _ { k + 1 } ) - \nabla h ( x _ { k } ) , x _ { \star } - x _ { k + 1 } \right. } \\ & { \qquad - B _ { k } \langle \nabla h ( x _ { k + 1 } ) - \nabla h ( x _ { k } ) , x _ { k } - x _ { k + 1 } \rangle - \alpha ( B _ { k } + 1 ) L D _ { h } ( x _ { k + 1 } , x _ { k } ) \leqslant 0 , } \end{array}\tag{18}
$$

where the terms involving h can be transformed as follows, using $\begin{array} { r } { 0 \leqslant \alpha \leqslant \frac { 1 } { L } } \end{array}$ :

$$
\begin{array} { r l r } {  { \langle \nabla h ( x _ { k + 1 } ) - \nabla h ( x _ { k } ) , x _ { \star } - x _ { k + 1 } \rangle + B _ { k } \langle \nabla h ( x _ { k + 1 } ) - \nabla h ( x _ { k } ) , x _ { k } - x _ { k + 1 } \rangle + \alpha ( B _ { k } + 1 ) L D _ { h } ( x _ { k + 1 } , x _ { k } ) } } \\ & { } & { \leqslant \langle \nabla h ( x _ { k + 1 } ) , x _ { \star } - x _ { k + 1 } \rangle - \langle \nabla h ( x _ { k } ) , x _ { \star } - x _ { k } \rangle } \\ & { } & { \quad + \underbrace { B _ { k } \langle \nabla h ( x _ { k + 1 } ) , x _ { k } - x _ { k + 1 } \rangle + ( B _ { k } + 1 ) \langle \nabla h ( x _ { k } ) , x _ { k + 1 } - x _ { k } \rangle + ( B _ { k } + 1 ) D _ { h } ( x _ { k + 1 } , x _ { k } ) } _ { = : H _ { k } } \cdot \begin{array} { c } { ( \nabla h ( x _ { k + 1 } ) - x _ { k + 1 } ) } \\ { ( x _ { k + 1 } ) } \end{array} } \end{array}\tag{9}
$$

Thus, an expression for $H _ { k }$ is $H _ { k } = ( B _ { k } + 1 ) ( h ( x _ { k + 1 } ) - h ( x _ { k } ) ) + B _ { k } \langle \nabla h ( x _ { k + 1 } ) , x _ { k } - x _ { k + 1 } \rangle$ , which we need to upper-bound. We proceed by invoking exp-concavity of h:

$$
\begin{array} { l } { { H _ { k } \leqslant ( B _ { k } + 1 ) \big ( h ( x _ { k + 1 } ) - h ( x _ { k } ) \big ) + B _ { k } \left( \nu - \nu \exp \left( \frac { h ( x _ { k + 1 } ) - h ( x _ { k } ) } { \nu } \right) \right) } } \\ { { \leqslant \nu \left[ ( B _ { k } + 1 ) \log \left( 1 + \frac { 1 } { B _ { k } } \right) - 1 \right] \leqslant \frac { \nu } { B _ { k } } , } } \end{array}
$$

where the second line is obtained by maximizing the right-hand side over $\Delta : = h ( x _ { k + 1 } ) - h ( x _ { k } )$ (the expression is concave in $\Delta )$ . In the third inequality, we use log $( 1 + a ) \leqslant a$ and simplify. The desired statement follows by combining (18) with (19) and the bound on $H _ { k }$ □

We are now ready to state the main convergence results for mirror descent (MD).

Theorem 2. Let $\{ x _ { i } \} _ { i = 0 , \ldots , k }$ be the iterates of (MD) with $\begin{array} { r } { \alpha = \frac { 1 } { L } } \end{array}$ . For any $k \geqslant 1$ , it holds that

$$
f ( x _ { k } ) - f _ { \star } \leqslant \frac { f ( x _ { 0 } ) - f _ { \star } + L \left( \langle \nabla h ( x _ { 0 } ) , x _ { 0 } - x _ { \star } \rangle + 2 \nu + \nu \log k \right) } { k + 1 } .\tag{20}
$$

Proof. Again, Lemma 4 suggests telescoping (14) with some $B _ { 0 } > 0$ . Setting $\begin{array} { r } { \alpha = \frac { 1 } { L } } \end{array}$ and $B _ { 0 } = 1$ gives $B _ { i } = i + 1$ , and hence

$$
( k + 1 ) \textstyle { \frac { 1 } { L } } \left( f ( x _ { k } ) - f _ { \star } \right) + \langle \nabla h ( x _ { k } ) , x _ { k } - x _ { \star } \rangle \leqslant \frac { 1 } { L } \big ( f ( x _ { 0 } ) - f _ { \star } \big ) + \langle \nabla h ( x _ { 0 } ) , x _ { 0 } - x _ { \star } \rangle + \nu \sum _ { i = 0 } ^ { k - 1 } \frac { 1 } { i + 1 } ,
$$

and consequently,

$$
\begin{array} { r } { ( k + 1 ) \frac { 1 } { L } \left( f ( x _ { k } ) - f _ { \star } \right) - \nu \leqslant \frac { 1 } { L } ( f ( x _ { 0 } ) - f _ { \star } ) + \langle \nabla h ( x _ { 0 } ) , x _ { 0 } - x _ { \star } \rangle + \nu \left( 1 + \log k \right) . } \end{array}
$$

A diferent and direct mirror descent result from the Bregman proximal-point perspective. A general MD algorithm is specified by a triple $( f , w , \alpha )$ , where $f$ is the objective to be minimized, w is the mirror function, and α is the stepsize. In the Euclidean case, we know that GD with unit stepsize is a particular case of MD applied to the triple $( f , { \frac { L } { 2 } } \| \cdot \| ^ { 2 } , 1 )$ . In turn, it can be seen as the proximal point algorithm applied to the triple $( f , \textstyle { \frac { L } { 2 } } | | \cdot | | ^ { 2 } - f , { \overline { { 1 } } } )$ . Evidently, this can be generalized by replacing $\textstyle { \frac { 1 } { 2 } } \left\| \cdot \right\| ^ { 2 }$ by generic convex $h ,$ , as this is just an algebraic manipulation. So, let us choose $\alpha _ { k } = 1$ and use the barrier $w = L h - f$ in (proxMD) (the choice of using w and not h for the log-barrier is for notational consistency within this section), and further assume w is ν-exp-concave. This yields the update

$$
x _ { k } = \underset { x \in \mathcal { S } } { \mathrm { a r g m i n } } \left\{ f ( x ) + D _ { w } ( x , x _ { k - 1 } ) \right\} = \underset { x \in \mathcal { S } } { \mathrm { a r g m i n } } \left\{ \langle \nabla f ( x _ { k - 1 } ) , x \rangle + L D _ { h } ( x , x _ { k - 1 } ) \right\} ,
$$

and Theorem 1 directly applies with $A _ { k } = k$

$$
f ( x _ { k } ) - f _ { \star } \leqslant \frac { \langle L \nabla h ( x _ { 0 } ) - \nabla f ( x _ { 0 } ) , x _ { 0 } - x _ { \star } \rangle + 2 \nu + \frac { \nu } { 4 } \log ( k ) } { k } .
$$

Of course, this assumption of $L h - f$ being exp-concave is more constraining than that of h being exp-concave, so such assumption should be motivated beyond the fact that it makes the result a direct consequence of the Bregman proximal-point analysis.

To motivate this, we simply claim that the D-optimal design problem [13, Section 2.2] satisfies in those assumptions. Let $H \in \overline { { \mathbb { R } ^ { m \times d } } }$ with $d > m$ , and consider $\begin{array} { r } { \bar { f ( x ) } = - \log \operatorname* { d e t } ( H \operatorname { d i a g } ( x ) H ^ { \dag } ) , \mathcal { C } = \mathbb { R } _ { + } ^ { d } } \end{array}$ $S = \{ x \colon \langle \mathbf { 1 } , x \rangle = 1 \}$ , and $\begin{array} { r } { h ( x ) = - \sum _ { i = 1 } ^ { d } \log x _ { i } } \end{array}$ . Now, let $L = 1$ and $\nu = d - m > 0$ . Since for any $t > 0$ we have logarithmic homogeneity

$$
w ( t x ) = w ( x ) - \nu \log t ,
$$

diferentiation at $t = 1$ gives $\nabla ^ { 2 } w ( x ) x = - \nabla w ( x )$ and $x ^ { \top } \nabla ^ { 2 } w ( x ) x = \nu$ . Moreover, since w is convex, $\nabla ^ { 2 } w ( x ) \succeq 0$ . Therefore, for any $v ,$ the Cauchy–Schwarz inequality gives

$$
\langle \nabla w ( x ) , v \rangle ^ { 2 } = \langle \nabla ^ { 2 } w ( x ) x , v \rangle ^ { 2 } \leqslant \langle \nabla ^ { 2 } w ( x ) x , x \rangle \langle \nabla ^ { 2 } w ( x ) v , v \rangle = \nu \langle \nabla ^ { 2 } w ( x ) v , v \rangle .
$$

Hence $\begin{array} { r } { \nabla ^ { 2 } w ( x ) \succcurlyeq \frac { 1 } { \nu } \nabla w ( x ) \nabla w ( x ) ^ { \top } } \end{array}$ and w is ν-exp-concave.

## 4.2 Mirror flow

Since the proofs of Theorems 1 and 2 may look somewhat involved, it might be useful to gain intuition from the corresponding continuous-time dynamics, namely a mirror flow. We consider a simplified setting to make the exposition more transparent.

Let $S = \mathbb { R } ^ { d }$ . Suppose, in addition to ν-exp-concavity, that $h \colon { \mathcal { C } } \to \mathbb { R } \cup \{ + \infty \}$ is twice diferentiable on $\operatorname { r i } ( \mathcal { C } )$ and strictly convex. Given an initial point $x _ { 0 }$ , the mirror flow is defined as

$$
\frac { \mathrm { d } } { \mathrm { d } t } \nabla h ( x _ { t } ) = - \nabla f ( x _ { t } ) , \qquad t \geqslant 0 .\tag{21}
$$

Alternatively, we can write it as $\nabla ^ { 2 } h ( x _ { t } ) \dot { x } _ { t } = - \nabla f ( x _ { t } )$

First, note that $f ( x _ { t } )$ is monotone. This is standard:

$$
\frac { \mathrm { d } } { \mathrm { d } t } f ( x _ { t } ) = \langle \nabla f ( x _ { t } ) , \dot { x } _ { t } \rangle = - \langle \nabla ^ { 2 } h ( x _ { t } ) \dot { x } _ { t } , \dot { x } _ { t } \rangle \leqslant 0 .
$$

Now from (21), we have

$$
\langle \nabla f ( { x } _ { t } ) + \frac { \mathrm { d } } { \mathrm { d } t } \nabla h ( { x } _ { t } ) , { x } _ { t } - { x } _ { \star } \rangle \leqslant 0 .
$$

$\mathrm { B y }$ convexity of $f$ and the chain rule applied to the second term, it follows that

$$
f ( x _ { t } ) - f _ { \star } + \frac { \mathrm { d } } { \mathrm { d } t } \langle \nabla h ( x _ { t } ) , x _ { t } - x _ { \star } \rangle - \langle \nabla h ( x _ { t } ) , \dot { x } _ { t } \rangle \leqslant 0 .\tag{22}
$$

The first two terms are exactly the same as in the Lyapunov function (4), and the last one is a continuous analog of the $H _ { k }$ term, the only problematic one. This time, however, we bound it diferently

$$
\begin{array} { r l } & { \langle \nabla h ( x _ { t } ) , \dot { x } _ { t } \rangle \leqslant \sqrt { \langle \nabla h ( x _ { t } ) , \dot { x } _ { t } \rangle ^ { 2 } } } \\ & { \qquad = \langle \nabla h ( x _ { t } ) \nabla h ( x _ { t } ) ^ { \top } \dot { x } _ { t } , \dot { x } _ { t } \rangle ^ { 1 / 2 } } \\ & { \leqslant \sqrt { \nu } \langle \nabla ^ { 2 } h ( x _ { t } ) \dot { x } _ { t } , \dot { x } _ { t } \rangle ^ { 1 / 2 } } \\ & { \qquad = \sqrt { \nu } \langle - \nabla f ( x _ { t } ) , \dot { x } _ { t } \rangle ^ { 1 / 2 } } \\ & { \qquad = \sqrt { \nu } \left[ - \frac { \mathrm { d } } { \mathrm { d } t } ( f ( x _ { t } ) - f _ { \star } ) \right] ^ { 1 / 2 } . } \end{array}
$$

(by exp-concavity)

It remains only to integrate this inequality carefully. For brevity, let $F ( t ) = f ( x _ { t } ) - f _ { \star }$ . Then integrating (22) from 0 to $T$ and using the latter bound for $\langle \nabla h ( x _ { t } ) , \dot { x } _ { t } \rangle$ yields

$$
\begin{array} { r l } { \displaystyle \int _ { 0 } ^ { T } F ( t ) \mathrm { d } t + \int _ { 0 } ^ { T } \frac { \mathrm { d } } { \mathrm { d } t } \langle \nabla h ( x _ { t } ) , x _ { t } - x _ { * } \rangle \mathrm { d } t \leqslant \sqrt { \nu } \displaystyle \int _ { 0 } ^ { T } \sqrt { - F ^ { \prime } ( t ) } \mathrm { d } t } & { } \\ { \displaystyle } & { = \sqrt { \nu } \displaystyle \int _ { 0 } ^ { T } \sqrt { - ( t + 1 ) F ^ { \prime } ( t ) } \cdot \frac { 1 } { \sqrt { t + 1 } } \mathrm { d } t } \\ { \displaystyle } & { \leqslant \left( \nu \displaystyle \int _ { T } ^ { 0 } ( t + 1 ) F ^ { \prime } ( t ) \mathrm { d } t \cdot \displaystyle \int _ { 0 } ^ { T } \frac { 1 } { t + 1 } \mathrm { d } t \right) ^ { 1 / 2 } } \\ { \displaystyle } & { \leqslant \int _ { T } ^ { 0 } ( t + 1 ) F ^ { \prime } ( t ) \mathrm { d } t + \frac { \nu } { 4 } \displaystyle \int _ { 0 } ^ { T } \frac { 1 } { t + 1 } \mathrm { d } t } \\ { \displaystyle } & { = F ( 0 ) - ( T + 1 ) F ( T ) + \displaystyle \int _ { 0 } ^ { T } F ( t ) \mathrm { d } t + \frac { \nu } { 4 } \log ( T + 1 ) , } \end{array}
$$

where in the second inequality we used Cauchy-Schwarz, in the third AM-GM, and in the last equation integration by parts. With this, we arrive at

$$
( T + 1 ) F ( T ) \leqslant F ( 0 ) + \langle \nabla h ( x _ { 0 } ) , x _ { 0 } - x _ { \star } \rangle + \nu + \frac { \nu } { 4 } \log ( T + 1 ) ,
$$

which essentially is exactly the same bound we had in the discrete case.

## 4.3 Linear objectives

This section illustrates that the analyses are much simpler with linear objectives, while not incurring any additional logarithmic terms. That is, we let $S = \mathbb { R } ^ { d }$ and $c \in \mathbb { R } ^ { d }$ and consider the simpler problem

$$
\operatorname* { m i n } _ { x \in { \mathcal C } } \left\{ f ( x ) \equiv \langle c , x \rangle \right\} ,
$$

under the same assumptions as before:

• Assumption 1 (main problem is well-defined);

• h is a ν-log-barrier of C;

• Assumption 2 (prox subproblem is well-defined);

$x _ { 0 } \in \mathrm { r i } ( \mathcal { C } )$

This setup is traditional from the interior-point literature where nonlinear objectives are typically delegated to the constraint set; see, e.g., [18, Section 5.3.4]. One should note that with a linear objective, (MD) and (proxMD) coincide. They also coincide with (IPM) up to a stepsize adjustment:

$$
x _ { k } = \underset { x \in \mathbb { R } ^ { d } } { \mathrm { a r g m i n } } \left\{ \left( \sum _ { i = 1 } ^ { k } \alpha _ { i } \right) \langle c , x \rangle + D _ { h } ( x , x _ { 0 } ) \right\} .
$$

In other words, using $\begin{array} { r } { \nabla h ( x _ { k } ) = \nabla h ( x _ { k - 1 } ) - \alpha _ { k } c = \nabla h ( x _ { 0 } ) - \sum _ { i = 1 } ^ { k } \alpha _ { i } c , } \end{array}$ one arrives at

$$
f ( x _ { k } ) - f _ { \star } = \langle c , x _ { k } - x _ { \star } \rangle = \frac { \langle \nabla h ( x _ { k } ) , x _ { \star } - x _ { k } \rangle + \langle \nabla h ( x _ { 0 } ) , x _ { k } - x _ { 0 } \rangle + \langle \nabla h ( x _ { 0 } ) , x _ { 0 } - x _ { \star } \rangle } { \sum _ { i = 1 } ^ { k } \alpha _ { i } }
$$

in general, or $\begin{array} { r } { f ( x _ { k } ) - f _ { \star } \leqslant \frac { \nu } { \sum _ { i = 1 } ^ { k } \alpha _ { i } } ~ \mathrm { i f } ~ \nabla h ( x _ { 0 } ) = 0 . } \end{array}$

Takeaways from linear objectives. Overall, looking at those very simple bounds and analyses, one might think that the bounds (9) or (10) are just weak and overly complicated.

We prove in the next section that the additional terms incurred by updating the prox-center are actually not an artifact from our proof, and that the linear-case analysis standard of the IPM literature, although simple and elegant, does not carry much of the dificulty of the general problem. In fact, looking back at (1), the linear analysis of this section is no longer generally valid when $S \neq \mathbb { R } ^ { d } \ ( \mathrm { o r }$ , in more details, when S is not an afine set), i.e., the proof requires h to encode all constraints.

## 5 Tightness of the bounds

In this section, we show tightness of the log(k)/k bound for (proxMD). For simplicity, we show this for $\nu = 1 , \alpha _ { k } = \alpha = 1$ , and $x _ { 0 }$ so that $\nabla h ( x _ { 0 } ) = 0$ . The more general case with $\alpha , \nu > 0$ can be obtained by appropriate scaling arguments. We fix $S = \mathbb { R } ^ { d }$ , while the set C is defined below.

Theorem 3. For any integer $k \geqslant 1$ , there exist a function f satisfying Assumption 1 and a 1-log-barrier h for a closed convex set C such that, when initialized at x for which $\nabla h ( x _ { 0 } ) = 0$ , the iterates of (proxMD) with $\alpha _ { k } = \alpha = 1$ , satisfy

$$
k ( f ( x _ { k } ) - f _ { \star } ) = 1 + \sum _ { i = 1 } ^ { k - 1 } ( \sqrt { i + 1 } - \sqrt { i } ) ^ { 2 } \sim 1 + \frac { 1 } { 4 } \log k .\tag{23}
$$

A concrete, but tedious, construction for any fixed k is provided below. The precise values presented below originate from the idea that one can identify which inequalities used in the proof must be tightened, while the construction is closely related to previous works on performance estimation problems [8, 9, 22]. The detailed construction of the corresponding PEP is deferred to the companion note [5]. The counterexample below was initially obtained through an interactive use of an LLM, which was provided with both this PEP formulation and the convergence proofs from Section $_ { 3 ; }$ the latter indicate which inequalities in the PEP are expected to be active.

## 5.1 Shape of the constructed example

The construction relies on interpolation arguments that reconstruct convex functions from samples (see, e.g., [22]). That is, we provide sampled representations of the objective and barrier functions, respectively in the form,

$$
S _ { f } = \{ ( x _ { i } , s _ { i } ^ { f } , f _ { i } ) \} _ { i \in \{ \star , 1 , 2 , \ldots , k \} } \qquad \mathrm { a n d } \qquad S _ { h } = \{ ( x _ { i } , s _ { i } ^ { h } , h _ { i } ) \} _ { i \in \{ 0 , 1 , 2 , \ldots , k \} } ,\tag{24}
$$

so that f and h respectively interpolate $S _ { f }$ and $S _ { h }$ . That is, $S _ { f }$ is such that $f$ is convex with $f ( x _ { i } ) = f _ { i }$ and $s _ { i } ^ { f } \in \partial f ( x _ { i } )$ (for all $i = { \star , 1 , 2 , \ldots , k } )$ , and $S _ { h }$ is such that h is 1-exp-concave with $h ( x _ { i } ) = h _ { i }$ and $s _ { i } ^ { h } \in \dot { \partial } h ( x _ { i } )$ (for all $i = 0 , 1 , 2 , \ldots , k )$ . Further, $S _ { f }$ and $S _ { h }$ are built so that when applying (proxMD) to the corresponding f and $h ,$ the bound in (23) is attained. To achieve this, f and h are coupled via the condition

$$
s _ { i + 1 } ^ { h } = s _ { i } ^ { h } - s _ { i + 1 } ^ { f } ,
$$

which encodes optimality conditions of (proxMD), and by sharing the same $x _ { i } \mathrm { { ' s } }$ . As $S \ : = \ : \mathbb { R } ^ { d }$ , and after the interpolation procedure to recover $f$ and $h ,$ this last condition corresponds to $\nabla h ( x _ { i + 1 } ) =$ $\nabla h ( x _ { i } ) - \nabla f ( x _ { i + 1 } )$ when h is diferentiable.

Before moving to specific sets $S _ { f }$ and $S _ { h }$ , we describe conditions on them to ensure that one can recover $f$ and h interpolating $S _ { f }$ and $S _ { h }$ .

Shape of $f$ and conditions on $S _ { f }$ . The function f is constructed in the form

$$
f ( x ) = \operatorname* { m a x } _ { i \in \{ \star , 1 , 2 , \ldots , k \} } \bigg \{ f _ { i } + \langle s _ { i } ^ { f } , x - x _ { i } \rangle \bigg \} ,\tag{25}
$$

where $x _ { \star } , x _ { 1 } , \ldots , x _ { k } \in \mathbb { R } ^ { d }$ are given points associated with some $\mathbf { \Phi } _  \mathsf { S } _ { \star } ^ { f } , \mathsf { S } _ { 1 } ^ { f } , \mathsf { \Phi } _ { \mathsf { I } } , \mathsf { \Phi } _ { \mathsf { I } } \mathsf { \Phi } _ { \mathsf { I } } , \mathsf { S } _ { k } ^ { f } \in \mathbb { R } ^ { d }$ and $f _ { \star } , f _ { 1 } , \ldots , f _ { k } \in \mathbb { R }$ This construction yields a convex function satisfying $s _ { i } ^ { f } \in \partial f ( x _ { i } )$ and $f _ { i } = f ( x _ { i } )$ , for all i, if and only if

$$
f _ { j } \geqslant f _ { i } + \langle s _ { i } ^ { f } , x _ { j } - x _ { i } \rangle , \qquad \forall i , j \in \{ \star , 1 , 2 , \ldots , k \} .\tag{26}
$$

The proof is straightforward and can be found in, e.g., [22, Section 2.3].

Shape of h and conditions on $S _ { h }$ . For h, we construct it in the form $h ( x ) = - \log ( - \phi ( x ) )$ , with the convention $h ( x ) = + \infty$ when $\phi ( x ) \geqslant 0$ , and with

$$
\phi ( \boldsymbol { x } ) = \operatorname* { m a x } _ { i \in \{ 0 , 1 , \ldots , k \} } \left\{ - \exp ( - h _ { i } ) \left( 1 - \langle s _ { i } ^ { h } , \boldsymbol { x } - \boldsymbol { x } _ { i } \rangle \right) \right\} ,\tag{27}
$$

where similarly $x _ { 0 } , x _ { 1 } , \dotsc , x _ { k } \in \mathbb { R } ^ { d } , s _ { 0 } ^ { h } , s _ { 1 } ^ { h } , \dotsc , s _ { k } ^ { h } \in \mathbb { R } ^ { d }$ and $h _ { 0 } , h _ { 1 } , \ldots , h _ { k } \in \mathbb { R }$ . This time, defining $\phi _ { i } = - \exp ( - h _ { i } )$ and $s _ { i } ^ { \phi } = - \phi _ { i } s _ { i } ^ { h } \left( i = 0 , 1 , \ldots , k \right)$ the constructed (27) is convex and satisfies $s _ { i } ^ { \phi } \in \partial \phi ( x _ { i } )$ and $\phi _ { i } = \phi ( x _ { i } )$ = − exp(−h<sub>i</sub>) for all i if and only if

$$
\phi _ { j } \geqslant \phi _ { i } + \langle s _ { i } ^ { \phi } , x _ { j } - x _ { i } \rangle , \qquad \forall i , j \in \{ 0 , 1 , \ldots , k \} .\tag{28}
$$

Note that $t \mapsto - \log ( - t )$ is increasing and convex, and hence subdiferential calculus $( \mathrm { e . g . , [ 1 2 }$ , Theorem 4.3.1]) allows us to link the subdiferentials of ϕ with those of h. Hence the function $h = - \log ( - \phi )$ is 1-exp-concave with $h ( x _ { i } ) = - \log ( - \phi _ { i } ) = h _ { i }$ and $s _ { i } ^ { h } \in \partial h ( x _ { i } )$ (for all i) if and only if (28). Noting that $\phi _ { i } < 0$ by construction, the corresponding proof is similarly straightforward as in the previous case.

Shape of the domain C. We do not explicitly construct C. Instead, it is implicitly constructed as $\mathcal { C } = \{ x : \phi ( x ) \leqslant 0 \}$ ; naturally, it follows from $\phi$ being convex that its sublevel sets are also convex and hence that C is convex. One can also easily verify that h is a log-barrier for ${ \mathcal { C } } ,$ as dom $h = \operatorname { r i } ( { \mathcal { C } } ) =$ $\{ x \colon \phi ( x ) < 0 \} , h ( x ) = - \log ( - \phi ( x ) )$ ) for x ∈ dom h, and $h ( x ) = + \infty$ otherwise.

## 5.2 Two-dimensional matching example

This section explicitly constructs the sample sets $S _ { f }$ and $S _ { h }$ that attain the bound in (23). We first introduce the quantities needed for the construction, and then verify that $S _ { f }$ and $S _ { h }$ satisfy the corresponding interpolation conditions (26) and (28). Set

$$
\begin{array} { r l r l r l } & { R _ { k } = 1 + \displaystyle \sum _ { i = 1 } ^ { k - 1 } ( \sqrt { i + 1 } - \sqrt { i } ) ^ { 2 } , } & { \gamma _ { i } = \displaystyle ( \sqrt { i + 1 } - \sqrt { i } ) \left( \frac { 1 } { \sqrt { i } } - \frac { 1 } { \sqrt { i + 1 } } \right) = \sqrt { \frac { i + 1 } { i } } + \sqrt { \frac { i } { i + 1 } } - 2 } & & { i = 1 , \dots , k - 1 . } \end{array}
$$

Then define the function values $\{ f _ { i } \} _ { i \in \{ \star , 1 , 2 , \ldots , k \} }$

$$
\begin{array} { r } { f _ { \star } = 0 , \qquad f _ { k } = \frac { R _ { k } } { k } , \qquad f _ { i } = f _ { i + 1 } + \gamma _ { i } , \qquad i = k - 1 , k - 2 , \ldots , 1 , } \end{array}\tag{29}
$$

Finally, define a sequence $\{ d _ { i } \} _ { i \in \{ 0 , 1 , \ldots , k \} }$ as

$$
\begin{array} { r } { d _ { k } = - 1 , \qquad \mathrm { a n d } \qquad d _ { i - 1 } = d _ { i } + f _ { i } + \sqrt { \frac { i - 1 } { i } } - 1 , \qquad i = k , k - 1 , \dots , 1 . } \end{array}\tag{30}
$$

Iterates and gradients. We work in $\mathbb { R } ^ { 2 }$ . Set $x _ { \star } = 0 , s _ { \star } ^ { f } = 0 , s _ { 0 } ^ { h } = 0$ , and define for $i = 1 , \ldots , k .$

$$
x _ { i } = \binom { 1 } { 1 / \sqrt { i } } , \quad s _ { i } ^ { h } = \binom { d _ { i } + 1 } { - \sqrt { i } } , \quad s _ { i } ^ { f } = \binom { d _ { i - 1 } - d _ { i } } { \sqrt { i } - \sqrt { i - 1 } } ,\tag{31}
$$

where one can verify $s _ { i } ^ { f } = s _ { i - 1 } ^ { h } - s _ { i } ^ { h }$ for all $i = 1 , \ldots , k$ . The essential part of this construction is the second coordinate of $x _ { i } , s _ { i } ^ { f }$ , and $s _ { i } ^ { h }$ . The first one only ensures that $\langle s _ { i } ^ { f } , x _ { i } \rangle = f _ { i }$ , as shown below.

Verification of the convex interpolation conditions for $f .$ Now, we verify that (26) holds. Note that, for any $i , j \in \{ 1 , 2 , \ldots , k \}$ , the first coordinate of $s _ { i } ^ { f }$ is irrelevant because $x _ { j } - x _ { i }$ has a zero first coordinate.

$( f . \mathrm { i } ) \ ( i = \star )$ . Since $s _ { \star } ^ { f } = 0$ and $f _ { \star } = 0$ , the interpolation inequalities reduce to $f _ { j } \geqslant 0 , j = 1 , \ldots , k .$ which holds by construction.

(f.ii) $( j = \star )$ . We need to verify $f _ { i } + \langle s _ { i } ^ { f } , x _ { \star } - x _ { i } \rangle \leqslant f _ { \star } = 0$ . This is true due to $\langle s _ { i } ^ { f } , x _ { i } \rangle = f _ { i }$ , which is easy to check

$$
\langle s _ { i } ^ { f } , x _ { i } \rangle = d _ { i - 1 } - d _ { i } + 1 - { \sqrt { \frac { i - 1 } { i } } } = f _ { i } .
$$

(f.iii) (All remaining cases). Let $i \geqslant 1$ . We use $\begin{array} { r } { \langle s _ { i } ^ { f } , x _ { j } - x _ { i } \rangle = ( \sqrt { i } - \sqrt { i - 1 } ) \left( \frac { 1 } { \sqrt { j } } - \frac { 1 } { \sqrt { i } } \right) } \end{array}$ . If j < i, then

$$
\begin{array} { l } { f _ { j } - f _ { i } = \displaystyle \sum _ { \ell = j } ^ { i - 1 } \gamma _ { \ell } = \displaystyle \sum _ { \ell = j } ^ { i - 1 } ( \sqrt { \ell + 1 } - \sqrt { \ell } ) \left( \frac { 1 } { \sqrt { \ell } } - \frac { 1 } { \sqrt { \ell + 1 } } \right) } \\ { \geqslant ( \sqrt { i } - \sqrt { i - 1 } ) \displaystyle \sum _ { \ell = j } ^ { i - 1 } \left( \frac { 1 } { \sqrt { \ell } } - \frac { 1 } { \sqrt { \ell + 1 } } \right) = ( \sqrt { i } - \sqrt { i - 1 } ) \left( \frac { 1 } { \sqrt { j } } - \frac { 1 } { \sqrt { i } } \right) = \langle s _ { i } ^ { f } , x _ { j } - x _ { i } \rangle , } \end{array}
$$

where we used that $\ell \mapsto \sqrt { \ell + 1 } - \sqrt { \ell }$ is decreasing. $\operatorname { I f } j > i ,$ then similarly

$$
f _ { i } - f _ { j } = \sum _ { \ell = i } ^ { j - 1 } \gamma _ { \ell } \leqslant ( { \sqrt { i } } - { \sqrt { i - 1 } } ) \sum _ { \ell = i } ^ { j - 1 } \left( { \frac { 1 } { \sqrt { \ell } } } - { \frac { 1 } { \sqrt { \ell + 1 } } } \right) = ( { \sqrt { i } } - { \sqrt { i - 1 } } ) \left( { \frac { 1 } { \sqrt { i } } } - { \frac { 1 } { \sqrt { j } } } \right) = - \langle s _ { i } ^ { f } , x _ { j } - x _ { i } \rangle .
$$

Thus the interpolation inequalities hold in all remaining cases.

Verification of the exp-concave interpolation conditions for h. Now, we verify that (28) holds. Recall (31) and further define

$$
\phi _ { 0 } = - 1 , \qquad h _ { 0 } = - \log ( - \phi _ { 0 } ) = 0 , \qquad s _ { 0 } ^ { h } = 0 , \qquad s _ { 0 } ^ { \phi } = 0
$$

and, for $i = 1 , \ldots , k$

$$
\begin{array} { r } { \phi _ { i } = - \frac { c } { \sqrt { i } } , \qquad h _ { i } = - \log ( - \phi _ { i } ) , \qquad s _ { i } ^ { \phi } = - \phi _ { i } s _ { i } ^ { h } , } \end{array}
$$

where $0 < c < 1$ . Finally, choose

$$
x _ { 0 } = { \binom { 1 } { 1 / c } } .
$$

We define $x _ { 0 }$ only now, because for (proxMD) interpolation of f does not require $x _ { 0 }$

(ϕ.i) $( i = 0 )$ . Since $s _ { 0 } ^ { \phi } = 0$ , the inequalities reduce to $\phi _ { j } \geqslant \phi _ { 0 } = - 1 , j = 0 , 1 , \ldots , k$ . This is immediate for $j = 0 .$ . For $j = 1 , \dots , k .$ , it follows from $\phi _ { j } = - \textstyle { \frac { c } { \sqrt { j } } } \geqslant - 1$ because $0 < c < 1$

(ϕ.ii) (j = 0). For $i = 1 , \ldots , k .$ , using $s _ { i } ^ { \phi } = - \phi _ { i } s _ { i } ^ { h } , x _ { 0 } - x _ { i } = \binom { 0 } { 1 / c - 1 / \sqrt { i } }$ , we get

$$
\begin{array} { r } { \phi _ { i } + \langle s _ { i } ^ { \phi } , x _ { 0 } - x _ { i } \rangle = \phi _ { i } \left( 1 - \langle s _ { i } ^ { h } , x _ { 0 } - x _ { i } \rangle \right) = - \frac { c } { \sqrt { i } } \left( 1 + \sqrt { i } \left( \frac { 1 } { c } - \frac { 1 } { \sqrt { i } } \right) \right) = - 1 = \phi _ { 0 } , } \end{array}
$$

so that the interpolation inequalities (28) are tight when $j = 0$

(ϕ.iii) $( i , j \in \{ 1 , \ldots , k \} )$ . Using that $s _ { i } ^ { \phi } = - \phi _ { i } s _ { i } ^ { h } , x _ { j } - x _ { i } = \binom { 0 } { 1 / \sqrt { j } - 1 / \sqrt { i } }$ , we have

$$
\begin{array} { r } { \phi _ { i } + \langle s _ { i } ^ { \phi } , x _ { j } - x _ { i } \rangle = \phi _ { i } \left( 1 - \langle s _ { i } ^ { h } , x _ { j } - x _ { i } \rangle \right) = \phi _ { i } \left( 1 + \sqrt { i } \left( \frac { 1 } { \sqrt { j } } - \frac { 1 } { \sqrt { i } } \right) \right) = \phi _ { i } \sqrt { \frac { i } { j } } = \phi _ { j } , } \end{array}
$$

so that the interpolation inequalities (28) are again tight for all $i , j \in \{ 1 , \ldots , k \}$

Domain verification. Finally, we need to show that $\phi ( x _ { \star } ) \geqslant 0 .$ , so that $x _ { \star } \in \mathcal { C }$ . Given the construction (27), it sufices to verify that one term in the max is zero, which follows from $1 - \left. s _ { k } ^ { h } , x _ { \star } - x _ { k } \right. = 0$

Conclusion. We have shown that the data $\{ ( x _ { i } , s _ { i } ^ { f } , f _ { i } ) \} _ { i \in \{ \star , 1 , \ldots , k \} }$ is correctly interpolated by the convex function (25), and that the data $\{ ( x _ { i } , s _ { i } ^ { h } , h _ { i } ) \} _ { i \in \{ 0 , 1 , \ldots , k \} }$ is correctly interpolated by the 1-expconcave function $- \log ( - \phi )$ , with the convex function ϕ from (27). As this two-dimensional construction also satisfies optimality conditions of (proxMD), it does correspond to a possible run of (proxMD) on the problem $\operatorname* { m i n } _ { x \in { \mathcal { C } } } f ( x )$ with barrier h. Finally, as this example achieves (23), we have proved Theorem 3.

## 6 Proximal mirror descent vs interior-point methods

In this section, we always assume that h is a ν-log-barrier.

In the mirror-descent setup, it is commonly assumed that the subproblems in (MD) or (proxMD) are simple enough to be solved in closed form or by inexpensive numerical procedures, such as a onedimensional Newton method; see, for example, [1,13]. In this case, the number of mirror-descent iterations is representative of the total computational cost. This is the first-order point of view — the method has a relatively slow convergence rate but a cheap iteration.

In contrast, IPMs have a fast convergence rate but an expensive iteration that typically requires an inner optimization routine, such as Newton’s method. In this case, the computational complexity is typically measured in terms of the total number of Newton iterations. This is the second-order point of view. Interestingly, both points of view apply to (proxMD), and this section presents the latter.

As we mentioned in the introduction, classical IPMs can be viewed as lazy versions of (proxMD)

$$
\begin{array} { r } { x _ { k } = \underset { x \in S } { \mathrm { a r g m i n } } \left. f ( x ) + \frac { 1 } { \alpha _ { k } } D _ { h } ( x , x _ { 0 } ) \right. , } \end{array}\tag{IPM}
$$

which enjoys the guarantee $( \mathrm { e . g . , [ 1 8 . }$ , Theorem 5.3.10] or (10) with $k = 1 )$ when $x _ { 0 }$ is the analytic center:

$$
f ( x _ { k } ) - f _ { \star } \leqslant \frac { \nu } { \alpha _ { k } } ,
$$

which is fast when $\alpha _ { k }$ grows geometrically $\alpha _ { k } = \alpha _ { k - 1 } ( 1 + \mu )$ for $\mu > 0$ . On the other hand, we know that (proxMD)

$$
\begin{array} { r } { x _ { k } = \underset { x \in \mathcal { S } } { \mathrm { a r g m i n } } \left. f ( x ) + \frac { 1 } { \alpha _ { k } } D _ { h } ( x , x _ { k - 1 } ) \right. , } \end{array}\tag{proxMD}
$$

can also be fast under quite broad assumptions. In particular, Corollary 2 states that, for geometrically growing stepsizes $\alpha _ { k } ,$ one has, crudely speaking and for a fixed $x _ { 0 } \colon$

$$
f ( x _ { k } ) - f _ { \star } \leqslant O \left( \frac { k \nu } { ( 1 + \mu ) ^ { k } } \right) .
$$

Clearly, individual iterations of (IPM) and (proxMD) are computationally equivalent (up to initialization), so it makes sense to conduct a thorough comparison.

We therefore have the following contest: an IPM with rigid assumptions and well-understood complexity versus a newcomer — (proxMD) — which is much more general but also less understood from a complexity perspective. Can the plain proximal method stand up to the established IPM? Not yet, but the contest is not entirely fair: to make the comparison possible, we must work under assumptions designed specifically for IPMs. Nevertheless, we show that, even in this setting, (proxMD) performs only slightly worse.

Below, we provide an idealized and simplified comparison in which all subproblems are solved exactly. This assessment should be understood as a proof of concept, while more detailed analyses, closer to those available for short-step interior-point methods, are left for future work.

## 6.1 Reminders on Newton’s method for self-concordant functions

We briefly recall a standard complexity bound for damped Newton’s method applied to a convex generalized self-concordant function $g : \mathbb { R } ^ { d }  \mathbb { R } \cup \{ + \infty \}$ . Consider

$$
\operatorname* { m i n } _ { y \in \mathbb { R } ^ { d } } { g ( y ) }
$$

and assume that $g$ is proper, convex, bounded below, and attains its minimum at some point $y _ { \star } ~ \in$ int(dom g). We further assume that g is $M _ { g } .$ -self-concordant on int(dom g) in the sense that

$$
\begin{array} { r } { | D ^ { 3 } g ( y ) [ u , u , u ] | \leq 2 M _ { g } \big ( D ^ { 2 } g ( y ) [ u , u ] \big ) ^ { 3 / 2 } \qquad \forall y \in \mathrm { i n t } ( \mathrm { d o m } g ) , \ \forall u \in \mathbb { R } ^ { d } , } \end{array}
$$

for some constant ${ { M } _ { g } } \mathrm { ~ > ~ 0 ~ }$ . Given $y \in \operatorname { i n t } ( \operatorname { d o m } g )$ , the Newton direction is $d ( y ) : = - [ \nabla ^ { 2 } g ( y ) ] ^ { - 1 } \nabla g ( y )$ and the classical Newton iteration is $y _ { + } = y + \gamma d ( y )$ for some $\gamma > 0$ . Standard analyses are stated in terms of convergence of the Newton decrement

$$
\lambda _ { g } ( y ) : = \sqrt { \langle \nabla g ( y ) , [ \nabla ^ { 2 } g ( y ) ] ^ { - 1 } \nabla g ( y ) \rangle } = \| \nabla g ( y ) \| _ { [ \nabla ^ { 2 } g ( y ) ] ^ { - 1 } } ,
$$

and yield global convergence results for various stepsize strategies, such as using a backtracking procedure or damped Newton steps with $\gamma = [ 1 + M _ { g } \lambda _ { g } ( y ) ] ^ { - 1 } ~ [ 1 6 , 1 8 ]$ . Then, standard analyses predict that at most

$$
A M _ { g } ^ { 2 } ( g ( y _ { 0 } ) - g _ { \star } ) + B \log \log \frac { 1 } { \varepsilon }\tag{32}
$$

iterations are required by Newton’s method to find some $\hat { y }$ such that $\lambda _ { g } ( \hat { y } ) \ \leqslant \ \varepsilon ; \ \mathrm { s e e } , \ \mathrm { e . g . } , \ [ 1 8$ , Eq. (5.2.11)]. In (32), the first term captures the cost of globalization and depends on the initial objective gap, while the second term corresponds to the final local quadratic phase where $\gamma \approx 1$ . The precise values of factors A and B are irrelevant for our purposes, as we only rely on the structure of this bound for our qualitative analyses below.

## 6.2 Total number of Newton iterations

This section outlines the computational complexity analysis for the problem

$$
\operatorname* { m i n } _ { x \in { \mathcal { C } } } f ( x ) ,
$$

where we assume that f is M -self-concordant on int(C). We also let h be 1-self-concordant on $\operatorname { i n t } ( \mathcal { C } )$ and the subproblem $x _ { k } = \mathrm { a r g m i n } _ { x } \{ G _ { k } ( x ) \}$ be given by $G _ { k } ( x ) = \alpha _ { k } f ( x ) + D _ { h } ( x , x _ { k - 1 } )$ . It follows from [18, Theorem 5.1.1] and $\alpha _ { k + 1 } \geqslant \alpha _ { k }$ , that $G _ { k }$ is M -self-concordant with $\begin{array} { r } { M _ { k } \leqslant \bar { M } : = \operatorname* { m a x } \left\{ \frac { M _ { f } } { \sqrt { \alpha _ { 1 } } } , 1 \right\} } \end{array}$

Outer iteration count. We aim to find ${ \hat { x } } \in { \mathcal { C } }$ such that $f ( \hat { x } ) - f _ { \star } \leqslant \delta$ for some prescribed accuracy $\delta > 0$ . For completeness, we recall $\textstyle A _ { k } : = \sum _ { i = 1 } ^ { k } \alpha _ { i }$ and define

$$
\mathcal { D } _ { x _ { 0 } , \nu } ^ { \mathrm { p r o x } } : = 2 \nu + \langle \nabla h ( x _ { 0 } ) , x _ { 0 } - x _ { \star } \rangle\tag{33}
$$

for the initialization-dependent constant. Similar to traditional IPMs, we choose for $k \geqslant 1$ some geometrically growing stepsizes $\alpha _ { k + 1 } = \mu A _ { k }$ with $\mu > 0$ for some $A _ { 1 } = \alpha _ { 1 } > 0$ . Using Corollary 2, one thus obtains

$$
f ( x _ { k } ) - f _ { \star } \leqslant \frac { \mathcal { D } _ { x _ { 0 } , \nu } ^ { \mathrm { p r o x } } + ( k - 1 ) \frac { \nu } { 4 } \log ( 1 + \mu ) } { \alpha _ { 1 } ( 1 + \mu ) ^ { k - 1 } } .\tag{34}
$$

Hence, the δ-accuracy requirement is satisfied after at most $\begin{array} { r } { k = O \left( \frac { \log \left( \nu / \delta \right) } { \log \left( 1 + \mu \right) } \right) } \end{array}$ outer iterations.

Inner iteration count. We bound the number of Newton steps necessary to solve the inner optimization problem given the initial estimate $x _ { k - 1 }$ . We assume that $x _ { k - 1 }$ minimizes $G _ { k - 1 }$ exactly. The bound (32) gives $\bar { N _ { k } } \leqslant A \bar { M } ^ { 2 } \bigl ( G _ { k } ( x _ { k - 1 } ) - G _ { k } ( x _ { k } ) \bigr ) + C$ where $N _ { k }$ is the number of inner Newton iterations to solve the k-th subproblem, and $\begin{array} { r } { C = B \log \log \frac { 1 } { \varepsilon } } \end{array}$ is treated as a constant. We proceed as follows:

$$
\begin{array} { l } { G _ { k } ( x _ { k - 1 } ) - G _ { k } ( x _ { k } ) = \alpha _ { k } \big ( f ( x _ { k - 1 } ) - f ( x _ { k } ) \big ) - D _ { h } ( x _ { k } , x _ { k - 1 } ) } \\ { \leqslant \alpha _ { k } \big ( f ( x _ { k - 1 } ) - f ( x _ { k } ) \big ) } \\ { \leqslant \alpha _ { k } \big ( f ( x _ { k - 1 } ) - f _ { \star } \big ) } \\ { \leqslant \mu \left( \mathcal { D } _ { x _ { 0 } , \nu } ^ { \mathrm { p r o x } } + ( k - 2 ) \frac { \nu } { 4 } \log ( 1 + \mu ) \right) , } \end{array}
$$

where the last inequality follows from (34) and $\alpha _ { k } = \mu A _ { k - 1 } = \mu \alpha _ { 1 } ( 1 + \mu ) ^ { k - 2 }$ . Thus, for $k \geqslant 2$ we reach

$$
N _ { k } \leqslant A \bar { M } ^ { 2 } \mu \left( \mathcal { D } _ { x _ { 0 } , \nu } ^ { \mathrm { p r o x } } + \frac { \nu } { 4 } \log ( 1 + \mu ) ( k - 2 ) \right) + C .
$$

In analogous way, for $k = 1$ we have $N _ { 1 } \leqslant A \bar { M } ^ { 2 } \alpha _ { 1 } \big ( f ( x _ { 0 } ) - f _ { \star } \big ) + C .$

Global Newton iteration count. The total number of Newton iterations is therefore upper-bounded by the sum of the previous estimates:

$$
\begin{array} { l } { \displaystyle \sum _ { i = 1 } ^ { k } N _ { i } \leqslant A \bar { M } ^ { 2 } \alpha _ { 1 } \big ( f ( x _ { 0 } ) - f _ { \star } \big ) + C k + A \bar { M } ^ { 2 } \mu \displaystyle \sum _ { i = 2 } ^ { k } \Big ( { \mathcal D } _ { x _ { 0 } , \nu } ^ { \mathrm { p r o x } } + \frac { \nu } { 4 } \log ( 1 + \mu ) ( i - 2 ) \Big ) } \\ { = A \bar { M } ^ { 2 } \alpha _ { 1 } \big ( f ( x _ { 0 } ) - f _ { \star } \big ) + C k + A \bar { M } ^ { 2 } \mu \displaystyle \left( { \mathcal D } _ { x _ { 0 } , \nu } ^ { \mathrm { p r o x } } k + \frac { \nu } { 4 } \log ( 1 + \mu ) \left( \frac { k ( k + 1 ) } { 2 } - 2 k \right) \right) . } \end{array}
$$

For fixed initial point $x _ { 0 }$ and stepsize $\alpha _ { 1 }$ , the leading term in k is $O \left( \mu \nu \log ( 1 + \mu ) k ^ { 2 } \right)$ . From the outer iteration count estimate $\begin{array} { r } { k = O \left( \frac { \log \left( \nu / \delta \right) } { \log \left( 1 + \mu \right) } \right) } \end{array}$ , the number of Newton iterations scales as $\begin{array} { r } { O \left( \frac { \mu \nu } { \log ( 1 + \mu ) } \log ^ { 2 } ( \nu / \delta ) \right) } \end{array}$ Thus, fixing $\mu > 0 ,$ we obtain the complexity bound $O \left( \nu \log ^ { 2 } \frac { 1 } { \delta } \right)$ . For comparison, a similar analysis of IPMs yields the classical bound $\begin{array} { r } { O ( \nu \log \frac { 1 } { \delta } ) } \end{array}$ When the objective is linear, one may also compare with short-step IPMs, which achieve $\begin{array} { r } { O ( \sqrt { \nu } \log \frac { 1 } { \delta } ) } \end{array}$ ; we note again that, for linear objectives, (proxMD) and $( \mathrm { I P M } )$ are equivalent up to a stepsize adjustment. For nonlinear objectives, the standard IPM approach is to move the objective into the constraints, for instance through an epigraph reformulation, so that the nonlinear objective is instead handled through the barrier.

## 7 Remarks and outlook

Technically, this work establishes log(k)/k-type bounds under exp-concavity of the mirror function. While earlier versions of the proof were obtained through analogies with known techniques (inspired by gradient accumulation such as in [24]), the final version presented in Section 3 was refined using performance estimation problems for exp-concave functions, which also allowed us to construct examples on which the main bounds are attained.

Potential ways to improve the bounds include a better model for log-barrier functionals beyond assuming that they are “just” exp-concave, such as log-homogeneity of the barriers, used in the study of interior-point methods. Beyond this, it is known that accelerated convergence rates are out of reach for general mirror descent methods [8], but it remains unclear how much this statement resists to stronger assumptions on the distance-generating function [7]. Other directions include exploring approximate solvers (as in the hybrid proximal extragradient framework [14, 15]) and investigating possible numeri cal advantages of proximal mirror methods combined with interior-point strategies, possibly for specific problem structures. In particular, interior-point strategies require exponentially growing stepsizes, typically driving the corresponding subproblem’s to strong ill-conditioning while proximal mirror schemes (updating the prox-center) allow for bounded but potentially large stepsizes.

## Acknowledgments

This research is supported by the European Union (ERC grant CASPER 101162889), by the Agence Nationale de la Recherche through the France 2030 program (grant reference ANR-23-IACL-0008, PR[AI]RIE-PSAI), and by the Austrian Science Fund (FWF 10.55776/STA223). Views and opinions expressed are those of the authors only and do not necessarily reflect those of the funding agencies or granting authorities, which cannot be held responsible for them.

## References

[1] H. H. Bauschke, J. Bolte, and M. Teboulle. A descent lemma beyond Lipschitz gradient continuity: First-order methods revisited and applications. Mathematics of Operations Research, 42(2):330–348, 2017. doi:10.1287/moor.2016.0817.

[2] A. Beck and M. Teboulle. Mirror descent and nonlinear projected subgradient methods for convex optimization. Operations Research Letters, 31(3):167–175, 2003. doi:10.1016/S0167-6377(02)00231-6.

[3] B. Birnbaum, N. R. Devanur, and L. Xiao. Distributed algorithms via gradient descent for Fisher markets. In Proceedings of the 12th ACM conference on Electronic commerce, pages 127–136, New York, NY, USA, 2011. Association for Computing Machinery. doi:10.1145/1993574.1993594.

[4] S. Cipolla and J. Gondzio. Proximal stabilized interior point methods and low-frequency-update preconditioning techniques. Journal of Optimization Theory and Applications, 197(3):1061–1103, 2023. doi:10.1007/s10957-023-02194-4

[5] A. De Marchi, Y. Malitsky, and A. B. Taylor. Mirror performance estimation with composed distancegenerating functions, 2026. In preparation.

[6] M. Doljansky and M. Teboulle. An interior proximal algorithm and the exponential multiplier method for semidefinite programming. SIAM Journal on Optimization, 9(1):1–13, 1998. doi:10.1137/S1052623496309405.

[7] R.-A. Dragomir. Bregman gradient methods for relatively-smooth optimization. PhD thesis, UT1 Capitole, 2021. URL https://inria.hal.science/tel-03389344.

[8] R.-A. Dragomir, A. B. Taylor, A. d’Aspremont, and J. Bolte. Optimal complexity and certification of Bregman first-order methods. Mathematical Programming, 194(1):41–83, 2022. doi:10.1007/s10107-021- 01618-1.

[9] Y. Drori and M. Teboulle. Performance of first-order methods for smooth convex minimization: a novel approach. Mathematical Programming, 145(1):451–482, 2014. doi:10.1007/s10107-013-0653-0.

[10] J. Eckstein. Nonlinear proximal point algorithms using Bregman functions, with applications to convex programming. Mathematics of Operations Research, 18(1):202–226, 1993. doi:10.1287/moor.18.1.202.

[11] J. Gondzio. Interior point methods in the year 2025. EURO Journal on Computational Optimization, 13:100105, 2025. doi:10.1016/j.ejco.2025.100105.

[12] J.-B. Hiriart-Urruty and C. Lemar´echal. Convex analysis and minimization algorithms I: Fundamentals, volume 305. Springer science & business media, 2013. doi:10.1007/978-3-662-02796-7.

[13] H. Lu, R. M. Freund, and Y. Nesterov. Relatively smooth convex optimization by first-order methods, and applications. SIAM Journal on Optimization, 28(1):333–354, 2018. doi:10.1137/16M1099546.

[14] R. D. Monteiro and B. F. Svaiter. On the complexity of the hybrid proximal extragradient method for the iterates and the ergodic mean. SIAM Journal on Optimization, 20(6):2755–2787, 2010. doi:10.1137/090753127.

[15] R. D. Monteiro and B. F. Svaiter. An accelerated hybrid proximal extragradient method for convex optimization and its implications to second-order methods. SIAM Journal on Optimization, 23(2):1092–1125, 2013. doi:10.1137/110833786.

[16] A. S. Nemirovski and M. J. Todd. Interior-point methods for optimization. Acta Numerica, 17:191–234, 2008. doi:10.1017/S0962492906370018.

[17] A. S. Nemirovskii and D. B. Yudin. Problem complexity and method eficiency in optimization. Wiley, New York, 1983. doi:10.1137/1027074.

[18] Y. Nesterov. Lectures on Convex Optimization, volume 137. Springer, 2nd edition, 2018. doi:10.1007/978- 3-319-91578-4.

[19] R. Polyak. Modified barrier functions (theory and methods). Mathematical programming, 54(1):177–222, 1992. doi:10.1007/BF01586050.

[20] S. Pougkakiotis and J. Gondzio. An interior point-proximal method of multipliers for convex quadratic programming. Computational Optimization and Applications, 78(2):307–351, 2021. doi:10.1007/s10589-020- 00240-9.

[21] R. T. Rockafellar. Augmented Lagrangians and applications of the proximal point algorithm in convex programming. Mathematics of operations research, 1(2):97–116, 1976. doi:10.1287/moor.1.2.97.

[22] A. B. Taylor, J. M. Hendrickx, and F. Glineur. Smooth strongly convex interpolation and exact worst-case performance of first-order methods. Mathematical Programming, 161(1):307–345, 2017. doi:10.1007/s10107- 016-1009-3.

[23] M. Teboulle. Entropic proximal mappings with applications to nonlinear programming. Mathematics of Operations Research, 17(3):670–690, 1992. doi:10.1287/moor.17.3.670.

[24] M. Teboulle and Y. Vaisbourd. An elementary approach to tight worst case complexity analysis of gradient based methods. Mathematical Programming, 201:63–96, 2023. doi:10.1007/s10107-022-01899-0.

[25] P. Tseng and D. P. Bertsekas. On the convergence of the exponential multiplier method for convex programming. Mathematical Programming, 60(1):1–19, 1993. doi:10.1007/BF01580598.