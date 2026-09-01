# KOLMOGOROV–ARNOLD AGAINST BOUNDED TRANSLATIONS

SVIATOSLAV V. DZHENZHER

Abstract. Historically originating from Hilbert’s 13th problem, the Kolmogorov–Arnold representation theorem (KART) has recently experienced a major revitalisation through its applications to neural networks, specifically Kolmogorov–Arnold Networks (KANs). While the exact representation is well established, its stability under continuous adversarial perturbations of the hidden layer remains a critical open question. In this paper, we investigate the robustness of KART against bounded adversarial translations. We provide an explicit, self-contained, and constructive proof of an approximate representation using fixed, piecewise linear “inner” functions. Crucially, our construction employs a single “outer” function that remains invariant for all summands and is independent of the specific adversarial translation, provided its maximum bound is known a priori.

Keywords: Kolmogorov–Arnold representation theorem, superposition of functions, Hilbert’s 13th problem, adversarial robustness, bounded translations, modulus of continuity, Kolmogorov–Arnold Networks (KANs).

MSC 2020: Primary: 26B40, 41A63; Secondary: 26B35, 68T07.

## 1. History and introduction

Historically, this field of research originates from Hilbert’s 13th problem [1, 2]. In 1836, Hamilton showed [3] that any seventh-degree equation can be reduced to an equation of the form

$$
x ^ { 7 } + a x ^ { 3 } + b x ^ { 2 } + c x + 1 = 0 .
$$

For the quadratic equation $a x ^ { 2 } + b x + c = 0$ , we all know the school formula that the solution to the equation $\textstyle x = x ( a , b , c ) = { \frac { - b \pm { \sqrt { b ^ { 2 } - 4 a c } } } { 2 a } }$ , considered a continuous function of three variables, can be expressed as the superposition of arithmetic operations. Analogously, Hilbert asked whether the solution $x = x ( a , b , c )$ of the seventh-degree equation can be expressed as the superposition of functions of two variables. For continuous functions, the problem was resolved afirmatively in 1956–57 by Kolmogorov and Arnold in their companion papers [4–6]. Nowadays, their result is known as the Kolmogorov–Arnold representation theorem (KART; see Theorem 2.1; in computer science, it is sometimes referred to as the Kolmogorov Superposition Theorem).

Initially, KART faced long-standing criticism for its non-constructive nature and the low smoothness of the resulting inner and outer functions. Specifically, Vitushkin [7, 8] and Fridman [9] demonstrated strict limitations regarding the diferentiability of these representations. It might seem amusing that after half a century, even Ismailov’s version [10] for discontinuous functions has found practical applications. Another line of research was strictly topological, where Ostrand [11] generalised KART to functions on finite-dimensional compact spaces, and Sternfeld [12, 13] proved the lower bound on the dimension of these compact sets (see also [14] for an overview). Nearly three decades passed before it was recognised [15–17] that KART also provides a significant contribution to the topic of neural networks (NNs). The approach was completely revitalised in 2024 by Kolmogorov–Arnold Networks (KANs) [18–20].

In light of the many practical implications of KART, the question of the stability of the Kolmogorov– Arnold representation was posed in [21]. More specifically, the sensitivity of KART under continuous reparameterisations of the hidden layer was investigated. Unlike coding theory, where the action of the noise is unknown, it was assumed that the action of the reparameterisation is known and that the output layer can be adjusted. A useful analogy is that of a known adversary attempting to perturb the hidden layer, and we try to withstand this attack by varying the output layer.

In [21], the stability of KART under countable families of adversarial homeomorphisms was established, and several natural questions were posed. In this paper, we address some of them.

The paper is organised as follows. Section 2 presents the necessary background and the main result (Theorem 2.3). Section 3 provides the proof of the main result. Section 4 is devoted to the discussion of results and their place among other results on the topic. Section 5 concludes the paper and outlines further research.

## 2. Background and main results

We say that a continuous tuple $\psi \colon [ 0 , 1 ] \  \ [ 0 , 1 ] ^ { k }$ is a tuple consisting of continuous functions $\psi _ { 1 } , . . . , \psi _ { k } \colon [ 0 , 1 ] \ \to \ [ 0 , 1 ]$ . Analogously, we say that a continuous matrix $\phi \colon [ 0 , 1 ] \  \ [ 0 , 1 ] ^ { m \times k }$ is a tuple consisting of continuous tuples $\phi _ { 1 } , \ldots , \phi _ { m } \colon [ 0 , 1 ] \to [ 0 , 1 ] ^ { k }$ ; in other words, this is a matrix consisting of continuous functions $\phi _ { i , j } \colon [ 0 , 1 ] \to [ 0 , 1 ]$ . We will use the analogous terminology for tuples and matrices with other continuity properties and ranges.

Theorem 2.1 (Kolmogorov, Arnold; 1956–57). Let $n > 1$ be an integer.

There exists a fixed “inner” continuous matrix $\phi \colon [ 0 , 1 ]  [ 0 , 1 ] ^ { n \times 2 n + 1 }$ such that

$$
f \colon [ 0 , 1 ] ^ { n } \to { \bar { \mathbb { R } } }
$$

there exist “outer” uniformly continuous functions $g _ { 1 } , . . . , g _ { 2 n + 1 } \colon \mathbb { R } \to \mathbb { R }$ such that

$$
f ( x ) = \sum _ { i = 1 } ^ { 2 n + 1 } g _ { i } \left( \sum _ { j = 1 } ^ { n } \phi _ { j , i } ( x _ { j } ) \right) \quad f o r a n y \quad x = ( x _ { 1 } , \ldots , x _ { n } ) \in [ 0 , 1 ] ^ { n } .
$$

In [22], the exposition is presented in which the inner functions are chosen of the form $\phi _ { j , i } ( x _ { j } ) =$ $\lambda _ { j } \phi _ { i } ( x _ { j } )$ for $\lambda _ { 1 } , \ldots , \lambda _ { n }$ being any rationally independent numbers; moreover, the outer functions $g _ { i }$ can be chosen equal. In [10], it is shown that the theorem holds for discontinuous functions $f$ if we allow the “outer” functions $g _ { i }$ to be discontinuous. See other versions of KART with diferent “inner” and “outer” functions in [23–25].

The following is the version from [21] on the stability of KART under “adversarial” attacks on the “inner” layer.

Theorem 2.2 (Dzhenzher, Freedman; 2025). Let $n > 1$ be an integer. Let H be a countable set of homeomorphisms $\mathbb { R } ^ { 2 n + 1 } \to { \dot { \mathbb { R } } } ^ { 2 n + 1 }$

There exists a fixed “inner” continuous matrix $\phi \colon [ 0 , 1 ]  [ 0 , 1 ] ^ { n \times 2 n + 1 }$ such that

for any “target” continuous function $f \colon [ 0 , 1 ] ^ { n }  \mathbb { R }$ and for any “adversarial” $h = ( h _ { 1 } , \ldots , h _ { n } ) \in { \mathcal { H } } ^ { n }$ 2 there exists a uniformly continuous function $g \colon { \mathbb { R } } $ R such that

$$
f ( x ) = \sum _ { i = 1 } ^ { 2 n + 1 } g \left( \sum _ { j = 1 } ^ { n } h _ { j , i } { \big ( } \phi _ { j } ( x _ { j } ) { \big ) } \right) \quad f o r \quad x = ( x _ { 1 } , \ldots , x _ { n } ) \in [ 0 , 1 ] ^ { n } .
$$

For example, if we take all $h _ { j }$ to be identities, then this result is a reformulation of Theorem 2.1. An analogous result was proved for the “inner” functions of the form $\phi _ { j , i } ( x _ { j } ) ~ = ~ \lambda _ { j } \phi _ { i } ( x _ { j } )$ , with the “adversarial” homeomorphism acting on $\phi$ but not on λ.

In [21], dificulties regarding equicontinuity arose, preventing similar results from being established for continuous groups of homeomorphisms. It was mentioned that the approximation trivially works for adversaries from the group of translations $\mathbb { R } ^ { 2 n + 1 } \to \mathbb { R } ^ { 2 n + 1 }$ if we are allowed to use diferent “outer” functions $g _ { i }$ . The following theorem partially solves the case for the unique “outer” function $g .$

Hereafter, $| \cdot | _ { \infty }$ is the max-metric in R<sup>m</sup>, and ∥·∥ is the sup-norm for continuous functions. Recall that the modulus of continuity $\omega _ { f } \colon [ 0 , + \infty )  [ 0 , + \infty )$ of the continuous function $f \colon [ 0 , 1 ] ^ { n }  \mathbb { R }$ is

$$
\omega _ { f } ( d ) : = \operatorname* { s u p } \{ | f ( x ) - f ( y ) | : | x - y | _ { \infty } \leqslant d \} .
$$

Theorem 2.3. Let $n > 1$ and $m \geqslant n + 1$ be integers. Let $C > 0 , \delta _ { 0 } > 0$ and $\textstyle 0 < \gamma < { \frac { 1 } { m } }$ be real numbers. Then

there exists an “inner” piecewise linear matrix $\phi \colon [ 0 , 1 ]  \mathbb { R } ^ { n \times m }$ such that

for any “target” continuous function $f \colon [ 0 , 1 ] ^ { n }  \mathbb { R }$

there exists an “outer” piecewise linear $\frac { 2 \| f \| } { \delta _ { 0 } ( m - n ) }$ -Lipschitz function $g \colon { \mathbb { R } }  { \mathbb { R } }$ such that $\begin{array} { r } { \| g \| \leqslant \frac { 1 } { m - n } } \end{array}$ ∥f∥   
and   
for any “adversarial” $t \in [ - C , C ] ^ { m }$

$$
\left| f ( x ) - \sum _ { i = 1 } ^ { m } g \left( \sum _ { j = 1 } ^ { n } \phi _ { j , i } ( x _ { j } ) + t _ { i } \right) \right| \leqslant \omega _ { f } ( \gamma ( m - 1 ) ) + { \frac { n } { m - n } } \left\| f \right\| \quad f o r \quad x = ( x _ { 1 } , \ldots , x _ { n } ) \in [ 0 , 1 ] ^ { n } .
$$

## 3. Proof of Theorem 2.3

This entire section is devoted to the proof of Theorem 2.3. The section is structured so that necessary constructions come first, followed by proofs that these constructions satisfy the required properties. Some technical facts are collected in the four lemmas below.

Denote $[ k ] : = \{ 1 , \dots , k \}$ for an integer k.

In the following three paragraphs, we construct the “inner” matrix $\phi \colon [ 0 , 1 ]  \mathbb { R } ^ { n \times m }$

The red segments of rank $i \in [ m ]$ are

$$
[ 0 , 1 ] \cap [ \gamma ( m k + i - ( m - 1 ) ) , \gamma ( m k + i ) ] \quad { \mathrm { ~ f o r ~ } } \ k \in \mathbb { Z } ;
$$

these are mostly closed segments of length $\gamma ( m - 1 )$ with possibly shorter segments or points near {0, 1}. Let $\begin{array} { r } { K \approx \lceil \frac { 1 } { \gamma m } \rceil } \end{array}$ be the maximal number of non-empty red segments of the same rank. Denote

$$
\delta : = 2 C + \delta _ { 0 } \quad \mathrm { a n d } \quad \Delta : = \delta K ^ { n } ;
$$

these represent the step sizes in the construction of $\phi .$

Now, let red segments of rank i be numbered like $S _ { i , 0 } , S _ { i , 1 } , \ldots , S _ { i , K _ { i } }$ in order from left to right, $K _ { i } < K$ For $i \in [ m ] , j \in [ n ]$ , construct piecewise linear functions $\phi _ { j , i }$ as follows: make them constant on the red segments $S _ { i , k }$ and equal to

$$
\varphi _ { j , i , k } : = i \Delta + \delta K ^ { j - 1 } k = \delta ( i K ^ { n } + k K ^ { j - 1 } ) ,
$$

and extend them linearly from these $\mathrm { \Omega ^ { \circ } f o r c e d { \Omega } ^ { \prime } }$ values between the red segments.

Now, fix any continuous function $f \colon [ 0 , 1 ] ^ { n }  \mathbb { R }$ . We need to construct the “outer” function g. A red n-solid of rank $i \in [ m ]$ is the Cartesian product of n red segments of rank $i .$ . For each red n-solid $S .$ choose a point $p ( S ) \in S$ . For an n-solid $S = S _ { i , k _ { 1 } } \times \ldots \times S _ { i , k _ { n } }$ , denote

$$
\Phi ( S ) : = \sum _ { j = 1 } ^ { n } \varphi _ { j , i , k _ { j } } \in [ i n \Delta , i n \Delta + \Delta - \delta ] = : W _ { i } .
$$

Finally, define the piecewise linear function $g \colon { \mathbb { R } }  { \mathbb { R } }$ by the “forced” values

$$
g ( y ) : = \frac { f ( p ( S ) ) } { m - n } , \quad y \in [ \Phi ( S ) - C , \Phi ( S ) + C ] .
$$

The definition is meaningful since the segments $\left[ \Phi ( S ) - C , \Phi ( S ) + C \right]$ do not intersect due to the following two lemmas.

Lemma 3.1. For two diferent n-solids $S , S ^ { \prime }$ of the same rank,

$$
| \Phi ( S ) - \Phi ( S ^ { \prime } ) | \geqslant \delta .
$$

Proof. This follows since $\Phi ( S )$ is, up to a factor of δ, a number in the numeral system with base K. □ Lemma 3.2. For two diferent n-solids $S , S ^ { \prime }$ of diferent ranks,

$$
| \Phi ( S ) - \Phi ( S ^ { \prime } ) | \geqslant ( n - 1 ) \Delta + \delta .
$$

Proof. This follows since the lower bound is the distance between neighbouring windows $W _ { i } , W _ { i + 1 }$ □

The fact that g is $\frac { 2 \| f \| } { \delta _ { 0 } ( m - n ) }$ -Lipschitz follows, since, by the previous lemmas, the distances between diferent intervals $[ \Phi ( \tilde { S ) } - \tilde { C } , \Phi ( S ) + C ]$ are at least $\delta - 2 C = \delta _ { 0 }$

Take any $t \in [ - C , C ] ^ { m }$ . Fix an arbitrary $x \in [ 0 , 1 ] ^ { n }$ . It remains to prove the final upper bound:

$$
\left| f ( x ) - \sum _ { i = 1 } ^ { m } g \left( \sum _ { j = 1 } ^ { n } \phi _ { j , i } ( x _ { j } ) + t _ { i } \right) \right| \leqslant \omega _ { f } ( \gamma ( m - 1 ) ) + { \frac { n } { m - n } } \left\| f \right\| .
$$

Lemma 3.3. Any point $x \in [ 0 , 1 ] ^ { n }$ lies in red n-solids of at least $m - n$ diferent ranks.

Proof. The argument is well known; for example, see [24]. It is suficient to prove that any $s \in [ 0 , 1 ]$ is placed in red segments of at least $m - 1$ diferent ranks. This holds since

$$
s \in [ 0 , 1 ] \cap [ \gamma ( m k + i - ( m - 1 ) ) , \gamma ( m k + i ) ] \Longleftrightarrow \frac { s } { \gamma } \in [ m k + i - ( m - 1 ) , m k + i ] ,
$$

and taking k such that $\frac { s } { \gamma } \in [ m k - ( m - 1 ) , m k + 1 )$ , we obtain that the inclusion

$$
\frac { s } { \gamma } \in [ m k + i - ( m - 1 ) , m k + i ]
$$

holds for at least $m - 1$ diferent ranks i.

By Lemma 3.3, for the given x we are able to take $m - n$ red n-solids $S _ { i } \ni x$ of diferent ranks $i \in I$ where the size of I is exactly $m - n$ . Then

$$
\boxed { f ( x ) - \sum _ { i = 1 } ^ { m } g \left( \sum _ { j = 1 } ^ { n } \phi _ { j , i } ( x _ { j } ) + t _ { i } \right) } \Bigg | \leqslant \sum _ { i \in I } \left| \frac { f ( x ) } { m - n } - g \left( \sum _ { j = 1 } ^ { n } \phi _ { j , i } ( x _ { j } ) + t _ { i } \right) \right| + \sum _ { i \notin I } \left| g \left( \sum _ { j = 1 } ^ { n } \phi _ { j , i } ( x _ { j } ) + t _ { i } \right) \right| ,
$$

and the following two lemmas conclude the proof.

Lemma 3.4.

$$
\sum _ { i \in I } \left. { \frac { f ( x ) } { m - n } } - g \left( \sum _ { j = 1 } ^ { n } \phi _ { j , i } ( x _ { j } ) + t _ { i } \right) \right. \leqslant \omega _ { f } ( \gamma ( m - 1 ) ) .
$$

Proof. Fix any $i \in I$ . Then

$$
\sum _ { j = 1 } ^ { n } \phi _ { j , i } ( x _ { j } ) = \Phi ( S _ { i } ) .
$$

Since $| t | _ { \infty } \leqslant C$ , and since g is constant on the intervals $\left[ \Phi ( S ) - C , \Phi ( S ) + C \right]$

$$
g \left( \sum _ { j = 1 } ^ { n } \phi _ { j , i } ( x _ { j } ) + t _ { i } \right) = g { \Big ( } \Phi ( S _ { i } ) { \Big ) } = { \frac { f ( p ( S _ { i } ) ) } { m - n } } .
$$

Since $| x - p ( S _ { i } ) | _ { \infty } \leqslant \gamma ( m - 1 )$

$$
| f ( x ) - f ( p ) | \leqslant \omega _ { f } ( \gamma ( m - 1 ) ) ,
$$

and thus each summand in the statement does not exceed $\frac { \omega _ { f } ( \gamma ( m - 1 ) ) } { m - n }$

Lemma 3.5.

$$
\sum _ { i \not \in I } \left| g \left( \sum _ { j = 1 } ^ { n } \phi _ { j , i } ( x _ { j } ) + t _ { i } \right) \right| \leqslant { \frac { n } { m - n } } \left\| f \right\| .
$$

Proof. It is clear since each summand does not exceed $\frac { \| f \| } { m - n }$

## 4. Discussion of the result

First, note that the idea of the proof relies on the combinatorial argument about red n-solids covering points, but it uses this argument slightly diferently. In the classical expositions [22, 23], the “inner” functions were constructed via rationally independent numbers (as linear rational combinations of square roots of prime numbers, to be precise). In our proof, we designed the “inner” functions in such a way that their values on the red segments are located far enough apart. It is worth mentioning that in the original KART and in [21], the “inner” matrices $\phi$ took values in the unit cube, while in Theorem 2.3, the “inner” functions take values in the entire R. Furthermore, as the perturbation C increases, the wider the range of the “inner” matrix becomes. Consequently, it could be argued that the problem posed in [21] is addressed at the expense of a minor trade-of.

The problem with the result being only an approximate, rather than an exact, outcome is that we cannot control the modulus of continuity of the function $f .$ If we knew that $f$ is L-Lipschitz, then we would be able to take $\gamma$ small enough so that

$$
\omega _ { f } ( \gamma ( m - 1 ) ) \leqslant \gamma ( m - 1 ) L < \varepsilon ,
$$

and we would obtain the approximation with an error of at most

$$
\varepsilon + { \frac { n } { m - n } } \left\| f \right\| ;
$$

for $m = 2 n + 1$ , it is exactly like [21, inequalities (1, 2)]. However, since the space of Lipschitz functions with bounded Lipschitz constants is not separable, we cannot approximate a function $f$ with a smooth enough Lipschitz function and fuel the recursive approximation, as is customary in classical expositions (for example, see [21, statements and applications of lemmas 3.2 and 4.2]). For the same reasons, we cannot prove the openness and the denseness of the appropriate matrices $\phi$ in order to apply the Baire category argument (which is originally due to Kahane [26]).

Note also that, despite the fact that Theorem 2.3 is formulated for m $\geqslant n + 1$ , it works well only for $m \geqslant 2 n + 1$ . Indeed, for $n + 1 \leqslant m \leqslant 2 n$ , the error ${ \frac { n } { m - n } } \left\| f \right\|$ can be defeated by the error $\| f \|$ using the function $g = 0$ without any complex preparations. The choice of $m$ involves a trade-of between two competing factors: on the one hand, with the growth of $m$ , the error $\omega _ { f } ( \gamma ( m - 1 ) )$ increases; on the other hand, the error ${ \frac { n } { m - n } } \left\| f \right\|$ decreases.

Finally, note that in [21], the “outer” function depends on the choice of the “adversarial” reparameterisation, while in Theorem 2.3, it does not. The cost of this independence is that we must know the maximal scope of the “adversarial” translation even before we construct the “inner” functions.

## 5. Conclusion and further research

In this paper, we have investigated the stability of the Kolmogorov–Arnold representation under “adversarial” reparameterisations of the hidden layer. The main result (Theorem 2.3) gives an explicit, piecewise linear construction of an inner matrix $\phi \colon [ 0 , 1 ]  \mathbb { R } ^ { n \times m }$ that satisfies two important additional properties: the “outer” function $g$ is taken to be the same for all m summands, and the construction (in contrast to Theorem 2.2) does not depend on the particular “adversarial” translation $t \in [ - C , C ] ^ { m }$ provided that the bound C is known in advance. The latter fact is the price we pay: the “inner” functions $\phi _ { j , i }$ have to take values in the whole real line, and the larger the allowed scope of the adversary, the wider the range of $\phi$

The proof of Theorem 2.3 is constructive and self-contained. It is interesting to compare this proof with the similar ideas from [27–29], where the first constructive proof of the KART was given.

In the course of the proof, we identified two important structural limitations of the result. First, the upper bound on the translation has to be known before the “inner” function is constructed, which makes the result “non-adaptive” with respect to the reparameterisation. Second, the loss of control of the modulus of continuity of f prevents us from obtaining a sharp result. Whether either of these obstructions can be overcome by a more sophisticated choice of $\phi$ is, to the best of our knowledge, an open question.

We believe that the techniques introduced in this paper can serve as a starting point for a finer study of the stability of KART under larger classes of adversarial actions, and we hope to return to these questions in future work.

## References

[1] Hilbert’s 13th problem. url: https://en.wikipedia.org/wiki/Hilbert%27s\_thirteenth\_ problem.

[2] S. A. Morris. “Hilbert 13: Are there any genuine continuous multivariate real-valued functions?” In: Bull. AMS 58.1 (2021), pp. 107–118. doi: 10.1090/bull/1698.

[3] W. R. Hamilton. “Inquiry into the Validity of a Method recently proposed by George B. Jerrard, Esq. for Transforming and Resolving Equations of Elevated Degrees”. In: Report of the Sixth Meeting of the British Association for the Advancement of Science (1837), pp. 295–348.

[4] A. N. Kolmogorov. “On the representation of continuous functions of several variables by superpositions of continuous functions of a smaller number of variables”. In: Dokl. Akad. Nauk SSSR 108 (1956), pp. 179–182. English translation: Twelve Papers on Algebra and Real Functions. Vol. 17. Amer. Math. Soc. Transl. (Ser. 2). 1961, pp. 369–373.

[5] V. I. Arnold. “On functions of three variables”. In: Dokl. Akad. Nauk SSSR 114 (1957), pp. 679– 681. English translation: Sixteen Papers on Analysis. Vol. 28. Amer. Math. Soc. Transl. (Ser. 2). 1963, pp. 51–54.

[6] A. N. Kolmogorov. “On the representation of continuous functions of many variables by superposition of continuous functions of one variable and addition”. In: Dokl. Akad. Nauk SSSR 114 (5 1957), pp. 953–965. English translation: in: Amer. Math. Soc. Transl. 28 (1963), pp. 55–59.

[7] A. G. Vitushkin. “On Hilbert’s thirteenth problem”. In: Dokl. Akad. Nauk SSSR 96 (1954). In Russian, pp. 701–704.

[8] A. G. Vitushkin. “On Hilbert’s thirteenth problem and related questions”. In: Russ. Math. Surv. 59 (1 2004), pp. 11–25. doi: 10.1070/RM2004v059n01ABEH000698.

[9] B. L. Fridman. “An improvement in the smoothness of the functions in A. N. Kolmogorov’s theorem on superpositions (in Russian)”. In: Dokl. Akad. Nauk SSSR 177 (5 1967), pp. 1019–1022. url: http://mi.mathnet.ru/dan33525.

[10] V. E. Ismailov. “Addressing common misinterpretations of KART and UAT in neural network literature”. In: Neural Networks 196 (2026), p. 108361. issn: 0893-6080. doi: 10.1016/j.neunet. 2025.108361

[11] P. A. Ostrand. “Dimension of metric spaces and Hilbert’s problem 13”. In: Bull. AMS 71 (1965), pp. 619–622. doi: 10.1090/S0002-9904-1965-11363-5.

[12] Y. Sternfeld. “Dimension, superposition of functions and separation of points in compact metric spaces”. In: Isr. J. Math. 50 (1985), pp. 13–53.

[13] Y. Sternfeld. “Hilbert’s 13th problem and dimension”. In: Lect. Notes Math. Vol. 1376. Springer, 1989, pp. 1–49.

[14] S. Dzhenzher. “A simpler proof of Sternfeld’s Theorem”. In: J. Topol. Anal. 17.06 (2025), pp. 1787– 1794. doi: 10.1142/S1793525324500080.

[15] R. Hecht-Nielsen. “Kolmogorov’s Mapping Neural Network Existence Theorem”. In: 1987. url: https://api.semanticscholar.org/CorpusID:118526925.

[16] V. Kůrková. “Kolmogorov’s Theorem Is Relevant”. In: Neural Comput. 3 (4 1991). PMID: 31167327, pp. 617–622. doi: 10.1162/neco.1991.3.4.617.

[17] V. Maiorov and A. Pinkus. “Lower bounds for approximation by MLP neural networks”. In: Neurocomputing 25.1 (1999), pp. 81–91. issn: 0925-2312. doi: 10.1016/S0925-2312(98)00111-8.

[18] Z. Liu et al. “KAN: Kolmogorov-Arnold Networks” (2024). arXiv:2404.19756.

[19] Z. Liu et al. “KAN 2.0: Kolmogorov-Arnold Networks Meet Science” (2024). arXiv:2408.10205.

[20] A. D. Bodner et al. “Convolutional Kolmogorov–Arnold Networks” (2024). arXiv:2406.13155.

[21] S. V. Dzhenzher and M. H. Freedman. “Kolmogorov–Arnold stability”. In: Adv. Theor. Math. Phys. 29.8 (2025), pp. 2285–2303. doi: https://dx.doi.org/10.4310/ATMP.260412174106.

[22] G. Lorentz, M. Golitschek, and Y. Makovoz. Constructive Approximation: Advanced Problems. 1996.

[23] T. Hedberg. “The Kolmogorov superposition theorem, Appendix II to H.S.Shapiro, Topics in Approximation Theory”. In: Lecture Notes in Math. Vol. 187. Springer, 1971, pp. 267–275.

[24] V. Brattka. “From Hilbert’s 13th Problem to the theory of neural networks: constructive aspects of Kolmogorov’s Superposition Theorem”. In: Kolmogorov’s Heritage in Mathematics. Ed. by É. Charpentier, A. Lesne, and N. K. Nikolski. Berlin, Heidelberg: Springer Berlin Heidelberg, 2007, pp. 253–280. isbn: 978-3-540-36351-4. doi: 10.1007/978-3-540-36351-4\_13.

[25] D. A. Sprecher. “On the structure of continuous functions of several variables”. In: Trans. AMS. 3rd ser. 115 (1965), pp. 340–355.

[26] J. Kahane. “Sur le theoreme de superposition de Kolmogorov”. In: J. Approx. Theory 13 (1975), pp. 229–234.

[27] D. A. Sprecher. “A Numerical Implementation of Kolmogorov’s Superpositions”. In: Neural Networks 9.5 (1996), pp. 765–772. issn: 0893-6080. doi: 10.1016/0893-6080(95)00081-X.

[28] D. A. Sprecher. “A Numerical Implementation of Kolmogorov’s Superpositions II”. In: Neural Networks 10.3 (1997), pp. 447–457. issn: 0893-6080. doi: 10.1016/S0893-6080(96)00073-1.

[29] J. Braun and M. Griebel. “On a Constructive Proof of Kolmogorov’s Superposition Theorem”. In: Constr. Approx. 30.3 (2009), pp. 653–675. doi: 10.1007/s00365-009-9054-2.