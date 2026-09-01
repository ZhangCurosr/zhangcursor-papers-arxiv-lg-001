# Exact Recovery Thresholds for Weighted Data Selection in Vector-Valued Linear Regression

Guangjian Zhang zgj1226029469@outlook.com

August 31, 2026

## Abstract

We resolve the threshold part of Question 4 of the COLT 2025 open problem “Data Selection for Regression Tasks” of Hanneke, Moran, Shlimovich and Yehudayof. In vectorvalued linear regression with square loss $\ell _ { ( x , y ) } ( W ) = \| W x - y \| _ { 2 } ^ { 2 }$ , where $x \in \mathbb { R } ^ { d } , y \in \mathbb { R } ^ { m }$ and the learner is the empirical risk minimizer of minimal Frobenius norm, we prove that the minimal budget of weighted examples that recovers the full-data loss on every finite dataset is exactly

$$
n ^ { \star } ( d , m ) = ( m + 1 ) d .
$$

We further determine two more values of the weighted selection profile $F _ { \mathrm { w e i g h t e d } } ( d , m , n )$ : at the near-threshold budget, $\begin{array} { r } { F _ { \mathrm { w e i g h t e d } } ( d , m , ( m + 1 ) d - 1 ) = 1 + \frac { 1 } { d m ^ { 2 } } } \end{array}$ , and at the spanning budget, $F _ { \mathrm { w e i g h t e d } } ( d , m , d ) = d + 1$ for every m, while $F _ { \mathrm { w e i g h t e d } } ( d , \overrightharpoon { m , } n ) = \infty$ for $n < d .$ . For the smallest open intermediate cell $( d , m ) = ( 2 , 2 )$ we prove $F _ { \mathrm { w e i g h t e d } } ( 2 , 2 , 3 ) \in [ 1 3 / 8 , 1 5 / 8 ]$ and $F _ { \mathrm { w e i g h t e d } } ( 2 , 2 , 4 ) \in [ 5 / 4 , 3 / 2 ]$ , reduce the conjectured exact values $1 3 / 8$ and $5 / 4$ to a finite moment problem on the circle with at most seven atoms, and establish strong structural evidence for the conjecture. The upper-bound techniques (a fixed-basis conic compression lemma, a determinant–facet rigidity theorem for maximal certificates, and sharp sparsifi cation lemmas for zero-mean weighted point systems) are of independent interest. As a byproduct we correct an erroneous claim circulating in a recent unrefereed preprint, exhibiting an explicit dataset with $m = 2$ on which no weighted selection of 2d points recovers the optimal loss. All results are new only for $m \geq 2 ;$ the scalar case $m = 1$ is due to Hanneke et al.

## 1 Introduction

How well can a fixed, natural learning rule perform when it is trained on only n examples selected from a larger dataset? Hanneke, Moran, Shlimovich and Yehudayof posed this dataselection question for basic regression tasks in a COLT 2025 open-problem note [HMSY25b] and companion paper [HMSY25a]. For scalar linear regression they obtained a complete taxonomy: with weighted selection and the minimum-norm empirical risk minimizer, the worst-case ratio between the loss of the selected model and the optimal full-data loss equals 1 for $n \geq 2 d .$ , equals $d + 1$ at $n = d ,$ , and is infinite for $n < d$ [HMSY25b, Theorem 1]. Their note concludes with a unified vector-valued formulation (predictors $W \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { m } }$ , square loss, minimum Frobeniusnorm ERM) and asks, as Question 4:

Given d, m, n, what is the value of $F _ { \mathrm { w e i g h t e d } } ( d , m , n ) \ ?$ In particular, what is the minimal $n = n ( d , m )$ such that this ratio equals 1?

This paper resolves the threshold question completely and determines several further values of the profile $F _ { \mathrm { w e i g h t e d } } ( d , m , \cdot )$

## 1.1 Main results

Throughout, $D = \{ z _ { i } = ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N } \subseteq \mathbb { R } ^ { d } \times \mathbb { R } ^ { m }$ is a finite multiset, $\ell _ { z } ( W ) = \| W x - y \| _ { 2 } ^ { 2 }$ for $\begin{array} { r } { W \in \mathbb { R } ^ { m \times d } , L _ { D } ( W ) = \frac { 1 } { N } \sum _ { i } \ell _ { z _ { i } } \bar { ( W ) } , L _ { D } ^ { \star } = \operatorname* { m i n } _ { W } L _ { D } ( W ) } \end{array}$ , and A is the ERM returning the empirical risk minimizer of minimal Frobenius norm. Weighted selection of budget n chooses $z _ { 1 } , \dotsc , z _ { n } \in D$ (repetitions allowed) and a convex combination $F \in \mathrm { c o n v } ( \ell _ { z _ { 1 } } , \ldots , \ell _ { z _ { n } } )$ , and sufers loss $L _ { D } ( A ( F ) )$ on the full dataset; $F _ { \mathrm { w e i g h t e d } } ( d , m , n )$ is the supremum over datasets of the ratio inf<sub>F</sub> $L _ { D } ( A ( F ) ) / L _ { D } ^ { \star }$ (Section 2 records the degenerate-ratio conventions). Let $n ^ { \star } ( d , m ) =$ min $\{ n : F _ { \mathrm { w e i g h t e d } } ( d , m , n ) = 1 \}$

Theorem 1.1 (Exact threshold; Theorem 3.7). For all $d , m \geq 2$

$$
n ^ { \star } ( d , m ) = ( m + 1 ) d .
$$

Moreover, for every dataset with feature rank r, some $( m + 1 ) r$ weighted points already recover a full-data optimum exactly, and there is an explicit integer dataset on $( m + 1 ) d$ points for which every selection of $( m + 1 ) d - 1$ weighted points has ratio exactly $\begin{array} { r } { 1 + \frac { m + 1 } { 2 m d ( m ^ { 2 } + m - 1 ) } > 1 } \end{array}$

Theorem 1.2 (Near-threshold value; Theorem 5.4). For all $d \ge 1 , m \ge 1$

$$
F _ { \mathrm { w e i g h t e d } } \left( d , m , ( m + 1 ) d - 1 \right) = 1 + \frac { 1 } { d m ^ { 2 } } .
$$

Theorem 1.3 (Low budgets; Theorem 4.1). For all $d , m \geq 1 \colon F _ { \mathrm { w e i g h t e d } } ( d , m , n ) = \infty f o r n < d ,$ and

$$
F _ { \mathrm { w e i g h t e d } } ( d , m , d ) = d + 1 ,
$$

independently of m.

Theorem 1.4 (Smallest intermediate cell; Theorem 7.7). $F _ { \mathrm { w e i g h t e d } } ( 2 , 2 , 3 ) \in \left[ \frac { 1 3 } { 8 } , \frac { 1 5 } { 8 } \right]$ and $F _ { \mathrm { w e i g h t e d } } ( 2 , 2 , 4 ) \in$ $\left[ \frac { 5 } { 4 } , \frac { 3 } { 2 } \right]$

We conjecture that the lower ends are the truth, $\begin{array} { r } { F _ { \mathrm { w e i g h t e d } } ( 2 , 2 , 3 ) = \frac { 1 3 } { 8 } } \end{array}$ and $F _ { \mathrm { w e i g h t e d } } ( 2 , 2 , 4 ) =$ $\frac { 5 } { 4 }$ (Conjecture 7.8), and we reduce this conjecture to a concrete finite question: a moment problem for at most seven atoms on the unit circle (Section 7). The evidence assembled there includes a complete solution of the two-direction class with a rigidity description of its equality systems, a proof that this equality fiber is a local maximum in a stratified sense, two exact obstructions showing which proof strategies cannot work, and exact extremal records for small systems.

Consistency checks: at $m = 1$ Theorems 1.1 and 1.2 specialize to $n ^ { \star } ( d , 1 ) = 2 d$ and $F _ { w } ( d , 2 d -$ $1 ) = 1 + { \frac { 1 } { d } }$ , matching [HMSY25b, Theorem 1] and the value announced there; at $d = 1 , x \equiv 1$ the problem contains weighted mean estimation in $\mathbb { R } ^ { m }$ , and $n ^ { \star } ( 1 , m ) = m + 1$ is the Carath´eodory bound.

Scope of novelty. For m = 1 all three regimes of Theorem 1.3 and the threshold $n ^ { \star } ( d , 1 ) = 2 d$ were established by Hanneke et al. [HMSY25b, HMSY25a] (their Theorem 8 and Example 2 give the exact scalar taxonomy), and the value $1 + { \textstyle { \frac { 1 } { d } } }$ at $n = 2 d - 1$ , announced in [HMSY25b], is proved—together with further exact values in the scalar intermediate regime—in the companion paper [Zha26]. All results of this paper are claimed as new only for $m \geq 2 ;$ our proofs happen to cover m = 1 uniformly, which we use purely as a consistency check.

## 1.2 Techniques

The threshold upper bound comes from a fixed-basis conic compression lemma (Lemma 3.1): after pinning a feature basis B of r points, the negated sum of their residual dyads is compressed inside the cone of the remaining dyads by conic Carath´eodory, at cost mr; together with the basis this yields an exact certificate on (m + 1)r points. Notably, this argument is simpler than the Steinitz-based route used for the scalar upper bound in [HMSY25a], and gives a data-dependent bound.

The near-threshold upper bound requires understanding datasets whose minimal exact certificates have the maximal size $( m + 1 ) d$ . We prove a determinant–facet rigidity theorem (Theorem 5.1): every facet of the normalized certificate polytope contributes a distinct linear factor to the degree-d polynomial $\begin{array} { r } { u \mapsto \operatorname* { d e t } ( \sum _ { i } u _ { i } x _ { i } x _ { i } ^ { \top } ) } \end{array}$ , forcing the polytope to be a simplex and the dataset to decompose into d disjoint (m + 1)-point circuits supported on d independent feature lines. A sharp k-point mean lemma for zero-mean weighted point systems (Lemma 5.3 for $k = m ;$ Theorem 6.1 in general) then finishes the assembly.

For the intermediate regime at (2, 2) we develop a complex-variable dictionary that identifies whitened instances with moment systems $( p _ { i } , t _ { i } , z _ { i } )$ on the unit circle and the selection cost with an explicit convex combination of two-point interpolation coeficients (Lemma 7.2), an identity that absorbs the ill-conditioning factor $( 1 - | \mu | ^ { 2 } ) ^ { - 1 }$ entirely.

## 1.3 Related and concurrent work

Published work. The problem and all scalar results are from [HMSY25b, HMSY25a]. The budget-d upper bound uses volume sampling and the exact expectation formulas of Derezi´nski and Warmuth [DW17]; we are careful to use their Theorem 5 in its correct form (equality under general position, inequality in general; see Section 4). The general k-point mean lemma relies on the full-dimensional Grace–Danielsson inequality (Egan’s conjecture), proved by Drozdov [Dro25]; none of our main theorems depend on that lemma beyond the elementary case $k = m$ for which we give a short self-contained proof (Lemma 5.3).

Concurrent unrefereed preprints. This open-problem family is being attacked by several automated pipelines, and a number of unrefereed, apparently machine-generated preprints on other questions of [HMSY25b] appeared recently: a preprint resolving Question 3 (unweighted selection, general m) $[ \mathrm { P ^ { + } 2 6 } ]$ ; two short manuscripts on Question 1 (mean estimation), one of which determines the budget-2 column of the mean-estimation table [Ano26b, Ano26a]; and a manuscript on the scalar weighted regime of Question 2 [Dew26]. None of these treats the weighted vector-valued Question 4, whose threshold and profile values are the subject of this paper. Since [Ano26b] advertises convex-geometric lemmas about few-atom distributions obtained by a variance-minimizing Carath´eodory reduction, we note the demarcation explicitly: our sparsification results (Lemma 5.3, Theorem 6.1) were obtained independently, are stated for zero-mean weighted systems with the sharp constant $\textstyle { \frac { M + 1 - k } { k M } }$ , and are used here as components of the $F _ { \mathrm { w e i g h t e d } }$ analysis; we make no priority claim on auxiliary convex-geometry statements that may overlap, and our headline results — the complete (m + 1)d characterization and the profile values — are not touched by any of the works above.

Remark 1.5 (Correction of a circulating claim). Remark 1.2 of $[ \mathrm { P ^ { + } 2 6 } ]$ asserts, in the notation $F ^ { w } ( d , m , n )$ , that the weighted scalar trichotomy of [HMSY25b, Theorem 1] — in particular, ratio 1 for all $n \geq 2 d -$ holds for general m, citing [HMSY25b] as the source. This is a misattribution: [HMSY25b] proves the trichotomy only for $m = 1$ , and the assertion is false for every $m \geq 2$ . By Theorem 1.1, exact recovery requires $( m + 1 ) d > 2 d$ points, and quantitatively $\begin{array} { r } { F _ { \mathrm { w e i g h t e d } } ( d , m , 2 d ) \geq F _ { \mathrm { w e i g h t e d } } ( d , m , ( m + 1 ) d - 1 ) = 1 + \frac { 1 } { d m ^ { 2 } } > 1 } \end{array}$ by monotonicity. We make the failure concrete in Proposition 3.9: on an explicit 6-point integer dataset with $d = m = 2$ , every weighted selection of $2 d = 4$ points has full-data loss at least $\bar { \frac { 2 3 } { 2 0 } } \cdot L _ { D } ^ { \star }$ , with equality attained.

## 1.4 Organization

Section 2 fixes definitions and records the basic structural lemmas. Section 3 proves the threshold theorem and the correction of Remark 1.5. Section 4 handles budgets $n \leq d .$ Section 5 proves the near-threshold value, via the rigidity theorem. Section 6 proves the general k-point mean lemma. Section 7 develops the (2, 2) intermediate regime: the interval theorem, the reduction of the conjectured exact values to a seven-atom moment problem, and the evidence. Section 8 discusses the general program and open problems.

## 2 Preliminaries

## 2.1 The model

A dataset is a finite multiset $D = \{ z _ { i } = ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N } \subseteq \mathbb { R } ^ { d } \times \mathbb { R } ^ { m }$ with $N \geq 1$ . A linear predictor is a matrix $W \in \mathbb { R } ^ { m \times d }$ , with loss $\ell _ { z } ( W ) = \| W x - y \| _ { 2 } ^ { 2 }$ and average loss $\begin{array} { r } { L _ { D } ( W ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \ell _ { z _ { i } } ( W ) } \end{array}$ write $L _ { D } ^ { \star } = \operatorname* { m i n } _ { W } L _ { D } ( W )$ . The learning rule A maps a nonnegatively weighted objective $F =$ $\sum _ { i } c _ { i } \ell _ { z _ { i } }$ (with $c _ { i } \geq 0 , \sum _ { i } c _ { i } = 1 )$ to the minimizer of $F$ of minimal Frobenius norm. Weighted selection with budget n is the value

$$
L _ { D } ^ { \star } ( n ; \mathrm { w e i g h t e d } ) = \operatorname* { i n f } _ { z _ { j _ { 1 } } , . . . , z _ { j _ { n } } \in D } L _ { D } \big ( A ( F ) \big ) ,
$$

and

$$
F _ { \mathrm { w e i g h t e d } } ( d , m , n ) = \operatorname* { s u p } _ { D \subseteq \mathbb { R } ^ { d } \times \mathbb { R } ^ { m } } \frac { L _ { D } ^ { \star } ( n ; \mathrm { w e i g h t e d } ) } { L _ { D } ^ { \star } } , \qquad n ^ { \star } ( d , m ) = \operatorname* { m i n } \{ n : F _ { \mathrm { w e i g h t e d } } ( d , m , n ) = 1 \} .
$$

Following the convention of [HMSY25a, Theorem 1] we set the ratio to 1 when numerator and denominator are both 0, and to ∞ when only the denominator is 0. We write the inner optimization as an infimum (as in [HMSY25a]); all our upper-bound certificates are attained, and our lower bounds bound the infimum, so nothing depends on attainment.

Repetitions among the $z _ { j _ { 1 } } , \dotsc , z _ { j _ { n } }$ are allowed. The following normalization is immediate and used silently.

Lemma 2.1 (Selection semantics). Selections of budget n are in value-preserving correspondence with weight vectors $c \in \mathbb { R } _ { \geq 0 } ^ { N } , \sum _ { i } { \dot { c } } _ { i } = 1$ , | supp c| ≤ n: repeated picks merge weights, zero-weight picks can be discarded, and any c with $| \operatorname { s u p p } c | \leq n$ is realizable with exactly n slots by repeating a chosen point and splitting its weight.

## 2.2 Row decomposition and minimum-norm geometry

Lemma 2.2 (Row decomposition). Write $w _ { i } ^ { \top }$ for the $j - t h$ row of W and $y _ { i j }$ for the $j - t h$ entry of y . Then $\begin{array} { r } { F _ { c } ( W ) = \sum _ { j = 1 } ^ { m } \sum _ { i } c _ { i } ( \langle w _ { j } , x _ { i } \rangle - \check { y } _ { i j } ) ^ { 2 } } \end{array}$ and $\begin{array} { r } { \| W \| _ { F } ^ { 2 } = \sum _ { j } \| w _ { j } \| ^ { 2 } } \end{array}$ . Hence the minimizer set of $F _ { c }$ is the Cartesian product over rows of scalar weighted least-squares solution sets, and

$A ( F _ { c } )$ consists of the minimal-ℓ<sub>2</sub>-norm scalar solutions row by row — with the single shared weight vector c.

Proof. Both the objective and the squared Frobenius norm are additive across rows, and a product set is minimized in norm coordinate-wise. □

Let $T = \operatorname { s p a n } \{ x _ { 1 } , \dots , x _ { N } \} , r = \dim T$ , and let $P _ { T }$ denote the orthogonal projection onto T. Let $W ^ { \circ }$ be the minimal-Frobenius-norm full-data minimizer, and set $\rho _ { i } = W ^ { \circ } x _ { i } - y _ { i }$ (residuals) and $M _ { i } = \rho _ { i } x _ { i } ^ { \top } \in \mathbb { R } ^ { m \times d }$ (residual dyads).

Lemma 2.3 (Full-data geometry). $W ^ { \circ } = W ^ { \circ } P _ { T }$ ; the first-order condition $\begin{array} { r } { \sum _ { i = 1 } ^ { N } M _ { i } = 0 } \end{array}$ holds; and for every $H \in \mathbb { R } ^ { m \times d }$

$$
L _ { D } ( W ^ { \circ } + H ) = L _ { D } ^ { \star } + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \lVert H x _ { i } \rVert ^ { 2 } .\tag{1}
$$

Consequently the full-data minimizer set is $\{ W ^ { \circ } + H : H P _ { T } = 0 \} ; \ i f T = \mathbb { R } ^ { d }$ the minimizer is unique.

Proof. Predictions depend on W only through $W P _ { T }$ , and $\| W \| _ { F } ^ { 2 } = \| W P _ { T } \| _ { F } ^ { 2 } + \| W P _ { T ^ { \perp } } \| _ { F } ^ { 2 }$ , so minimality of the norm forces $W ^ { \circ } P _ { T ^ { \perp } } = 0$ . The condition $\textstyle \sum _ { i } M _ { i } = 0$ is the vanishing gradient of the smooth convex $L _ { D }$ at $W ^ { \circ }$ . Expanding $L _ { D } ( W ^ { \circ } + H )$ , the cross term is $\begin{array} { r } { \frac { 2 } { N } \langle H , \sum _ { i } M _ { i } \rangle _ { F } = 0 . } \end{array}$ giving (1); equality in (1) holds if $H x _ { i } = 0$ for all i, i.e. H $P _ { T } = 0$ □

Lemma 2.4 (Exact certificate). Let $c \in \mathbb { R } _ { > 0 } ^ { N }$ have support S with $c _ { i } > 0$ on S. If

$$
\sum _ { i \in S } c _ { i } M _ { i } = 0 \qquad a n d \qquad \operatorname { s p a n } \{ x _ { i } : i \in S \} = T ,\tag{2}
$$

then $A ( F _ { c } ) = W ^ { \circ }$ ; in particular the selection realizes the full-data optimal loss, and the ratio is 1 (also when $L _ { D } ^ { \star } = 0$ , by the $0 / 0$ convention).

Proof. Using (2), for any H, $\begin{array} { r } { F _ { c } ( W ^ { \circ } + H ) - F _ { c } ( W ^ { \circ } ) = 2 \langle H , \sum _ { i \in S } c _ { i } M _ { i } \rangle _ { F } + \sum _ { i \in S } c _ { i } \| H x _ { i } \| ^ { 2 } = } \end{array}$ $\textstyle \sum _ { i \in S } c _ { i } \| H x _ { i } \| ^ { 2 }$ , which vanishes if $H x _ { i } = 0$ on S, if $H P _ { T } = 0$ (the support spans T). So the weighted minimizer set is $W ^ { \circ } + \{ H : H P _ { T } = 0 \}$ , the same set as the full-data minimizer set. Since $W ^ { \circ } = W ^ { \circ } P _ { T }$ and $H = H P _ { T } \mathrm { ~ . ~ }$ <sub>⊥</sub> are Frobenius-orthogonal, the unique minimal-norm element is $W ^ { \circ }$ □

We call a weight vector satisfying (2) a spanning zero certificate. Define

$$
\tau ( D ) \ = \ \operatorname * { m i n } \Bigl \{ | \mathrm { s u p p } c | \colon c \ \mathrm { i s \ a \ s p a n n i n g \ z e r o \ c e r t i f i c a t e } \Bigr \} .\tag{3}
$$

Lemma 2.5 (Scalar embedding). If all m columns of the labels are equal, $y _ { i } = t _ { i } \mathbf { 1 } _ { m }$ , then each row of the problem is the same scalar problem and $L _ { \tilde { D } } ( \cdot ) = m L _ { D _ { 0 } } ( \cdot )$ termwise. Consequently $F _ { \mathrm { w e i g h t e d } } ( d , m , n ) \geq F _ { w } ( d , n )$ for all $n ,$ where $F _ { w } ( d , \bar { n } )$ denotes the scalar weighted profile of [HMSY25b].

Proof. By Lemma 2.2 every row runs the identical scalar min-norm ERM with the shared weights, so numerator and denominator both scale by m. □

## 3 The exact threshold $( m + 1 ) d$

## 3.1 Upper bound: fixed-basis conic compression

Lemma 3.1 (Fixed-basis conic compression). Let $a _ { 1 } , \ldots , a _ { N }$ be elements of a finite-dimensional real vector space with $\begin{array} { r } { \sum _ { i = 1 } ^ { N } a _ { i } = 0 } \end{array}$ , and let $B \subseteq [ N ]$ . Put q<sub>B</sub> = dim span $\{ a _ { i } : i \notin B \}$ . Then there exist coeficients $c _ { i } \geq 0$ with $c _ { b } > 0$ for all $\textstyle b \in B , \sum _ { i } c _ { i } a _ { i } = 0$ , and $| \operatorname { s u p p } c | \leq | B | + q _ { B }$

Proof. From the global zero sum, $\begin{array} { r } { - \sum _ { b \in B } a _ { b } \ = \ \sum _ { i \notin B } a _ { i } } \end{array}$ , which is a nonnegative combination of $\{ a _ { i } \} _ { i \notin B }$ and hence lies in cone $\{ a _ { i } : i \notin B \}$ . By conic Carath´eodory (see e.g. [Ber09, Prop. $\mathrm { 1 . 2 . 1 ( \dot { a } ) ] ) }$ , every element of this cone is a positive combination of a linearly independent subfamily, $\begin{array} { r } { \mathrm { s o } - \sum _ { b \in B } a _ { b } = \sum _ { i \in C } \alpha _ { i } a _ { i } } \end{array}$ with $\alpha _ { i } > 0 , C \subseteq [ N ] \backslash B$ and $| C | \le q _ { B }$ (if the target is 0, take $C = \varnothing )$ . Set $c _ { b } = 1$ on $B , c _ { i } = \alpha _ { i }$ on C, and 0 elsewhere. □

Theorem 3.2 (Data-dependent upper bound). Every dataset with feature rank $r \geq 1$ admits a spanning zero certificate of support at most $( m + 1 ) r ;$ hence $\tau ( D ) \leq ( m + 1 ) r$ and $L _ { D } ^ { \star } \big ( ( m +$ $1 ) d ; \mathrm { w e i g h t e d } ) = L _ { D } ^ { \star }$ for every dataset. $I f r = 0$ , a single point sufices.

Proof. Pick $B \subseteq [ N ]$ with $| B | = r$ such that $\{ { x } _ { b } \} _ { b \in B }$ is a basis of T (possible: the features lie in and span T). Apply Lemma 3.1 to the residual dyads $a _ { i } = M _ { i }$ , whose global sum vanishes (Lemma 2.3). Every $M _ { i } = \rho _ { i } x _ { i } ^ { \top }$ has all rows proportional to $x _ { i } ^ { \top } \in T$ , so span $\{ M _ { i } : i \notin B \} \subseteq$ $\mathbb { R } ^ { m } \otimes T .$ , whence $q _ { B } \leq m r$ . The resulting c is strictly positive on a set $S \supseteq B$ of size at most $r + m r = ( m + 1 ) r .$ , satisfies $\textstyle \sum _ { i \in S } c _ { i } M _ { i } = 0$ , and span $\{ x _ { i } : i \in S \} \supseteq \operatorname { s p a n } \{ x _ { b } : b \in B \} = T ;$ since all features lie in T, the span equals T. Normalizing c to sum 1 (its support is nonempty) and invoking Lemma 2.4 gives $A ( F _ { c } ) = W ^ { \circ }$ , so the selected model attains the optimal full-data loss; by Lemma 2.1 the support fits in $( m + 1 ) d \geq ( m + 1 ) r$ slots. If $r = 0$ then all $x _ { i } = 0 \quad$ : every W has the same loss on every objective, and A returns $W = 0$ in both cases; one point sufices.

## 3.2 Lower bound: the axial simplex instance

Fix $d , m \geq 1$ and let $u = \mathbf { 1 } _ { m }$

$$
v _ { j } = e _ { j } ~ ( 1 \leq j \leq m ) , \qquad v _ { m + 1 } = - u ,
$$

so $\begin{array} { r } { \sum _ { j = 1 } ^ { m + 1 } v _ { j } = 0 } \end{array}$ , and the unique linear dependence among $v _ { 1 } , \ldots , v _ { m + 1 }$ has all coeficients equal; in particular no proper subset of $\{ v _ { j } \}$ admits a nonzero nonnegative zero-sum, equivalently $0 \not \in$ conv of any proper subset.

Definition 3.3 (Integer axial instance). $D _ { d , m }$ consists of the $N = ( m + 1 ) d$ points $z _ { i j } =$ $( e _ { i } , u - v _ { j } )$ for $i \in [ d ] , j \in [ m + 1 ]$ ; explicitly $y _ { i j } = u - e _ { j }$ for $j \leq m$ and $y _ { i , m + 1 } = 2 u$

Lemma 3.4. For $D _ { d , m } \colon$ the unique full-data minimizer is $W ^ { \star } = u \mathbf { 1 } _ { d } ^ { \top }$ , with residual $v _ { j }$ at $z _ { i j }$ and $\begin{array} { r } { L _ { D } ^ { \star } = \frac { 2 m } { m + 1 } } \end{array}$ . Moreover, for any weight vector c with per-axis totals $t _ { i } = \textstyle \sum _ { j } c _ { i j }$ , the returned model $\widehat { W } = A ( F _ { c } )$ satisfies: column i of $\widehat { W }$ equals the conditional weighted label mean $u - a _ { i }$ with $\begin{array} { r } { a _ { i } = \sum _ { j } \frac { c _ { i j } } { t _ { i } } v _ { j } } \end{array}$ when $t _ { i } > 0$ , and equals 0 when $t _ { i } = 0 ,$ and, writing $\delta _ { i } = \widehat { W } e _ { i } - u$

$$
L _ { D } ( \widehat { W } ) = L _ { D } ^ { \star } + \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \lVert \delta _ { i } \rVert ^ { 2 } .\tag{4}
$$

Proof. The full Gram matrix is $\begin{array} { r } { \sum _ { i j } e _ { i } e _ { i } ^ { \top } = ( m + 1 ) I \succ 0 } \end{array}$ and $\begin{array} { r } { \sum _ { i j } v _ { j } e _ { i } ^ { \top } = 0 } \end{array}$ , so $W ^ { \star }$ is the unique minimizer; $\begin{array} { r } { L _ { D } ^ { \star } = \frac { d \sum _ { j } \| v _ { j } \| ^ { 2 } } { N } = \frac { 2 m d } { ( m + 1 ) d } } \end{array}$ . Since the features are standard basis vectors, the weighted normal equations decouple across columns; a positive-weight column returns the conditional mean, and a zero-weight column is unconstrained, hence zeroed by the Frobenius rule (column-wise decomposition of the norm). For (4), expand $\| \delta _ { i } + v _ { j } \| ^ { 2 }$ and use $\begin{array} { r } { \sum _ { j } v _ { j } = 0 { : } } \end{array}$ $\begin{array} { r } { \sum _ { j } \| v _ { j } + \delta _ { i } \| ^ { 2 } = 2 m + ( m + 1 ) \| \delta _ { i } \| ^ { 2 } } \end{array}$ □

Lemma 3.5 (Facet distances). Let $Q _ { m } = m ^ { 2 } + m - 1$ . The squared distance from the origin to the convex hull of any proper subset of $\{ v _ { 1 } , \ldots , v _ { m + 1 } \}$ is at least $1 / Q _ { m }$ ; the minimum is attained on the facets omitting some $e _ { k }$ , while the facet omitting $v _ { m + 1 }$ has squared distance $1 / m$

Proof. Every proper subset is contained in a facet. For the facet conv $\{ e _ { 1 } , \ldots , e _ { m } \}$ the nearest point is $u / m .$ , of squared norm $1 / m$ . For the facet omitting $\textstyle e _ { k } .$ , let $n _ { k }$ hav $\beta - m$ in coordinate k and 1 elsewhere; the facet lies on the hyperplane $n _ { k } ^ { \top } x = 1$ and $\| n _ { k } \| ^ { 2 } = Q _ { m } ,$ so the distance to the hyperplane is $1 / \sqrt { Q _ { m } } ;$ the foot $n _ { k } / Q _ { m }$ lies in the facet since $\begin{array} { r } { n _ { k } / Q _ { m } = \frac { m } { Q _ { m } } ( - u ) + \sum _ { j \neq k } \frac { m + 1 } { Q _ { m } } e _ { j } } \end{array}$ is a convex combination. Finally $Q _ { m } \geq m$ with equality only at $m = 1$ (where both distances equal $1 = 1 / Q _ { 1 } )$ □

Theorem 3.6 (Exact deficit of the axial instance). For every $d , m \geq 1$

$$
\frac { L _ { D _ { d , m } } ^ { \star } \big ( ( m + 1 ) d - 1 ; \mathrm { w e i g h t e d } \big ) } { L _ { D _ { d , m } } ^ { \star } } = 1 + \frac { m + 1 } { 2 m d ( m ^ { 2 } + m - 1 ) } > 1 .
$$

Proof. A selection of support at most $N - 1$ either misses some axis entirely $( \lVert \delta _ { i } \rVert ^ { 2 } = \lVert u \rVert ^ { 2 } =$ m $\geq 1 / Q _ { m } )$ or uses a proper subset of the $m + 1$ points of some axis, in which case $\delta _ { i } = - a _ { i }$ lies in minus the convex hull of a proper subset of $\{ v _ { j } \}$ and $\lVert \delta _ { i } \rVert ^ { 2 } \geq 1 / Q _ { m }$ by Lemma 3.5. With (4), every such selection has $\begin{array} { r } { L _ { D } \geq L _ { D } ^ { \star } + \frac { 1 } { d Q _ { m } } } \end{array}$ . Conversely, omit $z _ { 1 , 1 }$ , give axis 1 the conditional weights of the foot $n _ { 1 } / Q _ { m }$ (namely $\frac { m } { Q _ { m } }$ on $v _ { m + 1 }$ and $\textstyle { \frac { m + 1 } { Q _ { m } } }$ on each $v _ { j } , 2 \leq j \leq m$ , scaled by $\begin{array} { r } { t _ { 1 } = \frac { 1 } { d } ) } \end{array}$ and every other axis uniform weights: the support is exactly $N - 1$ and $\begin{array} { r } { L _ { D } = L _ { D } ^ { \star } + \frac { 1 } { d Q _ { m } } } \end{array}$ Dividing by $\begin{array} { r } { L _ { D } ^ { \star } = \frac { 2 m } { m + 1 } } \end{array}$ gives the claim. □

Theorem 3.7 (Threshold). $n ^ { \star } ( d , m ) = ( m + 1 ) d$ for all d, $m \geq 1$ , and $F _ { \mathrm { w e i g h t e d } } ( d , m , n ) = 1$ for all $n \geq ( m + 1 ) d$

Proof. Combine Theorem 3.2 (ratio 1 at budget $( m + 1 ) d ,$ , every dataset) with Theorem 3.6 (some dataset has ratio $> 1$ at budget $( m + 1 ) d - 1 )$ □

Corollary 3.8. $n ^ { \star } ( 2 , 2 ) = 6$ . On the instance $D _ { 2 , 2 }$ (six integer points, labels (0, 1), (1, 0), (2, 2) over each $o f e _ { 1 } , e _ { 2 } )$ the optimal five-point ratio is exactly $\frac { 4 3 } { 4 0 }$

We emphasize that $4 3 / 4 0$ is the exact optimum of this instance; the worst case over all datasets at budget 5 is $F _ { \mathrm { w e i g h t e d } } ( 2 , 2 , 5 ) = 1 + { \textstyle { \frac { 1 } { 8 } } } = { \textstyle { \frac { 9 } { 8 } } }$ by Theorem 5.4.

## 3.3 Failure at $n = 2 d$ for $m \geq 2$

As announced in Remark 1.5, the scalar rule $^ { 6 6 } n \geq 2 d$ sufices” does not survive vector outputs.

Proposition 3.9. For all $d \geq 1 , m \geq 2 , \ F _ { \mathrm { w e i g h t e d } } ( d , m , 2 d ) \ \geq \ F _ { \mathrm { w e i g h t e d } } ( d , m , ( m + 1 ) d - 1 ) \ = \qquad $ $\textstyle 1 + { \frac { 1 } { d m ^ { 2 } } } \ > \ 1$ , since $2 d \leq ( m + 1 ) d - 1$ and $F _ { \mathrm { w e i g h t e d } }$ is nonincreasing in n. Concretely, for $d = m = 2$ the instance $D _ { 2 , 2 }$ of Corollary 3.8 satisfies: every weighted selection of at most $2 d = 4$ points has

$$
L _ { D } ( A ( F ) ) \geq { \frac { 2 3 } { 2 0 } } \cdot L _ { D } ^ { \star } ,
$$

with equality attained (allocate two points per axis, at conditional weights realizing the nearest facet points).

Proof. The monotonicity statement is immediate from the definitions. For $D _ { 2 , 2 } { \mathrm { , } }$ , by (4) and peraxis decoupling, a budget-4 selection allocates $( k _ { 1 } , k _ { 2 } )$ points to the two axes with $k _ { 1 } + k _ { 2 } \leq 4 ;$ the minimal per-axis costs are 0 (all three points), 1/5 (best two points, the facet $\{ e _ { 2 } , - u \}$ or $\{ e _ { 1 } , - u \}$ at squared distance $1 / Q _ { 2 } = 1 / 5 )$ , 1 (best single point), and 2 (empty axis). The minimum total cost over allocations is $2 \cdot \textstyle { \frac { 1 } { 5 } }$ at (2, 2), giving $\begin{array} { r } { L _ { D } = \mathrm { ~ \frac { 4 } { 3 } + \frac { 1 } { 2 } \cdot \frac { 2 } { 5 } = \frac { 2 3 } { 1 5 } ~ } } \end{array}$ and ratio ${ \frac { 2 3 / 1 5 } { 4 / 3 } } = { \frac { 2 3 } { 2 0 } }$ . (We verified this enumeration by exact symbolic computation as well.) □

## 4 Budgets $n \leq d$

Theorem 4.1. For all $d , m \geq 1 \colon F _ { \mathrm { w e i g h t e d } } ( d , m , n ) = \infty f o r$ every $n < d _ { \colon }$ , and $F _ { \mathrm { w e i g h t e d } } ( d , m , d ) =$ d + 1.

The lower bounds are inherited from the scalar case through Lemma 2.5: by [HMSY25b, Theorem 1], $F _ { w } ( d , n ) = \infty$ for $n < d$ and $F _ { w } ( d , d ) = d + 1$ , so $F _ { \mathrm { w e i g h t e d } } ( d , m , n ) \geq F _ { w } ( d , n )$ The content of this section is the upper bound $F _ { \mathrm { w e i g h t e d } } ( d , m , d ) \leq d + 1$ uniformly in m, which requires selecting one subset that serves all m output rows simultaneously.

Volume sampling. Let $X ~ \in ~ \mathbb { R } ^ { d \times N }$ have full row rank and let $\boldsymbol { y } ~ \in ~ \mathbb { R } ^ { N }$ . Size-d volume sampling draws a subset S of d column indices with probability proportional to det $( X _ { S } ) ^ { 2 }$ , and $w ( S )$ denotes the least-squares solution of the subproblem $( X _ { S } , y _ { S } )$ (the interpolant when $X _ { S }$ is invertible). Derezi´nski and Warmuth [DW17] proved: (i) unbiasedness $\mathbb { E } [ w ( S ) ] = w ^ { \star }$ for every label vector, requiring only full row rank (their Theorem 3 and Proposition 7); and (ii) for the total square loss L on the full data,

$$
\begin{array} { r } { \mathbb { E } \big [ L ( w ( S ) ) \big ] \ \leq \ ( d + 1 ) L ( w ^ { \star } ) , } \end{array}\tag{5}
$$

with equality when X is in general position (every d-column submatrix nonsingular) — their Theorem 5. We only use the inequality (5), which holds without general position; note that pointwise equality genuinely fails for degenerate X $( { \mathrm { e . g . ~ } } x _ { 1 } = x _ { 2 } = e _ { 1 } , x _ { 3 } = x _ { 4 } = e _ { 2 } , y =$ $( 1 , - 1 , 1 , - 1 )$ has $\mathbb { E } [ L ( w ( S ) ) ] = 2 L ( w ^ { \star } ) )$ ). Dividing by N converts (5) to average losses.

Proof of Theorem 4.1. Only $F _ { \mathrm { w e i g h t e d } } ( d , m , d ) \leq d + 1$ remains. Fix a dataset; let $r =$ rank of the features. If $r = 0$ the ratio is 1 (both rules return $W = 0 )$ . If $L _ { D } ^ { \star } = 0$ , choose r points whose features form a basis of T with positive weights: the weighted problem has minimum $0 ,$ its minimizer set is $W ^ { \circ } + \{ H : H P _ { T } = 0 \}$ , and the Frobenius rule returns $W ^ { \circ }$ ; the ratio is 1 by the $0 / 0$ convention.

Otherwise pick an orthonormal basis $U \in \mathbb { R } ^ { d \times r }$ of T and write $x _ { i } = U \xi _ { i } \mathrm { ~ w i t h } \Xi = [ \xi _ { 1 } \cdot \cdot \cdot \xi _ { N } ] \in$ $\mathbb { R } ^ { r \times N }$ of full row rank. Run size-r volume sampling on $\Xi ;$ crucially, the distribution over subsets depends only on the features. For the q-th output row, (5) (in dimension r) gives E $L _ { D } ^ { ( q ) } ( w _ { q , S } ) \leq$ $( r + 1 ) L _ { D } ^ { ( q ) , \star }$ . Summing over the m rows and using Lemma 2.2,

$$
\begin{array} { r } { \mathbb { E } \big [ L _ { D } ( W _ { S } ) \big ] ~ \le ~ ( r + 1 ) L _ { D } ^ { \star } ~ \le ~ ( d + 1 ) L _ { D } ^ { \star } , } \end{array}
$$

where $W _ { S } = \Theta _ { S } U ^ { \top }$ is the matrix whose rows are the per-row subproblem solutions in $\Xi _ { - }$ coordinates with the $T ^ { \perp }$ -component set to zero. Hence some subset $S \left( \mathrm { o f } r \leq d \mathrm { p o i n t s } \right)$ achieves $L _ { D } ( W _ { S } ) \ \leq \ ( d + 1 ) L _ { D } ^ { \star }$ . This $W _ { S }$ is realizable by a legal selection: the $r$ chosen features are linearly independent in $T .$ , so with any positive weights the weighted ERM set consists of all interpolants of the r selected points, and the Frobenius rule returns exactly $W _ { S }$ (the interpolant with vanishing $T ^ { \perp } { \mathrm { - c o m p o n e n t } } )$ . Padding to d slots by Lemma 2.1 completes the proof. □

## 5 The near-threshold value $1 + 1 / ( d m ^ { 2 } )$

Throughout this section fix a dataset with residual dyads $M _ { i } = \rho _ { i } x _ { i } ^ { \top }$ and recall the minimal certificate size $\tau ( D )$ from (3); Theorem 3.2 gives $\tau ( D ) \leq ( m + 1 ) r$ . The upper bound at budget $( m + 1 ) d - 1$ requires understanding the datasets with $\tau ( D ) = ( m + 1 ) d$ , which we call hard. The key is a rigidity theorem for maximal certificates.

## 5.1 Rigidity of maximal certificates

Theorem 5.1 (Determinant–facet rigidity). Let the features span $\mathbb { R } ^ { d }$ and suppose $\tau ( D ) = k : =$ $( m + 1 ) d .$ Let $c$ be a spanning zero certificate with $| \operatorname { s u p p } c | = k$ and support S. Then S splits uniquely (up to permutation) into d pairwise disjoint blocks $S = C _ { 1 } \sqcup \cdots \sqcup C _ { d }$ with $| C _ { j } | = m + 1$ 2 where each $C _ { j }$ supports a positive circuit of the dyads whose features span a one-dimensional line $U _ { j }$ , and $U _ { 1 } , \dots , U _ { d }$ are linearly independent.

Proof. Vectorize the dyads into ${ \cal A } = [ \mathrm { v e c } ( M _ { i } ) ] _ { i \in { \cal S } } \in \mathbb { R } ^ { m d \times k }$ and let $K =$ ker $A \cap \mathbb { R } _ { \geq 0 } ^ { k } , \ h \ =$ dim ker $A \geq k - m d = d .$ . Since $c > 0$ on S lies in ker A, a neighborhood of c in ker A stays nonnegative, so K spans ker A and the compact section $P = K \cap \{ \mathbf { 1 } ^ { \top } u = 1 \}$ is a polytope of dimension $h - 1$ whose relative interior consists exactly of the strictly positive points.

Consider $\begin{array} { r } { f ( u ) = \operatorname* { d e t } \bigl ( \sum _ { i \in S } u _ { i } x _ { i } x _ { i } ^ { \top } \bigr ) } \end{array}$ , a polynomial of degree at most d. On relint P all weights are positive and the features span $\mathbb { R } ^ { d } .$ , so $f > 0$ . At any boundary point some coordinate vanishes; if $f > 0$ there, the positively weighted features would span $\mathbb { R } ^ { d }$ and the point would be a spanning zero certificate of support $\le k - 1$ , contradicting $\tau = k$ . Hence $f \equiv 0$ on every facet of $P .$ . Parametrizing af $P$ by $\mathbb { R } ^ { h - 1 }$ , the restriction $g$ of $f$ is a nonzero polynomial of degree $\leq d$ vanishing on the relative interior of each facet, hence divisible by the (pairwise non-associated) afine linear forms of the distinct facet hyperplanes; therefore $P$ has at most d facets. Since a bounded $( h - 1 )$ )-dimensional polytope has at least h facets, we get $h \leq d .$ so $h = d , P$ has exactly d facets, and $P$ is a $( d - 1 )$ -simplex; K is a simplicial cone with extreme rays $v ^ { ( 1 ) } , \ldots , v ^ { ( d ) }$

Let $U _ { j } \ = \ \mathrm { s p a n } \{ x _ { i } \ : \ v _ { i } ^ { ( j ) } \ > \ 0 \}$ . As c is a strictly positive combination of all rays, the supports of the rays cover $S ,$ so $\textstyle \sum _ { j } U _ { j } = \mathbb { R } ^ { d }$ . Dropping ray $j$ gives a boundary point, where the features cannot span, so $\textstyle \sum _ { \ell \neq j } { \bar { U } } _ { \ell } \neq \mathbb { R } ^ { d }$ for each $j \colon$ the family $\{ U _ { j } \}$ is a minimal spanning family. Choose functionals $\phi _ { j }$ vanishing on $\textstyle \sum _ { \ell \neq j } U _ { \ell }$ and not on $U _ { j }$ , normalized against vectors $u _ { j } \in U _ { j }$ with $\phi _ { j } ( u _ { j } ) = 1 ;$ then $\left( \phi _ { j } \right)$ is a dual basis, and $U _ { j } \subseteq \cap _ { i \neq j }$ ker ϕ , a one-dimensional space, so dim $U _ { j } = 1$

Each extreme ray of the cone $\{ u \ge 0 : A u = 0 \}$ has inclusion-minimal support $C _ { j }$ , and dim ker $A _ { C _ { j } } = 1$ (otherwise a two-sided perturbation along a second kernel direction splits the $\mathrm { r a y } )$ . All dyads of $C _ { j }$ lie in $\mathbb { R } ^ { m } \otimes U _ { j }$ , a space of dimension $m ,$ so $| C _ { j } | =$ rank $A _ { C _ { i } } + 1 \le m + 1$ Finally $\begin{array} { r } { k = | S | \le \sum _ { j } | C _ { j } | \le d ( m + 1 ) = k } \end{array}$ forces every $| C _ { j } | = m + 1$ and pairwise disjoint supports. □

## 5.2 Hard datasets live on d lines

Theorem 5.2 (Line structure). Let the features span $\mathbb { R } ^ { d }$ and $\tau ( D ) = ( m + 1 ) d = k$ . Then:

1. every data point with $x _ { i } \neq 0$ belongs to some positive circuit of dyads supported on a single feature line and of size exactly $m + 1$ ;

2. there is no positive circuit of dyads whose features span a subspace of dimension $\geq 2$ , and no line circuit of size $\leq m ;$ in particular no point has $x _ { i } \neq 0$ and $\rho _ { i } = 0 _ { : }$

3. the nonzero features lie on exactly d linearly independent lines.

Consequently, any dataset whose nonzero features are not of this d-line form satisfies $\tau ( D ) \leq$ $( m + 1 ) d - 1$ and is recovered exactly within budget $( m + 1 ) d - 1$

Proof. (1) Extend $x _ { t }$ (t fixed, $x _ { t } \neq 0 )$ to a feature basis $B \ni t$ and apply Lemma 3.1: the resulting certificate has support at most k and, since $\tau = k$ , exactly $k ;$ Theorem 5.1 splits it into $( m + 1 ) { \mathrm { - p o i n t } }$ line circuits, one of which contains $t .$

(2) Let C be a positive circuit whose features span an s-dimensional space with $s \geq 2 ;$ then |C| ≤ ms + 1 (its dyads lie in an ms-dimensional space). Extend the features of C by $d - s$ data points to a basis of $\mathbb { R } ^ { d } .$ , and attach to each added point its $( m + 1 )$ )-point line circuit from (1). Summing all these positive relations gives a spanning zero certificate of support at most $m s + 1 + ( d - s ) ( m + 1 ) = k - s + 1 \leq k - 1$ , contradicting $\tau = k$ . A line circuit of size $\leq m$ similarly completes with $d - 1$ line circuits to support $\leq m + ( d - 1 ) ( m + 1 ) = k - 1$ . A point with $x _ { i } \neq 0 = \rho _ { i }$ is by itself a 1-point line “circuit” $( M _ { i } = 0 )$ and is excluded the same way.

(3) Suppose $d + 1$ distinct nonzero feature lines exist; take an $( m + 1 )$ -point line circuit on each (possible by (1)) and let J be their disjoint union, $| J | = ( d { + } 1 ) ( m { + } 1 )$ . Then dim ker $A _ { J } \geq$ $| J | - m d = d + m + 1$ . The sum of the circuits’ positive kernel vectors is strictly positive on $J ,$ so the nonnegative kernel cone spans ker $A _ { J }$ . But each extreme ray of that cone is a positive circuit, hence by (2) supported on a single line and of size $m + 1 ;$ on each line the kernel of the corresponding block is one-dimensional, so there are at most $d + 1$ extreme rays, spanning at most $d + 1$ dimensions — contradicting $d + m + 1 > d + 1$ . As the features span $\mathbb { R } ^ { d }$ , there are exactly d independent lines. □

## 5.3 The m-point mean lemma

Lemma 5.3 (m-point mean lemma). Let $z _ { i } \in \mathbb { R } ^ { m }$ carry weights $p _ { i } > 0$ with $\textstyle \sum _ { i } p _ { i } z _ { i } \ = \ 0$ and let $\begin{array} { r } { E = \sum _ { i } p _ { i } \| z _ { i } \| ^ { 2 } } \end{array}$ . Then some convex combination δ of at most m of the points satisfies $\lVert \delta \rVert ^ { 2 } \leq E / m ^ { 2 }$ . The constant is attained by the uniformly weighted regular simplex.

Proof. Decompose p inside the cone $\begin{array} { r } { \{ q \ge 0 : \sum _ { i } q _ { i } z _ { i } = 0 \} } \end{array}$ into positive circuits and normalize each to a probability vector; each circuit has support at most $m + 1$ and the second moments average to $E ,$ so some circuit $q$ has $E _ { q } \ \leq \ E$ . If $E _ { q } = 0$ , all its points vanish and a single point gives $\delta \ = \ 0$ . If its support has size $\leq m$ , its own zero mean is the desired $\delta \ = \ 0$

Otherwise the support is $q _ { 1 } , \ldots , q _ { m + 1 } > 0$ with $\begin{array} { r } { \sum _ { j } q _ { j } z _ { j } = 0 ; } \end{array}$ deleting point $j$ leaves the convex combination $\begin{array} { r } { \mu _ { j } \ = \ - \frac { q _ { j } } { 1 - q _ { j } } z _ { j } } \end{array}$ of the other m points. With $t _ { j } = q _ { j } \| z _ { j } \| ^ { 2 } / E _ { q } ( \mathrm { s o } \sum _ { j } t _ { j } = 1 )$ $\| \mu _ { j } \| ^ { 2 } / E _ { q } = q _ { j } t _ { j } / ( 1 - q _ { j } ) ^ { 2 }$ . Since $\begin{array} { r } { \sum _ { j } \frac { ( 1 - q _ { j } ) ^ { 2 } } { q _ { j } } = \sum _ { j } \frac { 1 } { q _ { j } } - 2 ( m + 1 ) + 1 \ge ( m + 1 ) ^ { 2 } - 2 ( m + 1 ) + 1 = m ^ { 2 } } \end{array}$ (Cauchy–Schwarz), not all $j$ can violate $\| \mu _ { j } \| ^ { 2 } \le E _ { q } / m ^ { 2 }$ , for otherwise summing $\begin{array} { r } { t _ { j } > \frac { ( 1 - q _ { j } ) ^ { 2 } } { m ^ { 2 } q _ { j } } } \end{array}$ over $j$ yields $1 > 1$ . Finally $E _ { q } \leq E$ . Tightness: for the regular simplex with uniform weights all $\| \mu _ { j } \| ^ { 2 } = E / m ^ { 2 }$ □

## 5.4 Assembly

Theorem 5.4 (Near-threshold value). For all d, m $\geq 2 , { \cal F } _ { \mathrm { w e i g h t e d } } \big ( d , m , ( m + 1 ) d - 1 \big ) = 1 + \frac { 1 } { d m ^ { 2 } }$

Proof. Upper bound. Let $n _ { 0 } = ( m + 1 ) d - 1$ and fix a dataset. If the feature rank is $r < d ,$ Theorem 3.2 gives a certificate of support $( m + 1 ) r \leq n _ { 0 }$ and the ratio is 1; the same holds if $L _ { D } ^ { \star } = 0$ (basis selection, as in Section 4) or if $\tau ( D ) \ \leq \ n _ { 0 }$ . In the remaining hard case $\tau ( D ) = ( m + 1 ) d .$ , Theorem 5.2 provides a basis $u _ { 1 } , \ldots , u _ { d }$ and scalars $s _ { i } \neq 0$ with $x _ { i } = s _ { i } u _ { j ( i ) }$ for every nonzero-feature point. Reading the first-order condition $\textstyle \sum _ { i } M _ { i } = 0$ against the basis $( u _ { j } )$ , each line satisfies $\begin{array} { r } { \sum _ { i \in I _ { j } } s _ { i } \rho _ { i } = 0 } \end{array}$ . Set, per line,

$$
z _ { i } = \rho _ { i } / s _ { i } , \qquad p _ { i } = s _ { i } ^ { 2 } / g _ { j } , \qquad g _ { j } = \sum _ { i \in I _ { j } } s _ { i } ^ { 2 } , \qquad R _ { j } = \sum _ { i \in I _ { j } } \lVert \rho _ { i } \rVert ^ { 2 } ,
$$

so that $\textstyle \sum _ { i \in I _ { j } } p _ { i } z _ { i } = 0$ and $\textstyle \sum _ { i } p _ { i } \| z _ { i } \| ^ { 2 } = R _ { j } / g _ { j }$ . Choose the lightest line $j _ { \star }$ with $\begin{array} { r } { R _ { j _ { \star } } \leq \frac 1 d \sum _ { j } R _ { j } \leq } \end{array}$ $\frac { N L _ { D } ^ { \star } } { d }$ (features equal to 0 only increase $L _ { D } ^ { \star } )$ . On each line $j \neq j$ pick, by conic Carath´eodory, at most $m + 1$ points with a positive zero-mean combination of the $z _ { i } ;$ on line $j _ { \star }$ pick at most m points and a convex combination $\delta$ with $\begin{array} { r } { \| \delta \| ^ { 2 } \le \frac { R _ { j _ { \star } } } { m ^ { 2 } g _ { j _ { \star } } } } \end{array}$ (Lemma 5.3). Translate the per-line convex coeficients $\alpha _ { i }$ into selection weights $c _ { i } = \alpha _ { i } / s _ { i } ^ { 2 } > 0$ (rescaling each line independently and normalizing globally); the weighted normal equations then decouple along the basis and give $\widehat { W } = W ^ { \circ } + H$ with $H u _ { j } = 0$ for $j \neq j _ { \star }$ and $H u _ { j _ { \star } } = \delta ;$ the selected features contain all d lines, so the weighted Gram matrix is positive definite and the minimizer is unique. The support totals $( d - 1 ) ( m + 1 ) + m = n _ { 0 }$ . By (1),

$$
L _ { D } ( \widehat { W } ) - L _ { D } ^ { \star } = \frac { g _ { j _ { \star } } \| \delta \| ^ { 2 } } { N } \leq \frac { R _ { j _ { \star } } } { N m ^ { 2 } } \leq \frac { L _ { D } ^ { \star } } { d m ^ { 2 } } .
$$

Lower bound. Take the regular-simplex axial instance: let $r _ { 1 } , \hdots , r _ { m + 1 } \in \mathbb { R } ^ { m }$ be regular simplex vertices with $\textstyle \sum _ { j } r _ { j } = 0$ and $\textstyle \| r _ { j } \| ^ { 2 } = { \frac { m } { m + 1 } }$ , whose facets are at squared distance $h ^ { 2 } =$ $\frac { 1 } { m ( m + 1 ) }$ from the origin; put $w = h e _ { 1 }$ and $D = \{ ( e _ { i } , w + r _ { j } ) \} _ { i \in [ d ] , j \in [ m + 1 ] } ,$ . The unique full minimizer is $W ^ { \circ } = [ w \cdots w ]$ with $\begin{array} { r } { L _ { D } ^ { \star } = \frac { m } { m + 1 } } \end{array}$ . For any selection of support at most $N - 1$ ， some axis is empty (column zeroed by the Frobenius rule; deviation $\| w \| ^ { 2 } = h ^ { 2 } )$ or uses a proper subset (deviation at least the facet distance $h ^ { 2 } )$ . By the analogue of (4), the loss exceeds $L _ { D } ^ { \star }$ by at least $h ^ { 2 } / d ,$ and deleting one vertex with uniform facet weights attains it. The ratio is $\begin{array} { r } { 1 + \frac { h ^ { 2 } / d } { m / ( m + 1 ) } = 1 + \frac { 1 } { d m ^ { 2 } } } \end{array}$ □

Remark 5.5 (General centered-simplex families). The same computation applies to any centered simplex $\begin{array} { r } { r _ { 1 } , \ldots , r _ { m + 1 } \left( \sum _ { j } r _ { j } = 0 , S = \sum _ { j } \| r _ { j } \| ^ { 2 } , h = \operatorname* { m i n } _ { j } \mathrm { d i s t } ( 0 , \operatorname { c o n v } \{ r _ { k } \} _ { k \neq j } ) \right) } \end{array}$ , provided the columns of $W ^ { \circ }$ are chosen with $\| W ^ { \circ } e _ { i } \| \geq h \colon$ the exact $( N - 1 )$ )-point ratio is $1 + { \frac { ( m + 1 ) h ^ { 2 } } { d S } }$ ， and within this family $h ^ { 2 } \le S / ( m ^ { 2 } ( m + 1 ) )$ with equality exactly for regular simplices. The integer instance of Definition 3.3 realizes the slightly smaller rational value of Theorem 3.6 with integer data.

## 6 The general k-point mean lemma

Lemma 5.3 is the case $k = M$ of a sharp family of sparsification bounds, which we record here both for its own sake and as a tool for Section 7. Unlike everything else in this paper, the general case relies on a recently proved deep geometric inequality — the full-dimensional Grace–Danielsson inequality conjectured by Egan and proved by Drozdov [Dro25]. None of Theorems 3.7, 4.1, 5.4 or 7.7 depends on this section; the only case used elsewhere is $k = M$ (Lemma 5.3) and the trivial $k = 1$ , both self-contained.

Theorem 6.1 (k-point mean lemma). Let $z _ { i } \in \mathbb { R } ^ { M }$ carry weights $p _ { i } > 0 , \sum _ { i } p _ { i } = 1$ , with $\textstyle \sum _ { i } p _ { i } z _ { i } = 0$ , and $\begin{array} { r } { E = \sum _ { i } p _ { i } \| z _ { i } \| ^ { 2 } } \end{array}$ . For every $1 \leq k \leq M + 1$ there is a convex combination δ of at most k of the points with

$$
\| \delta \| ^ { 2 } \leq E \cdot \frac { M + 1 - k } { k M } .
$$

The constant is sharp: for the uniformly weighted regular M-simplex, every convex combination supported on k points has squared norm at least $E \frac { \overset { \smile } { M } + 1 - k } { k M }$ , with equality for k vertices at equal weights.

The proof route: (i) reduce to a single positive circuit of support $s \leq M + 1$ (extreme points of the zero-mean polytope, as in Lemma 5.3); (ii) apply the following sparsification lemma with $a _ { i } = z _ { i }$

Lemma 6.2 (Simplex skeleton sparsification). Let $a _ { 1 } , \dots , a _ { s }$ lie in a Hilbert space, let q be a probability vector with full support s, and let $\begin{array} { r } { V _ { q } = \sum _ { i } q _ { i } \| a _ { i } - \bar { a } _ { q } \| ^ { 2 } } \end{array}$ where $\begin{array} { r } { \bar { a } _ { q } = \sum _ { i } q _ { i } a _ { i } } \end{array}$ . For every $1 \leq k \leq s$ there is a probability vector λ with | supp $\lambda | \leq k$ and

$$
\Big \| \sum _ { i } \lambda _ { i } a _ { i } - \bar { a } _ { q } \Big \| ^ { 2 } \ \leq \ \frac { s - k } { k ( s - 1 ) } V _ { q } .
$$

Proof sketch with full statements of the two nontrivial steps. We construct a martingale of probability vectors that successively kills coordinates. The engine is:

Boundary variance fact. Let q have $r \geq 2$ positive coordinates and let $H \succeq 0$ be a quadratic form. Then there is a random probability vector $Q _ { H }$ supported on the boundary $\partial \Delta _ { r } = \{ u \in$ $\Delta _ { r } : \operatorname* { m i n } _ { i } u _ { i } = 0 \}$ with $\mathbb { E } [ Q _ { H } ] = q$ and

$$
\mathbb { E } \big [ ( Q _ { H } - q ) ^ { \top } H ( Q _ { H } - q ) \big ] \ \leq \ \frac { \mathrm { t r } \big ( H ( \mathrm { D i a g } q - q q ^ { \top } ) \big ) } { ( r - 1 ) ^ { 2 } } .
$$

(One boundary variable per form H sufices for our purposes; a single Q working for all H simultaneously also exists, by separating the compact convex set of achievable covariances from $\{ K : K \preceq ( \mathrm { D i a g } q - q q ^ { \top } ) / ( r - 1 ) ^ { 2 } \}$ , but we do not need it.)

Proof of the fact. The left-hand side, minimized over admissible laws, is a finite moment problem on the compact set $\partial \Delta _ { r }$ ; strong duality (a supporting hyperplane to the compact convex set of achievable moment pairs) gives

$$
\Phi _ { H } ( q ) = \operatorname* { s u p } \Bigl \{ \ell ( q ) : \ell \mathrm { ~ a f f i n e , ~ } \ell ( u ) \leq ( u - q ) ^ { \top } H ( u - q ) \forall u \in \partial \Delta _ { r } \Bigr \} .
$$

Fix a feasible ℓ and (after adding $\varepsilon P _ { T }$ to H on the tangent space $T = \{ \mathbf { 1 } ^ { \top } y = 0 \}$ and letting $\varepsilon \downarrow 0 )$ assume $H \succ 0$ on T. Writing $\ell ( q + y ) = a + 2 \langle c , y \rangle _ { H }$ and $R ^ { 2 } = a + \| c \| _ { H } ^ { 2 }$ , feasibility says the H-ball $B _ { H } ( c , R )$ misses $\partial \Delta _ { r } - q ;$ since $0 \in B _ { H } ( c , R )$ when $a > 0$ , convexity forces $B _ { H } ( c , R ) \subseteq \Delta _ { r } - q$ The desired bound $\begin{array} { r } { ( r - 1 ) ^ { 2 } \ell ( q ) \leq \mathrm { t r } ( H ( \mathrm { D i a g } q - q q ^ { \top } ) ) = \sum _ { i } q _ { i } ( e _ { i } - q ) ^ { \top } H ( e _ { i } - q ) } \end{array}$ then follows from:

Sublemma G (simplex–ball interpolation). Let an n-simplex $T \subset \mathbb { R } ^ { n }$ contain the ball $B ( c , R )$ with $0 \in B ( c , R )$ , and let $\textstyle 0 = \sum _ { i } q _ { i } v _ { i }$ be the barycentric representation of the origin in the vertices $v _ { i } \ o f T$ . Then

$$
\sum _ { i } q _ { i } \| v _ { i } \| ^ { 2 } \geq n ^ { 2 } ( R ^ { 2 } - \| c \| ^ { 2 } ) .
$$

Proof of Sublemma G. Normalize the ball to the unit ball centered at the origin; the distinguished point becomes y with $u = \| y \| < 1$ , and the left side becomes the value at y of the afine interpolation of $\| \cdot \| ^ { 2 }$ on the vertices, which for any simplex with circumcenter O and circumradius R equals $P _ { T } ( y ) = \mathcal { R } ^ { 2 } - \Vert O - y \Vert ^ { 2 }$ plus $\| y \| ^ { 2 }$ . Slide each facet inward until tangent to the unit ball, obtaining the inscribed simplex $S \subseteq T$ with insphere the unit ball; by the variance decomposition of convex combinations (write each vertex of S in the vertices of $T$ and y in the vertices of S), the interpolation value only decreases: $P _ { T } ( y ) \ge P _ { S } ( y )$ . For S, let $d = \| O \|$ and write $\mathcal { R } = \boldsymbol { n } + \boldsymbol { s }$ with $s \geq 0$ . Drozdov’s theorem [Dro25] (the Grace–Danielsson inequality in every dimension: for any n-simplex with inradius r, circumradius R, and center distance $d ,$ one has $( \mathcal { R } - n r ) ( \mathcal { R } + ( n - 2 ) r ) \geq d ^ { 2 } \mathrm { ~ }$ ; here $r = 1 )$ gives $d ^ { 2 } \leq s ( s + 2 n - 2 )$ , whence $\mathcal { R } ^ { 2 } - d ^ { 2 } - n ^ { 2 } \ge 2 s$ and $s \geq \sqrt { d ^ { 2 } + ( n - 1 ) ^ { 2 } } - ( n - 1 )$ . Then

$$
{ \cal P } _ { S } ( y ) - n ^ { 2 } ( 1 - u ^ { 2 } ) \ \geq \ 2 s - 2 d u + ( n ^ { 2 } - 1 ) u ^ { 2 } ,
$$

and minimizing the right-hand side over $d \geq 0$ (optimum at $d = ( n - 1 ) u / \sqrt { 1 - u ^ { 2 } } )$ leaves $( n - 1 ) \big [ ( n + 1 ) u ^ { 2 } - 2 ( 1 - \sqrt { 1 - u ^ { 2 } } ) \big ] \geq 0$ , which holds since $1 - { \sqrt { 1 - u ^ { 2 } } } \leq u ^ { 2 } .$ . This proves Sublemma G, hence the boundary variance fact. □

Now iterate: starting from q with support s, apply the fact with $H = A ^ { \top } A$ (where $A =$ $[ a _ { 1 } \cdots a _ { s } ] )$ , obtaining a boundary law Q with $\mathbb { E } [ \bar { a } _ { Q } ] = \bar { a } _ { q }$ and $\mathbb { E } \| \bar { a } _ { Q } - \bar { a } _ { q } \| ^ { 2 } \leq V _ { q } / ( r - 1 ) ^ { 2 }$ ; the total-variance identity $\mathbb { E } [ V _ { Q } ] = V _ { q } - \mathbb { E } \lVert \bar { a } _ { Q } - \bar { a } _ { q } \rVert ^ { 2 }$ gives $\begin{array} { r } { \mathbb { E } [ V _ { Q } ] \geq \left( 1 - \frac { 1 } { ( r - 1 ) ^ { 2 } } \right) V _ { q } } \end{array}$ . A step may kill several coordinates at once; setting $\begin{array} { r } { f _ { r } = 1 - \frac { 1 } { ( r - 1 ) ^ { 2 } } } \end{array}$ and $\textstyle \Gamma _ { j } = \prod _ { t = k + 1 } ^ { j } f _ { t }$ (with $\Gamma _ { j } = 1$ for $j \leq k )$ , which is nonincreasing in j, a backward induction on the support size shows that the terminal variance obeys $\mathbb { E } [ V _ { \Lambda } ] \ge \Gamma _ { s } V _ { q }$ regardless of how many coordinates each step kills. Since the mean sequence is a martingale with orthogonal increments, $\mathbb { E } \| \bar { \boldsymbol { a } } _ { \Lambda } - \bar { \boldsymbol { a } } _ { q } \| ^ { 2 } = V _ { q } - \mathbb { E } [ V _ { \Lambda } ] \le \left( 1 - \Gamma _ { s } \right) V _ { q }$ , and the telescoping product $\begin{array} { r } { \Gamma _ { s } = \prod _ { r = k + 1 } ^ { s } { \frac { r ( r - 2 ) } { ( r - 1 ) ^ { 2 } } } = { \frac { s ( k - 1 ) } { k ( s - 1 ) } } } \end{array}$ yields the bound $\begin{array} { r } { \frac { s - k } { k ( s - 1 ) } V _ { q } ; } \end{array}$ some realization λ of the terminal law achieves it. □

Proof of Theorem 6.1. Reduce to a circuit q of support $s \leq M + 1$ and second moment $E _ { q } \leq$ E exactly as in Lemma 5.3 (including the degenerate branches $E _ { q } = 0$ and $s \leq k )$ . Apply Lemma 6.2 with $a _ { i } = z _ { i } , \bar { a } _ { q } = 0 \mathrm { . }$ : some k-sparse convex δ has $\begin{array} { r } { \| \delta \| ^ { 2 } \leq \frac { s - k } { k ( s - 1 ) } E _ { q } ; } \end{array}$ since $s \mapsto { \frac { s - k } { s - 1 } }$ is nondecreasing for $k \geq 1$ , the constant is at most $\textstyle { \frac { M + 1 - k } { k M } }$ . Sharpness: for regular simplex vertices, $\begin{array} { r } { \| \sum _ { i } \alpha _ { i } v _ { i } \| ^ { 2 } = R ^ { 2 } \big ( \frac { M + 1 } { M } \sum _ { i } \alpha _ { i } ^ { 2 } - \frac { 1 } { M } \big ) \geq R ^ { 2 } \frac { M + 1 - k } { k M } } \end{array}$ whenever | supp $\alpha | \le k$ , using $\sum \alpha _ { i } ^ { 2 } \geq 1 / k$ . □

## 7 The smallest intermediate cell: $( d , m ) = ( 2 , 2 )$

For $d < n < ( m + 1 ) d - 1$ the profile $F _ { \mathrm { w e i g h t e d } } ( d , m , n )$ remains open — as does its scalar counterpart, the intermediate regime of Question 2 of [HMSY25b]. In this section we treat the

smallest cell $( d , m ) = ( 2 , 2 )$ , where the open budgets are $n = 3 ,$ , 4. We prove the interval theorem (Theorem 7.7), reduce the conjectured exact values to a finite moment problem, and assemble structural evidence.

## 7.1 Lower bounds: an explicit axial instance

Proposition 7.1. There is an explicit dataset $D ^ { \triangle }$ with

$$
\frac { L _ { D ^ { \triangle } } ^ { \star } ( 3 ; \mathrm { w e i g h t e d } ) } { L _ { D ^ { \triangle } } ^ { \star } } = \frac { 1 3 } { 8 } , \qquad \frac { L _ { D ^ { \triangle } } ^ { \star } ( 4 ; \mathrm { w e i g h t e d } ) } { L _ { D ^ { \triangle } } ^ { \star } } = \frac { 5 } { 4 } .
$$

Hence $F _ { \mathrm { w e i g h t e d } } ( 2 , 2 , 3 ) \geq { \frac { 1 3 } { 8 } }$ and $F _ { \mathrm { w e i g h t e d } } ( 2 , 2 , 4 ) \geq { \frac { 5 } { 4 } }$

Proof. Let $r _ { 1 } , r _ { 2 } , r _ { 3 } \in \mathbb { R } ^ { 2 }$ be the vertices of a centered equilateral triangle with $\begin{array} { r } { \| r _ { j } \| ^ { 2 } = \frac { 2 } { 3 } } \end{array}$ (so the total is $S = 2$ , edge midpoint distance squared $\textstyle { \frac { 1 } { 6 } } )$ , and let $w \in \mathbb { R } ^ { 2 }$ with $\begin{array} { r } { \| w \| ^ { 2 } = \frac { 5 } { 6 } . } \end{array}$ Take $D ^ { \triangle } = \{ ( e _ { i } , w + r _ { j } ) \} _ { i \in [ 2 ] , j \in [ 3 ] }$ . As in Lemma 3.4, $\begin{array} { r } { W ^ { \circ } = [ w \ w ] , L _ { D } ^ { \star } = \frac { 2 } { 3 } } \end{array}$ , and a selection allocating $k _ { i }$ points to axis i incurs excess ${ \textstyle \frac { 1 } { 2 } } ( c ( k _ { 1 } ) + c ( k _ { 2 } ) )$ where $\begin{array} { r } { c ( 3 ) = 0 , c ( 2 ) = \frac { 1 } { 6 } } \end{array}$ (best edge), $\begin{array} { r } { c ( 1 ) = \frac { 2 } { 3 } } \end{array}$ (best vertex), $\begin{array} { r } { c ( 0 ) = \| w \| ^ { 2 } = \frac { 5 } { 6 } } \end{array}$ (empty axis). For budget 3 the minimum over allocations is $\begin{array} { r } { c ( 2 ) + c ( 1 ) = c ( 3 ) + c ( 0 ) = \frac { 5 } { 6 } } \end{array}$ , giving ratio $\textstyle 1 + { \frac { 5 / 1 2 } { 2 / 3 } } = { \frac { 1 3 } { 8 } }$ ; for budget 4 it is $\begin{array} { r } { 2 c ( 2 ) = \frac { 1 } { 3 } } \end{array}$ , giving $\textstyle 1 + { \frac { 1 / 6 } { 2 / 3 } } = { \frac { 5 } { 4 } }$ . (We verified both values by exhaustive exact enumeration of all supports.) □

## 7.2 A complex dictionary for the hard branches

Fix a (2, 2) dataset with feature rank 2 and $L _ { D } ^ { \star } > 0 ;$ after a linear change of features (which leaves the loss values invariant) assume the data whitened, $\begin{array} { r } { \sum _ { i } x _ { i } x _ { i } ^ { \top } = \gamma I _ { 2 } } \end{array}$ . Identify $\mathbb { R } ^ { 2 } \cong \mathbb { C } ;$ write $\xi _ { i }$ for the feature and $\eta _ { i }$ for the residual $\rho _ { i }$ of point i (viewed in C), and for $\xi _ { i } \neq 0$ set

$$
p _ { i } = \frac { | \xi _ { i } | ^ { 2 } } { \sum _ { j } | \xi _ { j } | ^ { 2 } } , \qquad t _ { i } = \frac { \bar { \xi } _ { i } } { \xi _ { i } } \in S ^ { 1 } , \qquad z _ { i } = \frac { \eta _ { i } } { \xi _ { i } } \in \mathbb { C } , \qquad E = \sum _ { i } p _ { i } | z _ { i } | ^ { 2 } .
$$

Whitening is equivalent to $\begin{array} { r } { \sum _ { i } \xi _ { i } ^ { 2 } = 0 , \mathrm { i . e . } \ \mathbb { E } _ { p } t = 0 } \end{array}$ , and the first-order condition $\begin{array} { r } { \sum _ { i } \rho _ { i } x _ { i } ^ { \top } = 0 } \end{array}$ is equivalent to the pair $\mathbb { E } _ { p } z = 0 , \mathbb { E } _ { p } ( \bar { t } z ) = 0$ . A selection with weights c corresponds to $q _ { i } \propto c _ { i } | \xi _ { i } | ^ { 2 }$ writing $\mu = \mathbb { E } _ { q } t , \nu = \mathbb { E } _ { q } z , \lambda = \mathbb { E } _ { q } ( \bar { t } z )$ , the selected model deviates from $W ^ { \circ }$ by the real-linear map $\xi \mapsto a \xi + b \bar { \xi }$ with

$$
\left( { \begin{array} { c c } { 1 } & { \mu } \\ { \bar { \mu } } & { 1 } \end{array} } \right) \left( { \begin{array} { c } { a } \\ { b } \end{array} } \right) = \binom { \nu } { \lambda } , \qquad | \mu | < 1 \iff \mathrm { s e l e c t e d ~ f e a t u r e s ~ s p a n ~ } \mathbb { R } ^ { 2 } ,\tag{6}
$$

and the excess loss satisfies $\big ( L _ { D } ( \widehat { W } ) - L _ { D } ^ { \star } \big ) / L _ { D } ^ { \star } = C ( q ) / E$ with $C ( q ) = | a | ^ { 2 } + | b | ^ { 2 }$ . (Zero-feature points only enlarge $L _ { D } ^ { \star }$ , so all upper bounds proved through the dictionary are conservative.) Conversely, every finite system $( p _ { i } , t _ { i } , z _ { i } )$ with the three moment conditions is realized by a whitened dataset $( \xi _ { i } = \sqrt { p _ { i } } s _ { i }$ with $\bar { s } _ { i } / s _ { i } = t _ { i } , \eta _ { i } = z _ { i } \xi _ { i } )$

For $k \in \{ 3 , 4 \}$ define

$$
\Gamma _ { k } = \operatorname* { s u p } _ { \mathrm { \tiny ~ m o m e n t ~ s y s t e m s } } \frac { 1 } { E } \operatorname* { i n f } _ { \begin{array} { c } { | \operatorname* { s u p } q | \leq k } \\ { | \mu ( q ) | < 1 } \end{array} } C ( q ) .
$$

## 7.3 Interpolation identities

Lemma 7.2 (Interpolation convex combination). For $t _ { i } \neq t _ { j }$ let $\begin{array} { r } { B _ { i j } = \left( \frac { t _ { i } z _ { j } - t _ { j } z _ { i } } { t _ { i } - t _ { j } } , \ \frac { z _ { i } - z _ { j } } { t _ { i } - t _ { j } } \right) \in \mathbb { C } ^ { 2 } } \end{array}$ (the coeficients of the afine function of t through the two points) and $d _ { i j } = | t _ { i } - t _ { j } | ^ { 2 }$ . Then for every admissible q,

$$
D ( q ) : = 1 - | \mu | ^ { 2 } = \sum _ { i < j } q _ { i } q _ { j } d _ { i j } , \qquad ( a , b ) = \sum _ { i < j \atop t _ { i } \neq t _ { j } } \frac { q _ { i } q _ { j } d _ { i j } } { D ( q ) } B _ { i j } .
$$

In particular $( a , b )$ is a convex combination of the pairwise interpolation coeficients; the conditioning factor $( 1 - | \mu | ^ { 2 } ) ^ { - 1 }$ is absorbed entirely.

Proof. $\begin{array} { r } { 1 - | \mu | ^ { 2 } = \frac { 1 } { 2 } \sum _ { i , j } q _ { i } q _ { j } | t _ { i } - t _ { j } | ^ { 2 } } \end{array}$ expands the first identity. With $w _ { i } = \bar { t } _ { i } z _ { i }$ one checks $d _ { i j } a _ { i j } \ : = \ : ( t _ { i } - t _ { j } ) ( w _ { i } - \ ' w _ { j } )$ and $d _ { i j } b _ { i j } = ( { \bar { t } } _ { i } - { \bar { t } } _ { j } ) ( z _ { i } - z _ { j } )$ ; summing against $q _ { i } q _ { j }$ and using $\begin{array} { r } { \sum _ { i < j } q _ { i } q _ { j } ( x _ { i } - x _ { j } ) ( y _ { i } - y _ { j } ) = \sum _ { i } q _ { i } x _ { i } y _ { i } - ( \sum _ { i } q _ { i } x _ { i } ) ( \sum _ { i } q _ { i } y _ { i } ) } \end{array}$ yields $\begin{array} { r } { \sum _ { i < j } q _ { i } q _ { j } d _ { i j } a _ { i j } = \nu - \mu \lambda } \end{array}$ and $\begin{array} { r } { \sum _ { i < j } q _ { i } q _ { j } d _ { i j } b _ { i j } = \lambda - \bar { \mu } \nu } \end{array}$ , which is $D ( q ) \cdot ( a , b )$ by (6). Same-direction pairs have $d _ { i j } = 0$ and are simply omitted. □

Lemma 7.3 (Three-point closure identity). For $2 \leq | S | \leq 3$ let $K _ { S } = \mathrm { c o n v } \{ B _ { i j } : i , j \in S , ~ t _ { i } \neq$ $t _ { j } \}$ (discard S with no valid pair). Then

$$
\operatorname* { i n f } _ { | \mu ( q ) | < 1 } C ( q ) \ = \ \mathrm { d i s t } ^ { 2 } \bigl ( 0 , K _ { S } \bigr ) , \qquad \operatorname* { i n f } _ { | \operatorname { s u p p } q | \leq 3 } C ( q ) \ = \ \operatorname* { m i n } _ { S } \mathrm { d i s t } ^ { 2 } ( 0 , K _ { S } ) .
$$

For three pairwise distinct directions the interior of the edge-weight triangle is attained by positive weights (via $\begin{array} { r } { q _ { 1 } { : } q _ { 2 } { : } q _ { 3 } = \frac { d _ { 2 3 } } { \alpha _ { 2 3 } } { : } \frac { d _ { 1 3 } } { \alpha _ { 1 3 } } { : } \frac { d _ { 1 2 } } { \alpha _ { 1 2 } } \big ) } \end{array}$ , the vertices by two-point supports, and non-degenerate edge interiors only as limits with $| \mu | \to 1$ ; if two of the three directions coincide, the corresponding segment of $K _ { S }$ is attained exactly. Moreover, for three distinct directions the three points $B _ { i j } , B _ { i k } , B _ { j k }$ , when distinct, are never collinear over R.

Proof. By Lemma 7.2, over supp q $\subseteq S$ the value $( a , b )$ ranges over convex combinations of the valid $B _ { i j }$ with weights $r _ { i j } \propto q _ { i } q _ { j } d _ { i j }$ ; the parametrization above shows all interior weight profiles are realized, and the boundary cases are checked directly (for a repeated direction, distributing weight inside the repeated pair moves along the segment with $| \mu | < 1$ preserved). Distances to closures equal infima of continuous functions over the realized sets. For non-collinearity: $B _ { i k } - B _ { i j }$ is a complex multiple of $\left( - t _ { i } , 1 \right)$ and $B _ { j k } - B _ { i j } \ \mathrm { o f } \ ( - t _ { j } , 1 )$ ; real proportionality of nonzero such vectors forces $t _ { i } = t _ { j }$ □

Lemma 7.4 (Global edge identities). Let $w _ { i j } = p _ { i } p _ { j } d _ { i j }$ . Then $\begin{array} { r } { \sum _ { i < j } w _ { i j } = 1 , \sum _ { i < j , t _ { i } \neq t _ { j } } w _ { i j } B _ { i j } = } \end{array}$ 0, and

$$
\sum _ { i < j \atop t _ { i } \neq t _ { j } } w _ { i j } \| B _ { i j } \| ^ { 2 } = 2 E - 2 \sum _ { G } \biggl ( P _ { G } { \sum _ { i \in G } } p _ { i } | z _ { i } | ^ { 2 } - \biggl | \sum _ { i \in G } p _ { i } z _ { i } \biggr | ^ { 2 } \biggr ) \leq 2 E ,
$$

where G ranges over the classes of equal direction and $\textstyle P _ { G } = \sum _ { i \in G } p _ { i }$ . In particular some valid pair has $\| B _ { i j } \| ^ { 2 } \leq 2 E$

Proof. The first identity is Lemma 7.2 at $q \ : = \ : p$ with $\mu = 0 ;$ the second is Lemma 7.2 at $q \ = \ p .$ , where $( a , b ) = ( 0 , 0 )$ by the moment conditions. For the third, expand $d _ { i j } \| B _ { i j } \| ^ { 2 } =$ $| t _ { i } z _ { j } - t _ { j } z _ { i } | ^ { 2 } + | z _ { i } - z _ { j } | ^ { 2 }$ and sum: over all pairs, each of the two sums telescopes to $E - | \mathbb { E } _ { p } ( { \bar { t } } z ) | ^ { 2 } = E$ and $E - | \mathbb { E } _ { p } z | ^ { 2 } = E$ respectively; same-direction pairs, for which $B _ { i j }$ is undefined, contribute $2 p _ { i } p _ { j } | z _ { i } - z _ { j } | ^ { 2 }$ , which regrouped over classes gives the correction term. □

## 7.4 Star localization and the two unconditional bounds

Lemma 7.5 (Star localization). Fix an atom i and set $\begin{array} { r } { P _ { i j } = \frac { 1 } { 2 } p _ { j } d _ { i j } } \end{array}$ for $j \neq i$ . Then $\textstyle \sum _ { j } P _ { i j } = 1$ the valid star coeficients $B _ { i j }$ lie on a real two-dimensional afine plane through $m _ { i }$ with $\| m _ { i } \| ^ { 2 } =$ $\frac { 1 } { 2 } \left| z _ { i } \right| ^ { 2 }$ , and $\begin{array} { r } { \sum _ { i } P _ { i j } B _ { i j } = m _ { i } , } \end{array}$ their variance is $\begin{array} { r } { V _ { i } = E + \frac { | z _ { i } | ^ { 2 } } { 2 } - L _ { i } } \end{array}$ where $\begin{array} { r } { L _ { i } = \sum _ { j : t _ { i } = t _ { i } } p _ { j } | z _ { j } - z _ { i } | ^ { 2 } \geq } \end{array}$ 0. Consequently there exist two star neighbours whose segment contains a point δ with

$$
\| \delta \| ^ { 2 } \leq \frac { E } { 4 } + \frac { 5 | z _ { i } | ^ { 2 } } { 8 } - \frac { L _ { i } } { 4 } ,
$$

and $\delta$ is a closure point of three-point selections (Lemma 7.3). Choosing i with $| z _ { i } | ^ { 2 } \leq E$

$$
\operatorname* { i n f } _ { | \mathbf { \phi } | < 1 } C ( q ) \ \leq \ { \frac { 7 E } { 8 } } , \qquad s o \qquad \Gamma _ { 3 } \leq { \frac { 7 } { 8 } } .\tag{7}
$$

Proof. Work in the real frame picture: choose $s _ { i } ^ { 2 } = t _ { i } ,$ , set $u _ { i } = \sqrt { 2 } ( \cos \theta _ { i } .$ , sin $\theta _ { i } ) ^ { \top }$ and $w _ { i } = \bar { s } _ { i } z _ { i } ;$ the moment conditions become the tight-frame identities $\begin{array} { r } { \sum _ { j } p _ { j } u _ { j } u _ { j } ^ { \top } = I _ { 2 } , \ \sum _ { j } p _ { j } u _ { j } w _ { j } \ = \ 0 } \end{array}$ and $d _ { i j } = \operatorname* { d e t } ( u _ { i } , u _ { j } ) ^ { 2 }$ With $n _ { i } ~ \perp ~ u _ { i }$ a unit vector, $P _ { i j } = p _ { j } ( u _ { j } ^ { \top } n _ { i } ) ^ { 2 }$ sums to $n _ { i } ^ { \top } I n _ { i } = 1 ;$ every valid pair coeficient can be written $\gamma _ { i j } = m _ { i } + n _ { i } \xi _ { i j }$ with $\begin{array} { r } { m _ { i } = \frac { 1 } { 2 } u _ { i } w _ { i } } \end{array}$ , and $\begin{array} { r } { \sum _ { j } P _ { i j } \xi _ { i j } = } \end{array}$ $\begin{array} { r } { { n } _ { i } ^ { \top } \big ( \sum _ { j } p _ { j } u _ { j } w _ { j } \big ) - { n } _ { i } ^ { \top } \big ( \sum _ { j } p _ { j } u _ { j } u _ { j } ^ { \top } \big ) m _ { i } = 0 } \end{array}$ , giving the mean identity; the variance computation gives ${ \dot { V } } _ { i } .$ Since the centered coeficients lie in a real plane (indeed a line through the origin of that plane after removing $m _ { i } )$ , Lemma 5.3 with $M = 2 , k = 2$ applied to the P-weighted zeromean system produces two neighbours and a point $\delta ^ { \prime }$ on their segment with $\lVert \delta ^ { \prime } - m _ { i } \rVert ^ { 2 } \leq V _ { i } / 4 ;$ orthogonality of $m _ { i }$ and the centered directions gives $\| \delta ^ { \prime } \| ^ { 2 } \leq \| m _ { i } \| ^ { 2 } + V _ { i } / 4$ , which is the stated bound. The bound (7) follows from $\operatorname* { m i n } _ { i } | z _ { i } | ^ { 2 } \leq E$ and $L _ { i } \geq 0$ □

Lemma 7.6 (Four-point $E / 2$ bound). For every moment system, inf $\textstyle | \operatorname { s u p p } q | \leq 4 , \ | \mu | < 1  C ( q ) \leq { \frac { E } { 2 } }$ ; hence $\Gamma _ { 4 } \leq \textstyle { \frac { 1 } { 2 } }$

Proof. Fix i with $| z _ { i } | ^ { 2 } \leq E$ . By Lemma 7.5 the star coeficients $\gamma _ { i j }$ lie in a real two-dimensional afine plane and average to $m _ { i }$ under the weights $P _ { i j } ;$ ; by Carath´eodory in the plane, $\begin{array} { r l } { m _ { i } } & { { } = } \end{array}$ $\textstyle \sum _ { r = 1 } ^ { 3 } \alpha _ { r } \gamma _ { i j }$ for some three neighbours and convex $\alpha .$ . Take $q _ { i } = 1 - \varepsilon , q _ { j _ { r } } = \varepsilon c _ { r }$ with $c _ { r } \propto \alpha _ { r } / d _ { i j _ { r } }$ by Lemma 7.2 the normalized edge weights of the star edges tend to $\alpha _ { r }$ while all leaf–leaf edges carry total weight $O ( \varepsilon )$ , so $( a , b ) \to m _ { i }$ as $\varepsilon \downarrow 0$ , along selections with support $\leq 4$ and $| \mu | < 1$ Hence the infimum is at most $\begin{array} { r } { \| m _ { i } \| ^ { 2 } = \frac 1 2 | z _ { i } | ^ { 2 } \leq \frac { E } { 2 } } \end{array}$ □

## 7.5 The interval theorem

Theorem 7.7. $\begin{array} { r } { F _ { \mathrm { w e i g h t e d } } ( 2 , 2 , 3 ) \in \left[ \frac { 1 3 } { 8 } , \frac { 1 5 } { 8 } \right] \ a n d \ F _ { \mathrm { w e i g h t e d } } ( 2 , 2 , 4 ) \in \left[ \frac { 5 } { 4 } , \frac { 3 } { 2 } \right] } \end{array}$

Proof. Lower bounds: Proposition 7.1. Upper bounds: fix a dataset and a budget $n \in \{ 3 , 4 \}$ If the feature rank is $r \leq 1$ , then $\tau ( D ) \leq ( m + 1 ) r \leq 3 \leq n$ (Theorem 3.2) and the ratio is 1; likewise if $L _ { D } ^ { \star } = 0 ~ \mathrm { o r } ~ \tau ( D ) \leq n$ . Otherwise the dictionary of Section 7.2 applies, and by (7) and Lemma 7.6 (a three-point selection is also a four-point selection) the ratio is at most $\begin{array} { r } { 1 + \Gamma _ { 3 } \le \frac { 1 5 } { 8 } } \end{array}$ for $n = 3$ and at most $\mathrm { 1 + r _ { 4 } \leq \frac { 3 } { 2 } }$ for $n = 4$ . The closure points used in Lemmas $7 . 5 \mathrm { - } 7 . 6$ are limits of admissible selections within the same budget, which sufices since the inner optimization in $F _ { \mathrm { w e i g h t e d } }$ is an infimum. □

## 7.6 The conjecture and its evidence

Conjecture 7.8. $F _ { \mathrm { w e i g h t e d } } ( 2 , 2 , 3 ) = { \textstyle { \frac { 1 3 } { 8 } } }$ and $F _ { \mathrm { w e i g h t e d } } ( 2 , 2 , 4 ) = { \frac { 5 } { 4 } }$ . Equivalently (by the branch analysis above and Lemma 7.3): for every moment system, inf<sub>| supp q|≤4,|µ|<1</sub> $\begin{array} { r } { C \le \frac { E } { 4 } } \end{array}$ and $\mathrm { i n f } _ { | \mathrm { s u p p } q | \leq 3 , | \mu | < 1 } C \leq$ $\frac { 5 E } { 8 }$

We record five independent pieces of structural evidence.

## 7.6.1 Finite reduction: at most seven atoms

Proposition 7.9. For fixed atoms $( t _ { i } , z _ { i } )$ , the admissible weight vectors p form a polytope cut out by at most 7 linear equalities, and the worst weight vector (minimizing E) may be taken at a basic feasible solution, of support at most 7. Consequently each inequality of Conjecture 7.8 holds for all finite systems if and only if it holds for all systems with at most seven atoms; and, since atoms may be split into identically placed copies without changing anything, it sufices to treat systems with exactly seven labelled atoms.

Proof. The inner quantity in $\complement _ { q } C ( q )$ depends only on the atom set, not on $p ; E ( p )$ is linear in $p ,$ constrained by normalization (1 equation) and the three complex moments (6 real equations). A basic optimal solution has support at most the constraint rank $\leq 7 ;$ its subsystem inherits the moment conditions, its witness lifts verbatim to the original system, and if the minimal $E ^ { \star } = 0$ all its supported $z _ { i }$ vanish while $\mathbb { E } t = 0$ guarantees two distinct directions, giving a two-point selection with $C = 0$ . Splitting an atom $( p _ { i } , t _ { i } , z _ { i } )$ into $( \theta p _ { i } , t _ { i } , z _ { i } ) , ( ( 1 - \theta ) p _ { i } , t _ { i } , z _ { i } )$ changes no moments; merging the copies in a witness does not increase its support. □

## 7.6.2 The two-direction class is solved sharply

Proposition 7.10. Suppose the atoms take exactly two directions. Then the directions are antipodal, each class has total weight $\textstyle { \frac { 1 } { 2 } }$ and zero conditional mean, and $C ( q ) = { \textstyle { \frac { 1 } { 2 } } } ( | m _ { + } | ^ { 2 } + | m _ { - } | ^ { 2 } )$ where $m _ { \pm }$ are the conditional means of the selected weights. Consequently

$$
\operatorname* { i n f } _ { | \mathrm { s u p p } q | \le 4 } C \le \frac { E } { 4 } , \qquad \operatorname* { i n f } _ { | \mathrm { s u p p } q | \le 3 } C \le \frac { 5 E } { 8 } ,
$$

and both constants are attained: for six equally weighted atoms with z-values the cube roots of unity over each direction, the three-point optimum is exactly $\frac { 5 E } { 8 }$ and the four-point optimum exactly $\textstyle { \frac { E } { 4 } }$ . Equality at $\frac { 5 L } { 8 }$ forces both classes to have equal energy, conditional weights $\textstyle { \frac { 1 } { 3 } }$ each, and equilateral z-triangles of equal radius. Furthermore, for five-atom two-direction systems the sharp constant at three points is $\textstyle { \frac { 4 } { 7 } } < { \frac { 5 } { 8 } }$ (attained by an equilateral triple against an antipodal pair with energy ratio 4:3).

Proof. $\mathbb { E } _ { p } t = 0$ with two unit directions forces antipodality and equal class weights; the two complex moments then force zero conditional class means. The regression fit at ± is the pair of selected conditional means, giving the formula for $C$ . For a class with conditional weights $\alpha _ { i }$ summing to 1 and zero mean, deleting one point leaves the two-point mean $- \frac { \alpha _ { i } z _ { i } } { 1 - \alpha _ { i } }$ , and $\begin{array} { r } { \sum _ { i } \frac { ( 1 - \alpha _ { i } ) ^ { 2 } } { \alpha _ { i } } \ge \frac { \angle } { \frac { \ d S } { \ d S } } , } \end{array}$ 4 for three atoms (Cauchy–Schwarz), so some deletion has squared norm $\le e _ { G } / 4 ;$ allocating (2, 2) or $( 2 , 1 )$ across the classes and balancing yields the bounds, with the stated equality analysis. The six-atom computation and the five-atom $\frac { 4 } { 7 }$ optimization (min $\{ e _ { 3 } , e _ { 3 } / 4 +$ $e _ { 2 } \} / ( e _ { 3 } + e _ { 2 } )$ maximized at $\begin{array} { r } { e _ { 2 } = \frac { 3 } { 4 } e _ { 3 } ) } \end{array}$ are elementary; we verified all exact values by enumeration.

## 7.6.3 The equality fiber is a stratified local maximum

Proposition 7.11. Let E be the set of 3+3 two-direction equality systems of Proposition 7.10. Then: (i) within the two-direction stratum, the three-point value drops linearly under energy imbalance: with class energies $E ( 1 \pm \eta )$ it equals $\frac { E } { 8 } ( 5 - 3 | \eta | ) \ : - \ : a$ downward cusp; (ii) for every sequence of moment systems converging to a point of E in which some direction class genuinely splits,

$$
\operatorname* { l i m } \operatorname* { s u p } \ \frac { 1 } { E } \operatorname* { i n f } _ { | \mathrm { s u p p } q | \leq 3 , | \mu | < 1 } C \ \leq \ \frac { 1 } { 2 } \ < \ \frac { 5 } { 8 } .
$$

Hence no nearby ascent direction exists: E is a local maximum of the three-point value in the stratified sense.

Proof. (i) is the balance computation in Proposition 7.10. For (ii), let atoms $j , k$ of the + class split: $\delta = t _ { j } - t _ { k } \to 0 , \delta \neq 0$ , with $t _ { j } , t _ { k } \to \tau$ and $z _ { j } , z _ { k }$ tending to distinct points of the limit equilateral triangle (radius $A , E \to A ^ { 2 } )$ Use the prediction coordinates $U _ { \pm } ( a , b ) = a \pm \tau b ,$ in which $\begin{array} { r } { C = \frac 1 2 ( | U _ { + } | ^ { 2 } + | U _ { - } | ^ { 2 } ) } \end{array}$ at the limit directions. Writing $t _ { j } = \tau + \eta _ { j } , t _ { k } = \tau + \eta _ { k }$ , the same-class edge $V = B _ { j k }$ satisfies the exact identities

$$
U _ { + } ( V ) = \frac { \eta _ { j } z _ { k } - \eta _ { k } z _ { j } } { \delta } , \qquad U _ { - } ( V ) = \frac { ( 2 \tau + \eta _ { j } ) z _ { k } - ( 2 \tau + \eta _ { k } ) z _ { j } } { \delta } ,
$$

so that $\left| \delta \right| U _ { + } ( V ) \to 0$ while $R _ { \delta } : = | \delta | U _ { - } ( V )$ has $| R _ { \delta } | \to 2 | z _ { j } - z _ { k } | > 0$ (its phase need not converge). Pick an atom ℓ of the − class; the two cross edges $B _ { j \ell } , B _ { k \ell }$ average, in the U-coordinates, to $\begin{array} { r } { \left( \frac { z _ { j } + z _ { k } } { 2 } , \ z _ { \ell } \right) + o ( 1 ) } \end{array}$ , whose first coordinate has modulus $\textstyle { \frac { A } { 2 } }$ . Among the three equilateral vertices $z _ { \ell }$ one always satisfies $\Re ( \bar { z } _ { \ell } R _ { \delta } ) \leq - \textstyle { \frac { 1 } { 2 } } A | R _ { \delta } |$ (some vertex makes an angle $\leq \pi / 3$ with $- R _ { \delta } )$ Give the same-class edge the convex weight $s _ { \delta } = c _ { \delta } | \delta |$ with $c _ { \delta } = - \Re ( \bar { z } _ { \ell } R _ { \delta } ) / | R _ { \delta } | ^ { 2 } = O ( 1 )$ , and the two cross edges weights $( 1 - s _ { \delta } ) / 2$ each; this is a legal point of the triangle closure $K _ { \{ j , k , \ell \} }$ (Lemma 7.3). In the limit the first coordinate is unchanged $( s _ { \delta } U _ { + } ( V ) \to 0 )$ and the second is optimized along the ray: $\begin{array} { r } { | z _ { \ell } + c _ { \delta } R _ { \delta } | ^ { 2 } \le \frac { 3 } { 4 } A ^ { 2 } + o ( 1 ) } \end{array}$ . Hence lim sup inf $\begin{array} { r } { C _ { 3 } \le \frac { 1 } { 2 } \big ( \frac { A ^ { 2 } } { 4 } + \frac { 3 A ^ { 2 } } { 4 } \big ) = \frac { A ^ { 2 } } { 2 } } \end{array}$ . If neither class splits, the system stays in the two-direction stratum, where the value is at most $\frac { 5 E } { 8 }$ with the rigid equality characterization of Proposition 7.10. (An explicit paired splitting family even attains limit value $\frac { E } { 8 }$ at three points while keeping its four-point value pinned at $\frac { E } { 4 } .  { \rangle }  { \quad }  { \sqcup }$

## 7.6.4 Two obstructions: what a proof cannot do

Proposition 7.12 $( \mu = 0$ obstruction). A proof of Conjecture 7.8 cannot restrict to selections with $\mu = 0$ . Concretely, for the seventh-roots system $t _ { j } = \zeta ^ { j } , \ z _ { j } = \zeta ^ { 3 j } , \ p _ { j } = \textstyle { \frac { 1 } { 7 } } \ ( \zeta = e ^ { 2 \pi i / 7 } )$ , every q with $\mu = 0$ satisfies $C \geq \frac { 7 / k - 1 } { 2 }$ for support k (a discrete Parseval identity), i.e. $\begin{array} { r } { C \ge \frac { 3 } { 8 } > \frac { 1 } { 4 } } \end{array}$ at $k = 4$ and $C \geq { \frac { 2 } { 3 } } > { \frac { 5 } { 8 } }$ at $k = 3 ;$ yet unconstrained selections achieve $\begin{array} { r } { C = { \frac { 2 } { 7 } } } \end{array}$ at three points (the diference set $\{ 0 , 1 , 3 \} )$ and $\begin{array} { r } { C < \frac { 1 } { 4 } } \end{array}$ at four consecutive points (an exact algebraic computation modulo $x ^ { 3 } + x ^ { 2 } - 2 x - 1 )$ . Similarly, for the regular hexagon third-harmonic system $( t _ { j } = e ^ { \pi i j / 3 }$ $\begin{array} { r } { z _ { j } = ( - 1 ) ^ { j } , p _ { j } = \frac { 1 } { 6 } ) } \end{array}$ all $\mu = 0$ selections of support $\leq 3$ have $C = 1$ , while $\{ 0 , 1 , 2 \}$ at weights 2:3:2 attains the exact three-point optimum $\begin{array} { r } { C = \frac { 1 } { 2 } } \end{array}$

Proposition 7.13 (Averaging obstruction). A proof cannot either rely on selections that keep the original weights on a subset. With $\begin{array} { r } { w _ { e } = p _ { i } p _ { j } d _ { i j } , y _ { e } = w _ { e } B _ { i j } , U = \sum _ { e } w _ { e } ^ { 2 } , Q = \sum _ { e } \lVert y _ { e } \rVert ^ { 2 } } \end{array}$ $\begin{array} { r } { R = 4 \sum _ { i } p _ { i } ^ { 2 } , V = 2 \sum _ { i } p _ { i } ^ { 2 } | z _ { i } | ^ { 2 } } \end{array}$ , the exact averages over all k-subsets S (with $\begin{array} { r } { D _ { S } = \sum _ { e \subset S } w _ { e . } } \end{array}$ $\begin{array} { r } { G _ { S } = \sum _ { e \subset S } y _ { e } } \end{array}$ , and binomials $\begin{array} { r } { A = { \binom { N - 2 } { k - 2 } } , B = { \binom { N - 3 } { k - 3 } } , H = { \binom { N - 4 } { k - 4 } } } \end{array}$ are

$$
\sum _ { | S | = k } ^ { \infty } \| G _ { S } \| ^ { 2 } = ( A - 2 B + H ) Q + ( B - H ) V , \qquad \sum _ { | S | = k } ^ { } D _ { S } ^ { 2 } = H + ( A - 2 B + H ) U + ( B - H ) R .
$$

For the fifth-roots system $\begin{array} { r } { t _ { j } = \zeta _ { 5 } ^ { j } , z _ { j } = t _ { j } ^ { 2 } , p _ { j } = \frac { 1 } { 5 } } \end{array}$ one gets $\begin{array} { r } { U = \frac { 3 } { 2 5 } , R = \frac { 4 } { 5 } , Q = \frac { 1 } { 5 } , V = \frac { 2 } { 5 } } \end{array}$ , so the best original-weight three-point ratio guaranteed by these averages is $\begin{array} { r } { \frac { Q + V } { U + R } = \frac { 1 5 } { 2 3 } > \frac { 5 } { 8 } . } \end{array}$ reweighting inside the selected triangle (the freedom quantified by Lemma 7.3) is essential.

Proof of Propositions 7.12 and 7.13. For the heptagon, $\mu = m _ { 1 } , \lambda = m _ { 2 } , \nu = m _ { 3 }$ where $m _ { r } =$ $\textstyle \sum _ { j } q _ { j } \zeta ^ { r j }$ , and Parseval gives $\begin{array} { r } { 1 + 2 \sum _ { r = 1 } ^ { 3 } | m _ { r } | ^ { 2 } = 7 \sum _ { j } q _ { j } ^ { 2 } \ge 7 / k ; } \end{array}$ with $m _ { 1 } = 0 , C = | m _ { 2 } | ^ { 2 } +$ $| m _ { 3 } | ^ { 2 }$ . The unconstrained witnesses are direct computations (for the four-point one, $\begin{array} { r } { x = 2 \cos { \frac { 2 \pi } { 7 } } } \end{array}$ satisfies $x ^ { 3 } + x ^ { 2 } - 2 x - 1 = 0$ and the claimed sign is that of $3 3 x ^ { 2 } + 7 1 x - 3 6 > 0 )$ . The hexagon classification of $\mu = 0$ supports (antipodal pairs and alternating triangles) and the exact optimum $\frac { 1 } { 2 }$ (via Lemma $7 . 3 ;$ minimal over the three dihedral types of triangles) are finite checks. The subset averages follow by counting occurrences of an edge, an adjacent edge pair, and a disjoint edge pair in k-subsets, together with the cross-term bookkeeping ${ \mathrm { A d j } } = V - 2 Q , { \mathrm { D i s j } } = Q - V$ (from $\textstyle \sum _ { e } y _ { e } = 0 )$ and their scalar analogues; the fifth-roots values use $\| B _ { i j } \| ^ { 2 } = 5 - d _ { i j }$ and the two edge lengths $d _ { \pm } ~ = ~ \frac { 5 \pm \sqrt { 5 } } { 2 }$ . All exact values in this subsection were also verified by independent symbolic computation. □

## 7.6.5 Records, closed classes, and a necessary condition

By deletion arguments (Sherman–Morrison in the frame picture; keeping original weights and deleting one atom i costs $C _ { - i } = p _ { i } | z _ { i } | ^ { 2 } h _ { i } / ( 1 - h _ { i } ) ^ { 2 }$ with $h _ { i } = 2 p _ { i }$ , and harmonic balancing gives min<sub>i</sub> $C _ { - i } \leq 2 E / ( s - 2 ) ^ { 2 }$ for s atoms), Conjecture 7.8 is already proved for: all systems with at most five atoms (four-point version, value $\leq { \frac { 2 E } { 9 } } )$ , all systems with at most four atoms (three-point version, $\leq \frac { E } { 2 } )$ , all two-direction systems (Proposition 7.10), and antipodally paired systems with equal in-pair z (an application of Lemma 5.3 with $M = 2 , k = 2$ to the pair means). By Lemma 7.5, any counterexample to the three-point inequality must satisfy $5 | z _ { i } | ^ { 2 } - 2 L _ { i } \geq 3 E$ for every atom (for distinct directions: $\begin{array} { r } { | z _ { i } | ^ { 2 } \geq \frac { 3 \bar { E } } { 5 } } \end{array}$ for all $i )$ . Exact records: the hexagon threepoint optimum is $\frac { 1 } { 2 }$ , its four-point value is at most ${ \frac { 1 9 } { 1 6 9 } } ;$ the five-atom two-direction worst case is exactly $\textstyle { \frac { 4 } { 7 } } ;$ ; extensive exact searches over random six- and seven-atom systems produced no value above the two-direction records.

## 8 Discussion and open problems

The emerging picture. For weighted selection with the minimum-Frobenius-norm ERM, the profile is now known at both ends for all $d , m \geq 1$ :

$$
F _ { \mathrm { w e i g h t e d } } ( d , m , n ) = \left\{ \begin{array} { l l } { \infty , } & { n < d , } \\ { d + 1 , } & { n = d , } \\ { \mathrm { o p e n } , } & { d < n < ( m + 1 ) d - 1 , } \\ { 1 + \frac { 1 } { d m ^ { 2 } } , } & { n = ( m + 1 ) d - 1 , } \\ { 1 , } & { n \geq ( m + 1 ) d . } \end{array} \right.
$$

The threshold splits conceptually as $( m + 1 ) d = d +$ md: d points to pin a feature basis, md points to cancel the $m \times d$ residual gradient — and the lower-bound instances realize this as d independent copies of weighted mean estimation in $\mathbb { R } ^ { m }$ , one per feature direction, each requiring its full simplex of $m + 1$ points. $\mathrm { A t } \ m = 1$ this collapses to the two mechanisms coinciding, which is why the scalar threshold 2d admits both the Steinitz-type proof of [HMSY25a] and the conic-compression proof given here.

## Open problems.

1. The intermediate curve. Determine $F _ { \mathrm { w e i g h t e d } } ( d , m , n )$ for $d < n < ( m + 1 ) d - 1$ . Already the scalar case is open ([HMSY25b], Question 2). The axial (line-decomposable) instances reduce to an allocation problem over per-line weighted mean-estimation deficits, for which Theorem 6.1 supplies the sharp per-line constants; but the scalar experience suggests that non-axial, higher-dimensional block constructions dominate deeper inside the intermediate range, so we expect the truth to be an allocation over a richer family of circuit structures.

2. The (2, 2) cell (Conjecture 7.8). By Proposition 7.9 this is a question about at most seven atoms on the circle. The obstructions of Section 7.6 show a proof must use selections with $\mu \neq 0$ and must reweight within the selected support; the evidence suggests the two-direction systems are extremal.

3. Unweighted vector selection thresholds. Question 3 asks for the unweighted profile $F ( d , m , n )$ a recent unrefereed preprint $[ \mathrm { P ^ { + } 2 6 } ]$ addresses it, and a rigorous account of the unweighted threshold and its relation to $( m + 1 ) d$ would complete the vector-valued picture.

4. Other learning rules. As emphasized in [HMSY25b], the questions make sense for any ERM selection rule; our certificate and rigidity machinery uses the Frobenius tie-breaking only through Lemma 2.4, and it would be interesting to determine how the threshold depends on the tie-breaking rule.

Verification. All closed-form values in this paper (the instance values $4 3 / 4 0 , 2 3 / 2 0 , 1 3 / 8 .$ 5/4, the profile values, the exact records ${ \frac { 1 } { 2 } } , { \frac { 1 9 } { 1 6 9 } } , { \frac { 4 } { 7 } } , { \frac { 2 } { 7 } } , { \frac { 1 5 } { 2 3 } }$ , and the tightness of the constants in Lemma 5.3 and Theorem 6.1) were additionally verified by exact rational or algebraic symbolic computation, and the lower-bound claims were stress-tested by independent numerical optimization over all supports of the relevant instances.

## References

[Ano26a] Anonymous. Dimension barrier for mean data selection. Unrefereed preprint, GitHub repository andotheror/dimension-barrier-mean-data-selection, 2026.

[Ano26b] Anonymous. Sharp pair selection for mean regression. Unrefereed preprint, Zenodo, 2026. August 22, 2026. DOI: 10.5281/zenodo.22061100.

[Ber09] Dimitri P. Bertsekas. Convex Optimization Theory. Athena Scientific, 2009.

[Dew26] P. Dewasurendra. One short of Steinitz: Weighted regression selection. Unrefereed preprint, Zenodo, 2026. DOI: 10.5281/zenodo.21881535. Scalar case (m = 1) only.

[Dro25] Sergei Drozdov. Egan conjecture holds. Discrete Applied Mathematics, 377:562– 572, 2025. arXiv:2310.10816.

[DW17] Micha l Derezi´nski and Manfred K. Warmuth. Unbiased estimates for linear regression via volume sampling. In Advances in Neural Information Processing Systems 30 (NeurIPS), 2017.

[HMSY25a] Steve Hanneke, Shay Moran, Alexander Shlimovich, and Amir Yehudayof. Data selection for ERMs. In Proceedings of the 38th Annual Conference on Learning Theory (COLT), volume 291 of Proceedings of Machine Learning Research, pages 2634–2665, 2025. arXiv:2504.14572.

[HMSY25b] Steve Hanneke, Shay Moran, Alexander Shlimovich, and Amir Yehudayof. Open problem: Data selection for regression tasks. In Proceedings of the 38th Annual Conference on Learning Theory (COLT), volume 291 of Proceedings of Machine Learning Research, pages 6225–6229, 2025.

[P<sup>+</sup>26] Peng et al. Data selection (unrefereed preprint on question 3 of hanneke et al., COLT 2025). Public repository pipeline-math, file papers/data-selection.pdf, 2026. June 26, 2026. No formal bibliographic data available; cited by repository path.

[Zha26] Guangjian Zhang. Exact risk ratios for weighted data selection in linear regression. arXiv:2608.28007, 2026.