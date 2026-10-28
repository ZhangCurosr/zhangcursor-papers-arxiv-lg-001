# Low-degree polynomials lower bounds for clustering in small dimension

Alexandra Carpentier<sup>1</sup> and Nicolas Verzelen<sup>2</sup>

1 Institut f¨ur Mathematik, Universit¨at Potsdam, Germany.

October 28, 2026

## Abstract

We study the computational limits of clustering n observations drawn from a mixture of K isotropic distributions in $\mathbb { R } ^ { d }$ , in the regime where the number K of groups is much larger than the dimension d. In this regime, tensor-based procedures recover the partition with a polylogarithmic separation between the centers, but they require a sample size that is a large and non-explicit polynomial of dK, whereas the existing low-degree lower bounds only show that n must be larger than $\bar { d K ^ { 2 } }$ . We introduce a mixture of 2K product Bernoulli distributions on $\{ 0 , 1 \} ^ { d }$ , with $K = ( d + 1 ) ^ { m }$ , whose centers are built from a dual BCH code. This construction ensures that the moment tensors of the observations coincide with those of the uniform distribution on $\{ - 1 , 1 \} ^ { d }$ up to the order 2m + 1. For any fixed m, we establish a low-degree polynomial lower bound showing that clustering is hard as long as the separation ∆ satisfies $\Delta ^ { 2 } \ll \overline { { { ( K ^ { 2 } d ^ { 3 / 2 } / n ) ^ { 1 / ( 2 m + 2 ) } } } } \wedge \sqrt { d } .$ up to polylogarithmic factors. In particular, with a polylogarithmic separation, low-degree polynomials require $n \gtrsim K ^ { 2 } d ^ { 3 / 2 }$ observations, up to polylogarithmic factors. Through a Gaussianization argument, this hardness carries over, in the sense of reductions, to isotropic Gaussian mixtures. Our proof relies on a family of polynomials of the observations, indexed by bipartite graphs, which is almost orthonormal. The moments of these polynomials are governed by the number of solutions of polynomial systems over the finite field $\mathbb { F } _ { d + 1 }$ that are induced by the BCH code, and the crux of our analysis is to connect the combinatorial structure of the bipartite graphs with the structure of the solution sets of these systems.

## 1 Introduction

Clustering n observations into K groups is one of the most basic tasks of unsupervised learning. Its most emblematic statistical formulation is the isotropic Gaussian mixture model, which has attracted a lot of attention both in statistics and in theoretical computer science [12, 17, 19, 21, 30, 34–36, 43, 48]. For binary data, the natural counterpart is a mixture of product Bernoulli distributions, also known as a latent class model [11, 20]. In this paper, we are interested in the computational limits of clustering in the regime where the number K of groups is much larger than the dimension d.

## 1.1 Clustering Gaussian and Bernoulli mixtures

Gaussian mixtures. For some unknown centers $\theta _ { 1 } , \dots , \theta _ { K } \in  { \mathbb { R } } ^ { d }$ and an unknown assignment $k ^ { * } : [ n ] $ [K], we observe independent random vectors $Y _ { i } \sim \mathcal { N } ( \theta _ { k ^ { \ast } ( i ) } , I _ { d } ) , i = 1 , \dots , n$ , the groups $( k ^ { * } ) ^ { - 1 } ( \{ k \} )$ having comparable sizes of order $n / K$ . The dificulty of the problem is driven by the separation $\Delta _ { * } : = \mathrm { m i n } _ { k \neq k ^ { \prime } } \Vert \theta _ { k } -$ $\theta _ { k ^ { \prime } } \| _ { 2 }$ , and the goal is to recover the partition of $[ n ]$ induced by $k ^ { * }$ better than a random guess. Setting aside computational constraints, this is possible as soon as $\Delta _ { * } ^ { 2 }$ is large compared to

$$
\Delta _ { \mathrm { s t a t } } ^ { 2 } : = \sqrt { \frac { d K \log ( K ) } { n } } \vee \log ( K ) ,
$$

see [17, 30, 43]. This threshold is achieved by the exact K-means estimator [17], which is however NP-hard to compute in the worst case [1], and even hard to approximate [2]. The central question is therefore whether a polynomial-time procedure can cluster at a separation close to $\Delta _ { \mathrm { s t a t } }$ , or whether a statistical-computational gap arises.

Bernoulli mixtures. In the Bernoulli counterpart, we observe $Y _ { i } ^ { * } \in \{ 0 , 1 \} ^ { d }$ whose coordinates are, conditionally on $k ^ { * }$ , independent with $\mathbb { P } [ Y _ { i j } ^ { * } = 1 ] = ( 1 + \theta _ { k ^ { * } ( i ) , j } ) / 2$ , where now $\theta _ { k } \in [ - 1 , 1 ] ^ { d }$ . The centered observations $Y _ { i } : = 2 Y _ { i } ^ { * } - 1 \in \{ - 1 , 1 \} ^ { d }$ have mean $\theta _ { k ^ { * } ( i ) }$ and independent sub-Gaussian coordinates with variance at most 1, so that the Gaussian and Bernoulli models share the same signal-to-noise ratio, measured again by $\Delta _ { * }$ . In particular, the guarantees of distance-based, spectral and semidefinite procedures, which only rely on sub-Gaussian concentration, are the same in both models [19, 21, 36]. Conversely, as explained later in this section, the Bernoulli model can be mapped to a Gaussian mixture with a separation smaller by a factor of order $\sqrt { \log ( n ) }$ , so that hardness results for the former carry over, in the sense of reductions, to the latter.

## 1.2 Polynomial-time algorithms and low-degree lower bounds

Low-degree polynomials. Since clustering is an average-case problem, worst-case hardness results are not informative and computational barriers are rather assessed within restricted classes of algorithms. In the low-degree polynomial framework [25, 28, 49], one considers estimators that are multivariate polynomials of degree at most D in the observations. For a large class of problems, polynomials of degree $D \asymp \log ( n )$ are as powerful as the best known polynomial-time algorithms, so that the failure of such polynomials is considered strong evidence of computational hardness [28, 49]. This heuristic has been formalized by Hopkins [23] as the low-degree conjecture, which predicts, for hypothesis testing problems with a product null distribution and a permutation-invariant planted distribution, that the failure of polynomials of degree slightly larger than $\log ( n )$ implies the failure of all polynomial-time algorithms, once a small amount of independent noise is added. Following counterexamples to some of its variants [6, 22], this conjecture has recently been disproved in its standard polynomial-time formulation [40]. Nevertheless, the failure of lowdegree polynomials, which match the best known polynomial-time algorithms for a wide range of problems, remains one of the strongest available arguments in favor of a computational barrier. Following [44], lowdegree lower bounds for estimation problems are formulated through the low-degree correlation between a functional x of the hidden partition, for instance $x = \mathbf { 1 } \{ k ^ { * } ( 1 ) = k ^ { * } ( 2 ) \}$ }, and the polynomials of degree at most D in the observations $Y = \left( Y _ { 1 } , \ldots , Y _ { n } \right)$ , namely

$$
{ \mathrm { C o r r } } _ { \leq D } : = \operatorname * { s u p } _ { P \in \mathbb { R } _ { D } [ Y ] : \ \mathbb { E } [ P ^ { 2 } ] \leq 1 } \mathbb { E } [ x P ] ,\tag{1}
$$

where $\mathbb { R } _ { D } [ Y ]$ stands for the space of polynomials with real coeficients and degree at most D in the entries of Y. Low-degree polynomials fail to estimate x when $\mathrm { C o r r } _ { \leq D }$ is not larger than the correlation $\mathbb { E } [ x ] = 1 / K$ achieved by constant polynomials.

High-dimensional regime. When the dimension is large $( n \leq d K )$ , the dificulty of clustering is driven by dimension reduction. Lesieur et al. [32] conjectured that, for a fixed number of groups, clustering is hard below the BBP threshold $[ 3 ] \Delta _ { * } ^ { 2 } \asymp \sqrt { d \dot { K } ^ { 2 } / n }$ . Even, Giraud and Verzelen [17, 18] established that low-degree polynomials fail as long as $\Delta _ { * } ^ { 2 } \ll _ { \mathrm { l o g } } \Delta _ { \mathrm { c o m p , H D } } ^ { 2 }$ , where the subscript in ${ \ll } _ { \mathrm { l o g } }$ indicates that polylogarithmic factors are hidden (see Section 1.6), and

$$
\Delta _ { \mathrm { c o m p , H D } } ^ { 2 } : = 1 \vee \left( \sqrt { d } \wedge \sqrt { \frac { d K ^ { 2 } } { n } } \right) \ .
$$

This bound is matched, up to polylogarithmic factors and when $n \leq d K$ , by a combination of hierarchical clustering – which succeeds as soon as $\Delta _ { * } ^ { 2 } \gtrsim \sqrt { d } \log ( n )$ , so that observations of the same group are closer than observations of distinct groups – and of a spectral projection onto a (K ∧ d)-dimensional subspace followed by hierarchical clustering [18, 48]. Overall, among distance-based and spectral procedures, including SDP relaxations and Lloyd’s algorithm [19, 21, 35, 36, 42], the best available guarantees require $\Delta _ { * } ^ { 2 } \gtrsim _ { \mathrm { l o g } }$ $( \sqrt { d K ^ { 2 } / n } \lor \sqrt { K } ) \land \sqrt { d } [ 1 8 ]$

Moderate-dimensional regime and many groups. When $n \geq d K$ , the picture is much less clear. In this regime, the lower bound $\Delta _ { \mathrm { c o m p , H D } } ^ { 2 }$ of [18] is still valid but it is no longer matched by spectral methods, whose guarantees reduce to $\Delta _ { * } ^ { \mathrm { 2 } } \gtrsim _ { \mathrm { l o g } } \sqrt { K } \wedge \sqrt { d }$ . A diferent stream of literature relies on high-order moment tensors, through the method of moments and tensor decompositions [26] or the sum-of-squares hierarchy [15, 24, 27]. The latter procedures cluster with a separation $\Delta _ { * } \geq K ^ { \gamma }$ , for any fixed $\gamma > 0$ , with a sample size and a running time that are polynomial in d and $K$ , but with an exponent that grows with $1 / \gamma$ Liu and Li [34] proved that a polynomial-time procedure based on iterative tensor projections recovers the partition when $\Delta _ { * } ^ { 2 } \gtrsim \log ( K ) ^ { 1 + \epsilon }$ for some $\epsilon > 0$ , almost matching $\Delta _ { \mathrm { s t a t } } ^ { 2 } ,$ provided that $n \geq ( d K ) ^ { C / \epsilon }$ for some large and non-explicit numerical constant C. On the lower bound side, we recently established in [10] that, when $d \geq K$ , low-degree polynomials fail as long as $\Delta _ { * } ^ { 2 } \ll _ { \log } \frac { K } { n ^ { 1 / 4 } } \wedge \sqrt { K }$ , and that this non-parametric rate is achieved by a non-spectral procedure for the corresponding instances when $d = K$

Statistical query lower bounds [13, 14] also exhibit statistical-computational gaps for clustering, but they rely on non-isotropic mixtures. For such mixtures, the computational barrier is much more severe when $K$ is large: if the covariance matrix is unknown and only bounded by $I _ { d }$ , any SQ procedure requires a complexity of order $d ^ { \Omega ( 1 / \gamma ) }$ for a separation $\Delta _ { * } = K ^ { \gamma } \left[ 1 3 \right]$ , so that, when $K$ grows polynomially with $d ,$ clustering with a polylogarithmic separation is out of reach for polynomial-time SQ procedures, whereas it is achievable in polynomial time for isotropic mixtures [34].

The regime where the number K of groups is much larger than the dimension d remains largely open. There, $\sqrt { K } \geq \sqrt { d }$ , so that distance-based and spectral procedures all require $\Delta _ { \ast } ^ { 2 } \gtrsim _ { \mathrm { l o g } } \sqrt { d } .$ , while, with a polylogarithmic separation, the only available procedures are tensor-based and require $n ~ \geq ~ ( d K ) ^ { c }$ for a large c. Conversely, the only low-degree lower bound in this regime is $\Delta _ { \mathrm { c o m p , H D } } ^ { 2 }$ from [18], which, for a polylogarithmic separation, only states that $n \gtrsim _ { \mathrm { l o g } } d K ^ { 2 }$ . This raises the following two questions:

• When the number K of groups is large compared to the dimension $d ,$ what is the minimal separation $\Delta _ { * } ,$ , as a function of n, K and d, for which clustering is possible in polynomial time?

• More specifically, in this regime, how large must the sample size n be compared to K and d for clustering in polynomial time with only a polylogarithmic separation?

## 1.3 Our contribution

Our results provide a partial answer to these questions on the lower bound side. For any fixed integer $m \geq 1$ we exhibit hard instances with $K = ( d + 1 ) ^ { m }$ groups, for which we establish a lower bound, as a function of n, K and d, on the separation required by low-degree polynomials. Importantly, this separation is smaller than the $\sqrt { d K ^ { 2 } / n }$ bound that corresponds to the spectral threshold and its scaling with respect to m reflects that higher-order tensors come into play.

We construct a Bernoulli mixture with 2K groups, where $K = ( d + 1 ) ^ { m }$ for an integer $m \geq 1$ and $d = 2 ^ { u } - 1$ , whose centers $\pm X _ { k }$ are, up to a random sign flip and a random permutation of the coordinates, the codewords of a dual BCH code [39, Chapter 9], rescaled by $\Delta / \sqrt { d }$ (Section 2). These centers are almost orthogonal, so that the separation $\Delta _ { * }$ lies between $\Delta$ and 2∆ as soon as $d \geq ( 4 \dot { m } ) ^ { m + 1 }$ (Lemma 2). The key property of this construction is that the moment tensors of the centers, and those of the observations, coincide with those of a Rademacher vector up to the order 2m + 1 (Lemma 1). In other words, the distribution of each observation cannot be distinguished from the uniform distribution on $\{ - 1 , 1 \} ^ { d }$ through its moments of order at most $2 m + 1$ . In particular, its covariance matrix is the identity, so that spectral methods based on the empirical covariance matrix are powerless, and the first moment tensor of the observations which carries information on the centers is of order $2 m + 2 .$ , with entries of order $( \Delta / \sqrt { d } ) ^ { 2 m + 2 }$

Our main result, Theorem 1, states that, for a fixed m, polynomials of degree $D$ polylogarithmic in n do not estimate $x = \mathbf { 1 } \{ k ^ { * } ( 1 ) = k ^ { * } ( 2 ) \}$ better than the trivial estimator $1 / K$ , as long as

$$
\Delta ^ { 2 } \ll _ { \mathrm { l o g } } \Big ( \frac { K ^ { 2 } d ^ { 3 / 2 } } { n } \Big ) ^ { 1 / ( 2 m + 2 ) } \wedge \sqrt { d } \ ,
$$

where ${ \ll } _ { \mathrm { l o g } }$ now also hides constants depending on m. Since $K = ( d + 1 ) ^ { m }$ , the first term is of order $d / ( n { \sqrt { d } } ) ^ { 1 / ( 2 m + 2 ) }$ , which is the form appearing in Condition $( \mathcal { H } _ { s i g n a l } )$ . Besides, since $\Delta \le \Delta _ { * } \le 2 \Delta$ , the same condition holds with $\Delta _ { * }$ in place of $\Delta .$ . In particular, with a polylogarithmic separation, low-degree polynomials require at least n $\gtrsim _ { \mathrm { l o g } } K ^ { 2 } d ^ { 3 / 2 }$ observations. This improves by a factor $\sqrt { d }$ over the bound $d K ^ { 2 }$ implied by [18], and, more generally, Theorem 1 is stronger than the bound of [18] as soon as $\Delta ^ { 4 m } \ll _ { \mathrm { l o g } } \sqrt { d } .$ To the best of our knowledge, this is the first computational lower bound showing that, when $K \gg d ,$ clustering with a polylogarithmic separation requires a sample size that is polynomially larger than $d K ^ { 2 }$ For $m = 1$ , that is $K = d + 1$ , the number of groups is of the order of the dimension, a regime which is essentially covered by [10] with a stronger bound, of order $K ^ { 4 } / \Delta ^ { 8 }$ instead of $K ^ { 2 } d ^ { 3 / 2 } / \Delta ^ { 8 } ;$ the novelty of Theorem 1 therefore lies in the regime $m \geq 2$ , that is $K \geq d ^ { 2 }$

Let us compare this bound with the following heuristic. Since the moment tensors of order $2 m + 2$ of the observations carry the first non-trivial information, it is natural to estimate their entries, whose nonzero values are of order $( \bar { \Delta } / \sqrt { d } ) ^ { 2 m + 2 }$ , by empirical averages over the n observations. Detecting a single nonzero entry then requires $\bar { \sqrt { n } } ( \Delta / \sqrt { d } ) ^ { 2 m + 2 } \gtrsim 1$ , that is $n \stackrel { \smile } { \sim } K ^ { 2 } d ^ { 2 } / \Delta ^ { 4 ( m + 1 ) }$ , which is larger than our lower bound by a factor ${ \sqrt { d } } .$ . Whether this remaining gap of order $\sqrt { d }$ can be closed, either by aggregating many entries of the empirical tensor or by improving the lower bound, is left open.

From a technical perspective, we follow the approach of [7, 8, 10], which consists in building a family of polynomials that is almost orthonormal under the distribution of the observations, and which is indexed here by bipartite graphs. As the polynomials only involve the algebraic structure of the BCH code through the number of solutions of polynomial systems over the finite field $\mathbb { F } _ { d + 1 }$ (Lemma 3), the crux of our analysis is twofold. First, we relate the combinatorial properties of the graphs to the moment tensors of the centers. Since these tensors coincide with those of a Rademacher vector up to the order $2 m + 1$ , the systems associated with many graphs have no solution (Lemmas 9 and 10). Second, we bound the number of solutions of the remaining systems through B´ezout-type bounds and a lower bound on the rank of their Jacobian, in terms of the numbers of nodes, edges and connected components of the graph (Lemma 11 and Section B).

From Bernoulli to Gaussian mixtures. Although we consider a mixture of Bernoulli distributions, our setting can be turned into a Gaussian mixture model with almost the same separation, using the Gaussianization techniques of $\left[ 4 , 5 , 3 8 \right]$ . This reduction only changes the separation threshold by a factor of order $\sqrt { \log ( n ) }$

## 1.4 Other related works

Proof techniques for low-degree lower bounds. For detection problems, low-degree lower bounds amount to bounding the norm of the low-degree projection of the likelihood ratio, which is explicit in many models [25, 28]. No such closed form is available for the low-degree correlation (1). Schramm and Wein [44] bound $\mathrm { C o r r } { \scriptstyle \leq _ { D } }$ in terms of joint cumulants of x and of the entries of Y. This versatile approach has been applied, among others, to submatrix estimation [44], graphon estimation [37] and Gaussian mixture clustering [17], but it may lead to involved computations and to thresholds that are only sharp up to polylogarithmic factors. It has been simplified and sharpened in latent variable models by conditioning on the latent variables [18], and generalized by Sohn and Wein [46], who capture sharp thresholds such as the BBP transition in the spiked Wigner model and the Kesten–Stigum threshold in the stochastic block model. A diferent strategy, used by Montanari and Wein [41] in rank-one matrix estimation and systematized in [7], consists in building a family of polynomials which is almost orthonormal under the distribution of the observations, so that $\mathrm { C o r r } _ { \leq D } ^ { 2 }$ is bounded, up to a factor close to one, by the sum of the squared correlations between x and the elements of this family. It has been applied to stochastic block models [8], clustering [10] and ranking [9], and our proof follows this strategy. Finally, Li [33] recently proposed a largely model-independent route, which turns any eficient recovery procedure into a test through a crossvalidation argument, and then relies on low-degree bounds for this testing problem. The resulting lower bounds are however conditional on the low-degree conjecture for detection problems, whereas the above strategies, including ours, provide unconditional bounds on $\mathrm { C o r r } { \le } D$

## 1.5 Organization of the manuscript

In Section 2, we introduce the model, we state the two key properties of the construction, namely the matching of the moment tensors (Lemma 1) and the separation of the centers (Lemma 2), and we state our main result. Section 3 describes the BCH construction and states its algebraic properties. All these

properties are proved in Section F. In Section 4, we discuss possible extensions. The proof of Theorem 1 is organized in Section 5, and its main steps are proved in Section 6 and in the appendix.

## 1.6 Notation

Sets and numbers. For a positive integer b, we write $[ b ] : = \{ 1 , \ldots , b \}$ and, for two integers $a \ \leq \ b .$ $[ a : b ] : = \{ a , a + 1 , \ldots , b \}$ . The cardinality of a finite set A is denoted by $| A |$ . For an event or an assertion $\mathcal { E } .$ $\mathbf { 1 } \{ \mathcal { E } \}$ is equal to 1 if E holds and to 0 otherwise. Given two sets A and B, we write A⊕ $B : = ( A \backslash B ) \cup ( B \backslash A )$ for their symmetric diference and, given finitely many sets $A _ { 1 } , \ldots , A _ { r }$ , we write $\oplus _ { i \le r } A _ { i } : = A _ { 1 } \oplus \cdots \oplus A _ { r }$ for the set of elements which belong to an odd number of the $A _ { i } \mathrm { ^ { * } s }$ . For two real numbers a and $b ,$ we write $a \vee b : = \operatorname* { m a x } ( a , b )$ and a $\cdot \wedge b : = \operatorname* { m i n } ( a , b )$ , and log stands for the natural logarithm.

Comparison of quantities. For two positive quantities a and b depending on the parameters $n , d , K$ and $\Delta$ of the problem, we write $a \lesssim b ,$ or equivalently $b \gtrsim a$ , when $a \leq C b$ for some constant $C > 0$ which does not depend on these parameters, and $a \asymp b$ when $a \lesssim b$ and $b \lesssim a$ . The informal notation $a \ll b$ means that $a / b$ is small, and the subscript in $\lesssim \mathrm { l o g } , \gtrsim \mathrm { l o g }$ and ${ \ll } _ { \mathrm { l o g } }$ indicates that polylogarithmic factors and constants depending on m are hidden as well. Finally, $O ( \cdot )$ stands for the usual Landau notation.

Vectors and tensors. For $x , y \in \mathbb { R } ^ { d }$ , we write $\textstyle \langle x , y \rangle : = \sum _ { j < d } x _ { j } y _ { j }$ and $\| x \| _ { 2 } : = \langle x , x \rangle ^ { 1 / 2 }$ for the Euclidean inner product and norm, and $I _ { d }$ stands for the identity matrix of size d. For a positive integer l, the l-th tensor power of $x \in \mathbb { R } ^ { d }$ is the tensor $x ^ { \otimes l }$ of order l whose entries are $\begin{array} { r } { ( x ^ { \otimes l } ) _ { j _ { 1 } , \dots , j _ { l } } : = \prod _ { s \leq l } x _ { j _ { s } } } \end{array}$ , for $j _ { 1 } , \ldots , j _ { l } \in [ d ]$ For a random vector $Z \in \mathbb { R } ^ { d }$ , the moment tensor $\mathbb { E } [ Z ^ { \otimes l } ]$ of order l is defined entrywise.

## 2 Model and main results

Let u and m denote two positive integers. Define $d = 2 ^ { u } - 1$ and $K = ( d + 1 ) ^ { m }$ . Our construction relies on a family $( \mu _ { k } ) _ { k = 1 , \dots , K }$ of vectors in $\{ - 1 , 1 \} ^ { d }$ which, up to renormalization, correspond to the codewords of a dual BCH code of length d. As their precise definition requires some algebra over finite fields, we postpone it to Section 3, see (9). In this section, we introduce the mixture model built from the $\mu _ { k } \mathrm { { ' s } }$ and we only state, in Lemma 1 below, the key properties of this construction that drive our lower bound.

Let σ denote a uniformly random permutation on [d] and let $a _ { j } , j = 1 , \dotsc , d ,$ denote i.i.d. Rademacher random variables, independent of $\sigma .$ For $\ell = 1 , \ldots , K$ , we define the vector $X _ { \ell } \in \mathbb { R } ^ { d }$ by

$$
X _ { \ell , j } = \frac { \Delta } { \sqrt { d } } a _ { j } \mu _ { \ell , \sigma ( j ) } , \mathrm { w h e r e } 0 < \Delta \leq \sqrt { d } .\tag{2}
$$

This range of $\Delta$ is what makes $( 1 + z _ { i } X _ { k ^ { * } ( i ) , j } ) / 2$ a valid probability below, since $| \mu _ { \ell , j } | = 1 ;$ ; it is in particular implied by Condition $( \mathcal { H } _ { s i g n a l } )$

Let the random function $k ^ { * } : [ n ] \to [ K ]$ be such that $k ^ { * } ( 1 ) , \ldots , k ^ { * } ( n )$ are i.i.d. $\operatorname { U n i f } ( [ K ] )$ and $z _ { i } ,$ for $i = 1 , \ldots , n$ , denote i.i.d. Rademacher random variables, with $\sigma , ( a _ { j } ) , k ^ { * } ( \cdot )$ , and $\left( z _ { i } \right)$ mutually independent. Then, for $i = 1 , \ldots , n$ , we observe the random vector $Y _ { i } ^ { * } \in \{ 0 , 1 \} ^ { \bar { d } }$ , which, conditionally on $\sigma , ( a _ { j } ) , k ^ { * } ( . )$ and (z<sub>i</sub>), follows a product distribution such that

$$
Y _ { i j } ^ { * } \sim { \mathcal { B } } { \Big ( } { \frac { 1 + z _ { i } X _ { k ^ { * } ( i ) , j } } { 2 } } { \Big ) } ~ .
$$

Hence, given $\sigma$ and $( a _ { j } )$ , the $( Y _ { i } ^ { * } )$ are independent observations from a mixture distribution with 2K groups with uniform weights. Our goal is to estimate the functional

$$
x = { \bf 1 } \{ k ^ { * } ( 1 ) = k ^ { * } ( 2 ) \} .\tag{3}
$$

For that we define

$$
Y _ { i j } = 2 Y _ { i j } ^ { * } - 1\tag{4}
$$

and we write $Y _ { i } = ( Y _ { i 1 } , \dots , Y _ { i d } ) \in \{ - 1 , 1 \} ^ { d }$ . Conditionally on $\sigma , ( a _ { j } ) , k ^ { * } ( i )$ and $z _ { i } ,$ the coordinates of $Y _ { i }$ are independent and $\mathbb { E } [ Y _ { i j } \mid \sigma , ( a _ { j } ) , k ^ { * } ( i ) , z _ { i } ] = z _ { i } X _ { k ^ { * } ( i ) , j }$ , so that the 2K centers of the mixture are the vectors $\pm X _ { k } , k \le K$ . We now state the key properties of the BCH construction of the $\mu _ { k } { } ^ { \mathrm { ' } } \mathrm { s } { } \mathrm { : }$ up to the order $2 m + 1$ the moment tensors of these centers, as well as those of the observations, are exactly those of a Rademacher random vector. This lemma is a variant of Lemma 4, which is stated in Section $^ { 3 , }$ where the $\mu _ { k } \mathrm { { ' s } }$ are defined; both lemmas are proved in Section F.

Lemma 1. Let $U = \left( U _ { 1 } , \ldots , U _ { d } \right)$ be a vector of d i.i.d. Rademacher random variables. Recall that, for any positive integer l and any $\begin{array} { r } { j _ { 1 } , \ldots , j _ { l } \in [ d ] , \mathbb { E } [ U ^ { \otimes l } ] _ { j _ { 1 } , \ldots , j _ { l } } = \mathbb { E } [ \prod _ { s < l } U _ { j _ { s } } ] } \end{array}$ is equal to 1 if each value appears an even number of times in $( j _ { 1 } , \dots , j _ { l } )$ , and to 0 otherwise. Then, for every $i \in [ n ]$ and every positive integer $l \leq 2 m + 1$ , the following holds.

(i) The order-l tensor of the centers is that of a rescaled Rademacher vector:

$$
\mathbb { E } \Big [ \big ( z _ { i } X _ { k ^ { * } ( i ) } \big ) ^ { \otimes l } \Big | \sigma , ( a _ { j } ) \Big ] = \frac { 1 } { 2 K } \sum _ { k = 1 } ^ { K } \sum _ { \varepsilon \in \{ - 1 , 1 \} } \big ( \varepsilon X _ { k } \big ) ^ { \otimes l } = \Big ( \frac { \Delta } { \sqrt { d } } \Big ) ^ { l } \mathbb { E } \big [ U ^ { \otimes l } \big ] \ .\tag{5}
$$

(ii) The order-l tensor of the observation $Y _ { i }$ is that of a Rademacher vector:

$$
\mathbb { E } \left[ Y _ { i } ^ { \otimes l } \left| ~ \sigma , ( a _ { j } ) \right. \right] = \mathbb { E } \left[ U ^ { \otimes l } \right] ~ .\tag{6}
$$

As the right-hand sides of (5) and (6) do not depend on σ and $( a _ { j } )$ , these two properties also hold unconditionally.

We now show that the vectors $\mu _ { k }$ are almost orthogonal, so that the 2K centers $\pm X _ { k }$ of the mixture are well separated, as long as m is small compared to log(d)/ log log(d). This lemma is also proved in Section F.

Lemma 2. Assume that d $\geq 2 m$ and let

$$
\eta _ { m } : = \left( \frac { 4 m ^ { m + 1 } } { d } \right) ^ { 1 / ( 2 m ) } .
$$

Then, for all $k \neq k ^ { \prime } \in [ K ]$ , we have $| \langle \mu _ { k } , \mu _ { k ^ { \prime } } \rangle | \leq \eta _ { m } d .$ . Consequently,

$$
2 ( 1 - \eta _ { m } ) d \le \| \mu _ { k } - \mu _ { k ^ { \prime } } \| _ { 2 } ^ { 2 } \le 2 ( 1 + \eta _ { m } ) d \ , \qquad a n d \qquad \| \mu _ { k } + \mu _ { k ^ { \prime } } \| _ { 2 } ^ { 2 } \ge 2 ( 1 - \eta _ { m } ) d \ .
$$

In particular, $i f d \ge ( 4 m ) ^ { m + 1 }$ , then $\eta _ { m } \leq 1 / 2$ , so that mi $\mathrm { n } _ { k \neq k ^ { \prime } } \| \mu _ { k } - \mu _ { k ^ { \prime } } \| _ { 2 } \geq \sqrt { d }$ , and the 2K centers $\pm X _ { k }$ $k \leq K$ , are pairwise at Euclidean distance at least $\Delta$

Since $K = ( d + 1 ) ^ { m }$ , the condition $d \geq ( 4 m ) ^ { m + 1 }$ holds as soon as $d \geq e ^ { e ^ { 2 } }$ and $K \leq d ^ { \log ( d ) / ( 2 \log \log ( d ) ) }$ , so that the number K of groups can be much larger than any fixed power of the dimension d.

Lemma 2 ensures that the mixture is well separated. Write $\Delta ,$ <sub>∗</sub> for its separation, that is the minimal Euclidean distance between two distinct centers $\varepsilon X _ { k }$ and $\varepsilon ^ { \prime } X _ { k ^ { \prime } }$ , with $\varepsilon , \varepsilon ^ { \prime } \in \{ - 1 , 1 \}$ and $k , k ^ { \prime } \in [ K ]$ . Since $\| X _ { k } \| _ { 2 } = \Delta$ and $\begin{array} { r } { \| \varepsilon X _ { k } - \varepsilon ^ { \prime } X _ { k ^ { \prime } } \| _ { 2 } ^ { 2 } = \frac { \Delta ^ { 2 } } { d } \big ( 2 d - 2 \varepsilon \varepsilon ^ { \prime } \langle \mu _ { k } , \mu _ { k ^ { \prime } } \rangle \big ) } \end{array}$ for k $\neq k ^ { \prime }$ , Lemma 2 gives

$$
\sqrt { 2 ( 1 - \eta _ { m } ) } \Delta \le \Delta _ { * } \le \sqrt { 2 ( 1 + \eta _ { m } ) } \Delta \ .
$$

In particular, $\Delta \le \Delta _ { * } \le 2 \Delta$ as soon as $d \geq ( 4 m ) ^ { m + 1 }$ , and $\Delta _ { * } = \sqrt { 2 } \Delta \left( 1 + o ( 1 ) \right)$ when m is not too large, namely when $m = o \big ( \log ( d ) / \log \log ( d ) \big )$ . Hence, up to numerical constants, the separation of the mixture is of the order of $\Delta .$ , and Theorem 1 below can be read in terms of $\Delta _ { * }$

In the following, we bound the low-degree correlation (1) between the functional x defined in (3) and the polynomials of degree at most $D$ in $Y = ( Y _ { i j } ) _ { i \in [ n ] , j \in [ d ] }$ , namely

$$
\mathrm { C o r r } _ { \leq D } = \operatorname* { s u p } _ { P \in \mathbb { R } _ { D } \left[ Y \right] : \ \mathbb { E } \left[ P ^ { 2 } \right] \leq 1 } \mathbb { E } [ x P ] .
$$

Since the constant polynomial $P = 1$ is admissible, we always have Corr $\leq D \geq \mathbb { E } [ x ] = 1 / K$

Assume that

$$
\rho : = \Big [ ( 3 2 D ^ { 3 } d ^ { 1 / 4 } n ^ { 1 / 2 } ) ^ { 1 / ( 2 m + 2 ) } \Big ( { \frac { 4 m \Delta } { \sqrt { d } } } \Big ) \Big ] \vee \Big [ ( 3 2 D ^ { 3 } ) { \frac { 4 m \Delta } { d ^ { 1 / 4 } } } \Big ] \leq D ^ { - 1 6 c _ { s } } \leq 1 \ .\tag{H<sub>signal</sub>}
$$

Under this condition, we bound the low-degree correlation as follows.

Theorem 1. Let D be a positive integer, and assume that $2 ^ { 8 } D ^ { 4 } \leq d , n \geq d ;$ and that Condition $( \mathcal { H } _ { s i g n a l } )$ holds for some $c _ { \mathrm { s } } \geq 1 0$ . Then, we have

$$
\mathrm { C o r r } _ { \le D } ^ { 2 } \le \frac { 1 } { K ^ { 2 } } \left( 1 + \frac { 1 6 } { D ^ { c _ { \mathrm { s } } / 2 } } + \frac { 1 6 D ^ { 2 } } { d } \right) .
$$

Since $\mathrm { C o r r } _ { \le D } \ge 1 / K$ , Theorem 1 entails that $\mathrm { C o r r } _ { \le D } = ( 1 + o ( 1 ) ) / K$ as soon as $D  \infty$ . Hence, no polynomial of degree at most D predicts whether the first two observations belong to the same group significantly better than the constant $1 / K$ , which ignores the data: low-degree polynomials do not improve over a random guess. Regarding the rates, for D of order $\log ( n )$ and since $K = ( d + 1 ) ^ { m }$ , Condition $( \mathcal { H } _ { s i g n a l } )$ amounts to

$$
\Delta ^ { 2 } \ll _ { \mathrm { l o g } } \frac { d } { ( n \sqrt { d } ) ^ { 1 / ( 2 m + 2 ) } } \asymp \Big ( \frac { K ^ { 2 } d ^ { 3 / 2 } } { n } \Big ) ^ { 1 / ( 2 m + 2 ) } ~ \mathrm { a n d } ~ \Delta ^ { 2 } \ll _ { \mathrm { l o g } } \sqrt { d } ~ ,
$$

where ${ \ll } _ { \mathrm { l o g } }$ hides polylogarithmic factors and constants depending on m. The second condition excludes the regime where distance-based procedures succeed. The first one entails that, with a polylogarithmic separation, low-degree polynomials require n $\gtrsim _ { \mathrm { l o g } } K ^ { 2 } d ^ { 3 / 2 }$ , which improves by a factor $\sqrt { d }$ over the bound $d K ^ { 2 }$ of [18], and is smaller by a factor $\sqrt { d }$ than the sample size at which the empirical tensor of order $2 m + 2$ becomes informative (Section 4).

## 3 Definition and Properties of the $\mu _ { k } { ' }$ s

We detail here the construction of the means $\mu _ { \ell } \mathrm { { ^ { s } } }$ used in our model. Our construction corresponds, up to renormalization, to codewords of the dual of the binary Bose–Chaudhuri–Hocquenghem (BCH) code of length d and designed distance $2 m + 1$ , see $\mathrm { e . g . }$ [39] for a textbook account. For the sake of completeness, we nevertheless provide a self-contained definition of these vectors, together with self-contained proofs of the properties used in this work, without relying on coding theory. Recall that u and m denote two positive integers. Write $d = 2 ^ { u } - 1$ and $K = ( d + 1 ) ^ { m }$

## 3.1 Definition

First, we recall one construction of the finite field $\mathbb { F } _ { 2 ^ { u } }$ . We consider $\mathbb { F } _ { 2 } [ T ]$ , namely the set of polynomials with binary coeficients, and we then quotient this set by a primitive polynomial $^ { 1 } f ( T )$ of degree u. Since f is irreducible, the quotient $\mathbb { F } _ { 2 } [ T ] / f ( T )$ is a field with $2 ^ { u }$ elements, and we henceforth identify it with the finite field $\mathbb { F } _ { 2 ^ { u } }$ . The underlying set of $\mathbb { F } _ { 2 ^ { u } }$ is furthermore in bijection with the set of integers $\{ 0 , \ldots , 2 ^ { u } - 1 \}$ and with the product set $\mathbb { F } _ { 2 } ^ { u } { \mathrm { : } }$ to a polynomial of $\mathbb { F } _ { 2 } [ T ] / f ( T )$ we associate, on the one hand, the vector of its ordered coeficients, which is an element of F<sup>u</sup>, and on the other hand its corresponding value when evaluated in base 2, which is an element of $\{ 0 , \ldots , 2 ^ { u } - 1 \}$ . Throughout the paper, we use the resulting bijection to identify the set of integers $\{ 0 , \ldots , 2 ^ { u } - 1 \}$ with the field $\mathbb { F } _ { 2 ^ { u } }$ . This identification only concerns the coordinate labels $j \in [ d ] = \{ 1 , \ldots , d \}$ , which are thereby identified with the nonzero elements of $\mathbb { F } _ { 2 ^ { u } }$ , and, in Section ${ \mathrm { B } } ,$ the variables $x _ { w } \colon$ whenever we add or multiply such elements – and, in particular, whenever we raise such an element to some power – the operations are those of the field $\mathbb { F } _ { 2 ^ { u } }$ , and not the operations of N. All the other arithmetic operations, such as k − 1, d + 1 or $2 a - 1$ , are the usual operations on integers.

Let us emphasize that the set $\mathbb { F } _ { 2 } ^ { u }$ plays a diferent role: it is only used as an additive group. Indeed, the bijection $\mathbb { F } _ { 2 ^ { u } } \to \mathbb { F } _ { 2 } ^ { u }$ described above is an isomorphism of $\mathbb { F } _ { 2 } \mathrm { - v e c t o r }$ spaces – hence, in particular, an isomorphism of additive groups, the addition on $\mathbb { F } _ { 2 } ^ { u }$ being the coordinatewise addition modulo 2 – but it is not a field isomorphism, and the product set F<sup>u</sup> endowed with the coordinatewise multiplication is not even a field as soon as $u \geq 2 .$ , since $\mathrm { e . g . ~ } ( 1 , 0 , \ldots , 0 )$ and $( 0 , 1 , 0 , \ldots , 0 )$ are nonzero elements with product 0. Accordingly, only the additive structure of $\mathbb { F } _ { 2 } ^ { u }$ is ever used below, all multiplicative operations being performed in $\mathbb { F } _ { 2 ^ { u } }$

In what follows, we write

$$
\psi : \{ 0 , \dots , 2 ^ { u } - 1 \} \to \mathbb { F } _ { 2 } ^ { u } ,
$$

for the bijection that associates to any element in $\{ 0 , \ldots , 2 ^ { u } - 1 \}$ its representation in base 2, as an element of $\mathbb { F } _ { 2 } ^ { u }$ . In other words, upon identifying $\{ 0 , \ldots , 2 ^ { u } - 1 \}$ with $\mathbb { F } _ { 2 ^ { u } }$ as explained above, the map ψ is exactly the isomorphism of $\mathbb { F } _ { 2 ^ { - } }$ vector spaces from $\mathbb { F } _ { 2 ^ { u } }$ onto $\mathbb { F } _ { 2 } ^ { u }$ described above. In particular, ψ is additive: $\ \psi ( v + v ^ { \prime } ) =$ $\psi ( v ) + \psi ( v ^ { \prime } )$ for all $v , v ^ { \prime } \in \mathbb { F } _ { 2 ^ { u } }$ , where the left-hand sum is taken in $\mathbb { F } _ { 2 ^ { u } }$ and the right-hand one coordinatewise modulo 2; and $\boldsymbol { \psi } ( \boldsymbol { v } ) = 0$ if and only if $v = 0$ . We shall only use these additive properties of $\psi .$ . We also write

$$
\bar { \psi } : \{ 0 , \ldots , 2 ^ { u m } - 1 \}  \{ 0 , 1 \} ^ { m \times u } ,
$$

for the bijection that associates to any $k \in \{ 0 , \ldots , 2 ^ { u m } - 1 \}$ its representation in base 2, but transformed from its vector representation in dimension um to the associated matrix representation of dimension $m \times u$ More precisely, we consider the base-2 representation as a vector, and we arrange its um coordinates into m rows of size u (row by row).

Let $v \in \{ 0 , \ldots , 2 ^ { u } - 1 \}$ , which we identify, as above, with an element of the finite field $\mathbb { F } _ { 2 ^ { u } } = \mathbb { F } _ { d + 1 }$ . We write for any integer $l \geq 1$

$$
\phi ^ { ( l ) } ( v ) = \psi \left( v ^ { l } \right) \in \mathbb { F } _ { 2 } ^ { u } \ ,\tag{7}
$$

where $v ^ { l }$ stands for the l-th power of v in the field $\mathbb { F } _ { 2 ^ { u } }$ , and where $\psi$ then transforms this field element into the associated binary code in $\mathbb { F } _ { 2 } ^ { u }$ . Let us stress that the power operation in (7) is the one of the field $\mathbb { F } _ { 2 ^ { u } }$ it is not a coordinatewise power of the binary vector $\psi ( v ) -$ which would be meaningless, as $\psi ( v ) ^ { l } = \psi ( v )$ for every $l \geq 1$ when powers are taken coordinatewise in $\{ 0 , 1 \}$ . Note also that $\phi ^ { ( l ) } ( 0 ) \stackrel { - } { = } \psi ( 0 ) = 0$ for every $l \geq 1$

Define for $v \in \{ 0 , \ldots , 2 ^ { u } - 1 \}$

$$
H _ { v } ( R ) = \prod _ { a = 1 } ^ { m } \prod _ { b = 1 } ^ { u } ( R _ { b } ^ { ( a ) } ) ^ { [ \phi ^ { ( 2 a - 1 ) } ( v ) ] _ { b } } ,\tag{8}
$$

where $R = ( R _ { b } ^ { ( a ) } ) _ { b \leq u , a \leq m }$ is such that $R _ { b } ^ { ( a ) } \in \{ - 1 , 1 \}$

We write for any ${ \bar { k \in } } \left\{ 1 , \ldots , 2 ^ { u m } \right\} = \left\{ 1 , \ldots , ( d + 1 ) ^ { m } \right\} = \{ 1 , \ldots , K \}$ and any $j \in \{ 1 , \ldots , 2 ^ { u } - 1 \} =$ $\{ 1 , \ldots , d \} - \mathrm { s o }$ that j ranges over the d nonzero elements of $\mathbb { F } _ { 2 ^ { u } } .$ , excluding the element 0 for which $\phi ^ { ( l ) } ( 0 ) = 0$ would otherwise yield a trivial (constant) coordinate:

$$
\mu _ { k , j } = H _ { j } \left( \left( 2 ( \bar { \psi } ( k - 1 ) ) _ { a , b } - 1 \right) _ { a \leq m , b \leq u } \right) \mathrm { ~ . ~ }\tag{9}
$$

Intuition. Since $\hat { \psi }$ is a bijection, when k is uniformly distributed on $[ K ]$ , the matrix $R : = \big ( 2 ( \bar { \psi } ( k - 1 ) ) _ { a , b } -$ $1 ) _ { a \leq m , b \leq u }$ has i.i.d. Rademacher entries. By (8) and (9), each coordinate $\mu _ { k , j } = H _ { j } ( R )$ is the product of the entries of R selected by the binary vector $h ( j ) : = \left( \phi ^ { ( 1 ) } ( j ) , \phi ^ { ( 3 ) } ( j ) , \ldots , \phi ^ { ( 2 m - 1 ) } ( j ) \right) \in \mathbb { F } _ { 2 } ^ { m \times u }$ . Hence, given a positive integer l and indices $j _ { 1 } , \ldots , j _ { l } \in [ d ]$ , not necessarily distinct, the product $\textstyle \prod _ { i = 1 } ^ { l } \mu _ { k , j _ { i } }$ is again a product of entries of R, whose expectation with respect to $k \sim \operatorname { U n i f } ( [ K ] )$ , namely $\begin{array} { r } { \frac { 1 } { K } \sum _ { k \leq K } \prod _ { i = 1 } ^ { l } \mu _ { k , j _ { i } } = } \end{array}$ $\begin{array} { r } { \left( \frac { 1 } { K } \sum _ { k \le K } \mu _ { k } ^ { \otimes l } \right) _ { j _ { 1 } , \ldots , j _ { l } } } \end{array}$ , is equal to 1 if each entry appears an even number of times, that is if $\begin{array} { r } { \sum _ { i = 1 } ^ { l } h ( j _ { i } ) = } \end{array}$ 0 in $\mathbb { F } _ { 2 } ^ { m \times u }$ , and to 0 otherwise: this is Lemma 3 below. In comparison, for a Rademacher vector $U ,$ $\mathbb { E } [ \prod _ { i = 1 } ^ { l } U _ { j _ { i } } ] = 1$ if and only if each index appears an even number of times in $( j _ { 1 } , \dots , j _ { l } )$ . Therefore, matching the moment tensors of $U$ up to the order 2m amounts to requiring that any 2m of the vectors $h ( 1 ) , \ldots , h ( d )$ are linearly independent over $\mathbb { F } _ { 2 }$ . This is where the choice of the odd powers $v , v ^ { 3 } , \ldots , v ^ { 2 m - 1 }$ comes into play. Identifying the indices $j _ { 1 } , \dotsc , j _ { l } \in [ d ]$ with nonzero elements $v _ { 1 } , \ldots , v _ { l }$ of $\mathbb { F } _ { 2 ^ { u } }$ , the condition $\begin{array} { r } { \sum _ { i = 1 } ^ { l } h ( j _ { i } ) = 0 } \end{array}$ means that the m odd power sums $\textstyle \sum _ { i = 1 } ^ { l } v _ { i } ^ { 2 a - 1 } , a \leq m$ , vanish in $\mathbb { F } _ { 2 ^ { u } }$ . In characteristic $2 ,$ , we hav $\textstyle { \mathrm { : } } ^ { \bar { 2 } } \sum _ { i = 1 } ^ { l } v _ { i } ^ { 2 k } = ( \sum _ { i = 1 } ^ { l } v _ { i } ^ { k } ) ^ { 2 }$ for any $k \geq 1$ , so that the first 2m power sums $\textstyle \sum _ { i = 1 } ^ { l } v _ { i } ^ { k } , k \leq 2 m$ , then vanish as well, which is impossible when $1 \leq l \leq$ 2m and $v _ { 1 } , \ldots , v _ { l }$ are pairwise distinct (Lemma 24 in Section F). This is why m blocks of u bits, that is $K = ( d + 1 ) ^ { m }$ vectors, sufice to match the moment tensors of U up to the order 2m, which is Lemma 4 below.

## 3.2 Properties

We state here the two properties of the BCH construction on which our analysis relies. Lemma 3 expresses the moment tensors of the $\mu _ { k } { \mathrm { ' s } } ,$ of any order, through the polynomial system $\begin{array} { r } { \sum _ { i } \phi ^ { ( 2 a - 1 ) } ( j _ { i } ) = 0 , a \le m , } \end{array}$ whose solutions are studied in Section B, and Lemma 4 states that these tensors coincide with those of a Rademacher vector up to the order 2m. Apart from Lemma 2, which ensures that the mixture is well separated, and from the fact that the $\mu _ { k }$ ’s belong to $\{ - 1 , 1 \} ^ { d }$ , these are the only properties of the $\mu _ { k }$ ’s that we use. Their proofs, as well as the proof of Lemma 1, are postponed to Section F.

Lemma 3. We have for any positive integer l

$$
\Big ( \frac { 1 } { K } \sum _ { k \le K } \mu _ { k } ^ { \otimes l } \Big ) _ { j _ { 1 } , \dots , j _ { l } } = \mathbf { 1 } \Big \{ \forall a \le m : \sum _ { i = 1 } ^ { l } \phi ^ { ( 2 a - 1 ) } ( j _ { i } ) = 0 \Big \} ,
$$

where the sum in the equation in the indicator is taken in $\mathbb { F } _ { 2 } ^ { u }$ - and therefore corresponds to a coordinate-wise sum of the u-dimensional vectors, modulo 2.

Lemma 4. We have for any $l \leq 2 m$ that

$$
\frac { 1 } { K } \sum _ { k \le K } \mu _ { k } ^ { \otimes l } = \mathbb { E } _ { \boldsymbol { U } \sim \boldsymbol { \mathcal { U } } \Big ( \{ - 1 , 1 \} ^ { d } \Big ) } \Big [ \boldsymbol { U } ^ { \otimes l } \Big ] = : \boldsymbol { \mathcal { U } } ^ { ( l ) } ,
$$

where $\mathcal { U } ( \{ - 1 , 1 \} ^ { d } )$ is the uniform distribution over the set $\{ - 1 , 1 \} ^ { d }$ . Note that $\mathcal { U } _ { j _ { 1 } , \dots , j _ { l } } ^ { ( l ) }$ is equal to 1 if and only if there exists a pairing<sup>3</sup> of the vector $( j _ { 1 } , \dots , j _ { l } )$ , and otherwise it is equal to 0. In this case, we say that $( j _ { 1 } , \dots , j _ { l } )$ is paired.

Remark 1. Lemma 4 deals with the tensors $\begin{array} { r } { \frac { 1 } { K } \sum _ { k \leq K } \mu _ { k } ^ { \otimes l } } \end{array}$ up to the order 2m only. The additional order $2 m + 1$ in Lemma 1 is due to the signs $z _ { i } .$ the set of centers $\{ \pm X _ { k } \}$ is symmetric, so that all its odd-order tensors vanish.

## 4 Discussion

Upper bound for the BCH instances. For the instances of Theorem 1, the tensor of order $2 m + 2$ of the observations can be estimated exactly. Indeed, conditionally on σ and $( a _ { j } )$ , Lemma 3 and the conditional independence of the coordinates of $Y _ { i }$ imply that, for pairwise distinct $j _ { 1 } , \ldots , j _ { 2 m + 2 } \in [ d ]$

$$
\mathbb { E } \Big [ \prod _ { s \leq 2 m + 2 } Y _ { i j s } \Big | \sigma , ( a _ { j } ) \Big ] = \Big ( \frac { \Delta } { \sqrt { d } } \Big ) ^ { 2 m + 2 } \prod _ { s \leq 2 m + 2 } a _ { j s } \ \mathbf { 1 } \Big \{ \forall a \leq m : \sum _ { s \leq 2 m + 2 } \phi ^ { ( 2 a - 1 ) } ( \sigma ( j _ { s } ) ) = 0 \Big \} \ .
$$

The empirical averages $\begin{array} { r } { \frac { 1 } { n } \sum _ { i \leq n } \prod _ { s } Y _ { i j } } \end{array}$ fluctuate at the scale $n ^ { - 1 / 2 }$ . By Hoefding’s inequality and a union bound over the at most $d ^ { 2 m + \overline { { 2 } } }$ tuples, thresholding them recovers, with high probability, both the support of this tensor and the signs $\Pi _ { s } a _ { j _ { s } }$ , as soon as

$$
n \gtrsim m \log ( d ) \left( { \frac { d } { \Delta ^ { 2 } } } \right) ^ { 2 m + 2 } \asymp m \log ( d ) { \frac { K ^ { 2 } d ^ { 2 } } { \Delta ^ { 4 ( m + 1 ) } } } .
$$

We conjecture that the centers $\pm X _ { k }$ of the mixture can be recovered in polynomial time from this estimated tensor and that, in turn, the observations can be clustered, for instance by assigning each of them to the center $\pm X _ { k }$ which maximizes its scalar product, as soon as $\Delta ^ { 2 } \gtrsim m \log ( d )$ . For these instances, clustering would then be possible in polynomial time as soon as $n \gtrsim _ { \mathrm { l o g } } K ^ { 2 } d ^ { 2 } / \Delta ^ { 4 ( m + 1 ) }$ and $\Delta ^ { 2 } \gtrsim$ m log(d), which difers from the lower bound of Theorem 1 by a factor $\sqrt { d }$ in n.

Even if this conjecture holds, such a procedure would heavily rely on the algebraic structure of the instances, namely on the knowledge that the centers form a permuted and sign-flipped dual BCH code. It is therefore not clear whether a sample size of order $K ^ { 2 } d ^ { 2 }$ , or $K ^ { 2 } d ^ { 3 / 2 }$ , characterizes the optimal complexity of clustering over all mixtures whose centers are ∆-separated. On the one hand, a procedure that is agnostic to the structure of the centers may require more observations. On the other hand, other configurations of centers may lead to harder instances.

On Gaussian designs. By Lemmas 4 and 1, the uniform distribution on $\{ \mu _ { k } , k \le K \}$ (resp. on $\{ \pm \mu _ { k } , k \le$ $K \}$ has the same moment tensors as a vector U of d i.i.d. Rademacher random variables, up to the order 2m $( \mathrm { r e s p . ~ } 2 m + 1 )$ . To introduce a tentative worst-case instance for Gaussian mixture models, we would rather use a design whose moment tensors match those of the standard normal distribution, that is a set $\{ \xi _ { 1 } , \ldots , \xi _ { K } \} \subset \mathbb { R } ^ { \bar { d } }$ such that

$$
\frac { 1 } { K } \sum _ { k \le K } \xi _ { k } ^ { \otimes l } = \mathbb { E } \bigl [ G ^ { \otimes l } \bigr ] \ , \qquad \mathrm { f o r ~ a l l } \ l \le 2 m + 1 \ , \qquad \mathrm { w h e r e ~ } G \sim \mathcal { N } ( 0 , I _ { d } ) \ .
$$

For general $m _ { ; }$ , the minimal size of such Gaussian designs remains largely unknown [16], even when the uniform weights $1 / K$ are replaced by positive weights, which amounts to a mixture with unequal proportions. Such a design has at least $\hat { \binom { d + m } { m } } \geq \hat { d ^ { m } } / m !$ points $[ 4 7 ]$ , which is of the same order as $K = ( d + 1 ) ^ { m }$ , whereas the known constructions with positive weights have $\dot { O ( d ^ { 2 m - 1 } ) }$ points [29]. This is the reason why we focused our interest in this manuscript on the Bernoulli case.

Towards a direct low-degree proof in the Gaussian case. The reduction described in the introduction transfers Theorem 1 to Gaussian mixtures at the price of a factor $\sqrt { \log ( n ) }$ in the separation, and the resulting low-degree bound only applies to entrywise truncated Gaussian observations. We believe that the strategy of this paper also gives a direct proof for the Gaussian mixture $\bar { Y } _ { i } = z _ { i } X _ { k ^ { * } ( i ) } + \xi _ { i }$ , where the $\xi _ { i } \sim \mathcal { N } ( 0 , I _ { d } )$ are independent and the centers $\pm X _ { k }$ are those of Section 2. In the Bernoulli model, $Y _ { i j } ^ { 2 } = 1$ , so that it sufices to consider multilinear polynomials, indexed by simple bipartite graphs. In the Gaussian model, an entry $\bar { Y } _ { i j }$ may appear with any power, so that the polynomials should be indexed by bipartite multigraphs. An edge $\boldsymbol { e } = \left( \boldsymbol { v } , \boldsymbol { w } \right)$ with multiplicity $\beta _ { e }$ is then associated with the Hermite polynomial $\mathrm { H e } _ { \beta _ { e } } \bigl ( \bar { Y } _ { \pi _ { 1 } ( v ) \pi _ { 2 } ( w ) } \bigr )$ instead of the monomial $Y _ { \pi _ { 1 } ( v ) \pi _ { 2 } ( w ) } ^ { \beta _ { e } } ,$ , similarly to what is done in [10] for Gaussian mixtures with orthogonal centers, where the polynomials are however indexed by multigraphs on the individuals.

## 5 Proof organization for Theorem 1

In this section, we introduce the main tools towards establishing Theorem 1. Parametrizing polynomials of Boolean variables by graphs is now standard in the analysis of low-degree polynomials [7, 25, 44]. First, we build upon the invariance of the distribution of $( x , Y )$ under permutations of the observations and of the coordinates, and under the symmetry of the random signs, to reduce the space of polynomials: we introduce a basis of permutation-invariant polynomials indexed by a family of bipartite graphs, and we show that graphs with a node of odd degree can be left out. Then, we establish that a suitable sub-family of these polynomials is almost orthogonal. In turn, this allows us to establish an explicit expression of the low-degree correlation $\mathrm { C o r r } _ { \leq D }$

## 5.1 Bipartite graph formalism and invariant polynomial basis

Throughout this work, we restrict our attention to simple and undirected bipartite graphs. In the sequel, we write $G = ( V , W , E )$ where V and W are two disjoint node sets and $E \subset V \times W$ . Given an edge $e \in V \times W$ we respectively write $( l ( e ) , r ( e ) )$ for the corresponding nodes in V and W. Given a node v of $G ,$ , we write $\mathcal { N } _ { G } ( v )$ for the set of neighbors of v in $G ,$ , and $\deg _ { G } ( v ) : = | \mathcal { N } _ { G } ( v ) |$ for the degree of $v ;$ the node v is isolated when $\deg _ { G } ( v ) = 0$ . Given two nodes v and $w ,$ we write v ${ \sim } G$ w when v and w are adjacent in G. When there is no possibility of confusion, we simply write $\mathcal { N } ( v )$ , deg(v) and $v \sim w$

Given a set $V = \{ v _ { 1 } , \ldots , v _ { r } \}$ of size at least 2, we denote by $\overline { { \Pi } } _ { V }$ the collection of all injective maps $\pi _ { 1 }$ from V to [n] such that $\pi _ { 1 } ( v _ { 1 } ) = 1$ and $\pi _ { 1 } ( v _ { 2 } ) = 2$ . Given a set W of the form $W = \{ w _ { 1 } , \dots , w _ { s } \}$ with $s \geq 1$ , we denote by $\Pi _ { W }$ the collection of all injective maps $\pi _ { 2 }$ from W to [d].

Given a graph $G , \pi _ { 1 } \in { \overline { { \Pi } } } _ { V }$ , and $\pi _ { 2 } \in \Pi _ { W }$ , we define the multivariate polynomial $P _ { G , \pi _ { 1 } , \pi _ { 2 } }$ by

$$
{ \cal P } _ { G , \pi _ { 1 } , \pi _ { 2 } } = \prod _ { e \in E } Y _ { \pi _ { 1 } ( l ( e ) ) , \pi _ { 2 } ( r ( e ) ) } .\tag{10}
$$

Two bipartite graphs $G = ( V , W , E )$ and $G ^ { \prime } = ( V ^ { \prime } , W ^ { \prime } , E ^ { \prime } )$ are said to be equivalent if there exist bijections $\tau _ { 1 }$ from V to $V ^ { \prime }$ such that $\tau _ { 1 } ( v _ { 1 } ) = v _ { 1 } ^ { \prime }$ and $\tau _ { 1 } ( v _ { 2 } ) = v _ { 2 } ^ { \prime }$ , and $\tau _ { 2 }$ from $W$ to $W ^ { \prime }$ such that the pair $( \tau _ { 1 } , \tau _ { 2 } )$ maps $E$ to $E ^ { \prime }$ , that is $E ^ { \prime } = \{ ( \tau _ { 1 } ( l ( e ) ) , \tau _ { 2 } ( r ( e ) ) ) : e \in E \}$

Let $\mathcal { G } _ { \leq D }$ denote a set of representatives of the equivalence classes of graphs $G = ( V , W , E )$ with $v _ { 1 } , v _ { 2 } \in V$ with at least one and at most $D$ edges, and such that each node $v \in V \setminus \{ v _ { 1 } , v _ { 2 } \}$ and each node $w \in W$ is not isolated.

Since $Y _ { i j } ^ { 2 } = 1$ , we identify $\mathbb { R } _ { D } [ Y ]$ with the space of multilinear polynomials of degree at most D in the entries of $Y .$ . Then, 1 and the distinct polynomials $P _ { G , \pi _ { 1 } , \pi _ { 2 } }$ , with $G \in { \mathcal { G } } _ { \leq D } , \pi _ { 1 } \in { \overline { { \Pi } } } _ { V }$ and $\pi _ { 2 } \in \Pi _ { W }$ , form a basis of $\mathbb { R } _ { D } [ Y ]$ . Define the polynomials $P _ { G }$ by summing over all labelings $\pi _ { 1 }$ and $\pi _ { 2 }$

$$
P _ { G } = \sum _ { \pi _ { 1 } \in \overline { { { \Pi } } } _ { V } } \sum _ { \pi _ { 2 } \in \Pi _ { W } } P _ { G , \pi _ { 1 } , \pi _ { 2 } } \ .
$$

We readily check that $( 1 , ( P _ { G } ) _ { G \in { \mathcal { G } } _ { < D } } )$ is a basis of the subspace of polynomials in $\mathbb { R } _ { D } [ Y ]$ that are invariant under permutations of the d columns and of the rows $i \in [ 3 : n ]$ . The following lemma is a consequence of the permutation invariance of the distribution of $Y$

Lemma 5. We have

$$
\mathrm { C o r r } _ { \le D } = \operatorname * { s u p } _ { { P \in \mathrm { s p a n } \big ( 1 , ( P _ { G } ) _ { G \in { \mathscr G } _ { \le D } } \big ) } \colon { \mathbb E } [ P ^ { 2 } ] \le 1 } \mathbb E [ x P ] .
$$

## 5.2 Correction and reduction to even-degree graphs

Next, we establish that we can leave out graphs G that contain at least one node whose degree is odd. First, the following lemma states that the corresponding polynomials are uncorrelated with the functional $x .$

Lemma 6. Consider any graph $G = ( V , W , E )$ and any labeling $~ \pi ~ = ~ ( \pi _ { 1 } , \pi _ { 2 } )$ in $\overline { { \Pi } } _ { V } \times \Pi _ { W }$ . We have $\mathbb { E } [ P _ { G , \pi } ] = 0$ and $\mathbb { E } [ x P _ { G , \pi } ] = 0$ if at least one of the following two conditions holds:

(a) The degree of a node $w \in W$ is odd.

(b) The degree of a node $v \in V$ is odd.

Condition (a) is a consequence of the symmetry of the Rademacher variables $( a _ { j } )$ in the definition (2) of $X _ { l , j }$ . Similarly, Condition (b) follows from the symmetry of the Rademacher random variables $\left( z _ { i } \right)$ in the definition of $Y _ { i }$

A graph $G = ( V , W , E ) \in \mathcal { G } _ { \leq D }$ is said to be of even degree if the degree of all nodes $v \in V$ and $w \in W$ is even. We write $\mathcal { G } _ { < D } ^ { \mathrm { e v e n } } \subset \mathcal { G } _ { \leq D }$ for the subcollection of $\mathscr { G } _ { \leq D }$ made of even-degree graphs. The following lemma states that polynomials $P _ { G }$ associated to $G \in { \mathcal { G } } _ { < D } ^ { \mathrm { e v e n } }$ are orthogonal to those with $G \in \mathcal { G } _ { \leq D } \setminus \mathcal { G } _ { < D } ^ { \mathrm { e v e n } }$

Lemma 7. Consider a graph $G ^ { ( 1 ) } = ( V ^ { ( 1 ) } , W ^ { ( 1 ) } , E ^ { ( 1 ) } ) \in \mathcal { G } _ { < D } ^ { \mathrm { e v e n } }$ and a graph $G ^ { ( 2 ) } = ( V ^ { ( 2 ) } , W ^ { ( 2 ) } , E ^ { ( 2 ) } ) \in$ $\mathcal { G } _ { \leq D } \setminus \mathcal { G } _ { \leq D } ^ { \mathrm { e v e n } }$ . For any labeling $\pi ^ { ( 1 ) } = ( \pi _ { 1 } ^ { ( 1 ) } , \pi _ { 2 } ^ { ( 1 ) } )$ and $\pi ^ { ( 2 ) } = ( \pi _ { 1 } ^ { \overline { { { ( 2 ) } } } } , \pi _ { 2 } ^ { ( 2 ) } )$ of those graphs, we have

$$
\mathbb { E } \left[ P _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } P _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right] = 0 \ .
$$

Again, the property follows from the symmetry of the Rademacher random variables $( a _ { j } )$ and $\left( z _ { i } \right)$ . As a consequence of the two previous lemmas, the low-degree correlation is achieved by the span of the $P _ { G } \mathrm { ^ { } s }$ where G is an even-degree graph.

Next, we shall transform the polynomials $P _ { G , \pi _ { 1 } , \pi _ { 2 } }$ to ensure that they are almost-orthogonal. Since the distribution of Y is invariant under permutations of [n] and under permutations of [d], we have that $\mathbb { E } [ P _ { G , \pi _ { 1 } , \pi _ { 2 } } ]$ does not depend on the choice of $\pi _ { 1 }$ and $\pi _ { 2 }$ . Given any $G \in { \mathcal { G } } _ { \leq D }$ , we write $\csc ( G )$ for the number of non-trivial connected components of $G ^ { 4 }$ . Then, we partition E into its $\csc ( G )$ non-trivial connected components $E _ { 1 } , E _ { 2 } , . . . , E _ { \mathrm { c c } ( G ) }$ , and we define accordingly the graphs $G _ { r } = ( V _ { r } , W _ { r } , E _ { r } )$ for $r = 1 , \ldots , \operatorname { c c } ( G )$ where $V _ { r } \subset V$ and $W _ { r } \subset W$ are the nodes induced by $E _ { r }$ (so that $G _ { \imath }$ has no isolated node). Then, we define $\Psi _ { G , \pi _ { 1 } , \pi _ { 2 } }$ by centering by connected components.

$$
\Psi _ { G , \pi _ { 1 } , \pi _ { 2 } } = \prod _ { r = 1 } ^ { \mathrm { c c } ( G ) } \left[ P _ { G _ { r } , \pi _ { 1 } , \pi _ { 2 } } - \mathbb { E } \left( P _ { G _ { r } , \pi _ { 1 } , \pi _ { 2 } } \right) \right] \ .\tag{11}
$$

Centering the multivariate polynomials by connected components has already been done for the analysis of low-degree polynomials in rank-one matrix estimation $[ 4 1 ]$ , in stochastic block models $[ 7 , 8 ]$ , and in clustering [10]. This centering drastically reduces the cross-moments $\mathbb { E } \big [ \Psi _ { G ^ { ( 1 ) } , \pi _ { 1 } ^ { ( 1 ) } , \pi _ { 2 } ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi _ { 1 } ^ { ( 2 ) } , \pi _ { 2 } ^ { ( 2 ) } } \big ]$ when no non-isolated node of $G ^ { ( 1 ) }$ shares its label with a non-isolated node of $G ^ { ( 2 ) } \mathrm { ~ - ~ }$ where the label of a node $v \in V ^ { ( i ) }$ (resp. $w \in W ^ { ( i ) } )$ is $\pi _ { 1 } ^ { ( i ) } ( v )$ (resp. $\pi _ { 2 } ^ { ( i ) } ( w ) ) \ - ,$ and more generally when the labels of some connected component of one graph are disjoint from those of the other graph (see Proposition 1 and the remark below it).

Finally, we define the polynomials $\Psi _ { G }$ and their normalized counterparts $\tilde { \Psi } _ { G }$ by summing over all labelings $\pi _ { 1 }$ and π<sub>2</sub>

$$
\Psi _ { G } = \sum _ { \pi _ { 1 } \in \overline { { { \Pi } } } _ { V } } \sum _ { \pi _ { 2 } \in \Pi _ { W } } \Psi _ { G , \pi _ { 1 } , \pi _ { 2 } } \ ; \qquad { \tilde { \Psi } } _ { G } = { \frac { n ^ { - ( | V | - 2 ) / 2 } d ^ { - | W | / 2 } } { \sqrt { | \mathrm { A u t } ( G ) | } } } \Psi _ { G } \ ,\tag{12}
$$

where $\operatorname { A u t } ( G )$ is the set of all automorphisms of G that preserve $v _ { 1 }$ and $v _ { 2 }$ . Then, we deduce from Lemmas 6 and $^ { 7 , }$ together with the parametrization of the $\Psi _ { G } \mathrm { ' s } .$ , the following result.

Lemma 8. We have

$$
\mathrm { C o r r } _ { \le D } = \operatorname * { s u p } _ { { P \in \mathrm { s p a n } \Big ( 1 , ( \Psi _ { G } ) _ { G \in { \mathscr G } _ { \le D } ^ { \mathrm { e v e n } } } \Big ) } \colon { \mathbb E } [ P ^ { 2 } ] \le 1 } \qquad \mathbb { E } [ x P ] .
$$

## 5.3 Almost-orthonormality of the $\tilde { \Psi } _ { G } { \bf \ ' _ { S } }$

The main step of the proof is to establish that the polynomials $\tilde { \Psi } _ { G }$ are almost-orthonormal. This is summarized in the following result.

Theorem 2. Assume that Condition $( \mathcal { H } _ { s i g n a l } )$ holds with a constant $c _ { \mathrm { s } } \geq 1 0$ , that $2 ^ { 8 } D ^ { 4 } \leq d , ~ D \geq 2$ , and that $n \geq d .$ Then, for any $( \alpha _ { \emptyset } , ( \alpha _ { G } ) _ { G \in { \mathcal { G } } _ { \leq D } ^ { \mathrm { e v e n } } } )$ , we have

$$
( 1 - 4 D ^ { - c _ { s } / 2 } - \frac { 4 D ^ { 2 } } { d } ) \left[ \alpha _ { \theta } ^ { 2 } + \sum _ { G \in \mathcal { G } _ { \Sigma D } ^ { \infty } } \alpha _ { G } ^ { 2 } \right] \leq \mathbb { E } \left[ \left( \alpha _ { \theta } + \sum _ { G \in \mathcal { G } _ { \Sigma D } ^ { \infty } } \alpha _ { G } \bar { \Psi } _ { G } \right) ^ { 2 } \right] \leq ( 1 + 4 D ^ { - c _ { s } / 2 } + \frac { 4 D ^ { 2 } } { d } ) \left[ \alpha _ { \theta } ^ { 2 } + \sum _ { G \in \mathcal { G } _ { \Sigma D } ^ { \infty } } \alpha _ { G } ^ { 2 } \right]
$$

This theorem is proved in Section C. The key ingredient for this proof is a tight control of the crossmoments $\mathbb { E } [ \Psi _ { G } \Psi _ { G ^ { \prime } } ]$ . This is done in Section 6 and this is where specific properties of our design come into play.

Remark: We believe that Condition $( \mathcal { H } _ { s i g n a l } )$ cannot be significantly weakened–say by some power of $d -$ Indeed, our bounds of the covariance between the $\tilde { \Psi } _ { G }$ are order-wise tight –see the comment below Lemma 11.

## 5.4 Proof of Theorem 1

Since $k ^ { * } ( 1 )$ and $k ^ { * } ( 2 )$ are i.i.d. $\operatorname { U n i f } ( [ K ] )$ , we have $\mathbb { E } [ x ] = \mathbb { P } [ k ^ { * } ( 1 ) = k ^ { * } ( 2 ) ] = 1 / K$ . If $D = 1$ , then $\mathcal { G } _ { < D } ^ { \mathrm { e v e n } }$ is empty, since the two nodes of a graph with a single edge have degree 1. By Lemma 8, the supremum defining $\mathrm { C o r r } { \_ } D$ then runs over constant polynomials $P = c$ with $c ^ { 2 } \leq 1$ , so that $\mathrm { C o r r } _ { \le D } = \mathbb { E } [ x ] = 1 / K$ and the bound of Theorem 1 trivially holds. Henceforth, we assume that $D \geq 2 .$ so that Theorem 2 and Lemma 9 below apply.

Combining Theorem 2 with Lemma $^ { 8 , }$ we readily get that the low-degree correlation satisfies

$$
\mathrm { C o r r } _ { \leq D } ^ { 2 } \leq \left[ 1 - 4 D ^ { - c _ { \mathrm { s } } / 2 } - \frac { 4 D ^ { 2 } } { d } \right] ^ { - 1 } \left[ \left( \mathbb { E } [ x ] \right) ^ { 2 } + \sum _ { G \in \mathcal { G } _ { \leq D } ^ { \mathrm { e v e n } } } \left( \mathbb { E } [ x \tilde { \Psi } _ { G } ] \right) ^ { 2 } \right] ~ .\tag{13}
$$

Recall that $( \mathbb { E } [ x ] ) ^ { 2 } = 1 / K ^ { 2 }$ . It only remains to bound $\mathbb { E } [ x \tilde { \Psi } _ { G } ]$ . Recall that $\rho$ is defined in $( \mathcal { H } _ { s i g n a l } )$

Lemma 9. Assume that Condition $( \mathcal { H } _ { s i g n a l } )$ holds with $c _ { \mathrm { s } } \geq 1 0$ , that $2 ^ { 8 } D ^ { 4 } \leq d , D \geq 2$ , and that $n \geq d .$ Consider a graph $G \in \mathcal { G } _ { \leq D } ^ { \mathrm { e v e n } }$ . If the degree of one node in $V \setminus  \{ v _ { 1 } , v _ { 2 } \}$ is smaller than $2 m + 2$ , we have $\mathbb { E } [ x \tilde { \Psi } _ { G } ] = 0$ . Otherwise, we have

$$
\left| \mathbb { E } [ x \tilde { \Psi } _ { G } ] \right| \leq \frac { 1 6 } { K } \rho ^ { | E | } \mathrm { ~ . ~ }
$$

The proof of this lemma is postponed to Section D. This implies that

$$
\sum _ { G \in \mathcal { G } _ { \leq D } ^ { \mathrm { e v e n } } } \left( \mathbb { E } [ x \tilde { \Psi } _ { G } ] \right) ^ { 2 } \leq \frac { 2 ^ { 8 } } { K ^ { 2 } } \sum _ { \substack { G \in \mathcal { G } _ { \leq D } ^ { \mathrm { e v e n } } : \mathrm { d e g } ( v ) \geq 2 m + 2 , \forall v \in V \backslash \{ v _ { 1 } , v _ { 2 } \} } } \rho ^ { 2 | E | } .
$$

The number of graphs $G \in { \mathcal { G } } _ { < D } ^ { \mathrm { e v e n } }$ that have e edges is smaller than $( D + 2 ) ^ { 2 e }$ since such graphs contain at most $D + 2$ nodes in $V$ and at most D nodes in W and since there are no isolated nodes with the possible exception of $v _ { 1 }$ and $v _ { 2 } .$ . Since $D \geq 2$ and since $\rho \le D ^ { - 1 6 c _ { \mathrm { s } } }$ (Condition $\left( \mathcal { H } _ { s i g n a l } \right) )$ , we have $\rho ^ { 2 } ( D + 2 ) ^ { 2 } \leq D ^ { 4 - 3 2 c _ { \mathrm { s } } }$ , using that $( D + 2 ) ^ { 2 } \leq D ^ { 4 }$ for $D \geq 2$ . Since graphs $G \in \mathcal { G } _ { \leq D } ^ { \mathrm { e v e n } }$ have an even number of edges, it follows that

$$
\begin{array} { l } { { \displaystyle \sum _ { G \in \mathcal { G } _ { \leq D } ^ { \mathrm { e v e n } } } \left( \mathbb { E } [ x \tilde { \Psi } _ { G } ] \right) ^ { 2 } \leq \frac { 2 ^ { 8 } } { K ^ { 2 } } \sum _ { e \geq 2 : \epsilon , \mathrm { e v e n } } \rho ^ { 2 e } ( D + 2 ) ^ { 2 e } = \frac { 2 ^ { 8 } } { K ^ { 2 } } \sum _ { e ^ { \prime } \geq 1 } \left[ \rho ^ { 2 } ( D + 2 ) ^ { 2 } \right] ^ { 2 e ^ { \prime } } \leq \frac { 2 ^ { 8 } } { K ^ { 2 } } \sum _ { e ^ { \prime } \geq 1 } D ^ { ( 8 - 6 4 c _ { \alpha } ) e ^ { \prime } } } } \\ { { \displaystyle \qquad \leq \frac { 2 ^ { 9 } } { K ^ { 2 } } D ^ { 8 - 6 4 c _ { \alpha } } \leq \frac { 4 } { K ^ { 2 } } D ^ { - c _ { \alpha } / 2 } \ , } } \end{array}
$$

where the second-to-last inequality bounds the geometric series (of ratio $D ^ { 8 - 6 4 c _ { \mathrm { s } } } \leq D ^ { - 6 3 2 }$ , since $c _ { \mathrm { s } } \geq 1 0 )$ by twice its first term, and the last inequality uses $1 2 8 D ^ { 8 - 6 3 . 5 c _ { \mathrm { s } } } \leq 1$ , which holds for $D \geq 2$ and $c _ { \mathrm { s } } \geq 1 0$ Together with Equation (13) and writing $\begin{array} { r } { a : = 4 D ^ { - c _ { \mathrm { s } } / 2 } + \frac { 4 D ^ { 2 } } { d } } \end{array}$ , this implies that

$$
\mathrm { C o r r } _ { \leq D } ^ { 2 } \leq \frac { 1 } { K ^ { 2 } } \cdot \frac { 1 + 4 D ^ { - c _ { \mathrm { s } } / 2 } } { 1 - a } \leq \frac { 1 } { K ^ { 2 } } \cdot \frac { 1 + a } { 1 - a } \leq \frac { 1 } { K ^ { 2 } } \left( 1 + \frac { 1 6 } { D ^ { c _ { \mathrm { s } } / 2 } } + \frac { 1 6 D ^ { 2 } } { d } \right) \enspace ,
$$

where we used $4 D ^ { - { c _ { \mathrm { s } } } / 2 } \le a$ for the second inequality, and $\begin{array} { r } { \frac { 1 + a } { 1 - a } \leq 1 + } \end{array}$ 4a for the last one, which holds since $a \leq 1 / 2$ (indeed, $D \geq 2 , c _ { \mathrm { s } } \geq 1 0$ and $2 ^ { 8 } D ^ { 4 } \leq d$ give $\textstyle a \leq 4 \cdot 2 ^ { - 5 } + \frac { 4 D ^ { 2 } } { 2 ^ { 8 } D ^ { 4 } } \leq \frac { 1 } { 8 } + \frac { 1 } { 2 5 6 } < 1 / 2 )$ . This concludes the proof of the theorem.

## 6 Moments of polynomials $P _ { G }$ and $\Psi _ { G }$

This section is where we build upon our construction of the design $\{ \mu _ { k } , \ k \le K \}$ . Before controlling the cross-moments between the $\Psi _ { G } \mathrm { ^ { * } s }$ , we first evaluate the first moments of the canonical polynomials $P _ { G , \pi _ { 1 } , \pi _ { 2 } }$

## 6.1 First moment

Consider any $G \in { \mathcal { G } } _ { < D } ^ { \mathrm { e v e n } }$ . Conditional on $k ^ { * } , \sigma , z _ { i }$ , and $a _ { j }$ , the expectation of $Y _ { i j }$ is equal to $\begin{array} { r } { \frac { \Delta } { \sqrt { d } } z _ { i } a _ { j } \mu _ { k ^ { * } ( i ) , \sigma ( j ) } } \end{array}$ Since the degree of all nodes is even, we deduce that

$$
\begin{array} { r l } & { \mathbb { E } [ P _ { G , \pi _ { 1 } , \pi _ { 2 } } ( Y ) ] = \mathbb { E } _ { k ^ { * } , \sigma } \Bigg [ \displaystyle \prod _ { e \in E } \frac { \Delta } { \sqrt { d } } \mu _ { k ^ { * } ( \pi _ { 1 } ( l ( e ) ) ) , \sigma ( \pi _ { 2 } ( r ( e ) ) ) } \Bigg ] } \\ & { \quad \quad \quad \quad = \left( \frac { \Delta } { \sqrt { d } } \right) ^ { | E | } \mathbb { E } _ { k ^ { * } , \sigma } \Bigg [ \displaystyle \prod _ { v \in V } \displaystyle \prod _ { w \in W : w \sim v } \mu _ { k ^ { * } ( \pi _ { 1 } ( v ) ) , \sigma ( \pi _ { 2 } ( w ) ) } \Bigg ] } \\ & { \quad \quad \quad = \left( \frac { \Delta } { \sqrt { d } } \right) ^ { | E | } \mathbb { E } _ { \sigma } \Bigg [ \displaystyle \prod _ { v \in V } \mathbb { E } _ { k ^ { * } ( \pi _ { 1 } ( v ) ) } \Big [ \displaystyle \prod _ { w \in W : w \sim v } \mu _ { k ^ { * } ( \pi _ { 1 } ( v ) ) , \sigma ( \pi _ { 2 } ( w ) ) } \Big ] \Bigg ] . } \end{array}
$$

The expectation $\begin{array} { r } { \mathbb { E } _ { k ^ { * } ( \pi _ { 1 } ( v ) ) } \Big [ \prod _ { w \in W : w \sim v } \mu _ { k ^ { * } ( \pi _ { 1 } ( v ) ) , \sigma ( \pi _ { 2 } ( w ) ) } \Big ] } \end{array}$ is equal to the evaluation of the order $\deg _ { G } ( v )$ tensor $\begin{array} { r } { \frac { 1 } { K } \sum _ { k \le K } \mu _ { k } ^ { \otimes \mathrm { d e g } _ { G } ( v ) } } \end{array}$ at the coordinate $( \sigma ( \pi _ { 2 } ( w ) ) )$ for the neighbors w of v –since this tensor is symmetric, this does not depend on the ordering of the neighbors w of v. In other words, we have

$$
\mathbb { E } _ { k ^ { * } ( \pi _ { 1 } ( v ) ) } \Big [ \prod _ { w \in W : w \sim v } \mu _ { k ^ { * } ( \pi _ { 1 } ( v ) ) , \sigma ( \pi _ { 2 } ( w ) ) } \Big ] = \Big ( \frac { 1 } { K } \sum _ { k \le K } \mu _ { k } ^ { \otimes \mathrm { d e g } _ { G } ( v ) } \Big ) _ { ( \sigma ( \pi _ { 2 } ( w ) ) ) _ { w \sim v } } \ .
$$

Lemma 3 states that this expectation takes values in {0, 1} and that it is equal to 1 if and only if, for all $1 \leq s \leq m$ , we have

$$
\sum _ { w : w \sim _ { G } v } \phi ^ { ( 2 s - 1 ) } ( \sigma ( \pi _ { 2 } ( w ) ) ) = 0 .\tag{14}
$$

Introduce the system $( S _ { G } )$ of equations

$$
\sum _ { w : w \sim G } \phi ^ { ( 2 s - 1 ) } ( \sigma ( \pi _ { 2 } ( w ) ) ) = 0 , \qquad \forall 1 \leq s \leq m , \forall v \in V ,\tag{S<sub>G</sub>}
$$

and define $\boldsymbol { \mathcal { A } } _ { G }$ as the event that the system of equations $( S _ { G } )$ is satisfied – we also write $\mathcal { A } [ G ]$ for $\boldsymbol { \mathcal { A } } _ { G }$ when G is given by a lengthy expression $- ^ { 5 }$ . Then, it follows that

$$
\mathbb { E } [ P _ { G , \pi _ { 1 } , \pi _ { 2 } } ( Y ) ] = \Big ( \frac { \Delta } { \sqrt { d } } \Big ) ^ { | E | } \mathbb { P } _ { \sigma } [ A _ { G } ] \ .\tag{15}
$$

Then, we control the probability $P _ { \sigma } [ \mathcal { A } _ { G } ]$ that the random system of equations $( S _ { G } )$ is satisfied. The proofs of the following lemmas are postponed to Section B.

Lemma 10. Consider a bipartite graph $G = ( V , W , E )$ such that the degree of all nodes is even. If there exists $V _ { 0 } \subset V$ such that $| \oplus _ { v \in V _ { 0 } } \mathcal { N } ( v ) | \in [ 1 : 2 m + 1 ]$ , then $\mathbb { P } _ { \sigma } [ \mathcal { A } _ { G } ] = 0$

Proof sketch. Assume that $( S _ { G } )$ holds and sum its equations over $v \in V _ { 0 }$ Since the sums are taken modulo 2, the coordinates w that are adjacent to an even number of nodes of $V _ { 0 }$ cancel out, so that $\begin{array} { r } { \sum _ { w \in \oplus _ { v \in V _ { \mathrm { 0 } } } , N ( v ) } \phi ^ { ( 2 s - 1 ) } ( \sigma ( \pi _ { 2 } ( w ) ) ) \ = \ 0 } \end{array}$ for all $s ~ \leq ~ m$ . Besides, $| \oplus _ { v \in V _ { 0 } } \mathcal { N } ( v ) |$ has the same parity as $\sum { _ { v \in V _ { 0 } } \deg ( v ) }$ , which is even, so that it lies in $[ 1 : 2 m ]$ . As the $\sigma ( \pi _ { 2 } ( w ) )$ are pairwise distinct, this contradicts Lemma 24, which states that such a vanishing sum of at most 2m terms is necessarily paired. The full proof is given in Section B.

In particular, the lemma implies that $\mathbb { E } _ { \sigma } [ { \bf 1 } \{ \mathcal { A } _ { G } \} ] = 0$ as soon as there exists a node $v \in V$ whose degree belongs to $[ 1 : 2 m + 1 ]$ . In light of this lemma, we introduce the following condition on the graph G.

Condition 1 $( \mathcal { W } _ { G } )$ . A graph $G = ( V , W , E )$ satisfies Condition $\left( \mathcal { W } _ { G } \right) ~ i f ~ ( i )$ the degree of each node $w \in W$ is even and (ii) for any $V _ { 0 } \subset V $ , we have

$$
\mid \oplus _ { v \in V _ { 0 } } \mathcal { N } ( v ) \mid \not \in [ 1 : 2 m + 1 ] \ .\tag{16}
$$

For any graph $G = ( V , W , E )$ , we define

$$
b ( G ) : = \frac { | W | } { 2 } - \frac { | E | } { 4 } + \frac { | V | } { 2 } ( m + 1 / 2 ) - \frac { \csc ( G ) } { 2 } .\tag{17}
$$

The following lemma bounds $\mathbb { P } _ { \sigma } [ \mathcal { A } _ { G } ]$ by evaluating the cardinality of the variety defined by $( S _ { G } )$ –see Section B.

Lemma 11. Consider any graph $G = ( V , W , E )$ that satisfies Condition $( \mathcal { W } _ { G } )$ , that does not contain any isolated node, and such that $| E | \le 2 D$ (this covers in particular any graph $G _ { \oplus }$ obtained as the symmetric diference of two graphs in $\mathscr { G } _ { \leq D } )$ . Assume that $D \geq 2$ and that d $\geq 2 ^ { 8 } D ^ { 4 }$ . We have

$$
\mathbb { P } _ { \sigma } [ \mathcal { A } _ { G } ] \le 2 ( 2 m ) ^ { | E | / 2 } \left( \frac { 1 } { d } \right) ^ { \lceil b ( G ) \rceil } \ .\tag{18}
$$

Remark. Consider a graph G such that the degree of all nodes is even. If G does not satisfy Condition $( \mathcal { W } _ { G } )$ the bound (18) still obviously holds by Lemma 10.

Remark (optimality of the exponent $\left\lceil b ( G ) \right\rceil )$ . The exponent $\lceil b ( G ) \rceil$ in (18) cannot be improved in general. Given an integer $r \geq 1$ , consider the graph $G ^ { \circ } = ( V ^ { \circ } , W ^ { \circ } , \dot { E } ^ { \circ } )$ made of a cycle of 2r pairs of twin nodes, defined as follows. For $t \in \mathbb { Z } / 2 r \mathbb { Z } .$ the twin nodes $v _ { t } , v _ { t } ^ { \prime } \in V ^ { \circ }$ have the same neighborhood, made of 2m $+ 2$ nodes of $W ^ { \circ } { : }$ 2m nodes $w _ { t , 1 } , \ldots , w _ { t , 2 m }$ of degree 2, which are only adjacent to $v _ { t }$ and $v _ { t } ^ { \prime } .$ and two nodes $u _ { t - 1 }$ and $u _ { t }$ of degree 4, where $u _ { t }$ is adjacent to $\boldsymbol { v } _ { t } , \ \boldsymbol { v } _ { t } ^ { \prime } , \ \boldsymbol { v } _ { t + 1 }$ and $v _ { t + 1 } ^ { \prime }$ . Hence, the nodes of degree 4 connect each pair of twin nodes to the rest of the cycle. The graph $G ^ { \circ }$ is connected, with $| V ^ { \circ } | = 4 r$ $| W ^ { \circ } | = 2 r ( 2 m + 1 )$ and $| E ^ { \circ } | = 8 r ( m + 1 )$ , so that $b ( G ^ { \circ } ) = 2 r m - 1 / 2$ and $\left\lceil b ( G ^ { \circ } ) \right\rceil = 2 r m$ . Besides, $G ^ { \circ }$ satisfies Condition $( \mathcal { W } _ { G } )$ : since twin nodes have the same neighborhood, $\oplus _ { v \in V _ { 0 } } \mathcal { N } ( v )$ only depends on the set T of indices t such that $V _ { 0 }$ contains exactly one of the nodes $v _ { t } , v _ { t } ^ { \prime } .$ , and its cardinality is equal to $2 m + 2$ if $| T | = 1$ and is at least $2 m | T | \geq 4 m { \mathrm { ~ i f ~ } } | T | \geq 2$ . We show in Section B.4 that, as $d \to \infty$ with m and r fixed,

$$
( 1 - o ( 1 ) ) \Big ( \frac { 1 } { d ^ { m } } \Big ) ^ { 2 r } \leq \mathbb { P } _ { \sigma } [ \mathcal { A } _ { G ^ { \circ } } ] \leq 2 ( 2 m ) ^ { 4 r ( m + 1 ) } \Big ( \frac { 1 } { d ^ { m } } \Big ) ^ { 2 r } \ ,
$$

so that $\mathbb { P } _ { \sigma } [ \mathcal { A } _ { G ^ { \circ } } ]$ is of the order of $( 1 / d ) ^ { \lceil b ( G ^ { \circ } ) \rceil }$ . In other words, each pair of twin nodes contributes a factor $d ^ { - m }$ , which corresponds to its m equations in $( S _ { G } )$ , the equations associated with $ { \boldsymbol { v } } _ { t } ^ { \prime }$ being identical to those associated with v . As noticed in Section B.4, the rank of the Jacobian matrix of the system $( S _ { G } )$ is then exactly equal to $\left\lceil b ( G ^ { \circ } ) \right\rceil$ .

## 6.2 Second Moments

In order to control the second moments of the form $\mathbb { E } [ P _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } P _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } ]$ , we need to introduce further graph notation.

Merged graph $G _ { \cup } ,$ intersection graph $G _ { \cap } ,$ and symmetric diference graph $G _ { \oplus }$ . Given a graph G and labelings π and $\pi _ { 2 }$ , we write $( \pi _ { 1 } , \pi _ { 2 } ) [ G ]$ for the corresponding labeled graph. For such labeled bipartite graphs $( \pi _ { 1 } ^ { ( 1 ) } , \pi _ { 2 } ^ { ( 1 ) } ) [ G ^ { ( 1 ) } ]$ and $( \pi _ { 1 } ^ { ( 2 ) } , \pi _ { 2 } ^ { ( 2 ) } ) [ G ^ { ( 2 ) } ]$ , we define $G _ { \cup }$ as the merged graph, with the convention that identical edges are merged into a single edge.

Similarly, we define the intersection graph $G _ { \cap } = \left( V _ { \cap } , W _ { \cap } , E _ { \cap } \right)$ and the symmetric diference graph $G _ { \oplus } =$ $( V _ { \oplus } , W _ { \oplus } , E _ { \oplus } )$ so that $E _ { \oplus } = E _ { \cup } \setminus E _ { \cap }$ . Here, $V _ { \cap }$ and $W _ { \cap }$ (resp. $V _ { \oplus }$ and $W _ { \oplus } )$ are the sets of nodes induced by the edges $E _ { \cap }$ (resp. $\scriptstyle { E _ { \oplus } } )$ so that $G _ { \cap }$ (resp. $G _ { \oplus } )$ does not contain any isolated node. A connected component of $( \pi _ { 1 } ^ { ( 1 ) } , \pi _ { 2 } ^ { ( 1 ) } ) [ G ^ { ( 1 ) } ]$ with at least one edge is said to be unmatched if none of its nodes is incident to an edge of $( \pi _ { 1 } ^ { ( 2 ) } , \pi _ { 2 } ^ { ( 2 ) } ) [ G ^ { ( 2 ) } ]$ , and symmetrically for the connected components of $( \pi _ { 1 } ^ { ( 2 ) } , \pi _ { 2 } ^ { ( 2 ) } ) [ G ^ { ( 2 ) } ]$ . In other words, the labels of an unmatched component are disjoint from those of the non-isolated nodes of the other labeled graph. We write $\operatorname { c c } _ { \mathrm { u } }$ for the total number of unmatched connected components of the two labeled graphs. This notion should not be confused with the sets of unmatched nodes introduced in Section C.1, which also account for isolated nodes.

Consider any two graphs $G ^ { ( 1 ) }$ and $G ^ { ( 2 ) }$ in $\mathcal { G } _ { \leq D } ^ { \mathrm { e v e n } }$ and any labelings $\pi ^ { ( 1 ) }$ and $\pi ^ { ( 2 ) }$ . Since $Y _ { i j } ^ { 2 } = 1$ for all $i , j ,$ we obtain as in (15) that

$$
\mathbb { E } [ P _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } P _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } ] = \Big ( \frac { \Delta } { \sqrt { d } } \Big ) ^ { \vert E _ { \oplus } \vert } \mathbb { P } _ { \sigma } [ A _ { G _ { \oplus } } ] \ .\tag{19}
$$

Then, we can control the probability $\mathbb { P } _ { \sigma } [ \mathcal { A } _ { G _ { \oplus } } ]$ thanks to Lemmas 10 and 11. Let us emphasize that neither the above identity nor the derivation of (15) relies on the membership of $G ^ { ( 1 ) }$ and $G ^ { ( 2 ) }$ in $\mathcal { G } _ { \leq D } ^ { \mathrm { e v e n } }$ : the derivation of (15) only requires that all the degrees of the (simple) bipartite graph G are even and that the labelings $\pi _ { 1 }$ and $\pi _ { 2 }$ are injective. Likewise, the above identity only requires that all the degrees of $G ^ { ( 1 ) }$ and $G ^ { ( 2 ) }$ are even. Accordingly, we shall freely apply the above identity to any two labeled bipartite graphs with even degrees and injective labelings, even when they do not belong to $\mathcal { G } _ { < D } ^ { \mathrm { e v e n } }$ nor carry distinguished nodes $v _ { 1 } , v _ { 2 }$

Thanks to the correction in the definition (11) of the polynomials $\bar { \Psi } _ { G , \pi } ,$ the cross-moment $\mathbb { E } \big [ \Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \big ]$ can be much smaller than $\mathbb { E } [ P _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } P _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } ]$ . The following proposition is one of the key steps of the proof.

Proposition 1. Consider any two graphs $G ^ { ( 1 ) } = ( V ^ { ( 1 ) } , W ^ { ( 1 ) } , E ^ { ( 1 ) } )$ and $G ^ { ( 2 ) } = ( V ^ { ( 2 ) } , W ^ { ( 2 ) } , E ^ { ( 2 ) } )$ , each with at most D edges and such that every node has even degree, and any injective labelings $\pi ^ { ( 1 ) } = ( \pi _ { 1 } ^ { ( 1 ) } , \pi _ { 2 } ^ { ( 1 ) } )$ and $\pi ^ { ( 2 ) } = ( \pi _ { 1 } ^ { ( 2 ) } , \pi _ { 2 } ^ { ( 2 ) } )$ , with $\pi _ { 1 } ^ { ( i ) } : V ^ { ( i ) } \to [ n ]$ and $\pi _ { 2 } ^ { ( i ) } : W ^ { ( i ) } \to [ d ]$ (not necessarily lying in $\overline { { \Pi } } _ { V } \times \Pi _ { W } ;$ in particular ${ \cal G } ^ { ( 1 ) } , { \cal G } ^ { ( 2 ) }$ need not belong to $\mathcal { G } _ { \leq D } ^ { \mathrm { e v e n } }$ nor carry distinguished nodes $v _ { 1 } , v _ { 2 } )$ . Assume that $D \geq 2$ $2 ^ { 8 } D ^ { 4 } \leq d , n \geq d ,$ and that Condition $( \mathcal { H } _ { s i g n a l } )$ holds for some $c _ { \mathrm { s } } \geq 1 0$ . We have the following

(a) If at least one node $v \in V ^ { ( 1 ) } \ ( r e s p . \ v \in V ^ { ( 2 ) } )$ satisfies de $\operatorname { J } _ { G ^ { ( 1 ) } } ( v ) \in [ 1 : 2 m + 1 ]$ and $\pi _ { 1 } ^ { ( 1 ) } ( v ) \notin \pi _ { 1 } ^ { ( 2 ) } ( V ^ { ( 2 ) } )$ (resp. $\deg _ { G ^ { ( 2 ) } } ( v ) \in [ 1 : 2 m + 1 ]$ and $\pi _ { 1 } ^ { ( 2 ) } ( v ) \notin \pi _ { 1 } ^ { ( 1 ) } ( V ^ { ( 1 ) } ) )$ , then

$$
\mathbb { E } \left[ \Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right] = 0 \ ; \qquad \mathbb { E } \left[ P _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right] = 0 \ .
$$

(b) In any case, we have

$$
\left| \mathbb { E } \left[ \Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right] \right| \leq 2 ( 3 2 D ^ { 3 } ) ^ { \mathrm { c c _ { u } } } \left( 4 m \frac { \Delta } { d ^ { 1 / 4 } } \right) ^ { | E _ { \oplus } | } \left( \frac { 1 } { d } \right) ^ { ( 2 m + 1 ) | V _ { \oplus } | / 4 + | W _ { \oplus } | / 2 - \mathrm { c c } ( G _ { \oplus } ) / 2 + \mathrm { c c _ { u } } / 2 ) }
$$

(c) If $E _ { \oplus } = \emptyset$ , then

$$
\left| \mathbb { E } \left[ \Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right] - 1 \right| \leq \frac { 8 m D \Delta } { d ^ { 1 / 4 } } .
$$

(d) If no non-isolated node of $\pi ^ { ( 1 ) } [ G ^ { ( 1 ) } ]$ shares its label with a non-isolated node of $\pi ^ { ( 2 ) } [ G ^ { ( 2 ) } ]$ , that is, if $\csc _ { \mathrm { u } } = \csc ( G ^ { ( 1 ) } ) + \csc ( G ^ { ( 2 ) } )$ , then

$$
\left| \mathbb { E } \left[ P _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right] \right| \leq ( 3 2 D ^ { 3 } ) ^ { \operatorname { c c } ( G ^ { ( 2 ) } ) } \left( 4 m \frac { \Delta } { d ^ { 1 / 4 } } \right) ^ { | E _ { \oplus } | } \left( \frac { 1 } { d } \right) ^ { ( 2 m + 1 ) | V _ { \oplus } | / 4 + | W _ { \oplus } | / 2 - \operatorname { c c } ( G ^ { ( 1 ) } ) / 2 }
$$

Remark: Benefit of centered polynomials. $\mathrm { B y }$ (19) and Lemma 11, the non-centered cross-moment satisfies

$$
\left| \mathbb { E } \left[ P _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } P _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right] \right| \leq 2 \left( { \sqrt { 2 m } } { \frac { \Delta } { d ^ { 1 / 4 } } } \right) ^ { | E _ { \oplus } | } \left( { \frac { 1 } { d } } \right) ^ { ( 2 m + 1 ) | V _ { \oplus } | / 4 + | W _ { \oplus } | / 2 - \operatorname { c c } ( G _ { \oplus } ) / 2 }
$$

Up to the factor $( 3 2 D ^ { 3 } ) ^ { \mathrm { { c c } _ { u } } }$ and to the constants, Bound (b) improves upon this bound by the factor $( 1 / d ) ^ { \mathrm { c c _ { u } / 2 } }$ that is by a factor $d ^ { - 1 / 2 }$ per unmatched connected component. This gain is due to the centering by connected components in (11). Indeed, consider an unmatched connected component of, say, $\pi ^ { ( 1 ) } [ G ^ { ( 1 ) } ]$ . Its rows and columns are distinct from all the other rows and columns involved in the cross-moment, so that, conditionally on $\sigma ,$ it is independent of the remaining factors of the product. Since it is centered, the cross-moment would vanish if the coordinates were not coupled through the random permutation σ. As σ only induces a weak dependence between the images of disjoint sets of columns, each unmatched component only contributes a factor of order $d ^ { - 1 / 2 }$ . In particular, when the two labeled graphs do not intersect, we have $\csc _ { \mathrm { u } } = \csc ( G ^ { ( 1 ) } ) +$ $\csc ( G ^ { ( 2 ) } )$ , and the gain is of order $d ^ { - ( \csc ( G ^ { ( 1 ) } ) + \csc ( G ^ { ( 2 ) } ) ) / 2 }$

## AI Disclosure

In an early phase, the authors used the AI assistant Gemini (Google) for bibliographical research on BCH codes, spherical design and on algebraic geometry (algebraic arguments of Lemma 11). In the final phase, the authors used the AI assistant Claude (Anthropic, Opus and Sonnet models 5.0 and 5.5), through the Claude Code interface, to help write and edit the main sections of this manuscript (Sections 1 to 4), and to audit the proofs. This assistance included drafting and rephrasing paragraphs of the introduction and of the discussion, reorganizing the presentation of the model and of the BCH construction, and harmonizing the notation. All the text produced with this assistance was reviewed and rephrased by the authors, who take full responsibility for the content of the manuscript, including all its mathematical statements and proofs.

## Acknowledgements

The work of A. Carpentier is partially supported by the Deutsche Forschungsgemeinschaft (DFG)- Project-ID 318763901 - SFB1294 ”Data Assimilation”, Project A03, by the DFG on the Forschungsgruppe FOR5381 ”Mathematical Statistics in the Information Age - Statistical Eficiency and Computational Tractability”, Project TP 02 (Project-ID 460867398), by the Universite franco-allemande (UFA) through the college doctoral franco-allemand CDFA-02-25 ”Statistisches Lernen f¨ur komplexe stochastische Prozesse”, by the European Research council (ERC) through the ERC Consolidator Grant SOCE (Nr: 101229569). Funded by the European Union (ERC, project SOCE, 101229569). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them. The work of A. Carpentier and N. Verzelen is also supported by the Universite franco-allemande (UFA) through the college doctoral franco-allemand CDFA-02-25 ”Statistisches Lernen f¨ur komplexe stochastische Prozesse”.

## A Proof of Proposition 1

In this section, our main purpose is to control E $\left[ \Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right]$ . Recall the definition

$$
\Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } = \prod _ { i \in [ \mathrm { c c } ^ { ( 1 ) } ] } \left( P _ { G _ { i } ^ { ( 1 ) } , \pi ^ { ( 1 ) } } - \mathbb { E } \left[ P _ { G _ { i } ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \right] \right) .
$$

First, we expand the products inside the definitions of $\Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } }$ and $\Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } }$ . Then, we integrate with respect to $k ^ { * }$ and Y. Given $T _ { 1 } \subset [ \mathrm { c c } ^ { ( 1 ) } ]$ , and $T _ { 2 } \subset [ \mathrm { c c } ^ { ( 2 ) } ]$ , we define $G _ { \oplus } ^ { T _ { 1 } ; T _ { 2 } }$ as the graph associated to the symmetric diference of the labeled graphs $( G ^ { ( 1 ) } , \pi ^ { ( 1 ) } )$ and $( G ^ { ( 2 ) } , \pi ^ { ( 2 ) } )$ from which we have removed the connected components in $[ \mathrm { c c } ^ { ( 1 ) } ] \backslash T _ { 1 }$ and $[ \mathrm { c c } ^ { ( 2 ) } ] \backslash T _ { 2 }$ . Then, we have

$$
\mathbb { E } \left[ \Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right] = \sum _ { \substack { T _ { 1 } \subset [ c ^ { ( 1 ) } ] T _ { 2 } \subset [ c ^ { ( 2 ) } ] } } \sum _ { 1 \atop \ell \in [ c ^ { ( 1 ) } ] } ( - 1 ) ^ { c c ^ { ( 1 ) } + \mathrm { c c } ^ { ( 2 ) } - | T _ { 1 } | - | T _ { 2 } | } \left( \frac { \Delta } { \sqrt { d } } \right) ^ { | E _ { \oplus } ^ { ( T _ { 1 } , T _ { 2 } ) } | + \sum _ { i \notin T _ { 1 } } | E _ { \mathrm { t } } ^ { ( 1 ) } | + \sum _ { i \notin T _ { 2 } } | E _ { \mathrm { t } } ^ { ( 2 ) } | }
$$

Denote by $R _ { 0 } ^ { ( 1 ) } \subset [ \mathrm { c c } ^ { ( 1 ) } ]$ (resp. $R _ { 0 } ^ { ( 2 ) } \subset [ \mathrm { c c } ^ { ( 2 ) } ] )$ the set of connected components of $G ^ { ( 1 ) } ( \mathrm { r e s p . } G ^ { ( 2 ) } )$ such that at least one of its nodes is matched to a node of $G ^ { ( 2 ) } ( \mathrm { r e s p . } \hat { G } ^ { ( 1 ) } )$ . Then, we decompose

E $\left[ \Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right]$ as follows.

$$
\mathbb { E } \left[ \Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right] = \sum _ { R ^ { ( 1 ) } \subset { \cal R } _ { 0 } ^ { ( 1 ) } } \sum _ { R ^ { ( 2 ) } \subset { \cal R } _ { 0 } ^ { ( 2 ) } } \left( \frac { \Delta } { \sqrt { d } } \right) ^ { u [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] } \omega [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \ ;\tag{20}
$$

$$
u [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] = | E _ { \oplus } ^ { R ^ { ( 1 ) } ; R ^ { ( 2 ) } } | + \sum _ { i \in [ \mathrm { c c } ^ { ( 1 ) } ] \backslash R ^ { ( 1 ) } } | E _ { i } ^ { ( 1 ) } | + \sum _ { i \in [ \mathrm { c c } ^ { ( 2 ) } ] \backslash R ^ { ( 2 ) } } | E _ { i } ^ { ( 2 ) } | \ ;\tag{21}
$$

$$
\omega [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] : = \sum _ { \substack { T _ { 1 } \subset [ \mathrm { c c } ^ { ( 1 ) } ] ; T _ { 1 } \cap R _ { 0 } ^ { ( 1 ) } = R ^ { ( 1 ) } } } ( - 1 ) ^ { \mathrm { { c c } ^ { ( 1 ) } + \mathrm { c c } ^ { ( 2 ) } - | T _ { 1 } | - | T _ { 2 } | } } \mathbb { P } _ { \sigma } \left[ \mathcal { A } [ G _ { \Phi } ^ { T _ { 1 } ; T _ { 2 } } ] \right] \prod _ { \substack { i \in [ \mathrm { c c } ^ { ( 1 ) } ] \backslash T _ { 1 } } } \mathbb { P } _ { \sigma } \left[ \mathcal { A } [ G _ { i } ^ { ( 1 ) } ] \right]
$$

The main challenge is to control $| \omega [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] |$

First, we prove Case (a) in Proposition 1. By symmetry, we may assume that $\textit { \textbf { v } } \in \textit { V } ^ { ( 1 ) }$ satisfies $\deg _ { G ^ { ( 1 ) } } ( v ) \ \in \ [ 1 \ : \ 2 m + 1 ]$ and $\pi _ { 1 } ^ { ( 1 ) } ( v ) \notin \pi _ { 1 } ^ { ( 2 ) } ( V ^ { ( 2 ) } )$ . Since v is non-isolated, $\pi _ { 1 } ^ { ( 1 ) } ( v )$ belongs to the (edge-induced) node set of at least one of $G _ { \oplus } ^ { T _ { 1 } ; T _ { 2 } } , G _ { i } ^ { ( 1 ) } , G _ { i } ^ { ( 2 ) }$ for any choice of $T _ { 1 }$ and $T _ { 2 }$ , and its degree in that graph is at most 2m + 1. Then, Lemma 10 implies that ${ \omega } [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] = 0$ . We conclude that $\mathbb { E } \left[ \Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right] = 0$ . Arguing similarly, we obtain $\mathbb { E } \left[ P _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right] = 0$

Let us turn to Case (b), which is the most challenging part of the proof. We have

$$
| \omega [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] | \le \left| \mathbb { E } _ { \sigma } \left[ L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right] \right| \prod _ { \substack { i \in R _ { 0 } ^ { ( 1 ) } \backslash R ^ { ( 1 ) } } } \mathbb { P } _ { \sigma } \left[ \mathcal { A } [ G _ { i } ^ { ( 1 ) } ] \right] \prod _ { \substack { i \in R _ { 0 } ^ { ( 2 ) } \backslash R ^ { ( 2 ) } } } \mathbb { P } _ { \sigma } \left[ \mathcal { A } [ G _ { i } ^ { ( 2 ) } ] \right] ,\tag{23}
$$

where

$$
\begin{array} { r l r } & { } & { L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] : = { \bf 1 } \left\{ { \cal A } [ G _ { \oplus } ^ { R ^ { ( 1 ) } ; R ^ { ( 2 ) } } ] \right\} \prod _ { i \in [ \mathrm { c c } ^ { ( 1 ) } ] \backslash R _ { 0 } ^ { ( 1 ) } } \left( { \bf 1 } \left\{ { \cal A } [ G _ { i } ^ { ( 1 ) } ] \right\} - \mathbb { P } _ { \sigma } \left[ { \cal A } [ G _ { i } ^ { ( 1 ) } ] \right] \right) } \\ & { } & { \cdot \prod _ { i \in [ \mathrm { c c } ^ { ( 2 ) } ] \backslash R _ { 0 } ^ { ( 2 ) } } \left( { \bf 1 } \left\{ { \cal A } [ G _ { i } ^ { ( 2 ) } ] \right\} - \mathbb { P } _ { \sigma } \left[ { \cal A } [ G _ { i } ^ { ( 2 ) } ] \right] \right) } \\ & { } & { i \in [ \mathrm { c c } ^ { ( 2 ) } ] \backslash R _ { 0 } ^ { ( 2 ) } } \end{array}
$$

Lemma 12. Under the assumptions of Proposition 1, we have

$$
\left| \mathbb { E } _ { \sigma } \left[ L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right] \right| \le ( 3 2 D ^ { 3 } ) ^ { \mathrm { c c } _ { \mathrm { u } } } ( 4 m ) ^ { \frac { | E _ { \oplus } ^ { R ^ { ( 1 ) } ; R ^ { ( 2 ) } } | + \Sigma _ { i \in [ \mathrm { c e } ^ { ( 1 ) } ] \setminus R _ { \oplus } ^ { ( 1 ) } } | ^ { R ^ { ( 1 ) } } | + \Sigma _ { i \in [ \mathrm { c e } ^ { ( 2 ) } ] \setminus R _ { 0 } ^ { ( 2 ) } } | ^ { R ^ { ( 2 ) } } | } { 2 } } \left( \frac { 1 } { d } \right) ^ { b _ { R } ( 1 ) , R ^ { ( 2 ) } }
$$

where

$$
\begin{array}{c} b _ { R ^ { ( 1 ) } , R ^ { ( 2 ) } } = - \frac { \vert E _ { \oplus } ^ { R ^ { ( 1 ) } ; R ^ { ( 2 ) } } \vert + \sum _ { i \in [ \mathrm { c c } ^ { ( 1 ) } ] \backslash R _ { 0 } ^ { ( 1 ) } } \vert E _ { i } ^ { ( 1 ) } \vert + \sum _ { i \in [ \mathrm { c c } ^ { ( 2 ) } ] \backslash R _ { 0 } ^ { ( 2 ) } } \vert E _ { i } ^ { ( 2 ) } \vert } { 4 }  \\ { + \left( 2 m + 1 \right) \frac { \vert V _ { \oplus } ^ { R ^ { ( 1 ) } ; R ^ { ( 2 ) } } \vert + \sum _ { i \in [ \mathrm { c c } ^ { ( 1 ) } ] \backslash R _ { 0 } ^ { ( 1 ) } } \vert V _ { i } ^ { ( 1 ) } \vert + \sum _ { i \in [ \mathrm { c c } ^ { ( 2 ) } ] \backslash R _ { 0 } ^ { ( 2 ) } } \vert V _ { i } ^ { ( 2 ) } \vert } { 4 } } \\ { - \frac { \mathrm { c c } \left( G _ { \oplus } ^ { R ^ { ( 1 ) } ; R ^ { ( 2 ) } } \right) } { 2 } + \frac { \vert W _ { \oplus } ^ { R ^ { ( 1 ) } ; R ^ { ( 2 ) } } \vert + \sum _ { i \in [ \mathrm { c c } ^ { ( 1 ) } ] \backslash R _ { 0 } ^ { ( 1 ) } } \vert W _ { i } ^ { ( 1 ) } \vert + \sum _ { i \in [ \mathrm { c c } ^ { ( 2 ) } ] \backslash R _ { 0 } ^ { ( 2 ) } } \vert W _ { i } ^ { ( 2 ) } \vert } { 2 } \ . } \end{array}
$$

Coming back to (23), we control the terms of the form $\mathbb { P } _ { \sigma } [ \mathcal { A } [ G _ { i } ^ { ( 1 ) } ] ]$ and $\mathbb { P } _ { \sigma } [ \mathcal { A } [ G _ { i } ^ { ( 2 ) } ] ]$ using Lemmas 10 and 11. This leads us to

$$
| \omega [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] | \leq ( 3 2 D ^ { 3 } ) ^ { \mathrm { c c } _ { \mathrm { u } } } ( 4 m ) ^ { u [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] / 2 } \left( \frac { 1 } { d } \right) ^ { b _ { R ^ { ( 1 ) } , R ^ { ( 2 ) } } ^ { \prime } }\tag{24}
$$

where

$$
\begin{array} { r l } { b _ { R ^ { ( 1 ) } , R ^ { ( 2 ) } } ^ { \prime } = - \frac { \big | E _ { \oplus } ^ { R ^ { ( 1 ) } ; R ^ { ( 2 ) } } \big | + \sum _ { i \in [ \mathrm { c c } ^ { ( 1 ) } ] \setminus R ^ { ( 1 ) } } \big | E _ { i } ^ { ( 1 ) } \big | + \sum _ { i \in [ \mathrm { c c } ^ { ( 2 ) } ] \setminus R ^ { ( 2 ) } } \big | E _ { i } ^ { ( 2 ) } \big | } { 4 } } & { } \\ { + \left( 2 m + 1 \right) \frac { \big | V _ { \oplus } ^ { R ^ { ( 1 ) } ; R ^ { ( 2 ) } } \big | + \sum _ { i \in [ \mathrm { c c } ^ { ( 1 ) } ] \setminus R ^ { ( 1 ) } } \big | V _ { i } ^ { ( 1 ) } \big | + \sum _ { i \in [ \mathrm { c c } ^ { ( 2 ) } ] \setminus R ^ { ( 2 ) } } \big | V _ { i } ^ { ( 2 ) } \big | } { 4 } } & { } \\ { - \frac { \underline { { \operatorname { c c } } } \left( G _ { \oplus } ^ { R ^ { ( 1 ) } ; R ^ { ( 2 ) } } \right) + \big | R _ { 0 } ^ { ( 1 ) } \setminus R ^ { ( 1 ) } \big | + \big | R _ { 0 } ^ { ( 2 ) } \setminus R ^ { ( 2 ) } \big | } { 2 } } & { } \\ { + \frac { \big | W _ { \oplus } ^ { R ^ { ( 1 ) } ; R ^ { ( 2 ) } } \big | + \sum _ { i \in [ \mathrm { c c } ^ { ( 1 ) } ] \setminus R ^ { ( 1 ) } } \big | W _ { i } ^ { ( 1 ) } \big | + \sum _ { i \in [ \mathrm { c c } ^ { ( 2 ) } ] \setminus R ^ { ( 2 ) } } \big | W _ { i } ^ { ( 2 ) } \big | } { 2 } . } & { } \end{array}
$$

Then, coming back to (20) and (21), we get

$$
\left| \mathbb { E } \left[ \Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right] \right| \leq ( 3 2 D ^ { 3 } ) ^ { \mathrm { c c _ { u } } } \left( \frac { \Delta 2 \sqrt { m } } { \sqrt { d } } \right) ^ { \mathrm { u } [ R _ { 0 } ^ { ( 1 ) } ; R _ { 0 } ^ { ( 2 ) } ] } \left( \frac { 1 } { d } \right) ^ { b _ { R _ { 0 } ^ { ( 1 ) } , R _ { 0 } ^ { ( 2 ) } } ^ { \prime } }
$$

When we move from $R ^ { ( 1 ) }$ to $R _ { 0 } ^ { ( 1 ) }$ , we reduce the number of connected components from $\csc ( { G } _ { \oplus } ^ { R ^ { ( 1 ) } ; R ^ { ( 2 ) } } ) +$ $\vert R _ { 0 } ^ { ( 1 ) } \setminus R ^ { ( 1 ) } \vert + \vert R _ { 0 } ^ { ( 2 ) } \setminus R ^ { ( 2 ) } \vert$ to $\csc ( \it G _ { \oplus } ^ { R _ { 0 } ^ { ( 1 ) } ; R _ { 0 } ^ { ( 2 ) } } )$ by merging some nodes. Since merging two nodes allows one to reduce the number of connected components by at most one and since $2 m + 1 \geq 2$ , we deduce that

$$
b _ { R ^ { ( 1 ) } , R ^ { ( 2 ) } } ^ { \prime } - b _ { R _ { 0 } ^ { ( 1 ) } , R _ { 0 } ^ { ( 2 ) } } ^ { \prime } \geq \frac { u [ R _ { 0 } ^ { ( 1 ) } ; R _ { 0 } ^ { ( 2 ) } ] - u [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] } { 4 } \ .
$$

Let us make this argument precise, as the passage from the $( R ^ { ( 1 ) } , R ^ { ( 2 ) } )$ -configuration to the $( R _ { 0 } ^ { ( 1 ) } , R _ { 0 } ^ { ( 2 ) } )$ configuration not only merges nodes, but also deletes the edges that cancel out in the symmetric diference. Write H for the disjoint union of $G _ { \oplus } ^ { \dot { R } ^ { ( 1 ) } ; R ^ { ( 2 ) } }$ , of the labeled components $\pi ^ { ( 1 ) } [ G _ { i } ^ { ( 1 ) } ]$ with $i \in R _ { 0 } ^ { ( 1 ) } \setminus R ^ { ( 1 ) }$ and of the labeled components $\pi ^ { ( 2 ) } [ G _ { i } ^ { ( 2 ) } ]$ with $i \in R _ { 0 } ^ { ( 2 ) } \setminus R ^ { ( 2 ) }$ . Since the components indexed by $[ \mathrm { c c } ^ { ( 1 ) } ] \backslash R _ { 0 } ^ { ( 1 ) }$ and $[ \mathrm { c c } ^ { ( 2 ) } ] \backslash R _ { 0 } ^ { ( 2 ) }$ contribute in the same way to $b _ { R ^ { ( 1 ) } , R ^ { ( 2 ) } } ^ { \prime }$ and to $b _ { R _ { 0 } ^ { ( 1 ) } , R _ { 0 } ^ { ( 2 ) } } ^ { \prime }$ , and since each excluded component $\pi ^ { ( j ) } [ G _ { i } ^ { ( j ) } ] , i \in R _ { 0 } ^ { ( j ) } \backslash R ^ { ( j ) }$ , is connected –so that the quantity cc $( { G _ { \oplus } ^ { R ^ { ( 1 ) } ; R ^ { ( 2 ) } } } ) + | R _ { 0 } ^ { ( 1 ) } \setminus R ^ { ( 1 ) } | + | R _ { 0 } ^ { ( 2 ) } \setminus R ^ { ( 2 ) } |$ arising in $b _ { R ^ { ( 1 ) } , R ^ { ( 2 ) } } ^ { \prime }$ is nothing but $\csc ( H ) - ,$ , we have

$$
b _ { R ^ { ( 1 ) } , R ^ { ( 2 ) } } ^ { \prime } - b _ { R _ { 0 } ^ { ( 1 ) } , R _ { 0 } ^ { ( 2 ) } } ^ { \prime } = - { \frac { \delta _ { E } } { 4 } } + ( 2 m + 1 ) { \frac { \delta _ { V } } { 4 } } + { \frac { \delta _ { W } } { 2 } } - { \frac { \delta _ { \mathrm { c c } } } { 2 } } ~ ,
$$

where $\delta _ { E } , \delta _ { V } , \delta _ { W }$ , and $\delta _ { \mathrm { c c } }$ respectively denote the numbers of edges, of nodes in $V _ { ; }$ of nodes in W, and of connected components of H minus the corresponding quantities for $G _ { \oplus } ^ { R _ { 0 } ^ { ( 1 ) } ; R _ { 0 } ^ { ( 2 ) } }$ . As $\delta _ { E } = u [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \ : -$ $u [ R _ { 0 } ^ { ( 1 ) } ; R _ { 0 } ^ { ( 2 ) } ]$ , the announced inequality is equivalent to

$$
( 2 m + 1 ) \delta _ { V } + 2 \delta _ { W } \geq 2 \delta _ { \mathrm { c c } } .\tag{25}
$$

Now, $G _ { \oplus } ^ { R _ { 0 } ^ { ( 1 ) } ; R _ { 0 } ^ { ( 2 ) } }$ is obtained from H in two steps: (i) we identify the nodes of H that carry the same label; (ii) we delete the pairs of edges that have become parallel in this identification –namely the labeled edges belonging to both $\pi ^ { ( 1 ) } [ G ^ { ( 1 ) } ]$ and $\pi ^ { ( 2 ) } [ G ^ { ( 2 ) } ]$ , which cancel out in the symmetric diference– as well as the nodes that have become isolated. Write $p _ { V }$ and $p _ { W }$ for the numbers of identifications performed at step (i) among the nodes of V and of $W$ , and $r _ { V }$ and $r _ { W }$ for the numbers of nodes of V and of W that are deleted at step (ii), so that $\delta _ { V } = p _ { V } + r _ { V }$ and $\delta _ { W } = p _ { W } + r _ { W } . \mathrm { ~ S t e p ~ ( i ) }$ reduces the number of connected components by at most $p _ { V } + p _ { W }$ , since each identification of two nodes merges at most two connected components. As for step (ii), each connected component of the graph obtained at step (i) that keeps at least one edge gives rise to at least one connected component of $G _ { \oplus } ^ { R _ { 0 } ^ { ( 1 ) } ; R _ { 0 } ^ { ( 2 ) } }$ , so that step (ii) reduces the number of connected components by at most the number k of connected components of the graph obtained at step (i) that lose all their edges. Any of these k components contains at least one edge –the connected components of the graph obtained at step (i) being unions of connected components of H, which all contain at least one edge–, hence at least one node of V and one node of W, all of which become isolated and are therefore deleted; consequently, $r _ { V } \geq k$ and $r _ { W } \geq k .$ . Gathering these bounds and using $2 m + 1 \geq 2$ , we get

$$
( 2 m + 1 ) \delta _ { V } + 2 \delta _ { W } \geq 2 ( p _ { V } + r _ { V } ) + 2 ( p _ { W } + r _ { W } ) \geq 2 ( p _ { V } + p _ { W } + k ) \geq 2 \delta _ { \mathrm { c c } } ,
$$

which is (25). We have that $u [ R _ { 0 } ^ { ( 1 ) } ; R _ { 0 } ^ { ( 2 ) } ] = | E _ { \oplus } |$ and $b _ { R _ { 0 } ^ { ( 1 ) } , R _ { 0 } ^ { ( 2 ) } } ^ { \prime } = - u [ R _ { 0 } ^ { ( 1 ) } ; R _ { 0 } ^ { ( 2 ) } ] / 4 + ( 2 m + 1 ) | V _ { \oplus } | / 4 +$ $| W _ { \oplus } | / 2 - \csc ( G _ { \oplus } ) / 2 + \csc _ { \mathrm { u } } / 2$ . We conclude that

$$
\left| \mathbb { E } \left[ \Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right] \right| \leq ( 3 2 D ^ { 3 } ) ^ { \mathrm { c c _ { u } } } \left( 2 \sqrt { m } \frac { \Delta } { d ^ { 1 / 4 } } \right) ^ { \left| E _ { \oplus } \right| } \left( \frac { 1 } { d } \right) ^ { ( 2 m + 1 ) | V _ { \oplus } | / 4 + | W _ { \oplus } | / 2 - \mathrm { c c } ( G _ { \oplus } ) / 2 + \mathrm { c c _ { u } } / 2 ) }
$$

Let $T _ { 0 } ^ { ( 1 ) } \subset { \cal R } _ { 0 } ^ { ( 1 ) }$ (resp. $T _ { 0 } ^ { ( 2 ) } \subset { \cal R } _ { 0 } ^ { ( 2 ) } )$ be the indices of the connected components such that all their edges are pruned in the construction of $E _ { \oplus }$ . As a consequence, we have

$$
u [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] - u [ R _ { 0 } ^ { ( 1 ) } ; R _ { 0 } ^ { ( 2 ) } ] \ge | T _ { 0 } ^ { ( 1 ) } \setminus R ^ { ( 1 ) } | + | T _ { 0 } ^ { ( 2 ) } \setminus R ^ { ( 2 ) } | \enspace .
$$

Since we assume that 8m $D \Delta \leq d ^ { 1 / 4 }$ –see Hypothesis $( \mathcal { H } _ { s i g n a l } ) -$ , we get

$$
\begin{array} { r l } & { \displaystyle \sum _ { R ^ { ( 1 ) } \subset { \cal R } _ { 0 } ^ { ( 1 ) } \subset { \cal R } _ { 0 } ^ { ( 2 ) } \subset { \cal R } _ { 0 } ^ { ( 2 ) } } \left( 2 \sqrt { m } \displaystyle \frac { \Delta } { d ^ { 1 / 4 } } \right) ^ { u [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] - u [ R _ { 0 } ^ { ( 1 ) } ; R _ { 0 } ^ { ( 2 ) } ] } } \\ & { \displaystyle \leq \sum _ { T ^ { ( 1 ) } \subset { \cal T } _ { 0 } ^ { ( 1 ) } \atop T ^ { ( 2 ) } \subset { \cal T } _ { 0 } ^ { ( 2 ) } R ^ { ( 1 ) } \subset { \cal R } _ { 0 } ^ { ( 1 ) } \setminus { \cal T } _ { 0 } ^ { ( 1 ) } } \left( 2 \sqrt { m } \displaystyle \frac { \Delta } { d ^ { 1 / 4 } } \right) ^ { | T ^ { ( 1 ) } | + | T ^ { ( 2 ) } | } } \\ & { \displaystyle \leq 2 | R _ { 0 } ^ { ( 1 ) } \setminus { \cal T } _ { 0 } ^ { ( 1 ) } | + | R _ { 0 } ^ { ( 2 ) } \setminus { \cal T } _ { 0 } ^ { ( 2 ) } | \displaystyle \sum _ { t = 0 } ^ { \infty } ( 2 D ) ^ { t } \left( 2 \sqrt { m } \displaystyle \frac { \Delta } { d ^ { 1 / 4 } } \right) ^ { t } } \\ & { \displaystyle \leq 2 \cdot 2 | { \cal E } _ { 0 } | \ . } \end{array}
$$

We conclude that

$$
\left| \mathbb { E } \left[ \Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right] \right| \leq 2 ( 3 2 D ^ { 3 } ) ^ { \mathrm { c c _ { \mathfrak { \alpha } } } } \left( 4 m \frac { \Delta } { d ^ { 1 / 4 } } \right) ^ { | E _ { \mathfrak { \varnothing } } | } \left( \frac { 1 } { d } \right) ^ { ( 2 m + 1 ) | V _ { \mathfrak { \varnothing } } | / 4 + | W _ { \mathfrak { \varnothing } } | / 2 - \mathrm { c c } ( G _ { \mathfrak { \varnothing } } ) / 2 + \mathrm { c c } _ { \mathfrak { \varnothing } } / 2 ) }
$$

We have established Case (b) in Proposition 1.

It remains to deal with Case (c). We build on the previous arguments, up to minor modifications. Since $E _ { \oplus } = \emptyset$ , we have $u [ R _ { 0 } ^ { ( 1 ) } ; R _ { 0 } ^ { ( 2 ) } ] = 0$ . Starting back from (20) and arguing as in Case (b), we get

$$
\begin{array} { r l } { \Big | \mathbb { E } \left[ \Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } - 1 \right] \Big | = \Bigg | } & { \displaystyle \sum _ { R ^ { ( 1 ) } \subset { R _ { 0 } ^ { ( 1 ) } } \atop 8 : \pi ^ { ( 2 ) } \subset { R _ { 0 } ^ { ( 2 ) } } } \left( \frac { \Delta } { \sqrt { d } } \right) ^ { u [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] } \omega [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \displaystyle \quad \quad \quad \quad \quad \quad \quad \times . . R ^ { ( 1 ) } \neq R _ { 0 } ^ { ( 2 ) } \textnormal { o r } R ^ { ( 2 ) } \neq R _ { 0 } ^ { ( 2 ) } } \\ & { \leq \Big | \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \bigg ( 2 m ^ { 1 / 2 } \frac { \Delta } { d ^ { 1 / 4 } } \Big ) ^ { u [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] } \Big | \ . } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad R ^ { ( 2 ) } \subset { R _ { 0 } ^ { ( 1 ) } } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \times . ~ R ^ { ( 2 ) } \neq R _ { 0 } ^ { ( 2 ) } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \times . ~ R ^ { ( 1 ) } \neq R _ { 0 } ^ { ( 2 ) } \neq R _ { 0 } ^ { ( 2 ) } } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad  \end{array}
$$

Since $E _ { \oplus } = \emptyset$ , we have

$$
u [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \ge | R _ { 0 } ^ { ( 1 ) } \setminus R ^ { ( 1 ) } | + | R _ { 0 } ^ { ( 2 ) } \setminus R ^ { ( 2 ) } | \ .
$$

Since we assume that 8m $D \Delta \leq d ^ { 1 / 4 }$ , we conclude

$$
\left| \mathbb { E } \left[ \Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } - 1 \right] \right| \leq \sum _ { r = 1 } ^ { \infty } \left( \frac { 4 m D \Delta } { d ^ { 1 / 4 } } \right) ^ { r } \leq \frac { 8 m D \Delta } { d ^ { 1 / 4 } } \ .
$$

It remains to establish the result (d). The proof is very close to that of (b) and we only sketch the diferences. Define $G _ { \oplus } ^ { T _ { 2 } }$ as the graph associated to the symmetric diference of the labeled graphs $( G ^ { ( 1 ) } , \pi ^ { ( 1 ) } )$ and $( G ^ { ( 2 ) } , \pi ^ { ( 2 ) } )$ from which we have removed the connected components in $[ \mathrm { c c } ^ { ( 2 ) } ] \backslash T _ { 2 }$ . Observe that $G _ { \oplus } ^ { T _ { 2 } }$ is made of a collection of $\csc ^ { ( 1 ) } + | T _ { 2 } |$ non-trivial connected components. Similarly to (20), we have

$$
\begin{array} { r l } & { \mathbb { E } \left[ P _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right] = \displaystyle \left( \frac { \Delta } { \sqrt { d } } \right) ^ { | { \cal E } ^ { ( 1 ) } | + | { \cal E } ^ { ( 2 ) } | } \mathbb { E } _ { \sigma } [ L ] \ ; } \\ & { \qquad L : = \displaystyle { \bf 1 } \left\{ { \cal A } [ G ^ { ( 1 ) } ] \right\} \prod _ { i \in [ \mathrm { c c } ^ { ( 2 ) } ] } \left( { \bf 1 } \left\{ { \cal A } [ G _ { i } ^ { ( 2 ) } ] \right\} - \mathbb { P } _ { \sigma } \left[ { \cal A } [ G _ { i } ^ { ( 2 ) } ] \right] \right) \ . } \end{array}\tag{26}
$$

Since ${ \cal A } [ G ^ { ( 1 ) } ]$ and $\boldsymbol { \mathcal { A } } [ G _ { i } ^ { ( 2 ) } ]$ depend only on the respective edge sets, we may replace $G ^ { ( 1 ) }$ and each $G _ { i } ^ { ( 2 ) }$ by their edge-induced subgraphs (removing any isolated node) before applying Lemma 11, exactly as for the pieces $G _ { 0 } ^ { * } , \ldots , G _ { l } ^ { * }$ in the proof of Lemma 12 below; this is consistent with $| V _ { \oplus } | , | W _ { \oplus } |$ below already denoting the node counts induced by $E ^ { ( 1 ) } \cup E ^ { ( 2 ) }$ . Analogously to Lemma 12, we get

$$
| \mathbb { E } _ { \sigma } \left[ L \right] | \leq ( 3 2 D ^ { 3 } ) ^ { \operatorname { c c } ( G ^ { ( 2 ) } ) } ( 4 m ) ^ { \frac { | E _ { \oplus } | } { 2 } } \left( \frac { 1 } { d } \right) ^ { - | E _ { \oplus } | / 4 + ( 2 m + 1 ) | V _ { \oplus } | / 4 + | W _ { \oplus } | / 2 - \operatorname { c c } ( G ^ { ( 1 ) } ) / 2 }
$$

Together with (26), we arrive at the desired result.

Proof of Lemma 12. Note that the labeled graphs $G _ { \oplus } ^ { R ^ { ( 1 ) } ; R ^ { ( 2 ) } } , \pi ^ { ( 1 ) } [ G _ { i } ^ { ( 1 ) } ]$ with $i \in [ \mathrm { c c } ^ { ( 1 ) } ] \setminus R _ { 0 } ^ { ( 1 ) }$ and $\pi ^ { ( 2 ) } [ G _ { i } ^ { ( 2 ) } ]$ with $i \in [ \mathrm { c c } ^ { ( 2 ) } ] \setminus R _ { 0 } ^ { ( 2 ) }$ do not have any node in common. We define $( G ^ { * } , \pi _ { 1 } ^ { * } , \pi _ { 2 } ^ { * } )$ as the labeled graph that contains $G _ { \oplus } ^ { R ^ { ( 1 ) } ; R ^ { ( 2 ) } }$ as well as the graphs $\pi ^ { ( 1 ) } [ G _ { i } ^ { ( 1 ) } ]$ with $i \in [ \mathrm { c c } ^ { ( 1 ) } ] \setminus R _ { 0 } ^ { ( 1 ) }$ and $\pi ^ { ( 2 ) } [ G _ { i } ^ { ( 2 ) } ]$ with $i \in [ \csc ^ { ( 2 ) } ] \setminus R _ { 0 } ^ { ( 2 ) }$ Write $l = \vert [ \mathrm { c c } ^ { ( 1 ) } ] \ : \backslash \ : R _ { 0 } ^ { ( 1 ) } \vert + \vert [ \mathrm { c c } ^ { ( 2 ) } ] \ : \backslash \ : R _ { 0 } ^ { ( 2 ) } \vert$ . For simplicity, we write $G _ { i } ^ { * } , i = 1 , \ldots , l$ for those l connected components. We also write $G _ { 0 } ^ { * }$ for the subgraph corresponding to $G _ { \oplus } ^ { R ^ { ( 1 ) } ; R ^ { ( 2 ) } }$ . Note that $G _ { 0 } ^ { * }$ is possibly empty especially $\mathrm { i f } | R ^ { ( 1 ) } | + | R ^ { ( 2 ) } | = 0$ . With this notation, we have

$$
L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] = \mathbf { 1 } \left\{ \mathcal { A } [ G _ { 0 } ^ { * } ] \right\} \prod _ { i = 1 } ^ { l } \left[ \mathbf { 1 } \left\{ \mathcal { A } [ G _ { i } ^ { * } ] \right\} - \mathbb { E } _ { \sigma } \left[ \mathbf { 1 } \left\{ \mathcal { A } [ G _ { i } ^ { * } ] \right\} \right] \right]\tag{27}
$$

The case $l = 0$ . Assume first that $l = 0$ , that is $R _ { 0 } ^ { ( 1 ) } = [ \mathrm { c c } ^ { ( 1 ) } ]$ and $R _ { 0 } ^ { ( 2 ) } = [ \mathrm { c c } ^ { ( 2 ) } ]$ (equivalently $\mathrm { c c } _ { \mathrm { u } } = 0 )$ . Then the product in (27) is empty, so that ${ \cal L } [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] = \mathbf { 1 } \bar { \{ { A [ G _ { 0 } ^ { * } ] } \} }$ and

$$
\left| \mathbb { E } _ { \sigma } \left[ L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right] \right| = \mathbb { P } _ { \sigma } \left[ \boldsymbol { A } [ G _ { 0 } ^ { * } ] \right] \ ,
$$

with $G ^ { * } = G _ { 0 } ^ { * } = G _ { \oplus } ^ { R ^ { ( 1 ) } ; R ^ { ( 2 ) } }$ . Writing $| E _ { 0 } ^ { * } | , | V _ { 0 } ^ { * } | , | W _ { 0 } ^ { * } |$ for the numbers of edges and of (non-isolated) nodes of $G _ { 0 } ^ { * }$ , the exponent in the statement of the lemma reduces, when $l = 0$ , to

$$
b _ { R ^ { ( 1 ) } , R ^ { ( 2 ) } } = - \frac { | E _ { 0 } ^ { * } | } { 4 } + ( 2 m + 1 ) \frac { | V _ { 0 } ^ { * } | } { 4 } - \frac { \csc ( G _ { 0 } ^ { * } ) } { 2 } + \frac { | W _ { 0 } ^ { * } | } { 2 } = b ( G _ { 0 } ^ { * } ) ,
$$

by the definition (17) of $b ( \cdot )$ , while the prefactor $( 3 2 D ^ { 3 } ) ^ { \mathrm { { c c } _ { u } } }$ equals one. Hence, it sufices to prove that $\begin{array} { r } { \tilde { \mathbb { P } } _ { \sigma } [ \boldsymbol { A } [ G _ { 0 } ^ { * } ] ] \leq ( 4 m ) ^ { | \dot { E } _ { 0 } ^ { * } | / 2 } ( 1 / \dot { d } ) ^ { \dot { b } ( G _ { 0 } ^ { * } ) } } \end{array}$ . If $G _ { 0 } ^ { * }$ is empty, the system $( S _ { G } )$ is empty, so that $\mathbb { P } _ { \sigma } [ \mathcal { A } [ G _ { 0 } ^ { * } ] ] = 1$ , whereas the above right-hand side is also equal to one since $| E _ { 0 } ^ { * } | = | V _ { 0 } ^ { * } | = | W _ { 0 } ^ { * } | = \csc ( G _ { 0 } ^ { * } ) = 0$ . If $G _ { 0 } ^ { * }$ is non-empty,

then it contains no isolated node (being induced by its edge set), all its degrees are even (a symmetric diference of graphs with even degrees has even degrees), and $| E _ { 0 } ^ { * } | \le | E ^ { ( 1 ) } | + | E ^ { ( 2 ) } | \le 2 D$ . If $G _ { 0 } ^ { * }$ satisfies Condition $( \mathcal { W } _ { G } )$ , Lemma 11 gives

$$
\mathbb { P } _ { \sigma } \left[ { \cal A } [ G _ { 0 } ^ { * } ] \right] \le 2 ( 2 m ) ^ { | E _ { 0 } ^ { * } | / 2 } \left( \frac { 1 } { d } \right) ^ { b ( G _ { 0 } ^ { * } ) } \le ( 4 m ) ^ { | E _ { 0 } ^ { * } | / 2 } \left( \frac { 1 } { d } \right) ^ { b ( G _ { 0 } ^ { * } ) } \ ,
$$

where we used $2 ~ \leq ~ 2 ^ { | E _ { 0 } ^ { * } | / 2 }$ , since $| E _ { 0 } ^ { * } | \ge 4$ (any non-empty connected component of a graph with even degrees contains at least 4 edges, see the end of this proof). If $G _ { 0 } ^ { * }$ does not satisfy Condition $( \mathcal { W } _ { G } )$ , then $\mathbb { P } _ { \sigma } [ \mathcal { A } [ G _ { 0 } ^ { * } ] ] = 0$ by Lemma 10 –see the Remark following Lemma 11– and the bound is trivial. This settles the case $l = 0$

In the remainder of this proof, we therefore assume that $l \geq 1$

Write $W _ { 0 } ^ { * } , W _ { 1 } ^ { * } , \ldots , W _ { l } ^ { * }$ for the sets of nodes corresponding to $G _ { 0 } ^ { * } , \ldots , G _ { l } ^ { * }$ . Note that these sets are disjoint. Define $\tilde { \mathbb { P } } _ { \sigma }$ as the distribution such that for $i = 0 , \ldots , l , \sigma ( \pi _ { 2 } ( w ) )$ with $w \in W _ { i } ^ { * }$ are sampled uniformly without replacement on $[ d ] ,$ but the $\sigma ( \pi _ { 2 } ( W _ { i } ^ { * } ) ) ^ { \prime } s$ are sampled independently. We write $\dot { \mathbb { E } } _ { \sigma }$ for the corresponding expectation. This distribution is chosen in such a way that $\mathbb { P } _ { \sigma } \left[ \boldsymbol { \mathcal { A } } [ G _ { i } ^ { * } ] \right] = \tilde { \mathbb { P } } _ { \sigma } \left[ \boldsymbol { \mathcal { A } } [ G _ { i } ^ { * } ] \right]$ and the events $A [ G _ { i } ^ { * } ] ^ { * }$ s are independent under ${ \tilde { P } } .$

Define the event $\mathcal { E }$ on σ such that, on this event, the sets $\sigma ( \pi _ { 2 } ( W _ { i } ^ { * } ) )$ ’s are disjoint for $i = 0 , \ldots , l$ . We have

$$
\mathbb { E } _ { \sigma } \left[ \mathbf { 1 } _ { \mathcal { E } } L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right] = \mathbb { \tilde { P } } _ { \sigma } ^ { - 1 } [ \mathcal { E } ] \tilde { \mathbb { E } } _ { \sigma } \left[ \mathbf { 1 } _ { \mathcal { E } } L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right] ~ .
$$

By a union bound over the $\binom { l + 1 } { 2 }$ pairs $i \neq j$ , each contributing at most $| W _ { i } ^ { * } | | W _ { j } ^ { * } | / d$ to $\widetilde { \mathbb { P } } _ { \sigma } [ \mathcal { E } ^ { c } ] \ ( 1 / d$ being the probability that two elements of $[ d ]$ , drawn independently and uniformly for $\ * { W _ { i } ^ { * } }$ and $W _ { j } ^ { * }$ , coincide), we get $\tilde { \mathbb { P } } _ { \sigma } [ \mathcal { E } ^ { c } ] \le | W ^ { * } | ^ { 2 } / ( 2 d ) \le 2 D ^ { 2 } / d$ , using $| W ^ { * } | \leq 2 D$ . Since $2 ^ { 8 } D ^ { 4 } \leq d$ and $D \geq 2$ , we have $\bar { 2 } D ^ { 2 } / d \geq 2 ^ { - 7 } D ^ { - 2 } \leq$ $1 / 2$ , which gives $\tilde { \mathbb { P } } _ { \sigma } [ \mathcal { E } ] \ge 1 / 2$ . Since, under $\tilde { \mathbb { P } } _ { \sigma }$ , the expectation of ${ \cal L } ^ { R ^ { ( 1 ) } ; R ^ { ( 2 ) } }$ is zero, we obtain

$$
\tilde { \mathbb { E } } _ { \sigma } \left[ \mathbf { 1 } _ { \varepsilon } L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right] = - \tilde { \mathbb { E } } _ { \sigma } \left[ \mathbf { 1 } _ { \varepsilon ^ { c } } L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right]
$$

In the sequel, $\mathcal { L }$ stands for a partition of $[ 0 : l ]$ . Given any element $L \in { \mathcal { L } }$ , we define the event $\boldsymbol { B } _ { L }$ such that for any $a _ { 0 }$ and $a ^ { \prime }$ in $L ,$ there exist $a _ { 1 } , \dots , a _ { s }$ in $L$ with $\sigma ( \pi _ { 2 } ( W _ { a _ { i } } ^ { * } ) ) \cap \sigma ( \pi _ { 2 } ( W _ { a _ { i + 1 } } ^ { * } ) ) \neq \emptyset$ for $i = 0 , \ldots , s - 1$ and $\sigma ( \pi _ { 2 } ( W _ { a _ { s } } ^ { * } ) ) \cap \sigma ( \pi _ { 2 } ( W _ { a ^ { \prime } } ^ { * } ) ) \neq \emptyset$ . If $L$ is a singleton, this event is of probability one. Since the connecting chain is required to remain inside $L$ , the event $\boldsymbol { B } _ { L }$ only involves the variables $\sigma ( \pi _ { 2 } ( W _ { a } ^ { * } ) )$ with $a \in L ;$ in particular, under $\tilde { \mathbb { P } } _ { \sigma } , \mathbf { 1 } _ { B _ { I } }$ is independent of the randomness attached to the blocks $W _ { a } ^ { * }$ with a /∈ L. Also, we define the event $\mathcal { E } _ { \mathcal { L } }$ such that for any a and $a ^ { \prime }$ in two groups $L$ and $L ^ { \prime }$ of $\mathcal { L } .$ we have $\sigma ( \pi _ { 2 } ( W _ { a } ^ { * } ) ) \cap \sigma ( \pi _ { 2 } ( W _ { a ^ { \prime } } ^ { * } ) ) = \emptyset$ Equipped with these definitions, we deduce that

$$
\tilde { \mathbb { E } } _ { \sigma } \left[ { \bf 1 } _ { \mathcal { E } } L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right] = - \sum _ { \mathcal { L } } \tilde { \mathbb { E } } _ { \sigma } \left[ { \bf 1 } _ { \mathcal { E } _ { \mathcal { L } } } \prod _ { L \in \mathcal { L } } { \bf 1 } _ { \mathcal { B } _ { L } } L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right] \ ,
$$

where the sum runs over all partitions $\mathcal { L }$ of $[ 0 : l ]$ that are distinct from $\{ \{ 0 \} , \{ 1 \} , \dots , \{ l \} \}$ . If the partition $\mathcal { L }$ contains a singleton distinct from {0}, then, we deduce from the definition of $L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ]$ that

$$
\tilde { \mathbb { E } } _ { \sigma } \left[ \prod _ { L \in \mathcal { L } } \mathbf { 1 } _ { B _ { L } } L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right] = 0
$$

so that

$$
\tilde { \mathbb { E } } _ { \sigma } \left[ \mathbf { 1 } _ { \mathcal { E } _ { \mathcal { L } } } \prod _ { L \in \mathcal { L } } \mathbf { 1 } _ { B _ { L } } L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right] = - \tilde { \mathbb { E } } _ { \sigma } \left[ \mathbf { 1 } _ { \mathcal { E } _ { \mathcal { L } } ^ { c } } \prod _ { L \in \mathcal { L } } \mathbf { 1 } _ { B _ { L } } L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right] \ .
$$

Hence, by a standard recursion, we arrive at

$$
\left. \tilde { \mathbb { E } } _ { \sigma } \left[ \mathbf { 1 } _ { \mathcal { E } } L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right] \right. \leq \sum _ { o = 1 } ^ { l + 1 } \sum _ { \mathcal { L } ^ { ( 1 ) } , \mathcal { L } ^ { ( 2 ) } , \ldots , \mathcal { L } ^ { ( o ) } } \mathbb { \tilde { E } } \left[ \mathbf { 1 } _ { \mathcal { E } _ { \mathcal { L } ^ { ( o ) } } } \prod _ { L \in \mathcal { L } ^ { ( o ) } } \mathbf { 1 } _ { \mathcal { B } _ { L } } \left. L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right. \right] \ ,
$$

where $\mathcal { L } ^ { ( a + 1 ) }$ is strictly coarser than $\mathcal { L } ^ { ( a ) }$ and where $\mathcal { L } ^ { ( o ) }$ is the only partition in the sequence that does not contain any singleton, with the possible exception of $\{ 0 \}$ . Write $\pmb { \mathcal { L } } ^ { * }$ for the collection of all partitions L that do not contain any singleton distinct from $\{ 0 \}$

Lemma 13. Given $\mathcal { L }$ in $\pmb { \mathcal { L } } ^ { * }$ , there exist no more than $l ^ { 2 l } 2 ^ { l }$ sequences $\mathcal { L } ^ { ( 1 ) } , \ldots , \mathcal { L } ^ { ( o ) }$ such that $\mathcal { L } ^ { ( o ) } = \mathcal { L }$

We conclude that

$$
\left| { \tilde { \mathbb { E } } } _ { \sigma } \left[ { \bf 1 } _ { \mathcal { E } } L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right] \right| \le l ^ { 2 l } 2 \sum _ { \mathcal { L } \in { \mathcal { L } } ^ { * } } { \tilde { \mathbb { E } } } _ { \sigma } \left[ { \bf 1 } _ { \mathcal { E } _ { \mathcal { L } } } \prod _ { L \in { \mathcal { L } } } { \bf 1 } _ { \mathcal { B } _ { L } } \left. L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right. \right] \ .\tag{28}
$$

In the following, we write $s$ for a partition of $W ^ { * }$ such that, for $i = 0 , \ldots , l .$ , elements of $W _ { i } ^ { * }$ are not in the same group. Let us define the event $\mathcal { C } _ { S }$ such that, for any w and $w ^ { \prime } \in W ^ { * }$ , we have $\sigma ( \pi _ { 2 } ( w ) ) = \sigma ( \pi _ { 2 } ( w ^ { \prime } ) )$ if and only if w and $w ^ { \prime }$ are in the same group in S. Given $\mathcal { L } \in \mathcal { L } ^ { * }$ , define $\pmb { S } [ \mathcal { L } ]$ as the collection of partitions S that are compatible with $\mathcal { E } _ { \mathcal { L } } \cap _ { L \in \mathcal { L } } B _ { L }$ . Finally, we define $\pmb { S } ^ { * } = \cup _ { \mathcal { L } \in \mathcal { L } ^ { * } } \pmb { S } [ \mathcal { L } ]$ . We then deduce from (27) and (28) that

$$
\begin{array} { r l } & { \left. \widetilde { \mathbb { E } } _ { \sigma } \left[ \mathbf { 1 } _ { \mathcal { E } } L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right] \right. \le l ^ { 2 l } 2 \underset { S \in \mathcal { S } ^ { * } } { \sum } \widetilde { \mathbb { E } } _ { \sigma } \left[ \mathbf { 1 } _ { \mathcal { C } _ { \delta } } \left. L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right. \right] } \\ & { \qquad \le l ^ { 2 l } 2 \underset { S \in \mathcal { S } ^ { * } } { \sum } \underset { 0 \in T \subset [ 0 ; l ] } { \sum } \widetilde { \mathbb { E } } _ { \sigma } \left[ \mathbf { 1 } _ { \mathcal { C } _ { \delta } } \prod \mathbf { 1 } \left\{ A [ G _ { i } ^ { * } ] \right\} \right] \underset { i \in [ 0 ; l ] \times T } { \prod } \mathbb { E } _ { \sigma } \left[ \mathbf { 1 } \left\{ A [ G _ { i } ^ { * } ] \right\} \right] \ . } \end{array}
$$

Define $\begin{array} { r } { r ( S ) = \sum _ { S \in { \cal S } } [ | S | - 1 ] } \end{array}$ . Since $| W ^ { * } | \leq 2 D$ and $d \geq 4 D$ (in particular, $d \geq 2 | W ^ { * } | )$ , we have

$$
\tilde { \mathbb { P } } _ { \sigma } \left[ \mathcal { C } _ { S } \right] \le \left( \frac { 2 } { d } \right) ^ { r ( S ) } \ .
$$

Also, given $T \subset [ 0 : l ]$ and a partition $s ,$ we define the graph $G _ { T , S } ^ { * }$ as the union of the graphs $G _ { i } ^ { * }$ for $i \in T$ where we also identify nodes w of those graphs that belong to the same group of the partition S. This leads us to

$$
\tilde { \mathbb { E } } _ { \sigma } \left[ \mathbf { 1 } _ { \mathcal { C } _ { S } } \prod _ { i \in T } \mathbf { 1 } \left\{ \boldsymbol { A } [ G _ { i } ^ { * } ] \right\} \right] = \tilde { \mathbb { P } } _ { \sigma } \left[ \mathcal { C } _ { S } \right] \mathbb { E } _ { \sigma } \left[ \mathbf { 1 } \left\{ \boldsymbol { A } [ G _ { T , S } ^ { * } ] \right\} \right]
$$

Putting everything together, and coming back from $\tilde { \mathbb { E } } _ { \sigma } [ \mathbf { 1 } _ { \varepsilon } L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] ]$ to $\mathbb { E } _ { \sigma } [ L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] ]$ thanks to the identity $\mathbb { E } _ { \sigma } \big [ \check { L } \big [ R ^ { ( 1 ) } ; R ^ { ( \tilde { 2 } ) } \big ] \big ] = \mathbb { E } _ { \sigma } \big [ \mathbf { 1 } _ { \mathcal { E } } L \big [ R ^ { ( \tilde { 1 } ) } ; R ^ { ( 2 ) } \big ] \big ] = \tilde { \mathbb { P } } _ { \sigma } ^ { - 1 } \big [ \check { \mathcal { E } } \big ] \tilde { \mathbb { E } } _ { \sigma } \big [ \mathbf { 1 } _ { \mathcal { E } } L \big [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } \big ] \big ]$ –the event E occurring $\mathbb { P } _ { \sigma ^ { - } }$ almost surely, since under $\mathbb { P } _ { \sigma }$ the map $w \mapsto \sigma ( \pi _ { 2 } ( w ) )$ is injective on $W ^ { * } -$ together with $\tilde { \mathbb { P } } _ { \sigma } ^ { - 1 } [ \mathcal { E } ] \le 2$ , which accounts for the extra factor 2 in the constant below, we arrive at

$$
\left. \mathbb { E } _ { \sigma } \left[ L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right] \right. \leq l ^ { 2 l } 2 ^ { 2 l + 1 } \sum _ { S \in S ^ { \star } } \left( \frac { 2 } { d } \right) ^ { r ( S ) } \operatorname* { m a x } _ { 0 \in T \subset [ 0 ; l ] } \mathbb { E } _ { \sigma } \left[ \mathbf { 1 } \left\{ { \mathcal A } [ G _ { T , S } ^ { \ast } ] \right\} \right] \prod _ { i \in [ 0 ; l ] \setminus T } \mathbb { E } _ { \sigma } \left[ \mathbf { 1 } \left\{ { \mathcal A } [ G _ { i } ^ { \ast } ] \right\} \right] \ .
$$

We apply Lemmas 10 and 11 to the graph $G _ { T , S } ^ { * }$ and to each of the pieces $G _ { i } ^ { * }$ with $i \in [ 0 : l ] \backslash T ;$ since $0 \in T$ , these pieces are connected components of $\dot { G } ^ { ( 1 ) }$ or of $G ^ { ( 2 ) }$ and are in particular non-empty. All these graphs have even degrees (a symmetric diference of graphs with even degrees has even degrees, connected components of $G ^ { ( 1 ) }$ and $G ^ { ( 2 ) }$ have even degrees, and identifying nodes of $W$ adds up their degrees), they do not contain any isolated node, and their edge sets are disjoint with total cardinality $| E ^ { * } | \leq | \check { E } ^ { ( 1 ) } | + | E ^ { ( \bar { 2 } ) } | \leq$ 2D. If one of them does not satisfy Condition $( \mathcal { W } _ { G } )$ , then Lemma 10 yields that the corresponding probability, hence the whole term indexed by $( T , S )$ , vanishes – see the Remark following Lemma 11 – so that we may assume that all of them satisfy Condition $( \mathcal { W } _ { G } )$ . Each of these graphs is non-empty, except possibly $G _ { T , S } ^ { * }$ when $T = \{ 0 \}$ and $G _ { 0 } ^ { * }$ is empty; a non-empty one has at least 4 edges (any non-empty connected component of a graph with even degrees does, see the end of this proof; and $G _ { T , S } ^ { * }$ contains a non-empty piece as soon as it is non-empty). Consequently, each application of Lemma 11 to a non-empty such graph H contributes a factor $2 ( 2 m ) ^ { \lceil E \rceil / 2 } \leq ( \mathring { 4 } m ) ^ { \lceil E \rceil / 2 }$ , since $2 \leq 2 ^ { | E ( H ) | / 2 }$ , whereas an empty $G _ { T , S } ^ { * }$ simply contributes the factor $\mathbb { P } _ { \sigma } [ \mathcal { A } [ G _ { T , \mathcal { S } } ^ { * } ] ] = 1$ (the System $( S _ { G } )$ being then empty), in accordance with $b ( \emptyset ) = 0$ . Multiplying these bounds and using that $\csc ( G _ { i } ^ { * } ) = 1$ for $i \in [ 0 : l ] \backslash T$ yields the prefactor $( 4 m ) ^ { | E ^ { * } | / 2 }$ and the exponent $\kappa _ { T , S }$ below, where $\begin{array} { r } { | E ^ { * } | = \sum _ { i = 0 } ^ { l } | E _ { i } ^ { * } | } \end{array}$

We obtain

$$
\left| \mathbb { E } _ { \sigma } \left[ L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right] \right| \le l ^ { 2 l } 2 ^ { 2 l + 1 } ( 4 m ) ^ { | { \cal E } ^ { \ast } | / 2 } \sum _ { S \in { \cal S } ^ { \ast } } \left( \frac { 2 } { d } \right) ^ { r ( S ) } \operatorname* { m a x } _ { 0 \in { \cal T } \subset [ 0 : l ] } \left( \frac { 1 } { d } \right) ^ { \kappa _ { { \cal T } , S } } \ ,
$$

where we define

$$
\kappa _ { T , S } : = \frac { | W _ { T , S } ^ { * } | + \sum _ { i \notin T } | W _ { i } ^ { * } | } { 2 } - \frac { | E ^ { * } | - ( 2 m + 1 ) | V ^ { * } | } { 4 } - \frac { \csc ( G _ { T , S } ^ { * } ) + ( l + 1 ) - | T | } { 2 } \enspace .\tag{29}
$$

Lemma 14. For any $S \in { \pmb S } ^ { * }$ and any $T \subset [ 0 : l ]$ with $0 \in T _ { \ O }$ , we have

$$
\kappa _ { T , S } + r ( S ) \geq - \frac { \left[ | E ^ { * } | - ( 2 m + 1 ) | V ^ { * } | \right] } { 4 } - \frac { \csc ( G _ { 0 } ^ { * } ) } { 2 } + \frac { | W ^ { * } | } { 2 } + \frac { ( r ( S ) - l / 2 ) _ { + } } { 2 } \ .
$$

Summing over all possible partitions $s ,$ , we conclude that

$$
\begin{array}{c} \left| \mathbb { E } _ { \sigma } \left[ L [ R ^ { ( 1 ) } ; R ^ { ( 2 ) } ] \right] \right| \leq l ^ { 2 l } 2 ^ { 2 l + 1 } ( 4 m ) ^ { | E ^ { * } | / 2 } \left( \frac { 1 } { d } \right) ^ { - \frac { [ | E ^ { * } | - ( 2 m + 1 ) | V ^ { * } | ] } { 4 } - \frac { \mathrm { c } ( E _ { 0 } ^ { * } ) } { 2 } + \frac { | W ^ { * } | } { 2 } } \sum _ { r = \left[ l ^ { \prime } 2 \right] } ^ { \infty } \int ^ { r } \left( \frac { 1 } { d } \right) ^ { ( r - l / 2 ) / 2 } | W ^ { * } | ^ { 2 r } d t ,  \\ { \leq l ^ { 2 l } 2 ^ { 3 l + 1 } ( 2 D ) ^ { l } ( 4 m ) ^ { | E ^ { * } | / 2 } \left( \frac { 1 } { d } \right) ^ { - \frac { [ | E ^ { * } | - ( 2 m + 1 ) | V ^ { * } | ] } { 4 } - \frac { \mathrm { c } ( E _ { 0 } ^ { * } ) } { 2 } + \frac { | W ^ { * } | } { 2 } } \sum _ { r = \left[ l ^ { \prime } 2 \right] } ^ { \infty } \left( \frac { 2 ^ { 6 } D ^ { 4 } } { d } \right) ^ { ( r - l / 2 ) / 2 } } \\ { \leq 2 ( 3 2 l ^ { 2 } D ) ^ { l } ( 4 m ) ^ { | E ^ { * } | / 2 } \left( \frac { 1 } { d } \right) ^ { - \frac { [ | E ^ { * } | - ( 2 m + 1 ) | V ^ { * } | ] } { 4 } - \frac { \mathrm { c } ( E _ { 0 } ^ { * } ) } { 2 } + \frac { | W ^ { * } | } { 2 } } , } \end{array}
$$

where we used that $| W ^ { * } | \leq 2 D$ in the second line, and that $2 ^ { 8 } D ^ { 4 } \leq d$ in the last line – indeed, writing $q : = 2 ^ { 6 } D ^ { 4 } / d \leq 1 / 4$ , the geometric-type sum satisfies $\begin{array} { r } { \sum _ { r \ge \lceil l / 2 \rceil } q ^ { ( r - l / 2 ) / 2 } = q ^ { \epsilon } / ( 1 - \sqrt { q } ) \le 1 / ( 1 - 1 / 2 ) = 2 . } \end{array}$ where $\epsilon = 0 \mathrm { i f } l$ is even and $\epsilon = 1 / 4$ if l is odd –.

To conclude, we note that $l \leq D / 2$ as any non-empty connected component of a graph with even degrees contains at least 4 edges. Hence,

$$
2 \left( 3 2 l ^ { 2 } D \right) ^ { l } \leq 2 \left( 8 D ^ { 3 } \right) ^ { l } \leq 4 ^ { l } ( 8 D ^ { 3 } ) ^ { l } = ( 3 2 D ^ { 3 } ) ^ { l } = ( 3 2 D ^ { 3 } ) ^ { \mathrm { c c } _ { \mathrm { u } } } \ ,
$$

where we used that $l = \vert [ \mathrm { c c } ^ { ( 1 ) } ] \setminus R _ { 0 } ^ { ( 1 ) } \vert + \vert [ \mathrm { c c } ^ { ( 2 ) } ] \setminus R _ { 0 } ^ { ( 2 ) } \vert = \mathrm { c c } _ { \mathrm { u } }$ , the connected components outside $R _ { 0 } ^ { ( 1 ) }$ and $R _ { 0 } ^ { ( 2 ) }$ being exactly the unmatched ones. Finally, the exponent appearing above is nothing but $b _ { R ^ { ( 1 ) } , R ^ { ( 2 ) } }$ , by the definition of $| E ^ { * } | , | V ^ { * } | , | W ^ { * } |$ and since $\csc ( G _ { 0 } ^ { * } ) = \csc ( { G _ { \oplus } ^ { R } } ^ { ( 1 ) } ; { R ^ { ( 2 ) } } )$ . This concludes the proof.

Proof of Lemma 13. Let us justify this counting bound (recall that $l \geq 1$ , as otherwise no partition $ { \mathcal { L } } \neq \hat { 0 }$ exists). Write $\hat { 0 } = \{ \{ 0 \} , \{ 1 \} , \ldots , \{ l \} \}$ for the finest partition of $[ 0 : l ]$ and order the partitions of $[ 0 : l ]$ by refinement: $\pi \preceq \pi ^ { \prime }$ if every group of π is contained in a group of $\pi ^ { \prime } ,$ and $\pi \prec \pi ^ { \prime }$ if moreover $\pi \neq \pi ^ { \prime }$ . Since every $\mathcal { L } ^ { ( a ) }$ satisfies $\hat { 0 } \prec \mathcal { L } ^ { ( 1 ) } \prec \cdot \cdot \prec \mathcal { L } ^ { ( o ) } = \mathcal { L }$ , it sufices to bound, for an arbitrary partition $ { \mathcal { L } } \neq \hat { 0 }$ of $[ 0 : l ] .$ , the number of such strictly increasing sequences of arbitrary length $o \geq 1$ ending at $\mathcal { L } ;$ forgetting the additional constraint that $\mathcal { L } ^ { ( 1 ) } , \ldots , \mathcal { L } ^ { ( o - 1 ) }$ contain a singleton distinct from {0} can only increase this number. Set $\nu = l + 1 = | [ 0 : l ] |$ , let $b _ { 1 } , \ldots , b _ { k }$ be the sizes of the $k \geq 1$ groups of ${ \mathcal { L } } ,$ and let

$$
\chi : = \nu - k = \sum _ { i = 1 } ^ { k } ( b _ { i } - 1 ) \geq 1 \ .
$$

Endow the partitions with the rank function $\pi \mapsto \nu - | \pi | ;$ then $\pi \prec \pi ^ { \prime }$ is a covering relation — that ${ \mathrm { i s } } , \pi ^ { \prime }$ is obtained from π by merging exactly two of its groups — if and only if the rank increases by exactly one, and the partitions π with ${ \hat { 0 } } \preceq \pi \preceq { \mathcal { L } }$ are exactly those obtained by partitioning each group of $\mathcal { L }$ separately, their rank ranging over $\{ 0 , 1 , \ldots , \chi \}$

Call saturated a chain $\ddot { 0 } = \tau _ { 0 } \prec \tau _ { 1 } \prec \cdot \cdot \cdot \prec \tau _ { \chi } = \mathcal { L }$ all of whose steps are covering relations. Inside a fixed group of $\mathcal { L }$ of size $b ,$ one goes from the b singletons to the whole group in $b - 1$ merges and, as long as $j$ groups remain, $\textstyle { \binom { j } { 2 } }$ pairs may be merged; there are therefore $\textstyle \prod _ { i = 2 } ^ { b } { \binom { j } { 2 } } = { \hat { b } } ! ( b - 1 ) ! / 2 ^ { b - 1 }$ such saturated chains. $\mathrm { A s }$ merges occurring in diferent groups of $\mathcal { L }$ can be freely interleaved, and as the k resulting sequences, of respective lengths $b _ { 1 } - 1 , \dotsc , b _ { k } - 1$ , can be interleaved into a single sequence of $\chi$ merges in $\begin{array} { r } { \chi ! / \prod _ { i } ( b _ { i } - 1 ) ! } \end{array}$ ways, the number of saturated chains from $\hat { 0 }$ to $\mathcal { L }$ equals

$$
{ \frac { \chi ! } { \prod _ { i = 1 } ^ { k } ( b _ { i } - 1 ) ! } } \prod _ { i = 1 } ^ { k } { \frac { b _ { i } ! ( b _ { i } - 1 ) ! } { 2 ^ { b _ { i } - 1 } } } = { \frac { \chi ! } { 2 ^ { \chi } } } \prod _ { i = 1 } ^ { k } b _ { i } ! ~ .\tag{30}
$$

Consider now any chain $\hat { 0 } = \mathcal { L } ^ { ( 0 ) } \prec \mathcal { L } ^ { ( 1 ) } \prec \dots \prec \mathcal { L } ^ { ( o ) } = \mathcal { L }$ . Since two comparable partitions are always related by successive merges of two groups, each step $\mathcal { L } ^ { ( a - 1 ) } \prec \mathcal { L } ^ { ( a ) }$ can be refined into a chain of covering relations, so that the whole chain admits at least one refinement into a saturated chain $( \tau _ { 0 } , \dots , \tau _ { \chi } )$ from $\hat { 0 }$ to $\mathcal { L } ;$ fix one such refinement for each chain. The ranks $R : = \{ \nu - | \mathcal { L } ^ { ( a ) } | , a = 1 , \ldots , o - 1 \}$ of the intermediate partitions form a subset of $\{ 1 , \ldots , \chi - 1 \}$ , and the pair $( ( \tau _ { 0 } , \dots , \tau _ { \chi } ) , R )$ recovers the original chain, which is nothing but $( \tau _ { r } ) _ { r \in R }$ followed by $\tau _ { \chi } = \mathcal { L }$ . This map being injective and there being $2 ^ { \chi - 1 }$ possible subsets $R ,$ we deduce from (30) that the number of chains ending at $\mathcal { L }$ is at most

$$
\frac { \chi ! } { 2 ^ { \chi } } \left( \prod _ { i = 1 } ^ { k } b _ { i } ! \right) 2 ^ { \chi - 1 } = \frac { \chi ! } { 2 } \prod _ { i = 1 } ^ { k } b _ { i } ! \leq \frac { l ! ( l + 1 ) ! } { 2 } ,
$$

where we used $\chi = \nu - k \leq \nu - 1 = l$ together with $\begin{array} { r } { \prod _ { i = 1 } ^ { k } b _ { i } ! \leq \nu ! = ( l + 1 ) ! } \end{array}$ , the latter because $\nu ! / \prod _ { i } b _ { i } ! =$ $\binom { \nu } { b _ { 1 } , \ldots , b _ { k } }$ is a positive integer. Finally, $j ! \leq j ^ { j }$ for any integer $j \geq 1$ , and $l + 1 \le 2 ^ { l }$ for any integer $l \geq 1$ , so that

$$
\frac { l ! ( l + 1 ) ! } { 2 } = \frac { ( l + 1 ) ( l ! ) ^ { 2 } } { 2 } \leq \frac { l + 1 } { 2 } l ^ { 2 l } \leq 2 ^ { l - 1 } l ^ { 2 l } \leq l ^ { 2 l } 2 ^ { l } \ ,
$$

which is the announced bound.

Proof of Lemma $1 \llcorner$ Fix any $T \subset [ 0 : l ]$ with $0 \in T$ . Define $r ( S , T ) = \lvert W ^ { * } \rvert - \lvert W _ { T , S } ^ { * } \rvert - \sum _ { i \notin T } \lvert W _ { i } ^ { * } \rvert$ . Define

$$
A _ { T } : = r ( \mathcal { S } ) + \frac { - r ( \mathcal { S } , T ) + \csc ( G _ { 0 } ^ { * } ) - ( l + 1 ) + | T | - \csc ( G _ { T , \mathcal { S } } ^ { * } ) } { 2 }
$$

It follows from the definition (29) of $\kappa _ { T , S }$ that

$$
\kappa _ { T , \mathcal { S } } + r ( \mathcal { S } ) = - \frac { [ | E ^ { * } | - ( 2 m + 1 ) | V ^ { * } | ] } { 4 } - \frac { \csc ( G _ { 0 } ^ { * } ) } { 2 } + \frac { | W ^ { * } | } { 2 } + A _ { T } \ ,
$$

and we only have to establish that $A _ { T } \ \geq \ ( r ( S ) - l / 2 ) _ { + } / 2 \ \geq \ 0$ for all such $T .$ . First, we establish that $A _ { T } \geq ( r ( S ) - l / 2 ) _ { + } / 2 \geq 0$ for $T = [ 0 : l ]$ . Indeed, for the latter we have

$$
2 A _ { [ 0 : l ] } = r ( { \cal S } ) + \csc ( { G } _ { 0 } ^ { \ast } ) - \csc ( G _ { [ 0 : l ] , { \cal S } } ^ { \ast } ) .
$$

Any partition $\boldsymbol { S } \in \boldsymbol { S } ^ { * }$ satisfies the following property:

for any $i \in [ l ]$ , at least one node of $W _ { i } ^ { * }$ belongs to the same group of $s$ as a node of $W ^ { * } \setminus W _ { i } ^ { * } . \neg ( \mathcal { P } _ { S } )$

Indeed, let $\mathcal { L } \in \mathcal { L } ^ { * }$ be such that $S \in { \pmb S } [ { \mathcal { L } } ]$ , and let $i \in [ l ]$ . Since $\mathcal { L }$ contains no singleton distinct from $\{ 0 \}$ the group $L \in { \mathcal { L } }$ containing i contains some $\boldsymbol { a } ^ { \prime } \neq i$ . On the event $\boldsymbol { B } _ { L }$ , the chain connecting i to $a ^ { \prime }$ has to leave i at some step, so that $\sigma ( \pi _ { 2 } ( W _ { i } ^ { * } ) ) \cap \sigma ( \pi _ { 2 } ( W _ { a } ^ { * } ) ) \neq \emptyset$ for some $a \in L \backslash \{ i \}$ ; by compatibility with $\mathcal { C } _ { S }$ , the two corresponding nodes, which lie respectively in $W _ { i } ^ { * }$ and in $W _ { a } ^ { * } \subset W ^ { * } \setminus W _ { i } ^ { * }$ , belong to the same group of ${ \mathcal { S } } .$ As a consequence of $( \mathcal { P } _ { S } )$ , at least l distinct nodes of $W ^ { * }$ – one in each $\dot { W } _ { i } ^ { * } , i \in [ l ]$ – belong to groups of $s$ of size at least 2. Since $| S | - 1 \ge | S | / 2$ for any such group $S _ { ; }$ we get $\begin{array} { r } { r ( S ) = \sum _ { S \in { \cal S } } ( | S | - 1 ) \ge l / 2 } \end{array}$

Let us also check that $\csc ( G _ { [ 0 : l ] , S } ^ { * } ) \leq l / 2 + \csc ( G _ { 0 } ^ { * } )$ . Call atoms the connected components of the disjoint union $G _ { 0 } ^ { * } \sqcup G _ { 1 } ^ { * } \sqcup \cdot \cdot \cdot \sqcup G _ { l } ^ { * } .$ : since each $G _ { i } ^ { * }$ with $i \in [ l ]$ is connected, the atoms are the $\mathrm { c c } ( G _ { 0 } ^ { * } )$ connected components of $G _ { 0 } ^ { * }$ together with the l graphs $G _ { i } ^ { * } , i \in [ l ]$ . Let Γ be the graph whose nodes are the atoms and where two distinct atoms are joined by an edge whenever some group of $s$ contains a node of the first atom and a node of the second one. As $G _ { [ 0 : l ] , s } ^ { * }$ is obtained from the above disjoint union by identifying the nodes lying in a common group of $s$ , two atoms belong to the same connected component of $G _ { [ 0 : l ] , s } ^ { * }$ if and only if they belong to the same connected component of Γ. Since every atom carries at least one edge, cc $( G _ { [ 0 : l ] , S } ^ { * } )$ is therefore exactly the number $p$ of connected components of Γ –isolated nodes of Γ being counted here. Besides, by Property $( \mathcal { P } _ { S } )$ , every atom $G _ { i } ^ { * }$ with $i \in [ l ]$ has degree at least one in $\Gamma _ { \mathrm { {  } } }$ , since a node of $W _ { i } ^ { * }$ belongs to the same group of $s$ as a node of another atom. Now, given a connected component $C$ of $\Gamma _ { \varpi }$ , write $g _ { C }$ for the number of its nodes that are connected components of $G _ { 0 } ^ { * }$ and $l _ { C }$ for the number of its nodes of the form $G _ { i } ^ { * }$ with $i \in [ l ]$ , so that $\begin{array} { r } { \sum _ { C } g _ { C } = \csc ( G _ { 0 } ^ { * } ) } \end{array}$ and $\Sigma _ { C } l _ { C } = l ;$ , the sums running over the $p$ connected components of Γ. If $g _ { C } \geq 1$ , then $1 \leq g _ { C } \leq g _ { C } + l _ { C } / 2$ . If $g _ { C } = 0$ , then all the nodes of $C$ are of the form $G _ { i } ^ { * }$ with $i \in [ l ] ;$ any of them has a neighbor in Γ, which necessarily lies in $C ,$ so that $l _ { C } \geq 2$ and again $1 \leq l _ { C } / 2 = g _ { C } + l _ { C } / 2$ . Summing these inequalities over the connected components of $\Gamma _ { \pm }$ , we get

$$
\csc ( G _ { [ 0 : l ] , S } ^ { * } ) = p \leq \sum _ { C } \left( g _ { C } + { \frac { l _ { C } } { 2 } } \right) = \csc ( G _ { 0 } ^ { * } ) + { \frac { l } { 2 } } ~ .
$$

This leads to $A _ { \lceil 0 : l \rceil } \geq ( r ( S ) - l / 2 ) _ { + } / 2 \geq 0$ . Finally, we show that $A _ { T } \ \geq \ A _ { [ 0 : l ] }$ for any $T \subset [ 0 : l ]$ with $0 \in T$ . To see this, observe that $G _ { [ 0 : l ] , s } ^ { * }$ is built from $G _ { T , S } ^ { * }$ and from the connected components $G _ { i } ^ { * }$ with $i \not \in T$ by identifying nodes. Since each node identification among the $r ( S ) - r ( S , T )$ ones merges at most two connected components, we arrive at

$$
r ( \mathcal { S } ) - r ( \mathcal { S } , T ) \geq + \mathrm { c c } ( G _ { T , \mathcal { S } } ^ { * } ) + ( l + 1 - | T | ) - \mathrm { c c } ( G _ { [ 0 : l ] , \mathcal { S } } ^ { * } )\tag{31}
$$

In turn, this implies that

$$
2 ( A _ { T } - A _ { [ 0 : l ] } ) = r ( \mathcal { S } ) - r ( \mathcal { S } , T ) + \csc ( G _ { [ 0 : l ] , \mathcal { S } } ^ { \ast } ) - \csc ( G _ { T , \mathcal { S } } ^ { \ast } ) - ( l + 1 - | T | )
$$

is non-negative.

## B Proofs for the number of solutions of $( S _ { G } )$

## B.1 Proof of Lemma 10

Assume that there exists $V ^ { \prime } \subset V$ such that $| \oplus _ { v \in V ^ { \prime } } \mathcal { N } ( v ) | \in [ 1 : 2 m ]$ - indeed, since we assumed that the degree of all nodes is even, the symmetric diferences of neighborhoods have even cardinality, so that we have excluded the case $| \oplus _ { v \in V ^ { \prime } } \mathcal { N } ( v ) | = 2 m + 1$

$$
\mathbb { P } _ { \sigma } [ \mathcal { A } _ { G } ] \le \mathbb { P } _ { \sigma } \Bigg [ \sum _ { v \in V ^ { \prime } } \sum _ { w : w \sim v } \phi ^ { ( 2 s - 1 ) } ( \sigma ( \pi _ { 2 } ( w ) ) ) = 0 , \forall 1 \le s \le m \Bigg ] ,
$$

since if the system of equations $( S _ { G } )$ is satisfied, then any sum over v in a subset of V of the associated Equation (14) needs to be equal to 0, for any $s \leq m$

Note that

$$
\sum _ { v \in V ^ { \prime } } \sum _ { w : w \sim v } \phi ^ { ( 2 s - 1 ) } ( \sigma ( \pi _ { 2 } ( w ) ) ) = \sum _ { w \in \Phi _ { v \in V ^ { \prime } } \backslash N ( v ) } \phi ^ { ( 2 s - 1 ) } ( \sigma ( \pi _ { 2 } ( w ) ) ) ,
$$

by definition of the sum operation in F<sup>u</sup> - as being the coordinatewise sum in $\mathbb { N } ,$ modulo 2. Hence, since $| \oplus _ { v \in V ^ { \prime } } \mathcal { N } ( v ) | \leq$ 2m and since the $\sigma ( \pi _ { 2 } ( w ) )$ ) are all distinct for $w \in W$ , we deduce from Lemma 24 that the system of equations

$$
\sum _ { v \in V ^ { \prime } } \sum _ { w : w \sim v } \phi ^ { ( 2 s - 1 ) } ( \sigma ( \pi _ { 2 } ( w ) ) ) = 0 , \qquad \forall 1 \leq s \leq m ,
$$

has no solution whatever the injection $\sigma$ is. This completes the proof.

## B.2 Proof of Lemma 11

This proof is based on standard arguments in algebraic geometry over finite fields. Recall that $d + 1 = 2 ^ { u }$ is a power of 2, and that, throughout, a column $w \in W$ is identified with the value $\sigma ( \pi _ { 2 } ( w ) ) \in \{ 1 , \ldots , d \}$ , itself identified with a nonzero element of $\mathbb { F } _ { d + 1 } = \mathbb { F } _ { 2 ^ { u } }$ as in Section $_ { 3 ; }$ we also use that G has no isolated node by assumption. Since the Zariski topology, dimension, and degree of an algebraic set are only meaningfully defined over an algebraically closed field, we view all the algebraic sets introduced below $( \mathbb { X } , \mathbb { U } , \overline { { \mathbb { U } } } ,$ and their irreducible components) as varieties over the algebraic closure $\overline { { \mathbb { F } } } _ { d + 1 } \mathrm { ~ o f ~ } \mathbb { F } _ { d + 1 }$ . For such a variety $\mathbb { Y } \subset \overline { { \mathbb { F } } } _ { d + 1 } ^ { W } ,$ we write |Y| for its number of $\mathbb { F } _ { d + 1 }$ -rational points, that is $| \mathbb { Y } \cap \mathbb { F } _ { d + 1 } ^ { W } | \ -$ this is the only quantity we are ultimately interested in, while dim and deg below refer to the geometric invariants of Y as a variety over $\overline { { \mathbb { F } } } _ { d + 1 }$

Define the afine algebraic set $\mathbb { X } \subset \overline { { \mathbb { F } } } _ { d + 1 } ^ { W }$ defined by the homogeneous equation system

$$
\sum _ { w : w \sim _ { G } v } x _ { w } ^ { 2 l - 1 } = 0 \ ,\tag{32}
$$

for $l = 1 , \ldots , m$ and for $v \in V$ . Note that this system is exactly the system $( S _ { G } )$ , read in $\mathbb { F } _ { d + 1 }$ rather than in $\mathbb { F } _ { 2 } ^ { u }$ : since $\phi ^ { ( 2 l - 1 ) } ( v ) = \psi ( v ^ { 2 l - 1 } )$ and since ψ is an isomorphism of additive groups from $\mathbb { F } _ { 2 ^ { u } }$ onto $\mathbb { F } _ { 2 } ^ { u }$ , we have, for any $v \in V$ and any $l \leq m$ ，

$$
\sum _ { w : w \sim _ { G } v } \phi ^ { ( 2 l - 1 ) } ( \sigma ( \pi _ { 2 } ( w ) ) ) = \psi \Big ( \sum _ { w : w \sim _ { G } v } \sigma ( \pi _ { 2 } ( w ) ) ^ { 2 l - 1 } \Big ) ,
$$

which vanishes in F<sup>u</sup> if and only if $\begin{array} { r } { \sum _ { w : w \sim _ { G } v } \sigma ( \pi _ { 2 } ( w ) ) ^ { 2 l - 1 } = 0 } \end{array}$ in $\mathbb { F } _ { d + 1 }$ . We define $\mathbb { U } \subset \mathbb { X }$ as the Zariski open subset such that the $x _ { w }$ are distinct and nonzero – matching the fact that $\sigma ( \pi _ { 2 } ( w ) ) \in \{ 1 , \ldots , d \}$ is identified with a nonzero element of $\mathbb { F } _ { d + 1 }$ . That is, $\mathbb { U } = \mathbb { X } \backslash \mathbb { Z }$ , where $\mathbb { Z }$ is the union of hyperplanes $x _ { w } - x _ { w ^ { \prime } } = 0$ for all w $\neq w ^ { \prime }$ , and of hyperplanes $x _ { w } = 0$ for all w. By definition of $( S _ { G } )$ and of the event $\mathcal { A } [ G ]$ , we have

$$
\mathbb { P } _ { \sigma } [ { \cal A } _ { G } ] = \frac { ( d - \vert W \vert ) ! } { d ! } \vert \mathbb { U } \vert \ ,\tag{33}
$$

where |U| is, as agreed above, the number of $\mathbb { F } _ { d + 1 }$ -rational points of $\mathbb { U } - \mathrm { n a m e l y }$ , the number of injective tuples $( x _ { w } ) _ { w \in W }$ with values in the d nonzero elements of $\mathbb { F } _ { d + 1 }$ satisfying the system (32). Thus, we look for an upper bound on this number of rational points of the set U. The proof proceeds in two steps. First, we bound the dimension of the geometric afine variety and then, we apply the B´ezout inequality.

The set U is a Zariski-open subset of $\mathbb { X } ,$ which is in general reducible and not closed. For this reason, we introduce $\overline { { \mathbb { U } } }$ as the Zariski closure of U in $\overline { { \mathbb { F } } } _ { d + 1 } ^ { W }$ . Then U is an afine algebraic set. Let V denote an irreducible component of U<sup>¯</sup>. Since U is dense in U, V ∩ U is a non-empty open subset of V. For any point $x \ = \ ( x _ { w } ) _ { w \in W } \ \in \ \mathbb { V } \cap \ \mathbb { U }$ , we consider the Jacobian matrix $\mathcal { T } _ { x } ~ \in ~ \bar { \mathbb { F } } _ { d + 1 } ^ { m \cdot | V | \times | W | }$ associated to the system (32) obtained by stacking vertically the blocks $\mathcal { I } _ { x , 1 } , \ldots , \mathcal { I } _ { x , m }$ , so that its rows are indexed by the pairs $( i , v ) \in [ m ] \times V$ , where

$$
( \mathcal { I } _ { x , i } ) _ { v , w } = x _ { w } ^ { 2 i - 2 } \mathbf { 1 } _ { w \in \mathcal { N } ( v ) } \ ,\tag{34}
$$

for all $v \in V$ and $w \in W$ . Here, we used that the derivative of $x _ { w } ^ { 2 i - 1 }$ with respect to $x _ { w }$ is $( 2 i - 1 ) x _ { w } ^ { 2 i - 2 } = x _ { w } ^ { 2 i - 2 }$ since $2 i - 1$ is odd and $\mathbb { F } _ { d + 1 }$ has characteristic 2.

Lemma 15. For any such $x \in \mathbb { V } \cap \mathbb { U }$ , we have

$$
r a n k ( \mathcal { T } _ { x } ) \geq b ( G )
$$

Then, since the equations (32) vanish on $\mathbb { V } ,$ the tangent space of $\mathbb { V }$ at x is contained in the kernel of the Jacobian matrix $\mathcal { I } _ { x }$ , and its dimension is at least dim(V) –see, e.g., [45, Chapter 2, Section 1]. Hence, by Lemma 15, and since $\operatorname { r a n k } ( \mathcal { I } _ { x } )$ is an integer,

$$
\mathrm { d i m } ( \mathbb { V } ) \leq | W | - \mathrm { r a n k } ( \mathcal { T } _ { x } ) \leq | W | - \lceil b ( G ) \rceil .
$$

Since the dimension of an algebraic set is the maximum of the dimensions of its components, we conclude

$$
\dim ( { \overline { { \mathbb { U } } } } ) \leq | W | - \lceil b ( G ) \rceil .
$$

In what follows, we use the cumulative degree of [31, Section 1], which we simply denote by deg: the degree deg(Y) of a possibly reducible afine algebraic set Y is the sum of the degrees of all the irreducible components of $\mathbb { Y }$ – and not only of those of maximal dimension –. We apply the B´ezout bound for the number of $\mathbb { F } _ { d + 1 } { \mathrm { - r a t i o n a l } }$ points on an afine algebraic set –see Theorem 2.1 in [31]– to U:

$$
| \mathbb { U } | \leq | \overline { { \mathbb { U } } } | \leq \deg ( \overline { { \mathbb { U } } } ) \cdot ( d + 1 ) ^ { \dim ( \overline { { \mathbb { U } } } ) } .
$$

Furthermore, since U is a union of irreducible components of the algebraic set $\mathbb { X } ,$ its degree is bounded by the degree of $ \mathbb { X } .$ , and more specifically by the product of the degrees of the equations (32) –see e.g. Theorem 1.1 and Equation (3) in [31].

$$
\deg ( \overline { { \mathbb { U } } } ) \leq \deg ( \mathbb { X } ) \leq \prod _ { j = 1 } ^ { m } ( 2 j - 1 ) ^ { | V | } .
$$

Since all nodes $v \in V$ have a degree larger than or equal to 2m $+ 2 -$ this follows from Condition $( \mathcal { W } _ { G } )$ applied with $V _ { 0 } = \{ v \}$ , which gives $\deg _ { G } ( v ) = | \mathcal { N } ( v ) | \notin [ 1 : 2 m + 1 ] ,$ , together with $\deg _ { G } ( v ) \geq 1$ , since G contains no isolated node –, we have $\begin{array} { r } { | E | = \sum _ { v \in V } \deg _ { G } ( v ) \geq ( 2 m + 2 ) | V | } \end{array}$ , and, in particular, $m | V | \leq | E | / 2$ We conclude – using that $2 j - 1 \le 2 m$ for any $j \leq m$ , so that $\begin{array} { r } { \prod _ { j = 1 } ^ { m } ( 2 j - 1 ) ^ { | V | } \leq ( 2 m ) ^ { m | V | } \leq ( 2 m ) ^ { | E | / 2 } } \end{array}$ 2 where the last inequality relies on $m | V | \leq | E | / 2$ and $2 m \geq 1 - \mathrm { { t h a t } }$

$$
| \overline { { \mathbb { U } } } | \leq ( 2 m ) ^ { | E | / 2 } ( d + 1 ) ^ { | W | - \lceil b ( G ) \rceil } .
$$

Coming back to (33), we now convert this bound in terms of $d + 1$ into a bound in terms of $d .$ Write $\begin{array} { r } { ( d + 1 ) ^ { | W | - \lceil b ( G ) \rceil } = d ^ { | W | - \lceil b ( G ) \rceil } \left( 1 + \frac { 1 } { d } \right) ^ { | W | - \lceil b ( G ) \rceil } } \end{array}$ . Since each node of W (resp. each of the $\csc ( G )$ non-trivial connected components) is incident to at least one edge, we have $| W | \leq | E |$ and $\operatorname { c c } ( G ) \leq | W | \leq | E | ;$ together with $| E | \le 2 D$ , this gives $| W | \le 2 D$ and $\operatorname { c c } ( G ) \leq 2 D$ . By definition (17) of $b ( G )$ and since $m \geq 0$ and $\left\lceil b ( G ) \right\rceil \geq b ( G )$

$$
| W | - \lceil b ( G ) \rceil \leq \ | W | - b ( G ) = \frac { | W | } { 2 } + \frac { | E | } { 4 } - \frac { | V | } { 2 } \Big ( m + \frac 1 2 \Big ) + \frac { \csc ( G ) } { 2 } \leq \frac { | W | } { 2 } + \frac { | E | } { 4 } + \frac { \csc ( G ) } { 2 } \leq \frac { 5 | E | } { 4 } \leq \frac { 5 D } { 2 } \ .
$$

Since $D \geq 2$ and $d \ge 2 ^ { 8 } D ^ { 4 }$ , we have $\begin{array} { r } { \frac { 5 D } { 2 d } \leq \frac { 5 D } { 2 \cdot 2 ^ { 8 } D ^ { 4 } } = \frac { 5 } { 2 ^ { 9 } D ^ { 3 } } \leq 2 ^ { - 8 } } \end{array}$ , so that $\begin{array} { r } { \left( 1 + \frac { 1 } { d } \right) ^ { | W | - \lceil b ( G ) \rceil } \leq \left( 1 + \frac { 1 } { d } \right) ^ { 5 D / 2 } \leq } \end{array}$ $e ^ { 5 D / ( 2 d ) } \leq e ^ { 2 ^ { - 8 } }$ . On the other hand, using $- \ln ( 1 - x ) \leq 2 x$ for $x \le 1 / 2$ together with $| W | \le 2 D$ and $| W | / d \le 1 / 2$ (which holds since $d \ge 2 ^ { 8 } D ^ { 4 } \ge 4 D \ge 2 | W | )$ , we get $\begin{array} { r } { \left( 1 - \frac { | W | } { d } \right) ^ { - | W | } \le e ^ { 2 | W | ^ { 2 } / d } \le e ^ { 8 D ^ { 2 } / d } \le } \end{array}$ $e ^ { 8 / ( 2 ^ { 8 } D ^ { 2 } ) } \ \leq \ e ^ { 2 ^ { - 7 } }$ , again using $D \ \geq \ 2$ . Multiplying the two bounds, $\begin{array} { r } { \left( 1 + \frac { 1 } { d } \right) ^ { | W | - \lceil b ( G ) \rceil } \left( 1 - \frac { | W | } { d } \right) ^ { - | W | } \leq } \end{array}$ $e ^ { 2 ^ { - 8 } + 2 ^ { - 7 } } = e ^ { 3 / 2 5 6 } \leq 2$ . We conclude that

$$
\mathbb { P } _ { \sigma } [ A _ { G } ] = \frac { ( d - | W | ) ! } { d ! } | \mathfrak { U } | \le ( 1 + \frac { 1 } { d } ) ^ { | W | - ( b ( G ) | } ( 1 - \frac { | W | } { d } ) ^ { - | W | } ( 2 m ) ^ { | E | / 2 } ( \frac { 1 } { d } ) ^ { [ b ( G ) | } \le 2 ( 2 m ) ^ { | E | / 2 } ( \frac { 1 } { d } ) ^ { [ b ( G ) | } \le 2 m ) ^ { | E | / 2 } )
$$

which establishes (18) and concludes the proof of Lemma 11.

Proof of Lemma 15. Given a graph $G = ( V , W , E )$ , a subset $\overline { { W } } \subset W$ satisfies the G-enumeration property if there exists an ordering $( v ^ { ( 1 ) } , \ldots , v ^ { ( | V | ) } )$ of the nodes in V such that, upon recursively defining $\overline { { W } } ^ { ( 0 ) } = \overline { { W } }$ and $\overline { { W } } ^ { ( t ) } = \overline { { W } } ^ { ( t - 1 ) } \setminus N ( v ^ { ( t ) } )$ , we have

$$
| \overline { { { W } } } ^ { ( t - 1 ) } \cap \mathcal { N } ( v ^ { ( t ) } ) | \leq m \qquad \forall t = 1 , \ldots , | V | .\tag{35}
$$

Lemma 16. Consider $\overline { { W } } \subset W$ that satisfies the G-enumeration property and such that every $w \in \overline { { W } }$ has at least one neighbor in $V \mathrm { ~ - ~ } w h i c h$ is in particular the case as soon as $G$ does not contain any isolated node –. Then, we have

$$
r a n k ( \mathcal { T } _ { x } ) \geq | \overline { { W } } |
$$

Hence, the key technical step of the proof amounts to finding $\overline { W }$ with $| { \overline { { W } } } | \geq b ( G )$ that satisfies the G-enumeration property. We prove this in several steps.

Lemma 17. Consider any simple connected bipartite graph $G = ( V , W , E )$ such that:

(i) Each node $w \in W$ satisfies $d e g _ { G } ( w ) = 2$

(ii) Each node $v \in V$ satisfies $d e g _ { G } ( v ) \geq m + 1$

Then, there exists $\overline { { W } } \subset W$ that satisfies the G-enumeration property and such that

$$
| \overline { { { W } } } | \geq \frac { | V | } { 2 } ( m + 1 / 2 ) - \frac { 1 } { 2 } .
$$

Consider a connected graph $G = ( V , W , E )$ . We say that a group $V ^ { * } \subset V$ is of type A if there exists a node $w ^ { * } \in W$ with d $\ g _ { G } ( w ^ { * } ) \geq 4$ such that

(i) Aside from $w ^ { \ast } , V ^ { \ast }$ is only adjacent to nodes w of degree 2

(ii) $w ^ { * }$ is connected to exactly two nodes in $V ^ { * }$

(iii) If we remove $w ^ { * }$ from $G ,$ then $V ^ { * }$ remains connected but is disconnected from $V \backslash V ^ { * }$

Lemma 18. Consider a connected graph $G = ( V , W , E )$ that satisfies Condition $\textit { 1 } ( \mathcal { W } _ { G } )$ and that does not contain any isolated node. $I f G$ does not contain any group of type A, then there exists $\overline { { W } } \subset W$ that satisfies the G-enumeration property and such that

$$
| \overline { { { W } } } | \geq \frac { | W | } { 2 } - \frac { | E | } { 4 } + \frac { | V | } { 2 } ( m + 1 / 2 ) - \frac { 1 } { 2 } .\tag{36}
$$

Lemma 19. Consider a graph $G = ( V , W , E )$ that satisfies Condition $\textit { 1 } ( \mathcal { W } _ { G } )$ and that does not contain any isolated node. There exists $\overline { { W } } \subset W$ that satisfies the G-enumeration property and such that

$$
| \overline { { { W } } } | \geq \frac { | W | } { 2 } - \frac { | E | } { 4 } + \frac { | V | } { 2 } ( m + 1 / 2 ) - \frac { \csc ( G ) } { 2 } .\tag{37}
$$

The result readily follows by combining Lemma 16 with the last lemma, the additional assumption of Lemma 16 being satisfied since the graph G of Lemma 11 does not contain any isolated node. □

Proof of Lemma 16. Consider a $\overline { { W } } \subset W$ that satisfies the G-enumeration property and let $( v ^ { ( 1 ) } , \ldots , v ^ { ( | V | ) } )$ be an ordering of the nodes that satisfies (35).

We shall prove that the columns of the matrix $\mathcal { I } _ { x }$ indexed by $\overline { W }$ are linearly independent. Given $w \in W$ we write $( \mathcal { I } _ { x } ) _ { \cdot , w }$ for the w-th column of $\mathcal { I } _ { x }$ . Consider any $( \lambda _ { w } ) _ { w \in \overline { { W } } } \in \overline { { \mathbb { F } } } _ { d + 1 } ^ { \overline { { W } } }$ such that

$$
\sum _ { w \in \overline { { \boldsymbol { W } } } } \lambda _ { w } ( \mathcal { I } _ { x } ) _ { \cdot , w } = 0 \mathrm { ~ . ~ }
$$

We shall prove by induction on t that $\lambda _ { w } = 0$ for all $w \in \overline { { W } } \backslash \overline { { W } } ^ { ( t ) }$ . Since we have $\overline { { W } } ^ { ( \vert V \vert ) } = \emptyset$ – indeed, $\overline { { W } } ^ { ( | V | ) } = \overline { { W } } \setminus \cup _ { t = 1 } ^ { | V | } \mathcal { N } ( v ^ { ( t ) } )$ is the set of the nodes $w \in \overline { { W } }$ that have no neighbor in V, and there is no such node by assumption –, this will imply that $\left( \lambda _ { w } \right) = 0$ and therefore that rank $( \mathcal { T } _ { x } ) \geq | \overline { { W } } |$

The property is obviously true for $t = 0$ . Let us assume that it holds at some $t \geq 0$ . This implies that

$$
\sum _ { w \in \overline { { \boldsymbol { W } } } ^ { ( t ) } } \lambda _ { w } ( \mathcal { T } _ { \boldsymbol { x } } ) _ { \cdot , w } = 0 \mathrm { ~ . ~ }\tag{38}
$$

Define $r = | \mathcal { N } ( v ^ { ( t + 1 ) } ) \cap \overline { { W } } ^ { ( t ) } |$ . By Condition (35), we have $r \in [ 0 : m ]$ . If $r = 0 .$ , we do not have anything to prove and we assume henceforth $r \in [ 1 : m ]$ . Let us focus on the rows indexed $( \mathcal { I } _ { x , i } ) _ { v ^ { ( t + 1 ) } }$ <sub>·</sub> for $i = 1 , \ldots , r .$ The expression (34) of the Jacobian matrix and Condition (38) imply that, for all $i = 1 , \ldots , r ,$ we have

$$
\sum _ { w \in \mathcal { N } ( v ^ { ( t + 1 ) } ) \cap \overline { { W } } ^ { ( t ) } } \lambda _ { w } ( x _ { w } ^ { 2 } ) ^ { i - 1 } = 0 .
$$

Since x $\in \mathbb { V } \cap \mathbb { U } .$ , the $x _ { w }$ are distinct, and since squaring is injective in characteristic $2 \ : ( \mathrm { i f } \ x _ { w } ^ { 2 } = x _ { w ^ { \prime } } ^ { 2 }$ then $( x _ { w } + x _ { w ^ { \prime } } ) ^ { 2 } = x _ { w } ^ { 2 } + x _ { w ^ { \prime } } ^ { 2 } = 0$ , so $x _ { w } = x _ { w ^ { \prime } } )$ , the $x _ { w } ^ { 2 }$ are distinct too; the corresponding Vandermonde matrix is thus invertible. This implies that $\lambda _ { w } = 0$ for all $w \in \mathcal { N } ( v ^ { ( t + 1 ) } ) \cap \overline { { W } } ^ { ( t ) }$ . Hence, we have $\lambda _ { w } = 0$ for all $w \in \overline { { W } } \setminus \overline { { W } } ^ { ( t + 1 ) }$ □

## B.3 Proof of technical lemmas

Proof of Lemma 17. Note first that $\vert V \vert \ge 2 \colon \mathrm { i f } \ V = \{ v \}$ , no node of W could have two distinct neighbors, so that $W = \emptyset$ , which contradicts de $\operatorname { g } _ { G } ( v ) \geq m + 1$ . We prove the lemma by strong induction on $| V |$

Step 0: pruning. We iteratively remove nodes of W (together with their two edges) as long as the resulting graph remains connected and all the nodes of V keep a degree at least $m + 1$ . Write $G _ { \mathrm { p } } = ( V , W _ { \mathrm { p } } , E _ { \mathrm { p } } )$ for the pruned graph. If $\overline { { W } } \subset W _ { \mathfrak { p } }$ satisfies the $G _ { \mathrm { p } } .$ -enumeration property, then it also satisfies the G-enumeration property, since $\mathcal { N } _ { G } ( v ) \cap A = \mathcal { N } _ { G _ { \mathrm { p } } } ( v ) \cap A$ A for all $v \in V$ and all $A \subset W _ { \mathfrak { p } }$ . As the claimed lower bound on $| \overline { { W } } |$ does not depend on W, it is therefore enough to prove the lemma for $G _ { \mathrm { p } } ,$ which we still denote $G = ( V , W , E )$ This graph is simple, connected, bipartite, satisfies (i) and (ii), and, by maximality of the pruning, it also satisfies

(iii) for any $w \in W$ such that the graph $G - w$ obtained by removing w is connected, one of the two neighbors of w has degree exactly $m + 1$

Step 1: a node of degree $m + 1$ on a cycle. All the nodes of G have degree at least 2, so that $G$ contains a cycle, and, G being bipartite, this cycle contains a node $w \in W$ . As $\deg _ { G } ( w ) = 2$ , both neighbors of w lie on the cycle, so that they remain connected after the removal of w; hence $G - w$ is connected and, by (iii), w has a neighbor v such that de $\zeta _ { G } ( v ) = m + 1$ . The cycle goes through v, which it leaves through w and through another node $w ^ { \prime } \in \mathcal { N } _ { G } ( v ) \setminus \{ w \}$

Step 2: peeling and induction. Set $\boldsymbol { v } ^ { ( 1 ) } : = \boldsymbol { v }$ . Then, as long as there exists a node $u \in V \setminus \{ v ^ { ( 1 ) } , \ldots , v ^ { ( t ) } \}$ such that $| \mathcal { N } _ { G } ( u ) \setminus \mathcal { N } _ { G } ( \{ v ^ { ( 1 ) } , \ldots , v ^ { ( t ) } \} ) | \ \leq m$ , where $v ^ { ( 1 ) } , \ldots , v ^ { ( t ) }$ are the nodes selected so far, we select such a node as $\stackrel { \cdot } { v } ^ { ( t + 1 ) }$ . Let $s \geq 1$ be the number of selected nodes, and write $V ^ { ( 0 ) } = \{ v ^ { ( 1 ) } , \ldots , v ^ { ( s ) } \}$ $W ^ { ( 0 ) } = { \mathcal { N } } _ { G } ( V ^ { ( 0 ) } )$ , and $\overline { { W } } ^ { ( 0 ) } = W ^ { ( 0 ) } \setminus \{ w \}$ . For $t = 1$ , we have $| \overline { { { W } } } ^ { ( 0 ) } \cap \mathcal { N } _ { G } ( v ^ { ( 1 ) } ) | = | \mathcal { N } _ { G } ( v ) \setminus \{ w \} | = m \mathrm { ~ a n d }$ for $t \ge 2 , \overline { { W } } ^ { ( 0 ) } \cap \mathcal { N } _ { G } ( v ^ { ( t ) } ) \setminus \bigvee _ { G } \big ( \{ v ^ { ( 1 ) } , \ldots , v ^ { ( t - 1 ) } \} \big )$ has at most m elements by the selection rule.

Let $G ^ { \prime }$ be the graph obtained from G by removing $V ^ { ( 0 ) }$ and $W ^ { ( 0 ) }$ , and let $C _ { 1 } , \ldots , C _ { r }$ be its connected components $( r = 0 { \mathrm { ~ i f ~ } } G ^ { \prime }$ is empty). A node of $W \setminus \breve { W } ^ { ( 0 ) }$ is not adjacent to $V ^ { ( 0 ) }$ , so that it keeps its two neighbors in $G ^ { \prime } { } .$ , and, by the stopping rule, every node u of $V \backslash V ^ { ( 0 ) }$ satisfies $\deg _ { G ^ { \prime } } ( u ) = | \mathcal N _ { G } ( u ) \backslash W ^ { ( 0 ) } | \geq m + 1$ Hence, each $C _ { k }$ is a simple connected bipartite graph that satisfies (i) and (ii) and has fewer nodes in V than G. By the induction hypothesis, there exists ${ \overline { { W } } } _ { k } \subset W ( C _ { k } )$ that satisfies the $C _ { k } \mathrm { { \mathrm { - e n u m e r a t i o n } } }$ property and such that $\begin{array} { r } { | \overline { { W } } _ { k } | \geq \frac { | V ( C _ { k } ) | } { 2 } ( m + 1 / 2 ) - \frac { 1 } { 2 } } \end{array}$

Define $\overline { { W } } = \overline { { W } } ^ { ( 0 ) } \cup \overline { { W } } _ { 1 } \cup \cdots \cup \overline { { W } } _ { r }$ , and order the nodes of V by first taking $v ^ { ( 1 ) } , \ldots , v ^ { ( s ) }$ , and then the nodes of $C _ { 1 } , \ldots , C _ { r } ,$ each $C _ { k }$ according to an ordering witnessing the $C _ { k } .$ -enumeration property of $\overline { { W } } _ { k }$ Then W satisfies the G-enumeration property. Indeed, the nodes of $\overline { { W } } _ { 1 } , \ldots , \overline { { W } } _ { r }$ are not adjacent to $V ^ { ( 0 ) }$ so that (35) holds for $v ^ { ( 1 ) } , \ldots , v ^ { ( s ) }$ by the above. When a node u of $C _ { k }$ is reached, all the nodes of $\overline { { W } } ^ { ( 0 ) }$ which are adjacent to $V ^ { ( 0 ) }$ , have already been removed, the nodes of $\overline { { W } } _ { j }$ with $j \neq k$ are not adjacent to $u ,$ and $\mathcal { N } _ { G } ( u ) \cap \overline { { W } } _ { k } = \mathcal { N } _ { C _ { k } } ( u ) \cap \overline { { W } } _ { k }$ , so that (35) holds as in $C _ { k }$

Step 3: counting. Let Σ be the number of edges between $W ^ { ( 0 ) }$ and $V \setminus V ^ { ( 0 ) }$ . Since every node of $W ^ { ( 0 ) }$ has degree 2 and all the neighbors of the nodes of $V ^ { ( 0 ) }$ belong to $W ^ { ( 0 ) }$ , we have

$$
2 | W ^ { ( 0 ) } | = \sum _ { u \in V ^ { ( 0 ) } } \deg _ { G } ( u ) + \Sigma \geq ( m + 1 ) s + \Sigma .
$$

$\operatorname { A s } | \overline { { W } } ^ { ( 0 ) } | = | W ^ { ( 0 ) } | - 1$ and $\begin{array} { r } { \sum _ { k < r } | V ( C _ { k } ) | = | V | - s , } \end{array}$ , we deduce that

$$
| \overline { { { W } } } | \geq \frac { ( m + 1 ) s + \Sigma } { 2 } - 1 + \frac { | V | - s } { 2 } ( m + 1 / 2 ) - \frac { r } { 2 } = \frac { | V | } { 2 } ( m + 1 / 2 ) - \frac { 1 } { 2 } + x , \qquad \mathrm { w h e r e ~ } x : = \frac { s } { 4 } + \frac { \Sigma - r } { 2 } - \frac { 1 } { 2 } \ .
$$

It remains to prove that $x \geq 0$ . Since $G$ is connected and the nodes of $W \setminus W ^ { ( 0 ) }$ have all their neighbors in $G ^ { \prime }$ , each component $C _ { k }$ is joined to the rest of G by at least one edge between $W ^ { ( 0 ) }$ and $V ( C _ { k } )$ ; hence $\Sigma \geq r .$ and $x \geq 0$ as soon as $s \geq 2 . \ \mathrm { H } \ s = 1$ , that is ${ \cal V } ^ { ( 0 ) } = \{ v \}$ , then $W ^ { ( 0 ) } = \mathcal { N } _ { G } ( v )$ and $\Sigma = m + 1$ , as each of the $m + 1$ neighbors of v has exactly one other neighbor. Besides, the part of the cycle of Step 1 that joins the other neighbor of w to the other neighbor of $w ^ { \prime }$ avoids v, hence also $\mathcal { N } _ { G } ( v )$ , since a node of $\mathcal { N } _ { G } ( v )$ has v as one of its only two neighbors; it is therefore a path in $G ^ { \prime } .$ . Thus, two of the $m + 1$ edges counted in Σ lead to the same component of $G ^ { \prime }$ and, as each component receives at least one of them, $r \leq m$ . This gives $\begin{array} { r } { x \ge \frac { 1 } { 4 } + \frac { 1 } { 2 } - \frac { 1 } { 2 } > 0 } \end{array}$ , which concludes the induction.

Proof of Lemma 18. The case where each node $w \in W$ satisfies $\deg _ { G } ( w ) \ = \ 2$ has been dealt with in Lemma 17. Hence, we assume henceforth that at least one node w $\in W$ satisfies $\deg _ { G } ( w ) \geq 4$

We shall build the set W iteratively. We define $W ^ { ( 0 ) } = W , V ^ { ( 0 ) } = V , \overline { { { W } } } ^ { ( 0 ) } = \emptyset$ , and $G ^ { ( 0 ) } = G$

First, since all nodes $v \in V$ have degree at least $2 m + 2 \textrm { - }$ – this follows from Condition $( \mathcal { W } _ { G } )$ applied with ${ \cal V } _ { 0 } = \{ v \}$ , which gives $\deg _ { G } ( v ) = | \mathcal N _ { G } ( v ) | \notin [ 1 : 2 m + 1 ]$ , together with $\deg _ { G } ( v ) \geq 1$ since G contains no isolated node –, we remove the node $w ^ { * }$ with the highest degree, which does not create any node $v \in V$ of degree at most m. In other words, we define $W ^ { ( 1 ) } = W ^ { ( 0 ) } \setminus \{ w ^ { * } \} , \overline { { W } } ^ { ( 1 ) } = \emptyset$ , and we fix $G ^ { ( 1 ) }$ to be the graph obtained from G by pruning $w ^ { * }$ . More generally, at time t, we check whether $W ^ { ( t - 1 ) }$ still contains a node w with degree $\deg _ { G } ( w ) \geq 4$ . If no such node exists, we stop and we define $t ^ { * } = t - 1$ . Otherwise, we take w to be a node in $\Breve { W } ^ { ( t - 1 ) }$ with the largest degree. We define $W ^ { ( t ) } = W ^ { ( t - 1 ) } \setminus \{ w \}$ and $V ^ { ( t ) } = V ^ { ( t - 1 ) }$ and we define $G ^ { ( t ) }$ as the graph induced by $V ^ { ( t ) }$ and $W ^ { ( t ) }$ , and we set $\overline { { W } } ^ { ( t ) } = \overline { { W } } ^ { ( t - 1 ) }$ . If all the nodes $v \in V ^ { ( t ) }$ satisfy d $\log _ { G ^ { ( t ) } } ( v ) \geq m + 1$ , then we go to $t + 1$ . If, instead, some node $v \in V ^ { ( t ) }$ satisfies $\deg _ { G ^ { ( t ) } } ( v ) \leq m .$ , then we update $\overline { { W } } ^ { ( t ) } = \overline { { W } } ^ { ( t ) } \cup [ W ^ { ( t ) } \cap \mathcal { N } _ { G ^ { ( t ) } } ( v ) ]$ ], we update $W ^ { ( t ) } = W ^ { ( t ) } \setminus [ W ^ { ( t ) } \cap \mathcal { N } _ { G ^ { ( t ) } } ( v ) ] , V ^ { ( t ) } = V ^ { ( t ) } \setminus \{ v \}$ $G ^ { ( t ) }$ as the graph induced by $V ^ { ( t ) }$ and $W ^ { ( t ) }$ . We continue updating $( \overline { { W } } ^ { ( t ) } , V ^ { ( t ) } , W ^ { ( t ) } , G ^ { ( t ) } )$ until the graph $G ^ { ( t ) }$ does not contain any node $v \in V ^ { ( t ) }$ with $\deg _ { G ^ { ( t ) } } ( v ) \leq m$

It readily follows from these definitions that the graph $G ^ { ( t ^ { * } ) }$ is either empty or is such that all $w \in W ^ { ( t ^ { * } ) }$ satisfy $\deg _ { G } ( w ) = 2$ and all $v \in V ^ { ( t ^ { * } ) }$ satisfy $\deg _ { G ^ { ( t ^ { * } ) } } ( v ) \geq m + 1$ . Denote by c the number of connected components of $G ^ { ( t ^ { * } ) }$ . Then, we apply Lemma 17 to construct c sets $\overline { { W } } ^ { * ( 1 ) } , \ldots , \overline { { W } } ^ { * ( c ) }$ such that $\cup _ { i = 1 } ^ { c } \overline { { W } } ^ { * ( i ) }$ satisfies the $G ^ { ( t ^ { * } ) }$ -enumeration property. Besides, we have

$$
\sum _ { i = 1 } ^ { c } \vert \overline { { { W } } } ^ { * ( i ) } \vert \geq \frac { \vert V ^ { ( t ^ { * } ) } \vert } { 2 } ( m + 1 / 2 ) - \frac { c } { 2 }\tag{39}
$$

One readily checks from the iterative construction that $\overline { { W } } ^ { ( t ^ { * } ) } \cup _ { i = 1 } ^ { c } \overline { { W } } ^ { * ( i ) }$ satisfies the G-enumeration property.

Let us bound $| \overline { { W } } ^ { ( t ^ { * } ) } |$ . For that purpose, we shall first look at the form of $\mathcal { N } _ { G } ( v ) \cap \overline { { W } } ^ { ( t ^ { * } ) }$ for each $v \in V \setminus V ^ { ( t ^ { * } ) }$

Lemma 20. Consider any node $v \in V \setminus V ^ { ( t ^ { * } ) }$ . We have $| \mathcal { N } _ { G } ( v ) \cap \overline { { W } } ^ { ( t ^ { * } ) } | \ge m$ and it is possible to order the neighbors w $\in \mathcal { N } _ { G } ( v )$ by non-decreasing degree $\deg _ { G } ( w )$ in such a way that the first m nodes of this ordering all belong to $\overline { { W } } ^ { ( t ^ { * } ) }$

This lemma is straightforward in light of the recursive definitions of the sets and we do not provide any proof. Let us define $V _ { \mathrm { a } } \subset V \setminus V ^ { ( t ^ { * } ) }$ as the subset of nodes v such that at least $m + 1$ nodes in $\mathcal { N } _ { G } ( v )$ are of degree 2.

Lemma 21. If $v \in V _ { \mathrm { a } }$ , then $\overline { W } ^ { ( t ^ { * } ) }$ contains at least $m + 1$ neighbors of v of degree 2.

Proof. First, observe that $\overline { { W } } ^ { ( t ^ { * } ) } \cup W ^ { ( t ^ { * } ) }$ contains all the degree-2 nodes of W. Since $v \in V _ { \mathrm { a } }$ has been pruned at some t, this means that all its neighbors have either been added to $\overline { W } ^ { ( t ) }$ or been pruned. Since no degree-2 node has been pruned, the result follows. □

Define $V _ { \mathrm { b } } = V \setminus ( V _ { \mathrm { a } } \cup V ^ { ( t ^ { * } ) } )$ . In order to bound $| \overline { { W } } ^ { ( t ^ { * } ) } |$ , we shall control the edges between $V$ and $\overline { { W } } ^ { ( t ^ { * } ) }$ and rescale this by the degree of the $w \in \overline { { W } } ^ { ( t ^ { * } ) }$ . For each node $v \in V _ { \mathrm { b } }$ , we denote by $d _ { v , i }$ the i-th smallest degree among $\mathcal { N } _ { G } ( v )$ . Then, we have

$$
| \overline { { W } } ^ { ( t ^ { * } ) } | \geq | V _ { \mathrm { a } } | \frac { m + 1 } { 2 } + \sum _ { v \in V _ { \mathrm { b } } } \sum _ { i = 1 } ^ { m } \frac { 1 } { d _ { v , i } } + \sum _ { v \in V ^ { ( t ^ { * } ) } , w \in \overline { { W } } ^ { ( t ^ { * } ) } } \frac { \mathbf { 1 } \{ w \sim _ { G } v \} } { \deg _ { G } ( w ) }
$$

Then, it follows from (39) that $\overline { { W } } = \overline { { W } } ^ { ( t ^ { * } ) } \cup _ { i = 1 } ^ { c } \overline { { W } } ^ { * ( i ) }$ satisfies

$$
| \overline { { W } } | \geq \sum _ { v \in V _ { \mathrm { a } } } \frac { m + 1 } { 2 } + \sum _ { v \in V _ { \mathrm { b } } } \sum _ { i = 1 } ^ { m } \frac { 1 } { d _ { v , i } } + \sum _ { v \in V ^ { ( t ^ { * } ) } } \left( \frac { m + 1 / 2 } { 2 } + \sum _ { w \in \overline { { W } } ^ { ( t ^ { * } ) } } \frac { \mathbf { 1 } \{ w \sim _ { G } v \} } { \deg ( w ) } \right) - \frac { c } { 2 }\tag{40}
$$

Lemma 22. Consider a node v such that $\deg _ { G } ( v ) \geq 2 m + 2 . \ I f d _ { v , m + 1 } \geq 4$ , then we have

$$
\sum _ { i = 1 } ^ { m } \frac { 1 } { d _ { v , i } } \geq \frac { 1 } { 2 } ( m + 1 / 2 ) + \sum _ { w \in \mathcal { N } _ { G } ( v ) } \left( \frac { 1 } { 2 \deg _ { G } ( w ) } - \frac { 1 } { 4 } \right)
$$

Proof of Lemma 22. Since the $d _ { v , i } \mathrm { ' s }$ are increasing in $i ,$ we have

$$
\sum _ { i = 1 } ^ { m } { \frac { 1 } { d _ { v , i } } } \geq \sum _ { i = 1 } ^ { 2 m } { \frac { 1 } { 2 d _ { v , i } } } = { \frac { m } { 2 } } + \sum _ { i = 1 } ^ { 2 m } \left( { \frac { 1 } { 2 d _ { v , i } } } - { \frac { 1 } { 4 } } \right) .
$$

Besides, we have $d _ { v , 2 m + 2 } \geq d _ { v , 2 m + 1 } \geq 4$ , which implies that $\begin{array} { r } { \frac { 1 } { 2 d _ { v , 2 m + 1 } } + \frac { 1 } { 2 d _ { v , 2 m + 2 } } \leq 1 / 4 } \end{array}$ . This allows us to conclude that

$$
\sum _ { i = 1 } ^ { m } \frac { 1 } { d _ { v , i } } \geq \frac { m + 1 / 2 } { 2 } + \sum _ { i = 1 } ^ { 2 m + 2 } \left( \frac { 1 } { 2 d _ { v , i } } - \frac { 1 } { 4 } \right) \geq \frac { 1 } { 2 } ( m + 1 / 2 ) + \sum _ { w \in \mathcal { N } _ { G } ( v ) } \left( \frac { 1 } { 2 \mathrm { d e g } _ { G } ( w ) } - \frac { 1 } { 4 } \right) \ ,
$$

where we used, for the last inequality, that $d _ { v , i } \geq 4$ for $i \geq m + 1$

Let us define η by

$$
\eta : = \sum _ { v \in V ^ { ( t ^ { * } ) } } \sum _ { w \in \mathcal { N } _ { G } ( v ) } \left( \frac { 1 } { 4 } - \frac { 1 } { 2 \mathrm { d e g } _ { G } ( w ) } + \frac { \mathbf { 1 } \{ w \in \overline { { W } } ^ { ( t ^ { * } ) } \} } { \mathrm { d e g } _ { G } ( w ) } \right) - \frac { c } { 2 } .\tag{41}
$$

Let us apply Lemma 22 to any $v \in V _ { \mathrm { b } }$ . Since $1 / ( 2 \deg _ { G } ( w ) ) \leq 1 / 4$ , it follows from (40) that

$$
| \overline { { W } } | \geq \eta + \sum _ { v \in V } \left( \frac { 1 } { 2 } ( m + 1 / 2 ) + \sum _ { w \in \mathcal { N } _ { G } ( v ) } \left( \frac { 1 } { 2 \mathrm { d e g } _ { G } ( w ) } - \frac { 1 } { 4 } \right) \right) = \eta + \frac { | V | } { 2 } ( m + 1 / 2 ) + \frac { | W | } { 2 } - \frac { | E | } { 4 }
$$

In light of (36), we only have to establish $\eta \ge - 1 / 2$ to conclude the proof.

Let us partition $V ^ { ( t ^ { * } ) }$ into its connected components $V _ { i } ^ { ( t ^ { * } ) }$ for $i = 1 , \ldots , c$ . Define $\eta _ { i }$ by

$$
\eta _ { i } : = \sum _ { v \in V _ { i } ^ { ( t ^ { * } ) } } \sum _ { w \in \mathcal { N } _ { G } ( v ) } \left( \frac { 1 } { 4 } - \frac { 1 } { 2 \mathrm { d e g } _ { G } ( w ) } + \frac { { \bf 1 } \{ w \in \overline { { W } } ^ { ( t ^ { * } ) } \} } { \mathrm { d e g } _ { G } ( w ) } \right) - \frac { 1 } { 2 } \ ,
$$

so that $\begin{array} { r } { \eta = \sum _ { i = 1 } ^ { c } \eta _ { i } } \end{array}$ . We shall prove that each $\eta _ { i }$ is non-negative, which in turn will imply that $\eta \geq 0$ . We consider two cases.

Case (i): There is a degree-2 node $w \in W$ that connects $V _ { i } ^ { ( t ^ { * } ) }$ and $V _ { \mathrm { a } } \cup V _ { \mathrm { b } }$ . By construction, this node belongs to $\overline { W } ^ { ( t ^ { * } ) }$ and we get $\eta _ { i } \ge 1 / 2 - 1 / 2 = 0$

Case (ii): No degree-2 node $w \in W$ is simultaneously connected to $V _ { i } ^ { ( t ^ { * } ) }$ and $V _ { \mathrm { a } } \cup V _ { \mathrm { b } }$ . By condition $( \mathcal { W } _ { G } )$ we know that

$$
\mid \oplus _ { v \in V _ { i } ^ { ( t ^ { * } ) } } \mathcal { N } _ { G } ( v ) \mid \not \in [ 1 : 2 m + 1 ] \ .
$$

$\mathrm { I f ~ } | \oplus _ { v \in V _ { \ast } ^ { ( t ^ { \ast } ) } } \mathcal { N } _ { G } ( v ) | \geq 2 m + 2$ , this implies that there are at least $2 m + 2$ edges between a node in $V _ { i } ^ { ( t ^ { * } ) }$ and a node $\dot { w } \in \oplus _ { v \in V _ { i } ^ { ( t ^ { * } ) } } \mathcal { N } _ { G } ( v )$ . All these nodes w are of degree at least 4 and we get

$$
\eta _ { i } \ge ( 2 m + 2 ) { \frac { 1 } { 8 } } - 1 / 2 \ge 0 \ ,
$$

since $m \geq 1$ . Finally, we consider the case where $\begin{array} { r } { \bigoplus _ { v \in V _ { i } ^ { ( t ^ { * } ) } } \mathcal { N } _ { G } ( v ) = \emptyset } \end{array}$ . Since G is connected, this implies that there is a positive and even number l of edges between $V _ { i } ^ { ( t ^ { * } ) }$ and nodes $w \in W$ of degree at least 4. ${ \mathrm { I f ~ } } l \geq 4 .$ we conclude as before that

$$
\eta _ { i } \ge l / 8 - 1 / 2 \ge 0 .
$$

Finally, we observe that the case $l = 2$ is not possible, as this would imply that $V _ { i } ^ { ( t ^ { * } ) }$ is a group of type A, while we assumed at the beginning that G does not contain any group of type A.

Proof of Lemma 19. We only prove the result for connected graphs G. We do it by induction on the size $| V |$ . First, the property holds for any such graph $G = ( V , W , E )$ with $| V | = 2$ . Indeed, the degree $\deg _ { G } ( w )$ of any $w \in W$ is equal to 2, and the result follows from Lemma 17.

Now, assume that the property holds for any graph $G = ( V , W , E )$ satisfying $( \mathcal { W } _ { G } )$ , containing no isolated node, and such that $| V | \leq r .$ Consider a graph $G = ( V , W , E )$ satisfying $( \mathcal { W } _ { G } )$ , containing no isolated node, and such that $| V | = r + 1$ . If G does not contain any group of type $\mathrm { A } ,$ the result follows from Lemma 18. Otherwise, let $V ^ { * }$ be a group of type A and le $w ^ { * } \in W$ be the unique node that connects $V ^ { * } \mathrm { t o } V ^ { \circ } : = V \backslash V ^ { * }$

Let us remove the two edges $e _ { 1 } ^ { * }$ and $e _ { 2 } ^ { * }$ between $V ^ { * }$ and $w ^ { * }$ . The graph $G ^ { \prime } = ( V , W , E \ : \backslash \ : \{ e _ { 1 } ^ { * } , e _ { 2 } ^ { * } \} )$ is therefore made of two connected components that respectively contain $V ^ { \circ }$ and $V ^ { * }$ . Let us denote by $G ^ { \circ } = ( V ^ { \circ } , W ^ { \circ } , E ^ { \circ } )$ and $G ^ { * } = ( V ^ { * } , W ^ { * } , E ^ { * } )$ the two connected components of $G ^ { \prime }$ containing $V ^ { \circ }$ and $V ^ { * }$ respectively; in particular, $w ^ { * }$ , which keeps at least two neighbors in $V ^ { \circ }$ , belongs to $W ^ { \circ }$ and not to $W ^ { * }$ One readily checks that $G ^ { \circ }$ is connected and satisfies $( \mathcal { W } _ { G } )$ , is such that $| V ^ { \circ } | \le r$ and does not contain any isolated node. By the induction hypothesis, there exists $\overline { W } ^ { \circ } \subset W ^ { \circ }$ that satisfies the $G ^ { \circ }$ -enumeration property and such that

$$
| \overline { { { W } } } ^ { \circ } | \geq \frac { | W ^ { \circ } | } { 2 } - \frac { | E ^ { \circ } | } { 4 } + \frac { | V ^ { \circ } | } { 2 } ( m + 1 / 2 ) - \frac { 1 } { 2 } .
$$

Similarly, $G ^ { * }$ satisfies the conditions of Lemma 17. Hence, there exists $\overline { W } ^ { * } \subset W ^ { * }$ that satisfies the $G ^ { * } \mathrm { . }$ enumeration property and such that

$$
| \overline { { { W } } } ^ { * } | \geq \frac { | V ^ { * } | } { 2 } ( m + 1 / 2 ) - \frac 1 2 = \frac { | W ^ { * } | } { 2 } - \frac { | E ^ { * } | } { 4 } + \frac { | V ^ { * } | } { 2 } ( m + 1 / 2 ) - \frac 1 2 \ ,
$$

since all the nodes $w \in W ^ { * }$ are of degree 2.

Since no node $v \in V ^ { \circ }$ is adjacent to $w \in W ^ { * } , \overline { { W } } = \overline { { W } } ^ { \circ } \cup \overline { { W } } ^ { * }$ satisfies the G-enumeration property. More precisely, let us order the nodes of V by first enumerating those of $V ^ { \circ }$ according to an ordering witnessing the $G ^ { \circ }$ -enumeration property of $\overline { W } ^ { \circ }$ , and then those of $V ^ { * }$ according to an ordering witnessing the $G ^ { * } .$ enumeration property of $\overline { W } ^ { * }$ . During the first phase, the nodes of $\Breve { \boldsymbol { W } } ^ { * }$ are never counted in the left-hand side of (35), as they are not adjacent to $V ^ { \circ }$ , and $\mathcal { N } _ { G } ( v ) = \mathcal { N } _ { G ^ { \circ } } ( v )$ for any $v \in V ^ { \circ }$ , so that (35) holds along this phase as it does in $G ^ { \circ }$ . Conversely, the only node of $\overline { W } ^ { \circ }$ that may be adjacent to $V ^ { * }$ is $w ^ { * }$ : any other node $w \in W ^ { \circ }$ is adjacent to $V ^ { \circ }$ , so that, if it were also adjacent to $V ^ { * }$ , it would still connect $V ^ { * }$ to $V \setminus V ^ { * }$ after the removal of $w ^ { * }$ , contradicting the property (iii) of a group of type A. Since $w ^ { * }$ is adjacent to at least one node of $V ^ { \circ } -$ its degree in $G ^ { \circ }$ is at least $^ { 2 , }$ as recalled above –, it has been removed from the sets $\overline { W } ^ { ( t ) }$ before the end of the first phase. Hence, along the second phase, only the nodes of $\overline { W } ^ { * }$ are counted in the left-hand side of (35), which therefore holds as it does in $G ^ { * }$ . We conclude that

$$
| \overline { { { W } } } | \geq \frac { | W ^ { * } | + | W ^ { \circ } | } { 2 } - \frac { | E ^ { * } | + | E ^ { \circ } | } { 4 } + \frac { | V ^ { * } | + | V ^ { \circ } | } { 2 } ( m + 1 / 2 ) - 1 = \frac { | W | } { 2 } - \frac { | E | } { 4 } + \frac { | V | } { 2 } ( m + 1 / 2 ) - \frac { 1 } { 2 } \ ,
$$

since $| E | = 2 + | E ^ { * } | + | E ^ { \circ } |$ . We have proved the induction property and the result follows.

The extension to graphs with multiple connected components is straightforward as we only need to consider each connected component independently.

## B.4 Optimality of the exponent in Lemma 11

We prove the claim of the remark following Lemma 11. Fix $r \geq 1$ , let $G ^ { \circ }$ be the cycle of 2r pairs of twin nodes defined there, and let $d \to \infty$ with m and r fixed. The upper bound follows from Lemma 11 applied with $D = 4 r ( m + 1 )$ , as soon as $d \ge 2 ^ { 8 } D ^ { 4 }$ . For the lower bound, we use a direct counting argument, which does not rely on Lang–Weil type estimates.

Syndromes. Write $\mathbb { F } _ { d + 1 } ^ { * } : = \mathbb { F } _ { d + 1 } \setminus \{ 0 \}$ and, for a finite subset $A \subset \mathbb { F } _ { d + 1 } ^ { * }$ , define its syndrome $s ( A ) : =$ $\begin{array} { r } { \big ( \sum _ { a \in A } a ^ { 2 l - 1 } \big ) _ { l < m } \in \mathbb { F } _ { d + 1 } ^ { \dot { m } } } \end{array}$ . Since $\mathbb { F } _ { d + 1 }$ has characteristic 2, we have $s ( A \oplus B ) = s ( A ) + s ( B )$ , where $A \oplus B$ is the symmetric diference of A and B. Besides, if $1 \leq | A | \leq 2 m$ , then $s ( A ) \neq 0 ;$ otherwise, Lemma 24, read in $\mathbb { F } _ { d + } .$ through ψ as in the proof of Lemma 11, would imply that the vector made of the pairwise distinct elements of A is paired, which is impossible. As a consequence:

(i) if |A| = |B| = m and s(A) = s(B), then A = B, since $| A \oplus B | \leq$ 2m and $s ( A \oplus B ) = 0 ;$

(ii) if $| A | = | B | = m + 1 , A \neq B$ and $s ( A ) = s ( B )$ , then $A \cap B = \emptyset$ , since otherwise $1 \leq | A \oplus B | \leq 2 m$ . In particular, $A \cup B$ is then a subset of size $2 m + 2$ with $s ( A \cup B ) = 0$

Counting. Since $\mathcal { N } ( v _ { t } ) = \mathcal { N } ( v _ { t } ^ { \prime } )$ , a tuple $( x _ { w } ) _ { w \in W ^ { \circ } }$ whose values on each neighborhood $\mathcal { N } ( v _ { t } )$ are pairwise distinct satisfies the system (32) for $G ^ { \circ }$ if and only if it satisfies the 2rm equations

$$
s \mathopen { } \mathclose \bgroup \left( \left\{ x _ { w _ { t , 1 } } , \ldots , x _ { w _ { t , 2 m } } \right\} \aftergroup \egroup \right) = s \mathopen { } \mathclose \bgroup \left( \left\{ x _ { u _ { t - 1 } } , x _ { u _ { t } } \aftergroup \egroup \right\} \aftergroup \egroup \right) ~ , \qquad t \in \mathbb { Z } / 2 r \mathbb { Z } ~ .\tag{42}
$$

For $a , a ^ { \prime } \in \mathbb { F } _ { d + 1 } ^ { * }$ with $a \ne a ^ { \prime }$ , let $M _ { a , a ^ { \prime } }$ be the number of injective tuples $\left( y _ { 1 } , \ldots , y _ { 2 m } \right)$ with values in $\mathbb { F } _ { d + 1 } ^ { * } \setminus \left\{ a , a ^ { \prime } \right\}$ such that $s ( \{ y _ { 1 } , . . . , y _ { 2 m } \} ) = s ( \{ a , a ^ { \prime } \} )$ ), and set $M _ { a , a } = 0$ . The $d \times d$ matrix M is symmetric with nonnegative entries. Expanding the trace,

$$
\mathrm { T r } ( M ^ { 2 r } ) = \sum _ { ( a _ { t } ) _ { t \in \mathbb { Z } / 2 r \mathbb { Z } } } \prod _ { t \in \mathbb { Z } / 2 r \mathbb { Z } } M _ { a _ { t - 1 } , a _ { t } }
$$

is the number of tuples $( x _ { w } ) _ { w \in W ^ { \circ } }$ with values in $\mathbb { F } _ { d + 1 } ^ { * }$ satisfying (42) and such that, for every t, the $2 m + 2$ values $x _ { w } , w \in \mathcal { N } ( v _ { t } )$ , are pairwise distinct. Such a tuple belongs to U, unless two nodes w, $w ^ { \prime } \in W ^ { \circ }$ without common neighbor receive the same value. For a given pair $\{ w , w ^ { \prime } \}$ of such nodes, we bound the number of these tuples with $x _ { w } = x _ { w ^ { \prime } }$ as follows. We first choose the values of the nodes $u _ { t } ,$ , and then, for each $t ,$ the values of m nodes among $w _ { t , 1 } , \ldots , w _ { t , 2 m }$ , including w or $w ^ { \prime }$ if one of them belongs to this block. By (i) and (42), the set of values of the m remaining nodes of each block is then uniquely determined, so that there are at most m! choices for these values. As $x _ { w } = x _ { w ^ { \prime } }$ removes one of the $2 r ( m + 1 )$ free choices, the number of such tuples is at most $( m ! ) ^ { 2 r } d ^ { 2 r ( m + 1 ) - 1 }$ . Summing over the at most $| W ^ { \circ } | ^ { 2 }$ pairs $\{ w , w ^ { \prime } \}$ , we get

$$
| \mathbb { U } | \geq \operatorname { T r } ( M ^ { 2 r } ) - | W ^ { \circ } | ^ { 2 } ( m ! ) ^ { 2 r } d ^ { 2 r ( m + 1 ) - 1 } \ .
$$

Since M is symmetric, its eigenvalues are real and, as $2 r$ is even, $\mathrm { T r } ( M ^ { 2 r } ) \geq \lambda _ { \operatorname* { m a x } } ( M ) ^ { 2 r } \geq ( { \bf 1 } ^ { \top } M { \bf 1 } / d ) ^ { 2 r }$ where 1 is the all-ones vector. The quantity $\mathbf { 1 } ^ { \top } M \mathbf { 1 }$ is the number of injective $( 2 m + 2 )$ -tuples with values in $\mathbb { F } _ { d + 1 } ^ { * }$ whose set of values has a zero syndrome, that is $( 2 m + 2 ) ! Z \cdot$ , where Z is the number of subsets of $\mathbb { F } _ { d + 1 } ^ { * }$ of size $2 m + 2$ with a zero syndrome. For $\varsigma \in \mathbb { F } _ { d + 1 } ^ { m }$ , let $n _ { \varsigma }$ be the number of subsets of $\mathbb { F } _ { d + 1 } ^ { * }$ of size $m + 1$ with syndrome ς. By (ii), any ordered pair $( A , B )$ of distinct such subsets with the same syndrome gives rise to the subset $A \cup B ,$ , which is counted in $Z ,$ and each subset counted in Z arises from at most $\binom { 2 m + 2 } { m + 1 }$ such pairs. Hence, by the Cauchy–Schwarz inequality and since there are $( d + 1 ) ^ { m }$ syndromes,

$$
{ \binom { 2 m + 2 } { m + 1 } } Z \ge \sum _ { c } n _ { c } ( n _ { c } - 1 ) \ge \frac { 1 } { ( d + 1 ) ^ { m } } { \binom { d } { m + 1 } } ^ { 2 } - { \binom { d } { m + 1 } } \ .
$$

Since $( 2 m + 2 ) ! / { \binom { 2 m + 2 } { m + 1 } } \ = \ ( ( m + 1 ) ! ) ^ { 2 }$ , we obtain $\mathbf { 1 } ^ { \top } M \mathbf { 1 } \ \geq \ ( 1 - o ( 1 ) ) d ^ { m + 2 }$ , so that $\mathrm { T r } ( M ^ { 2 r } ) ~ \geq ~ ( 1 ~ -$ $o ( 1 ) ) d ^ { 2 r ( m + 1 ) }$ and $| \mathbb { U } | \geq ( 1 - o ( 1 ) ) d ^ { 2 r ( m + 1 ) }$ . Finally, by (33) and since $d ! / ( d - | W ^ { \circ } | ) ! \leq d ^ { | W ^ { \circ } | }$

$$
\mathbb { P } _ { \sigma } [ \mathcal { A } _ { G ^ { \circ } } ] \geq \frac { | \mathbb { U } | } { d ^ { | W ^ { \circ } | } } \geq ( 1 - o ( 1 ) ) d ^ { 2 r ( m + 1 ) - 2 r ( 2 m + 1 ) } = ( 1 - o ( 1 ) ) \Big ( \frac { 1 } { d ^ { m } } \Big ) ^ { 2 r } .
$$

Rank of the Jacobian. For any $x \in \mathbb { U }$ , the rows of the Jacobian matrix $\mathcal { I } _ { x }$ defined in (34) that are indexed by $( i , v _ { t } )$ and $( i , v _ { t } ^ { \prime } )$ coincide, so that rank(J ) ≤ 2rm. Together with Lemma 15, which gives $\operatorname { r a n k } ( \mathcal { I } _ { x } ) \geq$ $b ( G ^ { \circ } ) = 2 r m - 1 / 2$ , we get rank $( \mathcal { I } _ { x } ) = \lceil b ( G ^ { \circ } ) \rceil$ , so that the lower bound on the rank used in the proof of Lemma 11 is attained for $G ^ { \circ }$

## C Proof of Theorem 2

We define the Gram matrix Γ associated to the family $( 1 , ( \tilde { \Psi } _ { G } ) _ { G \in \mathcal { G } _ { < D } ^ { \mathrm { e v e n } } } )$ by $\Gamma _ { G ^ { ( 1 ) } , G ^ { ( 2 ) } } = \mathbb { E } \left[ \tilde { \Psi } _ { G ^ { ( 1 ) } } \tilde { \Psi } _ { G ^ { ( 2 ) } } \right]$ for any $G ^ { ( 1 ) }$ and $G ^ { ( 2 ) }$ in $\mathcal { G } _ { < D } ^ { \mathrm { e v e n } } , \Gamma _ { \varnothing , G } = \Gamma _ { G , \varnothing } = \mathbb { E } [ \tilde { \Psi } _ { G } ]$ for any $G \in \mathcal { G } _ { < D } ^ { \mathrm { e v e n } }$ , and $\Gamma _ { \emptyset , \emptyset } = 1$ . Let us write Id for the identity matrix with the same dimension as Γ. Then, Theorem 2 is a straightforward corollary of the following result.

Proposition 2. Assume that Condition $( \mathcal { H } _ { s i g n a l } )$ holds for some $c _ { \mathrm { s } } \geq 1 0$ , that $2 ^ { 8 } D ^ { 4 } \leq d _ { \mathrm { { i } } }$ , that $D \geq 2 .$ , and that $n \geq d$ . Then, we have

$$
\| \Gamma - \mathrm { I d } \| _ { o p } \leq 4 D ^ { - c _ { \mathrm { s } } / 2 } + \frac { 4 D ^ { 2 } } { d } ,
$$

where $\| . \| _ { o p }$ stands for the operator norm.

In order to establish Proposition 2, we shall simply bound the operator norm of a symmetric matrix by the largest $l _ { 1 }$ norm of its rows. For that purpose, we need to evaluate $\vert \Gamma _ { G ^ { ( 1 ) } , G ^ { ( 2 ) } } - \mathbf { 1 } \{ G ^ { ( 1 ) } = G ^ { ( 2 ) } \} \vert$ . This is done in Proposition 3 below. This requires controlling the sum of moments of the form $\mathbb { E } \big [ \Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \big ]$ over all possible labelings $\pi ^ { ( 1 ) }$ and $\pi ^ { ( 2 ) }$ . For that purpose, we need to introduce additional graph notation.

## C.1 Additional graph notation

Node matching. Consider two graphs $G ^ { ( 1 ) } = ( V ^ { ( 1 ) } , W ^ { ( 1 ) } , E ^ { ( 1 ) } ) , G ^ { ( 2 ) } = ( V ^ { ( 2 ) } , W ^ { ( 2 ) } , E ^ { ( 2 ) } )$ . Given labelings $\pi ^ { ( 1 ) } = ( \pi _ { 1 } ^ { ( 1 ) } , \pi _ { 2 } ^ { ( 1 ) } )$ and $\pi ^ { ( 2 ) } = ( \pi _ { 1 } ^ { ( 2 ) } , \pi _ { 2 } ^ { ( 2 ) } )$ , recall that we say that two nodes $\boldsymbol { v } ^ { ( 1 ) }$ and $v ^ { ( 2 ) } \ ( \mathrm { r e s p . } w ^ { ( 1 ) }$ and $w ^ { ( 2 ) } )$ are matched i $\mathrm { ~ f ~ } \pi _ { 1 } ^ { ( 1 ) } ( v ^ { ( 1 ) } ) = \pi _ { 1 } ^ { ( 2 ) } ( v ^ { ( 2 ) } )$ (resp. $\pi _ { 2 } ^ { ( 1 ) } ( w ^ { ( 1 ) } ) = \pi _ { 2 } ^ { ( 2 ) } ( w ^ { ( 2 ) } ) )$ . More generally, a matching ${ { \bf { M } } _ { 1 } }$ stands for a set of pairs of nodes $( \hat { v ^ { ( 1 ) } } , v ^ { ( 2 ) } ) \in V ^ { ( 1 ) } \times \bar { V } ^ { ( 2 ) }$ where no node in $V ^ { ( 1 ) }$ or $V ^ { ( 2 ) }$ appears twice. Similarly, we define matchings $\mathbf { M } _ { 2 }$ for $W ^ { ( 1 ) }$ and $W ^ { ( 2 ) }$ . We denote by $\mathcal { M } _ { 1 }$ and $\mathcal { M } _ { 2 }$ the collections of all possible node matchings, where, for $\mathcal { M } _ { 1 }$ , we have the additional constraint that $( v _ { 1 } ^ { ( 1 ) } , v _ { 1 } ^ { ( 2 ) } )$ and $( v _ { 2 } ^ { ( 1 ) } , v _ { 2 } ^ { ( 2 ) } )$ belong to the matching - and we write $\mathcal { M } = \mathcal { M } _ { 1 } \times \mathcal { M } _ { 2 }$ . Given $\mathbf { M } _ { 1 } \in \mathcal { M } _ { 1 }$ and $\mathbf { M } _ { 2 } \in \mathcal { M } _ { 2 }$ , we define the set of labelings compatible with the matchings ${ { \bf { M } } _ { 1 } }$ and $\mathbf { M } _ { 2 }$ by

$$
\begin{array} { r l } & { \overline { { \Pi } } _ { 1 } ( \mathbf { M } _ { 1 } ) } \\ & { = \left\{ ( \overline { { \alpha } } _ { 1 } ^ { ( 1 ) } , \pi _ { 1 } ^ { ( 2 ) } ) \in \overline { { \Pi } } _ { V ^ { ( 1 ) } } \times \overline { { \Pi } } _ { V ^ { ( 2 ) } } : \forall ( v ^ { ( 1 ) } , v ^ { ( 2 ) } ) \in V ^ { ( 1 ) } \times V ^ { ( 2 ) } , \{ \pi _ { 1 } ^ { ( \mathfrak { U } ) } ( v ^ { ( 1 ) } ) = \pi _ { 1 } ^ { ( 2 ) } ( v ^ { ( 2 ) } ) \} \Longleftrightarrow \{ ( v ^ { ( 1 ) } , v ^ { ( 2 ) } ) \in \mathbf { M } _ { 1 } \} \right\} \mathrm { ~ ; ~ } } \\ & { \Pi _ { 2 } ( \mathbf { M } _ { 2 } ) } \\ & { = \left\{ ( \overline { { \alpha } } _ { 2 } ^ { ( 1 ) } , \pi _ { 2 } ^ { ( 2 ) } ) \in \Pi _ { W ^ { ( 1 ) } } \times \Pi _ { W ^ { ( 2 ) } } : \forall ( w ^ { ( 1 ) } , w ^ { ( 2 ) } ) \in W ^ { ( 1 ) } \times W ^ { ( 2 ) } , \{ \pi _ { 2 } ^ { ( 1 ) } ( w ^ { ( 1 ) } ) = \pi _ { 2 } ^ { ( 2 ) } ( w ^ { ( 2 ) } ) \} \Longleftrightarrow \{ ( w ^ { ( 1 ) } , w ^ { ( 2 ) } ) \in \mathbf { M } _ { 2 } \} \right\} \mathrm { ~ . ~ } } \end{array}
$$

For short, we write $\mathbf { M } = ( \mathbf { M } _ { 1 } , \mathbf { M } _ { 2 } ) \in \mathcal { M }$ and $\Pi ( \mathbf { M } )$ for the collection of labelings $( \pi _ { 1 } ^ { ( 1 ) } , \pi _ { 2 } ^ { ( 1 ) } , \pi _ { 1 } ^ { ( 2 ) } , \pi _ { 2 } ^ { ( 2 ) } )$ compatible with M. Given a matching M, we say that two edges $e \in E ^ { ( 1 ) }$ and $e ^ { \prime } \in \bar { E } ^ { ( 2 ) }$ are matched if the corre sponding incident nodes are matched. Importantly, as P is permutation invariant, $\mathbb { E } [ P _ { G ^ { ( 1 ) } , \pi _ { 1 } ^ { ( 1 ) } , \pi _ { 2 } ^ { ( 1 ) } } P _ { G ^ { ( 2 ) } , \pi _ { 1 } ^ { ( 2 ) } , \pi _ { 2 } ^ { ( 2 ) } } ]$ is the same for all $( \pi ^ { ( 1 ) } , \pi ^ { ( 2 ) } )$ in $\Pi ( \mathbf { M } )$

Note that, for a fixed matching M, all labeled graphs $G _ { \cup } \ ( \mathrm { r e s p . } \ G _ { \cap } , G _ { \oplus } )$ are isomorphic for $( \pi _ { 1 } ^ { ( 1 ) } , \pi _ { 1 } ^ { ( 2 ) } ) \in$ $\bar { \Pi } _ { 1 } ( \mathbf { M } _ { 1 } )$ and $( \pi _ { 2 } ^ { ( 1 ) } , \pi _ { 2 } ^ { ( 2 ) } ) \in \Pi _ { 2 } ( \mathbf { M } _ { 2 } )$ . Hence, we shall refer to quantities such as $| E _ { \oplus } | , | V _ { \oplus } | , | W _ { \oplus } | .$ . . associated to a matching M.

$$
\mathrm { W e ~ a l s o ~ h a v e } | E _ { \lfloor \omega \rfloor } | = | E ^ { ( 1 ) } | + | E ^ { ( 2 ) } | - | E _ { \cap } | \mathrm { ~ a n d ~ } | V _ { \cup } | = | V ^ { ( 1 ) } | + | V ^ { ( 2 ) } | - | \mathbf { M } _ { 1 } | \mathrm { ~ f o r ~ } ( \pi _ { 1 } ^ { ( 1 ) } , \pi _ { 1 } ^ { ( 2 ) } ) \in \bar { \Pi } _ { 1 } ( \mathbf { M } _ { 1 } ) .
$$

Sets of unmatched nodes and of semi-matched nodes. Write $U ^ { ( 1 ) } = ( U _ { 1 } ^ { ( 1 ) } , U _ { 2 } ^ { ( 1 ) } )$ , resp. $U ^ { ( 2 ) } =$ $( U _ { 1 } ^ { ( 2 ) } , U _ { 2 } ^ { ( 2 ) } )$ for the set of nodes in $( \pi _ { 1 } ^ { ( 1 ) } , \pi _ { 2 } ^ { ( 1 ) } ) [ G ^ { ( 1 ) } ]$ resp. $( \pi _ { 1 } ^ { ( 2 ) } , \pi _ { 2 } ^ { ( 2 ) } ) [ G ^ { ( 2 ) } ]$ that are not matched, namely the unmatched nodes, that is

$$
\begin{array} { r l r } { U _ { 1 } ^ { ( 1 ) } = \pi _ { 1 } ^ { ( 1 ) } ( V ^ { ( 1 ) } ) \setminus \pi _ { 1 } ^ { ( 2 ) } ( V ^ { ( 2 ) } ) \ ; } & { \qquad } & { U _ { 1 } ^ { ( 2 ) } = \pi _ { 1 } ^ { ( 2 ) } ( V ^ { ( 2 ) } ) \setminus \pi _ { 1 } ^ { ( 1 ) } ( V ^ { ( 1 ) } ) \ ; } \\ { U _ { 2 } ^ { ( 1 ) } = \pi _ { 2 } ^ { ( 1 ) } ( W ^ { ( 1 ) } ) \setminus \pi _ { 2 } ^ { ( 2 ) } ( W ^ { ( 2 ) } ) \ ; } & { \qquad } & { U _ { 2 } ^ { ( 2 ) } = \pi _ { 2 } ^ { ( 2 ) } ( W ^ { ( 2 ) } ) \setminus \pi _ { 2 } ^ { ( 1 ) } ( W ^ { ( 1 ) } ) \ ; } \end{array}
$$

Here, $V ^ { ( 1 ) }$ and $V ^ { ( 2 ) }$ include the possibly isolated nodes $v _ { 1 } ^ { ( i ) }$ and $v _ { 2 } ^ { ( i ) }$ , which are always matched since $\mathbf { M } _ { 1 } \in \mathcal { M } _ { 1 }$ . As the other nodes of $G ^ { ( 1 ) }$ and $G ^ { ( 2 ) }$ are not isolated, $U _ { 1 } ^ { ( 1 ) }$ and $U _ { 1 } ^ { ( 2 ) }$ only contain non-isolated nodes. Again, $| U _ { 1 } ^ { ( 1 ) } | , | U _ { 1 } ^ { ( 2 ) } | , | U _ { 2 } ^ { ( 1 ) } |$ , and $| U _ { 2 } ^ { ( 2 ) } |$ only depend on $( \pi ^ { ( 1 ) } , \pi ^ { ( 2 ) } )$ through the matching M. We have, for $i \in \{ 1 , 2 \}$

$$
| V ^ { ( i ) } | = | { \bf M } _ { 1 } | + | U _ { 1 } ^ { ( i ) } | \qquad | W ^ { ( i ) } | = | { \bf M } _ { 2 } | + | U _ { 2 } ^ { ( i ) } | \ .\tag{43}
$$

Write also

$$
\mathbf { M } _ { \mathrm { S M } } ( \mathbf { M } _ { 1 } ) \subset \mathbf { M } _ { 1 } , \qquad \mathbf { M } _ { \mathrm { S M } } ( \mathbf { M } _ { 2 } ) \subset \mathbf { M } _ { 2 } , \qquad \mathrm { a n d } \qquad \mathbf { M } _ { \mathrm { S M } } : = \mathbf { M } _ { \mathrm { S M } } ( \mathbf { M } ) = ( \mathbf { M } _ { \mathrm { S M } } ( \mathbf { M } _ { 1 } ) , \mathbf { M } _ { \mathrm { S M } } ( \mathbf { M } _ { 2 } ) )
$$

for the set of node matches of $( G ^ { ( 1 ) } , G ^ { ( 2 ) } )$ that are matched, and yet that are not pruned when creating the symmetric diference graph $G _ { \oplus }$ . This is the set of semi-matched nodes - i.e. at least one of their incident edges is not matched. The remaining pairs of nodes $\mathbf { M } \setminus \mathbf { M } _ { \mathrm { S M } }$ are said to be perfectly matched as all the edges incident to them are matched. We write $\mathbf { M } _ { \mathrm { P M } } = \mathbf { M } _ { \mathrm { P M } } ( \mathbf { M } )$ for the set of perfect matches in M. Note that

$$
| V ^ { ( 1 ) } | + | V ^ { ( 2 ) } | = | V _ { \oplus } | + | ( { \bf M } _ { \mathrm { S M } } ) _ { 1 } | + 2 | ( { \bf M } _ { \mathrm { P M } } ) _ { 1 } | ~ .\tag{44}
$$

Edit Distance between graphs. For any two graphs $G ^ { ( 1 ) }$ and $G ^ { ( 2 ) }$ , we define the so-called edit-distance.

$$
d ( G ^ { ( 1 ) } , G ^ { ( 2 ) } ) : = \operatorname* { m i n } _ { \mathbf { M } \in \mathcal { M } } | E _ { \oplus } | \ .\tag{45}
$$

Note that $d ( G ^ { ( 1 ) } , G ^ { ( 2 ) } ) = 0$ if and only if $G ^ { ( 1 ) }$ and $G ^ { ( 2 ) }$ are equivalent. As a consequence, if $G ^ { ( 1 ) }$ and $G ^ { ( 2 ) }$ are in $\mathcal { G } _ { \leq D }$ , the edit distance is equal to 0 if and only if $G ^ { ( 1 ) } = G ^ { ( 2 ) }$

## C.2 Entry-wise control of Γ

Proposition 3. Assume that Condition $( \mathcal { H } _ { s i g n a l } )$ holds for some $c _ { \mathrm { s } } \geq 1 0$ , that $2 ^ { 8 } D ^ { 4 } \leq d ,$ that $D \geq 2$ , and that $n \geq d$ . Consider two graphs $G ^ { ( 1 ) } , G ^ { ( 2 ) } \in \mathcal { G } _ { \leq D } ^ { \mathrm { e v e n } }$ . We have

$$
\left| \Gamma _ { G ^ { ( 1 ) } , G ^ { ( 2 ) } } - { \bf 1 } \{ G ^ { ( 1 ) } = G ^ { ( 2 ) } \} \right| = \left| \mathbb { E } \left[ \tilde { \Psi } _ { G ^ { ( 1 ) } } \tilde { \Psi } _ { G ^ { ( 2 ) } } \right] - { \bf 1 } \{ G ^ { ( 1 ) } = G ^ { ( 2 ) } \} \right| \leq 2 D ^ { - c _ { \mathrm { s } } ( d ( G ^ { ( 1 ) } , G ^ { ( 2 ) } ) \vee 1 ) } + \frac { 4 D ^ { 2 } } { d } { \bf 1 } \{ G ^ { ( 1 ) } = G ^ { ( 2 ) } \} \ ,
$$

where we recall that $d ( . , . )$ is the edit distance. The same bound holds for $G ^ { ( 2 ) } = \emptyset$ with the convention $d ( G ^ { ( 1 ) } , G ^ { ( 2 ) } ) = | E ^ { ( 1 ) } |$

In order to establish Proposition 2 in a self-contained way, we first bound, for a fixed graph, the number of graphs lying at a given edit distance from it.

Lemma 23. Fix $G ^ { ( 1 ) } \ \in \ { \mathcal { G } } _ { < D }$ and a positive integer ζ. The number of graphs $G ^ { ( 2 ) } \ \in \ { \mathcal G } _ { \leq D }$ such that $d ( G ^ { ( 1 ) } , G ^ { ( 2 ) } ) = \zeta$ is at most $\overline { { 4 } } \cdot 3 6 ^ { \zeta } D ^ { 3 \zeta + \bar { 1 } }$

Proof of Lemma 23. Since $G ^ { ( 1 ) }$ has no isolated node outside $\{ v _ { 1 } , v _ { 2 } \}$ , we have $| E ^ { ( 1 ) } | \le D , | V ^ { ( 1 ) } | \le D + 2 ;$ and $| W ^ { ( 1 ) } | \le D$ . Comparing $G ^ { ( 1 ) }$ and $G ^ { ( 2 ) }$ through the fully unmatched matching shows ${ \cal d } ( G ^ { ( 1 ) } , G ^ { ( 2 ) } ) \leq$ $| E ^ { ( 1 ) } | + | \dot { E ^ { ( 2 ) } } | \leq 2 D$ , so we only need to consider $\zeta \leq 2 D$

Fix a matching M achieving $d ( G ^ { ( 1 ) } , G ^ { ( 2 ) } ) = | \vec { E } _ { \oplus } | = \zeta ;$ , and write $a : = | E ^ { ( 1 ) } \cap E _ { \oplus } | \leq \zeta$ for the number of edges of $G ^ { ( 1 ) }$ that do not survive into $G ^ { ( 2 ) }$ , and $b : = \zeta - a = | E ^ { ( 2 ) } \cap E _ { \oplus } |$ for the number of edges of $G ^ { ( 2 ) }$ that are not already in $G ^ { ( 1 ) }$ . The graph $G ^ { ( 2 ) }$ (up to the isomorphism defining $\mathscr { G } _ { \leq D } )$ is entirely determined by (I) which a edges of $E ^ { ( 1 ) }$ are dropped, and (II) how the b remaining edges of $\scriptstyle { E _ { \oplus } }$ are attached.

(I) There are at most ${ \binom { | E ^ { ( 1 ) } | } { a } } \leq D ^ { a } \leq D ^ { \zeta }$ choices for a given $a \leq \zeta ,$ , hence at most $( \zeta + 1 ) D ^ { \zeta }$ choices summed over $a = 0 , \ldots , \zeta$ . Since $\zeta \leq 2 D$ , we have $\zeta + 1 \leq 3 D$ , so this contributes a factor at most $3 D ^ { \zeta + 1 }$

(II) By the no-isolated-node convention on $G _ { \oplus } .$ , the $b \leq \zeta$ new edges touch at most $2 b \leq 2 \zeta$ nodes that are not among the $\leq D + 2 \ ( { \mathrm { r e s p . } } \leq D ) $ original nodes of $\hat { V } ^ { ( 1 ) } ~ ( \mathrm { r e s p . } ~ W ^ { ( 1 ) } )$ . Hence each new edge has at most $( D + 2 ) + 2 \zeta \leq 6 D$ possible choices for its endpoint in V , and at most $D + 2 \zeta \leq 6 D$ possible choices for its endpoint in W (using $\zeta \leq 2 D )$ . Bounding the number of possible edge sets by the (larger) number of ordered edge sequences, there are at most $( 6 D \times 6 D ) ^ { b } = ( 3 6 D ^ { 2 } ) ^ { \bar { b } } \leq ( 3 6 D ^ { 2 } ) ^ { \bar { \zeta } }$ ways to place these b edges.

Multiplying (I) and (II), the number of possible $G ^ { ( 2 ) }$ is at most $3 D ^ { \zeta + 1 } \cdot ( 3 6 D ^ { \bar { 2 } } ) ^ { \zeta } = 3 \cdot 3 6 ^ { \zeta } D ^ { 3 \bar { \zeta } + 1 } \leq$ $4 \cdot 3 6 ^ { \zeta } D ^ { 3 \bar { \zeta } + 1 }$ □

Proof of Proposition 2. As explained earlier, we simply use

$$
\| \Gamma - \mathrm { I d } \| _ { \omega p } \leq \operatorname* { m a x } \left[ \operatorname* { m a x } _ { G ^ { ( 1 ) } \in \mathcal { G } _ { \leq D } ^ { \mathrm { e v e n } } } \left[ \left| \Gamma _ { G ^ { ( 1 ) } , \vartheta } \right| + \sum _ { G ^ { ( 2 ) } \in \mathcal { G } _ { \leq D } ^ { \mathrm { e v e n } } } \left| \Gamma _ { G ^ { ( 1 ) } , G ^ { ( 2 ) } } - { \bf 1 } \{ G ^ { ( 1 ) } = G ^ { ( 2 ) } \} \right| \right] , \sum _ { G ^ { ( 2 ) } \in \mathcal { G } _ { \leq D } ^ { \mathrm { e v e n } } } \left| \Gamma _ { G ^ { ( 2 ) } , \vartheta } \right| \right]\tag{46}
$$

Of-diagonal terms. Fix $G ^ { ( 1 ) } \in \mathcal { G } _ { < D } ^ { \mathrm { e v e n } }$ . Since both $G ^ { ( 1 ) }$ and $G ^ { ( 2 ) }$ have only even-degree nodes, so does their symmetric diference graph $G _ { \oplus }$ associated to any matching M – indeed, for any node $v , \deg _ { G _ { \oplus } } ( v ) =$ $\mathrm { d e g } _ { G ^ { ( 1 ) } } ( v ) + \mathrm { d e g } _ { G ^ { ( 2 ) } } ( v ) - 2 \mathrm { d e g } _ { \mathrm { s h a r e d } } ( v )$ (or simply $\deg _ { G ^ { ( 1 ) } } ( v )$ , resp. $\deg _ { G ^ { ( 2 ) } } ( v )$ , if v is unmatched), which is even whenever $\deg _ { G ^ { ( 1 ) } } ( v )$ and $\deg _ { G ^ { ( 2 ) } } ( v )$ are. Since $G _ { \oplus }$ is bipartite, $| E _ { \oplus } |$ equals the sum of the degrees on its $V _ { \oplus } { \cdot } \mathrm { s i d e }$ , hence is itself even. Moreover, if $G ^ { ( 1 ) } \neq G ^ { ( 2 ) }$ , then for the matching achieving $d ( G ^ { ( 1 ) } , G ^ { ( \bar { 2 } ) } ) = | E _ { \oplus } |$ the graph $G _ { \oplus }$ is non-empty, and (being simple, bipartite, with no isolated node, and all degrees even, hence $\geq 2 )$ it has at least 2 nodes of degree $\geq 2$ on each side, so $| E _ { \oplus } | \ge 4$ . We conclude that $d ( G ^ { ( 1 ) } , G ^ { ( 2 ) } )$ ), whenever nonzero, is an even integer $\geq 4$

Writing $N ( \zeta ) : = \# \{ G ^ { ( 2 ) } \in \mathcal { G } _ { \leq D } : d ( G ^ { ( 1 ) } , G ^ { ( 2 ) } ) = \zeta \}$ , Lemma 23 gives $N ( \zeta ) \leq 4 \cdot 3 6 ^ { \zeta } D ^ { 3 \zeta + 1 }$ (this bound applies a fortiori to $\mathcal { G } _ { < D } ^ { \mathrm { e v e n } } \subset \mathcal { G } _ { \leq D } )$ , so that, using Proposition 3 and the previous paragraph,

$$
\sum _ { G ^ { ( 2 ) } \in \mathcal { G } _ { \leq D } ^ { \mathrm { v e n } } , G ^ { ( 2 ) } \neq \mathcal { G } ^ { ( 1 ) } } | \Gamma _ { G ^ { ( 1 ) } , G ^ { ( 2 ) } } | \leq \sum _ { \zeta \geq 4 \atop \zeta \mathrm { ~ e v e n } } N ( \zeta ) \cdot 2 D ^ { - c _ { \mathrm { s } } \zeta } \leq \sum _ { \zeta \stackrel { ( 2 ) } { \mathrm { ~ e v e n ~ } } } 8 \cdot 3 6 ^ { \zeta } D ^ { 3 \zeta + 1 - c _ { \mathrm { s } } \zeta } \ .
$$

Writing $\theta : = 3 6 D ^ { 3 - c _ { \mathrm { s } } }$ , the right-hand side equals $\textstyle 8 D \sum _ { k \geq 2 } \theta ^ { 2 k } = 8 D \theta ^ { 4 } / ( 1 - \theta ^ { 2 } )$ as soon as $\theta < 1$

Diagonal and ∅-terms. By Proposition 3 applied with $G ^ { ( 2 ) } = G ^ { ( 1 ) }$ , for which $d ( G ^ { ( 1 ) } , G ^ { ( 1 ) } ) = 0$ so that $d ( G ^ { ( 1 ) } , G ^ { ( 1 ) } ) \vee 1 = 1$ , we get $\begin{array} { r } { | \Gamma _ { G ^ { ( 1 ) } , G ^ { ( 1 ) } } - 1 | \leq 2 D ^ { - c _ { \mathrm { s } } } + \frac { 4 D ^ { 2 } } { d } } \end{array}$ . Finally, using (12) together with the bound $\left| { \mathbb E } [ \tilde { \Psi } _ { G } ] \right| \leq 2 D ^ { - c _ { \mathrm { s } } | E | }$ established at the end of the proof of Proposition 3 below, and the same counting of graphs by number of edges as in the proof of Theorem 1 (at most $( D + 2 ) ^ { 2 e }$ graphs in $\mathcal { G } _ { \leq D } ^ { \mathrm { e v e n } }$ have e edges), we get, for $c _ { \mathrm { s } } \geq 1 0$ and $D \geq 2$

$$
\sum _ { G ^ { ( 2 ) } \in { \mathcal G } _ { \leq D } ^ { \mathrm { e v e n } } } | \Gamma _ { G ^ { ( 2 ) } , \emptyset } | \leq \sum _ { e \mathrm { ~ e v e n } } 2 ( D + 2 ) ^ { 2 e } D ^ { - c _ { \mathrm { s } } e } \leq 4 D ^ { - c _ { \mathrm { s } } / 2 } \ ,
$$

where the last inequality holds because $( D + 2 ) ^ { 2 } \leq D ^ { 4 }$ for $D \geq 2 \mathrm { : }$ : writing $y : = D ^ { 8 - 2 c _ { \mathrm { s } } } \leq 2 ^ { 8 - 2 c _ { \mathrm { s } } } \leq 2 ^ { - 1 2 }$ , the middle sum is at most $2 \textstyle \sum _ { k > 1 } y ^ { k } = 2 y / ( 1 - y ) \le 4 y \le 4 D ^ { - c _ { \mathrm { s } } / 2 }$ , since $8 - 2 c _ { \mathrm { s } } \leq - c _ { \mathrm { s } } / 2$ for $c _ { \mathrm { s } } \geq 1 0$ . Since $| E ^ { ( 1 ) } | \ge 1$ , the same bound $\left| { \mathbb E } [ \tilde { \Psi } _ { G } ] \right| \leq 2 D ^ { - c _ { \mathrm { s } } | E | }$ also gives the single-term bound $\left| \Gamma _ { G ^ { ( 1 ) } , \emptyset } \right| \leq 2 D ^ { - c _ { \mathrm { s } } }$ needed in the first branch of the max above.

Combining the three displays. For a fixed $G ^ { ( 1 ) }$ , the first part of the max in (46) is

$$
| \Gamma _ { G ^ { ( 1 ) } , \oplus } | + | \Gamma _ { G ^ { ( 1 ) } , G ^ { ( 1 ) } } - 1 | + \sum _ { G ^ { ( 2 ) } \neq G ^ { ( 1 ) } } | \Gamma _ { G ^ { ( 1 ) } , G ^ { ( 2 ) } } | \leq 2 D ^ { - c _ { \infty } } + \bigg ( 2 D ^ { - c _ { \infty } } + \frac { 4 D ^ { 2 } } { d } \bigg ) + 8 D \frac { \theta ^ { 4 } } { 1 - \theta ^ { 2 } } = 4 D ^ { - c _ { \infty } } + 8 D \frac { \theta ^ { 4 } } { 1 - \theta ^ { 2 } } + \frac { 4 D ^ { 2 } } { d } .
$$

It therefore sufices to check that $4 D ^ { - c _ { \mathrm { s } } } + 8 D \theta ^ { 4 } / ( 1 - \theta ^ { 2 } ) \le 4 D ^ { - c _ { \mathrm { s } } / 2 }$ . Since $D \geq 2$ and $c _ { \mathrm { s } } \geq 1 0$ , we have $\theta \leq 3 6 \cdot 2 ^ { - 7 } < 0 . 2 9$ , so that $8 / ( 1 - \theta ^ { 2 } ) \le 9$ and

$$
8 D { \frac { \theta ^ { 4 } } { 1 - \theta ^ { 2 } } } \le 9 \cdot 3 6 ^ { 4 } D ^ { 1 3 - 4 c _ { \mathrm { s } } } \le 9 \cdot 3 6 ^ { 4 } \cdot 2 ^ { - 2 2 } D ^ { - c _ { \mathrm { s } } / 2 } \le 3 . 7 D ^ { - c _ { \mathrm { s } } / 2 } ,
$$

where we used $D ^ { 1 3 - 7 c _ { \mathrm { s } } / 2 } \leq 2 ^ { 1 3 - 3 5 }$ . Besides, $4 D ^ { - c _ { \mathrm { s } } } \leq 4 \cdot 2 ^ { - 5 } D ^ { - c _ { \mathrm { s } } / 2 } = D ^ { - c _ { \mathrm { s } } / 2 } / 8 .$ and the claim follows by summing these two bounds. The second part, $\sum _ { G ^ { ( 2 ) } } \left| \Gamma _ { G ^ { ( 2 ) } , \emptyset } \right| \leq 4 D ^ { - c _ { \mathrm { s } } / 2 }$ , is bounded by the same quantity. This concludes the proof. □

## C.3 Proof of Proposition 3

The proof of this proposition follows the same steps as the proof of Proposition A.2 in [9], with some small changes. We start from the decomposition

$$
\mathbb { E } \left[ \bar { \Psi } _ { G ^ { ( 1 ) } } \bar { \Psi } _ { G ^ { ( 2 ) } } \right] ~ \sum _ { \mathbf { M } \in \mathcal { ( } G ^ { ( 2 ) } ) } \sum _ { | \mathbf { M } \in \mathcal { ( } G ^ { ( 2 ) } ) | } \sum _ { \mathbf { M } \in \mathcal { A } } n ^ { 2 - ( | V ^ { ( 1 ) } | + | V ^ { ( 2 ) } | ) / 2 } d ^ { - ( | W ^ { ( 1 ) } | + | W ^ { ( 2 ) } | ) / 2 } \sum _ { \{ \pi ^ { ( 1 ) } , \pi ^ { ( 2 ) } \} \in \Pi ( \mathbf { M } ) } \mathbb { E } [ \Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } ] ~ .
$$

We introduce $\mathcal { M } _ { \mathrm { P M } } \subset \mathcal { M }$ as the collection of perfect matchings, that is matchings M such that all the nodes in $( V ^ { ( 1 ) } , W ^ { ( 1 ) } )$ and $( V ^ { ( 2 ) } , W ^ { ( 2 ) } )$ are perfectly matched. Note that, if $\mathbf { M } \in \mathcal { M } _ { \mathrm { P M } }$ , then $G _ { \oplus }$ is the empty graph with $E _ { \oplus } = \emptyset$ . Besides, $\mathcal { M } _ { \mathrm { P M } } \neq \emptyset$ if and only if $G ^ { ( 1 ) }$ and $G ^ { ( 2 ) }$ are equivalent, which amounts to $G ^ { ( { \bar { 1 } } ) } = { \breve { G } } ^ { ( { \bar { 2 } } ) }$ for $G ^ { ( 1 ) } , \bar { G } ^ { ( 2 ) } \in \mathcal { G } _ { < D }$ . Let us call $\mathcal { M } ^ { \star }$ the collection of matchings M such that all nodes $v \in V ^ { ( 1 ) }$ (resp. in $V ^ { ( 2 ) } )$ whose degree is smaller than or equal to $2 m + 1$ arise in ${ { \bf { M } } _ { 1 } }$

Then, we shall bound the cross moment between $\tilde { \Psi } _ { G ^ { ( 1 ) } }$ and $\tilde { \Psi } _ { G ^ { ( 2 ) } }$ by applying Proposition 1. Write $\Pi _ { \mathrm { ( P M ) } } = \cup _ { \mathbf { M } \in \mathcal { M } _ { \mathrm { P M } } } \Pi ( \mathbf { M } )$ . Let us first argue that only the matchings $\mathbf { M } \in \mathcal { M } ^ { \star }$ contribute to the above sum over M. Consider $\mathbf { M } \in \mathcal { M } \backslash \mathcal { M } ^ { \star }$ and, by symmetry, assume that some node $v \in V ^ { ( 1 ) }$ with $\deg _ { G ^ { ( 1 ) } } ( v ) \leq 2 m + 1$ does not arise in ${ { \bf { M } } _ { 1 } }$ . Since $\mathcal { M } _ { 1 }$ enforces that $( v _ { 1 } ^ { ( 1 ) } , v _ { 1 } ^ { ( 2 ) } )$ and $( v _ { 2 } ^ { ( 1 ) } , v _ { 2 } ^ { ( 2 ) } )$ belong to the matching, we have $v \not \in \{ v _ { 1 } ^ { ( 1 ) } , v _ { 2 } ^ { ( 1 ) } \}$ , so that v is non-isolated by the definition of $\mathcal { G } _ { \leq D }$ and $\deg _ { G ^ { ( 1 ) } } ( v ) \in [ 1 : 2 m + 1 ]$ ]. Besides, for any $( \pi ^ { ( 1 ) } , \pi ^ { ( 2 ) } ) \in \Pi ( \mathbf { M } )$ , the equivalence in the definition of $\bar { \Pi } _ { 1 } ( { \mathbf { M } _ { 1 } } )$ ensures that $\pi _ { 1 } ^ { ( 1 ) } ( v ) \notin \pi _ { 1 } ^ { ( 2 ) } ( V ^ { ( 2 ) } )$ as v is matched to no node of $V ^ { ( 2 ) }$ . Hence, Proposition $\mathrm { 1 ( a ) }$ yields $\mathbb { E } \big [ \Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \big ] = 0$ for every $( \pi ^ { ( 1 ) } , \pi ^ { ( 2 ) } ) \in \Pi ( \mathbf { M } )$ , so that the term indexed by M vanishes. Note also that $\mathcal { M } _ { \mathrm { P M } } \subset \mathcal { M } ^ { \star }$ , all the nodes being matched for a perfect matching. Since $\mathbb { E } \big [ \Psi _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } \Psi _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \big ]$ only depends on $G ^ { ( 1 ) } , G ^ { ( 2 ) }$ , and M, we deduce from this proposition that

$$
\begin{array} { r l } & { \left| \mathbb { E } \left[ \tilde { \Psi } _ { G ^ { ( 1 ) } } \tilde { \Psi } _ { G ^ { ( 2 ) } } \right] - \mathbf { 1 } \{ G ^ { ( 1 ) } = G ^ { ( 2 ) } \} | \Pi _ { ( \mathrm { P M } ) } | n ^ { 2 - | V ^ { ( 1 ) } | } d ^ { - | W ^ { ( 1 ) } | } \frac { 1 } { | \mathrm { A u t } ( G ^ { ( 1 ) } ) | } \right| } \\ & { \leq 2 \frac { 1 } { \sqrt { | \mathrm { A u t } ( G ^ { ( 1 ) } ) | | \mathrm { A u t } ( G ^ { ( 2 ) } ) | } } \displaystyle \sum _ { \mathbf { M } \in \mathcal { A } ^ { \star } \setminus M _ { \mathrm { P M } } } ( 3 2 D ^ { 3 } ) ^ { \mathrm { c e } _ { n } } n ^ { ( | V ^ { ( 1 ) } | + | V ^ { ( 2 ) } | ) / 2 - | \mathbf { M } _ { 1 } | } d ^ { ( | W ^ { ( 1 ) } | + | W ^ { ( 2 ) } | ) / 2 - | \mathbf { M } _ { 2 } | } \left( \frac { 4 m \Delta } { d ^ { 1 / 4 } } \right) ^ { | E _ { 0 } | } } \\ & { \qquad \cdot \left( \frac { 1 } { d } \right) ^ { ( 2 m + 1 ) | V _ { \Psi } | / 4 + | W _ { \Psi } | / 2 - \mathrm { c } ( G _ { \Theta } ) / 2 + \mathrm { c e } _ { n } / 2 } + \frac { 8 m D \Delta } { d ^ { 1 / 4 } } \mathbf { 1 } \{ G ^ { ( 1 ) } = G ^ { ( 2 ) } \} \ , } \end{array}
$$

Write $C _ { \mathbf { M } }$ for the expression in the rhs of (47) in the sum over the matchings. We recall that $U _ { 1 } ^ { ( 1 ) }$ and $U _ { 1 } ^ { ( 2 ) }$ are the sets of unmatched nodes in resp. $V ^ { ( 1 ) }$ and $V ^ { ( 2 ) }$ , and we write $U _ { 1 } = U _ { 1 } ^ { ( 1 ) } \cup U _ { 1 } ^ { ( 2 ) }$ for the set of all unmatched nodes on the V-side, so that, by (43), $| U _ { 1 } | = | U _ { 1 } ^ { ( 1 ) } | + | U _ { 1 } ^ { ( 2 ) } |$ . Also, we have

$$
| V _ { \oplus } | = | V ^ { ( 1 ) } | + | V ^ { ( 2 ) } | - | \mathbf { M } _ { 1 } | - | ( \mathbf { M } _ { \mathrm { P M } } ) _ { 1 } | = | U _ { 1 } | + | \mathbf { M } _ { 1 } | - | ( \mathbf { M } _ { \mathrm { P M } } ) _ { 1 } |
$$

Similarly, $| W _ { \oplus } | = | W ^ { ( 1 ) } | + | W ^ { ( 2 ) } | - | \mathbf { M } _ { 2 } | - | ( \mathbf { M } _ { \mathrm { P M } } ) _ { 2 }$ . We deduce from (47) that

$$
C _ { \bf M } \le ( 3 2 D ^ { 3 } ) ^ { \mathrm { c c _ { u } } } \Big [ \sqrt { n } \Big ( { \frac { 4 m \Delta } { \sqrt { d } } } \Big ) ^ { 2 m + 2 } d ^ { 1 / 4 } \Big ] ^ { \vert U _ { 1 } \vert } \Big ( { \frac { 4 m \Delta } { d ^ { 1 / 4 } } } \Big ) ^ { \vert E _ { \oplus } \vert - ( 2 m + 2 ) \vert U _ { 1 } \vert } d ^ { b _ { \bf M } } \ ,
$$

where $b _ { \mathbf { M } } : = - ( 2 m + 1 ) [ | \mathbf { M } _ { \mathrm { 1 } } | - | ( \mathbf { M } _ { \mathrm { P M } } ) _ { \mathrm { 1 } } | ] / 4 - [ | \mathbf { M } _ { \mathrm { 2 } } | - | ( \mathbf { M } _ { \mathrm { P M } } ) _ { \mathrm { 2 } } | ] / 2 + \mathrm { c c } ( G _ { \oplus } ) / 2 - \mathrm { c c } _ { \mathrm { u } } / 2$ . Since $\csc ( G _ { \oplus } ) - \csc _ { \mathrm { u } }$ connected components of $G _ { \oplus }$ contain a matched node, we have cc $( G _ { \oplus } ) - \mathrm { c c } _ { \mathrm { u } } \leq | \mathbf { M } _ { 1 } | - | ( \mathbf { M } _ { \mathrm { P M } } ) _ { 1 } | + | \mathbf { M } _ { 2 } | -$ $| ( \mathbf { M } _ { \mathrm { P M } } ) _ { 2 } |$ . Combined with m $\geq 1$ , this yields $b _ { \mathbf { M } } \leq [ | \mathbf { M } _ { 1 } | - | ( \mathbf { M } _ { \mathrm { P M } } ) _ { 1 } | ] ( 1 - 2 m ) / 4 \leq 0 .$ , so that $d ^ { b _ { \mathbf { M } } } \leq 1$ Since we restrict our attention to $\mathbf { M } \in \mathcal { M } ^ { \star }$ , each node in $U _ { 1 }$ must have a degree at least equal to $2 m + 2$ in $G _ { \oplus }$ . This implies that $| E _ { \oplus } | \geq ( 2 m + 2 ) | U _ { 1 } |$ . We claim that, moreover,

$$
\mathrm { c c } _ { \mathrm { u } } \leq | U _ { 1 } | + \left[ | E _ { \oplus } | - ( 2 m + 2 ) | U _ { 1 } | \right] \ .\tag{48}
$$

Call an unmatched connected component exceptional if it contains no node of $U _ { 1 }$ , and write $\mathrm { c c } _ { \mathrm { u } } ^ { \mathrm { e x c } }$ for the number of such components, so that

$$
\mathrm { c c } _ { \mathrm { u } } \leq | U _ { 1 } | + \mathrm { c c } _ { \mathrm { u } } ^ { \mathrm { e x c } } ~ .\tag{49}
$$

Let us prove that $\mathrm { c c } _ { \mathrm { u } } ^ { \mathrm { e x c } } \leq 1$ and that $\mathrm { c c } _ { \mathrm { u } } ^ { \mathrm { e x c } } = 1$ enforces $| E _ { \oplus } | \geq ( 2 m + 2 ) | U _ { 1 } | + 4 .$ , which, together with (49), gives (48). Consider any exceptional connected component. By symmetry, we assume that it corresponds to a non-trivial connected component $G _ { r } ^ { ( 1 ) }$ of $G ^ { ( 1 ) }$ . Since the nodes $V _ { r } ^ { ( 1 ) }$ of $G _ { r } ^ { ( 1 ) }$ are matched, this implies that $V _ { r } ^ { ( 1 ) } \subset \{ v _ { 1 } ^ { ( 1 ) } , v _ { 2 } ^ { ( 1 ) } \}$ and that $v _ { 1 } ^ { ( 2 ) }$ and $v _ { 2 } ^ { ( 2 ) }$ are isolated in $G ^ { ( 2 ) }$ . Since $G ^ { ( 1 ) }$ has even degrees, we conclude $| V _ { r } ^ { ( 1 ) } | = 2$ . Thus, $\mathrm { c c } _ { \mathrm { u } } ^ { \mathrm { e x c } } \leq 1$ . If $\mathrm { c c } _ { \mathrm { u } } ^ { \mathrm { e x c } } = 1$ , then, since the degrees of $G ^ { ( 1 ) }$ and $G ^ { ( 2 ) }$ are even, $| E _ { \oplus } | - ( 2 m + 2 ) | U _ { 1 } |$ is larger than or equal to the number of edges in the unmatched exceptional connected component, which, in turn, is at least 4. We have proved (48).

Write $\rho _ { \mathrm { a } }$ and $\rho _ { \mathrm { b } }$ for the first and second terms in the maximum defining $\rho$ in Condition $( \mathcal { H } _ { s i g n a l } )$ , so that $\rho = \rho _ { \mathrm { a } } \lor \rho _ { \mathrm { b } } \lor \rho _ { \mathrm { a } }$ . Since $\sqrt { n } \big ( 4 m \Delta / \sqrt { d } \big ) ^ { 2 m + 2 } d ^ { 1 / 4 } = \rho _ { \mathrm { a } } ^ { 2 m + 2 } / ( 3 2 D ^ { 3 } )$ and $4 m \Delta / d ^ { 1 / 4 } = \rho _ { \mathrm { b } } / ( 3 2 D ^ { 3 } )$ , we deduce that

$$
C _ { \mathbf { M } } \le ( 3 2 D ^ { 3 } ) ^ { \mathrm { c c _ { u } } - | U _ { 1 } | - \left[ | E _ { \oplus } | - ( 2 m + 2 ) | U _ { 1 } | \right] } \rho _ { \mathsf { a } } ^ { ( 2 m + 2 ) | U _ { 1 } | } \rho _ { \mathsf { b } } ^ { | E _ { \oplus } | - ( 2 m + 2 ) | U _ { 1 } | } d ^ { b _ { \mathbf { M } } } \le \rho ^ { | E _ { \oplus } | } \ ,
$$

where we used $3 2 D ^ { 3 } \geq 1$ together with (48), then $d ^ { b _ { \mathbf { M } } } \leq 1$ and $\rho _ { \mathrm { a } } \vee \rho _ { \mathrm { b } } = \rho \le 1$ . We conclude that

$$
\left| \mathbb { E } \left[ \tilde { \Psi } _ { G ^ { ( 1 ) } } \tilde { \Psi } _ { G ^ { ( 2 ) } } \right] - \mathbf { 1 } \{ G ^ { ( 1 ) } = G ^ { ( 2 ) } \} \big | \Pi _ { \mathrm { ( P M ) } } \big | n ^ { 2 - | V ^ { ( 1 ) } | } d ^ { - | W ^ { ( 1 ) } | } \frac { 1 } { \big | \mathrm { A u t } \big ( G ^ { ( 1 ) } \big ) \big | } \right|
$$

$$
\leq 2 \frac { 1 } { \sqrt { | \mathrm { A u t } ( G ^ { ( 1 ) } ) | | \mathrm { A u t } ( G ^ { ( 2 ) } ) | } } \sum _ { \mathbf { M } \in \mathcal { M } ^ { \star } \setminus \mathcal { M } _ { \mathrm { P M } } } \rho ^ { | \cal { E } _ { \oplus } | } + \frac { 8 m D \Delta } { d ^ { 1 / 4 } } \mathbf { 1 } \{ G ^ { ( 1 ) } = G ^ { ( 2 ) } \} .
$$

Since

$$
| \Pi _ { \mathrm { ( P M ) } } | = | \mathrm { A u t } ( G ^ { ( 1 ) } ) | \frac { ( n - 2 ) ! } { ( n - | V ^ { ( 1 ) } | ) ! } \frac { d ! } { ( d - | W ^ { ( 1 ) } | ) ! } ,
$$

and $( | V ^ { ( 1 ) } | - 2 ) \vee | W ^ { ( 1 ) } | \leq D$ , we deduce that

$$
\begin{array} { r l } & { \left| | \Pi _ { ( \mathrm { P M } ) } | n ^ { 2 - | V ^ { ( 1 ) } | } d ^ { - | W ^ { ( 1 ) } | } \frac { 1 } { | \mathrm { A u t } ( G ^ { ( 1 ) } ) | } - 1 \right| \le | ( ( n - D - 4 ) / n ) ^ { D } ( ( d - D ) / d ) ^ { D } - 1 | } \\ & { \qquad \le 1 - \left( 1 - \displaystyle \frac { D + 4 } { n } - \displaystyle \frac { D } { d } \right) ^ { D } \le \displaystyle \frac { D ^ { 2 } + 4 D } { n } + \displaystyle \frac { D ^ { 2 } } { d } \enspace , } \end{array}
$$

since $D \leq ( n \land d ) / 4$ , which follows from $2 ^ { 8 } D ^ { 4 } \leq d , D \geq 2$ , and $n \geq d .$ . And so, using the assumption $n \geq d ,$ we conclude that

$$
\left| \mathbb { E } \left[ \tilde { \Psi } _ { G ( \mathfrak { t } ) } \hat { \Psi } _ { G ( \mathfrak { t } ^ { ( 2 ) } } \right] - \mathbf { 1 } \{ G ^ { ( 1 ) } = G ^ { ( 2 ) } \} \right| \leq \frac { 2 } { \sqrt { | \operatorname { A u t } ( G ^ { ( 1 ) } ) | | \operatorname { A u t } ( G ^ { ( 2 ) } ) | } } \sum _ { \mathbf { 1 } _ { \mathfrak { t } \in \mathcal { A } ^ { \star } \setminus A \log _ { \mathcal { A } } } } \rho ^ { | E _ { \mathfrak { t } } | } + \left( \frac { 4 D ^ { 2 } } { d } + \frac { 8 m D \Delta } { d ^ { 1 / 4 } } \right) \mathbf { 1 } \{ G ^ { ( 1 ) } = G ^ { ( 2 ) } \} \ .\tag{50}
$$

Shadow matchings. Given two sets $\bar { U } ^ { ( 1 ) } = ( \bar { U } _ { 1 } ^ { ( 1 ) } , \bar { U } _ { 2 } ^ { ( 1 ) } ) \subset V ^ { ( 1 ) } \times W ^ { ( 1 ) }$ and $\bar { U } ^ { ( 2 ) } = ( \bar { U } _ { 1 } ^ { ( 2 ) } , \bar { U } _ { 2 } ^ { ( 2 ) } ) \subset V ^ { ( 2 ) } \times$ $W ^ { ( 2 ) }$ and for a set of node matches $\mathbf { M } = ( \mathbf { M } _ { 1 } , \mathbf { \underline { { M } } } _ { 2 } ) \in \widetilde { \mathcal { M } } _ { 1 } \times \mathcal { M } _ { 2 }$ , where $\widetilde { \mathcal { M } } _ { 1 }$ stands for the collection of all matchings of $V ^ { ( 1 ) }$ and V<sup>(2)</sup> – without the additional constraint, imposed in the definition of $\mathcal { M } _ { 1 }$ , that $( v _ { 1 } ^ { ( 1 ) } , v _ { 1 } ^ { ( 2 ) } )$ and $( v _ { 2 } ^ { ( 1 ) } , v _ { 2 } ^ { ( 2 ) } )$ belong to the matching, as these two pairs, although always present in ${ { \bf { M } } _ { 1 } }$ , are usually not semi-matched –, we define $\mathcal { M } _ { \mathrm { s h a d o w } } ( \bar { \bar { U } } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) } , \underline { { \mathbf { M } } } )$ as the collection of matchings M satisfying

$$
{ \bf M } _ { \mathrm { S M } } ( { \bf M } ) = { \underline { { { \bf M } } } } , \quad ( \pi _ { b } ^ { ( a ) } ) ^ { - 1 } ( U _ { b } ^ { ( a ) } ) = { \bar { U } } _ { b } ^ { ( a ) } \mathrm { f o r } a = 1 , 2 \mathrm { a n d } b = 1 , 2 ,\tag{51}
$$

for any $( \pi ^ { ( 1 ) } , \pi ^ { ( 2 ) } ) \in \Pi ( \mathbf { M } )$ . Note that, as long as (51) is satisfied for one labeling $( \pi ^ { ( 1 ) } , \pi ^ { ( 2 ) } ) \in \Pi ( \mathbf { M } )$ , it is satisfied for all such $( \pi ^ { ( 1 ) } , \pi ^ { ( 2 ) } )$ . Equivalently, $\mathcal { M } _ { \mathrm { s h a d o w } } ( \bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) }$ , M) is the collection of all matchings that lead to the set M of semi-matched nodes and such that $\bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) }$ correspond to unmatched nodes in resp. $G ^ { ( 1 ) } , G ^ { ( 2 ) }$ . We say that these matchings satisfy a given shadow $( \bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) } , \underline { { \mathbf { M } } } )$ . The only thing that can vary between two elements of $\mathcal { M } _ { \mathrm { s h a d o w } } \bar { ( U ^ { ( 1 ) } , U ^ { ( 2 ) } }$ , M) is the matching of the nodes that are not in $\bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) }$ , nor part of a pair of nodes in M. This matching must however ensure that all of these nodes are perfectly matched.

Each matching M $\in \mathcal { M }$ gives rise to exactly one shadow, namely $\mathbf { s } ( \mathbf { M } ) : = \left( \bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) } , \mathbf { M } _ { \mathrm { S M } } ( \mathbf { M } ) \right)$ , so that the sets $\mathcal { M } _ { \mathrm { s h a d o w } } ( \bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) } , \underline { { \mathbf { M } } } )$ , when $( \bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) } , \underline { { \mathbf { M } } } )$ ranges over all shadows, form a partition of $\mathcal { M }$ . Let us identify those shadows whose matchings belong to $\mathcal { M } ^ { \star } \setminus \mathcal { M } _ { \mathrm { P M } }$ . Write $S ^ { \star }$ for the collection of shadows $( \bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) }$ , M) such that all nodes $v \in \bar { U } _ { 1 } ^ { ( 1 ) }$ (resp. $v \in \bar { U } _ { 1 } ^ { ( 2 ) } )$ have degree at least $2 m + 2$ in $G ^ { ( 1 ) }$ (resp. in $G ^ { ( 2 ) } )$ . Since the nodes of $V ^ { ( 1 ) } ~ ( \mathrm { r e s p . } ~ V ^ { ( 2 ) } )$ that do not arise in ${ { \bf { M } } _ { 1 } }$ are exactly those of $\bar { U } _ { 1 } ^ { ( 1 ) } ~ ( \mathrm { r e s p . } ~ \bar { U } _ { 1 } ^ { ( 2 ) } )$ , a matching M belongs to $\mathcal { M } ^ { \star }$ if and only $\mathrm { i f } \mathbf { s } ( \mathbf { M } ) \in \mathcal { S } ^ { \star } -$ membership in $\mathcal { M } ^ { \star }$ is a condition on the shadow only. Besides, a matching is perfect if and only if all its nodes are matched and none of them is semi-matched, that is if and only if its shadow is the empty shadow ${ \bf s } _ { \emptyset } : = ( \emptyset , \emptyset , \emptyset )$ , which does belong to $S ^ { \star }$ . Hence, $\mathcal { M } ^ { \star } \setminus \mathcal { M } _ { \mathrm { P M } }$ is the disjoint union of the sets $\mathcal { M } _ { \mathrm { s h a d o w } } ( \bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) } , \underline { { \mathbf { M } } } )$ over $( \bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) } , \underline { { \mathbf { M } } } ) \in \bar { \mathcal { S } } ^ { \star } \setminus \{ \mathbf { s } _ { \emptyset } \}$ . We then argue as in [9], proof of Proposition A.2, Steps 3 and 4. We have

$$
\begin{array} { r l } { \displaystyle \sum _ { \mathbf { M } \in \mathcal { M } ^ { \star } \setminus \mathcal { M } \mathbb { P } _ { \mathbf { M } } } \rho ^ { | E _ { \oplus } | } = } & { \displaystyle \sum _ { ( \bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) } , \mathbf { M } ) \in \mathcal { S } ^ { \star } \setminus \{ \mathbf { s } _ { \emptyset } \} } \sum _ { \mathbf { M } \in \mathcal { M } _ { \mathrm { s h a d o w } } ( \bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) } , \mathbf { M } ) } \rho ^ { | E _ { \oplus } | } } \\ & { \leq \rho ^ { [ d ( G ^ { ( 1 ) } , G ^ { ( 2 ) } ) \lor 1 ] / 2 } \displaystyle \sum _ { ( \bar { U } ^ { ( 2 ) } , \mathbf { M } ) \in \mathcal { S } ^ { \star } \setminus \{ \mathbf { s } _ { \emptyset } \} } \sum _ { \mathbf { M } \in \mathcal { M } _ { \mathrm { s h a d o w } } ( \bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) } , \mathbf { M } ) } \rho ^ { | E _ { \oplus } | / 2 } , } \end{array}
$$

since, by definition of the edit distance, $| E _ { \oplus } | \geq d ( G ^ { ( 1 ) } , G ^ { ( 2 ) } ) \vee 1$ as long as $\mathbf { M } \not \in \mathcal { M } _ { \mathrm { P M } }$

Arguing as in the proof of Lemma A.5 in [7] – see also Equation (117) in [7], which deals with matchings fixing both $v _ { 1 }$ and $v _ { 2 } .$ , and Lemma A.5 in $[ 9 ] \ - - ,$ we have

$$
| \mathcal { M } _ { \mathrm { s h a d o w } } ( \bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) } , { \bf M } ) | \leq \operatorname* { m i n } ( | \mathrm { A u t } ( G ^ { ( 1 ) } ) | , | \mathrm { A u t } ( G ^ { ( 2 ) } ) | ) ~ .
$$

Therefore,

$$
\sum _ { \mathbf { M } \in \mathcal { A } ^ { \star } \setminus \cup M _ { \mathrm { P M } } } \rho ^ { | E _ { \oplus } | / 2 } \leq \operatorname* { m i n } ( | \mathrm { A u t } ( G ^ { ( 1 ) } ) | , | \mathrm { A u t } ( G ^ { ( 2 ) } ) | ) \sum _ { ( \bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) } , \mathbf { M } ) \in \mathcal { S } ^ { \star } \setminus \{ \mathbf { s } _ { \emptyset } \} } \rho ^ { ( | \bar { U } ^ { ( 1 ) } | + | \bar { U } ^ { ( 2 ) } | + | \mathbf { { \vec { m } } } | ) / 4 } ,
$$

since, for any $\mathbf { M } \in \mathcal { M } _ { \mathrm { s h a d o w } } ( \bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) } , \underline { { \mathbf { M } } } )$ , each unmatched node and each matched node that is not perfectly matched connects to at least one edge in $E _ { \oplus }$ . This leads us to

$$
\begin{array} { r l } & { \left| \mathbb { E } \left[ \tilde { \Psi } _ { G ^ { ( 1 ) } } \tilde { \Psi } _ { G ^ { ( 2 ) } } \right] - \mathbf { 1 } \{ G ^ { ( 1 ) } = G ^ { ( 2 ) } \} \right| } \\ & { \leq 2 \rho ^ { \{ d ( G ^ { ( 1 ) } , G ^ { ( 2 ) } ) \vee 1 \} / 2 } \underset { ( \bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) } , \mathbf { M } ) \in \mathcal { S } ^ { \star } \backslash \{ \mathfrak { s } _ { 0 } \} } { \sum } \rho ^ { ( | \bar { U } ^ { ( 1 ) } | + | \bar { U } ^ { ( 2 ) } | + | \mathbf { M } | ) / 4 } + \left[ \frac { 4 D ^ { 2 } } { d } + \frac { 8 m D \Delta } { d ^ { 1 / 4 } } \right] \mathbf { 1 } \{ G ^ { ( 1 ) } = G ^ { ( 2 ) } \} . } \end{array}
$$

We then conclude as in the proof of Proposition $\mathrm { A . 2 ~ ( S t e p ~ 5 ) }$ in [9]: assuming that $\rho \le D ^ { - 1 6 c _ { \mathrm { s } } }$ with $c _ { \mathrm { s } } \geq 1 0$ and $D \geq 2$ , we have by summing over $\bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) }$ , M and grouping them by cardinality – the last sum being

bounded by the sum of $\rho ^ { ( | \bar { U } ^ { ( 1 ) } | + | \bar { U } ^ { ( 2 ) } | + | \underline { { \mathbf { M } } } | ) / 4 }$ over all non-empty shadows, that is by dropping the constraint $( \bar { U } ^ { ( 1 ) } , \bar { U } ^ { ( 2 ) } , \underline { { \mathbf { M } } } ) \in \mathcal { S } ^ { \star } - $

$$
\begin{array} { r l r } {  {  \mathbb { E } [ \tilde { \Psi } _ { G ^ { ( 1 ) } } \tilde { \Psi } _ { G ^ { ( 2 ) } } ] - \mathbf { 1 } \big \{ G ^ { ( 1 ) } = G ^ { ( 2 ) } \big \}  \leq D ^ { - c _ { \mathrm { s } } [ d ( G ^ { ( 1 ) } , G ^ { ( 2 ) } ) \vee 1 ] } + ( \frac { 4 D ^ { 2 } } { d } + \frac { 8 m D \Delta } { d ^ { 1 / 4 } } ) \mathbf { 1 } \big \{ G ^ { ( 1 ) } = G ^ { ( 2 ) } \big \} } } \\ & { } & \\ & { } & { \leq 2 D ^ { - c _ { \mathrm { s } } [ d ( G ^ { ( 1 ) } , G ^ { ( 2 ) } ) \vee 1 ] } + \frac { 4 D ^ { 2 } } { d } \mathbf { 1 } \big \{ G ^ { ( 1 ) } = G ^ { ( 2 ) } \big \} , ~ } \end{array}
$$

by definition of $\rho$ and Condition $( \mathcal { H } _ { s i g n a l } )$

Finally, it remains to bound the term $\mathbb { E } \big [ \tilde { \Psi } _ { G ^ { ( 1 ) } } \big ]$ which corresponds to the cross moment of 1 and $\tilde { \Psi } _ { G ^ { ( 1 ) } }$ Note that the previous arguments and derivations still hold in the degenerate situation where $G ^ { ( 2 ) }$ is the empty graph, so that $\Psi _ { G ^ { ( 2 ) } } = 1 , G _ { \oplus } = G ^ { ( 1 ) }$ and $| E _ { \oplus } | = | E ^ { ( 1 ) } |$ . There is then no matched node, $\begin{array} { r l } { \operatorname { c c } _ { \mathrm { { u } } } = } & { { } } \end{array}$ $\csc ( G ^ { ( 1 ) } )$ , and the set $U _ { 1 }$ of unmatched nodes on the V -side is the set of non-isolated nodes of $V ^ { ( 1 ) }$ ; as it contains ${ \cal V } ^ { ( 1 ) } \ \backslash \ \{ v _ { 1 } , v _ { 2 } \}$ , we have $n ^ { ( | V ^ { ( 1 ) } | - 2 ) / 2 } \leq n ^ { | U _ { 1 } | / 2 }$ . In that degenerate situation, every non-isolated node of $V ^ { ( 1 ) }$ has degree at least 2m + 2 –otherwise $\mathbb { E } \big [ \tilde { \Psi } _ { G ^ { ( 1 ) } } \big ] = 0$ by Proposition 1(a)– and we again have $\mathrm { c c } _ { \mathrm { u } } \leq | U _ { 1 } | + | E ^ { ( 1 ) } | - ( 2 m + 2 ) | U _ { 1 } |$ so that (48) remains valid. In particular, we readily get

$$
\left| \mathbb { E } [ \tilde { \Psi } _ { G ^ { ( 1 ) } } ] \right| \leq \frac { 2 } { \sqrt { \left| \mathrm { A u t } ( G ^ { ( 1 ) } ) \right| } } \rho ^ { | E ^ { ( 1 ) } | } \leq 2 D ^ { - c _ { s } | E ^ { ( 1 ) } | } \ .
$$

## D Proof of Lemma 9

Fix a graph G and a labeling π of G.

Case 1. First, we deal with the case where either $v _ { 1 }$ or $v _ { 2 }$ is isolated in G. Then, we readily check that $\begin{array} { r } { \mathbb { E } [ x \Psi _ { G , \pi } ] = \frac { 1 } { K } \mathbb { E } [ \Psi _ { G , \pi } ] } \end{array}$ . Besides, if G contains only one non-trivial connected component, then it follows from the definition of $\Psi _ { G , \pi }$ that $\mathbb { E } [ \Psi _ { G , \pi } ] = 0$ , which leads to $\mathbb { E } [ x \tilde { \Psi } _ { G } ] = 0$ . If G is made of several connected components, then we can decompose G into two disjoint graphs $G _ { 1 }$ and $G _ { 2 }$ and we are in a position to apply Proposition 1. In particular, we also have $\mathbb { E } [ x \tilde { \Psi } _ { G } ] = 0$ if one node $v \in V \setminus \{ v _ { 1 } , v _ { 2 } \}$ has a degree smaller than $2 m + 2$ . Otherwise, we have

$$
\left| \mathbb { E } \left[ \Psi _ { G , \pi } \right] \right| \leq 2 ( 3 2 D ^ { 3 } ) ^ { \mathrm { c c } ( G ) } \left( 4 m \frac { \Delta } { d ^ { 1 / 4 } } \right) ^ { \left| E \right| } \left( \frac { 1 } { d } \right) ^ { ( 2 m + 1 ) \left| V _ { \oplus } \right| / 4 + \left| W \right| / 2 } \ : \ : ,
$$

where $| V _ { \oplus } |$ is the number of non-isolated nodes of V in the disjoint union $G _ { 1 } \sqcup G _ { 2 }$ (Proposition $1 ( \mathrm { b } )$ is stated in terms of $V _ { \oplus }$ , which excludes isolated nodes), and where we also used $c c ( G ) \leq | V _ { \oplus } |$ (each non-trivial connected component contains at least one non-isolated node of $V )$ . Since only v and v may be isolated in a graph of $\mathscr { G } _ { \leq D }$ , we have $| V _ { \oplus } | \geq | V | - 2$ . Since $c c ( G ) \leq | V _ { \oplus } |$ and d is supposed to be large enough compared to D, we have

$$
( 3 2 D ^ { 3 } ) ^ { \mathrm { c e } ( G ) } ( 1 / d ) ^ { ( 2 m + 1 ) | V _ { \complement } | / 4 } \leq ( 3 2 D ^ { 3 } ) ^ { | V | - 2 } ( 1 / d ) ^ { ( 2 m + 1 ) ( | V | - 2 ) / 4 } = ( 3 2 D ^ { 3 } ) ^ { | V | - 2 } ( 1 / d ) ^ { ( 2 m + 1 ) | V | / 4 } \cdot d ^ { ( 2 m + 1 ) / 2 }
$$

Summing over all labelings π, we arrive at

$$
\begin{array} { l } { { \displaystyle \left| \mathbb { E } [ x \tilde { \Psi } _ { G } ] \right| \le 2 \frac { d ^ { ( 2 m + 1 ) / 2 } } { K d ^ { m + 1 / 2 } } ( 3 2 D ^ { 3 } ) ^ { | V | - 2 } \left[ \left( 4 m \frac { \Delta } { d ^ { 1 / 2 } } \right) ^ { 2 m + 2 } d ^ { 1 / 4 } \sqrt { n } \right] ^ { ( | V | - 2 ) } \left( 4 m \frac { \Delta } { d ^ { 1 / 4 } } \right) ^ { | E | - ( 2 m + 2 ) ( | V | - 2 ) } } } \\ { { \displaystyle \qquad \le 2 \frac { 1 } { K } \left[ \left( 4 m \frac { \Delta } { d ^ { 1 / 2 } } \right) ^ { 2 m + 2 } ( 3 2 D ^ { 3 } ) d ^ { 1 / 4 } \sqrt { n } \right] ^ { ( | V | - 2 ) } \left( 4 m \frac { \Delta } { d ^ { 1 / 4 } } \right) ^ { | E | - ( 2 m + 2 ) ( | V | - 2 ) } } } \\ { { \displaystyle \qquad \le \frac { 2 } { K } \rho ^ { | E | } ~ , } } \end{array}
$$

where the first line uses the bound on $( 3 2 D ^ { 3 } ) ^ { \mathrm { c c } ( G ) } ( 1 / d ) ^ { ( 2 m + 1 ) | V _ { \oplus } | / 4 }$ established above, the second line uses $d ^ { ( 2 m + 1 ) / 2 } / d ^ { m + 1 / 2 } = 1$ , and the last line uses that each node $v \in V \setminus \{ v _ { 1 } , v _ { 2 } \}$ has a degree at least $2 m + 2$ We have shown the desired inequality.

Let us turn to the more interesting situation where neither $v _ { 1 }$ nor $v _ { 2 }$ is isolated in $G .$ Case 2. First, we focus on the case where $v _ { 1 }$ and $v _ { 2 }$ belong to the same connected component of G. We decompose $G$ into $G _ { 0 }$ and $G _ { 1 }$ where $G _ { 1 }$ stands for this connected component and $G _ { 0 }$ contains all the remaining connected components. We use the convention that $\Psi _ { G _ { 0 } , \pi } = 1 \mathrm { ~ i f ~ } G _ { 0 }$ is the empty graph. Then, we can decompose

$$
\mathbb { E } \left[ x \Psi _ { G , \pi } \right] = \mathbb { E } [ x P _ { G _ { 1 } , \pi } \Psi _ { G _ { 0 } , \pi } ] - \mathbb { E } [ x \Psi _ { G _ { 0 } , \pi } ] \mathbb { E } [ P _ { G _ { 1 } , \pi } ] = \mathbb { E } [ x P _ { G _ { 1 } , \pi } \Psi _ { G _ { 0 } , \pi } ] - \frac { 1 } { K } \mathbb { E } [ P _ { G _ { 1 } , \pi } ] \mathbb { E } [ \Psi _ { G _ { 0 } , \pi } ] \ .\tag{52}
$$

Define the graph ${ \bar { G } _ { 1 } } = ( { \bar { V } _ { 1 } } , { \bar { W } _ { 1 } } , { \bar { E } _ { 1 } } )$ by identifying $v _ { 1 }$ and $v _ { 2 }$ into $v _ { 1 }$ and removing all edges that appear twice, that is all edges that connect $v _ { 1 }$ and $v _ { 2 }$ to a common neighbor $w \in W$ . One checks that

$$
\mathbb { E } [ x P _ { G _ { 1 } , \pi } \Psi _ { G _ { 0 } , \pi } ] = \frac { 1 } { K } \left( \frac { \Delta } { \sqrt { d } } \right) ^ { \vert E _ { 1 } \vert - \vert \bar { E } _ { 1 } \vert } \mathbb { E } [ P _ { \bar { G } _ { 1 } , \pi } \Psi _ { G _ { 0 } , \pi } ] \ .
$$

As in Case 1, we readily check that $\mathbb { E } \left[ x \Psi _ { G , \pi } \right] = 0$ if one node $v \in V \setminus \{ v _ { 1 } , v _ { 2 } \}$ has a degree smaller than $2 m + 2$ . Define $\bar { G } = ( \bar { V } , \bar { W } , \bar { E } )$ by combining $\bar { G } _ { 1 }$ and $G _ { 0 }$ and removing isolated nodes. To control the term $\mathbb { E } \big [ P _ { \bar { G } _ { 1 } , \pi } \Psi _ { G _ { 0 } , \pi } \big ] .$ , we apply Proposition 1 (d). Note that $\operatorname { c c } ( G _ { 0 } ) \leq | V | - 2$ , since $G _ { 0 }$ does not contain $v _ { 1 }$ or $v _ { 2 }$ (both being in $G _ { 1 } )$ and every non-trivial connected component has at least one node in $V .$ . This leads us to

$$
\left| \mathbb { E } [ x P _ { G _ { 1 } , \pi } \Psi _ { G _ { 0 } , \pi } ] \right| \le \frac { 1 } { K } ( 3 2 D ^ { 3 } ) ^ { \mathrm { c c } ( G _ { 0 } ) } \left( 4 m \frac { \Delta } { d ^ { 1 / 4 } } \right) ^ { | E | } \left( \frac { 1 } { d } \right) ^ { ( | E _ { 1 } | - | \widehat { E } _ { 1 } | ) / 4 + ( 2 m + 1 ) | \widehat { V } | / 4 + | \widehat { W } | / 2 - \mathrm { c c } ( \widehat { G } _ { 1 } ) / 2 }\tag{53}
$$

Applying Lemma 11 and Proposition 1, we get

$$
\frac { 1 } { K } \left| \mathbb { E } [ P _ { G _ { 1 } , \pi } ] \mathbb { E } [ \Psi _ { G _ { 0 } , \pi } ] \right| \leq \frac { 4 } { K } ( 3 2 D ^ { 3 } ) ^ { \mathrm { c c } ( G _ { 0 } ) } \left( 4 m \frac { \Delta } { d ^ { 1 / 4 } } \right) ^ { | E | } \left( \frac { 1 } { d } \right) ^ { | W | / 2 + ( 2 m + 1 ) | V | / 4 - 1 / 2 } \ .
$$

Then, we combine the two previous inequalities with the decomposition (52) and we sum over all possible labelings π. This leads us to

$$
\left| \mathbb { E } \left[ x \tilde { \Psi } _ { G } \right] \right| \leq \frac { 4 } { K } ( 3 2 D ^ { 3 } ) ^ { \mathrm { c c } ( G _ { 0 } ) } \left[ \left( 4 m \frac { \Delta } { d ^ { 1 / 2 } } \right) ^ { 2 m + 2 } d ^ { 1 / 4 } \sqrt { n } \right| ^ { | V | - 2 } \left( 4 m \frac { \Delta } { d ^ { 1 / 4 } } \right) ^ { | E | - ( 2 m + 2 ) ( | V | - 2 ) } \left[ \frac { 1 } { d ^ { m } } + \frac { 1 } { d ^ { a } } \right] ;\tag{54}
$$

$$
a : = ( | E _ { 1 } | - | \bar { E } _ { 1 } | ) / 4 + ( 2 m + 1 ) [ | \bar { V } | - | V | + 2 ] / 4 + [ | \bar { W } | - | W | ] / 2 - \csc ( \bar { G } _ { 1 } ) / 2 \ .
$$

Let us prove that a is non-negative. Set $r = | W | - | \bar { W } | ;$ : these r nodes are connected to $v _ { 1 }$ and $v _ { 2 }$ in $G$ and do not have any other neighbor. If $\csc ( \bar { G } _ { 1 } ) = 0$ , we readily deduce that $a \geq 0$ . If $\csc ( \bar { G } _ { 1 } ) > 0$ , then each connected component of $\bar { G } _ { 1 }$ that does not contain $v _ { 1 }$ contains a node w of degree at least 4 in $G$ which is also connected to $v _ { 1 }$ and $v _ { 2 }$ in $G .$ Denoting by cc<sub>−</sub> $\left( \hat { G } _ { 1 } \right)$ the number of connected components of $\bar { G } _ { 1 }$ that do not contain $v _ { 1 }$ , we deduce that

$$
( | E _ { 1 } | - | \bar { E } _ { 1 } | ) \geq 2 [ | W | - | \bar { W } | ] + 2 \mathrm { c c } _ { - } ( \bar { G } _ { 1 } ) \ .
$$

We conclude that

$$
a \geq ( 2 m + 1 ) [ \vert { \bar { V } } \vert - \vert { V } \vert + 2 ] / 4 - ( \csc ( { \bar { G } } _ { 1 } ) - \csc ( { \bar { G } } _ { 1 } ) ) / 2 \geq 0 ~ .
$$

Hence, we deduce from (54) and the definition of $\rho$ that

$$
\left| \mathbb { E } \left[ x \tilde { \Psi } _ { G } \right] \right| \leq \frac { 8 } { K } \rho ^ { | E | }
$$

Case 3. Finally, we consider the case where $v _ { 1 }$ and $v _ { 2 }$ belong to distinct connected components of G. As in the previous cases, we readily check that $\mathbb { E } \left[ x \Psi _ { G , \pi } \right] = 0$ if one node $v \in V \setminus \{ v _ { 1 } , v _ { 2 } \}$ has a degree smaller than $2 m + 2$ . We decompose $G$ into $G _ { 0 } , G _ { 1 }$ , and $G _ { 2 }$ where $G _ { 1 } \ ( \mathrm { r e s p . } \ G _ { 2 } )$ stands for the connected component of $v _ { 1 }$ (resp. v ) and $G _ { 0 }$ contains all the remaining connected components. We use the convention that $\Psi _ { G _ { 0 } , \pi } = 1$ if $G _ { 0 }$ is the empty graph. Define the graph ${ { \bar { G } } _ { 1 } } = \left( { { { \bar { V } } _ { 1 } } , { { \bar { W } } _ { 1 } } , { { \bar { E } } _ { 1 } } } \right)$ by identifying $v _ { 1 }$ and $v _ { 2 }$ into $v _ { 1 }$ in the disjoint union $G _ { 1 } \sqcup G _ { 2 }$ . Note that $\bar { G } _ { 1 }$ is connected. Then, we can decompose

$$
\begin{array} { r l } & { { \mathbb E } \left[ x \Psi _ { G , \pi } \right] = { \mathbb E } [ x P _ { G _ { 1 } , \pi } P _ { G _ { 2 } , \pi } \Psi _ { G _ { 0 } , \pi } ] + { \mathbb E } [ x \Psi _ { G _ { 0 } , \pi } ] { \mathbb E } [ P _ { G _ { 1 } , \pi } ] { \mathbb E } [ P _ { G _ { 2 } , \pi } ] } \\ & { \quad \quad \quad - { \mathbb E } [ x \Psi _ { G _ { 0 } , \pi } P _ { G _ { 1 } , \pi } ] { \mathbb E } [ P _ { G _ { 2 } , \pi } ] - { \mathbb E } [ x \Psi _ { G _ { 0 } , \pi } P _ { G _ { 2 } , \pi } ] { \mathbb E } [ P _ { G _ { 1 } , \pi } ] } \\ & { \quad \quad \quad = \displaystyle \frac { 1 } { K } { \mathbb E } [ P _ { \bar { G } _ { 1 } , \pi } \Psi _ { G _ { 0 } , \pi } ] + \displaystyle \frac { 1 } { K } { \mathbb E } [ \Psi _ { G _ { 0 } , \pi } ] { \mathbb E } [ P _ { G _ { 1 } , \pi } ] { \mathbb E } [ P _ { G _ { 2 } , \pi } ] } \\ & { \quad \quad \quad \quad - \displaystyle \frac { 1 } { K } { \mathbb E } [ \Psi _ { G _ { 0 } , \pi } P _ { G _ { 1 } , \pi } ] { \mathbb E } [ P _ { G _ { 2 } , \pi } ] - \displaystyle \frac { 1 } { K } { \mathbb E } [ \Psi _ { G _ { 0 } , \pi } P _ { G _ { 2 } , \pi } ] { \mathbb E } [ P _ { G _ { 1 } , \pi } ] \ . } \end{array}\tag{55}
$$

Each of these expressions is bounded using either Proposition 1 or Lemma 11, using as in Case 2 that $\operatorname { c c } ( G _ { 0 } ) \leq | V | - 2$ . This leads us to

$$
\left| \mathbb { E } \left[ x \Psi _ { G , \pi } \right] \right| \leq \frac { 1 6 } { K } ( 3 2 D ^ { 3 } ) ^ { \operatorname { c c } ( G _ { 0 } ) } \left( 4 m \frac { \Delta } { d ^ { 1 / 4 } } \right) ^ { \left| E \right| } \left( \frac { 1 } { d } \right) ^ { \left| W \right| / 2 + ( 2 m + 1 ) ( \left| V \right| - 1 ) / 4 - 1 / 2 }
$$

Summing over all possible labelings π yields

$$
\left| \mathbb { E } \left[ x \tilde { \Psi } _ { G } \right] \right| \leq \frac { 1 6 } { K } ( 3 2 D ^ { 3 } ) ^ { \mathrm { c c } ( G _ { 0 } ) } \left[ \left( 4 m \frac { \Delta } { d ^ { 1 / 2 } } \right) ^ { 2 m + 2 } d ^ { 1 / 4 } \sqrt { n } \right] ^ { | V | - 2 } \left( 4 m \frac { \Delta } { d ^ { 1 / 4 } } \right) ^ { | E | - ( 2 m + 2 ) ( | V | - 2 ) } d ^ { 1 / 2 - ( 2 m + 1 ) / 4 }\tag{56}
$$

Since $m \geq 1$ , we conclude that

$$
\left| \mathbb { E } \left[ x \tilde { \Psi } _ { G } \right] \right| \leq \frac { 1 6 } { K d ^ { 1 / 4 } } \rho ^ { | E | } \ .
$$

This concludes the proof.

## E Proof of reduction lemmas

Proof of Lemma 5. The distribution of $( x , Y )$ is invariant under any permutation of the rows in $[ 3 : n ]$ and any permutation of $[ 1 : d ]$ . As a consequence, the supremum in (1) may be restricted to polynomials $P$ that satisfy these invariance properties (indeed, for any $P ,$ , its symmetrization $\bar { P } : = \mathbb { E } _ { g } [ P \circ g ]$ over the invariance group satisfies $\mathbb { E } [ x \bar { P } ] = \mathbb { E } [ x P ]$ by invariance of the law of $( x , Y )$ , and $\mathbb { E } [ \bar { P } ^ { 2 } ] \leq \mathbb { E } [ \bar { P } ^ { 2 } ]$ by Jensen’s inequality, so that $\bar { P }$ is admissible in (1) as soon as P is, achieves the same value, and is itself invariant). Since $\left( 1 , ( P _ { G } ) _ { G \in { \mathcal G } _ { \leq D } } \right)$ is a basis of the invariant polynomials with degrees at most equal to $D$ (see Section 5), this concludes the proof. □

Proof of Lemma 6. Fix $\pi = ( \pi _ { 1 } , \pi _ { 2 } )$ . We have

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } [ P _ { G , \pi _ { 1 } , \pi _ { 2 } } ( Y ) ] = \mathbb { E } _ { a , z , k ^ { * } , \sigma } \Bigg [ \prod _ { e \in E } z _ { \pi _ { 1 } ( l ( e ) ) } X _ { k ^ { * } ( \pi _ { 1 } ( l ( e ) ) ) , \pi _ { 2 } ( r ( e ) ) } \Bigg ] } \\ & { \displaystyle \qquad = \left( \frac { \Delta } { \sqrt { d } } \right) ^ { | E | } \mathbb { E } _ { a , z , k ^ { * } , \sigma } \Bigg [ \prod _ { e \in E } z _ { \pi _ { 1 } ( l ( e ) ) } a _ { \pi _ { 2 } ( r ( e ) ) } \mu _ { k ^ { * } ( \pi _ { 1 } ( l ( e ) ) ) , \sigma ( \pi _ { 2 } ( r ( e ) ) ) } \Bigg ] ~ . } \end{array}
$$

Similarly, we have

$$
\mathbb { E } [ x P _ { G , \pi _ { 1 } , \pi _ { 2 } } ( Y ) ] = \left( \frac { \Delta } { \sqrt { d } } \right) ^ { | E | } \mathbb { E } _ { a , z , k ^ { * } , \sigma } \bigg [ x \prod _ { e \in E } z _ { \pi _ { 1 } ( l ( e ) ) } a _ { \pi _ { 2 } ( r ( e ) ) } \mu _ { k ^ { * } ( \pi _ { 1 } ( l ( e ) ) ) , \sigma ( \pi _ { 2 } ( r ( e ) ) ) } \bigg ] \enspace .
$$

Proof of a). Write $w ^ { * } \in W$ for (one of the) nodes of odd degree, and write α for its odd degree. We have

$$
\begin{array} { l } { { \displaystyle \mathbb { E } [ P _ { G , \pi _ { 1 } , \pi _ { 2 } } ( Y ) ] = \left( \frac { \Delta } { \sqrt { d } } \right) ^ { | E | } \mathbb { E } _ { a , z , k ^ { * } , \sigma } \Bigg [ \displaystyle \prod _ { \mathrm { { c } } \in E \backslash \{ ( e ) , E \tau ( e ) = - u ^ { * } \} } z _ { \pi _ { 1 } } ( I ( e ) ) a _ { \pi _ { 2 } ( \tau ( e ) ) } \mu _ { k ^ { * } } ( \pi _ { 1 } ( i ( e ) ) ) , \sigma ( \pi _ { 2 } ( \tau ( e ) ) ) } } \\ { { \displaystyle \qquad \cdot \prod _ { \mathrm { { c } } \in E \neq k ^ { * } \land \{ ( e ) \} } a _ { \pi _ { 2 } ( \tau ( e ) ^ { * } ) } \mu _ { k ^ { * } } ( \pi _ { 1 } ( i ( e ) ) ) , \sigma ( \pi _ { 2 } ( u ^ { * } ) ) \Bigg ] } } \\ { { \displaystyle = \left( \frac { \Delta } { \sqrt { d } } \right) ^ { | E | } \mathbb { E } _ { ( a _ { \pi _ { 2 } ( \sigma ) } ) _ { \infty ^ { * } \neq \pi _ { 2 } } , z , k ^ { * } , \sigma } \Bigg [ \displaystyle \prod _ { \mathrm { { c } } \in E \backslash J \in E \backslash \{ ( e ) , E \tau ( e ) = u ^ { * } \} ) } z _ { \pi _ { 1 } ( I ( e ) ) } a _ { \pi _ { 2 } ( \tau ( e ) ) } \mu _ { k ^ { * } } ( \pi _ { 1 } ( i ( e ) ) ) , \sigma ( \pi _ { 2 } ( \tau ( e ) ) ) } } \\   \displaystyle \qquad \cdot \qquad \prod _ { \mathrm { { c } } \in E \backslash \{ ( e ) \} } a _ { \pi _ { 1 } ( \ell ( e ) ) } \mu _ { k ^ { * } } ( \pi _ { 1 } ( i ( e ) ) ) , \sigma ( \pi _ { 2 } ( u ^ { * } ) ) \Bigg ] \mathbb { E } _  a _ { \pi _ { 2 } ( \sigma ^ { * } ) } \setminus \{ \left( \pi _ { 2 } ^ { \prime } \right) \}  \end{array}
$$

because α is odd and the entries of a are independent Rademacher random variables. Similarly, $a _ { \pi _ { 2 } ( w ^ { * } ) }$ is independent of the functional x and we obtain $\mathbb { E } [ x P _ { G , \pi _ { 1 } , \pi _ { 2 } } ( Y ) ] = 0$

Proof of b). Write $v ^ { * } \in V$ for (one of the) nodes of odd degree, and write α for its odd degree. We have

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ P _ { G , \pi _ { 1 } , \pi _ { 2 } } ( Y ) ] = \left( \frac { \Delta } { \sqrt { d } } \right) ^ { | E | } \mathbb { E } _ { \alpha _ { \pi _ { 1 } , \pi _ { 1 } } , \sigma _ { \sigma } } \Bigg [ \displaystyle \prod _ { \epsilon \in E \backslash \{ \epsilon \in E : I ( \epsilon ) = \nu ^ { * } \} } \xi _ { \pi _ { 1 } } ( l ( \epsilon ) ) ^ { a _ { \pi _ { 2 } } } ( r ( \epsilon ) ) ^ { \theta _ { k } * } \big \langle \pi _ { 1 } ( l ( \epsilon ) ) \big \rangle \sigma ( \pi _ { 2 } ( r ( \epsilon ) ) ) } \\ { \displaystyle \qquad \cdot \displaystyle \prod _ { \epsilon \in E : I ( \epsilon ) = \nu ^ { * } } z _ { \pi _ { 1 } ( \nu ^ { * } ) } a _ { \pi _ { 2 } ( r ( \epsilon ) ) } \mu _ { k * } \big \langle \pi _ { 1 } ( \epsilon _ { \pi } ^ { * } ) \big \rangle \sigma ( \pi _ { 2 } ( r ( \epsilon ) ) \big \rangle \Bigg ] } \\ { = \Bigg ( \frac { \Delta } { \sqrt { d } } \Bigg ) ^ { | E | } \mathbb { E } _ { \alpha _ { \pi _ { 1 } ( \pi _ { 1 } ( \epsilon ) ) , \pi _ { 2 } ( \epsilon ) } , \kappa ^ { * } } \cdot \sigma \Bigg [ \displaystyle \prod _ { \epsilon \in E \backslash \{ \epsilon \in E : I ( \epsilon ) = \nu ^ { * } \} } z _ { \pi _ { 1 } ( l ( \epsilon ) ) } a _ { \pi _ { 2 } ( r ( \epsilon ) ) } \| \mu _ { k * } \langle \pi _ { 1 } ( l ( \epsilon ) ) \rangle \sigma ( \pi _ { 2 } ( r ( \epsilon ) ) ) } \\  \displaystyle \qquad \cdot \displaystyle \prod _ { \epsilon \in E : I ( \epsilon ) = \nu ^ { * } } a _ { \pi _ { 2 } ( r ( \epsilon ) ) } \mu _ { k * } \big \langle \pi _ { 1 } ( \epsilon ) \big \rangle \sigma ( \pi _ { 2 } ( r ( \epsilon ) ) \big ) \Bigg ] \mathbb { E } _  \alpha _ { \pi _ { 1 } ( \epsilon * ) } \end{array}
$$

because α is odd and the entries of z follow independent Rademacher distributions. As the entries of $z$ are also independent of x, we also have $\mathbb { E } [ x P _ { G , \pi _ { 1 } , \pi _ { 2 } } ( Y ) ] = 0$ □

Proof of Lemma 7. Consider the labeled graph $G _ { \oplus }$ formed by the symmetric diference of $( \pi _ { 1 } ^ { ( 1 ) } , \pi _ { 2 } ^ { ( 1 ) } ) [ G ^ { ( 1 ) } ]$ and $( \pi _ { 1 } ^ { ( 2 ) } , \pi _ { 2 } ^ { ( 2 ) } ) [ G ^ { ( 2 ) } ]$ . We have

$$
\mathbb { E } \left[ P _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } P _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right] = \left( \frac { \Delta } { \sqrt { d } } \right) ^ { | E _ { \oplus } | } \mathbb { E } _ { a , z , k ^ { * } , \sigma } \left[ \prod _ { e \in E _ { \oplus } } z _ { l ( e ) } a _ { r ( e ) } \mu _ { k ^ { * } } ( l ( e ) ) , \sigma ( r ( e ) ) \right] .\tag{57}
$$

Suppose that there exists $v _ { 0 } \in V ^ { ( 2 ) }$ whose degree is odd. Then, in $E _ { \oplus }$ , the degree of $\pi _ { 1 } ^ { ( 2 ) } ( v _ { 0 } )$ is also odd. Hence, $z _ { \pi _ { 1 } ^ { ( 2 ) } ( v _ { 0 } ) }$ arises an odd number of times in (57). Since the $z _ { i } \mathrm { \dot { s } }$ are independent and follow independent Rademacher distributions, we deduce that $\mathbb { E } \left[ P _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } P _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right] = 0$

Now assume that there exists $w _ { 0 } \in W ^ { ( 2 ) }$ whose degree is odd. Then, in $\scriptstyle { E _ { \oplus } }$ , the degree of $\pi _ { 2 } ^ { ( 2 ) } ( w _ { 0 } )$ is also odd. Hence, ${ } ^ { a } \pi _ { 2 } ^ { ( 2 ) } ( w _ { 0 } )$ arises an odd number of times in (57). Since the $\boldsymbol { a } _ { j } ^ { \prime } \mathrm { { s } }$ are independent and follow independent Rademacher distributions, we deduce that $\mathbb { E } \left[ P _ { G ^ { ( 1 ) } , \pi ^ { ( 1 ) } } P _ { G ^ { ( 2 ) } , \pi ^ { ( 2 ) } } \right] = 0$

Proof of Lemma 8. By Lemma $5 ,$ we have Cor $\begin{array} { r } { \dot { \mathbf { \eta } } _ { \leq D } = \operatorname* { s u p } \left\{ \mathbb { E } [ x P ] : \ P \in \mathcal { S } , \ \mathbb { E } [ P ^ { 2 } ] \leq 1 \right\} } \end{array}$ , where we write $S : = \operatorname { s p a n } \left( 1 , ( P _ { G } ) _ { G \in { \mathcal { G } } < D } \right)$ , so that we may restrict our attention to polynomials lying in $s$ . Consider any polynomial $P \in { \mathcal { S } }$ such that $\mathbb { E } [ P ^ { 2 } ] \le 1$

Decompose $P = P _ { 1 } + P _ { 2 }$ where $P _ { 1 }$ lies in the span of 1 and the $P _ { G }$ with $G \in \mathcal { G } _ { < D } ^ { \mathrm { e v e n } }$ and $P _ { 2 }$ lies in the span of $P _ { G }$ with $G \in \mathcal { G } _ { \leq D } \ \backslash \ \mathcal { G } _ { < D } ^ { \mathrm { e v e n } }$ . By Lemmas 6 and 7, we have $\mathbb { E } [ x P _ { 2 } ] = 0$ and $\mathbb { E } [ P _ { 1 } P _ { 2 } ] = 0$ , so that $\mathbb { E } [ P _ { 1 } ^ { 2 } ] \le \mathbb { E } [ P _ { 1 } ^ { 2 } ] + \mathbb { E } [ P _ { 2 } ^ { 2 } ] = \mathbb { E } [ P ^ { 2 } ] \le 1$ and $\mathbb { E } [ x P ] = \mathbb { E } [ x P _ { 1 } ]$ , that is, $P _ { 1 }$ is admissible and achieves the same value as $P .$ Since, conversely, any polynomial in the span of 1 and of the $P _ { G }$ with $G \in \mathcal { G } _ { \leq D } ^ { \mathrm { e v e n } }$ belongs to $s { \mathrm { . } }$ we have

$$
\mathrm { C o r r } _ { \le D } = \operatorname * { s u p } _ { { P \in \mathrm { s p a n } \Big ( 1 , ( P _ { G } ) _ { G \in { \mathscr G } _ { < D } ^ { \mathrm { e v e n } } } \Big ) } \colon \mathbb E [ P ^ { 2 } ] \le 1 } \mathbb E [ x P ] .
$$

To conclude, it remains to check that span $\Big ( 1 , ( P _ { G } ) _ { G \in \mathcal { G } _ { \leq D } ^ { \mathrm { e v e n } } } \Big ) = \operatorname { s p a n } \Big ( 1 , ( \Psi _ { G } ) _ { G \in \mathcal { G } _ { \leq D } ^ { \mathrm { e v e n } } } \Big )$ . Fix any $G \in { \mathcal { G } } _ { \leq D } ^ { \mathrm { e v e n } }$ We easily check by expanding $\Psi _ { G }$ that it lies in the span of 1 and of the $P _ { G ^ { \prime } } \mathrm { \bar { s } }$ . Conversely, $P _ { G }$ is a linear combination of 1 and all $\Psi _ { G ^ { \prime } }$ where the $G ^ { \prime }$ are all subgraphs of $G$ (up to equivalence between graphs) where some of the connected components of G have been removed.

## F Proofs of the properties of the BCH construction

In this section, we prove Lemmas 3 and 4, stated in Section 3, as well as Lemmas 1 and 2 on the moments and the separation of the centers.

Proof of Lemma 3. We have

$$
\begin{array} { r l } {  { \Big ( \frac { 1 } { K } \sum _ { k \leq K } \mu _ { k } ^ { \otimes l } \Big ) _ { j _ { 1 } , \dots , j _ { l } } = \frac { 1 } { K } \sum _ { k \leq K } \prod _ { i \leq l } \mu _ { k , j _ { i } } } } \\ & { = \frac { 1 } { K } \sum _ { k \leq K } \prod _ { i \leq l } H _ { j _ { i } } ( \Big ( 2 ( \bar { \psi } ( k - 1 ) ) _ { a , b } - 1 \Big ) _ { a \leq m , b \leq u } ) } \\ & { = \mathbb { E } _ { R \sim \mathcal { R } ^ { m \times u } } \Big [ \prod _ { i \leq l } H _ { j _ { i } } ( R ) \Big ] , } \end{array}
$$

where R is the Rademacher distribution. This holds since $2 \bar { \psi } ( k { - } 1 ) { - } 1$ is uniformly distributed on $\{ - 1 , 1 \}$ m×u when k is uniformly distributed on $\{ 1 , \ldots , K \}$ – recall $K = 2 ^ { u m }$ . Hence, we have

$$
\begin{array} { r l } & { \mathbb { E } _ { R \sim { \mathcal R } ^ { m \times u } } \Big [ \displaystyle \prod _ { i \leq l } H _ { j _ { i } } ( R ) \Big ] = \mathbb { E } _ { R \sim { \mathcal R } ^ { m \times u } } \Big [ \displaystyle \prod _ { i = 1 } ^ { l } \displaystyle \prod _ { a = 1 } ^ { m } \displaystyle \prod _ { b = 1 } ^ { u } ( R _ { b } ^ { ( a ) } ) ^ { [ \phi ^ { ( 2 a - 1 ) } ( j _ { i } ) ] _ { b } } \Big ] } \\ & { \qquad = \mathbb { E } _ { R \sim { \mathcal R } ^ { m \times u } } \Big [ \displaystyle \prod _ { a = 1 } ^ { m } \displaystyle \prod _ { b = 1 } ^ { u } ( R _ { b } ^ { ( a ) } ) ^ { \sum _ { i = 1 } ^ { l } [ \phi ^ { ( 2 a - 1 ) } ( j _ { i } ) ] _ { b } } \Big ] } \\ & { \qquad = \mathbf { 1 } \Big \{ \forall a \leq m , \forall b \leq u : \displaystyle \sum _ { i = 1 } ^ { l } [ \phi ^ { ( 2 a - 1 ) } ( j _ { i } ) ] _ { b } \mathrm { ~ e v e n } \Big \} . } \end{array}
$$

This concludes the proof by remarking that the sum in $\mathbb { F } _ { 2 } ^ { u }$ has a given coordinate equal to 0 if and only if the standard sum is even. □

The following lemma is the key property of the BCH construction, see [39]. Here, we provide a selfcontained proof without reference to coding theory.

Lemma 24. Consider a positive integer $l \leq 2 m$ and consider l elements, not necessarily distinct, $v _ { 1 } , \ldots , v _ { l } \in$ $\mathbb { F } _ { 2 ^ { u } } \setminus \{ 0 \} = \{ 1 , \dots , d \}$ . Assume that for any $a \leq m$

$$
\sum _ { i = 1 } ^ { l } \phi ^ { ( 2 a - 1 ) } ( v _ { i } ) = 0 ,
$$

where the sum here is taken in $\mathbb { F } _ { 2 } ^ { u }$ - and therefore corresponds to a coordinate-wise sum of the u-dimensional vectors, modulo 2. Then l is even and the vector $( v _ { 1 } , \ldots , v _ { l } )$ is paired, that is its entries can be grouped two by two into $l / 2$ pairs of equal elements.

Proof of Lemma 24. For any integer $k \geq 1$ , write

$$
p _ { k } : = \sum _ { i = 1 } ^ { l } v _ { i } ^ { k } \ \in \mathbb { F } _ { 2 ^ { u } }
$$

for the k-th power sum of $v _ { 1 } , \ldots , v _ { l }$ , all the operations being those of the field $\mathbb { F } _ { 2 ^ { u } }$ , which has characteristic 2. Since $\phi ^ { ( k ) } \bar { ( } v ) = \psi ( v ^ { k } )$ by (7) and since $\psi$ is an isomorphism of additive groups from $\mathbb { F } _ { 2 ^ { u } }$ onto $\mathbb { F } _ { 2 } ^ { u }$ , we have, for every $a \leq m$

$$
\sum _ { i = 1 } ^ { l } \phi ^ { ( 2 a - 1 ) } ( v _ { i } ) = \sum _ { i = 1 } ^ { l } \psi \big ( v _ { i } ^ { 2 a - 1 } \big ) = \psi \Big ( \sum _ { i = 1 } ^ { l } v _ { i } ^ { 2 a - 1 } \Big ) = \psi \big ( p _ { 2 a - 1 } \big ) ,
$$

where the first two sums are taken in $\mathbb { F } _ { 2 } ^ { u }$ and the third one in $\mathbb { F } _ { 2 ^ { u } }$ . As $\psi ( x ) = 0$ if and only if $x = 0$ , the assumption of the lemma is therefore equivalent to

$$
p _ { 2 a - 1 } = 0 , \qquad \mathrm { f o r ~ e v e r y ~ } a = 1 , \dots , m .\tag{58}
$$

Step 1: all the power sums $p _ { 1 } , \ldots , p _ { 2 m }$ vanish. As $\mathbb { F } _ { 2 ^ { u } }$ has characteristic 2, we have $( x + y ) ^ { 2 } = x ^ { 2 } + 2 x y +$ $y ^ { 2 } = x ^ { \overset { \cdot } { 2 } } + y ^ { 2 }$ for all $x , y ,$ so that $x \mapsto x ^ { 2 }$ is additive (this is the Frobenius endomorphism). Consequently, for every integer $k \geq 1$

$$
p _ { 2 k } = \sum _ { i = 1 } ^ { l } v _ { i } ^ { 2 k } = \sum _ { i = 1 } ^ { l } \left( v _ { i } ^ { k } \right) ^ { 2 } = \Big ( \sum _ { i = 1 } ^ { l } v _ { i } ^ { k } \Big ) ^ { 2 } = p _ { k } ^ { 2 } .\tag{59}
$$

We prove by strong induction on k that $p _ { k } = 0$ for every $k = 1 , \ldots , 2 m$ . Let $1 \leq k \leq$ 2m and assume that $p _ { k ^ { \prime } } = 0$ for every $1 \leq k ^ { \prime } < k$ . If k is odd, then $k \leq 2 m - 1$ , so that $k = 2 a - 1$ with $a = ( k + 1 ) / 2 \leq m$ , and $p _ { k } = 0$ by (58). If k is even, write $k = 2 k ^ { \prime }$ with $k ^ { \prime } = k / 2 ;$ ; then $1 \leq k ^ { \prime } < k$ , so that $p _ { k ^ { \prime } } = 0$ by the induction hypothesis, and $p _ { k } = p _ { k ^ { \prime } } ^ { 2 } = 0$ by (59). This proves the claim, namely

$$
p _ { k } = 0 , \mathrm { f o r ~ e v e r y ~ } k = 1 , \ldots , 2 m .\tag{60}
$$

Step 2: reduction to pairwise distinct values. For $v \in \mathbb { F } _ { 2 ^ { u } } \ \backslash$ {0}, let $n _ { v } : = | \{ i \leq l : \ v _ { i } = v \} |$ denote the multiplicity of v in the list $( v _ { 1 } , \ldots , v _ { l } )$ , so that $\textstyle \sum _ { v } n _ { v } = l$ . Since $\mathbb { F } _ { 2 ^ { u } }$ has characteristic 2, we have $n _ { v } v ^ { k } = \left( n _ { v } { \mathrm { ~ m o d ~ } } 2 \right) v ^ { k }$ for every k, so that, letting $w _ { 1 } , \ldots , w _ { s }$ be the (pairwise distinct, nonzero) elements v whose multiplicity $n _ { v }$ is odd,

$$
p _ { k } = \sum _ { v } n _ { v } v ^ { k } = \sum _ { j = 1 } ^ { s } w _ { j } ^ { k } , \mathrm { ~ f o r ~ e v e r y ~ } k \ge 1 .\tag{61}
$$

Note that s is at most the number of distinct values among $v _ { 1 } , \ldots , v _ { l }$ , so that $s \leq l \leq 2 m$

We easily check that $( v _ { 1 } , \ldots , v _ { l } )$ is paired if and only if $s = 0$ . It therefore only remains to prove that $s = 0$

$S t e p ~ \ 3 \colon \ s = 0$ . Assume, for the sake of contradiction, that $s \geq 1$ . Note that $s = 1$ is not possible since $0 = p _ { 1 } = w _ { 1 }$ contradicts the fact that the $w _ { j } \mathrm { ' s }$ are nonzero. Thus, we assume that $s \geq 2$ and we introduce the two polynomials

$$
Q ( T ) : = \prod _ { j = 1 } ^ { s } ( 1 - w _ { j } T ) \ , \qquad N ( T ) : = \sum _ { j = 1 } ^ { s } w _ { j } T \prod _ { i \ne j } ( 1 - w _ { i } T ) \ ,
$$

with coeficients in $\mathbb { F } _ { 2 ^ { u } }$ , and where the second product ranges over $i \in \{ 1 , \ldots , s \} \setminus \{ j \}$ . We work in the ring $\mathbb { F } _ { 2 ^ { u } } [ [ T ] ]$ of formal power series over $\mathbb { F } _ { 2 ^ { u } }$ . Since $Q ( 0 ) = 1$ , the series Q is invertible in $\mathbb { F } _ { 2 ^ { u } } [ [ T ] ]$ , and, since $\begin{array} { r } { \sum _ { k \geq 1 } ( \bar { w } _ { j } T ) ^ { k } = w _ { j } T / ( 1 - w _ { j } T ) } \end{array}$ in $\mathbb { F } _ { 2 ^ { u } } [ [ T ] ]$ for every j, we deduce from (61) that

$$
\sum _ { k \geq 1 } p _ { k } T ^ { k } = \sum _ { j = 1 } ^ { s } \sum _ { k \geq 1 } ( w _ { j } T ) ^ { k } = \sum _ { j = 1 } ^ { s } { \frac { w _ { j } T } { 1 - w _ { j } T } } = { \frac { N ( T ) } { Q ( T ) } } ,
$$

the last equality being obtained by putting the s fractions over the common denominator $Q ( T )$ . By (60), the coeficients of ${ \check { T } } ^ { 0 } , T ^ { 1 } , \ldots , T ^ { 2 m }$ in the power series $\begin{array} { r } { N ( T ) / Q ( T ) = \sum _ { k > 1 } p _ { k } T ^ { k } } \end{array}$ all vanish, that is $N ( T ) / Q ( T ) \in$ $T ^ { 2 m + 1 } \mathbb { F } _ { 2 ^ { u } } [ [ T ] ]$ . Multiplying by $Q ( T )$ , we get $N ( T ) \in T ^ { 2 m + 1 } \mathbb { F } _ { 2 ^ { u } } [ [ \bar { T } ] ]$ . But N is a polynomial of degree at most $1 + ( s - 1 ) = s \leq 2 m < 2 m + 1$ , so that this forces

$$
N = 0 .\tag{62}
$$

We now contradict (62) by evaluating N at $T = 1 / w _ { 1 }$ , which is legitimate since $w _ { 1 } \neq 0$ . For any $j \neq 1$ , the product $\begin{array} { r } { \prod _ { i \neq j } ( 1 - w _ { i } T ) } \end{array}$ contains the factor $( 1 - w _ { 1 } T )$ , which vanishes at $T = 1 / w _ { 1 }$ . Hence, only the term $j = 1$ survives in the sum defining $N$ , and

$$
N ( 1 / w _ { 1 } ) = w _ { 1 } \cdot \frac { 1 } { w _ { 1 } } \cdot \prod _ { i \ne 1 } \left( 1 - \frac { w _ { i } } { w _ { 1 } } \right) = \prod _ { i = 2 } ^ { s } \left( 1 - \frac { w _ { i } } { w _ { 1 } } \right) ~ .
$$

Since $w _ { 1 } , \ldots , w _ { s }$ are pairwise distinct, each factor $1 - w _ { i } / w _ { 1 }$ with $i \neq 1$ is a nonzero element of the field $\mathbb { F } _ { 2 ^ { u } }$ so that the above product of nonzero elements is nonzero. This contradicts (62). Therefore $s = 0$ and, by Step 2, l is even and $( v _ { 1 } , \ldots , v _ { l } )$ is paired. □

Proof of Lemma $\it 4 .$ First note that for any $j _ { 1 } , \dotsc , j _ { l } \in \{ 1 , \dotsc , d \}$

$$
\mathcal { U } _ { j _ { 1 } , \dots , j _ { l } } ^ { ( l ) } = \mathbb { E } _ { U \sim \mathcal { U } \Big ( \{ - 1 , 1 \} ^ { d } \Big ) } \left[ \prod _ { i \leq l } U _ { j _ { i } } \right] = \mathbf { 1 } \Big \{ \mathrm { t h e r e ~ e x i s t s ~ a ~ p a i r i n g ~ f o r ~ } ( j _ { 1 } , \dots , j _ { l } ) \Big \} .
$$

We now check the converse of Lemma 24: if $( v _ { i } ) _ { i \leq l }$ is paired, then $\begin{array} { r } { \sum _ { i = 1 } ^ { l } \phi ^ { ( 2 a - 1 ) } ( v _ { i } ) = 0 } \end{array}$ for every $a \leq m$ Indeed, pairing $v _ { 1 } , \ldots , v _ { l }$ into $l / 2$ pairs of equal elements, each pair $( v _ { J _ { i } } , v _ { J _ { i } ^ { \prime } } )$ with $v _ { J _ { i } } = v _ { J _ { i } ^ { \prime } }$ contributes $\phi ^ { ( 2 a - 1 ) } ( v _ { J _ { i } } ) + \phi ^ { ( 2 a - 1 ) } ( v _ { J _ { i } ^ { \prime } } ) = 2 \phi ^ { ( 2 a - 1 ) } ( v _ { J _ { i } } ) = 0$ to the sum, since the sum is taken in F<sup>u</sup> (where $2 x = 0$ for every x); summing over all $l / 2$ pairs gives the identity. Together with Lemma 24, this shows that, for $l \leq 2 m$ and $v _ { 1 } , \ldots , v _ { l } \in \mathbb { F } _ { 2 ^ { u } } \setminus \{ 0 \}$ , we have $\begin{array} { r } { \sum _ { i = 1 } ^ { l } \phi ^ { ( 2 a - 1 ) } ( v _ { i } ) = 0 } \end{array}$ for all $a \leq m$ if and only $\mathrm { i f } \ ( v _ { i } ) _ { i \leq l }$ is paired. In particular, if l is odd, no pairing of $( v _ { 1 } , \ldots , v _ { l } )$ is possible, so by the contrapositive of Lemma 24, the system $\begin{array} { r } { \sum _ { i = 1 } ^ { l } \phi ^ { ( 2 a - 1 ) } ( v _ { i } ) = 0 } \end{array}$ , ∀a $\leq m$ , has no solution among $v _ { 1 } , \ldots , v _ { l } \in \mathbb { F } _ { 2 ^ { u } } \setminus \{ 0 \}$ – so the equivalence above holds vacuously (both sides are always false) for odd l as well.

Combining this equivalence, applied with $v _ { i } = j _ { i }$ for $i \leq l - \mathrm { r e c a l l }$ that $j _ { 1 } , \dotsc , j _ { l } \in \{ 1 , \dotsc , d \}$ are identified with nonzero elements of $\mathbb { F } _ { 2 ^ { u } } ~ -$ , with Lemma 3, we obtain

$$
\Big ( \frac { 1 } { K } \sum _ { k \leq K } \mu _ { k } ^ { \otimes l } \Big ) _ { j _ { 1 } , \ldots , j _ { l } } = \mathbf { 1 } \{ \forall a \leq m : \sum _ { i = 1 } ^ { l } \phi ^ { ( 2 a - 1 ) } ( j _ { i } ) = 0 \} = \mathbf { 1 } \{ ( j _ { 1 } , \ldots , j _ { l } ) \mathrm { ~ i s ~ p a i r e d } \} = \mathcal { U } _ { j _ { 1 } , \ldots , j _ { l } } ^ { ( l ) } \ ,
$$

which concludes the proof.

We now prove Lemma 1, stated in Section 2, which transfers the moment property of Lemma 4 to the centers $\pm X _ { k }$ and to the observations $Y _ { i }$

Proof of Lemma 1. Throughout the proof, we work conditionally on σ and $( a _ { j } )$ , and we fix $i \in [ n ]$ . Fix a positive integer $l \leq 2 m + 1$ and $j _ { 1 } , \dotsc , j _ { l } \in [ d ]$ . For $j \in [ d ]$ , let $n _ { j } : = | \{ s \leq l : j _ { s } = j \}$ | be the multiplicity of $j$ in $( j _ { 1 } , \ldots , j _ { l } )$ , and let $S : = \{ j \in [ d ] : n _ { j }$ is odd}. As shown in Step 2 of the proof of Lemma $2 4 , ( j _ { 1 } , \ldots , j _ { l } )$ is paired if and only if $S = \emptyset$ , so that $\mathbb { E } [ { \bar { U } } ^ { \otimes l } ] _ { j _ { 1 } , \dots , j _ { l } } = \mathbf { 1 } \{ S = \emptyset \}$ . Note also that $| S | \le l \le 2 m + 1$

Proof of $( i )$ . Since $k ^ { * } ( i )$ and $z _ { i }$ are independent, independent of $\sigma$ and $( a _ { j } )$ , and respectively uniform on $[ K ]$ and on $\{ - 1 , 1 \}$ , the first equality in (5) is the definition of the conditional expectation. By (2), the entry at $( j _ { 1 } , \dots , j _ { l } )$ of the left-hand side of (5) is

$$
\mathbb { E } \big [ z _ { i } ^ { l } \big ] \Big ( \frac { \Delta } { \sqrt { d } } \Big ) ^ { l } \prod _ { s \leq l } a _ { j s } \ \frac { 1 } { K } \sum _ { k \leq K } \prod _ { s \leq l } \mu _ { k , \sigma ( j _ { s } ) } \ .
$$

If l is odd, then $\mathbb { E } [ z _ { i } ^ { l } ] = 0$ , and $( j _ { 1 } , \dots , j _ { l } )$ cannot be paired, so that both sides of (5) vanish. If l is even, then $l \leq 2 m$ and $\mathbb { E } [ z _ { i } ^ { l } ] \stackrel { - } { = } 1$ . Applying Lemma 4 at the entry $( \sigma ( j _ { 1 } ) , \dots , \sigma ( j _ { l } ) )$ , we get that $\begin{array} { r } { \frac { 1 } { K } \sum _ { k \le K } \prod _ { s \le l } \mu _ { k , \sigma ( j _ { s } ) } } \end{array}$ is equal to 1 if $( \sigma ( j _ { 1 } ) , \dots , \sigma ( j _ { l } ) )$ is paired and to 0 otherwise. Since σ is a bijection, $( \sigma ( j _ { 1 } ) , \dots , \sigma ( j _ { l } ) )$ is paired if and only if $( j _ { 1 } , \dots , j _ { l } )$ is paired, that is if and only if $S = \emptyset$ . Finally, if $S = \emptyset .$ , then $\Pi _ { s \leq l } a _ { j _ { s } } =$ $\textstyle \prod _ { j \in [ d ] } a _ { j } ^ { n _ { j } } = 1$ since all the $n _ { j }$ are even and $a _ { j } ^ { 2 } = 1$ . Hence, the above entry is equal to $( \Delta / \sqrt { d } ) ^ { l } { \bf 1 } \{ S = \emptyset \} =$ $( \Delta / \sqrt { d } ) ^ { l } \mathbb { E } [ U ^ { \otimes l } ] _ { j _ { 1 } , \dots , j _ { l } }$ , which proves (5).

Proofof(ii). Since $Y _ { i j } \in \{ - 1 , 1 \}$ , we have $Y _ { i j } ^ { 2 } = 1$ , so that $\begin{array} { r } { \prod _ { s \leq l } Y _ { i j _ { s } } = \prod _ { j \in S } Y _ { i j } } \end{array}$ . Conditionally on $\sigma , ( a _ { j } )$ $k ^ { * } ( i )$ and $z _ { i } .$ , the coordinates of $Y _ { i }$ are independent and $\begin{array} { r } { \mathbb { E } [ Y _ { i j } \mid \sigma , ( a _ { j } ) , k ^ { * } ( i ) , z _ { i } ] = 2 \cdot \frac { 1 + z _ { i } X _ { k ^ { * } ( i ) , j } } { 2 } - 1 = z _ { i } X _ { k ^ { * } ( i ) , j } . } \end{array}$ As the elements of $S$ are pairwise distinct, we deduce that

$$
\mathbb { E } \Big [ \prod _ { s \leq l } Y _ { i j _ { s } } \Big | \sigma , ( a _ { j } ) , k ^ { * } ( i ) , z _ { i } \Big ] = \prod _ { j \in S } z _ { i } X _ { k ^ { * } ( i ) , j } \mathrm { ~ . ~ }
$$

If $S \ = \ \varnothing .$ , this product is empty and equal to 1. If $S \neq \emptyset$ , write $S = \{ j _ { 1 } ^ { \prime } , \ldots , j _ { | S | } ^ { \prime } \}$ . Integrating with respect to $k ^ { * } ( i )$ and $z _ { i } ,$ and applying (5) at the order $| S | \le 2 m + 1$ and at the entry $( j _ { 1 } ^ { \prime } , \ldots , j _ { | S | } ^ { \prime } )$ , we obtain $\begin{array} { r } { \mathbb { E } [ \prod _ { s \leq l } Y _ { i j _ { s } } \mid \sigma , ( a _ { j } ) ] = ( \Delta / \sqrt { d } ) ^ { \mid S \mid } \mathbb { E } [ U ^ { \otimes \mid S \mid } ] _ { j _ { 1 } ^ { \prime } , . . . , j _ { 1 S \mid } ^ { \prime } } = 0 , } \end{array}$ , because the $j _ { r } ^ { \prime }$ are pairwise distinct, so that $( j _ { 1 } ^ { \prime } , \ldots , j _ { | S | } ^ { \prime } )$ is not paired. In both cases, $\begin{array} { r } { \mathbb { E } [ \prod _ { s < l } Y _ { i j _ { s } } \mid \sigma , ( a _ { j } ) ] = \mathbf { 1 } \{ S = \emptyset \} = \mathbb { E } [ U ^ { \otimes l } ] _ { j _ { 1 } , \dots , j _ { l } } } \end{array}$ , which proves (6). Finally, the unconditional versions of (i) and (ii) follow by integrating with respect to σ and $( a _ { j } )$ , since the conditional quantities do not depend on them. □

Proof of Lemma 2. For $x , y \in \mathbb { F } _ { 2 } ^ { u }$ , write $\langle x , y \rangle _ { \mathbb { F } _ { 2 } } : = \sum _ { b < u } x _ { b } y _ { b } \in \mathbb { F } _ { 2 }$ . As in Section 3, we identify [d] with $\mathbb { F } _ { 2 ^ { u } } \setminus \{ 0 \}$ , and all the products of elements of [d] below are taken in the field $\mathbb { F } _ { 2 ^ { u } }$

Step 1: reduction to the sums $S _ { k }$ . For $k \in \left[ K \right]$ , let $\vartheta ( k ) \in \mathbb { F } _ { 2 } ^ { m \times u }$ be defined by $\vartheta ( k ) _ { a , b } : = 1 - ( \bar { \psi } ( k - 1 ) ) _ { a , b } .$ so that $2 ( \bar { \psi } ( k - 1 ) ) _ { a , b } - 1 = ( - 1 ) ^ { \vartheta ( k ) _ { a , b } }$ , and write $\vartheta ^ { ( a ) } ( k ) \in \mathbb { F } _ { 2 } ^ { u }$ for its a-th row. Since ψ<sup>¯</sup> is a bijection, so is $\vartheta : [ K ]  \mathbb { F } _ { 2 } ^ { m \times u }$ . By (8) and (9), we have, for all $k \in [ K ]$ and $v \in [ d ]$

$$
\mu _ { k , v } = \prod _ { a \leq m } \prod _ { b \leq u } ( - 1 ) ^ { \vartheta ( k ) _ { a , b } [ \phi ^ { ( 2 a - 1 ) } ( v ) ] _ { b } } = ( - 1 ) ^ { \sum _ { a \leq m } \langle \vartheta ^ { ( a ) } ( k ) , \phi ^ { ( 2 a - 1 ) } ( v ) \rangle _ { \mathbb { F } _ { 2 } } } .\tag{63}
$$

In particular, for k, $k ^ { \prime } \in [ K ]$ , the entrywise product of $\mu _ { k }$ and $\mu _ { k ^ { \prime } }$ is equal to $\mu _ { k ^ { \prime \prime } }$ , where $k ^ { \prime \prime }$ is defined by $\vartheta ( k ^ { \prime \prime } ) = \vartheta ( k ) + \vartheta ( k ^ { \prime } )$ . Let $k _ { 0 }$ be the index such that $\vartheta ( k _ { 0 } ) = 0$ , so that all the entries of $\mu _ { k _ { 0 } }$ are equal to 1. If $k \neq k ^ { \prime } .$ , then $\vartheta ( k ) \neq \vartheta ( k ^ { \prime } )$ , so that $\vartheta ( k ^ { \prime \prime } ) \neq 0$ , that is $\boldsymbol { k } ^ { \prime \prime } \ne k _ { 0 }$ . Writing $\begin{array} { r } { S _ { k } : = \sum _ { v \leq d } \mu _ { k , v } } \end{array}$ , we get $\langle \mu _ { k } , \mu _ { k ^ { \prime } } \rangle = S _ { k ^ { \prime \prime } }$ . Hence, it sufices to prove tha

$$
s _ { * } : = \operatorname* { m a x } _ { k \neq k _ { 0 } } | S _ { k } | \leq \eta _ { m } d .\tag{64}
$$

The bounds on $\| \mu _ { k } - \mu _ { k ^ { \prime } } \| _ { 2 } ^ { 2 }$ and $\| \mu _ { k } + \mu _ { k ^ { \prime } } \| _ { 2 } ^ { 2 }$ then follow from $\| \mu _ { k } \pm \mu _ { k ^ { \prime } } \| _ { 2 } ^ { 2 } = 2 d \pm 2 \langle \mu _ { k } , \mu _ { k ^ { \prime } } \rangle$ , since $\| \mu _ { k } \| _ { 2 } ^ { 2 } = d .$ Step 2: a moment bound. By Lemma 4 with $l = 2 m$

$$
\sum _ { k \leq K } S _ { k } ^ { 2 m } = \sum _ { v _ { 1 } , \ldots , v _ { 2 m } \in [ d ] } \sum _ { k \leq K } \prod _ { \substack { s \leq 2 m } } \mu _ { k , v _ { s } } = K \left| \left\{ ( v _ { 1 } , \ldots , v _ { 2 m } ) \in [ d ] ^ { 2 m } : ( v _ { 1 } , \ldots , v _ { 2 m } ) { \mathrm { ~ i s ~ p a i r e d } } \right\} \right| .
$$

Any paired $( v _ { 1 } , \ldots , v _ { 2 m } )$ is obtained by choosing a partition of $\{ 1 , \ldots , 2 m \}$ into m pairs – there are $( 2 m -$ $1 ) ! ! = ( 2 m - 1 ) ( 2 m - 3 ) \cdot \cdot \cdot 1$ of them – and a common value in [d] for each pair. Hence there are at most $( 2 m - 1 ) ! ! d ^ { m }$ paired vectors and, since $K = ( d + 1 ) ^ { m }$ ，

$$
\sum _ { k \neq k _ { 0 } } S _ { k } ^ { 2 m } \leq \sum _ { k \leq K } S _ { k } ^ { 2 m } \leq ( 2 m - 1 ) ! ! d ^ { m } ( d + 1 ) ^ { m } .\tag{65}
$$

Step 3: invariance under dilations. The sums $S _ { k } , k \ne k _ { 0 }$ , are in general not all equal, but we show in this step that $S _ { k }$ is invariant under the dilations $v \mapsto \lambda$ v of the coordinates. More precisely, we define an action $k \mapsto \lambda \cdot k$ of the nonzero elements $\lambda$ of $\mathbb { F } _ { 2 ^ { u } }$ on $[ K ]$ , and we prove that $S _ { \lambda \cdot k } = S _ { k }$ for all $k \in [ K ]$ and all $\lambda \neq 0$ . Together with Step 4, which shows that the orbits $\{ \lambda \cdot k : ~ \lambda \neq 0 \} , k \neq k _ { 0 }$ , contain at least $d / ( 2 m - 1 )$ indices, this will allow us to turn the moment bound (65) into the uniform bound (64). Fix $\lambda \in \dot { \mathbb { F } } _ { 2 ^ { u } } \setminus \{ 0 \}$ and $\vartheta \in \mathbb { F } _ { 2 } ^ { m \times u }$ . For each $a \leq m$ , the map $y \mapsto \langle \vartheta ^ { ( a ) } , \psi ( \lambda ^ { 2 a - 1 } y ) \rangle _ { \mathbb { F } _ { 2 } }$ is an $\mathbb { F } _ { 2 } .$ -linear form on $\mathbb { F } _ { 2 ^ { u } }$ , since ψ is additive. As ψ is moreover a bijection onto $\mathbb { F } _ { 2 } ^ { u }$ , any $\mathbb { F } _ { 2 } \cdot$ -linear form on $\mathbb { F } _ { 2 ^ { u } }$ is of the form $y \mapsto \langle r , \psi ( y ) \rangle _ { \mathbb { F } _ { 2 } }$ for a unique $r \in \mathbb { F } _ { 2 } ^ { u }$ . Hence, there exists a unique $\boldsymbol { \lambda } \cdot \boldsymbol { \vartheta } \in \mathbb { F } _ { 2 } ^ { m \times u }$ such that

$$
\begin{array} { r } { \langle ( \lambda \cdot \vartheta ) ^ { ( a ) } , \psi ( y ) \rangle _ { \mathbb { F } _ { 2 } } = \langle \vartheta ^ { ( a ) } , \psi ( \lambda ^ { 2 a - 1 } y ) \rangle _ { \mathbb { F } _ { 2 } } \ , \qquad \mathrm { f o r ~ a l l } \ a \le m \mathrm { ~ a n d ~ } y \in \mathbb { F } _ { 2 ^ { u } } \ . } \end{array}\tag{66}
$$

By uniqueness, $1 \cdot \vartheta = \vartheta , \lambda \cdot 0 = 0$ , and $\lambda \cdot ( \lambda ^ { \prime } \cdot \vartheta ) = ( \lambda \lambda ^ { \prime } )$ · ϑ for all nonzero $\lambda , \lambda ^ { \prime }$ , since both sides of the latter satisfy (66) with $\lambda \lambda ^ { \prime }$ in place of λ. In particular, $\vartheta \mapsto \lambda \cdot \vartheta$ is a bijection of $\mathbb { F } _ { 2 } ^ { m \times u }$ , with inverse $\vartheta \mapsto \lambda ^ { - 1 } \cdot \vartheta .$ and it maps nonzero elements to nonzero elements. For $k \in [ K ]$ , define $\lambda \cdot k \in [ K ]$ by $\vartheta ( \lambda \cdot k ) = \lambda \cdot \vartheta ( k )$ Since $\phi ^ { ( 2 a \bar { - } 1 ) } ( \lambda v ) = \psi ( \lambda ^ { 2 a - 1 } v ^ { 2 a - 1 } )$ , applying (66) with $y = v ^ { 2 a - 1 }$ in (63) yields $\mu _ { k , \lambda v } = \mu _ { \lambda \cdot k , v }$ for all $v \in [ d ]$ ${ \mathrm { A s ~ } } v \mapsto \lambda v$ is a bijection of ${ \mathbb { F } } _ { 2 ^ { u } } \setminus \{ 0 \} = [ d ]$ , we deduce that

$$
S _ { \lambda \cdot k } = \sum _ { v \leq d } \mu _ { k , \lambda v } = S _ { k } .
$$

Step $\it 4 .$ the orbits are large. Let $\vartheta \neq 0$ and let $a ^ { \star } \leq m$ be such that $\vartheta ^ { ( a ^ { \star } ) } \neq 0$ . Assume that $\lambda \cdot \vartheta = \vartheta$ Then, by (66) with $a = a ^ { \star }$ and by additivity of ψ, we have $\langle \vartheta ^ { ( a ^ { \star } ) } , \psi ( ( \lambda ^ { 2 a ^ { \star } - 1 } - 1 ) y ) \rangle _ { \mathbb { F } _ { 2 } } = 0$ for all $y \in \mathbb { F } _ { 2 ^ { u } }$ If $\lambda ^ { 2 a ^ { \star } - \bar { 1 } } \doteq 1$ , then $y \mapsto ( \lambda ^ { 2 a ^ { \star } - 1 } - 1 ) y$ is a bijection of $\mathbb { F } _ { 2 ^ { u } }$ and, ψ being onto $\mathbb { F } _ { 2 } ^ { u }$ , we get $\langle \vartheta ^ { ( a ^ { \star } ) } , x \rangle _ { \mathbb { F } _ { 2 } } = 0$ for all $\boldsymbol { x } \in \mathbb { F } _ { 2 } ^ { u }$ , that is $\vartheta ^ { ( a ^ { \star } ) } = 0$ , a contradiction. Hence λ is a root of the polynomial $x ^ { 2 \bar { a } ^ { \star } - 1 } - 1$ , which has at most $2 a ^ { \star } - 1 \leq 2 m - 1$ roots in the field $\mathbb { F } _ { 2 ^ { u } }$ . Now, if $\lambda \cdot \vartheta = \lambda ^ { \prime } \cdot \vartheta .$ , then $( \lambda ^ { \prime - 1 } \lambda ) \cdot \vartheta = \vartheta$ . Therefore, the map $\lambda \mapsto \lambda \cdot \vartheta$ , defined on the d nonzero elements of $\mathbb { F } _ { 2 ^ { u } }$ , takes each of its values at most $2 m - 1$ times, so that the orbit $\left\{ \lambda \cdot \vartheta : \lambda \neq 0 \right\}$ contains at least $d / ( 2 m - 1 )$ elements, all of them nonzero.

Step 5: conclusion. Let $k _ { 1 } \neq k _ { 0 }$ be such that $| S _ { k 1 } | = s _ { * }$ . By Steps 3 and 4 applied with $\vartheta = \vartheta ( k _ { 1 } ) \neq 0$ the set $\{ \lambda \cdot k _ { 1 } : \ \lambda \neq 0 \}$ contains at least $d / ( 2 m - 1 )$ indices, all distinct from $k _ { 0 }$ , and $\left| S _ { \lambda \cdot k _ { 1 } } \right| = s _ { * }$ for each of them. Together with (65), this gives

$$
\frac { d } { 2 m - 1 } s _ { * } ^ { 2 m } \le ( 2 m - 1 ) ! ! d ^ { m } ( d + 1 ) ^ { m } \ , \qquad \mathrm { t h a t ~ i s } \ \quad \left( \frac { s _ { * } } { d } \right) ^ { 2 m } \le \frac { ( 2 m - 1 ) ( 2 m - 1 ) ! ! ( 1 + 1 / d ) ^ { m } } { d } \ .
$$

By the arithmetic-geometric mean inequality, $\begin{array} { r } { ( 2 m - 1 ) ! ! = \prod _ { i = 1 } ^ { m } ( 2 i - 1 ) \leq \left( \frac { 1 } { m } \sum _ { i = 1 } ^ { m } ( 2 i - 1 ) \right) ^ { m } = m ^ { m } } \end{array}$ . Besides, $2 m - 1 \leq$ 2m and, since $d \geq 2 m , ( 1 + 1 / d ) ^ { m } \leq e ^ { m / d } \leq e ^ { 1 / 2 } \leq 2$ . Hence $( s _ { * } / d ) ^ { 2 m } \leq 4 m ^ { m + 1 } / d = \eta _ { m } ^ { 2 m }$ , which is (64).

Finally, assume that $d \geq ( 4 m ) ^ { m + 1 }$ . Then $d \geq 2 m$ and $\eta _ { m } ^ { 2 m } \leq 4 m ^ { m + 1 } / ( 4 m ) ^ { m + 1 } = 4 ^ { - m }$ , so that $\eta _ { m } \leq 1 / 2$ and $\| \mu _ { k } - \mu _ { k ^ { \prime } } \| _ { 2 } ^ { 2 } \geq 2 ( 1 - \eta _ { m } ) d \geq d$ for all $\boldsymbol { k } \neq \boldsymbol { k } ^ { \prime }$ . As for the centers, let $\varepsilon , \varepsilon ^ { \prime } \in \{ - 1 , 1 \}$ and $k , k ^ { \prime } \in [ K ]$ with $( \varepsilon , k ) \neq ( \varepsilon ^ { \prime } , k ^ { \prime } )$ . By (2), since $| a _ { j } | = 1$ and σ is a permutation, we have $\begin{array} { r } { \| \varepsilon X _ { k } - \varepsilon ^ { \prime } X _ { k ^ { \prime } } \| _ { 2 } = \frac { \Delta } { \sqrt { d } } \| \varepsilon \mu _ { k } - \varepsilon ^ { \prime } \mu _ { k ^ { \prime } } \| _ { 2 } . } \end{array}$ If $\begin{array} { r } { k \neq k ^ { \prime } , \mathrm { ~ t h e n ~ } \| \varepsilon \mu _ { k } - \varepsilon ^ { \prime } \mu _ { k ^ { \prime } } \| _ { 2 } ^ { 2 } = 2 d - 2 \varepsilon \varepsilon ^ { \prime } \langle \mu _ { k } , \mu _ { k ^ { \prime } } \rangle \geq 2 ( 1 - \eta _ { m } ) d \geq d . } \end{array}$ If $k = k ^ { \prime } .$ , then $\varepsilon ^ { \prime } = - \varepsilon$ and $\| \varepsilon \mu _ { k } - \varepsilon ^ { \prime } \mu _ { k ^ { \prime } } \| _ { 2 } ^ { 2 } = \| 2 \mu _ { k } \| _ { 2 } ^ { 2 } = 4 d .$ . In both cases, $\| \varepsilon X _ { k } - \varepsilon ^ { \prime } X _ { k ^ { \prime } } \| _ { 2 } \geq \Delta$ □

## References

[1] Daniel Aloise, Amit Deshpande, Pierre Hansen, and Preyas Popat. NP-hardness of Euclidean sum-ofsquares clustering. Machine Learning, 75(2):245–248, 2009.

[2] Pranjal Awasthi, Moses Charikar, Ravishankar Krishnaswamy, and Ali Kemal Sinop. The Hardness of Approximation of Euclidean k-Means. In 31st International Symposium on Computational Geometry (SoCG 2015), volume 34, pages 754–767. Schloss Dagstuhl – Leibniz-Zentrum f¨ur Informatik, 2015.

[3] Jinho Baik, G´erard Ben Arous, and Sandrine P´ech´e. Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices. The Annals of Probability, 33(5):1643 – 1697, 2005.

[4] Matthew Brennan and Guy Bresler. Reducibility and statistical-computational gaps from secret leakage. In Proceedings of Thirty Third Conference on Learning Theory, volume 125 of Proceedings of Machine Learning Research, pages 648–847. PMLR, 2020.

[5] Matthew Brennan, Guy Bresler, and Wasim Huleihel. Reducibility and computational lower bounds for problems with planted sparse structure. In Proceedings of the 31st Conference On Learning Theory, volume 75 of Proceedings of Machine Learning Research, pages 48–166. PMLR, 2018.

[6] Rares-Darius Buhai, Jun-Ting Hsieh, Aayush Jain, and Pravesh K. Kothari. The quasi-polynomial lowdegree conjecture is false. In 2025 IEEE 66th Annual Symposium on Foundations of Computer Science (FOCS), pages 2577–2590. IEEE, 2025.

[7] Alexandra Carpentier, Simone Maria Giancola, Christophe Giraud, and Nicolas Verzelen. Low-degree lower bounds via almost orthonormal bases. arXiv preprint arXiv:2509.09353, 2025.

[8] Alexandra Carpentier, Christophe Giraud, and Nicolas Verzelen. Phase transition for stochastic block model with more than n communities. In Proceedings of Thirty Ninth Conference on Learning Theory, volume 336 of Proceedings of Machine Learning Research, pages 938–1000. PMLR, 2026.

[9] Alexandra Carpentier and Nicolas Verzelen. Statistical and computational challenges in ranking. arXiv preprint arXiv:2512.21111, 2025.

[10] Alexandra Carpentier and Nicolas Verzelen. Low-degree lower bounds for clustering in moderate dimension. arXiv preprint arXiv:2602.23023, 2026.

[11] Sitan Chen and Ankur Moitra. Beyond the low-degree algorithm: mixtures of subcubes and their applications. In Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing, pages 869–880, 2019.

[12] Sanjoy Dasgupta. Learning mixtures of gaussians. In 40th Annual Symposium on Foundations of Computer Science (Cat. No. 99CB37039), pages 634–644. IEEE, 1999.

[13] Ilias Diakonikolas, Daniel M. Kane, Thanasis Pittas, and Nikos Zarifis. SQ lower bounds for learning mixtures of separated and bounded covariance Gaussians. In Gergely Neu and Lorenzo Rosasco, editors, Proceedings of Thirty Sixth Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pages 2319–2349. PMLR, 2023.

[14] Ilias Diakonikolas, Daniel M. Kane, and Alistair Stewart. Statistical query lower bounds for robust estimation of high-dimensional Gaussians and Gaussian mixtures. In 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS), pages 73–84. IEEE, 2017.

[15] Ilias Diakonikolas, Daniel M Kane, and Alistair Stewart. List-decodable robust mean estimation and learning mixtures of spherical gaussians. In Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing, pages 1047–1060, 2018.

[16] Travis Dillon. Fixed-strength spherical designs. SIAM Journal on Discrete Mathematics, 40(2):789–815, 2026.

[17] Bertrand Even, Christophe Giraud, and Nicolas Verzelen. Computation-information gap in highdimensional clustering. In Shipra Agrawal and Aaron Roth, editors, Proceedings of Thirty Seventh Conference on Learning Theory, volume 247 of Proceedings of Machine Learning Research, pages 1646– 1712. PMLR, 2024.

[18] Bertrand Even, Christophe Giraud, and Nicolas Verzelen. Computational lower bounds in latent models: clustering, sparse-clustering, biclustering. arXiv preprint arXiv:2506.13647, 2025.

[19] Yingjie Fei and Yudong Chen. Hidden integrality of sdp relaxations for sub-gaussian mixture models. In Proceedings of the 31st Conference On Learning Theory, volume 75 of Proceedings of Machine Learning Research, pages 1931–1965. PMLR, 2018.

[20] Jon Feldman, Ryan O’Donnell, and Rocco A. Servedio. Learning mixtures of product distributions over discrete domains. SIAM Journal on Computing, 37(5):1536–1564, 2008.

[21] Christophe Giraud and Nicolas Verzelen. Partial recovery bounds for clustering with the relaxed kmeans. Mathematical Statistics and Learning, 1(3/4):317–374, 2018.

[22] Justin Holmgren and Alexander S. Wein. Counterexamples to the low-degree conjecture. In 12th Innovations in Theoretical Computer Science Conference (ITCS 2021), volume 185 of Leibniz International Proceedings in Informatics (LIPIcs), pages 75:1–75:9. Schloss Dagstuhl – Leibniz-Zentrum f¨ur Informatik, 2021.

[23] Samuel Hopkins. Statistical inference and the sum of squares method. PhD thesis, Cornell University, 2018.

[24] Samuel B. Hopkins and Jerry Li. Mixture models, robustness, and sum of squares proofs. In Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing, pages 1021–1034. ACM, 2018.

[25] Samuel B. Hopkins and David Steurer. Eficient bayesian estimation from few samples: Community detection and related problems. In 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS), pages 379–390, 2017.

[26] Daniel Hsu and Sham M Kakade. Learning mixtures of spherical Gaussians: moment methods and spectral decompositions. In Proceedings of the 4th conference on Innovations in Theoretical Computer Science, pages 11–20, 2013.

[27] Pravesh K Kothari, Jacob Steinhardt, and David Steurer. Robust moment estimation and improved clustering via sum of squares. In Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing, pages 1035–1046, 2018.

[28] Dmitriy Kunisky, Alexander S. Wein, and Afonso S. Bandeira. Notes on computational hardness of hypothesis testing: Predictions using the low-degree likelihood ratio. In Paula Cerejeiras and Michael Reissig, editors, Mathematical Analysis, its Applications and Computation (ISAAC 2019), volume 385 of Springer Proceedings in Mathematics & Statistics, pages 1–50. Springer, Cham, 2022.

[29] Greg Kuperberg. Numerical cubature using error-correcting codes. SIAM Journal on Numerical Analysis, 44(3):897–907, 2006.

[30] Jeongyeol Kwon and Constantine Caramanis. The em algorithm gives sample-optimality for learning mixtures of well-separated gaussians. In Jacob Abernethy and Shivani Agarwal, editors, Proceedings of Thirty Third Conference on Learning Theory, volume 125 of Proceedings of Machine Learning Research, pages 2425–2487. PMLR, 2020.

[31] Gilles Lachaud and Robert Rolland. On the number of points of algebraic sets over finite fields. Journal of Pure and Applied Algebra, 219(11):5117–5136, 2015.

[32] Thibault Lesieur, Caterina De Bacco, Jess Banks, Florent Krzakala, Cris Moore, and Lenka Zdeborov´a. Phase transitions and optimal algorithms in high-dimensional Gaussian mixture clustering. In 2016 54th Annual Allerton Conference on Communication, Control, and Computing (Allerton), pages 601– 608. IEEE, 2016.

[33] Zhangsong Li. Algorithmic contiguity from low-degree heuristic II: Predicting detection-recovery gaps. arXiv preprint arXiv:2604.17410, 2026.

[34] Allen Liu and Jerry Li. Clustering mixtures with almost optimal separation in polynomial time. In Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of Computing, STOC 2022, pages 1248–1261, New York, NY, USA, 2022. Association for Computing Machinery.

[35] Matthias L¨ofler, Anderson Y Zhang, and Harrison H Zhou. Optimality of spectral clustering in the gaussian mixture model. The Annals of Statistics, 49(5):2506–2530, 2021.

[36] Yu Lu and Harrison H. Zhou. Statistical and Computational Guarantees of Lloyd’s Algorithm and its Variants. ArXiv e-prints, December 2016.

[37] Yuetian Luo and Chao Gao. Computational lower bounds for graphon estimation via low-degree polynomials. The Annals of Statistics, 52(5):2318–2348, 2024.

[38] Zongming Ma and Yihong Wu. Computational barriers in minimax submatrix detection. The Annals of Statistics, 43(3):1089–1116, 2015.

[39] F. Jessie MacWilliams and Neil J. A. Sloane. The Theory of Error-Correcting Codes, volume 16 of North-Holland Mathematical Library. North-Holland, Amsterdam, 1977.

[40] Songtao Mao. The polynomial-time low-degree conjecture is false. arXiv preprint arXiv:2607.20318, 2026.

[41] Andrea Montanari and Alexander S. Wein. Equivalence of approximate message passing and low-degree polynomials in rank-one matrix estimation. Probability Theory and Related Fields, 191(1–2):181–233, 2025.

[42] Mohamed Ndaoud. Sharp optimal recovery in the two component Gaussian mixture model. The Annals of Statistics, 50(4):2096 – 2126, 2022.

[43] Oded Regev and Aravindan Vijayaraghavan. On Learning Mixtures of Well-Separated Gaussians. In 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS), pages 85–96, 2017.

[44] Tselil Schramm and Alexander S Wein. Computational barriers to estimation from low-degree polynomials. The Annals of Statistics, 50(3):1833–1858, 2022.

[45] Igor R. Shafarevich. Basic Algebraic Geometry 1: Varieties in Projective Space. Springer, Berlin, Heidelberg, 3 edition, 2013.

[46] Youngtak Sohn and Alexander S. Wein. Sharp phase transitions in estimation with low-degree polynomials. In Proceedings of the 57th Annual ACM Symposium on Theory of Computing, pages 891–902. ACM, 2025.

[47] A. H. Stroud. Approximate Calculation of Multiple Integrals. Prentice-Hall Series in Automatic Computation. Prentice-Hall, Englewood Clifs, NJ, 1971.

[48] Santosh Vempala and Grant Wang. A spectral algorithm for learning mixture models. Journal of Computer and System Sciences, 68(4):841–860, 2004.

[49] Alexander S. Wein. Computational complexity of statistics: New insights from low-degree polynomials, 2025.