# Beyond Multi-Index Models: Learning Intersections of Degree-2 PTFs

Ilias Diakonikolas<sup>∗</sup> University of Wisconsin-Madison ilias@cs.wisc.edu

Mingchen Ma<sup>‡</sup> University of Wisconsin-Madison mingchen@cs.wisc.edu

Daniel M. Kane<sup>†</sup> University of California, San Diego dakane@ucsd.edu

Lisheng Ren<sup>§</sup> University of Wisconsin-Madison lren29@wisc.edu

Trung Tran University of California, San Diego trt023@ucsd.edu

## Abstract

We give an algorithm for PAC learning intersections of k degree-2 polynomial threshold functions over $\mathbb { R } ^ { d }$ under the standard Gaussian distribution. Our algorithm achieves classification error ϵ using $d ^ { O ( k ) } 2 ^ { O _ { k } ( \log ^ { 3 } ( 1 / \epsilon ) ) }$ time and samples. For every fixed $k ,$ this is polynomial in d and quasipolynomial in $1 / \epsilon ,$ , improving over the $\dot { d } ^ { O ( k ^ { 2 } / \epsilon ^ { 2 } ) }$ running time of the standard polynomial regression approach. To our knowledge, our result is the first to obtain a dimension exponent independent of the accuracy parameter, even for $k = 2$ . We complement our algorithmic result with a lower bound showing that the complexity dependence of $d ^ { \Omega ( k ) }$ is unavoidable for any Statistical Query learning algorithm.

Lifting the input to the space of linear and quadratic features turns the target into an intersection of k halfspaces. This suggests the classical approach of using the first two labeled moments to reduce the learning problem to at most k dimensions. In the lifted space, however, these moments can reveal nothing about the relevant subspace. Our starting point is a Gaussian comparison theorem showing that, when the defining quadratics are suficiently regular, the labeled moments needed for subspace recovery are approximately preserved. To handle arbitrary defining quadratics, we establish a new structural result showing that every direction with significant resampling influence can be detected through degree-O(k) Hermite moments. These results allow us to eficiently identify influential directions and reduce the problem to learning regular intersections through an adaptive restriction procedure, yielding an initial algorithm with running time $d ^ { O ( k ) } 2 ^ { \mathrm { p o l y } _ { k } ( 1 / \epsilon ) }$

To improve the exponential dependence on $1 / \epsilon$ , we develop a two-stage dimension-reduction approach that avoids the costly adaptive partitioning procedure. After identifying a subspace containing all directions with significant influence, we use a generalized notion of regularity and Gaussian comparison to recover the remaining relevant quadratic features through a new derivative-based moment operator. Together, the recovered features support an intersection that closely approximates the target. Combining these ingredients with a robust Hinge loss minimization method gives our quasipolynomial-time learning algorithm.

## Contents

1 Introduction 1   
1.1 Our Results and Technical Contributions . 2   
1.2 Technical Overview 5   
1.3 Related Work 12   
2 Preliminaries and Notations 13   
3 Gaussian Approximation via Regularity 15   
3.1 Gaussian Moment Comparison 16   
3.2 Gaussian Subspace Transfer 20   
3.3 Learning Regular Intersections 25   
4 Reduction to the Regular Case using Randomness 28   
5 Certificate Polynomial for Intersections of k Degree-2 PTFs 32   
5.1 Boundary Operator and Weighted Decision Boundary 32   
5.2 Certificate Polynomials for Generic Instance and Root Decompositions 35   
5.3 Analysis for Double Roots . 37   
5.4 Robust Factorization of Quadratic Polynomials under Gaussian Distribution 39   
5.5 Completing the Construction of Certificate Polynomials 52   
6 Eficient Correlation Finding through Influence Matrix 62   
A Tree-Based Algorithm for Learning Intersections of PTFs 64   
8 Polynomial Approximation with Hinge Loss 71   
9 Improved Complexity through Dimension Reduction and Hinge Loss Optimization 81   
9.1 Subspace Recovery through Gaussian Comparison 82   
9.2 Proof of Theorem 9.1 91   
10 Statistical Query Lower Bound 93

## 1 Introduction

We study the problem of PAC learning intersections of degree-2 polynomial threshold functions (PTFs) under the Gaussian distribution. A degree-2 PTF is a Boolean function obtained by thresholding a polynomial of degree at most two. An intersection of k such functions labels a point positive if it satisfies all k quadratic inequalities. When the defining polynomials are afine, this is the wellstudied problem of learning intersections of halfspaces [KP96, BK97, KS04, Vem10a, APS26, PV26]. Quadratic thresholds capture pairwise interactions between coordinates and include basic geometric concepts such as balls and ellipsoids. Intersections of quadratic PTFs also describe nonconvex regions, such as spherical shells.

Eficient algorithms are known for learning halfspaces and PTFs of fixed degree, even in the distribution-free setting [Ros58, MT94], and for learning generalized linear and single-index models under suitable assumptions [KKSK11]. For a single quadratic PTF, it sufices to solve a linear program after lifting the examples to quadratic features. For an intersection, a negative example tells us only that at least one of the defining inequalities failed. Even for two halfspaces, polynomial-time distribution-free PAC learning remains a tantalizing open problem. In fact, [APS26] recently gave the first subexponential-time algorithm for this case; see also [PV26].

This open problem has motivated the study of intersections of halfspaces under structured input distributions. Eficient algorithms are known for intersections of a constant number of halfspaces under the uniform distribution on a ball [BK97], Gaussian distributions [Vem10a, DKS18], and logconcave distributions when the bounding hyperplanes pass through the mean [Vem10b]. Finding a small set of relevant linear directions is also central to recent work on learning multi-index models (MIMs) under Gaussian inputs; see, e.g., [ABAM23, DIKZ25, DIKR25, DLB25].

A MIM is a function of the form $f ( \mathbf { x } ) = g ( \mathbf { W } \mathbf { x } )$ , where $\mathbf { W } \in \mathbb { R } ^ { r \times d }$ and the number of indices r is small compared to d. The function g is called the link function. Beyond intersections of halfspaces, MIMs also include many important function classes, such as shallow ReLU networks of bounded width [DKKZ20, CKM21, CDG<sup>+</sup>23, DK24, CN24] and multiclass linear classifiers [DIKZ25, DMRT25b]. We refer the reader to [BH25] for a survey.

Despite their broad applicability, multi-index models (MIMs) impose a specific form of lowdimensional structure: the target function must depend only on a small number of linear projections of the input. The restriction to linear indices matters even for functions of a single quadratic feature. For example, consider the spherical shell

$$
f ( \mathbf { x } ) = \mathbb { 1 } \left\{ a \leq \left\| \mathbf { x } \right\| _ { 2 } ^ { 2 } \leq b \right\} , \qquad 0 < a < b < \infty .
$$

The label depends on a single quadratic feature, $\| \mathbf { x } \| _ { 2 } ^ { 2 } .$ , but no projection onto a proper linear subspace of $\mathbb { R } ^ { d }$ determines it; existing guarantees for low-dimensional MIMs do not directly give useful learning guarantees for a function like this.

The shell example points to a broader research direction beyond MIMs: which functions of a few unknown nonlinear features can be learned eficiently under Gaussian inputs? See [BH25, Nic26] for discussion. From the perspective of learning geometric concepts, intersections of quadratic PTFs are arguably the most natural starting point. Consider the well-studied problem of learning intersections of halfspaces. If we keep the link function fixed and increase the degree of the defining polynomials from one to two, we obtain precisely the class studied here. The target then has the form

$$
f ( \mathbf { x } ) = g ( q _ { 1 } ( \mathbf { x } ) , \ldots , q _ { k } ( \mathbf { x } ) ) , \qquad g ( t _ { 1 } , \ldots , t _ { k } ) = \mathbb { 1 } \{ t _ { 1 } \geq 0 , \ldots , t _ { k } \geq 0 \} ,
$$

so the target depends on only k quadratic features, even though it may require d linear indices. The dificulty is to find useful quadratic features from labels that reveal only whether all k inequalities hold. To see where the linear case difers, consider an intersection of k halfspaces under Gaussian inputs. Its label depends only on the projection onto the span of its normal vectors. Estimates of the first two labeled moments (aka degree at most 2 Chow parameters), $\mathbf { E } [ f ( \mathbf { x } ) \mathbf { x } ]$ and $\mathbf { E } [ f ( \mathbf { x } ) ( \mathbf { x } \mathbf { x } ^ { \top } - \mathbf { I } _ { d } ) ]$ can be used to find a subspace $V$ of dimension $O ( k )$ such that an intersection with all its normal vectors in V approximates the target to error ϵ. Searching over intersections in the recovered subspace gives, for example, a running time of $d ^ { O ( 1 ) } ( k / \epsilon ) ^ { O ( k ^ { 2 } ) }$ , where d is the ambient dimension and the exponent of d is a universal constant independent of k and ϵ [Vem10a, DKS18].

This reduction does not directly extend to quadratic PTFs: even a single quadratic inequality need not be determined by the projection onto any proper linear subspace. One might instead lift the examples to the $\Theta ( d ^ { 2 } )$ -dimensional space of quadratic features, where the target becomes an intersection of k halfspaces. However, the lifted distribution is not Gaussian. Its coordinates are dependent, and the distributional properties used by the existing subspace-recovery algorithms no longer hold. Choosing an orthonormal Hermite basis makes these coordinates uncorrelated, but does not resolve this obstruction.

For arbitrary intersections of quadratic PTFs, the standard general-purpose (and the only known) approach is low-degree polynomial regression [KOS08]. Combining the Gaussian surface-area bound for PTFs [Kan11] with the realizable learning guarantee of [KOS08] gives a polynomial-regression algorithm for this class with running time $\breve { d } ^ { O ( k ^ { 2 } / \epsilon ^ { 2 } ) }$ .<sup>⋄</sup> Thus, even for $k = 2 .$ , the exponent of the dimension depends polynomially on $1 / \epsilon .$ This leaves the following basic question:

For every fixed k, can intersections of k degree-2 PTFs be learned under the Gaussian distribution in time $d ^ { O _ { k } ( 1 ) } F _ { k } ( 1 / \epsilon )$ , for some function $F _ { k }$ independent of d?

The above question allows arbitrary dependence on $1 / \epsilon ;$ it only asks that the exponent of d be inde pendent of the accuracy parameter. Our main result answers it afirmatively, with quasipolynomial dependence on $1 / \epsilon$

## 1.1 Our Results and Technical Contributions

We first define the target concept class and the PAC learning problem.

Definition 1.1 (Intersections of degree-2 PTFs). For $k \in \mathbb { Z } _ { + }$ , let $\mathcal { H } _ { k }$ be the class of functions $f : \mathbb { R } ^ { d }  \{ 0 , 1 \}$ of the form $f ( \mathbf { x } ) = \mathbb { 1 } \{ q _ { 1 } ( \mathbf { x } ) \geq 0 , \ldots , q _ { k } ( \mathbf { x } ) \geq 0 \}$ , where each $q _ { i } : \mathbb { R } ^ { d }  \mathbb { R }$ is a polynomial of degree at most two.

Definition 1.2 (PAC learning under the Gaussian distribution). Let D be an unknown distribution over $\mathbb { R } ^ { d } \times \{ 0 , 1 \}$ with marginal $D _ { \mathbf { x } } = \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ , and suppose that $y = f ^ { * } ( \mathbf { x } )$ almost surely for some $f ^ { \ast } \in \mathcal { H } _ { k }$ . Given i.i.d. sample access to D, an accuracy parameter $\epsilon \in ( 0 , 1 / 2 )$ , and a confidence parameter $\delta \in ( 0 , 1 )$ , the goal is to output a hypothesis $\widehat { f } : \mathbb { R } ^ { d } \to \{ 0 , 1 \}$ such that, with probability at least $1 - \delta , \mathrm { e r r } _ { D } ( \widehat { f } ) : = \mathbf { P r } _ { ( \mathbf { x } , y ) \sim D } [ \widehat { f } ( \mathbf { x } ) \neq y ] \leq \epsilon$ . The hypothesis need not belong to $\mathcal { H } _ { k }$

Our main algorithmic result is the following; see Theorem 9.1 for the formal statement and proof. For simplicity, we state the bound for constant confidence.

Theorem 1.3 (Main algorithmic result). There is an algorithm that PAC learns $\mathcal { H } _ { k }$ under the standard Gaussian distribution to error $\epsilon \in ( 0 , 1 / 2 )$ with probability at least 99%, using $d ^ { O ( k ) } 2 ^ { O _ { k } ( \log ^ { 3 } ( 1 / \epsilon ) ) }$ samples and time.

Here the constant implicit in $O _ { k } ( \cdot )$ may depend on k. For every fixed $k ,$ the running time is polynomial in d and quasipolynomial in $1 / \epsilon$ . Already for $k = 2$ , this gives $d ^ { O ( 1 ) } 2 ^ { O ( \log ^ { 3 } ( 1 / \epsilon ) ) }$ time, whereas polynomial regression gives $d ^ { O ( 1 / \epsilon ^ { 2 } ) }$ . Our algorithm is improper: it learns quadratic features $\phi _ { 1 } , \ldots , \phi _ { m }$ and outputs a hypothesis of the form

$$
{ \widehat { f } } ( \mathbf { x } ) = \mathbb { 1 } \{ P ( \phi _ { 1 } ( \mathbf { x } ) , \ldots , \phi _ { m } ( \mathbf { x } ) ) \geq 0 \} ,
$$

where $P$ is a polynomial and the number m of learned quadratic features is bounded by a function of k and ϵ. Thus, the output is a single PTF (whose degree may be larger than two).

Remark 1.4. In Theorem 7.1, we also develop a diferent learning algorithm with time complexity $d ^ { O ( k ) } 2 ^ { \mathrm { p o l y } _ { k } ( 1 / \epsilon ) }$ , whose output hypothesis is a decision tree of intersections of quadratic PTFs. We refer the reader to Section 7 for details.

We also prove a lower bound in the Statistical Query (SQ) model [Kea98], that qualitatively matches the complexity of our algorithm for constant error ϵ. In the $\mathrm { S Q }$ model, the learner accesses the labeled distribution through approximate expectations of bounded query functions. The following theorem shows that a linear dependence on k in the dimension exponent is necessary for SQ algorithms; see Theorem 10.2 for the formal statement and proof.

Theorem 1.5 (SQ lower bound). Fix a constant $\gamma \in ( 0 , 1 / 2 )$ . For d suficiently large as a function of k and $\gamma ,$ , any SQ algorithm that learns $\mathcal { H } _ { k }$ under the standard Gaussian distribution to error at most $1 / 2 - \gamma$ , with success probability at least $2 / 3$ , must either use a query of tolerance at most $d ^ { - \Omega ( k ) }$ or make at least $2 ^ { d ^ { \Omega ( 1 ) } }$ queries.

Thus, the $d ^ { O ( k ) }$ dependence of our upper bound is essentially optimal in the SQ framework, where complexity accounts for both the number of queries and their inverse tolerance. This gives a computational distinction from intersections of halfspaces. In the realizable Gaussian setting, the moment-based algorithms for halfspace intersections can be implemented using $d ^ { O ( 1 ) } ( k / \epsilon ) ^ { O ( k ^ { 2 } ) }$ statistical queries and tolerance $1 / \mathrm { p o l y } ( d , k , 1 / \epsilon )$ , with a universal exponent of d [Vem10a, DKS18].<sup>⋄</sup> Our SQ lower bound already holds when all the defining quadratics depend on the same unknown linear projection. It leaves open whether the running time can be polynomial in both d and $1 / \epsilon$ for every fixed k.

Technical contributions. Beyond our learning guarantees, we develop a collection of structural and algorithmic tools that may be useful for other learning problems involving nonlinear features under Gaussian inputs. Our approach combines Gaussian analysis, algebraic arguments, spectral methods, and polynomial approximation. Specifically, we highlight the following:

1. Low-degree certificates for directional influence. We develop a framework for analyzing the directional dependence of Boolean functions and constructing the corresponding low-degree certificate through their Gaussian-weighted decision boundaries. By restricting a Boolean function to Gaussian lines, we show (Lemma 5.7) that nontrivial resampling influence along a direction implies a nontrivial amount of Gaussian-weighted boundary mass along that direction. We further introduce a Gaussian boundary operator (Definition 5.2) that, through integration by parts, expresses the correlation between the Boolean function and a polynomial in terms of the change signed values of the polynomial at these decision boundaries (Lemma 5.6). This provides a systematic way to translate information about the changes in a Boolean function into polynomial correlations. For intersections of k quadratic PTFs, we develop a quantitative construction of such certificates, showing that any direction with nontrivial resampling influence admits a polynomial witness of degree O(k) with nontrivial correlation. (Sections 5-6.)

2. Robust algebraic tools for degenerate decision boundaries. We develop quantitative structural techniques for analyzing degenerate polynomial decision boundaries, which allow the boundarybased certificate framework to work without separation or genericity assumptions. Our key approach combines algebraic tools, including discriminants and resultants, with Gaussian anticoncentration to translate approximate root degeneracies on a nontrivial fraction of Gaussian slices into global small-norm polynomial relations. In particular, we establish structural simplifications for nearly double roots (Subsection (5.3)) and a robust factorization theorem (Lemma 5.19) for nearly shared roots. These structural alternatives allow us to simplify the defining intersection and construct polynomial certificates without imposing separation or genericity assumptions on its decision boundaries. (Section 5.3-5.5.)

3. Generalized regularity and Gaussian comparison. We develop a generalized notion of regularity (Definition 3.1) for quadratic polynomials relative to a subspace $V \subseteq \mathbb { R } ^ { d }$ , allowing arbitrary dependence on V while controlling the residual quadratic features in directions orthogonal to V. We show (Theorem 4.1) that an intersection with suficiently small resampling influence outside V can be approximated by one satisfying this regularity condition, even when its original defining polynomials are highly irregular. Under this generalized regularity condition, we establish a Gaussian comparison result (Lemma 3.3) that replaces the residual quadratic features by linear Gaussian features while preserving their dependence on V . Together, these results provide a framework for extending moment-based learning techniques from linear Gaussian models to nonlinear polynomial features.

4. Derivative-based subspace recovery in polynomial feature spaces. We develop new structural techniques for recovering relevant directions in a space of quadratic polynomial features. Unlike standard subspace recovery, the feature space we consider consists of residual linear, mixed, and quadratic terms, whose coordinates are not jointly Gaussian and do not admit the usual notion of directional diferentiation. To overcome this dificulty, we first establish a structural result (Lemma 9.2) for intersections of PTFs in an idealized Gaussian model, showing that influential directions in the residual feature space can be certified by low-degree polynomial correlations. We then introduce a new diferential operator for quadratic features and its adjoint under Gaussian measure (Lemma 9.4 and Lemma 9.5), allowing us to approximate the corresponding directional derivative correlations using labeled moments of the original input distribution. Under our generalized regularity condition, this yields a bilinear form that approximates an operator of rank at most k. Consequently, spectral methods recover a low-dimensional space of quadratic features suficient for learning the target, without requiring identification of the original defining polynomials.

5. Robust polynomial approximation and hinge-loss minimization. Building on polynomial approximation techniques for learning intersections of margin halfspaces [KS04], we develop a robust version suitable for intersections of quadratic PTFs under Gaussian inputs. Specifically, we construct polynomial approximations with simultaneously small expected hinge loss and controlled Gaussian $L _ { 2 }$ norm, while preserving the quadratic feature space of the target. The norm bound allows us to control the additional loss introduced by errors in the recovered feature representation. Consequently, we can learn over an approximately correct feature space through convex hinge-loss minimization.

## 1.2 Technical Overview

Subspace Recovery for Intersections of Halfspaces. We first recall the approach for learning intersections of halfspaces under a Gaussian distribution. Although learning intersections of halfspaces under general marginal distributions is a challenging problem, whose computational complexity is far from being understood (even for the case of $k = 2 )$ , the Gaussian distribution has nice structural properties that enable eficient algorithms; see, $\mathrm { e . g . }$ , [BK97, Vem10a, DKS18].

Specifically, consider an instance of an intersection of k halfspaces $f ( \mathbf { x } ) = \mathbb { 1 } \{ \mathbf { w } ^ { ( 1 ) } \mathbf { \cdot x } \geq t _ { 1 } , \mathbf { \cdot } \mathbf { \cdot } , \mathbf { w } ^ { ( k ) }$ $\textbf { x } \geq \ t _ { k } \}$ and let V be the (at most) k-dimensional subspace spanned by $\mathbf { w } ^ { ( 1 ) } , \ldots , \mathbf { w } ^ { ( k ) }$ The intuition is that if we look at the correlation between $f ( \mathbf { x } )$ and the first two moments of the marginal distribution, $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ f ( \mathbf { x } ) \mathbf { x } ]$ and $\mathbf E _ { \mathbf x \sim \mathcal { N } ( 0 , \mathbf I ) } [ f ( \mathbf { x } ) ( \mathbf { x x } ^ { \top } - \mathbf I ) ]$ , then for every $\mathbf { v } \perp V , \mathbf { v }$ is orthogonal to $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ f ( \mathbf { x } ) \mathbf { x } ]$ and E<sub>x∼N(0,I)</sub>[f(x)(xx<sup>⊤</sup> − I)] has 0 eigenvalue along $\mathbf { v } ,$ because $f ( \mathbf { x } )$ is independent of $\mathbf { v } \cdot \mathbf { x }$ . On the other hand, if $f ( \mathbf { x } )$ heavily depends on a unit direction v, then either $\big | \mathbf { v } \cdot \mathbf { E _ { x \sim \mathcal { N } ( 0 , \mathbf { I } ) } } [ f ( \mathbf { x } ) \mathbf { x } ] \big | \mathrm { ~ o r ~ } \big | \mathbf { v } ^ { \top } \mathbf { E _ { x \sim \mathcal { N } ( 0 , \mathbf { I } ) } } [ f ( \mathbf { x } ) ( \mathbf { x } \mathbf { x } ^ { \top } - \mathbf { I } ) ] \mathbf { v } \big |$ will be large.

As a result, if we take W to be the subspace spanned by $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ f ( \mathbf { x } ) \mathbf { x } ]$ and the eigenvectors of $\mathbf E _ { \mathbf x \sim \mathcal { N } ( 0 , \mathbf I ) } [ f ( \mathbf x ) ( \mathbf x \mathbf x ^ { \top } - \mathbf I ) ]$ ] corresponding to its k eigenvalues of largest absolute value, then there must be an instance whose defining vectors are in $W$ , close to $f ( \mathbf { x } )$

As a result, if we brute-force search over $W ,$ , we solve the learning problem in $d ^ { O ( 1 ) } ( k / \epsilon ) ^ { O ( k ^ { 2 } ) }$ time. In fact, as shown in [DKS18], even if the input distribution is corrupted by a small amount of nasty noise, we can still use the same method to learn a good hypothesis from the noisy distribution.

Polynomial Regularity and Gaussian Comparison. Given the discussion above, when working with an intersection of k PTFs, $f ( \mathbf { x } ) = \mathbb { 1 } \{ q _ { 1 } ( \mathbf { x } ) \geq 0 , \dots , q _ { k } ( \mathbf { x } ) \geq 0 \}$ , a natural attempt would be to use polynomial features. For a Gaussian distribution, we choose to use $\Phi ( \mathbf { x } ) \in \mathbb { R } ^ { d ( d + 3 ) / 2 }$ , where each coordinate of $\Phi ( \mathbf { x } )$ is a normalized Hermite polynomial with degree at most 2, so that the coordinates are orthogonal under the inner product defined under the Gaussian distribution. Under this feature basis, any $q _ { i } ( \mathbf { x } ) = \theta _ { i } \cdot \Phi ( \mathbf { x } ) - t _ { i }$ . The idea is that if $\Phi ( \mathbf { x } )$ is Gaussian distributed, then we can apply the same algorithm above to search for good parameters $( \hat { \theta } _ { i } , \hat { t } _ { i } ) , i \in [ k ]$ and return the corresponding intersections of PTFs. Unfortunately, this is not true, and the distribution of $\Phi ( \mathbf { x } )$ is far from a Gaussian distribution. However, inspired by techniques developed in pseudo-randomness literature $[ \mathrm { D K N 1 0 } , \mathrm { D H K ^ { + } 1 0 } , \mathrm { D D S 1 4 } ] ,$ for a nice linear function $L : \mathbb { R } ^ { d ( d + 3 ) / 2 } \to \mathbb { R } , L ( \Phi ( \mathbf { x } ) )$ , which corresponds to some quadratic function $q ( \mathbf { x } )$ , is approximately Gaussian when $\mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } )$ . We call a degree-2 polynomial $p$ to be τ-regular, if for every unit vector $\mathbf { v } , \| \partial _ { \mathbf { v } } p \| ^ { 2 } \leq \tau ^ { 2 } \mathbf { V a r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p ( \mathbf { x } ) ]$ ], where $\| \partial _ { \mathbf { v } } p \|$ is called the influence of p along the v direction. Equivalently, if we consider a quadratic function $p ( \mathbf { x } ) = \mathbf { b } \cdot \mathbf { x } + \mathbf { x } ^ { \top } \mathbf { A } \mathbf { x } - \mathrm { t r } ( \mathbf { A } ) - t ,$ then the regularity condition is equivalent to saying that $\| \mathbf b \|$ is small and $\| \mathbf { A } \| _ { \mathrm { o p } }$ is small. In fact, if $p _ { i } ( \mathbf { x } ) = \theta _ { i } \cdot \Phi _ { i } ( \mathbf { x } ) , i \in [ k ]$ , are all regular for some suficiently small $\tau ,$ then the regularity condition guarantees that the distribution of $( p _ { 1 } ( \mathbf { x } ) , \ldots , p _ { k } ( \mathbf { x } ) ) , \mathbf { x } \sim { \mathcal { N } } ( 0 , \mathbf { I } )$ approximates a Gaussian replacement vector $( \theta _ { 1 } \cdot G , \ldots , \theta _ { k } \cdot G )$ , where G is a standard Gaussian vector. To see why this helps us, we prove a Gaussian comparison lemma (Lemma 3.3) that for any smooth function $\Psi : \mathbb { R } ^ { k }  \mathbb { R }$ and every direction $\mathbf { u } \in \mathbb { R } ^ { d ( d + 3 ) / 2 } , j = 0 , 1 , 2$

$$
\operatorname { \bf { E } } _ { \sim \sim \mathcal { N } ( 0 , { \bf { I } } ) } [ \Psi ( p _ { 1 } ( { \bf { x } } ) , \dots , p _ { k } ( { \bf { x } } ) ) h _ { j } ( { \bf { u } } \cdot \Phi ( { \bf { x } } ) ) ] \approx \underset { G \sim \mathcal { N } ( 0 , { \bf { I } } ) } { \bf { E } } [ \Psi ( \theta _ { 1 } \cdot G , \dots , \theta _ { k } \cdot G ) h _ { j } ( { \bf { u } } \cdot G ) ] .
$$

That is to say, if the defining polynomials of $f ( \mathbf { x } ) , q _ { 1 } , \ldots , q _ { k }$ are suficiently regular, then the moment information and structure of the polynomial threshold function is suficiently close to an instance of intersection of halfspaces in $\mathbb { R } ^ { d ( d + 3 ) / 2 }$ under a Gaussian distribution, with the same defining vectors. Then we can simply treat an intersection of PTFs as its Gaussian replacement for the analysis. As a result, we can directly run the robust learning algorithm from [DKS18] over $\Phi ( \mathbf { x } )$ to eficiently learn a hypothesis with error depending on the regularity parameter τ .

Randomized Error and Extracting Influential Directions. Unfortunately, there is no guarantee that our defining polynomials $q _ { i }$ will be regular, and so our Gaussian comparison lemma may fail to hold. There are two possible reasons why this might be the case.

On the one hand, it is possible that the polynomials that we chose to represent $f$ are irregular, but that another choice would not have been. For example, suppose $f ( \mathbf { x } )$ contains a defining polynomial $q ( \mathbf { x } ) = \mathbf { x } _ { 1 } ^ { 2 } + \mathbf { x } _ { 2 } ^ { 2 }$ , which is highly irregular. On the other hand, since $q ( \mathbf { x } ) \geq 0$ always holds, we can remove it from $f ( \mathbf { x } )$ and find an equivalent representation for the target function. In fact, consider an irregular polynomial $p ( \mathbf { x } )$ . Since a bad direction must either correlate well with the linear part of $p ,$ or have large norm under the bilinear form defined by the quadratic part, we can find a low dimensional subspace (spanned by the linear part and the top few eigenvectors of the quadratic form) that is responsible for all of the bad directions. If we are allowed to remove the dependence on this subspace (for example, by replacing this component of the input by a random constant), then we can reduce $p$ to the regular case. Our goal is to apply this procedure to each defining polynomial $q _ { i }$ so that the resulting polynomials $\tilde { q } _ { i }$ are regular and the intersection they define is close to $f .$

However, to achieve this without introducing much error, we need to know that re-randomizing the input of $f$ along a low-dimensional subspace is unlikely to change the output. This will hold so long as $f$ is not heavily influenced by any particular direction (as we can change the input on a subspace by changing it one coordinate at a time). To quantify this influence, we introduce the notion of randomized error along a direction v as

$$
\mathrm { R a n d } _ { f } ( \mathbf { v } ) = \sum _ { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { \mathbf { v } ^ { \bot } } ) } \left[ \sum _ { z , z ^ { \prime } \sim \mathcal { N } ( 0 , 1 ) } [ f ( \mathbf { w } + z \mathbf { v } ) \neq f ( \mathbf { w } + z ^ { \prime } \mathbf { v } ) ] \right] .
$$

As we show in Theorem 4.1, if Rand ${ \bf \nabla } _ { f } ( { \bf v } )$ is small for every direction $\mathbf { v } ,$ then $f$ must be close to a τ-regular instance up to a small $L _ { 1 }$ error.

Unfortunately, sometimes $f$ will have directions of large influence, in which case the irregularity of its representation will be intrinsic. In particular, there may be some directions v with ${ \mathrm { R a n d } } _ { f } ( \mathbf { v } )$ reasonably large. In such a case, no matter how we represent it, some of the defining polynomials of $f$ will have large influence in the v-direction. To deal with this issue, we will actually need to handle these directions separately. The hope is that with sample access to $f ,$ we can directly identify these directions $\mathbf { v } ,$ , with ${ \mathrm { R a n d } } _ { f } ( \mathbf { v } )$ being nontrivial large. To do this, we show that a direction v will have large influence on f only if it has large influence on the low-degree part of $f$ (Theorem 5.1). In particular, letting $F _ { \ell } ( \mathbf { x } ) = \mathrm { p r o j } _ { P _ { < \ell } } ( f ( \mathbf { x } ) )$ be the degree-ℓ-part of $f$ for $\ell$ a large enough multiple of $k ,$ we show that for any direction v with Rand $f ( \mathbf { v } ) \geq \alpha$ that it must hold that $\mathbf { E } [ ( \partial _ { v } F _ { \ell } ) ^ { 2 } ] \geq \mathrm { p o l y } _ { k } ( \alpha )$ In particular, this means that defining the influence matrix as $G _ { \ell } : = \mathbf { E _ { x } } _ { \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \nabla F \nabla F ^ { \top } ]$ , that any influential direction for $f$ will have $\mathbf { v } ^ { T } G _ { \ell } \mathbf { v }$ relatively large. Hence, taking V to be the subspace spanned by the most significant eigenvectors of $G _ { \ell }$ , we can eficiently find a subspace $V$ so that for all v orthogonal to $V , \operatorname { R a n d } _ { f } ( \mathbf { v } )$ is very small (see Theorem 6.2). By Theorem 5.1, if $G _ { \ell }$ has no large eigenvalues, then $f$ has no direction of large resampling influence. If $\operatorname { R a n d } _ { f } ( \mathbf { v } )$ is large for some unit vector $\mathbf { v } ,$ then $G _ { \ell }$ has a large eigenvalue, and a top eigenvector gives a direction of large influence on the low-degree part of $f .$

Certificate Polynomial Construction. We now explain the main ideas behind our main structural result (Theorem 5.1). Specifically, we show that for an intersection of k degree-2 PTFs $f _ { i }$ if a direction v has Rand ${ \bf \nabla } _ { f } ( { \bf v } ) \geq \alpha$ , then there must be some $r \in [ 2 k ]$ and a unit-norm polynomial $p ( \mathbf { w } )$ over $\mathbf { v } ^ { \perp }$ such that

$$
\underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ f ( \mathbf { x } ) h _ { r } ( \mathbf { v } \cdot \mathbf { x } ) p ( \mathbf { w } ) ] \geq \mathrm { p o l y } _ { k } ( \alpha ) ,
$$

and $r + \deg ( p ) \leq 2 k$ . In other words, as long as ${ \mathrm { R a n d } } _ { f } ( \mathbf { v } )$ is non-trivially large, then $f ( \mathbf { x } )$ must have a non-trivial correlation with a normalized polynomial of degree at most 2k, with no-constant-in-v component.

To give the intuition behind Theorem 5.1, we decomposed the target function f according to the value $\textbf { w } \perp \textbf { v }$ . Fix a direction v and write $\mathbf { x } = z \mathbf { v } + \mathbf { w }$ , where $\textbf { w } \perp \textbf { v }$ . For each fixed w, the restriction $f _ { \mathbf { w } } ( z ) = f ( z \mathbf { v } + \mathbf { w } )$ is a one-dimensional union of intervals with at most 2k decision boundaries. Furthermore, if ${ \mathrm { R a n d } } _ { f } ( \mathbf { v } ) \geq \alpha$ , we show in Lemma 5.7 that these boundaries have total Gaussian-weighted mass at least $\Omega ( \alpha )$

As $f _ { \mathbf { w } } ( z )$ is a step function, directly computing correlations with f will be dificult. However, this means that $\partial _ { \mathbf { v } } f _ { \mathbf { w } }$ is a sum of δ-functions and much easier to work with. Furthermore, the operator $M _ { \mathbf { v } } Q = ( \mathbf { v } \cdot \mathbf { x } ) Q - \partial _ { \mathbf { v } } Q$ is adjoint to the derivative operator over the Gaussian distribution. Namely, for functions g and $h ,$ , we have ${ \bf E } [ ( \partial _ { \bf v } g ) h ] = { \bf E } [ g M _ { \bf v } h ]$ . Applying this with $g = f _ { \mathbf { w } }$ , we $\mathrm { g e t }$

$$
\underline { { \mathbf { E } } } [ f ( \mathbf { x } ) M _ { \mathbf { v } } Q ( \mathbf { x } ) ] = \underline { { \mathbf { E } } } \left[ \sum _ { s \in S _ { \mathbf { w } } } \Delta f _ { \mathbf { w } } ( s ) Q ( s \mathbf { v } + \mathbf { w } ) \phi ( s ) \right] \ ,
$$

where $S _ { \mathbf { w } }$ denotes the set of decision boundaries of $f _ { \mathbf { w } }$ and $\Delta f _ { \mathbf { w } } ( s ) \in \{ \pm 1 \}$ is the sign change of $f _ { \mathbf { w } }$ at boundary s. Thus, our goal is to construct a low-degree polynomial $Q$ whose value at each boundary has the same sign as the corresponding change in $f _ { \mathbf { w } }$ . Furthermore, using the polynomial $P = M _ { \mathbf { v } } Q$ has the advantage that the integral over v direction of $P$ is $\mathbf { E } [ M _ { \mathbf { v } } Q \cdot 1 ] = \mathbf { E } [ Q \cdot \partial _ { \mathbf { v } } 1 ] = 0$ Thus, this construction will naturally produce functions with no constant-in-v-component.

For a generic intersection, such a polynomial has a simple construction. Writing $f ( \mathbf { x } ) = \mathbf { 1 } \{ q _ { 1 } ( \mathbf { x } ) \geq$ $0 , \ldots , q _ { k } ( \mathbf { x } ) \geq 0 \}$ , consider $Q _ { \mathbf { v } } = \partial _ { \mathbf { v } } ( \prod _ { i = 1 } ^ { k } q _ { i } )$ . At every decision boundary, the sign of $Q _ { \mathbf { v } }$ agrees with the direction of the jump, implying that the inner product is always non-negative. Furthermore, if the derivative of the active polynomial and the values of all other defining polynomials are bounded away from zero, then the boundary contributes a nontrivial positive amount to the correlation. However, this argument can still fail to provide a good lower bound if $\textstyle \prod _ { i = 1 } ^ { k } q _ { i }$ has pairs of roots that are very close to each other—either due to some of the $q _ { i }$ having (approximate) double roots or some pairs approximately sharing roots.

To handle these degeneracies, we develop several structural tools that combine algebraic techniques with Gaussian concentration and anti-concentration to establish quantitative structural dichotomies. Our starting point is that, after fixing ${ \textbf { w } } \perp { \textbf { v } } ,$ each defining polynomial becomes a univariate quadratic whose coeficients are low-degree polynomials in $\mathbf { w } .$ . This allows us to use the discriminant to analyze nearly double roots and the resultant to analyze nearly shared roots. Importantly, both quantities are themselves low-degree polynomials in $\mathbf { w } _ { \mathrm { : } }$ so Gaussian anti-concentration allows us to convert their smallness on a nontrivial fraction of slices into a global small- $L _ { 2 } .$ -norm condition. These conditions enable us to derive global structural results to simplify the defining polynomials.

For nearly double roots, we establish a dichotomy (Subsection (5.3)): either such roots occur too infrequently to contribute significantly to the Gaussian-weighted decision boundary, or the corresponding threshold function can be approximated by a function independent of the v direction. For nearly shared roots, we develop a robust factorization argument (Subsection (5.4)) showing that either one polynomial is nearly independent of $\mathbf { v } ,$ the two polynomials are nearly proportional, or they can be approximated by polynomials sharing a common afine factor. In particular, if several defining polynomials both contain some afine function L as a linear factor, then this common afine factor $L$ allows us to construct a new certificate using the factors of the defining polynomials (Lemma 5.24). In either case, if the roots are close, these structural alternatives allow us to simplify the defining intersection, up to a controlled approximation error, whenever the generic certificate does not yield suficient correlation.

Finally, we combine these tools through an iterative construction to find the desired certificate polynomial (Subsection (5.5)). Starting from the target function $f ( \mathbf { x } )$ , at each step, we either obtain a certificate with nontrivial correlation or replace the current intersection by a simpler one by introducing a small $L _ { 1 }$ error. Importantly, after each replacement we reconsider the actual decision boundaries of the new intersection, since the previous boundaries need not be preserved. By the structural results for nearly double roots and nearly shared roots that we derived above, each unsuccessful step removes or decomposes a defining quadratic or identifies approximately proportional afine factors, so only $O ( k )$ changes are possible. By carefully controlling the accumulated approximation error and simplifying multiple constraints together, we obtain the correlation guarantee of Theorem 5.1.

A $d ^ { O ( k ) } 2 ^ { \mathrm { p o l y } _ { k } ( 1 / \epsilon ) }$ Time Learning Algorithm. The technical ingredients developed thus far are already enough to give us a learning algorithm for intersections of k degree-2 PTFss with $d ^ { O ( k ) }$ complexity dependence. As discussed above, our goal is to remove the dependence on the directions that have high influence on $f ( \mathbf { x } )$ , by randomizing the input along those directions. Naively, one would simply compute the high influence subspace $V$ for $f$ and then condition on the projection onto V (or more precisely, divide V into tiny bins and consider separately the samples that project onto each bin, efectively removing the dependence on the V-direction). Unfortunately, there is no guarantee that conditioning on the projection to $V$ will not cause there to be new directions of large influence. So instead we must apply this procedure adaptively. After restricting to a bin, we re-run our algorithm for finding one high influence direction, and if we find one, we condition further by partitioning the bin into many small bins along the found direction. Intuitively, we maintain a tree rooted at the target function $f ,$ whose leaves form a partition of $\mathbb { R } ^ { d }$ . Each node of the tree is associated with an intersection of $k \mathrm { P T F s } .$ obtained by freezing the input value of the instance at its parent node along a single direction through the above branching process.

It remains to show that this procedure actually terminates in a reasonable number of steps. To do this, we consider the $L _ { 2 }$ norm of the low-degree Hermite projection of $f .$ Specifically, suppose that we branch $f$ along a direction v and get a series of functions $f ^ { ( 1 ) } , \dots , f ^ { ( m ) }$ , with x · v frozen to be diferent values. Since the v influence of the low-degree part of $f$ must be large and since conditioning on the v direction efectively decreases the degree of this component, we will have that $\textstyle \sum _ { i < \ell } \| f _ { \leq i } \| _ { 2 } ^ { 2 }$ (where $f _ { \leq i }$ is the Hermite projection of $f$ of degree at most i) is strictly less than its average over the $f ^ { ( j ) }$ . Since this sum is never more than $\ell ,$ for $\ell = O ( k )$ , the branching operation will terminate after at most $\mathrm { p o l y } _ { k } ( 1 / \epsilon )$ rounds. Using Markov’s inequality, we can show that the tree has $2 ^ { \mathrm { p o l y } _ { k } ( 1 / \epsilon ) }$ many leaves. Furthermore, most of the leaf functions will have no directions of large influence, so as discussed above, we may approximate it by an intersection of regular PTFs. Since these polynomials do behave like Gaussians, we may employ the ideas of [DKS18] to find a low-dimensional space of degree- $- 2$ polynomials that nearly contains the defining polynomials of our intersection. We can then find these via a brute force search. As a result, by running the learning algorithm for regular instances $2 ^ { \mathrm { p o l y } _ { k } ( 1 / \epsilon ) }$ times, we learn a decision tree, with each leaf associated with an intersection of k PTFs that is $\mathrm { p o l y } ( \epsilon )$ close to the target function $f .$

Improved Algorithm via Hinge Loss Minimization. Unfortunately, the tree-based algorithm developed above has a running time that depends exponentially on a polynomial in $1 / \epsilon$ , since it repeatedly partitions the Gaussian space along influential directions. To improve this exponential dependence, we draw inspiration from prior works (see, e.g., [KS04]) that study learning intersections of halfspaces with γ-margin. Roughly speaking, if all examples x have unit norm and are γ-far from every decision boundary of an intersection of k halfspaces $f ( \mathbf { x } )$ , then there exists a polynomial $p$ of degree $O _ { k } ( \log ( 1 / \gamma ) )$ such that $\mathrm { s i g n } ( p ( \mathbf { x } ) ) = f ( \mathbf { x } )$ for every example in the support of the distribution.

So, a linear programming approach that runs in $d ^ { O _ { k } ( \log ( 1 / \gamma ) ) }$ time can output a single PTF that is close to $f .$ . This approach has two shortcomings. First, such an algorithm has no tolerance to label noise. Second, it does not work under any continuous distribution, where examples can be arbitraril close to the decision boundary. In fact, if we treat our target function $f ( \mathbf { x } )$ as an intersection of $k$ halfspaces in $\mathbb { R } ^ { d ( d + 3 ) / 2 }$ , defined by $\theta _ { i } \cdot \Phi ( \mathbf { x } )$ , then there is no similar known way for us to use some polynomial approximation-based algorithm with degree poly $\log ( 1 / \epsilon )$

In Section 8, we give a polynomial approximation that does not require a margin assumption. For every intersection $f : \mathbb { R } ^ { d }  \{ 0 , 1 \}$ of k quadratic PTFs, there is a polynomial $p$ of degree $O _ { k } ( \log ( 1 / \epsilon ) )$ such that

$$
\underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ \mathrm { H i n g e } ( 2 f ( \mathbf { x } ) - 1 , p ( \mathbf { x } ) ) ] \leq \epsilon , \qquad \| p \| _ { 2 } \leq 2 ^ { O _ { k } ( \log ^ { 2 } ( 1 / \epsilon ) ) } .
$$

Here Hinge $( y , t ) = \operatorname* { m a x } \{ 0 , 1 - y t \}$ for $y \in \{ \pm 1 \}$ . Our construction is based on rational approximations of the sign function. By combining these rational approximations and clearing their positive denominators, we obtain a polynomial that has a large margin away from the decision boundaries. Gaussian anti-concentration and tail bounds then allow us to control its expected hinge loss and $L _ { 2 }$ norm. Since hinge loss is convex in the polynomial coeficients, we can learn such a polynomial eficiently using stochastic convex optimization over the space of monomials. Importantly, we show that the sign of the learned polynomial is ϵ-close to the target function $f ( \mathbf { x } )$

However, directly performing this optimization over all $\deg \mathrm { r e e } - O _ { k } ( \log ( 1 / \epsilon ) )$ polynomials in $\mathbb { R } ^ { d }$ would require $d ^ { O _ { k } ( \log ( \bar { 1 } / \epsilon ) ) }$ time, which is even less desirable than our previous $\mathrm { \Phi } _ { d } O ( k ) _ { 2 } \mathrm { p o l y } _ { k } ( 1 / \epsilon )$ time algorithm. The reason is that the search space has too large a dimension. Suppose instead that we are given a low-dimensional subspace $\mathcal { H } \subseteq \mathcal { P } _ { \leq 2 } ( \mathbb { R } ^ { d } )$ such that the target function $f$ can be represented as an intersection of k quadratic PTFs with defining polynomials in $\mathcal { H } .$ . Then by running the hinge loss minimization algorithm over $\mathcal { H }$ , we get a dim $\bar { \mathbf { \eta } } _ { 1 } ( \bar { \mathcal { H } } ) ^ { \bar { O } _ { k } ( \bar { \log } ( 1 / \epsilon ) ) }$ time algorithm. Unfortunately, such a subspace is not available to us. And even if we do have one such a subspace, we can usually only promise that some intersection of PTFs from the subspace is close to $f ,$ which means the hinge loss minimization algorithm has to tolerate label noise. In fact, due to the norm bound of the approximate polynomial, we need a much stronger approximation guarantee than ϵ hinge loss error to make the algorithm noise tolerant. If $\begin{array} { r } { \mathbf { P r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ f ( \mathbf { x } ) \neq g ( \mathbf { x } ) ] \leq \rho , } \end{array}$ then the additional hinge loss incurred by a polynomial of norm at most $W$ can be bounded by $2 W _ { \parallel } / \bar { \rho } .$ . Thus, it sufices to find a space $\mathcal { H }$ that contains an intersection with error $\rho = 2 ^ { - O _ { k } ( \log ^ { 2 } ( 1 / \epsilon ) ) }$ . Our goal is to find such a space with dimension at most $2 ^ { O _ { k } ( \log ^ { 2 } ( 1 / \epsilon ) ) }$ , while maintaining a running time of $d ^ { O ( k ) }$ times a quasipolynomial function of $1 / \epsilon$

A Two-Stage Dimension Reduction Approach. To achieve this, we develop a two-stage dimension-reduction procedure in Section 9. In the first stage, we apply Theorem 5.1 and Theorem 6.2 to find a subspace $V \subseteq \mathbb { R } ^ { d }$ of dimension $\mathrm { p o l y } _ { k } ( 1 / \rho )$ , such that $f$ has suficiently small randomized error along every direction orthogonal to $V$ . Writing ${ \mathbf { x } } = { \mathbf { z } } + { \mathbf { w } }$ , where ${ \bf z } = \mathrm { p r o j } _ { V } ( { \bf x } )$ and ${ \bf w } = \mathrm { p r o j } _ { V ^ { \perp } } ( { \bf x } )$ For each defining polynomial $q _ { i } ,$ we write

$$
\begin{array} { r l } & { q _ { i } ( \mathbf { x } ) = a _ { i } ( \mathbf { z } ) + ( \mathbf { b } _ { i } + \mathbf { L } _ { i } ^ { \top } \mathbf { z } ) ^ { \top } \mathbf { w } + \mathbf { w } ^ { \top } \mathbf { A } _ { i } \mathbf { w } - \operatorname { t r } ( \mathbf { A } _ { i } ) } \\ & { N _ { \mathbf { u } _ { i } } ( \mathbf { z } , \mathbf { w } ) = ( \mathbf { b } _ { i } + \mathbf { L } _ { i } ^ { \top } \mathbf { z } ) ^ { \top } \mathbf { w } + \mathbf { w } ^ { \top } \mathbf { A } _ { i } \mathbf { w } - \operatorname { t r } ( \mathbf { A } _ { i } ) , } \end{array}
$$

where $a _ { i } ( \mathbf { z } )$ is a quadratic function that only depends on $\mathbf { z } ,$ and $\mathbf { A } _ { i }$ has no components in $V ,$ while $\mathbf { u } _ { i } = ( \mathbf { b } _ { i } , \mathbf { L } _ { i } , \mathbf { A } _ { i } )$ is the vector of coeficients of $N _ { { \mathbf { u } } _ { i } }$ . However, finding such a subspace $V$ does not mean that the target is close to an intersection supported only on $V .$ . Although every individual direction $\mathbf { v } \in V ^ { \perp }$ has small randomized error, the directions in $V ^ { \perp }$ may still collectively have a significant influence on the target. In terms of the above decomposition, this means that the residual polynomials $N _ { \mathbf { u } _ { i } } ( \mathbf { z } , \mathbf { w } )$ cannot simply be discarded. On the other hand, the space $\boldsymbol { \mathcal { S } }$ spanned by polynomials of the form $N _ { \mathbf { u } } ( \mathbf { z } , \mathbf { w } )$ might still have dimension $\Omega ( d ^ { 2 } )$ . As a result, we need a second stage of dimension reduction to find a low-dimensional subspace of $\boldsymbol { \mathcal { S } }$ that captures the influence of these residual polynomials $N _ { \mathbf { u } _ { i } } , i \in [ k ]$

The central idea that we use for this second stage of dimension reduction is based on a more general notion of regularity and Gaussian comparison lemma developed in Section 3. Specifically, let $V \subseteq \mathbb { R } ^ { d }$ be a subspace. We say that a quadratic polynomial $p$ is $( \tau , V )$ regular, if $\| \partial _ { \mathbf { v } } p \| ^ { 2 } \leq$ $\tau ^ { 2 } \mathbf { V a r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p ( \mathbf { x } ) ]$ for every unit vector $\mathbf { v } \perp V$ . In other words, the full regularity condition can be seen as a special case of $( \tau , V )$ regularity, where we take $V = \{ 0 \}$ . Unlike the tree-based algorithm, the $( \tau , V )$ -regularity condition allows us to flexibly keep the dependence on z instead of fixing its value. In fact, the regularity condition guarantees that the distribution of $( { \bf z } , N _ { { \bf u } _ { 1 } } ( { \bf z } , { \bf w } ) , \ldots , N _ { { \bf u } _ { k } } ( { \bf z } , { \bf w } ) )$ $\mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } _ { V } ) , \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { V ^ { \perp } } )$ approximates its Gaussian replacement $( \mathbf { z } , \mathbf { u } _ { 1 } \cdot G , \ldots , \mathbf { u } _ { k } \cdot G )$ . This allows us to perform our analysis over a Gaussian model, where we replace each $N _ { \mathbf { u } _ { i } } ( \mathbf { z } , \mathbf { w } )$ by $\langle \mathbf { u } _ { i } , G \rangle$ for G an independent standard Gaussian vector in the residual coeficient space. The resulting function

$$
\tilde { f } ( \mathbf { z } , G ) = \mathbb { 1 } \left\{ \tilde { q } _ { i } : = a _ { i } ( \mathbf { z } ) + \langle \mathbf { u } _ { i } , G \rangle \geq 0 , \forall i \in [ k ] \right\}
$$

is an intersection of halfspaces in $G$ for every fixed z. As we show in Lemma 3.4, if $q _ { 1 } , \ldots , q _ { k }$ are all $( \tau , V )$ regular, then for any smooth function $\Psi : \mathbb { R } ^ { k }  \mathbb { R }$ ，

$$
\mathbf { \underline { { E } } } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \Psi ( q _ { 1 } ( \mathbf { x } ) , \dots , q _ { k } ( \mathbf { x } ) ) ] \approx \underset { \mathbf { z } , G } { \mathbf { E } } [ \Psi ( \tilde { q } _ { 1 } ( \mathbf { z } , G ) , \dots , \tilde { q } _ { k } ( \mathbf { z } , G ) ) ] .
$$

Suppose that given sample access to ${ \tilde { f } } ,$ we would like to identify directions in G-space that are important for $\tilde { f } .$ In fact, by Theorem 4.1, the property of V ensures that $f$ must be suficiently close to an instance that is $( \tau , V )$ -regular. So, if this can be done, then the intuition is that we could potentially make use of the Gaussian comparison techniques to identify the subspace spanned by the $N _ { { \mathbf { u } } _ { i } }$ . For each fixed z, the first two labeled moments in $G$ sufice to find a subspace on which an intersection of halfspaces approximates $\tilde { f } ( { \bf z } , \cdot )$ , as in [DKS18]. Unfortunately, applying this idea directly would require being able to take measurements conditional on z, which can only be approximated using exponential in dim(V ) samples. However, the nice structure of $\tilde { f }$ still makes it possible to achieve this goal by using $O ( k )$ many moments. Specifically, in Lemma 9.2 we show that for every direction v in the G space that has non-trivial influence on ${ \tilde { f } } ,$ there exists a polynomial $p$ of degree 2k such that $\mathbf { E } _ { \mathbf { z } , G } [ \tilde { f } M _ { \mathbf { v } } p ]$ is non-trivially large. This gives a bilinear form with input v and polynomial $p .$ In fact, using integration by parts,

$$
\underset { \mathbf { z } , G } { \mathbf { E } } [ \tilde { f } ( \mathbf { z } , G ) M _ { \mathbf { v } } p ] = \underset { \mathbf { z } , G } { \mathbf { E } } [ p \partial _ { \mathbf { v } } \tilde { f } ] = \sum _ { i = 1 } ^ { k } ( \mathbf { v } \cdot \mathbf { u } _ { i } ) \underset { \mathbf { z } , G } { \mathbf { E } } [ p \frac { \partial \tilde { f } } { \partial \tilde { q } _ { i } } ] ,\tag{1}
$$

we know that such a bilinear form has at most k non-zero singular values. Thus, taking the subspace spanned by the top k singular vectors of the bilinear form, we find all directions in $G$ space that are important to $\tilde { f } .$

However, some additional work is needed to apply these ideas. First, we need to ensure that the Gaussian comparison techniques developed in Section 3 sufice to imply large singular values of the aforementioned bilinear form. This is a particular issue as we only have a comparison for functions of the form $\Psi ( q _ { 1 } , \dots , q _ { k } )$ , not an arbitrary smooth function support over $\mathbb { R } ^ { d }$ . Second, we work with a space S of polynomials of the form $N _ { \mathbf { u } } ( \mathbf { z } , \mathbf { w } )$ and have access only to labeled examples $\left( \mathbf { x } , f ( \mathbf { x } ) \right)$ . Our bilinear form involves a directional derivative along v in the Gaussian replacement space, and it is not clear how to define the corresponding operation on S. We address these two issues separately.

First, the certificate polynomial $p$ that we developed in Lemma 9.2 is a sum of products of the defining polynomials ${ \tilde { q } } _ { i } ,$ not an arbitrary polynomial. Thus, each term in the summation of the bilinear form (1) actually has the desired form of $\Psi ( q _ { 1 } , \dots , q _ { k } )$ . Let $f _ { h }$ be a smooth approximation of $f .$ It sufices to construct a bilinear form $B _ { f }$ such that

$$
B _ { f } ( \mathbf { u } , p ) \approx \sum _ { i = 1 } ^ { k } \langle \mathbf { u } , \mathbf { u } _ { i } \rangle \mathbf { E } \left[ p { \frac { \partial f _ { h } } { \partial q _ { i } } } \right]
$$

uniformly over unit residual coeficient directions u and unit-norm polynomials p of degree at most 2k. Let V<sup>¯</sup> be the span of the top k left singular vectors of $B _ { f }$ . Then $f$ is close to an intersection whose defining polynomials lie in

$$
{ \mathcal { H } } = { \mathcal { P } } _ { \leq 2 } ( V ) + \{ N _ { \mathbf { u } } : \mathbf { u } \in { \bar { V } } \} \ .
$$

It remains to address the second issue of how to build such a $B _ { f }$ . The observation is that under the Gaussian distribution $\mathbf { E _ { x \sim \mathcal { N } ( 0 , I ) } } [ \partial _ { \mathbf { v } } f ] = \mathbf { E _ { x \sim \mathcal { N } ( 0 , I ) } } [ ( \mathbf { v } \cdot \mathbf { x } ) f ]$ , which means that the behavior of $\partial _ { \mathbf { v } } { f }$ can be simulated through standard polynomial multiplication under the Gaussian distribution. The hope is that if we replace v by $N _ { \mathbf { v } } ( \mathbf { z } , \mathbf { w } )$ in the multiplication, then this would satisfy our purpose. In fact, using standard integration by parts,

$$
\operatorname { \mathbf { E } } _ { ( \mathbf { z } , \mathbf { w } ) \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ N _ { \mathbf { v } } ( \mathbf { z } , \mathbf { w } ) f ] = \operatorname { \mathbf { E } } _ { ( \mathbf { z } , \mathbf { w } ) \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ ( \mathbf { b } + \mathbf { L } ^ { \top } \mathbf { z } ) ^ { \top } \nabla _ { \mathbf { w } } f + \mathrm { t r } ( \mathbf { A } \nabla _ { \mathbf { w } } ^ { 2 } f ) ] .
$$

In Lemma 9.4, if we denote by $d _ { \mathbf { v } } ( f ) = ( \mathbf { b } + \mathbf { L } ^ { \top } \mathbf { z } ) ^ { \top } \nabla _ { \mathbf { w } } f + \mathrm { t r } ( \mathbf { A } \nabla _ { \mathbf { w } } ^ { 2 } f )$ , then under our regularity condition, we actually have that $\mathbf { E _ { z , w } } [ p d _ { \mathbf { u } } f ]$ is suficiently close to $\begin{array} { r } { \sum _ { i = 1 } ^ { k } \langle \mathbf { u } , \mathbf { u } _ { i } \rangle \mathbf { E } \left[ p \frac { \partial f _ { h } } { \partial q _ { i } } \right] } \end{array}$ and can be used as $B _ { f }$ . But since we do not have direct access to $f$ and cannot compute its gradients or Hessian, we still need an equivalent representation of $\mathbf { E _ { z , w } } [ p d _ { \mathbf { u } } f ]$ that can be directly computed from samples. That is, we want to find some operator $T _ { \mathbf { u } }$ such that ${ \bf E _ { z , w } } [ f T _ { \bf u } p ] = { \bf E _ { z , w } } [ p d _ { \bf u } f ]$ . Such an operator can be reverse-engineered using integration by parts. In Lemma $9 . 5 ,$ we show that $T _ { \mathbf { v } } p = p N _ { \mathbf { v } } ( \mathbf { z } , \mathbf { w } ) - ( \mathbf { b } + \mathbf { L } ^ { \top } \mathbf { z } + 2 \mathbf { A } \mathbf { w } ) ^ { \top } \nabla _ { \mathbf { w } } p + \mathrm { t r } ( \mathbf { A } \nabla _ { \mathbf { w } } ^ { 2 } p )$ , satisfies this property. Thus, we can find a subspace H of polynomials, where an intersection of k degree-2 PTFs supported on it is suficiently close to $f .$

Finally, we apply the hinge-loss minimization algorithm of Section 8 over the recovered space H. Since

$$
\dim ( { \mathcal { H } } ) \leq O ( \dim ( V ) ^ { 2 } + k ) = 2 ^ { O _ { k } ( \log ^ { 2 } ( 1 / \epsilon ) ) } ~ ,
$$

and we only need polynomials of degree $O _ { k } ( \log ( 1 / \epsilon ) )$ in a basis of $\mathcal { H }$ , the resulting feature space has dimension at most $2 ^ { O _ { k } ( \log ^ { 3 } ( 1 / \epsilon ) ) }$ . Combining the two dimension-reduction stages with hinge-loss minimization gives our final learning algorithm with running time $d ^ { O ( k ) } 2 ^ { O _ { k } ( \log ^ { 3 } ( \breve { 1 / \epsilon } ) ) }$

Extensions and Open Problems. A natural open question is whether our quasipolynomial dependence on $1 / \epsilon$ can be improved to polynomial dependence for every fixed k, yielding a running time of $d ^ { O ( k ) } ( 1 / \epsilon ) ^ { O _ { k } ( 1 ) }$ . It is a plausible conjecture that such an algorithm is possible. A second direction concerns higher-degree PTFs. Our hinge-loss approach extends to intersections of k degree-r PTFs under the standard Gaussian distribution, yielding a $d ^ { O _ { k , r } ( \log ^ { 2 } ( 1 / \epsilon ) ) }$ -time learning algorithm for any fixed degree r. It remains open whether our structural and dimension-reduction machinery can be generalized beyond degree two to obtain a running time polynomial in $d ,$ with an exponent depending only on k and r and not on ϵ. More broadly, it would be interesting to identify natural regularity or complexity conditions on the link function $g$ under which functions of the form $f ( x ) = g ( p _ { 1 } ( x ) , \ldots , p _ { k } ( x ) )$ , with quadratic $p _ { i }$ , can be learned eficiently, and to understand which of our techniques extend beyond the intersection structure.

## 1.3 Related Work

Learning intersections of halfspaces. Understanding the computational complexity of learning intersections of halfspaces is one of the central problems in computational learning theory. In the distribution-free setting, various computational hardness results have been established, such as hardness of proper learning [BR88, KS08], representation-based hardness [She10, She13], cryptographic and Statistical Query hardness on learning a super-constant number of halfspaces [KS07, KS09, Dan16, Tie24]. However, whether an intersection of two halfspaces can be learned in polynomial time in the distribution-free setting remains open. Motivated by this challenge, a line of work has focused on designing learning algorithms under structured distributions, such as learning two homogeneous halfspaces under symmetric distributions [Bau90], low-weighted intersections [KOS04], learning under margin assumptions [KS04] and under factorizable distributions [DMRT25a]. In particular, under the uniform distribution or Gaussian distributions, eficient learning algorithms are known for learning an intersection of a constant number of halfspaces [BK97, Vem10a]. These algorithms can also be made robust against adversarial noise [DKS18]. Under more general logconcave distributions, [Vem10b] gives similar algorithms for learning intersections of homogeneous halfspaces, and more recently [DKM26] gave approximate agnostic learners for functions of constantly many halfspaces under general logconcave marginals.

Multi-index models and beyond. The moment-based subspace recovery method for learning intersections of halfspaces has also inspired a rich line of work on the complexity of learning functions of a small number of linear projections (MIMs) under structured distributions, such as shallow ReLU networks [DKKZ20, CDG<sup>+</sup>23, DK24], multiclass linear classifiers [DMRT25b, DIM26] and general multi-index models [ABAM23, DLB25, DIKZ25, DIKR25]. In particular, [DIKZ25, DIKR25, DKM26] give general frameworks for robustly learning multi-index models under Gaussian and logconcave distributions. We also refer to [BH25] for a recent survey.

Beyond multi-index models, little is known about eficient learning with multiple nonlinear features. Recent works have made initial progress on learning structured hierarchical functions using 3-layer neural networks, including functions of a single polynomial feature [NDL23, WNL24] and polynomial functions of multiple quadratic features [FWNL25]. These guarantees rely on structural and nondegeneracy assumptions on the underlying features and link functions and do not extend to intersections of quadratic PTFs. In contrast, our work develops structural and algorithmic tools for learning intersections of arbitrary quadratic PTFs without additional assumptions.

Regularity and pseudorandomness for PTFs. A substantial line of work has developed regularity, decomposition, and Gaussian approximation techniques for analyzing low-degree polynomial threshold functions; see, e.g., [DHK<sup>+</sup>10, DSTW10, MZ10, DDS14, DS14]. A recurring theme is to separate the structured components of the defining polynomials from suficiently regular components. These techniques have played an important role in deterministic approximate counting and pseudorandomness, including for arbitrary Boolean functions of multiple PTFs [DKN10, DDS14]. In our setting, the defining polynomials are unknown. We must recover, from labeled examples, a space of quadratic features that supports an accurate approximation to the target. Our Gaussian comparison controls the labeled moments used to recover this space.

AI Disclosure. The main framework and algorithmic results of this paper were developed by the authors. In particular, the authors independently established Theorem 7.1 and developed the hinge-loss minimization framework of Section 8. The authors also observed that combining this framework with a suitable dimension-reduction method could improve the running time to $d ^ { O ( k ) } 2 ^ { \mathrm { p o l y } _ { k } ( \log ( 1 / \epsilon ) ) }$

To achieve this improvement, the authors first investigated an approach based on a stronger regularity lemma and showed that it does not hold in the required generality. The authors subsequently interacted with $\mathrm { G P T - 6 }$ Astra to explore alternative dimension-reduction strategies. These interactions led to an initial version of the approach using a slightly weaker notion of regularity, underlying Section 9. The authors then substantially simplified and refined this approach, established its connection to the framework they had previously developed, and obtained the final algorithm and analysis presented in the paper.

During manuscript preparation, GPT-6.1 was also used to improve the clarity of the exposition, check local proof arguments, and identify errors in calculations in earlier drafts.

Organization of the Paper. In Section 2, we list basic notations used throughout the paper. In Section 3, we introduce the notion of $( \tau , V )$ -regularity for quadratic polynomials and develop a Gaussian comparison lemma based on the regularity condition. In Section 4, we show that any intersection of $k$ degree two PTFs who has no high influence directions in $V ^ { \perp }$ must be close to a regular instance. In Section 5, we show our main structural result that a direction with high influence on the target function must have influence on its low degree part. In Section $6 ,$ we develop an eficient algorithm to find these high-influence directions using the structural result. In Section $^ { 7 , }$ we develop a $d ^ { O ( k ) } 2 ^ { \mathrm { p o l y } _ { k } ( 1 / \epsilon ) }$ time learning algorithm based on the developed technical ingredients. In Section $^ { 8 , }$ we develop a robust hinge loss minimization algorithm that will be used to improve the dependence on $1 / \epsilon .$ In Section 9, we develop a two-stage dimension reduction method and complete the main learning algorithm. In Section 10, we provide a matching SQ lower bound.

## 2 Preliminaries and Notations

Notations Let $k \in \mathbb { Z } _ { + }$ , we use $[ k ]$ to denote $\{ 1 , \ldots , k \}$ . We use the convention $\mathrm { s i g n } ( t ) = 1$ for $t \geq 0$ and sign $( t ) = - 1$ for $t < 0$ . When passing between $\{ 0 , 1 \}$ -valued and {±1}-valued functions, we use the correspondence $f \mapsto 2 f - 1$ , with inverse $g \mapsto ( g + 1 ) / 2$ . Thus, in either convention, membership in an intersection is represented by the label 1. The notation $O _ { k } ( \cdot )$ allows the implicit constant to depend on $k .$ . We use $\mathrm { p o l y } _ { k }$ for an unspecified polynomial dependence whose degree and constants may depend on k. In a positive lower bound with argument $t \in ( 0 , 1 )$ , this denotes a quantity of the form $c _ { k } t ^ { C _ { k } }$ for positive constants $c _ { k } , C _ { k }$ depending only on k. For $t \geq 1$ $\mathrm { q u a s i p o l y } _ { k } ( t )$ denotes a bound of the form $\hat { 2 } ^ { ( O _ { k } ( \log ( t ) ) }$

We use small boldface characters for vectors and capital bold characters for matrices and tensors. For $\mathbf { x } \in \mathbb { R } ^ { d }$ and $i \in [ d ] , \mathbf { x } _ { i }$ denotes the i-th coordinate of $\mathbf { x } .$ . Let V be a subspace. We use $V ^ { \perp }$ to denote its orthogonal complement, and use $\operatorname { p r o j } _ { V } ( \mathbf { x } )$ to denote the projection of x onto subspace $V .$ We will also use $\mathbf { x } _ { V }$ for short to denote $\operatorname { p r o j } _ { V } ( \mathbf { x } )$ . We use $\mathbb { S } ^ { d - 1 }$ to denote the unit sphere over $\mathbb { R } ^ { d }$

Unless otherwise specified, function norms are taken under the standard Gaussian distribution on the relevant space. For a function $f : \mathbb { R } ^ { d }  \mathbb { R }$ , we write $\| f \| _ { 2 } = ( \mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ f ( \mathbf { x } ) ^ { 2 } ] ) ^ { 1 / 2 }$ . For vectors, $\lVert \cdot \rVert _ { 2 }$ denotes the Euclidean norm; for matrices, $\left\| \cdot \right\| _ { \mathrm { o p } }$ and $\lVert \cdot \rVert _ { F }$ denote the operator and Frobenius norms, respectively. For convenience, we will also use $\lVert \cdot \rVert$ to denote $\ell _ { 2 } ( L _ { 2 } )$ norm.

## Boolean Functions and Randomized Error

Definition 2.1 (Randomized Error). Let $f : \mathbb { R } ^ { d }  \{ 0 , 1 \}$ be a Boolean function and let $\mathbf { v } \in \mathbb { S } ^ { d - 1 }$ be a unit vector. We define the randomized error of $f$ along direction v to be

$$
\mathrm { R a n d } _ { f } ( \mathbf { v } ) = \underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { \mathbf { v } ^ { \bot } } ) } { \mathbf { E } } \left[ \underset { z , z ^ { \prime } \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { P r } } [ f ( \mathbf { w } + z \mathbf { v } ) \neq f ( \mathbf { w } + z ^ { \prime } \mathbf { v } ) ] \right] ,
$$

where $\mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { \mathbf { v } ^ { \perp } } ) \perp \mathbf { v }$

Intuitively, Rand (v) quantifies the error of f under Gaussian distribution when we rerandomize the direction along v but fix the other directions. We give the following proposition of the randomized error.

Proposition 2.2. Let $f : \mathbb { R } ^ { d }  \{ 0 , 1 \}$ be a Boolean function and let $\mathbf { v } \in \mathbb { S } ^ { d - 1 }$ be a unit vector. Then,

$$
\mathrm { R a n d } _ { f } ( \mathbf { v } ) = 2 \sum _ { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } \left[ \mathrm { V a r } _ { z \sim \mathcal { N } ( 0 , 1 ) } [ f ( \mathbf { w } + z \mathbf { v } ) ] \right] ,
$$

where $\mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) \perp \mathbf { v }$

Proof of Proposition 2.2. We first fix $\mathbf { w } \perp \mathbf { v } .$ . Since $z , z ^ { \prime }$ are independent Gaussian, we have

$$
\begin{array} { r l } & { \quad \underset { z , z ^ { \prime } \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { P r } } [ f ( \mathbf { w } + z \mathbf { v } ) \neq f ( \mathbf { w } + z ^ { \prime } \mathbf { v } ) ] } \\ & { = \underset { z , z ^ { \prime } \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { P r } } [ f ( \mathbf { w } + z \mathbf { v } ) = 1 , f ( \mathbf { w } + z ^ { \prime } \mathbf { v } ) = 0 ] + \underset { z , z ^ { \prime } \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { P r } } [ f ( \mathbf { w } + z \mathbf { v } ) = 0 , f ( \mathbf { w } + z ^ { \prime } \mathbf { v } ) = 1 ] } \\ & { = 2 \underset { z \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { P r } } [ f ( \mathbf { w } + z \mathbf { v } ) = 1 ] \underset { z \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { P r } } [ f ( \mathbf { w } + z \mathbf { v } ) = 0 ] } \\ & { = 2 \mathrm { V a r } _ { z \sim \mathcal { N } ( 0 , 1 ) } [ f ( \mathbf { w } + z \mathbf { v } ) ] . } \end{array}
$$

Since w is independent of $z ,$ by taking the expectation over $\mathbf { w } _ { \mathrm { : } }$ , we have

$$
\mathrm { R a n d } _ { f } ( \mathbf { v } ) = 2 \sum _ { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } \left[ \mathrm { V a r } _ { z \sim \mathcal { N } ( 0 , 1 ) } [ f ( \mathbf { w } + z \mathbf { v } ) ] \right] .
$$

Definition 2.3 (Layer Decomposition). Let $f :  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ be a function. Let $\mathbf { v } \in \mathbb { S } ^ { d - 1 }$ be a unit vector and let w ⊥ v. We define the one dimensional function $f _ { \mathbf { w } } : \mathbb { R }  \mathbb { R }$ as $f _ { \mathbf { w } } ( s ) = f ( s \mathbf { v } + \mathbf { w } )$ . For a polynomial $q : \mathbb { R } ^ { d }  \mathbb { R }$ , we also use $q ^ { ( \mathbf { w } ) }$ to denote $q _ { \mathbf { w } }$

Hermite polynomials. For each integer $r \geq 0$ , let $h _ { r }$ denote the normalized probabilists’ Hermite polynomial of degree $r ,$ with positive leading coeficient. In particular, $h _ { 0 } ( t ) = 1 , h _ { 1 } ( t ) = t$ , and $h _ { 2 } ( t ) = ( t ^ { 2 } - 1 ) / \sqrt { 2 }$ . These polynomials satisfy

$$
\mathbf { \underset { Z \sim \mathcal { N } ( 0 , 1 ) } { E } } [ h _ { r } ( Z ) h _ { s } ( Z ) ] = \mathbb { 1 } \{ r = s \} , \qquad h _ { r } ^ { \prime } ( t ) = \sqrt { r } h _ { r - 1 } ( t ) \quad ( r \geq 1 ) .
$$

For a multi-index $\alpha \in \mathbb { Z } _ { \geq 0 } ^ { d } .$ write $\begin{array} { r } { | \alpha | = \sum _ { j = 1 } ^ { d } \alpha _ { j } } \end{array}$ and define $\begin{array} { r } { H _ { \alpha } ( \mathbf { x } ) = \prod _ { j = 1 } ^ { d } h _ { \alpha _ { j } } ( x _ { j } ) } \end{array}$ . The polynomials $H _ { \alpha }$ form an orthonormal basis for the space of square-integrable functions under $\mathcal { N } ( 0 , \mathbf { I } _ { d } )$ , and $H _ { \alpha }$ has total degree |α|.

Definition 2.4 (Hermite Projection). Let $k \in \mathbb N$ , and $V \subseteq \mathbb { R } ^ { d }$ be a subspace. We denote by $P _ { \leq k } ( V )$ the space of polynomials supported on V with degree no more than k. Let $f :  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ be a function. We denote by $f _ { \le k }$ to be the projection of f onto the space spanned by Hermite polynomials with degree no larger than k.

Gaussian polynomial inequalities. For $q \geq 1$ , we write $\| P \| _ { q } = ( \mathbf { E _ { x \sim \mathcal { N } ( 0 , I ) } } [ | P ( \mathbf { x } ) | ^ { q } ] ) ^ { 1 / q }$ . We will use the following Gaussian polynomial inequalities throughout.

Fact 2.5 (Gaussian hypercontractivity, see, e.g., [O’D14, Chapter 11]). Let $P : \mathbb { R } ^ { d }  \mathbb { R }$ be a polynomial of degree at most r. For every real $q \geq 2 , \ \| P \| _ { q } \leq ( q - 1 ) ^ { r / 2 } \| P \| _ { 2 }$ . In particular, there are universal constants $c , C > 0$ such that if $\deg ( P ) \leq 2$ , then, for every $t \geq 1$ ， $\mathrm { \bf P r } _ { { \bf x } \sim \mathcal { N } ( 0 , { \bf I } ) } [ | P ( { \bf x } ) | >$ $t \left\| P \right\| _ { 2 } ] \le C \exp ( - c t )$

Fact 2.6 (Gaussian anti-concentration [CW01, Theorem 8]). There is a universal constant $C > 0$ such that the following holds. Let $r \geq 1$ be an integer and let $P : \mathbb { R } ^ { d }  \mathbb { R }$ be a nonzero polynomial of degree at most r. For every $t > 0$ and $a \in \{ 1 , 2 \}$ $\begin{array} { r } { \mathbf { P r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ | P ( \mathbf { x } ) | \leq t \| P \| _ { a } ] \leq C r t ^ { 1 / r } } \end{array}$

Fact 2.7 (Gaussian product-norm bounds). For every pair of nonnegative integers $r , s ,$ , there are constants $c _ { r , s } , C _ { r , s } > 0$ , depending only on $r , s ,$ such that $c _ { r , s } \left\| P \right\| _ { 2 } \left\| Q \right\| _ { 2 } \leq \left\| P Q \right\| _ { 2 } \leq C _ { r , s } \left\| P \right\| _ { 2 } \left\| Q \right\| _ { 2 }$ for all polynomials $P , Q : \mathbb { R } ^ { d }  \mathbb { R }$ of degrees at most $r , s ,$ respectively.

Proof of Fact 2.7. By Cauchy–Schwarz and Fact 2.5, $\begin{array} { r } { \| P Q \| _ { 2 } \leq \| P \| _ { 4 } \| Q \| _ { 4 } \leq 3 ^ { ( r + s ) / 2 } \| P \| _ { 2 } \| Q \| _ { 2 } . } \end{array}$

For the lower bound, the claim is immediate if either polynomial is constant. Otherwise, Fact 2.6 provides constants $a _ { r } , a _ { s } > 0$ , depending only on $r , s ,$ such that

$$
\mathbf { P r } [ | P | < a _ { r } \| P \| _ { 2 } ] \leq 1 / 4 , \qquad \mathbf { P r } [ | Q | < a _ { s } \| Q \| _ { 2 } ] \leq 1 / 4 .
$$

With probability at least $1 / 2$ , both complementary inequalities hold. Hence $\left\| P Q \right\| _ { 2 } ^ { 2 } \geq a _ { r } ^ { 2 } a _ { s } ^ { 2 } \left\| P \right\| _ { 2 } ^ { 2 } \left\| Q \right\| _ { 2 } ^ { 2 } / 2$ which proves the lower bound. □

## 3 Gaussian Approximation via Regularity

In this section, we develop Gaussian comparison tools that allow us to reduce the learning of regular intersections of degree-2 PTFs to learning intersections of halfspaces under Gaussian distributions. The starting observation is that every degree-2 polynomial can be represented as a linear function of the degree-1 and degree-2 Hermite features. Thus, replacing the lifted feature vector by a standard Gaussian vector transforms an intersection of degree-2 PTFs into an intersection of halfspaces, for which powerful dimension-reduction techniques are available. The main challenge is to justify this replacement in a way that preserves the information needed for learning. In particular, we need to show that the relevant low-degree moments are preserved suficiently well to recover a useful subspace, and that a subspace capturing the Gaussian surrogate also yields a good approximation to the original target.

Our comparison results are developed in a more general setting that will also be useful in later sections. Specifically, we consider $( \tau , V )$ -regular polynomials, which may have unrestricted dependence on a fixed subspace V but satisfy regularity conditions in the orthogonal directions. The Gaussian replacement then preserves the original coordinates in V and acts only on the remaining polynomial contributions. The fully regular case corresponds to $V = \{ 0 \}$ . We first establish the moment-comparison and subspace-transfer guarantees in this general setting. We then specialize to the fully regular case and combine these results with the Gaussian halfspace recovery algorithm of [DKS18] to obtain an eficient learner for regular intersections of degree-2 PTFs.

Definition 3.1 ((τ, V)-regularity). Let $\tau \in ( 0 , 1 )$ , let $V \subseteq \mathbb { R } ^ { d }$ be a subspace, and let $p : \mathbb { R } ^ { d } $ R be a polynomial of degree at most two. For a unit vector $\textbf { v } \in \ \mathbb { R } ^ { d }$ , define its directional influence by $\begin{array} { r } { \operatorname* { i n f } _ { p } ( \mathbf { v } ) \ : = \ \mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } \left[ ( \mathbf { v } \cdot \nabla p ( \mathbf { x } ) ) ^ { 2 } \right] } \end{array}$ We say that $\textit { p } \ i s \ ( \tau , V )$ -regular if in $\mathrm { f } _ { p } ( \mathbf { v } ) \ \leq$ $\tau ^ { 2 } \mathbf { V a r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ p ( \mathbf { x } ) ]$ for every unit vector $\mathbf { v } \in V ^ { \perp }$ . We call p τ-regular when it is $( \tau , \{ 0 \} ) – r e g u l a r$

Regularity ensures that no input direction has a large influence relative to the total variance. In the fully regular case, an orthogonal change of coordinates expresses the centered polynomial as a sum of independent terms, each contributing only a small fraction of the polynomial’s variance. This gives a central-limit intuition for approximating the polynomial by a Gaussian with the same mean and variance.

Notation and Gaussian replacement. Throughout this section, for $\mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ , we use $\mathbf { z } = \mathrm { p r o j } _ { V } \mathbf { x }$ and $\mathbf { w } = \mathrm { p r o j } _ { V ^ { \perp } } \mathbf { x }$ to denote its components in V and $V ^ { \perp }$ , respectively, so that ${ \mathbf { x } } = { \mathbf { z } } + { \mathbf { w } }$ . Polynomial $L _ { 2 }$ norms are taken under the corresponding standard Gaussian distribution. Let $\Phi _ { 2 } : \mathbb { R } ^ { d }  \mathbb { R } ^ { N }$ , where $N = d ( d { + } 3 ) / 2$ , be the vector of normalized Gaussian Hermite polynomials of degrees one and two. These features have mean zero and satisfy $\mathbf { E } [ \Phi _ { 2 } ( \mathbf { x } ) \Phi _ { 2 } ( \mathbf { x } ) ^ { \top } ] = \mathbf { I } _ { N }$

Every polynomial $p _ { i }$ of degree at most two has a decomposition

$$
p _ { i } ( \mathbf { x } ) = q _ { i } ( \mathbf { z } ) + \mathbf { b } _ { i } ^ { \top } \mathbf { w } + \mathbf { z } ^ { \top } \mathbf { L } _ { i } \mathbf { w } + \mathbf { w } ^ { \top } \mathbf { A } _ { i } \mathbf { w } - \mathrm { t r } ( \mathbf { A } _ { i } ) ,
$$

where $\mathbf { b } _ { i } \in V ^ { \perp } , \mathbf { L } _ { i } : V ^ { \perp } \to V$ is linear, and $\mathbf { A } _ { i } : V ^ { \perp } \to V ^ { \perp }$ is symmetric. Here $q _ { i } ( \mathbf { z } ) = \mathbf { E } [ p _ { i } ( \mathbf { x } ) \mid \mathbf { z } ]$ so the remaining terms have conditional mean zero given z. The following fact translates regularity into bounds on these coeficients.

Fact 3.2 (Coeficient bounds from regularity). Suppose $p _ { i } \ i s \ ( \tau , V )  – r e g u l a r _ { ; }$ , and let $\sigma _ { i } ^ { 2 } : = \mathbf { V a r } _ { \mathbf { x } } [ p _ { i } ( \mathbf { x } ) ]$ Then the coeficients in the decomposition above satisfy $\| \mathbf { b } _ { i } \| _ { 2 } \leq \tau \sigma _ { i } , \| \mathbf { L } _ { i } \| _ { \mathrm { o p } } \leq \tau \sigma _ { i }$ , and $\| \mathbf { A } _ { i } \| _ { \mathrm { o p } } \leq$ $\tau \sigma _ { i } / 2$ . Moreover, $\left\| \mathbf { b } _ { i } \right\| _ { 2 } ^ { 2 } + \left\| \mathbf { L } _ { i } \right\| _ { F } ^ { 2 } + 2 \left\| \mathbf { A } _ { i } \right\| _ { F } ^ { 2 } \leq \sigma _ { i } ^ { 2 }$

Proof of Fact 3.2. For every unit vector $\mathbf { v } \in V ^ { \perp }$ , diferentiating the decomposition and using the independence of z and w gives $\mathbf { E } _ { \mathbf { x } } \left[ ( \mathbf { v } \cdot \nabla p _ { i } ( \mathbf { x } ) ) ^ { 2 } \right] = ( \mathbf { b } _ { i } ^ { \top } \mathbf { v } ) ^ { 2 } + \| \bar { \mathbf { L } _ { i } } \mathbf { v } \| _ { 2 } ^ { 2 } + 4 \left\| \bar { \mathbf { A } _ { i } } \mathbf { v } \right\| _ { 2 } ^ { 2 } \leq \tau ^ { 2 } \sigma _ { i } ^ { 2 }$ , where the inequality follows from regularity. Taking suprema over unit vectors $\mathbf { v } \in V ^ { \perp }$ yields the coeficient bounds. Finally, Gaussian orthogonality gives $\sigma _ { i } ^ { 2 } = \mathbf { V a r } _ { \mathbf { z } } [ q _ { i } ( \mathbf { z } ) ] + \lVert \mathbf { b } _ { i } \rVert _ { 2 } ^ { 2 } + \lVert \mathbf { L } _ { i } \rVert _ { F } ^ { 2 } + 2 \lVert \mathbf { A } _ { i } \rVert _ { F } ^ { 2 }$ , proving the last bound. □

Let $S _ { V } : = \left\{ \pmb \theta \in \mathbb { R } ^ { N } : \mathbf { E } \left[ \pmb \theta \cdot \Phi _ { 2 } ( \mathbf { x } ) \mid \mathbf { z } \right] = 0 \right\}$ be the subspace of residual coeficient vectors. We can therefore write $p _ { i } ( \mathbf { x } ) = q _ { i } ( \mathbf { z } ) + \pmb { \theta } _ { i } \cdot \pmb { \Phi } _ { 2 } ( \mathbf { x } )$ for a unique $\pmb \theta _ { i } \in S _ { V }$ . Throughout this section, we use $\mathbf { G } \sim \mathcal { N } ( 0 , \mathbf { I } _ { N } )$ , independent of $\mathbf { x } ,$ for the Gaussian replacement vector. Define the Gaussian replacement of $p _ { i }$ by $\widetilde { p } _ { i } ( \mathbf { z } , \mathbf { G } ) : = q _ { i } ( \mathbf { z } ) + \pmb { \theta } _ { i } \cdot \mathbf { G }$ . The same vector G is used for all i, preserving the pairwise covariances of the residual contributions. For an intersection $f ( \mathbf { x } ) = \mathbb { 1 } \{ p _ { i } ( \mathbf { x } ) \geq 0$ for all $i \in$ [k]}, write $\widetilde { f } ( \mathbf { z } , \mathbf { G } ) = \mathbb { 1 } \{ \widetilde { p } _ { i } ( \mathbf { z } , \mathbf { G } ) \geq 0$ for all $i \in [ k ] \}$ for its Gaussian surrogate. When $V = \{ 0 \}$ , each $q _ { i }$ is constant, and $\widetilde { f }$ is an intersection of k halfspaces under a standard Gaussian distribution.

## 3.1 Gaussian Moment Comparison

We now derive the Hermite moment comparison, treating $m = 0 , 1 , 2$ in turn.

Lemma 3.3 (Gaussian Hermite moment comparison). Let $\tau \in ( 0 , 1 )$ , and let $p _ { 1 } , \ldots , p _ { k }$ have the decompositions and Gaussian replacements defined above. For every $i \in [ k ]$ , assume that $\| p _ { i } \| _ { 2 } = 1$ and that the coeficient norms $\| \mathbf { b } _ { i } \| _ { 2 } , \| \mathbf { L } _ { i } \| _ { \mathrm { o p } } ,$ , and $\| \mathbf { A } _ { i } \| _ { \mathrm { o p } }$ are all at most τ. Let $\psi : \mathbb { R } ^ { k } $ R be a $C ^ { 4 }$ function $\begin{array} { r } { D _ { r } ( \psi ) : = \operatorname* { m a x } _ { i _ { 1 } , \dots , i _ { r } \in [ k ] } \| \partial _ { i _ { 1 } } \cdot \cdot \cdot \partial _ { i _ { r } } \dot { \psi } \| _ { \infty } < \infty \mathrm { ~ } f o \dot { r } \ r \in [ 4 ] } \end{array}$

For an integrable function $F ~ o f \mathbf { x }$ , define $\mathcal { P } _ { V } F : = F - \mathbf { E } [ F \mid \mathbf { z } ] + \mathbf { E } [ F ]$ . Then, for every unit vector $\mathbf { v } \in S _ { V }$ and every $m \in \{ 0 , 1 , 2 \}$ ，

$$
\left| \mathbf { E } \big [ \psi ( p _ { 1 } ( \mathbf { x } ) , \dots , p _ { k } ( \mathbf { x } ) ) \mathcal { P } _ { V } [ h _ { m } ( \mathbf { v } \cdot \Phi _ { 2 } ( \mathbf { x } ) ) ] \big ] - \mathbf { \Pi } _ { \mathbf { z } , \mathbf { G } } ^ { \mathbf { E } } \big [ \psi ( \widetilde { p } _ { 1 } ( \mathbf { z } , \mathbf { G } ) , \dots , \widetilde { p } _ { k } ( \mathbf { z } , \mathbf { G } ) ) h _ { m } ( \mathbf { v } \cdot \mathbf { G } ) \big ] \right| \leq \epsilon _ { m } ,
$$

where $h _ { m }$ is the normalized univariate Hermite polynomial of degree $m ,$ and $\epsilon _ { 0 } = O \left( k ^ { 2 } \tau D _ { 2 } ( \psi ) \right)$ $\begin{array} { r } { \epsilon _ { 1 } = O \big ( k ^ { 3 } \tau \operatorname* { m a x } _ { 1 \leq r \leq 3 } D _ { r } ( \psi ) \big ) , \epsilon _ { 2 } = O \big ( k ^ { 4 } \tau \operatorname* { m a x } _ { 1 \leq r \leq 4 } D _ { r } ( \psi ) \big ) } \end{array}$

The role of the comparison is easiest to see in the fully regular case $V = \{ 0 \}$ . The Gaussian surrogate is then an intersection of halfspaces, whose first two labeled moments provide the information needed for subspace recovery. To compare these moments, we work with a smooth function ψ of the defining polynomials and consider its correlations with the Hermite weights $h _ { m } ( \mathbf { v } \cdot \Phi _ { 2 } ( \mathbf { x } ) )$ for $m \in \{ 0 , 1 , 2 \}$ . These capture the expectation and the directional first and centered second moments, respectively. Our goal is to show that regularity preserves these quantities under Gaussian replacement, with error controlled by τ and the derivatives of $\psi$

We first establish the smooth comparison below, corresponding to the case without a Hermite weight. We gradually replace the residual polynomial contributions by a Gaussian vector with the same covariance, keeping the variables in V unchanged. After Gaussian integration by parts, the matching covariance terms cancel, and regularity controls the remaining error. To compare the first and second moments, we use integration by parts again to express the weighted expectations in terms of derivatives of $\psi ,$ up to similarly small errors, and apply the smooth comparison to these derivatives. When V is nontrivial, the conditional centering in the statement removes the contribution of the squared probe that depends only on the retained variables. The probe itself need not be regular, since the small factors controlling the errors come from the defining polynomials.

Lemma 3.4 (Gaussian comparison). Let $\tau , V , p _ { 1 } , \dotsc , p _ { k }$ satisfy the assumptions of Lemma 3.3, and use the same notation. Let $F : \mathbb { R } ^ { k }  \mathbb { R }$ be a $C ^ { 2 }$ function such that $\begin{array} { r } { D _ { 2 } ( F ) : = \operatorname* { m a x } _ { i , j \in [ k ] } \| \partial _ { i } \partial _ { j } F \| _ { \infty } < } \end{array}$ ∞. Then

$$
\Big | \operatorname { \mathbf { E } } _ { \mathbf { \theta } _ { \mathbf { x } } } \big [ F ( p _ { 1 } ( \mathbf { x } ) , \dots , p _ { k } ( \mathbf { x } ) ) \big ] - \operatorname { \mathbf { E } } _ { \mathbf { \theta } _ { \mathbf { x } , \mathbf { z } } } \big [ F ( \widetilde { p } _ { 1 } ( \mathbf { x } _ { V } , \mathbf { z } ) , \dots , \widetilde { p } _ { k } ( \mathbf { x } _ { V } , \mathbf { z } ) ) \big ] \Big | \leq O ( k ^ { 2 } \tau D _ { 2 } ( F ) ) .
$$

Proof of Lemma $\ 3 . 4 \cdot$ To prove Lemma 3.4, we keep the variables in V unchanged and gradually replace the residual terms by a Gaussian vector with the same covariance. Throughout, we use the assumptions and notation of Lemma 3.3. For $t \in [ 0 , 1 ]$ , define $p _ { t , i } = q _ { i } ( \mathbf { x } _ { V } ) + \sqrt { t } \left( p _ { i } ( \mathbf { x } ) - q _ { i } ( \mathbf { x } _ { V } ) \right)$ + ${ \sqrt { 1 - t } } \pmb { \theta } _ { i } \cdot \mathbf { z }$ , where z is an independent standard Gaussian vector, and set $\mathbf { P } _ { t } = ( p _ { t , 1 } , \dots , p _ { t , k } )$ and $H ( t ) = \mathbf { E } _ { \mathbf { x } , \mathbf { z } } [ F ( \mathbf { P } _ { t } ) ]$ . The two expectations in the lemma are $H ( 1 )$ and $H ( 0 )$ . Therefore, to bound $| H ( 0 ) - H ( 1 ) |$ , it sufices for us to bound $H ^ { \prime } ( t )$ for all $t \in ( 0 , 1 )$ . We will apply Gaussian integral by parts on $H ^ { \prime } ( t )$ and then show that the expression we get can be bounded by regularity.

For convenience, write $\mathbf { u } = \mathbf { x } _ { V } , \mathbf { y } = \mathbf { x } _ { \perp }$ , and $\mathbf { a } _ { i } ( \mathbf { u } ) = \mathbf { b } _ { i } + \mathbf { L } _ { i } ^ { \top } \mathbf { u } , R _ { i } = p _ { i } ( \mathbf { x } ) - q _ { i } ( \mathbf { u } )$ and $G _ { i } = \pmb \theta _ { i } \cdot \mathbf z$ Then $R _ { i } = { \bf { a } } _ { i } ( { \bf { u } } ) ^ { \top } { \bf { y } } + { \bf { y } } ^ { \top } { \bf { A } } _ { i } { \bf { y } } - \mathrm { t r } ( { \bf { A } } _ { i } )$ . Define

$$
\begin{array} { r l } & { \quad \quad \Sigma _ { i j } : = \mathbf { E } [ R _ { i } R _ { j } ] = \pmb { \theta } _ { i } \cdot \pmb { \theta } _ { j } = \underbrace { \mathbf { E } [ \mathbf { a } _ { i } ( \mathbf { u } ) ^ { \top } \mathbf { a } _ { j } ( \mathbf { u } ) ] } _ { \mathbf { u } } + 2 \mathrm { t r } ( \mathbf { A } _ { i } \mathbf { A } _ { j } ) , } \\ & { \quad \Gamma _ { i j } ( \mathbf { u } , \mathbf { y } ) : = \left( \mathbf { a } _ { i } ( \mathbf { u } ) + \mathbf { A } _ { i } \mathbf { y } \right) ^ { \top } \left( \mathbf { a } _ { j } ( \mathbf { u } ) + 2 \mathbf { A } _ { j } \mathbf { y } \right) . } \end{array}
$$

Thus Σ is the common covariance matrix of the centered vectors $( R _ { i } ) _ { i = 1 } ^ { k }$ and $( G _ { i } ) _ { i = 1 } ^ { k }$ . Unless otherwise indicated, expectations below are over the independent Gaussian vectors $\mathbf { u } , \mathbf { y } , \mathbf { z }$

Fact 3.5 (Integration-by-parts identity). For every $\begin{array} { r } { t \in ( 0 , 1 ) , H ^ { \prime } ( t ) = \frac { 1 } { 2 } \sum _ { i , j = 1 } ^ { k } \mathbf { E } \left[ \left( \Gamma _ { i j } - \Sigma _ { i j } \right) \partial _ { i } \partial _ { j } F ( \mathbf { P } _ { t } ) \right] } \end{array}$

Proof of Fact 3.5. Diferentiating with respect to t gives

$$
H ^ { \prime } ( t ) = { \frac { 1 } { 2 { \sqrt { t } } } \sum _ { i } \mathbf { E } [ R _ { i } \partial _ { i } F ( \mathbf { P } _ { t } ) ] - { \frac { 1 } { 2 { \sqrt { 1 - t } } } \sum _ { i } \mathbf { E } [ G _ { i } \partial _ { i } F ( \mathbf { P } _ { t } ) ] } } .
$$

For fixed u, Gaussian integration by parts gives $\mathbf { E } _ { \mathbf { y } } [ R _ { i } g ( \mathbf { y } ) ] = \mathbf { E } _ { \mathbf { y } } \left[ ( \mathbf { a } _ { i } ( \mathbf { u } ) + \mathbf { A } _ { i } \mathbf { y } ) ^ { \top } \nabla _ { \mathbf { y } } g ( \mathbf { y } ) \right]$ . For the quadratic terms, this follows by applying the identity $\mathbf { E } _ { \mathbf { y } } [ y _ { r } v ( \mathbf { y } ) ] { \stackrel { - } { = } } \mathbf { E } _ { \mathbf { y } } [ \partial _ { r } v ( \mathbf { y } ) ]$ with $v ( \mathbf { y } ) = y _ { s } g ( \mathbf { y } )$ the resulting constant term is canceled by $\operatorname { t r } ( \mathbf { A } _ { i } )$

Holding u, z fixed, apply this identity with $g ( \mathbf { y } ) = \partial _ { i } F ( \mathbf { P } _ { t } )$ . Since $\nabla _ { \mathbf { y } } p _ { t , j } = \sqrt { t } \left( \mathbf { a } _ { j } ( \mathbf { u } ) + 2 \mathbf { A } _ { j } \mathbf { y } \right)$ the chain rule gives $\begin{array} { r } { { \bf E } [ R _ { i } \partial _ { i } F ( { \bf P } _ { t } ) ] = \sqrt { t } \sum _ { i } { \bf E } [ \Gamma _ { i j } \partial _ { i } \partial _ { j } F ( { \bf P } _ { t } ) ] } \end{array}$ . Similarly, holding $\mathbf { u } , \mathbf { y }$ fixed and integrating by parts in z, we obtain $\begin{array} { r } { { \bf E } [ G _ { i } \partial _ { i } F ( { \bf P } _ { t } ) ] = \sqrt { 1 - t } \sum _ { j } \Sigma _ { i j } \ { \bf E } [ \partial _ { i } \partial _ { j } F ( { \bf P } _ { t } ) ] } \end{array}$ . Substituting these identities into $H ^ { \prime } ( t )$ proves the claim.

These operations are justified by Gaussian moment bounds, since bounded second derivatives imply that F and its first derivatives have at most quadratic and linear growth, respectively.

Fact 3.6 (Regularity bound). For every $i , j \in [ k ] , \mathbf { E } [ | \Gamma _ { i j } - \Sigma _ { i j } | ] \leq \left. \Gamma _ { i j } - \Sigma _ { i j } \right. _ { L _ { 2 } } \leq C \tau$ , where $C > 0$ is a universal constant.

Proof. Since $\mathbf { E } [ R _ { i } \ | \ \mathbf { u } ] = 0$ and $\mathbf { E } [ p _ { i } ^ { 2 } ] = 1$ , we have $\| \mathbf { b } _ { i } \| _ { 2 } ^ { 2 } + \| \mathbf { L } _ { i } \| _ { F } ^ { 2 } + 2 \| \mathbf { A } _ { i } \| _ { F } ^ { 2 } = \mathbf { E } [ R _ { i } ^ { 2 } ] \leq 1$ . In particular, $\mathbf { E _ { u } } [ \left\| \mathbf { a } _ { i } ( \mathbf { u } ) \right\| _ { 2 } ^ { 2 } ] = \| \mathbf { b } _ { i } \| _ { 2 } ^ { 2 } + \left\| \dot { \mathbf { L } _ { i } } \right\| _ { F } ^ { 2 } \leq 1$ . Expanding the discrepancy gives

$$
\Gamma _ { i j } - \Sigma _ { i j } = \mathbf { a } _ { i } ( \mathbf { u } ) ^ { \top } \mathbf { a } _ { j } ( \mathbf { u } ) - \mathbf { \phi } _ { \mathbf { u } } ^ { \mathbf { E } } [ \mathbf { a } _ { i } ( \mathbf { u } ) ^ { \top } \mathbf { a } _ { j } ( \mathbf { u } ) ] + \left( 2 \mathbf { A } _ { j } \mathbf { a } _ { i } ( \mathbf { u } ) + \mathbf { A } _ { i } \mathbf { a } _ { j } ( \mathbf { u } ) \right) ^ { \top } \mathbf { y } + 2 \big ( \mathbf { y } ^ { \top } \mathbf { A } _ { i } \mathbf { A } _ { j } \mathbf { y } - \operatorname { t r } ( \mathbf { A } _ { i } \mathbf { A } _ { j } ) \big ) .
$$

For a standard Gaussian vector Z and any square matrix $M , \left\| \mathbf Z ^ { \top } M \mathbf Z - \operatorname { t r } ( M ) \right\| _ { L ^ { \infty } } \leq \sqrt 2 \left\| M \right\| _ { F }$ Using this identity and independence of u and y, we obtain

$$
\begin{array} { r l } & { \quad \quad \left\| \mathbf { a } _ { i } ^ { \top } \mathbf { a } _ { j } - \frac { \mathbf { E } [ \mathbf { a } _ { i } ^ { \top } \mathbf { a } _ { j } ] } { \mathbf { u } } \right\| _ { L _ { 2 } } \leq \left\| \mathbf { L } _ { i } \mathbf { b } _ { j } + \mathbf { L } _ { j } \mathbf { b } _ { i } \right\| _ { 2 } + \sqrt { 2 } \left\| \mathbf { L } _ { i } \mathbf { L } _ { j } ^ { \top } \right\| _ { F } \leq C \tau , } \\ & { \quad \quad \left\| \left( 2 \mathbf { A } _ { j } \mathbf { a } _ { i } + \mathbf { A } _ { i } \mathbf { a } _ { j } \right) ^ { \top } \mathbf { y } \right\| _ { L _ { 2 } } \leq 2 \left\| \mathbf { A } _ { j } \right\| _ { \mathrm { o p } } + \left\| \mathbf { A } _ { i } \right\| _ { \mathrm { o p } } \leq 3 \tau , } \\ & { 2 \left\| \mathbf { y } ^ { \top } \mathbf { A } _ { i } \mathbf { A } _ { j } \mathbf { y } - \mathrm { t r } ( \mathbf { A } _ { i } \mathbf { A } _ { j } ) \right\| _ { L _ { 2 } } \leq 2 \sqrt { 2 } \left\| \mathbf { A } _ { i } \mathbf { A } _ { j } \right\| _ { F } \leq C \tau . } \end{array}
$$

The final bounds use $\| \mathbf { L } _ { i } \| _ { \mathrm { o p } } , \| \mathbf { A } _ { i } \| _ { \mathrm { o p } } \leq \tau$ , the coeficient bounds above, and $\lVert M N \rVert _ { F } \leq \lVert M \rVert _ { \mathrm { o p } } \lVert N \rVert _ { F }$ The triangle inequality and Cauchy–Schwarz prove the claim. □

By Facts 3.5 and 3.6, $\begin{array} { r } { | H ^ { \prime } ( t ) | \leq \frac { 1 } { 2 } D _ { 2 } ( F ) \sum _ { i , j = 1 } ^ { k } \mathbf { E } [ | \Gamma _ { i j } - \Sigma _ { i j } | ] \leq C k ^ { 2 } \tau D _ { 2 } ( F ) } \end{array}$ . Integrating over $t \in [ 0 , 1 ]$ , using continuity at the endpoints, gives $| H ( 1 ) - H ( 0 ) | \le C k ^ { 2 } \tau D _ { 2 } ( F )$ , as desired. □

We now prove Lemma 3.3 using the unweighted comparison established above. The case $m = 0$ follows directly from Lemma 3.4. For $m = 1 , 2$ , Gaussian integration by parts removes the Hermite weights, leaving derivatives of $\psi$ and remainder terms controlled by the coeficient bounds. We then apply Lemma 3.4 to these derivatives to obtain the desired comparison.

Proof of Lemma 3.3. For convenience, write $\mathbf { P } = ( p _ { 1 } ( \mathbf { x } ) , \ldots , p _ { k } ( \mathbf { x } ) ) , \widetilde { \mathbf { P } } = ( \widetilde { p } _ { 1 } ( \mathbf { x } _ { V } , \mathbf { z } ) , \ldots , \widetilde { p } _ { k } ( \mathbf { x } _ { V } , \mathbf { z } ) )$ and let $\mathbf { u } = \mathbf { x } _ { V }$ and $\mathbf { y } = \mathbf { x } _ { \perp }$

Case $m = 0$ . There is no Hermite weight in this case, so we apply Lemma 3.4 directly to $\psi$

Since $h _ { 0 } = 1$ and $\mathcal { P } _ { V } [ 1 ] = 1$ , Lemma 3.4 gives $\begin{array} { r } { \left| \mathbf { E } [ \psi ( \mathbf { P } ) ] - \mathbf { E } [ \psi ( \widetilde { \mathbf { P } } ) ] \right| \leq C k ^ { 2 } \tau D _ { 2 } ( \psi ) } \end{array}$

Case $m = 1$ We first use integration by parts to remove the Hermite factor, obtaining first derivatives of $\psi$ and a remainder controlled by regularity. We then apply Lemma 3.4 to these first derivatives.

Write $r = { \mathbf v } \cdot \Phi _ { 2 } ( { \mathbf x } ) , \widetilde r = { \mathbf v } \cdot { \mathbf z }$ and $c _ { i } = \mathbf { v } \cdot { \boldsymbol { \theta } } _ { i }$ . Since $\mathbf { v } \in S _ { V }$ , we can write

$$
r = \mathbf { a } _ { \mathbf { v } } ( \mathbf { u } ) ^ { \top } \mathbf { y } + \mathbf { y } ^ { \top } \mathbf { B } _ { \mathbf { v } } \mathbf { y } - \operatorname { t r } ( \mathbf { B } _ { \mathbf { v } } ) , \qquad \mathbf { a } _ { \mathbf { v } } ( \mathbf { u } ) = \mathbf { b } _ { \mathbf { v } } + \mathbf { L } _ { \mathbf { v } } ^ { \top } \mathbf { u } ,
$$

where $\mathbf { B _ { v } }$ is symmetric. In particular, ${ \bf E } [ r \mid { \bf u } ] = 0$ , so $\mathcal { P } _ { V } [ h _ { 1 } ( r ) ] = r .$ . The normalization $\| \mathbf { v } \| _ { 2 } = 1$ gives $\| \mathbf { b } _ { \mathbf { v } } \| _ { 2 } ^ { 2 } + \| \mathbf { L } _ { \mathbf { v } } \| _ { F } ^ { 2 } + 2 \| \mathbf { B } _ { \mathbf { v } } \| _ { F } ^ { 2 } = \mathbf { E } [ r ^ { 2 } ] { \stackrel { . } { = } } 1$ . Consequently, $\mathbf { E _ { u } } [ \lVert \mathbf { a _ { v } } ( \mathbf { u } ) \rVert _ { 2 } ^ { 2 } ] \leq 1 , \ \lvert c _ { i } \rvert \leq \lVert \pmb { \theta _ { i } } \rVert _ { 2 } ^ { - } \leq 1$ Also write $\mathbf { a } _ { i } ( \mathbf { u } ) = \mathbf { b } _ { i } + \mathbf { L } _ { i } ^ { \top } \mathbf { u } ,$ , so that $\nabla _ { \mathbf { y } } p _ { i } = \mathbf { a } _ { i } ( \mathbf { u } ) + 2 \mathbf { A } _ { i } \mathbf { y }$

We establish the first-moment expansion for a general smooth function $F : \mathbb { R } ^ { k }  \mathbb { R }$ , so that we can also use it in the case $m = 2$ . For fixed u, Gaussian integration by parts gives $\mathbf { E } _ { \mathbf { y } } [ r g ( \mathbf { y } ) ] = \mathbf { E } _ { \mathbf { y } } [ ( \mathbf { a } _ { \mathbf { v } } ( \mathbf { u } ) + \mathbf { B } _ { \mathbf { v } } \mathbf { y } ) ^ { \top } \nabla _ { \mathbf { y } } g ( \mathbf { y } ) ]$ . Applying this identity to $g ( \mathbf { y } ) = F ( \mathbf { P } )$ , define

$$
\begin{array} { r } { \Gamma _ { \mathbf { v } i } = ( \mathbf { a } _ { \mathbf { v } } ( \mathbf { u } ) + \mathbf { B } _ { \mathbf { v } } \mathbf { y } ) ^ { \top } ( \mathbf { a } _ { i } ( \mathbf { u } ) + 2 \mathbf { A } _ { i } \mathbf { y } ) , \qquad \Delta _ { i } = \Gamma _ { \mathbf { v } i } - c _ { i } . } \end{array}
$$

Since $\mathbf { E } [ \Gamma _ { \mathbf { v } i } ] = c _ { i }$ , we obtain

$$
\mathbf { E } [ r F ( \mathbf { P } ) ] = \sum _ { i = 1 } ^ { k } c _ { i } \mathbf { \delta E } [ \partial _ { i } F ( \mathbf { P } ) ] + \mathcal { R } _ { 1 } ( F ) ,
$$

where $\begin{array} { r } { \mathcal { R } _ { 1 } ( F ) = \sum _ { i = 1 } ^ { k } \mathbf { E } [ \Delta _ { i } \partial _ { i } F ( \mathbf { P } ) ] } \end{array}$

To bound the remainder, expand

$$
\Delta _ { i } = \mathbf { a } _ { \mathbf { v } } ( \mathbf { u } ) ^ { \top } \mathbf { a } _ { i } ( \mathbf { u } ) - \underline { { \mathbf { F } } } [ \mathbf { a } _ { \mathbf { v } } ( \mathbf { u } ) ^ { \top } \mathbf { a } _ { i } ( \mathbf { u } ) ] + \big ( 2 \mathbf { A } _ { i } \mathbf { a } _ { \mathbf { v } } ( \mathbf { u } ) + \mathbf { B } _ { \mathbf { v } } \mathbf { a } _ { i } ( \mathbf { u } ) \big ) ^ { \top } \mathbf { y } + 2 \big ( \mathbf { y } ^ { \top } \mathbf { B } _ { \mathbf { v } } \mathbf { A } _ { i } \mathbf { y } - \mathrm { t r } ( \mathbf { B } _ { \mathbf { v } } \mathbf { A } _ { i } ) \big ) .
$$

Using the same Gaussian second-moment estimates as in the proof of Lemma 3.4,

$$
\begin{array} { r } { \| \Delta _ { i } \| _ { L _ { 2 } } \leq \| \mathbf { L } _ { i } \mathbf { b } _ { \mathbf { v } } + \mathbf { L } _ { \mathbf { v } } \mathbf { b } _ { i } \| _ { 2 } + \sqrt { 2 } \left\| \mathbf { L } _ { \mathbf { v } } \mathbf { L } _ { i } ^ { \top } \right\| _ { F } + 2 \left\| \mathbf { A } _ { i } \right\| _ { \mathrm { o p } } + \left\| \mathbf { B } _ { \mathbf { v } } \mathbf { b } _ { i } \right\| _ { 2 } + \left\| \mathbf { B } _ { \mathbf { v } } \mathbf { L } _ { i } ^ { \top } \right\| _ { F } + 2 \sqrt { 2 } \left\| \mathbf { B } _ { \mathbf { v } } \mathbf { A } _ { i } \right\| _ { F } \leq C \tau . } \end{array}
$$

The last inequality follows from the normalization of v and the assumptions $\| \mathbf { b } _ { i } \| _ { 2 } , \| \mathbf { L } _ { i } \| _ { \mathrm { o p } } , \| \mathbf { A } _ { i } \| _ { \mathrm { o p } } \leq$ τ. Therefore, $| \mathcal { R } _ { 1 } ( F ) | \le C k \tau D _ { 1 } ( F )$

On the Gaussian side, integration by parts in z gives $\begin{array} { r } { \mathbf { E } [ \widetilde { r } \psi ( \widetilde { \mathbf { P } } ) ] = \sum _ { i = 1 } ^ { k } c _ { i } \mathbf { E } [ \partial _ { i } \psi ( \widetilde { \mathbf { P } } ) ] } \end{array}$ ]. Combining this identity with the first-moment expansion,

$$
\left| \mathbf { E } [ r \psi ( \mathbf { P } ) ] - \mathbf { E } [ \widetilde { r } \psi ( \widetilde { \mathbf { P } } ) ] \right| \leq \sum _ { i = 1 } ^ { k } \left| c _ { i } \right| \left| \mathbf { E } [ \partial _ { i } \psi ( \mathbf { P } ) ] - \mathbf { E } [ \partial _ { i } \psi ( \widetilde { \mathbf { P } } ) ] \right| + C k \tau D _ { 1 } ( \psi ) .
$$

Applying Lemma 3.4 to each $\partial _ { i } \psi$ , and using $| c _ { i } | \leq 1$ , bounds this by

$$
C k ^ { 3 } \tau D _ { 3 } ( \psi ) + C k \tau D _ { 1 } ( \psi ) \leq C k ^ { 3 } \tau \operatorname* { m a x } _ { 1 \leq j \leq 3 } D _ { j } ( \psi ) .
$$

Case $m = 2 .$ . Conditional centering removes the u-dependent variance of the probe. We then apply integration by parts and reuse the first-moment expansion to reduce the comparison to second derivatives of $\psi ,$ , which are handled by Lemma 3.4.

Let $s ( \mathbf { u } ) = \mathbf { E } _ { \mathbf { y } } [ r ^ { 2 } ] = \left. \mathbf { a } _ { \mathbf { v } } ( \mathbf { u } ) \right. _ { 2 } ^ { 2 } + 2 \left. \mathbf { B } _ { \mathbf { v } } \right. _ { F } ^ { 2 }$ . Since $\begin{array} { r } { { \bf { E } } [ r ^ { 2 } ] = 1 , \mathcal { P } _ { V } [ h _ { 2 } ( r ) ] = \frac { r ^ { 2 } - s ( { \bf { u } } ) } { \sqrt { 2 } } } \end{array}$ . Apply the integration-by-parts identity from the previous case to $g ( \mathbf { y } ) = r \psi ( \mathbf { P } )$ . Since

$$
( \mathbf { a } _ { \mathbf { v } } + \mathbf { B } _ { \mathbf { v } } \mathbf { y } ) ^ { \top } \nabla _ { \mathbf { y } } r = s ( \mathbf { u } ) + 3 \mathbf { a } _ { \mathbf { v } } ^ { \top } \mathbf { B } _ { \mathbf { v } } \mathbf { y } + 2 \big ( \mathbf { y } ^ { \top } \mathbf { B } _ { \mathbf { v } } ^ { 2 } \mathbf { y } - \mathrm { t r } ( \mathbf { B } _ { \mathbf { v } } ^ { 2 } ) \big ) ,
$$

we obtain

$$
\mathbf { E } [ ( r ^ { 2 } - s ( \mathbf { u } ) ) \psi ( \mathbf { P } ) ] = \sum _ { i } \mathbf { E } [ r \Gamma _ { \mathbf { v } i } \partial _ { i } \psi ( \mathbf { P } ) ] + \mathbf { E } [ J _ { \mathbf { v } } \psi ( \mathbf { P } ) ] ,\tag{2}
$$

where $J _ { \mathbf { v } } = 3 \mathbf { a } _ { \mathbf { v } } ( \mathbf { u } ) ^ { \top } \mathbf { B } _ { \mathbf { v } } \mathbf { y } + 2 \big ( \mathbf { y } ^ { \top } \mathbf { B } _ { \mathbf { v } } ^ { 2 } \mathbf { y } - \mathrm { t r } ( \mathbf { B } _ { \mathbf { v } } ^ { 2 } ) \big )$

For the first term in (2), write $\Gamma _ { \mathbf { v } i } = c _ { i } + \Delta _ { i }$ and apply the first-moment expansion to $F = \partial _ { i } \psi$ This gives

$$
\sum _ { i } \mathbf { E } [ r \Gamma _ { \mathbf { v } i } \partial _ { i } \psi ( \mathbf { P } ) ] = \sum _ { i , j } c _ { i } c _ { j } \mathbf { E } [ \partial _ { i } \partial _ { j } \psi ( \mathbf { P } ) ] + \sum _ { i } c _ { i } \mathcal { R } _ { 1 } ( \partial _ { i } \psi ) + \sum _ { i } \mathbf { E } [ r \Delta _ { i } \partial _ { i } \psi ( \mathbf { P } ) ] .
$$

By the bound on $\mathcal { R } _ { 1 }$ above and $\begin{array} { r } { | c _ { i } | \leq 1 , | \sum _ { i } c _ { i } \mathcal { R } _ { 1 } ( \partial _ { i } \psi ) | \leq C k ^ { 2 } \tau D _ { 2 } ( \psi ) } \end{array}$ . Also, Cauchy–Schwarz gives

$$
\left| \sum _ { i } \mathbf { E } [ r \Delta _ { i } \partial _ { i } \psi ( \mathbf { P } ) ] \right| \leq D _ { 1 } ( \psi ) \sum _ { i } \| r \| _ { L _ { 2 } } \| \Delta _ { i } \| _ { L _ { 2 } } \leq C k \tau D _ { 1 } ( \psi ) .
$$

It remains to bound the second term in (2). Another application of Gaussian integration by parts gives

$$
\mathbf { E } [ J _ { \mathbf { v } } \psi ( \mathbf { P } ) ] = \sum _ { i } \mathbf { E } \left[ ( 3 \mathbf { a } _ { \mathbf { v } } + 2 \mathbf { B } _ { \mathbf { v } } \mathbf { y } ) ^ { \top } \mathbf { B } _ { \mathbf { v } } \nabla _ { \mathbf { y } } p _ { i } \partial _ { i } \psi ( \mathbf { P } ) \right] .
$$

The normalization of v implies $\mathbf { E } [ \| 3 \mathbf { a } _ { \mathbf { v } } + 2 \mathbf { B } _ { \mathbf { v } } \mathbf { y } \| _ { 2 } ^ { 2 } ] \leq C$ , while the coeficient bounds give

$$
\mathbf { E } [ \| \mathbf { B } _ { \mathbf { v } } \nabla _ { \mathbf { y } } p _ { i } \| _ { 2 } ^ { 2 } ] = \| \mathbf { B } _ { \mathbf { v } } \mathbf { b } _ { i } \| _ { 2 } ^ { 2 } + \left\| \mathbf { B } _ { \mathbf { v } } \mathbf { L } _ { i } ^ { \top } \right\| _ { F } ^ { 2 } + 4 \left\| \mathbf { B } _ { \mathbf { v } } \mathbf { A } _ { i } \right\| _ { F } ^ { 2 } \leq C \tau ^ { 2 } .
$$

Therefore, Cauchy–Schwarz yields $| \mathbf { E } [ J _ { \mathbf { v } } \psi ( \mathbf { P } ) ] | \leq C k \tau D _ { 1 } ( \psi )$

Combining these estimates establishes the second-moment expansion

$$
{ \bf E } [ ( r ^ { 2 } - s ( { \bf u } ) ) \psi ( { \bf P } ) ] = \sum _ { i , j } c _ { i } c _ { j } \mathrm { \bf ~ E } [ \partial _ { i } \partial _ { j } \psi ( { \bf P } ) ] + { \mathcal R } _ { 2 } ( \psi ) ,
$$

with $| \mathcal { R } _ { 2 } ( \psi ) | \leq C \big ( k \tau D _ { 1 } ( \psi ) + k ^ { 2 } \tau D _ { 2 } ( \psi ) \big )$

On the Gaussian side, applying integration by parts twice gives

$$
\mathbf { E } [ \psi ( \widetilde { \mathbf { P } } ) h _ { 2 } ( \widetilde { r } ) ] = \frac { 1 } { \sqrt { 2 } } \sum _ { i , j } c _ { i } c _ { j } \ \mathbf { E } [ \partial _ { i } \partial _ { j } \psi ( \widetilde { \mathbf { P } } ) ] .
$$

Applying Lemma 3.4 to each $\partial _ { i } \partial _ { j } \psi$ , and using $| c _ { i } | \leq 1$ , bounds the diference between the two second Hermite moments by

$$
C \big ( k \tau D _ { 1 } ( \psi ) + k ^ { 2 } \tau D _ { 2 } ( \psi ) + k ^ { 4 } \tau D _ { 4 } ( \psi ) \big ) \leq C k ^ { 4 } \tau \operatorname* { m a x } _ { 1 \leq j \leq 4 } D _ { j } ( \psi ) .
$$

This proves the claimed bound for $m = 2$

## 3.2 Gaussian Subspace Transfer

Having compared the moments of the two models, we next show how a subspace recovered for the Gaussian surrogate can be used to approximate the original target. The following lemma shows that, if $\widetilde { f } ( \mathbf { z } , \mathbf { G } )$ has small randomized error in every unit direction of $S _ { V }$ orthogonal to a subspace $W \subseteq S _ { V }$ , then $f ( \mathbf { x } )$ is close to an intersection whose residual coeficient vectors lie in $W$ , while allowing unrestricted degree-2 dependence on z. The approximation error is controlled by τ and the directional randomized error. This provides the approximation guarantee needed to restrict the hypothesis search after recovering W in the Gaussian model.

Lemma 3.7 (Gaussian subspace transfer). Let $\tau , V , p _ { 1 } , \dots , p _ { k }$ satisfy the assumptions of Lemma ${ \it 3 . 3 , }$ and use the same notation. Define $f ( \mathbf { x } ) = \mathbf { 1 } \{ p _ { 1 } ( \mathbf { x } ) \geq 0 , \ldots , p _ { k } ( \mathbf { x } ) \geq 0 \}$ , and $\widetilde { f } ( \mathbf { x } _ { V } , \mathbf { z } ) = \mathbf { 1 } \{ q _ { i } ( \mathbf { x } _ { V } ) +$ $\pmb \theta _ { i } \cdot \mathbf z \geq 0$ for all $i \in [ k ] \}$ . Let $\delta \in [ 0 , 1 ]$ , and let $W \subseteq S _ { V }$ be a subspace such that Rand ${ \boldsymbol { \widetilde { f } } } ^ { ( \mathbf { w } ) } \leq \delta$ for every unit vector $\mathbf { w } \in W ^ { \perp }$ where $W ^ { \perp }$ is taken within $S _ { V }$

Then there exists an intersection $g ( \mathbf { x } ) = \mathbf { 1 } \{ g _ { 1 } ( \mathbf { x } ) \geq 0 , \ldots , g _ { k } ( \mathbf { x } ) \geq 0 \}$ such that $g _ { i } \in P _ { < 2 } ( V ) \circledast$ $\{ \mathbf { u } \cdot \Phi _ { 2 } ( \mathbf { x } ) : \mathbf { u } \in W \}$ for every $i \in [ k ]$ , where $P _ { \leq 2 } ( V )$ denotes the space of polynomials of degree at most two depending only on $\mathbf { x } _ { V }$ , such that $\bar { \mathbf { P r } } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ f ( \mathbf { x } ) \neq g ( \mathbf { x } ) ] \leq C k ^ { C } ( \delta + \tau ) ^ { 1 / 2 0 }$ , for a universal constant $C > 1$

Proof of Lemma 3.7. Throughout this proof, $C _ { k }$ denotes a positive quantity bounded by $C k ^ { C }$ for a universal constant $C > 1 ;$ its value may increase from line to line. We prove $\mathbf { P r } [ f ( \mathbf { x } ) \neq g ( \mathbf { x } ) ] \leq$ $C _ { k } ( \delta + \tau ) ^ { 1 / 2 0 }$ . The hypothesis says that, in the Gaussian model, W captures almost all the label information beyond $\mathbf { x } _ { V }$ . Our goal is to obtain an approximation whose residual coeficients lie in W and then use the same coeficients in the original model. Direct projection into W need not preserve the coeficient bounds required for Gaussian comparison. We therefore first find a controlled coeficient span, retain the part of that span almost contained in $W$ , and perform Gaussian comparison before making the final projection.

We use two standard estimates. First, if $P , Q$ are degree-at-most-two polynomials of a standard Gaussian input, $\| \cal { P } \| _ { 2 } \geq 1 / 2$ , and $\| P - Q \| _ { 2 } \leq s$ , then

$$
\mathbf { P r } [ \mathbf { 1 } \{ P \ge 0 \} \neq \mathbf { 1 } \{ Q \ge 0 \} ] \le C s ^ { 2 / 5 } .\tag{3}
$$

Indeed, degree-two Gaussian anticoncentration and Markov’s inequality give, for $t > 0$

$$
\mathbf { P r } [ \mathbf { 1 } \{ P \geq 0 \} \neq \mathbf { 1 } \{ Q \geq 0 \} ] \leq \mathbf { P r } [ | P | \leq t ] + \mathbf { P r } [ | P - Q | > t ] \leq C \sqrt { t } + \frac { s ^ { 2 } } { t ^ { 2 } } .
$$

Taking $t = s ^ { 4 / 5 }$ proves the claim for $s > 0 ;$ the case $s = 0$ is immediate.

Second, Lemma 3.4 implies the following comparison for disagreement probabilities. For a joint tuple of 2k defining polynomials whose $L _ { 2 }$ norms lie in $[ 1 / 2 , 1 ]$ and whose residual coeficient bounds are at most $\rho ,$ the disagreement probability of the two corresponding intersections difers between the original and Gaussian models by at most $C _ { k } \rho ^ { 1 / 5 }$ . The claim is immediate when $\rho = 0$ , and follows from the trivial bound 1 when $\rho \ge 1 / 2$ , after increasing $C _ { k }$ . Suppose $0 < \rho < 1 / 2$ . First normalize each polynomial by its positive $L _ { 2 }$ norm, which increases its coeficient bounds by at most a factor of two. Smooth each threshold on a scale t. If $A , B \in [ 0 , 1 ]$ are the resulting smooth approximations to the two intersections, then $A + B - 2 A B$ smoothly approximates their disagreement indicator and has second partial derivatives bounded by $C t ^ { - 2 }$ for a universal constant $C ,$ , since the undiferentiated factors lie in [0, 1]. Anticoncentration and a union bound give smoothing error in each model at most $C k { \sqrt { t } } .$ , since the Gaussian replacement preserves each polynomial’s $L _ { 2 }$ norm and is itself a degree-at-most-two polynomial of Gaussian variables. Applying Lemma 3.4 to the 2k normalized polynomials bounds the smooth comparison error by $C k ^ { 2 } \rho t ^ { - 2 }$ . Taking $t = \rho ^ { 2 / 5 }$ gives a total bound of $C k ^ { 2 } \rho ^ { 1 / 5 }$

Write $\mathrm { p r o j } _ { U }$ for orthogonal projection onto a coeficient subspace U. Let $\epsilon = \delta + \tau$ , and fix a suficiently small universal constant $c \in ( 0 , 2 ^ { - 8 } )$ . We may assume that $\epsilon \leq c k ^ { - 2 }$ . Otherwise, $1 \leq c ^ { - 1 / 2 0 } k ^ { \bar { 1 } / 1 0 } \epsilon ^ { 1 / 2 0 }$ , so the trivial error bound 1 gives the required conclusion with a polynomial prefactor in k. Choose parameters $a , b \in ( 0 , 1 / 2 )$ , to be specified at the end.

Step 1: Spectrally truncate the residual coeficient vectors. Although each $\pmb \theta _ { i }$ has small residual coeficients, a unit vector in their span can require a large linear combination of the $\pmb \theta _ { i } \mathrm { ^ { * } s }$ when these vectors nearly cancel. We remove the directions responsible for this poor conditioning. These directions have little total weight in the defining polynomials, so removing them changes the target only slightly.

Fact 3.8 (Spectral truncation). Let $\begin{array} { r } { \Theta : \mathbb { R } ^ { k }  S _ { V } , \Theta \alpha = \sum _ { i = 1 } ^ { k } \alpha _ { i } \pmb { \theta } _ { i } , M = \Theta \Theta ^ { \top } } \end{array}$ , and let T be the eigenspace of M with eigenvalues at least $a ^ { 2 }$ . Then $\| \pmb { \theta } _ { i } - \mathrm { p r o j } _ { T } \pmb { \theta } _ { i } \| _ { 2 } \leq a$ for every $i \in [ k ]$ . Moreover, for every unit vector $\mathbf { v } \in T$ , writing

$$
\begin{array} { r } { \mathbf v \cdot \Phi _ { 2 } ( \mathbf x ) = \mathbf b _ { \mathbf v } ^ { \top } \mathbf x _ { \perp } + \mathbf x _ { V } ^ { \top } \mathbf L _ { \mathbf v } \mathbf x _ { \perp } + \mathbf x _ { \perp } ^ { \top } \mathbf A _ { \mathbf v } \mathbf x _ { \perp } - \mathrm { t r } ( \mathbf A _ { \mathbf v } ) , } \end{array}
$$

we have

$$
\operatorname* { m a x } \left\{ \left\| \mathbf { b _ { v } } \right\| _ { 2 } , \left\| \mathbf { L _ { v } } \right\| _ { \mathrm { o p } } , \left\| \mathbf { A _ { v } } \right\| _ { \mathrm { o p } } \right\} \leq \frac { \sqrt { k } \tau } { a } .
$$

Proof of Fact 3.8. By the definition of $T , \| ( \mathbf { I } - \mathrm { p r o j } _ { T } ) \Theta \| _ { \mathrm { o p } } \leq a$ . Applying this operator to each coordinate vector of $\mathbb { R } ^ { k }$ proves the first claim.

Every unit $\mathbf { v } \in T$ has a representation $\mathbf { v } = \Theta \alpha$ with $\| \alpha \| _ { 2 } \le a ^ { - 1 }$ , obtained from the singular value decomposition of Θ restricted to the retained singular directions. The maps taking a residual coeficient vector to its $\mathbf { b } , \mathbf { L } , \mathbf { A }$ coeficients are linear. The triangle inequality and Cauchy–Schwarz therefore give

$$
\operatorname* { m a x } \left\{ \left\| \mathbf { b } _ { \mathbf { v } } \right\| _ { 2 } , \left\| \mathbf { L } _ { \mathbf { v } } \right\| _ { \mathrm { o p } } , \left\| \mathbf { A } _ { \mathbf { v } } \right\| _ { \mathrm { o p } } \right\} \leq \tau \sum _ { i } \left| \alpha _ { i } \right| \leq \sqrt { k } \tau \left\| \alpha \right\| _ { 2 } \leq \frac { \sqrt { k } \tau } { a } .
$$

Set

$$
\pmb { \theta } _ { i } ^ { 0 } = \mathrm { p r o j } _ { T } \pmb { \theta } _ { i } , \qquad p _ { i } ^ { 0 } ( \mathbf { x } ) = q _ { i } ( \mathbf { x } _ { V } ) + \pmb { \theta } _ { i } ^ { 0 } \cdot \Phi _ { 2 } ( \mathbf { x } ) .
$$

Let $f ^ { 0 } ( { \bf x } ) = { \bf 1 } \{ p _ { i } ^ { 0 } ( { \bf x } ) \geq 0$ for all $i \in [ k ] \}$ . Let $\widetilde { p } _ { i } ^ { 0 } ( \mathbf { u } , \mathbf { z } ) = q _ { i } ( \mathbf { u } ) + \pmb { \theta } _ { i } ^ { 0 } \cdot \mathbf { z }$ and let $\widetilde { f } ^ { 0 }$ be the corresponding Gaussian intersection.

Orthogonality of the Hermite features gives $\left\| p _ { i } - p _ { i } ^ { 0 } \right\| _ { 2 } \leq a _ { : }$ and the same bound holds in the Gaussian model. Consequently, (3) and a union bound imply

$$
\mathbf { P r } [ f \neq f ^ { 0 } ] \leq C _ { k } a ^ { 2 / 5 } , \qquad \mathbf { P r } [ \widetilde { f } \neq \widetilde { f } ^ { 0 } ] \leq C _ { k } a ^ { 2 / 5 } .\tag{4}
$$

Also, since the residual features have conditional mean zero given $\mathbf { x } _ { V }$

$$
n _ { i } ^ { 2 } : = \left\| p _ { i } ^ { 0 } \right\| _ { 2 } ^ { 2 } = \left\| q _ { i } \right\| _ { 2 } ^ { 2 } + \left\| \pmb { \theta } _ { i } ^ { 0 } \right\| _ { 2 } ^ { 2 } = 1 - \left\| \pmb { \theta } _ { i } - \pmb { \theta } _ { i } ^ { 0 } \right\| _ { 2 } ^ { 2 } \geq 1 - a ^ { 2 } .\tag{5}
$$

Write $\mathbf { u } \ = \ \mathbf { x } _ { V }$ , and let $\mathbf { z } , \mathbf { z } ^ { \prime } \sim \mathcal { N } ( 0 , \mathbf { I } _ { S _ { V } } )$ be independent of each other and of u. Let $U =$ span $\{ \mathrm { p r o j } _ { W ^ { \perp } } \pmb \theta _ { i } : i \in [ k ] \} \subsetneq W ^ { \perp }$ , and let $\mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { r }$ be an orthonormal basis of $U _ { : }$ , where $r =$ dim $U \leq k$ . The function $\dot { f } ( \mathbf { u } , \mathbf { z } )$ depends on the $W ^ { \perp }$ component of z only through its projection onto $U$ . Resampling the coordinates along $\mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { r }$ one at a time and applying a union bound therefore gives

$$
\mathbf { P r } \left[ \widetilde f ( \mathbf { u } , \mathbf { z } ) \neq \widetilde f ( \mathbf { u } , \mathrm { p r o j } _ { W } \mathbf { z } + \mathrm { p r o j } _ { W ^ { \perp } } \mathbf { z } ^ { \prime } ) \right] \leq \sum _ { j = 1 } ^ { r } \mathrm { R a n d } _ { \widetilde f } ( \mathbf { v } _ { j } ) \leq r \delta \leq k \delta .
$$

Both z and $\mathrm { p r o j } _ { W } { \bf z } + \mathrm { p r o j } _ { W ^ { \perp } } { \bf z } ^ { \prime }$ are standard Gaussian. Applying (4) at both endpoints and using another union bound gives

$$
\mathbf { P r } \left[ \widetilde { f } ^ { 0 } ( \mathbf { u } , \mathbf { z } ) \neq \widetilde { f } ^ { 0 } ( \mathbf { u } , \mathrm { p r o j } _ { W } \mathbf { z } + \mathrm { p r o j } _ { W ^ { \perp } } \mathbf { z } ^ { \prime } ) \right] \leq \eta , \qquad \eta : = k \delta + C _ { k } a ^ { 2 / 5 } .\tag{6}
$$

Here and below, constants denoted by $C _ { k }$ may increase from line to line.

Step 2: Retain the directions of T almost contained in W. A direction of $T$ that lies substantially outside W is substantially disturbed by rerandomizing $W ^ { \perp }$ . Since this rerandomization rarely changes the label, the target has little dependence on such directions. The following fact makes this statement quantitative.

Fact 3.9 (Gaussian resampling onto a nearby subspace). Let $T , W$ be subspaces of a finite-dimensional Euclidean space S. Let $\mathbf { z } , \mathbf { z } ^ { \prime }$ be independent standard Gaussians on S, and let u be independent of them. Suppose $H ( \mathbf { u } , \mathbf { z } ) \in \{ 0 , 1 \}$ depends on z only through proj z, and

$$
\mathbf { P r } [ H ( \mathbf { u } , \mathbf { z } ) \neq H ( \mathbf { u } , \mathrm { p r o j } _ { W } \mathbf { z } + \mathrm { p r o j } _ { W ^ { \perp } } \mathbf { z } ^ { \prime } ) ] \leq \eta .
$$

Let $E \subseteq T$ be the eigenspace of $K = \mathrm { p r o j } _ { T } \mathrm { p r o j } _ { W } \mathrm { p r o j } _ { T } : T  T$ with eigenvalues at least $1 - b ^ { 2 }$ . Then every $\mathbf { e } \in E$ satisfies $\left\| \mathbf { e } - \mathrm { p r o j } _ { W } \mathbf { e } \right\| _ { 2 } \leq b \left\| \mathbf { e } \right\| _ { 2 }$ . Moreover, $i f \mathbf { c }$ is an independent standard Gaussian on $T \cap E ^ { \perp }$ , then

$$
\mathbf { P r } [ H ( \mathbf { u } , \mathbf { z } ) \neq H ( \mathbf { u } , \mathrm { p r o j } _ { E } \mathbf { z } + \mathbf { c } ) ] \leq \frac { \eta } { b ^ { 2 } } .
$$

Proof of Fact 3.9. Choose an orthonormal eigenbasis $\mathbf { e } _ { 1 } , \ldots , \mathbf { e } _ { r }$ of $K$ , with eigenvalues $\lambda _ { j } \in [ 0 , 1 ]$ In this basis,

$$
\mathrm { p r o j } _ { T } \mathbf { z } \qquad \mathrm { a n d } \qquad \mathrm { p r o j } _ { T } ( \mathrm { p r o j } _ { W } \mathbf { z } + \mathrm { p r o j } _ { W ^ { \perp } } \mathbf { z } ^ { \prime } )
$$

are standard Gaussian vectors with cross-covariance $\mathrm { d i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { r } )$

For fixed u, expand H in the normalized multivariate Hermite basis:

$$
H \left( \mathbf { u } , \sum _ { j = 1 } ^ { r } s _ { j } \mathbf { e } _ { j } \right) = \sum _ { \alpha \in \mathbb { N } ^ { r } } \widehat { H } _ { \mathbf { u } } ( \alpha ) H _ { \alpha } ( \mathbf { s } ) .
$$

Because H is Boolean, disagreement probability equals expected squared diference. Orthogonality and the Gaussian Hermite correlation identity give

$$
\mathbf { P r } [ H ( \mathbf { u } , \mathbf { z } ) \neq H ( \mathbf { u } , \mathbf { p r o j } _ { W } \mathbf { z } + \mathbf { p r o j } _ { W } \mathbf { \bot z } ^ { \prime } ) ] = 2 \sum _ { \alpha } \left( 1 - \prod _ { j = 1 } ^ { r } \lambda _ { j } ^ { \alpha _ { j } } \right) \mathbf { E } [ \widehat { H } _ { \mathbf { u } } ( \alpha ) ^ { 2 } ] .\tag{7}
$$

Let $J = \{ j : \lambda _ { j } < 1 - b ^ { 2 } \}$ , so that $T \cap E ^ { \perp }$ is spanned by the $\mathbf { e } _ { j }$ with $j \in J$ . Independently replacing precisely these coordinates gives disagreement probability $\begin{array} { r } { 2 \sum _ { \alpha : \exists j \in J , \ \alpha _ { i } > 0 } \mathbf { E _ { u } } [ \widehat { H } _ { \mathbf { u } } ( \alpha ) ^ { 2 } ] } \end{array}$ . For each term in this sum, $\begin{array} { r } { 1 - \prod _ { j } \lambda _ { j } ^ { \alpha _ { j } } \ge b ^ { 2 } } \end{array}$ . Comparison with (7) proves the resampling bound. Finally, for $\mathbf { e } \in E .$

$$
\left\| \mathbf { e } - \mathrm { p r o j } _ { W } \mathbf { e } \right\| _ { 2 } ^ { 2 } = \left\| \mathbf { e } \right\| _ { 2 } ^ { 2 } - \left. \mathbf { e } , K \mathbf { e } \right. \leq b ^ { 2 } \left\| \mathbf { e } \right\| _ { 2 } ^ { 2 } ,
$$

which proves the first claim. If $T = \{ 0 \}$ , both conclusions are immediate.

Apply Fact 3.9 to $H = \widetilde f ^ { 0 }$ , using (6). Write $B = T \cap E ^ { \perp }$ and

$$
\pmb { \theta } _ { i } ^ { E } = \mathrm { p r o j } _ { E } \pmb { \theta } _ { i } ^ { 0 } , \qquad \pmb { \theta } _ { i } ^ { B } = \mathrm { p r o j } _ { B } \pmb { \theta } _ { i } ^ { 0 } .
$$

For an independent standard Gaussian c on $B ,$ define $\widetilde { r } _ { i } ( \mathbf { u } , \mathbf { z } , \mathbf { c } ) = q _ { i } ( \mathbf { u } ) + \pmb { \theta } _ { i } ^ { B } \cdot \mathbf { c } + \pmb { \theta } _ { i } ^ { E } \cdot \mathbf { z } .$ , and let $\widetilde { h } ( \mathbf { u } , \mathbf { z } , \mathbf { c } ) = \mathbf { 1 } \{ \widetilde { r } _ { i } ( \mathbf { u } , \mathbf { z } , \mathbf { c } ) \geq 0$ for all $i \in [ k ] \}$ . The fact gives

$$
\mathbf { P r } [ \widetilde { f } ^ { 0 } ( \mathbf { u } , \mathbf { z } ) \neq \widetilde { h } ( \mathbf { u } , \mathbf { z } , \mathbf { c } ) ] \leq \frac { \eta } { b ^ { 2 } } .\tag{8}
$$

Thus we have removed the directions of T outside E at a controlled cost. The remaining residual directions are both controlled by Fact 3.8 and close to $W$

Step 3: Transfer while the coeficients still lie in the controlled span. We now return to the original Hermite features, defining $r _ { i } ( \mathbf { x } , \mathbf { c } ) = q _ { i } ( \mathbf { x } _ { V } ) + \pmb { \theta } _ { i } ^ { B } \cdot \mathbf { c } + \pmb { \theta } _ { i } ^ { E } \cdot \Phi _ { 2 } ( \mathbf { x } )$ , and $h ( \mathbf { x } , \mathbf { c } ) = \mathbf { 1 } \{ r _ { i } ( \mathbf { x } , \mathbf { c } ) \geq$ 0 for all $i \in [ k ] \}$ . It is important to keep c random during this step. Fixing c prematurely could make some defining polynomials nearly zero in $L _ { 2 }$ , preventing a uniform anticoncentration bound. Keeping c random preserves their norms:

$$
\begin{array} { c } { \left\| r _ { i } \right\| _ { L _ { 2 } ( \mathbf { x } , \mathbf { c } ) } ^ { 2 } = \left\| q _ { i } \right\| _ { 2 } ^ { 2 } + \left\| \pmb { \theta } _ { i } ^ { B } \right\| _ { 2 } ^ { 2 } + \left\| \pmb { \theta } _ { i } ^ { E } \right\| _ { 2 } ^ { 2 } } \\ { = n _ { i } ^ { 2 } \in [ 1 - a ^ { 2 } , 1 ] . } \end{array}\tag{9}
$$

Treat the coordinates of c as additional retained Gaussian input coordinates, enlarging V to $V ^ { \prime } = V \oplus B _ { \mathbf { c } }$ , where $B _ { \mathbf { c } }$ denotes this added coordinate space. Then $q _ { i } ( { \bf x } _ { V } ) + \pmb { \theta } _ { i } ^ { B } \cdot { \bf c }$ is a degree-atmost-two polynomial on that enlarged subspace.

The residual coeficient vectors of the joint tuple $( p _ { 1 } ^ { 0 } , \ldots , p _ { k } ^ { 0 } , r _ { 1 } , \ldots , r _ { k } )$ are $\theta _ { i } ^ { 0 }$ and $\pmb { \theta } _ { i } ^ { E }$ , respectively. They lie in T and have Euclidean norm at most one. By Fact 3.8, all their coeficient bounds are at most $\sqrt { k } \tau / a$ . Embedding these residuals in the enlarged Hermite feature space introduces only zero coeficients on the additional features involving c.

By (5) and (9), the $L _ { 2 }$ norms of all 2k polynomials lie in $[ 1 / 2 , 1 ]$ . Applying the error comparison bound to their joint disagreement event, and then using (8), gives

$$
\mathbf { P r } [ f ^ { 0 } ( \mathbf { x } ) \neq h ( \mathbf { x } , \mathbf { c } ) ] \leq { \frac { \eta } { b ^ { 2 } } } + C _ { k } \left( { \frac { \tau } { a } } \right) ^ { 1 / 5 } .\tag{10}
$$

Indeed, with the choices $a = \epsilon ^ { 3 / 4 }$ and $b = \epsilon ^ { 1 / 8 }$ made below, our smallness assumption ensures $a , b \in ( 0 , 1 / 2 )$ . Normalization increases the residual coeficient bounds by at most a factor of two, and

$$
\frac { 2 \sqrt { k } \tau } { a } \leq 2 \sqrt { k } \epsilon ^ { 1 / 4 } \leq 2 c ^ { 1 / 4 } < 1 .
$$

Thus the normalized tuple satisfies the assumptions of Lemma 3.4.

Step 4: Move the coeficients into W and fix the extra Gaussian. We have completed Gaussian comparison while the residual coeficients remain in E. It remains to move these coeficients into W. This last move is small in coeficient norm, so it can be handled directly by sign stability.

Set

$$
\mathbf { w } _ { i } = \mathrm { p r o j } _ { W } \pmb { \theta } _ { i } ^ { E } , \qquad g _ { i } ( \mathbf { x } , \mathbf { c } ) = q _ { i } ( \mathbf { x } _ { V } ) + \pmb { \theta } _ { i } ^ { B } \cdot \mathbf { c } + \mathbf { w } _ { i } \cdot \Phi _ { 2 } ( \mathbf { x } ) ,
$$

and let

$$
g _ { \mathbf { c } } ( \mathbf { x } ) = \mathbf { 1 } \{ g _ { i } ( \mathbf { x } , \mathbf { c } ) \geq 0 { \mathrm { ~ f o r ~ a l l ~ } } i \in [ k ] \} .
$$

Fact 3.9 gives ${ \left\| \pmb { \theta } _ { i } ^ { E } - \mathbf { w } _ { i } \right\| } _ { 2 } \leq b \left\| \pmb { \theta } _ { i } ^ { E } \right\| _ { 2 } \leq b$ . Therefore, by orthonormality of the Hermite features, $\| r _ { i } - g _ { i } \| _ { L _ { 2 } ( \mathbf { x } , \mathbf { c } ) } \leq b .$ Using (9), (3), and a union bound, we obtain $\mathbf { P r } _ { \mathbf { x } , \mathbf { c } } [ h ( \mathbf { x } , \mathbf { c } ) \neq g _ { \mathbf { c } } ( \mathbf { x } ) ] \leq C _ { k } b ^ { 2 / 5 }$ Combining this with (4) and (10) yields

$$
\mathbf { P r } [ f ( \mathbf { x } ) \neq g _ { \mathbf { c } } ( \mathbf { x } ) ] \leq C _ { k } \left[ a ^ { 2 / 5 } + { \frac { \delta + a ^ { 2 / 5 } } { b ^ { 2 } } } + \left( { \frac { \tau } { a } } \right) ^ { 1 / 5 } + b ^ { 2 / 5 } \right] .
$$

Choose

$$
a = \epsilon ^ { 3 / 4 } , \qquad b = \epsilon ^ { 1 / 8 } , \qquad \epsilon = \delta + \tau .
$$

The four terms are bounded, respectively, by $\epsilon ^ { 3 / 1 0 } , \epsilon ^ { 3 / 4 } + \epsilon ^ { 1 / 2 0 } , \epsilon ^ { 1 / 2 0 }$ , and $\epsilon ^ { 1 / 2 0 }$ . Hence $\mathbf { P r _ { x , c } } [ f ( \mathbf { x } ) \neq$ $g _ { \mathbf { c } } ( \mathbf { x } ) ] \leq C _ { k } \epsilon ^ { 1 / 2 0 }$

Finally, averaging over c gives a deterministic $\mathbf { c } _ { * }$ with the same error bound over x. Let $g = g _ { \mathbf { c } _ { * } }$ Since $\pmb { \theta } _ { i } ^ { B } \cdot \mathbf { c } _ { * }$ is a constant and $\mathbf { w } _ { i } \in W$ , every defining polynomial satisfies

$$
g _ { i } ( \mathbf { \cdot } , \mathbf { c } _ { * } ) \in P _ { \leq 2 } ( V ) \oplus \{ \mathbf { w } \cdot \Phi _ { 2 } ( \mathbf { \cdot } ) : \mathbf { w } \in W \} .
$$

This proves the required conclusion.

□

## 3.3 Learning Regular Intersections

We now specialize to the fully regular case $V = \{ 0 \}$ and combine the preceding results with the moment-based dimension reduction of [DKS18] to obtain a learning algorithm for regular intersections. The algorithm estimates the first two labeled moments in the lifted feature space. The moment comparison in Lemma 3.3 shows that these estimates approximate the corresponding moments of the Gaussian surrogate. The Gaussian recovery guarantee then yields a coeficient subspace W of dimension at most $k ,$ outside which the surrogate has small directional randomized error. By Lemma 3.7, the original target admits a good approximation by an intersection supported on $W$ . We complete the algorithm by searching over a finite net of such intersections and selecting a hypothesis using independent validation samples.

Theorem 3.10 (Eficiently Learning a Regular Instance). There is an algorithm with the following guarantee. Let $d , k \geq 1$ be integers and let $\epsilon \in ( 0 , 1 / 2 )$ . Let D be a distribution over $\mathbb { R } ^ { d } \times \{ 0 , 1 \}$ with marginal $D _ { \mathbf { x } } = \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ . Suppose there exists an intersection $f ( \mathbf { x } ) = \mathbf { 1 } \{ p _ { 1 } ( \mathbf { x } ) \geq 0 , \ldots , p _ { k } ( \mathbf { x } ) \geq 0 \}$ where each $p _ { i } , i \in [ k ]$ , is a polynomial of degree at most two and is individually $( \tau , \{ 0 \} ) – r e g u l a r ,$ where $\begin{array} { r } { \tau \le \left( \frac { \epsilon } { k } \right) ^ { c } , \mathrm { P r } _ { ( \mathbf { x } , y ) \sim D } [ f ( \mathbf { x } ) \neq y ] \le \left( \frac { \epsilon } { k } \right) ^ { c } } \end{array}$ and c is a suficiently large universal constant.

Then given $k , \epsilon$ and i.i.d. sample access to D, the algorithm uses $d ^ { O ( 1 ) } \left( k / \epsilon \right) ^ { O ( k ^ { 2 } ) }$ time and sample complexity and with probability at least $2 / 3$ outputs a hypothesis $h : \mathbb { R } ^ { d }  \{ 0 , 1 \}$ with error $\mathbf { P r } _ { ( \mathbf { x } , y ) \sim D } [ h ( \mathbf { x } ) \neq y ] \leq \epsilon$

Proof. We analyze Algorithm 1. The proof uses Lemma 3.3 to compare the estimated moments with their Gaussian counterparts and Lemma 3.7 to transfer the resulting subspace approximation to the original distribution.

Let

$$
\nu = \operatorname* { P r } _ { ( \mathbf { x } , y ) \sim D } [ f ( \mathbf { x } ) \neq y ] , \qquad N = d ( d + 3 ) / 2 .
$$

Replacing identically zero defining polynomials by 1 and rescaling the others by positive constants, assume that $\| p _ { i } \| _ { 2 } = 1$ for every $i \in [ k ]$ . Write

$$
p _ { i } ( \mathbf { x } ) = c _ { i } + \pmb { \theta } _ { i } \cdot \Phi _ { 2 } ( \mathbf { x } ) , \qquad \widetilde { f } ( \mathbf { z } ) = \mathbf { 1 } \{ c _ { i } + \pmb { \theta } _ { i } \cdot \mathbf { z } \geq 0 \mathrm { ~ f o r ~ a l l ~ } i \in [ k ] \} .
$$

Here $\mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } _ { N } )$ . Fact 3.2 supplies the coeficient bounds required by Lemmas 3.3 and 3.7 with $V = \{ 0 \}$

Define

$$
\begin{array} { r } { \mu = \underset { \mathbf { z } } { \mathbf { E } } [ \widetilde { f } ( \mathbf { z } ) \mathbf { z } ] , \qquad M = \underset { \mathbf { z } } { \mathbf { E } } [ \widetilde { f } ( \mathbf { z } ) ( \mathbf { z } \mathbf { z } ^ { \top } - \mathbf { I } _ { N } ) ] , \qquad T = [ \mu \ M ] . } \end{array}
$$

Let ${ \widehat { \pmb { \mu } } } , { \widehat { M } }$ , and $\widehat { T }$ be the estimates computed by the algorithm. We compare $\widehat { T }$ with $T$ by bounding the contributions of Gaussian replacement, label disagreement, and sampling error.

Recall that $\lambda ~ = ~ ( \epsilon / k ) ^ { A }$ . Smoothing the thresholds in Lemma 3.3 and applying Gaussian anticoncentration bounds the Gaussian-comparison error by $k ^ { O ( 1 ) } \tau ^ { \Omega ( 1 ) }$ , with universal exponents.

```latex
Algorithm 1 RegularLearning
1: Input: $d , k , \epsilon ,$ and i.i.d. sample access to $D .$
2: Output: A hypothesis $h : \mathbb { R } ^ { d }  \{ 0 , 1 \}$
3: Set $N = d ( d + 3 ) / 2 , \lambda = ( \epsilon / k ) ^ { A } ,$ and $r = ( \epsilon / ( C k ) ) ^ { 5 / 2 }$ , for suficiently large universal constants
$A , C .$
4: Using fresh samples, estimate
$\underset { ( \mathbf { x } , y ) \sim D } { \mathbf { E } } [ y \Phi _ { 2 } ( \mathbf { x } ) ] \quad \mathrm { a n d } \quad \underset { ( \mathbf { x } , y ) \sim D } { \mathbf { E } } [ y ( \Phi _ { 2 } ( \mathbf { x } ) \Phi _ { 2 } ( \mathbf { x } ) ^ { \top } - \mathbf { I } _ { N } ) ]$
by $\widehat { \pmb { \mu } }$ and ${ \widehat { M } } ,$ with Euclidean and operator norm errors at most $\lambda / 6 ,$ respectively, and joint
failure probability at most $1 / 6 .$
5: Form $\widehat { T } = [ \widehat { \pmb { \mu } } \widehat { \pmb { M } } ]$ . Let W be the span of its top min $\{ k , N \}$ left singular vectors, and compute
an orthonormal basis $\mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { m }$ of $W .$
6: Construct an r-net $\mathcal { N }$ of the unit Gaussian $L _ { 2 }$ sphere in
${ \mathcal { Q } } _ { W } = \operatorname { s p a n } \{ 1 , \mathbf { v } _ { 1 } \cdot \Phi _ { 2 } , \ldots , \mathbf { v } _ { m } \cdot \Phi _ { 2 } \} ,$
using a Euclidean net of the coeficient unit sphere. Include the constant polynomials 1 and −1.
7: Form the candidate family
${ \mathcal { H } } = \{ \mathbf { x } \mapsto \mathbf { 1 } \{ q _ { i } ( \mathbf { x } ) \geq 0 { \mathrm { ~ f o r ~ a l l ~ } } i \in [ k ] \} : q _ { 1 } , \ldots , q _ { k } \in { \mathcal { N } } \}$
8: Draw an independent validation sample $\{ ( \mathbf { x } ^ { ( j ) } , y ^ { ( j ) } ) \} _ { j = 1 } ^ { n _ { \mathrm { v a l } } }$ , where $n _ { \mathrm { v a l } } = \lceil C \epsilon ^ { - 2 } \log ( 1 2 | \mathcal { H } | ) \rceil$
9: Return: A hypothesis
$\widehat { h } \in \arg \operatorname* { m i n } _ { h \in \mathcal { H } } \frac { 1 } { n _ { \mathrm { v a l } } } \sum _ { j = 1 } ^ { n _ { \mathrm { v a l } } } \mathbf { 1 } \{ h ( \mathbf { x } ^ { ( j ) } ) \neq y ^ { ( j ) } \} .$
```

The label disagreement contributes $O ( \sqrt \nu )$ : for every unit vector $\mathbf { v } ,$ , Gaussian hypercontractivity gives

$$
\left\| \mathbf { v } \cdot \Phi _ { 2 } \right\| _ { 2 } = 1 , \qquad \left\| ( \mathbf { v } \cdot \Phi _ { 2 } ) ^ { 2 } - 1 \right\| _ { 2 } = O ( 1 ) ,
$$

so Cauchy–Schwarz bounds the perturbations of the two moment blocks in Euclidean and operator norm, respectively.

The entries of the observed moment blocks have universally bounded variance. Entrywise medianof-means estimates with accuracy $\lambda / ( 6 N )$ therefore give the accuracies required by the algorithm using $\mathrm { p o l y } ( d , 1 / \lambda )$ time and samples. With probability at least $5 / 6$ , we consequently have

$$
\begin{array} { r } { \left\| \widehat { T } - T \right\| _ { \mathrm { o p } } \leq k ^ { O ( 1 ) } \tau ^ { \Omega ( 1 ) } + O ( \sqrt \nu ) + \lambda / 3 \leq \lambda , } \end{array}
$$

where the last inequality holds when the constant c in the theorem is suficiently large.

Let W be the subspace computed by the algorithm. Since $\widetilde { f }$ depends only on the projection onto span $\{ \pmb \theta _ { 1 } , \dots , \pmb \theta _ { k } \}$ , the range of $T$ lies in this subspace. Hence rank $( T ) \leq k$ , and $\| \mathrm { p r o j } _ { W ^ { \perp } } T \| _ { \mathrm { o p } } \leq 2 \lambda$ It follows that, for every unit $\mathbf { v } \in W ^ { \perp }$ , the correlation of $\widetilde { f }$ with any centered, normalized univariate polynomial of degree at most two in ${ \bf v } \cdot { \bf z }$ is $O ( \lambda )$ . By [DKS18, Proposition 5.2],

$$
\mathrm { R a n d } _ { \widetilde { f } } ( \mathbf { v } ) \leq O \left( k ^ { 4 / 1 1 } \lambda ^ { 1 / 1 1 } \log ^ { 2 / 1 1 } ( 2 k / \lambda ) \right) = : \delta _ { 0 } .
$$

Lemma 3.7 therefore gives an intersection g whose defining polynomials belong to

$$
\mathcal { Q } _ { W } = \mathbb { R } \oplus \{ { \bf w } \cdot \Phi _ { 2 } : { \bf w } \in W \} ,
$$

and such that

$$
\begin{array} { r } { \underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } { \mathbf { P r } } [ f ( \mathbf { x } ) \neq g ( \mathbf { x } ) ] \leq k ^ { O ( 1 ) } ( \delta _ { 0 } + \tau ) ^ { 1 / 2 0 } \leq \epsilon / 4 . } \end{array}
$$

Choosing A suficiently large and then c suficiently large ensures this inequality.

It remains to show that the candidate family H contains a good hypothesis. Since $\mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { m }$ are orthonormal coeficient vectors, the polynomials $1 , \mathbf { v } _ { 1 } \cdot \Phi _ { 2 } , \ldots , \mathbf { v } _ { m } \cdot \Phi _ { 2 }$ are orthonormal under $\mathcal { N } ( 0 , \mathbf { I } _ { d } )$ . Thus the Euclidean coeficient net used by the algorithm is an r-net in Gaussian $L _ { 2 }$ norm. Since $m \leq k , | \mathcal { H } | \leq ( C / r ) ^ { k ( k + 1 ) }$ . Normalize the nonzero defining polynomials of $^ { g , }$ replacing zero constraints by 1. The sign-stability estimate from the proof of Lemma 3.7 shows that rounding these polynomials to the net changes the intersection on a set of probability at most $C k r ^ { 2 / 5 }$ . The algorithm’s choice $r = ( \epsilon / ( C k ) ) ^ { 5 / 2 }$ therefore ensures that H contains a hypothesis $h _ { 0 }$ with

$$
\begin{array} { r } { \underset { ( \mathbf { x } , y ) \sim D } { \mathbf { P r } } [ h _ { 0 } ( \mathbf { x } ) \neq y ] \leq \mathbf { P r } [ f ( \mathbf { x } ) \neq g ( \mathbf { x } ) ] + \underset { \mathbf { x } } { \mathbf { P r } } [ g ( \mathbf { x } ) \neq h _ { 0 } ( \mathbf { x } ) ] + \nu \leq 3 \epsilon / 4 , } \end{array}
$$

where the probabilities over x are under $\mathcal { N } ( 0 , \mathbf { I } _ { d } )$

Conditioned on the preceding samples, the family H is fixed and the validation samples are independent. Hoefding’s inequality and a union bound imply that, with probability at least $5 / 6 .$ every candidate’s empirical error difers from its true error by at most $\epsilon / 8$ . The empirical minimizer returned by the algorithm therefore has true error at most $\epsilon / 4$ above that of $h _ { 0 }$ , and hence at most ϵ.

The overall success probability is at least $2 / 3$ . Moment estimation and subspace recovery take $\mathrm { p o l y } ( d , k / \epsilon )$ time, while $| \mathcal { H } | = ( k \bar { / } \epsilon ) ^ { O ( k ^ { 2 } ) }$ . The claimed time and sample bounds follow. □

## 4 Reduction to the Regular Case using Randomness

In this section, we show that if the randomized error is suficiently small along every direction orthogonal to V, then the target is close to an intersection whose defining polynomials are $( \tau , V )$ regular. The intuition for this is simple. We write each original defining polynomial as $q _ { i } ( \mathbf { x } ) =$ $q _ { i } ( \mathbf { z } , \mathbf { w } ) = a _ { i } ( \mathbf { z } ) + ( b _ { i } + L _ { i } ^ { \top } \mathbf { z } ) ^ { \top } \mathbf { w } + \mathbf { w } ^ { \top } \mathbf { A } _ { i } \mathbf { w } - \mathrm { t r } ( \mathbf { A } _ { i } )$ , where ${ \bf z } = \mathrm { p r o j } _ { V } ( { \bf x } )$ and ${ \bf w } = \mathrm { p r o j } _ { V ^ { \perp } } ( { \bf x } )$ . We call $( b _ { i } + L _ { i } ^ { \top } \mathbf { z } ) ^ { \top } \mathbf { w } + \mathbf { w } ^ { \top } \mathbf { A } _ { i } \mathbf { w } - \operatorname { t r } ( \mathbf { A } _ { i } )$ the remainder of $q _ { i }$ . Notice that a remainder consists of a linear term $b _ { i } \cdot \mathbf { w } .$ , a mixed term $( L _ { i } ^ { \top } \mathbf { z } ) \cdot \mathbf { w }$ and a pure quadratic term defined by $\mathbf { A } _ { i }$ . We select the subspace $U \subseteq V ^ { \bot }$ to be the subspace spanned by all $b _ { i }$ , right singular vectors of $L _ { i }$ corresponding to large singular values and large eigenvectors of $\mathbf { A } _ { i }$ . The intuition here is that if we fix the component of $\mathbf { x }$ in $U$ at some fixed point. Now the mixed term of the resulting polynomial has no large singular values and the quadratic term of it has no large eigenvalues. In fact, combining the fact that every direction in $V ^ { \perp }$ has a tiny randomized error and Gaussian anti-concentration, we are able to show that there exists a choice of the coordinates in $U$ such that the resulting intersection is close to the original one and each defining polynomial has a not too small norm.

However, this does not imply regularity yet, because the resulting polynomial could have new large linear terms. To deal with this, we apply the same idea above but only find a fix along the subspace spanned by these k linear terms.

As a result, the residual linear coeficient vectors and the operator norms of the residual mixed and quadratic coeficient matrices are small, while each resulting polynomial has a nonnegligible Gaussian $L _ { 2 }$ norm. With our parameter choices, we distinguish two cases. If the variance is suficiently large relative to the residual coeficient bounds, the directional-influence identity in the proof of Fact 3.2 gives $( \tau , V )$ -regularity. Otherwise, the variance is small compared with the squared $L _ { 2 }$ norm, so the polynomial has a nonnegligible mean. Chebyshev’s inequality then shows that its threshold function is close to a constant.

Theorem 4.1 (Regular approximation from small randomized error). There is a universal constant $C > 1$ such that the following holds. Let $k \geq 1 , \epsilon , \tau \in ( 0 , 1 )$ , and let $V \subseteq \mathbb { R } ^ { d }$ be a subspace. Suppose that $f : \mathbb { R } ^ { d }  \{ 0 , 1 \}$ is an intersection of at most k degree-2 PTFs and

$$
\operatorname { R a n d } _ { f } ( \mathbf { v } ) \leq \alpha _ { k } ( \epsilon , \tau ) : = \frac { \tau ^ { 2 } \epsilon ^ { 2 } } { C k ^ { 1 2 } } ,
$$

for every unit vector $\mathbf { v } \in V ^ { \perp }$ . Then there are $k ^ { \prime } \leq k$ polynomials $q _ { 1 } , \ldots , q _ { k ^ { \prime } }$ of degree at most two such that $g ( \mathbf { x } ) = \mathbf { 1 } \{ q _ { 1 } ( \mathbf { x } ) \geq 0 , \ldots , q _ { k ^ { \prime } } ( \mathbf { x } ) \geq 0 \}$ satisfies $\begin{array} { r } { \mathbf { P r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ f ( \mathbf { x } ) \neq g ( \mathbf { x } ) ] \leq \epsilon } \end{array}$ , and each $q _ { i }$ is $( \tau , V )$ -regular.

We first give two facts about random restrictions. The first allows us to choose the second set of directions after fixing the first set.

Fact 4.2. Let $f , h : \mathbb { R } ^ { d }  \{ 0 , 1 \}$ be Boolean functions. For every unit vector $\mathbf { v } \in \mathbb { R } ^ { d }$ , we have

$$
\mathrm { R a n d } _ { h } ( \mathbf { v } ) \leq \mathrm { R a n d } _ { f } ( \mathbf { v } ) + 2 \operatorname* { P r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ f ( \mathbf { x } ) \neq h ( \mathbf { x } ) ] .
$$

Moreover, let U be a subspace with orthonormal basis $\mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { m }$ . For $\textbf { z } \in \ U$ , define $f _ { \mathbf { Z } } ( \mathbf { x } ) =$ $f ( \mathbf { z } + \mathrm { p r o j } _ { U ^ { \perp } } \mathbf { x } )$ . Then

$$
\underset { \mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } _ { U } ) } { \mathbf { E } } \left[ \underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } { \mathbf { P r } } [ f ( \mathbf { x } ) \neq f _ { \mathbf { z } } ( \mathbf { x } ) ] \right] \leq \sum _ { j = 1 } ^ { m } \mathrm { R a n d } _ { f } ( \mathbf { v } _ { j } ) .
$$

Proof of Fact $4 \cdot 2 .$ Let $\mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ and $s \sim \mathcal { N } ( 0 , 1 )$ be independent, and set $\mathbf { x } ^ { \prime } = \operatorname { p r o j } _ { \mathbf { v } ^ { \perp } } \mathbf { x } + s \mathbf { v }$ By a union bound,

$$
\begin{array} { r l } & { { \mathrm { R a n d } } _ { h } ( \mathbf { v } ) = { \underset { \mathbf { x } , s } { \mathbf { P r } } } [ h ( \mathbf { x } ) \neq h ( \mathbf { x } ^ { \prime } ) ] } \\ & { \qquad \leq { \underset { \mathbf { x } , s } { \mathbf { P r } } } [ h ( \mathbf { x } ) \neq f ( \mathbf { x } ) ] + { \underset { \mathbf { x } , s } { \mathbf { P r } } } [ f ( \mathbf { x } ) \neq f ( \mathbf { x } ^ { \prime } ) ] + { \underset { \mathbf { x } , s } { \mathbf { P r } } } [ f ( \mathbf { x } ^ { \prime } ) \neq h ( \mathbf { x } ^ { \prime } ) ] } \\ & { \qquad = { \mathrm { R a n d } } _ { f } ( \mathbf { v } ) + 2 \underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } { \mathbf { P r } } [ f ( \mathbf { x } ) \neq h ( \mathbf { x } ) ] , } \end{array}
$$

where the last equality uses that both x and $\mathbf { x } ^ { \prime }$ are standard Gaussians.

For the second claim, draw $\mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ and $\mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } _ { U } )$ independently, and replace the coordinates of x along $\mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { m }$ one at a time by those of z. At each step, the input is standard Gaussian, and the probability that the label changes at step j is ${ \mathrm { R a n d } } _ { f } ( \mathbf { v } _ { j } )$ . Taking a union bound over the m steps proves the claim. □

The next lemma ensures that the defining polynomials do not become too small in $L _ { 2 }$ after fixing the coordinates.

Lemma 4.3. There is a universal constant $K \geq 1$ such that the following holds. Let $p : \mathbb { R } ^ { d } $ R be a nonzero polynomial of degree at most two, and let $U \subseteq \mathbb { R } ^ { d }$ be a subspace. For $\mathbf { z } \in U$ , define $p _ { \mathbf { z } } ( \mathbf { x } ) = p ( \mathbf { z } + \mathrm { p r o j } _ { U ^ { \perp } } \mathbf { x } )$ . Then, for every $\kappa \in ( 0 , 1 )$ ,

$$
\begin{array} { r } { \mathbf { P r } \mathbf { \| } p _ { \mathbf { z } } \| _ { 2 } < \kappa \| p \| _ { 2 } ] \leq K \sqrt { \kappa } . } \end{array}
$$

Proof of Lemma $4 . 3 .$ Define $Q ( \mathbf { z } ) = \mathbf { E } _ { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { r / \perp } ) } [ p ( \mathbf { z } + \mathbf { w } ) ^ { 2 } ]$ . Then $Q$ is a nonnegative polynomial of degree at most four, $Q ( \mathbf { z } ) = \| p _ { \mathbf { z } } \| _ { 2 } ^ { 2 }$ , and $\begin{array} { r } { \mathbf { E } _ { \mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } _ { U } ) } [ Q ( \mathbf { z } ) ] = \left. p \right. _ { 2 } ^ { 2 } } \end{array}$

By the Carbery–Wright inequality for degree-4 polynomials,

$$
\underset { \mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } _ { U } ) } { \mathbf { P r } } [ Q ( \mathbf { z } ) < \kappa ^ { 2 } \left. p \right. _ { 2 } ^ { 2 } ] \leq K ( \kappa ^ { 2 } ) ^ { 1 / 4 } = K \sqrt { \kappa } .
$$

Proof of Theorem $4 . 1$ . We may write

$$
f ( \mathbf { x } ) = \mathbf { 1 } \{ p _ { 1 } ( \mathbf { x } ) \geq 0 , \ldots , p _ { k } ( \mathbf { x } ) \geq 0 \} , \qquad \| p _ { i } \| _ { 2 } = 1 \quad \mathrm { f o r ~ e v e r y ~ } i \in [ k ] .
$$

Indeed, we can add constant-true constraints, replace an identically zero polynomial by 1, and positively rescale each defining polynomial. If $V ^ { \perp } = \{ 0 \}$ , the conclusion holds with $g = f$ . We therefore assume that $V ^ { \perp } \neq \{ 0 \}$

Let K be the constant from Lemma 4.3, and set

$$
\alpha = \frac { \tau ^ { 2 } \epsilon ^ { 2 } } { C k ^ { 1 2 } } , \qquad \kappa = ( 4 K k ) ^ { - 2 } , \qquad \delta ^ { 2 } = \frac { \tau ^ { 2 } \epsilon \kappa ^ { 4 } } { 2 0 0 k ^ { 2 } } .
$$

In particular, $\delta \in ( 0 , 1 )$ . Write $W = V ^ { \perp }$ , and, for $\mathbf { z } \in V$ and $\mathbf { w } \in W$ , decompose

$$
p _ { i } ( \mathbf { z } + \mathbf { w } ) = a _ { i } ( \mathbf { z } ) + \mathbf { b } _ { i } ^ { \top } \mathbf { w } + \mathbf { z } ^ { \top } \mathbf { L } _ { i } \mathbf { w } + \mathbf { w } ^ { \top } \mathbf { A } _ { i } \mathbf { w } - \mathrm { t r } ( \mathbf { A } _ { i } ) ,
$$

where $\mathbf { L } _ { i } : W \to V$ is linear and $\mathbf { A } _ { i } : W \to W$ is symmetric. By Gaussian orthogonality,

$$
\begin{array} { r } { \| \mathbf { b } _ { i } \| _ { 2 } ^ { 2 } + \| \mathbf { L } _ { i } \| _ { F } ^ { 2 } + 2 \| \mathbf { A } _ { i } \| _ { F } ^ { 2 } \leq 1 . } \end{array}\tag{11}
$$

First restriction. Let $U \subseteq W$ be the span, over all $i \in [ k ]$ , of $\mathbf { b } _ { i } .$ , the right singular vectors of $\mathbf { L } _ { i }$ with singular value greater than $\delta ,$ , and the eigenvectors of $\mathbf { A } _ { i }$ with eigenvalue of absolute value greater than δ. By (11), m := dim $U \leq 3 k \delta ^ { - 2 }$ . Let $R = W \cap U ^ { \perp }$ . By the choice of $U$ , for every $i \in [ k ]$

$$
\begin{array} { r } { \mathrm { p r o j } _ { R } \mathbf { b } _ { i } = 0 , \qquad \| \mathbf { L } _ { i } \mathrm { p r o j } _ { R } \| _ { \mathrm { o p } } \leq \delta , \qquad \| \mathbf { A } _ { i } \mathrm { p r o j } _ { R } \| _ { \mathrm { o p } } = \| \mathrm { p r o j } _ { R } \mathbf { A } _ { i } \| _ { \mathrm { o p } } \leq \delta . } \end{array}\tag{12}
$$

For $\mathbf { t } \in U$ , define

$$
p _ { i , { \bf t } } ( { \bf x } ) = p _ { i } ( \mathrm { p r o j } _ { V } { \bf x } + { \bf t } + \mathrm { p r o j } _ { R } { \bf x } ) , \qquad f _ { \bf t } ( { \bf x } ) = f ( \mathrm { p r o j } _ { V } { \bf x } + { \bf t } + \mathrm { p r o j } _ { R } { \bf x } ) .
$$

By Fact 4.2,

$$
\underset { \mathbf { t } \sim \mathcal { N } ( 0 , \mathbf { I } _ { U } ) } { \mathbf { E } } \left[ \underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } { \mathbf { P r } } [ f ( \mathbf { x } ) \neq f _ { \mathbf { t } } ( \mathbf { x } ) ] \right] \leq m \alpha .
$$

By Markov’s inequality, with probability at least $3 / 4$ over $\mathbf { t } \sim \mathcal { N } ( 0 , \mathbf { I } _ { U } )$ , the inner probability is at most 4mα. Moreover, Lemma 4.3 and a union bound imply that

$$
\operatorname* { P r } _ { \mathbf { t } \sim \mathcal { N } ( 0 , \mathbf { I } _ { U } ) } [ \exists i \in [ k ] : \| p _ { i , \mathbf { t } } \| _ { 2 } < \kappa ] \leq k K \sqrt { \kappa } = 1 / 4 .
$$

We can therefore fix $\mathbf { t } \in U$ such that

$$
e _ { 1 } : = \sum _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ f ( \mathbf { x } ) \neq f _ { \mathbf { t } } ( \mathbf { x } ) ] \leq 4 m \alpha , \qquad \| p _ { i , \mathbf { t } } \| _ { 2 } \geq \kappa \quad \mathrm { f o r ~ e v e r y ~ } i \in [ k ] .
$$

When $m = 0$ , the error is zero and the same conclusion holds with $\mathbf t = 0$

For $\mathbf { z } \in V$ and $\mathbf { w } \in R .$ , the restricted polynomial has the form

$$
\begin{array} { r } { p _ { i , { \bf t } } ( { \bf z } + { \bf w } ) = a _ { i , { \bf t } } ( { \bf z } ) + { \bf b } _ { i , { \bf t } } ^ { \top } { \bf w } + { \bf z } ^ { \top } { \bf L } _ { i , R } { \bf w } + { \bf w } ^ { \top } { \bf A } _ { i , R } { \bf w } - \mathrm { t r } ( { \bf A } _ { i , R } ) , } \end{array}
$$

where

$$
\mathbf { b } _ { i , \mathbf { t } } = 2 \mathrm { p r o j } _ { R } \mathbf { A } _ { i } \mathbf { t } , \qquad \mathbf { L } _ { i , R } = \mathbf { L } _ { i } { \big | } _ { R } , \qquad \mathbf { A } _ { i , R } = \mathrm { p r o j } _ { R } \mathbf { A } _ { i } { \big | } _ { R } .
$$

The operators $\mathbf { L } _ { i , R }$ and ${ \bf A } _ { i , R }$ have norm at most $\delta ,$ but $\mathbf { b } _ { i , \mathbf { t } }$ need not be small. We remove these linear terms by a second restriction.

Second restriction. Let

$$
H = \operatorname { s p a n } \{ \mathbf { b } _ { i , \mathbf { t } } : i \in [ k ] \} \subseteq R , \qquad S = R \cap H ^ { \perp } .
$$

We have dim $H \leq k$ . Although H depends on t, we have already fixed t. By Fact 4.2, for every unit $\mathbf { v } \in W$

$$
\operatorname { R a n d } _ { f _ { \mathbf { t } } } ( \mathbf { v } ) \leq \operatorname { R a n d } _ { f } ( \mathbf { v } ) + 2 e _ { 1 } \leq \alpha + 2 e _ { 1 } .
$$

For $\mathbf s \in H$ , define

$$
r _ { i , \mathrm { s } } ( \mathbf { x } ) = p _ { i } ( \mathrm { p r o j } _ { V } \mathbf { x } + \mathbf { t } + \mathbf { s } + \mathrm { p r o j } _ { S } \mathbf { x } ) , \qquad h _ { \mathrm { s } } ( \mathbf { x } ) = f ( \mathrm { p r o j } _ { V } \mathbf { x } + \mathbf { t } + \mathbf { s } + \mathrm { p r o j } _ { S } \mathbf { x } ) .
$$

Applying Fact 4.2 to $f _ { \mathbf { t } }$ and H, we obtain

$$
\underset { \mathbf { s } \sim \mathcal { N } ( 0 , \mathbf { I } _ { H } ) } { \mathbf { E } } \left[ \underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } { \mathbf { P r } } [ f _ { \mathbf { t } } ( \mathbf { x } ) \neq h _ { \mathbf { s } } ( \mathbf { x } ) ] \right] \leq k ( \alpha + 2 e _ { 1 } ) .
$$

Also, $\begin{array} { r l r } { \mathrm { \bf E } _ { { \bf s } \sim \mathcal { N } ( 0 , { \bf I } _ { H } ) } [ \| { \bf s } \| _ { 2 } ^ { 2 } ] } & { { } } & { } \end{array}$ dim $H \leq k$ . Applying Markov’s inequality to these two quantities, and applying Lemma 4.3 to $p _ { i , \mathbf { t } }$ , we can fix $\mathbf s \in H$ such that

$$
\begin{array} { r l r } & { e _ { 2 } : = \underset { \mathbf x \sim \mathcal N ( 0 , \mathbf I _ { d } ) } { \mathbf \mathbf \nabla } [ f _ { \mathbf t } ( \mathbf x ) \neq h _ { \mathbf s } ( \mathbf x ) ] \le 4 k ( \alpha + 2 e _ { 1 } ) , } & \\ & { } & \\ & { \| \mathbf s \| _ { 2 } ^ { 2 } \le 4 k , } & \\ & { } & { \| r _ { i , \mathbf s } \| _ { 2 } \ge \kappa \| p _ { i , \mathbf t } \| _ { 2 } \ge \kappa ^ { 2 } ~ \mathrm { ~ f o r ~ e v e r y ~ } i \in [ k ] . } & \end{array}\tag{13}
$$

Indeed, each of the first two conditions fails with probability at most $1 / 4$ , and the probability that any of the norm conditions fails is at most k $K \sqrt { \kappa } = 1 / 4$

Fix such an s, and write $r _ { i } = r _ { i , { \bf s } }$ and $h = h _ { \mathbf { s } }$ . The error after the two restrictions is bounded by

$$
\begin{array} { r } { \quad \operatorname* { P r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ f ( \mathbf { x } ) \neq h ( \mathbf { x } ) ] \leq e _ { 1 } + e _ { 2 } \leq ( 1 + 8 k ) e _ { 1 } + 4 k \alpha \leq 4 ( 1 + 8 k ) m \alpha + 4 k \alpha \leq 1 1 2 k ^ { 2 } \delta ^ { - 2 } \alpha . } \end{array}
$$

Regularity of the remaining polynomials. Since $\mathrm { p r o j } _ { S } \mathbf { b } _ { i , \mathbf { t } } = 0$ , the residual coeficients of $r _ { i }$ are

$$
\widetilde { \mathbf { b } } _ { i } = 2 \mathrm { p r o j } _ { S } \mathbf { A } _ { i , R } \mathbf { s } , \qquad \widetilde { \mathbf { L } } _ { i } = \mathbf { L } _ { i } | _ { S } , \qquad \widetilde { \mathbf { A } } _ { i } = \mathrm { p r o j } _ { S } \mathbf { A } _ { i } | _ { S } .
$$

By (12) and (13),

$$
\left\| \widetilde { \mathbf { b } } _ { i } \right\| _ { 2 } \leq 4 \delta \sqrt { k } , \qquad \left\| \widetilde { \mathbf { L } } _ { i } \right\| _ { \mathrm { o p } } \leq \delta , \qquad \left\| \widetilde { \mathbf { A } } _ { i } \right\| _ { \mathrm { o p } } \leq \delta .
$$

Every $r _ { i }$ ignores the coordinates in $U \oplus H$ . For a unit vector $\mathbf { v } \in V ^ { \perp }$ , let $\mathbf { w } = \mathrm { p r o j } _ { S } \mathbf { v }$ . Using the coeficient identity from Fact 3.2, we obtain

$$
\begin{array} { r l } & { \operatorname* { i n f } _ { r _ { i } } ( \mathbf { v } ) = \underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } { \mathbf { E } } [ ( \mathbf { v } \cdot \nabla r _ { i } ( \mathbf { x } ) ) ^ { 2 } ] } \\ & { \qquad = ( \widetilde { \mathbf { b } } _ { i } ^ { \top } \mathbf { w } ) ^ { 2 } + \left\| \widetilde { \mathbf { L } } _ { i } \mathbf { w } \right\| _ { 2 } ^ { 2 } + 4 \left\| \widetilde { \mathbf { A } } _ { i } \mathbf { w } \right\| _ { 2 } ^ { 2 } } \\ & { \qquad \leq ( 1 6 k + 5 ) \delta ^ { 2 } \leq 2 1 k \delta ^ { 2 } . } \end{array}
$$

Thus the second restriction introduces only small linear coeficients, and no further restriction is needed.

It remains to compare this bound with the variance of $r _ { i }$ . Set $\begin{array} { r } { \lambda = \frac { \epsilon \kappa ^ { 4 } } { 8 k } . \mathrm { ~ I f ~ } \mathrm { V a r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ r _ { i } ( \mathbf { x } ) ] \ge \lambda } \end{array}$ let $q _ { i } = r _ { i }$ . By the choice of $\delta ,$ for every unit $\mathbf { v } \in V ^ { \perp }$ 2

$$
\begin{array} { r } { \operatorname* { i n f } _ { q _ { i } } ( \mathbf { v } ) \leq 2 1 k \delta ^ { 2 } \leq \tau ^ { 2 } \lambda \leq \tau ^ { 2 } \operatorname { V a r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ q _ { i } ( \mathbf { x } ) ] . } \end{array}
$$

Therefore, $q _ { i }$ is $( \tau , V )$ -regular.

Otherwise, let $\mu _ { i } = \mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ r _ { i } ( \mathbf { x } ) ]$ . By (13),

$$
\begin{array} { r } { \mu _ { i } ^ { 2 } = \| r _ { i } \| _ { 2 } ^ { 2 } - \operatorname { V a r } _ { \mathbf x \sim \mathcal N ( 0 , \mathbf I _ { d } ) } [ r _ { i } ( \mathbf x ) ] \geq \kappa ^ { 4 } - \lambda \geq \kappa ^ { 4 } / 2 > 0 . } \end{array}
$$

Set $q _ { i } = 1$ if $\mu _ { i } > 0$ and $q _ { i } = - 1 \mathrm { i f } \mu _ { i } < 0$ . This constant polynomial is $( \tau , V )$ -regular. By Chebyshev’s inequality,

$$
\begin{array} { r l r } {  { \operatorname* { P r } _ { \mathbf { r } } } [ \mathbf { 1 } \{ r _ { i } ( \mathbf { x } ) \geq 0 \} ] \# \mathbf { 1 } \{ q _ { i } \geq 0 \} ] }  \\ & { } & { \quad \times \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } \\ & { } & { \quad \leq \operatorname* { P r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ | r _ { i } ( \mathbf { x } ) - \mu _ { i } | \geq | \mu _ { i } | ] } \\ & { } & { \quad \leq \frac { \mathrm { V a r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ r _ { i } ( \mathbf { x } ) ] } { \mu _ { i } ^ { 2 } } \leq \frac { 2 \lambda } { \kappa ^ { 4 } } = \frac { \epsilon } { 4 k } . } \end{array}
$$

Let $g ( \mathbf { x } ) = \mathbf { 1 } \{ q _ { i } ( \mathbf { x } ) \geq 0$ for every $i \in [ k ] \}$ . Combining the preceding bound on $\mathbf { P r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ f ( \mathbf { x } ) \neq$ $h ( \mathbf { x } ) ]$ with a union bound gives

$$
\mathbf { P r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ f ( \mathbf { x } ) \neq g ( \mathbf { x } ) ] \leq 1 1 2 k ^ { 2 } \delta ^ { - 2 } \alpha + \epsilon / 4 .
$$

Taking C suficiently large makes the first term in the preceding bound at most $\epsilon / 2$ , so the error is at most $3 \epsilon / 4 \le \epsilon$

Finally, positive rescaling preserves both the threshold and $( \tau , V )$ -regularity, so we may normalize every $q _ { i }$ to have $L _ { 2 }$ norm one. Constant-true constraints can be omitted, and an intersection containing a constant-false constraint can be represented by the single polynomial −1. If all constraints are omitted, use the single polynomial 1. Thus $g$ is an intersection of at most k degree-2 $\mathrm { P T F s }$ , regular with respect to the original subspace V . □

## 5 Certificate Polynomial for Intersections of k Degree-2 PTFs

In this section, we develop the main structural result, Theorem 5.1, for intersections of degree-2 polynomial threshold functions. We show that given a Boolean function $f ( \mathbf { x } )$ that is a product of an intersection of k degree-2 PTFs and any Boolean function $g ( \mathbf { w } )$ that is independent of a direction $\mathbf { v } \in \mathbb { R } ^ { d }$ . Suppose the randomized error of $f$ along v is at least $\alpha ,$ then there is a certificate polynomial of degree at most 2k correlates with $f .$ In particular, such a polynomial can be written as a product as one polynomial that only depends on v direction and one independent of v direction. As we show in Section 6, such a structure enables us to eficiently find such a direction v.

Theorem 5.1. Let $f ( \mathbf { x } ) = g ( \mathbf { w } ) \mathbb { 1 } \{ q _ { 1 } ( \mathbf { x } ) \geq 0 , \ldots , q _ { k } ( \mathbf { x } ) \geq 0 \} : \mathbb { R } ^ { d } \to \{ 0 , 1 \}$ , where for $i \in [ k ]$ $q _ { i }$ is a degree-2 polynomial, ${ \bf w } = \mathrm { p r o j } _ { { \bf v } ^ { \perp } } ( { \bf x } )$ and $g ( \mathbf { w } ) \in \mathbf { v } ^ { \perp } \to \{ 0 , 1 \} , \mathbf { v } \in \mathbb { S } ^ { d - 1 }$ . Suppose that Rand ${ \bf \nabla } _ { \cdot } f ( { \bf v } ) \geq \alpha$ , for some $\alpha \in ( 0 , 1 )$ , then there exists some $r \in [ 2 k ]$ and a polynomial $p ( \mathbf { w } ) : \mathbf { v } ^ { \perp } \to \mathbb { R }$ with $\| p \| = 1$ such that

$$
\underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ f ( \mathbf { x } ) h _ { r } ( \mathbf { v } \cdot \mathbf { x } ) p ( \mathbf { w } ) ] \geq \Omega ( \alpha ^ { 2 ^ { O ( k ) } } ) ,
$$

and $r + \deg ( p ) \leq 2 k$

To gain an intuition for why some degree 2k polynomial can correlate with such a function $f ,$ we fix some w $\perp \textbf { v }$ and treat $f ( \mathbf { x } )$ as a one-dimensional function along v direction. If $g ( \mathbf { w } ) = 1$ , then $f ( \mathbf { x } )$ is the indicator of a union of at most $k + 1$ intervals along the v direction, with at most 2k jump points. This implies that for such a fixed layer w, a degree 2k polynomial along direction v that crosses the boundaries of these intervals has a correlation with $f .$ . However, such a construction does not imply a certificate polynomial required in Theorem 5.1, because if we average the correlation for such a polynomial over w spaces, the correlations can cancel with each other. Thus, some multiplies $p ( \mathbf { w } )$ is needed to avoid such a cancellation.

Though, such an intuition does not directly give a way of constructing the required certificate, it motivates breaking down $f ( \mathbf { x } )$ into layers by fixing $\mathbf { w } \in \mathbf { v } ^ { \perp }$ and looking at the decision boundary of $f$ as a one-dimensional function along v. We introduce the following operator that will be heavily used throughout this work.

## 5.1 Boundary Operator and Weighted Decision Boundary

Definition 5.2 (Boundary Operator). Let P be the class of polynomials over $\mathbb { R } ^ { d }$ and $\mathbf { v } \in \mathbb { S } ^ { d - 1 }$ , we define the boundary operator $M _ { \mathbf { v } } : P  P$ as $M _ { \mathbf { v } } ( p ) = ( \mathbf { x } \cdot \mathbf { v } ) p - \partial _ { \mathbf { v } } p$

Before presenting the properties of the boundary operator, we first give the following facts about one-dimensional Hermite polynomials as well as the derivative of one-dimensional Boolean functions.

Fact 5.3. Let $k \in \mathbb { Z } _ { + }$ and let $h _ { k } ( z ) : \mathbb { R }  \mathbb { R }$ be the 1D degree-k Hermite polynomial. Then the following identities hold.

$$
d h _ { k } ( z ) / d z = \sqrt { k } h _ { k - 1 } ( z ) , \qquad z h _ { k } ( z ) = \sqrt { k + 1 } h _ { k + 1 } ( z ) + \sqrt { k } h _ { k - 1 } ( z ) .
$$

Fact 5.4. Let $f : \mathbb { R }  \{ 0 , 1 \}$ be a step function with a set of finite jump points S. For every $s \in S$ , define $\begin{array} { r } { \Delta _ { f } ( s ) = \operatorname* { l i m } _ { \tau \to 0 ^ { + } } f ( s + \tau ) - \operatorname* { l i m } _ { \tau \to 0 ^ { + } } f ( s - \tau ) } \end{array}$ . Then, in the sense of distributions, $\begin{array} { r } { d f / d x = \sum _ { s \in S } \Delta _ { f } ( s ) \delta _ { s } , } \end{array}$ , where $\delta _ { s }$ is the Dirac delta distribution at s.

The boundary operator has two important properties following Fact 5.3 that will be used for our proof. The first one is Fact 5.5.

Fact 5.5. Let $p ( \mathbf { x } ) \colon \ \mathbb { R } ^ { d } \ \to \ \mathbb { R }$ be a polynomial with degree r. Let $\textbf { v } \in \ \mathbb { S } ^ { d - 1 }$ . Write $p ( \mathbf { x } ) \mathbf { \alpha } = \mathbf { \alpha }$ $\textstyle \sum _ { i = 0 } ^ { r } h _ { i } ( \mathbf { v } \cdot \mathbf { x } ) M _ { i } ( \mathbf { w } )$ , where $\mathbf { w } = \mathrm { p r o j } _ { \mathbf { v } ^ { \perp } } ( \mathbf { x } )$ and $M _ { i } ( \mathbf { w } )$ is a polynomial over w with degree at most $\begin{array} { r } { r - i . \ T h e n , \ M _ { \mathbf { v } } p ( \mathbf { x } ) = \sum _ { i = 0 } ^ { r } \sqrt { i + 1 } h _ { i + 1 } ( \mathbf { v } \cdot \mathbf { x } ) M _ { i } ( \mathbf { w } ) } \end{array}$

Roughly speaking, for every polynomial $p ( \mathbf { x } )$ , the Hermite decomposition of $M _ { \mathbf { v } } p$ has no term that is totally independent of the v direction. In other words, if $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ f ( \mathbf { x } ) M _ { \mathbf { v } } p ( \mathbf { x } ) ] \ge \beta$ , then Fact 5.5 implies that there must be some $i \geq 1$ such that $\mathbf { E _ { x \sim \mathcal { N } ( 0 , \mathbf { I } ) } } [ f ( \mathbf { x } ) h _ { i } ( \mathbf { v } \cdot \mathbf { x } ) M _ { i - 1 } ( \mathbf { w } ) ] \geq \beta / ( r + 1 ) ^ { 3 / 2 }$ Thus, to prove Theorem 5.1, it sufices to construct some certificate polynomial $p ( \mathbf { x } )$ such that $M _ { \mathbf { v } } p$ has non-trivial correlation with $f .$

The second property, Lemma 5.6 is an application of integration by parts. Roughly speaking, consider the correlation between a one-dimensional Boolean function $f ( \mathbf { x } )$ along direction $\mathbf { v } ,$ and a polynomial $M _ { \mathbf { v } } p$ under a Gaussian distribution. The correlation can be written as the sum of the value of $p$ (up to the sign change of $f )$ at every boundary point weighted by its Gaussian density.

Lemma 5.6. Let $f : \mathbb { R } ^ { d }  \{ 0 , 1 \}$ be a Boolean function. For $\mathbf { v } \in \mathbb { S } ^ { d - 1 }$ , write ${ \bf x } = z { \bf v } + { \bf w }$ , where $z = { \bf v } \cdot { \bf x }$ and $\mathbf { w } = \mathrm { p r o j } _ { \mathbf { v } ^ { \perp } } ( \mathbf { x } )$ . For every $\mathbf { w } \perp \mathbf { v } ,$ denote by $f _ { \mathbf { w } } ( z ) = f ( z \mathbf { v } + \mathbf { w } )$ . Then for every polynomial $p : \mathbb { R } ^ { d }  \mathbb { R }$ , it holds

$$
\begin{array} { r } { \underline { { \mathbf { E } } } _ { z \sim \mathcal { N } ( 0 , 1 ) } [ f _ { \mathbf { w } } ( z ) M _ { \mathbf { v } } p ( z \mathbf { v } + \mathbf { w } ) ] = \underline { { \mathbf { E } } } _ { z \sim \mathcal { N } ( 0 , 1 ) } [ f _ { \mathbf { w } } ^ { \prime } ( z ) p ( z \mathbf { v } + \mathbf { w } ) ] } \end{array}
$$

In particular, if $f _ { \mathbf { w } } ( z )$ is a step function with the set of finite jump points $S _ { \mathbf { w } _ { \mathbf { \delta } } }$ , then

$$
\mathbf { \Psi } _ { z \sim \mathcal { N } ( 0 , 1 ) } ^ { \mathbf { E } } [ f _ { \mathbf { w } } ( z ) M _ { \mathbf { v } } p ( z \mathbf { v } + \mathbf { w } ) ] = \sum _ { s \in S _ { \mathbf { w } } } \Delta _ { f _ { \mathbf { w } } } ( s ) p ( s \mathbf { v } + \mathbf { w } ) \phi ( s ) .
$$

Proof of Lemma 5.6. We prove Lemma 5.6 using integration by parts. We have

$$
\begin{array} { r l } { \underset { z \sim \sqrt { ( 0 , 1 ) } } { \mathbf { E } } [ f _ { \mathbf { w } } ^ { \prime } ( z ) p ( z \mathbf { v } + \mathbf { w } ) ] = \displaystyle \int _ { \mathbb { R } } f _ { \mathbf { w } } ^ { \prime } ( z ) p _ { \mathbf { w } } ( z ) \phi ( z ) d z = - \int _ { \mathbb { R } } f _ { \mathbf { w } } ( z ) ( p _ { \mathbf { w } } ( z ) \phi ( z ) ) ^ { \prime } ~ } & { } \\ { = - \displaystyle \int _ { \mathbb { R } } f _ { \mathbf { w } } ( z ) ( p _ { \mathbf { w } } ^ { \prime } ( z ) \phi ( z ) - z p _ { \mathbf { w } } ( z ) \phi ( z ) ) = \displaystyle \int _ { \mathbb { R } } f _ { \mathbf { w } } ( z ) ( z p _ { \mathbf { w } } ( z ) - p _ { \mathbf { w } } ^ { \prime } ( z ) ) \phi ( z ) } \\ { = \underset { z \sim \sqrt { ( 0 , 1 ) } } { \mathbf { E } } [ f _ { \mathbf { w } } ( z ) M _ { \mathbf { v } } p ( z \mathbf { v } + \mathbf { w } ) ] . } & { } \end{array}
$$

If $f _ { \mathbf { w } } ( z )$ is a step function with finite jump points, then we have $\begin{array} { r } { f _ { \mathbf { w } } ^ { \prime } ( z ) = \sum _ { s \in S _ { \mathbf { w } } } \Delta _ { f _ { \mathbf { w } } } ( s ) \delta _ { s } ( z ) } \end{array}$ , which gives

$$
\mathbf { \Psi } _ { z \sim \mathcal { N } ( 0 , 1 ) } ^ { \mathbf { E } } [ f _ { \mathbf { w } } ( z ) M _ { \mathbf { v } } p ( z \mathbf { v } + \mathbf { w } ) ] = \sum _ { s \in S _ { \mathbf { w } } } \Delta _ { f _ { \mathbf { w } } } ( s ) p ( s \mathbf { v } + \mathbf { w } ) \phi ( s ) .
$$

Lemma 5.6 suggests if at every boundary point, the sign of $p$ matches with the sign of the derivative of $f ,$ then the correlation between $f$ and $M _ { \mathbf { v } } p$ must be strictly positive. However, even if $\Delta _ { f _ { \mathbf { w } } } ( s ) p ( s \mathbf { v } + \mathbf { w } ) > 0$ is very large for every boundary point s, it does not implies $f ( \mathbf { x } )$ has a large correlation with $M _ { \mathbf { v } } p ,$ because after weighted by $\phi ( s )$ , the whole sum could still be small. To give a quantitative lower bound for the correlation, we have to connect these weights with ${ \mathrm { R a n d } } _ { f } ( \mathbf { v } )$ Specifically, we give the following lemma using the Gaussian isoperimetric inequality.

Lemma 5.7. Let $f ( \mathbf { x } ) : \mathbb { R } ^ { d }  \{ 0 , 1 \}$ be an intersection of k degree-2 PTFs. Let $\mathbf { v } \in \mathbb { S } ^ { d - 1 }$ be a unit vector. For $\mathbf { w } \perp \mathbf { v } $ , write $f _ { \mathbf { w } } ( s ) = f ( s \mathbf { v } + \mathbf { w } ) : \mathbb { R } \to \{ 0 , 1 \}$ and let $S _ { \mathbf { w } } \subseteq \mathbb { R }$ be the set of jump points of $f _ { \mathbf { w } }$ . There is a constant $C > 1$ , such that for every $\alpha \in ( 0 , 1 )$ and for every $\mathbf { v } \in \mathbb { S } ^ { d - 1 }$ , if Rand $| { \bf \nabla } _ { f } ( { \bf v } ) > \alpha$ , then

$$
\underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } \left[ \sum _ { s \in S _ { \mathbf { w } } } \phi ( s ) \right] > \alpha / C .
$$

To prove Lemma 5.7, we will make use of the following 1-dimensional Gaussian isoperimetric inequality.

Fact 5.8. Let $S \subseteq \mathbb { R }$ be a set such that $| \partial S | < \infty$ . Denote by $p _ { S } = \mathbf { P r } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ z \in S ]$ . Then

$$
\sum _ { s \in \partial S } \phi ( s ) \geq \phi ( \Phi ^ { - 1 } ( p _ { S } ) ) ,
$$

where $\phi$ is the density function of a 1-dimensional Gaussian distribution and $\begin{array} { r } { \Phi ( t ) = \int _ { - \infty } ^ { t } \phi ( z ) d z } \end{array}$

Proof of Lemma 5.7. We first show that there exists a constant $c > 0$ such that for every $p \in [ 0 , 1 ]$ $\phi ( \Phi ^ { - 1 } ( p ) ) \geq c p ( 1 - p )$ . Without loss of generality, we assume that $p \leq 1 / 2$ . Let $s \leq 0$ such that $\Phi ( s ) = p$ . We show that $\phi ( s ) \geq c p ( 1 - p )$ . We consider two cases. If $| s | \leq 1$ , then we have

$$
\phi ( s ) \geq \phi ( 1 ) \geq 2 \phi ( 1 ) / 2 \geq 2 \phi ( 1 ) p .
$$

$\operatorname { I f } | s | > 1$ , then it holds $\Phi ( s ) \leq \phi ( s ) / s$ , which gives $\phi ( s ) \geq s \Phi ( s ) \geq p$ . This implies that for every $p \in ( 0 , 1 )$ , it holds $\phi ( \Phi ^ { - 1 } ( p ) ) \geq 2 \phi ( 1 ) p ( 1 - p )$

By Fact 5.8, we know that

$$
\begin{array} { r l } & { \displaystyle \sum _ { s \in S _ { \mathbf { w } } } \phi ( s ) \geq \phi ( \Phi ^ { - 1 } ( \underbrace { \mathbf { P r } } _ { \mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ f _ { \mathbf { w } } ( \mathbf { z } ) = 1 ] ) ) } \\ & { \qquad \geq 2 \phi ( 1 ) ( \underbrace { \mathbf { P r } } _ { \mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ f _ { \mathbf { w } } ( \mathbf { z } ) = 1 ] ) ( 1 - \underbrace { \mathbf { P r } } _ { \mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ f _ { \mathbf { w } } ( \mathbf { z } ) = 1 ] ) } \\ & { \qquad = 2 \phi ( 1 ) \mathbf { V a r } _ { \mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ f _ { \mathbf { w } } ( \mathbf { z } ) ] } \end{array}
$$

By taking expectation over w and Proposition 2.2, we get

$$
\underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ \sum _ { s \in S _ { \mathbf { w } } } \phi ( s ) ] \geq \phi ( 1 ) \mathrm { R a n d } _ { f } ( \mathbf { v } ) > \phi ( 1 ) \alpha .
$$

Lemma 5.7 implies that if ${ \mathrm { R a n d } } _ { f } ( \mathbf { v } )$ is large for some direction $\mathbf { v } ,$ then in expectation of weighted sum of $\phi ( s )$ over the boundary points of $f _ { \mathbf { w } }$ is not too small. This implies any $p$ such that $\Delta _ { f _ { \mathbf { w } } } ( s ) p ( s \mathbf { v } + \mathbf { w } ) > 0$ is non-trivially large for most of the boundary points; then $M _ { \mathbf { v } } p$ can be used as the certificate polynomial.

With this motivation, before giving the construction of our certificate polynomial, we first characterize the structure of boundary points of a one-dimensional intersection of $\mathrm { P T F }$

Fact 5.9. Let $f ( z ) = \mathbb { 1 } \{ p _ { 1 } ( z ) \geq t _ { 1 } , \ldots , p _ { k } ( z ) \geq t _ { k } \}$ , where for $i \in [ k ] , p _ { i } ( z )$ is a 1 dimensional degree-2 polynomial and $t _ { i } \in \mathbb { R }$ , be an intersection of k degree-2 PTFs over R. Let S be the set of jump points of $f$ . Then S satisfies the following properties.

1. For each $s \in S$ , let ${ C _ { s } = \{ p _ { i } \ | \ p _ { i } ( s ) - t _ { i } = 0 , i \in [ k ] \} }$ , then $C _ { s } \neq \varnothing$ and $p _ { i } ( s ) - t _ { i } > 0$ for every $p _ { i } \notin C _ { s }$

2. For each $s \in S$ , and for each $p _ { i } \in C _ { s }$ with $p _ { i } ^ { \prime } ( s ) \neq 0 , \Delta _ { f } ( s ) = \mathrm { s i g n } ( p _ { i } ^ { \prime } ( s ) )$

Proof of Fact 5.9. To prove the first item, we first suppose that $C _ { s } = \emptyset$ , which implies that $q _ { i } ( s ) - t _ { i } \neq$ 0 for each $i \in [ k ]$ . By continuity, there exists a neighborhood of $s ,$ , such that for every z in the neighborhood and for every $i \in [ k ] , \operatorname { s i g n } ( q _ { i } ( z ) - t _ { i } ) = \operatorname { s i g n } ( q _ { i } ( s ) - t _ { i } )$ . This means f does not change its sign at s, which gives a contradiction. Similarly, assume there exists some $q _ { i } \notin C _ { s }$ such that $q _ { i } ( s ) - t _ { i } < 0$ . Then there exists a neighborhood of s such that for every z in the neighborhood, $q _ { i } ( z ) - t _ { i } < 0$ , which implies that $f ( z ) = 0$ and gives a contradiction.

We next prove the second item. We first show that for every $q _ { i } \in C _ { s }$ , either $q _ { i } ^ { \prime } ( s ) = 0$ or sign $( q _ { i } ^ { \prime } ( s ) ) = \mathrm { s i g n } ( q _ { i } ^ { \prime } ( s ) )$ for some other $q _ { j } \in C _ { s }$ . Suppose not, then there exists $q _ { i } , q _ { j } \in C _ { s }$ such that $q _ { i } ^ { \prime } ( s ) q _ { j } ^ { \prime } ( s ) < 0$ . Since $q _ { i } ( s ) - t _ { i } = q _ { j } ( s ) - t _ { j } = 0$ . By continuity, this implies that there exists a small neighborhood of $s$ such that sign $( q _ { i } ( z ) - t _ { i } ) \mathrm { s i g n } ( q _ { j } ( z ) - t _ { j } ) < 0$ for every $z$ in the neighborhood. This implies that $f$ has no sign change at s. Now since $q _ { i } ( s ) - t _ { i } > 0$ , for every $q _ { i } \notin C _ { s }$ and $q _ { i } ^ { \prime } ( s )$ has the same sign for $q _ { i } \in C _ { s }$ , we know that $\Delta _ { f } ( s ) = \mathrm { s i g n } ( q _ { i } ^ { \prime } ( s ) )$ , for a $q _ { i } ^ { \prime } ( s ) \neq 0$ □

## 5.2 Certificate Polynomials for Generic Instance and Root Decompositions

According to Fact 5.9, we know that once we fix some $\textbf { w } \perp \textbf { v }$ , the decision boundaries of $f _ { \mathbf { w } }$ is a collection of roots of the defining polynomials $q _ { i } = p _ { i } - t _ { i }$ . Furthermore, for each boundary point $s ,$ we can cluster these $k$ polynomials into two groups. The first group of polynomials has s as their roots, and furthermore their derivatives at s have the same sign of $\Delta _ { f _ { \mathbf { w } } } ( s )$ . The other group of polynomials are strictly positive at s.

Based on this observation, we design a certificate polynomial, whose sign of value at every decision boundary of $f _ { \mathbf { w } }$ , for every w, matches $\Delta _ { f _ { \mathbf { w } } } ( s )$ . Specifically, if we consider polynomial $\begin{array} { r } { Q _ { \mathbf { v } } ( \mathbf { x } ) : = \partial _ { \mathbf { v } } ( \prod _ { i = 1 } ^ { k } q _ { i } ( \mathbf { x } ) ) = \sum _ { i = 1 } ^ { k } \prod _ { j \neq i } q _ { j } ( \mathbf { x } ) \partial _ { \mathbf { v } } q _ { i } ( \mathbf { x } ) } \end{array}$ , then at each $\mathbf { x } = s \mathbf { v } + \mathbf { w }$ , where s is a boundary of $f _ { \mathbf { w } }$ , each of the k terms in the summation is either 0 or has the same sign as $\Delta _ { f _ { \mathbf { w } } } ( s )$ . In particular, if a boundary point s is a root of a single $q _ { i }$ and $q _ { i }$ does not have a double root at $s ,$ then $Q _ { \mathbf { v } }$ gives a strictly positive contribution to the correlation between $f$ and $M _ { \mathbf { v } } Q _ { \mathbf { v } }$

However, we need to derive a quantitative lower bound for the correlation in order to use it to design an algorithm. To do this, we introduce parameters $\eta _ { d } , \eta _ { c } \in ( 0 , 1 ) . ~ \eta _ { d }$ is used to quantify the derivative of $q _ { i }$ at its root s. Intuitively, $\mathrm { i f } \ | \partial _ { \mathbf { v } } q _ { i } |$ is small at $s$ (and the coeficient of $z ^ { 2 }$ in $q _ { i } ( z \mathbf { v } + \mathbf { w } )$ is bounded away from zero), then there should be another root of $q _ { i }$ around s and s is approximately a double root of $q _ { i } , \eta _ { c }$ is used to quantify the value of $q _ { i }$ at $s ,$ where s is not a root of $q _ { i }$ but a root of a diferent defining polynomial $q _ { j }$ . The intuition here is that if $| q _ { i } |$ is tiny at $s ,$ then there must be a root of $q _ { i }$ around s and s a an approximately root shared by both $q _ { i }$ and $q _ { j }$ . Given this intuition, we can partition the boundary points $S _ { \mathbf { w } }$ of $f _ { \mathbf { w } }$ into three sets.

Definition 5.10. Let $f ( \mathbf { x } ) = g ( \mathbf { w } ) \mathbb { 1 } \{ p _ { 1 } ( \mathbf { x } ) \ge t _ { 1 } , \dots , p _ { k } ( \mathbf { x } ) \ge t _ { k } \} : \mathbb { R } ^ { d } \to \{ 0 , 1 \}$ , where for $i \in [ k ]$ $p _ { i }$ is a degree-2 polynomial with $\mathbf { E _ { x \sim \mathcal { N } ( 0 , I ) } } [ p _ { i } ( \mathbf { x } ) ] = 0 , \mathbf { V a r _ { x \sim \mathcal { N } ( 0 , I ) } } [ p _ { i } ( \mathbf { x } ) ] = 1 , t _ { i } \in \mathbb { R }$ and $g ( \mathbf { w } )$ $\mathbf { v } ^ { \perp } \to \{ 0 , 1 \} , \mathbf { v } \in \mathbb { S } ^ { d - 1 }$ . For every $\mathbf { w } \perp \mathbf { v } ,$ denote by $q ^ { ( \mathbf { w } ) } ( z ) = q ( z \mathbf { v } + \mathbf { w } )$ $F o r \ : i \in [ k ]$ , denote by $q _ { i } ( \mathbf { x } ) = p _ { i } - t _ { i }$ and $Q _ { \mathbf { v } } ( \mathbf { x } ) = \partial _ { \mathbf { v } } ( \prod _ { i = 1 } ^ { k } q _ { i } )$ , where $\mathbf { v } \in \mathbb { S } ^ { d - 1 }$ . For $\eta _ { d } , \eta _ { c } \in ( 0 , 1 )$ , $\mathbf { w } \perp \mathbf { v }$ , we define

1. $G _ { \mathbf { w } } ( \eta _ { d } , \eta _ { c } ) \subseteq S _ { \mathbf { w } }$ to be the set of jump points s of $f _ { \mathbf { w } }$ such that there exists some $i \in [ k ]$ such that $\left| ( q _ { i } ^ { ( \mathbf { w } ) } ) ^ { \prime } ( s ) \right| \geq \eta _ { d }$ and for every $j \neq i , q _ { j } ^ { ( \mathbf { w } ) } ( s ) \geq \eta _ { c }$

2. $B _ { \mathbf { w } } ( \eta _ { d } , \eta _ { c } ) \subseteq S _ { \mathbf { w } }$ to be the set of jump points s of $f _ { \mathbf { w } }$ , such that there exists some $i \in [ k ]$ , such that $q _ { i } ^ { ( \mathbf { w } ) } ( s ) = 0$ but $\left| ( q _ { i } ^ { ( \mathbf { w } ) } ) ^ { \prime } ( s ) \right| \leq \eta _ { d }$

$$
\begin{array} { r } { \mathcal { S } _ { \mathbf { * } } \left( \eta _ { d } , \eta _ { c } \right) = { S } _ { \mathbf { w } } \ \backslash \ \left( { G } _ { \mathbf { w } } ( \eta _ { d } , \eta _ { c } ) \cup { B } _ { \mathbf { w } } ( \eta _ { d } , \eta _ { c } ) \right) } \end{array}
$$

By Lemma $5 . 7 ,$ , we know that if Rand ${ \bf \nabla } _ { f } ( { \bf v } )$ is non-trivially large, then $\begin{array} { r } { \mathbf { E } _ { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \sum _ { s \in G _ { \mathbf { w } } ( \eta _ { d } , \eta _ { c } ) } \phi ( s ) + } \end{array}$ $\begin{array} { r } { \sum _ { s \in B _ { \mathbf { w } } ( \eta _ { d } , \eta _ { c } ) } \phi ( s ) + \sum _ { s \in C _ { \mathbf { w } } ( \eta _ { d } , \eta _ { c } ) } \phi ( s ) ] } \end{array}$ must be non-trivially large. By our above discussion, if the contribution from roots from $G _ { \mathbf { w } } ( \eta _ { d } , \eta _ { c } )$ is not too small, $M _ { \mathbf { v } } Q _ { \mathbf { v } }$ gives a nontrivially large correlation with respect to $f .$ Formally, we give the following Lemma 5.11.

Lemma 5.11 (Certificate Polynomials for Generic Intersections of PTFs). Let $f ( \mathbf { x } ) = g ( \mathbf { w } ) \mathbb { 1 } \{ p _ { 1 } ( \mathbf { x } ) \geq$ $t _ { 1 } , \dots , p _ { k } ( \mathbf { x } ) \geq t _ { k } \} : \mathbb { R } ^ { d } \to \{ 0 , 1 \}$ , where $f o r i \in [ k ] , p _ { i }$ is a degree-2 polynomial with $\mathbf { E _ { x \sim \mathcal { N } ( 0 , I ) } } [ p _ { i } ( \mathbf { x } ) ] =$ $0 , { \mathbf { V } } \mathbf { a r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p _ { i } ( \mathbf { x } ) ] = 1 , t _ { i } \in \mathbb { R }$ and $g ( \mathbf { w } ) : \mathbf { v } ^ { \perp } \to \{ 0 , 1 \}$ , $\textbf { v } \in \ \mathbb { S } ^ { d - 1 }$ For $i ~ \in ~ [ k ]$ , denote by $q _ { i } ( { \bf x } ) = p _ { i } - t _ { i }$ and $Q _ { \mathbf { v } } ( \mathbf { x } ) \ = \ \partial _ { \mathbf { v } } ( \prod _ { i = 1 } ^ { k } q _ { i } )$ , where $\textbf { v } \in \ \mathbb { S } ^ { d - 1 }$ . For $\eta _ { d } , \eta _ { c } \in ( 0 , 1 )$ $\textbf { w } \perp \textbf { v }$ , define $G _ { \mathbf { w } } ( \eta _ { d } , \eta _ { c } ) \ \subseteq \ S _ { \mathbf { w } }$ such that for each $s \in G _ { \mathbf { w } } ( \eta _ { d } , \eta _ { c } )$ , there exists some $i \in [ k ]$ such that $\left| ( q _ { i } ^ { ( \mathbf { w } ) } ) ^ { \prime } ( s ) \right| \geq \eta _ { d }$ and for every $j \neq i , q _ { j } ^ { ( \mathbf { w } ) } ( s ) \geq \eta _ { c }$ . Suppose that $\begin{array} { r } { \mathbf { E } _ { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \sum _ { s \in G _ { \mathbf { w } } ( \eta _ { d } , \eta _ { c } ) } \phi ( s ) ] \ge \alpha } \end{array}$ 2 then

$$
\begin{array} { r } { \underline { { \mathbf { E } } } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ f ( \mathbf { x } ) M _ { \mathbf { v } } Q _ { \mathbf { v } } ( \mathbf { x } ) ] \geq \eta _ { d } \eta _ { c } ^ { k - 1 } \alpha . } \end{array}
$$

Proof of Lemma 5.11. For every $\mathbf { w } \perp \mathbf { v } ,$ , denote by $f _ { \mathbf { w } } ( z ) = f ( z \mathbf { v } + \mathbf { w } )$ and define $S _ { \mathbf { w } } \subseteq \mathbb { R }$ , the set of jump points of $f _ { \mathbf { w } }$ . Notice that $\begin{array} { r } { \mathbf { E _ { x \sim \mathcal { N } ( 0 , \mathbf { I } ) } } [ f ( \mathbf { x } ) M _ { \mathbf { v } } Q _ { \mathbf { v } } ( \mathbf { x } ) ] = \mathbf { E _ { w \sim \mathcal { N } ( 0 , \mathbf { I } ) } } [ \mathbf { E } _ { z \sim \mathcal { N } ( 0 , 1 ) } [ f _ { \mathbf { w } } ( z ) M _ { \mathbf { v } } Q _ { \mathbf { v } } ( z \mathbf { v } + \mathbf { I } _ { 0 } ) ] . } \end{array}$ $\mathbf { w } ) ] ]$ . By Lemma 5.6, we have

$$
\begin{array} { l } { \underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ f ( \mathbf { x } ) M _ { \mathbf { v } } Q _ { \mathbf { v } } ( \mathbf { x } ) ] = \underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ \underset { z \sim \mathcal { N } ( 0 , \mathbf { 1 } ) } { \mathbf { E } } [ f _ { \mathbf { w } } ( z ) M _ { \mathbf { v } } Q _ { \mathbf { v } } ( z \mathbf { v } + \mathbf { w } ) ] ] } \\ { = \underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ \underset { s \in S _ { \mathbf { w } } } { \sum } \Delta _ { f _ { \mathbf { w } } } ( s ) Q _ { \mathbf { v } } ( s \mathbf { v } + \mathbf { w } ) \phi ( s ) ] } \\ { = \underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ \underset { s \in S _ { \mathbf { w } } } { \sum } \underset { i = 1 } { \overset { k } { \sum } } ( q _ { i } ^ { ( \mathbf { w } ) } ) ^ { \prime } ( s ) \underset { j \neq i } { \prod } q _ { j } ^ { ( \mathbf { w } ) } ( s ) \Delta _ { f _ { \mathbf { w } } } ( s ) \phi ( s ) ] . } \end{array}
$$

By Fact 5.9, for every $s \in S _ { \mathbf { w } }$ and for every $\begin{array} { r } { i \in [ k ] , ( q _ { i } ^ { ( \mathbf { w } ) } ) ^ { \prime } ( s ) \prod _ { j \neq i } q _ { j } ^ { ( \mathbf { w } ) } ( s ) \Delta _ { f _ { \mathbf { w } } } ( s ) \geq 0 } \end{array}$ . Furthermore, the strict inequality holds if and only if $( q _ { i } ^ { ( \mathbf { w } ) } ) ^ { \prime } ( s ) \neq 0$ and $q _ { j } ^ { ( \mathbf { w } ) } ( s ) > 0 , j \neq i$ , in which case, $( q _ { i } ^ { ( \mathbf { w } ) } ) ^ { \prime } ( s ) \Delta _ { f _ { \mathbf { w } } } ( s ) = \left| ( q _ { i } ^ { ( \mathbf { w } ) } ) ^ { \prime } ( s ) \right|$

This implies that

$$
\underset { \substack { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } } { \mathbf { E } } [ f ( \mathbf { x } ) M _ { \mathbf { v } } Q _ { \mathbf { v } } ( \mathbf { x } ) ] \geq \underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ \sum _ { s \in G _ { \mathbf { w } } ( \eta _ { d } , \eta _ { c } ) } \sum _ { i = 1 } ^ { k } \left| ( q _ { i } ^ { ( \mathbf { w } ) } ) ^ { \prime } ( s ) \right| \prod _ { j \neq i } q _ { j } ^ { ( \mathbf { w } ) } ( s ) \phi ( s ) ] \geq \eta _ { d } \eta _ { c } ^ { k - 1 } \alpha .
$$

Although for a generic good instance $f ,$ the simple construction of $Q _ { \mathbf { v } }$ can be used as a certificate polynomial, $Q _ { \mathbf { v } }$ can easily fail to give enough correlation for some very natural instance. For example, even if $f ( \mathbf { x } )$ is a simple intersection of two PTFs defined by $q _ { 1 } ( \mathbf { x } ) , q _ { 2 } ( \mathbf { x } )$ , the coeficients of $q _ { 1 } , q _ { 2 }$ could be approximately parallel, which implies almost no roots will be in $G _ { \bf w } ( \eta _ { d } , \eta _ { c } )$ and $M _ { \mathbf { v } } Q _ { \mathbf { v } }$ gives nearly zero correlation with respect to $f .$ . For larger $k ,$ the configuration of these k polynomials can be even more complicated. Thus, the most challenging part of proving Theorem 5.1 will be to construct certificate polynomials that work even if no roots are generic good. Our proof will provide analysis for $B _ { \mathbf { w } } ( \eta _ { d } , \eta _ { c } ,$ )(double roots) and $C _ { \mathbf { w } } ( \eta _ { d } , \eta _ { c } )$ (commonly sharing roots) separately. In Subsection (5.3), we use the discriminant of a quadratic as a tool to analyze the case where double roots have enough Gaussian mass. In Subsection (5.4), we develop a robust factorization lemma for quadratic functions under Gaussian distribution and use it to analyze the case where commonly sharing roots have enough Gaussian mass. Finally, In Subsection (5.5), we combine all the techniques developed in Section 5 to prove Theorem 5.1.

## 5.3 Analysis for Double Roots

Recall our discussion in the previous subsection, the generic polynomial that we construct will only give a tiny correlation, if a defining polynomial has a small derivative at its root. In this section, we provide analysis for this case. For a fixed polynomial $q ( \mathbf { x } ) , \mathbf { v } \in \mathbb { S } ^ { d - 1 }$ , and $\mathbf { w } \perp \mathbf { v } ,$ we consider the one dimensional $q ^ { ( \mathbf { w } ) } ( z ) = q ( z \mathbf { v } + \mathbf { w } )$ . The observation is that we can view $q ^ { ( \mathbf { w } ) } ( z )$ as a one-dimensional quadratic function along v direction, whose coeficients are polynomials in w. Specifically, we can always write $q ^ { ( \mathbf { w } ) } ( z ) = a z ^ { 2 } + b ( \mathbf { w } ) z + c ( \mathbf { w } )$ , where $\deg ( b ) \leq 1$ and $\deg ( c ) \leq 2$ Following elementary algebra, for s, a root of $q ^ { ( \mathbf { w } ) }$ , the square of the derivative of $q ^ { ( \mathbf { w } ) } ( s )$ is exactly equal to the discriminant of $q ^ { ( \mathbf { w } ) }$ . i.e. $\Delta ( \mathbf { w } ) : = b ( \mathbf { w } ) ^ { 2 } - 4 a c ( \mathbf { w } )$ . This implies that $\mathrm { i f } \ \big | ( q ^ { ( \mathbf { w } ) } ) ^ { \prime } ( s ) \big |$ is large, then $\Delta ( \mathbf { w } )$ must be large. On the other hand, if $\Delta ( \mathbf { w } )$ is small, then after normalization, the two roots of $q ^ { ( \mathbf { w } ) }$ must be very close to each other, which implies that by introducing a small error, $\mathbb { 1 } \{ q ^ { ( \mathbf { w } ) } \geq 0 \}$ is nearly 0 or 1.

Although, such an analysis is for a single fixed w, since $\Delta ( \mathbf { w } )$ is a polynomial of degree at most 2, Gaussian concentration and anti-concentration properties enable us to extend the analysis to all layers w. Specifically, in Lemma 5.12, we show that is $\| \Delta ( \mathbf { w } ) \|$ is not very small, then $\begin{array} { r } { \mathbf { E } _ { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \sum _ { s \in B _ { \mathbf { w } } ( \eta _ { d } , \eta _ { c } ) } \phi ( s ) ] } \end{array}$ must be very small and can be ignored. On the other hand, in Lemma $5 . 1 3 ,$ , we show that if $\| \Delta ( \mathbf { w } ) \|$ is small, then 1 $\{ q ( \mathbf { x } ) \geq 0 \}$ is close to some function $g ( \mathbf { w } )$ that only depends on $\mathbf { v } ^ { \perp }$ and thus can be removed from the intersection.

Lemma 5.12. Let $p ( \mathbf { x } )$ be a degree-2 polynomial such that $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p ( \mathbf { x } ) ] = 0$ and $\begin{array} { r l } {  { \mathbf { V a r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p ( \mathbf { x } ) ] = } } \end{array}$ 1. Let t ∈ R and $q ( { \bf x } ) : = p ( { \bf x } ) - t$ . For $\mathbf { v } \in \mathbb { S } ^ { d - 1 }$ , write $\mathbf { x } = z \mathbf { v } + \mathbf { w }$ , where $z = \mathbf { v } \cdot \mathbf { x }$ and $\mathbf { w } = \mathrm { p r o j } _ { \mathbf { v } ^ { \perp } } ( \mathbf { x } )$ For every $\mathbf { w } \perp \mathbf { v } ,$ denote by $q ^ { ( \mathbf { w } ) } ( z ) = q ( z \mathbf { v } + \mathbf { w } )$ . Write $q ^ { ( \mathbf { w } ) } ( z ) = a z ^ { 2 } + b ( \mathbf { w } ) z + c ( \mathbf { w } )$ , where $a \in \mathbb { R }$ $b ( \mathbf { w } ) , c ( \mathbf { w } )$ are polynomial over $\mathbf { w } ,$ with $\deg ( b ) \leq 1$ and $\deg ( c ) \leq 2$ . Let $\Delta ( \mathbf { w } ) = b ^ { 2 } ( \mathbf { w } ) - 4 a c ( \mathbf { w } )$ . For every $\alpha \in ( 0 , 1 / 4 )$ and $\eta _ { d } \in ( 0 , 1 )$ , suppose that $\mathbf { E } _ { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \Delta ( \mathbf { w } ) ^ { 2 } ] > \alpha$ , then $\mathbf { P r } _ { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \exists s , q ^ { ( \mathbf { w } ) } ( s ) =$ $0 , \big | ( q ^ { ( \mathbf { w } ) } ) ^ { \prime } ( s ) \big | \le \eta _ { d } \big ] \le O ( \eta _ { d } / \alpha ^ { 1 / 4 } )$

Proof of Lemma 5.12. Denote by $D ( \mathbf { w } , z ) = ( q ^ { ( \mathbf { w } ) } ) ^ { \prime } ( z )$ . For every w $\perp { \bf v }$ and for every $z \in \mathbb { R }$ , we have $D ( \mathbf { w } , z ) = 2 a z + b$ and thus $D ^ { 2 } ( \mathbf { w } , z ) = 4 a ^ { 2 } z ^ { 2 } + b ^ { 2 } ( \mathbf { w } ) + 4 a b ( \mathbf { w } ) z$ . If $q ^ { ( \mathbf { w } ) } ( z ) = 0$ , then we have

$$
D ^ { 2 } ( \mathbf { w } , z ) = b ^ { 2 } - 4 a c ( \mathbf { w } ) = \Delta ( \mathbf { w } ) .
$$

This implies that if $s _ { 1 } , s _ { 2 }$ are two roots of $q ^ { ( \mathbf { w } ) } ( z )$ , then $\left| ( q ^ { ( \mathbf { w } ) } ) ^ { \prime } ( s _ { 1 } ) \right| = \left| ( q ^ { ( \mathbf { w } ) } ) ^ { \prime } ( s _ { 2 } ) \right| = \sqrt { \Delta ( \mathbf { w } ) }$ . This implies that

$$
\begin{array} { r } { \quad \quad \mathbf { P r } _ { \sim \mathcal { N } ( 0 , \mathbf { I } ) } \big [ \exists s , q ^ { ( \mathbf { w } ) } ( s ) = 0 , \big | ( q ^ { ( \mathbf { w } ) } ) ^ { \prime } ( s ) \big | \le \eta _ { d } \big ] \le \underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { P r } } [ | \Delta ( \mathbf { w } ) | \le \eta _ { d } ^ { 2 } ] \le O ( \sqrt { \eta _ { d } ^ { 2 } / \| \Delta ( \mathbf { w } ) \| } ) \le O ( \eta _ { d } / \alpha ^ { 1 / 4 } ) . } \end{array}
$$

Lemma 5.13. Let $p ( \mathbf { x } )$ be a degree-2 polynomial such that $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p ( \mathbf { x } ) ] = 0$ and $\mathbf { V a r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p ( \mathbf { x } ) ] =$ 1. Let $t \in \mathbb { R }$ and $q ( \mathbf { x } ) : = p ( \mathbf { x } ) - t . \ F o r { \mathbf { v } } \in \mathbb { S } ^ { d - 1 }$ , write $\mathbf { x } = z \mathbf { v } + \mathbf { w } ,$ where $z = \mathbf { v } \cdot \mathbf { x }$ and $\mathbf { w } = \mathrm { p r o j } _ { \mathbf { v } ^ { \perp } } ( \mathbf { x } )$ $F o r$ every $\mathbf { w } \perp \mathbf { v } ,$ denote by $q ^ { ( \mathbf { w } ) } ( z ) = q ( z \mathbf { v } + \mathbf { w } )$ . Write $q ^ { ( \mathbf { w } ) } ( z ) = a z ^ { 2 } + b ( \mathbf { w } ) z + c ( \mathbf { w } )$ , where $a \in$ R and $b ( \mathbf { w } ) , c ( \mathbf { w } )$ are polynomials over w with $\deg ( b ) \leq 1$ and $\deg ( c ) \leq 2$ . Let $\Delta ( \mathbf { w } ) = b ( \mathbf { w } ) ^ { 2 } - 4 a c ( \mathbf { w } )$ For every $\alpha \in ( 0 , 1 / 4 )$ , suppose that $\mathbf { E } _ { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { \mathbf { v } \perp } ) } [ \Delta ( \mathbf { w } ) ^ { 2 } ] \leq \alpha$ . Then there exists a Boolean function $g : \mathbf { v } ^ { \perp } \to \{ 0 , 1 \}$ such that

$$
\Pr _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ g ( \mathbf { w } ) \neq \mathbb { 1 } \{ q ( \mathbf { x } ) \geq 0 \} ] \leq \tilde { O } ( \alpha ^ { 1 / 1 6 } ) .
$$

Proof of Lemma 5.13. We first normalize q rather than truncate the threshold t. Since p has mean zero and variance one, $s : = \| q \| = { \sqrt { 1 + t ^ { 2 } } } \geq 1$ . Replacing q by $q / s$ replaces $a , b , c$ by $a / s , b / s , c / s$ respectively, and its discriminant by $\Delta / s ^ { 2 }$ . This positive rescaling does not change $\mathbb { 1 } \{ q \ge 0 \}$ , and $\begin{array} { r } { \mathbf { E } _ { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { \mathbf { v } ^ { \perp } } ) } \left\lceil \left( \frac { \Delta ( \mathbf { w } ) } { s ^ { 2 } } \right) ^ { 2 } \right\rceil \leq \alpha } \end{array}$ . Thus, reusing the same notation for the rescaled quantities, we may assume that $\bar { \| } q \| = 1 , \| \bar { \Delta } \| \leq \alpha ^ { 1 / 2 }$

Set $\lambda = \alpha ^ { 1 / 3 }$ . We consider two cases. In each case, we construct a polynomial Q of degree at most two and a Boolean function $g ( \mathbf { w } )$ such that $\mathbb { 1 } \{ Q ( \mathbf { x } ) \geq 0 \} = g ( \mathbf { w } )$ almost surely and $\| q - Q \| \leq C \alpha ^ { 1 / 6 }$ , where $C > 0$ is a universal constant.

Case 1: $| a | \geq \lambda$ . Completing the square gives

$$
q ^ { ( \mathbf { w } ) } ( z ) = a \left( z + \frac { b ( \mathbf { w } ) } { 2 a } \right) ^ { 2 } - \frac { \Delta ( \mathbf { w } ) } { 4 a } .
$$

Define

$$
Q ( z \mathbf { v } + \mathbf { w } ) : = a \left( z + { \frac { b ( \mathbf { w } ) } { 2 a } } \right) ^ { 2 } , \qquad g ( \mathbf { w } ) : = \mathbb { 1 } \{ a > 0 \} .
$$

Since b is afine, Q has degree at most two. Moreover, $a \neq 0$ , so the square vanishes only on a Gaussian-null set. Hence $\mathbb { 1 } \{ Q ( \mathbf { x } ) \geq 0 \} = g ( \mathbf { w } )$ almost surely. Finally,

$$
\| q - Q \| = \frac { \| \Delta \| } { 4 | a | } \le \frac { \alpha ^ { 1 / 2 } } { 4 \lambda } = \frac { 1 } { 4 } \alpha ^ { 1 / 6 } .
$$

Case $\ 2 \colon \ | a | < \lambda$ . Define $d ( \mathbf { w } ) : = c ( \mathbf { w } ) + a = \mathbf { E } _ { z \sim \mathcal { N } ( 0 , 1 ) } [ q ( z \mathbf { v } + \mathbf { w } ) ]$ . Then

$$
q ( z \mathbf { v } + \mathbf { w } ) = a ( z ^ { 2 } - 1 ) + b ( \mathbf { w } ) z + d ( \mathbf { w } ) .
$$

Gaussian orthogonality gives $1 = \| q \| ^ { 2 } = 2 a ^ { 2 } + \| b \| ^ { 2 } + \| d \| ^ { 2 }$ . In particular, $\| d \| \leq 1$ , and therefore

$$
\left| \underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { \mathbf { v } ^ { \perp } } ) } { \mathbf { E } } [ c ( \mathbf { w } ) ] \right| \leq \left\| d \right\| + \left| a \right| \leq 1 + \left| a \right| .
$$

Using the discriminant identity $\Delta ( \mathbf { w } ) = b ( \mathbf { w } ) ^ { 2 } - 4 a c ( \mathbf { w } )$ , we obtain

$$
\begin{array} { r l } & { \| b \| ^ { 2 } = \underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { \mathbf { v } ^ { \bot } } ) } { \mathbf { E } } [ \Delta ( \mathbf { w } ) ] + 4 a \underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { \mathbf { v } ^ { \bot } } ) } { \mathbf { E } } [ c ( \mathbf { w } ) ] } \\ & { \quad \quad \leq \| \Delta \| + 4 | a | \underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { \mathbf { v } ^ { \bot } } ) } { \mathbf { E } } [ c ( \mathbf { w } ) ] | } \\ & { \quad \quad \leq \alpha ^ { 1 / 2 } + 4 | a | ( 1 + | a | ) } \\ & { \quad \quad \leq \alpha ^ { 1 / 2 } + 8 \lambda , } \end{array}
$$

where the last inequality uses $| a | < \lambda < 1$

Set

$$
Q ( z \mathbf { v } + \mathbf { w } ) : = d ( \mathbf { w } ) , \qquad g ( \mathbf { w } ) : = \mathbb { 1 } \{ d ( \mathbf { w } ) \geq 0 \} .
$$

Again by Gaussian orthogonality,

$$
\left\| q - Q \right\| ^ { 2 } = 2 a ^ { 2 } + \left\| b \right\| ^ { 2 } \leq 2 \lambda ^ { 2 } + \alpha ^ { 1 / 2 } + 8 \lambda \leq 1 1 \alpha ^ { 1 / 3 } .
$$

Thus $\| q - Q \| \leq \sqrt { 1 1 } \alpha ^ { 1 / 6 }$ , as required.

Converting the perturbation into disagreement. In both cases, $\| q \| = 1$ and $\left\| q - Q \right\| ^ { 2 } \leq$ $C \alpha ^ { 1 / 3 }$ . For every $u > 0$ , Gaussian anticoncentration for degree-two polynomials and Markov’s inequality give

$$
\begin{array} { r l } & { \quad \mathbf { P r } } \\ & { \quad \times \cdots \mathcal { N } ( 0 , \mathbf { I } ) [ 1 \{ q ( \mathbf { x } ) \geq 0 \} \neq g ( \mathbf { w } ) ] } \\ & { \quad \quad = \quad \mathbf { P r } } \\ & { \quad \quad = \quad \mathbf { { r } } \underset { { \mathbf { x } \times N ( 0 , \mathbf { I } ) } } { \mathbf { P r } } [ 1 \{ q ( \mathbf { x } ) \geq 0 \} \neq \mathbf { { 1 } } \{ Q ( \mathbf { x } ) \geq 0 \} ] } \\ & { \quad \quad \leq \quad \mathbf { P r } } \\ & { \quad \quad \quad \leq \quad \mathbf { P r } } \\ & { \quad \quad \quad \leq C \sqrt { u } + \frac { \| q - Q \| ^ { 2 } } { u ^ { 2 } } } \\ & { \quad \quad \quad \leq C \sqrt { u } + \frac { C \alpha ^ { 1 / 3 } } { u ^ { 2 } } . } \end{array}
$$

Taking $u = \alpha ^ { 2 / 1 5 }$ yields

$$
\operatorname* { P r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ 1 \{ q ( \mathbf { x } ) \ge 0 \} \neq g ( \mathbf { w } ) ] \le C \alpha ^ { 1 / 1 5 } .
$$

Together, Lemma 5.12 and Lemma 5.13 imply that by selecting a proper threshold for $\| \Delta \|$ any defining quadratic polynomial of the target function either has nearly no roots in $B _ { \mathbf { w } } ( \eta _ { d } , \eta _ { c } )$ in expectation, or up to a small error, we can remove it from the intersection and absorb it into some function $g ( \mathbf { w } )$ that only depends on w. This implies that if $\begin{array} { r } { \mathbf { E } _ { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \sum _ { s \in G _ { \mathbf { w } } ( \eta _ { d } , \eta _ { c } ) } \phi ( s ) ] } \end{array}$ is small, then most of the roots can be thought to be in $C _ { \mathbf { w } } ( \eta _ { d } , \eta _ { c } )$ . In Subsection (5.4), we provide a systematic way to analyze and construct certificate polynomials in this case.

## 5.4 Robust Factorization of Quadratic Polynomials under Gaussian Distribution

We now turn to the case where a significant portion of the decision boundary arises from roots that are approximately shared by multiple defining polynomials. As discussed in Subsection (5.3), we may assume that a defining polynomial has a not too small derivative at its roots for most of $\mathbf { w } \perp \mathbf { v } .$ However, the generic certificate given by Lemma 5.11 will still give only arbitrarily small correlation, if at a root of one defining polynomial, some other polynomial has nearly zero value, even though the corresponding boundary contributes significantly to the randomized error. In Lemma 5.14, we show that in this case, the two polynomials must have roots that are suficiently close to each other. However, this does not mean we can merge these commonly sharing roots naively into one root, because there dependence on w can be very complicated and even for a fixed w, the arrangement of the 2k roots of the k polynomials can be very complicated. Thus, to handle this case, we need a certificate that takes advantage of the structure of these nearly shared roots rather than treating them as separate boundaries.

Our key observation is that if two polynomials have nearly sharing roots at a non-trivial fraction of layers, $\mathbf { w } \perp \mathbf { v } ,$ then the two polynomials must approximately share some global algebraic structures. Such an observation starts from Euclidean algorithms for finding a greatest common divisor for one-dimensional polynomials. Similar to the previous section, for each fixed $\textbf { w } \perp \textbf { v }$ , and for each $i \in [ k ]$ , we write $q _ { i } ^ { ( \mathbf { w } ) } ( z ) = a _ { i } z ^ { 2 } + b _ { i } ( \mathbf { w } ) z + c _ { i } ( \mathbf { w } )$ as a univariate polynomial in z. Recall that two univariate polynomials have a common root if and only if their resultant (Definition 5.15) vanishes. This implies that if the resultant of $q _ { 1 } ^ { ( \mathbf { w } ) } ( z ) , q _ { 2 } ^ { ( \mathbf { w } ) } ( z )$ is suficiently small, we are able to find univariate afine polynomials $L , \ell _ { 1 } , \ell _ { 2 }$ such that $\left\| q _ { i } ^ { ( \mathbf { w } ) } ( z ) - L \ell _ { i } \right\|$ is small. In fact, in Fact 5.17, we point out the resultant of $q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) }$ as a function of w is a polynomial of degree at most 4. Thus, using Gaussian anti-concentration, in Lemma 5.18, we show that as long as $q _ { 1 } , q _ { 2 }$ have nearly shared roots on a non-trivial fraction of Gaussian layers, the $L _ { 2 }$ norm of their resultant must be small. The hope is that we would like to show in this case we have multivariate afine functions $L , \ell _ { 1 } , \ell _ { 2 }$ such that $q _ { i } \approx L \ell _ { i }$ . However, a direct application of Euclidean algorithm over these two polynomials does not imply this. Although the defining polynomials are univariate quadratics after fixing $\mathbf { w } _ { \mathrm { : } }$ their coeficients are themselves polynomials in w. The Euclidean algorithm therefore naturally produces rational functions of w, which need not correspond to low-degree polynomial factors in the original variables. Indeed, as we will mention later, even if two polynomials share a root on every slice, they need not have a common afine factor.

This motivates our main technical result, Lemma 5.19, which gives a robust factorization lemma for two multivariate quadratics whose resultant is small. In Lemma 5.19, we show that when the resultant is suficiently small, there are only three cases: either one quadratic is nearly independent of v direction, or the two quadratics are nearly parallel or they can be approximately factorized into the form of $L \ell _ { i }$ . If we are in the first two cases, then we can simply remove one defining polynomial from the intersection and work on the rest of the ones. In Subsection (5.5), we will show that once we find some L using the factorization lemma that contributes enough mass for the boundary points, then we can use $L$ as a pivot to cluster the k defining polynomials into two groups to construct a new certificate polynomials that bypass the dificulty of roots sharing.

Root Distance and Resultants To begin with, we start by showing that for a given two quadratics, under suitable normalization, if one polynomial evaluates very small at a root of another polynomial, then it has a nearby root.

Lemma 5.14. For $i \in [ 2 ]$ , let $p _ { i } : \mathbb { R } ^ { d }  \mathbb { R }$ be a degree-2 polynomial with $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p _ { i } ( \mathbf { x } ) ] = 0$ and $\mathbf { V a r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p _ { i } ( \mathbf { x } ) ] = 1$ . For $i \in [ 2 ]$ , denote by $q _ { i } ( \mathbf { x } ) = p _ { i } ( \mathbf { x } ) - t _ { i } , t _ { i } \in \mathbb { R }$ $F o r \textbf { v } \in \mathbb { S } ^ { d - 1 }$ , write $\mathbf { x } = z \mathbf { v } + \mathbf { w }$ , where $z = { \bf v } \cdot { \bf x }$ and $\mathbf { w } = \mathrm { p r o j } _ { \mathbf { v } ^ { \perp } } ( \mathbf { x } )$ . For every $\mathbf { w } \perp \mathbf { v } ,$ denote by $q _ { i } ^ { ( \mathbf { w } ) } ( z ) = q _ { i } ( z \mathbf { v } + \mathbf { w } )$ $L e t \ s _ { 1 } , s _ { 2 }$ be two roots of $q _ { i } ^ { ( \mathbf { w } ) } ( z )$ , if exist and let $t \in \mathbb { R }$ . Suppose that $| s _ { 1 } - t | \leq | s _ { 2 } - t |$ . Let $\eta _ { d } , \eta _ { c } \in ( 0 , 1 )$ . Suppose that $\left| ( q _ { i } ^ { ( \mathbf { w } ) } ) ^ { \prime } ( s _ { 1 } ) \right| \geq \eta _ { d }$ and $\left| q _ { i } ^ { ( \mathbf { w } ) } ( t ) \right| \leq \eta _ { c }$ , then $| s _ { 1 } - t | \leq 2 \eta _ { c } / \eta _ { d }$

Proof of Lemma $5 . 1 4$ . Write $q = q _ { i } ^ { ( \mathbf { w } ) }$ . If q is afine, then $| q ( t ) | = | q ^ { \prime } ( s _ { 1 } ) | | t - s _ { 1 } | \geq \eta _ { d } | t - s _ { 1 } |$ , which gives the desired bound.

Otherwise, write $q ( z ) = a _ { i } ( z - s _ { 1 } ) ( z - s _ { 2 } )$ . By the triangle inequality and the assumption $| s _ { 1 } - t | \leq | s _ { 2 } - t |$

$$
\left| s _ { 1 } - s _ { 2 } \right| \leq \left| s _ { 1 } - t \right| + \left| s _ { 2 } - t \right| \leq 2 \left| s _ { 2 } - t \right| .
$$

Since $| q ^ { \prime } ( s _ { 1 } ) | = | a _ { i } | | s _ { 1 } - s _ { 2 } |$ , we obtain

$$
\begin{array} { l } { { \eta _ { c } \geq | q ( t ) | = | a _ { i } | | s _ { 1 } - t | | s _ { 2 } - t | } } \\ { { \qquad \geq \displaystyle \frac { 1 } { 2 } \left| q ^ { \prime } ( s _ { 1 } ) \right| | s _ { 1 } - t | \geq \displaystyle \frac { \eta _ { d } } { 2 } \left| s _ { 1 } - t \right| . } } \end{array}
$$

Thus $| s _ { 1 } - t | \leq 2 \eta _ { c } / \eta _ { d }$

Next, we introduce the key concept, resultant that will be considered in the rest of the section.

Definition 5.15 (Resultant). For $i \in [ 2 ]$ , let $\begin{array} { r } { q _ { i } ( z ) = \sum _ { r = 0 } ^ { d _ { i } } a _ { i r } z ^ { r } : \mathbb { R } \to \mathbb { R } } \end{array}$ . The resultant for $( q _ { i } , q _ { j } )$ is defined as $\begin{array} { r } { R ( q _ { i } , q _ { j } ) = a _ { i d _ { i } } ^ { d _ { j } } \prod _ { t = 1 } ^ { d _ { i } } q _ { j } ( r _ { t } ) } \end{array}$ , where for $t \in [ d _ { i } ]$ , r<sub>t</sub> is the tth root (multiple roots and complex roots are also considered) of $q _ { i }$

It is not hard to see that two polynomials have an exact same root if and only if their resultant is zero. Besides, resultant also has the following useful property that will be used in our analysis later.

Fact 5.16. Let $q _ { i } : \mathbb { R }  \mathbb { R }$ be a polynomial, $i \in [ 2 ]$ . Then $| R ( q _ { 1 } , q _ { 2 } ) | = | R ( q _ { 2 } , q _ { 1 } ) |$

The central property that will be used in our proof is that a resultant for 2 quadratics $q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) }$ is always a polynomial in w.

Fact 5.17. Let $q _ { 1 } , q _ { 2 } : \mathbb { R } ^ { d }  \mathbb { R }$ be two polynomials with degree at most 2, then $R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { \mathbf { w } } )$ is a polynomial over R<sup>d</sup> with degree at most 4. Furthermore, for $\textbf { w } \perp \textbf { v }$ and $i \in [ 2 ]$ , write $q _ { i } ^ { ( \mathbf { w } ) } =$ $a _ { i } z ^ { 2 } + b _ { i } ( \mathbf { w } ) z + c ( \mathbf { w } )$ , where $a _ { i } \in \mathbb { R } , b _ { i } ( \mathbf { w } ) , c _ { i } ( \mathbf { w } )$ are polynomials over w, with de $\varsigma ( b _ { i } ) \leq 1$ and de $\varsigma ( c _ { i } ) \leq 2$ . Define polynomials over w, $U ( \mathbf { w } ) = a _ { 2 } b _ { 1 } ( \mathbf { w } ) - a _ { 1 } b _ { 2 } ( \mathbf { w } )$ and $V ( \mathbf { w } ) = a _ { 2 } c _ { 1 } ( \mathbf { w } ) - a _ { 1 } c _ { 2 } ( \mathbf { w } )$ $I f a _ { 1 } \ne 0$ , then

$$
R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) = a _ { 1 } ^ { \mathrm { d e g } ( q _ { 2 } ) - 2 } ( U ( \mathbf { w } ) ^ { 2 } c _ { 1 } ( \mathbf { w } ) / a _ { 1 } - U ( \mathbf { w } ) V ( \mathbf { w } ) b _ { 1 } ( \mathbf { w } ) / a _ { 1 } + V ^ { 2 } ( \mathbf { w } ) ) .
$$

Proof of Fact 5.17. For w $\perp { \textbf { v } }$ and $i \in [ 2 ]$ , write $q _ { i } ^ { ( \mathbf { w } ) } = a _ { i } z ^ { 2 } + b _ { i } ( \mathbf { w } ) z + c ( \mathbf { w } )$ , where $a _ { i } \in \mathbb { R }$ $b _ { i } ( \mathbf { w } ) , c _ { i } ( \mathbf { w } )$ are polynomials over w, with d $\operatorname { s g } ( b ) \leq 1$ and deg $\mathord { / { \vphantom { ( c ) } }  \kern - delimiterspace } c ) \leq 2$ . Define polynomials over w, $U ( \mathbf { w } ) = a _ { 2 } b _ { 1 } ( \mathbf { w } ) - a _ { 1 } b _ { 2 } ( \mathbf { w } )$ and $V ( \mathbf { w } ) = a _ { 2 } c _ { 1 } ( \mathbf { w } ) - a _ { 1 } c _ { 2 } ( \mathbf { w } )$ . Notice that

$$
a _ { 2 } q _ { 1 } ^ { ( \mathbf { w } ) } ( z ) - a _ { 1 } q _ { 2 } ^ { ( \mathbf { w } ) } ( z ) = U ( \mathbf { w } ) z + V ( \mathbf { w } ) .\tag{14}
$$

Case 1: $a _ { 1 } \neq 0$ In the first case, we consider the case where $a _ { 1 } \neq 0$ , which implies that $q _ { 1 } ^ { ( \mathbf { w } ) }$ always has degree 2. Let $r _ { 1 } , r _ { 2 }$ be two roots for $q _ { 1 } ^ { ( \mathbf { w } ) }$ and recall that $R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) = a _ { 1 } ^ { \mathrm { d e g } ( q _ { 2 } ) } q _ { 2 } ^ { ( \mathbf { w } ) } ( \dot { r _ { 1 } } ) q _ { 2 } ^ { ( \mathbf { w } ) } ( r _ { 2 } )$ We consider three diferent cases. By (14), we know that $q _ { 2 } ^ { ( \mathbf { w } ) } ( r _ { i } ) = - a _ { 1 } ^ { - 1 } ( U ( \mathbf { w } ) r _ { i } + V ( \mathbf { w } ) )$ . This implies that

$$
\begin{array} { r l } & { ( U ( \mathbf { w } ) r _ { 1 } + V ( \mathbf { w } ) ) ( U ( \mathbf { w } ) r _ { 2 } + V ( \mathbf { w } ) ) = U ( \mathbf { w } ) ^ { 2 } ( r _ { 1 } r _ { 2 } ) + U ( \mathbf { w } ) V ( \mathbf { w } ) ( r _ { 1 } + r _ { 2 } ) + V ^ { 2 } ( \mathbf { w } ) } \\ & { \qquad = U ( \mathbf { w } ) ^ { 2 } c _ { 1 } ( \mathbf { w } ) / a _ { 1 } - U ( \mathbf { w } ) V ( \mathbf { w } ) b _ { 1 } ( \mathbf { w } ) / a _ { 1 } + V ^ { 2 } ( \mathbf { w } ) , } \end{array}
$$

which is a polynomial over w with degree at most 4. In particular,

$$
R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) = a _ { 1 } ^ { \mathrm { d e g } ( q _ { 2 } ) - 2 } ( U ( \mathbf { w } ) ^ { 2 } c _ { 1 } ( \mathbf { w } ) / a _ { 1 } - U ( \mathbf { w } ) V ( \mathbf { w } ) b _ { 1 } ( \mathbf { w } ) / a _ { 1 } + V ^ { 2 } ( \mathbf { w } ) ) .\tag{15}
$$

Case 2: $a _ { 1 } = 0 , b _ { 1 } \neq 0$ In this case, we know that the only root of $q _ { 1 } ^ { ( \mathbf { w } ) }$ is $r _ { 1 } = - c _ { 1 } ( \mathbf { w } ) / b _ { 1 } ( \mathbf { w } )$ which gives $R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) = b _ { 1 } ( \mathbf { w } ) ^ { \mathrm { { d e g } } ( q _ { 2 } ^ { ( \mathbf { w } ) } ) } q _ { 2 } ^ { ( \mathbf { w } ) } ( - c _ { 1 } ( \mathbf { w } ) / b _ { 1 } ( \mathbf { w } ) )$ . By expanding $q _ { 2 } ^ { ( \mathbf { w } ) }$ , we have three cases. If $a _ { 2 } \neq 0$ , then

$$
R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) = a _ { 2 } c _ { 1 } ^ { 2 } ( \mathbf { w } ) - b _ { 1 } ( \mathbf { w } ) b _ { 2 } ( \mathbf { w } ) c _ { 1 } ( \mathbf { w } ) + b _ { 1 } ^ { 2 } ( \mathbf { w } ) c _ { 2 } ( \mathbf { w } ) ,
$$

which has a degree at most 4. If $a _ { 2 } = 0$ but $b _ { 2 } \neq 0$ , then

$$
R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) = b _ { 1 } ( \mathbf { w } ) c _ { 2 } ( \mathbf { w } ) - b _ { 2 } ( \mathbf { w } ) c _ { 1 } ( \mathbf { w } ) ,
$$

which has degree at most 3. If $a _ { 2 } = 0 , b _ { 2 } = 0$ , then $R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) = c _ { 2 } ( \mathbf { w } )$ , which has a degree at most 2.

Case 3: $a _ { 1 } = 0 , b _ { 1 } = 0$ In this case, we always have $R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) = c _ { 1 } ( \mathbf { w } ) ^ { \mathrm { d e g } ( q _ { 2 } ) }$ , which has degree at most 4. □

Using Fact 5.17, we next show that if there is a descent probability over w such that $\left| q _ { j } ^ { ( \mathbf { w } ) } \right|$ is small at a root of $q _ { i } ^ { ( \mathbf { w } ) }$ , then it must be the case that $R ( q _ { i } ^ { ( \mathbf { w } ) } , q _ { j } ^ { ( \mathbf { w } ) } )$ has a small norm.

Lemma 5.18. For $i \in [ k ] , \ l e t \ p _ { i } ( { \bf x } )$ be a degree-2 polynomial with $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p _ { i } ( \mathbf { x } ) ] \ = \ 0$ and $\mathbf { V a r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p _ { i } ( \mathbf { x } ) ] = 1$ . Let $t _ { i } \in \mathbb { R }$ and $q _ { i } ( \mathbf { x } ) : = p _ { i } ( \mathbf { x } ) - t _ { i }$ . Let $f ( \mathbf { x } ) = \mathbb { 1 } \{ q _ { 1 } ( \mathbf { x } ) \geq 0 , \dots , q _ { k } ( \mathbf { x } ) \geq 0 \}$ $L e t \eta _ { d } , \eta _ { c } \in ( 0 , 1 )$ . For w $\perp \mathbf { v } _ { i }$ , let $C _ { 1 2 } ^ { ( \mathbf { w } ) }$ be the set of jump points of $f _ { \mathbf { w } }$ such that $q _ { 1 } ^ { ( \mathbf { w } ) } ( s ) = 0$ , and $\big | q _ { 2 } ^ { ( \mathbf { w } ) } ( s ) \big | \le \eta _ { c } .$ . Suppose that $\begin{array} { r } { \mathbf { E } _ { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \sum _ { s \in C _ { 1 2 } ^ { ( \mathbf { w } ) } } \phi ( s ) ] \ge \alpha } \end{array}$ , then $\left\| R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) \right\| \le O ( \eta _ { c } \log ( 1 / \alpha ) / \alpha ^ { 4 } )$

Proof of Lemma 5.18. We write $q _ { i } ^ { ( \mathbf { w } ) } ( z ) = a _ { i } z ^ { 2 } + b _ { i } ( \mathbf { w } ) z + c _ { i } ( \mathbf { w } )$ , for $i \in [ 2 ]$ , where $\deg ( b _ { i } ) \leq 1$ and $\deg ( c _ { i } ) \leq 2$ . Notice that The degree of $q _ { i } ^ { ( \mathbf { w } ) } ( z )$ is at most 2. Our proof will thus consider two cases on the degree of $q _ { i } ^ { ( \mathbf { w } ) }$

Case 1: $\mathrm { d e g } ( q _ { 1 } ^ { ( \mathbf { w } ) } ) = 2$ We start with the case where $\mathrm { d e g } ( q _ { 1 } ^ { ( \mathbf { w } ) } ) = 2$ . Let $r _ { 1 } , r _ { 2 }$ be the two roots of $q _ { 1 } ^ { ( \mathbf { w } ) }$ . Then, we have

$$
\begin{array} { r l } & { q _ { 2 } ^ { ( \mathbf { w } ) } ( r _ { 1 } ) - q _ { 2 } ^ { ( \mathbf { w } ) } ( r _ { 2 } ) = a _ { 2 } ( r _ { 1 } - r _ { 2 } ) ( r _ { 1 } + r _ { 2 } ) + b _ { 2 } ( \mathbf { w } ) ( r _ { 1 } - r _ { 2 } ) } \\ & { \qquad = ( r _ { 1 } - r _ { 2 } ) ( a _ { 2 } ( r _ { 1 } + r _ { 2 } ) + b _ { 2 } ( \mathbf { w } ) ) } \\ & { \qquad = ( r _ { 1 } - r _ { 2 } ) ( b _ { 2 } ( \mathbf { w } ) - a _ { 2 } b _ { 1 } ( \mathbf { w } ) / a _ { 1 } ) } \\ & { \qquad = - ( r _ { 1 } - r _ { 2 } ) U ( \mathbf { w } ) / a _ { 1 } . } \end{array}
$$

On the other hand,

$$
a _ { 1 } ( r _ { 1 } - r _ { 2 } ) = - ( 2 a _ { 1 } r _ { 2 } + b _ { 1 } ( \mathbf { w } ) ) = - ( q _ { 1 } ^ { ( \mathbf { w } ) } ) ^ { \prime } ( r _ { 2 } ) .
$$

Together, we obtain that

$$
a _ { 1 } ^ { 2 } ( q _ { 2 } ^ { ( \mathbf { w } ) } ( r _ { 1 } ) - q _ { 2 } ^ { ( \mathbf { w } ) } ( r _ { 2 } ) ) = - a _ { 1 } ( r _ { 1 } - r _ { 2 } ) U ( \mathbf { w } ) = U ( \mathbf { w } ) ( q _ { 1 } ^ { ( \mathbf { w } ) } ) ^ { \prime } ( r _ { 2 } ) .
$$

This implies that

$$
\begin{array} { r l } & { R ( q _ { 1 } ^ { \mathbf { ( w ) } } , q _ { 2 } ^ { \mathbf { ( w ) } } ) = a _ { 1 } ^ { \mathbf { d e g } ( q _ { 2 } ) } q _ { 2 } ^ { \mathbf { ( w ) } } ( r _ { 1 } ) q _ { 2 } ^ { \mathbf { ( w ) } } ( r _ { 2 } ) } \\ & { \qquad = q _ { 2 } ^ { \mathbf { ( w ) } } ( r _ { 1 } ) ( a _ { 1 } ^ { \mathbf { d e g } ( q _ { 2 } ) - 2 } ( a _ { 1 } ^ { 2 } q _ { 2 } ^ { \mathbf { ( w ) } } ( r _ { 1 } ) + a _ { 1 } ( r _ { 1 } - r _ { 2 } ) U ( \mathbf { w } ) ) ) } \\ & { \qquad = q _ { 2 } ^ { \mathbf { ( w ) } } ( r _ { 1 } ) ( a _ { 1 } ^ { \mathbf { d e g } ( q _ { 2 } ) - 2 } ( a _ { 1 } ^ { 2 } q _ { 2 } ^ { \mathbf { ( w ) } } ( r _ { 1 } ) - U ( \mathbf { w ) } ( q _ { 1 } ^ { \mathbf { ( w ) } } ) ^ { \prime } ( r _ { 2 } ) ) ) } \end{array}
$$

We consider the degree of $q _ { 2 } ^ { ( \mathbf { w } ) }$ . If $\mathrm { d e g } ( q _ { 2 } ^ { ( \mathbf { w } ) } ) = 2$ , then

$$
R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) = q _ { 2 } ^ { ( \mathbf { w } ) } ( r _ { 1 } ) ( a _ { 1 } ^ { 2 } q _ { 2 } ^ { ( \mathbf { w } ) } ( r _ { 1 } ) - U ( \mathbf { w } ) ( q _ { 1 } ^ { ( \mathbf { w } ) } ) ^ { \prime } ( r _ { 2 } ) )\tag{16}
$$

Similarly, if $\mathrm { d e g } ( q _ { 2 } ^ { ( \mathbf { w } ) } ) = 1$ , then we have

$$
R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) = q _ { 2 } ^ { ( \mathbf { w } ) } ( r _ { 1 } ) ( a _ { 1 } q _ { 2 } ^ { ( \mathbf { w } ) } ( r _ { 1 } ) + b _ { 2 } ( \mathbf { w } ) ( q _ { 1 } ^ { ( \mathbf { w } ) } ) ^ { \prime } ( r _ { 2 } ) ) .\tag{17}
$$

And if $\mathrm { d e g } ( q _ { 2 } ^ { ( \mathbf { w } ) } ) = 0$ , then we have $R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) = ( q _ { 2 } ^ { ( \mathbf { w } ) } ( r _ { 1 } ) ) ^ { 2 }$

Recall that $q _ { i } ^ { ( \mathbf { w } ) } ( z ) = a _ { i } ( z ^ { 2 } - 1 ) + b _ { i } ( \mathbf { w } ) z + c _ { i } ( \mathbf { w } ) + a _ { i } = p _ { i } ( z \mathbf { v } + \mathbf { w } ) - t _ { i }$ . Since $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p _ { i } ( \mathbf { x } ) ] = 0$ and $\mathbf { V a r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p _ { i } ( \mathbf { x } ) ] = 1$ . This implies that $| a _ { i } | \leq 1 / \sqrt { 2 } \mathrm { a n d } \| b _ { i } ( \mathbf { w } ) \| \leq \sqrt { 2 }$ By the tail probability of Gaussian distribution, we know that for $T : = O ( \sqrt { \log ( 1 / \alpha ) } ) , \mathbf { P r _ { w \sim \mathcal { N } ( 0 , \mathbf { I } ) } } [ | b _ { i } | > T ] \le \alpha / 3 2$

For each $\mathbf { w } \perp \mathbf { v } ,$ we denote by $Z ^ { ( \mathbf { w } ) }$ the set of roots s of $q _ { 1 } ^ { ( \mathbf { w } ) }$ such that $\big | q _ { 2 } ^ { ( \mathbf { w } ) } ( s ) \big | \le \eta _ { c }$ , with $\phi ( s ) \geq \alpha / 1 6$ . Then, for each $s \in Z ^ { ( \mathbf { w } ) }$ , we have $| s | \le T$ . We will next show that for every w, such that $| b _ { 1 } ( \mathbf { w } ) | \leq T , | b _ { 2 } ( \mathbf { w } ) | \leq T$ and $Z ^ { ( \mathbf { w } ) } \neq \emptyset , R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } )$ is small. Let $r _ { 1 } = s$ . If $\mathrm { d e g } ( q _ { 2 } ^ { ( \mathbf { w } ) } ) = 2$ then by (16), we have

$$
\begin{array} { r l } & { \left| R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) \right| = \left| q _ { 2 } ^ { ( \mathbf { w } ) } ( s ) ( a _ { 1 } ^ { 2 } q _ { 2 } ^ { ( \mathbf { w } ) } ( s ) + U ( \mathbf { w } ) ( q _ { 1 } ^ { ( \mathbf { w } ) } ) ^ { \prime } ( s ) ) \right| } \\ & { \qquad \leq \eta _ { c } ^ { 2 } + \eta _ { c } \left| U ( \mathbf { w } ) \right| \left| 2 a _ { 1 } s + b _ { 1 } ( \mathbf { w } ) \right| } \\ & { \qquad \leq \eta _ { c } ^ { 2 } + \eta _ { c } ( \left| a _ { 1 } b _ { 2 } ( \mathbf { w } ) \right| + \left| a _ { 2 } b _ { 1 } ( \mathbf { w } ) \right| ) \left| 2 a _ { 1 } s + b _ { 1 } ( \mathbf { w } ) \right| } \\ & { \qquad \leq 1 0 \eta _ { c } T ^ { 2 } = O ( \eta _ { c } \log ( 1 / \alpha ) ) . } \end{array}
$$

If $\mathrm { d e g } ( q _ { 2 } ^ { ( \mathbf { w } ) } ) = 1$ , then by (17), we have

$$
\begin{array} { r l } & { \left| R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) \right| = \left| q _ { 2 } ^ { ( \mathbf { w } ) } ( s ) ( a _ { 1 } q _ { 2 } ^ { ( \mathbf { w } ) } ( s ) - b _ { 2 } ( \mathbf { w } ) ( q _ { 1 } ^ { ( \mathbf { w } ) } ) ^ { \prime } ( s ) ) \right| } \\ & { \quad \quad \quad \leq \eta _ { c } ^ { 2 } + \eta _ { c } \left| b _ { 2 } ( \mathbf { w } ) \right| \left| 2 a _ { 1 } s + b _ { 1 } ( \mathbf { w } ) \right| } \\ & { \quad \quad \quad \leq 1 0 \eta _ { c } T ^ { 2 } = O ( \eta _ { c } \log ( 1 / \alpha ) ) . } \end{array}
$$

If de $\grave { \{ q _ { 2 } ^ { ( \mathbf { w } ) } \} } = 0$ , then trivially, we have $\left| R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) \right| \leq \eta _ { c } ^ { 2 } .$

Now since $\begin{array} { r } { \mathbf { E } _ { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \sum _ { s \in C _ { 1 2 } ( \mathbf { w } ) } \phi ( s ) ] \stackrel { . } { \geq } \alpha } \end{array}$ , we know that ${ \bf E } _ { \bf w \sim \mathcal { N } ( 0 , I ) } [ \sum _ { s \in Z ( \bf w ) } \phi ( s ) ] \geq 7 \alpha / 8$ . Since $\phi ( s ) \leq 1 / \sqrt { 2 \pi }$ , this implies that

$$
\begin{array} { r } { \quad \underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { P r } } \big [ \big | R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) \big | \leq O ( \eta _ { c } \log ( 1 / \alpha ) ) ] \geq \underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { P r } } [ | b _ { 1 } ( \mathbf { w } ) | \leq T , | b _ { 2 } ( \mathbf { w } ) | \leq T , | Z | ^ { ( \mathbf { w } ) } \geq 1 ] \geq \alpha / 2 . } \end{array}
$$

Since $R ( q _ { i } ^ { ( \mathbf { w } ) } , q _ { j } ^ { ( \mathbf { w } ) } )$ is a polynomial with degree at most 4 by Fact 5.17, using Carbery–Wright anti-concentration inequality, we have for every $\tau > 0$ , we have

$$
\begin{array} { r } { \frac { \mathbf { P r } } { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \left| R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) \right| \le \tau ] \le O ( ( \tau / \left\| R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) \right\| ) ^ { 1 / 4 } ) . } \end{array}
$$

By choosing $\tau = { \cal O } ( \eta _ { c } \log ( 1 / \alpha ) )$ , we obtain that $\left\| R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) \right\| \le O ( \eta _ { c } \log ( 1 / \alpha ) / \alpha ^ { 4 } )$

Case 2: $\mathrm { d e g } ( q _ { 1 } ^ { ( \mathbf { w } ) } ) = 1$ In the rest of the proof, we work on the case where $\mathrm { d e g } ( q _ { 1 } ^ { ( \mathbf { w } ) } ) = 1$ . In this case, we have $q _ { 1 } ^ { ( \mathbf { w } ) } ( z ) = b _ { 1 } ( \mathbf { w } ) z + c _ { 1 } ( \mathbf { w } )$ and

$$
R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) = b _ { 1 } ( \mathbf { w } ) ^ { \mathrm { d e g } ( q _ { 2 } ^ { ( \mathbf { w } ) } ) } q _ { 2 } ^ { ( \mathbf { w } ) } ( r ) ,
$$

where $r$ is the root of $q _ { 1 } ^ { ( \mathbf { w } ) } ( z )$ . This gives for each $r \in Z ^ { ( \mathbf { w } ) }$ and for every w such that $| b _ { 1 } | \leq T$

$$
\left| R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) \right| \le \eta _ { c } \operatorname* { m a x } \{ 1 , | b _ { 1 } ( \mathbf { w } ) | ^ { 2 } \} \le O ( \eta _ { c } \log ( 1 / \alpha ) )
$$

Now since $\begin{array} { r } { \mathbf { E } _ { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \sum _ { s \in C _ { 1 2 } ( \mathbf { w } ) } \phi ( s ) ] \geq \alpha } \end{array}$ , we know that ${ \bf E } _ { \bf w \sim \mathcal { N } ( 0 , I ) } [ \sum _ { s \in Z ( \bf w ) } \phi ( s ) ] \geq 7 \alpha / 8$ . Since $\phi ( s ) \leq 1 / \sqrt { 2 \pi }$ , this implies that

$$
\begin{array} { r } { \quad \underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { P r } } \big [ \big | R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) \big | \leq O ( \eta _ { c } \log ( 1 / \alpha ) ) ] \geq \underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { P r } } [ | b _ { 1 } ( \mathbf { w } ) | \leq T , | b _ { 2 } ( \mathbf { w } ) | \leq T , | Z | ^ { ( \mathbf { w } ) } \geq 1 ] \geq \alpha / 2 . } \end{array}
$$

Since $R ( q _ { i } ^ { ( \mathbf { w } ) } , q _ { j } ^ { ( \mathbf { w } ) } )$ is a polynomial with degree at most 4 by Fact 5.17, using Carbery–Wright anti-concentration inequality, we have for every $\tau > 0$ , we have

$$
\begin{array} { r } { \frac { \mathbf { P r } } { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \left| R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) \right| \le \tau ] \le O ( ( \tau / \left\| R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) \right\| ) ^ { 1 / 4 } ) . } \end{array}
$$

By choosing $\tau = { \cal O } ( \eta _ { c } \log ( 1 / \alpha ) )$ , we obtain that $\left\| R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) \right\| \le O ( \eta _ { c } \log ( 1 / \alpha ) / \alpha ^ { 4 } )$

Robust Factorization Lemma With Lemma 5.18, we next present the main technical lemma, Lemma 5.19. We show that under the condition of Lemma 5.18, there are only three cases for a pair of degree 2 polynomials. Before giving Lemma 5.19, we first provide some intuition about what algebraic structure we can obtain from small resultants.

Our intuition starts from the Euclidean algorithm for finding the greatest common divisor for a pair of quadratics. Consider two univariate quadratic polynomials $q _ { i } = a _ { i } z ^ { 2 } + b _ { i } z + c _ { i } , i \in [ 2 ]$ . The first step of Euclidean algorithm is to eliminate the leading quadratic term, by considering

$$
a _ { 2 } q _ { 1 } - a _ { 1 } q _ { 2 } = U z + V ,
$$

where in Fact 5.17, we define $U = a _ { 2 } b _ { 1 } - a _ { 1 } b _ { 2 }$ and $V = a _ { 2 } c _ { 1 } - a _ { 1 } c _ { 2 }$ . If $U \neq 0$ and $q _ { 1 } , q _ { 2 }$ exactly share a root, then the root r must be $- V / U$ , which provides a way to factorizing the two polynomials into the form of $q _ { i } = L \ell _ { i }$ . The resultant of the two quadratics provides a quantitative version of this observation: it satisfies $R ( q _ { 1 } , q _ { 2 } ) = U ^ { 2 } q _ { 1 } ( r ) / a _ { 1 }$ . Consequently, whenever the resultant is small and U is bounded away from zero, both polynomials take small values at $r .$ . By polynomial division, each $q _ { i }$ can then be approximated by a product $( z - r ) \ell _ { i } ( z )$ , with an approximation error controlled by the resultant. On the other hand, if U is nearly zero, then small resultant implies that V is also nearly zero, which implies that $q _ { 1 }$ and $q _ { 2 }$ are nearly in parallel. Thus, a small resultant suggests either an approximate common afine factor or approximate proportionality of the two polynomials.

However, in our case, the coeficients of $q _ { i } ^ { ( \mathbf { w } ) }$ are polynomials on $\mathbf { w } .$ . If we directly carry the analysis of Euclidean algorithm, then the formal candidate of a common root is $r = - V ( \mathbf { w } ) / U ( \mathbf { w } )$ which is a rational function in $\mathbf { x } ,$ instead of a low degree polynomial. Furthermore, since $U ( \mathbf { w } )$ could be very small for some $\mathbf { w } \perp \mathbf { v } ,$ the instability from these slices would amplify the error in the approximate setting. Furthermore, there is also a genuine algebraic obstruction to demanding a common afine factor. For example, consider a simple example in $\mathbb { R } ^ { 2 }$ , where $q _ { 1 } ^ { ( w ) } ( z ) = q _ { 2 } ^ { ( w ) } ( z ) =$ $z w - 1$ . Although the two function shares an exact root $1 / w _ { ; }$ , for every $w \ne 0$ , they cannot be written as a product of two afine polynomials in $( z , w )$ . In fact, even if we consider a case of $q _ { 1 } ^ { ( w ) } ( z ) = q _ { 1 } ^ { ( w ) } ( z ) = z ^ { 2 } - w ^ { 2 } - 1 = ( z - \sqrt { w ^ { 2 } + 1 } ) ( z + \sqrt { w ^ { 2 } + 1 } )$ , where an exact factorization is possible, the factorization might be an irrational function in w.

We therefore seek a structural conclusion with several alternatives rather than an unconditional common-afine-factor theorem. Under the normalization and boundary-mass assumptions of Lemma 5.18, the goal is to show that either one defining polynomial is approximately independent of v direction, the two defining polynomials are approximately proportional, or both can be approximated by products of afine polynomials sharing a common afine factor.

Lemma 5.19. For $i \in [ 2 ]$ , let $p _ { i } ( { \bf x } )$ be a degree-2 polynomial with $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p _ { i } ( \mathbf { x } ) ] \ = \ 0$ and $\mathbf { V a r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p _ { i } ( \mathbf { x } ) ] = 1$ . Let $t _ { i } \in \mathbb { R }$ and $q _ { i } ( \mathbf { x } ) : = p _ { i } ( \mathbf { x } ) - t _ { i }$ . Let $f ( \mathbf { x } ) = \mathbb { 1 } \{ q _ { 1 } ( \mathbf { x } ) \geq 0 , q _ { 2 } ( \mathbf { x } ) \geq 0 \}$ . Let $\eta _ { c } \in ( 0 , 1 )$ and $\alpha \in ( 0 , 1 )$ . For w ⊥ v, denote by $F ( \mathbf { w } ) \subseteq \mathbb { R }$ , the set of jump points s of $f _ { \mathbf { w } }$ , such that $q _ { 1 } ^ { ( \mathbf { w } ) } ( s ) = 0$ , and $0 \leq q _ { 2 } ^ { ( \mathbf { w } ) } ( s ) \leq \eta _ { c }$ . Let $\alpha \in ( 0 , 1 )$ ). Suppose that $\begin{array} { r } { \mathbf { E } _ { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \sum _ { s \in F ( \mathbf { w } ) } \phi ( s ) ] \geq \alpha } \end{array}$ and $| t _ { i } | \le { \cal O } ( \log ( 1 / \alpha ) )$ . Denote by $\epsilon _ { f a c } = \tilde { O } ( ( \eta _ { c } / \alpha ^ { 4 } ) ^ { 1 / 2 0 0 0 } \alpha ^ { - 1 } )$ . Then the one of the follows holds.

1. for some $i \in [ 2 ]$ , there exists a degree-2 polynomial $\tilde { q } _ { i } ( \mathbf { w } )$ such that $\| q _ { i } - \tilde { q } _ { i } ( \mathbf { w } ) \| \leq \epsilon _ { f a c }$

2. there exists some $\lambda \in \mathbb { R }$ such that $\| q _ { i } - \lambda q _ { 3 - i } \| \leq \epsilon _ { f a c . }$ , for some $i \in [ 2 ]$

3. there exist polynomials $L , \ell _ { 1 } , \ell _ { 2 }$ with degree at most 1 such that $\| q _ { i } ( \mathbf { x } ) - L ( \mathbf { x } ) \ell _ { i } ( \mathbf { x } ) \| \leq \epsilon _ { f a c }$ for both $i \in [ 2 ]$

Before giving the proof, we first provide the intuition of the idea before Lemma 5.19. Recall that Lemma 5.18 shows that the resultant of the two defining polynomials has a small Gaussian $L _ { 2 }$ norm. Our goal is to use this information to obtain an approximate common afine factor, unless the polynomials fall into one of the degenerate configurations described in the lemma. Following the intuition from the Euclidean algorithm, we consider two cases, depending on whether at least one polynomial has a suficiently large quadratic coeficient along the direction v.

First, suppose that $| a _ { 1 } |$ is bounded below by some parameter $\kappa > 0$ . We eliminate the quadratic terms and write

$$
a _ { 2 } q _ { 1 } - a _ { 1 } q _ { 2 } = U ( \mathbf { w } ) z + V ( \mathbf { w } ) ,
$$

where $U = a _ { 2 } b _ { 1 } ( \mathbf { w } ) - a _ { 1 } b _ { 2 } ( \mathbf { w } )$ is afine and $V = a _ { 2 } c _ { 1 } ( \mathbf { w } ) - a _ { 1 } c _ { 2 } ( \mathbf { w } )$ is quadratic in w. If this remainder is suficiently small, then the two defining polynomials are approximately proportional, giving the second alternative of the lemma. Otherwise, we show that $U ( \mathbf { w } )$ must have a nontrivial $L _ { 2 }$ norm. Indeed, the resultant can be expressed in terms of $U$ and $V$ , and if $\| U \|$ is very small, the smallness of the resultant would force V to be small as well, contradicting our assumption that the remainder is nontrivial. This lower bound on $\lVert U \rVert$ is important because it allows us to make the polynomial division suggested by the Euclidean algorithm quantitatively stable.

The main technical step is then to construct an afine polynomial $h ( \mathbf { w } )$ such that $V ( \mathbf { w } ) \approx$ $U ( \mathbf { w } ) h ( \mathbf { w } )$ , without directly dividing by $U ( \mathbf { w } )$ . We distinguish two cases according to the size of the linear part of $U .$ . If it is suficiently large, we perform polynomial division along the corresponding direction and obtain a remainder that is independent of that direction. By restricting the resultant to the hyperplane $U ( \mathbf { w } ) = 0$ , we can control this remainder using the small $L _ { 2 }$ norm of the resultant and the bounded degree of the polynomials. On the other hand, if the linear part of $U$ is small, then its constant part must be bounded away from zero. In this case, we use the degree-four Hermite component of the resultant to show that the quadratic part of V must be small, allowing us to approximate $V / U$ by an afine polynomial. In either case, we obtain an afine h satisfying $\| V - U h \|$ being small. We then define $L = z + h ( \mathbf { w } )$ and use the resultant identity, together with a lower bound on the norm of products of low-degree Gaussian polynomials, to show that both $q _ { 1 }$ and $q _ { 2 }$ are close to products of $L$ with afine polynomials. This gives the desired approximate common factor. The case where $| a _ { 2 } |$ is bounded below follows symmetrically.

It remains to consider the case where both $| a _ { 1 } |$ and $| a _ { 2 } |$ are small. In this regime, the two defining polynomials are approximately linear in v direction, so we instead work with $\bar { q } _ { i } ( z , { \bf w } ) =$ $b _ { i } ( \mathbf { w } ) z + c _ { i } ( \mathbf { w } )$ . If both $b _ { i }$ have small norm, the polynomials are approximately independent of $\mathbf { v } ,$ giving the first alternative. Otherwise, we introduce the corresponding elimination polynomial $T ( \mathbf { w } ) = b _ { 2 } ( \mathbf { w } ) c _ { 1 } ( \mathbf { w } ) - b _ { 1 } ( \mathbf { w } ) c _ { 2 } ( \mathbf { w } )$ . Using the original assumption that the two polynomials have nearly shared roots on a nontrivial fraction of Gaussian slices, we show that $T$ has a small $L _ { 2 }$ norm. We then distinguish whether $b _ { 1 }$ and $b _ { 2 }$ are approximately proportional. If they are, the smallness of T implies that the defining polynomials are approximately proportional as well. Otherwise, an argument analogous to the previous case shows that $c _ { 1 }$ and $c _ { 2 }$ are approximately divisible by $b _ { 1 }$ and $b _ { 2 }$ , respectively, with a common afine quotient. This again yields an approximate common afine factor. Combining these cases establishes the desired structural alternatives.

Now, we are able to give the proof of Lemma 5.19. We first give the following polynomial division lemma that will be used several times in our proof.

Lemma 5.20 (Polynomial division under a Gaussian distribution). Let $\mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } )$ and v $\in \mathbb { S } ^ { d - 1 }$ Write $z = { \bf x } \cdot { \bf v }$ and $\mathbf { w } = \mathrm { p r o j } _ { \mathbf { v } ^ { \perp } } ( \mathbf { x } )$ . Consider an afine function $L ( z \mathbf { v } + \mathbf { w } ) = \lambda z + B ( \mathbf { w } ) , \lambda \neq 0$ where B is an afine function on w. Let $r _ { L } ( \mathbf { w } ) = - B ( \mathbf { w } ) / \lambda$ be its root in z. Let q be a polynomial of degree at most two, written as $q ( z \mathbf { v } + \mathbf { w } ) = a z ^ { 2 } + b ( \mathbf { w } ) z + c ( \mathbf { w } )$ , deg $b \leq 1$ , deg $c \leq 2$ . The following hold.

1. There are unique polynomials $\ell ( \mathbf { x } )$ and $m ( \mathbf { w } )$ , with deg $\ell \leq 1$ and deg $m \le 2$ , such that $q ( \mathbf { x } ) = L ( \mathbf { x } ) \ell ( \mathbf { x } ) + m ( \mathbf { w } )$ , given by

$$
\begin{array} { l } { { \displaystyle \ell ( z { \mathbf v } + { \mathbf w } ) = \frac { a } { \lambda } z + \frac { b ( { \mathbf w } ) } { \lambda } - \frac { a B ( { \mathbf w } ) } { \lambda ^ { 2 } } } , } \\ { { \displaystyle m ( { \mathbf w } ) = c ( { \mathbf w } ) - \frac { B ( { \mathbf w } ) b ( { \mathbf w } ) } { \lambda } + \frac { a B ( { \mathbf w } ) ^ { 2 } } { \lambda ^ { 2 } } = q ( r _ { L } ( { \mathbf w } ) { \mathbf v } + { \mathbf w } ) } . } \end{array}
$$

If q is afine, ℓ is constant.

2. For a universal constant $C _ { i }$

$$
\| \ell \| _ { 2 } \leq \frac { C } { | \lambda | } \left( 1 + \frac { \| B \| _ { 2 } } { | \lambda | } \right) \| q \| _ { 2 } .\tag{18}
$$

For every polynomial P of degree at most $l \leq 4$

$$
\left\| P ( r _ { L } ( \mathbf { w } ) \mathbf { v } + \mathbf { w } ) \right\| _ { L ^ { 2 } ( \mathbf { w } ) } \leq C _ { l } \left( 1 + \frac { \| B \| _ { 2 } } { | \lambda | } \right) ^ { l } \| P \| _ { L ^ { 2 } ( \mathbf { x } ) } ,\tag{19}
$$

where $C _ { l }$ depends only on l.

3. Let $E \subseteq \mathbf { v } ^ { \bot }$ have Gaussian probability at least $\beta \in ( 0 , 1 / 2 ]$ . Suppose that $s : E \to $ R is measurable, $| s ( \mathbf { w } ) | \leq T$ , and, for every $\mathbf { w } \in E , | q ( s ( \mathbf { w } ) \mathbf { v } + \mathbf { w } ) | \leq \varepsilon \ a n d \ | L ( s ( \mathbf { w } ) \mathbf { v } + \mathbf { w } ) | \leq \eta$ then

$$
\begin{array} { r } { \| m \| _ { 2 } \leq C \beta ^ { - 2 } \left( \varepsilon + \eta \big ( T + \sqrt { \log ( 2 / \beta ) } \big ) \| \ell \| _ { 2 } \right) . } \end{array}\tag{20}
$$

In particular, when ${ \cal L } ( s ( { \bf w } ) { \bf v } + { \bf w } ) = 0 ~ o n ~ { \cal E }$

$$
\| q - L \ell \| _ { 2 } = \| m \| _ { 2 } \leq C \beta ^ { - 2 } \varepsilon .\tag{21}
$$

Proof of Lemma 5.20. The first item can be proved by directly checking the definition of ℓ and m. For uniqueness, regard the polynomials as polynomials in z with coeficients depending on w. A nonzero multiple of $L ,$ whose leading coeficient in z is the nonzero constant $\lambda ,$ cannot be independent of z.

To prove (18), write $q = a ( z ^ { 2 } - 1 ) + b ( \mathbf { w } ) z + ( c ( \mathbf { w } ) + a )$ . Gaussian orthogonality gives $2 a ^ { 2 } +$ $\| b \| _ { 2 } ^ { 2 } + \| c + a \| _ { 2 } ^ { 2 } = \| q \| _ { 2 } ^ { 2 }$ . The bound follows from the division step and the triangle inequality.

For (19), expand in normalized Hermite polynomials along v:

$$
P ( z \mathbf { v } + \mathbf { w } ) = \sum _ { j = 0 } ^ { l } h _ { j } ( z ) P _ { j } ( \mathbf { w } ) , \qquad \deg P _ { j } \leq l - j , \qquad \sum _ { j = 0 } ^ { l } \| P _ { j } \| _ { 2 } ^ { 2 } = \| P \| _ { 2 } ^ { 2 } .
$$

The function $r _ { L }$ is afine. Gaussian moment bounds imply

$$
\| h _ { j } ( r _ { L } ( \mathbf { w } ) ) \| _ { 4 } \leq C _ { l } ( 1 + \| r _ { L } \| _ { 2 } ) ^ { j } ,
$$

and hypercontractivity gives $\lVert P _ { j } \rVert _ { 4 } \leq C _ { l } \lVert P _ { j } \rVert _ { 2 }$ . Hölder’s inequality and the triangle inequality therefore give

$$
\big \| P ( r _ { L } ( \mathbf { w } ) \mathbf { v } + \mathbf { w } ) \big \| _ { 2 } \leq \sum _ { j = 0 } ^ { l } \| h _ { j } ( r _ { L } ) \| _ { 4 } \| P _ { j } \| _ { 4 } \leq C _ { l } ( 1 + \| r _ { L } \| _ { 2 } ) ^ { l } \| P \| _ { 2 } .
$$

Substitute $\| r _ { L } \| _ { 2 } = \| B \| _ { 2 } / | \lambda |$

Finally, write $\ell ( z \mathbf { v } + \mathbf { w } ) = \gamma z + D ( \mathbf { w } )$ , where D is afine. Then $| \gamma | \leq \| \ell \| _ { 2 }$ and $\| D \| _ { 2 } \leq \| \ell \| _ { 2 }$ Gaussian tails imply that, except on a set of probability at most $\beta / 2 , | D ( \mathbf { w } ) | \leq C \| \ell \| _ { 2 } \sqrt { \log ( 2 / \beta ) }$ On a subset of E of probability at least $\beta / 2$ we consequently have $| \ell ( s ( \mathbf { w } ) \mathbf { v } + \mathbf { w } ) | \leq C ( T +$ $\sqrt { \log ( 2 / \beta ) } ) \| \ell \| _ { 2 }$ . Evaluate the polynomial division at the original selected point $s ( \mathbf { w } ) \mathbf { v } + \mathbf { w }$ . It gives

$$
| m ( \mathbf { w } ) | \leq \varepsilon + C \eta \big ( T + \sqrt { \log ( 2 / \beta ) } \big ) \| \ell \| _ { 2 }
$$

on that set. Degree-two Gaussian anti-concentration says that $\operatorname* { P r } _ { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ | m ( \mathbf { w } ) | \le h ] \le C ( h / \| m \| _ { 2 } ) ^ { 1 / 2 }$ for $m \neq 0$ . The probability lower bound $\beta / 2$ proves (20). The case $m = 0$ is immediate, and setting $\eta = 0$ gives (21). □

Now, we are able to prove Lemma 5.19

Proof of Lemma 5.19. Recall that, for $i \in [ 2 ] , q _ { i } ^ { ( \mathbf { w } ) } ( z ) = a _ { i } z ^ { 2 } + b _ { i } ( \mathbf { w } ) z + c _ { i } ( \mathbf { w } )$ . Write $B = 1 +$ $C \log ( 1 / \alpha )$ , with C suficiently large. The normalization in the statement gives $| a _ { i } | \leq 1 , \| b _ { i } \| _ { 2 } \leq$ $1 , \| c _ { i } \| _ { 2 } \leq B$ . Let $\kappa , \xi , \rho , \tau ^ { \prime } \in ( 0 , 1 )$ be cutofs, to be specified at the end, with $\tau ^ { \prime } \leq \sqrt { \rho } / 4$

Case 1: $| a _ { 1 } | \geq \kappa .$ . By (14), $a _ { 2 } q _ { 1 } ^ { ( \mathbf { w } ) } ( z ) - a _ { 1 } q _ { 2 } ^ { ( \mathbf { w } ) } ( z ) = U ( \mathbf { w } ) z + V ( \mathbf { w } )$ , where $U = a _ { 2 } b _ { 1 } - a _ { 1 } b _ { 2 } , V =$ $a _ { 2 } c _ { 1 } - a _ { 1 } c _ { 2 }$ . In particular, $\| U \| _ { 2 } \leq 2$ and $\| V \| _ { 2 } \leq 2 B$

Case 1.1: $\| a _ { 2 } q _ { 1 } - a _ { 1 } q _ { 2 } \| _ { 2 } \leq \xi$ . Then $\| q _ { 2 } - ( a _ { 2 } / a _ { 1 } ) q _ { 1 } \| _ { 2 } \le \xi / \kappa$ , which gives the approximately proportional alternative.

Case 1.2: $\| a _ { 2 } q _ { 1 } - a _ { 1 } q _ { 2 } \| _ { 2 } > \xi$ . Suppose that $\| R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) \| _ { 2 } \leq \eta$ , where $\eta \le \xi ^ { 2 } / 1 0$ . We will choose such an η using Lemma 5.18 at the end of the proof. Define the auxiliary polynomial

$$
\mathscr { R } : = \frac { U ^ { 2 } c _ { 1 } } { a _ { 1 } } - \frac { U V b _ { 1 } } { a _ { 1 } } + V ^ { 2 } .\tag{22}
$$

By (15), $\mathcal { R } = a _ { 1 } ^ { 2 - d _ { 2 } } R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } )$ , where $d _ { 2 } \in \{ 0 , 1 , 2 \}$ is the degree of $q _ { 2 } ^ { ( \mathbf { w } ) }$ in z for almost every w. Hence $\| \mathcal { R } \| _ { 2 } \leq \eta$

Claim 5.21. Under the assumptions of Case $1 . \mathcal { Q } , \| U \| _ { 2 } \ge c \kappa \xi ^ { 2 } / B$ for a universal constant $c > 0$

Proof of Claim 5.21. By (22) and Fact 2.7,

$$
\| \mathcal { R } - V ^ { 2 } \| _ { 2 } \leq \frac { C B } { \kappa } \big ( \| U \| _ { 2 } ^ { 2 } + \| U \| _ { 2 } \big ) .
$$

Since $\| V \| _ { 2 } ^ { 2 } \leq \| V ^ { 2 } \| _ { 2 }$ and $\| a _ { 2 } q _ { 1 } - a _ { 1 } q _ { 2 } \| _ { 2 } ^ { 2 } = \| U \| _ { 2 } ^ { 2 } + \| V \| _ { 2 } ^ { 2 }$

$$
\xi ^ { 2 } \le \| U \| _ { 2 } ^ { 2 } + \eta + \frac { C B } { \kappa } \bigl ( \| U \| _ { 2 } ^ { 2 } + \| U \| _ { 2 } \bigr ) \le \eta + \frac { C ^ { \prime } B } { \kappa } \| U \| _ { 2 } .
$$

The last inequality uses $\| U \| _ { 2 } \leq 2$ and $B / \kappa \geq 1$ . The assumption $\eta \le \xi ^ { 2 } / 1 0$ proves the claim.

We next show that V is approximately divisible by U, with an afine quotient.

Claim 5.22. Under the assumptions of Case 1.2, there is an afine polynomial $h ( \mathbf { w } )$ such that

$$
\| V - U h \| _ { 2 } \le C \frac { B ^ { 4 / 3 } \eta ^ { 1 / 6 } } { \kappa ^ { 2 / 3 } \xi ^ { 4 / 3 } } , \qquad \| h \| _ { 2 } \le C \frac { B ^ { 2 } } { \kappa \xi ^ { 2 } } .
$$

Proof of Claim 5.22. Put $\delta _ { * } : = ( B ^ { 4 / 3 } \eta ^ { 1 / 6 } ) / ( \kappa ^ { 2 / 3 } \xi ^ { 4 / 3 } )$ . If $\delta _ { * } \geq c _ { 0 } B$ , where $c _ { 0 } > 0$ is a suficiently small universal constant, take $h = 0$ . Since $\| V \| _ { 2 } \leq 2 B$ , the claim follows after increasing C. We therefore assume $\delta _ { * } < c _ { 0 } B$ and choose

$$
\tau = \left( \frac { \kappa \xi ^ { 2 } \sqrt { \eta } } { B ^ { 2 } } \right) ^ { 1 / 3 } = \frac { \kappa \xi ^ { 2 } } { B ^ { 2 } } \delta _ { * } .
$$

By Claim 5.21 and the choice of $c _ { 0 } , \tau \leq \| U \| _ { 2 } / 4 .$

If U is nonconstant, write $\mathbf { w } = t \mathbf { u } + \mathbf { w } _ { \perp }$ and $U ( \mathbf { w } ) = \beta _ { 1 } t + \beta _ { 0 }$ , where $\mathbf { u } \in \mathbf { v } ^ { \perp }$ is a unit vector, $\mathbf { w } _ { \perp } \perp \mathbf { u }$ , and $\beta _ { 1 } > 0$ . If U is constant, set $\beta _ { 1 } = 0 , \beta _ { 0 } = U$ , and omit the terms involving t in the second subcase below.

Suppose first that $\beta _ { 1 } \geq \tau$ and put $s = - \beta _ { 0 } / \beta _ { 1 }$ . Apply Lemma 5.20(1) in the Gaussian space $\mathbf { v } ^ { \perp }$ , with distinguished direction u, to divide V by U. It gives an afine polynomial $h ( \mathbf { w } )$ and a remainder $E ( \mathbf { w } _ { \perp } )$ such that

$$
V = U h + E , \qquad E ( \mathbf { w } _ { \perp } ) = V ( s \mathbf { u } + \mathbf { w } _ { \perp } ) .
$$

On the hyperplane $U = 0$ , (22) gives $\mathcal { R } ( s \mathbf { u } + \mathbf { w } _ { \perp } ) = E ( \mathbf { w } _ { \perp } ) ^ { 2 }$ . Since deg $\mathcal { R } \leq 4$ , the restriction bound in Lemma 5.20(2) implies

$$
\| E \| _ { 2 } ^ { 2 } \leq \| E ^ { 2 } \| _ { 2 } \leq C ( 1 + | s | ) ^ { 4 } \| \mathcal { R } \| _ { 2 } \leq C \eta / \tau ^ { 4 } .
$$

Here $| \beta _ { 0 } | \leq \| U \| _ { 2 } \leq 2$ . Thus $\| V - U h \| _ { 2 } \le C \sqrt { \eta } / \tau ^ { 2 } = C \delta _ { * }$

Suppose next that $\beta _ { 1 } ~ < ~ \tau$ . Since $\tau \leq \| U \| _ { 2 } / 4$ , we have $| \beta _ { 0 } | \geq \| U \| _ { 2 } / 2 \geq c \kappa \xi ^ { 2 } / B$ . Take $h = V _ { \Sigma 1 } / \beta _ { 0 }$ . By the upper bound in Fact 2.7,

$$
\| V - U h \| _ { 2 } \leq \| V _ { 2 } \| _ { 2 } + C \frac { B ^ { 2 } \tau } { \kappa \xi ^ { 2 } } .\tag{23}
$$

To bound $\| V _ { 2 } \| _ { 2 }$ , note that $\lVert ( V _ { 2 } ^ { 2 } ) _ { 4 } \rVert _ { 2 } \geq \sqrt { 2 } \lVert V _ { 2 } \rVert _ { 2 } ^ { 2 }$ . Indeed, after diagonalization write $\begin{array} { r } { V _ { 2 } = \sum _ { i } \gamma _ { i } h _ { 2 } ( w _ { i } ) } \end{array}$ Gaussian orthogonality gives

$$
\| ( V _ { 2 } ^ { 2 } ) _ { 4 } \| _ { 2 } ^ { 2 } = 6 \sum _ { i } \gamma _ { i } ^ { 4 } + 4 \sum _ { i < j } \gamma _ { i } ^ { 2 } \gamma _ { j } ^ { 2 } \geq 2 \left( \sum _ { i } \gamma _ { i } ^ { 2 } \right) ^ { 2 } .
$$

Taking degree-four Hermite components in (22),

$$
\mathcal { R } _ { 4 } = ( V _ { 2 } ^ { 2 } ) _ { 4 } + \frac { \beta _ { 1 } ^ { 2 } } { a _ { 1 } } \big ( t ^ { 2 } ( c _ { 1 } ) _ { 2 } \big ) _ { 4 } - \frac { \beta _ { 1 } } { a _ { 1 } } \big ( t V _ { 2 } ( b _ { 1 } ) _ { 1 } \big ) _ { 4 } .
$$

Consequently, by Fact 2.7,

$$
\sqrt { 2 } \| V _ { 2 } \| _ { 2 } ^ { 2 } \leq \eta + C \frac { B \tau ^ { 2 } } { \kappa } + C \frac { \tau } { \kappa } \| V _ { 2 } \| _ { 2 } .
$$

Solving this quadratic inequality yields

$$
\| V _ { 2 } \| _ { 2 } \le C \left( \sqrt { \eta } + \tau \sqrt { B / \kappa } + \tau / \kappa \right) .
$$

Combining this with (23), using $B \geq 1$ and $\kappa , \xi \leq 1$ , gives

$$
\| V - U h \| _ { 2 } \leq C \left( \sqrt { \eta } + \frac { B ^ { 2 } \tau } { \kappa \xi ^ { 2 } } \right) \leq C \delta _ { * } .
$$

Finally, in either nontrivial subcase, $\| U h \| _ { 2 } \leq \| V \| _ { 2 } + \| V - U h \| _ { 2 } \leq C B$ . The lower bound in Fact 2.7 and Claim 5.21 give

$$
\| h \| _ { 2 } \leq C B / \| U \| _ { 2 } \leq C B ^ { 2 } / ( \kappa \xi ^ { 2 } ) .
$$

Given h from Claim 5.22, set $L ( \mathbf { x } ) = z + h ( \mathbf { w } )$ . Apply Lemma 5.20(1) to each $q _ { i } .$ , now along the original direction v. It gives

$$
q _ { i } = L \ell _ { i } + E _ { i } ( { \bf w } ) , \qquad \ell _ { i } = a _ { i } z + b _ { i } - a _ { i } h , \qquad E _ { i } = c _ { i } - b _ { i } h + a _ { i } h ^ { 2 } .
$$

Put $E = V - U h$ . Substitution into (22) gives $\mathcal { R } = E ^ { 2 } + U E ( 2 h - b _ { 1 } / a _ { 1 } ) + U ^ { 2 } E _ { 1 } / a _ { 1 }$ , and therefore

$$
\begin{array} { r } { U ^ { 2 } E _ { 1 } = a _ { 1 } \mathcal { R } - a _ { 1 } E ^ { 2 } - U E \big ( 2 a _ { 1 } h - b _ { 1 } \big ) . } \end{array}\tag{24}
$$

Fact 2.7 imply

$$
\| E _ { 1 } \| _ { 2 } \leq \frac { C } { \| U \| _ { 2 } ^ { 2 } } \left( \eta + \| E \| _ { 2 } ^ { 2 } + \| U \| _ { 2 } \| E \| _ { 2 } ( 1 + \| h \| _ { 2 } ) \right) .\tag{25}
$$

Moreover, $a _ { 2 } E _ { 1 } - a _ { 1 } E _ { 2 } = V - U h = E .$ so

$$
\lVert E _ { 2 } \rVert _ { 2 } \leq \kappa ^ { - 1 } ( \lVert E \rVert _ { 2 } + \lVert E _ { 1 } \rVert _ { 2 } ) .\tag{26}
$$

These inequalities give the required errors in the common-afine alternative once the parameters are chosen below.

Case 2: $| a _ { 1 } | < \kappa$ and $| a _ { 2 } | \geq \kappa$ . Interchange the two polynomials and apply Case 1. Fact 5.16 preserves the absolute value, and hence the norm, of the resultant. No other change is needed.

Case 3: $| a _ { 1 } | < \kappa$ and $| a _ { 2 } | < \kappa .$ . Put $\bar { q } _ { i } = b _ { i } ( \mathbf { w } ) z + c _ { i } ( \mathbf { w } )$ . Then $\| q _ { i } - \bar { q } _ { i } \| _ { 2 } = \sqrt { 3 } | a _ { i } | \leq \sqrt { 3 } \kappa$ . We first bound the elimination polynomial $T ( \mathbf { w } ) = b _ { 2 } ( \mathbf { w } ) c _ { 1 } ( \mathbf { w } ) - b _ { 1 } ( \mathbf { w } ) c _ { 2 } ( \mathbf { w } )$ . Let $F ^ { ( \mathbf { w } ) }$ be the jump points in the hypothesis, at which $q _ { 1 } ^ { ( \mathbf { w } ) } ( s ) = 0$ and $0 \leq q _ { 2 } ^ { ( \mathbf { w } ) } ( s ) \leq \eta _ { c }$ . Their weighted mass is at least $\alpha .$ . Choose $R _ { \alpha } = C \sqrt { B }$ with $C$ suficiently large. Discard the roots with $| s | > R _ { \alpha }$ and the layers where $| b _ { 1 } ( \mathbf { w } ) | > R _ { \alpha }$ or $| b _ { 2 } ( \mathbf { w } ) | > R _ { \alpha }$ . There are at most two roots of $q _ { 1 } ^ { ( \mathbf { w } ) }$ on almost every layer. Gaussian tails for $b _ { 1 } , b _ { 2 }$ and $\phi ( R _ { \alpha } ) \leq \alpha / 1 0 0$ therefore show that the remaining weighted mass is at least $\alpha / 2$ . In particular, the remaining layers have probability at least cα.

At a retained root $s ,$ the identity $b _ { 2 } q _ { 1 } - b _ { 1 } q _ { 2 } = ( a _ { 1 } b _ { 2 } - a _ { 2 } b _ { 1 } ) z ^ { 2 } + T$ gives

$$
T ( \mathbf { w } ) = ( a _ { 2 } b _ { 1 } ( \mathbf { w } ) - a _ { 1 } b _ { 2 } ( \mathbf { w } ) ) s ^ { 2 } - b _ { 1 } ( \mathbf { w } ) q _ { 2 } ^ { ( \mathbf { w } ) } ( s ) .
$$

Consequently, on a set of probability at least cα, $| T ( \mathbf { w } ) | \le C ( \kappa B ^ { 3 / 2 } + \eta _ { c } B ^ { 1 / 2 } )$ . Since deg $T \le 3$ Gaussian anti-concentration gives

$$
\| T \| _ { 2 } \le C \alpha ^ { - 3 } ( \kappa B ^ { 3 / 2 } + \eta _ { c } B ^ { 1 / 2 } ) .\tag{27}
$$

The conclusion is immediate if $T = 0 ;$ otherwise apply $\mathbf { P r _ { w } } [ | T ( \mathbf { w } ) | \leq h ] \leq C ( h / \| T \| _ { 2 } ) ^ { 1 / 3 }$ with the displayed small-value threshold.

Without loss of generality, assume $\lVert b _ { 1 } \rVert _ { 2 } \geq \lVert b _ { 2 } \rVert _ { 2 }$ . If $\| b _ { 1 } \| _ { 2 } \leq \sqrt { \rho } ,$ then $\| q _ { i } - c _ { i } ( \mathbf { w } ) \| _ { 2 } \leq \sqrt { 3 } \kappa + \sqrt { \rho }$ for both $i ,$ giving the first alternative. Henceforth assume $\left\| \boldsymbol b _ { 1 } \right\| _ { 2 } > \sqrt { \rho }$ . Define

$$
\lambda = \frac { \mathbf { E _ { w \sim \mathcal { N } ( 0 , I _ { v ^ { \perp } } ) } } [ b _ { 1 } b _ { 2 } ] } { \| b _ { 1 } \| _ { 2 } ^ { 2 } } , \qquad b _ { 2 } = \lambda b _ { 1 } + r .
$$

Then $| \lambda | \le 1$ and $\mathbf { E _ { w \sim } } { \mathcal { N } } ( 0 , \mathbf { I _ { v ^ { \perp } } } ) \big [ b _ { 1 } r \big ] = 0 . \ \mathrm { I f } \ \| r \| _ { 2 } \leq \rho ,$ , the identity $T = b _ { 1 } ( \lambda c _ { 1 } - c _ { 2 } ) + r c _ { 1 }$ and Fact 2.7 gives $\| \lambda c _ { 1 } - c _ { 2 } \| _ { 2 } \le C ( \| T \| _ { 2 } + B \rho ) / \sqrt { \rho }$ . Thus $\| q _ { 2 } - \lambda q _ { 1 } \| _ { 2 } \le C \left( \kappa + B \sqrt { \rho } + \| T \| _ { 2 } / \sqrt { \rho } \right) \Big )$ , which gives the second alternative.

It remains to consider $\| r \| _ { 2 } > \rho$ . We construct an afine polynomial h for which both $E _ { i } : = c _ { i } - b _ { i } h$ are small. For nonconstant $b _ { 1 }$ , write $\mathbf { w } = t \mathbf { u } + \mathbf { w } _ { \perp }$ and $b _ { 1 } ( \mathbf { w } ) = \beta _ { 1 } t + \beta _ { 0 }$ , where $\beta _ { 1 } > 0$ . If $b _ { 1 }$ is constant, take $\beta _ { 1 } = 0 , \beta _ { 0 } = b _ { 1 }$ , and omit the terms involving t in the second subcase.

Suppose first that $\beta _ { 1 } \geq \tau ^ { \prime }$ and put $s = - \beta _ { 0 } / \beta _ { 1 } . \mathrm { \ A p p l y }$ Lemma 5.20(1) in the Gaussian space $\mathbf { v } ^ { \perp }$ , with distinguished direction u, to divide c<sub>1</sub> by $b _ { 1 }$ . We obtain an afine $h ( \mathbf { w } )$ and $E _ { 1 } ( \mathbf { w } _ { \perp } )$ such that $c _ { 1 } = b _ { 1 } h + E _ { 1 }$ . Restricting $T = b _ { 2 } c _ { 1 } - b _ { 1 } c _ { 2 }$ to $b _ { 1 } = 0$ gives $T ( s \mathbf { u } + \mathbf { w } _ { \perp } ) = r ( s \mathbf { u } + \mathbf { w } _ { \perp } ) E _ { 1 } ( \mathbf { w } _ { \perp } )$ The degree-three restriction bound in Lemma 5.20(2) implies

$$
\| r ( s \mathbf { u } + \mathbf { w } _ { \perp } ) E _ { 1 } ( \mathbf { w } _ { \perp } ) \| _ { 2 } \leq C ( 1 + | s | ) ^ { 3 } \| T \| _ { 2 } \leq C ( \tau ^ { \prime } ) ^ { - 3 } \| T \| _ { 2 } .
$$

Here $\left| \beta _ { 0 } \right| \le \left\| b _ { 1 } \right\| _ { 2 } \le 1$ . To bound the restricted r from below, write

$$
r ( t \mathbf { u } + \mathbf { w } _ { \perp } ) = r _ { 0 } + \gamma t + r _ { \perp } ( \mathbf { w } _ { \perp } ) , \qquad \mathbf { E } \left[ r _ { \perp } \right] = 0 .
$$

The orthogonality ${ \bf E } _ { \bf w } [ b _ { 1 } r ] = 0$ gives $\gamma = s r _ { 0 }$ . Therefore,

$$
\begin{array} { r } { \| r ( s \mathbf { u } + \mathbf { w } _ { \perp } ) \| _ { 2 } ^ { 2 } = ( 1 + s ^ { 2 } ) ^ { 2 } r _ { 0 } ^ { 2 } + \| r _ { \perp } \| _ { 2 } ^ { 2 } \geq \| r \| _ { 2 } ^ { 2 } > \rho ^ { 2 } . } \end{array}
$$

The lower bound in Fact 2.7 now yields

$$
\| E _ { 1 } \| _ { 2 } \leq C \frac { \| T \| _ { 2 } } { \rho ( \tau ^ { \prime } ) ^ { 3 } } .\tag{28}
$$

Suppose next that $\beta _ { 1 } ~ < ~ \tau ^ { \prime }$ . Since $\| b _ { 1 } \| _ { 2 } > { \sqrt { \rho } }$ and $\tau ^ { \prime } \leq \sqrt { \rho } / 4 , | \beta _ { 0 } | \geq \sqrt { \rho } / 2$ . Take $h \ =$ $( c _ { 1 } ) _ { \leq 1 } / \beta _ { 0 } , \| h \| _ { 2 } \leq 2 B / \sqrt { \rho }$ . By the upper bound in Fact 2.7,

$$
\| E _ { 1 } \| _ { 2 } \leq \| ( c _ { 1 } ) _ { 2 } \| _ { 2 } + C B \tau ^ { \prime } / \sqrt { \rho } .\tag{29}
$$

Write $r = r _ { 0 } + r _ { 1 }$ , where $r _ { 1 }$ is its degree-one Hermite component. Orthogonality gives

$$
| r _ { 0 } | | \beta _ { 0 } | = | \beta _ { 1 } \underline { { \mathbf { E } } } [ t r _ { 1 } ( \mathbf { w } ) ] | \leq \beta _ { 1 } \| r _ { 1 } \| _ { 2 } .
$$

It follows that $| r _ { 0 } | \le \| r _ { 1 } \| _ { 2 } / 2$ , and hence $\| r _ { 1 } \| _ { 2 } \ge \| r \| _ { 2 } / 2 > \rho / 2$ . Taking degree-three Hermite components in $T = b _ { 1 } ( \lambda c _ { 1 } - c _ { 2 } ) + r c _ { 1 }$ gives $T _ { 3 } = ( r _ { 1 } ( c _ { 1 } ) _ { 2 } ) _ { 3 } + \beta _ { 1 } \big ( t ( \lambda ( c _ { 1 } ) _ { 2 } - ( c _ { 2 } ) _ { 2 } ) \big ) _ { 3 } .$ For a centered afine polynomial P and a pure degree-two Hermite polynomial Q, we have $\| ( P Q ) _ { 3 } ^ { \circ } \| _ { 2 } \geq \| P \| _ { 2 } \| Q \| _ { 2 }$ Indeed, rotate so that $P = \| P \| _ { 2 } y$ , and write $Q = A h _ { 2 } ( y ) + y D ( \mathbf { r } ) + C ( \mathbf { r } )$ in orthogonal second-chaos components. Then

$$
\lVert ( y Q ) _ { 3 } \rVert _ { 2 } ^ { 2 } = 3 A ^ { 2 } + 2 \lVert D \rVert _ { 2 } ^ { 2 } + \lVert C \rVert _ { 2 } ^ { 2 } \geq \lVert Q \rVert _ { 2 } ^ { 2 } .
$$

Applying this inequality and the upper bound in Fact 2.7 to $T _ { 3 }$ , using $| \lambda | \le 1$ and $\| c _ { i } \| _ { 2 } \leq B$ , gives $\| ( c _ { 1 } ) _ { 2 } \| _ { 2 } \le C ( \| T \| _ { 2 } + B \tau ^ { \prime } ) / \rho$ . Together with (29), this gives

$$
\| E _ { 1 } \| _ { 2 } \leq C \frac { \| T \| _ { 2 } + B \tau ^ { \prime } } { \rho } .\tag{30}
$$

For either choice of $h ,$ put $E _ { 2 } = c _ { 2 } - b _ { 2 } h$ . Since $T = b _ { 2 } E _ { 1 } - b _ { 1 } E _ { 2 }$ , Fact 2.7 gives

$$
\| E _ { 2 } \| _ { 2 } \leq \frac { C } { \| b _ { 1 } \| _ { 2 } } ( \| T \| _ { 2 } + \| b _ { 2 } \| _ { 2 } \| E _ { 1 } \| _ { 2 } ) \leq \frac { C } { \sqrt { \rho } } ( \| T \| _ { 2 } + \| E _ { 1 } \| _ { 2 } ) .
$$

Take $L ( z \mathbf { v } + \mathbf { w } ) = z + h ( \mathbf { w } )$ and $\ell _ { i } ( \mathbf { w } ) = b _ { i } ( \mathbf { w } )$ . Then

$$
q _ { i } - L \ell _ { i } = a _ { i } z ^ { 2 } + E _ { i } ( \mathbf { w } ) , \qquad \| q _ { i } - L \ell _ { i } \| _ { 2 } \leq \sqrt { 3 } \kappa + \| E _ { i } \| _ { 2 } .
$$

Equations (28) and (30) bound the errors in the two subcases. Including the earlier independent and proportional alternatives, one of the three conclusions holds with error at most

$$
C \left( \kappa + B \sqrt { \rho } + \frac { \| T \| _ { 2 } } { \rho ^ { 3 / 2 } ( \tau ^ { \prime } ) ^ { 3 } } + \frac { B \tau ^ { \prime } } { \rho ^ { 3 / 2 } } \right) .\tag{31}
$$

Choice of parameters and conclusion. Let $D : = \eta _ { c } / \alpha ^ { 4 }$ . If $D \geq 1$ or $D ^ { 1 / 2 0 0 0 } \geq \alpha$ , the claimed error bound is at least an absolute constant, up to its implicit constant. The first alternative then follows by taking $\widetilde { q } _ { i } ( \mathbf { w } ) = c _ { i } ( \mathbf { w } )$ , since $\| q _ { i } - c _ { i } \| _ { 2 } \leq \sqrt { 3 } | a _ { i } | + \| b _ { i } \| _ { 2 } = O ( 1 )$ . We henceforth assume $D < 1$ and $D ^ { 1 / 2 0 0 0 } < \alpha$ . In particular,

$$
\alpha ^ { - 3 } < D ^ { - 3 / 2 0 0 0 } , \qquad B \le C ( 1 + \log ( 1 / D ) ) .
$$

By increasing the absolute constant in the asserted error bound, we may also assume that D is smaller than a suficiently small universal constant; the remaining values are covered by the same constant-error approximation.

Choose

$$
\kappa = D ^ { 3 5 / 2 0 0 0 } , \qquad \xi = D ^ { 3 7 / 2 0 0 0 } , \qquad \rho = D ^ { 4 / 2 0 0 0 } , \qquad \tau ^ { \prime } = { \textstyle { \frac { 1 } { 8 } } } D ^ { 8 / 2 0 0 0 } .
$$

These choices ensure $\tau ^ { \prime } \leq \sqrt { \rho } / 4$ . Here and below, $\widetilde O$ and $\widetilde \Omega$ hide only fixed powers of B and absolute constants. By (27),

$$
\| T \| _ { 2 } \le C \alpha ^ { - 3 } ( \kappa B ^ { 3 / 2 } + \eta _ { c } B ^ { 1 / 2 } ) \le \widetilde { O } ( D ^ { 3 2 / 2 0 0 0 } ) .
$$

Substituting into (31), we have

$$
B \sqrt { \rho } = \widetilde { O } ( D ^ { 2 / 2 0 0 0 } ) , \qquad \frac { \| T \| _ { 2 } } { \rho ^ { 3 / 2 } ( \tau ^ { \prime } ) ^ { 3 } } = \widetilde { O } ( D ^ { 2 / 2 0 0 0 } ) , \qquad \frac { B \tau ^ { \prime } } { \rho ^ { 3 / 2 } } = \widetilde { O } ( D ^ { 2 / 2 0 0 0 } ) .
$$

Thus Case 3 gives one of the required alternatives with error ${ \widetilde O } ( D ^ { 2 / 2 0 0 0 } )$

For Cases 1 and 2, take $\eta = C _ { 0 } B D$ , with $C _ { 0 }$ suficiently large that Lemma 5.18 gives $\| R ( q _ { 1 } ^ { ( \mathbf { w } ) } , q _ { 2 } ^ { ( \mathbf { w } ) } ) \| _ { 2 } \leq$ η. The smallness assumption on D ensures $\eta \le \xi ^ { 2 } / 1 0$ . The two claims give

$$
\| U \| _ { 2 } \ge \widetilde { \Omega } ( D ^ { 1 0 9 / 2 0 0 0 } ) , \qquad \| E \| _ { 2 } \le C \frac { B ^ { 4 / 3 } \eta ^ { 1 / 6 } } { \kappa ^ { 2 / 3 } \xi ^ { 4 / 3 } } \le \widetilde { O } ( D ^ { 2 6 0 / 2 0 0 0 } ) , \qquad \| h \| _ { 2 } \le \widetilde { O } ( D ^ { - 1 0 9 / 2 0 0 0 } ) .
$$

Using (25),

$$
\begin{array} { r l r } {  { \| E _ { 1 } \| _ { 2 } \le C ( \frac { \eta } { \| U \| _ { 2 } ^ { 2 } } + \frac { \| E \| _ { 2 } ^ { 2 } } { \| U \| _ { 2 } ^ { 2 } } + \frac { \| E \| _ { 2 } ( 1 + \| h \| _ { 2 } ) } { \| U \| _ { 2 } } ) } } \\ & { } & { \le \widetilde { O } ( D ^ { 1 7 8 2 / 2 0 0 0 } + D ^ { 3 0 2 / 2 0 0 0 } + D ^ { 1 5 1 / 2 0 0 0 } + D ^ { 4 2 / 2 0 0 0 } ) \le \widetilde { O } ( D ^ { 4 2 / 2 0 0 0 } ) . } \end{array}
$$

Equation (26) then gives $\| E _ { 2 } \| _ { 2 } \le \widetilde { \cal O } ( D ^ { 7 / 2 0 0 0 } )$ . The approximately proportional branch has error $\xi / \kappa = D ^ { 2 / 2 0 0 0 }$ . Case 2 is symmetric. Consequently, in every case one of the three alternatives holds with error ${ \widetilde O } ( D ^ { 2 / 2 0 0 0 } )$ . This is at most

$$
\widetilde { \cal O } \left( \alpha ^ { - 1 } D ^ { 1 / 2 0 0 0 } \right) = \widetilde { \cal O } \left( \alpha ^ { - 1 } ( \eta _ { c } / \alpha ^ { 4 } ) ^ { 1 / 2 0 0 0 } \right) ,
$$

which proves the claimed bound on $\boldsymbol { \epsilon } _ { \mathrm { f a c } }$

## 5.5 Completing the Construction of Certificate Polynomials

We now combine the techniques developed so far to prove Theorem 5.1. Our goal is to obtain correlation at least $( \alpha / ( C k ) ) ^ { 2 ^ { \frac { \cdot } { O } ( k ) } }$ while keeping the degree of the certificate at most 2k. Recall from Lemma 5.7 that large randomized error gives a nontrivial amount of Gaussian-weighted decision boundary along v. At a selected boundary, Lemma 5.11 gives a positive contribution when the derivative and the other defining polynomials are quantitatively separated from zero. If these conditions fail, the boundary corresponds to a nearly double root or a nearly shared root. The analysis in Subsection (5.3) handles nearly double roots: either their contribution to the weighted boundary is small, or a defining inequality can be approximated by a Boolean function of w and absorbed into $g ( \mathbf { w } )$ . For the remaining nearly shared roots, Lemma 5.19 either simplifies the instance through approximate proportionality or supplies a common afine polynomial; its approximately w-dependent alternative is excluded on the selected heavy boundary events in the proof below.

Once an afine polynomial L with $\partial _ { \mathbf { v } } L \neq 0$ has been found, Lemma 5.20 gives

$$
q _ { i } ( { \bf x } ) = L ( { \bf x } ) \ell _ { i } ( { \bf x } ) + m _ { i } ( { \bf w } ) , \qquad m _ { i } ( { \bf w } ) = q _ { i } ( r _ { L } ( { \bf w } ) { \bf v } + { \bf w } ) ,
$$

where $r _ { L } ( \mathbf { w } )$ is the root of L along v. Small remainders allow us to approximate additional polynomials using L, whereas remainders with nontrivial norm give separation at its roots by anti-concentration. This suggests grouping the defining polynomials according to whether they are approximately divisible by L. For an instance in which one group has the exact representation $q _ { i } = L \ell _ { i }$ , suppose that roots of L are active boundaries on a set of nontrivial weighted mass and that the cofactors $\ell _ { i }$ and the defining polynomials outside this group are quantitatively separated from zero there. Then Lemma 5.24 constructs a certificate with large correlation. We use separation of these polynomial values, rather than a condition stated only in terms of distances between roots.

Finding a factorization does not by itself guarantee these properties. The newly obtained afine polynomial might not give an active boundary, and replacing the defining polynomials by their approximations may remove the boundary from which the factorization was obtained. We therefore use an iterative construction. Denote by F the current approximating intersection. Each defining polynomial of F is either an undecomposed quadratic or has a recorded exact representation $q _ { i } = \ell _ { i 1 } \ell _ { i 2 }$ , where the afine polynomials may be constant and scalar signs are included in them. Once such a representation is obtained, we keep it and do not factorize the same quadratic again. At each round, the active boundaries of F are associated with undecomposed quadratics or afine polynomials already appearing in these representations. We select one such polynomial whose associated boundaries have nontrivial weighted mass, and handle its nearly double roots using Subsection (5.3) when it is quadratic.

If the selected boundaries satisfy the required separation conditions, the construction from Lemma 5.11 gives a certificate when the selected polynomial is quadratic, and the construction from Lemma 5.24 gives one when it is afine. Otherwise, at the chosen separation scales, other undecomposed quadratics or previously obtained afine polynomials have small values on substantial collections of these selected roots. We use this information to simplify F. If the selected polynomial is afine, say $L ,$ polynomial division allows us to replace the relevant undecomposed quadratics by $L \ell _ { j }$ . An afine polynomial that is small at these roots is instead replaced by a scalar multiple of $L _ { ; }$ consistently at all its current occurrences. Thus a round can make progress even when no quadratic remains to be decomposed.

If the selected polynomial is an undecomposed quadratic $q _ { i } .$ , we apply Lemma 5.19 to the relevant pairs. When all selected partners are approximately proportional to $q _ { i } .$ , we identify them. Otherwise, one pair supplies a decomposition $q _ { i } \approx \ell _ { i 1 } \ell _ { i 2 }$ . We then use these same two afine polynomials for all the selected partners: the division lemma approximates each partner using one of $\ell _ { i 1 } , \ell _ { i 2 }$ , with the choice allowed to depend on the partner. Their small-value events need not occur on the same layers. All these approximations are obtained using the selected roots of the current instance, before making the replacements. We then replace F by the resulting intersection $F ^ { \prime }$ and select an actual boundary of $F ^ { \prime }$ for the next round, without requiring the old boundary to survive.

Each unsuccessful round therefore removes or decomposes an undecomposed quadratic, or identifies afine polynomials already present. Each original quadratic can be removed or decomposed at most once, and at most 2k afine polynomials are introduced in total. Hence there are only $O ( k )$ such changes. Provided the accumulated approximation error is small enough to preserve nontrivial randomized error, the construction must reach an instance for which one of the two certificate constructions succeeds.

It remains to choose the approximation accuracies so that the eventual certificate also correlates with the original target. Write $F _ { 0 } = f , F _ { 1 } , \ldots , F _ { t }$ for the successive instances and $e _ { s } =$ $\mathbf { P r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ F _ { s } ( \mathbf { x } ) \neq F _ { s + 1 } ( \mathbf { x } ) ]$ . If a normalized certificate H satisfies $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ F _ { t } H ] \ge \beta$ , then

$$
\underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ f H ] \geq \beta - \sqrt { \sum _ { s = 0 } ^ { t - 1 } e _ { s } } .
$$

Thus the accumulated disagreement must be small compared with $\beta ^ { 2 }$ , not merely with $\alpha .$ . The factorization and division estimates give replacement errors of the form $e \leq b ^ { - A } u ^ { c }$ , where u is the small-value cutof, b is a fixed power of $\alpha / ( C k )$ , and $A , c > 0$ are universal constants. Consequently, preserving a later correlation requires only a fixed-power dependence of the current cutof on that correlation and b.

There is one additional issue: the certificate contains a product of several polynomial values, so using the same cutof for all of them could introduce an extra factor of k into the accuracy exponent at every round. We avoid this by simplifying several polynomials together. If j diferent polynomials are small, the same afine decomposition handles them with one common error bound and makes at least $\lceil j / 2 \rceil$ changes. We therefore choose diferent cutofs according to the number of changes they allow. In the case where a certificate is obtained, the logarithmic losses from the product are bounded by a geometric sum; in the other case, the replacement error is small enough for the certificate promised after those changes to transfer back. This keeps the accuracy loss per unit of progress at a universal power. After $O ( k )$ changes, including the normalization and Hermite-block extraction, we obtain correlation at least

$$
\left( \frac { \alpha } { C k } \right) ^ { D ^ { O ( k ) } } = \left( \frac { \alpha } { C k } \right) ^ { 2 ^ { O ( k ) } } ,
$$

where $D > 1$ is a universal constant. The detailed choice of cutofs below implements this calculation. We first give the following fact on the stability of Boolean functions under perturbation,

Fact 5.23 (Stability under polynomial perturbations). Suppose that f is determined by w and the signs of N polynomials of degree at most two and norm one. If each polynomial is replaced, at all its occurrences, by a polynomial within δ in $L _ { 2 }$ norm, then the resulting function fe satisfies

$$
\operatorname { \bf { P r } } _ { { \bf { x } } \sim \mathcal { N } ( 0 , { \bf { I } } ) } [ f \neq \widetilde { f } ] \le C N \delta ^ { 2 / 5 } .
$$

For any Boolean functions F, H, with $e = \mathbf { P r _ { x } } [ F \neq H ]$ , and every polynomial $p _ { : }$

$$
| \mathrm { R a n d } _ { F } ( \mathbf { v } ) - \mathrm { R a n d } _ { H } ( \mathbf { v } ) | \leq 2 e , \qquad | \mathbf { \phi } _ { \mathbf { x } \sim N ( 0 , \mathbf { I } ) } [ ( F - H ) p ] | \leq \sqrt { e } \| p \| _ { 2 } .
$$

A certificate around a given afine polynomial. Suppose a collection of defining inequalities has already been written using the same L. The following lemma gives the certificate and relates its size directly to the remainders of the other polynomials. Multiplication of an inequality by a positive constant does not change the target, so the normalizations in the statement cause no restriction.

Lemma 5.24 (Certificate using a common afine polynomial). Let $\| L \| _ { 2 } = 1$ and $a = \partial _ { \mathbf { v } } L \neq 0$ Suppose

$$
f ( \mathbf { x } ) = g ( \mathbf { w } ) \mathbf { 1 } \{ L ( \mathbf { x } ) \ell _ { i } ( \mathbf { x } ) \ge 0 ~ ( i \in S ) , ~ q _ { j } ( \mathbf { x } ) \ge 0 ~ ( j \notin S ) \} ,
$$

where $\varnothing \neq S \subseteq [ k ]$ , each $\ell _ { i }$ is afine or constant with norm one, and the remaining $q _ { j }$ have degree at most two and norm at most an absolute constant. Signs are included in $\ell _ { i }$ and $q _ { j }$ . Put $m = | S |$ and define

$$
Q _ { L } = a \left( \prod _ { i \in S } \ell _ { i } \right) \left( \prod _ { j \not \in S } q _ { j } \right) \left\{ \sum _ { i \in S } \ell _ { i } , \ : \ : \ : \ : m \ : \ : e v e n . \right.\tag{32}
$$

Then deg $Q _ { L } \leq 2 k - 1 , \| M _ { \mathbf { v } } Q _ { L } \| _ { 2 } \leq | a | ( C k ) ^ { C k }$ , and every boundary contribution $\Delta f _ { \mathbf { w } } ( s ) Q _ { L } ( s \mathbf { v } + \mathbf { w } )$ is nonnegative.

More quantitatively, let $r _ { L } ( \mathbf { w } )$ be the root of L and suppose it is a jump of $f _ { \mathbf { w } }$ on a measurable set E with

$$
\begin{array} { r } { \mathbf { E } [ \varphi ( r _ { L } ( \mathbf { w } ) ) \mathbf { 1 } _ { E } ( \mathbf { w } ) ] \geq b > 0 . } \end{array}
$$

For $i \in S$ put $m _ { i } ( \mathbf { w } ) = \ell _ { i } ( r _ { L } ( \mathbf { w } ) \mathbf { v } + \mathbf { w } )$ , and for $j \not \in S$ put $m _ { j } ( \mathbf { w } ) = q _ { j } ( r _ { L } ( \mathbf { w } ) \mathbf { v } + \mathbf { w } )$ . Write $d _ { i } = \operatorname* { m i n } \{ 1 , \| m _ { i } \| _ { 2 } \}$ . If every $d _ { i } > 0$ , then

$$
\frac { \mathbf { E _ { x } } [ f M _ { \mathbf { v } } Q _ { L } ] } { \| M _ { \mathbf { v } } Q _ { L } \| _ { 2 } } \geq \left( \frac { b } { C k } \right) ^ { C k } \prod _ { i = 1 } ^ { k } d _ { i } ^ { 2 } .\tag{33}
$$

Proof of Lemma 5.24. At an active root s of $L ,$ , Fact 5.9 implies that every nonzero outside polynomial $q _ { j }$ is positive. Moreover, for each $i \in S$ with $\ell _ { i } ( s \mathbf { v } + \mathbf { w } ) \neq 0$

$$
\Delta f _ { \mathbf { w } } ( s ) = \mathrm { s i g n } { \big ( } \partial _ { \mathbf { v } } ( L \ell _ { i } ) ( s \mathbf { v } + \mathbf { w } ) { \big ) } = \mathrm { s i g n } { \big ( } a \ell _ { i } ( s \mathbf { v } + \mathbf { w } ) { \big ) } .
$$

Thus all nonzero cofactors have the same sign. The product has the correct sign when $m$ is odd, and the additional sum gives the correct sign when m is even. If a cofactor or an outside polynomial vanishes, $Q _ { L }$ is zero. At a jump not arising from $L ,$ one of these polynomials must vanish, again by Fact 5.9. This proves the nonnegative contribution at every boundary. The degree bound follows directly from (32); Hölder’s inequality, hypercontractivity, and Fact 5.5 give the norm bound.

By Lemma 5.20(1), each $m _ { i }$ is the remainder obtained by dividing the corresponding $\ell _ { i }$ or $q _ { i }$ by L. In particular, deg $m _ { i } ~ \le ~ 2$ . Take $h ~ = ~ ( b / ( C k ) ) ^ { 2 }$ Gaussian anti-concentration gives $\begin{array} { r } { \mathbf { P r _ { w } } [ | m _ { i } ( \mathbf { w } ) | \leq h d _ { i } ] \leq C \sqrt { h } } \end{array}$ . A union bound discards weighted mass at most $b / 2$ for suficiently large C. On the remaining roots all the absolute values are at least $h d _ { i }$ . The sum in the even case has absolute value at least any one of its summands. Since $d _ { i } \leq 1$ and $h \leq 1$ , in both cases

$$
\Delta f _ { \mathbf { w } } ( r _ { L } ) Q _ { L } ( r _ { L } \mathbf { v } + \mathbf { w } ) \geq | a | h ^ { k + 1 } \prod _ { i = 1 } ^ { k } d _ { i } ^ { 2 } .
$$

Apply Lemma 5.6 and divide by the norm bound. The resulting factor $b h ^ { k + 1 } / ( 2 ( C k ) ^ { C k } )$ is at least $( b / ( C k ) ) ^ { C ^ { \prime } k }$ after increasing the absolute constant. Notice that $| a |$ cancels; no separate lower bound on $| \partial _ { \mathbf { v } } L |$ is needed for this normalized correlation estimate. □

Fact 5.25 (Evaluation at selected roots and normalization). Let $0 < b \leq b _ { * } , R _ { b } = \sqrt { 2 0 \log ( 1 / b ) }$ and let $s ( \mathbf { w } )$ be measurable on $E \subset \mathbf { v } ^ { \perp }$ with $| s ( \mathbf { w } ) | \leq R _ { b }$ . The following hold.

(i) If deg $e \le 2$ and $\| e \| _ { 2 } \le \delta _ { i }$ , then, outside a set of probability at most $b ^ { 3 } / 1 0 0$

$$
\operatorname* { s u p } _ { | z | \leq R _ { b } + 1 } \mathopen { } \mathclose \bgroup \left( | e ( z \mathbf { v } + \mathbf { w } ) | + | \partial _ { \mathbf { v } } e ( z \mathbf { v } + \mathbf { w } ) | \aftergroup \egroup | \aftergroup \egroup | \aftergroup \egroup \right) \leq C \delta b ^ { - 4 } .
$$

(ii) If deg $R \leq 2$ and $\begin{array} { r } { \mathbf { P r _ { w } } [ | R ( \mathbf { w } ) | \leq h ] \geq c _ { 0 } b } \end{array}$ , then $\| R \| _ { 2 } \le C _ { c _ { 0 } } b ^ { - 4 } h$

(iii) If deg $P \leq 2 , \| P \| _ { 2 } = 1$ , and

$$
\begin{array} { r } { \mathbf { E } [ \varphi ( s ( \mathbf { w } ) ) \mathbf { 1 } _ { E } ( \mathbf { w } ) \mathbf { 1 } \{ | P ( s ( \mathbf { w } ) \mathbf { v } + \mathbf { w } ) | \leq u \} ] \geq b , \qquad u \leq b ^ { A _ { 0 } } , } \end{array}
$$

for suficiently large absolute $A _ { 0 }$ , then $P$ is nonconstant. Writing $P = \mu + \sigma H$ , where $\mathbf { E _ { x } } [ H ] = 0$ and $\operatorname { V a r } _ { \mathbf { x } } ( H ) = 1$ , gives

$$
\sigma \geq \frac { c } { 1 + \log ( 1 / b ) } , \qquad | \mu | / \sigma \leq C ( 1 + \log ( 1 / b ) ) .
$$

Proof of Fact 5.25. For (i), write $e = a ( z ^ { 2 } - 1 ) + B ( \mathbf { w } ) z + C ( \mathbf { w } )$ . Orthogonality gives $2 a ^ { 2 } + \| B \| _ { 2 } ^ { 2 } +$ $\| C \| _ { 2 } ^ { 2 } \le \delta ^ { 2 }$ . Chebyshev bounds $| B | , | C |$ by $C \delta b ^ { - 3 }$ outside probability $C b ^ { 6 }$ . Since $R _ { b } + 1 \leq b ^ { - 1 }$ , the asserted uniform bound follows; decrease $b _ { * }$ to ensure $C b ^ { 6 } \leq b ^ { 3 } / 1 0 0$ . For (ii), degree-two Gaussian

anti-concentration gives $c _ { 0 } b \leq C ( h / \| R \| _ { 2 } ) ^ { 1 / 2 }$ when $R \neq 0$ . The resulting bound $C c _ { 0 } ^ { - 2 } b ^ { - 2 } h$ implies the stated one; the zero polynomial is immediate.

For (iii), a constant of norm one cannot satisfy the hypothesis, so $\sigma > 0$ . Decompose the centered unit-variance polynomial H in the same way as in (i). Gaussian tails for its afine and quadratic coeficient polynomials imply, outside probability $b / 1 0 0$

$$
\operatorname* { s u p } _ { | z | \leq R _ { b } } | H ( z \mathbf { v } + \mathbf { w } ) | \leq C ( 1 + \log ( 1 / b ) ) .
$$

The weighted event has ordinary probability at least $b / \varphi ( 0 )$ and intersects this set. Thus $| \mu | \leq$ $u + C \sigma ( 1 + \log ( 1 / b ) )$ ). Using $\mu ^ { 2 } + \sigma ^ { 2 } = 1 \qquad $ and $u \leq 1 / 4$ proves the lower bound on $\sigma$ and then the bound on $| \mu | / \sigma$ □

The iterative construction. Algorithm 2 summarizes the construction used in Theorem 5.1. It is a proof procedure with access to the defining polynomials and exact Gaussian boundary quantities, not a sample-based learning algorithm. We retain the normalization and replacement conventions of the proof. In particular, t counts changes, not rounds: a change is the removal or decomposition of an old undecomposed quadratic, or the disappearance of an old afine polynomial from the current expressions, counting proportional afine polynomials only once. An afine polynomial that remains as a new quotient is not counted as discarded.

Choose the universal constants as in the proof and set

$$
\begin{array} { r l r } & { \rho = \alpha / ( C k ) , \quad \quad b = \rho ^ { 8 } , \quad \beta _ { t } = \rho ^ { K k D ^ { 3 k - t } } , } \\ & { \theta _ { t } = \rho ^ { \Lambda k D ^ { 3 k - t - 1 } } , \quad \quad } & { u _ { t , j } = \rho ^ { \Lambda k D ^ { 3 k - t - \lceil j / 2 \rceil } } . } \end{array}\tag{34}
$$

In a round with selected roots $s ( \mathbf { w } )$ on $E ,$ put m ${ { \bf \xi } _ { E } } ( B ) = { \bf E } _ { { \bf w } } [ \varphi ( s ( { \bf w } ) ) { \bf 1 } _ { B } ( { \bf w } ) ]$ for $B \subseteq E$ , where w is standard Gaussian on $\mathbf { v } ^ { \perp }$ ; set $R _ { \rho } = \sqrt { 2 0 \log ( 1 / \rho ) }$ . If $P _ { 1 } , \dots , P _ { M }$ are the other diferent current polynomials, a batch $( j , B )$ means

$$
\begin{array} { r l r } & { } & { j \in [ M ] , \qquad B \subseteq \{ P _ { 1 } , \dots , P _ { M } \} , \qquad | B | = j , } \\ & { } & { m _ { E } ( \{ \mathbf { w } \in E : | P ( s ( \mathbf { w } ) \mathbf { v } + \mathbf { w } ) | \leq u _ { t , j } \} ) \geq b \quad \mathrm { f o r ~ e v e r y ~ } P \in \mathcal { B } . } \end{array}\tag{35}
$$

For $M = 0$ , no such batch exists. The expectations involving x below are under the standard Gaussian distribution on $\mathbb { R } ^ { d }$

The proof below excludes the approximately w-dependent alternative in the quadratic-pair branch and justifies all choices in the construction. After each update ending at $t ^ { \prime } { } _ { ; }$ , it establishes

$$
\begin{array} { r } { \mathbf { P r } [ F \neq F ^ { \prime } ] \leq \beta _ { t ^ { \prime } } ^ { 2 0 } , \qquad \mathrm { R a n d } _ { F ^ { \prime } } ( \mathbf { v } ) \geq \alpha / 2 . } \end{array}
$$

The small-discriminant branch makes at least one change; a batch of $j$ partners makes at least $\lceil j / 2 \rceil$ changes. A successful round gives $\mathbf { E _ { x } } [ F H ] \geq \beta _ { t }$ , and stability transfers this to $\mathbf { E } _ { \mathbf { x } } [ f H ] \geq \beta _ { 0 } / 2$ . The final paragraph of the proof then extracts the required normalized block $h _ { r } ( \mathbf { v } \cdot \mathbf { x } ) p ( \mathbf { w } )$ . The error estimates are guarantees of the proof, not additional tests performed by the procedure.

Proof of Theorem 5.1. Algorithm 2 records the construction. We may assume $0 < \alpha \le 1 / 2$ . Write $\mathbf { x } = z \mathbf { v } + \mathbf { w }$ , where $z \sim \mathcal { N } ( 0 , 1 )$ and $\mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { \mathbf { v } ^ { \perp } } )$ are independent. We construct successive approximating intersections, starting with $F = f$ . At each round we select an actual boundary of $F$ and either obtain a certificate or replace F by a simpler intersection.

Algorithm 2 Iterative construction of a directional certificate   
1: Input: The defining representation of the target function $f , { \mathrm { a } }$ unit direction $\mathbf { v } ,$ and $0 < \alpha \le 1 / 2$ with   
Rand ${ \bf \Pi } _ { f } ( { \bf v } ) \geq \alpha$   
2: Output: $H = M _ { \mathbf { v } } Q / \lVert M _ { \mathbf { v } } Q \rVert _ { 2 }$ <sub>2</sub> with deg $Q \leq 2 k - 1$ and $\begin{array} { r } { \mathbf { E } _ { \mathbf { x } } [ f H ] \geq \beta _ { 0 } / 2 . } \end{array}$   
3: Set $F  f$ and $t \gets 0 ;$ normalize and simplify as in the proof.   
4: while no certificate has been returned do   
5: Select a current polynomial $q _ { 0 }$ and actual jumps $s ( \mathbf { w } )$ of $F _ { \mathbf { w } }$ on $E ,$ associated with $q _ { 0 } .$ , with $m _ { E } ( E ) \geq$   
$2 \rho ^ { 4 }$ and $| s ( \mathbf { w } ) | \leq R _ { \rho } .$   
6: if $q _ { 0 } = a z ^ { 2 } + B ( \mathbf { w } ) z + C ( \mathbf { w } )$ is undecomposed and $\| B ^ { 2 } - 4 a C \| _ { 2 } \leq \theta _ { t }$ then   
7: Approximate its inequality by a Boolean function of w using Subsection (5.3); absorb it into the   
gate and remove $q _ { 0 } ,$ , giving $F ^ { \prime }$   
8: else   
9: if $q _ { 0 }$ is an undecomposed quadratic then   
10: Discard from E the layers where $| \partial _ { \mathbf { v } } q _ { 0 } ( s ( \mathbf { w } ) \mathbf { v } + \mathbf { w } ) | \leq \theta _ { t } .$   
11: List the other diferent current polynomials $P _ { 1 } , \dots , P _ { M } ;$ search for a batch $( j , B )$ satisfying (35).   
12: if no such batch exists then   
13: If $q _ { 0 } = q _ { i }$ is quadratic, take $\begin{array} { r } { Q = ( \partial _ { \mathbf { v } } \prod _ { j } q _ { j } ) } \end{array}$ ; otherwise use $Q _ { L }$ from Lemma 5.24.   
14: Reduce repeated afine powers by parity in the multiplicative part, as in the proof.   
15: return $H = M _ { \mathbf { v } } Q / \lVert M _ { \mathbf { v } } Q \rVert _ { 2 }$   
16: Fix one batch $( j , B )$ and put $\boldsymbol { u } \gets \boldsymbol { u } _ { t , j }$   
17: if $q _ { 0 } = L$ is afine then   
18: Keep L fixed. For each $P \in B ,$ use Lemma 5.20 to write $P = L \ell _ { P } + m _ { P } ( \mathbf { w } )$ ; prescribe $P \mapsto L \ell _ { P }$   
19: else   
20: Orient and normalize the pairs $( q _ { 0 } , P )$ and apply Lemma 5.19 as in the proof.   
21: if every selected partner is in the proportionality case then   
22: Prescribe scalar-multiple replacements using one common polynomial from $\{ q _ { 0 } \} \cup B ,$ choosing   
an afine one when available.   
23: else   
24: Obtain $q _ { 0 } \approx \kappa L _ { 1 } L _ { 2 }$ from one pair; prescribe $q _ { 0 } \mapsto \kappa L _ { 1 } L _ { 2 } .$   
25: For each $P \in B ,$ choose one $L ( P ) \in \{ L _ { 1 } , L _ { 2 } \}$ using its small-value event at the original   
selected roots $s ( \mathbf { w } )$   
26: Apply Lemma 5.20 with divisor $L ( P ) { \mathrm { ; } }$ prescribe $P \mapsto L ( P ) \ell _ { P }$   
27: Apply the prescribed replacements simultaneously at all old occurrences, without substituting   
again into new quotients, giving $F ^ { \prime }$   
28: Normalize $F ^ { \prime } { } _ { ; }$ retain scalar signs and decomposed forms, identify proportional copies, and simplify   
repeated, constant, and square inequalities.   
29: Let $\Delta t$ be the actual number of changes from the old representation of $F$ to the resulting representation   
of $F ^ { \prime }$   
30: Set $F  F ^ { \prime }$ and $t \gets t + \Delta t ;$ restart by selecting an actual boundary of this new $F .$

The representations and allowed changes. Normalize every undecomposed quadratic and every recorded afine polynomial to norm one, keeping scalar signs in the defining inequalities. An inequality that has been decomposed remains a product of afine polynomials; we never treat this product as an undecomposed quadratic again. Proportional afine polynomials are represented by the same normalized polynomial, with their scalars retained. Delete repeated inequalities and simplify constant and square inequalities. There are at most k inequalities and at most 2k diferent nonconstant polynomials in these expressions, counting each undecomposed quadratic and each recorded afine polynomial once.

An afine replacement $M \mapsto c L$ is made at every current occurrence of M, not by independently changing its coeficients in diferent inequalities. Replacements in a round are simultaneous: we do not substitute again into the newly obtained quotients. Count one change for each old undecomposed quadratic no longer present as such, and one for each old afine polynomial absent from the resulting expressions. In particular, an afine polynomial that still occurs as a new quotient is not counted as discarded.

Let t be the total number of changes so far, not the number of rounds. Each original quadratic is removed or decomposed at most once. New afine polynomials arise only from such decompositions, with at most two introductions per quadratic; a batch of $q + 1$ quadratics sharing two new afine polynomials introduces at most $q + 2 \leq 2 ( q + 1 )$ . Thus at most 2k afine polynomials are introduced in total, including initial afine inequalities and later reintroductions. If N diferent polynomials currently remain, then

$$
t + N \leq 3 k .\tag{36}
$$

Indeed, removed and remaining undecomposed quadratics total at most $k ,$ while discarded and remaining afine polynomials total the number introduced. This also rules out cycling: without a new quadratic decomposition an afine merger strictly reduces the current collection.

Accuracies to be maintained. Let $A > 1$ and $c \in ( 0 , 1 )$ be universal constants large and small enough, respectively, for the replacement estimates from Lemmas 5.19 and 5.20, Lemma 5.13, and Fact 5.23. The applications below establish disagreement bounds $b ^ { - A } u ^ { c }$ , for $u \leq b ^ { A }$ . Choose universal constants $K , \Lambda , D > 1$ , in that order, such that

$$
K \geq 4 C _ { * } , \qquad c \Lambda \geq 2 0 K + 8 A + 2 0 , \qquad { \frac { \Lambda } { D } } + { \frac { 6 \Lambda } { D - 1 } } \leq { \frac { K } { 2 } } ,\tag{37}
$$

where $C _ { * }$ absorbs the certificate norm and boundary-mass constants below. Set

$$
\rho = \frac { \alpha } { C k } , \qquad b = \rho ^ { 8 } , \qquad \beta _ { t } = \rho ^ { K k D ^ { 3 k - t } } \quad ( 0 \leq t \leq 3 k ) .
$$

We take $C$ suficiently large throughout. Every replacement ending at a new count $t ^ { \prime }$ will satisfy

$$
\begin{array} { r } { \mathbf { P r } [ F \neq F ^ { \prime } ] \leq \beta _ { t ^ { \prime } } ^ { 2 0 } . } \end{array}\tag{38}
$$

Since each replacement increases $t ,$ the accumulated disagreement is at most $\begin{array} { r } { \sum _ { s = 1 } ^ { 3 k } \beta _ { s } ^ { 2 0 } \le 3 k \rho ^ { 2 0 K k } < } \end{array}$ $\alpha / 4$ . Fact 5.23 then ensures Rand ${ \bf \Lambda } _ { F } ( { \bf v } ) \ge \alpha / 2$ . Initially this holds without any replacements; below we verify (38) for each possible update. In particular, positive randomized error ensures $N \geq 1$ and hence $t \le 3 k - 1$ at the start of every round.

Selecting a boundary and handling nearly double roots. At the beginning of a round let t be the current count. By Lemma 5.7, whose proof applies to every finite-jump Boolean layer,

$$
{ \underset { \mathbf { w } } { \mathbf { E } } } \left[ \sum _ { s \in S _ { \mathbf { w } } } \varphi ( s ) \right] \geq c \alpha .
$$

Assign each jump to a sign-changing zero of an undecomposed quadratic or a recorded afine polynomial. There are at most 2k choices and at most two assigned roots per choice on each layer. Pigeonhole to one polynomial $q _ { 0 }$ and retain its assigned root of largest Gaussian density on each layer. Discarding roots outside $R _ { \rho } = \sqrt { 2 0 \log ( 1 / \rho ) }$ loses at most $2 k \varphi ( R _ { \rho } ) = { \cal O } ( k \rho ^ { 1 0 } )$ in weighted mass. We obtain measurable selected roots $s ( \mathbf { w } )$ on a set $E$ with

$$
\begin{array} { r } { \mathbf { E } [ \varphi ( s ( \mathbf { w } ) ) \mathbf { 1 } _ { E } ( \mathbf { w } ) ] \geq 2 \rho ^ { 4 } , \qquad | s ( \mathbf { w } ) | \leq R _ { \rho } . } \end{array}
$$

Use fixed ordering and tie-breaking for these selections.

If $q _ { 0 }$ is quadratic, write $q _ { 0 } = a z ^ { 2 } + B ( \mathbf { w } ) z + C ( \mathbf { w } )$ and put $\Delta = B ^ { 2 } - 4 a C$ and $\theta = \rho ^ { \Lambda k D ^ { 3 k - t - 1 } }$ $\mathrm { I f ~ } \| \Delta \| _ { 2 } \leq \theta$ , use Fact 5.25(iii) to write $q _ { 0 } = \mu + \sigma p$ , where $p$ has mean zero and variance one, and $\sigma \geq c / ( 1 + \log ( 1 / b ) )$ . The discriminant of $q _ { 0 } / \sigma$ is $\Delta / \sigma ^ { 2 }$ . Lemma 5.13 therefore approximates the inequality by a Boolean function of $\mathbf { w } ,$ with error at most $b ^ { - A } \theta ^ { c }$ , after increasing A and decreasing c. Absorb that function into the gate. This removes a quadratic, so $t ^ { \prime } \geq t + 1$ , and

$$
b ^ { - A } \theta ^ { c } = \rho ^ { c \Lambda k D ^ { 3 k - t - 1 } - 8 A } \leq \beta _ { t + 1 } ^ { 2 0 } \leq \beta _ { t ^ { \prime } } ^ { 2 0 } .
$$

Thus the update satisfies (38); start the next round. Otherwise, the root–discriminant identity and the argument of Lemma 5.12 give

$$
\mathbf { P r } [ \exists s : q _ { 0 } ( s \mathbf { v } + \mathbf { w } ) = 0 , | \partial _ { \mathbf { v } } q _ { 0 } ( s \mathbf { v } + \mathbf { w } ) | \leq \theta ] \leq C \sqrt { \theta } .
$$

Discard these layers. The remaining selected mass is at least $\rho ^ { 4 }$ , and $| \partial _ { \mathbf { v } } q _ { 0 } | > \theta$ there. If $q _ { 0 }$ is afine, no such step is needed; define the same $\theta \leq 1$ for the bounds below. Write $m ( B ) = { \bf E } _ { \bf w } [ \varphi ( s ( { \bf w } ) ) { \bf 1 } _ { B } ( { \bf w } ) ]$ for $B \subseteq E$

A finite test for success or simplification. Let $P _ { 1 } , \dots , P _ { M }$ be the other diferent normalized polynomials currently appearing. By (36), $M \leq 3 k - t - 1$ . Put

$$
d _ { j } = \lceil j / 2 \rceil , \qquad u _ { j } = \rho ^ { \Lambda k D ^ { 3 k - t - d _ { j } } } \quad ( j \in [ M ] ) .
$$

We ask whether, for some $j ,$ , at least $j$ diferent polynomials satisfy

$$
m \big ( \{ \mathbf { w } \in E : | P _ { i } ( s ( \mathbf { w } ) \mathbf { v } + \mathbf { w } ) | \leq u _ { j } \} \big ) \geq b .\tag{39}
$$

If this happens, choose any $j$ such polynomials and set $u = u _ { j } ;$ ; they are the partners to be simplified in this round.

Suppose first that it does not happen. Relabel the polynomials so that

$$
m { \big ( } \{ | P _ { j } ( s ( \mathbf { w } ) \mathbf { v } + \mathbf { w } ) | \leq u _ { j } \} { \big ) } < b \quad { \mathrm { f o r ~ e v e r y ~ } } j \in [ M ] .
$$

Such a relabeling is obtained backwards: when assigning the jth position, $j$ polynomials remain unassigned, and fewer than $j$ polynomials in the whole collection satisfy (39) at $u _ { j }$ . Choose one of the remaining polynomials that does not satisfy it. Discard the union of these small-value events. Its weighted mass is at most $M b \le \rho ^ { 4 } / 2$ , so the remaining roots have mass at least $\rho ^ { 4 } / 2$ and $| P _ { j } | > u _ { j }$ for every $j$

If $q _ { 0 } = q _ { i }$ is an undecomposed defining quadratic, take $\begin{array} { r } { Q = ( \partial _ { \mathbf { v } } \prod _ { i } q _ { j } ) } \end{array}$ , as in Lemma 5.11. If $q _ { 0 } = L$ is afine, use the polynomial from Lemma 5.24. Fact 5.9 gives the required jump orientations. In the multiplicative part of $Q ,$ expand only the recorded afine products and reduce each repeated power $L ^ { a }$ to $L$ for odd $a$ and to $L ^ { 2 }$ for even $a .$ This preserves signs and zeros, and hence nonnegative contributions at every boundary. Each diferent polynomial then occurs at most twice. At active afine roots the cofactors have a common sign, so the possible additional sum costs at most one more clipped value.

The degree is at most $2 k - 1$ . Hypercontractivity and Fact 5.5 give $\| M _ { \mathbf { v } } Q \| _ { 2 } \leq ( C k ) ^ { C k }$ in the quadratic case and $\Vert M \mathbf { v } Q \Vert _ { 2 } \leq | \partial _ { \mathbf { v } } L | ( C k ) ^ { C k }$ in the afine case. In the latter case the same constant derivative occurs in the boundary contribution and cancels after normalization. Lemma 5.6 therefore gives $H = M _ { \mathbf { v } } Q / \lVert M _ { \mathbf { v } } Q \rVert _ { 2 }$ satisfying

$$
\begin{array} { l } { \displaystyle \mathbf { \frac { E } { x } } [ F H ] \geq \rho ^ { C * k } \theta \prod _ { j = 1 } ^ { M } u _ { j } ^ { 3 } = \rho ^ { C * k + \Lambda k D ^ { 3 k - t - 1 } + 3 \Lambda k \sum _ { j = 1 } ^ { M } D ^ { 3 k - t - d _ { j } } } \geq \rho ^ { K k D ^ { 3 k - t } } = \beta _ { t } . } \end{array}
$$

Here ${ \textstyle \sum _ { i = 1 } ^ { M } D ^ { 3 k - t - d _ { j } } \leq 2 D ^ { 3 k - t } / ( D - 1 ) }$ , and we used (37). For $M = 0$ the same argument uses an empty product. We stop with this certificate.

An unsuccessful round at an afine boundary. Now suppose (39) holds for the selected j partners, and $q _ { 0 } = L$ is afine. Its derivative along v is nonzero, and the selected points are its roots. For each selected partner $P _ { \mathrm { : } }$ the small-value event has ordinary probability at least cb. Lemma 5.20(1),(3) gives

$$
P = L \ell + m _ { P } ( { \bf w } ) , \qquad \deg \ell \leq \deg P - 1 , \qquad \| m _ { P } \| _ { 2 } \leq C b ^ { - 2 } u .\tag{40}
$$

Replace every selected undecomposed quadratic by $L \ell$ and every selected afine polynomial by the corresponding scalar multiple of $L .$ Thus the selected L is kept fixed, and already decomposed inequalities remain decomposed.

An unsuccessful round at a quadratic boundary. Suppose instead that $q _ { 0 }$ is an undecomposed quadratic, including the sign of its defining inequality. For each selected partner $P ,$ fix its sign on the side where the current intersection is one, retaining weighted mass at least $b / 2$ . By Fact 5.9 the selected root is a jump of this oriented pair. Fact 5.25(iii) allows us to divide their centered parts by standard deviations at least $c / ( 1 + \log ( 1 / b ) )$ . The normalized pair has thresholds $O ( 1 + \log ( 1 / b ) )$ and partner value at most $C u ( 1 + \log ( 1 / b ) )$ . Apply Lemma 5.19 and rescale back. Its error is at most $\delta _ { \mathrm { p a i r } } = C b ^ { - C _ { 1 } } u ^ { c _ { 1 } } \leq b ^ { 1 1 0 }$ , for universal $C _ { 1 } , c _ { 1 } > 0$ , by taking the cutof exponent A large enough. The alternative of being nearly independent of v is excluded: if a unit-norm polynomial is within $\delta _ { \mathrm { p a i r } }$ of $R ( \mathbf { w } )$ , then $\| R \| _ { 2 } \geq 1 - \delta _ { \mathrm { p a i r } }$ , whereas Fact $5 . 2 5 ( \mathrm { i } ) , ( \mathrm { i i } )$ on its selected small-value event gives

$$
\| R \| _ { 2 } \leq C b ^ { - 4 } ( u + C \delta _ { \mathrm { p a i r } } b ^ { - 4 } ) < 1 / 2 .
$$

If every selected partner is approximately proportional to $q _ { 0 }$ , retain one common polynomial, choosing an afine one when available, and replace the others by the appropriate scalar multiples. Unit norms ensure the scalars are bounded above and below in absolute value. Otherwise one pair supplies, after normalization,

$$
\begin{array} { r } { \| q _ { 0 } - \kappa L _ { 1 } L _ { 2 } \| _ { 2 } \leq \delta _ { 0 } \leq C \delta _ { \mathrm { p a i r } } , \qquad \| L _ { 1 } \| _ { 2 } = \| L _ { 2 } \| _ { 2 } = 1 , \qquad c _ { 0 } \leq | \kappa | \leq C _ { 0 } . } \end{array}\tag{41}
$$

The last bounds follow from Fact 2.7. Keep these same $L _ { 1 } , L _ { 2 }$ for all selected partners. At the original selected roots of $q _ { 0 }$ , Fact 5.25(i) gives $| L _ { 1 } L _ { 2 } | \le C \delta _ { 0 } b ^ { - 4 }$ outside probability $b ^ { 3 } / 1 0 0$ . For each partner, intersect with its small-value event and choose the smaller afine value. One fixed $L \in \{ L _ { 1 } , L _ { 2 } \}$ then satisfies

$$
| P ( s \mathbf { v } + \mathbf { w } ) | \leq u , \qquad | L ( s \mathbf { v } + \mathbf { w } ) | \leq C \sqrt { \delta _ { 0 } } b ^ { - 2 }
$$

on probability at least $c b$

Write $L ( z \mathbf { v } + \mathbf { w } ) = a z + B ( \mathbf { w } )$ . We have $| a | \geq b ^ { 4 } { \mathrm { : } }$ otherwise $\| B \| _ { 2 } \ge 1 / 2$ , but $| B | \le b ^ { 4 } R _ { b } +$ $C \sqrt { \delta _ { 0 } } b ^ { - 2 } \leq \dot { b } ^ { 3 }$ on probability at least cb, contradicting afine anti-concentration. Lemma 5.20(1),(2) now gives $P = L \ell + m _ { P }$ with $\| \ell \| _ { 2 } \le C b ^ { - 8 }$ . Part (3) of that lemma gives

$$
\| m _ { P } \| _ { 2 } \le C b ^ { - 2 } \bigl ( u + \sqrt { \delta _ { 0 } } b ^ { - 1 0 } \sqrt { \log ( 2 / b ) } \bigr ) \le C b ^ { - C _ { 2 } } ( u + \sqrt { \delta _ { 0 } } ) .\tag{42}
$$

Replace $q _ { 0 }$ by $\kappa L _ { 1 } L _ { 2 }$ and each partner by its approximation $L \ell .$ Diferent partners may use diferent choices of $L ,$ and their events need not overlap. All estimates use the old selected roots; no boundary of the new intersection has yet been assumed to survive.

Progress and error of an unsuccessful round. Make the replacements simultaneously as specified at the start, retain scalar signs, normalize the new afine polynomials, and simplify repeated, constant, and square inequalities. Let $F ^ { \prime }$ be the resulting intersection. The bounds above approximate every changed normalized polynomial within $C b ^ { - C _ { 3 } } u ^ { c _ { 3 } }$ , for universal $C _ { 3 } , c _ { 3 } > 0$ . Since their number is at most $2 k \leq b ^ { - 1 }$ , Fact 5.23 gives

$$
e : = \operatorname * { P r } _ { \mathbf x } [ F \neq F ^ { \prime } ] \leq b ^ { - A } u ^ { c } .\tag{43}
$$

This justifies the universal constants used earlier; the small-discriminant replacement is covered by the same constants after adjustment.

To count the changes, suppose $q$ selected partners are quadratic and a are afine, so $j = q + a$ At an afine boundary all $q$ quadratics are decomposed. Of the selected afine polynomials, at most $q$ can remain among the new quotients. Thus there are at least $q + \operatorname* { m a x } \{ 0 , a - q \} \geq \lceil j / 2 \rceil$ changes. In the proportional case there are at least $j$ changes. In the common-afine quadratic case, $q _ { 0 }$ and all $q$ quadratic partners are decomposed, and at most $q + 2$ selected afine polynomials can remain among $L _ { 1 } , L _ { 2 }$ and the new quotients. There are therefore at least $q + 1 + \operatorname* { m a x } \{ 0 , a - q - 2 \} \geq \lceil j / 2 \rceil$ changes. These counts allow a new quotient to coincide with an old polynomial. Hence the new count satisfies $t ^ { \prime } \geq t + d _ { j }$ . Since $u = u _ { j }$

$$
e \leq \rho ^ { c \Lambda k D ^ { 3 k - t - d _ { j } } - 8 A } \leq \beta _ { t + d _ { j } } ^ { 2 0 } \leq \beta _ { t ^ { \prime } } ^ { 2 0 } .
$$

This proves (38). All cutofs used above are valid: $u _ { j } , \theta \leq b ^ { A } , R _ { \rho } \leq R _ { b }$ , and $2 k \leq b ^ { - 1 }$ . Replace $F$ by $F ^ { \prime }$ and begin the next round by selecting an actual boundary of $F ^ { \prime }$ . We do not continue at an assumed boundary of $L _ { 1 }$ or $L _ { 2 }$

Termination and transfer to the original target. Every unsuccessful round increases t. By (36), at most 3k changes are possible; if no polynomial remained, the intersection would depend only on w and have zero randomized error. Since every update preserves randomized error at least $\alpha / 2$ a successful round must occur. Let $t _ { * }$ be its count, and let H be its normalized certificate. The sequence $\beta _ { t }$ is increasing. Thus the sum of the disagreement errors from all earlier rounds is at most $3 k \beta _ { t _ { * } } ^ { 2 0 }$ , and Fact 5.23 gives

$$
\begin{array} { r } { \mathbf { E } [ f H ] \geq \beta _ { t _ { * } } - \sqrt { 3 k } \beta _ { t _ { * } } ^ { 1 0 } \geq \frac { 1 } { 2 } \beta _ { 0 } . } \end{array}
$$

Finally deg $Q \leq 2 k - 1$ , so Fact 5.5 gives

$$
H ( z \mathbf { v } + \mathbf { w } ) = \sum _ { r = 1 } ^ { 2 k } h _ { r } ( z ) p _ { r } ( \mathbf { w } ) , \qquad \deg p _ { r } \leq 2 k - r , \qquad \sum _ { r } \| p _ { r } \| _ { 2 } ^ { 2 } = 1 .
$$

One summand has correlation at least $\beta _ { 0 } / ( 4 k )$ . Normalize its nonzero $p _ { r }$ , which can only increase that positive correlation. Since $K k D ^ { 3 k } = \dot { 2 } ^ { O ( k ) }$ , the resulting r, p satisfy

$$
\| p \| _ { 2 } = 1 , \qquad r + \deg p \leq 2 k , \qquad { \bf E } [ f ( { \bf x } ) h _ { r } ( { \bf v } \cdot { \bf x } ) p ( { \bf w } ) ] \geq \left( { \frac { \alpha } { C k } } \right) ^ { 2 ^ { O ( k ) } } .
$$

## 6 Eficient Correlation Finding through Influence Matrix

By Theorem 5.1, we know that if there is a direction $\mathbf { v } \in \mathbb { S } ^ { d - 1 }$ , such that Rand ${ \bf \nabla } _ { f } ( { \bf v } ) \geq \alpha$ , then there exists some polynomial $p ( \mathbf { w } )$ such that $\mathbf E _ { \mathbf x \sim \mathcal { N } ( 0 , \mathbf I ) } [ y h _ { k } ( \mathbf v \cdot \mathbf x ) p ( \mathbf w ) ] > \mathrm { p o l y } ( \alpha )$ . In this section, we give an algorithm to approximately find such a direction v. Notice that in general $k + \deg ( p ) \geq 2$ , so if x is sampled from a general distribution, then finding such a direction would require us to solve a higher-order tensor problem, which is generally hard [HL13]. However, the structure of the Gaussian distribution allows us to do this much more eficiently.

Suppose that $k + \deg ( p ) \leq \ell ,$ then we know that for every v and every p, $\mathbf { E } _ { ( \mathbf { x } , y ) \sim D } [ y h _ { k } ( \mathbf { v } \cdot \mathbf { x } ) p ( \mathbf { w } ) ]$ only depends on the projection of y onto the subspace spanned by Hermite polynomials with degree at most ℓ. Thus, we instead conside $F ( \mathbf { x } ) = \mathrm { p r o j } _ { P < \ell } ( y )$ , the low degree projection of $y .$ The central part of our analysis relies on Lemma 6.3, which consider the influential matrix defined by $G _ { \ell } : = $ $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \boldsymbol { \nabla } F \boldsymbol { \nabla } F ^ { \top } ]$ ]. In fact, denote by $i \in [ \ell ] , B _ { i } ( \mathbf { v } ) = \operatorname* { s u p } _ { p ( \mathbf { w } ) \in \mathcal { P } _ { \leq \ell - i } , \| p \| \leq 1 } \mathbf { E } _ { ( \mathbf { x } , y ) \sim D } [ y h _ { i } ( \mathbf { v } \cdot \mathbf { x } ) p ( \mathbf { w } ) ]$ then for every v, we have $\begin{array} { r } { \mathbf { v } ^ { \top } G _ { \ell } \mathbf { v } = \sum _ { i = 1 } ^ { \ell } i B _ { i } ^ { 2 } ( \mathbf { v } ) } \end{array}$ . This immediately gives two consequences. First, the top eigenvector must have large correlation with respect $y .$ Second, any eigenvector of $G _ { \ell }$ with a small eigenvalue must have tiny correlation with respect to y. In fact, since $\operatorname { t r } ( G ) \leq \ell ,$ there cannot be too many large eigenvectors of $G _ { \ell } .$ . Combine this with Theorem 5.1, we can find a subspace V of dimension $\mathrm { p o l y } _ { k } ( 1 / \alpha )$ that contains all directions in $\mathbb { R } ^ { d }$ with a randomized error at least α with respect to any intersections of k degree 2 PTFs $f ( \mathbf { x } )$ in one shot.

However, we still need to estimate $G _ { \ell }$ well enough. Fortunately, the under Hermite basis, the coeficient of F with respect to a Hermite polynomial $H _ { \gamma }$ is just $\mathbf { E } _ { ( \mathbf { x } , y ) \sim D } [ y H _ { \gamma } ( \mathbf { x } ) ]$ . Using basic properties of Hermite polynomials, this gives us a simple way to estimate $G _ { \ell }$ to a desired accuracy.

```tcl
Algorithm 3 CorrelationFinding
1: Input: Sample access from D, degree budget ℓ, correlation parameter α
2: Output: Subspace $V \subseteq \mathbb { R } ^ { d }$ and when $V \neq \{ 0 \}$ , a direction $\hat { \mathbf { v } } \in \mathbb { S } ^ { d - 1 }$ , along which y has a
non-trivial correlation
3: Draw $N = d ^ { O ( \ell ) } \mathrm { p o l y } ( 1 / \alpha )$ examples $\{ ( \mathbf { x } ^ { ( i ) } , y ^ { ( i ) } ) \} _ { i = 1 } ^ { N }$ from D
4: For each multi-index γ over [d] with $| \gamma | \leq \ell ,$ let $\begin{array} { r } { \hat { c } _ { \gamma } = N ^ { - 1 } \sum _ { i = 1 } ^ { N } ( y ^ { ( i ) } H _ { \gamma } ( \mathbf { x } ^ { ( i ) } ) ) } \end{array}$ , where $H _ { \gamma }$ is the
normalized Hermite polynomial with multi-index γ.
5: Construct matrix $\hat { G } \in \mathbb { R } ^ { \hat { d } \times d }$ such that for $\begin{array} { r } { i , j \in [ d ] , \hat { G } _ { i j } = \sum _ { \gamma : | \gamma | \leq \ell - 1 } \sqrt { ( \gamma _ { i } + 1 ) ( \gamma _ { j } + 1 ) } \hat { c } _ { \gamma + \mathbf { e } _ { i } } \hat { c } _ { \gamma + \mathbf { e } _ { j } } } \end{array}$
6: Return V as the subspace spanned by eigenvectors of $\hat { G }$ with large eigenvalue. And if $V \neq \{ 0 \}$
also return v the top eigenvector of $\hat { G } _ { \ell } .$
```

Theorem 6.1. Let D be a distribution over $\mathbb { R } ^ { d } \times \{ 0 , 1 \}$ such that $D _ { \mathbf { x } } = \mathcal { N } ( 0 , \mathbf { I } )$ There is an algorithm such that for $\alpha \in ( 0 , 1 )$ , suppose there exists a direction $\mathbf { v } \in \mathbb { S } ^ { d - 1 } , \ k \in \mathbb { N }$ and a degree r polynomial $p ( \mathbf { w } )$ , with $\| p ( \mathbf { w } ) \| \leq 1$ such that $\left| \mathbf { E } _ { ( \mathbf { x } , y ) \sim D } [ y h _ { k } ( \mathbf { v } \cdot \mathbf { x } ) p ( \mathbf { w } ) ] \right| > \alpha _ { : }$ then it draws $d ^ { ( r + k ) } \mathrm { p o l y } ( 1 / \alpha )$ examples from D, runs in poly $( d ^ { r + k } / \alpha )$ time and outputs a vector vˆ such that there exists some polynomial $q ( \hat { \mathbf { w } } ) ~ \| q ( \hat { \mathbf { w } } ) \| \leq 1$ , that satisfies $\left| \mathbf { E } _ { ( \mathbf { x } , y ) \sim D } [ y h _ { i } ( \hat { \mathbf { v } } \cdot \mathbf { x } ) q ( \hat { \mathbf { w } } ) ] \right| > \Omega ( \alpha / ( k + r ) )$ where $i + \deg ( q ) \leq r + k$

Furthermore, we give an even stronger algorithm that finds all such directions in one shot.

Theorem 6.2. Let D be a distribution over $\mathbb { R } ^ { d } \times \{ 0 , 1 \}$ such that $D _ { \mathbf { x } } = \mathcal { N } ( 0 , \mathbf { I } )$ and $\ell \in N _ { + }$ . There is an algorithm such that for $\beta \in ( 0 , 1 )$ , it draws $d ^ { O ( \ell ) } \mathrm { p o l y } ( 1 / \beta )$ examples from D, runs in $\scriptstyle \mathrm { ) o l y } ( d ^ { \ell } / \beta )$ time and outputs a subspace $V \subseteq \mathbb { R } ^ { d }$ with dimension at most ${ \cal O } ( \ell / \beta ^ { 2 } )$ such that for every $\mathbf { v } \perp V$ $\begin{array} { r } { \operatorname* { s u p } _ { p , k : \| p \| = 1 , k + \deg ( p ) \leq \ell , k \geq 1 } \mathbf { E } _ { ( \mathbf { x } , y ) \sim D } [ y h _ { k } ( \mathbf { v } \cdot \mathbf { x } ) p ( \mathbf { w } ) ] \leq \beta . } \end{array}$

To prove the above two theorems, we present the following technical lemmas.

Lemma 6.3. Let D be a distribution over $\mathbb { R } ^ { d } \times \{ 0 , 1 \}$ such that $D _ { \mathbf { x } } = \mathcal { N } ( 0 , \mathbf { I } )$ . For $\ell \in \mathbb { N } _ { + }$ , denote by $F _ { \ell } = \mathrm { p r o j } _ { < \ell } ( y )$ , the projection of y onto the subspace of polynomials of degree no larger than ℓ under Gaussian distribution and denote by $G _ { \ell } = \mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \nabla F ( \mathbf { x } ) \nabla F ( \mathbf { x } ) ^ { \top } ]$ . Then, for every $\mathbf { v } \in \mathbb { S } ^ { d - 1 }$ , we have

$$
{ \mathbf v } ^ { \top } G _ { \ell } { \mathbf v } = \sum _ { i = 1 } ^ { \ell } i B _ { i } ^ { 2 } ( { \mathbf v } ) ,
$$

where for $i \in [ \ell ] , B _ { i } ( \mathbf { v } ) = \operatorname* { s u p } _ { p ( \mathbf { w } ) \in \mathcal { P } _ { < \ell - i } , \| p \| \leq 1 } \mathbf { E } _ { ( \mathbf { x } , y ) \sim D } [ y h _ { i } ( \mathbf { v } \cdot \mathbf { x } ) p ( \mathbf { w } ) ]$ . Furthermore, there is an algorithm such that for every $\epsilon \in ( 0 , 1 )$ , it draws $d ^ { O ( \ell ) } \mathrm { p o l y } ( \ell / \epsilon )$ , runs in $d ^ { O ( \ell ) } \mathrm { p o l y } ( \ell / \epsilon )$ time and outputs a matrix $\hat { G }$ such that with probability at least 99%, $\left\| G _ { \ell } - \hat { G } \right\| _ { o p } \leq \epsilon$

Proof of Lemma 6.3. We expand $F _ { \ell }$ using normalized Hermite polynomials along direction v and its orthogonal subspace.

$$
F _ { \ell } ( { \mathbf { x } } ) = \sum _ { i , \alpha : | \alpha | + i \leq \ell } c _ { i , \alpha } h _ { i } ( { \mathbf { v } } \cdot { \mathbf { x } } ) H _ { \alpha } ( { \mathbf { w } } ) .
$$

Let $p ( \mathbf { w } )$ be any polynomial over $\mathbf { v } ^ { \perp }$ with degree at most $\ell - i .$ , with $i \leq \ell .$ . We expand $p ( \mathbf { w } )$ using normalized Hermite polynomial.

$$
p ( \mathbf { w } ) = \sum _ { \alpha : | \alpha | \leq \deg ( p ) } a _ { \alpha } H _ { \alpha } ( \mathbf { w } ) .
$$

In particular, we have $\begin{array} { r } { \| p \| ^ { 2 } = \sum _ { \alpha } a _ { \alpha } ^ { 2 } } \end{array}$ . Using orthogonality of Hermite polynomials, we have

$$
\underset { ( \mathbf { x } , y ) \sim D } { \mathbf { E } } [ y h _ { i } ( \mathbf { v } \cdot \mathbf { x } ) p ( \mathbf { w } ) ] = \underset { ( \mathbf { x } , y ) \sim D } { \mathbf { E } } [ F _ { \ell } h _ { i } ( \mathbf { v } \cdot \mathbf { x } ) p ( \mathbf { w } ) ] = \sum _ { \substack { \alpha : | \alpha | + i \leq \ell } } c _ { i , \alpha } a _ { \alpha } .
$$

Under the constraint $\textstyle \sum _ { \alpha } a _ { \alpha } ^ { 2 } \leq 1$ , Cauchy–Schwarz gives $\begin{array} { r } { B _ { i } ( \mathbf { v } ) ^ { 2 } = \sum _ { \alpha : | \alpha | + i \le \ell } c _ { i , \alpha } ^ { 2 } } \end{array}$ . When this sum is positive, the maximum is attained at

$$
p ( \mathbf { w } ) = \frac { \sum _ { \alpha : | \alpha | + i \le \ell } c _ { i , \alpha } H _ { \alpha } ( \mathbf { w } ) } { \left( \sum _ { \alpha : | \alpha | + i \le \ell } c _ { i , \alpha } ^ { 2 } \right) ^ { 1 / 2 } } .
$$

Otherwise, take $p = 0$ . On the other hand, for each $\mathbf { v } \in \mathbb { S } ^ { d - 1 }$ , we have

$$
\mathbf { v } ^ { \top } G _ { \ell } \mathbf { v } = \| \partial _ { \mathbf { v } } F _ { \ell } \| ^ { 2 } = \sum _ { i \geq 1 , \alpha : i + | \alpha | \leq \ell } i c _ { i , \alpha } ^ { 2 } = \sum _ { i = 1 } ^ { \ell } i B _ { i } ( \mathbf { v } ) ^ { 2 } .
$$

To finalize the proof of Lemma 6.3, we show that we can estimate $G _ { \ell }$ to a good accuracy using samples. Write $\begin{array} { r } { F _ { \ell } = \sum _ { \alpha : | \alpha | \le \ell } c _ { \alpha } H _ { \alpha } ( \mathbf { x } ) } \end{array}$ . We notice that for each α such that $\alpha _ { i } > 0$ , there exists a multi-index $\gamma$ such that $\alpha = \gamma + \mathbf { e } _ { i }$ , where $i \in [ d ]$ . This gives

$$
\partial _ { i } F _ { \ell } = \sum _ { \gamma : | \gamma | \leq \ell - 1 } \sqrt { \gamma _ { i } + 1 } c _ { \gamma + \mathbf { e } _ { i } } H _ { \gamma } ( \mathbf { x } )
$$

This implies that for each $i , j \in [ d ]$ , we have

$$
G _ { i j } = \underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ \partial _ { i } F _ { \ell } \partial _ { j } F _ { \ell } ] = \sum _ { \gamma } \sqrt { ( \gamma _ { i } + 1 ) ( \gamma _ { j } + 1 ) } c _ { \gamma + \mathbf { e } _ { i } } c _ { \gamma + \mathbf { e } _ { j } } .\tag{44}
$$

This implies that if we can estimate each $c _ { \alpha } = \mathbf { E } _ { ( \mathbf { x } , y ) \sim D } [ y H _ { \alpha } ( \mathbf { x } ) ]$ to error $\epsilon d ^ { - O ( \gamma ) }$ using some $\hat { c } _ { \alpha }$ then we have $\left\| { \hat { G } } - G \right\| \leq \epsilon$ . By Chebyshev inequality and union bound, we only need to use $d ^ { O ( r + k ) } \mathrm { p o l y } ( 1 / \stackrel { \cdot } { \alpha } )$ to run empirical estimation for all $c _ { \alpha } = \mathbf { E } _ { ( \mathbf { x } , y ) \sim D } [ y H _ { \alpha } ( \mathbf { x } ) ]$

With Lemma 6.3, the proofs of Theorem 6.2 and Theorem 6.1 are immediate.

Proof of Theorem 6.1. By Lemma 6.3 with $\ell = r + k$ , we know that using $d ^ { O ( r + k ) } \mathrm { p o l y } ( 1 / \alpha )$ examples, we can get a matrix $\hat { G }$ such that $\left\| \hat { G } - G \right\| \leq \alpha ^ { 2 } / 8$ . Since $\big | \mathbf { E } _ { ( \mathbf { x } , y ) \sim D } \big [ y h _ { k } ( \mathbf { v } \cdot \mathbf { x } ) p ( \mathbf { w } ) \big ] \big | > \alpha$ , we know that $G _ { \ell }$ has an eigenvalue at least $\alpha ^ { 2 }$ and thus $\hat { G }$ has an eigenvalue at least $7 \alpha ^ { 2 } / 8$ . Let u be the top eigenvector of $\hat { G }$ and we know that $\mathbf { u } ^ { \top } \hat { G } \mathbf { u } \geq 7 \alpha ^ { 2 } / 8$ . As a result $\mathbf { u } ^ { \top } G \mathbf { u } \geq 3 \alpha ^ { 2 } / 4$ . By Lemma 6.3, we know that there exists some $i \in [ \ell ]$ such that $B _ { i } ( { \mathbf { u } } ) \geq \Omega ( \alpha / \ell )$ □

Proof of Theorem 6.2. By Lemma 6.3 with ℓ, we know that using $d ^ { O ( \ell ) } \mathrm { p o l y } ( 1 / \beta )$ examples, we can get a matrix $\hat { G }$ such that $\Big \| \hat { G } - G \Big \| \leq \beta ^ { 2 } / d ^ { O ( \ell ) }$ $\mathrm { B y \ ( 4 4 ) }$ , since $\operatorname { t r } ( G ) \leq \ell ,$ we know that $\operatorname { t r } ( { \hat { G } } ) \leq 2 \ell$ We take V as the subspace spanned by all eigenvectors of $\hat { G }$ with an eigenvalue at least $\beta ^ { 2 } / 1 0 0$ Then dim $( V ) \le { \cal { O } } ( \ell / \beta ^ { 2 } )$ . Notice that for every $\mathbf { u } \perp V$ , we have $\left| { \bf u } ^ { \top } \hat { G } { \bf u } \right| \le \beta ^ { 2 } / 1 0 0$ , which implies $\mathbf { u } ^ { \top } G \mathbf { u } \leq \beta ^ { 2 }$ . By Lemma 6.3, we finish the proof. □

## 7 A Tree-Based Algorithm for Learning Intersections of PTFs

So far, we have already obtained all technical ingredients to obtain a learning algorithm that runs in $d ^ { O ( k ) } 2 ^ { \mathrm { p o l y } _ { k } ( 1 / \epsilon ) }$ time and outputs a hypothesis with error ϵ.

Theorem 7.1. Consider the problem of learning an intersection of k degree-2 PTFs under a Gaussian distribution. There is an algorithm such that given any $\epsilon \in ( 0 , 1 )$ , it draws $d ^ { O ( k ) } 2 ^ { \mathrm { p o l y } _ { k } ( 1 / \epsilon ) }$ samples from $D ,$ runs in $d ^ { O ( k ) } 2 ^ { \mathrm { p o l y } _ { k } ( 1 / \epsilon ) }$ time and outputs a hypothesis $\hat { h } : \mathbb { R } ^ { d }  \{ 0 , 1 \}$ such that with probability at least 99%, err $\mathbf { \Omega } _ { D } ( \hat { h } ) \leq \epsilon$

Recall that Theorem 5.1 says for any instance of intersections of k degree-two PTFs $f ,$ if there is a direction u such that Rand ${ \bf \nabla } _ { f } ( { \bf u } )$ is not very small, $f$ must have correlation with respect to some low degree Hermite polynomial that involves u direction. Furthermore, Theorem 6.1 shows that once such a direction exists, we can eficiently find one direction v, along which f actually has correlation. Once we have such a direction $\mathbf { v } ,$ we consider fixing $\mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } )$ along v direction using tiny grids. This can also be viewed as sampling $z \sim \mathcal { N } ( 0 , 1 )$ and considering the restricted function $f _ { z } ( \mathbf { w } ) : = f ( z \mathbf { v } + \mathbf { w } ) \ \mathrm { o r }$ 1 $\mathbf { v } ^ { \perp }$ . In Lemma 7.2, we show that for any fixed $j \in \mathbb N$ in expectation, $\left\| \mathrm { p r o j } _ { \leq j } ( f _ { z } ^ { * } ( \mathbf { w } ) ) \right\| _ { 2 } ^ { 2 }$ is at least $\left\| \operatorname { p r o j } _ { \leq j } ( f ^ { * } ( \mathbf { w } ) ) \right\| _ { 2 } ^ { 2 } .$ . In fact, we show in Lemma 7.3 that the property of v can in fact increase the low-degree Hermite energy by a non-trivial amount.

However, no matter how we fix these values, the low-degree Hermite energy cannot exceed 1. This implies that if we create a tree rooted at $f ^ { * }$ , the ground truth hypothesis, and grow it by creating children that represent the functions, which fix x along some directions, in expectation the tree cannot be too deep, which implies that there will be at most $2 ^ { \mathrm { p o l y } _ { k } ( 1 / \epsilon ) }$ many leaves. In other words, for all but $o ( \epsilon )$ fraction of the leaves, there is no direction in $\mathbb { R } ^ { d }$ that can have a large randomized error for the function represented at the leaf. Importantly, each function at a leaf is also an instance of intersections of k degree-2 PTFs under a Gaussian distribution.

Recall in Theorem 4.1, we proved that if for an instance of intersections of k degree-2 PTFs, there is no direction in $\mathbb { R } ^ { d }$ that has a large randomized error, then the instance is close to another instance that is regular. As a result, by Theorem 3.10, we can eficiently learn the function at the leaf with a high accuracy. This means by calling the direction finding algorithm, Algorithm 3 and Theorem 3.10 $2 ^ { \mathrm { p o l y } _ { k } ( 1 / \epsilon ) }$ times, we return a decision tree that is ϵ close to $f ^ { * }$

Lemma 7.2. Let $f ^ { * } ( \mathbf { x } ) : \mathbb { R } ^ { d }  [ 0 , 1 ]$ be a function. Let $\textbf { v } \in \mathbb { S } ^ { d - 1 }$ be a unit vector and write $f ^ { * } ( \mathbf { x } ) = f ^ { * } ( z \mathbf { v } + \mathbf { w } )$ , where $\mathbf { w } = \mathrm { p r o j } _ { \mathbf { v } ^ { \perp } } ( \mathbf { x } )$ . Write the $f ^ { * } ( \mathbf { x } )$ into Hermite decomposition,

$$
f ^ { * } ( { \bf x } ) = \sum _ { k \geq 0 } \sum _ { \alpha } \widehat { f ^ { * } } ( k , \alpha ) h _ { k } ( { \bf v } \cdot { \bf x } ) h _ { \alpha } ( { \bf w } ) .
$$

For each $z \in \mathbb { R }$ , we write $f _ { z } ^ { * } ( \mathbf { w } ) = f ^ { * } ( z \mathbf { v } + \mathbf { w } )$ as a function of w, when $\mathbf { v } \cdot \mathbf { x } = z , $ then for every $j \in \mathbb N$ , it holds

$$
\underset { z \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { E } } [ \| \mathrm { p r o j } _ { \le j } ( f _ { z } ^ { * } ( \mathbf { w } ) ) \| _ { 2 } ^ { 2 } ] - \| \mathrm { p r o j } _ { \le j } ( f ^ { * } ( \mathbf { x } ) ) \| _ { 2 } ^ { 2 } = \sum _ { \alpha : | \alpha | \le j } \sum _ { k : k + | \alpha | > j } \widehat { f ^ { * } } ( k , \alpha ) ^ { 2 } ,
$$

where for a function $f , \mathrm { p r o j } _ { \leq j } ( f )$ is the projection of f onto the subspace of polynomials with degree no larger than j and $\| f \| _ { 2 }$ is the $L _ { 2 }$ norm of f under $\mathcal { N } ( 0 , \bf { I } )$

Proof of Lemma 7.2. Notice that for each $z \in \mathbb { R }$ , we notice that $\begin{array} { r } { f _ { z } ^ { * } ( \mathbf { w } ) = \sum _ { k > 0 } \sum _ { \alpha } \widehat { f ^ { * } } ( k , \alpha ) h _ { k } ( z ) h _ { \alpha } ( \mathbf { w } ) } \end{array}$ This implies that $\begin{array} { r } { \mathrm { p r o j } _ { \le j } ( f _ { z } ^ { * } ( \mathbf { w } ) ) = \sum _ { \alpha : | \alpha | \le j } \left( \sum _ { k \ge 0 } \widehat { f ^ { * } } ( k , \alpha ) h _ { k } ( z ) \right) h _ { \alpha } ( \mathbf { w } ) } \end{array}$ and thus,

$$
\big \| \mathrm { p r o j } _ { \leq j } ( f _ { z } ^ { * } ( \mathbf { w } ) ) \big \| _ { 2 } ^ { 2 } = \sum _ { \alpha : | \alpha | \leq j } \left( \sum _ { k \geq 0 } \widehat { f ^ { * } } ( k , \alpha ) h _ { k } ( z ) \right) ^ { 2 } .
$$

By the orthogonality of the Hermite polynomials, after taking expectations over $z \sim \mathcal { N } ( 0 , 1 )$ , we obtain that

$$
\underset { z \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { E } } [ \| \mathrm { p r o j } _ { \leq j } ( f _ { z } ^ { * } ( \mathbf { w } ) ) \| _ { 2 } ^ { 2 } ] = \sum _ { \alpha : | \alpha | \leq j } \sum _ { k \geq 0 } \widehat { f ^ { * } } ( k , \alpha ) ^ { 2 } .
$$

Algorithm 4 FiniteGridPTFLearning   
1: Input: Sample access to D with $D _ { \mathbf { x } } = \mathcal { N } ( 0 , \mathbf { I } )$ and $y = f ^ { * } ( \mathbf { x } )$ for an intersection of k degree-two   
PTFs; accuracy ϵ and failure probability δ   
2: Output: A decision-tree hypothesis $\hat { f } : \mathbb { R } ^ { d }  \{ 0 , 1 \}$   
3: Set $\ell = 2 k + 1$ and $\rho = \epsilon / 1 0 ;$ choose $\epsilon _ { \mathrm { r e g } } , \eta , \epsilon _ { \mathrm { l e a r n } } , \tau$ so that Theorem 3.10 achieves leaf error at   
most $\epsilon / 2$ whenever a τ-regular intersection has error at most $\epsilon _ { \mathrm { r e g } } + \eta$   
4: Set $\alpha = \alpha _ { k } ( \epsilon _ { \mathrm { r e g } } , \tau )$ from Theorem 4.1, and $\beta = \beta _ { k } ( \alpha )$ from Theorem 5.1   
5: Set $\Delta _ { 0 } =$ min $\breve { \{ \beta ^ { 2 } / 1 6 , \eta \} } , B = \left\lceil 2 5 6 k \ell ^ { 4 } / ( \rho \beta ^ { 2 } \Delta _ { 0 } ) \right\rceil$ and $D _ { \mathrm { m a x } } = \lceil 1 2 8 \ell ^ { 4 } / ( \rho \beta ^ { 2 } ) \rceil$   
6: Let $\theta _ { b } = \Phi _ { \mathrm { G } } ^ { - 1 } ( b / B )$ for $b = 1 , \ldots , B - 1 .$ , with $\theta _ { 0 } = - \infty$ and $\theta _ { B } = + \infty$   
7: Set $L _ { \mathrm { m a x } } \ = \ B ^ { D _ { \mathrm { m a x } } } , \ m _ { \mathrm { m o m } } = ( d + 1 ) ^ { O ( \ell ) } \mathrm { p o l y } ( 1 / \beta , \log ( 8 L _ { \mathrm { m a x } } / \delta ) )$ for estimating each $G _ { \ell }$ to   
operator-norm error $\beta ^ { 2 } / 1 6$   
8: Let $m _ { \mathrm { r e g } }$ be the conditional sample budget of RegularLearn at failure probability $\delta / ( 8 L _ { \mathrm { m a x } } )$   
enlarged if necessary to be at least $C \epsilon ^ { - 2 } \log ( 8 L _ { \operatorname* { m a x } } / \delta )$   
9: Initialize a tree T whose root is the active cell $C _ { \mathcal { O } } = \mathbb { R } ^ { d }$   
10: for $t = 0 , 1 , \ldots , D _ { \mathrm { m a x } } - 1$ do   
11: Draw a fresh global batch $S _ { t }$ of $\left\lceil C B ^ { t } ( m _ { \mathrm { m o m } } + \log ( 8 L _ { \mathrm { m a x } } / \delta ) ) \right\rceil$ examples from $D ;$ route them   
to the active depth-t cells   
12: for each active cell $C$ at depth t do   
13: Let $U _ { C }$ be the span of the directions selected on the path to $C$ and let $W _ { C } = U _ { C } ^ { \perp }$   
14: if $W _ { C } = \{ 0 \}$ then   
15: Set $V _ { C } = \{ 0 \}$   
16: else   
17: From $\{ ( \mathrm { p r o j } _ { W _ { C } } \mathbf { x } , y ) : ( \mathbf { x } , y ) \in S _ { t } , \mathbf { x } \in C \}$ , run Algorithm 3 with parameters $( \ell , \beta )$   
obtaining $V _ { C }$ and, if $V _ { C } \neq \{ 0 \}$ , a direction $\mathbf { v } _ { C } \in W _ { C }$   
18: if $V _ { C } = \{ 0 \}$ then   
19: Mark C as a spectral terminal leaf   
20: else   
21: for $b = 1 , \dots , B$ do   
22: Create an active child cell $C _ { b } = C \cap \{ \mathbf { x } : \theta _ { b - 1 } \leq \mathbf { v } _ { C } \cdot \mathbf { x } < \theta _ { b } \}$   
23: Mark all active cells at depth $D _ { \mathrm { m a x } }$ as capped leaves   
24: Draw a fresh global batch $S _ { \mathrm { l e a f } }$ of $\left\lceil C B ^ { D _ { \mathrm { m a x } } } ( m _ { \mathrm { r e g } } + \log ( 8 L _ { \mathrm { m a x } } / \delta ) ) \right\rceil$ examples and route them to   
the spectral terminal leaves   
25: for each spectral terminal leaf C do   
26: if $W _ { C } = \{ 0 \}$ then   
27: Let $\hat { h } _ { C }$ be the empirical majority label among the conditional samples in $C$   
28: else   
29: Apply RegularLearn from Theorem 3.10 to the conditional samples $\{ ( \mathrm { p r o j } _ { W _ { C } } \mathbf { x } , y )$   
$( \mathbf { x } , y ) \in S _ { \mathrm { l e a f } } , \mathbf { x } \in C \}$ , with parameters $( \tau , \epsilon _ { \mathrm { l e a r n } } )$ , obtaining $\hat { h } _ { C }$   
30: Set $\hat { h } _ { C } \equiv 0$ on each capped leaf   
31: Return ${ \hat { f } } ( \mathbf { x } ) = { \hat { h } } _ { C ( \mathbf { x } ) } ( \mathrm { p r o j } _ { W _ { C ( \mathbf { x } ) } } \mathbf { x } )$ , where $C ( \mathbf { x } )$ is the leaf of $\tau$ containing x

$$
\begin{array} { l } { \mathrm { S i n c e ~ } \left\| \mathrm { p r o j } _ { \le j } ( f ^ { * } ( \mathbf { x } ) \right\| _ { 2 } ^ { 2 } = \sum _ { k , \alpha : k + | \alpha | \le j } \widehat { f ^ { * } } ( k , \alpha ) ^ { 2 } , \mathrm { ~ w e ~ g e t ~ } } \\ { \underset { z \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { E } } [ \left\| \mathrm { p r o j } _ { \le j } ( f _ { z } ^ { * } ( \mathbf { w } ) ) \right\| _ { 2 } ^ { 2 } ] - \left\| \mathrm { p r o j } _ { \le j } ( f ^ { * } ( \mathbf { x } ) ) \right\| _ { 2 } ^ { 2 } = \displaystyle \sum _ { \alpha : | \alpha | \le j } \displaystyle \sum _ { k : k > j - | \alpha | } \widehat { f ^ { * } } ( k , \alpha ) ^ { 2 } . } \end{array}
$$

Next, we will show that if $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ f ^ { * } ( \mathbf { x } ) h _ { k } ( \mathbf { v } \cdot \mathbf { x } ) p ( \mathbf { w } ) ]$ is lower bounded, then we can also lower bound the increment in Lemma 7.2.

Lemma 7.3. Let $f ^ { * } ( \mathbf { x } ) : \mathbb { R } ^ { d }  [ 0 , 1 ]$ be a function. Let $\textbf { v } \in \mathbb { S } ^ { d - 1 }$ be a unit vector and write $f ^ { * } ( \mathbf { x } ) = f ^ { * } ( z \mathbf { v } + \mathbf { w } )$ , where $\mathbf { w } = \mathrm { p r o j } _ { \mathbf { v } ^ { \perp } } ( \mathbf { x } )$ . For each $z \in \mathbb { R }$ , we write $f _ { z } ^ { * } ( \mathbf { w } ) = f ^ { * } ( z \mathbf { v } + \mathbf { w } )$ as a function of w, when $\mathbf { v } \cdot \mathbf { x } = z$ . Suppose there exist $k \geq 1$ and a degree-r polynomial $p ( \mathbf { w } )$ with $\| p ( \mathbf { w } ) \| _ { 2 } \leq 1$ such that $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ f ^ { * } ( \mathbf { x } ) h _ { k } ( \mathbf { v } \cdot \mathbf { x } ) p ( \mathbf { w } ) ] \geq \beta$ , for some $\beta > 0$ . Then, there exists some $j ^ { * } \leq r$ such that

$$
\begin{array} { r } { \underset { z \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { E } } [ \left\| \operatorname { p r o j } _ { \leq j ^ { * } } ( f _ { z } ^ { * } ( \mathbf { w } ) ) \right\| _ { 2 } ^ { 2 } ] - \left\| \operatorname { p r o j } _ { \leq j ^ { * } } ( f ^ { * } ( \mathbf { x } ) ) \right\| _ { 2 } ^ { 2 } \geq \beta ^ { 2 } / ( r + 1 ) , } \end{array}
$$

where for a function $f , \mathrm { p r o j } _ { \leq j ^ { * } } ( f )$ is the projection of f onto the subspace of polynomials with degree no larger than $j ^ { * }$ and $\| f \| _ { 2 }$ is the $L _ { 2 }$ norm of f under $\mathcal { N } ( 0 , \bf { I } )$

Proof of Lemma 7.3. We write $p ( \mathbf { w } )$ into Hermite decomposition, $\begin{array} { r } { p ( \mathbf { w } ) = \sum _ { \alpha : | \alpha | \leq r } \widehat { p } ( \alpha ) h _ { \alpha } ( \mathbf { w } ) } \end{array}$ . Since $\| p ( \mathbf { w } ) \| _ { 2 } \leq 1$ , we have $\textstyle \sum _ { \alpha } { \widehat { p } } ( \alpha ) ^ { 2 } \leq 1$ . Similarly, we write $\begin{array} { r } { f ^ { * } ( \mathbf { x } ) = \sum _ { k > 0 } \sum _ { \alpha } \widehat { f ^ { * } } ( k , \alpha ) h _ { k } ( \mathbf { v } \cdot \mathbf { x } ) h _ { \alpha } ( \mathbf { w } ) } \end{array}$ Then, we have

$$
\underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ f ^ { * } ( \mathbf { x } ) h _ { k } ( \mathbf { v } \cdot \mathbf { x } ) p ( \mathbf { w } ) ] = \sum _ { \alpha : | \alpha | \leq r } \widehat { f ^ { * } } ( k , \alpha ) \widehat { p } ( \alpha ) \geq \beta .
$$

By Cauchy–Schwarz inequality, we have $\textstyle \sum _ { \alpha : | \alpha | \leq r } { \widehat { f ^ { * } } } ( k , \alpha ) ^ { 2 } \geq \beta ^ { 2 }$ , which implies that there exists some $j ^ { * } \leq r$ such that $\textstyle \sum _ { \alpha : | \alpha | = j ^ { * } } { \widehat { f ^ { * } } } ( k , \alpha ) ^ { 2 } \geq \beta ^ { 2 } / ( r + 1 )$ . Apply Lemma 7.2 with $j = j ^ { * }$ , we have

$$
\begin{array} { r l } & { \quad \underset { z \sim \sqrt { ( 0 , 1 ) } } { \mathbf { E } } [ \| \mathrm { p r o j } _ { \leq j ^ { * } } ( f _ { z } ^ { * } ( \mathbf { w } ) ) \| _ { 2 } ^ { 2 } ] - \left\| \mathrm { p r o j } _ { \leq j ^ { * } } ( f ^ { * } ( \mathbf { x } ) ) \right\| _ { 2 } ^ { 2 } = \underset { \alpha : | \alpha | \leq j ^ { * } } { \sum } \underset { k : k > j ^ { * } - | \alpha | } { \sum } \widehat { f ^ { * } } ( k , \alpha ) ^ { 2 } } \\ & { \qquad \quad \geq \underset { \alpha : | \alpha | = j ^ { * } } { \sum } \widehat { f ^ { * } } ( k , \alpha ) ^ { 2 } \geq \beta ^ { 2 } / ( r + 1 ) . } \end{array}
$$

We now analyze the finite-grid implementation of the preceding freezing argument. Throughout the rest of this section, set $\ell = 2 k + 1$ and write

$$
\Phi _ { \ell } ( f ) = { \frac { 1 } { \ell } } \sum _ { j = 0 } ^ { \ell - 1 } \left\| \mathrm { p r o j } _ { \leq j } f \right\| _ { 2 } ^ { 2 } .
$$

We use the influence matrix $G _ { \ell }$ and its empirical estimate $\hat { G } _ { \ell }$ from Section 6.

Fix the B intervals $I _ { 1 } , \ldots , I _ { B }$ of Algorithm 4, each of Gaussian probability $1 / B$ , and let $z _ { b } =$ $\Phi _ { \mathrm { G } } ^ { - 1 } ( ( b - 1 / 2 ) / B ) \in I _ { b }$ be a representative of $I _ { b }$ . Consider a cell C at depth $t ,$ defined by mutually orthogonal unit vectors $\mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { t }$ and constraints $\mathbf { v } _ { i } \cdot \mathbf { x } \in I _ { b _ { i } }$ . Set

$$
U _ { C } = \mathrm { s p a n } \{ { \bf v } _ { 1 } , \ldots , { \bf v } _ { t } \} , \quad W _ { C } = U _ { C } ^ { \perp } , \quad a _ { C } = \sum _ { i = 1 } ^ { t } z _ { b _ { i } } { \bf v } _ { i } .
$$

Let $\nu _ { C }$ be the distribution of $\mathrm { p r o j } _ { U _ { C } } \mathbf { x }$ conditional on $\mathbf { x } \in C$ , where $\mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ . Conditional on $\mathbf { x } \in C ,$ the random variables $\mathbf { u } \sim \nu _ { C }$ and $\mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { W _ { C } } )$ are independent. Define

$$
g _ { C } ( \mathbf { w } ) = \underset { \mathbf { u } \sim \nu _ { C } } { \mathbf { E } } [ f ^ { * } ( \mathbf { u } + \mathbf { w } ) ] ,
$$

$$
h _ { C } ( \mathbf { w } ) = f ^ { * } ( a _ { C } + \mathbf { w } ) ,
$$

$$
\Delta _ { C } = \underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { W _ { C } } ) } { \mathbf { P r } } [ f ^ { * } ( \mathbf { u } + \mathbf { w } ) \neq h _ { C } ( \mathbf { w } ) ] .\tag{45}
$$

In particular, $h _ { C }$ is an intersection of at most k degree-two $\mathrm { P T F s }$ , whereas $g _ { C }$ is the conditional mean of the observed labels. At the root, $g _ { C _ { \otimes } } = h _ { C _ { \otimes } } = f ^ { * }$ and $\Delta _ { C _ { \mathcal { O } } } = 0$

Lemma 7.4 (Finite-grid approximation). Fix a cell C and a unit vector $\mathbf { v } \in W _ { C }$ . Put $E = W _ { C } \cap \mathbf { v } ^ { \perp }$ and $C _ { b } = C \cap \{ \mathbf x : \mathbf v \cdot \mathbf x \in I _ { b } \}$ . For $z \in \mathbb { R }$ and $\mathbf { w } ^ { \prime } \in E$ , write $g _ { C , z } ( \mathbf { w } ^ { \prime } ) = g _ { C } ( z \mathbf { v } + \mathbf { w } ^ { \prime } )$ . Then, for each $j \geq 0$ 2

$$
\frac { 1 } { B } \sum _ { b = 1 } ^ { B } \left\| \mathrm { p r o j } _ { \le j } g _ { C _ { b } } \right\| _ { 2 } ^ { 2 } \ge \underset { Z \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { E } } \left[ \left\| \mathrm { p r o j } _ { \le j } g _ { C , Z } \right\| _ { 2 } ^ { 2 } \right] - \frac { k } { 2 B } .\tag{46}
$$

Furthermore,

$$
\frac { 1 } { B } \sum _ { b = 1 } ^ { B } \Delta _ { C _ { b } } \leq \Delta _ { C } + \frac { 2 k } { B } ,\tag{47}
$$

and

$$
\begin{array} { r } { \underline { { \mathbf { E } } } _ { N \sim \mathcal { N } ( 0 , \mathbf { I } _ { W _ { C } } ) } [ | g _ { C } ( \mathbf { w } ) - h _ { C } ( \mathbf { w } ) | ] = \Delta _ { C } , \quad \quad \| g _ { C } - h _ { C } \| _ { 2 } ^ { 2 } \leq \Delta _ { C } . } \end{array}\tag{48}
$$

Proof. For fixed $\mathbf { u } \in U _ { C }$ and $\mathbf { w } ^ { \prime } \in E .$ define $F _ { \mathbf { u } , \mathbf { w } ^ { \prime } } ( z ) = f ^ { * } ( \mathbf { u } + z \mathbf { v } + \mathbf { w } ^ { \prime } )$ . Each defining quadratic has at most two zeros as a function of $z ,$ unless it is identically zero. Thus $F _ { \mathbf { u } , \mathbf { w } ^ { \prime } }$ is constant on all but at most $2 k$ of the intervals $I _ { 1 } , \ldots , I _ { B }$ , up to Gaussian-null sets. Since the variance of a Boolean random variable is at most $1 / 4$ , we have

$$
\frac { 1 } { B } \sum _ { b = 1 } ^ { B } \operatorname { V a r } _ { Z \sim \mathcal { N } ( 0 , 1 ) \mid Z \in I _ { b } } [ F _ { \mathbf { u } , \mathbf { w } ^ { \prime } } ( Z ) ] \leq \frac { k } { 2 B } .\tag{49}
$$

By the definition of $g _ { C }$ and Jensen’s inequality,

$$
\begin{array} { r } { \mathrm { V a r } _ { Z \sim \mathcal { N } ( 0 , 1 ) | Z \in I _ { b } } [ g _ { C } ( Z \mathbf { v } + \mathbf { w } ^ { \prime } ) ] \leq \underline { { \mathbf { E } } } _ { \mathbf { u } \sim \nu _ { C } } \left[ \mathrm { V a r } _ { Z \sim \mathcal { N } ( 0 , 1 ) | Z \in I _ { b } } [ F _ { \mathbf { u } , \mathbf { w } ^ { \prime } } ( Z ) ] \right] . } \end{array}
$$

Averaging over $\mathbf { w } ^ { \prime } \sim \mathcal { N } ( 0 , \mathbf { I } _ { E } )$ and $b \in [ B ]$ gives

$$
q _ { C } ( \mathbf { v } ) : = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \underbrace { \mathbf { F } } _ { \mathbf { w } ^ { \prime } \sim \mathcal { N } ( 0 , \mathbf { I } _ { E } ) } \left[ \mathrm { V a r } _ { Z \sim \mathcal { N } ( 0 , 1 ) | Z \in I _ { b } } [ g _ { C } ( Z \mathbf { v } + \mathbf { w } ^ { \prime } ) ] \right] \leq \frac { k } { 2 B } .\tag{50}
$$

Let $R _ { j }$ denote the orthogonal projection onto Gaussian Hermite polynomials of degree at most $j$ in $E .$ . Since

$$
g _ { C _ { b } } ( \mathbf w ^ { \prime } ) = \mathbf E _ { Z \sim \mathcal { N } ( 0 , 1 ) | Z \in I _ { b } } [ g _ { C , Z } ( \mathbf w ^ { \prime } ) ] ,
$$

$R _ { j } g _ { C _ { b } }$ is the conditional mean of $R _ { j } g _ { C , Z }$ over the same interval. The variance decomposition in the Hilbert space $L ^ { 2 } ( E )$ therefore gives

$$
\begin{array} { r l } & { \underset { Z \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { E } } \left[ \| R _ { j } g _ { C , Z } \| _ { 2 } ^ { 2 } \right] - \frac { 1 } { B } \underset { b = 1 } { \overset { B } { \sum } } \| R _ { j } g _ { C _ { b } } \| _ { 2 } ^ { 2 } } \\ & { = \frac { 1 } { B } \underset { b = 1 } { \overset { B } { \sum } } \underset { Z \sim \mathcal { N } ( 0 , 1 ) \vert Z \in I _ { b } } { \mathbf { E } } \left[ \| R _ { j } ( g _ { C , Z } - g _ { C _ { b } } ) \| _ { 2 } ^ { 2 } \right] } \\ & { \leq \frac { 1 } { B } \underset { b = 1 } { \overset { B } { \sum } } \underset { \pmb { w } ^ { \prime } \sim \mathcal { N } ( 0 , 1 ) \vert Z \in I _ { b } } { \mathbf { E } } \left[ ( g _ { C , Z } ( \mathbf { w } ^ { \prime } ) - g _ { C _ { b } } ( \mathbf { w } ^ { \prime } ) ) ^ { 2 } \right] = q _ { C } ( \mathbf { v } ) \leq \frac { k } { 2 B } . } \end{array}
$$

This proves (46).

For the approximation error, let $b ( Z )$ be the index of the interval containing Z. Since $\mathbf { u } \sim \nu _ { C }$ $Z \sim { \mathcal { N } } ( 0 , 1 )$ , and $\mathbf { w } ^ { \prime } \sim \mathcal { N } ( 0 , \mathbf { I } _ { E } )$ are mutually independent,

$$
\frac { 1 } { B } \sum _ { b = 1 } ^ { B } \Delta _ { C _ { b } } = \underset { { \bf w } \sim { \cal U } ( 0 , { \bf I } _ { E } ) } { \bf P r } [ f ^ { * } ( { \bf u } + Z { \bf v } + { \bf w } ^ { \prime } ) \neq f ^ { * } ( a _ { C } + z _ { b ( Z ) } { \bf v } + { \bf w } ^ { \prime } ) ]
$$

The first term equals $\Delta _ { C }$ . For every fixed $\mathbf { w } ^ { \prime } ,$ the defining polynomials that are not identically zero on the line $\mathbf { x } = a _ { C } + z \mathbf { v } + \mathbf { w } ^ { \prime }$ have at most 2k roots in total. Thus $z \mapsto f ^ { * } ( a _ { C } + z \mathbf { v } + \mathbf { w } ^ { \prime } )$ agrees with its interval representative except possibly on at most 2k intervals containing such roots, whose total Gaussian mass is at most $2 k / B$ . This proves (47). Finally, since $h _ { C } ( \mathbf { w } )$ is Boolean,

$$
\underset { \mathbf { u } \sim \nu _ { C } } { \mathbf { P r } } [ f ^ { * } ( \mathbf { u } + \mathbf { w } ) \neq h _ { C } ( \mathbf { w } ) ] = | g _ { C } ( \mathbf { w } ) - h _ { C } ( \mathbf { w } ) | .
$$

Averaging over $\mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { W _ { C } } )$ gives the first identity in (48), and the second follows from $| g _ { C } - h _ { C } | ^ { 2 } \leq$ $| g _ { C } - h _ { C } |$ □

Proof of Theorem 7.1. Let $\epsilon _ { \mathrm { r e g } } , \tau$ and α be the parameters of the randomized-error form of Theorem 4.1, and write $\beta = \beta _ { k } ( \alpha )$ for the correlation bound of Theorem 5.1. Let $\eta > 0$ be chosen so that the regular-case learner of Theorem 3.10 learns conditional labels to error at most $\epsilon / 2$ whenever the labels disagree with some τ-regular intersection with probability at most $\eta + \epsilon _ { \mathrm { r e g } }$ . This is the label-robustness guarantee required for the use of that learner. Put

$$
\rho = \epsilon / 1 0 , \qquad \Delta _ { 0 } = \operatorname* { m i n } \{ \beta ^ { 2 } / 1 6 , \eta \} , \qquad B = \left\lceil \frac { 2 5 6 k \ell ^ { 4 } } { \rho \beta ^ { 2 } \Delta _ { 0 } } \right\rceil , \qquad D _ { \operatorname* { m a x } } = \left\lceil \frac { 1 2 8 \ell ^ { 4 } } { \rho \beta ^ { 2 } } \right\rceil .
$$

Algorithm 4 makes the following spectral decision at an active cell C. If $W _ { C } = \{ 0 \}$ or the subspace $V _ { C }$ returned by Algorithm 3 is {0}, the algorithm stops at $C$ . Otherwise, it splits $C$ along the returned direction $\mathbf { v } _ { C }$ using the intervals $I _ { 1 } , \ldots , I _ { B }$

Potential progress and tree depth. At every internal node C, we have $V _ { C } \neq \{ 0 \}$ . By the spectral threshold in Algorithm 3 and the estimation guarantee $\| \widehat { G } _ { \ell } - G _ { \ell } \| _ { \mathrm { o p } } \leq \beta ^ { 2 } / 1 6$ , the returned direction

$\mathbf { v } _ { C }$ satisfies $\mathbf { v } _ { C } ^ { \top } G _ { \ell } \mathbf { v } _ { C } \geq 3 \beta ^ { 2 } / 1 6$ . By Lemma 6.3, there exist an integer $s \in [ \ell ]$ and a polynomial $p$ on $W _ { C } \cap \mathbf { v } _ { C } ^ { \perp }$ such that

$$
s + \deg ( p ) \leq \ell , \quad \| p \| _ { 2 } = 1 , \quad \ \underbrace { \mathbf { E } } _ { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { W _ { C } } ) } [ g _ { C } ( \mathbf { w } ) h _ { s } ( \mathbf { v } _ { C } \cdot \mathbf { w } ) p ( \mathrm { p r o j } _ { \mathbf { v } _ { C } ^ { \perp } } \mathbf { w } ) ] \geq \frac { \beta } { 4 \ell } .\tag{51}
$$

Apply Lemma 7.2 and Lemma 7.3 to $g _ { C }$ and the certificate (51). It supplies $j ^ { * } \leq \deg ( p ) \leq \ell - 1$ such that

$$
\underset { Z \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { E } } [ \big \| \mathrm { p r o j } _ { \le j ^ { * } } g _ { C , Z } \big \| _ { 2 } ^ { 2 } ] - \big \| \mathrm { p r o j } _ { \le j ^ { * } } g _ { C } \big \| _ { 2 } ^ { 2 } \ge \frac { \beta ^ { 2 } } { 1 6 \ell ^ { 3 } } .\tag{52}
$$

For every other $j \in \{ 0 , \ldots , \ell - 1 \}$ , Lemma 7.2 gives a nonnegative exact-freezing increment. Applying (46) of Lemma 7.4 for every such $j$ and averaging over $j$ yields

$$
\frac { 1 } { B } \sum _ { b = 1 } ^ { B } \Phi _ { \ell } ( g _ { C _ { b } } ) - \Phi _ { \ell } ( g _ { C } ) \geq \frac { \beta ^ { 2 } } { 1 6 \ell ^ { 4 } } - \frac { k } { 2 B } \geq \frac { \beta ^ { 2 } } { 3 2 \ell ^ { 4 } } = : \mu .\tag{53}
$$

The last inequality holds by the chosen value of $B$

For every node, put $\pi _ { C } = \mathbf { P r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ \mathbf { x } \in C ]$ . Each child has mass $\pi _ { C _ { b } } = \pi _ { C } / B$ , because the splitting direction is orthogonal to all earlier splitting directions on its path. Let I and $\mathcal { L }$ be the internal nodes and leaves, respectively. For any real-valued function A on the nodes, finite-tree telescoping gives

$$
\sum _ { C \in { \cal C } } \pi _ { C } A ( C ) - A ( C _ { \emptyset } ) = \sum _ { C \in { \cal Z } } \pi _ { C } \left( \frac { 1 } { B } \sum _ { b = 1 } ^ { B } A ( C _ { b } ) - A ( C ) \right) .\tag{54}
$$

Since $0 \leq \Phi _ { \ell } ( g _ { C } ) \leq 1$ , applying (54) to $A ( C ) = \Phi _ { \ell } ( g _ { C } )$ and using (53) gives

$$
\sum _ { C \in \mathbb { Z } } \pi _ { C } \leq \frac { 1 } { \mu } = \frac { 3 2 \ell ^ { 4 } } { \beta ^ { 2 } } .\tag{55}
$$

A fresh Gaussian point reaching depth $D _ { \mathrm { m a x } }$ passes through at least $D _ { \mathrm { m a x } }$ internal nodes. Thus, by Markov’s inequality,

$$
\underset { \substack { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } } { \mathbf { P r } } [ \mathbf { x } \mathrm { ~ r e a c h e s ~ a ~ c a p p e d ~ l e a f } ] \leq \frac { 3 2 \ell ^ { 4 } } { D _ { \mathrm { m a x } } \beta ^ { 2 } } \leq \rho / 4 .\tag{56}
$$

Accumulated gridding error. Using (54) with $A ( C ) = \Delta _ { C }$ , noting $\Delta _ { C _ { \emptyset } } = 0$ , and applying (47) and (55), we obtain

$$
\sum _ { C \in \mathcal { L } } \pi _ { C } \Delta _ { C } \leq \frac { 2 k } { B } \sum _ { C \in \mathcal { T } } \pi _ { C } \leq \frac { 6 4 k \ell ^ { 4 } } { B \beta ^ { 2 } } .
$$

In particular,

$$
\underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } { \mathbf { P r } } [ \Delta _ { C ( \mathbf { x } ) } > \Delta _ { 0 } ] \leq \frac { 6 4 k \ell ^ { 4 } } { B \beta ^ { 2 } \Delta _ { 0 } } \leq \rho / 4 ,\tag{57}
$$

where $C ( \mathbf { x } )$ is the leaf containing x. By (56) and $( 5 7 )$ , a fresh Gaussian point reaches a noncapped leaf with $\Delta _ { C } \leq \Delta _ { 0 }$ except with probability at most $\rho / 2$

Regularity at a good terminal cell. Fix such a leaf C with $W _ { C } \neq \{ 0 \}$ . By (48),

$$
\| g _ { C } - h _ { C } \| _ { 2 } \leq \sqrt { \Delta _ { C } } \leq \beta / 4 , \qquad \Delta _ { C } \leq \eta .
$$

Suppose some unit vector $\mathbf { u } \in W _ { C }$ satisfies $\mathrm { R a n d } _ { h _ { C } } ( { \bf u } ) \geq \alpha$ . By Theorem 5.1, there are $s \in [ \ell ]$ and a polynomial p with $\| p \| _ { 2 } = 1$ and $s + \deg ( p ) \leq \ell$ such that

$$
\underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { W _ { C } } ) } { \mathbf { E } } [ h _ { C } ( \mathbf { w } ) h _ { s } ( \mathbf { u } \cdot \mathbf { w } ) p ( \mathrm { p r o j } _ { \mathbf { u } ^ { \perp } } \mathbf { w } ) ] \geq \beta .
$$

Cauchy–Schwarz implies that the same correlation with $g _ { C }$ in place of $h _ { C }$ is at least $3 \beta / 4$ . On the other hand, since C is a spectral terminal leaf with $W _ { C } \neq \{ 0 \}$ , Algorithm 3 returned $V _ { C } = \{ 0 \}$ . By the spectral stopping condition and the accuracy of ${ \widehat { G } } _ { \ell } ,$ , for every unit vector $\mathbf { u } \in W _ { C } , \mathbf { u } ^ { \top } G _ { \ell } \mathbf { u } \le 5 \beta ^ { 2 } / 1 6$ By Lemma 6.3, for every $s \geq 1$ and polynomial $p$ with $\| p \| _ { 2 } = 1$ and $ s + \deg ( p ) \leq \ell ;$

$$
\left| \underset { \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { W _ { C } } ) } { \mathbf { E } } [ g _ { C } ( \mathbf { w } ) h _ { s } ( \mathbf { u } \cdot \mathbf { w } ) p ( \mathrm { p r o j } _ { \mathbf { u } ^ { \perp } } \mathbf { w } ) ] \right| \leq \frac { \sqrt { 5 } \beta } { 4 } < \frac { 3 \beta } { 4 } .
$$

This gives a contradiction. Hence,

$$
\mathrm { R a n d } _ { h _ { C } } ( { \mathbf { u } } ) < \alpha \qquad \mathrm { f o r ~ e v e r y ~ u n i t ~ } { \mathbf { u } } \in W _ { C } .\tag{58}
$$

If $W _ { C } = \{ 0 \}$ , this assertion is vacuous and $h _ { C }$ is constant.

By the randomized-error form of Theorem 4.1, h is $\epsilon _ { \mathrm { r e g } } { \mathrm { - c l o s e } }$ to a τ -regular intersection $\widetilde { h } _ { C }$ under $\mathcal { N } ( 0 , \mathbf { I } _ { W _ { C } } )$ . Moreover,

$$
\underset { \substack { \mathbf { u } \sim \nu _ { C } \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } _ { W _ { C } } ) } } { \mathbf { P r } } [ f ^ { * } ( \mathbf { u } + \mathbf { w } ) \neq \widetilde { h } _ { C } ( \mathbf { w } ) ] \leq \Delta _ { C } + \epsilon _ { \mathrm { r e g } } \leq \eta + \epsilon _ { \mathrm { r e g } } .
$$

Thus the stipulated robust regular-case learner returns a hypothesis of conditional error at most $\epsilon / 2$ at every good terminal cell. The exceptional cells have combined Gaussian probability at most $\rho / 2 .$ so the resulting decision tree has error at most $\epsilon / 2 + \rho / 2 < \epsilon$

Sample complexity. At a fixed cell $C ,$ conditioned on its earlier splitting directions, $\mathrm { p r o j } _ { W _ { C } } \mathbf { x } \sim$ $\mathcal { N } ( 0 , \mathbf { I } _ { W _ { C } } )$ under $D \mid \{ \mathbf { x } \in C \}$ . The moment-estimation guarantee of Lemma 6.3 therefore applies to these conditional samples. With fresh independent sample batches for each depth and a union bound over at most $B ^ { D _ { \mathrm { m a x } } }$ nodes, all calls of Algorithm 3 succeed with probability at least $1 - \delta / 2$ Every depth-t cell has Gaussian probability $B ^ { - t }$ , so a global batch of size $B ^ { t }$ times the number of samples required per node (up to logarithmic factors) supplies the needed conditional samples. A final fresh batch gives the samples for the regular-case learner at every terminal cell. Allocating failure probability $\delta / ( 2 B ^ { D _ { \mathrm { m a x } } } )$ per learner call and taking a union bound gives success probability at least $1 - \delta$ . The total running time is bounded by

$$
B ^ { D _ { \mathrm { m a x } } } \left( ( d + 1 ) ^ { O ( \ell ) } \mathrm { p o l y } \left( \beta ^ { - 1 } , \log \frac { B ^ { D _ { \mathrm { m a x } } } } { \delta } \right) + T _ { \mathrm { r e g } } \right) ,
$$

where $T _ { \mathrm { r e g } }$ is the cost of the regular-case learner per leaf. With $\ell = 2 k + 1$ and the parameter bounds required by Theorems 3.10 and 4.1, this is $d ^ { O ( k ) } 2 ^ { \mathrm { p o l y } _ { k } ( 1 / \epsilon ) }$ □

## 8 Polynomial Approximation with Hinge Loss

In this section, we give an algorithm for learning intersections of degree-2 PTFs supported on a known lifted subspace by minimizing hinge loss over bounded-norm polynomials. The main technical lemma establishes the existence of a low-degree polynomial with small expected hinge loss against the target intersection and controlled Gaussian $L _ { 2 }$ norm, while preserving the given lifted subspace.

This polynomial provides a good comparator for convex optimization, and its norm bound controls the additional loss caused by label disagreement. The resulting learning guarantee is stated in the following theorem. For fixed k, the running time is polynomial in the ambient dimension and quasipolynomial in the subspace dimension and inverse accuracy.

Theorem 8.1 (Learning in a known lifted subspace). Let $d , k \in \mathbb { Z } _ { + } , \ \epsilon \in ( 0 , 1 / 2 )$ and $L =$ $k ^ { 2 } \log ^ { 3 } ( k ) \log ^ { 2 } ( 1 / \epsilon )$ . Let $\Phi _ { 2 } : \mathbb { R } ^ { d }  \mathbb { R } ^ { N }$ , where $N = d ( d + 3 ) / 2$ , consist of the normalized Gaussian Hermite polynomials of degrees one and two, and let $S ~ \subseteq \mathbb { R } ^ { N }$ be a subspace of dimension n. Suppose D is a joint distribution of $\left( \mathbf { x } , y \right)$ over $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ with marginal $D _ { \mathbf { x } } = \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ , and there exists an intersection $f ( \mathbf { x } ) = { \textstyle \bigwedge } _ { i = 1 } ^ { k } \operatorname { s i g n } \bigl ( c _ { i } + \mathbf { u } _ { i } \cdot \Phi _ { 2 } ( \mathbf { x } ) \bigr )$ , where $c _ { i } \in \mathbb { R }$ and $\mathbf { u } _ { i } \in S$ such that $\operatorname { P r } _ { ( \mathbf { x } , y ) \sim D } [ f ( \mathbf { x } ) \neq y ] \leq 2 ^ { - C L }$

There is an algorithm based on hinge-loss minimization over bounded-norm polynomials with the following guarantee. Given d, k, ϵ, a basis for S, and i.i.d. sample access to D, the algorithm has sample complexity and running time at most $d ^ { O ( 1 ) } n ^ { O ( k ^ { 2 } \log ^ { 2 } ( 2 k ) \log \left( 1 / \epsilon \right) ) } 2 ^ { O \left( k ^ { 2 } \log ^ { 3 } ( 2 k ) \log ^ { 2 } ( 1 / \epsilon ) \right) }$ and, with probability at least $2 / 3$ , outputs a hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$ satisfying $\mathbf { P r } _ { ( \mathbf { x } , y ) \sim D } [ h ( \mathbf { x } ) \neq y ] \leq \epsilon$

The high-level intuition is as follows. For convenience, first consider the case $S = \mathbb { R } ^ { N }$ We minimize the expected hinge loss $\mathrm { H i n g e } ( y , y ^ { \prime } ) = \mathrm { m a x } \{ 0 , 1 - y y ^ { \prime } \}$ over polynomials of degree at most m and Gaussian $L _ { 2 }$ norm at most W, where m and W are chosen according to the technical lemma below. This objective is convex in the polynomial coeficients. The lemma guarantees the existence of a polynomial with small hinge loss against the target intersection, and its norm bound controls the additional loss caused by the label disagreement allowed in Theorem 8.1. Stochastic convex optimization then finds a polynomial p with small expected hinge loss. Since hinge loss upper-bounds the classification error of $h ( \mathbf { x } ) = \mathrm { s i g n } ( p ( \mathbf { x } ) )$ , thresholding at zero gives the desired hypothesis.

Using the normalized Hermite basis, this optimization has runtime and sample complexity polynomial in $M , W , 1 / \epsilon$ , where $M = \left( \begin{array} { c } { { d { + } m } } \\ { { m } } \end{array} \right)$ is the dimension of the space of polynomials of degree at most m. When a subspace $S \subseteq \mathbb { R } ^ { N }$ is given, the technical lemma further guarantees that the approximating polynomial can be written as

$$
p ( { \bf x } ) = P \big ( \mathrm { p r o j } _ { S } \Phi _ { 2 } ( { \bf x } ) \big ) , \qquad \mathrm { d e g } ( P ) \leq m / 2 .
$$

Writing $n = \dim ( S )$ , we can therefore restrict the search to a space spanned by at most $\binom { n + m / 2 } { m / 2 }$ monomial features. Together with bounds on the coeficients and feature moments, this yields a runtime that is polynomial in d and, for fixed k, quasipolynomial in n and $1 / \epsilon .$ , as stated in Theorem 8.1.

We now state the main technical lemma, which establishes the existence of the required polynomial, bounds its degree and Gaussian $L _ { 2 }$ norm, and shows that it preserves the given lifted subspace.

Lemma 8.2. Let $f : \mathbb { R } ^ { d }  \{ \pm 1 \}$ be an intersection of k degree-2 PTFs of the form $f ( \mathbf { x } ) =$ ${ \textstyle \bigwedge _ { i = 1 } ^ { k } \mathrm { s i g n } ( p _ { i } ( { \bf x } ) ) }$ . Then, for any $\epsilon \in ( 0 , 1 / 2 )$ , there exist an even integer m and a polynomial g of degree at most m with $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } \left[ \mathrm { H i n g e } ( f ( \mathbf { x } ) , g ( \mathbf { x } ) ) \right] \leq \epsilon$ . Moreover, g satisfies the following properties:

## 1. Degree and norm bounds. We have

$$
m = O \bigl ( k ^ { 2 } \log ^ { 2 } ( 2 k ) \log ( 1 / \epsilon ) \bigr ) , \qquad \| g \| _ { L _ { 2 } ( N ( 0 , { \bf I } _ { d } ) ) } \leq 2 ^ { O ( k ^ { 2 } \log ^ { 3 } ( 2 k ) \log ^ { 2 } ( 1 / \epsilon ) ) } .
$$

Furthermore, for every $\delta \in [ 0 , 1 ]$ and every measurable weight $a : \mathbb { R } ^ { d }  [ 0 , 1 ]$ with ${ \bf E } _ { { \bf x } \sim \mathcal { N } ( 0 , { \bf I } _ { d } ) } [ a ( { \bf x } ) ] \le$ δ,

$$
\begin{array} { r } { \underline { { \mathbf { E } } } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } \left[ | a ( \mathbf { x } ) g ( \mathbf { x } ) | \right] \leq \sqrt { \delta } ~ \mathrm { q u a s i p o l y } _ { k } ( 1 / \epsilon ) . } \end{array}
$$

2. Preservation of the lifted subspace. If all $p _ { i }$ are of the form $p _ { i } ( \mathbf { x } ) = c _ { i } + \mathbf { u } _ { i } \cdot \Phi _ { 2 } ( \mathbf { x } ) ~ f o r$ ${ \mathbf { u } } _ { i } \in S ,$ where $S \subseteq \mathbb { R } ^ { N }$ is a subspace. Then g can be chosen in the form $g ( { \bf x } ) = p ( \mathrm { p r o j } _ { S } \Phi _ { 2 } ( { \bf x } ) )$ $p$ is a polynomial of degree at most $m / 2$

Our construction builds on the rational-function approach to constructing a sign-matching polynomial for intersection of halfspaces with margins in [KOS04, KS04]. We adapt this approach to intersections of degree-2 PTFs under Gaussian marginals, without assuming a uniform positive margin. We retain the strategy of approximating the sign function by rational functions and rescaling by the denominators to obtain a polynomial. We use additional analysis to bound the expected hinge loss near the threshold boundaries and in the Gaussian tails, and Gaussian $\left( L _ { 2 } \right)$ norm of the rescaled polynomial.

The motivation is that directly approximating the value of sign function and then build the intersection incurs an $\Omega ( 1 / \eta )$ degree dependence on the boundary width η. Such construction does not leverage the fact that we only need sign matching. To avoid this, we approximate the individual sign values by rational functions (which has $O ( \log ( 1 / \eta ) )$ dependence), combine them to obtain a rational function whose sign agrees with the intersection away from the threshold boundaries, and then clear the denominators by rescaling. If this denominator is always positive, this scaling yields a polynomial with the same sign. We subsequently rescale it to obtain margin on a high-probability region and control its hinge loss elsewhere.

We first give the following fact on rational approximation of sign functions.

Fact 8.3 (Rational approximation of the sign function). For any boundary width $\eta \in ( 0 , 1 / 2 )$ and approximation error $\epsilon \in ( 0 , 1 / 2 )$ , let $m = \lceil c \log ( 1 / \eta ) \rceil$ and $s = \lceil c \log ( 1 / \epsilon ) \rceil$ , where c is a suficiently large universal constant and define $\begin{array} { r } { P _ { m } ( t ) = \prod _ { i = 0 } ^ { m } ( t + 2 ^ { - j } ) } \end{array}$ ，

$$
N ( t ) = P _ { m } ( t ) ^ { 2 s } - P _ { m } ( - t ) ^ { 2 s } , \qquad D ( t ) = P _ { m } ( t ) ^ { 2 s } + P _ { m } ( - t ) ^ { 2 s } .
$$

Each has degree at most $O ( \log ( 1 / \eta ) \log ( 1 / \epsilon ) )$ and the rational function $r ( t ) = N ( t ) / D ( t )$ satisfies:

1. Positive denominator: For all $t \in \mathbb { R } , D ( t ) > 0$

2. Boundedness and sign preservation: For all $t \in \mathbb { R } , | r ( t ) | \leq 1$ and $\mathrm { s i g n } ( r ( t ) ) = \mathrm { s i g n } ( t )$

3. Approximation away from 0: For $| t | \in [ \eta , 1 ] , | \mathrm { s i g n } ( t ) - r ( t ) | \leq \epsilon$

Proof of Fact 8.3. Both N and D have degree at most $2 s ( m + 1 )$ , which is $O ( \log ( 1 / \eta ) \log ( 1 / \epsilon ) )$

For the first property, notice that roots of $P _ { m } ( t )$ are all negative, while roots of $P _ { m } ( - t )$ are all positive. Therefore, $P _ { m } ( t )$ and $P _ { m } ( - t )$ have no common roots and $D ( t )$ must be always positive.

For the second property, $P _ { m } ( t ) ^ { 2 s }$ and $P _ { m } ( - t ) ^ { 2 s }$ are nonnegative, so $| N ( t ) | \leq D ( t )$ and hence $| r ( t ) | \leq 1$ for all t. For $t > 0$ , every factor in $P _ { m }$ satisfies $| t - 2 ^ { - j } | < t + 2 ^ { - j }$ . Thus $| P _ { m } ( - t ) | < P _ { m } ( t )$ and $N ( t ) , r ( t ) \neq 0$ for any $t \neq 0$ . Furthermore, since N is odd and D is even, we must have sign $( r ( t ) ) = \mathrm { s i g n } ( t )$

It only remains to prove the approximation property. For $t > 0$ , the goal of the construction is to have $| P _ { m } ( - t ) ^ { 2 s } |$ much smaller relative to $P _ { m } ( t ) ^ { 2 s }$ , so that $r ( t )$ is close to 1. Notice that this also sufices for the purpose of the negative side. At a fixed scale $a > 0$ , we have

$$
\frac { | a - t | } { a + t } \leq \frac { 1 } { 3 } \qquad \mathrm { w h e n ~ } a \leq t \leq 2 a ,
$$

and this ratio is at most 1 for every $t > 0$ . Thus one factor gives a constant separation over $[ a , 2 a ]$ . By multiplying over dyadic scales, we cover the entire desired interval without worsening the separation.

Using this to construct $P _ { m }$ , we get for any $t \in [ 2 ^ { - m } , 1 ]$

$$
{ \frac { | P _ { m } ( - t ) | } { P _ { m } ( t ) } } = \prod _ { j = 0 } ^ { m } { \frac { | 2 ^ { - j } - t | } { 2 ^ { - j } + t } } \leq { \frac { 1 } { 3 } } .
$$

We then take it to the power of $2 s$ to amplify this separation, this gives $| P _ { m } ( - t ) ^ { 2 s } | / | P _ { m } ( t ) ^ { 2 s } | \leq 1 / 3 ^ { 2 s }$ Therefore,

$$
| 1 - r ( t ) | = \left| \frac { 2 P _ { m } ( - t ) ^ { 2 s } } { P _ { m } ( - t ) ^ { 2 s } + P _ { m } ( t ) ^ { 2 s } } \right| \leq \frac 2 { 3 ^ { 2 s } + 1 } .
$$

By choosing c be a suficiently large universal constant in $m = \lceil c \log ( 1 / \eta ) \rceil$ and $s = \lceil c \log ( 1 / \epsilon ) \rceil$ gives $| 1 - r ( t ) | \leq \epsilon$ for $t \in [ \eta , 1 ]$ . Then the negative side follows the exact same argument. □

Given the rational function from the preceding fact, one can easily scale to construct a rational function $r ( t ) = N ( t ) / D ( t )$ with approximation error $\epsilon \leq 1 / ( 2 k )$ over the interval $[ - B , - \eta B ] \cup [ \eta B , B ]$ Then it is immediate that for the intersection of degree-2 PTFs $f ( \mathbf { x } ) = { \textstyle \bigwedge } _ { i = 1 } ^ { k } \operatorname { s i g n } ( p _ { i } ( \mathbf { x } ) )$ . We can construct a sign matching polynomial for $f$ by first having the following rational

$$
g ( \mathbf { x } ) = \sum _ { i = 1 } ^ { k } r ( p _ { i } ( \mathbf { x } ) ) - ( k - 1 ) = \sum _ { i = 1 } ^ { k } { \frac { N ( p _ { i } ( \mathbf { x } ) ) \Pi _ { j \in [ k ] \setminus i } D ( p _ { j } ( \mathbf { x } ) ) } { \Pi _ { j \in [ k ] } D ( p _ { j } ( \mathbf { x } ) ) } } - ( k - 1 ) .
$$

Since the preceding fact has shown that $D ( t )$ is always positive, we can simply have the sign matching polynomial as $p ( \mathbf { x } ) = g ( \mathbf { x } ) \Pi _ { j \in [ k ] } D ( p _ { j } ( \mathbf { x } ) )$ . It remains to check that 1. its hinge loss is bounded near the decision boundaries; 2. it is suficiently large for any point suficiently far from the decision boundary, so the overall Hinge can be upper bounded; and 3. the Hinge loss incurred outside the decision box is small due to Gaussian decaying tail. We first give the following fact on the sign matching polynomial.

Fact 8.4. Let $f : \mathbb { R } ^ { k }  \{ \pm 1 \}$ be an intersection $o f k$ sign functions of the form $f ( \mathbf { x } ) = \textstyle \bigwedge _ { i = 1 } ^ { k } \mathrm { s i g n } ( x _ { i } )$ For any $\eta \in ( 0 , 1 / 2 )$ , there exists a polynomial $q : \mathbb { R } ^ { k } $ R of degree at most $O ( k ^ { 2 } \log ( k ) \operatorname { l o g } ( 1 / \eta ) ) ~ o f$ the form $\begin{array} { r } { q ( \mathbf { x } ) = \sum _ { \alpha } c _ { \alpha } \mathbf { x } ^ { \alpha } } \end{array}$ where $\mathbf { x } ^ { \alpha }$ are monomials and $\begin{array} { r } { \sum _ { \alpha } \left| c _ { \alpha } \right| = 2 ^ { O ( k ^ { 2 } \log ( k ) \log ( 1 / \bar { \eta ) } ^ { 2 } ) } } \end{array}$ such that

1. Unit margin sign matching away from 0: $I f \eta \leq | x _ { i } | \leq 1$ for every $i ,$ , then $f ( \mathbf { x } ) q ( \mathbf { x } ) \geq 1$

2. Bounded pointwise hinge loss: $I f \left| x _ { i } \right| \leq 1$ for every $i ,$ then max $\{ 0 , 1 - f ( \mathbf { x } ) q ( \mathbf { x } ) \} \leq 2$

Proof of Fact $8 . 4 \cdot$ Set

$$
L = \lceil \log _ { 2 } ( 1 / \eta ) \rceil , \qquad \gamma = 2 ^ { - L } , \qquad \ell = 4 k L , \qquad \delta = 2 ^ { - \ell } ,
$$

and let $\begin{array} { r } { \rho = \frac { 1 } { 4 k } , \qquad s = \lceil \log _ { 9 } ( 8 k ) \rceil } \end{array}$ . We use the explicit rational construction $r = N / D$ from Fact 8.3 with product-depth parameter ℓ and power parameter s. Thus the boundary width and approximation error used there are $\delta$ and $\rho ,$ respectively. The estimate proved in Fact 8.3 gives

$$
| r ( t ) - \mathrm { s i g n } ( t ) | \leq { \frac { 2 } { 9 ^ { s } + 1 } } \leq \rho \qquad { \mathrm { f o r ~ } } \delta \leq | t | \leq 1 .
$$

Define the rational separator

$$
R ( { \bf x } ) = \sum _ { i = 1 } ^ { k } r ( x _ { i } ) - ( k - 1 ) .
$$

For Boolean inputs, the corresponding sum is 1 when all inputs are positive and at most −1 otherwise. The approximation changes this sum by at most $k \rho = 1 / 4$ , so

$$
f ( \mathbf { x } ) R ( \mathbf { x } ) \geq { \frac { 1 } { 2 } } \qquad { \mathrm { w h e n e v e r ~ } } \delta \leq | x _ { i } | \leq 1 { \mathrm { ~ f o r ~ e v e r y ~ } } i .
$$

Also, $| r | \leq 1$ gives $| R | \leq 2 k - 1$ everywhere.

Clear the positive denominator and normalize:

$$
S ( { \bf x } ) = R ( { \bf x } ) \prod _ { i = 1 } ^ { k } D ( x _ { i } ) , \qquad \lambda = \frac { 1 } { 2 } P _ { \ell } ( \gamma ) ^ { 2 s k } , \qquad q ( { \bf x } ) = \frac { S ( { \bf x } ) } { \lambda } .
$$

Then $q$ is a polynomial of degree at most $2 k s ( \ell + 1 ) = O \bigl ( k ^ { 2 } \log ( 2 k ) \log ( 1 / \eta ) \bigr )$

We now reduce the two required properties to denominator estimates. Since D has only even powers with nonnegative coeficients, it is even and nondecreasing on $[ 0 , \infty )$ . We will show

$$
D ( \gamma ) ^ { k } \geq 2 \lambda , \qquad D ( \delta ) D ( 1 ) ^ { k - 1 } \leq \frac { \lambda } { 2 k - 1 } .\tag{59}
$$

The first bound ensures a suficiently large denominator product when every coordinate is away from zero. The second ensures a suficiently small product when one coordinate is within width $\delta ,$ even if all other coordinates are as large as 1.

For the first bound, directly from the construction, $D ( \gamma ) \geq P _ { \ell } ( \gamma ) ^ { 2 s }$ , and hence $D ( \gamma ) ^ { k } \geq 2 \lambda$ . For the second bound, let $\begin{array} { r } { C _ { 0 } = \prod _ { i = 0 } ^ { \infty } ( 1 + 2 ^ { - j } ) < e ^ { 2 } < 8 } \end{array}$ . For $| t | \leq 1$ , we have $| P _ { \ell } ( \pm t ) | \le C _ { 0 }$ . For $| t | \leq \delta ,$ the smaller scale gives

$$
| P _ { \ell } ( \pm t ) | \le P _ { \ell } ( 0 ) \prod _ { j = 0 } ^ { \ell } ( 1 + 2 ^ { j - \ell } ) \le C _ { 0 } P _ { \ell } ( 0 ) .
$$

Therefore,

$$
D ( 1 ) \leq 2 C _ { 0 } ^ { 2 s } , \qquad D ( \delta ) \leq 2 C _ { 0 } ^ { 2 s } P _ { \ell } ( 0 ) ^ { 2 s } .
$$

Moreover,

$$
P _ { \ell } ( 0 ) = 2 ^ { - \ell ( \ell + 1 ) / 2 } , \qquad P _ { \ell } ( \gamma ) \geq \left( \prod _ { j = 0 } ^ { L - 1 } 2 ^ { - j } \right) \gamma ^ { \ell - L + 1 } = 2 ^ { - L \ell + L ( L - 1 ) / 2 } .
$$

Using $\ell = 4 k L$ , these estimates imply

$$
\frac { P _ { \ell } ( 0 ) ^ { 2 s } } { P _ { \ell } ( \gamma ) ^ { 2 s k } } \leq 2 ^ { - s ( ( 8 k ^ { 2 } + k ) L ^ { 2 } + 3 k L ) } .
$$

Consequently,

$$
\begin{array} { r l r } & { } & { \frac { ( 2 k - 1 ) D ( \delta ) D ( 1 ) ^ { k - 1 } } { \lambda } \leq 2 ( 2 k - 1 ) ( 2 C _ { 0 } ^ { 2 s } ) ^ { k } \frac { P _ { \ell } ( 0 ) ^ { 2 s } } { P _ { \ell } ( \gamma ) ^ { 2 s k } } } \\ & { } & { \qquad \leq 2 ^ { 9 s k - s ( ( 8 k ^ { 2 } + k ) L ^ { 2 } + 3 k L ) } \leq 1 . } \end{array}
$$

Here we used $2 ( 2 k - 1 ) ( 2 C _ { 0 } ^ { 2 s } ) ^ { k } \leq 2 ^ { 9 s k }$ , and the last inequality holds for $k , L , s \geq 1$ . This proves (59). We can now deduce the two properties. For Property 1, suppose $\eta \leq | x _ { i } | \leq 1$ for every i. Since $\delta < \gamma \leq \eta$ , monotonicity and the first bound in (59) give

$$
f ( \mathbf { x } ) q ( \mathbf { x } ) = \frac { f ( \mathbf { x } ) R ( \mathbf { x } ) } { \lambda } \prod _ { i } D ( x _ { i } ) \geq \frac { D ( \gamma ) ^ { k } } { 2 \lambda } \geq 1 .
$$

For Property $^ { 2 , }$ take any $\mathbf { x } \in [ - 1 , 1 ] ^ { k }$ . If $f ( \mathbf { x } ) q ( \mathbf { x } ) \geq 0$ , its hinge loss is at most 1. Otherwise, positivity of the denominator product implies $f ( { \bf x } ) R ( { \bf x } ) < 0$ , so some $| x _ { i } | < \delta$ . The second bound in (59) then gives

$$
| q ( \mathbf { x } ) | \leq { \frac { ( 2 k - 1 ) D ( \delta ) D ( 1 ) ^ { k - 1 } } { \lambda } } \leq 1 .
$$

Thus its hinge loss is at most $| 1 + | q ( \mathbf { x } ) | \leq 2 .$

Finally, we bound the monomial coeficients of $q .$ Both N and D have nonnegative coeficients: their definitions retain the odd and even coeficients of $P _ { \ell } ^ { 2 s }$ , respectively, each multiplied by 2. Their coeficient sums are therefore $N ( 1 )$ and $D ( 1 )$ . Expanding

$$
q ( \mathbf { x } ) = \frac { 1 } { \lambda } \left( \sum _ { i } N ( x _ { i } ) \prod _ { j \neq i } D ( x _ { j } ) - ( k - 1 ) \prod _ { i } D ( x _ { i } ) \right)
$$

and using $N ( 1 ) \leq D ( 1 )$ , we obtain

$$
\sum _ { \alpha } | c _ { \alpha } | \leq \frac { k N ( 1 ) D ( 1 ) ^ { k - 1 } + ( k - 1 ) D ( 1 ) ^ { k } } { \lambda } \leq \frac { ( 2 k - 1 ) D ( 1 ) ^ { k } } { \lambda } .
$$

The preceding lower bound on $P _ { \ell } ( \gamma )$ gives $\lambda ^ { - 1 } \leq 2 ^ { 1 + 8 s k ^ { 2 } L ^ { 2 } }$ , while $D ( 1 ) ^ { k } \leq ( 2 C _ { 0 } ^ { 2 s } ) ^ { k } = 2 ^ { O ( s k ) }$ . Hence

$$
\sum _ { \alpha } \left. c _ { \alpha } \right. \leq 2 ^ { O ( s k ^ { 2 } L ^ { 2 } ) } = 2 ^ { O ( k ^ { 2 } \log ( 2 k ) \log ^ { 2 } ( 1 / \eta ) ) } ,
$$

as required.

We now use Fact 8.4 to prove Lemma 8.2. Applying the fact to the rescaled defining polynomials $\{ p _ { i } / B \} _ { i \in [ k ] }$ for the intersection of PTFs, we obtain a polynomial $q$ whose hinge loss is zero whenever $\eta B \leq | p _ { i } ( \mathbf { \dot { x } } ) | \leq B$ for every i, by Property 1. It remains to control the contribution from points (i.e. $\{ p _ { i } / B \} _ { i \in [ k ] } )$ near the threshold boundaries and points in the tails.

Within the region where all $| p _ { i } ( \mathbf { x } ) | \leq B$ , Property 2 bounds the hinge loss by 2. Thus, the near-boundary contribution is controlled by its probability, which we bound using Gaussian anticoncentration. For the remaining tail region, we combine upper bound on the size of the constructed polynomial with Gaussian tail decay. With appropriate parameter choice of both B and η, the contribution from both regions will be suficiently small.

Proof of Lemma 8.2. The proof follows the following structure: We first choose the parameter η and B for rescaling the polynomial from Fact 8.4. Then as we discussed, Property 1 of Fact 8.4 already ensures the Hinge loss is 0 if $\eta B \leq | p _ { i } ( \mathbf { x } ) | \leq B$ for all i. We then show that for the near decision boundary case, i.e., there exists i such $p _ { i } ( \mathbf { x } ) \in [ - \eta B , \eta B ]$ . Since we have Property 2 of Fact 8.4, it sufices for us to bound $\mathbf { P r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ \exists i , p _ { i } ( \mathbf { x } ) \in [ - \eta B , \eta B ] ]$ from the anticoncentration. The other case that $p _ { i } ( \mathbf { x } ) \not \in [ - B , B ]$ can be analyzed from the Gaussian tail bound and the polynomial $L _ { 2 }$ norm. After that, we can simply put these together to bound the Hinge loss.

Step 1: Choose η, B and apply Fact 8.4 to construct the polynomial. We WLOG normalize the defining polynomials to satisfy $\| p _ { i } \| _ { L _ { 2 } ( \mathcal { N } ( 0 , \mathbf { I } _ { d } ) ) } = 1$ . Let c be a suficiently large universal constant and choose

$$
\eta = \left( \frac { \epsilon } { k } \right) ^ { C } , \qquad B = C k ^ { 2 } \log ( k ) ^ { 3 } \log ^ { 2 } ( 1 / \eta ) .
$$

Let q be the polynomial from Fact 8.4 and define

$$
g ( \mathbf { x } ) = q \left( \frac { p _ { 1 } ( \mathbf { x } ) } { B } , \ldots , \frac { p _ { k } ( \mathbf { x } ) } { B } \right) .
$$

The choice of $\eta$ makes the boundary region small, while B is chosen large enough that Gaussian tail decay dominates the possible growth of $g .$

Step 2: Control the contribution near the decision boundaries. Let $B = \mathbf { x } : \exists i , p _ { i } ( \mathbf { x } ) \in [ - \eta B , \eta B ]$ From Carbery–Wright anticoncentration and union bound, we must have that

$$
\underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } { \mathbf { P r } } [ \mathbf { x } \in \mathcal { B } ] \leq \sum _ { i } \underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } { \mathbf { P r } } [ p _ { i } ( \mathbf { x } ) \in [ - \eta B , \eta B ] ] = O ( k \sqrt { B \eta } ) .
$$

Using the second property of Fact 8.4 directly implies that

$$
\underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } { \mathbf { E } } [ \mathrm { H i n g e } ( f ( \mathbf { x } ) , g ( \mathbf { x } ) ) \mathbf { 1 } _ { B } ( \mathbf { x } ) ] = O ( k \sqrt { B \eta } ) .
$$

Step 3: Control the contribution from the tails. Let $\mathcal { T } = \{ \mathbf { x } : \mathrm { m a x } _ { i } | p _ { i } ( \mathbf { x } ) | > B \}$ denote the tail region. We first bound the $L _ { 2 }$ norms of $g$ and ${ \bf 1 } _ { T }$ , then combine them to control the hinge contribution from this region.

Since each $p _ { i }$ has degree at most 2 and $\| p _ { i } \| _ { 2 } = 1$ , hypercontractivity gives

$$
\left. \frac { p _ { i } } { B } \right. _ { u } \leq \frac { u - 1 } { B } , \qquad u \geq 2 .
$$

Taking a power requires a higher moment: for every integer $a \geq 1$ ，

$$
\left\| \left( { \frac { p _ { i } } { B } } \right) ^ { a } \right\| _ { 2 } = \left\| { \frac { p _ { i } } { B } } \right\| _ { 2 a } ^ { a } \leq \left( { \frac { 2 a - 1 } { B } } \right) ^ { a } .
$$

More generally, for a monomial of total degree $r = | \alpha | \geq 1$ , Hölder’s inequality gives

$$
\left\| \prod _ { i } ( p _ { i } / B ) ^ { \alpha _ { i } } \right\| _ { 2 } \leq \prod _ { i } \left\| { \frac { p _ { i } } { B } } \right\| _ { 2 r } ^ { \alpha _ { i } } \leq \left( { \frac { 2 r - 1 } { B } } \right) ^ { r } \leq 1 ,
$$

where the last inequality uses $r \leq \deg { q }$ and $B \geq 2 \deg { q } .$ . The constant monomial also has norm 1. Write $\begin{array} { r } { q ( \mathbf { z } ) = \sum _ { \alpha } c _ { \alpha } \mathbf { z } ^ { \alpha } } \end{array}$ . By the triangle inequality, the preceding bound on each substituted monomial, and the coeficient bound in Fact 8.4, we have

$$
\| g \| _ { 2 } \leq \sum _ { \alpha } | c _ { \alpha } | \left\| \prod _ { i } ( p _ { i } / B ) ^ { \alpha _ { i } } \right\| _ { 2 } \leq \sum _ { \alpha } | c _ { \alpha } | \leq 2 ^ { O ( k ^ { 2 } \log ( 2 k ) \log ^ { 2 } ( 2 k / \eta ) ) } .
$$

Next, the squared $L _ { 2 }$ norm of the indicator is exactly the tail probability. Hypercontractivity and Markov’s inequality give $\mathbf { P r } [ | p _ { i } ( \mathbf { x } ) | > B ] \leq e ^ { - c B }$ for a universal constant $c > 0$ . Thus a union bound yields

$$
\| \mathbf { 1 } _ { \mathcal { T } } \| _ { 2 } ^ { 2 } = \mathbf { P r } [ \mathbf { x } \in \mathcal { T } ] \leq \sum _ { i = 1 } ^ { k } \mathbf { P r } [ | p _ { i } ( \mathbf { x } ) | > B ] \leq k e ^ { - c B } .
$$

We can now bound the $L _ { 1 }$ norm of $g$ restricted to the tail region. By Cauchy–Schwarz,

$$
\| g \mathbf { 1 } _ { \mathcal { T } } \| _ { 1 } \leq \| g \| _ { 2 } \| \mathbf { 1 } _ { \mathcal { T } } \| _ { 2 } \leq \sqrt { k } \ \| g \| _ { 2 } e ^ { - c B / 2 } .
$$

Since Hinge $\left( f , g \right) \leq 1 + \left| g \right|$ , the hinge contribution from this region satisfies

$$
\mathbf { E } \left[ \mathrm { H i n g e } ( f ( \mathbf { x } ) , g ( \mathbf { x } ) ) \mathbf { 1 } _ { \mathcal { T } } ( \mathbf { x } ) \right] \leq \mathbf { P r } [ \mathbf { x } \in \mathcal { T } ] + \| g \mathbf { 1 } _ { \mathcal { T } } \| _ { 1 } \leq k e ^ { - c B } + \sqrt { k } \ \| g \| _ { 2 } e ^ { - c B / 2 } .
$$

Step 4: Combine the bounds. Taking C suficiently large makes

$$
O \Big ( k \sqrt { B \eta } \Big ) \le \epsilon / 2 , \qquad k e ^ { - c B } + \sqrt { k } \ \| g \| _ { 2 } e ^ { - c B / 2 } \le \epsilon / 2 .
$$

For the first inequality, the power $( \epsilon / k ) ^ { C / 2 }$ dominates the polynomial and logarithmic factors in $k \sqrt { B }$ For the second, B is a suficiently large multiple of the exponent in the norm bound. Therefore,

$$
\mathbf { E } [ \mathrm { H i n g e } ( f ( \mathbf { x } ) , g ( \mathbf { x } ) ) ] \leq \epsilon .
$$

Finally, set $m = 2 \deg { q }$ . Then m is even, and quadratic substitution gives deg $g \le m$ . Since

$$
\log ( 2 k / \eta ) = O ( \log ( 2 k / \epsilon ) ) = O ( \log ( 2 k ) \log ( 1 / \epsilon ) ) ,
$$

the degree and norm bounds become

$$
m = O \bigl ( k ^ { 2 } \log ^ { 2 } ( 2 k ) \log ( 1 / \epsilon ) \bigr ) , \qquad \| g \| _ { 2 } \leq 2 ^ { O ( k ^ { 2 } \log ^ { 3 } ( 2 k ) \log ^ { 2 } ( 1 / \epsilon ) ) } .
$$

The weighted-error estimate follows from Cauchy–Schwarz and $a ^ { 2 } \leq a$ . For the preservation claim, recall that $g = q ( p _ { 1 } / B , \dots , p _ { k } / B )$ with deg $q = m / 2$ . Under the lifted-subspace assumption, each $p _ { i }$ is afine in $\mathrm { p r o j } _ { S } \Phi _ { 2 } ( { \bf x } )$ , so g has the required form. □

Given Lemma 8.2, we can now prove Theorem 8.1 by applying standard stochastic convex optimization over the corresponding polynomial ball to find a polynomial $g$ with small expected hinge loss. Then obtain the hypothesis $h ( \mathbf x ) = \mathrm { s i g n } ( g ( \mathbf x ) )$ with small 0–1 error, since any point that h makes mistake for 0–1 error would also incur constant error for Hinge loss. We will use the following fact about stochastic convex optimization and the whole algorithm procedure is summarized in Algorithm 5.

Fact 8.5 (Projected stochastic subgradient descent; [Duc18], Corollary 3.4.9). Let $\mathcal { C } \subseteq \mathbb { R } ^ { N }$ be a nonempty compact convex set of diameter at most $R > 0$ , with eficiently computable Euclidean projections, and let $F : { \mathcal { C } }  \mathbb { R }$ be continuous and convex. Suppose a stochastic oracle, using fresh independent randomness on each call, satisfies

$$
\begin{array} { r } { \underline { { \mathbf { E } } } [ G ( \mathbf { w } , \boldsymbol { \xi } ) ] \in \partial F ( \mathbf { w } ) , \qquad \underline { { \mathbf { E } } } \Big [ \| G ( \mathbf { w } , \boldsymbol { \xi } ) \| _ { 2 } ^ { 2 } \Big ] \leq M ^ { 2 } } \end{array}
$$

for every ${ \textbf { w } } \in { \mathcal { C } } ,$ where $M > 0$ . Then projected stochastic subgradient descent with step sizes $\alpha _ { t } = R / ( M \sqrt { t } )$ and iterate averaging returns ${ \widehat { \mathbf { w } } } \in { \mathcal { C } }$ after $T \geq 1$ oracle calls such that

$$
\mathbf { E } [ F ( { \widehat { \mathbf { w } } } ) ] \leq \operatorname* { m i n } _ { \mathbf { w } \in { \mathcal { C } } } F ( \mathbf { w } ) + { \frac { 3 R M } { 2 { \sqrt { T } } } } ~ .
$$

Algorithm 5 HingeLearning   
1: Input: $d , k , \epsilon , \mathrm { ~ a ~ }$ basis for $S \subseteq \mathbb { R } ^ { N }$ , and i.i.d. sample access to $D$   
2: Output: A hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$   
3: Let m be the even degree bound from Lemma 8.2 with accuracy $\epsilon / 1 2 .$ , and set $t = m / 2$ . Choose   
the coeficient radius $R$ as in the proof of Theorem 8.1.   
4: Compute an orthonormal basis $\mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { n }$ of $S ,$ where n = dim S, and define   
$\Psi _ { t } ( \mathbf { x } ) = \left( \prod _ { j = 1 } ^ { n } \langle \mathbf { v } _ { j } , \Phi _ { 2 } ( \mathbf { x } ) \rangle ^ { \alpha _ { j } } \right) _ { \alpha \in \mathbb { Z } _ { \geq 0 } ^ { n } , | \alpha | \leq t } .$   
5: Using fresh samples from D, apply standard projected stochastic subgradient descent to   
$\underset { \| \mathbf { w } \| _ { 2 } \leq R ( \mathbf { x } , y ) \sim D } { \operatorname* { m i n } } [ \operatorname* { m a x } \{ 0 , 1 - y \langle \mathbf { w } , \Psi _ { t } ( \mathbf { x } ) \rangle \} ]$   
to expected additive optimization error at most $\epsilon / 1 2 .$ . Let $\widehat { \bf w }$ denote its output.   
6: Return: $h ( \mathbf { x } ) = \mathrm { s i g n } ( \langle \widehat { \mathbf { w } } , \Psi _ { t } ( \mathbf { x } ) \rangle )$

We now prove Theorem 8.1.

Proof of Theorem 8.1. We analyze Algorithm 5. Apply the construction in the proof of Lemma 8.2 with accuracy ϵ/12. Let

$$
m = O \big ( k ^ { 2 } \log ^ { 2 } ( 2 k ) \log ( 1 / \epsilon ) \big ) , \qquad t = m / 2 , \qquad W = 2 ^ { O ( k ^ { 2 } \log ^ { 3 } ( 2 k ) \log ^ { 2 } ( 1 / \epsilon ) ) } ,
$$

where $m$ is even and the constant in the definition of W is suficiently large. This gives a polynomial $g ^ { \star }$ satisfying

$$
\begin{array} { r } { \underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } { \mathbf { E } } [ \mathrm { H i n g e } ( f ( \mathbf { x } ) , g ^ { \star } ( \mathbf { x } ) ) ] \leq \epsilon / 1 2 , \qquad \| g ^ { \star } \| _ { 2 } \leq W . } \end{array}
$$

We first bound the coeficients of $g ^ { \star }$ in the feature representation used by the algorithm. Normalize the defining polynomials

$$
p _ { i } ( \mathbf { x } ) = c _ { i } + \mathbf { u } _ { i } \cdot \Phi _ { 2 } ( \mathbf { x } )
$$

to have Gaussian $L _ { 2 }$ norm one. The construction gives

$$
g ^ { \star } ( \mathbf { x } ) = q ( p _ { 1 } ( \mathbf { x } ) / B , \ldots , p _ { k } ( \mathbf { x } ) / B ) ,
$$

where $B \geq 1$ , deg $q \leq t ,$ and, by Fact 8.4, the sum of the absolute coeficients of $q$ is at most $W$ after increasing the constant defining $W$

Let $\mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { n }$ be the orthonormal basis of $S$ computed by the algorithm, and write

$$
z _ { j } ( \mathbf { x } ) = \langle \mathbf { v } _ { j } , \Phi _ { 2 } ( \mathbf { x } ) \rangle , \qquad p _ { i } ( \mathbf { x } ) = c _ { i } + \sum _ { j = 1 } ^ { n } b _ { i j } z _ { j } ( \mathbf { x } ) .
$$

Orthonormality of the Hermite features and the normalization of $p _ { i }$ imply

$$
c _ { i } ^ { 2 } + \sum _ { j = 1 } ^ { n } b _ { i j } ^ { 2 } = 1 , \qquad | c _ { i } | + \sum _ { j = 1 } ^ { n } | b _ { i j } | \leq \sqrt { n + 1 } .
$$

Expanding the composition defining $g ^ { \star }$ , we therefore obtain

$$
g ^ { \star } ( \mathbf { x } ) = \langle \mathbf { w } ^ { \star } , \Psi _ { t } ( \mathbf { x } ) \rangle , \qquad \| \mathbf { w } ^ { \star } \| _ { 2 } \leq \| \mathbf { w } ^ { \star } \| _ { 1 } \leq W ( n + 1 ) ^ { t / 2 } .
$$

Accordingly, set $R = W ( n + 1 ) ^ { t / 2 }$ . This radius is determined by the known parameters and does not require knowledge of the defining polynomials.

Next, we bound the comparator’s hinge loss under the observed labels. Let $\delta = \mathbf { P r } _ { ( \mathbf { x } , y ) \sim D } [ y \neq$ $f ( \mathbf { x } ) ]$ ]. The hinge loss satisfies

$$
\mathrm { H i n g e } ( y , a ) \leq \mathrm { H i n g e } ( f ( \mathbf { x } ) , a ) + 2 | a | \mathbf { 1 } _ { \{ y \neq f ( \mathbf { x } ) \} } .
$$

Since the input marginal of D is Gaussian, Cauchy–Schwarz gives

$$
\begin{array} { r l r } { \underset { ( \mathbf { x } , y ) \sim D } { \mathbf { E } } [ \mathrm { H i n g e } ( y , g ^ { \star } ( \mathbf { x } ) ) ] \leq \epsilon / 1 2 + 2 \underset { ( \mathbf { x } , y ) \sim D } { \mathbf { E } } \left[ | g ^ { \star } ( \mathbf { x } ) | \mathbf { 1 } _ { \{ y \neq f ( \mathbf { x } ) \} } \right] } & { } & \\ { \leq \epsilon / 1 2 + 2 W \sqrt { \delta } \leq \epsilon / 4 , } \end{array}
$$

where the last inequality follows from the theorem’s noise assumption for a suficiently large universal constant $C .$

We now verify the moment bound needed to apply Fact 8.5. Let $M = \left( { { n + t } atop { t } } \right)$ be the number of coordinates of $\Psi _ { t }$ . Each $z _ { j }$ is a polynomial of degree at most two with Gaussian $L _ { 2 }$ norm one. Thus, for every multi-index α with $1 \leq | \alpha | = r \leq t$ , Hölder’s inequality and Gaussian hypercontractivity give

$$
\left\| \prod _ { j = 1 } ^ { n } z _ { j } ^ { \alpha _ { j } } \right\| _ { 2 } \leq \prod _ { j = 1 } ^ { n } \| z _ { j } \| _ { 2 r } ^ { \alpha _ { j } } \leq ( 2 r - 1 ) ^ { r } \leq ( 2 t ) ^ { t } .
$$

The constant feature has norm one, so

$$
\begin{array} { r } { \underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } { \mathbf { E } } \left\| \Psi _ { t } ( \mathbf { x } ) \right\| _ { 2 } ^ { 2 } \leq M ( 2 t ) ^ { 2 t } = : \sigma ^ { 2 } . } \end{array}
$$

Consider the convex objective

$$
F ( \mathbf { w } ) = \underset { ( \mathbf { x } , y ) \sim D } { \mathbf { E } } \left[ \operatorname { H i n g e } \bigl ( y , \langle \mathbf { w } , \boldsymbol { \Psi } _ { t } ( \mathbf { x } ) \rangle \bigr ) \right]
$$

over $\mathcal { C } = \left\{ \mathbf { w } \in \mathbb { R } ^ { M } : \left\| \mathbf { w } \right\| _ { 2 } \leq R \right\}$ . It is continuous, since

$$
| F ( \mathbf { w } ) - F ( \mathbf { v } ) | \leq \sigma \left\| \mathbf { w } - \mathbf { v } \right\| _ { 2 } .
$$

A fresh labeled example provides the stochastic subgradient

$$
G ( \mathbf { w } ; \mathbf { x } , y ) = - y \Psi _ { t } ( \mathbf { x } ) \mathbf { 1 } _ { \{ y \langle \mathbf { w } , \Psi _ { t } ( \mathbf { x } ) \rangle < 1 \} } ,
$$

whose expectation belongs to $\partial F ( \mathbf { w } )$ and whose second moment is at most $\sigma ^ { 2 }$ . Moreover, C has diameter 2R and admits an $O ( M )$ -time Euclidean projection.

Applying Fact 8.5 with

$$
T = \left\lceil \frac { 1 2 9 6 R ^ { 2 } \sigma ^ { 2 } } { \epsilon ^ { 2 } } \right\rceil
$$

samples therefore returns ${ \widehat { \mathbf { w } } } \in { \mathcal { C } }$ satisfying

$$
\mathbf { E } [ F ( \widehat { \mathbf { w } } ) ] \leq F ( \mathbf { w } ^ { \star } ) + \frac { 3 R \sigma } { \sqrt { T } } \leq \epsilon / 4 + \epsilon / 1 2 = \epsilon / 3 .
$$

For the returned hypothesis

$$
h ( \mathbf { x } ) = \mathrm { s i g n } \langle \widehat { \mathbf { w } } , \Psi _ { t } ( \mathbf { x } ) \rangle ,
$$

misclassification loss is at most hinge loss. Hence

$$
\begin{array} { r } { \mathbf { E } \left[ \underset { ( \mathbf { x } , y ) \sim D } { \mathbf { P r } } [ h ( \mathbf { x } ) \neq y ] \right] \leq \epsilon / 3 . } \end{array}
$$

Markov’s inequality implies that h has classification error at most ϵ with probability at least $2 / 3$

Finally, the algorithm uses $T$ samples. Computing the lifted coordinates and their monomials, followed by a subgradient step and projection, takes $( \bar { d + 1 } ) ^ { O ( 1 ) } M$ arithmetic operations per iteration. The basis and feature preprocessing also fit within the claimed bound. Using $M \ \leq \ ( n + 1 ) ^ { t }$ $R ^ { 2 } = W ^ { 2 } ( n + 1 ) ^ { t }$ , and $\sigma ^ { 2 } = M ( 2 t ) ^ { 2 t }$ , we obtain

$$
M T = O \left( \frac { W ^ { 2 } ( n + 1 ) ^ { 3 t } ( 2 t ) ^ { 2 t } } { \epsilon ^ { 2 } } \right) .
$$

Since $t = O ( k ^ { 2 } \log ^ { 2 } ( 2 k ) \log ( 1 / \epsilon ) )$ and

$$
t \log ( 2 t ) + \log ( W / \epsilon ) = O \bigl ( k ^ { 2 } \log ^ { 3 } ( 2 k ) \log ^ { 2 } ( 1 / \epsilon ) \bigr ) ,
$$

both the sample complexity and arithmetic running time are at most

$$
d ^ { O ( 1 ) } n ^ { O ( k ^ { 2 } \log ^ { 2 } ( k ) \log ( 1 / \epsilon ) ) } 2 ^ { O ( k ^ { 2 } \log ^ { 3 } ( k ) \log ^ { 2 } ( 1 / \epsilon ) ) } .
$$

This proves the theorem.

## 9 Improved Complexity through Dimension Reduction and Hinge Loss Optimization

In this section, we combine two stages of dimension reduction with hinge-loss minimization to improve the dependence on the accuracy parameter from $2 ^ { \mathrm { p o l y } _ { k } ( 1 / \epsilon ) } \ \mathrm { t o } \ 2 ^ { \mathrm { p o l y } _ { k } ( \log ( 1 / \epsilon ) ) }$ . The main idea is to identify a low-dimensional space of polynomial features that supports a suficiently accurate approximation to the target, allowing us to apply the hinge-loss learner of Theorem 8.1. Our first dimension-reduction step identifies a subspace $V$ such that the randomized error is small along every direction orthogonal to $V .$ . While this provides the regularity needed to control the residual polynomial contributions, the resulting feature space may still be too large for eficient hinge-loss minimization. To overcome this, we develop a second dimension-reduction procedure that uses labeled moments to recover a low-dimensional subspace $U$ capturing the relevant residual polynomial features, without explicitly identifying the defining polynomials. We show that the combined space $\mathcal { P } _ { 2 } ( V ) + U$ supports a suficiently accurate intersection, enabling hinge-loss minimization in this reduced feature space.

Theorem 9.1 (Learning by dimension reduction and hinge loss). Consider the problem of learning an intersection of k degree-2 PTFs under a Gaussian distribution. There is an algorithm such that given any $\epsilon , \delta \in ( 0 , 1 )$ , it draws $d ^ { O ( k ) } 2 ^ { O _ { k } ( \log ^ { 3 } ( 1 / \epsilon ) ) } \log ( 1 / \delta )$ samples from $D ,$ , runs in $d ^ { O ( k ) } 2 ^ { O _ { k } ( \log ^ { 3 } ( 1 / \epsilon ) ) } \log ( 1 / \delta )$ time and outputs a hypothesis $\hat { h } : \mathbb { R } ^ { d }  \{ 0 , 1 \}$ such that with probability at least $1 - \delta , \mathrm { e r r } _ { D } ( \hat { h } ) \leq \epsilon$

We give the intuition below. Notice that Algorithm 4 makes use of Theorem 5.1 to adaptively find directions that are important to the current intersections of PTFs, and when we cannot do so, the resulting instance is close to a jointly regular instance, where we can make use of the subspace recovery algorithm over the subspace spanned by the degree-2 Hermite polynomials. However, such a method will essentially produces $2 ^ { \mathrm { p o l y } ( 1 / \epsilon ) }$ many subproblems to solve, which introduces a $2 ^ { \mathrm { p o l y } ( 1 / \epsilon ) }$ complexity. The intuition is that the regularity condition required for implementing the algorithm from [DKS18] turns out to be much stronger than what the learning task really needs. To obtain an improved complexity, our motivation is that combined with Theorem 5.1, we want to construct a single subspace of the space of degree-2 Hermite polynomials, such that some intersection of k degree-2 PTFs from that subspace is very close to the target instance.

Our approach works as follows. By Theorem 5.1 and Theorem 6.2, we can find a single subspace $V \subseteq \mathbb { R } ^ { d }$ of dimension $2 ^ { O _ { k } ( \log ^ { 2 } ( 1 / \epsilon ) ) }$ such that for every unit vector $\textbf { v } \perp V$ , Rand $f * ( \mathbf { v } )$ is tiny. We write ${ \mathbf { x } } = { \mathbf { z } } + { \mathbf { w } }$ , where $\mathbf { z } \in V$ and $\mathbf { w } \perp V$ . To simplify the notation, we denote by $n = \dim ( V )$ and $m = d - n$ . Notice that for each quadratic function $q _ { i } ( \mathbf { x } )$ that defines the target function, we can write

$$
q _ { i } ( \mathbf { z } , \mathbf { w } ) = a _ { i } ( \mathbf { z } ) + ( \mathbf { b } _ { i } + \mathbf { L } _ { i } ^ { \top } \mathbf { z } ) ^ { \top } \mathbf { w } + \mathbf { w } ^ { \top } \mathbf { A } _ { i } \mathbf { w } - \operatorname { t r } ( \mathbf { A } _ { i } ) ,
$$

where $a _ { i } ( \mathbf { z } )$ is a quadratic function that purely depends on z, $\mathbf { b } _ { i } \in \mathbb { R } ^ { m } , ~ \mathbf { L } _ { i } : \mathbb { R } ^ { m }  \mathbb { R } ^ { n }$ and $\mathbf { A } _ { i } \in \mathbb { R } ^ { m \times m }$ . We will use the subspace V that we found to construct another subspace $\bar { V } \subseteq P _ { \leq 2 } ( \mathbb { R } ^ { d } )$ such that $P _ { \leq 2 } ( V )$ together with $\bar { V }$ contains some intersections of k degree-2 PTFs that is close to $f ^ { * }$ Similar to Section 3, where we use algorithm from [DKS18], and look at the second moment matrix over the space $P _ { \leq 2 } ( \mathbb { R } ^ { d } )$ and find a subspace that almost spans the linear combination coeficients. Our method will focus on the subspace of the polynomials of the form $( \mathbf { b } + \mathbf { L } ^ { \top } \mathbf { z } ) ^ { \top } \mathbf { w } + \mathbf { w } ^ { \top } \mathbf { A } \mathbf { w } - \mathrm { t r } ( \mathbf { A } )$ In particular, we will use $\mathbf { u } = ( \mathbf { b } , \mathbf { L } , \mathbf { A } )$ to vectorize the coeficient of the polynomial. Denote by S this space of coeficients. We will build a bilinear map $\mathcal { M } : \mathcal { S } \times P _ { \leq 2 k } ( \mathbb { R } ^ { d } ) \to \mathbb { R }$ , such that the required subspace V<sup>¯</sup> would be the space of large singular vectors of $\mathcal { M } .$

## 9.1 Subspace Recovery through Gaussian Comparison

To begin with, we first work on an idealized Gaussian model, to get a stronger property Lemma 9.2 than the one we used from [DKS18].

Lemma 9.2. Let $V \subseteq \mathbb { R } ^ { d }$ . Write $\mathbf { x } = \mathbf { z } + \mathbf { w } \in \mathbb { R } ^ { d }$ , where $\mathbf { z } \in V$ and $\pmb { w } \in V ^ { \perp }$ . Let $f ( \mathbf { x } ) = \mathbb { 1 } \{ q _ { 1 } ( \mathbf { x } ) \geq$ $0 , \ldots , q _ { k } ( \mathbf { x } ) \geq 0 \}$ , where $f o r ~ i \in [ k ] , ~ q _ { i } ( \mathbf { x } ) = a _ { i } ( \mathbf { z } ) + \mathbf { u } _ { i } \cdot \mathbf { w } , ~ \| q _ { i } ( \mathbf { x } ) \| = 1$ ,where $a _ { i } ( \mathbf { z } )$ is a polynomial of degree at most 2 that only depends on z. Let $\mathbf { v } \in V ^ { \perp }$ be any unit vector and let $\alpha \in ( 0 , 1 )$ . Suppose $\mathrm { R a n d } _ { f } ( \mathbf { v } ) \geq \alpha _ { \mathrm { \ m } }$ , there exists a polynomial $p : \mathbb { R } ^ { d }  \mathbb { R }$ , with $\deg ( p ) \leq 2 k$ and $\| p \| \leq 1$ such that $\mathbf { E _ { x \sim \mathcal { N } ( 0 , \mathbf { I } ) } } [ f ( \mathbf { x } ) M _ { \mathbf { v } } p ( \mathbf { x } ) ] \geq \Omega ( ( \alpha / k ) ^ { 3 k } )$ . In particular, there exists a polynomial $P : \mathbb { R } ^ { k }  \mathbb { R }$ such that $p ( \mathbf { x } ) = P ( q _ { 1 } ( \mathbf { x } ) , \ldots , q _ { k } ( \mathbf { x } ) )$

Roughly speaking, Lemma 9.2 considers a family of intersections of k degree-2 polynomials, whose defining polynomials have a special form of $q _ { i } ( \mathbf { x } ) = a _ { i } ( \mathbf { z } ) + \mathbf { u } _ { i } \cdot \mathbf { w }$ , where $a _ { i }$ is a quadratic function that only depends on a given subspace $V _ { \mathrm { : } }$ , while in $V ^ { \perp } , q _ { i }$ only has a linear dependence. Notice that if $V = \emptyset$ , then the instance degenerates to an intersections of k halfspaces. The property given by [DKS18] says that if a direction $\mathbf { v } \in V ^ { \perp }$ has large randomized error, then there is a degree-2 polynomial associated with v to certify this fact. Thus, by using the first 2 Chow-moments, we are able to find all these important directions. Here, for a general subspace $V ,$ we still want to identify all directions in $V ^ { \perp }$ that have large randomized error through moment methods. However, the existence of the quadratic functions $a _ { i } ( \mathbf { z } )$ can hide the signals from the first two moments and thus we need to use higher moments for this purpose.

The idea is similar to the certificate polynomial construction in Section $5 ,$ where fix a pair of $\mathbf { z } \in V , \mathbf { u } \in V ^ { \perp } , \mathbf { u } \perp \mathbf { v } , f _ { ( \mathbf { z } , \mathbf { u } ) }$ gives a one dimensional Boolean function. We would like to construct a polynomial h such that $\Delta _ { f _ { ( \mathbf { z } , \mathbf { u } ) } } ( s ) h ( s ) \phi ( s )$ is non-trivially large at any boundary of $f _ { ( \mathbf { z } , \mathbf { u } ) }$ . The special form of f implies that once $( \mathbf { z } , \mathbf { u } )$ is fixed, the function is just an indicator function of an interval. This fact makes the construction much easier than Section 5. In fact, we can simply choose $\begin{array} { r } { h ( \mathbf { x } ) = \prod _ { i : \mathbf { u } _ { i } \cdot \mathbf { v } < 0 } q _ { i } ( \mathbf { x } ) - \prod _ { i : \mathbf { u } _ { i } \cdot \mathbf { v } > 0 } q _ { i } ( \mathbf { x } ) } \end{array}$ . Once $( \mathbf { z } , \mathbf { u } )$ is fixed, each $q _ { i }$ is a monotone linear function along v direction, which naturally gives the certificate polynomial that we want.

Proof of Lemma 9.2. We consider polynomial $\begin{array} { r } { h ( \mathbf { x } ) = \prod _ { i : \mathbf { u } _ { i } \cdot \mathbf { v } < 0 } q _ { i } ( \mathbf { x } ) - \prod _ { i : \mathbf { u } _ { i } \cdot \mathbf { v } \geq 0 } q _ { i } ( \mathbf { x } ) } \end{array}$ . We write $\mathbf { x } = \mathbf { z } + s \mathbf { v } + \mathbf { u }$ , where $\mathbf { u } \perp V , \mathbf { u } \perp \mathbf { v } , s \in \mathbb { R }$ . By Lemma 5.6, we know that

$$
\begin{array} { r l } & { \quad \underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ f ( \mathbf { x } ) M _ { \mathbf { v } } h ( \mathbf { x } ) ] } \\ & { = \underset { \mathbf { z } , \mathbf { u } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ \underset { s \in S _ { ( \mathbf { z } , \mathbf { u } ) } } { \sum } \Delta _ { f _ { ( \mathbf { z } , \mathbf { u } ) } } ( s ) h ( s \mathbf { v } + \mathbf { u } + \mathbf { z } ) \phi ( s ) ] . } \end{array}
$$

Since each $q _ { i } ( \mathbf { x } )$ depends linearly on $\mathbf { w } _ { \mathrm { : } }$ , for each fixed $( \mathbf { z } , \mathbf { u } )$ the function $f _ { ( \mathbf { z } , \mathbf { u } ) } ( s )$ agrees almost everywhere with the indicator of an interval, possibly empty or all of R. Layers on which $f _ { ( \mathbf { z } , \mathbf { u } ) }$ is constant almost everywhere contribute zero to both the correlation and the randomized error, and are omitted from the following layerwise analysis. For each remaining layer, denote the interval by $( l _ { ( \mathbf { z } , \mathbf { u } ) } , r _ { ( \mathbf { z } , \mathbf { u } ) } )$ , allowing infinite endpoints. All endpoint identities below are asserted only at finite endpoints. Boundary terms at infinite endpoints are interpreted as zero by taking limits, and any condition in the definition of $B _ { t }$ involving an infinite endpoint is omitted. This implies that

$$
\begin{array} { r } { \displaystyle \sum _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ f ( \mathbf { x } ) M _ { \mathbf { v } } h ( \mathbf { x } ) ] = \displaystyle \sum _ { \mathbf { z } , \mathbf { u } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ h ( l _ { ( \mathbf { z } , \mathbf { u } ) } \mathbf { v } + \mathbf { u } + \mathbf { z } ) \phi ( l _ { ( \mathbf { z } , \mathbf { u } ) } ) - h ( r _ { ( \mathbf { z } , \mathbf { u } ) } \mathbf { v } + \mathbf { u } + \mathbf { z } ) \phi ( r _ { ( \mathbf { z } , \mathbf { u } ) } ) ] . } \end{array}
$$

Notice that for some $i \in [ k ]$ . If $\mathbf { v } \cdot \mathbf { u } _ { i } < 0 .$ , then $q _ { i } ( s \mathbf { v } + \mathbf { u } + \mathbf { z } )$ is a decreasing function and $q _ { i } ( l _ { ( \mathbf { z } , \mathbf { u } ) } \mathbf { v } + \mathbf { z } + \mathbf { u } ) \geq 0$ , while if $\mathbf { v } \cdot \mathbf { u } _ { i } \geq 0$ , then $q _ { i } ( s \mathbf { v } + \mathbf { u } + \mathbf { z } )$ is a nondecreasing function and $q _ { i } ( r _ { ( \mathbf { z } , \mathbf { u } ) } \mathbf { v } + \mathbf { z } + \mathbf { u } ) \geq 0$ . In particular, this implies $\begin{array} { r } { \prod _ { i : { \bf u } _ { i } \cdot { \bf v } > 0 } q _ { i } ( l _ { ( { \bf z } , { \bf u } ) } { \bf v } + { \bf z } + { \bf u } ) = 0 } \end{array}$ and $\begin{array} { r } { \prod _ { i : { \bf u } _ { i } \cdot { \bf v } < 0 } q _ { i } ( r _ { ( { \bf z } , { \bf u } ) } { \bf v } + { \bf z } + { \bf u } ) = 0 } \end{array}$ . Otherwise, it contradicts the definition of the $r _ { ( \mathbf { z } , \mathbf { u } ) } , l _ { ( \mathbf { z } , \mathbf { u } ) }$ . So, we have

$$
\boldsymbol { \underline { { \bf { E } } } } _ { \times \sim \mathcal { N } ( 0 , 1 ) } [ f ( { \bf x } ) M _ { \bf v } h ( { \bf x } ) ] = \boldsymbol { \underline { { \bf { E } } } } _ { \times \sim \mathcal { N } ( 0 , 1 ) } [ \prod _ { i : { \bf u } _ { i } , { \bf v } < 0 } q _ { i } ( l _ { ( z , { \bf u } ) } { \bf v } + { \bf z } + { \bf u } ) \phi ( l _ { ( z , { \bf u } ) } ) + \prod _ { i : { \bf u } _ { i } , { \bf v } > 0 } q _ { i } ( r _ { ( z , { \bf u } ) } { \bf v } + { \bf z } + { \bf u } ) \phi ( r _ { ( z , { \bf u } ) } ) ] ,
$$

which is nonnegative. We will show that ${ \mathrm { R a n d } } _ { f } ( \mathbf { v } )$ can be upper bounded in terms of $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ f ( \mathbf { x } ) M _ { \mathbf { v } } h ( \mathbf { x } ) ]$ Recall that

$$
\begin{array} { r l } & { \mathrm { R a n d } _ { f } ( \mathbf { v } ) = \underset { \mathbf { u } , \mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } \left[ \underset { s , s ^ { \prime } \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { P r } } [ f ( \mathbf { u } + \mathbf { z } + s \mathbf { v } ) \neq f ( \mathbf { u } + \mathbf { z } + s ^ { \prime } \mathbf { v } ) ] \right] } \\ & { \quad \quad = \underset { \mathbf { u } , \mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } \left[ 2 \underset { s \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { P r } } [ f ( \mathbf { u } + \mathbf { z } + s \mathbf { v } ) = 1 ] ( 1 - \underset { s \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { P r } } [ f ( \mathbf { u } + \mathbf { z } + s \mathbf { v } ) = 1 ] ) \right] . } \end{array}
$$

To simplify the notations, we denote by $I _ { - } = \{ i \in [ k ] \ | \ \mathbf { u } _ { i } \cdot \mathbf { v } < 0 \}$ and $I _ { + } = \{ i \in [ k ] \ | \ \mathbf { u } _ { i } \cdot \mathbf { v } \geq 0 \}$ Let $B _ { t }$ be the event over $( \mathbf { z } , \mathbf { u } )$ such that there exists some $q _ { i } ( l _ { ( \mathbf { z } , \mathbf { u } ) } \mathbf { v } + \mathbf { z } + \mathbf { u } ) \leq t , \ i \in I _ { - }$ or $q _ { i } ( r _ { ( \mathbf { z } , \mathbf { u } ) } \mathbf { v } + \mathbf { z } + \mathbf { u } ) \leq t , i \in I _ { + }$ . We will separate the analysis for $( \mathbf { u } , \mathbf { z } ) \in B _ { t }$ and $( \mathbf { u } , \mathbf { z } ) \notin B _ { t }$

We start by analyzing the case where $( \mathbf { u } , \mathbf { z } ) \in B _ { t }$ . By the monotonicity of $q _ { i }$ as a function of s along direction v, we know that if $f ( \mathbf { x } ) = 1$ and $B _ { t }$ happen together, then there must be some $i \in [ k ]$ such that $| q _ { i } ( \mathbf { x } ) | \leq t .$ . This implies that

$$
\underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { P r } } [ f ( \mathbf { x } ) = 1 , B _ { t } ] \leq \sum _ { i \in [ k ] } [ \underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { P r } } | q _ { i } ( \mathbf { x } ) | \leq t ] \leq O ( k \sqrt { t } ) .
$$

This implies that

$$
\begin{array} { r } { \underset { \mathbf { u } , \mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } \bigg [ 2 \underset { s \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { P r } } [ f ( \mathbf { u } + \mathbf { z } + s \mathbf { v } ) = 1 ] ( 1 - \underset { s \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { P r } } [ f ( \mathbf { u } + \mathbf { z } + s \mathbf { v } ) = 1 ] ) \mathbb { 1 } \{ ( \mathbf { u } , \mathbf { z } ) \in B _ { t } \} \bigg ] \leq O ( k \sqrt { t } ) . } \end{array}
$$

Next, we analyze the case where $( \mathbf { u } , \mathbf { z } ) \notin B _ { t }$ . In this case, we have

$$
\begin{array} { r l } & { \left( \displaystyle \prod _ { i : \mathbf { u } _ { i } . \mathbf { v } < 0 } q _ { i } ( l _ { ( \mathbf { z } , \mathbf { u } ) } \mathbf { v } + \mathbf { z } + \mathbf { u } ) \phi ( l _ { ( \mathbf { z } , \mathbf { u } ) } ) + \displaystyle \prod _ { i : \mathbf { u } _ { i } . \mathbf { v } \geq 0 } q _ { i } ( r _ { ( \mathbf { z } , \mathbf { u } ) } \mathbf { v } + \mathbf { z } + \mathbf { u } ) \phi ( r _ { ( \mathbf { z } , \mathbf { u } ) } ) \right) \mathbb { 1 } \{ ( \mathbf { u } , \mathbf { z } ) \not \in B _ { t } \} } \\ & { \geq t ^ { k } ( \phi ( l _ { ( \mathbf { z } , \mathbf { u } ) } ) + \phi ( r _ { ( \mathbf { z } , \mathbf { u } ) } ) ) \mathbb { 1 } \{ ( \mathbf { u } , \mathbf { z } ) \not \in B _ { t } \} } \end{array}
$$

By Fact 5.8, we know that

$$
\phi ( l _ { ( \mathbf { z } , \mathbf { u } ) } ) + \phi ( r _ { ( \mathbf { z } , \mathbf { u } ) } ) \ge \Omega ( \operatorname* { \bf \Pi } _ { s \sim \mathcal { N } ( 0 , 1 ) } [ f ( \mathbf { u } + \mathbf { z } + s \mathbf { v } ) = 1 ] ( 1 - \operatorname* { \bf \Pi } _ { s \sim \mathcal { N } ( 0 , 1 ) } [ f ( \mathbf { u } + \mathbf { z } + s \mathbf { v } ) = 1 ] ) ) ,
$$

which implies that

$$
\begin{array} { r l } & { t ^ { - k } \underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } \big [ f ( \mathbf { x } ) M _ { \mathbf { v } } h ( \mathbf { x } ) \big ] } \\ & { \ \geq \Omega \big ( \underset { \mathbf { u } , \mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } \left[ 2 \underset { s \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { P r } } [ f ( \mathbf { u } + \mathbf { z } + s \mathbf { v } ) = 1 ] ( 1 - \underset { s \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { P r } } [ f ( \mathbf { u } + \mathbf { z } + s \mathbf { v } ) = 1 ] ) \mathbb { 1 } \{ ( \mathbf { u } , \mathbf { z } ) \notin B _ { t } \} \right] \big ) . } \end{array}
$$

Together, this gives

$$
\mathrm { R a n d } _ { f } ( \mathbf { v } ) \leq O ( t ^ { - k } \underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ f ( \mathbf { x } ) M _ { \mathbf { v } } h ( \mathbf { x } ) ] + k \sqrt { t } ) .
$$

Now suppose that ${ \mathrm { R a n d } } _ { f } ( \mathbf { v } ) \geq \alpha$ . Choose $t = ( c \alpha / k ) ^ { 2 }$ , where $c > 0$ is a suficiently small universal constant. By the preceding inequality, $\mathbf { E _ { x \sim \mathcal { N } ( 0 , \mathbf { I } ) } } [ f ( \mathbf { x } ) M _ { \mathbf { v } } h ( \mathbf { x } ) ] \geq \Omega ( \alpha t ^ { k } )$ . By Hölder’s inequality and Gaussian hypercontractivity, for every nonempty $I \subseteq [ k ]$

$$
\left\| \prod _ { i \in I } q _ { i } \right\| _ { 2 } \leq ( 2 | I | ) ^ { | I | } \leq ( 2 k ) ^ { k } .
$$

Consequently, $\| h \| _ { 2 } \leq H _ { k } = 2 ( 2 k ) ^ { k }$ . Taking $\begin{array} { r } { p ( \mathbf { x } ) = \frac { h ( \mathbf { x } ) } { H _ { k } } } \end{array}$ , we finish the proof.

That is to say once we can write each $q _ { i }$ that defines $f ( \mathbf { x } )$ in a special form of $a _ { i } ( { \bf z } ) + { \bf u } _ { i } \cdot { \bf w }$ then we can run a singular value decomposition of the bilinear form $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ f ( \mathbf { x } ) M _ { \mathbf { v } } p ]$ to find the directions in the w space that is important to $f .$ . To use this fact to design a learning algorithm for learning a general intersection of k PTFs, the intuition follows from Section 3 and Section 4.

By Theorem 6.2, the subspace V can be chosen so that for every $\mathbf { v } \perp V$ , Rand<sub>f</sub>(v) is tiny. By Theorem 4.1, we know that $f$ must be close to a $( \tau , V )$ -regular instance for some suficiently small $\tau .$ This implies some form of Gaussian comparison can be done, the random polynomial vector, $( { \bf z } , N _ { { \bf u } _ { 1 } } ( { \bf z } , { \bf w } ) , \ldots , N _ { { \bf u } _ { k } } ( { \bf z } , { \bf w } ) )$ , where ${ \bf z } , { \bf w } \sim \mathcal { N } ( 0 , { \bf I } )$ and $N _ { { \bf u } _ { i } } ( { \bf z } , { \bf w } ) = ( { \bf b } _ { i } { + } { \bf L } _ { i } ^ { \top } { \bf z } ) ^ { \top } { \bf w } { + } { \bf w } ^ { \top } { \bf A } _ { i } { \bf w } { - } \mathrm { t r } ( { \bf A } _ { i } )$ is close to $( \mathbf { z } , \mathbf { u } _ { 1 } \cdot G , \dots , \mathbf { u } _ { k } \cdot G )$ , where $G$ is an independent Gaussian. Then, since we are able to use the SVD type method in the idealized Gaussian model, we translate this into the original model over $\mathbb { R } ^ { d }$ that we care about.

Recall that the bilinear form we consider in Lemma 9.2 satisfies

$$
\underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ f ( \mathbf { x } ) M _ { \mathbf { v } } p ] = \underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ p \partial _ { \mathbf { v } } f ] ,
$$

which is the key property used in the proof of Lemma 9.2. However, in the set up we care about a vector v represents a polynomial $N _ { \mathbf { v } }$ instead of a point in $\mathbb { R } ^ { d }$ , which makes the notion $\partial _ { v } f$ not well defined in our set up. ${ \mathrm { S o } } ,$ the next step will be to properly design a bilinear form in the space we are working on to preserve the property of ${ \bf E _ { x \sim \mathcal { N } ( 0 , { \bf I } ) } } [ f ( { \bf x } ) M _ { \bf v } p ] = { \bf E _ { x \sim \mathcal { N } ( 0 , { \bf I } ) } } [ p \partial _ { \bf v } f ]$ in some way so that we can use Gaussian comparison to show that an SVD over it can recover important $N _ { \mathbf { v } }$

Specifically, under the idealized Gaussian model, since $f ( \mathbf { x } ) = \mathbb { 1 } \{ q _ { 1 } ( \mathbf { z } , G ) \geq 0 , \dots , q _ { k } ( \mathbf { z } , G ) \geq 0 \}$ 2 which can be seen as a function of $q _ { 1 } , \ldots , q _ { k }$ , for a unit vector v in $V ^ { \perp }$ , we have

$$
\underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ f ( \mathbf { x } ) M _ { \mathbf { v } } p ] = \underset { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ p \partial _ { \mathbf { v } } f ] = \sum _ { i = 1 } ^ { k } ( \mathbf { v } \cdot \mathbf { u } _ { i } ) \underset { \mathbf { z } , \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ p \frac { \partial f } { \partial q _ { i } } ] .\tag{60}
$$

We would like to build another bilinear form $B _ { f } : \mathcal { S } \times P _ { \leq 2 k } ( \mathbb { R } ^ { d } )$ such that for every $\mathbf { v } \in { \mathcal { S } }$ and for every polynomial $\begin{array} { r } { p , \ \left| B _ { f } ( \mathbf { v } , p ) - \sum _ { i = 1 } ^ { k } ( \mathbf { v } \cdot \mathbf { u } _ { i } ) \mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p \frac { \partial f } { \partial q _ { i } } ] \right| } \end{array}$ is small. If this approximation holds, we can recover the subspace $\bar { V }$ by using the large singular vector spaces of $B _ { f }$ . The proof follows Gaussian comparison results that we have established earlier. We give an overview about this. On the one hand, for every $\mathbf v \perp \mathrm { s p a n } ( \mathbf u _ { 1 } , \ldots , \mathbf u _ { k } ) , \vert B _ { f } ( \mathbf v , p ) \vert$ is small, which implies that $B _ { f }$ is approximately low-rank. On the other hand, by the special certificate polynomial we construct in Lemma 9.2 and gaussian comparison, every important direction v must be captured in the large singular vector spaces of $B _ { L }$ . This implies that if we take the top k singular vectors of this bilinear form $B _ { f }$ and span a subspace V<sup>¯</sup> , then any vector $\mathbf { u } \perp \bar { V }$ cannot introduce large error in the idealized Gaussian model. Thus, Lemma 3.7 immediately shows that we can search an intersection of k PTFs in the found subspace together with $P _ { \leq 2 } ( V )$ that is suficiently close to $f .$ . We next formalize this argument.

We first describe the approximation property required of the bilinear form, without specifying how it is constructed. Fix a smooth function $\chi : \mathbb { R } \to [ 0 , 1 ]$ equal to zero on $( - \infty , - 1 ]$ and one on $[ 1 , \infty )$ . For $h \in ( 0 , 1 )$ , define

$$
\psi _ { h } ( t _ { 1 } , \dots , t _ { k } ) = \prod _ { i = 1 } ^ { k } \chi ( t _ { i } / h ) , \qquad f _ { h } = \psi _ { h } ( q _ { 1 } , \dots , q _ { k } ) , \qquad \widetilde { f } _ { h } = \psi _ { h } ( \widetilde { q } _ { 1 } , \dots , \widetilde { q } _ { k } ) .
$$

We use the notation

$$
\frac { \partial f _ { h } } { \partial q _ { i } } : = \partial _ { i } \psi _ { h } ( q _ { 1 } , . . . , q _ { k } ) .
$$

These regularized partial derivatives are used only in the analysis. In particular, the algorithm will not need the defining polynomials or values of $f _ { h }$

For a bilinear form on $\mathcal { S } \times \mathcal { P } _ { < 2 k } ( \mathbb { R } ^ { d } )$ , set

$$
\| B \| _ { \mathrm { o p } } = \operatorname* { s u p } _ { \| \mathbf { v } \| \leq 1 , \| p \| _ { 2 } \leq 1 } | B ( \mathbf { v } , p ) | .
$$

Representing both spaces in orthonormal bases identifies this with the usual matrix operator norm; the left singular vectors belong to ${ \mathcal { S } } .$

Lemma 9.3 (Subspace recovery from derivative approximation). Let $q _ { 1 } , \ldots , q _ { k }$ satisfy the normalization above and suppose each q<sub>i</sub> is $( \tau , V )$ -regular, with $\tau \in ( 0 , 1 )$ . Let $h \in ( 0 , 1 )$ and $\alpha \in ( 0 , 1 / 2 )$ Suppose a bilinear form $B _ { f } : \mathcal { S } \times \mathcal { P } _ { \leq 2 k } ( \mathbb { R } ^ { d } ) \to \mathbb { R }$ satisfies, for every $\mathbf { v } \in { S }$ and $p \in \mathcal { P } _ { \leq 2 k } ( \mathbb { R } ^ { d } )$

$$
\left| B _ { f } ( \mathbf { v } , p ) - \sum _ { i = 1 } ^ { k } \langle \mathbf { v } , \mathbf { u } _ { i } \rangle \operatorname { \bf \Pi } _ { \mathbf { z } , \mathbf { w } } \left[ p \frac { \partial f _ { h } } { \partial q _ { i } } \right] \right| \leq \xi \| \mathbf { v } \| \| p \| _ { 2 } .\tag{61}
$$

Let $\widehat { B }$ satisfy $\| \widehat { B } - B _ { f } \| _ { \mathrm { o p } } \leq \nu ,$ and let $\bar { V } \subseteq { S }$ be the span of its top min $\{ k ,$ dim $s \}$ left singular vectors. There is $C _ { k } \doteq 2 ^ { \cal O ( k ) }$ such that, if

$$
2 \xi + 2 \nu + C _ { k } \big ( h ^ { 1 / 4 } + \tau h ^ { - 3 } \big ) \leq \frac { \beta _ { k } ( \alpha ) } { 2 } ,\tag{62}
$$

then there exists an intersection $g = \mathbf { 1 } \{ g _ { i } \geq 0$ for all $i \in [ k ] \}$ with

$$
g _ { i } \in \mathcal { P } _ { \leq 2 } ( V ) \oplus \{ N _ { \mathbf { u } } : \mathbf { u } \in \bar { V } \} , \qquad \mathbf { P r } [ f \neq g ] \leq C _ { \mathrm { t r } , k } ( k \alpha + \tau ) ^ { 1 / 2 0 } .
$$

Here $C _ { \mathrm { t r } , k }$ is the constant in the quantitative conclusion of Lemma 3.7.

Proof of Lemma 9.3. If $\bar { V } = { \mathcal { S } }$ , take $g = f$ . We henceforth assume $k < \dim { \mathcal { S } }$

Define the bilinear form

$$
A _ { h } ( \mathbf { v } , p ) = \sum _ { i = 1 } ^ { k } \langle \mathbf { v } , \mathbf { u } _ { i } \rangle \operatorname* { \bf { \Pi } } _ { \mathbf { z } , \mathbf { w } } \left[ p \frac { \partial f _ { h } } { \partial q _ { i } } \right] .
$$

It has rank at most $k ,$ and (61) gives $\| B _ { f } - A _ { h } \| _ { \mathrm { o p } } \leq \xi$ . Thus $\sigma _ { k + 1 } ( \widehat { B } ) \leq \xi + \nu$ . By the definition of ${ \bar { V } } .$ , for every unit $\mathbf { v } \perp { \bar { V } }$

$$
\operatorname* { s u p } _ { \deg ( p ) \leq 2 k , \| p \| _ { 2 } \leq 1 } | B _ { f } ( \mathbf { v } , p ) | \leq \xi + 2 \nu .\tag{63}
$$

We show that every such direction has small randomized error in the ideal Gaussian model. Suppose instead that Rand $\mathbf { \Delta } _ { \cdot \widetilde { f } } ( \mathbf { v } ) \geq \alpha$ . Let P be the certificate from Lemma 9.2. Then

$$
\begin{array} { r } { \underline { { \mathbf { E } } } _ { , G } [ P ( \widetilde { q } ) \partial _ { \mathbf { v } } \widetilde { f } ] \geq \beta _ { k } ( \alpha ) . } \end{array}\tag{64}
$$

We will compare this quantity with $A _ { h } ( { \bf v } , P ( q ) )$

First, if $Q _ { 1 } , \ldots , Q _ { k }$ are degree-at-most-two polynomials of any standard Gaussian input with $\| Q _ { i } \| _ { 2 } \leq 1$ , Hölder’s inequality and Gaussian hypercontractivity give

$$
\left\| \prod _ { i \in J } Q _ { i } \right\| _ { 2 } \leq \prod _ { i \in J } \| Q _ { i } \| _ { 2 | J | } \leq ( 2 | J | - 1 ) ^ { | J | } \leq ( 2 k ) ^ { k } \quad ( J \neq \emptyset ) .
$$

The empty product has norm one. Every partial derivative of $P$ of order at most two is a signed sum of at most two such products divided by $H _ { k }$ . Consequently,

$$
\| P ( Q ) \| _ { 2 } \leq 1 , \qquad \| ( \partial _ { a } P ) ( Q ) \| _ { 2 } \leq 1 , \qquad \| ( \partial _ { a b } P ) ( Q ) \| _ { 2 } \leq 1 .\tag{65}
$$

In particular, $P ( q )$ and $P ( \widetilde q )$ have degree at most 2k and norm at most one.

For $s \in [ 0 , 1 ]$ , introduce the interpolation

$$
q _ { s , i } = a _ { i } ( \mathbf { z } ) + \sqrt { s } N _ { \mathbf { u } _ { i } } ( \mathbf { z } , \mathbf { w } ) + \sqrt { 1 - s } \langle \mathbf { u } _ { i } , G \rangle , \qquad Q _ { s } = ( q _ { s , 1 } , \ldots , q _ { s , k } ) .
$$

The residuals are centered conditional on $\mathbf { z } ,$ and $G$ is independent of $( \mathbf { z } , \mathbf { w } )$ . Therefore every $q _ { s , i }$ has degree at most two and $L _ { 2 }$ norm one under the joint input $( \mathbf { z } , \mathbf { w } , G )$

Fix $i \in [ k ]$ and put $J _ { i } ( t ) = P ( t ) \partial _ { i } \psi _ { h } ( t )$ . The interpolation identity in Fact 3.5, applied to $J _ { i \cdot }$ gives

$$
\frac { d } { d s } \mathop { \bf E } _ { { \bf z } , { \bf w } , G } ^ { } [ J _ { i } ( Q _ { s } ) ] = \frac { 1 } { 2 } \sum _ { a , b = 1 } ^ { k } \mathop { \bf E } _ { { \bf z } , { \bf w } , G } ^ { } \bigl [ ( \Gamma _ { a b } - \Sigma _ { a b } ) \partial _ { a b } J _ { i } ( Q _ { s } ) \bigr ] ,\tag{66}
$$

where, writing $c _ { i } ( { \bf z } ) = b _ { i } + L _ { i } ^ { \top } { \bf z } .$

$$
\Gamma _ { a b } = \left( c _ { a } + \mathbf { A } _ { a } \mathbf { w } \right) ^ { \top } ( c _ { b } + 2 \mathbf { A } _ { b } \mathbf { w } ) , \qquad \Sigma _ { a b } = \langle \mathbf { u } _ { a } , \mathbf { u } _ { b } \rangle .
$$

The regularity estimate of Fact 3.6 gives $\| \Gamma _ { a b } - \Sigma _ { a b } \| _ { 2 } \leq C \tau$ . Although $J _ { i }$ need not have bounded second derivatives, it and its derivatives have polynomial growth. Thus the diferentiation and Gaussian integrations by parts in Fact 3.5 remain valid. We use their identity directly, before taking a supremum norm of the second derivatives.

For $j \le 3$ , the order-j partial derivatives of $\psi _ { h }$ are bounded by $C _ { j } h ^ { - j }$ . Moreover,

$$
\partial _ { a b } J _ { i } = ( \partial _ { a b } P ) \partial _ { i } \psi _ { h } + ( \partial _ { a } P ) \partial _ { b i } \psi _ { h } + ( \partial _ { b } P ) \partial _ { a i } \psi _ { h } + P \partial _ { a b i } \psi _ { h } .
$$

By (65),

$$
\operatorname* { s u p } _ { s \in [ 0 , 1 ] } \lVert \partial _ { a b } J _ { i } ( Q _ { s } ) \rVert _ { 2 } \leq C h ^ { - 3 } .
$$

Cauchy–Schwarz in (66), followed by integration over $s ,$ therefore gives

$$
\begin{array} { r } { \Big | \underset { \mathbf { z } , \mathbf { w } } { \mathbf { E } } [ P ( q ) \partial _ { i } \psi _ { h } ( q ) ] - \underset { \mathbf { z } , G } { \mathbf { E } } [ P ( \widetilde { q } ) \partial _ { i } \psi _ { h } ( \widetilde { q } ) ] \Big | \leq C k ^ { 2 } \tau h ^ { - 3 } . } \end{array}\tag{67}
$$

All endpoint expectations are continuous by Gaussian moment bounds and polynomial growth.

Since $\| \mathbf { u } _ { i } \| \leq 1$ and $\| \mathbf { v } \| = 1$ , summing (67) with weights $\langle \mathbf { v } , \mathbf { u } _ { i } \rangle$ and applying the ideal chain rule gives

$$
\left| A _ { h } ( \mathbf { v } , P ( q ) ) - \underset { \mathbf { z } , G } { \mathbf { E } } [ P ( \widetilde { q } ) \partial _ { \mathbf { v } } \widetilde { f } _ { h } ] \right| \leq C k ^ { 3 } \tau h ^ { - 3 } .\tag{68}
$$

Here the exact identity used on the ideal side is

$$
\partial _ { \mathbf { v } } \widetilde { f } _ { h } = \sum _ { i = 1 } ^ { k } \langle \mathbf { v } , \mathbf { u } _ { i } \rangle \partial _ { i } \psi _ { h } ( \widetilde { q } ) .
$$

It remains to remove smoothing in this ideal derivative correlation. Degree-two Gaussian anti-concentration and a union bound give

$$
\| \widetilde { f } - \widetilde { f } _ { h } \| _ { 2 , \mathbf { z } , G } ^ { 2 } \le \sum _ { i = 1 } ^ { k } \mathbf { P r } [ | \widetilde { q } _ { i } | \le h ] \le C k \sqrt { h } .
$$

For a degree-at-most-2k polynomial $R ( \mathbf { z } , G )$ , Gaussian integration by parts and the normalized Hermite recurrence give

$$
\begin{array} { r } { \underline { { \mathbf { E } } } _ { G } [ R \partial _ { \mathbf { v } } H ] = \underline { { \mathbf { E } } } _ { G } [ H ( ( \mathbf { v } \cdot G ) R - \partial _ { \mathbf { v } } R ) ] , \qquad \| ( \mathbf { v } \cdot G ) R - \partial _ { \mathbf { v } } R \| _ { 2 } \le \sqrt { 2 k + 1 } \| R \| _ { 2 } . } \end{array}
$$

These identities also hold for bounded H in the weak sense. Taking $R = P ( \widetilde { q } )$ and $H = \widetilde { f } - \widetilde { f } _ { h }$ yields

$$
\begin{array} { r } { \left| \underline { { \mathbf { E } } } _ { \mathfrak { c } } [ P ( \widetilde { q } ) \partial _ { \mathbf { v } } ( \widetilde { f } - \widetilde { f } _ { h } ) ] \right| \le C \sqrt { k ( 2 k + 1 ) } h ^ { 1 / 4 } . } \end{array}
$$

Together with (68), this proves

$$
\begin{array} { r } { \left| A _ { h } ( \mathbf { v } , P ( q ) ) - \underset { \mathbf { z } , G } { \mathbf { E } } [ P ( \widetilde { q } ) \partial _ { \mathbf { v } } \widetilde { f } ] \right| \leq C _ { k } ( h ^ { 1 / 4 } + \tau h ^ { - 3 } ) . } \end{array}\tag{69}
$$

Using (61), (64), and (69), we obtain

$$
B _ { f } ( \mathbf { v } , P ( q ) ) \geq \beta _ { k } ( \alpha ) - \xi - C _ { k } ( h ^ { 1 / 4 } + \tau h ^ { - 3 } ) .
$$

Since $\| \ b { P } ( \ b { q } ) \| _ { 2 } \leq 1$ , comparison with (63) would imply

$$
\beta _ { k } ( \alpha ) \le 2 \xi + 2 \nu + C _ { k } ( h ^ { 1 / 4 } + \tau h ^ { - 3 } ) \le \beta _ { k } ( \alpha ) / 2 ,
$$

a contradiction. Hence

$$
\mathrm { R a n d } _ { \widetilde { f } } ( \mathbf { v } ) < \alpha \qquad \mathrm { f o r ~ e v e r y ~ u n i t ~ } \mathbf { v } \perp \bar { V } .
$$

Set

$$
E = \mathrm { s p a n } \{ \mathrm { p r o j } _ { \bar { V } ^ { \perp } } \mathbf { u } _ { i } : i \in [ k ] \} .
$$

Then $E \subseteq { \bar { V } } ^ { \bot } , \dim ( E ) \leq k$ , and $\widetilde { f }$ depends on the $\bar { V } ^ { \perp }$ coordinates only through their projection onto E. Choose an orthonormal basis of E and independently resample its coordinates one at a time. At each step the current Gaussian vector is standard and independent of $\mathbf { z } ,$ so the probability that the label changes is less than α. A union bound therefore gives

$$
\Pr _ { \mathbf { z } , G , G ^ { \prime } } \left[ \widetilde { f } ( \mathbf { z } , G ) \neq \widetilde { f } ( \mathbf { z } , \mathrm { p r o j } _ { \bar { V } } G + \mathrm { p r o j } _ { \bar { V } ^ { \perp } } G ^ { \prime } ) \right] \leq k \alpha ,
$$

where $G ^ { \prime }$ is an independent standard Gaussian on S. Applying Lemma 3.7 with $\delta = \operatorname* { m i n } \{ k \alpha , 1 \}$ gives an intersection with defining polynomials in $\mathcal { P } _ { \leq 2 } ( V ) \oplus \{ N _ { \mathbf { u } } : \mathbf { u } \in \bar { V } \}$ and error at most $C _ { \mathrm { t r } , k } ( k \alpha + \tau ) ^ { 1 / 2 0 }$ , as required. □

Finally it sufices to design a bilinear form that satisfies the requirement in the statement of Lemma 9.3. The intuition of designing this bilinear form is simple. Recall that in the Gaussian model, by integral by part, for a direction v and a (smooth) function $f$ and polynomial p, we have $\mathbf { E _ { x \sim \mathcal { N } ( 0 , I ) } } [ ( \mathbf { v } \cdot \mathbf { x } ) f ( \mathbf { x } ) ] = \mathbf { E _ { x \sim \mathcal { N } ( 0 , I ) } } [ \partial _ { \mathbf { v } } f ]$ , as well as ${ \bf E _ { x \sim \mathcal { N } ( 0 , { \bf I } ) } } [ f ( { \bf x } ) M _ { \bf v } p ] = { \bf E _ { x \sim \mathcal { N } ( 0 , { \bf I } ) } } [ \partial _ { \bf v } f p ]$ We would like to map these two properties to the setup, where we replace v by the corresponding polynomial $N _ { \mathbf { v } }$ . Using integral by part, we have

$$
\begin{array} { r l } & { \underset { ( \mathbf { z } , \mathbf { w } ) \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ N _ { \mathbf { u } } ( \mathbf { z } , \mathbf { w } ) f ] = \underset { ( \mathbf { z } , \mathbf { w } ) \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ ( ( \mathbf { b } + \mathbf { L } ^ { \top } \mathbf { z } ) ^ { \top } \mathbf { w } + \mathbf { w } ^ { \top } \mathbf { A } \mathbf { w } - \mathrm { t r } ( \mathbf { A } ) ) f ] } \\ & { \quad \quad \quad \quad = \underset { ( \mathbf { z } , \mathbf { w } ) \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } [ ( \mathbf { b } + \mathbf { L } ^ { \top } \mathbf { z } ) ^ { \top } \nabla _ { \mathbf { w } } f + \mathrm { t r } ( \mathbf { A } \nabla _ { \mathbf { w } } ^ { 2 } f ) ] } \end{array}
$$

We simplify the notation by defining $d _ { \mathbf { u } } ( f ) = ( \mathbf { b } + \mathbf { L } ^ { \top } \mathbf { z } ) ^ { \top } \nabla _ { \mathbf { w } } f + \mathrm { t r } ( \mathbf { A } \nabla _ { \mathbf { w } } ^ { 2 } f )$ , which is used as a suitable replacement of partial derivative. We give the following guarantee under the regularity assumption.

Lemma 9.4. Let $V \subseteq \mathbb { R } ^ { d } ;$ , and write ${ \mathbf { x } } = { \mathbf { z } } + { \mathbf { w } }$ , where $\textbf { z } \in \textbf { V }$ and $\mathbf { w } \in V ^ { \perp }$ are independent standard Gaussian vectors. For $i \in [ k ]$ , let $q _ { i } ( \mathbf { z } , \mathbf { w } ) = a _ { i } ( \mathbf { z } ) + N _ { \mathbf { u } _ { i } } ( \mathbf { z } , \mathbf { w } )$ , where $\mathbf { u } _ { i } = ( \mathbf { b } _ { i } , \mathbf { L } _ { i } , \mathbf { A } _ { i } )$ and $N _ { { \bf { u } } _ { i } } ( { \bf { z } } , { \bf { w } } ) = ( { \bf { b } } _ { i } + { \bf { L } } _ { i } ^ { \top } { \bf { z } } ) ^ { \top } { \bf { w } } + { \bf { w } } ^ { \top } { \bf { A } } _ { i } { \bf { w } } - \mathrm { t r } ( { \bf { A } } _ { i } )$ . Suppose that each $q _ { i }$ is $( \tau , V ) – r e g u l a r ~ a n d$ $\mathrm { V a r } ( q _ { i } ) \leq 1$

Let $\psi : \mathbb { R } ^ { k }  \mathbb { R }$ be a $C ^ { 2 }$ function with bounded first and second derivatives, and set

$$
F ( \mathbf { z } , \mathbf { w } ) = \psi ( q _ { 1 } ( \mathbf { z } , \mathbf { w } ) , \ldots , q _ { k } ( \mathbf { z } , \mathbf { w } ) ) .
$$

For $j \in \{ 1 , 2 \}$ , define

$$
D _ { j } ( \psi ) = \operatorname* { m a x } _ { i _ { 1 } , \ldots , i _ { j } \in [ k ] } \| \partial _ { i _ { 1 } } \cdot \cdot \cdot \partial _ { i _ { j } } \psi \| _ { \infty } .
$$

Then for every unit coeficient direction $\mathbf { v } = ( b , L , \mathbf { A } )$ and every polynomial $p _ { ; }$

$$
\left| \underset { \substack { \mathbf z , \mathbf w \sim \mathcal { N } ( 0 , 1 ) } } { \mathbf E } [ p d _ { \mathbf v } F ] - \sum _ { i = 1 } ^ { k } \langle \mathbf v , \mathbf u _ { i } \rangle \underset { \substack { \mathbf z , \mathbf w \sim \mathcal { N } ( 0 , 1 ) } } { \mathbf E } [ p \partial _ { i } \psi ( q _ { 1 } , \dots , q _ { k } ) ] \right| \leq C \tau \big ( k D _ { 1 } ( \psi ) + k ^ { 2 } D _ { 2 } ( \psi ) \big ) \| p \| _ { 2 } ,
$$

where $C > 0$ is a universal constant.

Proof of Lemma $9 . 4 \cdot$ To simplify the notation, write $a ( \mathbf { z } ) = b + L ^ { \top } \mathbf { z } , a _ { i } ( \mathbf { z } ) = b _ { i } + L _ { i } ^ { \top } \mathbf { z } ,$ , and let $g _ { i } ( \mathbf { z } , \mathbf { w } ) = \nabla _ { \mathbf { w } } q _ { i } ( \mathbf { z } , \mathbf { w } ) = a _ { i } ( \mathbf { z } ) + 2 \mathbf { A } _ { i } \mathbf { w }$ . Since $\nabla _ { \mathbf { w } } ^ { 2 } q _ { i } \ = \ 2 \mathbf { A } _ { i } .$ , the chain rule gives $\nabla _ { \mathbf { w } } F =$ $\textstyle \sum _ { i = 1 } ^ { k } \partial _ { i } \psi ( q ) g _ { i }$ and $\begin{array} { r } { \nabla _ { \mathbf { w } } ^ { 2 } F = \sum _ { i = 1 } ^ { k } 2 \partial _ { i } \psi ( q ) \mathbf { A } _ { i } + \sum _ { i , j = 1 } ^ { k } \partial _ { i j } \psi ( q ) g _ { i } g _ { j } ^ { \intercal } } \end{array}$ , where $q = ( q _ { 1 } , \dots , q _ { k } )$ Substituting these expressions into the definition of $d _ { \mathbf { v } } F$ , we obtain

$$
d _ { \mathbf { v } } F = a ( \mathbf { z } ) ^ { \mathsf { T } } \nabla _ { \mathbf { w } } F + \operatorname { t r } ( \mathbf { A } \nabla _ { \mathbf { w } } ^ { 2 } F ) = \sum _ { i = 1 } ^ { k } \partial _ { i } \psi ( q ) \left( a ( \mathbf { z } ) ^ { \top } g _ { i } + 2 \langle \mathbf { A } , \mathbf { A } _ { i } \rangle _ { F } \right) + \sum _ { i , j = 1 } ^ { k } \partial _ { i j } \psi ( q ) g _ { i } ^ { \top } \mathbf { A } g _ { j } .
$$

For $i \in [ k ]$ , define $K _ { i } ( { \bf z } , { \bf w } ) = a ( { \bf z } ) ^ { \top } g _ { i } ( { \bf z } , { \bf w } ) + 2 \langle { \bf A } , { \bf A } _ { i } \rangle _ { F }$ . By independence and Gaussian orthogonality,

$$
\begin{array} { r } { \underset { { \bf z } , { \bf w } \sim \mathcal { N } ( 0 , { \bf I } ) } { \bf E } [ K _ { i } ] = { \bf b } ^ { \top } { \bf b } _ { i } + \langle { \bf L } , { \bf L } _ { i } \rangle _ { F } + 2 \langle { \bf A } , { \bf A } _ { i } \rangle _ { F } = \langle { \bf v } , { \bf u } _ { i } \rangle . } \end{array}
$$

We first show that $\lVert K _ { i } - \left. \mathbf { v } , \mathbf { u } _ { i } \right. \rVert _ { 2 } \leq C \tau .$ . Expanding the centered polynomial gives

$$
K _ { i } - \left. \mathbf { v } , \mathbf { u } _ { i } \right. = \mathbf { b } ^ { \top } \mathbf { L } _ { i } ^ { \top } \mathbf { z } + ( \mathbf { L } ^ { \top } \mathbf { z } ) ^ { \top } \mathbf { b } _ { i } + \mathbf { z } ^ { \top } \mathbf { L } \mathbf { L } _ { i } ^ { \top } \mathbf { z } - \left. \mathbf { L } , \mathbf { L } _ { i } \right. _ { F } + 2 ( \mathbf { b } + \mathbf { L } ^ { \top } \mathbf { z } ) ^ { \top } \mathbf { A } _ { i } \mathbf { w } .
$$

By $( \tau , V )$ -regularity and Fact 3.2, $\| \mathbf { b } _ { i } \| _ { 2 } , \| \mathbf { L } _ { i } \| _ { \mathrm { o p } } , \| \mathbf { A } _ { i } \| _ { \mathrm { o p } } \leq \tau$ . Moreover, since v is a unit coeficient direction, $\| \mathbf { b } \| _ { 2 } ^ { 2 } + \| \mathbf { L } \| _ { F } ^ { 2 } + 2 \| \mathbf { A } \| _ { F } ^ { 2 } = 1$ . Consequently,

$$
\begin{array} { r l } & { \quad \quad \| \mathbf { b } ^ { \top } \mathbf { L } _ { i } ^ { \top } \mathbf { z } \| _ { 2 } = \| \mathbf { L } _ { i } \mathbf { b } \| _ { 2 } \leq \tau , } \\ & { \quad \quad \| ( \mathbf { L } ^ { \top } \mathbf { z } ) ^ { \top } \mathbf { b } _ { i } \| _ { 2 } = \| \mathbf { L } \mathbf { b } _ { i } \| _ { 2 } \leq \tau , } \\ & { \quad \quad \| \mathbf { z } ^ { \top } \mathbf { L } \mathbf { L } _ { i } ^ { \top } \mathbf { z } - \langle \mathbf { L } , \mathbf { L } _ { i } \rangle _ { F } \| _ { 2 } \leq \sqrt { 2 } \| \mathbf { L } \mathbf { L } _ { i } ^ { \top } \| _ { F } \leq \sqrt { 2 } \tau . } \end{array}
$$

For the remaining term, conditioning on z and taking the expectation over w gives

$$
\begin{array} { r } { \| 2 ( \mathbf { b } + \mathbf { L } ^ { \top } { \mathbf z } ) ^ { \top } \mathbf { A } _ { i } \mathbf { w } \| _ { 2 } ^ { 2 } = 4 \mathbf { E } \left[ \| \mathbf { A } _ { i } ( \mathbf { b } + \mathbf { L } ^ { \top } { \mathbf z } ) \| _ { 2 } ^ { 2 } \right] \leq 4 \| \mathbf { A } _ { i } \| _ { \mathrm { o p } } ^ { 2 } \big ( \| \mathbf { b } \| _ { 2 } ^ { 2 } + \| \mathbf { L } \| _ { F } ^ { 2 } \big ) \leq 4 \tau ^ { 2 } . } \end{array}
$$

Combining the four bounds proves $\lVert K _ { i } - \left. \mathbf { v } , \mathbf { u } _ { i } \right. \rVert _ { 2 } \leq C \tau$

Next, we show that for every $i , j \in [ k ] , \| g _ { i } ^ { \top } \mathbf { A } g _ { j } \| _ { 2 } \leq C \tau$ . By the variance bound on $q _ { i }$ and Gaussian orthogonality, $\| \mathbf { b } _ { i } \| _ { 2 } ^ { 2 } + \| \mathbf { L } _ { i } \| _ { F } ^ { 2 } + 2 \| \mathbf { A } _ { i } \| _ { F } ^ { 2 } \leq \bar { 1 }$ . It follows that

$$
\begin{array} { r } { \underset { { \bf z } , { \bf w } \sim \mathcal { N } ( 0 , { \bf I } ) } { \bf E } \| g _ { i } \| _ { 2 } ^ { 2 } = \| { \bf b } _ { i } \| _ { 2 } ^ { 2 } + \| { \bf L } _ { i } \| _ { F } ^ { 2 } + 4 \| { \bf A } _ { i } \| _ { F } ^ { 2 } \leq 2 . } \end{array}
$$

Since $g _ { i }$ is an afine Gaussian vector, Gaussian fourth-moment bounds imply $\begin{array} { r } { \left( \mathbf { E } _ { \mathbf { z } , \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } \| g _ { i } \| _ { 2 } ^ { 4 } \right) ^ { 1 / 4 } \leq } \end{array}$ $C .$ On the other hand,

$$
\begin{array} { r l } & { \quad \underset { \mathbf { z } , \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } { \mathbf { E } } \| \mathbf { A } g _ { j } \| _ { 2 } ^ { 2 } = \| \mathbf { A } \mathbf { b } _ { j } \| _ { 2 } ^ { 2 } + \| \mathbf { A } \mathbf { L } _ { j } ^ { \top } \| _ { F } ^ { 2 } + 4 \| \mathbf { A } \mathbf { A } _ { j } \| _ { F } ^ { 2 } } \\ & { \qquad \leq \| \mathbf { A } \| _ { F } ^ { 2 } \left( \| \mathbf { b } _ { j } \| _ { 2 } ^ { 2 } + \| \mathbf { L } _ { j } \| _ { \mathrm { o p } } ^ { 2 } + 4 \| \mathbf { A } _ { j } \| _ { \mathrm { o p } } ^ { 2 } \right) } \\ & { \qquad \leq C \tau ^ { 2 } . } \end{array}
$$

Applying the same fourth-moment bound gives $\left( \mathbf { E } _ { \mathbf { z } , \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } \| \mathbf { A } g _ { j } \| _ { 2 } ^ { 4 } \right) ^ { 1 / 4 } \leq C \tau$ . Thus, by Cauchy– Schwarz,

$$
\| g _ { i } ^ { \top } \mathbf { A } g _ { j } \| _ { 2 } \leq \left( \mathbf { E } \| g _ { i } \| _ { 2 } ^ { 4 } \right) ^ { 1 / 4 } \left( \mathbf { E } \| \mathbf { A } g _ { j } \| _ { 2 } ^ { 4 } \right) ^ { 1 / 4 } \leq C \tau .
$$

Finally, using the expansion of $d _ { \mathbf { v } } F _ { \mathbf { \delta } }$

$$
\begin{array} { l } { \displaystyle \sum _ { \mathbf { z } , \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p d _ { \mathbf { v } } F ] - \sum _ { i = 1 } ^ { k } \langle \mathbf { v } , \mathbf { u } _ { i } \rangle \displaystyle \sum _ { \mathbf { z } , \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p \partial _ { i } \psi ( q ) ] \Bigg \rvert } \\ { \displaystyle \leq \sum _ { i = 1 } ^ { k } \left. \sum _ { \mathbf { z } , \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } \left[ p \partial _ { i } \psi ( q ) ( K _ { i } - \langle \mathbf { v } , \mathbf { u } _ { i } \rangle ) \right] \right. + \sum _ { i , j = 1 } ^ { k } \left. \sum _ { \mathbf { z } , \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } \left[ p \partial _ { i j } \psi ( q ) g _ { i } ^ { \top } \mathbf { A } g _ { j } \right] \right. } \\ { \displaystyle \leq \| p \| _ { 2 } \left( D _ { 1 } ( \psi ) \displaystyle \sum _ { i = 1 } ^ { k } \| K _ { i } - \langle \mathbf { v } , \mathbf { u } _ { i } \rangle \| _ { 2 } + D _ { 2 } ( \psi ) \displaystyle \sum _ { i , j = 1 } ^ { k } \| g _ { i } ^ { \top } \mathbf { A } g _ { j } \| _ { 2 } \right) } \\ { \displaystyle \leq C \tau \big ( k D _ { 1 } ( \psi ) + k ^ { 2 } D _ { 2 } ( \psi ) \big ) \| p \| _ { 2 } . } \end{array}
$$

This completes the proof.

To complete the proof, we only need to find an operator $T _ { \mathbf { u } }$ such that $\mathbf { E } _ { ( \mathbf { z } , \mathbf { w } ) \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p d _ { \mathbf { u } } f ] =$ $\mathbf { E } _ { ( \mathbf { z } , \mathbf { u } ) \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ f T _ { \mathbf { u } } p ]$ , for every polynomial $p .$ Such an operator is obtained through integration by parts.

Lemma 9.5. Let $\mathbf { v } = ( b , L , \mathbf { A } )$ be a residual coeficient direction and let $p$ be a polynomial. Define

$$
T _ { \mathbf { v } } p = p N _ { \mathbf { v } } ( \mathbf { z } , \mathbf { w } ) - ( \mathbf { b } + \mathbf { L } ^ { \top } \mathbf { z } + 2 \mathbf { A } \mathbf { w } ) ^ { \top } \nabla _ { \mathbf { w } } p + \mathrm { t r } ( \mathbf { A } \nabla _ { \mathbf { w } } ^ { 2 } p ) ,
$$

where

$$
N _ { \mathbf { v } } ( \mathbf { z } , \mathbf { w } ) = ( \mathbf { b } + \mathbf { L } ^ { \top } \mathbf { z } ) ^ { \top } \mathbf { w } + \mathbf { w } ^ { \top } \mathbf { A } \mathbf { w } - \operatorname { t r } ( \mathbf { A } ) .
$$

Then, for every suficiently smooth function $F _ { ; }$

$$
\underset { { \bf z } , { \bf w } \sim \mathcal { N } ( 0 , { \bf I } ) } { \bf E } [ p d _ { \bf v } F ] = \underset { { \bf z } , { \bf w } \sim \mathcal { N } ( 0 , { \bf I } ) } { \bf E } [ F T _ { \bf v } p ] .
$$

The identity also holds for bounded measurable F when $d _ { \mathbf { v } } F$ is interpreted weakly.

Proof of Lemma 9.5. Recall that

$$
d _ { \mathbf { v } } F = ( \mathbf { b } + \mathbf { L } ^ { \top } \mathbf { z } ) ^ { \top } \nabla _ { \mathbf { w } } F + \mathrm { t r } ( \mathbf { A } \nabla _ { \mathbf { w } } ^ { 2 } F ) .
$$

We apply Gaussian integration by parts with respect to $\mathbf { w } _ { \mathrm { : } }$ , keeping z fixed. For the linear part, since $b + L ^ { \top } { \mathbf z }$ does not depend on $\mathbf { w } ,$ we have

$$
\mathbf { \frac { E } { w } } [ p ( \mathbf { b } + \mathbf { L } ^ { \top } \mathbf { z } ) ^ { \top } \nabla _ { \mathbf { w } } F ] = \mathbf { \frac { E } { w } } \left[ F \left( p ( \mathbf { b } + \mathbf { L } ^ { \top } \mathbf { z } ) ^ { \top } \mathbf { w } - ( \mathbf { b } + \mathbf { L } ^ { \top } \mathbf { z } ) ^ { \top } \nabla _ { \mathbf { w } } p \right) \right] .
$$

For the quadratic part, using the symmetry of A and applying Gaussian integration by parts once gives

$$
\begin{array} { r } { \frac { \mathbf { E } } { \mathbf { w } } [ p \mathrm { t r } ( \mathbf { A } \nabla _ { \mathbf { w } } ^ { 2 } F ) ] = \frac { \mathbf { E } } { \mathbf { w } } [ ( p \mathbf { A } \mathbf { w } - \mathbf { A } \nabla _ { \mathbf { w } } p ) ^ { \top } \nabla _ { \mathbf { w } } F ] . } \end{array}
$$

Applying Gaussian integration by parts a second time, we obtain

$$
\mathbf { { \frac { E } { w } } } [ ( p \mathbf { A } \mathbf { w } - \mathbf { A } \nabla _ { \mathbf { w } } p ) ^ { \top } \nabla _ { \mathbf { w } } F ] = \mathbf { { \frac { E } { w } } } \left[ F \big ( \mathbf { w } ^ { \top } ( p \mathbf { A } \mathbf { w } - \mathbf { A } \nabla _ { \mathbf { w } } p ) - \mathrm { { d i v } } _ { \mathbf { w } } ( p \mathbf { A } \mathbf { w } - \mathbf { A } \nabla _ { \mathbf { w } } p ) \big ) \right] .
$$

Notice that $\operatorname { d i v } _ { \mathbf { w } } ( p \mathbf { A } \mathbf { w } ) = ( \mathbf { A } \mathbf { w } ) ^ { \top } \nabla _ { \mathbf { w } } p + p \mathrm { t r } ( \mathbf { A } ) , \operatorname { d i v } _ { \mathbf { w } } ( \mathbf { A } \nabla _ { \mathbf { w } } p ) = \mathrm { t r } ( \mathbf { A } \nabla _ { \mathbf { w } } ^ { 2 } p )$ . Therefore,

$$
\mathbf { w } ^ { \top } ( p \mathbf { A } \mathbf { w } - \mathbf { A } \nabla _ { \mathbf { w } } p ) - \mathrm { d i v } _ { \mathbf { w } } ( p \mathbf { A } \mathbf { w } - \mathbf { A } \nabla _ { \mathbf { w } } p ) = p ( \mathbf { w } ^ { \top } \mathbf { A } \mathbf { w } - \operatorname { t r } ( \mathbf { A } ) ) - 2 ( \mathbf { A } \mathbf { w } ) ^ { \top } \nabla _ { \mathbf { w } } p + \operatorname { t r } ( \mathbf { A } \nabla _ { \mathbf { w } } ^ { 2 } p ) .
$$

This gives

$$
\begin{array} { r } { \frac { \mathbf { E } } { \mathbf { w } } [ p \mathrm { t r } ( \mathbf { A } \nabla _ { \mathbf { w } } ^ { 2 } F ) ] = \frac { \mathbf { E } } { \mathbf { w } } \left[ F \big ( p ( \mathbf { w } ^ { \top } \mathbf { A } \mathbf { w } - \mathrm { t r } ( \mathbf { A } ) ) - 2 ( \mathbf { A } \mathbf { w } ) ^ { \top } \nabla _ { \mathbf { w } } p + \mathrm { t r } ( \mathbf { A } \nabla _ { \mathbf { w } } ^ { 2 } p ) \big ) \right] . } \end{array}
$$

Combining the linear and quadratic parts, and taking expectation over z, we obtain

$$
\begin{array} { r } { \underline { { \mathbf { E } } } _ { , \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ p d _ { \mathbf { v } } F ] = \underline { { \mathbf { E } } } _ { , \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } \left[ F \big ( p N _ { \mathbf { v } } ( \mathbf { z } , \mathbf { w } ) - ( \mathbf { b } + \mathbf { L } ^ { \top } \mathbf { z } + 2 \mathbf { A } \mathbf { w } ) ^ { \top } \nabla _ { \mathbf { w } } p + \mathrm { t r } ( \mathbf { A } \nabla _ { \mathbf { w } } ^ { 2 } p ) \big ) \right] = \underline { { \mathbf { E } } } _ { \mathbf { z } , \mathbf { w } \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ F T \mathbf { v } p ] , } \end{array}
$$

as required.

For bounded measurable F, the same identity defines the weak action of $d _ { \mathbf { v } } F$ on polynomial test functions, and the conclusion follows by smooth approximation. □

## 9.2 Proof of Theorem 9.1

We now combine the dimension-reduction guarantees with the known-lifted-subspace learner of Theorem 8.1. The regular intersection supplied by Theorem 4.1 and the approximating intersection supplied by Lemma 9.3 to prove Theorem 9.1.

Proof of Theorem 9.1. Let $C _ { 8 }$ be the constant in the approximation requirement of Theorem 8.1, and put

$$
L = k ^ { 2 } \log ^ { 3 } ( 2 k ) \log ^ { 2 } ( 1 / \varepsilon ) , \qquad \rho = 2 ^ { - C _ { 8 } L } .
$$

Increase $C _ { \mathrm { t r } , k }$ to be at least one. Choose

$$
a = \left( \frac { \rho } { 2 C _ { \mathrm { t r } , k } } \right) ^ { 2 0 } , \quad \quad \quad \quad \quad \alpha = \frac { a } { 2 k } , \quad \quad \quad \quad \beta = \beta _ { k } ( \alpha ) ,
$$

$$
\tau = \operatorname* { m i n } \left\{ \frac { a } { 2 } , \left( \frac { \beta } { 8 A _ { k } } \right) ^ { 1 3 } \right\} , \qquad \nu _ { \mathrm { e s t } } = \frac { \beta } { 1 6 } , \qquad \eta _ { 0 } = \operatorname* { m i n } \left\{ \frac { \rho } { 2 } , \left( \frac { \beta } { 1 6 K _ { k } } \right) ^ { 2 } \right\} .
$$

These choices imply

$$
\eta _ { 0 } + C _ { \mathrm { t r } , k } ( k \alpha + \tau ) ^ { 1 / 2 0 } \leq \rho \quad \mathrm { a n d } \quad A _ { k } \tau ^ { 1 / 1 3 } + 2 \nu _ { \mathrm { e s t } } + 2 K _ { k } \sqrt { \eta _ { 0 } } \leq \frac { 3 \beta } { 8 } .\tag{70}
$$

All logarithms of the inverse parameters are $O _ { k } ( L )$

Finding the physical subspace. Set

$$
\theta = \alpha _ { k } ( \eta _ { 0 } , \tau ) = \frac { \tau ^ { 2 } \eta _ { 0 } ^ { 2 } } { C _ { 4 } k ^ { 1 2 } } , \qquad \ell = 2 k + 1 ,
$$

where $C _ { 4 }$ is the constant in Theorem 4.1. Let $2 \gamma$ be a valid certificate lower bound from Theorem 5.1 at randomized-error level $\theta ;$ we may take $\gamma = \overset { 1 } { \underset { 2 } { \boldsymbol { \mathbf { \rho } } } } ( \theta / ( C _ { 5 } k ) ) ^ { 2 ^ { C _ { 5 } k } }$ for a suficiently large universal $C _ { 5 }$ Apply Theorem 6.2 to the original labels with degree budget ℓ and correlation threshold $\gamma .$ With failure probability at most $\delta / 3$ , it returns a subspace V such that

$$
n : = \dim V \leq O ( \ell / \gamma ^ { 2 } ) , \qquad \mathrm { R a n d } _ { f ^ { * } } ( { \mathbf v } ) < \theta \quad \mathrm { f o r ~ e v e r y ~ u n i t ~ } { \mathbf v } \perp V .
$$

The randomized-error conclusion follows by contradiction from Theorem 5.1. Theorem 4.1 then supplies an intersection f of at most k normalized $( \tau , V )$ -regular quadratics with $\mathbf { P r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ f ^ { * } ( \mathbf { x } ) \neq$ $f ( \mathbf { x } ) ] \leq \eta _ { 0 }$ . The required confidence in the moment-estimation step is obtained by applying entrywise median-of-means amplification to the moment estimates used in the proof of Lemma 6.3.

Recovering the residual polynomial subspace. Condition on the first-stage samples and use fresh examples to estimate $B _ { y }$ . Let R = dim $\begin{array} { r } { S \le d ( d + 3 ) / 2 } \end{array}$ and $J = { \binom { d + 2 k } { 2 k } }$ . For orthonormal bases $( e _ { a } ) _ { a = 1 } ^ { R }$ and $( H _ { \gamma } ) _ { | \gamma | \leq 2 k }$ , estimate

$$
B _ { y } ( e _ { a } , H _ { \gamma } ) = \underset { ( \mathbf { x } , y ) \sim D } { \mathbf { E } } [ y T _ { e _ { a } } H _ { \gamma } ( \mathbf { x } ) ] .
$$

Each sample contribution has second moment at most $K _ { k } ^ { 2 }$ . Entrywise median-of-means estimates with accuracy $\nu _ { \mathrm { e s t } } / \sqrt { R J }$ therefore give $\| \widehat { B } - B _ { y } \| _ { \mathrm { o p } } \leq \nu _ { \mathrm { e s t } }$ with failure probability at most $\delta / 3 .$ , using

$$
O \bigg ( K _ { k } ^ { 2 } R J \nu _ { \mathrm { e s t } } ^ { - 2 } \log \frac { 2 R J } { \delta } \bigg )
$$

examples. The operator-norm conclusion follows from the Frobenius-norm bound. If $R = 0$ , this estimation step is omitted. Constructing the matrix and taking its singular value decomposition uses $( d + 1 ) ^ { O ( k ) } \mathrm { p o l y } _ { k } ( 1 / \nu _ { \mathrm { e s t } } , \log ( 2 / \delta ) )$ arithmetic operations.

By (70), Lemma 9.3 applies to its leading left singular subspace $\bar { V }$ , of dimension at most k. It supplies an intersection $g$ with defining polynomials in $\mathcal { H } = \mathcal { P } _ { < 2 } ( V ) \oplus \{ N _ { \mathbf { u } } : \mathbf { u } \in \bar { V } \}$ and

$$
\Pr _ { ( \mathbf { x } , y ) \sim D } [ g ( \mathbf { x } ) \neq y ] \leq \eta _ { 0 } + C _ { \mathrm { t r } , k } ( k \alpha + \tau ) ^ { 1 / 2 0 } \leq \rho .
$$

Let $S \subseteq \mathbb { R } ^ { N }$ be the coeficient subspace satisfying

$$
\mathcal { H } = \mathbb { R } \oplus \{ u \cdot \Phi _ { 2 } ( \cdot ) : u \in S \} .
$$

Constants are excluded from S and are treated separately in Theorem 8.1. Since the residual polynomials are orthogonal to $\mathcal { P } _ { \leq 2 } ( V )$

$$
r : = \dim S \leq { \binom { n + 2 } { 2 } } - 1 + k = { \frac { n ( n + 3 ) } { 2 } } + k .\tag{71}
$$

Learning in the recovered subspace. Use a fresh batch and apply Theorem 8.1 to $S$ and the relabeled examples $( \mathbf { x } , 2 y - 1 )$ , with accuracy ε and failure probability $\delta / 3$ . The function $2 g - 1$ satisfies its approximation hypothesis because $\rho = 2 ^ { - C _ { 8 } L }$ . If its output is $h : \mathbb { R } ^ { d }  \{ - 1 , 1 \}$ , return ${ \widehat { f } } = ( h + 1 ) / 2$ . Conditionally on the successful preceding stages, its error against the original labels is at most ε. A union bound gives overall success probability at least $1 - \delta$ . The degree statement follows from Theorem 8.1, since substituting quadratic features at most doubles the degree.

Complexity. The parameter choices and Theorem 5.1 give

$$
\log ( 1 / \gamma ) = O _ { k } ( L ) , \qquad \log ( n + 2 ) = O _ { k } ( L ) , \qquad \log ( r + 2 ) = O _ { k } ( L ) .
$$

The two dimension-reduction stages have sample and arithmetic cost at most

$$
( d + 1 ) ^ { O ( k ) } 2 ^ { O _ { k } ( L ) } \log ^ { 2 } ( 2 / \delta ) .
$$

Here polynomial factors in log $( d + 2 )$ are absorbed into $( d + 1 ) ^ { O ( k ) }$

By Theorem 8.1, the hinge-loss stage has sample and arithmetic cost at most

$$
( d + 1 ) ^ { O ( 1 ) } ( r + 2 ) ^ { O ( k ^ { 2 } \log ^ { 2 } ( 2 k ) \log ( 1 / \varepsilon ) ) } 2 ^ { O ( k ^ { 2 } \log ^ { 3 } ( 2 k ) \log ^ { 2 } ( 1 / \varepsilon ) ) } \log ^ { 2 } ( 2 / \delta ) .
$$

Since log $( r + 2 ) = O _ { k } ( L )$ and $L = O _ { k } ( \log ^ { 2 } ( 1 / \varepsilon ) )$ , this is at most

$$
( d + 1 ) ^ { O ( 1 ) } 2 ^ { O _ { k } ( L \log ( 1 / \varepsilon ) ) } \log ^ { 2 } ( 2 / \delta ) \leq ( d + 1 ) ^ { O ( 1 ) } 2 ^ { O _ { k } ( \log ^ { 3 } ( 1 / \varepsilon ) ) } \log ^ { 2 } ( 2 / \delta ) .
$$

Combining the bounds for the two dimension-reduction stages and hinge-loss minimization proves the claimed dependence on d and ε. □

## 10 Statistical Query Lower Bound

In this section, we prove a statistical query (SQ) lower bound for learning intersections of degree-2 PTFs. We first recall the SQ model. Throughout this section, we use the equivalent {±1}-label convention.

The SQ model was introduced by Kearns [Kea98] for learning Boolean functions in the PAC setting [Val84]. It provides access to approximate expectations of bounded functions through an oracle. The model has also been studied in unsupervised learning [Fel16], and many familiar techniques admit SQ implementations, including spectral algorithms, methods based on moments and tensors, and local search such as EM [FGR<sup>+</sup>17, FGV17].

Definition 10.1 (Statistical Query Model). Let D be a distribution over $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ . For a tolerance parameter $\tau > 0$ , the oracle $\mathrm { S T A T } _ { D } ( \tau )$ accepts a query function $\phi : \mathbb { R } ^ { d } \times \{ \pm 1 \}  [ - 1 , 1 ]$ and returns a value a satisfying $| a - \mathbf { E } _ { ( \mathbf { x } , y ) \sim D } [ \phi ( \mathbf { x } , y ) ] | \leq \tau$ . An SQ algorithm accesses D through these oracle calls and may choose each query and its tolerance adaptively based on previous responses. Its guarantee must hold for every sequence of responses satisfying the tolerance bounds.

The main theorem of this section is the following.

Theorem 10.2. Let $k \in \mathbb { Z } _ { + }$ and let $d \in \mathbb { Z } _ { + }$ be suficiently large depending on k. Any SQ algorithm that learns intersections of k degree-2 PTFs with Gaussian marginals (as defined in Definition 1.2) to any constant error smaller than $1 / 2$ must either (i) use a query with tolerance at most $d ^ { - \Omega ( k ) }$ , or (ii) make at least $2 ^ { d ^ { \Omega ( 1 ) } }$ queries.

The main technical fact of the section is the following. It states that there exists a function $g : \mathbb { R }  \{ \pm 1 \}$ that is discontinuous at at most 2k points and satisfies $\mathbf { E } _ { t \sim \mathcal { N } ( 0 , 1 ) } [ g ( t ) p ( t ) ] = 0$ for every polynomial p of degree at most $2 k - 1$

Fact 10.3. For every $k \in \mathbb { Z } _ { + }$ , there exists a function $g : \mathbb { R }  \{ \pm 1 \}$ that is discontinuous at at most 2k points and satisfies $\mathbf { E } _ { t \sim \mathcal { N } ( 0 , 1 ) } [ g ( t ) t ^ { m } ] = 0$ for all $m \in \{ 0 , \ldots , 2 k - 1 \}$

```latex
Algorithm 6 LiftedSubspaceLearning
1: Input: Sample access from $D ,$ a subspace $V \subseteq \mathbb { R } ^ { d }$ satisfying the condition of Section $^ { 9 , }$ accuracy
ϵ and failure probability $\delta$
2: Output: $\mathrm { A }$ hypothesis $\hat { f } : \mathbb { R } ^ { d }  \{ 0 , 1 \}$
3: Let $m , W$ be the degree and norm bounds from Lemma 8.2 with accuracy $\epsilon / 8 .$ Set $t = m / 2$ and
$\eta = ( \epsilon / ( 2 0 W ) ) ^ { 2 } .$ . Choose $\xi$ as required by the subspace recovery guarantee for approximation
error $\eta .$
4: Write $\begin{array} { r } { \mathbf { x } = \mathbf { z } + \mathbf { w } , } \end{array}$ where $\mathbf { z } = \mathrm { p r o j } _ { V } \mathbf { x }$ and $\mathbf { w } = \mathrm { p r o j } _ { V ^ { \perp } } \mathbf { x } .$ . Let $s$ be the space of residual degree-2
polynomials $N _ { \mathbf { v } } ( \mathbf { z } , \mathbf { w } )$ defined in Section 9.
5: Let $R _ { 1 } , \dots , R _ { M }$ be an orthonormal basis of $\boldsymbol { \mathcal { S } }$ under the Gaussian $L _ { 2 }$ inner product.
6: Draw $N = d ^ { O ( k ) } \mathrm { p o l y } ( 1 / \xi , \log ( 1 / \delta ) )$ independent examples $\{ ( \mathbf { x } ^ { ( i ) } , y ^ { ( i ) } ) \} _ { i = 1 } ^ { N }$ from $D .$
7: Construct a matrix $\hat { B } \in \mathbb { R } ^ { M \times \dim ( \mathcal { P } _ { \leq 2 k } ( \mathbb { R } ^ { d } ) ) }$ such that, for every $a \in [ M ]$ and multi-index $\gamma$ with
$| \gamma | \leq 2 k ,$
$\hat { B } _ { a , \gamma } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } y ^ { ( i ) } T _ { R _ { a } } H _ { \gamma } ( \mathbf { x } ^ { ( i ) } ) ,$
where $T _ { R _ { a } }$ is the operator in Lemma 9.5.
8: Let $\mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { r }$ be the top r left singular vectors of ${ \hat { B } } ,$ , where $r = \operatorname* { m i n } \{ k , M \}$ . Set
$U = \mathrm { s p a n } \left\{ \sum _ { a = 1 } ^ { M } ( { \mathbf { u } } _ { j } ) _ { a } R _ { a } : j \in [ r ] \right\} .$
9: Set $\mathcal { H } = \mathcal { P } _ { \leq 2 } ( V ) \oplus U .$ , and let $1 , \phi _ { 1 } , \ldots , \phi _ { s - 1 }$ be an orthonormal basis of $\mathcal { H } .$
10: Construct the lifted polynomial feature map
$\Psi _ { t } ( \mathbf { x } ) = \left( \prod _ { j = 1 } ^ { s - 1 } \phi _ { j } ( \mathbf { x } ) ^ { \gamma _ { j } } \right) _ { | \gamma | \leq t } .$
11: Using fresh samples from $D ,$ apply the bounded-norm hinge-loss minimization method of
Section $^ 8$ to the features $\Psi _ { t } ( \mathbf { x } )$ and labels $2 y - 1$ , with degree budget t and the corresponding
coeficient-norm bound $R _ { \epsilon , s }$
12: Return $\hat { f } ( \mathbf { x } ) = \mathbf { 1 } \{ \langle \hat { \theta } , \Psi _ { t } ( \mathbf { x } ) \rangle \geq 0 \}$ , where $\hat { \theta }$ is the coeficient vector returned by hinge-loss
minimization.
```

Proof of Fact 10.3. We construct $g$ by taking a best $L _ { 1 }$ approximation $p ^ { * }$ to $t ^ { 2 k }$ among polynomials of degree at most $2 k - 1$ , and setting

$$
g ( t ) = \mathrm { s i g n } \big ( p ^ { * } ( t ) - t ^ { 2 k } \big ) ,
$$

where sign(0) = 1. The idea is that optimality of $p ^ { * }$ forces $g$ to be orthogonal to every polynomial of degree at most $2 k - 1$ , while the degree of $p ^ { * } ( t ) - t ^ { 2 k }$ bounds the number of discontinuities. We justify this construction by checking existence of the minimizer, the moment identities, and the discontinuity bound.

First, let $\mathcal { P } _ { 2 k - 1 }$ be the space of real polynomials of degree at most $2 k - 1$ , and define

$$
F ( p ) : = \underset { t \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { E } } \big [ | p ( t ) - t ^ { 2 k } | \big ] .
$$

The functional F is continuous and satisfies

$$
F ( p ) \geq \| p \| _ { 1 } - \underset { t \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { E } } [ | t | ^ { 2 k } ] ,
$$

where $\left\| \cdot \right\| _ { 1 }$ denotes the Gaussian $L _ { \mathrm { 1 } } \mathrm { - n o r m }$ . Since this is a norm on the finite-dimensional space $\mathcal { P } _ { 2 k - 1 }$ , the sublevel sets of $F$ are compact. Hence, F attains its minimum at some $p ^ { \ast } \in \mathcal { P } _ { 2 k - }$ <sub>−1</sub>.

Next, fix m $\in \{ 0 , \ldots , 2 k - 1 \}$ and consider the perturbation $p ^ { * } ( t ) + s t ^ { m }$ . The residual $p ^ { * } ( t ) - t ^ { 2 k }$ is a polynomial of degree exactly 2k, so it is nonzero almost surely under the Gaussian distribution. Moreover, for $s \neq 0$

$$
\left| \frac { | p ^ { * } ( t ) + s t ^ { m } - t ^ { 2 k } | - | p ^ { * } ( t ) - t ^ { 2 k } | } { s } \right| \leq | t | ^ { m } .
$$

Since $\mathbf { E } _ { t \sim \mathcal { N } ( 0 , 1 ) } [ | t | ^ { m } ] < \infty$ , dominated convergence justifies diferentiating under the expectation. Optimality of $p ^ { * }$ therefore gives

$$
0 = \frac { d } { d s } \underset { t \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { E } } \big [ | p ^ { * } ( t ) + s t ^ { m } - t ^ { 2 k } | \big ] \biggr \rvert _ { s = 0 } = \underset { t \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { E } } [ g ( t ) t ^ { m } ] .
$$

Finally, g can be discontinuous only at a real root of $p ^ { * } ( t ) - t ^ { 2 k }$ . This polynomial has degree 2k, so there are at most 2k such points. Thus, g satisfies both required properties, proving the fact.

Given Fact 10.3, we are ready to prove Theorem 10.2. Let $g$ be the function constructed in the proof of Fact 10.3. For each unit vector $\mathbf { v } \in \mathbb { S } ^ { d - 1 }$ , define $D _ { \mathbf { v } }$ to be the joint distribution of $\left( \mathbf { x } , y \right)$ obtained by sampling $\mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ and setting $y = g ( \mathbf { v } \cdot \mathbf { x } )$ . As we verify below, the labels under $D _ { \mathbf { v } }$ , after conversion to $\{ 0 , 1 \}$ , are consistent with an intersection of k degree-2 PTFs. Let $D _ { 0 }$ be the joint distribution of $\left( \mathbf { x } , y \right)$ , where $\mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ and $y \sim \operatorname { U n i f } ( \{ \pm 1 \} )$ ) independently.

Using the framework introduced in [DKS17], we can show that any $\mathrm { S Q }$ algorithm that distinguish between $D _ { \mathbf { v } }$ with unknown v and $D _ { 0 }$ must either (i) use a query with tolerance at most $d ^ { - \Omega ( k ) }$ , or (ii) make at least $2 ^ { d ^ { \Omega ( 1 ) } }$ queries. Notice that an algorithm for learning intersections of k degree-2 PTFs can be used to distinguish between $D _ { \mathbf { v } }$ and $D _ { 0 }$ , as when input distribution is $D _ { \mathbf { v } }$ , the algorithm must with high probability return a hypothesis with constant error better than $1 / 2$ , while if the input distribution is $D _ { 0 }$ , no such hypothesis could exists.

For convenience, we invoke the SQ lower bound of [DKPZ21], which is based on the argument outlined above. We recall the relevant definition and proposition below.

Definition 10.4 ([DKPZ21], Definition 18). Let $g : \mathbb { R }  [ - 1 , 1 ]$ be a measurable function. For each $\mathbf { v } \in \mathbb { S } ^ { d - 1 }$ , let $D _ { \mathbf { v } }$ be the distribution of $\left( \mathbf { x } , y \right)$ over $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ satisfying $\mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ and $\mathbf { E } [ y \mid \mathbf { x } ] = g ( \mathbf { v } \cdot \mathbf { x } )$ . Define ${ \mathcal { D } } _ { q } : = \{ D _ { \mathbf { v } } : \mathbf { v } \in \mathbb { S } ^ { d - 1 } \}$ and $D _ { 0 } : = \mathcal { N } ( 0 , \mathbf { I } _ { d } ) \times \operatorname { U n i f } ( \{ \pm 1 \} )$ . We denote by $B ( \mathcal { D } _ { g } , D _ { 0 } )$ the problem of distinguishing $\boldsymbol { D } \in \mathcal { D } _ { g }$ from $D = D _ { 0 }$ , given $S Q$ access to an unknown distribution D promised to satisfy one of these conditions.

Fact 10.5 ([DKPZ21], Proposition 19). Let $\ell \in \mathbb { Z } _ { + }$ , and let d be suficiently large depending on ℓ. Suppose that $g : \mathbb { R }  [ - 1 , 1 ]$ is measurable and satisfies $\mathbf { E } _ { t \sim \mathcal { N } ( 0 , 1 ) } [ g ( t ) t ^ { j } ] = 0$ for all $j \in \{ 0 , \ldots , \ell - 1 \}$ Any SQ algorithm that solves $B ( \mathcal { D } _ { g } , D _ { 0 } )$ with probability at least $2 / 3$ must either $( i )$ use a query with tolerance at most d $^ { . \Omega ( \ell ) } , o r \ ( i i )$ make at least $2 ^ { d ^ { \Omega ( 1 ) } }$ queries.

We now prove Theorem 10.2.

Proof of Theorem 10.2. The proof has two parts. First, we use Facts 10.3 and 10.5 to obtain a hard distinguishing problem of the form in Definition 10.4. Second, we show that an SQ learner would solve this problem by learning from the input distribution and checking the error of its output hypothesis.

For the first part, let $g$ be the function constructed in the proof of Fact 10.3. Applying Fact 10.5 with $\ell = 2 k$ , distinguishing an unknown ${ \cal D } _ { \mathbf { v } } \in { \cal D } _ { g }$ from $D _ { 0 }$ requires either a query of tolerance at most $d ^ { - c k }$ or at least $N : = 2 ^ { d ^ { c _ { 0 } } }$ queries, for universal constants $c , c _ { 0 } > 0$

Moreover, each $D _ { \mathbf { v } }$ is realizable by the target class. Indeed, $g$ is negative on both tails, and its positive set is a union of at most k bounded intervals. A union of $r$ bounded intervals can be expressed using r quadratic inequalities: one restricting to their convex hull and $r - 1$ excluding the intervening gaps. Substituting $t = { \bf v } \cdot { \bf x }$ gives the required intersection of quadratic PTFs, up to Gaussian-null endpoints and conversion of the labels. We use the equivalent {±1}-label convention below.

For the second part, fix a constant $\gamma \in ( 0 , 1 / 2 )$ . Suppose for contradiction that an $\mathrm { S Q }$ learner achieves error at most $1 / 2 - \gamma$ with probability at least $2 / 3$ , using fewer than $N / 2$ queries, all of tolerance greater than $d ^ { - c k }$ . Since γ is a positive constant, for every $k \geq 1$ and suficiently large d depending on k and $\gamma ,$ we have $\gamma / 4 > d ^ { - c k }$

Given an input distribution D, run this learner, enforcing its claimed complexity bounds, and use one additional $\mathrm { S Q }$ query of tolerance $\gamma / 4$ to estimate the error of its output hypothesis h. If $D = D _ { \mathbf { v } }$ then with probability at least $2 / 3$ , the error of h is at most $1 / 2 - \gamma _ { \mathrm { { ; } } }$ , so the estimate is at most $1 / 2 - 3 \gamma / 4$ . If $D = D _ { 0 }$ , every hypothesis has error exactly $1 / 2$ , so the estimate is at least $1 / 2 - \gamma / 4$ Comparing the estimate with $1 / 2 - \gamma / 2$ therefore distinguishes the two cases with probability at least $2 / 3$

For suficiently large d, this distinguishing algorithm uses fewer than N queries, all of tolerance greater than $d ^ { - c k }$ , contradicting Fact 10.5. □

## References

[ABAM23] E. Abbe, E. Boix-Adserà, and T. Misiakiewicz. SGD learning on neural networks: Leap complexity and saddle-to-saddle dynamics. In Proceedings of the Thirty-Sixth Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pages 2552–2623, 2023. URL https://proceedings.mlr.press/v195/abbe23a.html.

[APS26] J. Alman, S. Patel, and R. A. Servedio. Learning functions of halfspaces. In Proceedings of the 58th Annual ACM Symposium on Theory of Computing, STOC ’26. Association for Computing Machinery, 2026. doi:10.1145/3798129.3800866.

[Bau90] E. B. Baum. A polynomial time algorithm that learns two hidden unit nets. Neural Computation, 2(4):510–522, 1990. doi:10.1162/neco.1990.2.4.510.

[BH25] J. Bruna and D. Hsu. Survey on algorithms for multi-index models. Statistical Science, 40(3):378–391, 2025. doi:10.1214/25-STS993.

[BK97] A. L. Blum and R. Kannan. Learning an intersection of a constant number of halfspaces over a uniform distribution. Journal of Computer and System Sciences, 54(2):371–380, 1997.

[BR88] A. Blum and R. L. Rivest. Training a 3-node neural network is NP-complete. In D. S. Touretzky, editor, Advances in Neural Information Processing Systems 1, pages 494–501, 1988.

[CDG<sup>+</sup>23] S. Chen, Z. Dou, S. Goel, A. Klivans, and R. Meka. Learning narrow one-hidden-layer ReLU networks. In The Thirty Sixth Annual Conference on Learning Theory, pages 5580–5614. PMLR, 2023.

[CKM21] S. Chen, A. R. Klivans, and R. Meka. Learning deep ReLU networks is fixed-parameter tractable. In 62nd IEEE Annual Symposium on Foundations of Computer Science, FOCS 2021, Denver, CO, USA, February 7-10, 2022, pages 696–707. IEEE, 2021. doi:10.1109/FOCS52979.2021.00073.

[CN24] S. Chen and S. Narayanan. A faster and simpler algorithm for learning shallow networks. In S. Agrawal and A. Roth, editors, The Thirty Seventh Annual Conference on Learning Theory, volume 247 of Proceedings of Machine Learning Research, pages 981–994. PMLR, 2024. URL https://proceedings.mlr.press/v247/chen24b.html.

[CW01] A. Carbery and J. Wright. Distributional and L<sup>q</sup> norm inequalities for polynomials over convex bodies in R<sup>n</sup>. Mathematical Research Letters, 8(3):233–248, 2001.

[Dan16] A. Daniely. Complexity theoretic limitations on learning halfspaces. In Proceedings of the forty-eighth annual ACM symposium on Theory of Computing, pages 105–117, 2016.

[DDS14] A. De, I. Diakonikolas, and R. A. Servedio. Deterministic approximate counting for juntas of degree-2 polynomial threshold functions. In 2014 IEEE 29th Conference on Computational Complexity (CCC), pages 229–240. IEEE, 2014.

[DHK<sup>+</sup>10] I. Diakonikolas, P. Harsha, A. Klivans, R. Meka, P. Raghavendra, R. A. Servedio, and L.-Y. Tan. Bounding the average sensitivity and noise sensitivity of polynomial threshold functions. In Proceedings of the forty-second ACM symposium on Theory of computing, pages 533–542, 2010.

[DIKR25] I. Diakonikolas, G. Iakovidis, D. M. Kane, and L. Ren. Algorithms and SQ lower bounds for robustly learning real-valued multi-index models. In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 5debeab774db7ca61ea15f1117f7fa33-Abstract-Conference.html.

[DIKZ25] I. Diakonikolas, G. Iakovidis, D. M. Kane, and N. Zarifis. Robust learning of multi-index models via iterative subspace approximation. In 66th IEEE Annual Symposium on Foundations of Computer Science, FOCS 2025, Sydney, Australia, December 14-17, 2025, pages 2231–2239. IEEE, 2025.

[DIM26] I. Diakonikolas, G. Iakovidis, and M. Ma. Polynomial-time robust multiclass linear classification under Gaussian marginals. In Advances in Neural Information Processing Systems, 2026. URL https://arxiv.org/abs/2605.21428. To appear.

[DK24] I. Diakonikolas and D. M. Kane. Eficiently learning one-hidden-layer ReLU networks via Schur polynomials. In The Thirty Seventh Annual Conference on Learning Theory, pages 1364–1378. PMLR, 2024.

[DKKZ20] I. Diakonikolas, D. M. Kane, V. Kontonis, and N. Zarifis. Algorithms and SQ lower bounds for PAC learning one-hidden-layer ReLU networks. In Conference on Learning Theory, pages 1514–1539. PMLR, 2020.

[DKM26] I. Diakonikolas, D. M. Kane, and M. Ma. Robust learning of multi-index models under logconcave distributions via weak partial learners. In Proceedings of the 67th Annual IEEE Symposium on Foundations of Computer Science (FOCS), 2026.

[DKN10] I. Diakonikolas, D. M. Kane, and J. Nelson. Bounded independence fools degree-2 threshold functions. In 2010 IEEE 51st Annual Symposium on Foundations of Computer Science, pages 11–20. IEEE, 2010.

[DKPZ21] I. Diakonikolas, D. M. Kane, T. Pittas, and N. Zarifis. The optimality of polynomial regression for agnostic learning under Gaussian marginals in the SQ model. In M. Belkin and S. Kpotufe, editors, Proceedings of the 34th Conference on Learning Theory, volume 134 of Proceedings of Machine Learning Research, pages 1552–1584, 2021. URL https://proceedings.mlr.press/v134/diakonikolas21c.html.

[DKS17] I. Diakonikolas, D. M. Kane, and A. Stewart. Statistical query lower bounds for robust estimation of high-dimensional Gaussians and Gaussian mixtures. In 58th IEEE Annual Symposium on Foundations of Computer Science (FOCS), pages 73–84, 2017. doi:10.1109/FOCS.2017.16.

[DKS18] I. Diakonikolas, D. Kane, and A. Stewart. Learning geometric concepts with nasty noise. In Proceedings of the 50<sup>th</sup> Annual ACM SIGACT Symposium on Theory of Computing, STOC 2018, pages 1061–1073, 2018.

[DLB25] A. Damian, J. D. Lee, and J. Bruna. The generative leap: Tight sample complexity for eficiently learning Gaussian multi-index models. In Advances in Neural Information Processing Systems, volume 38, 2025. doi:10.52202/085713-0953.

[DMRT25a] I. Diakonikolas, M. Ma, L. Ren, and C. Tzamos. Learning intersections of two margin halfspaces under factorizable distributions. In Proceedings of the Thirty Eighth Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 1472–1530, 2025. URL https://proceedings.mlr.press/v291/diakonikolas25a. html.

[DMRT25b] I. Diakonikolas, M. Ma, L. Ren, and C. Tzamos. Statistical query hardness of multiclass linear classification with random classification noise. In International Conference on Machine Learning, pages 13693–13712. PMLR, 2025.

[DS14] A. De and R. A. Servedio. Eficient deterministic approximate counting for low-degree polynomial threshold functions. In D. B. Shmoys, editor, Symposium on Theory of Computing, STOC 2014, New York, NY, USA, May 31 - June 03, 2014, pages 832–841. ACM, 2014. doi:10.1145/2591796.2591800.

[DSTW10] I. Diakonikolas, R. A. Servedio, L.-Y. Tan, and A. Wan. A regularity lemma, and low-weight approximators, for low-degree polynomial threshold functions. In 2010 IEEE 25th Annual Conference on Computational Complexity, pages 211–222. IEEE, 2010.

[Duc18] J. C. Duchi. Introductory lectures on stochastic optimization. In M. W. Mahoney, J. C. Duchi, and A. C. Gilbert, editors, The Mathematics of Data, volume 25 of IAS/Park City Mathematics Series, pages 99–185. 2018.

[Fel16] V. Feldman. Statistical query learning. In Encyclopedia of Algorithms, pages 2090–2095. 2016.

[FGR<sup>+</sup>17] V. Feldman, E. Grigorescu, L. Reyzin, S. Vempala, and Y. Xiao. Statistical algorithms and a lower bound for detecting planted cliques. J. ACM, 64(2):8:1–8:37, 2017.

[FGV17] V. Feldman, C. Guzman, and S. S. Vempala. Statistical query algorithms for mean vector estimation and stochastic convex optimization. In P. N. Klein, editor, Proceedings of the Twenty-Eighth Annual ACM-SIAM Symposium on Discrete Algorithms, SODA 2017, pages 1265–1277. SIAM, 2017.

[FWNL25] H. Fu, Z. Wang, E. Nichani, and J. D. Lee. Learning hierarchical polynomials of multiple nonlinear features. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 599221d7ebf6b3403190f38a3f282a1c-Abstract-Conference.html.

[HL13] C. J. Hillar and L.-H. Lim. Most tensor problems are NP-hard. Journal of the ACM, 60(6):45:1–45:39, 2013. doi:10.1145/2512329.

[Kan11] D. M. Kane. The Gaussian surface area and noise sensitivity of degree-d polynomial threshold functions. Computational Complexity, 20(2):389–412, 2011.

[Kea98] M. J. Kearns. Eficient noise-tolerant learning from statistical queries. Journal of the ACM, 45(6):983–1006, 1998.

[KKSK11] S. Kakade, V. Kanade, O. Shamir, and A. Kalai. Eficient learning of generalized linear and single index models with isotonic regression. Advances in Neural Information Processing Systems, 24, 2011.

[KOS04] A. Klivans, R. O’Donnell, and R. Servedio. Learning intersections and thresholds of halfspaces. Journal of Computer & System Sciences, 68(4):808–840, 2004.

[KOS08] A. Klivans, R. O’Donnell, and R. Servedio. Learning geometric concepts via Gaussian surface area. In Proc. 49th IEEE Symposium on Foundations of Computer Science (FOCS), pages 541–550, Philadelphia, Pennsylvania, 2008.

[KP96] S. Kwek and L. Pitt. PAC learning intersections of halfspaces with membership queries. In Proceedings of the ninth annual conference on Computational learning theory, pages 244–254, 1996.

[KS04] A. Klivans and R. Servedio. Learning intersections of halfspaces with a margin. In Proceedings of the 17<sup>th</sup> Annual Conference on Learning Theory, pages 348–362, 2004.

[KS07] A. R. Klivans and A. A. Sherstov. Unconditional lower bounds for learning intersections of halfspaces. Machine Learning, 69(2–3):97–114, 2007.

[KS08] S. Khot and R. Saket. On hardness of learning intersection of two halfspaces. In Proceedings of the fortieth annual ACM symposium on Theory of computing, pages 345–354, 2008.

[KS09] A. R. Klivans and A. A. Sherstov. Cryptographic hardness for learning intersections of halfspaces. Journal of Computer and System Sciences, 75(1):2–12, 2009.

[MT94] W. Maass and G. Turan. How fast can a threshold gate learn? In S. Hanson, G. Drastal, and R. Rivest, editors, Computational Learning Theory and Natural Learning Systems, pages 381–414. MIT Press, 1994.

[MZ10] R. Meka and D. Zuckerman. Pseudorandom generators for polynomial threshold functions. In STOC, pages 427–436, 2010.

[NDL23] E. Nichani, A. Damian, and J. D. Lee. Provable guarantees for nonlinear feature learning in three-layer neural networks. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, 2023.

[Nic26] E. Nichani. A tutorial on feature learning in shallow neural networks. https: //eshaannichani.com/blog/multi-index-models/index.html, September 27 2026. Section 5: Open Directions.

[O’D14] R. O’Donnell. Analysis of Boolean Functions. Cambridge University Press, 2014.

[PV26] S. Patel and S. Vempala. Tight bounds for learning polyhedra with a margin. In Proceedings of the IEEE Symposium on Foundations of Computer Science (FOCS). IEEE, 2026. URL https://arxiv.org/abs/2604.14614.

[Ros58] F. Rosenblatt. The perceptron: A probabilistic model for information storage and organization in the brain. Psychological Review, 65(6):386–408, 1958. doi:10.1037/h0042519.

[She10] A. A. Sherstov. Optimal bounds for sign-representing the intersection of two halfspaces by polynomials. In Proceedings of the Forty-second ACM Symposium on Theory of Computing, pages 523–532, 2010.

[She13] A. A. Sherstov. The intersection of two halfspaces has high threshold degree. SIAM Journal on Computing, 42(6):2329–2374, 2013.

[Tie24] S. Tiegel. Improved hardness results for learning intersections of halfspaces. In The Thirty Seventh Annual Conference on Learning Theory, pages 4764–4786. PMLR, 2024.

[Val84] L. Valiant. A theory of the learnable. Communications of the ACM, 27(11):1134–1142, 1984.

[Vem10a] S. Vempala. Learning convex concepts from Gaussian distributions with PCA. In 51<sup>st</sup> Annual IEEE Symposium on Foundations of Computer Science, FOCS, pages 124–130, 2010.

[Vem10b] S. Vempala. A random-sampling-based algorithm for learning intersections of halfspaces. J. ACM, 57(6):32:1–32:14, 2010.

[WNL24] Z. Wang, E. Nichani, and J. D. Lee. Learning hierarchical polynomials with three-layer neural networks. In The Twelfth International Conference on Learning Representations, ICLR. OpenReview.net, 2024.