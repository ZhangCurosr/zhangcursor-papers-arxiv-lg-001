# Gaussian Partitioning for Reliable Learning of Multi-Index Models

Ilias Diakonikolas<sup>∗</sup> University of Wisconsin-Madison ilias@cs.wisc.edu

Daniel M. Kane<sup>‡</sup> University of California, San Diego dakane@cs.ucsd.edu

Giannis Iakovidis<sup>†</sup> University of Wisconsin-Madison iakovidis@wisc.edu

Mingchen Ma<sup>§</sup> University of Wisconsin-Madison mingchen@cs.wisc.edu

## Abstract

We develop Gaussian partitioning, a general algorithmic primitive for distributions that are smooth with respect to a Gaussian. Given samples from such a distribution, our algorithm partitions the space into polyhedral regions so that almost all of the probability mass lies in thin strips: regions where the coordinates along a small number of mutually orthogonal directions lie in short intervals, while the remaining directions are unrestricted. Additionally, within each such region, the conditional distribution approximately matches low-degree Gaussian moments in the unrestricted directions. Our construction builds on the logconcave partitioning method of Diakonikolas, Kane, and Ma (FOCS 2026).

We apply this primitive to develop new algorithms for learning Boolean multi-index models. Our results are as follows:

• Agnostic and Testable Learning. Under input marginals smooth with respect to the standard Gaussian, we give learners with error (1 + c)opt + ϵ for broad classes of multi-index models, including functions of halfspaces, low-dimensional convex concepts, and low-dimensional polynomial threshold functions. For Boolean functions of k halfspaces, the dimension dependence is $d ^ { \mathrm { p o l y } ( k / c ) }$ , improving the previous exponential dependence on $k / c$ in the exponent. For functions of k halfspaces, we also obtain a testable agnostic learner with the same error guarantee and dimension dependence $d ^ { \mathrm { p o l y } ( k / c ) }$

• Learning from Positive Examples with Imperfect Unlabeled Data. In the PUI model, the learner receives positive examples from an unknown input distribution that is smooth with respect to the standard Gaussian reference. We give general algorithms and Statistical Query (SQ) lower bounds that qualitatively characterize the complexity of PUI learning for a broad family of multi-index models through Gaussian moment matching. For fixed smoothness and positive-mass parameters, we obtain a PUI learner for intersections of k halfspaces with dimension dependence $d ^ { \bar { O } ( \log ( k / \epsilon ) ) }$ and matching SQ lower bounds, improving the previous $d ^ { \mathrm { p o l y } ( 1 / \epsilon ) }$ upper bound even for a single halfspace.

• Agnostic Reliable Learning. We give a simple general reduction from agnostic reliable learning to PUI learning. Under input marginals smooth with respect to the standard Gaussian, this yields negative reliable learners for intersections of k halfspaces with dimension dependence $d ^ { O ( \log ( k / \epsilon ) ) }$ for fixed smoothness parameters.

Finally, for PQ learning and Testable learning with Distribution Shift (TDS), we give qualitatively optimal SQ lower bounds of $d ^ { \Omega ( 1 / \epsilon ^ { 2 } ) }$ , even for intersections of two halfspaces with Gaussian training data and arbitrary test distributions. These bounds rule out, within the SQ framework, the logarithmic dependence on $1 / \epsilon$ in the dimension exponent achieved by our PUI and one-sided reliable learners.

## Contents

1 Introduction 1   
1.1 Gaussian Partitioning 1   
1.2 Algorithmic Applications in Learning Theory 2   
1.2.1 Semi-Agnostic Learning 3   
1.2.2 Learning from Positive Examples with Imperfect Unlabeled Data 4   
1.2.3 Agnostic Reliable Learning 5   
1.3 Detailed Overview of Techniques 7   
1.3.1 From Logconcave Partitioning to Thin Gaussian Cells 7   
1.3.2 Constructing the Partition 8   
1.3.3 Learning Theory Applications via Reductions to Gaussian Partitioning 10   
1.3.4 SQ Lower Bounds via Moment Matching 12   
1.4 Related Work 12   
2 Preliminaries and Notations 13   
3 G i S i i i i Q i l i l i 14   
3.1 Partitioning Algorithm 16   
3.2 Proof of the Partitioning Theorem 26   
3.3 Eficient Moment Discrepancy Finding Algorithm 29   
4 Eficient Agnostic Learning via Gaussian Partitioning 35   
4.1 Eficient Weak Learning Algorithm for Smooth Distributions 37   
4.2 Eficient Agnostic Learning under Gaussian-Smooth Distributions 41   
5 Learning with Positive and Imperfect Unlabeled Examples 44   
5.1 Faster Algorithm for PUI Learning with Gaussian Reference 45   
5.2 SQ Lower Bound for PUI Learning of MIMs with Gaussian Reference 51   
5.3 PUI Learning Intersections of Halfspaces and Beyond 53   
6 Agnostic Reliable Learning 60   
7 SQ Lower Bound for Realizable TDS and PQ Learning under the Gaussian Distribution 63   
A Omitted Preliminaries 69   
B Omitted Facts and Technical Proofs 70   
C Omitted Details from Section 3 73   
C.1 Confidence Amplification of Discrepancy Finder 73   
D Omitted Details from Section 4 76   
D.1 Applications to Well-Studied Distribution Families 76   
E Multiplicative Approximation SQ Lower Bound 77   
F Testable Learning of MIMs from Gaussian Partitioning 78   
F.1 Testable Weak Learner . 80   
F.2 Testable Learner for MIMs under Smooth Distributions 82

## 1 Introduction

A broad range of supervised learning algorithms use properties of Gaussian inputs, such as low-degree polynomial approximations and moment bounds. In several learning settings, however, the observed examples need not follow a Gaussian distribution. For example, boosting changes the weights of examples during its execution, and a learner given only positive examples observes the original distribution conditioned on an unknown event. Thus, even when the original input distribution is Gaussian, the distribution observed by the learner need not be.

Motivated by such applications in learning theory, we develop a general algorithmic primitive fo partitioning the input space so that Gaussian approximation bounds become applicable within the resulting regions. Given a Gaussian reference distribution P and samples from a distribution $Q ,$ we assume that events of suficiently small probability under P also have small probability under $Q .$ . This smoothness assumption, defined in Definition 1.1, allows $Q$ to be otherwise very diferent from P. Our algorithm partitions $\mathbb { R } ^ { d }$ into polyhedral regions so that, on cells containing almost all of the Q-mass, the conditional distribution approximately matches low-degree Gaussian moments in the unrestricted directions. We call this primitive Gaussian partitioning.

Our main result, Theorem 1.2, gives a partition whose number of cells is independent of the ambient dimension d. Cells containing almost all of the Q-mass are thin strips: their projections onto the span of their constraint normals have small diameter, while the orthogonal coordinates remain unrestricted. The dimension of this span is independent of d. Within each such cell R, the conditional distribution Q | R approximately matches low-degree Gaussian moments in the unrestricted directions. The small diameter guarantee is crucial for our learning theory applications. For a target Boolean function of bounded Gaussian surface area, it allows us to fix the constrained coordinates with small total classification error. We can therefore analyze the target within a cell using Gaussian moment and approximation bounds in the unrestricted directions.

Our construction builds on the logconcave partitioning method of [DKM26]. The earlier method compares moments with conditional logconcave distributions; our thin cells allow the learning step to use an untruncated Gaussian reference in the unrestricted directions. This distinction is important because Gaussian approximation bounds can require substantially fewer moments.

Our Gaussian partitioning primitive gives new algorithms for learning Boolean multi-index models in several settings: agnostic learning, testable agnostic learning, learning from positive examples with imperfect unlabeled data (PUI learning), and one-sided reliable learning. In Subsection 1.1, we formally describe our partitioning result and in Subsection 1.2 we present its learning theory applications.

## 1.1 Gaussian Partitioning

Smoothness with respect to the Gaussian distribution is the distributional assumption underlying our learning algorithms. Intuitively, events that are suficiently rare under the reference distribution must also have small probability under the input distribution. We use the following definition from [LMZ26], specialized to distributions on $\mathbb { R } ^ { d }$

Definition 1.1 $( ( q , \sigma )$ -Smoothness [LMZ26]). Let $P , Q$ be probability distributions on $\mathbb { R } ^ { d }$ , let $q \geq 1$ , and let $\sigma \in ( 0 , 1 ]$ . We say that $Q$ is $( q , \sigma )$ )-smooth with respect to P $i f ,$ for every measurable event $E \subseteq \mathbb { R } ^ { d }$ $Q ( E ) \leq ( 1 / \bar { \sigma } ) P ( E ) ^ { 1 / q }$

For $q = 1$ , this condition is equivalent to obtaining $Q$ from P by rejection sampling with acceptance probability at least $\sigma .$ We also say that $Q$ has weight at least σ with respect to P. Allowing $q > 1$ permits an unbounded density ratio while still controlling the probability of events that are unlikely under $P$ For example, $\mathcal { N } ( \pmb { \mu } , \mathbf { I } _ { d } )$ is $( 2 , e ^ { - \| \pmb { \mu } \| _ { 2 } ^ { 2 } / 2 } )$ -smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ ; see Lemma D.1. More generally, bounded $\chi ^ { 2 }$ divergence or bounded Rényi divergence of order greater than one implies smoothness; see [LMZ26, Appendix D.4].

To formally state our main result, we need some additional terminology. For a subspace $V \subseteq \mathbb { R } ^ { d }$ , let proj denote the orthogonal projection onto V . For a polyhedral cell $R ,$ let $V _ { R }$ be the span of the normal vectors of its defining inequalities. We measure the diameter of a cell in its constrained directions by diam $( R ) : = \operatorname* { s u p } _ { \mathbf { u } , \mathbf { v } \in \operatorname { p r o j } _ { V _ { R } } ( R ) } \| \mathbf { u } - \mathbf { v } \| _ { 2 }$ . When this quantity is small, R is a thin strip along $V _ { R } ^ { \perp }$ . For a standard Gaussian reference $P ,$ , the projection of $P \mid R$ onto $V _ { R } ^ { \perp }$ is still standard Gaussian: conditioning on R restricts only the $V _ { R } { \mathrm { - } }$ -coordinates.

We say that $P ~ \mid ~ R$ and ${ \cal Q } \ | \ { \cal R } ,$ , the corresponding conditional distributions on $R ,$ match their first k directional moments on $V _ { R } ^ { \perp }$ with slack $\Delta \ { \mathrm { i f } } .$ , for every unit vector $\mathbf { v } \in V _ { R } ^ { \perp }$ and every $i \in [ k ]$ ， $| \mathbf { E _ { x \sim Q | R } } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { i } ] - \mathbf { E _ { x \sim P | R } } [ ( \mathbf { v } \cdot \grave { \mathbf { x } ) ^ { i } } ] | \leq \Delta$ . This is the notion of approximate moment matching used throughout the paper; see Definition 2.2. With this setup, we now state our main algorithmic result.

Theorem 1.2 (Gaussian Partitioning; Informal Version of Theorem 3.1). Let $P = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ , and let $Q$ be a distribution on $\mathbb { R } ^ { d }$ that is $( q , \sigma )$ -smooth with respect to $P$ . For $k \in \mathbb { Z } _ { + }$ and $\eta , \delta , \Delta \in ( 0 , 1 / 2 )$ , there is an algorithm using at most $d ^ { k } B$ samples from $Q$ and $d ^ { O ( k ) } B$ time, where $\begin{array} { r } { B = \exp \left( \frac { ( q k ) ^ { O ( k ) } } { \Delta ^ { O ( 1 ) } } \log ^ { O ( k ) } \frac { 2 } { \sigma \eta } \log \frac { 2 } { \delta } \right) } \end{array}$ that, with probability at least $2 / 3$ , returns a polyhedral partition U $o f \mathbb { R } ^ { d }$ and a subcollection $\mathcal { G } \subseteq \mathcal { U }$ such that $| \mathcal { U } | \le B$ and $Q ( \bigcup _ { R \in { \mathcal { G } } } R ) \geq 1 - \eta$ . Moreover, every $R \in { \mathcal { G } }$ satisfies:

(i) $Q ( R ) \geq B ^ { - 1 }$

(ii) diam(R) ≤ δ.

(iii) $P \mid R$ and $Q \mid R$ match their first k directional moments on $V _ { R } ^ { \perp }$ with slack $\Delta$

We call the cells in $\mathcal { G }$ certified. The parameter $\delta$ bounds the constrained diameter of every certified cell, as stated in item (ii). For fixed $q , k , \Delta$ , the complexity of our algorithm is polynomial in d and quasi-polynomial in $1 / ( \sigma \eta \delta )$ . We emphasize that the underlying construction uses only samples from $Q$ and is independent of any target function or labels.

We next explain why the partitioning theorem is useful for learning. For concept classes of bounded Gaussian surface area, a suficiently small diameter bound allows us to fix the $V _ { R } { \mathrm { - } }$ -coordinates in each certified cell with small total error. The learning problem in each cell then involves only the unrestricted coordinates, where the conditional distribution approximately matches Gaussian moments.

Comparison with [DKM26]. Recent work [DKM26] developed a related partitioning subroutine as part of a general algorithm for agnostically learning Multi-Index Models (MIMs) under logconcave marginals. Given a logconcave distribution P and a distribution $Q$ obtained by rejection sampling from $P _ { \mathrm { : } }$ their algorithm partitions the space so that $P \mid R$ and $Q \mid R$ approximately match low-degree moments on cells containing almost all of the Q-mass. The comparison is made after an afine normalization of each cell. This is often enough to construct a weak learner over these conditional distributions, and these can be combined by a boosting algorithm.

The reference distribution in each cell is therefore $P \ | \ R _ { \mathrm { : } }$ , which is logconcave but not necessarily Gaussian, even when P itself is Gaussian. Their analysis uses approximation bounds for general logconcave distributions, which can require (exponentially) higher degree than the corresponding Gaussian bounds. The thinness of our cells allows us to fix the constrained coordinates with small total error and apply Gaussian approximation and moment bounds in the unrestricted directions.

## 1.2 Algorithmic Applications in Learning Theory

Our learning applications concern Boolean multi-index models, namely functions that depend only on the projection onto an unknown low-dimensional subspace. Formally, we have the following definition.

Definition 1.3 (Multi-Index Model). A function $f : \mathbb { R } ^ { d }  \{ \pm 1 \}$ is called a K-MIM if there exists a subspace $U \subseteq \mathbb { R } ^ { d }$ of dimension at most K such that $f ( \mathbf { x } ) = f ( \mathrm { p r o j } _ { U } \mathbf { x } )$ for every $\mathbf { x } \in \mathbb { R } ^ { d }$ . We call such a subspace $\textit { U a }$ relevant subspace of $f$ . Equivalently, $f ( \mathbf { x } ) = g ( \mathbf { W } \mathbf { x } )$ for a link function $g : \mathbb { R } ^ { K }  \{ \pm 1 \}$ and a matrix $\mathbf { W } \in \mathbb { R } ^ { K \times d }$

The relevant subspace is unknown to the learner. The link function may also be unknown, as for arbitrary Boolean functions of halfspaces; in special cases, such as intersections of halfspaces, its form is specified by the concept class. In the settings of interest, K is much smaller than the ambient dimension d. Examples include functions of a small number of halfspaces, low-dimensional convex concepts, and low-dimensional polynomial threshold functions.

A growing line of work studies algorithms for MIMs under diferent distributional and noise assumptions. We discuss this work in Subsection 1.4. We now present the learning problems addressed by our Gaussian partitioning primitive.

## 1.2.1 Semi-Agnostic Learning

We start with the standard notion of distribution-specific agnostic learning with approximate error guarantees<sup>⋄</sup>.

Definition 1.4 (Semi-Agnostic Learning under Smooth Marginals). Let $\mathcal { F }$ be a class of Boolean functions on $\mathbb { R } ^ { d } .$ , and let D be a distribution on $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ whose x-marginal $D _ { \textbf { x } } i s \ ( q , \sigma )$ -smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ . No assumptions are made on the labels. For a hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$ , define $\mathrm { e r r } _ { D } ( h ) : =$ $\mathbf { P r } _ { ( \mathbf { x } , y ) \sim D } [ h ( \mathbf { x } ) \neq y ]$ and ${ \mathrm { o p t } } _ { D } ( { \mathcal { F } } ) : = { \mathrm { i n f } } _ { f \in { \mathcal { F } } } { \mathrm { e r r } } _ { D } ( f )$ Given i.i.d. samples from D and parameters $c , \epsilon \in ( 0 , 1 )$ , the goal of a $( 1 + c )$ -approximate agnostic learner is to output a hypothesis h satisfying err $\begin{array} { r } { \mathbf { \dot { \phi } } _ { D } ( h ) \leq ( 1 + c ) \mathrm { o p t } _ { D } ( \mathcal { F } ) + \epsilon } \end{array}$ with probability at least $2 / 3$

Our general algorithmic result for this model applies to MIM classes that are closed under afine restrictions, admit Gaussian sandwiching polynomials, and have uniformly bounded Gaussian surface area.

Definition 1.5 ((ℓ, K)-Well-Behaved MIMs). Let $\ell : ( 0 , 1 ] \to \mathbb { Z } _ { + }$ be non-increasing, and let $\mathcal { F }$ be a class of Boolean K-MIMs on $\mathbb { R } ^ { d }$ . We say that F is (ℓ, K)-well-behaved if the following hold.

(i) Closure under afine restrictions. For every $f \in { \mathcal { F } }$ , every subspace $W \subseteq \mathbb { R } ^ { d }$ , and every $\mathbf { b } \in W ^ { \perp }$ , the function $f _ { W , { \bf b } } ( { \bf x } ) : = f ( { \bf b } + \mathrm { p r o j } _ { W } { \bf x } )$ belongs to $\mathcal { F }$

(ii) Gaussian sandwiching. For every $f \in { \mathcal { F } }$ and every $a \in ( 0 , 1 )$ , there are polynomials $p ^ { l } , p ^ { u }$ of degree at most $\ell ( a )$ such that $p ^ { l } ( { \bf x } ) \le f ( { \bf x } ) \le p ^ { u } ( { \bf x } )$ for every $\mathbf { x } \in \mathbb { R } ^ { d }$ , and $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } ) } [ p ^ { u } ( \mathbf { x } ) - p ^ { l } ( \mathbf { x } ) ] \leq a$

(iii) Bounded Gaussian surface area. The class satisfies $\Gamma ( { \mathcal F } ) : = \operatorname* { s u p } _ { f \in { \mathcal F } } \Gamma ( f ) < \infty$ , where $\Gamma ( f )$ denotes the Gaussian surface area of $f ;$ see Definition A.1.

The three conditions in Definition 1.5 serve diferent purposes. Closure under afine restrictions keeps the target in the class after the constrained coordinates are fixed. Sandwiching polynomials transfer Gaussian approximation guarantees to a distribution with approximately matching moments. The surface-area bound controls the total classification error introduced by fixing the constrained coordinates.

Our main agnostic learning result is the following:

Theorem 1.6 (Semi-Agnostic Learning of MIMs; Informal Version of Theorem 4.2). Let F be an $( \ell , K )$ well-behaved MIM class and D be a distribution on $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ such that $D _ { \textbf { x } } i s \ ( q , \sigma )$ -smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ . For $c , \epsilon \in ( 0 , 1 / 2 )$ , set $\ell _ { c } = \ell ( c _ { 0 } \cdot c )$ for a suficiently small universal constant $c _ { 0 } > 0$ There is an algorithm that draws $n = d ^ { O ( \ell _ { c } ) } \exp \Bigl ( ( q K \ell _ { c } ) ^ { O ( \ell _ { c } ) } \log ^ { O ( \ell _ { c } ) } \frac { 2 } { c \sigma \epsilon } \log ( 2 + \Gamma ( \mathcal { F } ) ) \Bigr )$ samples from $D ,$ runs in poly(n) time, and, with probability at least $2 / 3$ , outputs a hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$ satisfying err $\begin{array} { r } { \mathbf { \dot { \phi } } _ { D } ( h ) \leq ( 1 + c ) \mathrm { o p t } _ { D } ( \mathcal { F } ) + \epsilon } \end{array}$

The main point is that the degree determining the exponent of d depends on the approximation factor $c ,$ but not on the excess error ϵ. For fixed class parameters, q, and $c ,$ the algorithm has polynomial dependence on d and quasipolynomial dependence on $1 / ( \sigma \epsilon )$ . The error guarantee is the same as in [DKM26]; the improvement comes from using Gaussian sandwiching degrees in place of logconcave ones.

<table><tr><td>Concept class</td><td>Dimension dependence</td></tr><tr><td>Intersections of k halfspaces</td><td> $\widetilde { d ^ { O ( k ^ { 3 } / c ^ { 4 } ) } }$ </td></tr><tr><td>Arbitrary Boolean functions of k halfspaces</td><td> $d ^ { \widetilde { O } ( k ^ { 5 } / c ^ { 4 } ) }$ </td></tr><tr><td>Convex concepts of intrinsic dimension at most K</td><td> $d ^ { \widetilde { O } ( K ^ { 5 } / c ^ { 4 } ) }$ </td></tr><tr><td>Degree-at-most-r PTFs of intrinsic dimension at most K</td><td> $d ^ { \widetilde { O } ( r ^ { 6 } K ^ { 5 } / c ^ { 4 } ) }$ </td></tr></table>

Table 1: Dimension dependence for $( 1 + c )$ -approximate agnostic learning under marginals that are $( q , \sigma )$ -smooth with respect to the standard Gaussian. For fixed class parameters, $q ,$ and $^ { c , }$ the remaining factor is quasipolynomial in $1 / ( \sigma \epsilon )$

Combining Theorem 1.6 with known Gaussian sandwiching bounds [KKM13, KSV26] gives Table 1. The table records only the dependence on $d ;$ the theorem gives the remaining factors. In particular, for arbitrary Boolean functions of k halfspaces, the dimension dependence improves from $d \mathrm { e x p } ( O ( ( k / c ) ^ { 4 } ) )$ in [DKM26] to $d ^ { \mathrm { p o l y } ( k / c ) }$

We also prove an $\mathrm { S Q }$ lower bound for agnostic learning with error $( 1 + c ) { \mathrm { o p t } } + \epsilon .$ , even under standard Gaussian marginals; see Section E. When $\epsilon \leq c ,$ , this bound shows that $\mathrm { S Q }$ algorithms require either exponentially many queries or tolerance at most $d ^ { - \Omega ( \ell ( c ) ) }$ , where $\ell ( c )$ denotes the Gaussian $L _ { 1 }$ approximation degree at accuracy $O ( c )$ . This lower bound is tight up to polynomial factors in the dimension exponent for the classes considered in our applications. In fact, for halfspaces, $L _ { 1 }$ sandwiching and $L _ { 1 }$ approximation have the same degree $\Theta ( 1 / c ^ { 2 } )$ , yielding tightness up to constant factors in the exponent.

Extension to Testable Agnostic Learning. For functions of k halfspaces, our guarantees also extend to testable agnostic learning. Given samples from an arbitrary labeled distribution, the learner either rejects or outputs a hypothesis with error at most $( 1 + c ) { \mathrm { o p t } } + \epsilon .$ with probability at least $2 / 3$ . Whenever the input marginal is $( q , \sigma )$ -smooth with respect to the standard Gaussian, the learner accepts and returns a hypothesis satisfying this guarantee with probability at least $2 / 3$ . The dimension dependence is $d ^ { \widetilde { O } ( k ^ { 5 } / c ^ { 4 } ) }$ for arbitrary functions of k halfspaces and $d ^ { \widetilde { O } ( k ^ { 3 } / c ^ { 4 } ) }$ for intersections; see Corollaries F.7 and F.8.

## 1.2.2 Learning from Positive Examples with Imperfect Unlabeled Data

As our second learning-theoretic application, we give new algorithms for learning from positive examples with imperfect unlabeled data (PUI learning). For intersections of halfspaces and low-dimensional convex concepts, our algorithms have logarithmic dependence on the inverse accuracy in the dimension exponent. These results follow from a qualitative characterization of PUI learning complexity in terms of Gaussian moment matching, via an algorithmic upper bound and an SQ lower bound.

Definition 1.7 (PUI Learning [LMZ26]). Let F be a class of Boolean functions on $\mathbb { R } ^ { d }$ , let D be a reference distribution, and let $D ^ { * }$ be an unknown distribution that is $( q , \sigma )$ -smooth with respect to D. For an unknown target $f \in { \mathcal { F } } .$ , the learner receives the smoothness parameters $q , \sigma _ { ; }$ , sample access to D and $D _ { + } ^ { * } : = D ^ { * } \mid \{ f = 1 \}$ , and a lower bound $p _ { + } > 0$ on $D ^ { * } ( f = 1 )$ . Given $\epsilon \in ( 0 , 1 )$ , the goal is to output $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$ satisfying $\mathbf { P r } _ { \mathbf { x } \sim D ^ { * } } [ h ( \mathbf { x } ) \neq f ( \mathbf { x } ) ] \leq \epsilon$ with probability at least $2 / 3$

We consider the standard Gaussian reference $D = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ . For a class ${ \mathcal { F } } ,$ we define the Gaussian PUI learning exponent $m ^ { * } ( \epsilon , \Delta , \sigma , q )$ to be the least positive integer m such that, whenever the positive conditional distribution $D _ { + } ^ { * }$ matches the first m Gaussian directional moments to error $\Delta ,$ we have $D ^ { * } ( f = - 1 ) < \epsilon$ . This implication must hold for every target $f \in { \mathcal { F } }$ and every distribution $D ^ { * }$ that is $( q , \sigma )$ -smooth with respect to D and satisfies $D ^ { * } ( f = 1 ) > 0$ . Here, moment matching to additive error $\Delta$ means that $| \mathbf { E } _ { \mathbf { x } \sim D _ { + } ^ { * } } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { i } ] - \mathbf { E } _ { z \sim \mathcal { N } ( 0 , 1 ) } [ z ^ { i } ] | \leq \Delta$ for every unit vector $\mathbf { v } \in \mathbb { R } ^ { d }$ and every $1 \leq i \leq m$ . If no such degree exists, we set $m ^ { * } = \infty ;$ see Definition 5.2. Thus, matching suficiently many moments of the positive examples certifies that predicting positive has small error. Gaussian partitioning allows us to apply this criterion within individual cells.

<table><tr><td>Concept class</td><td>Dimension dependence</td></tr><tr><td>Intersections of k halfspaces</td><td> $\overline { { d ^ { O ( \log ( 2 k ) + q \log ( 2 q / ( \sigma \epsilon p _ { + } ) ) ) } } }$ </td></tr><tr><td>Convex concepts of intrinsic dimension K</td><td> $d ^ { O ( K + q \log ( 2 q / ( \sigma \epsilon p _ { + } ) ) ) }$ </td></tr></table>

Table 2: Dimension dependence of our PUI learners with a standard Gaussian reference. For each row, the sample complexity is $n = d ^ { m } \exp ( m ^ { O ( m ) } )$ , with m a suficiently large constant multiple of the quantity inside $O ( \cdot )$ in the displayed exponent. The running time is poly(n).

Our general complexity result for PUI learning is informally stated below (see Theorem 5.3 and Theorem 5.4 for detailed formal statements).

Theorem 1.8 (PUI Learning; Informal). Let F be a class of Boolean functions on $\mathbb { R } ^ { d }$ , and consider PUI learning with reference $D = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ and $( q , \sigma )$ -smooth input distributions, where $q \geq 1$ and $\sigma \in ( 0 , 1 ]$

(i) Upper Bound. Suppose F is closed under afine restrictions and $\Gamma ( \mathcal { F } ) < \infty$ . For $\epsilon , \Delta \ \in$ $( 0 , 1 / 2 )$ and a given lower bound $p _ { + } > 0$ , let $\begin{array} { r } { m = m ^ { * } ( c _ { 1 } \epsilon , \Delta , \frac { c _ { 2 } \epsilon \sigma p _ { + } } { a } , 2 q ) < \infty } \end{array}$ , where $c _ { 1 } , c _ { 2 } > 0$ are suficiently small universal constants. Then PUI learning to error ϵ is possible using $n =$ $\begin{array} { r } { d ^ { m } \exp \left( \frac { ( q m ) ^ { O ( m ) } \log ^ { O ( m ) } \left( 2 / \left( \sigma \epsilon p _ { + } \right) \right) \log \left( 2 + \Gamma ( \mathcal { F } ) \right) } { \Delta ^ { O ( 1 ) } } \right) } \end{array}$ samples and poly(n) time.

(ii) SQ Lower Bound. Suppose F contains all rotations of a K-MIM f admitting a $( q , \sigma )$ -smooth input distribution with positive mass $p _ { + } > 0$ and negative mass at least $\rho > 0$ , whose positive conditional distribution matches Gaussian moments up to degree $r \geq 2$ . Then, PUI learning to error $p _ { + } ( \sigma \rho ) ^ { q } / 2$ requires either $2 ^ { d ^ { \Omega ( 1 ) } }$ SQ queries or tolerance at most $d ^ { - \Omega ( r ) }$

Our algorithmic result does not require the concepts to be MIMs. For the SQ lower bound, it sufices that the class contain all rotations of a suitable low-dimensional target. The characterization is qualitative: the upper and lower bounds use diferent accuracy and smoothness parameters, and the lower bound requires exact moment matching; see Remark 5.5.

For fixed smoothness parameters and a fixed positive-mass lower bound, the previous algorithms [LMZ26] had dimension dependence $d ^ { \mathrm { p o l y } ( 1 / \epsilon ) }$ , even for a single halfspace. Our bounds reduce the dependence on inverse accuracy in this exponent to logarithmic.

Specifically, for intersections of halfspaces we obtain qualitatively matching upper and SQ lower bounds.

Theorem 1.9 (PUI Learning Intersections of Halfspaces; Informal Versions of Theorems 5.12 and 5.14).   
For intersections of k halfspaces with a standard Gaussian reference, the following hold.

(i) For fixed smoothness parameters q, σ and a fixed lower bound $p _ { + } > 0$ , there is a PUI learner under input distributions that are $( q , \sigma )$ -smooth with respect to the standard Gaussian achieving error ϵ with probability at least $2 / 3$ , with complexity $d ^ { O ( \log ( k / \epsilon ) ) }$ , suppressing factors independent of d.

(ii) For suitable constant smoothness parameters and a constant positive-mass lower bound, any SQ learner achieving error ϵ requires either $2 ^ { d ^ { \Omega ( 1 ) } }$ queries or tolerance at most $d ^ { - \Omega ( \log ( k / \epsilon ) ) }$

Our framework also gives PUI learners for low-dimensional convex concepts; see Theorem 5.13. Table 2 summarizes our upper bounds.

## 1.2.3 Agnostic Reliable Learning

Reliable learning models situations in which one type of mistake is more costly than the other. For example, a spam filter should make very few false positives while detecting as much spam as possible. The agnostic reliable model of [KKM12] formalizes this requirement by minimizing one type of error subject to making almost none of the other. Unlike ordinary agnostic learning, the comparator is required to make zero errors of the protected type. We follow the convention of [DRZ24].

Definition 1.10 (Agnostic Reliable Learning). Let F be a class of Boolean functions on $\mathbb { R } ^ { d }$ containing the constant function −1, and let D be a distribution on $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ . Define $\mathcal { G } _ { D } : = \{ f \in \mathcal { F } : \mathbf { P r } _ { ( \mathbf { x } , y ) \sim D } [ f ( \mathbf { x } ) =$ $1 , y = - 1 ] = 0 \}$ . A positive reliable learner, given i.i.d. samples from D and $\epsilon \in \mathsf { \Gamma } ( 0 , 1 )$ , outputs a hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$ such that, with probability at least $2 / 3 , \mathbf { P r } _ { ( \mathbf { x } , y ) \sim D } [ h ( \mathbf { x } ) = 1 , y = - 1 ] \le \epsilon$ and $\begin{array} { r } { \mathbf { P r } _ { ( \mathbf { x } , y ) \sim D } [ h ( \mathbf { x } ) = - 1 , y = 1 ] \leq \operatorname* { i n f } _ { f \in \mathcal { G } _ { D } } \mathbf { P r } _ { ( \mathbf { x } , y ) \sim D } [ f ( \mathbf { x } ) = - 1 , y = 1 ] \overset {  } { + } \epsilon . } \end{array}$ . Negative reliable learning is defined by interchanging the labels 1 and −1 and requiring that F contain the constant function 1.

Our reliable-learning algorithms follow from a reduction to PUI learning. The reduction applies to any class with a PUI learner; it does not require a separate partitioning argument.

Theorem 1.11 (From PUI to Reliable Learning; Informal Version of Lemma 6.3). Suppose that F contains the constant function 1 and has a PUI learner with sample complexity $N ( \epsilon , \sigma , p _ { + } , q )$ , running time $T ( \epsilon , \sigma , p _ { + } , q )$ , and failure probability at most 1/12. Then F has a negative reliable learner under $( q , \sigma )$ -smooth Gaussian marginals using $O \big ( N ( \epsilon / 2 , \sigma \epsilon / 4 , \epsilon / 4 , q ) / \epsilon + 1 / \epsilon ^ { 2 } \big )$ labeled samples and the analogous time bound with N replaced by T, with success probability at least $2 / 3$ . If F contains the constant function −1 and $- \mathcal { F }$ admits the corresponding PUI learner, applying the reduction to $- \mathcal { F }$ gives positive reliable learning of F.

Combining this reduction with our general PUI learner gives negative reliable learning for classes closed under afine restrictions, containing 1, and having finite Gaussian surface area. The dimension exponent is controlled by $m = m ^ { * } ( c _ { 0 } \epsilon , \Delta , c _ { 0 } \sigma \epsilon ^ { 3 } / q , 2 q ) < \infty$ , for $\epsilon , \Delta \in ( 0 , 1 / 2 )$ and a suficiently small universal constant $c _ { 0 } > 0 ;$ see Theorem 6.2. For intersections and unions of halfspaces, we obtain the following.

Theorem 1.12 (Reliable Learning of Intersections and Unions; Informal Version of Corollary 6.4). Let D be a distribution on $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ whose marginal $D _ { \mathbf { x } }$ is $( q , \sigma )$ -smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ . For $\epsilon \in ( 0 , 1 / 2 )$ , there is a negative reliable learner for intersections of k halfspaces (and a positive reliable learner for unions of k halfspaces) using at most $n = d ^ { m } \exp \bigl ( m ^ { O ( m ) } \bigr )$ labeled samples and $\mathrm { p o l y } ( n )$ time, where $\begin{array} { r } { m = O ( \log ( k ) + q \log { \frac { q } { \sigma \epsilon } } ) } \end{array}$

A conceptual contribution of the above is that reliable learning can be qualitatively easier than fully agnostic learning for a range of concept classes. For Gaussian halfspaces, this separation was established by [DRZ24]: reliable learning admits a logarithmic dimension exponent in inverse accuracy, whereas fully agnostic learning to error opt + ϵ has $\mathrm { S Q }$ complexity $d ^ { \Omega ( 1 / \epsilon ^ { 2 } ) }$ [DKPZ21]. Our result extends the algorithmic side of this separation to intersections and unions and to marginals that are only smooth with respect to a Gaussian. For a fixed number of halfspaces, the logarithmic dependence on $1 / \epsilon$ is already necessary in the SQ model, by the halfspace lower bound of [DRZ24].

Finally, we remark that our reduction also gives negative reliable learning of convex concepts of intrinsic dimension K and positive reliable learning of their complements; see Lemma 6.3 and Theorem 5.13.

Computational Limits for PQ and TDS learning. Our preceding results raise the question of whether Gaussian partitioning can also improve the complexity dependence in other models of reliable learning with distribution shift. We consider the realizable setting, with a common target function, a Gaussian training marginal, and an arbitrary test marginal. In PQ learning [GKKM20, KK21], the learner receives labeled training examples and unlabeled test examples and may abstain. It must abstain rarely on the training distribution and make few incorrect, non-abstaining predictions on the test distribution. In testable learning with distribution shift (TDS), the learner may reject the test distribution; whenever it accepts, its hypothesis must have small test error, and it must accept when the training and tes distributions agree. The formal definitions are in Definitions 7.1 and 7.2. Recent work gives algorithms and reductions for these models, including algorithms for Gaussian halfspaces and their intersections based on polynomial approximation and testing [KSV24b, GSSV24, CKK<sup>+</sup>24, KPSV26, KSV26].

For intersections of halfspaces without balance assumptions, the known general guarantees have inversepolynomial accuracy dependence in the dimension exponent. We show that this dependence cannot be reduced to logarithmic within the SQ framework. Even for intersections of two halfspaces and Gaussian training data, any learner newwith SQ access to both the labeled training distribution and the unlabeled test distribution must make $2 ^ { d ^ { \Omega ( 1 ) } }$ queries or use tolerance at most d<sup>−</sup> $- \Omega ( 1 / \epsilon ^ { 2 } )$ ; see Theorem 7.3. This significantly improves the prior known SQ lower bound of $d ^ { \tilde { \Omega } ( \log ( 1 / \epsilon ) ) }$ from [KSV24a]. For TDS, the accuracy exponent matches the known general bound for any fixed $k \geq 2$ , up to logarithmic factors [KSV24b]. For PQ learning, it rules out the logarithmic dependence achieved by our PUI and one-sided reliable learners.

## 1.3 Detailed Overview of Techniques

Our main technical contribution is a Gaussian partitioning primitive that produces thin regions covering almost all of the input mass. Within each such region, the conditional input distribution approximately matches low-degree Gaussian moments in the unrestricted directions. We first explain the geometry of the partition and how to construct it eficiently. We then describe the learning applications and the moment-matching constructions underlying our SQ lower bounds.

Brief Overview of Approach. We start with the entire space and build the partition recursively. On each current region, we check whether the conditional input distribution approximately matches low-degree Gaussian moments in every unrestricted direction. If we find a moment discrepancy, we subdivide the region into narrow intervals along the corresponding direction and continue on the resulting regions. Regions where we certify moment matching are kept. The main challenges are to find discrepancy directions eficiently and to control the number of regions produced by the recursion. We address the first through a low-dimensional search. For the second, we use a potential function to measure progress under subdivision, together with smoothness bounds that control the recursion depth and the total mass discarded.

The geometry of the regions is what makes the partition useful for learning. Conditioning the Gaussian reference on a region changes only its constrained coordinates, leaving the remaining coordinates standard Gaussian. At the same time, the narrow intervals ensure that the constrained coordinates vary little within each retained region. For a target function with bounded Gaussian surface area, we can fix the constrained coordinates within each certified region with small total error over the regions. The resulting functions depend only on the unrestricted coordinates, where the conditional input distributions approximately match Gaussian moments. This allows us to apply Gaussian approximation bounds within the partition.

For agnostic learning, moment matching lets us transfer Gaussian polynomial upper and lower approximations to the conditional input distributions. These approximations give a weak learner, which can then be boosted. For PUI learning, we use a diferent consequence of moment matching: for the classes we consider, if the positive examples match suficiently many Gaussian moments, then the target has small negative probability. Applying this certificate within the regions gives our PUI learners, and a reduction then yields one-sided reliable learning. Our SQ lower bounds use distributions that match many Gaussian moments exactly. Embedding these distributions in unknown low-dimensional subspaces produces instances that are dificult for SQ algorithms to distinguish from a Gaussian.

We now turn to the partition itself. We first explain why conditioning on a region creates a dificulty for applying Gaussian approximation bounds, and how this motivates the thinness condition.

## 1.3.1 From Logconcave Partitioning to Thin Gaussian Cells

The partitioning procedure of [DKM26] takes samples from a distribution Q obtained by rejection sampling from a logconcave reference P. It recursively partitions the space until, on cells containing almost all of the Q-mass, the conditional distributions $Q \mid R$ and $P \mid R$ approximately match low-degree directional moments after an afine normalization. For fixed moment and discarded-mass parameters, the number of cells is polynomial in the inverse acceptance probability. This reduces learning under Q to weak learning under distributions whose moments are close to those of a logconcave distribution.

A limitation of this approach is that its reference distribution on a cell is $P \mid R .$ . General logconcave distributions have weaker tail bounds than a Gaussian, and the available polynomial approximation bounds can require exponentially higher degree. For example, the approach in [DKM26] agnostically learns arbitrary functions of K halfspaces within error $( 1 + c ) { \mathrm { o p t } } + \epsilon$ by matching moments up to degree $\exp ( O ( ( K / c ) ^ { 4 } ) )$ . [DKM26] uses this degree bound even when the original reference is Gaussian: conditioning on a cell produces a truncated Gaussian, and afine normalization does not in general restore a Gaussian distribution. Thus, that earlier approach cannot leverage the sharper Gaussian approximation bounds.

We instead construct cells whose coordinates along a set of mutually orthogonal directions lie in short intervals, while the remaining coordinates are unrestricted. Let $V _ { R }$ denote the span of these constrained directions. For every certified cell we retain, the projection $\mathrm { p r o j } _ { V _ { R } } ( R )$ has small diameter. The cell therefore resembles a thin strip: its constrained coordinates lie in a small set, while its orthogonal coordinates remain unrestricted. Under the Gaussian reference $P = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ , conditioning on R changes only the constrained coordinates; the marginal on $V _ { R } ^ { \perp }$ remains standard Gaussian. Our moment guarantee states that $Q \mid R$ approximately matches low-degree Gaussian moments in every direction in $V _ { R } ^ { \perp }$

The purpose of the diameter bound is to remove the target’s dependence on the constrained coordinates. For a Boolean function f of bounded Gaussian surface area, we show that there are points ${ \mathbf z } _ { R } \in \mathrm { p r o j } _ { V _ { R } } ( R )$ such that the afine restrictions $f _ { R } ( { \bf x } ) = f ( { \bf z } _ { R } + { \bf x } _ { V _ { R } ^ { \perp } } )$ approximate f on the certified cells with small total error under the Gaussian reference P. To prove this, within each certified cell we resample the constrained coordinates from their conditional Gaussian distribution and keep the orthogonal coordinates fixed. Outside these cells we leave the point unchanged. This coupling preserves the Gaussian marginal and moves each point by at most the constrained diameter; see Lemma 4.5 for the Gaussian smoothing argument.

By smoothness, the same afine restrictions have small total error under Q. By choosing the grid resolution suficiently small, we make this error as small as required by the application. The guarantee is averaged over the cells; it does not assert a small conditional error in every cell. The points $\mathbf { z } _ { R }$ are used only in the analysis. When the class is closed under afine restrictions, each $f _ { R }$ remains in the class and depends only on the coordinates where Gaussian moment matching holds.

## 1.3.2 Constructing the Partition

At each cell R, we search for a direction in $V _ { R } ^ { \perp }$ along which one of the first k moments of $Q \mid R$ difers significantly from the corresponding Gaussian moment. We call the procedure used for this search a moment discrepancy finder. A certificate that the first k directional moments approximately match their Gaussian values makes R a certified cell; a discrepancy supplies a new direction along which to subdivide R. The procedure can also reject a cell when the checks needed for an eficient search fail. Starting from the entire space, we apply this procedure recursively. We first explain how a moment discrepancy leads to a useful subdivision and how to control the recursion depth. We then describe the discrepancy finder and show how conditional smoothness provides the guarantees needed by both steps.

Subdividing Cells and Controlling the Depth. On a cell with the required conditional smoothness, a moment discrepancy along a direction v implies that there is a threshold t for which $\mathbf { P r } _ { \mathbf { x } \sim Q | R } [ \mathbf { v } \cdot \mathbf { x } \leq t ]$ difers from $\mathbf { P r } _ { z \sim \mathcal { N } ( 0 , 1 ) } [ z \leq t ]$ by a nontrivial amount; see Lemma 3.5. We estimate these probabilities with fresh samples and use the split only if its gap is large enough. A single threshold would split R into two cells. To obtain thin cells, we also cut the central range of the coordinate $\mathbf { v } \cdot \mathbf { x }$ into intervals of width at most $\delta _ { 0 }$ , including the chosen threshold among the endpoints. Here $\delta _ { 0 } > 0$ is the grid resolution, chosen suficiently small relative to the recursion-depth bound to ensure that each retained cell has small diameter in its constrained directions. The two unbounded tail intervals are discarded, and we recurse on the bounded intervals.

Each new direction lies in the orthogonal complement of all previous constraint directions. Thus, after r subdivisions along a path, a remaining cell restricts r orthonormal coordinates to intervals of width at most $\delta _ { 0 }$ . Its constrained diameter is at most $\delta _ { 0 } { \sqrt { r } } .$ . In particular, a depth bound L gives both dim $( V _ { R } ) \le L$ and diam $( R ) \leq \delta _ { 0 } \sqrt { L }$ . The depth therefore controls both the number of constrained directions and the diameter of the cell in those directions.

We measure progress using the afinity potential $\begin{array} { r } { \Psi _ { t } = \sum _ { R \in { \cal A } _ { t } } \sqrt { P ( R ) Q ( R ) } } \end{array}$ , where $A _ { t }$ consists of the cells split in round t. The validated probability gap makes the conditional laws of the children under $P$ and $Q$ diferent. Quantitatively, their squared Hellinger distance is bounded below, so a split reduces the total afinity contribution of its parent by a factor $1 - \kappa$ for a positive κ determined by the gap. Subdividing further into short intervals cannot increase this contribution. Consequently, $\Psi _ { t + 1 } \leq ( 1 - \kappa ) \Psi _ { t } ;$ see Lemma 3.7.

For $q = 1$ , the bounded density ratio converts small afinity directly into small remaining Q-mass. For $q > 1$ , smoothness implies that the density ratio is bounded outside a set of small Q-mass, and the same argument applies on the remaining set; see Lemma 3.8. After suficiently many rounds, the cells that remain active have small total Q-mass. We stop the recursion at this point and discard these cells. The required number of rounds is polylogarithmic in the inverse smoothness and discarded-mass parameters when the moment parameters and $q$ are fixed. We also adapt the low-mass potential of [DKM26] to subdivisions with many children. It bounds the total mass lost when cells become too small for conditional sampling. Together with Gaussian tail bounds and the conditional smoothness analysis below, this accounts for the discarded cells and gives the stated quasipolynomial complexity; see Lemmas 3.9 and 3.10.

Finding a Discrepancy from Hermite Moments. We reduce the search for a moment discrepancy to a low-dimensional subspace. Fix a cell R and consider the projection of $Q \mid R$ onto $V _ { R } ^ { \perp }$ . Let $\mathcal { H } _ { i }$ be the normalized order-i Hermite tensor, and let $\mathbf { A } _ { i }$ be its expectation under this projected distribution. Every directional raw moment discrepancy through degree k is a linear combination of contractions $\mathbf { A } _ { i } \cdot \mathbf { v } ^ { \otimes i }$

We estimate these tensors from samples, flatten each along one index, and let W be the span of the left singular vectors with suficiently large singular values. Passing the Frobenius-norm test bounds the dimension of $W$ by $k ^ { O ( k ) } H ^ { 2 } / \Delta ^ { 2 }$ , where H is the norm cutof and $\Delta$ is the desired moment slack. The tensor components involving directions orthogonal to $W$ are small. Consequently, projecting a direction onto W approximately preserves the Hermite contractions that determine its moment discrepancies.

The projection need not have unit length. We therefore adjoin one direction in $V _ { R } ^ { \perp }$ orthogonal to W, when such a direction exists, and use it to restore the lost norm. This gives a unit vector whose directional moment discrepancies are close to those of the original direction, without changing their degrees. Searching a suficiently fine net of the unit sphere in this enlarged subspace then either finds a discrepancy of size $\Omega ( \Delta )$ or certifies moment matching in every direction; see Proposition 3.3 and Claim 3.19.

Conditional smoothness supplies the required tensor norm bounds. Contracting $\mathcal { H } _ { i }$ with a tensor of unit Frobenius norm gives a degree-i polynomial of Gaussian $L _ { 2 }$ norm at most one. Gaussian hypercontractivity and smoothness bound its expectation under the projected conditional distribution. Taking the supremum over these tensors bounds $\| \mathbf { A } _ { i } \| _ { F } ;$ see Lemma 3.13. The polynomial second-moment bound controls estimation error, while the Frobenius-norm cutof controls the dimension of the search.

The Hermite and singular-value approach already appears as motivation in [DKM26]. Our geometry allows us to use it throughout the recursion because the reference on the remaining coordinates is always Gaussian.

Controlling Conditional Smoothness. Both the subdivision step and the discrepancy finder require control of the conditional distribution. Without control of the tails, a moment discrepancy could come from a tiny amount of mass far from the origin, and no threshold need give a useful gap. Conditional smoothness supplies the required tail bounds. It also bounds the Hermite moment tensors used to find a discrepancy eficiently. The dificulty is that conditioning on a cell may weaken smoothness, so we need these guarantees on cells containing most of the mass.

For $q = 1$ , the density ratio of $Q$ to P is bounded. After conditioning on a cell $R ,$ this bound is multiplied by $P ( R ) / Q ( R )$ , but cells where this factor is too large have small total Q-mass. For $q > 1$ , the original density ratio can be unbounded, so this argument is insuficient. Instead, we prove an averaging statement for conditional densities: for every finite measurable partition, cells containing at least $1 - \eta$ of the Q-mass satisfy that $Q \mid R$ is $( 2 q , \sigma \eta / ( 2 q ) ,$ )-smooth with respect to $P \ | \ R .$ The proof bounds a suitable $L _ { p }$ norm of the global density ratio and then bounds the Q-weighted average of the corresponding conditional norms; see Lemma 3.11. We apply this statement to the final partition, so the exponent does not double at every level of the recursion.

Note that we cannot directly test whether a conditional distribution has these smoothness parameters. Instead, we check conditions that sufice for reliable moment estimation, eficient discrepancy finding, and progress under subdivision. First, we estimate the mass of a cell and stop processing it if obtaining enough conditional samples would be too expensive. For every cell passing this test, global smoothness gives a uniform second-moment bound for low-degree polynomials under $Q \mid R ,$ relative to their Gaussian second moments. This makes their empirical expectations reliable. Next, we test whether the empirical Hermite moment tensors have small Frobenius norms, which is the condition needed for an eficient search fo a discrepancy. Finally, before subdividing a cell we check that the proposed split actually separates its conditional distribution from the Gaussian reference by a suficiently large probability gap. We discard cells that fail the norm or split test and leave them unchanged in the final partition. On a cell with the required conditional smoothness, the norm test passes and any detected moment discrepancy yields a split with the required gap. Applying the averaging statement to the final partition therefore bounds the total Q-mass lost to these rejections. The uniform second-moment bound ensures that every returned moment certificate is sound, including on cells without the stated conditional smoothness guarantee.

## 1.3.3 Learning Theory Applications via Reductions to Gaussian Partitioning

Agnostic Learning. Our weak learner first applies Gaussian partitioning to the input marginal. On each cell with suficiently large estimated mass, it fits a low-degree polynomial to the observed labels using $L _ { 1 }$ regression and thresholds it to obtain a Boolean classifier. It combines these classifiers into a single hypothesis, predicting 1 on the remaining cells. We use this weak learner within smooth agnostic boosting, constructing a new partition for each reweighted input distribution.

The key implication of sandwiching is that correlation with the target gives correlation with a lowdegree polynomial. For the analysis, we replace the target on each certified cell by its afine restriction $f _ { R }$ Gaussian sandwiching polynomials $p _ { R } ^ { l } , p _ { R } ^ { u }$ satisfy the pointwise inequalities $p _ { R } ^ { l } \le f _ { R } \le p _ { R } ^ { u }$ . Their expected gap remains small under the actual conditional input distribution because its low-degree moments match the Gaussian moments. To make the required moment slack independent of the ambient dimension, we average the polynomials over irrelevant Gaussian coordinates. They then depend on at most K directions, so their coeficient bounds depend on the intrinsic dimension K rather than $d ;$ see Fact B.8.

In particular, either sandwiching polynomial approximates $f _ { R }$ in $L _ { 1 }$ under the conditional input distribution. Since $| y | = 1$ , this gives $\left| \mathbf { E } [ y f _ { R } ( \mathbf { x } ) \mid R ] - \mathbf { E } [ y p _ { R } ^ { u } ( \mathbf { x } ) \mid R ] \right| \leq \mathbf { E } [ | f _ { R } ( \mathbf { x } ) - p _ { R } ^ { u } ( \mathbf { x } ) | \mid R ]$ . Thus, a significant correlation between the labels and $f _ { R }$ implies a correlation with a low-degree polynomial. The conditional $L _ { 1 }$ loss of $p _ { R } ^ { u }$ against the labels is at most twice the conditional classification error of $f _ { R }$ plus the expected gap between $p _ { R } ^ { u }$ and $p _ { R } ^ { l }$ on $R .$ We use $L _ { 1 }$ polynomial regression and thresholding to find cellwise classifiers, which can be viewed as a partial hypothesis on the retained cells. Averaging over cells recovers the original label correlation, up to the total freezing and approximation errors; no individual cell is required to have a small noise rate. Since the omitted cells have small mass, completing this hypothesis gives the weak learner of Theorem 4.4.

As in [DKM26], smooth agnostic boosting then gives error $( 1 + c ) { \mathrm { o p t } } + \epsilon$ . The sandwiching accuracy, and hence its degree, depend on c rather than $\epsilon ;$ boosting only changes the smoothness parameter from σ to σϵ. Our partition allows this reduction to use Gaussian sandwiching degrees in place of those fo general logconcave distributions; see Fact 4.3 and Theorem 4.2.

Testable Agnostic Learning. The preceding agnostic learning argument combines properties of the partition—bounds on cell masses and constrained diameters, together with moment matching on certified cells—with smoothness to control the error from fixing the constrained coordinates. The partition properties can be checked empirically. Testable learning, however, requires an error guarantee whenever the learner accepts, even if the input marginal is not smooth. Thus it sufices to replace the direction fixing argument with an argument that it works for general distributions that pass eficient tests. We do this by assuming a stronger sandwiching result that holds for well-known function classes.

For any Boolean function of k halfspaces, we show that on each thin cell a single pair of low-degree polynomials in the unrestricted coordinates sandwiches the target for every choice of the constrained coordinates within the cell. Their expected Gaussian gap, averaged over the cells using Gaussian cell masses, is controlled by the constrained diameters and the polynomial approximation error; see Definition F.1. Essentially, this holds because there exist upper and lower bounding functions in the unrestricted coordinates that are close and they belong in the class. Using Gaussian sandwiching bounds for these functions gives the required polynomials; see Lemma F.6.

Combining this property with checks on cell diameters, empirical cell masses relative to their Gaussian masses, and conditional empirical moments gives a testable weak learner (Theorem F.3). Boosting then yields an eficient testable learner with error $( 1 + c ) \mathrm { o p t } + \epsilon ;$ see Theorem F.2. Moreover, our result extends beyond the Gaussian setting: our testable learner applies to any input marginal that is $( q , \sigma )$ -smooth with respect to the Gaussian. Our result gives eficient testable learning algorithms for functions of halfspaces

PUI Learning via Moment Certificates. For PUI learning, we partition the observed positive distribution $D _ { + } ^ { * } = D ^ { * } \mid \{ f = 1 \}$ , using $D = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ as the reference. Our hypothesis is positive on certified cells that pass the additional mass test described below, and negative elsewhere. The partition guarantee ensures that almost all positive examples lie in certified cells, so predicting negative on the uncertified cells incurs only a small false-negative error. To control false positives, we need a diferent consequence of moment matching: for the classes of interest, if the positive conditional distribution matches enough Gaussian moments, then the negative region has small probability under the target distribution. This is the moment certificate quantified by the Gaussian PUI learning exponent in Definition 5.2. It replaces the sandwiching assumption used for agnostic learning.

For a halfspace, the certificate follows from a univariate polynomial that is nonpositive on the positive halfspace but has positive Gaussian expectation. Its degree and coeficients show that a halfspace excluding Gaussian mass $\rho$ cannot support a distribution that matches the first $O ( \log ( 2 / \rho ) )$ Gaussian moments to a suficiently small constant slack. For an intersection of k halfspaces excluding mass $\rho ,$ one defining halfspace excludes mass at least $\rho / k ,$ giving degree $O ( \log ( 2 k / \rho ) ) ,$ ). Smoothness converts negative target mass into a lower bound on negative Gaussian mass. This gives the logarithmic degree bound used by the learner; see Lemma 5.15 and Corollary 5.16.

Applying this certificate within a cell requires some care. After fixing the constrained coordinates, the target becomes $f _ { R } ,$ , but the moments supplied by the partition come from examples conditioned on $f = 1$ not on $f _ { R } = 1$ . A small unconditional disagreement between these functions could change their positive conditional distributions substantially if the positive event were too rare. We therefore estimate $D _ { + } ^ { * } ( R )$ on each certified cell and discard the cell if this estimate is too small relative to its Gaussian mass $D ( R )$ . The cells discarded by this test have small total $D _ { + } ^ { * } { \mathrm { - m a s s } }$ . On the remaining cells, apart from a collection of small total $D ^ { * } { \mathrm { - m a s s } }$ , the mass test and smoothness give a lower bound on $D ^ { * } ( f = 1 \mid R )$ . Conditional smoothness also bounds the second moments of the directional monomials used in the certificate. Togethe with the bound on the disagreement between $f$ and $f _ { R }$ , these estimates control how much the moments change when we condition on $f _ { R } = 1$ instead of $f = 1$ . We can then apply the certificate to $f _ { R }$ to bound the false-positive error. The resulting algorithm and analysis appear in Algorithm 4 and Theorem 5.3.

From PUI to Reliable Learning. For negative reliable learning, consider a comparator with no false negatives and condition the labeled distribution on the event that its label agrees with the comparator. The comparator is realizable on this event, and all positive examples survive, so their conditional distribution is unchanged. The conditioned feature distribution remains smooth with respect to the Gaussian. We can therefore run the PUI learner directly on the observed positive examples. The conditioning is used only in the analysis. The resulting hypothesis has small false-negative error, since all positive examples lie in the agreement event. Its false-positive error is at most the comparator’s false-positive error plus the PUI learning error. A constant hypothesis handles the case of very few positive labels. Complementing labels and concepts gives positive reliable learning of unions; see Lemma 6.3 and Corollary 6.4.

## 1.3.4 SQ Lower Bounds via Moment Matching

PUI Learning. Our SQ lower bounds use the non-Gaussian component analysis (NGCA) method introduced in [DKS17], in the form developed in [DKRS23]. The method embeds a non-Gaussian distribution that matches many Gaussian moments in an unknown low-dimensional subspace and uses an independent Gaussian on its orthogonal complement. Distinguishing the resulting distribution from a standard Gaussian requires either many SQ queries or very small tolerance; see Fact 5.10. We show that a successful PUI learner would perform this distinction, yielding Theorem 5.4.

For halfspaces and their intersections, the main construction is a distribution supported on the positive region that matches Gaussian moments exactly through logarithmic degree. In one dimension, we delete a small Gaussian tail and repair its missing moments by a density correction on a central interval. The correction is small enough that the resulting density remains nonnegative and bounded by a small multiple of the Gaussian density. Products of these distributions give the construction for intersections; see Lemmas 5.18 and 5.22 and Theorem 5.14.

PQ and TDS Learning. The PQ/TDS lower bound uses the same NGCA method with a diferent moment-matching distribution. The discrete construction of [DKS17] matches Θ(m) Gaussian moments while placing mass $\Theta ( 1 / \sqrt { m } )$ at the origin. We embed it in a hidden direction and take the target to be a suficiently narrow slab around the corresponding hyperplane. By taking the slab suficiently narrow, we can simulate the labeled training SQ oracle using the Gaussian distribution with constant −1 labels, within the requested tolerances and without knowing the hidden direction. The test distribution still places substantial mass inside the slab. Setting $m = \Theta ( 1 / \epsilon ^ { 2 } )$ forces a successful PQ or TDS learner to distinguish the planted test distribution from a Gaussian. Applying the NGCA lower bound of Fact 5.10 gives the inverse-quadratic accuracy dependence in Theorem 7.3.

## 1.4 Related Work

Multi-index models have a long history in statistics, including projection pursuit regression and suficient dimension reduction [FS81, Li91, XTLZ02]. Here we focus on algorithmic results for learning Boolean MIMs. Several distribution-specific algorithms are known for Boolean MIMs. [Vem10] gave a PAC learner for intersections of halfspaces under logconcave distributions when the defining hyperplanes pass through the mean. Under Gaussian inputs, polynomial regression and Gaussian surface-area bounds give agnostic learners for broad classes of Boolean functions [KKMS08, KOS08]. [KKM13] extended this approach to agnostically learn functions of a constant number of halfspaces under logconcave distributions.

In the distribution-free realizable setting, [APS26] recently gave a PAC learner for arbitrary functions of k halfspaces whose running time is subexponential in the ambient dimension for constant k. Under an additional margin assumption, [PV26] obtained improved bounds for learning intersections of halfspaces. These results require subexponential time even for k = 2.

For general MIMs under Gaussian inputs, [DKK<sup>+</sup>24] gave agnostic learners with error opt + ϵ using membership queries, with fixed-degree polynomial dependence on the ambient dimension. For learning from random examples, [DIKZ25] developed a general approach for broad classes of finite-valued MIMs in the presence of adversarial label noise. Their applications include approximate agnostic learners for intersections of halfspaces and multiclass linear classifiers. [DIKR25] (see also [DLB25] for a similar result) developed an analogous framework for real-valued MIMs under square loss. Both works use conditional low-degree moments to iteratively approximate the relevant subspace, and establish qualitatively matching

SQ lower bounds on the dimension dependence. Their conditional-moment criteria also generalize the generative-exponent condition for single-index parameter recovery studied in [DPVLB24].

Other work on real-valued SIMs and MIMs includes analyses of gradient methods and moment-based subspace recovery [DLS22, ABAM23, LVM26], as well as agnostic learners for SIMs [WZDD24, WZDD25].

The closest prior work is [DKM26], which developed a logconcave partitioning procedure and used it to construct weak partial learners that can be boosted to approximate agnostic and testable agnostic learners.

AI Disclosure. The authors set the research direction, developed all the ideas, and proved all the theorems in this paper. AI was used to audit and edit the proofs and algorithms for minor errors, and to improve the presentation. The authors reviewed the feedback and incorporated minor corrections.

Organization. The paper is structured as follows. In Section 2, we introduce the necessary definitions and preliminaries. In Section 3, we develop our partitioning primitive for smooth distributions and establish its guarantees. In Section 4, we use this primitive to obtain our agnostic learning algorithm, establishing Theorem 4.2, and derive consequences for concrete concept classes. In Section 5, we present and prove our PUI learning result, Theorem 5.3, as well as our statistical query lower bound, Theorem 5.4. In Subsection 5.3, we give PUI learners for intersections of halfspaces and low-dimensional convex concepts, together with an SQ lower bound for intersections. In Section 6, we reduce one-sided agnostic reliable learning to PUI learning. In Section 7, we show that the logarithmic dependence on $1 / \epsilon$ in the dimension exponent is impossible for PQ and TDS learning in the SQ model, even for intersections of two halfspaces. The appendix also contains an SQ lower bound for agnostic learning with a multiplicative approximation guarantee (Section E) and our testable agnostic learning results (Section F).

## 2 Preliminaries and Notations

Basic notation. For $n \in \mathbb { Z } _ { + }$ , let $[ n ] : = \{ 1 , \dots , n \}$ . We use bold lowercase letters for vectors and bold uppercase letters for matrices and tensors. For $\mathbf { x } \in \mathbb { R } ^ { d } , x _ { i }$ denotes its i-th coordinate, and $\| \mathbf { x } \| _ { 2 } =$ $( \Sigma _ { i = 1 } ^ { d } x _ { i } ^ { 2 } ) ^ { 1 / 2 }$ denotes its Euclidean norm. We often omit the subscript and write $\| \mathbf { x } \|$ . We write $\mathbf x \cdot \mathbf y$ for the inner product and $\mathbb { S } ^ { d - 1 } : = \{ \mathbf { x } \in \mathbb { R } ^ { d } : \| \mathbf { x } \| _ { 2 } = 1 \}$ for the unit sphere. For a subspace $V \subseteq \mathbb { R } ^ { d }$ , let $V ^ { \perp }$ denote its orthogonal complement and write proj x, or $\mathbf { x } _ { V }$ , for the orthogonal projection of x onto V . We denote the linear span of a set of vectors L by span(L), and write $V + W : = \{ \mathbf { v } + \mathbf { w } : \mathbf { v } \in V , \mathbf { w } \in W \}$ . Let $\mathbf { e } _ { i }$ be the i-th standard basis vector and $\mathbf { I } _ { d }$ the $d \times d$ identity matrix. The Frobenius norm $\| \mathbf { A } \| _ { F }$ of a matrix or tensor is the square root of the sum of its squared entries. For $A \subseteq \mathbb { R } ^ { d }$ , define dist $( \mathbf { x } , A ) : = \operatorname* { i n f } _ { \mathbf { a } \in A } \| \mathbf { x } - \mathbf { a } \| _ { 2 }$ Let $R = \{ \mathbf { x } \mid \mathbf { w } ^ { ( i ) } \cdot \mathbf { x } \leq b _ { i } , i \in [ m ] \}$ be a polyhedral cell in $\mathbb { R } ^ { d }$ and let $V _ { R } = \operatorname { s p a n } ( \mathbf { w } ^ { ( 1 ) } , \ldots , \mathbf { w } ^ { ( m ) } )$ be the subspace spanned by the constraints that define R. We define diam $( R ) = \operatorname* { s u p } \{ \| \mathbf { u } - \mathbf { v } \| \mid \mathbf { u } , \mathbf { v } \in \operatorname { p r o j } _ { V _ { R } } ( R ) \}$ to be the diameter of $R .$

We use standard asymptotic notation, with ${ \widetilde { O } } ( \cdot )$ suppressing polylogarithmic factors. We write $a \lesssim b$ when $a \leq C b$ for an absolute constant $C > 0$ . The notation $\mathrm { p o l y } ( \cdot )$ and polylog(·) denotes polynomial and polylogarithmic dependence, respectively.

Probability notation. We write $\mathbf { E } _ { \mathbf { x } \sim D } [ g ( \mathbf { x } ) ]$ for expectation under D and $\mathbf { P r } _ { \mathbf { x } \sim D } [ \mathcal { E } ]$ for the probability of an event E. We omit the distribution when it is clear from context and write $D ( A ) : = \mathbf { P r } _ { \mathbf { x } \sim D } [ \mathbf { x } \in A ]$ For a joint distribution D over $\left( \mathbf { x } , y \right)$ , let $D _ { \mathbf { x } }$ denote its feature marginal. We write $\mathcal { N } ( \pmb { \mu } , \mathbf { I } _ { d } )$ for the Gaussian distribution with mean $\pmb { \mu }$ and identity covariance; $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ is the standard Gaussian. We write $Q \ll P { \mathrm { ~ i f ~ } } $ Q is absolutely continuous with respect to $P ,$ meaning that $P ( E ) = 0$ implies $Q ( E ) = 0$ for every measurable set E. This guarantees the existence of the density ratio $r = \mathrm { d } Q / \mathrm { d } P$ , uniquely defined P-almost everywhere. For discrete probability distributions $P = ( p _ { i } ) _ { i = 1 } ^ { m }$ and $Q = ( q _ { i } ) _ { i = 1 } ^ { m }$ , their squared Hellinger distance is $\begin{array} { r } { H ^ { 2 } ( P , Q ) : = 1 - \sum _ { i = 1 } ^ { m } \sqrt { p _ { i } q _ { i } } } \end{array}$ . For a distribution $P ,$ a measurable function $f : \mathbb { R } ^ { d }  \mathbb { R }$ and $1 \leq p < \infty$ , we define the $L _ { p } ( P )$ norm of $f$ by $\| f \| _ { L _ { p } ( P ) } : = ( \mathbf { E } _ { \mathbf { x } \sim P } [ | f ( \mathbf { x } ) | ^ { p } ] ) ^ { 1 / p }$

Definition 2.1 (Degree-ℓ $L _ { 1 }$ sandwiching polynomials). Let H be a class of Boolean functions on $\mathbb { R } ^ { K }$ let D be a class of distributions on $\mathbb { R } ^ { K }$ , and let $\ell : ( 0 , 1 ] \to \mathbb { Z } _ { + }$ be integer-valued. We say that H admits degree-ℓ $L _ { 1 }$ sandwiching polynomials over D if, for every $c \in ( 0 , 1 )$ , every $g \in { \mathcal { H } }$ , and every $\mu \in \mathcal { D }$ , there exist polynomials $p ^ { l } , p ^ { u }$ of degree at most $\ell ( c )$ such that $p ^ { l } ( \mathbf { z } ) \leq g ( \mathbf { z } ) \leq p ^ { u } ( \mathbf { z } )$ for every $\mathbf { z } \in \mathbb { R } ^ { K }$ , and $\begin{array} { r } { \mathbf { E } _ { \mathbf { z } \sim \mu } \big [ | p ^ { u } ( \mathbf { z } ) - p ^ { l } ( \mathbf { z } ) | \big ] \leq c . } \end{array}$

Definition 2.2 (Directional Moment matching). Let P and $Q$ be distributions over $\mathbb { R } ^ { d } .$ , and let $W \subseteq \mathbb { R } ^ { d }$ be a subspace. We say that P and Q match their first k directional moments on W with slack $\Delta \ i f , \ f o r$ every unit vector $\mathbf { v } \in W$ and every $i \in [ k ] , \big | \mathbf { E } _ { \mathbf { x } \sim P } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { i } ] - \mathbf { E } _ { \mathbf { x } \sim Q } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { i } ] \big | \leq \Delta$ . When $W = \mathbb { R } ^ { d }$ , we simply say that P and Q match their first k directional moments with slack $\Delta$

We refer the reader to Section A for the omitted preliminaries.

## 3 Gaussian Space Partitioning in Quasi-Polynomial Time

In this section, we give the main algorithmic result in this paper, Theorem 3.1, which formalizes our Gaussian Partitioning primitive. First, we give an overview of our approach. In Subsection 3.1, we present the algorithm and prove the main helper lemmas supporting the analysis. In Subsection 3.2, we provide the complete analysis of our algorithm, and in Subsection 3.3, we prove the guarantees of the Moment Discrepancy Finder subroutine (Proposition 3.3), which is needed for the analysis.

As mentioned in the introduction, our algorithm, given samples from a distribution Q that is smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ , eficiently partitions the space and marks a subcollection of cells, which we will refer to as certified. Each marked cell R satisfies a mass lower bound, a diameter bound (within the subspace $V _ { R }$ spanned by its constraint normals), and a directional moment matching guarantee in $V _ { R } ^ { \perp }$ , as stated below. These certified cells together contain almost all of the probability mass of $Q .$ . We state the formal result below.

Theorem 3.1 (Gaussian Partitioning Algorithm). Let $k , d \in \mathbb { N } _ { + } , q \geq 1 , \sigma \in ( 0 , 1 ]$ . Let $P = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ and suppose that Q is a distribution on $\mathbb { R } ^ { d }$ that is $( q , \sigma )$ -smooth with respect to $P$ . There is an algorithm such that given $\eta , \delta , \Delta \ \in \ ( 0 , 1 / 2 )$ , it draws at most $d ^ { k } B$ samples from $Q _ { i }$ , runs in time $d ^ { O ( k ) } B$ for $\begin{array} { r } { B = \exp \left( \frac { O ( ( q k ) ^ { O ( k ) } ) } { \Delta ^ { O ( 1 ) } } \log ^ { O ( k ) } ( \frac { 2 } { \sigma \eta } ) \log ( \frac { 2 } { \delta } ) \right) } \end{array}$ , and, with probability at least $2 / 3$ , returns a polyhedral partition U of $\mathbb { R } ^ { d }$ with $| \mathcal { U } | \le B$ and a collection $\mathcal { G } \subseteq \mathcal { U }$ satisfying $Q \left( \bigcup _ { R \in { \mathcal { G } } } R \right) \geq 1 - \eta$ . Every $R \in { \mathcal { G } }$ satisfies:

(i) $Q ( R ) \geq 1 / B$

(ii) $\mathrm { d i a m } ( R ) \leq \delta$ , where the diameter is measured in the constraint subspace $V _ { R }$

(iii) $P \mid R$ and $Q \mid R$ match their first k directional moments on $V _ { R } ^ { \perp }$ with slack $\Delta$

To comment on the complexity bound, it is useful to consider how the parameters of Theorem 3.1 are chosen in our subsequent learning applications. For example, in our agnostic learning application (Theorem 4.2), the moment order $k ,$ moment slack $\Delta ,$ and discarded mass $\eta$ depend only on the complexity of the model class and the relative approximation parameter $c .$ For a class of bounded complexity, such as Boolean functions of a constant number of halfspaces, and constant $^ { c , }$ these parameters can therefore be held constant. The desired additive accuracy ϵ enters through boosting, which uses distributions with smoothness parameter $\sigma \epsilon$ . The diameter parameter $\delta$ can be taken polynomially small in $\sigma \epsilon ,$ contributing only a factor of $\log ( 2 / \delta )$ to the exponent. These choices imply that, for fixed $q .$ , the complexity is polynomial in d and quasi-polynomial in $1 / ( \sigma \epsilon )$

Remark 3.2 (Comparison with Logconcave Partitioning [DKM26]). A quasipolynomial-complexity logconcave partitioning result also appears in [DKM26], but our theorem provides several improvements in the Gaussian setting. The main improvement is the diameter bound in outcome (ii) of Theorem 3.1. As discussed in the introduction, for targets with bounded Gaussian surface area, small cell diameter allows us to fix the coordinates in each cell’s constraint subspace with only a small total approximation error. The resulting function on each cell depends only on $V _ { R } ^ { \perp }$ , where the directional moment-matching guarantee applies. This allows us to transfer certain Gaussian learning guarantees to the conditional distributions on the certified cells, which is vital for our improved complexity results. Second, our partitioning theorem applies to general $( q , \sigma )$ -smooth distributions, whereas [DKM26] considers only the case $q = 1$ . Finally, our complexity bound has an improved dependence on the moment order k.

Before diving into the details of Theorem 3.1, we first give a rough overview of the intuition behind the algorithm. As in [DKM26], our starting point is a moment discrepancy finder. This is a routine that, given sample access to a distribution $Q _ { i }$ , either certifies that $Q$ approximately matches the first k directional moments of a standard Gaussian or returns a direction v along which some moment of order $i \in [ k ]$ does not match.

We show in Subsection 3.3 that if $Q$ is $( q , \sigma )$ -smooth with respect to a Gaussian distribution, or, more generally, if the expected Hermite tensors under $Q$ have small Frobenius norms, then such a routine can be implemented eficiently. In fact, the latter condition gives us a way to test whether the routine can perform the required moment discrepancy task reliably within a fixed computational budget. Formally, we state the following result for the task of moment discrepancy finding, whose proof and analysis are deferred to Subsection 3.3.

Proposition 3.3 (Moment Discrepancy Finder). Let $k , d \in \mathbb { Z } _ { + } , \Delta \in ( 0 , 1 / 2 )$ , and $V , H \geq 1$ . Let $P = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ and let Q be a distribution over $\mathbb { R } ^ { d }$ satisfying $\mathbf { E } _ { \mathbf { x } \sim Q } [ p ( \mathbf { x } ) ^ { 2 } ] \leq V \Vert p \Vert _ { L _ { 2 } ( P ) } ^ { 2 }$ for every polynomial p of degree at most k.

Algorithm $\mathcal { L }$ draws $d ^ { k } O ( k ^ { O ( k ) } ) \frac { V } { \Delta ^ { 2 } }$ samples from $Q ,$ , runs in $d ^ { O ( k ) }$ exp $\begin{array} { r } { \left( O ( k ^ { O ( k ) } ) \frac { H ^ { 2 } } { \Delta ^ { 2 } } \log \frac { H } { \Delta } \right) V ^ { O ( 1 ) } } \end{array}$ time and returns Reject, None, or a unit direction. With probability at least ${ \dot { 2 } } / 3$ , all of the following hold:

1. If the algorithm returns a direction w, then for some $\ell \in [ k ]$

$$
\underset { \mathbf { x } \sim Q } { \mathbf { E } } [ ( \mathbf { w } \cdot \mathbf { x } ) ^ { \ell } ] - \underset { \mathbf { x } \sim P } { \mathbf { E } } [ ( \mathbf { w } \cdot \mathbf { x } ) ^ { \ell } ] \bigg | \geq 3 \Delta / 8 .
$$

2. If the algorithm returns None, then for every unit vector v and every $\ell \in [ k ]$

$$
\left| \underset { \mathbf { x } \sim Q } { \mathbf { E } } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { \ell } ] - \underset { \mathbf { x } \sim P } { \mathbf { E } } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { \ell } ] \right| < \Delta .
$$

3. If every Hermite moment tensor $\mathbf { A } _ { i } = \mathbf { E } _ { \mathbf { x } \sim Q } [ \mathcal { H } _ { i } ( \mathbf { x } ) ] , i \in [ k ]$ , satisfies $\| \mathbf { A } _ { i } \| _ { F } \leq H$ , the algorithm does not return Reject.

We now give the intuition for constructing the desired partition using a moment discrepancy finder as a black box. We focus on the case $q = 1$ . Suppose that we find a direction v from our moment discrepancy finding algorithm. Since $Q$ has a moment discrepancy along v and is smooth with respect to the standard Gaussian P, the argument in [DKM26] yields a threshold $t \in \mathbb { R }$ such that the Bernoulli distributions of $\mathbb { 1 } \{ \mathbf { v } \cdot \mathbf { x } \geq t \}$ under $\mathbf { x } \sim Q$ and $\mathbf { x } \sim P$ have large Kolmogorov distance. This threshold splits the space into two halfspaces. To obtain a partition into cells of small diameter, in addition to the split used in [DKM26], we further refine each half into short intervals along v. The observation is that in expectation, the weight of $Q \mid R$ with respect to $P \mid R$ is not smaller than the weight of $Q$ with respect to $P$ . This implies that, if we recursively run this procedure over each $R ,$ we would expect the weight of $Q \mid R$ with respect to $P \mid R$ to become either extreme (very small or very large) for most cells. Thus, the procedure should stop after a small number of rounds, with the desired moment matching holding on cells covering all but a smal amount of Q-mass.

However, such a naive analysis does not immediately imply Theorem 3.1. There are still several important challenges. First, we must ensure that the total number of cells does not grow exponentially in $1 / \sigma$ . Second, although each step introduces a new direction v and a fine grid along v, the diameter of a cell R also depends on the dimension of the subspace defining $R ,$ hence we must show that this dimension does not grow too large. Furthermore, the intuition above requires $Q \mid R$ to remain suficiently smooth with respect to $P \mid R$ as we recursively apply the procedure to each cell R. For $q > 1$ , we must therefore show that this smoothness property holds on cells containing most of the Q-mass.

## 3.1 Partitioning Algorithm

In this subsection, we present the Gaussian partitioning algorithm, using a moment discrepancy finder as a black box. We first show how moment discrepancies yield useful one-dimensional splits that allow us to keep the partition depth small. We then describe the algorithm and establish the lemmas needed to bound the total Q-mass of uncertified cells.

Following the discussion above, we introduce our goals when splitting cells along each direction. Namely, we aim to obtain a large separation between the probabilities assigned by P and $Q$ to the two sides of the split (which will help us control the number of cells), and to subdivide all but the tail regions into intervals of width at most $\delta _ { 0 } ~ ( \delta _ { 0 }$ will be set much smaller than $\delta$ so that the overall diameter is at most δ).

Definition 3.4 (Gaussian Good Split and Refinement). Let $\kappa \in ( 0 , 1 )$ and $T , \delta _ { 0 } > 0$ be parameters. Let $P = \mathcal { N } ( 0 , 1 )$ and let $Q$ be a distribution on R. A threshold t is a $\kappa { - } G$ aussian good split $i f$ the Bernoulli distributions with parameters $p _ { t } : = \mathbf { P r } _ { z \sim P } [ z \geq t ]$ and $q _ { t } : = \mathbf { P r } _ { z \sim Q } [ z \geq t ]$ satisfy $H ^ { 2 } ( p _ { t } , q _ { t } ) =$ $1 - \sqrt { p _ { t } q _ { t } } - \sqrt { ( 1 - p _ { t } ) ( 1 - q _ { t } ) } \geq \kappa$ , where $H ^ { 2 }$ is the squared Hellinger distance. For $t \in [ - T , T ]$ , subdivide each nonempty interval $[ - T , t )$ and $[ t , T )$ into consecutive half-open intervals of length $\delta _ { 0 }$ , with possibly shorter end intervals, and include the two intervals $( - \infty , - T )$ and $[ T , \infty )$ . The resulting partition of R refines the binary split at t and contains at most $\left\lceil 2 T / \delta _ { 0 } \right\rceil + 3$ intervals.

We start by showing that, if Q is $( q , \sigma )$ -smooth with respect to $P ,$ then a moment discrepancy of size $\Delta$ between $Q$ and P implies the existence of a κ-Gaussian good split, with κ inverse-polylogarithmic in $1 / \sigma$ for fixed $q , k , \Delta$ . Furthermore, such a good split can be eficiently found and refined. The intuition here is that, for a general distribution Q, a moment discrepancy does not necessarily imply the existence of a quantitatively good split, because a small amount of mass far in the tails can create a large moment discrepancy. However, the smoothness of Q controls the contribution of the tails to its low-degree moments. Thus, once a discrepancy exists, there must be a threshold t in a bounded interval such that $Q ( z > t )$ difers quantitatively from $P ( z > t )$ , which also gives a quantitative lower bound on the squared Hellinger distance. By checking empirical tail probabilities at candidate thresholds, we can eficiently find such a good split and refine it.

Lemma 3.5 (Moment Discrepancy implies a Gaussian Good Split). Let $P = \mathcal { N } ( 0 , 1 )$ , and let $Q$ be a distribution over R that is $( q , \sigma )$ -smooth with respect to P, where $q \geq 1$ and $\sigma \in ( 0 , 1 ]$ . Let $k \in \mathbb { N } _ { + }$ and $\Delta \in ( 0 , 1 )$ , and suppose that $\begin{array} { r } { \left| \mathbf { E } _ { z \sim P } [ z ^ { i } ] - \mathbf { E } _ { z \sim Q } [ z ^ { i } ] \right| \geq \Delta } \end{array}$ for some $i \in [ k ]$ . Define $\Lambda : = q ( k \log ( 2 q k ) +$ $\log ( 1 6 / ( \sigma \Delta ) ) )$ . There are universal constants $C _ { 0 } , C > 1$ such that, for every $b \ge C _ { 0 } \sqrt { \Lambda }$ , there exists a threshold $t \in [ - b , b ]$ satisfying

$$
\left| \underset { z \sim P } { \mathbf { P r } } [ z \ge t ] - \underset { z \sim Q } { \mathbf { P r } } [ z \ge t ] \right| \ge \frac { \Delta } { 4 b ^ { k } } .
$$

The threshold can also be chosen to be a κ-Gaussian good split, where $\kappa : = \Delta ^ { 2 } / ( C \Lambda ) ^ { k }$ . Moreover, for every $\delta \in ( 0 , 1 )$ , there is an algorithm that draws $\begin{array} { r } { O \left( \frac { ( C \Lambda ) ^ { k } } { \Delta ^ { 2 } } \mathrm { l o g } \frac { 2 } { \delta } \right) = \mathrm { p o l y l o g } _ { q , k , \Delta } ( 2 / \sigma ) \log ( 2 / \delta ) } \end{array}$ samples from $Q$ runs in sample-polynomial time, and outputs a threshold satisfying these conclusions with probability at least $1 - \delta$

Proof of Lemma 3.5. Fix any $i \in [ k ]$ witnessing the moment discrepancy. Smoothness implies $Q \ll P$ since $P ( E ) = 0$ implies $Q ( E ) = 0$ . Thus, Q is absolutely continuous with respect to Lebesgue measure. We first show that truncating to an interval of radius $O ( \sqrt { \Lambda } )$ changes the moment discrepancy by at most $\Delta / 8$ . Let $L = C _ { 0 } \sqrt { \Lambda } \ge 1$ , where $C _ { 0 } > 1$ is a suficiently large universal constant. Cauchy–Schwarz, the bound $\mathbf { E } _ { z \sim P } [ | z | ^ { 2 i } ] \leq ( 2 i ) ^ { i }$ , and the standard Gaussian tail bound give

$$
\begin{array} { r l r } {  { \underset { z \sim P } { \mathbf { E } } [ | z | ^ { i } \Im \{ | z | > L \} ] \leq ( \underset { z \sim P } { \mathbf { E } } [ | z | ^ { 2 i } ] ) ^ { 1 / 2 } \underset { z \sim P } { \mathbf { P r } } [ | z | > L ] ^ { 1 / 2 } } } \\ & { } & { \leq \sqrt { 2 } ( 2 i ) ^ { i / 2 } e ^ { - L ^ { 2 } / 4 } \leq \Delta / 1 6 . } \end{array}
$$

For $t > 0$ , smoothness and the Gaussian tail bound give

$$
\mathbf { P r } \left[ | z | > t \right] \leq \frac { \mathbf { P r } _ { z \sim P } [ | z | > t ] ^ { 1 / q } } { \sigma } \leq \frac { 2 ^ { 1 / q } } { \sigma } e ^ { - t ^ { 2 } / ( 2 q ) } \leq \frac { 2 } { \sigma } e ^ { - t ^ { 2 } / ( 2 q ) } .
$$

Writing the moment as an integral of tail probabilities, we obtain

$$
\underset { z \sim Q } { \mathbf { E } } [ \vert z \vert ^ { 2 i } ] = \int _ { 0 } ^ { \infty } 2 i t ^ { 2 i - 1 } \underset { Q } { \mathbf { P r } } [ \vert z \vert > t ] \mathrm { d } t \leq \frac { 2 } { \sigma } \int _ { 0 } ^ { \infty } 2 i t ^ { 2 i - 1 } e ^ { - t ^ { 2 } / ( 2 q ) } \mathrm { d } t = \frac { 2 } { \sigma } ( 2 q ) ^ { i } i ! \leq \frac { 2 } { \sigma } ( 2 q i ) ^ { i } .
$$

In particular, all the moments used below are finite. Applying Cauchy–Schwarz under Q gives

$$
\underset { z \sim Q } { \mathbf { E } } \left[ \vert z \vert ^ { i } \mathbf { 1 } \{ \vert z \vert > L \} \right] \leq \left( \underset { Q } { \mathbf { E } } [ \vert z \vert ^ { 2 i } ] \right) ^ { 1 / 2 } \mathbf { P r } [ \vert z \vert > L ] ^ { 1 / 2 } \leq \frac { 2 } { \sigma } ( 2 q i ) ^ { i / 2 } e ^ { - L ^ { 2 } / ( 4 q ) } \leq \frac { \Delta } { 1 6 } .
$$

The last inequalities in both tail estimates hold for a suficiently large universal $C _ { 0 }$ , uniformly over $i \in [ k ]$ 2 since

$$
\frac { L ^ { 2 } } { q } = C _ { 0 } ^ { 2 } \left( k \log ( 2 q k ) + \log \frac { 1 6 } { \sigma \Delta } \right) \geq C _ { 0 } ^ { 2 } \left( i \log ( 2 q i ) + \log \frac { 1 6 } { \sigma \Delta } \right) .
$$

Let $\mathrm { c l i p } _ { L } ( z ) : = \operatorname* { m a x } \{ - L , \operatorname* { m i n } \{ z , L \} \}$ and define $g ( z ) : = \mathrm { c l i p } _ { L } ( z ) ^ { i }$ . Since $| z ^ { i } - g ( z ) | \leq | z | ^ { i } \mathbb { 1 } \{ | z | > L \}$ , the assumed moment discrepancy and the preceding two tail estimates yield

$$
\left| \underset { z \sim P } { \mathbf { E } } [ g ( z ) ] - \underset { z \sim Q } { \mathbf { E } } [ g ( z ) ] \right| \geq \Delta - \frac { \Delta } { 1 6 } - \frac { \Delta } { 1 6 } = \frac { 7 \Delta } { 8 } .\tag{1}
$$

Define the restricted Kolmogorov discrepancy

$$
d _ { \mathrm { K } , L } ( P , Q ) : = \operatorname* { m a x } _ { t \in [ - L , L ] } \left| \mathbf { P r } \left[ z \geq t \right] - \mathbf { P r } \left[ z \geq t \right] \right| .
$$

The maximum exists because both tail probabilities are continuous.

Claim 3.6. We have $d _ { \mathrm { K } , L } ( P , Q ) \geq 7 \Delta / ( 1 6 L ^ { k } )$

Proof of Claim 3.6. For every $x \in \mathbb { R }$ , the clipped monomial satisfies the tail-integral identity

$$
g ( x ) = \int _ { 0 } ^ { L } i t ^ { i - 1 } \left( \mathbb { 1 } \{ x \geq t \} + ( - 1 ) ^ { i } \mathbb { 1 } \{ x \leq - t \} \right) \mathrm { d } t .
$$

Taking expectations and applying Fubini’s theorem gives

$$
\mathbf { E } _ { P } ^ { [ g ] } - \mathbf { E } _ { Q } ^ { [ g ] } = \int _ { 0 } ^ { L } { \dot { u } } ^ { i - 1 } \left( \mathbf { P } _ { P } [ z \geq \ell ] - \mathbf { P } _ { Q } [ z \geq \ell ] \right) \mathrm { d } \ell + ( - 1 ) ^ { i } \int _ { 0 } ^ { L } { \dot { u } } ^ { i - 1 } \left( \mathbf { P } _ { P } [ z \leq - \ell ] - \mathbf { P } _ { Q } [ z \leq - \ell ] \right) \mathrm { d } \ell .
$$

Since P and $Q$ are absolutely continuous, a lower-tail discrepancy at −t equals the corresponding upper-tail discrepancy in absolute value. All thresholds in these integrals lie in $[ - L , L ]$ , so

$$
\begin{array} { r l r } {  { \biggl | \frac { { \bf E } [ g ] - { \bf E } [ g ] } { { \bf E } } \biggr | \le 2 d _ { \mathrm { K } , L } ( P , Q ) \int _ { 0 } ^ { L } i t ^ { i - 1 } \mathrm { d } t } } \\ & { } & { = 2 L ^ { i } d _ { \mathrm { K } , L } ( P , Q ) \le 2 L ^ { k } d _ { \mathrm { K } , L } ( P , Q ) , } \end{array}
$$

where the last inequality uses $i \leq k$ and $L \geq 1$ . Combining this with (1) gives the claimed lower bound. This completes the proof of Claim 3.6 □

Set $\rho : = \Delta / ( 4 L ^ { k } )$ . By Claim 3.6, there is a threshold $t _ { \star } \in [ - L , L ]$ such that $| p _ { \star } - q _ { \star } | \geq 7 \Delta / ( 1 6 L ^ { k } ) =$ $7 \rho / 4 ,$ , where $p _ { \star } : = \mathbf { P r } _ { z \sim P } [ z \geq t _ { \star } ]$ and $q _ { \star } : = \mathbf { P r } _ { z \sim Q } [ z \geq t _ { \star } ]$ . For Bernoulli distributions, $H ^ { 2 } ( \mathrm { B e r n } ( p ) , \mathrm { B e r n } ( q ) ) \geq$ $( p - q ) ^ { 2 } / 2$ . Since $L ^ { 2 k } = \dot { C } _ { 0 } ^ { 2 k } \Lambda ^ { k }$ , choosing the universal constant C suficiently large gives

$$
H ^ { 2 } ( \mathrm { B e r n } ( p _ { \star } ) , \mathrm { B e r n } ( q _ { \star } ) ) \geq \frac { \rho ^ { 2 } } { 2 } \geq \frac { \Delta ^ { 2 } } { ( C \Lambda ) ^ { k } } = \kappa .
$$

For every $b \geq L ,$ , the same threshold belongs to $[ - b , b ]$ and has gap at least $\rho \ge \Delta / ( 4 b ^ { k } )$ , proving that such a threshold exists whenever some degree $i \in [ k ]$ witnesses the moment discrepancy.

It remains to find such a threshold eficiently. We draw n samples from Q and sort them as $z _ { ( 1 ) } \leq$ $\cdots \leq z _ { ( n ) }$ . For each sample point $z _ { ( a ) } \in [ - L , L ]$ , the empirical probabilities of $z \geq z _ { ( a ) }$ and $z > z _ { ( a ) }$ are $( n - a + 1 ) / n$ and $( n - a ) / n ,$ respectively. We compare both with the Gaussian upper-tail probability $1 - \Phi ( z _ { ( a ) } )$ . We also check the endpoints −L and L using their empirical upper-tail probabilities, and choose the threshold $\widehat { t } \mathrm { g i v i n g }$ the largest absolute diference. Since $Q$ is absolutely continuous, it assigns zero probability to each point, so $Q ( z \geq t ) = Q ( z > t )$

Between consecutive sample points, the empirical upper-tail probability is constant and the Gaussian upper-tail probability is monotone. Thus, checking both endpoints and both sides of each sample point in the interval sufices to maximize the empirical discrepancy over $[ - L , L ]$

By the DKW inequality (see Fact B.2), for

$$
n = \left\lceil 8 \rho ^ { - 2 } \log \frac { 2 } { \delta } \right\rceil = O \left( \rho ^ { - 2 } \log \frac { 2 } { \delta } \right) ,
$$

with probability at least $1 - \delta$ , all one-sided empirical tail probabilities are within $\rho / 4$ of their population values. On this event, the empirical discrepancy at $t _ { \star }$ is at least $7 \rho / 4 - \rho / 4 = 3 \rho / 2$ . Consequently, the maximizing threshold satisfies

$$
\lvert p _ { \widehat { t } } - q _ { \widehat { t } } \rvert \geq 3 \rho / 2 - \rho / 4 = 5 \rho / 4 \geq \rho \geq \frac { \Delta } { 4 b ^ { k } } .
$$

It follows that

$$
H ^ { 2 } \left( \mathrm { B e r n } ( p _ { \hat { t } } ) , \mathrm { B e r n } ( q _ { \hat { t } } ) \right) \geq { \frac { \rho ^ { 2 } } { 2 } } \geq { \frac { \Delta ^ { 2 } } { ( C \Lambda ) ^ { k } } } = \kappa .
$$

Moreover, $\begin{array} { r } { | \widehat { t } | \leq L \leq b } \end{array}$ . Since $\rho = \Delta / ( 4 L ^ { k } )$ and $L ^ { 2 k } = C _ { 0 } ^ { 2 k } \Lambda ^ { k }$ , the claimed sample bound follows. Sorting the samples gives sample-polynomial running time. All estimates use the same k-dependent radius and apply to any witnessing degree $i \in [ k ]$ ; the algorithm does not need to know i. This completes the proof of Lemma 3.5. □

Given Lemma 3.5, we provide an overview of our main partitioning algorithm, Algorithm 1. The algorithm maintains a partition $\mathcal { U } _ { t }$ of $\mathbb { R } ^ { d }$ . At each iteration t, we select a subcollection $A _ { t } \subseteq \mathcal { U } _ { t }$ of cells that have not yet been certified to satisfy the requirements of Theorem 3.1. For each $R \in A _ { t }$ , we call the moment discrepancy finder to find a direction $\mathbf { v } \perp V _ { R }$ and then subdivide R into small intervals along v.

```latex
Algorithm 1 GaussianSpacePartition
1: Input: Samples from $Q$ and parameters $d , k , q , \sigma , \eta , \delta , \Delta .$ , a moment discrepancy finder $\mathcal { A } _ { m } ^ { \beta }$ (Corol
lary C.1).
2: Output: A partition $\mathcal { U }$ and certified cells ${ \mathcal { G } } .$
3: Set parameters according to (Parameter Choices of Algorithm 1), $\mathcal { U } _ { 1 } = \{ \mathbb { R } ^ { d } \} ; \mathcal { G } = \mathcal { B } = \mathcal { T } = \mathcal { E } = \emptyset .$
4: for $t = 1 , \dots , L$ do
5: $\mathcal { U } _ { t + 1 } = \varnothing .$
6: Draw $\lceil 8 \xi ^ { - 2 } \log ( 2 N _ { * } / \beta ) \rceil$ fresh samples from $Q ;$ let $\widehat { q } _ { R }$ be their empirical mass in each $R \in { \mathcal { U } } _ { t }$
7: for $R \in { \mathcal { U } } _ { t }$ do
8: if R is not terminal, classify it into B if ${ \widehat { q } } R \leq \xi / 2 .$
9: if $R \in \mathcal G \cup B \cup \mathcal T \cup \mathcal E$ then Add R to $\mathcal { U } _ { t + 1 }$
10: else
11: Run $\mathcal { A } _ { m } ^ { \beta }$ on the $V _ { R } ^ { \perp }$ projection of Q | R, with parameters $k , \Delta , V , H ;$ if $V _ { R } ^ { \perp } = \{ \mathbf { 0 } \}$ , take the
output to be None.
12: if the output is None then Add R to $\mathcal { G }$ and $\mathcal { U } _ { t + 1 }$
13: else if the output is Reject then Add R to E and $\mathcal { U } _ { t + 1 }$
14: else
15: Let $\mathbf { v } \in V _ { R } ^ { \perp }$ be the returned unit direction.
16: Obtain n fresh samples from $Q \mid R .$
17: Find a threshold $s \in [ - T , T ]$ maximizing the empirical Gaussian upper-tail gap in
direction v, as in Lemma 3.5.
18: if this maximum is below $2 g$ then Add R to $\mathcal { E }$ and $\mathcal { U } _ { t + 1 }$
19: else
20: Refine R along a maximizing threshold using Definition 3.4.
21: Add every child to $\mathcal { U } _ { t + 1 } ;$ Add the two tail children to $\tau .$
22: return $( \mathcal { U } _ { L + 1 } , \mathcal { G } )$
```

Equivalently, the algorithm builds a tree rooted at $\mathbb { R } ^ { d }$ , with each node corresponding to a cell created during its execution. The active cells in $A _ { t }$ are those split at iteration t, and their children lie one level deeper in the tree. The final partition consists of the cells corresponding to the leaves of this tree. Below, we give an overview of the analysis, which also explains some additional steps needed beyond this general template.

Bounding the Recursion Depth. The above structure implies that the key quantity to control during the analysis is the depth L of the tree at termination. Since each split creates many new cells, the total number of cells can grow exponentially with $L .$ The depth also controls the cell diameters: if a cell is defined by intervals of width at most $\mu$ along at most $L$ mutually orthogonal directions, then its diameter in its constraint subspace is at most $\mu \sqrt { L }$

The way we keep track of the depth is to keep track of a potential function $\begin{array} { r } { \Psi _ { t } : = \sum _ { R \in A _ { t } } \sqrt { p _ { R } q _ { R } } } \end{array}$ , where $p _ { R } = \mathbf { P r } _ { P } [ R ]$ and $q _ { R } = \mathbf { P r } _ { Q } [ R ]$ . As we will see later, under the smoothness condition, for most of the cells, the ratio $q _ { R } / p _ { R }$ can be controlled. Thus, $\textstyle \sum _ { R \in A _ { t } } q _ { R }$ can be controlled as a linear function of $\Psi _ { t }$ up to a small additive error. This implies that if we can show $\Psi _ { t }$ drops fast with respect to $t ,$ then we can stop when $\Psi _ { t }$ is small enough to make the tree have a reasonable depth.

To see why a fast decreasing rate is possible, suppose we split a single cell R into new cells $R _ { 1 } , \dots , R _ { M }$ Then we have $\begin{array} { r l r } {  { \sum _ { j = 1 } ^ { M } \sqrt { p _ { R _ { j } } q _ { R _ { j } } } = \sqrt { p _ { R } q _ { R } } \sum _ { j = 1 } ^ { M } \sqrt { P _ { R } ( R _ { j } ) Q _ { R } ( R _ { j } ) } } } \end{array}$ , where $P _ { R } ( Q _ { R } )$ is the distribution of $P ( Q )$ conditioned over $R$ . Notice that $\begin{array} { r } { 1 - \sum _ { j = 1 } ^ { M } \sqrt { P _ { R } ( R _ { j } ) Q _ { R } ( R _ { j } ) } } \end{array}$ is exactly the squared Hellinger distance between the distributions supported over $R _ { i } , i \in [ M ]$ with probability $P _ { R } , Q _ { R }$ . This implies that as long as the intervals we construct along a direction v can create a Hellinger distance at least $\kappa , \Psi _ { t }$ drops at a rate of $\exp ( - \kappa t )$ . This is exactly the outcome of Lemma 3.5.

Selecting Cells to Split. Thus, the criteria for selecting the cells in $A _ { t }$ to be split are clear. First, $Q \mid R$ must have a moment discrepancy relative to $P \mid R$ in some direction in $V _ { R } ^ { \perp }$ . Second, $q _ { R }$ must be suficiently large to allow eficient sampling from $Q \mid R$ . Third, R must not have any tail constraints, which would make its diameter unbounded. Finally, $Q \mid R$ must be suficiently smooth with respect to $P \mid R ,$ so that we can reliably run the moment discrepancy finder and find a good split that provides the progress needed to control the recursion depth.

Conditional Smoothness. Notice that the first three conditions can be checked using empirical estimates. However, verifying that $Q \mid R$ is suficiently smooth with respect to $P \mid R$ can be much harder, even when $Q$ itself is smooth with respect to $P .$ To get an intuition, we consider the special case of $q = 1$ where the smoothness is equivalent to the weight between $Q \mid R$ and $P \mid R .$ . In this case, since $Q$ has σ weight with respect to $P ,$ the weight between $Q \mid R$ and $P \mid R$ is exactly $\sigma q _ { R } / p _ { R }$ . This implies that we only need to check the ratio between $q _ { R }$ and $p _ { R }$ . And clearly, the total $Q$ mass of the cells that are not smooth cannot be too large. However, for the case of $q > 1$ , the situation is diferent. Since we only know that $Q ( S ) \leq P ( S ) ^ { 1 / q } / \sigma$ , for a set $S \subseteq R$ , even if $q _ { R } / p _ { R }$ is large, $( Q \mid _ { R } ( S ) ) / ( P \mid _ { R } ( S ) ^ { 1 / q } )$ still has a linear dependence on $p _ { R } ^ { 1 / q - 1 }$ . So, if $p _ { R }$ is small, potentially $Q \mid R$ can be highly unsmooth with respect to $P \mid R$ under the q-smooth definition. This yields two undesired consequences. First, without a smoothness guarantee, we may not be able to run the moment discrepancy finder reliably within the given computational budget. In particular, failure to return a direction does not necessarily imply that $Q \mid R$ and $P \ | \ R$ approximately match moments. So potentially, the moment discrepancy finder may miss the cells that are not good enough. Second, even if we have found a direction $\mathbf { v } ,$ along which the moment discrepancy exists, as discussed in Lemma 3.5, a quantitative good split could not exist due to the non-smoothness, which prevents us from controlling the progress and the depth of Algorithm 1. To avoid this, we notice that the success of Lemma 3.5 can be checked by looking at the empirical probability. If $\mathrm { a }$ moment discrepancy exists but we do not find a good split, then we can simply discard this cell, because of the unsmoothness. Furthermore, Proposition 3.3 indeed succeeds under a broader testable condition, where the expectation of Hermite tensor is not very large under $Q .$ . Since a smooth distribution must satisfy this condition, when we find the algorithm may not reliably succeed, we can simply discard this cell. The only remaining issue is to show that we will not discard too much Q-mass. Fortunately, in Lemma 3.11, we show that if Q is $( q , \sigma )$ -smooth with respect to P, then outside a collection of cells of small total Q-mass, $Q \mid R$ is smooth with respect to $P \mid R$ , with a slightly worse parameter, which means we will only discard a tiny $Q$ mass in this way.

To conclude the analysis, it remains to show that when Algorithm 1 stops, the fraction of cells that have low Q-mass or contain a tail inequality is small. In Lemma 3.9 and Lemma 3.10, we provide separate analyses.

Before the formal analysis, we first state the parameter choices of Algorithm 1.

Parameter Choices of Algorithm 1. Choose universal constants $C _ { 0 } , C > 1$ suficiently large, with $C _ { 0 }$ at least the constant required by Lemma 3.5. Define the error tolerances $\lambda = \varepsilon = \tau _ { 0 } = \eta / 1 0 0$ , the conditional smoothness parameter $\sigma _ { * } = \sigma \eta / ( 4 0 0 q )$ , the polynomial norm threshold $H = e ( 4 q \log ( 4 e q / \sigma _ { * } ) ) ^ { k / 2 }$ , the auxiliary tail parameter $\Lambda _ { * } = 2 q \left( k \log ( 4 q k ) + \log ( 6 4 / ( \sigma _ { * } \Delta ) ) \right)$ , the truncation bound $b = C _ { 0 } \sqrt { \Lambda _ { * } }$ , the probability-gap parameter $g = \Delta / ( 1 0 0 b ^ { k } )$ , and the potential contraction parameter $\kappa = g ^ { 2 } / 8$

Define the potential bound $M = 1 / ( \sigma ^ { q } \tau _ { 0 } ^ { q - 1 } )$ , the recursion depth bound $L = 1 + \left\lceil \kappa ^ { - 1 } \log ( 1 0 0 \sqrt { M } / \eta ) \right\rceil$ 2 the grid width $\delta _ { 0 } = \delta / \sqrt { L }$ , the tail cutof T = max $\left\{ b , \sqrt { 2 \log ( 2 0 0 L / ( \sigma \eta / 1 0 0 ) ^ { q } ) } \right\}$ , the grid-size bound $m = \lceil 2 T / \delta _ { 0 } \rceil + 3$ , and the small-mass potential exponent $s _ { 0 } = \kappa / ( 2 ( \kappa + \log m ) )$

Finally, define the Q-mass cutof $\xi = \big ( ( \sigma \varepsilon ) ^ { q } \lambda ^ { 1 / 2 + s _ { 0 } } \big ) ^ { 1 / ( 2 s _ { 0 } ) }$ , the cell-count bound $N _ { * } = ( m + 1 ) ^ { L + 1 }$ , the per-call failure probability $\beta = 1 / ( 1 0 0 0 ( N _ { * } + L + 1 ) )$ ), the polynomial variance bound $V = 8 q ( 4 q - 1 ) ^ { k } / ( \sigma \xi )$ and the CDF-estimation sample size $n _ { \mathrm { C D F } } = \left\lceil 8 g ^ { - 2 } \log ( 2 / \beta ) \right\rceil$

Mass Analysis of Active Cells. We first use a potential analysis in Lemma 3.7 to show that the Q-mass of the cells that are active during Algorithm 1 drops exponentially fast. We remark that instead of directly working with a smooth distribution $Q _ { i }$ , we relax the assumption and work with a distribution Q such that most of the Q-mass is supported over points whose density ratio $d Q / d P$ is bounded. As we show later in Lemma 3.8, any smooth distribution Q satisfies this assumption.

Lemma 3.7 (Mass of Active Cells Decreases Exponentially). Let $P = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ , let Q be $( q , \sigma )$ -smooth with respect to $P ,$ , where $q \geq 1$ and $\sigma \in ( 0 , 1 ]$ , and write $r = \mathrm { d } Q / \mathrm { d } P$ . Assume $M \geq 1$ and $\tau _ { 0 } \in [ 0 , 1 ]$ such that $Q ( r > M ) \leq \tau _ { 0 }$ . Let $g > 0$ be the split-validation parameter used by Algorithm 1. Let $L \in \mathbb { N } _ { + }$ and $\kappa \in ( 0 , 1 )$ satisfy $\kappa \leq g ^ { 2 } / 2$ . Consider Algorithm 1 through iteration L. Assume the event that all empirical $C D F s$ used to validate splits in these iterations difer from their population CDFs by at most $g / 4$ at every threshold.

For $t \in [ L ]$ , denote by $A _ { t }$ the cells that are split in round t, and write $p _ { R } = P ( R )$ and $q _ { R } = Q ( R )$ Then

$$
\sum _ { R \in A _ { t } } q _ { R } \leq \sqrt M e ^ { - \kappa ( t - 1 ) } + \tau _ { 0 } .
$$

In particular, for $\eta \in ( 0 , 1 )$ , setting $L = 1 + \lceil \kappa ^ { - 1 } \log ( 1 0 0 \sqrt { M } / \eta ) \rceil$ makes their total mass at most $\eta / 1 0 0 { + \tau _ { 0 } }$ Proof of Lemma 3.7. For $t \in [ L ]$ , define $\begin{array} { r } { \Psi _ { t } : = \sum _ { R \in A _ { t } } \sqrt { p _ { R } q _ { R } } } \end{array}$ . We first show that $\Psi _ { t + 1 } \leq ( 1 - \kappa ) \Psi _ { t }$ for $t < L$

By assumption all empirical probabilities used to validate splits are within $g / 4$ of their true values. Consider a split of a cell R in a unit direction $\mathbf { v } \in V _ { R } ^ { \perp }$ at threshold s. Since the constraints defining R depend only on $\mathrm { p r o j } _ { V _ { R } } \mathbf { x }$ , the distribution of $\mathbf { v } \cdot \mathbf { x }$ under $P \mid R$ remains $\mathcal { N } ( 0 , 1 )$ . The algorithm makes this split only if the empirical estimate of $\mathbf { P r } _ { \mathbf { x } \sim Q | R } [ \mathbf { v } \cdot \mathbf { x } \geq s ]$ difers from $\mathbf { P r } _ { z \sim \mathcal { N } ( 0 , 1 ) } [ z \geq s ]$ by at least $2 g$ . The corresponding true probabilities therefore difer by at least $2 g - g / 4 \geq g$ . The squared Hellinger distance between the resulting Bernoulli distributions is consequently at least $\frac { g ^ { 2 } } { 2 } \geq \kappa$ . Thus, every split made by the algorithm is κ-Gaussian good.

Fix $R \in A _ { t } .$ , and let $R _ { 1 } , \ldots , R _ { m }$ be all of its children, including the two terminal tail children. Let $P _ { R } , Q _ { R }$ be the distributions on these children, defined by $P _ { R } ( R _ { j } ) = P ( R _ { j } \mid R ) , Q _ { R } ( R _ { j } ) = Q ( R _ { j } \mid R ) , j \in [ m ]$ . Both distributions have total mass one. Since the full child partition refines the κ-good binary split,

$$
H ^ { 2 } ( P _ { R } , Q _ { R } ) = 1 - \sum _ { j = 1 } ^ { m } \sqrt { P _ { R } ( R _ { j } ) Q _ { R } ( R _ { j } ) } \geq \kappa .
$$

Indeed, the κ-good split partitions the space $\mathbb { R } ^ { d }$ restricted on R into two parts, creating two Bernoulli distributions, with a squared Hellinger distance at least κ. Since $P _ { R } , Q _ { R }$ come from a refinement of the good split, their Hellinger distance cannot be smaller. Consequently,

$$
\sum _ { j = 1 } ^ { m } \sqrt { p _ { R _ { j } } q _ { R _ { j } } } = \sqrt { p _ { R } q _ { R } } \sum _ { j = 1 } ^ { m } \sqrt { P _ { R } ( R _ { j } ) Q _ { R } ( R _ { j } ) } \le ( 1 - \kappa ) \sqrt { p _ { R } q _ { R } } .
$$

The sum over children that are split in the next round is no larger. All stopped or certified cells remain unchanged and are never retested. Thus, every cell in $A _ { t + 1 }$ is a child of a cell in $A _ { t }$ . Summing over $R \in A _ { t }$ gives $\Psi _ { t + 1 } \leq ( 1 - \kappa ) \Psi _ { t }$ . Since $\Psi _ { 1 } \leq 1$ , we obtain

$$
\Psi _ { t } \leq ( 1 - \kappa ) ^ { t - 1 } \leq e ^ { - \kappa ( t - 1 ) } .
$$

Now we will use the above potential to bound the $Q -$ mass of $A _ { t }$ . Let $b _ { R }$ be the $Q -$ -mass of points of R with a small density ratio, $\mathrm { i . e . }$

$$
b _ { R } : = \int _ { R } r \mathbb { 1 } \{ r \leq M \} \mathrm { d } P .
$$

Since $b _ { R } \leq M p _ { R }$ and $b _ { R } \leq q _ { R }$ , we have $b _ { R } ^ { 2 } \le M p _ { R } q _ { R }$ . Therefore,

$$
\begin{array} { r l r } {  { \sum _ { R \in { \cal A } _ { t } } q _ { R } = \sum _ { R \in { \cal A } _ { t } } b _ { R } + Q ( ( \bigcup _ { R \in { \cal A } _ { t } } R ) \cap \{ r > M \} ) } } \\ & { } & { \leq \sqrt { M } \sum _ { R \in { \cal A } _ { t } } \sqrt { p _ { R } q _ { R } } + \tau _ { 0 } \leq \sqrt { M } e ^ { - \kappa ( t - 1 ) } + \tau _ { 0 } . } \end{array}
$$

Every unprocessed leaf remaining after round L is a child of a cell in $A _ { L } ,$ , because every previously processed cell was either split or stopped permanently. The children partition their parents, so the total mass of these leaves is at most

$$
\sum _ { R \in A _ { L } } q _ { R } \le \sqrt { M } e ^ { - \kappa ( L - 1 ) } + \tau _ { 0 } .
$$

For the displayed choice of $L , \sqrt { M } e ^ { - \kappa ( L - 1 ) } \leq \eta / 1 0 0$ . This proves the lemma.

The following lemma shows that the density-ratio assumption in Lemma 3.7 follows from $( q , \sigma ) \mathrm { . }$ smoothness. For every $\epsilon \in ( 0 , 1 )$ , choosing $M = 1 / ( \sigma ^ { q } \epsilon ^ { q - 1 } )$ ensures that the density ratio is at most M outside a set of Q-mass at most ϵ.

Lemma 3.8 (Approximately Bounded Density Ratio). Let $P , Q$ be distributions on the same measurable space, with $Q \ ( q , \sigma )$ -smooth with respect to $P _ { i }$ where $q \geq 1$ and $\sigma \in ( 0 , 1 ]$ . Write $f = \mathrm { d } Q / \mathrm { d } P$ . For every $\varepsilon \in ( 0 , 1 )$ , the set

$$
G = \left\{ { \bf x } : f ( { \bf x } ) \leq \frac { 1 } { \sigma ^ { q } \varepsilon ^ { q - 1 } } \right\}
$$

satisfies $Q ( G ) \geq 1 - \varepsilon$ . Moreover, $Q \mid G$ is $\left( 1 , ( 1 - \varepsilon ) \sigma ^ { q } \varepsilon ^ { q - 1 } \right)$ -smooth with respect to $P _ { - }$

We defer the proof of Lemma 3.8 to Section C.

Mass Analysis of Discarded Cells. We next analyze the Q-mass of cells that will be discarded during the execution of Algorithm 1. In particular, these are cells whose defining constraints include an unbounded interval and cells with small Q-mass.

First, we bound the total Q-mass of cells whose defining constraints include a tail segment (the collection T in Algorithm 1).

Lemma 3.9 (Negligible Mass of Gaussian Tail Cells). Let $P = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ , let Q be $( q , \sigma )$ -smooth with respect to P, where $q \geq 1$ and $\sigma \in ( 0 , 1 ]$ , and let $L \in \mathbb { N } _ { + }$ and $\eta \in ( 0 , 1 )$ . Choose $T > 0$ satisfying $T ^ { 2 } \geq 2 \log ( 2 0 0 L / ( \sigma \eta / 1 0 0 ) ^ { q } )$ . Consider Algorithm 1 through iteration L. Then, for every $t \leq L + 1$

$$
\sum _ { R \in \mathcal { T } _ { t } } q _ { R } \leq \frac { ( 2 L e ^ { - T ^ { 2 } / 2 } ) ^ { 1 / q } } { \sigma } \leq \frac { \eta } { 1 0 0 } .
$$

Proof of Lemma 3.9. Tail cells are carried forward unchanged, so $\mathcal T _ { t } \subseteq \mathcal T _ { t + 1 }$ . It therefore sufices to bound the total mass of $\mathcal { T } _ { L + 1 }$

Let $A _ { s }$ be the cells split in round s. Consider a set that will be split in round $s , R \in A _ { s }$ and let $\mathbf { v } _ { R }$ be its split unit direction. Denote the tail children by $E _ { R } : = R \cap \left( \left\{ \mathbf { x } : \mathbf { v } _ { R } \cdot \mathbf { x } < - T \right\} \cup \left\{ \mathbf { x } : \mathbf { v } _ { R } \cdot \mathbf { x } \geq T \right\} \right)$

Since $\mathbf { v } _ { R } \in V _ { R } ^ { \perp }$ by Gaussian independence

$$
P ( E _ { R } ) = 2 P ( R ) \Phi ( - T ) \leq 2 P ( R ) e ^ { - T ^ { 2 } / 2 } .
$$

Let $\begin{array} { r } { E = \bigcup _ { R \in \mathcal { T } _ { L + 1 } } R } \end{array}$ . Using the above bound we have

$$
\begin{array} { l } { { \displaystyle P ( E ) \le \sum _ { s = 1 } ^ { L } \sum _ { R \in A _ { s } } P ( E _ { R } ) } } \\ { { \displaystyle \phantom { \sum _ { s = 1 } ^ { L } \sum _ { R \in A _ { s } } ^ { L } P ( E _ { R } ) } \le 2 e ^ { - T ^ { 2 } / 2 } \sum _ { s = 1 } ^ { L } \sum _ { R \in A _ { s } } P ( R ) \le 2 L e ^ { - T ^ { 2 } / 2 } \ : , } } \end{array}
$$

where we used that the cells in $A _ { s }$ are disjoint. Applying smoothness to the event $E$

$$
\sum _ { R \in \mathcal { T } _ { L + 1 } } q _ { R } = Q ( E ) \leq \frac { P ( E ) ^ { 1 / q } } { \sigma } \leq \frac { ( 2 L e ^ { - T ^ { 2 } / 2 } ) ^ { 1 / q } } { \sigma } .
$$

The choice of $T$ gives

$$
2 L e ^ { - T ^ { 2 } / 2 } \leq \frac { 1 } { 1 0 0 } \left( \frac { \sigma \eta } { 1 0 0 } \right) ^ { q } \leq \left( \frac { \sigma \eta } { 1 0 0 } \right) ^ { q } .
$$

Consequently, $Q ( E ) \leq \eta / 1 0 0$ , which concludes the proof.

Second, we bound the total Q-mass of cells with small individual Q-mass (the collection B in Algorithm 1).

Lemma 3.10 (Negligible Total Mass of Low-Q-Mass Cells). Let Q be $( q , \sigma )$ -smooth with respect to P. For $t \leq L + 1$ , consider the partition U of Algorithm 1, and let $B _ { t } = \{ R \in \mathcal { U } _ { t } : Q ( R ) \leq \xi \}$ . Let $m \geq 2$ bound the number of children in every refinement, and set $s = \kappa / ( 2 ( \kappa + \log m ) )$ . Assuming that every executed split is κ-Gaussian good, for every $\lambda \in ( 0 , 1 )$ ，

$$
\sum _ { R \in \mathcal { B } _ { t } } Q ( R ) \leq \lambda + \frac { 1 } { \sigma } \left( \frac { \xi ^ { 2 s } } { \lambda ^ { 1 / 2 + s } } \right) ^ { 1 / q } .
$$

In particular, $f o r \varepsilon \in ( 0 , 1 )$ , the choice $\xi = \left( ( \sigma \varepsilon ) ^ { q } \lambda ^ { 1 / 2 + s } \right) ^ { 1 / ( 2 s ) }$ makes this total Q-mass at most $\lambda + \varepsilon .$

Proof of Lemma 3.10. Write $p _ { R } = P ( R )$ and $q _ { R } = Q ( R )$ . For each iteration $t ,$ define the potential

$$
F _ { t } : = \sum _ { R \in \mathcal { U } _ { t } } \left( p _ { R } q _ { R } \right) ^ { 1 / 2 - s } .
$$

Since $\mathcal { U } _ { 1 } = \{ \mathbb { R } ^ { d } \}$ , we have $F _ { 1 } = 1$ . We show that a refinement cannot increase this potential. Consider a cell R split into $R _ { 1 } , \ldots , R _ { j }$ , where $j \leq m$ , and put $p _ { i } = P ( R _ { i } \mid R )$ and $q _ { i } = Q ( R _ { i } \mid R )$ . All children, including both terminal tails, are retained in the partition, so $( p _ { i } ) _ { i = 1 } ^ { j }$ and $( q _ { i } ) _ { i = 1 } ^ { j }$ are probability vectors. The refinement contains the validated binary split and hence by Fact B.1 satisfies

$$
\sum _ { i = 1 } ^ { j } { \sqrt { p _ { i } q _ { i } } } \leq 1 - \kappa .
$$

Since $0 < 1 - 2 s < 1$ , concavity gives

$$
\sum _ { i = 1 } ^ { j } ( p _ { i } q _ { i } ) ^ { 1 / 2 - s } \leq j ^ { 2 s } \left( \sum _ { i = 1 } ^ { j } { \sqrt { p _ { i } q _ { i } } } \right) ^ { 1 - 2 s } \leq m ^ { 2 s } ( 1 - \kappa ) ^ { 1 - 2 s } \leq 1 .
$$

The last inequality follows since by the definition of $s ,$ 2s log $m + ( 1 - 2 s ) \log ( 1 - \kappa ) \leq 2 s \log m - ( 1 - 2 s ) \kappa = 0$ Consequently,

$$
\sum _ { i = 1 } ^ { j } ( p _ { R _ { i } } q _ { R _ { i } } ) ^ { 1 / 2 - s } \leq ( p _ { R } q _ { R } ) ^ { 1 / 2 - s } .
$$

Stopped and certified cells are carried forward unchanged. Applying the preceding inequality to each split in an iteration proves $F _ { t + 1 } \leq F _ { t }$ , hence $F _ { t } \leq 1$ for every t.

Let $B _ { t } ^ { - }$ consist of the cells in $B _ { t }$ with $q _ { R } \leq \lambda p _ { R }$ , and put $B _ { t } ^ { + } = B _ { t } \setminus B _ { t } ^ { - }$ . Disjointness implies

$$
\sum _ { R \in \mathsf { B } _ { t } ^ { - } } q _ { R } \leq \lambda \sum _ { R \in \mathsf { B } _ { t } ^ { - } } p _ { R } \leq \lambda .
$$

For each $R \in B _ { t } ^ { + }$ we have $p _ { R } < q _ { R } / \lambda \leq \xi / \lambda$ and $( p _ { R } q _ { R } ) ^ { 1 / 2 - s } \ge \lambda ^ { 1 / 2 - s } p _ { R } ^ { 1 - 2 s }$ . Therefore,

$$
P \left( \bigcup _ { R \in \mathcal { B } _ { t } ^ { + } } R \right) = \sum _ { R \in \mathcal { B } _ { t } ^ { + } } p _ { R } ^ { 1 - 2 s } p _ { R } ^ { 2 s } \leq ( \xi / \lambda ) ^ { 2 s } \sum _ { R \in \mathcal { B } _ { t } ^ { + } } p _ { R } ^ { 1 - 2 s } \leq ( \xi / \lambda ) ^ { 2 s } \lambda ^ { - ( 1 / 2 - s ) } F _ { t } \leq \frac { \xi ^ { 2 s } } { \lambda ^ { 1 / 2 + s } } .
$$

Apply global smoothness once to this union of cells to obtain

$$
\sum _ { R \in \mathcal { B } _ { t } ^ { + } } q _ { R } \leq \frac { 1 } { \sigma } \left( \frac { \xi ^ { 2 s } } { \lambda ^ { 1 / 2 + s } } \right) ^ { 1 / q } .
$$

Adding the bounds for the two collections proves the claim. Substitution of the choice of $\xi$ completes the proof. □

Conditional Smoothness and Polynomial Moment Bounds. Finally, conditioning on a cell R can worsen the smoothness parameters, so the moment discrepancy finder may return Reject. We prove two lemmas to show that cells rejected in this way have small total Q-mass. First, Lemma 3.11 shows that if $Q$ is smooth with respect to P, then, for every partition, most of the Q-mass lies in cells R such that $Q \mid R$ is smooth with respect to $P \mid R _ { : }$ , with slightly worse smoothness parameters. Second, Lemma 3.13 bounds the second moments of low-degree polynomials and the norms of the Hermite moment tensors under a smooth distribution, providing the bounds required by the moment discrepancy finder.

Lemma 3.11 (Conditional smoothness on most cells). Let $P , Q$ be probability distributions on $\mathbb { R } ^ { d }$ , let $q \geq 1$ and $\sigma , \eta \in ( 0 , 1 )$ , and suppose that $Q$ is $( q , \sigma )$ -smooth with respect to P. For every finite measurable partition P of $\mathbb { R } ^ { d }$ , there is a collection $\mathcal { G } \subseteq \mathcal { P }$ of cells of positive Q-mass such that $\begin{array} { r } { \sum _ { R \in \mathcal { G } } Q ( R ) \geq 1 - \eta } \end{array}$ and, for every $R \in { \mathcal { G } }$ , the conditional distribution $Q \mid R$ is $( 2 q , \sigma \eta / ( 2 q ) )$ -smooth with respect to $P \mid R$

Proof of Lemma 3.11. Smoothness implies that $Q \ll P$ , hence the density ratio $r = \mathrm { d } Q / \mathrm { d } P$ is defined. Let $a = 2 q / ( 2 q - 1 )$ . We first show that r has bounded $L _ { a }$ norm under $P .$

Claim 3.12. It holds that $\| r \| _ { L _ { a } ( P ) } \leq 2 q / \sigma$

Proof of Claim 3.12. For $q = 1$ , this follows from $r \leq 1 / \sigma$ almost surely. For $q > 1$ , smoothness gives

$$
t P ( r > t ) \leq Q ( r > t ) \leq \frac { P ( r > t ) ^ { 1 / q } } { \sigma } ~ .
$$

Hence $P ( r > t ) \leq \operatorname* { m i n } \{ 1 , ( \sigma t ) ^ { - b } \}$ , for $b = q / ( q - 1 )$ . Integrating this tail bound yields

$$
\operatorname { E } [ r ^ { a } ] = a \int _ { 0 } ^ { \infty } t ^ { a - 1 } P ( r > t ) \mathrm { d } t \le a \int _ { 0 } ^ { 1 / \sigma } t ^ { a - 1 } \mathrm { d } t + a \sigma ^ { - b } \int _ { 1 / \sigma } ^ { \infty } t ^ { a - b - 1 } \mathrm { d } t = \sigma ^ { - a } \left( 1 + { \frac { a } { b - a } } \right) = ( 2 q - 1 ) \sigma ^ { - a } .
$$

This proves that $\| r \| _ { L _ { a } ( P ) } \leq 2 q / \sigma$ and completes the proof of Claim 3.12.

For each cell with $q _ { R } : = Q ( R ) > 0$ , we have that $p _ { R } : = P ( R ) > 0$ . Define the conditional density ratio $s _ { R } ( x ) : = \mathrm { d } ( Q \mid R ) / \mathrm { d } ( P \mid R ) ( x ) = ( p _ { R } / q _ { R } ) r ( x )$ for $x \in R$

By Hölder’s inequality and disjointness of the cells,

$$
\begin{array} { r l r } {  { \sum _ { R : q _ { R } > 0 } q _ { R } \| s _ { R } \| _ { L _ { a } ( P | R ) } = \displaystyle \sum _ { R : q _ { R } > 0 } p _ { R } ^ { 1 - 1 / a } ( \int _ { R } r ^ { a } \mathrm { d } P ) ^ { 1 / a } } } \\ & { } & { \leq ( \displaystyle \sum _ { R : q _ { R } > 0 } p _ { R } ) ^ { 1 - 1 / a } ( \sum _ { R : q _ { R } > 0 } \int _ { R } r ^ { a } \mathrm { d } P ) ^ { 1 / a } } \\ & { } & { \leq \| r \| _ { L _ { a } ( P ) } \leq \frac { 2 q } { \sigma } . } \end{array}
$$

Consequently, the cells satisfying $\| s _ { R } \| _ { L _ { a } ( P | R ) } > 2 q / ( \sigma \eta )$ have total $Q \cdot$ -mass at most $\eta .$

Let $\mathcal { G }$ consist of the remaining cells of positive Q-mass. For every $R \in { \mathcal { G } }$ and measurable event $F _ { ; }$ another application of Hölder’s inequality gives

$$
\begin{array} { r l } & { ( Q \mid R ) ( F ) = \displaystyle \int _ { F } s _ { R } \mathrm { d } ( P \mid R ) \le \left( \int _ { F } s _ { R } ^ { a } \mathrm { d } ( P \mid R ) \right) ^ { 1 / a } \left( \int _ { F } 1 \mathrm { d } ( P \mid R ) \right) ^ { 1 - 1 / a } } \\ & { \qquad \le \| s _ { R } \| _ { L _ { a } ( P \mid R ) } ( P \mid R ) ( F ) ^ { 1 - 1 / a } \le \frac { 2 q } { \sigma \eta } ( P \mid R ) ( F ) ^ { 1 / ( 2 q ) } . } \end{array}
$$

This completes the proof of Lemma 3.11.

We next show that under the smoothness assumption, we can bound the norm of a fixed degree polynomial $p$ under $Q$ in terms of its norm under the standard Gaussian distribution.

Lemma 3.13 (Polynomial Norm under Smoothness). Let $P = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ , let Q be $( q , \sigma )$ -smooth with respect to $P _ { \varepsilon }$ where $q \geq 1$ and $\sigma \in ( 0 , 1 ]$ , and let $k \in \mathbb { N } _ { + }$ . For every polynomial p of degree at most $k$

$$
\underset { Q } { \mathbf { E } } [ p ^ { 2 } ] \leq \frac { 2 q } { \sigma } ( 4 q - 1 ) ^ { k } \Vert p \Vert _ { L _ { 2 } ( P ) } ^ { 2 } .
$$

Moreover, for the normalized Hermite tensors $\mathcal { H } _ { i } , \mathrm { ~ } i \in [ k ]$ , we have

$$
\| \mathbf { \Delta E } [ \mathcal { H } _ { i } ] \| _ { F } \leq e \left( 2 q \log \frac { 2 e q } { \sigma } \right) ^ { i / 2 } \leq e \left( 2 q \log \frac { 2 e q } { \sigma } \right) ^ { k / 2 } .
$$

Proof of Lemma 3.13. Smoothness implies $Q \ll P$ , therefore we can define the density ratio $r = \mathrm { d } Q / \mathrm { d } P$ Let $a = 2 q / ( 2 q - 1 )$ . By Claim 3.12, $\| r \| _ { L _ { a } ( P ) } \leq 2 q / \sigma$ . Therefore, Hölder’s inequality and Gaussian hypercontractivity give

$$
\begin{array} { r l r } {  { \mathbf { E } [ p ^ { 2 } ] = \mathbf { E } [ r p ^ { 2 } ] } } \\ & { } & { \leq \| r \| _ { L _ { a } ( P ) } \| p ^ { 2 } \| _ { L _ { 2 q } ( P ) } = \| r \| _ { L _ { a } ( P ) } \| p \| _ { L _ { 4 q } ( P ) } ^ { 2 } } \\ & { } & { \leq \frac { 2 q } { \sigma } ( 4 q - 1 ) ^ { k } \| p \| _ { L _ { 2 } ( P ) } ^ { 2 } , } \end{array}
$$

where we used that $1 / a + 1 / ( 2 q ) = 1$ , which proves the first assertion.

For the tensor bound, fix $i \in [ k ]$ . For every order-i tensor T with $\| \mathbf { T } \| _ { F } = 1$ , define $p _ { \mathbf { T } } ( \mathbf { x } ) = \mathbf { T } \cdot \mathcal { H } _ { i } ( \mathbf { x } )$ This polynomial has degree at most i and $\| p _ { \mathbf { T } } \| _ { L _ { 2 } ( P ) } \leq 1$ . By duality of the Frobenius norm,

$$
\| \mathbf { \frac { E } { Q } } [ \mathcal { H } _ { i } ] \| _ { F } = \operatorname* { s u p } _ { \| \mathbf { T } \| _ { F } = 1 } | \mathbf { T } \cdot \mathbf { \frac { E } { Q } } [ \mathcal { H } _ { i } ] | = \operatorname* { s u p } _ { \| \mathbf { T } \| _ { F } = 1 } | \mathbf { \frac { E } { Q } } [ p _ { \mathbf { T } } ] | .
$$

Thus, it sufices to bound expectations under $Q$ of polynomials of degree at most i whose Gaussian $L _ { 2 }$ norm is at most 1.

Since $\| r \| _ { L _ { 1 } ( P ) } = 1$ and $\| r \| _ { L _ { a } ( P ) } \leq 2 q / \sigma$ , interpolation gives, for every $t \geq 2 q$

$$
\| r \| _ { L _ { t / ( t - 1 ) } ( P ) } \leq \| r \| _ { L _ { 1 } ( P ) } ^ { 1 - 2 q / t } \| r \| _ { L _ { a } ( P ) } ^ { 2 q / t } \leq \left( \frac { 2 q } { \sigma } \right) ^ { 2 q / t } .
$$

For completeness, if $c = t / ( t - 1 )$ and $\theta = ( c - 1 ) / ( a - 1 )$ , then $c = ( 1 - \theta ) + a \theta$ . Hölder’s inequality applied to $r ^ { c } = r ^ { 1 - \theta } ( r ^ { a } ) ^ { \theta }$ gives this interpolation inequality after taking cth roots, since aθ $\% c = 2 q / t$

Choose $t = 2 q \log ( 2 e q / \sigma ) \geq 2 q \geq 2$ . Then $( 2 q / \sigma ) ^ { 2 q / t } = \exp ( \log ( 2 q / \sigma ) / \log ( 2 e q / \sigma ) ) \leq e$ . For every polynomial p of degree at most $i ,$ Hölder’s inequality and Gaussian hypercontractivity give

$$
\begin{array} { r l } & { | \underset { Q } { \mathbf { E } } [ p ] | \leq \underset { Q } { \mathbf { E } } [ | p | ] = \underset { P } { \mathbf { E } } [ r | p | ] } \\ & { \qquad \leq \| r \| _ { L _ { t / ( t - 1 ) } ( P ) } \| p \| _ { L _ { t } ( P ) } } \\ & { \qquad \leq e ( t - 1 ) ^ { i / 2 } \| p \| _ { L _ { 2 } ( P ) } \leq e t ^ { i / 2 } \| p \| _ { L _ { 2 } ( P ) } . } \end{array}
$$

Applying this bound to $p _ { \mathbf { T } }$ and taking the supremum over $\| \mathbf { T } \| _ { F } = 1$ yields

$$
\| \mathbf { E } [ \mathcal { H } _ { i } ] \| _ { F } \leq e t ^ { i / 2 } = e \left( 2 q \log \frac { 2 e q } { \sigma } \right) ^ { i / 2 } .
$$

This concludes the proof of Lemma 3.13.

## 3.2 Proof of the Partitioning Theorem

Proof of Theorem 3.1. Run Algorithm 1, and write $p _ { R } : = P ( R )$ and $q _ { R } : = Q ( R )$ . Let $\mathcal { U } _ { L + 1 }$ denote the final partition and G the collection of certified cells. Let $B , \tau$ , and E denote the collections of cells stopped by the mass test, tail rule, and either finder rejection or failed split validation, respectively. Finally, let $\mathcal { D } = \mathcal { U } _ { L + 1 } \setminus ( \mathcal { G } \cup B \cup \mathcal { T } \cup \mathcal { E } )$ denote the leaves still unprocessed after round L.

We prove the result by first showing that every certified cell $R \in { \mathcal { G } }$ satisfies the moment-matching and small-diameter properties and has nontrivial Q-mass, with the parameters specified in the statement. It then remains to show that the remaining cells in $\mathcal { U } _ { L + 1 } \setminus \mathcal { G }$ have total Q-mass at most $\eta .$

We work on the simultaneous success event Ω on which all empirical cell masses have error at most $\xi / 4$ every amplified finder satisfies the conclusions of Corollary C.1, every fresh empirical CDF has uniform error at most $g / 4$ , and every conditional sampling budget succeeds. We verify later in the proof that $\mathbf { P r } [ \Omega ] \geq 9 9 / 1 0 0$

Claim 3.14 (Guarantees on Certified Cells). Every $R \in { \mathcal { G } }$ satisfies $q _ { R } > \xi / 4$ and diam $( R ) \le \delta _ { 0 } \sqrt { L }$ Moreover, $P \mid R$ and $Q \mid R$ match their first k moments on $V _ { R } ^ { \perp }$ with slack $\Delta$

Proof of Claim 3.14. A cell reaches the finder only after passing the empirical mass test. On Ω, this implies $q _ { R } \geq \widehat { q } _ { R } - \xi / 4 > \xi / 4$ . We establish the finder’s second-moment assumption uniformly for every such cell. For a polynomial $p$ on $V _ { R } ^ { \perp }$ , its extension $p ( \operatorname { p r o j } _ { V _ { R } ^ { \perp } } \mathbf { x } )$ has the same Gaussian $L _ { 2 }$ norm. By Lemma 3.13, for every such polynomial of degree at most k,

$$
\underset { Q | R } { \mathbf { E } } [ p ^ { 2 } ] \leq \frac { \mathbf { E } _ { Q } [ p ^ { 2 } ] } { q _ { R } } \leq V \| p \| _ { L _ { 2 } ( \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { V _ { R } ^ { \perp } } ) ) } ^ { 2 } ,
$$

where $\begin{array} { r } { V = \frac { 8 q } { \sigma \xi } ( 4 q - 1 ) ^ { k } } \end{array}$ , using $q _ { R } > \xi / 4$ . This single bound applies to every finder input at every depth. Membership in R depends only on $V _ { R } ,$ , so Gaussian independence implies that the projection of $P \mid R$ onto $V _ { R } ^ { \perp }$ is standard Gaussian. A cell is certified only when the amplified finder returns None; hence Corollary C.1 gives the asserted moment matching for the full distribution $Q \mid R$ . If $V _ { R } ^ { \perp } = \{ \mathbf { 0 } \}$ , the assertion is vacuous, as in Definition 2.2.

Every new constraint direction is a unit vector orthogonal to the preceding constraint subspace. The directions along a root-to-leaf path are therefore orthonormal. By Definition 3.4, each constrained coordinate of a nontail leaf lies in an interval of width at most $\delta _ { 0 }$ . Certified leaves are nontail and have at most L constrained coordinates, giving diam $( R ) \leq \delta _ { 0 } \sqrt { L }$ . This concludes the proof of Claim 3.14. □

It remains to bound the mass of the uncertified final leaves. We first control the unprocessed, low-mass, and tail leaves.

Claim 3.15 (Mass of $\mathcal { D } , B , \tau )$ . The final collections satisfy $\begin{array} { r } { \sum _ { R \in { \mathcal { D } } \cup { \mathcal { B } } \cup { \mathcal { T } } } q _ { R } \leq \frac { 5 \eta } { 1 0 0 } } \end{array}$

Proof of Claim 3.15. Every executed split has empirical CDF gap at least $2 g$ . On Ω, its true gap is at least $2 g - g / 4 \geq g$ . The reference marginal is standard Gaussian, and the squared Hellinger distance between Bernoulli distributions is at least half the squared diference of their parameters. Thus every executed split is κ-Gaussian good, since $g ^ { 2 } / 2 \geq \kappa ;$ see Definition 3.4.

Write $r = \mathrm { d } Q / \mathrm { d } P$ . By Lemma 3.8, $M = 1 / ( \sigma ^ { q } \tau _ { 0 } ^ { q - 1 } )$ satisfies $Q ( r > M ) \leq \tau _ { 0 }$ . Every leaf in $\mathcal { D }$ is a child of a cell split in round L. Hence Lemma 3.7 and the choice of L give

$$
\sum _ { R \in \cal D } q _ { R } \le \sqrt { M } e ^ { - \kappa ( L - 1 ) } + \tau _ { 0 } \le \eta / 1 0 0 + \tau _ { 0 } = 2 \eta / 1 0 0 .
$$

On Ω, every cell in B has $q _ { R } \leq 3 \xi / 4 < \xi$ . Since all executed splits are κ-Gaussian good, Lemma 3.10 applies to $\mathcal { U } _ { L + 1 }$ with $s = s _ { 0 }$ . The choices of $\xi , \lambda$ , and ε therefore give

$$
\sum _ { R \in \mathcal { B } } q _ { R } \leq \lambda + \varepsilon = 2 \eta / 1 0 0 .
$$

Applying Lemma 3.9 with our chosen value of $T$ gives $\begin{array} { r } { \sum _ { R \in \mathcal { T } } q _ { R } \leq \sigma ^ { - 1 } ( 2 L e ^ { - T ^ { 2 } / 2 } ) ^ { 1 / q } \leq \eta / 1 0 0 } \end{array}$ . Combining the above bounds concludes the proof of Claim 3.15. □

Claim 3.16 (Mass Stopped by Rejection or Failed Split Validation). The remaining terminal collection satisfies $\sum _ { R \in \mathcal { E } } q _ { R } \leq \eta / 1 0 0$

Proof of Claim 3.16. Apply Lemma 3.11 to the final partition $\mathcal { U } _ { L + 1 }$ , with parameters $( q , \sigma / 2 , \eta / 1 0 0 )$ . This is valid because $Q$ is also $( q , \sigma / 2 )$ -smooth and $\sigma / 2 < 1$ . It gives a collection $S \subseteq \mathcal { U } _ { L + 1 }$ such that

$$
\sum _ { R \in \mathcal { S } } q _ { R } \geq 1 - \eta / 1 0 0 , \qquad Q \mid R \mathrm { ~ i s ~ } ( 2 q , \sigma _ { * } ) \mathrm { - s m o o t h ~ w i t h ~ r e s p e c t ~ t o ~ } P \mid R \quad ( R \in \mathcal { S } ) ,
$$

where $\sigma _ { * } : = \sigma \eta / ( 4 0 0 q )$ . We show that $\mathcal { E } \cap \mathcal { S } = \emptyset$

Suppose $R \in { \mathcal { E } } \cap { \mathcal { S } }$ . The cell was tested at an earlier round and then carried forward unchanged, so its conditional distribution and constraint subspace at that test are the same as in the final partition. By Fact B.4 smoothness is preserved by projection. Since the projection of $P \mid R$ onto $V _ { R } ^ { \perp }$ is standard Gaussian, the corresponding projection of $Q \mid R$ is $( 2 q , \sigma _ { * } )$ -smooth. Lemma 3.13 therefore gives, for each $i \leq k$

$$
\left\| \underset { Q | R } { \mathbf { E } } [ \mathcal { H } _ { i } ( \mathrm { p r o j } _ { V _ { R } ^ { \perp } } \mathbf { x } ) ] \right\| _ { F } \leq e \left( 4 q \log \frac { 4 e q } { \sigma _ { * } } \right) ^ { i / 2 } \leq H .
$$

By the third conclusion of Corollary C.1, the finder cannot reject this cell.

The cell must therefore have been stopped by failed split validation after the finder returned a direction. By Corollary C.1, this direction has raw moment discrepancy at least $\Delta / 4$ in some degree in [k]. Applying Lemma 3.5 with parameters $( 2 q , \sigma _ { * } )$ and discrepancy $\Delta / 4$ gives a threshold $u _ { * } \in [ - b , b ]$ whose Gaussian upper-tail gap is at least $\Delta / ( 1 6 b ^ { k } ) \geq 6 g$ . Because $T \geq b$ and the empirical CDF error is at most $g / 4$ , the empirical maximum searched by Algorithm 1 is at least $6 g - g / 4 > 2 g$ . Thus split validation also succeeds, a contradiction. We conclude that ${ \mathcal { E } } \subseteq { \mathcal { U } } _ { L + 1 } \setminus S$ , which proves Claim 3.16. □

All uncertified final cells belong to $B \cup \mathcal { T } \cup \mathcal { E } \cup \mathcal { D }$ . The preceding bounds give

$$
Q \left( \bigcup _ { R \in \mathcal { U } _ { L + 1 } \backslash \mathcal { G } } R \right) \leq \frac { ( 2 + 2 + 1 + 1 ) \eta } { 1 0 0 } = \frac { 6 \eta } { 1 0 0 } < \eta .
$$

Together with Claim 3.14, this establishes the structural and moment-matching guarantees. By Defini tion 3.4, every split creates at most m children, so at most $N _ { * } : = ( 1 + m ) ^ { L + 1 }$ cells are created. We now verify the probability of Ω and the quantitative mass, diameter, and complexity bounds.

Claim 3.17 (Probability of Success Event). With the parameters in (Parameter Choices of Algorithm 1), the event Ω satisfies $\mathbf { P r } [ \Omega ] \geq 1 - ( L + 4 N _ { * } ) \beta \geq 9 9 / 1 0 0$

Proof of Claim 3.17. There are at most N<sub>∗</sub> calls to the moment discrepancy finding algorithm, N<sub>∗</sub> CDF validation stages to check whether the found split is good enough, and $2 N _ { * }$ conditional sampling stages from the cells R created during the implementation.

Condition successively on the preceding execution, up to the first failure. Fresh samples and Hoefding’s inequality give failure probability at most $\beta / N _ { * }$ per mass estimate, hence at most $L \beta$ in total. While these estimates are accurate, the uniform V-bound established in Claim 3.14 applies to every finder input. Thus Corollary C.1 bounds finder failures by $N _ { * } \beta$ in total. The Dvoretzky–Kiefer–Wolfowitz inequality similarly bounds CDF estimation failures by $N _ { * } \beta$

Couple each conditional sampling stage with an unlimited rejection-sampling stream, whose accepted observations are independent samples from ${ \cal Q } \ | \ { \cal R }$ . Since every tested cell has $q _ { R } > \xi / 4$ , a binomial lower-tail bound and the prescribed sampling budget give probability at most $\beta$ of obtaining too few observations. These failures total at most $2 N _ { * } \beta$ . Summing the failure bounds proves Claim 3.17. □

Complexity. Set

$$
\mathsf { E } = \frac { O ( ( q k ) ^ { O ( k ) } ) } { \Delta ^ { 3 } } \log ^ { k + 2 } \frac { 2 } { \sigma \eta } \log \frac { 2 } { \delta _ { 0 } } .
$$

Expanding (Parameter Choices of Algorithm 1) gives

$$
L \leq \frac { O ( ( q k ) ^ { O ( k ) } ) } { \Delta ^ { 3 } } \log ^ { k + 1 } \frac { 2 } { \sigma \eta } , \qquad \log \frac { N _ { * } V } { \xi g \beta } + k ^ { O ( k ) } \frac { H ^ { 2 } } { \Delta ^ { 2 } } \log \frac { H } { \Delta } \leq \mathsf { E } .
$$

By Definition 3.4, at most $N _ { * }$ cells are created, each requiring at most one finder call and one CDF validation stage. By Corollary C.1, each finder call uses at most $d ^ { k } \exp ( O ( \mathsf { E } ) )$ conditional samples and $d ^ { O ( k ) } \exp ( O ( \mathsf { E } ) )$ time, for every $H \geq 1$ . Each CDF stage uses $O ( g ^ { - 2 } \log ( 2 / \beta ) )$ conditional samples. Rejection sampling contributes a factor $O ( 1 / \xi )$ and logarithmic overhead, while mass estimation uses $O ( L \xi ^ { - 2 } \log ( 2 N _ { * } / \beta ) )$ additional draws. Routing samples, computing Gaussian cell probabilities, and scanning empirical CDFs add polynomial overhead. Thus, the total sample complexity is at most $d ^ { k } \exp ( O ( \mathsf { E } ) )$ and the running time is at most $d ^ { O ( k ) } \exp ( O ( \mathsf { E } ) )$

Finally, by the choice $\delta _ { 0 } = \delta / \sqrt { L }$ , diam $( R ) \leq \delta _ { 0 } \sqrt { L } = \delta$ for every $R \in { \mathcal { G } }$ , and the preceding bound give the stated sample and running-time complexities, $N _ { * } \leq B$ , and $\xi / 4 \ge 1 / B$ , upon choosing the implicit constants in B suficiently large. Together with Claim 3.17, this establishes all the stated guarantees. This completes the proof of Theorem 3.1. □

## 3.3 Eficient Moment Discrepancy Finding Algorithm

We now describe the moment discrepancy finding algorithm for Proposition 3.3. In Corollary C.1, we show how to boost Proposition 3.3 into a high probability regime.

Algorithm 2 GaussianDiscrepancyFinder   
1: Input: Sample access from $Q$ over $\mathbb { R } ^ { d }$ , parameters $k \in \mathbb { Z } _ { + } , \Delta \in ( 0 , 1 / 2 )$ , and $V , H \geq 1$   
2: Output: Reject, None, or a direction $\mathbf { w } \in \mathbb { S } ^ { d - 1 }$ , with the guarantees of Proposition 3.3.   
3: Set $F _ { k } = ( 2 k ) ^ { 2 k } , \tau = \Delta / ( 1 0 0 k 2 ^ { k } F _ { k } )$ , and $n = \lceil 3 V ( d + k ) ^ { k } / \tau ^ { 2 } \rceil$   
4: Draw $\mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( n ) }$ independently from $Q .$   
5: for $i \in [ k ]$ do   
6: Compute $\begin{array} { r } { \widehat { \mathbf { A } } _ { i } = ( 1 / n ) \sum _ { s = 1 } ^ { n } \mathcal { H } _ { i } ( \mathbf { x } ^ { ( s ) } ) } \end{array}$   
7: if $\Vert \widehat { \mathbf { A } } _ { i } \Vert _ { F } > 2 H$ then   
8: return Reject   
9: Flatten $\widehat { \mathbf { A } } _ { i }$ as a matrix and let $U _ { i }$ be the span of its left singular vectors with singular value at   
least $\tau .$   
10: Let $W = \operatorname { s p a n } ( U _ { 1 } , \dots , U _ { k } )$   
11: If $W \neq \mathbb { R } ^ { d } ;$ , choose any $\mathbf { e } \in W ^ { \perp } \cap \mathbb { S } ^ { d - 1 }$ and set $W ^ { \prime } = W + \mathrm { s p a n } \{ \mathbf { e } \}$ ; otherwise set $W ^ { \prime } = W$   
12: Let C be a $\rho \mathrm { - }$ net of $W ^ { \prime } \cap \mathbb { S } ^ { d - 1 }$ for $\rho = \Delta / ( 1 0 0 k F _ { k } H )$   
13: Let   
$( \ell _ { * } , \widehat { \mathbf { w } } ) \in \mathop { \mathrm { a r g m a x } } _ { \ell \in [ k ] , \mathbf { w } \in \mathcal { C } } \left| \frac { 1 } { n } \sum _ { s = 1 } ^ { n } ( \mathbf { w } \cdot \mathbf { x } ^ { ( s ) } ) ^ { \ell } - \underset { Z \sim { \mathcal { N } ( 0 , 1 ) } } { \mathbf { E } } [ Z ^ { \ell } ] \right|$   
14: if   
$\left| \frac { 1 } { n } \sum _ { s = 1 } ^ { n } ( \widehat { \mathbf { w } } \cdot \mathbf { x } ^ { ( s ) } ) ^ { \ell _ { * } } - \underset { Z \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { E } } [ Z ^ { \ell _ { * } } ] \right| \geq \Delta / 2$   
then   
15: return $\widehat { \bf w }$   
16: else   
17: return None

When a nonempty sample is supplied explicitly to Algorithm 2, set n to its size and use the supplied points in all empirical averages, omitting the sample-size selection and sampling steps. Repeated observations are counted with their multiplicities.

The intuition behind the algorithm is as follows. Though for an arbitrary distribution $Q ,$ , finding a moment discrepancy between Q and P might require us to do a brute force search over $\mathbb { R } ^ { d } .$ , if $Q$ satisfies the condition of Proposition 3.3, we would like to find some low-dimensional subspace W such that the moment discrepancy is preserved over $W .$ . Once this is done, we only need to run the brute force search over W, which significantly reduces the complexity of the algorithm.

To see why such a low-dimensional subspace exists, we consider the term $\mathbf { E } _ { \mathbf { x } \sim Q } [ p ] - \mathbf { E } _ { \mathbf { x } \sim P } [ p ]$ , for an arbitrary polynomial $p$ with degree at most k. Using Hermite polynomial decomposition, we can show there exists some degree k polynomial $G ,$ , such that $\mathbf { E } _ { \mathbf { x } \sim Q } [ p ] - \mathbf { E } _ { \mathbf { x } \sim P } [ p ] = \mathbf { E } _ { \mathbf { x } \sim P } [ G ( \mathbf { x } ) p ( \mathbf { x } ) ]$ . Recall that we can always expand $\begin{array} { r } { G ( \mathbf { x } ) = \sum _ { i = 1 } ^ { k } \mathbf { A } _ { i } \cdot \mathcal { H } _ { i } ( \mathbf { x } ) } \end{array}$ using Hermite tensors. If we choose $p ( \mathbf { x } ) = ( \mathbf { w } \cdot \mathbf { x } ) ^ { k }$ , then ${ \bf E } _ { { \bf x } \sim P } [ G ( { \bf x } ) p ( { \bf x } ) ]$ is a linear combination of $\mathbf { w } ^ { \otimes i } \cdot \mathbf { A } _ { i }$ . So if we flatten $\mathbf { A } _ { i }$ as a bilinear form over $\mathbb { R } ^ { d \times d ^ { i - 1 } }$ and choose W to be the span of large singular vectors of all $\mathbf { A } _ { i } .$ then adjoining one unit direction from $W ^ { \perp }$ gives a subspace $W ^ { \prime }$ that approximately preserves the moment discrepancy. However, it remains to show that W has a low dimension.

The assumption on $Q$ implies that $G$ has a small $L _ { 2 }$ norm, which further implies that each of the $\mathbf { A } _ { i }$ must have small Frobenius norm. So there is no $\mathbf { A } _ { i }$ that can have many large singular vectors and $W$ must have low dimension.

Proof of Proposition 3.3. We first construct a polynomial representing the low-degree moments of $Q .$ . We use the normalized probabilists’ Hermite polynomials $h _ { j } ( t ) = \mathrm { H e } _ { j } ( t ) / \sqrt { j ! }$ , with $h _ { 0 } = 1$ . For a multi index $\beta ,$ put $\begin{array} { r } { H _ { \beta } ( \mathbf { x } ) = \prod _ { a = 1 } ^ { d } h _ { \beta _ { a } } ( \mathbf { x } _ { a } ) } \end{array}$ . The polynomials $H _ { \beta } , | \beta | \le k$ , form an orthonormal basis for the degree-at-most-k polynomials in $L _ { 2 } ( P )$ . Define

$$
g _ { k } ( \mathbf { x } ) = \sum _ { | \beta | \leq k } \mathbf { E } _ { \sim Q } [ H _ { \beta } ( \mathbf { y } ) ] H _ { \beta } ( \mathbf { x } ) .
$$

All its coeficients are finite: the V-assumption and Cauchy–Schwarz give $| \mathbf { E } _ { Q } H _ { \beta } | \leq ( \mathbf { E } _ { Q } H _ { \beta } ^ { 2 } ) ^ { 1 / 2 } \leq \sqrt { V }$ For every polynomial $p$ of degree at most $k ,$ expansion in this orthonormal basis gives

$$
\begin{array} { r l r } {  { \mathbf { \underline { { E } } } _ { \times \sim Q } [ p ( \mathbf { x } ) ] { = } \sum _ { | \beta | \leq k } \langle p , H _ { \beta } \rangle _ { L _ { 2 } ( P ) } \ \mathbf { \underline { { E } } } [ H _ { \beta } ] } } \\ & { } & { = \sum _ { \mathbf { x } \sim P } [ p ( \mathbf { x } ) g _ { k } ( \mathbf { x } ) ] . } \end{array}
$$

In particular, ${ \bf E } _ { P } [ g _ { k } ] = { \bf E } _ { Q } [ 1 ] = 1$ . Moreover, by duality in the finite-dimensional polynomial space and the V -assumption,

$$
\begin{array} { r l } { \| g _ { k } \| _ { L _ { 2 } ( P ) } = } & { \underset { \deg p \leq k , \ \| p \| _ { L _ { 2 } ( P ) } = 1 } { \operatorname* { s u p } } | \mathbf { E } [ p g _ { k } ] | } \\ { = } & { \underset { \deg p \leq k , \ \| p \| _ { L _ { 2 } ( P ) } = 1 } { \operatorname* { s u p } } | \mathbf { E } [ p ] | } \\ & { \leq } \end{array}
$$

Thus, $G = g _ { k } - 1$ has degree at most $k ,$ has Gaussian mean zero, and satisfies $\| G \| _ { L _ { 2 } ( P ) } ^ { 2 } \le V - 1$ . For every polynomial $p ( \mathbf { x } )$ of degree no larger than $k ,$ , it holds

$$
\underset { \mathbf { x } \sim Q } { \mathbf { E } } [ p ( \mathbf { x } ) ] - \underset { \mathbf { x } \sim P } { \mathbf { E } } [ p ( \mathbf { x } ) ] = \underset { \mathbf { x } \sim P } { \mathbf { E } } [ G ( \mathbf { x } ) p ( \mathbf { x } ) ] .
$$

Let $\mathcal { H } _ { i }$ be the ith order Hermite tensor. We expand $G ( \mathbf { x } )$ using the Hermite decomposition $G ( \mathbf { x } ) =$ $\begin{array} { r } { \sum _ { i = 1 } ^ { k } \mathbf { A } _ { i } \cdot \mathcal { H } _ { i } ( \mathbf { x } ) } \end{array}$ , where for $i \in [ k ] , \mathbf { A } _ { i }$ is an order-i symmetric tensor over $\mathbb { R } ^ { d }$ . Parseval’s identity therefore gives

$$
\| \mathbf { A } _ { i } \| _ { F } ^ { 2 } \leq \sum _ { j = 1 } ^ { k } \| \mathbf { A } _ { j } \| _ { F } ^ { 2 } = \| G \| _ { L _ { 2 } ( P ) } ^ { 2 } \leq V - 1 .
$$

Moreover, for every $\mathbf { w } \in \mathbb { S } ^ { d - 1 }$ , we write simultaneously for $\ell \in \left\lceil k \right\rceil$

$$
( \mathbf { w } \cdot \mathbf { x } ) ^ { \ell } = \sum _ { i = 0 } ^ { \ell } a _ { \ell , i } \mathbf { w } ^ { \otimes i } \cdot \mathcal { H } _ { i } ( \mathbf { x } ) ,
$$

where $\begin{array} { r } { \sum _ { i = 0 } ^ { \ell } | a _ { \ell , i } | \le F _ { k } : = ( 2 k ) ^ { 2 k } = O ( k ^ { O ( k ) } ) } \end{array}$ for every $\ell \in [ k ]$

Explicitly, when $\ell - i$ is nonnegative and even,

$$
a _ { \ell , i } = \frac { \ell ! } { 2 ^ { ( \ell - i ) / 2 } ( ( \ell - i ) / 2 ) ! \sqrt { i ! } } ,
$$

and $a _ { \ell , i } = 0$ otherwise. This follows by expanding a monomial in normalized probabilists’ Hermite polynomials. Every displayed coeficient is at most $k !$ so $\begin{array} { r } { \sum _ { i = 0 } ^ { \ell } | a _ { \ell , i } | \leq ( k + 1 ) k ! \leq ( 2 k ) ^ { 2 k } } \end{array}$ . Thus the bound also includes $k = 1$

Define the degree ℓ moment discrepancy at direction w as

$$
\begin{array} { r l } & { { \cal F } _ { \ell } ( { \bf w } ) { : = } \frac { { \bf E } } { Q } [ ( { \bf w } \cdot { \bf x } ) ^ { \ell } ] - \frac { { \bf E } } { P } [ ( { \bf w } \cdot { \bf x } ) ^ { \ell } ] } \\ & { \quad \quad \quad = \frac { { \bf E } } { P } [ G ( { \bf x } ) ( { \bf w } \cdot { \bf x } ) ^ { \ell } ] = \displaystyle \sum _ { i = 1 } ^ { \ell } a _ { \ell , i } { \bf A } _ { i } \cdot { \bf w } ^ { \otimes i } , \qquad \ell \in [ k ] . } \end{array}
$$

Let $\tau \in ( 0 , 1 )$ be a parameter that will be determined later. Also for $i \in [ k ]$ let

$$
\widehat { \mathbf { A } } _ { i } : = \frac { 1 } { n } \sum _ { s = 1 } ^ { n } \mathcal { H } _ { i } ( \mathbf { x } ^ { ( s ) } )
$$

be the empirical Hermite tensor, where the $\mathbf { x } ^ { ( s ) }$ are i.i.d. samples from $Q .$

Claim 3.18 (Empirical tensor estimation). $I f n \geq 3 V ( d + k ) ^ { k } / \tau ^ { 2 }$ , with probability at least $2 / 3$ we have simultaneously for every $i \in [ k ]$ that $\left\| \widehat { \mathbf { A } } _ { i } - \mathbf { A } _ { i } \right\| _ { F } \leq \tau .$

Proof. We estimate the tensors $\mathbf { A } _ { i }$ directly from samples. By our normalization of the Hermite tensors, ${ \bf A } _ { i } =$ $\mathbf { E } _ { \mathbf { x } \sim Q } [ \mathcal { H } _ { i } ( \mathbf { x } ) ]$ , and hence $\mathbf { E } [ \widehat { \mathbf { A } } _ { i } ] = \mathbf { A } _ { i }$ . Moreover, by the assumption that $\mathbf { E _ { x \sim Q } } [ p ( \mathbf { x } ) ^ { 2 } ] \leq V \mathbf { E _ { x \sim P } } [ p ( \mathbf { x } ) ^ { 2 } ]$ for every polynomial $p$ of degree at most $k ,$ we have

$$
\mathbf { E } \left[ \left\| \widehat { \mathbf { A } } _ { i } - \mathbf { A } _ { i } \right\| _ { F } ^ { 2 } \right] \leq \frac { 1 } { n } \mathbf { \underline { { \mathbf { E } } } } _ { N } \left[ \left\| \mathcal { H } _ { i } ( \mathbf { x } ) \right\| _ { F } ^ { 2 } \right] \leq \frac { V } { n } \mathbf { \underline { { \mathbf { E } } } } _ { N } \left[ \left\| \mathcal { H } _ { i } ( \mathbf { x } ) \right\| _ { F } ^ { 2 } \right] \ .
$$

For the second inequality, apply the assumed V -bound to each entry of $\mathcal { H } _ { i }$ , which is a polynomial of degree $i ,$ and sum. The tensor normalization gives the exact count

$$
\underset { P } { \mathbf { E } } \Vert \mathcal { H } _ { i } ( \mathbf { x } ) \Vert _ { F } ^ { 2 } = \sum _ { | \beta | = i } \frac { i ! } { \beta ! } \frac { \beta ! } { i ! } \underset { P } { \mathbf { E } } [ H _ { \beta } ^ { 2 } ] = \binom { d + i - 1 } { i } .
$$

Consequently,

$$
\begin{array} { r l } & { \mathbf { E } \left[ \displaystyle \sum _ { i = 1 } ^ { k } \left\| \widehat { \mathbf { A } } _ { i } - \mathbf { A } _ { i } \right\| _ { F } ^ { 2 } \right] \leq \displaystyle \frac { V } { n } \sum _ { i = 1 } ^ { k } \binom { d + i - 1 } { i } } \\ & { \qquad = \displaystyle \frac { V } { n } \left( \binom { d + k } { k } - 1 \right) \leq \frac { V ( d + k ) ^ { k } } { n } . } \end{array}
$$

It follows from Markov’s inequality that, for

$$
n \geq { \frac { 3 V ( d + k ) ^ { k } } { \tau ^ { 2 } } } ,
$$

with probability at least $2 / 3$ we have $\left\| \widehat { \mathbf { A } } _ { i } - \mathbf { A } _ { i } \right\| _ { F } \leq \tau$ simultaneously for every $i \in [ k ]$ . Indeed, the probability that the sum of squared errors exceeds $\tau ^ { 2 }$ is at most $1 / 3 { \mathrm { . } }$ , and controlling this sum controls every individual tensor. This completes the proof of Claim 3.18. □

For the remainder of the proof we condition on the event in Claim 3.18. The algorithm first checks whether some $\Vert \widehat { \mathbf { A } } _ { i } \Vert _ { F } > 2 H$ . If so, it returns Reject. If every $\| \mathbf { A } _ { i } \| _ { F } \leq H$ , then

$$
\| \widehat { \mathbf { A } } _ { i } \| _ { F } \leq \| \mathbf { A } _ { i } \| _ { F } + \tau \leq H + \tau \leq 2 H ,
$$

since $H \geq 1$ and $\tau < 1$ . Thus rejection is impossible in that case on the event under consideration. In all subsequent steps assume that the algorithm does not reject, so $\Vert \widehat { \mathbf { A } } _ { i } \Vert _ { F } \leq 2 H$ for every $i \in \left\lceil k \right\rceil$

$i \in [ k ]$ , flatten $\widehat { \mathbf { A } } _ { i }$ as a matrix in $\mathbb { R } ^ { d \times d ^ { i - 1 } }$ , and let $U _ { i }$ be the subspace spanned by its left singular vectors with corresponding singular value at least $\tau .$ Let $W = \operatorname { s p a n } ( U _ { 1 } , \dots , U _ { k } )$ . If $W \neq \mathbb { R } ^ { d }$ , fix any $\mathbf { e } \in W ^ { \perp } \cap \mathbb { S } ^ { d - 1 }$ and set $W ^ { \prime } = W + \mathrm { s p a n } \{ \mathbf { e } \}$ ; otherwise set $W ^ { \prime } = W$

Claim 3.19 (Low-dimensional Reduction). On the estimation event, assuming that the algorithm does not reject,

$$
\dim ( W ^ { \prime } ) \leq 1 + { \frac { 4 k H ^ { 2 } } { \tau ^ { 2 } } } .
$$

Moreover, for every $\mathbf { u } \in \mathbb { S } ^ { d - 1 }$ there exists a unit vector $\bar { \mathbf { u } } \in W ^ { \prime }$ such that, simultaneously for every $\ell \in [ k ]$

$$
| F _ { \ell } ( \bar { \mathbf { u } } ) - F _ { \ell } ( \mathbf { u } ) | \leq 4 F _ { k } 2 ^ { k } \tau .
$$

In particular, $| F _ { \ell } ( \bar { \mathbf { u } } ) | \geq | F _ { \ell } ( \mathbf { u } ) | - 4 F _ { k } 2 ^ { k } \tau$

Proof. For $i \in [ k ]$ , we flatten the empirical tensor $\widehat { \mathbf { A } } _ { i }$ as a matrix in $\mathbb { R } ^ { d \times d ^ { i - 1 } }$ and let $U _ { i }$ be the subspace spanned by its left singular vectors with corresponding singular value at least τ . Since the algorithm does not reject, $\Vert \widehat { \mathbf { A } } _ { i } \Vert _ { F } \leq 2 H$ , so

$$
\mathrm { d i m } ( U _ { i } ) \leq \frac { \| \widehat { \mathbf { A } } _ { i } \| _ { F } ^ { 2 } } { \tau ^ { 2 } } \leq \frac { 4 H ^ { 2 } } { \tau ^ { 2 } } .
$$

Let $W = \operatorname { s p a n } ( U _ { 1 } , \dots , U _ { k } )$ . Then

$$
\dim ( W ) \leq \sum _ { i = 1 } ^ { k } \dim ( U _ { i } ) \leq { \frac { 4 k H ^ { 2 } } { \tau ^ { 2 } } } .
$$

Since dim $\operatorname { \mathrm { \Omega } } ( W ^ { \prime } ) \leq \dim ( W ) + 1$ , this proves the dimension bound. For a unit vector $\mathbf { z } ,$ write

$$
F _ { \ell } ( \mathbf { z } ) = \underset { P } { \mathbf { E } } [ G ( \mathbf { x } ) ( \mathbf { z } \cdot \mathbf { x } ) ^ { \ell } ] = \sum _ { i = 1 } ^ { \ell } a _ { \ell , i } \mathbf { A } _ { i } \cdot \mathbf { z } ^ { \otimes i } , \qquad \ell \in [ k ] ,
$$

and, for $\mathbf { y } \in W$ , define the auxiliary expression

$$
\widetilde F _ { \ell } ( \mathbf { y } ) : = \sum _ { i = 1 } ^ { \ell } a _ { \ell , i } \mathbf { A } _ { i } \cdot \mathbf { y } ^ { \otimes i } .
$$

Write $\mathbf { u } = \mathbf { v } + \mathbf { w }$ , with $\mathbf { v } = \mathrm { p r o j } _ { W } \mathbf { u }$ and $\mathbf { w } = \mathrm { p r o j } _ { W ^ { \perp } } \mathbf { u }$ . Then we have

$$
{ \bf u } ^ { \otimes j } = ( { \bf v } + { \bf w } ) ^ { \otimes j } = { \bf v } ^ { \otimes j } + \sum _ { S \subseteq [ j ] } \bigotimes _ { i = 1 } ^ { j } { \bf z } _ { i } ( S ) ,
$$

where for $S \subseteq [ j ]$ and $i \in [ j ] , \mathbf { z } _ { i } ( S ) = \mathbf { w } { \mathrm { ~ i f ~ } } i \in S$ and $\mathbf { z } _ { i } ( S ) = \mathbf { v }$ otherwise. This implies that

$$
\mathbf { A } _ { j } \cdot \mathbf { u } ^ { \otimes j } - \mathbf { A } _ { j } \cdot \mathbf { v } ^ { \otimes j } = \mathbf { A } _ { j } \cdot \sum _ { S \subseteq [ j ] \atop S \neq \emptyset } \bigotimes _ { i = 1 } ^ { j } \mathbf { z } _ { i } ( S ) .
$$

By symmetry, we may without loss of generality assume that $\mathbf { z } _ { 1 } ( S ) = \mathbf { w }$ . Since $\mathbf { w } \in W ^ { \perp }$ and $U _ { j } \subseteq W$ , it is orthogonal to $U _ { j }$ . By the definition of $U _ { j }$ , the restriction of the flattening of $\widehat { \mathbf { A } } _ { j }$ to $U _ { j } ^ { \bot }$ has operator norm at most τ . Equivalently, if $\widehat { \mathbf { M } } _ { j }$ denotes that flattening, then $\| \mathrm { p r o j } _ { U _ { j } ^ { \perp } } \widehat { \mathbf { M } } _ { j } \| _ { \mathrm { o p } } \leq \tau$ . Therefore, for a fixed $S _ { i }$

$$
\begin{array} { r l } & { \left| \mathbf { A } _ { j } \cdot \bigotimes _ { i = 1 } ^ { j } \mathbf { z } _ { i } ( S ) \right| \leq \left| \widehat { \mathbf { A } } _ { j } \cdot \bigotimes _ { i } ^ { j } \mathbf { z } _ { i } ( S ) \right| + \left| ( \mathbf { A } _ { j } - \widehat { \mathbf { A } } _ { j } ) \cdot \bigotimes _ { i = 1 } ^ { j } \mathbf { z } _ { i } ( S ) \right| } \\ & { \qquad \leq \tau + \left\| \mathbf { A } _ { j } - \widehat { \mathbf { A } } _ { j } \right\| _ { F } \leq 2 \tau , } \end{array}
$$

where we used that $\| \mathbf { v } \| _ { 2 } , \| \mathbf { w } \| _ { 2 } \leq 1$ . This implies that

$$
\begin{array} { r l r } {  { \Big | F _ { \ell } ( \mathbf { u } ) - \widetilde { F } _ { \ell } ( \mathbf { v } ) \Big | \le 2 \sum _ { i = 1 } ^ { \ell } | a _ { \ell , i } | ( 2 ^ { i } - 1 ) \tau } } \\ & { } & { \le 2 F _ { k } 2 ^ { k } \tau , \qquad \ell \in [ k ] . } \end{array}
$$

If $W = \mathbb { R } ^ { d } .$ , take $\bar { \bf u } = { \bf u }$ . Otherwise, using the fixed unit vector $\mathbf { e } \in W ^ { \perp }$ , set $\bar { \mathbf { u } } = \mathbf { v } + \sqrt { 1 - \left\| \mathbf { v } \right\| _ { 2 } ^ { 2 } } \mathbf { e }$ . In this latter case, put $\mathbf { w } ^ { \prime } = \sqrt { 1 - \| \mathbf { v } \| _ { 2 } ^ { 2 } } \mathbf { e }$ . Then $\mathbf { w } ^ { \prime } \in W ^ { \perp } , \| \mathbf { w } ^ { \prime } \| _ { 2 } \leq 1$ , and $\bar { \mathbf { u } } = \mathbf { v } + \mathbf { w } ^ { \prime } \in W ^ { \prime } \cap \mathbb { S } ^ { d - 1 }$ . Expanding $( \mathbf { v } + \mathbf { w } ^ { \prime } ) ^ { \otimes j }$ gives the same explicit sum over nonempty subsets as above, with each factor equal to v or $\mathbf { w } ^ { \prime }$ Each such term has contraction at most $2 \tau$ , since $\mathbf { w } ^ { \prime } \perp U _ { j }$ and the same operator-norm and estimation bounds apply. Therefore,

$$
| F _ { \ell } ( \bar { \mathbf { u } } ) - \widetilde { F } _ { \ell } ( \mathbf { v } ) | \leq 2 F _ { k } 2 ^ { k } \tau , \qquad \ell \in [ k ] .
$$

The triangle inequality now gives

$$
| F _ { \ell } ( \bar { \mathbf { u } } ) - F _ { \ell } ( { \mathbf { u } } ) | \leq 4 F _ { k } 2 ^ { k } \tau , \qquad \ell \in [ k ] ,
$$

which establishes the required bound. This completes the proof of Claim 3.19.

Choose

$$
\tau = \frac { \Delta } { 1 0 0 k 2 ^ { k } F _ { k } } , \qquad n = \left\lceil \frac { 3 V ( d + k ) ^ { k } } { \tau ^ { 2 } } \right\rceil .
$$

Then $\tau \in ( 0 , 1 )$ $4 F _ { k } 2 ^ { k } \tau = \Delta / ( 2 5 k ) \le \Delta / 2 5$ , and Claim 3.19 shows that a discrepancy of size $\Delta$ in any degree has a unit representative in $W ^ { \prime }$ with discrepancy at least $2 4 \Delta / 2 5$ in that degree. Since $F _ { k } = O ( k ^ { O ( k ) } )$ and $( d + k ) ^ { k } \leq d ^ { k } ( k + 1 ) ^ { k }$ , the sample size is $d ^ { k } { \cal \bar { O } } ( k ^ { \bar { O } ( k ) } ) V / \Delta ^ { 2 }$

Moreover, on the event in Claim 3.18, for every unit vector $\mathbf { z } ,$

$$
\left| \sum _ { i = 1 } ^ { \ell } a _ { \ell , i } \widehat { \mathbf { A } } _ { i } \cdot \mathbf { z } ^ { \otimes i } - \sum _ { i = 1 } ^ { \ell } a _ { \ell , i } \mathbf { A } _ { i } \cdot \mathbf { z } ^ { \otimes i } \right| \leq \sum _ { i = 1 } ^ { \ell } | a _ { \ell , i } | \tau \leq F _ { k } \tau \leq \Delta / 1 0 0 , \qquad \ell \in [ k ] .
$$

Define

$$
{ \widehat F } _ { \ell } ( { \mathbf { z } } ) : = \sum _ { i = 1 } ^ { \ell } a _ { \ell , i } { \widehat { \mathbf { A } } } _ { i } \cdot { \mathbf { z } } ^ { \otimes i } .
$$

The polynomial expansion also shows that

$$
\widehat { F } _ { \ell } ( \mathbf { z } ) = \frac { 1 } { n } \sum _ { s = 1 } ^ { n } ( \mathbf { z } \cdot \mathbf { x } ^ { ( s ) } ) ^ { \ell } - \underset { Z \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { E } } Z ^ { \ell } .
$$

Thus the same samples estimate all directional discrepancies uniformly, including directions selected from those samples. The search over $W ^ { \prime }$ can therefore be performed using the empirical tensors $\widehat { \mathbf { A } } _ { i }$

Claim 3.20 (Search over the Reduced Subspace). On the estimation event, assuming that the algorithm does not reject, a search over $W ^ { \prime } \cap \mathbb { S } ^ { d - 1 }$ can be implemented in time

$$
d ^ { O ( k ) } \exp \left( O ( k ^ { O ( k ) } ) \frac { H ^ { 2 } } { \Delta ^ { 2 } } \log \frac { H } { \Delta } \right) .
$$

It returns a direction with discrepancy at least $3 \Delta / 8$ in some degree, or certifies that all directional discrepancies in degrees $1 , \ldots , k$ are smaller than $\Delta$

Proof. Let $r = \dim ( W ^ { \prime } )$ . The dimension estimate and the chosen value of $\tau$ give

$$
r \le 1 + \frac { 4 k H ^ { 2 } } { \tau ^ { 2 } } \le { \cal O } ( k ^ { { \cal O } ( k ) } ) \frac { H ^ { 2 } } { \Delta ^ { 2 } } .
$$

On the branch under consideration, $\Vert \widehat { \mathbf { A } } _ { i } \Vert _ { F } \leq 2 H$ for all $i \in [ k ]$ . Therefore, for unit vectors $\mathbf { u } , \mathbf { v } \in W ^ { \prime }$ we have

$$
\begin{array} { r l r } & { } & { \displaystyle { \left| \sum _ { i = 1 } ^ { \ell } a _ { \ell , i } \widehat { \mathbf { A } } _ { i } \cdot \mathbf { u } ^ { \otimes i } - \sum _ { i = 1 } ^ { \ell } a _ { \ell , i } \widehat { \mathbf { A } } _ { i } \cdot \mathbf { v } ^ { \otimes i } \right| } \leq \sum _ { i = 1 } ^ { \ell } \vert a _ { \ell , i } \vert \Vert \widehat { \mathbf { A } } _ { i } \Vert _ { F } \Vert \mathbf { u } ^ { \otimes i } - \mathbf { v } ^ { \otimes i } \Vert _ { F } } \\ & { } & { \leq 2 k H F _ { k } \Vert \mathbf { u } - \mathbf { v } \Vert _ { 2 } . } \end{array}
$$

For completeness, the last step uses the telescoping identity

$$
\mathbf { u } ^ { \otimes i } - \mathbf { v } ^ { \otimes i } = \sum _ { a = 1 } ^ { i } \mathbf { u } ^ { \otimes ( a - 1 ) } \otimes ( \mathbf { u } - \mathbf { v } ) \otimes \mathbf { v } ^ { \otimes ( i - a ) } ,
$$

which implies $\| \mathbf { u } ^ { \otimes i } - \mathbf { v } ^ { \otimes i } \| _ { F } \leq i \| \mathbf { u } - \mathbf { v } \| _ { 2 }$ . The bound holds simultaneously for all $\ell \in \left\lceil k \right\rceil$ . Let $\mathcal { C }$ be a ρ-net of $W ^ { \prime } \cap \mathbb { S } ^ { d - 1 }$ , where $\rho = \Delta / ( 1 0 0 k F _ { k } H )$ . Such a net can be chosen with $| { \mathcal C } | \le ( 3 / \rho ) ^ { r }$ . An explicit construction also fits the claimed time bound: identify $W ^ { \prime }$ with $\mathbb { R } ^ { r }$ , round coordinates of the unit sphere to a grid of spacing $\rho / ( 4 \sqrt { r } )$ , and normalize every nonzero grid point. Rounding changes a unit vector by at most $\rho / 8$ , and normalization adds at most the same error. Enumerating the grid in $[ - 1 - \rho , 1 + \rho ] ^ { r }$ therefore produces a ρ-net with at most $( C \sqrt { r } / \rho ) ^ { r }$ points. We may use this constructible net in the algorithm. For its size,

$$
\log | { \mathcal { C } } | \leq r \log { \frac { C \sqrt { r } } { \rho } } .
$$

The choice of $\rho$ gives

$$
\log \frac { C \sqrt { r } } { \rho } = O \left( k \log ( 2 k ) + \log \frac { H } { \Delta } \right) \leq O ( k \log ( 2 k ) ) \log \frac { H } { \Delta } ,
$$

where we used $F _ { k } = ( 2 k ) ^ { 2 k } , H / \Delta > 2$ , and $r \le { O } ( k ^ { O ( k ) } ) H ^ { 2 } / \Delta ^ { 2 }$ . Consequently,

$$
\log | \mathcal { C } | \leq O ( k ^ { O ( k ) } ) \frac { H ^ { 2 } } { \Delta ^ { 2 } } \log \frac { H } { \Delta } ,
$$

and hence

$$
| \mathcal { C } | \le \exp \left( O ( k ^ { O ( k ) } ) \frac { H ^ { 2 } } { \Delta ^ { 2 } } \log \frac { H } { \Delta } \right) .
$$

By Claim 3.19, if $| F _ { \ell } ( { \mathbf { u } } ) | \geq \Delta$ for some unit u and some $\ell \in [ k ]$ , there is a unit vector $\bar { \mathbf { u } } \in W ^ { \prime }$ satisfying $| F _ { \ell } ( \bar { \mathbf { u } } ) | \geq 2 4 \Delta / 2 5$ . The uniform empirical approximation bound then gives

$$
| \widehat { F } _ { \ell } ( \bar { \mathbf { u } } ) | \geq 2 4 \Delta / 2 5 - \Delta / 1 0 0 = 1 9 \Delta / 2 0 .
$$

Let $\mathbf { z } \in { \mathcal { C } }$ satisfy $\left\| \mathbf { z } - \bar { \mathbf { u } } \right\| _ { 2 } \leq \rho$ . By the choice of $\rho$ and the Lipschitz bound,

$$
| \widehat { F } _ { \ell } ( { \bf z } ) | \geq 1 9 \Delta / 2 0 - 2 k H F _ { k } \rho = 9 3 \Delta / 1 0 0 > \Delta / 2 .
$$

Enumerate $\mathcal { C }$ and all $\ell \in [ k ]$ . If $\begin{array} { r } { \operatorname* { m a x } _ { \ell \in [ k ] , \mathbf { z } \in \mathcal { C } } | \widehat { F } _ { \ell } ( \mathbf { z } ) | \geq \Delta / 2 } \end{array}$ , return a maximizing direction w. Otherwise return None. The same empirical approximation bound then gives, whenever a direction w is returned,

$$
| F _ { \ell } ( \mathbf { w } ) | \geq | \widehat { F } _ { \ell } ( \mathbf { w } ) | - \Delta / 1 0 0 \geq 4 9 \Delta / 1 0 0 \geq 3 \Delta / 8
$$

for its selected degree ℓ. Using the defining property of $F _ { \ell } .$ , this is precisely

$$
\begin{array} { r } { \left| \mathbf { E } [ ( \mathbf { w } \cdot \mathbf { x } ) ^ { \ell } ] - \mathbf { E } [ ( \mathbf { w } \cdot \mathbf { x } ) ^ { \ell } ] \right| \geq 3 \Delta / 8 . } \end{array}
$$

Conversely, if a discrepancy of size at least $\Delta$ existed in any direction and degree, the preceding net argument would force the empirical maximum above $\Delta / 2$ . Thus returning None certifies that every such discrepancy is smaller than $\Delta$

Evaluating the empirical expression at one point of the net takes $d ^ { O ( k ) }$ time, with factors depending only on k absorbed into the displayed exponential bound. Hence the total search time is

$$
d ^ { O ( k ) } | \mathcal { C } | \leq d ^ { O ( k ) } \exp \left( O ( k ^ { O ( k ) } ) \frac { H ^ { 2 } } { \Delta ^ { 2 } } \log \frac { H } { \Delta } \right) ,
$$

which establishes the required running-time bound. This completes the proof of Claim 3.20.

To finish, the algorithm draws the stated n samples, computes the empirical Hermite tensors, rejects if some empirical tensor norm exceeds 2H, and otherwise constructs the singular subspaces and performs the net search just specified. Computing the tensors, their norms, the flattenings, and the singular subspaces takes $\mathrm { p o l y } ( n , ( d + k ) ^ { k } )$ arithmetic operations. Together with the last claim, this gives the stated running time. There are no additional sampling stages in this finder.

The single event in Claim 3.18 has probability at least $2 / 3$ . On this event the preceding arguments simultaneously prove the returned-direction guarantee, the None certificate, and the absence of rejection whenever all $\| \mathbf { A } _ { i } \| _ { F } \leq H$ This completes the proof of Proposition 3.3. □

## 4 Eficient Agnostic Learning via Gaussian Partitioning

As our first application of Theorem 3.1, we give an agnostic learning algorithm for a broad class of multi-index models under the standard Gaussian and, more generally, under marginal distributions that are $( q , \sigma )$ -smooth with respect to it. We first present our result and the intuition behind our approach. We begin by defining the class of multi-index models we consider.

Definition 4.1 $( ( \ell , K )$ -well-behaved MIMs). Let $\ell : ( 0 , 1 ] \to \mathbb { Z } _ { + }$ be non-increasing, $K \in \mathbb { Z } _ { + }$ . We say that a class $\mathcal { F } \subseteq \{ f : \mathbb { R } ^ { d } \to \{ \pm 1 \} \}$ of K-MIMs is (ℓ, K)-well-behaved if the following hold.

1. (Closure under afine restrictions) For every $f \in { \mathcal { F } }$ , every subspace $W \subseteq \mathbb { R } ^ { d }$ , and every b $\in W ^ { \perp }$ , the function $f _ { W , { \mathbf { b } } } ( { \mathbf { x } } ) : = f \bigl ( { \mathbf { b } } + \mathrm { p r o j } _ { W } { \mathbf { x } } \bigr )$ belongs to $\mathcal { F }$

2. (Gaussian sandwiching) F admits degree-ℓ $L _ { 1 }$ sandwiching polynomials under the standard Gaussian.

3. (Bounded Gaussian surface area) The class satisfies $\Gamma ( { \mathcal F } ) : = \operatorname* { s u p } _ { f \in { \mathcal F } } \Gamma ( f ) < \infty .$ , where $\Gamma ( f )$ denotes the Gaussian surface area of $f .$

For this class of functions, we prove the existence of an agnostic QPTAS under any marginal distribution that is smooth with respect to the standard Gaussian. Specifically, our main result is the following:

Theorem 4.2 (Agnostic QPTAS under Smoothness Relative to a Gaussian). Fix $q \ge 1 , c , \epsilon \in ( 0 , 1 / 2 )$ and $\sigma \in ( 0 , 1 ]$ . Let ${ \mathcal { F } } \subseteq \{ \mathbb { R } ^ { d } \to \{ \pm 1 \} \}$ be an $( \ell , K )$ -well-behaved MIM class. Let D be a distribution over $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ whose x-marginal $D _ { \mathbf { x } }$ is $( q , \sigma )$ -smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ . Let $\ell _ { c } = \ell ( c ^ { \prime } c )$ for a suficiently small universal constant $c ^ { \prime }$ . There exists an algorithm that draws

$$
n = d ^ { O ( \ell _ { c } ) } \exp \left( ( q K \ell _ { c } ) ^ { O ( \ell _ { c } ) } \log ^ { O ( \ell _ { c } ) } ( 2 / ( c \sigma \epsilon ) ) \log ( 2 + \Gamma ( \mathcal { F } ) ) \right)
$$

samples from $D _ { i }$ runs in poly(n) time, and, with probability at least $2 / 3 ,$ , outputs a hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$ such that

$$
\mathrm { e r r } _ { D } ( h ) \leq ( 1 + c ) \mathrm { o p t } _ { D } ( \mathcal { F } ) + \epsilon .
$$

We now briefly outline the intuition behind our result before we move towards the proof.

As in $[ \mathrm { D K M 2 6 } ] ,$ a core ingredient of our agnostic learning algorithm is that, given labeled samples whose x-marginal $Q$ is smooth with respect to a Gaussian distribution, we can agnostically learn a weak hypothesis over $Q$ in $d ^ { O ( 1 ) }$ time as long as the label noise is not too large. By Fact 4.3, such a weak learner sufices to obtain our agnostic approximation guarantee through boosting.

Let D be a distribution over $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ whose x-marginal $D _ { \mathbf { x } } = Q$ is $( q , \sigma )$ -smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ . The intuition here is that no matter how the label distribution changes, as long as we have sample access to $Q ,$ , we use Theorem 3.1 to create a partition $\mathcal { U }$ of $\mathbb { R } ^ { d }$ such that, ignoring a tiny amount of $Q$ mass, all the cells $R \in { \mathcal { U } }$ satisfy the three properties:

1. The $Q$ mass of each cell R is not too small and thus can be eficiently sampled from.

2. The radius of each cell R is suficiently small.

3. The low-degree moments of $Q \mid R$ approximately match those of a Gaussian for every direction in $V _ { R } ^ { \perp }$

In fact, any partition of $\mathbb { R } ^ { d }$ that satisfies the above three properties is suficient for us to design an eficient weak learning algorithm. We can apply our partitioning algorithm throughout boosting, since boosting only uses distributions obtained by mild conditioning, which preserves smoothness with slightly worse parameters.

Now given such a partition, since the diameter of a cell R is very small, if we additionally assume that our target function has some noise sensitivity property (such as bounded Gaussian surface area), then we would be able to show that our function behaves approximately like a function over $V _ { R } ^ { \perp }$ restricted to each $R .$ Indeed, fixing a representative point $\mathbf { z } _ { R } \in V _ { R }$ , we may approximate the target concept on R by the function $f _ { R } ( \mathbf { x } _ { V _ { R } ^ { \perp } } ) : = f ( \mathbf { z } _ { R } + \mathbf { x } _ { V _ { R } ^ { \perp } } )$ (see Lemma 4.5).

Recall that our partition guarantees that $Q \mid R$ approximately matches the low-degree directional moments of the standard Gaussian on $V _ { R } ^ { \perp }$ , with slack $\Delta$ . Since $f _ { R }$ depends only on this subspace and our class is closed under afine restrictions, we can use Gaussian sandwiching polynomials for $f _ { R }$ with the degree guaranteed for ${ \mathcal F }$ . By choosing the diameter suficiently small, we also ensure that $f _ { R }$ is close to $f$ on average over the partition. Using these sandwiching polynomials and moment matching, standard polynomial regression [KKMS08] yields error bounds on each cell. Averaging these bounds over the partition controls the overall error in terms of the original noise rate and the total approximation error. Choosing the Gaussian approximation degree large enough to make the approximation error a suficiently small constant then yields a weak learner, provided that the noise rate is not too large.

Note that our Gaussian Partition guarantees small cell diameter and moment matching in the orthogonal, unconstrained directions, allowing us to retain the Gaussian approximation degree. This degree can be exponentially smaller than the logconcave approximation degree used in [DKM26], giving an exponentially smaller exponent on d.

We structure the remaining section as follows: In Subsection 4.1, we construct a weak learner given a Gaussian Partition with suitable parameters. In Subsection 4.2, we combine this learner with boosting to prove our main agnostic learning result Theorem 4.2. Finally, in Subsection D.1, we extend the result to mixtures of shifted standard Gaussians and to smooth distributions with an unknown Gaussian reference mean.

## 4.1 Eficient Weak Learning Algorithm for Smooth Distributions

In this section, we show that, given a Gaussian partition satisfying the guarantees of Theorem 3.1 and a well-behaved function class (Definition 4.1), we can construct a weak learner (Theorem 4.4).

Before stating this result, we recall the following boosting procedure. It yields a relative error guarantee using a weak learner that achieves error nontrivially below $1 / 2$ whenever the noise rate is below a suitable constant.

Fact 4.3 (Corollary of Theorem 3.5 and Remark 3.6 in [Fel10]; see [DKM26]). Fix $a , \beta \in ( 0 , 1 / 2 )$ . Let H be a hypothesis class over $\mathbb { R } ^ { d }$ and let A be an algorithm with the following guarantee: for every distribution D over $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ such that $\mathrm { o p t } _ { D } ( \mathcal { H } ) \leq 1 / 2 - a$ , the algorithm draws m examples from D, runs in time $T _ { i }$ , and outputs a hypothesis $\widehat { h }$ such that err $D ( \hat { h } ) \leq 1 / 2 - \beta .$ , with probability at least 99%. Then there is an algorithm B with the following guarantee: for every distribution D over $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ and for every $\epsilon > 0$ , B makes poly $( \beta ^ { - 1 } , \epsilon ^ { - 1 } )$ many calls to A, it runs in time $\mathrm { p o l y } ( T , \beta ^ { - 1 } , \epsilon ^ { - 1 } )$ , and outputs a hypothesis $\widehat { h }$ such that $\mathrm { e r r } _ { D } ( \widehat { h } ) \leq \mathrm { o p t } _ { D } ( \mathcal { H } ) / ( 1 - 2 a ) + \epsilon _ { \lambda }$ , with probability 99%. In particular, each call to A is defined over a distribution D<sup>′</sup> satisfying $D ^ { \prime } ( { \bf x } , y ) = D ( { \bf x } , y ) w ( { \bf x } , y )$ for some weight function $w : \mathbb { R } ^ { d } \times \{ \pm 1 \}  [ 0 , 1 / \epsilon ]$ with ${ \bf E } _ { D } [ w ] = 1$ . For a fixed input distribution D and accuracy ϵ, the same conclusion holds if the weak-learning guarantee is available only for all normalized reweightings $D ^ { \prime }$ of this form, since the proof uses the weak learner only on these distributions.

We are now ready to state our weak learning result for well-behaved MIMs under the same weak-learning assumption.

Theorem 4.4 (Weak Learner). Fix $q \ge 1 , \sigma \in ( 0 , 1 ] , \eta , \delta , \Delta , \tau \in ( 0 , 1 / 2 )$ , and $k , d \in \mathbb { Z } _ { + }$ . Let D be a distribution on $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ whose x-marginal $Q = D _ { \mathbf { x } }$ is $( q , \sigma )$ -smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ . Let $\mathcal { F } \subseteq$ $\{ \mathbb { R } ^ { d }  \{ \pm 1 \} \}$ be a class that is (ℓ, K)-well-behaved and assume that opt $_ D ( \mathcal { F } ) \leq 1 / ( 2 ( 1 + c ) ) , c \in ( 0 , 1 / 2 )$ Let P be a partition that satisfies the guarantees of Theorem 3.1 for Q with parameters $( q , \sigma , \eta , \delta , k , \Delta )$ where $\eta = \tau = c / C _ { 1 } , k = \ell ( c / C _ { 1 } ) , \Delta = ( 2 K k ) ^ { - C _ { 1 } k }$ , and $\delta = ( c \sigma / C _ { 1 } ) ^ { 3 q } / ( 1 + \Gamma ( \mathcal { F } ) ) ^ { 2 }$ for a suficiently large constant $C _ { 1 } > 1$

Then there is a weak learner that, given the partition P, draws

$$
n = d ^ { O ( k ) } \exp \left( ( q K k ) ^ { O ( k ) } \log ^ { O ( k ) } ( 2 / ( c \sigma ) ) \log ( 2 + \Gamma ( \mathcal { F } ) ) \right) ,
$$

samples from D and runs in poly(n) time and with probability at least 99%, it outputs a hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$ such that $\mathbf { P r } _ { ( \mathbf { x } , y ) \sim D } [ h ( \mathbf { x } ) \neq y ] \leq 1 / 2 - c / 1 2$

The weak learner from Theorem 4.4 is given explicitly in Algorithm 3. Before we move on to the proof of Theorem 4.4, we first prove the corresponding perturbation bound that allows us to approximate a function on a small-diameter cell by fixing its constraint coordinates.

Lemma 4.5 (Perturbation on a Subcollection). Let $f : \mathbb { R } ^ { d }  \{ \pm 1 \}$ be a concept with Gaussian surface area $\Gamma ( f )$ , let P be a partition of $\mathbb { R } ^ { d }$ , and let $\mathcal { G } \subseteq \mathcal { P }$ be such that diam $( R ) \leq \delta$ for every $R \in { \mathcal { G } }$ . Then, for every $\rho \in ( 0 , 1 )$ , there exist points ${ \mathbf z } _ { R } \in \mathrm { p r o j } _ { V _ { R } } ( R )$ , one for each $R \in { \mathcal { G } }$ , such that

$$
\sum _ { R \in { \mathcal G } } \operatorname* { P r } _ { \mathbf x \sim \mathcal N ( \mathbf 0 , \mathbf I ) } [ \mathbf x \in R ] \operatorname* { P r } _ { \mathbf x \sim \mathcal N ( \mathbf 0 , \mathbf I ) } \left[ f ( \mathbf x ) \neq f \left( \mathbf z _ { R } + \mathbf x _ { V _ { R } ^ { \perp } } \right) \Big | \mathbf x \in R \right] \lesssim \sqrt { \rho } \Gamma ( f ) + \frac \delta \rho .
$$

Algorithm 3 WeakLearner   
1: Input: Sample access to $D ,$ a partition $\mathcal { P }$ satisfying the guarantees of Theorem 3.1, the bound $B ,$   
degree $\ell ,$ and regression error τ   
2: Output: A hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$   
3: Set $m = d ^ { O ( \ell ) } \mathrm { p o l y } ( 1 / \tau , \log ( 1 2 B ) )$   
4: Draw $N _ { 0 } = O ( B ^ { 2 } \log B )$ fresh samples from $D .$   
5: for $R \in \mathcal { P }$ do   
6: Let $\widehat { q } _ { R }$ be the empirical fraction of these samples whose x-coordinate lies in $R .$   
7: Let $\mathcal { H } = \{ R \in \mathcal { P } : \widehat { q } _ { R } \geq 3 / ( 4 B ) \}$   
8: Draw an additional fresh batch of $N _ { 1 } = O ( B ( m + \log B ) )$ samples from $D$ and assign each sample to   
its cell in $\mathcal { P } .$   
9: for $R \in { \mathcal { H } }$ do   
10: if R contains at least $m$ samples from the second batch then   
11: Run the $L _ { 1 }$ polynomial regression routine of Fact B.3 on m of these samples with degree $\ell ,$   
regression error $\epsilon = \tau .$ , and failure probability $1 / ( 1 2 B )$ , obtaining $q _ { R }$ and $t _ { R } \in [ - 1 , 1 ]$   
12: Set $h _ { R } ( { \bf x } ) = \mathrm { s i g n } ( q _ { R } ( { \bf x } ) - t _ { R } )$   
13: else   
14: Set $h _ { R } ( \mathbf { x } ) = 1 .$   
15: Define $h ( \mathbf { x } ) = h _ { R } ( \mathbf { x } )$ when $\mathbf { x } \in R \in \mathcal { H } .$ , and define $h ( \mathbf { x } ) = 1$ on the remaining cells.   
16: return $h .$

Proof. For each cell $R \in { \mathcal { G } }$ , let $\mathbf { u } _ { R }$ be an independent draw from the conditional distribution of $\mathbf { x } _ { V _ { R } }$ given $\mathbf { x } \in R$ , where $\mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . Given $\mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , define

$$
\mathbf { x } ^ { \prime } : = \left\{ \begin{array} { l l } { \mathbf { u } _ { R } + \mathbf { x } _ { V _ { R } ^ { \perp } } , } & { \mathrm { i f ~ } \mathbf { x } \in R \mathrm { ~ f o r ~ s o m e ~ } R \in \mathcal { G } , } \\ { \mathbf { x } , } & { \mathrm { i f ~ } \mathbf { x } \in R \mathrm { ~ f o r ~ s o m e ~ } R \in \mathcal { P } \setminus \mathcal { G } . } \end{array} \right.
$$

If $R \in { \mathcal { G } }$ , then R is determined only by its coordinates in $V _ { R } ,$ , so conditional on $\mathbf { x } \in R$ the random variables $\mathbf { x } _ { V _ { R } }$ and $\mathbf { x } _ { V _ { R } ^ { \perp } }$ are independent, and $\mathbf { x } _ { V _ { R } ^ { \perp } }$ remains a standard Gaussian on $V _ { R } ^ { \perp }$ . Consequently, conditional on $\mathbf { x } \in R ,$ , the random variable $\mathbf { x } ^ { \prime }$ has the same distribution as x. The same conclusion holds trivially when $R \in { \mathcal { P } } \setminus { \mathcal { G } }$ , since then $\mathbf { x } ^ { \prime } = \mathbf { x }$ . Since $\mathcal { P }$ is a partition, it follows that $\mathbf { x } ^ { \prime } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ marginally.

Let $g = T _ { \rho } f$ , where $T _ { \rho }$ denotes the Gaussian noise operator. By the Gaussian noise-sensitivity bound for functions of Gaussian surface area $\Gamma ( f )$ , we have $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) } \left[ | f ( \mathbf { x } ) - g ( \mathbf { x } ) | \right] \lesssim \sqrt { \rho } \Gamma ( f )$ . Moreover, $g$ is $O ( 1 / \rho )$ -Lipschitz. If $\mathbf { x } \in R$ for some $R \in { \mathcal { G } }$ , then $\mathbf { x } , \mathbf { x } ^ { \prime } \in R$ and their $V _ { R } ^ { \perp }$ components are identical, so

$$
\| \mathbf { x } - \mathbf { x } ^ { \prime } \| _ { 2 } = \| \mathbf { x } _ { V _ { R } } - \mathbf { u } _ { R } \| _ { 2 } \leq \mathrm { d i a m } ( R ) \leq \delta .
$$

If x lies in a cell of ${ \mathcal { P } } \setminus { \mathcal { G } } ,$ , then $\mathbf { x } ^ { \prime } = \mathbf { x }$ . Thus, $\| \mathbf { x } - \mathbf { x } ^ { \prime } \| _ { 2 } \leq \delta$ in all cases. Therefore,

$$
\begin{array} { l } { \displaystyle \mathbf { P r } [ f ( \mathbf { x } ) \neq f ( \mathbf { x } ^ { \prime } ) ] = \frac { 1 } { 2 } \mathbf { E } \left[ | f ( \mathbf { x } ) - f ( \mathbf { x } ^ { \prime } ) | \right] } \\ { \displaystyle \qquad \leq \frac { 1 } { 2 } \mathbf { E } \left[ | f ( \mathbf { x } ) - g ( \mathbf { x } ) | \right] + \frac { 1 } { 2 } \mathbf { E } \left[ | g ( \mathbf { x } ) - g ( \mathbf { x } ^ { \prime } ) | \right] + \frac { 1 } { 2 } \mathbf { E } \left[ | g ( \mathbf { x } ^ { \prime } ) - f ( \mathbf { x } ^ { \prime } ) | \right] } \\ { \displaystyle \qquad \lesssim \sqrt { \rho } \Gamma ( f ) + \frac { \delta } { \rho } , } \end{array}
$$

where in the last inequality we used that $\mathbf { x } ^ { \prime } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$

For each $R \in { \mathcal { G } }$ and $\mathbf { z } \in \mathrm { p r o j } _ { V _ { R } } ( R )$ , define

$$
e _ { R } ( \mathbf { z } ) : = \operatorname* { P r } _ { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) } \left[ f ( \mathbf { x } ) \neq f \left( \mathbf { z } + \mathbf { x } _ { V _ { R } ^ { \bot } } \right) \Big | \mathbf { x } \in R \right] .
$$

By construction,

$$
\begin{array} { r } { \mathbf { \Delta E } _ { \mathbf { u } _ { R } } \left[ e _ { R } ( \mathbf { u } _ { R } ) \right] = \mathbf { P r } \left[ f ( \mathbf { x } ) \neq f ( \mathbf { x } ^ { \prime } ) \ : | \ : \mathbf { x } \in R \right] . } \end{array}
$$

Hence, there exists ${ \mathbf z } _ { R } \in \mathrm { p r o j } _ { V _ { R } } ( R )$ such that

$$
e _ { R } ( \mathbf { z } _ { R } ) \leq \mathbf { P r } \left[ f ( \mathbf { x } ) \neq f ( \mathbf { x } ^ { \prime } ) \vert \mathbf { x } \in R \right] .
$$

Since $\mathbf { x } ^ { \prime } = \mathbf { x }$ whenever x lies in a cell of ${ \mathcal { P } } \backslash { \mathcal { G } }$ , these cells contribute no disagreement. Therefore, multiplying by $\mathbf { P r _ { x \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) } } [ \mathbf { x } \in R ]$ and summing over $R \in { \mathcal { G } }$ gives

$$
\begin{array} { r l } & { \displaystyle \sum _ { R \in \mathcal { G } } \displaystyle \sum _ { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) } [ \mathbf { x } \in R ] e _ { R } ( \mathbf { z } _ { R } ) } \\ & { \quad \quad \quad \le \displaystyle \sum _ { R \in \mathcal { G } } \displaystyle \sum _ { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) } [ \mathbf { x } \in R ] \mathbf { P r } \left[ f ( \mathbf { x } ) \neq f ( \mathbf { x } ^ { \prime } ) \right. \left| \mathbf { x } \in R \right. ] } \\ & { \quad \quad = \mathbf { P r } [ f ( \mathbf { x } ) \neq f ( \mathbf { x } ^ { \prime } ) ] } \\ & { \quad \quad \quad \lesssim \sqrt { \rho } \Gamma ( f ) + \frac { \delta } { \rho } , } \end{array}
$$

which proves the claim.

We are now ready to prove that Gaussian sandwiching approximations along with a Gaussian Partition can yield a weak learner when the noise rate is bounded away from $1 / 2$ . The intuition is as follows. Let $g$ be the piecewise function defined on each certified cell R by $g ( \mathbf { x } ) : = f ( \mathbf { z } _ { R } + \mathbf { x } _ { V _ { R } ^ { \bot } } )$ , where $\mathbf { z } _ { R }$ is supplied by the previous lemma. By that lemma, $g$ is close to f on average over the partition. Moreover, on each certified cell $R ,$ the restriction of $g$ depends only on $\mathbf { x } _ { V _ { R } ^ { \perp } }$ and, by the assumptions on $\mathcal { F }$ , admits Gaussian sandwiching polynomials. Together with moment matching, these polynomials give a lower bound on the correlation with the labels on each certified cell.

A particular cell, however, may have a substantially larger conditional noise rate than the original distribution, and $f$ need not be close to g on that cell. We therefore average the correlation bounds over the partition. This bounds the noise contribution by the global noise rate of $f$ and the approximation contribution by the total error between $f$ and $^ { g , }$ yielding a nontrivial overall correlation with the labels.

Proof of Theorem $4 . 4 \cdot$ . Let $\mathcal { P }$ be a partition that satisfies the guarantees of Theorem 3.1 for $Q$ with parameters $( q , \sigma , \eta , \delta , k , \Delta )$ . We bound the error and complexity of Algorithm 3 given a partition with the stated parameters and then set those parameters accordingly to get the stated weak learning guarantee.

First, we establish the existence of a function that is a low-degree polynomial on each cell of the partition and whose approximation error is close to the stated bound. We then apply $L _ { 1 }$ polynomial regression and thresholding on suficiently high-mass cells to eficiently obtain a Boolean hypothesis.

Fix $\zeta > 0$ and choose $f \in { \mathcal { F } }$ such that err $D ( f ) \leq \mathrm { o p t } _ { D } ( \mathcal { F } ) + \zeta$

Let B be an upper bound on the quantity supplied by the partition theorem,

$$
{ \cal B } = \exp \left( ( q k ) ^ { O ( k ) } \log ^ { O ( k ) } ( 2 / ( \sigma \eta ) ) \log ( 1 / \delta ) / \Delta ^ { O ( 1 ) } \right) \ .
$$

Let $\mathcal { G } \subseteq \mathcal { P }$ be a collection of good cells whose existence is guaranteed by Theorem 3.1. By item (ii) of the partitioning theorem, diam $( R ) \leq \delta$ for every $R \in { \mathcal { G } }$

Let $S = \cup _ { R \in \mathcal { G } } R$ . For each fixed cell $R \in { \mathcal { G } }$ , define the function $f _ { R } ( { \bf x } ) = f ( { \bf z } _ { R } + { \bf x } _ { V _ { R } ^ { \perp } } )$ where ${ \mathbf z } _ { R } \in \mathrm { p r o j } _ { V _ { R } } ( R )$ is as defined in Lemma 4.5, and let $g : S  \mathbb { R }$ be given by $\begin{array} { r } { g ( \mathbf { x } ) : = \sum _ { R \in \mathcal { G } } f _ { R } ( \mathbf { x } ) \overset { * } { \mathbb { 1 } } \{ \mathbf { x } \in R \} } \end{array}$ By $( q , \sigma )$ -smoothness and Lemma 4.5 we have that

$$
\begin{array} { r l r } {  { \operatorname* { P r } _ { \mathbf { x } \sim D _ { \mathbf { x } } } [ f ( \mathbf { x } ) \neq g ( \mathbf { x } ) , \mathbf { x } \in S ] \leq ( 1 / \sigma ) \operatorname* { P r } _ { \mathbf { x } \sim N ( \mathbf { 0 } , \mathbf { I } ) } [ f ( \mathbf { x } ) \neq g ( \mathbf { x } ) , \mathbf { x } \in S ] ^ { 1 / q } } } \\ & { } & { \lesssim \frac { ( \delta ( 1 + \Gamma ( f ) ) ^ { 2 } ) ^ { 1 / ( 3 q ) } } { \sigma } , } \end{array}
$$

where we apply Lemma 4.5 with diameter bound δ and choose $\rho = ( \delta / ( 1 + \Gamma ( f ) ) ) ^ { 2 / 3 } \in ( 0 , 1 )$ . As a result, $\mathbf { P r } _ { D } [ y \neq g ( \mathbf { x } ) , \mathbf { x } \in S ] \leq \mathrm { e r r } _ { D } ( f ) + O \left( ( \delta ( 1 + \Gamma ( f ) ) ^ { 2 } ) ^ { 1 / ( 3 q ) } / \sigma \right)$

Fix a cell $R \in { \mathcal { G } }$ . Since f is a K-MIM, so is $f _ { R } ( \mathbf { x } ) : = f ( \mathbf { z } _ { R } + \mathbf { x } _ { V _ { R } ^ { \perp } } )$ , and hence there exists a subspace $U _ { R } \subseteq V _ { R } ^ { \bot }$ , with dim $( U _ { R } ) \leq K$ , on which $f _ { R }$ depends. Moreover, since $\mathcal { F }$ is $( \ell , K )$ -well-behaved, $f _ { R } \in \mathcal { F }$ and therefore there exist $\mathrm { d e g r e e } { - \ell } = \ell ( \tau )$ polynomials $p _ { R } ^ { \pm } : \mathbb { R } ^ { d } \to \mathbb { R }$ such that $p _ { R } ^ { - } ( \mathbf { x } ) \leq f _ { R } ( \mathbf { x } ) \leq p _ { R } ^ { + } ( \mathbf { x } )$ for all $\mathbf { x } \in \mathbb { R } ^ { d }$ and $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } ) } [ p _ { R } ^ { + } ( \mathbf { x } ) - p _ { R } ^ { - } ( \mathbf { x } ) ] \leq \tau$ . Averaging each $p _ { R } ^ { \pm }$ over the standard Gaussian on $U _ { R } ^ { \perp }$ preserves both the sandwiching inequalities and their Gaussian error, since $f _ { R }$ depends only on $U _ { R }$ . Thus, after relabeling, we may assume that $p _ { R } ^ { + }$ and $p _ { R } ^ { - }$ , and hence $s _ { R } : = p _ { R } ^ { + } - p _ { R } ^ { - }$ , depend only on $U _ { R }$

Let $Q _ { R }$ be the distribution of $\mathbf { x } _ { U _ { R } }$ under $D _ { \mathbf { x } } \mid R .$ Since R is a good cell and $k \geq \ell , Q _ { R }$ and $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { U _ { R } } )$ match their first ℓ moments with slack $\Delta$ . Moreover, $s _ { R } \geq 0$ and $\| s _ { R } \| _ { L _ { 1 } ( \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { U _ { R } } ) ) } \leq \tau$ . Therefore, Fact B.8 gives

$$
\left| \underset { \mathbf { x } \sim Q _ { R } } { \mathbf { E } } [ s _ { R } ( \mathbf { x } ) ] - \underset { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { U _ { R } } ) } { \mathbf { E } } [ s _ { R } ( \mathbf { x } ) ] \right| \leq ( K \ell ) ^ { O ( \ell ) } \Delta \tau .
$$

As a result,

$$
\begin{array} { r l } & { \mathbf { E } [ \left| y - p _ { R } ^ { + } ( \mathbf { x } _ { V _ { R } ^ { \perp } } ) \right| \mid R ] \leq 2 \mathbf { P r } [ y \neq g ( \mathbf { x } ) \mid R ] + \mathbf { E } [ p _ { R } ^ { + } ( \mathbf { x } _ { V _ { R } ^ { \perp } } ) - p _ { R } ^ { - } ( \mathbf { x } _ { V _ { R } ^ { \perp } } ) \mid R ] } \\ & { \qquad \leq 2 \mathbf { P r } [ y \neq g ( \mathbf { x } ) \mid R ] + \mathbf { \sum } _ { N ( \mathbf { 0 } , \mathbf { I } ) } [ p _ { R } ^ { + } ( \mathbf { x } _ { V _ { R } ^ { \perp } } ) - p _ { R } ^ { - } ( \mathbf { x } _ { V _ { R } ^ { \perp } } ) ] + ( K \ell ) ^ { O ( \ell ) } \Delta \tau } \\ & { \qquad \leq 2 \mathbf { P r } [ y \neq g ( \mathbf { x } ) \mid R ] + \tau + ( K \ell ) ^ { O ( \ell ) } \Delta \tau , } \end{array}
$$

where we used moment matching and the sandwiching error.

For $R \in { \mathcal { G } }$ define $p _ { R } ( \mathbf { x } ) : = p _ { R } ^ { + } ( \mathbf { x } _ { V _ { R } ^ { \bot } } )$ and for $R \not \in { \mathcal { G } }$ define $p _ { R } ( \mathbf { x } ) = 0$ . Let $\begin{array} { r } { p ( \mathbf { x } ) = \sum _ { R \in \mathcal { P } } p _ { R } ( \mathbf { x } ) \mathbb { 1 } \{ \mathbf { x } \in R \} } \end{array}$ By averaging the preceding cellwise bound and using $\begin{array} { r } { \mathbf { P r } _ { D } [ \mathbf { x } \notin \bigcup _ { R \in \mathcal { G } } R ] \le \eta } \end{array}$ , we obtain

$$
\begin{array} { l } { \displaystyle \mathbf { E } [ | { \boldsymbol { y } } - { \boldsymbol { p } } ( \mathbf { x } ) | ] = \sum _ { R \in \mathcal { P } } \mathbf { P r } [ R ] \ \mathbf { E } _ { D } [ | { \boldsymbol { y } } - { \boldsymbol { p } } _ { R } ( \mathbf { x } ) | \ | \ R ] } \\ { \displaystyle \qquad \leq 2 \mathrm { e r r } _ { D } ( f ) + O \left( \frac { ( \delta ( 1 + \Gamma ( f ) ) ^ { 2 } ) ^ { 1 / ( 3 q ) } } { \sigma } \right) + \eta + \tau + ( K \ell ) ^ { O ( \ell ) } \Delta \tau . } \end{array}
$$

We next analyze Algorithm $^ { 3 , }$ which replaces the existential polynomials $p _ { R }$ by eficiently computable classifiers without using G in the algorithm. Draw $N _ { 0 } = O ( B ^ { 2 } \log B )$ fresh samples from D and, for each $R \in \mathcal { P }$ , let $\widehat { q } _ { R }$ be the empirical fraction of samples whose x-coordinate lies in R. Since $| \mathcal { P } | \le B$ , Hoefding’s inequality and a union bound imply that, with probability at least $1 1 / 1 2$ $| \widehat { q } _ { R } - \mathbf { P r } _ { D } [ R ] | \ \leq \ 1 / ( 4 B )$ simultaneously for every $R \in \mathcal { P }$ . Let $\mathcal { H } : = \{ R \in \mathcal { P } : \widehat { q } _ { R } \geq 3 / ( 4 B ) \}$ . On this event, every $R \in { \mathcal { G } }$ belongs to H, since $\mathbf { P r } _ { D } [ R ] \geq 1 / B$ , and every $R \in { \mathcal { H } }$ satisfies $\mathbf { P r } _ { D } [ R ] \geq 1 / ( 2 B )$

For every $R \in { \mathcal { H } }$ , apply Fact B.3 to $D \mid R .$ , with degree ℓ, regression error $\epsilon = \tau$ , and failure probability $1 / ( 1 2 B )$ . Each invocation needs $m = d ^ { O ( \ell ) } \mathrm { p o l y } ( 1 / \tau , \log ( 1 2 B ) )$ ) samples from $D \mid R$ and the same amount of time. Draw an additional fresh batch of $N _ { 1 } = O ( B ( m + \log B ) )$ samples from D and assign each sample to its cell. Conditional on the first sampling event, every $R \in { \mathcal { H } }$ has mass at least $1 / ( 2 B )$ , so a Chernof bound and a union bound imply that, with probability at least $1 1 / 1 2$ , every cell in H contains enough conditional samples. Since $| \mathcal { H } | \leq B$ , the regression calls take total time $B d ^ { O ( \ell ) } \mathrm { p o l y } ( 1 / \tau , \log ( 1 2 B ) )$ Another union bound over their failure probabilities shows that, with probability at least $1 1 / 1 2$ , all of these regression calls succeed. In particular, since $\mathcal { G } \subseteq \mathcal { H }$ , simultaneously for all $R \in { \mathcal { G } }$ we obtain polynomials $q _ { R }$ and thresholds $t _ { R } \in [ - 1 , 1 ]$ such that

$$
\mathbf { P r } [ \mathrm { s i g n } ( q _ { R } ( \mathbf { x } ) - t _ { R } ) \neq y \mid R ] \leq { \frac { 1 } { 2 } } \mathbf { E } [ \left| y - p _ { R } ( \mathbf { x } ) \right| \mid R ] + \tau .
$$

The intersection of the preceding three events has probability at least $3 / 4$ , and hence at least $2 / 3$ . Define $h ( { \bf x } ) = \mathrm { s i g n } ( q _ { R } ( { \bf x } ) - t _ { R } )$ when ${ \mathbf x } \in R \in \mathcal { H }$ , and define it arbitrarily on the remaining cells. In the analysis

we only use the guarantees on $R \in { \mathcal { G } }$ . Since $\mathbf { P r } _ { D } [ \mathbf { x } \notin S ] \leq \eta$ , averaging the preceding guarantees gives

$$
\begin{array} { r l } & { \mathbf { P } _ { D } [ h ( \mathbf { x } ) \neq y ] \leq \mathbf { P } _ { D } [ \mathbf { x } \neq S ] + \displaystyle \frac { 1 } { 2 } \sum _ { R \in \mathcal { G } } \mathbf { P } _ { D } [ R ] \mathbf { E } [ | y - p _ { R } ( \mathbf { x } ) | \mid R ] + \tau } \\ & { \qquad \leq \displaystyle \frac { 1 } { 2 } \mathbf { E } [ | y - p ( \mathbf { x } ) | ] + \frac { 1 } { 2 } \mathbf { P r } [ \mathbf { x } \notin S ] + \tau } \\ & { \qquad \leq \operatorname { e r r } _ { D } ( f ) + O \left( \displaystyle \frac { ( \delta ( 1 + \Gamma ( f ) ) ^ { 2 } ) ^ { 1 / ( 3 q ) } } { \sigma } \right) + \eta + 2 \tau + ( K \ell ) ^ { O ( \ell ) } \Delta \tau } \\ & { \qquad \leq \operatorname { o p t } _ { D } ( \mathcal { F } ) + \zeta + O \left( \displaystyle \frac { ( \delta ( 1 + \Gamma ( \mathcal { F } ) ) ^ { 2 } ) ^ { 1 / ( 3 q ) } } { \sigma } \right) + \eta + 2 \tau + ( K \ell ) ^ { O ( \ell ) } \Delta \tau . } \end{array}
$$

Since $\zeta > 0$ is arbitrary, this gives the stated guarantee. Since $| \mathcal { P } | \le B$ , the first sampling stage uses $O ( B ^ { 2 } \log B )$ samples and the second uses $O ( B d ^ { O ( \ell ) } \mathrm { p o l y } ( 1 / \tau , \log B ) )$ samples. These quantities are bounded by $d ^ { O ( \ell ) } \mathrm { p o l y } ( 1 / \tau ) \exp \left( ( q k ) ^ { O ( k ) } \log ^ { O ( k ) } ( 2 / ( \sigma \eta ) ) \log ( 1 / \delta ) / \Delta ^ { O ( 1 ) } \right)$ for suficiently large constants.

Let $c \in ( 0 , 1 / 2 )$ , and suppose that op $\mathit { \Pi } _ { \overline { { D } } } ( \mathcal { F } ) \leq 1 / ( 2 ( 1 + \mathit { \dot { c } } ) )$ . Choose a suficiently large universal constant $C _ { 1 }$ and set $\eta = \tau = c / C _ { 1 } , \ell = \ell ( \tau ) , k = \ell$ , and $\Delta : = ( 2 K k ) ^ { - C _ { 1 } k } , \delta : = ( c \sigma / C _ { 1 } ) ^ { 3 q } / ( 1 + \Gamma ( \mathcal { F } ) ) ^ { 2 }$ Then

$$
\operatorname { e r r } _ { D } ( h ) \leq { \frac { 1 } { 2 ( 1 + c ) } } + O \left( { \frac { c } { C _ { 1 } } } \right) \leq { \frac { 1 } { 2 } } - { \frac { c } { 6 } } .
$$

Repeating the procedure a constant number of times and selecting a hypothesis using fresh validation samples with error estimates accurate to $c / 2 4$ yields, with probability at least 99%, a hypothesis with error at most $1 / 2 - c / 1 2$ . The sample complexity, including amplification, is

$$
n = d ^ { O ( k ) } \exp \left( ( q K k ) ^ { O ( k ) } \log ^ { O ( k ) } ( 2 / ( c \sigma ) ) \log ( 2 + \Gamma ( \mathcal { F } ) ) \right) ,
$$

and the running time is poly(n).

## 4.2 Eficient Agnostic Learning under Gaussian-Smooth Distributions

In this section, we use the weak learner designed in the previous section in order to obtain our main agnostic learning algorithm (Theorem 4.2). We later apply our results to several MIM classes, and Table 3 summarizes our concrete applications.

<table><tr><td>Concept class</td><td>Complexity</td></tr><tr><td>(l, K)-well-behaved MIMs</td><td> $d ^ { O ( \ell ( c ^ { \prime } c ) ) }$ </td></tr><tr><td>Intersections or unions of k halfspaces</td><td> $d ^ { \widetilde { O } ( k ^ { 3 } / c ^ { 4 } ) }$ </td></tr><tr><td>Arbitrary Boolean functions of k halfspaces</td><td> $d ^ { \widetilde { O } ( k ^ { 5 } / c ^ { 4 } ) }$ </td></tr><tr><td>Convex concepts of intrinsic dimension K</td><td> $d ^ { \widetilde { O } ( K ^ { 5 } / c ^ { 4 } ) }$ </td></tr><tr><td>Degree-r PTFs of intrinsic dimension K</td><td> $d ^ { \widetilde { O } ( r ^ { 6 } K ^ { 5 } / c ^ { 4 } ) }$ </td></tr></table>

Table 3: Agnostic learning results. All results hold for marginal distributions that are $( q , \sigma )$ -smooth with respect to the standard Gaussian. The displayed complexities have an additional factor quasi-polynomial in $1 / ( \sigma \epsilon )$ for fixed class parameters, q, and c.

Combining the boosting procedure in Fact 4.3 with our weak learner from Theorem 4.4, we can now prove Theorem 4.2.

Proof of Theorem 4.2. Set $\alpha = \sigma \epsilon$ . Every distribution $D ^ { \prime }$ used by Fact 4.3 satisfies $D ^ { \prime } ( { \bf x } , y ) = D ( { \bf x } , y ) w ( { \bf x } , y )$ for some $w : \mathbb { R } ^ { d } \times \{ \pm 1 \}  [ 0 , 1 / \epsilon ]$ with ${ \bf E } _ { D } [ w ] = 1$ . Writing $G = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ , for every measurable $E \subseteq \mathbb { R } ^ { d }$ we have

$$
D _ { \mathbf { x } } ^ { \prime } ( E ) \leq D _ { \mathbf { x } } ( E ) / \epsilon \leq G ( E ) ^ { 1 / q } / ( \sigma \epsilon ) = G ( E ) ^ { 1 / q } / \alpha .
$$

Thus $D _ { \mathbf { x } } ^ { \prime }$ is $( q , \alpha )$ -smooth with respect to $G .$

Apply Theorem 3.1 to $D _ { \mathbf { x } } ^ { \prime }$ with the parameters specified in Theorem 4.4, using smoothness parameter α. Whenever opt $_ { D ^ { \prime } } ( \mathcal { F } ) \leq 1 / ( 2 ( 1 + c ) )$ , the resulting partition and Theorem 4.4 give a hypothesis with error at most $1 / 2 - c / 1 2$ with probability at least 99%. This supplies the weak learner required by Fact 4.3. Consequently,

$$
\operatorname { e r r } _ { D } ( h ) \leq \frac { \operatorname { o p t } _ { D } ( \mathcal { F } ) } { 1 - 2 ( 1 / 2 - 1 / ( 2 ( 1 + c ) ) ) } + \epsilon = ( 1 + c ) \mathrm { o p t } _ { D } ( \mathcal { F } ) + \epsilon .
$$

To verify the complexity, take $c ^ { \prime } \leq 1 / C _ { 1 }$ . Since ℓ is non-increasing, $\ell ( c / C _ { 1 } ) \leq \ell ( c ^ { \prime } c ) = \ell _ { c }$ . The partition construction and the amplified weak learner therefore use at most

$$
n _ { \mathrm { w } } = d ^ { O ( \ell _ { c } ) } \exp \left( ( q K \ell _ { c } ) ^ { O ( \ell _ { c } ) } \log ^ { O ( \ell _ { c } ) } ( 2 / ( c \sigma \epsilon ) ) \log ( 2 + \Gamma ( \mathcal { F } ) ) \right)
$$

samples per call and polynomial time.

Samples from $D ^ { \prime }$ are simulated by accepting samples from D with probability $\epsilon w ( \mathbf { x } , y )$ , whose expectation is ϵ. Let $R = \mathrm { p o l y } ( 1 / c , 1 / \epsilon )$ bound the number of boosting calls. Drawing $O ( ( n _ { \mathrm { w } } + \log ( 1 0 0 R ) ) / \epsilon )$ samples for each weak learner call supplies the weak learner with $n _ { w }$ samples with failure probability at most $1 / 1 0 0$ , by the Hoefding and union bounds. Combining this with the 99% success probability of boosting gives overall success probability at least 98%. Noticing that the number of weak learner calls as well as the time complexity of the weak learner are polynomial justifies our sample and time complexity bounds, completing the proof. □

We first apply Theorem 4.2 to intersections of halfspaces.

Corollary 4.6 (Agnostic Learning of Intersections of Halfspaces). Let $q \geq 1$ , k, $d \in  { \mathbb { N } } _ { + } , c , \epsilon \in ( 0 , 1 / 2 )$ and $\sigma \in ( 0 , 1 ]$ . Let $\mathcal { F } _ { k }$ be the class of intersections of at most k afine halfspaces over $\mathbb { R } ^ { d }$ . Let D be a distribution over $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ whose x-marginal is $( q , \sigma )$ -smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ . There is an algorithm that draws

$$
n = d ^ { \tilde { O } ( k ^ { 3 } / c ^ { 4 } ) } \exp \left( ( q k / c ) ^ { ( k / c ) ^ { O ( 1 ) } } \log ^ { ( k / c ) ^ { O ( 1 ) } } ( 2 / ( \sigma \epsilon ) ) \right)
$$

samples from $D ,$ runs in time $\mathrm { p o l y } ( n )$ , and, with probability at least $2 / 3$ , outputs a hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$ satisfying err $_ { D } ( h ) \leq ( 1 + c ) \mathrm { o p t } _ { D } ( \mathcal { F } _ { k } ) + \epsilon$

Proof of Corollary $4 . 6 .$ The class $\mathcal { F } _ { k }$ consists of k-MIMs and is closed under afine restrictions. By Facts B.6 and B.7, it admits Gaussian $L _ { 1 }$ sandwiching polynomials of degree $\ell ( \tau ) = \widetilde O ( k ^ { 3 } / \tau ^ { 4 } )$ and satisfies $\Gamma ( \mathcal { F } _ { k } ) = O ( \sqrt { \log ( 2 k ) } )$ . Thus $\mathcal { F } _ { k }$ is $( \ell , k )$ -well-behaved. Applying Theorem 4.2 with $K = k$ and $\ell _ { c } = \ell ( \Omega ( c ) ) = \widetilde { O } ( k ^ { 3 } / c ^ { 4 } )$ gives the result. □

We next apply Theorem 4.2 to arbitrary Boolean functions of halfspaces.

Corollary 4.7 (Agnostic Learning of Functions of Halfspaces). Let $q \ge 1 , k , d \in \mathbb { N } _ { + } , c , \epsilon \in ( 0 , 1 / 2 )$ , and $\sigma \in ( 0 , 1 ]$ . Let $\mathcal { H } _ { k }$ denote the class of arbitrary Boolean functions of at most k afine halfspaces over $\mathbb { R } ^ { d }$ Let D be a distribution over $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ whose x-marginal $D _ { \mathbf { x } }$ is $( q , \sigma )$ -smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$

There exists an algorithm that draws

$$
n = d ^ { \widetilde { O } ( k ^ { 5 } / c ^ { 4 } ) } \exp \left( ( q k / c ) ^ { ( k / c ) ^ { O ( 1 ) } } \log ^ { ( k / c ) ^ { O ( 1 ) } } \left( 2 / ( \sigma \epsilon ) \right) \right)
$$

samples from $D ,$ runs in time polynomial in $n ,$ and, with probability at least $2 / 3$ , outputs a hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$ satisfying

$$
\mathrm { e r r } _ { D } ( h ) \leq ( 1 + c ) \mathrm { o p t } _ { D } ( \mathcal { H } _ { k } ) + \epsilon .
$$

Proof. The class $\mathcal { H } _ { k }$ is closed under afine restrictions, since restricting an afine halfspace gives an afine halfspace or a constant function. Moreover, every $f \in \mathcal { H } _ { k }$ depends only on the span of the normals of it defining halfspaces, and hence is a k-MIM.

The decision boundary of $f$ is contained in the union of at most k hyperplanes. Gaussian anticoncentration therefore gives an $O ( k )$ boundary-smoothness bound, and $\Gamma ( \mathcal { H } _ { k } ) = O ( k )$ . By Fact $\mathrm { B } . 6 , \mathcal { H } _ { k }$ admits Gaussian $L _ { 1 }$ sandwiching polynomials of degree $\ell ( \tau ) = \widetilde O ( k ^ { 5 } / \tau ^ { 4 } )$ . Thus $\mathcal { H } _ { k }$ is (ℓ, k)-well-behaved. Applying Theorem 4.2 with smoothness parameters $( q , \sigma )$ and $K = k$ and substituting $\ell ( \Omega ( c ) ) = \widetilde { O } ( k ^ { 5 } / c ^ { 4 } )$ and $\Gamma ( \mathcal { H } _ { k } ) = O ( k )$ yields the stated sample complexity, running time, and error guarantee. □

Applying our general theorem for Low-Dimensional convex sets gives the following corollary.

Corollary 4.8 (Agnostic Learning of Low-Dimensional Convex Sets). Let $q \ge 1 , K , d \in \mathbb { N } _ { + } , c , \epsilon \in ( 0 , 1 / 2 )$ and $\sigma \in ( 0 , 1 ]$ . Let $\mathcal { C } _ { K }$ denote the class of Boolean K-MIMs over $\mathbb { R } ^ { d }$ whose positive regions are convex. Let D be a distribution over $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ whose x-marginal $D _ { \mathbf { x } }$ is $( q , \sigma )$ -smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$

There exists an algorithm that draws

$$
n = d ^ { \widetilde { O } ( K ^ { 5 } / c ^ { 4 } ) } \exp \left( ( q K / c ) ^ { ( K / c ) ^ { O ( 1 ) } } \log ^ { ( K / c ) ^ { O ( 1 ) } } ( 2 / ( \sigma \epsilon ) ) \right)
$$

samples from $D$ , runs in time polynomial in $n _ { ; }$ , and, with probability at least $2 / 3$ , outputs a hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$ satisfying

$$
\mathrm { e r r } _ { D } ( h ) \leq ( 1 + c ) \mathrm { o p t } _ { D } ( \mathcal { C } _ { K } ) + \epsilon .
$$

Proof. The class $\mathcal { C } _ { K }$ is closed under afine restrictions, since afine preimages preserve convexity and do not increase the intrinsic dimension.

By Fact B.7, $\sigma ( \mathcal { C } _ { K } ) = O ( K \log ( 2 K ) )$ and $\Gamma ( \mathcal { C } _ { K } ) \ \leq \ \sigma ( \mathcal { C } _ { K } )$ . By Fact B.6, $\mathcal { C } _ { K }$ admits Gaussian $L _ { 1 }$ sandwiching polynomials of degree $\ell ( \tau ) = \widetilde O ( K ^ { 5 } / \tau ^ { 4 } )$ . Thus $\mathcal { C } _ { K }$ is $( \ell , K )$ -well-behaved. Applying Theorem 4.2 with smoothness parameters $( q , \sigma )$ and substituting $\ell ( \Omega ( c ) ) = \widetilde { O } ( K ^ { 5 } / c ^ { 4 } )$ and the above surface area bound yields the stated sample complexity, running time, and error guarantee. □

Applying our general theorem also gives a result for Low-Dimensional $\mathrm { P T F s }$

Corollary 4.9 (Agnostic Learning of Low-Dimensional PTFs). Let $q \ge 1 , r , K , d \in \mathbb { N } _ { + } , c , \epsilon \in ( 0 , 1 / 2 )$ and $\sigma \in ( 0 , 1 ]$ . Let $\mathcal { P } _ { K , r }$ denote the class of polynomial threshold functions over $\mathbb { R } ^ { d } \ o f$ degree at most r that are $K { - } M I M s$ . Let D be a distribution over $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ whose x-marginal $D _ { \mathbf { x } }$ is $( q , \sigma )$ -smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$

There exists an algorithm that draws

$$
n = d ^ { \tilde { O } ( r ^ { 6 } K ^ { 5 } / c ^ { 4 } ) } \exp \left( ( q r K / c ) ^ { ( r K / c ) ^ { O ( 1 ) } } \log ^ { ( r K / c ) ^ { O ( 1 ) } } ( 2 / ( \sigma \epsilon ) ) \right)
$$

samples from $D _ { ; }$ runs in time polynomial in $n _ { ; }$ , and, with probability at least $2 / 3$ , outputs a hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$ satisfying

$$
\mathrm { e r r } _ { D } ( h ) \leq ( 1 + c ) \mathrm { o p t } _ { D } ( \mathcal { P } _ { K , r } ) + \epsilon .
$$

Proof. The class $\mathcal { P } _ { K , r }$ is closed under afine restrictions, since composing a polynomial with an afine map does not increase its degree, and the intrinsic dimension remains at most $K$

By Fact B.7, $\sigma ( \mathcal P _ { K , r } ) = O ( r ^ { 3 } K )$ and $\Gamma ( \mathcal { P } _ { K , r } ) \leq \sigma ( \mathcal { P } _ { K , r } )$ . By Fact B.6, $\mathcal { P } _ { K , i }$ <sub>r</sub> admits Gaussian $L _ { 1 }$ sandwiching polynomials of degree $\ell ( \tau ) = \widetilde O ( r ^ { 6 } K ^ { 5 } / \tau ^ { 4 } )$ . Thus $\mathcal { P } _ { K , 1 }$ <sub>r</sub> is $( \ell , K )$ -well-behaved. Applying Theorem 4.2 with smoothness parameters $( q , \sigma )$ and substituting $\ell ( \Omega ( c ) ) = { \widetilde O } ( r ^ { 6 } K ^ { 5 } / c ^ { 4 } )$ and the above surface area bound yields the stated sample complexity, running time, and error guarantee. □

Remark 4.10 (Applications to Natural Distribution Families). Our agnostic learning algorithm works for all $( q , \sigma )$ -smooth distributions with respect to the standard Gaussian. We are able to show that this includes some natural distribution families. In fact, mixtures of Gaussians with identity covariance and component means of norm at most $M$ are $( 2 , e ^ { - M ^ { 2 } / 2 } )$ -smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ (see Lemma D.1), so Theorem 4.2 applies directly. Moreover, for classes closed under translations, the same learning guarantee holds for marginals that are $( q , \sigma )$ -smooth with respect to $\mathcal { N } ( \pmb { \mu } , \mathbf { I } _ { d } )$ for an unknown $\pmb { \mu } ,$ without sample access to the reference Gaussian. See Subsection D.1 for details.

## 5 Learning with Positive and Imperfect Unlabeled Examples

In this section, we give another application of Theorem 3.1 to a diferent reliable learning problem, [LMZ26], learning from positive examples and imperfect unlabeled data.

Definition 5.1 (PUI learning with Imperfect Data). Let $\mathcal { F } \subseteq \{ f : \mathbb { R } ^ { d } \to \{ \pm 1 \} \}$ be a hypothesis class. Let D<sup>∗</sup> be a distribution over $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ such that there exists some $f ^ { * } \in { \mathcal { F } }$ with ${ \bf P r } _ { ( { \bf x } , y ) \sim D ^ { * } } [ f ^ { * } ( { \bf x } ) \neq y ] = 0$ and $\mathbf { P r } _ { ( \mathbf { x } , y ) \sim D ^ { * } } [ y = 1 ] \geq p _ { + }$ , where $p _ { + } \in ( 0 , 1 )$ . Suppose that there exists a reference distribution D over $\mathbb { R } ^ { d }$ such that $D _ { \mathbf { x } } ^ { * }$ is $( q , \sigma )$ -smooth with respect to D. Given sample access to D and $D _ { + } ^ { * }$ , where $D _ { + } ^ { * } : = D _ { \mathbf { x } } ^ { * } \mid y = 1$ , and $\epsilon \in ( 0 , 1 )$ , the goal is to output a hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$ such that $\mathrm { e r r } _ { D ^ { * } } ( h ) : = \mathbf { P r } _ { ( \mathbf { x } , y ) \sim D ^ { * } } [ h ( \mathbf { x } ) \neq y ] \leq \epsilon$

The algorithm proposed by [LMZ26] uses an L<sub>1</sub>-approximating polynomial developed in the agnostic learning literature as a subroutine. Roughly speaking, their algorithm requires an L -approximate polynomial to functions in $\mathcal { F }$ to an accuracy of $( \epsilon \sigma p _ { + } ) ^ { O ( q ) }$ . Even if the reference distribution $D = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ and $\mathcal { F }$ is the class of halfspaces, this results in a time complexity of $d ^ { ( \epsilon \sigma p _ { + } ) ^ { - O ( q ) } }$ . We would like to mention that if the function class $\mathcal { F }$ is adversarially constructed, for example, as the one used in [DKPZ24], then according to [LMZ26], a time complexity of $d ^ { \Omega ( 1 / \epsilon ^ { 2 } ) }$ is unavoidable. However, as we show in this section, for many natural concept classes, one can potentially achieve a much better complexity. In fact, we will see that the Gaussian partitioning algorithm, Theorem 3.1, directly gives a much more eficient algorithm for the problem when the reference distribution $D = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . Our starting point is to introduce the following definition called the Gaussian PUI learning Exponent, which qualitatively determines the computational complexity of PUI learning with a Gaussian reference.

Definition 5.2 (Gaussian PUI Learning Exponent). Let $\mathcal { F } \subseteq \{ f : \mathbb { R } ^ { d } \to \{ \pm 1 \} \}$ be a class of functions closed under afine transformations and let $D : = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ . We define the Gaussian PUI learning exponent $m ^ { * } ( \epsilon , \Delta , \sigma , q )$ to be the minimum integer $k \in \mathbb { Z } _ { + }$ such that, for every $f \in { \mathcal { F } }$ and every distribution $D ^ { * }$ that is $( q , \sigma )$ -smooth with respect to D satisfying $D ^ { * } ( f ( \mathbf { x } ) = - 1 ) \geq \epsilon$ and $D ^ { * } ( f ( \mathbf { x } ) = 1 ) > 0$ , the positive conditional distribution $D _ { + } ^ { * } : = D ^ { * } \mid f ( \mathbf { x } ) = 1$ cannot match the first k moments of $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ with slack $\Delta$ in the sense of Definition 2.2. If no such integer exists, set $m ^ { * } ( \epsilon , \Delta , \sigma , q ) = \infty$

The exponent $m ^ { * }$ allows us to both design eficient PUI learning algorithms and analyze the hardness of the learning problem. Specifically, the main results in this section are the following pair of theorems that quantify the computational complexity of PUI learning (for multi-index models).

Our main algorithmic result in this section is the following:

Theorem 5.3 (PUI Learning with Gaussian References). Let $\mathcal { F }$ be a class of functions that is closed under afine restrictions and has bounded Gaussian surface area $\Gamma ( \mathcal { F } ) < \infty \ ( s a t i s f i e s$ items (1) and $( 3 )$ of $D e f i n i t i o n \ 4 . 1 )$ , let the reference distribution be the standard Gaussian $D = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ , and let $D ^ { * }$ be a target distribution that is $( q , \sigma )$ -smooth with respect to $D , q \geq 1$ and $\sigma \in ( 0 , 1 ]$ . Let $f \in { \mathcal { F } }$ be an unknown function, let $p _ { + } > 0$ satisfy $D ^ { * } ( f = 1 ) \geq p _ { + }$ , and set $D _ { + } ^ { * } = D ^ { * } \mid f ( \mathbf { x } ) = 1$ . Let $\epsilon , \Delta \in ( 0 , 1 / 2 )$ and set $k = m ^ { * } ( c \epsilon , \Delta , c ^ { \prime } \epsilon \sigma p _ { + } / q , 2 q )$ for suficiently small universal constants $c , c ^ { \prime } > 0$ , and assume $k < \infty$ There exists a PUI learning algorithm that, given $\epsilon , \sigma , p _ { + } , \Delta , q$ and $d ^ { k } \exp ( ( q k ) ^ { O ( k ) } \log ^ { O ( k ) } ( 2 / ( \sigma \epsilon p _ { + } ) )$ log(2+ $\Gamma ( \mathcal { F } ) ) / \Delta ^ { O ( 1 ) } )$ samples from $D _ { + } ^ { * }$ , runs in time polynomial in its sample complexity and with probability at least $2 / 3$ outputs a hypothesis h such that $\mathbf { P r } _ { \mathbf { x } \sim D ^ { * } } [ h ( \mathbf { x } ) \neq f ( \mathbf { x } ) ] \leq \epsilon$

Perhaps surprisingly, we show a qualitatively matching SQ lower bound:

Theorem 5.4 (SQ Lower Bounds for PUI Learning of Gaussian MIMs). Let d, $K , m ^ { * } \in \mathbb { Z } _ { + }$ with $m ^ { * } \geq 2$ and max $\{ K , m ^ { * } \} \ \leq \ d ^ { c }$ , for a suficiently small absolute constant $c > 0$ and suficiently large d. Let $\sigma , \epsilon \in ( 0 , 1 ]$ and $q \geq 1$ . Let ${ \mathcal { F } } \subseteq \{ \mathbb { R } ^ { d } \to \{ \pm 1 \} \}$ be a class of Boolean functions. Suppose there exists $f \in { \mathcal { F } }$ such that:

1. f depends only on the projection onto a subspace $V \subseteq \mathbb { R } ^ { d }$ of dimension $K _ { i }$ $i . e . , f ( \mathbf { x } ) = g ( \mathbf { W } \mathbf { x } )$ for $\mathbf { W } \in \mathbb { R } ^ { K \times d }$ satisfying $\mathbf { W } \mathbf { W } ^ { \top } = \mathbf { I } _ { K }$

2. $\mathcal { F }$ contains every rotation of $f ,$ i. $\mathbf { \sigma } . e . , \mathbf { x } \mapsto g ( \mathbf { U } \mathbf { x } )$ belongs to $\mathcal { F }$ for every $\mathbf { U } \in \mathbb { R } ^ { K \times d }$ satisfying $\mathbf { U U } ^ { \top } = \mathbf { I } _ { K }$

3. There exists a distribution $D ^ { * }$ that is $( q , \sigma )$ -smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ such that $p _ { + } : = { \bf P r } _ { \mathbf x \sim D ^ { * } } [ f ( \mathbf x ) =$ $1 ] \in ( 0 , 1 ) , \operatorname* { P r } _ { \mathbf { x } \sim D ^ { * } } [ f ( \mathbf { x } ) = - 1 ] \geq \epsilon$ , and $( \sigma \epsilon ) ^ { q } , p _ { + } \geq d ^ { - c m ^ { * } }$ . Moreover, $D _ { + } ^ { * } : = D ^ { * } \mid \{ f ( \mathbf { x } ) = 1 \}$ and $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ match moments through degree $m ^ { * }$

Then any $S Q$ algorithm that learns $\mathcal { F }$ over distributions that are $( q , \sigma )$ -smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ ， using $S Q$ access only to the positive conditional distribution $D _ { + } ^ { * }$ , to error $p _ { + } ( \sigma \epsilon ) ^ { q } / 2$ with probability at least $2 / 3$ , requires either $2 ^ { d ^ { \Omega ( 1 ) } }$ queries or a query of tolerance at most $d ^ { - \Omega ( m ^ { * } ) }$

Remark 5.5 (Qualitative Tightness). The upper and lower bounds establish a qualitative correspondence between moment matching and $\mathrm { S Q }$ complexity, rather than a quantitative characterization by the same value of the PUI learning exponent. The proof of Theorem 5.3 achieves error ϵ with dimension exponent $m ^ { * } ( \epsilon / 8 , \Delta , \epsilon \sigma p _ { + } / ( 1 2 8 0 0 q ) , 2 q )$ . In contrast, under the assumptions of Theorem 5.4, an instance whose positive conditional distribution matches Gaussian moments exactly through degree $m ,$ and whose negative mass is at least $\rho ,$ yields $\mathrm { S Q }$ complexity $d ^ { \Omega ( m ) }$ for learning error $p _ { + } ( \sigma \rho ) ^ { q } / 2$ . Such a witness implies $m ^ { * } ( \rho , \Delta , \sigma , q ) > m$ for every $\Delta > 0 ;$ ; the inequality for a fixed positive slack alone does not sufice to invoke the lower bound.

To compare the bounds at learning error ϵ, one therefore sets $\rho ~ = ~ ( 2 \epsilon / p _ { + } ) ^ { 1 / q } / \sigma$ , provided the resulting parameters satisfy the lower-bound theorem’s assumptions. The parameter diferences arise from conditioning in the upper-bound proof, which changes the smoothness parameters, and from the accuracy required by the reduction from PUI learning to distinguishing in the lower-bound proof.

If $q$ is fixed and $\sigma ^ { q } p _ { + } \geq \epsilon ^ { 1 - a }$ for some fixed $a \in ( 0 , 1 )$ , then $\rho \le 2 ^ { 1 / q } \epsilon ^ { a / q }$ , so the accuracy rescaling changes ϵ only by a fixed power. For logarithmic moment thresholds, these parameter changes can preserve the dimension exponent up to constant factors depending on $q .$ . This occurs for $q = 1$ in the regime of Theorem 5.14.

Roadmap of Section 5 In Subsection 5.1, we give the proof of Theorem 5.3, giving the general PUI learning algorithm with a Gaussian reference. In Subsection 5.2, we prove Theorem $5 . 4 ,$ showing that the $d ^ { m ^ { * } }$ dependence is qualitatively optimal for multi-index models. In Subsection 5.3, we focus on natural concept classes such as halfspaces, intersections of halfspaces, and general K-dimensional convex sets. In particular, for the class of intersections of k halfspaces $\mathcal { F } _ { k } ,$ we show that $m ^ { * } = \Theta ( \log ( k / ( \sigma \epsilon ) ) )$ ), implying these natural classes have much more eficient PUI learning algorithms.

## 5.1 Faster Algorithm for PUI Learning with Gaussian Reference

In this section, we give the proof of Theorem 5.3. Our learning algorithm is stated as Algorithm 4. The definition of $m ^ { * }$ suggests a very simple PUI learning algorithm. For intuition, we first consider a one-dimensional PUI learning problem. We will use the contrapositive part of $m ^ { * }$ . Suppose that $D _ { + } ^ { * }$ matches its first $m ^ { * }$ moments with respect to a standard Gaussian distribution within a slack $\Delta$ . Then this implies that the fraction of negative examples under $D ^ { * }$ must be at most $\epsilon .$ Thus, we can safely output a hypothesis that labels every example to be positive. However, the moment matching condition in general fails for a general distribution $D _ { + } ^ { * }$ over $\mathbb { R } ^ { d }$ . Fortunately, since $D ^ { * }$ is smooth with respect to a Gaussian, as long as $p _ { + }$ is not too small, $D _ { + } ^ { * }$ is also smooth with respect to the Gaussian distribution. We therefore apply our Gaussian partitioning algorithm with $Q = D _ { + } ^ { * }$ and $k = m ^ { * }$

Algorithm 4 GaussianPUILearner   
1: Input: Sample access to $D _ { + } ^ { * }$ and parameters $\epsilon , \Delta , \sigma , q ,$ and a lower bound $0 < p _ { + } \leq D ^ { * } ( f = 1 )$   
2: Output: A hypothesis h satisfying the guarantees of Theorem 5.3.   
3: Set $\Gamma = 1 + \Gamma ( \mathcal { F } ) , \alpha = \sigma p _ { + } / 4 .$ and $\eta = \epsilon / 8 .$   
4: Set $\sigma _ { * } = \alpha \eta / ( 4 0 0 q )$ and $k = m ^ { * } ( \eta , \Delta , \sigma _ { * } , 2 q )$   
5: Set $M _ { k } = \sqrt { ( 2 k - 1 ) ! ! }$ and $\Delta _ { 0 } = \Delta / ( 2 ^ { k + 2 } M _ { k } )$   
6: Set $\delta = \Gamma ^ { - 2 } \left( \frac { \alpha \eta \Delta } { C q k } \right) ^ { C q ( k + q ) }$ , for a suficiently large universal constant $C > 1 .$   
7: Run the Gaussian partitioning procedure of Theorem 3.1 on $D _ { + } ^ { * }$ with parameters $( q , \alpha , \eta , \delta , k , \Delta _ { 0 } )$   
obtaining a partition $\mathcal { P }$ and its certified subcollection $\mathcal { G } _ { 0 } .$   
8: Set $\lambda = \eta / 1 0 0 ;$ and let B be the cell-count bound in Theorem 3.1.   
9: Draw $\lceil 8 B ^ { 2 } \log ( 2 0 0 B ) \rceil$ fresh samples from $D _ { + } ^ { * }$ , let $\widehat { q } _ { R }$ be their empirical mass in each $R \in \mathcal G _ { 0 }$   
10: Set $\mathcal { G } = \{ R \in \mathcal { G } _ { 0 } : \widehat { q } _ { R } > 2 \lambda D ( R ) \}$   
11: return the hypothesis   
$h ( \mathbf { x } ) = \left\{ { \begin{array} { l l } { 1 , } & { \mathbf { x } \in \bigcup _ { R \in \mathcal { G } } R , } \\ { - 1 , } & { { \mathrm { o t h e r w i s e . } } } \end{array} } \right.$

Recall Theorem 3.1, our Gaussian partitioning algorithm partitions $\mathbb { R } ^ { d }$ into cells such that most of $D _ { + } ^ { * }$ mass falls into some R with small radius and satisfies the moment-matching condition over $V _ { R } ^ { \perp }$ Furthermore, $D _ { + } ^ { * } \mid R$ is still smooth. The small radius condition together with a mild noise sensitivity condition (Gaussian surface area) implies that we can assume, restricted at $R ,$ the target function $f$ only depends on directions in $V _ { R } ^ { \perp }$

Following our discussion above, the moment matching condition together with the smoothness condition implies that if we label all examples in R as positive, we introduce at most ϵ conditional error. Thus, if we label all examples over these moment matching cells R as positive and the other examples as negative, then the fraction of false positives is at most $O ( \epsilon )$ . On the other hand, since the fraction of false negatives under $D _ { + } ^ { * }$ is at most $O ( \epsilon )$ , it is also at most $O ( \epsilon )$ under $D ^ { * }$ . Thus, we get a PUI learning algorithm with complexity $d ^ { m ^ { * } }$

Proof of Theorem 5.3. Set $\Gamma : = 1 + \Gamma ( \mathcal { F } )$ , so that $\Gamma ( f ) \leq \Gamma$ . Let P and $\mathcal { G } _ { 0 }$ be the partition and certified subcollection returned by Theorem 3.1 for $D _ { + } ^ { * }$ with parameters $( q , \alpha , \eta , \delta , k , \Delta _ { 0 } )$ as in Algorithm 4, and let ${ \mathcal { G } } \subseteq { \mathcal { G } } _ { 0 }$ be the cells retained by its ratio test (Line 10). We take $\alpha = \sigma p _ { + } / 4 \le \sigma D ^ { * } ( f = 1 )$ , so that $D _ { + } ^ { * }$ is $( q , \alpha )$ -smooth with respect to D and $\alpha < 1 / 2$ , as required by Theorem 3.1. We use $\eta = \epsilon / 8$ and $k = m ^ { * } ( \eta , \Delta , \sigma _ { * } , 2 q )$ with $\sigma _ { * } = \alpha \eta / ( 4 0 0 q )$ . For the analysis, set $M _ { k } : = \sqrt { ( 2 k - 1 ) ! ! }$ $M : = 1 / ( \sigma ^ { q } \eta ^ { q - 1 } )$ , $\pi _ { 0 } : = p _ { + } \eta / ( 1 0 0 M ) , ~ A _ { q , k } : = 8 \sqrt { 4 q } ( 8 q - 1 ) ^ { k / 2 } M _ { k }$ , and $r _ { 0 } : = ( \sigma _ { * } \pi _ { 0 } \Delta / A _ { q , k } ) ^ { 4 q }$ . Let B be the bound in Theorem 3.1 for these parameters. The theorem guarantees that diam $( R ) \leq \delta$ for every $R \in \mathcal G _ { 0 }$ and $| \mathcal { P } | \leq B$ . Every $R \in \mathcal G _ { 0 }$ also satisfies $D _ { + } ^ { * } ( R ) \geq 1 / B$

Let $\lambda = \eta / 1 0 0$ . Hoefding’s inequality and a union bound give $| { \widehat { q } } _ { R } - Q ( R ) | \leq 1 / ( 4 B ) \leq Q ( R ) / 4$ simultaneously for all $R \in \mathcal { G } _ { 0 }$ , with probability at least $9 9 / 1 0 0$ , for the estimates of Line 9. On this event, every retained cell satisfies $Q ( R ) \geq ( 4 / 5 ) { \widehat { q } } _ { R } > \lambda D ( R )$ . Every discarded cell satisfies $Q ( R ) \leq ( 4 / 3 ) { \widehat { q } } _ { R } \leq$ $( 8 / 3 ) \lambda D ( R ) \leq 3 \lambda D ( R )$ . Therefore, summing the aforementioned inequality we have that on its estimation event, every retained cell satisfies $D _ { + } ^ { * } ( R ) / D ( R ) \geq \eta / 1 0 0$ , and $D _ { + } ^ { * } ( \bigcup _ { R \in { \mathcal { G } } _ { 0 } \backslash { \mathcal { G } } } R ) \leq 3 \lambda$

The Gaussian masses $D ( R )$ are computable from the orthogonal interval constraints defining each cell. In what follows, work on the joint success event of the partitioning routine and the estimation event.

Let h be defined by $h ( \mathbf { x } ) = 1 { \mathrm { ~ i f ~ } } \mathbf { x } \in R$ for some $R \in { \mathcal { G } }$ , and $h ( \mathbf { x } ) = - 1$ otherwise.

Claim 5.6 (Small Mass of False Negatives). We have $\operatorname* { P r } _ { \mathbf { x } \sim D ^ { * } } [ h ( \mathbf { x } ) = - 1 , f ( \mathbf { x } ) = 1 ] \leq \eta + 3 \lambda .$

Proof. Note that by Theorem 3.1 we had that the total mass of $D _ { + } ^ { * }$ outside $\mathcal { G } _ { 0 }$ is at most η. Therefore, since only 3λ mass of $D _ { + } ^ { * }$ belongs in $\mathcal { G } _ { 0 } \backslash \mathcal { G }$ we have that $D _ { + } ^ { * } ( S ) \geq 1 - \eta - 3 \lambda$ , for $S : = \cup _ { R \in \mathcal { G } } R$

Since $h ( \mathbf { x } ) = - 1$ exactly when ${ \textbf { x } } \notin S$ and $D _ { + } ^ { * } = D ^ { * } \mid f ( \mathbf { x } ) = 1$ , it follows that $\mathbf { P r } _ { \mathbf { x } \sim D ^ { * } } [ h ( \mathbf { x } ) =$ $\begin{array} { r } { - 1 , f ( \mathbf { x } ) = 1 \big ] = D ^ { * } ( f ( \mathbf { x } ) = 1 ) D _ { + } ^ { * } ( S ^ { c } ) \leq ( \eta + 3 \lambda ) D ^ { * } ( f ( \mathbf { x } ) = 1 ) \leq \eta + 3 \lambda . } \end{array}$ □

Lemma 5.7 (Small Mass of False Positives). We have $\begin{array} { r } { \mathbf { P r _ { x \sim D ^ { * } } } [ h ( \mathbf { x } ) = 1 , f ( \mathbf { x } ) = - 1 ] \leq 3 \eta + O \left( \frac { 1 } { \sigma } \left( \frac { ( \delta \Gamma ^ { 2 } ) ^ { 1 / 3 } } { r _ { 0 } } \right) ^ { 1 / q } \right) } \end{array}$

Proof of Lemma 5.7. Applying Lemma 4.5 with $\rho = ( \delta / \Gamma ) ^ { 2 / 3 }$ gives points ${ \mathbf z } _ { R } \in \mathrm { p r o j } _ { V _ { R } } ( R )$ such that, for $f _ { R } ( { \bf x } ) : = f ( { \bf z } _ { R } + \mathrm { p r o j } _ { V _ { D } ^ { \perp } } { \bf x } )$ , we have $\begin{array} { r } { \sum _ { R \in \mathcal { G } } \mathbf { P r } _ { \mathbf { x } \sim D } [ f ( \mathbf { x } ) \neq f _ { R } ( \mathbf { x } ) \ | \ R ] D ( R ) = O ( ( \delta \Gamma ^ { 2 } ) ^ { 1 / 3 } ) } \end{array}$ . Moreover, by closure under afine restrictions, $f _ { R } \in \mathcal { F }$

Write $\begin{array} { r } { A : = \sum _ { R \in \mathcal { G } } D ( R ) \mathbf { P r } _ { D } [ f \neq f _ { R } \ | \ R ] = O ( ( \delta \Gamma ^ { 2 } ) ^ { 1 / 3 } ) } \end{array}$ . Applying $( q , \sigma )$ -smoothness once to the union $\cup _ { R \in \mathcal { G } } ( R \cap \{ f \neq f _ { R } \} )$ , we have

$$
\mathbf { P r } _ { \mathbf { x } \sim D ^ { * } } [ h ( \mathbf { x } ) = 1 , f ( \mathbf { x } ) = - 1 ] \leq \sum _ { R \in \mathcal { G } } \mathbf { P r } _ { \mathbf { x } \sim D ^ { * } } [ f _ { R } ( \mathbf { x } ) = - 1 \mid R ] D ^ { * } ( R ) + \frac { A ^ { 1 / q } } { \sigma } .
$$

Note that, moreover, by Theorem 3.1, $D _ { + } ^ { * } \mid R$ matches moments with $D \mid R$ on $V _ { R } ^ { \perp }$ . Also, $D | R$ on $V _ { R } ^ { \perp }$ is a standard Gaussian.

We aim to apply the definition of $m ^ { * }$ for the functions $f _ { R }$ for cells $R \in { \mathcal { G } }$ . In order to do that we show that for almost all cells in ${ \mathcal { G } } ,$ measured by their $D ^ { * }$ -mass, the distribution $Q _ { R }$ obtained by projecting $D ^ { * } \mid R$ onto $V _ { R } ^ { \perp }$ satisfies that $Q _ { R } \mid f _ { R } = 1$ approximately matches the first k moments of the standard Gaussian on $\bar { V } _ { R } ^ { \perp }$ . We obtain this by using the fact that $f$ is very close to $f _ { R }$ on average.

Claim 5.8 $( Q _ { R } | f _ { R } ( { \bf x } ) = 1$ matches moments with $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { V _ { R } ^ { \perp } } ) \big )$ . Suppose the partition is constructed with moment slack $\Delta _ { 0 } \leq \Delta / 4$ . There exists a subcollection $\mathcal { H } \subseteq \mathcal { G }$ such that

$$
\sum _ { R \in \mathcal { G } \backslash \mathcal { H } } D ^ { * } ( R ) \leq 2 \eta + \frac { 1 } { \sigma } \left( \frac { A } { r _ { 0 } } \right) ^ { 1 / q } .
$$

For every $R \in \mathcal H _ { \mathrm { : } }$ , let $Q _ { R }$ be the distribution of $\operatorname { p r o j } _ { V _ { R } ^ { \perp } } \mathbf { x } f o r \mathbf { x } \sim D ^ { * } \mid R$ . Then $Q _ { R } \mid f _ { R } = 1$ matches the first k moments of the standard Gaussian on $V _ { R } ^ { \perp }$ with slack $\Delta$ . Moreover, $Q _ { R }$ is $( 2 q , \sigma _ { * } )$ -smooth with respect to this Gaussian, and $Q _ { R } ( f _ { R } = 1 ) \ge \pi _ { 0 } / 2 > 0$

Proof of Claim 5.8. Let H be the collection of certified cells satisfying the following three conditions:

$$
\begin{array} { r } { \mathcal { H } : = \{ R \in \mathcal { G } : \ D ^ { * } \mid R \mathrm { ~ i s ~ } ( 2 q , \sigma _ { * } ) \mathrm { - s m o o t h ~ w . r . t . ~ } D \mid R , \operatorname* { P r } [ f = 1 \mid R ] \geq \pi _ { 0 } , \operatorname* { P r } _ { D ^ { * } } [ f \neq f _ { R } \mid R ] \leq r _ { 0 } ^ { 1 / q } \} . } \end{array}
$$

First we show that only a small mass of certified cells does not belong in H and afterwards we show that for all $R \in \mathcal { H } , Q _ { R } | f _ { R } = 1$ matches moments with the corresponding standard Gaussian.

Mass of excluded certified cells. For the first condition, apply Lemma 3.11 to $D ^ { * }$ and D on the entire partition, with parameters $( q , \sigma / 2 , \eta )$ . Since $\sigma \eta / ( 4 q ) \geq \sigma _ { * }$ , the certified cells failing this condition have total $D ^ { * }$ -mass at most η.

For the second condition, recall $M = 1 / ( \sigma ^ { q } \eta ^ { q - 1 } )$ The laws of the cell index under $D ^ { * }$ and D satisfy the same $( q , \sigma )$ -smoothness inequality. Applying Lemma 3.8 to these laws shows that cells with $D ^ { * } ( R ) / D ( R ) > M$ have total $D ^ { * }$ -mass at most η. Every retained cell with $D ^ { * } ( R ) / D ( R ) \leq M$ satisfies ${ \bf P r } _ { D ^ { * } } [ f = 1 \ | \ R ] = D ^ { * } ( f = 1 ) D _ { + } ^ { * } ( R ) / D ^ { * } ( R ) \ge p _ { + } \eta / ( 1 0 0 M ) = \pi _ { 0 }$ . Therefore, the retained cells failing the second condition have total $D ^ { * } { \mathrm { - m a s s } }$ at most $\eta .$

For the third condition, the union of the cellwise disagreement events has D-mass A, so global $( q , \sigma )$ -smoothness gives

$$
\sum _ { R \in { \mathcal { G } } } D ^ { * } ( R ) \operatorname* { \bf P r } _ { D ^ { * } } [ f \neq f _ { R } \ | \ R ] \leq \frac { A ^ { 1 / q } } { \sigma } .
$$

Markov’s inequality therefore bounds the total $D ^ { * }$ -mass of certified cells failing this condition by

$$
r _ { 0 } ^ { - 1 / q } \sum _ { R \in \mathcal { G } } D ^ { * } ( R ) \operatorname { \bf P r } [ f \neq f _ { R } \mid R ] \leq \frac { 1 } { \sigma } \left( \frac { A } { r _ { 0 } } \right) ^ { 1 / q } .
$$

Combining the three conditions by a union bound yields

$$
\sum _ { R \in \mathcal { G } \backslash \mathcal { H } } D ^ { * } ( R ) \leq 2 \eta + \frac { 1 } { \sigma } \left( \frac { A } { r _ { 0 } } \right) ^ { 1 / q } .
$$

Moment matching on retained cells. Now fix $R \in { \mathcal { H } }$ . It sufices to show that $\ ( \mathrm { p r o j } _ { V _ { R } ^ { \perp } } ( D ^ { * } \mid R ) ) \mid f _ { R } = 1$ has approximately the same first k moments as $\mathrm { p r o j } _ { V _ { R } ^ { \perp } } ( D ^ { * } \mid ( R , f = 1 ) )$ . Since $f _ { R }$ depends only on the $V _ { R } ^ { \perp }$ coordinates, we have $( \mathrm { p r o j } _ { V _ { p } ^ { \bot } } ( D ^ { * } \mid R ) ) \mid f _ { R } = 1 = \mathrm { p r o j } _ { V _ { p } ^ { \bot } } ( D ^ { * } \mid ( R , f _ { R } = 1 ) )$

Let $P = D ^ { * } \mid R , p = P ( f = 1 ) , p ^ { \prime } = P ( f _ { R } = 1 )$ , and fix a unit vector $\mathbf { v } \in V _ { R } ^ { \perp }$ and $j \in [ k ]$ . Expanding the conditioning indicators gives

$$
\begin{array} { r l } & { \underset { P | f _ { n } = 1 } { \mathbf { E } } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { j } ] - \underset { P | f = 1 } { \mathbf { E } } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { j } ] = \frac { \mathbf { E } _ { P } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { j } ] \left\{ f _ { R } ( \mathbf { x } ) = 1 \right\} ] } { p ^ { \prime } } - \frac { \mathbf { E } _ { P } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { j } ] \left\{ f ( \mathbf { x } ) = 1 \right\} ] } { p } } \\ & { \qquad = \frac { \mathbf { E } _ { P } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { j } ( 1 \left\{ f _ { R } ( \mathbf { x } ) = 1 \right\} - 1 \left\{ f ( \mathbf { x } ) = 1 \right\} ) ] } { p ^ { \prime } } + \frac { p - p ^ { \prime } } { p ^ { \prime } } \underset { P | f = 1 } { \mathbf { E } } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { j } ] . } \end{array}
$$

We bound each term separately. Since $R \in { \mathcal { H } } , p \geq \pi _ { 0 }$ and $P ( f \neq f _ { R } ) \leq r _ { 0 } ^ { 1 / q }$ . Therefore,

$$
| p - p ^ { \prime } | \leq { \bf { P r } } [ f ( { \bf { x } } ) \neq f _ { R } ( { \bf { x } } ) \ | \ R ] \leq r _ { 0 } ^ { 1 / q } , \qquad p ^ { \prime } \geq \pi _ { 0 } - r _ { 0 } ^ { 1 / q } \geq { \frac { \pi _ { 0 } } { 2 } } ,
$$

where the last inequality follows from $r _ { 0 } ^ { 1 / q } = ( \sigma _ { * } \pi _ { 0 } \Delta / A _ { q , k } ) ^ { 4 } \leq \pi _ { 0 } / 2 .$

By Fact B.4, the projection of $P$ onto $V _ { R } ^ { \perp }$ is $( 2 q , \sigma _ { * } ) – \mathrm { s m o o t h }$ with respect to the standard Gaussian. Hence Lemma 3.13 and Cauchy–Schwarz give

$$
\begin{array} { r l } & { \left| \underset { P } { \mathbf { E } } \left[ ( \mathbf { v } \cdot \mathbf { x } ) ^ { j } ( \mathbb { 1 } \{ f _ { R } ( \mathbf { x } ) = \mathbb { 1 } \} ) \right] \right| \leq \sqrt { \underset { P } { \mathbf { E } } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { 2 j } ] } \sqrt { P ( f \neq f _ { R } ) } } \\ & { \qquad \leq \frac { \sqrt { 4 q } ( 8 q - 1 ) ^ { k / 2 } M _ { k } } { \sqrt { \sigma _ { * } } } r _ { 0 } ^ { 1 / ( 2 q ) } = \frac { A _ { q , k } } { 8 \sqrt { \sigma _ { * } } } r _ { 0 } ^ { 1 / ( 2 q ) } , } \end{array}
$$

since $\mathbf { v } \cdot \mathbf { x }$ under $D \mid R$ is standard Gaussian. The certified moment-matching guarantee with slack $\Delta / 4$ also implies

$$
\left| \underset { P | f = 1 } { \mathbf { E } } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { j } ] \right| \leq M _ { k } + \Delta / 4 \leq 2 M _ { k } .
$$

Substituting these bounds into the preceding identity yields

$$
\begin{array} { r l } & { \left| \underset { P | f _ { R } = 1 } { \mathbf { E } } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { j } ] - \underset { P | f = 1 } { \mathbf { E } } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { j } ] \right| \leq \frac { 2 } { \pi _ { 0 } } \left( \frac { A _ { q , k } } { 8 \sqrt { \sigma _ { * } } } r _ { 0 } ^ { 1 / ( 2 q ) } + 2 M _ { k } r _ { 0 } ^ { 1 / q } \right) } \\ & { \qquad \leq \frac { A _ { q , k } } { 2 \sigma _ { * } \pi _ { 0 } } r _ { 0 } ^ { 1 / ( 4 q ) } = \frac { \Delta } { 2 } , } \end{array}
$$

where we used $r _ { 0 } \leq 1 , \sigma _ { * } \leq 1$ , and $A _ { q , k } \geq 1 6 M _ { k }$ . Combining this with the partition’s moment slack $\Delta / 4$ shows that $Q _ { R } \mid f _ { R } = 1$ matches the first k Gaussian moments with slack $\Delta .$ . The conditional smoothness of $Q _ { R }$ follows from Fact B.4, and the bound on $p ^ { \prime }$ proves its positive mass assertion. This completes the proof of Claim 5.8. □

Fix $R \in { \mathcal { H } }$ . By Claim 5.8, $Q _ { R }$ is $( 2 q , \sigma _ { * } )$ -smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { V _ { R } ^ { \perp } } )$ . To apply the definition of $m ^ { * }$ on $\mathbb { R } ^ { d }$ , let $T _ { R }$ be the distribution of ${ \mathbf { u } } + { \mathbf { z } } ,$ where ${ \mathbf { u } } \sim Q _ { R }$ and $\mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { V _ { R } } )$ are independent. Independent Gaussian extension preserves the same smoothness parameters: if $E _ { \mathbf { z } } : = \{ \mathbf { u } \in V _ { R } ^ { \perp } : \mathbf { u } + \mathbf { z } \in E \}$ then smoothness and Jensen’s inequality give

$$
T _ { R } ( E ) \leq \frac { 1 } { \sigma _ { * } } { \mathbf { \Delta z } } [ \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { V _ { R } ^ { \perp } } ) ( E _ { \mathbf { z } } ) ^ { 1 / ( 2 q ) } ] \leq \frac { D ( E ) ^ { 1 / ( 2 q ) } } { \sigma _ { * } } .
$$

Expanding directional monomials shows that the corresponding lift of $\operatorname { p r o j } _ { V _ { R } ^ { \perp } } ( D _ { + } ^ { * } \mid R )$ matches the first k moments of D with slack at most $2 ^ { k } M _ { k } \Delta _ { 0 } = \Delta / 4$ . The indicator comparison in the proof of Claim 5.8 applies to these lifted moments as well: under the reference law $\mathbf { x } \sim D \mid R ,$ independently of $\mathbf { z } ,$ the vector $\mathrm { p r o j } _ { V _ { R } ^ { \perp } } \mathbf { x } + \mathbf { z }$ is standard Gaussian, and $f _ { R } ( \mathrm { p r o j } _ { V _ { R } ^ { \perp } } { \bf x } + { \bf z } ) = f _ { R } ( { \bf x } )$ . For the moment-comparison bound, use Cauchy–Schwarz under $( \mathbf { x } , \mathbf { z } ) \sim ( D ^ { * } \mid R ) \otimes \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { V _ { R } } )$ , bound the disagreement probability by $r _ { 0 } ^ { 1 / q }$ , and bound the polynomial second moment using the $( 2 q , \sigma _ { * }$ )-smooth law $T _ { R }$ . Thus the same bounds show that $T _ { R } \ : | \ : f _ { R } = 1$ matches the first k moments of D with slack $\Delta$

Since $f _ { R } \in \mathcal { F } , T _ { R }$ is $( 2 q , \sigma _ { * } )$ -smooth with respect to D and $T _ { R } ( f _ { R } = 1 ) > 0$ , and $k \geq m ^ { * } ( \eta , \Delta , \sigma _ { * } , 2 q )$ 2 the definition of $m ^ { * }$ implies $T _ { R } ( f _ { R } = - 1 ) < \eta$ . As $f _ { R }$ depends only on $V _ { R } ^ { \perp }$ , we obtain

$$
\operatorname* { P r } _ { \mathbf { x } \sim D ^ { * } } [ f _ { R } ( \mathbf { x } ) = - 1 \mid R ] = Q _ { R } ( f _ { R } = - 1 ) = T _ { R } ( f _ { R } = - 1 ) < \eta .
$$

Consequently, using the initial false-positive bound and Claim 5.8,

$$
\begin{array} { l } { \displaystyle \mathbf { P r } _ { \mathbf { r } } \left[ h ( \mathbf { x } ) = 1 , f ( \mathbf { x } ) = - 1 \right] \leq \sum _ { R \in \mathcal { H } } D ^ { * } ( R ) \operatorname* { \mathbf { P r } } _ { \mathbf { x } \sim D ^ { * } } [ f _ { R } ( \mathbf { x } ) = - 1 \mid R ] + \sum _ { R \in \mathcal { G } \backslash \mathcal { H } } D ^ { * } ( R ) + \frac { A ^ { 1 / q } } { \sigma } } \\ { \displaystyle \leq 3 \eta + \frac { 1 } { \sigma } \left( \frac { A } { r _ { 0 } } \right) ^ { 1 / q } + \frac { A ^ { 1 / q } } { \sigma } } \\ { \displaystyle \leq 3 \eta + O \left( \frac { 1 } { \sigma } \left( \frac { ( \delta \Gamma ^ { 2 } ) ^ { 1 / 3 } } { r _ { 0 } } \right) ^ { 1 / q } \right) . } \end{array}
$$

This completes the proof of Lemma 5.7.

Set $\eta : = \epsilon / 8 , \sigma _ { \ast } : = \alpha \eta / ( 4 0 0 q ) , k : = m ^ { \ast } ( \eta , \Delta , \sigma _ { \ast } , 2 q )$ . Take the internal moment slack to be $\Delta _ { 0 } : =$ $\Delta / ( 2 ^ { k + 2 } M _ { k } )$ . The partition guarantees diam $( R ) \leq \delta$ on every certified cell and has at most $B$ cells. The choice in Algorithm 4 satisfies

$$
\delta = \Gamma ^ { - 2 } \left( \frac { \alpha \eta \Delta } { C q k } \right) ^ { C q ( k + q ) } \leq \frac { c _ { 0 } r _ { 0 } ^ { 3 } } { \Gamma ^ { 2 } } \left( \frac { \sigma \epsilon } { 8 } \right) ^ { 3 q } ,
$$

where $c _ { 0 } > 0$ is a suficiently small fixed universal constant and $C > 1$ is suficiently large. Indeed, $M _ { k } ^ { 2 } \le ( 2 k ) ^ { k } , \alpha \le \sigma$ , and $\sigma _ { * } \pi _ { 0 } = ( \alpha \eta ) ^ { 2 } ( \sigma \eta ) ^ { q - 1 } / ( 1 0 ^ { 4 } q )$ give

$$
\begin{array} { c } { { \log \displaystyle \frac { 1 } { c _ { 0 } r _ { 0 } ^ { 3 } ( \sigma \eta ) ^ { 3 q } } = \log \displaystyle \frac { 1 } { c _ { 0 } } + 1 2 q \log \displaystyle \frac { A _ { q , k } } { \sigma _ { * } \pi _ { 0 } \Delta } + 3 q \log \displaystyle \frac { 1 } { \sigma \eta } } } \\ { { = { \cal O } \left( q ( k + q ) \log \displaystyle \frac { 2 q k } { \alpha \eta \Delta } \right) . } } \end{array}
$$

For suficiently large $C ,$ this is at most $C q ( k + q ) \log ( C q k / ( \alpha \eta \Delta ) )$ , proving the displayed inequality. The Gaussian freezing error $A$ from the proof above satisfies $A = O ( ( \delta \Gamma ^ { 2 } ) ^ { 1 / 3 } )$ . By the preceding inequality, a suficiently small universal $c _ { 0 }$ ensures $A \le r _ { 0 } ( \sigma \epsilon / 8 ) ^ { q }$ . As $r _ { 0 } \leq 1$ , this gives

$$
\frac { 1 } { \sigma } \left( \frac { A } { r _ { 0 } } \right) ^ { 1 / q } + \frac { A ^ { 1 / q } } { \sigma } \leq \epsilon / 4 .
$$

Combining Claim 5.6 and Lemma 5.7, on the joint success event, gives

$$
\underset { \mathbf { x } \sim D ^ { * } } { \mathbf { P r } } [ h ( \mathbf { x } ) \neq f ( \mathbf { x } ) ] { \le } 4 \eta + 3 \lambda + \epsilon / 4 < \epsilon ,
$$

since $\eta = \epsilon / 8$ and $\lambda = \eta / 1 0 0$ . By Claim 3.17, the partitioning routine succeeds with probability at least $9 9 / 1 0 0 ;$ conditional on its successful output, the estimation event fails with probability at most $1 / 1 0 0$ Thus the joint event has probability at least $9 8 / 1 0 0 > 2 / 3$

Complexity. By Theorem 3.1, applied to $D _ { + } ^ { * }$ with moment slack $\Delta _ { 0 }$ and diameter bound $\delta ,$ the partitioning stage uses at most $d ^ { k } B$ samples and runs in time $d ^ { O ( k ) } B$ , with $B$ as above. The ratio test uses an additional $O ( B ^ { 2 } \log ( 2 B ) ) ,$ ) positive samples and time polynomial in $B$ and $d .$ These costs are absorbed by increasing the universal constants in the exponential below. Substituting the parameter choices and simplifying the dimension-independent factors, a suficient sample bound is

$$
N = d ^ { k } \mathrm { e x p } \left( { \frac { ( q k ) ^ { O ( k ) } } { \Delta ^ { O ( 1 ) } } } \log ^ { O ( k ) } { \frac { 2 } { \sigma \epsilon p _ { + } } } \log ( 2 + \Gamma ( \mathcal { F } ) ) \right) ,
$$

with running time polynomial in $N .$

Similar to agnostic learning that we discussed in Subsection D.1, for PUI learning we can also remove the assumption of having sample access to distribution D. In particular, assuming $D ^ { * }$ is smooth with respect to some unknown mixture of Gaussians or some unknown Gaussian, we can even perform eficient PUI learning with only sample access to $D _ { + } ^ { * }$

Corollary 5.9 (PUI learning under Gaussian mixtures and unknown means). Let $\mathcal { F }$ satisfy the hypotheses of Theorem 5.3, let $f \in { \mathcal { F } }$ , and suppose that $D ^ { * } ( f = 1 ) \geq p _ { + } > 0 . \ I f D ^ { * } \leq Q / \sigma$ , where $\begin{array} { r } { Q = \sum _ { i = 1 } ^ { s } \lambda _ { i } \mathcal { N } ( \pmb { \mu } _ { i } , \mathbf { I } _ { d } ) } \end{array}$ and $\| \mu _ { i } \| _ { 2 , } \leq$ M for every $i ,$ then the guarantees of Theorem 5.3 hold with $q = 2$ and $\sigma$ replaced by $\bar { \sigma } = \sigma e ^ { - M ^ { 2 } / 2 }$

Suppose additionally that $\mathcal { F }$ is closed under translations. If $D ^ { * } \leq \mathcal { N } ( \mu , \mathbf { I } _ { d } ) / \sigma$ for an unknown $\pmb { \mu } _ { ; }$ , the same guarantees hold with $q = 2$ and $\bar { \sigma } = e ^ { - 1 } \sigma ^ { 3 } p _ { + } ^ { 2 }$ . The algorithm requires sample access only to $D _ { + } ^ { * }$ ; neither knowledge of $\pmb { \mu }$ nor samples from $\mathcal { N } ( \pmb { \mu } , \mathbf { I } _ { d } )$ are needed.

Proof. In the mixture case, Lemma D.1 gives

$$
D ^ { * } ( E ) \leq \frac { Q ( E ) } { \sigma } \leq \frac { e ^ { M ^ { 2 } / 2 } } { \sigma } { \mathcal { N } } ( { \bf 0 } , { \bf I } _ { d } ) ( E ) ^ { 1 / 2 }
$$

for every measurable event $E .$ . Thus Theorem 5.3 applies with parameters $( 2 , { \bar { \sigma } } )$

For the unknown-mean case, conditioning on the positives gives $D _ { + } ^ { * } \leq \mathcal { N } ( \pmb { \mu } , \mathbf { I } _ { d } ) / ( \sigma p _ { + } )$ . The Gaussian moment-generating formula and Jensen’s inequality imply

$$
\left\| { \bf { \underline { { E } } } } _ { \bf { \Delta } \times \Delta \mathcal { D } _ { + } ^ { * } } [ { \bf { x } } ] - \pmb { \mu } \right\| _ { 2 } \leq \sqrt { 2 \log \frac { 1 } { \sigma p _ { + } } } .
$$

Gaussian tail bounds also give $\mathbf { E _ { x \sim D _ { \perp } ^ { * } } } [ \| \mathbf { x } - \pmb { \mu } \| _ { 2 } ^ { 2 } ] = O ( d \log ( 2 / ( \sigma p _ { + } ) ) )$ . Consequently, using $O ( d \log ( 2 / ( \sigma p _ { + } ) ) )$ positive samples, we can estimate their mean by m with Euclidean error at most 1, with high constant probability. On this event,

$$
\| \pmb { \mu } - \mathbf { m } \| _ { 2 } ^ { 2 } \leq 4 \log \frac { 1 } { \sigma p _ { + } } + 2 .
$$

The centered target distribution is dominated by $\mathcal { N } ( \pmb { \mu } - \mathbf { m } , \mathbf { I } _ { d } ) / \sigma$ . By Lemma D.1, it is therefore $( 2 , { \bar { \sigma } } )$ -smooth with respect to the standard Gaussian, since

$$
\sigma e ^ { - \| \mu - \mathbf { m } \| _ { 2 } ^ { 2 } / 2 } \geq e ^ { - 1 } \sigma ^ { 3 } p _ { + } ^ { 2 } = \bar { \sigma } .
$$

Apply Theorem 5.3 to fresh centered positive samples. Translation closure ensures that the centered target concept belongs to ${ \mathcal F } .$ . If the learner returns $h ,$ output $H ( \mathbf { x } ) = h ( \mathbf { x } - \mathbf { m } )$ , which preserves the target error. Choosing the constant failure probabilities in mean estimation and partitioning suficiently small gives the stated success guarantee. The additional mean-estimation cost is absorbed into the sample and running-time bounds. □

## 5.2 SQ Lower Bound for PUI Learning of MIMs with Gaussian Reference

Here we show that a dependence similar to $d ^ { m ^ { * } }$ in Theorem 5.3 is essentially unavoidable for $\mathrm { S Q }$ learning algorithms when $\mathcal { F }$ is a class of K-multi-index models. The intuition comes directly from the definition of $m ^ { * }$ . In particular, there exists a K-dimensional instance $D ^ { * }$ whose positive conditional distribution $D _ { + } ^ { * }$ matches the first $m ^ { * } - 1$ moments of a standard Gaussian, even though $D ^ { * }$ assigns at least ϵ probability mass to negative examples.

We use this moment-matching instance to construct a hard instance of non-Gaussian component analysis (NGCA). Specifically, we randomly choose a K-dimensional subspace U of $\mathbb { R } ^ { d }$ and construct a distribution $Q$ such that its marginal over U is $D _ { + } ^ { * }$ , while its marginal over $\mathbf { U } ^ { \perp }$ is an independent standard Gaussian. Since $D _ { + } ^ { * }$ matches the first $m ^ { * } - 1$ Gaussian moments, distinguishing $Q$ from $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ is an instance of the SQ-hard NGCA problem studied in [DKS17, DKRS23]. We also refer the reader to Section A for background on SQ learning.

To obtain the lower bound, suppose that there exists an SQ algorithm that learns the corresponding PUI learning problem to suficiently small error. Given an unknown input distribution that is either Q or $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , we run this learner using the input distribution as the positive-example distribution and $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ as the unlabeled distribution. Let $\widehat { h }$ be the resulting hypothesis. We then compare the probability that bh predicts a negative label under the input distribution with the corresponding probability under $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$

If the input distribution is $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , these two probabilities are identical. On the other hand, if the input distribution is $Q _ { i }$ , the learning guarantee forces $\widehat { h }$ to predict a significant fraction of negative examples under $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , since the underlying instance $D ^ { * }$ has at least ϵ negative mass. At the same time, because $Q$ is the positive conditional distribution, the learner must predict negative labels on only a small fraction of $Q .$ . Consequently, the two probabilities difer significantly, allowing us to distinguish $Q$ from $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ This contradicts the $\mathrm { S Q }$ lower bound for NGCA and establishes the claimed dependence on d and $m ^ { * }$

Proof of Theorem $5 . 4$ . Write $G _ { j } : = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { j } )$ . Let $\mu$ be the distribution of Wx for $\mathbf { x } \sim D ^ { * }$ , and let $\mu _ { + } ~ : = ~ \mu ~ | ~ \{ g ( \mathbf { z } ) ~ = ~ 1 \}$ . Since the conditioning event depends only on Wx, $\mathbf { P r } _ { \mathbf { z } \sim \mu } [ g ( \mathbf { z } ) ~ = ~ - 1 ] ~ =$ $\mathbf { P r _ { x \sim D ^ { * } } } [ f ( \mathbf { x } ) = - 1 ] \geq \epsilon , \mathbf { P r _ { z \sim \mu } } [ g ( \mathbf { z } ) = 1 ] = p _ { + }$ , and $\mu _ { + }$ is the distribution of Wx for $\mathbf { x } \sim D _ { + } ^ { * }$

Every polynomial in $\mathbf { z } \in \mathbb { R } ^ { K }$ of degree at most $m ^ { * }$ becomes a polynomial in x of the same degree after substituting $\mathbf { z } = \mathbf { W } \mathbf { x }$ . Therefore, $\mu _ { + }$ and $G _ { K }$ match moments through degree $m ^ { * }$ . By Fact B.4, $\mu$ is $( q , \sigma )$ -smooth with respect to $G _ { K }$ . Conditioning gives $( q , \sigma p _ { + } )$ -smoothness of $\mu _ { + }$

We will use the following fact about the hardness of NGCA.

Fact 5.10 (NGCA SQ Lower Bound [DKRS23, Theorem 1.5]). Fix $\lambda \in ( 0 , 1 )$ and $a \in ( 0 , ( 1 - \lambda ) / 4 )$ . Let $d , K , r \in \mathbb { Z } _ { + }$ , with $r \geq 2$ even and max $\{ K , r \} \le d ^ { \lambda } /$ log d. Let A be a distribution over $\mathbb { R } ^ { K }$ matching the moments of $G _ { K }$ through degree r.

For $\mathbf { U } \in \mathbb { R } ^ { K \times d }$ satisfying $\mathbf { U U } ^ { \top } = \mathbf { I } _ { K }$ , let $P _ { \mathbf { U } , A }$ denote the distribution of $\mathbf { U } ^ { \top } \mathbf { z } + ( \mathbf { I } _ { d } - \mathbf { U } ^ { \top } \mathbf { U } ) \mathbf { g }$ , where $\mathbf { z } \sim A$ and $\mathbf { g } \sim G _ { d }$ are independent. For suficiently large $d ,$ any SQ algorithm distinguishing $G _ { d }$ from $P _ { \mathbf { U } , A }$ for a uniformly random U, with probability at least $2 / 3$ , requires either $2 ^ { d ^ { \Omega ( a ) } }$ queries or a query of tolerance at most $\dot { C _ { K , r } } d ^ { - ( ( 1 - \lambda ) / 4 - a ) r }$ , where $C _ { K , r } = O \big ( \Gamma ( ( K + r ) / 2 ) / \Gamma ( K / 2 ) \big )$

For each $\mathbf { U } \in \mathbb { R } ^ { K \times d }$ satisfying $\mathbf { U U } ^ { \top } = \mathbf { I } _ { K }$ , define $f _ { \mathbf { U } } ( \mathbf { x } ) : = g ( \mathbf { U } \mathbf { x } ) , D _ { \mathbf { U } } ^ { * } : = P _ { \mathbf { U } , \mu }$ , and $D _ { + , { \mathbf { U } } } ^ { * } : = P _ { { \mathbf { U } } , \mu _ { + } }$ By the rotation assumption, $f _ { \mathbf { U } } \in \mathcal { F }$ . Since $\mathbf { U } \mathbf { x } = \mathbf { z }$ in the definition of $P _ { \mathbf { U } , A }$

$$
\underset { \mathbf { x } \sim D _ { \mathbf { U } } ^ { * } } { \mathbf { P r } } \left[ f _ { \mathbf { U } } ( \mathbf { x } ) = - 1 \right] = \underset { \mathbf { z } \sim \mu } { \mathbf { P r } } [ g ( \mathbf { z } ) = - 1 ] \geq \epsilon ,
$$

$$
\underset { \mathbf { x } \sim D _ { \mathbf { U } } ^ { * } } { \mathbf { P r } } \left[ f _ { \mathbf { U } } ( \mathbf { x } ) = 1 \right] = p _ { + } ,
$$

$$
D _ { + , { \mathbf { U } } } ^ { * } = D _ { \mathbf { U } } ^ { * } \mid \{ f _ { \mathbf { U } } ( { \mathbf { x } } ) = 1 \} .
$$

Note that $D _ { \mathbf { U } } ^ { * }$ is $( q , \sigma )$ -smooth with respect to $G _ { d } = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ . Indeed, for any measurable event $E \subseteq \mathbb { R } ^ { d }$ smoothness and Jensen’s inequality give

$$
\begin{array} { r l } & { \quad \quad \quad \quad \quad \quad \quad \mathbf { P r } _ { \mathbf { U } } \left[ \mathbf { x } \in E \right] = \underset { \mathbf { g } \sim G _ { d } } { \mathbf { E } } \left[ \mathbf { P r } \left[ \mathbf { U } ^ { \top } \mathbf { z } + ( \mathbf { I } _ { d } - \mathbf { U } ^ { \top } \mathbf { U } ) \mathbf { g } \in E \right] \right] } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \sigma \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \sigma } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad  \end{array}
$$

Here $P _ { \mathbf { U } , G _ { K } } \ = \ G _ { d }$ because, when $\mathbf { z } \sim G _ { K }$ , the independent Gaussian components have covariance matrices summing to $\mathbf { I } _ { d } .$ . Conditioning gives $\begin{array} { r } { \mathbf { P r _ { x \sim D _ { + , U } ^ { * } } } [ \mathbf { x } \in E ] \leq \mathbf { P r _ { x \sim G _ { d } } } [ \mathbf { x } \in E ] ^ { 1 / q } / ( \sigma p _ { + } ) } \end{array}$ , so $D _ { + , { \bf { U } } } ^ { * }$ is $( q , \sigma p _ { + } )$ -smooth.

Set $r : = 2 \lfloor m ^ { * } / 2 \rfloor$ . Then $r \geq 2$ is even and $m ^ { * } / 2 \leq r \leq m ^ { * }$ . Thus $\mu _ { + }$ matches Gaussian moments through degree r. Apply Fact 5.10 with $A = \mu _ { + } , \lambda = 1 / 4$ , and $a = 1 / 1 6$ . For suficiently small c and suficiently large d, the assumption max $\{ K , m ^ { * } \} \stackrel { < } { \simeq } d ^ { c }$ ensures that max $\{ K , r \} \leq d ^ { 1 / 4 } / \log d .$ Furthermore,

$$
\frac { \Gamma ( ( K + r ) / 2 ) } { \Gamma ( K / 2 ) } = \prod _ { j = 0 } ^ { r / 2 - 1 } \left( \frac { K } { 2 } + j \right) \le \left( \frac { K + r } { 2 } \right) ^ { r / 2 } \le d ^ { c r / 2 } .
$$

Consequently, distinguishing $G _ { d }$ from $D _ { + , { \bf { U } } } ^ { * }$ for a uniformly random U requires either $2 ^ { d ^ { \Omega ( 1 ) } }$ queries or a query of tolerance at most $\tau _ { d } : = C _ { K , r } d ^ { - r / 8 }$ , where $\tau _ { d } = { \cal O } \big ( d ^ { - ( 1 / 8 - c / 2 ) r } \big ) = d ^ { - \Omega ( m ^ { * } ) }$

Claim 5.11 (Reduction from PUI Learning to Distinguishing). Suppose an SQ learner, given access only to $D _ { + , { \bf { U } } } ^ { * } ,$ uses at most Q queries and, for every U, outputs $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$ satisfying ${ \bf P r } _ { { \bf x } \sim D _ { { \bf \textup { I I } } } ^ { * } } [ h ( { \bf x } ) \neq $ $f _ { \mathbf { U } } ( \mathbf { x } ) \big ] \leq p _ { + } ( \sigma \epsilon ) ^ { q } / 2$ with probability at least $2 / 3$ . Then it yields an SQ algorithm distinguishing $G _ { d }$ from $D _ { + , { \bf { U } } } ^ { * }$ with probability at least $2 / 3$ . Using the known Gaussian expectation, the algorithm makes the learner’s queries and one additional query of tolerance $( \sigma \epsilon ) ^ { q } / 8$

Proof of Claim 5.11. Let P denote the unknown distribution, which is either $G _ { d }$ $D _ { + , { \bf { U } } } ^ { * }$ . Run the learner using the SQ oracle for P, and let h be its output. Define $a : = \mathbf { P r } _ { \mathbf { x } \sim G _ { d } } [ h ( \mathbf { x } ) = - 1 ]$ . Make one additional SQ query to obtain v satisfying $| v - \mathbf { P r } _ { \mathbf { x } \sim P } [ h ( \mathbf { x } ) = - 1 ] | \leq ( \sigma \epsilon ) ^ { q } / 8$ . Declare the planted case if $a - v > ( \sigma \epsilon ) ^ { q } / 4$ and otherwise declare the Gaussian case.

If $P = G _ { d }$ , then $\mathbf { P r } _ { \mathbf { x } \sim P } [ h ( \mathbf { x } ) = - 1 ] = a$ , regardless of the learner’s output. Hence $| a - v | \leq ( \sigma \epsilon ) ^ { q } / 8$ and the test declares the Gaussian case.

Now suppose $P = D _ { + , { \bf U } } ^ { * }$ and the learner succeeds. Write $e _ { + } : = \mathbf { P r } _ { \mathbf { x } \sim D _ { \tau } ^ { * } } [ f _ { \mathbf { U } } ( \mathbf { x } ) = 1 , h ( \mathbf { x } ) = - 1 ]$ and $e _ { - } : = \mathbf { P r } _ { \mathbf { x } \sim D _ { \mathbf { r } \mathrm { { I } } } ^ { * } } [ f _ { \mathbf { U } } ( \mathbf { x } ) = - 1 , h ( \mathbf { x } ) = 1 ]$ . Then $e _ { + } + e _ { - } \le p _ { + } ( \sigma \epsilon ) ^ { q } / 2$ and $\mathbf { P r _ { x \sim D _ { + , U } ^ { * } } } [ h ( \mathbf { x } ) = - 1 ] = e _ { + } / p _ { + }$ By smoothness and the negative-mass bound,

$$
\begin{array} { r } { \underset { \mathbf { x } \sim G _ { d } } { \mathbf { P r } } [ h ( \mathbf { x } ) = - 1 ] \geq \big ( \sigma \underset { \mathbf { x } \sim D _ { \mathbf { U } } ^ { * } } { \mathbf { P r } } [ h ( \mathbf { x } ) = - 1 ] \big ) ^ { q } \geq \big ( \sigma ( \epsilon - e _ { - } ) \big ) ^ { q } . } \end{array}
$$

Since $p _ { + } \leq 1 - \epsilon$ , we have $p _ { + } q \sigma ^ { q } \epsilon ^ { q - 1 } \le q ( 1 - \epsilon ) \epsilon ^ { q - 1 } \le 1 - \epsilon ^ { q } \le 1$ . Convexity therefore gives

$$
\begin{array} { r l } & { \mathrm { \bf ~ P r } _ { } \left[ h ( { \bf x } ) = - 1 \right] - \underset { { \bf x } \sim { \cal D } _ { + , { \bf U } } ^ { * } } { \bf P r } [ h ( { \bf x } ) = - 1 ] \geq ( \sigma \epsilon ) ^ { q } - q \sigma ^ { q } \epsilon ^ { q - 1 } e _ { - } - \frac { e _ { + } } { p _ { + } } } \\ & { \quad \quad \quad \quad \quad \quad \quad \geq ( \sigma \epsilon ) ^ { q } - \frac { e _ { - } + e _ { + } } { p _ { + } } } \\ & { \quad \quad \quad \quad \quad \geq \frac { ( \sigma \epsilon ) ^ { q } } { 2 } . } \end{array}
$$

Consequently, $a - v \ge 3 ( \sigma \epsilon ) ^ { q } / 8$ , so the test declares the planted case whenever the learner succeeds.

It remains to ensure that the additional query cannot itself supply the precision required by Fact 5.10.   
The parameter assumptions give $( \sigma \epsilon ) ^ { q } / 8 \geq d ^ { - c m ^ { * } } / 8$ . On the other hand, $r \ge m ^ { * } / 2$ implies $\tau _ { d } =$   
$O \big ( d ^ { - } ^ { - } ( 1 / 1 6 - c / 4 ) m ^ { * } \big )$ . Choosing c suficiently small therefore ensures $( \sigma \epsilon ) ^ { q } / 8 > 2 \tau _ { d }$ for suficiently large d. Suppose the learner uses fewer than exponentially many queries and every query has tolerance greater   
than $2 \tau _ { d }$ . By Claim 5.11, this yields an NGCA distinguisher using the learner’s queries and one additional   
query, all with tolerance greater than $\tau _ { d } ,$ contradicting Fact 5.10. Thus the learner requires either $2 ^ { d ^ { \Omega ( 1 ) } }$   
queries or a query of tolerance at most $2 \tau _ { d } = d ^ { - \Omega ( m ^ { * } ) }$ □

Combining Theorem 5.4 with a moment-matching construction for a class of MIMs yields a PUI learning lower bound for that class.

## 5.3 PUI Learning Intersections of Halfspaces and Beyond

As concrete applications of Theorem 5.3 and Theorem 5.4, we derive PUI learning algorithms for the class of halfspaces, intersections of halfspaces, and general K-dimensional convex sets. We also derive matching SQ lower bounds for the class of halfspaces and intersections of halfspaces. Let $\mathcal { F } _ { k }$ denote the class of intersections of at most k halfspaces, including the constant functions. The main results in this section are the following theorems.

We next apply our PUI framework to intersections of halfspaces, obtaining a learner whose exponent in d depends logarithmically on k and $1 / \epsilon$

Theorem 5.12 (PUI Learning Intersections of Halfspaces). Let $k , d \in \mathbb { N } _ { + } , \ \epsilon \in ( 0 , 1 / 2 ) , \ q \geq 1$ , and $\sigma \in ( 0 , 1 ]$ . Let $D = \mathcal { N } ( 0 , I _ { d } )$ and let $D ^ { * }$ be a distribution that is $( q , \sigma )$ -smooth with respect to D. Let $f \in \mathcal { F } _ { k }$ be an unknown target intersection of k halfspaces, and set $p _ { + } : = D ^ { * } ( f = 1 ) > 0 , D _ { + } ^ { * } : = D ^ { * } \mid \{ f = 1 \}$ Set $m : = \lceil C ( \log ( 2 k ) + q \log ( 2 q / ( \sigma \epsilon p _ { + } ) ) ) \rceil$ for a suficiently large universal constant $C > 0$ . There is an algorithm which, given $\epsilon , \sigma , p _ { + } , q$ and sample access to $D _ { + } ^ { * }$ , draws at most

$$
N = d ^ { m } \exp \Bigl ( ( 2 m ) ^ { O ( m ) } \Bigr )
$$

samples. The algorithm runs in time polynomial in N and, with probability at least $2 / 3$ , outputs a hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$ satisfying $\mathbf { P r } _ { x \sim D ^ { * } } [ h ( \mathbf { x } ) \neq f ( \mathbf { x } ) ] \leq \epsilon$

For convex concepts of intrinsic dimension at most K, our PUI general algorithm yields a learner whose exponent in d is linear in K and logarithmic in $1 / \epsilon$

Theorem 5.13 (PUI Learning Low-Dimensional Convex $\mathrm { S e t s } )$ . Let $K , d \in \mathbb { N } _ { + } , \epsilon \in ( 0 , 1 / 2 ) , q \geq 1$ , and $\sigma \in ( 0 , 1 ]$ . Let $D = \mathcal { N } ( 0 , I _ { d } )$ and let $D ^ { * }$ be a distribution that is $( q , \sigma )$ -smooth with respect to D. Let $f \in { \mathcal { C } } _ { K }$ be an unknown target K-dimensional convex set, and set $p _ { + } : = D ^ { * } ( f = 1 ) > 0 , D _ { + } ^ { * } : = D ^ { * } \mid \{ f = 1 \}$ . Set $m : = \lceil C ( K + q \log ( 2 q / ( \sigma \epsilon p _ { + } ) ) ) \rceil$ for a suficiently large universal constant $C > 0$ . There is an algorithm which, given $\epsilon , \sigma , p _ { + } , q$ and sample access to $D _ { + } ^ { * }$ , draws at most

$$
N = d ^ { m } \exp \Bigl ( ( 2 m ) ^ { O ( m ) } \Bigr )
$$

samples. The algorithm runs in time polynomial in N and, with probability at least $2 / 3$ , outputs a hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$ satisfying $\mathbf { P r } _ { x \sim D ^ { * } } [ h ( \mathbf { x } ) \neq f ( \mathbf { x } ) ] \leq \epsilon$

We next give an $\mathrm { S Q }$ lower bound of $d ^ { \Omega ( \log ( k / \epsilon ) ) }$ , matching the dependence on d of our sample complexity upper bound when the smoothness parameters are constant and $p _ { + }$ is bounded below by a constant.

Theorem 5.14 (SQ Lower Bound for PUI Learning Intersections of Halfspaces). There exists a suficiently small universal constant $c > 0$ such that the following holds for suficiently large d. Let $k , d \in \mathbb { Z } _ { + } .$ $\epsilon \in ( 0 , 1 / 2 )$ and $m ^ { * } = \log ( k / \epsilon )$ . Assume that max $\{ k , m ^ { * } \} \leq d ^ { c } , \epsilon \leq c / k ^ { 3 }$ . Then any SQ algorithm that learns $\mathcal { F } _ { k }$ over distributions that are $( 1 , 1 / 2 )$ -smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ , using SQ access $o n l y$ to the positive conditional distribution, to error $\epsilon ( 1 - \epsilon ) / 4$ with probability at least $2 / 3$ , requires either $2 ^ { d ^ { \Omega ( 1 ) } }$ queries $o r$ a query of tolerance at most $d ^ { - \Omega ( m ^ { * } ) }$

The main technical ingredient is to show that $m ^ { * } ( \epsilon , \Delta , \sigma , q ) = O ( \log ( k / ( \sigma \epsilon ) ^ { q } ) )$ , while on the other hand even for $q = 1$ and $\sigma = 1 / 2 , m ^ { * } = \Omega ( \log ( k / \epsilon ) )$ . Together, these show that $d ^ { \Theta ( \log ( k / \epsilon ) ) }$ time complexity is both suficient and necessary for PUI learning intersections of halfspaces with a Gaussian reference.

Upper Bounds for PUI Learning Exponents of Intersections of Halfspaces We start with the proof of Theorem 5.12. The idea of upper bounding $m ^ { * }$ for intersections of k halfspaces is to first derive an upper bound for a single halfspace. Specifically, given a one-dimensional halfspace $\mathbb { 1 } \{ x < t \}$ if $t = \Theta ( \sqrt { \log ( 1 / \epsilon ) } )$ , then under the Gaussian distribution, the fraction of negative examples is roughly ϵ. We show that any distribution supported on $( - \infty , t ]$ cannot match its first $O ( t ^ { 2 } )$ moments within a slack of $1 / 4$ with those of a standard Gaussian. Since an intersection of k halfspaces is a conjunction of k halfspaces, we know that if the fraction of negative examples under Gaussian is at least $\epsilon ,$ there must be at least one halfspace that contains at least $\epsilon / k$ negative examples under Gaussian. Combining this fact with the structural results on a single halfspace, we know that any distribution sampled from the positive region of the intersection cannot match $\log ( k / \epsilon )$ moments up to a constant slack. Notice that once this is proven, a smoothness assumption immediately gives an upper bound of $m ^ { * } = O ( \log ( 1 / ( \sigma \epsilon ) ^ { q } ) )$ for a single halfspace.

Lemma 5.15 (Structural Result for Halfspaces). Let $G = \mathcal { N } ( 0 , 1 )$ , let $t \geq 0$ , and let P be supported on $( - \infty , t ]$ with finite moments through degree $2 \lceil 1 + t ^ { 2 } \rceil + 1$ . Then P and G do not match their first $2 \lceil 1 + t ^ { 2 } \rceil + 1$ moments to slack less than $1 / 4$

Let $\epsilon < 1 / 2$ $I f \mathbf { P r } _ { x \sim G } [ x \geq t ] \geq \epsilon ;$ , then $t = O ( \sqrt { \log ( 1 / \epsilon ) } )$ . Hence, P and G cannot match all moments through degree $\lfloor C \log ( 1 / \epsilon ) \rfloor$ to slack less than 1/4 for a suficiently large constant $C > 0$

Proof of Lemma 5.15. Let $r = \lceil 1 + t ^ { 2 } \rceil , a = { \sqrt { 2 r - 1 } } , n = 2 r - 1 = a ^ { 2 }$ and define the polynomial $q ( x ) = ( x - t ) x ^ { 2 r - 2 } ( x + a ) ^ { 2 }$

Since $q \leq 0$ on the support of $P _ { \mathrm { : } }$ we have $\mathbf { E } _ { x \sim P } [ q ( x ) ] \leq 0$ . Denote $\mu _ { r } : = { \bf E } _ { x \sim G } [ x ^ { 2 r - 2 } ] \ge 1$ . Expanding q and using that the odd Gaussian moments vanish, we get

$$
\begin{array} { r l } & { \underset { x \sim G } { \mathbf { E } } [ q ( x ) ] = ( 2 a - t ) \underset { x \sim G } { \mathbf { E } } [ x ^ { 2 r } ] - t a ^ { 2 } \underset { x \sim G } { \mathbf { E } } [ x ^ { 2 r - 2 } ] } \\ & { = \big [ ( 2 a - t ) ( 2 r - 1 ) - t a ^ { 2 } \big ] \mu _ { r } } \\ & { = \big [ n ( 2 a - t ) - t a ^ { 2 } \big ] \mu _ { r } } \\ & { = 2 a ^ { 2 } ( a - t ) \mu _ { r } . } \end{array}
$$

Here the second line uses $\mathbf { E } _ { x \sim G } [ x ^ { 2 r } ] = \left( 2 r - 1 \right) \mathbf { E } _ { x \sim G } [ x ^ { 2 r - 2 } ] = ( 2 r - 1 ) \mu _ { r }$ , and the last line uses $n = a ^ { 2 }$ Write $B = 2 a ^ { 2 } ( a - t )$ . Since $n \geq 2 t ^ { 2 } + 1$ , we have $| t | / a \leq 1 / \sqrt { 2 }$ , so $B > 0$

Note that expanding q gives that the sum of the absolute values of its non-constant monomial coeficients is

$$
C = 1 + | 2 a - t | + | a ^ { 2 } - 2 a t | + \Im \{ r \geq 2 \} | t | a ^ { 2 } .
$$

Therefore $P$ and $G$ cannot match their first $2 r + 1$ moments to slack less than $( B / C ) \mu _ { r }$

It remains to bound $B / C .$ . If $t \leq 0$ , set $v = - t / a$ . Since $a \ge 1$

$$
C \leq 1 + ( 2 + v ) a + ( 1 + 2 v ) a ^ { 2 } + v a ^ { 3 } \leq 4 ( 1 + v ) a ^ { 3 } = 2 B .
$$

If $t > 0$ , then $r \geq 2$ , so $a \geq { \sqrt { 3 } } .$ . Setting $v = t / a \leq 1 / \sqrt { 2 }$ , we obtain

$$
C \leq 1 + 2 a + a ^ { 2 } + { \frac { a ^ { 3 } } { \sqrt 2 } } < { \frac { 9 } { 4 } } a ^ { 3 } , \qquad B = 2 ( 1 - v ) a ^ { 3 } \geq ( 2 - \sqrt 2 ) a ^ { 3 } > { \frac { 9 } { 1 6 } } a ^ { 3 } .
$$

Thus $B / C > 1 / 4$ in this case as well. Since $\mu _ { r } \geq 1$ , we have $( B / C ) \mu _ { r } \geq 1 / 4$ , proving the claim. □

For an intersection of k halfspaces, the union bound implies that one defining halfspace excludes at least a $1 / k$ share of that mass. Applying the single-halfspace result to the projection onto its normal vector then gives a moment discrepancy in that direction. Now we state our result for intersections of halfspaces.

Corollary 5.16 (Structural Result for Intersections of Halfspaces). There exists a suficiently large constant $C > 0$ such that the following holds. Let $G = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ , and let $f$ be an intersection of k halfspaces such that $\mathbf { P r } _ { \mathbf { x } \sim G } [ f ( \mathbf { x } ) = - 1 ] \geq \epsilon .$ , where $0 < \epsilon < 1 / 2$ . Let P be a distribution supported on $\{ \mathbf { x } : f ( \mathbf { x } ) = 1 \}$ that has finite moments through degree $m = C \lceil \log ( k / \epsilon ) \rceil$ . Then, P and G do not match moments through degree m to slack less than $1 / 4$ in the directional sense of Definition 2.2.

Proof of Corollary 5.16. Denote by

$$
H = \bigcap _ { i = 1 } ^ { k } \{ \mathbf x : \mathbf v _ { i } \cdot \mathbf x \leq t _ { i } \} , \qquad \| \mathbf v _ { i } \| _ { 2 } = 1
$$

the positive region of $f .$ . Since

$$
H ^ { c } = \bigcup _ { i = 1 } ^ { k } \{ \mathbf x : \mathbf v _ { i } \cdot \mathbf x > t _ { i } \} ,
$$

the union bound implies that some $i \in [ k ]$ satisfies

$$
\underset { \mathbf { x } \sim G } { \mathbf { P r } } [ \mathbf { v } _ { i } \cdot \mathbf { x } > t _ { i } ] \geq \frac { \epsilon } { k } .
$$

Assume first that $t _ { i } \geq 0$ . Under $G ,$ , the random variable $\mathbf { v } _ { i } \cdot \mathbf { x }$ has distribution $\mathcal { N } ( 0 , 1 )$ . Under $P ,$ it is supported on $( - \infty , t _ { i } ]$

Therefore, by Lemma $5 . 1 5 ,$ , the distribution of $\mathbf { v } _ { i } \cdot \mathbf { x }$ under $P$ cannot match the Gaussian moments through degree $m = C \lceil \log ( k / \epsilon ) \rceil$ with slack less than $1 / 4$ , for a suficiently large universal constant $C > 0$

If $t _ { i } \leq 0$ , then the distribution of $\mathbf { v } _ { i } \cdot \mathbf { x }$ under $P$ is also supported on $( - \infty , 0 ]$ . Applying Lemma 5.15 with $t = 0$ rules out matching the first three Gaussian moments with slack less than $1 / 4$ . Since $m \geq 3$ for a suficiently large $C _ { i }$ , this also rules out moment matching through degree $m$ . In either case, $P$ and G cannot match moments through degree m with slack less than $1 / 4$ in the sense of Definition 2.2, completing the proof. □

We apply Theorem 5.3 to intersections of halfspaces. Let $\mathcal { F } _ { k }$ denote the class of intersections of at most k halfspaces, including the constant functions. By Corollary 5.16, for every fixed $\Delta \in ( 0 , 1 / 4 )$ $m ^ { * } ( \epsilon , \Delta , \sigma , q ) = { \cal O } \left( \log ( 2 k / ( \sigma \epsilon ) ^ { q } ) \right)$ . Indeed, $( q , \sigma )$ -smoothness implies that if $D ^ { * } ( f = - 1 ) \geq \epsilon .$ , then $D ( f = - 1 ) \geq ( \sigma \epsilon ) ^ { q }$ . Combining this bound with Theorem 5.3 yields the following corollary.

Proof of Theorem 5.12. The class $\mathcal { F } _ { k }$ is closed under afine restrictions and satisfies $\Gamma ( \mathcal { F } _ { k } ) = O ( k )$ . Taking $\Delta = 1 / 8$ , Corollary 5.16 gives

$$
m ^ { * } \bigg ( \frac { \epsilon } { 8 } , \frac { 1 } { 8 } , \frac { \epsilon \sigma p _ { + } } { 1 2 8 0 0 q } , 2 q \bigg ) = O \bigg ( \log ( 2 k ) + q \log \frac { 2 q } { \sigma \epsilon ^ { 2 } p _ { + } } \bigg ) \leq m .
$$

Run Algorithm 4 with moment order $m ;$ its proof only requires the order to be at least the displayed PUI learning exponent. Since $\Gamma ( \mathcal { F } _ { k } ) = O ( k )$ and $q , \log ( 2 / ( \sigma \epsilon p _ { + } ) ) , \log ( 2 k ) = O ( m )$ , the exponential factor in Theorem 5.3 is at most

$$
\exp \Biggl ( ( q m ) ^ { O ( m ) } \log ^ { O ( m ) } \Biggl ( \frac { 2 } { \sigma \epsilon p _ { + } } \Biggr ) \log ( 2 k ) \Biggr ) \leq \exp \Bigl ( ( 2 m ) ^ { O ( m ) } \Bigr ) .
$$

Thus the sample complexity is $d ^ { m } \exp ( ( 2 m ) ^ { O ( m ) } )$ and the running time is polynomial in this quantity. The error guarantee follows from Theorem 5.3. Here k counts halfspaces and m is the moment order used by the PUI algorithm. □

Upper Bounds for PUI Learning Exponents of Low Dimensional Convex Sets Similar analysis also holds for general K-dimensional convex sets. The idea is as follows. Suppose that $C ^ { * } \subseteq \mathbb { R } ^ { K }$ is a K-dimensional convex set such that $\mathbf { P r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { K } ) } [ \mathbf { x } \notin C ^ { * } ] > \epsilon$ . Then we claim that there is some halfspace $h = \{ \mathbf { x } \mid \mathbf { u } \cdot \mathbf { x } \leq t \}$ for some moderate $t > 0$ that contains $C ^ { * }$ . In other words, $C ^ { * }$ must exclude some point that is not extremely far from the origin. By Lemma 5.15, we know that any distribution sampled from $h$ cannot match $\Omega ( 1 + t ^ { 2 } )$ many moments with respect to a Gaussian, and therefore the distribution sampled from $C ^ { * }$ cannot. In fact, we can show that the threshold t can be chosen as $O ( \sqrt { K } + \sqrt { \log ( 1 / \epsilon ) } )$ we know that the PUI learning exponent of a K-dimensional convex set is at most $O ( K + q \log ( 1 / ( \sigma \epsilon ) ) )$ ).

Lemma 5.17 (Structural Results for K-Dimensional Convex Sets). There exists a suficiently large constant $C > 0$ such that the following holds. Let $G = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ , and let $C ^ { * }$ be a K-dimensional convex set such that $\mathbf { P r } _ { \mathbf { x } \sim G } [ \mathbf { x } \notin C ^ { * } ] \ge \epsilon$ , where $0 < \epsilon < 1 / 2$ . Let P be a distribution supported on $C ^ { * }$ that has finite moments through degree $m = C \lceil ( K + \log ( 1 / \epsilon ) ) \rceil$ . Then, P and G do not match moments through degree m to slack less than $1 / 4$ in the directional sense of Definition 2.2.

Proof of Lemma 5.17. Without loss of generality, we assume that $C ^ { * } \subseteq \mathbb { R } ^ { K }$ . Let $R ^ { 2 } = 2 K \log ( 2 ) +$ $4 \log ( 2 / \epsilon )$ . Using the bound on the norm of a K dimensional Gaussian vector, we know that $\mathbf { P r } _ { \mathbf { x } \sim \mathcal { N } ( 0 , \mathbf { I } _ { K } ) } [ \| \mathbf { x } \| >$ $R ] < \epsilon / 2$ . Since $\mathbf { P r } _ { \mathbf { x } \sim G } [ \mathbf { x } \notin C ^ { * } ] \ge \epsilon$ , we know that there must be some point $\mathbf { x } ^ { * }$ such that $\| \mathbf { x } ^ { * } \| \leq R .$ , while $\mathbf { x } ^ { * } \notin C ^ { * }$ . Thus, applying the separating hyperplane theorem to $C ^ { * }$ and $\mathbf { x } ^ { * }$ , we know that there exists a halfspace $h = \{ \mathbf { x } \mid \mathbf { u } \cdot \mathbf { x } \leq t \}$ , where $\mathbf { u } \in \mathbb { S } ^ { K - 1 }$ and $0 \leq t < R$ such that $C ^ { * } \subseteq h$

Since P has support $C ^ { * }$ , we know that $P$ is also supported on $h .$ By Lemma 5.15, we know that P cannot match $O ( 1 + t ^ { 2 } ) = O ( K + \log ( 2 / \epsilon ) )$ moments with a standard Gaussian within a slack of $1 / 4 . \ \square$

Proof of Theorem 5.13. The class $\mathcal { C } _ { K }$ is closed under afine restrictions and satisfies $\Gamma ( \mathcal { C } _ { K } ) = O ( K ^ { 1 / 4 } )$ Taking $\Delta = 1 / 8$ , Lemma 5.17 gives

$$
m ^ { * } \bigg ( \frac { \epsilon } { 8 } , \frac { 1 } { 8 } , \frac { \epsilon \sigma p _ { + } } { 1 2 8 0 0 q } , 2 q \bigg ) = O \bigg ( K + q \log \frac { 2 q } { \sigma \epsilon ^ { 2 } p _ { + } } \bigg ) \leq m .
$$

Run Algorithm 4 with moment order $m ;$ its proof only requires the order to be at least the displayed PUI learning exponent. Since $\Gamma ( \mathcal { C } _ { K } ) = O ( K ^ { 1 / 4 } )$ the exponential factor in Theorem 5.3 is at most

$$
\exp \Biggl ( ( q m ) ^ { O ( m ) } \log ^ { O ( m ) } \Biggl ( \frac { 2 } { \sigma \epsilon p _ { + } } \Biggr ) \log ( 2 K ) \Biggr ) \leq \exp \Bigl ( ( 2 m ) ^ { O ( m ) } \Bigr ) .
$$

Thus the sample complexity is $d ^ { m } \exp ( ( 2 m ) ^ { O ( m ) } )$ and the running time is polynomial in this quantity. The error guarantee follows from Theorem 5.3. □

Lower Bounds for PUI Learning Exponents of Intersections of Halfspaces We next investigate the tightness of Corollary 5.16. We seek an intersection f of k halfspaces and a distribution $D ^ { \star } \leq 2 G \ ( { \mathrm { i . e . } }$ $\sigma = 1 / 2 )$ , such that $\mathbf { P r } _ { \mathbf { x } \sim D ^ { \star } } [ f ( \mathbf { x } ) = - 1 ] \geq \epsilon$ while the positive conditional distribution $D _ { + } ^ { \star } = D ^ { \star } \mid f ( { \bf x } ) = 1$ matches G through degree $\Omega ( \log ( k / \epsilon ) )$ . The construction below shows that the logarithmic moment bound is tight up to constants in the regime $\epsilon \leq c / k ^ { 3 }$ , for a suficiently small universal constant $c > 0$

We first settle the case of a single halfspace. In particular we prove that there exists a moment matching distribution supported on a halfspace that is very close to the Gaussian (the domination parameter approaches 1) of nontrivial negative mass. The construction deletes the Gaussian right tail and repairs its missing moments by a bounded correction on a central interval.

Lemma 5.18 (Moment Matching Lower bound for Halfspaces). Let $G = \mathcal { N } ( 0 , 1 )$ , let $t \geq 6$ , and set $\epsilon : = \mathbf { P r } _ { x \sim G } [ x > t ]$ and $m : = \lfloor t ^ { 2 } / 3 2 \rfloor$ . There exists a probability distribution Q supported on $( - \infty , t ]$ such that $Q \leq ( \bar { 1 ^ { ' } } + 2 e ^ { - 5 t ^ { 2 } / 3 2 } ) G$ and Q matches the Gaussian moments through degree m. Moreover, for the halfspace $f _ { t } ( x ) : = 2 \mathbb { 1 } \{ x \leq t \} - 1$ , there exists a probability distribution $D ^ { \star } \leq ( 1 + 2 e ^ { - 5 t ^ { 2 } / 3 2 } ) G$ such that $\mathbf { P r } _ { x \sim D ^ { \star } } [ f _ { t } ( x ) = - 1 ] = \epsilon , D _ { + } ^ { \star } : = D ^ { \star } \mid \{ f _ { t } ( x ) = 1 \} = Q$

Proof of Lemma 5.18. Define the vector of Hermite polynomials $\mathbf v ( x ) : = ( h _ { 0 } ( x ) , \dots , h _ { m } ( x ) ) ^ { \top }$ . Removing the Gaussian tail $( t , \infty )$ loses the Hermite moments

$$
{ \mathbf { b } } : = \mathbf { \phi } _ { x \sim G } [ \mathbb { 1 } \{ x > t \} { \mathbf { v } } ( x ) ] .
$$

Since $h _ { 0 } = 1$ , the first coordinate of b is the lost probability mass ϵ.

We will restore these moments by adjusting the density on the interval $I : = \left\lceil - R , R \right\rceil$ , where $R : =$ $\sqrt { 8 ( m + 1 ) }$ . Since $t \geq 6$ , the choice of m gives $m \geq 1$ and $R ^ { 2 } \leq t ^ { 2 } / 4 + 8 < t ^ { 2 }$ , so $I \subseteq ( - \infty , t ]$ In particular, we seek a correction polynomial of the form $r ( x ) = \mathbf { v } ( x ) ^ { \top } \mathbf { c }$ satisfying

$$
\underset { x \sim G } { \mathbf { E } } [ \mathbb { 1 } \{ x \in I \} r ( x ) \mathbf { v } ( x ) ] = \mathbf { b } .
$$

Substituting r shows that it sufices to solve the linear system $\mathbf { A } \mathbf { c } = \mathbf { b }$ , where

$$
\mathbf { A } : = \mathbf { \underset { x \sim G } { E } } [ \mathbb { 1 } \{ x \in I \} \mathbf { v } ( x ) \mathbf { v } ( x ) ^ { \top } ] .
$$

The following claim shows that such a system is solvable.

Claim 5.19 (Invertibility of A). We have $\mathbf { A } \succeq \frac 1 2 \mathbf { I } _ { m + 1 }$ . In particular, A is invertible and $\| \mathbf { A } ^ { - 1 } \| _ { \mathrm { o p } } \leq 2$

Proof of Claim 5.19. Fix $\mathbf { a } \in \mathbb { R } ^ { m + 1 }$ and write $p ( x ) : = \mathbf { a } ^ { \top } \mathbf { v } ( x )$ . By orthonormality, $\| p \| _ { L _ { 2 } ( G ) } = \| \mathbf { a } \| _ { 2 }$ We will use the following recurrence about Hermite polynomials $x h _ { j } ( x ) = \sqrt { j + 1 } h _ { j + 1 } ( x ) + \sqrt { j } h _ { j - 1 } ( x )$ Applying the recurrence, the triangle inequality, and orthonormality within each of the two sums gives

$$
\| x p \| _ { L _ { 2 } ( G ) } \leq \left( \sum _ { j = 0 } ^ { m } ( j + 1 ) \mathbf { a } _ { j } ^ { 2 } \right) ^ { 1 / 2 } + \left( \sum _ { j = 1 } ^ { m } j \mathbf { a } _ { j } ^ { 2 } \right) ^ { 1 / 2 } \leq 2 \sqrt { m + 1 } \| p \| _ { L _ { 2 } ( G ) } .
$$

On $\{ | x | > R \}$ , we have $1 \leq x ^ { 2 } / R ^ { 2 }$ . Hence

$$
\underset { x \sim G } { \mathbf { E } } [ \mathbb { 1 } \{ | x | > R \} p ( x ) ^ { 2 } ] \leq \frac { \mathbf { E } _ { x \sim G } [ x ^ { 2 } p ( x ) ^ { 2 } ] } { R ^ { 2 } } \leq \frac { 4 ( m + 1 ) } { R ^ { 2 } } \| \mathbf { a } \| _ { 2 } ^ { 2 } = \frac { 1 } { 2 } \| \mathbf { a } \| _ { 2 } ^ { 2 } .
$$

Subtracting this bound from $\mathbf { E } _ { \widetilde { x } \widetilde { \sim } G } [ p ( x ) ^ { 2 } ] = \| \mathbf { a } \| _ { 2 } ^ { 2 }$ yields $\mathbf { a } ^ { \top } \mathbf { A } \mathbf { a } \geq \frac { 1 } { 2 } \| \mathbf { a } \| _ { 2 } ^ { 2 }$ , proving Claim 5.19.

By Claim 5.19, we may take $\mathbf { c } = \mathbf { A } ^ { - 1 } \mathbf { b }$ . With this choice of $^ { r , }$ define $w ( x ) : = \mathbb { 1 } \{ x \leq t \} + \mathbb { 1 } \{ x \in I \} r ( x )$ We show that reweighting G by w preserves its moments through degree m.

Claim 5.20 (Moment Matching). For every polynomial p of degree at most m,

$$
\underset { x \sim G } { \mathbf { E } } [ w ( x ) p ( x ) ] = \underset { x \sim G } { \mathbf { E } } [ p ( x ) ] .
$$

In particular, ${ \bf E } _ { x \sim G } [ w ( x ) ] = 1$

Proof of Claim 5.20. The definitions of A and r give

$$
\underset { x \sim G } { \mathbf { E } } [ \mathbb { 1 } \{ x \in I \} r ( x ) \mathbf { v } ( x ) ] = \underset { x \sim G } { \mathbf { E } } [ \mathbb { 1 } \{ x \in I \} \mathbf { v } ( x ) \mathbf { v } ( x ) ^ { \top } ] \mathbf { A } ^ { - 1 } \mathbf { b } = \mathbf { b } .
$$

Consequently,

$$
\underset { x \sim G } { \mathbf { E } } [ w ( x ) \mathbf { v } ( x ) ] = \underset { x \sim G } { \mathbf { E } } [ \mathbf { v } ( x ) ] - \mathbf { b } + \mathbf { b } = \underset { x \sim G } { \mathbf { E } } [ \mathbf { v } ( x ) ] .
$$

Every polynomial of degree at most m is a linear combination of $h _ { 0 } , \ldots , h _ { m } .$ , so the asserted equality follows by linearity. Taking $p = 1$ gives normalization. This concludes the proof of Claim 5.20. □

It remains to bound the correction term r to show that w is nonnegative and hence that $\mathrm { d } Q = w \mathrm { d } G$ defines a probability distribution.

Claim 5.21 (Bounded Correction). The polynomial r satisfies $\begin{array} { r } { \operatorname* { s u p } _ { x \in I } | r ( x ) | \leq 2 e ^ { - 5 t ^ { 2 } / 3 2 } } \end{array}$

Proof of Claim 5.21. We start with the bound

$$
\operatorname* { s u p } _ { x \in I } | r ( x ) | \leq \operatorname* { s u p } _ { x \in I } \| \mathbf { v } ( x ) \| _ { 2 } \| \mathbf { A } ^ { - 1 } \| _ { \mathrm { o p } } \| \mathbf { b } \| _ { 2 } .
$$

By Claim 5.19, the middle factor is at most 2. It remains to bound $\| \mathbf b \| _ { 2 }$ and $\operatorname { s u p } _ { x \in I } \| \mathbf { v } ( x ) \| _ { 2 }$

For every $\mathbf { a } \in \mathbb { R } ^ { m + 1 }$ with $\| \mathbf { a } \| _ { 2 } = 1$ , Cauchy–Schwarz gives

$$
| \mathbf { a } ^ { \top } \mathbf { b } | \leq { \sqrt { \mathbf { P r } \left[ x > t \right] } } \sqrt { \mathbf { E } _ { x \sim G } [ ( \mathbf { a } ^ { \top } \mathbf { v } ( x ) ) ^ { 2 } ] } = { \sqrt { \epsilon } } .
$$

Thus $\| \mathbf { b } \| _ { 2 } \leq { \sqrt { \epsilon } }$ . We next bound v on I. Set $B : = R + { \sqrt { m } }$ . We show by induction that $| h _ { j } ( x ) | \leq B ^ { j } / \sqrt { j ! }$ for $x \in I$ and $0 \leq j \leq m$ . The cases $j = 0 , 1$ follow from $h _ { 0 } = 1$ and $h _ { 1 } ( x ) = x$ . For the induction step, the Hermite recurrence gives

$$
| h _ { j + 1 } ( x ) | \leq \frac { | x | | h _ { j } ( x ) | + \sqrt { j } | h _ { j - 1 } ( x ) | } { \sqrt { j + 1 } } \leq \frac { R B ^ { j } + j B ^ { j - 1 } } { \sqrt { ( j + 1 ) ! } } \leq \frac { B ^ { j + 1 } } { \sqrt { ( j + 1 ) ! } } ,
$$

where we used $R B + j \le R B + m \le B ^ { 2 }$ . Since $B \geq \sqrt { m }$ , the sequence $B ^ { j } / \sqrt { j ! }$ is nondecreasing for $0 \leq j \leq m$ . Also, $R \leq 4 { \sqrt { m } }$ , so $B \leq 5 { \sqrt { m } }$ . Using m! $\ge ( m / e ) ^ { m }$ , we obtain

$$
| h _ { j } ( x ) | \leq { \frac { B ^ { m } } { \sqrt { m ! } } } \leq ( 5 { \sqrt { e } } ) ^ { m } \qquad ( x \in I , \ 0 \leq j \leq m ) .
$$

As $m + 1 \le 2 ^ { m }$ for $m \geq 1$ , this implies

$$
\operatorname* { s u p } _ { x \in I } \| \mathbf { v } ( x ) \| _ { 2 } \leq { \sqrt { m + 1 } } ( 5 { \sqrt { e } } ) ^ { m } \leq ( 5 { \sqrt { 2 e } } ) ^ { m } \leq e ^ { 3 m } .
$$

Combining the three bounds gives $\begin{array} { r } { \operatorname* { s u p } _ { x \in I } | r ( x ) | \le 2 e ^ { 3 m } \sqrt { \epsilon } } \end{array}$ . The Gaussian tail bound gives $\epsilon \leq e ^ { - t ^ { 2 } / 2 }$ Therefore, using $m \leq t ^ { 2 } / 3 2$ , we obtain $2 e ^ { 3 m } \sqrt { \epsilon } \le 2 e ^ { 3 t ^ { 2 } / 3 \dot { 2 } - t ^ { 2 } / 4 } = 2 e ^ { - 5 t ^ { 2 } / 3 2 }$ . This concludes the proof of Claim 5.21. □

Note that for $t \geq 6 , \ \operatorname* { s u p } _ { x \in I } | r ( x ) | < 1$ . Hence, since $I \subseteq ( - \infty , t ]$ , Claim 5.21 implies $0 \leq w \leq$ $1 + 2 e ^ { - 5 t ^ { 2 } / 3 2 }$ and $w = 0 \mathrm { o n } \ ( t , \infty )$ . By Claim 5.20, the measure $Q$ defined by $\mathrm { d } Q = w \mathrm { d } G$ is therefore a probability distribution matching the Gaussian moments through degree m.

To construct the full target distribution, set $D ^ { \star } : = ( 1 - \epsilon ) Q + \epsilon G ( { \bf \cdot } | x > t )$ . Since $0 < \epsilon < 1$ , this is a probability distribution. Its density ratio with respect to $G$ is

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } D ^ { \star } } { \mathrm { d } G } ( x ) = ( 1 - \epsilon ) w ( x ) + \epsilon \frac { 1 \{ x > t \} } { \epsilon } } \\ { = ( 1 - \epsilon ) w ( x ) + 1 \{ x > t \} } \\ { \displaystyle \qquad \leq \operatorname* { m a x } \Big \{ ( 1 - \epsilon ) ( 1 + 2 e ^ { - 5 t ^ { 2 } / 3 2 } ) , 1 \Big \} } \\ { \displaystyle \qquad \leq 1 + 2 e ^ { - 5 t ^ { 2 } / 3 2 } , } \end{array}
$$

because the two summands have disjoint supports. The two mixture components are supported on $\{ f _ { t } ( x ) = 1 \}$ and $\{ f _ { t } ( x ) = - 1 \}$ , respectively. Thus $\mathbf { P r } _ { x \sim D ^ { \star } } [ f _ { t } ( x ) = - 1 ] = \epsilon , \mathbf { P r } _ { x \sim D ^ { \star } } [ f _ { t } ( x ) = 1 ] = 1 - \epsilon ,$ and $D _ { + } ^ { \star } = Q$ , completing the proof. □

Now we generalize this lower bound to intersections of halfspaces.

Lemma 5.22 (Moment Matching Lower Bound for Intersections of Halfspaces). There exists a universal constant $c > 0$ such that the following holds. Let $k , d \in \mathbb { Z } _ { + }$ with $k \leq d$ and $\epsilon \in ( 0 , 1 / 2 )$ . Assume that $\epsilon \leq c / k ^ { 3 }$ . There exists an intersection of k halfspaces $f : \mathbb { R } ^ { d }  \{ \pm 1 \}$ and a probability distribution $D ^ { * } \leq 2 \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ such that $\mathbf { P r } _ { \mathbf { x } \sim D ^ { * } } [ f ( \mathbf { x } ) = - 1 ] = \epsilon$ and $D _ { + } ^ { * } : = D ^ { * } \mid \{ f ( \mathbf { x } ) = 1 \}$ matches the Gaussian moments through total degree $\Omega ( \log ( k / \epsilon ) )$

Proof. Set $\tau : = 1 - ( 1 - \epsilon ) ^ { 1 / k }$ and let t satisfy $\mathbf { P r } _ { z \sim \mathcal { N } ( 0 , 1 ) } [ z > t ] = \tau$ . Since $\epsilon / k \le \tau \le 2 \epsilon / k \le 2 c _ { ; }$ taking c suficiently small ensures that $t \geq 6$ . Set $\delta : = 2 e ^ { - 5 t ^ { 2 } / 3 2 }$ and $m : = \lfloor t ^ { 2 } / 3 2 \rfloor$ . Applying Lemma 5.18 with tail mass $\tau _ { : }$ , we obtain a probability distribution $D ^ { \star }$ satisfying $D ^ { \star } \leq ( 1 + \delta ) \mathcal { N } ( 0 , 1 )$ and $\mathbf { P r } _ { z \sim D ^ { \star } } [ z > t ] = \tau$ ， such that $D ^ { \star } \mid \{ z \leq t \}$ matches the Gaussian moments through degree m.

Define

$$
D _ { k } ^ { \star } : = ( D ^ { \star } ) ^ { \otimes k } \times \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d - k } ) , \qquad H : = \bigcap _ { i = 1 } ^ { k } \{ \mathbf { x } \in \mathbb { R } ^ { d } : \mathbf { x } _ { i } \leq t \} , \qquad f ( \mathbf { x } ) : = 2 \mathbb { 1 } \{ \mathbf { x } \in H \} - 1 .
$$

We will show that $D _ { k } ^ { \star }$ satisfies the lemma’s conditions. Write $D _ { k , + } ^ { \star } : = D _ { k } ^ { \star } \mid \{ f ( \mathbf { x } ) = 1 \}$ . First, we prove the moment matching condition.

Claim 5.23. $D _ { k , + } ^ { \star }$ matches the moments of $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ through total degree $m$

Proof of Claim 5.23. Since H is a coordinatewise intersection, independence gives $D _ { k , + } ^ { \star } = ( D ^ { \star } \mid \{ z \leq$ $t \ v { \} } ) ^ { \otimes k } \times \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d - k } )$ . Fix a multi-index $\alpha \in \mathbb { Z } _ { > 0 } ^ { d }$ with $| \alpha | \le m$ . Each $\alpha _ { i } \leq m .$ , so one-dimensional moment matching and independence give

$$
\underset { \substack { \mathbf { x } \sim D _ { k , + } ^ { * } } } { \mathbf { E } } [ \mathbf { x } ^ { \alpha } ] = \left( \prod _ { i = 1 } ^ { k } \underset { z \sim D \setminus D ^ { * } } { \mathbf { E } } [ z ^ { \alpha _ { i } } \mid z \leq t ] \right) \prod _ { i = k + 1 } ^ { d } \underset { z \sim N ( 0 , 1 ) } { \mathbf { E } } [ z ^ { \alpha _ { i } } ] = \prod _ { i = 1 } ^ { d } \underset { z \sim N ( 0 , 1 ) } { \mathbf { E } } [ z ^ { \alpha _ { i } } ] = \underset { \mathbf { x } \sim \sqrt { ( 0 , \mathbf { I } _ { d } ) } } { \mathbf { E } } [ \mathbf { x } ^ { \alpha _ { i } } ]
$$

The claim follows by linearity of expectation.

Second, we bound the density ratio of $D _ { k } ^ { \star }$ with respect to the Gaussian.

Claim 5.24. It holds that $D _ { k } ^ { \star } \le ( 1 + \delta ) ^ { k } \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } ) \le 2 \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$

Proof $o f \ C l a i m \ 5 . \mathcal { 2 } 4$ . Since $D ^ { \star } \leq ( 1 + \delta ) \mathcal { N } ( 0 , 1 )$ , taking products gives $D _ { k } ^ { \star } = ( D ^ { \star } ) ^ { \otimes k } \times \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d - k } ) \ \leq$ $( 1 + \delta ) ^ { k } \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ For $t \geq 6$ , the Gaussian tail lower bound gives $\tau \geq \frac { t } { \sqrt { 2 \pi } ( t ^ { 2 } + 1 ) } e ^ { - t ^ { 2 } / 2 } \geq e ^ { - 5 t ^ { 2 } / 8 }$ . Consequently, $\delta =$ $2 e ^ { - 5 t ^ { 2 } / 3 2 } \leq 2 \tau ^ { 1 / 4 }$ . Using $\tau \leq 2 \epsilon / k$ and $\epsilon \leq c / k ^ { 3 }$ , we obtain $k \delta \leq 2 k \tau ^ { 1 / 4 } \leq 2 ( 2 k ^ { 3 } \epsilon ) ^ { 1 / 4 } \leq 2 ( 2 c ) ^ { 1 / 4 }$ . Taking c suficiently small therefore gives $( 1 + \delta ) ^ { k } \leq e ^ { k \delta } \leq 2 \qquad $ , proving the claim. □

Finally, we compute the negative mass of f under $D _ { k } ^ { \star }$

Claim 5.25. It holds that $\mathbf { P r _ { x \sim D _ { k } ^ { \star } } } [ f ( \mathbf { x } ) = - 1 ] = 1 - ( 1 - \tau ) ^ { k } .$

Proof of Claim 5.25. Each of the first k coordinates independently satisfies $\begin{array} { r } { \mathbf { P r } _ { \mathbf { x } \sim D _ { k } ^ { \star } } [ \mathbf { x } _ { i } \le t ] = 1 - \tau } \end{array}$ . Thus k $\mathbf { P r } _ { \mathbf { x } \sim D _ { k } ^ { \star } } [ f ( \mathbf { x } ) = 1 ] = ( 1 - \tau ) ^ { k }$ , and taking the complement proves the claim. □

By the choice $\tau = 1 - ( 1 - \epsilon ) ^ { 1 / k }$ , Claim 5.25 gives $\mathbf { P r } _ { \mathbf { x } \sim D _ { \imath } ^ { \star } } [ f ( \mathbf { x } ) \ = \ - 1 ] \ = \ \epsilon$ . By Claim 5.24, $D _ { k } ^ { \star } \leq 2 \mathcal { N } ( { \bf 0 } , { \bf I } _ { d } )$ . Moreover, Gaussian tail estimates give $t ^ { 2 } = \Theta ( \log ( \ddot { 1 } / \tau ) )$ . Since $t \geq 6$ and $\epsilon / k \le \tau \le 2 \epsilon / k$ we have $m = \lfloor t ^ { 2 } / 3 2 \rfloor = \Omega ( \log ( k / \epsilon ) )$ . Together with Claim 5.23, this shows that $D _ { k } ^ { \star }$ is the required distribution. □

Proof of Theorem 5.14. By Lemma 5.22, there exist an intersection f of k halfspaces and a distribution $D ^ { * } \leq 2 \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ such that $\mathbf { P r } _ { \mathbf { x } \sim D ^ { * } } [ f ( \mathbf { x } ) = - 1 ] = \epsilon$ and $D _ { + } ^ { * }$ matches Gaussian moments through degree $m = \Theta ( m ^ { * } )$ . The construction depends only on the first k coordinates, and $\mathcal { F } _ { k }$ contains every rotation of f. Moreover, $p _ { + } = 1 - \epsilon \ge 1 / 2$ . For suficiently small c and suficiently large d, the stated parameter assumptions satisfy the dimension and mass conditions of Theorem 5.4, with $q = 1 , \sigma = 1 / 2 , K = k$ , and moment degree m. Applying that theorem gives the claimed lower bound. □

## 6 Agnostic Reliable Learning

As a third application, we show that the PUI learning algorithm through Gaussian partitioning also directly gives eficient learning algorithms for one-sided agnostic reliable learning defined as follows.

Definition 6.1 (Agnostic Reliable learning). Let ${ \mathcal { F } } \subseteq \{ \mathbb { R } ^ { d } \to \{ \pm 1 \} \}$ be a class of Boolean functions containing the function $f ( \mathbf { x } ) = - 1$ and let D be a distribution on $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ . We say that an algorithm positively reliably learns $\mathcal { F }$ under D if it returns a hypothesis h such that $\mathbf { P r } _ { ( \mathbf { x } , y ) \sim D } [ h ( \mathbf { x } ) = 1 , y = - 1 ] \leq \epsilon$ and

$$
\underset { ( \mathbf { x } , y ) \sim D } { \mathbf { P r } } [ h ( \mathbf { x } ) = - 1 , y = 1 ] \leq \underset { f \in \mathcal { G } _ { D } } { \operatorname* { i n f } } \underset { ( \mathbf { x } , y ) \sim D } { \mathbf { P r } } [ f ( \mathbf { x } ) = - 1 , y = 1 ] + \epsilon ,
$$

where $\mathcal { G } _ { D } = \left\{ f \in \mathcal { F } : \mathbf { P r } _ { ( \mathbf { x } , y ) \sim D } [ f ( \mathbf { x } ) = 1 , y = - 1 ] = 0 \right\}$

Negative reliable learning is defined analogously, with the roles of the labels 1 and −1 interchanged and with $\mathcal { F }$ containing the constant function $f ( \mathbf { x } ) = 1$

Our main result is that if a concept class F has a bounded PUI learning exponent $m ^ { * }$ , then it also implies a reliable learning algorithm in time $d ^ { O ( m ^ { * } ) }$ under Gaussian marginal distribution.

Theorem 6.2 (Agnostic Reliable Learning under Gaussian Distribution). Let F be a class of functions that is closed under afine restrictions and has bounded Gaussian surface area $\Gamma ( \mathcal { F } ) < \infty$ (satisfies items $( 1 )$ and (3) of Definition $4 . 1 )$ and contains the constant function 1. Let D be a distribution over $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ whose x-marginal is $( q , \sigma )$ -smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ , where $q \geq 1$ and $\sigma \in ( 0 , 1 ]$ . Let $\epsilon \in ( 0 , 1 / 2 )$ and $\Delta \in ( 0 , 1 / 2 )$ , and set $m : = m ^ { * } ( c \epsilon , \Delta , c \sigma \epsilon ^ { 3 } / q , 2 q )$ for a suficiently small universal constant $c > 0$ . Assume $m < \infty$ . There is an algorithm which, given $\epsilon , \sigma , \Delta , q$ and sample access to $D ,$ uses at most

$$
{ \cal N } = d ^ { m } \exp \Bigl ( ( q m ) ^ { O ( m ) } \log ^ { O ( m ) } ( 1 / ( \sigma \epsilon ) ) \log ( 2 + \Gamma ( \mathcal { F } ) ) / \Delta ^ { O ( 1 ) } \Bigr )
$$

labeled samples, runs in time polynomial in $N _ { \cdot }$ , and negatively reliably learns F under D with probability at least $2 / 3$

The proof of Theorem 6.2 relies on the following reduction from agnostic reliable learning to PUI learning.

Lemma 6.3 (Reduction from Reliable Learning to Smoothed PUI Learning). Let $G = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ , let D be a distribution over $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ whose x-marginal is $( q , \sigma )$ -smooth with respect to $G ,$ where $q \geq 1$ and $\sigma \in ( 0 , 1 ]$ , and let $\epsilon \in ( 0 , 1 / 2 )$ . Assume that $\mathcal { F }$ contains the constant function 1 and admits a $P U I$ learner A with failure probability at most $1 / 1 2$ , sample complexity $N ( \epsilon , \sigma , p _ { + } , q )$ , and running time $T ( \epsilon , \sigma , p _ { + } , q )$ where $p _ { + } > 0$ is a given lower bound on the probability that the target function equals 1.

Then there exists an algorithm that negatively reliably learns $\mathcal { F }$ under D with probability at least $2 / 3$ using $O ( N ( \epsilon / 2 , \sigma \epsilon / 4 , \epsilon / 4 , q ) / \epsilon + 1 / \epsilon ^ { 2 } )$ labeled samples and $O ( T ( \epsilon / 2 , \sigma \epsilon / 4 , \epsilon / 4 , q ) / \epsilon + 1 / \epsilon ^ { 2 } )$ time.

Consequently, if $\mathcal { F }$ contains the constant function −1 and −F admits such a PUI learner, then $\mathcal { F }$ is positively reliably learnable under D, with the same complexity bounds for the learner $o f - \mathcal { F }$

The intuition of Lemma 6.3 is simple. To perform agnostic reliable learning, it is suficient to learn a hypothesis with error ϵ over the distribution $D ^ { * }$ , which is the distribution conditioned on the examples with $f ^ { * } ( \mathbf { x } ) = y$ . However, we do not have sample access to $D ^ { * }$ directly. Fortunately, the assumption of reliable learning guarantees that $D \mid \{ y = 1 \} = D ^ { * } \mid \{ f ^ { * } ( \mathbf { x } ) = 1 \}$ . Since $D ^ { * }$ is smooth, by running our PUI learning algorithm over $D \mid \{ y = 1 \}$ , we immediately get a desired hypothesis.

Proof of Lemma 6.3. We prove the negative reliable case. Let $\rho : = \mathbf { P r } _ { ( \mathbf { x } , y ) \sim D } [ y = 1 ]$ . Using $O ( \epsilon ^ { - 2 } )$ labeled samples, obtain an estimate $\widehat { \rho }$ such that $| \widehat { \rho } - \rho | \leq \epsilon / 4$ with probability at least $1 1 / 1 2$ . Condition on this event. If $\widehat { \rho } \leq \epsilon / 2$ , return the constant hypothesis $h = - 1$ . Its false-negative error is $\rho \le 3 \epsilon / 4$ , and its false-positive error is zero, so it satisfies the required guarantees.

Otherwise, $\rho > \epsilon / 4$ . Let OPT denote the negative reliable optimum, and choose $f ^ { * } \in { \mathcal { F } }$ such that

$$
\begin{array} { r } { \underset { ( \mathbf { x } , y ) \sim D } { \mathbf { P r } } [ f ^ { * } ( \mathbf { x } ) = - 1 , y = 1 ] = 0 , \quad \quad \underset { ( \mathbf { x } , y ) \sim D } { \mathbf { P r } } [ f ^ { * } ( \mathbf { x } ) = 1 , y = - 1 ] \leq \mathrm { O P T } + \epsilon / 2 . } \end{array}
$$

Define $A : = \{ y = f ^ { * } ( \mathbf { x } ) \}$ , write $w : = \mathbf { P r } _ { ( \mathbf { x } , y ) \sim D } [ ( \mathbf { x } , y ) \in A ]$ , and let $D ^ { * }$ be the distribution of x under D conditioned on $A$ . Every positive example belongs to $A _ { i }$ , so $w \ge \rho > \epsilon / 4$ . Consequently, for every measurable $E \subseteq \mathbb { R } ^ { d }$

$$
\underset { \substack { \mathbf { x } \sim D ^ { * } } } { \mathbf { P r } } [ \mathbf { x } \in E ] \leq \frac { \mathbf { P r } _ { \mathbf { x } \sim D _ { \mathbf { x } } } [ \mathbf { x } \in E ] } { w } \leq \frac { \mathbf { P r } _ { \mathbf { x } \sim G } [ \mathbf { x } \in E ] ^ { 1 / q } } { \sigma w } \leq \frac { \mathbf { P r } _ { \mathbf { x } \sim G } [ \mathbf { x } \in E ] ^ { 1 / q } } { \sigma \epsilon / 4 } .
$$

Thus $D ^ { * }$ is $( q , \sigma \epsilon / 4 )$ -smooth with respect to $G .$ Moreover, $D ^ { * } \mid \{ f ^ { * } ( { \bf x } ) = 1 \} = D | y = 1$ and $\mathbf { P r } _ { \mathbf { x } \sim D ^ { * } } [ f ^ { * } ( \mathbf { x } ) = 1 ] = \rho / w \geq \rho > \epsilon / 4$ . Thus we can supply the PUI learner with the required conditional samples by rejection sampling from D, without knowing $f ^ { * } , D ^ { * }$ , or $\mathrm { O P T }$

Write $n : = \lceil N ( \epsilon / 2 , \sigma \epsilon / 4 , \epsilon / 4 , q ) \rceil$ . Draw fresh labeled samples from D and retain their x-coordinates exactly when $y = 1$ , until n samples have been accepted or $M : = \lceil C ( n + 1 ) / \epsilon \rceil$ labeled samples have been drawn. If fewer than n samples are accepted, return $h = - 1$ . On the event considered above, each draw is accepted independently with probability $\rho > \epsilon / 4$ . The number $X$ of accepted samples among M draws has mean $\mu = M \rho > C ( n + 1 ) / 4$ . For a suficiently large absolute constant $C .$ , a Chernof bound gives

$$
\mathbf { P r } [ X < n ] \leq \mathbf { P r } [ X \leq \mu / 2 ] \leq \exp ( - \mu / 8 ) \leq \exp ( - C ( n + 1 ) / 3 2 ) \leq 1 / 1 2 .
$$

The accepted points are independent samples from $D | y = 1$ , including conditional on obtaining n of them before the cutof, since the cutof depends only on the labels. This procedure uses $O ( ( n + 1 ) / \epsilon )$ labeled samples and time.

Run $\mathcal { A }$ at accuracy $\epsilon / 2$ , with smoothness parameters $( q , \sigma \epsilon / 4 )$ and positive-mass lower bound $\epsilon / 4$ using the accepted samples. Enforce its stated sample and running-time limits, returning $h = - 1$ if either is exceeded. On the valid branch, these limits do not afect the PUI guarantee. The learner yields a hypothesis h satisfying $\mathbf { P r } _ { \mathbf { x } \sim D ^ { * } } [ h ( \mathbf { x } ) \neq f ^ { * } ( \mathbf { x } ) ] \leq \epsilon / 2$ with probability at least $1 1 / 1 2$ . On this success event,

$$
\begin{array} { r } { \underset { ( \mathbf { x } , y ) \sim { D } } { \mathbf { P r } } [ h ( \mathbf { x } ) = - 1 , y = 1 ] = w \underset { \mathbf { x } \sim { D } ^ { * } } { \mathbf { P r } } [ h ( \mathbf { x } ) = - 1 , f ^ { * } ( \mathbf { x } ) = 1 ] \leq w \epsilon / 2 \leq \epsilon / 2 . } \end{array}
$$

Furthermore,

$$
\operatorname* { P r } _ { ( \mathbf { x } , y ) \sim D } [ h ( \mathbf { x } ) \neq y ] \leq \operatorname* { P r } _ { ( \mathbf { x } , y ) \sim D } [ ( \mathbf { x } , y ) \in A ^ { c } ] + w \operatorname* { P r } _ { \mathbf { x } \sim D ^ { * } } [ h ( \mathbf { x } ) \neq f ^ { * } ( \mathbf { x } ) ] \leq \mathrm { O P T } + \epsilon .
$$

In particular, the false-positive error is at most ${ \mathrm { O P T } } + \epsilon .$ . The mass estimate, rejection sampling, and PUI learner each have failure probability at most $1 / 1 2$ , so the overall success probability is at least $3 / 4$ , and hence at least $2 / 3$

The preliminary mass estimate uses $O ( \epsilon ^ { - 2 } )$ samples and time. Together with the rejection-sampling cutof, this gives the stated sample bound. The total running time is

$$
O \left( T ( \epsilon / 2 , \sigma \epsilon / 4 , \epsilon / 4 , q ) + \frac { N ( \epsilon / 2 , \sigma \epsilon / 4 , \epsilon / 4 , q ) + 1 } { \epsilon } + \frac { 1 } { \epsilon ^ { 2 } } \right) .
$$

Since $T \geq N$ and $\epsilon < 1$ , this is bounded by the running time in the statement.

For the positive reliable case, apply the negative reliable reduction to the relabeled distribution $\left( \mathbf { x } , - y \right)$ and the class $- \mathcal { F }$ , which contains the constant function 1. The samples supplied to its PUI learner are drawn from D conditioned on $y = - 1$ . If that learner returns ${ \widehat { g } } ,$ return $h = - { \widehat { g } } .$ . The two reliable guarantees follow by interchanging the labels. □

Combining Theorem 5.3 with Lemma 6.3 yields the proof of Theorem 6.2.

Proof of Theorem 6.2. First amplify the success probability of the PUI learner in Theorem 5.3 to at least 11/12 by taking the pointwise majority of a suficiently large constant number of independent runs at accuracy $\epsilon / 2 4$ . By the Hoefding bound, with probability at least $1 1 / 1 2 .$ , at least $7 / 1 2$ of the runs succeed. Whenever the majority is wrong, at least $1 / 1 2$ of all runs are successful runs that are wrong at that point; hence the majority’s error is at most $1 2 ( \epsilon / 2 4 ) = \epsilon / 2$

Lemma 6.3 invokes this learner with smoothness parameter $\sigma ^ { \prime } = \sigma \epsilon / 4$ and positive-mass lower bound $p _ { + } ^ { \prime } = \epsilon / 4$ . Each independent PUI run uses accuracy $\epsilon ^ { \prime } = \epsilon / 2 4$ . Thus the PUI algorithm uses $\alpha = \sigma ^ { \prime } p _ { + } ^ { \prime } \dot { / } 4 = \sigma \epsilon ^ { 2 } / 6 4$ and conditional smoothness parameter $\sigma _ { * } = { \alpha } { \epsilon } ^ { \prime } / ( 3 2 0 0 q ) = \Theta ( \sigma { \epsilon } ^ { 3 } / q )$ . For suficiently small $c ,$ the moment order required by each run is at most m.

The parameter substitution gives

$$
\log \frac { 1 } { \sigma ^ { \prime } \epsilon ^ { \prime } p _ { + } ^ { \prime } } = O \biggl ( \log \frac { 1 } { \sigma \epsilon } \biggr ) .
$$

Applying the explicit PUI bounds from the proof of Theorem 5.3, and using $\log ( 1 / \Delta ) \leq 1 / \Delta$ , yields a suficient sample bound

$$
d ^ { m } \exp \Bigl ( ( q m ) ^ { O ( m ) } \log ^ { O ( m ) } ( 1 / ( \sigma \epsilon ) ) \log ( 2 + \Gamma ( \mathcal { F } ) ) / \Delta ^ { O ( 1 ) } \Bigr ) .
$$

Here the exponent $O ( 1 )$ of $1 / \Delta$ is universal and independent of m. Logarithms of $1 / ( \sigma \epsilon )$ from the resolution are absorbed into lo $\stackrel { \cdot } { \mathrm { g } } ^ { \hat { O } ( m ) } ( \stackrel { \cdot } { 1 } / ( \sigma \epsilon ) )$ , while the surface-area factor remains log $\left( 2 + \Gamma ( \mathcal { F } ) \right)$ . The constant number of amplification runs, the rejection-sampling factor $1 / \epsilon ,$ and the $O ( \epsilon ^ { - 2 } )$ estimation cost are also absorbed into this exponential factor. The total running time is polynomial in the displayed sample bound. □

As an application of Theorem 6.2 and Theorem 5.12, we immediately obtain reliable learning algorithms for learning intersections and unions of halfspaces.

Corollary 6.4 (Reliable Learning of Intersections and Unions of Halfspaces). Let k, $d \in \mathbb { N } _ { + } , \epsilon \in ( 0 , 1 / 2 )$ ， $q \geq 1$ , and $\sigma \in ( 0 , 1 ]$ . Let D be a distribution over $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ whose x-marginal is $( q , \sigma ) – s m o o t h \ w i t h$ respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ . Set $m : = \lceil C ( \log ( 2 k ) + q \log ( 2 q / ( \sigma \epsilon ) ) ) \rceil$ for a suficiently large universal constant $C > 0$ . There is an algorithm which, given $\epsilon , \sigma , q$ and sample access to $D$ , uses at most

$$
N = d ^ { m } \exp \Bigl ( ( 2 m ) ^ { O ( m ) } \Bigr )
$$

labeled samples, runs in time polynomial in N, and negatively reliably learns the class $\mathcal { F } _ { k }$ of intersections of at most k halfspaces with probability at least $2 / 3$ . With the same sample and running-time bounds, unions of at most k halfspaces are positively reliably learnable under D.

Proof of Corollary $6 . 4 \cdot$ The class $\mathcal { F } _ { k }$ contains the constant functions, is closed under afine restrictions, and satisfies $\Gamma ( \mathcal { F } _ { k } ) = O ( k )$ . Taking $\Delta = 1 / 8$ , Corollary 5.16 gives

$$
\begin{array} { c } { { m ^ { * } ( c \epsilon , 1 / 8 , c \sigma \epsilon ^ { 3 } / q , 2 q ) = O \left( \log ( 2 k ) + q \log \displaystyle \frac { 2 q } { \sigma \epsilon ^ { 4 } } \right) } } \\ { { = O \left( \log ( 2 k ) + q \log \displaystyle \frac { 2 q } { \sigma \epsilon } \right) \leq m . } } \end{array}
$$

Apply Theorem 6.2 with $\Delta = 1 / 8$ and moment order m. Since $\Gamma ( \mathcal { F } _ { k } ) = O ( k )$ and $q , \log ( 1 / ( \sigma \epsilon ) ) , \log ( 2 + k ) =$ $O ( m )$ , its sample bound simplifies to

$$
d ^ { m } \exp \Bigl ( ( q m ) ^ { O ( m ) } \log ^ { O ( m ) } ( 1 / ( \sigma \epsilon ) ) \log ( 2 + k ) \Bigr ) \leq d ^ { m } \exp \Bigl ( ( 2 m ) ^ { O ( m ) } \Bigr ) .
$$

This proves the stated complexity and the negative reliable guarantee. The complement of a union of at most k halfspaces is an intersection of at most k halfspaces, so the positive reliable guarantee follows from the last part of Lemma 6.3. □

## 7 SQ Lower Bound for Realizable TDS and PQ Learning under the Gaussian Distribution

Theorem 6.2 yields eficient algorithms for agnostic reliable learning. For intersections of k halfspaces under the standard Gaussian, our algorithm has dimension dependence $d ^ { O ( \log ( k / \epsilon ) ) }$ . This raises a natural question: can we obtain the same logarithmic dependence on $1 / \epsilon$ in the exponent of d for closely related reliable learning models, such as PQ learning [GKKM20] and TDS learning [KSV24b]? We first recall their definitions.

Definition 7.1 (PQ learning). Let ${ \mathcal { F } } \subseteq \{ \mathbb { R } ^ { d } \to \{ \pm 1 \} \}$ , and let $P , Q$ be distributions on $\mathbb { R } ^ { d }$ . We say that an algorithm $( \epsilon , \delta ) \ / – P Q$ learns $\mathcal { F }$ under $P \ i f , \ f o r$ every target $f \in { \mathcal { F } }$ and every test distribution $Q ,$ given labeled samples $\left( \mathbf { x } , f ( \mathbf { x } ) \right)$ with $\textbf { x } \sim  { P }$ and unlabeled samples $\mathbf { x } \sim Q$ , it returns a selective hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 , \bot \}$ such that, with probability at least $1 - \delta$ $\mathbf { P r _ { x \sim P } } [ h ( \mathbf { x } ) = \bot ] \ \leq \ \epsilon$ and $\begin{array} { r } { \mathbf { P r } _ { \mathbf { x } \sim Q } [ h ( \mathbf { x } ) \neq f ( \mathbf { x } ) , h ( \mathbf { x } ) \neq \bot ] \leq \epsilon } \end{array}$

Definition 7.2 (TDS learning). Let ${ \mathcal { F } } \subseteq \{ \mathbb { R } ^ { d } \to \{ \pm 1 \} \}$ and let P be a distribution on $\mathbb { R } ^ { d }$ . We say that an algorithm $( \epsilon , \delta ) \ – T D S$ learns $\mathcal { F }$ under $P \ i f ,$ for every target $f \in { \mathcal { F } }$ and every test distribution $Q _ { i }$ , given labeled samples $\left( \mathbf { x } , f ( \mathbf { x } ) \right)$ with $\mathbf { x } \sim P$ and unlabeled samples $\mathbf { x } \sim Q$ , it either rejects or accepts and returns a hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$ satisfying:

1. (Completeness.) $I f Q = P$ , the algorithm accepts with probability at least $1 - \delta$

2. (Soundness.) With probability at least $1 - \delta _ { i }$ , whenever the algorithm accepts, its hypothesis satisfies $\mathbf { P r _ { x \sim Q } } [ h ( \mathbf { x } ) \neq f ( \mathbf { x } ) ] \leq \epsilon$

Recent works [KK21, KPSV26] have shown that, in the distribution-free setting, these models are equivalent to agnostic reliable learning. Under a Gaussian training marginal, however, known algorithms based on polynomial regression [KSV24b] have dimension dependence $d ^ { \mathrm { p o l y } ( k / \epsilon ) }$

We show that an improvement analogous to our reliable learning result is impossible in the SQ model, even for realizable learning of intersections of two halfspaces. Specifically, any PQ or TDS learner with SQ access to both the labeled training distribution and the unlabeled test distribution must either make $2 ^ { d ^ { \widetilde { \Omega } ( 1 ) } }$ queries or use tolerance at most $\ : d ^ { - \Omega ( 1 / \epsilon ^ { 2 } ) } \ :$ (see Theorem 7.3). This establishes a separation between the SQ complexities of agnostic reliable learning and realizable PQ/TDS learning in the Gaussian setting, despite their distribution-free equivalence. We remark that [KSV24a] also obtained an SQ lower bound for the class of intersections of two halfspaces in the TDS learning setting. However, they only established a $d ^ { \tilde { \Omega } ( \log ( 1 / \epsilon ) ) }$ lower bound, which is not suficient to establish the separation for our purpose.

Theorem 7.3 (SQ Lower Bound for PQ and TDS learning). Let F be the class of intersections of two halfspaces over $\mathbb { R } ^ { d }$ . There exist universal constants $c , c ^ { \prime } > 0$ such that, for every suficiently small $\epsilon > 0$ and suficiently large d depending on ϵ, the following holds. Any algorithm $( \epsilon , 1 / 2 0 ) – P Q$ learns or (ϵ, 1/20)-TDS learns F under $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , using SQ access to the labeled training distribution and the unlabeled test distribution, must either make at least $2 ^ { d ^ { c ^ { \prime } } }$ SQ queries in total or make a query to either oracle of tolerance at most $d ^ { - c / \epsilon ^ { 2 } }$

Our construction follows directly from the discrete Gaussian distribution used in [DKS17], which constructs a one-dimensional discrete distribution $D _ { \mathbf { v } }$ that matches $\Omega ( 1 / \epsilon ^ { 2 } )$ moments with a one-dimensional standard Gaussian, but puts $\Omega ( \epsilon )$ mass at 0. Using standard NGCA techniques from [DKRS23], any SQ-algorithm would take $\dot { d } ^ { \Omega ( 1 / \mathrm { p o l y } ( \epsilon ) ) }$ time to distinguish a standard Gaussian distribution from $Q _ { \mathbf { v } }$ . But on the other hand, if we set $f ^ { * } = 2 1 \{ \mathbf { x } \mid | \mathbf { v } \cdot \mathbf { x } | \leq \delta  \} - 1$ , for some $\delta$ that is arbitrarily close to 0, then the training distribution would just be $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) \times \{ - 1 \}$ , while a successful PQ/TDS learning algorithm would need to distinguish $Q _ { \nu }$ from $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ so that the trained hypothesis has ϵ error over the test distribution.

Proof of Theorem 7.3. We reduce the problem of non-Gaussian component analysis (NGCA) to PQ or TDS learning. Fix $r = \lfloor c _ { 0 } / \epsilon ^ { 2 } \rfloor$ for a suficiently small universal constant $c _ { 0 } > 0$ . By [DKS17], there exists a discrete distribution B that matches the first $4 r + 1$ moments of $\mathcal { N } ( 0 , 1 )$ and assigns mass

$$
w _ { 0 } = \frac { 4 ^ { r } } { ( 2 r + 1 ) { \binom { 2 r } { r } } } = \Theta ( r ^ { - 1 / 2 } ) \geq 2 0 \epsilon
$$

to zero. For an unknown unit vector $\mathbf { v } ,$ let $Q _ { \mathbf { v } }$ denote the distribution whose marginal along v is B and whose orthogonal marginal is an independent standard Gaussian. By Fact 5.10, distinguishing $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ from $Q _ { \mathbf { v } }$ , for a uniformly random hidden direction v, requires either $2 ^ { \bar { d } ^ { \Omega ( 1 ) } }$ SQ queries or tolerance at most $d ^ { - \Omega ( r ) }$

Suppose that there exists a PQ or TDS SQ learner using at most T SQ queries, each of tolerance at least $\tau > 0$ . Choose $t > 0$ suficiently small that

$$
\rho : = \operatorname* { P r } _ { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) } [ | \mathbf { v } \cdot \mathbf { x } | \leq t ] \leq \operatorname* { m i n } \{ \epsilon , \tau / 8 \} .
$$

Consider the target

$$
f _ { \mathbf { v } } ( \mathbf { x } ) = { \left\{ \begin{array} { l l } { 1 , } & { | \mathbf { v } \cdot \mathbf { x } | \leq t , } \\ { - 1 , } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }
$$

This is an intersection of two general halfspaces. Moreover,

$$
\underset { \mathbf { x } \sim Q _ { \mathbf { v } } } { \mathbf { P r } } \left[ f _ { \mathbf { v } } ( \mathbf { x } ) = 1 \right] \geq w _ { 0 } \geq 2 0 \epsilon .
$$

We construct an SQ distinguisher for NGCA using the assumed learner.

We show that the performance of the SQ learner over the training distribution can be the same as one trained over a Gaussian distribution with examples all negative. Specifically, for any SQ query $\phi : \mathbb { R } ^ { d } \times \{ \pm 1 \}  [ - 1 , 1 ]$ , we return an estimate of

$$
a _ { \phi } = \underset { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) } { \mathbf { E } } [ \phi ( \mathbf { x } , - 1 ) ] .
$$

Since $f _ { \mathbf { v } } ( \mathbf { x } ) = - 1$ outside the slab $| { \bf v } \cdot { \bf x } | \le t$ , we have

$$
\left| a _ { \phi } - \frac { \mathbf { E } } { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 , I } ) } [ \phi ( \mathbf { x } , f _ { \mathbf { v } } ( \mathbf { x } ) ) ] \right| \leq 2 \operatorname* { P r } _ { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 , I } ) } [ f _ { \mathbf { v } } ( \mathbf { x } ) = 1 ] \leq 2 \rho \leq \tau / 4 .
$$

This implies that the performance of the learning algorithm can be seen as identical to a distribution whose label is always negative and the training stage provides no information to the learner.

We provide the unknown NGCA distribution Q as the learner’s unlabeled test SQ oracle.

First suppose that the learner is a TDS learner. If it rejects, declare that the test distribution is $Q _ { \mathbf { v } }$ Otherwise, let h be its output and query $p = \mathbf { P r } _ { \mathbf { x } \sim Q } [ h ( \mathbf { x } ) = 1 ]$ with tolerance ϵ.

If $Q = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , completeness guarantees that the learner accepts, and soundness gives

$$
p \leq \operatorname* { P r } _ { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) } [ h ( \mathbf { x } ) \neq f _ { \mathbf { v } } ( \mathbf { x } ) ] + \rho \leq 2 \epsilon .
$$

If $Q = Q \mathbf { \sigma } _ { \mathbf { v } }$ , soundness implies that, whenever the learner accepts,

$$
p \geq \underset { \mathbf { x } \sim Q _ { \mathbf { v } } } { \mathbf { P r } } \left[ f _ { \mathbf { v } } ( \mathbf { x } ) = 1 \right] - \underset { \mathbf { x } \sim Q _ { \mathbf { v } } } { \mathbf { P r } } \left[ h ( \mathbf { x } ) \neq f _ { \mathbf { v } } ( \mathbf { x } ) \right] \geq 1 9 \epsilon .
$$

Consequently, thresholding the SQ estimate at 10ϵ distinguishes the two cases.

For PQ learning, let $h : \mathbb { R } ^ { d }  \{ \pm 1 , \bot \}$ be the output and instead query $p = \mathbf { P r } _ { \mathbf { x } \sim Q } [ h ( \mathbf { x } ) \neq - 1 ]$ with tolerance ϵ. When $Q = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , the coverage and soundness guarantees give

$$
p \leq \operatorname* { P r } _ { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) } [ h ( \mathbf { x } ) = \perp ] + \operatorname* { P r } _ { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) } [ h ( \mathbf { x } ) = 1 , f _ { \mathbf { v } } ( \mathbf { x } ) = - 1 ] + \rho \leq 3 \epsilon .
$$

When $Q = Q \mathbf { \sigma } _ { \mathbf { v } }$ , soundness gives

$$
p \geq \underset { \mathbf { x } \sim Q _ { \mathbf { v } } } { \mathbf { P r } } \left[ f _ { \mathbf { v } } ( \mathbf { x } ) = 1 \right] - \underset { \mathbf { x } \sim Q _ { \mathbf { v } } } { \mathbf { P r } } \left[ h ( \mathbf { x } ) = - 1 , f _ { \mathbf { v } } ( \mathbf { x } ) = 1 \right] \geq 1 9 \epsilon .
$$

Again, one $\mathrm { S Q }$ query of tolerance ϵ distinguishes the two cases.

In either case, we obtain an SQ distinguisher for NGCA using at most $T + 1$ queries to the unknown distribution, all of tolerance at least min $\{ \tau , \epsilon \}$ . Applying the NGCA lower bound of Fact 5.10, we conclude that either

$$
T + 1 \geq 2 ^ { d ^ { \Omega ( 1 ) } } \quad \mathrm { o r } \quad \operatorname* { m i n } \{ \tau , \epsilon \} \leq d ^ { - \Omega ( r ) } .
$$

Since $r = \Theta ( 1 / \epsilon ^ { 2 } )$ , this gives the claimed SQ lower bound in the stated parameter regime.

## References

[ABAM23] E. Abbe, E. Boix-Adserà, and T. Misiakiewicz. SGD learning on neural networks: Leap complexity and saddle-to-saddle dynamics. In Proceedings of the Thirty Sixth Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pages 2552–2623. PMLR, 2023.

[APS26] J. Alman, S. Patel, and R. A. Servedio. Learning functions of halfspaces. In Proceedings of the 58th Annual ACM Symposium on Theory of Computing. ACM, 2026.

[CKK<sup>+</sup>24] G. Chandrasekaran, A. R. Klivans, V. Kontonis, K. Stavropoulos, and A. Vasilyan. Eficient discrepancy testing for learning with distribution shift. In Advances in Neural Information Processing Systems, volume 37, pages 137263–137308, 2024.

[DIKR25] I. Diakonikolas, G. Iakovidis, D. M. Kane, and L. Ren. Algorithms and SQ lower bounds for robustly learning real-valued multi-index models. In Advances in Neural Information Processing Systems, volume 38, pages 72352–72422, 2025.

[DIKZ25] I. Diakonikolas, G. Iakovidis, D. M. Kane, and N. Zarifis. Robust learning of multi-index models via iterative subspace approximation. In 66th IEEE Annual Symposium on Foundations of Computer Science, FOCS 2025, Sydney, Australia, December 14-17, 2025, pages 2231–2239. IEEE, 2025.

[DKK<sup>+</sup>24] I. Diakonikolas, D. M. Kane, V. Kontonis, C. Tzamos, and N. Zarifis. Agnostically learning multi-index models with queries. In Proceedings of the IEEE Symposium on Foundations of Computer Science (FOCS). IEEE, 2024.

[DKM26] I. Diakonikolas, D. Kane, and M. Ma. Robust learning of multi-index models under logconcave distributions via weak partial learners. In Proceedings of the IEEE Symposium on Foundations of Computer Science (FOCS). IEEE, 2026.

[DKPZ21] I. Diakonikolas, D. M. Kane, T. Pittas, and N. Zarifis. The optimality of polynomial regression for agnostic learning under Gaussian marginals in the SQ model. In Proceedings of The 34<sup>th</sup> Conference on Learning Theory, COLT, 2021.

[DKPZ24] I. Diakonikolas, D. M. Kane, T. Pittas, and N. Zarifis. Statistical query lower bounds for learning truncated Gaussians. In S. Agrawal and A. Roth, editors, Proceedings of the Thirty Seventh Conference on Learning Theory, volume 247 of Proceedings of Machine Learning Research, pages 1336–1363. PMLR, 2024.

[DKRS23] I. Diakonikolas, D. Kane, L. Ren, and Y. Sun. SQ lower bounds for non-Gaussian component analysis with weaker assumptions. Advances in Neural Information Processing Systems, 36:4199–4212, 2023.

[DKS17] I. Diakonikolas, D. M. Kane, and A. Stewart. Statistical query lower bounds for robust estimation of high-dimensional Gaussians and Gaussian mixtures. In 2017 IEEE 58<sup>th</sup> Annual Symposium on Foundations of Computer Science (FOCS), pages 73–84, 2017.

[DKW56] A. Dvoretzky, J. Kiefer, and J. Wolfowitz. Asymptotic minimax character of the sample distribution function and of the classical multinomial estimator. Ann. Mathematical Statistics, 27(3):642–669, 1956.

[DLB25] A. Damian, J. D. Lee, and J. Bruna. The generative leap: Tight sample complexity for eficiently learning Gaussian multi-index models. In Advances in Neural Information Processing Systems, volume 38, pages 32081–32116, 2025.

[DLS22] A. Damian, J. Lee, and M. Soltanolkotabi. Neural networks can learn representations with gradient descent. In Proceedings of the Thirty Fifth Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 5413–5452. PMLR, 2022.

[DPVLB24] A. Damian, L. Pillaud-Vivien, J. D. Lee, and J. Bruna. Computational-statistical gaps in Gaussian single-index models (extended abstract). In Proceedings of the Thirty Seventh Conference on Learning Theory, volume 247 of Proceedings of Machine Learning Research, page 1262. PMLR, 2024.

[DRZ24] I. Diakonikolas, L. Ren, and N. Zarifis. Reliable learning of halfspaces under Gaussian marginals. In Advances in Neural Information Processing Systems, volume 37, pages 45038– 45072, 2024.

[Fel10] V. Feldman. Distribution-specific agnostic boosting. In Proceedings of Innovations in Computer Science, pages 241–250, 2010.

[Fel16] V. Feldman. Statistical query learning. In Encyclopedia of Algorithms, pages 2090–2095. Springer, 2016.

[FGR<sup>+</sup>13] V. Feldman, E. Grigorescu, L. Reyzin, S. Vempala, and Y. Xiao. Statistical algorithms and a lower bound for detecting planted cliques. In Proceedings of STOC’13, pages 655–664, 2013. Full version in Journal of the ACM, 2017.

[FGV17] V. Feldman, C. Guzman, and S. S. Vempala. Statistical query algorithms for mean vector estimation and stochastic convex optimization. In P. N. Klein, editor, Proceedings of the Twenty-Eighth Annual ACM-SIAM Symposium on Discrete Algorithms, SODA 2017, pages 1265–1277. SIAM, 2017.

[FS81] J. H. Friedman and W. Stuetzle. Projection pursuit regression. Journal of the American Statistical Association, 76(376):817–823, 1981.

[GKKM20] S. Goldwasser, A. T. Kalai, Y. Kalai, and O. Montasser. Beyond perturbations: Learning guarantees with arbitrary adversarial test examples. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.

[GSSV24] S. Goel, A. Shetty, K. Stavropoulos, and A. Vasilyan. Tolerant algorithms for learning with arbitrary covariate shift. In Advances in Neural Information Processing Systems, volume 37, 2024.

[Kea93] M. J. Kearns. Eficient noise-tolerant learning from statistical queries. In Proceedings of the Twenty-Fifth Annual ACM Symposium on Theory of Computing, pages 392–401, 1993.

[KK21] A. T. Kalai and V. Kanade. Eficient learning with arbitrary covariate shift. In Algorithmic Learning Theory, pages 850–864. PMLR, 2021.

[KKM12] A. T. Kalai, V. Kanade, and Y. Mansour. Reliable agnostic learning. Journal of Computer and System Sciences, 78(5):1481–1495, 2012.

[KKM13] D. Kane, A. Klivans, and R. Meka. Learning halfspaces under log-concave densities: Polynomial approximations and moment matching. In Conference on Learning Theory, pages 522–545. PMLR, 2013.

[KKMS08] A. T. Kalai, A. R. Klivans, Y. Mansour, and R. A. Servedio. Agnostically learning halfspaces. SIAM Journal on Computing, 37(6):1777–1805, 2008.

[KOS08] A. R. Klivans, R. O’Donnell, and R. A. Servedio. Learning geometric concepts via Gaussian surface area. In 49th Annual IEEE Symposium on Foundations of Computer Science, pages 541–550, Philadelphia, Pennsylvania, 2008. IEEE.

[KPSV26] A. R. Klivans, S. Patel, K. Stavropoulos, and A. Vasilyan. Equivalence of coarse and fine-grained models for learning with distribution shift. In Proceedings of the Thirty Ninth Conference on Learning Theory, volume 336 of Proceedings of Machine Learning Research, pages 4022–4049. PMLR, 2026.

[KSV24a] A. Klivans, K. Stavropoulos, and A. Vasilyan. Learning intersections of halfspaces with distribution shift: Improved algorithms and sq lower bounds. In The Thirty Seventh Annual Conference on Learning Theory, pages 2944–2978. PMLR, 2024.

[KSV24b] A. Klivans, K. Stavropoulos, and A. Vasilyan. Testable learning with distribution shift. In The Thirty Seventh Annual Conference on Learning Theory, volume 247 of Proceedings of Machine Learning Research, pages 2887–2943. PMLR, 2024.

[KSV26] A. R. Klivans, K. Stavropoulos, and A. Vasilyan. Sandwiching polynomials for geometric concepts with low intrinsic dimension. arXiv preprint arXiv:2602.24178, 2026.

[Li91] K.-C. Li. Sliced inverse regression for dimension reduction. Journal of the American Statistical Association, 86(414):316–327, 1991.

[LMZ26] J. H. Lee, A. Mehrotra, and M. Zampetakis. Smoothed analysis of learning from positive samples. In Proceedings of the 58th Annual ACM Symposium on Theory of Computing, STOC ’26, pages 42–53, New York, NY, USA, 2026. Association for Computing Machinery.

[LVM26] H. Latourelle-Vigeant and T. Misiakiewicz. Statistical-computational trade-ofs in learning multi-index models via harmonic analysis. arXiv preprint arXiv:2602.09959, 2026.

[PV26] S. Patel and S. Vempala. Tight bounds for learning polyhedra with a margin. In Proceedings of the IEEE Symposium on Foundations of Computer Science (FOCS). IEEE, 2026.

[RV23] R. Rubinfeld and A. Vasilyan. Testing distributional assumptions of learning algorithms. In Proceedings of the 55th Annual ACM Symposium on Theory of Computing, pages 1643–1656, 2023.

[Val84] L. G. Valiant. A theory of the learnable. In Proc. 16th Annual ACM Symposium on Theory of Computing (STOC), pages 436–445. ACM Press, 1984.

[Vem10] S. Vempala. A random-sampling-based algorithm for learning intersections of halfspaces. J. ACM, 57(6):32:1–32:14, 2010.

[WZDD24] P. Wang, N. Zarifis, I. Diakonikolas, and J. Diakonikolas. Sample and computationally eficient robust learning of Gaussian single-index models. In Advances in Neural Information Processing Systems, volume 37, pages 58376–58422, 2024.

[WZDD25] P. Wang, N. Zarifis, I. Diakonikolas, and J. Diakonikolas. Robustly learning monotone single-index models. In Advances in Neural Information Processing Systems, volume 38, pages 75304–75341, 2025.

[XTLZ02] Y. Xia, H. Tong, W. K. Li, and L.-X. Zhu. An adaptive estimation of dimension reduction space. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 64(3):363–410, 2002.

## Appendix

The appendix is structured as follows. Section A contains additional preliminaries. Section B collects standard facts and supporting technical proofs used in the main text. Section C contains the omitted technical proofs for our partitioning algorithm (Section 3). Section D contains the extensions to Gaussian mixtures and unknown Gaussian reference means. Section E presents an SQ lower bound for agnostic learning with a multiplicative approximation guarantee. Finally, in Section F, we prove our testable agnostic learning results.

## A Omitted Preliminaries

Gaussian space: We denote by $L _ { 2 } ( \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } ) )$ the vector space of all functions $f : \mathbb { R } ^ { d }  \mathbb { R }$ such that $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } ) } [ f ( \mathbf { x } ) ^ { 2 } ] < \infty$ . We use Hermite polynomials normalized with respect to the standard Gaussian measure. Let $\mathrm { H e } _ { m }$ denote the probabilists’ Hermite polynomial of degree $m ,$ , and define

$$
h _ { m } ( t ) : = \frac { \mathrm { H e } _ { m } ( t ) } { \sqrt { m ! } } = \frac { ( - 1 ) ^ { m } } { \sqrt { m ! } } e ^ { t ^ { 2 } / 2 } \frac { d ^ { m } } { d t ^ { m } } e ^ { - t ^ { 2 } / 2 } .
$$

In particular, $h _ { 0 } ( t ) = 1 , h _ { 1 } ( t ) = t ,$ and $h _ { 2 } ( t ) = ( t ^ { 2 } - 1 ) / \sqrt { 2 }$ . For a multi-index $\beta \in \mathbb { Z } _ { > 0 } ^ { d } .$ define $\begin{array} { r } { { \cal H } _ { \beta } ( { \bf x } ) : = \prod _ { r = 1 } ^ { d } h _ { \beta _ { r } } ( { \bf x } _ { r } ) , | \beta | : = \sum _ { r = 1 } ^ { d } \beta _ { r } } \end{array}$ . These polynomials form an orthonormal basis of $L _ { 2 } ( \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } ) )$ , $\operatorname { i . e . , } \mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } ) } [ H _ { \beta } ( \mathbf { x } ) H _ { \gamma } ( \mathbf { x } ) ] = \mathbb { 1 } \{ \beta = \gamma \}$ . We also use the order-m symmetric Hermite tensor $\mathcal { H } _ { m } (  { \mathbf { x } } )$ ， whose entries are

$$
( \mathcal { H } _ { m } ( { \bf x } ) ) _ { j _ { 1 } , \dots , j _ { m } } : = \sqrt { \frac { \beta ! } { m ! } } H _ { \beta } ( { \bf x } ) , \qquad \beta _ { r } : = | \{ s \in [ m ] : j _ { s } = r \} | ,
$$

where $\begin{array} { r } { \beta ! : = \prod _ { r = 1 } ^ { d } \beta _ { r } ! } \end{array}$ . This normalization gives $\langle \mathcal { H } _ { m } ( \mathbf { x } ) , \mathbf { u } ^ { \otimes m } \rangle = h _ { m } ( \mathbf { u } \cdot \mathbf { x } )$ for every unit vector u. Moreover, for symmetric order-m tensors S, T, $\mathbf { E } _ { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } ) } [ \langle \mathbf { S } , \mathcal { H } _ { m } ( \mathbf { x } ) \rangle \langle \mathbf { T } , \mathcal { H } _ { m } ( \mathbf { x } ) \rangle ] = \langle \mathbf { S } , \mathbf { T } \rangle _ { F }$

Definition A.1 (Gaussian surface area). For a Borel set $A \subseteq \mathbb { R } ^ { d }$ and $t > 0$ , let $A _ { t } : = \{ \mathbf { x } \in \mathbb { R } ^ { d }$ : dist $( \mathbf { x } , A ) \leq t \}$ . The Gaussian surface area of A is defined as

$$
\Gamma ( A ) : = \operatorname* { l i m } _ { t \downarrow 0 } \operatorname* { i n f } _ { \mathbf { \Theta } } { \frac { \mathbf { P r } _ { \mathbf { x } \sim { \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } ) } } [ \mathbf { x } \in A _ { t } \setminus A ] } { t } } .
$$

For a Borel measurable Boolean function $f : \mathbb { R } ^ { d }  \{ - 1 , + 1 \}$ , let $A _ { b } : = \{ { \bf x } : f ( { \bf x } ) = b \}$ and define $\Gamma ( f ) : = \Gamma ( A _ { - 1 } ) + \Gamma ( A _ { + 1 } )$ . For a class F of such functions, define $\Gamma ( \mathcal { F } ) : = \operatorname* { s u p } _ { f \in \mathcal { F } } \Gamma ( f )$

Definition A.2 (Gaussian noise operator). For $\rho \in ( 0 , 1 )$ and a bounded measurable function $f : \mathbb { R } ^ { d }  \mathbb { R }$ define

$$
T _ { \rho } f ( \mathbf { x } ) = \underset { \mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } ) } { \mathbf { E } } \left[ f \left( \sqrt { 1 - \rho ^ { 2 } } \mathbf { x } + \rho \mathbf { z } \right) \right] .
$$

Fact A.3 (Properties of Gaussian smoothing). Let $\rho \in ( 0 , 1 )$ . The following properties hold:

1. For every bounded measurable $f :  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ , the function $T _ { \rho } f$ is diferentiable and

$$
\| \nabla T _ { \rho } f ( \mathbf { x } ) \| _ { 2 } \leq \frac { \| f \| _ { \infty } } { \rho } .
$$

2. For every $f : \mathbb { R } ^ { d }  \{ \pm 1 \}$ with finite Gaussian surface area $\Gamma ( f )$

$$
\begin{array} { r } { \underline { { \mathbf { E } } } _ { \boldsymbol { \mathbf { x } } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } ) } \left[ | f ( \mathbf { x } ) - T _ { \rho } f ( \mathbf { x } ) | \right] \le 2 \sqrt { \pi } \rho \Gamma ( f ) . } \end{array}
$$

Basics on SQ Model SQ algorithms are a broad class of algorithms that, instead of having direct access to samples, are allowed to query expectations of bounded functions of the distribution.

Definition A.4 (SQ algorithms). Let D be a distribution on X. A statistical query is a bounded function $q : \mathcal { X }  [ - 1 , 1 ]$ For $\tau > 0$ , the oracle $\mathrm { S T A T } ( \tau )$ responds to a query q with a value v satisfying $| v - \mathbf { E } _ { \mathbf { x } \sim D } [ q ( \mathbf { x } ) ] | \leq \tau$ . We call τ the tolerance of the statistical query. A Statistical Query algorithm is an algorithm whose objective is to learn some information about an unknown distribution D by making adaptive calls to the corresponding oracle.

The SQ model was introduced in [Kea93] as a natural restriction of the PAC model [Val84]. Subsequently, the model has been extensively studied in a range of contexts, see, e.g., [Fel16]. The class of SQ algorithms is broad and captures a range of known supervised learning algorithms. More broadly, several known algorithmic techniques in machine learning are known to be implementable using SQs (see, e.g., [FGR<sup>+</sup>13, FGV17]).

Testable Learning. Testable learning [RV23] is a popular model in which a learner may test the distributional assumptions on the input distribution. For every input distribution, with high probability, whenever the test passes, the learner returns a hypothesis with the prescribed accuracy. Moreover, the learner must accept with high probability whenever the distributional assumptions hold.

Definition A.5 (Testable learning). Let ${ \mathcal { F } } \subseteq \{ \mathbb { R } ^ { d } \to \{ \pm 1 \} \}$ be a concept class, and let D be a family of distributions over $\mathbb { R } ^ { d }$ . We say that an algorithm A testably agnostically learns F to error $\tau + \epsilon$ with probability $1 - \delta$ with respect to D if the following holds.

For every distribution D over $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ , on input $\epsilon , \delta \in ( 0 , 1 / 2 )$ and $m ( \epsilon , \delta )$ i.i.d. samples from D, the algorithm outputs either Reject or $\left( A C C E P T , h \right)$ , where $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$ , satisfying:

1. Soundness. The probability that A accepts and outputs a hypothesis h with $\mathrm { e r r } _ { D } ( h ) > \tau + \epsilon$ is at most δ.

2. Completeness. If $D _ { \mathbf { x } } \in \mathcal { D }$ , then A accepts with probability at least $1 - \delta$

## B Omitted Facts and Technical Proofs

In this section, we collect standard facts used in our proofs.

The following fact shows that Hellinger distance cannot decrease under refinement. We use it in the analysis of our partitioning algorithm.

Fact B.1 (Squared Hellinger distance under refinement). Let $P = ( p _ { i } ) _ { i = 1 } ^ { j }$ and $Q = ( q _ { i } ) _ { i = 1 } ^ { j }$ be probability distributions, and let $I \subseteq [ j ]$ . If the two distributions obtained by grouping coordinates into I and $[ j ] \setminus I$ have squared Hellinger distance at least $\kappa ,$ then ${ \textstyle \sum _ { i = 1 } ^ { j } \sqrt { p _ { i } q _ { i } } \le 1 - \kappa }$

Proof. By Cauchy–Schwarz applied separately to the two groups,

$$
\begin{array} { r l r } {  { \sum _ { i = 1 } ^ { j } { \sqrt { p _ { i } q _ { i } } } \le \sqrt { ( \sum _ { i \in I } p _ { i } ) ( \sum _ { i \in I } q _ { i } ) } + \sqrt { ( \sum _ { i \notin I } p _ { i } ) ( \sum _ { i \notin I } q _ { i } ) } } } \\ & { } & { \le 1 - \kappa , } \end{array}
$$

where the last inequality follows from the assumed squared Hellinger distance between the grouped distributions. □

The following standard fact bounds the diference between the true and empirical cumulative distribution functions uniformly over all thresholds.

Fact B.2 (Dvoretzky–Kiefer–Wolfowitz (DKW) inequality [DKW56]). Let D be a distribution over R, and let $x _ { 1 } , \ldots , x _ { n }$ be independent samples from D. Define the cumulative distribution function $F ( t ) =$ $\mathbf { P r } _ { x \sim D } [ x \leq t ]$ and its empirical counterpart

$$
{ \widehat { F } } _ { n } ( t ) = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \mathbb { 1 } \{ x _ { i } \leq t \} .
$$

For every $\epsilon > 0$

$$
\mathbf { P r } \left[ \operatorname* { s u p } _ { t \in \mathbb { R } } \left| { \widehat { F } } _ { n } ( t ) - F ( t ) \right| > \epsilon \right] \leq 2 \exp ( - 2 n \epsilon ^ { 2 } ) .
$$

In particular, for $\epsilon , \delta \in ( 0 , 1 )$ , if $\begin{array} { r } { n \geq \frac { 1 } { 2 \epsilon ^ { 2 } } } \end{array}$ log $\left( \frac { 2 } { \delta } \right)$ , then, with probability at least $1 - \delta , \Big | \widehat { F } _ { n } ( t ) - F ( t ) \Big | \le \epsilon$ simultaneously for all $t \in \mathbb { R }$

The following fact shows that $L _ { 1 }$ polynomial regression and thresholding eficiently produce a classifier whose error is controlled by the polynomial approximation error.

Fact B.3 $( L _ { 1 }$ polynomial regression [KKMS08]). Let D be a distribution over $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ and let $p$ be a degree-ℓ polynomial. For every $\epsilon , \zeta > 0$ , using $d ^ { O ( \ell ) } \mathrm { p o l y } ( 1 / \epsilon , \mathrm { l o g } ( 1 / \zeta ) )$ time and samples, one can output a degree-ℓ polynomial q and a threshold $t \in [ - 1 , 1 ]$ such that, with probability at least $1 - \zeta$ $\mathbf { P r } _ { D } [ \mathrm { s i g n } ( q ( \mathbf { x } ) - t ) \neq y ] \leq \mathbf { E } _ { D } [ | y - p ( \mathbf { x } ) | ] / 2 + \epsilon$

The following fact shows that projecting both distributions onto the same subspace preserves smoothness with the same parameters. This fact is used in the analysis of our partitioning algorithm.

Fact B.4 (Smoothness under projection). Let $P , Q$ be distributions on $\mathbb { R } ^ { d } .$ , and suppose that $Q$ is $( q , \sigma ) \cdot$ smooth with respect to P. For every subspace $W \subseteq \mathbb { R } ^ { d }$ , the projection $o f Q$ onto $W$ is $( q , \sigma )$ -smooth with respect to the projection of P onto $W$

Proof. Let ${ \mathbf { I I } } = \mathrm { p r o j } _ { W }$ . For any measurable $A \subseteq W$ , apply smoothness to $\Pi ^ { - 1 } ( A ) = \{ \mathbf { x } : \Pi \mathbf { x } \in A \}$ to obtain

$$
\operatorname* { P r } _ { \mathbf { x } \sim Q } [ \mathbf { I I x } \in A ] \leq \frac { \mathbf { P r } _ { \mathbf { x } \sim P } [ \mathbf { I I x } \in A ] ^ { 1 / q } } { \sigma } .
$$

This is the required smoothness inequality for the projections.

We use the following notion of the δ-surface of a concept.

Definition B.5 (δ-surface). For a Boolean function f on $\mathbb { R } ^ { d }$ and $\delta > 0$ , define

$$
S _ { \delta } ( f ) : = \left\{ \mathbf { u } \in \mathbb { R } ^ { d } : \exists \mathbf { v } \in \mathbb { R } ^ { d } , \ \| \mathbf { u } - \mathbf { v } \| _ { 2 } \leq \delta , \ f ( \mathbf { u } ) \neq f ( \mathbf { v } ) \right\} .
$$

We next state facts that bound the Gaussian sandwiching degree in terms of a noise sensitivity quantity.

Fact B.6 (Gaussian sandwiching from boundary smoothness [KSV26, Theorem 3.2]). Let F be a class of $k { - } M I M s$ over $\mathbb { R } ^ { d }$ . Define $\begin{array} { r } { \sigma _ { \mathcal { F } } : = \operatorname* { m a x } \left\{ 1 , \operatorname* { s u p } _ { f \in \mathcal { F } } \operatorname* { s u p } _ { \delta > 0 } ( \mathbf { P r } _ { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) } [ \mathbf { x } \in S _ { \delta } ( f ) ] / \delta ) \right\} . \ I f \sigma _ { \mathcal { F } } < \infty , } \end{array}$ , then, for every $\epsilon \in ( 0 , 1 )$ , every $f \in { \mathcal { F } }$ admits ϵ-sandwiching polynomials of degree at most $\ell ( \epsilon ) = \widetilde O \left( \sigma _ { \mathcal { F } } ^ { 2 } k ^ { 3 } / \epsilon ^ { 4 } \right)$ with respect to the Gaussian distribution.

The following fact states known bounds for the quantity $\sigma ( \mathcal { F } )$ for well studied function classes. This helps us apply our results to concrete function families.

Fact B.7 (Gaussian boundary smoothness and surface area). For a Boolean concept class $\mathcal { F }$ over $\mathbb { R } ^ { d } .$ , let $\sigma ( \mathcal { F } )$ denote the quantity of Fact B.6. Note that, by definition, $\Gamma ( \mathcal { F } ) \leq \sigma ( \mathcal { F } )$ . Moreover, for $K , q \in \mathbb { Z } _ { + }$ with $K \leq d _ { \mathrm { : } }$ , the following bounds hold (see e.g., [KSV26] Lemmas 4.7, 4.5, 4.12 and $4 . 1 4 )$

1. $I f { \mathcal { F } }$ is the class of convex concepts with intrinsic dimension at most $K _ { i }$ then $\sigma ( \mathcal { F } ) = O ( K \log ( 2 K ) )$ .

2. If F is the class of polynomial threshold functions of degree at most r and intrinsic dimension at most K, then $\sigma ( \mathcal { F } ) = O ( r ^ { 3 } K )$

3. $I f { \mathcal { F } }$ is the class of intersections of at most k halfspaces, then $\sigma ( \mathcal { F } ) = O ( \sqrt { \log ( 2 k ) } )$

4. $I f { \mathcal { F } }$ is the class of arbitrary Boolean functions of at most k halfspaces, then $\sigma ( \mathcal { F } ) = O ( k )$

The following fact translates a bound on directional moment discrepancies into a bound on the diference in expectations of a low-degree polynomial under the two distributions.

Fact B.8 $( L _ { 1 }$ polynomials under moment matching). Let $P = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ , and let $Q$ be a distribution over $\mathbb { R } ^ { d }$ such that $P$ and $Q$ match their first k moments with slack $\Delta$ according to Definition 2.2. Let $p : \mathbb { R } ^ { d } $ R be a polynomial of degree at most k satisfying $\| p \| _ { L _ { 1 } ( P ) } \leq M .$ . Then

$$
\biggl | \underset { \mathbf { x } \sim P } { \mathbf { E } } [ p ( \mathbf { x } ) ] - \underset { \mathbf { x } \sim Q } { \mathbf { E } } [ p ( \mathbf { x } ) ] \biggr | \leq ( d k ) ^ { O ( k ) } M \Delta .
$$

Proof. We first relate the Gaussian $L _ { 1 }$ norm of $p$ to its coeficients. Expand $p$ in the normalized multivariate Hermite basis as

$$
p ( \mathbf { x } ) = \sum _ { | \alpha | \leq k } { \widehat { p _ { \alpha } } } H _ { \alpha } ( \mathbf { x } ) .
$$

The number of terms in this expansion is at most ${ \binom { d + k } { k } } \leq ( d + k ) ^ { k }$ , and orthonormality gives

$$
\sum _ { | \alpha | \leq k } \widehat { p } _ { \alpha } ^ { 2 } = \| p \| _ { L _ { 2 } ( P ) } ^ { 2 } .
$$

Every normalized Hermite polynomial $H _ { \alpha }$ of degree at most k has the sum of the absolute values of its monomial coeficients bounded by $k ^ { O ( k ) }$ . Consequently, if

$$
p ( \mathbf { x } ) = \sum _ { | \beta | \leq k } c _ { \beta } \mathbf { x } ^ { \beta }
$$

is the monomial expansion of $p ,$ Cauchy–Schwarz implies

$$
\sum _ { | \beta | \leq k } | c _ { \beta } | \leq ( d k ) ^ { O ( k ) } \| p \| _ { L _ { 2 } ( P ) } .\tag{2}
$$

We next bound the Gaussian $L _ { 2 }$ norm by the Gaussian $L _ { 1 }$ norm. Gaussian hypercontractivity gives $\| p \| _ { L _ { 4 } ( P ) } \le 3 ^ { k / 2 } \| p \| _ { L _ { 2 } ( P ) }$ . Applying the Paley–Zygmund inequality to $p ( \mathbf { x } ) ^ { 2 }$ therefore yields

$$
\underset { { \bf x } \sim P } { \bf P r } \left[ | p ( { \bf x } ) | \geq \frac { \| p \| _ { L _ { 2 } ( P ) } } { \sqrt { 2 } } \right] \geq \frac { 1 } { 4 \cdot 9 ^ { k } } .
$$

It follows that

$$
\| p \| _ { L _ { 2 } ( P ) } \leq 4 \sqrt { 2 } 9 ^ { k } \| p \| _ { L _ { 1 } ( P ) } \leq 4 \sqrt { 2 } 9 ^ { k } M .
$$

Together with (2), this shows that

$$
\sum _ { | \beta | \leq k } | c _ { \beta } | \leq ( d k ) ^ { O ( k ) } M .\tag{3}
$$

It remains to use moment matching. Fix a monomial $\mathbf { x } ^ { \beta }$ of degree $r = | \beta | \in [ k ]$ , and choose indices $i _ { 1 } , \ldots , i _ { r } \in [ d ]$ , with repetitions, such that $\begin{array} { r } { \mathbf { x } ^ { \beta } = \prod _ { j = 1 } ^ { r } \mathbf { x } _ { i _ { j } } } \end{array}$ . By the polarization identity,

$$
\prod _ { j = 1 } ^ { r } \mathbf { x } _ { i _ { j } } = \frac { 1 } { 2 ^ { r } r ! } \sum _ { \varepsilon \in \{ \pm 1 \} ^ { r } } \left( \prod _ { j = 1 } ^ { r } \varepsilon _ { j } \right) \left( \sum _ { j = 1 } ^ { r } \varepsilon _ { j } \mathbf { x } _ { i _ { j } } \right) ^ { r } .
$$

For each $\varepsilon \in \{ \pm 1 \} ^ { r }$ , the vector $\begin{array} { r } { \mathbf { w } _ { \varepsilon } = \sum _ { j = 1 } ^ { r } \varepsilon _ { j } \mathbf { e } _ { i _ { j } } } \end{array}$ satisfies $\| \mathbf { w } _ { \varepsilon } \| _ { 2 } \leq r$ . Applying moment matching to $\mathbf { w } _ { \varepsilon } / \| \mathbf { w } _ { \varepsilon } \| _ { 2 }$ when $\mathbf { w } _ { \varepsilon } \neq \mathbf { 0 }$ gives

$$
\left| \underset { \mathbf { x } \sim P } { \mathbf { E } } [ \mathbf { x } ^ { \beta } ] - \underset { \mathbf { x } \sim Q } { \mathbf { E } } [ \mathbf { x } ^ { \beta } ] \right| \leq \frac { r ^ { r } } { r ! } \Delta \leq e ^ { k } \Delta .
$$

The constant term has the same expectation under P and $Q .$ Hence, by linearity, the triangle inequality, and (3),

$$
\begin{array} { r } { \biggr | _ { \mathbf { x } \sim P } [ p ( \mathbf { x } ) ] - \underset { \mathbf { x } \sim Q } { \mathbf { E } } [ p ( \mathbf { x } ) ] \biggr | \leq e ^ { k } \Delta \underset { 1 \leq | \beta | \leq k } { \sum } | c _ { \beta } | } \\ { \leq ( d k ) ^ { O ( k ) } M \Delta , } \end{array}
$$

as claimed.

## C Omitted Details from Section 3

We now prove Lemma 3.8, which shows that the density ratio of a smooth distribution Q with respect to P is bounded outside a set of small $Q { \mathrm { - m a s s } }$

Proof of Lemma 3.8. By smoothness, $Q ( E ) \leq P ( E ) ^ { 1 / q } / \sigma$ for every measurable event E. In particular, $P ( E ) = 0$ implies $Q ( E ) = 0$ , so the density ratio $f = \mathrm { d } Q / \mathrm { d } P$ is well-defined P-almost everywhere.

If $q = 1$ , smoothness gives $f \leq 1 / \sigma$ P-almost everywhere. Hence $G = \{ f \leq 1 / \sigma \}$ has Q-mass one, and $Q \mid G = Q { \mathrm { ~ i s ~ } } ( 1 , \sigma )$ -smooth, hence also $( 1 , ( 1 - \varepsilon ) \sigma )$ -smooth. This proves the lemma for $q = 1$

Suppose that $q > 1$ . For $t > 0$ , define $E _ { t } = \{ \mathbf { x } : f ( \mathbf { x } ) > t \}$ . Since

$$
Q ( E _ { t } ) = \int _ { E _ { t } } f \mathrm { d } P \geq t P ( E _ { t } ) ,
$$

smoothness gives

$$
Q ( E _ { t } ) \leq \frac { P ( E _ { t } ) ^ { 1 / q } } { \sigma } \leq \frac { ( Q ( E _ { t } ) / t ) ^ { 1 / q } } { \sigma } .
$$

If $Q ( E _ { t } ) ~ > ~ 0$ , raising to the qth power and dividing by $Q ( E _ { t } )$ yields $Q ( E _ { t } ) ^ { q - 1 } \leq 1 / ( \sigma ^ { q } t )$ Taking $t = 1 / ( \sigma ^ { q } \varepsilon ^ { q - 1 } )$ therefore gives $Q ( E _ { t } ) \leq \varepsilon$ . This also holds if $Q ( E _ { t } ) = 0$ . Since $G = E _ { t } ^ { c }$ , we obtain $Q ( G ) \geq 1 - \varepsilon$ . Finally,

$$
\frac { \mathrm { d } ( Q \mid G ) } { \mathrm { d } P } = \frac { f \mathbb { 1 } \left\{ \mathbf { x } \in G \right\} } { Q ( G ) } \leq \frac { 1 } { ( 1 - \varepsilon ) \sigma ^ { q } \varepsilon ^ { q - 1 } } ,
$$

which proves the claimed smoothness of $Q \mid G$

## C.1 Confidence Amplification of Discrepancy Finder

In this subsection, we use standard amplification techniques to increase the success probability of the discrepancy finder in Proposition 3.3.

Corollary C.1 (Confidence amplification of the moment discrepancy finder). Under the hypotheses of Proposition 3.3, for every $\beta \in ( 0 , 1 / 2 )$ , Algorithm 5 draws $d ^ { k } O ( k ^ { O ( k ) } ) V / \Delta ^ { 2 } \log ( 2 / \beta )$ samples and runs in time

$$
d ^ { O ( k ) } \exp \left( O ( k ^ { O ( k ) } ) \frac { H ^ { 2 } } { \Delta ^ { 2 } } \log \frac { H } { \Delta } \right) V ^ { O ( 1 ) } \log ^ { O ( 1 ) } \frac { 2 } { \beta } .
$$

It returns Reject, None, or a unit direction. With probability at least $1 - \beta _ { i }$ , all of the following hold:

1. For every returned direction, there is a degree in [k] for which the corresponding directional moment difers from its Gaussian reference value by at least $\Delta / 4$

2. If the routine returns None, the distribution matches directional moments with the standard Gaussian to slack ∆.

3. If every population Hermite moment tensor has Frobenius norm at most H, the routine does not return Reject.

Algorithm 5 AmplifiedGaussianDiscrepancyFinder   
1: Input: Sample access to Q over $\mathbb { R } ^ { d } ;$ parameters $k , \Delta , V , H$ as in Proposition 3.3, and $\beta \in ( 0 , 1 / 2 )$   
2: Output: Reject, None, or a unit direction.   
3: Set $J = \lceil C \log ( 2 / \beta ) \rceil$ , where C is a suficiently large universal constant.   
4: Run Algorithm 2 independently J times, with parameters $k , \Delta , V , H$   
5: if a strict majority of the outputs are None then   
6: return None.   
7: Let D be the list of returned directions.   
8: if D is empty then   
9: return Reject.   
10: Set $n _ { 0 } = \lceil C V ( 2 k ) ^ { k } / \Delta ^ { 2 } \rceil$ and $B = 2 \lceil C \log ( 2 k J / \beta ) \rceil + 1$   
11: Draw one fresh batch of $B n _ { 0 }$ samples from $Q ,$ , and divide it into B blocks of size $n _ { 0 }$   
12: for each $\mathbf { v } \in \mathcal { D }$ and $\ell \in [ k ]$ do   
13: Let $\widehat { m } _ { \mathbf { v } , \ell }$ be the median of the B block averages of $( \mathbf { v } \cdot \mathbf { x } ) ^ { \ell } .$   
14: if some $\mathbf { v } \in \mathcal { D }$ and $\ell \in [ k ]$ satisfy $\left| \widehat { m } _ { \mathbf { v } , \ell } - \mathbf { E } _ { Z \sim \mathcal { N } ( 0 , 1 ) } [ Z ^ { \ell } ] \right| \geq 5 \Delta / 1 6$ then   
15: return any such v.   
16: else   
17: return Reject.

Proof. For each of the J independent complete finder runs, call the run successful if all three conclusions of Proposition 3.3 hold. Its success probability is at least $2 / 3 .$ and these events are independent. Hoefding’s inequality gives

Pr [at most J/2 complete runs are successful] $\leq e ^ { - J / 1 8 } \leq \beta / 2$

for a suficiently large universal C. We will work on the complementary event, on which strictly more than half of the complete runs are successful.

We next justify the fresh scalar validation stage. Condition on all samples and outputs of the complete finder runs, so the list D is fixed. It contains at most J unit directions. For every $\mathbf { v } \in \mathcal { D }$ and $\ell \in [ k ]$ , the polynomial $( { \bf v } \cdot { \bf x } ) ^ { \ell }$ has degree at most k. The V-assumption gives

$$
\mathbf { \Sigma } _ { Q } ^ { \mathbf { E } } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { 2 \ell } ] \leq V \mathbf { \Sigma } _ { Z \sim \mathcal { N } ( 0 , 1 ) } ^ { \mathbf { E } } [ Z ^ { 2 \ell } ] \leq V ( 2 k ) ^ { k } .
$$

In particular, its variance is at most $V ( 2 k ) ^ { k }$ . If $\overline { { m } } _ { \mathbf { v } , \ell }$ denotes its mean in one fresh block of $n _ { 0 }$ samples, Chebyshev’s inequality gives

$$
\mathbf { P r } \left[ | \overline { { m } } _ { \mathbf { v } , \ell } - \underline { { \mathbf { E } } } _ { Q } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { \ell } ] | > \Delta / 3 2 \right] \leq \frac { 3 2 ^ { 2 } V ( 2 k ) ^ { k } } { n _ { 0 } \Delta ^ { 2 } } \leq \frac { 1 } { 1 0 } ,
$$

after increasing $C .$

For each fixed pair $( \mathbf { v } , { \boldsymbol { \ell } } )$ the blocks are independent. Hoefding’s inequality therefore bounds the probability that at least half of its block means have error greater than $\Delta / 3 2$ by $e ^ { - c B }$ , for a universal

$c > 0$ . If fewer than half have such an error, their median has error at most $\Delta / 3 2$ . There are at most Jk pairs, so the choice of B and a union bound show that, with conditional probability at least $1 - \beta / 2$

$$
\left| \widehat { m } _ { \mathbf { v } , \ell } - \underline { { \mathbf { E } } } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { \ell } ] \right| \leq \Delta / 3 2 \qquad \mathrm { f o r ~ a l l ~ } \mathbf { v } \in \mathcal { D } \mathrm { ~ a n d ~ } \ell \in [ k ] .
$$

The same batch is used for every pair; independence between diferent pairs is not needed. Integrating over the finder outputs gives this validation guarantee unconditionally as well. If the algorithm has already returned or D is empty, we take the validation event to hold automatically.

By a union bound, the majority-success event and the validation event hold together with probability at least $1 - \beta$ . We prove all three conclusions on this joint event.

If a strict majority of the complete runs returns None, at least one of those runs is successful. Its certificate proves that every raw directional moment through degree $k$ difers from its Gaussian reference value by less than $\Delta .$ . Thus the amplified None certificate is valid.

If a direction v passes validation in degree $\ell ,$ its true raw discrepancy is at least

$$
\left| \mathbf { E } [ ( \mathbf { v } \cdot \mathbf { x } ) ^ { \ell } ] - \mathbf { \underset { Z \sim \mathcal { N } ( 0 , 1 ) } { \mathbf { E } } } [ Z ^ { \ell } ] \right| \geq \frac { 5 \Delta } { 1 6 } - \frac { \Delta } { 3 2 } = \frac { 9 \Delta } { 3 2 } \geq \frac { \Delta } { 4 } .
$$

This proves the returned-direction guarantee, even if the direction came from an unsuccessful complete run.

Finally, suppose all the true Hermite moment tensors have Frobenius norm at most H. No successful complete run can reject. If a strict majority returns None, the amplified algorithm returns None as required. Otherwise, since strictly more than half of the complete runs are successful, at least one successful run must return a direction ${ \bf v } .$ For some degree $\ell \in [ k ]$ , this direction has true discrepancy at least $3 \Delta / 8$ Its estimated discrepancy is therefore at least $3 \Delta / 8 - \Delta / 3 2 = 1 1 \Delta / 3 2 > 5 \Delta / 1 6$ . It passes validation, so the amplified routine cannot reject. This proves the three asserted conclusions.

It remains to bound resources. The J full finder runs use

$$
d ^ { k } { \cal O } ( k ^ { { \cal O } ( k ) } ) \frac { V } { \Delta ^ { 2 } } \log \frac { 2 } { \beta }
$$

samples. The shared validation batch has size

$$
B n _ { 0 } = O \left( \frac { V ( 2 k ) ^ { k } } { \Delta ^ { 2 } } \log \frac { 2 k J } { \beta } \right) .
$$

Since $J = O ( \log ( 2 / \beta ) )$ and $\beta < 1 / 2$

$$
\log \frac { 2 k J } { \beta } = O \left( \log ( 2 k ) + \log \frac { 2 } { \beta } \right) .
$$

Factors depending only on $k$ are absorbed into $O ( k ^ { O ( k ) } )$ , so the validation batch is bounded by the same displayed sample bound. Crucially, this is one common batch, rather than a separate batch for every proposal.

The J full runs multiply the base running time by $O ( \log ( 2 / \beta ) )$ . Computing all block means and medians requires at most poly $( d , k , J , B , n _ { 0 } )$ additional arithmetic operations. Using $k \geq 1 , d \geq 1 , V , H \geq 1$ , and $\Delta < 1 / 2$ , these operations are absorbed into

$$
d ^ { O ( k ) } \exp \left( O ( k ^ { O ( k ) } ) \frac { H ^ { 2 } } { \Delta ^ { 2 } } \log \frac { H } { \Delta } \right) V ^ { O ( 1 ) } \log ^ { O ( 1 ) } \frac { 2 } { \beta } .
$$

This proves the sample and time bounds.

## D Omitted Details from Section 4

## D.1 Applications to Well-Studied Distribution Families

Theorem 4.2 gives an agnostic learning algorithm for marginal distributions that are $( q , \sigma )$ -smooth with respect to a known Gaussian reference. In this subsection, we show that this assumption holds for natural distribution families.

First, we show that mixtures of Gaussians with identity covariance are smooth with respect to the standard Gaussian, with a smoothness parameter that depends on the distances of their means from the origin.

Lemma D.1 (Smoothness of shifted Gaussians and their mixtures). Let $G = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ and let $Q =$ $\textstyle \sum _ { i = 1 } ^ { s } \lambda _ { i } { \mathcal { N } } ( \pmb { \mu } _ { i } , \mathbf { I } _ { d } )$ , where $\lambda _ { i } \geq 0$ and $\textstyle \sum _ { i = 1 } ^ { s } \lambda _ { i } = 1$ . Then $Q \ i s \ ( 2 , \sigma )$ -smooth with respect to G, where

$$
\sigma : = \left( \sum _ { i = 1 } ^ { s } \lambda _ { i } e ^ { \| { \pmb { \mu } } _ { i } \| _ { 2 } ^ { 2 } / 2 } \right) ^ { - 1 } .
$$

In particular, $\mathcal { N } ( \pmb { \mu } , \mathbf { I } _ { d } )$ is $( 2 , e ^ { - \| \pmb { \mu } \| _ { 2 } ^ { 2 } / 2 } )$ -smooth with respect to G. $I f \parallel \mu _ { i } \parallel _ { 2 } \leq \mu$ for every i, then the mixture Q is $( 2 , e ^ { - \mu ^ { 2 } / 2 } )$ -smooth with respect to $G$

Proof of Lemma D.1. The density ratio of $\mathcal { N } ( \pmb { \mu } _ { i } , \mathbf { I } _ { d } )$ with respect to G is

$$
r _ { i } ( \mathbf { x } ) = \exp \left( \pmb { \mu } _ { i } \cdot \mathbf { x } - \frac { \| \pmb { \mu } _ { i } \| _ { 2 } ^ { 2 } } { 2 } \right) .
$$

Completing the square in the Gaussian exponent gives $\mathbf { E } _ { \mathbf { x } \sim G } [ r _ { i } ( \mathbf { x } ) ^ { 2 } ] = e ^ { \| \pmb { \mu } _ { i } \| _ { 2 } ^ { 2 } }$ . Consequently, for every measurable event $E \subseteq \mathbb { R } ^ { d }$ , Cauchy–Schwarz gives

$$
\begin{array} { l } { { \displaystyle Q ( E ) = \sum _ { i = 1 } ^ { s } \lambda _ { i } \mathbf { \Delta } _ { \mathbf { x } \sim G } [ r _ { i } ( \mathbf { x } ) \mathbb { 1 } \{ \mathbf { x } \in E \} ] } \ ~ } \\ { { \displaystyle \leq \sum _ { i = 1 } ^ { s } \lambda _ { i } e ^ { \| \pmb { \mu } _ { i } \| _ { 2 } ^ { 2 } / 2 } G ( E ) ^ { 1 / 2 } = \frac { G ( E ) ^ { 1 / 2 } } { \sigma } } . } \end{array}
$$

This proves the smoothness claim. The single-Gaussian case follows by taking $s = 1$ , and the uniform norm bound follows from $\begin{array} { r } { \sum _ { i } \lambda _ { i } e ^ { \| \pmb { \mu } _ { i } \| _ { 2 } ^ { 2 } / 2 } \leq e ^ { \mu ^ { 2 } / 2 } } \end{array}$ □

By Lemma D.1, our agnostic learner applies to Gaussian mixtures with parameters controlled by the norms of their means. Moreover, for classes closed under translations, the choice of origin for the Gaussian reference is essentially without loss of generality: an unknown reference mean can be handled by first estimating the mean of the observed feature distribution.

Corollary D.2 (Gaussian mixtures and unknown Gaussian reference means). Let $\mathcal { F }$ be an $( \ell , K ) \ – w e l l -$ behaved MIM class, and fix $c , \epsilon \in ( 0 , 1 / 2 )$ and $\sigma \in ( 0 , 1 ]$ . Write $\ell _ { c } = \ell ( c ^ { \prime } c )$ for a suficiently small universal constant $c ^ { \prime } > 0$ . The following hold.

1. Suppose that $D _ { \mathbf { x } } \leq Q / \sigma$ , where $\begin{array} { r } { Q = \sum _ { i = 1 } ^ { s } \lambda _ { i } \mathcal { N } ( \pmb { \mu } _ { i } , \mathbf { I } _ { d } ) , \lambda _ { i } \geq 0 , \sum _ { i } \lambda _ { i } = 1 } \end{array}$ , and $\| \pmb { \mu } _ { i } \| _ { 2 } \le M$ for every i. There is an algorithm using at most

$$
N = d ^ { O ( \ell _ { c } ) } \exp \left( ( K \ell _ { c } ) ^ { O ( \ell _ { c } ) } \left( M ^ { 2 } + \log \frac { 2 } { c \sigma \epsilon } \right) ^ { O ( \ell _ { c } ) } \log ( 2 + \Gamma ( \mathcal { F } ) ) \right)
$$

samples from D.

2. Suppose additionally that $\mathcal { F }$ is closed under translations. Let $q \geq 1$ , and suppose that $D _ { \mathbf { x } }$ is $( q , \sigma )$ smooth with respect to $\mathcal { N } ( \pmb { \mu } , \mathbf { I } _ { d } )$ for an unknown $\pmb { \mu } \in \mathbb { R } ^ { d }$ . There is an algorithm using at most

$$
N = d ^ { O ( \ell _ { c } ) } \exp \left( ( q K \ell _ { c } ) ^ { O ( \ell _ { c } ) } \log ^ { O ( \ell _ { c } ) } \frac { 2 } { c \sigma \epsilon } \log ( 2 + \Gamma ( \mathcal { F } ) ) \right)
$$

samples from D, without knowing µ or having sample access to $\mathcal { N } ( \pmb { \mu } , \mathbf { I } _ { d } )$

In both cases, the algorithm runs in time polynomial in N and, with probability at least $2 / 3$ , outputs a hypothesis H satisfying err $_ { D } ( H ) \leq ( 1 + c ) \mathrm { o p t } _ { D } ( \mathcal { F } ) + \epsilon$

Proof of Corollary D.2. For the first assertion, Lemma D.1 gives, for every measurable event $E ,$

$$
D _ { \mathbf { x } } ( E ) \leq \frac { Q ( E ) } { \sigma } \leq \frac { e ^ { M ^ { 2 } / 2 } } { \sigma } \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } ) ( E ) ^ { 1 / 2 } .
$$

Thus $D _ { \mathbf { x } }$ is $( 2 , \sigma e ^ { - M ^ { 2 } / 2 } )$ -smooth with respect to the standard Gaussian. Applying Theorem 4.2 with these parameters proves the first assertion.

For the second assertion, smoothness and the Gaussian tail bound give, for every unit vector v and $t > 0$

$$
\operatorname* { P r } _ { \mathbf x \sim D _ { \mathbf x } } \left[ | \mathbf v \cdot ( \mathbf x - \mu ) | > t \right] \leq \operatorname* { m i n } \left\{ 1 , \frac { 2 } { \sigma } e ^ { - t ^ { 2 } / ( 2 q ) } \right\} .
$$

Integrating this bound yields $\| \mathbf { E _ { x \sim D _ { x } } } [ \mathbf { x } ] - \pmb { \mu } \| _ { 2 } = O ( \sqrt { q \log ( 2 / \sigma ) } )$ and $\mathbf { E _ { x \sim D _ { x } } } [ \| \mathbf { x } - \pmb { \mu } \| _ { 2 } ^ { 2 } ] = O ( d q \log ( 2 / \sigma ) )$ Consequently, using $O ( d q \log ( 2 / \sigma ) )$ independent examples, the empirical mean m satisfies $\begin{array} { r } { \| \mathbf { m } { - } \mathbf { E } _ { \mathbf { x } \sim D _ { \mathbf { x } } } [ \mathbf { x } ] \| _ { 2 } \leq } \end{array}$ 1 with probability at least 99/100. On this event,

$$
\| \pmb { \mu } - \mathbf { m } \| _ { 2 } ^ { 2 } \leq C q \log ( 2 / \sigma )
$$

for a universal constant $C > 0$

Condition on such an estimate, and let $D ^ { \prime }$ be the distribution of $( \mathbf { x } - \mathbf { m } , y )$ for $( \mathbf { x } , y ) \sim D$ . Write $\pmb { \mu } ^ { \prime } = \pmb { \mu } - \mathbf { m }$ . The centered feature distribution is $( q , \sigma )$ -smooth with respect to $\mathcal { N } ( \pmb { \mu } ^ { \prime } , \mathbf { I } _ { d } )$ . Hence, by Lemma D.1,

$$
\begin{array} { l } { \displaystyle { D _ { \mathbf { x } } ^ { \prime } ( E ) \leq \frac { \mathcal { N } ( \pmb { \mu } ^ { \prime } , \mathbf { I } _ { d } ) ( E ) ^ { 1 / q } } { \sigma } } } \\ { \displaystyle { \leq \frac { e ^ { \| \pmb { \mu } ^ { \prime } \| _ { 2 } ^ { 2 } / ( 2 q ) } } { \sigma } \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } ) ( E ) ^ { 1 / ( 2 q ) } . } } \end{array}
$$

Therefore $D _ { \mathbf { x } } ^ { \prime }$ is $( 2 q , \tau )$ -smooth for a known choice $\tau = c _ { 0 } \sigma ^ { C _ { 0 } }$ , where $c _ { 0 } > 0$ and $C _ { 0 } > 1$ are universal constants.

Apply Theorem 4.2 to fresh centered examples, using smoothness parameters $( 2 q , \tau )$ . Since $\log ( 2 / \tau ) =$ $O ( \log ( 2 / \sigma ) )$ , its sample complexity has the stated form. Translation closure gives $\mathrm { o p t } _ { D ^ { \prime } } ( \mathcal { F } ) = \mathrm { o p t } _ { D } ( \mathcal { F } )$ so translating the output back by setting $H ( \mathbf { x } ) = h ( \mathbf { x } - \mathbf { m } )$ preserves the error guarantee. Standard confidence amplification, with a smaller internal accuracy parameter, accounts for the mean-estimation failure probability and gives overall success probability at least $2 / 3$ . The mean-estimation and validation costs are absorbed into the displayed bound. □

## E Multiplicative Approximation SQ Lower Bound

We next prove near-optimality of our agnostic learning algorithm as a simple consequence of [DKPZ21]. First, we define the $L _ { 1 }$ approximation degree that quantifies the complexity of the lower bound. For a Boolean function $f : \mathbb { R } ^ { K }  \{ \pm 1 \}$ and $\epsilon \in ( 0 , 1 )$ , define

$$
\deg _ { 1 , \epsilon } ( f ) : = \operatorname* { m i n } \left\{ r : \exists { \mathrm { ~ a ~ p o l y n o m i a l ~ } } p : \mathbb { R } ^ { K } \to \mathbb { R } , \deg ( p ) \leq r , \underbrace { \mathbf { E } } _ { \mathbf { z } \sim \sqrt { \left( \mathbf { 0 } , \mathbf { I } _ { K } \right) } } \ [ | p ( \mathbf { z } ) - f ( \mathbf { z } ) | ] < \epsilon \right\} .
$$

Fact E.1 ([DKPZ21, Theorem 4]). Fix a constant $a \in ( 0 , 1 / 2 )$ . There is a suficiently small constant $b > 0$ for which the following holds. Let $K \leq d ^ { a }$ and $\epsilon \geq d ^ { - b }$ , fix $f : \mathbb { R } ^ { K }  \{ \pm 1 \}$ , and set $r = \deg _ { 1 , 2 \epsilon } ( f )$ . Suppose that ${ \mathcal { F } } \subseteq \{ \mathbb { R } ^ { d } \to \{ \pm 1 \} \}$ contains every orthogonal embedding of $f ,$ namely $\{ f ( \mathbf { P } \mathbf { x } ) : \mathbf { P } \in \mathbb { R } ^ { K \times d }$ ,PP<sup>⊤</sup> = $\mathbf { I } _ { K } \} \subseteq \mathcal { F }$ . Then any SQ algorithm that agnostically learns $\mathcal { F }$ under $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ to error opt + ϵ either makes a query of tolerance at most $d ^ { - \Omega ( r ) }$ or makes at least $2 ^ { d ^ { \Omega ( 1 ) } }$ queries.

The preceding fact immediately gives the desired tradeof for a multiplicative approximation guarantee.

Theorem E.2 (Multiplicative Approximation SQ Lower Bound). Fix a constant $a \in ( 0 , 1 / 2 )$ . There is a suficiently small constant $b > 0$ for which the following holds. Fix $c \in ( 0 , 1 / 4 )$ and $\epsilon \in \mathsf { \Gamma } ( 0 , c ]$ and assume $2 c \geq d ^ { - b }$ . Let $K \ \leq \ d ^ { a }$ , fix $f : \mathbb { R } ^ { K }  \{ \pm 1 \}$ , and let ${ \mathcal { F } } \subseteq \{ \mathbb { R } ^ { d } \to \{ \pm 1 \} \}$ such that $\{ f ( \mathbf { P x } ) : \mathbf { P } \in \mathbb { R } ^ { K \times d }$ $\mathbf { P } \mathbf { P } ^ { \top } = \mathbf { I } _ { K } \big \} \subseteq { \mathcal { F } }$ . Set $r : = \deg _ { 1 , 4 c } ( f )$ . Then any $S Q$ algorithm that agnostically learns $\mathcal { F }$ under $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ to error $( 1 + c ) { \mathrm { o p t } } + \epsilon$ either makes a query of tolerance at most $d ^ { - \Omega ( r ) }$ or makes at least $2 ^ { d ^ { \Omega ( 1 ) } }$ queries.

Proof. Assume that there exists an algorithm that agnostically learns $\mathcal { F }$ to error $( 1 + c ) { \mathrm { o p t } } + \epsilon .$ Since opt $\leq 1$ and $\epsilon \leq c ,$

$$
( 1 + c ) { \mathrm { o p t } } + \epsilon \leq { \mathrm { o p t } } + c + \epsilon \leq { \mathrm { o p t } } + 2 c .
$$

Thus such an algorithm also agnostically learns $\mathcal { F }$ to additive error opt + 2c. Applying Fact E.1 completes the proof of the claim. □

## F Testable Learning of MIMs from Gaussian Partitioning

In this section, we use our Gaussian partitioning primitive to design a testable agnostic learner for wellbehaved MIM classes under smooth distributions with respect to the Gaussian. Recall the definition of testable learning in Definition A.5. The proofs share several technical similarities with those of our agnostic learning results, hence we present it here in the appendix.

The intuition is as follows. Under our sandwiching assumption, a partition satisfying the guarantees of Theorem 3.1 sufices to obtain a weak learner. This holds without distributional assumptions (subject to a caveat discussed below), since sandwiching polynomials certify the approximation error on regions that match the required moments.

We can check the cell diameters directly from the partition description and test the cell masses and conditional moments empirically using fresh samples. Thus, testing the output partition certifies the properties needed by the weak learner. Boosting then converts this testable weak learner into a testable learner with a relative approximation guarantee.

A caveat is that our original weak learner also uses smoothness to transfer a Gaussian noise sensitivity bound to $Q .$ . This allows us to freeze the constrained coordinates within each cell and reduce to function depending only on the orthogonal directions.

We achieve this by assuming that, on suficiently thin cells, the target admits accurate sandwiching polynomials depending only on the orthogonal directions. This eliminates the need for the perturbation argument (Lemma 4.5). This property seems natural, and it holds for functions of halfspaces (Lemma F.6).

One way to picture this is the following. For each cell R, define the bounding functions on $V _ { R } ^ { \perp }$ by $f _ { R } ^ { + } ( { \bf x } ) : = \operatorname* { s u p } _ { { \bf y } \in \mathrm { p r o j } _ { V _ { R } } ( R ) } f ( { \bf x } + { \bf y } ) , f _ { R } ^ { - } ( { \bf x } ) : = \operatorname* { i n f } _ { { \bf y } \in \mathrm { p r o j } _ { V _ { R } } ( R ) } f ( { \bf x } + { \bf y } )$ . An upper sandwiching polynomial for $f _ { R } ^ { + }$ and a lower sandwiching polynomial for $f _ { R } ^ { - }$ automatically sandwich every restriction $f ( \mathbf { x } + \mathbf { y } )$ with $\mathbf { y } \in \mathrm { p r o j } _ { V _ { R } } ( R )$ . Thus, it sufices that these bounding functions admit accurate Gaussian sandwiching polynomials and that their Gaussian disagreement probability, averaged over cells with weights $P ( R )$ becomes small as the cell diameters decrease.

We first present our new well-behaved definition with this additional assumption:

Definition F.1 (Testable well-behaved MIMs). Let $\ell : ( 0 , 1 ] \to \mathbb { Z } _ { + }$ be non-increasing, let $K \in \mathbb { Z } _ { + }$ , and fix $A > 0$ and $\Lambda \geq 1$ . We say that a class F of Boolean $K { - } M I M s$ is $( \ell , K , A , \Lambda )$ )-testable well-behaved if the following hold.

1. Closure under afine restrictions. For every $f \in { \mathcal { F } }$ , every subspace $W \subseteq \mathbb { R } ^ { d }$ , and every $ { \mathbf { b } } \in W ^ { \perp }$ , the function $\mathbf { x } \mapsto f ( \mathbf { b } + \mathrm { p r o j } _ { W } \mathbf { x } )$ belongs to $\mathcal { F }$

2. Sandwiching on thin cells. Let $f \in { \mathcal { F } }$ and fix $\delta , \tau \in ( 0 , 1 )$ . For every finite polyhedral partition P of $\mathbb { R } ^ { d }$ and every $\mathcal { G } \subseteq \mathcal { P }$ satisfying diam $( R ) \leq \delta$ for all $R \in { \mathcal { G } } ,$ there exist polynomials $p _ { R } ^ { - } , p _ { R } ^ { + } : V _ { R } ^ { \perp } \to \mathbb { R }$ one pair for each $R \in { \mathcal { G } }$ , of degree at most $\ell ( \tau )$ , both depending on a common subspace of dimension at most $K ,$ such that

$$
p _ { R } ^ { - } ( { \bf x } ) \le f ( { \bf x } + { \bf y } ) \le p _ { R } ^ { + } ( { \bf x } ) \qquad f o r e v e r y { \bf x } \in V _ { R } ^ { \perp } a n d { \bf y } \in \mathrm { p r o j } _ { V _ { R } } ( R ) .
$$

Furthermore, these polynomials satisfy

$$
\sum _ { R \in { \mathcal { G } } } Q ( R ) \mathbf { \Pi } _ { \mathbf { x } \sim { \mathcal { N } } ( \mathbf { 0 } , \mathbf { I } _ { V _ { R } ^ { \perp } } ) } [ p _ { R } ^ { + } ( \mathbf { x } ) - p _ { R } ^ { - } ( \mathbf { x } ) ] \leq A \Lambda \delta + \tau
$$

for every probability distribution Q satisfying $Q ( R ) \leq \Lambda \mathbf { P r } _ { \mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } ) } [ \mathbf { x } \in R ]$ for all $R \in { \mathcal { G } }$

Using this definition, we modify our previous agnostic learning algorithm to obtain the following result. For simplicity of notation, let $\mathcal { D } _ { q , \sigma }$ denote the family of distributions over $\mathbb { R } ^ { d }$ that are $( q , \sigma )$ -smooth with respect to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ , where $q \geq 1$ and $\sigma \in ( 0 , 1 ]$

Theorem F.2 (Testable agnostic learning of MIMs). Let $\mathcal { F }$ be an $( \ell , K , A , \Lambda )$ -testable well-behaved class for every $\Lambda \geq 1$ . Fix $q \geq 1 , \sigma \in ( 0 , 1 ]$ , and $c , \epsilon \in ( 0 , 1 / 2 )$ . For a suficiently large universal constant C, let $\ell _ { c } = \ell ( c / C )$

There is an algorithm that, given samples from any distribution D on $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ , testably learns $\mathcal { F }$ under $\mathcal { D } _ { q , \sigma }$ to error $( 1 + c ) \mathrm { o p t } _ { D } ( \mathcal { F } ) + \epsilon$ with probability at least 99%, using

$$
{ \cal N } = d ^ { O ( \ell _ { c } ) } \exp \left( ( q K \ell _ { c } ) ^ { O ( \ell _ { c } ) } \log ^ { O ( \ell _ { c } ) } \left( \frac { 2 } { c \sigma \epsilon } \right) \log ( 2 + A ) \right)
$$

samples from D and running in time poly(N).

Our results are summarized in Table 4.

<table><tr><td>Concept class</td><td>Dimension dependence</td></tr><tr><td> $( \ell , K , A , \Lambda )$  -testable well-behaved MIMs</td><td> $d ^ { O ( \ell _ { c } ) }$ </td></tr><tr><td>Intersections of k halfspaces</td><td> $d ^ { \widetilde { O } ( k ^ { 3 } / c ^ { 4 } ) }$ </td></tr><tr><td>Boolean functions of k halfspaces</td><td> $d ^ { \widetilde { O } ( k ^ { 5 } / c ^ { 4 } ) }$ </td></tr></table>

Table 4: Testable learning under $\mathcal { D } _ { q , \sigma }$ to error $( 1 + c ) \mathrm { o p t } _ { D } ( \mathcal { F } ) +$ ϵ with confidence 99%. The table displays the dependence on d in the sample complexity and running time; the remaining factors are given in Theorem F.2. Here $\ell _ { c } = \ell ( c / C )$ , and the general class is assumed to be testable well-behaved.

The remainder of this section is organized as follows. In Subsection F.1, we present our testable weak learner and prove its guarantees. In Subsection F.2, we prove our main testable learning result and apply it to intersections of halfspaces.

## F.1 Testable Weak Learner

We obtain a testable weak learner by checking the partition on fresh samples before running polynomial regression. The sandwiching condition in Definition F.1 replaces the freezing argument in the proof of Theorem 4.4, so it sufices to check cell masses, diameters, and conditional moments. The full algorithm is given in Algorithm 6.

Theorem F.3 (Testable weak learner). Let F be an $( \ell , K , A , \Lambda )$ -testable well-behaved class for every $\Lambda \geq 1$ Fix $q \geq 1 , \sigma \in ( 0 , 1 ]$ , and $c , \beta \in ( 0 , 1 / 2 )$ . For a suficiently large universal constant C, let $\ell _ { c } = \ell ( c / C )$

There is an algorithm that, given samples from any distribution D on $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ , testably learns $\mathcal { F }$ under $\mathcal { D } _ { q , \sigma }$ with probability at least $1 - \beta _ { ; }$ , to error $1 / 2 - c / 1 2$ whenever $\mathrm { o p t } _ { D } ( \mathcal { F } ) \leq 1 / ( 2 ( 1 + c ) )$ . The algorithm uses

$$
n = d ^ { O ( \ell _ { c } ) } \exp \left( ( q K \ell _ { c } ) ^ { O ( \ell _ { c } ) } \log ^ { O ( \ell _ { c } ) } ( 2 / ( c \sigma ) ) \log ( 2 + A ) \right) \log ^ { O ( 1 ) } ( 2 / \beta )
$$

samples from D and runs in time poly(n). Its completeness guarantee holds for every D with $D _ { \mathbf { x } } \in \mathcal { D } _ { q , \sigma }$ ， regardless of $\mathrm { o p t } _ { D } ( \mathcal { F } )$

```latex
Algorithm 6 TestableWeakLearner
1: Input: Sample access to D, parameters $q , \sigma , c ,$ and class parameters $\ell , K , A .$
2: Output: Reject or a hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$
3: Set $\eta , \tau , \ell _ { c } , \Delta , \Lambda , \delta , n _ { 0 }$ as in the proof of Theorem $\mathrm { F . 3 . }$
4: Run Algorithm 1 with parameters $( q , \sigma , \eta / 1 0 , \delta , \ell _ { c } , \Delta / 1 0 )$ and its prescribed sample and time budgets.
Denote its output by $( \mathcal { P } , \mathcal { G } _ { 0 } )$ , and let B, H be its cell bound and Hermite norm threshold.
5: if Algorithm 1 does not finish within its budgets or $| \mathcal { P } | > B$ then return Reject.
6: Draw a fresh labeled sample S of size $n _ { 0 }$ from $D ,$ and let $\widehat { Q }$ be its empirical x-marginal.
7: Set $\mathcal { G } = \emptyset$ and $P = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$
8: for $R \in \mathcal { G } _ { 0 }$ do
9: if diam $( R ) \leq \delta , \widehat { Q } ( R ) \geq 1 / ( 2 B )$ , and ${ \widehat Q } ( R ) \leq \Lambda P ( R )$ then
10: If $V _ { R } ^ { \perp } \neq \{ \mathbf { 0 } \}$ , run Algorithm 2 using the $V _ { R } ^ { \perp }$ projections of $S \cap R$ as its supplied sample, with
degree $\ell _ { c } ,$ slack $\Delta ,$ and Hermite cap $H .$
11: If $V _ { R } ^ { \perp } = \{ \mathbf { 0 } \}$ , take the output to be None.
12: if the output is None then add R to ${ \mathcal { G } } .$
13: if $\begin{array} { r } { \sum _ { R \in \mathcal { G } } \widehat { Q } ( R ) < 1 - \eta } \end{array}$ then return Reject.
14: for $R \in { \mathcal { G } }$ do
15: Using the same sample $S \cap R ,$ find a polynomial $q _ { R }$ on $V _ { R } ^ { \perp }$ of degree at most $\ell _ { c }$ whose empirical $L _ { 1 }$
loss is within τ of the minimum.
16: Choose $t _ { R } \in [ - 1 , 1 ]$ minimizing the empirical classification error of sign $( q _ { R } ( \mathrm { p r o j } _ { V _ { R } ^ { \perp } } { \bf x } ) - t _ { R } )$ on
$S \cap R$
17: Set $h _ { R } ( { \bf x } ) = \mathrm { s i g n } ( q _ { R } ( \mathrm { p r o j } _ { V _ { R } ^ { \perp } } { \bf x } ) - t _ { R } )$
18: Define $h ( \mathbf { x } ) = h _ { R } ( \mathbf { x } )$ when $\mathbf { x } \in R \in { \mathcal { G } } _ { : }$ , and set $h ( \mathbf { x } ) = 1$ on the remaining cells.
19: return h.
```

Proof of Theorem F.3. We first show that Algorithm 6 satisfies the guarantees of the theorem with constant success probability, and then amplify this probability to $1 - \beta$ . Let $Q = D _ { \mathbf { x } }$ and $P = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ . Let $\eta = \tau = c / C$ and $\ell _ { c } = \ell ( \tau )$ . Set $\Delta = ( 2 K \ell _ { c } ) ^ { - C \ell _ { c } } , \Lambda = 2 / ( \sigma ^ { q } ( \eta / 1 0 0 ) ^ { q - 1 } )$ , and $\delta = \tau / ( ( 1 + A ) \Lambda )$ . Run Algorithm 1 with parameters $( q , \sigma , \eta / 1 0 , \delta , \ell _ { c } , \Delta / 1 0 )$ , obtaining a partition $\mathcal { P }$ and candidate certified cells $\mathcal { G } _ { 0 }$ . Let B be the bound supplied by Theorem 3.1. We reject if the partitioning procedure exceeds its stated sampling or running-time budget, or returns more than B cells.

Set $n _ { 0 } = d ^ { O ( \ell _ { c } ) } \mathrm { p o l y } ( B , \Lambda , 1 / \sigma , 1 / \tau )$ , with suficiently large implicit constants. Draw a fresh labeled sample S of size $n _ { 0 }$ , independently of the partition, and denote its empirical distribution by $\widehat { D }$ , with x-marginal $\widehat { Q }$ . Use the available Gaussian cell probabilities $P ( R )$ . For each $R \in \mathcal { G } _ { 0 }$ , retain $R$ only if:

1. diam $( R ) \leq \delta , \widehat { Q } ( R ) \geq 1 / ( 2 B )$ , and ${ \widehat Q } ( R ) \leq \Lambda P ( R )$ (Line 9).

2. The conditional empirical distribution, projected onto $V _ { R } ^ { \perp }$ , passes the moment discrepancy test with slack $\Delta$ (Line 12).

The diameter is checked from the recorded orthogonal cuts. For the moment test, when $V _ { R } ^ { \perp } \neq \{ \mathbf { 0 } \}$ , run Algorithm 2 using the $V _ { R } ^ { \perp }$ projections of $S \cap R$ as its supplied sample, with degree $\ell _ { c } ,$ slack $\Delta$ , and Hermite cap H. Retain R only when the output is None. If $V _ { R } ^ { \perp } = \{ \mathbf { 0 } \}$ , the moment test passes automatically.

Let G be the retained collection. Reject if $\widehat { Q } ( \bigcup _ { R \in \mathcal { G } } R ) < 1 - \eta$ . Otherwise, run $\mathrm { d e g r e e } { - \ell _ { c } }$ empirical $L _ { 1 }$ regression on each $S \cap R ,$ followed by threshold selection as in Algorithm $^ { 3 , }$ with accuracy τ. Label the remaining cells arbitrarily and return the resulting hypothesis $h$

We first show Completeness, i.e., if $Q$ is smooth the algorithm returns a hypothesis with constant probability.

Claim F.4 (Completeness). $I f Q \in { \mathcal { D } } _ { q , \sigma }$ , then Algorithm 6 accepts with probability at least $1 / 2$

Proof of Claim $F . 4 \AA$ By Theorem 3.1, with probability at least $2 / 3$ , the cells in $\mathcal { G } _ { 0 }$ have total Q-mass at least $1 - \eta / 1 0$ , and every $R \in \mathcal { G } _ { 0 }$ has Q-mass at least $1 / B$ , diameter at most $\delta ,$ and conditional moments on $V _ { R } ^ { \perp }$ matching the Gaussian moments with slack $\Delta / 1 0$

Applying smoothness to unions of cells shows that the discrete distribution $( Q ( R ) ) _ { R \in \mathcal { P } }$ is $( q , \sigma )$ -smooth with respect to $( P ( R ) ) _ { R \in \mathcal { P } }$ . Now applying Lemma 3.8 to these distributions shows that cells satisfying $Q ( R ) > ( \Lambda / 2 ) P ( R )$ have total Q-mass at most $\eta / 1 0 0$

Conditional on the partition, Hoefding’s inequality and a union bound over at most B cells ensure, with probability at least $7 / 8 ,$ , that $| \widehat { Q } ( R ) - Q ( R ) | \leq \eta / ( 1 0 B )$ for every cell R. The candidate cells satisfying $Q ( R ) \leq ( \Lambda / 2 ) P ( R )$ have total Q-mass at least $1 - 1 1 \eta / 1 0 0$ , and each satisfies $Q ( R ) \geq 1 / B$ . For every such cell, $\widehat { Q } ( R ) \geq Q ( R ) - \eta / ( 1 0 B ) \geq 1 / ( 2 B )$ and ${ \widehat { Q } } ( R ) \leq Q ( R ) + \eta / ( 1 0 B ) \leq 2 Q ( R ) \leq \Lambda P ( R )$ . Thus these cells pass the mass checks on Line 9. Since there are at most B such cells, their total mass estimation error is at most $\eta / 1 0$ , so their total empirical mass is at least $1 - 2 1 \eta / 1 0 0$

By Lemmas 3.11 and $3 . 1 3 ,$ all but $\eta / 1 0 0$ of the Q-mass lies in cells R satisfying $\| \mathbf { E } _ { \mathbf { x } \sim Q | R } [ \mathcal { H } _ { i } ( \mathrm { p r o j } _ { V _ { B } ^ { \perp } } \mathbf { x } ) ] \| _ { F } \leq$ H for every $i \in [ \ell _ { c } ]$ , and $\mathbf { E } _ { Q | R } [ p ^ { 2 } ] \leq V \| p \| _ { L _ { 2 } ( P ) } ^ { 2 }$ for every polynomial p of degree at most $\ell _ { c }$ depending only on $V _ { R } ^ { \perp }$ , where $V = C q ^ { 2 } ( 8 q - 1 ) ^ { \ell _ { c } } / ( \sigma \eta )$ for a suficiently large universal constant $C$

For each candidate cell satisfying the bounds above, conditional on its sample count, the projected observations in $S \cap R$ are independent samples from the projected distribution $Q \mid R$ . By the tensorestimation bound in the proof of Claim 3.18, taking $n _ { 0 }$ suficiently large makes the probability that a cell contains at least $n _ { 0 } / ( 2 B )$ observations but fails the finder’s estimation event at most $1 / ( 8 B )$ . A union bound gives simultaneous success for all cells with this many observations with probability at least $7 / 8$ On the cell-mass event above, all relevant cells have this many observations. The proof of Proposition 3.3 then shows that the finder does not reject. It cannot return a direction either, since any returned direction would have population moment discrepancy at least $3 \Delta / 8 ,$ , contradicting the conditional moment-matching bound $\Delta / 1 0$ . Hence every such cell returns None. Together with the cell-mass event above, both events hold with conditional probability at least $3 / 4$ . On this event, cells of total Q-mass at least $1 - 1 2 \eta / 1 0 0$ pass both tests, and their empirical mass is at least $1 - 2 2 \eta / 1 0 0 \geq 1 - \eta$ . Thus the algorithm accepts. This completes the proof of Claim F.4. □

Claim F.5 (Soundness). For every distribution $D ,$ with probability at least $1 1 / 1 2$ over the fresh sample $S _ { i }$ whenever the algorithm accepts, its output satisfies

$$
\mathrm { e r r } _ { D } ( h ) \leq \mathrm { o p t } _ { D } ( \mathcal { F } ) + 7 \tau .
$$

Proof of Claim F.5. Fix $f \in { \mathcal { F } }$ with err $_ { D } ( f ) \leq \mathrm { o p t } _ { D } ( \mathcal { F } ) + \tau$ . By the cell-mass test, acceptance gives ${ \widehat Q } ( R ) \leq \Lambda P ( R )$ for every $R \in { \mathcal { G } }$ . Apply Definition F.1 to this partition and $f ,$ and write $s _ { R } : = p _ { R } ^ { + } - p _ { R } ^ { - } \ge 0$ The resulting polynomials satisfy

$$
\sum _ { R \in { \mathcal G } } { \widehat Q } ( R ) \ { \underline { { { \mathbf { F } } } } } \left[ { s _ { R } } \right] \leq A \Lambda \delta + \tau \leq 2 \tau \ ,
$$

where $G _ { R } = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { V _ { R } ^ { \bot } } )$ . Let $\widehat { Q } _ { R }$ denote the projection of ${ \widehat { Q } } \mid R$ onto $V _ { R } ^ { \perp }$ . Since the finder uses the exact empirical moments, its None certificate ensures that $\widehat { Q } _ { R }$ and $G _ { R }$ match directional moments through degree $\ell _ { c }$ with slack $\Delta$ . Since $s _ { R }$ depends on at most K directions, empirical moment matching and Fact B.8 give

$$
\sum _ { R \in \mathcal { G } } \widehat { Q } ( R ) \ \underline { { \mathbf { F } } } _ { R } [ s _ { R } ] \leq \left( 1 + ( K \ell _ { c } ) ^ { O ( \ell _ { c } ) } \Delta \right) \sum _ { R \in \mathcal { G } } \widehat { Q } ( R ) \ \underline { { \mathbf { F } } } _ { R } [ s _ { R } ] \leq 4 \tau .
$$

Conditional on $S ,$ apply Fact B.3 to the projected labeled empirical distribution in each $R \in { \mathcal { G } }$ . A union bound gives probability at least $2 3 / 2 4$ that all regression calls satisfy $\mathrm { e r r } _ { \widehat { D } | R } ( h _ { R } ) \leq { \bf E } _ { \widehat { D } | R } [ | y - p _ { R } ^ { + } ( { \bf x } _ { V _ { R } ^ { \perp } } ) | ] / 2 + \tau$ The pointwise sandwich and the preceding gap bound therefore give err $\ B ( h ) \leq \operatorname { e r r } _ { \widehat { D } } ( f ) + \eta + 3 \tau$ , where η accounts for the empirical mass outside $\mathcal { G }$

Conditional on the independently constructed partition, piecewise degree-ℓ<sub>c</sub> polynomial threshold hypotheses have VC dimension at most $( d + \ell _ { c } ) ^ { O ( \ell _ { c } ) } \bar { B }$ . Taking S suficiently large ensures, with probability at least $1 1 / 1 2$ , that er $\mathrm { r } _ { D } ( h ) \le \mathrm { e r r } _ { \hat { D } } ( h ) + \tau$ uniformly over these hypotheses, and e $\operatorname { r r } _ { \widehat { D } } ( f ) \leq \operatorname { e r r } _ { D } ( f ) + \tau$ for the fixed $f .$ . On this event, whenever the algorithm accepts,

$$
\operatorname { e r r } _ { D } ( h ) \leq \operatorname { e r r } _ { D } ( f ) + \eta + 5 \tau \leq \mathrm { o p t } _ { D } ( \mathcal { F } ) + \eta + 6 \tau = \mathrm { o p t } _ { D } ( \mathcal { F } ) + 7 \tau .
$$

This completes the proof of Claim F.5.

Under the assumption opt $_ { D } ( \mathcal { F } ) \leq 1 / ( 2 ( 1 + c ) )$ , Claim F.5 gives

$$
\mathrm { e r r } _ { D } ( h ) \leq \frac { 1 } { 2 ( 1 + c ) } + \frac { 7 c } { C } \leq \frac { 1 } { 2 } - \frac { c } { 1 2 }
$$

for a suficiently large universal constant $C$

The partition construction, empirical moment tests, regression, and generalization require $d ^ { O ( \ell _ { c } ) } \mathrm { p o l y } ( B , \Lambda , 1 / \sigma , 1 / \tau )$ samples and time polynomial in this quantity.

To obtain failure probability at most $\beta ,$ use $O ( \log ( 1 / \beta ) )$ independent runs and return the first accepted hypothesis, rejecting if every run rejects. Choose the fresh sample sizes so that the sum of the soundness failure probabilities across these runs is at most $\beta / 2$ . By Claim F.4, if $Q \in \mathcal { D } _ { q , \sigma }$ , the probability that all runs reject is at most $\beta / 2$ . Thus both guarantees hold with probability at least $1 - \beta$ , with overhead polynomial in $\log ( 1 / \beta )$ □

## F.2 Testable Learner for MIMs under Smooth Distributions

In this section, we prove our main testable learning result, Theorem F.2, and apply it to intersections and arbitrary Boolean functions of k halfspaces.

The proof applies the agnostic boosting fact to the testable weak learner from Theorem F.3. The only modification to the boosting algorithm is that it rejects whenever a weak-learning call rejects.

Proof of Theorem F.2. Let $Q = D _ { \mathbf { x } }$ and $P = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ . We use Fact 4.3 with $\alpha = c / ( 2 ( 1 + c ) ) , \gamma = c / 1 2$ and confidence $1 - 1 / 1 0 0 0$ . These choices give $1 / 2 - \alpha = 1 / ( 2 ( 1 + c ) )$ and $1 / ( 1 - 2 \alpha ) = 1 + c$ . Thus Theorem F.3 supplies exactly the weak-learning guarantee required by the boosting fact whenever a call accepts.

Algorithm 7 TestableBoosting   
1: Input: Sample access to $D ,$ parameters $q , \sigma , c , \epsilon ,$ and class parameters $\ell , K , A .$   
2: Output: Reject or a hypothesis $h : \mathbb { R } ^ { d }  \{ \pm 1 \}$   
3: Set $\alpha = c / ( 2 ( 1 + c ) )$ and $\gamma = c / 1 2 .$   
4: Let $T = \mathrm { p o l y } ( 1 / c , 1 / \epsilon )$ and $\Gamma = O ( 1 / \epsilon )$ be the bounds from the agnostic boosting fact on the number   
of weak-learning calls and on $d D _ { t } / d D$ , respectively.   
5: Set $\sigma _ { * } = \sigma / \Gamma$ and $\beta = 1 / ( 1 0 0 0 T )$   
6: Run the boosting algorithm from Fact 4.3 with parameters $( \alpha , \gamma , \epsilon )$ and confidence $1 - 1 / 1 0 0 0 .$ , answering   
its weak-learning queries as follows.   
7: for each weak-learning query on a distribution $D _ { t }$ do   
8: Apply Theorem F.3 to fresh samples from $D _ { t }$ with parameters $( \boldsymbol { q } , \sigma _ { * } , c )$ and failure probability $\beta .$   
9: if the weak learner returns Reject then   
10: return Reject.   
11: Supply the returned hypothesis $h _ { t }$ to the boosting algorithm.   
12: return the hypothesis produced by the boosting algorithm.

Apply Theorem F.3 with failure probability $\beta = 1 / ( 1 0 0 0 T )$ for each weak-learning query. By the union bound, all weak-learning calls are successful with probability at least $9 9 \%$

For soundness, assume that all weak-learning calls and the boosting procedure satisfy their guarantees. If any weak-learning call rejects, the algorithm rejects. Thus, if the algorithm outputs a hypothesis, every call has returned a hypothesis satisfying the required weak-learning guarantee. By Fact 4.3,

$$
\operatorname { e r r } _ { D } ( h ) \leq { \frac { \mathrm { o p t } _ { D } ( { \mathcal { F } } ) } { 1 - 2 \alpha } } + \epsilon = ( 1 + c ) \mathrm { o p t } _ { D } ( { \mathcal { F } } ) + \epsilon .
$$

For completeness, suppose that $Q ~ \in ~ { \cal D } _ { q , \sigma }$ . The boosting fact guarantees that every query distribution satisfies $d D _ { t } / d D \leq \Gamma$ for $\Gamma = { \cal O } ( 1 / \epsilon )$ . Consequently, its x-marginal $Q _ { t }$ satisfies $Q _ { t } ( E ) \ \leq$ $\Gamma Q ( E )$ , for every measurable $E \subseteq \mathbb { R } ^ { d }$ . Therefore, $Q _ { t } \in \mathcal { D } _ { q , \sigma / \Gamma }$ . Hence all weak-learning calls accept with probability 99%, so the boosting algorithm finishes and returns a hypothesis.

Finally, the weak learner is run with $\sigma _ { * } = \sigma / \Gamma$ and degree $\ell _ { c } = \ell ( c / C )$ . Since $\Gamma = O ( 1 / \epsilon )$ , substituting $\sigma _ { * }$ into the bound of Theorem F.3 gives

$$
d ^ { O ( \ell _ { c } ) } \exp \left( ( q K \ell _ { c } ) ^ { O ( \ell _ { c } ) } \log ^ { O ( \ell _ { c } ) } \left( \frac { 2 } { c \sigma \epsilon } \right) \log ( 2 + A ) \right)
$$

samples per call. The boosting fact uses $\mathrm { { . 9 o l y } } ( 1 / c , 1 / \epsilon )$ calls and polynomial sampling and computational overhead. This proves the stated sample complexity and running time, which completes the proof of Theorem F.2. □

We next show that functions of halfspaces satisfy Condition 2 of Definition F.1, yielding a testable learning algorithm for this class.

For each thin cell R, we construct two functions of halfspaces $h _ { R } ^ { \pm }$ on $V _ { R } ^ { \perp }$ such that $h _ { R } ^ { - } ( { \bf x } _ { V _ { R } ^ { \perp } } ) \leq h ( { \bf x } ) \leq$ $h _ { R } ^ { + } ( \mathbf { x } _ { V _ { R } ^ { \perp } } )$ for every $\mathbf { x } \in R ,$ , where h is the target function. We obtain these bounding functions by adjusting each halfspace’s threshold according to its variation along $V _ { R }$ within R. The two thresholds difer by at most the cell diameter. Gaussian anti-concentration then bounds the Gaussian disagreement between $h _ { R } ^ { - }$ and $h _ { R } ^ { + }$ , averaged over the partition. Finally, applying Gaussian sandwiching to these two functions of halfspaces gives the required polynomials.

Lemma F.6 (Sandwiching functions of halfspaces on thin cells). For every $k \in \mathbb { N } _ { + }$ and $\Lambda \geq 1$ , the class of arbitrary Boolean functions of at most k afine halfspaces satisfies Condition 2 of Definition F.1 with parameters $( \ell , k , 2 k , \Lambda )$ , where $\ell ( \tau ) = \widetilde O ( k ^ { 5 } / \tau ^ { 4 } )$ . Moreover, intersections of at most k halfspaces satisfy the same condition with $\ell ( \tau ) = \widetilde O ( k ^ { 3 } / \tau ^ { 4 } )$

Proof of Lemma $F . 6 .$ Fix $\delta , \tau \in ( 0 , 1 )$ and let $P = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ . Let $\mathcal { P }$ be a finite polyhedral partition and let $\mathcal { G } \subseteq \mathcal { P }$ consist of cells of diameter at most δ in their constraint subspaces. For each cell, write $A _ { R } : = \mathrm { p r o j } _ { V _ { R } } ( R )$ and $G _ { R } : = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { V _ { R } ^ { \perp } } )$ , so that $R = A _ { R } + V _ { R } ^ { \perp }$ and diam $( A _ { R } ) \leq \delta$ for $R \in { \mathcal { G } }$ . We may omit empty cells, assigning zero polynomials to them.

Let $f$ be a Boolean function of at most k halfspaces. If f is constant, take $p _ { R } ^ { - } = p _ { R } ^ { + } = f$ . Otherwise, absorbing constant halfspaces into the Boolean link and repeating the remaining halfspaces if necessary, write $f ( \mathbf { u } ) = F ( h _ { 1 } ( \mathbf { u } ) , \dots , h _ { k } ( \mathbf { u } ) )$ , where $F : \{ \pm 1 \} ^ { k }  \{ \pm 1 \}$ and $h _ { j } ( \mathbf { u } ) = 2 \mathbb { 1 } \{ \mathbf { w } _ { j } \cdot \mathbf { u } \leq t _ { j } \} - 1$ with $\| \mathbf { w } _ { j } \| _ { 2 } = 1$ . For each nonempty $R \in { \mathcal { G } }$ , define $\begin{array} { r } { \alpha _ { R , j } : = \operatorname* { i n f } _ { \mathbf { y } \in A _ { R } } \mathbf { w } _ { j } \cdot \mathbf { y } } \end{array}$ and $\beta _ { R , j } : = \operatorname* { s u p } _ { \mathbf { y } \in A _ { R } } \mathbf { w } _ { j } \cdot \mathbf { y }$ . Then $0 \leq \beta _ { R , j } - \alpha _ { R , j } \leq \delta$ . Define shifted halfspaces on $V _ { R } ^ { \perp }$ V<sup>⊥</sup> by

$$
\begin{array} { r } { h _ { R , j } ^ { - } ( \mathbf { x } ) : = 2 \mathbb { 1 } \{ \mathbf { w } _ { j } \cdot \mathbf { x } \leq t _ { j } - \beta _ { R , j } \} - 1 , } \\ { h _ { R , j } ^ { + } ( \mathbf { x } ) : = 2 \mathbb { 1 } \{ \mathbf { w } _ { j } \cdot \mathbf { x } \leq t _ { j } - \alpha _ { R , j } \} - 1 . } \end{array}
$$

For every $\mathbf { x } \in V _ { R } ^ { \perp } \ \mathrm { a n d } \ \mathbf { y } \in A _ { R } , h _ { R , j } ^ { - } ( \mathbf { x } ) \leq h _ { j } ( \mathbf { x } + \mathbf { y } ) \leq h _ { R , j } ^ { + } ( \mathbf { x } )$

Let $E _ { R }$ be the set of $\mathbf { x } \in V _ { R } ^ { \bot }$ for which some pair $h _ { R , j } ^ { - } ( \mathbf { x } ) , h _ { R , j } ^ { + } ( \mathbf { x } )$ disagrees, and define

$$
h _ { R } ^ { \pm } ( \mathbf x ) : = \left\{ \begin{array} { l l } { F ( h _ { R , 1 } ^ { - } ( \mathbf x ) , \ldots , h _ { R , k } ^ { - } ( \mathbf x ) ) , } & { \mathbf x \notin E _ { R } , } \\ { \pm 1 , } & { \mathbf x \in E _ { R } . } \end{array} \right.
$$

Outside $E _ { R } ,$ every underlying halfspace has the same value for all $\mathbf { y } \in A _ { R }$ . On $E _ { R }$ , the two bounds are −1 and 1. Hence $h _ { R } ^ { - } ( { \mathbf { x } } ) \leq f ( { \mathbf { x } } + { \mathbf { y } } ) \leq h _ { R } ^ { + } ( { \mathbf { x } } )$ for every $\mathbf { x } \in V _ { R } ^ { \bot }$ and $\mathbf { y } \in A _ { R }$

Both functions are Boolean functions of the same 2k shifted halfspaces and depend only on $W _ { R } : =$ $\operatorname { s p a n } \{ { \mathrm { p r o j } } _ { V _ { R } ^ { \perp } } \mathbf { w } _ { j } : j \in [ k ] \}$ , which has dimension at most k. By Facts B.6 and B.7, applied on $W _ { R }$ with error $\tau / 2$ , we obtain polynomials of degree at most $\ell ( \tau ) = \widetilde O ( k ^ { 5 } / \tau ^ { 4 } )$ . Extending them by projection give $p _ { R } ^ { \pm }$ , both depending on $W _ { R }$ , with $p _ { R } ^ { - } \leq h _ { R } ^ { - } , h _ { R } ^ { + } \leq p _ { R } ^ { + }$ , and

$$
\begin{array} { r } { \underline { { \mathbf { E } } } _ { R } [ h _ { R } ^ { - } - p _ { R } ^ { - } ] \leq \tau / 2 , \qquad \underline { { \mathbf { E } } } _ { R } [ p _ { R } ^ { + } - h _ { R } ^ { + } ] \leq \tau / 2 . } \end{array}
$$

If $W _ { R } = \{ \mathbf { 0 } \}$ , take the corresponding constant polynomials instead. Thus the required pointwise sandwich holds. Writing $b _ { R } : = G _ { R } ( E _ { R } )$ , we have ${ \bf E } _ { G _ { R } } [ p _ { R } ^ { + } - p _ { R } ^ { - } ] \leq 2 b _ { R } + \tau$

We now bound the weighted disagreement. If ${ \mathbf u } = { \mathbf x } + { \mathbf y } \in R$ with ${ \bf x } \in E _ { R }$ , then for some $j \in [ k ]$ $t _ { j } - \beta _ { R , j } < \mathbf { w } _ { j } \cdot \mathbf { x } \leq t _ { j } - \alpha _ { R , j }$ . Together with $\alpha _ { R , j } \leq \mathbf { w } _ { j } \cdot \mathbf { y } \leq \beta _ { R , j }$ , this implies $| { \bf w } _ { j } \cdot { \bf u } - t _ { j } | \leq \delta$ Independence of the orthogonal Gaussian projections gives

$$
P ( R ) b _ { R } = \mathop { \bf P r } _ { \scriptscriptstyle \mathbf u \sim P } [ { \bf u } \in R , \mathrm { \ p r o j } _ { V _ { R } ^ { \perp } } { \bf u } \in E _ { R } ] .
$$

By disjointness of the cells and Gaussian anti-concentration,

$$
\sum _ { R \in { \mathcal G } } P ( R ) b _ { R } \leq \sum _ { j = 1 } ^ { k } \underset { \mathbf { u } \sim P } { \mathbf { P r } } [ | \mathbf { w } _ { j } \cdot \mathbf { u } - t _ { j } | \leq \delta ] \leq k \delta .
$$

The polynomials have been constructed independently of $Q .$ . For any distribution Q satisfying $Q ( R ) \leq$ $\Lambda P ( R )$ on $\mathcal { G }$ , we therefore obtain

$$
\begin{array} { r l } { \displaystyle \sum _ { R \in \mathcal { G } } Q ( R ) \ \mathbf { E } _ { { G } _ { R } } [ p _ { R } ^ { + } - p _ { R } ^ { - } ] \leq 2 \displaystyle \sum _ { R \in \mathcal { G } } Q ( R ) b _ { R } + \tau \displaystyle \sum _ { R \in \mathcal { G } } Q ( R ) } & { } \\ { \leq 2 \Lambda \displaystyle \sum _ { R \in \mathcal { G } } P ( R ) b _ { R } + \tau } & { } \\ { \leq 2 \Lambda k \delta + \tau . } \end{array}
$$

This proves Condition 2 with $A = 2 k$

If $f$ is an intersection, choose the $h _ { j }$ as its defining halfspaces and use instead

$$
h _ { R } ^ { - } ( \mathbf { x } ) : = 2 \mathbb { 1 } \{ \mathbf { w } _ { j } \cdot \mathbf { x } \leq t _ { j } - \beta _ { R , j } { \mathrm { ~ f o r ~ e v e r y ~ } } j \in [ k ] \} - 1 ,
$$

$$
h _ { R } ^ { + } ( \mathbf { x } ) : = 2 \mathbb { 1 } \{ \mathbf { w } _ { j } \cdot \mathbf { x } \leq t _ { j } - \alpha _ { R , j } { \mathrm { ~ f o r ~ e v e r y ~ } } j \in [ k ] \} - 1 .
$$

These functions satisfy the same pointwise sandwich, and their disagreement is contained in $E _ { R }$ , so the preceding weighted gap bound applies. Since both functions are intersections of at most k halfspaces, the same two facts give Gaussian sandwiching polynomials of degree ${ \widetilde O } ( k ^ { 3 } / \tau ^ { 4 } )$ with error $\tau / 2$ on their common subspace $W _ { R }$ . The same argument therefore proves Condition 2 with this smaller degree. □

For intersections, the sandwiching degree along with our main theorem gives the following testable learning guarantee.

Corollary F.7 (Testable learning of intersections of halfspaces). Let $\mathcal { F } _ { k }$ be the class of intersections of at most k halfspaces in $\mathbb { R } ^ { d }$ , where $k \in \mathbb { N } _ { + }$ . Fix $q \ge 1 , \sigma \in ( 0 , 1 ]$ , and $c , \epsilon \in ( 0 , 1 / 2 )$ . There is an algorithm that, given samples from any distribution D on $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ , testably learns $\mathcal { F } _ { k }$ under $\mathcal { D } _ { q , \sigma }$ to error $( 1 + c ) \mathrm { o p t } _ { D } ( \mathcal { F } _ { k } ) + \epsilon$ with probability at least 99%, using

$$
{ \cal N } = d ^ { \widetilde O ( k ^ { 3 } / c ^ { 4 } ) } \exp \left( \left( \frac { q k } { c } \right) ^ { \widetilde O ( k ^ { 3 } / c ^ { 4 } ) } \log ^ { \widetilde O ( k ^ { 3 } / c ^ { 4 } ) } \left( \frac { 2 } { c \sigma \epsilon } \right) \right)
$$

samples from D and running in time poly(N).

Proof of Corollary F.7. The class $\mathcal { F } _ { k }$ consists of k-MIMs and is closed under afine restrictions. By Lemma F.6, it is $( \ell , k , 2 k , \Lambda )$ -testable well-behaved for every $\Lambda \geq 1$ , with $\ell ( \tau ) = \widetilde O ( k ^ { 3 } / \tau ^ { 4 } )$ . Applying Theorem F.2 with $K = k , A = 2 k$ , and $\ell _ { c } = \ell ( c / C ) = \widetilde { O } ( k ^ { 3 } / c ^ { 4 } )$ gives the result. □

Using the sandwiching degree for functions of halfspaces we get a testable learning result for that class.

Corollary F.8 (Testable learning of functions of halfspaces). Let $\mathcal { H } _ { k }$ be the class of arbitrary Boolean functions of at most k afine halfspaces in $\mathbb { R } ^ { d }$ , where $k \in \mathbb { N } _ { + }$ . Fix $q \geq 1 , \sigma \in ( 0 , 1 ]$ , and $c , \epsilon \in ( 0 , 1 / 2 )$ There is an algorithm that, given samples from any distribution D on $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ , testably learns $\mathcal { H } _ { k }$ under $\mathcal { D } _ { q , \sigma }$ to error $( 1 + c ) \mathrm { o p t } _ { D } ( \mathcal { H } _ { k } ) + \epsilon$ with probability at least 99%, using

$$
{ \cal N } = d ^ { \widetilde { O } ( k ^ { 5 } / c ^ { 4 } ) } \exp \left( \left( \frac { q k } { c } \right) ^ { \widetilde { O } ( k ^ { 5 } / c ^ { 4 } ) } \log ^ { \widetilde { O } ( k ^ { 5 } / c ^ { 4 } ) } \left( \frac { 2 } { c \sigma \epsilon } \right) \right)
$$

samples from D and running in time poly(N).

Proof of Corollary F.8. Every member of $\mathcal { H } _ { k }$ depends only on the span of its halfspace normals and is therefore a k-MIM. Afine restrictions preserve the class: each halfspace remains a halfspace or becomes constant, and constant inputs can be absorbed into the Boolean link. By Lemma F.6, $\mathcal { H } _ { k }$ is consequently $( \ell , k , 2 k , \Lambda )$ -testable well-behaved for every $\Lambda \geq 1$ , where $\ell ( \tau ) = \widetilde O ( k ^ { 5 } / \tau ^ { 4 } )$ . Applying Theorem F.2 with $K = k , A = 2 k$ , and $\ell _ { c } = \ell ( c / C ) = \widetilde { O } ( k ^ { 5 } / c ^ { 4 } )$ gives the result. □