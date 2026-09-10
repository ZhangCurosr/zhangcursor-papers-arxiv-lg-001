# Algorithmic stability via ensembling

Rina Foygel Barber<sup>∗</sup>

Richard J. Samworth<sup>†</sup>

September 10, 2026

## Abstract

Algorithmic stability refers to the property of an algorithm being insensitive to perturbations of the input data, where the type of perturbation may vary depending on the setting. In this work, we develop a general framework to quantify the extent to which any ensembling strategy defined via averaging can yield stability guarantees for any type of data perturbation. Our main theoretical result is a guarantee on the stability of this ensembled algorithm, given in terms of the norm of a certain covariance operator that describes the ensembling process. We show how our general framework yields interpretable and intuitive insights in several examples of perturbations of practical interest, and provides much sharper guarantees than those obtained from privacy considerations.

## 1 Introduction

In modern data analysis, there are many factors that play a role in choosing an algorithm for learning from data: we may wish to select an algorithm based on features such as accuracy (in recovering some underlying model or parameter), computational cost, privacy, robustness to outliers or to model errors, and many more. In this work, we are interested in the property of algorithmic stability, which measures the extent to which the output of an algorithm is sensitive to small perturbations of the training data; in particular, in the statistics literature, stability is commonly defined by considering the change in the output when deleting, or resampling, small amounts of the training data, but other types of perturbations have also been considered. Stability is widely considered to be a fundamental desirable property of learning algorithms (Meinshausen and B¨uhlmann, 2010; Yu, 2013; Shah and Samworth, 2013; Murdoch et al., 2019), and is closely connected with other statistical aims and properties such as generalization, learnability, reproducibility, and robustness (Mukherjee et al., 2006; Shalev-Shwartz et al., 2010; Yu and Kumbier, 2020).

Certain types of algorithms are provably stable—for instance, algorithms that are defined by strongly convex regularized optimization problems, where the regularization enforces low sensitivity to perturbations (Bousquet and Elisseef, 2002; Wibisono et al., 2009). While the algorithms used in practice in many applications are often more complex, empirically they may appear to exhibit stability, but it is known to be impossible to empirically certify stability properties when data are limited (Kim and Barber, 2023; Luo and Barber, 2024).

Consequently, a recent question of interest is whether it is possible to introduce a posthoc correction to an arbitrary base algorithm, enforcing stability without losing its favorable performance. The recent work of Solof et al. (2024a,b) shows that $b a g g i n g ^ { 1 }$ (that is, repeatedly resampling the data, and then averaging the algorithm’s output) ensures a guarantee of stability in the sense of low sensitivity to deleting small amounts of data. In this work, we develop a framework to study this phenomenon more generally: given any base algorithm, and any desired notion of stability (i.e., stability with respect to some sort of perturbation), can we use some form of ensembling in order to ensure that this stability property will hold?

## 1.1 Algorithmic stability and algorithm ensembling

Let A be a map that inputs data $z \in { \mathcal { Z } }$ and returns an output $\mathcal { A } ( z ) \in [ 0 , 1 ]$ . We would like to assess whether A is sensitive to perturbations of the input data. To motivate this question, we now introduce several key examples that we will study throughout the paper. First, we might consider stability with respect to dropping a single data point:

Example 1 (Deleting one data point). Let $z = ( z _ { 1 } , \ldots , z _ { n } )$ be a collection of training data points, and write $z _ { - i } = ( z _ { 1 } , \ldots , z _ { i - 1 } , z _ { i + 1 } , \ldots , z _ { n } )$ to denote the same data set when the ith data point has been deleted. Then the stability of A, with respect to deleting a data point from the training dataset $z _ { i }$ can be summarized by computing

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } \big ( \mathcal { A } ( z ) - \mathcal { A } ( z _ { - i } ) \big ) ^ { 2 } ,\tag{1}
$$

which is the average change in the output of A when a single data point is deleted at random from z.

In this example, even though the data vector $z$ is fixed (and might be chosen adversarially), the quantity (1) nonetheless captures an ‘average’ rather than ‘worst-case’ notion of stability, since the data value $z _ { i }$ being deleted is being chosen at random from $i \in [ n ]$

Second, it is common to consider stability with respect to data corruption:

Example 2 (Corrupting part of the data). For a random data vector $Z \in \mathbb { R } ^ { n }$ , we can define a perturbed version of Z by replacing a randomly chosen subset of data point with contaminated or corrupted values. Specifically, suppose that we select a subset $S \subseteq [ n ]$ at random, with $| S | = s$ , and each entry $Z _ { i }$ for $i \in S$ is replaced by some corrupted value $\zeta _ { i } .$ we write

$$
Z \circ \mathbf { 1 } _ { S ^ { c } } + \zeta \circ \mathbf { 1 } _ { S }
$$

to denote the corrupted data vector, where $\mathbf { 1 } _ { S } \in \{ 0 , 1 \} ^ { n }$ is the indicator vector for the set S (and similarly for $\mathbf { 1 } _ { S ^ { c } } )$ , and ◦ denotes elementwise product. Then the quantity

$$
\frac { 1 } { \binom { n } { s } } \sum _ { \stackrel { S \subseteq [ n ] } { | S | = s } } { \mathbb { E } } _ { Z \sim \pi , \zeta \sim \nu } \left[ \Big ( A ( Z ) - A \big ( Z \circ \mathbf { 1 } _ { S ^ { c } } + \zeta \circ \mathbf { 1 } _ { S } \big ) \Big ) ^ { 2 } \right]
$$

characterizes the stability of A in this setting, with respect to randomly sampled data drawn as $Z \sim \pi _ { ; }$ , and corruptions drawn as $\zeta \sim \nu$ , with the indices S of the corrupted entries chosen uniformly at random.

It is natural to ask how one might make an algorithm more stable to this type of perturbation, and bagging provides one common approach. Here we present two common variants (Breiman, 1996a; Andonova et al., 2002):

Example A (Bagging). For constructing the bagged version of an algorithm A, we may average the output of A with respect to drawing a random subset of size m, sampled uniformly without replacement from [n] (also called ‘subbagging’):

$$
\mathcal { A } _ { \mathrm { b a g } } ( z ) = \frac { 1 } { n ! / ( n - m ) ! } \sum _ { \substack { i _ { 1 } , \ldots , i _ { m } \in [ n ] } } \mathcal { A } \big ( ( z _ { i _ { 1 } } , \ldots , z _ { i _ { m } } ) \big ) .
$$

Alternatively, we may sample m indices uniformly with replacement (‘the m out of n bootstrap’):

$$
\mathcal { A } _ { \mathrm { b a g } } ( z ) = \frac { 1 } { n ^ { m } } \sum _ { i _ { 1 } , \ldots , i _ { m } \in [ n ] } \mathcal { A } \big ( ( z _ { i _ { 1 } } , \ldots , z _ { i _ { m } } ) \big ) .
$$

Another common technique for increasing stability is to ‘smooth’ the algorithm by averaging over noisy versions of the data:

Example B (Adding noise). Given data $z \in \mathbb { R } ^ { n }$ , we define an ensembled algorithm by averaging over noisy versions of the data, $z + \xi .$ , where $\xi \in \mathbb { R } ^ { n }$ is a vector of noise, with entries drawn i.i.d. from some density h on R. This leads to a smoothed algorithm:

$$
\mathcal { A } _ { \mathrm { s m t h } } ( z ) = \mathbb { E } _ { \xi _ { i } \sim h } \left[ \boldsymbol { A } ( z + \xi ) \right] .
$$

The goal of this paper is to study how ensembling strategies, such as bagging or averaging over added noise as in Examples A and B, relate to the stability properties of the resulting ensembled algorithms, such as stability with respect to deleting or corrupting data as in Examples 1 and 2. As we will see in the overview of related work below, some existing literature has established this type of connection for specific examples; in this paper, our aim is to build a broader and more general understanding of the connection between ensembling and stability.

## 1.2 Related work

Stability has been extensively studied in the statistics and learning theory literature. It is often viewed as a desirable algorithmic property in its own right, connected to qualitative goals such as interpretability or trustworthiness (Breiman, 1996b; Yu, 2013). Yu and Kumbier (2020) discuss the necessity of stability with respect to many types of perturbations or choices made by the analyst, including variability of the data, randomness inherent to the procedure, and decisions made in constructing the model or in data preprocessing.

From a theoretical standpoint, the most commonly studied notion of stability is that of stability with respect to deleting—or resampling—a single data point or a small fraction of the data (as in Example 1); see Bousquet and Elisseef (2002) for background. This type of stability condition is known to be suficient for establishing properties such as generalization (Bousquet and Elisseef, 2002; Poggio et al., 2004), learnability (Shalev-Shwartz et al., 2010), asymptotic convergence of cross-validation-based estimates of risk (Austern and Zhou, 2025; Bayle et al., 2020), avoidance of sample splitting (Lundborg et al., 2024), and validity of crossvalidation-based predictive inference (Barber et al., 2021; Amann et al., 2023). Chakraborty et al. (2026) examine the tradeof between stability and accuracy via a constrained minimax framework.

While stability with respect to data deletion or resampling has been extensively studied, a broad range of diferent notions of stability (under various names) have been proposed in statistics and in related fields. Here we mention several examples. The field of robust statistics examines the sensitivity of statistical procedures to data contamination or outliers; see Huber and Ronchetti (2011) for background. In Bayesian statistics, prior sensitivity analysis refers to examining whether the conclusions (the posterior) are stable with respect to various choices of the prior (Gelman et al., 1995). In optimization, it is common to study stability with respect to the choice of initialization point, or with respect to a perturbation to the optimization problem—for instance, Poliquin and Rockafellar (1998) study the tilt stability of a minimization problem, i.e., the perturbation to the solution that results from adding a random linear term (a ‘tilt’) to the function being minimized.

There are many statistical tools and procedures that have been developed to improve stability (and other properties) empirically, most notably bagging (Breiman, 1996a) and subbagging (Andonova et al., 2002), as described in Example A. Bagging was introduced as a tool for constructing stable models by Breiman (1996a,b), and plays a key role in statistics and machine learning, underlying many common procedures, such as random forests (Breiman, 2001). In certain situations, it is known to improve performance by reducing variance (B¨uhlmann and Yu, 2002; Hall and Samworth, 2005; Biau et al., 2010; Samworth, 2012). As mentioned earlier, the work of Solof et al. (2024a,b) establishes that bagging any base algorithm leads to a particular type of stability guarantee, namely, stability with respect to deleting one data point (Example 1); we will give details on these results in Section 4.1.

## 1.3 Our contributions

In Section 2, we introduce definitions that quantify the stability of an algorithm to an arbitrary notion of data perturbation, and construct an ensembled version of the algorithm that averages outputs over data drawn using a Markov kernel (or ‘channel’), generalizing the various special cases introduced in the examples above.

Our main theoretical results are presented in Section 3, providing an upper bound on the stability parameter of the ensembled algorithm, given in terms of a certain operator norm of the covariance operator derived from a kernel associated with the channel. (This bound is presented for algorithms taking values in the unit interval [0, 1] but can be extended via Grothendieck’s inequality to more general settings where the algorithm takes values in a real, separable Hilbert space, e.g., algorithms that return a fitted function—see Appendix B.) Section 4 is devoted to several examples that showcase the utility of our general framework.

In each case, we show how our theory yields interpretable and intuitive bounds on the stability parameter.

In general, one way of ensuring stability is via privacy: a suficiently noisy channel will make the diferences between original and perturbed data undetectable. In Section 5, we explore the connections between stability and privacy in detail, reaching the conclusion that our theory provides much tighter bounds on stability than those implied via privacy. We end with a discussion of related ideas and possible extensions in Section 6. Most proofs are deferred to Appendix A.

## 2 Framework

In this section, we will develop a unified framework and notation for describing stability with respect to perturbations (for instance, perturbations such as deleting data or corrupting data, as described in Examples 1 and 2 above), and for developing a generic notion of ensembled algorithms (for example, bagging as in Example $\mathrm { A } ,$ or averaging with respect to added noise as in Example B).

## 2.1 A general definition of stability

We begin by introducing notation that will allow us to generalize to an arbitrary notion of data perturbation. Assume the data z lies in some measurable space $\mathcal { Z }$ and let $P$ be a distribution on $\mathcal { Z } \times \mathcal { Z }$ . In a random pair $( Z , Z ^ { \prime } ) \sim P ,$ Z represents the original unperturbed data, while $Z ^ { \prime }$ is its perturbed version. Our goal is to guarantee that our algorithm returns similar outputs when trained on $Z$ versus on $Z ^ { \prime }$ , when we sample $( Z , Z ^ { \prime } ) \sim P$

To return to our motivating examples:

Example 1 (Deleting one data point). Given a data vector $z = ( z _ { 1 } , \ldots , z _ { n } )$ , define

$$
P = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { ( z , z _ { - i } ) } .
$$

This is the uniform distribution over all possible pairs $( z , z _ { - i } ) - t h a t \ i s .$ , the original data z, and the perturbed data $z _ { - i }$ obtained by deleting one data point $i \in [ n ]$ , with $i \in [ n ]$ chosen at random, as described in Example 1.

Example 2 (Corrupting part of the data). For the corrupted data model described in $E x -$ ample 2 above, we can define $P$ as the distribution of

$$
\left( Z , Z \circ \mathbf { 1 } _ { S ^ { c } } + \zeta \circ \mathbf { 1 } _ { S } \right)
$$

that is induced by sampling $Z , \zeta , S$ independently as

$$
Z \sim \pi , \quad \zeta \sim \nu , \quad S \sim \mathrm { U n i f } \left( { \binom { [ n ] } { s } } \right) ,
$$

where $\binom { [ n ] } { s }$ denotes the collection of all subsets of [n] with cardinality s.

Given a joint distribution $P ,$ , we define the stability parameter

$$
\beta _ { P } ^ { 2 } ( A ) =  { \mathbb { E } } _ { ( Z , Z ^ { \prime } ) \sim P } \left[ \left( A ( Z ) - A ( Z ^ { \prime } ) \right) ^ { 2 } \right] ,
$$

where the expectation is taken with respect to randomly sampling $( Z , Z ^ { \prime } ) \sim P$ , i.e., the original data $Z$ and its perturbed version $Z ^ { \prime }$ . Intuitively, an algorithm A is stable to perturbations if this quantity is small. But depending on the nature of the algorithm, this may not immediately be the case. The aim of ensembling is to stabilize the algorithm $\mathcal { A } \mathrm { : }$ we aim to modify the algorithm in such a way that this stability parameter is guaranteed to be small.

## 2.2 Defining an ensembled algorithm

As discussed in Section 1.1, to help ensure that an algorithm A achieves some notion of stability, it is common to modify $A$ by taking an average over perturbations of the data. We have seen two specific mechanisms that are commonly used: subsampling data (Example A) and adding noise to data (Example B).

To work within a more general framework that captures these existing mechanisms as well as other possible notions of ensembling, we now introduce some additional notation. Let $Q$ denote a Markov kernel, with $Q ( \cdot \mid z )$ defining a distribution on ${ \mathcal { Z } } ,$ for each $z \in { \mathcal { Z } }$ Given a choice of $Q .$ , we define

$$
\begin{array} { r } { \mathcal { A } _ { Q } ( z ) = \mathbb { E } _ { Z ^ { \prime } \sim Q ( \cdot \vert z ) } \left[ \mathcal { A } ( Z ^ { \prime } ) \right] . } \end{array}\tag{2}
$$

We can think of this as an ensembled version of the algorithm ${ \mathcal { A } } ,$ obtained by averaging over the draw of the perturbed data $Z ^ { \prime } \sim Q ( \cdot \mid z ) . ^ { 2 }$ Borrowing terminology from the fields of information theory and diferential privacy, we can refer to $Q$ as a ‘channel’ that converts data z into a randomly sampled output drawn from $Q ( \cdot \mid z )$ . Since $Q$ defines an ensembled version of the algorithm ${ \mathcal { A } } ,$ we will call it the ensembling channel.

To return to our earlier examples:

Example A (Bagging). Let $z ~ = ~ ( z _ { 1 } , \ldots , z _ { n } )$ denote the data. For subbagging (sampling uniformly without replacement), we define

$$
Q ( \cdot \mid z ) = { \frac { 1 } { n ! / ( n - m ) ! } } \sum _ { \substack { i _ { 1 } , \ldots , i _ { m } \in [ n ] } } \delta _ { ( z _ { i _ { 1 } } , \ldots , z _ { i _ { m } } ) } .
$$

For bootstrapping (sampling uniformly with replacement), we instead have

$$
Q ( \cdot \mid z ) = { \frac { 1 } { n ^ { m } } } \sum _ { i _ { 1 } , \ldots , i _ { m } \in [ n ] } \delta _ { ( z _ { i _ { 1 } } , \ldots , z _ { i _ { m } } ) } .
$$

Example B (Adding noise). For adding noise $\xi$ with entries sampled i.i.d. from some density h on $\mathbb { R }$ , we define the ensembling channel $Q$ via the conditional density

$$
Q ( w \mid z ) = \prod _ { i = 1 } ^ { n } h ( w _ { i } - z _ { i } ) .
$$

## 3 Main results

The aim of our theoretical analysis is to derive a simple and assumption-lean bound on the stability parameter for the ensembled algorithm $\mathcal { A } _ { Q } , \mathrm { i . e . } , \beta _ { P } ^ { 2 } ( \mathcal { A } _ { Q } )$ , working in the general framework defined above. Intuitively, in order for this quantity to be small, we need to choose the channel $Q$ to be suficiently noisy so as to “mask” the efect of the perturbation, i.e., the diference between data $Z$ and its perturbed version $Z ^ { \prime }$ when $( Z , Z ^ { \prime } ) \sim P$

As mentioned above, such bounds have already been established in the specific setting of deleting a small portion of the data: it is known that bagging $( \mathrm { i . e . }$ , the ensembling mechanism defined in Example A) leads to stability with respect to deleting data $( \mathrm { i . e . }$ , the notion of stability defined in Example 1). However, in this work our aim is to develop a general theory that will enable stability bounds to be proved more broadly, and will also ofer new insights on the reason that bagging ofers stability in the setting of Example 1.

## 3.1 Preliminaries

We will assume that the ensembling channel $Q$ is chosen such that $Q ( \cdot \mid z )$ has a density with respect to some σ-finite base measure $\mu$ on $\mathcal { Z } .$ for every $z \in { \mathcal { Z } }$ (for instance, the counting measure for the discrete setting of Example $\mathrm { A }$ , or the Lebesgue measure on $\mathbb { R } ^ { n }$ for Example B). Overloading notation, we will use $Q ( \cdot \mid z )$ to denote the density or the distribution, as appropriate.

First we define a kernel

$$
K _ { P , Q } ( z , z ^ { \prime } ) = \mathbb { E } _ { ( Z _ { 0 } , Z _ { 1 } ) \sim P } \left[ \left( Q ( z \mid Z _ { 0 } ) - Q ( z \mid Z _ { 1 } ) \right) \cdot \left( Q ( z ^ { \prime } \mid Z _ { 0 } ) - Q ( z ^ { \prime } \mid Z _ { 1 } ) \right) \right] .
$$

Note that $K _ { P , Q }$ is positive semidefinite by construction. Let $T _ { P , Q }$ be the associated operator: for any bounded measurable function $f : { \mathcal { Z } } \to \mathbb { R }$ , define $T _ { P , Q } f : \mathcal { Z }  \mathbb { R }$ as

$$
[ T _ { P , Q } f ] ( z ) = \int _ { \mathcal { Z } } K _ { P , Q } ( z , z ^ { \prime } ) f ( z ^ { \prime } ) ~ \mathrm { d } \mu ( z ^ { \prime } ) .
$$

The following lemma gives an interpretation of this operator:

Lemma 1. Let $f , g : { \mathcal { Z } } \to \mathbb { R }$ be bounded functions. Define their ensembled versions

$$
f _ { Q } ( z ) = \mathbb { E } _ { Z \sim Q ( \cdot | z ) } \left[ f ( Z ) \right] , \quad g _ { Q } ( z ) = \mathbb { E } _ { Z \sim Q ( \cdot | z ) } \left[ g ( Z ) \right] .
$$

Then

$$
\int _ { \mathcal { Z } } f ( z ) \cdot [ T _ { P , Q } { g } ] ( z ) \mathrm { d } \mu ( z ) = \mathbb { E } _ { ( Z _ { 0 } , Z _ { 1 } ) \sim P } \left[ \left( f _ { Q } ( Z _ { 0 } ) - f _ { Q } ( Z _ { 1 } ) \right) \cdot \left( g _ { Q } ( Z _ { 0 } ) - g _ { Q } ( Z _ { 1 } ) \right) \right] .
$$

We also define an operator norm on $T _ { P , Q }$ as

$$
\| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } = \operatorname* { s u p } \left\{ \| T _ { P , Q } f \| _ { L _ { 1 } } : \| f \| _ { L _ { \infty } } \le 1 \right\} ,
$$

where for any measurable $f : \mathcal { Z } \to \mathbb { R }$ we define $\begin{array} { r } { \| f \| _ { L _ { 1 } } = \int _ { \mathcal Z } | f ( z ) | \ \mathsf { d } \mu ( z ) } \end{array}$ , and $\| f \| _ { L _ { \infty } } =$ $\operatorname* { s u p } _ { z \in { \mathcal { Z } } } | f ( z ) |$ . By construction (and using the fact that $K _ { P , Q }$ is positive semidefinite), we can equivalently write

$$
| | T _ { P , Q } | | _ { L _ { \infty } \to L _ { 1 } } = \operatorname* { s u p } \left\{ \int _ { \mathcal { Z } } \int _ { \mathcal { Z } } K _ { P , Q } ( z , z ^ { \prime } ) f ( z ) f ( z ^ { \prime } ) \mathrm { d } \mu ( z ) \mathrm { d } \mu ( z ^ { \prime } ) : | | f | | _ { L _ { \infty } } \le 1 \right\} .\tag{3}
$$

## 3.2 Stability guarantee

Our main result shows that the stability $\beta _ { P } ^ { 2 } ( \mathcal { A } _ { Q } )$ can be characterized by the norm of the associated operator $T _ { P , Q }$

Theorem 2. For any $\mathcal { A } : \mathcal { Z } \to [ 0 , 1 ]$ , it holds that

$$
\beta _ { P } ^ { 2 } ( A _ { Q } ) \leq \frac { 1 } { 4 } \| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } .
$$

In particular, note that this upper bound does not depend on the particular base algorithm A. As we will see below, this bound is tight when viewed as a result required to hold simultaneously over all A—in fact, it is possible to construct an algorithm A and an ensembling channel $Q$ such that this upper bound is achieved for every $P .$ However, given a single algorithm ${ \mathcal { A } } ,$ the bound may be loose, i.e., a particular ensembled algorithm $\mathcal { A } _ { Q }$ may exhibit more stability than is guaranteed by this upper bound.

Theorem 2 can also be extended to the setting where the output of our algorithm takes values in a separable Hilbert space; see Theorem 10 in Appendix B.

Proof of Theorem 2. Define $\begin{array} { r } { f ( z ) = \mathcal { A } ( z ) - \frac { 1 } { 2 } } \end{array}$ . Note that $\begin{array} { r } { \mathcal { A } _ { Q } ( z ) = f _ { Q } ( z ) + \frac { 1 } { 2 } } \end{array}$ for all $z ,$ by construction. Then

$$
\begin{array} { r l } & { \beta _ { P } ^ { 2 } ( \mathcal { A } _ { Q } ) = \mathbb { E } _ { P } \left[ \left( \mathcal { A } _ { Q } ( Z _ { 0 } ) - \mathcal { A } _ { Q } ( Z _ { 1 } ) \right) ^ { 2 } \right] } \\ & { \quad \quad \quad = \mathbb { E } _ { P } \left[ \left( f _ { Q } ( Z _ { 0 } ) - f _ { Q } ( Z _ { 1 } ) \right) \cdot \left( f _ { Q } ( Z _ { 0 } ) - f _ { Q } ( Z _ { 1 } ) \right) \right] } \\ & { \quad \quad \quad = \displaystyle \int _ { Z } f ( z ) \cdot [ T _ { P , Q } f ] ( z ) \ \mathrm { d } \mu ( z ) \mathrm { ~ b y ~ L e m m a ~ 1 ~ } ( \mathrm { a p p l i e d ~ w i t h ~ } f = g ) } \\ & { \quad \quad \quad \le \| f \| _ { L _ { \infty } } \| T _ { P , Q } f \| _ { L _ { 1 } } \le \| f \| _ { L _ { \infty } } ^ { 2 } \| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } } \\ & { \quad \quad \quad \le \frac { 1 } { 4 } \| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } , } \end{array}
$$

where the last step holds since f takes values in $[ - \frac { 1 } { 2 } , \frac { 1 } { 2 } ]$ (because A takes values in [0, 1]).

Next, we provide a useful relaxation of this bound, which allows us to work with a diferent operator norm.

Corollary 3. Let ψ be any density with respect to the base measure $\mu _ { ; }$ with $\psi ( z ) > 0$ for all z. Then for any $\mathcal { A } : \mathcal { Z } \to [ 0 , 1 ]$ , it holds that

$$
\beta _ { P } ^ { 2 } ( A _ { Q } ) \leq \frac { 1 } { 4 } \| \tilde { T } _ { P , Q } ^ { \psi } \| _ { L _ { 2 }  L _ { 2 } } ,
$$

where we define the operator $\tilde { T } _ { P , Q } ^ { \psi }$ as

$$
[ \tilde { T } _ { P , Q } ^ { \psi } f ] ( z ) = \int _ { \mathcal { Z } } \frac { K _ { P , Q } ( z , z ^ { \prime } ) } { \sqrt { \psi ( z ) \psi ( z ^ { \prime } ) } } \cdot f ( z ^ { \prime } ) \mathrm { d } \mu ( z ^ { \prime } ) .
$$

Proof of Corollary 3. Fix any $f : \mathcal { Z } \to \mathbb { R }$ with $\| f \| _ { L _ { \infty } } \leq 1$ . A straightforward calculation shows that

$$
\int _ { \mathcal { Z } } \int _ { \mathcal { Z } } K _ { P , Q } ( z , z ^ { \prime } ) f ( z ) f ( z ^ { \prime } ) \mathrm { d } \mu ( z ) \mathrm { d } \mu ( z ^ { \prime } ) = \int _ { \mathcal { Z } } \tilde { f } ( z ) \cdot [ \tilde { T } _ { P , Q } ^ { \psi } \tilde { f } ] ( z ) \mathrm { d } \mu ( z ) ,
$$

where ${ \tilde { f } } ( z ) = { \sqrt { \psi ( z ) } } \cdot f ( z )$ . Note that $\begin{array} { r } { \int _ { \mathcal { Z } } \tilde { f } ( z ) ^ { 2 } \mathop { } \mathrm { d } \mu ( z ) = \int _ { \mathcal { Z } } \psi ( z ) f ( z ) ^ { 2 } \mathop { } \mathrm { d } \mu ( z ) \le 1 } \end{array}$ , since $\psi$ is a density and $\| f \| _ { L _ { \infty } } \leq 1$ . Therefore,

$$
\int _ { \mathcal { Z } } \int _ { \mathcal { Z } } K _ { P , Q } ( z , z ^ { \prime } ) f ( z ) f ( z ^ { \prime } ) \mathrm { d } \mu ( z ) \mathrm { d } \mu ( z ^ { \prime } ) \leq \| \tilde { T } _ { P , Q } ^ { \psi } \| _ { L _ { 2 } \to L _ { 2 } } .
$$

Since this holds for any f with $\| f \| _ { L _ { \infty } } \leq 1$ , applying (3) we have

$$
\| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } \le \| \tilde { T } _ { P , Q } ^ { \psi } \| _ { L _ { 2 } \to L _ { 2 } } .
$$

Combined with Theorem 2, this completes the proof.

## 3.3 Universality

Next, we show that ensembling is a universal tool for obtaining stable algorithms. In particular, we will show that for any algorithm ${ \mathcal { A } } ,$ we can formulate A as the ensembled version of some base algorithm $\mathcal { A } ^ { \ast }$ such that our theory above exactly characterizes the stability of $\mathcal { A } \mathrm { : }$ that is, we construct an algorithm $\mathcal { A } ^ { \ast }$ such that

$$
\boldsymbol { \mathcal { A } } = [ \boldsymbol { \mathcal { A } } ^ { * } ] _ { Q } ,
$$

and such that the stability $\beta _ { P } ^ { 2 } ( A )$ of the algorithm A can be fully described by the result of Theorem 2. (We note however that the relaxation provided in Corollary 3 may not be tight.)

Theorem 4. Let $\mathcal { A } : \mathcal { Z }  [ 0 , 1 ]$ be any algorithm. Then there exists another algorithm $\mathcal { A } ^ { \ast } : \mathcal { Z }  [ 0 , 1 ]$ and an ensembling channel $Q ( \cdot \mid z )$ such that

$$
\mathcal { A } ( z ) = [ \mathcal { A } ^ { \ast } ] _ { Q } ( z ) \mathrm { ~ f o r ~ a l l ~ } z \in \mathcal { Z }
$$

and such that

$$
\beta _ { P } ^ { 2 } ( \mathcal { A } ) = \frac { 1 } { 4 } \| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } \mathrm { ~ f o r ~ a l l ~ d i s t r i b u t i o n s ~ } P \mathrm { ~ o n ~ } \mathcal { Z } \times \mathcal { Z } .
$$

## 4 Stability guarantees for examples

Next we examine the implications of our main result, Theorem 2, in the settings of our examples. We will see how bagging (Example A) can ensure stability with respect to deleting data (Example 1), and how smoothing by adding noise (Example B) can ensure stability with respect to the corrupted data model (Example 2). We then also develop one additiona example: stability with respect to missing data.

## 4.1 Bagging & deleted data (Example $\mathbf { 1 } + \mathbf { A } )$

We first need some notation: for any sequence of indices $r = ( i _ { 1 } , \ldots , i _ { m } ) \in [ n ] ^ { m }$ , we define the corresponding subvector of z as

$$
z _ { r } = ( z _ { i _ { 1 } } , \dots , z _ { i _ { m } } ) ,
$$

and let $p _ { r }$ denote the probability of selecting this ‘bag’. With this notation, the bagged algorithm is given by

$$
\mathcal { A } _ { \mathrm { b a g } } ( z ) = \sum _ { r } p _ { r } \mathcal { A } ( z _ { r } ) ,\tag{4}
$$

for a data vector $z = ( z _ { 1 } , \ldots , z _ { n } )$ of length n. We can make this more concrete by considering the two most common examples:

• For subbagging (sampling m indices uniformly at random without replacement), we have $\begin{array} { r } { p _ { r } \ = \ \frac { 1 } { n ! / ( n - m ) ! } } \end{array}$ for all $r \in [ n ] ^ { m }$ consisting of m distinct indices, and $p _ { r } ~ = ~ 0$ otherwise.

• For m out of n bootstrapping (sampling m indices uniformly at random with replacement), we have $\begin{array} { r } { p _ { r } = \frac { 1 } { n ^ { m } } } \end{array}$ for all $r \in [ n ] ^ { m }$

We will assume that the bagged algorithm on the data vector $z _ { - i }$ (which has length $n - 1 )$ obeys

$$
\mathcal { A } _ { \mathrm { b a g } } ( z _ { - i } ) = \frac { \sum _ { r } \mathbb { 1 } _ { i \notin r } p _ { r } \mathcal { A } ( z _ { r } ) } { \sum _ { r } \mathbb { 1 } _ { i \notin r } p _ { r } } ,\tag{5}
$$

that is, sampling a random bag from $[ n ] \backslash \{ i \}$ is equivalent to sampling a bag r with probabilities $p _ { r }$ , conditional on the event $i \not \in r$ . This holds for both subbagging and bootstrapping, for any choice of m $( \mathrm { e . g . }$ , sampling m entries from $z _ { - i } .$ , without replacement, is equivalent to sampling m entries without replacement from z and conditioning on the event that entry i was not sampled).

The work of Solof et al. (2024a,b) shows a stability guarantee for $\mathcal { A } _ { \mathrm { b a g } }$ for certain bagging schemes (we give details below). We will now see that our general framework can generalize this result, while recovering the bound (7) as a special case. To state our result, define $\pi _ { i } =$ $\textstyle \sum _ { r } \mathbb { I } _ { i \in r } p _ { r }$ (the probability that a randomly sampled bag contains $i )$ , and $\begin{array} { r } { \sum _ { i j } = \sum _ { r } \mathbb { 1 } _ { i , j \in r } p _ { r } - } \end{array}$ $\pi _ { i } \pi _ { j }$ (the covariance between the events $i \in r$ and $j \in r )$ .

Proposition 5. Let A be any base algorithm returning outputs in [0, 1]. The bagged algorithm $A _ { \mathrm { b a g } } ~ ( 4 )$ satisfies

$$
\operatorname* { s u p } _ { z = ( z _ { 1 } , \ldots , z _ { n } ) } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \big ( A _ { \mathrm { b a g } } ( z ) - A _ { \mathrm { b a g } } ( z _ { - i } ) \big ) ^ { 2 } \leq \frac { 1 } { 4 } \| M _ { \mathrm { b a g } } \| _ { 2 } ,
$$

where $\| \cdot \| _ { 2 }$ denotes the usual matrix operator norm, and where $M _ { \mathrm { b a g } } \in \mathbb { R } ^ { n \times n }$ is the matrix defined by entries

$$
( M _ { \mathrm { b a g } } ) _ { i j } = \frac { 1 } { n } \frac { \Sigma _ { i j } } { ( 1 - \pi _ { i } ) ( 1 - \pi _ { j } ) } .
$$

We now see how Solof et al. $( 2 0 2 4 \mathrm { a } , \mathrm { b } ) \colon$ s results can be recovered as a special case. Their work assumes a symmetry condition ensuring that the n indices are treated interchangeably in the process of sampling a bag r, and moreover assumes correlation cannot be positive:

$$
\pi _ { i } = p \mathrm { ~ f o r ~ a l l ~ } i , \quad \Sigma _ { i j } = - q \leq 0 \mathrm { ~ f o r ~ a l l ~ } i \neq j\tag{6}
$$

for some $p , q$ (this holds for both subbagging and bootstrapping). Under this condition, we can then calculate

$$
\begin{array} { r } { ( M _ { \mathrm { b a g } } ) _ { i j } = \left\{ \frac { 1 } { n } \cdot \frac { p } { 1 - p } , \ ~ i = j , \right. } \\ { \frac { 1 } { n } \cdot \frac { - q } { ( 1 - p ) ^ { 2 } } , \ \left. ~ i \neq j , \right. } \end{array}
$$

which implies

$$
\| M _ { \mathrm { b a g } } \| _ { 2 } = \frac { 1 } { n } \cdot \left( \frac { p } { 1 - p } + \frac { q } { ( 1 - p ) ^ { 2 } } \right) \leq \frac { 1 } { n - 1 } \cdot \frac { p } { 1 - p } ,
$$

where the inequality holds since we must have $\begin{array} { r } { q \leq \frac { p ( 1 - p ) } { n - 1 } } \end{array}$ (because $\Sigma$ is positive semidefinite). Combined with Proposition 5, this implies the stability bound

$$
\operatorname* { s u p } _ { z = ( z _ { 1 } , \ldots , z _ { n } ) } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \big ( A _ { \mathrm { b a g } } ( z ) - A _ { \mathrm { b a g } } ( z _ { - i } ) \big ) ^ { 2 } \leq \frac { 1 } { 4 ( n - 1 ) } \cdot \frac { p } { 1 - p } ,\tag{7}
$$

which exactly recovers the main result of Solof et al. (2024a,b).

## 4.2 Adding noise $\&$ corrupted data (Example 2+B)

Next we derive stability bounds for the corrupted data setting. For $x \in \mathbb { R } ^ { s }$ , define

$$
\Delta _ { h } ( x ) = \mathrm { d } _ { \chi ^ { 2 } } ( x + W \vert \vert W ) \mathrm { ~ w h e r e ~ } W = ( W _ { 1 } , \dots , W _ { s } ) \mathrm { ~ f o r ~ } W _ { 1 } , \dots , W _ { s } \overset { \mathrm { i i d } } { \sim } h ,
$$

where we recall that $h$ is the density of the entries of the ensembling noise, and where $\mathrm { d } _ { \chi ^ { 2 } }$ denotes the $\chi ^ { 2 } \mathrm { - d i v e r g e n c e }$ . Essentially, $\Delta _ { h } ( x )$ is small if noise drawn from h can mask a shift of magnitude $| x _ { i } |$ across draws $i = 1 , \dots , s$

Proposition 6. In the setting and notation above, the smoothed algorithm $A _ { \mathrm { s m t h } }$ satisfies

$$
\frac { 1 } { \binom { n } { s } } \sum _ { S \in \binom { [ n ] } { s } } \mathbb { E } _ { Z \sim \pi , \zeta \sim \nu } \left[ \left( \mathcal { A } _ { \mathrm { s m t h } } ( Z ) - \mathcal { A } _ { \mathrm { s m t h } } ( Z \circ { \bf 1 } _ { S ^ { c } } + \zeta \circ { \bf 1 } _ { S } ) \right) ^ { 2 } \right] \leq \frac { s } { 4 n } \mathbb { E } \left[ \operatorname* { m a x } _ { S \in \binom { [ n ] } { s } } \Delta _ { h } ( \zeta _ { S } - Z _ { S } ) \right] .
$$

For example, suppose that h is the density of the ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ distribution. Since

$$
\mathrm { d } _ { \chi ^ { 2 } } \big ( \mathcal { N } ( \mu , \sigma ^ { 2 } \mathbf { I } _ { s } ) \| \mathcal { N } ( \mathbf { 0 } _ { s } , \sigma ^ { 2 } \mathbf { I } _ { s } ) \big ) = e ^ { \| \mu \| _ { 2 } ^ { 2 } / \sigma ^ { 2 } } - 1
$$

for any $\mu \in \mathbb { R } ^ { s }$ , we therefore have

$$
\Delta _ { h } ( x ) = e ^ { \| x \| _ { 2 } ^ { 2 } / \sigma ^ { 2 } } - 1 ,
$$

and so the stability bound above can be simplified to

$$
\frac { s } { 4 n } \mathbb { E } \left[ \exp \left\{ \operatorname* { m a x } _ { S \in \binom { [ n ] } { s } } \| Z _ { S } - \zeta _ { S } \| _ { 2 } ^ { 2 } / \sigma ^ { 2 } \right\} - 1 \right] .
$$

For instance, if $\| Z \| _ { \infty } , \| \zeta \| _ { \infty } \leq B$ almost surely $( \mathrm { i . e . , ~ } \pi$ and $\nu$ are supported on $[ - B , B ] ^ { n } )$ , then the stability result can be bounded by

$$
{ \frac { s \left( e ^ { 4 B ^ { 2 } s / \sigma ^ { 2 } } - 1 \right) } { 4 n } } .\tag{8}
$$

As another example, we may consider adding more heavy-tailed noise, which will more efectively mask corruptions (and will allow us to avoid assuming boundedness). Let $h$ be the standard Cauchy distribution. By calculating the $\chi ^ { 2 }$ divergence between Cauchy distributions (Nielsen and Okamura, 2022), we have

$$
\Delta _ { h } ( x ) = \frac { x ^ { 2 } } { 2 } ,
$$

and so the stability bound above can be simplified to

$$
\frac { s } { 8 n } \mathbb { E } \left[ \operatorname* { m a x } _ { S \in \left( \left[ n \right] \right) } \| Z _ { S } - \zeta _ { S } \| _ { 2 } ^ { 2 } \right] .
$$

For instance, if each $Z _ { i } , \zeta _ { i }$ is subgaussian, we then have a bound that is

$$
\mathcal { O } \left( \frac { s ^ { 2 } \log n } { n } \right) .
$$

## 4.3 Stability with respect to missing data

We next develop an additional example, in the context of missing data. Let $Z$ denote the complete data (with no missingness), while $Z _ { \Omega }$ denotes data with missingness, where Ω is the ‘mask’ specifying which parts of the data are observed. Specifically, we assume $Z$ takes values in a space $\mathcal { Z } = \mathbb { R } ^ { \tau }$ , for instance:

• For matrix-valued data $Z \in \mathbb { R } ^ { n \times m }$ , we can equivalently write $Z \in \mathbb { R } ^ { [ n ] \times [ m ] }$ , by representing a $n \times m$ matrix as a function mapping an index $( i , j ) \in \mathcal { I } = [ n ] \times [ m ]$ to a value $Z _ { i j } \in \mathbb { R }$

$Z$ may consist of functional data: we may have $Z \in \mathbb { R } ^ { [ 0 , 1 ] }$ in the setting where we observe real-valued data over the time period $t \in [ 0 , 1 ]$ or $Z \in \mathbb { R } ^ { [ 0 , 1 ] \times [ 0 , 1 ] }$ if we observe real-valued data over a 2-dimensional spatial domain.

The observation mask Ω is then a subset of $\mathcal { T } ,$ specifying which part of $Z$ is observed (with $\Omega = \mathcal { T }$ corresponding to the case of no missingness, i.e., $Z _ { \mathcal { T } } = Z )$ . Our goal is to ensure that $\boldsymbol { \mathcal { A } } ( \boldsymbol { Z } _ { \Omega } )$ , the output of the algorithm $A$ when the input $Z _ { \Omega }$ has some missingness, is typically similar to $\mathcal A ( Z )$ , the output when $\mathcal { A }$ is trained on the fully observed data $Z .$

Now we formalize the general setting. Let $\pi$ be a joint distribution for the pair $( Z , \Omega )$ For convenience we assume that the possible values of $\Omega ,$ under this joint distribution $\pi ,$ is at most countably infinite, with $\Omega \in \{ \omega _ { 1 } , \omega _ { 2 } , . . . \}$ (where $\omega _ { i } \subseteq \mathbb { Z }$ for each $i \geq 1 )$ We will write $\pi _ { Z }$ to denote the marginal distribution of $Z$ , and $\pi _ { i } ( z ) = \mathbb { P } _ { \pi } \left\{ \Omega = \omega _ { i } \mid Z = z \right\}$ to denote the conditional distribution of $\Omega .$ , which in general depends on Z (note that the common missing completely at random assumption would instead require $\Omega \perp \perp Z )$

Our goal is to be stable with respect to missingness, i.e., to bound

$$
\begin{array} { r } { \mathbb { E } _ { ( Z , \Omega ) \sim \pi } \left[ ( A ( Z ) - A ( Z _ { \Omega } ) ) ^ { 2 } \right] . } \end{array}
$$

We now define an ensembling procedure that will allow for this type of stability guarantee. Let $q = ( q _ { 1 } , q _ { 2 } , \dots )$ be a distribution on $\{ \omega _ { 1 } , \omega _ { 2 } , \ldots \}$ . We ensemble via a Poissonized missingness mechanism, by averaging over the following process: first sample $M \sim \mathrm { P o i s s o n } ( \lambda )$ then sample $\Omega _ { 1 } , \ldots , \Omega _ { M } \stackrel { \mathrm { i i d } } { \sim } q$ and use the observation mask $\cap _ { j = 1 } ^ { M } \Omega _ { j }$ (we take the convention that the empty intersection is the entire set I—that is, if $M \stackrel { . } { = } 0$ , then the observation mask is $\mathcal { T } _ { : }$ i.e., no missingness). More formally, we can write

$$
\mathcal { A } _ { \mathrm { P M } ( \lambda ) } ( z ) = \mathbb { E } _ { M \sim \mathrm { P o i s s o n } ( \lambda ) , \Omega _ { j } \sim q } \left[ A ( Z _ { \cap _ { j = 1 } ^ { M } \Omega _ { j } } ) \right] .\tag{9}
$$

This ensembling kernel leads to the following stability guarantee:

Proposition 7. In the setting and notation above, then the algorithm $\boldsymbol { \mathcal { A } } _ { \mathrm { P M } ( \lambda ) }$ constructed via Poissonized missingness satisfies

$$
\begin{array} { r } { \mathbb { E } _ { ( Z , \Omega ) \sim \pi } \left[ ( \mathcal { A } _ { \mathrm { P M } ( \lambda ) } ( z ) - \mathcal { A } _ { \mathrm { P M } ( \lambda ) } ( z _ { \Omega } ) ) ^ { 2 } \right] \leq \frac { \mathbb { E } _ { Z \sim \pi _ { Z } } \left[ \operatorname* { s u p } _ { i } \{ \pi _ { i } ( Z ) / q _ { i } \} \right] } { \lambda } . } \end{array}
$$

In particular, if $\operatorname* { s u p } _ { i } \{ \pi _ { i } ( Z ) / q _ { i } \}$ is bounded (which may be possible if the dependence between Ω and Z is not too strong), we see that the stability guarantee holds with parameter $\asymp \lambda ^ { - 1 }$

For intuition, suppose that the original distribution π typically leads to an ϵ portion of the data being missing $( \mathrm { e . g . , } \approx \epsilon n m$ entries are missing, from an $n \times m$ matrix). The Poissonized missingness mechanism will then lead to $\approx \lambda \epsilon$ missingness, since we have $\mathbb { E } \left[ M \right] = \lambda \ ( \mathrm { e . g . }$ ≈ λϵnm entries are missing, from an $n \times m \mathrm { m a t r i x } )$ . This means that to ensure that stability holds with parameter $\asymp \lambda ^ { - 1 }$ , we pay the price of increasing the fraction of missing data by the factor λ in the ensembling procedure.

## 5 Connections with privacy

Our framework for studying stability can be interpreted as saying that the channel $Q$ needs to be suficiently noisy so as to ‘mask’ the diference between the data $Z$ and its perturbed version $Z ^ { \prime }$ . On the surface, this intuition resembles the idea of privacy: it seems that we need the distributions $Q ( \cdot \mid Z )$ and $Q ( \cdot \mid Z ^ { \prime } )$ to be similar when $( Z , Z ^ { \prime } ) \sim P$ . Indeed, the following proposition makes this connection explicit:

Proposition 8. The operator $T _ { P , Q }$ defined above satisfies

$$
\begin{array} { r } { \| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } \le 4 \mathbb { E } _ { ( Z , Z ^ { \prime } ) \sim P } \left[ \left( \mathrm { d } _ { \mathrm { T V } } ( Q ( \cdot \mid Z ) , Q ( \cdot \mid Z ^ { \prime } ) ) \right) ^ { 2 } \right] , } \end{array}
$$

where $\mathrm { d } _ { \mathrm { T V } }$ denotes the total variation distance.

This upper bound relates to the statistical framework of total variation privacy (Barber and Duchi, 2014): for instance, for the setting of deleting one data point, we might impose a privacy constraint

$$
\operatorname { d } _ { \operatorname { T V } } ( Q ( \cdot \mid z ) , Q ( \cdot \mid z _ { - i } ) ) \leq \epsilon { \mathrm { ~ f o r ~ a l l ~ } } z = ( z _ { 1 } , \ldots , z _ { n } ) { \mathrm { ~ a n d ~ a l l ~ } } i \in [ n ] .\tag{10}
$$

(This is strictly weaker than the more commonly used diferential privacy condition (Dwork et al., 2006); see Dwork and Roth (2014) and Su (2025) for an overview of various privacy frameworks.)

Under a total variation privacy assumption (10), Proposition 8 ensures $\| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } \le$ $4 \epsilon ^ { 2 }$ , which then leads to a stability guarantee $\beta _ { P } ^ { 2 } ( \mathcal { A } _ { Q } ) \le \epsilon ^ { 2 }$ by Theorem 2. However, the inequality in Proposition 8 is often extremely loose. We can see this in the settings of our two examples:

Example 1+A. Recall that $Q ( \cdot \mid z )$ is the distribution over all bags r given by probabilities $p _ { \tau }$ (as in (4)), while $Q ( \cdot \mid z _ { - i } )$ is the same distribution conditioned on the event $i \notin r \ ( a s$ in (5)). Since this event has probability $1 - \pi _ { i }$ , we can therefore calculate $\operatorname { d _ { T V } } ( Q ( \cdot \mid z ) , Q ( \cdot \mid z _ { - i } ) ) = \pi _ { i }$ In particular, in the special case of assumptions (6),

$$
\operatorname { d _ { T V } } ( Q ( \cdot \mid z ) , Q ( \cdot \mid z _ { - i } ) ) = p
$$

for all $i \in [ n ]$ . Therefore, Proposition 8 yields the bound

$$
\operatorname* { s u p } _ { z = ( z _ { 1 } , \ldots , z _ { n } ) } { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \left( { \cal A } ( z ) - { \cal A } ( z _ { - i } ) \right) ^ { 2 } \leq p ^ { 2 } .
$$

This is a much weaker bound than the stability guarantee (7) (which recovers the upper bound ${ \frac { 1 } { 4 ( n - 1 ) } } \cdot { \frac { p } { 1 - p } }$ established by Solof et al. $( 2 0 \% a ) )$ . In particular, this bound does not decrease with n, and thus illustrates the looseness of Proposition 8.

Example 2+B. Recall that $\begin{array} { r } { Q ( w \mid Z ) = \prod _ { i = 1 } ^ { n } h ( w _ { i } - Z _ { i } ) } \end{array}$ . Let $Z ^ { \prime } = Z \circ \mathbf { 1 } _ { S ^ { c } } + \zeta \circ \mathbf { 1 } _ { S }$ denote the corrupted version of the data Z. Then we can calculate

$$
\begin{array} { r } { \mathrm { d } _ { \mathrm { T V } } \big ( Q ( \cdot \mid Z ) , Q ( \cdot \mid Z ^ { \prime } ) \big ) = \Delta _ { h } ^ { \mathrm { T V } } ( \zeta _ { S } - Z _ { S } ) , } \end{array}
$$

where we define

$$
\Delta _ { h } ^ { \mathrm { T V } } ( x ) = \operatorname { d } _ { \mathrm { T V } } ( x + W \Vert W ) \ \mathrm { w i t h } \ W = ( W _ { 1 } , \dots , W _ { s } ) \ \mathrm { f o r } \ W _ { 1 } , \dots , W _ { s } \overset { \mathrm { i i } } { \sim } h
$$

for $x \in \mathbb { R } ^ { s }$ . For instance, if h is given by the ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ density and we assume $\| Z \| _ { \infty } , \| \zeta \| _ { \infty } \le$ B (as in the discussion after Proposition $6 )$ , we then have $\Delta _ { h } ^ { \mathrm { T V } } ( x ) \leq \| x \| _ { 2 } / \sqrt { 2 \pi } \sigma$ and so $\mathrm { d _ { T V } } \big ( Q ( \cdot \mid Z ) , Q ( \cdot \mid Z ^ { \prime } ) \big ) \leq 2 B \sqrt { s } / \sqrt { 2 \pi } \sigma$ . Therefore, Proposition 8 implies

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { Z \sim \pi , \zeta \sim \nu } [ ( \mathcal { A } _ { \mathrm { s m t h } } ( Z ) - \mathcal { A } _ { \mathrm { s m t h } } ( Z \circ \mathbf { 1 } _ { S ^ { c } } + \zeta \circ \mathbf { 1 } _ { S } ) ^ { 2 } ] \leq \frac { 8 B ^ { 2 } s } { \pi \sigma ^ { 2 } } .
$$

Since this bound does not decrease with n, this is a weaker bound than the stability guarantee obtained in (8) under the same assumptions, thus providing another illustration of the gap in Proposition 8.

In other words, we can see via these examples that it is not the case that stability is simply a consequence of (total variation) privacy. The stability achieved by ensembling (which is characterized by $\| T _ { P , Q } \| _ { L _ { \infty }  L _ { 1 } }$ , as in the results of Theorem 2) may often be much stronger than the amount of stability implied via privacy (i.e., via the bound in Proposition 8).

## 6 Discussion

Our goal in this work has been to introduce a unified framework that quantifies the stability guarantees aforded by ensembling. We have seen that diferent types of data perturbation can be stabilized through diferent ensembling strategies.

These results naturally lead to some important open questions. First, our theoretical guarantees rely on a boundedness assumption: we assume in Theorem 2 that A returns outputs lying in [0, 1] (or, in the extension to Hilbert space-valued output in Appendix B, that the output lies in a bounded subset of a Hilbert space). While this is reasonable for certain applications, relaxing these conditions or allowing for data-dependent bounds is an important direction for future work (see Solof et al. (2024a) for some extensions of this type in the specific setting of bagging for stability with respect to data deletion).

Next, the goal of ensembling in this setting is to provide stability while retaining the good performance of the original algorithm A—that is, we would like to return output that is similar to the output of the base algorithm A, if possible (without this aim, one could achieve stability trivially: simply return a constant output, or the output of a provably stable algorithm, while ignoring A). It would be interesting to quantify the extent to which diferent ensembling strategies navigate this tradeof between the resulting stability guarantee, and the deviation from the original A.

Finally, there are many practical notions of stability that we have not considered as examples here: for instance, stability with respect to a choice of prior; with respect to a choice of bandwidth or tuning parameter; with respect to the initialization point in an optimization problem; etc. In our framework, the ‘data’ Z can denote any input to the algorithm—that is, Z can contain a choice of tuning parameters, etc, in addition to the random variables that comprise the input data. Therefore, it would be interesting to study whether our framework can give meaningful stability guarantees for questions of this type.

## Acknowledgements

R.F.B. was supported by the Ofice of Naval Research via grant N00014-24-1-2544. R.J.S.   
was supported by European Research Council Advanced Grant 101019498.

## References

Nicolai Amann, Hannes Leeb, and Lukas Steinberger. Uncertainty quantification via crossvalidation and its variants under algorithmic stability. arXiv preprint arXiv:2312.14596, 2023.

Savina Andonova, Andre Elisseef, Theodoros Evgeniou, and Massimiliano Pontil. A simple algorithm for learning stable machines. In ECAI, volume 2, pages 513–517, 2002.

Morgane Austern and Wenda Zhou. Asymptotics of cross-validation. Ann. Inst. H. Poincar´e Probab. Statist., 61:2804–2865, 2025.

Rina Foygel Barber and John C Duchi. Privacy and statistical risk: Formalisms and minimax bounds. arXiv preprint arXiv:1412.4451, 2014.

Rina Foygel Barber, Emmanuel J Cand\`es, Aaditya Ramdas, and Ryan J Tibshirani. Predictive inference with the jackknife+. The Annals of Statistics, 49(1):486–507, 2021.

Pierre Bayle, Alexandre Bayle, Lucas Janson, and Lester Mackey. Cross-validation confidence intervals for test error. Advances in Neural Information Processing Systems, 33:16339– 16350, 2020.

G´erard Biau, Fr´ed´eric C´erou, and Arnaud Guyader. On the rate of convergence of the bagged nearest neighbor estimate. Journal of Machine Learning Research, 11(2):687–712, 2010.

St´ephane Boucheron, G´abor Lugosi, and Pascal Massart. Concentration Inequalities. Oxford University Press, 2013.

Olivier Bousquet and Andr´e Elisseef. Stability and generalization. The Journal of Machine Learning Research, 2:499–526, 2002.

Leo Breiman. Bagging predictors. Machine Learning, 24(2):123–140, 1996a.

Leo Breiman. Heuristics of instability and stabilization in model selection. The Annals of Statistics, 24(6):2350–2383, 1996b.

Leo Breiman. Random forests. Machine Learning, 45(1):5–32, 2001.

Peter B¨uhlmann and Bin Yu. Analyzing bagging. The Annals of Statistics, 30(4):927–961, 2002.

Abhinav Chakraborty, Yuetian Luo, and Rina Foygel Barber. Stability and accuracy tradeofs in statistical estimation. arXiv preprint arXiv:2601.11701, 2026.

Cynthia Dwork and Aaron Roth. The algorithmic foundations of diferential privacy. Foundations and Trends® in Theoretical Computer Science, 9(3-4):211–487, 2014.

Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography Conference, pages 265– 284. Springer, 2006.

Andrew Gelman, John B Carlin, Hal S Stern, and Donald B Rubin. Bayesian Data Analysis. Chapman and Hall, 1995.

Alexander Grothendieck. Sur certaines classes de suites dans les espaces de banach et le th´eor\`eme de dvoretzky–rogers. Bol. Soc. Mat. S˜ao Paulo, 8:81–110, 1953.

Peter Hall and Richard J Samworth. Properties of bagged nearest neighbour classifiers. Journal of the Royal Statistical Society Series B: Statistical Methodology, 67(3):363–379, 2005.

Peter J Huber and Elvezio M Ronchetti. Robust Statistics. John Wiley & Sons, 2011.

Byol Kim and Rina Foygel Barber. Black-box tests for algorithmic stability. Information and Inference: A Journal of the IMA, 12(4):2690–2719, 2023.

Anton Rask Lundborg, Ilmun Kim, Rajen D Shah, and Richard J Samworth. The projected covariance measure for assumption-lean variable significance testing. The Annals of Statistics, 52(6):2851–2878, 2024.

Yuetian Luo and Rina Foygel Barber. Is algorithmic stability testable? A unified framework under computational constraints. arXiv preprint arXiv:2405.15107, 2024.

Nicolai Meinshausen and Peter B¨uhlmann. Stability selection. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(4):417–473, 2010.

Sayan Mukherjee, Partha Niyogi, Tomaso Poggio, and Ryan Rifkin. Learning theory: stability is suficient for generalization and necessary and suficient for consistency of empirical risk minimization. Advances in Computational Mathematics, 25(1):161–193, 2006.

W James Murdoch, Chandan Singh, Karl Kumbier, Reza Abbasi-Asl, and Bin Yu. Definitions, methods, and applications in interpretable machine learning. Proceedings of the National Academy of Sciences, 116(44):22071–22080, 2019.

Frank Nielsen and Kazuki Okamura. On f-divergences between Cauchy distributions. IEEE Transactions on Information Theory, 69(5):3150–3171, 2022.

Gilles Pisier. Grothendieck’s theorem, past and present. Bull. Amer. Math. Soc., 49:237–323, 2012.

Tomaso Poggio, Ryan Rifkin, Sayan Mukherjee, and Partha Niyogi. General conditions for predictivity in learning theory. Nature, 428(6981):419–422, 2004.

RA Poliquin and R Tyrrell Rockafellar. Tilt stability of a local minimum. SIAM Journal on Optimization, 8(2):287–299, 1998.

Richard J Samworth. Optimal weighted nearest neighbour classifiers. The Annals of Statistics, 40:2733–2763, 2012.

Rajen D Shah and Richard J Samworth. Variable selection with error control: another look at stability selection. Journal of the Royal Statistical Society Series B: Statistical Methodology, 75(1):55–80, 2013.

Shai Shalev-Shwartz, Ohad Shamir, Nathan Srebro, and Karthik Sridharan. Learnability, stability and uniform convergence. The Journal of Machine Learning Research, 11(90): 2635–2670, 2010.

Jake A Solof, Rina Foygel Barber, and Rebecca Willett. Bagging provides assumption-free stability. Journal of Machine Learning Research, 25(131):1–35, 2024a.

Jake A Solof, Rina Foygel Barber, and Rebecca Willett. Stability via resampling: statistical problems beyond the real line. arXiv preprint arXiv:2405.09511, 2024b.

Weijie J Su. A statistical viewpoint on diferential privacy: Hypothesis testing, representation, and Blackwell’s theorem. Annual Review of Statistics and Its Application, 12(1): 157–175, 2025.

Andre Wibisono, Lorenzo Rosasco, and Tomaso Poggio. Suficient conditions for uniform stability of regularization algorithms. Computer Science and Artificial Intelligence Laboratory Technical Report, MIT-CSAIL-TR-2009-060, 156, 2009.

Bin Yu. Stability. Bernoulli, 19:1484–1500, 2013.

Bin Yu and Karl Kumbier. Veridical data science. Proceedings of the National Academy of Sciences, 117(8):3920–3929, 2020.

## A Proofs

## A.1 Proofs of main results

Proof of Lemma 1. We calculate

$$
\begin{array} { r l } & { \displaystyle \int _ { z } f ( z ) \cdot \big [ T r , g { \mathfrak { g } } \big ] \big ( z \big ) \mathrm { d } \mu ( z ) } \\ { } & { = \displaystyle \int _ { z } f ( z ) \left[ \int _ { z } K _ { r , g } ( z , z ^ { \prime } ) \cdot g ( z ^ { \prime } ) \mathrm { d } \mu ( z ^ { \prime } ) \right] \mathrm { d } \mu ( z ) } \\ { } & { = \displaystyle \int _ { z } \int _ { z } \mathbb { E } _ { r } \left[ \left( Q ( z \mid Z _ { 0 } ) - Q ( z \mid Z _ { 1 } ) \right) \cdot \left( Q ( z ^ { \prime } \mid Z _ { 0 } ) - Q ( z ^ { \prime } \mid Z _ { 1 } ) \right) \right] \cdot f ( z ) g ( z ^ { \prime } ) \mathrm { d } \mu ( z ^ { \prime } ) \mathrm { d } \mu ( z ) } \\ { } & { = \mathbb { E } _ { \mathbb { P } } \left[ \displaystyle \int _ { z } \int _ { z } \left( Q ( z \mid Z _ { 0 } ) - Q ( z \mid Z _ { 1 } ) \right) \cdot \left( Q ( z ^ { \prime } \mid Z _ { 0 } ) - Q ( z ^ { \prime } \mid Z _ { 1 } ) \right) \cdot f ( z ) g ( z ^ { \prime } ) \mathrm { d } \mu ( z ^ { \prime } ) \mathrm { d } \mu ( z ) \right] } \\ { } & { = \mathbb { E } _ { \mathbb { P } } \left[ \displaystyle \int _ { z } \left( Q ( z \mid Z _ { 0 } ) - Q ( z \mid Z _ { 1 } ) \right) \cdot f ( z ) \mathrm { d } \mu ( z ) \cdot \displaystyle \int _ { z } \left( Q ( z ^ { \prime } \mid Z _ { 0 } ) - Q ( z ^ { \prime } \mid Z _ { 1 } ) \right) \cdot g ( z ^ { \prime } ) \mathrm { d } \mu ( z ^ { \prime } ) \right] } \\ { } & { = \mathbb { E } _ { \mathbb { P } } \left[ \left( f _ { g } ( Z _ { 0 } ) - f _ { g } ( Z _ { 1 } ) \right) \cdot \left( g ( Z _ { 0 } ) - g _ { G } ( Z _ { 1 } ) \right) \right] , } \end{array}
$$

where the second step holds by definition of $K _ { P , Q }$ , the third step holds by Fubini’s theorem, and the last step holds by definition of $f _ { Q } , g _ { Q }$ □

Proof of Theorem $\it 4 .$ We can assume $| \mathcal { Z } | > 1$ to avoid the trivial case. Let $z ^ { ( 0 ) } , z ^ { ( 1 ) } \in \mathcal { Z }$ be distinct points. Define

$$
\mathcal { A } ^ { \ast } ( z ) = \mathbb { 1 } \left\{ z = z ^ { ( 1 ) } \right\} ,
$$

and define an ensembling channel

$$
Q ( \cdot \mid z ) = ( 1 - A ( z ) ) \cdot \delta _ { z ^ { ( 0 ) } } + A ( z ) \cdot \delta _ { z ^ { ( 1 ) } } .
$$

For any input data $z ,$ this channel places all its mass on the two data points, $z ^ { ( 0 ) }$ and $z ^ { ( 1 ) }$ with weights determined by the output $\boldsymbol { \mathcal { A } } ( \boldsymbol { z } )$ of the original algorithm. (Recall that we require that $Q ( \cdot \mid z )$ is a density with respect to a base measure $\mu ,$ for all $z ;$ this holds by choosing $\mu$ to be the counting measure.) We then calculate that, for any $z \in { \mathcal { Z } }$

$$
[ A ^ { * } ] _ { Q } ( z ) = \mathbb { E } _ { Z \sim Q ( \cdot \vert z ) } \left[ A ^ { * } ( Z ) \right] = \mathbb { E } _ { Z \sim Q ( \cdot \vert z ) } \left[ \mathbb { 1 } \left\{ Z = z ^ { ( 1 ) } \right\} \right] = \mathbb { P } _ { Z \sim Q ( \cdot \vert z ) } \left\{ Z = z ^ { ( 1 ) } \right\} = \mathcal { A } ( z ) .
$$

In other words, $\mathcal { A } = [ \mathcal { A } ^ { * } ] _ { Q } , \mathrm { i . e . }$ ., the original algorithm $A$ is equivalent to the ensembled version of the base algorithm $\mathcal { A } ^ { \ast }$

Next, let $P$ be any distribution on $\mathcal { Z }$ . By Theorem $2$ we know that

$$
\beta _ { P } ^ { 2 } ( A ) = \beta _ { P } ^ { 2 } ( [ A ^ { * } ] _ { Q } ) \leq \frac { 1 } { 4 } \| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } .
$$

We next need to show that this is in fact an equality. We first calculate the associated kernel $K _ { P , Q }$ . We have

$$
\begin{array} { r l } & { K _ { P , Q } ( z ^ { ( 1 ) } , z ^ { ( 1 ) } ) = \operatorname { \mathbb { E } } _ { ( Z _ { 0 } , Z _ { 1 } ) \sim P } \left[ \left( Q ( z ^ { ( 1 ) } \mid Z _ { 0 } ) - Q ( z ^ { ( 1 ) } \mid Z _ { 1 } ) \right) ^ { 2 } \right] } \\ & { \phantom { K p a c e : \ } = \operatorname { \mathbb { E } } _ { ( Z _ { 0 } , Z _ { 1 } ) \sim P } \left[ \left( A ( Z _ { 0 } ) - A ( Z _ { 1 } ) \right) ^ { 2 } \right] = \beta _ { P } ^ { 2 } ( A ) , } \end{array}
$$

by definition of $Q .$ . Similar calculations verify that

$$
K _ { P , Q } ( z ^ { ( 0 ) } , z ^ { ( 0 ) } ) = \beta _ { P } ^ { 2 } ( \mathcal { A } ) , \quad K _ { P , Q } ( z ^ { ( 0 ) } , z ^ { ( 1 ) } ) = K _ { P , Q } ( z ^ { ( 1 ) } , z ^ { ( 0 ) } ) = - \beta _ { P } ^ { 2 } ( \mathcal { A } ) .
$$

And, $K _ { P , Q } ( z , z ^ { \prime } ) = 0$ for any other pair $( z , z ^ { \prime } )$ , since $Q$ only returns outputs in $\{ z ^ { ( 0 ) } , z ^ { ( 1 ) } \}$ Next we calculate the operator $T _ { P , Q }$ . For any function $f : \mathcal { Z } \to \mathbb { R }$ , by definition of $T _ { P , Q }$ we can verify that

$$
[ T _ { P , Q } f ] ( z ) = \left\{ \begin{array} { l l } { \beta _ { P } ^ { 2 } ( \mathcal { A } ) \cdot ( f ( z ^ { ( 0 ) } ) - f ( z ^ { ( 1 ) } ) ) , } & { z = z ^ { ( 0 ) } , } \\ { - \beta _ { P } ^ { 2 } ( \mathcal { A } ) \cdot ( f ( z ^ { ( 0 ) } ) - f ( z ^ { ( 1 ) } ) ) , } & { z = z ^ { ( 1 ) } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

Therefore, for any $f$ with $\| f \| _ { L _ { \infty } } \leq 1$ ，

$$
\| T _ { P , Q } f \| _ { L _ { 1 } } = 2 \beta _ { P } ^ { 2 } ( \mathcal { A } ) \cdot | f ( z ^ { ( 0 ) } ) - f ( z ^ { ( 1 ) } ) | \le 4 \beta _ { P } ^ { 2 } ( \mathcal { A } ) \cdot \| f \| _ { L _ { \infty } } \le 4 \beta _ { P } ^ { 2 } ( \mathcal { A } ) ,
$$

which completes the proof.

## A.2 Proofs for examples

Proof of Proposition 5. First we recall how this example can be written in our unified notation. Given the definition of the bagged algorithm $\boldsymbol { A } _ { \mathrm { b a g } } ,$ , we can see that the ensembling channel Q satisfies

$$
Q ( \cdot \mid z ) = \sum _ { r } p _ { r } \delta _ { z _ { r } } , \quad Q ( \cdot \mid z _ { - i } ) = \frac { \sum _ { r } \mathbb { 1 } _ { i \notin r } p _ { r } \delta _ { z _ { r } } } { \sum _ { r } \mathbb { 1 } _ { i \notin r } p _ { r } } .
$$

(We can view these as densities with respect to the counting measure.) We also have

$$
P = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { ( z , z _ { - i } ) } ,
$$

since the perturbed data is determined by deleting one index i at random.

Below, we will verify that the resulting operator $T _ { P , Q }$ can be characterized by the following identity: for any bounded measurable functions $f , g .$

$$
\int _ { z ^ { \prime } } f ( z ^ { \prime } ) \cdot [ T _ { P , Q } { g } ] ( z ^ { \prime } ) \mathrm { d } \mu ( z ^ { \prime } ) = \sum _ { i = 1 } ^ { n } \left( \sum _ { r } M _ { i , r } f ( z _ { r } ) \right) \cdot \left( \sum _ { r } M _ { i , r } { g } ( z _ { r } ) \right) ,\tag{11}
$$

where we define a matrix M with entries<sup>3</sup>

$$
M _ { i , r } = \frac { 1 } { \sqrt { n } } \cdot p _ { r } \left( 1 - \frac { \mathbb { 1 } _ { i \notin r } } { \sum _ { r ^ { \prime } } \mathbb { 1 } _ { i \notin r ^ { \prime } } p _ { r ^ { \prime } } } \right) .
$$

Next let v, w be vectors with entries $v _ { r } = f ( z _ { r } ) , w _ { r } = g ( z _ { r } )$ . Therefore,

$$
\int _ { z ^ { \prime } } f ( z ^ { \prime } ) \cdot [ T _ { P , Q } g ] ( z ^ { \prime } ) \mathrm { d } \mu ( z ^ { \prime } ) = \sum _ { i = 1 } ^ { n } \left( \sum _ { r } M _ { i , r } v _ { r } \right) \cdot \left( \sum _ { r } M _ { i , r } w _ { r } \right) = v ^ { \top } M ^ { \top } M w .
$$

Therefore,

$$
\operatorname* { s u p } _ { \| \boldsymbol { f } \| _ { L _ { \infty } } , \| \boldsymbol { g } \| _ { L _ { \infty } } \le 1 } \int _ { \boldsymbol { z } ^ { \prime } } \boldsymbol { f } ( \boldsymbol { z } ^ { \prime } ) \cdot [ T _ { P , Q } \boldsymbol { g } ] ( \boldsymbol { z } ^ { \prime } ) \mathrm { d } \mu ( \boldsymbol { z } ^ { \prime } ) = \operatorname* { s u p } _ { \| \boldsymbol { v } \| _ { \infty } , \| \boldsymbol { w } \| _ { \infty } \le 1 } \boldsymbol { v } ^ { \top } \boldsymbol { M } ^ { \top } \boldsymbol { M } \boldsymbol { w } ,
$$

or in other words,

$$
\| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } = \| M ^ { \top } M \| _ { \infty \to 1 } ,\tag{12}
$$

where $\| A \| _ { \infty \to 1 } = \operatorname* { s u p } _ { \| w \| _ { \infty } \leq 1 } \| A w \| _ { 1 }$

Next, define another matrix M<sup>˜</sup> with entries

$$
\tilde { M } _ { i , r } = \frac { 1 } { \sqrt { n } } \cdot \sqrt { p _ { r } } \left( 1 - \frac { \mathbb { 1 } _ { i \notin r } } { \sum _ { r ^ { \prime } } \mathbb { 1 } _ { i \notin r ^ { \prime } } p _ { r ^ { \prime } } } \right) ,
$$

so that $M _ { i , r } = \sqrt { p _ { r } } \tilde { M } _ { i , r }$ . Since $\left( p _ { r } \right)$ is a probability vector, a straightforward calculation shows that<sup>4</sup>

$$
\| M ^ { \top } M \| _ { \infty  1 } \leq \| \tilde { M } ^ { \top } \tilde { M } \| _ { 2 } = \| \tilde { M } \tilde { M } ^ { \top } \| _ { 2 } .
$$

We can also calculate, for each $i , j \in [ n ]$ 2

$$
( \tilde { M } \tilde { M } ^ { \top } ) _ { i j } = \frac { 1 } { n } \sum _ { r } p _ { r } \left( 1 - \frac { \mathbb { 1 } _ { i \notin r } } { \sum _ { r ^ { \prime } } \mathbb { 1 } _ { i \notin r ^ { \prime } } p _ { r ^ { \prime } } } \right) \cdot \left( 1 - \frac { \mathbb { 1 } _ { j \notin r } } { \sum _ { r ^ { \prime } } \mathbb { 1 } _ { j \notin r ^ { \prime } } p _ { r ^ { \prime } } } \right) = ( M _ { \mathrm { b a g } } ) _ { i j } ,
$$

and so combining everything, we have showed that

$$
\| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } \le \| M _ { \mathrm { b a g } } \| _ { 2 } .
$$

Applying Theorem 2, this verifies the stability claim for $\mathcal { A } _ { \mathrm { b a g } }$

To complete the proof, we need to verify the identity stated in (11). First, let $f _ { Q }$ be defined as in Lemma 1: that is, we have

$$
f _ { Q } ( z ^ { \prime } ) = \mathbb { E } _ { Z \sim Q ( \cdot | z ^ { \prime } ) } \left[ f ( Z ) \right] .
$$

We can evaluate this function at $z ^ { \prime } = z$ as

$$
f _ { Q } ( z ) = \mathbb { E } _ { Z \sim Q ( \cdot | z ) } \left[ f ( Z ) \right] = \sum _ { r } Q ( z _ { r } \mid z ) \cdot f ( z _ { r } ) = \sum _ { r } p _ { r } f ( z _ { r } ) ,
$$

and similarly at $z ^ { \prime } = z _ { - i }$ (for any $i \in [ n ] )$ as

$$
f _ { Q } ( z ) = \mathbb { E } _ { Z \sim Q ( \cdot \vert z ) } \left[ f ( Z ) \right] = \sum _ { r } Q ( z _ { r } \mid z _ { - i } ) \cdot f ( z _ { r } ) = \frac { \sum _ { r } \mathbb { 1 } _ { i \notin r } p _ { r } f ( z _ { r } ) } { \sum _ { r } \mathbb { 1 } _ { i \notin r } p _ { r } } ,
$$

by definition of $Q ( \cdot \mid z )$ and $Q ( \cdot \mid z _ { - i } )$ . The analogous calculations hold for $g _ { Q }$ . Then

$$
\begin{array} { r l } & { \displaystyle \int _ { \varepsilon ^ { \prime } } f ( z ^ { \prime } ) \cdot \big [ \mathcal { P } r _ { \varepsilon \theta } g \big ] ( z ^ { \prime } ) \mathrm { d } \mu ( z ^ { \prime } ) } \\ & { = \mathrm { E } _ { ( z _ { \varepsilon } , z _ { 1 } ) \sim } \big [ \big ( \int _ { \varepsilon } ( z _ { 0 } ) - \mathfrak { P } _ { \varepsilon } ( z _ { 1 } ) \big ) \cdot \big ( \mathcal { Q } ( z _ { 0 } ) - \mathcal { L } _ { \theta } ( z _ { 1 } ) \big ) \big ] \mathrm { ~ b y ~ L e m m a ~ 1 ~ } } \\ & { = \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \big ( \int _ { \mathbb { R } ^ { n } } ( z ) - f _ { \varepsilon \theta } ( z _ { \varepsilon - i } ) \big ) \cdot \big ( \mathcal { Q } _ { \theta } ( z ) - \mathcal { L } _ { \theta } ( z _ { \varepsilon - i } ) \big ) } \\ & { = \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } ( \sum _ { r } \beta _ { r } f ( z _ { \varepsilon } ) - \frac { \sum _ { r } \lambda _ { i \neq r } \beta _ { r } f ( z _ { \varepsilon } ) } { \sum _ { r } 1 _ { i \neq r } \beta _ { r } } ) \cdot ( \sum _ { r } p _ { \varepsilon } \mathcal { L } ( z _ { \varepsilon } ) - \frac { \sum _ { r } \lambda _ { i \neq r } \beta _ { r } f ( z _ { \varepsilon } ) } { \sum _ { r } 1 _ { i \neq r } \beta _ { r } } ) } \\ &  = \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } ( \sum _ { r } ( 1 - \frac { \lambda _ { i \neq r } } { \sum _ { r } 1 _ { i \neq r } \beta _ { r } f ( z _ { \varepsilon } ) } ) \cdot p _ { r } f ( z _ { \varepsilon } ) ) \cdot ( \sum _ { r } ( 1 - \frac { \lambda _ { i \neq r } } { \sum _ { r } 1 _ { i \neq r } \beta _ { r } f ( z _ { \varepsilon } ) } )  \end{array}
$$

by definition of M.

Proof of Proposition 6. First, recall that by (3), it holds that

$$
\| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } = \operatorname* { s u p } \left\{ \int _ { \mathcal { Z } } \int _ { \mathcal { Z } } f ( z _ { 0 } ) f ( z _ { 1 } ) K _ { P , Q } ( z _ { 0 } , z _ { 1 } ) \ \mathsf { d } \mu ( z _ { 0 } ) \ \mathsf { d } \mu ( z _ { 1 } ) \right\} ,
$$

where the supremum is taken over all functions f with $\| f \| _ { L _ { \infty } } \leq 1$ . We will now fix any such function $f _ { ; }$ , and will bound this integral for the example.

Let $z _ { 0 } , z _ { 1 } \in \mathbb { R } ^ { n }$ . Recall that when $( Z , Z ^ { \prime } ) \sim P _ { ; }$ , the perturbed data point $Z ^ { \prime }$ is defined by replacing $Z _ { i }$ with $\zeta _ { i }$ for all $i \in S$ , where $\dot { S } \in ( \mathbb { \Gamma } _ { s } ^ { [ n ] } )$ is sampled uniformly at random, and where $Z \sim \pi , \zeta \sim \nu$ . Then

$$
{ \frac { Q ( z _ { 0 } \mid Z ^ { \prime } ) } { Q ( z _ { 0 } \mid Z ) } } = { \frac { \prod _ { i = 1 } ^ { n } h ( z _ { 0 i } - Z _ { i } ^ { \prime } ) } { \prod _ { i = 1 } ^ { n } h ( z _ { 0 i } - Z _ { i } ) } } = { \frac { \prod _ { i \in S } h ( z _ { 0 i } - \zeta _ { i } ) } { \prod _ { i \in S } h ( z _ { 0 i } - Z _ { i } ) } } ,
$$

and similarly for $z _ { 1 }$ in place of $z _ { 0 } ,$ , which yields

$$
\begin{array} { r l } & { K _ { P , Q } ( z _ { 0 } , z _ { 1 } ) = \mathbb { E } _ { ( Z , Z ^ { \prime } ) \sim P } \left[ ( Q ( z _ { 0 } \mid Z ) - Q ( z _ { 0 } \mid Z ^ { \prime } ) ) ( Q ( z _ { 1 } \mid Z ) - Q ( z _ { 1 } \mid Z ^ { \prime } ) ) \right] } \\ & { = \mathbb { E } _ { ( Z , Z ^ { \prime } ) \sim P } \left[ Q ( z _ { 0 } \mid Z ) Q ( z _ { 1 } \mid Z ) \left( \frac { \prod _ { i = 1 } ^ { n } h \left( z _ { 0 i } - Z _ { i } ^ { \prime } \right) } { \prod _ { i = 1 } ^ { n } h \left( z _ { 0 i } - Z _ { i } \right) } - 1 \right) \left( \frac { \prod _ { i = 1 } ^ { n } h \left( z _ { 1 i } - Z _ { i } ^ { \prime } \right) } { \prod _ { i = 1 } ^ { n } h \left( z _ { 1 i } - Z _ { i } \right) } - 1 \right) \right] } \\ & { = \frac { 1 } { { \binom { n } { s } } } \sum _ { s \in { \binom { [ n ] } { s } } } \mathbb { E } _ { z \sim \nu } \left[ Q ( z _ { 0 } \mid Z ) Q ( z _ { 1 } \mid Z ) \left( \frac { \prod _ { i \in S } h \left( z _ { 0 i } - \zeta _ { i } \right) } { \prod _ { i \in S } h \left( z _ { 0 i } - Z _ { i } \right) } - 1 \right) \left( \frac { \prod _ { i \in S } h \left( z _ { 1 i } - \zeta _ { i } \right) } { \prod _ { i \in S } h \left( z _ { 1 i } - Z _ { i } \right) } - 1 \right) \right] . } \end{array}
$$

Therefore, applying Fubini’s theorem, we can write

$$
\begin{array} { r l } & { \displaystyle \int _ { \mathcal { Z } } \int _ { \mathcal { Z } } f ( z _ { 0 } ) f ( z _ { 1 } ) K _ { P , Q } ( z _ { 0 } , z _ { 1 } ) \mathrm { d } \mu ( z _ { 0 } ) \mathrm { d } \mu ( z _ { 1 } ) } \\ & { \qquad = \frac { 1 } { \binom { n } { s } } \displaystyle \sum _ { S \in \binom { [ n ] } { s } } \mathbb { E } \left[ f ( Z _ { 0 } ) f ( Z _ { 1 } ) \cdot \left( \frac { \prod _ { i \in S } h ( Z _ { 0 i } - \zeta _ { i } ) } { \prod _ { i \in S } h ( Z _ { 0 i } - Z _ { i } ) } - 1 \right) \left( \frac { \prod _ { i \in S } h ( Z _ { 1 i } - \zeta _ { i } ) } { \prod _ { i \in S } h ( Z _ { 1 i } - Z _ { i } ) } - 1 \right) \right] , } \end{array}
$$

where the expected value is taken with respect to

$$
Z \sim \pi , \zeta \sim \nu , Z _ { 0 } , Z _ { 1 } \mid Z \stackrel { \mathrm { i i d } } { \sim } Q ( \cdot \mid Z ) .
$$

Since $Z _ { 0 } , Z _ { 1 }$ are conditionally i.i.d. given $Z , \zeta$ , we can rewrite this again as

$$
\frac { 1 } { \binom { n } { s } } \sum _ { S \in \binom { [ n ] } { s } } \mathbb { E } \left[ \mathbb { E } \left[ f ( Z _ { 0 } ) \cdot \left( \frac { \prod _ { i \in S } h ( Z _ { 0 i } - \zeta _ { i } ) } { \prod _ { i \in S } h ( Z _ { 0 i } - Z _ { i } ) } - 1 \right) \Big | Z , \zeta \right] ^ { 2 } \right] .
$$

Next, define

$$
f _ { S } ( Z _ { 0 S } ) = \mathbb { E } \left[ f ( Z _ { 0 } ) \mid Z _ { 0 S } , Z , \zeta \right] .
$$

By the tower law, for each $S$

$$
\begin{array} { r l } & { \mathbb { E } \left[ f ( Z _ { 0 } ) \cdot \left( \frac { \prod _ { i \in S } h \left( Z _ { 0 i } - \zeta _ { i } \right) } { \prod _ { i \in S } h \left( Z _ { 0 i } - Z _ { i } \right) } - 1 \right) \ \middle | \ Z , \zeta \right] } \\ & { = \mathbb { E } \left[ \mathbb { E } \left[ f ( Z _ { 0 } ) \cdot \left( \frac { \prod _ { i \in S } h \left( Z _ { 0 i } - \zeta _ { i } \right) } { \prod _ { i \in S } h \left( Z _ { 0 i } - Z _ { i } \right) } - 1 \right) \ \middle | \ Z _ { 0 S } , Z , \zeta \right] \ \middle | \ Z , \zeta \right] } \\ & { = \mathbb { E } \left[ \mathbb { E } \left[ f ( Z _ { 0 } ) \ \middle | \ Z _ { 0 S } , Z , \zeta \right] \cdot \left( \frac { \prod _ { i \in S } h \left( Z _ { 0 i } - \zeta _ { i } \right) } { \prod _ { i \in S } h \left( Z _ { 0 i } - Z _ { i } \right) } - 1 \right) \ \middle | \ Z , \zeta \right] } \\ & { = \mathbb { E } \left[ f _ { S } ( Z _ { 0 S } ) \cdot \left( \frac { \prod _ { i \in S } h \left( Z _ { 0 i } - \zeta _ { i } \right) } { \prod _ { i \in S } h \left( Z _ { 0 i } - Z _ { i } \right) } - 1 \right) \ \middle | \ Z , \zeta \right] . } \end{array}
$$

Next consider the random variable $\begin{array} { r } { \frac { \prod _ { i \in S } h ( Z _ { 0 i } - \zeta _ { i } ) } { \prod _ { i \in S } h ( Z _ { 0 i } - Z _ { i } ) } - 1 } \end{array}$ . By construction, we have

$$
\mathbb { E } \left[ \left. \frac { \prod _ { i \in S } h ( Z _ { 0 i } - \zeta _ { i } ) } { \prod _ { i \in S } h ( Z _ { 0 i } - Z _ { i } ) } - 1 \right| Z , \zeta \right] = 0 , \mathbb { E } \left[ \left. \left( \frac { \prod _ { i \in S } h ( Z _ { 0 i } - \zeta _ { i } ) } { \prod _ { i \in S } h ( Z _ { 0 i } - Z _ { i } ) } - 1 \right) ^ { 2 } \right| Z , \zeta \right] = \Delta _ { h } ( Z _ { S } - \zeta _ { S } ) .
$$

Therefore, by Cauchy–Schwarz,

$$
\mathbb { E } \left[ f _ { S } ( Z _ { 0 S } ) \cdot \left( \frac { \prod _ { i \in S } h ( Z _ { 0 i } - \zeta _ { i } ) } { \prod _ { i \in S } h ( Z _ { 0 i } - Z _ { i } ) } - 1 \right) \middle | Z , \zeta \right] \leq \mathrm { V a r } ( f _ { S } ( Z _ { 0 S } ) \mid Z , \zeta ) ^ { 1 / 2 } \cdot \Delta _ { h } ( Z _ { S } - \zeta _ { S } ) ^ { 1 / 2 } .
$$

Returning to our work above we then have

$$
\begin{array} { r l r } {  { \int _ { { \mathcal Z } } \int _ { { \mathcal Z } } f ( z _ { 0 } ) f ( z _ { 1 } ) K _ { P , Q } ( z _ { 0 } , z _ { 1 } ) \mathrm { d } \mu ( z _ { 0 } ) \mathrm { d } \mu ( z _ { 1 } ) } } \\ & { } & { \leq \frac { 1 } { \binom { n } { s } } \sum _ { S \in \binom { [ n ] } { s } } \mathbb { E } [ \mathrm { V a r } ( f _ { S } ( Z _ { 0 S } ) \mid Z , \zeta ) \cdot \Delta _ { h } ( Z _ { S } - \zeta _ { S } ) ] } \\ & { } & { \leq \mathbb { E } [ \frac { 1 } { \binom { n } { s } } \sum _ { S \in \binom { [ n ] } { s } } \mathrm { V a r } ( f _ { S } ( Z _ { 0 S } ) \mid Z , \zeta ) \cdot \operatorname* { m a x } _ { S \in \binom { [ n ] } { s } } \Delta _ { h } ( Z _ { S } - \zeta _ { S } ) ] . } \end{array}
$$

Since we have $f _ { S } ( Z _ { 0 S } ) = \mathbb { E } \left[ f ( Z _ { 0 } ) \mid Z _ { 0 S } , Z , \zeta \right]$ by definition, where $Z _ { 0 1 } , \ldots , Z _ { 0 n }$ are independent conditional on $Z , \zeta .$ , we can apply Lemma 9 (given below), conditional on $Z , \zeta$ to obtain

$$
{ \frac { 1 } { \binom { n } { s } } } \sum _ { S \in { \binom { [ n ] } { s } } } \operatorname { V a r } ( f _ { S } ( Z _ { 0 S } ) \mid Z , \zeta ) \leq { \frac { s } { n } } \operatorname { V a r } ( f ( Z _ { 0 } ) \mid Z , \zeta ) .
$$

And, since $\Vert f \Vert _ { L _ { \infty } } \leq 1 , \operatorname { V a r } ( f ( Z _ { 0 } ) \mid Z , \zeta ) \leq 1$ . Therefore,

$$
\int _ { \mathcal { Z } } \int _ { \mathcal { Z } } f ( z _ { 0 } ) f ( z _ { 1 } ) K _ { P , Q } ( z _ { 0 } , z _ { 1 } ) \mathrm { d } \mu ( z _ { 0 } ) \mathrm { d } \mu ( z _ { 1 } ) \leq \frac { s } { n } \mathbb { E } \left[ \operatorname* { m a x } _ { s \in ( \frac { [ n ] } { s } ) } \Delta _ { h } ( Z _ { S } - \zeta _ { S } ) \right] .
$$

In combination with Theorem 2, this completes the proof.

Proof of Proposition 7. First we relate this example to our unified notation. The distribution $P$ on $( Z , Z ^ { \prime } )$ can be defined as follows:

P is the distribution of $( Z , Z _ { \Omega } )$ induced by $( Z , \Omega ) \sim \pi$

The ensembling kernel Q is given by

$$
Q ( \cdot \mid z ) = \sum _ { m \geq 0 } \sum _ { j _ { 1 } , \ldots , j _ { m } \geq 1 } \frac { \lambda ^ { m } e ^ { - \lambda } } { m ! } \cdot \prod _ { k = 1 } ^ { m } q _ { j _ { k } } \cdot \delta _ { z _ { \cap _ { k = 1 } ^ { m } \omega _ { j _ { k } } } } .
$$

(Note that for any $z , Q ( \cdot \mid z )$ can be expressed as a density with respect to the counting measure $\mu$ on ${ \mathcal { Z } } ,$ since it is a discrete distribution.)

Now we rewrite the ensembling procedure one more time. Given a binary vector ${ \bf a } =$ $( a _ { 1 } , a _ { 2 } , \ldots )$ with finitely many 1s, let $\omega _ { \mathbf { a } }$ denote the intersection of all selected masks as indicated by the 1s, i.e.,

$$
\begin{array} { r } { \omega _ { \mathbf { a } } = \cap _ { j \geq 1 , a _ { j } = 1 } \omega _ { j } . } \end{array}
$$

(As before, we allow for an empty intersection: if $\mathbf { a } = ( 0 , 0 , \ldots )$ then $\omega _ { \mathbf { a } } ~ = ~ \mathcal { T }$ , i.e., no missingness.)

Under the Poissonized missingness mechanism, to generate a sample from $Q ( \cdot \mid z )$ , we return $z _ { \Omega }$ where, to construct $\Omega$ , we draw $M \sim \mathrm { P o i s s o n } ( \lambda )$ and then sample M times from the Multinomia $\mathsf { l } ( q )$ distribution. By properties of the Poisson and Multinomial, this is equivalent to sampling $n _ { j } \sim \mathrm { P o i s s o n } ( \lambda q _ { j } )$ , independently for each $j \geq 1$ , and then the set $\{ j _ { 1 } , \dotsc , j _ { M } \}$ consists of (a random permutation of) these selected indices $( \mathrm { i . e . }$ , each index $j$ is included $n _ { j }$ times). By construction, the output $z _ { \Omega }$ of the ensembling kernel $Q$ is therefore determined by

$$
\begin{array} { r } { \Omega = \cap _ { k = 1 } ^ { M } \omega _ { j _ { k } } = \omega _ { \mathbf { a } } \mathrm { ~ w h e r e ~ } \mathbf { a } = ( \mathbb { 1 } _ { n _ { 1 } \geq 1 } , \mathbb { 1 } _ { n _ { 2 } \geq 1 } , \dots ) . } \end{array}
$$

In particular, we have $a _ { j } \sim \mathrm { B e r n o u l l i } ( 1 - e ^ { - \lambda q _ { j } } )$ , independently for each $j ~ \geq ~ 1$ , since $\mathbb { P } \left\{ n _ { j } \geq 1 \right\} = \mathbb { P } \left\{ { \mathrm { P o i s s o n } } ( \lambda q _ { j } ) \geq 1 \right\} = 1 - e ^ { - \lambda q _ { j } }$ . Therefore, we can write

$$
Q ( \cdot \mid z ) = \sum _ { \textbf { a } } \prod _ { j \geq 1 , a _ { j } = 1 } ( 1 - e ^ { - \lambda q _ { j } } ) \cdot \prod _ { j \geq 1 , a _ { j } = 0 } e ^ { - \lambda q _ { j } } \cdot \delta _ { z _ { \infty } } = \sum _ { \textbf { a } } e ^ { - \lambda } \cdot \prod _ { j \geq 1 } ( e ^ { \lambda q _ { j } } - 1 ) ^ { a _ { j } } \cdot \delta _ { z _ { \infty } } .
$$

In particular, given $( Z , \Omega ) \sim \pi$ , we can calculate

$$
Q ( \cdot \mid Z ) = \sum _ { \bf a } e ^ { - \lambda } \prod _ { j \geq 1 } ( e ^ { \lambda q _ { j } } - 1 ) ^ { a _ { j } } \cdot \delta _ { Z _ { \omega _ { \bf a } } } ,
$$

and, on the event $\Omega = \omega _ { i }$ (for each $i \geq 1 )$ ,

$$
\begin{array} { r l } & { Q ( \cdot  { \mid } Z _ { \Omega } ) = Q ( \cdot  { \mid } Z _ { \omega _ { i } } ) = \displaystyle \sum _ { \mathbf a } e ^ { - \lambda } \prod _ { j \geq 1 } ( e ^ { \lambda q _ { j } } - 1 ) ^ { a _ { j } } \cdot \delta _ { Z _ { \omega _ { \Omega } ( \cdot ) \omega _ { i } } } } \\ & { \qquad = \displaystyle \sum _ { \mathbf a } \frac { \mathbb { I } _ { a _ { i } = 1 } } { 1 - e ^ { - \lambda q _ { i } } } \cdot e ^ { - \lambda } \prod _ { j \geq 1 } ( e ^ { \lambda q _ { j } } - 1 ) ^ { a _ { j } } \cdot \delta _ { Z _ { \omega _ { \Omega } } } . } \end{array}
$$

From this point on, the proof follows a similar structure as the proof of Proposition 5. In this example, the operator $T _ { P , Q }$ can be characterized by the following identity: for any bounded measurable functions $f , g$

$$
\int _ { \varSigma } f ( z ^ { \prime } ) \cdot [ T _ { P , Q } { g } ] ( z ^ { \prime } ) \mathrm { d } \mu ( z ^ { \prime } ) = \mathbb { E } _ { Z \sim \pi _ { Z } } \left[ \sum _ { i \geq 1 } \left( \sum _ { \mathbf { a } } M _ { i , \mathbf { a } } ( Z ) f ( Z _ { \omega _ { \mathbf { a } } } ) \right) \cdot \left( \sum _ { \mathbf { a } } M _ { i , \mathbf { a } } ( Z ) g ( Z _ { \omega _ { \mathbf { a } } } ) \right) \right] ,\tag{13}
$$

where for each $z \in { \mathcal { Z } }$ , we define a (countably-infinite-dimensional) matrix $M ( z )$ with entries

$$
M _ { i , { \bf a } } ( z ) = \sqrt { \pi _ { i } ( z ) } \cdot e ^ { - \lambda } \cdot \prod _ { j \geq 1 } ( e ^ { \lambda q _ { j } } - 1 ) ^ { a _ { j } } \cdot \left( 1 - \frac { \mathbb { 1 } _ { a _ { i } = 1 } } { 1 - e ^ { - \lambda q _ { i } } } \right) ,
$$

indexed by $i \geq 1$ and by infinite binary vectors a with finitely many 1s. (The proof of the claim (13) is analogous to the proof of (11) in Proposition $5 . )$ Similarly to the proof of Proposition 5, this establishes that

$$
\| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } \le \mathbb { E } _ { Z \sim \pi _ { Z } } \left[ \| M ( Z ) ^ { \top } M ( Z ) \| _ { \infty \to 1 } \right] .
$$

Now define a rescaled version of the matrix,

$$
\tilde { M } _ { i , { \bf a } } ( z ) = \sqrt { \pi _ { i } ( z ) \cdot e ^ { - \lambda } \cdot \prod _ { j \geq 1 } ( e ^ { \lambda q _ { j } } - 1 ) ^ { a _ { j } } } \cdot \left( 1 - \frac { \mathbb { 1 } _ { a _ { i } = 1 } } { 1 - e ^ { - \lambda q _ { i } } } \right) .
$$

Since $\begin{array} { r } { \mathbf { a } \mapsto e ^ { - \lambda } \cdot \prod _ { i > 1 } ( e ^ { \lambda q _ { j } } - 1 ) ^ { a _ { j } } } \end{array}$ is the probability mass function of a distribution $( \mathrm { i . e . }$ , of the distribution of a determined by the Poissonized missingness mechanism), as in the proof of Proposition 5 we have

$$
\| M ( z ) ^ { \top } M ( z ) \| _ { \infty  1 } \leq \| \tilde { M } ( z ) ^ { \top } \tilde { M } ( z ) \| _ { 2 } = \| \tilde { M } ( z ) \tilde { M } ( z ) ^ { \top } \| _ { 2 } .
$$

Finally, we calculate $\tilde { M } ( z ) \tilde { M } ( z ) ^ { \top }$ . For any $i , k \geq 1$ , we compute

$$
\begin{array} { r l } & { ( \tilde { M } ( z ) \tilde { M } ( z ) ^ { \top } ) _ { i k } = \displaystyle \sum _ { \mathbf { a } } \tilde { M } _ { i , \mathbf { a } } ( z ) \tilde { M } _ { k , \mathbf { a } } ( z ) } \\ & { = \sqrt { \pi _ { i } ( z ) \pi _ { k } ( z ) } \cdot \displaystyle \sum _ { \mathbf { a } } e ^ { - \lambda } \cdot \displaystyle \prod _ { j \geq 1 } ( e ^ { \lambda q _ { j } } - 1 ) ^ { a _ { j } } \cdot \left( 1 - \frac { \mathbb { I } _ { a _ { i } = 1 } } { 1 - e ^ { - \lambda q _ { k } } } \right) \cdot \left( 1 - \frac { \mathbb { I } _ { a _ { k } = 1 } } { 1 - e ^ { - \lambda q _ { k } } } \right) } \\ & { = \sqrt { \pi _ { i } ( z ) \pi _ { k } ( z ) } \cdot \mathbb { E } _ { \mathbf { a } } \left[ \left( 1 - \frac { \mathbb { I } _ { a _ { i } = 1 } } { 1 - e ^ { - \lambda q _ { i } } } \right) \cdot \left( 1 - \frac { \mathbb { I } _ { a _ { k } = 1 } } { 1 - e ^ { - \lambda q _ { k } } } \right) \right] } \\ & { = \sqrt { \pi _ { i } ( z ) \pi _ { k } ( z ) } \cdot \left( 1 - \frac { \mathbb { P } _ { \mathbf { a } } \left\{ a _ { i } = 1 \right\} } { 1 - e ^ { - \lambda q _ { i } } } - \frac { \mathbb { P } _ { \mathbf { a } } \left\{ a _ { k } = 1 \right\} } { 1 - e ^ { - \lambda q _ { k } } } + \frac { \mathbb { P } _ { \mathbf { a } } \left\{ a _ { i } = a _ { k } = 1 \right\} } { \left( 1 - e ^ { - \lambda q _ { i } } \right) \left( 1 - e ^ { - \lambda q _ { k } } \right) } \right) , } \end{array}
$$

where the last two lines are computed with respect to the distribution of a under the Poissonized missingness mechanism. Under this distribution, we have ${ \mathbb { P } } _ { \mathbf { a } } \left\{ a _ { i } = 1 \right\} = 1 - e ^ { - \lambda q _ { i } }$ and $\mathbb { P } _ { \mathbf { a } } \left\{ a _ { k } = 1 \right\} = 1 - e ^ { - \lambda q _ { k } }$ , so the above simplifies to

$$
( \tilde { M } ( z ) \tilde { M } ( z ) ^ { \top } ) _ { i k } = \sqrt { \pi _ { i } ( z ) \pi _ { k } ( z ) } \cdot \left( \frac { { \mathbb P } _ { { \mathbf a } } \left\{ a _ { i } = a _ { k } = 1 \right\} } { ( 1 - e ^ { - \lambda q _ { i } } ) ( 1 - e ^ { - \lambda q _ { k } } ) } - 1 \right) .
$$

$\mathrm { I f } \ i \ne k$ , then $a _ { i } \perp \perp a _ { k }$ , and so ${ \mathbb P } _ { \mathbf a } \left\{ a _ { i } = a _ { k } = 1 \right\} = ( 1 - e ^ { - \lambda q _ { i } } ) ( 1 - e ^ { - \lambda q _ { k } } )$ , meaning that

$$
( \tilde { M } ( z ) \tilde { M } ( z ) ^ { \top } ) _ { i k } = 0 \mathrm { ~ f o r ~ } i \neq k .
$$

On the other hand, for the diagonal entries $i = k$ , we have ${ { \mathbb { P } } _ { \mathbf { a } } } \left\{ { { a } _ { i } } = { { a } _ { k } } = 1 \right\} = { { \mathbb { P } } _ { \mathbf { a } } } \left\{ { { a } _ { i } } = 1 \right\} =$ $1 - e ^ { - \lambda q _ { i } }$ , and so

$$
( \tilde { M } ( z ) \tilde { M } ( z ) ^ { \top } ) _ { i i } = \pi _ { i } ( z ) \cdot \left( \frac { 1 - e ^ { - \lambda q _ { i } } } { ( 1 - e ^ { - \lambda q _ { i } } ) ^ { 2 } } - 1 \right) = \frac { \pi _ { i } ( z ) } { e ^ { \lambda q _ { i } } - 1 } .
$$

Therefore, $\tilde { M } ( z ) \tilde { M } ( z ) ^ { \top }$ is a diagonal matrix, with

$$
\| \tilde { M } ( z ) \tilde { M } ( z ) ^ { \top } \| _ { 2 } = \operatorname* { s u p } _ { i } ( \tilde { M } ( z ) \tilde { M } ( z ) ^ { \top } ) _ { i i } = \operatorname* { s u p } _ { i } \frac { \pi _ { i } ( z ) } { e ^ { \lambda q _ { i } } - 1 } \leq \frac { \operatorname* { s u p } _ { i } \{ \pi _ { i } ( z ) / q _ { i } \} } { \lambda } ,
$$

since $e ^ { \lambda q _ { i } } \geq 1 + \lambda q _ { i }$ . This completes the proof.

## A.3 Additional technical results

Proof of Proposition 8. By Lemma 1, we have

$$
\begin{array} { r } { \left. T _ { P , Q } \right. _ { L _ { \infty } \to L _ { 1 } } = \operatorname* { s u p } \left\{ \mathbb E _ { ( Z , Z ^ { \prime } ) \sim P } \left[ \left( f _ { Q } ( Z ) - f _ { Q } ( Z ^ { \prime } ) \right) \cdot \left( g _ { Q } ( Z ) - g _ { Q } ( Z ^ { \prime } ) \right) \right] : \left. f \right. _ { L _ { \infty } , } \left. g \right. _ { L _ { \infty } } \le 1 \right\} , } \end{array}
$$

where $f _ { Q } , g _ { Q }$ are defined as in Lemma 1. For any function f taking values in $[ - 1 , 1 ]$ , we can write

$$
\begin{array} { r } { f _ { Q } ( Z ) - f _ { Q } ( Z ^ { \prime } ) = \mathbb { E } _ { Z ^ { \prime \prime } \sim Q ( \cdot \vert Z ) } \left[ f ( Z ^ { \prime \prime } ) \right] - \mathbb { E } _ { Z ^ { \prime \prime } \sim Q ( \cdot \vert Z ^ { \prime } ) } \left[ f ( Z ^ { \prime \prime } ) \right] \le 2 \mathrm { d } _ { \mathrm { T V } } ( Q ( \cdot \vert Z ) , Q ( \cdot \vert Z ^ { \prime } ) ) , } \end{array}
$$

by definition of total variation distance. A similar bound holds for $g _ { Q }$ . Therefore,

$$
\begin{array} { r } { \mathbb { E } _ { ( Z , Z ^ { \prime } ) \sim P } \left[ \left( f _ { Q } ( Z ) - f _ { Q } ( Z ^ { \prime } ) \right) \cdot \left( g _ { Q } ( Z ) - g _ { Q } ( Z ^ { \prime } ) \right) \right] \le \mathbb { E } _ { ( Z , Z ^ { \prime } ) \sim P } \left[ 4 \left( \mathrm { d } _ { \mathrm { T V } } ( Q ( \cdot \mid Z ) , Q ( \cdot \mid Z ^ { \prime } ) ) \right) ^ { 2 } \right] } \end{array}
$$

holds for any $f , g$ with $\| f \| _ { L _ { \infty } } , \| g \| _ { L _ { \infty } } \leq 1$ , which completes the proof.

Lemma 9. Fix any integer $n \geq 1$ . Let $X _ { 1 } , \ldots , X _ { n }$ be independent and let $f ( X _ { 1 } , \ldots , X _ { n } )$ be any square-integrable function. For any subset $S \subseteq [ n ]$ , define a random variable

$$
E _ { S } = \mathbb { E } \left[ f ( X _ { 1 } , \dots , X _ { n } ) \mid X _ { S } \right] ,
$$

the conditional expectation given $X _ { S } = ( X _ { i } ) _ { i \in S }$ . Then, for any $s \in [ n ]$

$$
{ \frac { 1 } { \binom { n } { s } } } \sum _ { | S | = s } \operatorname { V a r } ( E _ { S } ) \leq { \frac { s } { n } } \cdot \operatorname { V a r } ( f ( X _ { 1 } , \ldots , X _ { n } ) ) .
$$

We can interpret the lemma as follows: if we condition on $X _ { S }$ for a random subset $S$ of size $s ,$ we expect to capture a proportional amount of the variability of the function, i.e., $\frac { s } { n }$ of the total variance.

Proof of Lemma 9. We will prove this by induction on n. For the base case $n = 1$ , we must have $s = 1$ and so the claim holds trivially.

Now fix some $n \geq 2$ . If $s = n$ , then again the result is trivial since the only possible subset is $S = [ n ]$ . Now assume $s \leq n - 1$

Fix any $i \in [ n ]$ and define $E _ { - i } = \mathbb { E } \left[ Y \mid X _ { - i } \right]$ where $Y = f ( X _ { 1 } , \ldots , X _ { n } )$ . Note that, if $S \subseteq [ n ] \setminus \{ i \}$ , then

$$
E _ { S } = \mathbb { E } \left[ Y \mid X _ { S } \right] = \mathbb { E } \left[ \mathbb { E } \left[ Y \mid X _ { - i } \right] \mid X _ { S } \right] = \mathbb { E } \left[ E _ { - i } \mid X _ { S } \right] .
$$

Now apply the result of the lemma, with $n - 1$ in place of $n .$ , and with $E _ { - }$ <sub>−i</sub> in place of $Y { : }$ we obtain

$$
\sum _ { | S | = s } \operatorname { V a r } ( E _ { S } ) \leq { \binom { n - 2 } { s - 1 } } \cdot \operatorname { V a r } ( E _ { - i } ) .\tag{14}
$$

Next we sum over all i. For the right-hand side of (14) we calculate

$$
\begin{array} { l } { \displaystyle \sum _ { i = 1 } ^ { n } \operatorname { V a r } ( E _ { - i } ) = \sum _ { i = 1 } ^ { n } \operatorname { ( V a r } ( Y ) - \mathbb { E } \left[ \operatorname { V a r } ( Y \mid X _ { - i } ) \right] ) \mathrm { ~ b y ~ t h e ~ L a w ~ o f ~ T o t a l ~ V a r i a n c e } } \\ { \displaystyle = n \operatorname { V a r } ( Y ) - \sum _ { i = 1 } ^ { n } \mathbb { E } \left[ \operatorname { V a r } ( Y \mid X _ { - i } ) \right] } \\ { \displaystyle \leq ( n - 1 ) \mathrm { V a r } ( Y ) , } \end{array}
$$

where the last step holds by the Efron–Stein inequality (Boucheron et al., 2013, Theorem 3.1), which tells us that

$$
\operatorname { V a r } ( f ( X _ { 1 } , \dots , X _ { n } ) ) \leq \sum _ { i = 1 } ^ { n } \mathbb { E } \left[ \operatorname { V a r } ( f ( X _ { 1 } , \dots , X _ { n } ) \mid X _ { - i } ) \right] .
$$

On the other hand, for the left-hand side of (14), we have

$$
\sum _ { i = 1 } ^ { n } \sum _ { { S \subseteq [ n ] \backslash \{ i \} } \atop | S | = s } { \mathrm { V a r } ( E _ { S } ) } = \sum _ { { S \subseteq [ n ] \atop | S | = s } } \left( \sum _ { i = 1 } ^ { n } { \mathbb { 1 } _ { i \not \in S } } \right) { \mathrm { V a r } ( E _ { S } ) } = \sum _ { { S \subseteq [ n ] \atop | S | = s } } { \mathrm { V a r } ( E _ { S } ) } \cdot ( n - s ) .
$$

Returning to (14), then, we have shown that

$$
\sum _ { | S | = s } \operatorname { V a r } ( E _ { S } ) \cdot ( n - s ) \leq { \binom { n - 2 } { s - 1 } } \cdot ( n - 1 ) \operatorname { V a r } ( Y ) .
$$

After simplifying, this proves the desired claim.

## B Extension to Hilbert space-valued outputs

In this section, we extend our main stability guarantee to avoid requiring that the algorithm A returns real-valued (i.e., one-dimensional) outputs. In particular, let H be a separable Hilbert space—this includes examples such as $\mathbb { R } ^ { d }$ (for multivariate outputs) or $L _ { 2 } ( \mathbb { R } )$ (i.e., square-integrable functions $f : \mathbb { R } \to \mathbb { R }$ , for function-valued outputs). Let $\mathcal { W } \subseteq \mathcal { H }$ be a convex and closed subset, and define its radius

$$
\mathrm { r a d } _ { \mathcal { H } } ( \mathcal { W } ) = \operatorname* { i n f } _ { \boldsymbol { w } \in \mathcal { W } } \operatorname* { s u p } _ { \boldsymbol { w ^ { \prime } } \in \mathcal { W } } \| \boldsymbol { w } - \boldsymbol { w ^ { \prime } } \| _ { \mathcal { H } } .
$$

We will assume that A returns outputs in W, i.e., we can write the algorithm as a map $\mathcal { A } : \mathcal { Z }  \mathcal { W }$ . Our goal is to examine its stability,

$$
\beta _ { P } ^ { 2 } ( A ) = \mathbb { E } _ { ( Z , Z ^ { \prime } ) \sim P } \left[ \| A ( Z ) - A ( Z ^ { \prime } ) \| _ { \mathcal { H } } ^ { 2 } \right] ,
$$

where now we measure the change in the output using the Hilbert space norm $\| \cdot \| _ { \mathcal { H } }$

The work of Solof et al. (2024b) establishes the stability properties of bagging in the setting of a Hilbert-space-valued output. We now show an analogous result working in our general framework:

Theorem 10 (Extension of Theorem 2 to Hilbert-space-valued output). Let H be a separable Hilbert space and let $\mathcal { W } \subseteq \mathcal { H }$ be a convex and closed subset with $\operatorname { r a d } _ { \mathcal { H } } ( \mathcal { W } ) < \infty$ . For any $\mathcal { A } : \mathcal { Z }  \mathcal { W } _ { : }$ , it holds that

$$
\begin{array} { r } { \beta _ { P } ^ { 2 } ( \mathcal { A } _ { Q } ) \leq \mathrm { r a d } _ { \mathcal { H } } ( \mathcal { W } ) ^ { 2 } \cdot \| T _ { P , Q } \| _ { L _ { \infty } ( \mathcal { H } ) \to L _ { 1 } ( \mathcal { H } ) } . } \end{array}
$$

Essentially, the result of Theorem 10 is a direct extension of the real-valued-case: in particular, we recover the result of Theorem 2 as a special case by taking $\mathcal { H } = \mathbb { R }$ and $\mathcal { W } = [ 0 , 1 ]$ since in this case we have $\operatorname { r a d } _ { \mathcal { H } } ( \mathcal { W } ) = 1 / 2$

In the result above, the operator $T _ { P , Q }$ is defined exactly as before, but its norm is now defined with respect to H-valued functions. Specifically, let $f : \mathcal { Z } \to \mathcal { H }$ be bounded and measurable. We define

$$
[ T _ { P , Q } f ] ( z ) = \int _ { \mathcal { Z } } K _ { P , Q } ( z , z ^ { \prime } ) f ( z ^ { \prime } ) ~ \mathrm { d } \mu ( z ^ { \prime } ) ,
$$

as before, where now the returned function $T _ { P , Q } f$ is also H-valued. Then write

$$
\| T _ { P , Q } \| _ { L _ { \infty } ( \mathcal { H } ) \to L _ { 1 } ( \mathcal { H } ) } = \operatorname* { s u p } \left\{ \| T _ { P , Q } f \| _ { L _ { 1 } ( \mathcal { H } ) } : \| f \| _ { L _ { \infty } ( \mathcal { H } ) } \le 1 \right\} ,
$$

where for any measurable $f : \mathcal { Z } \ \to \ \mathcal { H }$ we define $\begin{array} { r } { \Vert f \Vert _ { L _ { 1 } ( \mathcal { H } ) } \ = \ \int _ { \mathcal { Z } } \Vert f ( z ) \Vert _ { \mathcal { H } } \ \mathrm { d } \mu ( z ) } \end{array}$ , and $\begin{array} { r } { \| f \| _ { L _ { \infty } ( \mathcal { H } ) } = \operatorname* { s u p } _ { z \in \mathcal { Z } } \| f ( z ) \| _ { \mathcal { H } } } \end{array}$ . We show in Proposition 13 that by an extension of Grothendieck’s inequality (Grothendieck, 1953),

$$
\| T _ { P , Q } \| _ { L _ { \infty } ( \mathcal { H } ) \to L _ { 1 } ( \mathcal { H } ) } \le C _ { \mathrm { G } } \| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } ,
$$

where $C _ { \mathrm { G } }$ denotes Grothendieck’s constant, known to satisfy $1 . 6 7 < C _ { \mathrm { G } } < 1 . 7 9$ . In particular, this means that the stability bound shown in Theorem 10 is (up to this mild constant) no worse than the bound for real-valued algorithms, given in Theorem 2.

Proof of Theorem 10. Let $w ^ { \ast } \in \mathcal { W }$ attain the infimum, in $\begin{array} { r } { \operatorname { f } _ { w \in { \mathcal { W } } } \operatorname* { s u p } _ { w ^ { \prime } \in { \mathcal { W } } } \| w - w ^ { \prime } \| _ { { \mathcal { H } } } } \end{array}$ , so that

$$
\operatorname { r a d } _ { \mathcal { H } } ( \mathcal { W } ) = \operatorname* { s u p } _ { w \in \mathcal { W } } \| w - w ^ { * } \| _ { \mathcal { H } } .
$$

Define $f ( z ) = \mathcal { A } ( z ) - w ^ { * }$ . Note that $\mathcal { A } _ { Q } ( z ) = f _ { Q } ( z ) + w ^ { * }$ for all z, by construction.

From this point on the proof is identical to that of Theorem 2, except that we work in the Hilbert space H. We calculate

$$
\begin{array} { r l } & { \beta _ { P } ^ { 2 } ( \mathcal { A } _ { Q } ) = \mathbb { E } _ { P } \left[ \left\| \boldsymbol { A } _ { Q } ( Z _ { 0 } ) - \boldsymbol { A } _ { Q } ( Z _ { 1 } ) \right\| _ { \mathcal { H } } ^ { 2 } \right] } \\ & { \quad \quad \quad = \mathbb { E } _ { P } \left[ \left. f _ { Q } ( Z _ { 0 } ) - f _ { Q } ( Z _ { 1 } ) , f _ { Q } ( Z _ { 0 } ) - f _ { Q } ( Z _ { 1 } ) \right. \right] } \\ & { \quad \quad \quad = \displaystyle \int _ { \mathcal { Z } } \left. f ( z ) , [ T _ { P , Q } f ] ( z ) \right. \mathrm { d } \mu ( z ) \mathrm { b y ~ L e m m a ~ 1 2 ~ ( a p p l i e d ~ w i t h ~ } f = g ) } \\ & { \quad \quad \quad \le \| f \| _ { L _ { \infty } ( \mathcal { H } ) } \| T _ { P , Q } f \| _ { L _ { 1 } ( \mathcal { H } ) } \le \| f \| _ { L _ { \infty } ( \mathcal { H } ) } ^ { 2 } \| T _ { P , Q } \| _ { L _ { \infty } ( \mathcal { H } ) \to L _ { 1 } ( \mathcal { H } ) } } \\ & { \quad \quad \quad \le \mathrm { r a d } _ { \mathcal { H } } ( \mathcal { W } ) ^ { 2 } \cdot \| T _ { P , Q } \| _ { L _ { \infty } ( \mathcal { H } ) \to L _ { 1 } ( \mathcal { H } ) } , } \end{array}
$$

where the last step holds since f takes values in $\{ w - w ^ { * } : w \in \mathcal { W } \}$ and therefore $\| f \| _ { L _ { \infty } ( \mathcal { H } ) } \leq$ $\operatorname { r a d } _ { \mathcal { H } } ( \mathcal { W } )$ , by construction. □

We can also take a relaxation of this result, as for the real-valued case:

Corollary 11. Let ψ be any density with respect to the base measure $\mu ,$ , with $\psi ( z ) > 0$ for all z. Then for any $\mathcal { A } : \mathcal { Z }  \mathcal { W }$ , it holds that

$$
\beta _ { P } ^ { 2 } ( \mathcal { A } _ { Q } ) \leq \mathrm { r a d } _ { \mathcal { H } } ( \mathcal { W } ) ^ { 2 } \cdot \| \tilde { T } _ { P , Q } ^ { \psi } \| _ { L _ { 2 }  L _ { 2 } } ,
$$

where we define the operator $\tilde { T } _ { P , Q } ^ { \psi }$ as

$$
[ \tilde { T } _ { P , Q } ^ { \psi } f ] ( z ) = \int _ { \mathcal { Z } } \frac { K _ { P , Q } ( z , z ^ { \prime } ) } { \sqrt { \psi ( z ) \psi ( z ^ { \prime } ) } } \cdot f ( z ^ { \prime } ) \mathrm { d } \mu ( z ^ { \prime } ) .
$$

Note that, once we relax to the $L _ { 2 }  L _ { 2 }$ norm, there is no price to pay for extending from real-valued to Hilbert-valued functions: the quantity $\Vert \tilde { T } _ { P , Q } ^ { \psi } \Vert _ { L _ { 2 }  L _ { 2 } }$ appearing in the bound is the same norm as for the real-valued case.

Proof of Corollary 11. Exactly as in the proof of Corollary 3, we bound

$$
\| T _ { P , Q } \| _ { L _ { \infty } ( \mathcal { H } ) \to L _ { 1 } ( \mathcal { H } ) } \le \| \tilde { T } _ { P , Q } ^ { \psi } \| _ { L _ { 2 } ( \mathcal { H } ) \to L _ { 2 } ( \mathcal { H } ) } .
$$

And, by properties of the Hilbert norm, we also have

$$
\| \tilde { T } _ { P , Q } ^ { \psi } \| _ { L _ { 2 } ( \mathcal { H } ) \to L _ { 2 } ( \mathcal { H } ) } = \| \tilde { T } _ { P , Q } ^ { \psi } \| _ { L _ { 2 } \to L _ { 2 } } .
$$

Lemma 12 (Extension of Lemma 1 to Hilbert-space-valued output). Let $f , g : \mathcal { Z } \to \mathcal { H }$ be bounded functions. Define their ensembled versions

$$
f _ { Q } ( z ) = \mathbb { E } _ { Z \sim Q ( \cdot | z ) } \left[ f ( Z ) \right] , \quad g _ { Q } ( z ) = \mathbb { E } _ { Z \sim Q ( \cdot | z ) } \left[ g ( Z ) \right] .
$$

Then

$$
\int _ { \mathcal { Z } } \left. f ( z ) , [ T _ { P , Q } g ] ( z ) \right. \mathrm { d } \mu ( z ) = \mathbb { E } _ { P } \left[ \left. f _ { Q } ( Z _ { 0 } ) - f _ { Q } ( Z _ { 1 } ) , g _ { Q } ( Z _ { 0 } ) - g _ { Q } ( Z _ { 1 } ) \right. \right] .
$$

Proof of Lemma 12. The proof is identical to that of Lemma 1, except that we now work with inner products in the separable Hilbert space H. We calculate

$$
\begin{array} { r l } & { \int _ { \mathcal { Z } }  f ( z ) , [ T r _ { 2 } g \mathrm { g } ] ( z )  \mathrm { d } \mu ( z ) } \\ & { = \int _ { \mathcal { Z } }  f ( z ) , \int _ { \mathcal { Z } } K _ { P , \mathcal { Q } } ( z , z ^ { \prime } ) \cdot g ( z ^ { \prime } ) \mathrm { d } \mu ( z ^ { \prime } )  \mathrm { d } \mu ( z ) } \\ & { = \int _ { \mathcal { Z } } \int _ { \mathcal { Z } } \mathbb { E } _ { P } [ ( Q ( z \mid Z _ { 0 } ) - Q ( z \mid Z _ { 1 } ) ) \cdot ( Q ( z ^ { \prime } \mid Z _ { 0 } ) - Q ( z ^ { \prime } \mid Z _ { 1 } ) ) ] \cdot  f ( z ) , g ( z ^ { \prime } )  \mathrm { d } \mu ( z ^ { \prime } ) \mathrm { d } \mu ( z ) } \\ & { = \mathbb { E } _ { P } [ \int _ { \mathcal { Z } } \int _ { \mathcal { Z } } ( Q ( z \mid Z _ { 0 } ) - Q ( z \mid Z _ { 1 } ) ) \cdot ( Q ( z ^ { \prime } \mid Z _ { 0 } ) - Q ( z ^ { \prime } \mid Z _ { 1 } ) ) \cdot  f ( z ) , g ( z ^ { \prime } )  \mathrm { d } \mu ( z ^ { \prime } ) \mathrm { d } \mu ( z ) ] } \\ & { = \mathbb { E } _ { P } [  \int _ { \mathcal { Z } } ( Q ( z \mid Z _ { 0 } ) - Q ( z \mid Z _ { 1 } ) ) f ( z ) \mathrm { d } \mu ( z ) , \int _ { \mathcal { Z } } ( Q ( z ^ { \prime } \mid Z _ { 0 } ) - Q ( z ^ { \prime } \mid Z _ { 1 } ) ) g ( z ^ { \prime } ) \mathrm { d } \mu ( z ^ { \prime } )  ] } \\ &  = \mathbb { E } _ { P } [  \int _ { \mathcal { Z } } ( Q ( z \mid Z _ { 0 } ) - Q ( z \mid Z _ { 1 } ) ) f ( z ) \mathrm { d } \mu ( z ) , \int _  \end{array}
$$

where the second step holds by definition of $K _ { P , Q }$ , the third step holds by Fubini’s theorem (applied in the separable Hilbert space H), and the last step holds by definition of $f _ { Q } , g _ { Q } . \quad \bigsqcup$

The following lemma concerns an extension of the simplest, finite-dimensional form of Grothendieck’s inequality to separable Hilbert spaces; since the proof is self-contained and direct, we provide it for the reader’s convenience.

## Proposition 13. We have

$$
\| T _ { P , Q } \| _ { L _ { \infty } ( \mathcal { H } ) \to L _ { 1 } ( \mathcal { H } ) } \le C _ { \mathrm { G } } \| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } .
$$

Proof. First define functions $F , G : { \mathcal { Z } } \to { \mathcal { H } }$ that take only finitely many values via

$$
F ( z ^ { \prime } ) = \sum _ { j = 1 } ^ { n } u _ { j } \mathbb { 1 } _ { A _ { j } } ( z ^ { \prime } ) , \quad G ( z ) = \sum _ { i = 1 } ^ { n } v _ { i } \mathbb { 1 } _ { B _ { i } } ( z ) ,
$$

where $A _ { 1 } , \ldots , A _ { n }$ are measurable, pairwise disjoint subsets of $\mathcal { Z }$ , as are $B _ { 1 } , \ldots , B _ { n }$ , and both $\mathrm { n a x } _ { j \in [ n ] } \| u _ { j } \| _ { \mathcal { H } } \leq 1$ and ma $\mathsf { r } _ { i \in [ n ] } \| v _ { i } \| _ { \mathcal { H } } \leq 1$ . Thus $\| F \| _ { L _ { \infty } ( \mathcal { H } ) } , \| G \| _ { L _ { \infty } ( \mathcal { H } ) } \leq 1$ . Define $A = ( a _ { i j } ) \in \bar { \mathbb { R } } ^ { n \times n }$ by

$$
a _ { i j } = \int _ { B _ { i } } \int _ { A _ { j } } K _ { P , Q } ( z , z ^ { \prime } ) \ : \mathsf { d } \mu ( z ^ { \prime } ) \ : \mathsf { d } \mu ( z ) .
$$

Now let $s _ { 1 } , \ldots , s _ { n } , t _ { 1 } , \ldots , t _ { n } \in [ - 1 , 1 ]$ , and define simple functions $f , g : { \mathcal { Z } } \to \mathbb { R }$ by

$$
f ( z ^ { \prime } ) = \sum _ { j = 1 } ^ { n } s _ { j } \mathbb { 1 } _ { A _ { j } } ( z ^ { \prime } ) , \quad g ( z ) = \sum _ { i = 1 } ^ { n } t _ { i } \mathbb { 1 } _ { B _ { i } } ( z ) .
$$

Then $\| f \| _ { \infty } , \| g \| _ { \infty } \leq 1$ , so

$$
\displaystyle \left| \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } a _ { i j } s _ { j } t _ { i } \right| = \displaystyle \left| \int _ { \mathcal { Z } } \int _ { \mathcal { Z } } K _ { P , Q } ( z , z ^ { \prime } ) f ( z ^ { \prime } ) g ( z ) \mathrm { d } \mu ( z ^ { \prime } ) \mathrm { d } \mu ( z ) \right| \le \| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } .
$$

It follows by the finite-dimensional Grothendieck inequality (e.g. Pisier, 2012, Theorem 1.1), that

$$
\left| \int _ { \mathcal { Z } } \int _ { \mathcal { Z } } K _ { P , Q } ( z , z ^ { \prime } ) \langle F ( z ^ { \prime } ) , G ( z ) \rangle _ { \mathcal { H } } \mathrm { d } \mu ( z ^ { \prime } ) \mathrm { d } \mu ( z ) \right| = \left| \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } a _ { i j } \langle u _ { j } , v _ { i } \rangle _ { \mathcal { H } } \right| \le C _ { G } \| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } .
$$

Now let $F : \mathcal { Z }  \mathcal { H }$ be measurable with $\| F \| _ { L _ { \infty } ( \mathcal { H } ) } \leq 1$ . Let $( e _ { \ell } ) _ { \ell \in \mathbb { N } }$ denote an orthonormal basis for $\mathcal { H } .$ , so that we may write

$$
F ( z ) = \sum _ { \ell = 1 } ^ { \infty } \lambda _ { \ell } ( z ) e _ { \ell } ,
$$

where $\begin{array} { r } { \operatorname* { s u p } _ { z \in \mathcal { Z } } \sum _ { \ell = 1 } ^ { \infty } \lambda _ { \ell } ^ { 2 } ( z ) \le 1 } \end{array}$ . Now let ${ \mathcal { H } } _ { n } = \operatorname { s p a n } ( e _ { 1 } , \ldots , e _ { n } )$ , let $B _ { \mathcal { H } _ { n } } = \{ x \in \mathcal { H } _ { n } : \| x \| _ { \mathcal { H } } \leq$ 1}, and let $\{ u _ { n , 1 } , \ldots , u _ { n , N _ { n } } \} \subseteq { \mathcal { H } } _ { n }$ denote $\mathrm { ~ a ~ } ( 1 / n ) – \mathrm { n e t }$ for $B _ { \mathcal { H } _ { n } }$ . Thus, given any $x \in B _ { \mathcal { H } _ { n } }$ we can find $r \in [ N _ { n } ]$ such that $\| x - u _ { n , r } \| _ { \mathcal { H } } \leq 1 / n$ . Define

$$
E _ { n , 1 } = \left\{ z \in \mathcal { Z } : \left\| \sum _ { \ell = 1 } ^ { n } \lambda _ { \ell } ( z ) e _ { \ell } - u _ { n , 1 } \right\| _ { \mathcal { H } } \leq \frac { 1 } { n } \right\}
$$

and

$$
E _ { n , r } = \left\{ z \in \mathcal { Z } : \left\| \sum _ { \ell = 1 } ^ { n } \lambda _ { \ell } ( z ) e _ { \ell } - u _ { n , r } \right\| _ { \mathcal { H } } \leq \frac { 1 } { n } \right\} \setminus \bigcup _ { s = 1 } ^ { r - 1 } E _ { n , s }
$$

for $r \in [ N _ { n } ] \setminus \{ 1 \}$ . The sets $( E _ { n , r } ) _ { r \in [ N _ { n } ] }$ form a measurable partition of $\mathcal { Z }$ , because we have $\begin{array} { r } { \operatorname* { s u p } _ { z \in \mathcal { Z } } \big \| \sum _ { \ell = 1 } ^ { n } \lambda _ { \ell } ( z ) e _ { \ell } \big \| _ { \mathcal { H } } ^ { 2 } = \operatorname* { s u p } _ { z \in \mathcal { Z } } \sum _ { \ell = 1 } ^ { n } \lambda _ { \ell } ^ { 2 } ( z ) \leq 1 } \end{array}$ . Now define a measurable function $F _ { n } : { \mathcal { Z } } \to { \mathcal { H } }$ taking only finitely many values by

$$
F _ { n } ( z ) = \sum _ { r = 1 } ^ { N _ { n } } u _ { n , r } \mathbb { 1 } _ { E _ { n , r } } ( z ) .
$$

Then $\begin{array} { r } { \operatorname* { s u p } _ { z \in \mathcal { Z } } \| F _ { n } ( z ) \| _ { \mathcal { H } } \leq 1 } \end{array}$ and for every $z \in { \mathcal { Z } }$

$$
\begin{array} { r l } & { \| F _ { n } ( z ) - F ( z ) \| _ { \mathcal { H } } \leq \left\| \displaystyle \sum _ { r = 1 } ^ { N _ { n } } u _ { n , r } \mathbb { 1 } _ { E _ { n , r } } ( z ) - \displaystyle \sum _ { \ell = 1 } ^ { n } \lambda _ { \ell } ( z ) e _ { \ell } \right\| _ { \mathcal { H } } + \left\| \displaystyle \sum _ { \ell = 1 } ^ { n } \lambda _ { \ell } ( z ) e _ { \ell } - F ( z ) \right\| _ { \mathcal { H } } } \\ & { \qquad \leq \displaystyle \frac { 1 } { n } + \left( \displaystyle \sum _ { \ell = n + 1 } ^ { \infty } \lambda _ { \ell } ^ { 2 } ( z ) \right) ^ { 1 / 2 } \to 0 } \end{array}
$$

as $n \to \infty$

Since $| \langle F _ { n } ( z ^ { \prime } ) , F _ { n } ( z ) \rangle _ { \mathcal { H } } | \le \| F _ { n } ( z ^ { \prime } ) \| _ { \mathcal { H } } \| F _ { n } ( z ) \| _ { \mathcal { H } } \le 1$ , we may apply dominated convergence to deduce that

$$
\begin{array} { r l } { \displaystyle \left| \int _ { \mathcal { Z } } \int _ { \mathcal { Z } } K _ { P , Q } ( z , z ^ { \prime } ) \langle F ( z ^ { \prime } ) , F ( z ) \rangle _ { \mathbb { \boldsymbol { \mathscr { H } } } } \mathrm { d } \mu ( z ^ { \prime } ) \mathrm { d } \mu ( z ) \right| } & { } \\ { = \displaystyle \operatorname* { l i m } _ { n \to \infty } \left| \int _ { \mathcal { Z } } \int _ { \mathcal { Z } } K _ { P , Q } ( z , z ^ { \prime } ) \langle F _ { n } ( z ^ { \prime } ) , F _ { n } ( z ) \rangle _ { \mathbb { \boldsymbol { \mathscr { H } } } } \mathrm { d } \mu ( z ^ { \prime } ) \mathrm { d } \mu ( z ) \right| } & { } \\ { \leq C _ { \mathrm { G } } \| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } . } \end{array}
$$

We conclude that

$$
\begin{array} { r l } & { \| T _ { P , Q } \| _ { L _ { \infty } ( \mathcal H ) \to L _ { 1 } ( \mathcal H ) } = \operatorname* { s u p } \biggr \{ \biggr \{ \underset { Z } { K } _ { P , Q } ( z , z ^ { \prime } ) \langle F ( z ^ { \prime } ) , F ( z ) \rangle _ { \mathcal H } \mathrm d \mu ( z ^ { \prime } ) \mathrm d \mu ( z ) : \| F \| _ { L _ { \infty } ( \mathcal H ) } \le 1 \biggr \} } \\ & { \qquad \le C _ { \mathrm G } \| T _ { P , Q } \| _ { L _ { \infty } \to L _ { 1 } } , } \end{array}
$$

as required.