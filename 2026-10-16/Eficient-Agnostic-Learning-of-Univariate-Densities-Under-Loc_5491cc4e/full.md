# Eficient Agnostic Learning of Univariate Densities Under Local Diferential Privacy

Mohammad Afzali University of British Columbia moafzali@cs.ubc.ca

Hassan Ashtiani   
McMaster University   
Vector Institute   
zokaeiam@mcmaster.ca

McMaster University Canada AI CIFAR Chair, Vector Institute sabatos@mcmaster.ca

October 16, 2026

## Abstract

We propose a general framework for non-interactive locally diferentially private (LDP) learning of structured univariate distributions that can be approximated by piece-wise polynomials. This gives the first sample-eficient LDP learners for several distribution classes, including mixtures of log-concave distributions, such as Gaussian mixture models (GMMs). These learners also provide agnostic guarantees and are computationally eficient. We also prove a lower bound for learning GMMs under LDP, improving on the standard pure-DP packing lower bounds of Bun et al. [BKSW19]. To obtain our general learning framework, we develop an eficient non-interactive LDP algorithm for synthetic data generation over large domains, which may be of independent interest. We then use it together with a deterministic adaptation of the non-private piecewise polynomial learner of Acharya et al. [ADLS17].

## 1 Introduction

Distribution learning is a fundamental problem in statistics and machine learning, where the goal is to estimate an unknown probability distribution P from i.i.d. samples [DL01]. In the realizable setting, P is assumed to belong to a known class F, such as Gaussian distributions, Gaussian mixtures, or log-concave distributions. The learner seeks to return a distribution Pb that approximates P in total variation distance. In practice, the chosen class may only approximate P, leading to model misspecification. Agnostic distribution learning captures this scenario by allowing P to be arbitrary and measuring the learner’s error relative to the best approximation to P within F [Yat85; DL01; LRV16; DKKLMS19; ABHLMP20; DK23]. Our paper is concerned with computationally eficient agnostic distribution learning for structured distributions.

In a typical learning scenario, each data point may be contributed by an individual. Maintaining privacy of those individuals then introduces a further challenge: although distribution learning aims to characterize a population, the samples may reveal sensitive information about the individuals. Diferential privacy provides a rigorous framework for limiting the efect of any individual’s data on the distribution of the learner’s output [DMNS06]. We seek distribution learners that combine a form of this protection with agnostic accuracy guarantees.

In this work, we consider the local diferential privacy (LDP) model [War65; EGS03; KLNRS11; DJW13]. Unlike the central model of privacy, the local model removes the need for individuals/users to entrust their raw observations to a data collector: each user sends a randomized report to the server based on their data point, and the server aggregates these privatized reports. We focus on non-interactive protocols, in which the local randomization protocols are fixed in advance (as opposed to interactive protocols where the randomization protocol can be adjusted based on the reports collected so far). Non-interactive LDP protocols are easier to implement at large scales and have been adopted by major technology companies, including Google [EPK14] and Apple [App17].

It has been observed that a wide range of structured univariate distributions—such as Gaussian mixture models and more generally mixtures of log-concave distributions—can be well approximated by piecewise polynomials [CDSS14]. Without privacy, piecewise polynomials can be learned with sample-and-time-eficient algorithms [CDSS14; ADLS17]. These algorithms are also agnostic, and therefore can be readily used to learn structured distributions. This paradigm has been extended to the central diferential privacy setting by Diakonikolas et al. [DHS15]. However, there is no known algorithm that works in the local model of privacy. This raises the following questions:

Is there a sample eficient learning algorithm for learning structured distributions under LDP? Can this algorithm be made agnostic, time-eficient, and non-interactive?

Summary of our contribution In this work we answer the above questions positively. We give the first general framework for sample-eficient LDP learning of structured univariate distributions that can be approximated by piece-wise polynomials. Our learning algorithm is non-interactive, provides agnostic guarantees and is computationally eficient. Our framework applies to a broad range of distribution classes that admit suitable piece-wise polynomial approximations. As applications, we obtain the first sample-eficient non-interactive LDP learners for bounded-parameter Gaussian mixture models (GMMs) and mixtures of log-concave distributions.

Paper organization We start by formally defining the problem setup in Section 2. We then provide a summary of our main results and contributions in Section 3 and discuss related work in Section 4. Next, in Section 5, we present the preliminaries. Sections 6 and 7 develop private synthetic data generation and our generic agnostic LDP learner, respectively. Section 8 applies the framework to Gaussian and log-concave mixtures, and Section G establishes sample complexity lower bounds. Omitted proofs and further discussion on related work appear in the appendix.

## 2 Locally private agnostic learning of distributions

In this section, we introduce formal notions required to introduce the problem setup. A locally private mechanism/algorithm can access data points only through local randomizers.

Definition 2.1 (Local randomizer). Let $x , y$ be two domains, and $\varepsilon > 0$ be a parameter. A randomized function $\mathcal { M } : \mathcal { X }  \mathcal { Y }$ is called an ε-local randomizer if for all $x , x ^ { \prime } \in { \mathcal { X } }$ , and any measurable subset $Y \subseteq \mathcal { V }$ , it holds that $\mathbb { P } [ \mathcal { M } ( \boldsymbol { x } ) \in Y ] \leq e ^ { \varepsilon } \mathbb { P } [ \mathcal { M } ( \boldsymbol { x } ^ { \prime } ) \in Y ]$ , where the probabilities are over the randomness of M.

Definition 2.2 (Non-interactive local diferential privacy). Let $x , y$ be two domains, $\varepsilon > 0$ , and $n \in \mathbb N$ . An algorithm $\mathcal { A } : \mathcal { X } ^ { n }  \mathcal { Y }$ is non-interactive ε-locally diferentially private $\left( \varepsilon - L D P \right)$ if there exist ε-local randomizers $\mathcal { M } _ { i } : \mathcal { X }  \mathcal { Z } _ { i } ~ f o r ~ i \in [ n ]$ , and a (possibly randomized) aggregation function $\begin{array} { r } { \mathcal { T } : \prod _ { i = 1 } ^ { n } \mathcal { Z } _ { i }  \mathcal { Y } } \end{array}$ , such that $A ( x _ { 1 } , \ldots , x _ { n } ) = { \mathcal { T } } ( { \mathcal { M } } _ { 1 } ( x _ { 1 } ) , \ldots , { \mathcal { M } } _ { n } ( x _ { n } ) )$ . Each of the local randomizers and the aggregation function have independent randomness.

We can now state the learning problem. Let $\mathcal { F }$ be a known class of probability distributions over $\mathbb { R } .$ and let $P$ be an arbitrary unknown distribution over R. There are n users, each holding one sample, with the samples drawn independently from P. Given a privacy parameter $\varepsilon \in ( 0 , 1 )$ , an accuracy parameter $\alpha \in ( 0 , 1 )$ , and a failure probability $\beta \in ( 0 , 1 )$ , the goal is to design a non-interactive ε-LDP algorithm that outputs a probability distribution $\widehat { P }$ satisfying

$$
\mathrm { d _ { T V } } ( \widehat { P } , P ) \leq C \operatorname* { i n f } _ { Q \in \mathcal { F } } \mathrm { d _ { T V } } ( Q , P ) + \alpha\tag{2.1}
$$

with probability at least $1 - \beta$ over the samples and the randomness of the algorithm, where $C \geq 1$ is a universal constant. We call such an algorithm a $( C , \alpha , \beta )$ -agnostic learner for $\mathcal { F }$ . We allow $\widehat { P }$ to lie outside $\mathcal { F } ;$ that is, the learner may be non-proper. In the realizable case, where $P \in { \mathcal { F } }$ , the guarantee reduces to $\mathrm { d } _ { \mathrm { T V } } ( \widehat { P } , P ) \leq \alpha$ . Note that the privacy requirement should hold even if the dataset is not generated i.i.d. by a distribution $P .$

In general, unbounded distribution classes (e.g. unbounded univariate Gaussians) cannot be learned with a uniform sample complexity upper bound under LDP due to packing lower bounds in pure diferential privacy Bun et al. [BKSW19]. Thus, to achieve any meaningful result, we restrict the learning task to classes admitting some “boundedness” assumptions on the underlying class parameters (e.g. bounds on mean and variance). To measure the density estimation error introduced by such boundedness assumptions, we introduce the following definition.

Definition 2.3 (Discretization error). Let $R , \delta \ > \ 0$ , and let $\mathcal { F }$ be a class of probability distributions over R. The $( R , \delta )$ -discretization error of class $\mathcal { F }$ is defined as $\mathrm { D i s c - E r r } _ { R , { \delta } } ( { \mathcal F } ) \ : =$ $\begin{array} { r } { \operatorname* { s u p } _ { F \in \mathcal { F } } \| \operatorname { D i s c } _ { R , \delta } ( F ) - F \| _ { 1 } } \end{array}$ , where Disc $^ { \mathbf { \delta } } R , \delta$ converts a density into a histogram using a grid of step size δ on $[ - R , R ]$ and two additional bins for the tails (see Definition 6.2 below).

Next, we define the class of piece-wise polynomial distributions, which plays a key role as a proxy for eficiently learning structured distributions.

Definition 2.4 (Piecewise polynomials). For integers $s \geq 1$ and $d \geq 0$ , let $\mathcal { P } _ { s , d }$ be the set $o f$ nonnegative integrable functions $p : \mathbb { R } \to \mathbb { R } _ { \geq 0 }$ for which R can be partitioned into at most s intervals $\{ I _ { 1 } , \cdots , I _ { k } \}$ (where $k \leq s )$ , such that for each interval $I _ { i } ,$ there exists a real polynomial $q _ { i } : \mathbb { R } $ R of degree at most d satisfying $p ( x ) = q _ { i } ( x )$ for all (interior points) $x \in I _ { i }$

The next definition measures how well a class can be approximated by piecewise polynomials.

Definition 2.5 (Piece-wise polynomial approximation error). Let $s \geq 1$ and $d \geq 0$ be integers, and let $\mathcal { F }$ be a class of probability distributions over R. The (s, d)-piece-wise polynomial approximation error of class $\mathcal { F }$ is $\begin{array} { r } { \mathrm { P o l y - A p p r o x } _ { s , d } ( \mathcal { F } ) : = \operatorname* { s u p } _ { F \in \mathcal { F } } \operatorname* { i n f } _ { P \in \mathcal { P } _ { s , d } } \| P - F \| _ { 1 } } \end{array}$

## 3 Main results and contributions

Our main contribution is the introduction of a generic framework for non-interactive LDP learning of structured univariate distributions that can be approximated by piece-wise polynomials.

Theorem 3.1 (Informal version of 7.1). Let F be a class of univariate probability distributions. $I f \operatorname { P o l y - A p p r o x } _ { s , d } ( { \mathcal { F } } ) + \operatorname { D i s c - E r r } _ { R , \delta } ( { \mathcal { F } } ) \leq \Theta ( \alpha )$ , then, F admits a computationally eficient, noninteractive $\varepsilon \ – L D P \ ( 1 0 , \Theta ( \alpha ) , 0 . 1 )$ -agnostic learner using $\begin{array} { r } { n = \widetilde { O } \left( \frac { s ^ { 2 } ( d + 1 ) ^ { 2 } } { \varepsilon ^ { 2 } \alpha ^ { 2 } } \log ^ { 3 } { \left( \frac { R } { \delta } \right) } \right) } \end{array}$ samples.

This is the first algorithm for this setting whose sample complexity is polynomial in all relevant parameters. Moreover, our learner provides agnostic guarantees and is computationally eficient.

Our learning framework builds on the eficient non-private piece-wise polynomial learner of Acharya et al. [ADLS17]. A deterministic adaptation of their algorithm—as also observed in [DHS15] in the context of central privacy—can work with a synthetic dataset whose empirical distribution is close to the true distribution in Kolmogorov distance. The key challenge is to generate such a synthetic dataset privately, while ensuring both sample and computational eficiency. We show that this task is possible, by presenting the following eficient, non-interactive LDP algorithm for synthetic data generation, which may be of independent interest.

Theorem 3.2 (Informal version of 6.3). Let P be a univariate probability distribution that satisfies Disc-Err $R , \delta \bigl ( \bigl \{ P \bigr \} \bigr ) \leq \Theta ( \alpha )$ . Then there exists a computationally eficient, non-interactive ${ \varepsilon } { - } L D P$ algorithm that, given $\begin{array} { r } { n = { \widetilde { O } } \left( \frac { 1 } { \varepsilon ^ { 2 } \alpha ^ { 2 } } \log ^ { 3 } \left( \frac { R } { \delta } \right) \right) } \end{array}$ samples from P, generates a synthetic dataset of size $O ( 1 / \alpha )$ whose empirical distribution approximates P to Kolmogorov accuracy $\Theta ( \alpha )$ with probability at least 0.9.

As discussed earlier, time eficiency is measured relative to the optimal sample complexity, which requires eficiency with respect to the underlying domain size. Although existing LDP mechanisms for answering prefix queries are sample eficient, they require processing the entire domain, resulting in running time polynomial in the domain size [CKS19; ENU20]. To overcome this challenge, we provide a lazy adaptation of the mechanism proposed by Cormode et al. [CKS19], obtaining an algorithm that answers range queries in time polylogarithmic in the domain size. We then use this mechanism as a subroutine to eficiently generate private synthetic data.

Our generic learning framework applies to a broad range of structured distribution classes. As applications, we provide the first sample-eficient non-interactive LDP learners—which also provide agnostic guarantees and are computationally eficient—for the following classes:

Theorem 3.3 (Log-concave mixtures, informal version of 8.5). Let $\varepsilon , \alpha \in ( 0 , 1 )$ , let $m \geq 1$ be an integer, and let $R \geq 0$ and $0 < \sigma _ { \mathrm { m i n } } \le \sigma _ { \mathrm { m a x } } < \infty$ . The class of mixtures of m logconcave distributions, each with mean $| \mu | \leq R$ and standard deviation $\sigma \in [ \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ]$ , admits a computationally eficient, non-interactive $\varepsilon { - } L D P \left( 1 0 , \alpha , 0 . 1 \right)$ -agnostic learner with sample complexity $\begin{array} { r } { n = \widetilde { O } \left( \frac { m ^ { 2 } \log ^ { 3 } D } { \varepsilon ^ { 2 } \alpha ^ { 3 } } \right) } \end{array}$ and total running time $\tilde { O } \left( \frac { m ^ { 3 } \log ^ { 4 } D } { \varepsilon ^ { 2 } \alpha ^ { 9 / 2 } } \right)$ , where $\begin{array} { r } { D = \frac { 2 ( R + \sigma _ { \mathrm { m a x } } ) } { \sigma _ { \mathrm { m i n } } } } \end{array}$

Theorem 3.4 (Gaussian mixtures, informal version of F.2). Under the same parameter assumptions as in Theorem 3.3, the class of mixtures of m Gaussians admits a computationally eficient, noninteractive ε-LDP $( 1 0 , \alpha , 0 . 1 )$ -agnostic learner with sample complexity $\begin{array} { r } { \bar { n } = \tilde { O } \left( \frac { m ^ { 2 } \log ^ { 3 } \bar { D } } { \varepsilon ^ { 2 } \alpha ^ { 2 } } \right) } \end{array}$ and total running time $\tilde { O } \left( \textstyle { \frac { m ^ { 3 } \log ^ { 4 } D } { \varepsilon ^ { 2 } \alpha ^ { 3 } } } \right)$ .<sup>1</sup>

We further prove a lower bound showing that restrictions on class parameters, such as bounded mean and variance, are necessary in general for a uniform finite sample complexity bound under LDP. The lower bound provided in the following theorem for non-interactive LDP learning of GMMs holds even in the realizable setting with non-proper outputs. Since every Gaussian distribution is log-concave, this lower bound also applies to mixtures of log-concave distributions.

Theorem 3.5 (Gaussian mixture lower bound, informal version of G.1). Let $m \geq 2 , \varepsilon \in ( 0 , 1 ]$ , and $\alpha > 0$ be suficiently small. Under the same bounds on component means and standard deviations as in Theorem 3.3, if the parameter range $\begin{array} { r } { \Lambda : = \frac { R } { \sigma _ { \operatorname* { m i n } } } + \log \left( \frac { \sigma _ { \operatorname* { m a x } } } { \sigma _ { \operatorname* { m i n } } } \right) \geq \Omega ( m ) } \end{array}$ is suficiently large, then any non-interactive ${ \varepsilon } { - } L D P$ algorithm that learns mixtures of m Gaussians to total variation accuracy α with probability at least 0.9 requires $\begin{array} { r } { n = \Omega \left( \frac { m ^ { 2 } } { \varepsilon ^ { 2 } \alpha ^ { 2 } } \log { \left( e ^ { } + \frac { \Lambda } { m } \right) } \right) } \end{array}$ samples.

This matches the dependence of our GMM sample complexity upper bound on $m , \varepsilon ,$ and $\alpha ,$ up to logarithmic factors. In particular, this improves over the standard packing lower bounds for pure diferential privacy, which apply in both the central and local models [BKSW19]. Indeed, packing for mixtures of m components usually has a size exponential in $m ,$ which then translate to lower bounds of $\Omega ( m )$ . We show a stronger lower bound of $\Omega ( m ^ { 2 } )$ by reducing the problem of LDP estimation of sparse discrete distributions studied by Acharya et al. [AKLS20] to the problem of LDP learning of Gaussian mixtures.

Our framework also improves prior LDP learning results for specialized distribution classes:

• Univariate Gaussians. Theorem 3.4 gives an agnostic learner for univariate Gaussians using $\begin{array} { r } { \tilde { O } \left( \frac { \log ^ { 3 } \left( D \right) } { \varepsilon ^ { 2 } \alpha ^ { 2 } } \right) } \end{array}$ samples. This obtains the same sample complexity rates as for realizable Gaussian mean estimation [GRS19; JKMW19] up to logarithmic factors, while providing agnostic guarantees for density estimation in total variation distance.

• (Sparse) discrete distributions. For learning discrete distributions over a domain of size D, Acharya et al. [ASZ19] achieves the optimal sample complexity of $\Theta \left( \frac { D ^ { 2 } } { \varepsilon ^ { 2 } \alpha ^ { 2 } } \right)$ . When the unknown distribution has support of size at most s $( { \mathrm { i . e . } }$ , is s-sparse), Acharya et al. [AKLS20] gives sharper sample complexity bound of Θ $\left( \frac { s ^ { 2 } \operatorname* { m a x } \{ \log ( D / s ) , 1 \} } { \varepsilon ^ { 2 } \alpha ^ { 2 } } \right)$ . Both classes can be embedded into classes of piecewise constant densities. Our framework provides agnostic guarantees for these classes while losing only logarithmic factors in sample complexity.

All our density learners above are non-proper. In Section H, we show how to convert these into proper learners with the same asymptotic sample complexity, at the cost of a larger agnostic approximation factor and potentially ineficient algorithm. In particular, we show that our result achieves an exponential improvement in sample complexity over generic LDP hypothesis selection methods for proper learning of structured distributions.

## 4 Related Work

A closely related result in private learning of structured distributions is provided by Diakonikolas et al. [DHS15]. They study this problem in the central diferential privacy model and present a connection between private Kolmogorov approximation and structured distribution learning. They state a formal piece-wise constant (histogram) learner and sketch the reduction, explaining that it can be extended to piece-wise polynomials. They separately develop and evaluate a heuristic for private Kolmogorov approximation. However, privatization techniques for Kolmogorov approximation difer substantially between the central and local models. In particular, the central DP guarantee used in their theoretical reduction relies on the approximately diferentially private threshold-release algorithm of Bun et al. [BNSV15], which builds on their private algorithm for the interior point problem.

A closely related line of work studies locally private hypothesis selection (LPHS). Given a finite class $\mathcal { F }$ of probability distributions and i.i.d. samples from an arbitrary distribution $P _ { \mathrm { : } }$ , the goal is to select $\widehat { P } \in \mathcal { F }$ whose total variation distance to $P$ is within a constant factor of $\mathrm { i n f } _ { Q \in { \mathcal { F } } } \mathrm { d _ { T V } } ( P , Q )$ up to an additive error α, while satisfying ε-local diferential privacy. For this problem, Gopi et al. [GKKNWZ20] give a non-interactive algorithm with sample complexity $\widetilde { O } ( | \mathcal { F } | ^ { 2 } / ( \varepsilon ^ { 2 } \alpha ^ { 2 } ) )$ , which was later improved by Kamath et al. [KPRW25] to ${ \tilde { O } } ( | \mathcal { F } | ^ { 3 / 2 } / ( \varepsilon ^ { 2 } \alpha ^ { \hat { 2 } } ) )$ . A worst-case lower bound of $\Omega ( | \mathcal { F } | / ( \varepsilon ^ { 2 } \alpha ^ { 2 } ) )$ holds even for interactive protocols [GKKNWZ20; PAA24]. Thus, without exploiting additional structure in $\mathcal { F }$ , a generic hypothesis-selection method cannot achieve sublinear dependence on $| \mathcal F |$ . The running time of these algorithms also scales polynomially with $| \mathcal F |$ , as they typically need to scan the entire class.

These hypothesis selection methods also yield agnostic learners for some infinite distribution classes by applying hypothesis selection to a finite total-variation cover. However, such covers can be exponentially large in terms of class parameters, making this approach ineficient in both sample complexity and running time. For example, discretizing the parameters of a single Gaussian gives polynomially many choices in domain size $D ;$ enumerating m components and discretizing their weights yields a GMM cover exponential in m. Applying hypothesis selection inherits this dependence. On the other hand, our framework exploits distributional structure to obtain sample complexity and running time polynomial in m and polylogarithmic in $D$ . See Section H for further discussion of LPHS and Section A for more on related work.

## 5 Preliminaries

For $N \in \mathbb { N } .$ , let $[ N ] : = \{ 1 , 2 , \dots , N \}$ , and $[ N ] _ { 0 } : = \{ 0 , 1 , 2 , \ldots , N \}$ . The total variation distance between two probability distributions $P$ and Q on X is defined as $\operatorname { d _ { T V } } ( P , Q ) : = \operatorname* { s u p } _ { A } | P ( A ) - Q ( A ) | =$ ${ \frac { 1 } { 2 } } \parallel P - Q \parallel _ { 1 }$ , where A ranges over measurable sets and $\| \cdot \| _ { 1 }$ denotes the total variation norm. For densities $f$ and $^ { g , }$ this equals $\begin{array} { r } { { \frac { 1 } { 2 } } \int _ { \mathcal { X } } | f ( x ) - g ( x ) | d x } \end{array}$ . Let $\log _ { + } ( x ) : = \operatorname* { m a x } \{ 1 , \log x \}$ . For a probability distribution P over R, we use the notation $F _ { P }$ to denote its CDF. For two probability distributions

P, Q over R, their Kolmogorov distance is defined as $\begin{array} { r } { \| F _ { P } - F _ { Q } \| _ { \infty } = \operatorname* { s u p } _ { x \in \mathbb { R } } | F _ { P } ( x ) - F _ { Q } ( x ) | . } \end{array}$

Throughout the paper, we measure running time in a unit-cost RAM model with exact real arithmetic and integer words large enough to store all indices and counters used by our algorithms. The word size is logarithmic in the relevant input and algorithmic parameters, including their inverses when appropriate. Arithmetic operations, comparisons, memory access, shifts, bitwise operations, word-level parity, evaluation of the elementary functions used below, and sampling from the Bernoulli and uniform distributions used by our algorithms all take constant time.

In the following two subsections, we first introduce LDP primitives for answering range queries and then present the non-private piece-wise polynomial learner used in our downstream learning algorithm.

## 5.1 Locally private prefix frequencies

We present tools for answering range-frequency queries over a finite ordered domain under LDP. These queries ask for the fraction of observations lying in a specified interval. We focus on prefix queries, which provide the cumulative frequency information used to generate our private synthetic dataset described in Section 6. General interval frequencies can be obtained by taking the diference of two prefix frequencies.

A key ingredient in answering prefix queries eficiently in both samples and time is the dyadic decomposition, which partitions each prefix into logarithmically many disjoint intervals from a fixed binary hierarchy. This hierarchy underlies the binary-tree mechanisms for private counting [DNPR10; CSS11] and their local analogues discussed by Cormode et al. [CKS19] and Edmonds et al. [ENU20]. Following the approach to locally private range queries of Cormode et al. [CKS19], we combine this hierarchy with the Hadamard randomized response to estimate interval frequencies. We first define the dyadic hierarchy and then introduce the Hadamard characters used in this construction.

Definition 5.1 (Dyadic tree representation). Let [N] be an ordered domain, where $N = 2 ^ { L }$ . Then, for every depth $\ell \in [ L ] _ { 0 }$ and every $v \in [ 2 ^ { \ell } - 1 ] _ { 0 }$ , the dyadic interval is defined as

$$
I _ { \ell , v } : = \left\{ v 2 ^ { L - \ell } + 1 , \ldots , ( v + 1 ) 2 ^ { L - \ell } \right\} .
$$

The interval $I _ { 0 , 0 } = [ N ]$ is the root of the tree, and the intervals $I _ { L , v } = \{ v + 1 \}$ are its leaves. For $x \in [ N ]$ , let $\begin{array} { r } { v _ { \ell } ( x ) : = \left\lfloor \frac { x - 1 } { 2 ^ { L - \ell } } \right\rfloor } \end{array}$ denote the index of the unique depth-ℓ node containing x.

If the domain size is not a power of two, we pad it to the next power of two with unused elements.

![](images/1f899e8c1a44cb8910c3827c98663028ec0904830e383ab82cda79d12d5bd859.jpg)  
Figure 1: For $N = 4$ , the blue nodes $\mathcal { C } ( 3 ) = \{ I _ { 1 , 0 } , I _ { 2 , 2 } \}$ partition the prefix [1, 3], using at most one node per depth.

The following lemma shows that any proper prefix [j] of [N] can be partitioned into at most $\log _ { 2 } N$ dyadic intervals, with at most one interval at each depth, and that this partition can be constructed in ${ \cal O } ( \log N )$ time. Summing the frequencies of these disjoint intervals therefore gives the prefix frequency. The full prefix [N] has frequency one and requires no estimation. See Figure 1 for an illustration. For completeness, C provides the decomposition algorithm and the proof.

Lemma 5.2 (Canonical dyadic decomposition of a prefix). Let $L \in \mathbb { N } , N = 2 ^ { L }$ , and let $j \in [ N - 1 ]$ Then, Algorithm 3, upon receiving j, returns a collection $\mathcal { C } ( j )$ of dyadic nodes satisfying:

(1) the nodes in $\mathcal { C } ( j )$ are pairwise disjoint and $\begin{array} { r } { [ j ] = \bigcup _ { ( \ell , v ) \in \mathcal { C } ( j ) } I _ { \ell , v } } \end{array}$ (2) for every $\ell \in [ L ]$ , the collection $\mathcal { C } ( j )$ contains at most one node at depth $\ell ;$ therefore, $| { \mathcal { C } } ( j ) | \leq L$ (3) construction of ${ \mathfrak { C } } ( j )$ takes $O ( L )$ time.

To estimate dyadic interval frequencies, we use the Hadamard randomized-response construction of Cormode et al. [CKS19] (see Algorithm 4 in C), which uses Hadamard characters:

Definition 5.3 (Hadamard characters). $F o r \ \ell \in \mathbb { N }$ and $v \in [ 2 ^ { \ell } - 1 ] _ { 0 }$ , the Hadamard character indexed by v is the function $\operatorname { H a d } _ { \ell } ( v , \cdot ) : [ 2 ^ { \ell } - 1 ] _ { 0 } \to \{ - 1 , + 1 \}$ defined, for every $u \in [ 2 ^ { \ell } - 1 ] _ { 0 } ,$ , by $\mathrm { H a d } _ { \ell } ( v , u ) : = ( - 1 ) ^ { \langle v , u \rangle _ { 2 } }$ , where $\langle v , u \rangle _ { 2 }$ is the binary inner product of the ℓ-bit representations of v and u. Each $\operatorname { H a d } _ { \ell } ( v , u )$ can be computed in $O ( 1 )$ time using a bitwise AND and a parity operation.

At a high level, each user helps estimate prefix frequencies by contributing information at a randomly selected depth $\ell _ { u }$ of the dyadic tree. To do so, user u randomly chooses a node with index $r _ { u }$ at depth $\ell _ { u }$ and computes Ha $\mathrm { l } _ { \ell _ { u } } ( v _ { u } , r _ { u } )$ , where $v _ { u }$ is the index of their true node, namely the node containing $X _ { u }$ at the same depth. They then apply randomized response to this sign and report it together with $\ell _ { u }$ and $r _ { u }$ . We show in Section 6.1 how the server can eficiently reuse each user’s report to answer diferent prefix queries using the orthogonality of Hadamard characters.

## 5.2 A deterministic non-private piecewise polynomial learner

Another key primitive is the non-private density estimator of Acharya et al. [ADLS17]. We use a deterministic adaptation of their algorithm. Their original estimator takes i.i.d. samples from an unknown distribution. As we formalize in Appendix B, the estimator can instead operate on synthetic data whose empirical distribution satisfies a suitable Kolmogorov-distance bound. We define the class of piecewise polynomial functions and state the deterministic guarantee.

Proposition 5.4 (Deterministic post-sampling General–Merging [ADLS17]). There exists a universal integer $C > 2$ such that the following holds. Let $I \subset \mathbb { R }$ be a known finite interval, and let $Z = \{ z _ { 1 } , . . . , z _ { M } \} \in I ^ { M }$ be an arbitrary multiset, where $M \geq 1$ , with empirical probability measure $\begin{array} { r } { H = \frac { 1 } { M } \sum _ { i \in [ M ] } \delta _ { z _ { i } } } \end{array}$ . Let $s \geq 1$ and $d \geq 0$ be integers, and let $\eta > 0$ . There exists a deterministic algorithm which, given $I , Z , s , d ,$ and η, returns a nonnegative function $\widehat { P } \in \mathcal { P } _ { C s , d }$ supported on I with the following guarantee:

For every probability distribution P<sup>⋆</sup> with a density supported on I satisfying $\begin{array} { r } { \| F _ { H } - F _ { P ^ { \star } } \| _ { \infty } \leq \frac { \eta } { 2 ( C + 1 ) ( d + 1 ) s } , } \end{array}$ the output of the algorithm satisfies $\| \widehat { P } - P ^ { \star } \| _ { 1 } \leq 4 ( \mathrm { O P T } + \eta )$ , where $\begin{array} { r } { \mathrm { O P T } : = \operatorname* { i n f } _ { P \in \mathcal { P } _ { s , d } } \| P - P ^ { \star } \| _ { 1 } } \end{array}$ Moreover, the running time is $\begin{array} { r } { \widetilde { O } \left( M ( d + 1 ) ^ { \omega + 2 } \log _ { + } ^ { 3 } \left( \frac { s } { \eta } \right) \left( 1 + \log _ { + } \left( \frac { M } { s } \right) \right) \right) } \end{array}$ , where ω is the matrix multiplication exponent, and $\tilde { O }$ hides only logarithmic factors in M and $d + 1$

```latex
Algorithm 1 Hierarchical LDP PrefixQuery
Input: Privacy parameter $\varepsilon \in ( 0 , 1 )$ , domain size $D \in \mathbb { N }$ with $D \geq 2 ,$ a query index $j \in [ D ] _ { 0 }$ , and a
transcript ${ \mathrm { T r } } = ( T _ { 1 } , \ldots . . . , T _ { n } )$ , where $n \geq 1$
Output: An estimate $\widehat { Z } _ { j } \in [ 0 , 1 ]$ of the prefix frequency $\begin{array} { r } { Z _ { j } ( X ) = \frac { 1 } { n } \sum _ { u \in [ n ] } { \mathbf 1 } \{ X _ { u } \leq j \} } \end{array}$
1: If j = 0 return $\widehat { Z } _ { j } \gets 0 .$
2: If $j = D$ return $\hat { Z } _ { j } \gets 1 .$
3: Set $L \gets \lceil \log _ { 2 } D \rceil , N \gets 2 ^ { L }$ , and $\textstyle \theta _ { \varepsilon } \gets { \frac { e ^ { \varepsilon } - 1 } { e ^ { \varepsilon } + 1 } }$
4: Apply Algorithm 3 (from Lemma 5.2) to the prefix index j in the dyadic tree over [N]. Represent
the returned nodes as $\mathcal { C } ( j ) = \{ ( \ell _ { i } , v _ { i } ) : i \in [ k ] \}$ , where $[ j ] = \bigcup _ { ( \ell _ { i } , v _ { i } ) \in \mathcal { C } ( j ) } I _ { \ell _ { i } , v _ { i } } , k \le L$ , and the
depths $\ell _ { i }$ are distinct. Store the nodes of $\mathcal { C } ( j )$ indexed by depth.
5: For $u \in [ n ]$ , set $\begin{array} { r l } { Z _ { u } \gets \frac { L } { \theta _ { \varepsilon } } } & { { } \sum _ { \mathbf { \lambda } } \mathrm { ~ \sum ~ } _ { u } \mathrm { H a d } _ { \ell _ { i } } ( v _ { i } , r _ { u } ) } \end{array}$
(ℓ<sub>i</sub>,v<sub>i</sub>)∈C(j)
$\ell _ { i } \overset { \cdot } { = } \ell _ { u }$
6: return $\begin{array} { r } { \widehat { Z } _ { j } \gets \operatorname* { m i n } \bigg \{ 1 , \operatorname* { m a x } \bigg \{ 0 , \frac { 1 } { n } \sum _ { u \in [ n ] } Z _ { u } \bigg \} \bigg \} . } \end{array}$ ▷ Average the contributions and clip to [0, 1].
```

## 6 Eficient locally private synthetic data generation

In this section, we develop a sample- and time-eficient non-interactive LDP algorithm for generating synthetic data with the Kolmogorov approximation guarantee required by the downstream nonprivate piece-wise polynomial learner. We do this in two steps. First, in Section 6.1, we develop an eficient LDP oracle for answering prefix queries on a fixed dataset. Then, in Section 6.2, we use this oracle to generate a synthetic dataset whose empirical distribution approximates the discretized input distribution in Kolmogorov distance.

## 6.1 Lazy dyadic prefix mechanism

In this section, we introduce PrefixQuery (Algorithm 1), a non-interactive ε-LDP algorithm for estimating prefix frequencies. Given a discrete domain [D], and n users holding values $X _ { 1 } , \dots , X _ { n } \in$ [D], the algorithm estimates the fraction of observations in [j] for any queried index $j \in [ D ]$ . Each query takes $O ( n + \log D )$ time, and $\begin{array} { r } { n = O \left( \frac { \log ^ { 2 } D } { \varepsilon ^ { 2 } \alpha ^ { 2 } } \log \frac { D } { \beta } \right) } \end{array}$ users sufice to achieve an additive error of at most α simultaneously over all prefixes, with probability at least $1 - \beta$ over the randomness of the LDP mechanism.

The algorithm uses the same local randomizer as the binary hierarchical mechanism of Cormode et al. [CKS19], as presented in Algorithm 4. However, while the server-side reconstruction described by Cormode et al. [CKS19] materializes the full hierarchy, requiring $O ( n + D \log D )$ time, our server-side implementation evaluates requested prefixes lazily, without constructing the full hierarchy. This gives a running time of $O ( q ( n + \log D ) )$ for answering q prefix queries. This is especially useful when the number of queries q is logarithmic in the domain size D. As we will see in the following sections, this is indeed the case for the downstream non-private algorithm.

The following theorem formalizes the utility and privacy guarantees of PrefixQuery.

Theorem 6.1 (Non-interactive LDP prefix oracle). Let $\varepsilon , \alpha , \beta \in ( 0 , 1 )$ , and $D \in \mathbb { N }$ with $D \geq 2$ be parameters, and let $X = ( X _ { 1 } , \cdot \cdot \cdot , X _ { n } ) \in [ D ] ^ { n }$ be any fixed dataset. Define its prefix frequencies $b y$

$$
Z _ { j } ( X ) : = { \frac { 1 } { n } } \sum _ { u \in [ n ] } { \mathbf { 1 } } \{ X _ { u } \leq j \} , \qquad j \in [ D ] _ { 0 } .
$$

Let M be the local randomizer in Algorithm $^ { 4 , }$ and set the users’ transcripts to be $\operatorname { T r } _ { \mathcal { M } } ( X ) : =$ $( \mathcal { M } ( X _ { 1 } , \varepsilon , D ) , \ldots , \mathcal { M } ( X _ { n } , \varepsilon , D ) )$ . For an input query $j \in [ D ] _ { 0 }$ , let $\widehat { Z } _ { j } = \mathsf { P r e f i x Q u e r y } ( \varepsilon , D , j , \mathrm { T r } _ { \mathcal { M } } ( X ) )$ be the output of Algorithm 1. Then,

(1) The mechanism PrefixQuery(ε, $D , j , \operatorname { T r } _ { \mathcal { M } } ( . ) )$ is non-interactive ${ \varepsilon } { - } L D P$ for every $j \in [ D ] _ { 0 }$

(2) $\begin{array} { r } { I f n \ge \frac { 2 8 8 ( \log D ) ^ { 2 } } { \varepsilon ^ { 2 } \alpha ^ { 2 } } } \end{array}$ log $\left( { \frac { 2 D } { \beta } } \right)$ , then with probability at least $1 - \beta$ over the randomness of M, it holds that $\left| \widehat { Z } _ { j } - Z _ { j } ( X ) \right| \leq \alpha$ for every $j \in [ D ] _ { 0 }$ simultaneously.

(3) Generating $\operatorname { T r } _ { \mathcal { M } } ( X )$ takes $O ( n )$ total run-time. Each call to PrefixQuery takes $O ( n + \log D )$ server side run-time.

At a high level, PrefixQuery decomposes the queried prefix $[ j ]$ into the disjoint dyadic nodes in $\mathcal { C } ( j )$ and estimates their total frequency. To estimate the frequency of a node $( \ell _ { i } , v _ { i } ) \in \mathcal { C } ( j )$ , the server takes user reports with $\ell _ { u } = \ell _ { i }$ and multiplies the reported sign $Y _ { u }$ by H $\{ \mathrm { a d } _ { \ell _ { i } } ( v _ { i } , r _ { u } )$ . This works because Hadamard orthogonality (Claim C.1) gives

$$
\begin{array} { r } { \mathbb { E } _ { r \sim \mathrm { U n i f } ( [ 2 \ell _ { i } - 1 ] _ { 0 } ) } \left[ \mathrm { H a d } _ { \ell _ { i } } ( v _ { \ell _ { i } } ( X _ { u } ) , r ) \mathrm { H a d } _ { \ell _ { i } } ( v _ { i } , r ) \right] = \mathbf { 1 } \{ X _ { u } \in I _ { \ell _ { i } , v _ { i } } \} . } \end{array}
$$

Here, $v _ { \ell _ { i } } ( X _ { u } )$ is the index of the node containing $X _ { u }$ at depth $\ell _ { i }$ . After rescaling to account for randomized response, the contribution $Z _ { u }$ in Algorithm 1 satisfies ${ \mathbb E } [ Z _ { u } \mid X _ { u } ] = { \bf 1 } \{ X _ { u } \leq j \}$ Averaging these contributions therefore gives an unbiased estimate of $Z _ { j } ( X )$ . Since orthogonality holds for every node and the reports are generated without reference to j, the server can reuse the same reports to estimate diferent prefixes.

Since $\mathcal { C } ( j )$ contains at most one node per depth, each report takes $O ( 1 )$ time to process, giving $O ( n + \log D )$ time per query. This also bounds each user’s rescaled contribution $Z _ { u }$ , allowing concentration bounds to establish simultaneous accuracy over all prefixes. See C in the appendix for the formal proof.

## 6.2 Eficient LDP synthetic dataset generation

In this section, we develop an eficient non-interactive LDP algorithm for generating a synthetic dataset whose empirical distribution approximates the discretized input distribution in Kolmogorov distance. We first introduce a discretization that maps real-valued samples to a finite ordered set of bin indices, allowing us to apply the prefix oracle from Section 6.1. We then use this oracle to locate approximate quantiles and place synthetic points at these locations. Although the distributions considered in our applications may have unbounded support, suitable parameter bounds allow us to control both the tail mass and the approximation error introduced by binning.

Definition 6.2 (Uniform discretization). Let $R , \delta > 0$ be parameters, and $D = \lceil 2 R / \delta \rceil + 2$ . Let $c _ { 0 } = - R - \delta , c _ { D } = R + \delta$ , and set $c _ { j } = - R + \operatorname* { m i n } \{ ( j - 1 ) \delta , 2 R \}$ for $j \in [ D - 1 ]$ . For a probability distribution P over R, define the discretized distribution Disc $\begin{array} { r } { { \bf \Phi } _ { \mathrm { \cdot } { R } , \delta } ( P ) : = \sum _ { j = 1 } ^ { D } P ( B _ { j } ) } \end{array}$ Unif $\big ( [ c _ { j - 1 } , c _ { j } ) \big )$ , where $B _ { 1 } = ( - \infty , - R ) , B _ { D } = [ R , + \infty )$ , and $B _ { j } = [ c _ { j - 1 } , c _ { j } )$ for $j \in \{ 2 , \cdots , D - 1 \}$ . By a slight abuse of notation, for a point $x \in \mathbb { R }$ , we write Disc ${ } _ { R , \delta } ( x ) = j \ i f f x \in B _ { j }$ , which denotes the index of the interval containing $x .$ Each endpoint $c _ { j }$ and the index of the interval containing a given x can be computed in constant time.

```latex
Algorithm 2 Non-interactive LDP synthetic data generation
Input: Parameters $\varepsilon , \alpha \in ( 0 , 1 )$ 2 $R , \delta > 0 ,$ , and n users holding values $X _ { 1 } , \ldots , X _ { n } \in \mathbb { R } .$
1: Set $D \gets \lceil 2 R / \delta \rceil + 2$ and $M \gets \lceil 1 / \alpha \rceil$ , and let the endpoints $c _ { j }$ be as in Definition 6.2.
2: Each user computes $\tilde { X } _ { u } \gets \mathrm { D i s c } _ { R , \delta } ( X _ { u } ) \in [ D ]$ locally. Let M be the local randomizer in
Algorithm 4, and set the users’ transcripts to be $\mathrm { T r } = \left( \mathcal { M } \left( \widetilde { X } _ { 1 } , \varepsilon , D \right) , \ldots , \mathcal { M } \left( \widetilde { X } _ { n } , \varepsilon , D \right) \right)$
3: for $\ell = 1 , \ldots , M$ do
4: Set $u _ { \ell } \gets \frac { \ell - 1 / 2 } { M } .$ ▷ Target quantile level for the ℓ-th point.
5: Initialize $( \vec { a } , \vec { b } )  ( \vec { 0 } , \vec { D } )$ ▷ Indices of the current search endpoints.
6: Initialize $( A , B )  ( 0 , 1 )$ ▷ CDF estimates at $c _ { a } , c _ { b } ;$ maintain $A < u _ { \ell } \leq B$
7: while $b > a + 1$ do
8: Let $j \gets \lfloor ( a + b ) / 2 \rfloor$
9: Compute $\begin{array} { r } { \hat { Z } _ { j } \gets \dot { \bf \Phi } . } \end{array}$ PrefixQuery(j, Tr). ▷ Estimated CDF at $c _ { j }$ using Algorithm 1.
10: if $\widehat { Z } _ { j } < u _ { \ell }$ then
11: $( a , A )  ( j , \widehat { Z } _ { j } ) .$
12: else
13: $( b , B )  ( j , \widehat { Z } _ { j } )$
14: Set $\begin{array} { r } { \lambda _ { \ell }  \frac { u _ { \ell } - A } { B - A } . } \end{array}$ ▷ Relative position of $u _ { \ell }$ between A and B.
15: Set $z _ { \ell } \gets c _ { b - 1 } + \lambda _ { \ell } ( c _ { b } - c _ { b - 1 } )$ ▷ Place the synthetic point at this position in bin b.
16: return the synthetic dataset $\mathcal { Z }  \{ z _ { 1 } , . . . , z _ { M } \}$ , passed as a multiset.
```

The following theorem presents the privacy and utility guarantees of our synthetic data generation algorithm (Algorithm 2). Utility is measured in Kolmogorov distance with respect to the discretized input distribution $\mathrm { D i s c } _ { R , \delta } ( P )$ . We later show how choosing R and δ appropriately for the distribution class of interest allows us to relate this guarantee to the original distribution P, using suitable tail bounds and binning arguments.

Theorem 6.3 (LDP synthetic dataset). Let $\varepsilon , \alpha , \beta \in ( 0 , 1 )$ , $R > 0 , \delta \in ( 0 , R / 2 ]$ , and $n \in \mathbb N$ be parameters, and let P be an unknown distribution over R. Then, the following holds:

(1) For any dataset $X \in \mathbb { R } ^ { n }$ , Algorithm 2 is non-interactive ${ \varepsilon } { - } L D P .$

(2) If the dataset X consists of n i.i.d. samples from P, where $\begin{array} { r } { n = \Omega \left( \frac { 1 } { \varepsilon ^ { 2 } \alpha ^ { 2 } } \log ^ { 2 } \left( \frac { R } { \delta } \right) \log \left( \frac { R } { \beta \delta } \right) \right) } \end{array}$ ， then with probability at least $1 - \beta$ over the sampling and local randomizers, the empirical probability measure $\begin{array} { r } { H : = \frac { 1 } { M } \sum _ { i \in [ M ] } \delta _ { z _ { i } } } \end{array}$ of the output, satisfies $\left\| F _ { H } - F _ { \mathrm { D i s c } _ { R , \delta } ( P ) } \right\| _ { \infty } \leq \alpha$ (3) The total run-time of Algorithm 2, including user preprocessing and server side computation, is $\begin{array} { r } { O \left( \frac { 1 } { \alpha } \log \left( \frac { R } { \delta } \right) \left( n + \log \left( \frac { R } { \delta } \right) \right) \right) } \end{array}$

Let $\tilde { P } : = \operatorname { D i s c } _ { R , \delta } ( P )$ . At a high level, Algorithm 2 represents $\widetilde { P }$ by M equally weighted points, one near each quantile level $u _ { \ell } .$ . Using the estimates $\widehat { Z } _ { j }$ returned by PrefixQuery in Algorithm 1, we perform binary search to find adjacent bin endpoints whose estimated CDF values A, B satisfy

$A < u _ { \ell } \leq B$ . The interpolation weight $\lambda _ { \ell }$ then places $z _ { \ell }$ within this bin so that its estimated cumulative probability matches $u _ { \ell }$

Since $F _ { \widetilde { P } }$ is linear within each bin, the errors in the endpoint estimates translate directly into errors between $F _ { \widetilde { P } } ( z _ { \ell } )$ and $u _ { \ell }$ . Since the quantile levels $u _ { \ell }$ are evenly spaced and each point $z _ { \ell }$ has mass $1 / M$ in $H ,$ these errors control how closely $F _ { H }$ follows $F _ { \widetilde { P } }$ , with an additional error of order $1 / M$ from using finitely many points. All searches reuse the same transcript and require only O(log D) prefix queries per point. Appendix C provides the formal proof.

## 7 Eficient LDP agnostic piece-wise polynomial learner

In this section, we combine the ingredients provided above to obtain an eficient LDP agnostic piece-wise polynomial learner. We feed the private synthetic dataset (from section 6.2) into the deterministic adaptation of the non-private piece-wise polynomial learner of Acharya et al. [ADLS17] (See Proposition 5.4). To relate the resulting guarantee to the unknown distribution, we account for the error introduced by discretization (Definition 2.3), as well as the error from approximating the target class by piece-wise polynomials (Definition 2.5).

Putting these ingredients together gives our general eficient LDP agnostic learning framework theorem. See D in the appendix for the proof.

Theorem 7.1. Let $\varepsilon , \alpha , \beta \in ( 0 , 1 ) , R > 0 , 0 < \delta \leq R / 2$ be parameters, let $s \geq 1 , d \geq 0$ be integers, let P be an unknown distribution over $\mathbb { R }$ , and let F be a known class of probability distributions over R. Then, there exist a universal integer $C \geq 1$ and a non-interactive ${ \varepsilon } { - } L D P$ algorithm that, using $\begin{array} { r } { n = \Theta \left( \frac { s ^ { 2 } ( d + 1 ) ^ { 2 } } { \varepsilon ^ { 2 } \alpha ^ { 2 } } \log ^ { 2 } { \left( \frac { R } { \delta } \right) } \log { \left( \frac { R } { \beta \delta } \right) } \right) } \end{array}$ i.i.d. samples from $P _ { z }$ , outputs a distribution $\widehat { P }$ with a nonnegative piece-wise polynomial density in $\mathcal { P } _ { C s , d }$ satisfying:

$$
\begin{array} { r } { \| \widehat { P } - P \| _ { 1 } \leq 1 0 0 \mathrm { P T } + 8 \alpha + 9 \mathrm { D i s c } \mathrm { - E r r } _ { R , \delta } ( \mathcal { F } ) + 8 \mathrm { P o l y } \mathrm { - A p p r o x } _ { s , d } ( \mathcal { F } ) } \end{array}
$$

with probability at least $1 - \beta$ over sampling and local randomizers, where $\begin{array} { r } { \operatorname { O P T } : = \operatorname* { i n f } _ { F \in \mathcal { F } } \| F - P \| _ { 1 } } \end{array}$ Moreover, the total running time, including user preprocessing and server side computation, is $\begin{array} { r } { \widetilde { O } \left( \frac { s ( d + 1 ) } { \alpha } \left( n \log \left( \frac { R } { \delta } \right) + ( d + 1 ) ^ { \omega + 2 } \right) \right) } \end{array}$ , where ω denotes the matrix multiplication exponent, and $\widetilde O$ hides only logarithmic factors in $s , d + 1$ , and $1 / \alpha$

## 8 Eficient LDP Agnostic Learning of Mixtures of Log-concave distributions

In this section, as an application of our framework, we provide an eficient non-interactive LDP agnostic learner for mixtures of log-concave distributions with bounded component parameters. The class of mixtures of log-concave distributions is quite expressive and includes GMMs and mixtures of uniform, exponential, Laplace, and logistic distributions. It also allows mixtures whose components belong to diferent distribution families. Restrictions on the component parameter ranges are necessary for a uniform finite bound on sample complexity in the non-interactive LDP setting, as we demonstrate below by the packing lower bound for GMMs in Theorem G.1. All proofs omitted from this section are given in Section E.

## 8.1 Mixture distributions

Here, we define mixture distributions and show that bounding their polynomial approximation and discretization errors reduces to bounding the corresponding errors for the component class.

Definition 8.1 (Mixture class). Let Q be an arbitrary class of distributions. We denote the class of m-mixtures $o f \mathcal { Q }$ by $\boldsymbol { m } \cdot \boldsymbol { m } i x ( \mathcal { Q } ) = \Delta _ { m } \times \mathcal { Q } ^ { m }$ where $\begin{array} { r } { \Delta _ { m } = \Big \{ w \in \mathbb { R } ^ { m } : w _ { i } \geq 0 , \sum _ { i \in [ m ] } w _ { i } = 1 \Big \} } \end{array}$ is the $( m - 1 )$ -dimensional probability simplex.

Lemma 8.2 (Discretization and polynomial approximation of mixtures). Let $\mathcal { F }$ be a nonempty class of probability distributions over R with densities. Let $m , s \geq 1$ and $d \geq 0$ be integers, and let $R , \delta > 0$ . Then,

(1) Disc-Err ${ } _ { R , \delta } \left( m \mathbf { - } m i x ( \mathcal { F } ) \right) = \operatorname { D i s c \mathbf { - } E r r } _ { R , \delta } ( \mathcal { F } )$

(2) $\mathrm { P o l y - A p p r o x } _ { m ( s - 1 ) + 1 , d } \left( m - m i x ( \mathcal { F } ) \right) \leq \mathrm { P o l y - A p p r o x } _ { s , d } ( \mathcal { F } )$

## 8.2 An upper bound for mixtures of log-concave distributions

The formal definition of bounded-parameter log-concave distributions is given as follows.

Definition 8.3 (Log-concave functions). A function $f : \mathbb { R } \to \mathbb { R } _ { > 0 }$ is log-concave $i f ,$ for every $x , y \in \mathbb { R }$ and $\lambda \in ( 0 , 1 )$ , it holds that $f ( \lambda x + ( 1 - \lambda ) y ) \geq f ( x ) ^ { \lambda } f ( y ) ^ { 1 - \lambda }$

Definition 8.4 (Bounded-Parameter Log-concave Distributions). Let $R \geq 0$ , and $0 < \sigma _ { \mathrm { m i n } } \le \sigma _ { \mathrm { m a x } } <$ ∞ be parameters. Then the class of bounded-parameter log-concave distributions over R is defined by $\mathcal { L } ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ) = \{ f : f$ is a log-concave probability density, $| \mu _ { f } | \le R , \sigma _ { f } \in [ \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ] \}$ , where $\mu _ { f } = \mathbb { E } _ { X \sim f } [ X ]$ , and $\sigma _ { f } ^ { 2 } = \mathbb { E } _ { X \sim f } [ ( X - \mu _ { f } ) ^ { 2 } ]$

The following theorem gives the resulting guarantee for eficient non-interactive LDP agnostic learning of log-concave mixtures.

Theorem 8.5 (Eficient LDP Agnostic Learning of Log-concave Mixtures). Let $\varepsilon , \alpha , \beta \in ( 0 , 1 )$ , let $m \geq 1$ be an integer, and let $R \geq 0$ and $0 < \sigma _ { \mathrm { m i n } } \le \sigma _ { \mathrm { m a x } } < \infty$ be parameters. Then, there exists a non-interactive $\varepsilon { - } L D P \left( 1 0 , \alpha , \beta \right) { - } a g \imath$ nostic learner for the class ${ \it m } { \cdot } m i x ( \mathcal { L } ( R , \sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } } ) )$ , that uses $\begin{array} { r } { n = { \widetilde { O } } \left( \frac { m ^ { 2 } } { { \varepsilon } ^ { 2 } \alpha ^ { 3 } } \log ^ { 2 } \left( \frac { 2 ( R + \sigma _ { \operatorname* { m a x } } ) } { \alpha \sigma _ { \operatorname* { m i n } } } \right) \log \left( \frac { 2 ( R + \sigma _ { \operatorname* { m a x } } ) } { \alpha \beta \sigma _ { \operatorname* { m i n } } } \right) \right) } \end{array}$ , samples and its output has a nonnegative piece-wise linear density with $\widetilde { O } ( m / \sqrt { \alpha } )$ pieces. Moreover, the total running time is $\begin{array} { r } { \tilde { O } \left( \frac { n m } { \alpha ^ { 3 / 2 } } \log \left( \frac { 2 ( R + \sigma _ { \operatorname* { m a x } } ) } { \alpha \sigma _ { \operatorname* { m i n } } } \right) \right) } \end{array}$ Here, $\tilde { O }$ hides only logarithmic factors in $1 / \alpha$ in the sample and piece-count bounds, and in m and $1 / \alpha$ in the running-time bound.

To obtain this result, it sufices to bound the polynomial approximation and discretization errors of the single-component class $\mathcal { L } ( R , \sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } } )$ . The following two lemmas provide these bounds, which extend to the mixture class by Lemma 8.2, allowing us to apply the LDP piece-wise polynomial learner from Theorem 7.1.

For log-concave distributions, the piece-wise linear approximation result of Chan et al. [CDSS14] gives the required polynomial approximation bound. We state it below in terms of our class approximation error.

Lemma 8.6 (Piece-wise polynomial approximation of log-concave distributions). For every α ∈ (0, 1), there exists an integer $s _ { \alpha } = \widetilde { O } ( 1 / \sqrt { \alpha } )$ such that, for every $R \geq 0$ and $0 < \sigma _ { \mathrm { m i n } } \le \sigma _ { \mathrm { m a x } } < \infty ,$ it holds that $\mathrm { P o l y - A p p r o x } _ { s _ { \alpha } , 1 } \left( \mathcal { L } ( R , \sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } } ) \right) \leq \alpha$ . Here, Oe hides only logarithmic factors in $1 / \alpha$

Standard tail bounds for log-concave distributions control the error from truncating the domain. Within the retained intervals, we use unimodality and a bound on the density to control the error introduced by discretization. Combining these arguments gives the following lemma.

Lemma 8.7 (Discretization of Bounded-Parameter Log-concave Distributions). Let $\eta \in ( 0 , 1 )$ $R \geq 0$ , and $0 < \sigma _ { \mathrm { m i n } } \le \sigma _ { \mathrm { m a x } } < \infty$ be parameters. Let $\begin{array} { r } { T = R + \sigma _ { \operatorname* { m a x } } \left( 1 + \log \left( \frac { 4 } { \eta } \right) \right) , \ J = \left\lceil \frac { 8 T } { \eta \sigma _ { \operatorname* { m i n } } } \right\rceil } \end{array}$ 2 and $\begin{array} { r } { \delta = \frac { 2 T } { J } } \end{array}$ . Then, it holds that $\operatorname { D i s c - E r r } _ { T , \delta } \left( \mathcal { L } ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ) \right) \leq \eta$

Together with the piece-wise linear approximation bound above, this allows us to apply Theorem 7.1 and obtain the eficient non-interactive ε-LDP agnostic learner stated in Theorem 8.5.

## AI Disclosure

The authors initially worked with the factorization mechanism of Edmonds et al. [ENU20] for answering prefix frequency queries. While the authors were looking for ways to make their algorithm time-eficient, AI (GPT-5.6 Sol) highlighted the connection to the local randomizer of Cormode et al. [CKS19]. The authors also used AI to check proofs and identify edge cases in algorithms and theorem statements, which they subsequently addressed. All text, theorems, and proofs were written by the authors. AI assistance with writing was limited to language and grammar refinement.

## Acknowledgments

Mohammad Afzali was supported in part by a Four Year Doctoral Fellowship (4YF) from the University of British Columbia. Sivan Sabato holds a Canada AI CIFAR Chair at the Vector Institute. Hassan Ashtiani is also a faculty member at the Vector Institute. We acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC), funding reference numbers RGPIN-2024-05907 and RGPIN-2019-06018. Resources used in preparing this research were provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute; see https://vectorinstitute.ai/partnerships current-partners/.

## References

[AAK21] Ishaq Aden-Ali, Hassan Ashtiani, and Gautam Kamath “On the Sample Complexity of Privately Learning Unbounded High-Dimensional Gaussians”. In: Proceedings of the 32nd International Conference on Algorithmic Learning Theory. Vol. 132. Proceedings of Machine Learning Research. 2021, pp. 185–216.

[AAL21] Ishaq Aden-Ali, Hassan Ashtiani, and Christopher Liaw “Privately Learning Mixtures of Axis-Aligned Gaussians”. In: Advances in Neural Information Processing Systems. Vol. 34. 2021.

[AAL23] Jamil Arbas, Hassan Ashtiani, and Christopher Liaw “Polynomial Time and Private Learning of Unbounded Gaussian Mixture Models”. In: Proceedings of the 40th International Conference on Machine Learning. Vol. 202. Proceedings of Machine Learning Research. 2023, pp. 1018–1040.

[AAL24] Mohammad Afzali, Hassan Ashtiani, and Christopher Liaw “Mixtures of Gaussians are Privately Learnable with a Polynomial Number of Samples”. In: Proceedings of The 35th International Conference on Algorithmic Learning Theory. Vol. 237. Proceedings of Machine Learning Research. 2024, pp. 47–73.

[AAL25] Mohammad Afzali, Hassan Ashtiani, and Christopher Liaw “Agnostic Private Density Estimation for GMMs via List Global Stability”. In: Proceedings of The 36th International Conference on Algorithmic Learning Theory. Vol. 272. Proceedings of Machine Learning Research. 2025, pp. 41–66.

[ABHLMP20] Hassan Ashtiani, Shai Ben-David, Nicholas JA Harvey, Christopher Liaw, Abbas Mehrabian, and Yaniv Plan “Near-optimal sample complexity bounds for robust learning of gaussian mixtures via compression schemes”. In: Journal of the ACM (JACM) 67.6 (2020), pp. 1–42.

[ADLS17] Jayadev Acharya, Ilias Diakonikolas, Jerry Li, and Ludwig Schmidt “Sample-Optimal Density Estimation in Nearly-Linear Time”. In: Proceedings of the Twenty-Eighth Annual ACM-SIAM Symposium on Discrete Algorithms. SODA ’17. 2017, pp. 1278–1289.

[AKLS20] Jayadev Acharya, Peter Kairouz, Yuhan Liu, and Ziteng Sun “Estimating sparse discrete distributions under local privacy and communication constraints”. In: arXiv preprint arXiv:2011.00083 (2020).

[AKTVZ23] Daniel Alabi, Pravesh K. Kothari, Pranay Tankala, Prayaag Venkat, and Fred Zhang “Privately Estimating a Gaussian: Eficient, Robust, and Optimal”. In: Proceedings of the 55th Annual ACM Symposium on Theory of Computing. 2023, pp. 483–496.

[AL22] Hassan Ashtiani and Christopher Liaw “Private and polynomial time algorithms for learning Gaussians and beyond”. In: Proceedings of Thirty Fifth Conference on Learning Theory. Vol. 178. Proceedings of Machine Learning Research. 2022, pp. 1075–1076.

[AMN24] Hassan Ashtiani, Mahbod Majid, and Shyam Narayanan “Sample-Eficient Private Learning of Mixtures of Gaussians”. In: Advances in Neural Information Processing Systems. Vol. 37. 2024.

[App17] Apple Diferential Privacy Team Learning with Privacy at Scale. 2017.

[ASZ19] Jayadev Acharya, Ziteng Sun, and Huanyu Zhang “Hadamard Response: Estimating Distributions Privately, Eficiently, and with Little Communication”. In: Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics. Vol. 89. Proceedings of Machine Learning Research. 2019, pp. 1120–1129.

[BDKS20] Cristina Butucea, Amandine Dubois, Martin Kroll, and Adrien Saumard “Local diferential privacy: Elbow efect in optimal density estimation and adaptation over Besov ellipsoids”. In: Bernoulli 26.3 (2020), pp. 1727–1764.

[BGSUZ21] Gavin Brown, Marco Gaboardi, Adam Smith, Jonathan Ullman, and Lydia Zakynthinou “Covariance-Aware Private Mean Estimation Without Private Covariance Estimation”. In: Advances in Neural Information Processing Systems. Vol. 34. 2021.

[BHS23] Gavin Brown, Samuel Hopkins, and Adam Smith “Fast, Sample-Eficient, Afine-Invariant Private Mean and Covariance Estimation for Subgaussian Distributions”. In: Proceedings of Thirty Sixth Conference on Learning Theory. Vol. 195. Proceedings of Machine Learning Research. 2023, pp. 5578–5579.

[BKSW19] Mark Bun, Gautam Kamath, Thomas Steinke, and Zhiwei Steven Wu “Private Hypothesis Selection”. In: Advances in Neural Information Processing Systems. Vol. 32. 2019.

[BNSV15] Mark Bun, Kobbi Nissim, Uri Stemmer, and Salil Vadhan “Diferentially Private Release and Learning of Threshold Functions”. In: 2015 IEEE 56th Annual Symposium on Foundations of Computer Science. 2015.

[CDSS14] Siu-On Chan, Ilias Diakonikolas, Rocco A Servedio, and Xiaorui Sun “Eficient density estimation via piecewise polynomial approximation”. In: Proceedings of the forty-sixth annual ACM symposium on Theory of computing. 2014, pp. 604–613.

[CKS19] Graham Cormode, Tejas Kulkarni, and Divesh Srivastava “Answering Range Queries Under Local Diferential Privacy”. In: Proceedings of the VLDB Endowment 12.10 (2019), pp. 1126–1138.

[CSS11] T.-H. Hubert Chan, Elaine Shi, and Dawn Song “Private and Continual Release of Statistics”. In: ACM Transactions on Information and System Security 14.3 (2011), 26:1–26:24.

[DHS15] Ilias Diakonikolas, Moritz Hardt, and Ludwig Schmidt “Diferentially Private Learning of Structured Discrete Distributions”. In: Advances in Neural Information Processing Systems. Vol. 28. 2015.

[DJW13] John C Duchi, Michael I Jordan, and Martin J Wainwright “Local privacy and statistical minimax rates”. In: 2013 IEEE 54th annual symposium on foundations of computer science. IEEE. 2013, pp. 429–438.

[DJW18] John C. Duchi, Michael I. Jordan, and Martin J. Wainwright “Minimax Optimal Procedures for Locally Private Estimation”. In: Journal of the American Statistical Association 113.521 (2018), pp. 182–201.

[DK23] Ilias Diakonikolas and Daniel M Kane Algorithmic high-dimensional robust statistics. Cambridge university press, 2023.

[DKKLMS19] Ilias Diakonikolas, Gautam Kamath, Daniel Kane, Jerry Li, Ankur Moitra, and Alistair Stewart “Robust estimators in high-dimensions without the computational intractability”. In: SIAM Journal on Computing 48.2 (2019), pp. 742–864.

[DL01] Luc Devroye and Gábor Lugosi Combinatorial Methods in Density Estimation. Springer Series in Statistics. Springer, 2001.

[DMNS06] Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith “Calibrating Noise to Sensitivity in Private Data Analysis”. In: Theory of Cryptography Conference. 2006, pp. 265–284.

[DNPR10] Cynthia Dwork, Moni Naor, Toniann Pitassi, and Guy N. Rothblum “Diferential Privacy under Continual Observation”. In: Proceedings of the Forty-Second ACM Symposium on Theory of Computing. 2010, pp. 715–724.

[EGS03] Alexandre Evfimievski, Johannes Gehrke, and Ramakrishnan Srikant “Limiting privacy breaches in privacy preserving data mining”. In: Proceedings of the twentysecond ACM SIGMOD-SIGACT-SIGART symposium on Principles of database systems. 2003, pp. 211–222.

[ENU20] Alexander Edmonds, Aleksandar Nikolov, and Jonathan Ullman “The Power of Factorization Mechanisms in Local and Central Diferential Privacy”. In: Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing. STOC ’20. 2020, pp. 425–438.

[EPK14] Úlfar Erlingsson, Vasyl Pihur, and Aleksandra Korolova “RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response”. In: Proceedings of the 2014 ACM SIGSAC Conference on Computer and Communications Security. ACM, 2014.

[GKKNWZ20] Sivakanth Gopi, Gautam Kamath, Janardhan Kulkarni, Aleksandar Nikolov, Zhiwei Steven Wu, and Huanyu Zhang “Locally Private Hypothesis Selection”. In: Proceedings of Thirty Third Conference on Learning Theory. Vol. 125. Proceedings of Machine Learning Research. 2020, pp. 1785–1816.

[GRS19] Marco Gaboardi, Ryan Rogers, and Or Shefet “Locally Private Mean Estimation: Z-test and Tight Confidence Intervals”. In: Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics. Vol. 89. Proceedings of Machine Learning Research. 2019, pp. 2545–2554.

[HKMN23] Samuel B. Hopkins, Gautam Kamath, Mahbod Majid, and Shyam Narayanan “Robustness Implies Privacy in Statistical Estimation”. In: Proceedings of the 55th Annual ACM Symposium on Theory of Computing. 2023, pp. 497–506.

[JKMW19] Matthew Joseph, Janardhan Kulkarni, Jieming Mao, and Steven Z. Wu “Locally Private Gaussian Estimation”. In: Advances in Neural Information Processing Systems. Vol. 32. 2019.

[KDH23] Rohith Kuditipudi, John Duchi, and Saminul Haque “A Pretty Fast Algorithm for Adaptive Private Mean Estimation”. In: Proceedings of Thirty Sixth Conference on Learning Theory. Vol. 195. Proceedings of Machine Learning Research. 2023, pp. 2511–2551.

[KLNRS11] Shiva Prasad Kasiviswanathan, Homin K. Lee, Kobbi Nissim, Sofya Raskhodnikova, and Adam Smith “What Can We Learn Privately?” In: SIAM Journal on Computing 40.3 (2011), pp. 793–826.

[KLSU19] Gautam Kamath, Jerry Li, Vikrant Singhal, and Jonathan Ullman “Privately Learning High-Dimensional Distributions”. In: Proceedings of the Thirty-Second Conference on Learning Theory. Vol. 99. Proceedings of Machine Learning Research. 2019, pp. 1853–1902.

[KMSSU22] Gautam Kamath, Argyris Mouzakis, Vikrant Singhal, Thomas Steinke, and Jonathan Ullman “A Private and Computationally-Eficient Estimator for Unbounded Gaussians”. In: Proceedings of Thirty Fifth Conference on Learning Theory. Vol. 178. Proceedings of Machine Learning Research. 2022, pp. 544–572.

[KMV22] Pravesh K. Kothari, Pasin Manurangsi, and Ameya Velingker “Private Robust Estimation by Stabilizing Convex Relaxations”. In: Proceedings of Thirty Fifth Conference on Learning Theory. Vol. 178. Proceedings of Machine Learning Research. 2022, pp. 723–777.

[KPRW25] Gautam Kamath, Alireza F. Pour, Matthew Regehr, and David P. Woodruf “Query-Eficient Locally Private Hypothesis Selection via the Schefé Graph”. In: Advances in Neural Information Processing Systems. Vol. 38. 2025.

[KSSU19] Gautam Kamath, Or Shefet, Vikrant Singhal, and Jonathan Ullman “Diferentially Private Algorithms for Learning Mixtures of Separated Gaussians”. In: Advances in Neural Information Processing Systems. Vol. 32. 2019.

[KV18] Vishesh Karwa and Salil Vadhan “Finite Sample Diferentially Private Confidence Intervals”. In: 9th Innovations in Theoretical Computer Science Conference. Vol. 94. Leibniz International Proceedings in Informatics. 2018.

[LBY23] Mengchu Li, Thomas B. Berrett, and Yi Yu “On robustness and local diferential privacy”. In: The Annals of Statistics 51.2 (2023), pp. 717–737.

[LKO22] Xiyang Liu, Weihao Kong, and Sewoong Oh “Diferential privacy and robust statistics in high dimensions”. In: Proceedings of Thirty Fifth Conference on Learning Theory. Vol. 178. Proceedings of Machine Learning Research. 2022, pp. 1167–1246.

[LRV16] Kevin A Lai, Anup B Rao, and Santosh Vempala “Agnostic estimation of mean and covariance”. In: 2016 IEEE 57th Annual Symposium on Foundations of Computer Science (FOCS). IEEE. 2016, pp. 665–674.

[LV07] László Lovász and Santosh Vempala “The Geometry of Logconcave Functions and Sampling Algorithms”. In: Random Structures & Algorithms 30.3 (2007), pp. 307– 358.

[ODo14] Ryan O’Donnell Analysis of Boolean Functions. Cambridge University Press, 2014.

[PAA24] Alireza F. Pour, Hassan Ashtiani, and Shahab Asoodeh “Sample-Optimal Locally Private Hypothesis Selection and the Provable Benefits of Interactivity”. In: Proceedings of Thirty Seventh Conference on Learning Theory. Vol. 247. Proceedings of Machine Learning Research. 2024, pp. 4240–4275.

[War65] Stanley L Warner “Randomized response: A survey technique for eliminating evasive answer bias”. In: Journal of the American statistical association 60.309 (1965), pp. 63–69.

[Yat85] Yannis G Yatracos “Rates of convergence of minimum distance estimators and Kolmogorov’s entropy”. In: The Annals of Statistics 13.2 (1985), pp. 768–774.

[ZZ25] Manolis Zampetakis and Felix Zhou “Private Statistical Estimation via Truncation”. In: Advances in Neural Information Processing Systems. Vol. 38. 2025.

## A More on the related work

In this section, we discuss further related work and its connections to our framework.

In the realizable setting, prior work has studied nonparametric density estimation under LDP under smoothness assumptions. Duchi et al. [DJW18] study Sobolev density estimation under squared $L _ { 2 }$ loss, while Butucea et al. [BDKS20] study Besov classes under integrated $L _ { r }$ loss, including adaptation to unknown smoothness. Both works assume that the sampling density belongs to a specified smoothness class on a fixed bounded interval, and give non-interactive estimators with polynomial running time for fixed smoothness and support parameters, assuming eficient basis evaluation. Although their losses imply TV bounds on bounded supports, these guarantees still require the sampling density to belong to the specified smoothness class. Moreover, their rates depend on smoothness norms and support parameters and do not directly recover our dependence on the number of mixture components or our polylogarithmic dependence on the component parameter range. Our framework instead gives an agnostic TV guarantee based on piece-wise polynomia approximation, covering bounded-parameter mixtures with unbounded support or discontinuities.

Li et al. [LBY23] study Huber contamination, where samples come from $P = ( 1 - \eta ) F + \eta G$ and the goal is to recover the clean distribution F. Their density-estimation results assume that $F$ belongs to a Sobolev class and that both F and G are supported on [0, 1]. They give non-interactive estimators with polynomial running time for fixed smoothness parameters, assuming eficient basis evaluation, and minimax guarantees under squared $L _ { 2 }$ and $L _ { \infty }$ losses. These imply TV bounds after conversion to a density, but do not establish an agnostic guarantee for arbitrary sampling distributions or our dependence on mixture complexity and component parameter ranges. Our guarantee allows arbitrary P and, under the contamination model above with $F \in { \mathcal { F } } .$ , implies $\mathrm { d _ { T V } } ( \widehat { P } , F ) \leq ( C + 1 ) \eta + \alpha$ . However, it does not provide their stronger $L _ { 2 }$ or $L _ { \infty }$ guarantees.

In the central DP model, learning a single Gaussian has been studied extensively, from univariate confidence intervals [KV18] to high-dimensional learning and agnostic estimation of unrestricted Gaussians [KLSU19; AAK21]. Further work studies covariance-aware estimation and robustness [BG-SUZ21; LKO22; KMV22]. Several works provide computationally eficient algorithms for Gaussian mean, covariance, or density estimation [KMSSU22; AL22; AKTVZ23; HKMN23; BHS23; KDH23]. More recently, Zampetakis and Zhou [ZZ25] develop a truncation-based framework for private statistical estimation, including eficient Gaussian mean and covariance estimation.

For mixtures, Aden-Ali et al. [AAL21] give central DP sample complexity bounds for mixtures of axisaligned Gaussians, and Afzali et al. [AAL24] establish polynomial sample complexity for mixtures of arbitrary multivariate Gaussians. Ashtiani et al. [AMN24] improve these bounds and, in particular, obtain a realizable univariate learner using $\widetilde { \cal O } ( m / \alpha ^ { 2 } + m \log ( 1 / \delta ) / ( \varepsilon \alpha ) )$ samples under centra (ε, δ)-DP, where δ is the privacy failure parameter. In the agnostic setting, Afzali et al. [AAL25] give a central DP learner for unrestricted multivariate GMMs; their univariate specialization uses ${ \widetilde O } ( ( m ^ { 2 } + m \log ( 1 / \delta ) ) / ( \varepsilon \alpha ^ { 2 } ) )$ samples. These unrestricted density-learning algorithms are computationally ineficient. For mixture parameter estimation, Kamath et al. [KSSU19] give eficient central DP algorithms under strong separation assumptions. Arbas et al. [AAL23] allow weaker separation and unbounded parameters, but their resulting algorithm has sample complexity and running time exponential in the number of components m, with a polynomial-time guarantee for fixed m. The privatization techniques in the central and local models difer substantially: a central algorithm can access the dataset jointly, whereas each user in our non-interactive local algorithms privatizes their own sample independently. Our results provide eficient agnostic learners for bounded-parameter univariate mixtures in this local setting, with only logarithmic factors in the component parameter ranges.

## B Non-private piece-wise polynomial learner with synthetic data

Here, we provide a proof sketch for the deterministic adaptation of the non-private piece-wise polynomial learner of Acharya et al. [ADLS17] that we use in our framework. Their original estimator takes i.i.d. samples from an unknown distribution, but it can also operate on synthetic data whose empirical distribution satisfies a suitable Kolmogorov-distance bound. We first define the $\mathcal { A } _ { k }$ norm used in their analysis.

Definition B.1 $( \mathcal { A } _ { k } \mathrm { { - n o r m } \ [ A D L S 1 7 ] ) }$ . Let $I \subseteq \mathbb { R }$ be an interval, and $k \geq 1$ be an integer. For a finite signed measure $\mu$ on R, the $\mathcal { A } _ { k }$ norm restricted to I is defined as:

$$
\| \mu \| _ { \mathcal { A } _ { k } , I } : = \operatorname* { s u p } _ { \substack { 0 \leq r \leq k } } \sum _ { \substack { I _ { 1 } , \ldots , I _ { r } \subseteq I } } | \mu ( I _ { i } ) | .
$$

The intervals may have arbitrary real endpoints. We allow open, closed, half-open, and singleton intervals. When $I = \mathbb { R }$ , we write $\| \mu \| _ { \mathcal { A } _ { k } }$ instead of $\| \mu \| _ { \mathcal { A } _ { k } , \mathbb { R } }$

Lemma B.2 (A -distance from CDF error). Let $P , Q$ be two probability distributions over $\mathbb { R } _ { : }$ , with CDFs $F _ { P } , F _ { Q }$ , and let $k \geq 1$ be an integer. Then,

$$
\| P - Q \| _ { \mathcal { A } _ { k } } \le k \| P - Q \| _ { \mathcal { A } _ { 1 } } \le 2 k \| F _ { P } - F _ { Q } \| _ { \infty } .
$$

Proof. For any interval $I = ( a , b ]$ , we have

$$
\begin{array} { r l } & { | ( P - Q ) ( I ) | = | ( F _ { P } - F _ { Q } ) ( b ) - ( F _ { P } - F _ { Q } ) ( a ) | } \\ & { \qquad \le | ( F _ { P } - F _ { Q } ) ( b ) | + | ( F _ { P } - F _ { Q } ) ( a ) | \le 2 \| F _ { P } - F _ { Q } \| _ { \infty } . } \end{array}
$$

The same bound holds for other endpoint conventions and unbounded intervals by taking the corresponding one-sided limits of the CDFs. Taking the supremum over intervals gives

$$
\begin{array} { r } { \| P - Q \| _ { \cal A _ { 1 } } \le 2 \| F _ { P } - F _ { Q } \| _ { \infty } . } \end{array}
$$

Moreover, for any collection of $r \leq k$ disjoint intervals $I _ { 1 } , \cdots , I _ { r } ,$

$$
\sum _ { i \in [ r ] } | ( P - Q ) ( I _ { i } ) | \leq r \| P - Q \| _ { \ r { A _ { 1 } } } \leq k \| P - Q \| _ { \ r { A _ { 1 } } } .
$$

Taking the supremum over all such collections proves the result.

Proof of Proposition $5 . 4 .$ . Fix an arbitrary probability distribution $P ^ { \star }$ satisfying the assumptions of the Proposition 5.4, and set $k = ( C + 1 ) ( d + 1 ) \colon$ s. By Lemma B.2,

$$
\| H - P ^ { \star } \| _ { { \cal A } _ { k } } \le 2 k \| F _ { H } - F _ { P ^ { \star } } \| _ { \infty } \le 2 ( C + 1 ) ( d + 1 ) s \cdot \frac { \eta } { 2 ( C + 1 ) ( d + 1 ) s } = \eta .
$$

We use the deterministic post-sampling adaptations of Theorem 17 of Acharya et al. [ADLS17] for $d > 0$ and Theorem 10 for $d = 0$ , with the supplied multiset Z replacing the sampling step. Their error analyses apply once the required $\scriptstyle A _ { k } - \mathrm { d i s t a n c e }$ bound holds, which we have established above. With suitable universal constants and approximation tolerances, they therefore give

$$
\| \widehat { P } - P ^ { \star } \| _ { 1 } \leq 4 \big ( \mathrm { O P T } + \eta \big ) .
$$

The output has at most Cs pieces of degree at most $d ,$ and the running time follows from the same deterministic algorithms on M input points. The conversion from Kolmogorov distance is used only in the analysis and requires no additional computation. □

## C Omitted proofs from sections 5 and 6

```latex
Algorithm 3 Canonical dyadic decomposition of a prefix
Input: Domain size $N = 2 ^ { L }$ and a query index $j \in [ N - 1 ]$
Output: A collection $\mathcal { C } ( j )$ of pairwise disjoint dyadic nodes whose union is [j].
1: Let V be the $2 N - 1$ nodes of the complete dyadic tree over [N] (See Definition 5.1). Index
them as
$I _ { \ell , v } = \{ v 2 ^ { L - \ell } + 1 , \ldots , ( v + 1 ) 2 ^ { L - \ell } \}$
for depth $\ell \in \{ 0 , \ldots , L \}$ and $v \in \{ 0 , \ldots , 2 ^ { \ell } - 1 \}$
2: Initialize $\mathscr { C } ( j ) \gets \emptyset$ and $s  0 .$
3: for each depth $\ell \in \{ 1 , \ldots , L \}$ do
4: $q  2 ^ { L \cdot }$ −ℓ ▷ Length of a node at depth ℓ
5: if $\lfloor j / q \rfloor$ mod $2 = 1$ then
6: $v  s / q$
7: ${ \mathcal { C } } ( j ) \gets { \mathcal { C } } ( j ) \cup \{ I _ { \ell , v } \}$
8: $s \gets s + q$
9: return $\mathcal { C } ( j )$
```

Proof of Lemma 5.2. We prove that the algorithm 3 satisfies the requirements of the lemma. Let $\begin{array} { r } { j = \sum _ { \ell \in [ L ] } b _ { \ell } 2 ^ { L - \ell } , } \end{array}$ , be the L-bit binary representation of $j$ , where $b _ { \ell } \in \{ 0 , 1 \}$ . At depth $\ell ,$ Algorithm 3 adds a node to $\mathcal { C } ( j )$ when $b _ { \ell } ~ = ~ 1$ . Let $s _ { 0 } ~ = ~ 0$ , and for each $\ell \in \ [ L ]$ define $\begin{array} { r } { s _ { \ell } : = \sum _ { r \in \left[ \ell \right] } b _ { r } 2 ^ { L - r } } \end{array}$ . We show that, after processing depth $\ell ,$ the nodes selected so far are pairwise disjoint and their union is exactly $[ 1 , s _ { \ell } ]$ . If $b _ { \ell } = 0$ , the algorithm does not add a node and $s _ { \ell } = s _ { \ell - 1 }$ so the claim holds. Otherwise, $b _ { \ell } = 1$ . Let $q = 2 ^ { L - \ell }$ . Note that at depth $\ell ,$ the variable s is initially equal to $s \ell - 1$ , and every term in $s \ell - 1$ is divisible by $q ,$ therefore $\begin{array} { r } { v = \frac { s _ { \ell - 1 } } { q } } \end{array}$ is an integer. So, the node selected by the algorithm is $I _ { \ell , v } = \{ v q + 1 , \ldots , ( v + 1 ) q \} = \{ s _ { \ell - 1 } + 1 , \ldots , s _ { \ell - 1 } + q \}$ . This node is disjoint from all previously selected nodes and is immediately adjacent to their union. Since $s _ { \ell } = s _ { \ell - 1 } + q$ , the union of the selected nodes after depth ℓ is exactly [1, s<sub>ℓ</sub>]. In particular at depth $L ,$ we have $\begin{array} { r } { s _ { L } = \sum _ { \ell \in \left[ L \right] } b _ { \ell } 2 ^ { L - \ell } = j } \end{array}$ . Consequently, the nodes in $\mathcal { C } ( j )$ are pairwise disjoint and exactly cover $[ 1 , j ]$ , proving (1).

Algorithm 4 Hierarchical LDP local randomizer M   
Input: A user’s value $X _ { u } \in [ D ]$ , privacy parameter $\varepsilon \in ( 0 , 1 )$ , and domain size $D \in \mathbb { N }$ with $D \geq 2$   
Output: A user transcript $T _ { u } = ( \ell _ { u } , r _ { u } , Y _ { u } )$   
1: Set $L \gets \lceil \log _ { 2 } D \rceil$   
2: Sample a depth $\ell _ { u }$ uniformly from $[ L ] .$   
3: Sample $r _ { u }$ uniformly from $\{ 0 , \ldots , 2 ^ { \ell _ { u } } - 1 \}$   
4: Set $\begin{array} { r } { v _ { u } \gets v _ { \ell _ { u } } ( X _ { u } ) = \left\lfloor \frac { X _ { u } - 1 } { 2 ^ { L - \ell _ { u } } } \right\rfloor } \end{array}$   
5: Set $B _ { u } \gets \mathrm { H a d } _ { \ell _ { u } } ( v _ { u } , r _ { u } ) \in \{ - 1 , + 1 \}$   
6: Set $p _ { \varepsilon }  \frac { e ^ { \varepsilon } } { e ^ { \varepsilon } + 1 }$ , and $Y _ { u } \gets \left\{ { B _ { u } , } \atop { - B _ { u } , } \right.$ with probability $p _ { \varepsilon } ,$   
, with probability $1 - p _ { \varepsilon }$   
7: return $T _ { u } \gets ( \ell _ { u } , r _ { u } , Y _ { u } )$

The algorithm considers each depth $\ell \in [ L ]$ exactly once and adds at most one node during that iteration. Thus, $\mathcal { C } ( j )$ contains at most one node at every depth, proving (2). It also follows that $\begin{array} { r } { | \mathcal { C } ( j ) | = \sum _ { \ell \in [ L ] } b _ { \ell } \leq L } \end{array}$

Finally, the algorithm performs L iterations, with a constant amount of operations at every iteration. Hence, it constructs $\mathcal { C } ( j )$ in $O ( L )$ time, which completes the proof. □

The following is the standard orthogonality identity for Hadamard characters: distinct characters have zero average product, while each character has average squared value one.

Claim C.1 (Orthogonality of Hadamard characters, Theorem 1.5 of O’Donnell [ODo14]). Let $\ell \in \mathbb { N } .$ and $u , v \in \{ 0 , \cdot \cdot \cdot , 2 ^ { \ell } - 1 \}$ , then Had<sub>ℓ</sub> satisfies the orthogonality identity:

$$
{ \frac { 1 } { 2 ^ { \ell } } } \sum _ { r = 0 } ^ { 2 ^ { \ell } - 1 } { \mathrm { H a d } } _ { \ell } ( v , r ) { \mathrm { H a d } } _ { \ell } ( u , r ) = \mathbf { 1 } \{ v = u \} .\tag{C.1}
$$

Proof of Theorem 6.1. Now, we prove the utility and privacy gaurantee of our lazy PrefixQuery Algorithm 1.

Set $L : = \lceil \log _ { 2 } D \rceil$ . We first prove (1). Fix a user $u \in [ n ]$ , and let $T _ { u } : = \mathcal { M } ( X _ { u } , \varepsilon , D )$ . Note that the depth $\ell _ { u }$ and Hadamard coordinate $r _ { u }$ are sampled independently of the input. Therefore, it is enough to compare the conditional probabilities of the reported bit $Y _ { u }$ . For any $x , x ^ { \prime } \in [ D ]$ and any possible transcript $( \ell , r , y )$

$$
{ \frac { \mathbb { P } \left[ T _ { u } = ( \ell , r , y ) ~ | ~ X _ { u } = x \right] } { \mathbb { P } \left[ T _ { u } = ( \ell , r , y ) ~ | ~ X _ { u } = x ^ { \prime } \right] } } = { \frac { \mathbb { P } \left[ Y _ { u } = y ~ | ~ \ell _ { u } = \ell , r _ { u } = r , X _ { u } = x \right] } { \mathbb { P } \left[ Y _ { u } = y ~ | ~ \ell _ { u } = \ell , r _ { u } = r , X _ { u } = x ^ { \prime } \right] } } .
$$

For any $( \ell , r )$ , the unperturbed signs corresponding to x and $x ^ { \prime }$ are $b = \mathrm { H a d } _ { \ell } ( v _ { \ell } ( x ) , r )$ and $b ^ { \prime } =$ $\mathrm { H a d } _ { \ell } ( v _ { \ell } ( x ^ { \prime } ) , r )$ , respectively. I ${ \mathrm { ~ f ~ } } b = b ^ { \prime }$ , then the two conditional distributions are identical. $\operatorname { I f } b = - b ^ { \prime }$ then the largest possible likelihood ratio is $\frac { p _ { \varepsilon } } { 1 - p _ { \varepsilon } } = e ^ { \varepsilon }$ , which proves that M is ε-LDP.

Also, note that each user produces its transcript $T _ { u }$ using only its own input and independent local randomness. All calls to PrefixQuery are performed by the server after the transcript $\operatorname { T r } _ { \mathcal { M } } ( X ) =$

$( T _ { 1 } , \ldots , T _ { n } )$ has been collected. These calls only post-process $\operatorname { T r } _ { \mathcal { M } } ( X )$ , so the protocol is noninteractive $\varepsilon { \mathrm { - L D P } }$ , regardless of the number of queries.

Now, we prove (2). The prefixes $j = 0$ and $j = D$ are exact by definition. Let $j \in \{ 1 , \cdots , D - 1 \}$ be any remaining query index, and $x \in [ D ]$ be any domain point. Also, let $\mathcal { C } ( j ) = \{ ( \ell _ { i } , v _ { i } ) : i \in [ k ] \}$ and

$$
Z _ { u } : = \frac { L } { \theta _ { \varepsilon } } \sum _ { ( \ell _ { i } , v _ { i } ) \in \mathcal { C } ( j ) } \mathbf { 1 } \{ \ell _ { u } = \ell _ { i } \} Y _ { u } \operatorname { H a d } _ { \ell _ { i } } ( v _ { i } , r _ { u } )
$$

be defined as in Algorithm 1.

We show that:

(I) $\mathbb { E } [ Z _ { u } \mid X _ { u } = x ] = { \bf 1 } \{ x \leq j \} ;$

(II) $| Z _ { u } | \le L / \theta _ { \varepsilon }$ almost surely.

Here, all expectations are over the depth sampling $( \ell _ { u } )$ , the Hadamard-coordinate sampling $\left( r _ { u } \right)$ and randomized response $( Y _ { u } )$

Conditioned on $X _ { u } = x , \ell _ { u } = \ell ;$ , and $r _ { u } = r$ , randomized response satisfies:

$$
\operatorname { \mathbb { E } } [ Y _ { u } \mid X _ { u } = x , \ell _ { u } = \ell , r _ { u } = r ] = \theta _ { \varepsilon } { \mathrm { H a d } } _ { \ell } ( v _ { \ell } ( x ) , r ) .
$$

Since $\ell _ { u }$ is uniform over $\{ 1 , \ldots , L \}$ , it follows from Hadamard orthogonality (Claim C.1) that

$$
\begin{array} { r l } & { \mathbb { E } [ Z _ { u } \mid \boldsymbol { X } _ { u } = \boldsymbol { x } ] = \displaystyle \frac { L } { \theta _ { \varepsilon } } \sum _ { ( \ell _ { i } , v _ { i } ) \in \mathcal { C } ( j ) } \frac { 1 } { L } \frac { 1 } { 2 ^ { \ell _ { i } } } \sum _ { r = 0 } ^ { 2 ^ { \ell _ { i } } - 1 } \theta _ { \varepsilon } \mathrm { H a d } _ { \ell _ { i } } ( v _ { \ell _ { i } } ( \boldsymbol { x } ) , r ) \mathrm { H a d } _ { \ell _ { i } } ( v _ { i } , r ) } \\ & { \quad \quad \quad \quad = \displaystyle \sum _ { ( \ell _ { i } , v _ { i } ) \in \mathcal { C } ( j ) } \mathbf { 1 } \{ v _ { \ell _ { i } } ( \boldsymbol { x } ) = v _ { i } \} . } \end{array}
$$

Recall that $\begin{array} { r } { v _ { \ell _ { i } } ( x ) = \left\lfloor \frac { x - 1 } { 2 ^ { L - \ell _ { i } } } \right\rfloor } \end{array}$ denotes the index of the unique $\mathrm { d e p t h } { \cdot } \ell _ { i }$ node containing x. Since the intervals in $\mathcal { C } ( j )$ are disjoint and their union is $[ 1 , j ]$ , the last expression equals $\mathbf { 1 } \{ x \leq j \}$ , proving (I).

By Lemma 5.2, the cover contains at most one node at every depth. Since the user samples only one depth, at most one term in the definition of $Z _ { u }$ can be nonzero. Furthermore, $Y _ { u }$ and every Hadamard entry belong to $\{ - 1 , + 1 \}$ . Hence $| Z _ { u } | \le L / \theta _ { \varepsilon }$ , proving (II).

For the fixed dataset X, the variables $Z _ { 1 } , \cdots , Z _ { n }$ are independent, although they need not be identically distributed. Let $\widetilde { Z } _ { j } = \frac { 1 } { n } \sum _ { u \in [ n ] } Z _ { u }$ . By (I),

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left[ \widetilde { Z } _ { j } \mid X \right] = \frac { 1 } { n } \sum _ { u \in [ n ] } \mathbb { E } [ Z _ { u } \mid X _ { u } ] } \\ { = \frac { 1 } { n } \sum _ { u \in [ n ] } \mathbf { 1 } \{ X _ { u } \leq j \} = Z _ { j } ( X ) . } \end{array}
$$

Using (II) and Hoefding’s inequality for independent bounded random variables implies:

$$
\mathbb { P } \left[ | \widetilde { Z } _ { j } - Z _ { j } ( \boldsymbol { X } ) | > \alpha \mid \boldsymbol { X } \right] \le 2 \exp \left( - \frac { 2 n ^ { 2 } \alpha ^ { 2 } } { \sum _ { \boldsymbol { u } \in [ n ] } ( 2 L / \theta _ { \varepsilon } ) ^ { 2 } } \right)
$$

$$
= 2 \exp \left( - \frac { n \theta _ { \varepsilon } ^ { 2 } \alpha ^ { 2 } } { 2 L ^ { 2 } } \right) .
$$

Taking a union bound over the at most D prefix queries results in:

$$
\mathbb { P } \left[ \operatorname* { m a x } _ { j \in \{ 1 , \cdots , D - 1 \} } | \widetilde { Z } _ { j } - Z _ { j } ( X ) | > \alpha \mid X \right] \le 2 D \exp \left( - \frac { n \theta _ { \varepsilon } ^ { 2 } \alpha ^ { 2 } } { 2 L ^ { 2 } } \right) .
$$

Since $\theta _ { \varepsilon } \geq \varepsilon / 4$ for $0 < \varepsilon \le 1$ and $L \leq 3 \log D$ for $D \geq 2$ , the right-hand side is at most $\beta$ under the stated sample-size condition. Finally, clipping ${ \tilde { Z } } _ { j }$ to $[ 0 , 1 ]$ cannot increase its distance from $Z _ { j } ( X ) \in [ 0 , 1 ]$ , which concludes the proof for (2).

Finally, we prove (3). Each user constructs and stores its report in $O ( 1 )$ time and $O ( 1 )$ words, so user preprocessing takes $O ( n )$ total time. For each prefix query, the server constructs the dyadic cover in $O ( L )$ time and scans the n reports in $O ( n )$ time, using a cover representation indexed by depth. Thus, one query takes $O ( n + \log D )$ time, and $q \geq 1$ queries take $O ( q ( n + \log D ) )$ total time including user preprocessing, without materializing the full tree or a length-D vector. □

Lemma C.2 $( L _ { 1 }$ Stability of Discretization). Let $R , \delta > 0$ be parameters, and let P and $P ^ { \prime }$ be two probability distributions over R. Then,

$$
\begin{array} { r } { \| \mathrm { D i s c } _ { R , { \delta } } ( P ) - \mathrm { D i s c } _ { R , { \delta } } ( P ^ { \prime } ) \| _ { 1 } \leq \| P - P ^ { \prime } \| _ { 1 } . } \end{array}
$$

Proof. Let $B _ { j } { } ^ { \dag } \mathrm { s }$ be the intervals from Definition 6.2. Since the intervals $B _ { j }$ ’s are pairwise disjoint, we have:

$$
\begin{array} { l l } { \displaystyle \| \mathrm { D i s c } _ { R , \delta } ( P ) - \mathrm { D i s c } _ { R , \delta } ( P ^ { \prime } ) \| _ { 1 } = \sum _ { j = 1 } ^ { D } | P ( B _ { j } ) - P ^ { \prime } ( B _ { j } ) | } \\ { \displaystyle \leq \| P - P ^ { \prime } \| _ { 1 } , } \end{array}
$$

where the inequality follows because $( B _ { j } ) _ { j = 1 } ^ { D }$ is a measurable partition of R.

Proof of Theorem 6.3. Putting together, we are ready to prove the utility and privacy guarantee of our LDP synthetic data generator in Algorithm 2.

Set $D : = \left\lceil 2 R / \delta \right\rceil + 2$ and $L : = \lceil \log _ { 2 } D \rceil$ . Since $\delta \le R / 2$ , we have $R / \delta \geq 2$ , and $\begin{array} { r } { \frac { 2 R } { \delta } \leq D \leq \frac { 4 R } { \delta } } \end{array}$ . In particular, log $D = O ( \log ( R / \delta ) )$ and $L = O ( \log ( R / \delta ) )$

Let the real line partitioning $B _ { 1 } = ( - \infty , - R ) , B _ { j } = [ c _ { j - 1 } , c _ { j } ) , B _ { D } = [ R , + \infty )$ , be as in Definition 6.2, where $c _ { 0 } = - R - \delta , c _ { j } = - R + \operatorname* { m i n } \{ ( j - 1 ) \delta , 2 R \} , c _ { D } = R + \delta$

We first prove (1). Each user computes the interval index $\widetilde { X } _ { u } = \mathrm { D i s c } _ { R , \delta } ( X _ { u } )$ locally and applies the randomizer in Algorithm 4. For any two real inputs, their indices belong to [D], so Theorem 6.1 proves ε-local diferential privacy. The protocol is non-interactive, and all remaining steps are post-processing of the transcripts Tr.

Now, we prove (2). Let $\widetilde { P } = \mathrm { D i s c } _ { R , \delta } ( P )$ , and write $F _ { \widetilde { P } } ( x ) = \mathbb { P } _ { X \sim \widetilde { P } } ( X \le x )$ for its CDF. By Definition 6.2, $F _ { \widetilde { P } }$ is continuous and linear on each interval $[ c _ { j - 1 } , c _ { j } ]$ , and

$$
F _ { \widetilde { P } } ( c _ { j } ) = \sum _ { i = 1 } ^ { j } P ( B _ { i } ) = \mathbb { P } _ { X \sim P } \left[ \mathrm { D i s c } _ { R , \delta } ( X ) \le j \right] , \qquad j \in \{ 0 , \cdots , D \} .
$$

Let $\widetilde { X } = \left( \widetilde { X } _ { 1 } , \cdots , \widetilde { X } _ { n } \right)$ be the discretized dataset, and define its prefix frequencies by

$$
Z _ { j } \left( \widetilde { X } \right) : = \frac { 1 } { n } \sum _ { u \in [ n ] } \mathbf { 1 } \left\{ \widetilde { X } _ { u } \leq j \right\} , \qquad j \in \{ 0 , \cdots , D \} .
$$

Let $\alpha ^ { \prime } : = \alpha / 2$ . Conditioned on any realization of $\widetilde { X }$ , Theorem 6.1, with accuracy parameter $\alpha ^ { \prime } / 2$ and failure probability $\beta / 2$ , gives

$$
\operatorname* { m a x } _ { j \in \{ 0 , \cdots , D \} } \left| { \widehat { Z } } _ { j } - Z _ { j } \left( { \widetilde { X } } \right) \right| \leq { \frac { \alpha ^ { \prime } } { 2 } }
$$

with probability at least $1 - \beta / 2$ over the local randomizers, provided that $\begin{array} { r } { n \geq \frac { 2 8 8 ( \log D ) ^ { 2 } } { \varepsilon ^ { 2 } ( \alpha ^ { \prime } / 2 ) ^ { 2 } } \log \left( \frac { 4 D } { \beta } \right) } \end{array}$ Since the prefix-oracle guarantee holds for every fixed dataset, the same probability bound holds over the joint sampling and local randomness.

Furthermore, since $X _ { 1 } , \cdots , X _ { n }$ are drawn i.i.d. from $P ,$ we have $\mathbb { E } _ { X \sim P } \left[ Z _ { j } \left( \mathrm { D i s c } _ { R , \delta } ( X ) \right) \right] \ =$ $\mathbb { E } _ { X \sim P } \left[ Z _ { j } \left( \widetilde { X } \right) \right] = F _ { \widetilde { P } } ( c _ { j } )$ . Using Hoefding’s inequality, followed by a union bound over the prefixes, gives

$$
\mathbb { P } \left[ \operatorname* { m a x } _ { j \in \{ 0 , \cdots , D \} } \left| Z _ { j } \left( \widetilde { X } \right) - F _ { \widetilde { P } } ( c _ { j } ) \right| > \frac { \alpha ^ { \prime } } { 2 } \right] \leq 2 D \exp \left( - \frac { n ( \alpha ^ { \prime } ) ^ { 2 } } { 2 } \right) \leq \frac { \beta } { 2 }
$$

Combining the two bounds, with probability at least $1 - \beta$ over the sampling and local randomizers, the triangle inequality gives

$$
\left| \widehat { Z } _ { j } - F _ { \widetilde { P } } ( c _ { j } ) \right| \leq \left| \widehat { Z } _ { j } - Z _ { j } \left( \widetilde { X } \right) \right| + \left| Z _ { j } \left( \widetilde { X } \right) - F _ { \widetilde { P } } ( c _ { j } ) \right| \leq \frac { \alpha ^ { \prime } } { 2 } + \frac { \alpha ^ { \prime } } { 2 } = \alpha ^ { \prime }
$$

for every $j \in \{ 0 , \cdots , D \}$ simultaneously. We condition on this event throughout the rest of the proof. Furthermore, $\hat { Z } _ { 0 } = 0$ and $\widehat { Z } _ { D } = 1$ , but the $\hat { Z } _ { j } { ' } \mathrm { s }$ are not necessarily monotone.

For each $\ell \in [ M ]$ , the binary search maintains the condition $A = \widehat { Z } _ { a } < u _ { \ell } \leq \widehat { Z } _ { b } = B$ . At termination, $a = b - 1$ , so that $B - A > 0$ and $\lambda _ { \ell } = ( u _ { \ell } - A ) / ( B - A ) \in ( 0 , 1 ]$ . Thus, the interpolation step is well-defined, and it holds that $u _ { \ell } = ( 1 - \lambda _ { \ell } ) \widehat { Z } _ { b - 1 } + \lambda _ { \ell } \widehat { Z } _ { b } ,$ , and $F _ { \widetilde { P } } ( z _ { \ell } ) = ( 1 - \lambda _ { \ell } ) F _ { \widetilde { P } } ( c _ { b - 1 } ) + \lambda _ { \ell } F _ { \widetilde { P } } ( c _ { b } )$ Therefore, the simultaneous accuracy bound implies:

$$
\begin{array} { r } { | F _ { \widetilde { P } } ( z _ { \ell } ) - u _ { \ell } | \leq ( 1 - \lambda _ { \ell } ) | F _ { \widetilde { P } } ( c _ { b - 1 } ) - \widehat { Z } _ { b - 1 } | + \lambda _ { \ell } | F _ { \widetilde { P } } ( c _ { b } ) - \widehat { Z } _ { b } | \leq \alpha ^ { \prime } . } \end{array}
$$

Write $F _ { H }$ for the CDF of H. Fix any $x \in \mathbb { R }$ . By the preceding accuracy bound and monotonicity of $F _ { \widetilde { P } } , { \mathrm { ~ i f ~ } } z _ { \ell } \leq x$ , then $u _ { \ell } \le F _ { \widetilde { P } } ( z _ { \ell } ) + \alpha ^ { \prime } \le F _ { \widetilde { P } } ( x ) + \alpha ^ { \prime }$ . Similarly, if $z _ { \ell } > x$ , then $u _ { \ell } \ge F _ { \widetilde { P } } ( z _ { \ell } ) - \alpha ^ { \prime } \ge$ $F _ { \widetilde { P } } ^ { ' } ( x ) - \alpha ^ { \prime }$

Since $u _ { \ell } = ( \ell - 1 / 2 ) / M$ , these inequalities give

$$
{ \cal F } _ { H } ( x ) \le { \frac { 1 } { M } } \sum _ { \ell \in [ M ] } { { \bf 1 } \{ u _ { \ell } \le F _ { \widetilde { \cal P } } ( x ) + \alpha ^ { \prime } \} } \le { \cal F } _ { \widetilde { \cal P } } ( x ) + \alpha ^ { \prime } + { \frac { 1 } { 2 M } } ,
$$

$$
1 - F _ { H } ( x ) \leq \frac { 1 } { M } \sum _ { \ell \in [ M ] } { \mathbf { 1 } \{ u _ { \ell } \geq F _ { \widetilde { P } } ( x ) - \alpha ^ { \prime } \} } \leq 1 - F _ { \widetilde { P } } ( x ) + \alpha ^ { \prime } + \frac { 1 } { 2 M } .
$$

Putting together, gives

$$
\| F _ { H } - F _ { \widetilde { P } } \| _ { \infty } = \operatorname* { s u p } _ { x \in \mathbb { R } } | F _ { H } ( x ) - F _ { \widetilde { P } } ( x ) | \le \alpha ^ { \prime } + \frac { 1 } { 2 M } \le \frac { \alpha } { 2 } + \frac { \alpha } { 2 } = \alpha .
$$

This argument does not require the estimated prefix values to be monotone.

Finally, to see (3), computing the interval index and generating the local randomized response each take $O ( 1 )$ time per user. Thus, user preprocessing takes $O ( n )$ total time. Generating the synthetic dataset requires M binary searches, each with at most L prefix queries and constant additional work per step. By Theorem 6.1, each query takes $O ( n + L )$ time. Thus, the total run-time of the algorithm is

$$
O \left( n + M L ( n + L ) \right) = O \left( \frac { 1 } { \alpha } \log { \left( \frac { R } { \delta } \right) } \left( n + \log { \left( \frac { R } { \delta } \right) } \right) \right) ,
$$

where we used $M = \lceil 1 / \alpha \rceil = O ( 1 / \alpha )$ and $L = O ( \log ( R / \delta ) )$ . The assumption $\delta \le R / 2$ is used only to simplify the bounds. □

## D Proof of the main theorem from section 7

Proof of Theorem 7.1. We now prove the guarantees of our LDP agnostic learning algorithm.

Let $C \geq 3$ be a universal constant for which Proposition 5.4 holds, and set $t = ( C + 1 ) ( d + 1 ) \varepsilon$ , and $\begin{array} { r } { M = \left\lceil \frac { 2 t } { \alpha } \right\rceil } \end{array}$

Let $\widetilde { P } : = \operatorname { D i s c } _ { R , \delta } ( P )$ with support $I _ { R , \delta } : = [ - R - \delta , R + \delta ]$ . Using Theorem 6.3, with accuracy parameter $\alpha / ( 2 t )$ , the stated sample size sufices to obtain a synthetic multiset $\mathcal { Z } \subseteq I _ { R , \delta }$ of size M, with empirical probability measure H, such that

$$
\left\| F _ { H } - F _ { \widetilde { P } } \right\| _ { \infty } \le \frac { \alpha } { 2 t } = \frac { \alpha } { 2 ( C + 1 ) ( d + 1 ) s }\tag{D.1}
$$

with probability at least $1 - \beta$ . We condition on this event for the rest of the proof.

Run the deterministic algorithm from Proposition 5.4 on $I _ { R , \delta } , \mathcal { Z } , s , d _ { \colon }$ , with parameter $\eta = \alpha$ . Since $\widetilde { P }$ has a density supported on $I _ { R , \delta }$ , the Kolmogorov bound (D.1) satisfies the requirement of the proposition for $P ^ { \star } = \widetilde { P } ^ { }$ . Thus, the output of the deterministic piece-wise polynomial learner $Q \in \mathcal { P } _ { C s , d }$ is nonnegative and satisfies:

$$
\| Q - \widetilde { P } \| _ { 1 } \leq 4 \operatorname* { i n f } _ { P ^ { \prime } \in \mathcal { P } _ { s , d } } \| P ^ { \prime } - \widetilde { P } \| _ { 1 } + 4 \alpha .\tag{D.2}
$$

Fix any $F \in { \mathcal { F } }$ . We can write:

$$
\begin{array} { r } { \| F - \widetilde { P } \| _ { 1 } \leq \| F - \mathrm { D i s c } _ { R , \delta } ( F ) \| _ { 1 } + \| \mathrm { D i s c } _ { R , \delta } ( F ) - \mathrm { D i s c } _ { R , \delta } ( P ) \| _ { 1 } } \end{array}\tag{D.3}
$$

$$
\leq \mathrm { D i s c - E r r } _ { R , \delta } ( \mathcal { F } ) + \| F - P \| _ { 1 } ,\tag{D.4}
$$

where the first inequality follows from the triangle inequality, and the second follows from Definition 2.3 and the fact that Disc $^ { \mathbf { \delta } } R , \delta$ is a Markov operator and does not increase $L _ { 1 }$ distance (see Lemma C.2).

Consequently, we can write:

$$
\begin{array} { r l } & { \underset { P ^ { \prime } \in \mathcal { P } _ { s , d } } { \operatorname* { i n f } } \| P ^ { \prime } - \widetilde { P } \| _ { 1 } \leq \underset { P ^ { \prime } \in \mathcal { P } _ { s , d } } { \operatorname* { i n f } } \| P ^ { \prime } - F \| _ { 1 } + \| F - \widetilde { P } \| _ { 1 } } \\ & { \qquad \leq \mathrm { P o l y – A p p r o x } _ { s , d } ( \mathcal { F } ) + \mathrm { D i s c \mathrm { - } E r r } _ { R , \delta } ( \mathcal { F } ) + \| F - P \| _ { 1 } . } \end{array}
$$

Putting together with (D.2), we get:

$$
\begin{array} { r } { \| Q - \widetilde { P } \| _ { 1 } \leq 4 \alpha + 4 \mathrm { P o l y - A p p r o x } _ { s , d } ( \mathcal { F } ) + 4 \mathrm { D i s c - E r r } _ { R , \delta } ( \mathcal { F } ) + 4 \| F - P \| _ { 1 } . } \end{array}\tag{D.5}
$$

Since $Q$ is a piece-wise polynomial supported on the bounded interval $I _ { R , \delta }$ , it is integrable. Let $\begin{array} { r } { m _ { 0 } : = \int _ { \mathbb { R } } Q ( x ) d x < \infty } \end{array}$ , and define the normalized output distribution by:

$$
\widehat { P } : = \left\{ \begin{array} { l l } { Q / m _ { 0 } , } & { m _ { 0 } > 0 , } \\ { { \mathrm { U n i f } } ( I _ { R , \delta } ) , } & { m _ { 0 } = 0 . } \end{array} \right.
$$

If $m _ { 0 } > 0$ , then

$$
\| \hat { P } - \widetilde { P } \| _ { 1 } \leq \| \hat { P } - Q \| _ { 1 } + \| Q - \widetilde { P } \| _ { 1 } = | 1 - m _ { 0 } | + \| Q - \widetilde { P } \| _ { 1 } \leq 2 \| Q - \widetilde { P } \| _ { 1 } .\tag{D.6}
$$

If $m _ { 0 } = 0$ , then $\| Q - \widetilde { P } \| _ { 1 } = 1$ and $\| \widehat { P } - \widetilde { P } \| _ { 1 } \leq 2$ . Therefore, in both cases, we have $\| \widehat { P } - \widetilde { P } \| _ { 1 } \leq$ $2 \lVert Q - \widetilde P \rVert _ { 1 }$ . Note that $\widehat { P }$ remains a piece-wise polynomial, since normalization preserves the number and degrees of the polynomial pieces, and the fallback density $\mathrm { U n i f } ( I _ { R , \delta } )$ has at most three pieces on R.

Moreover, we can write:

$$
\| \widetilde { P } - P \| _ { 1 } \le \| \widetilde { P } - F \| _ { 1 } + \| F - P \| _ { 1 }\tag{D.7}
$$

$$
\leq \| \mathrm { D i s c } _ { R , \delta } ( P ) - \mathrm { D i s c } _ { R , \delta } ( F ) \| _ { 1 } + \| \mathrm { D i s c } _ { R , \delta } ( F ) - F \| _ { 1 } + \| F - P \| _ { 1 }\tag{D.8}
$$

$$
\leq \| P - F \| _ { 1 } + \| \operatorname { D i s c } _ { R , \delta } ( F ) - F \| _ { 1 } + \| F - P \| _ { 1 }\tag{D.9}
$$

$$
\leq 2 \| F - P \| _ { 1 } + \operatorname { D i s c - E r r } _ { R , \delta } ( \mathcal { F } ) ,\tag{D.10}
$$

where the first two inequalities follow from the triangle inequality, the third follows because Disc $R , \delta$ is a Markov operator and does not increase $L _ { 1 }$ distance (see Lemma C.2), and the last follows from Definition 2.3.

Finally, we can write:

$$
\begin{array} { r l } & { \| \hat { P } - P \| _ { 1 } \leq \| \hat { P } - \widetilde { P } \| _ { 1 } + \| \widetilde { P } - P \| _ { 1 } } \\ & { \qquad \leq 2 \| Q - \widetilde { P } \| _ { 1 } + \| \widetilde { P } - P \| _ { 1 } } \\ & { \qquad \leq 8 \alpha + 8 \| F - P \| _ { 1 } + 8 \mathrm { D i s c } \mathrm { - } \mathrm { E r r } _ { R , \delta } ( \mathcal { F } ) + 8 \mathrm { P o l y } \mathrm { - } \mathrm { A p p r o x } _ { s , d } ( \mathcal { F } ) + \| \widetilde { P } - P \| _ { 1 } } \\ & { \qquad \leq 8 \alpha + 1 0 \| F - P \| _ { 1 } + 9 \mathrm { D i s c } \mathrm { - } \mathrm { E r r } _ { R , \delta } ( \mathcal { F } ) + 8 \mathrm { P o l y } \mathrm { - } \mathrm { A p p r o x } _ { s , d } ( \mathcal { F } ) , } \end{array}
$$

where the first inequality follows from the triangle inequality, the second follows from (D.6), the third follows from (D.5), and the last follows from (D.10). Taking the infimum over $F \in { \mathcal { F } }$ gives

$$
\| \hat { P } - P \| _ { 1 } \leq 1 0 \operatorname* { i n f } _ { F \in \mathcal { F } } \| F - P \| _ { 1 } + 8 \alpha + 9 \mathrm { D i s c } . \mathrm { E r r } _ { R , \delta } ( \mathcal { F } ) + 8 \mathrm { P o l y } . \mathrm { A p p r o x } _ { s , d } ( \mathcal { F } ) .
$$

Privacy and non-interactivity follow from Theorem 6.3, since fitting and normalization are postprocessing operations. By Theorem 6.3 and Proposition 5.4, the total running time is

$$
O \left( \frac { t } { \alpha } \log \left( \frac { R } { \delta } \right) \left( n + \log \left( \frac { R } { \delta } \right) \right) \right) + \widetilde { O } \left( M ( d + 1 ) ^ { \omega + 2 } \log _ { + } ^ { 3 } \left( \frac { s } { \alpha } \right) \left( 1 + \log _ { + } \left( \frac { M } { s } \right) \right) \right) ,
$$

where log ${ \bf \Phi } _ { + } ( x ) : = \operatorname* { m a x } \{ 1 , \log x \}$ , and the $\widetilde O$ term hides only logarithmic factors in M and $d + 1$

For the chosen sample size, $n = \Omega ( \log ( R / \delta ) )$ , so $n + \log ( R / \delta ) = O ( n ) , t = O ( s ( d + 1 ) )$ ， $M =$ $O ( s ( d + 1 ) / \alpha )$ , and $M / s = O ( ( d + 1 ) / \alpha )$ . Absorbing only logarithmic factors in $s , d + 1$ , and $1 / \alpha$ therefore gives the stated runtime bound in the theorem. Integrating and normalizing the polynomial pieces takes $O ( s ( d + 1 ) )$ additional operations, which is absorbed in the bound. □

## E Omitted proofs from section 8

Proof of Lemma 8.2. We prove that bounding the polynomial approximation and discretization errors of mixture classes reduces to bounding the corresponding errors for the single-component classes.

Fix an arbitrary mixture $\begin{array} { r } { F = \sum _ { i \in [ m ] } w _ { i } F _ { i } \in m { \mathrm { - m i x } } ( { \mathcal F } ) } \end{array}$

First we prove (1). By Definition 6.2, Dis $\begin{array} { r } { { \mathrm { : } } _ { R , \delta } ( F ) = \sum _ { i \in [ m ] } w _ { i } \operatorname { D i s c } _ { R , \delta } ( F _ { i } ) } \end{array}$ . Therefore, the triangle inequality gives

$$
\begin{array} { r l } {  { \| \mathrm { D i s c } _ { R , \delta } ( F ) - F \| _ { 1 } \le \sum _ { i \in [ m ] } w _ { i } \| \mathrm { D i s c } _ { R , \delta } ( F _ { i } ) - F _ { i } \| _ { 1 } } } \\ & { \le \mathrm { D i s c } - \mathrm { E r r } _ { R , \delta } ( \mathcal { F } ) . } \end{array}
$$

Taking the supremum over $F \in \operatorname* { m - m i x } ( { \mathcal { F } } )$ proves one direction. The reverse inequality follows from ${ \mathcal { F } } \subseteq m { \mathrm { - m i x } } ( { \mathcal { F } } )$

Next, we show (2). Fix $\eta > 0$ . For each $i \in [ m ]$ , choose $P _ { i } \in \mathcal { P } _ { s , d }$ satisfying

$$
\begin{array} { r } { \| P _ { i } - F _ { i } \| _ { 1 } \leq \mathrm { P o l y - A p p r o x } _ { s , d } ( \mathcal { F } ) + \eta , } \end{array}
$$

and set $\begin{array} { r } { P : = \sum _ { i \in [ m ] } w _ { i } P _ { i } } \end{array}$ . Each $P _ { i }$ has at most $s - 1$ breakpoints. Their common refinement therefore has at most $m ( s { \ - } - 1 ) + 1$ interval pieces, and taking weighted sums does not increase the degree of the underlying polynomials. Since $P$ is nonnegative and integrable, we have $P \in \mathcal P _ { m ( s - 1 ) + 1 , d } .$ Moreover,

$$
\| P - F \| _ { 1 } \leq \sum _ { i \in [ m ] } w _ { i } \| P _ { i } - F _ { i } \| _ { 1 } \leq \mathrm { P o l y } \mathrm { - A p p r o x } _ { s , d } ( \mathcal { F } ) + \eta .
$$

Consequently,

$$
\operatorname* { i n f } _ { P \in \mathcal { P } _ { m ( s - 1 ) + 1 , d } } \| P - F \| _ { 1 } \leq \mathrm { P o l y - A p p r o x } _ { s , d } ( \mathcal { F } ) + \eta .
$$

Taking the supremum over $F \in \operatorname* { m - m i x } ( { \mathcal { F } } )$ and letting η tend to zero proves the result. □

Proof of Lemma 8.7. Next, we prove a bound on the discretization error of log-concave distributions. Fix an arbitrary distribution $F \in \mathcal { L } ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } )$ , and let f be its density, with mean $\mu$ and standard deviation σ.

Define $g ( x ) : = \sigma f ( \mu + \sigma x )$ for $x \in \mathbb { R }$ . The function g is log-concave, and the change of variables $y = \mu + \sigma x$ gives

$$
\begin{array} { r l } & { \displaystyle \int _ { \mathbb R } g ( x ) d x = \int _ { \mathbb R } f ( y ) d y = 1 , } \\ & { \displaystyle \int _ { \mathbb R } x g ( x ) d x = \frac 1 { \sigma } \int _ { \mathbb R } ( y - \mu ) f ( y ) d y = 0 , } \\ & { \displaystyle \int _ { \mathbb R } x ^ { 2 } g ( x ) d x = \frac 1 { \sigma ^ { 2 } } \int _ { \mathbb R } ( y - \mu ) ^ { 2 } f ( y ) d y = 1 . } \end{array}
$$

Thus, g is a log-concave probability density with mean zero and variance one. By Lemma $5 . 5 ( \mathrm { a } )$ of Lovász and Vempala [LV07], $\textstyle \operatorname* { s u p } _ { x \in \mathbb { R } } g ( x ) \leq 1$ . Since $\begin{array} { r } { f ( x ) = \frac { 1 } { \sigma } g \left( \frac { x - \mu } { \sigma } \right) } \end{array}$ , we obtain $\operatorname* { s u p } _ { x \in \mathbb { R } } f ( x ) =$ $\begin{array} { r } { \frac { 1 } { \sigma } \operatorname* { s u p } _ { z \in \mathbb { R } } g ( z ) \leq \frac { 1 } { \sigma } } \end{array}$

We next bound the total tail mass. For $X \sim f .$ , the random variable $Y = ( X - \mu ) / \sigma$ has density $g .$ Since $\mathbb { E } [ Y ^ { 2 } ] = 1$ , Lemma 5.17 of Lovász and Vempala [LV07] gives, for every $t > 1$

$$
\mathbb { P } \left[ \left| X - \mu \right| > t \sigma \right] = \mathbb { P } \left[ \left| Y \right| > t \right] \le e ^ { 1 - t } .
$$

Since $| \mu | \leq R$ and $\sigma \le \sigma _ { \mathrm { m a x } }$ , it follows that

$$
\mathbb { P } [ | X | > T ] \le \mathbb { P } \left[ \frac { | X - \mu | } { \sigma } > \frac { T - R } { \sigma _ { \operatorname* { m a x } } } \right] \le \exp \left( 1 - \frac { T - R } { \sigma _ { \operatorname* { m a x } } } \right) = \frac { \eta } { 4 } .
$$

Therefore, $F ( \mathbb { R } \setminus [ - T , T ] ) \leq \eta / 4$

Let $D = J + 2$ , and let $B _ { j } \mathrm { ^ { * } s }$ and $c _ { j } \mathrm { ^ { * } s }$ be the intervals and endpoints from Definition 6.2, with parameters $( T , \delta )$ . For each $j \in \{ 2 , \cdots , D - 1 \}$ , let $h _ { j } = c _ { j } - c _ { j - 1 } \leq \delta$ and

$$
\bar { f } _ { j } : = \frac { 1 } { h _ { j } } \int _ { B _ { j } } f ( y ) d y .
$$

For every $x \in B _ { j }$ , we have

$$
{ \begin{array} { r l } { | f ( x ) - { \bar { f } } _ { j } | \leq { \frac { 1 } { h _ { j } } } \int _ { B _ { j } } | f ( x ) - f ( y ) | d y } \\ { \quad } & { \leq \operatorname* { s u p } _ { y \in B _ { j } } f ( y ) - \operatorname* { i n f } _ { y \in B _ { j } } f ( y ) . } \end{array} }
$$

Since $f$ is nonnegative and unimodal, the sum of its binwise diferences between the supremum and infimum is bounded by its total increase plus its total decrease, each of which is at most $\operatorname { s u p } _ { x \in \mathbb { R } } f ( x )$ Therefore,

$$
\sum _ { j = 2 } ^ { D - 1 } \left( \operatorname* { s u p } _ { x \in B _ { j } } f ( x ) - \operatorname* { i n f } _ { x \in B _ { j } } f ( x ) \right) \leq 2 \operatorname* { s u p } _ { x \in \mathbb { R } } f ( x ) \leq { \frac { 2 } { \sigma } } .
$$

Integrating over $x \in B _ { j }$ and summing over the internal intervals yields

$$
\sum _ { j = 2 } ^ { D - 1 } \int _ { B _ { j } } | f ( x ) - \bar { f } _ { j } | d x \leq \delta \sum _ { j = 2 } ^ { D - 1 } \left( \operatorname* { s u p } _ { x \in B _ { j } } f ( x ) - \operatorname* { i n f } _ { x \in B _ { j } } f ( x ) \right) \leq \frac { 2 \delta } { \sigma } \leq \frac { 2 \delta } { \sigma _ { \operatorname* { m i n } } } .
$$

The original and discretized measures restricted to the two tails each have total mass $F ( \mathbb { R } \setminus [ - T , T ] )$ Their $L _ { 1 }$ diference is therefore at most twice this mass. Combining the two bounds gives

$$
\| \mathrm { D i s c } _ { T , \delta } ( F ) - F \| _ { 1 } \le \frac { 2 \delta } { \sigma _ { \operatorname* { m i n } } } + \frac { \eta } { 2 } = \frac { 4 T } { J \sigma _ { \operatorname* { m i n } } } + \frac { \eta } { 2 } \le \eta .
$$

Taking the supremum over F proves the result.

Lemma E.1 (Piece-wise polynomial approximation of log-concave distributions, Lemma 27 of Chan et al. [CDSS14]). For every $\alpha \in ( 0 , 1 )$ , there exists an integer $s _ { \alpha } = \widetilde { O } ( 1 / \sqrt { \alpha } )$ such that, for every log-concave probability density f over R, there exists a (normalized) piece-wise polynomial $p \in \mathcal { P } _ { s _ { \alpha } , 1 }$ satisfying $\| f - p \| _ { 1 } \leq \alpha$ . Here, Oe hides only logarithmic factors in $1 / \alpha$

Proof of Lemma 8.6. We prove a bound on the piece-wise polynomial approximation of log-concave distributions.

Let $s _ { \alpha }$ be the integer from Lemma E.1. For every $f \in \mathcal L ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } )$ , that lemma gives a normalized $P \in \mathcal { P } _ { s _ { \alpha } , 1 }$ satisfying $\| P - f \| _ { 1 } \leq \alpha$ . Therefore, $\mathrm { i n f } _ { P \in { \mathcal { P } } _ { s _ { \alpha } , 1 } } \| P - f \| _ { 1 } \leq \alpha$ . Taking the supremum over $f \in \mathcal { L } ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } )$ proves the result. □

Proof of Theorem 8.5. Putting together, we obtain our upper bound for LDP agnostic learning log-concave distributions.

Let $\mathcal { F } = m \mathrm { - m i x } \left( \mathcal { L } ( R , \sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } } ) \right)$ be the class of log-concave mixtures. Set $\begin{array} { r } { \rho = \frac { \alpha } { 2 5 } , s _ { 0 } = m ( s _ { \rho } - } \end{array}$ \~ $1 ) + 1 , d = 1$ , where $s _ { \rho }$ is the integer from Lemma 8.6 at accuracy $\rho .$ Thus, $s _ { 0 } = O ( m / \sqrt { \rho } )$

Also let the discretization parameters $\begin{array} { r } { T = R + \sigma _ { \operatorname* { m a x } } \Big ( 1 + \log \Big ( \frac { 4 } { \rho } \Big ) \Big ) , ~ J = \Big \lceil \frac { 8 T } { \rho \sigma _ { \operatorname* { m i n } } } \Big \rceil } \end{array}$ , and $\begin{array} { r } { \delta = \frac { 2 T } { J } } \end{array}$ be defined as per Lemma 8.7. Since $T > 0$ and $J \geq 4 .$ , we have $0 < \delta \le { T } / 2$ . We can apply Theorem 7.1 with class ${ \mathcal { F } } ,$ , discretization parameters $( T , \delta )$ , privacy parameter $\varepsilon ,$ failure probability $\beta _ { i }$ accuracy parameter $\rho ,$ and polynomial parameters $( s _ { 0 } , d )$ , using sample size $\begin{array} { r } { n = O \left( \frac { s _ { 0 } ^ { 2 } ( d + 1 ) ^ { 2 } } { \varepsilon ^ { 2 } \rho ^ { 2 } } \log ^ { 2 } \left( \frac { T } { \delta } \right) \log \left( \frac { T } { \beta \delta } \right) \right) } \end{array}$ The output then satisfies

$$
\begin{array} { r } { \| \widehat { P } - P \| _ { 1 } \leq 1 0 0 \mathrm { P T } ^ { * } + 8 \rho + 9 \mathrm { D i s c } \mathrm { - E r r } _ { T , \delta } ( \mathcal { F } ) + 8 \mathrm { P o l y } \mathrm { - A p p r o x } _ { s _ { 0 } , d } ( \mathcal { F } ) , } \end{array}
$$

with probability at least $1 - \beta$ , where $\begin{array} { r } { \mathrm { O P T ^ { * } } = \operatorname* { i n f } _ { F \in \mathcal { F } } \| P - F \| _ { 1 } } \end{array}$

By Lemmas 8.6 and 8.2,

$$
\begin{array} { r } { \mathrm { P o l y - A p p r o x } _ { s _ { 0 } , d } ( \mathcal { F } ) \leq \mathrm { P o l y - A p p r o x } _ { s _ { \rho , 1 } } \left( \mathcal { L } ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ) \right) \leq \rho . } \end{array}
$$

Moreover, Lemmas 8.7 and 8.2 give

$$
\operatorname { D i s c - E r r } _ { T , \delta } ( \mathcal F ) = \operatorname { D i s c - E r r } _ { T , \delta } \left( \mathcal L ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ) \right) \le \rho .
$$

Substituting these two bounds gives

$$
\| \widehat { P } - P \| _ { 1 } \leq 1 0 \mathrm { O P T } ^ { * } + 8 \rho + 9 \rho + 8 \rho = 1 0 \mathrm { O P T } ^ { * } + \alpha .
$$

Finally, $s _ { 0 } = \tilde { O } ( m / \sqrt { \rho } ) , d = 1$ , and our parameter choices give $\begin{array} { r } { J = O \left( \frac { R + \sigma _ { \mathrm { m a x } } } { \rho \sigma _ { \mathrm { m i n } } } \log \left( \frac { 4 } { \rho } \right) \right) } \end{array}$ , and log $\begin{array} { r } { \left( \frac { T } { \delta } \right) = O \left( \log \left( \frac { 2 ( R + \sigma _ { \operatorname* { m a x } } ) } { \alpha \sigma _ { \operatorname* { m i n } } } \right) \right) } \end{array}$ . Substituting the chosen parameters into the sample and runningtime bounds of Theorem 7.1 gives the stated complexities, suppressing only logarithmic factors in $1 / \alpha$ in the sample bound, and in m and $1 / \alpha$ in the running time. The output has $\widetilde { O } ( m / \sqrt { \alpha } )$ polynomial pieces and degree at most 1. Privacy and non-interactivity also follow from Theorem 7.1. □

## F Upper bound for GMMs

In this section, we consider agnostic LDP learning of GMMs. Although GMMs are mixtures of log-concave distributions, Gaussian components admit a more specialized piece-wise polynomial approximation, whose number of pieces is independent of the accuracy parameter $\alpha .$ . This allows us to obtain an optimal sample dependence on the accuracy parameter. The class of bounded-parameter Gaussian distributions is defined as follows.

Definition F.1 (Bounded Gaussians). Let $R \geq 0$ , and $0 < \sigma _ { m i n } \leq \sigma _ { m a x } < \infty$ be parameters. Then the class of bounded Gaussians over the real line is defined by:

$$
\mathcal { G } ( R , \sigma _ { m i n } , \sigma _ { m a x } ) = \left\{ \mathcal { N } ( \mu , \sigma ^ { 2 } ) : | \mu | \leq R , \sigma \in [ \sigma _ { m i n } , \sigma _ { m a x } ] \right\} .
$$

The following theorem gives the resulting guarantee for eficient non-interactive LDP agnostic learning of GMMs.

Theorem F.2 (Eficient LDP Agnostic Learning of GMMs). Under the same assumptions of Theorem 8.5, there exists a non-interactive $\varepsilon \mathrm { - } L D P \left( 1 0 , \alpha , \beta \right)$ -agnostic learner for the class ${ \mathit { m } } { - } { \mathit { m i x } } ( \mathcal { G } ( R , \sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } } ) )$ that uses $\begin{array} { r } { n = O \left( \frac { m ^ { 2 } \log ^ { 2 } ( 2 / \alpha ) } { \varepsilon ^ { 2 } \alpha ^ { 2 } } \log ^ { 2 } \left( \frac { 2 ( R + \sigma _ { \operatorname* { m a x } } ) } { \alpha \sigma _ { \operatorname* { m i n } } } \right) \log \left( \frac { 2 ( R + \sigma _ { \operatorname* { m a x } } ) } { \alpha \beta \sigma _ { \operatorname* { m i n } } } \right) \right) } \end{array}$ many samples and its output has a nonnegative piece-wise polynomial density with $O ( m )$ pieces and degree $O ( \log ( 2 / \alpha ) )$ . Moreover, the total running time, is $\begin{array} { r } { \tilde { O } \left( \frac { n m } { \alpha } \log \left( \frac { 2 ( R + \sigma _ { \operatorname* { m a x } } ) } { \alpha \sigma _ { \operatorname* { m i n } } } \right) \right) } \end{array}$ , where $\widetilde O$ hides only logarithmic factors in m and $1 / \alpha$

Proof. Let $\mathcal { F } = m \mathrm { - m i x } \left( \mathcal { G } ( R , \sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } } ) \right)$ be the class of GMMs. Set $\textstyle \rho = { \frac { \alpha } { 2 5 } } , s _ { 0 } = 2 m + 1 , d =$ $\left\lceil C \log \left( { \frac { 2 } { \rho } } \right) \right\rceil$ , where $C > 0$ is the universal constant from Lemma F.5.

Also let the discretization parameters $\begin{array} { r } { T = R + \sigma _ { \operatorname* { m a x } } \left( 1 + \log \left( \frac { 4 } { \rho } \right) \right) , J = \left\lceil \frac { 8 T } { \rho \sigma _ { \operatorname* { m i n } } } \right\rceil } \end{array}$ , and $\begin{array} { r } { \delta = \frac { 2 T } { J } } \end{array}$ be defined as per Lemma F.3. Since $T > 0$ and $J \ge 4$ , we have $0 < \dot { \delta } \le T / 2$ , we can apply Theorem 7.1 with class $\mathcal { F }$ , discretization parameters $( T , \delta )$ , privacy parameter $\varepsilon ,$ failure probability $\beta$ , accuracy parameter $\rho ,$ and polynomial parameters $( s _ { 0 } , d )$ , using sample size $\begin{array} { r } { n = O \left( \frac { s _ { 0 } ^ { 2 } ( d + 1 ) ^ { 2 } } { \varepsilon ^ { 2 } \rho ^ { 2 } } \log ^ { 2 } \left( \frac { T } { \delta } \right) \log \left( \frac { T } { \beta \delta } \right) \right) } \end{array}$ The output then satisfies

$$
\begin{array} { r } { \| \widehat { P } - P \| _ { 1 } \leq 1 0 0 \mathrm { P T } ^ { * } + 8 \rho + 9 \mathrm { D i s c } \mathrm { - E r r } _ { T , \delta } ( \mathcal { F } ) + 8 \mathrm { P o l y } \mathrm { - A p p r o x } _ { s _ { 0 } , d } ( \mathcal { F } ) . } \end{array}
$$

with probability at least $1 - \beta .$ , where $\begin{array} { r } { \mathrm { O P T ^ { * } } : = \operatorname* { i n f } _ { G \in \mathcal { F } } \| P - G \| _ { 1 } } \end{array}$

By Lemmas F.5 and 8.2,

$$
\begin{array} { r } { \mathrm { P o l y - A p p r o x } _ { s _ { 0 } , d } ( \mathcal { F } ) \le \mathrm { P o l y - A p p r o x } _ { 3 , d } \left( \mathcal { G } ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ) \right) \le \rho . } \end{array}
$$

Moreover, Lemmas F.3 and 8.2 give

$$
\operatorname { D i s c - E r r } _ { T , \delta } ( \mathcal { F } ) = \operatorname { D i s c - E r r } _ { T , \delta } \left( \mathcal { G } ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ) \right) \leq \rho .
$$

Substituting these two bounds gives

$$
\| \widehat { P } - P \| _ { 1 } \leq 1 0 \mathrm { O P T } ^ { * } + 8 \rho + 9 \rho + 8 \rho = 1 0 \operatorname* { i n f } _ { G \in \mathcal { F } } \| P - G \| _ { 1 } + \alpha .
$$

Finally, our choices of T and δ give log $\begin{array} { r } { \left( \frac { T } { \delta } \right) \ = \ O \left( \log \left( \frac { 2 ( R + \sigma _ { \operatorname* { m a x } } ) } { \alpha \sigma _ { \operatorname* { m i n } } } \right) \right) } \end{array}$ . Substituting the chosen parameters into the sample and running-time bounds of Theorem 7.1 gives the stated complexities, suppressing only logarithmic factors in m and $1 / \alpha$ in the running time. The output has $O ( m )$ polynomial pieces and degree $O ( \log ( 2 / \alpha ) )$ . Privacy and non-interactivity also follow from Theorem 7.1. □

Lemma F.3 (Discretization of Bounded-Parameter Gaussians). Let $\eta \in ( 0 , 1 ) , \ R \geq 0$ , and $0 < \sigma _ { \mathrm { m i n } } \le \sigma _ { \mathrm { m a x } } < \infty$ be parameters. Let $\begin{array} { r } { T = R + \sigma _ { \operatorname* { m a x } } \left( 1 + \log \left( \frac { 4 } { \eta } \right) \right) , \ J = \left\lceil \frac { 8 T } { \eta \sigma _ { \operatorname* { m i n } } } \right\rceil } \end{array}$ , and $\begin{array} { r } { \delta = \frac { 2 T } { J } } \end{array}$ Then, it holds that $\mathrm { D i s c - E r r } _ { T , \delta } \left( \mathcal { G } ( R , \sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } } ) \right) \leq \eta$

Proof. Every Gaussian density is log-concave. Therefore, ${ \mathcal G } ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ) \subseteq { \mathcal L } ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } )$

By Lemma 8.7, we get Disc-Err $\begin{array} { r } { \cdot _ { T , \delta } \left( \mathcal { G } ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ) \right) \leq \mathrm { D i s c } - \mathrm { E r r } _ { T , \delta } \left( \mathcal { L } ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ) \right) \leq \eta . } \end{array}$

Lemma F.4 (Piece-wise polynomial approximation of Gaussians, Lemma 36 of Chan et al. [CDSS14]). There exists a universal constant $C > 0$ such that, for every $\alpha \in ( 0 , 1 )$ and every Gaussian distribution $g = \mathcal { N } ( \mu , \sigma ^ { 2 } )$ with $\mu \in \mathbb { R }$ and $\sigma > 0$ , there exists a (normalized) piece-wise polynomial $p \in \mathcal { P } _ { s , d }$ with $s = 3$ , and $\begin{array} { r } { d = \lceil C \log ( \frac { 2 } { \alpha } ) \rceil } \end{array}$ satisfying $\| g - p \| _ { 1 } \leq \alpha$

We use the Gaussian approximation bound of Chan et al. [CDSS14] to obtain the following lemma.

Lemma F.5 (Piece-wise polynomial approximation of Gaussians). There exists a universal constant $C > 0$ such that the following holds. Let $\alpha \in ( 0 , 1 ) , R \geq 0$ , and $0 < \sigma _ { \mathrm { m i n } } \le \sigma _ { \mathrm { m a x } } < \infty$ be parameters. Set $s = 3$ , and $\begin{array} { r } { d = \lceil C \log ( \frac { 2 } { \alpha } ) \rceil } \end{array}$ . Then, it holds that $\mathrm { P o l y \mathrm { - } A p p r o x } _ { s , d } \left( \mathcal { G } ( R , \sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } } ) \right) \leq \alpha$

Proof. Let $C > 0$ be the universal constant from Lemma F.4. For every $g \in \mathcal { G } ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } )$ , that lemma gives a normalized $P \in \mathcal { P } _ { s , d }$ satisfying $\| P - g \| _ { 1 } \leq \alpha$ . Therefore, inf ${ P \in { \mathcal { P } } _ { s , d } } \left\| P - g \right\| _ { 1 } \leq \alpha$ Taking the supremum over $g \in \mathcal { G } ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } )$ proves the result. □

## G Lower bound for LDP-learning of GMMs

In this section, we prove a lower bound for non-interactive LDP learning of GMMs. When the ranges of the component means and variances are suficiently large, the lower bound matches the dependence on the number of mixture components, accuracy and privacy parameters of our upper bound in Theorem F.2 up to logarithmic factors. Since every Gaussian distribution is log-concave, this lower bound also applies to mixtures of log-concave distributions with the same component parameter bounds.

In particular, our lower bound improves on the standard packing lower bounds in pure diferential privacy [BKSW19]: packings for mixtures of m components usually have sizes exponential in $m _ { \ i }$ yielding lower bounds of $\Omega ( m )$ , whereas we obtain a lower bound of $\Omega ( m ^ { 2 } )$

We obtain this result by reducing LDP estimation of sparse discrete distributions to GMM learning and applying the lower bound of Acharya et al. [AKLS20]. The main idea is to associate each element of a finite domain with a known Gaussian component, so that a sparse probability vector becomes the vector of mixture weights. We choose components separated in location or scale, with each component placing most of its probability mass in its own disjoint region. This separation ensures that an accurate estimate of the mixture yields an accurate estimate of the sparse probability vector. The reduction preserves local privacy, so a GMM learner would also solve the sparse estimation problem with the same number of samples. The lower bound for sparse estimation therefore gives the desired GMM lower bound.

Theorem G.1 (GMM Lower Bound). Let $\varepsilon \in ( 0 , 1 ]$ and $\alpha \in ( 0 , \alpha _ { 0 } ]$ for a suficiently small universal constant $\alpha _ { 0 } ~ > ~ 0$ . Let m $\geq 2$ be an integer, $R \geq 0$ , and $0 < \sigma _ { \mathrm { m i n } } \le \sigma _ { \mathrm { m a x } } < \infty$ . Assume that $\Lambda : = R / \sigma _ { \operatorname* { m i n } } + \log ( \sigma _ { \operatorname* { m a x } } / \sigma _ { \operatorname* { m i n } } ) \geq C _ { 0 } m$ , where $C _ { 0 } > 0$ is a suficiently large universal constant. Then, any non-interactive ε-LDP algorithm that learns ${ \mathfrak { m } } { \cdot } m i x ( \mathcal { G } ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ) )$ to total variation distance α with probability at least 0.9 requires $\begin{array} { r } { n = \Omega \left( \frac { m ^ { 2 } } { \varepsilon ^ { 2 } \alpha ^ { 2 } } \log { \left( e + \frac { \Lambda } { m } \right) } \right) } \end{array}$ samples. This lower bound holds even for non-proper learners.

Since every Gaussian density is log-concave, Theorem G.1 immediately implies the same lower bound for learning mixtures of log-concave distributions m-mix $( \mathcal { L } ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ) )$

Theorem G.2 (Sparse distribution estimation lower bound $\left[ \mathrm { A K L S 2 0 } \right] )$ . There exists a universal constant $c _ { 0 } > 0$ such that the following holds. Let $\varepsilon \in ( 0 , 1 ] , \alpha \in ( 0 , c _ { 0 } ]$ , and let $k , s$ be integers satisfying $2 \leq s \leq k / 1 0 0$ . Let $S _ { k , s } : = \{ p \in \Delta _ { k } : \| p \| _ { 0 } \leq s \}$ be the class of distributions supported on at most s elements, where $\begin{array} { r } { \Delta _ { k } = \left\{ ( w _ { 1 } , \dots , w _ { k } ) \in \mathbb { R } _ { \ge 0 } ^ { k } : \sum _ { i \in [ k ] } w _ { i } = 1 \right\} } \end{array}$ . Then, any non-interactive ε-LDP algorithm that learns $\mathcal { S } _ { k , s }$ to total variation distance α with probability at least 0.9 requires

$$
n = \Omega \left( \frac { s ^ { 2 } \log ( k / s ) } { \varepsilon ^ { 2 } \alpha ^ { 2 } } \right)
$$

samples. The output is not required to be s-sparse.

Proof of Theorem G.1. Now, we give our lower bound proof for LDP learning of GMMs.

Define parameters $\begin{array} { r } { K _ { \mu } = 1 + \left\lfloor \frac { R } { \sigma _ { \operatorname* { m i n } } } \right\rfloor , K _ { \sigma } = 1 + \left\lfloor \frac { \log ( \sigma _ { \operatorname* { m a x } } / \sigma _ { \operatorname* { m i n } } ) } { \log 1 6 } \right\rfloor , K = \operatorname* { m a x } \{ K _ { \mu } , K _ { \sigma } \} } \end{array}$ , and $M =$ min $\{ m , K \}$

Taking $C _ { 0 } \geq 1 0 0 ( 1 + \log 1 6 )$ , the assumption gives $K \geq$ max $\begin{array} { r } { \left\{ \frac { R } { \sigma _ { \operatorname* { m i n } } } , \frac { \log \left( \sigma _ { \operatorname* { m a x } } / \sigma _ { \operatorname* { m i n } } \right) } { \log 1 6 } \right\} \geq \frac { \Lambda } { 1 + \log 1 6 } \geq 1 0 0 m } \end{array}$ Thus, $M \ = \ m$ and $2 \leq M \leq K / 1 0 0$ , as required by the sparse estimation lower bound in Theorem G.2.

We construct K Gaussian components $g _ { 1 } , \dots , g _ { K } \in \mathcal G ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } )$ and disjoint measurable sets $B _ { 1 } , \ldots , B _ { K } \subset \mathbb { R }$ such that $g _ { j } ( B _ { j } ) \geq 2 / 3$ for all $j \in [ K ]$ . Let $\Phi$ denote the standard normal CDF. We use the larger of the following two constructions.

(1) Location Gaussians. For $j \in [ K _ { \mu } ]$ , set $\mu _ { j } = ( 2 j - K _ { \mu } - 1 ) \sigma _ { \operatorname* { m i n } } , g _ { j } = \mathcal { N } ( \mu _ { j } , \sigma _ { \operatorname* { m i n } } ^ { 2 } )$ , and $B _ { j } =$ $[ \mu _ { j } - \sigma _ { \mathrm { m i n } } , \mu _ { j } + \sigma _ { \mathrm { m i n } } )$ . Then, the means satisfy $| \mu _ { j } | \leq ( K _ { \mu } - 1 ) \sigma _ { \operatorname* { m i n } } \leq R$ . The set $B _ { j } \mathrm { ^ { * } s }$ are disjoint, and

$$
g _ { j } ( B _ { j } ) = 2 \Phi ( 1 ) - 1 > \frac { 2 } { 3 } .
$$

(2) Scale Gaussians. For $j \in [ K _ { \sigma } ]$ , set $\sigma _ { j } = 1 6 ^ { j - 1 } \sigma _ { \operatorname* { m i n } } , g _ { j } = \mathcal { N } ( 0 , \sigma _ { j } ^ { 2 } )$ , and $B _ { j } = \{ x : \sigma _ { j } / 4 \leq | x | <$ $4 \sigma _ { j } \}$ . We have ${ \sigma _ { j } } \in [ \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ]$ . Then, the sets $B _ { j }$ ’s are disjoint, and

$$
g _ { j } ( B _ { j } ) = 2 \big ( \Phi ( 4 ) - \Phi ( 1 / 4 ) \big ) > \frac { 2 } { 3 } .
$$

For $p \in \Delta _ { K }$ , write $\begin{array} { r } { f _ { p } : = \sum _ { j \in [ K ] } p _ { j } g _ { j } } \end{array}$ , and let $\gamma = 1 / 3$ . Since the sets are disjoint, $\begin{array} { r } { \sum _ { i \neq j } g _ { j } ( B _ { i } ) \leq } \end{array}$ $1 - g _ { j } ( B _ { j } )$ . Thus, the reverse triangle inequality gives, for any $p , q \in \Delta _ { K }$ ，

$$
\begin{array} { l } { \displaystyle \mathrm { d } _ { \mathrm { T V } } ( f _ { p } , f _ { q } ) \geq \frac { 1 } { 2 } \displaystyle \sum _ { i \in [ K ] } \displaystyle \left. \sum _ { j \in [ K ] } ( p _ { j } - q _ { j } ) g _ { j } ( B _ { i } ) \right. } \\ { \displaystyle \qquad \geq \frac { 1 } { 2 } \displaystyle \sum _ { j \in [ K ] } \vert p _ { j } - q _ { j } \vert \left( g _ { j } ( B _ { j } ) - \sum _ { i \neq j } g _ { j } ( B _ { i } ) \right) } \\ { \displaystyle \qquad \geq \frac { 1 } { 2 } \sum _ { j \in [ K ] } \vert p _ { j } - q _ { j } \vert \big ( 2 g _ { j } ( B _ { j } ) - 1 \big ) } \\ { \displaystyle \qquad \geq \gamma \mathrm { d } _ { \mathrm { T V } } ( p , q ) . } \end{array}
$$

We reduce learning $\mathcal { S } _ { K , M }$ to learning the Gaussian mixture class. Given a discrete sample $Y = j$ , the corresponding user independently draws $X \sim g _ { j }$ and applies the local randomizer of the Gaussian mixture learner to $X$ . For any such randomizer $Q$ and measurable message event $A ,$ integrating its privacy inequality gives

$$
\int Q ( A \mid x ) d g _ { i } ( x ) \leq e ^ { \varepsilon } \int Q ( A \mid x ) d g _ { j } ( x )
$$

for all $i , j \in [ K ]$ . Hence, the resulting protocol remains non-interactive $\varepsilon { \mathrm { - L D P } } .$ . If $Y \sim p \in S _ { K , M }$ then $X \sim f _ { p }$ , and $f _ { p }$ has at most $M \leq m$ Gaussian components.

Let $\widehat { f }$ be the output of the learner, and define $\widehat { p } \in$ arg min $\boldsymbol { \mathbf { \ell } } _ { q \in \mathcal { S } _ { K , M } } \operatorname { d _ { T V } } ( \widehat { f } , f _ { q } )$ . A minimizer exists because $\mathcal { S } _ { K , M }$ is compact and the objective is continuous in $q .$

On the event $\mathrm { d } _ { \mathrm { T V } } ( \widehat { f } , f _ { p } ) \leq \alpha$ , we have

$$
\gamma \mathrm { d } _ { \mathrm { T V } } ( \widehat { p } , p ) \leq \mathrm { d } _ { \mathrm { T V } } ( f _ { \widehat { p } } , f _ { p } )
$$

$$
\begin{array} { r l } & { \leq \mathrm { d } _ { \mathrm { T V } } ( f _ { \widehat { p } } , \widehat { f } ) + \mathrm { d } _ { \mathrm { T V } } ( \widehat { f } , f _ { p } ) } \\ & { \leq 2 \alpha . } \end{array}
$$

This decoding is post-processing and returns an M-sparse probability vector. The resulting discrete learner has accuracy $2 \alpha / \gamma$ and succeeds with probability at least 0.9. Choosing α so that $2 \alpha _ { 0 } / \gamma \leq c _ { 0 }$ , Theorem G.2 therefore gives $\begin{array} { r } { n = \Omega \left( \frac { m ^ { 2 } \log \left( K / m \right) } { \varepsilon ^ { 2 } \alpha ^ { 2 } } \right) } \end{array}$ . Finally, since $K = \Theta ( \Lambda )$ and $\begin{array} { r } { K / m \geq 1 0 0 , \log ( K / m ) = \Theta \left( \log \left( e + \frac { \Lambda } { m } \right) \right) } \end{array}$ , which gives the stated bound. □

## H Discussion on proper learning and connections to Private Hypothesis Selection

The results in the preceding sections concern non-proper learning. In this section, we convert these learners into proper learners by projecting their outputs onto a finite cover of the target class. This preserves the sample complexity up to a constant rescaling of the accuracy parameter, at the cost of a larger agnostic approximation factor and potentially ineficient computation.

A generic alternative for non-eficient proper learning a class $\mathcal { F }$ is to apply locally private hypothesis selection to a finite α-cover ${ \mathcal { C } } \subseteq { \mathcal { F } }$ . The selector returns a member of ${ \mathcal { C } } ,$ so its output belongs to the target class. After a constant rescaling of α to account for the cover error, the interactive algorithm of Pour et al. [PAA24] gives sample complexity ${ \cal \tilde { O } } ( | { \cal { C } } | / ( \varepsilon ^ { 2 } \alpha ^ { 2 } ) )$ with agnostic approximation factor $^ { 9 , }$ while the non-interactive algorithm of Kamath et al. [KPRW25] gives $\bar { O } ( | C | ^ { 3 / 2 } / ( \varepsilon ^ { 2 } \alpha ^ { 2 } ) )$ with agnostic approximation factor 13. Here, $\widetilde O$ suppresses logarithmic factors in |C| and $1 / \beta$

However, such covers can be exponentially large. For example, for GMMs, the cover size is roughly $\begin{array} { r } { \tilde { O } \left( \left( \frac { m R } { \alpha ^ { 3 } \sigma _ { \mathrm { m i n } } } \right) ^ { m } \right) } \end{array}$ . Applying LDP hypothesis selection to such a cover yields sample complexity bound that is exponential in m and polynomial in range parameters $R / \alpha \sigma _ { \mathrm { m i n } }$ . In comparison, our proper learning result in Corollary H.3 preserves the sample bound of our non-proper learner, with quadratic dependence on m (up to logarithmic factors) and only polylogarithmic dependence on the component range parameters. The cover size afects the computation required for the final projection, but does not enter the sample bound. The resulting agnostic approximation factor is 21.

For an arbitrary class of distributions ${ \mathcal { F } } _ { : }$ , an $\Omega ( | \mathcal { F } | / ( \varepsilon ^ { 2 } \alpha ^ { 2 } ) )$ worst-case sample lower bound holds even for interactive $\mathrm { L D P }$ hypothesis selection [GKKNWZ20; PAA24]. This worst-case bound prevents a generic improvement based only on $| { \mathcal { F } } |$ ; it does not impose the same dependence for structured distribution classes.

Lemma H.1 (From non-proper to proper learning). Let $C \geq 1 , \alpha \in ( 0 , 1 / 4 )$ , and $\beta \in ( 0 , 1 )$ . Let $\mathcal { F }$ be a class of distributions with a given finite α-cover ${ \mathcal { C } } \subseteq { \mathcal { F } }$ in total variation distance. Let $\mathcal { A }$ be a non-proper $( C , \alpha , \beta )$ -agnostic learner for $\mathcal { F }$ . Suppose there exists an oracle that, given any $F \in { \mathcal { F } }$ and any distribution $Q _ { i }$ , approximates $\mathrm { d } _ { T V } ( Q , F )$ with additive error at most $\alpha / 2$ in time $T _ { \mathrm { d i s t } }$ . Then, the following holds:

There exists a proper $( 2 C + 1 , 4 \alpha , \beta )$ )-agnostic learner for F that outputs a member of $\mathcal { C } _ { i }$ uses the same number of samples, and takes $O ( | { \mathcal { C } } | ( T _ { \mathrm { d i s t } } + 1 ) )$ ) additional time. Moreover, $i f . A$ is non-interactive $\varepsilon { - } L D P ,$ then the resulting proper learner is also non-interactive ${ \varepsilon } { - } L D P$

The proof, given later in this section, follows by approximately projecting the non-proper learner’s output onto C and applying the triangle inequality.

The next lemma provides such a finite cover for the class of bounded-parameter Gaussian mixtures.   
Its proof appears later in this section.

Lemma H.2. Let $\alpha \in ( 0 , 1 ) , R \geq 0 , m \geq 1$ be an integer, $0 < \sigma _ { \mathrm { m i n } } \le \sigma _ { \mathrm { m a x } } < \infty$ , and $c > 0$ be a universal constant. Then the class ${ \mathit { m } } { - } { \mathit { m i x } } ( \mathcal { G } ( R , \sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } } ) )$ admits an α-cover in total variation distance of size at most $\begin{array} { r } { \left[ \frac { c m } { \alpha } \left( 1 + \frac { R } { \alpha \sigma _ { \mathrm { m i n } } } \right) \left( 1 + \frac { \log \left( \sigma _ { \mathrm { m a x } } / \sigma _ { \mathrm { m i n } } \right) } { \alpha } \right) \right] ^ { m } } \end{array}$

Combining the proper conversion from H.1 with Lemma H.2 and Theorem F.2 gives the following corollary.

Corollary H.3 (LDP Agnostic Proper Learning of GMMs). Under the same assumptions of Theorem F.2, there exists a non-interactive $\varepsilon \ – L D P \ ( 2 1 , \alpha , \beta ) \ – a g \imath$ ostic proper learner for the class of ${ \mathfrak { m } } { \cdot } { \mathfrak { m i x } } ( { \mathcal { G } } ( R , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ) )$ that uses $\begin{array} { r } { O \left( \frac { m ^ { 2 } \log ^ { 2 } ( 2 / \alpha ) } { \varepsilon ^ { 2 } \alpha ^ { 2 } } \log ^ { 2 } \left( \frac { 2 ( R + \sigma _ { \operatorname* { m a x } } ) } { \alpha \sigma _ { \operatorname* { m i n } } } \right) \log \left( \frac { 2 ( R + \sigma _ { \operatorname* { m a x } } ) } { \alpha \beta \sigma _ { \operatorname* { m i n } } } \right) \right) } \end{array}$ samples. The algorithm is not claimed to be computationally eficient.

Proof of Lemma H.1. Now, we prove how to obtain a proper learner from a non-proper learner.

Run A to obtain a distribution ${ \widehat { P } } .$ . Approximate each distance $\operatorname { d } _ { \mathrm { T V } } ( \widehat { P } , G )$ , for $G \in { \mathcal { C } } .$ , to additive error at most $\alpha / 2 ,$ and return a member ${ \tilde { P } } \in { \mathcal { C } }$ with the smallest estimated distance, breaking ties according to a fixed ordering of C. This ensures that

$$
\mathrm { d _ { T V } } ( \widehat { P } , \widetilde { P } ) \leq \operatorname* { m i n } _ { G \in \mathcal { C } } \mathrm { d _ { T V } } ( \widehat { P } , G ) + \alpha .
$$

Let P be the sampling distribution. With probability at least $1 - \beta$ , the guarantee of A gives

$$
\mathrm { d _ { T V } } ( \widehat { P } , P ) \leq C \operatorname* { i n f } _ { F \in \mathcal { F } } \mathrm { d _ { T V } } ( P , F ) + \alpha .
$$

Condition on this event. Fix any $F ^ { \star } \in { \mathcal { F } }$ . Since $\mathcal { C }$ is an α-cover, there exists $G ^ { \star } \in { \mathcal { C } }$ such that $\mathrm { d } _ { \mathrm { T V } } ( F ^ { \star } , G ^ { \star } ) \leq \alpha$ . Therefore,

$$
\begin{array} { r l } & { \mathrm { d } _ { \mathrm { T V } } ( \widetilde { P } , P ) \leq \mathrm { d } _ { \mathrm { T V } } ( \widetilde { P } , \widehat { P } ) + \mathrm { d } _ { \mathrm { T V } } ( \widehat { P } , P ) } \\ & { \qquad \leq \mathrm { d } _ { \mathrm { T V } } ( G ^ { \star } , \widehat { P } ) + \mathrm { d } _ { \mathrm { T V } } ( \widehat { P } , P ) + \alpha } \\ & { \qquad \leq \mathrm { d } _ { \mathrm { T V } } ( F ^ { \star } , \widehat { P } ) + \mathrm { d } _ { \mathrm { T V } } ( \widehat { P } , P ) + 2 \alpha } \\ & { \qquad \leq \mathrm { d } _ { \mathrm { T V } } ( F ^ { \star } , P ) + 2 \mathrm { d } _ { \mathrm { T V } } ( \widehat { P } , P ) + 2 \alpha } \\ & { \qquad \leq ( 2 C + 1 ) \mathrm { d } _ { \mathrm { T V } } ( P , F ^ { \star } ) + 4 \alpha . } \end{array}
$$

Taking the infimum over $F ^ { \star } \in { \mathcal { F } }$ proves the accuracy guarantee. The additional errors account for the cover and the approximate minimization.

The conversion computes |C| distance approximations and makes $O ( | \mathcal { C } | )$ additional comparisons, giving $O ( | C | ( T _ { \mathrm { d i s t } } + 1 ) ) ,$ additional running time. It uses only the output of A, so it requires no additional samples and preserves privacy and non-interactivity by post-processing.

For a target additive error $\tau \in ( 0 , 1 )$ , applying this construction with $\alpha = \tau / 4$ and a cover of radius $\tau / 4$ gives a proper $( 2 C + 1 , \tau , \beta )$ -agnostic learner using $n _ { \cal A } ( \tau / 4 , \beta )$ samples. □

Here, we explain the construction of a cover for bounded-parameter GMMs.

Claim H.4. Let $\alpha \in ( 0 , 1 ]$ . There is an α-cover for the $( m - 1 )$ -dimensional probability simplex $\begin{array} { r } { \Delta _ { m } = \Big \{ ( w _ { 1 } , w _ { 2 } , . . . , w _ { m } ) \in \mathbb { R } _ { \geq 0 } ^ { m } : \sum _ { i \in [ m ] } w _ { i } = 1 \Big \} } \end{array}$ with respect to $\ell _ { \infty }$ of size at most $\lceil 1 / \alpha \rceil ^ { m }$

Proof. Partition $[ 0 , 1 ]$ into $\lceil 1 / \alpha \rceil$ intervals of length at most $\alpha ,$ and use their Cartesian products to partition $[ 0 , 1 ] ^ { m }$ . For every cell that intersects $\Delta _ { m } ,$ select one point from the intersection. Any two points in the same cell have $\ell _ { \infty }$ distance at most $\alpha ,$ so the selected points form an α-cover of $\Delta _ { m }$ There are at most $\lceil 1 / \alpha \rceil ^ { m }$ cells. □

Proof of Lemma H.2. We prove the cover construction for GMMs.

Let $\begin{array} { r } { f = \sum _ { i \in [ m ] } w _ { i } \mathcal { N } ( \mu _ { i } , \sigma _ { i } ^ { 2 } ) } \end{array}$ be an arbitrary member of the class. We construct a cover by discretizing the mixture weights, means, and standard deviations.

For the weights, let $C _ { w }$ be an α/m-cover of $\Delta _ { m }$ in $\ell _ { \infty } ,$ of size at most $\lceil m / \alpha \rceil ^ { m }$ , as given by Claim H.4.

For the means, let $\mathcal { C } _ { \mu }$ be a grid on $[ - R , R ]$ containing both endpoints, with consecutive points at distance at most $\alpha \sigma _ { \mathrm { m i n } } / 2$ . When $R = 0$ , take ${ \mathcal C } _ { \mu } = \{ 0 \}$ . The grid can be chosen to have at most $1 + \lceil 4 R / ( \alpha \sigma _ { \mathrm { m i n } } ) \rceil$ points, and every $\mu \in [ - R , R ]$ has a grid point $\tilde { \mu }$ satisfying $| \mu - \tilde { \mu } | \leq \alpha \sigma _ { \mathrm { m i n } } / 4$

For the standard deviations, take

$$
\mathcal { C } _ { \sigma } = \left\{ \sigma _ { \operatorname* { m i n } } ( 1 + \alpha / 4 ) ^ { j } : j = 0 , \dots , \left\lfloor \frac { \log ( \sigma _ { \operatorname* { m a x } } / \sigma _ { \operatorname* { m i n } } ) } { \log ( 1 + \alpha / 4 ) } \right\rfloor \right\} .
$$

For every $\sigma \in [ \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ]$ , there is a grid point $\tilde { \sigma }$ such that $\tilde { \sigma } \leq \sigma < ( 1 + \alpha / 4 ) \tilde { \sigma }$ , and hence $| \log ( \sigma / \tilde { \sigma } ) | \le \alpha / 4$

Let C contain all mixtures formed from weights in $C _ { w }$ , means in $\mathcal { C } _ { \mu } .$ , and standard deviations in $\mathcal { C } _ { \sigma }$ Every member of C belongs to the target class.

For univariate Gaussians, varying the mean and then the standard deviation gives

$$
\mathrm { d _ { T V } } \left( \mathcal { N } ( \mu , \sigma ^ { 2 } ) , \mathcal { N } ( \tilde { \mu } , \tilde { \sigma } ^ { 2 } ) \right) \leq \frac { | \mu - \tilde { \mu } | } { 2 \sigma } + \left| \log \left( \frac { \sigma } { \tilde { \sigma } } \right) \right| .
$$

Choose the corresponding grid parameters for $f ,$ and write $\begin{array} { r } { \tilde { f } = \sum _ { i \in [ m ] } \tilde { w } _ { i } \mathcal { N } ( \tilde { \mu } _ { i } , \tilde { \sigma } _ { i } ^ { 2 } ) \in \mathcal { C } } \end{array}$ . Then

$$
\begin{array} { r l } & { \mathrm { d } _ { \mathrm { T V } } ( f , \tilde { f } ) \leq \displaystyle \frac { 1 } { 2 } \sum _ { i \in [ m ] } | w _ { i } - \tilde { w } _ { i } | + \sum _ { i \in [ m ] } \tilde { w } _ { i } \mathrm { d } _ { \mathrm { T V } } \left( \mathcal { N } ( \mu _ { i } , \sigma _ { i } ^ { 2 } ) , \mathcal { N } ( \tilde { \mu } _ { i } , \tilde { \sigma } _ { i } ^ { 2 } ) \right) } \\ & { \qquad \leq \displaystyle \frac { \alpha } { 2 } + \sum _ { i \in [ m ] } \tilde { w } _ { i } \left( \frac { \alpha \sigma _ { \operatorname* { m i n } } / 4 } { 2 \sigma _ { \operatorname* { m i n } } } + \frac { \alpha } { 4 } \right) } \\ & { \qquad = \displaystyle \frac { \alpha } { 2 } + \frac { 3 \alpha } { 8 } \leq \alpha . } \end{array}
$$

Finally,

$$
| { \mathcal C } | \le | C _ { w } | | { \mathcal C } _ { \mu } | ^ { m } | { \mathcal C } _ { \sigma } | ^ { m }
$$

$$
\begin{array} { l } { \displaystyle \leq \left\lceil \frac { m } { \alpha } \right\rceil ^ { m } \left( 1 + \left\lceil \frac { 4 R } { \alpha \sigma _ { \operatorname* { m i n } } } \right\rceil \right) ^ { m } \left( 1 + \left\lfloor \frac { \log \left( \sigma _ { \operatorname* { m a x } } / \sigma _ { \operatorname* { m i n } } \right) } { \log ( 1 + \alpha / 4 ) } \right\rfloor \right) ^ { m } } \\ { \displaystyle \leq \left\lceil \frac { c m } { \alpha } \left( 1 + \frac { R } { \alpha \sigma _ { \operatorname* { m i n } } } \right) \left( 1 + \frac { \log \left( \sigma _ { \operatorname* { m a x } } / \sigma _ { \operatorname* { m i n } } \right) } { \alpha } \right) \right\rceil ^ { m } } \end{array}
$$

for a universal constant $c > 0$