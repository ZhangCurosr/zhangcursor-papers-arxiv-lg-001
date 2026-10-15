# Hypothesis Selection via Weighted Medians

Liang-Yu Zou

Aarhus University

zou@cs.au.dk

## Abstract

Given � distributions (hypotheses) $\mathcal { H } = \{ H _ { 1 } , \ldots , H _ { n } \}$ and samples from an unknown target distribution $P ,$ proper hypothesis selection seeks a distribution whose total variation distance to � is close to the best achievable within the known distributions. We give an algorithm that, using $O ( \varepsilon ^ { - 2 } \log ( n / \delta ) )$ target samples, running in $O ( n \varepsilon ^ { - 2 } \log ( n / \delta ) )$ time, with probability at least $1 - \delta ,$ returns $H _ { \hat { i } }$ satisfying $\begin{array} { r } { d _ { \mathrm { T V } } ( P , H _ { \hat { i } } ) \leq 3 \operatorname* { m i n } _ { i \in [ n ] } d _ { \mathrm { T V } } ( P , H _ { i } ) + \varepsilon . } \end{array}$ . The algorithm achieves the optimal approximation factor and sample complexity. The running time improves the dependence on the failure probability from polynomial to logarithmic in the state-of-the-art bound $\tilde { O } ( n \bar { \varepsilon } ^ { - 2 } \delta ^ { - 1 } )$ of Aamand, Aliakbarpour, Chen, and Silwal (COLT 2026), thereby answering their Question 2.

## 1 Introduction

Selecting a probability distribution from a finite class is a basic task in statistical learning. The target distribution need not belong to the class: the aim is to compete with the best available distribution using samples from the target. In hypothesis selection, the quality of this approximation is measured in total variation distance, and the challenge is to obtain statistical guarantees with a running time that scales favorably with the number of distributions.

Let $\mathcal { H } = \left\{ H _ { 1 } , \ldots , H _ { n } \right\}$ be a class of distributions, each a probability measure on a measurable space $( \mathcal { X } , \mathcal { B } )$ and let $P$ be the unknown target distribution. Fix a common dominating measure $\mu$ for $P , H _ { 1 } , \ldots , H _ { n }$ , and write $h _ { i } = d H _ { i } / d \mu$ for the density of $H _ { i }$ . We have sample access to $P$ and sample and density-query access to every $H _ { i } ;$ the density of $P$ is not available to the algorithm. Writing $\begin{array} { r } { \mathrm { O P T } = \operatorname* { m i n } _ { i \in [ n ] } d _ { \mathrm { T V } } ( P , H _ { i } ) } \end{array}$ , the goal is to return a probability measure $Q$ such that, with probability at least $1 - \delta .$

$$
d _ { \mathrm { T V } } ( P , Q ) \leq C \cdot \mathrm { O P T } + \varepsilon .
$$

Here $ { \varepsilon } \in ( 0 , 1 )$ is the additive error and $\delta \in ( 0 , 1 )$ is the failure probability. The selection is proper if $Q \in { \mathcal { H } } ;$ otherwise, the algorithm is allowed to be improper.

Statistical guarantees. Classical minimum-distance methods achieve approximation factor $C = 3$ using $O ( \varepsilon ^ { - 2 } \log ( n / \delta ) )$ samples from � [Yat85, DL01]. The factor 3 is also optimal for proper selection with a sample size independent of the domain size [BKM19, AACS26]. Allowing an improper output improves the achievable factor to 2 [BKM19, BBK<sup>+</sup>22]. Thus, for proper selection, the remaining issue is how eficiently the optimal statistical guarantee can be attained.

Computational guarantees. Classical factor-3 selectors have cubic or quadratic running times in � [DL01, MS08]. Aliakbarpour, Bun, and Smith [ABS24] obtained a nearly linear dependence on �, with running time $\tilde { O } ( n / ( \delta ^ { 3 } \varepsilon ^ { 3 } ) )$ and optimal target-sample complexity. Aamand, Aliakbarpour, Chen, and Silwal [AACS26] improved the running time to $\tilde { O } ( n / ( \delta \varepsilon ^ { 2 } ) )$ for $\delta > 1 / n$ . Selected bounds are compared in

<table><tr><td>Algorithm</td><td>Running time</td><td>Failure probability Comments</td><td></td></tr><tr><td>Minimum distance [Yat85, DL01]</td><td> $O ( n ^ { 3 } \varepsilon ^ { - 2 } \log n )$ </td><td> $1 / \mathrm { p o l y } ( n )$ </td><td></td></tr><tr><td>Modified minimum distance [MS08]</td><td> $O ( n ^ { 2 } \varepsilon ^ { - 2 } \log n )$ </td><td> $1 / \mathrm { p o l y } ( n )$ </td><td></td></tr><tr><td>Approx min distance [ABS24]</td><td> $\tilde { O } ( n / ( \delta ^ { 3 } \varepsilon ^ { 3 } ) )$ </td><td>δ</td><td></td></tr><tr><td>Semi-distance threshold [AACS26, Theorem 1.3]</td><td> $\tilde { O } ( n / ( \delta \varepsilon ^ { 2 } ) )$ </td><td>δ</td><td> $\delta > 1 / n$ </td></tr><tr><td>This paper, Theorem 1.1</td><td> $O ( n \varepsilon ^ { - 2 } \log ( n / \delta ) )$ </td><td>δ</td><td></td></tr></table>

Table 1: Proper hypothesis selection with approximation factor 3. All rows use $O ( \varepsilon ^ { - 2 } \log ( n / \delta ) )$ target samples; for polynomially small failure probability, this is $O ( \varepsilon ^ { - 2 } \log n )$ . More results are collected in [AACS26, Table 1].

Table 1. Aamand et al. ask whether proper selection can simultaneously achieve factor 3, $O ( \varepsilon ^ { - 2 } \log ( n / \delta ) )$ target samples, and running time $\tilde { O } ( n \log ( 1 / \delta ) / \varepsilon ^ { 2 } )$ [AACS26, Question 2]. We answer this question afirmatively.

Theorem 1.1. Let $n \geq 1 , 0 < \varepsilon \leq 1$ , and $0 < \delta < 1$ . Algorithm 1 returns a distribution $H _ { \hat { i } } \in \mathcal { H }$ such that, with probability at least $1 - \delta ,$

$$
d _ { \mathrm { T V } } ( P , H _ { \hat { i } } ) \leq 3 ~ \mathrm { O P T } + \varepsilon .
$$

The algorithm uses $O ( \varepsilon ^ { - 2 } \log ( n / \delta ) )$ target samples, and runs in $O ( n \varepsilon ^ { - 2 } \log ( n / \delta ) )$ time.

## 1.1 Overview of the method

The algorithm maintains positive weights $\mathbf { w } _ { t }$ summing to one. At each point �, let $\boldsymbol u _ { t } ( \boldsymbol x )$ be the lower weighted median of $h _ { 1 } ( x ) , \ldots , h _ { n } ( x )$ under these weights, and let $U _ { t }$ be the measure with density $u _ { t }$ . We use $U _ { t }$ as an intermediate measure for comparing the input distributions with the target. Although $U _ { t }$ need not have mass one, its distance to each $H _ { i }$ can be measured by $\begin{array} { r l } {  { \frac { 1 } { 2 } \| U _ { t } - H _ { i } \| _ { 1 } } } \end{array}$ , where $\begin{array} { r } { \| U _ { t } - H _ { i } \| _ { 1 } = \int \ \left| u _ { t } - h _ { i } \right| d \mu } \end{array}$ is the total variation norm of their diference.

For each $i ,$ define a test $\sigma _ { t , i } ( x ) \in [ - 1 , 1 ]$ that equals +1 when $h _ { i } ( x )$ is below the median and −1 when it is above. A fractional value on ties ensures the pointwise identities

$$
\sum _ { i } w _ { t , i } \sigma _ { t , i } ( x ) = 0 , \qquad \sigma _ { t , i } ( x ) \big ( u _ { t } ( x ) - h _ { i } ( x ) \big ) = | u _ { t } ( x ) - h _ { i } ( x ) | .
$$

The first identity balances the tests under the current weights; the second connects them to distances from the median measure.

In each round, the algorithm draws one sample from � and one sample from a proposal mixture $M _ { t } ^ { \prime }$ of the input distributions. The proposal mixture is constructed to be close to the mixture $\begin{array} { r } { M _ { t } = \sum _ { i } w _ { t , i } H _ { i } } \end{array}$ of the input distributions under the current weights, and can be sampled eficiently. These two samples produce a bounded gain estimate $g _ { t , i }$ for each distribution, with conditional expectation $\begin{array} { r l r } { \mathrm { ~ } } & { { } } & { \frac { 1 } { 2 } \int \ \sigma _ { t , i } d ( U _ { t } - P ) } \end{array}$ given the previous rounds. $\mathsf { A }$ positive gain indicates that the corresponding distribution is closer to the median than � is. A multiplicative update then increases the weights of distributions with positive gain.

The balance identity gives $\begin{array} { r } { \sum _ { i } w _ { t , i } g _ { t , i } = 0 } \end{array}$ for every realization, so the update preserves the sum of the weights without renormalization. A standard multiplicative-weights argument [AHK12] bounds the average realized gain of every distribution by $O ( \varepsilon )$ . Concentration for bounded martingale diferences transfers this bound to the average conditional gains along the adaptive run. The second identity above yields, for every �,

$$
\frac { 1 } { 2 } \| U _ { t } - H _ { i } \| _ { 1 } \leq \mathbb { E } [ g _ { t , i } \mid \mathcal { F } _ { t - 1 } ] + d _ { \mathrm { T V } } ( P , H _ { i } ) ,
$$

where $\mathcal { F } _ { t - 1 }$ records the randomness through round $t - 1$ . Consequently, for a best input distribution $H _ { i ^ { \star } }$ , the average distance $\begin{array} { r } { \frac { 1 } { T } \sum _ { t } \frac { 1 } { 2 } \| \boldsymbol { U } _ { t } - \boldsymbol { H } _ { i ^ { * } } \| _ { 1 } } \end{array}$ is at most $\mathrm { O P T } + O ( \varepsilon )$ with high probability. Using $H _ { i ^ { * } }$ as the intermediate measure in the triangle inequality gives

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \frac { 1 } { 2 } \| U _ { t } - P \| _ { 1 } \leq \mathrm { O P T } + \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \frac { 1 } { 2 } \| U _ { t } - H _ { i ^ { \ast } } \| _ { 1 } \leq 2 \mathrm { O P T } + O ( \varepsilon ) .
$$

To obtain a proper output, the algorithm also uses the proposal samples to form bounded, unbiased estimates of $\begin{array} { r l } {  { { \frac { 1 } { 2 } } \| U _ { t } - H _ { i } \| } } \end{array}$ <sub>1</sub> for every �. These estimates include a mass correction because $U _ { t }$ is not normalized. The algorithm returns $H _ { \hat { i } }$ minimizing the estimated average distance to the median measures. Uniform concentration of the distance estimates ensures that its true average distance is at most $\mathrm { O P T } + O ( \varepsilon )$ . Averaging the triangle inequality through $U _ { t }$ therefore yields

$$
d _ { \mathrm { T V } } ( P , H _ { \hat { i } } ) \leq \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left( \frac { 1 } { 2 } \| P - U _ { t } \| _ { 1 } + \frac { 1 } { 2 } \| U _ { t } - H _ { \hat { i } } \| _ { 1 } \right) \leq 3 \mathrm { O P T } + O ( \varepsilon ) .
$$

The analysis in Section 4 tracks the constants to obtain $3 \operatorname { O P T } + \varepsilon$ with probability at least $1 - \delta$ after $T = O ( \varepsilon ^ { - 2 } \log ( n / \delta ) )$ rounds. Each round uses one target sample and $O ( n )$ time, giving the claimed sample complexity and running time.

## 2 Preliminaries

## 2.1 Problem setting and computational model

Let $H _ { 1 } , \ldots , H _ { n }$ be probability distributions on a measurable space $( \mathcal { X } , { B } )$ , and let $P$ be the unknown target distribution on the same space. Fix a common �-finite dominating measure $\mu$ for $P , H _ { 1 } , \ldots , H _ { n }$ , and write

$$
p = \frac { d P } { d \mu } , \qquad h _ { i } = \frac { d H _ { i } } { d \mu } \quad ( i \in [ n ] ) .
$$

Here $h _ { i }$ is the density of the distribution $H _ { i }$ relative to $\mu .$ The target density $\boldsymbol { p }$ is used only as mathematical notation and is not available to the algorithm. We access � through a sample oracle, and each $H _ { i }$ through a sample oracle and a density oracle that evaluates $h _ { i } ( x )$ at any supplied point $x \in \mathcal { X }$ . Every sample-oracle call returns a fresh independent draw from the requested measure.

For accuracy $0 ~ < ~ \varepsilon ~ \leq ~ 1$ and failure probability $0 ~ < ~ \delta ~ < ~ 1$ , our goal is to return a distribution $H _ { \hat { i } } \in \{ H _ { 1 } , \ldots , H _ { n } \}$ such that

$$
d _ { \mathrm { T V } } ( P , H _ { \bar { i } } ) \leq 3 \mathrm { O P T } + \varepsilon , \qquad \mathrm { O P T } = \operatorname* { m i n } _ { i \in [ n ] } d _ { \mathrm { T V } } ( P , H _ { i } ) ,
$$

with probability at least $1 - \delta$ over the samples and the algorithm’s internal randomness. The selection is proper because the output must be one of the input distributions.

We use an exact-real oracle model: sample-oracle calls, density evaluations, arithmetic operations (addition, subtraction, multiplication, or division), comparisons, ceiling operations, and draws from uniform distributions each take unit time. The algorithm’s running time is the total number of such operations. The sample complexity only counts the number of samples drawn from the target distribution $P .$

## 2.2 Notation and density conventions

Write $[ n ] = \left\{ 1 , \ldots , n \right\}$ , use ln for the natural logarithm and log for the base-two logarithm, and set $z _ { + } =$ $\operatorname* { m a x } \{ z , 0 \}$ . Throughout the paper, fix $i ^ { \star } \in \mathrm { a r g m i n } _ { i \in [ n ] } d _ { \mathrm { T V } } ( P , H _ { i } )$ , breaking ties by a fixed rule.

Uppercase symbols denote measures and their lowercase counterparts denote densities relative to $\mu .$ In particular, $m , m ^ { \prime } ;$ , and � denote the densities of the mixtures $M , M ^ { \prime }$ , and the median measure $U$ introduced in Section 3.

Since $h _ { i }$ is the density of $H _ { i }$ with respect to $\textstyle \mu , H _ { i } ( E ) = \int _ { E } h _ { i }$ �� denotes the probability of a measurable set $E \in B .$ . The same convention applies to $M , M ^ { \prime }$ , and �, although � need not have mass one.

## 2.3 Total variation and finite measures

For a finite signed measure � on $( \mathcal { X } , { B } )$ , write $\| V \| _ { 1 } = | V | ( \mathcal { X } )$ for its total variation norm. If � has density � relative to $\mu ,$ then $\begin{array} { r } { \| V \| _ { 1 } = \int } \end{array}$ |�| $d \mu$ . For probability measures $A , B ,$ we use

$$
d _ { \operatorname { T V } } ( A , B ) = \operatorname* { s u p } _ { E \in { \mathcal { B } } } | A ( E ) - B ( E ) | = { \frac { 1 } { 2 } } \| A - B \| _ { 1 } .
$$

For finite nonnegative measures �, � and any measurable function $f$ with $| f | \le 1$ , the signed-measure norm gives

$$
\left| \int f d ( A - B ) \right| \leq \| A - B \| _ { 1 } .\tag{1}
$$

We also use the triangle inequality for this norm, which applies even when the intermediate measure is not a probability measure. In particular, for probability measures $A , B$ and a finite nonnegative measure $U ,$

$$
d _ { \mathrm { T V } } ( A , B ) \leq \frac { 1 } { 2 } \| A - U \| _ { 1 } + \frac { 1 } { 2 } \| U - B \| _ { 1 } .
$$

When $U ( \mathcal { X } ) \neq 1$ , the quantity ${ \frac { 1 } { 2 } } \| U - A \| _ { 1 }$ denotes half the signed-measure norm, rather than the total variation distance between two probability measures.

A useful identity accounts explicitly for the mass of such a measure. If � has density � and mass $\begin{array} { r } { U ( \mathcal { X } ) = \int \ u \ d \mu } \end{array}$ , then

$$
\begin{array} { l } { \displaystyle \frac { 1 } { 2 } \| \boldsymbol { U } - H _ { i } \| _ { 1 } = \frac { 1 } { 2 } \int | \boldsymbol { u } - h _ { i } | d \mu } \\ { \displaystyle \ = \int ( \boldsymbol { u } - h _ { i } ) _ { + } d \mu - \frac { 1 } { 2 } \int ( \boldsymbol { u } - h _ { i } ) d \mu } \\ { \displaystyle \ = \ \int ( \boldsymbol { u } - h _ { i } ) _ { + } d \mu + \frac { 1 - U ( \mathcal { X } ) } { 2 } . } \end{array}\tag{2}
$$

Indeed, $| z | = 2 z _ { + } - z$ and $\textstyle \int h _ { i } d \mu = 1$ give this identity directly. The correction $( 1 - U ( \mathcal { X } ) ) / 2$ is needed because the median measures used by the algorithm are not normalized.

## 3 The algorithm

## 3.1 Weighted medians and balanced tests

Let $\mathbf { w } = ( w _ { 1 } , \dots , w _ { n } )$ be a vector of positive weights with $\textstyle \sum _ { i } w _ { i } = 1$ , and let $\begin{array} { r } { M = \sum _ { i } w _ { i } H _ { i } } \end{array}$ be the corresponding mixture of the distributions. Then � is a probability measure with density $\begin{array} { r } { m = \sum _ { i } w _ { i } h _ { i } } \end{array}$

Let �(�) be the lower weighted median of $h _ { 1 } ( x ) , \ldots , h _ { n } ( x )$ : the smallest input value whose cumulative weight is at least $1 / 2 .$ The function � is measurable, since it is selected from finitely many measurable density functions using measurable comparisons. Define the median measure � by $d U = u d \mu$ . Its mass $U ( \mathcal X )$ need not be one, so we do not normalize it.

Let $\begin{array} { r } { W _ { < } ( x ) = \sum _ { i : h _ { i } ( x ) < u ( x ) } w _ { i } } \end{array}$ be the total weight on density values below $u ( x )$ , with $W _ { = } ( x )$ and $W _ { > } ( x )$ defined analogously. Define the balanced median tests $\sigma _ { i } ( x )$ by

$$
\sigma _ { i } ( x ) = \left\{ \begin{array} { l l } { 1 , } & { h _ { i } ( x ) < u ( x ) , } \\ { \displaystyle \frac { W _ { > } ( x ) - W _ { < } ( x ) } { W _ { = } ( x ) } , } & { h _ { i } ( x ) = u ( x ) , } \\ { - 1 , } & { h _ { i } ( x ) > u ( x ) . } \end{array} \right.\tag{3}
$$

The tests are measurable signed functions that indicate whether $h _ { i } ( x )$ is below or above the median, with a fractional tie value that exactly balances the total weight on each side. The denominator $W _ { = } ( x )$ is positive because the median is an input value and all weights are positive.

Lemma 3.1 (Weighted-median identities). For every $x \in \mathcal { X }$ and $i \in [ n ]$

$$
\begin{array} { c } { | \sigma _ { i } ( x ) | \le 1 , } \\ { \displaystyle \sum _ { i } w _ { i } \sigma _ { i } ( x ) = 0 , } \\ { \sigma _ { i } ( x ) ( u ( x ) - h _ { i } ( x ) ) = | u ( x ) - h _ { i } ( x ) | , } \\ { 0 \le u ( x ) \le 2 m ( x ) . } \end{array}
$$

In particular, � is a finite nonnegative measure with $U ( { \mathcal { X } } ) \leq 2 .$

Proof of Lemma 3.1. Fix $x \in \mathcal { X }$ . The lower-median definition gives $W _ { < } ( x ) < 1 / 2$ and $W _ { > } ( x ) \leq 1 / 2$ . Since $W _ { = } ( x ) = 1 - W _ { < } ( x ) - W _ { > } ( x )$ , these inequalities imply

$$
- W _ { = } ( x ) \leq W _ { > } ( x ) - W _ { < } ( x ) \leq W _ { = } ( x ) .
$$

Thus the tie value lies in [−1, 1], proving the first identity.

Summing the three cases of Eq. (3) gives

$$
\sum _ { i } w _ { i } \sigma _ { i } ( x ) = W _ { < } ( x ) - W _ { > } ( x ) + W _ { = } ( x ) \frac { W _ { > } ( x ) - W _ { < } ( x ) } { W _ { = } ( x ) } = 0 .
$$

The third identity follows directly from the sign of $u ( x ) - h _ { i } ( x )$

Finally, at least half the weight lies on values at least $u ( x )$ , so

$$
m ( x ) = \sum _ { i \in [ n ] } w _ { i } h _ { i } ( x ) \geq u ( x ) \sum _ { i : h _ { i } ( x ) \geq u ( x ) } w _ { i } \geq { \frac { u ( x ) } { 2 } } .
$$

Integrating gives $U ( { \mathcal { X } } ) \leq 2$

The fractional tie value is essential to exact balance: replacing it by a fixed sign or by zero does not preserve $\begin{array} { r } { \sum _ { i } w _ { i } \sigma _ { i } ( x ) = 0 } \end{array}$ in general. At any point where all candidate densities vanish, $u ( x ) = 0 { \mathrm { . } }$ $W _ { < } ( x ) = W _ { > } ( x ) = 0$ , and $W _ { = } ( x ) = 1 , s 0 \sigma _ { i } ( x ) = 0$ for every �. This includes target samples that fall in the measurable set $\{ x : h _ { 1 } ( x ) = \cdots = h _ { n } ( x ) = 0 \}$

## 3.2 Rounded weights and proposal sampling

We construct a proposal distribution $M ^ { \prime }$ by rounding the weights to dyadic probabilities. This construction ensures $M ^ { \prime } \ge M / 2$ and allows exact sampling using a uniform integer draw followed by a sample from one of the input distributions.

Choose the least power of two $N \geq n$ , so $n \leq N < 2 n .$ once for the entire run. Given �, first set $a _ { i } = \lceil N w _ { i } \rceil$ . Increase $a _ { 1 }$ by $2 N - \textstyle \sum _ { i } a _ { i } ,$ so that the final integer weights sum to $2 N _ { ; }$ , and define the rounded weights, mixture, and density by

$$
w _ { i } ^ { \prime } = \frac { a _ { i } } { 2 N } , \qquad M ^ { \prime } = \sum _ { i } w _ { i } ^ { \prime } H _ { i } , \qquad m ^ { \prime } ( x ) = \sum _ { i } w _ { i } ^ { \prime } h _ { i } ( x ) .
$$

Lemma 3.2 (Proposal properties). For every $x \in \mathcal { X }$ and $i \in [ n ]$

$$
w _ { i } ^ { \prime } \geq \frac { w _ { i } } { 2 } , \qquad w _ { i } ^ { \prime } \geq \frac { 1 } { 2 N } , \qquad m ^ { \prime } ( x ) \geq \frac { m ( x ) } { 2 } , \qquad m ^ { \prime } ( x ) \geq \frac { u ( x ) } { 4 } .
$$

ProofofLemma 3.2. Before increasing $a _ { 1 }$ , positivity of the weights gives max $\{ N w _ { i } , 1 \} \le a _ { i } < N w _ { i } + 1$ , and hence $\textstyle \sum _ { i } a _ { i } < N + n \leq 2 N$ . The remaining integer mass is nonnegative, and adding it to $a _ { 1 }$ preserves $a _ { i } \geq N w _ { i }$ and $a _ { i } \geq 1$ . Dividing by 2� gives the first two inequalities. Multiplying $w _ { i } ^ { \prime } \geq w _ { i } / 2$ by $h _ { i } ( x )$ and summing gives $m ^ { \prime } ( x ) \geq m ( x ) / 2 ;$ the last inequality follows from Lemma 3.1. □

Sampling from $M ^ { \prime }$ . Draw $I \sim \mathrm { U n i f } ( [ 2 N ] )$ in unit time. Scan the cumulative sums $\textstyle \sum _ { k \leq i } a _ { k }$ to find the unique index � such that $\textstyle \sum _ { k < i } a _ { k } < I \leq \sum _ { k \leq i } a _ { k }$ , and return a fresh sample from $H _ { i } .$ . The index equals � with probability $a _ { i } / ( 2 N ) = w _ { i } ^ { \prime } ,$ so the returned point has law $M ^ { \prime }$ . The sampling time is $O ( n )$ : the uniform draw and candidate-sample call each take unit time, and the cumulative scan takes $O ( n )$ time.

## 3.3 The iterative procedure

The algorithm is given in Algorithm 1. Define the number of rounds

$$
T = \left\lceil 4 0 0 \varepsilon ^ { - 2 } \left\lceil \log \frac { 3 n } { \delta } \right\rceil \right\rceil .\tag{4}
$$

Let $( \boldsymbol { \mathcal { F } } _ { t } ) _ { t = 0 } ^ { T }$ be the filtration generated by the random choices in the algorithm, with $\mathcal { F } _ { 0 }$ containing the fixed input. Conditional on $\mathcal { F } _ { t - 1 }$ , the weights $\mathbf { w } _ { t } = ( w _ { t , 1 } , \dots , w _ { t , n } )$ , median measure $U _ { t }$ , tests $\sigma _ { t , i }$ , and proposal $M _ { t } ^ { \prime }$ are fixed. Draw conditionally independent fresh samples $X _ { t } \sim P$ and $Y _ { t } \sim M _ { t } ^ { \prime }$ , and define

$$
\rho _ { t } = \frac { u _ { t } ( Y _ { t } ) } { m _ { t } ^ { \prime } ( Y _ { t } ) } ,\tag{5}
$$

$$
g _ { t , i } = \frac { 1 } { 2 } \big ( \rho _ { t } \sigma _ { t , i } ( Y _ { t } ) - \sigma _ { t , i } ( X _ { t } ) \big ) ,\tag{6}
$$

$$
\hat { D } _ { t , i } = \frac { ( u _ { t } ( Y _ { t } ) - h _ { i } ( Y _ { t } ) ) _ { + } } { m _ { t } ^ { \prime } ( Y _ { t } ) } + \frac { 1 - \rho _ { t } } { 2 } .\tag{7}
$$

Since every $w _ { t , i } ^ { \prime } > 0$ , if $m _ { t } ^ { \prime } ( x ) = 0$ then all $h _ { i } ( x )$ and $\boldsymbol u _ { t } ( \boldsymbol x )$ vanish. Ratios with denominator $m _ { t } ^ { \prime } ( x )$ are defined to be zero. This convention preserves the integral identities, and $M _ { t } ^ { \prime }$ assigns that set probability zero. The expectation calculations in Section 4.2 make this explicit.

Gains and distance estimates. Starting from uniform weights, we increase the weight of distributions with positive gain through the multiplicative update in Algorithm 1. The gain compares the median measure with the target through the balanced tests, while the distance estimates the distance from a distribution to the median measure. More precisely, Section 4.2 proves that

$$
\mathbb { E } [ g _ { t , i } \mid \mathcal { F } _ { t - 1 } ] = \frac { 1 } { 2 } \int \sigma _ { t , i } d ( U _ { t } - P ) , \qquad \mathbb { E } [ \hat { D } _ { t , i } \mid \mathcal { F } _ { t - 1 } ] = \frac { 1 } { 2 } \| U _ { t } - H _ { i } \| _ { 1 } .
$$

Algorithm 1: Proper hypothesis selection by unnormalized weighted medians   
Input: Sample oracle for $P ;$ sample and density oracles for $H _ { 1 } , \dots , H _ { n } ;$ accuracy � and failure   
probability �.   
Output: An input distribution $H _ { \hat { i } } .$   
1 Choose � as the least power of two at least $n ;$ set $T$ as in Eq. (4)   
2 Initialize $w _ { 1 , i } = 1 / \ i$ � and $\overline { { \hat { D } } } _ { i } = 0$ for every $i \in [ n ]$   
3 for $t = 1 , \dots , T$ do   
4 Use $\mathbf { w } _ { t }$ to define $U _ { t }$ and $\sigma _ { t , i }$ by Eq. (3)   
5 Construct $\mathbf { w } _ { t } ^ { \prime }$ and $M _ { t } ^ { \prime }$ using the rounding procedure of Lemma 3.2   
6 Draw independent fresh samples $X _ { t } \sim P$ and $Y _ { t } \sim M _ { t } ^ { \prime }$   
7 Evaluate all $h _ { i } ( X _ { t } ) , h _ { i } ( Y _ { t } )$ , the two weighted medians and their tests, and $\begin{array} { r } { m _ { t } ^ { \prime } ( Y _ { t } ) = \sum _ { i } w _ { t , i } ^ { \prime } h _ { i } ( Y _ { t } ) } \end{array}$   
8 Compute $\rho _ { t }$ by Eq. (5)   
9 for $i = 1 , \ldots , n$ do   
10 Compute $g _ { t , i }$ and $\hat { D } _ { t , i }$ by Eq. (6) and Eq. (7)   
11 $w _ { t + 1 , i } \gets w _ { t , i } \big ( 1 + ( \varepsilon / 5 0 ) g _ { t , i } \big )$   
12 $\overline { { \hat { D } } } _ { i } \gets \overline { { \hat { D } } } _ { i } + \hat { D } _ { t , i } / T$   
13 Return $H _ { \hat { i } }$ for $\begin{array} { r } { \hat { i } \in \mathrm { a r g m i n } _ { i \in [ n ] } \overline { { \hat { D } } } _ { i } . } \end{array}$ , breaking ties by a fixed rule

Each round takes linear time. The algorithm represents $U _ { t }$ implicitly through the weights and the densities of the candidate distributions. In each round, it computes the weighted median and tests only at $X _ { t }$ and $Y _ { t } .$ . These computations take $O ( n )$ time by Section A. Rounding, proposal sampling, density evaluation, and the updates also take $O ( n )$ time, giving a total of $O ( n )$ per round.

## 4 Analysis of the algorithm

## 4.1 Bounded estimates and valid weights

Suppose that the current weights are positive and sum to one. By Lemma 3.2, $m _ { t } ^ { \prime } \ge u _ { t } / 4 .$ , so $0 \leq \rho _ { t } \leq 4$ and

$$
0 \leq \frac { ( u _ { t } ( Y _ { t } ) - h _ { i } ( Y _ { t } ) ) _ { + } } { m _ { t } ^ { \prime } ( Y _ { t } ) } \leq \rho _ { t } .
$$

Together with $| \sigma _ { t , i } | \leq 1$ , these bounds imply

$$
| g _ { t , i } | \leq \frac { 5 } { 2 } , \qquad - \frac { 3 } { 2 } \leq \hat { D } _ { t , i } \leq \frac { 5 } { 2 } .\tag{8}
$$

Validity of the update. The initial weights are positive and sum to one. The balance identity in Lemma 3.1 holds pointwise, and therefore

$$
\sum _ { i } w _ { t , i } g _ { t , i } = \frac { 1 } { 2 } \left( \rho _ { t } \sum _ { i } w _ { t , i } \sigma _ { t , i } ( Y _ { t } ) - \sum _ { i } w _ { t , i } \sigma _ { t , i } ( X _ { t } ) \right) = 0 .
$$

Consequently,

$$
\sum _ { i } w _ { t + 1 , i } = \sum _ { i } w _ { t , i } + { \frac { \varepsilon } { 5 0 } } \sum _ { i } w _ { t , i } g _ { t , i } = 1 .
$$

Moreover, $| ( \varepsilon / 5 0 ) g _ { t , i } | \le \varepsilon / 2 0 \le 1 / 2 0$ , so every update factor is positive. Induction shows that all weight vectors, medians, and proposals are well defined, and that $\operatorname { E q . } \left( 8 \right)$ holds in every round.

The following standard multiplicative-weights argument [AHK12] uses exact balance to obtain a bound for every distribution.

Lemma 4.1 (Deterministic gain bound). For every sample sequence and every $i \in [ n ]$

$$
{ \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } g _ { t , i } \leq { \frac { 5 0 \ln n } { \varepsilon T } } + { \frac { \varepsilon } { 8 } } \leq { \frac { 7 \varepsilon } { 3 2 } } .
$$

Proof. Since $w _ { t + 1 , i } = w _ { t , i } ( 1 + ( \varepsilon / 5 0 ) g _ { t , i } )$ and $w _ { 1 , i } = 1 / n$ , it holds that

$$
\begin{array} { r l r } {  { \ln w _ { T + 1 , i } = - \ln n + \sum _ { t = 1 } ^ { T } \ln \bigl ( 1 + ( \varepsilon / 5 0 ) g _ { t , i } \bigr ) } } \\ & { } & \\ & { \geq - \ln n + \frac { \varepsilon } { 5 0 } \displaystyle \sum _ { t = 1 } ^ { T } g _ { t , i } - \frac { \varepsilon ^ { 2 } } { 2 5 0 0 } \sum _ { t = 1 } ^ { T } g _ { t , i } ^ { 2 } } \\ & { } & \\ & { \geq - \ln n + \frac { \varepsilon } { 5 0 } \displaystyle \sum _ { t = 1 } ^ { T } g _ { t , i } - \frac { \varepsilon ^ { 2 } T } { 4 0 0 } , } \end{array}
$$

where the first inequality uses ln $\begin{array} { r } { \iota ( 1 + x ) \geq x - x ^ { 2 } \mathrm { f o r } | x | \leq 1 / 2 } \end{array}$ , and the second uses $g _ { t , i } ^ { 2 } \leq 2 5 / 4$ . The weights are positive and sum to one, so ln $w _ { T + 1 , i } \leq 0$ . Rearranging gives

$$
{ \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } g _ { t , i } \leq { \frac { 5 0 \ln n } { \varepsilon T } } + { \frac { \varepsilon } { 8 } } .
$$

Finally, since ln $2 < 3 / 4$ , we have ln $n \leq \ln ( 3 n / \delta ) \leq 3 [ \log ( 3 n / \delta ) ] / 4$ . Together with $T \geq 4 0 0 \varepsilon ^ { - 2 } [ \log ( 3 n / \delta ) ]$ this gives

$$
\frac { 5 0 \ln n } { \varepsilon T } + \frac { \varepsilon } { 8 } \leq \frac { 3 \varepsilon } { 3 2 } + \frac { \varepsilon } { 8 } = \frac { 7 \varepsilon } { 3 2 } .
$$

□

## 4.2 Conditional expectations

Recall the filtration $\textstyle \mathcal { F } _ { t }$ from Section 3.3. Define

$$
D _ { t , i } = \frac 1 2 \| U _ { t } - H _ { i } \| _ { 1 } , \qquad \overline { { { D } } } _ { i } = \frac 1 T \sum _ { t = 1 } ^ { T } D _ { t , i } .
$$

These quantities use the signed-measure norm because $U _ { t }$ need not have mass one. By Lemma 3.1, $0 \leq U _ { t } ( \mathcal { X } ) \leq 2$ , and hence

$$
0 \leq D _ { t , i } \leq { \frac { 1 } { 2 } } \int ( u _ { t } + h _ { i } ) d \mu = { \frac { U _ { t } ( { \mathcal { X } } ) + 1 } { 2 } } \leq { \frac { 3 } { 2 } } .
$$

Lemma 4.2 (Conditional expectations). For every round $t \in [ T ]$ and coordinate $i \in [ n ]$

$$
D _ { t , i } \leq \mathbb { E } [ g _ { t , i } \ | \ \mathcal { F } _ { t - 1 } ] + d _ { \mathrm { T V } } ( P , H _ { i } ) , \qquad \mathbb { E } [ \hat { D } _ { t , i } \ | \ \mathcal { F } _ { t - 1 } ] = D _ { t , i } .
$$

Proof. Conditional on $\mathcal { F } _ { t - 1 }$ , the current weights, median, tests, and proposal are fixed, and $X _ { t } \sim P$ and $Y _ { t } \sim M _ { t } ^ { \prime }$ are fresh and independent. By the sign identity in Lemma 3.1,

$$
\begin{array} { l } { { \displaystyle D _ { t , i } = \frac { 1 } { 2 } \int | u _ { t } ( x ) - h _ { i } ( x ) | d \mu } } \\ { { \displaystyle \quad = \frac { 1 } { 2 } \int \sigma _ { t , i } d ( U _ { t } - H _ { i } ) } } \\ { { \displaystyle \quad = \frac { 1 } { 2 } \int \sigma _ { t , i } d ( U _ { t } - P ) + \frac { 1 } { 2 } \int \sigma _ { t , i } d ( P - H _ { i } ) . } } \end{array}\tag{9}
$$

Importance sampling gives

$$
\begin{array} { c } { { \mathbb E [ \rho _ { t } \sigma _ { t , i } ( Y _ { t } ) \mid { \mathcal F } _ { t - 1 } ] = \displaystyle \int _ { \{ x : m _ { t } ^ { \prime } ( x ) > 0 \} } \frac { u _ { t } ( x ) \sigma _ { t , i } ( x ) } { m _ { t } ^ { \prime } ( x ) } m _ { t } ^ { \prime } ( x ) d \mu ( x ) } } \\ { { = \displaystyle \int \sigma _ { t , i } d U _ { t } . } } \end{array}
$$

The first equality is the definition of expectation under $M _ { t } ^ { \prime } .$ . The last equality uses $u _ { t } = 0$ wherever $m _ { t } ^ { \prime } = 0$ with the ratio defined to be zero on this null set of $M _ { t } ^ { \prime } .$ . Since $\begin{array} { r } { \mathbb { E } [ \sigma _ { t , i } ( X _ { t } ) \mid \mathcal { F } _ { t - 1 } ] = \int \ \sigma _ { t , i } \ : d P } \end{array}$ , we obtain

$$
\mathbb { E } [ g _ { t , i } \mid { \mathcal { F } } _ { t - 1 } ] = { \frac { 1 } { 2 } } \int \sigma _ { t , i } d ( U _ { t } - P ) .\tag{10}
$$

Combining Eqs. (9) and (10) and applying Eq. (1) yields

$$
\begin{array} { r l r } {  { D _ { t , i } = \mathbb { E } \bigl [ g _ { t , i } \mid \mathcal { F } _ { t - 1 } \bigr ] + \frac { 1 } { 2 } \int \sigma _ { t , i } d ( P - H _ { i } ) } } \\ & { } & { \leq \mathbb { E } \bigl [ g _ { t , i } \mid \mathcal { F } _ { t - 1 } \bigr ] + d _ { \mathrm { T V } } ( P , H _ { i } ) . \quad } \end{array}
$$

For the distance estimate, Eq. (2) gives

$$
D _ { t , i } = \int ( u _ { t } ( x ) - h _ { i } ( x ) ) _ { + } d \mu ( x ) + \frac { 1 - U _ { t } ( \mathcal { X } ) } { 2 } .\tag{11}
$$

Both $u _ { t }$ and $\left( { { u } _ { t } } - { { h } _ { i } } \right)$ vanish wherever $m _ { t } ^ { \prime } = 0$ , so the same importance-sampling calculation yields

$$
\mathbb { E } \left[ \left. \frac { ( u _ { t } ( Y _ { t } ) - h _ { i } ( Y _ { t } ) ) _ { + } } { m _ { t } ^ { \prime } ( Y _ { t } ) } \right| { \mathcal { F } } _ { t - 1 } \right] = \int ( u _ { t } - h _ { i } ) _ { + } d \mu ,
$$

Substituting into Eq. (7) proves

$$
\mathbb E [ \hat { D } _ { t , i } \mid \mathcal F _ { t - 1 } ] = D _ { t , i } .\tag{12}
$$

Why use the mass correction? The direct importance estimate $| u _ { t } ( Y _ { t } ) - h _ { i } ( Y _ { t } ) | / ( 2 m _ { t } ^ { \prime } ( Y _ { t } ) )$ introduces the ratio $h _ { i } ( Y _ { t } ) / m _ { t } ^ { \prime } ( Y _ { t } )$ , for which the bound $1 / w _ { t , i } ^ { \prime } \le 2 N$ depends on �. In contrast, $0 \leq ( u _ { t } - h _ { i } ) _ { + } \leq u _ { t } \leq 4 m _ { t } ^ { \prime } .$ so the positive-part ratio in Eq. (7) is at most four. The mass correction $\frac { 1 - U _ { t } ( \mathcal { X } ) } { 2 }$ in Eq. (11) makes this estimate unbiased even though $U _ { t }$ is not normalized. The resulting estimate $\hat { D } _ { t , i }$ may be negative, but its conditional mean is nonnegative and its range has length four, independently of � and the current weights.

## 4.3 Concentration for the adaptive sequence

The medians and proposals depend on previous samples, so we use a conditional form of the Azuma– Hoefding inequality [Azu67, Hoe63].

Lemma 4.3 (Azuma–Hoefding inequality). Let $( \boldsymbol { \mathcal { F } } _ { t } ) _ { t = 0 } ^ { T }$ be a filtration and let $( \xi _ { t } ) _ { t = 1 } ^ { T }$ be a sequence ofrandom variables, each $\textstyle \mathcal { F } _ { t }$ -measurable with $\mathbb { E } [ \xi _ { t } \mid { \mathcal { F } } _ { t - 1 } ] = 0$ . Suppose that $a _ { t } \leq \xi _ { t } \leq b _ { t }$ almost surely, where $a _ { t } , b _ { t }$ are $\mathcal { F } _ { t - 1 }$ -measurable and $b _ { t } - a _ { t } \leq c$ for a constant $c > 0$ . Then for $s > 0$

$$
\mathrm { P r } \left\{ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \xi _ { t } > s \right\} \leq \exp \left( - \frac { 2 T s ^ { 2 } } { c ^ { 2 } } \right) .
$$

The same bound holds for the lower tail.

Lemma 4.4 (Concentration of gains and distance estimates). With probability at least $1 - \delta$ , the following inequalities hold simultaneously for every $i \in [ n ]$

$$
\begin{array} { c l c r } { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } [ g _ { t , i } \mid { \mathcal F } _ { t - 1 } ] \leq \frac { 1 } { T } \sum _ { t = 1 } ^ { T } g _ { t , i } + \frac { 5 \varepsilon } { 3 2 } , } & \\ { \displaystyle \left| \overline { { \hat { D } } } _ { i } - \overline { { D } } _ { i } \right| \leq \frac { \varepsilon } { 8 } . } & \end{array}
$$

Proof. The choice of � in Eq. (4) and ln $2 < 3 / 4$ imply

$$
{ \frac { T \varepsilon ^ { 2 } } { 5 1 2 } } \geq { \frac { 2 5 } { 3 2 } } \left\lceil \log { \frac { 3 n } { \delta } } \right\rceil \geq \ln { \frac { 3 n } { \delta } } .
$$

For a fixed $i ,$ the diferences $\begin{array} { r } { \xi _ { t } = \mathbb { E } [ g _ { t , i } \mid \mathcal { F } _ { t - 1 } ] - g _ { t , i } } \end{array}$ have conditional mean zero and conditional range of length five by Eq. (8); centering shifts an interval without changing its length. Apply Lemma 4.3 with $c = 5$ and $s = 5 \varepsilon / 3 2$ to bound the probability that the first inequality fails for this � by exp $\left( - T \varepsilon ^ { 2 } / 5 1 2 \right) \le \delta / ( 3 n )$ A union bound over $i \in [ n ]$ gives failure probability at most $\delta / 3$

By Lemma 4.2, $\hat { D } _ { t , i } - D _ { t , i }$ has conditional mean zero, and its conditional range has length four by Eq. (8). Apply both tails of Lemma 4.3 with $c = 4$ and $s = \varepsilon / 8$ to bound the probability that the second inequality fails for this � by $2 \exp ( - T \varepsilon ^ { 2 } / 5 1 2 ) \le 2 \delta / ( 3 n )$ . A union bound over $i \in [ n ]$ gives failure probability at most $2 \delta / 3$

Combining the two failure probabilities proves the simultaneous guarantee. Neither independence across coordinates nor independence between the gain and score estimates is needed. □

Combining Lemmas 4.1, 4.2 and 4.4 gives the following corollary.

Corollary 4.5. With probability at least $1 - \delta _ { z }$ the following inequalities hold simultaneously for every � ∈ [�]:

$$
\overline { { { D } } } _ { i } \leq d _ { \mathrm { T V } } ( P , H _ { i } ) + \frac { 3 \varepsilon } { 8 } , \qquad | \overline { { { \hat { D } } } } _ { i } - \overline { { { D } } } _ { i } | \leq \frac { \varepsilon } { 8 } .
$$

## 4.4 Proof of Theorem 1.1

Approximation factor. Work on the event in Corollary 4.5, which has probability at least $1 - \delta$ . Fix $i ^ { * } \in \mathrm { a r g m i n } _ { i } d _ { \mathrm { T V } } ( P , H _ { i } )$ . Since the output index <sup>̂</sup>� minimizes the estimated average distance, the average distance between the median measures and the output distribution is bounded by

$$
\overline { { D } } _ { \hat { i } } \leq \overline { { \hat { D } } } _ { \hat { i } } + \frac { \varepsilon } { 8 } \leq \overline { { \hat { D } } } _ { \hat { i } ^ { \ast } } + \frac { \varepsilon } { 8 } \leq \overline { { D } } _ { \hat { i } ^ { \ast } } + \frac { \varepsilon } { 4 } \leq \mathrm { O P T } + \frac { 5 \varepsilon } { 8 } .
$$

where the first, third, and fourth inequalities follow from Corollary 4.5. Moreover, the median measures are close to � on average, since

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \frac { 1 } { 2 } \| P - U _ { t } \| _ { 1 } \leq \frac { 1 } { T } \displaystyle \sum _ { t = 1 } ^ { T } \frac { 1 } { 2 } \left( \| P - H _ { i ^ { \star } } \| _ { 1 } + \| H _ { i ^ { \star } } - U _ { t } \| _ { 1 } \right) } & { } \\ { \leq d _ { \mathrm { T V } } ( P , H _ { i ^ { \star } } ) + \overline { { D } } _ { i ^ { \star } } } & { } \\ { \leq 2 \mathrm { O P T } + \frac { 3 \varepsilon } { 8 } . } \end{array}
$$

where the last inequality uses Corollary 4.5. For every �, use $U _ { t }$ as the intermediate measure in the triangle inequality:

$$
d _ { \mathrm { T V } } ( P , H _ { \hat { i } } ) \leq \frac { 1 } { 2 } \| P - U _ { t } \| _ { 1 } + \frac { 1 } { 2 } \| U _ { t } - H _ { \hat { i } } \| _ { 1 } = \frac { 1 } { 2 } \| P - U _ { t } \| _ { 1 } + D _ { t , \hat { i } } .
$$

Averaging the triangle inequality through � rounds yields

$$
\begin{array} { r l } {  { d _ { \mathrm { T V } } ( P , H _ { \widehat { i } } ) \leq \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \frac { 1 } { 2 } \| P - U _ { t } \| _ { 1 } + \overline { { D } } _ { \widehat { i } } } } \\ & { \leq 2 \mathrm { O P T } + \frac { 3 \varepsilon } { 8 } + \mathrm { O P T } + \frac { 5 \varepsilon } { 8 } } \\ & { = 3 \mathrm { O P T } + \varepsilon . } \end{array}
$$

Sample complexity. Each round draws one target sample, thus the algorithm uses exactly � targetsample calls, where $T = O ( \varepsilon ^ { - 2 } \log ( n / \delta ) )$ .

Running time and space. As the analysis in Section 3.3 shows, each round takes $O ( n )$ time. The total running time is therefore $O ( n T ) = O ( n \varepsilon ^ { - 2 } \log ( n / \delta ) )$ .

## AI Disclosure

ChatGPT was used extensively during the preparation of this work. The author suggested using a pointwise median to obtain an improper distribution with approximation factor 2 on the lower-bound instance of [AACS26]. Then, ChatGPT proposed combining weighted medians with multiplicative weight updates. ChatGPT and Claude also assisted with exposition and editing. The author takes full responsibility for the results, proofs, and references.

## A Finding the weighted median in linear time

Ordinary order-statistic selection can be implemented deterministically in linear time by the median-ofmedians algorithm of Blum et al. $\left[ \mathrm { B F P ^ { + } 7 3 } \right]$ . Partition the input into groups of five, sort each group, and recursively select a median of the group medians as a pivot. Partition around this pivot and recurse on the strict side containing the requested rank, unless the rank belongs to the values equal to the pivot. At least half the group medians lie weakly on either side of the pivot. Discarding the possibly incomplete group and the group containing the pivot loses at most two groups on either side. Each remaining full group contributes three values weakly on that side, so each strict side contains at most $7 m / 1 0 + 6$ values, including when values coincide. Thus the worst-case cost satisfies, for suficiently large �,

$$
A ( m ) \leq A ( \lceil m / 5 \rceil ) + A ( \lfloor 7 m / 1 0 \rfloor + 6 ) + O ( m ) = O ( m ) ,
$$

where the linear bound follows by induction since $1 / 5 + 7 / 1 0 < 1$ , with smaller instances handled directly.   
We use this selection procedure to find a weighted median as follows.

Lemma A.1 (Lower weighted quantile selection). Given real values $v _ { 1 } , \dots , v _ { m }$ with positive weights and a target mass � satisfying $\begin{array} { r } { 0 < \tau \leq \sum _ { i = 1 } ^ { m } w _ { i } } \end{array}$ , the smallest input value whose cumulative weight is at least � can be found deterministically in $O ( m )$ worst-case time and $O ( m )$ working space.

Proof. Maintain an active set � and a residual target satisfying $\begin{array} { r } { 0 < \tau \le \sum _ { i \in A } w _ { i } } \end{array}$ . Select an ordinary median by count of the active values, namely an order statistic of rank $\lceil \lvert A \rvert / 2 \rceil$ , using the algorithm above. Write � for its value, partition � into

$$
\begin{array} { r } { A _ { < } = \{ i \in A : v _ { i } < z \} , \qquad A _ { = } = \{ i \in A : v _ { i } = z \} , \qquad A _ { > } = \{ i \in A : v _ { i } > z \} , \qquad A _ { > } = \{ i \in A : v _ { i } > z \} , } \end{array}
$$

and compute $\begin{array} { r } { W _ { L } = \sum _ { i \in A _ { < } } } \end{array}$ �<sub>�</sub> and $\begin{array} { r } { W _ { E } = \sum _ { i \in A _ { = } } w _ { i } } \end{array}$ . Apply the following rules:

1. If $\tau \leq W _ { L }$ , retain $A _ { < }$ and leave � unchanged.

2. If $W _ { L } < \tau \leq W _ { L } + W _ { E }$ , return �.

3. $\mathrm { I f } \ \tau > W _ { L } + W _ { E } ,$ retain $A _ { > }$ and replace � by $\tau - W _ { L } - W _ { E }$

These rules preserve the residual-target invariant and the desired lower quantile. In particular, when $\tau = W _ { L }$ , the target has already been reached strictly below �, so the first branch is necessary. In the last branch the discarded values all precede every remaining value, which justifies subtracting exactly their total weight.

Both strict sides of an ordinary median by count contain at most $\lfloor \left| A \right| / 2 \rfloor$ values. Therefore each nonterminal step at least halves the active-set size, including in the presence of ties. Selection, partitioning, and weight summation take $O ( | A | )$ operations per step, for a total of

$$
O ( m ) + O ( m / 2 ) + O ( m / 4 ) + \cdots = O ( m ) .
$$

The procedure terminates because each nonterminal branch strictly reduces the active set. Arrays for values, weights, and partitions, together with the recursive selection workspace, use $O ( m )$ stored numbers and indices. □

For a density profile $( h _ { 1 } ( x ) , \ldots , h _ { n } ( x ) )$ in round �, apply Lemma A.1 with weights $w _ { t , i }$ and target $1 / 2$ to obtain $\boldsymbol u _ { t } ( \boldsymbol x )$ . Then scan the entire original profile to compute the total weights strictly below, equal to, and strictly above $\boldsymbol u _ { t } ( \boldsymbol x )$ , and scan once more to form all $\sigma _ { t , i } ( x )$ . These totals include the items discarded during selection. The scans cost $O ( n )$ , so computing the median and all signed tests costs $O ( n )$ , with no sorting of the full profile.

## References

[AACS26] Anders Aamand, Maryam Aliakbarpour, Justin Y. Chen, and Sandeep Silwal. How fast can you find a good hypothesis? In Steve Hanneke and Tor Lattimore, editors, Proceedings of Thirty Ninth Conference on Learning Theory, volume 336 of Proceedings of Machine Learning Research, pages 1–2. PMLR, 29 Jun–03 Jul 2026. 1, 2, 11

[ABS24] Maryam Aliakbarpour, Mark Bun, and Adam Smith. Optimal hypothesis selection in (almost) linear time. Advances in Neural Information Processing Systems, 37:141490–141527, 2024. 1, 2

[AHK12] Sanjeev Arora, Elad Hazan, and Satyen Kale. The multiplicative weights update method: a meta-algorithm and applications. Theory ofcomputing, 8(1):121–164, 2012. 2, 8

[Azu67] Kazuoki Azuma. Weighted sums of certain dependent random variables. Tohoku Mathematical Journal, Second Series, 19(3):357–367, 1967. 10

[BBK<sup>+</sup>22] Olivier Bousquet, Mark Braverman, Gillat Kol, Klim Efremenko, and Shay Moran. Statistically near-optimal hypothesis selection. In 2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS), pages 909–919. IEEE, 2022. 1

[BFP<sup>+</sup>73] Manuel Blum, Robert W. Floyd, Vaughan R. Pratt, Ronald L. Rivest, Robert Endre Tarjan, et al. Time bounds for selection. J. Comput. Syst. Sci., 7(4):448–461, 1973. 11

[BKM19] Olivier Bousquet, Daniel Kane, and Shay Moran. The optimal approximation factor in density estimation. In Conference on Learning Theory, pages 318–341. PMLR, 2019. 1

[DL01] Luc Devroye and Gábor Lugosi. Combinatorial methods in density estimation, volume 1. Springer, 2001. 1, 2

[Hoe63] Wassily Hoefding. Probability inequalities for sums of bounded random variables. Journal of the American statistical association, 58(301):13–30, 1963. 10

[MS08] Satyaki Mahalanabis and Daniel Stefankovic. Density estimation in linear time. In Rocco A. Servedio and Tong Zhang, editors, 21st Annual Conference on Learning Theory - COLT 2008, Helsinki, Finland, July 9-12, 2008, pages 503–512. Omnipress, 2008. 1, 2

[Yat85] Yannis G Yatracos. Rates of convergence of minimum distance estimators and kolmogorov’s entropy. The Annals of Statistics, 13(2):768–774, 1985. 1, 2