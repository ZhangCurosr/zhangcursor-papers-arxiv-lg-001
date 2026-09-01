# Minimax bounds for watermarked and masked recursive discrete distribution estimation

Millen Kanabar and Michael Gastpar School of Computer and Communication Sciences, EPFL, Switzerland

## Abstract

Watermarking has been proposed as a way to identify synthetic samples in estimation settings where no metadata is available to distinguish them from real samples, but its precise effects remain unexplored. In the absence of a distinguishing mechanism, it has been shown that adding synthetic samples significantly reduces the marginal efficacy of new real samples. In this work, we study the minimax loss of such recursive discrete distribution estimation in the presence of watermarks in contrast to the unassisted and oracle-assisted losses. When the fraction of real samples vanishes asymptotically, we provide a lower bound that shows that it is impossible to improve performance by adding watermarks unless the false negative rate of detection also vanishes. Additionally, we show that in most regimes, the worst-case losses of a sequence of simple deterministic estimators match the corresponding lower bounds up to constants. Finally, we propose masking, a randomization procedure that narrows the gap in the remaining regimes to a Jensen gap. We conjecture that a tighter lower bound argument can close this gap.

## I. INTRODUCTION

In the generative machine learning literature, model collapse refers to the phenomenon where the underlying distributions of models trained only on their previous generation’s outputs degrade with each iteration. This stands to reason, since finitely many samples disproportionately represent higher-probability outcomes. This leads to increasingly noisy estimates for lower probability outcomes, until the output distribution eventually gets corrupted into a point mass by this sampling noise.

Accumulating each batch of samples used for training and including fresh real samples in each batch can lead to eventual convergence to the ground truth, so long as real samples do not get ‘lost’ in a sea of synthetic ones. Even so, if synthetic samples are not distinguished from real samples, the reinforcement of higher probability symbols in the empirical distribution of the corpus acts as noise and affects performance. In a previous work, we quantify the effect of this noise in the discrete distribution estimation setting [1] via upper and lower minimax bounds. These bounds show that even in the regime where the expected number of real samples asymptotically grows to infinity, if synthetic sampling outpaces real sampling, the margina utility of new real samples is greatly reduced.

In the literature and in practice, watermarks have been proposed as a mechanism to identify synthetic samples and detect their provenances. When reliable, they can offer a useful way to identify and discard synthetic samples from the corpus. In this work, we use reductions from the watermarking-assisted setting to the unassisted setting studied in our previous work to find conditions where watermarking makes a difference.

The problem setting in our work is depicted in Figure 1. New samples added at stage t are assumed to be drawn from the ground truth distribution p with probability $\alpha _ { t }$ and from the previous estimate $\hat { P } _ { t - 1 }$ otherwise. We assume that the mixing factors $\alpha _ { t }$ are known to the estimator beforehand. We also assume that samples do not change appreciably with the application of watermarks, and therefore ignore any distortion in the distribution of synthetic samples due to watermarking. This assumption mirrors watermarking in image models, where a small amount of structured noise is added as a watermark.

The estimator is equipped with a watermark detector. Since watermarking is not a perfect procedure to identify sample provenance (see e.g. [3], [4]), we model the outcome of the detector as a random variable correlated with the provenance of the sample with known output distributions given whether it is real or synthetic. This subsumes both detectors that provide a ‘confidence rating’ instead of a binary outcome and training methods that might use additional metadata collected with each sample as soft watermarks.

Through our lower bound, we observe that watermarking can improve performance significantly only if the false negative rate of the watermark detector shrinks with the fraction of real samples, matching the oracle-assisted setting (equivalently, perfect watermarking) up to constant factors whenever this rate is linear or faster. We also observe that a matching upper bound (up to constant factors) is achieved by simple deterministic estimators in most regimes. However, when synthetic sampling outpaces real sampling extremely rapidly, the performance of these estimators falls short of the lower bound. A similar effect was also observed in the unassisted setting in our previous work. With this in mind, we propose a randomization procedure (which we refer to as masking) that narrows it to a Jensen gap. We conjecture that a tighter lower bound might close this gap.

![](images/aa74d7d520f3015546cd2f7688c1e4cc491f483c029dce56afc29bb2d01d2ec9.jpg)  
Fig. 1. Stage t in the accumulation workflow with watermarking

## A. Notation

Deterministic quantities and functions are represented by lowercase Greek and Roman alphabets; random variables are represented by uppercase Roman alphabets. The probability simplex corresponding to the alphabet of size k is denoted as $\Delta _ { k }$ We use $[ a : b ]$ to denote $\{ x \in \mathbb { N } : a \leqslant x \leqslant b \}$ . The unit vector in with component j equal to 1 is denoted as $e _ { j }$

The $j ^ { \mathrm { t h } }$ component of a random or deterministic vector P or $p$ is indexed by square brackets as $P [ j ]$ and $p [ j ]$ respectively, and the probability of an event A under distribution Q is denoted as $Q \{ A \}$ . The n-fold product of a distribution Q is denoted as $Q ^ { n }$

For $t \geqslant 0$ and functions $f , g ,$ , we write $g ( t ) = o ( f ( t ) )$ if $\begin{array} { r } { \operatorname* { l i m } _ { t \uparrow \infty } g ( t ) / f ( t ) = 0 ; g ( t ) = \mathcal { O } ( f ( t ) ) } \end{array}$ q if there exists a constant $a \geqslant 0$ such that $g ( t ) \leqslant a f ( t ) \forall t \geqslant 0$ and $g ( t ) = \Theta ( f ( t ) ) { \mathrm { ~ i f ~ } } g ( t ) = { \mathcal { O } } ( f ( t ) )$ and $f ( t ) = \mathcal { O } ( g ( t ) )$ .

## B. Related work

Model collapse has been widely observed empirically e.g. in [5]–[8] and analyzed theoretically for discrete distributions [5], [9], [10], parametric models [11]–[14], regression and kernel density estimation [15]–[21], and diffusion models [15], [22]. Various mechanisms to mitigate model collapse have also been proposed, such as accumulating data [9], [12], [17], [23], [24], watermarking [5], [25]–[27], and data curation [18], [28], [29]. Our masking procedure is inspired, in part, by recent work [30] that reduces recursive learning to the PU learning setting [31]. Additionally, in our setting, we only consider samples from the ground truth distribution and known estimates; similar bounds for settings such as [32] that use samples from unknown distributions close to the ground truth remain open.

## II. PROBLEM SETUP

Let $p \in \Delta _ { k }$ be the distribution being estimated, and $\left( n _ { t } , \alpha _ { t } \right) _ { t \geqslant 0 }$ be a sequence over $\mathbb { N } \times [ 0 , 1 ]$ . The estimators might also be equipped with additional randomness $U _ { t } ;$ ; the randomized estimates are available to the estimator at each subsequent stage of estimation. Sampling and estimation then follows the following procedure:

## A. Sampling procedure

a) Initial samples and estimates: At stage 0, the estimator receives $n _ { 0 }$ samples $X _ { 0 } ^ { n _ { 0 } }$ sampled i.i.d. from $p .$ The estimator releases the estimate $\widehat { P } _ { 0 } = \widehat { p } ( X _ { 0 } ^ { n _ { 0 } } , U _ { 0 } )$ and retains it for subsequent stages of estimation.

b) Recursive estimation: At every subsequent stage $t \geqslant 1$ , random variables $Y _ { t } ^ { n _ { t } } \sim ( \mathrm { B e r } ( \alpha _ { t } ) ) ^ { n _ { t } }$ are drawn. These will determine the provenances of the samples. Subsequently, samples $X _ { t } ^ { n _ { t } }$ are drawn conditionally independently from $\bar { P } ( X | \widehat { P } _ { t - 1 } , Y )$ given as

$$
\begin{array} { r } { \bar { P } ( X = x | \widehat { P } _ { t - 1 } , Y ) = \left\{ \begin{array} { l l } { p [ x ] } & { Y = 1 } \\ { \widehat { P } _ { t - 1 } [ x ] } & { Y = 0 . } \end{array} \right. } \end{array}\tag{1}
$$

The estimate $\widehat { P } _ { t } : = \widehat { p } _ { t } \left( ( X _ { i } ^ { n _ { i } } , U _ { i } ) _ { t \in [ 0 : t ] } \right)$ is released and retained for future estimation. For ease of notation, we define $Y _ { 0 } ^ { n _ { 0 } }$ as the all-ones vector. We will refer to the joint distribution of the whole sequence $( U _ { i } , X _ { i } ^ { n _ { i } } ) _ { i \in [ 0 : t ] }$ as $\bar { \mathbf { P } } _ { t , p }$

c) Watermarking: Additionally, when a watermarking mechanism is present, we model the output of the watermarking detector as random variables $W _ { t } ^ { n _ { t } } \in \mathcal { W } ^ { n _ { t } }$ , with $| \mathcal { W } | = k _ { W }$ . Given $Y _ { t }$ , these are drawn conditionally independently from

$$
\bar { P } _ { W _ { t } | Y _ { t } } ( w | y ) = \left\{ \begin{array} { l l } { Q _ { 0 , t } [ w ] } & { y = 0 } \\ { Q _ { 1 , t } [ w ] } & { y = 1 . } \end{array} \right.\tag{2}
$$

Once again, for ease of notation, we define $W _ { 0 } ^ { n _ { 0 } }$ as the all-ones vector. For conciseness, we will drop the dependence on t and only write $Q _ { 0 }$ and $Q _ { 1 }$ , however, the proofs do not depend on these staying constant. In this setting, the watermark-aided estimates $\widehat { P } _ { t } ^ { ( \mathrm { w } ) } : = \widehat { p } _ { t } ^ { ( \mathrm { w } ) } \ : \big ( ( X _ { i } ^ { n _ { i } } , W _ { i } ^ { n _ { i } } , U _ { i } ) _ { i \in [ 0 : t ] } \big )$ are released and retained for subsequent stages. We also add a superscript to the joint distribution of all the information available $\bar { \mathbf { P } } _ { t , p } ^ { ( \mathrm { w } ) }$ to indicate the watermarking setting. The marginal distribution of the watermark $\alpha _ { t } Q _ { 1 } + ( 1 - \alpha _ { t } ) Q _ { 0 }$ is denoted as $\textstyle { \bar { Q } } _ { \alpha _ { t } }$

For the rest of the paper, we use the term ‘watermark’ to refer to the output of the detector.

d) Oracle-assisted baseline: We consider estimators to be ‘oracle-assisted’ when the sample provenances $( Y _ { i } ^ { n _ { i } } )$ are also available to the estimator via an oracle; this information is lost otherwise. In this setting, oracle-assisted estimates, denoted as $\widehat { P } _ { t } ^ { ( \mathrm { o } ) } : = \widehat { p } _ { t } ^ { ( \mathrm { o } ) } \left( ( X _ { i } ^ { n _ { i } } , Y _ { i } ^ { n _ { i } } , U _ { i } ) _ { i \in [ 0 : t ] } \right)$ , are released and retained in memory. As with watermarking, we also add the superscript $\bar { \mathbf { P } } _ { t , p } ^ { ( \mathrm { o } ) }$ to indicate the oracle-assisted setting.

We use the unassisted and oracle-assisted performance of estimators as lower and upper baselines in this work: good watermarks and will get the loss as close to the oracle-assisted performance as possible, whereas a loss close to the unassisted performance will signal ineffective watermarks.

## B. Measuring performance

We use the minimax expected $\ell _ { 2 } ^ { 2 }$ loss as the figure of merit.

Definition 1 (Minimax expected loss): The minimax expected loss $\boldsymbol { r } _ { t , k }$ of the sequence $\widehat { p } _ { t }$ is given as

$$
r _ { t , k } : = \operatorname* { i n f } _ { \hat { p } _ { t } } \operatorname* { s u p } _ { p \in \Delta _ { k } } E _ { \bar { \mathbf { P } } _ { t , p } } \left[ \ell _ { 2 } ^ { 2 } \left( \widehat { P } _ { t } , p \right) \right] .\tag{3}
$$

The oracle-assisted and watermark-assisted minimax losses $r _ { t , k } ^ { ( \mathrm { o } ) }$ and $r _ { t , k } ^ { ( \mathrm { w } ) }$ are defined similarly.

When estimating a discrete distribution $p \in \Delta _ { k }$ from n i.i.d. samples, the minimax $\ell _ { 2 } ^ { 2 }$ loss is $\Theta ( ^ { 1 } / n )$ . Along similar lines, in the recursive estimation settings in this work, we informally refer to the multiplicative inverse of the expected $\ell _ { 2 } ^ { 2 }$ losses (suitably normalized by constants) as the ‘effective sample size’. For example, using [1, Theorems 1 & 2], under most regimes in the unassisted setting, we find that the minimax loss is upper and lower bounded as

$$
\frac { c _ { 1 } } { \displaystyle \sum _ { i = 0 } ^ { t } n _ { i } \alpha _ { i } ^ { 2 } } \leqslant r _ { t , k } \leqslant \frac { c _ { 2 } } { \displaystyle \sum _ { i = 0 } ^ { t } n _ { i } \alpha _ { i } ^ { 2 } }
$$

for some positive constants $c _ { 1 }$ and $c _ { 2 }$ independent of the sequences $n _ { t } , \alpha _ { t }$ . Consequently, we refer to the denominator $\sum _ { i = 0 } ^ { t } n _ { i } \alpha _ { i } ^ { 2 }$ as the effective sample size, and the individual terms $n _ { i } \alpha _ { i } ^ { 2 }$ as the effective sample size at stage i.

## III. MAIN RESULTS

To help state the results, we first define the memory term that encapsulates the effect estimates have on lower bounds for future stages.

Definition 2 (Memory terms for the lower bound): The memory term $g _ { t } ( \rho )$ (and, similarly, $g _ { t } ^ { ( \mathrm { w } ) } ( \rho ) )$ is defined as

$$
g _ { t } ( \rho ) = \operatorname* { s u p } _ { \stackrel { p \in \Delta _ { k } } { x \in \mathcal { X } } } { \bar { \mathbf { P } } } _ { t , p } \left\{ \left| { \widehat { P } } _ { t - 1 } [ x ] - p [ x ] \right| \geqslant \rho \right\} .\tag{4}
$$

The term $h _ { t } ( \rho )$ (and similarly $h _ { t } ^ { ( \mathrm { w } ) } ( \rho ) )$ is defined as

$$
h _ { t } ( \boldsymbol { \rho } ) = \sum _ { i = 1 } ^ { t } n _ { i } \alpha _ { i } g _ { i } ( \boldsymbol { \rho } ) .\tag{5}
$$

Notice that the memory terms depend on the deviation probabilities of the previous estimators. We comment on the necessity of these in our bounds in Section IV-B where we discuss the masking estimator.

## A. Minimax bounds with watermarking

Theorem 1 (Lower bound with watermarks): Consider the watermark-aided recursive distribution estimation problem with tuples $( U _ { i } , X _ { i } ^ { n _ { i } } , W _ { i } ^ { n _ { i } } ) _ { i \in [ 0 : t ] }$ distributed according to $\bar { \mathbf { P } } _ { t , p } ^ { ( \mathrm { w } ) }$ . There exists a constant m independent of t such that

$$
r _ { t , k } ^ { ( \mathrm { w } ) } \geqslant \frac { m } { \sum _ { i = 0 } ^ { t } n _ { i } \alpha _ { i } ^ { 2 } \left( 1 + \chi ^ { 2 } \left( Q _ { 1 } \| \bar { Q } _ { \alpha _ { i } } \right) \right) + h _ { t } ^ { ( \mathrm { w } ) } \left( { 1 } / { 4 k } \right) }\tag{6}
$$

whenever $\begin{array} { r } { \sum _ { i = 0 } ^ { t } n _ { i } \alpha _ { i } ^ { 2 } \left( 1 + \chi ^ { 2 } \left( Q _ { 1 } \lVert \bar { Q } _ { \alpha _ { i } } \right) \right) + h _ { t } ^ { ( \mathrm { w } ) } \left( { 1 } / { 4 k } \right) \geqslant k / 4 , } \end{array}$ , with $h _ { t } ^ { ( \mathrm { w } ) } ( ^ { { 1 } / { 4 k } } )$ defined in Definition 2.

We also show that the lower bound at each stage t is achievable by a sequence of deterministic estimators in most regimes under mild parametric assumptions.

Theorem 2 (Upper bound with watermarks): For the problem setup given above, there exists a sequence of deterministic estimators for which the worst-case loss is bounded above as

$$
\operatorname* { s u p } _ { p } E [ \ell _ { 2 } ^ { 2 } ( \widehat { P } _ { t } ^ { ( \mathrm { w } ) } , p ) ] \leqslant \frac { 3 } { \sum _ { i = 0 } ^ { t } n _ { i } \alpha _ { i } ^ { 2 } ( 1 + \chi ^ { 2 } \left( Q _ { 1 } \lVert \bar { Q } _ { \alpha _ { i } } \right) ) }\tag{7}
$$

whenever, for every $t \geqslant 1$

$$
\frac { 8 \log \left( k _ { W } \left( 1 + \chi ^ { 2 } \left( Q _ { 1 } \| \bar { Q } _ { \alpha _ { t } } \right) \right) \right) } { \operatorname* { m i n } _ { w } \bar { Q } _ { \alpha _ { t } } [ w ] } \leqslant n _ { t } \leqslant \frac { 1 } { \alpha _ { t } } \sum _ { i = 0 } ^ { t } n _ { i } \alpha _ { i } ^ { 2 } .\tag{8}
$$

We note that the hypothesis conditions (8) can be dispensed with in favor of more complicated results and proofs: the first inequality allows us to match a loose lower bound via concentration of the denominator, and the second is necessary only to ensure that the corresponding estimates lie in $\Delta _ { k }$ with probability 1 (as opposed lying outside $\Delta _ { k }$ with only exponentially small probability). From the above bounds, we see that when the memory term is small, the effective sample size for batch i in the watermarking setting is $n _ { i } \alpha _ { i } ^ { 2 } \left( 1 + \chi ^ { 2 } \left( Q _ { 1 } \lVert \bar { Q } _ { \alpha _ { i } } \right) \right)$ .

## B. Performance of the masked estimator

The above upper and lower bounds do not match when the memory term is not small. Making progress towards closing this gap, we show that probabilistically masking the output estimate reduces this to a Jensen gap.

1) The masking process: Let $B _ { t }$ be a (not necessarily i.i.d.) process such that $B _ { t } \sim \mathrm { B e r } \left( b _ { t } \right)$ for some sequence $b _ { t }$ . Let $\hat { P } _ { t } ^ { \mathrm { ( p r e ) } }$ be an unbiased pre-release estimate of $p .$ Sample $\dot { S } _ { t } \sim \hat { P } _ { t } ^ { \mathrm { ( p r e ) } }$ . The released estimate is given as

$$
\widehat { P } _ { t } [ x ] = \left\{ \begin{array} { l l } { \widehat { P } _ { t } ^ { ( \mathrm { p r e } ) } [ x ] } & { B _ { t } = 0 } \\ { \mathbb { 1 } \{ S _ { t } = x \} } & { B _ { t } = 1 } \end{array} . \right.
$$

We refer to this as masking to refer to the probabilistic replacement of the pre-release estimate with the k deterministic distributions $e _ { x } , x \in \mathcal { X }$ . When the estimate is indeed masked at time t, each synthetic sample in batch $t + 1$ is equal to $S _ { t }$

2) Performance: For conciseness of expression and proof, we state the upper bound in the unassisted setting, applying the masking process to a sequence of simple linear estimators. An equivalent expression with the corresponding $\chi ^ { 2 }$ divergence terms for each stage i can be shown by masking the sequence of estimators proposed in Theorem 2 instead.

Theorem 3: There exists a sequence of estimators $\widehat { p } _ { t }$ such that, for each $t \geqslant 0 ,$

$$
\operatorname* { s u p } _ { p } E _ { \bar { \mathbf { P } } _ { t , p } } \left[ \ell _ { 2 } ^ { 2 } \left( \widehat { P } _ { t } , p \right) \right] \leqslant 2 \sigma _ { t } ^ { 2 } \cdot ( 1 + c ) ,\tag{9}
$$

$$
\sigma _ { t } ^ { 2 } : = E \left[ \frac { 1 - 1 / k } { n _ { 0 } + \displaystyle \sum _ { i = 1 } ^ { t } n _ { i } \alpha _ { i } ^ { 2 } \left( 1 - B _ { i - 1 } \right) + n _ { i } \alpha _ { i } B _ { i - 1 } } \right]\tag{10}
$$

for any process $( B _ { i } ) _ { i \geqslant 0 }$ with marginal distributions $B _ { t } \sim \mathrm { B e r } \left( b _ { t } \right)$ , as long as $b _ { i } \leqslant c \sigma _ { i } ^ { 2 }$ for some constant $c > 0$ for every $i .$ The proofs of these theorems can be found in Appendix A

## IV. DISCUSSION

To get a sense of the impact(s) of watermarking and masking, we compare our results to the previously known unassisted and oracle-assisted minimax bounds. Via [1, Theorems 1 and 2 and Lemma 1], the effective sample sizes at stage t are $n _ { t } \alpha _ { t } ^ { 2 } + n _ { t } \alpha _ { t } g _ { t } ( \mathbb { 1 } / 4 k )$ for the lower bound and $n _ { t } \alpha _ { t } ^ { 2 }$ for the upper bound in the unassisted setting, and $n _ { t } \alpha _ { t }$ in both senses for the oracle-assisted setting. Watermarking therefore improves the effective samples size of stage t by a factor of 1 $- \chi ^ { 2 } \left( Q _ { 1 } \| \bar { Q } _ { \alpha _ { t } } \right)$

## A. Watermarking in recursive settings

Since perfect watermarking is exactly equivalent to the availability of oracle information $( Y _ { i } ^ { n _ { i } } ) _ { i \in [ 0 : t ] } .$ , we can compare effective sample sizes to find how fragile the benefit of watermarking is; a ‘good’ watermark would lead to an increase in effective sample size by a factor of $\Omega ( 1 / \alpha _ { i } )$ at each stage i. Given the statistics of the watermark, we have the following Proposition bounding the magnitude of this increment.

Proposition 1: For any $\alpha \in [ 0 , 1 ]$ , Let $\underline { { \epsilon } } = \mathrm { m i n } _ { w } \ : Q _ { 0 } [ w ] / Q _ { 1 } [ w ]$ Then $\chi ^ { 2 } \left( Q _ { 1 } \| \bar { Q } _ { \alpha } \right)$ is bounded from above as

$$
1 + \chi ^ { 2 } \left( Q _ { 1 } \| \bar { Q } _ { \alpha } \right) \leqslant \frac { 1 } { \alpha + \underline { { \epsilon } } - \alpha \underline { { \epsilon } } } \leqslant \frac { 1 } { \operatorname* { m a x } \{ \alpha , \underline { { \epsilon } } \} }\tag{11}
$$

Proof: The proof is direct. We have

$$
\begin{array} { l } { \displaystyle 1 + \chi ^ { 2 } \left( Q _ { 1 } \| \bar { Q } _ { \alpha } \right) = \displaystyle \sum _ { w \in \mathcal { W } } Q _ { 1 } [ w ] \frac { Q _ { 1 } [ w ] } { \alpha Q _ { 1 } [ w ] + ( 1 - \alpha ) Q _ { 0 } [ w ] } } \\ { \displaystyle = \sum _ { w \in \mathcal { W } } Q _ { 1 } [ w ] \frac { 1 } { \alpha + ( 1 - \alpha ) \frac { Q _ { 0 } [ w ] } { Q _ { 1 } [ w ] } } } \\ { \displaystyle \leqslant \frac { 1 } { \alpha + ( 1 - \alpha ) \underline { { \epsilon } } } \cdot \sum _ { w \in \mathcal { W } } Q _ { 1 } [ w ] . } \end{array}
$$

This bound provides a concise condition on $Q _ { 0 }$ and $Q _ { 1 }$ for watermarking to be useful: $\underline { { \epsilon } } _ { t }$ should decay at least as quickly as $\alpha _ { t }$ . Since $Q _ { 1 } [ w ] \leqslant 1$ , this implies that the watermarking detector should ‘misidentify’ synthetic samples as real samples (via at least one high confidence ‘real’ signal $w ^ { * } )$ with smaller and smaller probabilities as real samples become rarer and the estimates get closer to the ground truth distribution

We present the exact performance in two well-known cases, where the watermarking detection statistics describe the output of the sample provenances $Y$ when passed through two well-known channels: the BEC and the BSC.

Claim $I ~ ( B E C ) .$ : Let the conditional distributions $Q _ { 0 }$ and $Q _ { 1 }$ denote the output distributions of a binary erasure channel (BEC) with error probability ϵ given inputs 0 and 1 respectively. $\chi ^ { 2 } ( Q _ { 1 } \| \bar { Q } _ { \alpha } )$ is then given as $1 - \epsilon / \alpha$

This is exactly the behavior we expect, since the detection outputs for synthetic samples are either 0 or ‘error’. This might be difficult to implement in practice: watermarks still have relatively high false negative rates (see [2]–[4]). A more realistic watermarking scheme is described by a BSC:

Claim $2 \ ( B S C ) .$ : For $Q _ { 0 }$ and $Q _ { 1 }$ describing the conditional output distributions of a binary symmetric channel (BSC) with crossover probability $\epsilon < 1 / 2$ , the corresponding $\chi ^ { 2 } ( Q _ { 1 } \| \bar { Q } _ { \alpha } )$ is given as

$$
( 1 - 2 \epsilon ) \cdot \left( \frac { 1 } { \frac { \alpha } { 1 - \alpha } + \frac { \epsilon } { 1 - \epsilon } } - \frac { 1 } { \frac { \alpha } { 1 - \alpha } + \frac { 1 - \epsilon } { \epsilon } } \right) \leqslant \frac { 1 } { \operatorname* { m a x } \{ \alpha , \epsilon \} } .
$$

Thus, whenever the probability of misidentification is not very small (here, when ϵ is not close to α), watermarking only increases the effective sample size very marginally.

## B. Masking estimators

In a large selection of regimes, the term $h _ { t } ( 1 / 4 k )$ (and similarly, $h _ { t } ^ { ( \mathrm { w } ) } ( ^ { 1 / 4 k } ) )$ is dominated by the other term in the denominator [1, Propositions 1 and 2]. However, we see that there is a gap between the upper and lower bounds whenever the memory term $h _ { t } ( 1 / 4 k )$ dominates the effective sample size.

1) The linear estimation gap: Using Chebyshev’s inequality, we can upper bound the memory term $g _ { i } ( \rho ) \leqslant 1 6 k ^ { 2 } \sigma _ { i - } ^ { 2 }$ and consequently, upper bound the total effective sample size by $\begin{array} { r } { n _ { 0 } + \sum _ { i = 1 } ^ { t } 1 6 k ^ { 2 } n _ { i } \alpha _ { i } \sigma _ { i - 1 } ^ { 2 } } \end{array}$

Now, any sequence of linear estimators of $\dot { \mathbf { \rho } } _ { p }$ will lead to subgaussian estimates $\widehat { P } _ { i } ^ { ( \mathrm { w } ) }$ (since the empirical distributions of each batch are subgaussian). Additionally, the bias of any such estimate has to be $\mathcal { O } ( \mathrm { v a r } ( \widehat { P } _ { i } ^ { ( \mathrm { w } ) } ) )$ , which is $o ( 1 )$ , i.e. the mean of these estimates is at an $o ( 1 )$ distance from $p .$ Consequently, the corresponding memory term g<sub>i</sub>—the probability of deviating from $p$ by a constant distance $1 / 4 k { \mathrm { - } } { \mathrm { w i l l } }$ be exponentially small due to the concentration of these subgaussians around their mean. If $\alpha _ { i }$ (asymptotically) decreases quicker than $\sigma _ { i - 1 } ^ { 2 }$ , using linear estimators cannot lead to an effective sample size larger than $\textstyle \sum _ { i = 0 } ^ { t } n _ { i } \alpha _ { i } ^ { 2 }$ , while our Chebyshev upper bound is of the order $\begin{array} { r } { n _ { 0 } + \sum _ { i = 1 } ^ { t } n _ { i } \alpha _ { i } \sigma _ { i - 1 } ^ { 2 } } \end{array}$ . We show that the non-linearity introduced by the masking procedure narrows this to a Jensen gap.

2) Partially bridging the gap via masking: Notice that when $b _ { i }$ is indeed equal to $\sigma _ { i } ^ { 2 }$ (or equivalently within constant factors), the expected value of the denominator in the RHS of (10) is

$$
E \left[ n _ { 0 } + \sum _ { i = 1 } ^ { t } n _ { i } \alpha _ { i } ^ { 2 } \left( 1 - B _ { i - 1 } \right) + n _ { i } \alpha _ { i } B _ { i - 1 } \right] = \sum _ { i = 0 } ^ { t } n _ { i } \alpha _ { i } ^ { 2 } + \sum _ { i = 1 } ^ { t } n _ { i } \alpha _ { i } \sigma _ { i } ^ { 2 } .\tag{12}
$$

While on the wrong side of Jensen’s inequality, the loss of the sequence of masking estimators converges to the corresponding lower bound in [1, Theorem 1] in the regimes where this denominator concentrates around its expected value, thereby closing the gap. We conjecture that a more careful analysis which accounts for sample path averages over each of the events in the memory term can further close this gap by raising the lower bound for the regimes in which concentration does not occur.

3) Masking with distributions $o f$ support larger than 1: For some alphabet $x ,$ let the released estimate $\widehat { P } _ { i - 1 } [ x ] \ = \ 0$ Conditioned on x getting masked in this way, the marginal probability of drawing x at stage i is given as $\alpha _ { i } p [ x ]$ . The ‘corrected’ empirical estimate $\scriptstyle { \frac { 1 } { \alpha _ { i } } } { \widehat { p } } _ { \mathrm { e m p } } [ x ]$ has variance exactly

$$
\frac { 1 } { n _ { i } ^ { 2 } \alpha _ { i } ^ { 2 } } \cdot n _ { i } \cdot \alpha _ { i } p [ x ] ( 1 - \alpha _ { i } p [ x ] ) = \frac { p [ x ] ( 1 - \alpha _ { i } p [ x ] ) } { n _ { i } \alpha _ { i } } ,\tag{13}
$$

which directly raises the effective sample size for the estimate of $p [ x ]$ from $n _ { i } \alpha _ { i } ^ { 2 }$ to $n _ { i } \alpha _ { i }$ . Therefore, masking each estimate with distributions that have support size $1 < \underline { { k } } < k$ would also lead to a similar upper bound, provided that masking is done often enough for each alphabet. We persist with masking using distributions of support size 1 for simplicity of proof.

4) Masking as a watermark: Similar to implementations in text generation, masking itself acts as a watermark: given $\widehat { P } _ { t - 1 } [ x ] = 0$ , all samples in stage t with value x can be directly classified as real with probability 1. It does, however, distort the synthetic distribution, and therefore offers a more limited advantage compared to ‘invisible’ watermarks. Practicality is further limited by the fact that our modeling assumption that only samples from $\hat { P } _ { t - 1 }$ are mixed in at stage t might no longer be valid in realistic settings.

## REFERENCES

[1] M. Kanabar and M. Gastpar, “Model Non-Collapse: Minimax Bounds for Recursive Discrete Distribution Estimation,” IEEE Transactions on Information Theory, vol. 72, pp. 1817–1830, Mar. 2026.

[2] Q. Pang, S. Hu, W. Zheng, and V. Smith, “No Free Lunch in LLM Watermarking: Trade-offs in Watermarking Design Choices,” Advances in Neural Information Processing Systems, vol. 37, pp. 138756–138788, Dec. 2024.

[3] H. Zhang, B. L. Edelman, D. Francati, D. Venturi, G. Ateniese, and B. Barak, “Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models,” Apr. 2024.

[4] S. Gowal, R. Bunel, F. Stimberg, D. Stutz, G. Ortiz-Jimenez, C. Kouridi, M. Vecerik, J. Hayes, S.-A. Rebuffi, P. Bernard, C. Gamble, M. Z. Horvath,´ F. Kaczmarczyck, A. Kaskasoli, A. Petrov, I. Shumailov, M. Thotakuri, O. Wiles, J. Yung, Z. Ahmed, V. Martin, S. Rosen, C. Savcak, A. Senoner,ˇ N. Vyas, and P. Kohli, “SynthID-Image: Image watermarking at internet scale,” Oct. 2025. arXiv:2510.09263.

[5] I. Shumailov, Z. Shumaylov, Y. Zhao, N. Papernot, R. Anderson, and Y. Gal, “AI models collapse when trained on recursively generated data,” Nature, vol. 631, July 2024. arXiv:2305.17493.

[6] S. Alemohammad, J. Casco-Rodriguez, L. Luzi, A. I. Humayun, H. Babaei, D. LeJeune, A. Siahkoohi, and R. Baraniuk, “Self-consuming generative models go MAD,” in The Twelfth International Conference on Learning Representations, 2024. arXiv:2307.01850.

[7] J. Kazdan, R. Schaeffer, A. Dey, M. Gerstgrasser, R. Rafailov, D. L. Donoho, and S. Koyejo, “Collapse or Thrive? Perils and Promises of Synthetic Data in a Self-Generating World,” Feb. 2025. arXiv:2410.16713.

[8] L. Shi, M. Wu, H. Zhang, Z. Zhang, M. Tao, and Q. Qu, “A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective,” Advances in Neural Information Processing Systems, vol. 38, pp. 40658–40691, Apr. 2026.

[9] M. E. A. Seddik, S.-W. Chen, S. Hayou, P. Youssef, and M. A. Debbah, “How bad is training on synthetic data? a statistical analysis of language model collapse,” in First Conference on Language Modeling, 2024. arxiv:2404.05090.

[10] A. T. Suresh, A. Thangaraj, and A. N. K. Khandavally, “Rate of model collapse in recursive training,” in The 28th International Conference on Artificial Intelligence and Statistics, 2025. arXiv:2412.17646.

[11] Q. Bertrand, J. Bose, A. Duplessis, M. Jiralerspong, and G. Gidel, “On the stability of iterative retraining of generative models on their own data,” in The Twelfth International Conference on Learning Representations, 2024. arXiv:2310.00429.

[12] M. Marchi, S. Soatto, P. Chaudhari, and P. Tabuada, “Heat Death of Generative Models in Closed-Loop Learning,” Aug. 2024. arXiv:2404.02325.

[13] S. Xu, H. He, and G. Cheng, “A Probabilistic Perspective on Model Collapse,” May 2025. arXiv:2505.13947.

[14] D. Barzilai and O. Shamir, “When Models Don’t Collapse: On the Consistency of Iterative MLE,” Advances in Neural Information Processing Systems, vol. 38, pp. 76813–76854, Apr. 2026.

[15] S. Fu, S. Zhang, Y. Wang, X. Tian, and D. Tao, “Towards theoretical understandings of self-consuming generative models,” in Forty-first International Conference on Machine Learning, 2024. arXiv:2402.11778.

[16] E. Dohmatob, Y. Feng, and J. Kempe, “Model collapse demystified: The case of regression,” in The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. arXiv:2402.07712.

[17] M. Gerstgrasser, R. Schaeffer, A. Dey, R. Rafailov, T. Korbak, H. Sleight, R. Agrawal, J. Hughes, D. B. Pai, A. Gromov, D. Roberts, D. Yang, D. L. Donoho, and S. Koyejo, “Is model collapse inevitable? breaking the curse of recursion by accumulating real and synthetic data,” in First Conference on Language Modeling, 2024. arXiv:2404.01413.

[18] Y. Feng, E. Dohmatob, P. Yang, F. Charton, and J. Kempe, “Beyond model collapse: Scaling up with synthesized data requires reinforcement,” in ICML 2024 Workshop on Theoretical Foundations of Foundation Models, 2024. arXiv:2406.07515.

[19] E. Dohmatob, Y. Feng, A. Subramonian, and J. Kempe, “Strong model collapse,” in The Thirteenth International Conference on Learning Representations, 2025. arXiv:2410.04840.

[20] H. A. Vu, G. Reeves, and E. Wenger, “What happens when generative AI models train recursively on each others’ outputs?,” Oct. 2025. arXiv:2505.21677.

[21] A. Garg, S. Bhattacharya, and P. Sur, “Preventing Model Collapse Under Overparametrization: Optimal Mixing Ratios for Interpolation Learning and Ridge Regression,” Feb. 2026. arXiv:2509.22341.

[22] N. B. Khelifa, R. E. Turner, and R. Venkataramanan, “Error Propagation and Model Collapse in Diffusion Models: A Theoretical Study,” Feb. 2026. arXiv:2602.16601.

[23] S. Fu, Y. Wang, Y. Chen, X. Tian, and D. Tao, “A Theoretical Perspective: How to Prevent Model Collapse in Self-consuming Training Loops,” Feb 2025. arXiv:2502.18865.

[24] J. Kazdan, R. Schaeffer, A. Dey, M. Gerstgrasser, R. Rafailov, D. L. Donoho, and S. Koyejo, “Collapse or thrive? perils and promises of synthetic data in a self-generating world,” 2025. arXiv:2410.16713.

[25] H. He, S. Xu, and G. Cheng, “Golden Ratio Weighting Prevents Model Collapse,” Mar. 2025. arXiv:2502.18049.

[26] W. Ji, W. Yuan, E. Getzen, K. Cho, M. I. Jordan, S. Mei, J. Weston, W. J. Su, J. Xu, and L. Zhang, “An Overview of Large Language Models for Statisticians,” The American Statistician, vol. 0, pp. 1–106, Apr. 2026. eprint: https://doi.org/10.1080/00031305.2026.2657480.

[27] G. Racca, M. Valko, and A. Sanyal, “Language Generation with Replay: A Learning-Theoretic View of Model Collapse,” Mar. 2026. arXiv:2603.11784

[28] E. Dohmatob, M. Pezeshki, and R. Askari-Hemmat, “Why Less is More (Sometimes): A Theory of Data Curation,” Nov. 2025. arXiv:2511.03492.

[29] K. Amin, S. Babakniya, A. Bie, W. Kong, U. Syed, and S. Vassilvitskii, “Escaping collapse: The strength of weak data for large language model training,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. arXiv.2502.08924.

[30] K. Amin, A. Bie, W. Kong, U. Syed, and S. Vassilvitskii, “Learning from synthetic data: Limitations of ERM,” in 37th International Conference on Algorithmic Learning Theory, 2026.

[31] F. Mansouri and S. Ben-David, “Learning from positive and unlabeled examples -Finite size sample bounds,” Oct. 2025.

[32] A. Jain, A. Montanari, and E. Sasoglu, “Scaling laws for learning with real and surrogate data,” Advances in Neural Information Processing Systems vol. 37, Dec. 2024.

[33] L. Paninski, “A Coincidence-Based Test for Uniformity Given Very Sparsely Sampled Discrete Data,” IEEE Transactions on Information Theory, vol. 54, Oct. 2008.

[34] B. Yu, “Assouad, Fano, and Le Cam,” in Festschrift for Lucien Le Cam: Research Papers in Probability and Statistics (D. Pollard, E. Torgersen, and G. L. Yang, eds.), New York, NY: Springer, 1997.

## APPENDIX A

## PROOF OF THE LOWER BOUND

## A. The lower bound for watermarking

As in [1], we use the Paninski construction [33] along with Assouad’s method (see e.g. [34]) to find lower bounds in the various settings.

Lemma 4 (Restated Lemma 7 from $I I J .$ Let $\boldsymbol { Z } \sim \bar { \mathbf { P } } _ { t , p }$ . Consider the set of vectors $\mathcal { V } = \{ + 1 , - 1 \} ^ { k / 2 }$ . With each vector $v \in \mathcal { V } ,$ associate the distribution

$$
p _ { v } : = p _ { u } + \frac { \delta } { k } \left[ \begin{array} { c } { { v } } \\ { { - v } } \end{array} \right]\tag{14}
$$

where p is the uniform distribution. Let $p _ { u }$ $\mathcal { P } _ { k } : = \{ p _ { v } : v \in \mathcal { V } \}$ . The $\ell _ { 2 } ^ { 2 }$ minimax estimation risk of an estimator $\widehat { P } _ { t } = \widehat { p } _ { t } ( Z )$ as described in Definition 1 is lower bounded as

$$
r _ { t , k } \geqslant \frac { \delta ^ { 2 } } { 4 k } \left( 1 - \sqrt \frac { 2 } { k } \sum _ { j = 1 } ^ { k / 2 } \operatorname* { m a x } _ { v \in \mathcal { V } _ { \vdots } } D \Big ( \bar { \mathbf { P } } _ { t , p _ { v } } \Big \Vert \bar { \mathbf { P } } _ { t , p _ { v - 2 e _ { j } } } \Big ) \right)\tag{15}
$$

We are now ready to prove the lower bound.

Proof of Theorem 1: The proof proceeds by showing an upper bound on $D \left( \bar { \mathbf { P } } _ { t , p _ { v } } ^ { ( \mathrm { w } ) } \bigg \| \bar { \mathbf { P } } _ { t , p _ { v - 2 e _ { j } } } ^ { ( \mathrm { w } ) } \right)$ , with $p _ { v }$ as defined in (14). Define the conditional distributions (with some abuse of notation)

$$
\begin{array} { r l } & { \widetilde { \mathbf { P } } _ { t , p } ^ { ( w ) } ( ( x _ { t } , w _ { t } ) ^ { n _ { t } } | ( x _ { i } , w _ { i } ) _ { i \in [ 0 , t - 1 ] } ^ { n _ { t } } ) : = \overline { { \mathbf { P } } } _ { t , p } ^ { ( w ) } \{ ( X _ { t } , W _ { t } ) ^ { n _ { t } } = ( x _ { t } , w _ { t } ) ^ { n _ { t } } \big | ( X _ { i } , W _ { i } ) _ { i \in [ 0 , t - 1 ] } ^ { n _ { t } } = ( x _ { i } , w _ { i } ) _ { i \in [ 0 , t - 1 ] } ^ { n _ { t } } \} , } \\ & { \qquad \widetilde { \mathbf { P } } _ { t , p } ^ { ( w ) } ( x _ { t } , w _ { t } | ( x _ { i } , w _ { i } ) _ { i \in [ 0 , t - 1 ] } ^ { n _ { t } } ) : = \overline { { \mathbf { P } } } _ { t , p } ^ { ( w ) } \{ ( X _ { t } [ j ] , W _ { t } [ j ] ) = ( x _ { t } , w _ { t } ) \Big | ( X _ { i } , W _ { i } ) _ { i \in [ 0 , t - 1 ] } ^ { n _ { t } } = ( x _ { i } , w _ { i } ) _ { i \in [ 0 , t - 1 ] } ^ { n _ { t } } \} \quad j \in [ 1 : n _ { t } ] , } \\ &  \mathrm { a n d ~ } \widetilde { \mathbf { P } } _ { t , p , w _ { t } } ^ { ( w ) } ( x _ { t } \Big | ( x _ { i } , w _ { i } ) _ { i \in [ 0 , t - 1 ] } ^ { n _ { t } } ) : = \overline { { \mathbf { P } } } _ { t , p } ^ { ( w ) } \{ X _ { t } [ j ] = x _ { t } \Big | ( X _ { i } , W _ { i } ) _ { i \in [ 0 , t - 1 ] } ^ { n _ { t } } = ( x _ { i } , w _ { i } ) _ { i \in [ 0 , t - 1 ] } ^ { n _ { t } } , W  \end{array}
$$

Then

$$
\begin{array} { r l } { D ( \widetilde { \mathbf { P } } _ { t , p _ { v } } ^ { ( \mathbf { w } ) } \middle | \overline { { \mathbf { P } } } _ { t , p _ { v } - 2 _ { s _ { j } } } ^ { ( \mathbf { w } ) } ) \stackrel { ( a ) } { = } \underset { i = 0 } { \overset { t } { \sum } } D ( \widetilde { \mathbf { P } } _ { i , p _ { v } } ^ { ( \mathbf { w } ) } \middle | \widetilde { \mathbf { P } } _ { i , p _ { v - 2 _ { s } } } ^ { ( \mathbf { w } ) } \middle | \overline { { \mathbf { P } } } _ { i - 1 , p _ { v } } ^ { ( \mathbf { w } ) } ) } & { } \\ & { = \underset { i = 0 } { \overset { t } { \sum } } n _ { i } E _ { \widetilde { \mathbf { P } } _ { i - 1 , p _ { v } } ^ { ( \mathbf { w } ) } } [ D ( \widetilde { P } _ { i , p _ { v } } ^ { ( \mathbf { w } ) } \middle | \widetilde { P } _ { i , p _ { v - 2 _ { s } } } ^ { ( \mathbf { w } ) } ) ] } \\ & { = \underset { i = 0 } { \overset { t } { \sum } } n _ { i } E _ { \widetilde { \mathbf { P } } _ { i - 1 , p _ { v } } ^ { ( \mathbf { w } ) } } [ E _ { W \sim \widetilde { Q } _ { \alpha _ { t } } } [ D ( \widetilde { P } _ { i , p _ { v } , W } ^ { ( \mathbf { w } ) } \middle | \widetilde { P } _ { i , p _ { v - 2 _ { s } } } ^ { ( \mathbf { w } ) } , W ) ] ] } \\ &  \overset { ( b ) } { = } \underset { i = 0 } { \overset { t } { \sum } } E _ { W \sim \widetilde { Q } _ { \alpha _ { t } } } [ n _ { i } E _ { \widetilde { \mathbf { P } } _ { i - 1 , p _ { v } } ^ { ( \mathbf { w } ) } } [ D ( \widetilde { P } _ { i , p _ { v } , W } ^ { ( \mathbf { w } ) } \middle | \widetilde { P } _  i , p _  v - 2 _ \end{array}\tag{16}
$$

where paq follows from the chain rule of KL Divergences and pbq follows because of the independence of the distribution of the watermark and the samples themselves. Observe that

$$
\widetilde { P } _ { t , p , w } ^ { ( \mathrm { w } ) } = \widetilde { \alpha } _ { i , w } p + ( 1 - \widetilde { \alpha } _ { i , w } ) \widehat { P } _ { t - 1 } ^ { ( \mathrm { w } ) }\tag{17}
$$

where

$$
\widetilde \alpha _ { i , w } ^ { ( \mathrm { w } ) } : = \frac { \alpha _ { i } Q _ { 1 } [ w ] } { \bar { Q } _ { \alpha _ { i } } [ w ] }
$$

$$
i \in [ 0 : t ] .\tag{18}
$$

Thus, conditioned on each watermark detection output $W _ { t } [ j ] = w _ { t }$ , the distributions of the samples $X _ { t } [ j ]$ are equal to the marginal distributions of unwatermarked samples with equivalent mixing factors ${ \widetilde \alpha } _ { i , w _ { t } } ^ { ( \mathrm { w } ) }$ . We can therefore directly use results from the unwatermarked setting to bound (16). From [1, Equation 17], for each $W = w$

$$
n _ { i } E _ { \widetilde { \mathbf { P } } _ { i - 1 , p _ { \mathrm { v } } } ^ { ( \infty ) } } \left[ D \left( \widetilde { P } _ { i , p _ { \mathrm { v } } , W } ^ { ( \infty ) } \Big \| \widetilde { P } _ { i , p _ { \mathrm { v } } - 2 e _ { j } , W } ^ { ( \infty ) } \right) \right] \leqslant \frac { 8 \delta ^ { 2 } } { k \left( \frac { 9 } { 1 6 } - \delta ^ { 2 } \right) } \left( n _ { i } \left( \widetilde { \alpha } _ { i , w } ^ { ( \mathrm { w } ) } \right) ^ { 2 } + n _ { i } \widetilde { \alpha } _ { i , w } ^ { ( \mathrm { w } ) } g _ { i } ( { ^ { 1 / 4 k } } ) \right) .\tag{19}
$$

Consequently,

$$
D \left( \mathbf { \widetilde { P } } _ { t , p _ { v } } ^ { ( \infty ) } \Big \| \mathbf { \widetilde { P } } _ { t , p _ { v } - 2 \epsilon _ { j } } ^ { ( \infty ) } \right) \leqslant \frac { 8 \delta ^ { 2 } } { k \left( \frac { 9 } { 1 6 } - \delta ^ { 2 } \right) } E _ { W \sim \bar { Q } _ { \alpha _ { t } } } \left[ \sum _ { i = 0 } ^ { t } n _ { i } \Big ( \widetilde { \alpha } _ { i , W } ^ { ( \infty ) } \Big ) ^ { 2 } + n _ { i } \widetilde { \alpha } _ { i , W } ^ { ( \infty ) } g _ { i } ( 1 / 4 k ) \right] .
$$

Since

$$
E _ { W \sim \bar { Q } _ { \alpha _ { t } } } \left[ \left( \widetilde { \alpha } _ { i , W } ^ { ( \mathrm { w } ) } \right) ^ { 2 } \right] = E _ { W \sim \bar { Q } _ { \alpha _ { t } } } \left[ \alpha _ { t } ^ { 2 } \left( \frac { Q _ { 1 } [ w ] } { \bar { Q } _ { \alpha _ { t } } [ x ] } \right) ^ { 2 } \right] = \alpha _ { i } ^ { 2 } \left( 1 + \chi ^ { 2 } \left( Q _ { 1 } \| \bar { Q } _ { \alpha _ { i } } \right) \right)\tag{20}
$$

$$
\mathrm { a n d ~ } E _ { W \sim \bar { Q } _ { \alpha _ { t } } } \left[ \widetilde { \alpha } _ { i , W } ^ { ( \mathrm { w } ) } \right] = E _ { W \sim \bar { Q } _ { \alpha _ { t } } } \left[ \alpha _ { t } \frac { Q _ { 1 } [ w ] } { \bar { Q } _ { \alpha _ { t } } [ x ] } \right] = \alpha _ { i } ,\tag{21}
$$

choosing

$$
\delta ^ { 2 } = \frac { k } { 6 4 \sum _ { i = 0 } ^ { t } \left( n _ { i } \alpha _ { i } ^ { 2 } \left( 1 + \chi ^ { 2 } \left( Q _ { 1 } \Vert \bar { Q } _ { \alpha _ { i } } \right) \right) + n _ { i } \alpha _ { i } g _ { i } ( 1 / 4 k ) \right) }\tag{22}
$$

gives us

$$
D \left( \bar { \mathbf { P } } _ { t , p _ { v } } ^ { ( \mathrm { w } ) } \bigg \| \bar { \mathbf { P } } _ { t , p _ { v - 2 e _ { j } } } ^ { ( \mathrm { w } ) } \right) \leqslant \frac { 1 } { 4 }
$$

whenever $\begin{array} { r } { \sum _ { i = 0 } ^ { t } n _ { i } \alpha _ { i } ^ { 2 } \left( 1 + \chi ^ { 2 } \left( Q _ { 1 } \lVert \bar { Q } _ { \alpha _ { i } } \right) \right) + h _ { t } ^ { ( \mathrm { w } ) } ( ^ { 1 } / 4 k ) \geqslant k / 4 } \end{array}$ . Invoking Lemma 4 then leads to a lower bound of

$$
r _ { t , k } ^ { ( \mathrm { w } ) } \geqslant \frac { 1 } { 5 1 2 \sum _ { i = 0 } ^ { t } n _ { i } \alpha _ { i } ^ { 2 } ( 1 + \chi ^ { 2 } ( Q _ { 1 } \mathopen { } \mathclose \bgroup \| \bar { Q } _ { \alpha _ { i } } ) ) + h _ { t } \mathopen { } \mathclose \bgroup ( ^ { 1 / 4 k } \aftergroup \egroup ) } ,\tag{23}
$$

completing the proof.

## APPENDIX B PROOFS OF THE UPPER BOUNDS

## A. Useful Lemmas

We also have the following Lemma showing the performance of a linear estimator in the presence of recursive sampling. Lemma 5 (Restated Lemma 9 from $I I J .$ Let $\hat { P } _ { 0 }$ be an unbiased estimate of $\mathbf { \Phi } _ { p }$ with a variance $\eta ^ { 2 } [ x ] \cdot p [ x ] ( 1 - p [ x ] )$ for each component $x \in \mathcal { X }$ . Let samples $Z ^ { n } \sim ( \alpha p + ( 1 - \alpha ) \widehat { P } _ { 0 } ) ^ { n }$ . Denote the empirical distribution of samples $X ^ { n }$ as ${ \widehat { p } } _ { \mathrm { e m p } } ( X ^ { n } )$ . Then the intermediate estimate

$$
\widehat { P } _ { \mathrm { 1 , i n t } } = \widehat { p } _ { \mathrm { 1 , i n t } } ( X _ { 1 } ^ { n _ { 1 } } ) : = \frac { 1 } { \alpha } \left( \widehat { p } _ { \mathrm { e m p } } ( X _ { 1 } ^ { n _ { 1 } } ) - ( 1 - \alpha ) \widehat { P } _ { 0 } \right)\tag{24}
$$

has expectation $p$ and is component-wise uncorrelated with $ { \hat { P } } _ { 0 }$ . Additionally, each component $x \in \mathcal { X }$ has variance

$$
E \left[ \left( \hat { P } _ { 1 , \mathrm { i n t } } [ x ] - p [ x ] \right) ^ { 2 } \right] = \frac { p [ x ] ( 1 - p [ x ] ) } { n _ { 1 } \alpha ^ { 2 } } \left( 1 - ( 1 - \alpha ) ^ { 2 } \eta ^ { 2 } [ x ] \right) .\tag{25}
$$

Finally, we state the following elementary lemma that will be useful in combining estimates when upper bounds on thei variances are known.

Lemma $6 \mathrm { : }$ Let $\widehat { \theta _ { 1 } }$ and $\widehat { \theta } _ { 2 }$ be two unbiased and uncorrelated estimates of a parameter $\theta \in \Theta$ , and let their variances be bounded above by $\sigma _ { 1 } ^ { 2 }$ and $\sigma _ { 2 } ^ { 2 }$ . We can then construct a linear combination $\bar { \theta }$ of $\dot { \theta _ { 1 } }$ and $\widehat { \theta } _ { 2 }$ that is unbiased and with variance $\overline { { \sigma } } ^ { 2 }$ bounded above as $\begin{array} { r } { \bar { \sigma } ^ { 2 } \leqslant 1 / \Bigl ( \frac { 1 } { \sigma _ { 1 } ^ { 2 } } + \frac { 1 } { \sigma _ { 2 } ^ { 2 } } \Bigr ) } \end{array}$

## B. Proof of the watermarking upper bound

Proof of Theorem 2: The proof proceeds inductively. Assume that at stage t ´ 1, each component of $\hat { P } _ { t - 1 }$ is unbiased and has variance upper bounded as

$$
\operatorname { v a r } \Big ( \widehat { P } _ { t - 1 } ^ { ( \mathrm { w } ) } [ x ] \Big ) \leqslant \bar { \sigma } _ { t - 1 } ^ { 2 } [ x ] = \frac { 3 \cdot p [ x ] ( 1 - p [ x ] ) } { \sum _ { i = 0 } ^ { t } n _ { i } \alpha _ { i } ^ { 2 } ( 1 + \chi ^ { 2 } \left( Q _ { 1 } \| \bar { Q } _ { \alpha _ { i } } \right) ) } .
$$

It is easy to verify that the empirical estimator satisfies these conditions for $t = 0$

Let the number of samples in batch t watermarked as w be $N _ { t } [ w ]$ , and let the corresponding samples be $X _ { t . w } ^ { N _ { t } [ w ] }$ . Recall from (17) that given $\widehat { P } _ { t - 1 } ^ { ( \mathrm { w } ) }$ and $W _ { t } = w , X _ { t , w } ^ { N _ { t } [ w ] }$ is a conditionally i.i.d. vector with marginal distributions $\tilde { \alpha } _ { t , w } ^ { ( \mathrm { w } ) } p + ( 1 - \tilde { \alpha } _ { t , w } ^ { ( \mathrm { w } ) } ) \hat { P } _ { t - 1 } ^ { ( \mathrm { w } ) }$ Therefore, invoking Lemma 5, for each $x \in \mathcal X$ , the conditional variance of the $x ^ { \mathrm { t h } }$ component of intermediate estimates $\widehat { P } _ { t , w , \mathrm { i n t } } : = \widehat { p } _ { t , \mathrm { i n t } } \left( \bar { X } _ { t , w } ^ { N _ { t } [ w ] } \right)$ given $W _ { t } ^ { n _ { t } }$ and $\widehat { P } _ { t - 1 } ^ { ( \mathrm { w } ) }$ is bounded above by

$$
\mathrm { v a r } \left( \widehat { P } _ { t , w , \mathrm { i n t } } [ x ] \Big | W _ { t } ^ { n _ { t } } , \widehat { P } _ { t - 1 } ^ { ( \mathrm { w } ) } \right) \leqslant \frac { p [ x ] ( 1 - p [ x ] ) } { N _ { t } [ w ] \left( \widetilde { \alpha } _ { i , w } ^ { ( \mathrm { w } ) } \right) ^ { 2 } } .
$$

For each $x \in \mathcal { X }$ , all estimates $\widehat { P } _ { t , w , \mathrm { i n t } } [ x ] , w \in \mathcal { W }$ are also unbiased and conditionally independent given $W _ { t } ^ { n _ { t } }$ and $\widehat { P } _ { t - 1 } ^ { ( \mathrm { w } ) }$ . The combined intermediate estimate $\begin{array} { r } { \widehat { P } _ { t , \mathrm { i n t } } = \sum _ { w \in \mathcal { W } } \frac { N _ { t } \left[ w \right] \left( \widetilde { \alpha } _ { i , w } ^ { ( \mathrm { w } ) } \right) ^ { 2 } } { \sum _ { w ^ { \prime } \in \mathcal { W } } N _ { t } \left[ w ^ { \prime } \right] \left( \widetilde { \alpha } _ { i , w ^ { \prime } } ^ { ( \mathrm { w } ) } \right) ^ { 2 } } \cdot \widehat { P } _ { t , w , \mathrm { i n t } } } \end{array}$ therefore has a conditional variance bounded above as

$$
\operatorname { v a r } \Big ( \widehat { P } _ { t , \mathrm { i n t } } [ x ] \Big | W _ { t } ^ { n _ { t } } , \widehat { P } _ { t - 1 } ^ { ( \mathrm { w } ) } \Big ) \leqslant \frac { p [ x ] ( 1 - p [ x ] ) } { \sum _ { w \in \mathcal { W } } N _ { t } [ w ] \left( \widetilde { \alpha } _ { i , w } ^ { ( \mathrm { w } ) } \right) ^ { 2 } } ,\tag{26}
$$

(via Lemma 6) where the denominator has an expected value $n _ { t } \alpha _ { t } ^ { 2 } \left( 1 + \chi ^ { 2 } \left( Q _ { 1 } \lVert \bar { Q } _ { \alpha _ { t } } \right) \right)$ ˘. Now,

$$
\begin{array} { r l } { \bar { \mathbf { P } } _ { t , p } ^ { \left( \mathrm { w } \right) } \left\{ \displaystyle \sum _ { w \in \mathcal { W } } N _ { t } [ w ] \left( \widetilde { \alpha } _ { i , w } ^ { \left( \mathrm { w } \right) } \right) ^ { 2 } \leqslant \frac { 1 } { 2 } n _ { t } \alpha _ { t } ^ { 2 } \left( 1 + \chi ^ { 2 } \left( Q _ { 1 } \| \bar { Q } _ { \alpha _ { t } } \right) \right) \right\} \stackrel { \scriptscriptstyle ( a ) } { \leqslant } \bar { \mathbf { P } } _ { t , p } ^ { \left( \mathrm { w } \right) } \left\{ \bigcup _ { w \in \mathcal { W } } \left\{ N _ { t } [ w ] \leqslant \frac { n _ { t } \bar { Q } _ { \alpha _ { t } } [ w ] } { 2 } \right\} \right\} } & \\ { \stackrel { \scriptscriptstyle ( b ) } { \leqslant } \displaystyle \sum _ { w \in \mathcal { W } } \bar { \mathbf { P } } _ { t , p } ^ { \left( \mathrm { w } \right) } \left\{ \nabla _ { t } [ w ] \leqslant \frac { n _ { t } \bar { Q } _ { \alpha _ { t } } [ w ] } { 2 } \right\} } & \\ { \leqslant k _ { W } \displaystyle \operatorname* { m a x } _ { w \in \mathcal { W } } \bar { \mathbf { P } } _ { t , p } ^ { \left( \mathrm { w } \right) } \left\{ N _ { t } [ w ] \leqslant \frac { n _ { t } \bar { Q } _ { \alpha _ { t } } [ w ] } { 2 } \right\} } & \\ { \stackrel { \scriptscriptstyle ( e ) } { \leqslant } k _ { W } \exp \left( - n _ { t } \frac { \operatorname* { m i n } _ { \bar { Q } _ { \alpha _ { t } } [ w ] } } { 8 } \right) } & \\ { \stackrel { \scriptscriptstyle ( d ) } { \leqslant } \frac { 1 } { 1 + \chi ^ { 2 } \left( Q _ { 1 } \| \bar { Q } _ { \alpha _ { t } } \right) } , } & \end{array}\tag{27}
$$

where paq follows from $W _ { t } [ i ] \sim \bar { Q } _ { \alpha _ { t } }$ and $\tilde { \alpha } _ { i , w } ^ { \mathrm { ( w ) } } = Q _ { 1 } [ w ] / \bar { Q } _ { \alpha _ { t } } [ w ]$ , pbq follows from the union bound, pcq follows from Hoeffding’s inequality, and pdq follows from the lower bound assumption on $n _ { t }$ in (8). Define the pre-release estimate as

$$
\begin{array} { r } { \widehat { P } _ { t , \mathrm { p r e } } : = \left\{ \begin{array} { l l } { \widehat { P } _ { t , \mathrm { i n t } } } & { \mathrm { i f ~ } \sum _ { w \in \mathcal { W } } N _ { t } [ w ] \left( \widetilde { \alpha } _ { i , w } ^ { ( \mathrm { w } ) } \right) ^ { 2 } > \frac { 1 } { 2 } n _ { t } \alpha _ { t } ^ { 2 } \left( 1 + \chi ^ { 2 } \left( Q _ { 1 } \| \hat { Q } _ { \alpha _ { t } } \right) \right) } \\ { \widehat { P } _ { t , \mathrm { i n t } } ( X _ { t } ^ { n _ { t } } ) } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}
$$

Consequently, from (26), Lemma 5, and (27),

$$
\begin{array} { r l } & { E \left[ \mathrm { v a r } \left( \widehat { P } _ { t , \mathrm { p r e } } [ x ] \Big | W _ { t } ^ { n _ { t } } , \widehat { P } _ { t - 1 } ^ { ( \mathbf { w } ) } \right) \right] \leqslant \frac { 2 \cdot p [ x ] ( 1 - p [ x ] ) } { n _ { t } \alpha _ { t } ^ { 2 } \left( 1 + \chi ^ { 2 } \left( Q _ { 1 } \| Q _ { \alpha _ { t } } \right) \right) } } \\ & { \qquad + \frac { p [ x ] ( 1 - p [ x ] ) } { n _ { t } \alpha _ { t } ^ { 2 } } \cdot \widehat { \mathbf { P } } _ { t , p } ^ { ( \mathbf { w } ) } \left\{ \displaystyle \sum _ { w \in \mathcal { W } } N _ { t } [ w ] \left( \widehat { \alpha } _ { i , w } ^ { ( \mathbf { w } ) } \right) ^ { 2 } \leqslant \frac { 1 } { 2 } n _ { t } \alpha _ { t } ^ { 2 } \left( 1 + \chi ^ { 2 } \left( Q _ { 1 } \| \bar { Q } _ { \alpha _ { t } } \right) \right) \right\} } \\ & { \leqslant \frac { 3 \cdot p [ x ] ( 1 - p [ x ] ) } { \sum _ { i = 0 } ^ { t } n _ { i } \alpha _ { t } ^ { 2 } \left( 1 + \chi ^ { 2 } \left( Q _ { 1 } \| \bar { Q } _ { \alpha _ { t } } \right) \right) } . } \end{array}
$$

$\widehat { P } _ { t , \mathrm { p r e } }$ is unbiased and component-wise uncorrelated with $\widehat { P } _ { t - 1 } ^ { ( \mathrm { w } ) }$ (since $\widehat { p } _ { t , \mathrm { i n t } } ( X _ { t } ^ { n _ { t } } )$ and $\widehat { P } _ { t , \mathrm { i n t } }$ are also unbiased and componentwise uncorrelated). Using Lemma 6 again to combine $\widehat { P } _ { t , \mathrm { p r e } }$ and $\widehat { P } _ { t - 1 } ^ { ( \mathrm { w } ) }$ completes the induction step.

## C. Prooffor the masking estimator upper bound

Recall that the masking procedure takes an unbiased pre-release estimate $\widehat { P } _ { t } ^ { \mathrm { ( p r e ) } }$ and masking decision $B _ { t } \in \{ 0 , 1 \}$ as inputs and releases the estimate

$$
\widehat { P } _ { t } [ x ] = \left\{ \begin{array} { l l } { \widehat { P } _ { t } ^ { ( \mathrm { p r e } ) } [ x ] } & { B _ { t } = 0 } \\ { \mathbb { 1 } \{ S _ { t } = x \} } & { B _ { t } = 1 } \end{array} \right.
$$

where $S _ { t } \sim \widehat { P } _ { t } ^ { \mathrm { ( p r e ) } }$

Proof of Theorem 3: Fix $B _ { i } { = } u _ { i } { \in } \{ 0 , 1 \}$ for each $i \in [ 0 ; t ]$ . Mirroring the notation in Lemma 5, define $\eta _ { i } ^ { 2 } [ x ]$ as the scalar such that

$$
\operatorname { v a r } \Big ( \widehat { P } _ { i } [ x ] \Big | ( B _ { i } ) _ { i \in [ 1 : t ] } = ( u _ { i } ) _ { i \in [ 1 : t ] } \Big ) = \eta _ { i } ^ { 2 } [ x ] \cdot p [ x ] ( 1 - p [ x ] ) .
$$

If $\widehat { P } _ { t } ^ { \mathrm { ( p r e ) } }$ is unbiased, $E [ \mathbb { 1 } \left\{ S _ { t } = x \right\} ] = p$ and therefore $\hat { P } _ { t }$ is also unbiased.

Using the empirical estimator $\widehat { p } _ { 0 , \mathrm { i n t } } = \widehat { p } _ { \mathrm { e m p } }$ in stage 0, and subsequently $\widehat { p } _ { i , \mathrm { i n t } }$ from Lemma 5 in stage $i \geqslant 1$ , we have

$$
\mathrm { v a r } \left( \widehat { P } _ { 0 , \mathrm { i n t } } \Big | ( B _ { i } ) _ { i \in [ 0 : t ] } = ( u _ { i } ) _ { i \in [ 0 : t ] } \right) \leqslant \frac { p [ x ] ( 1 - p [ x ] ) } { n _ { 0 } }
$$

$$
i = 0
$$

$$
\operatorname { v a r } \Big ( \widehat { P } _ { i , \operatorname* { i n t } } [ x ] \Big | ( B _ { j } ) _ { j \in [ 0 : t ] } = ( u _ { j } ) _ { j \in [ 0 : t ] } \Big ) \leqslant \frac { p [ x ] ( 1 - p [ x ] ) } { n _ { i } \alpha _ { i } ^ { 2 } } \cdot ( 1 - ( 1 - \alpha _ { i } ) ^ { 2 } \eta _ { i - 1 } ^ { 2 } [ x ] )
$$

$$
i \geqslant 1
$$

These are all unbiased and uncorrelated with the previous estimate. Therefore, using Lemma 6, we can find the pre-release estimate $\widehat { P } _ { t } ^ { \mathrm { ( p r e ) } }$ with a variance upper bound of

$$
\operatorname { v a r } \Big ( \widehat { P } _ { t } ^ { ( \mathrm { p r e } ) } \Big | ( B _ { i } ) _ { i \in [ 0 : t ] } = ( u _ { i } ) _ { i \in [ 0 : t ] } \Big ) \leqslant \frac { p [ x ] ( 1 - p [ x ] ) } { n _ { 0 } + \displaystyle \sum _ { i = 1 } ^ { t } \frac { n _ { i } \alpha _ { i } ^ { 2 } } { 1 - ( 1 - \alpha _ { i } ^ { 2 } ) \eta _ { i - 1 } ^ { 2 } [ x ] } }
$$

We now consider two cases:

1) The variance of $\hat { P } _ { i } [ x ]$ given $B _ { i } = 1$ is

$$
E \left[ ( 1 - p [ x ] ) ^ { 2 } \widehat { P } _ { i } ^ { ( \mathrm { p r e } ) } [ x ] + p [ x ] ^ { 2 } ( 1 - \widehat { P } _ { i } ^ { ( \mathrm { p r e } ) } ) \right] = ( 1 - p [ x ] ) ^ { 2 } p [ x ] + p [ x ] ^ { 2 } ( 1 - p [ x ] ) = p [ x ] ( 1 - p [ x ] ) .
$$

Thus, if $u _ { i - 1 } = 1$ , then $\eta _ { i - 1 } ^ { 2 } [ x ] = 1$ for all $x \in \mathcal { X }$ , and consequently

$$
1 - ( 1 - \alpha _ { i } ) ^ { 2 } \eta _ { i - 1 } ^ { 2 } [ x ] = 2 \alpha _ { i } .
$$

2) If $B _ { i } = 0$ , we simply use the trivial upper bound of 1 for $1 - ( 1 - \alpha _ { i } ) ^ { 2 } \eta _ { i - 1 } ^ { 2 } [ x ]$

Consequently, given $( B _ { i } ) _ { i \in [ 0 : t ] } = ( u _ { i } ) _ { i \in [ 0 : t ] }$ , we have

$$
\mathrm { v a r } \left( \widehat { P } _ { t } ^ { ( \mathrm { p r e } ) } \Big | ( B _ { i } ) _ { i \in [ 0 : t ] } = ( u _ { i } ) _ { i \in [ 0 : t ] } \right) \leqslant \bar { \sigma } _ { t } ^ { 2 } { : = } \frac { p [ x ] ( 1 - p [ x ] ) } { n _ { 0 } + \displaystyle \sum _ { i = 1 } ^ { t } \frac { n _ { i } \alpha _ { i } } { 2 } u _ { i } + n _ { i } \alpha _ { i } ^ { 2 } ( 1 - u _ { i } ) }
$$

Taking the sum over x and expectation over the joint distribution of $B _ { i }$ and noticing that masking can only increase the loss by $b _ { t }$ , and therefore only a factor of c when $b _ { t } \leqslant c \overline { { \sigma } } _ { t } ^ { 2 }$ concludes the proof.