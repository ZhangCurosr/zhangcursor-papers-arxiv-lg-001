# Eficient and Optimal Learning of Discrete Distributions with Person-Level Privacy

Gautam Kamath<sup>‗</sup> Mahbod Majid<sup>†</sup> Ankur Moitra<sup>‡</sup>

Argyris Mouzakis<sup>§</sup> Jonathan Ullman<sup>¶</sup>

## Abstract

We study learning a discrete distribution under person-level diferential privacy in the batched setting. We are given � users, each contributing a batch of � i.i.d. samples from an unknown distribution $\mu$ over [�], and seek an estimate of $\mu$ up to total-variation error �. We give polynomial-time algorithms under both pure and approximate diferential privacy. Under pure (�<sub>,</sub> 0)-DP, we achieve

$$
n = \widetilde { O } \left( \frac { k } { \alpha ^ { 2 } m } + \frac { k } { \alpha \sqrt { m } \varepsilon } + \frac { k } { \varepsilon } \right) ,
$$

the first polynomial-time pure-DP guarantee for this problem that matches the informationtheoretic optimum up to logarithmic factors. Under approximate (� �)-DP, we achieve

$$
n = \widetilde { O } \biggl ( \frac { k } { \alpha ^ { 2 } m } + \frac { k } { \alpha \sqrt { m } \varepsilon } + k + \frac { \log ( 1 / \delta ) } { \varepsilon } \biggr ) .
$$

At $\delta = 2 ^ { - \Theta ( k ) }$ , the bound remains $\widetilde O ( k )$ for fixed $\alpha , \varepsilon , m ,$ and failure probability. Our algorithms use a new degree-6 sum-of-squares estimator for learning from untrusted batches. Its key ingredient is a trace-bounded diagonal majorizer for the diference between the empirical covariance of the repaired batch averages and the multinomial covariance predicted by their mean. We prove that clean data admit this certificate through semidefinite-programming duality and a new application of the stronger symmetric form of Grothendieck’s inequality due to Friedland and Lim. The certificate preserves the mean-dependent variance of multinomial batch averages, yielding the near-optimal robust rate $O ( \eta \sqrt { \log ( 1 / \eta ) / m } )$ and providing the score required by the privacy reduction. We also prove a statistical query lower bound that rules out a uniformly smaller order of error under fixed polynomial SQ resource bounds.

## Contents

1 Introduction 3   
1.1 Results 4   
1.2 Techniques . 7   
1.3 Related Work 11   
2 Preliminaries 12   
2.1 Data Model and Notation 13   
2.2 Probability Simplex . 13   
2.3 Concentration Bounds 15   
2.4 Sum-of-Squares 16   
2.5 Subset-Variance Control and the Diagonal-Majorizer SDP 17   
2.6 Robustness to Privacy . 19   
3 Robust Estimation for Discrete Batches 21   
3.1 Sum-of-Squares Based Robust Algorithm 22   
3.2 Proof of the Main SoS Lemma . 27   
3.3 Covariance Closeness Guarantees 30   
3.4 Heterogeneous Clean Batches 32   
4 Private Estimation for Discrete Batches 33   
4.1 Properties of the Score 36   
4.1.1 Well-Definedness of the Score Function 36   
4.1.2 Bounded Sensitivity 36   
4.1.3 Quasi Convexity 37   
4.1.4 Eficient Computability 37   
4.2 Problem-Specific Consequences 38   
4.2.1 Eficiently Finding Low Scoring Points . 38   
4.2.2 Volume . 39   
4.2.3 Proof of the Pure-DP Theorem 43   
4.2.4 Proof of the Approximate-DP Theorem 44   
5 Statistical Query Lower Bound for Discrete Batches 50   
5.1 Testing Problem 50   
5.2 SQ Hardness . 51   
5.3 From Estimation to Testing . 54   
5.4 Binomial Moment Construction 54   
References 57   
A Details for the Approximate-DP Section 62   
B Pure DP Lower Bounds 68   
B.1 Projection Notation for Lower Bounds 68

## 1 Introduction

Federated learning $[ \mathrm { M M R ^ { + } } 1 7 , \mathrm { K M A ^ { + } } 2 1 ]$ is motivated by the tension between two important goals in modern machine learning:

1. to train machine learning models from large and diverse data sets, and

2. while doing so, to preserve the data privacy, security, and ownership of the data, since data is increasingly distributed across a network of devices.

Motivated by such considerations, Qiao and Valiant [QV18] introduced the following problem which they called learningfrom untrusted batches. There exists an unknown discrete distribution $\mu$ on [�] and data is drawn in � batches, each of size $m .$ . When all batches are drawn i.i.d. from $\mu$ and can be used without restriction, the empirical distribution achieves total-variation error $\alpha ,$ provided that the number of batches satisfies

$$
n = \Theta \left( \frac { k + \log ( 1 / \beta ) } { \alpha ^ { 2 } m } \right) ,
$$

where $\beta$ is the failure probability. This follows from the classical bound for �� independent samples [Can20].

However, in federated learning, we can rarely assume that all the batches are sampled i.i.d. from $\mu ,$ and instead must consider the possibility that some are corrupted, possibly in an adversarial way. To address this question, Qiao and Valiant studied the problem with the twist that an �-fraction of the batches are adversarially corrupted. Again, it is helpful to compare to a naive baseline. If we ignore the batch structure and treat the data as a single dataset of �� samples, then it is easy to see that we cannot learn the distribution to accuracy better than $\eta$ in total variation distance, no matter how much data we have. Qiao and Valiant demonstrated that when we exploit the batched structure of the data, it is possible to do much better. In particular, they showed that it is information-theoretically possible to achieve error $O ( \eta / \sqrt { m } )$ , and that this is the best possible. Batch information makes it possible to robustly distinguish distributions at much finer scale. Without batch information, we cannot distinguish two distributions $\mu$ and $\tilde { \mu }$ that are �-close in total variation distance in the face of an adversary, because they can convert one distribution to the other by moving just a $\gamma$ fraction of the probability mass. But when data is batched and the adversary $r _ { \mathrm { { S } } }$ corruptions must respect the batch structure, the total variation distance between $\mu ^ { \otimes m }$ and $\tilde { \mu } ^ { \otimes m }$ grows at least as Ω(min $\{ 1 , \gamma \sqrt { m } \} ,$ ), meaning the adversary has to corrupt a much larger fraction of batches to make these distributions indistinguishable.

Qiao and Valiant gave two algorithms that solve the learning from untrusted batches problem, running in time exponential in � or �, respectively. Jain and Orlitsky [JO20b] gave a semidefiniteprogramming algorithm based on Grothendieck’s inequality that roughly matches the informationtheoretic bounds and runs in polynomial time. In independent and concurrent work, Chen, Li, and Moitra [CLM20a] gave similar algorithms and explored extensions to more complex settings, including improving sample complexity when there are shape constraints on $\mu .$

Qiao and Valiant motivated their model by the need to learn from batches supplied by potentially malicious agents. Privacy, communication, and computational costs are also central concerns in federated learning. Keeping raw data local does not by itself ensure privacy: shared gradients can still leak private training data [ZLH19].

In this work, we revisit the learning from batches problem using the lens of diferential privacy [DMNS06]. There is an unknown discrete distribution $\mu$ on [�] and data is drawn in � batches of size �. As is common in federated learning, we think of each batch of data as belonging to one person (e.g. a batch consists of the data on one person’s device), and want to design a batch-level diferentially private<sup>1</sup> algorithm that, informally, hides the information of any one batch of data. Formally, two datasets are adjacent if they difer in one person’s entire batch. Batch-level privacy protects a person’s entire contribution, whereas sample-level privacy protects only one data item. Diferential privacy bounds how much the output distribution can change when one person’s data is replaced.

We pursue three goals. First, we require the strong guarantee of pure batch-level diferential privacy, formally $( \varepsilon , 0 ) \mathopen { } \mathclose \bgroup  - \mathrm { D P } . ^ { 2 }$ Second, we seek the information-theoretically optimal user complexity. Prior work established that, up to logarithmic factors, the optimum for learning to total-variation error � is [GKK<sup>+</sup>23b, Theorem 43]

$$
n = \widetilde { O } \left( \frac { k } { \alpha ^ { 2 } m } + \frac { k } { \alpha \sqrt { m } \varepsilon } + \frac { k } { \varepsilon } \right) .
$$

Our final goal is to attain this bound in polynomial time.

Several groups have considered private learning from batches, but none give an algorithm satisfying all three goals. Liu et al. [LSY 20] introduced the problem, giving an eficient algorithm achieving the weaker notion of approximate diferential privacy, or, formally, $( \varepsilon , \delta )$ -diferential privacy. One can strengthen the privacy guarantee of their algorithm to $( \varepsilon , 0 )$ -diferentially private by appealing to basic composition of pure diferential privacy instead of advanced composition of approximate diferential privacy [DRV10], but doing so would increase the required number of batches by a poly(�) factor and would not achieve optimal sample complexity. Ghazi, Kamath, Kumar, Manurangsi, Meka, and Zhang [GKK<sup>+</sup>23b] presented a pure diferentially private algorithm with the information-theoretically optimal sample complexity, but their stated construction applies private hypothesis selection over an exponentially large cover and is not polynomial-time.

We use the eficient robustness-to-privacy reduction of Hopkins, Kamath, Majid, and Narayanan [HKMN23]. Our contribution is to identify the robust certificate that makes this reduction work for discrete batches. It is a single degree-6 sum-of-squares (SoS) certificate that represents discarded batches, mean feasibility, and the mean-dependent covariance of multinomial batch averages. This certificate simultaneously yields a robust estimator and the score used for privacy.

## 1.1 Results

Our certificate is satisfied by clean data at the right scale, forces every feasible pseudo-mean to be close to the truth, remains stable when batches are discarded, and admits polynomial-time optimization. It yields the robust estimator in Theorem 1.1 and, through the reduction of Hopkins et al. [HKMN23], the private algorithms in Theorem 1.3.

Theorem 1.1 (Robust SoS Estimation from Discrete Batches). Let $\mu \in \Delta ^ { k - 1 } , \beta \in ( 0 , 1 )$ , and $m \in \mathbb { N } .$ . Let $\eta > 0$ be smaller than a suficiently small absolute constant. Then there exists a polynomial-time algorithm, obtained by solving a degree-6 SoS relaxation, that takes in � user batches of size � forming an �-corruption of independent clean batches

$$
B _ { i } ^ { \star } = ( U _ { i , 1 } ^ { \star } , \ldots , U _ { i , m } ^ { \star } ) , \qquad U _ { i , 1 } ^ { \star } , \ldots , U _ { i , m } ^ { \star } \stackrel { \mathrm { i i d } } { \sim } \mu ,
$$

<sup>1</sup>This notion of privacy is also frequently called person-level or user-level diferential privacy in the literature.

and outputs an estimate $\hat { \mu }$ such that, with probability at least $1 - \beta ,$

$$
\| \hat { \mu } - \mu \| _ { \mathrm { T V } } \leqslant O \Bigl ( \eta \sqrt { \log ( 1 / \eta ) / m } \Bigr ) ,
$$

provided that

$$
n = \Omega \left( \frac { k + \log ( 1 / \beta ) } { \alpha ^ { 2 } m } \right) \qquad f o r \alpha = \Theta \left( \eta \sqrt { \log ( 1 / \eta ) / m } \right) .
$$

Remark 1.2 (Robustness to per-batch distribution shift). Our robust estimator also tolerates mild heterogeneity across the clean batches, with the modified covariance allowance in Corollary 3.11. Namely, in addition to the �-corruption model in Theorem 1.1, one may allow the samples in each clean batch � to be drawn i.i.d. from a distribution $\mu _ { i }$ satisfying $\| \mu _ { i } - \mu \| _ { \mathrm { T V } } \leqslant \omega$ . In that case the guarantee worsens by only an additive $O ( \omega )$ term, giving

$$
\| \hat { \mu } - \mu \| _ { \mathrm { T V } } \leqslant O \left( \eta \sqrt { \log ( 1 / \eta ) / m } + \omega \right) .
$$

The diagonal-majorizer certificate defines a low-sensitivity, quasi-convex score (Definition 4.5), so the eficient robustness-to-privacy reduction gives the private guarantees stated next.

Theorem 1.3 (Main Result, Private Estimation from Discrete Batches). Let $\mu \in \Delta ^ { k - 1 }$ be a discrete distribution over [�]. Let � $, \beta \in ( 0 , 1 ) , \varepsilon > 0$ , and $m \in \mathbb { N } .$ . Suppose that the batches are independent across users and each of the � users contributes a batch

$$
B _ { i } = ( U _ { i , 1 } , \ldots , U _ { i , m } ) , \qquad U _ { i , 1 } , \ldots , U _ { i , m } \overset { \mathrm { i i d } } { \sim } \mu .
$$

Then the following guarantees hold.

(i) There is a polynomial-time $( \varepsilon , 0 ) – D P$ algorithm that outputs $\hat { \mu }$ satisfying $\| { \hat { \boldsymbol { \mu } } } - \mu \| _ { \mathrm { T V } } \leqslant \alpha$ with probability at least $1 - \beta ,$ , provided

$$
n = \widetilde O \left( \frac { k + \log ( 1 / \beta ) } { \alpha ^ { 2 } m } + \frac { k + \log ( 1 / \beta ) } { \alpha \sqrt { m } \varepsilon } + \frac { k + \log ( 1 / \beta ) } { \varepsilon } \right) .
$$

(ii) For every $\delta \in ( 0 , 1 )$ , there is a polynomial-time $( \varepsilon , \delta ) – D P$ algorithm with the same accuracy andfailure probability, provided

$$
n = \widetilde { O } \left( k + \frac { k + \log ( 1 / \beta ) } { \alpha ^ { 2 } m } + \frac { k + \log ( 1 / \beta ) } { \alpha \sqrt { m } \varepsilon } + \frac { \log ( 1 / \delta ) + \log ( 1 / \beta ) } { \varepsilon } \right) .
$$

The $\widetilde { O }$ notation suppresses logarithmic factors in $1 / \alpha ;$ Remark 4.16 gives explicit bounds, including the square-root logarithm in the approximate-DP privacy term. Our pure-DP rate matches known lower bounds up to logarithmic factors $[ \mathrm { L S Y ^ { + } } 2 0 , \mathrm { G K K ^ { + } } 2 3 \mathrm { b } ] ;$ Theorem B.5 gives the dependence on the failure probability explicitly.

Prior polynomial-time approximate-DP results cover diferent user regimes. Suppressing logarithmic factors other than those displayed below, the relevant privacy terms and user-count requirements are

$$
\mathrm { L i u ~ e t ~ a l . ~ } [ \mathrm { L S Y ^ { + } 2 0 } ] : \quad \frac { k \sqrt { \log ( 1 / \delta ) } } { \alpha \varepsilon \sqrt { m } } , \qquad \quad n \gtrsim \frac { \sqrt { k \log ( 1 / \delta ) } } { \varepsilon } ,
$$

$$
\mathrm { ( f e w ~ u s e r s ) ~ E s f a n d i a r i ~ e t ~ a l . ~ [ E M N 2 2 ] } \colon \frac { k \log ( k / \delta ) \sqrt { \log ( 1 / \delta ) \log ( k n / \beta ) } } { \alpha \varepsilon \sqrt { m } } , \quad n \gtrsim \frac { \log ( 1 / ( \beta \delta ) ) } { \varepsilon } .
$$

Esfandiari et al.’s few-user guarantee removes the dimension-dependent user floor required by Liu et al. Their all-regime result combines this guarantee with earlier algorithms for the high-user regime; its larger logarithmic factor is the cost of reaching the few-user regime. For negligible approximate-privacy parameters $\delta = 2 ^ { - k ^ { \Omega ( 1 ) } }$ and fixed $\alpha , \varepsilon , m ,$ the privacy terms of both Liu et al. and Esfandiari et al. are superlinear in �. Our bound has additive dependence on $\log ( 1 / \delta )$ and includes a $\widetilde O ( k )$ user term for clean-data feasibility. At $\delta = 2 ^ { - \Theta ( k ) }$ , our bound is $\widetilde O ( k )$ for fixed $\alpha , \varepsilon , m$ , and failure probability.

Dependence on the batch size. For constant failure probability and $0 < \varepsilon \leqslant 1$ , when $\alpha \sqrt { m } \leqslant 1$ the term $k / ( \alpha \sqrt { m } \varepsilon )$ absorbs $k / \varepsilon ,$ so the pure-DP theorem already gives the approximate-DP rate up to logarithmic factors. When $\alpha \sqrt { m } \geqslant 1$ , we cap the internal corruption parameter at a suficiently small constant. The robust error is then at most $\alpha ,$ giving a pure-DP rate of $\widetilde { O } ( k / \varepsilon )$

The additional � in the approximate-DP bound is inherent in using our robust estimator at its stated accuracy. For any fixed suficiently small corruption fraction $\eta ,$ Theorem 1.1 guarantees error $O ( \eta \sqrt { \log ( 1 / \eta ) / m } ) = O ( 1 / \sqrt { m } )$ , which requires $\Omega ( k )$ users even without corruptions or privacy. Indeed, the nonprivate sample bound $\Omega ( k \bar { / } ( \alpha ^ { 2 } m ) )$ becomes Ω(�) at this accuracy. This requirement is already implicit in the robust theorem through $\alpha = \Theta ( \eta \sqrt { \log ( 1 / \eta ) / m } )$ , and is absorbed by $k / \varepsilon$ for pure DP. It appears explicitly for approximate DP because the other terms can be smaller than $k .$

For larger batches, the target error � can be much larger than $1 / { \sqrt { m } }$ , and approximate DP can use fewer than $k$ users [EMN22]. Exploiting this regime through robustness-to-privacy requires a robust guarantee adapted to the coarser target accuracy. Remark 4.17 describes how splitting batches reduces the user requirement while retaining additive dependence on log(1/�), and how to combine our estimator with that of Esfandiari et al. using only the public parameters.

On optimality under eficient computation. Our polynomial-time robust algorithm achieves error $O ( \eta \sqrt { \log ( 1 / \eta ) / m } )$ , while the information-theoretic optimum is $O ( \eta / \sqrt { m } )$ [QV18]. Such an extra $\sqrt { \log ( 1 / \eta ) }$ factor has precedent in statistical query (SQ) lower bounds for robust estimation of binary product distributions [DKS22] and Gaussians [DKS17]. In Section $5 ,$ we prove a lower bound directly for categorical batches. For every fixed integer $t \geqslant 1$ , we give a query–tolerance tradeof at error a constant $c _ { t } > 0$ times

$$
\eta \operatorname* { m i n } \biggl \{ 1 , \sqrt { \frac { \log ( 1 / \eta ) } { m } } \biggr \} .
$$

For all suficiently large � depending on � and every batch size, achieving this error requires either a query of tolerance less than $2 k ^ { - ( \check { t } + 1 ) / 8 }$ or $\exp ( \Omega ( { \sqrt { k } } ) )$ queries. Consequently, no SQ estimator with fixed polynomial resource bounds in $k , m , 1 / \eta$ achieves uniformly smaller order of error. This rate matches the dependence on � and $\eta$ of our robust estimator when � $\geqslant \log ( 1 / \eta )$ . For smaller batches, the contaminated empirical distribution attains error ${ \cal O } ( \eta )$ with suficiently many samples. Our construction uses moment matching as in [DKS22], with a discrete polynomial correction and a balanced subset of the original alphabet. If this SQ lower bound captures the computational threshold for robust estimation, then the privacy-to-robustness connection of Georgiev and Hopkins [GH22] predicts a corresponding threshold for private estimation with high success probability. For $0 < \varepsilon \leqslant 1$ and $\beta = \exp ( - \Theta ( k ) )$ , group privacy gives robustness to an $\eta = \Theta ( k / ( n \varepsilon ) )$ fraction of corrupted batches, provided this fraction is suficiently small. When $\alpha \sqrt { m }$ is small and $m \geqslant \log ( 1 / \eta )$ , substituting this value into the robust threshold predicts a privacy term

of order

$$
\frac { k \sqrt { \log ( 1 / ( \alpha \sqrt { m } ) ) } } { \alpha \sqrt { m } \varepsilon } .
$$

This agrees with the explicit privacy term in Remark 4.16, including the square-root logarithm. Georgiev and Hopkins derive an analogous conditional square-root logarithmic gap for Gaussian mean estimation [GH22, full version, Corollary 4.2]. This interpretation also applies to approximate DP for $\delta \leqslant \varepsilon \exp ( - C k )$ with a suficiently large constant �.

Remark 1.4 (Robustness of the private estimators to corruptions). For $0 \textless \varepsilon \leqslant 1$ , both private estimators tolerate an �-fraction of fully corrupted user batches, under their respective sample bounds in Theorem 1.3, provided

$$
\alpha \geqslant C \eta \sqrt { \log ( 1 / \eta ) / m }
$$

for a suficiently large absolute constant � and suficiently small �. Both algorithms also tolerate clean-user distributions $\mu _ { i }$ satisfying $\| \mu _ { i } - \mu \| _ { \mathrm { T V } } \leqslant \omega ,$ provided the algorithm is given � and the accuracy floor includes an additive �� term. For approximate DP, a suficient sample bound includes an additional $\widetilde { O } ( k \omega / ( \alpha \varepsilon ) )$ term, which is absorbed by the stated bound when � $\sqrt { m } \leqslant 1$ or when � is constant. We prove these claims in Corollary 4.18.

## 1.2 Techniques

We first describe the subset-variance condition that identifies the unknown distribution, then encode it by a polynomial-size semidefinite constraint. We define the score used by the privacy reduction as the minimum number of batches that must be discarded to satisfy this constraint.

Subset-variance control. Let $y _ { i } \in \Delta ^ { k - 1 }$ denote the empirical distribution of the �th batch. Fix a subset $S \subseteq [ k ]$ and let $s \in \{ 0 , 1 \} ^ { k }$ be its indicator. Then $\left. y _ { i } , s \right.$ is simply the fraction of the �th batch that falls in $S ,$ and has variance $\mu ( S ) ( 1 - \mu ( S ) ) / m$ . We write this variance as $V ( \nu , s ) : = \langle \nu , s \rangle ( 1 - \langle \nu , s \rangle ) / m$ when the underlying distribution is �.

A statistically natural goal is to find a set $T \subseteq [ n ]$ with $| T | \geqslant ( 1 - \eta ) n$ such that, for $\bar { \mu } _ { T } : =$ $\begin{array} { r } { | T | ^ { - 1 } \sum _ { i \in T } y _ { i } , } \end{array}$

$$
\left| \frac { 1 } { | T | } \sum _ { i \in T } \langle y _ { i } - \bar { \mu } _ { T } , s \rangle ^ { 2 } - V ( \bar { \mu } _ { T } , s ) \right| \leqslant O \left( \frac { \eta \log ( 1 / \eta ) } { m } \right) \qquad \mathrm { f o r ~ e v e r y ~ } s \in \{ 0 , 1 \} ^ { k } .\tag{1}
$$

This condition says that, in every subset direction, the empirical variance should agree with the multinomial variance predicted by the empirical mean $\bar { \mu } _ { T }$

Representing this condition poses a problem: there are exponentially many subset directions, and listing all of them does not give the succinct, low-sensitivity certificate needed for privacy. Earlier robust algorithms do not directly provide such a certificate. Jain and Orlitsky [JO20b] obtain the right rate through filtering and randomized rounding. Their guarantee does not directly supply the eficiently computable, sensitivity-one, quasi-convex score needed by the privacy reduction. Chen, Li, and Moitra [CLM20a] give an SoS formulation. Bounding only second moments in their argument gives $O ( { \sqrt { \eta / m } } )$ error, polynomially worse in � than our rate. Their higher-moment formulation gives error $O ( \eta ^ { 1 - 1 / t } \sqrt { t / m } )$ using degree-�(�) constraints; obtaining $O ( \eta \sqrt { \log ( 1 / \eta ) / m } )$ by this approach uses $t = \Theta ( \log ( 1 / \eta ) )$ and hence quasipolynomial resources.

Compressing the subset constraints. The subset-variance condition (1) contains one inequality for each of the $2 ^ { k }$ subsets of [�], so it cannot be inserted directly into a polynomial-size program. We replace this family by one semidefinite condition: the diference between the empirical covariance and the multinomial covariance predicted by the empirical mean must lie below a diagonal matrix with small trace. We call this diagonal matrix a majorizer.

Two facts justify the replacement. First, clean data admit such a majorizer; proving this uses semidefinite duality and the stronger symmetric Grothendieck inequality of Friedland and Lim [FL20]. Second, testing any feasible majorizer on sign vectors recovers the required upper variance bound in every subset direction. Thus the majorizer is a polynomial-size encoding of the one-sided subset control we need.

Robust SoS program. We define the relaxation in Section 3.1. Its constraints have the following schematic interpretation: retain almost all batches, repair the discarded ones, let $\mu ^ { \prime }$ be the mean of the repaired data, and require the repaired covariance to lie below the multinomial covariance plus a low-trace diagonal matrix. We write $\begin{array} { r } { \mathbb { E } _ { i } z _ { i } : = n ^ { - 1 } \sum _ { i = 1 } ^ { n } z _ { i } } \end{array}$ and search over selectors $w _ { i , }$ repaired batch averages $y _ { i } ^ { \prime } ,$ a candidate mean $\mu ^ { \prime } .$ , a diagonal vector $\lambda ,$ , and a matrix � satisfying

$$
w _ { i } ^ { 2 } = w _ { i } , \qquad w _ { i } y _ { i } ^ { \prime } = w _ { i } y _ { i } , \qquad \sum _ { i = 1 } ^ { n } w _ { i } \geqslant ( 1 - \eta ) n ,
$$

$$
y _ { i } ^ { \prime } \in \Delta ^ { k - 1 } \operatorname { f o r } \mathrm { a l l } i , \qquad \mu ^ { \prime } \in \Delta ^ { k - 1 } , \qquad \mu ^ { \prime } = \mathbb { F } y _ { i } ^ { \prime } ,
$$

$$
\widehat { \Sigma } ^ { \prime } + M M ^ { \top } = \Sigma ( \mu ^ { \prime } ) + \mathrm { D i a g } ( \lambda ) , \qquad \sum _ { j = 1 } ^ { k } \lambda _ { j } \leqslant \tau _ { \mathrm { c o v } } ,
$$

where

$$
\widehat { \Sigma } ^ { \prime } : = \mathbb { E } ( y _ { i } ^ { \prime } - \mu ^ { \prime } ) ( y _ { i } ^ { \prime } - \mu ^ { \prime } ) ^ { \top } , \qquad \Sigma ( \nu ) : = \frac { 1 } { m } \big ( \mathrm { d i a g } ( \nu ) - \nu \nu ^ { \top } \big ) , \qquad \tau _ { \mathrm { c o v } } \asymp \frac { \eta \log ( 1 / \eta ) } { m } .
$$

We denote this system by $\mathcal { A } _ { \eta }$ . In the first line, we select the retained batches and enforce consistency with the observed data; in the second, we require the repaired batch averages and their mean to be distributions. For every subset indicator $s ,$ we have $s ^ { \top } \Sigma ( \nu ) s = V ( \nu , s )$ , so the covariance line encodes the upper bound in (1) for the repaired data. Here � is an auxiliary slack variable: writing the positive-semidefinite slack as $M M ^ { \top }$ encodes the inequality $\widehat \Sigma ^ { \prime } - \Sigma ( \mu ^ { \prime } ) \precsim \mathrm { D i a g } ( \lambda )$ as a polynomial equality.

The program repairs discarded batches rather than removing and renormalizing them. On the clean witness, a discarded batch is filled in by its latent clean value; in the soundness proof, the influence of all repaired batches is controlled through their second moment. This represents the upper bound in (1) by a polynomial system over all � batches.

We solve the degree-6 relaxation and use its pseudo-mean $\widetilde { \mathbb { E } } \mu ^ { \prime }$ as the estimate, with an $\ell _ { 1 }$ projection onto the simplex to account for numerical error. There is no filtering or randomized rounding afterward. It remains to justify the covariance constraint: clean data must admit a low-trace majorizer, and every feasible majorizer must recover the required subset bounds. We prove these statements in this order.

Clean-data feasibility. We assign the selector, repaired batches, and candidate mean directly from the latent clean data. Let $r _ { i } = \mathbb { 1 } [ y _ { i } = y _ { i } ^ { * } ]$ and $\bar { \boldsymbol { \mu } } = \mathbb { E } _ { i } \boldsymbol { y } _ { i } ^ { * }$ , and set

$$
w _ { i } = r _ { i } , \qquad y _ { i } ^ { \prime } = y _ { i } ^ { * } , \qquad \mu ^ { \prime } = { \bar { \mu } } .
$$

These assignments satisfy the selector and simplex constraints by inspection. It remains to construct �. Define

$$
\widehat { \Sigma } ^ { \star } : = \mathbb { E } ( y _ { i } ^ { \star } - \bar { \mu } ) ( y _ { i } ^ { \star } - \bar { \mu } ) ^ { \top } , \qquad \Delta _ { \mathrm { c o v } } ^ { \star } : = \widehat { \Sigma } ^ { \star } - \Sigma ( \bar { \mu } ) .
$$

If $\Delta _ { \mathrm { c o v } } ^ { \star }$ has a diagonal majorizer with trace at most $\tau _ { \mathrm { { C O V } } } ,$ then we choose $M M ^ { \top } = \mathrm { D i a g } ( \lambda ) - { \Delta } _ { \mathrm { c o v } } ^ { \star }$ and the clean witness is complete.

The only probabilistic input is the �-good clean-sample event. Under the sample-size assumption in Theorem 1.1, Corollary 3.3 guarantees this event with the required probability. For feasibility, we use only the following two consequences:

$$
\operatorname* { s u p } _ { s \in \{ 0 , 1 \} ^ { k } } \Big | s ^ { \top } \Delta _ { \mathrm { c o v } } ^ { \star } s \Big | \leqslant O \bigg ( \frac { \eta \log ( 1 / \eta ) } { m } \bigg ) , \qquad \Delta _ { \mathrm { c o v } } ^ { \star } \mathbf { 1 } = 0 .
$$

We prove the subset-discrepancy bound in Lemma 3.4, and the identity $\Delta _ { \mathrm { c o v } } ^ { \star } \mathbf { 1 } = 0$ in Lemma 3.5.   
From this point on, the feasibility argument is deterministic.

From subset bounds to a diagonal majorizer. Let Δ be any symmetric matrix satisfying $\Delta \mathbf { 1 } = 0$ Combining diagonal-majorizer duality from Corollary 2.16, the Friedland–Lim inequality from Corollary 2.14 [FL20], and the substitution $\boldsymbol { \sigma } = 2 s - \mathbf { 1 }$ gives

$$
\begin{array} { r l } { \underset { \lambda \in \mathbb { R } ^ { k } : \mathrm { \mathrm { ~ D i a g } } ( \lambda ) \approx \Delta } { \mathrm { m i n } } \underset { j = 1 } { \overset { k } { \sum } } \lambda _ { j } = \underset { X \underset { \forall 0 } { \mathrm { m a x } } } { \mathrm { m a x } } \langle \Delta , X \rangle } & { } \\ { \leqslant \underset { X \underset { \forall 0 } { \mathrm { m a x } } } { \mathrm { m a x } } | \langle \Delta , X \rangle | } & { } \\ { \leqslant C _ { \mathrm { F L } } \underset { \sigma \in \{ 1 \} } { \mathrm { m a x } } | \langle \Delta ^ { \top } \Delta \sigma \big \vert = 4 C _ { \mathrm { F L } } \underset { s \in \{ 0 , 1 \} ^ { k } } { \mathrm { m a x } } \big \vert s ^ { \top } \Delta s \big \vert , } & { } \end{array}
$$

where $C _ { \mathrm { F L } }$ is a universal constant. The final equality follows by expanding $\boldsymbol { \sigma } = 2 s - \mathbf { 1 }$ and using $\Delta \mathbf { 1 } = 0$ . This explains why the nullspace condition matters: it turns every sign-vector quadratic form into four times a subset quadratic form. Friedland–Lim passes from sign vectors to the correlation SDP, and duality turns the SDP bound into the diagonal vector � used by the program. See Lemma 2.11 and corollaries 2.14 and 2.16 for the details.

Applying this chain to $\Delta _ { \mathrm { c o v } } ^ { \star }$ produces a majorizer of trace ${ \cal O } ( \eta \log ( 1 / \eta ) / m )$ . Choosing the constant in $\tau _ { \mathrm { c o v } }$ suficiently large therefore completes the clean witness. We next recover the one-sided subset bound needed for identifiability.

Recovering the subset bounds. To recover the subset bounds, let $\Delta _ { \mathrm { c o v } } ^ { \prime } : = \widehat { \Sigma } ^ { \prime } - \Sigma ( \mu ^ { \prime } )$ . Both terms annihilate 1, so $\Delta _ { \mathrm { c o v } } ^ { \prime } \mathbf { 1 } = 0$ . For any fixed $s \in \{ 0 , 1 \} ^ { k }$ , set $\boldsymbol { \sigma } = 2 s - \mathbf { 1 }$ . Testing the majorizer on � gives

$$
4 s ^ { \top } \Delta _ { \mathrm { c o v } } ^ { \prime } s = \sigma ^ { \top } \Delta _ { \mathrm { c o v } } ^ { \prime } \sigma \leqslant \sigma ^ { \top } \mathrm { D i a g } ( \boldsymbol { \lambda } ) \sigma = \sum _ { j = 1 } ^ { k } \lambda _ { j } .
$$

Thus testing the majorizer on the sign vector associated with � recovers the desired bound in that subset direction. Since $s ^ { \top } \Sigma ( \mu ^ { \prime } ) s = \bar { V } ( \mu ^ { \prime } , s )$ , we obtain the following degree-6 SoS upper bound on the repaired directional variance:

$$
\mathcal { A } _ { \eta } \Big | _ { \mathsf { R } } \left\{ \mathbb { E } \langle y _ { i } ^ { \prime } - \mu ^ { \prime } , s \rangle ^ { 2 } \leqslant V ( \mu ^ { \prime } , s ) + O \left( \frac { \eta \log ( 1 / \eta ) } { m } \right) \right\} .
$$

This is the upper bound in (1), applied to the repaired data. The two-sided subset-discrepancy bound in Lemma 3.4 was needed to construct the majorizer. For identifiability, the program uses only the repaired-variance upper bound, together with the clean-retained second-moment lower bound in (9).

Low-degree SoS identifiability proof. We argue in one dimension. Fix a subset direction $s \in \{ 0 , 1 \} ^ { k }$ . Readers unfamiliar with SoS may read $\mathcal { A } _ { \eta } \ \middle \vert _ { 6 } \ \{ \cdots \}$ as “the program certifies the inequality inside the braces.” The degree only records that the certificate remains valid under the degree-6 pseudoexpectation; statistically, this is the usual directional robust-mean argument.

On the �-good clean-sample event, at most a 2� fraction of the batches fail to be both clean and retained. Weighted Cauchy–Schwarz reduces their contribution to a comparison between the repaired and clean-retained second moments. Since $\mu ^ { \prime } = \mathbb { E } _ { i } y _ { i } ^ { \prime } ,$ , expanding $y _ { i } ^ { \prime } - \mu = ( y _ { i } ^ { \prime } - \mu ^ { \prime } ) + ( \mu ^ { \prime } - \mu )$ makes the cross term vanish. The majorizer then upper-bounds the repaired second moment through $V ( \mu ^ { \prime } , s )$ , while the clean-sample event lower-bounds the clean-retained second moment through $V ( \mu , s )$ , giving

$$
\mathcal { A } _ { \eta }  _ { 6 } \{ \langle \mu ^ { \prime } - \mu , s \rangle ^ { 2 } \leqslant O ( \frac { \eta ^ { 2 } \log ( 1 / \eta ) } { m } ) + 4 \eta \big ( V ( \mu ^ { \prime } , s ) - V ( \mu , s ) \big ) + 4 \eta \langle \mu ^ { \prime } - \mu , s \rangle ^ { 2 } \} .\tag{2}
$$

We give the full calculation in Section 3.2. In (2), the $O ( \eta ^ { 2 } \log ( 1 / \eta ) / m )$ term has the desired size, and the term $4 \eta \langle \mu ^ { \prime } - \mu , s \rangle ^ { 2 }$ can be absorbed into the left-hand side. It remains to control the variance diference $V ( \mu ^ { \prime } , s ) - V ( \mu , s )$

To control the variance diference, we keep the covariance predicted by $\mu ^ { \prime }$ rather than use a universal upper bound. For every distribution �, we have $s ^ { \top } \Sigma ( \nu ) s = V ( \nu , s ) = \langle \nu , s \rangle ( 1 - \langle \nu , s \rangle ) / m$ It follows that $V ( \mu ^ { \prime } , s ) - V ( \mu , s )$ is controlled by the directional mean error $\langle \mu ^ { \prime } - \mu , s \rangle$ . By contrast, using only $V ( \mu ^ { \prime } , s ) \leqslant 1 / ( 4 m )$ would leave an ${ \cal O } ( \eta / m )$ term and yield only ${ \cal O } ( \sqrt { \eta } / m )$ directional error. Bounding second moments in the formulation of Chen, Li, and Moitra [CLM20a] encounters this limitation; their higher-moment constraints improve the rate.

To absorb the term $4 \eta ( V ( \mu ^ { \prime } , s ) - V ( \mu , s ) )$ in (2), Lemma 3.10 bounds the square of the variance diference by the squared mean error, as shown in the first line below. Apply Young’s inequality from Fact 2.8 with $a = V ( \mu ^ { \prime } , s ) - V ( \mu , s ) , b = 4 \eta$ , and $\lambda = m ^ { 2 } / 1 6$ . This gives the second line, and substituting the first line gives the third:

$$
\begin{array} { r l r } {  { \mathcal { A } _ { \eta }  _ { 6 }  _ { 6 } \{ 4 \eta \bigl ( V ( \mu ^ { \prime } , s ) - V ( \mu , s ) \bigr ) ^ { 2 } \leqslant \frac { 4 } { m ^ { 2 } } \langle \mu ^ { \prime } - \mu , s \rangle ^ { 2 } ,  } } \\ & { } & {  \mathcal { A } _ { \eta }  _ { 6 } \{ 4 \eta \bigl ( V ( \mu ^ { \prime } , s ) - V ( \mu , s ) \bigr ) \leqslant \frac { m ^ { 2 } } { 1 6 } \bigl ( V ( \mu ^ { \prime } , s ) - V ( \mu , s ) \bigr ) ^ { 2 } + \frac { 6 4 \eta ^ { 2 } } { m ^ { 2 } } \} } \\ & { } & { \leqslant \frac { 1 } { 4 } \langle \mu ^ { \prime } - \mu , s \rangle ^ { 2 } + \frac { 6 4 \eta ^ { 2 } } { m ^ { 2 } } . } \end{array}
$$

Substituting into (2) leaves $( 1 / 4 + 4 \eta ) \langle \mu ^ { \prime } - \mu , s \rangle ^ { 2 }$ on the right. For suficiently small � this term is absorbed, while $6 4 \eta ^ { 2 } / m ^ { 2 } = O ( \eta ^ { 2 } \log ( 1 / \eta ) / m )$ . Hence

$$
\mathcal { A } _ { \eta } \left. _ { 6 } \right. \left\{ \langle \mu ^ { \prime } - \mu , s \rangle ^ { 2 } \leqslant O \left( \frac { \eta ^ { 2 } \log ( 1 / \eta ) } { m } \right) \right\} .
$$

Finally, pseudoexpectation Cauchy–Schwarz and total-variation duality turn this directional certificate into $\Vert \hat { \mu } - \mu \Vert _ { \mathrm { T V } } \leqslant O ( \eta \sqrt { \log ( 1 / \eta ) / m } )$ . Here � is chosen, only in the analysis, to indicate the coordinates on which $\hat { \mu } - \mu$ is positive.

From the robust certificate to a private score. Clean data admit a polynomial-size certificate, and every certificate with a small discard budget is accurate. For privacy, we retain these two facts and allow the discard budget $T$ to vary. The score of a candidate � is the smallest � for which $\mathcal { A } _ { \eta } ,$ with the count constraint replaced by $\begin{array} { r } { \sum _ { i } w _ { i } \geqslant n - T . } \end{array}$ , admits a degree-6 pseudoexpectation whose pseudo-mean lies within $\ell _ { 1 } { \mathrm { - d i s t a n c e } }$ � of $\nu .$ Thus a low score says that the data admit a covariance-aware certificate near � after only a small number of deletions.

The score satisfies the conditions of the robustness-to-privacy reduction: changing one batch costs one additional discard, so sensitivity is one; combining certificates gives quasi-convexity; and the polynomial-size semidefinite formulation gives eficient computation. One further ingredient is needed: the low-score sets must have controlled volume.

$\ell _ { 1 }$ geometry of low-score sets. To compare the volumes of low-score sets, we use an inner and an outer ball. An exact clean certificate certifies the full $\ell _ { 1 } { \mathrm { - } } \mathrm { b a l l }$ of radius � around its mean. For a suficiently small absolute constant $\eta ^ { * }$ , robust soundness places all points of score at most $\eta ^ { * } n$ inside the concentric ball of radius $\alpha + O ( 1 / { \sqrt { m } } )$ (Lemma 4.12). Since the volume of an $\ell _ { 1 } { \mathrm { - } } \mathrm { b a l l }$ scales as the �th power of its radius, these two balls bound the ratio used by the pure-DP reduction of Hopkins, Kamath, Majid, and Narayanan [HKMN23]. Part 3 of Lemma 4.12 gives a sharper radius bound at each score level.

Approximate DP requires the analogous comparison at score levels $0 . 8 \eta ^ { * } n$ and $\eta ^ { * } n$ for an arbitrary dataset, where there may be no true mean to use as a center. We obtain the needed outer ball by adapting the one-dimensional obstruction argument of [HKMN23] to sign directions. We use signs because of exact $\ell _ { 1 }$ duality:

$$
\| x \| _ { 1 } = \max _ { \sigma \in \{ \pm 1 \} ^ { k } } \langle x , \sigma \rangle .
$$

If the pseudo-means of two certificates with discard budget at most $\eta ^ { * } n$ were farther than $O ( 1 / \sqrt { m } )$ in $\ell _ { 1 } ,$ some sign vector would separate them by more than $O ( 1 / \sqrt { m } )$ , contradicting their directional variance bounds. This gives the required $\ell _ { 1 } { \mathrm { - d i a m e t e r } }$ bound directly, without an $\ell _ { 2 } \mathrm { - t o - } \ell _ { 1 }$ conversion and its extra $\sqrt { k }$ loss. Adding the �-radius candidate neighborhoods gives an intermediate logvolume ratio of $O ( k \log ( 1 + C / ( \alpha { \sqrt { m } } ) ) )$ . Keeping the inner radius close to $\alpha ,$ as in Lemma 4.13, is needed when $\alpha \sqrt { m }$ is large. The clean-data certificate still requires $n = { \widetilde \Omega } ( k )$ in this regime, which accounts for the extra user term in the approximate-DP theorem. We prove the volume comparison in Corollary 4.15, and give the diameter argument in Section A.

## 1.3 Related Work

Private learning discrete distributions from batches was introduced by Liu et al. $[ \mathrm { L S Y ^ { + } 2 0 } ]$ , and was subsequently studied by others [EMN22, GKK<sup>+</sup>23b]. This problem also fits into the broader literature on user-level privacy [LSA<sup>+</sup>21], which studies adjacent tasks including mean estimation [GDD22, EMN22, CFMT22, GRST24, ZLS<sup>+</sup>24, AKM<sup>+</sup>25], PAC learning [GKM21], and convex optimization [GKK<sup>+</sup>23a, BS23, AL24, GKLM26]. There is likewise an extensive literature on private estimation in the item-level setting, corresponding to $m = 1$ ; see [KU20] for a partial survey. In that setting, estimating a discrete distribution over [�] under item-level $( \varepsilon , 0 )$ -diferential privacy is straightforward: applying the Laplace mechanism to the empirical distribution achieves the optimal sample complexity [DHS15]. This estimator, followed by $\ell _ { 1 }$ projection onto the simplex, tolerates replacement of an �-fraction of the items: with constant probability its total-variation error is $O ( \eta + \sqrt { k / n } + k / ( n \varepsilon ) )$ . Indeed, the replacements change the empirical distribution by at most � in total variation, and projection increases its distance from the true distribution by at most a factor of two.

Qiao and Valiant [QV18] introduced the batch-robust setting, which was subsequently studied by several groups [JO20b, CLM20a], as well as in related settings including structured distributions [JO20a, JO21, CLM20b], linear regression [DJKS23, JSK<sup>+</sup>24, DKK<sup>+</sup>25], and local diferential privacy [CS23, ALS23]. Our work, together with many of these results, also fits into the broader line of algorithmic robust statistics, which began with work on robust mean estimation $[ \mathrm { D K K ^ { + } } 1 6 ,$ LRV16] and has since expanded substantially; see [DK23] for a textbook treatment. As in the private setting, the interesting algorithmic issues for learning discrete distributions arise because of the batch structure: when � = 1, the information-theoretic optimum is already achieved by the contaminated empirical distribution.

Our private algorithm uses the robustness-to-privacy framework introduced by Hopkins, Kamath, Majid, and Narayanan [HKMN23] and independently by Asi, Ullman, and Zakynthinou [AUZ23], which converts robust estimators into private ones. Eficient versions of this framework have already yielded near-optimal private and robust algorithms for Gaussian estimation [HKMN23], parameter estimation in random graphs [CDHS24], and linear regression [ABMT25]. Related sumof-squares exponential mechanisms also give private algorithms for graphon estimation $[ \mathrm { C D d } ^ { + } 2 4 ]$ Robustness-to-privacy methods also underlie private Bayesian estimation [CDMM26] and improved sample bounds for privately learning Gaussian mixtures [AMN24]. To our knowledge, this is the first polynomial-time application of this framework that converts batch-level robustness into user-level privacy for learning discrete distributions. Georgiev and Hopkins [GH22] prove a partial converse, showing settings in which privacy automatically yields robustness.

Starting from the original work of Dwork and Lei [DL09], there has been substantial work on connections between robustness and privacy [BKSW19, KSU20, LKO22, BGS<sup>+</sup>21, RC21, KMV22, HKM22, $\mathrm { A K T ^ { + } } 2 3 ]$ . Some of these papers give alternative general methods for converting robust estimators into private ones [LKO22, KMV22]. Other work from the statistics community studies these connections from a diferent perspective [AM20, AM21, RJC25, SM22], including in the local model [LBY23]. A recent line of work studies estimators that are simultaneously private and robust for high-dimensional problems [BKSW19, GKMN21, LKKO21, EMN22, AL22, KMV22, TCK<sup>+</sup>22, LKO22, $\mathrm { L J K } ^ { + } 2 3 ]$ . See the survey of Kamath [Kam24] for a broader discussion of the robustness–privacy interface.

Paper roadmap. Section 2 shows how subset control produces a diagonal majorizer. Section 3 uses the majorizer in both directions—first for clean-data feasibility and then for degree-6 SoS identifiability. Section 4 turns this certificate into a score function and proves the categorical private theorems. Section 5 proves the SQ lower bound for robust learning from discrete batches. The appendices give the approximate-DP $\ell _ { 1 }$ obstruction and the matching pure-DP lower bound.

## 2 Preliminaries

Problem 2.1 (Robust Learning from Discrete Batches). Let $\mu \in \Delta ^ { k - 1 }$ be an unknown discrete distribution over [�]. Let $m \in \mathbb { N }$ be the batch size, and $\eta$ be the corruptionfraction. Let $\{ B _ { i } ^ { * } \} _ { i \in [ n ] }$ be � batches of samples, where each clean batch

$$
B _ { i } ^ { * } = \{ U _ { i , 1 } ^ { * } , \ldots , U _ { i , m } ^ { * } \}
$$

and each $U _ { i , j } ^ { * } \sim \mu$ is drawn independently from $\mu .$ An �-fraction of these batches are then adversarially corrupted to form the observed batches $\{ B _ { i } \} _ { i \in [ n ] } ,$ , meaning that there exists a set $S \subseteq [ n ]$ with $| S | \leqslant \eta n$ such that $B _ { i } = B _ { i } ^ { * }$ for every $i \not \in S$ . We seek an algorithm that, given the observed batches $\{ B _ { i } \} _ { i \in [ n ] }$ , outputs a distribution $\hat { \mu } \in \Delta ^ { k - 1 }$ such that with high probability,

$$
\| { \hat { \boldsymbol { \mu } } } - { \boldsymbol { \mu } } \| _ { \mathrm { T V } } \leqslant { \boldsymbol { \alpha } } ,
$$

where $\alpha \geqslant \eta \sqrt { \log ( 1 / \eta ) / m }$

## 2.1 Data Model and Notation

We use calligraphic letters such as $\mathcal { B }$ and � for datasets. A star marks a latent clean batch or average, and a prime marks a repaired point or candidate mean.

For a batch $B _ { i } = \{ U _ { i , 1 } , \ldots , U _ { i , m } \}$ of symbols in $[ k ] ,$ , define its empirical average vector by

$$
y _ { i } : = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } e _ { U _ { i , j } } \in \Delta ^ { k - 1 } ,
$$

where $e _ { 1 } , \ldots , e _ { k }$ are the standard basis vectors in $\mathbb { R } ^ { k }$ . Likewise, each clean batch $B _ { i } ^ { * }$ induces a clean batch-average vector

$$
y _ { i } ^ { * } : = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } e _ { U _ { i , j } ^ { * } } \in \Delta ^ { k - 1 } .
$$

We write

$$
\begin{array} { r } { \begin{array} { r } { \mathcal { B } ^ { * } = \{ B _ { i } ^ { * } \} _ { i \in [ n ] } , \qquad \mathcal { B } = \{ B _ { i } \} _ { i \in [ n ] } , \qquad \mathcal { Y } ^ { * } = \{ y _ { i } ^ { * } \} _ { i \in [ n ] } , \qquad \mathcal { Y } = \{ y _ { i } \} _ { i \in [ n ] } . } \end{array} } \end{array}
$$

From now on, we work with the averaged vectors $y ^ { * }$ and $y .$ . The robust program uses repaired points $y _ { i } ^ { \prime }$ and their mean $\mu ^ { \prime } ; \hat { \mu }$ denotes the output estimate, and � denotes a candidate for the private score.

## 2.2 Probability Simplex

We will repeatedly make use of the probability simplex of distributions over [�].

Definition 2.1 (Probability Simplex $\Delta ^ { k - 1 } )$ . Let $\Delta ^ { k - 1 }$ denote

$$
\begin{array} { r } { \Delta ^ { k - 1 } : = \{ x  { |  \mathbf { 1 } , x  } = 1 , x \in \mathbb { R } _ { \geqslant 0 } ^ { k } \} . } \end{array}
$$

Remark 2.2 (Total Variation Convention). Whenever $\boldsymbol { x } \in \mathbb { R } ^ { k }$ satisfies $\langle \mathbf { 1 } , x \rangle = 0$ , we write

$$
\| x \| _ { \mathrm { T V } } : = { \frac { 1 } { 2 } } \| x \| _ { 1 } = \operatorname* { m a x } _ { s \in \{ 0 , 1 \} ^ { k } } \langle x , s \rangle .
$$

In particular, for probability vectors $\mu , \nu \in \Delta ^ { k - 1 } , \| \mu - \nu \| _ { \mathrm { T V } }$ denotes the usual total variation distance.

Fact 2.3 (Diagonal minus rank-one is PSD on the simplex). Let $\boldsymbol { x } \in \mathbb { R } _ { \geqslant 0 } ^ { k }$ with $\textstyle s : = \sum _ { i = 1 } ^ { k } x _ { i } \leqslant 1$ (in particular, for $x \in \Delta ^ { k - 1 }$ one has $s = 1 )$ . Define

$$
M ( x ) : = \mathrm { d i a g } ( x ) - x x ^ { \top } \in \mathbb { R } ^ { k \times k } .
$$

Then $M ( x ) \succeq 0$ . Moreover, on supp(�),

$$
M ( x ) \geq ( 1 - s ) \mathrm { d i a g } ( x ) .
$$

Proof. Let $D : = \mathrm { d i a g } ( x )$ and $y : = D ^ { 1 / 2 } \mathbf { 1 }$ . Then $M ( x )$ factors as

$$
M ( x ) \ : = \ : D ^ { 1 / 2 } \big ( I - y y ^ { \top } \big ) D ^ { 1 / 2 } , \qquad \| y \| ^ { 2 } = { \bf 1 } ^ { \top } D { \bf 1 } = s \leqslant 1 .
$$

Here $I - y y ^ { \top }$ has eigenvalues 1 (with multiplicity $k - 1$ on $y ^ { \perp } )$ and $1 - \| y \| ^ { 2 } = 1 - s \geqslant 0$ (in the direction of �). Hence $I - y y ^ { \top } \succeq 0 ,$ , and conjugation by $D ^ { 1 / 2 }$ preserves positive semidefiniteness, yielding $M ( x ) \succeq 0$ . To prove $M ( x ) \geq ( 1 - s ) \mathrm { d i a g } ( x )$ , observe that

$$
\begin{array} { r } { u ^ { \top } M ( x ) u \ = \ \| D ^ { 1 / 2 } u \| ^ { 2 } - \langle D ^ { 1 / 2 } u , y \rangle ^ { 2 } \ \geqslant \ ( 1 - \| y \| ^ { 2 } ) \| D ^ { 1 / 2 } u \| ^ { 2 } \ = \ ( 1 - s ) u ^ { \top } D u , } \end{array}
$$

by Cauchy–Schwarz. In the special case $s = 1$ (the standard simplex), $I - y y ^ { \top }$ has a simple zero eigenvalue, so $M ( x )$ has rank at most $k - 1$ with 1 in its nullspace; equivalently

$$
u ^ { \top } M ( x ) u ~ = ~ \frac { 1 } { 2 } \sum _ { i , j } x _ { i } x _ { j } ( u _ { i } - u _ { j } ) ^ { 2 } ~ \geqslant ~ 0 .
$$

Fact 2.4 (Mean and Covariance of a Clean Batch Average). Let $\mu \in \Delta ^ { k - 1 }$ be a discrete distribution over [�], let $m \in \mathbb { N } _ { i }$ , let

$$
U _ { 1 } , \ldots , U _ { m } \stackrel { \mathrm { i i d } } { \sim } \mu , \qquad Y : = \frac { 1 } { m } \sum _ { \ell = 1 } ^ { m } e _ { U _ { \ell } } \in \Delta ^ { k - 1 } ,
$$

and write

$$
\Sigma ( \mu ) : = \frac { 1 } { m } \big ( \mathrm { d i a g } ( \mu ) - \mu \mu ^ { \intercal } \big ) .
$$

Then

$$
\begin{array} { r l } & { \mathbb { E } [ Y ] = \mu , } \\ & { \mathbb { E } [ Y Y ^ { \mathsf { T } } ] - \mathbb { E } [ Y ] \mathbb { E } [ Y ] ^ { \mathsf { T } } = \Sigma ( \mu ) . } \end{array}
$$

Proof. Linearity of expectation gives the mean. For the second moment, we have

$$
\begin{array} { r l } {  { \mathbb { E } [ Y Y ^ { \top } ] = \mathbb { E } [ \frac { 1 } { m ^ { 2 } } \sum _ { \ell , t \in [ m ] } e _ { U _ { \ell } } e _ { U _ { t } } \tau ] ^ { \tau } } } \\ & { = \frac { 1 } { m ^ { 2 } } \Bigg ( \sum _ { \ell \in [ m ] } \mathbb { E } [ e _ { U _ { \ell } } e _ { U _ { \ell } } \tau ] + \sum _ { \ell \neq t } \mathbb { E } [ e _ { U _ { \ell } } ] \mathbb { E } [ e _ { U _ { t } } ] ^ { \top } \Bigg ) } \\ & { = \frac { 1 } { m ^ { 2 } } \big ( m \mathrm { d i a g } ( \mu ) + m ( m - 1 ) \mu \mu ^ { \top } \big ) . } \end{array}
$$

Thus,

$$
\begin{array} { l } { { \displaystyle \mathbb { E } [ Y Y ^ { \mathsf { T } } ] - \mathbb { E } [ Y ] \mathbb { E } [ Y ] ^ { \mathsf { T } } = \frac { 1 } { m ^ { 2 } } \big ( m { \mathrm { ~ d i a g } } ( \mu ) + m ( m - 1 ) \mu \mu ^ { \mathsf { T } } \big ) - \mu \mu ^ { \mathsf { T } } } \ } \\ { \displaystyle \quad \quad = \frac { 1 } { m } \big ( { \mathrm { d i a g } } ( \mu ) - \mu \mu ^ { \mathsf { T } } \big ) . } \end{array}
$$

We will also repeatedly use the variance of a clean batch average along a direction $s \in \{ 0 , 1 \} ^ { k }$ whose formula is immediate from Fact 2.4.

Definition 2.5 (Subset-Direction Mean and Variance). Let $\mu \in \Delta ^ { k - 1 }$ , and let � be a �-dimensional vector in $\{ 0 , 1 \} ^ { k }$ . We define the mean and variance of a clean batch average along direction � by

$$
\begin{array} { c } { { \mu ( s ) : = \langle s , \mu \rangle , } } \\ { { V ( \mu , s ) : = \displaystyle \frac { \langle s , \mu \rangle ( 1 - \langle s , \mu \rangle ) } { m } . } } \end{array}
$$

## 2.3 Concentration Bounds

We use a version of the clean-sample condition in [JO20b] with an explicit failure probability.

Fact 2.6 (cf. Lemma 3 in [JO20b]). Let $\mu \in \Delta ^ { k - 1 } , \eta , \beta \in ( 0 , 1 / 2 )$ , and $m \in \mathbb { N } .$ Let

$$
B _ { i } ^ { * } = ( U _ { i , 1 } ^ { * } , \ldots , U _ { i , m } ^ { * } ) , \qquad U _ { i , 1 } ^ { * } , \ldots , U _ { i , m } ^ { * } \overset { \mathrm { i i d } } { \sim } \mu , \qquad y _ { i } ^ { * } : = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } e _ { U _ { i , j } ^ { * } }
$$

for each $i \in [ n ]$ . Then as long as

$$
n \geqslant \Omega \left( \frac { k + \log ( 1 / \beta ) } { \eta ^ { 2 } \log ( 1 / \eta ) } \right) ,
$$

with probability $1 - \beta ,$ , for all subsets of indices $T \subseteq [ n ] w i t h | T | \geqslant ( 1 - \eta ) n$ , and all vectors $s \in \{ 0 , 1 \} ^ { k }$ , we have

$$
\left| \frac { 1 } { | T | } \sum _ { i \in T } \langle y _ { i } ^ { * } , s \rangle - \langle \mu , s \rangle \right| \leqslant O \left( \eta \sqrt { \log ( 1 / \eta ) / m } \right) ,
$$

and

$$
\left| \frac { 1 } { | T | } \sum _ { i \in T } \langle s , y _ { i } ^ { * } - \mu \rangle ^ { 2 } - V ( \mu , s ) \right| \leqslant O \left( \eta \log ( 1 / \eta ) / m \right) .
$$

Proof. Write $\xi _ { i } ( s ) = \langle y _ { i } ^ { * } - \mu , s \rangle$ . For each fixed �, Hoefding’s inequality gives $\mathbb { P } [ | \xi _ { i } ( s ) | > t ] \ \leqslant$ $2 \exp ( - 2 m t ^ { 2 } )$ . Thus $\xi _ { i } ( \dot { s } ) ^ { 2 }$ has subexponential norm $O ( 1 / m )$ . Applying the corresponding mean and second-moment concentration bounds and taking a union bound over the $2 ^ { k }$ choices of � gives, with probability at least $1 - \beta / 2$

$$
\begin{array} { r l } & { ~ \bigg | \underset { i } { \mathbb { E } } \xi _ { i } ( s ) \bigg | \leqslant O \left( \sqrt { \frac { k + \log ( 2 / \beta ) } { n m } } \right) , } \\ & { \bigg | \underset { i } { \mathbb { E } } \xi _ { i } ( s ) ^ { 2 } - V ( \mu , s ) \bigg | \leqslant \frac { O ( 1 ) } { m } \left( \sqrt { \frac { k + \log ( 2 / \beta ) } { n } } + \frac { k + \log ( 2 / \beta ) } { n } \right) } \end{array}
$$

simultaneously for all �. The assumed sample size ensures �� $\geqslant 1$ . For a fixed set � of size ⌈��⌉, the exponential-moment bound gives $\begin{array} { r } { \sum _ { i \in U } \bar { \xi } _ { i } ( s ) ^ { 2 } \leqslant O ( ( \lceil \eta n \rceil + t ) / m ) } \end{array}$ except with probability $e ^ { - t }$ A union bound over � and the $\binom { \breve { n } } { \lceil \eta n \rceil } \leqslant ( e n / \lceil \eta n \rceil ) ^ { \lceil \eta n \rceil }$ such sets therefore gives, with probability at least $1 - \beta / 2$

$$
\operatorname* { m a x } _ { | U | \leqslant \eta n } \frac { 1 } { n } \sum _ { i \in U } \xi _ { i } ( s ) ^ { 2 } \leqslant \frac { O ( 1 ) } { m } \left( \eta \log ( 1 / \eta ) + \frac { k + \log ( 2 / \beta ) } { n } \right) \leqslant O \left( \frac { \eta \log ( 1 / \eta ) } { m } \right)
$$

for every �. For $T = [ n ] \backslash U ,$ , Cauchy–Schwarz bounds $\begin{array} { r } { | n ^ { - 1 } \sum _ { i \in U } \xi _ { i } ( s ) | } \end{array}$ by $\textstyle { \sqrt { ( | U | / n ) ( n ^ { - 1 } \sum _ { i \in U } \xi _ { i } ( s ) ^ { 2 } ) } }$ Subtracting the contribution of � from the full mean and second moment, and dividing by $| T | / n \geqslant 1 / 2 ,$ , proves the mean and second-moment bounds for �. For the second moment, the normalization contributes at most $2 \eta V ( \mu , s ) \leqslant \eta / ( 2 m )$ □

## 2.4 Sum-of-Squares

Fact 2.7 (Triangle inequality). Let � and � be indeterminates, then

$$
\begin{array} { r } { \frac { | a , b } { 2 } \left\{ ( a + b ) ^ { 2 } \leqslant 2 a ^ { 2 } + 2 b ^ { 2 } \right\} . } \end{array}
$$

Fact 2.8 (Young’s inequality). Let � and � be indeterminates and let $\lambda > 0$ be a fixed scalar. Then

$$
\left| { \frac { a , b } { 2 } } \right. \left\{ a b \leqslant \lambda a ^ { 2 } + { \frac { b ^ { 2 } } { 4 \lambda } } \right\} .
$$

Proof. We can write

$$
\begin{array} { r } { | \frac { a , b } { 2 } \{ \displaystyle \lambda a ^ { 2 } + \frac { b ^ { 2 } } { 4 \lambda } - a b = ( \sqrt \lambda a - \frac { b } { 2 \sqrt \lambda } ) ^ { 2 } \} , } \\ { \geqslant 0 \qquad \geqslant 0 } \end{array}
$$

as desired.

Lemma 2.9 (SoS Selector Lemma [ABMT25]). Let $B , a _ { 1 } , \ldots , a _ { n } \in \mathbb { R }$ and $k \in \mathbb { N } .$ . Suppose $\textstyle \sum _ { i \in T } a _ { i } \leqslant B$ for every $T \subseteq [ n ]$ with $\left. T \right. \leqslant k$ . Then

$$
\left\{ \forall i \in [ n ] : 0 \leqslant b _ { i } \leqslant 1 , \sum _ { i = 1 } ^ { n } b _ { i } = k \right\} \left| { \frac { b _ { 1 } , \ldots , b _ { n } } { 2 } } \left\{ \sum _ { i = 1 } ^ { n } b _ { i } a _ { i } \leqslant B \right\} . \right.
$$

Additionally, if the $a _ { i } ^ { \prime } s$ are non-negative, the constraint $\begin{array} { r } { \sum _ { i \in [ n ] } b _ { i } = k } \end{array}$ may be replaced by $\begin{array} { r } { \sum _ { i \in [ n ] } b _ { i } \leqslant k , } \end{array}$ retaining $0 \leqslant b _ { i } \leqslant 1$ for every �. More generally, let $t \in [ 0 , n ]$ be real, let $a _ { i } \geqslant 0 .$ , and suppose $\textstyle \sum _ { i \in T } a _ { i } \leqslant B$ whenever $\left| T \right| \leqslant t$ . Then

$$
\left\{ \forall i : b _ { i } ^ { 2 } = b _ { i } , \sum _ { i } b _ { i } \leqslant t \right\} \Big | _ { 4 } \left\{ \sum _ { i } a _ { i } b _ { i } \leqslant 2 B \right\} .
$$

Proof of the real-budget extension. For $t \geqslant 1 ,$ , order the coeficients so that $a _ { 1 } \geqslant \cdots \geqslant a _ { n }$ . By the Boolean constraints, $0 \leqslant b _ { i } \leqslant 1$ , and hence

$$
\begin{array} { r l r } {  { \sum _ { i } a _ { i } b _ { i } \leqslant a _ { \lfloor t \rfloor } \sum _ { i } b _ { i } + \sum _ { i \leqslant \lfloor t \rfloor } ( a _ { i } - a _ { \lfloor t \rfloor } ) } } \\ & { } & { \leqslant \sum _ { i \leqslant \lfloor t \rfloor } a _ { i } + ( t - \lfloor t \rfloor ) a _ { \lfloor t \rfloor } } \\ & { } & { \leqslant \frac { t } { \lfloor t \rfloor } \sum _ { i \leqslant \lfloor t \rfloor } a _ { i } \leqslant 2 B . } \end{array}
$$

These are degree-2 SoS inequalities. If $t < 1$ , the Boolean identities give the degree-4 certificate

$$
( t - 1 ) b _ { i } = b _ { i } ^ { 2 } \left( t - \sum _ { j } b _ { j } \right) + \sum _ { j \neq i } ( b _ { i } b _ { j } ) ^ { 2 } \geqslant 0 .
$$

Thus $b _ { i } \leqslant 0$ , and $\begin{array} { r } { \sum _ { i } a _ { i } b _ { i } \leqslant 0 \leqslant 2 B . } \end{array}$

Definition 2.10 (Approximately Satisfying Linear Functionals). Let $n \in  { \mathbb { N } } , T \in [ 0 , n ]$ . Let $\mathcal { A } _ { T }$ be a system of polynomial constraints parameterized by � in indeterminates $w = ( w _ { 1 } , \dots , w _ { n } )$ and � (of arbitrary size) such that

$$
\mathcal { A } _ { T } = \left\{ \forall i \in [ m _ { \geqslant } ] : q _ { i } ( z , w ) \geqslant 0 \right\} \cup \left\{ \forall j \in [ m _ { \mathrm { e q } } ] : r _ { j } ( z , w ) = 0 \right\} \cup \left\{ \sum _ { i \in [ n ] } w _ { i } \geqslant n - T \right\} .
$$

Let ℛ denote the operator that maps a polynomial to a vector representing its coeficients. Let $\tau \geqslant 0$ . We say that a linear functional ℒ �-approximately satisfies $\mathcal { A } _ { T }$ at degree � if

1. $\mathcal { L } 1 = 1$

2. For all degree at most $D / 2$ polynomials ${ \boldsymbol { p } } ,$ such that $\| \mathcal { R } ( p ) \| _ { 2 } \leqslant 1$ , we have that $\mathcal { L } p ^ { 2 } \geqslant - \tau T$

3. For all $i \in [ m _ { \geqslant } ]$ and polynomials � such that deg $( p ^ { 2 } q _ { i } ) \leqslant D$ , and $\| \mathcal { R } ( p ) \| _ { 2 } \leqslant 1$ , we have

$$
\begin{array} { r } { \mathcal { L } p ^ { 2 } q _ { i } \geqslant - \tau T . } \end{array}
$$

4. For all $j \in [ m _ { \mathrm { e q } } ]$ and polynomials � such that $\deg ( p ^ { 2 } r _ { j } ) \leqslant D$ , and $\| \mathcal { R } ( p ) \| _ { 2 } \leqslant 1$ , we have

$$
\left| \mathcal { L } p ^ { 2 } r _ { j } \right| \leqslant \tau T .
$$

5. For all polynomials � such that $\begin{array} { r } { \mathrm { d e g } ( p ^ { 2 } ( \sum _ { i } w _ { i } - n + T ) ) \leqslant D } \end{array}$ , and $\| \mathcal { R } ( p ) \| _ { 2 } \leqslant 1$ , we have that $\begin{array} { r } { \mathcal { L } \left[ p ^ { 2 } ( \sum _ { i } w _ { i } - n + T ) \right] \geqslant - 5 \tau T n . } \end{array}$

## 2.5 Subset-Variance Control and the Diagonal-Majorizer SDP

We isolate the abstract bridge from subset-direction control of a covariance-gap matrix to a tracebounded diagonal majorizer. Let $\Delta \in \mathbb { R } ^ { k \times k }$ be symmetric. We consider the Boolean quadratic $s ^ { \top } \Delta s$ over subset directions $s \in \{ 0 , 1 \} ^ { k }$

Lemma 2.11 (Subset-to-sign reduction under a nullspace condition). Let $\Delta \in \mathbb { R } ^ { k \times k }$ be symmetric and satisfy $\Delta \mathbf { 1 } = 0$ . Then for every $s \in \{ 0 , 1 \} ^ { k }$ and $\boldsymbol { \sigma } = 2 s - \mathbf { 1 }$ 1

$$
\left| \sigma ^ { \top } \Delta \sigma \right| = 4 \left| s ^ { \top } \Delta s \right| .
$$

Proof.

$$
\sigma ^ { \top } \Delta \sigma = ( 2 s - \mathbf { 1 } ) ^ { \top } \Delta ( 2 s - \mathbf { 1 } ) = 4 s ^ { \top } \Delta s
$$

because $\Delta \mathbf { 1 } = 0$

□

Definition 2.12 (Elliptope). Define

$$
\mathscr { E } _ { k } : = \left\{ X \in \mathbb { R } ^ { k \times k } \ : \middle | \ : X \succeq 0 , \ : \mathrm { d i a g } ( X ) = { \bf 1 } \right\} .
$$

Each rank-one matrix $\sigma \sigma ^ { \top }$ with $\sigma \in \left\{ \pm 1 \right\} ^ { k }$ belongs to $\mathcal { E } _ { k }$

Theorem 2.13 (Symmetric Grothendieck inequality, real �-case of Friedland–Lim [FL20]). Let $A = ( a _ { i j } ) \in \mathbb { R } ^ { k \times k }$ be symmetric, and let $d \geqslant k$ . Then

$$
\operatorname* { m a x } _ { \substack { x _ { 1 } , \ldots , x _ { k } \in \mathbb { R } ^ { d } } } \left| \sum _ { i , j = 1 } ^ { k } a _ { i j } \langle x _ { i } , x _ { j } \rangle \right| \leqslant \sinh ( \pi / 2 ) \operatorname* { m a x } _ { \sigma \in \{ \pm 1 \} ^ { k } } \left| \sum _ { i , j = 1 } ^ { k } a _ { i j } \sigma _ { i } \sigma _ { j } \right| .
$$

Equivalently, this is the real symmetric �-seminorm inequality from Friedland–Lim, Theorem 4.3, equation (55), together with the bound on the constant from equation (57).

Corollary 2.14 (Correlation-matrix form). Let $A \in \mathbb { R } ^ { k \times k }$ be symmetric. Then

$$
\operatorname* { m a x } _ { X \approx 0 , \ \mathrm { d i a g } ( X ) = 1 } \bigl | \langle A , X \rangle \bigr | \leqslant \sinh ( \pi / 2 ) \operatorname* { m a x } _ { \sigma \in \{ \pm 1 \} ^ { k } } \bigl | \sigma ^ { \top } A \sigma \bigr | .
$$

Proof. Every feasible � is a correlation matrix, hence a Gram matrix: there exist unit vectors $x _ { 1 } , \ldots , x _ { k } \in \mathbb { R } ^ { r }$ for some $r \leqslant k$ such that

$$
X _ { i j } = \left. x _ { i } , x _ { j } \right. \qquad { \mathrm { f o r ~ a l l ~ } } i , j \in [ k ] .
$$

Therefore

$$
\langle A , X \rangle = \sum _ { i , j = 1 } ^ { k } a _ { i j } X _ { i j } = \sum _ { i , j = 1 } ^ { k } a _ { i j } \langle x _ { i } , x _ { j } \rangle ,
$$

and the claim follows from Theorem 2.13.

We use the real symmetric indefinite absolute-value form from Friedland–Lim because Δ is not assumed positive semidefinite.

Fact 2.15 (Semidefinite duality with equality constraints). Let $C , A _ { 1 } , \dots , A _ { m } \in \mathbb { R } _ { \mathrm { s y m } } ^ { d \times d }$ and let $b \in \mathbb { R } ^ { m }$ Consider the primal–dual pair

(P)

$$
\operatorname* { m a x } _ { X \neq 0 } \left. C , X \right. \qquad { \mathrm { s u c h ~ t h a t } } \left. A _ { t } , X \right. = b _ { t } { \mathrm { ~ f o r ~ a l l ~ } } t \in [ m ] ,\tag{D}
$$

$$
\operatorname* { m i n } _ { y \in \mathbb { R } ^ { m } } \sum _ { t = 1 } ^ { m } b _ { t } y _ { t } \qquad { \mathrm { s u c h ~ t h a t ~ } } \sum _ { t = 1 } ^ { m } y _ { t } A _ { t } - C \succcurlyeq 0 .
$$

If both programs are strictly feasible, then strong duality holds and the two optimal values are equal.

Corollary 2.16 (Diagonal-majorizer SDP duality). Let $\Delta \in \mathbb { R } ^ { k \times k }$ be symmetric. Then

$$
\operatorname* { m a x } _ { X \approx 0 , \ \mathrm { d i a g } ( X ) = 1 } \langle \Delta , X \rangle = \operatorname* { m i n } _ { \lambda \in \mathbb { R } ^ { k } } \left\{ \sum _ { j \in [ k ] } \lambda _ { j } \ \left| \ \mathrm { D i a g } ( \lambda ) - \Delta \approx 0 \right. \right\} .
$$

Moreover, both programs are strictly feasible.

Proof. Apply Fact 2.15 with $A _ { j } = e _ { j } e _ { j } ^ { \top } , b _ { j } = 1$ , and $C = \Delta$ . Strict primal feasibility is witnessed by $X = I _ { k } .$ , and strict dual feasibility is witnessed by $\lambda = C \mathbf { 1 }$ for any suficiently large scalar �. □

Proposition 2.17 (Subset Control Implies Diagonal-Majorizer Feasibility). Let $\Delta \in \mathbb { R } ^ { k \times k }$ be symmetric. Assume that

$$
\Delta { \mathbf { 1 } } = 0 , \qquad \quad { \underset { s \in \{ 0 , 1 \} ^ { k } } { \operatorname* { s u p } } } \left| s ^ { \top } \Delta s \right| \leqslant \rho .
$$

Then there exists $\boldsymbol { \lambda } \in \mathbb { R } ^ { k }$ such that

$$
\mathrm { D i a g } ( \lambda ) - \Delta \succcurlyeq 0 , \qquad \sum _ { j \in [ k ] } \lambda _ { j } \leqslant O ( \rho ) .
$$

Proof. By Lemma 2.11,

$$
\operatorname* { m a x } _ { \sigma \in \left\{ \pm 1 \right\} ^ { k } } \left| \sigma ^ { \top } \Delta \sigma \right| \leqslant 4 \rho .
$$

By Corollary 2.14,

$$
\operatorname* { m a x } _ { X \neq 0 , \dim \& ( X ) = 1 } \langle \Delta , X \rangle \leqslant \operatorname* { m a x } _ { X \neq 0 , \dim \& ( X ) = 1 } | \langle \Delta , X \rangle | \leqslant \sinh ( \pi / 2 ) \operatorname* { m a x } _ { \sigma \in \{ \pm 1 \} ^ { k } } \left| \sigma ^ { \top } \Delta \sigma \right| \leqslant O ( \rho ) .
$$

Applying Corollary 2.16 yields the claim.

## 2.6 Robustness to Privacy

We apply the following theorem in order to transform robust algorithms into private ones.

Theorem 2.18 ([HKMN23]). Let $\varepsilon > 0 , \beta \in ( 0 , 1 )$ , and $0 < \eta , r < 1 < R$ be fixed parameters. Suppose we have a scorefunction $S ( \theta , y ) \in [ 0 , n ]$ that takes as input a dataset $\pmb { y } = \{ y _ { 1 } , \dots , y _ { n } \}$ and a parameter $\theta \in \Theta \subset \mathbb { B } ( R ) ^ { d }$ (where Θ is convex and contained in a ball of radius �), with thefollowing properties:

• (Bounded Sensitivity) For any two adjacent datasets $y , y \prime$ and any $\theta \in \Theta , | S ( \theta , y ) - S ( \theta , y \prime ) | \leqslant 1$

• (Quasi-Convexity) For any fixed dataset � , any $\theta , \theta ^ { \prime } \in \Theta$ , and any $0 \leqslant \lambda \leqslant 1$ , we have that $S ( \lambda \theta + ( 1 - \lambda ) \theta ^ { \prime } , y ) \leqslant \operatorname* { m a x } ( S ( \theta , y ) , S ( \theta ^ { \prime } , y ) )$

• (Eficiently Computable) For any given $\theta \in \Theta$ and dataset �, we can compute $\textstyle S ( \theta , y )$ up to error � in $\begin{array} { r } { \mathrm { { \dot { ~ } p o l y } } ( n , d , \log \frac { R } { r } , \log \gamma ^ { - 1 } ) } \end{array}$ time for any $\gamma > 0$

• (Robust algorithm finds low-scoring point) For a given dataset � , let $T = \mathrm { m i n } _ { \theta _ { 0 } \in \Theta } S ( \theta _ { 0 } , y )$ . Then, we can find some point � such that for all $\theta ^ { \prime }$ within distance � of $\theta , S ( \theta ^ { \prime } , y ) \leqslant T + 1$ , in time $\scriptstyle \operatorname { p o l y } ( n , d , \log { \frac { R } { r } } )$

• (Volume) For any given dataset � and $\eta ^ { \prime } \geqslant \eta ,$ , let $V _ { \eta ^ { \prime } } ( y )$ represent the �-dimensional volume ofpoints $\theta \in \Theta \subset \mathbb { R } ^ { d }$ with score at most �<sup>′</sup>�. (Note that $V _ { 1 } ( \boldsymbol { y } )$ is thefull volume of Θ).

Then, we have a pure �-DP algorithm � on datasets of size �, that runs in poly(�<sub>,</sub> �<sub>,</sub> log $\textstyle { \frac { R } { r } } )$ time, with the following property. For any dataset �, if there exists � with $S ( \theta , y ) \leqslant \eta n$ and if

$$
n \geqslant \Omega \left( \operatorname* { m a x } _ { \eta ^ { \prime } : \eta \leqslant \eta ^ { \prime } \leqslant 1 } \frac { \log ( V _ { \eta ^ { \prime } } ( \mathcal { Y } ) / V _ { \eta } ( \mathcal { Y } ) ) + \log ( 1 / ( \beta \cdot \eta ) ) } { \varepsilon \cdot \eta ^ { \prime } } \right) ,
$$

then $\mathcal { A } ( \boldsymbol { y } )$ outputs some $\theta \in \Theta$ of score at most 2�� with probability $1 - \beta .$

For context we also record the corresponding approximate-DP reduction theorem. Relative to Theorem 2.18, the only extra input is the intermediate volume-ratio condition at levels 0<sub>.</sub>8� and $\eta ^ { * }$

Theorem 2.19 ([HKMN23]). Let $\varepsilon > 0 , \delta , \beta \in ( 0 , 1 ) , 0 < \eta < 0 . 1$ , and $0 < r < 1 < R$ befixed parameters. Suppose we have a score function $S ( \theta , y ) \in [ 0 , \infty )$ that takes as input a dataset $\pmb { y } = \{ y _ { 1 } , \dots , y _ { n } \}$ and a parameter $\theta \in \Theta \subset \mathbb { R } ^ { d }$ (where Θ is convex and contained in a ball of radius $R ) ,$ satisfying the properties in theorem 2.18. In addition,fix some parameter $\eta ^ { * } \in [ 1 0 \eta , 1 ]$ . Suppose that

$$
n \geqslant \Omega \left( \frac { \log ( 1 / \delta ) + \log ( V _ { \eta ^ { * } } ( \pmb { \mathscr { Y } } ) / V _ { 0 . 8 \eta ^ { * } } ( \pmb { \mathscr { Y } } ) ) } { \varepsilon \cdot \eta ^ { * } } \right)\tag{3}
$$

for all � such that there exists � with $S ( \theta , y ) \leqslant 0 . 7 \eta ^ { * } n$ . Then, there exists an $( \varepsilon , \delta ) – D P$ algorithm ${ \mathcal { A } } ,$ that runs in poly $\begin{array} { r } { \mathbf { \nabla } ( n , d , \log ( \frac { R } { r } ) ) } \end{array}$ time, such that for any dataset � , if there exists � with $S ( \theta , y ) \leqslant \eta n$ and if

$$
n \geqslant \Omega \left( \operatorname* { m a x } _ { \eta ^ { \prime } : \eta \leqslant \eta ^ { \prime } \leqslant \eta ^ { * } } \frac { \log ( V _ { \eta ^ { \prime } } ( \mathcal { Y } ) / V _ { \eta } ( \mathcal { Y } ) ) + \log ( 1 / ( \beta \cdot \eta ) ) } { \varepsilon \cdot \eta ^ { \prime } } \right) ,\tag{4}
$$

then $\mathcal { A } ( \boldsymbol { y } )$ outputs some $\theta \in \Theta$ satisfying $S ( \theta , y ) \leqslant 2 r _ { i }$ � with probability at least $1 - \beta .$

Lemma 2.20 (A refined utility bound). Suppose $0 < \varepsilon \leqslant 1$ and the score-�� set contains a Euclidean ball of radius �. In the sample-size condition of Theorem 2.18 and in (4), the term $\log ( 1 / ( \beta \eta ) )$ can be replaced by $\log ( 2 / \beta )$ . All other hypotheses remain required, including the worst-case volume condition (3) for approximate DP.

Proof. We refine the tail calculation for the samplers of [HKMN23, full version, Lemmas 4.7 and 4.8]. These samplers assign weights proportional to $\exp ( - c \varepsilon S )$ up to constant multiplicative factors; the approximate-DP sampler also truncates at score $0 . 9 \eta ^ { * } n + O ( 1 )$ . Their grid-point counts agree with sublevel-set volumes up to constant factors, by [HKMN23, Fact 4.6]. The inner-ball assumption supplies this comparison at level �.

Each nonempty shell with score in $[ 2 ^ { j } \eta n , 2 ^ { j + 1 } \eta n ) , j \geqslant 1$ , has probability at most

$$
A \exp \left( - c \varepsilon n ( 2 ^ { j } - 1 ) \eta \right) \left\{ \begin{array} { l l } { \displaystyle \frac { V _ { \mathrm { m i n } \{ 2 ^ { j + 1 } \eta , 1 \} } ( \mathcal { Y } ) } { V _ { \eta } ( \mathcal { Y } ) } , } & { \mathrm { f o r ~ p u r e ~ D P } , } \\ { \displaystyle \frac { V _ { \mathrm { m i n } \{ 2 ^ { j + 1 } \eta , \eta ^ { * } \} } ( \mathcal { Y } ) } { V _ { \eta } ( \mathcal { Y } ) } , } & { \mathrm { f o r ~ a p p r o x i m a t e ~ D P } , } \end{array} \right.
$$

for absolute constants $A , c > 0$ . Additive �(1) score errors afect only �, since $\varepsilon \leqslant 1$ . If

$$
n \geqslant \left\{ \begin{array} { l l } { C \displaystyle \operatorname* { m a x } _ { \eta \leqslant u \leqslant 1 } \frac { \log ( V _ { u } ( \pmb { y } ) / V _ { \eta } ( \pmb { y } ) ) + \log ( 2 / \beta ) } { \varepsilon u } , } & { \mathrm { f o r ~ p u r e ~ D P } , } \\ { C \displaystyle \operatorname* { m a x } _ { \eta \leqslant u \leqslant \eta ^ { + } } \frac { \log ( V _ { u } ( \pmb { y } ) / V _ { \eta } ( \pmb { y } ) ) + \log ( 2 / \beta ) } { \varepsilon u } , } & { \mathrm { f o r ~ a p p r o x i m a t e ~ D P } , } \end{array} \right.\tag{5}
$$

then the capped volume levels are at most $2 ^ { j + 1 } \eta \leqslant 4 ( 2 ^ { j } - 1 ) \eta ,$ which lets us absorb the volume ratio into half of the exponential decay by choosing � suficiently large. Taking $u = \eta$ in (5) also gives $\varepsilon n \eta \geqslant C \log ( 2 / \beta )$ . Consequently the sum of the shell probabilities is at most

$$
A \sum _ { j \geqslant 1 } \exp \bigl ( - \frac { c } { 2 } \varepsilon n ( 2 ^ { j } - 1 ) \eta \bigr ) \leqslant \beta / 2 .
$$

For approximate DP, the reduction’s test based on the minimum score accepts whenever that minimum is at most ��: indeed $\eta ^ { * } \geqslant 1 0 \eta ,$ and (5) gives �� $\geqslant C \log ( 2 / \beta ) / \varepsilon$ , which exceeds a suficiently large constant. Its sampler has support below $\eta ^ { * } n ,$ , since $0 . 9 \eta ^ { * } n + O ( 1 ) < \eta ^ { * } n$ . Set its numerical failure probability to at most min $\{ \beta / 2 , c _ { 0 } \delta \}$ for a suficiently small absolute constant $c _ { 0 } > 0$ . Rescaling the privacy and failure parameters by absolute constants, as in the cited reductions, proves the claim. □

## 3 Robust Estimation for Discrete Batches

In this section, we design a robust estimation algorithm. In Section 4 we will show how to use this robust algorithm (Theorem 3.1) and Theorem 2.18 to obtain a private algorithm. For the technical arguments in this section, fix a suficiently small absolute constant $\eta _ { \mathrm { a b s } } \in ( 0 , 1 / 2 0 )$

Theorem 3.1 (Robust Estimation from Discrete Batches). Let $\mu \in \Delta ^ { k - 1 } , \eta \in ( 0 , \eta _ { \mathrm { a b s } } ) , \beta \in ( 0 , 1 )$ , and $m \in \mathbb { N }$ . Then there exists a polynomial-time algorithm, obtained by solving a degree-6 SoS relaxation, that takes in � user batches of size � forming an �-corruption of independent clean batches

$$
B _ { i } ^ { \star } = ( U _ { i , 1 } ^ { \star } , \ldots , U _ { i , m } ^ { \star } ) , \qquad U _ { i , 1 } ^ { \star } , \ldots , U _ { i , m } ^ { \star } \stackrel { \mathrm { i i d } } { \sim } \mu ,
$$

and outputs an estimate $\hat { \mu }$ such that $\| { \hat { \boldsymbol { \mu } } } - { \boldsymbol { \mu } } \| _ { \mathrm { T V } } \leqslant \alpha .$ , with probability at least $1 \ : - \ : \beta .$ , where $\alpha = O ( \eta \sqrt { \log ( 1 / \eta ) / m } )$ , provided that

$$
n = \Omega \left( \frac { k + \log ( 1 / \beta ) } { \alpha ^ { 2 } m } \right) .
$$

We will view the robust constraints through the variance discrepancy for a kept subset with empirical mean $\bar { \mu } _ { T }$ and covariance $\widehat { \Sigma } _ { T } \colon$

$$
\langle s s ^ { \top } , \widehat { \Sigma } _ { T } \rangle - \frac { 1 } { m } \big ( \langle \bar { \mu } _ { T } , s \rangle - \langle \bar { \mu } _ { T } , s \rangle ^ { 2 } \big ) , \qquad s \in \{ 0 , 1 \} ^ { k } .
$$

This is the object optimized by the SDP-based algorithms of Jain and Orlitsky [JO20b] and Chen, Li, and Moitra [CLM20b]. Our succinct SoS certificate will be the dual object: a trace-bounded diagonal majorizer.

We isolate the deterministic properties of the clean samples needed in the robust analysis.

Definition 3.2 (�-Good Samples). Let $\mu \in \Delta ^ { k - 1 } , \eta \in ( 0 , 1 / 2 )$ , and $m \in \mathbb { N } .$ . Let $y _ { 1 } ^ { * } , \ldots , y _ { n } ^ { * } \in \mathbb { R } ^ { k }$ be any points. For $s \in \{ 0 , 1 \} ^ { k }$ , write

$$
\xi _ { i } ( s ) : = \langle y _ { i } ^ { * } - \mu , s \rangle .
$$

We say that $y _ { 1 } ^ { * } , \ldots , y _ { n } ^ { * }$ are �-good $i f f o r$ every $s \in \{ 0 , 1 \} ^ { k }$ we simultaneously have

1. $| \mathbb { E } _ { i } \xi _ { i } ( s ) | \leqslant O ( \eta \sqrt { \log ( 1 / \eta ) / m } )$

2. $\left| \mathbb { E } _ { i } \xi _ { i } ( s ) ^ { 2 } - V ( \mu , s ) \right| \leqslant O \bigl ( \eta \log ( 1 / \eta ) / m \bigr ) .$

3. For every subset $T \subseteq [ n ]$ with $| T | \leqslant 2 \eta n$

$$
\frac 1 n \sum _ { i \in T } \xi _ { i } ( s ) ^ { 2 } \leqslant O \bigl ( \eta \log ( 1 / \eta ) / m \bigr ) .
$$

Clean batch-average vectors are �-good with high probability, as shown next.

Corollary 3.3 (True Samples Are �-Good with High Probability). Let $\mu \in \Delta ^ { k - 1 } , \eta \in ( 0 , 1 / 4 )$ $\beta \in \left( 0 , 1 / 2 \right)$ , and $m \in \mathbb { N }$ . Let

$$
B _ { i } ^ { * } = ( U _ { i , 1 } ^ { * } , \ldots , U _ { i , m } ^ { * } ) , \qquad U _ { i , 1 } ^ { * } , \ldots , U _ { i , m } ^ { * } \overset { \mathrm { i i d } } { \sim } \mu , \qquad y _ { i } ^ { * } : = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } e _ { U _ { i , j } ^ { * } }
$$

for each $i \in [ n ]$ . Then as long as $n \geqslant \Omega ( ( k + \log ( 1 / \beta ) ) / ( \eta ^ { 2 } \log ( 1 / \eta ) ) )$ , with probability $1 - \beta ,$ the samples $y _ { 1 } ^ { * } , \ldots , y _ { n } ^ { * }$ are �-good in the sense of Definition 3.2.

Proof. Apply Fact 2.6 with parameter $2 \eta .$ . Its sample-size requirement is satisfied by the hypothesis of Corollary $3 . 3 ,$ up to absolute constants, so the concentration event holds with probability $1 - \beta$ Taking the subset $T = [ n ]$ gives Properties 1 and 2 of Definition 3.2.

For the third property, fix a subset $T \subseteq [ n ]$ with $| T | \leqslant 2 \eta n$ , and let $S : = [ n ] \backslash T$ . Then $\begin{array} { r } { | S | \geqslant ( 1 - 2 \eta ) n , } \end{array}$ and

$$
\frac { 1 } { n } \sum _ { i \in \cal { T } } \xi _ { i } ( s ) ^ { 2 } = \mathop { \mathbb { E } } _ { i } \xi _ { i } ( s ) ^ { 2 } - \frac { | S | } { n } \cdot \frac { 1 } { | S | } \sum _ { i \in \cal { S } } \xi _ { i } ( s ) ^ { 2 } .
$$

Therefore,

$$
\begin{array} { c l } { \displaystyle \frac { 1 } { n } \sum _ { i \in T } \xi _ { i } ( s ) ^ { 2 } \leqslant \left| \mathbb { E } \xi _ { i } ( s ) ^ { 2 } - V ( \mu , s ) \right| + \displaystyle \frac { | S | } { n } \left| \displaystyle \frac { 1 } { | S | } \sum _ { i \in S } \xi _ { i } ( s ) ^ { 2 } - V ( \mu , s ) \right| + \displaystyle \frac { | T | } { n } V ( \mu , s ) } \\ { \displaystyle \leqslant O \left( \displaystyle \frac { \eta \log ( 1 / \eta ) } { m } \right) + O \left( \displaystyle \frac { \eta \log ( 1 / \eta ) } { m } \right) + 2 \eta \cdot \displaystyle \frac { 1 } { 4 m } } \\ { \displaystyle } \end{array}
$$

Here we used $V ( \mu , s ) \leqslant 1 / ( 4 m )$ . This proves Property 3 of Definition 3.2, and hence the samples are �-good. □

## 3.1 Sum-of-Squares Based Robust Algorithm

Let

$$
B _ { i } ^ { * } = ( U _ { i , 1 } ^ { * } , \ldots , U _ { i , m } ^ { * } ) , \qquad U _ { i , 1 } ^ { * } , \ldots , U _ { i , m } ^ { * } \overset { \mathrm { i i d } } { \sim } \mu , \qquad y _ { i } ^ { * } : = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } e _ { U _ { i , j } ^ { * } }
$$

for each $i \in [ n ]$ . Let $\left\{ y _ { i } \right\} _ { i = 1 } ^ { n }$ be an �-corrupted version of these clean batch-average vectors, and let $\left\{ y _ { i } ^ { \prime } \right\} _ { i = 1 } ^ { n }$ be the repaired batch-average indeterminates. Write

$$
\widehat { \Sigma } ^ { \prime } : = \mathbb { E } ( y _ { i } ^ { \prime } - \mu ^ { \prime } ) ( y _ { i } ^ { \prime } - \mu ^ { \prime } ) ^ { \top } , \qquad \Sigma ( \nu ) : = \frac { 1 } { m } \big ( \mathrm { d i a g } ( \nu ) - \nu \nu ^ { \top } \big ) , \qquad \tau _ { \mathrm { c o v } } : = C \frac { \eta \log ( 1 / \eta ) } { m } ,
$$

for a suficiently large universal constant �. With variables $\lambda \in \mathbb { R } ^ { k }$ and $M \in \mathbb { R } ^ { k \times k }$ , consider the following constraint system:

selector/data consistency:

$$
\forall i \in [ n ] : w _ { i } ^ { 2 } = w _ { i } , \quad w _ { i } y _ { i } ^ { \prime } = w _ { i } y _ { i } , \quad \sum _ { i = 1 } ^ { n } w _ { i } \geqslant ( 1 - \eta ) n , \Bigg |
$$

simplex constraints:

$$
\forall i , j \in [ n ] \times [ k ] : y _ { i , j } ^ { \prime } \geqslant 0 , \quad \forall i \in [ n ] : \sum _ { j \in [ k ] } y _ { i , j } ^ { \prime } = 1 ,
$$

$$
\forall j \in [ k ] : \ \mu _ { j } ^ { \prime } \geqslant 0 , \quad \sum _ { j \in [ k ] } \mu _ { j } ^ { \prime } = 1 ,
$$

covariance-majorizer constraints:

$$
\widehat { \Sigma } ^ { \prime } + M M ^ { \top } = \Sigma ( \mu ^ { \prime } ) + \mathrm { D i a g } ( \lambda ) , ~ \sum _ { j \in [ k ] } \lambda _ { j } \leqslant \tau _ { \mathrm { c o v } } ,
$$

$$
\mathrm { w h e r e } \mu ^ { \prime } = \underset { i } { \mathbb { E } } y _ { i } ^ { \prime } .
$$

Selector/data consistency determines which observed batch averages are kept and ties the repaired points to the observed data on those indices. Simplex constraints keep both the repaired points and the pseudo-mean inside the simplex. To replace all subset-direction variance checks succinctly, the covariance-majorizer block uses a diagonal slack that upper-bounds the diference between the empirical and multinomial covariance simultaneously in every subset direction.

On the �-good event, the clean data provide a witness for this system. If $r _ { i } = \mathbb { 1 } [ y _ { i } = y _ { i } ^ { * } ]$ and $\bar { \mu } = \mathbb { E } _ { i } y _ { i } ^ { * }$ , set $w _ { i } = r _ { i } , y _ { i } ^ { \prime } = y _ { i } ^ { * } ,$ , and $\mu ^ { \prime } = \bar { \mu }$ . These choices satisfy the selector/data-consistency and simplex constraints. By Proposition 2.17, feasibility of the covariance-majorizer constraint follows once we verify subset discrepancy and the nullspace condition for the clean covariance-gap matrix $\Delta _ { \mathrm { c o v } } ^ { \star }$

Lemma 3.4 (Clean Covariance Gap Has Small Subset Discrepancy). Let $y _ { 1 } ^ { * } , \ldots , y _ { n } ^ { * }$ be clean batchaverage vectors, and suppose they are �-good in the sense of Definition 3.2. Let

$$
\bar { \mu } : = \mathbb { E } \ y _ { i } ^ { * } , \qquad \Delta _ { \mathrm { c o v } } ^ { \star } : = \mathbb { E } ( y _ { i } ^ { * } - \bar { \mu } ) ( y _ { i } ^ { * } - \bar { \mu } ) ^ { \top } - \frac { 1 } { m } \big ( \mathrm { d i a g } ( \bar { \mu } ) - \bar { \mu } \bar { \mu } ^ { \top } \big ) \ .
$$

Then for every $s \in \{ 0 , 1 \} ^ { k }$

$$
\left| s ^ { \top } \Delta _ { \mathrm { c o v } } ^ { \star } s \right| \leqslant O \left( \frac { \eta \log ( 1 / \eta ) } { m } \right) .
$$

Proof. Let $h : = \bar { \mu } - \mu$ . Fix $s \in \{ 0 , 1 \} ^ { k }$ , and write $\xi _ { i } ( s ) : = \langle y _ { i } ^ { * } - \mu , s \rangle$ . Then

$$
\begin{array} { r l } & { s ^ { \top } \Delta _ { \mathrm { c o v } } ^ { \star } s = \underline { { \mathbb { F } } } ( y _ { i } ^ { \ast } - \bar { \mu } , s ) ^ { 2 } - V ( \bar { \mu } , s ) } \\ & { \qquad = \underline { { \mathbb { F } } } ( \xi _ { i } ( s ) - \langle h , s \rangle ) ^ { 2 } - V ( \bar { \mu } , s ) } \\ & { \qquad = \underline { { \mathbb { F } } } \xi _ { i } ( s ) ^ { 2 } - \langle h , s \rangle ^ { 2 } - V ( \bar { \mu } , s ) } \\ & { \qquad = \bigg ( \underline { { \mathbb { F } } } \xi _ { i } ( s ) ^ { 2 } - V ( \mu , s ) \bigg ) - \langle h , s \rangle ^ { 2 } + \big ( V ( \mu , s ) - V ( \bar { \mu } , s ) \big ) . } \end{array}
$$

By Property 1 of Definition 3.2,

$$
\left| \left. h , s \right. \right| = \left| \mathbb { E } \xi _ { i } ( s ) \right| \leqslant O \left( \eta \sqrt { \frac { \log ( 1 / \eta ) } { m } } \right) .
$$

Since $x \mapsto x ( 1 - x )$ is 1-Lipschitz on [0 1], we obtain

$$
\bigl | V ( \mu , s ) - V ( \bar { \mu } , s ) \bigr | \leqslant \frac { | \langle h , s \rangle | } { m } \leqslant O \left( \frac { \eta \sqrt { \log ( 1 / \eta ) } } { m ^ { 3 / 2 } } \right) \leqslant O \left( \frac { \eta \log ( 1 / \eta ) } { m } \right) .
$$

Therefore

$$
\begin{array} { r l } & { \left| s ^ { \top } \Delta _ { \mathrm { c o v } } ^ { \star } s \right| \leqslant \left| \underset { i } { \mathbb { E } } \xi _ { i } ( s ) ^ { 2 } - V ( \mu , s ) \right| + \left| \langle h , s \rangle \right| ^ { 2 } + \left| V ( \mu , s ) - V ( \bar { \mu } , s ) \right| } \\ & { \qquad \leqslant O \left( \frac { \eta \log ( 1 / \eta ) } { m } \right) + O \left( \frac { \eta ^ { 2 } \log ( 1 / \eta ) } { m } \right) + O \left( \frac { \eta \log ( 1 / \eta ) } { m } \right) } \\ & { \qquad \leqslant O \left( \frac { \eta \log ( 1 / \eta ) } { m } \right) . } \end{array}
$$

Lemma 3.5 (Clean Covariance Gap Kills the All-Ones Direction). Let $y _ { 1 } ^ { * } , \ldots , y _ { n } ^ { * }$ be clean batch-average vectors, and define

$$
\bar { \mu } : = \mathbb { E } \ y _ { i } ^ { * } , \qquad \Delta _ { \mathrm { c o v } } ^ { \star } : = \mathbb { E } ( y _ { i } ^ { * } - \bar { \mu } ) ( y _ { i } ^ { * } - \bar { \mu } ) ^ { \top } - \frac { 1 } { m } \big ( \mathrm { d i a g } ( \bar { \mu } ) - \bar { \mu } \bar { \mu } ^ { \top } \big ) \ .
$$

Then

$$
\begin{array} { r } { \Delta _ { \mathrm { c o v } } ^ { \star } \mathbf { 1 } = 0 . } \end{array}
$$

Proof. Since each $y _ { i } ^ { * }$ and $\bar { \mu }$ lie in the simplex,

$$
\langle y _ { i } ^ { * } - { \bar { \mu } } , \mathbf { 1 } \rangle = 0 \qquad { \mathrm { f o r ~ a l l ~ } } i \in [ n ] ,
$$

and hence

$$
\mathbb { E } ( y _ { i } ^ { * } - \bar { \mu } ) ( y _ { i } ^ { * } - \bar { \mu } ) ^ { \mathsf { T } } { \bf 1 } = 0 .
$$

Also,

$$
\big ( \mathrm { d i a g } ( \bar { \mu } ) - \bar { \mu } \bar { \mu } ^ { \top } \big ) { \bf 1 } = \bar { \mu } - \bar { \mu } \langle \bar { \mu } , { \bf 1 } \rangle = 0 .
$$

Corollary 3.6 (Diagonal-Majorizer Feasibility). Let $y _ { 1 } ^ { * } , \ldots , y _ { n } ^ { * }$ be clean batch-average vectors, and suppose these vectors are �-good in the sense of Definition 3.2. Let

$$
\bar { \mu } : = \mathbb { E } \ y _ { i } ^ { * } , \qquad \Delta _ { \mathrm { c o v } } ^ { \star } : = \mathbb { E } ( y _ { i } ^ { * } - \bar { \mu } ) ( y _ { i } ^ { * } - \bar { \mu } ) ^ { \top } - \frac { 1 } { m } \big ( \mathrm { d i a g } ( \bar { \mu } ) - \bar { \mu } \bar { \mu } ^ { \top } \big ) .
$$

Then there exists $\boldsymbol { \lambda } \in \mathbb { R } ^ { k }$ such that

$$
\mathrm { D i a g } ( \lambda ) - \Delta _ { \mathrm { c o v } } ^ { \star } \amalg \asymp 0 , \qquad \sum _ { j \in [ k ] } \lambda _ { j } \leqslant O \left( \frac { \eta \log ( 1 / \eta ) } { m } \right) .
$$

Proof. By Proposition 2.17, it remains only to verify the subset-discrepancy and nullspace hypotheses. Those are exactly Lemmas 3.4 and 3.5 applied to $\Delta = \Delta _ { \mathrm { c o v } } ^ { \star }$ □

This proves feasibility of the covariance-majorizer block for the clean witness.

If $\Delta ^ { \prime } \mathbf { 1 } = 0$ and $\mathrm { D i a g } ( \lambda ) - \Delta ^ { \prime } \succcurlyeq 0$ , then for every subset direction $s \in \{ 0 , 1 \} ^ { k }$ and $\boldsymbol { \sigma } = 2 s - \mathbf { 1 }$

$$
4 s ^ { \top } \Delta ^ { \prime } s = \sigma ^ { \top } \Delta ^ { \prime } \sigma \leqslant \sigma ^ { \top } \mathrm { D i a g } ( \lambda ) \sigma = \sum _ { j \in [ k ] } \lambda _ { j } .
$$

Thus one matrix majorizer simultaneously controls every subset direction. We next give the SoS version of this calculation for $\Delta _ { \mathrm { c o v } } ^ { \prime } = \widehat { \Sigma ^ { \prime } } - \bar { \Sigma } ( \mu ^ { \prime } )$

Lemma 3.7 (Diagonal Majorizer Implies Program-Side Subset-Direction Variance Control). Under $\mathcal { A } _ { \eta }$ and the Boolean constraints $\{ \forall j \in \mathbf { \widetilde { [ } } k ] : s _ { j } ^ { 2 } = s _ { j } \}$ , we have

$$
\mathcal { A } _ { \eta } \cup \left\{ \forall j \in [ k ] : s _ { j } ^ { 2 } = s _ { j } \right\} \Big | _ { \mathsf { \overline { { 6 } } } } \left\{ s ^ { \top } \Big ( \widehat { \Sigma } ^ { \prime } - \Sigma ( \mu ^ { \prime } ) \Big ) s \leqslant \frac { 1 } { 4 } \sum _ { j \in [ k ] } \lambda _ { j } \right\} .
$$

Consequently,

$$
\mathcal { A } _ { \eta } \cup \{ \forall j \in [ k ] : s _ { j } ^ { 2 } = s _ { j } \} | _ { \overline { { 6 } } } \{ \mathbb { E } \langle y _ { i } ^ { \prime } - \mu ^ { \prime } , s \rangle ^ { 2 } \leqslant \frac { \langle \mu ^ { \prime } , s \rangle \big ( 1 - \langle \mu ^ { \prime } , s \rangle \big ) } { m } + O \bigg ( \frac { \eta \log ( 1 / \eta ) } { m } \bigg ) \} .
$$

Proof. Write

$$
\Delta _ { \mathrm { c o v } } ^ { \prime } : = \widehat { \Sigma } ^ { \prime } - \Sigma ( \mu ^ { \prime } ) , \qquad \sigma : = 2 s - { \bf 1 } .
$$

We first show that $\mathcal { A } _ { \eta } \left| _ { 6 } \right. \{ \Delta _ { \mathrm { c o v } } ^ { \prime } \mathbf { 1 } = 0 \}$ . Indeed, for every $i \in [ n ]$

$$
\mathcal { A } _ { \eta } \left| _ { 6 } \right. \left\{ \langle y _ { i } ^ { \prime } - \mu ^ { \prime } , \mathbf { 1 } \rangle = \sum _ { j \in [ k ] } y _ { i , j } ^ { \prime } - \sum _ { j \in [ k ] } \mu _ { j } ^ { \prime } = 0 \right\} .
$$

Therefore

$$
\begin{array} { r } { \mathcal { A } _ { \eta } | _ { 6 } \{ \widehat { \Sigma } ^ { \prime } \mathbf { 1 } = \mathbb { E } ( y _ { i } ^ { \prime } - \mu ^ { \prime } ) \langle y _ { i } ^ { \prime } - \mu ^ { \prime } , \mathbf { 1 } \rangle = 0 \} . } \end{array}
$$

Also,

$$
\mathcal { A } _ { \eta }  _ { 6 } \{ \Sigma ( \mu ^ { \prime } ) { \bf 1 } = \frac { 1 } { m } \big ( \mu ^ { \prime } - \mu ^ { \prime } \langle \mu ^ { \prime } , { \bf 1 } \rangle \big ) = 0 \} .
$$

and hence $\Delta _ { \mathrm { c o v } } ^ { \prime } \mathbf { 1 } = 0$

Using this identity, we have

$$
\begin{array} { r } { \mathcal { A } _ { \eta } \cup \{ \forall j \in [ k ] : s _ { j } ^ { 2 } = s _ { j } \} | _ { 6 } \{ \sigma ^ { \top } \Delta _ { \mathrm { c o v } } ^ { \prime } \sigma = ( 2 s - { \bf 1 } ) ^ { \top } \Delta _ { \mathrm { c o v } } ^ { \prime } ( 2 s - { \bf 1 } ) = 4 s ^ { \top } \Delta _ { \mathrm { c o v } } ^ { \prime } s \} . } \end{array}
$$

Now the diagonal-majorizer constraint in $\mathcal { A } _ { \eta }$ gives

$$
\begin{array} { r } { \mathcal { A } _ { \eta } \cup \left\{ \forall j \in [ k ] : s _ { j } ^ { 2 } = s _ { j } \right\} \big | _ { 6 } \left\{ \sigma ^ { \top } \Delta _ { \mathrm { c o v } } ^ { \prime } \sigma \leqslant \sigma ^ { \top } \mathrm { D i a g } ( \lambda ) \sigma \right\} . } \end{array}
$$

since

$$
\begin{array} { r } { \sigma ^ { \top } \big ( \mathrm { D i a g } ( \lambda ) - \Delta _ { \mathrm { c o v } } ^ { \prime } \big ) \sigma = \| M ^ { \top } \sigma \| _ { 2 } ^ { 2 } \geqslant 0 . } \end{array}
$$

Under the Boolean constraints on �, we have

$$
\left\{ \forall j \in [ k ] : s _ { j } ^ { 2 } = s _ { j } \right\} \big | _ { 6 } \left\{ \sigma _ { j } ^ { 2 } = ( 2 s _ { j } - 1 ) ^ { 2 } = 1 \right\}
$$

for every $j \in [ k ]$ , and therefore

$$
\mathcal { A } _ { \eta } \cup \left\{ \forall j \in [ k ] : s _ { j } ^ { 2 } = s _ { j } \right\} \Big | _ { 6 } \left\{ \sigma ^ { \top } \operatorname { D i a g } ( \lambda ) \sigma = \sum _ { j \in [ k ] } \lambda _ { j } \right\} .
$$

Substituting $\sigma ^ { \top } \Delta _ { \mathrm { c o v } } ^ { \prime } \sigma = 4 s ^ { \top } \Delta _ { \mathrm { c o v } } ^ { \prime } s$ and $\begin{array} { r } { \sigma ^ { \top } \operatorname { D i a g } ( \lambda ) \sigma = \sum _ { j } \lambda _ { j } } \end{array}$ into the diagonal-majorizer inequality proves the first claim.

For the variance bound, observe that

$$
\{ \forall j \in [ k ] : s _ { j } ^ { 2 } = s _ { j } \} | _ { 6 } \{ s ^ { \top } \widehat { \Sigma } ^ { \prime } s = \underset { i } { \mathbb { F } } \langle y _ { i } ^ { \prime } - \mu ^ { \prime } , s \rangle ^ { 2 } \}
$$

and

$$
{ \mathcal { A } } _ { \eta } \cup \{ \forall j \in [ k ] : s _ { j } ^ { 2 } = s _ { j } \} | _ { 6 } \{ s ^ { \top } \Sigma ( \mu ^ { \prime } ) s = { \frac { \langle \mu ^ { \prime } , s \rangle - \langle \mu ^ { \prime } , s \rangle ^ { 2 } } { m } } = { \frac { \langle \mu ^ { \prime } , s \rangle \big ( 1 - \langle \mu ^ { \prime } , s \rangle \big ) } { m } } \} .
$$

Using the bound $\begin{array} { r } { s ^ { \top } \Delta _ { \mathrm { c o v } } ^ { \prime } s \leqslant \frac { 1 } { 4 } \sum _ { j } \lambda _ { j } } \end{array}$ together with $\Sigma _ { j } \lambda _ { j } \leqslant \tau _ { \mathrm { c o v } }$ gives

$$
\displaystyle \mathcal { A } _ { \eta } \cup \left\{ \forall j \in [ k ] : s _ { j } ^ { 2 } = s _ { j } \right\} \Big | _ { \overline { { 6 } } } \left\{ \underset { i } { \mathbb { E } } \langle y _ { i } ^ { \prime } - \mu ^ { \prime } , s \rangle ^ { 2 } \leqslant \frac { \langle \mu ^ { \prime } , s \rangle \big ( 1 - \langle \mu ^ { \prime } , s \rangle \big ) } { m } + \frac { 1 } { 4 } \tau _ { \mathrm { c o v } } \right\} .
$$

which is the desired bound.

Algorithm 3.1 (Robust Estimation from Discrete Batches).

Input: An �-corrupted batch dataset $\{ B _ { i } \} _ { i \in [ n ] } ,$ , where each batch has size �. Output: An estimate $\hat { \mu }$ of the underlying discrete distribution $\mu .$

## Operations:

1. Convert each observed batch to its average vector

$$
y _ { i } : = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } e _ { U _ { i , j } } \in \Delta ^ { k - 1 } \qquad \mathrm { f o r ~ a l l ~ } i \in [ n ] .
$$

2. Form the SoS constraint system $\mathcal { A } _ { \eta }$ from Section 3.1, with variables $( w _ { i } ) _ { i \in [ n ] } ,$ $( y _ { i } ^ { \prime } ) _ { i \in [ n ] } , \mu ^ { \prime } , \lambda ,$ and $M .$

3. Find a degree-6 pseudoexpectation $\tilde { \mathbb { E } }$ satisfying $\mathcal { A } _ { \eta } .$

4. Output an $\ell _ { 1 }$ projection of $\widetilde { \mathbb { E } } \mu ^ { \prime }$ onto $\Delta ^ { k - 1 }$ . For an exact pseudoexpectation, this is $\hat { \mu } = \mathbf { \hat { \mathbb { E } } } \mu ^ { \prime } .$

We can now state our main technical lemma.

Lemma 3.8 (Main SoS Lemma). Suppose $\boldsymbol { \mathcal { Y } ^ { * } } = \{ \boldsymbol { y } _ { i } ^ { * } \} _ { i \in [ n ] }$ is the clean batch-average dataset and that it is �-good (as in Definition 3.2). Then for any fixed binary vector $s \in \left\{ 0 , 1 \right\} ^ { k }$ the following SoS proof exists.

$$
\mathcal { A } _ { \eta }  _ { 6 } \{ \langle \mu ^ { \prime } - \mu , s \rangle ^ { 2 } \leqslant O ( ( \eta \sqrt { \frac { \log ( 1 / \eta ) } { m } } ) ^ { 2 } ) \}
$$

Algorithm 3.1 summarizes the robust estimator promised by Theorem 3.1.

We use exact pseudoexpectation notation in the analysis. For computation, we may restrict the SDP to a ball of radius poly(� �) in moment space. Indeed, the clean witness has $w _ { i } , y _ { i , j } ^ { \prime } , \mu _ { j } ^ { \prime } \in [ 0 , 1 ]$ Since $( \Delta _ { \mathrm { c o v } } ^ { \star } ) _ { j j } \geqslant - 1 / ( 4 m )$ , its diagonal majorizer satisfies

$$
- \frac { 1 } { 4 m } \leqslant \lambda _ { j } \leqslant \tau _ { \mathrm { c o v } } + \frac { k - 1 } { 4 m } , \qquad \| M \| _ { \mathrm { F } } ^ { 2 } = \sum _ { j } \lambda _ { j } - \operatorname { t r } ( \Delta _ { \mathrm { c o v } } ^ { \star } ) \leqslant \tau _ { \mathrm { c o v } } + \frac { k } { 4 m } .
$$

Thus the point-evaluation functional at this witness has polynomially bounded degree-6 moments. In the standard bit model, solve the bounded SDP to an inverse-polynomial feasibility tolerance � chosen so that

$$
\tau \ \mathrm { p o l y } ( n , k , m ) = o \Bigg ( \frac { \eta ^ { 2 } \log ( 1 / \eta ) } { m } \Bigg ) .
$$

Since the degree-6 certificates in the proof of Lemma 3.8 have coeficient norms bounded by $\mathrm { p o l y } ( n , k , m )$ , approximate feasibility changes the squared-error bound by at most � poly $( n , k , m )$ Projecting onto the simplex in $\ell _ { 1 }$ changes the error by at most a constant factor. This preserves the error rate in Theorem 3.1 and polynomial running time, as in Lemma 4.11.

We now derive Theorem 3.1 from Lemma 3.8.

Proof of Theorem 3.1. Apply Corollary 3.3 with failure probability min $\{ \beta , 1 / 4 \}$ . With probability at least $1 - \beta ,$ , the clean batch-average vectors satisfy the three properties in Definition 3.2. Conditioned on this event, Lemma 3.8 applies. Let E<sup>˜</sup> be a degree-6 pseudoexpectation satisfying $\mathcal { A } _ { \eta } ,$ and let $\hat { \mu } = \tilde { \mathbb { E } } \mu ^ { \prime }$ be the output of the algorithm. Let $s \in \{ 0 , 1 \} ^ { k }$ be defined coordinate-wise by $s _ { i } = \mathbb { 1 } [ \hat { \mu } _ { i } > \mu _ { i } ]$ Then we have

$$
\begin{array} { r l } & { \| \hat { \mu } - \mu \| _ { \mathrm { T V } } ^ { 2 } = \langle \hat { \mu } - \mu , s \rangle ^ { 2 } } \\ & { \qquad = \langle \widetilde { \mathbb E } \mu ^ { \prime } - \mu , s \rangle ^ { 2 } } \\ & { \qquad \leqslant \widetilde { \mathbb E } \langle \mu ^ { \prime } - \mu , s \rangle ^ { 2 } } \\ & { \qquad \leqslant O \left( \left( \eta \sqrt { \frac { \log ( 1 / \eta ) } { m } } \right) ^ { 2 } \right) . } \end{array}
$$

Therefore,

$$
\| \hat { \mu } - \mu \| _ { \mathrm { T V } } \leqslant O \left( \eta \sqrt { \frac { \log ( 1 / \eta ) } { m } } \right) ,
$$

as desired.

## 3.2 Proof of the Main SoS Lemma

Proof. We proceed in three steps: first control the clean-kept and corrupted/discarded contributions in a fixed direction; then use the diagonal-majorizer certificate to upper bound the repaired second moment; finally control and absorb the remaining variance diference using the covariance formula $V ( \mu , s ) = \langle \mu , s \rangle ( 1 - \langle \mu , s \rangle ) / m$

Let $r _ { i }$ denote the indicator function of uncorrupted samples, i.e. $r _ { i } = \mathbb { 1 } [ y _ { i } ^ { * } = y _ { i } ]$ . Let $s \in \left\{ 0 , 1 \right\} ^ { k }$ be fixed, and write

$$
\Delta _ { s } : = \langle \mu ^ { \prime } - \mu , s \rangle , \qquad \xi _ { i } : = \langle y _ { i } ^ { * } - \mu , s \rangle , \qquad \xi _ { i } ^ { \prime } : = \langle y _ { i } ^ { \prime } - \mu , s \rangle .
$$

Multiplying by $r _ { i } w _ { i }$ selects indices that are both clean and kept; $r _ { i } ( 1 - w _ { i } )$ selects clean indices discarded by the selector; and $1 - r _ { i } w _ { i }$ is the complement.

Split into the clean-kept part and its complement. Applying SoS triangle inequality we can write

$$
\begin{array} { r } { \mathcal { A } _ { \eta } | \underset { 6 } { \overline { { \mathbf { \Theta } } } } \Delta _ { s } ^ { 2 } \leqslant 2 \underbrace { ( \frac { \mathbb { E } } { i } r _ { i } w _ { i } \xi _ { i } ^ { \prime } ) ^ { 2 } } _ { \mathrm { c l e a n - k e p t \ c o n t r i b u t i o n } } + 2 \underbrace { ( \frac { \mathbb { E } ( 1 - r _ { i } w _ { i } ) \xi _ { i } ^ { \prime } } { i } ) ^ { 2 } } _ { \mathrm { c o r r u p t e d - o r - d i s c a r d e d \ c o n t r i b u t i o n } } . } \end{array}\tag{6}
$$

Clean-kept contribution. Observing that $r _ { i } w _ { i } y _ { i } ^ { * } = r _ { i } w _ { i } y _ { i } ^ { \prime } .$ , we can write

$$
\begin{array} { r l } & { \mathcal { A } _ { \eta } \big | _ { \overline { { 6 } } } \left\{ \begin{array} { l l } { \left( \mathbb { E } r _ { i } w _ { i } \xi _ { i } ^ { \prime } \right) ^ { 2 } = \bigg ( \mathbb { E } r _ { i } w _ { i } \xi _ { i } \bigg ) ^ { 2 } } \\ { \qquad \leqslant 2 \bigg ( \mathbb { E } r _ { i } \xi _ { i } \bigg ) ^ { 2 } + 2 \bigg ( \mathbb { E } r _ { i } ( 1 - w _ { i } ) \xi _ { i } \bigg ) ^ { 2 } \Bigg \} . } \end{array} \right. } \end{array}
$$

Now

$$
\begin{array} { r l } & { \left( \frac { \mathbb { E } } { i } r _ { i } \xi _ { i } \right) ^ { 2 } = \left( \frac { \mathbb { E } } { i } \xi _ { i } - \frac { \mathbb { E } } { i } ( 1 - r _ { i } ) \xi _ { i } \right) ^ { 2 } } \\ & { \qquad \leqslant 2 \left( \frac { \mathbb { E } } { i } \xi _ { i } \right) ^ { 2 } + 2 \left( \frac { \mathbb { E } } { i } ( 1 - r _ { i } ) \xi _ { i } \right) ^ { 2 } } \\ & { \qquad \leqslant O \left( \frac { \eta ^ { 2 } \log ( 1 / \eta ) } { m } \right) + 2 \frac { \mathbb { E } } { i } ( 1 - r _ { i } ) \cdot \frac { \mathbb { E } } { i } ( 1 - r _ { i } ) \xi _ { i } ^ { 2 } } \\ & { \qquad \leqslant O \left( \frac { \eta ^ { 2 } \log ( 1 / \eta ) } { m } \right) . } \end{array}
$$

Here we used Property 1 of Definition 3.2 in the third line and Property 3 in the last line, since the corrupted indices form a subset of size at most ��. Therefore

$$
\mathcal { A } _ { \eta } \left| \frac { \ d _ { \mathsf { F } } } { \ d _ { 6 } } \left( \mathbb { E } r _ { i } w _ { i } \xi _ { i } ^ { \prime } \right) ^ { 2 } \right. \leqslant O \left( \frac { \eta ^ { 2 } \log ( 1 / \eta ) } { m } \right) + 2 \underbrace { \vphantom { \frac { 1 } { \eta ^ { 2 } \log ( 1 / \eta ) } } \left( \mathbb { E } r _ { i } ( 1 - w _ { i } ) \xi _ { i } \right) ^ { 2 } } _ { \mathrm { d i s c a r d e d - c l e a n ~ c o n t r i b u t i o n } } .\tag{7}
$$

We next bound the discarded-clean contribution. We have

$$
\begin{array} { r l } & { \mathcal { A } _ { \eta } | _ { 6 } \{ ( \frac { \mathbb { E } } { i } r _ { i } ( 1 - w _ { i } ) \xi _ { i } ) ^ { 2 } \leqslant ( \frac { \mathbb { E } } { i } r _ { i } ( 1 - w _ { i } ) ) \cdot ( \frac { \mathbb { E } } { i } r _ { i } ( 1 - w _ { i } ) \xi _ { i } ^ { 2 } ) \} } \\ & { \qquad \leqslant \eta \cdot ( \frac { \mathbb { E } } { i } ( 1 - w _ { i } ) \xi _ { i } ^ { 2 } ) } \end{array}
$$

Property 3 of Definition 3.2 together with Lemma 2.9 implies

$$
\begin{array} { r } { \mathcal { A } _ { \eta } \Big | _ { 6 } \left\{ \left( \begin{array} { l } { \mathbb { E } r _ { i } ( 1 - w _ { i } ) \xi _ { i } \Big ) ^ { 2 } \leqslant \eta \cdot O ( \eta \log ( 1 / \eta ) / m ) } . } \\ { = O ( \eta ^ { 2 } \log ( 1 / \eta ) / m ) } \end{array} \right\} . } \end{\right\array} \end{array}
$$

Therefore,

$$
\mathcal { A } _ { \eta } | _ { 6 } \{ ( \mathbb { E } _ { i } r _ { i } w _ { i } \xi _ { i } ^ { \prime } ) ^ { 2 } \leqslant O ( \frac { \eta ^ { 2 } \log ( 1 / \eta ) } { m } ) \} .
$$

Corrupted-or-discarded contribution. We can write

$$
\begin{array} { r } { \mathcal { \mathcal { R } } _ { \eta } | _ { \mathrm { t } } \{ \begin{array} { l l } { ( \underline { { \mathbb { E } } } ( 1 - r _ { i } w _ { i } ) \xi _ { i } ^ { \prime } ) ^ { 2 } \leqslant ( \underline { { \mathbb { E } } } ( 1 - r _ { i } w _ { i } ) ) ( \underline { { \mathbb { E } } } ( 1 - r _ { i } w _ { i } ) \xi _ { i } ^ { \prime 2 } ) } \\ { \qquad \leqslant 2 \eta ( \underline { { \mathbb { E } } } ( 1 - r _ { i } w _ { i } ) \xi _ { i } ^ { \prime 2 } ) } \\ { \qquad = 2 \eta ( \underbrace { \underline { { \mathbb { F } } } \xi _ { i } ^ { \prime 2 } } _ { \mathrm { r e p a i r e d ~ s e c o n d ~ m o m e n t } } - \underbrace { \underline { { \mathbb { F } } } r _ { i } w _ { i } \xi _ { i } ^ { \prime 2 } } _ { \mathrm { c l e a n - k e p t s e c o n d ~ m o m e n t } } ) ^ { \mathrm { t } } . } \end{array} \} . } \end{array}\tag{8}
$$

For completeness, the weighted Cauchy–Schwarz step in (8) is the degree-6 bottleneck. With $b _ { i } : = 1 - r _ { i } w _ { i }$ and $z _ { i } : = \xi _ { i } ^ { \prime } ,$ the Boolean identities $b _ { i } ^ { 2 } = b _ { i }$ give

$$
\begin{array} { r } { \mathcal { R } _ { \eta } \Big | _ { \overline { { 6 } } } \left\{ \begin{array} { l l } { \left( \underline { { \mathbb { F } } } \ b _ { i } \right) \left( \underline { { \mathbb { F } } } \ b _ { i } z _ { i } ^ { 2 } \right) - \left( \underline { { \mathbb { F } } } \ b _ { i } z _ { i } \right) ^ { 2 } = \frac { 1 } { 2 n ^ { 2 } } \displaystyle \sum _ { i , j } b _ { i } b _ { j } ( z _ { i } - z _ { j } ) ^ { 2 } } \\ { \qquad \quad = \frac { 1 } { 2 n ^ { 2 } } \displaystyle \sum _ { i , j } \left( b _ { i } b _ { j } ( z _ { i } - z _ { j } ) \right) ^ { 2 } \left\{ \begin{array} { l l } { \begin{array} { r l r } \end{array} } \end{array} \right. } \\ { \qquad \geqslant 0 } \end{array} \right. } \end{array}
$$

Since the polynomials being squared have degree 3, this certificate has degree 6. We first bound the repaired second moment. Since

$$
\begin{array} { r } { \underset { i } { \mathbb { E } } \xi _ { i } ^ { \prime 2 } = \underset { i } { \mathbb { E } } \langle y _ { i } ^ { \prime } - \mu ^ { \prime } , s \rangle ^ { 2 } + \Delta _ { s } ^ { 2 } , } \end{array}
$$

Lemma 3.7 with subset direction � gives

$$
\begin{array} { r } { \mathcal { A } _ { \eta } \big \vert _ { 6 } \left\{ \begin{array} { l } { \underset { i } { \mathbb { E } } \xi _ { i } ^ { \prime 2 } \leqslant \frac { \big \langle \mu ^ { \prime } , s \big \rangle \big ( 1 - \big \langle \mu ^ { \prime } , s \big \rangle \big ) } { m } + O \bigg ( \displaystyle \frac { \eta \log ( 1 / \eta ) } { m } \bigg ) + \Delta _ { s } ^ { 2 } } \\ { \qquad = V ( \mu ^ { \prime } , s ) + O \bigg ( \displaystyle \frac { \eta \log ( 1 / \eta ) } { m } \bigg ) + \Delta _ { s } ^ { 2 } } \end{array} \right\} , } \end{array}
$$

Next we lower bound the clean-kept second moment. Since $r _ { i } w _ { i } \xi _ { i } ^ { \prime 2 } = r _ { i } w _ { i } \xi _ { i } ^ { 2 } .$ , we have

$$
\mathcal { R } _ { \eta } | \frac { \mathbb { \Gamma } _ { i } } { 6 } \{ \begin{array} { l l } { \mathbb { E } \displaystyle r _ { i } w _ { i } \xi _ { i } ^ { \prime 2 } = \mathbb { E } r _ { i } w _ { i } \xi _ { i } ^ { 2 } } \\ { \quad \quad = \mathbb { E } \xi _ { i } ^ { 2 } - \mathbb { E } ( 1 - r _ { i } w _ { i } ) \xi _ { i } ^ { 2 } } \\ { \quad \quad \geqslant V ( \mu , s ) - O ( \eta \log ( 1 / \eta ) / m ) - \underbrace { \mathbb { E } ( 1 - r _ { i } w _ { i } ) \xi _ { i } ^ { 2 } } _ { \mathrm { m i s s i n g ~ c l e a n ~ s e c o n d ~ m o m e n t } } } \end{array} \} ,\tag{9}
$$

Here the first inequality follows from Property 2 of Definition 3.2, and the remaining error comes from the missing clean second moment. Since

$$
\sum _ { i } ( 1 - r _ { i } w _ { i } ) \leqslant \sum _ { i } ( 1 - r _ { i } ) + \sum _ { i } r _ { i } ( 1 - w _ { i } ) \leqslant 2 \eta n ,
$$

Property 3 of Definition 3.2 together with Lemma 2.9 gives

$$
\begin{array} { r } { \mathcal { A } _ { \eta } \big | _ { 6 } \mathbb { E } ( 1 - r _ { i } w _ { i } ) \xi _ { i } ^ { 2 } \leqslant O ( \eta \log ( 1 / \eta ) / m ) . } \end{array}
$$

Directional identifiability inequality. Apply the clean-kept contribution bound to (6), and use (8) for the complementary contribution. Combining the upper bound on $\mathbb { E } _ { i } \xi _ { i } ^ { \prime 2 }$ with the lower bound on $\mathbb { E } _ { i } r _ { i } w _ { i } \xi _ { i } ^ { \prime 2 }$ then gives

$$
\mathcal { A } _ { \eta } \big | _ { 6 } \left\{ \begin{array} { l l } { \displaystyle \Delta _ { s } ^ { 2 } \leqslant O \left( \frac { \eta ^ { 2 } \log ( 1 / \eta ) } { m } \right) + 4 \eta \left( \mathbb { E } \xi _ { i } ^ { \prime 2 } - \mathbb { E } r _ { i } w _ { i } \xi _ { i } ^ { \prime 2 } \right) } \\ { \displaystyle \leqslant O \left( \frac { \eta ^ { 2 } \log ( 1 / \eta ) } { m } \right) + 4 \eta \big ( V ( \mu ^ { \prime } , s ) - V ( \mu , s ) \big ) + 4 \eta \Delta _ { s } ^ { 2 } } \end{array} \right\} .\tag{10}
$$

Absorbing the variance diference. The term $4 \eta ( V ( \mu ^ { \prime } , s ) - V ( \mu , s ) )$ in (10) is linear in the variance diference, whereas Lemma 3.10 controls the square of that diference. We use Young’s inequality to pass from the linear term to the square. In Fact 2.8, take $a = V ( \mu ^ { \prime } , s ) - V ( \mu , s ) , b = 4 \eta .$ , and $\lambda \overset { = } { m } ^ { 2 } / 1 6$ . This gives

$$
\begin{array} { r } { \mathcal { A } _ { \eta } \left| \frac { 1 } { 4 } \left\{ \begin{array} { l } { \displaystyle 4 \eta \big ( V ( \mu ^ { \prime } , s ) - V ( \mu , s ) \big ) \leqslant \frac { m ^ { 2 } } { 1 6 } \big ( V ( \mu ^ { \prime } , s ) - V ( \mu , s ) \big ) ^ { 2 } + \frac { ( 4 \eta ) ^ { 2 } } { 4 ( m ^ { 2 } / 1 6 ) } } \\ { \displaystyle = \frac { m ^ { 2 } } { 1 6 } \big ( V ( \mu ^ { \prime } , s ) - V ( \mu , s ) \big ) ^ { 2 } + \frac { 6 4 \eta ^ { 2 } } { m ^ { 2 } } } \end{array} \right. \right\} . } \end{array}
$$

We chose $\lambda = m ^ { 2 } / 1 6$ so that Lemma 3.10 turns the first term on the right into $\Delta _ { s } ^ { 2 } / 4$ . Indeed, by the definition of $V _ { \ast }$

$$
\begin{array} { r l } & { \mathcal { R } _ { \eta }  \frac { \ l } { 6 } \{ \cfrac { m ^ { 2 } } { 1 6 } ( V ( \mu ^ { \prime } , s ) - V ( \mu , s ) ) ^ { 2 } = \cfrac { 1 } { 1 6 } ( \langle \mu ^ { \prime } , s \rangle \big ( 1 - \langle \mu ^ { \prime } , s \rangle \big ) - \langle \mu , s \rangle \big ( 1 - \langle \mu , s \rangle \big ) ) ^ { 2 } \} . } \\ & { \qquad \leqslant \cfrac { 1 } { 4 } \Delta _ { s } ^ { 2 } } \end{array}
$$

Combining these two inequalities, we obtain

$$
\mathcal { A } _ { \eta } \big | _ { 6 } \Big \{ 4 \eta \big ( V ( \mu ^ { \prime } , s ) - V ( \mu , s ) \big ) \leqslant \frac { 1 } { 4 } \Delta _ { s } ^ { 2 } + \frac { 6 4 \eta ^ { 2 } } { m ^ { 2 } } \Big \} .\tag{11}
$$

Substituting (11) into (10) gives

$$
\mathcal { A } _ { \eta } \left| _ { 6 } \right. \left\{ \Delta _ { s } ^ { 2 } \leqslant O \left( \frac { \eta ^ { 2 } \log ( 1 / \eta ) } { m } \right) + \left( \frac { 1 } { 4 } + 4 \eta \right) \Delta _ { s } ^ { 2 } + \frac { 6 4 \eta ^ { 2 } } { m ^ { 2 } } \right\} .
$$

Since $\eta \leqslant \eta _ { \mathsf { a b s } }$ and $\eta _ { \mathrm { a b s } }$ is a suficiently small absolute constant, we may absorb the middle term into the left-hand side and conclude

$$
\mathcal { A } _ { \eta } \big | _ { \overline { { 6 } } } \left\{ \Delta _ { s } ^ { 2 } \leqslant O \left( \frac { \eta ^ { 2 } \log ( 1 / \eta ) } { m } \right) + O \left( \frac { \eta ^ { 2 } } { m ^ { 2 } } \right) = O \left( \frac { \eta ^ { 2 } \log ( 1 / \eta ) } { m } \right) \right\} ,
$$

as desired. The weighted Cauchy–Schwarz step in (8) has degree $6 ,$ and every other step has degree at most 6. Hence the entire proof has degree 6. □

## 3.3 Covariance Closeness Guarantees

We first prove the following lemma that will help us in bounding the diferences of covariances by the diferences of the means inside SoS.

Lemma 3.9. Let � be an indeterminate and � a scalar in [0 1]. Then

$$
\begin{array} { r } { \left\{ 0 \leqslant x \leqslant 1 \right\} \frac { \left| x \right. } { 6 } \left\{ \left( x ( 1 - x ) - a ( 1 - a ) \right) ^ { 2 } \leqslant 4 ( x - a ) ^ { 2 } \right\} } \end{array}
$$

Proof. We have

$$
\begin{array} { r l } & { 4 ( x - a ) ^ { 2 } - ( x ( 1 - x ) - a ( 1 - a ) ) ^ { 2 } = 4 ( x - a ) ^ { 2 } - ( x - a ) ^ { 2 } ( x + a - 1 ) ^ { 2 } } \\ & { \qquad = 3 ( x - a ) ^ { 2 } + ( x - a ) ^ { 2 } \bigl ( 1 - ( x + a - 1 ) ^ { 2 } \bigr ) } \\ & { \qquad = 3 ( x - a ) ^ { 2 } + \bigl ( 1 - ( a - 1 ) ^ { 2 } \bigr ) ( 1 - x ) ( x - a ) ^ { 2 } + ( 1 - a ^ { 2 } ) x ( x - a ) ^ { 2 } } \\ & { \qquad \quad + x \bigl [ ( x - a ) ( 1 - x ) \bigr ] ^ { 2 } + ( 1 - x ) \bigl [ x ( x - a ) \bigr ] ^ { 2 } } \\ & { \qquad \geqslant 0 . } \end{array}
$$

where every term in the last two lines is a square times either $1 , x ,$ or $1 - x _ { \mathrm { { , } } }$ , with a nonnegative scalar coeficient, since $1 - ( a - 1 ) ^ { 2 } \geqslant 0$ and $1 ^ { - } - a ^ { 2 } \geqslant 0$ . Thus,

$$
\begin{array} { r } { \{ 0 \leqslant x \leqslant 1 \} | \frac { x } { 6 } \{ ( x ( 1 - x ) - a ( 1 - a ) ) ^ { 2 } \leqslant 4 ( x - a ) ^ { 2 } \} . } \end{array}
$$

Now using this lemma we can bound the diference between the covariances inside SoS.

Lemma 3.10. Let � be any fixed binary vector in $\{ 0 , 1 \} ^ { k }$ . Then

$$
\begin{array} { r } { \mathcal { A } _ { \eta } \left| _ { 6 } \right. \left\{ ( \langle \mu ^ { \prime } , s \rangle ( 1 - \langle \mu ^ { \prime } , s \rangle ) - \langle \mu , s \rangle ( 1 - \langle \mu , s \rangle ) ) ^ { 2 } \leqslant 4 \langle \mu ^ { \prime } - \mu , s \rangle ^ { 2 } \right\} . } \end{array}
$$

Proof. Note that

$$
\mathcal { A } _ { \eta }  _ { \overline { { 6 } } } \{ ^ { \langle \mu ^ { \prime } , s \rangle = } \sum _ { j \in [ k ] } \mu _ { j } ^ { \prime } s _ { j } \} ,
$$

where the inequality comes from $\mathcal { A } _ { \eta } \vdash \{ \forall j \in [ k ] : \mu _ { j } ^ { \prime } \geqslant 0 \}$ , and that $s _ { j } ^ { \prime } \mathbf { s }$ are fixed scalars larger than or equal to 0. Moreover,

$$
\begin{array} { r l } & { \mathcal { A } _ { \eta }  _ { 6 } \{ { \begin{array} { l } { \langle \mu ^ { \prime } , s \rangle = \displaystyle \sum _ { j \in [ k ] } \mu _ { j } ^ { \prime } s _ { j } } } \\ { \leqslant 1 } \end{array} } \} ,  \\ & { \qquad \leqslant 1 } \end{array}
$$

where the inequality comes from the fact that $s _ { j } ^ { \prime } \mathsf { s }$ s are fixed scalars in $\{ 0 , 1 \}$ , and that $\mathcal { A } _ { \eta } \vdash \{ \forall j \in$ $[ k ] : \mu _ { j } ^ { \prime } \geqslant 0 , \sum _ { j } \mu _ { j } ^ { \prime } = 1 \}$ . Therefore,

$$
\begin{array} { r } { \mathcal { A } _ { \eta } \left| _ { 6 } \right. \left\{ 0 \leqslant \langle \mu ^ { \prime } , s \rangle \leqslant 1 \right\} , } \end{array}
$$

Applying Lemma 3.9 with $x = \langle \mu ^ { \prime } , s \rangle$ and the fixed scalar $a = \langle \mu , s \rangle \in [ 0 , 1 ]$ , we obtain

$$
\begin{array} { r } { \mathcal { A } _ { \eta } \left| _ { 6 } \right. \left\{ \left( \langle \mu ^ { \prime } , s \rangle ( 1 - \langle \mu ^ { \prime } , s \rangle ) - \langle \mu , s \rangle ( 1 - \langle \mu , s \rangle ) \right) ^ { 2 } \leqslant 4 \langle \mu ^ { \prime } - \mu , s \rangle ^ { 2 } \right\} , } \end{array}
$$

as desired.

## 3.4 Heterogeneous Clean Batches

Corollary 3.11. Suppose the clean batches are independent and batch � consists of � i.i.d. samples from $\mu _ { i } \in \Delta ^ { k - 1 }$ , where $\| \mu _ { i } - \mu \| _ { \mathrm { T V } } \leqslant \omega$ for every � and some known $\omega \in [ 0 , 1 ]$ . Let $\eta \in ( 0 , \eta _ { \mathrm { a b s } } )$ and $\beta \in ( 0 , 1 )$ . Replace the covariance allowance in Algorithm 3.1 by

$$
\tau _ { \mathrm { c o v } } = C \left( \frac { \eta \log ( 1 / \eta ) } { m } + \frac { \omega } { m } + \omega ^ { 2 } \right)
$$

for a suficiently large absolute constant �. $I f n \geqslant \Omega ( ( k + \log ( 1 / \beta ) ) / ( \eta ^ { 2 } \log ( 1 / \eta ) ) )$ , then on an �-corruption of these batches the algorithm returns, with probability at least $1 - \beta ,$

$$
\| \hat { \mu } - \mu \| _ { \mathrm { T V } } \leqslant O \left( \eta \sqrt { \frac { \log ( 1 / \eta ) } { m } } + \omega \right) .
$$

Proof. For a fixed $s \in \{ 0 , 1 \} ^ { k }$ , write $d _ { i } = \langle \mu _ { i } - \mu , s \rangle$ and $Z _ { i } = \langle y _ { i } ^ { * } - \mu _ { i } , s \rangle$ , so that $\left| d _ { i } \right| \leqslant \omega$ and $\xi _ { i } = Z _ { i } + d _ { i }$ . Each $Z _ { i }$ is centered and subgaussian with parameter ${ \mathrm { \dot { O } } } ( 1 / m )$ . By independence, the proof of Fact 2.6 applies to these variables even though their distributions may difer. In addition, the weighted average $\mathbb { E } _ { i } d _ { i } Z _ { i }$ has subgaussian parameter $O ( \omega ^ { 2 } / ( n m ) )$ . A union bound over �, the inequality $| V ( \mu _ { i } , s ) - V ( \mu , s ) | \leqslant \omega / m _ { \cdot }$ , and $( Z _ { i } \dot { + } d _ { i } ) ^ { 2 } \leqslant 2 Z _ { i } ^ { 2 } + 2 d _ { i } ^ { 2 }$ give the following simultaneous bounds with probability at least $1 - \beta \colon$

$$
\begin{array} { c } { \displaystyle | \mathbb { E } \xi _ { i } | \leqslant O \left( \eta \sqrt { \frac { \log ( 1 / \eta ) } { m } } + \omega \right) , } \\ { \displaystyle | \mathbb { E } \xi _ { i } ^ { 2 } - V ( \mu , s ) | \leqslant O \left( \frac { \eta \log ( 1 / \eta ) } { m } + \frac { \omega } { m } + \omega ^ { 2 } \right) , } \\ { \displaystyle \operatorname* { m a x } _ { | T | \leqslant 2 \eta n } \frac { 1 } { n } \sum _ { i \in T } \xi _ { i } ^ { 2 } \leqslant O \left( \frac { \eta \log ( 1 / \eta ) } { m } + \eta \omega ^ { 2 } \right) . } \end{array}
$$

Here the cross term in the second-moment estimate is at most $O ( \omega \eta \sqrt { \log ( 1 / \eta ) / m } ) \leqslant O ( \omega ^ { 2 } +$ $\eta ^ { 2 } \log ( 1 / \eta ) / m )$ , which is absorbed by the $\omega ^ { 2 }$ and $\eta \log ( 1 / \eta ) / m$ terms. Substituting the mean and second-moment estimates in the proof of Lemma 3.4 bounds the clean covariance discrepancy by $O ( \eta \log ( 1 / \eta ) / m + \omega / m + \omega ^ { 2 } )$ in every subset direction. Moreover, the matrix $\Delta _ { \mathrm { c o v } } ^ { \star }$ defined in that lemma satisfies $\Delta _ { \mathrm { c o v } } ^ { \star } \mathbf { 1 } = 0$ because each $y _ { i } ^ { * }$ and $\bar { \boldsymbol { \mu } } = \mathbb { E } _ { i } \boldsymbol { y } _ { i } ^ { * }$ has coordinate sum one. Thus Proposition 2.17 gives a diagonal majorizer with trace at most $\tau _ { \mathrm { { C O V } } } ,$ proving feasibility.

For soundness, the clean-kept contribution in the proof of Lemma 3.8 is now ${ \cal O } ( \eta ^ { 2 } \log ( 1 / \eta ) / m +$ $\omega ^ { 2 } )$ . Comparing the repaired and clean-kept second moments, as in (10), gives

$$
\mathcal { A } _ { \eta } | _ { - } \{ \Delta _ { s } ^ { 2 } \leqslant O ( \frac { \eta ^ { 2 } \log ( 1 / \eta ) } { m } + \omega ^ { 2 } + \frac { \eta \omega } { m } ) + 4 \eta \big ( V ( \mu ^ { \prime } , s ) - V ( \mu , s ) \big ) + 4 \eta \Delta _ { s } ^ { 2 } \} .
$$

By Young’s inequality, ��/� is at most $( \omega ^ { 2 } + \eta ^ { 2 } / m ^ { 2 } ) / 2$ . Applying (11) and absorbing $( 1 / 4 + 4 \eta ) \Delta _ { s } ^ { 2 }$ into the left-hand side gives $\Delta _ { s } ^ { 2 } \leqslant O ( \eta ^ { 2 } \log ( 1 / \eta ) / m + \omega ^ { 2 } )$ at degree 6. Taking pseudoexpectations and using total-variation duality completes the proof. □

## 4 Private Estimation for Discrete Batches

This section instantiates the reduction of Hopkins, Kamath, Majid, and Narayanan [HKMN23] on the batch-average dataset $\boldsymbol { \mathcal { Y } } = \{ \boldsymbol { y } _ { i } \} _ { i \in [ n ] }$ . It requires only a quasi-convex score on candidate means from Section 3, implemented here through low-degree pseudoexpectations for the diagonal-majorizer system.

Theorem 4.1 (Pure-DP Estimation from Discrete Batches). Let $\mu \in \Delta ^ { k - 1 }$ . Let �<sub>,</sub> $\begin{array} { r } { \beta \in ( 0 , 1 ) , \varepsilon > 0 , } \end{array}$ , and $m \in \mathbb { N }$ . Then there exists a polynomial-time $( \varepsilon , 0 ) – D P$ algorithm that takes in � independent user batches, each of the form

$$
B _ { i } = ( U _ { i , 1 } , \ldots , U _ { i , m } ) , \qquad U _ { i , 1 } , \ldots , U _ { i , m } \stackrel { \mathrm { i i d } } { \sim } \mu ,
$$

and outputs an estimate $\hat { \mu }$ such that $\| { \hat { \boldsymbol { \mu } } } - { \boldsymbol { \mu } } \| _ { \mathrm { T V } } \leqslant { \boldsymbol { \alpha } } ,$ , with probability at least $1 - \beta ,$ provided that

$$
n \geqslant \tilde { O } \left( \frac { k + \log ( 1 / \beta ) } { \alpha ^ { 2 } m } + \frac { k + \log ( 1 / \beta ) } { \alpha \sqrt { m } \varepsilon } + \frac { k + \log ( 1 / \beta ) } { \varepsilon } \right) .
$$

Theorem 4.2 (Approximate-DP Estimation from Discrete Batches). Let $\mu \in \Delta ^ { k - 1 }$ . Let $\alpha , \beta , \delta \in ( 0 , 1 )$ $\varepsilon > 0$ , and $m \in \mathbb { N } .$ . Then there exists a polynomial-time $( \varepsilon , \delta ) – D P$ algorithm that takes in � independent user batches, each of the form

$$
B _ { i } = ( U _ { i , 1 } , \ldots , U _ { i , m } ) , \qquad U _ { i , 1 } , \ldots , U _ { i , m } \stackrel { \mathrm { i i d } } { \sim } \mu ,
$$

and outputs an estimate $\hat { \mu }$ such that $\| { \hat { \boldsymbol { \mu } } } - \mu \| _ { \mathrm { T V } } \leqslant { \boldsymbol { \alpha } } .$ , with probability at least $1 - \beta ,$ provided that

$$
n \geqslant \tilde { O } \left( k + \frac { k + \log ( 1 / \beta ) } { \alpha ^ { 2 } m } + \frac { k + \log ( 1 / \beta ) } { \alpha \sqrt { m } \varepsilon } + \frac { \log ( 1 / \delta ) + \log ( 1 / \beta ) } { \varepsilon } \right) .
$$

Both privacy arguments use one score function. For the approximate-DP theorem, we also need a worst-case bound on the intermediate ratio $V _ { \eta ^ { * } } ( { \boldsymbol { y } } ) / V _ { 0 . 8 \eta ^ { * } } ( { \boldsymbol { y } } )$ ; we obtain it from an $\ell _ { 1 } { \mathrm { - d i a m e t e r } }$ bound for the score-�<sup>∗</sup>� set.

Our score function reuses the succinct diagonal-majorizer system introduced in Section 3.1. For a dataset $\boldsymbol { \mathcal { Y } } = \{ \boldsymbol { y } _ { i } \} _ { i \in [ n ] }$ and trimming budget $T _ { \mathbf { \delta } }$ , let $\mathcal { B } _ { T } ( y )$ denote the system from Section 3.1 on the observed batch averages $y ,$ , with the selector/data-consistency, simplex, and covariance-majorizer blocks of $\mathcal { A } _ { \eta } ,$ but with the counting constraint changed to

$$
\sum _ { i = 1 } ^ { n } w _ { i } \geqslant n - T .
$$

Relative to $\mathcal { A } _ { \eta } ,$ , this is the only structural change: the certificate may now discard up to $T$ observed batch averages. Write

$$
\widehat { \Sigma } ^ { \prime } : = \mathbb { E } ( y _ { i } ^ { \prime } - \mu ^ { \prime } ) ( y _ { i } ^ { \prime } - \mu ^ { \prime } ) ^ { \top } , \qquad \Sigma ( \nu ) : = \frac { 1 } { m } \big ( \mathrm { d i a g } ( \nu ) - \nu \nu ^ { \top } \big ) , \qquad \tau _ { \mathrm { c o v } } : = C \frac { \eta \log ( 1 / \eta ) } { m } ,
$$

for the same suficiently large universal constant � as in Section 3.1. The selector/data-consistency block decides which observed batch averages are kept and ties the repaired points to the observed data on those indices. Simplex constraints keep the repaired points and the pseudo-mean inside the simplex, while the covariance-majorizer block enforces the subset-direction variance control in Lemma 3.7.

Given a query point �, we will consider degree-6 pseudoexpectations that satisfy

1. $\widetilde { \mathbb { E } } \left| = \mathcal { B } _ { T } ( \mathcal { Y } ) \right.$ at degree 6.

2. Closeness constraint: $\| \widetilde { \mathbb { E } } \mu ^ { \prime } - \nu \| _ { 1 } \leqslant \alpha .$

Informally, the score of a candidate mean is the smallest trimming budget for which there exists a low-degree certificate consistent with that candidate. We first define certifiable means, then use them to define the score.

Definition 4.3 $( ( \alpha , \tau , T )$ -Certifiable Mean). Let $\alpha , \tau \geqslant 0 , k , n \in \mathbb { N } , T \in [ 0 , n ]$ . Let $\begin{array} { r } { y = \left\{ y _ { i } \right\} _ { i \in [ n ] } \subseteq \Delta ^ { k - 1 } } \end{array}$ We call a point $\nu \in \Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \tau n + \alpha ) \subseteq \mathbb { R } ^ { k }$ an $( \alpha , \tau , T )$ certifiable mean for � if there exists a linear functional ℒ which �-approximately (according to Definition 2.10) satisfies $\mathcal { B } _ { T } ( y )$ at degree 6 such that, writing $\mu _ { \mathcal { L } } : = \mathcal { L } \mu ^ { \prime } ,$

1. $\| \mathcal { R } ( \mathcal { L } ) \| _ { \mathrm { F } } \leqslant R ^ { \prime } + \tau ,$ , where $R ^ { \prime } = \mathrm { p o l y } ( n , k )$ is suficiently large and $\mathcal { R } ( \mathcal { L } )$ is the matrix representation of the coeficients of $\mathcal { L }$

2. The query point � is close to the assignment given by the approximate pseudoexpectation $\mathcal { L } \colon$

$$
\| \mu _ { \mathcal { L } } - \nu \| _ { 1 } \leqslant \alpha + \tau T .
$$

Moreover, we will call such a functional ℒ an $\left( \alpha , \tau , T \right) \ – c e r t i f i c a t e f o r \left( \mathcal { Y } , \nu \right)$

Lemma 4.4 (Approximate Certificates Are Close to the Simplex). Let $\mathcal { L }$ be a linear functional that �-approximately satisfies $\mathcal { B } _ { T } ( y )$ , and let

$$
x : = \mathcal { L } \mu ^ { \prime } \in \mathbb { R } ^ { k } .
$$

Then for every $j \in [ k ]$

$$
x _ { j } \geqslant - \tau T , \qquad \left| \sum _ { j = 1 } ^ { k } x _ { j } - 1 \right| \leqslant \tau T .
$$

Consequently, there exists a point $\Pi ( x ) \in { \Delta } ^ { k - 1 }$ such that

$$
\| x - \Pi ( x ) \| _ { 1 } \leqslant C k \tau T
$$

for an absolute constant �.

Proof. For the coordinate lower bound, apply item 3 of Definition 2.10 to the constraint $\mu _ { j } ^ { \prime } \geqslant 0$ with $p = 1$ . This gives

$$
x _ { j } = \mathcal { L } \mu _ { j } ^ { \prime } \geqslant - \tau T
$$

for every $j \in [ k ]$ . For the constraint $\begin{array} { r } { \sum _ { j \in [ k ] } \mu _ { j } ^ { \prime } - 1 = 0 } \end{array}$ , item 4 of Definition 2.10, with $p = 1$ and $\mathcal { L } 1 = 1$ , gives

$$
\left| \sum _ { j = 1 } ^ { k } x _ { j } - 1 \right| = \left| { \mathcal { L } } \left[ \sum _ { j = 1 } ^ { k } \mu _ { j } ^ { \prime } - 1 \right] \right| \leqslant \tau T .
$$

Set $y _ { j } : = \operatorname* { m a x } \{ x _ { j } , 0 \}$ for every $j \in [ k ]$ . Then

$$
\| x - y \| _ { 1 } = \sum _ { j = 1 } ^ { k } \operatorname* { m a x } \{ - x _ { j } , 0 \} \leqslant k \tau T .
$$

Also,

$$
1 - \tau T \leqslant \sum _ { j = 1 } ^ { k } x _ { j } \leqslant \sum _ { j = 1 } ^ { k } y _ { j } \leqslant \sum _ { j = 1 } ^ { k } x _ { j } + k \tau T \leqslant 1 + ( k + 1 ) \tau T .
$$

Let $\textstyle s : = \sum _ { j = 1 } ^ { k } y _ { j }$ , and define

$$
\Pi ( x ) : = { \left\{ \begin{array} { l l } { y / s } & { { \mathrm { i f ~ } } s > 0 , } \\ { e _ { 1 } } & { { \mathrm { i f ~ } } s = 0 , } \end{array} \right. }
$$

where $e _ { 1 }$ is the first standard basis vector. Then $\Pi ( x ) \in { \Delta } ^ { k - 1 }$ . If $s > 0$ , then

$$
\| y - \Pi ( x ) \| _ { 1 } = \sum _ { j = 1 } ^ { k } y _ { j } \bigg | 1 - \frac { 1 } { s } \bigg | = | s - 1 | \leqslant ( k + 1 ) \tau T .
$$

I $\therefore s = 0 .$ , then $y = 0$ and hence $x _ { j } \leqslant 0$ for every �. Together with $\begin{array} { r } { \left| \sum _ { j } x _ { j } - 1 \right| \leqslant \tau T } \end{array}$ , this implies $\tau T \geqslant 1$ and therefore

$$
\| x - \Pi ( x ) \| _ { 1 } = \| x - e _ { 1 } \| _ { 1 } \leqslant 1 + k \tau T \leqslant 2 k \tau T .
$$

For $s > 0$

$$
\| x - \Pi ( x ) \| _ { 1 } \leqslant \| x - y \| _ { 1 } + \| y - \Pi ( x ) \| _ { 1 } \leqslant ( 2 k + 1 ) \tau T \leqslant 3 k \tau T .
$$

So the claim holds with $C = 3$

Henceforth choose

$$
0 < \tau \leqslant \frac { \operatorname* { m i n } \{ \alpha , 1 / \sqrt { m } \} } { 1 0 0 C k n } .\tag{12}
$$

This bound permits an inverse-polynomial choice of $\tau ,$ which we may decrease further to meet the numerical error bound in Lemma 4.11. We now define the score using Definition 4.3.

Definition 4.5 (Score Function). Let $\Delta ^ { k - 1 }$ be the probability simplex (see Definition 2.1). Consider $\Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \tau n + \alpha ) \subseteq \mathbb { R } ^ { k }$ , the fattening of $\Delta ^ { k - 1 }$ by the $\ell _ { 1 }$ -ball of radius $\tau n + \alpha$ in � dimensions. Let $\alpha , \tau \geqslant 0 ,$ $\boldsymbol { y } = \left\{ \boldsymbol { y } _ { i } \right\} _ { i \in [ n ] } \subseteq \Delta ^ { k - 1 }$ , and $\nu \in \Delta ^ { k - 1 } { + } \mathbb { B } _ { 1 } ^ { k } ( \tau n { + } \alpha )$ . We define the score function $S : \Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \tau n + \alpha )  \mathbb { R } _ { \geqslant 0 }$ (viewed as a function of �) as

$$
S ( \nu ; \boldsymbol { \mathcal { Y } } , \alpha , \tau ) : = \operatorname* { m i n } _ { T \geqslant 0 } s u c h t h a t \nu i s a \left( \alpha , \tau , T \right) c e r t i f i a b l e m e a n f o r \boldsymbol { \mathcal { Y } } .
$$

For the rest of the section, we fix $\alpha , \tau ,$ , and the dataset $y ,$ and abbreviate

$$
S y ( \nu ) : = S ( \nu ; y , \alpha , \tau ) .
$$

We next verify the score properties required by Hopkins, Kamath, Majid, and Narayanan [HKMN23]: well-definedness, sensitivity, quasi-convexity, eficient computation, a low-scoring neighborhood, and bounds on sublevel-set volumes.

## 4.1 Properties of the Score

## 4.1.1 Well-Definedness of the Score Function

Lemma 4.6 (score function upper bound). Let $\alpha , \tau \geqslant 0 , y = \left\{ y _ { i } \right\} _ { i \in [ n ] }$ be any set of inputs, and $\nu \in \Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \tau n + \alpha )$ . Then $S y ( \nu ) \leqslant n$ , where � is defined in Definition 4.5.

Proof. Let � be any point in $\Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \tau n + \alpha )$ . It sufices to find an $( \alpha , \tau , T )$ certificate $\mathcal { L }$ for $( \boldsymbol { y } , \nu )$ where $T = n$

Let $\mu _ { 0 } : = \mathrm { P r o j } _ { \ell _ { 1 } } ( \nu ; \Delta ^ { k - 1 } )$ , so that

$$
\| \nu - \mu _ { 0 } \| _ { 1 } \leqslant \alpha + \tau n = \alpha + \tau T
$$

because $\nu \in \Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \alpha + \tau n )$ . Now let $\mathcal { L }$ be the linear functional corresponding to the single-point distribution that sets

$$
\forall i \in [ n ] : \qquad w _ { i } = 0 , \qquad y _ { i } ^ { \prime } = \mu _ { 0 } , \qquad \mu ^ { \prime } = \mu _ { 0 } , \qquad \lambda = 0 .
$$

All constraints except the diagonal-majorizer equality are then immediate. Since every $y _ { i } ^ { \prime } = \mu _ { 0 }$ we have $\widehat { \Sigma } ^ { \prime } = 0$ . Because $\mu _ { 0 } \in \bar { \Delta } ^ { k - 1 }$ , Fact 2.3 implies that $\Sigma ( \mu _ { 0 } )$ is positive semidefinite, so we may choose � such that

$$
\boldsymbol { M } \boldsymbol { M } ^ { \intercal } = \boldsymbol { \Sigma } ( \mu _ { 0 } ) .
$$

Thus

$$
\widehat { \Sigma } ^ { \prime } + M M ^ { \top } = \Sigma ( \mu _ { 0 } ) = \Sigma ( \mu ^ { \prime } ) + \mathrm { D i a g } ( \lambda ) , \qquad \sum _ { j \in [ k ] } \lambda _ { j } = 0 \leqslant \tau _ { \mathrm { c o v } } .
$$

Since the entries of $\mu _ { 0 }$ and � have absolute value at most 1, the degree-6 moment vector has polynomial norm and satisfies item 1 of Definition 4.3 for suficiently large $R ^ { \prime }$ . By construction, the closeness constraint holds, so $S y ( \nu ) \leqslant n$ . Compactness of the feasible pairs $( T , \mathcal { L } )$ ensures that the minimum in Definition 4.5 is attained □

## 4.1.2 Bounded Sensitivity

Lemma 4.7 (Bounded Sensitivity). For every fixed $\nu \in \Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \tau n + \alpha )$ , the score function $y \mapsto$ $S ( \nu ; y , \alpha , \tau )$ has sensitivity 1 with respect to changing the input data $\dot { \boldsymbol { \mathscr { y } } } = \left\{ \boldsymbol { y } _ { i } \right\} _ { i \in [ n ] } .$

Proof. Suppose $\pmb { y } = \left\{ y _ { 1 } , \ldots , y _ { n } \right\}$ and $y ^ { \prime } = \left\{ { \bar { y } } _ { 1 } , y _ { 2 } , \dots , y _ { n } \right\}$ are two datasets that difer in a single batch average. Moreover, assume that for some $\nu \in \Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \tau n + \alpha ) , S y ( \nu ) = T$ . This means that there exists an $( \alpha , \tau , T )$ certificate $\mathcal { L }$ for $( \boldsymbol { y } , \nu )$ . We will show that there exists a $( \alpha , \tau , T + 1 )$ certificate $\mathcal { L } ^ { \prime }$ for $( \boldsymbol { y } ^ { \prime } , \boldsymbol { \nu } )$

We define $\mathcal { L } ^ { \prime }$ as follows. For every monomial $p$ that does not involve $w _ { 1 } ,$ , we set $\mathcal { L } ^ { \prime } ( p ) = \mathcal { L } ( p )$ For monomials � that involve $w _ { 1 , }$ , we set $\mathcal { L } ^ { \prime } ( p ) = 0$

If $T + 1 \geqslant n ,$ , the claim follows from Lemma 4.6, so suppose $T + 1 < n$ . Substitution sets moment coordinates to zero, and hence cannot increase $\Vert \mathcal { R } ( \mathcal { L } ) \Vert _ { \mathrm { F } }$ . It preserves normalization and the pseudo-mean. For a polynomial $p$ with $\| \mathcal { R } ( p ) \| _ { 2 } \leqslant 1$ , write $q = p | _ { w _ { 1 } = 0 } ,$ , so $\| \mathcal { R } ( q ) \| _ { 2 } \leqslant 1$ . If a constraint polynomial � does not involve $w _ { 1 }$ , then

$$
\ { \mathcal { L } } ^ { \prime } [ s p ^ { 2 } ] = { \mathcal { L } } [ s q ^ { 2 } ] .
$$

Thus the constraints independent of $w _ { 1 }$ , square positivity, and the closeness constraint retain their error bounds at budget $T$ . The Boolean and data-consistency constraints involving $w _ { 1 }$ become identically zero. For the counting constraint in item 5 of Definition 2.10,

$$
\begin{array} { r l } { \mathcal { L } ^ { \prime } \Bigg [ \Bigg ( \displaystyle \sum _ { i = 1 } ^ { n } w _ { i } - n + T + 1 \Bigg ) p ^ { 2 } \Bigg ] } & { = \mathcal { L } \Bigg [ \Bigg ( \displaystyle \sum _ { i \neq 1 } w _ { i } - n + T + 1 \Bigg ) q ^ { 2 } \Bigg ] } \\ & { = \mathcal { L } \Bigg [ \Bigg ( \displaystyle \sum _ { i = 1 } ^ { n } w _ { i } - n + T \Bigg ) q ^ { 2 } \Bigg ] + \mathcal { L } [ ( 1 - w _ { 1 } ) ^ { 2 } q ^ { 2 } ] + \mathcal { L } [ w _ { 1 } ( 1 - w _ { 1 } ) q ^ { 2 } ] } \\ & { \geqslant - 5 \tau T n - 4 \tau T - \tau T } \\ & { \geqslant - 5 \tau ( T + 1 ) n . } \end{array}
$$

Here deg $q \leqslant 2 ,$ , so every expression has degree at most 6. The square term $\mathcal { L } [ ( 1 - w _ { 1 } ) ^ { 2 } q ^ { 2 } ]$ is at least $- 4 \tau T$ because $\| \mathcal { R } ( ( 1 - w _ { 1 } ) q ) \| _ { 2 } \leqslant 2 ;$ the term $\mathcal { L } [ w _ { 1 } ( 1 - w _ { 1 } ) q ^ { 2 } ]$ is at least $- \tau T$ by item 4 of Definition 2.10, applied to $w _ { 1 } ^ { 2 } - w _ { 1 } = 0$ . Interchanging � and $y \prime$ proves the sensitivity bound. □

## 4.1.3 Quasi Convexity

Lemma 4.8 (Quasi Convexity). For every fixed dataset � , the score function $\nu \mapsto S y ( \nu )$ is quasi-convex on $\Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \tau n + \alpha )$

Proof. Let $\nu _ { 1 } , \nu _ { 2 } \in \Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \tau n + \alpha )$ and $T _ { 1 } , T _ { 2 } \in [ 0 , n ]$ with $S y ( \nu _ { 1 } ) = T _ { 1 }$ and $S y ( \nu _ { 2 } ) = T _ { 2 }$ . For any $\lambda \in [ 0 , 1 ]$ and $T = S y ( \lambda \nu _ { 1 } \bar { + } ( 1 - \lambda ) \nu _ { 2 } )$ , we want to show that $T \leqslant \mathrm { m a x } \{ T _ { 1 } , T _ { 2 } \}$ . By definition 4.5, we get that $\nu _ { 1 }$ is an $( \alpha , \tau , T _ { 1 }$ )-certifiable mean for � , and an analogous property holds for $\nu _ { 2 }$ . Choose certificates $\mathcal { L } _ { 1 } , \mathcal { L } _ { 2 }$ for $( y , \nu _ { 1 } )$ and $( y , \nu _ { 2 } )$ at budgets $T _ { 1 } , T _ { 2 }$ , respectively, as in Definition 4.3. We construct $\begin{array} { r } { \mathcal { L } = \lambda \mathcal { L } _ { 1 } + ( 1 - \lambda ) \mathcal { L } _ { 2 } } \end{array}$ , and show that it certifies $\lambda \nu _ { 1 } + ( 1 - \lambda ) \nu _ { 2 }$ at budget max $\{ T _ { 1 } , T _ { 2 } \}$ . The moment-norm bound, the closeness constraint, and all constraints independent of � are convex in $\mathcal { L }$ and their permitted errors increase with �. It remains to check the counting constraint in item 5 of Definition 2.10. Let � be any polynomial with $\| \mathcal { R } ( p ) \| _ { 2 } \leqslant 1$ . Assuming that $T _ { 1 } \leqslant T _ { 2 }$ , we have that:

$$
\begin{array} { r l } & { \quad \mathcal { L } \Bigg [ \Bigg | \Bigg ( \displaystyle \sum _ { i = 1 } ^ { n } w _ { i } + T _ { 2 } - n \Bigg ) p ^ { 2 } \Bigg ] } \\ & { = \lambda \mathcal { L } _ { 1 } \Bigg [ \displaystyle \sum _ { i = 1 } ^ { n } w _ { i } + T _ { 1 } + ( T _ { 2 } - T _ { 1 } ) - n \Bigg | p ^ { 2 } \Bigg ] + ( 1 - \lambda ) \mathcal { L } _ { 2 } \Bigg [ \Bigg ( \displaystyle \sum _ { i = 1 } ^ { n } w _ { i } + T _ { 2 } - n \Bigg ) p ^ { 2 } \Bigg ] } \\ & { = \lambda \mathcal { L } _ { 1 } \Bigg [ \displaystyle \left[ \displaystyle \sum _ { i = 1 } ^ { n } w _ { i } + T _ { 1 } - n \Bigg | p ^ { 2 } \right] + \lambda ( T _ { 2 } - T _ { 1 } ) \mathcal { L } _ { 1 } p ^ { 2 } + ( 1 - \lambda ) \mathcal { L } _ { 2 } \Bigg [ \left( \displaystyle \sum _ { i = 1 } ^ { n } w _ { i } + T _ { 2 } - n \right) p ^ { 2 } \Bigg ] } \\ & { \geqslant \lambda ( - 5 \tau T _ { 1 } n ) + \lambda ( T _ { 2 } - T _ { 1 } ) ( - 5 \tau T _ { 1 } ) + ( 1 - \lambda ) ( - 5 \tau T _ { 2 } n ) } \\ & { \geqslant \lambda ( - 5 \tau T _ { 1 } n ) + \lambda [ - 5 \tau n ( T _ { 2 } - T _ { 1 } ) ] + ( 1 - \lambda ) ( - 5 \tau T _ { 2 } n ) } \\ & { \geqslant - 5 \tau T _ { 2 } n , } \end{array}
$$

where we used that $T _ { 1 } \leqslant n$ (lemma 4.6) in the second-to-last inequality.

## 4.1.4 Eficient Computability

Lemma 4.9 (Eficient Computation of Score Function). Let � be any point in $\Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \tau n + \alpha )$ . Let $T = S _ { y } ( \nu )$ be its score. Then we can compute � to additive accuracy � in time polynomial in �, �, log(1/�),

log $( 1 / \tau )$ , and the input bit length.

Proof. There are $\mathrm { { p o l y } } ( n , k )$ moments of degree at most 6. At fixed $T ,$ approximate positivity and the inequality constraints are positive-semidefinite constraints on moment and localizing matrices; each approximate equality is a pair of such constraints. Polynomial-time separation also applies to the moment-norm and ℓ<sub>1</sub>-closeness constraints. Feasible sets are nested in $T ,$ as follows from the counting-constraint calculation in Lemma 4.8.

For completeness, after fixing the normalization $\mathcal { L } 1 = 1$ , the positive tolerance supplies the interior needed for weak feasibility. Increasing � by � increases the permitted error in each constraint by at least $\tau \gamma ;$ the counting constraint retains slack of order $\tau n \gamma ,$ using $T \leqslant n$ and $\mathcal { L } p ^ { 2 } \geqslant - \tau T$ . If the moment-norm bound is tight, mix the functional with a suficiently small multiple of the bounded point-mass functional from Lemma 4.6. Taking $R ^ { \prime }$ larger than twice the norm of that functional creates norm slack, while changing every other constraint by at most a fixed polynomial times the mixing weight. A mixing weight of order $\tau \gamma / \mathrm { p o l y } ( n , k )$ therefore leaves a ball of radius $\tau \gamma / \mathrm { p o l y } ( n , k )$ in the feasible set at the enlarged budget. We apply the ellipsoid argument of Appendix C in [HKMN23], followed by binary search in $T ,$ to compute the score to additive error �. Its bit complexity is polynomial also in the input bit length and $\log ( 1 / \tau ) ;$ throughout, � is inverse-polynomial in the problem parameters. □

## 4.2 Problem-Specific Consequences

## 4.2.1 Eficiently Finding Low Scoring Points

Lemma 4.10. There exists an algorithm, running in time polynomial in $n , k , \log ( 1 / \tau )$ , and the input bit length, that outputs $\hat { \mu }$ such that

$$
\begin{array} { r } { S y ( \hat { \mu } ) \leqslant \underset { \nu \in \Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \tau n + \alpha ) } { \operatorname* { m i n } } S y ( \nu ) + 1 , } \end{array}
$$

and every point within $\ell _ { 1 }$ distance $\alpha / 2$ of the returned $\hat { \mu }$ has score at most min $\mathsf { l } _ { \nu \in \Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \tau n + \alpha ) } S _ { y } ( \nu ) + 1$

Proof. Consider the certificate constraints in Definition 4.3 with the closeness constraint in item 2 omitted. Without the closeness constraint, we may still apply the weak-feasibility argument in Lemma 4.9. Binary search, with an additive budget allowance smaller than 1, returns a feasible budget $T _ { \mathrm { m i n } }$ and a corresponding functional $\mathcal { L }$ with

$$
T _ { \operatorname* { m i n } } \leqslant \operatorname* { m i n } _ { \substack { \nu \in \Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } \left( \tau n + \alpha \right) } } S y ( \nu ) + 1 .
$$

Indeed, dropping the closeness constraint can only decrease the optimum. Define

$$
\mu _ { \mathcal { L } } : = \mathcal { L } \mu ^ { \prime } , \qquad \hat { \mu } : = \Pi ( \mu _ { \mathcal { L } } ) .
$$

By Lemma 4.4 and (12), we have

$$
\begin{array} { r } { \| \hat { \mu } - \mu _ { \mathcal { L } } \| _ { 1 } \leqslant C k \tau T _ { \operatorname* { m i n } } \leqslant C k \tau n \leqslant \alpha / 1 0 0 \leqslant \alpha / 1 0 . } \end{array}
$$

In particular, $\hat { \mu } \in \Delta ^ { k - 1 } \subseteq \Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \tau n + \alpha )$ . Thus $\mathcal { L }$ certifies that $\hat { \mu }$ has score at most $T _ { \mathrm { m i n } } ,$ , since

$$
\| \hat { \mu } - \mu _ { \mathcal { L } } \| _ { 1 } \leqslant \alpha / 1 0 \leqslant \alpha + \tau T _ { \operatorname* { m i n } } .
$$

Now let $\nu$ be any point with $\| \nu - \hat { \mu } \| _ { 1 } \leqslant \alpha / 2$ . Then

$$
\| \nu - \mu _ { \mathcal { L } } \| _ { 1 } \leqslant \| \nu - \hat { \mu } \| _ { 1 } + \| \hat { \mu } - \mu _ { \mathcal { L } } \| _ { 1 } \leqslant \frac { \alpha } { 2 } + \frac { \alpha } { 1 0 } < \alpha .
$$

Hence $\mathcal { L }$ also witnesses that � has score at most $T _ { \mathrm { m i n } }$

Both score bounds follow from $\begin{array} { r } { T _ { \operatorname* { m i n } } \leqslant \operatorname* { m i n } _ { \nu \in \Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \tau n + \alpha ) } S y ( \nu ) + 1 . } \end{array}$

Lemma 4.11 (Soundness of Low-Trimming Certificates). Let �<sub>0</sub> $\in \ ( 0 , \eta _ { \mathrm { a b s } } )$ . Assume $\tau _ { \mathrm { c o v } } ~ \leqslant$ $C \eta _ { 0 } \log ( 1 / \eta _ { 0 } ) / m$ . Suppose $y ^ { * } = \{ y _ { i } ^ { * } \} _ { i \in [ n ] }$ is � -good, and $\boldsymbol { \mathcal { Y } } = \{ \boldsymbol { y } _ { i } \} _ { i \in [ n ] }$ is an �<sub>0</sub>-corruption of $y ^ { * }$ Let $\mathcal { L }$ be an $( \alpha , \tau , T )$ -certificate for $( \boldsymbol { y } , \nu )$ with $T \leqslant \eta _ { 0 } n$ , and write $\mu _ { \mathcal { L } } : = \mathcal { L } \mu ^ { \prime }$ . Then

$$
\| \mu _ { \mathcal { L } } - \mu \| _ { 1 } \leqslant O \left( \eta _ { 0 } \sqrt { \frac { \log ( 1 / \eta _ { 0 } ) } { m } } \right) .
$$

Proof. For exact certificates, this is exactly the deterministic soundness proof from lemma 3.8 and the proof of Theorem 3.1, with trimming budget $T \leqslant \eta _ { 0 } n$ in place of the robust theorem’s corruption budget. For approximate certificates, each inference in this fixed degree-6 proof incurs at most $\tau \mathrm { p o l y } ( n , k , m , \bar { \eta } _ { 0 } ^ { - 1 } ) T$ coeficient error. We choose the inverse-polynomial tolerance � suficiently small that

$$
\tau \ \mathrm { p o l y } ( n , k , m , \eta _ { 0 } ^ { - 1 } ) T = o \left( \frac { \eta _ { 0 } ^ { 2 } \log ( 1 / \eta _ { 0 } ) } { m } \right) .
$$

Thus the perturbation is lower order than the squared directional-error bound in Lemma 3.8. Applying that directional bound in both subset directions and using $| \textstyle \sum _ { j } ( \mu _ { \mathcal { L } } ) _ { j } - 1 | \leqslant \tau T$ from Lemma 4.4 gives the claimed $\ell _ { 1 }$ bound. □

## 4.2.2 Volume

Lemma 4.12 (Volume). Assume $n \geqslant \Omega ( ( k + \log ( 1 / \beta ) ) / ( \eta ^ { 2 } \log ( 1 / \eta ) ) )$ . Let $\eta ^ { * } \geqslant \eta$ be a suficiently small absolute constant. $L e t$

$$
B _ { i } ^ { * } = ( U _ { i , 1 } ^ { * } , \ldots , U _ { i , m } ^ { * } ) , \qquad U _ { i , 1 } ^ { * } , \ldots , U _ { i , m } ^ { * } \overset { \mathrm { i i d } } { \sim } \mu , \qquad y _ { i } ^ { * } : = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } e _ { U _ { i , j } ^ { * } }
$$

for each $i \in [ n ]$ , and let $\boldsymbol { y ^ { * } } = \left\{ \boldsymbol { y } _ { 1 } ^ { * } , \ldots , \boldsymbol { y } _ { n } ^ { * } \right\}$ . Let $\boldsymbol { \mathcal { Y } } = \left\{ \boldsymbol { y } _ { i } \right\} _ { i \in [ n ] }$ be any �-corruption of $y ^ { * }$ . The following three statements hold with probability $1 - \beta$ over the draw of $\bar { y ^ { * } }$

1. There exists a point $\mu _ { 0 }$ such that every point at $\ell _ { 1 }$ distance at most � from $\mu _ { 0 }$ has score at most $\eta n$

2. All points which have score at most �<sup>∗</sup>� are contained in an $\ell _ { 1 }$ -ball of radius $\alpha + O ( 1 / \sqrt { m } )$ centered at the same point $\mu _ { 0 }$

3. Simultaneously for every $\eta ^ { \prime } \in \left[ \eta , \eta ^ { * } \right]$ , all points of score at most $\eta ^ { \prime }$ � lie in the $\ell _ { 1 }$ -ball about $\mu _ { 0 }$ of radius

$$
\alpha + O \left( \eta ^ { \prime } \sqrt { \frac { \log ( 1 / \eta ^ { \prime } ) } { m } } \right) .
$$

Proof. Take $\mu _ { 0 }$ to be the empirical average of the true samples, $\bar { \mu } .$ . For each integer $0 \leqslant j \leqslant \lceil \log _ { 2 } ( \eta ^ { * } / \eta ) \rceil .$ apply Corollary 3.3 at min $\bar { \{ 2 ^ { j } \eta , \eta ^ { * } \} }$ with failure probability $\beta / 2 ^ { j + 2 }$ . These events hold simultaneously with probability at least $1 - \beta$ . To check the sample size, observe that

$$
( 2 q ) ^ { 2 } \log ( 1 / ( 2 q ) ) \geqslant 2 q ^ { 2 } \log ( 1 / q ) \qquad ( 0 < q \leqslant 1 / 8 ) .
$$

For the final, possibly shorter step, we therefore obtain

$$
\operatorname* { m i n } \{ 2 ^ { j } \eta , \eta ^ { * } \} ^ { 2 } \log \left( \frac { 1 } { \operatorname* { m i n } \{ 2 ^ { j } \eta , \eta ^ { * } \} } \right) \geqslant 2 ^ { j - 1 } \eta ^ { 2 } \log ( 1 / \eta ) \qquad ( j \geqslant 1 ) .
$$

Since $k + \log ( 2 ^ { j + 2 } / \beta ) = O ( k + \log ( 1 / \beta ) + j )$ and $j + 1 = O ( 2 ^ { j } )$ , the sample-size hypothesis of this lemma supplies the good-sample event at every level min $\{ 2 ^ { j } \eta , \eta ^ { * } \}$ . Work on their intersection. Let $r _ { i }$ be the indicator of the uncorrupted samples. By Corollary 3.6, there exists � such that

$$
\mathrm { D i a g } ( \lambda ) - \left( \mathbb { E } ( y _ { i } ^ { * } - \bar { \mu } ) ( y _ { i } ^ { * } - \bar { \mu } ) ^ { \top } - \frac { 1 } { m } \big ( \mathrm { d i a g } ( \bar { \mu } ) - \bar { \mu } \bar { \mu } ^ { \top } \big ) \right) \succcurlyeq 0
$$

and $\begin{array} { r } { \sum _ { j } \lambda _ { j } \leqslant O ( \eta \log ( 1 / \eta ) / m ) \leqslant \tau _ { \mathrm { c o v } } } \end{array}$ , where the last inequality holds by the choice of the constant in $\tau _ { \mathrm { c o v } }$ . Therefore, taking

$$
w _ { i } = r _ { i } , \qquad y _ { i } ^ { \prime } = y _ { i } ^ { * } , \qquad \mu ^ { \prime } = \bar { \mu } ,
$$

and letting � be a square root of the positive semidefinite matrix

$$
\mathrm { D i a g } ( \lambda ) - \left( \mathbb { E } ( y _ { i } ^ { * } - \bar { \mu } ) ( y _ { i } ^ { * } - \bar { \mu } ) ^ { \mathsf { T } } - \frac { 1 } { m } \big ( \mathrm { d i a g } ( \bar { \mu } ) - \bar { \mu } \bar { \mu } ^ { \mathsf { T } } \big ) \right) ,
$$

gives a feasible point-mass functional. The diagonal entries of the majorizer imply $\lambda _ { j } \geqslant - 1 / ( 4 m ) ;$ together with its trace bound, this bounds each $\lambda _ { j }$ and $\| M \| _ { \mathrm { F } }$ by poly(�). Thus its degree-6 moments also satisfy item 1 of Definition 4.3 for suficiently large $R ^ { \prime } { } _ { i }$ , and $\bar { \mu }$ is an $( \alpha , \tau , \eta n )$ -certifiable mean. Hence every � such that

$$
\| \nu - \bar { \mu } \| _ { 1 } \leqslant \alpha
$$

also has score at most ��, proving the first part.

For the third part, fix $\eta ^ { \prime } \in \left[ \eta , \eta ^ { \ast } \right]$ and choose the smallest � with min $\{ 2 ^ { j } \eta , \eta ^ { * } \} \geqslant \eta ^ { \prime } .$ , so that min $\{ 2 ^ { j } \eta , \eta ^ { * } \} \leqslant 2 \eta ^ { \prime }$ . The fixed covariance allowance obeys

$$
\tau _ { \mathrm { c o v } } = C \frac { \eta \log ( 1 / \eta ) } { m } \leqslant C \frac { \operatorname* { m i n } \{ 2 ^ { j } \eta , \eta ^ { * } \} \log ( 1 / \operatorname* { m i n } \{ 2 ^ { j } \eta , \eta ^ { * } \} ) } { m } .
$$

The dataset is a min $\{ 2 ^ { j } \eta , \eta ^ { * } \}$ -corruption of samples that are good at that level. Thus Lemma 4.11 applies to every certificate of score at most $\eta ^ { \prime } n ,$ giving

$$
\begin{array} { r l } & { \| \mathcal { L } \mu ^ { \prime } - \mu \| _ { 1 } \leqslant O \left( \operatorname* { m i n } \{ 2 ^ { j } \eta , \eta ^ { * } \} \sqrt { \frac { \log ( 1 / \operatorname* { m i n } \{ 2 ^ { j } \eta , \eta ^ { * } \} ) } { m } } \right) } \\ & { \qquad \leqslant O \left( \eta ^ { \prime } \sqrt { \frac { \log ( 1 / \eta ^ { \prime } ) } { m } } \right) . } \end{array}
$$

Choose � suficiently small for soundness at every grid point and so that $\tau n \leqslant \eta \sqrt { \log ( 1 / \eta ) / m }$ One inverse-polynomial choice sufices, since every level is at least $\eta .$ Property 1 of the $\eta { \mathrm { - g o o d } }$ event bounds $\| { \bar { \mu } } - \mu \| _ { 1 }$ by $O ( \eta \sqrt { \log ( 1 / \eta ) / m } )$ . Adding these bounds and the certificate’s closeness constraint proves the third part. Taking $\eta ^ { \prime } = \eta ^ { * }$ gives the second part. □

Lemma 4.13 (Lower Volume at Level $0 . 8 \eta ^ { * } n )$ . Let $\boldsymbol { \mathcal { Y } } = \left\{ \boldsymbol { y } _ { i } \right\} _ { i \in [ n ] }$ and let $\eta ^ { * }$ be a suficiently small absolute constant. Suppose $\mathcal { L } _ { 0 }$ is an $( \alpha , \tau , 0 . 7 \eta ^ { * } n )$ -certificate. Set

$$
\mu ^ { \circ } = \Pi ( \mathcal { L } _ { 0 } \mu ^ { \prime } ) .
$$

Every point � with $\| \nu - \mu ^ { \circ } \| _ { 1 } \leqslant \alpha - C$ ��� has score at most $0 . 8 \eta ^ { * } n$ , where $C$ is the constant in Lemma 4.4. Proof. The ball lies in $\Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \tau n + \alpha )$ because $\mu ^ { \circ } \in \Delta ^ { k - 1 }$ . By Lemma $4 . 4 ,$

$$
\| \nu - \mathcal { L } _ { 0 } \mu ^ { \prime } \| _ { 1 } \leqslant \| \nu - \mu ^ { \circ } \| _ { 1 } + \| \mu ^ { \circ } - \mathcal { L } _ { 0 } \mu ^ { \prime } \| _ { 1 } \leqslant \alpha - C k \tau n + C k \tau n = \alpha .
$$

Thus this certificate gives score at most $0 . 7 \eta ^ { * } n$ , and hence at most $0 . 8 \eta ^ { * } n$

Lemma 4.14 (Worst-Case $\ell _ { 1 }$ Diameter of the Score-�<sup>∗</sup>� Set). Let $\eta ^ { * }$ be a suficiently small absolute constant and let $\tau _ { \mathrm { c o v } } \geqslant 0 . \ I f \nu _ { 1 } , \nu _ { 2 } \in \Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \tau n + \alpha )$ both have score at most $\eta ^ { * } n$ , then

$$
\lVert \nu _ { 1 } - \nu _ { 2 } \rVert _ { 1 } \leqslant O \left( \alpha + \sqrt { \frac { 1 } { m } + \tau _ { \mathrm { c o v } } } \right) .
$$

More precisely, any corresponding certificates $\mathcal { L } _ { 1 } , \mathcal { L } _ { 2 }$ satisfy

$$
\| \mathcal { L } _ { 1 } \mu ^ { \prime } - \mathcal { L } _ { 2 } \mu ^ { \prime } \| _ { 1 } \leqslant O \left( \sqrt { \frac { 1 } { m } + \tau _ { \mathrm { c o v } } } \right) .
$$

Proof. See Section A for the full proof.

Corollary 4.15 (Intermediate Volume Ratio). Let $\begin{array} { r } { \boldsymbol { \mathcal { Y } } = \left\{ \boldsymbol { y } _ { i } \right\} _ { i \in [ n ] } , } \end{array}$ let $\eta ^ { * }$ be a suficiently small absolute constant, and let $\tau _ { \mathrm { c o v } } \geqslant 0$ . Suppose there is a point of score at most $\phantom { - } 0 . 7 \eta ^ { * } n$ . Then

$$
\log \left( \frac { V _ { \eta ^ { * } } ( y ) } { V _ { 0 . 8 \eta ^ { * } } ( y ) } \right) \leqslant O \left( k \log \left( 1 + \frac { \sqrt { 1 / m + \tau _ { \mathrm { c o v } } } } { \alpha } \right) \right) .
$$

Proof. Choose a certificate $\mathcal { L } _ { 0 }$ for a point of score at most $0 . 7 \eta ^ { * } n ,$ , and let $\mu ^ { \circ } = \Pi ( \mathcal { L } _ { 0 } \mu ^ { \prime } )$ . Let � be the constant in Lemma 4.4. By Lemma 4.13, the score-0<sub>.</sub>8�<sup>∗</sup>� set contains the $\ell _ { 1 }$ ball of radius $\alpha - C k \tau n$ centered at $\mu ^ { \circ }$ . For any point � of score at most $\eta ^ { * } n$ , choose a certificate $\mathcal { L }$ . The pseudo-mean diameter bound in Lemma 4.14 gives

$$
\begin{array} { r l } & { \| \nu - \mu ^ { \circ } \| _ { 1 } \leqslant \| \nu - \mathcal { L } \mu ^ { \prime } \| _ { 1 } + \| \mathcal { L } \mu ^ { \prime } - \mathcal { L } _ { 0 } \mu ^ { \prime } \| _ { 1 } + \| \mathcal { L } _ { 0 } \mu ^ { \prime } - \mu ^ { \circ } \| _ { 1 } } \\ & { \qquad \leqslant \alpha + \tau n + O \left( \sqrt { \displaystyle \frac { 1 } { m } + \tau _ { \mathrm { c o v } } } \right) + C k \tau n . } \end{array}
$$

Choosing � as in (12) ensures $C k \tau n \leqslant \mathrm { m i n } \{ \alpha / 1 0 0 , 1 / ( 1 0 0 \sqrt { m } ) \}$ and $\tau n \leqslant C k \tau n$ . Comparing volumes of the two $\ell _ { 1 }$ balls therefore gives

$$
\frac { V _ { \eta ^ { * } } ( y ) } { V _ { 0 . 8 \eta ^ { * } } ( J ) } \leqslant \left( \frac { \alpha + O \left( \sqrt { \frac { 1 } { m } + \tau _ { \mathrm { c o v } } } \right) + 2 C k \tau n } { \alpha - C k \tau n } \right) ^ { k } \leqslant \left( 1 + \frac { C ^ { \prime } \sqrt { 1 / m + \tau _ { \mathrm { c o v } } } } { \alpha } \right) ^ { k }
$$

for an absolute constant $C ^ { \prime } { } _ { - }$ , proving the claim.

Algorithm 4.1 (Private Estimation from Discrete Batches).

Input: User batches $\{ B _ { i } \} _ { i \in [ n ] } ,$ , target accuracy $\alpha ,$ failure probability $\beta ,$ and privacy parameters $( \varepsilon , 0 )$ or $( \varepsilon , \delta )$

Output: $\mathrm { A }$ private estimate $\hat { \mu }$ of the underlying distribution.

## Operations:

1. If $\varepsilon > 1$ , return the estimate from the pure-DP construction in Section 4.2.3.

2. If approximate DP is requested and $\delta > 1 - \beta / 2$ , return the empirical distribution of all �� samples with probability $\delta ,$ and the fixed distribution $e _ { 1 }$ otherwise.

3. Replace � by $\alpha / 2$ in the remaining steps.

4. Convert each observed batch to its average vector

$$
y _ { i } : = \frac 1 m \sum _ { j = 1 } ^ { m } e _ { U _ { i , j } } , \qquad y : = \left\{ y _ { i } \right\} _ { i \in [ n ] } .
$$

5. Define the score function $S _ { y } ( \nu )$ from Definition 4.5: the smallest trimming budget $T$ for which � admits an $( \alpha , \tau , T )$ )-certificate via the diagonal-majorizer system $\mathcal { B } _ { T } ( y )$

6. Set the internal trimming parameter

$$
\eta _ { \mathrm { p r i v } } : = \operatorname* { m i n } \bigl \{ \eta _ { \mathrm { a b s } } / 2 0 , \tilde { \Theta } ( \alpha \sqrt { m } ) \bigr \}
$$

and instantiate the score system with

$$
\tau _ { \mathrm { c o v } } : = { C } \frac { \eta _ { \mathrm { p r i v } } \log ( 1 / \eta _ { \mathrm { p r i v } } ) } { m } .
$$

Run the reduction from Theorem 2.18 (for pure DP) or Theorem 2.19 (for approximate DP) on the domain $\Theta = \Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \tau \bar { n } + \alpha )$ with score function $S _ { y }$

7. Output a nearest point in $\Delta ^ { k - 1 }$ to the returned point in $\ell _ { 1 }$ distance.

Algorithm 4.1 summarizes the private estimator obtained from the robustness-to-privacy reduction.

For the proofs of Theorems 4.1 and 4.2, fix $\eta ^ { * } = \eta _ { \mathrm { a b s } } / 2$ and instantiate the score system with the capped internal parameter

$$
\eta _ { \mathrm { p r i v } } : = \operatorname* { m i n } \bigl \{ \eta _ { \mathrm { a b s } } / 2 0 , ~ \tilde { \Theta } ( \alpha \sqrt { m } ) \bigr \} , \qquad \tau _ { \mathrm { c o v } } : = C \frac { \eta _ { \mathrm { p r i v } } \log ( 1 / \eta _ { \mathrm { p r i v } } ) } { m } ,
$$

so that $1 0 \eta _ { \mathrm { p r i v } } \leqslant \eta ^ { * }$ , and where the hidden constant in $\tilde { \Theta } ( \alpha \sqrt { m } )$ is chosen small enough that whenever $\eta _ { \mathrm { p r i v } } < \eta _ { \mathrm { a b s } } / 2 0$ , we have

$$
O \left( \eta _ { \mathrm { p r i v } } \sqrt { \frac { \log ( 1 / \eta _ { \mathrm { p r i v } } ) } { m } } \right) \leqslant \alpha / 1 0 .
$$

Equivalently, after fixing the hidden constant once and for all, there is an absolute split threshold such that:

1. if $\alpha \sqrt { m }$ is below that threshold, then $\eta _ { \mathrm { p r i v } } = \tilde { \Theta } ( \alpha \sqrt { m } ) ;$

2. if $\alpha \sqrt { m }$ is above that threshold, then $\eta _ { \mathrm { p r i v } } = \eta _ { \mathrm { a b s } } / 2 0$ , so $2 \eta _ { \mathrm { p r i v } } < 1 / 1 0$ and

$$
O \left( \eta _ { \mathrm { p r i v } } \sqrt { \frac { \log ( 1 / \eta _ { \mathrm { p r i v } } ) } { m } } \right) = O \left( \frac { 1 } { \sqrt { m } } \right) \leqslant \alpha / 1 0 .
$$

## 4.2.3 Proof of the Pure-DP Theorem

First suppose $0 < \varepsilon \leqslant 1$ . We allocate failure probability $\beta / 3$ to the $2 \eta _ { \mathrm { p r i v } } { - \xi }$ ood event, the volume event of Lemma 4.12, and the privacy mechanism; this changes only absolute constants and logarithmic factors. We apply Theorem 2.18, using the score function defined in Definition 4.5, with $\begin{array} { r } { \Theta = \Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( \tau n + \alpha ) } \end{array}$ , internal score threshold $\eta _ { \mathrm { p r i v } } ,$ and Euclidean radius $\alpha / ( 2 \sqrt { k } )$ . The $\ell _ { 1 }$ ball in Lemma 4.10 contains a Euclidean ball of this radius. All score properties required by Theorem 2.18 follow from Lemmas 4.6 to 4.10. It remains to verify the low-score witness, bound the volume term, and invoke soundness.

Both the good-sample event from Corollary 3.3 at parameter $2 \eta _ { \mathrm { p r i v } }$ and the event in Lemma 4.12 at parameter $\eta _ { \mathrm { p r i v } }$ hold once

$$
n \geqslant \Omega \left( \frac { k + \log ( 1 / \beta ) } { \eta _ { \mathrm { p r i v } } ^ { 2 } \log ( 1 / \eta _ { \mathrm { p r i v } } ) } \right) .\tag{13}
$$

In the below-threshold regime $\eta _ { \mathrm { p r i v } } = \tilde { \Theta } ( \alpha \sqrt { m } )$ , the requirement in (13) becomes

$$
\tilde { O } \left( \frac { k + \log ( 1 / \beta ) } { \alpha ^ { 2 } m } \right) .
$$

In the above-threshold regime $\eta _ { \mathrm { p r i v } } = \eta _ { \mathrm { a b s } } / 2 0$ , this requirement is only $O ( k + \log ( 1 / \beta ) )$ , which is absorbed by the $\widetilde { O } ( ( k + \log ( 1 / \beta ) ) / \varepsilon )$ term when $\varepsilon \leqslant 1$

For the volume term, let $\mu _ { 0 }$ be the point from the first part of Lemma 4.12 at parameter $\eta _ { \mathrm { p r i v } } .$ Because the witnessing certificate there has pseudo-mean exactly $\mu _ { 0 } ,$ every point within $\ell _ { 1 }$ distance � of $\mu _ { 0 }$ also has score at most $\eta _ { \mathrm { p r i v } } n$ . We use Lemma 2.20; the �-radius $\ell _ { 1 }$ ball about $\mu _ { 0 }$ contains a Euclidean ball of radius $\alpha / ( 2 \sqrt { k } )$ . Split the maximization in (5) into $\eta ^ { \prime } \in \left[ \eta _ { \mathrm { p r i v } } , \eta ^ { * } \right]$ and $\eta ^ { \prime } \in [ \eta ^ { * } , 1 ]$ For the first interval, item 3 of Lemma 4.12 gives

$$
\log \biggl ( \frac { V _ { \eta ^ { \prime } } ( y ) } { V _ { \eta _ { \mathrm { p i v } } } ( y ) } \biggr ) \leqslant k \log \biggl ( 1 + C \frac { \eta ^ { \prime } \sqrt { \log ( 1 / \eta ^ { \prime } ) } } { \alpha \sqrt { m } } \biggr ) \leqslant k \log \biggl ( 1 + C ^ { \prime } \frac { \eta ^ { \prime } } { \eta _ { \mathrm { p r i v } } } \biggr ) ,
$$

where the last step uses $\eta _ { \mathrm { p r i v } } \sqrt { \log ( 1 / \eta _ { \mathrm { p r i v } } ) } = O ( \alpha \sqrt { m } )$ . Using $\operatorname* { s u p } _ { u \geqslant 1 } \log ( 1 + C ^ { \prime } u ) / u = O ( 1 )$ , we obtain

$$
\operatorname* { m a x } _ { \eta _ { \mathrm { p r i v } } \leqslant \eta ^ { \prime } \leqslant \eta ^ { \ast } } \frac { \log ( V _ { \eta ^ { \prime } } ( \mathcal { Y } ) / V _ { \eta _ { \mathrm { p r i v } } } ( \mathcal { Y } ) ) + \log ( 2 / \beta ) } { \varepsilon \eta ^ { \prime } } \leqslant O \biggl ( \frac { k + \log ( 2 / \beta ) } { \varepsilon \eta _ { \mathrm { p r i v } } } \biggr ) .\tag{14}
$$

The right-hand side of (14) is $\widetilde { O } ( ( k + \log ( 2 / \beta ) ) / ( \alpha \sqrt { m } \varepsilon ) )$ below the split threshold, and $O ( ( k +$ log $( 2 / \beta ) / \varepsilon )$ above it. For $\eta ^ { \prime } \geqslant \eta ^ { * }$ , bounding $V _ { \eta ^ { \prime } }$ by the volume $V _ { 1 }$ of the full score domain gives

$$
\frac { \log ( V _ { 1 } ( \boldsymbol { \mathcal { Y } } ) / V _ { \eta _ { \mathrm { p r i v } } } ( \boldsymbol { \mathcal { Y } } ) ) + \log ( 2 / \beta ) } { \varepsilon \cdot \eta ^ { * } } \leqslant \tilde { O } \left( \frac { k + \log ( 1 / \beta ) } { \varepsilon } \right) .
$$

Therefore, as long as

$$
n \geqslant \tilde { O } \left( \frac { k + \log ( 1 / \beta ) } { \alpha ^ { 2 } m } + \frac { k + \log ( 1 / \beta ) } { \alpha \sqrt { m } \varepsilon } + \frac { k + \log ( 1 / \beta ) } { \varepsilon } \right) ,
$$

the sampler of Theorem 2.18, with utility analyzed in Lemma 2.20, returns a point � with score at most $2 \eta _ { \mathrm { p r i v } } n$ with probability at least $1 - \beta$

Let $\mathcal { L }$ be a corresponding $( \alpha , \tau , 2 \eta _ { \mathrm { p r i v } } n )$ -certificate for $( \boldsymbol { y } , \nu )$ . Since $2 \eta _ { \mathrm { p r i v } } \leqslant \eta _ { \mathrm { a b s } } / 1 0 < \eta _ { \mathrm { a b s } } ,$ Corollary 3.3 and Lemma 4.11 with parameter $\eta _ { 0 } = 2 \eta _ { \mathrm { p r i v } } \mathrm { g i v e }$

$$
\| \mathcal { L } \mu ^ { \prime } - \mu \| _ { 1 } \leqslant O \left( \eta _ { \mathrm { p r i v } } \sqrt { \frac { \log ( 1 / \eta _ { \mathrm { p r i v } } ) } { m } } \right) \leqslant \alpha / 1 0 .
$$

Using the closeness constraint and taking � suficiently small,

$$
\| \nu - \mu \| _ { 1 } \leqslant \| \nu - \mathcal { L } \mu ^ { \prime } \| _ { 1 } + \| \mathcal { L } \mu ^ { \prime } - \mu \| _ { 1 } \leqslant \alpha + 2 \tau \eta _ { \mathrm { p r i v } } n + \alpha / 1 0 \leqslant 6 \alpha / 5 .
$$

If $\hat { \mu }$ is a nearest point to � in $\Delta ^ { k - 1 }$ in $\ell _ { 1 }$ distance, then

$$
\begin{array} { r } { \| \hat { \boldsymbol { \mu } } - \boldsymbol { \mu } \| _ { \mathrm { T V } } = \frac { 1 } { 2 } \| \hat { \boldsymbol { \mu } } - \boldsymbol { \mu } \| _ { 1 } \leqslant \| \boldsymbol { \nu } - \boldsymbol { \mu } \| _ { 1 } \leqslant 6 \alpha / 5 . } \end{array}\tag{15}
$$

Running the construction with internal accuracy $\alpha / 2$ gives the claimed error � and changes the sample bound by only an absolute factor. Projection is deterministic postprocessing and preserves privacy. To cover $\varepsilon > 1$ , first suppose $\varepsilon \leqslant m$ . Partition each user’s batch into $\lceil \varepsilon \rceil$ disjoint subbatches of size $\lfloor m / \lceil \varepsilon \rceil \rfloor$ , discarding any remainder, and run the construction just proved for privacy parameters at most 1 on the $n \lceil \varepsilon \rceil$ subbatches with privacy parameter $\varepsilon / { \lceil \varepsilon \rceil } \leqslant 1$ . Changing one original batch changes at most $\lceil \varepsilon \rceil$ subbatches, so group privacy gives �-DP. Since $\lfloor m / \lceil \varepsilon \rceil \rfloor \geqslant m / ( 2 \lceil \varepsilon \rceil )$ and $\lceil \varepsilon \rceil \leqslant 2 \varepsilon$ the required number of original users is

$$
\widetilde { O } \Bigg ( \frac { k + \log ( 1 / \beta ) } { \alpha ^ { 2 } m } + \frac { k + \log ( 1 / \beta ) } { \alpha \sqrt { m \varepsilon } } + \frac { k + \log ( 1 / \beta ) } { \varepsilon } \Bigg ) .
$$

By the arithmetic–geometric mean inequality, the middle term is at most the sum of the other two. If $\varepsilon > m ,$ apply the Laplace mechanism to the pooled empirical distribution, whose $\ell _ { 1 }$ sensitivity under replacement of one batch is $2 / n$ , and project to the simplex. This uses

$$
\widetilde { O } \left( \frac { k + \log ( 1 / \beta ) } { \alpha ^ { 2 } m } + \frac { k + \log ( 1 / \beta ) } { \alpha \varepsilon } \right)
$$

users; its second term is absorbed by the first because $\varepsilon > m$ and $\alpha < 1$ . This proves Theorem 4.1.

## 4.2.4 Proof of the Approximate-DP Theorem

If $\delta > 1 - \beta / 2$ , release the empirical distribution with probability � and a fixed distribution otherwise. This is $( 0 , \delta ) \mathsf { - D P } .$ With $n = O ( ( k + \log ( 1 / \beta ) ) / ( \alpha ^ { 2 } m ) )$ , the empirical distribution has error at most � with probability at least $1 - \beta / 2 ,$ so the released estimate succeeds with probability at least $\delta ( 1 - \beta / 2 ) > ( 1 - \beta / 2 ) ^ { 2 } \geqslant 1 - \beta$ . We may therefore assume $\delta \leqslant 1 - \beta / 2$ . Then $\beta \delta \leqslant 1 / 2 ,$ , and the additive constant in $\log ( 2 / ( \beta \delta ) )$ is absorbed by $\log ( 1 / \delta ) + \log ( 1 / \beta )$

First suppose $0 < \varepsilon \leqslant 1$ , and apply Theorem 2.19 with the score, internal threshold $\eta _ { \mathrm { p r i v } }$ , and Euclidean radius $\alpha / ( 2 \sqrt { k } )$ from Section 4.2.3. That section verifies the oracle properties, and item 1 of Lemma 4.12 supplies the low-score witness.

To obtain the good-sample event used in Section 4.2.3, we need

$$
n \geqslant \widetilde { O } \left( k + \log ( 1 / \beta ) + \frac { k + \log ( 1 / \beta ) } { \alpha ^ { 2 } m } \right) ,\tag{16}
$$

because $\eta _ { \mathrm { p r i v } } = \operatorname* { m i n } \{ \eta _ { \mathrm { a b s } } / 2 0 , \widetilde { \Theta } ( \alpha \sqrt { m } ) \}$ . For the utility condition (5) on this event, the balls in items 1 and 2 of Lemma 4.12 give

$$
\log \left( \frac { V _ { \eta ^ { \prime } } ( y ) } { V _ { \eta _ { \mathrm { p r i v } } } ( y ) } \right) \leqslant O \left( k \log \left( 1 + \frac { 1 } { \alpha \sqrt { m } } \right) \right) \qquad ( \eta _ { \mathrm { p r i v } } \leqslant \eta ^ { \prime } \leqslant \eta ^ { * } ) .\tag{17}
$$

Apply Lemma 2.20, using the inner ball in item 1 of Lemma 4.12. Below the fixed split threshold, (14) gives $O ( ( k + \log ( 2 / \beta ) ) / ( \varepsilon \eta _ { \mathrm { p r i v } } ) )$ . Above that threshold, $\eta _ { \mathrm { p r i v } }$ is constant and $\log ( 1 + 1 / ( \alpha \sqrt { m } ) ) \leqslant$ $1 / ( \alpha \sqrt { m } )$ , so (17) bounds the utility requirement by

$$
\widetilde { O } \left( \frac { k } { \alpha \sqrt { m } \varepsilon } + \frac { \log ( 2 / \beta ) } { \varepsilon } \right) .
$$

Thus (5) is implied by

$$
n \geqslant \widetilde { O } \Bigg ( \frac { k + \log ( 1 / \beta ) } { \alpha \sqrt { m } \varepsilon } + \frac { \log ( 2 / \beta ) } { \varepsilon } \Bigg ) .
$$

For the worst-case volume condition (3), choose $\eta _ { \mathrm { a b s } }$ small enough that $m \tau _ { \mathrm { c o v } } \leqslant 1 / 2$ . Then Corollary 4.15 gives

$$
n \geqslant O \left( \frac { \log ( 1 / \delta ) + k \log ( 1 + 1 / ( \alpha \sqrt { m } ) ) } { \varepsilon } \right)
$$

as a suficient condition. Since log(1 $+ 1 / ( \alpha \sqrt { m } ) ) \leqslant 1 / ( \alpha \sqrt { m } )$ , this requirement is absorbed by the sample bound in Theorem 4.2. Finally, $k + ( \log ( 1 / \delta ) + \log ( 1 / \beta ) ) / \varepsilon \geqslant k + \log ( 1 / \beta )$ for $\varepsilon \leqslant 1$ , so that theorem’s sample bound also supplies (16).

Allocating failure probability $\beta / 3$ to the sample events and the reduction, we obtain a point � with score at most $2 \eta _ { \mathrm { p r i v } } n$ with probability at least $1 - \beta$ . Applying Lemma 4.11 at $2 \eta _ { \mathrm { p r i v } }$ and projecting as in (15) gives error at most �, after running the construction with internal accuracy $\alpha / 2$

If $\varepsilon > 1$ , use the pure-DP estimator from Theorem 4.1. Its last sample term satisfies

$$
\frac { k + \log ( 1 / \beta ) } { \varepsilon } \leqslant k + \frac { \log ( 1 / \delta ) + \log ( 1 / \beta ) } { \varepsilon } ,
$$

so it also satisfies the sample bound in Theorem 4.2 and is $( \varepsilon , \delta ) – \mathrm { D P } .$ This proves Theorem 4.2.

Remark 4.16 (Explicit logarithmic factors). For $0 < \varepsilon \leqslant 1$ , the proofs of Theorems 4.1 and 4.2 give suficient sample bounds with ordinary � notation. For pure DP, the bound is

$$
n = O \left( \frac { k + \log ( 2 / \beta ) } { \alpha ^ { 2 } m } + \frac { ( k + \log ( 2 / \beta ) ) \sqrt { \log ( e + 1 / ( \alpha \sqrt { m } ) ) } } { \alpha \sqrt { m } \varepsilon } + \frac { k \log ( e / \alpha ) + \log ( 2 / \beta ) } { \varepsilon } \right) .
$$

For approximate DP, the bound is

$$
n = O \left( k + \frac { k + \log ( 2 / \beta ) } { \alpha ^ { 2 } m } + \frac { ( k + \log ( 2 / \beta ) ) \sqrt { \log ( e + 1 / ( \alpha \sqrt { m } ) ) } } { \alpha \sqrt { m } \varepsilon } + \frac { \log ( 1 / \delta ) + \log ( 2 / \beta ) } { \varepsilon } \right) .
$$

Indeed, choose

$$
\eta _ { \mathrm { p r i v } } = c \operatorname* { m i n } \left\{ 1 , { \frac { \alpha { \sqrt { m } } } { \sqrt { \log ( e + 1 / ( \alpha { \sqrt { m } } ) ) } } } \right\}
$$

with $c > 0$ a suficiently small absolute constant. Then $\eta _ { \mathrm { p r i v } } ^ { 2 } \log ( 1 / \eta _ { \mathrm { p r i v } } ) = \Theta ( \operatorname* { m i n } \{ 1 , \alpha ^ { 2 } m \} )$ , so (13) requires $O ( k + \log ( 2 / \beta ) + ( k + \log ( 2 / \beta ) ) / ( \alpha ^ { 2 } m ) )$ users. Below the fixed split threshold, (14) requires

$$
O \Bigg ( \frac { k + \log ( 2 / \beta ) } { \varepsilon \eta _ { \mathrm { p r i v } } } \Bigg ) = O \Bigg ( \frac { ( k + \log ( 2 / \beta ) ) \sqrt { \log ( e + 1 / ( \alpha \sqrt { m } ) ) } } { \alpha \sqrt { m } \varepsilon } \Bigg ) .
$$

Above the split threshold, (17) gives $O ( k / ( \alpha \sqrt { m } \varepsilon ) + \log ( 2 / \beta ) / \varepsilon )$ . Pure $D P$ may also use the $O ( ( k +$ log $( 2 / \beta ) / \varepsilon )$ bound, which is absorbed by $( k \log ( e / \alpha ) + \log ( 2 / \beta ) ) / \varepsilon$ . For pure DP, the full score domain lies in a fixed-radius $\ell _ { 1 }$ ball, giving log $V _ { 1 } / V _ { \eta _ { \mathrm { p r i v } } } ) = O ( k \log ( e / \alpha ) )$ . For approximate DP, (3) contributes

$$
O \Bigg ( \frac { \log ( 1 / \delta ) + k \log ( 1 + 1 / ( \alpha \sqrt { m } ) ) } { \varepsilon } \Bigg ) .
$$

Its dimension-dependent term is absorbed by

$$
\frac { ( k + \log ( 2 / \beta ) ) \sqrt { \log ( e + 1 / ( \alpha \sqrt m ) ) } } { \alpha \sqrt m \varepsilon } .
$$

Thus the privacy termfor small $\alpha \sqrt { m }$ contains only a square-root logarithm.

Remark 4.17 (Larger batches). For clean data, splitting batches gives a refinement of Theorem 4.2. Suppose $0 < \varepsilon \leqslant 1 , \beta \leqslant 1 / 3$ , and $\delta \leqslant 1 / 2$ . There is a polynomial-time $( \varepsilon , \delta ) – D P$ estimator with the same accuracy and failure probability using

$$
\begin{array} { c } { { n = \widetilde O \displaystyle \left( \frac { k + \log ( 1 / \beta ) } { \alpha ^ { 2 } m } + \frac { k + \log ( 1 / \beta ) } { \alpha \sqrt { m } \varepsilon } \right. } } \\ { { \displaystyle \left. + \operatorname* { m i n } \Biggl \{ k , \frac { k + \log ( 1 / \beta ) } { ( \alpha \sqrt { m } \varepsilon ) ^ { 2 / 3 } } \right\} } } \\ { { \displaystyle \left. + \frac { \log ( 1 / \delta ) + \log ( 1 / \beta ) } { \varepsilon } \right) . } } \end{array}
$$

When $\alpha \sqrt { m } \varepsilon \leqslant 1$ , this follows directly from Theorem 4.2. Otherwise, partition each user’s batch into $\lceil ( \alpha \sqrt { m } \varepsilon ) ^ { 2 / 3 } \rceil$ disjoint blocks, each of size

$$
\mathsf { L } _ { \overline { { \mathsf { J } ( \alpha \sqrt { m } } \varepsilon ) ^ { 2 / 3 } } \overline { { \mathsf { J } } } } ^ { m } \mathsf { J } ,
$$

discarding any remaining samples. The number of blocks per user is at most � because $\alpha \varepsilon \leqslant 1$ . Apply Theorem 4.2 to the $n [ ( \alpha \sqrt { m } \varepsilon ) ^ { 2 / 3 } ]$ independent blocks with privacy parameters

$$
\left( \frac { \varepsilon } { \lceil ( \alpha \sqrt { m } \varepsilon ) ^ { 2 / 3 } \rceil } , \quad \frac { \delta } { e ^ { \varepsilon } \lceil ( \alpha \sqrt { m } \varepsilon ) ^ { 2 / 3 } \rceil } \right) .
$$

Changing one original user changes at most $\lceil ( \alpha \sqrt { m } \varepsilon ) ^ { 2 / 3 } \rceil$ blocks, so group privacy gives $( \varepsilon , \delta ) – D P .$ : the accumulated additive privacy error is at most

$$
\frac { \delta } { e ^ { \varepsilon } \Gamma ( \alpha \sqrt { m } \varepsilon ) ^ { 2 / 3 } \rceil } \sum _ { j = 0 } ^ { \lceil ( \alpha \sqrt { m } \varepsilon ) ^ { 2 / 3 } \rceil - 1 } \exp \left( \frac { j \varepsilon } { \lceil ( \alpha \sqrt { m } \varepsilon ) ^ { 2 / 3 } \rceil } \right) \leqslant \delta .
$$

Each block has size at least $m / ( 2 \lceil ( \alpha \sqrt { m } \varepsilon ) ^ { 2 / 3 } \rceil )$ . Dividing the required number of blocks by $\lceil ( \alpha \sqrt { m } \varepsilon ) ^ { 2 / 3 } \rceil$ gives

$$
\begin{array} { r l } & { n = \widetilde { O } \Bigg ( \frac { k } { \lceil ( \alpha \sqrt { m } \varepsilon ) ^ { 2 / 3 } \rceil } + \frac { k + \log ( 1 / \beta ) } { \alpha ^ { 2 } m } } \\ & { \qquad + \frac { ( k + \log ( 1 / \beta ) ) \sqrt { \lceil ( \alpha \sqrt { m } \varepsilon ) ^ { 2 / 3 } \rceil } } { \alpha \sqrt { m } \varepsilon } } \\ & { \qquad + \frac { \log ( 1 / \delta ) + \log ( 1 / \beta ) } { \varepsilon } } \\ & { \qquad + \frac { \log \lceil ( \alpha \sqrt { m } \varepsilon ) ^ { 2 / 3 } \rceil } { \varepsilon } \Bigg ) . } \end{array}
$$

Since $\alpha { \sqrt { m } } \varepsilon > 1$ , rounding the number of blocks up changes it by at most a factor of two. Taking the better of this bound and the bound of Theorem 4.2 applied to the original batches proves the claim. Here $\widetilde { O }$ suppresses the additional factor log $\lceil ( \alpha \sqrt { m } \varepsilon ) ^ { 2 / 3 } \rceil$ but no additional dependence on $\delta .$ For constant $\beta$ and $\alpha { \sqrt { m } } \varepsilon > 1$ , the fixed � term is thus replaced by $k / ( \alpha \sqrt { m } \varepsilon ) ^ { 2 / 3 }$ . This argument does not preserve the guarantee of Corollary 4.18: splitting reduces the batch size that controls the corruption error.

For clean data, we may also choose between this estimator and that of Esfandiari et al. [EMN22] using only the public parameters, obtaining the smaller of the two sample bounds without any privacy composition. In particular,for constant $\beta$ and $m \geqslant C k ^ { 2 } / \alpha ^ { 2 }$ , their ball-localization algorithm gives $n = O ( \log ( 1 / \delta ) / \varepsilon )$ users. Indeed, each user’s empirical distribution satisfies $\mathbb { E } \| y _ { i } - \mu \| _ { 2 } ^ { 2 } \leqslant 1 / m ,$ so Markov’s inequality and a Chernof bound place at least two-thirds of the users within $O ( 1 / \sqrt { m } )$ of $\vert \mu$ in $\ell _ { 2 } .$ Applying [EMN22, Theorem 3.3], followed by Euclidean projection onto the simplex, gives $\ell _ { 2 }$ error $O ( \sqrt { k / m } )$ , hence total-variation error $O ( k / \sqrt { m } )$ . For general batch sizes, their few-user bound has the additional logarithmic dependence on $1 / \delta$ stated in [EMN22, Theorem 4.1].

Corollary 4.18 (Robustness of the Private Estimators). There are absolute constants $c , C > 0$ for which the following holds. Let $0 < \varepsilon \leqslant 1 , 0 \leqslant \eta \leqslant c .$ , and $\alpha , \beta \in ( 0 , 1 )$ . Suppose the clean batches are independent and batch � consists of � i.i.d. samples from $\mu _ { i } ,$ where $\| \mu _ { i } - \mu \| _ { \mathrm { T V } } \leqslant \omega$ for every � and a known bound $\omega \in [ 0 , 1 ]$ . There is a polynomial-time, pure ${ \varepsilon } { - } D P$ estimator that, on any �-corruption of these batches, returns $\hat { \mu } \in \Delta ^ { k - 1 }$ with $\| \hat { \mu } - \mu \| _ { \mathrm { T V } } \leqslant$ � with probability at least $1 - \beta ,$ , provided

$$
\alpha \geqslant C \left( \eta \sqrt { \frac { \log ( 1 / \eta ) } { m } } + \omega \right)
$$

and

$$
n \geqslant \widetilde { O } \left( \frac { k + \log ( 1 / \beta ) } { \alpha ^ { 2 } m } + \frac { k + \log ( 1 / \beta ) } { \alpha \sqrt { m } \varepsilon } + \frac { k + \log ( 1 / \beta ) } { \varepsilon } \right) .
$$

The term involving � is defined to be zero when $\eta = 0$ . For every $\delta \in ( 0 , 1 )$ , there is also a polynomial-time $( \varepsilon , \delta ) – D P$ estimator with this accuracy guarantee, provided

$$
n \geqslant \widetilde { O } \bigg ( k + \frac { k + \log ( 1 / \beta ) } { \alpha ^ { 2 } m } + \frac { k + \log ( 1 / \beta ) } { \alpha \sqrt { m } \varepsilon } + \frac { k \omega / \alpha + \log ( 1 / \delta ) + \log ( 1 / \beta ) } { \varepsilon } \bigg ) .
$$

This recovers the sample bound of Theorem 4.2 when $\omega = O ( 1 / { \sqrt { m } } + \alpha \varepsilon ) ;$ in particular, it does so when � $\sqrt { m } \leqslant 1$ or when � is constant.

Proof. We first prove the pure-DP claim. Use the score construction from Section 4, with internal accuracy $c _ { 1 } \alpha$ for a suficiently small absolute $c _ { 1 } > 0$ . Choose its internal trimming parameter �<sub>priv</sub> $\leqslant \eta _ { \mathrm { a b s } } / 2 0$ as in Section 4.2.3, so that

$$
\eta _ { \mathrm { p r i v } } = \operatorname* { m i n } \Bigl \{ \eta _ { \mathrm { a b s } } / 2 0 , ~ \widetilde { \Theta } ( \alpha \sqrt { m } ) \Bigr \} , \qquad O \left( \eta _ { \mathrm { p r i v } } \sqrt { \frac { \log ( 1 / \eta _ { \mathrm { p r i v } } ) } { m } } \right) \leqslant \alpha / 1 0 .
$$

The constants � in the accuracy assumption and � in $\eta \leqslant c$ ensure $\eta \leqslant \eta _ { \mathrm { p r i v } }$ . Replace the covariance allowance by

$$
\tau _ { \mathrm { c o v } } = C _ { 0 } \bigg ( \frac { \eta _ { \mathrm { p r i v } } \log ( 1 / \eta _ { \mathrm { p r i v } } ) } { m } + \frac { \omega } { m } + \omega ^ { 2 } \bigg ) ,\tag{18}
$$

where $C _ { 0 }$ is a suficiently large absolute constant. All these parameters are public. The dataconsistency constraints and the dependence of the score on its trimming budget are unchanged, so Lemmas 4.7 to 4.10 apply without modification. The witness bounds remain polynomial because $\omega \leqslant 1$

We verify the statistical properties needed by the pure-DP reduction. The concentration estimates in the proof of Corollary 3.11 hold simultaneously at parameters $\eta _ { \mathrm { p r i v } } , 2 \eta _ { \mathrm { p r i v } } ,$ and $\eta ^ { * } = \eta _ { \mathrm { a b s } } / 2$ , with failure probability at most $\beta / 2 ,$ , once

$$
n \geqslant \Omega \left( \frac { k + \log ( 1 / \beta ) } { \eta _ { \mathrm { p r i v } } ^ { 2 } \log ( 1 / \eta _ { \mathrm { p r i v } } ) } \right) .
$$

For the low-score witness, take $w _ { i }$ to indicate the uncorrupted batches, set $y _ { i } ^ { \prime } \ = \ y _ { i } ^ { * } .$ , and set $\mu ^ { \prime } = \bar { \mu } = \mathbb { E } _ { i } y _ { i } ^ { * }$ . By the proof of Corollary 3.11, we obtain a diagonal majorizer with allowance (18). Since at most $\eta n \leqslant \eta _ { \mathrm { p r i v } } n$ batches are corrupted, this gives a score-�<sub>priv</sub>� certificate at $\bar { \mu }$ and throughout the $\ell _ { 1 }$ ball of radius $c _ { 1 } \alpha$ around ${ \bar { \mu } } .$

For $\eta _ { 0 } \in \left. 2 \eta _ { \mathrm { p r i v } } , \eta ^ { * } \right.$ , the allowance in (18) is at most $C _ { 0 } ( \eta _ { 0 } \log ( 1 / \eta _ { 0 } ) / m + \omega / m + \omega ^ { 2 } )$ . Thus the degree-6 soundness argument in Corollary 3.11, with the numerical tolerance argument of Lemma 4.11, shows that every certificate with trimming budget at most $\eta _ { 0 } n$ has

$$
\| \mathcal { L } \mu ^ { \prime } - \mu \| _ { 1 } \leqslant O \left( \eta _ { 0 } \sqrt { \frac { \log ( 1 / \eta _ { 0 } ) } { m } } + \omega \right) .\tag{19}
$$

In particular, the score-�<sup>∗</sup>� set is contained in an $\ell _ { 1 }$ ball around $\mu$ of radius $O ( c _ { 1 } \alpha + 1 / \sqrt { m } + \omega )$ Because $\omega \leqslant \alpha / C = O ( c _ { 1 } \alpha )$ , this radius is $O ( c _ { 1 } \alpha + 1 / \sqrt { m } )$ . Comparing this outer ball with the

$\mathbf { s c o r e } – \eta _ { \mathrm { p r i v } } n$ ball of radius $c _ { 1 } \alpha$ about $\bar { \mu }$ gives

$$
\log \left( \frac { V _ { \eta ^ { \prime } } ( \boldsymbol { y } ) } { V _ { \eta _ { \mathrm { p r i v } } } ( \boldsymbol { y } ) } \right) \leqslant O \left( k \log \left( 2 + \frac { 1 } { c _ { 1 } \alpha \sqrt { m } } \right) \right) \qquad ( \eta _ { \mathrm { p r i v } } \leqslant \eta ^ { \prime } \leqslant \eta ^ { * } ) .
$$

The full score domain is stil $\Delta ^ { k - 1 } + \mathbb { B } _ { 1 } ^ { k } ( c _ { 1 } \alpha + \tau n )$ , so the bound on $V _ { 1 } / V _ { \eta _ { \mathrm { p r i v } } }$ from Section 4.2.3 also applies.

It follows from Theorem 2.18 that the sample bound in Corollary 4.18 returns a point with score at most $2 \eta _ { \mathrm { p r i v } } n$ with probability at least $1 - \beta / 2$ on the concentration event. In the capped regime, the additional statistical requirement $n = \Omega ( k + \log ( 1 / \beta ) )$ is absorbed by $( k + \log ( 1 / \beta ) ) / \varepsilon$ because $\varepsilon \leqslant 1$ Apply (19) with $\eta _ { 0 } = 2 \eta _ { \mathrm { p r i v } } ,$ , add the closeness allowance from item $2$ of Definition $4 . 3 ,$ and project onto the simplex in $\ell _ { 1 }$ . For suficiently small $c _ { 1 }$ and suficiently large $C ,$ the resulting total-variation error is at most $\alpha$ . Together, the concentration and privacy-mechanism failure probabilities sum to at most $\beta .$

For approximate DP, use the heterogeneous covariance allowance (18). If $\delta > 1 - \beta / 2 ,$ , apply the release-with-probability-� construction from Section 4.2.4 to the robust estimator of Corollary 3.11, with trimming parameter $\eta _ { \mathrm { p r i v } }$ and failure probability $\beta / 2$ . Its sample requirement is

$$
\widetilde { O } \left( k + \log ( 1 / \beta ) + \frac { k + \log ( 1 / \beta ) } { \alpha ^ { 2 } m } \right) ,
$$

which the stated approximate-DP bound supplies since $\varepsilon \leqslant 1$ . Its success probability is at least $\delta ( 1 - \beta / 2 ) > 1 - \beta$

Otherwise, use the approximate-DP reduction with internal accuracy $c _ { 1 } \alpha$ and trimming parameter $\eta _ { \mathrm { p r i v } }$ . Since $\omega \leqslant 1$ , (18) gives

$$
\sqrt { 1 / m + \tau _ { \mathrm { c o v } } } \leqslant O ( 1 / \sqrt { m } + \omega ) .
$$

Thus Corollary 4.15 bounds the worst-case intermediate log-volume ratio by

$$
\log \left( \frac { V _ { \eta ^ { * } } ( \mathcal { Y } ) } { V _ { 0 . 8 \eta ^ { * } } ( \mathcal { Y } ) } \right) \leqslant O \left( k \log \left( 1 + \frac { 1 } { c _ { 1 } \alpha \sqrt { m } } + \frac { \omega } { c _ { 1 } \alpha } \right) \right) .\tag{20}
$$

Using $\log ( 1 + x ) \leqslant x ,$ the privacy condition (3) is therefore implied by

$$
n \geqslant O \biggl ( \frac { k } { \alpha \sqrt { m } \varepsilon } + \frac { k \omega / \alpha + \log ( 1 / \delta ) } { \varepsilon } \biggr ) .
$$

On the concentration event used for pure DP, the score-�<sub>priv</sub>� set contains the ball of radius $c _ { 1 } \alpha$ about $\bar { \mu } .$ . By (19) at $\eta _ { 0 } = \eta ^ { * }$ and the mean estimate in Corollary 3.11, every point of score at most $\eta ^ { * } n$ lies within $c _ { 1 } \alpha + O ( 1 / \sqrt { m } + \omega )$ of $\bar { \mu } .$ . Comparing these balls bounds log $( V _ { \eta ^ { \prime } } ( \boldsymbol { y } ) / V _ { \eta _ { \mathrm { p r i v } } } ( \boldsymbol { y } ) )$ by the right-hand side of (20) for $\eta _ { \mathrm { p r i v } } \leqslant \eta ^ { \prime } \leqslant \eta ^ { * }$ . When � $\sqrt { m }$ is below the fixed split threshold in Section $4 . 2 . 3 , \omega \leqslant \alpha / C$ makes the heterogeneity contribution a constant inside the logarithm, so the original $\widetilde { O } ( ( k + \log ( 1 / \beta ) ) / ( \alpha \sqrt { m } \ \varepsilon ) )$ utility bound applies. Above that threshold, $\eta _ { \mathrm { { p r i v } } }$ is constant, and Lemma 2.20 requires only

$$
n \geqslant O \left( \frac { k } { \alpha \sqrt { m } \varepsilon } + \frac { k \omega / \alpha + \log ( 2 / \beta ) } { \varepsilon } \right) .
$$

Both requirements are covered by the stated sample bound, which also supplies the concentration event. The reduction returns score at most $2 \eta _ { \mathrm { p r i v } } n ;$ applying (19) and projecting as in (15) gives accuracy �. Allocating failure probability $\beta / 2$ to concentration and $\beta / 2$ to the reduction proves the claim. □

## 5 Statistical Query Lower Bound for Discrete Batches

In this section, we prove a statistical query (SQ) lower bound for robust learning from discrete batches. We show that no SQ algorithm with polynomially many queries of inverse-polynomial tolerance achieves error �(� min{1 $\cdot \sqrt { \log ( 1 / \eta ) / m } \} ,$ ) uniformly over the parameters. We adapt the moment-matching strategy and polynomial-correction construction of Diakonikolas, Kane, and Sun [DKS22] to categorical batches in $[ k ] ^ { m }$ . We correct a biased binomial distribution to match the low-order moments of the unbiased binomial, and use this distribution for the number of batch elements in a hidden balanced subset of the alphabet. The resulting moment-matching lemma applies to every batch size. We first prove hardness of testing and then reduce testing to robust estimation.

## 5.1 Testing Problem

Let $\mu \in \Delta ^ { k - 1 }$ be the clean distribution. We model corruption by allowing the observed batch distribution $Q$ on $[ k ] ^ { m }$ to difer from the clean batch distribution $\mu ^ { \otimes m }$ by at most � in total variation:

$$
\| Q - \mu ^ { \otimes m } \| _ { \mathrm { T V } } \leqslant \eta .
$$

An SQ query to $Q$ consists of a function $\phi \colon [ k ] ^ { m }  [ - 1 , 1 ]$ and a tolerance $\tau > 0 ;$ the oracle returns a number within � of $\mathbb { E } _ { B \sim Q } \big [ \phi ( B ) \big ]$ . We allow randomized, adaptive algorithms, whose guarantees must hold for every valid oracle. Query bounds are worst-case bounds. Allowing queries on the full ordered batch also covers SQ algorithms whose queries depend only on the batch average.

We give the construction and proof for even �; the case of odd � follows similarly by leaving one symbol unused. Let $\mu _ { 0 }$ be uniform on [�], and set $P = \mu _ { 0 } ^ { \otimes m }$ . Pair the alphabet symbols as $\{ 1 , 2 \} , \{ 3 , 4 \} , \dots , \{ k - 1 , k \}$ . A hidden subset � contains exactly one symbol from each pair, so $| S | = k / 2$ . For $0 < \Delta < 1 / 4$ , increase the probabilities on � and decrease those on its complement:

$$
\mu _ { S } ( j ) : = \left\{ \begin{array} { l l } { ( 1 + 2 \Delta ) / k , } & { j \in S , } \\ { ( 1 - 2 \Delta ) / k , } & { j \notin S . } \end{array} \right. \qquad \| \mu _ { S } - \mu _ { 0 } \| _ { \mathrm { T V } } = \Delta .
$$

For a batch $x = ( x _ { 1 } , \ldots , x _ { m } ) \in [ k ] ^ { m }$ , let

$$
N _ { S } ( x ) : = \sum _ { i = 1 } ^ { m } \mathbb { 1 } [ x _ { i } \in S ]
$$

be the number of its elements that lie in the hidden subset. A single draw lies in � with probability $1 / 2$ under $\mu _ { 0 }$ and $1 / 2 + \Delta$ under $\mu _ { S }$ . Writing $p _ { s } : = \mathrm { B i n } ( m , 1 / 2 + s )$ for $\vert s \vert < 1 / 2$ , we therefore have

$$
N _ { S } ( X ) \sim \left\{ { p _ { 0 } , \quad X \sim P , } \atop { p _ { \Delta } , \quad X \sim \mu _ { S } ^ { \otimes m } . }  \right.
$$

Conditioned on the count $N _ { S , }$ , both batch laws are uniform on the same set of batches. This lets us modify the count distribution while keeping the conditional batch distribution fixed.

To compare diferent hidden subsets, encode the choice from each pair by $\sigma \in \left\{ - 1 , 1 \right\} ^ { k / 2 }$ : put $2 u - 1$ in $S _ { \sigma }$ if $\sigma _ { u } = 1$ , and 2� in $S _ { \sigma }$ if $\sigma _ { u } = - 1$ . Write $\mu _ { \sigma } = \mu _ { S _ { c } }$ <sub>�</sub> and $N _ { \sigma } = N _ { S _ { \sigma } }$ . The sign encoding

$$
v _ { \sigma } ( j ) : = 2 \mathbb { 1 } [ j \in S _ { \sigma } ] - 1 , \qquad v _ { \sigma } = ( \sigma _ { 1 } , - \sigma _ { 1 } , \ldots , \sigma _ { k / 2 } , - \sigma _ { k / 2 } )
$$

satisfies $\mu _ { \sigma } ( j ) = ( 1 + 2 \Delta v _ { \sigma } ( j ) ) / k$ . We consider the testing problem

$$
\begin{array} { r l } { H _ { 0 } : } & { { } ( \mu , Q ) = ( \mu _ { 0 } , P ) , } \\ { H _ { 1 } : } & { { } \mu = \mu _ { \sigma } \mathrm { ~ f o r ~ s o m e ~ } \sigma \in \{ - 1 , 1 \} ^ { k / 2 } , \qquad \| Q - \mu _ { \sigma } ^ { \otimes m } \| _ { \mathrm { T V } } \leqslant \eta / 2 0 . } \end{array}\tag{21}
$$

Given $\mathrm { S Q }$ access to �, a tester must identify the clean hypothesis with probability at least $2 / 3$ on every instance. Each hypothesis specifies both the clean and observed distributions. If an observed distribution occurs under both hypotheses, testing is already information-theoretically impossible.

## 5.2 SQ Hardness

Theorem 5.1 (SQ Lower Bound for Testing). There is an absolute constant $c > 0$ such that, for every integer $t \geqslant 1 .$ , there are constants $c _ { t } > 0$ and $k _ { 0 } ( t )$ with thefollowing property. For all $k \geqslant k _ { 0 } ( t )$ $m \in \mathbb { N } .$ and $\eta \in ( 0 , 1 / 4 ]$ , set $\Delta = c _ { t } \eta$ min $\{ 1 , \sqrt { \log ( 1 / \eta ) / m } \}$ . Any SQ algorithm solving (21) must either make a query of tolerance less than

$$
2 k ^ { - ( t + 1 ) / 8 }
$$

or make at least exp $) ( c { \sqrt { k } } )$ queries.

Proof roadmap. For each hidden subset $S _ { \sigma } ,$ , we construct a valid corrupted batch law $Q _ { \sigma }$ . The goal is to make its centered likelihood ratio $f _ { \sigma } = d Q _ { \sigma } / d P - 1$ satisfy

$$
\begin{array} { r } { \| f _ { \sigma } \| _ { L ^ { 2 } ( P ) } ^ { 2 } \leqslant 2 , \qquad \left| \langle f _ { \sigma } , f _ { \omega } \rangle _ { L ^ { 2 } ( P ) } \right| \leqslant 2 | \rho ( \sigma , \omega ) | ^ { t + 1 } . } \end{array}
$$

Here $\begin{array} { r } { \rho ( \sigma , \omega ) : = ( 2 / k ) \sum _ { u = 1 } ^ { k / 2 } \sigma _ { u } \omega _ { u } } \end{array}$ is the centered overlap of the hidden subsets. Lemma 5.2 supplies the count distribution used to define $Q _ { \sigma }$ . Its total-variation bound keeps $Q _ { o }$ within the corruption budget; its moment identities eliminate Fourier degrees $1 , \ldots , t ;$ and its chi-square bound controls the remaining Fourier mass. Independent uniformly chosen hidden subsets have small $| \rho |$ with high probability. The resulting correlation bound then implies the query lower bound through Lemma 5.3. Finally, Section 5.3 reduces robust estimation to this testing problem. The binomial construction is proved in Section 5.4.

Lemma 5.2 (Binomial Moment Matching). For every integer $t \geqslant 1$ , there is a constant $c _ { t } > 0$ with the following property. Let $m \in \mathbb { N }$ and $\eta \in ( 0 , 1 / 4 ]$ , and set

$$
\Delta : = c _ { t } \eta \sqrt { \operatorname * { m i n } \biggl \{ 1 , \frac { \log ( 1 / \eta ) } { m } \biggr \} } , \qquad p _ { s } : = \mathrm { B i n } ( m , 1 / 2 + s ) .
$$

There is a probability distribution $\widetilde { p }$ on $\{ 0 , \ldots , m \}$ such that

$$
\mathbb { E } _ { N \sim \widetilde { \eta } } \big [ N ^ { i } \big ] = \operatorname * { \mathbb { E } } _ { N \sim p _ { 0 } } \big [ N ^ { i } \big ] \quad ( 0 \leqslant i \leqslant t ) , \qquad \| \widetilde { \eta } - p _ { \Delta } \| _ { \mathrm { T V } } \leqslant \eta / 2 0 , \qquad \chi ^ { 2 } ( \widetilde { p } \| p _ { 0 } ) \leqslant 2 .\tag{22}
$$

Here $\begin{array} { r } { \chi ^ { 2 } ( \widetilde { p } \| p ) : = \sum _ { i } ( \widetilde { p } ( j ) - p ( j ) ) ^ { 2 } / p ( j ) } \end{array}$ denotes the chi-square divergence.

We separate the general SQ argument from the categorical construction in the following lemma; compare the SQ correlation bound in [DKS22, Lemma 11]. We prove this lemma after Theorem 5.1.

Lemma 5.3 (SQ Testing from Correlation Tails). Let � and $\left\{ Q _ { \theta } : \theta \in \Theta \right\}$ be distributions on a finite domain �, with Θ finite and nonempty and $Q _ { \theta } \ll P$ . Set $f _ { \theta } = d Q _ { \theta } / d P - 1$ . Suppose that, for some $B > 0$ and $\gamma , \xi \geqslant 0$ , every $\theta \in \Theta$ satisfies

$$
\| f _ { \theta } \| _ { L ^ { 2 } ( P ) } ^ { 2 } \leqslant B , \qquad \underset { \omega \sim \mathrm { U n i f } ( \Theta ) } { \mathbb { P } } \big [ \big | \langle f _ { \theta } , f _ { \omega } \rangle _ { L ^ { 2 } ( P ) } \big | > \gamma \big ] \leqslant \xi .
$$

Let $\tau > 0$ satisfy $\tau ^ { 2 } > \gamma$ . Any randomized, adaptive $S Q$ tester that distinguishes � from every $Q _ { \theta }$ with success probability at least $2 / 3$ against every valid oracle, makes at most $q$ queries in the worst case, and uses tolerances at least �, must satisfy

$$
q B \xi \geqslant \frac { \tau ^ { 2 } - \gamma } { 3 } .
$$

Proof of Theorem 5.1. Take $\widetilde { p }$ from Lemma 5.2. For each $\sigma \in \left\{ - 1 , 1 \right\} ^ { k / 2 }$ , we construct an observed distribution $Q _ { \sigma }$ satisfying $\Vert Q _ { \sigma } - \mu _ { \sigma } ^ { \otimes m } \Vert _ { \mathrm { T V } } \leqslant \eta / 2 0$

Hard distributions. To draw a batch from $Q _ { \sigma } ,$ draw its count $N _ { \sigma }$ from $\widetilde { p }$ and then choose uniformly among the batches with that count. Equivalently, define

$$
F _ { \sigma } ( x ) : = \frac { d Q _ { \sigma } } { d P } ( x ) = \frac { \widetilde { p } ( N _ { \sigma } ( x ) ) } { p _ { 0 } ( N _ { \sigma } ( x ) ) } , \qquad f _ { \sigma } : = F _ { \sigma } - 1 .
$$

Under $P ,$ the count $N _ { \sigma }$ has distribution ${ p } _ { 0 } ,$ , so $Q _ { \sigma }$ is a probability distribution. $\mu _ { \sigma } ^ { \otimes m }$ has likelihood ratio $p _ { \Delta } ( N _ { \sigma } ) / p _ { 0 } ( N _ { \sigma } )$ with respect to $P .$ . Consequently,

$$
\| Q _ { \sigma } - \mu _ { \sigma } ^ { \otimes m } \| _ { \mathrm { T V } } = \| \widetilde { p } - p _ { \Delta } \| _ { \mathrm { T V } } \leqslant \eta / 2 0 , \qquad \| f _ { \sigma } \| _ { L ^ { 2 } ( P ) } ^ { 2 } = \chi ^ { 2 } ( \widetilde { p } \| p _ { 0 } ) \leqslant 2 .\tag{23}
$$

Bounding correlations. For $z \in \{ - 1 , 1 \} ^ { m }$ , let $N ( z )$ count its positive coordinates, and write the Fourier expansion

$$
g ( z ) : = \frac { \widetilde { p } ( N ( z ) ) } { p _ { 0 } ( N ( z ) ) } = \sum _ { A \subseteq [ m ] } \hat { g } ( A ) \prod _ { i \in A } z _ { i } .
$$

The batch likelihood ratios satisfy $F _ { \sigma } ( x ) = g ( v _ { \sigma } ( x _ { 1 } ) , \ldots , v _ { \sigma } ( x _ { m } ) )$ . Here $\hat { g } ( \emptyset ) = 1$ . Moment matching implies $\hat { g } ( A ) = 0$ for $1 \leqslant \left| A \right| \leqslant t$ . Indeed, the conditional expectation of $\Pi _ { i \in A } z _ { i }$ given $N ( z ) = j$ under uniform signs, is

$$
{ \binom { m } { | A | } } ^ { - 1 } \sum _ { r = 0 } ^ { | A | } ( - 1 ) ^ { | A | - r } { \binom { j } { r } } { \binom { m - j } { | A | - r } } .
$$

This is a polynomial in � of degree at most $| A |$ . Its expectation under $\widetilde { p }$ therefore equals its expectation under $p _ { 0 } ,$ , which is zero.

For � uniform on $[ k ] ,$ , the signs $v _ { \sigma } ( U )$ and $v _ { \omega } ( U )$ are unbiased and satisfy $\mathbb { E } [ v _ { \sigma } ( U ) v _ { \omega } ( U ) ] =$ $\rho ( \sigma , \omega )$ . In the product of the two Fourier expansions, terms indexed by distinct sets have expectation zero: a position in their symmetric diference contributes one unbiased sign. Terms indexed by the same set � have expectation $\rho ( \sigma , \omega ) ^ { | A | }$ . Thus

$$
\langle f _ { \sigma } , f _ { \omega } \rangle _ { L ^ { 2 } ( P ) } = \sum _ { | A | > t } \hat { g } ( A ) ^ { 2 } \rho ( \sigma , \omega ) ^ { | A | } , \qquad \left| \langle f _ { \sigma } , f _ { \omega } \rangle _ { L ^ { 2 } ( P ) } \right| \leqslant 2 | \rho ( \sigma , \omega ) | ^ { t + 1 } .\tag{24}
$$

Parseval’s identity and (23) give the inequality. If $t \geqslant m$ , the sum is empty and $f _ { \sigma } = 0$ for every $\sigma .$

Applying the SQ lemma. For each fixed $\sigma ,$ Hoefding’s inequality gives

$$
\begin{array} { r } { \mathbb { P } _ { \omega \sim \mathrm { U n i f } ( \{ \pm 1 \} ^ { k / 2 } ) } \big [ | \rho ( \sigma , \omega ) | > k ^ { - 1 / 4 } \big ] \leqslant 2 e ^ { - \sqrt { k } / 8 } . } \end{array}
$$

Together with (23) and (24), this verifies the hypotheses of Lemma 5.3 with

$$
B = 2 , \qquad \gamma = 2 k ^ { - ( t + 1 ) / 4 } , \qquad \xi = 2 e ^ { - \sqrt { k } / 8 } , \qquad \tau = 2 k ^ { - ( t + 1 ) / 8 } .
$$

If all query tolerances are at least �, the lemma yields

$$
q \geqslant \frac { \tau ^ { 2 } - \gamma } { 3 B \xi } = \frac { 1 } { 6 } k ^ { - ( t + 1 ) / 4 } e ^ { \sqrt { k } / 8 } \geqslant e ^ { \sqrt { k } / 2 4 }
$$

for suficiently large � depending only on �. This proves the theorem with $c = 1 / 2 4$

Proof of Lemma 5.3. All inner products and norms below are in $L ^ { 2 } ( P )$ , and indices are independent and uniform on Θ unless otherwise specified.

One query. Fix $\phi \colon X  [ - 1 , 1 ]$ . Let

$$
\begin{array} { r } { \mathcal { T } _ { \phi } : = \left\{ \theta : \left. \langle \phi , f _ { \theta } \rangle \right. > \tau \right\} , \qquad p : = \mathbb { P } \left[ \theta \in \mathcal { T } _ { \phi } \right] , \qquad G : = \frac { \mathbb { F } } { \theta } \big [ \mathbb { 1 } [ \theta \in \mathcal { T } _ { \phi } ] \mathrm { s i g n } ( \langle \phi , f _ { \theta } \rangle ) f _ { \theta } \big ] . } \end{array}
$$

If $p > 0 ,$ , Cauchy–Schwarz and $\| \phi \| _ { 2 } \leqslant 1$ give

$$
p ^ { 2 } \tau ^ { 2 } < \langle \phi , G \rangle ^ { 2 } \leqslant \| G \| _ { 2 } ^ { 2 } \leqslant \gamma p ^ { 2 } + B \xi p .
$$

For the last inequality, pairs with $\left| \langle f _ { \theta } , f _ { \omega } \rangle \right| \leqslant \gamma$ contribute at most $\gamma { { p } ^ { 2 } }$ . For each $\theta \in \mathcal { T } _ { \phi } ,$ the remaining � have probability at most $\xi ,$ and $\left| \langle f _ { \theta } , f _ { \omega } \rangle \right| \leqslant B$ by the norm bound. Rearranging gives

$$
p \leqslant \frac { B \xi } { \tau ^ { 2 } - \gamma } .\tag{25}
$$

This bound also holds when $p = 0$ . Since $\mathbb { E } _ { Q _ { \theta } } \phi - \mathbb { E } _ { P } \phi = \langle \phi , f _ { \theta } \rangle$ , it bounds the fraction of alternatives for which the exact null answer is outside tolerance �.

Adaptive queries. Run the tester against the oracle returning exact expectations under �. Conditional on its random coins, the at most $q$ queries on this execution are fixed. By (25) and a union bound, these answers are all valid for a uniform $Q _ { \theta }$ at the requested tolerances except with probability at most $q B \xi / ( \tau ^ { 2 } - \gamma )$ , also after averaging over the coins.

For each $\theta ,$ define an oracle that returns $\mathbb { E } _ { P } \phi$ whenever this is within the requested tolerance of $\mathbb { E } _ { Q _ { \theta } } \phi _ { \varepsilon }$ , and returns $\mathbb { E } _ { Q _ { \theta } } \phi$ otherwise. This is a valid oracle on every execution and does not depend on the tester’s random coins. Couple the null and alternative executions using the same coins. Their transcripts difer with probability at most $q B \xi / ( \tau ^ { 2 } - \gamma )$ , averaged over $\theta .$ . Success probability $2 / 3$ under each hypothesis requires an acceptance-probability gap of at least $1 / 3$ . Therefore

$$
\frac { q B \xi } { \tau ^ { 2 } - \gamma } \geqslant \frac 1 3 .
$$

## 5.3 From Estimation to Testing

Corollary 5.4 (SQ Lower Bound for Robust Estimation). For every integer $t \geqslant 1$ , let $c _ { t }$ and $k _ { 0 } ( t )$ be as in Theorem 5.1. For all $k \geqslant k _ { 0 } ( t ) , m \in \mathbb { N } _ { }$ , and $\eta \in ( 0 , 1 / 4 ]$ , any $S Q$ estimator that,for every $\mu \in \Delta ^ { k - 1 }$ and every $Q$ with $\| Q - \mu ^ { \otimes m } \| _ { \mathrm { T V } } \leqslant \eta ,$ , guarantees

$$
\| \hat { \mu } - \mu \| _ { \mathrm { T V } } \leqslant \frac { c _ { t } } { 4 } \eta \operatorname* { m i n } \left\{ 1 , \sqrt { \frac { \log ( 1 / \eta ) } { m } } \right\}
$$

with probability at least $2 / 3$ must either make a query of tolerance less than $2 k ^ { - ( t + 1 ) / 8 }$ or make at least <sub>exp(�</sub>√<sub>�)</sub> <sub>queries.</sub>

Consequently, no $S Q$ algorithm with fixed polynomial bounds on query count and inverse tolerance in $k , m , 1 / \eta$ achieves error �(� min $\{ 1 , \sqrt { \log ( 1 / \eta ) / m } \} )$ ) uniformly over the parameters. This holds already along the sequence $\eta = 1 / k$ and $m = \lceil ( \log k ) ^ { 2 } \rceil a s k  \infty$

Proof. Run the estimator and declare $H _ { 1 } \mathrm { i f } \| \hat { \mu } - \mu _ { 0 } \| _ { \mathrm { T V } } > \Delta / 2$ . On the estimator’s success event, this distance is at most $\Delta / 4$ under $H _ { 0 }$ and at least $3 \Delta / 4$ under $H _ { 1 }$ , by the triangle inequality. Thus the estimator solves (21), and Theorem 5.1 applies.

For the asymptotic conclusion, set $\eta = 1 / k$ and $m = \lceil ( \log k ) ^ { 2 } \rceil$ . Any fixed polynomial resource bounds become $q \leqslant k ^ { a }$ and tolerances at least $k ^ { - b }$ , for fixed $a , b$ and suficiently large $k .$ Choose a fixed � with $( t + 1 ) / 8 > b .$ . An �(� min $\{ 1 , \sqrt { \log ( 1 / \eta ) / m } \} )$ error guarantee is eventually smaller than $c _ { t } \eta$ min $\{ 1 , \sqrt { \log ( 1 / \eta ) / m } \} / 4 ,$ , while $k ^ { - b } \geqslant \bar { 2 } k ^ { - ( t + 1 ) / 8 }$ and $k ^ { a } < \exp ( \sqrt { k } / 2 4 )$ . This contradicts Theorem 5.1. □

With suficiently many batches, the polynomial-time estimator of Jain and Orlitsky [JO20b], together with the pooled histogram for $m \leqslant \log ( 1 / \eta )$ , achieves error $O ( \eta \operatorname* { m i n } \{ 1 , \sqrt { \log ( 1 / \eta ) / m } \} )$ . Information-theoretically, the corruption dependence is $\Theta ( \eta / \sqrt { m } ) \ [ \mathrm { Q V 1 8 } ]$ . In Theorem 5.1, the error scale matches the dependence on � and � of the known polynomial-time corruption rate. Its constant $c _ { t }$ depends on the moment order, and hence on the polynomial bound on inverse tolerance. We bound the attainable corruption error, rather than the number of batches required to attain it.

One can also obtain the distributions used in the SQ lower bound, up to exponentially small total-variation error, by replacing at most $\lfloor \eta n \rfloor$ batches in a finite dataset. Indeed, independently maximally couple � clean batches to batches with law $Q _ { \sigma }$ . Replace the mismatched batches if there are at most ⌊��⌋ mismatches, and make no replacements otherwise. This is a legal batch-corruption adversary. Since $\lVert Q _ { \sigma } - \mu _ { \sigma } ^ { \otimes m } \rVert _ { \mathrm { T V } } \leqslant \eta / 2 0$ , a Chernof bound shows that its output distribution is within $\exp ( - c ^ { \prime } \eta n )$ of $Q _ { \sigma } ^ { \otimes n }$ in total variation, for an absolute constant $c ^ { \prime } > 0$

## 5.4 Binomial Moment Construction

We follow the polynomial-correction method of Diakonikolas, Kane, and Sun [DKS22, Proposition 25]. For each fixed moment order $t ,$ we solve the moment equations directly on the discrete count grid. The resulting estimates, together with a direct treatment of the remaining parameter range, give the separation scale in Lemma 5.2 for every batch size.

Proof of Lemma 5.2. Constants denoted $C _ { t }$ may change from line to line and depend only on �. First suppose min $\{ m , \log ( 1 / \eta ) \} \geqslant R _ { t }$ , where $R _ { t } > t + 1$ is a suficiently large constant depending only on �. Set

$$
I : = \left\{ j \in \{ 0 , \ldots , m \} : | j - m / 2 | \leqslant { \frac { { \sqrt { m \operatorname* { m i n } \{ m , \log ( 1 / \eta ) \} } } } { 1 6 } } \right\} ,
$$

$$
z _ { j } : = \frac { 1 6 ( j - m / 2 ) } { \sqrt { m \operatorname* { m i n } \{ m , \log ( 1 / \eta ) \} } } .
$$

Since min $\{ m , \log ( 1 / \eta ) \} \leqslant m$ , we have

$$
\frac { 1 } { 1 6 } \operatorname* { m i n } \{ m , \log ( 1 / \eta ) \} \leqslant \frac { \sqrt { m \operatorname* { m i n } \{ m , \log ( 1 / \eta ) \} } } { 1 6 } \leqslant m / 1 6
$$

and $| I | = \Theta ( \sqrt { m \operatorname* { m i n } \{ m , \log ( 1 / \eta ) \} } / 1 6 )$ . Consider the $( t + 1 ) \times ( t + 1 )$ matrix

$$
M _ { u v } : = \frac { 1 } { | I | } \sum _ { j \in I } z _ { j } ^ { u + v } , \qquad 0 \leqslant u , v \leqslant t .
$$

These Riemann sums converge to

$$
H _ { u v } = \frac { 1 } { 2 } \int _ { - 1 } ^ { 1 } x ^ { u + v } d x \qquad \mathrm { a s } \frac { \sqrt { m \operatorname* { m i n } \{ m , \log ( 1 / \eta ) \} } } { 1 6 } \to \infty ,
$$

with error $O _ { t } ( 1 6 / \sqrt { m \operatorname* { m i n } \{ m , \log ( 1 / \eta ) \} } )$ . The matrix � is positive definite, since its quadratic form is the integral of the square of a nonzero polynomial. Thus, by increasing $R _ { t } ,$ , we have

$$
\| M ^ { - 1 } \| _ { \mathrm { o p } } \leqslant C _ { t } .\tag{26}
$$

The bound is uniform over � and �; odd � only shifts the grid by half a step.

Correcting the moments. Let $b _ { i } : = \mathbb { E } _ { N \sim p _ { 0 } } \big [ z _ { N } ^ { i } \big ] - \mathbb { E } _ { N \sim p _ { \Delta } } \big [ z _ { N } ^ { i } \big ]$ . We claim that

$$
| b _ { i } | \leqslant C _ { t } \frac { 1 6 m \Delta } { \sqrt { m \operatorname* { m i n } \{ m , \log ( 1 / \eta ) \} } } \leqslant C _ { t } c _ { t } \eta , \qquad 0 \leqslant i \leqslant t .\tag{27}
$$

Diferentiating the binomial formula gives, for any function $f$ on $\{ 0 , \ldots , m \}$

$$
\frac { d } { d s } \mathop { \mathbb { E } } _ { N \sim p _ { s } } ^ { \mathbb { E } } \bigl [ f ( N ) \bigr ] = m \mathop { \mathbb { E } } _ { Y \sim \mathrm { B i n } ( m - 1 , 1 / 2 + s ) } ^ { } \bigl [ f ( Y + 1 ) - f ( Y ) \bigr ] .
$$

For $1 \leqslant i \leqslant t$ and $f ( j ) = z _ { j } ^ { i } ,$ , the finite diference is

$$
\frac { 1 6 i } { \sqrt { m \operatorname * { m i n } \{ m , \log ( 1 / \eta ) \} } } \int _ { 0 } ^ { 1 } \left( \frac { 1 6 ( Y + u - m / 2 ) } { \sqrt { m \operatorname * { m i n } \{ m , \log ( 1 / \eta ) \} } } \right) ^ { i - 1 } d u .
$$

For $0 \leqslant s \leqslant \Delta$ and $0 \leqslant u \leqslant 1$ , the random variable $1 6 ( Y + u - m / 2 ) / \sqrt { m \operatorname * { m i n } \{ m , \log ( 1 / \eta ) \} }$ has bounded absolute moments of every fixed order. Its centered fluctuations have scale $1 6 / \sqrt { \operatorname* { m i n } \{ m , \log ( 1 / \eta ) \} } \leqslant$ $^ { 1 6 , }$ and its mean has absolute value at most $1 6 ( m \Delta + 1 ) / \sqrt { m \operatorname* { m i n } \{ m , \log ( 1 / \eta ) \} } \leqslant 2 .$ , after taking $c _ { t } \leqslant \frac { 1 } { 1 6 }$ and $R _ { t }$ large. Here we use $\mathbb { E } [ | Y - \mathbb { E } Y | ^ { r } ] \leqslant C _ { r } m ^ { r / 2 } ,$ , which follows by integrating Hoefding’s tail bound. Integrating the derivative over $[ 0 , \Delta ]$ proves (27); also $b _ { 0 } = 0$

Solve $M \lambda = b$ and define

$$
h ( z ) : = \sum _ { i = 0 } ^ { t } \lambda _ { i } z ^ { i } , \qquad d _ { j } : = \left\{ { h ( z _ { j } ) } / { | I | } , \begin{array} { l l } { j \in I , } \\ { 0 , } \end{array} \right. \quad \widetilde { p } ( j ) : = p _ { \Delta } ( j ) + d _ { j } .
$$

Then $\textstyle \sum _ { j } d _ { j } z _ { j } ^ { i } = b _ { i }$ for $0 \leqslant i \leqslant t$ , giving the moment identities in (22) and total mass one. By (26) and (27),

$$
\operatorname* { s u p } _ { | z | \leqslant 1 } | h ( z ) | \leqslant C _ { t } c _ { t } \eta , \qquad \sum _ { j } | d _ { j } | \leqslant C _ { t } c _ { t } \eta , \qquad \operatorname* { m a x } _ { j } | d _ { j } | \leqslant \frac { 1 6 C _ { t } c _ { t } \eta } { \sqrt { m \operatorname* { m i n } \{ m , \log ( 1 / \eta ) \} } } .\tag{28}
$$

Nonnegativity and distance bounds. For $j \in I ,$ both $j / m$ and $1 / 2 + \Delta$ lie in $[ 1 / 4 , 3 / 4 ]$ , and

$$
| j - m ( 1 / 2 + \Delta ) | \leqslant \frac { \sqrt { m \operatorname* { m i n } \{ m , \log ( 1 / \eta ) \} } } { 8 }
$$

since $m \Delta \leqslant \sqrt { m \operatorname* { m i n } \{ m , \log ( 1 / \eta ) \} } / 1 6$ . Stirling’s formula gives

$$
p _ { \Delta } ( j ) \geqslant \frac { c } { \sqrt { m } } \exp ( - m D ( j / m  1 / 2 + \Delta ) ) ,
$$

where $D ( \cdot \| \cdot )$ is binary relative entropy. The inequality log $x \leqslant x - 1$ implies

$$
D \left( j / m \parallel 1 / 2 + \Delta \right) \leqslant \frac { ( j / m - 1 / 2 - \Delta ) ^ { 2 } } { ( 1 / 2 + \Delta ) ( 1 / 2 - \Delta ) } \leqslant \frac { 1 6 } { 3 } ( j / m - 1 / 2 - \Delta ) ^ { 2 } .
$$

Therefore

$$
p _ { \Delta } ( j ) \geqslant \frac { c } { \sqrt { m } } e ^ { - \operatorname* { m i n } \{ m , \log ( 1 / \eta ) \} / 1 2 } \geqslant \frac { c \sqrt { \eta } } { \sqrt { m } } .
$$

Consequently,

$$
\frac { | d _ { j } | } { p _ { \Delta } ( j ) } \leqslant C _ { t } c _ { t } \frac { \sqrt { \eta } } { \sqrt { \operatorname* { m i n } \{ m , \log ( 1 / \eta ) \} } } \leqslant \frac { 1 } { 2 }
$$

after reducing $c _ { t }$ . Thus $0 \leqslant \widetilde { p } \leqslant \frac { 3 } { 2 } p _ { \Delta }$ , and (28) gives $\Vert \widetilde { p } - p _ { \Delta } \Vert _ { \mathrm { T V } } \leqslant \eta / 2 0$

By the binomial theorem,

$$
\begin{array} { l } { { \displaystyle \sum _ { j = 0 } ^ { m } \frac { p _ { \Delta } ( j ) ^ { 2 } } { p _ { 0 } ( j ) } = \sum _ { j = 0 } ^ { m } \binom { m } { j } \left[ 2 ( 1 / 2 + \Delta ) ^ { 2 } \right] ^ { j } \left[ 2 ( 1 / 2 - \Delta ) ^ { 2 } \right] ^ { m - j } } } \\ { { = ( 1 + 4 \Delta ^ { 2 } ) ^ { m } . } } \end{array}\tag{29}
$$

Hence

$$
1 + \chi ^ { 2 } ( \widetilde { p } \| p _ { 0 } ) \leqslant \frac { 9 } { 4 } ( 1 + 4 \Delta ^ { 2 } ) ^ { m } \leqslant 3 .
$$

The last inequality holds for suficiently small $c _ { t } ,$ since � $\Delta ^ { 2 } \leqslant c _ { t } ^ { 2 } \eta ^ { 2 } \log ( 1 / \eta )$ is uniformly small.

Remaining parameter range. I $\mathrm { f } \operatorname* { m i n } \{ m , \log ( 1 / \eta ) \} < R _ { t } .$ , take $\widetilde { p } = p _ { 0 }$ . By (29) and Cauchy–Schwarz,

$$
\Vert p _ { 0 } - p _ { \Delta } \Vert _ { \mathrm { T V } } \leqslant \frac { 1 } { 2 } \sqrt { ( 1 + 4 \Delta ^ { 2 } ) ^ { m } - 1 } \leqslant C \Delta \sqrt { m } \leqslant C c _ { t } \eta \sqrt { R _ { t } } \leqslant \eta / 2 0 .
$$

All moments match and the chi-square divergence is zero. Every choice of constants depends only on �. □

AI Disclosure. We used Codex and GPT Pro to assist with technical development and exposition, including the diagonal-majorizer and semidefinite-duality formulation, the application of the symmetric Grothendieck inequality, the approximate-DP obstruction argument, and the SQ lowerbound proof. The authors are responsible for the final manuscript.

## References

[ABMT25] Prashanti Anderson, Ainesh Bakshi, Mahbod Majid, and Stefan Tiegel. Sample-optimal private regression in polynomial time. In Proceedings of the 57th Annual ACM Symposium on the Theory of Computing, STOC ’25, pages 2341–2349. ACM, 2025.

[AKM<sup>+</sup>25] Sushant Agarwal, Gautam Kamath, Mahbod Majid, Argyris Mouzakis, Rose Silver, and Jonathan Ullman. Private mean estimation with person-level diferential privacy. In Proceedings of the 36th Annual ACM-SIAM Symposium on Discrete Algorithms, SODA ’25, pages 2819–2880. SIAM, 2025.

[AKT<sup>+</sup>23] Daniel Alabi, Pravesh K Kothari, Pranay Tankala, Prayaag Venkat, and Fred Zhang. Privately estimating a Gaussian: Eficient, robust, and optimal. In Proceedings ofthe 55th Annual ACM Symposium on the Theory ofComputing, STOC ’23, pages 483–496. ACM, 2023.

[AL22] Hassan Ashtiani and Christopher Liaw. Private and polynomial time algorithms for learning Gaussians and beyond. In Proceedings of the 35th Annual Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 1075–1076. PMLR, 2022.

[AL24] Hilal Asi and Daogao Liu. User-level diferentially private stochastic convex optimization: Eficient algorithms with optimal rates. In Proceedings of the 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pages 4240–4248. PMLR, 2024.

[ALS23] Jayadev Acharya, Yuhan Liu, and Ziteng Sun. Discrete distribution estimation under user-level local diferential privacy. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 8561–8585. PMLR, 2023.

[AM20] Marco Avella-Medina. The role of robust statistics in private data analysis. Chance, 33(4):37–42, 2020.

[AM21] Marco Avella-Medina. Privacy-preserving parametric inference: a case for robust statistics. Journal of the American Statistical Association, 116(534):969–983, 2021.

[AMN24] Hassan Ashtiani, Mahbod Majid, and Shyam Narayanan. Sample-eficient private learning of mixtures of Gaussians. In Advances in Neural Information Processing Systems 37, NeurIPS ’24, pages 98994–99039. Curran Associates, Inc., 2024.

[AUZ23] Hilal Asi, Jonathan Ullman, and Lydia Zakynthinou. From robustness to privacy and back. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 1121–1146. PMLR, 2023.

[BGS<sup>+</sup>21] Gavin Brown, Marco Gaboardi, Adam Smith, Jonathan Ullman, and Lydia Zakynthinou. Covariance-aware private mean estimation without private covariance estimation. In Advances in Neural Information Processing Systems 34, NeurIPS ’21, pages 7950–7964. Curran Associates, Inc., 2021.

[BKSW19] Mark Bun, Gautam Kamath, Thomas Steinke, and Zhiwei Steven Wu. Private hypothesis selection. In Advances in Neural Information Processing Systems 32, NeurIPS ’19, pages 156–167. Curran Associates, Inc., 2019.

[BS23] Raef Bassily and Ziteng Sun. User-level private stochastic convex optimization with optimal rates. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 1838–1851. PMLR, 2023.

[Can20] Clément L. Canonne. A short note on learning discrete distributions. arXiv preprint arXiv:2002.11457, 2020.

[CDd<sup>+</sup>24] Hongjie Chen, Jingqiu Ding, Tommaso d’Orsi, Yiding Hua, Chih-Hung Liu, and David Steurer. Private graphon estimation via sum-of-squares. In Proceedings of the 56th Annual ACM Symposium on the Theory of Computing, STOC ’24, pages 172–182. ACM, 2024.

[CDHS24] Hongjie Chen, Jingqiu Ding, Yiding Hua, and David Steurer. Private edge density estimation for random graphs: Optimal, eficient and robust. In Advances in Neural Information Processing Systems 37, NeurIPS ’24, pages 90771–90817. Curran Associates, Inc., 2024.

[CDMM26] Sitan Chen, Jingqiu Ding, Mahbod Majid, and Walter McKelvie. Computation-utilityprivacy tradeofs in Bayesian estimation. In Proceedings of the 58th Annual ACM Symposium on the Theory of Computing, STOC ’26, pages 1616–1626. ACM, 2026.

[CFMT22] Rachel Cummings, Vitaly Feldman, Audra McMillan, and Kunal Talwar. Mean estimation with user-level privacy under data heterogeneity. In Advances in Neural Information Processing Systems 35, NeurIPS ’22, pages 29139–29151. Curran Associates, Inc., 2022.

[CLM20a] Sitan Chen, Jerry Li, and Ankur Moitra. Eficiently learning structured distributions from untrusted batches. In Proceedings of the 52nd Annual ACM Symposium on the Theory of Computing, STOC ’20, pages 960–973. ACM, 2020.

[CLM20b] Sitan Chen, Jerry Li, and Ankur Moitra. Learning structured distributions from untrusted batches: Faster and simpler. In Advances in Neural Information Processing Systems 33, NeurIPS ’20, pages 4512–4523. Curran Associates, Inc., 2020.

[CS23] Julien Chhor and Flore Sentenac. Robust estimation of discrete distributions under local diferential privacy. In Proceedings ofthe 34th International Conference on Algorithmic Learning Theory, volume 201 of Proceedings of Machine Learning Research, pages 411–446. PMLR, 2023.

[DHS15] Ilias Diakonikolas, Moritz Hardt, and Ludwig Schmidt. Diferentially private learning of structured discrete distributions. In Advances in Neural Information Processing Systems 28, NIPS ’15, pages 2566–2574. Curran Associates, Inc., 2015.

[DJKS23] Abhimanyu Das, Ayush Jain, Weihao Kong, and Rajat Sen. Eficient list-decodable regression using batches. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 7025–7065. PMLR, 2023.

[DK23] Ilias Diakonikolas and Daniel M. Kane. Algorithmic High-Dimensional Robust Statistics. Cambridge University Press, 2023.

[DKK<sup>+</sup>16] Ilias Diakonikolas, Gautam Kamath, Daniel M. Kane, Jerry Li, Ankur Moitra, and Alistair Stewart. Robust estimators in high dimensions without the computational intractability. In Proceedings of the 57th Annual IEEE Symposium on Foundations of Computer Science, FOCS ’16, pages 655–664. IEEE Computer Society, 2016. Journal version: SIAM Journal on Computing, 48(2):742–864, 2019.

[DKK<sup>+</sup>25] Ilias Diakonikolas, Daniel Kane, Sushrut Karmalkar, Sihan Liu, and Thanasis Pittas. Batch list-decodable linear regression via higher moments. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 13622–13642. PMLR, 2025.

[DKS17] Ilias Diakonikolas, Daniel M. Kane, and Alistair Stewart. Statistical query lower bounds for robust estimation of high-dimensional Gaussians and Gaussian mixtures. In Proceedings of the 58th Annual IEEE Symposium on Foundations of Computer Science, FOCS ’17, pages 73–84. IEEE Computer Society, 2017.

[DKS22] Ilias Diakonikolas, Daniel M. Kane, and Yuxin Sun. Optimal SQ lower bounds for robustly learning discrete product distributions and Ising models. In Proceedings of the 35th Annual Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 3936–3978. PMLR, 2022.

[DL09] Cynthia Dwork and Jing Lei. Diferential privacy and robust statistics. In Proceedings of the 41st Annual ACM Symposium on the Theory of Computing, STOC ’09, pages 371–380. ACM, 2009.

[DMNS06] Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Proceedings of the 3rd Conference on Theory of Cryptography, TCC ’06, pages 265–284, Berlin, Heidelberg, 2006. Springer. Journal version: Journal of Privacy and Confidentiality, 7(3):17–51, 2017.

[DRV10] Cynthia Dwork, Guy N. Rothblum, and Salil Vadhan. Boosting and diferential privacy. In Proceedings of the 51st Annual IEEE Symposium on Foundations of Computer Science, FOCS ’10, pages 51–60. IEEE Computer Society, 2010.

[EMN22] Hossein Esfandiari, Vahab Mirrokni, and Shyam Narayanan. Tight and robust private mean estimation with few users. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 16383–16412. PMLR, 2022.

[FL20] Shmuel Friedland and Lek-Heng Lim. Symmetric Grothendieck inequality. arXiv preprint arXiv:2003.07345, 2020.

[GDD22] Antonious M Girgis, Deepesh Data, and Suhas Diggavi. Distributed user-level private mean estimation. In Proceedings of the 2022 IEEE International Symposium on Information Theory, ISIT ’22, pages 2196–2201. IEEE, 2022.

[GH22] Kristian Georgiev and Samuel B Hopkins. Privacy induces robustness: Informationcomputation gaps and sparse mean estimation. In Advances in Neural Information Processing Systems 35, NeurIPS ’22, pages 6829–6842. Curran Associates, Inc., 2022.

[GKK<sup>+</sup>23a] Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Raghu Meka, and Chiyuan Zhang. On user-level private convex optimization. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 11283–11299. PMLR, 2023.

[GKK 23b] Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Raghu Meka, and Chiyuan Zhang. User-level diferential privacy with few examples per user. In Advances in Neural Information Processing Systems 36, NeurIPS ’23, pages 19263–19290. Curran Associates, Inc., 2023.

[GKLM26] Badih Ghazi, Ravi Kumar, Daogao Liu, and Pasin Manurangsi. Nearly linear-time user-level DP-SCO with optimal rates. In Proceedings ofthe 39th Annual Conference on Learning Theory, volume 336 of Proceedings ofMachine Learning Research, pages 2601–2636. PMLR, 2026.

[GKM21] Badih Ghazi, Ravi Kumar, and Pasin Manurangsi. User-level diferentially private learning via correlated sampling. In Advances in Neural Information Processing Systems 34, NeurIPS ’21, pages 20172–20184. Curran Associates, Inc., 2021.

[GKMN21] Badih Ghazi, Ravi Kumar, Pasin Manurangsi, and Thao Nguyen. Robust and private learning of halfspaces. In The 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pages 1603–1611. PMLR, 2021.

[GRST24] Anand Jerry George, Lekshmi Ramesh, Aditya Vikram Singh, and Himanshu Tyagi. Continual mean estimation under user-level privacy. IEEE Journal on Selected Areas in Information Theory, 5:28–43, 2024.

[HKM22] Samuel B Hopkins, Gautam Kamath, and Mahbod Majid. Eficient mean estimation with pure diferential privacy via a sum-of-squares exponential mechanism. In Proceedings of the 54th Annual ACM Symposium on the Theory of Computing, STOC ’22, pages 1406–1417. ACM, 2022.

[HKMN23] Samuel B Hopkins, Gautam Kamath, Mahbod Majid, and Shyam Narayanan. Robustness implies privacy in statistical estimation. In Proceedings of the 55th Annual ACM Symposium on the Theory of Computing, STOC ’23, pages 497–506. ACM, 2023. Full version: arXiv:2212.05015.

[JO20a] Ayush Jain and Alon Orlitsky. A general method for robust learning from batches. In Advances in Neural Information Processing Systems 33, NeurIPS ’20, pages 21775–21785. Curran Associates, Inc., 2020.

[JO20b] Ayush Jain and Alon Orlitsky. Optimal robust learning of discrete distributions from batches. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 4651–4660. PMLR, 2020.

[JO21] Ayush Jain and Alon Orlitsky. Robust density estimation from batches: The best things in life are (nearly) free. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 4698–4708. PMLR, 2021.

[JSK<sup>+</sup>24] Ayush Jain, Rajat Sen, Weihao Kong, Abhimanyu Das, and Alon Orlitsky. Linear regression using heterogeneous data batches. In Advances in Neural Information Processing Systems 37, NeurIPS ’24, pages 87061–87103. Curran Associates, Inc., 2024.

[Kam24] Gautam Kamath. The broader landscape of robustness in algorithmic statistics. IEEE BITS the Information Theory Magazine, 4(3):3–21, 2024.

[KMA<sup>+</sup>21] Peter Kairouz, H. Brendan McMahan, Brendan Avent, et al. Advances and open problems in federated learning. Foundations and Trends in Machine Learning, 14(1–2):1– 210, 2021.

[KMV22] Pravesh K Kothari, Pasin Manurangsi, and Ameya Velingker. Private robust estimation by stabilizing convex relaxations. In Proceedings of the 35th Annual Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 723–777. PMLR, 2022.

[KSU20] Gautam Kamath, Vikrant Singhal, and Jonathan Ullman. Private mean estimation of heavy-tailed distributions. In Proceedings of the 33rd Annual Conference on Learning Theory, volume 125 of Proceedings of Machine Learning Research, pages 2204–2235. PMLR, 2020.

[KU20] Gautam Kamath and Jonathan Ullman. A primer on private statistics. arXiv preprint arXiv:2005.00010, 2020.

[LBY23] Mengchu Li, Thomas B Berrett, and Yi Yu. On robustness and local diferential privacy. The Annals of Statistics, 51(2):717–737, 2023.

[LJK<sup>+</sup>23] Xiyang Liu, Prateek Jain, Weihao Kong, Sewoong Oh, and Arun Sai Suggala. Label robust and diferentially private linear regression: Computational and statistical eficiency. In Advances in Neural Information Processing Systems 36, NeurIPS ’23, pages 23019–23033. Curran Associates, Inc., 2023.

[LKKO21] Xiyang Liu, Weihao Kong, Sham Kakade, and Sewoong Oh. Robust and diferentially private mean estimation. In Advances in Neural Information Processing Systems 34, NeurIPS ’21, pages 3887–3901. Curran Associates, Inc., 2021.

[LKO22] Xiyang Liu, Weihao Kong, and Sewoong Oh. Diferential privacy and robust statistics in high dimensions. In Proceedings of the 35th Annual Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 1167–1246. PMLR, 2022.

[LRV16] Kevin A. Lai, Anup B. Rao, and Santosh Vempala. Agnostic estimation of mean and covariance. In Proceedings of the 57th Annual IEEE Symposium on Foundations of Computer Science, FOCS ’16, pages 665–674. IEEE Computer Society, 2016.

[LSA<sup>+</sup>21] Daniel Levy, Ziteng Sun, Kareem Amin, Satyen Kale, Alex Kulesza, Mehryar Mohri, and Ananda Theertha Suresh. Learning with user-level privacy. In Advances in Neural

Information Processing Systems 34, NeurIPS ’21, pages 12466–12479. Curran Associates, Inc., 2021.

[LSY<sup>+</sup>20] Yuhan Liu, Ananda Theertha Suresh, Felix Yu, Sanjiv Kumar, and Michael Riley. Learning discrete distributions: User vs item-level privacy. In Advances in Neural Information Processing Systems 33, NeurIPS ’20, pages 20965–20976. Curran Associates, Inc., 2020.

[MMR<sup>+</sup>17] H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas. Communication-eficient learning of deep networks from decentralized data. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, volume 54 of Proceedings of Machine Learning Research, pages 1273–1282. PMLR, 2017.

[QV18] Mingda Qiao and Gregory Valiant. Learning discrete distributions from untrusted batches. In Proceedings of the 9th Conference on Innovations in Theoretical Computer Science, ITCS ’18, pages 47:1–47:20, Dagstuhl, Germany, 2018. Schloss Dagstuhl–Leibniz-Zentrum fuer Informatik.

[RC21] Kelly Ramsay and Shoja’eddin Chenouri. Diferentially private depth functions and their associated medians. arXiv preprint arXiv:2101.02800, 2021.

[RJC25] Kelly Ramsay, Aukosh Jagannath, and Shoja’eddin Chenouri. Diferentially private multivariate medians. Journal of Machine Learning Research, 26(255):1–52, 2025.

[SM22] Aleksandra Slavkovic and Roberto Molinari. Perturbed M-estimation: A further investigation of robust statistics for diferential privacy. In Alicia L. Carriquiry, Judith M. Tanur, and William F. Eddy, editors, Statistics in the Public Interest: In Memory of Stephen E. Fienberg, pages 337–361. Springer, 2022.

[TCK<sup>+</sup>22] Eliad Tsfadia, Edith Cohen, Haim Kaplan, Yishay Mansour, and Uri Stemmer. Friendly-Core: Practical diferentially private aggregation. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 21828–21863. PMLR, 2022.

[Wai19] Martin J. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint. Cambridge University Press, 2019.

[ZLH19] Ligeng Zhu, Zhĳian Liu, and Song Han. Deep leakage from gradients. In Advances in Neural Information Processing Systems 32, NeurIPS ’19, pages 14774–14784. Curran Associates, Inc., 2019.

[ZLS<sup>+</sup>24] Puning Zhao, Lifeng Lai, Li Shen, Qingming Li, Jiafei Wu, and Zhe Liu. A Huber loss minimization approach to mean estimation under user-level diferential privacy. In Advances in Neural Information Processing Systems 37, NeurIPS ’24, pages 130018–130056. Curran Associates, Inc., 2024.

## A Details for the Approximate-DP Section

This appendix proves the obstruction argument of Hopkins, Kamath, Majid, and Narayanan [HKMN23] in sign directions $\sigma \in \{ \pm 1 \} ^ { k }$ , which is the geometric step needed for

the $\ell _ { 1 }$ -diameter bound in Lemma 4.14. We follow Lemma 5.13 in the full version of [HKMN23], with sign directions replacing Euclidean ones and exact $\ell _ { 1 }$ duality replacing an $\ell _ { 2 } \mathrm { - t o - } \ell _ { 1 }$ conversion.

Lemma A.1 (One-dimensional obstruction for low-variance certificates). Let $z _ { 1 } , \dots , z _ { n } \in \mathbb { R }$ and suppose that at least $n / 4$ of the $z _ { i } \prime _ { s }$ are at least 20. Let $\mathcal { L }$ be a degree-6 pseudoexpectation over variables $\{ w _ { i } , z _ { i } ^ { \prime } \} _ { i \in [ n ] }$ such that

1. $w _ { i } ^ { 2 } = w _ { i } f o r e v e r y i \in [ n ] ,$

2. $\begin{array} { r } { \sum _ { i = 1 } ^ { n } w _ { i } \geqslant 0 . 9 9 n , } \end{array}$

3. $w _ { i } ( z _ { i } ^ { \prime } - z _ { i } ) = 0 f o r e v e r y i \in [ n ] ,$

4. $\begin{array} { r } { \mathcal { L } \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } z _ { i } ^ { \prime } \right] = 0 . } \end{array}$

Then

$$
\mathcal { L } \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( z _ { i } ^ { \prime } - \frac { 1 } { n } \sum _ { j = 1 } ^ { n } z _ { j } ^ { \prime } \right) ^ { 2 } \right] \geqslant 2 .
$$

Proof. Repeating each index four times does not change the hypotheses or conclusion, so we may assume that � is divisible by 4. Write $\begin{array} { r } { \bar { z } ^ { \prime } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } z _ { i } ^ { \prime } . } \end{array}$ . Suppose for contradiction that

$$
\mathcal { L } \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( z _ { i } ^ { \prime } - \bar { z } ^ { \prime } ) ^ { 2 } \right] < 2 .\tag{30}
$$

Because $w _ { i } ^ { 2 } = w _ { i } ,$ , we also have $( 1 - w _ { i } ) ^ { 2 } = 1 - w _ { i }$ . Hence $0 \leqslant \mathcal { L } [ w _ { i } ] \leqslant 1$ for every $i ,$ and

$$
w _ { i } w _ { j } - w _ { i } - w _ { j } + 1 = ( 1 - w _ { i } ) ( 1 - w _ { j } )
$$

is a square modulo the Boolean equalities, so $\mathcal { L } [ w _ { i } w _ { j } ] \geqslant \mathcal { L } [ w _ { i } ] + \mathcal { L } [ w _ { j } ] - 1$ . Also $0 \leqslant \mathcal { L } [ w _ { i } w _ { j } ] \leqslant 1$ Write $z _ { ( 1 ) } \leqslant \cdots \leqslant z _ { ( n ) }$ for the ordered observations and set $C _ { 1 } = z _ { ( n / 4 ) }$ and $C _ { 2 } = z _ { ( 3 n / 4 + 1 ) }$ . If $C _ { 2 } - C _ { 1 } > 8$ , then there exist sets $S _ { + } , S _ { - } \subseteq [ n ]$ with $\vert S _ { + } \vert = \vert S _ { - } \vert = n / 4$ such that $z _ { i } - z _ { j } \geqslant 8$ for every $i \in S _ { + }$ and $j \in S _ { - }$ . Using Condition 3, we obtain

$$
\begin{array} { r l } { \mathcal { L } \Bigg [ \displaystyle \sum _ { i = 1 , j = 1 } ^ { n } ( z _ { i } ^ { \prime } - z _ { j } ^ { \prime } ) ^ { 2 } \Bigg ] \approx } & { \mathcal { L } \Bigg [ \displaystyle \sum _ { i , j = 1 } ^ { n } w _ { i } w _ { j } ( z _ { i } ^ { \prime } - z _ { j } ^ { \prime } ) ^ { 2 } \Bigg ] } \\ & { = \displaystyle \sum _ { i , j = 1 } ^ { n } ( z _ { i } - z _ { i } ) ^ { 2 } \mathcal { L } [ w _ { i } w _ { j } ] } \\ & { \geqslant 2 \displaystyle \sum _ { i \in \mathbb { N } _ { \neq j } \mathbb { S } _ { - } } ( z _ { i } - z _ { j } ) ^ { 2 } \mathcal { L } [ w _ { i } w _ { j } ] } \\ & { \geqslant 2 \cdot 8 ^ { 2 } \displaystyle \sum _ { i \in \mathbb { N } _ { \neq j } \mathbb { S } _ { - } } \mathcal { L } [ w _ { i } w _ { j } ] } \\ & { \geqslant 2 \cdot 8 ^ { 2 } \displaystyle \sum _ { i \in \mathbb { S } _ { - } \mathbb { S } _ { - } } ( \mathcal { L } [ w _ { i } ] + \mathcal { L } [ w _ { j } ] - 1 ) . } \end{array}
$$

Since $| S _ { + } | = | S _ { - } | = n / 4$ and $\textstyle \sum _ { i } { \mathcal { L } } [ w _ { i } ] \geqslant 0 . 9 9 n$ , the total expected weight on $S _ { + } \cup S _ { - }$ is at least $0 . 4 9 n$ Therefore,

$$
\begin{array} { l } { \displaystyle \sum _ { i \in S _ { + } , j \in S _ { - } } \left( \mathcal { L } [ w _ { i } ] + \mathcal { L } [ w _ { j } ] - 1 \right) = \displaystyle \frac { n } { 4 } \sum _ { i \in S _ { + } } \mathcal { L } [ w _ { i } ] + \frac { n } { 4 } \sum _ { j \in S _ { - } } \mathcal { L } [ w _ { j } ] - \frac { n ^ { 2 } } { 1 6 } } \\ { \displaystyle \geqslant \frac { n } { 4 } ( 0 . 4 9 n ) - \frac { n ^ { 2 } } { 1 6 } > 0 . 0 6 n ^ { 2 } . } \end{array}
$$

Thus $\mathcal { L } [ \sum _ { i , j } ( z _ { i } ^ { \prime } - z _ { j } ^ { \prime } ) ^ { 2 } ] > 6 n ^ { 2 }$ . Using the identity

$$
\sum _ { i , j = 1 } ^ { n } ( z _ { i } ^ { \prime } - z _ { j } ^ { \prime } ) ^ { 2 } = 2 n \sum _ { i = 1 } ^ { n } ( z _ { i } ^ { \prime } - { \bar { z } } ^ { \prime } ) ^ { 2 } ,
$$

we conclude that

$$
\mathcal { L } \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( z _ { i } ^ { \prime } - \bar { z } ^ { \prime } ) ^ { 2 } \right] > 3 ,
$$

contradicting (30). Hence we may assume that $C _ { 2 } - C _ { 1 } \leqslant 8 .$

Let � be the middle $n / 2$ indices after sorting the $z _ { i } \prime \mathrm { s }$ . Then there exists $C \geqslant$ 16 such that

$$
z _ { i } \in [ C - 4 , C + 4 ] \qquad { \mathrm { f o r ~ a l l ~ } } i \in S ,
$$

because at least $n / 4$ of the $z _ { i } \mathrm { ' s }$ are at least 20. Condition 2 implies that $\begin{array} { r } { \sum _ { i \in S } \mathcal { L } [ w _ { i } ] \geqslant 0 . 4 9 n } \end{array}$ , and so

$$
\mathcal { L } \left[ \sum _ { i \in S } w _ { i } z _ { i } ^ { \prime } \right] = \sum _ { i \in S } z _ { i } \mathcal { L } [ w _ { i } ] \geqslant ( C - 4 ) \sum _ { i \in S } \mathcal { L } [ w _ { i } ] \geqslant 0 . 4 9 ( C - 4 ) n .\tag{31}
$$

Likewise,

$$
\mathcal { L } \left[ \left( \sum _ { i \in S } w _ { i } z _ { i } ^ { \prime } \right) ^ { 2 } \right] = \sum _ { i , j \in S } z _ { i } z _ { j } \mathcal { L } [ w _ { i } w _ { j } ] \leqslant ( C + 4 ) ^ { 2 } \Big ( \frac { n } { 2 } \Big ) ^ { 2 } .\tag{32}
$$

Next, write

$$
\bar { z } _ { S } ^ { \prime } : = \frac 1 { | S | } \sum _ { i \in S } z _ { i } ^ { \prime } .
$$

Because $\left| S \right| = n / 2$ , we have

$$
\sum _ { i \in S } ( z _ { i } ^ { \prime } ) ^ { 2 } = \sum _ { i \in S } ( z _ { i } ^ { \prime } - \bar { z } _ { S } ^ { \prime } ) ^ { 2 } + \frac { 1 } { | S | } \Bigg ( \sum _ { i \in S } z _ { i } ^ { \prime } \Bigg ) ^ { 2 } .
$$

Since $\bar { z } _ { S } ^ { \prime }$ minimizes $\textstyle \sum _ { i \in S } ( z _ { i } ^ { \prime } - c ) ^ { 2 }$ over all constants $c ,$ we have

$$
\sum _ { i \in S } ( z _ { i } ^ { \prime } - \bar { z } _ { S } ^ { \prime } ) ^ { 2 } \leqslant \sum _ { i \in S } ( z _ { i } ^ { \prime } - \bar { z } ^ { \prime } ) ^ { 2 } \leqslant \sum _ { i = 1 } ^ { n } ( z _ { i } ^ { \prime } - \bar { z } ^ { \prime } ) ^ { 2 } .
$$

Therefore,

$$
\sum _ { i \in S } ( z _ { i } ^ { \prime } ) ^ { 2 } \leqslant \sum _ { i = 1 } ^ { n } ( z _ { i } ^ { \prime } - \bar { z } ^ { \prime } ) ^ { 2 } + \frac { 2 } { n } \Bigg ( \sum _ { i \in S } z _ { i } ^ { \prime } \Bigg ) ^ { 2 } .
$$

Taking ℒ of both sides and using (30) yields

$$
\mathcal { L } \left[ \sum _ { i \in S } ( z _ { i } ^ { \prime } ) ^ { 2 } \right] \leqslant 2 n + \frac { 2 } { n } \mathcal { L } \left[ \left( \sum _ { i \in S } z _ { i } ^ { \prime } \right) ^ { 2 } \right] .\tag{33}
$$

Define

$$
E _ { \mathrm { d r o p } } : = \mathcal { L } \left[ \left( \sum _ { i \in S } ( 1 - w _ { i } ) z _ { i } ^ { \prime } \right) ^ { 2 } \right] .
$$

Apply the degree-6 polynomial weighted Cauchy–Schwarz certificate from the proof of Lemma 3.8 with $b _ { i } = 1 - w _ { i }$ . Condition 2 and the Boolean identities for $b _ { i }$ then give

$$
\begin{array} { l } { \displaystyle E _ { \mathrm { d r o p } } \leqslant \mathcal { L } \Bigg [ \left( \displaystyle \sum _ { i \in S } ( 1 - w _ { i } ) \right) \left( \displaystyle \sum _ { i \in S } ( 1 - w _ { i } ) ( z _ { i } ^ { \prime } ) ^ { 2 } \right) \Bigg ] } \\ { \leqslant 0 . 0 1 n \mathcal { L } \Bigg [ \displaystyle \sum _ { i \in S } ( 1 - w _ { i } ) ( z _ { i } ^ { \prime } ) ^ { 2 } \Bigg ] } \\ { \leqslant 0 . 0 1 n \mathcal { L } \Bigg [ \displaystyle \sum _ { i \in S } ( z _ { i } ^ { \prime } ) ^ { 2 } \Bigg ] . } \end{array}
$$

Combining this with (33), and then using

$$
\sum _ { i \in S } z _ { i } ^ { \prime } = \sum _ { i \in S } ( 1 - w _ { i } ) z _ { i } ^ { \prime } + \sum _ { i \in S } w _ { i } z _ { i } ^ { \prime } ,
$$

followed by $( a + b ) ^ { 2 } \leqslant 2 a ^ { 2 } + 2 b ^ { 2 }$ , gives

$$
\begin{array} { l } { \displaystyle { E _ { \mathrm { d r o p } } \leqslant 0 . 0 1 n \left[ 2 n + \frac { 2 } { n } \mathcal { L } \left[ \left( \sum _ { i \in S } z _ { i } ^ { \prime } \right) ^ { 2 } \right] \right] } } \\ { \displaystyle { \leqslant 0 . 0 2 n ^ { 2 } + 0 . 0 4 E _ { \mathrm { d r o p } } + 0 . 0 4 \mathcal { L } \left[ \left( \sum _ { i \in S } w _ { i } z _ { i } ^ { \prime } \right) ^ { 2 } \right] } . } \end{array}
$$

Rearranging and applying (32),

$$
E _ { \mathrm { d r o p } } \leqslant \frac { 1 } { 2 4 } n ^ { 2 } + \frac { 1 } { 9 6 } ( C + 4 ) ^ { 2 } n ^ { 2 } \leqslant 0 . 0 2 C ^ { 2 } n ^ { 2 } ,
$$

where the last inequality uses $C \geqslant 1 6$ . Hence

$$
\left| \mathcal { L } \left[ \sum _ { i \in S } ( 1 - w _ { i } ) z _ { i } ^ { \prime } \right] \right| \leqslant \sqrt { E _ { \mathrm { d r o p } } } \leqslant 0 . 1 5 C n .\tag{34}
$$

From (31) and (34), and using $C \geqslant 1 6 ,$ , we obtain

$$
\mathcal { L } \left[ \sum _ { i \in S } z _ { i } ^ { \prime } \right] = \mathcal { L } \left[ \sum _ { i \in S } w _ { i } z _ { i } ^ { \prime } \right] + \mathcal { L } \left[ \sum _ { i \in S } ( 1 - w _ { i } ) z _ { i } ^ { \prime } \right] \geqslant 0 . 3 5 C n - 0 . 1 5 C n \geqslant 0 . 2 C n .
$$

Condition 4 says $\begin{array} { r } { \mathcal { L } [ \sum _ { i = 1 } ^ { n } z _ { i } ^ { \prime } ] = 0 } \end{array}$ , so

$$
\mathcal { L } \left[ \sum _ { i \in S } z _ { i } ^ { \prime } - \frac { 1 } { 2 } \sum _ { i = 1 } ^ { n } z _ { i } ^ { \prime } \right] \geqslant 0 . 2 C n .
$$

Another application of Cauchy–Schwarz yields

$$
\mathcal { L } \left[ \left( \sum _ { i \in S } z _ { i } ^ { \prime } - \frac { 1 } { 2 } \sum _ { i = 1 } ^ { n } z _ { i } ^ { \prime } \right) ^ { 2 } \right] \geqslant 0 . 0 4 C ^ { 2 } n ^ { 2 } .\tag{35}
$$

Let $R = [ n ] \backslash S ,$ so $| R | = n / 2$ . Then

$$
\sum _ { i \in S } z _ { i } ^ { \prime } - \frac { 1 } { 2 } \sum _ { i = 1 } ^ { n } z _ { i } ^ { \prime } = \frac { 1 } { 2 } \Biggl ( \sum _ { i \in S } z _ { i } ^ { \prime } - \sum _ { j \in R } z _ { j } ^ { \prime } \Biggr ) = \frac { 1 } { n } \sum _ { i \in S , j \in R } ( z _ { i } ^ { \prime } - z _ { j } ^ { \prime } ) .
$$

Applying Cauchy–Schwarz once more,

$$
\left( \frac { 1 } { n } \sum _ { i \in S , j \in R } ( z _ { i } ^ { \prime } - z _ { j } ^ { \prime } ) \right) ^ { 2 } \leqslant \frac { 1 } { 4 } \sum _ { i \in S , j \in R } ( z _ { i } ^ { \prime } - z _ { j } ^ { \prime } ) ^ { 2 } .
$$

Combining this with (35) shows

$$
\mathcal { L } \left[ \sum _ { i \in S , j \in R } ( z _ { i } ^ { \prime } - z _ { j } ^ { \prime } ) ^ { 2 } \right] \geqslant 0 . 1 6 C ^ { 2 } n ^ { 2 } .
$$

Since the left-hand side is at most $\mathcal { L } [ \Sigma _ { i , j } ( z _ { i } ^ { \prime } - z _ { j } ^ { \prime } ) ^ { 2 } ]$ , we conclude that

$$
2 n \cdot \mathcal { L } \left[ \sum _ { i = 1 } ^ { n } ( z _ { i } ^ { \prime } - \bar { z } ^ { \prime } ) ^ { 2 } \right] = \mathcal { L } \left[ \sum _ { i , j = 1 } ^ { n } ( z _ { i } ^ { \prime } - z _ { j } ^ { \prime } ) ^ { 2 } \right] \geqslant 0 . 1 6 C ^ { 2 } n ^ { 2 } > 4 n ^ { 2 } ,
$$

again contradicting (30). This contradiction proves the lemma.

Remark A.2 (Approximate certificates). We first treat exact pseudoexpectations in the proof of Lemma 4.14. Every displayed inference in its one-dimensional obstruction argument is part of a fixed degree-6 derivation whose coeficient norm is bounded by $\mathrm { p o l y } ( n , k , m )$ . Since $( 1 / m + \tau _ { \mathrm { c o v } } ) ^ { - 1 / 2 } \ \leqslant \ \sqrt { m }$ , normalizing the projections by this factor preserves the polynomial coeficient bound. Therefore, if a linear functional ℒ only �-approximately satisfies the constraints in $\mathcal { B } _ { \eta ^ { * } n } ( \boldsymbol { y } )$ , the derivation incurs an additive loss

$$
{ \cal O } \left( \tau \mathrm { p o l y } ( n , k , m ) \eta ^ { * } n \right) .
$$

Choosing

$$
\tau = { \frac { 1 } { \mathrm { p o l y } ( n , k , m , \alpha ^ { - 1 } ) } }
$$

makes the accumulated coeficient error �(1), so the one-dimensional obstruction argument also applies to approximate certificates.

Full proofofLemma 4.14. Let $\mathcal { L } _ { 1 } , \mathcal { L } _ { 2 }$ be $( \alpha , \tau , \eta ^ { * } n )$ -certificates for $\nu _ { 1 } , \nu _ { 2 } ,$ , and write $\mu _ { \mathcal { L } _ { j } } : = \mathcal { L } _ { j } \mu ^ { \prime }$ for $j \in \{ 1 , 2 \}$ . For a fixed certificate $\mathcal { L } _ { j } ,$ we work with the observed and repaired one-dimensional projections of the batch-average vectors in $y$ and the pseudo-points $y _ { i } ^ { \prime }$ from $\mathcal { B } _ { \eta ^ { * } n } ( \boldsymbol { y } )$ . We first show that for every sign vector $\sigma \in \{ \pm 1 \} ^ { k }$

$$
\left| \langle \mu _ { \mathcal { L } _ { 1 } } - \mu _ { \mathcal { L } _ { 2 } } , \sigma \rangle \right| \leqslant O \left( \sqrt { \frac { 1 } { m } + \tau _ { \mathrm { c o v } } } \right) .\tag{36}
$$

Fix $j \in \{ 1 , 2 \}$ and a sign vector �. Set

$$
a _ { i } : = \frac { 1 } { \sqrt { 1 / m + \tau _ { \mathrm { c o v } } } } \langle y _ { i } - \mu _ { \mathcal { L } _ { j } } , \sigma \rangle \qquad \mathrm { a n d } \qquad a _ { i } ^ { \prime } : = \frac { 1 } { \sqrt { 1 / m + \tau _ { \mathrm { c o v } } } } \langle y _ { i } ^ { \prime } - \mu _ { \mathcal { L } _ { j } } , \sigma \rangle .
$$

We first treat the exact case $\tau = 0$ . Then $\mathscr { L } _ { j }$ exactly satisfies the constraints in $\mathcal { B } _ { \eta ^ { * } n } ( \boldsymbol { y } )$ . In particular,

1. $w _ { i } ^ { 2 } = w _ { i } ,$

2. $\begin{array} { r } { \sum _ { i = 1 } ^ { n } w _ { i } \geqslant ( 1 - \eta ^ { * } ) n , } \end{array}$

3. $w _ { i } ( a _ { i } ^ { \prime } - a _ { i } ) = ( 1 / m + \tau _ { \mathrm { c o v } } ) ^ { - 1 / 2 } \sigma ^ { \top } w _ { i } ( y _ { i } ^ { \prime } - y _ { i } ) = 0 ,$

4. $\begin{array} { r } { \mathcal { L } _ { j } [ n ^ { - 1 } \sum _ { i = 1 } ^ { n } a _ { i } ^ { \prime } ] = ( 1 / m + \tau _ { \mathrm { c o v } } ) ^ { - 1 / 2 } \langle \mathcal { L } _ { j } \mu ^ { \prime } - \mu _ { \mathcal { L } _ { j } } , \sigma \rangle = 0 . } \end{array}$

Choosing $\eta ^ { * }$ smaller than 0<sub>.</sub>01, Lemma $\mathrm { A . 1 }$ shows that if at least $n / 4$ of the $a _ { i } ^ { \prime } \mathbf { s }$ were at least $^ { 2 0 , }$ then

$$
\mathcal { L } _ { j } \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( a _ { i } ^ { \prime } - \frac { 1 } { n } \sum _ { \ell = 1 } ^ { n } a _ { \ell } ^ { \prime } \right) ^ { 2 } \right] \geqslant 2 .\tag{37}
$$

By definition of $a _ { i } ^ { \prime }$ and $\mu \mathcal { L } _ { j } .$ , the left-hand side is exactly

$$
\frac { 1 } { 1 / m + \tau _ { \mathrm { c o v } } } \mathcal { L } _ { j } \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \langle y _ { i } ^ { \prime } - \mu ^ { \prime } , \sigma \rangle ^ { 2 } \right] .
$$

Applying the covariance constraint from $\mathcal { B } _ { \eta ^ { * } n } ( \boldsymbol { y } )$ gives

$$
\begin{array} { r l r } {  { \frac { 1 } { 1 / m + \tau _ { \mathrm { c o v } } } \mathscr { L } _ { j } [ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \langle y _ { i } ^ { \prime } - \mu ^ { \prime } , \sigma \rangle ^ { 2 } ] \leqslant \frac { \mathscr { L } _ { j } [ \sigma ^ { \top } ( \mathrm { d i a g } ( \mu ^ { \prime } ) - \mu ^ { \prime } \mu ^ { \prime \top } ) \sigma ] + m \mathscr { L } _ { j } [ \sum _ { \ell = 1 } ^ { k } \lambda _ { \ell } ] } { 1 + m \tau _ { \mathrm { c o v } } } } } \\ & { } & { \leqslant \frac { \mathscr { L } _ { j } [ 1 - \langle \mu ^ { \prime } , \sigma \rangle ^ { 2 } ] + m \tau _ { \mathrm { c o v } } } { 1 + m \tau _ { \mathrm { c o v } } } \leqslant 1 , ~ } \end{array}
$$

since $\mu ^ { \prime }$ lies in the simplex in the exact case and $\sigma _ { \ell } ^ { 2 } = 1$ for every $\ell \in [ k ]$ . This contradicts (37). Hence fewer than $n / 4$ of the $a _ { i } ^ { \prime } \mathbf { s }$ can be at least 20. Applying this argument to −� shows that fewer than $n / 4$ of the $a _ { i } ^ { \prime } \mathbf { s }$ can be at most −20. Therefore, the median of $\{ \langle y _ { i } , \sigma \rangle \} _ { i \in [ n ] }$ lies within $2 0 \sqrt { 1 / m + \tau _ { \mathrm { c o v } } }$ of $\langle \mu _ { \mathcal { L } _ { i } } , \sigma \rangle$

By Remark A.2, the bound on the distance from $\langle \mu _ { \mathcal { L } _ { j } } , \sigma \rangle$ to the observed-data median remains valid for approximate certificates.

Both projected certificate means $\langle \mu _ { \mathcal { L } _ { 1 } } , \sigma \rangle$ and $\langle \mu _ { \mathcal { L } _ { 2 } } , \sigma \rangle$ therefore lie within $O ( \sqrt { 1 / m + \tau _ { \mathrm { c o v } } } )$ of the same median of $\{ \langle y _ { i } , \sigma \rangle \} _ { i \in [ n ] } .$ , which proves (36). By exact $\ell _ { 1 }$ duality, (36) implies

$$
\| \mu _ { \mathcal { L } _ { 1 } } - \mu _ { \mathcal { L } _ { 2 } } \| _ { 1 } = \operatorname* { m a x } _ { \sigma \in \{ \pm 1 \} ^ { k } } \langle \mu _ { \mathcal { L } _ { 1 } } - \mu _ { \mathcal { L } _ { 2 } } , \sigma \rangle \leqslant O \left( \sqrt { \frac { 1 } { m } + \tau _ { \mathrm { c o v } } } \right) .
$$

Finally, since $\nu _ { j }$ is certifiable via $\mathcal { L } _ { j } ,$ the closeness condition from Definition 4.3 gives

$$
\| \nu _ { j } - \mu _ { \angle _ { j } } \| _ { 1 } \leqslant \alpha + \tau \eta ^ { * } n = O ( \alpha ) \qquad { \mathrm { f o r ~ e a c h ~ } } j \in \{ 1 , 2 \} .
$$

Hence

$$
\lVert \nu _ { 1 } - \nu _ { 2 } \rVert _ { 1 } \leqslant O \left( \alpha + \sqrt { \frac { 1 } { m } + \tau _ { \mathrm { c o v } } } \right) .
$$

This proves Lemma 4.14.

## B Pure DP Lower Bounds

This appendix proves the pure-DP lower bound in Theorem B.5. Its three terms come from three diferent arguments: classical nonprivate estimation, a centered code packing around the uniform distribution, and a packing inside the simplex in $( k - 1 )$ coordinates.

## B.1 Projection Notation for Lower Bounds

We work in $( k - 1 )$ coordinates for the packing argument, and introduce the required projection notation here.

Definition B.1 (Nonnegative $\ell _ { 1 }$ -ball (radius 1) in $\mathbb { R } ^ { k - 1 } )$ . Let $\Delta _ { \leqslant } ^ { k - 1 }$ denote

$$
\Delta _ { \leqslant } ^ { k - 1 } : = \left\{ x \big \vert \langle \mathbf { 1 } , x \rangle \leqslant 1 , x \in \mathbb { R } _ { \geqslant 0 } ^ { k - 1 } \right\} .
$$

Definition B.2 (Restriction and Extension Operators). We define Res: $\mathbb { R } ^ { k } \to \mathbb { R } ^ { k - 1 }$ and Ext: $\mathbb { R } ^ { k - 1 } \to \mathbb { R } ^ { k }$ by

$$
\begin{array} { l } { \displaystyle \mathrm { R e s } ( v ) : = v _ { [ k - 1 ] } } \\ { \displaystyle \mathrm { E x t } ( v ) : = ( v , 1 - \sum _ { i \in [ k - 1 ] } v _ { i } ) . } \end{array}
$$

In words, Res restricts a vector to its first � − 1 coordinates, and Ext appends the final simplex coordinate.

Fact B.3 (ℓ<sub>1</sub> Distance Between Restrictions and Extensions). For any $u , v \in \mathbb { R } ^ { k - 1 }$

$$
\left. \operatorname { E x t } ( u ) - \operatorname { E x t } ( v ) \right. _ { 1 } = \left. u - v \right. _ { 1 } + \left. \sum _ { i = 1 } ^ { k - 1 } ( u _ { i } - v _ { i } ) \right. \in \left[ \Vert u - v \Vert _ { 1 } , 2 \Vert u - v \Vert _ { 1 } \right] .
$$

Moreover, for any $x , y \in \mathbb { R } ^ { k }$ with $\textstyle \sum _ { i = 1 } ^ { k } x _ { i } = \sum _ { i = 1 } ^ { k } y _ { i }$ (for example �<sub>,</sub> $y \in \Delta ^ { k - 1 } )$

$$
\left\| \operatorname { R e s } ( x ) - \operatorname { R e s } ( y ) \right\| _ { 1 } \leqslant \ \left\| x - y \right\| _ { 1 } \leqslant \ 2 \left\| \operatorname { R e s } ( x ) - \operatorname { R e s } ( y ) \right\| _ { 1 } .
$$

Proof. For the extension inequalities, we can write

$$
\operatorname { E x t } ( u ) - \operatorname { E x t } ( v ) = { \big ( } u - v , \ 1 - \sum _ { i = 1 } ^ { k - 1 } u _ { i } - ( 1 - \sum _ { i = 1 } ^ { k - 1 } v _ { i } ) { \big ) } = { \big ( } u - v , \ - \sum _ { i = 1 } ^ { k - 1 } ( u _ { i } - v _ { i } ) { \big ) } ,
$$

so

$$
\| \operatorname { E x t } ( u ) - \operatorname { E x t } ( v ) \| _ { 1 } = \sum _ { i = 1 } ^ { k - 1 } | u _ { i } - v _ { i } | + \left| \sum _ { i = 1 } ^ { k - 1 } ( u _ { i } - v _ { i } ) \right| \in \big [ \| u - v \| _ { 1 } , 2 \| u - v \| _ { 1 } \big ]
$$

by the triangle inequality. For the restriction inequalities, the lower bound is immediate since restriction can only decrease $\ell _ { 1 }$ norm. If $\textstyle \sum _ { i } x _ { i } = \sum _ { i } y _ { i } .$ , then $\begin{array} { r } { x _ { k } - y _ { k } = - \sum _ { i = 1 } ^ { k - 1 } ( x _ { i } - y _ { i } ) } \end{array}$ and hence

$$
\| x - y \| _ { 1 } = \sum _ { i = 1 } ^ { k - 1 } | x _ { i } - y _ { i } | + \Big | \sum _ { i = 1 } ^ { k - 1 } ( x _ { i } - y _ { i } ) \Big | \leqslant 2 \| \operatorname { R e s } ( x ) - \operatorname { R e s } ( y ) \| _ { 1 } .
$$

We invoke the following result.

Theorem B.4 (Packing Lower Bounds [HKM22]). Let $T \geqslant 2$ and $0 < \beta \leqslant 1 / 3$ . Let $\mathcal { P } = \{ P _ { 1 } , \ldots , P _ { T } \}$ be a set of distributions, and $P _ { 0 }$ be a distribution such that for every $P _ { i } \in \mathcal { P } , \| P _ { i } - P _ { 0 } \| _ { \mathrm { T V } } \leqslant \gamma$ . Let $\mathcal { G } = \{ G _ { 1 } , \ldots , G _ { T } \}$ be a collection of disjoint subsets of some set �. If there is an �-DP algorithm � such that $\mathbb { P } _ { X \sim P _ { i } ^ { n } } [ M ( X ) \in G _ { i } ] \geqslant 1 - \beta f o r a l l i \in [ T ]$ , then

$$
n \geqslant \Omega { \Bigg ( } { \frac { \log T + \log ( 1 / \beta ) } { \gamma ( e ^ { 2 \varepsilon } - 1 ) } } { \Bigg ) }
$$

Note that for the usual regime $\varepsilon \leqslant 1$ , we can replace the $e ^ { 2 \varepsilon } - 1$ in the denominator with $\varepsilon .$

We use Theorem B.4 to show the following lower bound under pure DP. Similar lower bounds already exist in $[ \mathrm { L S Y ^ { + } 2 0 } ]$ . Here we focus on the dependence on the failure probability $\beta .$

Theorem B.5. Assume $k \geqslant 2 , 0 < \alpha \leqslant 1 / 8 , 0 < \beta \leqslant 1 / 3$ , and $0 < \varepsilon \leqslant 1$ . Any ${ \varepsilon } { - } D P$ algorithm that, for every distribution $\mu \in \Delta ^ { k - 1 }$ , takes � independent batches of � i.i.d. samples from � and outputs an estimate $\hat { \mu }$ satisfying $\| \hat { \mu } - \mu \| _ { 1 } \leqslant \alpha$ with probability at least $1 - \beta$ must use

$$
n = \Omega \Bigg ( \frac { k + \log ( 1 / \beta ) } { \alpha ^ { 2 } m } + \frac { k + \log ( 1 / \beta ) } { \alpha \sqrt { m } \varepsilon } + \frac { k \log ( 1 / \alpha ) + \log ( 1 / \beta ) } { \varepsilon } \Bigg ) .
$$

Proof. First term: nonprivate estimation. Classical nonprivate estimation of a distribution on [�] to $\ell _ { 1 }$ error � with failure probability $\beta$ requires $\Omega ( ( k + \bar { \log } ( 1 / \beta ) ) / \alpha ^ { 2 } )$ individual samples [Can20]. Since there are �� samples, division by � gives the first term.

Third term: packing inside the simplex. From Fact B.3 we have

$$
\| \mathrm { R e s } ( \mu ) - \mathrm { R e s } ( \mu ^ { \prime } ) \| _ { 1 } \leqslant \| \mu - \mu ^ { \prime } \| _ { 1 } \leqslant 2 \| \mathrm { R e s } ( \mu ) - \mathrm { R e s } ( \mu ^ { \prime } ) \| _ { 1 } .
$$

If we find some points $v _ { 1 } , \dotsc , v _ { T } \in \Delta _ { \leqslant } ^ { k - 1 }$ (as in Definition B.1) such that $\| v _ { i } - v _ { j } \| _ { 1 } \geqslant 3 \alpha$ for all $i \neq j$ and let $\mu _ { i } = \operatorname { E x t } ( v _ { i } )$ , then define

$$
P _ { i } : = \mu _ { i } ^ { \otimes m } \qquad \mathrm { f o r } \ i \in [ T ] , \qquad P _ { 0 } : = \mu _ { 1 } ^ { \otimes m } .
$$

Since total variation distance between any two probability distributions is at most 1, we have $\| P _ { i } - P _ { 0 } \| _ { \mathrm { T V } } \leqslant 1$ for every $i \in [ T ]$ . Thus Theorem B.4 applies with $\gamma = 1$ to the family $\{ P _ { i } \} _ { i \in [ T ] }$ and the sets

$$
G _ { i } = \left\{ \hat { \mu } \ : \Big | \ : \| \hat { \mu } - \mu _ { i } \| _ { 1 } \leqslant \alpha \right\} .
$$

To construct the 3�-separated points $v _ { 1 } , \ldots , v _ { T } .$ , consider all mirror images of $\Delta _ { \leqslant } ^ { k - 1 }$ , namely $C _ { s , k } = \{ x \ | \ \| x \| _ { 1 } \leqslant 1 , s _ { i } x _ { i } \geqslant 0 \}$ for $s \in \{ \pm 1 \} ^ { k - 1 }$ . These sets partition the $\ell _ { 1 }$ ball of radius 1 in $\mathbb { R } ^ { k - 1 }$ . If there exist $T ^ { \prime }$ points in this ball at pairwise distance at least $3 \alpha$ , then reflecting its most populated orthant into the nonnegative orthant gives at least $T = 2 ^ { - k + 1 } T ^ { \prime }$ points in $\Delta _ { \leqslant } ^ { k - 1 }$ at the same separation. Standard $\ell _ { 1 }$ packing bounds (see e.g. [Wai19]) give $T ^ { \prime } \geqslant ( 1 / ( 3 \alpha ) ) ^ { k - 1 }$ , hence $T \geqslant ( 1 / ( 6 \alpha ) ) ^ { \bar { k } - 1 }$ , and since $\alpha \leqslant 1 / 8 ,$ , Theorem B.4 yields the lower bound

$$
n \geqslant \Omega \left( \frac { k \log ( 1 / \alpha ) + \log ( 1 / \beta ) } { \varepsilon } \right) .
$$

Second term: centered code packing around the uniform distribution. Let $\mu _ { o }$ be the uniform distribution on the first $2 \lfloor k / 2 \rfloor$ elements of [�]. For each $x \in \{ \pm 1 \} ^ { \lfloor k / 2 \rfloor }$ , define $s ( x ) \in \{ - 1 , 0 , + 1 \} ^ { k }$ by

$$
s ( x ) _ { 2 u - 1 } = x _ { u } , \qquad s ( x ) _ { 2 u } = - x _ { u } \qquad \mathrm { f o r ~ } u \in [ \lfloor k / 2 \rfloor ] ,
$$

and if � is odd set the last coordinate of $s ( x )$ to zero. Define

$$
\mu ( x ) : = \mu _ { o } + \frac { 4 \alpha } { 2 \lfloor k / 2 \rfloor } s ( x ) .
$$

Since $\begin{array} { r } { \sum _ { i = 1 } ^ { k } s ( x ) _ { i } = 0 } \end{array}$ and $4 \alpha \leqslant 1 / 2$ under $\alpha \leqslant 1 / 8$ , we have $\mu ( x ) \in \Delta ^ { k - 1 }$ for every �. Now note that for any such � we have

$$
\begin{array} { l } { { \displaystyle \mathrm { K L } ( \mu ( x ) \| \mu _ { o } ) = \sum _ { i = 1 } ^ { 2 \lfloor k / 2 \rfloor } \mu ( x ) ( i ) \log \biggl ( \frac { \mu ( x ) ( i ) } { \mu _ { o } ( i ) } \biggr ) } } \\ { { = \displaystyle \frac { 1 } { 2 \lfloor k / 2 \rfloor } \sum _ { i = 1 } ^ { 2 \lfloor k / 2 \rfloor } \left( 1 + 4 \alpha s ( x ) _ { i } \right) \log ( 1 + 4 \alpha s ( x ) _ { i } ) , } } \end{array}
$$

For $\vert t \vert \leqslant 1 / 2$ , the inequality $( 1 + t ) \log ( 1 + t ) \leqslant t + 2 t ^ { 2 }$ holds. Applying it with $t = 4 \alpha s ( x ) _ { i }$ � and using $\begin{array} { r } { \sum _ { i } s ( { \boldsymbol x } ) _ { i } = 0 } \end{array}$ , we obtain

$$
\mathrm { K L } ( \mu ( x ) \| \mu _ { o } ) \leqslant \frac { 3 2 \alpha ^ { 2 } } { 2 \lfloor k / 2 \rfloor } \sum _ { i = 1 } ^ { 2 \lfloor k / 2 \rfloor } s ( x ) _ { i } ^ { 2 } \leqslant 3 2 \alpha ^ { 2 } = O ( \alpha ^ { 2 } ) .
$$

By tensorization of KL and Pinsker’s inequality,

$$
\| \mu ( x ) ^ { \otimes m } - \mu _ { o } ^ { \otimes m } \| _ { \mathrm { T V } } \leqslant O ( \alpha \sqrt { m } ) .
$$

A greedy binary-code packing gives $x _ { 1 } , \ldots , x _ { M } \in \{ \pm 1 \} ^ { \lfloor k / 2 \rfloor }$ with $M = 2 ^ { \Omega ( \lfloor k / 2 \rfloor ) } = 2 ^ { \Omega ( k ) }$ and pairwise Hamming distance strictly greater than $\lfloor k / 2 \rfloor / 4 $ : each chosen word removes at most

$$
\sum _ { j = 0 } ^ { \lfloor \lfloor k / 2 \rfloor / 4 \rfloor } { \binom { \lfloor k / 2 \rfloor } { j } } \leqslant 2 ^ { H _ { 2 } ( 1 / 4 ) \lfloor k / 2 \rfloor }
$$

words, where $H _ { 2 } ( 1 / 4 ) < 1$ is the binary entropy. For $i \neq j ,$

$$
\| \mu ( x _ { i } ) - \mu ( x _ { j } ) \| _ { 1 } = \frac { 8 \alpha } { \lfloor k / 2 \rfloor } d _ { H } ( x _ { i } , x _ { j } ) > \frac { 4 \alpha } { 2 } = 2 \alpha .
$$

$$
n \geqslant \Omega \left( \frac { k + \log ( 1 / \beta ) } { \alpha \sqrt { m } \varepsilon } \right) .
$$

Thus the closed $\ell _ { 1 }$ balls of radius � about the code distributions are disjoint. Therefore, applying Theorem B.4 with $\gamma = \operatorname* { m i n } \{ 1 , O ( \alpha \sqrt { m } ) \}$ gives

□