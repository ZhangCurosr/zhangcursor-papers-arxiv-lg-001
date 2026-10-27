# Balanced Sampling in Stochastic Multi-objective Bandits

Mengfan Xu<sup>∗</sup>   
Mechanical and Industrial Engineering   
University of Massachusetts Amherst mengfanxu@umass.edu

October 26, 2026

## Abstract

In stochastic multi-objective bandits, Pareto optimal arms may represent diferent trade-ofs among objectives. Balanced sampling across them is important in practical settings such as online advertising. Yet, there seems to be little consensus on how to measure fairness, to our best knowledge. Achieving fairness is also tough as the set of Pareto optimal arms is unknown, and early sampling decisions may create imbalances that are dificult to erase afterwards. We measure unfairness by the KL divergence from the uniform distribution over the true Pareto set to the empirical measure induced by the sampling trajectory. Nicely, this measure admits an exact decomposition that separates the cost of sampling suboptimal arms from the imbalance among Pareto optimal arms. With a recursive form for unfairness, we develop adaptive top-two sampling, which combines sampling counts and empirical evidence of Pareto membership to define a gain for each arm and guide sampling without committing to an estimated Pareto set. When the Pareto set is stable under small perturbations, we establish an $O ( \log T / T )$ and $O ( T ^ { - 1 } )$ bound on expected unfairness and variance, respectively. Information-theoretic lower bounds show that the log T /T rate can be unavoidable for uniformly fair policies on stable instances, matching our upper bound. More surprisingly, stability is indeed crucial, as the expected unfairness can be $\omega ( \log T / T )$ without it. Finally, we create a variant that encourages exploration of undersampled arms to reduce the risk of persistent underallocation. At suficiently large horizons, it preserves the rate on the expected unfairness while admitting a sharper bound of ${ \bar { O ( \log ^ { 2 } T / T ^ { 2 } ) } }$ on variance. Numerical studies support these findings and suggest practical insights.

## 1 Introduction

Multi-armed bandits have supported sequential decision making widely, where a decision maker repeatedly selects an option (arm) and receives information about its performance in the form of a reward [Auer et al., 2002a,b, Bubeck and Cesa-Bianchi, 2012]. Classical bandits typically consider a scalar reward and measure performance by regret, which captures the cumulative cost of selecting suboptimal arms. However, many decisions involve multiple objectives beyond a single scalar reward. Multi-objective bandits (MO-MAB) have emerged to address such problems [Drugan and Nowe, 2013, Busa-Fekete et al., 2017, Mehrotra et al., 2020, Lu et al., 2019]. Here, each arm exhibits a vector of rewards, and Pareto dominance provides a new notion of optimality. An arm is Pareto optimal if no other arm performs at least as well in every objective and strictly better in at least one.

Importantly, multiple arms may be Pareto optimal, and the set of such arms is called the Pareto set by convention. In fact, diferent Pareto optimal arms may represent diferent trade-ofs among objectives. For example, in online recommendation, diferent choices may favor user engagement or supplier diversity, while in network routing, diferent routes may ofer trade-ofs between latency and reliability [Mehrotra et al., 2020, Lu et al., 2019]. Repeatedly selecting only a subset of Pareto optimal arms may favor particular trade-ofs, thereby incurring opportunity costs for others. A balanced selection across Pareto optimal arms helps to balance these diferent forms of trade-ofs in real-world scenarios, and thus demands our attention. In fact, in classical bandits, fairness [Wang et al., 2021, Singh and Joachims, 2018, Simchi-Levi and Wang, 2023] has been well motivated and understood. Yet we can imagine that these insights may not translate into MO-MAB given Pareto dominance.

A substantial line of research on MO-MAB has studied how to learn and select Pareto optimal arms. Amongst, two most common goals are 1) to minimize Pareto regret, which measures the cumulative cost of selecting Pareto suboptimal arms [Drugan and Nowe, 2013, Turgay et al., 2018, Lu et al., 2019, Guan and Xu, 2026], and 2) to identify the Pareto set through careful sampling with as small sample complexity as possible [Auer et al., 2016, Ararat and Tekin, 2023, Garivier et al., 2024, Kone et al., 2024]. Yet, neither guarantees a balanced (or fair) allocation among these Pareto optimal arms. To see this, consider two arms with mean reward vectors (1, 0) and (0, 1). Both arms are Pareto optimal, and selecting either incurs zero Pareto regret. In fact, a learner may select the first arm almost exclusively and still achieve zero regret, although the second arm receives almost no observations. Pareto set identification, on the other hand, concerns which arms belong to the set and stops once we figure out both arms are Pareto optimal. By doing so, we emphasize their tota sampling frequencies that sum over arms and time, but not their relative sampling frequencies. The latter indeed matters a lot when we care about the sampling trajectory across arms and time steps.

In parallel, fairness in MO-MAB has received relatively less attention, to the extent of our knowledge. Existing work has approached it, but there has been a lack of consensus on which definition we should adopt in general. The first debate concerns fairness across objectives [Ann and Oh, 2026] or among arms [Yahyaa et al., 2014a, Mehrotra et al., 2020]. The latter is more appropriate when we want to balance the trade-ofs among objectives entangled at the arm level via future sampling, whereas the former is more about the disentangled trade-ofs among objectives when looking backward. Herein, we focus on the latter, in line with classical bandits and our goal in an online setting. Within this layer, multiple measures still exist, mostly based on information entropy. On the positive side, this means that researchers have already considered this problem; on the flip side, it may imply that we have not yet converged on a coherent definition, as we can still find cases where the existing measures are less appropriate. For example, Yahyaa et al. [2014a] use Shannon entropy [Sethna et al., 2006] to measure unfairness, which increases as the allocation among the Pareto set becomes more uniform. Yet, uniform allocation can actually be the fairest. A consequence of the lack of consensus is that there have been almost no theoretical analyses of fairness in MO-MAB, to the best of our knowledge.

Operationally, achieving balanced allocation is challenging because the Pareto set is unknown, while fairness requires controlling the sampling process over time. If the Pareto set were known, we could simply favor its least-sampled arms. In practice, however, we must infer Pareto membership from noisy observations while deciding how to allocate samples. A natural approach is to estimate the Pareto set and balance sampling within it. Yet, an incorrect estimate may have substantial consequences. A Pareto optimal arm mistakenly classified as dominated may receive few further observations, delaying the correction of its estimate and causing persistent underallocation. Such selections would increase the cumulative cost as well. Even if the Pareto set is eventually identified correctly, the learning process may have already consumed a substantial portion of the horizon. We may not have enough time to compensate for the underallocation of newly discovered Pareto optimal arms. This dificulty is particularly pronounced near a boundary of Pareto dominance, where small estimation errors can flip Pareto membership entirely. Thus, learning Pareto membership and balancing allocation necessitate a joint treatment, as each selection afects both. We hope to find out:

Can we find a satisfying measure of fairness and achieve fair allocation over an unknown Pareto set?

Main contributions. We answer this question firmly. We first measure terminal unfairness by the KL divergence $\mathcal { F } _ { T } = D _ { \mathrm { K L } } ( \pi ^ { \star } \| \widehat { \pi } _ { T } )$ , where $\pi ^ { \star }$ is the uniform distribution over the true Pareto set and $\widehat { \pi } _ { T }$ is the empirical allocation. Intuitively, the closer the empirical allocation is to a uniform distribution across Pareto-optimal arms, the greater the fairness. Interestingly, this measure admits an exact decomposition:

$$
\mathcal { F } _ { T } = - \log ( 1 - B _ { T } / T ) + D _ { \mathrm { K L } } ( U _ { m } \| q _ { T } ) ,\tag{1}
$$

where $B _ { T }$ is the number of dominated-arm pulls, m is the Pareto set size, and $q _ { T }$ is the empirical allocation conditional on Pareto-optimal pulls. The two terms capture the cost of sampling dominated arms and the imbalance among Pareto-optimal arms, respectively. This provides a natural connection between fairness and Pareto regret while revealing the additional requirement of balanced allocation. After careful comparison with existing measures, we find that this measure is coherent and appropriate while preserving the information-theoretic structure.

The one-step change in unfairness shows that the benefit of sampling an arm depends on both its sampling count and Pareto membership. To this end, we develop adaptive top-two sampling, which defines a step-wise fairness gain based on the sampling count and an exponential weight reflecting empirical evidence of Pareto membership. It selects between the two arms with the largest gains, promoting exploration without committing to an estimated Pareto set. When the Pareto set is stable under small changes in mean rewards, we show that unfairness has mean $O _ { \mu } ( \log T / T )$ and variance $O _ { \mu } ( T ^ { - 1 } )$ . Moreover, as a by-product, the expected Pareto regret grows at most logarithmically. With information-theoretic lower bounds, we show that the log $T / T$ rate can be unavoidable for uniformly fair policies on instances where the Pareto set is stable. More surprisingly, at a boundary where the Pareto set is unstable, we prove a cost of $\omega ( \log T / T )$ for any uniformly fair policy, which also justifies the assumption of stability. These results speak to the power of Pareto geometry.

Finally, a variance of order $O _ { \mu } ( T ^ { - 1 } )$ could require $\Omega ( T )$ independent runs to estimate expected fairness with suficient precision. To improve reliability across trajectories, we introduce a method that encourages more exploration of undersampled arms and thus reduces mistakes about Pareto membership that can lead to catastrophic unfairness. When the horizon is suficiently large, it preserves the $O _ { \mu } ( \log T / T )$ rate for the expected unfairness while admitting a sharper upper bound of $O _ { \mu } ( \log ^ { 2 } T / T ^ { 2 } )$ on variance. At smaller horizons, however, it may increase pulls of Pareto suboptimal arms, suggesting a subtle trade-of whose consequences depend on the stability of the Pareto set.

## 2 Problem Formulation

We consider a stochastic multi-objective bandit problem, which itself has been by large studied in recent years, but from a new perspective of balanced allocation across good arms that first demands a new framework for measuring ”being balanced.” This section is thus devoted to such conceptualization. Let us consider $K \geq 2$ arms and $d \geq 1$ reward coordinates for each arm $i \in [ K ]$ Each arm is associated with an independent reward tape $X _ { i } = ( X _ { i } ^ { ( 1 ) } , \ldots , X _ { i } ^ { ( d ) } )$ which is independent across arms and time steps and identically distributed across time; however, diferent coordinates of a reward vector of the same arm may be dependent. The reward vector of arm i has a unknown mean vector $\mu _ { i } = ( \mu _ { i } ^ { ( 1 ) } , \ldots , \mu _ { i } ^ { ( d ) } ) \in \mathcal { M }$ , where $\begin{array} { r } { \mathcal { M } = \prod _ { r = 1 } ^ { d } [ a _ { r } , b _ { r } ] } \end{array}$ is a Cartesian product of the coordinate ranges. Without loss of generality, we assume $a _ { r } , b _ { r }$ are known; typically $a _ { r } = 0 , b _ { r } = 1$ in the literature. Every reward coordinate $X _ { i } ^ { ( r ) }$ follows a $\sigma ^ { 2 }$ -sub-Gaussian distribution with known variance proxy $\sigma ^ { 2 }$ , formally

$$
\mathbb { E } \exp \bigl ( \lambda ( X _ { i } ^ { ( r ) } - \mu _ { i } ^ { ( r ) } ) \bigr ) \leq \exp ( \sigma ^ { 2 } \lambda ^ { 2 } / 2 ) , \qquad \lambda \in \mathbb { R } .\tag{2}
$$

Let us denote the time horizon of the bandit problem by T. At time $t \in [ T ]$ , a player, equipped with a (possibly random) policy, chooses arm $A _ { t }$ based on what it has observed thus far without any other information beyond the history. It then receives a reward vector $X _ { A _ { t } , t }$ which is one random realization of $X _ { A _ { t } }$ of the selected arm $A _ { t }$ at time t.

The goal of the player is to select the ”good” arms in a ”balanced” way, and let us formalize these terminologies one by one. We measure ”good” through the celebrated Pareto dominance [Drugan and Nowe, 2013, Mehrotra et al., 2020]. An arm j dominates arm i when $\mu _ { j } ^ { ( r ) } \geq \mu _ { i } ^ { ( r ) }$ for every $r ,$ with at least one strict inequality. Let $\mathcal { A } ^ { \star } = \mathcal { A } ^ { \star } ( \mu )$ be the nondominated arm set, and thus represent the set of ”good” arms, which is oftentimes called Pareto optimal set; let us also denote its cardinality by $m = | \mathcal { A } ^ { \star } |$

A learning policy $\pi = ( \pi _ { t } ) _ { t \geq 1 }$ specifies a probability distribution $\pi _ { t } ( \cdot \ | \ H _ { t - 1 } )$ on the measurable space of arms $( [ K ] , 2 ^ { [ K ] } )$ at each time t, conditional on the observed history $H _ { t - 1 } = ( A _ { s } , X _ { A _ { s } , s } ) _ { s = 1 } ^ { t - 1 }$ Jointly, the policy and reward induce a probability measure $\mathbb { P } _ { X } ^ { \pi }$ over the sequence of actions and rewards.

Let $\begin{array} { r } { N _ { i } ( t ) = \sum _ { s = 1 } ^ { t } \mathbf { 1 } \{ A _ { s } = i \} } \end{array}$ denote the total number of pulls of arm i up to and including time t. The empirical frequency of arm i is then $\widehat { \pi } _ { t , i } = N _ { i } ( t ) / t$ . From a probabilistic perspective, these frequencies define an empirical probability measure $\widehat { \pi } _ { t }$ on $( [ K ] , 2 ^ { [ K ] } )$ , with $\widehat { \pi } _ { t } ( \{ i \} ) = \widehat { \pi } _ { t , i }$ . This is a random probability measure constructed from the action sequence $( A _ { 1 } , \ldots , A _ { t } )$ , whose random variables are generally dependent and non-identically distributed, with randomness governed by $\mathbb { P } _ { X } ^ { \pi }$

Subsequently, we are clear that the target of the player is the ideal measure defined through $\pi _ { i } ^ { \star } = m ^ { - 1 } \mathbf { 1 } \{ i \in \mathcal { A } ^ { \star } \}$ , which assigns equal probabilities to all Pareto optimal arms and 0 probabilities to suboptimal arms, though of course $\mathcal { A } ^ { \star }$ is unknown to us as the mean vectors are unknown. This language in probability motivates and also prepares us to measure terminal unfairness by the KL divergence between the ideal measure and the empirical measure

$$
\mathcal { F } _ { T } = D _ { \mathrm { K L } } ( \pi ^ { \star } \| \widehat { \pi } _ { T } ) = \frac { 1 } { m } \sum _ { i \in \mathcal { A } ^ { \star } } \log \frac { T } { m N _ { i } ( T ) } .\tag{3}
$$

Naturally, we can also consider the unfairness of the expected empirical measure where E is over the randomness of the allocation policy itself,

$$
\bar { \mathcal { F } } _ { T } = D _ { \mathrm { K L } } ( { \boldsymbol \pi } ^ { \star } \| \mathbb { E } \widehat { \boldsymbol { \pi } } _ { T } ) = \frac { 1 } { m } \sum _ { i \in \mathcal { A } ^ { \star } } \log \frac { T } { m \mathbb { E } [ N _ { i } ( T ) ] } .
$$

By the Jensen’s inequality, we have that $E [ \mathcal { F } _ { T } ] \ge \bar { \mathcal { F } } _ { T }$

We would like to compare our measure with existing measures of fairness across Pareto optimal arms. These difer from measures of fairness across objectives, so for the sake of space and relevance, we omit that discussion here and refer readers to [Busa-Fekete et al., 2017, Mehrotra et al., 2020, Ann and Oh, 2026] for a detailed introduction. It is surprising that our fairness also presents as an upper bound for the objective fairness in [Ann and Oh, 2026], and thus provides a natural bridge between objective fairness and fairness over Pareto optimal arms herein. Note that an existing measure of unfairness measures the KL divergence between the optimal sampling policy and the actual sampling policy [Yahyaa et al., 2014a]. However, it does not restrict arms to be Pareto optimal and is ill-defined if we consider only Pareto optimal arms. The KL measure coincides with our measure when the target distribution is uniform over all Pareto optimal arms and all arms are Pareto optimal. Another existing measure is based on Shannon entropy [Yahyaa et al., 2014a], namely, the entropy of the actual sampling frequencies restricted to Pareto optimal arms, which is essentially the empirical measure of the conditional distribution of the sampling policy. Higher entropy represents greater unfairness under their definition. This is fundamentally flawed, however, because it is well known that the uniform distribution has maximum entropy among all discrete distributions and, based on their definition, represents the most unfair allocation. At the same time, the uniform distribution is the fairest allocation when there is no preference over the priority of the Pareto optimal arms. This contradiction suggests that it may be less appropriate to use this measure. Instead, we directly compare the empirical measure with the fairest allocation, that is, the uniform distribution, and a smaller distance indicates a fairer allocation. In the meantime, our measure still admits an entropy formulation through the relationship between KL divergence and Shannon entropy, which we believe is satisfying.

More broadly, while we use the KL divergence distance, many other distance measures from probability theory can be adopted. Examples include the Kolmogorov, Wasserstein, and total variation distances. Our result provides a conservative guarantee, as the Kolmogorov distance is bounded by total variation, and the Wasserstein distance is bounded by a constant multiple of total variation on a finite space with a bounded ground metric [Ross, 2011]. Moreover, total variation is bounded by the square root of half the KL divergence through Pinsker’s inequality. Thus, by providing an upper bound on the KL divergence that defines balancedness, we also obtain upper bounds on all these distance measures in some sense.

Beyond its role as a proper measure of balancedness, surprisingly, it can also connect well to existing metrics of eficiency of the selected arms in multi-objective bandits. Taking a closer look at this metric, we find that it admits a useful exact decomposition. To see this, let $\begin{array} { r } { B _ { T } = \sum _ { i \notin \mathcal { A } ^ { \star } } N _ { i } ( T ) } \end{array}$ denote the total number of pulls of dominated arms, which we refer to the less eficient pulls, and let $H _ { T } = T - B _ { T }$ be the total number of pulls of Pareto optimal arms. For each $i \in \mathcal { A } ^ { \star }$ , let us define $q _ { T , i } = N _ { i } ( T ) / H _ { T }$ as its sampling proportion conditional on pulling a Pareto optimal arm. Finally, let $U _ { m }$ denote the uniform distribution on m points.

Throughout, we assume that $N _ { i } ( T ) \geq 2$ for all arm $i \in [ K ]$ and $T \geq 2 K$ . This is concerned about the initialization, where the policy pulls each arm at least twice; we note that it is quite common to pull arms at least once, and this twice condition, though requiring a bit more, is still mild.

Lemma 2.1. For every trajectory with positive counts on $A ^ { \star }$ , which automatically holds by the condition of $N _ { i } ( T ) \geq 2$ for every $i ,$ we have

$$
\mathcal { F } _ { T } = - \log ( 1 - B _ { T } / T ) + D _ { \mathrm { K L } } ( U _ { m } \| q _ { T } ) , D _ { \mathrm { K L } } ( U _ { m } \| q _ { T } ) \le \frac { 1 } { H _ { T } } \sum _ { i \in A ^ { \star } } \frac { ( N _ { i } ( T ) - H _ { T } / m ) ^ { 2 } } { N _ { i } ( T ) }\tag{4}
$$

In particular, $\mathcal { F } _ { T } \ge B _ { T } / T$ and $0 \leq \mathcal { F } _ { T } \leq \log ( T / m )$

Proof. Let us first substitute $N _ { i } ( T ) / T = ( H _ { T } / T ) q _ { T , i }$ into equation 3. By the relationship $D _ { \mathrm { K L } } ( P \Vert Q ) \leq$ $\chi ^ { 2 } ( P \| Q )$ , with $P = U _ { m }$ and $Q \ = \ q _ { T }$ , we obtain that $\begin{array} { r } { \chi ^ { 2 } ( U _ { m } \| q _ { T } ) = \sum _ { i } ( 1 / m - q _ { T , i } ) ^ { 2 } / q _ { T , i } = } \end{array}$ $\begin{array} { r } { H _ { T } ^ { - 1 } \sum _ { i } ( N _ { i } ( T ) - H _ { T } / m ) ^ { 2 } / N _ { i } ( T ) } \end{array}$ . Lastly, the bound $\mathcal { F } _ { T } \leq \log ( T / m )$ follows from $N _ { i } ( T ) \geq 2$ . It is worth noting that both terms in equation 4 are nonnegative, and $- \log ( 1 - z ) \geq z$ which implies that $\mathcal { F } _ { T } \ge B _ { T } / T$ □

The first term is closely related to the concept of Pareto regret [Drugan and Nowe, 2013, Xu and Klabjan, 2023], which penalizes pulls of dominated arms through a weighted sum of $N _ { i } ( T )$ , where the weight is determined by arm i’s distance to the Pareto front. In contrast, this new metric penalizes $B _ { T }$ through a logarithmic transformation. But interestingly, $- \log ( 1 - B _ { T } / T ) \approx B _ { T } / T$ when $B _ { T } / T$ is small, so it resembles a normalized version of Pareto regret when all weights are equal. This decomposition is very useful in that it shows that this target not only requires the allocation to be as close to uniform across Pareto optimal arms as possible, but also necessitates allocating as much as possible to the Pareto optimal set of arms. Hence, optimizing this target gives the best of both worlds, in terms of both eficiency and balance, in a rough sense; later, we provide precise evidence for this.

## 3 Methodology

Toward this goal, we hope to design a new learning method, which is the focus of this section. Our starting point is the observation that the one-step change in unfairness satisfies

$$
\mathcal { F } _ { t } - \mathcal { F } _ { t - 1 } = \log \frac { t } { t - 1 } - \frac { \mathbb { I } \{ A _ { t } \in \mathcal { A } ^ { \star } \} } { m } \log \left( 1 + \frac { 1 } { N _ { A _ { t } } ( t - 1 ) } \right) .
$$

If the Pareto set were known, minimizing the one-step fairness loss would amount to maximizing

$$
\frac { \mathbb { I } \{ i \in \mathcal { A } ^ { \star } \} } { m } \log \left( 1 + \frac { 1 } { N _ { i } ( t ) } \right) .
$$

The arm selection then becomes quite straightforward. Interestingly, this greedy selection is closely related to the Frank–Wolfe method [Jaggi, 2013, Lacoste-Julien and Jaggi, 2015] for constrained optimization. To see this, we are essentially minimizing the KL divergence $\mathcal { F } ( \pi ) = D _ { \mathrm { K L } } ( \pi ^ { \star } \| \pi )$ over the probability simplex on $( [ K ] , 2 ^ { [ K ] } )$ where $\pi$ has to be an empirical measure. At each step, Frank–Wolfe selects a vertex of the simplex that minimizes the first-order approximation of the objective. Concretely, a vertex $e _ { i }$ corresponds to an arm $i ,$ and the partial derivative is exactly

$$
\frac { \partial \mathcal { F } ( \pi ) } { \partial \pi _ { i } } = - \frac { \mathbb { I } \{ i \in \mathcal { A } ^ { \star } \} } { m \pi _ { i } } .
$$

Thus, it amounts to selecting the Pareto optimal arm with the smallest $\widehat { \pi } _ { t , i } = N _ { i } ( t ) / t$ . This coincides with our rule by noting that $\textstyle { \frac { \mathbb { I } \{ i \in A ^ { \star } \} } { m } } \log \left( 1 + { \frac { 1 } { N _ { i } ( t ) } } \right) = { \frac { 1 } { m } } \log ( 1 + 1 / N _ { i } ( t ) )$ for such arms. Moreover, pulling arm i updates the empirical allocation as

$$
\widehat { \pi } _ { t + 1 } = \widehat { \pi } _ { t } + \frac { 1 } { t + 1 } ( e _ { i } - \widehat { \pi } _ { t } ) ,
$$

which is precisely a Frank–Wolfe update with step size $1 / ( t + 1 )$ . It is inspiring to see this connection although we use the exact change whereas Frank–Wolfe uses the first-order approximation.

Unfortunately, Pareto membership is unknown, which is why this problem is hard and why we believe this work is worth developing. A direct idea is to estimate the Pareto front $\mathcal { A } ^ { \star }$ , and replace the quantity $\mathcal { A } ^ { \star }$ in the above loss with this estimation. However, as we shall foresee, this estimation can be notoriously hard, and a wrong estimation of the Pareto membership of just one arm could flip the binary variable $I \{ i \in { \mathcal { A } } ^ { \star } \}$ entirely and thus directly propagates to the loss estimation and further the decision making. Each observation afects not only the empirical allocation but also the learner’s knowledge of the target distribution, resembling a connection to the classical knowledge-gradient principle and the value of information. To this end, we hope to construct a soft target that incorporates both the empirical dominance margin and the statistical evidence for Pareto membership. The soft target replaces this discrete weight $I \{ i \in { \mathcal { A } } ^ { \star } \}$ by a continuous score using dominance and statistical evidence of arms accumulated thus far, and we use the resulting fairness gains to guide arm sampling.

More specifically, at time t and for arm i, let $\widehat { \mu } _ { t , i }$ be the empirical mean, and $\widetilde { \mu } _ { t , i } = \Pi _ { \mathcal { M } } ( \widehat { \mu } _ { t , i } )$ its coordinate-wise projection onto M. We first measure how strongly arm i appears to be dominated through the empirical dominance margin

$$
\widehat { \Delta } _ { t , i } = [ \operatorname* { m a x } _ { j \neq i } \operatorname* { m i n } _ { r \in [ d ] } ( \widetilde { \mu } _ { t , j } ^ { ( r ) } - \widetilde { \mu } _ { t , i } ^ { ( r ) } ) ] _ { + } , \quad R _ { t , i } = \sqrt { N _ { i } ( t ) } / \sigma \widehat { \Delta } _ { t , i } .\tag{5}
$$

A positive $\widehat { \Delta } _ { t , i }$ indicates that another arm has a larger empirical mean in every objective and thus arm i is empirically dominated; we would prefer arms with small $\widehat { \Delta } _ { t , i }$ . The quantity $R _ { t , i }$ further accounts for the number of observations collected from arm i in an efort to balance exploration (through $N _ { i } ( t ) )$ and exploitation (through $\widehat { \Delta } _ { t , i } )$ . Importantly, we also need to take into account the sensitivity of Pareto membership to estimation error as these quantities are all empirical, and thus we thus adopt a soft-max style criteria as follows by incorporating a temperature. Specifically, for any candidate instance $x \in \mathcal { M } ^ { K }$ , let

$$
{ \mathrm { A l t } } _ { i } ( x ) = \left\{ \nu \in { \mathcal { M } } ^ { K } : { \mathbf { 1 } } \{ i \in { \mathcal { A } } ^ { \star } ( \nu ) \} \neq { \mathbf { 1 } } \{ i \in { \mathcal { A } } ^ { \star } ( x ) \} \right\}
$$

denote the set of alternative instances under which arm i has diferent Pareto membership. We then define

$$
Z _ { t , i } = \operatorname* { i n f } _ { \nu \in \mathrm { A l t } _ { i } ( \widetilde { \mu } _ { t } ) } \sum _ { a = 1 } ^ { K } N _ { a } ( t ) / 2 \sigma ^ { 2 } \| \widetilde { \mu } _ { t , a } - \nu _ { a } \| _ { \infty } ^ { 2 } , \quad \operatorname* { i n f } _ { \mathbf { \lambda } } \varpi = + \infty .\tag{6}
$$

Thus, $Z _ { t , i }$ measures the smallest change in the empirical means, weighted by their sample counts, needed to reverse the Pareto membership of arm i. It organically conveys the sensitivity of the current estimation of arm i; a larger value means more stable and thus should be more preferred. We are now ready to define an adaptive temperature by reversely depending on the value of $Z _ { t , i }$ For $c _ { \tau } > 0$ , let us define

$$
\tau _ { t , i } = { c \tau } / { \sqrt { 1 + Z _ { t , i } } } .\tag{7}
$$

By default, we set $c _ { \tau } = 1$ unless explicitly stated otherwise. Then we find the soft target $w _ { t , i } =$ exp(− $\cdot \underline { { R _ { t , i } } } / \tau _ { t , i } )$ as a proxy for the original target $\frac { \mathbb { I } \{ i \in \mathcal { A } ^ { \star } \} } { m }$ . As $Z _ { t , i }$ increases, the temperature   
$\overline { { \sum _ { j = 1 } ^ { K } \exp ( - R _ { t , j } / \tau _ { t , j } ) } }$   
decreases, the weight for arm i also drops. By doing so, it serves as a penalty for the stability to   
incentivize exploring less stable arms.

Finally, we derive the one-step fairness gain by $w _ { t , i } \log ( 1 + 1 / N _ { i } ( t ) )$ . Since the normalization in $w _ { t , i }$ is common to all arms, the same ranking follows from

$$
G _ { t , i } = \exp \{ - R _ { t , i } \sqrt { 1 + Z _ { t , i } } \ / c _ { \tau } \} \log ( 1 + 1 / N _ { i } ( t ) \big ) .\tag{8}
$$

Though greedy selection appears appealing, it can be less exploratory given the uncertainties that we face in multiple coordinates. A top-two sampling idea has been proposed [Russo, 2020, Qin and You, 2026] to sharply cope with this by maintaining two plausibly good arms and randomly selecting from them. We customize this idea through the design of the criterion and the sampling probability. Precisely, let $b _ { t } , c _ { t }$ be the two distinct arms with largest and second largest gains (i.e., the top two maximizers of $G _ { t , i } )$ , and let the empirical Pareto front be $\widehat { \mathcal { A } ^ { \star } } _ { t } = \mathcal { A } ^ { \star } ( \widetilde { \mu } _ { t } )$

There are two alternatives that we can leverage to decide how to select among the top two arms, which are typical hard parts of adopting top-two algorithms (but very fair enough as a price for their elegancy and interpretability). First, the leader is selected with a constant probability p, e.g., 0.70, and the challenger otherwise. When $| \widehat { \mathcal { A } ^ { \star } } _ { t } | = 1$ , the leader is selected surely.

Second, the probability adaptively responds to the diference in the two candidates’ sample counts:

$$
p _ { t } = \left\{ \begin{array} { l l } { 1 , } & { | \widehat { \mathcal { A } } _ { t } | = 1 , } \\ { { \mathrm { c l i p } } _ { [ 0 . 4 5 , 0 . 9 5 ] } [ 0 . 7 0 + 0 . 2 5 \operatorname { t a n h } \left( \frac { N _ { c _ { t } } ( t ) - N _ { b _ { t } } ( t ) } { 1 + 0 . 1 5 \{ N _ { b _ { t } } ( t ) + N _ { c _ { t } } ( t ) \} } \right) ] , } & { b _ { t } , c _ { t } \in \widehat { \mathcal { A } } _ { t } , } \\ { p , e . g . , 0 . 7 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{9}
$$

These parameter choices, such as 0.7, are for analytical convenience; in practice, we can tune them as needed. Subsequently, we choose the leader arm $b _ { t }$ with probability $p _ { t }$ , and the challenger arm $c _ { t }$ otherwise. Notably, after reliable identification, the two largest gains correspond to the two least-sampled true Pareto arms, for which $R _ { t , i } = 0$ and the gains reduce to log $\left( 1 + 1 / N _ { i } ( t ) \right)$ . The leader, which has the smaller arm count, exhibits a probability at least $0 . 7 0 > 1 / 2$ , thereby favoring exploration. The full procedure of the proposed method reads as Algorithm 1.

Algorithm 1 Soft-target adaptive top-two sampling   
Require: Horizon $T \ge 2 K , c _ { \tau } > 0 .$ and a fixed choice of the rule for $p _ { t }$   
1: Pull every arm twice; initialize $N _ { i } ( 2 K )$ and $\widetilde { \mu } _ { 2 K , i }$   
2: for $t = 2 K + 1 , \dots , T$ do   
3: Compute $R _ { t , i } , Z _ { t , i } ,$ and $G _ { t , i }$ using equation 5–equation 8   
4: Find the two distinct arms $b _ { t } , c _ { t }$ with largest gains   
5: Determine $p _ { t }$ using the chosen rule   
6: Select $A _ { t } = b _ { t }$ with probability p<sub>t</sub>; otherwise select $A _ { t } = c _ { t }$   
7: Pull $A _ { t }$ and update the counts and empirical means   
8: end for

## 4 Theoretical Guarantee

To the best of our knowledge, progress on provably balanced allocation has been quite limited in multi-objective bandits, largely due to the lack of a coherent and well-defined measure of “balance”. It is our understanding that no bounds on the fairness measure across Pareto optimal arms exist. Fortunately, the new framework developed herein allows us to take steps in this direction with the powerful handle of the mature theory on KL divergence and distributional convergence, as well as the new method. We proceed as follows: we first derive an upper bound for this method in Section 4.1 and then sketch an information-theoretic lower bound for any reasonable policy in Section 4.2.

## 4.1 Upper Bound

Traditionally, we prove upper bounds on performance measures in expectation, as commonly seen in the bandit literature. Therefore, we first bound the expected fairness in this section. Naturally, an upper bound on expected fairness also bounds the pseudo-fairness $\hat { \mathcal { F } } _ { T }$ , defined over the expected allocation, by the convexity of KL divergence. Beyond this, it also seems both appealing and necessary to study the variance of fairness. By definition, the KL divergence depends inversely on the empirical measure induced by a sequence of random arm selections and thus may vary substantially across trajectories. The variance thus provides a measure of how reliably the policy achieves fair allocation, as ultimately we have a limited number of trajectories in practice.

For the analysis, we introduce a separation margin that measures the sensitivity of the Pareto membership. Specifically, let us define

$$
\eta _ { \mu } ^ { \mathrm { P } } = \operatorname* { m i n } _ { i \in \mathcal { A } ^ { \star } } \operatorname* { m i n } _ { j \neq i } \operatorname* { m a x } _ { r \in [ d ] } ( \mu _ { i } ^ { ( r ) } - \mu _ { j } ^ { ( r ) } ) , \quad \eta _ { \mu } ^ { \mathrm { D } } = \operatorname* { m i n } _ { i \notin \mathcal { A } ^ { \star } } \operatorname* { m a x } _ { j \neq i } \operatorname* { m i n } _ { r \in [ d ] } ( \mu _ { j } ^ { ( r ) } - \mu _ { i } ^ { ( r ) } ) ,\tag{10}
$$

where $\eta _ { \mu } ^ { \mathrm { D } } = + \infty$ if all arms are Pareto optimal. The first margin, $\eta _ { \mu } ^ { \mathrm { P } }$ , measures the smallest advantage that a Pareto optimal arm has over any other arm in at least one objective. The second margin, $\eta _ { \mu } ^ { \mathrm { D } }$ , measures how clearly we can establish that an arm is dominated. Since it sufices to find one arm that outperforms it in every objective, we consider the arm with the largest minimum advantage across objectives. Taking the minimum over all dominated arms then identifies the hardest boundary.

Let $\eta _ { \mu } = \operatorname* { m i n } \{ \eta _ { \mu } ^ { \mathrm { P } } , \eta _ { \mu } ^ { \mathrm { D } } \}$ . We assume $\eta _ { \mu } > 0$ , which ensures that every Pareto optimal arm is strictly better than every other arm in at least one objective, while every dominated arm is strictly dominated by some arm in all objectives, in analogue to strict dominance [Kone et al., 2024]. Consequently, Pareto membership is preserved under suficiently small perturbations of the mean vectors. It can be arbitrarily small and is unknown; smaller ones mean more sensitivity and harder problems.

Throughout the following theorems, we consider that M is bounded, $\eta _ { \mu } > 0$ , and

$$
c _ { \tau } > 3 2 \sqrt { 2 } { \frac { B _ { \mathcal { M } } ^ { 2 } } { \eta _ { \mu } ^ { 2 } } } , \qquad B _ { \mathcal { M } } = \operatorname* { m a x } _ { r \in [ d ] } ( b _ { r } - a _ { r } ) .\tag{11}
$$

To state the bounds, let us define $\begin{array} { r } { s = 1 - \frac { 3 2 \sqrt { 2 } B _ { \mathcal { M } } ^ { 2 } } { c _ { \tau } \eta _ { \mu } ^ { 2 } } \in \left( 0 , 1 \right) } \end{array}$ and

$$
T _ { 0 } = \lceil 4 \operatorname * { m a x } \{ [ 1 2 8 K ( 1 + { \frac { 3 8 4 \sigma ^ { 2 } } { s \eta _ { \mu } ^ { 2 } } } ) ] ^ { 8 / s } , { \frac { 4 K d } { s } } ( 1 + { \frac { 3 2 \sigma ^ { 2 } } { \eta _ { \mu } ^ { 2 } } } ) , [ { \frac { 3 2 ( K - m ) c _ { \tau } ^ { 2 } \sigma ^ { 2 } } { \eta _ { \mu } ^ { 2 } } } ] ^ { 2 } \} \rceil .\tag{12}
$$

The quantity s measures how far the temperature parameter lies above its suficient lower bound. The three terms in $T _ { 0 }$ account for exploration of all arms, accumulated estimation errors, and pulls of dominated arms, respectively.

Theorem 4.1. Under the preceding conditions, for every $T \geq T _ { 0 }$ , we have

$$
\bar { \mathcal { F } } _ { T } \leq \mathbb { E } _ { \mu } \mathcal { F } _ { T } \leq \frac { 8 } { T } \left[ ( K - m ) \left( 3 + \frac { c _ { \tau } ^ { 2 } \sigma ^ { 2 } } { \eta _ { \mu } ^ { 2 } } \log ( T + 1 ) \right) + \sqrt { T _ { 0 } } + 1 6 m \right] + \frac { 5 1 2 T _ { 0 } } { T ^ { 2 } } \log \frac { T } { m } .\tag{13}
$$

Moreover,

$$
\mathrm { V a r } _ { \mu } ( \mathcal { F } _ { T } ) \leq \frac { 1 2 8 } { T ^ { 2 } } [ ( K - m ) ( 3 + \frac { c _ { \tau } ^ { 2 } \sigma ^ { 2 } } { \eta _ { \mu } ^ { 2 } } \log ( T + 1 ) ) + \sqrt { T _ { 0 } } ] ^ { 2 } + \frac { 1 6 3 8 4 m ^ { 2 } } { T } + \frac { 5 1 2 T _ { 0 } } { T ^ { 2 } } \log ^ { 2 } \frac { T } { m } .\tag{14}
$$

In other words, we have $\mathbb { E } _ { \mu } \mathcal { F } _ { T } = O ( \frac { \log T } { T } )$ and $\begin{array} { r } { \operatorname { V a r } _ { \mu } ( \mathcal { F } _ { T } ) = O ( \frac { 1 } { T } ) } \end{array}$ . Here, $T _ { 0 }$ and the coeficients in the bounds jointly reflect the dificulty of the problem instance through their dependence on $\eta _ { \mu } , \sigma _ { \mathrm { { \rangle } } }$ $B _ { \mathcal { M } } , K$ , and $m = | \mathcal { A } ^ { \star } |$ . In particular, $T _ { 0 }$ increases as the temperature approaches its lower bound, corresponding to $s \downarrow 0$ , which may result from a small $\eta _ { \mu }$ , a feature of a hard problem instance. Indeed, we can further decompose the bounds into contributions from pulls of dominated arms and imbalance among Pareto optimal arms, which more precisely identify the sources of unfairness.

Theorem 4.2 (Arm pulls and imbalance). Under the same conditions, recall that $\begin{array} { r } { B _ { T } = \sum _ { i \notin \mathcal { A } ^ { \star } } N _ { i } ( T ) , \Phi _ { T } = } \end{array}$ $\begin{array} { r } { \sum _ { i \in \mathcal { A } ^ { \star } } \Big ( N _ { i } ( T ) - \frac { T - B _ { T } } { m } \Big ) ^ { 2 } } \end{array}$ . For every $T \geq 2 K$

$$
\mathbb { E } _ { \mu } B _ { T } \leq \sqrt { \mathbb { E } _ { \mu } B _ { T } ^ { 2 } } \leq ( K - m ) \left( 3 + \frac { c _ { \tau } ^ { 2 } \sigma ^ { 2 } } { \eta _ { \mu } ^ { 2 } } \log ( T + 1 ) \right) + \sqrt { T _ { 0 } } .\tag{15}
$$

The imbalance potential satisfies

$$
\begin{array} { r } { \mathbb { E } _ { \mu } \Phi _ { T } \leq T + T _ { 0 } , \qquad \mathbb { E } _ { \mu } \Phi _ { T } ^ { 2 } \leq T ^ { 2 } ( T + T _ { 0 } ) . } \end{array}\tag{16}
$$

Notably, when $m = 1 , \Phi _ { T } = 0$ identically, which is reduced to the optimization of eficiency in spirit to minimizing the Pareto regret; the functional is diferent though.

The proof sketch is as follows; we defer the complete proof to Appendix A. We first show that the condition on the temperature $c _ { \tau }$ ensures logarithmic exploration of every arm with high probability. Combining this with concentration of the empirical means and positive Pareto separation, we bound the second moment of the number of rounds that fail the required sampling or accuracy conditions by $T _ { 0 }$ . On the remaining rounds, the empirical Pareto set is correct, and a dominated arm can be selected only while its count is at most $( c _ { \tau } ^ { 2 } \sigma ^ { 2 } / \eta _ { \mu } ^ { 2 } ) \log ( t + 1 )$ , since its gain otherwise falls below those of the Pareto optimal arms. This gives the bounds on dominated-arm pulls. Meanwhile, the gains of Pareto optimal arms reduce to $\log ( 1 + 1 / N _ { i } ( t ) )$ , and the selection rule simply favors smaller counts. We use this to show that the conditional expected increase of $\Phi _ { t }$ is at most one on these rounds, yielding $\mathbb { E } _ { \mu } \Phi _ { T } \le T + T _ { 0 }$ . Its second moment then follows from $\Phi _ { T } ^ { 2 } \le T ^ { 2 } \Phi _ { T }$ . To translate these bounds into fairness guarantees, we show that, for $T \geq T _ { 0 }$ , every Pareto optimal arm receives at least $T / ( 8 m )$ pulls except on an event of probability at most $5 1 2 T _ { 0 } / T ^ { 2 }$ . Then the exact decomposition yields $\mathcal { F } _ { T } \leq 8 B _ { T } / T + 6 4 m \Phi _ { T } / T ^ { 2 }$ ; otherwise, we use the fact that $\mathcal { F } _ { T } \leq \log ( T / m )$ in Lemma 2.1. Taking first and second moments yields the expected unfairness and variance bounds. Finally, convexity of KL divergence gives $\bar { \mathcal { F } } _ { T } \leq \mathbb { E } _ { \mu } \mathcal { F } _ { T }$

It is useful to compare our suficient horizon $T _ { 0 }$ with the sample complexity for Pareto set identification in Garivier et al. [2024]. The latter, which is a pure exploration problem, derives the optimal sample complexity $T ^ { \star } ( \mu ) \log ( 1 / \delta )$ as the error probability $\delta$ vanishes. For Gaussian observations with covariance $\sigma ^ { 2 } I _ { d } .$ , our positive separation condition implies $\begin{array} { r } { T ^ { \star } ( \mu ) \leq \frac { 8 K \sigma ^ { 2 } } { \eta _ { \mu } ^ { 2 } } } \end{array}$ , since perturbing every mean coordinate by less than $\eta _ { \mu } / 2$ cannot change the Pareto set. Thus, both $T ^ { \star } ( \mu )$ and $T _ { 0 }$ reflect the dificulty of distinguishing Pareto membership through the separation margin. However, $T _ { 0 }$ also depends on the temperature parameter $c _ { \tau }$ and can be substantially larger than $T ^ { \star } ( \mu )$ . This reflects the additional dificulty of balancing allocation, the exploration-exploitation trade-of, and hypothetically, the conservatism of the choice of $T _ { 0 }$

## 4.1.1 Implications for Pareto regret

As mentioned earlier, fairness, by its definition, already measures the eficiency of the selected arms. To see this more precisely, let us introduce Pareto regret, a widely used metric for eficiency, and

then derive bounds on it. We define the Pareto suboptimality gap of arm i [Drugan and Nowe, 2013, Guan and Xu, 2026] by $\begin{array} { r } { \Delta _ { i } ^ { \mathrm { P } } = [ \operatorname* { m a x } _ { j \neq i } \operatorname* { m i n } _ { r \in [ d ] } \big ( \mu _ { j } ^ { ( r ) } - \mu _ { i } ^ { ( r ) } \big ) ] _ { - } } \end{array}$ <sub>+</sub>, and let $\Delta _ { \mathrm { m a x } } ^ { \mathrm { P } } = \mathrm { m a x } _ { i \in [ K ] } \Delta _ { i } ^ { \mathrm { P } }$ . The cumulative Pareto regret is

$$
R _ { T } ^ { \mathrm { P } } = \sum _ { t = 1 } ^ { T } \Delta _ { A _ { t } } ^ { \mathrm { P } } = \sum _ { i \notin \mathcal { A } ^ { \star } } \Delta _ { i } ^ { \mathrm { P } } N _ { i } ( T ) .
$$

Under the condition of positive Pareto separation, we immediately obtain $\Delta _ { i } ^ { \mathrm { P } } \geq \eta _ { \mu } > 0$ for every dominated arm.

Corollary 4.3 (Pareto regret). Under the assumptions of Theorem 4.2, for every $T \geq 2 K$

$$
\mathbb { E } _ { \mu } R _ { T } ^ { \mathrm { P } } \leq c _ { \tau } ^ { 2 } \sigma ^ { 2 } \log ( T + 1 ) \sum _ { i \notin { \mathcal { A } ^ { \star } } } \frac { 1 } { \Delta _ { i } ^ { \mathrm { P } } } + 3 \sum _ { i \notin { \mathcal { A } ^ { \star } } } \Delta _ { i } ^ { \mathrm { P } } + \Delta _ { \operatorname* { m a x } } ^ { \mathrm { P } } \sqrt { T _ { 0 } } ,\tag{17}
$$

where $T _ { 0 }$ is the same explicit quantity used in Theorem 4.2. Moreover, on every trajectory,

$$
R _ { T } ^ { \mathrm { P } } \leq \Delta _ { \operatorname* { m a x } } ^ { \mathrm { P } } B _ { T } \leq \Delta _ { \operatorname* { m a x } } ^ { \mathrm { P } } T \mathcal { F } _ { T } .\tag{18}
$$

Consequently, on the positive side, the same policy achieves logarithmic expected Pareto regret, consistent with existing work on Pareto regret [Drugan and Nowe, 2013, Guan and Xu, 2026], while additionally approaching a balanced allocation over the Pareto set. In terms of rates, it achieves the best of both worlds for these two goals. Importantly, we are not in a position to make the same claim about the coeficients (and thus the full term of the bound) without a formal comparison. In fact, on the flip side, this advantage comes at a cost; the Pareto regret has an additional term involving $\sqrt { T _ { 0 } }$ and a potentially larger coeficient of the leading log T term through the temperature parameter $c _ { \tau }$

## 4.2 Information-theoretic Lower Bounds

Whenever we propose a new metric, we are interested in establishing a lower bound on this metric for any reasonable policy. In other words, for any such policy and any problem instance, the expected fairness is at least a certain value. To do so, for a problem instance, we construct an alternative instance that are statistically dificult to distinguish, but on which similar sampling behavior can lead to substantially diferent fairness guarantees. We then show that the average expected fairness over these two instances must be at least a certain value, so that at least one of them has to attain this value.

In our context, this derivation has a dual purpose. On the one hand, it reveals the gap between the established upper bound and this unavoidable cost, thereby assessing the sharpness of the method. Under positive Pareto separation, we show that the method is rate-optimal, up to diferences in the coeficients. On the other hand, without this condition, we can sense how far a method can go. In fact, we show that the absence of positive separation leads to a lower bound with a larger rate. As a by-product, this suggests that the condition is necessary for the rate. More deeply, it precisely manifests the sensitivity of the Pareto front, a genuinely hard aspect of multi-objective problems.

## 4.2.1 Positive Pareto separation

We first prove a lower bound on the fairness achievable by any reasonable policy under positive Pareto separation. The key observation is that a dominated arm may become Pareto optimal under an alternative instance, and distinguishing the two instances yields exploration cost. We characterize this cost through the KL divergence between their reward distributions.

For an instance with mean vectors $\mu ,$ let $P _ { i }$ denote the reward distribution of arm $i ,$ and as before we denote $\mathbb { E } _ { \mu }$ as the expectation under these distributions. We focus on policies that are consistent across instances with respect to fairness (as opposed to regret [Lai and Robbins, 1985, Kaufmann et al., 2016, Lattimore and Szepesv´ari, 2019]), namely uniformly fair below.

Definition 4.4. A policy is uniformly fair if, for every admissible collection of reward distributions with mean vectors $\mu$ satisfying $\eta _ { \mu } > 0$ and every $\zeta \in ( 0 , 1 )$ ,

$$
\mathbb { E } _ { \mu } \mathcal { F } _ { T } = o ( T ^ { - 1 + \zeta } ) .
$$

Intuitively, it means that the policy should be sublinear in $T$ w.r.t. $T \mathbb { E } _ { \mu } \mathcal { F } _ { T }$ , in analogue to sublinear regret [Kaufmann et al., 2016, Lattimore and Szepesv´ari, 2019]. For a dominated arm $i ,$ consider replacing its reward distribution $P _ { i }$ with an admissible distribution $Q ;$ admissibility means the same sub-Gaussian assumption in $\mathcal { M }$ . Let $\nu$ denote the resulting mean vectors, so that $\nu _ { i } = \mathbb { E } _ { X \sim Q } X$ and $\nu _ { a } = \mu _ { a }$ for $a \neq i$ . Let us define

$$
\kappa _ { i } ^ { \mathrm { K L } } ( \mu ) = \operatorname* { i n f } _ { Q : \eta _ { \nu } > 0 \atop i \in \mathcal { A } ^ { \star } ( \nu ) } D _ { \mathrm { K L } } ( P _ { i } \| Q ) ,
$$

with inf $\varnothing = + \infty$ and $1 / \infty = 0$ . This quantity measures, among all alternatives ν under which arm i becomes Pareto optimal, the smallest KL divergence to $\nu .$ It thus measures the statistical dificulty of determining the Pareto membership of arm $i ;$ smaller values indicate that the instances are harder to distinguish. Consequently, we are ready to derive the following instance-dependent lower bound.

Theorem 4.5. For every instance with $\eta _ { \mu } > 0$ and every uniformly fair policy,

$$
\operatorname* { l i m } _ { T \to \infty } \frac { T \mathbb { E } _ { \mu } \mathcal { F } _ { T } } { \log T } \geq \sum _ { i \notin \mathcal { A } ^ { \star } ( \mu ) } \frac { 1 } { \kappa _ { i } ^ { \mathrm { K L } } ( \mu ) } .\tag{19}
$$

The lower bound increases as $\kappa _ { i } ^ { \mathrm { K L } } ( \mu )$ decreases, reflecting the greater exploration required to distinguish the Pareto membership of arm i. Whenever the right-hand side of equation 19 is positive, the expected fairness converge with a log $T / T$ rate within this policy class. This lower bound matches the rate of our upper bound, and we hope to inspire future work that improves the coeficients.

A careful reader may observe that the bound is asymptotic. Noting that the upper bound holds for a finite horizon $( \geq T _ { 0 } )$ , it may be favorable to also prove a finite-time lower bound. For this purpose,

Theorem 4.6 (Finite-time fairness lower bound). Fix an instance $\mu$ with $\eta _ { \mu } > 0$ and a nonempty subset $\mathcal { T } \subseteq [ K ] \backslash \mathcal { A } ^ { \star } ( \mu )$ . For each $i \in \mathcal { Z }$ , we choose an admissible reward distribution $Q _ { i }$ and let $\nu ^ { ( i ) }$ be the instance obtained by replacing only $P _ { i }$ with $Q _ { i }$ , s.t. $\eta _ { \nu ^ { ( i ) } } > 0 , i \in \mathcal { A } ^ { \star }$ , and $0 < D _ { \mathrm { K L } } ( P _ { i } \Vert Q _ { i } ) < \infty$ Then, for any policy and every $T \geq$ max{K, 4eK(K + 1)S} where $\begin{array} { r } { S = \sum _ { i \in \mathcal { T } } { 1 } / { D _ { \mathrm { K L } } ( P _ { i } | | Q _ { i } ) } } \end{array}$

$$
\operatorname* { m a x } \left\{ \mathbb { E } _ { \mu } \mathcal { F } _ { T } , \operatorname* { m a x } _ { i \in \mathcal { T } } \mathbb { E } _ { \nu ^ { ( i ) } } \mathcal { F } _ { T } \right\} \geq \frac { S } { T } \left( \log \frac { T } { 4 K ( K + 1 ) S } - \log \log \frac { T } { 4 K ( K + 1 ) S } \right) .\tag{20}
$$

In particular, the worst-case expected unfairness over these instances is at least $( 1 - o ( 1 ) ) S$ log $T / T$ uniformly over policies. The right-hand side is asymptotic to $\begin{array} { r } { \log T / T \sum _ { i \in \mathcal { T } } \frac { 1 } { D _ { \mathrm { K L } } ( P _ { i } \| Q _ { i } ) } ; } \end{array}$ when ${ \mathcal { T } } ^ { c } =$ $\mathcal { A } ^ { \star } ( \mu )$ , this is in spirit with the asymptotic bound above. Intuitively, statistically closer instances

may require longer horizons before the asymptotic lower bound becomes informative. The constants can be consistent with Theorem 4.1, thus supporting the rate optimality of the method for suficiently large $T$ whenever $\kappa _ { i } ^ { \mathrm { K L } } < \infty$ . Taking a closer look at the coeficient, we observe that

$$
\sum _ { i \in \mathcal { T } } \frac { 1 } { D _ { \mathrm { K L } } ( P _ { i } \| Q _ { i } ) } \leq \sum _ { i \notin \mathcal { A } ^ { \star } ( \mu ) } \frac { 1 } { \kappa _ { i } ^ { \mathrm { K L } } ( \mu ) } .
$$

Thus, the finite-time bound is more conservative, as a price we pay for a nonasymptotic guarantee. In a like manner, this conservatism appears in finite-time lower bounds on regret for classical bandits [Lattimore and Szepesv´ari, 2019].

## 4.2.2 Zero Pareto separation

When the separation is zero, it becomes a diferent story. Thereon, an arbitrarily small change in the mean rewards may change the Pareto set and, consequently, the ideal distribution. We show that this instability prevents any reasonable policy from achieving the aforementioned rate both at such a boundary instance.

To see this, consider two arms with two objectives and independent Gaussian noise $\mathcal { N } ( 0 , \sigma ^ { 2 } I _ { 2 } )$ where $\sigma > 0$ and $g \in ( 0 , 1 )$ are fixed:

$$
\begin{array} { r l } & { \nu _ { 0 } : \quad \mu _ { 1 } = ( 0 , 0 ) , \qquad \mu _ { 2 } = ( g , 0 ) , } \\ & { \nu _ { \varepsilon } : \quad \mu _ { 1 } = ( 0 , \varepsilon ) , \qquad \mu _ { 2 } = ( g , 0 ) , \quad 0 < \varepsilon < 1 . } \end{array}\tag{21}
$$

Under $\nu _ { 0 } .$ arm 2 dominates arm 1, but $\eta _ { \nu _ { 0 } } = 0$ because their second coordinates are equal. Under $\nu _ { \varepsilon }$ , both arms are Pareto optimal, with $\eta _ { \nu _ { \varepsilon } } = \operatorname* { m i n } \{ g , \varepsilon \} > 0$ . Thus, the target distribution across arms changes from $( 0 , 1 )$ to $( 1 / 2 , 1 / 2 )$ , regardless of how small ε is. That is, sampling arm 1 is necessary for fairness under $\nu _ { \varepsilon }$ , but accumulates unfairness under $\nu _ { 0 }$ . This dilemma diverts a policy from achieving the same fairness rate at $\nu _ { 0 }$ and $\nu _ { \varepsilon }$ . The formal statement is as follows.

Theorem 4.7. For any policy that is uniformly fair under positive Pareto separation, the instances in equation 21 satisfy

$$
\frac { \mathbb { E } _ { \nu _ { 0 } } N _ { 1 } ( T ) } { \log T } \longrightarrow \infty , \qquad \frac { T \mathbb { E } _ { \nu _ { 0 } } \mathcal { F } _ { T } } { \log T } \longrightarrow \infty .\tag{22}
$$

We observe a subtle dificulty in fairness learning at the boundary of Pareto optimality. An arbitrarily small change in the mean vectors (ϵ) can alter the Pareto set. Statistically distinguishing these instances requires increasingly many observations as the separation $\eta _ { \nu _ { \varepsilon } }$ approaches zero, while failing to do so can lead to unfair allocations of rate $\omega ( \log T / T )$

## 5 Variance Reduction

To quantify the relative variability of fairness across trajectories, let us adopt the signal-to-noise ratio (SNR), commonly defined as

$$
\mathrm { S N R } _ { T } = \mathbb { E } [ \mathcal { F } _ { T } ] \big / \sqrt { \mathrm { V a r } ( \mathcal { F } _ { T } ) } .
$$

The aforementioned upper and lower bounds imply that $\mathbb { E } [ \mathcal { F } _ { T } ] = \Theta ( \log T / T )$ and $\operatorname { V a r } ( \mathcal { F } _ { T } ) = O ( 1 / T )$ The standard deviation decreases more slowly than the expected fairness. Consequently, we obtain $\mathrm { S N R } _ { T } = \Omega ( \log T / \sqrt { T } )$ . This has direct implications for empirical evaluation. Precisely, given R

independent runs $( { \mathrm { i . e . } }$ , trajectories), we compute the sample mean of fairness $\overline { { \mathcal { F } } } _ { T , R }$ . Then its relative standard error is

$$
\sqrt { \mathrm { V a r } ( \overline { { \mathcal { F } } } _ { T , R } ) } / \mathbb { E } [ \mathcal { F } _ { T } ] = 1 / \sqrt { R } \mathrm { S N R } _ { T } = O \left( 1 / \log T \sqrt { T / R } \right) .
$$

This means that the evaluation is computationally expensive and thus less desirable, as maintaining a fixed relative estimation precision may require $R = \Omega ( T / \log ^ { 2 } T )$ in the worst-case scenario where $\mathrm { V a r } ( \mathcal { F } _ { T } ) = \Theta ( 1 / T )$ . Even when expected fairness vanishes at the rate $\Theta ( \log T / T )$ , reliable estimation of this expectation by the sample mean $\overline { { \mathcal { F } } } _ { T , R }$ may require a number of runs that grows nearly linearly with the horizon. To this end, we hope to propose a new method for variance reduction, so that we can increase $\mathrm { S N R } _ { T }$ and decrease R.

## 5.1 Methodology

The original top-two rule selects arms according to their empirical gains in equation 8, which are in the direction of reducing empirical unfairness. A true Pareto optimal arm whose mean is underestimated may receive few further observations, delaying the correction of its estimate and causing persistent underallocation. Indeed, such trajectories likely exhibit extreme unfairness and thus can contribute substantially to the variance of fairness. To fix this, we introduce two changes. We first promote extra exploration based directly on arm counts in the early stages, and add a confidence test for terminating this exploration timely. The complete procedure is summarized in Algorithm 2.

For a horizon $T > 2 K$ , we set $q _ { 0 } = 9 6$ and $p _ { g } = 0 . 7 0$ , and define the exploration quota

$$
\begin{array} { r l } & { \rho _ { T } = [ \sqrt { T } - q _ { 0 } \log ( 1 + T / 2 K ) ] _ { + } , } \\ & { h _ { t , T } = 2 + q _ { 0 } \log ( 1 + t / 2 K ) + \rho _ { T } \operatorname* { m i n } \{ 1 , t / [ \log T ] \} . } \end{array}\tag{23}
$$

The logarithmic term in $h _ { t , T }$ provides continuing exploration, while $\rho _ { T } \sim \sqrt { T }$ allows additional exploration when the separation is too small for this term alone to discover Pareto membership. The factor multiplying $\rho _ { T }$ raises the quota early in the horizon, ideally before imbalances accumulate.

While the exploration is active, let $j _ { t } \in \arg \operatorname* { m i n } _ { i } N _ { i } ( t )$ be a least-sampled arm. If $N _ { j _ { t } } ( t ) < h _ { t , T }$ , we select $j _ { t }$ with probability $p _ { g } ;$ with probability $1 - p _ { g }$ , we select the leader $b _ { t }$ or the challenger $c _ { t } \mathrm { i f } \ j _ { t }$ is already the leader $b _ { t }$ . Otherwise, we use the original top-two rule. The main diference is that an arm receives this protection regardless of its gain or estimated Pareto membership.

While this directly addresses the prolonged underallocation, we need to stop it once suficient statistical evidence is available, to prevent imbalance. The key is to use the empirical separation relative to the uncertainty in the estimates. Let $\begin{array} { r } { \Lambda _ { T } = \log ( 4 K d T ^ { 7 } ) , n _ { \operatorname* { m i n } } ( t ) = \operatorname* { m i n } _ { i } { N _ { i } ( t ) } } \end{array}$ , and we define the empirical separation and uncertainty as

$$
\widehat { \eta } _ { t } ^ { \mathrm { P } } = \operatorname* { m i n } _ { i \in \widehat { A ^ { \star } } _ { t } } \operatorname* { m i n } _ { j \neq i } \operatorname* { m a x } _ { r \in [ d ] } ( \widetilde { \mu } _ { t , i } ^ { ( r ) } - \widetilde { \mu } _ { t , j } ^ { ( r ) } ) ,\tag{24}
$$

$$
\widehat { \eta } _ { t } ^ { \mathrm { D } } = \operatorname* { m i n } _ { i \notin \widehat { A ^ { \star } } _ { t } } \operatorname* { m a x } _ { j \neq i } \operatorname* { m i n } _ { r \in [ d ] } ( \widetilde { \mu } _ { t , j } ^ { ( r ) } - \widetilde { \mu } _ { t , i } ^ { ( r ) } ) ,
$$

$$
\widehat { \eta } _ { t } = \operatorname* { m i n } \{ \widehat { \eta } _ { t } ^ { \mathrm { P } } , \widehat { \eta } _ { t } ^ { \mathrm { D } } \} , \qquad r _ { t } ^ { \operatorname* { m a x } } = \sqrt { 2 \sigma ^ { 2 } \Lambda _ { T } / n _ { \operatorname* { m i n } } ( t ) } .
$$

Here, $\widehat { \eta } _ { t } ^ { \mathrm { D } } = + \infty$ when all arms are empirically Pareto optimal. These are the empirical counterparts of the separation margins, and $r _ { t } ^ { \operatorname* { m a x } }$ bounds the uncertainty by leveraging the smallest arm count.

Algorithm 2 Evidence-gated adaptive top-two sampling (EG-ATT)   
Require: Horizon $T > 2 K ,$ , proxy $\sigma , q _ { 0 } = 9 6 , p _ { g } = 0 . 7 0 , c _ { \tau } = 1$   
1: Pull each arm twice; set resolved ← false   
2: Compute $\rho _ { T }$ from equation 23 and set $\Lambda _ { T } = \log ( 4 K d T ^ { 7 } )$   
3: for $t = 2 K + 1 , . . . , T$ do   
4: Compute $\widetilde { \mu } _ { t } , \widehat { A ^ { \star } } _ { t } ,$ and the gains $G _ { t , i }$ from equation $8$   
5: Identify the two distinct arms $b _ { t } , c _ { t }$ with the largest gains, using fixed tie-breaking   
6: Compute $p _ { t }$ from equation 9, $h _ { t , T }$ from equation 23, and $n _ { \mathrm { m i n } } ( t ) = \mathrm { m i n } _ { i } N _ { i } ( t )$   
7: if resolved = false then   
8: Compute $\widehat { \eta _ { t } }$ and $r _ { t } ^ { \operatorname* { m a x } }$ from equation 24   
9: if $\widehat { \eta _ { t } } \geq 8 r _ { t } ^ { \operatorname* { m a x } }$ then   
10: resolved ← true   
11: end if   
12: end if   
13: if resolved = false and $n _ { \mathrm { m i n } } ( t ) < h _ { t , T }$ then   
14: Let $j _ { t } \in \arg \operatorname* { m i n } _ { i } N _ { i } ( t )$ , breaking ties deterministically   
15: $k _ { t }  b _ { t }$ if $j _ { t } \neq b _ { t } ,$ else $k _ { t }  c _ { t }$   
16: Draw $A _ { t } = j _ { t }$ with probability $p _ { g } .$ , else $A _ { t } = k _ { t }$   
17: else   
18: Draw $A _ { t } = b _ { t }$ with probability $p _ { t } ,$ else $A _ { t } = c _ { t }$   
19: end if   
20: Pull $A _ { t }$ and update the counts and empirical means   
21: end for

The choice of $\Lambda _ { T }$ ensures that the confidence bounds hold simultaneously through coordinates and time T with probability at least $1 - 1 / ( 2 T ^ { 6 } )$

We permanently disable the exploration after $\widehat { \eta } _ { t } \geq 8 r _ { t } ^ { \operatorname* { m a x } }$ . The empirical comparisons are then suficiently separated relative to their uncertainty to assign Pareto membership on the confidence event. Thereafter, we follow the original top-two rule to promote balanced allocation as before.

## 5.2 Theoretical Guarantee

We show that the new mechanism controls the number of pulls of dominated arms, while the adaptive top-two rule controls the imbalance among Pareto optimal arms after release. These properties yield bounds on the expected fairness, which should remain of the same order as before since we use the same imbalance control, and on its variance, which decreases due to the additional control.

Theorem 5.1. Suppose $\eta _ { \mu } > 0$ and consider Algorithm 2. Define the constant $C _ { \mu } = 2 K + 5 +$ $2 / p _ { g } [ K ( 1 + 5 1 2 \sigma ^ { 2 } / \eta _ { \mu } ^ { 2 } ) + 8 ]$ . For every horizon $T > 2 K$ satisfying $\rho _ { T } \ge \operatorname* { m a x } \{ 2 , \lceil 5 1 2 \sigma ^ { 2 } \Lambda _ { T } / \eta _ { \mu } ^ { 2 } \rceil \}$ and $T \geq ( 1 + 8 m ) C _ { \mu } \Lambda _ { T }$ , there exists a constant $D > 0$ , depending only on $| A ^ { \star } |$ , such that

$$
\mathbb { E } _ { \mu } \mathcal { F } _ { T } \le \frac { 2 C _ { \mu } \Lambda _ { T } } { T } + \frac { D } { T ^ { 2 } } , \quad \mathrm { V a r } _ { \mu } ( \mathcal { F } _ { T } ) \le \frac { 8 C _ { \mu } ^ { 2 } \Lambda _ { T } ^ { 2 } } { T ^ { 2 } } + \frac { D \log ^ { 2 } T } { T ^ { 4 } } .\tag{25}
$$

Moreover, we can bound the moments ofthe arm pulls and imbalance potential as $\mathbb { E } _ { \mu } B _ { T } ^ { 4 } \le C _ { \mu } ^ { 4 } \Lambda _ { T } ^ { 4 } + T ^ { - 2 }$ and $\mathbb { E } _ { \mu } \Phi _ { T } ^ { 2 } \le D + T ^ { - 2 }$

In terms of order, we obtain $\mathbb { E } _ { \mu } \mathcal { F } _ { T } = O _ { \mu } ( \frac { \log T } { T } )$ and $\begin{array} { r } { \mathrm { V a r } _ { \mu } ( \mathcal { F } _ { T } ) = O _ { \mu } ( \frac { \log ^ { 2 } T } { T ^ { 2 } } ) } \end{array}$ by noting that $\boldsymbol { \Lambda } _ { T } =$ O(log T). On the positive side, compared to Theorem 4.1, we preserve the same order of expected

fairness but substantially reduce the order of the variance, from $O \big ( \frac { 1 } { T } \big )$ to $O _ { \mu } ( \frac { \log ^ { 2 } T } { T ^ { 2 } } )$ . This quadratic decay has a clear practical advantage. To see this, the relative standard error becomes

$$
\sqrt { \mathrm { V a r } ( \overline { { \mathcal F } } _ { T , R } ) } / \mathbb { E } [ \mathcal { F } _ { T } ] = O ( R ^ { - 1 / 2 } ) ;
$$

it leverages the fact that the upper bound on the expected fairness implies that the policy is uniformly fair (sublinearity holds by the logarithmic rate). Therefore, maintaining a fixed relative estimation precision may only require a constant number of runs, independent of the time horizon $T _ { i }$ , in contrast to the previous order of $T / \log ^ { 2 } T$ runs.

However, on the flip side, the upper bounds difer in their leading coeficients. The consequences, and thus the decision regarding which method to use, depend on the interplay between the separation margin $\eta _ { \mu }$ and the horizon T. Precisely, for the mean fairness, with $m = | \mathcal { A } ^ { \star } |$ , the coeficient of log $T / T$ is $8 ( K - m ) c _ { \tau } ^ { 2 } \sigma ^ { 2 } / \eta _ { \mu } ^ { 2 }$ for the original top-two rule where $c _ { \tau } > 3 2 \sqrt { 2 } B _ { \mathcal { M } } ^ { 2 } / \eta _ { \mu } ^ { 2 }$ , compared with $1 4 C _ { \mu } = 6 8 K + 3 9 0 + 2 0 4 8 0 K \sigma ^ { 2 } / \eta _ { \mu } ^ { 2 }$ for the new variant. We observe that when $\eta _ { \mu }$ is really small, which means that it is hard for us to sort out Pareto membership, using the new variant may ofer a greater advantage, as its coeficient is of order $1 / \eta _ { \mu } ^ { 2 }$ , whereas the original rule exhibits $1 / \eta _ { \mu } ^ { 6 }$ . This appears very natural and consistent with our earlier motivation. When $\eta _ { \mu }$ is large, then the plausibly larger coeficient of the new variant reflects the trade-of introduced by the variance reduction mechanism. The quota allocates additional pulls to undersampled arms, including dominated arms, and thus is more conservative, at the cost of balancing the Pareto optimal arms. When the horizon $T$ is very large (e.g., at least $1 0 ^ { 5 } )$ , it is still surely more advantageous to use the new variant for its rate advantage. At smaller horizons, such an advantage fades away as the larger coeficient unfortunately becomes the bottleneck. Perhaps when the problem is easier, it is better to be less conservative, as the original method does. In fact, our previous concern about $R = \Omega ( T )$ may also fade, and then the original top-two method remains a safer choice.

For the variance, the comparison is more subtle, as the original top-two rule has both $\mathrm { ~ a ~ } 1 / T$ term and terms of order $\mathrm { l o g } ^ { 2 } T / T ^ { 2 }$ in its upper bound. The coeficient of the former is $1 6 3 8 4 m ^ { 2 }$ independent of $\eta _ { \mu } ,$ , whereas the coeficient of the latter includes $1 2 8 ( K - m ) ^ { 2 } c _ { \tau } ^ { 4 } \sigma ^ { 4 } / \eta _ { \mu } ^ { 4 }$ , which grows at the order of $1 \dot { / } \eta _ { \mu } ^ { 1 2 }$ when $c _ { \tau } \asymp 1 / \eta _ { \mu } ^ { 2 }$ . In comparison, the new variant has a variance bound of order $\log ^ { 2 } T / T ^ { 2 }$ , with coeficient $3 9 2 C _ { \mu } ^ { 2 } \stackrel { . } { = } 2 ( 6 8 K + 3 9 0 + 2 0 4 8 0 K \sigma ^ { 2 } / \eta _ { \mu } ^ { 2 } ) ^ { 2 }$ , which grows at the order of $1 / \eta _ { \mu } ^ { 4 }$ . Therefore, when $\eta _ { \mu }$ is small, the new variant may ofer an advantage not only through its faster variance decay but also through its dependence on the separation margin. When $\eta _ { \mu }$ is larger, the advantage still holds when the horizon is large enough; nevertheless, it likewise becomes less clear at smaller horizons.

## 6 Simulation Experiments

Conventionally, upper bounds are conservative and thus reflect performance in a pessimistic sense. A pathway to examine potentially more optimistic performance is to conduct numerical studies and report the observed performance. Oftentimes, simulation experiments sufice for this purpose. We explore this idea in the context of fairness herein. Another advantage is that we can sense how the separation margin and the time horizon afect the actual mean and variance of unfairness under the two methods. Moreover, this supports comparisons with existing methods for stochastic multi-objective bandits. Though these methods are mostly empirical or heuristic, they sometimes do work in practice; we believe that it is crucial to understand when they work, when they do not, which ones work, and why.

Experimental setup. We consider a six-arm Bernoulli instance with four Pareto-optimal arms and a five-arm Gaussian instance with three Pareto-optimal arms. The separation margins are $\eta _ { \mu } \in \{ 0 . 0 0 5 , 0 . 0 1 , 0 . 1 \}$ for Bernoulli rewards and $\eta _ { \mu } \in \{ 0 . 0 0 5 , 0 . 0 2 , 0 . 1 \}$ for Gaussian rewards. Within each family, we vary the margin through afine transformations of the mean reward vectors while preserving the Pareto set. All mean vectors lie in $[ 0 , 1 ] ^ { 2 }$ , and the Gaussian rewards have standard deviation $\sigma = 0 . 1 2$ . The mean vectors are reported in Appendix B.

We consider $T \in \{ 5 , 0 0 0 , 5 0 , 0 0 0 \}$ and initialize every arm with two observations. For each instance, we measure terminal unfairness $\mathcal { F } _ { T } = D _ { \mathrm { K L } } ( \pi ^ { \star } \| \widehat { \pi } _ { T } )$ , where $\pi ^ { \star } = \operatorname { U n i f } ( A ^ { \star } )$ and $\widehat { \pi } _ { T } ( i ) = N _ { i } ( T ) / T$ We report its sample mean and variance over 240 independent held-out runs.

The baselines include Pareto-UCB1 [Drugan and Nowe, 2013], Pareto Thompson sampling (PTS) [Yahyaa et al., 2014a], Width-guided UCB [Guan and Xu, 2026], Pareto-KG [Yahyaa et al., 2014b], and Annealing-Pareto [Yahyaa et al., 2014a]. Their parameters mostly follow from what have been reported in their respective work. We tune ATT and EG-ATT separately for each instance and horizon through noisy Bayesian optimization (BO), using two objectives:

$$
J _ { \mathrm { m e a n + S D } } = \widehat { \mathbb { E } } \mathcal { F } _ { T } + \frac { 1 } { 2 } \sqrt { \widehat { \mathrm { V a r } } ( \mathcal { F } _ { T } ) } , J _ { \mathrm { l o g } } = \frac { 1 } { 2 } \log \left( \widehat { \mathbb { E } } \mathcal { F } _ { T } + \mathbf { l } \mathbf { 0 } ^ { - 4 } \right) + \frac { 1 } { 2 } \log \left( \widehat { \mathrm { V a r } } ( \mathcal { F } _ { T } ) + \mathbf { l } \mathbf { 0 } ^ { - 6 } \right) .
$$

We vary the temperature and leader probability of both methods, and additionally the quota and gate parameters of EG-ATT. Each search evaluates 20 configurations on 28 pilot runs, compares the leading candidates on 90 validation runs, and evaluates on the held-out runs.

Tables 1 –2 summarize the results. We hope to interpret them and extract the insights in what follows.

Efect of the separation margin and horizon. A larger separation margin reduces mean unfairness for both proposed methods in every instance considered. For example, at $T = 5 0 { , } 0 0 0 { \mathrm { . } }$ , the mean unfairness of ATT on Gaussian rewards decreases from 0.0713 at $\eta _ { \mu } = 0 . 0 0 5$ to $2 . 9 6 \times 1 0 ^ { - 4 }$ at $\eta _ { \mu } = 0 . 1$ , while that of EG-ATT decreases from 0.0699 to 0.00168. This coincides with the theory, where the bounds scale reversely with the margin $\eta$ and also the intuition that a smaller margin may indicate a harder instance. Increasing the horizon from 5,000 to 50,000 further reduces the mean for both methods in all six instances, also sort of predictable by the rate $O ( \log T / T )$ . The variance exhibits a diferent pattern. On Bernoulli rewards with $\eta _ { \mu } = 0 . 0 1$ , the mean unfairness of ATT decreases from 0.3041 to 0.1368, whereas its variance increases from 0.00904 to 0.0282. Hypothetically, this may be due to that the horizon $T$ has not reached the suficient horizon for the upper bound on variance.

The role of quota protection. Quota protection helps balance the allocation among Paretooptimal arms, but can also incur additional pulls of dominated arms. The relative importance of these two efects determines its impact on mean unfairness and variance. For Bernoulli rewards at $T = 5 0 { , } 0 0 0$ , EG-ATT reduces the variance by $7 7 \%$ at $\eta _ { \mu } = 0 . 0 1$ , with similar mean unfairness, and improves both metrics at $\eta _ { \mu } = 0 . 1$ . Likewise, for gaussian rewards, EG-ATT also improves both metrics at $\eta _ { \mu } = 0 . 0 2$ . Conversely, in some cases, for example at $\eta _ { \mu } = 0 . 1$ , it achieves nearly zero variance at the cost of larger mean unfairness.

Efect of the BO objective. The tuning objective also shapes the trade-of of mean and variance we observe. The distinction is particularly visible at $T = 5 { , } 0 0 0$ . On Bernoulli rewards with $\eta _ { \mu } = 0 . 0 1$ log-tuned EG-ATT attains variance $7 . 5 0 \times 1 0 ^ { - 1 0 }$ but mean unfairness 0.4053, whereas mean+SD tuning yields mean 0.3020 and variance 0.0100. We observe a similar outcome for Gaussian rewards at $\eta _ { \mu } = 0 . 0 2 \colon$ log-tuned EG-ATT has variance $6 . 0 7 \times 1 0 ^ { - 1 5 }$ and mean unfairness 0.2196, compared with 0.0020 and 0.0390 under mean+SD tuning.

Table 1: Mean and variance of terminal KL unfairness on Bernoulli rewards for $T \in \{ 5 , 0 0 0 , 5 0 , 0 0 0 \}$ The smallest mean and variance for each separation margin and horizon are in bold.
<table><tr><td></td><td colspan="2"> $\eta _ { \mu } = 0 . 0 0 5$ </td><td colspan="2"> $\eta _ { \mu } = 0 . 0 1$ </td><td colspan="2"> $\eta _ { \mu } = 0 . 1$ </td></tr><tr><td>Method</td><td>Mean</td><td>Variance</td><td>Mean</td><td>Variance</td><td>Mean</td><td>Variance</td></tr><tr><td>Horizon  $T = 5 , 0 0 0$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ATT (mean+SD BO)</td><td>0.3655</td><td>0.0042</td><td>0.3041</td><td>0.0090</td><td>0.0246</td><td>0.0028</td></tr><tr><td>ATT (log BO)</td><td>0.3651</td><td>0.0043</td><td>0.3099</td><td>0.0048</td><td>0.0246</td><td>0.0028</td></tr><tr><td>EG-ATT (mean+SD BO)</td><td>0.3719</td><td>0.0014</td><td>0.3020</td><td>0.0100</td><td>0.0227</td><td> $6 . 5 5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>EG-ATT (log BO)</td><td>0.4048</td><td> $\mathbf { 2 . 7 0 \times 1 0 ^ { - 6 } }$ </td><td>0.4053</td><td> $\mathbf { 7 . 5 0 \times 1 0 ^ { - 1 0 } }$ </td><td>0.2650</td><td> $\mathbf { 1 . 9 2 \times 1 0 ^ { - 1 4 } }$ </td></tr><tr><td>Pareto-UCB1</td><td>0.3686</td><td>0.0015</td><td>0.3291</td><td>0.0015</td><td>0.1116</td><td>0.00034</td></tr><tr><td>PTS</td><td>0.4355</td><td>0.0319</td><td>0.3418</td><td>0.0259</td><td>0.0222</td><td>0.0025</td></tr><tr><td>Width-guided UCB</td><td>0.3753</td><td>0.0016</td><td>0.3345</td><td>0.0020</td><td>0.2208</td><td>0.0067</td></tr><tr><td>Pareto-KG (adapted)</td><td>0.6248</td><td>0.2845</td><td>0.4734</td><td>0.2805</td><td>0.3119</td><td>0.2331</td></tr><tr><td>Annealing-Pareto</td><td>3.0503</td><td>0.9311</td><td>3.0684</td><td>0.8474</td><td>2.0157</td><td>0.6267</td></tr><tr><td>Horizon T = 50, 000</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ATT (mean+SD BO)</td><td>0.2495</td><td>0.0168</td><td>0.1368</td><td>0.0282</td><td>0.0068</td><td>0.0057</td></tr><tr><td>ATT (log BO)</td><td>0.2758</td><td>0.0044</td><td>0.1578</td><td>0.0044</td><td>0.0140</td><td>0.0182</td></tr><tr><td>EG-ATT (mean+SD BO)</td><td>0.2488</td><td>0.0200</td><td>0.1415</td><td>0.0066</td><td>0.0028</td><td> $7 . 7 2 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>EG-ATT (log BO)</td><td>0.2730</td><td>0.0047</td><td>0.1566</td><td>0.0044</td><td>0.0036</td><td> $\mathbf { 2 . 0 8 \times 1 0 ^ { - 7 } }$ </td></tr><tr><td>Pareto-UCB1</td><td>0.3079</td><td>0.0012</td><td>0.2460</td><td>0.0012</td><td>0.0306</td><td> $1 . 6 4 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>PTS</td><td>0.2846</td><td>0.0276</td><td>0.1837</td><td>0.0272</td><td>0.0033</td><td> $2 . 6 8 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Width-guided UCB</td><td>0.2940</td><td>0.0021</td><td>0.2032</td><td>0.0012</td><td>1.6001</td><td>0.0106</td></tr><tr><td>Pareto-KG (adapted)</td><td>0.2556</td><td>0.0628</td><td>0.1436</td><td>0.0457</td><td>0.2565</td><td>0.3260</td></tr><tr><td>Annealing-Pareto</td><td>4.7997</td><td>1.5698</td><td>4.5389</td><td>1.7251</td><td>2.8636</td><td>1.3302</td></tr></table>

Comparison with existing methods. Among all the methods, our newly proposed methods almost always outperform, with only one exception where the new methods and PTS are more or less the same. In many cases, it achieves an order-of-magnitude improvement. Across 120 pairwise comparisons with the five baselines, they reduce mean unfairness by more than 50% in 60 cases (50.0%), and by more than 90% in 37 cases (30.8%); e.g., a 96.9% reduction relative to Annealing-Pareto on Bernoulli rewards with $\eta _ { \mu } = 0 . 0 1$ at $T = 5 0 { , } 0 0 0$ . Particularly, PTS is competitive when the separation margin and the time horizon are large. For Bernoulli rewards with $\eta _ { \mu } = 0 . 1$ and $T = 5 { , } 0 0 0$ , it achieves the smallest mean unfairness, 0.0222, among all; the proposed method achieves 0.0227. At $T = 5 0 { , } 0 0 0 .$ , its mean is 0.00325 on Bernoulli rewards and $5 . 1 8 \times 1 0 ^ { - 4 }$ on Gaussian rewards. The hypothesis is that by sampling from a posterior Pareto set, PTS can spread pulls among Pareto-optimal arms once their membership becomes clearer which is more feasible with a larger separation margin. Meanwhile, Pareto-KG can achieve a similar mean, such as 0.1436 on Bernoulli rewards with $\eta _ { \mu } = 0 . 0 1$ at $T = 5 0 { , } 0 0 0$ , but its variance 0.0457 is much larger. It samples uniformly from a Pareto set adjusted by knowledge-gradient bonuses, which can yield competitive mean fairness. However, reward fluctuations afect which arms enter this set and for how long, thereby yielding larger fluctuations in fairness across runs.

On the contrary, Pareto-UCB1 often has a small variance, attaining the smallest variance among the methods in the table on the two more dificult Bernoulli instances at $T = 5 0 { , } 0 0 0$ . Its mean unfairness, however, remains larger than that of the proposed methods. By design, it samples uniformly from an estimated Pareto set using their upper confidence estimates. This tends to stabilize the allocation as the estimates can converge nicely, but the set itself may also contain suboptimal arms. By doing ${ \mathrm { s o } } ,$ it also considers the fairness, though only in part, which explains why it is not the worst. Diferently, Width-guided UCB provides a more extreme example. It aims to aggressively select one arm that wins on an objective, en route to minimizing Pareto regret by design. The price of such aggressiveness is that it mostly fails to balance the pulls among all Pareto-optimal arms. Such price only gets higher as the horizon increases. On Bernoulli rewards with $\eta _ { \mu } = 0 . 1$ , its mean unfairness increases from 0.2208 at $T = 5 { , } 0 0 0$ to 1.6001 at $T = 5 0 { , } 0 0 0$ Annealing-Pareto is among the worst-performing. It incurs much larger unfairness on the Bernoulli instances, while its performance improves substantially on the Gaussian instances as the separation increases. Interestingly, its performance mostly downgrades as the time horizon increases, consistent with the Width-guided UCB but as opposed to the majority of the methods. This certainly makes it less favorable, however, with a clear reason. The algorithm aims to minimize the Shannon entropy based on their measure; as introduced earlier, this is flawed by definition.

Table 2: Mean and variance of terminal KL unfairness on Gaussian rewards for $T \in \{ 5 , 0 0 0 , 5 0 , 0 0 0 \}$ The smallest mean and variance for each separation margin and horizon are in bold.
<table><tr><td rowspan="2"></td><td colspan="2"> $\eta _ { \mu } = 0 . 0 0 5$ </td><td colspan="2"> $\eta _ { \mu } = 0 . 0 2$ </td><td colspan="2"> $\eta _ { \mu } = 0 . 1$ </td></tr><tr><td>Mean</td><td>Variance</td><td>Mean</td><td>Variance</td><td>Mean</td><td>Variance</td></tr><tr><td colspan="7">Horizon T = 5, 000</td></tr><tr><td>ATT (mean+SD BO)</td><td>0.3188</td><td>0.0098</td><td>0.0516</td><td>0.0123</td><td>0.0018</td><td> $1 . 9 4 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>ATT (log BO)</td><td>0.3468</td><td>0.0078</td><td>0.0516</td><td>0.0123</td><td>0.0026</td><td> $1 . 1 2 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>EG-ATT (mean+SD BO)</td><td>0.2826</td><td>0.0196</td><td>0.0390</td><td>0.0020</td><td>0.0072</td><td> $\mathbf { < 1 0 ^ { - 1 5 } }$ </td></tr><tr><td>EG-ATT (log BO)</td><td>0.4923</td><td> $\mathbf { 4 . 8 9 \times 1 0 ^ { - 5 } }$ </td><td>0.2196</td><td> $\mathbf { 6 . 0 7 \times 1 0 ^ { - 1 5 } }$ </td><td>0.0072</td><td> $2 . 1 9 \times 1 0 ^ { - 9 }$ </td></tr><tr><td>Pareto-UCB1</td><td>0.4698</td><td>0.00024</td><td>0.3860</td><td>0.00027</td><td>0.1462</td><td> $3 . 9 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>PTS</td><td>0.3175</td><td>0.0321</td><td>0.0513</td><td>0.0081</td><td>0.0035</td><td> $1 . 7 5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Width-guided UCB</td><td>0.4859</td><td>0.00023</td><td>0.4351</td><td>0.00023</td><td>0.3289</td><td>0.00014</td></tr><tr><td>Pareto-KG (adapted)</td><td>0.3118</td><td>0.0243</td><td>0.0757</td><td>0.0086</td><td>0.0176</td><td>0.0255</td></tr><tr><td>Annealing-Pareto</td><td>0.3928</td><td>0.0380</td><td>0.1352</td><td>0.0147</td><td>0.0443</td><td> $9 . 7 2 \times 1 0 ^ { - 6 }$ </td></tr><tr><td colspan="7">Horizon T = 50, 000</td></tr><tr><td>ATT (mean+SD BO)</td><td>0.0713</td><td>0.0168</td><td>0.0066</td><td> $7 . 4 4 \times 1 0 ^ { - 6 }$ </td><td>0.00030</td><td> $1 . 4 7 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>ATT (log BO)</td><td>0.0946</td><td>0.0017</td><td>0.0066</td><td> $7 . 4 4 \times 1 0 ^ { - 6 }$ </td><td>0.00018</td><td> $2 . 0 3 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>EG-ATT (mean+SD BO)</td><td>0.0699</td><td>0.0046</td><td>0.0046</td><td> $3 . 2 2 \times 1 0 ^ { - 6 }$ </td><td>0.0017</td><td> $\mathbf { < 1 0 ^ { - 1 5 } }$ </td></tr><tr><td>EG-ATT (log BO)</td><td>0.0603</td><td>0.0055</td><td>0.0069</td><td> $\mathbf { 1 . 3 4 \times 1 0 ^ { - 6 } }$ </td><td>0.0013</td><td> $6 . 6 8 \times 1 0 ^ { - 1 2 }$ </td></tr><tr><td>Pareto-UCB1</td><td>0.4194</td><td>0.00020</td><td>0.2375</td><td> $8 . 1 1 \times 1 0 ^ { - 5 }$ </td><td>0.0382</td><td> $1 . 9 5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>PTS</td><td>0.0919</td><td>0.0273</td><td>0.0070</td><td> $8 . 6 0 \times 1 0 ^ { - 6 }$ </td><td>0.00052</td><td> $3 . 5 0 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>Width-guided UCB</td><td>0.4538</td><td>0.00015</td><td>0.3667</td><td>0.00011</td><td>0.7097</td><td>0.00027</td></tr><tr><td>Pareto-KG (adapted)</td><td>0.1254</td><td>0.0060</td><td>0.0139</td><td>0.0071</td><td>0.0099</td><td>0.0216</td></tr><tr><td>Annealing-Pareto</td><td>0.8993</td><td>0.2025</td><td>0.1864</td><td>0.1639</td><td>0.0044</td><td> $8 . 4 8 \times 1 0 ^ { - 8 }$ </td></tr></table>

## 7 Conclusion and AI Use Statement

Stochastic multi-objective bandits have shown promise in online advertising where a single objective can be far from suficient for decision makers. With the progress of the community, we have a decent understanding of how to identify and select Pareto optimal arms. However, it seems that we know much less about how to select among these optimal arms in a balanced manner. One barrier is that there seems to be a lack of consensus on a proper measure for it, and another barrier is that the set of Pareto optimal arms is typically unknown. Surmounting these barriers requires more careful control of the sampling trajectory while learning the Pareto set. We take steps herein toward this goal. We first introduce a measure for unfairness based on the KL divergence from the ideal measure, the uniform distribution over the Pareto optimal arms, to the actual measure, the empirical measure of the sampling trajectory induced by a policy. It resembles several existing measures or corrects their limitations while remaining well defined in any circumstance. An exact decomposition of this measure further provides pathways for method design. We propose an adaptive top-two sampling method, which defines a gain criterion for each arm at each time step based on a soft target for the true fairness gain, and randomly selects between the top two arms to promote exploration. The upper bound of the method exhibits a satisfactory rate for the expected unfairness, O(log T/T), whereas the variance seems to pose computational challenges. Hence, we further propose a variant aiming to reduce variance, which adds more exploration at earlier stages to avoid misidentified Pareto membership and thus fluctuations in fairness. Numerical experiments also help interpret the theory, show the advantages of the proposed methods, and ofer practical insights into existing methods.

Generative AI tools, particularly GPT-6, have assisted the authors with coding, drafting the proofs of the mathematical claims, manuscript polishing, and reviewing the content. The tools did not participate in the ideation, conceptualization, or formalization of the paper. They also helped with the rigorous steps in the method design after the authors sketched the rough picture of the method. The authors have independently revised the content and take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## References

Heesang Ann and Min-hwan Oh. Blessings of multiple good arms in multi-objective linear bandits. arXiv preprint arXiv:2602.12901, 2026.

C¸ a˘gın Ararat and Cem Tekin. Vector optimization with stochastic bandit feedback. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 2165–2190. PMLR, 2023. URL https:// proceedings.mlr.press/v206/ararat23a.html.

Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2):235–256, 2002a.

Peter Auer, Nicol\`o Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. The nonstochastic multiarmed bandit problem. SIAM Journal on Computing, 32(1):48–77, 2002b. doi: 10.1137/ S0097539701398375.

Peter Auer, Chao-Kai Chiang, Ronald Ortner, and Madalina Drugan. Pareto front identification from stochastic bandit feedback. In Proceedings of the 19th International Conference on Artificial Intelligence and Statistics, volume 51 of Proceedings of Machine Learning Research, pages 939–947. PMLR, 2016. URL https://proceedings.mlr.press/v51/auer16.html.

S´ebastien Bubeck and Nicolo Cesa-Bianchi. Regret analysis of stochastic and nonstochastic multiarmed bandit problems. Foundations and Trends® in Machine Learning, 5(1):1–122, 2012.

R´obert Busa-Fekete, Bal´azs Sz¨or´enyi, Paul Weng, and Shie Mannor. Multi-objective bandits: Optimizing the generalized gini index. In International conference on machine learning, pages 625–634. PMLR, 2017.

Madalina M Drugan and Ann Nowe. Designing multi-objective multi-armed bandits algorithms: A study. In The 2013 international joint conference on neural networks (IJCNN), pages 1–8. IEEE, 2013.

John Duchi and Saminul Haque. An information-theoretic lower bound in time-uniform estimation. In The Thirty Seventh Annual Conference on Learning Theory, pages 1486–1500. PMLR, 2024.

Aur´elien Garivier, Wouter M Koolen, et al. Sequential learning of the pareto front for multi-objective bandits. In International Conference on Artificial Intelligence and Statistics, pages 3583–3591. PMLR, 2024.

Changkun Guan and Mengfan Xu. Are stochastic multi-objective bandits harder than single-objective bandits? arXiv preprint arXiv:2604.07096, 2026.

Martin Jaggi. Revisiting frank-wolfe: Projection-free sparse convex optimization. In International conference on machine learning, pages 427–435. PMLR, 2013.

Emilie Kaufmann, Olivier Capp´e, and Aur´elien Garivier. On the complexity of best-arm identification in multi-armed bandit models. The Journal of Machine Learning Research, 17(1):1–42, 2016.

Cyrille Kone, Emilie Kaufmann, and Laura Richert. Bandit pareto set identification: the fixed budget setting. In International Conference on Artificial Intelligence and Statistics, pages 2548–2556. PMLR, 2024.

Simon Lacoste-Julien and Martin Jaggi. On the global linear convergence of frank-wolfe optimization variants. Advances in neural information processing systems, 28, 2015.

T. L. Lai and Herbert Robbins. Asymptotically eficient adaptive allocation rules. Advances in Applied Mathematics, 6(1):4–22, 1985. doi: 10.1016/0196-8858(85)90002-8. URL https: //doi.org/10.1016/0196-8858(85)90002-8.

Tor Lattimore and Csaba Szepesv´ari. Bandit algorithms. Cambridge University, 355, 2019.

Shiyin Lu, Guanghui Wang, Yao Hu, and Lijun Zhang. Multi-objective generalized linear bandits. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, pages 3080–3086. International Joint Conferences on Artificial Intelligence Organization, 2019. doi: 10.24963/ijcai.2019/427.

Rishabh Mehrotra, Niannan Xue, and Mounia Lalmas. Bandit based optimization of multiple objectives on a music streaming platform. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 3224–3233, 2020.

Chao Qin and Wei You. Dual-directed algorithm design for eficient pure exploration. Operations Research, 74(2):1104–1125, 2026.

Nathan Ross. Fundamentals of stein’s method. 2011.

Daniel Russo. Simple bayesian algorithms for best-arm identification. Operations Research, 68(6): 1625–1647, 2020.

James P Sethna et al. Statistical mechanics: entropy, order parameters, and complexity, volume 14. Oxford University Press, Oxford, 2006.

David Simchi-Levi and Chonghuan Wang. Multi-armed bandit experimental design: Online decisionmaking and adaptive inference. In International Conference on Artificial Intelligence and Statistics, pages 3086–3097. PMLR, 2023.

Ashudeep Singh and Thorsten Joachims. Fairness of exposure in rankings. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 2219–2228, 2018.

Eralp Turgay, Doruk Oner, and Cem Tekin. Multi-objective contextual bandit problem with similarity<sup>¨</sup> information. In Proceedings of the 21st International Conference on Artificial Intelligence and Statistics, volume 84 of Proceedings of Machine Learning Research, pages 1673–1681. PMLR, 2018. URL https://proceedings.mlr.press/v84/turgay18a.html.

Lequn Wang, Yiwei Bai, Wen Sun, and Thorsten Joachims. Fairness of exposure in stochastic bandits. In International Conference on Machine Learning, pages 10686–10696. PMLR, 2021.

Mengfan Xu and Diego Klabjan. Pareto regret analyses in multi-objective multi-armed bandit. In International Conference on Machine Learning, pages 38499–38517. PMLR, 2023.

Saba Q Yahyaa, Madalina M Drugan, and Bernard Manderick. Annealing-pareto multi-objective multi-armed bandit algorithm. In 2014 IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning (ADPRL), pages 1–8. IEEE, 2014a.

Saba Q Yahyaa, Madalina M Drugan, and Bernard Manderick. Knowledge gradient for multiobjective multi-armed bandit algorithms. In International Conference on Agents and Artificial Intelligence, volume 2, pages 74–83. SCITEPRESS, 2014b.

The appendix proceeds as follows. Section A presents the proofs of our theoretical results. Since fairness essentially depends on the sampling process, it is in fact more convenient to start with the sampling process, such as arm pulls and the imbalance potential. Hence, we first present the proofs for bounds on arm pulls and imbalance, followed by those for the bounds on expected unfairness and variance. This is in the reverse order of the results presented in the main body, and we believe this exposition is natural, as the two orders serve diferent purposes. We then derive the proof for Pareto regret. We then prove the information-theoretic lower bounds, including the asymptotic and finite-time bounds under positive Pareto separation and the lower bound at zero separation. Finally, we establish the guarantees for EG-ATT designed for variance reduction. At the same time, to enhance the reproducibility of the results, Section B provides the reward instances, baseline implementations, and tuning details for the numerical experiments.

## A Proofs

## A.1 Proof for Arm Pulls and Imbalance

Proof of Theorem 4.2. We first control inaccurate rounds, then use this control to bound dominatedarm pulls and the imbalance potential.

Step 1: exploration and estimation errors. For this proof, set

$$
\alpha = \frac { ( 6 4 + 3 2 s ) \sigma ^ { 2 } } { \eta _ { \mu } ^ { 2 } } , \qquad t _ { * } = \biggl \lceil \frac { \sqrt { T _ { 0 } } } { 2 } \biggr \rceil .
$$

Let $\mathcal { E } _ { t }$ be the event that

$$
\operatorname* { m i n } _ { i } { N _ { i } ( t ) } \ge \alpha \log ( t + 1 ) , \qquad \operatorname* { m a x } _ { i , r } | \widetilde { \mu } _ { t , i } ^ { ( r ) } - \mu _ { i } ^ { ( r ) } | \le \eta _ { \mu } / 4 .
$$

We will show that

$$
2 \sum _ { t = 2 K } ^ { \infty } ( t + 1 ) \mathbb { P } _ { \mu } ( \mathcal { E } _ { t } ^ { c } ) \leq T _ { 0 } .\tag{26}
$$

When $\widehat { \Delta } _ { t , i } > 0$ , changing only arm i’s projected mean to the upper corner of M makes that arm Pareto optimal. Hence

$$
Z _ { t , i } \leq \frac { N _ { i } ( t ) B _ { \mathcal { M } } ^ { 2 } } { 2 \sigma ^ { 2 } } .
$$

Using $\widehat { \Delta } _ { t , i } \leq B _ { \mathcal { M } }$ and $\sqrt { x ^ { 2 } + x } \le x + 1 / 2$ gives

$$
\frac { R t , i \sqrt { 1 + Z _ { t , i } } } { c _ { \tau } } \leq \frac { B _ { \mathcal { M } } ^ { 2 } N _ { i } ( t ) } { \sqrt { 2 } \sigma ^ { 2 } c _ { \tau } } + \frac { 1 } { \sqrt { 2 } c _ { \tau } } .\tag{27}
$$

When $R _ { t , i } = 0$ , the exponent is zero by convention, and the same bound holds.

Our choice of α satisfies

$$
\frac { \alpha B _ { \mathcal M } ^ { 2 } } { \sqrt { 2 } \sigma ^ { 2 } c _ { \tau } } = ( 1 - s ) ( 1 + s / 2 ) \leq 1 - s / 2 , \qquad \frac { \alpha \eta _ { \mu } ^ { 2 } } { 3 2 \sigma ^ { 2 } } = 2 + s .
$$

Also, $\eta _ { \mu } \leq B _ { \mathcal { M } }$ and equation 11 imply $c _ { \tau } > 3 2 { \sqrt { 2 } } .$ , so exp $\{ 1 / ( \sqrt { 2 } c _ { \tau } ) \} < 2$

Fix $t \geq t _ { * }$ and suppose that $N _ { a } ( t ) < \alpha \log ( t + 1 )$ for some arm a. For every earlier round ℓ, monotonicity of the counts and equation 27 imply

$$
G _ { \ell , a } \geq \frac { ( t + 1 ) ^ { - 1 + s / 2 } } { 2 ( 1 + \alpha \log ( { t + 1 } ) ) } .
$$

Since the leader maximizes the gain and $G _ { \ell , b _ { \ell } } \leq 1 / N _ { b _ { \ell } } ( \ell )$

$$
N _ { b _ { \ell } } ( \ell ) \leq 2 ( 1 + \alpha \log ( t + 1 ) ) ( t + 1 ) ^ { 1 - s / 2 } .
$$

Thus, over $\ell = \lceil t / 2 \rceil , \ldots , t - 1$ , the total number of pulls made as the leader is at most

$$
K \left[ 2 ( 1 + \alpha \log ( t + 1 ) ) ( t + 1 ) ^ { 1 - s / 2 } + 2 \right] \leq { \frac { t } { 1 6 } } .
$$

For the last inequality, use $\alpha \leq 9 6 \sigma ^ { 2 } / \eta _ { \mu } ^ { 2 } , \log ( t + 1 ) \leq 4 ( t + 1 ) ^ { s / 4 } / s$ , and

$$
t \geq t _ { * } \geq \left[ 1 2 8 K \left( 1 + \frac { 3 8 4 \sigma ^ { 2 } } { s \eta _ { \mu } ^ { 2 } } \right) \right] ^ { 4 / s } ,
$$

which follows from equation 12.

At every round, the leader is selected with probability at least 0.45, and therefore at least $2 / 5$ Using independent uniform random variables to implement these selections, the number of leader pulls over these rounds is bounded below by a binomial random variable with parameters $\lfloor t / 2 \rfloor$ and $2 / 5$ . A Chernof bound consequently gives

$$
\mathbb { P } _ { \mu } \left\{ \operatorname* { m i n } _ { i } N _ { i } ( t ) < \alpha \log ( t + 1 ) \right\} \leq e ^ { - t / 6 4 } , \qquad t \geq t _ { * } .
$$

Projection onto $\mathcal { M }$ cannot increase the error relative to the true mean. Applying the sub-Gaussian bound to the reward tapes and summing over all sample sizes $n \geq \lceil \alpha \log ( t + 1 ) \rceil$ yields

$$
\begin{array} { l } { { \displaystyle { \mathbb P } _ { \mu } \big ( { \mathcal E } _ { t } ^ { c } \big ) \leq e ^ { - t / 6 4 } + \frac { 2 K d } { 1 - e ^ { - \eta _ { \mu } ^ { 2 } / ( 3 2 \sigma ^ { 2 } ) } } ( t + 1 ) ^ { - 2 - s } } } \\ { { \qquad \leq e ^ { - t / 6 4 } + 2 K d \left( 1 + \frac { 3 2 \sigma ^ { 2 } } { \eta _ { \mu } ^ { 2 } } \right) ( t + 1 ) ^ { - 2 - s } , \qquad t \geq t _ { * } . } } \end{array}
$$

Here we used $( 1 - e ^ { - x } ) ^ { - 1 } \leq 1 + x ^ { - 1 }$ for $x > 0$ . For earlier rounds, use the trivial probability bound of one. Therefore,

$$
\begin{array} { r l r } {  { 2 \sum _ { t = 2 K } ^ { \infty } ( t + 1 ) \mathbb { P } _ { \mu } ( \mathcal { E } _ { t } ^ { c } ) \leq t _ { * } \big ( t _ { * } + 1 \big ) + 8 4 5 0 + \frac { 4 K d } { s } ( 1 + \frac { 3 2 \sigma ^ { 2 } } { \eta _ { \mu } ^ { 2 } } ) } } \\ & { } & \\ & { } & { \leq T _ { 0 } . } \end{array}
$$

Indeed, the first term is at most $T _ { 0 } / 2$ , and each of the remaining terms is at most $T _ { 0 } / 4$ , by the explicit definition of $T _ { 0 }$ . The numerical bound 8450 follows from

$$
2 \sum _ { t = 0 } ^ { \infty } ( t + 1 ) e ^ { - t / 6 4 } = { \frac { 2 } { ( 1 - e ^ { - 1 / 6 4 } ) ^ { 2 } } } \leq 2 ( 1 + 6 4 ) ^ { 2 } .
$$

This proves equation 26.

Let

$$
M _ { T } = \sum _ { t = 2 K } ^ { T - 1 } \mathbf { 1 } \{ \mathcal { E } _ { t } ^ { c } \}
$$

count inaccurate rounds. Expanding the square and using $\mathbb { P } _ { \mu } ( \mathcal { E } _ { u } ^ { c } \cap \mathcal { E } _ { t } ^ { c } ) \le \mathbb { P } _ { \mu } ( \mathcal { E } _ { t } ^ { c } )$ for $u \leq t ,$ , we obtain

$$
\mathbb { E } _ { \mu } M _ { T } ^ { 2 } \leq 2 \sum _ { t = 2 K } ^ { T - 1 } ( t + 1 ) \mathbb { P } _ { \mu } ( \mathcal { E } _ { t } ^ { c } ) \leq T _ { 0 } .\tag{28}
$$

Step 2: dominated-arm pulls. On $\mathcal { E } _ { t }$ , the empirical Pareto set equals $\mathcal { A } ^ { \star }$ , and every true Pareto arm has gain log $\left( 1 + 1 / N _ { i } ( t ) \right)$ .

For each dominated arm $i ,$ choose a Pareto optimal arm $p _ { i }$ such that

$$
\mu _ { p _ { i } } ^ { ( r ) } - \mu _ { i } ^ { ( r ) } \geq \eta _ { \mu } \qquad \mathrm { f o r ~ e v e r y ~ } r .
$$

Such an arm exists by the definition of $\eta _ { \mu } ^ { \mathrm { D } }$ : if a dominating arm is not Pareto optimal, replacing it by an arm that dominates it cannot decrease any coordinatewise advantage.

On $\mathcal { E } _ { t }$ , each empirical advantage of $p _ { i }$ over i is at least $\eta _ { \mu } / 2$ . Consequently,

$$
R _ { t , i } \geq \frac { \eta _ { \mu } \sqrt { N _ { i } ( t ) } } { 2 \sigma } , \qquad Z _ { t , i } \geq \frac { \eta _ { \mu } ^ { 2 } } { 8 \sigma ^ { 2 } } \frac { N _ { i } ( t ) N _ { p _ { i } } ( t ) } { N _ { i } ( t ) + N _ { p _ { i } } ( t ) } .
$$

For the second inequality, any alternative that makes arm i Pareto optimal must break its domination by $p _ { i } .$ . If the two mean vectors change by amounts u and v in the infinity norm, this requires $u + v \ge \eta _ { \mu } / 2$ . Minimizing the corresponding weighted sum of squares gives the stated bound. It follows that

$$
\frac { R _ { t , i } \sqrt { 1 + Z _ { t , i } } } { c _ { \tau } } \geq \frac { \eta _ { \mu } ^ { 2 } } { 4 \sqrt { 2 } c _ { \tau } \sigma ^ { 2 } } N _ { i } ( t ) \sqrt { \frac { N _ { p _ { i } } ( t ) } { N _ { i } ( t ) + N _ { p _ { i } } ( t ) } } .
$$

If a dominated arm is selected on $\mathcal { E } _ { t } ,$ its gain is at least $1 / ( t + 1 )$ . When $m \geq 2$ , two true Pareto arms have gains at least this large. When $m = 1$ , the singleton rule selects the leader, whose gain is at least that of the true Pareto arm. Since $\log ( 1 + 1 / N _ { i } ( t ) ) \leq 1$ , the exponent in its gain is therefore at most $\log ( t + 1 )$ .

Using $N _ { p _ { i } } ( t ) \geq \alpha \log ( t + 1 )$ and considering $N _ { i } ( t ) \leq N _ { p _ { i } } ( t )$ and $N _ { i } ( t ) > N _ { p _ { i } } ( t )$ separately gives

$$
\begin{array} { l } { { N _ { i } ( t ) \leq \operatorname* { m a x } \left\{ \displaystyle \frac { 8 c _ { \tau } \sigma ^ { 2 } } { \eta _ { \mu } ^ { 2 } } , \frac { 6 4 c _ { \tau } ^ { 2 } \sigma ^ { 4 } } { \alpha \eta _ { \mu } ^ { 4 } } \right\} \log ( t + 1 ) } } \\ { { \leq \displaystyle \frac { c _ { \tau } ^ { 2 } \sigma ^ { 2 } } { \eta _ { \mu } ^ { 2 } } \log ( t + 1 ) . } } \end{array}
$$

The last inequality uses $\alpha \geq 6 4 \sigma ^ { 2 } / \eta _ { \mu } ^ { 2 }$ and $c _ { \tau } > 8$

Accounting for initialization, the last pull on an accurate round, and pulls on inaccurate rounds yields the pathwise bound

$$
B _ { T } \leq \left( K - m \right) \left( 3 + \frac { c _ { \tau } ^ { 2 } \sigma ^ { 2 } } { \eta _ { \mu } ^ { 2 } } \log ( T + 1 ) \right) + M _ { T } .\tag{29}
$$

Taking the $L ^ { 2 }$ norm and applying equation 28 proves equation 15.

Step 3: imbalance among Pareto optimal arms. When $m = 1 , \Phi _ { T } = 0$ . Suppose $m \geq 2$ and write

$$
H _ { t } = \sum _ { i \in \mathcal { A } ^ { \star } } N _ { i } ( t ) , \qquad x _ { i } ( t ) = N _ { i } ( t ) - H _ { t } / m .
$$

A pull of Pareto arm i changes the potential by

$$
\Phi _ { t + 1 } - \Phi _ { t } = 2 x _ { i } ( t ) + 1 - \frac { 1 } { m } ,
$$

whereas a dominated-arm pull leaves it unchanged.

On $\mathcal { E } _ { t } ,$ , if exactly one candidate is Pareto optimal, it is a least-sampled true Pareto arm and has a nonpositive centered count. If both candidates are Pareto optimal, they are the two least-sampled true Pareto arms. Their centered counts sum to at most zero, and the smaller-count candidate is selected with probability at least one half. Thus,

$$
\mathbb { E } _ { \mu } [ \Phi _ { t + 1 } - \Phi _ { t } \mid { \mathcal { F } } _ { t } ] \leq 1 \qquad { \mathrm { o n ~ } } { \mathcal { E } } _ { t } .
$$

On any round, $| \Phi _ { t + 1 } - \Phi _ { t } | \leq 2 t + 1$ . Since $\Phi _ { 2 K } = 0$ , equation 26 gives

$$
\mathbb { E } _ { \mu } \Phi _ { T } \le T + \sum _ { t = 2 K } ^ { T - 1 } ( 2 t + 1 ) \mathbb { P } _ { \mu } ( \mathcal { E } _ { t } ^ { c } ) \le T + T _ { 0 } .
$$

Finally,

$$
0 \le \Phi _ { T } = \sum _ { i \in \mathcal { A } ^ { \star } } N _ { i } ( T ) ^ { 2 } - \frac { H _ { T } ^ { 2 } } m \le H _ { T } ^ { 2 } \le T ^ { 2 } .
$$

Therefore,

$$
\mathbb { E } _ { \mu } \Phi _ { T } ^ { 2 } \le T ^ { 2 } \mathbb { E } _ { \mu } \Phi _ { T } \le T ^ { 2 } ( T + T _ { 0 } ) ,
$$

which completes the proof.

□

## A.2 Proof for the Expected Fairness and Variance

Proof of Theorem $4 . 1 .$ Fix $T \geq T _ { 0 }$ and let $n = T - 2 K$ . The definition of $T _ { 0 }$ ensures $T \geq 1 2 K$ , so $n \ge 5 T / 6$ . It also gives

$$
( K - m ) \left( 3 + \frac { c _ { \tau } ^ { 2 } \sigma ^ { 2 } } { \eta _ { \mu } ^ { 2 } } \log ( T + 1 ) \right) \leq \frac { T } { 1 6 } \leq \frac { n } { 8 } ,
$$

using $\log ( T + 1 ) \leq { \sqrt { T } }$ . Hence equation 29 implies

$$
B _ { T } \leq n / 8 + M _ { T } .
$$

We first verify that every Pareto optimal arm receives a positive fraction of the observations, except on an event whose contribution to the fairness moments is small.

Suppose $m \geq 2 .$ . Let $J _ { t }$ indicate that $\mathcal { E } _ { t }$ holds and at least one candidate is a true Pareto arm. Let $I _ { t }$ indicate that the selected arm is a currently least-sampled true Pareto arm. On a round counted by $J _ { t } ,$ , the probability of such a selection is at least 0.30. Indeed, when both candidates are Pareto

optimal, the leader is least sampled and is selected with probability at least 0.70; when exactly one is Pareto optimal, it is least sampled and is selected with probability 0.30 or 0.70. Therefore,

$$
\mathbb { E } _ { \mu } [ I _ { t } \mid \mathcal { F } _ { t } ] \geq 0 . 3 0 J _ { t } .
$$

Moreover,

$$
\sum _ { t = 2 K } ^ { T - 1 } J _ { t } \geq n - B _ { T } - M _ { T } \geq \frac { 7 n } { 8 } - 2 M _ { T } .
$$

On $\lbrace M _ { T } \leq n / 1 6 \rbrace$ , this sum is at least $3 n / 4$ . By equation 28, Markov’s inequality, and Azuma’s inequality,

$$
\mathbb { P } _ { \mu } \left\{ \sum _ { t = 2 K } ^ { T - 1 } I _ { t } < \frac { 3 n } { 2 0 } \right\} \leq \frac { 2 5 6 T _ { 0 } } { n ^ { 2 } } + \exp \left\{ - \frac { 9 n } { 3 2 0 0 } \right\} .
$$

Starting from two pulls per arm, at most $m$ selections of a currently least-sampled Pareto arm are needed to increase the minimum Pareto-arm count by one. Other selections cannot decrease any count. Thus,

$$
\operatorname* { m i n } _ { i \in \mathcal { A } ^ { \star } } N _ { i } ( T ) \geq 2 + \left\lfloor \frac { 1 } { m } \sum _ { t = 2 K } ^ { T - 1 } I _ { t } \right\rfloor .
$$

If the sum is at least $3 n / 2 0$ , then

$$
\operatorname* { m i n } _ { i \in \mathcal { A } ^ { \star } } N _ { i } ( T ) \geq \frac { 3 n } { 2 0 m } \geq \frac { T } { 8 m } .
$$

When $m = 1$ , the same count bound follows directly on $\lbrace M _ { T } \leq n / 1 6 \rbrace$ from

$$
N _ { p } ( T ) = T - B _ { T } \geq T - { \frac { n } { 8 } } - { \frac { n } { 1 6 } } \geq { \frac { 1 3 T } { 1 6 } } ,
$$

where $p$ is the unique Pareto optimal arm.

Consequently, for

$$
\mathcal { C } _ { T } = \left\{ \operatorname* { m i n } _ { i \in \mathcal { A } ^ { \star } } N _ { i } ( T ) \geq \frac { T } { 8 m } \right\} ,
$$

we have

$$
\mathbb { P } _ { \mu } ( \mathcal { C } _ { T } ^ { c } ) \leq \frac { 2 5 6 T _ { 0 } } { n ^ { 2 } } + \exp \left\{ - \frac { 9 n } { 3 2 0 0 } \right\} \leq \frac { 5 1 2 T _ { 0 } } { T ^ { 2 } } .\tag{30}
$$

For the final inequality, use $n \ge 5 T / 6$ and

$$
e ^ { - 9 n / 3 2 0 0 } \leq e ^ { - 3 T / 1 2 8 0 } \leq \frac { 2 ( 1 2 8 0 / 3 ) ^ { 2 } } { T ^ { 2 } } \leq \frac { T _ { 0 } } { T ^ { 2 } } ,
$$

where the last inequality follows from the explicit size of $T _ { 0 }$

On $\mathcal { C } _ { T }$ , we have $H _ { T } = T - B _ { T } \geq T / 8$ . The exact decomposition and its chi-square bound give

$$
\begin{array} { l } { \displaystyle \mathcal { F } _ { T } \leq \frac { B _ { T } } { H _ { T } } + \frac { 1 } { H _ { T } } \sum _ { i \in \mathcal { A } ^ { \star } } \frac { ( N _ { i } ( T ) - H _ { T } / m ) ^ { 2 } } { N _ { i } ( T ) } } \\ { \displaystyle \leq \frac { 8 B _ { T } } { T } + \frac { 6 4 m \Phi _ { T } } { T ^ { 2 } } . } \end{array}
$$

On its complement, initialization gives $0 \leq \mathcal { F } _ { T } \leq \log ( T / m )$ . Splitting the expectation over these events yields

$$
\mathbb { E } _ { \mu } \mathcal { F } _ { T } \leq \frac { 8 \mathbb { E } _ { \mu } B _ { T } } { T } + \frac { 6 4 m \mathbb { E } _ { \mu } \Phi _ { T } } { T ^ { 2 } } + \frac { 5 1 2 T _ { 0 } } { T ^ { 2 } } \log { \frac { T } { m } } .
$$

Similarly, using $\operatorname { V a r } _ { \mu } ( \mathcal { F } _ { T } ) \leq \mathbb { E } _ { \mu } \mathcal { F } _ { T } ^ { 2 }$ 2

$$
\mathrm { V a r } _ { \mu } ( \mathcal { F } _ { T } ) \leq \frac { 1 2 8 \mathbb { E } _ { \mu } B _ { T } ^ { 2 } } { T ^ { 2 } } + \frac { 8 1 9 2 m ^ { 2 } \mathbb { E } _ { \mu } \Phi _ { T } ^ { 2 } } { T ^ { 4 } } + \frac { 5 1 2 T _ { 0 } } { T ^ { 2 } } \log ^ { 2 } \frac { T } { m } .
$$

Substitute Theorem 4.2 and use $T + T _ { 0 } \leq 2 T$ to obtain equation 13 and equation 14.

Finally, convexity of KL divergence in its second argument gives

$$
\begin{array} { r } { \bar { \mathcal { F } } _ { T } = D _ { \mathrm { K L } } ( \pi ^ { \star } \| \mathbb { E } _ { \mu } \widehat { \pi } _ { T } ) \leq \mathbb { E } _ { \mu } D _ { \mathrm { K L } } ( \pi ^ { \star } \| \widehat { \pi } _ { T } ) = \mathbb { E } _ { \mu } \mathcal { F } _ { T } . } \end{array}
$$

This completes the proof.

## A.3 Proof for the Pareto Regret

Proof. We retain the individual arm gaps in the proof of Theorem 4.2. Let $\mathcal { E } _ { t }$ and $M _ { T }$ be the accurate-round event and the number of inaccurate rounds defined there. In particular, $\mathbb { E } _ { \mu } M _ { T } ^ { 2 } \le T _ { 0 }$

For a dominated arm i, choose a Pareto optimal arm $p _ { i }$ attaining its domination margin:

$$
\operatorname* { m i n } _ { r \in [ d ] } \big ( \mu _ { p _ { i } } ^ { ( r ) } - \mu _ { i } ^ { ( r ) } \big ) = \Delta _ { i } ^ { \mathrm { P } } .
$$

Such a maximizing arm can be chosen in $\mathcal { A } ^ { \star }$ , since replacing a dominated maximizer by an arm that dominates it cannot decrease any coordinatewise advantage. On $\mathcal { E } _ { t }$ , every empirical advantage of $p _ { i }$ over i is at least

$$
\Delta _ { i } ^ { \mathrm { P } } - \frac { \eta _ { \mu } } { 2 } \geq \frac { \Delta _ { i } ^ { \mathrm { P } } } { 2 } .
$$

Therefore, the gain comparison in Step 2 of the arm-count proof applies with $\Delta _ { i } ^ { \mathrm { P } }$ in place of $\eta _ { \mu }$ . Whenever arm i is selected on $\mathcal { E } _ { t } ,$ it gives

$$
\begin{array} { r l } & { N _ { i } ( t ) \leq \sigma ^ { 2 } \operatorname* { m a x } \left\{ \cfrac { 8 c _ { \tau } } { ( \Delta _ { i } ^ { \mathrm { P } } ) ^ { 2 } } , \frac { c _ { \tau } ^ { 2 } \eta _ { \mu } ^ { 2 } } { ( \Delta _ { i } ^ { \mathrm { P } } ) ^ { 4 } } \right\} \log ( t + 1 ) } \\ & { \qquad \leq \cfrac { c _ { \tau } ^ { 2 } \sigma ^ { 2 } } { ( \Delta _ { i } ^ { \mathrm { P } } ) ^ { 2 } } \log ( t + 1 ) . } \end{array}
$$

The last inequality uses $\Delta _ { i } ^ { \mathrm { P } } \geq \eta _ { \mu }$ and $c _ { \tau } > 8$ , which follows from the temperature condition and $\eta _ { \mu } \leq B _ { \mathcal { M } }$

Accounting for initialization and the last pull on an accurate round, and charging each inaccurate round at most $\Delta _ { \mathrm { m a x } } ^ { \mathrm { P } }$ regret, yields

$$
R _ { T } ^ { \mathrm { { P } } } \leq 3 \sum _ { i \notin { \mathcal { A } } ^ { \star } } \Delta _ { i } ^ { \mathrm { { P } } } + c _ { \tau } ^ { 2 } \sigma ^ { 2 } \log ( T + 1 ) \sum _ { i \notin { \mathcal { A } } ^ { \star } } \frac { 1 } { \Delta _ { i } ^ { \mathrm { { P } } } } + \Delta _ { \operatorname* { m a x } } ^ { \mathrm { { P } } } M _ { T } .
$$

Taking expectations and using $\mathbb { E } _ { \mu } M _ { T } \leq \sqrt { \mathbb { E } _ { \mu } M _ { T } ^ { 2 } } \leq \sqrt { T _ { 0 } }$ proves equation 17.

Finally, $R _ { T } ^ { \mathrm { P } } \leq \Delta _ { \operatorname* { m a x } } ^ { \mathrm { P } } B _ { T }$ follows directly from the definition of regret. The exact unfairness decomposition gives $\mathcal { F } _ { T } \ge B _ { T } / T$ , proving equation 18. □

## A.4 Proofs for the Lower Bounds

Proof of Theorem 4.5. Fix a dominated arm i and an admissible alternative $Q _ { i }$ such that the resulting instance $\nu ^ { ( i ) }$ satisfies

$$
\eta _ { \nu ^ { ( i ) } } > 0 , \qquad i \in \mathcal { A } ^ { \star } ( \nu ^ { ( i ) } ) , \qquad 0 < D _ { \mathrm { K L } } ( P _ { i } \| Q _ { i } ) < \infty .
$$

For notation convenience, we write $n _ { i } = \mathbb { E } _ { \mu } N _ { i } ( T )$ and $E _ { i } = \{ N _ { i } ( T ) \leq \sqrt { T } \}$

We first establish an event bound under the alternative instance. Let $m _ { i } ^ { \prime } = | \mathcal { A } ^ { \star } ( \nu ^ { ( i ) } ) | \leq K$ . By data processing for KL divergence, on the event $E _ { i }$

$$
\begin{array} { r l } & { \mathcal { F } _ { T } \geq \mathrm { k l } \left( \displaystyle \frac { 1 } { m _ { i } ^ { \prime } } , \frac { N _ { i } ( T ) } { T } \right) } \\ & { \quad \geq \displaystyle \frac { 1 } { m _ { i } ^ { \prime } } \log \frac { T } { N _ { i } ( T ) } - \log 2 } \\ & { \quad \geq \displaystyle \frac { \log T } { 2 K } - \log 2 \geq \frac { \log T } { 4 K } , } \end{array}
$$

for all suficiently large horizons. Here, the second inequality follows from the fact that binary entropy is at most log 2. Markov’s inequality therefore gives

$$
\mathbb { P } _ { \nu ^ { ( i ) } } ( E _ { i } ) \leq \frac { 4 K } { \log T } \mathbb { E } _ { \nu ^ { ( i ) } } \mathcal { F } _ { T } .
$$

For every fixed $\zeta \in ( 0 , 1 )$ , uniform fairness implies

$$
\mathbb { E } _ { \nu ^ { ( i ) } } \mathscr { F } _ { T } = o ( T ^ { - 1 + \zeta } ) .
$$

Consequently,

$$
\mathbb { P } _ { \nu ^ { ( i ) } } ( E _ { i } ) \leq T ^ { - 1 + \zeta }
$$

for all suficiently large T.

We next apply the change-of-measure argument explicitly.

Since arm i is dominated under $\mu$ and the policy is uniformly fair,

$$
n _ { i } \leq T \mathbb { E } _ { \mu } \mathcal { F } _ { T } = o ( T ^ { 1 / 4 } ) .
$$

By Markov’s inequality,

$$
\mathbb { P } _ { \mu } ( E _ { i } ) \geq 1 - \frac { n _ { i } } { \sqrt { T } } \longrightarrow 1 .
$$

Moreover, the adaptive KL chain rule gives

$$
D _ { \mathrm { K L } } ( \mathbb { P } _ { \mu } ^ { T } \Vert \mathbb { P } _ { \nu ^ { ( i ) } } ^ { T } ) = n _ { i } D _ { \mathrm { K L } } ( P _ { i } \Vert Q _ { i } ) < \infty ,
$$

which implies $\mathbb { P } _ { \mu } ^ { T } \ll \mathbb { P } _ { \nu ^ { ( i ) } } ^ { T }$ . Therefore, $\mathbb { P } _ { \nu ^ { ( i ) } } ( E _ { i } ) > 0$ for all suficiently large $T .$

When the alternative event has positive probability, i.e., $\mathbb { P } _ { \nu ^ { ( i ) } } ( E _ { i } ) > 0$ , the adaptive KL chain rule and data processing give

$$
\begin{array} { r l } & { n _ { i } D _ { \mathrm { K L } } ( P _ { i } \| Q _ { i } ) \geq \mathrm { k l } \big ( \mathbb { P } _ { \mu } ( E _ { i } ) , \mathbb { P } _ { \nu ^ { ( i ) } } ( E _ { i } ) \big ) } \\ & { \qquad \geq \mathbb { P } _ { \mu } ( E _ { i } ) \log \frac { 1 } { \mathbb { P } _ { \nu ^ { ( i ) } } ( E _ { i } ) } - \log 2 . } \end{array}
$$

The second inequality again uses the binary entropy bound. Hence, we obtain, for all suficiently large horizons,

$$
n _ { i } D _ { \mathrm { K L } } ( P _ { i } \| Q _ { i } ) \geq \left( 1 - \frac { n _ { i } } { \sqrt { T } } \right) ( 1 - \zeta ) \log T - \log 2 .
$$

Thus,

$$
\mathbb { E } _ { \mu } N _ { i } ( T ) \geq \frac { ( 1 - \zeta ) \log T - \log 2 } { D _ { \mathrm { K L } } ( P _ { i } \Vert Q _ { i } ) + \frac { ( 1 - \zeta ) \log T } { \sqrt { T } } } .
$$

Dividing by log T and taking lower limits yields

$$
\operatorname* { l i m } _ { T \to \infty } \operatorname* { l i m } _ { \log T } \geq \frac { 1 - \zeta } { D _ { \mathrm { K L } } ( P _ { i } \| Q _ { i } ) } .
$$

Letting $\zeta \downarrow 0$ and taking the supremum over all admissible finite-KL alternatives gives

$$
\operatorname* { l i m } _ { T \to \infty } \frac { \mathbb { E } _ { \mu } N _ { i } ( T ) } { \log T } \geq \frac { 1 } { \kappa _ { i } ^ { \mathrm { K L } } ( \mu ) } .
$$

Every admissible alternative has strictly positive KL divergence. Indeed, if $D _ { \mathrm { K L } } ( P _ { i } \Vert Q _ { i } ) = 0$ , then $P _ { i } = Q _ { i }$ by Pinsker’s inequality. Since the distributions of all other arms remain unchanged, the resulting instance coincides with $\mu ,$ contradicting $i \in \mathcal { A } ^ { \star } ( \nu ^ { ( i ) } )$ and $i \not \in \mathcal { A } ^ { \star } ( \mu )$

Finally, the unfairness decomposition implies

$$
T \mathbb { E } _ { \mu } \mathcal { F } _ { T } \geq \sum _ { i \notin \mathcal { A } ^ { \star } ( \mu ) } \mathbb { E } _ { \mu } N _ { i } ( T ) .
$$

Since the sum is finite and its terms are nonnegative,

$$
\begin{array} { r l } { \displaystyle \operatorname* { l i m i n f } _ { T \to \infty } \frac { T \mathbb { E } _ { \mu } \mathcal { F } _ { T } } { \log T } \geq \sum _ { i \notin \mathcal { A } ^ { \star } ( \mu ) } \operatorname* { l i m i n f } _ { T \to \infty } \frac { \mathbb { E } _ { \mu } N _ { i } ( T ) } { \log T } } & { } \\ { \geq \sum _ { i \notin \mathcal { A } ^ { \star } ( \mu ) } \frac { 1 } { \kappa _ { i } ^ { \mathrm { K L } } ( \mu ) } . } \end{array}
$$

This proves equation 19.

Proof of Theorem 4.6. Again, for each arm in the chosen subset, we highlight that we use an alternative that changes only its reward distribution, makes it Pareto optimal, and satisfies

$$
\eta _ { \nu ^ { ( i ) } } > 0 , \qquad i \in \mathcal { A } ^ { \star } ( \nu ^ { ( i ) } ) , \qquad 0 < D _ { \mathrm { K L } } ( P _ { i } \| Q _ { i } ) < \infty .
$$

Let

$$
r _ { T } = \operatorname* { m a x } \left\{ \mathbb { E } _ { \mu } \mathcal { F } _ { T } , \operatorname* { m a x } _ { i \in \mathcal { T } } \mathbb { E } _ { \nu ^ { ( i ) } } \mathcal { F } _ { T } \right\} .
$$

The result is immediate if $r _ { T } = \infty$ , so suppose $r _ { T } < \infty$

For each $i \in \mathcal { Z }$ , consider the event

$$
E _ { i } = \{ N _ { i } ( T ) \leq T / ( 2 K ) \} .
$$

Since arm i is dominated under $\mu$ and $\mathcal { F } _ { T } \geq N _ { i } ( T ) / T$ , Markov’s inequality gives

$$
\mathbb { P } _ { \mu } ( E _ { i } ^ { c } ) \le \frac { 2 K } { T } \mathbb { E } _ { \mu } N _ { i } ( T ) \le 2 K r _ { T } .
$$

Under $\nu ^ { ( i ) }$ , arm i becomes Pareto optimal. Writing $m _ { i } ^ { \prime } = | { \mathcal { A } } ^ { \star } ( \nu ^ { ( i ) } ) | \leq K$ , data processing and Pinsker’s inequality give

$$
\mathcal { F } _ { T } \geq \mathrm { k l } \left( \frac { 1 } { m _ { i } ^ { \prime } } , \frac { N _ { i } ( T ) } { T } \right) \geq 2 \left( \frac { 1 } { m _ { i } ^ { \prime } } - \frac { N _ { i } ( T ) } { T } \right) ^ { 2 } .
$$

On $E _ { i } ,$ we have

$$
\frac { 1 } { m _ { i } ^ { \prime } } - \frac { N _ { i } ( T ) } { T } \geq \frac { 1 } { 2 K } ,
$$

and therefore $\mathcal { F } _ { T } \geq 1 / ( 2 K ^ { 2 } )$ . Consequently,

$$
\begin{array} { r } { \mathbb { P } _ { \nu ^ { ( i ) } } ( E _ { i } ) \leq 2 K ^ { 2 } r _ { T } . } \end{array}
$$

The instances $\mu$ and $\nu ^ { ( i ) }$ difer only in the reward distribution of arm i. The adaptive KL chain rule and the Bretagnolle–Huber inequality (Lemma 7 in Duchi and Haque [2024]) imply

$$
\begin{array} { r l } & { \mathbb { P } _ { \mu } ( E _ { i } ^ { c } ) + \mathbb { P } _ { \nu ^ { ( i ) } } ( E _ { i } ) } \\ & { \qquad \geq \displaystyle \frac { 1 } { 2 } \exp \left( - D _ { \mathrm { K L } } ( P _ { i } \| Q _ { i } ) \mathbb { E } _ { \mu } N _ { i } ( T ) \right) . } \end{array}
$$

Combining the preceding bounds yields

$$
2 K ( K + 1 ) r _ { T } \geq \frac { 1 } { 2 } \exp \left( - D _ { \mathrm { K L } } ( P _ { i } \| Q _ { i } ) \mathbb { E } _ { \mu } N _ { i } ( T ) \right) .
$$

Since $D _ { \mathrm { K L } } ( P _ { i } \Vert Q _ { i } ) < \infty$ and $\mathbb { E } _ { \mu } N _ { i } ( T ) \le T < \infty$ , the right-hand side is strictly positive, so $r _ { T } > 0$ Taking logarithms gives

$$
\mathbb { E } _ { \mu } N _ { i } ( T ) \geq \frac { 1 } { D _ { \mathrm { K L } } ( P _ { i } \| Q _ { i } ) } \log \frac { 1 } { 4 K ( K + 1 ) r _ { T } } .
$$

Summing over the arms and using the pathwise inequality

$$
T \mathcal { F } _ { T } \ge B _ { T } \ge \sum _ { i \in \mathcal { I } } N _ { i } ( T ) ,
$$

we obtain, after taking expectations,

$$
T r _ { T } \geq \sum _ { i \in { \mathcal { I } } } \mathbb { E } _ { \mu } N _ { i } ( T ) \geq S \log { \frac { 1 } { 4 K ( K + 1 ) r _ { T } } } .
$$

Let

$$
x = \frac { T r _ { T } } { S } , \qquad z = \frac { T } { 4 K ( K + 1 ) S } .
$$

The preceding inequality gives $x e ^ { x } \geq z$ . By assumption, $z \geq e$ . Moreover,

$$
( \log z - \log \log z ) e ^ { \log z - \log \log z } = z \left( 1 - { \frac { \log \log z } { \log z } } \right) \leq z .
$$

Since $x e ^ { x }$ is increasing for $x \geq 0$ , we conclude that

$$
x \geq \log z - \log \log z .
$$

Substituting the definitions of x and z proves equation 20.

Proof of Theorem $4 . 7 .$ Let $x _ { T } = \mathbb { E } _ { \nu _ { 0 } } N _ { 1 } ( T )$ and $E _ { T } = \{ N _ { 1 } ( T ) \geq T / 4 \}$ . Markov’s inequality gives

$$
\mathbb { P } _ { \nu _ { 0 } } ( E _ { T } ) \leq \frac { 4 x _ { T } } { T } .
$$

Under $\nu _ { \varepsilon }$ , both arms are Pareto optimal. On $E _ { T } ^ { c }$ , their allocation is suficiently unbalanced that

$$
\mathcal { F } _ { T } = \frac { 1 } { 2 } \log \frac { T ^ { 2 } } { 4 N _ { 1 } ( T ) ( T - N _ { 1 } ( T ) ) } \ge \kappa : = \frac { 1 } { 2 } \log \frac { 4 } { 3 } > 0 .
$$

Since $\nu _ { \varepsilon }$ has positive Pareto separation, uniform fairness implies that, for every fixed $\varepsilon \in ( 0 , 1 )$ and $\zeta \in ( 0 , 1 )$ ，

$$
\mathbb { P } _ { \nu _ { \varepsilon } } ( E _ { T } ^ { c } ) \leq \frac { \mathbb { E } _ { \nu _ { \varepsilon } } \mathcal { F } _ { T } } { \kappa } = o ( T ^ { - 1 + \zeta } ) .
$$

Let $\mathbb { P } _ { \nu } ^ { T }$ denote the law of the actions and observed rewards through time $T$ under instance ν. The policy uses the same conditional action probabilities under both instances, and only arm 1’s second reward coordinate changes. The chain rule for KL divergence therefore gives

$$
D _ { \mathrm { K L } } ( \mathbb { P } _ { \nu _ { 0 } } ^ { T } \| \mathbb { P } _ { \nu _ { \varepsilon } } ^ { T } ) = \frac { \varepsilon ^ { 2 } } { 2 \sigma ^ { 2 } } \mathbb { E } _ { \nu _ { 0 } } N _ { 1 } ( T ) = \frac { \varepsilon ^ { 2 } x _ { T } } { 2 \sigma ^ { 2 } } .
$$

By the Bretagnolle–Huber inequality (Lemma 7 in Duchi and Haque [2024]),

$$
\mathbb { P } _ { \nu _ { 0 } } ( E _ { T } ) + \mathbb { P } _ { \nu _ { \varepsilon } } ( E _ { T } ^ { c } ) \geq \frac { 1 } { 2 } \exp \left\{ - \frac { \varepsilon ^ { 2 } x _ { T } } { 2 \sigma ^ { 2 } } \right\} .
$$

Combining the preceding bounds yields

$$
\frac { 4 x _ { T } } { T } + o ( T ^ { - 1 + \zeta } ) \geq \frac { 1 } { 2 } \exp \left\{ - \frac { \varepsilon ^ { 2 } x _ { T } } { 2 \sigma ^ { 2 } } \right\}
$$

for every fixed $\varepsilon \in ( 0 , 1 )$ and $\zeta \in ( 0 , 1 )$

Suppose that $x _ { T } /$ log T did not tend to infinity. Then there would exist a finite $M > 0$ and a sequence $T _ { k } \to \infty$ such that $x _ { T _ { k } } \leq M$ log $T _ { k }$ . Choose a fixed $\varepsilon \in ( 0 , 1 )$ suficiently small that

$$
a : = \frac { \varepsilon ^ { 2 } M } { 2 \sigma ^ { 2 } } < 1 ,
$$

and then choose $\zeta \in ( 0 , 1 - a )$ . Along this sequence, the preceding inequality implies

$$
\frac { 1 } { 2 } \leq \frac { 4 M \log T _ { k } } { T _ { k } ^ { 1 - a } } + o \left( T _ { k } ^ { - ( 1 - a - \zeta ) } \right) .
$$

Both terms on the right-hand side tend to zero, a contradiction. Therefore,

$$
\frac { \mathbb { E } _ { \nu _ { 0 } } N _ { 1 } ( T ) } { \log T } \longrightarrow \infty .
$$

Finally, under $\nu _ { 0 } .$ , only arm 2 is Pareto optimal, so

$$
\mathcal { F } _ { T } = - \log \left( 1 - \frac { N _ { 1 } ( T ) } { T } \right) \geq \frac { N _ { 1 } ( T ) } { T } .
$$

Taking expectations and using the preceding limit gives

$$
\frac { T \mathbb { E } _ { \nu _ { 0 } } \mathcal { F } _ { T } } { \log T } \longrightarrow \infty ,
$$

completing the proof.

## A.5 Proof for Results in Section 5

Proof. We first show that the evidence gate is released within $O _ { \mu } ( \log T )$ rounds with high probability. We then bound the imbalance among Pareto optimal arms after release and use these bounds to control fairness.

Release of the quota. Let $\mathcal { E } _ { T }$ be the event on which, for every arm, objective, and sample count $2 \leq n \leq T$ , the empirical mean difers from its true mean by at most

$$
r ( n ) = { \sqrt { \frac { 2 \sigma ^ { 2 } \Lambda _ { T } } { n } } } .
$$

Since projection onto M does not increase the estimation error, sub-Gaussian concentration and a union bound over the arm reward tapes give

$$
\mathbb { P } _ { \mu } ( \mathcal { E } _ { T } ^ { c } ) \leq 2 K d T e ^ { - \Lambda _ { T } } = \frac { 1 } { 2 T ^ { 6 } } .
$$

The event remains valid under adaptive sampling.

For the analysis, let

$$
n _ { \mu } = \operatorname * { m a x } \left\{ 2 , \left\lceil \frac { 5 1 2 \sigma ^ { 2 } \Lambda _ { T } } { \eta _ { \mu } ^ { 2 } } \right\rceil \right\} .
$$

Whenever $n _ { \mathrm { m i n } } ( t ) \geq n _ { \mu }$ , we have $r _ { t } ^ { \operatorname* { m a x } } \leq \eta _ { \mu } / 1 6$ . On $\mathcal { E } _ { T }$ , the empirical Pareto set is then correct, and

$$
\widehat { \eta } _ { t } \geq \eta _ { \mu } - 2 r _ { t } ^ { \operatorname* { m a x } } \geq 8 r _ { t } ^ { \operatorname* { m a x } } .
$$

Thus, the evidence gate is activated once every arm has received $n _ { \mu }$ observations, unless it has already been released.

By the assumed condition $\rho _ { T } \ge n _ { \mu }$ , the quota satisfies

$$
h _ { t , T } \geq n _ { \mu } , \qquad t \geq \operatorname* { m a x } \{ 2 K , \lceil \log T \rceil \} .
$$

Before release, whenever $n _ { \mathrm { m i n } } ( t ) < n _ { \mu }$ , the quota branch therefore selects a least-sampled arm with probability $p _ { g }$ . Each such selection decreases

$$
\sum _ { i = 1 } ^ { K } [ n _ { \mu } - N _ { i } ( t ) ] _ { + }
$$

by one. Since every arm is initially sampled twice, at most $K ( n _ { \mu } - 2 )$ successful quota selections are required.

Define

$$
s = \operatorname* { m a x } \{ 2 K , \lceil \log T \rceil \} + \Biggl \lceil \frac { 2 } { p _ { g } } \bigl [ K ( n _ { \mu } - 2 ) + 8 \Lambda _ { T } \bigr ] \Biggr \rceil + 1 .
$$

Coupling the quota decisions with independent Bernoulli variables of parameter $p _ { g }$ , a Chernof bound gives

$$
\mathbb { P } \left\{ \mathrm { B i n } \left( \left\lceil \frac 2 { p _ { g } } [ K ( n _ { \mu } - 2 ) + 8 \Lambda _ { T } ] \right\rceil , p _ { g } \right) < K ( n _ { \mu } - 2 ) \right\} \le e ^ { - 2 \Lambda _ { T } } .
$$

By the definition of $C _ { \mu } ,$ we have $s \leq C _ { \mu } \Lambda _ { T }$ . Consequently, there exists an event $\mathcal { G } _ { T }$ such that the gate is released by time s on $\mathcal { G } _ { T } .$ , and

$$
\mathbb { P } _ { \mu } ( \mathcal { G } _ { T } ^ { c } ) \leq \frac { 1 } { 2 T ^ { 6 } } + e ^ { - 2 \Lambda _ { T } } \leq T ^ { - 6 } .\tag{31}
$$

Arm selection after release. Let $\tau \leq s$ be the release time on $\mathcal { G } _ { T }$ . By the gate condition, $\widehat { \eta } _ { \tau } \geq 8 r _ { \tau } ^ { \mathrm { m a x } }$ . On ${ \mathcal { E } } _ { T }$ , this certifies the true Pareto set and gives

$$
\eta _ { \mu } \geq \widehat { \eta } _ { \tau } - 2 r _ { \tau } ^ { \operatorname* { m a x } } \geq 6 r _ { \tau } ^ { \operatorname* { m a x } } .
$$

Since $r _ { t } ^ { \operatorname* { m a x } }$ is nonincreasing, the empirical Pareto set remains correct for every $t \geq \tau$ , with

$$
\widehat { \eta } _ { t } \geq \eta _ { \mu } - 2 r _ { t } ^ { \operatorname* { m a x } } \geq 4 r _ { t } ^ { \operatorname* { m a x } } .
$$

We now compare the gains in equation 8. For a dominated arm i, its empirical domination margin satisfies

$$
\begin{array} { r } { \widehat { \Delta } _ { t , i } \geq \widehat { \eta } _ { t } \geq 4 r _ { t } ^ { \operatorname* { m a x } } . } \end{array}
$$

Moreover, any alternative that changes the Pareto membership of arm i must perturb at least one mean vector involved in an empirical domination relation by at least $\widehat { \eta _ { t } } / 2$ in the infinity norm. By the definition of $Z _ { t , i }$

$$
Z _ { t , i } \geq \frac { n _ { \operatorname* { m i n } } ( t ) \widehat { \eta } _ { t } ^ { 2 } } { 8 \sigma ^ { 2 } } \geq 4 \Lambda _ { T } .
$$

Thus, recalling $R _ { t , i } = \sqrt { N _ { i } ( t ) } \widehat { \Delta } _ { t , i } / \sigma$

$$
R _ { t , i } \sqrt { 1 + Z _ { t , i } } \ge 8 \sqrt { 2 } \Lambda _ { T } .
$$

Since $c _ { \tau } = 1$ , the unnormalized gain of every dominated arm satisfies

$$
\widetilde { G } _ { t , i } \leq e ^ { - 8 \sqrt { 2 } \Lambda _ { T } } \log \left( 1 + \frac { 1 } { N _ { i } ( t ) } \right) < \frac { 1 } { T + 1 } .
$$

For a Pareto optimal arm $j ,$ however, $\widehat { \Delta } _ { t , j } = 0$ , and hence

$$
\widetilde { G } _ { t , j } = \log \left( 1 + \frac { 1 } { N _ { j } ( t ) } \right) \geq \frac { 1 } { T + 1 } .
$$

Since normalization does not change the ordering of gains, all Pareto optimal arms have larger gains than dominated arms.

When $m \geq 2 ,$ the two largest gains therefore correspond to the two least-sampled Pareto optimal arms. The adaptive probability in equation 9 assigns probability at least 0.70 to the arm with the smaller count. When $m = 1$ , the rule selects the unique Pareto optimal arm with probability one.

It follows that no dominated arm is selected after release on $\mathcal { G } _ { T }$ . In particular,

$$
B _ { T } \leq s \qquad \mathrm { o n } ~ \mathcal { G } _ { T } .
$$

Since $B _ { T } \leq T$ otherwise,

$$
\begin{array} { r } { \mathbb { E } _ { \mu } B _ { T } ^ { 4 } \le s ^ { 4 } + T ^ { 4 } \mathbb { P } _ { \mu } ( \mathcal { G } _ { T } ^ { c } ) \le C _ { \mu } ^ { 4 } \Lambda _ { T } ^ { 4 } + T ^ { - 2 } . } \end{array}\tag{32}
$$

Moment bound for the imbalance. Suppose m $\geq 2 .$ . Starting from the actual counts at the deterministic time $s ,$ construct an auxiliary process that always selects the two least-sampled true Pareto optimal arms, using the adaptive probability in equation 9.

Couple its randomization with EG-ATT. On $\mathcal { G } _ { T }$ , the two processes coincide after s. Importantly, the auxiliary process is defined on every trajectory, so its transition probabilities can be analyzed without conditioning on $\mathcal { G } _ { T }$

Let $x _ { i } ( t )$ denote the centered count of Pareto optimal arm i in this process, and write

$$
D _ { t } = \left( \sum _ { i \in \mathcal { A } ^ { \star } } x _ { i } ( t ) ^ { 2 } \right) ^ { 1 / 2 } .
$$

Order these counts as $x _ { 1 } \leq x _ { 2 } \leq \cdot \cdot \cdot \leq x _ { m }$ . Since the smaller count is selected with probability at least $p _ { \star } = 0 . 7 0$

$$
\mathbb { E } [ x _ { A _ { t + 1 } } ( t ) \mid \mathcal { H } _ { t } ] \leq p _ { \star } x _ { 1 } + ( 1 - p _ { \star } ) x _ { 2 } ,
$$

where $\mathcal { H } _ { t }$ denotes the auxiliary process history.

Using $\textstyle \sum _ { i } x _ { i } = 0$ , we have

$$
x _ { 2 } \leq - \frac { x _ { 1 } } { m - 1 } , \qquad D _ { t } \leq - \sqrt { m ( m - 1 ) } x _ { 1 } .
$$

Consequently,

$$
\mathbb { E } [ x _ { A _ { t + 1 } } ( t ) \mid \mathcal { H } _ { t } ] \leq \frac { m p _ { \star } - 1 } { m - 1 } x _ { 1 }
$$

$$
\leq - \frac { D _ { t } } { 2 m } ,
$$

where the last inequality holds for every $m \geq 2$ when $p _ { \star } = 0 . 7 0$

Each selection changes the centered count vector by $e _ { i } - \mathbf { 1 } / m$ . Therefore,

$$
D _ { t + 1 } ^ { 2 } - D _ { t } ^ { 2 } = 2 x _ { A _ { t + 1 } } ( t ) + 1 - \frac { 1 } { m } ,
$$

which gives the drift bound

$$
\mathbb { E } [ D _ { t + 1 } ^ { 2 } - D _ { t } ^ { 2 } \mid \mathcal { H } _ { t } ] \leq - \frac { D _ { t } } { m } + 1 - \frac { 1 } { m } .\tag{33}
$$

We next establish a uniform fourth-moment bound. For $D _ { t } \geq 2 ( m - 1 )$ , concavity of the square root and equation 33 give

$$
\mathbb { E } [ D _ { t + 1 } - D _ { t } \mid \mathcal { H } _ { t } ] \leq - \frac { 1 } { 4 m } .
$$

Since $| D _ { t + 1 } - D _ { t } | \leq 1$ , Hoefding’s lemma, with $\theta = 1 / ( 4 m )$ , implies

$$
\mathbb { E } [ e ^ { \theta D _ { t + 1 } } \mid \mathcal { H } _ { t } ] \le e ^ { - 1 / ( 3 2 m ^ { 2 } ) } e ^ { \theta D _ { t } } .
$$

When $D _ { t } < 2 ( m - 1 )$ , the left-hand side is bounded by $e ^ { \theta ( 2 m - 1 ) }$ . Combining these cases and iterating from time s, we obtain

$$
\mathbb { E } e ^ { \theta D _ { T } } \leq \exp \left( \frac { s } { 4 m } - \frac { T - s } { 3 2 m ^ { 2 } } \right) + \frac { e ^ { ( 2 m - 1 ) / ( 4 m ) } } { 1 - e ^ { - 1 / ( 3 2 m ^ { 2 } ) } } .
$$

The suficient horizon condition yields

$$
T - s \geq 8 m s .
$$

Hence the first term is at most one, and the exponential moment is bounded by a constant depending only on $m$ . Using $x ^ { 4 } \leq 2 4 e ^ { \theta x } / \theta ^ { 4 }$ $x \geq 0$ , we conclude that

$$
\mathbb { E } D _ { T } ^ { 4 } \le M _ { m } ,\tag{34}
$$

where $M _ { m } < \infty$ depends only on $m$

On $\mathcal { G } _ { T }$ , the actual imbalance satisfies $\Phi _ { T } = D _ { T } ^ { 2 }$ . On its complement, $\Phi _ { T } ^ { 2 } \le T ^ { 4 }$ . Together with equation 31 and equation 34, this gives

$$
\mathbb { E } _ { \mu } \Phi _ { T } ^ { 2 } \le M _ { m } + T ^ { - 2 } .\tag{35}
$$

Moments of fairness. Recall the decomposition

$$
\mathcal { F } _ { T } = - \log \left( 1 - \frac { B _ { T } } { T } \right) + D _ { \mathrm { K L } } ( U _ { m } \| q _ { T } ) ,
$$

where $q _ { T }$ is the empirical allocation conditioned on Pareto optimal pulls.

Let $H = T - s$ and $\begin{array} { r } { a = - \log ( 1 - s / T ) } \end{array}$ . On ${ \mathcal { G } } _ { T } ,$ , we have $B _ { T } \ \leq \ s$ . If the auxiliary imbalance additionally satisfies $D _ { T } < H / ( 2 m )$ , every Pareto optimal arm has at least half its balanced allocation. The inequality between KL divergence and chi-squared divergence gives

$$
D _ { \mathrm { K L } } ( U _ { m } \| q _ { T } ) \leq \frac { 2 m \Phi _ { T } } { ( T - B _ { T } ) ^ { 2 } } \leq \frac { 2 m D _ { T } ^ { 2 } } { H ^ { 2 } } .
$$

Therefore,

$$
\mathcal { F } _ { T } \leq a + \frac { 2 m D _ { T } ^ { 2 } } { H ^ { 2 } }
$$

on this event.

By Markov’s inequality and equation 34, the probability that this bound does not apply is at most

$$
\delta _ { T } = T ^ { - 6 } + \frac { 1 6 m ^ { 4 } M _ { m } } { H ^ { 4 } } .
$$

On the complementary event, we use the general bound $\mathcal { F } _ { T } \leq \log ( T / m )$

Taking expectations and using $\mathbb { E } D _ { T } ^ { 2 } \le \sqrt { M _ { m } }$ gives

$$
\mathbb { E } _ { \mu } \mathcal { F } _ { T } \leq a + \frac { 2 m \sqrt { M _ { m } } } { H ^ { 2 } } + \delta _ { T } \log ( T / m ) .
$$

Similarly, since $\operatorname { V a r } _ { \mu } ( \mathcal { F } _ { T } ) \leq \mathbb { E } _ { \mu } \mathcal { F } _ { T } ^ { 2 }$

$$
\mathrm { V a r } _ { \mu } ( \mathcal { F } _ { T } ) \leq 2 a ^ { 2 } + \frac { 8 m ^ { 2 } M _ { m } } { H ^ { 4 } } + \delta _ { T } \log ^ { 2 } ( T / m ) .
$$

Finally, the horizon condition implies $s \leq C _ { \mu } \Lambda _ { T } \leq T / 2$ , so

$$
a \leq \frac { 2 C _ { \mu } \Lambda _ { T } } { T } , \qquad H \geq \frac { T } { 2 } , \qquad \delta _ { T } \leq T ^ { - 6 } + \frac { 2 5 6 m ^ { 4 } M _ { m } } { T ^ { 4 } } .
$$

Substitution yields

$$
\mathbb { E } _ { \mu } \mathcal { F } _ { T } \leq \frac { 2 C _ { \mu } \Lambda _ { T } } { T } + \frac { D } { T ^ { 2 } } ,
$$

and

$$
\operatorname { V a r } _ { \mu } ( \mathcal { F } _ { T } ) \leq \frac { 8 C _ { \mu } ^ { 2 } \Lambda _ { T } ^ { 2 } } { T ^ { 2 } } + \frac { D \log ^ { 2 } T } { T ^ { 4 } } ,
$$

for a suficiently large constant D depending only on m. Increasing D if necessary also gives the claimed bound on $\mathbb { E } _ { \mu } \Phi _ { T } ^ { 2 }$ .

Case $m = 1$ . When the Pareto set contains a single arm, $\Phi _ { T } = 0$ . On $\mathcal { G } _ { T }$ , the unique Pareto optimal arm is selected after release, so $\mathcal { F } _ { T } \leq a$ . Consequently,

$$
\begin{array} { r } { \mathbb { E } _ { \mu } \mathcal { F } _ { T } \leq a + T ^ { - 6 } \log T , \qquad \mathrm { V a r } _ { \mu } ( \mathcal { F } _ { T } ) \leq a ^ { 2 } + T ^ { - 6 } \log ^ { 2 } T . } \end{array}
$$

These satisfy the stated bounds with a suitable constant D.

Together with equation 32, this completes the proof.

## B Experimental Details

## B.1 Reward Instances

The three instances for each reward model are listed below. Arm indices are consistent across separation margins. The Pareto optimal arms are {1, 2, 3, 4} for Bernoulli rewards and {1, 2, 3} for Gaussian rewards. The Gaussian rewards have standard deviation $\sigma = 0 . 1 2$

Table 3: Mean reward vectors for the bernoulli instances.
<table><tr><td>Arm</td><td> $\eta _ { \mu } = 0 . 0 0 5$ </td><td> $\eta _ { \mu } = 0 . 0 1$ </td><td> $\eta _ { \mu } = 0 . 1$ </td></tr><tr><td>1</td><td>(0.525,0.5)</td><td>(0.55, 0.5)</td><td>(0.6,0.1)</td></tr><tr><td>2</td><td>(0.515, 0.505)</td><td>(0.53,0.51)</td><td>(0.4, 0.2)</td></tr><tr><td>3</td><td>(0.51,0.52)</td><td>(0.52, 0.54)</td><td>(0.3, 0.5)</td></tr><tr><td>4</td><td>(0.5, 0.535)</td><td>(0.5,0.57)</td><td>(0.1, 0.8)</td></tr><tr><td>5</td><td>(0.505, 0.505)</td><td>(0.51,0.51)</td><td>(0.2,0.2)</td></tr><tr><td>6</td><td>(0.5,0.5)</td><td>(0.5,0.5)</td><td>(0.1,0.1)</td></tr></table>

Table 4: Mean reward vectors for the gaussian instances.
<table><tr><td>Arm</td><td> $\eta _ { \mu } = 0 . 0 0 5$ </td><td> $\eta _ { \mu } = 0 . 0 2$ </td><td> $\eta _ { \mu } = 0 . 1$ </td></tr><tr><td>1</td><td>(0.5425,0.53)</td><td>(0.67, 0.62)</td><td>(0.8,0.55)</td></tr><tr><td>2</td><td>(0.53, 0.5425)</td><td>(0.62, 0.67)</td><td>(0.55, 0.8)</td></tr><tr><td>3</td><td>(0.535, 0.535)</td><td>(0.64, 0.64)</td><td>(0.65, 0.65)</td></tr><tr><td>4</td><td>(0.5275, 0.5275)</td><td>(0.61,0.61)</td><td>(0.5,0.5)</td></tr><tr><td>5</td><td>(0.525, 0.535)</td><td>(0.6,0.64)</td><td>(0.45,0.65)</td></tr></table>

## B.2 Baseline Implementations

We include the five baselines compared in the main text: Pareto-UCB1, Pareto Thompson sampling (PTS), Width-guided UCB, Pareto-KG, and Annealing-Pareto. Pareto-UCB1 uses the optimistic confidence radius without a Gaussian noise multiplier. Bernoulli PTS uses independent Beta(1, 1) priors, while Gaussian PTS uses normal mean posteriors. Width-guided UCB uses the theoremmatched coeficient $c = 2$ . Pareto-KG uses a posterior-uncertainty plug-in and is therefore an adaptation of the original method. Annealing-Pareto uses decay 0.4 for Bernoulli rewards and the fixed benchmark choice 0.97 for Gaussian rewards. Applying confidence rules originally developed for bounded rewards to Gaussian observations is also an adaptation.

## B.3 Tuning Details.

For each reward family, separation margin, and horizon, we tuned ATT and EG-ATT separately under two Bayesian-optimization (BO) objectives: the mean-plus-standard-deviation objective (M) and the log-scale objective (L), as defined in the main text. Each search evaluated 20 configurations using 28 pilot runs, compared leading configurations using 90 validation runs, and evaluated the selected configuration on 240 independent held-out runs. Tables 5 and 6 report the selected parameters together with the held-out mean and variance of terminal unfairness for every setting. The parameter vectors are $\pmb { \theta } = ( c _ { \tau } , p _ { 0 } , \delta , \lambda , p _ { - } , p _ { + } )$ , for the temperature, leader-probability intercept, count-deficit strength, denominator scale, and clipping limits, and $\boldsymbol { \omega } = ( q _ { 0 } , p _ { g } , a _ { \rho } , a _ { g } )$ , for the quota coeficient, quota probability, reserve multiplier, and gate multiplier; the latter are used only by EG-ATT. Numerical parameter values are rounded for display. In all 48 selected configurations, the observed gate-release fraction was zero, so these finite-horizon EG-ATT results reflect quota protection without gate release.

Lastly, we observe that the terminating condition is never met in the reported EG-ATT runs. Thus, these experiments isolate the efect of quota protection at the tested horizons. This may suggest that the terminating condition is conservative or that longer horizons are needed for it to be met. Further investigation is beyond the scope of this paper but may be of independent interest.

Table 5: Selected BO parameters and held-out unfairness for Bernoulli rewards. M denotes mean+SD tuning; L denotes log-scale tuning.
<table><tr><td>ημ</td><td></td><td>T Method</td><td>BO θ</td><td></td><td>ω</td><td>Mean</td><td>Variance</td></tr><tr><td>0.005</td><td>5,000</td><td>ATT</td><td>M</td><td>(2.87, 0.658, 0.010, 0.155, 0.452, 0.994)</td><td></td><td>0.3655</td><td>0.004249</td></tr><tr><td>0.005</td><td>5,000</td><td>ATT</td><td>L</td><td>(2.88, 0.597, 0.200, 0.669, 0.317, 0.994)</td><td></td><td>0.3651</td><td>0.004257</td></tr><tr><td>0.005</td><td>5,000</td><td>EG-ATT M</td><td></td><td>(2.90, 0.589, 0.267, 0.523, 0.325, 0.994)</td><td>(107.52, 0.552, 0.494, 9.67)</td><td>0.3719</td><td>0.001447</td></tr><tr><td>0.005</td><td>5,000</td><td>EG-ATT</td><td>L</td><td>(1.88, 0.581, 0.028, 0.189, 0.414, 0.912)</td><td>(155.33, 0.762, 0.673, 8.62)</td><td>0.4048</td><td>2.7 × 10−6</td></tr><tr><td>0.005</td><td>50,000</td><td>ATT</td><td>M</td><td>(1.57, 0.578, 0.280, 0.026, 0.482, 0.920)</td><td></td><td>0.2495</td><td>0.01683</td></tr><tr><td>0.005</td><td>50,000</td><td>ATT</td><td>L</td><td>(3.00, 0.801, 0.266, 0.116, 0.507, 0.936)</td><td></td><td>0.2758</td><td>0.004433</td></tr><tr><td>0.005</td><td>50,000</td><td>EG-ATT</td><td>M</td><td>(1.40, 0.782, 0.417,0.276, 0.536, 0.983)</td><td>(8.37, 0.949, 0.750, 9.82)</td><td>0.2488</td><td>0.01999</td></tr><tr><td>0.005</td><td>50,000</td><td>EG-ATT</td><td>L</td><td>(2.89, 0.622, 0.088, 0.203, 0.464, 0.926)</td><td>(159.27, 0.633, 0.347, 9.22)</td><td>0.273</td><td>0.004707</td></tr><tr><td>0.01</td><td>5,000</td><td>ATT</td><td>M</td><td>(2.18, 0.718, 0.261, 0.042, 0.488, 0.937)</td><td></td><td>0.3041</td><td>0.009042</td></tr><tr><td>0.01</td><td>5,000</td><td>ATT</td><td>L</td><td>(2.95, 0.558, 0.169, 0.044, 0.478, 0.986)</td><td></td><td>0.3099</td><td>0.004766</td></tr><tr><td>0.01</td><td>5,000</td><td>EG-ATT</td><td>M</td><td>(2.03, 0.869, 0.107, 0.538, 0.390, 0.980)</td><td>(2.17, 0.592, 0.666, 7.18)</td><td>0.302</td><td>0.01001</td></tr><tr><td>0.01</td><td>5,000</td><td>EG-ATT</td><td>L</td><td>(2.91, 0.721, 0.468, 0.190, 0.437, 0.932)</td><td>(148.17,0.918, 0.551, 7.05)</td><td>0.4053</td><td>7.5 × 10−10</td></tr><tr><td>0.01</td><td>50,000</td><td>ATT</td><td>M</td><td>(1.00, 0.700, 0.250, 0.150, 0.450, 0.950</td><td></td><td>0.1368</td><td>0.02818</td></tr><tr><td>0.01</td><td>50,000</td><td>ATT</td><td>L</td><td>(2.36, 0.567, 0.084, 0.068, 0.504, 0.910)</td><td></td><td>0.1578</td><td>0.004353</td></tr><tr><td>0.01</td><td>50,000</td><td>EG-ATT M</td><td></td><td>(1.77, 0.669, 0.130, 0.148, 0.484, 0.960)</td><td>(35.36, 0.504, 0.930, 10.94)</td><td>0.1415</td><td>0.006578</td></tr><tr><td>0.01</td><td>50,000</td><td>EG-ATT</td><td>L</td><td>(2.32, 0.769, 0.029, 0.561, 0.525, 0.973)</td><td>(2.62, 0.651, 0.785, 11.58)</td><td>0.1566</td><td>0.004438</td></tr><tr><td>0.1</td><td>5,000</td><td>ATT</td><td>M</td><td>(1.00, 0.700, 0.250, 0.150, 0.450, 0.950)</td><td></td><td>0.0246</td><td>0.002773</td></tr><tr><td>0.1</td><td>5,000</td><td>ATT</td><td>L</td><td>(1.00, 0.700, 0.250, 0.150, 0.450, 0.950)</td><td></td><td>0.0246</td><td>0.002773</td></tr><tr><td>0.1</td><td>5,000</td><td>EG-ATT M</td><td></td><td>(0.739, 0.693, 0.244, 0.428, 0.538, 0.966)</td><td>(2.38, 0.784, 0.449, 8.96)</td><td>0.02273</td><td>6.55 × 10−5</td></tr><tr><td>0.1</td><td>5,000</td><td>EG-ATT</td><td>L</td><td>(1.00, 0.700, 0.250, 0.150, 0.450, 0.950)</td><td>(96.00, 0.700, 1.00, 8.00)</td><td>0.265</td><td>1.93 × 10−14</td></tr><tr><td>0.1</td><td>50,000</td><td>ATT</td><td>M</td><td>(0.517, 0.869, 0.049, 0.656, 0.349, 0.970)</td><td></td><td>0.006803</td><td>0.005705</td></tr><tr><td>0.1</td><td>50,000</td><td>ATT</td><td>L</td><td>(0.426, 0.895, 0.493, 0.385, 0.441, 0.998)</td><td></td><td>0.01402</td><td>0.01824</td></tr><tr><td>0.1</td><td>50,000</td><td>EG-ATT M</td><td></td><td>(0.443, 0.893, 0.483, 0.666, 0.534, 0.977)</td><td>(3.62, 0.567, 0.118, 6.99)</td><td>0.002758</td><td> $7 . 7 2 \times 1 0 ^ { - 7 }$ </td></tr><tr><td></td><td>0.1 50,000</td><td>EG-ATT L</td><td></td><td>(0.322, 0.893, 0.416, 0.524, 0.313, 0.996)</td><td>(3.73,0.782, 0.270, 9.41)</td><td>0.003614</td><td> $2 . 0 8 \times 1 0 ^ { - 7 }$ </td></tr></table>

Table 6: Selected BO parameters and held-out unfairness for Gaussian rewards. M denotes mean+SD tuning; L denotes log-scale tuning.
<table><tr><td> $\eta _ { \mu }$ </td><td></td><td>T Method</td><td>BO θ</td><td></td><td>ω</td><td>Mean</td><td>Variance</td></tr><tr><td>0.005</td><td>5,000</td><td>ATT</td><td>M</td><td>(2.20, 0.586, 0.221, 0.080, 0.460, 0.915)</td><td></td><td>0.3188</td><td>0.009786</td></tr><tr><td>0.005</td><td>5,000</td><td>ATT</td><td>L</td><td>(2.92, 0.560, 0.446, 0.580, 0.307, 0.993)</td><td></td><td>0.3468</td><td>0.007814</td></tr><tr><td>0.005</td><td>5,000</td><td>EG-ATT M</td><td></td><td>(0.758, 0.813, 0.068, 0.351, 0.450, 0.921)</td><td>(42.86, 0.943, 0.004, 6.00)</td><td>0.2826</td><td>0.01957</td></tr><tr><td>0.005</td><td>5,000</td><td>EG-ATT L</td><td></td><td>(2.05, 0.662, 0.389, 0.172, 0.386, 0.922)</td><td>(155.36, 0.580, 0.806, 8.67)</td><td>0.4923</td><td>4.89 × 10−5</td></tr><tr><td>0.005</td><td>50,000</td><td>ATT</td><td>M</td><td>(1.00, 0.700, 0.250, 0.150, 0.450, 0.950)</td><td></td><td>0.07131</td><td>0.0168</td></tr><tr><td>0.005</td><td>50,000</td><td>ATT</td><td>L</td><td>(1.80, 0.697, 0.270, 0.036, 0.440, 0.939)</td><td></td><td>0.09463</td><td>0.001683</td></tr><tr><td>0.005</td><td>50,000</td><td>EG-ATT M</td><td></td><td>(1.00, 0.700, 0.250, 0.150, 0.450, 0.950)</td><td>(96.00, 0.700, 1.00, 8.00)</td><td>0.06995</td><td>0.004597</td></tr><tr><td>0.005</td><td>50,000</td><td>EG-ATT L</td><td></td><td>(0.823, 0.677, 0.068, 0.082, 0.479,0.921)</td><td>(65.22, 0.881, 0.665, 9.33)</td><td>0.06028</td><td>0.005503</td></tr><tr><td>0.02</td><td>5,000</td><td>ATT</td><td>M</td><td>(1.00, 0.700, 0.250, 0.150, 0.450, 0.950)</td><td></td><td>0.05165</td><td>0.01231</td></tr><tr><td>0.02</td><td>5,000</td><td>ATT</td><td>L</td><td>(1.00, 0.700, 0.250, 0.150, 0.450, 0.950)</td><td></td><td>0.05165</td><td>0.01231</td></tr><tr><td>0.02</td><td>5,000</td><td>EG-ATT M</td><td></td><td>(0.707,0.870, 0.361, 0.545, 0.477,0.912)</td><td>(4.14,0.791,0.635, 11.16)</td><td>0.03897</td><td>0.001997</td></tr><tr><td>0.02</td><td>5,000</td><td>EG-ATT L</td><td></td><td>(0.784, 0.801, 0.108, 0.102, 0.430, 0.926)</td><td>(78.86, 0.535, 0.996, 9.69)</td><td>0.2196</td><td>6.07 × 10−15</td></tr><tr><td>0.02</td><td>50,000</td><td>ATT</td><td>M</td><td>(1.00, 0.700, 0.250, 0.150, 0.450, 0.950)</td><td></td><td>0.006584</td><td>7.45 × 10−6</td></tr><tr><td>0.02</td><td>50,000</td><td>ATT</td><td>L</td><td>(1.00, 0.700, 0.250, 0.150, 0.450, 0.950)</td><td></td><td>0.006584</td><td>7.45 × 10−6</td></tr><tr><td>0.02</td><td>50,000</td><td>EG-ATT M</td><td></td><td>(0.416, 0.896, 0.043, 0.556, 0.497, 0.950)</td><td>(3.29, 0.578, 0.303, 8.77)</td><td>0.004619</td><td>3.22 × 10−6</td></tr><tr><td>0.02</td><td>50,000</td><td>EG-ATT L</td><td></td><td>(0.257, 0.894, 0.484, 0.329, 0.510, 0.971)</td><td>(12.30, 0.935, 0.481, 11.44)</td><td>0.00686</td><td>1.34 × 10−6</td></tr><tr><td>0.1</td><td>5,000</td><td>ATT</td><td>M</td><td>(0.405,0.712, 0.119,0.151,0.323, 0.954)</td><td></td><td>0.001771</td><td>1.94 × 10−6</td></tr><tr><td>0.1</td><td>5,000</td><td>ATT</td><td>L</td><td>(0.703, 0.551, 0.040, 0.651, 0.465, 0.956)</td><td></td><td>0.002576</td><td>1.12 × 10−6</td></tr><tr><td>0.1</td><td>5,000</td><td>EG-ATT M</td><td></td><td>(0.157,0.796, 0.269, 0.239, 0.362, 0.988)</td><td>(2.09, 0.763, 0.035, 9.25)</td><td>0.007226</td><td>4.92 × 10−16</td></tr><tr><td>0.1</td><td>5,000</td><td>EG-ATT L</td><td></td><td>(0.274,0.582, 0.155, 0.326, 0.370, 0.987)</td><td>(2.32,0.698, 0.026, 11.96)</td><td>0.00723</td><td>2.19 × 10−9</td></tr><tr><td>0.1</td><td>50,000</td><td>ATT</td><td>M</td><td>(0.584,0.796, 0.479, 0.079, 0.533, 0.995)</td><td></td><td>0.000296</td><td>1.47 × 10−8</td></tr><tr><td>0.1</td><td>50,000</td><td>ATT</td><td>L</td><td>(0.242, 0.589, 0.496, 0.597, 0.319, 0.950)</td><td></td><td>0.0001842</td><td>2.03 × 10−7</td></tr><tr><td>0.1</td><td>50,000</td><td>EG-ATT M</td><td></td><td>(0.216, 0.869, 0.012, 0.328, 0.515, 0.970)</td><td>(2.13, 0.895, 0.105, 10.87)</td><td>0.001681</td><td>2.42 × 10−20</td></tr><tr><td>0.1</td><td>50,000</td><td>EG-ATT L</td><td></td><td>(0.190, 0.659, 0.231, 0.410, 0.489, 0.923)</td><td>(2.43, 0.851, 0.044, 11.95)</td><td>0.001281</td><td>6.68 × 10−12</td></tr></table>