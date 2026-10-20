# A Simple and General Approach to top-� Selection

Motti Goldberger

Yale School of Management, motti.goldberger@yale.edu

Nils Rudi

Yale School of Management, nils.rudi@yale.edu

Abstract. This paper considers the Bayesian top-� identification problem, wherein a decision maker is tasked with selecting � items from a pool of items of unknown quality. To assess the items, the decision maker adaptively allocates a fixed sampling budget, where each sample of an item yields a noisy observation of its quality, with the objective of minimizing the expected opportunity cost of the selected set. We propose an approach that allocates each batch of samples by minimizing a surrogate objective derived from an upper bound on the expected opportunity cost. The approach is general: samples can be allocated in batches, the selection size � is arbitrary, and it applies to many sampling distributions. It is also simple and intuitive. We prove that the method’s posterior expected opportunity cost decays at an exponential rate of at least half the best rate achievable by any sampling rule. In simulations across a wide range of settings, the method is competitive with, and often improves upon, the best existing algorithm in each setting

Key words: Ranking-and-Selection, Best-Arm Identification, Top-� Identification

## 1. Introduction

This paper studies the problem of a decision maker who faces a large pool of items with unknown quality, can assess them through noisy and costly samples, and wants to identify and select the best item(s). This is commonly referred to as a ranking and selection task in the operations literature (Chen et al. 2000, Chen and Lee 2010, Hong et al. 2021), and as a best-arm(s) identification task in the multi-armed bandit literature (Kalyanakrishnan et al. 2012, Kaufmann et al. 2016, Russo 2020). This problem arises in many domains, including clinical trials (Villar et al. 2015, Chick et al. 2022), A/B/n testing (Scott 2015, Ju et al. 2019), machine-learning model selection and hyperparameter optimization (Snoek et al. 2012, Jamieson and Talwalkar 2016, Li et al. 2018), and simulation optimization (Fu et al. 2005). In each setting, the key question is how to adaptively allocate a fixed sampling budget to maximize the quality of the final selection.

A central goal of this paper is to treat the problem with generality so that the method is useful across a range of settings. Two aspects of this generality are worth discussing up front. The first concerns how samples are collected. The most adaptive option is to allocate one sample at a time and wait to observe the outcome before choosing the next. However, this can be impractical to implement. For example, in clinical trials, one may not be able to wait to observe an outcome before assigning the next treatment (Robertson et al. 2023), and in simulation optimization, many simulations may be run in parallel (Hong et al. 2021). We therefore consider the partially adaptive case, in which batches of samples are chosen in rounds, and outcomes are observed between rounds. The second aspect is the target number of items to select. Choosing the single best one (� = 1) is the most common and most studied case, but some of the applications above instead call for selecting several items. Our method can handle any target selection size �, giving it a wider range of applicability.

There are two common metrics for how “good” a sample allocation procedure is. One is its probability of correct selection (PCS), the probability that the selected items are the true best items (Chen et al. 2000, Glynn and Juneja 2004, Li et al. 2025). The other is its expected opportunity cost (EOC), sometimes referred to as simple regret, which is the expected shortfall in value between the selected set and the true best set (Chick and Inoue 2001, Gao and Chen 2015). PCS treats every incorrect selection equally, while EOC makes bigger mistakes more costly. For many tasks, EOC, which we adopt as our objective, is the more natural fit since a big mistake can be far worse than a marginal one. That said, the two are closely related. In simulations, a method that performs well on one metric tends to perform well on the other, and our theoretical guarantees apply to both objectives.

Given the objective, the decision maker’s problem can be formulated as a finite-horizon dynamic program. In each round, given the current state (the posterior distribution over item values) and the available budget, one chooses the optimal batch of samples by accounting for all possible outcomes and future states. However, the complexity of this program grows rapidly with the budget size, so solving it exactly is out of reach (Frazier et al. 2008, Hong et al. 2021, Zhang et al. 2023). This has motivated a large body of literature, which we group into two broad categories. The first set of methods allocates samples by solving a sequence of static optimization problems, one per round, in place of the dynamic program. The second group constructs allocation rules directly from the current state without solving an optimization problem. We take the first approach and construct a per-round surrogate objective whose minimizer gives the sample allocation.

There are several ways to form a sequence of one-round static objectives. In a Bayesian formulation, one common approach is to maximize the expected value of information (EVI), the improvement in the quality of the final selection that collecting the next batch would produce (Chick and Inoue 2001, Chick et al. 2010). When samples follow a Gaussian distribution and are chosen one at a time, Frazier et al. (2008, 2009) show that this can be computed exactly for the top-1 EOC objective, yielding the knowledge gradient (KG) policy. In general, however, even the one-step look-ahead objective is intractable, so the literature resorts to approximations. Examples targeting PCS include AOAP for top-1 identification (Peng et al. 2018), and AOAm (Zhang et al. 2023) and EMIp (Liu et al. 2024) for top-�, all designed for Gaussian samples. For AOAm, Zhang et al. (2023) also prove that the sampling proportions converge to the proportions that maximize the asymptotic error-decay rate, assuming their necessary sample-proportion conditions have a unique solution.

A second approach in this category, referred to as optimal computing budget allocation (OCBA) (Chen and Lee 2010), assumes that in each round the current estimates of the means are the true means and finds the sample allocation that would be optimal under this assumption (Chen et al. 2000). However, even with this assumption, the problem is generally intractable. It is therefore commonly approached by constructing a bound on the objective, a lower bound for PCS or an upper bound for EOC, and deriving allocations that optimize the bound as the budget goes to infinity. Most of this work concerns Gaussian samples and top-1 selection under PCS, with extensions to top-� (Chen et al. 2008, Gao and Chen 2016), and to top-1 and top-� selection under EOC (He et al. 2007, Gao and Chen 2015).

An alternative approach targets learning the allocation that maximizes the exponential rate at which the probability of incorrect selection decays as the budget grows, the large-deviation rate (Glynn and Juneja 2004). Algorithms of this kind run sequentially, again treating the current mean estimates as the truth, and choose allocations using the optimality conditions for this rate. Chen and Ryzhov (2023) establish such an algorithm that applies to general distributions in the � = 1 setting and prove that the sampling proportions converge to the optimal proportions as the budget approaches infinity.

Next, we review methods that construct sampling rules directly from the current state. Methods similar in spirit to our approach are confidence and gap-based frequentist approaches, which allocate samples based on how close items are to the current estimated top-� cutof and how uncertain their estimated values are, such as gap-e (Gabillon et al. 2011, Bubeck et al. 2013) and KL-LUCB (Kaufmann and Kalyanakrishnan 2013). Our allocation algorithm is based on the same two quantities, but difers in how the top-� cutof is defined, how the distance from it is measured, and how uncertainty is used.

Elimination algorithms (Audibert et al. 2010, Karnin et al. 2013) divide sampling into rounds, allocate samples uniformly among the remaining items within a round, and eliminate the empirically weakest items according to an elimination schedule. The successive accepts and rejects (SAR) approach (Bubeck et al. 2013) also “accepts” empirically strong items, avoiding spending samples on items already determined to likely be in the top-�.

Algorithms based on the expected improvement (EI) rule, which originates from the Bayesian optimization literature (Jones et al. 1998), have been designed for top-1 selection problems (Ryzhov 2016, Qin et al. 2017, Chen and Ryzhov 2019). For each item, the algorithm computes the expected amount by which the item’s value exceeds that of the current best item, and samples the item with the largest improvement score. In the Gaussian case, Chen and Ryzhov (2019) show that a variant of EI achieves the optimal sampling proportions asymptotically.

A family of Bayesian algorithms referred to as top two algorithms allocates a sample in each round by identifying a “leader” item and a “challenger” item, then flipping a weighted coin to decide which of the two to sample (Russo 2020). Various methods have been proposed for choosing the leader and challenger (Jourdan et al. 2022), for example, based on Thompson sampling (Russo 2020) or expected improvement scores (Qin et al. 2017). For � = 1 and a broad class of sampling distributions, Russo (2020) proves that with an unbiased coin, the posterior probability of an incorrect selection decays at an exponential rate of at least half the best achievable rate. Russo (2020) further shows that with suitable tuning of the coin flip probability, it attains the optimal decay rate. Qin et al. (2017) establish the corresponding results for top two expected improvement with Gaussian samples. You et al. (2023) and Qin and You (2026) extend top two methods to the top-� setting and give a rigorous way to weight the coin such that in the top-1 Gaussian setting, the posterior probability of an incorrect selection decays at the optimal exponential rate without a tuning parameter, though their general top-� sampling rules do not come with posterior error decay rate guarantees. Establishing optimality for general � is more dificult because the conditions governing how samples must be allocated among items have a more complicated structure (Zhang et al. 2023, Qin and You 2026). Russo (2020) conjectures that his top-� extension of top two Thompson sampling achieves a posterior error decay rate of at least half the optimum.

Given the rich literature on this problem, our motivation for this work stems from a related project. In Goldberger and Rudi (2026), we study the top-� identification problem in a diferent setting with three key components that existing methods do not address in combination. First, the decision maker can choose among diferent binary sample forms (in particular, a binary rating of an item or a comparison of two items), each with a diferent cost and signal quality. Methods that allocate directly from the current state have no clear way to handle this, whereas methods that solve a static problem may, because the costs and signal quality of each sample form can be incorporated into the objective. Second, that paper considers “small” budget settings, so analysis in the asymptotic regime is not of primary importance. Third, samples may be collected in batches. Among the methods above, OCBA-style and elimination-based algorithms support batching (Jun et al. 2016). However, OCBA algorithms generally rely on asymptotic analysis, and elimination-based algorithms are often outperformed in simulation by other methods such as gap-e and KL-LUCB (Bubeck et al. 2013, Kaufmann and Kalyanakrishnan 2013, Zhang et al. 2023). This motivated us to develop a new method for that specific problem. In this paper, building on insights from that work, we develop a simpler, far more general method for Bayesian top-� identification.

Our approach. We take the approach of solving a sequence of tractable static optimization problems, which we develop in two steps. First, we give an upper bound on EOC, structurally diferent from those in the OCBA literature on EOC, and discuss why this structure has the potential for an improved algorithm. Second, to address the intractability of minimizing this bound as a one-round static problem, we approximate the future posterior distribution after a batch via expected moment matching rather than relying on asymptotic analysis. This idea originates in the EVI and OCBA literature with Gaussian samples by Chen et al. (1997), He et al. (2007), and Zhang et al. (2023), and we generalize it to other distributions. These two steps yield a simple, interpretable per-round static surrogate objective. A key property of the surrogate is that it is separable across items, enabling eficient algorithms for solving the batch problem without relying on heuristics.

Results and theoretical guarantees. Similar to the top two and modified expected-improvement methods for � = 1, our method is conceptually simple and intuitive, yet remarkably comes with strong theoretical performance guarantees for general �. For a broad class of sampling distributions, we prove that posterior EOC (as well as the posterior probability of an incorrect selection) decays at an exponential rate of at least half the best achievable rate. This establishes the type of general-� guarantee conjectured by Russo (2020), but for our sampling rule.

In a range of simulations, we demonstrate that our method is competitive with and often improves on the current state-of-the-art approaches, including those specialized to $k = 1$ as well as those for general $k > 1$ those specialized to Gaussian as well as other distributions, and both at small and large budgets. These comparisons are conducted in the non-batched setting to include as many existing methods as possible. In the batched setting, we show that our method significantly outperforms existing batch algorithms and its performance degrades less as the batch size grows.

We believe this method can be valuable to both researchers and practitioners. It is flexible across a wide range of practical settings, is competitive with the best algorithms in each of these settings, and extends naturally to richer settings, such as those with varying sample forms and sampling costs.

The rest of the paper is organized as follows: §2 formulates the model and objective, §3 derives the surrogate objective and the batch allocation algorithm, §4 establishes the theoretical guarantees, §5 presents the numerical experiments, and §6 concludes.

## 2. Model and objective

We consider � items, indexed $i = 1 , \ldots , I ,$ , each with an unknown mean $\theta _ { i } \in \mathcal { I }$ , where $\mathcal { T } \subseteq \mathbb { R }$ is an open interval. The goal is to select the $k < I$ items with the largest means. Each item � has a prior distribution $\Pi _ { i , 0 }$ on $\theta _ { i }$ with density $\pi _ { i , 0 }$ . For item �, let $y _ { i }$ be the outcome space, and let $\{ P _ { i } ( \cdot | \theta ) : \theta \in \mathcal { T } \}$ be a known mean-parametrized family of distributions on $y _ { i }$ with density or mass function $p _ { i } ( \cdot \mid \theta )$ . Conditional on $\theta _ { i }$ , the �th observation $Y _ { i , n }$ from item � is drawn i.i.d. from $P _ { i } ( \cdot \mid \theta _ { i } )$ , with $\mathbb { E } [ Y _ { i , n } \mid \theta _ { i } ] = \theta _ { i }$ . We make the following regularity assumption.

ASSUMPTION 1 (Regular one-parameter Bayesian model). For each $i = 1 , \ldots , I ,$ , the prior $\Pi _ { i , 0 }$ and sampling family $\{ P _ { i } ( \cdot | \theta ) : \theta \in \mathcal { T } \}$ are such that for every $n \geq 0$ and every $( y _ { 1 } , . . . , y _ { n } ) \in \mathcal { Y } _ { i } ^ { n }$ the posterior of $\theta _ { i }$ conditioned on $\{ Y _ { i , 1 } = y _ { 1 } , \dots , Y _ { i , n } = y _ { n } \}$ has continuous positive density on J and finite second moment, and the corresponding posterior predictive distribution of $\mathrm { Y } _ { i , n + 1 }$ hasfinite second moment. In addition, the priors and observations are independent across items.

Remark 1 gives examples of standard conjugate pairs (ESCP) satisfying Assumption 1 that we will refer to throughout the paper.

REMARK 1 (ESCP). ESCP covered by Assumption 1:

– Gaussian: $Y _ { i , n } \mid \theta _ { i } \sim N ( \theta _ { i } , \sigma _ { i } ^ { 2 } )$ with $\sigma _ { i } ^ { 2 }$ known, prior $\theta _ { i } \sim N ( m _ { i } ( 0 ) , s _ { i } ^ { 2 } ( 0 ) )$ , ${ \mathcal { T } } = \mathbb { R } .$

– Bernoulli: $Y _ { i , n } \mid \theta _ { i } \sim \mathrm { B e r n } ( \theta _ { i } )$ , prior $\theta _ { i } \sim \mathrm { B e t a } ( a _ { i } ( 0 ) , b _ { i } ( 0 ) ) , ~ \mathcal { T } = ( 0 , 1 )$

– Poisson: $Y _ { i , n } \mid \theta _ { i } \sim \mathrm { P o i s } ( \theta _ { i } )$ , prior $\theta _ { i } \sim \mathrm { G a m m a } ( a _ { i } ( 0 ) , b _ { i } ( 0 ) )$ with rate $b _ { i } ( 0 )$ $\mathcal { T } = ( 0 , \infty )$

– Gamma with known shape: $Y _ { i , n } \mid \theta _ { i } \sim \mathrm { G a m m a } ( \kappa _ { i } , \theta _ { i } / \kappa _ { i } )$ with scale $\theta _ { i } / \kappa _ { i }$ and $\kappa _ { i }$ known $( \kappa _ { i } = 1$ gives the exponential distribution), prior $\theta _ { i } \sim$ InvGamma $( a _ { i } ( 0 ) , b _ { i } ( 0 ) )$ with $a _ { i } ( 0 ) > 2 , b _ { i } ( 0 ) > 0 ,$ $\mathcal { T } = ( 0 , \infty )$

– Lognormal with known log-variance: $Y _ { i , n } \mid \boldsymbol { \theta } _ { i }$ ∼ Lognormal(log $\theta _ { i } - \sigma _ { i } ^ { 2 } / 2 , \sigma _ { i } ^ { 2 } )$ with $\sigma _ { i } ^ { 2 }$ known, prior log $\theta _ { i } \sim N ( m _ { i } ( 0 ) , s _ { i } ^ { 2 } ( 0 ) ) , ~ \mathcal { T } = ( 0 , \infty )$

Sampling proceeds in $T { + } 1$ rounds, an optional warm-up round indexed $t = 0$ followed by � adaptive rounds. In the warm-up round, each item is sampled $n _ { 0 } \in \mathbb { Z } _ { \geq 0 }$ times. In rounds $t = 1 , \ldots , T$ , we choose a batch of $B _ { t } \in \mathbb { Z } _ { + }$ samples, where the per-round batch sizes $\{ B _ { t } \} _ { t = 1 } ^ { T }$ are pre-specified. The choice of samples in round � is denoted by the vector $\Delta ^ { t } = ( \Delta _ { 1 } ^ { t } , \dots , \Delta _ { I } ^ { t } )$ , where $\Delta _ { i } ^ { t } \in \mathbb { Z } _ { \geq 0 }$ is the number of samples assigned to item �. Let $N _ { i } ( t )$ denote the total samples allocated to item � up to but not including round $t ,$ so that $N _ { i } ( 0 ) = 0$ $N _ { i } ( 1 ) = n _ { 0 }$ , and

$$
N _ { i } ( t + 1 ) = N _ { i } ( t ) + \Delta _ { i } ^ { t } , \qquad t = 1 , \ldots , T .
$$

Let $\mathcal { H } _ { t }$ denote the history at time �, which is the collection of priors $\{ \Pi _ { i , 0 } \} _ { i = 1 } ^ { I }$ and observations $\{ Y _ { i , n } : 1 \leq i \leq$ $I , 1 \leq n \leq N _ { i } ( t ) \}$ . We write $\mathbb { E } _ { t } \left[ \cdot \right]$ and $\mathbb { P } _ { t } ( \cdot )$ for conditional expectation and probability given $\mathcal { H } _ { t }$ . Let $\Pi _ { i , t }$ denote the posterior distribution of $\theta _ { i }$ given $\mathcal { H } _ { t }$ and define $\mu _ { i } ( t ) : = \mathbb { E } _ { t } [ \theta _ { i } ] , \nu _ { i } ( t ) : = \mathbb { V } _ { t } ( \theta _ { i } ) , F _ { i , t } ( x ) : = \mathbb { P } _ { t } ( \theta _ { i } \leq$ �). In round �, we refer to $\Pi _ { i , t }$ as the current prior.

In any round �, if we were to select items based on the current prior, we would pick the � items with the largest posterior means. Thus, it is convenient to define the following: for any vector $\mathbf { x } \in \mathbb { R } ^ { I }$ , let $S _ { k } ( { \mathbf x } )$ denote the set of indices of the � largest elements of x. Write $\pmb { \mu } ( t ) = ( \mu _ { 1 } ( t ) , \ldots , \mu _ { I } ( t ) )$ and $\pmb { \theta } = ( \theta _ { 1 } , \ldots , \theta _ { I } )$ . For any $\pmb { \theta } \in \mathbb { R } ^ { I }$ , the opportunity cost of the current selection $S _ { k } ( \pmb { \mu } ( t ) )$ is

$$
\operatorname { O C } ( S _ { k } ( \pmb { \mu } ( t ) ) , \pmb { \theta } ) : = \sum _ { i \in S _ { k } ( \pmb { \theta } ) } \theta _ { i } - \sum _ { i \in S _ { k } ( \pmb { \mu } ( t ) ) } \theta _ { i } .
$$

Thus the Bayesian expected opportunity cost given $\mathcal { H } _ { t }$ is

$$
\mathrm { E O C } ( \mathcal { H } _ { t } ) = \mathbb { E } _ { \pmb { \theta } | \mathcal { H } _ { t } } \left[ \mathrm { O C } ( S _ { k } ( \pmb { \mu } ( t ) ) , \pmb { \theta } ) \right] .\tag{1}
$$

The decision maker’s problem is to sequentially choose the sample batches $\{ \Delta ^ { t } \} _ { t = 1 } ^ { T }$ that minimize the terminal expected opportunity cost $\mathrm { E O C } ( \mathcal { H } _ { T + 1 } )$ . This can be formulated as a dynamic program, but it is intractable to solve, as discussed in the introduction. We next formulate a static surrogate objective that guides the choice of batch in each round.

## 3. Surrogate objective via EOC bound

EOC $( \mathcal { H } _ { t } )$ does not generally admit an expression amenable to analysis, so (1) is not immediately useful for choosing which batch of samples to collect. Thus, we construct the per-round surrogate objective in two steps. First, Lemma 1 gives an upper bound on $\operatorname { E O C } ( \mathcal { H } _ { t } )$ that is amenable to analysis. Second, in §3.1, we make this bound a function of the round-� allocation to guide the choice of future samples. We then discuss how the resulting per-round objective difers from those in prior work in §3.2 and give eficient algorithms for finding the batch that minimizes it in §3.3. §3.4 presents the sequential sampling procedure, and §3.5 gives explicit expressions of the sampling rule for the ESCP. We begin with the EOC upper bound, which is based on comparing each item’s $\theta _ { i }$ to a threshold.

LEMMA 1 (Threshold upper bound on EOC). (i) For every threshold $q \in \mathbb { R } ,$

$$
\mathrm { E O C } ( \mathcal { H } _ { t } ) \leq \mathbb { E } _ { t } \left[ \sum _ { \substack { i \in S _ { k } ( \mu ( t ) ) } } ( q - \theta _ { i } ) ^ { + } + \sum _ { \substack { i \notin S _ { k } ( \mu ( t ) ) } } ( \theta _ { i } - q ) ^ { + } \right] .
$$

(ii) The right-hand side of (i) is convex in � and has a unique minimizer $q ^ { * } ( t )$ characterized by

$$
\mathbb { E } _ { t } \left[ | \{ i : \theta _ { i } \geq q ^ { * } ( t ) \} | \right] = \sum _ { i = 1 } ^ { I } \{ 1 - F _ { i , t } ( q ^ { * } ( t ) ) \} = k .\tag{2}
$$

Proof All proofs are provided in the Appendix.

Part (i) gives a family of upper bounds on EOC, each separable across items. For an item in the current top-� set, its corresponding term in the bound is the expected shortfall of its true mean below threshold $q .$ For an item outside this set, its corresponding term is the expected amount its true mean exceeds the threshold. Part (ii) states that the tightest bound in this family is given by $q ^ { * } ( t )$ , the threshold such that, in expectation, exactly � item means are larger than it. This makes $q ^ { * } ( t )$ a natural threshold choice when constructing the surrogate objective. We also consider another top-� threshold motivated by the theoretical analysis in $\ S 4$ Write $d _ { i } ( x , y ) : = \mathrm { K L } ( P _ { i } ( \cdot | x ) \| P _ { i } ( \cdot | y ) )$ for the Kullback–Leibler (KL) divergence between the sampling distributions of item � at means � and $y _ { : }$ , and let $\mu _ { [ 1 ] } ( t ) \geq \cdot \cdot \cdot \geq \mu _ { [ I ] } ( t )$ denote the ordered posterior means. We define the threshold

$$
q ^ { \dagger } ( t ) : = \left\{ \begin{array} { l l } { \arg \operatorname* { m i n } _ { \substack { \pi _ { \lfloor k + 1 \rfloor } ( t ) < q < \mu _ { [ k ] } ( t ) } } \sum _ { i = 1 } ^ { I } \frac { 1 } { d _ { i } ( \mu _ { i } ( t ) , q ) } , } & { \mu _ { [ k ] } ( t ) > \mu _ { [ k + 1 ] } ( t ) , } \\ { \mu _ { [ k ] } ( t ) , } & { \mu _ { [ k ] } ( t ) = \mu _ { [ k + 1 ] } ( t ) . } \end{array} \right.\tag{3}
$$

$\ S 4$ explains why this choice of top-� threshold is “good”. In what follows, let $q ( t )$ denote the top-� threshold used in round �, which can be either $q ^ { * } ( t )$ or $q ^ { \dagger } ( t )$

## 3.1. Forward-looking objective

Given $q ( t )$ , the threshold bound at state $\mathcal { H } _ { t }$ is

$$
\mathbb { E } _ { t } \left[ \sum _ { i \in S _ { k } ( \mu ( t ) ) } ( \boldsymbol { q } ( t ) - \boldsymbol { \theta } _ { i } ) ^ { + } + \sum _ { i \notin S _ { k } ( \mu ( t ) ) } ( \boldsymbol { \theta } _ { i } - \boldsymbol { q } ( t ) ) ^ { + } \right] .\tag{4}
$$

To make this a function of a proposed round-� allocation $\Delta = \left( \Delta _ { 1 } , \ldots , \Delta _ { I } \right)$ , we take the expectation with respect to an approximation of the random posterior of $\theta _ { i }$ after allocating � and observing its outcomes. This approximation preserves the shape of $\Pi _ { i , t }$ and has mean and variance equal to the expected posterior mean and variance. We say two distributions have the same shape if they are equivalent after centering by their mean and scaling by their standard deviation. Put together, these conditions say that we replace the random future posterior with a deterministic, “shrunken” copy of the current prior, as it has the same mean as the prior, keeps the same shape, and changes only through a contraction around its mean by the expected amount of uncertainty that the proposed samples will remove. Figure 1 gives an example.

![](images/5d2cb7b4b6224514ecc92aff4c4a5851878c061d512b2131d38d4a3c70924e2a.jpg)

![](images/938697d031db7a561aaa628185aba4a55ac2b2d7e3c3f661c0a2a8202f25a62b.jpg)  
Figure 1 Structure of the surrogate loss $\mathcal { L } _ { t } ( \cdot )$ and allocation for $I = 3 , k = 1$ using $q ( t ) = q ^ { * } ( t )$ . For each item $i \in \{ A , B , C \}$ the left distribution is its current prior for $\theta _ { i } ,$ , and the right is the posterior moment-matched distribution $\widetilde { \theta } _ { i } ( \Delta _ { i } )$ . The allocation shown minimizes (8) with $B _ { t } = 1 0$ . The shaded area is the item’s contribution to the loss.

Formally, we replace each $\theta _ { i }$ in (4) by the random variable $\widetilde { \theta } _ { i } ( \Delta _ { i } )$ that satisfies

$$
\begin{array} { r } { \mathrm { ( i ) } \frac { \widetilde { \theta } _ { i } ( \Delta _ { i } ) - \mathbb { E } _ { t } [ \widetilde { \theta } _ { i } ( \Delta _ { i } ) ] } { \sqrt { \mathbb { V } _ { t } ( \widetilde { \theta } _ { i } ( \Delta _ { i } ) ) } } \stackrel { d } { = } \frac { \theta _ { i } - \mu _ { i } ( t ) } { \sqrt { \nu _ { i } ( t ) } } , } \end{array}
$$

$$
\mathrm { ( i i ) } \mathbb { E } _ { t } \left[ \widetilde { \theta } _ { i } ( \Delta _ { i } ) \right] = \mathbb { E } \left[ \mu _ { i } ( t + 1 ) \vert \Delta , \mathcal { H } _ { t } \right] = \mu _ { i } ( t ) ,
$$

$$
\begin{array} { r } { \mathrm { ( i i i ) } \mathbb { V } _ { t } \big ( \widetilde \theta _ { i } ( \Delta _ { i } ) \big ) = \mathbb { E } \big [ \mathbb { V } ( \theta _ { i } \mid \mathcal { H } _ { t + 1 } ) \mid \Delta , \mathcal { H } _ { t } \big ] = : \bar { \nu } _ { i } ( t , \Delta _ { i } ) . } \end{array}
$$

For a given $\mathcal { H } _ { t }$ , this distribution is unique and is given by

$$
\widetilde { \theta } _ { i } ( \Delta _ { i } ) = \mu _ { i } ( t ) + c _ { i } ( t , \Delta _ { i } ) \big ( \theta _ { i } - \mu _ { i } ( t ) \big ) , \qquad c _ { i } ( t , \Delta _ { i } ) : = \sqrt { \frac { \bar { \nu } _ { i } ( t , \Delta _ { i } ) } { \nu _ { i } ( t ) } } .\tag{5}
$$

We then define the surrogate loss induced by the allocation � as

$$
\begin{array} { r l } & { \mathcal { L } _ { t } ( \Delta ) : = \mathbb { E } _ { t } \left[ \displaystyle \sum _ { i \in S _ { k } ( \mu ( t ) ) } ( q ( t ) - \widetilde { \theta } _ { i } ( \Delta _ { i } ) ) ^ { + } + \displaystyle \sum _ { i \notin S _ { k } ( \mu ( t ) ) } ( \widetilde { \theta } _ { i } ( \Delta _ { i } ) - q ( t ) ) ^ { + } \right] } \\ & { \quad \quad = \displaystyle \sum _ { i = 1 } ^ { I } \mathbb { E } _ { t } \left[ ( q ( t ) - \widetilde { \theta } _ { i } ( \Delta _ { i } ) ) ^ { + } \right] + \displaystyle \sum _ { i \notin S _ { k } ( \mu ( t ) ) } ( \mu _ { i } ( t ) - q ( t ) ) , } \end{array}\tag{6}
$$

where the second equality follows from $( x - q ) ^ { + } = ( x - q ) + ( q - x ) ^ { + }$ . The second sum is constant with respect to �. For the first sum we have

$$
\mathcal { \ell } _ { i , t } ( \Delta _ { i } ) : = \mathbb { E } _ { t } [ ( q ( t ) - \widetilde { \theta } _ { i } ( \Delta _ { i } ) ) ^ { + } ] = c _ { i } ( t , \Delta _ { i } ) \mathbb { E } _ { t } [ ( z _ { i } ( t , \Delta _ { i } ) - \theta _ { i } ) ^ { + } ] ,\tag{7}
$$

where $z _ { i } ( t , \Delta _ { i } ) : = \mu _ { i } ( t ) + ( q ( t ) - \mu _ { i } ( t ) ) / c _ { i } ( t , \Delta _ { i } )$ . Choosing the batch that minimizes (6) is thus equivalent to

$$
\operatorname* { m i n } _ { \mathbf { A } \in \mathbb { Z } _ { \geq 0 } ^ { I } } \sum _ { i = 1 } ^ { I } \ell _ { i , t } ( \Delta _ { i } ) \qquad \mathrm { s . t . } \qquad \sum _ { i = 1 } ^ { I } \Delta _ { i } = B _ { t } ,\tag{8}
$$

which is the per-round surrogate program. Conceptually, allocating according to (8) sends samples to the items whose posterior has substantial mass on both sides of the top-� threshold $q ( t )$ , weighted by the distance of that mass from $q ( t )$ . These are the items whose membership in the top-� set is uncertain and whose misclassification, if it occurs, could be costly.

## 3.2. Relation to OCBA-style objectives

Before analyzing how to solve (8), we discuss how its structure difers from static objectives in the OCBA literature for EOC. These objectives often use an upper bound on EOC based on a sum of losses associated with pairwise misorderings. For example, for top-1 EOC with Gaussian samples, He et al. (2007) use

$$
\mathrm { E O C } ( \mathcal { H } _ { t } ) \leq \sum _ { i \neq i _ { ( 1 ) } } \mathbb { E } _ { t } \big [ \big ( \theta _ { i } - \theta _ { i _ { ( 1 ) } } \big ) ^ { + } \big ] .
$$

Gao and Chen (2015) give the following bound on top-� frequentist EOC:

$$
\mathrm { E O C } \leq \sum _ { j = 1 } ^ { k } \sum _ { \substack { i \notin \{ i _ { ( 1 ) } , \ldots , i _ { ( j ) } \} } } \sum _ { n = 1 } ^ { j } ( \mu _ { i _ { ( n ) } } ( t ) - \mu _ { i } ( t ) ) \Phi \left( - \frac { \mu _ { i _ { ( n ) } } ( t ) - \mu _ { i } ( t ) } { \sqrt { \nu _ { i _ { ( n ) } } ( t ) + \nu _ { i } ( t ) } } \right) ,
$$

where Φ is the standard normal CDF, and $i _ { ( 1 ) } , \ldots , i _ { ( I ) }$ order the items by their current posterior means. Their static objective is then based on driving this bound down. The potential issue with using such a bound is that the candidate top-� items $\{ i _ { ( 1 ) } , \ldots , i _ { ( k ) } \}$ are in far more terms in the sum than the non-top-� items. For instance, when $k = 1 , i _ { ( 1 ) }$ is in all �−1 terms, while all other items are only in one term. Thus, an objective based on such a bound may over-count the benefit of sampling the top items. Our bound avoids these pairwise terms by comparing each item to the same threshold $q ( t )$ , resulting in a single term for each item regardless of whether or not it is in $S _ { k } ( \pmb { \mu } ( t ) )$ . Figure 2 shows the distribution of sample allocation across items for our method, called TEAk, and the method of Gao and Chen (2015), called EOC-m. EOC-m systematically allocates more samples to items in the top-�, even items far above the top-� boundary, which is likely one of the main factors driving our method’s improvement over EOC-m, which we demonstrate in §5.

## 3.3. Batch allocation algorithm

Although (8) is an integer program, its separability across items allows us to solve it eficiently by dynamic programming and, in some cases, using a greedy procedure. Both are standard techniques. The main technical result of this subsection is Proposition 2(ii)–(iii), which gives conditions on the posterior distributions under which the greedy procedure is optimal.

The dynamic programming procedure in Algorithm 1 has two phases. First, we compute $\ell _ { i , t } ( d )$ for every item � and candidate allocation $d = 0 , \ldots , B _ { t }$ . Second, for each $r = 1 , \ldots , I$ and $b = 0 , \ldots , B _ { t }$ , we solve the subproblem of allocating exactly � samples among the first � items by forward induction, and then recover the optimal allocation by backtracking. Let $V _ { r } ( b )$ denote the minimum loss for this subproblem so that $V _ { I } ( B _ { t } )$ is the optimal value of (8).

![](images/24cf034952e592d8df13239982729601b0c281ac2ef229a7dffb4bf7f0f9e427.jpg)  
Figure 2 Budget allocation across items on the Gaussian instance described in §5, with $I = 2 0$ at $k = 3 , 5 , 1 0 .$ The plots show the proportion of the total budget spent on each item, with items ordered by their true mean. TEAk uses $q ^ { * } ( t )$ . The dashed line marks the top-� boundary.

```latex
Algorithm 1 Dynamic Programming Batch Builder (DPBB)
1: Precompute cost values: for each $i = 1 , \ldots , I$ and $d = 0 , \ldots , B _ { t }$ , compute $\ell _ { i , t } ( d )$
2: Initialize: set $V _ { 0 } ( 0 ) = 0$ and $V _ { 0 } ( b ) = + \infty$ for $b = 1 , \dots , B _ { t }$
3: Solve subproblems:
4: for $r = 1 , \ldots , I$
5: for $b = 0 , \ldots , B _ { t }$
6: Compute $\begin{array} { r } { V _ { r } ( b ) = \operatorname* { m i n } _ { \Delta \in \{ 0 , \ldots , b \} } \left\{ V _ { r - 1 } ( b - \Delta ) + \ell _ { r , t } ( \Delta ) \right\} } \end{array}$ and record an argmin $\Delta _ { r } ^ { * } ( b )$
7: Recover the allocation: set $b \gets B _ { t }$
8: for $r = I , \ldots , 1$
9: Set $\Delta _ { r } ^ { t } = \Delta _ { r } ^ { * } ( b )$ and update $b \gets b \textless \Delta _ { r } ^ { t }$
10: Return $\mathbf { \dot { \Delta } } \mathbf { \Delta } ^ { t } = ( \Delta _ { 1 } ^ { t } , \dots , \Delta _ { I } ^ { t } )$
```

PROPOSITION 1 (Optimality of DPBB). DPBB returns a global optimizer of (8). Let $C _ { \ell }$ denote the cost of computing each value $\ell _ { i , t } ( d )$ . The total computational cost is $O ( I B _ { t } C _ { \ell } + I B _ { t } ^ { 2 } )$ .

For the greedy procedure, define the marginal decrease from allocating one more sample to item � by $D _ { i , t } ( \Delta _ { i } ) : = \ell _ { i , t } ( \Delta _ { i } ) - \ell _ { i , t } ( \Delta _ { i } + 1 )$ . The procedure in Algorithm 2 sequentially assigns the $B _ { t }$ samples to the item with the current largest marginal decrease.

Algorithm 2 Greedy Batch Builder (GBB)   
1: Initialize: �<sup>�</sup> = 0   
2: for $n = 1 , \ldots , B _ { t }$   
3: For each $i = 1 , \ldots , I ,$ compute $D _ { i , t } ( \Delta _ { i } ^ { t } ) = \ell _ { i , t } ( \Delta _ { i } ^ { t } ) - \ell _ { i , t } ( \Delta _ { i } ^ { t } + 1 )$   
4: Choose �<sup>∗</sup> ∈ arg ma $\mathfrak { i } _ { 1 \le i \le I } D _ { i , t } ( \Delta _ { i } ^ { t } )$   
5: $\Delta _ { i ^ { * } } ^ { t } \gets \Delta _ { i ^ { * } } ^ { t } + 1$   
6: Return $\Delta ^ { t } \overset { \iota } { = } ( \Delta _ { 1 } ^ { t } , \ldots \ldots , \Delta _ { I } ^ { t } )$   
PROPOSITION 2 (Conditions for greedy optimality). (i) If,for every item �,   
$D _ { i , t } ( 0 ) \geq D _ { i , t } ( 1 ) \geq \cdots \geq D _ { i , t } ( B _ { t } - 1 ) ,$ (9)   
then GBB returns a global optimizer of (8). Its computational cost is $O ( I B _ { t } C _ { \ell } )$

(ii) Suppose that at round �, for every item � there exists $\lambda _ { i } ( t ) > 0$ such that

$$
c _ { i } ( t , d ) ^ { 2 } = \frac { 1 } { 1 + \lambda _ { i } ( t ) d } \qquad f o r e \nu e r y d \in \mathbb { Z } _ { \ge 0 } .\tag{10}
$$

Then (9) holds, and GBB returns a global optimizer of (8).

(iii) Suppose that for every item �, the sampling family is a regular one-parameter natural exponential family, meaning it has an open natural parameter space $\mathcal { E } _ { i }$ and density or massfunction $p _ { i } ( y \mid \eta _ { i } ) =$ $h _ { i } ( y ) \exp \{ \eta _ { i } y - A _ { i } ( \eta _ { i } ) \}$ , where $\theta _ { i } = A _ { i } ^ { \prime } ( \eta _ { i } )$ . Suppose also that the prior on $\eta _ { i }$ induced by $\Pi _ { i , 0 }$ has density proportional to exp $\left\{ a _ { i } \eta _ { i } { - } b _ { i } A _ { i } ( \eta _ { i } ) \right\}$ on $\mathcal { E } _ { i }$ for some $a _ { i } \in \mathbb { R }$ and $b _ { i } > 0 .$ . Then (10) holds at every round, and GBB returns a global optimizer of (8).

Part (i) says that GBB finds an optimal allocation whenever the gain from each additional sample is nonincreasing. However, checking this condition requires computing all marginal decreases in every round. Part (ii) avoids that computation by giving a suficient condition for (9) based on how the expected posterior variance decreases with additional samples, and part (iii) establishes that the condition holds for regular natural exponential families with conjugate priors.

REMARK 2 (SAMPLING COSTS). In some settings, items may difer in their sampling costs. Let $w _ { i } \in \mathbb { Z } _ { > 0 }$ denote the cost ofone sample ofitem �. The constraint in (8) becomes $\begin{array} { r } { \sum _ { i = 1 } ^ { I } w _ { i } \Delta _ { i } \leq B _ { t } } \end{array}$ and the same dynamic programming approach finds the optimal batch by accounting for these costs in the remaining budget.

## 3.4. Sequential algorithm

Algorithm 3 gives the full sampling procedure. In each round, we update the threshold from the current prior and allocate samples by solving (8) using one of the batch builders above.

Algorithm 3 Threshold-EOC allocation for top-� selection (TEAk)   
1: Input: priors $( \Pi _ { i , 0 } ) _ { i = 1 } ^ { I } ,$ , sampling distributions $( P _ { i } ( \cdot | \theta _ { i } ) ) _ { i = 1 } ^ { I }$ , warm-up �<sub>0</sub>, batch sizes $( B _ { t } ) _ { t = 1 } ^ { T }$   
2: Initialize: $N _ { i } ( 0 ) = \dot { 0 }$ for all �   
3: Warm-up: for each �, allocate $n _ { 0 }$ samples, observe outcomes, and update $N _ { i } ( 1 ) = n _ { 0 }$ and $\Pi _ { i , 1 }$   
4: for $t = 1 , \ldots , T$   
5: Compute �(�) using (2) or (3)   
6: Run DPBB to obtain �<sup>�</sup>, or GBB when (9) holds for every item   
7: For each �, allocate $\Delta _ { i } ^ { t }$ samples and observe outcomes   
8: Update $N _ { i } ( t { + } 1 ) = N _ { i } \dot { ( t ) } + \hat { \Delta } _ { i } ^ { t }$ and $\Pi _ { i , t + 1 }$   
9: Return $S _ { k } ( \pmb { \mu } ( T + 1 ) )$

## 3.5. ESCP explicit expressions for Algorithm 3

This section instantiates Algorithm 3 for the five ESCP in Remark 1. Recall from (7) that

$$
\ell _ { i , t } ( \Delta _ { i } ) = c _ { i } ( t , \Delta _ { i } ) \mathbb { E } _ { t } \big [ ( z _ { i } ( t , \Delta _ { i } ) - \theta _ { i } ) ^ { + } \big ] , \qquad z _ { i } ( t , \Delta _ { i } ) : = \mu _ { i } ( t ) + \frac { q ( t ) - \mu _ { i } ( t ) } { c _ { i } ( t , \Delta _ { i } ) } .
$$

Computing the round � allocation thus requires explicit formulas for $c _ { i } ( t , \Delta _ { i } )$ as well as $\mathbb { E } _ { t } \left[ ( z - \theta _ { i } ) ^ { + } \right]$ for each family, which are given in Proposition 3.

PROPOSITION 3. Fix round � and write $\mu _ { i } : = \mu _ { i } ( t )$ . For (ii)–(iv), write $a _ { i } : = a _ { i } ( t )$ and $b _ { i } : = b _ { i } ( t )$

(i) Gaussian, known variance $\sigma _ { i } ^ { 2 }$ . Posterior $N ( m _ { i } , s _ { i } ^ { 2 } ) , \mu _ { i } = m _ { i }$

$$
c _ { i } ( t , \Delta _ { i } ) ^ { 2 } = \frac { 1 } { 1 + ( s _ { i } ^ { 2 } / \sigma _ { i } ^ { 2 } ) \Delta _ { i } } \qquad \quad \mathbb { E } _ { t } \left[ ( z - \theta _ { i } ) ^ { + } \right] = ( z - m _ { i } ) \Phi \big ( \frac { z - m _ { i } } { s _ { i } } \big ) + s _ { i } \phi \big ( \frac { z - m _ { i } } { s _ { i } } \big ) ,
$$

where $\phi$ is the standard normal density.

(ii) Binomial/Bernoulli. Observations are binomial with success probability $p _ { i }$ and number oftrials $r \in \mathbb { Z } _ { + }$

Posterior $p _ { i } \sim \mathrm { B e t a } ( a _ { i } , b _ { i } ) , \mu _ { i } = r a _ { i } / n _ { i }$ where $n _ { i } = a _ { i } + b _ { i } .$

$$
c _ { i } ( t , \Delta _ { i } ) ^ { 2 } = \frac { 1 } { 1 + ( r / n _ { i } ) \Delta _ { i } } \qquad \quad \mathbb { E } _ { t } \big [ ( z - \theta _ { i } ) ^ { + } \big ] = \left\{ \begin{array} { l l } { 0 , } & { z \leq 0 , } \\ { z I _ { z / r } ( a _ { i } , b _ { i } ) - \mu _ { i } I _ { z / r } ( a _ { i } + 1 , b _ { i } ) , } & { 0 < z < r , } \\ { z - \mu _ { i } , } & { z \geq r } \end{array} \right.
$$

where $I _ { z } ( a , b )$ is the regularized incomplete betafunction.

(iii) Poisson. Posterior Gamma $( a _ { i } , b _ { i } )$ with shape $a _ { i }$ and rate $b _ { i } , \mu _ { i } = a _ { i } / b _ { i }$

$$
c _ { i } ( t , \Delta _ { i } ) ^ { 2 } = \frac { 1 } { 1 + \Delta _ { i } / b _ { i } }
$$

$$
\mathbb { E } _ { t } \left[ ( z - \theta _ { i } ) ^ { + } \right] = \left\{ { \begin{array} { l l } { 0 , \qquad \quad \quad z \leq 0 , } \\ { z \gamma ( a _ { i } , b _ { i } z ) - \mu _ { i } \gamma ( a _ { i } + 1 , b _ { i } z ) , \quad \quad z > 0 } \end{array} } \right.
$$

where $\gamma ( a , x )$ is the lower regularized incomplete gamma function.

(iv) Gamma, known shape $\kappa _ { i } .$ . Posterior InvGamma $( a _ { i } , b _ { i } ) , \mu _ { i } = b _ { i } / ( a _ { i } - 1 )$

$$
c _ { i } ( t , \Delta _ { i } ) ^ { 2 } = \frac { 1 } { 1 + ( \kappa _ { i } / ( a _ { i } - 1 ) ) \Delta _ { i } } \qquad \mathbb { B } _ { t } \left[ ( z - \theta _ { i } ) ^ { + } \right] = \left\{ \begin{array} { l l } { 0 , } & { z \leq 0 , } \\ { z \Gamma ( a _ { i } , b _ { i } / z ) - \mu _ { i } \Gamma ( a _ { i } - 1 , b _ { i } / z ) , } & { z > 0 } \end{array} \right.
$$

where $\Gamma ( a , x )$ is the upper regularized incomplete gamma function.

(v) Lognormal. Posterior lognormal with log-mean $m _ { i } ,$ , log-variance $s _ { i } ^ { 2 } , \mu _ { i } = e ^ { m _ { i } + s _ { i } ^ { 2 } / 2 }$ , and $s _ { i } ^ { 2 } ( \Delta _ { i } ) : =$ (1/�<sup>2</sup>+Δ<sub>�</sub>/�<sup>2</sup>)<sup>−1</sup>:

$$
c _ { i } ( t , \Delta _ { i } ) ^ { 2 } = \frac { 1 - e ^ { - s _ { i } ^ { 2 } ( \Delta _ { i } ) } } { 1 - e ^ { - s _ { i } ^ { 2 } } } \qquad \mathbb { E } _ { t } \left[ ( z - \theta _ { i } ) ^ { + } \right] = \left\{ \begin{array} { l l } { 0 , } & { z \leq 0 , } \\ { z \Phi \big ( \frac { \log z - m _ { i } } { s _ { i } } \big ) - \mu _ { i } \Phi \big ( \frac { \log z - m _ { i } - s _ { i } ^ { 2 } } { s _ { i } } \big ) , } & { z > 0 } \end{array} \right. .
$$

For $\mathrm { ( i ) } { - } ( \mathrm { i v } ) , c _ { i } ( t , \Delta _ { i } ) ^ { 2 }$ has the form (10), so GBB solves (8) exactly by Proposition $2 ( i i )$ . For $( \mathbf { v } ) , c _ { i } ( t , \Delta _ { i } ) ^ { 2 }$ does not have this form in general.

## 4. Theoretical guarantees

The guarantees in $\ S 3$ concern the optimization of the surrogate objective (8), not the performance of TEAk in reducing EOC. In this section, we give asymptotic performance guarantees for TEAk. We begin by showing that TEAk is consistent (EOC(H<sub>�</sub>) converges to zero) and then make additional distributional assumptions to obtain a much stronger result. In particular, with threshold $q ^ { \dagger } ( t )$ , we show that the asymptotic exponential rate at which $\operatorname { E O C } ( \mathcal { H } _ { t } )$ converges to zero, referred to as the posterior convergence rate, is at least half of the best rate achievable by any sequential sampling rule.

Throughout this section, we fix a true mean vector $\pmb { \theta } ^ { 0 } \in \mathcal { T } ^ { I }$ . We write $\mathbb { P } _ { \pmb { \theta } ^ { 0 } }$ and $\mathbb { E } _ { \theta ^ { 0 } }$ for probability and expectation over observations generated from $P _ { i } ( \cdot | \theta _ { i } ^ { 0 } )$ . All convergence statements hold $\mathbb { P } _ { \pmb { \theta } ^ { 0 } }$ -almost surely. Although the true means are fixed, $\operatorname { E O C } ( \mathcal { H } _ { t } )$ remains the Bayesian posterior expectation in (1). Our rate guarantees also hold for the posterior error probability $\mathbb { P } _ { t } ( S _ { k } ( \pmb \theta ) \neq S _ { k } ( \pmb \theta ^ { 0 } ) )$ , the quantity studied in the $k = 1$ analyses of Russo (2020), Qin et al. (2017), and Shang et al. (2020).

These posterior guarantees should not be confused with guarantees for the frequentist EOC $\mathbb { E } _ { \pmb { \theta } ^ { 0 } } [ \mathrm { O C } ( S _ { k } ( \pmb { \mu } ( t ) ) , \pmb { \theta } ^ { 0 } ) ]$ and the frequentist probability of incorrect selection $\mathbb { P } _ { \pmb { \theta } ^ { 0 } } ( S _ { k } ( \pmb { \mu } ( t ) ) \neq S _ { k } ( \pmb { \theta } ^ { 0 } ) )$ . The posterior quantities measure the EOC and posterior error probability conditional on an observed history, while the frequentist quantities average over observation histories. This distinction matters for exponential-rate guarantees: a sampling rule that achieves the optimal posterior convergence rate need not achieve the corresponding frequentist rate. In fact, no single sampling rule can generally guarantee a constant fraction of the best exponential decay rate of the frequentist probability of incorrect selection (likewise frequentist expected opportunity cost) uniformly across problem instances (Goldberger 2026, Degenne 2023).

## 4.1. Consistency

PROPOSITION 4 (Consistency). Run TEAkfor infinitely many rounds with threshold $q ^ { * } ( t )$ or $q ^ { \dagger } ( t )$ . For $q ^ { \dagger } ( t )$ , assume that the minimum in (3) is attained whenever $\mu _ { [ k ] } ( t ) > \mu _ { [ k + 1 ] } ( t )$ . There is a set $\Theta ^ { * } \subseteq \mathcal { T } ^ { I }$ with prior probability one such thatfor every $\pmb { \theta } ^ { 0 } \in \Theta ^ { * }$ , thefollowing hold as $t \to \infty ;$

(i) $N _ { i } ( t ) \to \infty , \mu _ { i } ( t ) \to \theta _ { i } ^ { 0 }$ , and $\nu _ { i } ( t ) \to 0$ for every �.

(ii) $\mathrm { E O C } ( \mathcal { H } _ { t } )  0 .$

(iii) $I f \theta _ { [ k ] } ^ { 0 } > \theta _ { [ k + 1 ] } ^ { 0 } .$ , then $S _ { k } ( { \pmb \mu } ( t ) ) = S _ { k } ( { \pmb \theta } ^ { 0 } )$ for all suficiently large �.

Part (i) says that every item is sampled infinitely often, and consequently its posterior mean converges to the true mean, and its posterior variance to 0. Parts (ii) and (iii) then follow: the posterior EOC converges to 0 and, when the true top-� gap is positive, there is a finite time after which $S _ { k } ( { \pmb \mu } ( t ) ) = S _ { k } ( { \pmb \theta } ^ { 0 } )$

## 4.2. The optimal exponential rate

For the remainder of this section, assume $1 \leq B _ { t } \leq B$ for every � where � is a fixed positive integer, and assume $\theta _ { 1 } ^ { 0 } \geq \cdot \cdot \cdot \geq \theta _ { k } ^ { 0 } > \theta _ { k + 1 } ^ { 0 } \geq \cdot \cdot \cdot \geq \theta _ { I } ^ { 0 }$ . We next establish an upper bound on the posterior convergence rate that any sequential sampling rule can achieve, and characterize the sample allocation that attains this bound. This gives the benchmark against which we will evaluate TEAk. We rely on the general posterior large-deviation analysis of Russo (2020), which we then apply to our top-� EOC objective. Accordingly, we follow the distributional assumptions of Russo (2020).

ASSUMPTION 2 (Bounded exponential-family model). In addition to Assumption 1, $\mathcal { T } = ( L , U )$ with $- \infty < L < U < \infty$ . For every item �, the likelihood has the form

$$
p _ { i } ( y \mid x ) = h _ { i } ( y ) \exp \{ \eta _ { i } ( x ) T _ { i } ( y ) - A _ { i } ( \eta _ { i } ( x ) ) \} , \qquad x \in \mathcal { T } ,
$$

where $T _ { i }$ is strictly increasing and $A _ { i }$ is twice diferentiable. Write $\mathcal { E } _ { i } : = \eta _ { i } ( \mathcal { T } ) = ( \underline { { \eta } } _ { i } , \overline { { \eta } } _ { i } )$ , with $- \infty < \underline { { \eta } } _ { i } <$ $\overline { { \eta } } _ { i } < \infty$ , and let $\widetilde { \pi } _ { i , 0 }$ denote the prior density of $\eta _ { i } ( \theta _ { i } )$ . We assume

$$
\operatorname* { s u p } _ { \eta \in \mathcal { E } _ { i } } | A _ { i } ^ { \prime } ( \eta ) | < \infty , \qquad 0 < \operatorname* { i n f } _ { \eta \in \mathcal { E } _ { i } } \widetilde { \pi } _ { i , 0 } ( \eta ) \leq \operatorname* { s u p } _ { \eta \in \mathcal { E } _ { i } } \widetilde { \pi } _ { i , 0 } ( \eta ) < \infty .
$$

The boundedness assumptions are used in the posterior convergence analysis primarily to facilitate generality across this broad range of families. As Russo (2020, p. 1635) notes, “Analogous results are easily established for certain unbounded conjugate priors.” All five ESCP in Remark 1 satisfy Assumption 2 when their priors are restricted and normalized to a bounded interval whose closure is contained in $\mathcal { T }$

The next result shows that the exponential decay rate of a posterior probability is determined by the empirical sampling proportions. Let $\Delta _ { I } : = \{ w \in \mathbb { R } ^ { I } : w _ { i } \geq 0 , ~ \sum _ { i } w _ { i } = 1 \}$ , and define the empirical sampling proportions by $\widehat { w } _ { i } ( t ) : = N _ { i } ( t ) / \sum _ { j } N _ { j } ( t )$ for $t \geq 2$

LEMMA 2 (Russo (2020, Proposition 5)). Under any sequential sampling rule, for every nonempty open set $O \subseteq { \mathcal { T } } ^ { I }$

$$
- \frac { \log \mathbb { P } _ { t } ( \pmb { \theta } \in O ) } { \sum _ { i } N _ { i } ( t ) } - \operatorname* { i n f } _ { \pmb { x } \in O } \sum _ { i = 1 } ^ { I } \widehat { w } _ { i } ( t ) d _ { i } ( \theta _ { i } ^ { 0 } , x _ { i } ) \longrightarrow 0 .\tag{11}
$$

To interpret the first term in (11), $\{ \pmb { \theta } \in O \}$ is the “bad” event whose probability we want to control. For top-� selection, � is the set of alternative parameters: parameters whose top-� set is diferent from $S _ { k } ( \pmb \theta ^ { 0 } )$ Ignoring ties which have a posterior probability of zero, these are the parameters where some pair of items $( i , j )$ satisfies $i \leq k < j$ , and yet $\theta _ { j } > \theta _ { i }$ . Accordingly, define the alternative space

$$
{ \mathrm { A l t } } ( \pmb { \theta } ^ { 0 } ) : = \bigcup _ { i \leq k < j } \{ \pmb { x } \in \mathcal { T } ^ { I } : x _ { j } > x _ { i } \}
$$

so that $\mathbb { P } _ { t } \left( \pmb { \theta } \in \mathrm { A l t } ( \pmb { \theta } ^ { 0 } ) \right) = \mathbb { P } _ { t } \left( S _ { k } ( \pmb { \theta } ) \neq S _ { k } ( \pmb { \theta } ^ { 0 } ) \right)$ . We will apply Lemma 2 with $O = \mathrm { { A l t } } ( \theta ^ { 0 } )$ . In the second term, for fixed sampling proportions �, the quantity $\begin{array} { r } { \sum _ { i } w _ { i } d _ { i } ( \theta _ { i } ^ { 0 } , x _ { i } ) } \end{array}$ is the expected log-likelihood ratio per observation between the true parameter $\pmb { \theta } ^ { 0 }$ and an alternative �. The infimum over � therefore identifies the alternative parameter that this allocation provides the least evidence to distinguish from the true parameter. For $w \in \Delta _ { I }$ , denote this infimum at $O = \mathrm { A l t } ( \pmb { \theta } ^ { 0 } )$ by $\Gamma ( w ; \pmb \theta ^ { 0 } )$ . Lemma 3 characterizes its structure.

LEMMA 3 (Pairwise representation of Γ). For every $w \in \Delta _ { I }$

$$
\Gamma ( w ; \theta ^ { 0 } ) : = \operatorname* { i n f } _ { x \in \mathrm { A l t } ( \theta ^ { 0 } ) } \sum _ { \ell = 1 } ^ { I } w _ { \ell } d _ { \ell } ( \theta _ { \ell } ^ { 0 } , x _ { \ell } ) = \operatorname* { m i n } _ { i \le k < j } \operatorname* { m i n } _ { x \in [ \theta _ { j } ^ { 0 } , \theta _ { i } ^ { 0 } ] } \{ w _ { i } d _ { i } ( \theta _ { i } ^ { 0 } , x ) + w _ { j } d _ { j } ( \theta _ { j } ^ { 0 } , x ) \} .\tag{12}
$$

Lemma 3 says that $\Gamma ( \boldsymbol { w } ; \boldsymbol { \theta } ^ { 0 } )$ is determined by the pair of items satisfying $i \leq k < j$ whose relative ranking is hardest to learn under the sampling design �. Maximizing over � gives

$$
\Gamma ^ { * } ( \pmb { \theta } ^ { 0 } ) : = \operatorname* { m a x } _ { \pmb { w } \in \Delta _ { I } } \Gamma ( \pmb { w } ; \pmb { \theta } ^ { 0 } ) .\tag{13}
$$

The next proposition establishes that $\Gamma ^ { * } ( \pmb { \theta } ^ { 0 } )$ is the optimal exponential decay rate of posterior EOC and that it is attained by any sampling rule whose empirical proportions converge to a maximizer of (13).

PROPOSITION 5 (Optimal exponential rate of posterior EOC). (i) The maximum in (13) is attained, and $0 < \Gamma ^ { * } ( \theta ^ { 0 } ) < \infty .$

(ii) Every sampling rule satisfies

$$
\operatorname* { l i m } _ { t \to \infty } \operatorname* { l p } _ { - } \frac { \log \mathrm { E O C } ( \mathcal { H } _ { t } ) } { \sum _ { i } N _ { i } ( t ) } \le \Gamma ^ { * } ( \pmb { \theta } ^ { 0 } ) .
$$

(iii) If the sampling proportions converge to a maximizer of (13), then

$$
\operatorname* { l i m } _ { t \to \infty } - \frac { \log \mathrm { E O C } ( \mathcal { H } _ { t } ) } { \sum _ { i } N _ { i } ( t ) } = \Gamma ^ { * } ( \pmb { \theta } ^ { 0 } ) .
$$

We next show that, with threshold $q ^ { \dagger } ( t )$ , TEAk’s posterior convergence rate is at least $\Gamma ^ { * } ( \pmb { \theta } ^ { 0 } ) / 2$

## 4.3. The rate of TEAk

To determine the rate TEAk achieves, it is suficient to characterize the sample proportions produced asymptotically. First, we do this assuming the threshold $q ( t )$ converges to some point $q \in ( \theta _ { k + 1 } ^ { 0 } , \theta _ { k } ^ { 0 } )$ . For such a �, define the proportions

$$
w _ { i } ( q ) : = \frac { 1 / d _ { i } ( \theta _ { i } ^ { 0 } , q ) } { \sum _ { j = 1 } ^ { I } 1 / d _ { j } ( \theta _ { j } ^ { 0 } , q ) } , \qquad i = 1 , \ldots , I .\tag{14}
$$

THEOREM 1 (Allocation and rate for a convergent threshold). Assume $q ( t ) \to q _ { 0 } \in ( \theta _ { k + 1 } ^ { 0 } , \theta _ { k } ^ { 0 } )$ . Then

(i) $\widehat { w } _ { i } ( t ) \to w _ { i } ( q _ { 0 } )$ for every �,

(ii) $\begin{array} { r } { - \frac { \log \mathrm { E O C } ( \mathcal { H } _ { t } ) } { \sum _ { i } N _ { i } ( t ) } \longrightarrow \Gamma \big ( \pmb { w } ( q _ { 0 } ) ; \pmb { \theta } ^ { 0 } \big ) , } \end{array}$

(iii) $\begin{array} { r } { \Gamma ( w ( q _ { 0 } ) ; \theta ^ { 0 } ) \geq \big ( \sum _ { i = 1 } ^ { I } 1 / d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } ) \big ) ^ { - 1 } . } \end{array}$

Part (i) says that the asymptotic sampling proportions of TEAk are described by (14). To prove this, we show that

$$
- \frac { \log D _ { i , t } ( d ) } { N _ { i } ( t ) } \longrightarrow d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } ) \qquad \mathrm { f o r ~ e v e r y ~ } i \mathrm { ~ a n d ~ } d = 0 , \dots , B - 1 .
$$

Thus, $d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } )$ gives the exponential decay rate of the marginal decrease in the surrogate per sample of item �. Since TEAk allocates samples to the items with the largest marginal decreases (the argument under DPBB is a bit more technical but also holds), it asymptotically samples the items with the smallest $N _ { i } ( t ) d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } )$ which leads to balancing $N _ { i } ( t ) d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } )$ across all items. Therefore, the sampling proportions converge to $w _ { i } ( q _ { 0 } )$ , which makes $w _ { i } ( q _ { 0 } ) d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } )$ equal across items. Part (ii) says that TEAk’s posterior convergence rate is then obtained by plugging these proportions into Γ. Part (iii) gives a lower bound on TEAk’s rate, and we establish the one-half guarantee by comparing this lower bound, rather than $\Gamma ( \pmb { w } ( q _ { 0 } ) ; \pmb { \theta } ^ { 0 } )$ itself, with the optimal rate $\Gamma ^ { * } ( \pmb { \theta } ^ { 0 } )$ . This lower bound motivates choosing the threshold that makes it tightest,

$$
q ^ { \dag } \in \underset { \theta _ { k + 1 } ^ { 0 } < q < \theta _ { k } ^ { 0 } } { \arg \operatorname* { m i n } } \sum _ { i = 1 } ^ { I } \frac { 1 } { d _ { i } ( \theta _ { i } ^ { 0 } , q ) } .
$$

Replacing the true means with the posterior means results in our choice of threshold $q ^ { \dagger } ( t )$ in (3). This parallels our earlier choice of $q ^ { * } ( t )$ : there we minimize an upper bound on the current EOC, whereas here we maximize a lower bound on its exponential decay rate.

It remains to show that $q ^ { \dagger } ( t )  q ^ { \dagger }$ . To establish this convergence, we assume the minimizer is unique.

ASSUMPTION 3 (Unique threshold). For every $\pmb { x } \in \mathcal { T } ^ { I }$ with $x _ { [ k ] } \ > \ x _ { [ k + 1 ] }$ , the function $q \mapsto$ $\textstyle \sum _ { i = 1 } ^ { I } 1 / d _ { i } ( x _ { i } , q )$ has a unique minimizer on $\left( x _ { [ k + 1 ] } , x _ { [ k ] } \right)$ .

REMARK 3. Each likelihood family in Remark 1 satisfies Assumption 3.

LEMMA 4 (Convergence of $q ^ { \dagger } ( t ) )$ . Run Algorithm 3 with threshold $q ^ { \dagger } ( t )$ . Then $q ^ { \dagger } ( t )  q ^ { \dagger }$

By Lemma 4, Theorem 1 applies with $q _ { 0 } = q ^ { \dagger }$ . We further show that $\begin{array} { r } { \left( \sum _ { i = 1 } ^ { I } 1 / d _ { i } ( \theta _ { i } ^ { 0 } , q ^ { \dagger } ) \right) ^ { - 1 } \geq \Gamma ^ { * } ( \theta ^ { 0 } ) / 2 . } \end{array}$ which yields our main theoretical result.

THEOREM 2 (Exponential rate of TEAk). Run Algorithm 3 with threshold $q ^ { \dagger } ( t )$ . Then

$$
\operatorname* { l i m } _ { t \to \infty } - \frac { \log \mathrm { E O C } ( \mathcal { H } _ { t } ) } { \sum _ { i } N _ { i } ( t ) } \geq \frac { 1 } { 2 } \Gamma ^ { * } ( \pmb { \theta } ^ { 0 } ) .
$$

Thus, the posterior convergence rate of TEAk is at least half the optimal posterior convergence rate. The conclusions in Proposition 5 and in Theorems 1 and 2 also hold when $\operatorname { E O C } ( \mathcal { H } _ { t } )$ is replaced by the posterior error probability $\mathbb { P } _ { t } ( S _ { k } ( \pmb \theta ) \neq S _ { k } ( \pmb \theta ^ { 0 } ) )$ .

## 4.4. Numerical illustration

We illustrate Theorems 1 and 2 on a Gaussian instance with $I = 2 0 , k = 5$ , likelihood variance of 1, and true means $\theta _ { i } ^ { 0 } = - ( i - 1 ) / 2$ so that the gap between consecutive means is 0.5. On this instance $\Gamma ^ { * } ( \pmb { \theta } ^ { 0 } ) = 0 . 0 2 4 3$ and $q ^ { \dagger } = - 2 . 2 5$ , the midpoint between $\theta _ { k } ^ { 0 }$ and $\theta _ { k + 1 } ^ { 0 }$ . The rate of the threshold allocation is $\Gamma ( w ( q ^ { \dagger } ) ; \pmb { \theta } ^ { 0 } ) =$ $0 . 5 7 \Gamma ^ { * } ( \pmb { \theta } ^ { 0 } )$ and the lower bound on it in Theorem 1(iii) is $0 . 5 3 \Gamma ^ { * } ( \theta ^ { 0 } )$ .

We run TEAk with $q ^ { \dagger } ( t )$ on this instance with $B _ { t } = 1$ , a sampling budget of 20,000, and a $N ( 0 , 1 0 ^ { 2 } )$ prior over 50 replications. We also run two fixed allocations: the oracle $\pmb { w } ^ { * }$ that maximizes (13) using the true means, and equal allocation. Figure 3 shows the results.

The left panel shows the average sampling proportions at the end of the run. TEAk’s proportions coincide with $w ( q ^ { \dag } )$ , as guaranteed by Theorem 1(i), and are a bit of from �<sup>∗</sup>: TEAk puts 83% of the budget on the �th and (�+1)th items, whereas $\pmb { w } ^ { * }$ puts 78%. The middle panel shows an approximation<sup>1</sup> of $- \log \operatorname { E O C } ( \mathcal { H } _ { t } ) / t$ along each replication as a fraction of $\Gamma ^ { * } ( \pmb { \theta } ^ { 0 } )$ . For TEAk, it decreases towards $\Gamma ( \pmb { w } ( q ^ { \dagger } ) ; \pmb { \theta } ^ { 0 } )$ in line with Theorem 1(ii), and for the oracle allocation it approaches $\Gamma ^ { * } ( \pmb { \theta } ^ { 0 } )$ . The right panel shows the convergence $q ^ { \dagger } ( t )  q ^ { \dagger }$ on each replication.

![](images/df55acfb79458e39dc9ad774b37b40e757e00620cc4f0dde3f205b72793c1c5f.jpg)

![](images/85e6d037ebf659cb5777a4844e3e9f4d161788fba9981ffc6d7f9229d5eb78d9.jpg)

![](images/85132d7e66779e3ef9d8e85c483ff91308aad36ec1ee7e6755116ba0382f2249.jpg)  
Figure 3 Left: TEAk’s sampling proportions at $t = 2 0 { , } 0 0 0 ,$ averaged over replications, as well as $w ( q ^ { \dagger } )$ and $\pmb { w } ^ { * }$ . Middle: $- \log \mathrm { E O C } ( \mathcal { H } _ { t } ) / ( t \Gamma ^ { * } ( \pmb { \theta } ^ { 0 } ) )$ for TEAk, the oracle allocation, and equal allocation on each of the 50 replications (median in bold). Right: the threshold $q ^ { \dagger } ( t )$ on each of the 50 replications (median in bold).

## 5. Numerical experiments

In this section, we evaluate TEAk’s performance numerically by comparing it to the most competitive algorithms in the literature. For this comparison, we use a batch size of 1, as most benchmarks are designed for the fully sequential problem. §5.3 then turns to the batched setting and compares the performance of TEAk across batch sizes. Throughout, TEAk and $\operatorname { T E A k } \mathbf { - } q ^ { \dag } \left( t \right)$ denote Algorithm 3 with the threshold $q ^ { * } ( t )$ and $q ^ { \dagger } ( t )$ respectively. Across all settings, the two are nearly identical in both performance and allocation.

## 5.1. Setup

We consider $I \in \{ 2 0 , 5 0 , 1 0 0 \}$ , with a sampling budget of up to 1,000· � for each setting. For $I = 2 0$ we consider $k \in \{ 1 , 3 , 5 , 1 0 \}$ , and for $I \in \{ 5 0 , 1 0 0 \}$ we consider $k \in \{ 1 , 5 , 1 0 , 2 0 \}$ }. On each $( I , k )$ pair, we consider four likelihood families: Gaussian, exponential, Bernoulli, and lognormal. For each of the four families, we consider three generative distributions for their true means �, given in Table 1. These ranges, although somewhat arbitrary, are targeted so that the dificulty of the identification problem is approximately the same across the likelihood families. The results are generally the same across these three generative distributions of the true means, so we report the results in the middle column in this section, giving the rest in Appendix D.

Table 1 The four likelihood families and the three distributions used to generate the true means.
<table><tr><td>Likelihood  $Y _ { i , n } \mid \boldsymbol { \theta } _ { i }$ </td><td colspan="3">Distribution of true means  $\theta _ { i }$ </td></tr><tr><td> $N ( \theta _ { i } , 1 )$ </td><td>Unif(−2,2)</td><td> $\mathrm { U n i f } ( - 1 , 1 )$ </td><td> $\operatorname { U n i f } ( - 0 . 5 , 0 . 5 )$ </td></tr><tr><td> $\textstyle \operatorname { E x p } ( 1 / \theta _ { i } )$ </td><td>Unif(1,9)</td><td>Unif(1,5)</td><td>Unif(1,3)</td></tr><tr><td>Bern(θi)</td><td>Unif(0.05,0.95)</td><td>Unif(0.2,0.8)</td><td>Unif(0.35,0.65)</td></tr><tr><td> $\mathrm { L o g n o r m a l } ( \log \theta _ { i } { - } 1 / 2 , 1 )$ </td><td>Unif(1,9)</td><td>Unif(1,5)</td><td>Unif(1,3)</td></tr></table>

Fair comparison. We use three measures to ensure each method is on equal footing. First, within a replication, all methods have the same true means and the outcomes come from the same stream, so the �th observation of an item � is identical across methods. Second, to ensure fairness across Bayesian and frequentist methods, the Bayesian methods all use the same uninformative (Jefreys) prior for each distribution family: the flat prior for the Gaussian mean, $\propto 1 / \theta$ for the exponential mean, Beta(1/2, 1/2) for Bernoulli, and the flat prior on log � for the lognormal mean (Jefreys 1946). Third, since some benchmark methods require a warm-up, each algorithm receives a warm-up of $n _ { 0 } = 3$ samples per item.

Benchmarks. We compare against the following benchmarks, all of which aim to maximize PCS unless otherwise specified.

– Gaussian-Specificfrom the Ranking and Selection Literature: For � = 1 AOAP from Peng et al. (2018) and for � > 1 AOAm from Zhang et al. (2023) and EMIp from Liu et al. (2024). For the top-� EOC setting, EOC-m from Gao and Chen (2015).

– General � = 1: Top two Thompson sampling (TTTS) of Russo (2020) with $\beta = 1 / 2$ and its variant that does not rely on a tuning parameter, TTTS-IDS from Qin and You (2026). BOLD (Chen and Ryzhov 2023).

– General � > 1 from the Best Arm(s) Identification Literature: The top-� Thompson sampling approaches TS-PPS-IDS and TS-KKT(.5)-IDS from You et al. (2023), Qin and You (2026), KL-LUCB from Kaufmann and Kalyanakrishnan (2013), and gap-e from Bubeck et al. (2013).

– Uniform: Equal allocation across all items.

Metrics. Since most benchmarks aim to maximize PCS, we report both EOC and PCS, estimated from the 10,000 replications. Consistent with previous literature and the discussion in the introduction, the two are closely aligned. Since none of the methods other than gap-e depend on the total budget, we have their EOC and PCS for every fixed budget up to 1,000·�.

## 5.2. Results

To keep the paper focused, we report PCS at the largest budget (1,000·�), and for EOC, we report performance curves across budgets for the Gaussian setting. The PCS and EOC performance curves for all settings are in Appendix D.

% difference in PCS from TEAk at budget 1000 ·I (positive = benchmark better), symlog scale, linear between -0.1 and 0.1

![](images/9e12489e3d702aa044cdc8f12070f6edbb7f685066b15a3f7791333f50b6c50d.jpg)  
Figure 4 % diference in PCS between benchmarks and TEAk at budget 1,000·� (positive means benchmark is better).

![](images/80d33acc23b46e65c14aea5645698bd56ed3177d9f8dfe2bff1315dda358d6e0.jpg)  
Figure 5 The distribution across all settings in Figure 4 for the four strongest benchmarks and $\operatorname { T E A k } \mathbf { - } q ^ { \dag } ( t )$

PCS. Figure 4 shows the PCS of each benchmark relative to TEAk for all configurations. TEAk generally has the highest PCS, and when it does not, it is within a few percent of the best. TEAk’s biggest advantage over the benchmarks is on larger problems (� = 100) and larger �. To show more clearly how TEAk improves over the best benchmarks and how it compares with $\operatorname { T E A k } \mathbf { - } q ^ { \dag } \left( t \right)$ , Figure 5 takes the four best-performing benchmarks, together with $\operatorname { T E A k } \mathbf { - } q ^ { \dag } \left( t \right)$ , and plots the distribution over all configurations of their PCS relative to TEAk. Each benchmark falls short of TEAk by 5% or more in some configurations, while none beats it by more than 0.2% in any configuration. In addition, TEAk outperforms $\operatorname { T E A k } \mathbf { - } q ^ { \dag } \left( t \right)$ in 36 of the 48 configurations, and in every configuration the PCS of TEAk- $\cdot q ^ { \dagger } ( t )$ is between 0.8% below and 0.3% above that of TEAk.

EOC over the whole budget. Figure 6 plots EOC on a log scale against cumulative samples spent for the Gaussian family and each (�, �) pair. $\mathrm { A t } \ k = 1$ , the most competitive benchmarks are the TTTS variants, and for general � the most competitive are gap-e, KL-LUCB, and the top-� Thompson sampling methods. Which benchmark is best (or near-best) varies across configurations and varies with the budget within a configuration. TEAk is the exception: in every configuration, its EOC is the lowest, or close to it, across all budgets. The plots for the other three sampling families show a similar picture.

![](images/3f438f36cf4d2fef4111b4971531872086a7e6b26bedf79cb07b8edb35adc283.jpg)  
Figure 6 Gaussian: EOC (log scale) against budget (max budget = 1,000·�). Rows vary the problem size � and columns vary �.

## 5.3. Batching

We now investigate TEAk’s performance in the batched setting, one of its main motivations. We compare against BatchSAR (Jun et al. 2016), a version of the successive accepts and rejects algorithm of Bubeck et al. (2013) designed specifically for fixed-budget top-� identification with batched sampling. We consider the Gaussian and Bernoulli families with $I \in \{ 2 0 , 5 0 , 1 0 0 \}$ , � = 5, and the middle column of Table 1. Since BatchSAR’s schedule depends on the total budget, we cannot run it with a large budget and read of its performance at every point along the way, so we fix a budget of 50·� and consider batch sizes of 1%, 2%, 5%, and 10% of the budget. Figure 7 plots PCS, the objective BatchSAR is designed to maximize. Not only does TEAk have a higher PCS at the smallest batch size, but it also decreases less as the batch size grows: from batches of 1% to 10% of the budget, its PCS falls by up to 12%, and BatchSAR’s by up to 29%.

![](images/bd1eada229354902bea0fa1b0673c0a452c145c13b485ba398b1539d766996bd.jpg)  
Figure 7 PCS at the final budget of 50·� against the batch size for the Gaussian (top) and Bernoulli (bottom) families with � = 5.

## 6. Discussion

We give a simple and interpretable algorithm for top-� selection that accommodates batching in a principled way and can be applied to a wide range of sampling distribution families. We prove that it is consistent and that with the threshold $q ^ { \dagger } ( t )$ , its posterior convergence rate is at least half of the best rate achievable by any sampling rule. In simulation, it is competitive with the state-of-the-art methods across likelihood families, problem sizes, values of �, and batch sizes.

## References

Audibert JY, Bubeck S, Munos R (2010) Best arm identification in multi-armed bandits. Proceedings of the 23rd Conference on Learning Theory (COLT).

Bubeck S, Wang T, Viswanathan N (2013) Multiple identifications in multi-armed bandits. Proceedings ofthe 30th International Conference on Machine Learning (ICML), 258–265.

Chen CH, He D, Fu M, Lee LH (2008) Eficient simulation budget allocation for selecting an optimal subset. INFORMS Journal on Computing 20:579–595.

Chen CH, Lee LH (2010) Stochastic Simulation Optimization: An Optimal Computing Budget Allocation (World Scientific).

Chen CH, Lin J, Y¨ucesan E, Chick SE (2000) Simulation budget allocation for further enhancing the eficiency of ordinal optimization. Discrete Event Dynamic Systems 10:251–270.

Chen HC, Dai L, Chen CH, Y¨ucesan E (1997) New development of optimal computing budget allocation for discrete event simulation. Proceedings of the 29th Winter Simulation Conference, 334–341.

Chen Y, Ryzhov IO (2019) Complete expected improvement converges to an optimal budget allocation. Advances in Applied Probability 51:209–235.

Chen Y, Ryzhov IO (2023) Balancing optimal large deviations in sequential selection. Management Science 69:3457– 3473.

Chick SE, Branke J, Schmidt C (2010) Sequential sampling to myopically maximize the expected value of information. INFORMS Journal on Computing 22:71–80.

Chick SE, Gans N, Yapar O (2022) Bayesian sequential learning for clinical trials of multiple correlated medical <sup>¨</sup> interventions. Management Science 4919–4938.

Chick SE, Inoue K (2001) New two-stage and sequential procedures for selecting the best simulated system. Operations Research 49:732–743.

Degenne R (2023) On the existence of a complexity in fixed budget bandit identification. Proceedings ofthe 36th Annual Conference on Learning Theory.

Diaconis P, Ylvisaker D (1979) Conjugate priors for exponential families. The Annals of Statistics 7:269–281.

Doob JL (1949) Application of the theory of martingales. Actes du Colloque International Le Calcul des Probabilites et´ ses Applications (Lyon, 28 juin–3 juillet 1948), 23–27 (CNRS).

Frazier PI, Powell WB, Dayanik S (2008) A knowledge-gradient policy for sequential information collection. SIAM Journal on Control and Optimization 47:2410–2439.

Frazier PI, Powell WB, Dayanik S (2009) The knowledge-gradient policy for correlated normal beliefs. INFORMS Journal on Computing 21:599–613.

Fu MC, Glover FW, April J (2005) Simulation optimization: A review, new developments, and applications. Proceedings of the Winter Simulation Conference, 83–95.

Gabillon V, Ghavamzadeh M, Lazaric A, Bubeck S (2011) Multi-bandit best arm identification. Advances in Neural Information Processing Systems (NeurIPS).

Gao S, Chen W (2015) Eficient subset selection for the expected opportunity cost. Automatica 59:19–26.

Gao S, Chen W (2016) A new budget allocation framework for selecting top simulated designs. IIE Transactions 48:855–863.

Glynn PW, Juneja S (2004) A large deviations perspective on ordinal optimization. Proceedings ofthe 2004 Winter Simulation Conference, 577–585.

Goldberger M (2026) Fundamental limitations of fixed-budget best-arm identification. arXiv:2607.11635 .

Goldberger M, Rudi N (2026) Selecting the best few with human evaluators: Adaptively combining ratings and pairwise comparisons, working paper.

He D, Chick SE, Chen CH (2007) Opportunity cost and OCBA selection procedures in ordinal optimization for a fixed number of alternative systems. IEEE Transactions on Systems, Man, and Cybernetics, Part C 37:951–961.

Hong LJ, Fan W, Luo J (2021) Review on ranking and selection: A new perspective. Frontiers of Engineering Management 8:321–343.

Jamieson K, Talwalkar A (2016) Non-stochastic best arm identification and hyperparameter optimization. Proceedings ofthe 19th International Conference on Artificial Intelligence and Statistics, 240–248.

Jefreys H (1946) An invariant form for the prior probability in estimation problems. Proceedings of the Royal Society of London. Series A. Mathematical and Physical Sciences 186:453–461.

Jones DR, Schonlau M, Welch WJ (1998) Eficient global optimization of expensive black-box functions. Journal of Global Optimization 13:455–492.

Jourdan M, Degenne R, Baudry D, de Heide R, Kaufmann E (2022) Top two algorithms revisited. Advances in Neural Information Processing Systems (NeurIPS).

Ju N, Hu D, Henderson A, Hong L (2019) A sequential test for selecting the better variant: Online a/b testing, adaptive allocation, and continuous monitoring. Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining, 492–500.

Jun KS, Jamieson K, Nowak R, Zhu X (2016) Top arm identification in multi-armed bandits with batch arm pulls. Proceedings of the 19th International Conference on Artificial Intelligence and Statistics (AISTATS), 139–148.

Kalyanakrishnan S, Tewari A, Auer P, Stone P (2012) PAC subset selection in stochastic multi-armed bandits. Proceedings of the 29th International Conference on Machine Learning (ICML), 655–662.

Karnin Z, Koren T, Somekh O (2013) Almost optimal exploration in multi-armed bandits. Proceedings of the 30th International Conference on Machine Learning (ICML), 1238–1246.

Kaufmann E, Cappe O, Garivier A (2016) On the complexity of best-arm identification in multi-armed bandit models.´ Journal ofMachine Learning Research 17:1–42.

Kaufmann E, Kalyanakrishnan S (2013) Information complexity in bandit subset selection. Proceedings ofthe Conference on Learning Theory (COLT).

Li L, Jamieson K, DeSalvo G, Rostamizadeh A, Talwalkar A (2018) Hyperband: A novel bandit-based approach to hyperparameter optimization. Journal ofMachine Learning Research 18:1–52.

Li Z, Fan W, Hong LJ (2025) The (surprising) sample optimality of greedy procedures for large-scale ranking and selection. Management Science 71:1238–1259.

Liu H, Xing Z, Xiao H, Chew EP (2024) Marginal improvement procedures for top-� selection. Automatica 170. Miller JW (2018) A detailed treatment of Doob’s theorem. arXiv:1801.03122 .

Peng Y, Chong EKP, Chen CH, Fu MC (2018) Ranking and selection as stochastic control. IEEE Transactions on Automatic Control 63:2359–2373.

Qin C, Klabjan D, Russo D (2017) Improving the expected improvement algorithm. Advances in Neural Information Processing Systems, volume 30.

Qin C, You W (2026) Dual-directed algorithm design for eficient pure exploration. Operations Research 74:1104–1125.

Robertson DS, Lee KM, Lopez-Kolkovska BC, Villar SS (2023) Response-adaptive randomization in clinical trials:´ From myths to practical considerations. Statistical Science 38:185–208.

Russo D (2020) Simple bayesian algorithms for best-arm identification. Operations Research 68:1625–1647.

Ryzhov IO (2016) On the convergence rates of expected improvement methods. Operations Research 64:1515–1528.

Scott SL (2015) Multi-armed bandit experiments in the online service economy. Applied Stochastic Models in Business and Industry 31:37–45.

Shang X, de Heide R, Menard P, Kaufmann E, Valko M (2020) Fixed-confidence guarantees for bayesian best-arm identification. Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, 1823–1832.

Snoek J, Larochelle H, Adams RP (2012) Practical bayesian optimization of machine learning algorithms. Advances in Neural Information Processing Systems 25.

Villar SS, Bowden J, Wason J (2015) Multi-armed bandit models for the optimal design of clinical trials: Benefits and challenges. Statistical Science 30:199–215.

You W, Qin C, Wang Z, Yang S (2023) Information-directed selection for top-two algorithms. Proceedings of the 36th Conference on Learning Theory (COLT).

Zhang G, Peng Y, Zhang J, Zhou E (2023) Asymptotically optimal sampling policy for selecting top-� alternatives. INFORMS Journal on Computing 35:1261–1285.

## Appendix A: Proofs for §3

Proof of Lemma $I ( i )$ Fix any realized $\pmb { \theta } \in \mathbb { R } ^ { I }$ and any $q \in \mathbb { R }$ . Then

$$
\begin{array} { r l } { \operatorname { O C } ( S _ { k } ( \mu ( t ) ) , \theta ) = \displaystyle \sum _ { i \in S _ { k } ( \theta ) } \theta _ { i } - \displaystyle \sum _ { i \in S _ { k } ( \mu ( t ) ) } \theta _ { i } } & { } \\ { \overset { ( a ) } { = } } & { \displaystyle \sum _ { i \in S _ { k } ( \theta ) \backslash S _ { k } ( \mu ( t ) ) } \theta _ { i } - \displaystyle \sum _ { i \in S _ { k } ( \mu ( t ) ) \backslash S _ { k } ( \theta ) } \theta _ { i } } \\ { \overset { ( b ) } { = } } & { \displaystyle \sum _ { i \in S _ { k } ( \theta ) \backslash S _ { k } ( \mu ( t ) ) } ( \theta _ { i } - q ) + \displaystyle \sum _ { i \in S _ { k } ( \mu ( t ) ) \backslash S _ { k } ( \theta ) } ( q - \theta _ { i } ) } \\ { \overset { ( b ) } { \leq } } & { \displaystyle ( \theta _ { i } - \sum _ { i } \theta _ { i } ) ( \theta _ { i } - q ) ^ { + } + \displaystyle \sum _ { i \in S _ { k } ( \mu ( t ) ) \backslash ( q - \theta _ { i } ) } ( q - \theta _ { i } ) ^ { + } } \\ { \leq } & { \displaystyle \sum _ { i \notin S _ { k } ( \mu ( t ) ) } ( \theta _ { i } - q ) ^ { + } + \displaystyle \sum _ { i \in S _ { k } ( \mu ( t ) ) } ( q - \theta _ { i } ) ^ { + } , } \end{array}
$$

where (a) holds because indices in $S _ { k } ( \pmb { \mu } ( t ) ) \cap S _ { k } ( \pmb { \theta } )$ cancel, and (b) holds because $\vert S _ { k } ( \pmb \mu ( t ) ) \setminus S _ { k } ( \pmb \theta ) \vert =$ $\vert S _ { k } ( \pmb \theta ) \backslash S _ { k } ( \pmb \mu ( t ) ) \vert$ . Taking posterior expectation proves the claim. □

Proof of Lemma 1(ii) Define

$$
G _ { t } ( \boldsymbol { q } ) : = \mathbb { E } _ { t } \left[ \sum _ { i \in S _ { k } ( \mu ( t ) ) } ( \boldsymbol { q } - \boldsymbol { \theta _ { i } } ) ^ { + } + \sum _ { i \notin S _ { k } ( \mu ( t ) ) } ( \boldsymbol { \theta } _ { i } - \boldsymbol { q } ) ^ { + } \right] .
$$

By Assumption 1 each posterior CDF $F _ { i , t }$ is continuous on $\mathcal { T }$ , so diferentiating under the expectation gives

$$
\frac { d } { d q } \mathbb { E } _ { t } [ ( q - \theta _ { i } ) ^ { + } ] = F _ { i , t } ( q ) , \qquad \frac { d } { d q } \mathbb { E } _ { t } [ ( \theta _ { i } - q ) ^ { + } ] = F _ { i , t } ( q ) - 1 .
$$

Hence

$$
G _ { t } ^ { \prime } ( q ) = \sum _ { i \in S _ { k } ( \mu ( t ) ) } F _ { i , t } ( q ) + \sum _ { i \notin S _ { k } ( \mu ( t ) ) } ( F _ { i , t } ( q ) - 1 ) = \sum _ { i = 1 } ^ { I } F _ { i , t } ( q ) - ( I - k ) .
$$

Additionally, by Assumption 1 each posterior density is positive on $\mathcal { T }$ , so each $F _ { i , t }$ is strictly increasing on $\mathcal { T }$ Thus $G _ { t } ^ { \prime }$ is continuous and strictly increasing on $\mathcal { T }$ , which gives convexity of $G _ { t }$ . Taking limits, $F _ { i , t } ( q ) \to 0$ as $q \to - \infty$ and $F _ { i , t } ( q ) \to 1$ as � → +∞, so $G _ { t } ^ { \prime } ( q ) \to - ( I - k ) < 0$ as $q \to - \infty$ and $G _ { t } ^ { \prime } ( q ) \to k > 0$ as $q \to + \infty$ By the intermediate value theorem, $G _ { t } ^ { \prime }$ has a unique zero $q ^ { * } ( t ) \in \mathcal { T }$ , characterized by $\begin{array} { r } { \sum _ { i } F _ { i , t } ( q ^ { * } ( t ) ) = I { - } k } \end{array}$ equivalently (2), and $q ^ { * } ( t )$ is the unique minimizer of $G _ { t }$ . □

Proof of Proposition 1 Starting with $r = 0 ,$ , set $V _ { 0 } ( 0 ) = 0$ and $V _ { 0 } ( b ) = + \infty$ for $b = 1 , \dots , B _ { t }$ . For the inductive step, suppose $V _ { r - 1 } ( \cdot )$ is already known and consider the $( r , b )$ subproblem. Because the objective in (8) is separable across items, fixing item $r { } _ { \mathrm { { s } } }$ allocation at some $\delta \in \{ 0 , \ldots , b \}$ splits the subproblem into two pieces: the cost $\ell _ { r , t } ( \delta )$ incurred by item �, plus the cost of optimally allocating exactly the residual $b { - } \delta$ samples among items $1 , \ldots , r { - } 1$ , which is $V _ { r - 1 } ( b - \delta )$ . Therefore, for each $( r , b ) \in \{ ( x , y ) : 1 \leq x \leq I , 0 \leq y \leq B _ { t } \}$ the optimal value function is given by

$$
V _ { r } ( b ) = \operatorname* { m i n } _ { \delta \in \{ 0 , \ldots , b \} } \{ V _ { r - 1 } ( b - \delta ) + \ell _ { r , t } ( \delta ) \} , \qquad \Delta _ { r } ^ { * } ( b ) \in \mathrm { ~ a r g \operatorname* { m i n } _ { \delta \in \{ 0 , \ldots , b \} } ~ } \{ V _ { r - 1 } ( b - \delta ) + \ell _ { r , t } ( \delta ) \} .
$$

Once we solve this for each $( r , b )$ , an optimal allocation vector $\Delta ^ { t }$ is recovered by backtracking. $\Delta _ { I } ^ { t } = \Delta _ { I } ^ { * } ( B _ { t } )$ , leaving a residual budget $B _ { t } - \Delta _ { I } ^ { t }$ for items $1 , \ldots , I - 1$ . Then $\Delta _ { I - 1 } ^ { t } = \Delta _ { I - 1 } ^ { * } ( B _ { t } - \Delta _ { I } ^ { t } )$ and so on down to item 1:

$$
\Delta _ { r } ^ { t } = \Delta _ { r } ^ { * } \Big ( B _ { t } - \sum _ { j = r + 1 } ^ { I } \Delta _ { j } ^ { t } \Big ) , \qquad r = I , I - 1 , \ldots , 1 .
$$

By construction, backtracking recovers an allocation whose components sum to $B _ { t }$ and whose objective value is $V _ { I } ( B _ { t } )$ . It is therefore a global optimizer of (8). The total runtime is the cost of precomputing the $I ( B _ { t } { + } 1 )$ values $\ell _ { i , t } ( d )$ which is $O ( I B _ { t } C _ { \ell } )$ and the cost of filling in the $O ( I B _ { t } )$ DP states (each requiring a minimization over at most $B _ { t } + 1$ values of �) which is $O ( I B _ { t } ^ { 2 } )$ ). □

Proof of Proposition 2(i) For any allocation �, the improvement relative to assigning no samples in the current round is

$$
\sum _ { j = 1 } ^ { I } \ell _ { j , t } ( 0 ) - \sum _ { j = 1 } ^ { I } \ell _ { j , t } ( \Delta _ { j } ) = \sum _ { j = 1 } ^ { I } \sum _ { h = 0 } ^ { \Delta _ { j } - 1 } \left[ \ell _ { j , t } ( h ) - \ell _ { j , t } ( h + 1 ) \right] = \sum _ { j = 1 } ^ { I } \sum _ { h = 0 } ^ { \Delta _ { j } - 1 } D _ { j , t } ( h ) ,
$$

where the inner sum is zero when $\Delta _ { j } = 0$ . Minimizing $\Sigma _ { j } \ell _ { j , t } ( \Delta _ { j } )$ is therefore equivalent to maximizing

$$
G ( \Delta ) : = \sum _ { j = 1 } ^ { I } \sum _ { h = 0 } ^ { \Delta _ { j } - 1 } D _ { j , t } ( h )
$$

subject to $\begin{array} { r } { \sum _ { j } \Delta _ { j } = B _ { t } } \end{array}$ . Let $\Delta ^ { G }$ denote the greedy allocation and � any other feasible allocation. We will show $G ( \Delta ^ { G } ) \geq G ( \Delta )$ . If $\Delta = \Delta ^ { G }$ , there is nothing to prove. Otherwise, since both allocations sum to $B _ { t }$ , there exist indices $i , j$ with $\Delta _ { i } > \Delta _ { i } ^ { G }$ and $\Delta _ { j } < \Delta _ { j } ^ { G }$ . We show that one can transfer a sample from � to � without decreasing $G .$

Because $\Delta _ { i } > \Delta _ { i } ^ { G }$ , the term $D _ { i , t } ( \Delta _ { i } - 1 )$ appears in $G ( \Delta )$ . Since $\Delta _ { i } - 1 \geq \Delta _ { i } ^ { G }$ , (9) gives

$$
D _ { i , t } ( \Delta _ { i } - 1 ) \leq D _ { i , t } ( \Delta _ { i } ^ { G } ) .
$$

Because $\Delta _ { j } < \Delta _ { j } ^ { G }$ , the term $D _ { j , t } ( \Delta _ { j } )$ does not appear in $G ( \Delta )$ . Since $\Delta _ { j } \leq \Delta _ { j } ^ { G } - 1 , ( 9 )$ gives

$$
D _ { j , t } ( \Delta _ { j } ) \geq D _ { j , t } ( \Delta _ { j } ^ { G } - 1 ) .
$$

Consider the greedy step at which the $( \Delta _ { j } ^ { G } )$ )th sample was assigned to item $j .$ Before that step, $j$ had received $\Delta _ { j } ^ { G } - 1$ samples and � had received at most $\Delta _ { i } ^ { G }$ samples. The marginal gain of allocating to $j$ at that step was $D _ { j , t } ( \Delta _ { j } ^ { G } - 1 )$ and the marginal gain of � was at least $D _ { i , t } ( \Delta _ { i } ^ { G } )$ by (9). Since greedy chose $j ,$

$$
D _ { j , t } ( \Delta _ { j } ^ { G } - 1 ) \geq D _ { i , t } ( \Delta _ { i } ^ { G } ) .
$$

Combining the three inequalities,

$$
D _ { j , t } ( \Delta _ { j } ) \geq D _ { j , t } ( \Delta _ { j } ^ { G } - 1 ) \geq D _ { i , t } ( \Delta _ { i } ^ { G } ) \geq D _ { i , t } ( \Delta _ { i } - 1 ) .
$$

Define �<sup>′</sup> by transferring one sample from � to $j \colon \Delta _ { i } ^ { \prime } = \Delta _ { i } - 1 , \Delta _ { j } ^ { \prime } = \Delta _ { j } + 1$ , and $\Delta _ { \ell } ^ { \prime } = \Delta _ { \ell }$ for $\ell \notin \{ i , j \}$ . Then $\begin{array} { r } { \sum _ { \ell } \Delta _ { \ell } ^ { \prime } = B _ { t } } \end{array}$ , and

$$
G ( \Delta ^ { \prime } ) - G ( \Delta ) = D _ { j , t } ( \Delta _ { j } ) - D _ { i , t } ( \Delta _ { i } - 1 ) \geq 0 .
$$

Repeating these exchanges transforms � into $\Delta ^ { G }$ without ever decreasing the gain. Therefore $G ( \Delta ^ { G } ) \geq G ( \Delta )$ 9 and $\Delta ^ { G }$ minimizes $\textstyle \sum _ { j } \ell _ { j , t } ( \Delta _ { j } )$ over the feasible set. □

Proof of Proposition $2 ( i i )$ Fix item � and round $t ,$ and write $c ( d ) : = c _ { i } ( t , d ) , \lambda : = \lambda _ { i } ( t ) , \ell ( d ) : = \ell _ { i , t } ( d )$ $\theta : = \theta _ { i } , \mu : = \mu _ { i } ( t ) , q : = q ( t )$ , and $\pi ( \cdot )$ for the density of the posterior $\Pi _ { i , t } . \operatorname { B y }$ the definitions of $\widetilde { \theta } _ { i } ( d )$ and $\ell _ { i , t } ( d )$

$$
\ell ( d ) = \operatorname { \mathbb { E } } \left[ ( q - \mu - c ( d ) ( \theta - \mu ) ) ^ { + } \right] .
$$

We will relax $\ell ( \cdot )$ to the reals $d \geq 0$ and show ℓ is convex in $d ,$ thus $D _ { i , t } ( d ) = \ell ( d ) - \ell ( d + 1 )$ is nonincreasing in $d \in \mathbb { Z } _ { \geq 0 }$ , which is (9).

We first compute the derivatives of �. By assumption, $c ( d ) = ( 1 + \lambda d ) ^ { - 1 / 2 }$ , so

$$
c ^ { \prime } ( d ) = - { \frac { \lambda } { 2 } } ( 1 + \lambda d ) ^ { - 3 / 2 } = - { \frac { \lambda } { 2 } } c ( d ) ^ { 3 } .
$$

Diferentiating again,

$$
c ^ { \prime \prime } ( d ) = - { \frac { \lambda } { 2 } } \cdot 3 c ( d ) ^ { 2 } \cdot c ^ { \prime } ( d ) = - { \frac { \lambda } { 2 } } \cdot 3 c ( d ) ^ { 2 } \cdot \left( - { \frac { \lambda } { 2 } } c ( d ) ^ { 3 } \right) = { \frac { 3 \lambda ^ { 2 } } { 4 } } c ( d ) ^ { 5 } \geq 0 .
$$

Next we diferentiate ℓ. Let $U ( d ) : = \mu + ( q - \mu ) / c ( d )$ so

$$
\ell ( d ) = \int _ { \mathbb { R } } \bigl ( q - \mu - c ( d ) ( \theta - \mu ) \bigr ) ^ { + } \pi ( \theta ) d \theta = \int _ { - \infty } ^ { U ( d ) } \bigl ( q - \mu - c ( d ) ( \theta - \mu ) \bigr ) \pi ( \theta ) d \theta .
$$

Define $\begin{array} { r } { I ( \tau ) : = \int _ { - \infty } ^ { \tau } ( \theta - \mu ) \pi ( \theta ) d \theta } \end{array}$ , taking $\pi = 0$ outside J. Since $\mathbb { E } [ | \theta - \mu | ] < \infty$ , diferentiating under the expectation gives

$$
\ell ^ { \prime } ( d ) = - c ^ { \prime } ( d ) \int _ { - \infty } ^ { U ( d ) } ( \theta - \mu ) \pi ( \theta ) d \theta = - c ^ { \prime } ( d ) I ( U ( d ) ) .
$$

For $U ( d ) \in { \mathcal { T } }$ , diferentiating again gives

$$
\begin{array} { r l } & { \ell ^ { \prime \prime } ( d ) = - c ^ { \prime \prime } ( d ) I ( U ( d ) ) - c ^ { \prime } ( d ) I ^ { \prime } ( U ( d ) ) U ^ { \prime } ( d ) } \\ & { \qquad = - c ^ { \prime \prime } ( d ) I ( U ( d ) ) - c ^ { \prime } ( d ) \cdot \frac { q - \mu } { c ( d ) } \pi ( U ( d ) ) \cdot \left( - \frac { q - \mu } { c ( d ) ^ { 2 } } c ^ { \prime } ( d ) \right) } \\ & { \qquad = - c ^ { \prime \prime } ( d ) I ( U ( d ) ) + c ^ { \prime } ( d ) ^ { 2 } \frac { ( q - \mu ) ^ { 2 } } { c ( d ) ^ { 3 } } \pi ( U ( d ) ) . } \end{array}
$$

Both terms are nonnegative. To see this, $c ^ { \prime \prime } ( d ) \geq 0$ , and $I ( \tau ) \leq 0$ for $\tau \leq \mu$ . For $\tau \geq \mu .$ , using $\mathbb { E } [ \theta - \mu ] = 0$ gives

$$
I ( \tau ) = - \int _ { \tau } ^ { \infty } ( \theta - \mu ) \pi ( \theta ) d \theta \leq 0 .
$$

Thus $\ell ^ { \prime \prime } ( d ) \geq 0$ whenever $U ( d ) \in { \mathcal { T } }$ . On any interval where $U ( d )$ is outside J, ℓ is constant. Since $\ell ^ { \prime } ( d ) = - c ^ { \prime } ( d ) I ( U ( d ) )$ is continuous also when $U ( d )$ crosses an endpoint of $\mathcal { T } , \ell ^ { \prime }$ is nondecreasing on $[ 0 , \infty )$ . Therefore ℓ is convex, so $D _ { i , t } ( d )$ is nonincreasing in �. □

Proof of Proposition $2 ( i i i )$ Fix an item � and a round �. By conjugacy, the posterior density of $\eta _ { i }$ given $\mathcal { H } _ { t }$ is proportional to exp $\{ a \eta { - } b A _ { i } ( \eta ) \}$ }, where

$$
a = a _ { i } + \sum _ { n = 1 } ^ { N _ { i } ( t ) } Y _ { i , n } , \qquad b = b _ { i } + N _ { i } ( t ) .
$$

The standard moment identity for this natural exponential family conjugate posterior gives $\begin{array} { r } { \mu _ { i } ( t ) = \frac { a } { b } } \end{array}$ (Diaconis and Ylvisaker 1979). Additionally,

$$
0 = \int _ { { \mathcal { E } } _ { i } } \frac { d } { d \eta } \left[ \left( A _ { i } ^ { \prime } ( \eta ) - \frac { a } { b } \right) \exp \{ a \eta - b A _ { i } ( \eta ) \} \right] d \eta ,
$$

which, after expanding the derivative and normalizing, gives

$$
\mathbb { E } _ { t } \left[ A _ { i } ^ { \prime \prime } ( \eta _ { i } ) \right] = b \mathbb { E } _ { t } \left[ \left( A _ { i } ^ { \prime } ( \eta _ { i } ) - \frac a b \right) ^ { 2 } \right] = b \nu _ { i } ( t ) .
$$

Fix $d \geq 1$ , and let $\begin{array} { r } { S _ { d } = \sum _ { n = 1 } ^ { d } Y _ { i , N _ { i } ( t ) + n } } \end{array}$ denote the sum of the � additional observations. Conditional on $\eta _ { i }$ $\mathbb { E } _ { t } \left[ S _ { d } \mid \eta _ { i } \right] = d A _ { i } ^ { \prime } ( \eta _ { i } ) = d \theta _ { i }$ and $\mathbb { V } _ { t } \left( S _ { d } \mid \eta _ { i } \right) = d A _ { i } ^ { \prime \prime } ( \eta _ { i } )$ . Therefore, by the law of total variance,

$$
\mathbb { V } _ { t } \left( S _ { d } \right) = \mathbb { E } _ { t } \left[ \mathbb { V } _ { t } \left( S _ { d } \mid \eta _ { i } \right) \right] + \mathbb { V } _ { t } \left( \mathbb { E } _ { t } \left[ S _ { d } \mid \eta _ { i } \right] \right) = d b \nu _ { i } ( t ) + d ^ { 2 } \nu _ { i } ( t ) = d ( b + d ) \nu _ { i } ( t ) .
$$

After observing these additional samples, the posterior density of $\eta _ { i }$ is proportional to exp $\left\{ ( a + S _ { d } ) \eta - ( b + d ) A _ { i } ( \eta ) \right\}$ . Applying the same conjugate-posterior mean identity gives

$$
\mathbb { E } \left[ \theta _ { i } \mid \mathcal { H } _ { t } , S _ { d } \right] = \frac { a + S _ { d } } { b + d } .
$$

Hence,

$$
\mathbb { V } _ { t } \left( \mathbb { E } \left[ \theta _ { i } \mid \mathcal { H } _ { t } , S _ { d } \right] \right) = \frac { \mathbb { V } _ { t } ( S _ { d } ) } { ( b + d ) ^ { 2 } } = \frac { d } { b + d } \nu _ { i } ( t ) .
$$

Finally, the law of total variance gives

$$
\nu _ { i } ( t ) = \bar { \nu } _ { i } ( t , d ) + \mathbb { V } _ { t } \left( \mathbb { E } \left[ \theta _ { i } \mid \mathcal { H } _ { t } , S _ { d } \right] \right) .
$$

Therefore

$$
c _ { i } ( t , d ) ^ { 2 } = \frac { \bar { \nu } _ { i } ( t , d ) } { \nu _ { i } ( t ) } = \frac { b } { b + d } = \frac { 1 } { 1 + \frac { 1 } { b } d } .
$$

Since $b = b _ { i } + N _ { i } ( t ) > 0$ , (10) holds with $\begin{array} { r } { \lambda _ { i } ( t ) = \frac { 1 } { b _ { i } + N _ { i } ( t ) } } \end{array}$ □

## Appendix B: Closed forms for the conjugate families of Proposition 3

In each subsection, we first define the round-� posterior parameters and then suppress their argument � in the remaining formulas.

## B.1. Gaussian observations with known variance

Suppose $Y _ { i , n } \mid \theta _ { i } \sim N ( \theta _ { i } , \sigma _ { i } ^ { 2 } )$ with $\sigma _ { i } ^ { 2 }$ known, and prior $\theta _ { i } \sim N ( m _ { i } ( 0 ) , s _ { i } ^ { 2 } ( 0 ) )$ . At the start of round $t ,$

$$
\theta _ { i } | \mathcal { H } _ { t } \sim N ( m _ { i } ( t ) , s _ { i } ^ { 2 } ( t ) ) , \qquad s _ { i } ^ { 2 } ( t ) = \left( \frac { 1 } { s _ { i } ^ { 2 } ( 0 ) } + \frac { N _ { i } ( t ) } { \sigma _ { i } ^ { 2 } } \right) ^ { - 1 } , \quad m _ { i } ( t ) = s _ { i } ^ { 2 } ( t ) \left( \frac { m _ { i } ( 0 ) } { s _ { i } ^ { 2 } ( 0 ) } + \frac { 1 } { \sigma _ { i } ^ { 2 } } \sum _ { n = 1 } ^ { N _ { i } ( t ) } Y _ { i , n } \right) .
$$

Then $\begin{array} { r } { \mu _ { i } ( t ) = m _ { i } , \nu _ { i } ( t ) = s _ { i } ^ { 2 } , F _ { i , t } ( q ) = \Phi \Bigl ( \frac { q - m _ { i } } { s _ { i } } \Bigr ) } \end{array}$ and $q ^ { * } ( t )$ solves

$$
\sum _ { i = 1 } ^ { I } \left[ 1 - \Phi \biggl ( \frac { q - m _ { i } } { s _ { i } } \biggr ) \right] = k .
$$

We next compute the expected post-round-� posterior variance. After $\Delta _ { i }$ further observations, the posterior variance of $\theta _ { i }$ is deterministic: it depends on $\Delta _ { i }$ but not on the realized values. This deterministic update is a special property of the Gaussian-Gaussian pair that does not hold for the other families. Hence

$$
\bar { \nu } _ { i } ( t , \Delta _ { i } ) = \left( \frac { 1 } { s _ { i } ^ { 2 } } + \frac { \Delta _ { i } } { \sigma _ { i } ^ { 2 } } \right) ^ { - 1 } , \qquad c _ { i } ( t , \Delta _ { i } ) ^ { 2 } = \frac { 1 } { 1 + s _ { i } ^ { 2 } \Delta _ { i } / \sigma _ { i } ^ { 2 } } .
$$

This has the form (10), so GBB is exact.

Finally, for any $z \in \mathbb { R }$

$$
\mathbb { E } _ { t } [ ( z - \theta _ { i } ) ^ { + } ] = ( z - m _ { i } ) \Phi \Bigg ( \frac { z - m _ { i } } { s _ { i } } \Bigg ) + s _ { i } \phi \Bigg ( \frac { z - m _ { i } } { s _ { i } } \Bigg ) .
$$

Combined with $c _ { i } ( t , \Delta _ { i } )$ and (7), this yields the closed form for $\ell _ { i , t } ( \Delta _ { i } )$

## B.2. Binomial with beta posterior

Suppose $Y _ { i , n } \mid p _ { i } \sim \mathrm { { B i n } } ( r , p _ { i } )$ with $r \in \mathbb { Z } _ { + }$ known, mean parameter $\theta _ { i } = r p _ { i }$ , and prior $p _ { i } \sim \mathrm { B e t a } ( a _ { i } ( 0 ) , b _ { i } ( 0 ) )$ . Let $\begin{array} { r } { S _ { i } ( t ) : = \sum _ { n = 1 } ^ { N _ { i } ( t ) } Y _ { i , n \cdot } \mathrm { A t } } \end{array}$ the start of round �,

$$
p _ { i } \mid { \mathcal { H } } _ { t } \sim { \mathrm { B e t a } } ( a _ { i } ( t ) , b _ { i } ( t ) ) , \qquad a _ { i } ( t ) : = a _ { i } ( 0 ) + S _ { i } ( t ) , \quad b _ { i } ( t ) : = b _ { i } ( 0 ) + r N _ { i } ( t ) - S _ { i } ( t ) .
$$

Set $n _ { i } : = a _ { i } + b _ { i }$ . Then

$$
\mu _ { i } ( t ) = r \frac { a _ { i } } { n _ { i } } , \qquad \nu _ { i } ( t ) = r ^ { 2 } \frac { a _ { i } b _ { i } } { n _ { i } ^ { 2 } ( n _ { i } + 1 ) } , \qquad F _ { i , t } ( q ) = I _ { q / r } ( a _ { i } , b _ { i } ) \quad ( 0 < q < r ) ,
$$

and $q ^ { * } ( t )$ solves

$$
\sum _ { i = 1 } ^ { I } \left[ 1 - I _ { q / r } ( a _ { i } , b _ { i } ) \right] = k .
$$

We next compute the expected post-round-� posterior variance. Allocating $\Delta _ { i }$ further binomial observations is equivalent to $r \Delta _ { i }$ further Bernoulli trials. Conditional on the future number of successes $X$

$$
p _ { i } \mid \mathcal { H } _ { t } , X \sim \mathrm { B e t a } ( a _ { i } + X , b _ { i } + r \Delta _ { i } - X ) ,
$$

so taking expectation of $\mathbb { V } ( \theta _ { i } \mid \mathcal { H } _ { t } , X )$ over � gives

$$
\bar { \nu } _ { i } \left( t , \Delta _ { i } \right) = r ^ { 2 } \frac { a _ { i } b _ { i } } { n _ { i } \left( n _ { i } + 1 \right) \left( n _ { i } + r \Delta _ { i } \right) } = \nu _ { i } \left( t \right) \frac { n _ { i } } { n _ { i } + r \Delta _ { i } } , \qquad c _ { i } \left( t , \Delta _ { i } \right) ^ { 2 } = \frac { n _ { i } } { n _ { i } + r \Delta _ { i } } = \frac { 1 } { 1 + \left( r / n _ { i } \right) \Delta _ { i } } .
$$

This has the form (10), so GBB is exact.

Finally, for $0 < z < r$

$$
\mathbb { E } _ { t } \left[ ( z - \theta _ { i } ) ^ { + } \right] = z I _ { z / r } ( a _ { i } , b _ { i } ) - r \frac { a _ { i } } { n _ { i } } I _ { z / r } ( a _ { i } + 1 , b _ { i } ) ,
$$

extending to $z \le 0$ and $z \geq r$ with values 0 and $z { - } \mu _ { i } ( t )$ respectively. Combined with $c _ { i } ( t , \Delta _ { i } )$ and $( 7 )$ , this yields the closed form for $\ell _ { i , t } ( \Delta _ { i } )$

## B.3. Poisson with gamma posterior

Suppose $Y _ { i , n } \mid \theta _ { i } \sim \mathrm { P o i s } ( \theta _ { i } )$ with prior $\theta _ { i } \sim \mathrm { G a m m a } ( a _ { i } ( 0 ) , b _ { i } ( 0 ) )$ , where $b _ { i } ( 0 )$ is the rate. After $N _ { i } ( t )$ observations,

$$
\theta _ { i } \mid { \mathcal { H } } _ { t } \sim \operatorname { G a m m a } ( a _ { i } ( t ) , b _ { i } ( t ) ) , \qquad a _ { i } ( t ) : = a _ { i } ( 0 ) + \sum _ { n = 1 } ^ { N _ { i } ( t ) } Y _ { i , n } , \quad b _ { i } ( t ) : = b _ { i } ( 0 ) + N _ { i } ( t ) .
$$

The posterior density is $( b _ { i } ^ { a _ { i } } / \Gamma ( a _ { i } ) ) \theta ^ { a _ { i } - 1 } e ^ { - b _ { i } \theta } \mathrm { o n } \theta > 0$ . Then

$$
\mu _ { i } ( t ) = \frac { a _ { i } } { b _ { i } } , \qquad \nu _ { i } ( t ) = \frac { a _ { i } } { b _ { i } ^ { 2 } } , \qquad F _ { i , t } ( q ) = \gamma ( a _ { i } , b _ { i } q ) \quad ( q > 0 ) ,
$$

and $q ^ { * } ( t )$ solves

$$
\sum _ { i = 1 } ^ { I } \Gamma ( a _ { i } , b _ { i } q ) = k .
$$

We next compute the expected post-round-� posterior variance. The future total $\begin{array} { r } { X : = \sum _ { \ell = 1 } ^ { \Delta _ { i } } Y _ { i , \ell } ^ { \mathrm { n e w } } } \end{array}$ satisfies $X \mid \theta _ { i } \sim \operatorname { P o i s } ( \Delta _ { i } \theta _ { i } )$ , so

$$
\theta _ { i } \mid \mathcal { H } _ { t } , X \sim \mathrm { G a m m a } ( a _ { i } + X , b _ { i } + \Delta _ { i } ) , \qquad \mathbb { V } ( \theta _ { i } \mid \mathcal { H } _ { t } , X ) = \frac { a _ { i } + X } { ( b _ { i } + \Delta _ { i } ) ^ { 2 } } .
$$

Using $\mathbb { E } _ { t } \left[ X \right] = \Delta _ { i } a _ { i } / b _ { i }$

$$
\bar { \nu } _ { i } \left( t , \Delta _ { i } \right) = \frac { a _ { i } } { b _ { i } ( b _ { i } + \Delta _ { i } ) } = \nu _ { i } ( t ) \frac { b _ { i } } { b _ { i } + \Delta _ { i } } , \qquad c _ { i } ( t , \Delta _ { i } ) ^ { 2 } = \frac { b _ { i } } { b _ { i } + \Delta _ { i } } = \frac { 1 } { 1 + \Delta _ { i } / b _ { i } } .
$$

This has the form (10), so GBB is exact.

Finally, for $z > 0$

$$
\mathbb { E } _ { t } \left[ ( z - \theta _ { i } ) ^ { + } \right] = z \gamma ( a _ { i } , b _ { i } z ) - \frac { a _ { i } } { b _ { i } } \gamma ( a _ { i } + 1 , b _ { i } z ) .
$$

Combined with $c _ { i } ( t , \Delta _ { i } )$ and (7), this gives the closed form for $\ell _ { i , t } ( \Delta _ { i } )$

## B.4. Gamma observations with known shape and inverse-gamma posterior

Suppose $Y _ { i , n } \mid \theta _ { i } \sim \mathrm { G a m m a } ( \kappa _ { i } , \theta _ { i } / \kappa _ { i } )$ (shape $\kappa _ { i }$ , scale $\theta _ { i } / \kappa _ { i } )$ with $\kappa _ { i } > 0$ known (the case $\kappa _ { i } = 1$ is the exponential model $Y _ { i , n } \mid \theta _ { i } \sim \mathrm { E x p } ( 1 / \theta _ { i } ) )$ ). The conjugate prior is $\theta _ { i } \sim$ InvGamma $( a _ { i } ( 0 ) , b _ { i } ( 0 ) )$ with $a _ { i } ( 0 ) > 2$ and $b _ { i } ( 0 ) > 0$ , ensuring $\nu _ { i } ( t )$ is finite at every round. After $N _ { i } ( t )$ observations,

$$
\theta _ { i } \mid \mathcal { H } _ { t } \sim \operatorname { I n v G a m m a } ( a _ { i } ( t ) , b _ { i } ( t ) ) , \qquad a _ { i } ( t ) : = a _ { i } ( 0 ) + \kappa _ { i } N _ { i } ( t ) , \quad b _ { i } ( t ) : = b _ { i } ( 0 ) + \kappa _ { i } \sum _ { n = 1 } ^ { N _ { i } ( t ) } Y _ { i , n } .
$$

The posterior density is $( b _ { i } ^ { a _ { i } } / \Gamma ( a _ { i } ) ) \theta ^ { - a _ { i } - 1 } e ^ { - b _ { i } / \theta }$ on $\theta > 0$ . Then

$$
\mu _ { i } ( t ) = \frac { b _ { i } } { a _ { i } - 1 } , \qquad \nu _ { i } ( t ) = \frac { b _ { i } ^ { 2 } } { ( a _ { i } - 1 ) ^ { 2 } ( a _ { i } - 2 ) } , \qquad F _ { i , t } ( q ) = \Gamma ( a _ { i } , b _ { i } / q ) \quad ( q > 0 ) ,
$$

and $q ^ { * } ( t )$ solves

$$
\sum _ { i = 1 } ^ { I } \gamma ( a _ { i } , b _ { i } / q ) = k .
$$

We next compute the expected post-round-� posterior variance. Let $\begin{array} { r } { W : = \kappa _ { i } \sum _ { \ell = 1 } ^ { \Delta _ { i } } Y _ { i , \ell } ^ { \mathrm { n e w } } } \end{array}$ . Each gamma observation contributes likelihood proportional to $\theta ^ { - \kappa _ { i } } e ^ { - \kappa _ { i } y / \theta }$ <sup>�</sup> , so

$$
\theta _ { i } \mid \mathcal { H } _ { t } , W \sim \mathrm { I n v G a m m a } ( a _ { i } + \kappa _ { i } \Delta _ { i } , b _ { i } + W ) , \qquad \Psi ( \theta _ { i } \mid \mathcal { H } _ { t } , W ) = \frac { ( b _ { i } + W ) ^ { 2 } } { ( a _ { i } + \kappa _ { i } \Delta _ { i } - 1 ) ^ { 2 } ( a _ { i } + \kappa _ { i } \Delta _ { i } - 2 ) } .
$$

Conditional on $\theta _ { i } , W \sim \mathrm { G a m m a } ( \kappa _ { i } \Delta _ { i } , \theta _ { i } )$ (shape $\kappa _ { i } \Delta _ { i }$ , scale $\theta _ { i } )$ , so $\mathbb { E } _ { t } \left[ W \right] = \kappa _ { i } \Delta _ { i } b _ { i } / ( a _ { i } - 1 )$ and $\mathbb { E } _ { t } [ W ^ { 2 } ] =$ $\kappa _ { i } \Delta _ { i } ( \kappa _ { i } \Delta _ { i } { + } 1 ) b _ { i } ^ { 2 } / [ ( a _ { i } { - } 1 ) ( a _ { i } { - } 2 ) ]$ , giving

$$
\mathbb { E } _ { t } { [ ( b _ { i } + W ) ^ { 2 } ] } = \frac { b _ { i } ^ { 2 } ( a _ { i } + \kappa _ { i } \Delta _ { i } - 1 ) ( a _ { i } + \kappa _ { i } \Delta _ { i } - 2 ) } { ( a _ { i } - 1 ) ( a _ { i } - 2 ) }
$$

and therefore

$$
\begin{array} { l } { \displaystyle \bar { \nu } _ { i } ( t , \Delta _ { i } ) = \frac { b _ { i } ^ { 2 } } { ( a _ { i } - 1 ) ( a _ { i } - 2 ) ( a _ { i } + \kappa _ { i } \Delta _ { i } - 1 ) } = \nu _ { i } ( t ) \frac { a _ { i } - 1 } { a _ { i } + \kappa _ { i } \Delta _ { i } - 1 } , } \\ { \displaystyle c _ { i } ( t , \Delta _ { i } ) ^ { 2 } = \frac { a _ { i } - 1 } { a _ { i } + \kappa _ { i } \Delta _ { i } - 1 } = \frac { 1 } { 1 + ( \kappa _ { i } / ( a _ { i } - 1 ) ) \Delta _ { i } } . } \end{array}
$$

This has the form (10), so GBB is exact.

Finally, for $z > 0 ,$

$$
\mathbb { E } _ { t } \left[ ( z - \theta _ { i } ) ^ { + } \right] = z \Gamma ( a _ { i } , b _ { i } / z ) - \frac { b _ { i } } { a _ { i } - 1 } \Gamma ( a _ { i } - 1 , b _ { i } / z ) .
$$

Combined with $c _ { i } ( t , \Delta _ { i } )$ and (7), this gives the closed form for $\ell _ { i , t } ( \Delta _ { i } )$

## B.5. Lognormal observations with known log-variance

Suppose $Y _ { i , n } \mid \theta _ { i } \sim$ Lognormal(log $\theta _ { i } - \sigma _ { i } ^ { 2 } / 2 , \sigma _ { i } ^ { 2 } )$ with $\sigma _ { i } ^ { 2 }$ known, parameterized so $\mathbb { E } [ Y _ { i , n } \mid \theta _ { i } ] = \theta _ { i }$ . Let $\eta _ { i } : = \log \theta _ { i }$ and $X _ { i , n } : = \log Y _ { i , n } + \sigma _ { i } ^ { 2 } / 2$ , so $X _ { i , n } = \eta _ { i } + \varepsilon _ { i , n }$ with $\varepsilon _ { i , n } \sim N ( 0 , \sigma _ { i } ^ { 2 } )$ . With prior $\eta _ { i } \sim N ( m _ { i } ( 0 ) , s _ { i } ^ { 2 } ( 0 ) )$ , the transformed problem in $\eta _ { i }$ is Gaussian-Gaussian conjugate:

$$
\eta _ { i } | \mathcal { H } _ { t } \sim N ( m _ { i } ( t ) , s _ { i } ^ { 2 } ( t ) ) , \qquad s _ { i } ^ { 2 } ( t ) = \left( \frac { 1 } { s _ { i } ^ { 2 } ( 0 ) } + \frac { N _ { i } ( t ) } { \sigma _ { i } ^ { 2 } } \right) ^ { - 1 } , \quad m _ { i } ( t ) = s _ { i } ^ { 2 } ( t ) \left( \frac { m _ { i } ( 0 ) } { s _ { i } ^ { 2 } ( 0 ) } + \frac { 1 } { \sigma _ { i } ^ { 2 } } \sum _ { n = 1 } ^ { N _ { i } ( t ) } X _ { i , n } \right) .
$$

Hence $\theta _ { i } = e ^ { \eta _ { i } }$ has lognormal posterior with

$$
\mu _ { i } ( t ) = \exp \left( m _ { i } + \frac { s _ { i } ^ { 2 } } { 2 } \right) , \qquad \nu _ { i } ( t ) = ( e ^ { s _ { i } ^ { 2 } } - 1 ) e ^ { 2 m _ { i } + s _ { i } ^ { 2 } } , \qquad F _ { i , t } ( q ) = \Phi \left( \frac { \log q - m _ { i } } { s _ { i } } \right) \quad ( q > 0 ) ,
$$

and $q ^ { * } ( t )$ solves

$$
\sum _ { i = 1 } ^ { I } \left[ 1 - \Phi \left( { \frac { \log q - m _ { i } } { s _ { i } } } \right) \right] = k .
$$

We next compute the expected post-round-� posterior variance of $\theta _ { i } = e ^ { \eta _ { i } }$ , rather than of $\eta _ { i }$ . After $\Delta _ { i }$ further samples, $\eta _ { i }$ has Gaussian posterior with deterministic variance

$$
s _ { i } ^ { 2 } ( \Delta _ { i } ) : = \left( \frac { 1 } { s _ { i } ^ { 2 } } + \frac { \Delta _ { i } } { \sigma _ { i } ^ { 2 } } \right) ^ { - 1 }
$$

and random mean $M _ { i } ( \Delta _ { i } )$ . The Gaussian update and the law of total variance give

$$
\begin{array} { r } { M _ { i } ( \Delta _ { i } ) \mid \mathcal { H } _ { t } \sim N ( m _ { i } , s _ { i } ^ { 2 } - s _ { i } ^ { 2 } ( \Delta _ { i } ) ) . } \end{array}
$$

Conditional on $M _ { i } ( \Delta _ { i } ) , \theta _ { i }$ is lognormal with parameters $( M _ { i } ( \Delta _ { i } ) , s _ { i } ^ { 2 } ( \Delta _ { i } ) )$ , so

$$
\mathbb { V } ( \theta _ { i } \mid M _ { i } ( \Delta _ { i } ) ) = ( e ^ { s _ { i } ^ { 2 } ( \Delta _ { i } ) } - 1 ) e ^ { 2 M _ { i } ( \Delta _ { i } ) + s _ { i } ^ { 2 } ( \Delta _ { i } ) } .
$$

The Gaussian MGF gives $\mathbb { E } _ { t } \left[ e ^ { 2 M _ { i } ( \Delta _ { i } ) } \right] = \exp ( 2 m _ { i } + 2 ( s _ { i } ^ { 2 } - s _ { i } ^ { 2 } ( \Delta _ { i } ) ) )$ , so

$$
\bar { \nu } _ { i } ( t , \Delta _ { i } ) = ( e ^ { s _ { i } ^ { 2 } ( \Delta _ { i } ) } - 1 ) e ^ { 2 m _ { i } + 2 s _ { i } ^ { 2 } - s _ { i } ^ { 2 } ( \Delta _ { i } ) } , \qquad c _ { i } ( t , \Delta _ { i } ) ^ { 2 } = \frac { 1 - e ^ { - s _ { i } ^ { 2 } ( \Delta _ { i } ) } } { 1 - e ^ { - s _ { i } ^ { 2 } } } .
$$

This does not have the form (10) in general, so Proposition 2(ii) does not apply.

Finally, for $z > 0$

$$
\mathbb { E } _ { t } \left[ ( z - \theta _ { i } ) ^ { + } \right] = z \Phi \left( \frac { \log z - m _ { i } } { s _ { i } } \right) - \mu _ { i } ( t ) \Phi \left( \frac { \log z - m _ { i } - s _ { i } ^ { 2 } } { s _ { i } } \right) .
$$

Combined with $c _ { i } ( t , \Delta _ { i } )$ and (7), this yields the closed form for $\ell _ { i , t } ( \Delta _ { i } )$ .

## Appendix C: Proofs for §4

## C.1. Proof of consistency

To prove Proposition 4 we first prove three useful lemmas.

LEMMA 5 (Conditional posterior consistency). Under Assumption 1, there exists a set $\Theta ^ { * } \subseteq \mathcal { T } ^ { I }$ with prior probability one such that, for every $\pmb { \theta } ^ { 0 } \in \Theta ^ { * }$ , almost surely under $\mathbb { P } _ { \pmb { \theta } ^ { 0 } } \colon i f N _ { i } ( t )  \infty a s t  \infty ,$ then $\mu _ { i } ( t ) \to \theta _ { i } ^ { 0 }$ and $\nu _ { i } ( t ) \to 0$

Proof For each �, Assumption 1 provides the conditions for Doob’s theorem (Doob 1949, Miller 2018): the mean-parametrization $\mathbb { E } _ { \theta } [ Y _ { i , n } ] = \theta$ makes the family $\{ P _ { i } ( \cdot | \theta ) : \theta \in \mathcal { T } \}$ identifiable, and $\mathbb { E } _ { \Pi _ { i , 0 } } [ \theta _ { i } ^ { 2 } ] < \infty$ allows the choice $g ( \theta _ { i } ) = \theta _ { i } ^ { 2 }$ . Doob’s theorem applied to $g ( \theta _ { i } ) = \theta _ { i }$ and $g ( \theta _ { i } ) = \theta _ { i } ^ { 2 }$ gives a set $J _ { i } ^ { * } \subseteq { \mathcal { I } }$ with $\Pi _ { i , 0 } ( J _ { i } ^ { * } ) = 1$ such that, for every $\mathcal { \theta } _ { i } ^ { 0 } \in J _ { i } ^ { * } , \mathbb { P } _ { \theta _ { i } ^ { 0 } }$ -almost surely: if $N _ { i } ( t )  \infty$ , then $\mu _ { i } ( t ) \to \theta _ { i } ^ { 0 }$ and $\mathbb { E } _ { t } [ \theta _ { i } ^ { 2 } ]  ( \theta _ { i } ^ { 0 } ) ^ { 2 }$ . Thus, $\nu _ { i } ( t ) = \mathbb { E } _ { t } [ \theta _ { i } ^ { 2 } ] - \mu _ { i } ( t ) ^ { 2 } \to 0 ,$ . Define $\Theta ^ { * } : = J _ { 1 } ^ { * } \times \dots \times J _ { I } ^ { * } \subseteq \mathcal { T } ^ { I }$ . Since the priors are independent across items, $\Theta ^ { * }$ has prior probability $\begin{array} { r } { \prod _ { i = 1 } ^ { I } \Pi _ { i , 0 } ( J _ { i } ^ { * } ) = 1 } \end{array}$ □

LEMMA 6 (Uniform improvement from one observation). Let Π be a distribution on J with continuous positive density, mean $\mu : = \mathbb { E } _ { \Pi } [ \theta ]$ , and variance $\nu : = \mathbb { V } _ { \Pi } ( \theta ) > 0 .$ . Let $\theta \sim \Pi$ and $Y \mid \theta \sim P ( \cdot \mid \theta )$ with $\mathbb { E } [ Y \mid \theta ] = \theta$ and finite predictive second moment $\mathbb { E } [ Y ^ { 2 } ] < \infty$ . Define $\bar { \nu } : = \mathbb { E } [ \mathbb { V } ( \theta \mid Y ) ] , \ c : = \sqrt { \bar { \nu } / \nu }$ , and $X _ { c } : = \mu + c ( \theta - \mu )$ . Then $c < 1$ , and for every compact $K \subset \mathcal { T }$ there exists $\eta > 0$ such that

$$
\begin{array} { r } { \mathbb { E } [ ( q - \theta ) ^ { + } ] - \mathbb { E } [ ( q - X _ { c } ) ^ { + } ] \geq \eta \qquad f o r e \nu e r y q \in K . } \end{array}
$$

Proof We first show $c < 1$ . By the law of total variance,

$$
\nu = \operatorname { \mathbb { E } } [ \mathbb { V } ( \theta \mid Y ) ] + \mathbb { V } ( \operatorname { \mathbb { E } } [ \theta \mid Y ] ) = { \bar { \nu } } + \mathbb { V } ( \operatorname { \mathbb { E } } [ \theta \mid Y ] ) .
$$

$\mathbb { V } ( \mathbb { E } [ \theta \mid Y ] ) > 0$ since

$$
\operatorname { C o v } ( \mathbb { E } [ \theta \mid Y ] , Y ) = \mathbb { E } [ \mathbb { E } [ \theta \mid Y ] \cdot Y ] - \mathbb { E } [ \mathbb { E } [ \theta \mid Y ] ] \mathbb { E } [ Y ] = \mathbb { E } [ \theta Y ] - \mu ^ { 2 } = \nu ,
$$

so $\bar { \nu } < \nu$ and $c = \sqrt { \bar { \nu } / \nu } < 1$

Next we show the inequality for a fixed �. Fix $q \in \mathcal { T }$ and let $\phi _ { q } ( x ) : = ( q - x ) ^ { + }$ . Since $X _ { c } = c \theta + ( 1 - c ) \mu$ is a convex combination of � and $\mu ,$ convexity of $\phi _ { q }$ gives,

$$
\phi _ { q } ( X _ { c } ) \leq c \phi _ { q } ( \theta ) + ( 1 - c ) \phi _ { q } ( \mu ) .
$$

Taking expectations,

$$
\begin{array} { r } { \mathbb { E } [ \phi _ { q } ( X _ { c } ) ] \leq c \mathbb { E } [ \phi _ { q } ( \theta ) ] + ( 1 - c ) \phi _ { q } ( \mu ) < \mathbb { E } [ \phi _ { q } ( \theta ) ] . } \end{array}
$$

The second inequality follows from Jensen’s inequality, and is strict since Π has continuous positive density on $\mathcal { T }$ and $\mathcal { T }$ is open so $\mathbb { P } ( \theta < q ) > 0$ and $\mathbb { P } ( \theta > q ) > 0$ . Define

$$
\delta ( q ) : = \mathbb { E } [ ( q - \theta ) ^ { + } ] - \mathbb { E } [ ( q - X _ { c } ) ^ { + } ] > 0 .
$$

For any $q ^ { \prime } \in \mathbb { R } , | ( q ^ { \prime } - x ) ^ { + } - ( q - x ) ^ { + } | \leq | q ^ { \prime } - q |$ so $\delta ( q )$ is 2-Lipschitz and thus is continuous in $q .$ . Thus for compact $K , \eta : = \operatorname* { i n f } _ { q \in K } \delta ( q ) > 0$ □

LEMMA 7 (Every item is sampled infinitely often). Under the hypotheses of Proposition 4, for every $\pmb { \theta } ^ { 0 } \in \Theta ^ { * }$

$$
N _ { i } ( t ) \to \infty \qquad f o r e \nu e r y i = 1 , \dots , I
$$

P<sub>�</sub>0 -almost surely.

Proof Fix $\pmb { \theta } ^ { 0 } \in \Theta ^ { * }$ and a sample path in the probability-one event on which the conclusions of Lemma 5 hold. For the asymptotic claim, we consider the procedure of Algorithm 3 extended to infinitely many rounds, with $B _ { t } \in \mathbb { Z } _ { + }$ <sub>+</sub> for every �, thus $\textstyle \sum _ { t = 1 } ^ { T } B _ { t } \to$ ∞ as $T \to \infty$ . Assume by contradiction that some items are sampled only finitely often, and define

$$
F : = \{ i : \operatorname* { s u p } _ { t \geq 1 } N _ { i } ( t ) < \infty \} .
$$

Since the sum of samples diverges, $F ^ { c } \neq \varnothing$ . All items $i \in F$ eventually stop getting sampled, so for all suficiently large � we write $\Pi _ { i , t } = \Pi _ { i } ^ { \infty } , \mu _ { i } ( t ) = \mu _ { i } ^ { \infty }$ , and $\nu _ { i } ( t ) = \nu _ { i } ^ { \infty } > 0$ . For each $j \notin F , N _ { j } ( t ) \to \infty$ , so by Lemma 5, $\mu _ { j } ( t ) \to \theta _ { j } ^ { 0 }$ and $\nu _ { j } ( t ) \to 0$ . We begin by showing that each of $q ^ { * } ( t )$ and $q ^ { \dagger } ( t )$ is contained in a compact interval for all suficiently large �. Choose � $, b \in { \mathcal { T } }$ with $a < \min _ { j \in F ^ { c } } \theta _ { j } ^ { 0 }$ and $b > \operatorname* { m a x } _ { j \in F ^ { c } } \theta _ { j } ^ { 0 }$ . For each $j \in F ^ { c } , \mu _ { j } ( t ) \to \theta _ { i } ^ { 0 }$ and $\nu _ { j } ( t ) \to 0$ , so Chebyshev’s inequality gives $\mathbb { P } _ { t } ( \theta _ { j } \geq a )  1$ and $\mathbb { P } _ { t } ( \theta _ { j } \geq b )  0$

Moreover, since the distributions of the items in � are fixed, � and � can be chosen such that, for all suficiently large �,

$$
\sum _ { i = 1 } ^ { I } \mathbb { P } _ { t } \left( \theta _ { i } \geq a \right) > k , \qquad \sum _ { i = 1 } ^ { I } \mathbb { P } _ { t } \left( \theta _ { i } \geq b \right) < k .
$$

Since $\begin{array} { r } { q \mapsto \sum _ { i } \mathbb { P } _ { t } ( \theta _ { i } \geq q ) } \end{array}$ is nonincreasing in � and equals $k$ at $q ^ { * } ( t )$ by (2), $q ^ { * } ( t ) \in K : = [ a , b ]$ for all suficiently large �.

For $q ^ { \dagger } ( t )$ , since every posterior mean converges to a point in $\mathcal { T } _ { : }$ , we can enlarge � within $\mathcal { T }$ , if necessary, so that it also contains every $\mu _ { i } ( t )$ for all suficiently large �. By (3), $\mu _ { [ k + 1 ] } ( t ) \leq q ^ { \dag } ( t ) \leq \mu _ { [ k ] } ( t ) , \operatorname { s o } q ^ { \dag } ( t ) \in K$ as well.

Applying Lemma 6 to $\Pi _ { i } ^ { \infty }$ with the compact set � gives $\eta _ { i } > 0$ such that

$$
\mathbb { E } _ { \Pi _ { i } ^ { \infty } } [ ( q ( t ) - \theta _ { i } ) ^ { + } ] - \mathbb { E } [ ( q ( t ) - X _ { c _ { i } ^ { \infty } } ) ^ { + } ] \ge \eta _ { i } ,
$$

where $X _ { c _ { i } ^ { \infty } } : = \mu _ { i } ^ { \infty } + c _ { i } ^ { \infty } ( \theta _ { i } - \mu _ { i } ^ { \infty } )$ and $c _ { i } ^ { \infty } : = \sqrt { \bar { \nu } _ { i } ^ { \infty } ( 1 ) / \nu _ { i } ^ { \infty } }$ . Once $\Pi _ { i , t } = \Pi _ { i } ^ { \infty }$ , the left-hand side is exactly $\ell _ { i , t } ( 0 ) { - } \ell _ { i , t } ( 1 )$ . Hence $\ell _ { i , t } ( 0 ) { - } \ell _ { i , t } ( 1 ) \geq \eta _ { i }$ for all suficiently large �. Let $c _ { F } : = \mathrm { m i n } _ { i \in F } \eta _ { i } > 0$ . Then for all suficiently large �,

$$
\ell _ { i , t } ( 0 ) - \ell _ { i , t } ( 1 ) \geq c _ { F } \qquad { \mathrm { ~ f o r ~ e v e r y ~ } } i \in F .
$$

To get a contradiction, we next show that for every $j \not \in F$ , the maximum possible improvement from allocating samples to item � converges to 0. For $j \not \in F$ , let

$$
R _ { j , t } : = \ell _ { j , t } ( 0 ) - \operatorname* { i n f } _ { d \geq 0 } \ell _ { j , t } ( d )
$$

be the maximum possible improvement from allocating any number of samples to item $j .$ . Since $x \mapsto ( q - x ) ^ { + }$ is 1-Lipschitz, for any random �,

$$
\left| \mathbb { E } [ ( q - X ) ^ { + } ] - ( q - \mathbb { E } [ X ] ) ^ { + } \right| \leq \mathbb { E } | X - \mathbb { E } [ X ] | \leq { \sqrt { \mathbb { V } ( X ) } } .
$$

The current distribution for item � has variance $\nu _ { j } ( t ) = \bar { \nu } _ { j } ( t , 0 )$ , and $\bar { \nu } _ { j } ( t , d ) \leq \bar { \nu } _ { j } ( t , 0 )$ . Therefore

$$
R _ { j , t } \leq 2 \sqrt { \nu _ { j } ( t ) }  0 .
$$

Since $F ^ { c }$ is finite, $\begin{array} { r } { \sum _ { j \notin F } R _ { j , t } \to 0 } \end{array}$ , and for all suficiently large $t , \sum _ { j \notin F } R _ { j , t } < c _ { F }$ . Thus, for suficiently large $t ,$ the feasible allocation that places one sample on some $i _ { 0 } \in F$ and distributes the remaining $B _ { t } { - } 1$ samples arbitrarily achieves total improvement

$$
\left( \ell _ { i _ { 0 } , t } ( 0 ) - \ell _ { i _ { 0 } , t } ( 1 ) \right) + \sum _ { j \neq i _ { 0 } } \left( \ell _ { j , t } ( 0 ) - \ell _ { j , t } ( d _ { j } ) \right) \geq c _ { F } .
$$

By contrast, any allocation that assigns no samples to items in � has a total improvement of at most $\textstyle \sum _ { j \notin F } R _ { j , t } < c _ { F }$ . Therefore, the optimal solution allocates a sample to at least one item in � for every suficiently large �, contradicting the assumption. Hence $F = \varnothing$ □

Proof of Proposition 4 Fix $\pmb { \theta } ^ { 0 } \in \Theta ^ { * }$ . By Lemma $7 , N _ { i } ( t ) $ ∞ for every �, almost surely. Hence $\mu _ { i } ( t ) \to \theta _ { i } ^ { 0 }$ and $\nu _ { i } ( t ) \to 0$ for every �, almost surely. Next, for any realized �,

$$
\begin{array} { l } { \displaystyle \mathrm { O C } ( S _ { k } ( \mu ( t ) ) , \pmb \theta ) = \sum _ { i \in S _ { k } ( \pmb \theta ) } \theta _ { i } - \sum _ { i \in S _ { k } ( \mu ( t ) ) } \theta _ { i } } \\ { \displaystyle \qquad \leq \sum _ { i \in S _ { k } ( \pmb \theta ) } \vert \theta _ { i } - \mu _ { i } ( t ) \vert + \sum _ { i \in S _ { k } ( \mu ( t ) ) } \vert \theta _ { i } - \mu _ { i } ( t ) \vert } \\ { \displaystyle \qquad \leq 2 k \Vert \pmb \theta - \mu ( t ) \Vert _ { \infty } . } \end{array}
$$

Taking expectation,

$$
\mathrm { E O C } ( \mathcal { H } _ { t } ) \leq 2 k \mathbb { E } _ { t } \Vert \theta - \mu ( t ) \Vert _ { \infty } \leq 2 k \sum _ { i = 1 } ^ { I } \mathbb { E } _ { t } \vert \theta _ { i } - \mu _ { i } ( t ) \vert \leq 2 k \sum _ { i = 1 } ^ { I } \sqrt { \nu _ { i } ( t ) } .
$$

Since $\nu _ { i } ( t ) \to 0$ for every $i , \mathrm { E O C } ( \mathcal { H } _ { t } )  0$ almost surely.

Finally, assume $\theta _ { [ k ] } ^ { 0 } > \theta _ { [ k + 1 ] } ^ { 0 }$ , and let $g : = \theta _ { [ k ] } ^ { 0 } - \theta _ { [ k + 1 ] } ^ { 0 } > 0$ . Since $\mu _ { i } ( t ) \to \theta _ { i } ^ { 0 }$ for every �, eventually $| \mu _ { i } ( t ) - \theta _ { i } ^ { 0 } | < g / 3 . \mathrm { I f } i \in S _ { k } ( \pmb { \theta } ^ { 0 } )$ and $j \notin S _ { k } ( \pmb \theta ^ { 0 } )$ , then $\theta _ { i } ^ { 0 } \geq \theta _ { [ k ] } ^ { 0 }$ and $\theta _ { j } ^ { 0 } \leq \theta _ { [ k + 1 ] } ^ { 0 }$ . Hence, eventually,

$$
\mu _ { i } ( t ) \geq \theta _ { [ k ] } ^ { 0 } - g / 3 > \theta _ { [ k + 1 ] } ^ { 0 } + g / 3 \geq \mu _ { j } ( t ) .
$$

Thus $S _ { k } ( { \pmb \mu } ( t ) ) = S _ { k } ( { \pmb \theta } ^ { 0 } )$ eventually almost surely. □

## C.2. The optimal exponential rate

Throughout this subsection and the next, assume Assumption 2 and $1 \leq B _ { t } \leq B$ for every �, where � is a fixed positive integer. All convergence statements hold $\mathbb { P } _ { \pmb { \theta } ^ { 0 } }$ -almost surely.

Proof of Lemma 3 By the definition of $\mathrm { A l t } ( \pmb \theta ^ { 0 } )$ ,

$$
\begin{array}{c} \Gamma ( w ; \theta ^ { 0 } ) = \operatorname* { i n f } _ { x \in \mathrm { A l t } ( \theta ^ { 0 } ) } \sum _ { \ell = 1 } ^ { I } w _ { \ell } d _ { \ell } ( \theta _ { \ell } ^ { 0 } , x _ { \ell } )  \\ { = \operatorname* { m i n } _ { i \le k < j } \operatorname* { i n f } _ { x \in \mathcal { I } _ { i } ^ { I } } \sum _ { \ell = 1 } ^ { I } w _ { \ell } d _ { \ell } ( \theta _ { \ell } ^ { 0 } , x _ { \ell } ) } \\ { = \operatorname* { m i n } _ { i \le k < j } \operatorname* { i n f } _ { u , v \in \mathcal { J } } \{ w _ { i } d _ { i } ( \theta _ { i } ^ { 0 } , u ) + w _ { j } d _ { j } ( \theta _ { j } ^ { 0 } , \nu ) \} . } \end{array}
$$

The last equality follows by setting $x _ { \ell } = \theta _ { \ell } ^ { 0 }$ for $\ell \notin \{ i , j \}$ . We conclude by showing that the inner infimum equals

$$
\operatorname* { m i n } _ { x \in [ \theta _ { j } ^ { 0 } , \theta _ { i } ^ { 0 } ] } \{ w _ { i } d _ { i } ( \theta _ { i } ^ { 0 } , x ) + w _ { j } d _ { j } ( \theta _ { j } ^ { 0 } , x ) \} .
$$

Under Assumption 2, $d _ { \ell } ( \theta _ { \ell } ^ { 0 } , x )$ is continuous and decreases as � moves toward $\theta _ { \ell } ^ { 0 }$ . By continuity of $d _ { \ell } ( \theta _ { \ell } ^ { 0 } , \cdot )$ we may replace $u < \nu$ with $u \leq \nu$ . Since moving � toward $\theta _ { i } ^ { 0 }$ and � toward $\theta _ { j } ^ { 0 }$ cannot increase the sum, we may restrict to $\theta _ { j } ^ { 0 } \leq u \leq \nu \leq \theta _ { i } ^ { 0 }$ . Within this interval, replacing � by � cannot increase $d _ { i } ( \theta _ { i } ^ { 0 } , u )$ . Thus we may take $u = \nu = x \in [ \theta _ { j } ^ { 0 } , \theta _ { i } ^ { 0 } ]$ , and the infimum is a minimum by continuity and compactness. □

LEMMA 8 (EOC and posterior error rates). For every sampling rule,

$$
- \frac { \log \mathrm { E O C } ( \mathcal { H } _ { t } ) } { \sum _ { i } N _ { i } ( t ) } - \Gamma ( \widehat { w } ( t ) ; \theta ^ { 0 } ) \longrightarrow 0 \qquad a n d \qquad - \frac { \log \mathbb { P } _ { t } ( S _ { k } ( \theta ) \ne S _ { k } ( \theta ^ { 0 } ) ) } { \sum _ { i } N _ { i } ( t ) } - \Gamma ( \widehat { w } ( t ) ; \theta ^ { 0 } ) \longrightarrow 0 .
$$

Proof The posterior error statement follows from Lemma 2 with $O = \mathrm { A l t } ( \pmb { \theta } ^ { 0 } )$ . To prove the EOC statement, we first bound the posterior expected loss of selecting the true top-� set, $R _ { t } : = \mathbb { E } _ { t } [ \mathrm { O C } ( S _ { k } ( \pmb { \theta } ^ { 0 } ) , \pmb { \theta } ) ]$ . We then compare $R _ { t }$ with EOC(H<sub>�</sub> ).

For $0 < \delta < U - L$ , define

$$
O _ { \delta } : = \bigcup _ { i \leq k < j } \{ { \pmb x } \in \mathcal { T } ^ { I } : x _ { j } > x _ { i } + \delta \} .
$$

On $O _ { \delta }$ , the opportunity cost of selecting $S _ { k } ( \pmb \theta ^ { 0 } )$ is at least �. Also, opportunity cost is at most $k ( U { - } L )$ and is zero outside $\mathrm { A l t } ( \pmb \theta ^ { 0 } )$ . Therefore,

$$
\delta \mathbb { P } _ { t } ( \pmb { \theta } \in O _ { \delta } ) \leq R _ { t } \leq k ( U - L ) \mathbb { P } _ { t } ( \pmb { \theta } \in \mathrm { A l t } ( \pmb { \theta } ^ { 0 } ) ) .\tag{15}
$$

Applying Lemma 2 to the probabilities in (15) gives

$$
\Gamma ( \widehat { w } ( t ) ; \pmb { \theta } ^ { 0 } ) - o ( 1 ) \leq - \frac { \log R _ { t } } { \sum _ { i } N _ { i } ( t ) } \leq \operatorname* { i n f } _ { \pmb { y } \in O _ { \delta } } \sum _ { \ell } \widehat { w } _ { \ell } ( t ) d _ { \ell } ( \theta _ { \ell } ^ { 0 } , y _ { \ell } ) + o ( 1 ) .
$$

It therefore sufices to show that for every $\varepsilon > 0$ , there is a $\delta > 0$ such that, for every $w \in \Delta _ { I }$

$$
\Gamma ( w ; \pmb \theta ^ { 0 } ) \leq \operatorname* { i n f } _ { \pmb { y } \in O _ { \delta } } \sum _ { \ell } w _ { \ell } d _ { \ell } ( \pmb \theta _ { \ell } ^ { 0 } , \mathbf { y } _ { \ell } ) \leq \Gamma ( \pmb w ; \pmb \theta ^ { 0 } ) + \varepsilon .\tag{16}
$$

The first inequality follows from $O _ { \delta } \subseteq \mathrm { A l t } ( \theta ^ { 0 } )$ . For the second, choose $\delta > 0$ small enough that $x \pm \delta \in \mathcal { T }$ and

$$
| d _ { i } ( \theta _ { i } ^ { 0 } , x \pm \delta ) - d _ { i } ( \theta _ { i } ^ { 0 } , x ) | \leq \varepsilon \qquad { \mathrm { f o r ~ e v e r y ~ } } i { \mathrm { ~ a n d ~ } } x \in [ \theta _ { I } ^ { 0 } , \theta _ { 1 } ^ { 0 } ] .
$$

Fix � and choose $i , j , x$ attaining the minimum in (12). Setting $y _ { i } = x - \delta , y _ { j } = x + \delta$ , and $y _ { \ell } = \theta _ { \ell } ^ { 0 }$ for all other items gives a vector in $O _ { \delta }$ . Hence

$$
\begin{array} { l } { \displaystyle \operatorname* { i n f } _ { y \in O _ { \delta } } \displaystyle \sum _ { \ell } w _ { \ell } d _ { \ell } ( \theta _ { \ell } ^ { 0 } , y _ { \ell } ) \leq w _ { i } d _ { i } ( \theta _ { i } ^ { 0 } , x - \delta ) + w _ { j } d _ { j } ( \theta _ { j } ^ { 0 } , x + \delta ) } \\ { \displaystyle \qquad \leq w _ { i } d _ { i } ( \theta _ { i } ^ { 0 } , x ) + w _ { j } d _ { j } ( \theta _ { j } ^ { 0 } , x ) + \varepsilon ( w _ { i } + w _ { j } ) } \\ { \displaystyle \qquad \leq \Gamma ( w ; \theta ^ { 0 } ) + \varepsilon . } \end{array}
$$

This proves (16). Combining it with the preceding bounds on − log $R _ { t } / \sum _ { i } N _ { i } ( t )$ and letting $\varepsilon \downarrow 0$ gives

$$
- \frac { \log R _ { t } } { \sum _ { i } N _ { i } ( t ) } - \Gamma ( \widehat { \pmb { w } } ( t ) ; \pmb { \theta } ^ { 0 } ) \longrightarrow 0 .\tag{17}
$$

We next compare $R _ { t }$ with $\operatorname { E O C } ( \mathcal { H } _ { t } )$ . Let $g : = \theta _ { k } ^ { 0 } - \theta _ { k + 1 } ^ { 0 } > 0$ and $V : = \{ \pmb { x } \in \mathcal { T } ^ { I } : \| \pmb { x } - \pmb { \theta } ^ { 0 } \| _ { \infty } < g / 4 \}$ so that for every ${ \pmb x } \in V , S _ { k } ( { \pmb x } ) = S _ { k } ( { \pmb \theta } ^ { 0 } )$ . If $S _ { k } ( { \pmb \mu } ( t ) ) \neq S _ { k } ( { \pmb \theta } ^ { 0 } )$ , its opportunity cost is at least $g / 2$ when $\pmb \theta \in V$ , thus $\operatorname { E O C } ( \mathcal { H } _ { t } ) \geq ( g / 2 ) \mathbb { P } _ { t } ( \pmb { \theta } \in V ) . \operatorname { I f } S _ { k } ( \pmb { \mu } ( t ) ) = S _ { k } ( \pmb { \theta } ^ { 0 } )$ , then $\mathrm { E O C } ( \mathcal { H } _ { t } ) = R _ { t }$ . Since selecting the largest posterior means minimizes posterior expected opportunity cost, we also have $\mathrm { E O C } ( \mathcal { H } _ { t } ) \leq R _ { t }$ in both cases. Hence

$$
\operatorname* { m i n } \{ R _ { t } , ( g / 2 ) \mathbb { P } _ { t } ( \pmb { \theta } \in V ) \} \le \mathrm { E O C } ( \mathcal { H } _ { t } ) \le R _ { t } .
$$

Since $\pmb { \theta } ^ { 0 } \in V$ , the infimum in Lemma 2 is zero for $O = V ,$ , giving

$$
- \frac { \log \mathbb { P } _ { t } ( \pmb { \theta } \in V ) } { \sum _ { i } N _ { i } ( t ) } \longrightarrow 0 .
$$

Taking logarithms and using (17) proves the EOC statement. □

Proof of Proposition 5 For part (i), Γ is continuous in � by (12), since each inner minimum is over a fixed compact interval. It therefore attains a finite maximum on $\Delta _ { I }$ . If every $w _ { i } > 0$ , each pair minimum is positive because the two KL terms cannot both vanish when $\theta _ { i } ^ { 0 } > \theta _ { j } ^ { 0 }$ . Thus $\Gamma ^ { * } ( \pmb { \theta } ^ { 0 } ) > 0$

We will also use that every maximizing allocation has positive elements. To see this, take $w _ { i } = 0$ and choose some � on the other side of the true top-� boundary. Evaluating the pair at $x = \theta _ { j } ^ { 0 }$ gives

$$
0 \leq \Gamma ( w ; \theta ^ { 0 } ) \leq w _ { i } d _ { i } ( \theta _ { i } ^ { 0 } , \theta _ { j } ^ { 0 } ) + w _ { j } d _ { j } ( \theta _ { j } ^ { 0 } , \theta _ { j } ^ { 0 } ) = 0 .
$$

Such an allocation cannot attain the positive maximum.

For part (ii), Lemma 8 and the definition of Γ<sup>∗</sup> give

$$
\operatorname* { l i m } _ { t \to \infty } \operatorname* { l p } _ { - } \frac { \log \mathrm { E O C } ( \mathcal { H } _ { t } ) } { \sum _ { i } N _ { i } ( t ) } \le \Gamma ^ { * } ( \pmb { \theta } ^ { 0 } ) .
$$

For part (iii), if $\widehat { \pmb { w } } ( t )  \pmb { w } ^ { * }$ for a maximizing allocation $\pmb { w } ^ { * }$ , continuity gives

$$
- \frac { \log \mathrm { E O C } ( \mathcal { H } _ { t } ) } { \sum _ { i } N _ { i } ( t ) } \longrightarrow \Gamma ( w ^ { * } ; \pmb { \theta } ^ { 0 } ) = \Gamma ^ { * } ( \pmb { \theta } ^ { 0 } ) .
$$

The posterior error result follows from Lemma 8. □

## C.3. The rate of TEAk

We now prove Theorems 1 and 2. We first establish Theorem 1 for $q ( t ) \to q _ { 0 } \in ( \theta _ { k + 1 } ^ { 0 } , \theta _ { k } ^ { 0 } )$ . We then prove Remark 3 and Lemma 4 and apply Theorem 1 with $q _ { 0 } = q ^ { \dagger }$ . Finally, we compare the resulting rate with $\Gamma ^ { * } ( \pmb { \theta } ^ { 0 } )$ to prove Theorem 2.

To prove Theorem 1, we first determine the exponential rate of $D _ { i , t } ( d )$ . For this, we need the following lemma on the posterior density, mean, and variance. The mean and variance convergence was already established in Lemma 5 for prior-almost every true parameter. Here, under the stronger assumptions of Assumption 2, we establish it for every fixed true parameter, together with the additional estimates needed to determine the rate of $D _ { i , t } ( d )$ . The proof relies on Russo (2020, Proposition 4 and Lemmas EC.5, EC.7, and EC.8).

LEMMA 9 (Posterior density, mean, and variance). Let $\Pi _ { i } ^ { ( n ) }$ be the posterior after thefirst � observations ofitem $i ,$ with density $\pi _ { i } ^ { ( n ) }$ , mean $\mu _ { i , n } ,$ , and variance $\nu _ { i , n } .$ . Thus $\Pi _ { i } ^ { ( 0 ) } = \Pi _ { i , 0 }$ and $\Pi _ { i , t } = \Pi _ { i } ^ { ( N _ { i } ( t ) ) }$ . For every �,

$$
\varepsilon _ { i , n } : = \operatorname* { s u p } _ { x \in \mathcal { T } } \left. \frac { 1 } { n } \log \frac { \pi _ { i } ^ { ( n ) } ( x ) } { \pi _ { i , 0 } ( x ) } + d _ { i } ( \theta _ { i } ^ { 0 } , x ) \right. \longrightarrow 0 ,\tag{18}
$$

$$
\mu _ { i , n } \longrightarrow \theta _ { i } ^ { 0 } , \qquad \nu _ { i , n } \longrightarrow 0 , \qquad - \frac { 1 } { n } \log \nu _ { i , n } \longrightarrow 0 .
$$

$I f N _ { i } ( t ) \to \infty ,$ then for every fixed integer $m \geq 0 ,$

$$
\frac { \bar { \nu } _ { i } ( t , m ) } { \nu _ { i } ( t ) } \longrightarrow 1 .\tag{19}
$$

Proof Fix � and let $\widetilde { \pi } _ { i } ^ { ( n ) }$ be the posterior density of $\eta _ { i } ( \theta _ { i } )$ . By Russo (2020, Lemmas EC.5, EC.7, and EC.8), with the latter two applied over $\mathcal { E } _ { i }$

$$
\operatorname* { s u p } _ { x \in \mathcal { T } } \left| \frac { 1 } { n } \log \widetilde { \pi } _ { i } ^ { ( n ) } ( \eta _ { i } ( x ) ) + d _ { i } ( \theta _ { i } ^ { 0 } , x ) \right| \longrightarrow 0 .
$$

Since $\widetilde { \pi } _ { i , 0 }$ is bounded above and bounded away from zero, the same limit holds after dividing $\widetilde { \pi } _ { i } ^ { ( n ) }$ by $\widetilde { \pi } _ { i , 0 }$ The posterior-to-prior density ratio is unchanged by the transformation $\eta _ { i }$ , proving (18).

We next establish convergence of the posterior mean and variance. By Russo (2020, Proposition 4), the posterior concentrates at $\theta _ { i } ^ { 0 }$ . Since $\mathcal { T }$ is bounded, this gives $\mu _ { i , n } \to \theta _ { i } ^ { 0 }$ and $\nu _ { i , n } \to 0$ . His proof of that proposition in Section EC.6.2 also shows that, for every suficiently small $\delta > 0$ , there is a constant $c _ { i , \delta } > 0$ such that, for all suficiently large �,

$$
\Pi _ { i } ^ { ( n ) } ( | \theta _ { i } - \theta _ { i } ^ { 0 } | \geq \delta ) \leq e ^ { - n c _ { i , \delta } } .\tag{20}
$$

We next show that $- n ^ { - 1 }$ log $\nu _ { i , n } \to 0$ . Fix $\varepsilon > 0$ and choose $\delta > 0$ so that $V : = ( \theta _ { i } ^ { 0 } - 2 \delta , \theta _ { i } ^ { 0 } - \delta ) \subset \mathcal { T }$ and $d _ { i } ( \theta _ { i } ^ { 0 } , x ) \leq \varepsilon$ for every $x \in V .$ . Since $\mu _ { i , n } \to \theta _ { i } ^ { 0 }$ , eventually $\mu _ { i , n } \geq \theta _ { i } ^ { 0 } { - } \delta / 2$ , giving

$$
\nu _ { i , n } \geq \int _ { V } ( x - \mu _ { i , n } ) ^ { 2 } \Pi _ { i } ^ { ( n ) } ( d x ) \geq \frac { \delta ^ { 2 } } { 4 } \Pi _ { i } ^ { ( n ) } ( V ) .
$$

Taking logarithms and applying Lemma 2,

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname* { s u p } - \frac { 1 } { n } \log \nu _ { i , n } \leq \operatorname* { i n f } _ { x \in V } d _ { i } ( \theta _ { i } ^ { 0 } , x ) \leq \varepsilon .
$$

Letting $\varepsilon \downarrow 0$ and using that the variance is bounded gives $- n ^ { - 1 }$ log $\nu _ { i , n } \to 0$

It remains to prove (19). The claim is trivial for $m = 0$ . Fix $m \geq 1$ . The law of total variance gives $\bar { \nu } _ { i } ( t , m ) / \nu _ { i } ( t ) \leq 1$ . We will show that lim $\textstyle \operatorname* { i n f } _ { t \to \infty } \bar { \nu } _ { i } ( t , m ) / \nu _ { i } ( t ) \geq 1$

Let $Y _ { 1 } , \dots , Y _ { m }$ denote the new observations of item �. The expectation $\mathbb { E } _ { \theta _ { i } ^ { 0 } }$ is over these additional observations only. By Bayes’ formula and minimization of expected squared error,

$$
\begin{array} { r l } & { \overline { { \mathbb { P } } } _ { \ell } ( \ell , m ) = \beth _ { \mathbb { C } } [ \frac { \operatorname* { m i n } \int _ { \mathcal { I } } ( X - a ) ^ { 2 } \prod _ { j } ^ { m } p _ { i } ( Y _ { \ell } ) | X \prod _ { i , \ell } ( d x ) } { \int _ { \mathcal { I } } \prod _ { r = 1 } ^ { m } \int _ { 0 } ^ { \infty } \prod _ { i } ( Y _ { \ell } ) \prod _ { i , \ell } ( d z ) } ] } \\ & { \qquad = \mathbb { E } _ { \ell } [  \frac { \prod _ { j } ^ { m } \prod _ { j } ^ { m } ( Y _ { \ell } \mid Z ) \prod _ { i , \ell } ( d z ) } { \underset { \ell = 1 } { \prod } }   } \\ & { \qquad  = \mathbb { E } _ { \ell } (  \frac { \prod _ { j } ^ { m } ( Y _ { \ell } \mid Z ) \prod _ { i , \ell } ( X _ { \ell } ) \prod _ { i , \ell } ( d z ) } { \underset { \ell = 1 } { \prod } }    } \\ & { \qquad   \prod _ { j } ^ { m } | P _ { \ell } ( X _ { \ell } ) \prod _ { i } ^ { ( j ) } )  \times \frac { \operatorname* { m i n } \int _ { \mathcal { I } } \int _ { \mathcal { I } } ^ { m } \prod _ { i } ^ { m } ( Y _ { \ell } \mid Z ) \prod _ { i , \ell } ( d z ) } { \int _ { \mathcal { I } } \prod _ { r = 1 } ^ { m } \int _ { 0 } ^ { \infty } \prod _ { i } ^ { m } ( Y _ { \ell } \mid Z ) \prod _ { i , \ell } ( d z ) } ] } \\ &  \qquad = \mathbb { E } _ { \ell } [  \operatorname* { m i n } \int _ { \mathcal { I } } ( X - a ) ^ { 2 } \prod _ { j = 1 } ^ { m } \frac { p _ { i } ( Y _ { \ell } \mid X ) }  p _ { i } ( Y _ { \ell } ) \prod _ { i } ( Y  \end{array}
$$

We first bound the logarithm of the likelihood product term. We have

$$
\begin{array} { r l r } {  { \log \prod _ { r = 1 } ^ { m } \frac { p _ { i } ( Y _ { r } \mid x ) } { p _ { i } ( Y _ { r } \mid \theta _ { i } ^ { 0 } ) } = \{ \eta _ { i } ( x ) - \eta _ { i } ( \theta _ { i } ^ { 0 } ) \} \sum _ { r = 1 } ^ { m } T _ { i } ( Y _ { r } ) - m \{ A _ { i } ( \eta _ { i } ( x ) ) - A _ { i } ( \eta _ { i } ( \theta _ { i } ^ { 0 } ) ) \} } } \\ & { } & { \geq - | \eta _ { i } ( x ) - \eta _ { i } ( \theta _ { i } ^ { 0 } ) | | \sum _ { r = 1 } ^ { m } T _ { i } ( Y _ { r } ) | - m | A _ { i } ( \eta _ { i } ( x ) ) - A _ { i } ( \eta _ { i } ( \theta _ { i } ^ { 0 } ) ) | . } \end{array}
$$

To bound $\begin{array} { r } { \left| \sum _ { r = 1 } ^ { m } T _ { i } ( Y _ { r } ) \right| } \end{array}$ , fix $ { \varepsilon } \in ( 0 , 1 )$ and choose � so that

$$
\mathbb { P } _ { \theta _ { i } ^ { 0 } } \Big ( \operatorname* { m a x } _ { 1 \leq r \leq m } | T _ { i } ( Y _ { r } ) | \leq M \Big ) \geq 1 - \varepsilon .
$$

If max $_ { \cdot 1 \leq r \leq m } | T _ { i } ( Y _ { r } ) | \leq M$ , then $\begin{array} { r } { \left| \sum _ { r = 1 } ^ { m } T _ { i } ( Y _ { r } ) \right| \leq m M } \end{array}$ . By continuity of $\eta _ { i }$ and $A _ { i } ,$ , choose $V = [ \theta _ { i } ^ { 0 } - \delta , \theta _ { i } ^ { 0 } + \delta ] \subset$ $\mathcal { T }$ so that for $x \in V$

$$
m M | \eta _ { i } ( x ) - \eta _ { i } ( \theta _ { i } ^ { 0 } ) | + m | A _ { i } ( \eta _ { i } ( x ) ) - A _ { i } ( \eta _ { i } ( \theta _ { i } ^ { 0 } ) ) | \leq \varepsilon .
$$

Thus, whenever $\operatorname* { m a x } _ { 1 \leq r \leq m } | T _ { i } ( Y _ { r } ) | \leq M$ , for $x \in V$ we have

$$
\prod _ { r = 1 } ^ { m } { \frac { p _ { i } ( Y _ { r } \mid x ) } { p _ { i } ( Y _ { r } \mid \theta _ { i } ^ { 0 } ) } } \geq e ^ { - \varepsilon } .
$$

Plugging this into the expression for $\bar { \nu } _ { i } ( t , m )$ gives

$$
\bar { \nu } _ { i } ( t , m ) \geq e ^ { - \varepsilon } ( 1 - \varepsilon ) \operatorname* { m i n } _ { a \in \mathbb { R } } \int _ { V } ( x - a ) ^ { 2 } \Pi _ { i , t } ( d x ) \geq e ^ { - \varepsilon } ( 1 - \varepsilon ) \big \{ \nu _ { i } ( t ) - ( U - L ) ^ { 2 } \Pi _ { i , t } ( V ^ { c } ) \big \} .
$$

For the second inequality, the minimizing � belongs to �. The integral over $V ^ { c }$ is therefore at most $( U - L ) ^ { 2 } \Pi _ { i , t } ( V ^ { c } )$ , while the integral over J is at least $\nu _ { i } ( t )$

By (20) and $- n ^ { - 1 } \log \nu _ { i , n } \longrightarrow 0 , \Pi _ { i , t } ( V ^ { c } ) / \nu _ { i } ( t ) \longrightarrow 0$ when $N _ { i } ( t ) \to \infty$ . Dividing by $\nu _ { i } ( t )$ and taking the liminf gives

$$
\operatorname* { l i m i n f } _ { t \to \infty } { \frac { { \bar { \nu } } _ { i } ( t , m ) } { \nu _ { i } ( t ) } } \geq e ^ { - \varepsilon } ( 1 - \varepsilon ) .
$$

Letting $\varepsilon \downarrow 0$ proves (19). □

Proof of Theorem 1 We first show that every item is sampled infinitely often. Since $q ( t ) \to q _ { 0 } \in ( \theta _ { k + 1 } ^ { 0 } , \theta _ { k } ^ { 0 } )$ 9 eventually $q ( t )$ belongs to a compact interval $K \subset \mathcal { T }$ . The proof of Lemma 7 then applies, using Lemma 9 for items sampled infinitely often and Lemma 6 for items that stop receiving samples. Thus $N _ { i } ( t ) $ ∞ and $\mu _ { i } ( t ) \to \theta _ { i } ^ { 0 }$ for every �.

For part (i), we first establish the rates of $D _ { i , t } ( d )$ . We will show that

$$
\operatorname* { m a x } _ { 0 \leq d < B } \left. - \frac { \log D _ { i , t } ( d ) } { N _ { i } ( t ) } - d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } ) \right. \longrightarrow 0 .\tag{21}
$$

We will then use these rates and optimality of each batch to prove $\widehat { w } _ { i } ( t ) \to w _ { i } ( q _ { 0 } )$ . To prove (21), define

$$
J _ { i , t } ( d ) : = \ell _ { i , t } ( d ) - ( q ( t ) - \mu _ { i } ( t ) ) ^ { + } .
$$

We first establish the bounds

$$
\frac { \bar { \nu } _ { i } ( t , B - 1 ) } { 2 \mathbb { V } _ { t } ( Y ) } J _ { i , t } ( B - 1 ) \leq D _ { i , t } ( d ) \leq J _ { i , t } ( 0 ) , \qquad 0 \leq d < B ,\tag{22}
$$

where � is an additional observation of item �.

By (5), $\widetilde { \theta } _ { i } ( d + 1 )$ is a contraction of $\widetilde { \theta } _ { i } ( d )$ about $\mu _ { i } ( t )$ with factor $c _ { i } ( t , d + 1 ) / c _ { i } ( t , d )$ . The convexity argument in the proof of Lemma 6 gives

$$
0 \leq \left( 1 - \frac { c _ { i } ( t , d + 1 ) } { c _ { i } ( t , d ) } \right) J _ { i , t } ( d ) \leq D _ { i , t } ( d ) = J _ { i , t } ( d ) - J _ { i , t } ( d + 1 ) \leq J _ { i , t } ( d ) \leq J _ { i , t } ( 0 ) .\tag{23}
$$

This proves the upper bound in (22). For the lower bound, we next bound $1 - c _ { i } ( t , d + 1 ) / c _ { i } ( t , d )$ from below.

Let $Z _ { d } = ( Y _ { 1 } , \ldots , Y _ { d } )$ be the next � observations of item �, and let $Y = Y _ { d + 1 }$ . Applying the variance and covariance identities in the proof of Lemma 6 conditional on $( \mathcal { H } _ { t } , Z _ { d } )$ , followed by Cauchy–Schwarz, gives

$$
\bar { \nu } _ { i } ( t , d ) - \bar { \nu } _ { i } ( t , d + 1 ) \geq \mathbb { E } _ { t } \left[ \frac { \mathbb { V } ( \theta _ { i } | \mathcal { H } _ { t } , Z _ { d } ) ^ { 2 } } { \mathbb { V } ( Y | \mathcal { H } _ { t } , Z _ { d } ) } \right] \geq \frac { \bar { \nu } _ { i } ( t , d ) ^ { 2 } } { \mathbb { E } _ { t } \left[ \mathbb { V } ( Y | \mathcal { H } _ { t } , Z _ { d } ) \right] } \geq \frac { \bar { \nu } _ { i } ( t , d ) ^ { 2 } } { \mathbb { V } _ { t } ( Y ) } .
$$

The last inequality follows from the law of total variance. Therefore,

$$
1 - \frac { c _ { i } ( t , d + 1 ) } { c _ { i } ( t , d ) } = \frac { \bar { \nu } _ { i } ( t , d ) - \bar { \nu } _ { i } ( t , d + 1 ) } { \bar { \nu } _ { i } ( t , d ) ( 1 + c _ { i } ( t , d + 1 ) / c _ { i } ( t , d ) ) } \geq \frac { \bar { \nu } _ { i } ( t , d ) } { 2 \mathbb { V } _ { t } ( Y ) } \geq \frac { \bar { \nu } _ { i } ( t , B - 1 ) } { 2 \mathbb { V } _ { t } ( Y ) } .
$$

Also, (23) gives $J _ { i , t } ( d ) \geq J _ { i , t } ( B - 1 )$ for $d < B .$ . Substituting into the lower bound in (23) proves (22).

We now show that both $J _ { i , t } ( 0 )$ and $J _ { i , t } ( B { - } 1 )$ have exponential rate $d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } )$

$$
- { \frac { \log J _ { i , t } ( d ) } { N _ { i } ( t ) } } \longrightarrow d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } ) , \qquad d \in \{ 0 , B - 1 \} .\tag{24}
$$

By Lemma 9,

$$
c _ { i } ( t , d )  1 , \qquad z _ { i } ( t , d ) = \mu _ { i } ( t ) + { \frac { q ( t ) - \mu _ { i } ( t ) } { c _ { i } ( t , d ) } }  q _ { 0 } .
$$

Since $\mu _ { i } ( t ) \to \theta _ { i } ^ { 0 }$ and $q _ { 0 } \in ( \theta _ { k + 1 } ^ { 0 } , \theta _ { k } ^ { 0 } )$ , (7) gives, for all suficiently large �,

$$
J _ { i , t } ( d ) = \left\{ \begin{array} { l l } { c _ { i } ( t , d ) \mathbb { E } _ { t } [ ( z _ { i } ( t , d ) - \theta _ { i } ) ^ { + } ] , } & { i \leq k , } \\ { c _ { i } ( t , d ) \mathbb { E } _ { t } [ ( \theta _ { i } - z _ { i } ( t , d ) ) ^ { + } ] , } & { i > k . } \end{array} \right.
$$

Write $n = N _ { i } ( t )$ . By (18) and monotonicity of $d _ { i } ( \theta _ { i } ^ { 0 } , x )$ ,

$$
J _ { i , t } ( d ) \leq ( U - L ) e ^ { - n ( d _ { i } ( \theta _ { i } ^ { 0 } , z _ { i } ( t , d ) ) - \varepsilon _ { i , n } ) } .
$$

For a lower bound when $i \leq k$ , choose a suficiently small $\delta > 0$ and let $V = [ q _ { 0 } - 2 \delta , q _ { 0 } - \delta ] \subset \mathcal { T }$ . Eventually $| z _ { i } ( t , d ) - q _ { 0 } | < \delta / 2$ , so

$$
J _ { i , t } ( d ) \geq c _ { i } ( t , d ) \frac { \delta } { 2 } \Pi _ { i , t } ( V ) \geq c _ { i } ( t , d ) \frac { \delta } { 2 } \Pi _ { i , 0 } ( V ) e ^ { - n ( d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } - 2 \delta ) + \varepsilon _ { i , n } ) } .
$$

For $i > k$ , take $V = \left[ q _ { 0 } + \delta , q _ { 0 } + 2 \delta \right]$ and obtain the same bound with $q _ { 0 } + 2 \delta$ in place of $q _ { 0 } - 2 \delta$ . Since $c _ { i } ( t , d ) \to 1$ and $\Pi _ { i , 0 } ( V ) > 0$ , taking logarithms, letting $t \to \infty$ , and then letting $\delta \downarrow 0$ proves (24).

To conclude (21) from (22), it remains to show that

$$
- \frac { 1 } { n } \log \frac { \bar { \nu } _ { i } ( t , B - 1 ) } { 2 \mathbb { V } _ { t } ( Y ) } \longrightarrow 0 .
$$

The numerator is covered by Lemma 9. For the denominator, let $\begin{array} { r } { M _ { i } : = \int _ { \mathcal { T } } \mathbb { E } _ { x } [ Y ^ { 2 } ] \Pi _ { i , 0 } ( d x ) < \infty } \end{array}$ , by Assumption 1. Equation (18) gives

$$
\mathbb { V } _ { t } ( Y ) \le \mathbb { E } _ { t } [ Y ^ { 2 } ] \le M _ { i } e ^ { n \varepsilon _ { i , n } } .
$$

Since $\mathbb { V } _ { t } ( Y ) \ge \nu _ { i } ( t ) \ge \bar { \nu } _ { i } ( t , B - 1 )$

$$
0 \leq - \frac { 1 } { n } \log \frac { \bar { \nu } _ { i } ( t , B - 1 ) } { 2 \mathbb { V } _ { t } ( Y ) } \leq - \frac { 1 } { n } \log \bar { \nu } _ { i } ( t , B - 1 ) + \varepsilon _ { i , n } + \frac { \log ( 2 M _ { i } ) } { n } \longrightarrow 0 .
$$

Combining this with (24) in (22) proves (21).

We now use these rates and batch optimality to show that

$$
\frac { d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } ) N _ { i } ( t ) - \operatorname* { m i n } _ { j } d _ { j } ( \theta _ { j } ^ { 0 } , q _ { 0 } ) N _ { j } ( t ) } { \sum _ { j } N _ { j } ( t ) } \longrightarrow 0 \qquad \mathrm { f o r ~ e v e r y ~ } i .
$$

We first compare the sample counts for items sampled in the current round. Write

$$
\varepsilon _ { t } : = \operatorname* { m a x } _ { i , 0 \leq d < B } \frac { | - \log D _ { i , t } ( d ) - d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } ) N _ { i } ( t ) | } { \sum _ { j } N _ { j } ( t ) } .
$$

By (21), $\varepsilon _ { t } \to 0$ . If $\Delta _ { i } ^ { t } > 0$ and $j \neq i ,$ , transferring one sample from � to � is feasible. Optimality of $\Delta ^ { t }$ gives

$$
\begin{array} { r } { 0 \le \ell _ { i , t } ( \Delta _ { i } ^ { t } - 1 ) + \ell _ { j , t } ( \Delta _ { j } ^ { t } + 1 ) - \ell _ { i , t } ( \Delta _ { i } ^ { t } ) - \ell _ { j , t } ( \Delta _ { j } ^ { t } ) = D _ { i , t } ( \Delta _ { i } ^ { t } - 1 ) - D _ { j , t } ( \Delta _ { j } ^ { t } ) . } \end{array}
$$

Since $0 \leq \Delta _ { i } ^ { t } - 1 \leq B - 1$ and $0 \leq \Delta _ { j } ^ { t } \leq B - 1$

$$
\begin{array} { r l r } {  { d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } ) N _ { i } ( t ) \le - \log D _ { i , t } ( \Delta _ { i } ^ { t } - 1 ) + \varepsilon _ { t } \sum _ { \ell } N _ { \ell } ( t ) } } \\ & { } & \\ & { } & { \qquad \le - \log D _ { j , t } ( \Delta _ { j } ^ { t } ) + \varepsilon _ { t } \sum _ { \ell } N _ { \ell } ( t ) } \\ & { } & \\ & { } & { \qquad \le d _ { j } ( \theta _ { j } ^ { 0 } , q _ { 0 } ) N _ { j } ( t ) + 2 \varepsilon _ { t } \sum _ { \ell } N _ { \ell } ( t ) . } \end{array}
$$

Taking the minimum over � gives

$$
\Delta _ { i } ^ { t } > 0 \quad \Longrightarrow \quad d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } ) N _ { i } ( t ) \leq \operatorname* { m i n } _ { j } d _ { j } ( \theta _ { j } ^ { 0 } , q _ { 0 } ) N _ { j } ( t ) + 2 \varepsilon _ { t } \sum _ { j } N _ { j } ( t ) .\tag{25}
$$

To extend this analysis to every item, we apply it at the last round in which that item was sampled. Let $\tau _ { i } ( t ) < t$ be that round. Since $N _ { i } ( t ) \to \infty$ , we have $\tau _ { i } ( t ) \to \infty$ , and boundedness of the batches gives

$$
N _ { i } ( t ) = N _ { i } ( \tau _ { i } ( t ) ) + \Delta _ { i } ^ { \tau _ { i } ( t ) } \leq N _ { i } ( \tau _ { i } ( t ) ) + B .
$$

Also, min<sub>�</sub> $d _ { j } ( \theta _ { j } ^ { 0 } , q _ { 0 } ) N _ { j } ( t )$ is nondecreasing. Applying (25) at $\tau _ { i } ( t )$ therefore gives

$$
\begin{array} { l } { \displaystyle 0 \le \frac { d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } ) N _ { i } ( t ) - \operatorname* { m i n } _ { j } d _ { j } ( \theta _ { j } ^ { 0 } , q _ { 0 } ) N _ { j } ( t ) } { \sum _ { j } N _ { j } ( t ) } } \\ { \displaystyle \le \frac { B d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } ) } { \sum _ { j } N _ { j } ( t ) } + 2 \varepsilon _ { \tau _ { i } ( t ) } \frac { \sum _ { j } N _ { j } ( \tau _ { i } ( t ) ) } { \sum _ { j } N _ { j } ( t ) } } \end{array}
$$

$$
\leq \frac { B d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } ) } { \sum _ { j } N _ { j } ( t ) } + 2 \varepsilon _ { \tau _ { i } ( t ) } \longrightarrow 0 .
$$

Dividing by $d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } )$ , summing over �, and using $\begin{array} { r } { \sum _ { i } \widehat { w } _ { i } ( t ) = 1 } \end{array}$ gives

$$
\frac { \operatorname* { m i n } _ { j } d _ { j } ( \theta _ { j } ^ { 0 } , q _ { 0 } ) N _ { j } ( t ) } { \sum _ { j } N _ { j } ( t ) } { \longrightarrow } \left( \sum _ { i } \frac { 1 } { d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } ) } \right) ^ { - 1 } .
$$

Hence

$$
\widehat { w } _ { i } ( t ) \longrightarrow \frac { 1 / d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } ) } { \sum _ { j } { 1 / d _ { j } ( \theta _ { j } ^ { 0 } , q _ { 0 } ) } } = w _ { i } ( q _ { 0 } ) ,
$$

proving part (i). Part (ii) follows from Lemma 8 and continuity of Γ.

For part (iii), $\begin{array} { r } { w _ { i } ( q _ { 0 } ) d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } ) = \left( \sum _ { \ell = 1 } ^ { I } 1 / d _ { \ell } ( \theta _ { \ell } ^ { 0 } , q _ { 0 } ) \right) ^ { - 1 } } \end{array}$ for every �. For any $i \leq k < j$ and $x \in [ \theta _ { j } ^ { 0 } , \theta _ { i } ^ { 0 } ]$ monotonicity gives

$$
\begin{array} { r } { w _ { i } ( q _ { 0 } ) d _ { i } ( \theta _ { i } ^ { 0 } , x ) + w _ { j } ( q _ { 0 } ) d _ { j } ( \theta _ { j } ^ { 0 } , x ) \geq \left\{ \begin{array} { l l } { w _ { i } ( q _ { 0 } ) d _ { i } ( \theta _ { i } ^ { 0 } , q _ { 0 } ) = \left( \sum _ { \ell = 1 } ^ { I } 1 / d _ { \ell } ( \theta _ { \ell } ^ { 0 } , q _ { 0 } ) \right) ^ { - 1 } , } & { x \leq q _ { 0 } , } \\ { w _ { j } ( q _ { 0 } ) d _ { j } ( \theta _ { j } ^ { 0 } , q _ { 0 } ) = \left( \sum _ { \ell = 1 } ^ { I } 1 / d _ { \ell } ( \theta _ { \ell } ^ { 0 } , q _ { 0 } ) \right) ^ { - 1 } , } & { x \geq q _ { 0 } . } \end{array} \right. } \end{array}
$$

Taking the minimum over � and then over $i \leq k < j$ in (12) gives $\begin{array} { r } { \Gamma ( w ( q _ { 0 } ) ; \theta ^ { 0 } ) \geq \left( \sum _ { \ell = 1 } ^ { I } 1 / d _ { \ell } ( \theta _ { \ell } ^ { 0 } , q _ { 0 } ) \right) ^ { - 1 } } \end{array}$ □

ProofofRemark 3 Fix � with $x _ { [ k ] } > x _ { [ k + 1 ] }$ . The objective $\textstyle \sum _ { i } 1 / d _ { i } ( x _ { i } , q )$ is continuous on $( x _ { [ k + 1 ] } , x _ { [ k ] } )$ and tends to infinity at its endpoints, so it attains a minimum. We prove uniqueness for each family by strict convexity in � or in a strictly increasing function of $q .$

For Gaussian observations, each term is strictly convex in �, since

$$
\frac { 1 } { d _ { i } ( x _ { i } , q ) } = \frac { 2 \sigma _ { i } ^ { 2 } } { ( x _ { i } - q ) ^ { 2 } } , \qquad \frac { d ^ { 2 } } { d q ^ { 2 } } \frac { 1 } { d _ { i } ( x _ { i } , q ) } = \frac { 1 2 \sigma _ { i } ^ { 2 } } { ( x _ { i } - q ) ^ { 4 } } > 0 .
$$

For lognormal observations, set $z = \log { q }$ . Then

$$
\frac { 1 } { d _ { i } ( x _ { i } , e ^ { z } ) } = \frac { 2 \sigma _ { i } ^ { 2 } } { ( \log { x _ { i } } - z ) ^ { 2 } } , \qquad \frac { d ^ { 2 } } { d z ^ { 2 } } \frac { 1 } { d _ { i } ( x _ { i } , e ^ { z } ) } = \frac { 1 2 \sigma _ { i } ^ { 2 } } { ( \log { x _ { i } } - z ) ^ { 4 } } > 0 .
$$

For Bernoulli observations, write $q = e ^ { z } / ( 1 + e ^ { z } )$ . We show that $1 / d _ { i } ( x _ { i } , q )$ is strictly convex in $z .$ Diferentiating gives

$$
\frac { d ^ { 2 } } { d z ^ { 2 } } \frac { 1 } { d _ { i } ( x _ { i } , q ) } = \frac { 2 ( q - x _ { i } ) ^ { 2 } - d _ { i } ( x _ { i } , q ) q ( 1 - q ) } { d _ { i } ( x _ { i } , q ) ^ { 3 } } .
$$

To show that the numerator is positive, we bound $d _ { i } ( x _ { i } , q )$ from above. Using log $u \leq u - 1$

$$
d _ { i } ( x _ { i } , q ) = x _ { i } \log { \frac { x _ { i } } { q } } + ( 1 - x _ { i } ) \log { \frac { 1 - x _ { i } } { 1 - q } } \leq x _ { i } \left( { \frac { x _ { i } } { q } } - 1 \right) + ( 1 - x _ { i } ) \left( { \frac { 1 - x _ { i } } { 1 - q } } - 1 \right) = { \frac { ( x _ { i } - q ) ^ { 2 } } { q ( 1 - q ) } } .
$$

Substituting gives

$$
\frac { d ^ { 2 } } { d z ^ { 2 } } \frac { 1 } { d _ { i } ( x _ { i } , q ) } \geq \frac { ( q - x _ { i } ) ^ { 2 } } { d _ { i } ( x _ { i } , q ) ^ { 3 } } > 0 ,
$$

since $q \in ( x _ { [ k + 1 ] } , x _ { [ k ] } )$ implies $q \neq x _ { i }$

For Poisson and Gamma observations, write $q = e ^ { z }$ . Then

$$
d _ { i } ( x _ { i } , q ) = \left\{ \begin{array} { l l } { x _ { i } ( e ^ { \nu } - 1 - \nu ) , } & { \nu = z - \log x _ { i } , \quad \mathrm { P o i s s o n } , } \\ { \kappa _ { i } ( e ^ { \nu } - 1 - \nu ) , } & { \nu = \log x _ { i } - z , \quad \mathrm { G a m m a } . } \end{array} \right.
$$

Since $x _ { i } , \kappa _ { i } > 0$ and $d \nu / d z = \pm 1$ , it sufices to show that $1 / ( e ^ { \nu } { - } 1 { - } \nu )$ is strictly convex for $\nu \neq 0$ . Diferentiating gives

$$
\frac { d ^ { 2 } } { d \nu ^ { 2 } } \frac { 1 } { e ^ { \nu } - 1 - \nu } = \frac { e ^ { \nu } ( e ^ { \nu } + 2 e ^ { - \nu } + \nu - 3 ) } { ( e ^ { \nu } - 1 - \nu ) ^ { 3 } } .
$$

The denominator is positive for $\nu \neq 0$ . To show that the numerator is positive, use $e ^ { \nu } > 1 { + } \nu$ and $e ^ { - \nu } > 1 - \nu$ to get

$$
e ^ { \nu } + 2 e ^ { - \nu } + \nu - 3 > ( 1 + \nu ) + 2 ( 1 - \nu ) + \nu - 3 = 0 .
$$

Thus the second derivative is positive, proving strict convexity in � for both families. □

ProofofLemma 4 By Lemma 9 and the proof of Lemma $7 , \mu ( t ) \to \pmb \theta ^ { 0 }$ . Define

$$
H _ { t } ( q ) : = \sum _ { i } \frac { 1 } { d _ { i } ( \mu _ { i } ( t ) , q ) } , \qquad H ( q ) : = \sum _ { i } \frac { 1 } { d _ { i } ( \theta _ { i } ^ { 0 } , q ) } .
$$

We will show that every limit of $q ^ { \dagger } ( t )$ belongs to $( \theta _ { k + 1 } ^ { 0 } , \theta _ { k } ^ { 0 } )$ and minimizes �. Uniqueness then gives convergence to $q ^ { \dagger }$

Since $\mu _ { [ k ] } ( t ) \to \theta _ { k } ^ { 0 }$ and $\mu _ { \left[ k + 1 \right] } ( t ) \to \theta _ { k + 1 } ^ { 0 }$ , eventually $\mu _ { [ k + 1 ] } ( t ) < q ^ { \dagger } < \mu _ { [ k ] } ( t )$ . Since $q ^ { \dagger } ( t )$ minimizes $H _ { t }$ on this interval,

$$
H _ { t } ( q ^ { \dagger } ( t ) ) \leq H _ { t } ( q ^ { \dagger } ) \longrightarrow H ( q ^ { \dagger } ) < \infty .\tag{26}
$$

Let $t _ { n } \to \infty$ satisfy $q ^ { \dagger } ( t _ { n } ) \to \bar { q }$ . Since $\mu _ { [ k + 1 ] } ( t ) \leq q ^ { \dagger } ( t ) \leq \mu _ { [ k ] } ( t )$ , we have $\bar { q } \in [ \theta _ { k + 1 } ^ { 0 } , \theta _ { k } ^ { 0 } ]$ . If $\bar { q } = \theta _ { k } ^ { 0 }$ (and likewise $\bar { q } = \theta _ { k + 1 } ^ { 0 } )$ , then

$$
H _ { t _ { n } } ( q ^ { \dagger } ( t _ { n } ) ) \geq { \frac { 1 } { d _ { k } ( \mu _ { k } ( t _ { n } ) , q ^ { \dagger } ( t _ { n } ) ) } } { \longrightarrow } \infty ,
$$

contradicting (26). Therefore $\bar { q } \in ( \theta _ { k + 1 } ^ { 0 } , \theta _ { k } ^ { 0 } )$ , and continuity gives

$$
H ( { \bar { q } } ) = \operatorname* { l i m } _ { n \to \infty } H _ { t _ { n } } ( q ^ { \dagger } ( t _ { n } ) ) \leq H ( q ^ { \dagger } ) .
$$

Thus $\bar { q }$ minimizes $H ,$ and Assumption 3 gives $\bar { q } = q ^ { \dagger }$ . Since $q ^ { \dagger } ( t )$ is eventually contained in a compact interval and every such limit equals $q ^ { \dagger }$ , we conclude that $q ^ { \dagger } ( t )  q ^ { \dagger }$ □

Proof of Theorem 2 By Lemma 4 and Theorem 1(ii),

$$
- \frac { \log \mathrm { E O C } ( \mathcal { H } _ { t } ) } { \sum _ { i } N _ { i } ( t ) } \longrightarrow \Gamma ( \pmb { w } ( \boldsymbol { q } ^ { \dag } ) ; \pmb { \theta } ^ { 0 } ) \le \Gamma ^ { * } ( \pmb { \theta } ^ { 0 } ) .
$$

For the lower bound, write $\begin{array} { r } { H ( q ) : = \sum _ { i } 1 / d _ { i } ( \theta _ { i } ^ { 0 } , q ) } \end{array}$ . Part (iii) of Theorem 1 gives

$$
\Gamma ( w ( q ^ { \dagger } ) ; { \pmb \theta } ^ { 0 } ) \geq \frac { 1 } { H ( q ^ { \dagger } ) } .
$$

It therefore remains to show $\Gamma ^ { * } ( { \pmb \theta } ^ { 0 } ) / 2 \leq 1 / H ( q ^ { \dagger } )$ .

Let $\pmb { w } ^ { * }$ maximize Γ. By the proof of Proposition 5, $w _ { i } ^ { * } > 0$ for every �. The functions (of �)

$$
\operatorname* { m i n } _ { i \leq k } w _ { i } ^ { * } d _ { i } ( \theta _ { i } ^ { 0 } , q ) , \qquad \operatorname* { m i n } _ { j > k } w _ { j } ^ { * } d _ { j } ( \theta _ { j } ^ { 0 } , q )
$$

are continuous on $[ \theta _ { k + 1 } ^ { 0 } , \theta _ { k } ^ { 0 } ]$ . The first is positive at $\theta _ { k + 1 } ^ { 0 }$ and zero at $\theta _ { k } ^ { 0 }$ , while the second is zero at $\theta _ { k + 1 } ^ { 0 }$ and positive at $\theta _ { k } ^ { 0 }$ . Thus they are equal at some $\bar { q } \in ( \theta _ { k + 1 } ^ { 0 } , \theta _ { k } ^ { 0 } )$ . Write

$$
r : = \operatorname* { m i n } _ { i \leq k } w _ { i } ^ { * } d _ { i } ( \theta _ { i } ^ { 0 } , \bar { q } ) = \operatorname* { m i n } _ { j > k } w _ { j } ^ { * } d _ { j } ( \theta _ { j } ^ { 0 } , \bar { q } ) > 0 .
$$

Choose $i \leq k < j$ attaining the two minima. Evaluating (12) at this pair and $x = \bar { q }$ gives

$$
\Gamma ^ { * } ( \pmb { \theta } ^ { 0 } ) \leq w _ { i } ^ { * } d _ { i } ( \theta _ { i } ^ { 0 } , \bar { q } ) + w _ { j } ^ { * } d _ { j } ( \theta _ { j } ^ { 0 } , \bar { q } ) = 2 r .
$$

Since $w _ { i } ^ { * } d _ { i } ( \theta _ { i } ^ { 0 } , \bar { q } ) \geq r$ for every �,

$$
r H ( \bar { q } ) = \sum _ { i } \frac { r } { d _ { i } ( \theta _ { i } ^ { 0 } , \bar { q } ) } \leq \sum _ { i } w _ { i } ^ { * } = 1 .
$$

Since $q ^ { \dagger }$ minimizes �,

$$
\frac { 1 } { 2 } \Gamma ^ { * } ( \pmb { \theta } ^ { 0 } ) \leq r \leq \frac { 1 } { H ( \bar { q } ) } \leq \frac { 1 } { H ( q ^ { \dag } ) } \leq \Gamma ( \pmb { w } ( q ^ { \dag } ) ; \pmb { \theta } ^ { 0 } ) .
$$

The corresponding result for the posterior error follows from Lemma 8. □

## Appendix D: Additional simulation results

This section presents the remaining plots of the experiments in §5. §D.1 gives the EOC plots for the Bernoulli, exponential, and lognormal families and the PCS counterparts of all the EOC plots, which use the middle column in Table 1 for distributions of the true means. §D.2 and §D.3 give the EOC and PCS plots for the right and left columns, respectively. Table 1 is repeated below for reference.

<table><tr><td>Likelihood  $Y _ { i , n } \mid \boldsymbol { \theta } _ { i }$ </td><td colspan="3">Distribution of true means  $\theta _ { i }$ </td></tr><tr><td> $N ( \theta _ { i } , 1 )$ </td><td>Unif(−2,2)</td><td>Unif(−1,1)</td><td>Unif(−0.5,0.5)</td></tr><tr><td> $\textstyle \operatorname { E x p } ( 1 / \theta _ { i } )$ </td><td>Unif(1,9)</td><td>Unif(1,5)</td><td>Unif(1,3)</td></tr><tr><td>Bern(θi)</td><td>Unif(0.05,0.95)</td><td>Unif(0.2,0.8)</td><td>Unif(0.35,0.65)</td></tr><tr><td>Lognormal(log  $\theta _ { i } { - } 1 / 2 , 1 )$ </td><td>Unif(1,9)</td><td>Unif(1,5)</td><td>Unif(1,3)</td></tr></table>

## D.1. Middle column

![](images/636ef9e4095387280a634ed710d231396772ec67929ec863ba7b0db5c52fd0b9.jpg)  
Figure 8 Bernoulli: same format as Figure 6.

![](images/928bcab3d24ff44596028d5df3e6661fdc2a54971168417840a555693ad1daf2.jpg)  
Figure 9 Exponential: same format as Figure 6.

![](images/454d54f5820c651a14aa79f30c0f73f80b4ab7c690fcbaac09eb7f486b64704c.jpg)  
Figure 10 Lognormal: same format as Figure 6.

![](images/fac732151744d1d802e07d1d5e900e98b99fc22e56fbed693a99b107c9b3f179.jpg)  
Figure 11 Gaussian: PCS.

![](images/8ea5483b63ac8d594e9a26441e1a5158a436e5876f61811c7ef0683c98b7d691.jpg)  
Figure 12 Bernoulli: PCS.

![](images/4963b16cbdb7c0539cc07648d8082048351184e8af3b3b41e6beedc0c5995f5e.jpg)  
Figure 13 Exponential: PCS.

![](images/eaccd1971694a158a9f013804f56a2df0ede718b38323f34f69039abd33707a3.jpg)  
Figure 14 Lognormal: PCS.

## D.2. Right column

![](images/a995880e1d42fd8c6b37507635e7b45f26c60d12d8e109332f6a2ea2f696874e.jpg)  
Figure 15 Gaussian: EOC.

![](images/c0338d6399ad174313571093ee93760a6ea6ba074d4cd52342ebe40f9c4b6884.jpg)  
Figure 16 Bernoulli: EOC.

![](images/1d60097146569f60b0eceb3a7b1a938f579499a496dd0f0ac70eeb33cbcd2acc.jpg)  
Figure 17 Exponential: EOC.

![](images/67c4c803878d76ee9c759200cb96d429b3fed23f021672dc757de29420baab76.jpg)  
Figure 18 Lognormal: EOC.

![](images/42baf7aa82007df32437a95cc29110f857d2a9042d1e24b8a61f98796d777917.jpg)  
Figure 19 Gaussian: PCS.

![](images/fc8818e0f6ab754a6876dea21b8f269781b720359655c6d2bfab0b75fba4e62b.jpg)  
Figure 20 Bernoulli: PCS.

![](images/559573e494e3897a29bd5199d7e31a846efa4b97ccddb69182460eebc6ba35ca.jpg)  
Figure 21 Exponential: PCS.

![](images/63fae729f44b7ac188d452538c683666c866d35822eabe26b7a3215b3f073d60.jpg)  
Figure 22 Lognormal: PCS.

## D.3. Left column

![](images/4af99b95d906f6bab1b34ae4aa0a46f294369254e8f979ce7f048d78b317c555.jpg)  
Figure 23 Gaussian: EOC.

![](images/f4e7bf6c2425632e8aa52fb15c796f5cd09f391e2d5d34529f88cd4c566d5f7c.jpg)  
Figure 24 Bernoulli: EOC.

![](images/866d0843c280422aaf5028ebd8f5971f6dfb0a8a692c82eab3d92c3a348be7d6.jpg)  
Figure 25 Exponential: EOC.

![](images/5dfce29ce3bf1f4ac549a964277035ed58f223a15dc6fe7b367d6ba7e96301d5.jpg)  
Figure 26 Lognormal: EOC.

![](images/538ddb999ceb9cb4aa50b921ff9804c5fcc33284ece5d341973da2a7c5f7943e.jpg)  
Figure 27 Gaussian: PCS.

![](images/c77e2152bf481ad48d488639dcc7c56abf04113965ed1e7bce5ba96c2c65822b.jpg)  
Figure 28 Bernoulli: PCS.

![](images/30b5bc94698f09059b51eb32cb5370038240f802f080c3624986b2e7d8bb1934.jpg)  
Figure 29 Exponential: PCS.

![](images/50cd3c80c05f4da834210b3d9eb9620a93da88acc2a6104b26e1493623bb3b29.jpg)  
Figure 30 Lognormal: PCS.